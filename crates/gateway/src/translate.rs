//! Chat Completions ↔ Messages ↔ Responses translation for a managed catalog walk.
//!
//! Triggered when the inbound path names Chat Completions, Messages, or Responses (including the
//! `/auto` suffix) and the catalog row speaks a different one of those three. Same-wire walks stay
//! a byte relay — including `/{provider}/v1/responses`. `/{provider}/…` never translates.
//!
//! Responses ↔ Messages is composed through Chat Completions so thinking / `cache_control` /
//! `reasoning_effort` keep the slice-1 mappings.
//!
//! What crosses, and how:
//! - **Dropped:** Responses-only session fields (`store`, `previous_response_id`, `include`,
//!   `truncation`, …) when leaving Responses; `stream_options` on a Responses or Anthropic body.
//!   Same-endpoint Responses is a byte relay: those session fields pass through. Hints with no
//!   equivalent and no effect on the response's shape (`seed`, penalties, `logit_bias`, `top_k`,
//!   a message's `name`, Responses `reasoning` items, and off OpenAI's own API `prompt_cache_key`,
//!   `service_tier` and `verbosity`) are dropped, as is OpenAI JSON mode (`json_object`):
//!   Anthropic's format needs a schema, and OpenAI already requires a JSON-mode prompt to ask for
//!   JSON. Another vendor's `thinking` blocks and Anthropic `cache_control` are dropped onto
//!   Responses, which reads neither.
//! - **Forwarded for the provider to reject:** input and options that change what the client gets
//!   back and have no equivalent on the target (`input_audio`, a `file_id` or URL document or
//!   image, a non-base64 data URI, `n` > 1, `logprobs`, audio output, `stop` onto Responses or a
//!   model that rejects it, hosted / server / `custom` tools, `mcp_servers`, a Responses `prompt`
//!   template). Translation runs after the request headers went upstream, so the gateway cannot 400
//!   here; passing the field through gets the provider's 400 naming it, instead of an answer about
//!   input the model never saw or without a tool the client offered.
//! - **Images, both ways:** base64 data-URI images convert to Anthropic `base64` sources and back;
//!   `http(s)` image URLs become Anthropic `url` sources and back, and pass through between Chat
//!   Completions and Responses. Nothing is fetched: every one of those upstreams downloads the URL
//!   itself. Amazon Bedrock is the exception — it rejects `url` sources — so a URL image that fails
//!   over onto a Bedrock candidate gets Bedrock's 400 rather than a silently image-less answer.
//! - **PDFs, both ways:** a Chat Completions `file` part with inline `file_data` ↔ an Anthropic
//!   base64 `document` ↔ a Responses `input_file`.
//! - **Structured output, both ways:** `response_format` `json_schema` ↔ Anthropic
//!   `output_config.format` ↔ Responses `text.format`.
//! - **Passed both ways:** `thinking` / `redacted_thinking` blocks, `cache_control` on tools and
//!   content, `parallel_tool_calls: false` ↔ `tool_choice.disable_parallel_tool_use`, `user` ↔
//!   `metadata.user_id` (an email address, which Anthropic rejects, becomes a stable hash), tool
//!   `strict`, mid-conversation `system` / `developer` messages in place (see `place_system`), and
//!   images or files a tool returned (a Chat Completions `tool` message holds text, so they follow
//!   it as a user message). Legacy `functions` / `function_call` become the `tools` loop.
//! - **Model-aware onto Messages** (see `ClaudeModel`): `reasoning_effort`, or a client's own
//!   `thinking` object, becomes adaptive thinking plus `output_config.effort` on Claude 4.6 and
//!   later, and a `budget_tokens` below `max_tokens` before that. `temperature` / `top_p` are
//!   dropped where the model rejects them (4.7 and later, and anything with thinking on), clamped
//!   to 0–1, and `top_p` yields to `temperature`. Budget thinking yields to forced tool use.
//!   Conversation-binding models get `thinking.block_binding` (see
//!   `bind_thinking_with_drop_block`). Tool ids are rewritten to Anthropic's pattern.
//! - **Model-aware onto Chat Completions and Responses** (see `OpenAiModel`): OpenAI's own ids get
//!   `max_completion_tokens`, sampling only where the model takes it, and an effort from the set
//!   the family accepts (none on a model that does not reason).
//! - **Added onto Messages:** default `cache_control` breakpoints when the client set none (see
//!   `auto_cache_breakpoints`). An OpenAI SDK never marks anything, and without a marker Anthropic
//!   caches nothing.
//! - **Added onto Responses:** `store: false` unless the client asked to store; Chat Completions
//!   stores nothing by default, Responses stores everything.
//! - **Required mapping:** system/messages/`input`, `max_tokens`/`max_output_tokens`, stop,
//!   stream, tools, `tool_choice`, text + tool_use/tool_result, usage. Anthropic requires
//!   `max_tokens`; a missing OpenAI value becomes 4096.
//!
//! Usage/billing parse the **upstream** body. This module only reshapes bytes the client sees.

use crate::route::Endpoint;
use serde_json::{Map, Value, json};

/// Anthropic requires this; OpenAI does not. Used when the Chat Completions body omitted it.
const DEFAULT_MAX_TOKENS: u64 = 4096;

/// Most bytes translation holds for one response: a whole non-streaming body, or one SSE event not
/// yet terminated. Sized for the largest legitimate case — a Responses `response.completed` event
/// repeats the entire output — with room to spare; past it `proxy` aborts the response rather than
/// let one upstream grow the gateway's memory without bound.
pub const MAX_TRANSLATE_BUFFER: usize = 32 * 1024 * 1024;

/// Per-request translate state, boxed on [`crate::proxy`]'s model-routed path only.
pub struct TranslateState {
    /// Inbound endpoint — what the client sent and what it must receive.
    pub client: Endpoint,
    /// SSE translator, created in `response_filter` once the upstream is known to stream.
    pub sse: Option<SseBridge>,
    /// Non-stream JSON, withheld until end-of-stream so we can map the object.
    pub json_buf: Vec<u8>,
}

impl TranslateState {
    pub fn new(client: Endpoint) -> Self {
        Self {
            client,
            sse: None,
            json_buf: Vec::new(),
        }
    }

    /// Append a non-streaming response chunk. `false`, with nothing appended, once the body would
    /// pass [`MAX_TRANSLATE_BUFFER`].
    #[must_use]
    pub fn push_json(&mut self, chunk: &[u8]) -> bool {
        if self.json_buf.len().saturating_add(chunk.len()) > MAX_TRANSLATE_BUFFER {
            return false;
        }
        self.json_buf.extend_from_slice(chunk);
        true
    }
}

/// Map a buffered request body from `from` (inbound) to `to` (upstream endpoint).
///
/// `upstream_model` is the id this attempt's candidate will receive. Onto Messages it decides
/// which reasoning, sampling and tool-choice controls the Claude model accepts (see `ClaudeModel`).
///
/// Unparseable JSON is returned unchanged so the provider 400s rather than us 502ing after
/// headers have already gone upstream. The candidate `model` id is spliced by the caller
/// **after** this returns.
pub fn request(from: Endpoint, to: Endpoint, body: &[u8], upstream_model: &str) -> Vec<u8> {
    if from == to {
        return body.to_vec();
    }
    let Ok(v) = serde_json::from_slice::<Value>(body) else {
        return body.to_vec();
    };
    if !v.is_object() {
        return body.to_vec();
    }
    encode(&map_request(from, to, &v, Upstream::of(upstream_model)))
}

/// What this attempt's upstream model accepts, parsed once from the id it will receive.
#[derive(Debug, Clone, Copy)]
struct Upstream {
    claude: ClaudeModel,
    openai: OpenAiModel,
}

impl Upstream {
    fn of(model: &str) -> Self {
        Self {
            claude: ClaudeModel::of(model),
            openai: OpenAiModel::of(model),
        }
    }
}

fn map_request(from: Endpoint, to: Endpoint, v: &Value, up: Upstream) -> Value {
    // Legacy `functions` / `function_call` are the same tool loop in an older shape.
    let chat = (from == Endpoint::ChatCompletions).then(|| legacy_functions_to_tools(v));
    let chat = chat.as_ref().and_then(Option::as_ref).unwrap_or(v);
    match (from, to) {
        (Endpoint::ChatCompletions, Endpoint::Messages) => openai_req_to_anthropic(chat, up.claude),
        (Endpoint::Messages, Endpoint::ChatCompletions) => anthropic_req_to_openai(v, up.openai),
        (Endpoint::Responses, Endpoint::ChatCompletions) => responses_req_to_openai(v, up.openai),
        (Endpoint::ChatCompletions, Endpoint::Responses) => {
            openai_req_to_responses(chat, up.openai)
        }
        (Endpoint::Responses, Endpoint::Messages) => {
            let mut chat = responses_req_to_openai(v, up.openai);
            // Intermediate only (never sent): `reasoning.summary` has no Chat Completions field, but
            // it decides `thinking.display` on Messages.
            if let (Some(r), Some(obj)) = (v.get("reasoning"), chat.as_object_mut()) {
                obj.insert("reasoning".into(), r.clone());
            }
            openai_req_to_anthropic(&chat, up.claude)
        }
        (Endpoint::Messages, Endpoint::Responses) => {
            openai_req_to_responses(&anthropic_req_to_openai(v, up.openai), up.openai)
        }
        (a, b) if a == b => v.clone(),
        _ => v.clone(),
    }
}

/// Map a non-stream JSON response from `upstream` into `client`. Error objects are reshaped
/// into the client's error envelope.
pub fn response_json(upstream: Endpoint, client: Endpoint, body: &[u8]) -> Vec<u8> {
    if upstream == client {
        return body.to_vec();
    }
    let Ok(v) = serde_json::from_slice::<Value>(body) else {
        return body.to_vec();
    };
    if looks_like_error(&v) {
        return encode(&map_error(&v, client));
    }
    encode(&map_response(upstream, client, &v))
}

fn map_response(upstream: Endpoint, client: Endpoint, v: &Value) -> Value {
    match (upstream, client) {
        (Endpoint::Messages, Endpoint::ChatCompletions) => anthropic_resp_to_openai(v),
        (Endpoint::ChatCompletions, Endpoint::Messages) => openai_resp_to_anthropic(v),
        (Endpoint::ChatCompletions, Endpoint::Responses) => openai_resp_to_responses(v),
        (Endpoint::Responses, Endpoint::ChatCompletions) => responses_resp_to_openai(v),
        (Endpoint::Messages, Endpoint::Responses) => {
            openai_resp_to_responses(&anthropic_resp_to_openai(v))
        }
        (Endpoint::Responses, Endpoint::Messages) => {
            openai_resp_to_anthropic(&responses_resp_to_openai(v))
        }
        (a, b) if a == b => v.clone(),
        _ => v.clone(),
    }
}

fn encode(v: &Value) -> Vec<u8> {
    serde_json::to_vec(v).unwrap_or_else(|_| {
        br#"{"error":{"message":"translate failed","type":"api_error"}}"#.to_vec()
    })
}

fn looks_like_error(v: &Value) -> bool {
    if v.get("error").is_none() {
        return false;
    }
    // Success bodies never carry a top-level `error` next to `choices` / `content` / `output`.
    if v.get("choices").is_some() || v.get("content").is_some() || v.get("output").is_some() {
        return false;
    }
    if v.get("type").and_then(Value::as_str) == Some("message") {
        return false;
    }
    if v.get("object").and_then(Value::as_str) == Some("response") {
        return false;
    }
    true
}

fn map_error(v: &Value, client: Endpoint) -> Value {
    let (typ, msg) = extract_error(v);
    match client {
        Endpoint::Messages => json!({ "type": "error", "error": { "type": typ, "message": msg } }),
        Endpoint::ChatCompletions | Endpoint::Responses | Endpoint::Embeddings => {
            json!({ "error": { "message": msg, "type": typ } })
        }
    }
}

fn extract_error(v: &Value) -> (String, String) {
    let err = v.get("error").unwrap_or(v);
    let typ = err
        .get("type")
        .and_then(Value::as_str)
        .or_else(|| v.get("type").and_then(Value::as_str))
        .unwrap_or("api_error");
    let msg = err
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("upstream error");
    (typ.to_owned(), msg.to_owned())
}

// --- request: OpenAI → Anthropic --------------------------------------------

fn openai_req_to_anthropic(v: &Value, claude: ClaudeModel) -> Value {
    let mut out = Map::new();
    copy_if(&mut out, v, "model");
    let max_tokens = max_tokens_of(v).unwrap_or(DEFAULT_MAX_TOKENS);
    out.insert("max_tokens".into(), json!(max_tokens));
    openai_reasoning_to_anthropic(v, &mut out, claude, max_tokens);
    copy_if(&mut out, v, "stream");
    if let Some(format) = openai_format_to_anthropic(v.get("response_format")) {
        set_output_config(&mut out, "format", format);
    }
    if let Some(user) = v
        .get("user")
        .or_else(|| v.get("safety_identifier"))
        .and_then(Value::as_str)
    {
        out.insert(
            "metadata".into(),
            json!({ "user_id": anthropic_user_id(user) }),
        );
    }
    // No Anthropic equivalent, and each changes what the client gets back. Forwarded verbatim so
    // the provider rejects them by name; dropping them would answer a different question.
    for key in ["audio", "web_search_options", "top_logprobs"] {
        copy_if(&mut out, v, key);
    }
    if v.get("n").and_then(Value::as_u64).is_some_and(|n| n > 1) {
        copy_if(&mut out, v, "n");
    }
    if v.get("logprobs").and_then(Value::as_bool) == Some(true) {
        copy_if(&mut out, v, "logprobs");
    }
    if v.get("modalities")
        .and_then(Value::as_array)
        .is_some_and(|m| m.iter().any(|x| x != "text"))
    {
        copy_if(&mut out, v, "modalities");
    }
    if let Some(stop) = v.get("stop") {
        match stop {
            Value::String(s) => {
                out.insert("stop_sequences".into(), json!([s]));
            }
            Value::Array(_) => {
                out.insert("stop_sequences".into(), stop.clone());
            }
            _ => {}
        }
    }
    if let Some(tools) = v.get("tools").and_then(Value::as_array) {
        let mapped: Vec<Value> = tools.iter().filter_map(openai_tool_to_anthropic).collect();
        if !mapped.is_empty() {
            out.insert("tools".into(), Value::Array(mapped));
        }
    }
    // Models that reject forced tool use get `auto` plus a closing system instruction naming what
    // must be called (appended after the messages below).
    let mut must_call: Option<String> = None;
    if let Some(choice) = v.get("tool_choice") {
        let mut mapped = openai_tool_choice_to_anthropic(choice);
        let forced = matches!(
            mapped.get("type").and_then(Value::as_str),
            Some("any" | "tool")
        );
        // Budget-thinking models 400 on thinking alongside forced tool use ("Thinking may not be
        // enabled when tool_choice forces tool use"). The forced call is the client's contract (it
        // expects a tool call back); reasoning is a quality hint. The contract wins.
        if forced
            && claude.forced_tool_choice
            && out.get("thinking").and_then(|t| t.get("type")) == Some(&json!("enabled"))
        {
            out.remove("thinking");
        }
        if !claude.forced_tool_choice {
            must_call = match mapped.get("type").and_then(Value::as_str) {
                Some("any") => Some("Respond by calling one of the provided tools.".to_owned()),
                Some("tool") => mapped
                    .get("name")
                    .and_then(Value::as_str)
                    .map(|n| format!("Respond by calling the `{n}` tool.")),
                _ => None,
            };
            if must_call.is_some() {
                mapped = json!({ "type": "auto" });
            }
        }
        out.insert("tool_choice".into(), mapped);
    }
    if v.get("parallel_tool_calls").and_then(Value::as_bool) == Some(false)
        && out.contains_key("tools")
    {
        let choice = out
            .entry("tool_choice")
            .or_insert_with(|| json!({ "type": "auto" }));
        if choice.get("type").and_then(Value::as_str) != Some("none")
            && let Some(obj) = choice.as_object_mut()
        {
            obj.insert("disable_parallel_tool_use".into(), json!(true));
        }
    }
    openai_sampling_to_anthropic(v, &mut out, claude);

    let mut system_parts: Vec<Value> = Vec::new();
    let mut messages: Vec<Value> = Vec::new();
    if let Some(arr) = v.get("messages").and_then(Value::as_array) {
        let mut pending_tool_results: Vec<Value> = Vec::new();
        // Mid-conversation system messages waiting for a place Messages accepts them.
        let mut pending_system: Vec<Value> = Vec::new();
        let mut in_conversation = false;
        for m in arr {
            let role = m.get("role").and_then(Value::as_str).unwrap_or("");
            match role {
                "system" | "developer" if !in_conversation => {
                    system_parts.extend(openai_system_blocks(m));
                }
                "system" | "developer" if claude.mid_system => {
                    flush_tool_results(&mut messages, &mut pending_tool_results);
                    pending_system.extend(openai_system_blocks(m));
                }
                "system" | "developer" => {
                    // No mid-conversation system role on this model: the top-level prompt is the
                    // only place the instruction keeps system authority.
                    flush_tool_results(&mut messages, &mut pending_tool_results);
                    system_parts.extend(openai_system_blocks(m));
                }
                "tool" => {
                    in_conversation = true;
                    if let Some(tr) = openai_tool_result(m) {
                        pending_tool_results.push(tr);
                    }
                }
                "assistant" => {
                    in_conversation = true;
                    flush_tool_results(&mut messages, &mut pending_tool_results);
                    place_system(&mut messages, &mut pending_system, &mut system_parts);
                    let m = assistant_refusal_as_text(m);
                    push_anth_message(&mut messages, "assistant", openai_assistant_content(&m));
                }
                _ => {
                    // user (and anything else treated as user)
                    in_conversation = true;
                    flush_tool_results(&mut messages, &mut pending_tool_results);
                    push_anth_message(&mut messages, "user", openai_user_content(m));
                }
            }
        }
        flush_tool_results(&mut messages, &mut pending_tool_results);
        place_system(&mut messages, &mut pending_system, &mut system_parts);
    }
    sanitize_tool_ids(&mut messages);
    if !system_parts.is_empty() {
        out.insert("system".into(), anthropic_system_value(system_parts));
    }
    out.insert("messages".into(), Value::Array(messages));
    auto_cache_breakpoints(&mut out);
    // After the breakpoints, so the instruction sits past the cached prefix. A mid-conversation
    // system message must follow a user turn (or another system message that does); when the
    // request ends on an assistant turn (a prefill, which these models also reject), there is
    // nowhere valid to put it.
    if let Some(text) = must_call
        && let Some(messages) = out.get_mut("messages").and_then(Value::as_array_mut)
        && messages.last().is_some_and(|m| {
            matches!(
                m.get("role").and_then(Value::as_str),
                Some("user" | "system")
            )
        })
    {
        messages.push(json!({ "role": "system", "content": text }));
    }
    if claude.binding_controls {
        bind_thinking_with_drop_block(&mut out);
    }
    Value::Object(out)
}

/// Emit mid-conversation system messages where Messages accepts one: right after a user turn,
/// followed by the next assistant turn or ending the array. Called before each assistant message
/// and at the end, so a system message between two user turns, or right after an assistant turn,
/// moves past the next user turn. That keeps it in the conversation (appending one never rewrites
/// the prefix earlier thinking blocks are bound to, nor the prompt cache) and still ahead of the
/// reply it is meant to shape. With no user turn to follow (the request ends on an assistant
/// turn), it joins the top-level prompt, the only other place it keeps system authority.
fn place_system(
    messages: &mut Vec<Value>,
    pending: &mut Vec<Value>,
    system_parts: &mut Vec<Value>,
) {
    if pending.is_empty() {
        return;
    }
    let blocks = std::mem::take(pending);
    if messages
        .last()
        .is_some_and(|m| m.get("role").and_then(Value::as_str) == Some("user"))
    {
        messages.push(json!({ "role": "system", "content": anthropic_system_value(blocks) }));
    } else {
        system_parts.extend(blocks);
    }
}

/// The `anthropic-beta` value that lets a translated request set
/// `thinking.block_binding.prefix_mismatch_behavior`.
pub const THINKING_BINDING_BETA: &str = "thinking-binding-controls-2026-08-01";

/// The `anthropic-beta` value a request translated onto Messages needs for `upstream_model`, if
/// any. `proxy` sends it on that attempt's request headers, which leave before the body is
/// translated, so it is decided from the model id alone (the same fact [`request`] uses to add
/// `block_binding`).
pub fn messages_beta(upstream_model: &str) -> Option<&'static str> {
    ClaudeModel::of(upstream_model)
        .binding_controls
        .then_some(THINKING_BINDING_BETA)
}

/// Preserved thinking on a translated conversation: never let the conversation check 400 it.
///
/// On models that bind a thinking block to the conversation that produced it (Claude Fable 5.1,
/// Opus 5.5, Sonnet 5.5), a replayed block whose prefix changed is a 400 on enforced accounts. A
/// translated conversation cannot promise an unchanged prefix: the forced-tool instruction above
/// exists only in the request the gateway built, so the client's next turn replays that turn's
/// thinking without it. `drop_block` makes the API drop such a block (and every one after it)
/// instead of failing. Set on every request to these models, not only ones replaying thinking:
/// a thinking parameter that changes between turns would also restart the prompt cache.
///
/// Requires [`THINKING_BINDING_BETA`], which `proxy` adds for the same models. Not with
/// `between_tools` (the API rejects `block_binding` there) or with thinking off.
fn bind_thinking_with_drop_block(out: &mut Map<String, Value>) {
    let binding = json!({ "prefix_mismatch_behavior": "drop_block" });
    match out.get_mut("thinking") {
        Some(Value::Object(t)) => {
            if matches!(
                t.get("type").and_then(Value::as_str),
                Some("adaptive" | "enabled")
            ) {
                t.insert("block_binding".into(), binding);
            }
        }
        // These models think by default; an explicit `adaptive` is the same request.
        None => {
            out.insert(
                "thinking".into(),
                json!({ "type": "adaptive", "block_binding": binding }),
            );
        }
        Some(_) => {}
    }
}

/// Sampling onto Messages. Current Claude models 400 on non-default sampling, and every model 400s
/// on it alongside thinking. OpenAI clients send `temperature` by habit; it is a hint, so it goes.
/// Where it stays: Anthropic's range is 0–1 (OpenAI's is 0–2), and Claude 4.5-era models reject
/// `temperature` and `top_p` together, so `top_p` yields to `temperature`.
fn openai_sampling_to_anthropic(v: &Value, out: &mut Map<String, Value>, claude: ClaudeModel) {
    let thinking_on = out
        .get("thinking")
        .and_then(|t| t.get("type"))
        .and_then(Value::as_str)
        .is_some_and(|t| t != "disabled");
    if claude.reasoning == ClaudeGen::Adaptive || thinking_on {
        return;
    }
    if let Some(t) = v.get("temperature").and_then(Value::as_f64) {
        out.insert("temperature".into(), json!(t.clamp(0.0, 1.0)));
    } else {
        copy_if(out, v, "top_p");
    }
}

/// A Chat Completions assistant turn with its refusal as text. A refusal is what the assistant
/// said; Messages has no refusal part, and dropping it would leave an empty turn.
fn assistant_refusal_as_text(m: &Value) -> std::borrow::Cow<'_, Value> {
    use std::borrow::Cow;
    let content_empty = match m.get("content") {
        None | Some(Value::Null) => true,
        Some(Value::String(s)) => s.is_empty(),
        Some(Value::Array(a)) => a.is_empty(),
        _ => false,
    };
    // The `refusal` field stands in for content only when there is none.
    let refusal = m
        .get("refusal")
        .and_then(Value::as_str)
        .filter(|r| !r.is_empty() && content_empty);
    let has_part = m.get("content").and_then(Value::as_array).is_some_and(|p| {
        p.iter()
            .any(|x| x.get("type").and_then(Value::as_str) == Some("refusal"))
    });
    if !has_part && refusal.is_none() {
        return Cow::Borrowed(m);
    }
    let mut m = m.clone();
    if let Some(parts) = m.get_mut("content").and_then(Value::as_array_mut) {
        for p in parts.iter_mut() {
            if p.get("type").and_then(Value::as_str) == Some("refusal") {
                let text = p.get("refusal").cloned().unwrap_or(json!(""));
                *p = json!({ "type": "text", "text": text });
            }
        }
    }
    if let Some(r) = refusal {
        m["content"] = json!(r);
    }
    Cow::Owned(m)
}

/// Anthropic tool ids must match `^[a-zA-Z0-9_-]+$`; other providers mint ids like
/// `functions.get_weather:0` (Kimi). Rewritten the same way on the `tool_use` and on its
/// `tool_result`, so the pair still matches, and on every turn, so the history stays byte-stable.
fn sanitize_tool_ids(messages: &mut [Value]) {
    for m in messages {
        let Some(blocks) = m.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        for b in blocks {
            let key = match b.get("type").and_then(Value::as_str) {
                Some("tool_use") => "id",
                Some("tool_result") => "tool_use_id",
                _ => continue,
            };
            if let Some(id) = b.get(key).and_then(Value::as_str)
                && let Some(clean) = anthropic_tool_id(id)
            {
                b[key] = json!(clean);
            }
        }
    }
}

/// `None` when `id` is already valid. Otherwise each disallowed character becomes `_`, plus a hash
/// of the original so two ids that differ only in those characters stay distinct.
fn anthropic_tool_id(id: &str) -> Option<String> {
    let ok = |c: char| c.is_ascii_alphanumeric() || c == '_' || c == '-';
    if !id.is_empty() && id.chars().all(ok) {
        return None;
    }
    let mut clean: String = id.chars().map(|c| if ok(c) { c } else { '_' }).collect();
    clean.push_str(&format!("_{:016x}", fnv1a64(id.as_bytes())));
    Some(clean)
}

/// Anthropic rejects a `metadata.user_id` that looks like an email address ("send a uuid or hash
/// value instead"; measured: names, phone numbers and 300-character ids pass). Such a value becomes
/// a hash of itself, so the same user still maps to the same id and the address itself never
/// reaches the provider. It is an abuse-tracking key, not a secret: an unsalted hash of a known
/// address is recomputable. Anything else passes unchanged.
fn anthropic_user_id(user: &str) -> String {
    if user.contains('@') {
        format!("{:016x}", fnv1a64(user.as_bytes()))
    } else {
        user.to_owned()
    }
}

/// FNV-1a, 64-bit: stable across builds, platforms and processes (unlike `std`'s hasher), which is
/// what an id that must repeat on every turn needs.
fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Chat Completions' pre-`tools` function calling (`functions`, `function_call`, an assistant
/// `function_call`, `role: "function"` results) rewritten as the `tools` loop it is equivalent to,
/// so every mapping below sees one shape. `None` when the body uses none of it. A legacy call has
/// no id; it gets one derived from its message position, which the `function` result that answers
/// it (the next one) reuses, and which is the same on every turn.
fn legacy_functions_to_tools(v: &Value) -> Option<Value> {
    let msgs = v.get("messages").and_then(Value::as_array);
    let legacy_history = msgs.is_some_and(|ms| {
        ms.iter().any(|m| {
            m.get("function_call").is_some()
                || m.get("role").and_then(Value::as_str) == Some("function")
        })
    });
    let legacy_tools = v.get("tools").is_none() && v.get("functions").is_some();
    let legacy_choice = v.get("tool_choice").is_none() && v.get("function_call").is_some();
    if !legacy_history && !legacy_tools && !legacy_choice {
        return None;
    }
    let mut out = v.clone();
    let obj = out.as_object_mut()?;
    if legacy_tools && let Some(Value::Array(fs)) = obj.remove("functions") {
        let tools = fs
            .into_iter()
            .map(|f| json!({ "type": "function", "function": f }))
            .collect();
        obj.insert("tools".into(), Value::Array(tools));
    }
    if legacy_choice && let Some(fc) = obj.remove("function_call") {
        let choice = match fc {
            Value::Object(o) => match o.get("name") {
                Some(name) => json!({ "type": "function", "function": { "name": name } }),
                None => json!("auto"),
            },
            other => other,
        };
        obj.insert("tool_choice".into(), choice);
    }
    if legacy_history && let Some(Value::Array(ms)) = obj.get_mut("messages") {
        let mut last_call: Option<String> = None;
        for (i, m) in ms.iter_mut().enumerate() {
            let Some(mo) = m.as_object_mut() else {
                continue;
            };
            if mo.get("role").and_then(Value::as_str) == Some("function") {
                let id = last_call
                    .take()
                    .unwrap_or_else(|| format!("call_legacy_{i}"));
                mo.insert("role".into(), json!("tool"));
                mo.insert("tool_call_id".into(), json!(id));
                mo.remove("name");
                continue;
            }
            if let Some(fc) = mo.remove("function_call")
                && !mo.contains_key("tool_calls")
            {
                let id = format!("call_legacy_{i}");
                mo.insert(
                    "tool_calls".into(),
                    json!([{ "id": id, "type": "function", "function": fc }]),
                );
                last_call = Some(id);
            }
        }
    }
    Some(out)
}

/// Default prompt-cache breakpoints for a translated request that set none.
///
/// OpenAI caches a repeated prefix on its own; Anthropic caches only up to an explicit
/// `cache_control` marker. A stock OpenAI SDK never sends one, so every Chat Completions or
/// Responses call that translated onto Messages paid full input price for its whole prefix on
/// every turn: an agent loop re-bought its system prompt, tools and history each request.
///
/// A client that marked anything keeps full control: one `cache_control` anywhere and nothing is
/// added. Otherwise, at most two markers (Anthropic allows four):
///
/// - **The static prefix**: the last system block, or the last tool when there is no system.
///   Tools render before system, so one marker covers both. An app's system prompt and tools repeat
///   across its requests, so this is written once and read on every later call.
/// - **The conversation so far**: the last cacheable block of the last message, only once the
///   request holds an assistant turn. The next turn appends to this prefix and reads it back
///   (Anthropic looks back up to 20 blocks for the previous marker). A single-turn request skips it:
///   a write costs 1.25× input and a one-shot never reads it.
///
/// Prefixes shorter than the model's minimum (1024 tokens on most Claude models) are simply not
/// cached; the marker costs nothing there.
fn auto_cache_breakpoints(out: &mut Map<String, Value>) {
    if has_cache_control(out) {
        return;
    }
    let ephemeral = || json!({ "type": "ephemeral" });
    let mut marked_prefix = false;
    if let Some(system) = out.get_mut("system") {
        if let Value::String(t) = system
            && !t.is_empty()
        {
            *system = json!([{ "type": "text", "text": std::mem::take(t) }]);
        }
        if let Some(last) = system.as_array_mut().and_then(|a| a.last_mut())
            && let Some(obj) = last.as_object_mut()
        {
            obj.insert("cache_control".into(), ephemeral());
            marked_prefix = true;
        }
    }
    if !marked_prefix
        && let Some(last) = out
            .get_mut("tools")
            .and_then(Value::as_array_mut)
            .and_then(|a| a.last_mut())
            .and_then(Value::as_object_mut)
    {
        last.insert("cache_control".into(), ephemeral());
    }
    let Some(messages) = out.get_mut("messages").and_then(Value::as_array_mut) else {
        return;
    };
    if !messages
        .iter()
        .any(|m| m.get("role").and_then(Value::as_str) == Some("assistant"))
    {
        return;
    }
    let Some(last) = messages.last_mut() else {
        return;
    };
    if let Some(Value::String(t)) = last.get("content")
        && !t.is_empty()
    {
        let t = t.clone();
        last["content"] = json!([{ "type": "text", "text": t }]);
    }
    // Thinking blocks cannot carry a marker; walk back to the last block that can.
    if let Some(block) = last
        .get_mut("content")
        .and_then(Value::as_array_mut)
        .and_then(|blocks| {
            blocks.iter_mut().rev().find(|b| {
                !matches!(
                    b.get("type").and_then(Value::as_str),
                    Some("thinking" | "redacted_thinking")
                )
            })
        })
        .and_then(Value::as_object_mut)
    {
        block.insert("cache_control".into(), ephemeral());
    }
}

/// Whether the translated request already carries a `cache_control` on a tool, a system block,
/// or a message content block.
fn has_cache_control(out: &Map<String, Value>) -> bool {
    let marked = |v: &Value| v.get("cache_control").is_some();
    let in_array = |v: Option<&Value>| {
        v.and_then(Value::as_array)
            .is_some_and(|a| a.iter().any(marked))
    };
    in_array(out.get("tools"))
        || in_array(out.get("system"))
        || out
            .get("messages")
            .and_then(Value::as_array)
            .is_some_and(|ms| ms.iter().any(|m| in_array(m.get("content"))))
}

fn flush_tool_results(messages: &mut Vec<Value>, pending: &mut Vec<Value>) {
    if pending.is_empty() {
        return;
    }
    let blocks = std::mem::take(pending);
    push_anth_message(messages, "user", Value::Array(blocks));
}

fn push_anth_message(messages: &mut Vec<Value>, role: &str, content: Value) {
    if let Some(last) = messages.last_mut()
        && last.get("role").and_then(Value::as_str) == Some(role)
    {
        merge_content(last, content);
        return;
    }
    messages.push(json!({ "role": role, "content": content }));
}

fn merge_content(msg: &mut Value, add: Value) {
    let existing = msg.get_mut("content");
    let Some(existing) = existing else {
        msg["content"] = add;
        return;
    };
    match (&*existing, &add) {
        (Value::Array(a), Value::Array(b)) => {
            let mut n = a.clone();
            n.extend(b.iter().cloned());
            *existing = Value::Array(n);
        }
        (Value::String(a), Value::String(b)) => {
            *existing = Value::String(format!("{a}{b}"));
        }
        (Value::String(a), Value::Array(b)) => {
            let mut n = vec![json!({ "type": "text", "text": a })];
            n.extend(b.iter().cloned());
            *existing = Value::Array(n);
        }
        (Value::Array(a), Value::String(b)) => {
            let mut n = a.clone();
            n.push(json!({ "type": "text", "text": b }));
            *existing = Value::Array(n);
        }
        _ => *existing = add,
    }
}

fn openai_tool_to_anthropic(t: &Value) -> Option<Value> {
    // `custom` (free-form input) and hosted tools have no Messages equivalent. Forwarded for the
    // provider to reject by name: dropping one would let the model answer without a tool the
    // client offered.
    if t.get("type")
        .and_then(Value::as_str)
        .is_some_and(|ty| ty != "function")
    {
        return Some(t.clone());
    }
    let func = t.get("function").unwrap_or(t);
    let name = func.get("name").and_then(Value::as_str)?;
    let mut m = Map::new();
    m.insert("name".into(), json!(name));
    if let Some(d) = func.get("description") {
        m.insert("description".into(), d.clone());
    }
    let schema = func
        .get("parameters")
        .cloned()
        .unwrap_or_else(|| json!({ "type": "object", "properties": {} }));
    m.insert("input_schema".into(), schema);
    if func.get("strict").and_then(Value::as_bool) == Some(true) {
        m.insert("strict".into(), json!(true));
    }
    copy_cache_control(&mut m, t);
    if !m.contains_key("cache_control") {
        copy_cache_control(&mut m, func);
    }
    Some(Value::Object(m))
}

/// Which reasoning and sampling controls a Claude model accepts. Decided from the id the candidate
/// will receive, because the same OpenAI `reasoning_effort` must become different requests: the
/// old `thinking.budget_tokens` shape is a 400 on every current model, and adaptive thinking is a
/// 400 before 4.6.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ClaudeGen {
    /// Claude 3.x through 4.5, and non-Claude models behind a Messages-compatible API:
    /// `thinking: {type: enabled, budget_tokens}` (less than `max_tokens`), sampling allowed
    /// without thinking.
    Budget,
    /// Claude 4.6: adaptive thinking with `output_config.effort` `low`..`max` (no `xhigh`),
    /// sampling allowed without thinking.
    Adaptive46,
    /// Claude 4.7 and later, Fable, Mythos: adaptive thinking and effort (`xhigh` included).
    /// `budget_tokens` and non-default `temperature` / `top_p` are 400s, and several reject
    /// `thinking: disabled`, so "no reasoning" is an omitted `thinking` at effort `low`.
    Adaptive,
}

/// What the Claude model behind a candidate accepts, parsed from the id it will receive (any
/// spelling: `claude-opus-4-8`, OpenRouter's `anthropic/claude-opus-4.8`, Bedrock's
/// `global.anthropic.claude-haiku-4-5-…`). A dated suffix (`claude-sonnet-4-20250514`) is not a
/// minor version. An unrecognized Claude family is assumed to behave like the newest models; a
/// model that is not Claude at all keeps the long-standing shapes Messages-compatible APIs
/// implement (`budget_tokens`, forced `tool_choice`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ClaudeModel {
    pub(crate) reasoning: ClaudeGen,
    /// Whether forced `tool_choice` (`any` / `tool`) is accepted. Claude Fable 5.1, Mythos 5.1,
    /// Opus 5.5 and Sonnet 5.5 reject it with a 400.
    pub(crate) forced_tool_choice: bool,
    /// Whether `{"role": "system"}` may appear inside `messages`: Opus 4.8 and later, Fable,
    /// Mythos, Sonnet 5.5 and later. Not Sonnet 5 or older (the docs list it as unsupported), and
    /// never a model that is not Claude.
    pub(crate) mid_system: bool,
    /// Whether the request should carry `thinking.block_binding` (see
    /// `bind_thinking_with_drop_block`): a model that binds thinking to its conversation (Fable
    /// 5.1, Opus 5.5 and Sonnet 5.5 and later; not Mythos 5.1, which skips that check), spelled as
    /// Anthropic's own API spells it. Bedrock (`…anthropic.claude-…`) and OpenRouter
    /// (`anthropic/…`) spellings are excluded: whether those hosts accept the beta header that the
    /// field requires is unverified, and a field without its header is a 400.
    pub(crate) binding_controls: bool,
    /// Whether `thinking: {type: between_tools}` exists: Sonnet 5.5 and later Sonnets only.
    pub(crate) between_tools: bool,
}

impl ClaudeModel {
    pub(crate) fn of(model: &str) -> Self {
        const NOT_CLAUDE: ClaudeModel = ClaudeModel {
            reasoning: ClaudeGen::Budget,
            forced_tool_choice: true,
            mid_system: false,
            binding_controls: false,
            between_tools: false,
        };
        let Some(at) = model.find("claude-") else {
            return NOT_CLAUDE;
        };
        let first_party = at == 0;
        let newest = ClaudeModel {
            reasoning: ClaudeGen::Adaptive,
            forced_tool_choice: false,
            mid_system: true,
            binding_controls: first_party,
            between_tools: false,
        };
        let mut parts = model[at + "claude-".len()..].split(['-', '.']);
        let family = parts.next().unwrap_or("");
        if family.starts_with(|c: char| c.is_ascii_digit()) {
            // `claude-3-haiku`, `claude-3-5-sonnet-…`.
            return NOT_CLAUDE;
        }
        let Some(major) = parts.next().and_then(|p| p.parse::<u32>().ok()) else {
            return newest;
        };
        let minor = parts
            .next()
            .filter(|p| p.len() <= 2)
            .and_then(|p| p.parse::<u32>().ok())
            .unwrap_or(0);
        let version = (major, minor);
        match family {
            "opus" | "sonnet" | "haiku" => ClaudeModel {
                reasoning: match version {
                    (5.., _) | (4, 7..) => ClaudeGen::Adaptive,
                    (4, 6) => ClaudeGen::Adaptive46,
                    _ => ClaudeGen::Budget,
                },
                forced_tool_choice: version < (5, 5),
                mid_system: match family {
                    "opus" => version >= (4, 8),
                    _ => version >= (5, 5),
                },
                binding_controls: first_party && version >= (5, 5),
                between_tools: family == "sonnet" && version >= (5, 5),
            },
            "fable" | "mythos" => ClaudeModel {
                reasoning: ClaudeGen::Adaptive,
                forced_tool_choice: version < (5, 1),
                mid_system: true,
                binding_controls: first_party && family == "fable" && version >= (5, 1),
                between_tools: false,
            },
            _ => newest,
        }
    }
}

/// Which reasoning and sampling controls an OpenAI model accepts, parsed from the id this
/// attempt's candidate receives. Measured live on 2026-09-30 against Chat Completions and
/// Responses; where the two differ (o-series `xhigh`, GPT-6 `max`), the set is what both accept.
///
/// Applied only to OpenAI's own ids. Other hosts (xAI, DeepSeek, Groq's `openai/gpt-oss-…`) keep
/// every field as the client sent it. OpenRouter's `openai/…` normalizes `max_tokens` and drops
/// sampling on its own, but passes the effort through (`none` on GPT-5 is "Reasoning is mandatory",
/// `xhigh` on o4-mini a 400), so only the effort is clamped there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OpenAiModel {
    /// OpenAI's own API (bare id): `max_tokens` is rejected on reasoning models, and non-default
    /// sampling is rejected unless reasoning is off.
    pub(crate) native: bool,
    pub(crate) reasoning: OpenAiReasoning,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpenAiReasoning {
    /// Not an OpenAI model this table knows: effort and sampling pass through.
    Unknown,
    /// No reasoning (`gpt-4`, `gpt-4o`, `gpt-4.1`): `reasoning_effort` is an unrecognized argument.
    Off,
    /// A reasoning model and the `reasoning_effort` values it accepts, lowest first.
    Efforts(&'static [&'static str]),
}

/// Effort names, lowest first. Anthropic's `max` is the top rung.
const EFFORT_LADDER: [&str; 7] = ["none", "minimal", "low", "medium", "high", "xhigh", "max"];

impl OpenAiModel {
    const UNKNOWN: Self = Self {
        native: false,
        reasoning: OpenAiReasoning::Unknown,
    };

    pub(crate) fn of(model: &str) -> Self {
        let (native, id) = match model.split_once('/') {
            None => (true, model),
            Some(("openai", id)) => (false, id),
            Some(_) => return Self::UNKNOWN,
        };
        let is_openai = id.starts_with("chatgpt-")
            || (id.starts_with("gpt-") && !id.starts_with("gpt-oss"))
            || id
                .strip_prefix('o')
                .is_some_and(|r| r.starts_with(|c: char| c.is_ascii_digit()));
        if !is_openai {
            return Self::UNKNOWN;
        }
        Self {
            native,
            reasoning: openai_reasoning_of(id),
        }
    }

    /// The field that caps output on Chat Completions: OpenAI's own API rejects `max_tokens` on
    /// every reasoning model and accepts `max_completion_tokens` on all of them.
    fn max_tokens_key(self) -> &'static str {
        if self.native {
            "max_completion_tokens"
        } else {
            "max_tokens"
        }
    }

    /// The `reasoning_effort` to send for a requested effort, or `None` to send none. A value this
    /// model lacks becomes the nearest one above it (`none` on a model that always reasons becomes
    /// its lowest), or its highest when nothing is above. An unrecognized name passes through for
    /// the provider to reject by name.
    fn effort(self, requested: &str) -> Option<String> {
        let requested = match requested {
            "off" | "disabled" => "none",
            other => other,
        };
        match self.reasoning {
            OpenAiReasoning::Unknown => Some(requested.to_owned()),
            OpenAiReasoning::Off => None,
            OpenAiReasoning::Efforts(accepted) => {
                let rank = |e: &str| EFFORT_LADDER.iter().position(|x| *x == e);
                let Some(want) = rank(requested) else {
                    return Some(requested.to_owned());
                };
                accepted
                    .iter()
                    .find(|a| rank(a).is_some_and(|r| r >= want))
                    .or_else(|| accepted.last())
                    .map(|e| (*e).to_owned())
            }
        }
    }

    /// Whether `temperature` / `top_p` may go upstream given the effort being sent. OpenAI's
    /// reasoning models accept only the default unless reasoning is `none`; OpenRouter drops them
    /// itself, and a model this table does not know keeps them.
    fn keeps_sampling(self, effort: Option<&str>) -> bool {
        !self.native
            || !matches!(self.reasoning, OpenAiReasoning::Efforts(_))
            || effort == Some("none")
    }
}

/// Accepted efforts by family, measured live (2026-09-30). Pro variants take fewer.
fn openai_reasoning_of(id: &str) -> OpenAiReasoning {
    const O_SERIES: &[&str] = &["low", "medium", "high"];
    const GPT5: &[&str] = &["minimal", "low", "medium", "high"];
    const GPT5_PRO: &[&str] = &["high"];
    const GPT51: &[&str] = &["none", "low", "medium", "high"];
    const GPT52: &[&str] = &["none", "low", "medium", "high", "xhigh"];
    const GPT6: &[&str] = &["low", "medium", "high", "xhigh"];
    const PRO: &[&str] = &["medium", "high", "xhigh"];
    if id.starts_with('o') {
        return OpenAiReasoning::Efforts(O_SERIES);
    }
    let Some(rest) = id.strip_prefix("gpt-") else {
        // `chatgpt-4o-latest`.
        return OpenAiReasoning::Off;
    };
    fn digits(s: &str) -> (Option<u32>, &str) {
        let n = s.bytes().take_while(u8::is_ascii_digit).count();
        (s[..n].parse::<u32>().ok(), &s[n..])
    }
    let (Some(major), after) = digits(rest) else {
        return OpenAiReasoning::Unknown;
    };
    let (minor, suffix) = match after.strip_prefix('.') {
        Some(m) => {
            let (minor, suffix) = digits(m);
            (minor.unwrap_or(0), suffix)
        }
        None => (0, after),
    };
    let pro = suffix.split('-').any(|s| s == "pro");
    if major <= 4 {
        return OpenAiReasoning::Off;
    }
    if suffix.split('-').any(|s| s == "chat") {
        // ChatGPT's instant snapshots (`gpt-5.2-chat`): not verified; leave them alone.
        return OpenAiReasoning::Unknown;
    }
    OpenAiReasoning::Efforts(match (major, minor, pro) {
        (5, 0, true) => GPT5_PRO,
        (5, 0, false) => GPT5,
        (5, 1, _) => GPT51,
        (5, _, true) | (6.., _, true) => PRO,
        (5, _, false) => GPT52,
        _ => GPT6,
    })
}

/// A thinking request, whichever shape the client used, before it is fit to the model.
enum ThinkingIntent<'a> {
    Off,
    BetweenTools,
    Budget(u64),
    Effort(&'a str),
}

fn openai_reasoning_to_anthropic(
    v: &Value,
    out: &mut Map<String, Value>,
    claude: ClaudeModel,
    max_tokens: u64,
) {
    let effort = v
        .get("reasoning_effort")
        .and_then(Value::as_str)
        .or_else(|| v.pointer("/reasoning/effort").and_then(Value::as_str));
    // A client may send Anthropic's own `thinking` object on the OpenAI wire. It is a request for
    // thinking, not a finished shape: `budget_tokens` is a 400 on current models and adaptive a
    // 400 before 4.6, so it goes through the same model-aware mapping as `reasoning_effort`.
    let native = v.get("thinking").filter(|t| t.is_object());
    let intent = match native.and_then(|t| t.get("type")).and_then(Value::as_str) {
        Some("disabled") => ThinkingIntent::Off,
        Some("between_tools") => ThinkingIntent::BetweenTools,
        Some("enabled") => match native
            .and_then(|t| t.get("budget_tokens"))
            .and_then(Value::as_u64)
        {
            Some(n) if effort.is_none() => ThinkingIntent::Budget(n),
            _ => ThinkingIntent::Effort(effort.unwrap_or("medium")),
        },
        Some("adaptive") => ThinkingIntent::Effort(effort.unwrap_or("high")),
        Some(_) => {
            // An unknown thinking type: the provider names it.
            if let Some(t) = native {
                out.insert("thinking".into(), t.clone());
            }
            return;
        }
        None => match effort {
            Some("none" | "off" | "disabled") => ThinkingIntent::Off,
            Some(e) => ThinkingIntent::Effort(e),
            None => return,
        },
    };
    let intent = match intent {
        ThinkingIntent::BetweenTools if !claude.between_tools => ThinkingIntent::Off,
        other => other,
    };
    match claude.reasoning {
        ClaudeGen::Budget => {
            let budget = match intent {
                ThinkingIntent::Off | ThinkingIntent::BetweenTools => {
                    out.insert("thinking".into(), json!({ "type": "disabled" }));
                    return;
                }
                ThinkingIntent::Budget(n) => n,
                ThinkingIntent::Effort(e) => budget_for_effort(e).min(max_tokens / 2),
            };
            // `budget_tokens` must be at least 1024 and below `max_tokens`, which also has to hold
            // the answer. Leave the answer at least half; too small a request thinks not at all.
            if max_tokens <= MIN_THINKING_BUDGET {
                return;
            }
            let budget = budget.clamp(MIN_THINKING_BUDGET, max_tokens - 1);
            let mut t = json!({ "type": "enabled", "budget_tokens": budget });
            copy_display(&mut t, native);
            out.insert("thinking".into(), t);
        }
        ClaudeGen::Adaptive46 | ClaudeGen::Adaptive => {
            let requested = match intent {
                ThinkingIntent::BetweenTools => {
                    out.insert("thinking".into(), json!({ "type": "between_tools" }));
                    return;
                }
                ThinkingIntent::Off => "none",
                ThinkingIntent::Budget(n) => effort_from_budget(n),
                ThinkingIntent::Effort(e) => e,
            };
            let level = match effort_alias(requested) {
                "none" => "low",
                "xhigh" if claude.reasoning == ClaudeGen::Adaptive46 => "max",
                "xhigh" if requested == "max" => "max",
                other => other,
            };
            if !matches!(intent, ThinkingIntent::Off) {
                let mut t = json!({ "type": "adaptive" });
                copy_display(&mut t, native);
                // A Responses client asking for a reasoning summary: current models default to
                // `omitted`, which streams empty thinking blocks.
                if claude.reasoning == ClaudeGen::Adaptive
                    && t.get("display").is_none()
                    && v.pointer("/reasoning/summary")
                        .and_then(Value::as_str)
                        .is_some_and(|s| s != "none")
                    && let Some(obj) = t.as_object_mut()
                {
                    obj.insert("display".into(), json!("summarized"));
                }
                out.insert("thinking".into(), t);
            }
            set_output_config(out, "effort", json!(level));
        }
    }
}

/// Carry a client's `thinking.display` (`summarized` / `omitted` / …) onto the mapped object.
fn copy_display(t: &mut Value, native: Option<&Value>) {
    if let Some(d) = native.and_then(|n| n.get("display"))
        && let Some(obj) = t.as_object_mut()
    {
        obj.insert("display".into(), d.clone());
    }
}

/// Anthropic's floor for `thinking.budget_tokens`.
const MIN_THINKING_BUDGET: u64 = 1024;

/// Set one field of the request's `output_config` (effort, structured-output format).
fn set_output_config(out: &mut Map<String, Value>, key: &str, value: Value) {
    if let Some(Value::Object(m)) = out.get_mut("output_config") {
        m.insert(key.into(), value);
        return;
    }
    let mut m = Map::new();
    m.insert(key.into(), value);
    out.insert("output_config".into(), Value::Object(m));
}

/// OpenAI `response_format` → Anthropic `output_config.format`. Only `json_schema` has an
/// equivalent. `json_object` (JSON mode, no schema) has none: Anthropic's format needs a schema,
/// and OpenAI already requires a JSON-mode prompt to ask for JSON, so it is dropped. `text` is the
/// default.
fn openai_format_to_anthropic(rf: Option<&Value>) -> Option<Value> {
    let rf = rf?;
    if rf.get("type").and_then(Value::as_str) != Some("json_schema") {
        return None;
    }
    let schema = rf.pointer("/json_schema/schema")?;
    Some(json!({ "type": "json_schema", "schema": schema }))
}

/// Anthropic `output_config.format` → OpenAI `response_format`.
fn anthropic_format_to_openai(v: &Value) -> Option<Value> {
    let format = v.pointer("/output_config/format")?;
    if format.get("type").and_then(Value::as_str) != Some("json_schema") {
        return None;
    }
    let schema = format.get("schema")?;
    Some(json!({
        "type": "json_schema",
        "json_schema": {
            "name": "response",
            "strict": openai_strict_schema(schema),
            "schema": schema,
        },
    }))
}

fn budget_for_effort(effort: &str) -> u64 {
    match effort {
        "minimal" | "low" => 1024,
        "high" => 8192,
        "xhigh" | "max" => 16384,
        _ => 4096,
    }
}

fn effort_from_thinking(thinking: &Value) -> Option<&'static str> {
    match thinking.get("type").and_then(Value::as_str) {
        Some("disabled") => Some("none"),
        Some("adaptive") => Some(
            thinking
                .get("effort")
                .and_then(Value::as_str)
                .map(effort_alias)
                .unwrap_or("high"),
        ),
        Some("enabled") => {
            let budget = thinking
                .get("budget_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(4096);
            Some(effort_from_budget(budget))
        }
        _ => None,
    }
}

fn effort_alias(s: &str) -> &'static str {
    match s {
        "none" | "off" | "disabled" => "none",
        "minimal" | "low" => "low",
        "high" => "high",
        "xhigh" | "max" => "xhigh",
        _ => "medium",
    }
}

fn effort_from_budget(budget: u64) -> &'static str {
    if budget <= 1024 {
        "low"
    } else if budget <= 4096 {
        "medium"
    } else if budget <= 8192 {
        "high"
    } else {
        "xhigh"
    }
}

fn copy_cache_control(out: &mut Map<String, Value>, src: &Value) {
    if let Some(cc) = src.get("cache_control") {
        out.insert("cache_control".into(), cc.clone());
    }
}

fn openai_system_blocks(m: &Value) -> Vec<Value> {
    let cc = m.get("cache_control");
    match m.get("content") {
        Some(Value::String(s)) => vec![text_block(s, cc)],
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| {
                let t = p
                    .get("text")
                    .and_then(Value::as_str)
                    .or_else(|| p.as_str())?;
                let part_cc = p.get("cache_control").or(cc);
                Some(text_block(t, part_cc))
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn text_block(text: &str, cache_control: Option<&Value>) -> Value {
    let mut m = json!({ "type": "text", "text": text });
    if let Some(cc) = cache_control
        && let Some(obj) = m.as_object_mut()
    {
        obj.insert("cache_control".into(), cc.clone());
    }
    m
}

fn anthropic_system_value(blocks: Vec<Value>) -> Value {
    if blocks.len() == 1
        && blocks[0].get("cache_control").is_none()
        && let Some(t) = blocks[0].get("text").cloned()
    {
        return t;
    }
    Value::Array(blocks)
}

fn openai_tool_choice_to_anthropic(choice: &Value) -> Value {
    match choice {
        Value::String(s) => match s.as_str() {
            "none" => json!({ "type": "none" }),
            "required" => json!({ "type": "any" }),
            _ => json!({ "type": "auto" }),
        },
        Value::Object(o) => {
            // `allowed_tools` narrows which tools may be called this turn. Messages has no subset
            // (and trimming `tools` would rewrite the cached, thinking-bound prefix), so the mode
            // carries: `required` over one tool forces that tool, over several forces any tool.
            if let Some(allowed) = o.get("allowed_tools") {
                let tools = allowed.get("tools").and_then(Value::as_array);
                if allowed.get("mode").and_then(Value::as_str) != Some("required") {
                    return json!({ "type": "auto" });
                }
                return match tools.map(Vec::as_slice) {
                    Some([only]) => openai_tool_choice_to_anthropic(only),
                    _ => json!({ "type": "any" }),
                };
            }
            if let Some(name) = o
                .get("function")
                .or_else(|| o.get("custom"))
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
                .or_else(|| o.get("name").and_then(Value::as_str))
            {
                json!({ "type": "tool", "name": name })
            } else {
                json!({ "type": "auto" })
            }
        }
        _ => json!({ "type": "auto" }),
    }
}

fn openai_tool_result(m: &Value) -> Option<Value> {
    let id = m
        .get("tool_call_id")
        .and_then(Value::as_str)
        .unwrap_or("call_0");
    // A tool message may carry images (OpenAI accepts `image_url` parts there); `tool_result`
    // content holds images too, so they stay with the result instead of being flattened away.
    let content = match m.get("content") {
        Some(Value::Array(parts))
            if parts.iter().any(|p| {
                p.get("type")
                    .and_then(Value::as_str)
                    .is_some_and(|t| t != "text")
            }) =>
        {
            let blocks: Vec<Value> = parts
                .iter()
                .filter_map(openai_tool_part_to_anthropic)
                .collect();
            Value::Array(blocks)
        }
        _ => json!(message_text(m).unwrap_or_default()),
    };
    Some(json!({
        "type": "tool_result",
        "tool_use_id": id,
        "content": content,
    }))
}

fn openai_tool_part_to_anthropic(p: &Value) -> Option<Value> {
    match p.get("type").and_then(Value::as_str) {
        Some("text") => Some(text_block(
            p.get("text").and_then(Value::as_str)?,
            p.get("cache_control"),
        )),
        Some("image_url") => openai_image_part_to_anthropic(p),
        Some("file") => Some(openai_file_to_anthropic(p)),
        _ => Some(p.clone()),
    }
}

fn openai_assistant_content(m: &Value) -> Value {
    let mut blocks: Vec<Value> = Vec::new();
    match m.get("content") {
        Some(Value::String(s)) if !s.is_empty() => {
            blocks.push(json!({ "type": "text", "text": s }));
        }
        Some(Value::Array(parts)) => {
            for p in parts {
                match p.get("type").and_then(Value::as_str) {
                    Some("thinking") => blocks.push(thinking_block_from_openai(p)),
                    Some("redacted_thinking") => blocks.push(redacted_block_from_openai(p)),
                    Some("text") | None => {
                        if let Some(t) =
                            p.get("text").and_then(Value::as_str).or_else(|| p.as_str())
                            && !t.is_empty()
                        {
                            let mut b = json!({ "type": "text", "text": t });
                            if let Some(obj) = b.as_object_mut() {
                                copy_cache_control(obj, p);
                            }
                            blocks.push(b);
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
    if let Some(extra) = m.get("thinking").and_then(Value::as_array) {
        for p in extra.iter().rev() {
            let block = match p.get("type").and_then(Value::as_str) {
                Some("redacted_thinking") => redacted_block_from_openai(p),
                _ => thinking_block_from_openai(p),
            };
            blocks.insert(0, block);
        }
    } else if blocks
        .iter()
        .all(|b| b.get("type").and_then(Value::as_str) != Some("thinking"))
        && let Some(reasoning) = m
            .get("reasoning_content")
            .and_then(Value::as_str)
            .or_else(|| m.get("reasoning").and_then(Value::as_str))
        && !reasoning.is_empty()
    {
        blocks.insert(0, json!({ "type": "thinking", "thinking": reasoning }));
    }
    if let Some(calls) = m.get("tool_calls").and_then(Value::as_array) {
        for c in calls {
            let id = c.get("id").and_then(Value::as_str).unwrap_or("call_0");
            let func = c.get("function").unwrap_or(c);
            let name = func.get("name").and_then(Value::as_str).unwrap_or("");
            let input = parse_arguments(func.get("arguments"));
            blocks.push(json!({
                "type": "tool_use",
                "id": id,
                "name": name,
                "input": input,
            }));
        }
    }
    if blocks.len() == 1 && blocks[0].get("type").and_then(Value::as_str) == Some("text") {
        return blocks[0]
            .get("text")
            .cloned()
            .unwrap_or(Value::String(String::new()));
    }
    if blocks.is_empty() {
        return Value::String(String::new());
    }
    Value::Array(blocks)
}

fn thinking_block_from_openai(p: &Value) -> Value {
    let mut m = Map::new();
    m.insert("type".into(), json!("thinking"));
    let text = p
        .get("thinking")
        .and_then(Value::as_str)
        .or_else(|| p.get("text").and_then(Value::as_str))
        .unwrap_or("");
    m.insert("thinking".into(), json!(text));
    if let Some(sig) = p.get("signature") {
        m.insert("signature".into(), sig.clone());
    }
    Value::Object(m)
}

fn redacted_block_from_openai(p: &Value) -> Value {
    json!({
        "type": "redacted_thinking",
        "data": p.get("data").cloned().unwrap_or(json!("")),
    })
}

fn openai_user_content(m: &Value) -> Value {
    let msg_cc = m.get("cache_control");
    match m.get("content") {
        Some(Value::String(s)) => {
            if let Some(cc) = msg_cc {
                return Value::Array(vec![text_block(s, Some(cc))]);
            }
            Value::String(s.clone())
        }
        Some(Value::Array(parts)) => {
            let mut blocks: Vec<Value> = Vec::new();
            for p in parts {
                match p.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        if let Some(t) = p.get("text").and_then(Value::as_str) {
                            blocks.push(text_block(t, p.get("cache_control").or(msg_cc)));
                        }
                    }
                    Some("image_url") => {
                        if let Some(mut b) = openai_image_part_to_anthropic(p) {
                            if let Some(obj) = b.as_object_mut() {
                                copy_cache_control(obj, p);
                            }
                            blocks.push(b);
                        }
                    }
                    Some("file") => {
                        let mut b = openai_file_to_anthropic(p);
                        if let Some(obj) = b.as_object_mut() {
                            copy_cache_control(obj, p);
                        }
                        blocks.push(b);
                    }
                    _ => {
                        if let Some(t) = p.get("text").and_then(Value::as_str) {
                            blocks.push(text_block(t, p.get("cache_control").or(msg_cc)));
                        } else {
                            // `input_audio` and anything else without text: no Messages
                            // equivalent. Forwarded so the provider rejects it by name; dropping it
                            // would get an answer about input the model never saw.
                            blocks.push(p.clone());
                        }
                    }
                }
            }
            if blocks.len() == 1
                && blocks[0].get("type").and_then(Value::as_str) == Some("text")
                && blocks[0].get("cache_control").is_none()
            {
                return blocks[0]
                    .get("text")
                    .cloned()
                    .unwrap_or(Value::String(String::new()));
            }
            if blocks.is_empty() {
                return Value::String(String::new());
            }
            Value::Array(blocks)
        }
        _ => Value::String(String::new()),
    }
}

/// Chat Completions `file` part → Anthropic `document`. Inline data (`file_data`, a data URI) maps
/// to a base64 source. A `file_id` names an OpenAI Files upload Anthropic cannot read, so the part
/// is forwarded as-is for the provider to reject by name.
fn openai_file_to_anthropic(part: &Value) -> Value {
    let file = part.get("file");
    let Some((media_type, data)) = file
        .and_then(|f| f.get("file_data"))
        .and_then(Value::as_str)
        .and_then(parse_data_uri)
    else {
        return part.clone();
    };
    let mut doc = json!({
        "type": "document",
        "source": { "type": "base64", "media_type": media_type, "data": data },
    });
    if let Some(name) = file
        .and_then(|f| f.get("filename"))
        .filter(|n| n.is_string())
        && let Some(obj) = doc.as_object_mut()
    {
        obj.insert("title".into(), name.clone());
    }
    doc
}

/// A Chat Completions image part onto Messages. A data URI that is not base64
/// (`data:image/svg+xml;utf8,…`) has no Messages image source: it is forwarded for the provider to
/// reject, since dropped, the model would describe a picture it never saw. Other schemes
/// (`file://`) are never forwarded in any shape.
fn openai_image_part_to_anthropic(p: &Value) -> Option<Value> {
    if let Some(b) = openai_image_to_anthropic(p) {
        return Some(b);
    }
    p.pointer("/image_url/url")
        .or_else(|| p.get("url"))
        .and_then(Value::as_str)
        .is_some_and(|u| u.starts_with("data:"))
        .then(|| p.clone())
}

fn openai_image_to_anthropic(part: &Value) -> Option<Value> {
    let url = part
        .pointer("/image_url/url")
        .and_then(Value::as_str)
        .or_else(|| part.get("url").and_then(Value::as_str))?;
    if is_http_url(url) {
        return Some(json!({
            "type": "image",
            "source": { "type": "url", "url": url },
        }));
    }
    let (media_type, data) = parse_data_uri(url)?;
    Some(json!({
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": data,
        }
    }))
}

/// An image the upstream downloads itself. Carried as a URL on every wire rather than dropped:
/// a silently image-less request produces a confident answer about a picture the model never saw.
fn is_http_url(url: &str) -> bool {
    url.starts_with("https://") || url.starts_with("http://")
}

fn parse_data_uri(url: &str) -> Option<(&str, &str)> {
    let rest = url.strip_prefix("data:")?;
    let (meta, data) = rest.split_once(";base64,")?;
    if meta.is_empty() || data.is_empty() {
        return None;
    }
    Some((meta, data))
}

fn parse_arguments(v: Option<&Value>) -> Value {
    match v {
        Some(Value::String(s)) => serde_json::from_str(s).unwrap_or_else(|_| json!({})),
        Some(o) if o.is_object() => o.clone(),
        _ => json!({}),
    }
}

fn max_tokens_of(v: &Value) -> Option<u64> {
    v.get("max_tokens")
        .and_then(Value::as_u64)
        .or_else(|| v.get("max_completion_tokens").and_then(Value::as_u64))
}

fn message_text(m: &Value) -> Option<String> {
    match m.get("content") {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Array(parts)) => {
            let t: String = parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Value::as_str).or_else(|| p.as_str()))
                .collect();
            Some(t)
        }
        Some(Value::Null) | None => None,
        _ => None,
    }
}

fn copy_if(out: &mut Map<String, Value>, v: &Value, key: &str) {
    if let Some(x) = v.get(key) {
        out.insert(key.to_owned(), x.clone());
    }
}

/// Root-level Responses session field that cannot be honored off `/v1/responses`.
///
/// `None` means the body is a `store: false` one-shot (no `previous_response_id`). Unparseable
/// JSON is `Some("store")` so a catalog walk fail-closes onto a real Responses arm rather than
/// silently stripping session state.
pub fn responses_session_field(body: &[u8]) -> Option<&'static str> {
    let Ok(v) = serde_json::from_slice::<Value>(body) else {
        return Some("store");
    };
    if previous_response_id_set(&v) {
        return Some("previous_response_id");
    }
    match v.get("store") {
        Some(Value::Bool(false)) => None,
        _ => Some("store"),
    }
}

fn previous_response_id_set(v: &Value) -> bool {
    match v.get("previous_response_id") {
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Null) | None => false,
        Some(_) => true,
    }
}

/// Drop Responses-only session/control fields and reshape `input` onto Chat Completions
/// `messages`. Used when a `store: false` one-shot is allowed to leave the Responses endpoint.
/// Same-endpoint Responses must not call this — those fields pass through as a byte relay.
pub fn responses_to_chat(body: &[u8]) -> Vec<u8> {
    request(Endpoint::Responses, Endpoint::ChatCompletions, body, "")
}

// --- request: Anthropic → OpenAI --------------------------------------------

fn anthropic_req_to_openai(v: &Value, openai: OpenAiModel) -> Value {
    let mut out = Map::new();
    copy_if(&mut out, v, "model");
    if let Some(t) = v.get("max_tokens") {
        out.insert(openai.max_tokens_key().into(), t.clone());
    }
    copy_if(&mut out, v, "stream");
    // Remote MCP servers have no Chat Completions equivalent and change what the model can do.
    copy_if(&mut out, v, "mcp_servers");
    if let Some(stop) = v.get("stop_sequences") {
        out.insert("stop".into(), stop.clone());
    }
    if let Some(tools) = v.get("tools").and_then(Value::as_array) {
        let mapped: Vec<Value> = tools.iter().filter_map(anthropic_tool_to_openai).collect();
        if !mapped.is_empty() {
            out.insert("tools".into(), Value::Array(mapped));
        }
    }
    if let Some(choice) = v.get("tool_choice") {
        out.insert(
            "tool_choice".into(),
            anthropic_tool_choice_to_openai(choice),
        );
    }
    // Anthropic's effort lives in `output_config.effort`; `thinking` alone implies one. Fit to what
    // the upstream model accepts (see `OpenAiModel`).
    let effort = v
        .pointer("/output_config/effort")
        .and_then(Value::as_str)
        .map(effort_alias)
        .or_else(|| v.get("thinking").and_then(effort_from_thinking))
        .and_then(|e| openai.effort(e));
    if openai.keeps_sampling(effort.as_deref()) {
        copy_if(&mut out, v, "temperature");
        copy_if(&mut out, v, "top_p");
    }
    if let Some(effort) = effort {
        out.insert("reasoning_effort".into(), json!(effort));
    }
    if let Some(format) = anthropic_format_to_openai(v) {
        out.insert("response_format".into(), format);
    }
    if v.pointer("/tool_choice/disable_parallel_tool_use")
        .and_then(Value::as_bool)
        == Some(true)
    {
        out.insert("parallel_tool_calls".into(), json!(false));
    }
    if let Some(user) = v.pointer("/metadata/user_id").filter(|u| u.is_string()) {
        out.insert("user".into(), user.clone());
    }

    let mut messages: Vec<Value> = Vec::new();
    if let Some(sys) = v.get("system")
        && let Some(msg) = anthropic_system_to_openai(sys)
    {
        messages.push(msg);
    }
    if let Some(arr) = v.get("messages").and_then(Value::as_array) {
        for m in arr {
            let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
            match role {
                "assistant" => messages.push(anthropic_assistant_to_openai(m)),
                // A mid-conversation system message keeps its role and its place. Chat Completions
                // takes `system` anywhere. A directive-only one (`content: []` plus
                // `output_config`) says nothing a Chat model can read, so it has no message.
                "system" => messages.extend(m.get("content").and_then(anthropic_system_to_openai)),
                _ => messages.extend(anthropic_user_to_openai(m)),
            }
        }
    }
    out.insert("messages".into(), Value::Array(messages));
    Value::Object(out)
}

fn anthropic_system_to_openai(sys: &Value) -> Option<Value> {
    match sys {
        Value::String(s) if !s.is_empty() => Some(json!({ "role": "system", "content": s })),
        Value::Array(blocks) => {
            let has_cc = blocks.iter().any(|b| b.get("cache_control").is_some());
            if has_cc {
                let parts: Vec<Value> = blocks
                    .iter()
                    .filter_map(|b| {
                        let t = b
                            .get("text")
                            .and_then(Value::as_str)
                            .or_else(|| b.as_str())?;
                        let mut p = json!({ "type": "text", "text": t });
                        if let Some(obj) = p.as_object_mut() {
                            copy_cache_control(obj, b);
                        }
                        Some(p)
                    })
                    .collect();
                if parts.is_empty() {
                    return None;
                }
                Some(json!({ "role": "system", "content": parts }))
            } else {
                let t = anthropic_system_text(sys)?;
                (!t.is_empty()).then(|| json!({ "role": "system", "content": t }))
            }
        }
        _ => None,
    }
}

fn anthropic_system_text(sys: &Value) -> Option<String> {
    match sys {
        Value::String(s) => Some(s.clone()),
        Value::Array(blocks) => Some(
            blocks
                .iter()
                .filter_map(|b| b.get("text").and_then(Value::as_str).or_else(|| b.as_str()))
                .collect::<Vec<_>>()
                .join("\n"),
        ),
        _ => None,
    }
}

fn anthropic_tool_to_openai(t: &Value) -> Option<Value> {
    // Server tools (`web_search_…`, `code_execution_…`) and Anthropic-defined client tools
    // (`bash_…`, `text_editor_…`, `computer_…`) run a contract no Chat Completions function
    // carries: as an empty-schema function the model would call it and nothing would run it.
    // Forwarded as-is for the provider to reject by name.
    if t.get("type")
        .and_then(Value::as_str)
        .is_some_and(|ty| ty != "custom")
    {
        return Some(t.clone());
    }
    let name = t.get("name").and_then(Value::as_str)?;
    let mut func = Map::new();
    func.insert("name".into(), json!(name));
    if let Some(d) = t.get("description") {
        func.insert("description".into(), d.clone());
    }
    let params = t
        .get("input_schema")
        .cloned()
        .unwrap_or_else(|| json!({ "type": "object", "properties": {} }));
    // OpenAI's strict mode needs a stricter schema than Anthropic's (see `openai_strict_schema`).
    if t.get("strict").and_then(Value::as_bool) == Some(true) {
        func.insert("strict".into(), json!(openai_strict_schema(&params)));
    }
    func.insert("parameters".into(), params);
    let mut tool = json!({ "type": "function", "function": Value::Object(func) });
    if let Some(obj) = tool.as_object_mut() {
        copy_cache_control(obj, t);
    }
    Some(tool)
}

/// Whether OpenAI accepts `schema` under `strict: true`: every object sets
/// `additionalProperties: false` and lists all of its properties in `required`. Anthropic's
/// structured outputs allow optional properties, so a schema valid there is a 400 here with
/// `strict: true`; sent with `strict: false` it still shapes the output, without the guarantee.
fn openai_strict_schema(schema: &Value) -> bool {
    let Some(obj) = schema.as_object() else {
        return true;
    };
    let is_object =
        obj.get("type").and_then(Value::as_str) == Some("object") || obj.contains_key("properties");
    if is_object {
        if obj.get("additionalProperties") != Some(&Value::Bool(false)) {
            return false;
        }
        let required: Vec<&str> = obj
            .get("required")
            .and_then(Value::as_array)
            .map(|r| r.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        if let Some(props) = obj.get("properties").and_then(Value::as_object)
            && !props.keys().all(|k| required.contains(&k.as_str()))
        {
            return false;
        }
    }
    let nested = |key: &str| -> Vec<&Value> {
        match obj.get(key) {
            Some(Value::Object(m)) if matches!(key, "properties" | "$defs" | "definitions") => {
                m.values().collect()
            }
            Some(Value::Array(a)) => a.iter().collect(),
            Some(v @ Value::Object(_)) => vec![v],
            _ => Vec::new(),
        }
    };
    [
        "properties",
        "$defs",
        "definitions",
        "items",
        "prefixItems",
        "anyOf",
        "oneOf",
        "allOf",
    ]
    .into_iter()
    .flat_map(nested)
    .all(openai_strict_schema)
}

fn anthropic_tool_choice_to_openai(choice: &Value) -> Value {
    match choice.get("type").and_then(Value::as_str) {
        Some("none") => json!("none"),
        Some("any") => json!("required"),
        Some("tool") => {
            let name = choice.get("name").and_then(Value::as_str).unwrap_or("");
            json!({ "type": "function", "function": { "name": name } })
        }
        _ => json!("auto"),
    }
}

fn anthropic_assistant_to_openai(m: &Value) -> Value {
    let mut text = String::new();
    let mut thinking_text = String::new();
    let mut thinking_blocks: Vec<Value> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    match m.get("content") {
        Some(Value::String(s)) => text = s.clone(),
        Some(Value::Array(blocks)) => {
            for b in blocks {
                match b.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        if let Some(t) = b.get("text").and_then(Value::as_str) {
                            text.push_str(t);
                        }
                    }
                    Some("thinking") => {
                        if let Some(t) = b
                            .get("thinking")
                            .and_then(Value::as_str)
                            .or_else(|| b.get("text").and_then(Value::as_str))
                        {
                            thinking_text.push_str(t);
                        }
                        thinking_blocks.push(b.clone());
                    }
                    Some("redacted_thinking") => {
                        thinking_blocks.push(b.clone());
                    }
                    Some("tool_use") => {
                        let id = b.get("id").and_then(Value::as_str).unwrap_or("call_0");
                        let name = b.get("name").and_then(Value::as_str).unwrap_or("");
                        let args = b
                            .get("input")
                            .map(|i| serde_json::to_string(i).unwrap_or_else(|_| "{}".into()))
                            .unwrap_or_else(|| "{}".into());
                        tool_calls.push(json!({
                            "id": id,
                            "type": "function",
                            "function": { "name": name, "arguments": args },
                        }));
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
    let mut msg = Map::new();
    msg.insert("role".into(), json!("assistant"));
    if tool_calls.is_empty() {
        msg.insert("content".into(), json!(text));
    } else {
        msg.insert(
            "content".into(),
            if text.is_empty() {
                Value::Null
            } else {
                json!(text)
            },
        );
        msg.insert("tool_calls".into(), Value::Array(tool_calls));
    }
    if !thinking_text.is_empty() {
        msg.insert("reasoning_content".into(), json!(thinking_text));
    }
    if !thinking_blocks.is_empty() {
        msg.insert("thinking".into(), Value::Array(thinking_blocks));
    }
    Value::Object(msg)
}

fn anthropic_user_to_openai(m: &Value) -> Vec<Value> {
    match m.get("content") {
        Some(Value::String(s)) => vec![json!({ "role": "user", "content": s })],
        Some(Value::Array(blocks)) => {
            let mut out = Vec::new();
            let mut user_parts: Vec<Value> = Vec::new();
            let flush_user = |parts: &mut Vec<Value>, out: &mut Vec<Value>| {
                if parts.is_empty() {
                    return;
                }
                let p = std::mem::take(parts);
                if p.len() == 1 && p[0].get("type").and_then(Value::as_str) == Some("text") {
                    out.push(json!({
                        "role": "user",
                        "content": p[0].get("text").cloned().unwrap_or(json!("")),
                    }));
                } else {
                    out.push(json!({ "role": "user", "content": Value::Array(p) }));
                }
            };
            // Chat Completions wants every `tool` message right after the assistant turn that
            // called it, so the tool results go first and the rest of the turn follows as one
            // user message (Anthropic puts `tool_result` blocks first anyway).
            for b in blocks {
                match b.get("type").and_then(Value::as_str) {
                    Some("tool_result") => {
                        out.push(anthropic_tool_result_to_openai(b, &mut user_parts));
                    }
                    Some("image") => user_parts.extend(anthropic_image_part(b)),
                    Some("text") | None => {
                        if let Some(t) =
                            b.get("text").and_then(Value::as_str).or_else(|| b.as_str())
                        {
                            user_parts.push(text_block(t, b.get("cache_control")));
                        }
                    }
                    Some("document") => user_parts.push(anthropic_document_to_openai(b)),
                    // Anything else (`search_result`, `container_upload`, …) has no Chat
                    // Completions part: forwarded for the provider to reject by name.
                    Some(_) => user_parts.push(b.clone()),
                }
            }
            flush_user(&mut user_parts, &mut out);
            if out.is_empty() {
                out.push(json!({ "role": "user", "content": "" }));
            }
            out
        }
        _ => vec![json!({ "role": "user", "content": "" })],
    }
}

/// Anthropic `tool_result` → a Chat Completions `tool` message holding its text. A tool message
/// holds text only, so images and documents in the result (a screenshot, a rendered PDF) move to
/// `carried`, which the caller sends as the user message right after the tool messages; dropping
/// them would get an answer about output the model never saw. `is_error` has no field: the text
/// says so instead.
fn anthropic_tool_result_to_openai(b: &Value, carried: &mut Vec<Value>) -> Value {
    let id = b
        .get("tool_use_id")
        .and_then(Value::as_str)
        .unwrap_or("call_0");
    let mut text = String::new();
    match b.get("content") {
        Some(Value::String(s)) => text.push_str(s),
        Some(Value::Array(inner)) => {
            for x in inner {
                match x.get("type").and_then(Value::as_str) {
                    Some("text") | None => {
                        if let Some(t) = x.get("text").and_then(Value::as_str) {
                            text.push_str(t);
                        }
                    }
                    Some("image") => carried.extend(anthropic_image_part(x)),
                    Some("document") => carried.push(anthropic_document_to_openai(x)),
                    Some(_) => carried.push(x.clone()),
                }
            }
        }
        _ => {}
    }
    if b.get("is_error").and_then(Value::as_bool) == Some(true) {
        text = if text.is_empty() {
            "Error".to_owned()
        } else {
            format!("Error: {text}")
        };
    }
    json!({ "role": "tool", "tool_call_id": id, "content": text })
}

/// Anthropic `image` → a Chat Completions `image_url` part. A source Chat Completions cannot
/// reference (a Files API `file_id`) is forwarded as-is for the provider to reject by name. A
/// `url` source that is not http(s) is never forwarded in any shape.
fn anthropic_image_part(b: &Value) -> Option<Value> {
    let Some(mut part) = anthropic_image_to_openai(b) else {
        let url_source = b.pointer("/source/type").and_then(Value::as_str) == Some("url");
        return (!url_source).then(|| b.clone());
    };
    if let Some(obj) = part.as_object_mut() {
        copy_cache_control(obj, b);
    }
    Some(part)
}

/// Anthropic `document` → a Chat Completions part. A base64 PDF becomes a `file` part with a data
/// URI and a plain-text document becomes text. A URL or Files-API document has no Chat Completions
/// equivalent and is forwarded as-is, so the provider rejects it by name instead of the model
/// answering without it.
fn anthropic_document_to_openai(b: &Value) -> Value {
    let src = b.get("source");
    let kind = src.and_then(|s| s.get("type")).and_then(Value::as_str);
    let media = src
        .and_then(|s| s.get("media_type"))
        .and_then(Value::as_str);
    let data = src.and_then(|s| s.get("data")).and_then(Value::as_str);
    match (kind, media, data) {
        (Some("base64"), Some(media), Some(data)) => {
            let filename = b
                .get("title")
                .and_then(Value::as_str)
                .unwrap_or("document.pdf");
            json!({
                "type": "file",
                "file": {
                    "filename": filename,
                    "file_data": format!("data:{media};base64,{data}"),
                }
            })
        }
        (Some("text"), _, Some(text)) => json!({ "type": "text", "text": text }),
        _ => b.clone(),
    }
}

fn anthropic_image_to_openai(b: &Value) -> Option<Value> {
    let src = b.get("source")?;
    match src.get("type").and_then(Value::as_str) {
        Some("base64") => {}
        Some("url") => {
            let url = src
                .get("url")
                .and_then(Value::as_str)
                .filter(|u| is_http_url(u))?;
            return Some(json!({
                "type": "image_url",
                "image_url": { "url": url },
            }));
        }
        _ => return None,
    }
    let media = src.get("media_type").and_then(Value::as_str)?;
    let data = src.get("data").and_then(Value::as_str)?;
    Some(json!({
        "type": "image_url",
        "image_url": { "url": format!("data:{media};base64,{data}") },
    }))
}

// --- response JSON ----------------------------------------------------------

fn anthropic_resp_to_openai(v: &Value) -> Value {
    let id = v.get("id").cloned().unwrap_or(json!("msg_translated"));
    let model = v.get("model").cloned().unwrap_or(json!(""));
    let assistant = anthropic_assistant_to_openai(&json!({
        "role": "assistant",
        "content": v.get("content").cloned().unwrap_or(json!([])),
    }));
    let finish = map_stop_to_openai(v.get("stop_reason").and_then(Value::as_str));
    let usage = map_usage_to_openai(v.get("usage"));
    json!({
        "id": id,
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": assistant,
            "finish_reason": finish,
        }],
        "usage": usage,
    })
}

fn openai_resp_to_anthropic(v: &Value) -> Value {
    let id = v.get("id").cloned().unwrap_or(json!("chatcmpl_translated"));
    let model = v.get("model").cloned().unwrap_or(json!(""));
    let choice = v
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|a| a.first());
    let message = choice.and_then(|c| c.get("message")).unwrap_or(v);
    let content = openai_assistant_content(message);
    let finish = choice
        .and_then(|c| c.get("finish_reason"))
        .and_then(Value::as_str);
    let usage = map_usage_to_anthropic(v.get("usage"));
    json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": normalize_anth_content(content),
        "stop_reason": map_stop_to_anthropic(finish),
        "usage": usage,
    })
}

fn normalize_anth_content(content: Value) -> Value {
    match content {
        Value::String(s) => json!([{ "type": "text", "text": s }]),
        other => other,
    }
}

fn map_stop_to_openai(reason: Option<&str>) -> &'static str {
    match reason {
        Some("max_tokens") => "length",
        Some("tool_use") => "tool_calls",
        _ => "stop",
    }
}

fn map_stop_to_anthropic(reason: Option<&str>) -> &'static str {
    match reason {
        Some("length") => "max_tokens",
        Some("tool_calls") => "tool_use",
        _ => "end_turn",
    }
}

fn map_usage_to_openai(usage: Option<&Value>) -> Value {
    let u = usage.unwrap_or(&Value::Null);
    let input = u
        .get("input_tokens")
        .and_then(Value::as_u64)
        .or_else(|| u.get("prompt_tokens").and_then(Value::as_u64))
        .unwrap_or(0);
    let output = u
        .get("output_tokens")
        .and_then(Value::as_u64)
        .or_else(|| u.get("completion_tokens").and_then(Value::as_u64))
        .unwrap_or(0);
    let mut m = json!({
        "prompt_tokens": input,
        "completion_tokens": output,
        "total_tokens": input.saturating_add(output),
    });
    if let Some(obj) = m.as_object_mut() {
        let cached = u
            .get("cache_read_input_tokens")
            .and_then(Value::as_u64)
            .or_else(|| {
                u.pointer("/prompt_tokens_details/cached_tokens")
                    .and_then(Value::as_u64)
            })
            .unwrap_or(0);
        if cached > 0 {
            obj.insert(
                "prompt_tokens_details".into(),
                json!({ "cached_tokens": cached }),
            );
        }
        let reasoning = u
            .pointer("/output_tokens_details/thinking_tokens")
            .and_then(Value::as_u64)
            .or_else(|| {
                u.pointer("/completion_tokens_details/reasoning_tokens")
                    .and_then(Value::as_u64)
            });
        if let Some(r) = reasoning {
            obj.insert(
                "completion_tokens_details".into(),
                json!({ "reasoning_tokens": r }),
            );
        }
    }
    m
}

fn map_usage_to_anthropic(usage: Option<&Value>) -> Value {
    let u = usage.unwrap_or(&Value::Null);
    let input = u
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .or_else(|| u.get("input_tokens").and_then(Value::as_u64))
        .unwrap_or(0);
    let output = u
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .or_else(|| u.get("output_tokens").and_then(Value::as_u64))
        .unwrap_or(0);
    let mut m = json!({ "input_tokens": input, "output_tokens": output });
    if let Some(obj) = m.as_object_mut() {
        if let Some(c) = u
            .pointer("/prompt_tokens_details/cached_tokens")
            .and_then(Value::as_u64)
            .or_else(|| u.get("cache_read_input_tokens").and_then(Value::as_u64))
            && c > 0
        {
            obj.insert("cache_read_input_tokens".into(), json!(c));
        }
        if let Some(w) = u.get("cache_creation_input_tokens").and_then(Value::as_u64)
            && w > 0
        {
            obj.insert("cache_creation_input_tokens".into(), json!(w));
        }
        if let Some(r) = u
            .pointer("/completion_tokens_details/reasoning_tokens")
            .and_then(Value::as_u64)
            .or_else(|| {
                u.pointer("/output_tokens_details/thinking_tokens")
                    .and_then(Value::as_u64)
            })
        {
            obj.insert(
                "output_tokens_details".into(),
                json!({ "thinking_tokens": r }),
            );
        }
    }
    m
}

// --- request: Responses ↔ Chat Completions ----------------------------------

fn responses_req_to_openai(v: &Value, openai: OpenAiModel) -> Value {
    let mut out = Map::new();
    copy_if(&mut out, v, "model");
    copy_if(&mut out, v, "stream");
    if let Some(t) = v
        .get("max_output_tokens")
        .or_else(|| v.get("max_tokens"))
        .or_else(|| v.get("max_completion_tokens"))
    {
        out.insert(openai.max_tokens_key().into(), t.clone());
    }
    let effort = v
        .get("reasoning_effort")
        .and_then(Value::as_str)
        .or_else(|| v.pointer("/reasoning/effort").and_then(Value::as_str))
        .and_then(|e| openai.effort(e));
    if openai.keeps_sampling(effort.as_deref()) {
        copy_if(&mut out, v, "temperature");
        copy_if(&mut out, v, "top_p");
    }
    if let Some(effort) = effort {
        out.insert("reasoning_effort".into(), json!(effort));
    }
    if let Some(tools) = v.get("tools").and_then(Value::as_array) {
        let mapped: Vec<Value> = tools.iter().map(responses_tool_to_openai).collect();
        if !mapped.is_empty() {
            out.insert("tools".into(), Value::Array(mapped));
        }
    }
    if let Some(choice) = v.get("tool_choice") {
        out.insert(
            "tool_choice".into(),
            responses_tool_choice_to_openai(choice),
        );
    }
    for key in ["parallel_tool_calls", "user", "safety_identifier"] {
        copy_if(&mut out, v, key);
    }
    // OpenAI's Chat Completions takes these too. Elsewhere they are hints (cache routing, latency
    // tier, answer length) and go.
    if openai.native {
        for key in ["prompt_cache_key", "service_tier"] {
            copy_if(&mut out, v, key);
        }
        if let Some(verbosity) = v.pointer("/text/verbosity") {
            out.insert("verbosity".into(), verbosity.clone());
        }
    }
    // A stored prompt template decides what the model is asked. Chat Completions has no
    // equivalent: forwarded for the provider to reject by name.
    copy_if(&mut out, v, "prompt");
    // Responses nests the output format under `text.format` and flattens `json_schema`.
    if let Some(format) = v.pointer("/text/format") {
        let rf = match format.get("type").and_then(Value::as_str) {
            Some("json_schema") => {
                let mut js = Map::new();
                for key in ["name", "schema", "strict", "description"] {
                    copy_if(&mut js, format, key);
                }
                Some(json!({ "type": "json_schema", "json_schema": js }))
            }
            Some("json_object") => Some(json!({ "type": "json_object" })),
            _ => None,
        };
        if let Some(rf) = rf {
            out.insert("response_format".into(), rf);
        }
    }

    let mut messages: Vec<Value> = Vec::new();
    if let Some(instr) = v.get("instructions") {
        messages.push(responses_instructions_to_system(instr));
    }
    messages.extend(responses_input_to_messages(v.get("input")));
    out.insert("messages".into(), Value::Array(messages));
    Value::Object(out)
}

/// The smallest `max_output_tokens` the Responses API accepts.
const RESPONSES_MIN_OUTPUT_TOKENS: u64 = 16;

fn openai_req_to_responses(v: &Value, openai: OpenAiModel) -> Value {
    let mut out = Map::new();
    copy_if(&mut out, v, "model");
    copy_if(&mut out, v, "stream");
    // The Responses API rejects a limit under 16 ("integer_below_min_value"); Chat Completions
    // and Messages accept 1. Raising a tiny limit to the floor answers the request instead of
    // forwarding a guaranteed 400.
    if let Some(t) = max_tokens_of(v).or_else(|| v.get("max_output_tokens").and_then(Value::as_u64))
    {
        out.insert(
            "max_output_tokens".into(),
            json!(t.max(RESPONSES_MIN_OUTPUT_TOKENS)),
        );
    }
    // Chat Completions stores nothing unless asked; Responses stores every response by default.
    out.insert(
        "store".into(),
        v.get("store")
            .filter(|s| s.is_boolean())
            .cloned()
            .unwrap_or(json!(false)),
    );
    let effort = v
        .get("reasoning_effort")
        .and_then(Value::as_str)
        .or_else(|| v.pointer("/reasoning/effort").and_then(Value::as_str))
        .and_then(|e| openai.effort(e));
    if openai.keeps_sampling(effort.as_deref()) {
        copy_if(&mut out, v, "temperature");
        copy_if(&mut out, v, "top_p");
    }
    if let Some(effort) = effort {
        out.insert(
            "reasoning".into(),
            json!({ "effort": effort, "summary": "auto" }),
        );
    }
    if let Some(tools) = v.get("tools").and_then(Value::as_array) {
        let mapped: Vec<Value> = tools.iter().filter_map(openai_tool_to_responses).collect();
        if !mapped.is_empty() {
            out.insert("tools".into(), Value::Array(mapped));
        }
    }
    if let Some(choice) = v.get("tool_choice") {
        out.insert(
            "tool_choice".into(),
            openai_tool_choice_to_responses(choice),
        );
    }
    for key in [
        "parallel_tool_calls",
        "user",
        "safety_identifier",
        "prompt_cache_key",
        "service_tier",
        "metadata",
        "top_logprobs",
    ] {
        copy_if(&mut out, v, key);
    }
    // No Responses field, and each changes what the client gets back: forwarded for the provider to
    // reject by name. (`mcp_servers` arrives here from a Messages client.)
    for key in ["stop", "audio", "web_search_options", "mcp_servers"] {
        copy_if(&mut out, v, key);
    }
    if v.get("n").and_then(Value::as_u64).is_some_and(|n| n > 1) {
        copy_if(&mut out, v, "n");
    }
    if v.get("logprobs").and_then(Value::as_bool) == Some(true) {
        copy_if(&mut out, v, "logprobs");
    }
    if v.get("modalities")
        .and_then(Value::as_array)
        .is_some_and(|m| m.iter().any(|x| x != "text"))
    {
        copy_if(&mut out, v, "modalities");
    }
    let mut text = Map::new();
    if let Some(rf) = v.get("response_format") {
        let format = match rf.get("type").and_then(Value::as_str) {
            Some("json_schema") => {
                let mut f = Map::new();
                f.insert("type".into(), json!("json_schema"));
                if let Some(js) = rf.get("json_schema") {
                    for key in ["name", "schema", "strict", "description"] {
                        copy_if(&mut f, js, key);
                    }
                }
                Some(Value::Object(f))
            }
            Some("json_object") => Some(json!({ "type": "json_object" })),
            _ => None,
        };
        if let Some(format) = format {
            text.insert("format".into(), format);
        }
    }
    copy_if(&mut text, v, "verbosity");
    if !text.is_empty() {
        out.insert("text".into(), Value::Object(text));
    }

    let mut instructions: Vec<String> = Vec::new();
    let mut input: Vec<Value> = Vec::new();
    if let Some(arr) = v.get("messages").and_then(Value::as_array) {
        let mut in_conversation = false;
        for m in arr {
            let role = m.get("role").and_then(Value::as_str).unwrap_or("");
            match role {
                "system" | "developer" if !in_conversation => {
                    instructions.extend(
                        openai_system_to_responses_blocks(m)
                            .iter()
                            .filter_map(|b| b.get("text").and_then(Value::as_str))
                            .map(str::to_owned),
                    );
                }
                // Responses takes system and developer messages in `input`; a mid-conversation one
                // stays where the client put it.
                "system" | "developer" => input.push(json!({
                    "type": "message",
                    "role": role,
                    "content": chat_content_to_responses(m.get("content"), false),
                })),
                "tool" => {
                    in_conversation = true;
                    input.push(chat_tool_to_function_call_output(m));
                }
                "assistant" => {
                    in_conversation = true;
                    // The turn's text comes before the calls it made, as Responses emits them.
                    let content = chat_content_to_responses(m.get("content"), true);
                    let mut content = content.as_array().cloned().unwrap_or_default();
                    if let Some(r) = m
                        .get("refusal")
                        .and_then(Value::as_str)
                        .filter(|r| !r.is_empty())
                        && content.is_empty()
                    {
                        content.push(json!({ "type": "refusal", "refusal": r }));
                    }
                    if !content.is_empty() {
                        input.push(json!({
                            "type": "message",
                            "role": "assistant",
                            "content": content,
                        }));
                    }
                    if let Some(calls) = m.get("tool_calls").and_then(Value::as_array) {
                        input.extend(calls.iter().map(chat_tool_call_to_responses_item));
                    }
                }
                _ => {
                    in_conversation = true;
                    input.push(json!({
                        "type": "message",
                        "role": "user",
                        "content": chat_content_to_responses(m.get("content"), false),
                    }));
                }
            }
        }
    }
    if !instructions.is_empty() {
        // `instructions` is a string; a list of text blocks is not one of its shapes.
        out.insert("instructions".into(), json!(instructions.join("\n\n")));
    }
    out.insert("input".into(), Value::Array(input));
    Value::Object(out)
}

/// A Chat Completions message's content as Responses input content. `cache_control` is Anthropic's
/// and goes; `thinking` / `redacted_thinking` blocks carry another vendor's reasoning, which a
/// Responses model cannot read (it is the earlier turn's own scratch work, not input); parts with
/// no Responses equivalent (`input_audio`) are forwarded for the provider to reject by name.
fn chat_content_to_responses(content: Option<&Value>, assistant: bool) -> Value {
    let text_type = if assistant {
        "output_text"
    } else {
        "input_text"
    };
    let text = |t: &Value| json!({ "type": text_type, "text": t });
    match content {
        Some(Value::String(s)) => json!([text(&json!(s))]),
        Some(Value::Array(parts)) => Value::Array(
            parts
                .iter()
                .filter_map(|p| match p.get("type").and_then(Value::as_str) {
                    Some("text") | None => p
                        .get("text")
                        .cloned()
                        .or_else(|| p.as_str().map(|s| json!(s)))
                        .map(|t| text(&t)),
                    Some("refusal") if assistant => Some(json!({
                        "type": "refusal",
                        "refusal": p.get("refusal").cloned().unwrap_or(json!("")),
                    })),
                    Some("refusal") => p.get("refusal").map(text),
                    Some("image_url") => Some(chat_image_to_responses(p)),
                    Some("file") => {
                        let mut m = Map::new();
                        m.insert("type".into(), json!("input_file"));
                        if let Some(file) = p.get("file") {
                            for key in ["file_data", "file_id", "filename"] {
                                copy_if(&mut m, file, key);
                            }
                        }
                        Some(Value::Object(m))
                    }
                    Some("thinking" | "redacted_thinking") => None,
                    Some(_) => Some(p.clone()),
                })
                .collect(),
        ),
        _ => json!([]),
    }
}

fn chat_image_to_responses(p: &Value) -> Value {
    let Some(url) = p.pointer("/image_url/url").and_then(Value::as_str) else {
        return p.clone();
    };
    let mut m = json!({ "type": "input_image", "image_url": url });
    if let Some(detail) = p.pointer("/image_url/detail")
        && let Some(obj) = m.as_object_mut()
    {
        obj.insert("detail".into(), detail.clone());
    }
    m
}

/// A Chat Completions `tool` message → a Responses `function_call_output`, whose `output` is a
/// string or a list of input parts (text, images, files).
fn chat_tool_to_function_call_output(m: &Value) -> Value {
    let output = match m.get("content") {
        Some(Value::String(s)) => json!(s),
        Some(content @ Value::Array(_)) => chat_content_to_responses(Some(content), false),
        _ => json!(""),
    };
    json!({
        "type": "function_call_output",
        "call_id": m.get("tool_call_id").cloned().unwrap_or(json!("call_0")),
        "output": output,
    })
}

/// A Chat Completions tool call → a Responses `function_call` (or `custom_tool_call`) item.
fn chat_tool_call_to_responses_item(c: &Value) -> Value {
    let call_id = c.get("id").cloned().unwrap_or(json!("call_0"));
    if c.get("type").and_then(Value::as_str) == Some("custom") {
        let custom = c.get("custom").unwrap_or(c);
        return json!({
            "type": "custom_tool_call",
            "call_id": call_id,
            "name": custom.get("name").cloned().unwrap_or(json!("")),
            "input": custom.get("input").cloned().unwrap_or(json!("")),
        });
    }
    let func = c.get("function").unwrap_or(c);
    json!({
        "type": "function_call",
        "call_id": call_id,
        "name": func.get("name").cloned().unwrap_or(json!("")),
        "arguments": func.get("arguments").cloned().unwrap_or(json!("{}")),
    })
}

/// A Responses tool → Chat Completions. `custom` (free-form input) has a Chat Completions shape;
/// hosted tools (`web_search`, `file_search`, `mcp`, …) do not, and are forwarded as-is for the
/// provider to reject by name. Dropping one would let the model answer without a tool the client
/// offered.
fn responses_tool_to_openai(t: &Value) -> Value {
    if t.get("function").is_some() || t.get("custom").is_some() {
        return t.clone();
    }
    let typ = t.get("type").and_then(Value::as_str).unwrap_or("function");
    let Some(name) = t.get("name").and_then(Value::as_str) else {
        return t.clone();
    };
    let keys: &[&str] = match typ {
        "function" => &["description", "parameters", "strict"],
        "custom" => &["description", "format"],
        _ => return t.clone(),
    };
    let mut inner = Map::new();
    inner.insert("name".into(), json!(name));
    for key in keys {
        copy_if(&mut inner, t, key);
    }
    let mut out = Map::new();
    out.insert("type".into(), json!(typ));
    out.insert(typ.into(), Value::Object(inner));
    copy_cache_control(&mut out, t);
    Value::Object(out)
}

/// A Chat Completions tool → Responses (flat). `cache_control` is Anthropic's and goes.
fn openai_tool_to_responses(t: &Value) -> Option<Value> {
    let typ = t.get("type").and_then(Value::as_str).unwrap_or("function");
    let (inner, keys): (&Value, &[&str]) = match typ {
        "function" => (
            t.get("function").unwrap_or(t),
            &["description", "parameters", "strict"],
        ),
        "custom" => (t.get("custom").unwrap_or(t), &["description", "format"]),
        // Hosted and Anthropic server tools: forwarded for the provider to reject by name.
        _ => return Some(t.clone()),
    };
    let name = inner.get("name").and_then(Value::as_str)?;
    let mut m = Map::new();
    m.insert("type".into(), json!(typ));
    m.insert("name".into(), json!(name));
    for key in keys {
        copy_if(&mut m, inner, key);
    }
    Some(Value::Object(m))
}

/// A tool reference inside `tool_choice` / `allowed_tools`, Responses (flat) → Chat (nested).
fn responses_tool_ref_to_openai(r: &Value) -> Value {
    match (
        r.get("type").and_then(Value::as_str),
        r.get("name").and_then(Value::as_str),
    ) {
        (Some(typ @ ("function" | "custom")), Some(name)) => {
            json!({ "type": typ, typ: { "name": name } })
        }
        _ => r.clone(),
    }
}

/// A tool reference, Chat (nested) → Responses (flat).
fn openai_tool_ref_to_responses(r: &Value) -> Value {
    match r.get("type").and_then(Value::as_str) {
        Some(typ @ ("function" | "custom")) => {
            match r
                .get(typ)
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
            {
                Some(name) => json!({ "type": typ, "name": name }),
                None => r.clone(),
            }
        }
        _ => r.clone(),
    }
}

/// Responses `tool_choice` → Chat Completions. Strings are the same; a named tool nests its name,
/// and `allowed_tools` nests its mode and list. A hosted-tool choice (`{"type": "web_search"}`)
/// is forwarded for the provider to reject.
fn responses_tool_choice_to_openai(choice: &Value) -> Value {
    match choice.get("type").and_then(Value::as_str) {
        Some("allowed_tools") if choice.get("allowed_tools").is_none() => {
            let tools: Vec<Value> = choice
                .get("tools")
                .and_then(Value::as_array)
                .map(|ts| ts.iter().map(responses_tool_ref_to_openai).collect())
                .unwrap_or_default();
            json!({
                "type": "allowed_tools",
                "allowed_tools": {
                    "mode": choice.get("mode").cloned().unwrap_or(json!("auto")),
                    "tools": tools,
                },
            })
        }
        Some("function" | "custom") => responses_tool_ref_to_openai(choice),
        _ => choice.clone(),
    }
}

/// Chat Completions `tool_choice` → Responses: the inverse of [`responses_tool_choice_to_openai`].
fn openai_tool_choice_to_responses(choice: &Value) -> Value {
    match choice.get("type").and_then(Value::as_str) {
        Some("allowed_tools") => {
            let allowed = choice.get("allowed_tools").unwrap_or(choice);
            let tools: Vec<Value> = allowed
                .get("tools")
                .and_then(Value::as_array)
                .map(|ts| ts.iter().map(openai_tool_ref_to_responses).collect())
                .unwrap_or_default();
            json!({
                "type": "allowed_tools",
                "mode": allowed.get("mode").cloned().unwrap_or(json!("auto")),
                "tools": tools,
            })
        }
        Some("function" | "custom") => openai_tool_ref_to_responses(choice),
        _ => choice.clone(),
    }
}

fn responses_instructions_to_system(instr: &Value) -> Value {
    match instr {
        Value::String(s) => json!({ "role": "system", "content": s }),
        Value::Array(parts) => json!({
            "role": "system",
            "content": parts.iter().filter_map(responses_part_to_openai).collect::<Vec<_>>(),
        }),
        other => json!({ "role": "system", "content": other }),
    }
}

fn responses_input_to_messages(input: Option<&Value>) -> Vec<Value> {
    let items = match input {
        Some(Value::String(s)) => return vec![json!({ "role": "user", "content": s })],
        Some(Value::Array(items)) => items,
        _ => return Vec::new(),
    };
    let mut out: Vec<Value> = Vec::new();
    // Images and files a tool returned. A Chat Completions tool message holds text, so they follow
    // the run of tool messages as one user message.
    let mut carried: Vec<Value> = Vec::new();
    for item in items {
        let typ = item.get("type").and_then(Value::as_str).unwrap_or("");
        match typ {
            "function_call" | "custom_tool_call" => {
                flush_carried(&mut out, &mut carried);
                let call = responses_call_to_openai(item);
                // Parallel calls arrive as consecutive items; Chat Completions wants them on one
                // assistant message (split across several, OpenAI rejects the history), after
                // whatever text that turn said.
                match out.last_mut().and_then(Value::as_object_mut) {
                    Some(last) if last.get("role").and_then(Value::as_str) == Some("assistant") => {
                        match last.get_mut("tool_calls").and_then(Value::as_array_mut) {
                            Some(calls) => calls.push(call),
                            None => {
                                last.insert("tool_calls".into(), json!([call]));
                            }
                        }
                    }
                    _ => out.push(json!({
                        "role": "assistant",
                        "content": Value::Null,
                        "tool_calls": [call],
                    })),
                }
            }
            "function_call_output" | "custom_tool_call_output" => {
                out.push(responses_output_to_tool_message(item, &mut carried));
            }
            // The model's own earlier reasoning, and hosted-tool call records: no Chat Completions
            // equivalent, and nothing the client wrote.
            "reasoning" => {}
            _ if typ == "message" || item.get("role").is_some() => {
                flush_carried(&mut out, &mut carried);
                let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
                out.push(json!({
                    "role": role,
                    "content": responses_content_to_openai(item.get("content")),
                }));
            }
            _ => {}
        }
    }
    flush_carried(&mut out, &mut carried);
    out
}

fn flush_carried(out: &mut Vec<Value>, carried: &mut Vec<Value>) {
    if !carried.is_empty() {
        out.push(json!({ "role": "user", "content": std::mem::take(carried) }));
    }
}

/// A Responses `function_call` / `custom_tool_call` item → a Chat Completions tool call.
fn responses_call_to_openai(item: &Value) -> Value {
    let id = item
        .get("call_id")
        .or_else(|| item.get("id"))
        .cloned()
        .unwrap_or(json!("call_0"));
    let name = item.get("name").cloned().unwrap_or(json!(""));
    if item.get("type").and_then(Value::as_str) == Some("custom_tool_call") {
        return json!({
            "id": id,
            "type": "custom",
            "custom": { "name": name, "input": item.get("input").cloned().unwrap_or(json!("")) },
        });
    }
    json!({
        "id": id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": match item.get("arguments") {
                Some(Value::String(s)) => Value::String(s.clone()),
                Some(other) => json!(value_string(other)),
                None => json!("{}"),
            },
        }
    })
}

/// A Responses tool output → a Chat Completions `tool` message. An output that is a list of parts
/// keeps its text here; its images and files go to `carried` (see `responses_input_to_messages`).
fn responses_output_to_tool_message(item: &Value, carried: &mut Vec<Value>) -> Value {
    let mut m = Map::new();
    m.insert("role".into(), json!("tool"));
    if let Some(id) = item.get("call_id").or_else(|| item.get("id")) {
        m.insert("tool_call_id".into(), id.clone());
    }
    let content = match item.get("output") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => {
            let mut text = String::new();
            for p in parts {
                match p.get("type").and_then(Value::as_str) {
                    Some("input_text" | "output_text" | "text") | None => {
                        if let Some(t) = p.get("text").and_then(Value::as_str) {
                            text.push_str(t);
                        }
                    }
                    _ => carried.extend(responses_part_to_openai(p)),
                }
            }
            text
        }
        Some(other) => value_string(other),
        None => String::new(),
    };
    m.insert("content".into(), json!(content));
    Value::Object(m)
}

fn responses_content_to_openai(content: Option<&Value>) -> Value {
    match content {
        Some(Value::String(s)) => Value::String(s.clone()),
        Some(Value::Array(parts)) => {
            let mapped: Vec<Value> = parts.iter().filter_map(responses_part_to_openai).collect();
            if mapped.len() == 1
                && mapped[0].get("type").and_then(Value::as_str) == Some("text")
                && mapped[0].get("cache_control").is_none()
            {
                mapped[0]
                    .get("text")
                    .cloned()
                    .unwrap_or(Value::Array(mapped))
            } else {
                Value::Array(mapped)
            }
        }
        Some(other) => other.clone(),
        None => Value::String(String::new()),
    }
}

fn responses_part_to_openai(part: &Value) -> Option<Value> {
    let typ = part.get("type").and_then(Value::as_str).unwrap_or("text");
    match typ {
        "input_text" | "output_text" | "text" => {
            let mut m = Map::new();
            m.insert("type".into(), json!("text"));
            m.insert(
                "text".into(),
                part.get("text").cloned().unwrap_or(json!("")),
            );
            copy_cache_control(&mut m, part);
            Some(Value::Object(m))
        }
        "input_image" | "image_url" => {
            let url = part
                .pointer("/image_url/url")
                .or_else(|| part.get("image_url"))
                .or_else(|| part.get("url"));
            let url = match url {
                Some(Value::String(s)) => s.as_str(),
                // A Files API image (`file_id`): Chat Completions images are URLs only. Forwarded
                // for the provider to reject by name, not answered without the picture.
                _ => return Some(part.clone()),
            };
            let mut image_url = json!({ "url": url });
            if let (Some(d), Some(obj)) = (part.get("detail"), image_url.as_object_mut()) {
                obj.insert("detail".into(), d.clone());
            }
            let mut m = json!({ "type": "image_url", "image_url": image_url });
            if let Some(obj) = m.as_object_mut() {
                copy_cache_control(obj, part);
            }
            Some(m)
        }
        // A `file_url` document has no Chat Completions `file` field: forwarded as-is.
        "input_file" if part.get("file_url").is_some() => Some(part.clone()),
        "input_file" => {
            let mut file = Map::new();
            for key in ["file_data", "file_id", "filename"] {
                copy_if(&mut file, part, key);
            }
            Some(json!({ "type": "file", "file": file }))
        }
        "thinking" | "redacted_thinking" => Some(part.clone()),
        _ => None,
    }
}

fn openai_message_to_responses_content(m: &Value, input: bool) -> Value {
    let text_type = if input { "input_text" } else { "output_text" };
    match m.get("content") {
        Some(Value::String(s)) => json!([{ "type": text_type, "text": s }]),
        Some(Value::Array(parts)) => Value::Array(
            parts
                .iter()
                .filter_map(|p| openai_part_to_responses(p, text_type))
                .collect(),
        ),
        Some(Value::Null) | None => json!([]),
        Some(other) => other.clone(),
    }
}

fn openai_part_to_responses(part: &Value, text_type: &str) -> Option<Value> {
    match part.get("type").and_then(Value::as_str) {
        Some("text") | None if part.get("text").is_some() || part.is_string() => {
            let mut m = Map::new();
            m.insert("type".into(), json!(text_type));
            m.insert(
                "text".into(),
                part.get("text")
                    .cloned()
                    .or_else(|| part.as_str().map(|s| json!(s)))
                    .unwrap_or(json!("")),
            );
            copy_cache_control(&mut m, part);
            Some(Value::Object(m))
        }
        Some("image_url") => {
            let url = part.pointer("/image_url/url").and_then(Value::as_str)?;
            let mut m = json!({
                "type": "input_image",
                "image_url": url,
            });
            if let Some(obj) = m.as_object_mut() {
                copy_cache_control(obj, part);
            }
            Some(m)
        }
        Some("file") => {
            let mut m = Map::new();
            m.insert("type".into(), json!("input_file"));
            if let Some(file) = part.get("file") {
                for key in ["file_data", "file_id", "filename"] {
                    copy_if(&mut m, file, key);
                }
            }
            Some(Value::Object(m))
        }
        Some("thinking") | Some("redacted_thinking") => Some(part.clone()),
        _ => part
            .as_str()
            .map(|s| json!({ "type": text_type, "text": s })),
    }
}

fn content_is_empty(content: &Value) -> bool {
    match content {
        Value::Array(a) => a.is_empty(),
        Value::String(s) => s.is_empty(),
        Value::Null => true,
        _ => false,
    }
}

fn openai_tool_call_to_function_call(c: &Value) -> Value {
    let func = c.get("function").unwrap_or(c);
    json!({
        "type": "function_call",
        "call_id": c.get("id").cloned().unwrap_or(json!("call_0")),
        "name": func.get("name").cloned().unwrap_or(json!("")),
        "arguments": func.get("arguments").cloned().unwrap_or(json!("{}")),
    })
}

fn openai_system_to_responses_blocks(m: &Value) -> Vec<Value> {
    match m.get("content") {
        Some(Value::String(s)) => {
            let mut b = json!({ "type": "text", "text": s });
            if let Some(obj) = b.as_object_mut() {
                copy_cache_control(obj, m);
            }
            vec![b]
        }
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| {
                let text = p
                    .get("text")
                    .and_then(Value::as_str)
                    .or_else(|| p.as_str())?;
                let mut b = json!({ "type": "text", "text": text });
                if let Some(obj) = b.as_object_mut() {
                    copy_cache_control(obj, p);
                    if !obj.contains_key("cache_control") {
                        copy_cache_control(obj, m);
                    }
                }
                Some(b)
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn openai_resp_to_responses(v: &Value) -> Value {
    let id = v.get("id").cloned().unwrap_or(json!("resp_translated"));
    let model = v.get("model").cloned().unwrap_or(json!(""));
    let choice = v
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|a| a.first());
    let message = choice.and_then(|c| c.get("message")).unwrap_or(v);
    let mut output = Vec::new();
    if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
        for c in calls {
            output.push(openai_tool_call_to_function_call(c));
        }
    }
    let content = openai_message_to_responses_content(message, false);
    if !content_is_empty(&content) {
        output.insert(
            0,
            json!({
                "type": "message",
                "id": "msg_translated",
                "role": "assistant",
                "content": content,
                "status": "completed",
            }),
        );
    }
    json!({
        "id": id,
        "object": "response",
        "status": "completed",
        "model": model,
        "output": output,
        "usage": map_usage_to_responses(v.get("usage")),
    })
}

fn responses_resp_to_openai(v: &Value) -> Value {
    let id = v.get("id").cloned().unwrap_or(json!("resp_translated"));
    let model = v.get("model").cloned().unwrap_or(json!(""));
    let output = v.get("output").and_then(Value::as_array);
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    if let Some(items) = output {
        for item in items {
            match item.get("type").and_then(Value::as_str) {
                Some("function_call") => tool_calls.push(json!({
                    "id": item.get("call_id").or_else(|| item.get("id")).cloned().unwrap_or(json!("call_0")),
                    "type": "function",
                    "function": {
                        "name": item.get("name").cloned().unwrap_or(json!("")),
                        "arguments": item.get("arguments").cloned().unwrap_or(json!("{}")),
                    }
                })),
                _ => {
                    if let Some(content) = item.get("content").and_then(Value::as_array) {
                        for p in content {
                            if let Some(t) = p.get("text").and_then(Value::as_str) {
                                text.push_str(t);
                            }
                        }
                    } else if let Some(t) = item.get("content").and_then(Value::as_str) {
                        text.push_str(t);
                    }
                }
            }
        }
    }
    let mut message = json!({ "role": "assistant", "content": text });
    let finish = if tool_calls.is_empty() {
        "stop"
    } else {
        if let Some(obj) = message.as_object_mut() {
            obj.insert("tool_calls".into(), Value::Array(tool_calls));
            obj.insert("content".into(), Value::Null);
        }
        "tool_calls"
    };
    json!({
        "id": id,
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish,
        }],
        "usage": map_usage_to_openai(v.get("usage")),
    })
}

fn map_usage_to_responses(usage: Option<&Value>) -> Value {
    let u = usage.unwrap_or(&Value::Null);
    let input = u
        .get("input_tokens")
        .and_then(Value::as_u64)
        .or_else(|| u.get("prompt_tokens").and_then(Value::as_u64))
        .unwrap_or(0);
    let output = u
        .get("output_tokens")
        .and_then(Value::as_u64)
        .or_else(|| u.get("completion_tokens").and_then(Value::as_u64))
        .unwrap_or(0);
    let mut m = json!({
        "input_tokens": input,
        "output_tokens": output,
        "total_tokens": input.saturating_add(output),
    });
    if let Some(obj) = m.as_object_mut() {
        let cached = u
            .get("cache_read_input_tokens")
            .and_then(Value::as_u64)
            .or_else(|| {
                u.pointer("/prompt_tokens_details/cached_tokens")
                    .and_then(Value::as_u64)
            })
            .or_else(|| {
                u.pointer("/input_tokens_details/cached_tokens")
                    .and_then(Value::as_u64)
            })
            .unwrap_or(0);
        if cached > 0 {
            obj.insert(
                "input_tokens_details".into(),
                json!({ "cached_tokens": cached }),
            );
        }
        if let Some(r) = u
            .pointer("/completion_tokens_details/reasoning_tokens")
            .and_then(Value::as_u64)
            .or_else(|| {
                u.pointer("/output_tokens_details/reasoning_tokens")
                    .and_then(Value::as_u64)
            })
            .or_else(|| {
                u.pointer("/output_tokens_details/thinking_tokens")
                    .and_then(Value::as_u64)
            })
        {
            obj.insert(
                "output_tokens_details".into(),
                json!({ "reasoning_tokens": r }),
            );
        }
    }
    m
}

// --- SSE --------------------------------------------------------------------

/// Event-by-event SSE translator. Incomplete events stay in `buf`; complete events are mapped
/// immediately. Does not wait for `[DONE]` before forwarding deltas.
pub struct SseBridge {
    client: Endpoint,
    upstream: Endpoint,
    buf: Vec<u8>,
    /// Bytes of `buf` already searched for an event end, so a long unterminated event is scanned
    /// once rather than from the start on every chunk.
    scanned: usize,
    ant_to_oai: AntToOai,
    oai_to_ant: OaiToAnt,
    oai_to_resp: OaiToResp,
    resp_to_oai: RespToOai,
    /// An `error` event already went to the client. Flush must not invent a success close
    /// (`message_start`/`message_stop` or a trailing `[DONE]`-only envelope that implies a message).
    errored: bool,
}

#[derive(Default)]
struct AntToOai {
    id: String,
    model: String,
    next_tool: u32,
    input_tokens: u64,
    cache_read: u64,
    thinking_tokens: Option<u64>,
    done: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum OpenBlock {
    #[default]
    None,
    Text,
    Tool,
    Thinking,
}

#[derive(Default)]
struct OaiToAnt {
    started: bool,
    id: String,
    model: String,
    next_block: u32,
    open: OpenBlock,
    pending_stop: Option<&'static str>,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    finished: bool,
}

#[derive(Default)]
struct OaiToResp {
    id: String,
    model: String,
    created: bool,
    completed: bool,
    usage: Option<Value>,
}

#[derive(Default)]
struct RespToOai {
    done: bool,
}

impl SseBridge {
    pub fn new(client: Endpoint, upstream: Endpoint) -> Self {
        Self {
            client,
            upstream,
            buf: Vec::new(),
            scanned: 0,
            ant_to_oai: AntToOai::default(),
            oai_to_ant: OaiToAnt::default(),
            oai_to_resp: OaiToResp::default(),
            resp_to_oai: RespToOai::default(),
            errored: false,
        }
    }

    /// Bytes held for an event whose terminating blank line has not arrived yet.
    pub fn pending_len(&self) -> usize {
        self.buf.len()
    }

    /// Feed upstream SSE bytes. Returns client-dialect SSE bytes (possibly empty).
    pub fn feed(&mut self, data: &[u8], end: bool) -> Vec<u8> {
        self.buf.extend_from_slice(data);
        let mut out = Vec::new();
        while let Some(raw) = take_event(&mut self.buf, &mut self.scanned) {
            out.extend(self.map_event(&raw));
        }
        if end {
            if !self.buf.is_empty() {
                self.scanned = 0;
                let rest = std::mem::take(&mut self.buf);
                out.extend(self.map_event(&rest));
            }
            out.extend(self.flush());
        }
        out
    }

    fn map_event(&mut self, raw: &[u8]) -> Vec<u8> {
        let (event, data) = parse_sse(raw);
        if data.is_empty() && event.is_empty() {
            return Vec::new();
        }
        match (self.upstream, self.client) {
            (Endpoint::Messages, Endpoint::ChatCompletions) => self.ant_event_to_oai(&event, &data),
            (Endpoint::ChatCompletions, Endpoint::Messages) => self.oai_event_to_ant(&event, &data),
            (Endpoint::ChatCompletions, Endpoint::Responses) => {
                self.oai_event_to_resp(&event, &data)
            }
            (Endpoint::Responses, Endpoint::ChatCompletions) => {
                self.resp_event_to_oai(&event, &data)
            }
            (Endpoint::Messages, Endpoint::Responses) => {
                let chat = self.ant_event_to_oai(&event, &data);
                self.rewrite_chat_sse(&chat, |s, e, d| s.oai_event_to_resp(e, d))
            }
            (Endpoint::Responses, Endpoint::Messages) => {
                let chat = self.resp_event_to_oai(&event, &data);
                self.rewrite_chat_sse(&chat, |s, e, d| s.oai_event_to_ant(e, d))
            }
            (a, b) if a == b => raw.to_vec(),
            _ => Vec::new(),
        }
    }

    fn rewrite_chat_sse(
        &mut self,
        bytes: &[u8],
        mut map: impl FnMut(&mut Self, &str, &str) -> Vec<u8>,
    ) -> Vec<u8> {
        let mut buf = bytes.to_vec();
        let mut scanned = 0;
        let mut out = Vec::new();
        while let Some(raw) = take_event(&mut buf, &mut scanned) {
            let (event, data) = parse_sse(&raw);
            if data.is_empty() && event.is_empty() {
                continue;
            }
            out.extend(map(self, &event, &data));
        }
        out
    }

    fn flush(&mut self) -> Vec<u8> {
        if self.errored {
            return match self.client {
                Endpoint::ChatCompletions => {
                    if self.ant_to_oai.done || self.resp_to_oai.done {
                        Vec::new()
                    } else {
                        self.ant_to_oai.done = true;
                        self.resp_to_oai.done = true;
                        sse_data("[DONE]")
                    }
                }
                Endpoint::Messages | Endpoint::Responses | Endpoint::Embeddings => Vec::new(),
            };
        }
        match (self.upstream, self.client) {
            (_, Endpoint::ChatCompletions) => {
                if self.ant_to_oai.done || self.resp_to_oai.done {
                    Vec::new()
                } else {
                    self.ant_to_oai.done = true;
                    self.resp_to_oai.done = true;
                    sse_data("[DONE]")
                }
            }
            (_, Endpoint::Messages) => self.oai_to_ant.finish(),
            (Endpoint::Messages, Endpoint::Responses) => {
                let chat = if self.ant_to_oai.done {
                    Vec::new()
                } else {
                    self.ant_to_oai.done = true;
                    sse_data("[DONE]")
                };
                let mut out = self.rewrite_chat_sse(&chat, |s, e, d| s.oai_event_to_resp(e, d));
                out.extend(self.oai_to_resp.finish());
                out
            }
            (_, Endpoint::Responses) => self.oai_to_resp.finish(),
            // Embeddings never stream and never translate.
            (_, Endpoint::Embeddings) => Vec::new(),
        }
    }

    fn ant_event_to_oai(&mut self, event: &str, data: &str) -> Vec<u8> {
        if event == "ping" {
            return Vec::new();
        }
        if data == "[DONE]" {
            if self.errored || self.ant_to_oai.done {
                return Vec::new();
            }
            self.ant_to_oai.done = true;
            return sse_data("[DONE]");
        }
        let Ok(v) = serde_json::from_str::<Value>(data) else {
            return Vec::new();
        };
        if looks_like_error(&v) || event == "error" {
            self.errored = true;
            return sse_data(&value_string(&map_error(&v, Endpoint::ChatCompletions)));
        }
        let typ = if event.is_empty() {
            v.get("type").and_then(Value::as_str).unwrap_or("")
        } else {
            event
        };
        match typ {
            "message_start" => {
                let msg = v.get("message").unwrap_or(&v);
                if let Some(id) = msg.get("id").and_then(Value::as_str) {
                    self.ant_to_oai.id = id.to_owned();
                }
                if let Some(model) = msg.get("model").and_then(Value::as_str) {
                    self.ant_to_oai.model = model.to_owned();
                }
                if let Some(u) = msg.get("usage") {
                    self.ant_to_oai.input_tokens =
                        u.get("input_tokens").and_then(Value::as_u64).unwrap_or(0);
                    self.ant_to_oai.cache_read = u
                        .get("cache_read_input_tokens")
                        .and_then(Value::as_u64)
                        .unwrap_or(0);
                }
                self.emit_oai_delta(json!({ "role": "assistant", "content": "" }), None)
            }
            "content_block_start" => {
                let block = v.get("content_block").unwrap_or(&Value::Null);
                match block.get("type").and_then(Value::as_str) {
                    Some("tool_use") => {
                        let idx = self.ant_to_oai.next_tool;
                        self.ant_to_oai.next_tool = self.ant_to_oai.next_tool.saturating_add(1);
                        let id = block.get("id").and_then(Value::as_str).unwrap_or("call_0");
                        let name = block.get("name").and_then(Value::as_str).unwrap_or("");
                        self.emit_oai_delta(
                            json!({
                                "tool_calls": [{
                                    "index": idx,
                                    "id": id,
                                    "type": "function",
                                    "function": { "name": name, "arguments": "" },
                                }]
                            }),
                            None,
                        )
                    }
                    Some("redacted_thinking") => {
                        let data = block.get("data").cloned().unwrap_or(json!(""));
                        self.emit_oai_delta(
                            json!({
                                "thinking": [{ "type": "redacted_thinking", "data": data }]
                            }),
                            None,
                        )
                    }
                    // text / thinking: OpenAI has no block-start
                    _ => Vec::new(),
                }
            }
            "content_block_delta" => {
                let delta = v.get("delta").unwrap_or(&Value::Null);
                match delta.get("type").and_then(Value::as_str) {
                    Some("text_delta") => {
                        let text = delta.get("text").and_then(Value::as_str).unwrap_or("");
                        if text.is_empty() {
                            return Vec::new();
                        }
                        self.emit_oai_delta(json!({ "content": text }), None)
                    }
                    Some("input_json_delta") => {
                        let partial = delta
                            .get("partial_json")
                            .and_then(Value::as_str)
                            .unwrap_or("");
                        let idx = self.ant_to_oai.next_tool.saturating_sub(1);
                        self.emit_oai_delta(
                            json!({
                                "tool_calls": [{
                                    "index": idx,
                                    "function": { "arguments": partial },
                                }]
                            }),
                            None,
                        )
                    }
                    Some("thinking_delta") => {
                        let text = delta
                            .get("thinking")
                            .and_then(Value::as_str)
                            .or_else(|| delta.get("text").and_then(Value::as_str))
                            .unwrap_or("");
                        if text.is_empty() {
                            return Vec::new();
                        }
                        self.emit_oai_delta(
                            json!({ "reasoning_content": text, "reasoning": text }),
                            None,
                        )
                    }
                    Some("signature_delta") => {
                        let sig = delta.get("signature").and_then(Value::as_str).unwrap_or("");
                        if sig.is_empty() {
                            return Vec::new();
                        }
                        self.emit_oai_delta(json!({ "thinking_signature": sig }), None)
                    }
                    _ => Vec::new(),
                }
            }
            "message_delta" => {
                let stop = v
                    .pointer("/delta/stop_reason")
                    .and_then(Value::as_str)
                    .or_else(|| v.get("stop_reason").and_then(Value::as_str));
                let finish = map_stop_to_openai(stop);
                if let Some(u) = v.get("usage")
                    && let Some(o) = u.get("output_tokens").and_then(Value::as_u64)
                {
                    if let Some(t) = u
                        .pointer("/output_tokens_details/thinking_tokens")
                        .and_then(Value::as_u64)
                    {
                        self.ant_to_oai.thinking_tokens = Some(t);
                    }
                    let mut finish_bytes = self.emit_oai_delta(json!({}), Some(finish));
                    let mut usage = json!({
                        "prompt_tokens": self.ant_to_oai.input_tokens,
                        "completion_tokens": o,
                        "total_tokens": self.ant_to_oai.input_tokens.saturating_add(o),
                    });
                    if let Some(obj) = usage.as_object_mut() {
                        if self.ant_to_oai.cache_read > 0 {
                            obj.insert(
                                "prompt_tokens_details".into(),
                                json!({ "cached_tokens": self.ant_to_oai.cache_read }),
                            );
                        }
                        if let Some(t) = self.ant_to_oai.thinking_tokens {
                            obj.insert(
                                "completion_tokens_details".into(),
                                json!({ "reasoning_tokens": t }),
                            );
                        }
                    }
                    finish_bytes.extend(self.emit_oai_usage(usage));
                    return finish_bytes;
                }
                self.emit_oai_delta(json!({}), Some(finish))
            }
            "message_stop" => {
                if self.errored || self.ant_to_oai.done {
                    Vec::new()
                } else {
                    self.ant_to_oai.done = true;
                    sse_data("[DONE]")
                }
            }
            "content_block_stop" => Vec::new(),
            _ => Vec::new(),
        }
    }

    fn emit_oai_delta(&mut self, delta: Value, finish: Option<&str>) -> Vec<u8> {
        let mut choice = json!({ "index": 0, "delta": delta });
        if let Some(f) = finish
            && let Some(obj) = choice.as_object_mut()
        {
            obj.insert("finish_reason".into(), json!(f));
        }
        let chunk = json!({
            "id": self.ant_to_oai.id,
            "object": "chat.completion.chunk",
            "model": self.ant_to_oai.model,
            "choices": [choice],
        });
        sse_data(&value_string(&chunk))
    }

    fn emit_oai_usage(&self, usage: Value) -> Vec<u8> {
        let chunk = json!({
            "id": self.ant_to_oai.id,
            "object": "chat.completion.chunk",
            "model": self.ant_to_oai.model,
            "choices": [],
            "usage": usage,
        });
        sse_data(&value_string(&chunk))
    }

    fn oai_event_to_ant(&mut self, _event: &str, data: &str) -> Vec<u8> {
        if data == "[DONE]" {
            if self.errored {
                return Vec::new();
            }
            return self.oai_to_ant.finish();
        }
        let Ok(v) = serde_json::from_str::<Value>(data) else {
            return Vec::new();
        };
        if looks_like_error(&v) {
            self.errored = true;
            let err = map_error(&v, Endpoint::Messages);
            return sse_named("error", &value_string(&err));
        }
        let mut out = Vec::new();
        if let Some(id) = v.get("id").and_then(Value::as_str)
            && !id.is_empty()
        {
            self.oai_to_ant.id = id.to_owned();
        }
        if let Some(model) = v.get("model").and_then(Value::as_str)
            && !model.is_empty()
        {
            self.oai_to_ant.model = model.to_owned();
        }
        if !self.oai_to_ant.started {
            out.extend(self.oai_to_ant.start());
        }
        let choice = v
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|a| a.first());
        if let Some(delta) = choice.and_then(|c| c.get("delta")) {
            if let Some(content) = delta.get("content").and_then(Value::as_str)
                && !content.is_empty()
            {
                out.extend(self.oai_to_ant.text_delta(content));
            }
            let reasoning = delta
                .get("reasoning_content")
                .and_then(Value::as_str)
                .or_else(|| delta.get("reasoning").and_then(Value::as_str));
            if let Some(r) = reasoning
                && !r.is_empty()
            {
                out.extend(self.oai_to_ant.thinking_delta(r));
            }
            if let Some(sig) = delta.get("thinking_signature").and_then(Value::as_str)
                && !sig.is_empty()
            {
                out.extend(self.oai_to_ant.signature_delta(sig));
            }
            if let Some(blocks) = delta.get("thinking").and_then(Value::as_array) {
                for b in blocks {
                    if b.get("type").and_then(Value::as_str) == Some("redacted_thinking") {
                        out.extend(self.oai_to_ant.redacted_thinking(b));
                    }
                }
            }
            if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
                for c in calls {
                    out.extend(self.oai_to_ant.tool_delta(c));
                }
            }
        }
        if let Some(finish) = choice
            .and_then(|c| c.get("finish_reason"))
            .and_then(Value::as_str)
            && !finish.is_empty()
            && finish != "null"
        {
            self.oai_to_ant.pending_stop = Some(map_stop_to_anthropic(Some(finish)));
        }
        if let Some(u) = v.get("usage") {
            self.oai_to_ant.prompt_tokens = u.get("prompt_tokens").and_then(Value::as_u64);
            self.oai_to_ant.completion_tokens = u.get("completion_tokens").and_then(Value::as_u64);
        }
        if self.oai_to_ant.pending_stop.is_some() && self.oai_to_ant.completion_tokens.is_some() {
            out.extend(self.oai_to_ant.finish());
        }
        out
    }

    fn oai_event_to_resp(&mut self, _event: &str, data: &str) -> Vec<u8> {
        if data == "[DONE]" {
            return self.oai_to_resp.finish();
        }
        let Ok(v) = serde_json::from_str::<Value>(data) else {
            return Vec::new();
        };
        if looks_like_error(&v) {
            self.errored = true;
            return sse_data(&value_string(&map_error(&v, Endpoint::Responses)));
        }
        if let Some(id) = v.get("id").and_then(Value::as_str)
            && self.oai_to_resp.id.is_empty()
        {
            self.oai_to_resp.id = id.to_owned();
        }
        if let Some(model) = v.get("model").and_then(Value::as_str)
            && self.oai_to_resp.model.is_empty()
        {
            self.oai_to_resp.model = model.to_owned();
        }
        let mut out = Vec::new();
        if !self.oai_to_resp.created {
            self.oai_to_resp.created = true;
            if self.oai_to_resp.id.is_empty() {
                self.oai_to_resp.id = "resp_translated".into();
            }
            let created = json!({
                "type": "response.created",
                "response": {
                    "id": self.oai_to_resp.id,
                    "object": "response",
                    "status": "in_progress",
                    "model": self.oai_to_resp.model,
                }
            });
            out.extend(sse_named("response.created", &value_string(&created)));
        }
        let choice = v
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|a| a.first());
        let delta = choice.and_then(|c| c.get("delta")).unwrap_or(&Value::Null);
        if let Some(text) = delta
            .get("content")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            let ev = json!({
                "type": "response.output_text.delta",
                "delta": text,
            });
            out.extend(sse_named("response.output_text.delta", &value_string(&ev)));
        }
        if let Some(reason) = delta
            .get("reasoning_content")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            let ev = json!({
                "type": "response.reasoning_text.delta",
                "delta": reason,
            });
            out.extend(sse_named(
                "response.reasoning_text.delta",
                &value_string(&ev),
            ));
        }
        if let Some(u) = v.get("usage") {
            self.oai_to_resp.usage = Some(u.clone());
            out.extend(self.oai_to_resp.finish());
            return out;
        }
        out
    }

    fn resp_event_to_oai(&mut self, event: &str, data: &str) -> Vec<u8> {
        if data == "[DONE]" {
            if self.resp_to_oai.done {
                return Vec::new();
            }
            self.resp_to_oai.done = true;
            return sse_data("[DONE]");
        }
        let Ok(v) = serde_json::from_str::<Value>(data) else {
            return Vec::new();
        };
        if looks_like_error(&v) || event == "error" {
            self.errored = true;
            return sse_data(&value_string(&map_error(&v, Endpoint::ChatCompletions)));
        }
        let typ = if event.is_empty() {
            v.get("type").and_then(Value::as_str).unwrap_or("")
        } else {
            event
        };
        match typ {
            "response.output_text.delta" => {
                let text = v
                    .get("delta")
                    .and_then(|d| {
                        d.as_str()
                            .map(str::to_owned)
                            .or_else(|| d.get("text").and_then(Value::as_str).map(str::to_owned))
                    })
                    .or_else(|| v.get("text").and_then(Value::as_str).map(str::to_owned))
                    .unwrap_or_default();
                if text.is_empty() {
                    return Vec::new();
                }
                sse_data(&value_string(&json!({
                    "id": "chatcmpl_translated",
                    "object": "chat.completion.chunk",
                    "choices": [{ "index": 0, "delta": { "content": text } }],
                })))
            }
            "response.reasoning_text.delta" | "response.reasoning.delta" => {
                let text = v.get("delta").and_then(Value::as_str).unwrap_or("");
                if text.is_empty() {
                    return Vec::new();
                }
                sse_data(&value_string(&json!({
                    "id": "chatcmpl_translated",
                    "object": "chat.completion.chunk",
                    "choices": [{ "index": 0, "delta": { "reasoning_content": text } }],
                })))
            }
            "response.completed" => {
                let resp = v.get("response").unwrap_or(&v);
                let usage = map_usage_to_openai(resp.get("usage"));
                let mut out = sse_data(&value_string(&json!({
                    "id": "chatcmpl_translated",
                    "object": "chat.completion.chunk",
                    "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }],
                    "usage": usage,
                })));
                if !self.resp_to_oai.done {
                    self.resp_to_oai.done = true;
                    out.extend(sse_data("[DONE]"));
                }
                out
            }
            _ => Vec::new(),
        }
    }
}

impl OaiToResp {
    fn finish(&mut self) -> Vec<u8> {
        if self.completed {
            return Vec::new();
        }
        self.completed = true;
        if self.id.is_empty() {
            self.id = "resp_translated".into();
        }
        let usage = self
            .usage
            .as_ref()
            .map(|u| map_usage_to_responses(Some(u)))
            .unwrap_or_else(|| map_usage_to_responses(None));
        let ev = json!({
            "type": "response.completed",
            "response": {
                "id": self.id,
                "object": "response",
                "status": "completed",
                "model": self.model,
                "usage": usage,
            }
        });
        sse_named("response.completed", &value_string(&ev))
    }
}

impl OaiToAnt {
    fn start(&mut self) -> Vec<u8> {
        self.started = true;
        if self.id.is_empty() {
            self.id = "chatcmpl_translated".into();
        }
        let msg = json!({
            "type": "message_start",
            "message": {
                "id": self.id,
                "type": "message",
                "role": "assistant",
                "model": self.model,
                "content": [],
                "usage": { "input_tokens": 0, "output_tokens": 0 },
            }
        });
        sse_named("message_start", &value_string(&msg))
    }

    fn close_open(&mut self) -> Vec<u8> {
        if self.open == OpenBlock::None {
            return Vec::new();
        }
        let idx = self.next_block.saturating_sub(1);
        self.open = OpenBlock::None;
        sse_named(
            "content_block_stop",
            &value_string(&json!({ "type": "content_block_stop", "index": idx })),
        )
    }

    fn text_delta(&mut self, text: &str) -> Vec<u8> {
        let mut out = Vec::new();
        if self.open == OpenBlock::Tool || self.open == OpenBlock::Thinking {
            out.extend(self.close_open());
        }
        if self.open != OpenBlock::Text {
            let idx = self.next_block;
            self.next_block = self.next_block.saturating_add(1);
            self.open = OpenBlock::Text;
            out.extend(sse_named(
                "content_block_start",
                &value_string(&json!({
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": { "type": "text", "text": "" },
                })),
            ));
        }
        let idx = self.next_block.saturating_sub(1);
        out.extend(sse_named(
            "content_block_delta",
            &value_string(&json!({
                "type": "content_block_delta",
                "index": idx,
                "delta": { "type": "text_delta", "text": text },
            })),
        ));
        out
    }

    fn tool_delta(&mut self, call: &Value) -> Vec<u8> {
        let mut out = Vec::new();
        if self.open == OpenBlock::Text || self.open == OpenBlock::Thinking {
            out.extend(self.close_open());
        }
        let id = call.get("id").and_then(Value::as_str);
        let name = call
            .pointer("/function/name")
            .and_then(Value::as_str)
            .or_else(|| call.get("name").and_then(Value::as_str));
        if self.open != OpenBlock::Tool || id.is_some() {
            if self.open == OpenBlock::Tool {
                out.extend(self.close_open());
            }
            let idx = self.next_block;
            self.next_block = self.next_block.saturating_add(1);
            self.open = OpenBlock::Tool;
            out.extend(sse_named(
                "content_block_start",
                &value_string(&json!({
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {
                        "type": "tool_use",
                        "id": id.unwrap_or("call_0"),
                        "name": name.unwrap_or(""),
                        "input": {},
                    },
                })),
            ));
        }
        if let Some(args) = call.pointer("/function/arguments").and_then(Value::as_str)
            && !args.is_empty()
        {
            let idx = self.next_block.saturating_sub(1);
            out.extend(sse_named(
                "content_block_delta",
                &value_string(&json!({
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": { "type": "input_json_delta", "partial_json": args },
                })),
            ));
        }
        out
    }

    fn thinking_delta(&mut self, text: &str) -> Vec<u8> {
        let mut out = Vec::new();
        if self.open == OpenBlock::Text || self.open == OpenBlock::Tool {
            out.extend(self.close_open());
        }
        if self.open != OpenBlock::Thinking {
            let idx = self.next_block;
            self.next_block = self.next_block.saturating_add(1);
            self.open = OpenBlock::Thinking;
            out.extend(sse_named(
                "content_block_start",
                &value_string(&json!({
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": { "type": "thinking", "thinking": "" },
                })),
            ));
        }
        let idx = self.next_block.saturating_sub(1);
        out.extend(sse_named(
            "content_block_delta",
            &value_string(&json!({
                "type": "content_block_delta",
                "index": idx,
                "delta": { "type": "thinking_delta", "thinking": text },
            })),
        ));
        out
    }

    fn signature_delta(&mut self, sig: &str) -> Vec<u8> {
        let mut out = Vec::new();
        if self.open != OpenBlock::Thinking {
            out.extend(self.thinking_delta(""));
        }
        let idx = self.next_block.saturating_sub(1);
        out.extend(sse_named(
            "content_block_delta",
            &value_string(&json!({
                "type": "content_block_delta",
                "index": idx,
                "delta": { "type": "signature_delta", "signature": sig },
            })),
        ));
        out
    }

    fn redacted_thinking(&mut self, block: &Value) -> Vec<u8> {
        let mut out = Vec::new();
        if self.open != OpenBlock::None {
            out.extend(self.close_open());
        }
        let idx = self.next_block;
        self.next_block = self.next_block.saturating_add(1);
        self.open = OpenBlock::Thinking;
        out.extend(sse_named(
            "content_block_start",
            &value_string(&json!({
                "type": "content_block_start",
                "index": idx,
                "content_block": {
                    "type": "redacted_thinking",
                    "data": block.get("data").cloned().unwrap_or(json!("")),
                },
            })),
        ));
        out
    }

    fn finish(&mut self) -> Vec<u8> {
        if self.finished {
            return Vec::new();
        }
        // A usage-only / `[DONE]`-only upstream stream never opened a text block. `start()`'s
        // bytes used to be discarded (`let _ =`), so the client got `message_delta`/`message_stop`
        // with no `message_start` — which a stock Anthropic SDK treats as a broken stream.
        let mut out = if self.started {
            Vec::new()
        } else {
            self.start()
        };
        self.finished = true;
        out.extend(self.close_open());
        let stop = self.pending_stop.unwrap_or("end_turn");
        let output = self.completion_tokens.unwrap_or(0);
        let input = self.prompt_tokens.unwrap_or(0);
        out.extend(sse_named(
            "message_delta",
            &value_string(&json!({
                "type": "message_delta",
                "delta": { "stop_reason": stop },
                "usage": { "input_tokens": input, "output_tokens": output },
            })),
        ));
        out.extend(sse_named(
            "message_stop",
            &value_string(&json!({ "type": "message_stop" })),
        ));
        out
    }
}

/// Remove and return the first complete event. `scanned` is how much of `buf` an earlier call
/// already searched without finding an end; the search resumes there, not from byte 0.
fn take_event(buf: &mut Vec<u8>, scanned: &mut usize) -> Option<Vec<u8>> {
    // A terminator is judged at its first `\n`, which needs up to two bytes after it — so the last
    // two positions searched before may have been undecidable and are searched again.
    match find_event_end(buf, scanned.saturating_sub(2)) {
        Some(end) => {
            *scanned = 0;
            Some(buf.drain(..end).collect())
        }
        None => {
            *scanned = buf.len();
            None
        }
    }
}

/// End (exclusive) of the first `\n\n` or `\r\n\r\n`, judging only `\n`s at or after `from`.
///
/// One forward pass over newlines. The earlier two-`windows` form ran both searches to completion,
/// so a stream using only one line ending walked the whole buffer for the other on every call.
fn find_event_end(buf: &[u8], from: usize) -> Option<usize> {
    let mut at = from;
    while let Some(off) = memchr::memchr(b'\n', buf.get(at..)?) {
        let nl = at + off;
        match buf.get(nl + 1) {
            Some(b'\n') => return Some(nl + 2),
            Some(b'\r')
                if nl > 0 && buf.get(nl - 1) == Some(&b'\r') && buf.get(nl + 2) == Some(&b'\n') =>
            {
                return Some(nl + 3);
            }
            _ => {}
        }
        at = nl + 1;
    }
    None
}

fn parse_sse(raw: &[u8]) -> (String, String) {
    let text = String::from_utf8_lossy(raw);
    let mut event = String::new();
    let mut data = String::new();
    for line in text.split('\n') {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if let Some(rest) = line.strip_prefix("event:") {
            event = rest.trim().to_owned();
        } else if let Some(rest) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(rest.trim_start());
        }
    }
    (event, data)
}

fn sse_data(data: &str) -> Vec<u8> {
    let mut o = Vec::with_capacity(data.len() + 8);
    o.extend_from_slice(b"data: ");
    o.extend_from_slice(data.as_bytes());
    o.extend_from_slice(b"\n\n");
    o
}

fn sse_named(event: &str, data: &str) -> Vec<u8> {
    let mut o = Vec::with_capacity(event.len() + data.len() + 16);
    o.extend_from_slice(b"event: ");
    o.extend_from_slice(event.as_bytes());
    o.extend_from_slice(b"\ndata: ");
    o.extend_from_slice(data.as_bytes());
    o.extend_from_slice(b"\n\n");
    o
}

fn value_string(v: &Value) -> String {
    serde_json::to_string(v).unwrap_or_else(|_| "{}".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A current (adaptive-generation) Claude id, what most catalog walks land on.
    const OPUS: &str = "claude-opus-4-8";

    #[test]
    fn json_buffer_refuses_past_the_cap_and_keeps_what_it_has() {
        let mut t = TranslateState::new(Endpoint::ChatCompletions);
        assert!(t.push_json(&vec![b' '; MAX_TRANSLATE_BUFFER - 1]));
        assert!(t.push_json(b"{"), "exactly at the cap is allowed");
        assert!(!t.push_json(b"}"), "one byte past is refused");
        assert_eq!(
            t.json_buf.len(),
            MAX_TRANSLATE_BUFFER,
            "a refused chunk appends nothing"
        );
    }

    #[test]
    fn an_unterminated_event_is_held_and_reported_as_pending() {
        let mut b = SseBridge::new(Endpoint::ChatCompletions, Endpoint::Messages);
        let big = format!("event: content_block_delta\ndata: {}", "x".repeat(10_000));
        for chunk in big.as_bytes().chunks(1000) {
            assert!(b.feed(chunk, false).is_empty());
        }
        assert_eq!(b.pending_len(), big.len());
    }

    /// The resumable search must find exactly what a from-scratch search finds, including a
    /// terminator split across chunks at every possible point, for both line endings.
    #[test]
    fn resumable_event_scan_matches_a_fresh_scan_at_every_split() {
        for stream in [
            &b"data: a\n\ndata: bb\n\n: c\n\n"[..],
            &b"data: a\r\n\r\ndata: bb\r\n\r\n"[..],
            &b"data: a\r\ndata: b\n\ndata: c\r\n\r\n"[..],
        ] {
            let mut fresh = stream.to_vec();
            let mut want = Vec::new();
            let mut s = 0;
            while let Some(e) = take_event(&mut fresh, &mut s) {
                want.push(e);
                s = 0;
            }
            for split in 0..=stream.len() {
                let mut buf = Vec::new();
                let mut scanned = 0;
                let mut got = Vec::new();
                for part in [&stream[..split], &stream[split..]] {
                    buf.extend_from_slice(part);
                    while let Some(e) = take_event(&mut buf, &mut scanned) {
                        got.push(e);
                    }
                }
                assert_eq!(
                    got,
                    want,
                    "split at {split} of {:?}",
                    String::from_utf8_lossy(stream)
                );
            }
        }
    }

    fn oai_req() -> Value {
        json!({
            "model": "claude-opus-4-8",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"}
            ],
            "max_tokens": 16,
            "temperature": 0.2,
            "stream": true,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
                }
            }],
            "tool_choice": "auto",
            "stream_options": {"include_usage": true}
        })
    }

    fn to_messages(body: &Value) -> Value {
        let out = request(
            Endpoint::ChatCompletions,
            Endpoint::Messages,
            &serde_json::to_vec(body).unwrap(),
            OPUS,
        );
        serde_json::from_slice(&out).unwrap()
    }

    fn count_markers(v: &Value) -> usize {
        match v {
            Value::Object(m) => {
                usize::from(m.contains_key("cache_control"))
                    + m.values().map(count_markers).sum::<usize>()
            }
            Value::Array(a) => a.iter().map(count_markers).sum(),
            _ => 0,
        }
    }

    /// A stock OpenAI SDK sends no `cache_control`. Without a marker Anthropic caches nothing, so
    /// the static prefix gets one.
    #[test]
    fn an_unmarked_request_caches_its_system_prompt() {
        let v = to_messages(&oai_req());
        assert_eq!(v["system"][0]["text"], "be brief");
        assert_eq!(v["system"][0]["cache_control"]["type"], "ephemeral");
        assert!(
            v["tools"][0].get("cache_control").is_none(),
            "one marker covers tools + system"
        );
        // Single turn: no conversation marker (a write costs 1.25x and a one-shot never reads it).
        assert_eq!(v["messages"][0]["content"], "hi");
        assert_eq!(count_markers(&v), 1, "{v}");
    }

    #[test]
    fn without_a_system_prompt_the_last_tool_is_marked() {
        let mut req = oai_req();
        req["messages"] = json!([{"role": "user", "content": "hi"}]);
        req["tools"]
            .as_array_mut()
            .unwrap()
            .push(json!({"type": "function", "function": {"name": "second"}}));
        let v = to_messages(&req);
        assert!(v.get("system").is_none());
        assert!(v["tools"][0].get("cache_control").is_none());
        assert_eq!(v["tools"][1]["cache_control"]["type"], "ephemeral");
        assert_eq!(count_markers(&v), 1, "{v}");
    }

    /// Once the request is a conversation, the whole prefix is marked so the next turn reads it.
    #[test]
    fn a_conversation_marks_the_end_of_its_last_message() {
        let mut req = oai_req();
        req["messages"] = json!([
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
            }]},
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            {"role": "user", "content": "and tomorrow?"}
        ]);
        let v = to_messages(&req);
        let last = v["messages"].as_array().unwrap().last().unwrap().clone();
        assert_eq!(last["role"], "user");
        let blocks = last["content"].as_array().unwrap();
        assert!(blocks[0].get("cache_control").is_none(), "{last}");
        assert_eq!(blocks.last().unwrap()["cache_control"]["type"], "ephemeral");
        assert_eq!(count_markers(&v), 2, "system + conversation: {v}");
    }

    #[test]
    fn a_string_last_message_becomes_a_marked_text_block() {
        let mut req = oai_req();
        req["messages"] = json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "again"}
        ]);
        let v = to_messages(&req);
        assert_eq!(
            v["messages"][2]["content"],
            json!([{"type": "text", "text": "again", "cache_control": {"type": "ephemeral"}}])
        );
    }

    /// Anthropic rejects `cache_control` on a thinking block.
    #[test]
    fn the_conversation_marker_skips_trailing_thinking_blocks() {
        let mut req = oai_req();
        req["messages"] = json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "hello"},
                {"type": "thinking", "thinking": "hm", "signature": "sig"}
            ]}
        ]);
        let v = to_messages(&req);
        let blocks = v["messages"][1]["content"].as_array().unwrap();
        assert!(
            blocks.iter().any(|b| b["type"] == "thinking"),
            "fixture must carry a thinking block: {blocks:?}"
        );
        for b in blocks {
            if matches!(b["type"].as_str(), Some("thinking" | "redacted_thinking")) {
                assert!(b.get("cache_control").is_none(), "{b}");
            }
        }
        assert!(
            blocks.iter().any(|b| b.get("cache_control").is_some()),
            "a non-thinking block carries the marker: {blocks:?}"
        );
    }

    /// A client that manages its own caching keeps full control.
    #[test]
    fn a_client_marker_anywhere_disables_the_defaults() {
        let mut req = oai_req();
        req["messages"] = json!([
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": [
                {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral", "ttl": "1h"}}
            ]},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "again"}
        ]);
        let v = to_messages(&req);
        assert_eq!(count_markers(&v), 1, "{v}");
        assert_eq!(v["messages"][0]["content"][0]["cache_control"]["ttl"], "1h");
    }

    #[test]
    fn responses_onto_messages_gets_the_same_defaults() {
        let body = json!({
            "model": "claude-opus-4-8",
            "instructions": "be brief",
            "input": "hi"
        });
        let out = request(
            Endpoint::Responses,
            Endpoint::Messages,
            &serde_json::to_vec(&body).unwrap(),
            OPUS,
        );
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["system"][0]["cache_control"]["type"], "ephemeral", "{v}");
    }

    #[test]
    fn openai_request_maps_system_tools_and_drops_stream_options() {
        let body = serde_json::to_vec(&oai_req()).unwrap();
        let out = request(Endpoint::ChatCompletions, Endpoint::Messages, &body, OPUS);
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["system"][0]["text"], "be brief");
        assert_eq!(v["max_tokens"], 16);
        assert_eq!(v["stream"], true);
        assert!(v.get("stream_options").is_none(), "{v}");
        assert_eq!(v["messages"][0]["role"], "user");
        assert_eq!(v["messages"][0]["content"], "hi");
        assert_eq!(v["tools"][0]["name"], "get_weather");
        assert_eq!(
            v["tools"][0]["input_schema"]["properties"]["city"]["type"],
            "string"
        );
        assert_eq!(v["tool_choice"]["type"], "auto");
        assert_eq!(v["model"], "claude-opus-4-8");
    }

    /// Chat Completions nests a tool's description under `function`; the Responses shape is flat.
    /// Every field has to come from the nested object, or the model loses what the tool is for.
    #[test]
    fn openai_tools_keep_their_description_on_responses() {
        let body = serde_json::to_vec(&oai_req()).unwrap();
        for from in [Endpoint::ChatCompletions, Endpoint::Messages] {
            let src = if from == Endpoint::Messages {
                request(Endpoint::ChatCompletions, Endpoint::Messages, &body, OPUS)
            } else {
                body.clone()
            };
            let v: Value =
                serde_json::from_slice(&request(from, Endpoint::Responses, &src, OPUS)).unwrap();
            let tool = &v["tools"][0];
            assert_eq!(tool["name"], "get_weather", "{from:?}: {v}");
            assert_eq!(tool["description"], "weather", "{from:?}: {v}");
            assert_eq!(tool["parameters"]["properties"]["city"]["type"], "string");
        }
    }

    #[test]
    fn openai_request_defaults_max_tokens() {
        let body = br#"{"model":"x","messages":[{"role":"user","content":"hi"}]}"#;
        let v: Value = serde_json::from_slice(&request(
            Endpoint::ChatCompletions,
            Endpoint::Messages,
            body,
            OPUS,
        ))
        .unwrap();
        assert_eq!(v["max_tokens"], DEFAULT_MAX_TOKENS);
    }

    #[test]
    fn tool_loop_round_trips_openai_through_anthropic() {
        let oai = json!({
            "model": "m",
            "max_tokens": 8,
            "messages": [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"}
                }]},
                {"role": "tool", "tool_call_id": "call_1", "content": "64F"}
            ],
            "tools": [{"type": "function", "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {}}
            }}]
        });
        let anth_bytes = request(
            Endpoint::ChatCompletions,
            Endpoint::Messages,
            &serde_json::to_vec(&oai).unwrap(),
            OPUS,
        );
        let anth: Value = serde_json::from_slice(&anth_bytes).unwrap();
        assert_eq!(anth["messages"][1]["role"], "assistant");
        assert_eq!(anth["messages"][1]["content"][0]["type"], "tool_use");
        assert_eq!(anth["messages"][1]["content"][0]["input"]["city"], "SF");
        assert_eq!(anth["messages"][2]["role"], "user");
        assert_eq!(anth["messages"][2]["content"][0]["type"], "tool_result");
        assert_eq!(anth["messages"][2]["content"][0]["tool_use_id"], "call_1");

        let back_bytes = request(
            Endpoint::Messages,
            Endpoint::ChatCompletions,
            &anth_bytes,
            OPUS,
        );
        let back: Value = serde_json::from_slice(&back_bytes).unwrap();
        assert_eq!(back["messages"][1]["tool_calls"][0]["id"], "call_1");
        assert_eq!(back["messages"][2]["role"], "tool");
        assert_eq!(back["messages"][2]["content"], "64F");
        assert_eq!(back["tools"][0]["function"]["name"], "get_weather");
    }

    #[test]
    fn anthropic_stream_true_does_not_inject_stream_options_here() {
        // include_usage is spliced by the proxy onto the *translated OpenAI* body, not here.
        let body = br#"{"model":"gpt-4o-mini","max_tokens":8,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#;
        let v: Value = serde_json::from_slice(&request(
            Endpoint::Messages,
            Endpoint::ChatCompletions,
            body,
            OPUS,
        ))
        .unwrap();
        assert_eq!(v["stream"], true);
        assert!(v.get("stream_options").is_none(), "{v}");
        assert_eq!(v["messages"][0]["role"], "user");
        assert_eq!(v["max_tokens"], 8);
    }

    #[test]
    fn response_json_maps_usage_and_text() {
        let anth = json!({
            "id": "msg_mock",
            "type": "message",
            "model": "claude-opus-4-8",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 13, "output_tokens": 7}
        });
        let oai: Value = serde_json::from_slice(&response_json(
            Endpoint::Messages,
            Endpoint::ChatCompletions,
            &serde_json::to_vec(&anth).unwrap(),
        ))
        .unwrap();
        assert_eq!(oai["object"], "chat.completion");
        assert_eq!(oai["choices"][0]["message"]["content"], "hi");
        assert_eq!(oai["usage"]["prompt_tokens"], 13);
        assert_eq!(oai["usage"]["completion_tokens"], 7);

        let back: Value = serde_json::from_slice(&response_json(
            Endpoint::ChatCompletions,
            Endpoint::Messages,
            &serde_json::to_vec(&oai).unwrap(),
        ))
        .unwrap();
        assert_eq!(back["type"], "message");
        assert_eq!(back["content"][0]["text"], "hi");
        assert_eq!(back["usage"]["input_tokens"], 13);
    }

    #[test]
    fn error_bodies_map_into_the_client_envelope() {
        let oai_err = br#"{"error":{"message":"nope","type":"invalid_request_error"}}"#;
        let anth: Value = serde_json::from_slice(&response_json(
            Endpoint::ChatCompletions,
            Endpoint::Messages,
            oai_err,
        ))
        .unwrap();
        assert_eq!(anth["type"], "error");
        assert_eq!(anth["error"]["message"], "nope");

        let back: Value = serde_json::from_slice(&response_json(
            Endpoint::Messages,
            Endpoint::ChatCompletions,
            &serde_json::to_vec(&anth).unwrap(),
        ))
        .unwrap();
        assert_eq!(back["error"]["type"], "invalid_request_error");
    }

    #[test]
    fn anthropic_sse_becomes_chat_completion_chunk_without_waiting_for_stop() {
        let mut b = SseBridge::new(Endpoint::ChatCompletions, Endpoint::Messages);
        let start = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_x\",\"model\":\"claude-opus-4-8\",\"usage\":{\"input_tokens\":13,\"output_tokens\":1}}}\n\n",
        );
        let delta = concat!(
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
        );
        let out = String::from_utf8(b.feed(start.as_bytes(), false)).unwrap();
        assert!(out.contains("chat.completion.chunk"), "{out}");
        let out = String::from_utf8(b.feed(delta.as_bytes(), false)).unwrap();
        assert!(out.contains("chat.completion.chunk"), "{out}");
        assert!(out.contains("\"hi\""), "{out}");
        assert!(
            !out.contains("[DONE]"),
            "must not wait for message_stop: {out}"
        );
    }

    #[test]
    fn sse_event_split_across_chunks_is_held() {
        let mut b = SseBridge::new(Endpoint::ChatCompletions, Endpoint::Messages);
        let first = b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\"";
        let out = b.feed(first, false);
        assert!(out.is_empty(), "incomplete event must be withheld");
        let rest = b",\"delta\":{\"type\":\"text_delta\",\"text\":\"z\"}}\n\n";
        let out = String::from_utf8(b.feed(rest, false)).unwrap();
        assert!(out.contains("\"z\""), "{out}");
    }

    #[test]
    fn openai_sse_becomes_anthropic_events_and_does_not_wait_for_done() {
        let mut b = SseBridge::new(Endpoint::Messages, Endpoint::ChatCompletions);
        let chunk = "data: {\"id\":\"chatcmpl-x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n";
        let out = String::from_utf8(b.feed(chunk.as_bytes(), false)).unwrap();
        assert!(out.contains("event: message_start"), "{out}");
        assert!(out.contains("text_delta"), "{out}");
        assert!(out.contains("hi"), "{out}");
        assert!(
            !out.contains("message_stop"),
            "must not wait for [DONE]: {out}"
        );
    }

    #[test]
    fn same_wire_is_a_byte_copy() {
        let body = br#"{"model":"x"}"#;
        assert_eq!(
            request(
                Endpoint::ChatCompletions,
                Endpoint::ChatCompletions,
                body,
                OPUS
            ),
            body
        );
        assert_eq!(
            response_json(Endpoint::Messages, Endpoint::Messages, body),
            body
        );
    }

    #[test]
    fn openai_sse_flush_without_deltas_still_emits_message_start() {
        let mut b = SseBridge::new(Endpoint::Messages, Endpoint::ChatCompletions);
        let out = String::from_utf8(b.feed(b"", true)).unwrap();
        assert!(out.contains("event: message_start"), "{out}");
        assert!(out.contains("event: message_stop"), "{out}");
    }

    #[test]
    fn anthropic_tool_sse_becomes_openai_tool_calls() {
        let mut b = SseBridge::new(Endpoint::ChatCompletions, Endpoint::Messages);
        let src = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_x\",\"model\":\"claude-opus-4-8\",\"usage\":{\"input_tokens\":13,\"output_tokens\":1}}}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get_weather\",\"input\":{}}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\\\"SF\\\"}\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":7}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );
        let mut out = Vec::new();
        for byte in src.as_bytes() {
            out.extend(b.feed(&[*byte], false));
        }
        out.extend(b.feed(b"", true));
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("\"tool_calls\""), "{text}");
        assert!(text.contains("get_weather"), "{text}");
        assert!(text.contains("toolu_1"), "{text}");
        assert!(text.contains("city"), "{text}");
        assert!(text.contains("\"finish_reason\":\"tool_calls\""), "{text}");
        assert!(text.contains("\"prompt_tokens\":13"), "{text}");
        assert!(text.contains("[DONE]"), "{text}");
    }

    #[test]
    fn openai_tool_sse_becomes_anthropic_tool_use() {
        let mut b = SseBridge::new(Endpoint::Messages, Endpoint::ChatCompletions);
        let src = concat!(
            "data: {\"id\":\"chatcmpl-x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]}}]}\n\n",
            "data: {\"id\":\"chatcmpl-x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\":\\\"SF\\\"}\"}}]}}]}\n\n",
            "data: {\"id\":\"chatcmpl-x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: {\"id\":\"chatcmpl-x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o\",\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":9}}\n\n",
            "data: [DONE]\n\n",
        );
        let out = String::from_utf8(b.feed(src.as_bytes(), true)).unwrap();
        assert!(out.contains("event: message_start"), "{out}");
        assert!(out.contains("\"type\":\"tool_use\""), "{out}");
        assert!(out.contains("get_weather"), "{out}");
        assert!(out.contains("call_1"), "{out}");
        assert!(out.contains("input_json_delta"), "{out}");
        assert!(out.contains("\"stop_reason\":\"tool_use\""), "{out}");
        assert!(out.contains("event: message_stop"), "{out}");
        assert!(!out.contains("chat.completion.chunk"), "{out}");
    }

    #[test]
    fn sse_error_events_map_into_the_client_envelope() {
        let mut oai = SseBridge::new(Endpoint::ChatCompletions, Endpoint::Messages);
        let anth_err = concat!(
            "event: error\n",
            "data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"try again\"}}\n\n",
        );
        let out = String::from_utf8(oai.feed(anth_err.as_bytes(), true)).unwrap();
        assert!(out.contains("\"message\":\"try again\""), "{out}");
        assert!(out.contains("overloaded_error"), "{out}");
        assert!(!out.contains("event: error"), "{out}");

        let mut anth = SseBridge::new(Endpoint::Messages, Endpoint::ChatCompletions);
        let oai_err = "data: {\"error\":{\"message\":\"try again\",\"type\":\"server_error\"}}\n\n";
        let out = String::from_utf8(anth.feed(oai_err.as_bytes(), true)).unwrap();
        assert!(out.contains("event: error"), "{out}");
        assert!(out.contains("\"type\":\"error\""), "{out}");
        assert!(out.contains("try again"), "{out}");
        assert!(
            !out.contains("event: message_start"),
            "error stream must not invent a success close: {out}"
        );
    }

    #[test]
    fn nonstream_tool_json_round_trips() {
        let anth = json!({
            "id": "msg_mock",
            "type": "message",
            "model": "claude-opus-4-8",
            "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "SF"}}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 13, "output_tokens": 7}
        });
        let oai: Value = serde_json::from_slice(&response_json(
            Endpoint::Messages,
            Endpoint::ChatCompletions,
            &serde_json::to_vec(&anth).unwrap(),
        ))
        .unwrap();
        assert_eq!(oai["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            oai["choices"][0]["message"]["tool_calls"][0]["id"],
            "toolu_1"
        );
        assert_eq!(
            oai["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
        let args = oai["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .unwrap();
        assert!(args.contains("SF"), "{args}");

        let back: Value = serde_json::from_slice(&response_json(
            Endpoint::ChatCompletions,
            Endpoint::Messages,
            &serde_json::to_vec(&oai).unwrap(),
        ))
        .unwrap();
        assert_eq!(back["stop_reason"], "tool_use");
        assert_eq!(back["content"][0]["type"], "tool_use");
        assert_eq!(back["content"][0]["input"]["city"], "SF");
    }

    #[test]
    fn responses_session_field_names_previous_response_id_first() {
        let body = br#"{"model":"gpt-4o","previous_response_id":"resp_1","store":false}"#;
        assert_eq!(responses_session_field(body), Some("previous_response_id"));
        let omitted = br#"{"model":"gpt-4o","input":"hi"}"#;
        assert_eq!(responses_session_field(omitted), Some("store"));
        let stored = br#"{"model":"gpt-4o","store":true}"#;
        assert_eq!(responses_session_field(stored), Some("store"));
        let one_shot = br#"{"model":"gpt-4o","store":false,"input":"hi"}"#;
        assert_eq!(responses_session_field(one_shot), None);
        let empty_prev = br#"{"model":"gpt-4o","previous_response_id":"","store":false}"#;
        assert_eq!(responses_session_field(empty_prev), None);
        assert_eq!(responses_session_field(b"not-json"), Some("store"));
    }

    #[test]
    fn responses_to_chat_drops_session_fields_and_maps_input() {
        let body = serde_json::to_vec(&json!({
            "model": "gpt-4o",
            "input": [{"role":"user","content":[{"type":"input_text","text":"hi"}]}],
            "max_output_tokens": 16,
            "store": false,
            "include": ["reasoning.encrypted_content"],
            "truncation": "auto",
            "previous_response_id": "resp_x",
        }))
        .unwrap();
        let v: Value = serde_json::from_slice(&responses_to_chat(&body)).unwrap();
        assert!(v.get("store").is_none(), "{v}");
        assert!(v.get("previous_response_id").is_none(), "{v}");
        assert!(v.get("include").is_none(), "{v}");
        assert!(v.get("truncation").is_none(), "{v}");
        assert!(v.get("input").is_none(), "{v}");
        assert_eq!(v["max_tokens"], 16);
        assert_eq!(v["messages"][0]["role"], "user");
        assert_eq!(v["messages"][0]["content"], "hi");
        assert_eq!(v["model"], "gpt-4o");
    }

    #[test]
    fn openai_cache_control_and_reasoning_reach_anthropic_fields() {
        let oai = json!({
            "model": "claude-opus-4-8",
            "reasoning_effort": "high",
            "max_tokens": 16,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
                ]
            }],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}}
                },
                "cache_control": {"type": "ephemeral"}
            }]
        });
        let v: Value = serde_json::from_slice(&request(
            Endpoint::ChatCompletions,
            Endpoint::Messages,
            &serde_json::to_vec(&oai).unwrap(),
            OPUS,
        ))
        .unwrap();
        // Opus 4.8 rejects `budget_tokens`: effort goes to `output_config`.
        assert_eq!(v["thinking"]["type"], "adaptive");
        assert_eq!(v["output_config"]["effort"], "high");
        assert!(v["thinking"].get("budget_tokens").is_none(), "{v}");
        assert_eq!(
            v["messages"][0]["content"][0]["cache_control"]["type"],
            "ephemeral"
        );
        assert_eq!(v["tools"][0]["cache_control"]["type"], "ephemeral");
        assert!(
            v.get("reasoning_effort").is_none(),
            "Anthropic takes thinking, not reasoning_effort: {v}"
        );
        assert!(
            v.get("max_output_tokens").is_none() && v.get("input").is_none(),
            "Responses-only fields must stay dropped: {v}"
        );
    }

    #[test]
    fn anthropic_thinking_becomes_openai_reasoning_effort() {
        let anth = json!({
            "model": "m",
            "max_tokens": 8,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "messages": [{"role": "user", "content": "hi"}]
        });
        let v: Value = serde_json::from_slice(&request(
            Endpoint::Messages,
            Endpoint::ChatCompletions,
            &serde_json::to_vec(&anth).unwrap(),
            OPUS,
        ))
        .unwrap();
        assert_eq!(v["reasoning_effort"], "low");
        assert!(v.get("thinking").is_none(), "{v}");
    }

    #[test]
    fn thinking_and_redacted_thinking_round_trip_in_history() {
        let anth = json!({
            "model": "m",
            "max_tokens": 8,
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "plan", "signature": "sig"},
                    {"type": "redacted_thinking", "data": "redacted"},
                    {"type": "text", "text": "hi"}
                ]
            }]
        });
        let oai: Value = serde_json::from_slice(&request(
            Endpoint::Messages,
            Endpoint::ChatCompletions,
            &serde_json::to_vec(&anth).unwrap(),
            OPUS,
        ))
        .unwrap();
        assert_eq!(oai["messages"][0]["content"], "hi");
        assert_eq!(oai["messages"][0]["reasoning_content"], "plan");
        assert_eq!(oai["messages"][0]["thinking"][0]["type"], "thinking");
        assert_eq!(
            oai["messages"][0]["thinking"][1]["type"],
            "redacted_thinking"
        );

        let back: Value = serde_json::from_slice(&request(
            Endpoint::ChatCompletions,
            Endpoint::Messages,
            &serde_json::to_vec(&oai).unwrap(),
            OPUS,
        ))
        .unwrap();
        let content = &back["messages"][0]["content"];
        assert_eq!(content[0]["type"], "thinking");
        assert_eq!(content[0]["thinking"], "plan");
        assert_eq!(content[0]["signature"], "sig");
        assert_eq!(content[1]["type"], "redacted_thinking");
        assert_eq!(content[2]["type"], "text");
        assert_eq!(content[2]["text"], "hi");
    }

    #[test]
    fn http_image_urls_become_anthropic_url_sources() {
        let oai = json!({
            "model": "m",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "see"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}
                ]
            }]
        });
        let v: Value = serde_json::from_slice(&request(
            Endpoint::ChatCompletions,
            Endpoint::Messages,
            &serde_json::to_vec(&oai).unwrap(),
            OPUS,
        ))
        .unwrap();
        let content = &v["messages"][0]["content"];
        assert_eq!(content[0]["text"], "see");
        assert_eq!(content[1]["type"], "image");
        assert_eq!(content[1]["source"]["type"], "url");
        assert_eq!(content[1]["source"]["url"], "https://example.com/x.png");
    }

    #[test]
    fn anthropic_url_sources_become_openai_image_urls() {
        let msg = json!({
            "model": "m",
            "max_tokens": 16,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "url", "url": "https://example.com/x.png"}},
                    {"type": "text", "text": "see"}
                ]
            }]
        });
        let v: Value = serde_json::from_slice(&request(
            Endpoint::Messages,
            Endpoint::ChatCompletions,
            &serde_json::to_vec(&msg).unwrap(),
            OPUS,
        ))
        .unwrap();
        let content = &v["messages"][0]["content"];
        assert_eq!(content[0]["type"], "image_url");
        assert_eq!(content[0]["image_url"]["url"], "https://example.com/x.png");
        assert_eq!(content[1]["text"], "see");
    }

    /// A `url` source that is not http(s) (`file:`, `ftp:`, garbage) is not forwarded as an
    /// OpenAI `image_url` the upstream would try, and fail, to fetch.
    #[test]
    fn non_http_url_sources_are_not_forwarded() {
        let msg = json!({
            "model": "m",
            "max_tokens": 16,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "url", "url": "file:///etc/passwd"}},
                    {"type": "text", "text": "see"}
                ]
            }]
        });
        let v: Value = serde_json::from_slice(&request(
            Endpoint::Messages,
            Endpoint::ChatCompletions,
            &serde_json::to_vec(&msg).unwrap(),
            OPUS,
        ))
        .unwrap();
        assert_eq!(v["messages"][0]["content"], "see");
    }

    #[test]
    fn http_image_urls_pass_between_responses_and_chat() {
        let resp = json!({
            "model": "m",
            "store": false,
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "see"},
                    {"type": "input_image", "image_url": "https://example.com/x.png"}
                ]
            }]
        });
        let chat: Value = serde_json::from_slice(&request(
            Endpoint::Responses,
            Endpoint::ChatCompletions,
            &serde_json::to_vec(&resp).unwrap(),
            OPUS,
        ))
        .unwrap();
        let content = &chat["messages"][0]["content"];
        assert_eq!(content[1]["type"], "image_url");
        assert_eq!(content[1]["image_url"]["url"], "https://example.com/x.png");

        let back: Value = serde_json::from_slice(&request(
            Endpoint::ChatCompletions,
            Endpoint::Responses,
            &serde_json::to_vec(&chat).unwrap(),
            OPUS,
        ))
        .unwrap();
        let content = &back["input"][0]["content"];
        assert_eq!(content[1]["type"], "input_image");
        assert_eq!(content[1]["image_url"], "https://example.com/x.png");

        // And on through Messages: Responses → Messages is composed through Chat Completions.
        let msgs: Value = serde_json::from_slice(&request(
            Endpoint::Responses,
            Endpoint::Messages,
            &serde_json::to_vec(&resp).unwrap(),
            OPUS,
        ))
        .unwrap();
        let content = &msgs["messages"][0]["content"];
        assert_eq!(content[1]["source"]["type"], "url");
        assert_eq!(content[1]["source"]["url"], "https://example.com/x.png");
    }

    #[test]
    fn anthropic_thinking_sse_reappears_on_openai_stream() {
        let mut b = SseBridge::new(Endpoint::ChatCompletions, Endpoint::Messages);
        let src = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_x\",\"model\":\"claude-opus-4-8\",\"usage\":{\"input_tokens\":13,\"cache_read_input_tokens\":4}}}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"plan\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":7,\"output_tokens_details\":{\"thinking_tokens\":3}}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );
        let out = String::from_utf8(b.feed(src.as_bytes(), true)).unwrap();
        assert!(out.contains("\"reasoning_content\":\"plan\""), "{out}");
        assert!(out.contains("\"hi\""), "{out}");
        assert!(out.contains("\"reasoning_tokens\":3"), "{out}");
        assert!(out.contains("\"cached_tokens\":4"), "{out}");
    }

    #[test]
    fn openai_reasoning_sse_becomes_anthropic_thinking() {
        let mut b = SseBridge::new(Endpoint::Messages, Endpoint::ChatCompletions);
        let src = concat!(
            "data: {\"id\":\"chatcmpl-x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"reasoning_content\":\"plan\"}}]}\n\n",
            "data: {\"id\":\"chatcmpl-x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
            "data: [DONE]\n\n",
        );
        let out = String::from_utf8(b.feed(src.as_bytes(), true)).unwrap();
        assert!(out.contains("\"type\":\"thinking\""), "{out}");
        assert!(out.contains("thinking_delta"), "{out}");
        assert!(out.contains("plan"), "{out}");
        assert!(out.contains("text_delta"), "{out}");
        assert!(out.contains("hi"), "{out}");
    }

    #[test]
    fn client_usage_maps_cache_and_reasoning_without_changing_shape_of_tools() {
        let anth = json!({
            "id": "msg_mock",
            "type": "message",
            "model": "claude-opus-4-8",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 13,
                "output_tokens": 7,
                "cache_read_input_tokens": 4,
                "output_tokens_details": {"thinking_tokens": 3}
            }
        });
        let oai: Value = serde_json::from_slice(&response_json(
            Endpoint::Messages,
            Endpoint::ChatCompletions,
            &serde_json::to_vec(&anth).unwrap(),
        ))
        .unwrap();
        assert_eq!(oai["usage"]["prompt_tokens_details"]["cached_tokens"], 4);
        assert_eq!(
            oai["usage"]["completion_tokens_details"]["reasoning_tokens"],
            3
        );
    }

    fn stock_responses(model: &str) -> Value {
        json!({
            "model": model,
            "input": [{
                "role": "user",
                "content": [{"type": "input_text", "text": "hi"}]
            }],
            "max_output_tokens": 16,
            "stream": true,
            "store": false,
            "reasoning": { "effort": "high" },
        })
    }

    #[test]
    fn stock_responses_body_becomes_chat_completions_and_drops_store() {
        let v: Value = serde_json::from_slice(&request(
            Endpoint::Responses,
            Endpoint::ChatCompletions,
            &serde_json::to_vec(&stock_responses("gpt-4o-mini")).unwrap(),
            OPUS,
        ))
        .unwrap();
        assert_eq!(v["messages"][0]["role"], "user");
        assert_eq!(v["messages"][0]["content"], "hi");
        assert_eq!(v["max_tokens"], 16);
        assert_eq!(v["stream"], true);
        assert_eq!(v["reasoning_effort"], "high");
        assert!(v.get("store").is_none(), "{v}");
        assert!(v.get("input").is_none(), "{v}");
        assert!(v.get("max_output_tokens").is_none(), "{v}");
    }

    #[test]
    fn responses_body_with_a_claude_id_becomes_messages() {
        let v: Value = serde_json::from_slice(&request(
            Endpoint::Responses,
            Endpoint::Messages,
            &serde_json::to_vec(&stock_responses("claude-opus-4-8")).unwrap(),
            OPUS,
        ))
        .unwrap();
        assert_eq!(v["messages"][0]["role"], "user");
        assert_eq!(v["messages"][0]["content"], "hi");
        assert_eq!(v["max_tokens"], 16);
        assert_eq!(v["thinking"]["type"], "adaptive");
        assert!(v.get("store").is_none(), "{v}");
        assert!(v.get("input").is_none(), "{v}");
    }

    #[test]
    fn openai_sse_becomes_responses_events_without_waiting_for_done() {
        let mut b = SseBridge::new(Endpoint::Responses, Endpoint::ChatCompletions);
        let chunk = "data: {\"id\":\"chatcmpl-x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n";
        let out = String::from_utf8(b.feed(chunk.as_bytes(), false)).unwrap();
        assert!(out.contains("response.output_text.delta"), "{out}");
        assert!(out.contains("\"hi\""), "{out}");
        assert!(
            !out.contains("response.completed"),
            "must not wait for [DONE]: {out}"
        );
        let rest = "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":9}}\n\ndata: [DONE]\n\n";
        let out = String::from_utf8(b.feed(rest.as_bytes(), true)).unwrap();
        assert!(out.contains("response.completed"), "{out}");
        assert!(out.contains("\"input_tokens\":5"), "{out}");
        assert!(out.contains("\"output_tokens\":9"), "{out}");
    }

    #[test]
    fn anthropic_sse_becomes_responses_via_chat() {
        let mut b = SseBridge::new(Endpoint::Responses, Endpoint::Messages);
        let src = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_x\",\"model\":\"claude-opus-4-8\",\"usage\":{\"input_tokens\":13,\"output_tokens\":1}}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":7}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );
        let out = String::from_utf8(b.feed(src.as_bytes(), true)).unwrap();
        assert!(out.contains("response.output_text.delta"), "{out}");
        assert!(out.contains("\"hi\""), "{out}");
        assert!(out.contains("response.completed"), "{out}");
        assert!(out.contains("\"input_tokens\":13"), "{out}");
    }

    #[test]
    fn responses_json_round_trips_text_and_usage() {
        let chat = json!({
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "hi" },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18 }
        });
        let resp: Value = serde_json::from_slice(&response_json(
            Endpoint::ChatCompletions,
            Endpoint::Responses,
            &serde_json::to_vec(&chat).unwrap(),
        ))
        .unwrap();
        assert_eq!(resp["object"], "response");
        assert_eq!(resp["output"][0]["content"][0]["text"], "hi");
        assert_eq!(resp["usage"]["input_tokens"], 11);
        assert_eq!(resp["usage"]["output_tokens"], 7);
        assert!(resp.get("choices").is_none(), "{resp}");
    }

    // ---- Model-aware reasoning and sampling ----

    #[test]
    fn claude_generation_parses_every_candidate_spelling() {
        use ClaudeGen::*;
        // (id, reasoning, forced tool_choice accepted)
        for (id, reasoning, forced) in [
            ("claude-opus-4-8", Adaptive, true),
            ("anthropic/claude-opus-4.8", Adaptive, true),
            ("global.anthropic.claude-opus-4-7-v1:0", Adaptive, true),
            ("claude-opus-5", Adaptive, true),
            ("claude-opus-5-5", Adaptive, false),
            ("anthropic/claude-opus-5.5", Adaptive, false),
            ("claude-sonnet-5", Adaptive, true),
            ("claude-sonnet-5-5", Adaptive, false),
            ("claude-fable-5", Adaptive, true),
            ("claude-fable-5-1", Adaptive, false),
            ("claude-mythos-5-1", Adaptive, false),
            ("claude-opus-4-6", Adaptive46, true),
            ("claude-sonnet-4-6", Adaptive46, true),
            ("claude-haiku-4-5", Budget, true),
            ("anthropic.claude-haiku-4-5-20251001-v1:0", Budget, true),
            ("claude-sonnet-4-20250514", Budget, true),
            ("claude-opus-4-1", Budget, true),
            ("claude-3-haiku", Budget, true),
            ("claude-3-5-sonnet-latest", Budget, true),
            ("claude-nova", Adaptive, false),
            ("kimi-k3", Budget, true),
            ("", Budget, true),
        ] {
            let m = ClaudeModel::of(id);
            assert_eq!(
                (m.reasoning, m.forced_tool_choice),
                (reasoning, forced),
                "{id}"
            );
        }
    }

    fn chat(extra: Value) -> Value {
        let mut req = json!({
            "model": "m",
            "max_tokens": 16000,
            "messages": [{"role": "user", "content": "hi"}]
        });
        if let (Some(r), Some(e)) = (req.as_object_mut(), extra.as_object()) {
            for (k, v) in e {
                r.insert(k.clone(), v.clone());
            }
        }
        req
    }

    fn to_claude(body: &Value, model: &str) -> Value {
        serde_json::from_slice(&request(
            Endpoint::ChatCompletions,
            Endpoint::Messages,
            &serde_json::to_vec(body).unwrap(),
            model,
        ))
        .unwrap()
    }

    /// Current Claude models 400 on `budget_tokens` and on `temperature` / `top_p`.
    #[test]
    fn current_claude_gets_adaptive_thinking_and_no_sampling() {
        let v = to_claude(
            &chat(json!({"reasoning_effort": "xhigh", "temperature": 0.2, "top_p": 0.9})),
            "claude-opus-4-8",
        );
        assert_eq!(v["thinking"], json!({"type": "adaptive"}));
        assert_eq!(v["output_config"]["effort"], "xhigh");
        assert!(v.get("temperature").is_none(), "{v}");
        assert!(v.get("top_p").is_none(), "{v}");
    }

    /// Several current models reject `thinking: disabled`; "no reasoning" is the lowest effort.
    #[test]
    fn reasoning_none_on_current_claude_omits_thinking() {
        let v = to_claude(
            &chat(json!({"reasoning_effort": "none"})),
            "claude-opus-5-5",
        );
        // Opus 5.5 always thinks; the explicit `adaptive` only carries `block_binding`.
        assert_eq!(v["thinking"]["type"], "adaptive", "{v}");
        assert_eq!(v["output_config"]["effort"], "low");
    }

    #[test]
    fn claude_4_6_has_no_xhigh() {
        let v = to_claude(
            &chat(json!({"reasoning_effort": "xhigh"})),
            "claude-sonnet-4-6",
        );
        assert_eq!(v["output_config"]["effort"], "max");
    }

    #[test]
    fn older_claude_gets_a_budget_below_max_tokens() {
        let v = to_claude(
            &chat(json!({"reasoning_effort": "high"})),
            "claude-haiku-4-5",
        );
        assert_eq!(v["thinking"]["type"], "enabled");
        assert_eq!(
            v["thinking"]["budget_tokens"], 8000,
            "half of max_tokens 16000: {v}"
        );
        assert!(
            v.get("output_config").is_none(),
            "Haiku 4.5 rejects effort: {v}"
        );

        let mut small = chat(json!({"reasoning_effort": "high"}));
        small["max_tokens"] = json!(1000);
        let v = to_claude(&small, "claude-haiku-4-5");
        assert!(
            v.get("thinking").is_none(),
            "no room to think under 1024: {v}"
        );
    }

    /// Sampling is fine on older models, except alongside thinking.
    #[test]
    fn older_claude_keeps_sampling_unless_thinking() {
        let v = to_claude(&chat(json!({"temperature": 0.2})), "claude-haiku-4-5");
        assert_eq!(v["temperature"], 0.2);
        let v = to_claude(
            &chat(json!({"temperature": 0.2, "reasoning_effort": "low"})),
            "claude-haiku-4-5",
        );
        assert!(v.get("temperature").is_none(), "{v}");
    }

    // ---- Fields that used to vanish ----

    fn schema() -> Value {
        json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": false
        })
    }

    #[test]
    fn json_schema_response_format_becomes_output_config_format() {
        let v = to_claude(
            &chat(json!({"response_format": {
                "type": "json_schema",
                "json_schema": {"name": "a", "strict": true, "schema": schema()}
            }})),
            OPUS,
        );
        assert_eq!(
            v["output_config"]["format"],
            json!({"type": "json_schema", "schema": schema()})
        );
        assert!(v.get("response_format").is_none(), "{v}");
    }

    #[test]
    fn format_and_effort_share_one_output_config() {
        let v = to_claude(
            &chat(json!({
                "reasoning_effort": "low",
                "response_format": {"type": "json_schema", "json_schema": {"name": "a", "schema": schema()}}
            })),
            OPUS,
        );
        assert_eq!(v["output_config"]["effort"], "low");
        assert_eq!(v["output_config"]["format"]["type"], "json_schema");
    }

    #[test]
    fn output_config_format_becomes_response_format() {
        let anth = json!({
            "model": "claude-opus-4-8",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "output_config": {"format": {"type": "json_schema", "schema": schema()}, "effort": "medium"}
        });
        let v: Value = serde_json::from_slice(&request(
            Endpoint::Messages,
            Endpoint::ChatCompletions,
            &serde_json::to_vec(&anth).unwrap(),
            "gpt-5",
        ))
        .unwrap();
        assert_eq!(v["response_format"]["type"], "json_schema");
        assert_eq!(v["response_format"]["json_schema"]["schema"], schema());
        assert_eq!(v["reasoning_effort"], "medium");
    }

    #[test]
    fn parallel_tool_calls_false_disables_parallel_tool_use_both_ways() {
        let v = to_claude(
            &chat(json!({
                "parallel_tool_calls": false,
                "tools": [{"type": "function", "function": {"name": "f"}}]
            })),
            OPUS,
        );
        assert_eq!(
            v["tool_choice"],
            json!({"type": "auto", "disable_parallel_tool_use": true})
        );
        let back: Value = serde_json::from_slice(&request(
            Endpoint::Messages,
            Endpoint::ChatCompletions,
            &serde_json::to_vec(&v).unwrap(),
            "gpt-5",
        ))
        .unwrap();
        assert_eq!(back["parallel_tool_calls"], false);
    }

    #[test]
    fn user_becomes_metadata_user_id_both_ways() {
        let v = to_claude(&chat(json!({"user": "u-123"})), OPUS);
        assert_eq!(v["metadata"], json!({"user_id": "u-123"}));
        let back: Value = serde_json::from_slice(&request(
            Endpoint::Messages,
            Endpoint::ChatCompletions,
            &serde_json::to_vec(&v).unwrap(),
            "gpt-5",
        ))
        .unwrap();
        assert_eq!(back["user"], "u-123");
    }

    #[test]
    fn an_inline_pdf_becomes_a_document_and_back() {
        let v = to_claude(
            &chat(json!({"messages": [{"role": "user", "content": [
                {"type": "file", "file": {"filename": "q3.pdf", "file_data": "data:application/pdf;base64,JVBERi0="}},
                {"type": "text", "text": "summarize"}
            ]}]})),
            OPUS,
        );
        let doc = &v["messages"][0]["content"][0];
        assert_eq!(doc["type"], "document");
        assert_eq!(
            doc["source"],
            json!({"type": "base64", "media_type": "application/pdf", "data": "JVBERi0="})
        );
        assert_eq!(doc["title"], "q3.pdf");
        let back: Value = serde_json::from_slice(&request(
            Endpoint::Messages,
            Endpoint::ChatCompletions,
            &serde_json::to_vec(&v).unwrap(),
            "gpt-5",
        ))
        .unwrap();
        assert_eq!(
            back["messages"][0]["content"][0],
            json!({"type": "file", "file": {"filename": "q3.pdf", "file_data": "data:application/pdf;base64,JVBERi0="}})
        );
    }

    /// No Messages equivalent: forwarded so the provider names the problem. Dropping them would
    /// answer a question about input the model never saw.
    #[test]
    fn unmappable_input_is_forwarded_for_the_provider_to_reject() {
        let audio =
            json!({"type": "input_audio", "input_audio": {"data": "UklG", "format": "wav"}});
        let by_id = json!({"type": "file", "file": {"file_id": "file-abc"}});
        let v = to_claude(
            &chat(
                json!({"n": 2, "logprobs": true, "messages": [{"role": "user", "content": [
                    audio.clone(), by_id.clone(), {"type": "text", "text": "what is this"}
                ]}]}),
            ),
            OPUS,
        );
        assert_eq!(v["n"], 2);
        assert_eq!(v["logprobs"], true);
        assert_eq!(v["messages"][0]["content"][0], audio);
        assert_eq!(v["messages"][0]["content"][1], by_id);
    }

    #[test]
    fn harmless_defaults_are_not_forwarded() {
        let v = to_claude(&chat(json!({"n": 1, "logprobs": false, "seed": 7})), OPUS);
        for key in ["n", "logprobs", "seed"] {
            assert!(v.get(key).is_none(), "{key}: {v}");
        }
    }

    #[test]
    fn responses_text_format_and_input_file_reach_messages() {
        let body = json!({
            "model": "claude-opus-4-8",
            "store": false,
            "text": {"format": {"type": "json_schema", "name": "a", "strict": true, "schema": schema()}},
            "input": [{"role": "user", "content": [
                {"type": "input_file", "filename": "q3.pdf", "file_data": "data:application/pdf;base64,JVBERi0="},
                {"type": "input_text", "text": "summarize"}
            ]}]
        });
        let v: Value = serde_json::from_slice(&request(
            Endpoint::Responses,
            Endpoint::Messages,
            &serde_json::to_vec(&body).unwrap(),
            OPUS,
        ))
        .unwrap();
        assert_eq!(v["output_config"]["format"]["schema"], schema());
        assert_eq!(v["messages"][0]["content"][0]["type"], "document");
    }

    #[test]
    fn chat_response_format_becomes_responses_text_format() {
        let v: Value = serde_json::from_slice(&request(
            Endpoint::ChatCompletions,
            Endpoint::Responses,
            &serde_json::to_vec(&chat(json!({"response_format": {
                "type": "json_schema",
                "json_schema": {"name": "a", "strict": true, "schema": schema()}
            }})))
            .unwrap(),
            "gpt-5",
        ))
        .unwrap();
        assert_eq!(
            v["text"]["format"],
            json!({"type": "json_schema", "name": "a", "strict": true, "schema": schema()})
        );
    }

    // ---- Forced tool use on models that reject it ----

    fn tools_req(choice: Value) -> Value {
        chat(json!({
            "tool_choice": choice,
            "tools": [
                {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}}},
                {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}}
            ]
        }))
    }

    #[test]
    fn required_is_forced_where_the_model_allows_it() {
        let v = to_claude(&tools_req(json!("required")), "claude-opus-4-8");
        assert_eq!(v["tool_choice"], json!({"type": "any"}));
        assert!(
            v["messages"]
                .as_array()
                .unwrap()
                .iter()
                .all(|m| m["role"] != "system"),
            "{v}"
        );
    }

    /// Opus 5.5 / Sonnet 5.5 / Fable 5.1 400 on `any` and `tool`: `auto` plus a closing
    /// instruction, which is Anthropic's documented migration for these models.
    #[test]
    fn required_becomes_auto_plus_an_instruction_where_forcing_is_rejected() {
        for model in ["claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1"] {
            let v = to_claude(&tools_req(json!("required")), model);
            assert_eq!(v["tool_choice"], json!({"type": "auto"}), "{model}");
            let last = v["messages"].as_array().unwrap().last().unwrap().clone();
            assert_eq!(last["role"], "system", "{model}: {v}");
            assert!(last["content"].as_str().unwrap().contains("provided tools"));
        }
    }

    #[test]
    fn a_named_tool_choice_names_the_tool_in_the_instruction() {
        let v = to_claude(
            &tools_req(json!({"type": "function", "function": {"name": "get_time"}})),
            "claude-opus-5-5",
        );
        assert_eq!(v["tool_choice"], json!({"type": "auto"}));
        let last = v["messages"].as_array().unwrap().last().unwrap().clone();
        assert_eq!(last["role"], "system");
        assert!(
            last["content"].as_str().unwrap().contains("`get_time`"),
            "{last}"
        );
    }

    #[test]
    fn unforced_choice_keeps_disable_parallel_tool_use() {
        let mut req = tools_req(json!("required"));
        req["parallel_tool_calls"] = json!(false);
        let v = to_claude(&req, "claude-opus-5-5");
        assert_eq!(
            v["tool_choice"],
            json!({"type": "auto", "disable_parallel_tool_use": true})
        );
    }

    /// The instruction goes after the cache marker, so it never invalidates the cached prefix.
    #[test]
    fn the_instruction_follows_the_conversation_cache_marker() {
        let mut req = tools_req(json!("required"));
        req["messages"] = json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "weather?"}
        ]);
        let v = to_claude(&req, "claude-opus-5-5");
        let msgs = v["messages"].as_array().unwrap();
        assert_eq!(msgs.last().unwrap()["role"], "system");
        assert!(msgs.last().unwrap().get("cache_control").is_none());
        let user = &msgs[msgs.len() - 2];
        assert_eq!(
            user["content"][0]["cache_control"]["type"], "ephemeral",
            "{v}"
        );
    }

    #[test]
    fn auto_and_none_are_untouched_everywhere() {
        for choice in ["auto", "none"] {
            let v = to_claude(&tools_req(json!(choice)), "claude-opus-5-5");
            assert_eq!(v["tool_choice"]["type"], choice);
            assert!(
                v["messages"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|m| m["role"] != "system")
            );
        }
    }
}

#[cfg(test)]
#[path = "translate_request_tests.rs"]
mod request_tests;
