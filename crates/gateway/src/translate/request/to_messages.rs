//! Request: Chat Completions → Messages.

use crate::translate::*;

/// The mapped body, and whether the gateway added cache breakpoints (see
/// `auto_cache_breakpoints`).
pub(in crate::translate) fn openai_req_to_anthropic(
    v: &Value,
    claude: ClaudeModel,
) -> (Value, bool) {
    let mut out = Map::new();
    copy_if(&mut out, v, "model");
    let max_tokens = max_tokens_of(v).unwrap_or(DEFAULT_MAX_TOKENS);
    out.insert("max_tokens".into(), json!(max_tokens));
    let thinking_off = openai_reasoning_to_anthropic(v, &mut out, claude, max_tokens);
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
    // `prompt` is a Responses client's stored template, carried here by the intermediate Chat
    // body of a Responses → Messages walk.
    for key in ["audio", "web_search_options", "top_logprobs", "prompt"] {
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
                // A Responses input item with no Chat Completions shape (see
                // `responses_input_to_messages`): forwarded whole, never merged into a user turn
                // it would vanish from, so the provider rejects it by name.
                //
                // So is a message whose role no dialect has (a typo, a framework's private role),
                // role unchanged: the provider's 400 names it, as it does called directly.
                _ if !matches!(role, "user" | "") || m.get("role").is_none() => {
                    in_conversation = true;
                    flush_tool_results(&mut messages, &mut pending_tool_results);
                    messages.push(m.clone());
                }
                _ => {
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
    // Budget thinking in a tool loop: the final assistant turn must open with its thinking block.
    // A client that did not send the signed block back (most Chat Completions and Responses SDKs)
    // leaves nothing to replay, and Anthropic accepts the request without thinking, not with it.
    if claude.reasoning == ClaudeGen::Budget
        && out
            .get("thinking")
            .and_then(|t| t.get("type"))
            .and_then(Value::as_str)
            == Some("enabled")
        && tool_turn_lacks_thinking(&messages)
    {
        out.remove("thinking");
    }
    drop_thinking_when_off(&out, &mut messages, claude);
    if !system_parts.is_empty() {
        out.insert("system".into(), anthropic_system_value(system_parts));
    }
    out.insert("messages".into(), Value::Array(messages));
    let gateway_cache = auto_cache_breakpoints(&mut out);
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
    if thinking_off && claude.between_tools {
        thinking_off_between_tools(&mut out, claude);
    }
    if claude.binding_controls {
        bind_thinking_with_drop_block(&mut out);
    }
    (Value::Object(out), gateway_cache)
}

/// "No reasoning" on a model whose lowest setting is `between_tools` (Sonnet 5.5): it rejects
/// `thinking: disabled`, and an omitted `thinking` is adaptive thinking, so `reasoning_effort:
/// "none"` would think anyway. `between_tools` does no extended thinking (the model's short notes
/// between tool calls still come back as thinking blocks) and takes effort `high` or below — the
/// effort here is `low`.
///
/// Except when the history holds thinking and the request will carry `block_binding`: the API
/// rejects `block_binding` alongside `between_tools`, and without it a replayed block whose prefix
/// the gateway changed is a 400 on enforced accounts (see `bind_thinking_with_drop_block`). Keeping
/// the conversation answerable outranks skipping the thinking, so that request stays adaptive at
/// `low`.
pub(in crate::translate) fn thinking_off_between_tools(
    out: &mut Map<String, Value>,
    claude: ClaudeModel,
) {
    if out.contains_key("thinking") {
        return;
    }
    let history_thinks = out
        .get("messages")
        .and_then(Value::as_array)
        .is_some_and(|ms| {
            ms.iter().any(|m| {
                m.get("content")
                    .and_then(Value::as_array)
                    .is_some_and(|bs| bs.iter().any(is_thinking_block))
            })
        });
    if claude.binding_controls && history_thinks {
        return;
    }
    out.insert("thinking".into(), json!({ "type": "between_tools" }));
}

/// Root Chat Completions keys that turn reasoning on for a Claude model a Chat host (OpenRouter)
/// serves: OpenAI's `reasoning_effort`, OpenRouter's `reasoning` and legacy `include_reasoning`,
/// and an Anthropic-style `thinking` some clients pass through.
pub(in crate::translate) const CHAT_REASONING_KEYS: [&str; 4] = [
    "reasoning_effort",
    "reasoning",
    "include_reasoning",
    "thinking",
];

/// [`drop_unreplayable_chat_reasoning`] and [`drop_chat_reasoning_when_off`] on a translated
/// Chat Completions body bound for a Claude model.
pub(in crate::translate) fn claude_chat_reasoning_guard(mut body: Value, up: Upstream) -> Value {
    if up.claude_thinking
        && let Some(out) = body.as_object_mut()
    {
        drop_unreplayable_chat_reasoning(out);
        drop_chat_reasoning_when_off(out, up.claude);
    }
    body
}

/// Chat Completions side: a body that asks a Claude model for no reasoning (none of
/// [`CHAT_REASONING_KEYS`] on, after [`drop_unreplayable_chat_reasoning`]) loses the thinking its
/// assistant messages carry (`reasoning_details`, and the gateway's `thinking` array). Anthropic on
/// Bedrock, which serves Claude behind OpenRouter, rejects thinking in the final turn ("When
/// thinking is disabled, an `assistant` message in the final position cannot contain `thinking`"),
/// and counts every call of a tool loop as that turn. A request that does not think has no use for
/// any of it, and Anthropic ignores it, so it goes from every assistant message rather than from a
/// turn boundary the gateway would have to guess. Not on an adaptive-generation model, which thinks
/// adaptively with no reasoning asked (see [`drop_thinking_when_off`]). `true` when one was removed.
pub(in crate::translate) fn drop_chat_reasoning_when_off(
    out: &mut Map<String, Value>,
    claude: ClaudeModel,
) -> bool {
    if claude.reasoning == ClaudeGen::Adaptive
        || CHAT_REASONING_KEYS
            .iter()
            .any(|k| out.get(*k).is_some_and(reasoning_value_on))
    {
        return false;
    }
    let Some(messages) = out.get_mut("messages").and_then(Value::as_array_mut) else {
        return false;
    };
    let mut removed = false;
    for m in messages {
        if m.get("role").and_then(Value::as_str) == Some("assistant")
            && let Some(m) = m.as_object_mut()
        {
            removed |= m.remove("reasoning_details").is_some();
            removed |= m.remove("thinking").is_some();
        }
    }
    removed
}

/// Messages side (see [`drop_chat_reasoning_when_off`]): when the request does not
/// think, assistant messages lose their thinking blocks, unless they are all one holds (an empty
/// message is a 400 of its own). "Does not think" is `thinking: disabled`, or no `thinking` on a
/// model where that means none (an adaptive-generation model thinks adaptively without one).
fn drop_thinking_when_off(out: &Map<String, Value>, messages: &mut [Value], claude: ClaudeModel) {
    let off = match out.get("thinking") {
        Some(t) => t.get("type").and_then(Value::as_str) == Some("disabled"),
        None => claude.reasoning != ClaudeGen::Adaptive,
    };
    if !off {
        return;
    }
    for m in messages {
        if m.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        if let Some(blocks) = m.get_mut("content").and_then(Value::as_array_mut)
            && blocks.iter().any(|b| !is_thinking_block(b))
        {
            blocks.retain(|b| !is_thinking_block(b));
        }
    }
}

/// Anthropic's signed-thinking rule for a Claude model behind Chat Completions (OpenRouter's `anthropic/…`, same wire or
/// translated): OpenRouter sends the request's reasoning to Anthropic as enabled thinking, and
/// Anthropic 400s a tool turn whose last assistant message does not open on a signed thinking
/// block ("a final `assistant` message must start with a thinking block"). OpenRouter replays a
/// block only from `reasoning_details`; a client that sends none back for that turn (pi, the stock
/// OpenAI SDK, LangChain: a plain `reasoning` string at most) gets the request without reasoning,
/// which Anthropic accepts. `true` when a key was removed.
pub(in crate::translate) fn drop_unreplayable_chat_reasoning(out: &mut Map<String, Value>) -> bool {
    if !CHAT_REASONING_KEYS
        .iter()
        .any(|k| out.get(*k).is_some_and(reasoning_value_on))
    {
        return false;
    }
    let lacks = out
        .get("messages")
        .and_then(Value::as_array)
        .is_some_and(|ms| chat_tool_turn_lacks_reasoning(ms));
    if !lacks {
        return false;
    }
    for k in CHAT_REASONING_KEYS {
        out.remove(k);
    }
    true
}

/// Whether a reasoning control's value asks for reasoning: not `null`, `false`, `"none"`,
/// `{"effort": "none"}`, `{"enabled": false}` or `{"type": "disabled"}`.
pub(in crate::translate) fn reasoning_value_on(v: &Value) -> bool {
    match v {
        Value::Null | Value::Bool(false) => false,
        Value::String(s) => s != "none",
        Value::Object(o) => {
            o.get("effort").and_then(Value::as_str) != Some("none")
                && o.get("enabled").and_then(Value::as_bool) != Some(false)
                && o.get("type").and_then(Value::as_str) != Some("disabled")
        }
        _ => true,
    }
}

/// Whether the last assistant message of a Chat Completions body calls a tool without any thinking
/// OpenRouter can replay (a signed `reasoning.text` or an encrypted entry in `reasoning_details`).
fn chat_tool_turn_lacks_reasoning(messages: &[Value]) -> bool {
    let Some(m) = messages
        .iter()
        .rev()
        .find(|m| m.get("role").and_then(Value::as_str) == Some("assistant"))
    else {
        return false;
    };
    if m.get("tool_calls")
        .and_then(Value::as_array)
        .is_none_or(Vec::is_empty)
    {
        return false;
    }
    let mut blocks = Vec::new();
    if let Some(details) = m.get("reasoning_details").and_then(Value::as_array) {
        let mut gather = Gather::default();
        for d in details {
            gather.detail(d, &mut blocks);
        }
        gather.close(&mut blocks);
    }
    !blocks.iter().any(is_replayable_thinking)
}

/// Whether the last assistant turn calls a tool without opening on a thinking block (signed or
/// redacted: unsigned ones never reach here), which Anthropic rejects while thinking is enabled.
fn tool_turn_lacks_thinking(messages: &[Value]) -> bool {
    let Some(blocks) = messages
        .iter()
        .rev()
        .find(|m| m.get("role").and_then(Value::as_str) == Some("assistant"))
        .and_then(|m| m.get("content"))
        .and_then(Value::as_array)
    else {
        return false;
    };
    !blocks.first().is_some_and(is_thinking_block)
        && blocks
            .iter()
            .any(|b| b.get("type").and_then(Value::as_str) == Some("tool_use"))
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

/// The `anthropic-beta` value Anthropic fast mode needs beside `speed: "fast"`
/// (platform.claude.com/docs/en/build-with-claude/fast-mode). A managed client may send it to
/// direct Anthropic itself; `proxy` adds it to a translated attempt that [`request_with_tools`]
/// asks for fast mode.
pub const FAST_MODE_BETA: &str = "fast-mode-2026-02-01";

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
pub(in crate::translate) fn bind_thinking_with_drop_block(out: &mut Map<String, Value>) {
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
pub(in crate::translate) fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Chat Completions' pre-`tools` function calling (`functions`, `function_call`, an assistant
/// `function_call`, `role: "function"` results) rewritten as the `tools` loop it is equivalent to,
/// so every mapping below sees one shape. `None` when the body uses none of it. A legacy call has
/// no id; it gets one derived from its message position, which the `function` result that answers
/// it (the next one) reuses, and which is the same on every turn.
pub(in crate::translate) fn legacy_functions_to_tools(v: &Value) -> Option<Value> {
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
///
/// Returns whether it added any marker: the writes they cause are then the gateway's cost.
pub(in crate::translate) fn auto_cache_breakpoints(out: &mut Map<String, Value>) -> bool {
    if has_cache_control(out) {
        return false;
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
        marked_prefix = true;
    }
    let Some(messages) = out.get_mut("messages").and_then(Value::as_array_mut) else {
        return marked_prefix;
    };
    if !messages
        .iter()
        .any(|m| m.get("role").and_then(Value::as_str) == Some("assistant"))
    {
        return marked_prefix;
    }
    let Some(last) = messages.last_mut() else {
        return marked_prefix;
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
        .and_then(|blocks| blocks.iter_mut().rev().find(|b| !is_thinking_block(b)))
        .and_then(Value::as_object_mut)
    {
        block.insert("cache_control".into(), ephemeral());
        return true;
    }
    marked_prefix
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

/// Append `add` to `msg`'s content, in place. Each block moves once: a run of same-role messages
/// merges in time linear in its length (copying the gathered blocks on every merge made a long run
/// quadratic).
fn merge_content(msg: &mut Value, add: Value) {
    let Some(existing) = msg.get_mut("content") else {
        msg["content"] = add;
        return;
    };
    let text = |t: String| json!({ "type": "text", "text": t });
    match (existing, add) {
        (Value::Array(a), Value::Array(b)) => a.extend(b),
        (Value::Array(a), Value::String(b)) => a.push(text(b)),
        // Two messages stay two blocks: fused, "Hello" + "World" would read "HelloWorld".
        // An empty side adds nothing (Messages rejects an empty text block).
        (Value::String(_), Value::String(b)) if b.is_empty() => {}
        (existing @ Value::String(_), Value::String(b)) => {
            let Value::String(a) = std::mem::take(existing) else {
                return;
            };
            *existing = if a.is_empty() {
                Value::String(b)
            } else {
                Value::Array(vec![text(a), text(b)])
            };
        }
        (existing @ Value::String(_), Value::Array(b)) => {
            let Value::String(a) = std::mem::take(existing) else {
                return;
            };
            let mut n = Vec::with_capacity(b.len().saturating_add(1));
            n.push(text(a));
            n.extend(b);
            *existing = Value::Array(n);
        }
        (existing, add) => *existing = add,
    }
}

/// Set one field of the request's `output_config` (effort, structured-output format).
pub(in crate::translate) fn set_output_config(
    out: &mut Map<String, Value>,
    key: &str,
    value: Value,
) {
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

pub(in crate::translate) fn copy_cache_control(out: &mut Map<String, Value>, src: &Value) {
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

pub(in crate::translate) fn text_block(text: &str, cache_control: Option<&Value>) -> Value {
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
    let mut result = json!({
        "type": "tool_result",
        "tool_use_id": id,
        "content": content,
    });
    if let Some(obj) = result.as_object_mut() {
        copy_cache_control(obj, m);
    }
    Some(result)
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

/// A Chat Completions assistant turn as Messages content. Thinking crosses only as a block
/// Anthropic can verify — signed, or redacted — from our `thinking` array, OpenRouter's
/// `reasoning_details`, or thinking parts in `content`. Bare reasoning text (`reasoning_content`,
/// `reasoning`) is never signed, and an unsigned block is a 400 ("thinking.signature: Field
/// required") on every later turn, so it goes; Anthropic accepts a turn without its thinking.
fn openai_assistant_content(m: &Value) -> Value {
    let mut blocks: Vec<Value> = Vec::new();
    let mut thinking_parts = false;
    match m.get("content") {
        Some(Value::String(s)) if !s.is_empty() => {
            blocks.push(json!({ "type": "text", "text": s }));
        }
        Some(Value::Array(parts)) => {
            for p in parts {
                match p.get("type").and_then(Value::as_str) {
                    Some("thinking" | "redacted_thinking") => {
                        thinking_parts = true;
                        let b = anthropic_thinking_block(p);
                        if is_replayable_thinking(&b) {
                            blocks.push(b);
                        }
                    }
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
    if !thinking_parts {
        let signed = chat_thinking_blocks(m)
            .into_iter()
            .filter(is_replayable_thinking);
        blocks.splice(0..0, signed);
    }
    if let Some(calls) = m.get("tool_calls").and_then(Value::as_array) {
        for c in calls {
            let id = c.get("id").and_then(Value::as_str).unwrap_or("call_0");
            // A `custom` call's input is free-form text, and `tool_use.input` must be an object.
            let (func, input) = match c.get("custom") {
                Some(custom) => (
                    custom,
                    json!({ "input": custom.get("input").cloned().unwrap_or(json!("")) }),
                ),
                None => {
                    let func = c.get("function").unwrap_or(c);
                    (func, parse_arguments(func.get("arguments")))
                }
            };
            let name = func.get("name").and_then(Value::as_str).unwrap_or("");
            blocks.push(json!({
                "type": "tool_use",
                "id": id,
                "name": name,
                "input": input,
            }));
        }
    }
    // A marker on the whole message caches through its end: its last block (thinking blocks take
    // none, and only ever lead).
    if let Some(cc) = m.get("cache_control")
        && let Some(last) = blocks
            .last_mut()
            .filter(|b| !is_thinking_block(b))
            .and_then(Value::as_object_mut)
        && !last.contains_key("cache_control")
    {
        last.insert("cache_control".into(), cc.clone());
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
                        // `None` only for a URL `route::refused_input` refuses before any translation.
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
