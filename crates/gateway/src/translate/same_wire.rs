//! Byte-span edits on a body that is not translated: the same wire is relayed, with only the spans these functions name changed.

use super::*;

/// A same-wire Chat Completions body relayed to a Claude model on a Chat host, less its reasoning
/// controls when its tool turn has no thinking to replay (see
/// [`drop_unreplayable_chat_reasoning`]), and less its assistant messages' thinking when it asks
/// for none (see [`drop_chat_reasoning_when_off`]). Both rules read only keys that open with
/// `"reasoning` or `"thinking` (`reasoning_effort`, `reasoning_details`, a `thinking` list) and
/// `include_reasoning`, so a body that spells none of them as a quoted key prefix is returned
/// untouched and unparsed: prose that says "reasoning" never does, since a quote inside a JSON
/// string is escaped. `upstream_model` is the id the candidate will receive.
pub fn claude_chat_relay_reasoning(body: Vec<u8>, upstream_model: &str) -> Vec<u8> {
    let says = |needle: &[u8]| memchr::memmem::find(&body, needle).is_some();
    if !(says(b"\"reasoning") || says(b"\"thinking") || says(b"\"include_reasoning\"")) {
        return body;
    }
    let Ok(mut v) = serde_json::from_slice::<Value>(&body) else {
        return body;
    };
    let claude = ClaudeModel::of(upstream_model);
    // Both rules, always (`|`, not `||`).
    if v.as_object_mut().is_some_and(|o| {
        drop_unreplayable_chat_reasoning(o) | drop_chat_reasoning_when_off(o, claude)
    }) {
        encode(&v)
    } else {
        body
    }
}

/// Root-level Responses session field that cannot be honored off `/v1/responses`.
///
/// `None` means the body is a one-shot: `store: false`, no `previous_response_id` and no
/// `conversation` (the Conversations API: the request continues a conversation whose items OpenAI
/// holds, the same state `previous_response_id` points into, so it is relayed or refused the same
/// way, never translated with the history dropped). An omitted
/// (or `null`) `store` is OpenAI's default `true`, so it is session state only where the row has a
/// Responses arm to keep it (`responses_arm`); elsewhere it is the stock `responses.create()` call,
/// which translates rather than 400s. An `item_reference` input item that stands for an earlier
/// turn is session state too ([`turn_reference_in_input`]): a pointer into a store that a
/// row with no Responses arm cannot resolve (the gateway stores no customer content). One inside a
/// tool step is not: it is dropped. A body that is not one well-formed object is `Some("store")`
/// so a catalog walk fail-closes onto a real Responses arm rather than silently stripping session
/// state.
///
/// Read by span, never into a `Value`: this runs on every Responses body before the walk,
/// up to `MAX_REQUEST_BODY`, and a `Value` costs 10-250 bytes of heap per byte of JSON. Each field
/// is the last of its key, as a provider's parser keeps it.
pub fn responses_session_field(body: &[u8], responses_arm: bool) -> Option<&'static str> {
    const FAIL_CLOSED: Option<&str> = Some("store");
    let Some(open) = peek::root_open(body) else {
        return FAIL_CLOSED;
    };
    let (mut previous, mut conversation, mut store, mut input) = (None, None, None, None);
    let mut members = peek::members(body, open);
    for m in members.by_ref() {
        let Some(m) = m else {
            return FAIL_CLOSED;
        };
        if m.key_is(body, "previous_response_id") {
            previous = Some(m.value);
        } else if m.key_is(body, "conversation") {
            conversation = Some(m.value);
        } else if m.key_is(body, "store") {
            store = Some(m.value);
        } else if m.key_is(body, "input") {
            input = Some(m.value);
        }
    }
    let trailing = members.end().and_then(|end| body.get(end..));
    if !trailing.is_some_and(|t| t.iter().all(u8::is_ascii_whitespace)) {
        return FAIL_CLOSED;
    }
    let raw = |span: (usize, usize)| body.get(span.0..span.1).unwrap_or_default();
    // An id string or `{"id": …}`; an empty id or `null` is not set.
    let set =
        |span: Option<(usize, usize)>| span.is_some_and(|s| !matches!(raw(s), b"null" | b"\"\""));
    if set(previous) {
        return Some("previous_response_id");
    }
    if set(conversation) {
        return Some("conversation");
    }
    match input.map(|span| turn_reference_in_input(body, span)) {
        Some(None) => return FAIL_CLOSED,
        Some(Some(true)) => return Some("item_reference"),
        _ => {}
    }
    match store.map(raw) {
        Some(b"false") => None,
        Some(b"null") | None => responses_arm.then_some("store"),
        _ => Some("store"),
    }
}

/// Whether `input` carries an `item_reference` that stands for an earlier conversation turn:
/// the one kind a row with no Responses arm refuses rather than drops.
///
/// The Vercel AI SDK's default Responses model sends each earlier assistant text or reasoning item
/// that has an id as `{"type":"item_reference","id":…}`, but each client-executed tool call in
/// full (`function_call`, then its `function_call_output`). So `input` reads as model steps: a
/// **step** is a maximal run of `item_reference`, `reasoning`, `function_call` and
/// `custom_tool_call` items, ended by any other item (a message, a tool output) or by the end of
/// `input`. A reference in a step that holds a call is preamble text or reasoning of the step that
/// made the call, wherever in the run it sits: the call and its output say what that step did, so
/// the reference is dropped (translation skips it, [`strip_item_references`] cuts it from a
/// Responses relay). A reference in a step without a call is an earlier answer, a turn the model
/// would answer without: `true`, refused. `span` is `input`'s value; `None` when it is a malformed
/// array. A `memmem` keeps an `input` that never says `item_reference` unwalked; otherwise one pass
/// over its items by span, no allocation.
fn turn_reference_in_input(body: &[u8], span: (usize, usize)) -> Option<bool> {
    let region = body.get(span.0..span.1).unwrap_or_default();
    if region.first() != Some(&b'[') || memchr::memmem::find(region, b"item_reference").is_none() {
        return Some(false);
    }
    let (mut refs, mut calls) = (false, false);
    for item in peek::elements(body, span.0) {
        let (start, _) = item?;
        match item_type(body, start)? {
            ItemKind::Reference => refs = true,
            ItemKind::Call => calls = true,
            ItemKind::Reasoning => {}
            ItemKind::Other => {
                if refs && !calls {
                    return Some(true);
                }
                (refs, calls) = (false, false);
            }
        }
    }
    Some(refs && !calls)
}

/// What a Responses `input` item is, for [`turn_reference_in_input`].
enum ItemKind {
    Reference,
    Call,
    Reasoning,
    Other,
}

/// The kind of the `input` item starting at `start`, from its last `type` member, read by span.
/// `None` when the item is a malformed object.
fn item_type(body: &[u8], start: usize) -> Option<ItemKind> {
    if body.get(start) != Some(&b'{') {
        return Some(ItemKind::Other);
    }
    let Some(typ) = peek::last_member(body, start, "type")? else {
        return Some(ItemKind::Other);
    };
    let is = |want: &str| peek::str_is(body, typ.value, want);
    Some(if is("item_reference") {
        ItemKind::Reference
    } else if is("function_call") || is("custom_tool_call") {
        ItemKind::Call
    } else if is("reasoning") {
        ItemKind::Reasoning
    } else {
        ItemKind::Other
    })
}

/// The 400 for a Responses session field a row cannot honor. An `item_reference` also says how to
/// send the item instead: the gateway keeps no copy to resolve it from.
pub fn session_field_refusal(field: &str, model: &str) -> String {
    if field == "item_reference" {
        format!(
            "an item_reference to an earlier turn cannot be honored for {model}: no upstream on \
             this route keeps responses, and the gateway stores none. Send earlier items in full \
             by setting store: false (Vercel AI SDK: providerOptions.openai.store = false)"
        )
    } else {
        format!("{field} cannot be honored for {model} (no Responses upstream)")
    }
}

/// Leads the `encrypted_content` of every `reasoning` item the gateway mints (see
/// [`reasoning_item`]), ahead of the Anthropic signature. It marks the item as ours even when a
/// client replays it without its `id` (Codex, the Agents SDK), and no provider's own encrypted
/// reasoning can start with it: `:` is outside the base64 alphabet both use.
pub(super) const GATEWAY_SIGNATURE_PREFIX: &str = "rs_gw:";

/// Whether a Responses input item is a `reasoning` item this gateway minted: our `rs_gw…` id
/// (items minted before [`GATEWAY_SIGNATURE_PREFIX`]) or our prefix on its `encrypted_content`.
/// [`is_gateway_reasoning`] for the item object at `start`, read by span: an item can be most of a
/// 100 MiB body, and a `Value` of it would cost many times that.
fn is_gateway_reasoning_span(body: &[u8], start: usize) -> bool {
    let field = |key: &str| peek::last_member(body, start, key).flatten();
    let starts = |key: &str, prefix: &str| {
        field(key)
            .and_then(|m| peek::str_value(body, m.value))
            .is_some_and(|v| v.starts_with(prefix))
    };
    field("type").is_some_and(|m| peek::str_is(body, m.value, "reasoning"))
        && (starts("id", "rs_gw") || starts("encrypted_content", GATEWAY_SIGNATURE_PREFIX))
}

pub(super) fn is_gateway_reasoning(item: &Value) -> bool {
    item.get("type").and_then(Value::as_str) == Some("reasoning")
        && (item
            .get("id")
            .and_then(Value::as_str)
            .is_some_and(|id| id.starts_with("rs_gw"))
            || item
                .get("encrypted_content")
                .and_then(Value::as_str)
                .is_some_and(|c| c.starts_with(GATEWAY_SIGNATURE_PREFIX)))
}

/// A one-shot Responses body walked onto a Responses candidate of a row with no Responses arm
/// (every grok row reaches xAI over `/v1/responses`), as `store: false`. Such a row walks its
/// candidates only when the body carries no session state (`responses_session_field`: an explicit
/// `store: true` or a `previous_response_id` is a 400 there), so an omitted or null `store` is a
/// one-shot the client expects not to be kept; xAI stores every response for 30 days unless told
/// not to (https://docs.x.ai/developers/model-capabilities/text/comparison.md). Every root `store`
/// member is cut out by span and `"store":false` goes first; every other byte stays as sent. `true`
/// when the body changed; a body already saying `store: false` once is untouched.
pub fn store_false(body: &mut Vec<u8>) -> bool {
    let Some(members) = peek::root_members(body) else {
        return false;
    };
    let stores: Vec<_> = members.iter().filter(|m| m.key_is(body, "store")).collect();
    if let [only] = stores.as_slice()
        && &body[only.value.0..only.value.1] == b"false"
    {
        return false;
    }
    peek::set_root_member(body, "store", b"false")
}

/// Cut every `item_reference` out of a Responses body walked onto a Responses candidate of a row
/// with no Responses arm (xAI's, as `store: false`): nothing there holds the item (xAI answers one
/// 422 "unknown item type"). Such a row walks its candidates only when every reference is a tool
/// step's (`responses_session_field`: one for an earlier turn is a 400 there), so this drops what
/// translation drops for any other candidate. Every other byte stays as sent. `true` when
/// the body changed; a body without the string is untouched and unparsed.
pub fn strip_item_references(body: &mut Vec<u8>) -> bool {
    const NEEDLE: &[u8] = b"item_reference";
    if memchr::memmem::find(body, NEEDLE).is_none() {
        return false;
    }
    // The last `input` is the one a provider's parser keeps.
    let Some(input) = peek::root_members(body)
        .and_then(|m| m.into_iter().rev().find(|m| m.key_is(body, "input")))
    else {
        return false;
    };
    if body.get(input.value.0) != Some(&b'[') {
        return false;
    }
    let Some(items) = peek::array_elements(body, input.value.0) else {
        return false;
    };
    // Each item's `type` by span: one item can be most of a 100 MiB body.
    let refs: Vec<bool> = items
        .iter()
        .map(|&(s, e)| {
            memchr::memmem::find(body.get(s..e).unwrap_or_default(), NEEDLE).is_some()
                && matches!(item_type(body, s), Some(ItemKind::Reference))
        })
        .collect();
    peek::remove_items(body, &items, |k| refs.get(k).copied().unwrap_or(false))
}

/// A Messages body for a candidate that does not accept fine-grained tool input streaming
/// (`providers::catalog::serves_eager_input_streaming`: Bedrock's older serving stack 400s
/// the field, and a 400 is not failed over): every `eager_input_streaming` member of each root
/// `tools` element is cut out by span. The tools then stream the API's default way, each parameter
/// buffered and validated before it is sent. Every other byte stays as sent; one `memmem` keeps a
/// body without the field unwalked. `true` when the body changed.
pub fn strip_eager_input_streaming(body: &mut Vec<u8>) -> bool {
    if memchr::memmem::find(body, b"eager_input_streaming").is_none() {
        return false;
    }
    // Every root `tools`: a parser may keep either one, and neither may carry the field.
    let Some(members) = peek::root_members(body) else {
        return false;
    };
    let opens: Vec<usize> = members
        .iter()
        .filter(|m| m.key_is(body, "tools") && body.get(m.value.0) == Some(&b'['))
        .map(|m| m.value.0)
        .collect();
    // Last first: a cut never moves an earlier array.
    opens.into_iter().rev().fold(false, |changed, open| {
        peek::remove_element_members(body, open, "eager_input_streaming") || changed
    })
}

/// A body for a candidate that answers only streams (`providers::catalog::stream_only`),
/// asking for a stream when the client did not: every root `stream` member (and, with `usage`,
/// every `stream_options`) is cut out by span, and `"stream":true` (with
/// `"stream_options":{"include_usage":true}`, so a Chat Completions stream ends in its usage chunk)
/// goes first; every other byte stays as sent. `false`, untouched, when the body already streams
/// (its last `stream` is `true`, as the provider's parser reads it): the client then gets the
/// stream it asked for.
pub fn force_stream(body: &mut Vec<u8>, usage: bool) -> bool {
    let Some(members) = peek::root_members(body) else {
        return false;
    };
    if members
        .iter()
        .rev()
        .find(|m| m.key_is(body, "stream"))
        .is_some_and(|m| &body[m.value.0..m.value.1] == b"true")
    {
        return false;
    }
    // Each set reports its own change, and neither answer is dropped: the caller re-scans on it.
    let options =
        usage && peek::set_root_member(body, "stream_options", br#"{"include_usage":true}"#);
    peek::set_root_member(body, "stream", b"true") || options
}

/// A Chat Completions body for a host that does not read the `developer` role
/// (`providers::catalog::reads_developer_role`): each root `messages` element whose `role` is
/// `"developer"` says `"system"` instead, which is what `developer` means for a model that is not
/// OpenAI's. Only those role values change; every other byte stays as sent. One `memmem` keeps a
/// body without `developer` unwalked. `true` when the body changed.
pub fn developer_as_system(body: &mut Vec<u8>) -> bool {
    if memchr::memmem::find(body, b"developer").is_none() {
        return false;
    }
    let Some(messages) = peek::root_members(body)
        .and_then(|m| m.into_iter().rev().find(|m| m.key_is(body, "messages")))
    else {
        return false;
    };
    if body.get(messages.value.0) != Some(&b'[') {
        return false;
    }
    let Some(items) = peek::array_elements(body, messages.value.0) else {
        return false;
    };
    let roles: Vec<(usize, usize)> = items
        .iter()
        .filter_map(|&(start, _)| {
            let role = peek::last_member(body, start, "role").flatten()?;
            (body.get(role.value.0..role.value.1) == Some(b"\"developer\"")).then_some(role.value)
        })
        .collect();
    if roles.is_empty() {
        return false;
    }
    // One pass: a splice per role moved the whole tail each time.
    let mut out = Vec::with_capacity(body.len());
    let mut at = 0;
    for &(start, end) in &roles {
        out.extend_from_slice(body.get(at..start).unwrap_or_default());
        out.extend_from_slice(b"\"system\"");
        at = end;
    }
    out.extend_from_slice(body.get(at..).unwrap_or_default());
    *body = out;
    true
}

/// A Chat Completions body for a candidate whose thinking conflicts with tools
/// (`providers::catalog::ToolThinking`), when it offers tools and the rule says so.
/// Thinking off: every root `reasoning` and `reasoning_effort` member is cut out by span and
/// `"reasoning":{"enabled":false}` goes first. A forced `tool_choice` (`"required"` or a named
/// function) turns thinking off under `OffWhenForced` and `OffWithTools`, since the candidate
/// refuses it while thinking; under `OffWithTools` any tool request does, unless the client asked
/// for reasoning itself. Under `ForcedAsInstruction` (the candidate cannot think off) a forced
/// request goes as `"tool_choice":"auto"` with a closing system message naming what must be called
/// instead. Under `AutoOnly` a forced request goes that way too, and a `"tool_choice":"none"`
/// request goes with its root `tools` and `tool_choice` cut out by span. Every other byte
/// stays as sent. `true` when the body changed.
pub fn fit_tools_to_thinking(body: &mut Vec<u8>, rule: providers::catalog::ToolThinking) -> bool {
    use providers::catalog::ToolThinking;
    if rule == ToolThinking::Free || memchr::memmem::find(body, b"\"tools\"").is_none() {
        return false;
    }
    let Some(members) = peek::root_members(body) else {
        return false;
    };
    let last = |key: &str| members.iter().rev().find(|m| m.key_is(body, key));
    let offers_tools = last("tools").is_some_and(|m| {
        body[m.value.0] == b'['
            && peek::array_elements(body, m.value.0).is_some_and(|items| !items.is_empty())
    });
    if !offers_tools {
        return false;
    }
    let choice = last("tool_choice").map(|m| &body[m.value.0..m.value.1]);
    let forced = choice.is_some_and(|v| v == b"\"required\"" || v.first() == Some(&b'{'));
    if rule == ToolThinking::AutoOnly && choice == Some(&b"\"none\""[..]) {
        let tools = peek::remove_root_members(body, "tools");
        return peek::remove_root_members(body, "tool_choice") || tools;
    }
    if matches!(
        rule,
        ToolThinking::ForcedAsInstruction | ToolThinking::AutoOnly
    ) {
        let Some(choice) = choice.filter(|_| forced) else {
            return false;
        };
        let must = match serde_json::from_slice::<Value>(choice) {
            Ok(Value::Object(o)) => match o.get("function").and_then(|f| f.get("name")) {
                Some(Value::String(n)) => format!("Respond by calling the `{n}` tool."),
                _ => return false,
            },
            _ => "Respond by calling one of the provided tools.".to_owned(),
        };
        return forced_as_instruction(body, &must);
    }
    let asked_reasoning = ["reasoning", "reasoning_effort"]
        .iter()
        .any(|k| last(k).is_some_and(|m| &body[m.value.0..m.value.1] != b"null"));
    let off = forced || (rule == ToolThinking::OffWithTools && !asked_reasoning);
    if !off {
        return false;
    }
    peek::remove_root_members(body, "reasoning_effort");
    peek::set_root_member(body, "reasoning", br#"{"enabled":false}"#)
}

/// `"tool_choice":"auto"` first, and `{"role":"system","content":must}` closing the root
/// `messages`. Untouched, `false`, when `messages` is not an array.
fn forced_as_instruction(body: &mut Vec<u8>, must: &str) -> bool {
    let close = peek::root_members(body)
        .and_then(|m| m.into_iter().rev().find(|m| m.key_is(body, "messages")))
        .filter(|m| body.get(m.value.0) == Some(&b'['))
        .and_then(|m| {
            Some((
                m.value.1.checked_sub(1)?,
                peek::array_elements(body, m.value.0)?,
            ))
        });
    let Some((close, items)) = close else {
        return false;
    };
    let mut msg = if items.is_empty() {
        Vec::new()
    } else {
        b",".to_vec()
    };
    msg.extend_from_slice(
        json!({"role": "system", "content": must})
            .to_string()
            .as_bytes(),
    );
    body.splice(close..close, msg);
    // The splice above changed the body, whatever the set reports.
    peek::set_root_member(body, "tool_choice", b"\"auto\"");
    true
}

/// A same-wire Responses body, minus the `reasoning` items this gateway minted from Claude's
/// thinking. A catalog walk relays the body byte for byte to an OpenAI Responses upstream (a
/// Responses-first row, a GPT row's Responses arm, a mixed-row failover), where a foreign id and an
/// Anthropic signature as `encrypted_content` mean nothing and are rejected. One `memmem` decides:
/// a body without `rs_gw` is returned untouched, unparsed.
///
/// The items are cut out by span ([`peek::remove_items`]); every other byte stays as the client
/// sent it. A `Value` round-trip would sort every object's keys (a strict structured output's
/// schema property order, and with it the prompt-cache prefix) and respace the body. Only an
/// `input` element that mentions `rs_gw` is parsed, to check it is ours.
pub fn strip_gateway_reasoning(mut body: Vec<u8>) -> Vec<u8> {
    if memchr::memmem::find(&body, b"rs_gw").is_none() {
        return body;
    }
    // The last `input` is the one a provider's parser keeps.
    let Some(input) = peek::root_members(&body)
        .and_then(|m| m.into_iter().rev().find(|m| m.key_is(&body, "input")))
    else {
        return body;
    };
    if body.get(input.value.0) != Some(&b'[') {
        return body;
    }
    let Some(items) = peek::array_elements(&body, input.value.0) else {
        return body;
    };
    let ours: Vec<bool> = items
        .iter()
        .map(|&(s, e)| {
            memchr::memmem::find(body.get(s..e).unwrap_or_default(), b"rs_gw").is_some()
                && is_gateway_reasoning_span(&body, s)
        })
        .collect();
    peek::remove_items(&mut body, &items, |k| ours.get(k).copied().unwrap_or(false));
    body
}
