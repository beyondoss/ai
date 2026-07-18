//! Anthropic Messages wire (`/v1/messages`).
//!
//! The harness's internal model was chosen to be Anthropic-shaped (content blocks,
//! `tool_use`/`tool_result`), so the request mapping is nearly an identity and the SSE decoder is a
//! direct translation of Anthropic's `content_block_*` / `message_*` events.

use std::sync::Arc;

use serde::Deserialize;
use serde_json::{Map, Value, json};

use super::StreamDecoder;
use crate::error::{Error, Result};
use crate::message::{StopReason, StreamEvent, TokenUsage, ToolDef};
use crate::transport::{ModelRequest, ToolChoice};

/// Anthropic's OAuth-gated (Claude Pro/Max subscription) endpoint expects to see exactly Claude Code's
/// own request shape, not just its headers (see `CLAUDE_CODE_BETA`'s doc comment in `client.rs` for why
/// presenting this tool as Claude Code is a deliberate, user-confirmed choice — applied here to the
/// body, not a separate decision). Two parts to that shape, both gated on the same `is_oauth` flag
/// threaded in from `client.rs`: the system prompt is prefixed with Claude Code's own identity sentence
/// ([`CLAUDE_CODE_IDENTITY`]), and every tool name — advertised in `tools`, replayed in assistant
/// history, and decoded back out of a live `tool_use` block — is round-tripped through Claude Code's
/// canonical casing. Mirrors pi's `buildParams`/`convertTools`/`convertMessages`/`fromClaudeCodeName`
/// (`anthropic-messages.ts:916-931, 949-955, 1111, 592-598`).
const CLAUDE_CODE_IDENTITY: &str = "You are Claude Code, Anthropic's official CLI for Claude.";

/// Claude Code's own canonical tool-name casing — pi's `claudeCodeTools` table
/// (`anthropic-messages.ts:72-91`; source: https://cchistory.mariozechner.at/data/prompts-2.1.11.md).
/// A short, hand-maintained list rather than a generated one: it only needs to match Claude Code's own
/// naming closely enough that Anthropic's OAuth-gated endpoint recognizes the request shape, not to
/// track Claude Code's tool set exactly — a name with no case-insensitive match here passes through
/// unchanged.
const CLAUDE_CODE_TOOLS: &[&str] = &[
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Grep",
    "Glob",
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "KillShell",
    "NotebookEdit",
    "Skill",
    "Task",
    "TaskOutput",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
];

/// Canonicalize `name` to Claude Code's own casing when it matches one of [`CLAUDE_CODE_TOOLS`]
/// case-insensitively, otherwise leave it untouched. Mirrors pi's `toClaudeCodeName`.
fn to_claude_code_name(name: &str) -> String {
    CLAUDE_CODE_TOOLS
        .iter()
        .find(|cc| cc.eq_ignore_ascii_case(name))
        .map(|cc| cc.to_string())
        .unwrap_or_else(|| name.to_string())
}

/// Reverse of [`to_claude_code_name`]: given a name the model just produced in a live `tool_use` block
/// (under OAuth, potentially Claude Code's canonical casing rather than ours), find the real advertised
/// tool whose name matches case-insensitively and use its real casing. Falls back to the incoming name
/// verbatim if nothing advertised matches (a tool this turn never offered — pass it through rather than
/// guess). Mirrors pi's `fromClaudeCodeName`.
fn from_claude_code_name(name: &str, tools: &[ToolDef]) -> String {
    tools
        .iter()
        .find(|t| t.name.eq_ignore_ascii_case(name))
        .map(|t| t.name.clone())
        .unwrap_or_else(|| name.to_string())
}

/// Build the streaming request body. `system` is hoisted to a top-level field (Anthropic keeps it
/// out of `messages`); `messages` and `tools` serialize straight from the internal model. `is_oauth`
/// selects Claude Code's own identity/tool-naming shape (see [`CLAUDE_CODE_IDENTITY`]'s doc comment) —
/// threaded in from `client.rs`, which already knows whether this credential is a genuine
/// direct-to-Anthropic OAuth login.
///
/// Three prompt-cache breakpoints are stamped in. An agent loop re-sends an ever-growing prefix —
/// tools, then system, then the whole prior conversation — on every turn; without caching each turn
/// re-bills that entire prefix at full input price, an O(n²) token cost over an n-step run.
/// Anthropic caches the request prefix up to each `cache_control` mark (reads cost ~10% of input
/// tokens), so we anchor one breakpoint on the fixed tool block, one on the system prompt (a stable
/// anchor that survives Anthropic's ~20-block breakpoint lookback on tool-heavy turns), and roll a
/// third onto the last message to capture the conversation so far. The TTL is 5 min, or 1 hour when
/// `cache_long` is set (see [`cache_control`]).
pub fn build_body(req: &ModelRequest, is_oauth: bool) -> Value {
    // `is_codex`/`is_azure` are both OpenAI-only route flags (always `false` here, since Anthropic
    // requests never carry either) — passed through anyway for the same reason every other dialect's
    // `build_body` does: `capabilities_for_route` is a complete no-op for any non-OpenAI id (see its
    // own `..._leaves_non_openai_ids_completely_unaffected` test), so this stays a plain
    // `capabilities(&req.model)` in every real case while keeping one call shape across all dialects.
    let caps = crate::models::capabilities_for_route(&req.model, req.is_codex, req.is_azure);
    let mut map = Map::new();
    map.insert("model".into(), Value::String(req.model.clone()));
    map.insert(
        "max_tokens".into(),
        Value::from(super::clamp_max_tokens_to_context(req, &caps)),
    );
    map.insert("stream".into(), Value::Bool(true));

    // The 1-hour TTL is only valid on models that support long cache retention; Anthropic 400s
    // otherwise. Gate the request's `cache_long` opt-in on the model's capability so an unsupported
    // model silently falls back to the standard 5-minute TTL instead of erroring the turn.
    let long = req.cache_long && caps.supports_long_cache;
    // `no_cache` skips every breakpoint below: a genuinely one-off request (no follow-up turn to read
    // the cache back) would otherwise eat the ~1.25x cache-write premium for an entry nothing reads.
    let cc = (!req.no_cache).then(|| cache_control(long));

    // Rolling breakpoint: cache the conversation prefix (tools + system + every prior message) up to
    // the final block, so next turn the whole accumulated transcript is a cache read, not a re-bill.
    //
    // Built field-by-field (`role`/`content` only) rather than `serde_json::to_value`-ing the whole
    // `Message` struct — `Message` also carries several internal-only provenance/accounting fields
    // (`model_id`/`error_message`/`aborted`/`usage`/`stop_reason`) that must never reach the wire:
    // Anthropic's schema is strict about unknown fields on a message object and 400s the entire
    // request if any leaks through. An earlier version of this dialect instead serialized the whole
    // `Message` and hand-stripped each internal field by name afterward (a `strip_internal_fields`
    // blocklist) — that mechanism already caused one live production Critical bug (two fields shipped
    // several passes before the strip list caught up to them) and has no way to catch a *future* field
    // addition the same way. Building the wire shape from exactly these two fields instead makes an
    // unlisted field structurally unreachable, matching the OpenAI/OpenAI-Responses dialects' own
    // field-by-field construction (see this module's own doc comment) rather than relying on a
    // hand-maintained blocklist staying in sync with `Message`'s field list forever. See
    // `build_body_wire_messages_never_carry_any_field_besides_role_and_content` for the canary test
    // this shape is designed to keep passing even if `Message` gains a field this dialect forgets to
    // account for.
    let mut messages = Value::Array(
        req.messages
            .iter()
            .map(|m| json!({ "role": m.role, "content": m.content }))
            .collect(),
    );
    // …and the same argument, one level down. The `role`/`content` shape above makes a stray *message*
    // field unreachable, but `content` is then serialized straight off `ContentBlock`, which carries
    // fields no Anthropic block has: `Text::id`/`Text::phase` (OpenAI Responses item ids) and
    // `ToolUse::thought_signature`. They are `None` on anything Anthropic itself produced — and are
    // populated the moment a transcript passes through another dialect, which a session outlives.
    // `run --model gpt-5 …` then `run --continue --model claude-…` 400s on
    // `messages.N.content.0.text.id: Extra inputs are not permitted`, and cross-provider model switching
    // is now an ordinary thing to do (`agent::gateway_credential`'s direct routes), not an exotic one.
    prune_non_anthropic_block_fields(&mut messages);
    normalize_cross_model_tool_ids(&mut messages, &req.messages, &req.model);
    downgrade_unsigned_thinking(&mut messages);
    // Vision is gated on *both* the model's own real support and the request's explicit
    // wire-level opt-out (`ModelRequest::block_images`) — an image is downgraded to a text
    // placeholder when either is unsupported/blocked, regardless of when/how it entered history.
    downgrade_unsupported_images(&mut messages, caps.supports_vision && !req.block_images);
    encode_tool_result_images(&mut messages);
    if is_oauth {
        // Replayed assistant tool_use blocks in history need the same canonical casing the live model
        // just saw advertised in `tools` below — the stored name is always our own real one (this
        // dialect's `Decoder` already reverses it back via `from_claude_code_name` at decode time), so
        // this is a pure forward rename, never conditional on which turn produced the block. Mirrors
        // pi's `convertMessages` (`anthropic-messages.ts:1111`).
        canonicalize_tool_use_names(&mut messages);
    }
    if let Some(cc) = &cc {
        mark_last_block(&mut messages, cc);
    }
    map.insert("messages".into(), messages);

    // System as a single cached text block — a *dedicated* third breakpoint. Anthropic's breakpoint
    // lookback only walks back ~20 content blocks; on a tool-heavy turn (N tool_use + N tool_result
    // blocks) the rolling message breakpoint can fall outside that window, so this stable anchor keeps
    // the (large, fixed) system prompt a cache read. `no_cache` drops the breakpoint but keeps the
    // system block itself (still needed on the wire either way).
    if is_oauth {
        // Anthropic's OAuth-gated endpoint requires this exact identity sentence as the *first* system
        // block — the real system prompt, if any, is appended as a second block, never substituted for
        // it. Both blocks get the same cache breakpoint pi stamps on each (`anthropic-messages.ts:
        // 916-931`), not just the last one.
        let mut system = vec![system_block(CLAUDE_CODE_IDENTITY, &cc)];
        if let Some(prompt) = &req.system {
            system.push(system_block(prompt, &cc));
        }
        map.insert("system".into(), Value::Array(system));
    } else if let Some(system) = &req.system {
        map.insert(
            "system".into(),
            Value::Array(vec![system_block(system, &cc)]),
        );
    }
    // Anthropic forbids `temperature` alongside extended thinking (thinking requires an implicit
    // temperature of 1) — matches pi's own `!options?.thinkingEnabled` gate (`anthropic-messages.ts`).
    // Separately, a handful of models (`claude-opus-4-7`/`claude-opus-4-8` — our own default model)
    // reject `temperature` outright regardless of thinking state (pi: `compat.supportsTemperature`) —
    // gated on the capability table rather than thinking state alone.
    if let (Some(temperature), None, true) =
        (req.temperature, &req.thinking, caps.supports_temperature)
    {
        map.insert("temperature".into(), json!(temperature));
    }
    if let Some(thinking) = &req.thinking {
        // Extended thinking. Anthropic requires `max_tokens > budget_tokens` and forbids `temperature`
        // alongside it. Newer models (the capability table's `Adaptive`
        // shape) take an effort-based shape instead of an explicit budget, with `output_config.effort`
        // as a *sibling top-level request field*, not nested under `thinking` — a request-shape detail
        // easy to get wrong. Both shapes explicitly set `display`: Anthropic's own API default for
        // `adaptive` is "omitted" (no visible reasoning text at all), so leaving it unset on an
        // adaptive model silently produces empty thinking output unless the caller explicitly opted
        // into `ThinkingDisplay::Omitted` themselves (pi: `thinkingDisplay: "omitted"`, for faster
        // time-to-first-text-token when the UI doesn't surface thinking).
        let display = thinking.display.as_str();
        match caps.thinking {
            crate::models::ThinkingShape::Adaptive => {
                map.insert(
                    "thinking".into(),
                    json!({ "type": "adaptive", "display": display }),
                );
                if let Some(effort) = req.reasoning_effort {
                    let wire = crate::models::anthropic_adaptive_effort_wire(&caps, effort);
                    map.insert("output_config".into(), json!({ "effort": wire }));
                }
            }
            _ => {
                map.insert(
                    "thinking".into(),
                    json!({
                        "type": "enabled",
                        "budget_tokens": thinking.budget_tokens,
                        "display": display,
                    }),
                );
            }
        }
    } else if caps.reasoning_disableable {
        // No thinking requested this turn, but the model can be told so explicitly rather than
        // relying on Anthropic's own undocumented default for whatever it does when the field is
        // omitted entirely.
        map.insert("thinking".into(), json!({ "type": "disabled" }));
    }
    if !req.tools.is_empty() {
        // Anchor breakpoint: the tool definitions (ten JSON schemas) are identical every turn and sit
        // at the front of the cache order, so this entry stays warm even when the rolling message
        // breakpoint is rewritten each turn. Requires stable tool ordering — see `definitions()`.
        let mut tools = serde_json::to_value(req.tools.as_ref()).unwrap_or(Value::Null);
        if is_oauth {
            canonicalize_tool_names(&mut tools);
        }
        if caps.supports_eager_tool_streaming {
            mark_eager_tool_streaming(&mut tools);
        }
        if let Some(cc) = &cc {
            if caps.supports_cache_control_on_tools {
                mark_last_tool(&mut tools, cc);
            }
        }
        map.insert("tools".into(), tools);
    }
    // Constrain tool use only when the caller asked: an unset `tool_choice` emits nothing, leaving
    // Anthropic's default (auto when tools are present), so the common request shape is untouched.
    if let Some(choice) = &req.tool_choice {
        map.insert("tool_choice".into(), tool_choice(choice));
    }
    // Anthropic-specific abuse-detection/rate-limiting hint — see `ModelRequest::user_id`'s doc
    // comment. Unset by default, matching pi's own `metadata` passthrough (never populated by its own
    // CLI, but available to a caller embedding the library).
    if let Some(user_id) = &req.user_id {
        map.insert("metadata".into(), json!({ "user_id": user_id }));
    }
    Value::Object(map)
}

/// Map a [`ToolChoice`] to Anthropic's `tool_choice` object. Anthropic spells "must call some tool"
/// as `any` and pins a specific tool with `{type:"tool", name}`.
fn tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!({ "type": "auto" }),
        ToolChoice::None => json!({ "type": "none" }),
        ToolChoice::Required => json!({ "type": "any" }),
        ToolChoice::Tool(name) => json!({ "type": "tool", "name": name }),
    }
}

/// The `cache_control` object to stamp on a breakpoint: ephemeral, with the 1-hour TTL when `long`.
fn cache_control(long: bool) -> Value {
    if long {
        json!({ "type": "ephemeral", "ttl": "1h" })
    } else {
        json!({ "type": "ephemeral" })
    }
}

/// A single Anthropic system-prompt text block, with `cache_control` stamped on when `cc` is set.
fn system_block(text: &str, cc: &Option<Value>) -> Value {
    match cc {
        Some(cc) => json!({ "type": "text", "text": text, "cache_control": cc }),
        None => json!({ "type": "text", "text": text }),
    }
}

/// Canonicalize every advertised tool's `name` to Claude Code's own casing — pi's `convertTools`
/// (`anthropic-messages.ts:949-955, 1200`). Only called when `is_oauth`.
fn canonicalize_tool_names(tools: &mut Value) {
    if let Some(list) = tools.as_array_mut() {
        for tool in list.iter_mut().filter_map(Value::as_object_mut) {
            if let Some(name) = tool.get("name").and_then(Value::as_str) {
                let canonical = to_claude_code_name(name);
                tool.insert("name".into(), Value::String(canonical));
            }
        }
    }
}

/// The keys each Anthropic content-block type is allowed to carry on the wire. Anything else is dropped
/// by [`prune_non_anthropic_block_fields`].
///
/// An **allowlist**, not a blocklist, and deliberately so — the same reasoning [`build_body`]'s
/// `role`/`content` construction spells out for message-level fields. A blocklist has to be updated
/// every time `ContentBlock` gains a field, and the one time it lags is a 400 on every request carrying
/// that field. An allowlist drops a new internal field by default: the failure mode of forgetting to
/// update it is "a field Anthropic wanted goes missing", which the dialect's own tests catch loudly,
/// rather than "a field Anthropic rejects gets sent", which only shows up as a live 400.
///
/// `cache_control` is absent on purpose: it is *added* by a later pass, after this one runs.
/// `tool_result.images` is present even though Anthropic has no such key — a later pass consumes it into
/// the real `content` array shape and removes it (see `attach_tool_result_images`), so it has to survive
/// this one.
const ANTHROPIC_BLOCK_KEYS: &[(&str, &[&str])] = &[
    ("text", &["type", "text"]),
    ("thinking", &["type", "thinking", "signature"]),
    ("redacted_thinking", &["type", "data"]),
    ("tool_use", &["type", "id", "name", "input"]),
    (
        "tool_result",
        &["type", "tool_use_id", "content", "is_error", "images"],
    ),
    ("image", &["type", "source"]),
];

/// Drop every content-block field Anthropic's schema doesn't accept — see [`ANTHROPIC_BLOCK_KEYS`].
///
/// The fields this actually removes today (`text.id`, `text.phase`, `tool_use.thought_signature`) are all
/// set by *other* dialects, and only ever reach here on a transcript that has crossed dialects: a session
/// started on an OpenAI model and continued on a Claude one. Anthropic rejects unknown block fields
/// outright (`Extra inputs are not permitted`), so this is a hard 400 on the whole request, not a quiet
/// degradation.
///
/// A block whose `type` isn't in the table is left untouched: this prunes known shapes, and a shape it
/// doesn't know about is not one it should be silently emptying.
fn prune_non_anthropic_block_fields(messages: &mut Value) {
    let Some(msgs) = messages.as_array_mut() else {
        return;
    };
    for m in msgs {
        let Some(content) = m.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        for block in content.iter_mut() {
            let Some(obj) = block.as_object_mut() else {
                continue;
            };
            let Some(ty) = obj.get("type").and_then(Value::as_str) else {
                continue;
            };
            let Some((_, allowed)) = ANTHROPIC_BLOCK_KEYS.iter().find(|(name, _)| *name == ty)
            else {
                continue;
            };
            obj.retain(|key, _| allowed.contains(&key.as_str()));
        }
    }
}

/// Canonicalize every `tool_use` block's `name` in the message history to Claude Code's own casing —
/// see [`build_body`]'s `is_oauth` branch for why this is always a forward rename.
fn canonicalize_tool_use_names(messages: &mut Value) {
    let Some(msgs) = messages.as_array_mut() else {
        return;
    };
    for m in msgs {
        let Some(content) = m.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        for block in content.iter_mut() {
            let Some(obj) = block.as_object_mut() else {
                continue;
            };
            if obj.get("type").and_then(Value::as_str) != Some("tool_use") {
                continue;
            }
            if let Some(name) = obj.get("name").and_then(Value::as_str) {
                let canonical = to_claude_code_name(name);
                obj.insert("name".into(), Value::String(canonical));
            }
        }
    }
}

/// Normalize a `tool_use.id` produced by a *different* model than `req.model` is about to see it, to
/// match Anthropic's required `^[a-zA-Z0-9_-]+$`, max-64-char shape — mirrors pi's
/// `normalizeToolCallId` (`anthropic-messages.ts:1006-1009`), applied by its `transformMessages`
/// whenever `!isSameModel`. A cross-model tool-call id can be non-conformant in two ways Anthropic
/// itself never produces: the OpenAI Responses API's combined `"call_id|item_id"` shape (already
/// truncated to `call_id` by `Session::scrub_cross_model_state` before a *persisted* session reaches
/// this point, but that scrub only ever runs at an explicit model-switch — this is the belt-and-
/// suspenders check at the point the id actually reaches the wire, covering every other path: a
/// same-turn multi-model fan-out, a hand-edited or externally-loaded session, a future caller that
/// builds a `ModelRequest` directly), or a non-standard OpenAI-compatible provider's own id shape
/// (GitHub Copilot's 450+-char blobs, arbitrary punctuation). `Message::model_id` is the same
/// per-message provenance field `scrub_cross_model_state` keys off; a message with `model_id ==
/// Some(&req.model)` (produced by the very model about to see it again) is left untouched — its ids
/// are guaranteed Anthropic-native already, and are also candidates a paired-later `ToolResult` might
/// still reference, so skipping them here doubles as skipping unnecessary remap bookkeeping. Every
/// rewrite is recorded and replayed onto the matching `tool_result` in a second pass, the same
/// two-pass shape `scrub_cross_model_state` already uses — a `ToolResult` block never carries
/// `model_id` itself, so its pairing can only be kept correct by propagating the assistant-side
/// rewrite forward.
fn normalize_cross_model_tool_ids(
    messages: &mut Value,
    typed: &[crate::message::Message],
    model: &str,
) {
    let Some(msgs) = messages.as_array_mut() else {
        return;
    };
    let mut id_remap: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    for (m, src) in msgs.iter_mut().zip(typed) {
        if src.model_id.as_deref() == Some(model) {
            continue;
        }
        let Some(content) = m.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        for block in content.iter_mut() {
            let Some(obj) = block.as_object_mut() else {
                continue;
            };
            if obj.get("type").and_then(Value::as_str) != Some("tool_use") {
                continue;
            }
            let Some(id) = obj.get("id").and_then(Value::as_str) else {
                continue;
            };
            let normalized = normalize_tool_call_id(id);
            if normalized != id {
                id_remap.insert(id.to_string(), normalized.clone());
                obj.insert("id".into(), Value::String(normalized));
            }
        }
    }
    if id_remap.is_empty() {
        return;
    }
    for m in msgs.iter_mut() {
        let Some(content) = m.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        for block in content.iter_mut() {
            let Some(obj) = block.as_object_mut() else {
                continue;
            };
            if obj.get("type").and_then(Value::as_str) != Some("tool_result") {
                continue;
            }
            if let Some(new_id) = obj
                .get("tool_use_id")
                .and_then(Value::as_str)
                .and_then(|id| id_remap.get(id))
            {
                obj.insert("tool_use_id".into(), Value::String(new_id.clone()));
            }
        }
    }
}

/// Replace every character outside `[a-zA-Z0-9_-]` with `_`, then truncate to 64 chars — Anthropic's
/// own `tool_use.id`/`tool_result.tool_use_id` pattern requirement. Every substituted character is
/// ASCII, so byte-truncation and char-truncation coincide; a plain already-conformant id (the common
/// case) round-trips as a no-op beyond the linear scan.
fn normalize_tool_call_id(id: &str) -> String {
    let mut out: String = id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    out.truncate(64);
    out
}

/// Stamp a cache breakpoint onto the last content block of the last message. No-op if the history is
/// empty or the final message carries no content blocks.
fn mark_last_block(messages: &mut Value, cc: &Value) {
    if let Some(block) = messages
        .as_array_mut()
        .and_then(|msgs| msgs.last_mut())
        .and_then(|m| m.get_mut("content"))
        .and_then(Value::as_array_mut)
        .and_then(|content| content.last_mut())
        .and_then(Value::as_object_mut)
    {
        block.insert("cache_control".into(), cc.clone());
    }
}

/// Downgrade a `thinking` block with no (or empty) `signature` to a plain `text` block, matching pi's
/// `convertMessages` (`anthropic-messages.ts`: "e.g., from an aborted stream"). Anthropic requires a
/// signed thinking block to replay it verbatim on a later tool turn; our own architecture generally
/// avoids persisting a partial/cancelled turn in the first place (narrowing the window this ever fires
/// in versus pi), but a non-conformant proxy, or a bug that delivers `message_stop` before a thinking
/// block's `signature_delta`, would otherwise send an empty `signature` Anthropic likely rejects rather
/// than degrading gracefully. A no-op for the common case (every thinking block already signed).
///
/// A block whose `thinking` text is empty or whitespace-only (e.g. a stream aborted before any
/// delta landed) is dropped rather than downgraded — *before* the signed/unsigned distinction is
/// even considered, matching pi's own ordering (`anthropic-messages.ts`'s `convertMessages`:
/// `if (block.thinking.trim().length === 0) continue;` runs ahead of its signature check). This
/// applies even to a *signed* empty block: Anthropic's `thinking` content block requires non-empty
/// text just as its `text` block does, so a signed-but-empty block would just trade one 400
/// (missing signature) for another (empty text) if kept as-is, and downgrading it would trade that
/// for a third (`{"type": "text", "text": ""}`). Only a non-empty block reaches the signed/unsigned
/// branch below: signed survives verbatim for replay, unsigned downgrades to `text`. Mirrors
/// [`crate::session::Session::scrub_cross_model_state`]'s same empty-thinking-drops-instead-of-degrades
/// rule.
fn downgrade_unsigned_thinking(messages: &mut Value) {
    let Some(msgs) = messages.as_array_mut() else {
        return;
    };
    for m in msgs {
        let Some(content) = m.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        content.retain_mut(|block| {
            let Some(obj) = block.as_object_mut() else {
                return true;
            };
            if obj.get("type").and_then(Value::as_str) != Some("thinking") {
                return true;
            }
            let text = obj
                .get("thinking")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            if text.trim().is_empty() {
                return false;
            }
            let signed = obj
                .get("signature")
                .and_then(Value::as_str)
                .is_some_and(|s| !s.is_empty());
            if signed {
                return true;
            }
            *block = json!({ "type": "text", "text": text });
            true
        });
    }
}

/// A model that can't accept images placeholder-text for one, matching pi's `transform-messages.ts`
/// (`downgradeUnsupportedImages`).
const USER_IMAGE_PLACEHOLDER: &str = "(image omitted: model does not support images)";
/// Same idea, for a tool result's images specifically — a distinct string so a transcript reader can
/// tell which shape produced it.
const TOOL_IMAGE_PLACEHOLDER: &str = "(tool image omitted: model does not support images)";
/// Default text for a vision-capable tool result that carries images but no text at all — matching
/// pi's `convertContentBlocks` (`anthropic-messages.ts`), which unshifts this exact string as a text
/// block before the image blocks rather than sending a content array with no text part. Distinct from
/// [`TOOL_IMAGE_PLACEHOLDER`]/[`USER_IMAGE_PLACEHOLDER`] above: those two stand in for an image the
/// model can't see at all (vision unsupported); this one accompanies a real image the model *can* see.
const TOOL_RESULT_IMAGE_ONLY_TEXT: &str = "(see attached image)";

/// Replace image content with a text placeholder when `supports_vision` is `false` — sending an image
/// to a model that doesn't accept one would otherwise 400 the whole turn instead of degrading
/// gracefully. Runs *before* [`encode_tool_result_images`], while a `tool_result`'s images still live
/// in their own `images` field (simpler to clear there than to un-splice them from `content` after).
/// A no-op when `supports_vision` is `true` — the overwhelmingly common case costs nothing beyond the
/// flag check.
fn downgrade_unsupported_images(messages: &mut Value, supports_vision: bool) {
    if supports_vision {
        return;
    }
    let Some(msgs) = messages.as_array_mut() else {
        return;
    };
    for m in msgs {
        let Some(content) = m.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        for block in content.iter_mut() {
            let Some(obj) = block.as_object_mut() else {
                continue;
            };
            if obj.get("type").and_then(Value::as_str) != Some("tool_result") {
                continue;
            }
            let had_images =
                matches!(obj.get("images"), Some(Value::Array(imgs)) if !imgs.is_empty());
            if !had_images {
                continue;
            }
            obj.remove("images");
            let mut text = obj
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str(TOOL_IMAGE_PLACEHOLDER);
            obj.insert("content".into(), Value::String(text));
        }
        // Collapse any run of consecutive user-turn `{"type":"image"}` blocks into one placeholder
        // text block, matching pi (rather than one placeholder per image).
        let old = std::mem::take(content);
        let mut pending_placeholder = false;
        for block in old {
            let is_image = block.get("type").and_then(Value::as_str) == Some("image");
            if is_image {
                pending_placeholder = true;
                continue;
            }
            if pending_placeholder {
                content.push(json!({ "type": "text", "text": USER_IMAGE_PLACEHOLDER }));
                pending_placeholder = false;
            }
            content.push(block);
        }
        if pending_placeholder {
            content.push(json!({ "type": "text", "text": USER_IMAGE_PLACEHOLDER }));
        }
    }
}

/// Rewrite `tool_result` blocks carrying images into Anthropic's content-array shape. The derived
/// JSON is `{type:"tool_result", content:"text", images:[…]}`, but Anthropic wants the images *inside*
/// `content`: `{content:[{type:"text",text},{type:"image",source}…]}`. A no-op for the common
/// text-only result (no `images` key), so the existing wire is untouched.
fn encode_tool_result_images(messages: &mut Value) {
    let Some(msgs) = messages.as_array_mut() else {
        return;
    };
    for m in msgs {
        let Some(content) = m.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        for block in content {
            let Some(obj) = block.as_object_mut() else {
                continue;
            };
            if obj.get("type").and_then(Value::as_str) != Some("tool_result") {
                continue;
            }
            let images = match obj.remove("images") {
                Some(Value::Array(imgs)) if !imgs.is_empty() => imgs,
                // No images (or the key was already absent): leave the string `content` as-is.
                _ => continue,
            };
            let text = obj
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let mut parts: Vec<Value> = Vec::new();
            if !text.is_empty() {
                parts.push(json!({ "type": "text", "text": text }));
            } else {
                // Images with no accompanying text: default to pi's own placeholder rather than
                // sending a content array whose first (and only, until the loop below) block is an
                // image with nothing labeling it.
                parts.push(json!({ "type": "text", "text": TOOL_RESULT_IMAGE_ONLY_TEXT }));
            }
            // A serialized `ImageSource` is exactly Anthropic's `source` object (`type:"base64"`, …).
            for source in images {
                parts.push(json!({ "type": "image", "source": source }));
            }
            obj.insert("content".into(), Value::Array(parts));
        }
    }
}

/// Mark every tool definition eager-input-streaming-capable — mutually exclusive with the
/// `fine-grained-tool-streaming-2025-05-14` beta header (see `client.rs`), which only applies to models
/// where this capability is absent.
fn mark_eager_tool_streaming(tools: &mut Value) {
    if let Some(list) = tools.as_array_mut() {
        for tool in list.iter_mut().filter_map(Value::as_object_mut) {
            tool.insert("eager_input_streaming".into(), json!(true));
        }
    }
}

/// Stamp a cache breakpoint onto the last tool definition. Only called by [`build_body`] when
/// [`crate::models::ModelCaps::supports_cache_control_on_tools`] is `true` for the active model — see
/// that field's own doc comment for why this is no longer unconditional.
fn mark_last_tool(tools: &mut Value, cc: &Value) {
    if let Some(tool) = tools
        .as_array_mut()
        .and_then(|t| t.last_mut())
        .and_then(Value::as_object_mut)
    {
        tool.insert("cache_control".into(), cc.clone());
    }
}

fn map_stop_reason(s: Option<&str>) -> StopReason {
    match s {
        Some("end_turn") => StopReason::EndTurn,
        Some("tool_use") => StopReason::ToolUse,
        Some("max_tokens") => StopReason::MaxTokens,
        Some("stop_sequence") => StopReason::StopSequence,
        Some("refusal") => StopReason::Refusal,
        // `pause_turn` is Anthropic pausing a long-running turn it expects the client to *resubmit* to
        // continue — not a natural end. We have no resubmit step in the loop, so map it to `Other`
        // rather than `EndTurn`: reading it as a clean end-of-turn would silently truncate a turn the
        // model meant to keep going. (A fully distinct `PauseTurn` variant that drives a resubmit would
        // need a `message.rs` enum change plus agent-loop handling — out of scope for this fix.)
        Some("pause_turn") => StopReason::Other,
        // Content flagged by safety filters mid-generation — not yet a named variant in Anthropic's own
        // SDK types, but a real terminal state the reference agent treats as an error, not a clean end.
        // We don't have a distinct explanation to surface for it (unlike `refusal`, which carries one in
        // `stop_details.explanation`), so it shares `Refusal`'s variant rather than earning a new one —
        // both mean "the model was blocked from completing," and the loop already lets a caller tell
        // either apart from a normal end-of-turn instead of reading it as success.
        Some("sensitive") => StopReason::Refusal,
        Some(other) => {
            // A genuinely unrecognized value (Anthropic added a new terminal state we don't know about
            // yet) silently collapsing into `Other` — which the loop treats identically to a normal
            // `EndTurn` — would hide a real change in provider behavior. `warn!` so it's at least
            // visible, without hard-failing the turn (the reference agent throws here; we're more
            // conservative since a false-positive on this match would abort an otherwise-fine turn).
            tracing::warn!(
                stop_reason = other,
                "unrecognized Anthropic stop_reason; treating as Other"
            );
            StopReason::Other
        }
        None => StopReason::Other,
    }
}

/// A synthetic block index for the refusal-explanation text `message_delta` synthesizes (see its own
/// handler below) — guaranteed never to collide with a real Anthropic content-block index (always
/// small, starting at 0).
const REFUSAL_EXPLANATION_INDEX: usize = usize::MAX;

/// Decodes Anthropic SSE. Tracks token usage (input + cache reads/writes from `message_start`,
/// output from `message_delta`) and the stop reason, emitting a single `Usage` + `MessageStop` at
/// `message_stop`. `saw_start`/`saw_stop` let `finish` reject a stream truncated before its terminal
/// `message_stop`. `is_oauth`/`tools` are only needed to reverse a live `tool_use` block's name back
/// out of Claude Code's canonical casing (see [`from_claude_code_name`]) — `Default` (used directly by
/// the bench harness, which never exercises OAuth) leaves `is_oauth` `false`, a no-op for that reversal.
#[derive(Default)]
pub struct Decoder {
    usage: TokenUsage,
    stop_reason: StopReason,
    saw_start: bool,
    saw_stop: bool,
    is_oauth: bool,
    tools: Arc<[ToolDef]>,
}

impl Decoder {
    /// A decoder for an OAuth (`is_oauth`) or plain request, given the tool list this same turn
    /// advertised (via `build_body`) — needed only to reverse `from_claude_code_name` on a live
    /// `tool_use` block when `is_oauth`.
    pub fn new(is_oauth: bool, tools: Arc<[ToolDef]>) -> Self {
        Self {
            is_oauth,
            tools,
            ..Self::default()
        }
    }
}

/// Typed shape of a `content_block_delta` event — the crate's single highest-frequency SSE event
/// (hundreds to thousands per turn: every streamed text/thinking token and every tool-argument
/// fragment). Deserializing directly into this instead of a generic `Value` skips the AST allocation
/// entirely for the common case; see `StreamDecoder::try_fast_path`'s doc comment. The `#[serde(tag =
/// "type")]` on `FastDelta` makes the whole parse fail (falling back to the general path) for any
/// `delta.type` this crate doesn't recognize — the same "unrecognized delta type; dropping" case
/// `Decoder::push`'s own `content_block_delta` arm already warns and drops, so a future Anthropic
/// delta kind added to the wire degrades to the slow path's existing warning rather than being
/// silently misparsed here.
///
/// `kind` (the outer event's own `type`) is captured and checked explicitly in `try_fast_path` rather
/// than assumed from the `index`+`delta` shape alone: today no other Anthropic SSE event carries both
/// an `index` and a `delta.type`-tagged object, so this can't currently misfire, but nothing stops a
/// future event type from coincidentally matching that shape — the explicit check keeps that failure
/// mode closed rather than relying on it staying true forever.
#[derive(Deserialize)]
struct FastContentBlockDelta {
    #[serde(rename = "type")]
    kind: String,
    index: usize,
    delta: FastDelta,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum FastDelta {
    #[serde(rename = "text_delta")]
    Text { text: String },
    #[serde(rename = "thinking_delta")]
    Thinking { thinking: String },
    #[serde(rename = "signature_delta")]
    Signature { signature: String },
    #[serde(rename = "input_json_delta")]
    InputJson { partial_json: String },
}

impl StreamDecoder for Decoder {
    fn try_fast_path(&mut self, payload: &str) -> Option<Vec<StreamEvent>> {
        // A bare, top-level `#[serde(tag = "type")]` enum (no wrapper struct) would also happily
        // accept a payload whose `type` is `"text_delta"` etc. with no outer `content_block_delta`
        // envelope at all — never a real shape on this wire, but cheap to rule out precisely by
        // requiring the outer `FastContentBlockDelta`'s own fields (`index`/`delta`) instead of
        // trusting the inner tag alone.
        let parsed: FastContentBlockDelta = serde_json::from_str(payload).ok()?;
        if parsed.kind != "content_block_delta" {
            return None;
        }
        let index = parsed.index;
        Some(vec![match parsed.delta {
            FastDelta::Text { text } => StreamEvent::TextDelta { index, text },
            FastDelta::Thinking { thinking } => StreamEvent::ThinkingDelta {
                index,
                text: thinking,
            },
            FastDelta::Signature { signature } => StreamEvent::SignatureDelta { index, signature },
            FastDelta::InputJson { partial_json } => StreamEvent::InputJsonDelta {
                index,
                partial_json,
            },
        }])
    }

    fn push(&mut self, data: &Value) -> Vec<StreamEvent> {
        let kind = data.get("type").and_then(Value::as_str).unwrap_or("");
        match kind {
            "message_start" => {
                self.saw_start = true;
                let usage = data.get("message").and_then(|m| m.get("usage"));
                self.usage.input_tokens = u32_at(usage, "input_tokens");
                self.usage.output_tokens = u32_at(usage, "output_tokens");
                // Cache accounting is reported only on `message_start` in the real API; capturing it
                // is what makes the prompt-cache breakpoints we stamp in `build_body` observable.
                self.usage.cache_read_tokens = u32_at(usage, "cache_read_input_tokens");
                self.usage.cache_write_tokens = u32_at(usage, "cache_creation_input_tokens");
                // The 1h/5m TTL split lives one level deeper, only when the provider breaks it out.
                self.usage.cache_write_1h_tokens = usage
                    .and_then(|u| u.get("cache_creation"))
                    .map(|cc| u32_at(Some(cc), "ephemeral_1h_input_tokens"))
                    .unwrap_or(0);
                vec![StreamEvent::MessageStart]
            }
            "content_block_start" => {
                let Some(index) = usize_at(Some(data), "index") else {
                    tracing::warn!(
                        "dropping Anthropic content_block_start event: missing/malformed index"
                    );
                    return Vec::new();
                };
                let block = data.get("content_block");
                match block.and_then(|b| b.get("type")).and_then(Value::as_str) {
                    Some("tool_use") => {
                        let id = str_at(block, "id").to_string();
                        let raw_name = str_at(block, "name");
                        // Reverse Claude Code's canonical casing back to our own real tool name — the
                        // OAuth-shaped mirror of `canonicalize_tool_names`' forward rename in
                        // `build_body`. A no-op for a non-OAuth decoder (`is_oauth: false`).
                        let name = if self.is_oauth {
                            from_claude_code_name(raw_name, &self.tools)
                        } else {
                            raw_name.to_string()
                        };
                        vec![StreamEvent::ToolUseStart { index, id, name }]
                    }
                    // A redacted-thinking block is fully delivered here (no deltas follow): its opaque
                    // `data` must be replayed verbatim so the model keeps reasoning continuity.
                    Some("redacted_thinking") => vec![StreamEvent::RedactedThinking {
                        index,
                        data: str_at(block, "data").to_string(),
                    }],
                    // Text and (clear) thinking blocks open empty and accrue via deltas — no event.
                    Some("text") | Some("thinking") => Vec::new(),
                    Some(other) => {
                        // A genuinely new content-block type (Anthropic has added several over time —
                        // `server_tool_use`, `web_search_tool_result`, `mcp_tool_result`) silently
                        // dropping the whole block with no signal would hide a provider capability
                        // change until a user notices missing content.
                        tracing::warn!(
                            block_type = other,
                            "unrecognized Anthropic content_block_start type; dropping block"
                        );
                        Vec::new()
                    }
                    None => Vec::new(),
                }
            }
            "content_block_delta" => {
                let Some(index) = usize_at(Some(data), "index") else {
                    tracing::warn!(
                        "dropping Anthropic content_block_delta event: missing/malformed index"
                    );
                    return Vec::new();
                };
                let delta = data.get("delta");
                match delta.and_then(|d| d.get("type")).and_then(Value::as_str) {
                    Some("text_delta") => {
                        vec![StreamEvent::TextDelta {
                            index,
                            text: str_at(delta, "text").to_string(),
                        }]
                    }
                    Some("thinking_delta") => {
                        vec![StreamEvent::ThinkingDelta {
                            index,
                            text: str_at(delta, "thinking").to_string(),
                        }]
                    }
                    Some("signature_delta") => {
                        vec![StreamEvent::SignatureDelta {
                            index,
                            signature: str_at(delta, "signature").to_string(),
                        }]
                    }
                    Some("input_json_delta") => {
                        vec![StreamEvent::InputJsonDelta {
                            index,
                            partial_json: str_at(delta, "partial_json").to_string(),
                        }]
                    }
                    Some(other) => {
                        tracing::warn!(
                            delta_type = other,
                            "unrecognized Anthropic content_block_delta type; dropping delta"
                        );
                        Vec::new()
                    }
                    None => Vec::new(),
                }
            }
            "content_block_stop" => match usize_at(Some(data), "index") {
                Some(index) => vec![StreamEvent::ContentBlockStop { index }],
                None => {
                    tracing::warn!(
                        "dropping Anthropic content_block_stop event: missing/malformed index"
                    );
                    Vec::new()
                }
            },
            "message_delta" => {
                let delta = data.get("delta");
                self.stop_reason = map_stop_reason(
                    delta
                        .and_then(|d| d.get("stop_reason"))
                        .and_then(Value::as_str),
                );
                let usage = data.get("usage");
                let out = u32_at(usage, "output_tokens");
                if out > 0 {
                    self.usage.output_tokens = out;
                }
                // Real Anthropic only reports cache fields on `message_start`, never here — but a
                // proxy sitting in front of it could, and a stale `message_start`-only snapshot would
                // then silently under/over-report the rest of the turn. Refresh only when present, so
                // this is a no-op against the real API's actual behavior.
                if let Some(read) = usage.and_then(|u| u.get("cache_read_input_tokens")) {
                    if let Some(read) = read.as_u64() {
                        self.usage.cache_read_tokens = read as u32;
                    }
                }
                if let Some(write) = usage.and_then(|u| u.get("cache_creation_input_tokens")) {
                    if let Some(write) = write.as_u64() {
                        self.usage.cache_write_tokens = write as u32;
                    }
                }
                if let Some(cc) = usage.and_then(|u| u.get("cache_creation")) {
                    self.usage.cache_write_1h_tokens =
                        u32_at(Some(cc), "ephemeral_1h_input_tokens");
                }
                // Reasoning tokens, when broken out separately, are still *included* in
                // `output_tokens`; capture them so a caller can see the thinking share of the spend.
                let thinking = usage
                    .and_then(|u| u.get("output_tokens_details"))
                    .map(|d| u32_at(Some(d), "thinking_tokens"))
                    .unwrap_or(0);
                if thinking > 0 {
                    self.usage.reasoning_tokens = thinking;
                }
                // On a refusal, Anthropic carries a human-readable reason in
                // `delta.stop_details.explanation`. Surface it as a text delta so it lands in the
                // assembled assistant message instead of being dropped — otherwise a refusal arrives as
                // an empty turn with only a `Refusal` stop reason, and the caller can't tell the user
                // *why*. (The block has already closed by `message_delta`, so this trailing text is
                // flushed as its own block; see the loop's `Accumulator`.)
                if self.stop_reason == StopReason::Refusal {
                    let explanation = delta
                        .and_then(|d| d.get("stop_details"))
                        .and_then(|sd| sd.get("explanation"))
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    if !explanation.is_empty() {
                        // A fresh, never-real index: every actual content block has already closed
                        // by the time `message_delta` arrives, so this can't collide with one — and
                        // it never gets an explicit `ContentBlockStop` (there isn't a wire event for
                        // it), relying on `Accumulator::finish()`'s own "flush whatever's still open"
                        // pass at the end of the turn, exactly as it did before this event carried an
                        // index at all.
                        return vec![StreamEvent::TextDelta {
                            index: REFUSAL_EXPLANATION_INDEX,
                            text: explanation.to_string(),
                        }];
                    }
                }
                Vec::new()
            }
            "message_stop" => {
                self.saw_stop = true;
                vec![
                    StreamEvent::Usage(self.usage),
                    StreamEvent::MessageStop {
                        stop_reason: self.stop_reason,
                    },
                ]
            }
            // A real, frequent no-op: Anthropic sends a periodic `ping` keepalive on a long-running
            // stream — expected, not worth logging every time it fires.
            "ping" => Vec::new(),
            other => {
                // Anthropic adding a new top-level SSE event type (this crate doesn't request any of
                // the server-side tool capabilities that would trigger one today, but both providers
                // have added streaming event types before) would otherwise silently drop that event's
                // entire content with no signal — unlike `stop_reason`'s own unrecognized-value
                // handling above, which does warn.
                tracing::warn!(
                    event_type = other,
                    "unrecognized Anthropic SSE event type; dropping"
                );
                Vec::new()
            }
        }
    }

    fn finish(&mut self) -> Result<Vec<StreamEvent>> {
        // A stream that opened (`message_start`) but never delivered `message_stop` was truncated
        // mid-flight — a dropped connection or a gateway cut. Reject it rather than let the partial
        // turn pass as a clean completion.
        if self.saw_start && !self.saw_stop {
            return Err(Error::Transport(
                "Anthropic stream ended before message_stop".into(),
            ));
        }
        Ok(Vec::new())
    }

    fn is_terminal(&self) -> bool {
        self.saw_stop
    }

    // pi-parity fix: pi tolerates a malformed-but-recoverable Anthropic event body (an invalid
    // backslash escape, a raw control character inside a string value) by repairing it before
    // parsing — see `StreamDecoder::repairs_json`'s doc comment and
    // `packages/ai/test/anthropic-sse-parsing.test.ts:82-167`. Anthropic is the only dialect that
    // opts in, matching pi's own scoping (every other dialect's outer SSE parse is owned by that
    // provider's SDK, not hand-rolled the way Anthropic's is).
    fn repairs_json(&self) -> bool {
        true
    }
}

fn str_at<'a>(v: Option<&'a Value>, key: &str) -> &'a str {
    v.and_then(|v| v.get(key))
        .and_then(Value::as_str)
        .unwrap_or("")
}

fn u32_at(v: Option<&Value>, key: &str) -> u32 {
    v.and_then(|v| v.get(key))
        .and_then(Value::as_u64)
        .map_or(0, super::saturating_u32)
}

/// Anthropic's own `content_block_start`/`_delta`/`_stop` all carry a real `index` — read it straight
/// through rather than discarding it, so genuinely interleaved blocks (should Anthropic ever deliver
/// them; not observed in practice today, but the wire already carries the field) accumulate correctly
/// instead of relying on an assumption of strict sequential delivery. `None` on a missing or
/// non-numeric `index` (a malformed event, or an intermediary that mangled it) — never defaulted to
/// `0`: that's a real, commonly-open index in a parallel-tool-call turn, so silently defaulting to it
/// would misattribute this event's content into whichever block happens to occupy slot 0 instead of
/// dropping the malformed event, matching how `dialect::openai`'s own index-ambiguity handling drops
/// rather than guesses.
fn usize_at(v: Option<&Value>, key: &str) -> Option<usize> {
    v.and_then(|v| v.get(key))
        .and_then(Value::as_u64)
        .map(|n| n as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::decode_sse;
    use crate::message::{ContentBlock, Message, ToolDef};
    use serde_json::json;

    #[test]
    fn build_body_hoists_system_and_keeps_blocks() {
        let req = ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 256)
            .with_system("be brief")
            .with_tools(vec![ToolDef {
                name: "read".into(),
                description: "read a file".into(),
                input_schema: json!({ "type": "object" }),
            }]);
        let body = build_body(&req, false);
        assert_eq!(body["model"], "claude-opus-4-8");
        assert_eq!(body["stream"], true);
        // System is a cached text-block array (a dedicated breakpoint), not a bare string.
        assert_eq!(body["system"][0]["type"], "text");
        assert_eq!(body["system"][0]["text"], "be brief");
        assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"][0]["type"], "text");
        assert_eq!(body["tools"][0]["name"], "read");
        assert_eq!(body["tools"][0]["input_schema"]["type"], "object");
    }

    #[test]
    fn build_body_sends_temperature_when_set_and_not_thinking() {
        // A model that actually supports `temperature` (unlike our own default `claude-opus-4-8` —
        // see `build_body_omits_temperature_for_a_model_that_rejects_it_outright` below).
        let req = ModelRequest::new("claude-sonnet-4-5", vec![Message::user("hi")], 256)
            .with_temperature(0.4);
        let body = build_body(&req, false);
        assert_eq!(body["temperature"], 0.4);
    }

    #[test]
    fn build_body_omits_temperature_when_unset() {
        let req = ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 256);
        let body = build_body(&req, false);
        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn build_body_omits_temperature_when_thinking_is_enabled() {
        // Anthropic forbids `temperature` alongside extended thinking — matches pi's own
        // `!options?.thinkingEnabled` gate. Uses a gen6+ model that *does* otherwise support
        // `temperature` (opus-4-6, not opus-4-8), isolating this thinking-state gate from the
        // separate per-model `supports_temperature` gate covered by the test below.
        let req = ModelRequest::new("claude-opus-4-6", vec![Message::user("hi")], 4096)
            .with_temperature(0.9)
            .with_thinking(1024);
        let body = build_body(&req, false);
        assert!(
            body.get("temperature").is_none(),
            "got: {:#?}",
            body.get("temperature")
        );
    }

    #[test]
    fn build_body_omits_temperature_for_a_model_that_rejects_it_outright() {
        // pi-parity: `claude-opus-4-7`/`claude-opus-4-8` (`compat.supportsTemperature: false` in
        // `anthropic.models.ts`) reject `temperature` unconditionally — not only while thinking is on.
        // `claude-opus-4-8` is Beyond's own `DEFAULT_MODEL`, so this is a live, reachable 400 without
        // the gate: thinking is left at its default (which resolves to an explicit `{"type":
        // "disabled"}` on the wire for this model, since it's `reasoning_disableable`) and a
        // `temperature` is explicitly requested — both must still not produce a `temperature` field.
        let req = ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 256)
            .with_temperature(0.7);
        let body = build_body(&req, false);
        assert_eq!(body["thinking"]["type"], "disabled");
        assert!(
            body.get("temperature").is_none(),
            "claude-opus-4-8 must never receive `temperature`, got: {:#?}",
            body.get("temperature")
        );
    }

    #[test]
    fn build_body_omits_temperature_for_opus_4_7_too() {
        // Same gate, the sibling id pi also marks `supportsTemperature: false` for.
        let req = ModelRequest::new("claude-opus-4-7", vec![Message::user("hi")], 256)
            .with_temperature(0.7);
        let body = build_body(&req, false);
        assert!(
            body.get("temperature").is_none(),
            "got: {:#?}",
            body.get("temperature")
        );
    }

    #[test]
    fn build_body_never_leaks_model_id_onto_the_wire() {
        // `Message::model_id` is internal-only provenance for `Session::scrub_cross_model_state` — a
        // live 400 (`messages.N.model_id: Extra inputs are not permitted`) proved this was leaking
        // straight through `serde_json::to_value(req.messages)` before this dialect stripped it (now
        // structurally unreachable — see `build_body`'s own doc comment on its field-by-field
        // `{role, content}` message construction).
        let req = ModelRequest::new(
            "claude-opus-4-8",
            vec![
                Message::user("hi"),
                Message::assistant(vec![ContentBlock::text("hello")])
                    .with_model_id("claude-opus-4-8"),
            ],
            256,
        );
        let body = build_body(&req, false);
        for m in body["messages"].as_array().unwrap() {
            assert!(
                m.get("model_id").is_none(),
                "model_id must never reach the wire: {m}"
            );
        }
    }

    #[test]
    fn build_body_never_leaks_error_message_or_aborted_onto_the_wire() {
        // Same leak class as `model_id` above, caught before it ever reached a live request: a
        // `Message::error`/`Message::with_aborted` closing record is exactly the kind of message a
        // whole-run retry or a client's follow-up `prompt` replays on a real request, and Anthropic
        // rejects any message object carrying a field outside its schema.
        let req = ModelRequest::new(
            "claude-opus-4-8",
            vec![
                Message::user("hi"),
                Message::error("transport error: boom"),
                Message::user("try again"),
                Message::assistant(vec![ContentBlock::text("partial")])
                    .with_model_id("claude-opus-4-8")
                    .with_aborted(),
            ],
            256,
        );
        let body = build_body(&req, false);
        for m in body["messages"].as_array().unwrap() {
            assert!(
                m.get("error_message").is_none(),
                "error_message must never reach the wire: {m}"
            );
            assert!(
                m.get("aborted").is_none(),
                "aborted must never reach the wire: {m}"
            );
        }
    }

    #[test]
    fn build_body_never_leaks_usage_or_stop_reason_onto_the_wire() {
        // CRITICAL regression: `Agent::run_events_steered` stamps `usage` and `stop_reason` onto
        // *every* real assistant turn (not just a synthetic error/abort record), so any ordinary
        // multi-turn conversation replays them on the very next request. Empirically confirmed live
        // against real Anthropic before this fix: `messages.1.stop_reason: Extra inputs are not
        // permitted` (400). `build_body`'s field-by-field message construction must never carry either.
        let req = ModelRequest::new(
            "claude-opus-4-8",
            vec![
                Message::user("hi"),
                Message::assistant(vec![ContentBlock::text("hello")])
                    .with_model_id("claude-opus-4-8")
                    .with_usage(TokenUsage {
                        input_tokens: 10,
                        output_tokens: 5,
                        ..Default::default()
                    })
                    .with_stop_reason(StopReason::EndTurn),
                Message::user("go on"),
            ],
            256,
        );
        let body = build_body(&req, false);
        for m in body["messages"].as_array().unwrap() {
            assert!(
                m.get("usage").is_none(),
                "usage must never reach the wire: {m}"
            );
            assert!(
                m.get("stop_reason").is_none(),
                "stop_reason must never reach the wire: {m}"
            );
        }
    }

    /// Generic safety-net canary (pi-parity, Task B): unlike the three tests above, which each name a
    /// specific field, this asserts the wire message object's key *set* is exactly `{role, content}` —
    /// so a future field added to `Message` (that this dialect's `build_body` doesn't explicitly thread
    /// through) fails this test immediately instead of silently reaching the wire and 400ing in
    /// production, the way `usage`/`stop_reason` did for two whole passes before anyone noticed. Builds
    /// a message with every current optional field populated to prove the field-by-field
    /// `{role, content}` construction really does exclude all of them, not just the three already
    /// covered above.
    #[test]
    fn build_body_wire_messages_never_carry_any_field_besides_role_and_content() {
        let req = ModelRequest::new(
            "claude-opus-4-8",
            vec![
                Message::user("hi"),
                Message::assistant(vec![ContentBlock::text("hello")])
                    .with_model_id("claude-opus-4-8")
                    .with_aborted()
                    .with_usage(TokenUsage {
                        input_tokens: 10,
                        output_tokens: 5,
                        ..Default::default()
                    })
                    .with_stop_reason(StopReason::EndTurn),
                Message::error("transport error: boom"),
            ],
            256,
        );
        let body = build_body(&req, false);
        for m in body["messages"].as_array().unwrap() {
            let keys: std::collections::BTreeSet<&str> =
                m.as_object().unwrap().keys().map(String::as_str).collect();
            assert_eq!(
                keys,
                std::collections::BTreeSet::from(["role", "content"]),
                "a wire message must carry exactly role/content, got: {m}"
            );
        }
    }

    #[test]
    fn build_body_marks_every_tool_eager_input_streaming() {
        let req =
            ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 256).with_tools(vec![
                ToolDef {
                    name: "read".into(),
                    description: "read a file".into(),
                    input_schema: json!({ "type": "object" }),
                },
                ToolDef {
                    name: "write".into(),
                    description: "write a file".into(),
                    input_schema: json!({ "type": "object" }),
                },
            ]);
        let body = build_body(&req, false);
        assert_eq!(body["tools"][0]["eager_input_streaming"], true);
        assert_eq!(body["tools"][1]["eager_input_streaming"], true);
    }

    #[test]
    fn build_body_stamps_cache_breakpoints() {
        let req = ModelRequest::new(
            "claude-opus-4-8",
            vec![
                Message::user("hi"),
                Message::tool_result("tu_1", "out", false),
            ],
            256,
        )
        .with_tools(vec![
            ToolDef {
                name: "read".into(),
                description: "read a file".into(),
                input_schema: json!({ "type": "object" }),
            },
            ToolDef {
                name: "write".into(),
                description: "write a file".into(),
                input_schema: json!({ "type": "object" }),
            },
        ]);
        let body = build_body(&req, false);
        // Anchor breakpoint on the last (only the last) tool definition.
        assert!(body["tools"][0].get("cache_control").is_none());
        assert_eq!(body["tools"][1]["cache_control"]["type"], "ephemeral");
        // Rolling breakpoint on the last block of the last message, and nowhere earlier.
        assert!(
            body["messages"][0]["content"][0]
                .get("cache_control")
                .is_none()
        );
        assert_eq!(
            body["messages"][1]["content"][0]["cache_control"]["type"],
            "ephemeral"
        );
    }

    #[test]
    fn fireworks_anthropic_wire_models_never_get_a_tool_cache_breakpoint() {
        // pi-parity (Task #2): pi's `fireworks.models.ts` sets `supportsCacheControlOnTools: false` on
        // every one of its 14 Anthropic-wire ids (DeepSeek-V4, GLM-5.1, gpt-oss-120b/20b, Kimi-K2.6/
        // K2.7-Code + fast/turbo variants, MiniMax-M2.7/M3, Qwen3.7-Plus) — a later routing change
        // (`is_fireworks_anthropic_wire_model`) now sends exactly these ids through this dialect, so
        // stamping `cache_control` on their last tool unconditionally (as this dialect used to) would
        // diverge from pi and risk a 400 Fireworks doesn't accept a `cache_control` breakpoint for.
        let req = ModelRequest::new(
            "accounts/fireworks/models/deepseek-v4-pro",
            vec![Message::user("hi")],
            256,
        )
        .with_tools(vec![ToolDef {
            name: "read".into(),
            description: "read a file".into(),
            input_schema: json!({ "type": "object" }),
        }]);
        let body = build_body(&req, false);
        assert!(
            body["tools"][0].get("cache_control").is_none(),
            "a Fireworks Anthropic-wire model must never get a tool cache breakpoint: {body:?}"
        );
        // The rolling message breakpoint is untouched by this gate — only tools are affected.
        assert_eq!(
            body["messages"][0]["content"][0]["cache_control"]["type"],
            "ephemeral"
        );

        // An ordinary Claude id is completely unaffected — still gets its tool breakpoint.
        let req =
            ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 256).with_tools(vec![
                ToolDef {
                    name: "read".into(),
                    description: "read a file".into(),
                    input_schema: json!({ "type": "object" }),
                },
            ]);
        let body = build_body(&req, false);
        assert_eq!(body["tools"][0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn no_cache_skips_every_breakpoint() {
        let req = ModelRequest::new(
            "claude-opus-4-8",
            vec![
                Message::user("hi"),
                Message::tool_result("tu_1", "out", false),
            ],
            256,
        )
        .with_system("be brief")
        .with_tools(vec![ToolDef {
            name: "read".into(),
            description: "read a file".into(),
            input_schema: json!({ "type": "object" }),
        }])
        .with_no_cache(true);
        let body = build_body(&req, false);
        assert!(body["tools"][0].get("cache_control").is_none());
        assert!(
            body["messages"][1]["content"][0]
                .get("cache_control")
                .is_none()
        );
        assert!(body["system"][0].get("cache_control").is_none());
        // The system block itself is still present, just uncached.
        assert_eq!(body["system"][0]["text"], "be brief");
    }

    #[test]
    fn thinking_is_explicitly_disabled_when_not_requested_on_a_disable_capable_model() {
        // No `thinking` requested on a model that supports turning it off explicitly (gen6+, minus
        // claude-fable-5) → an explicit `{"type":"disabled"}`, not silent reliance on whatever
        // Anthropic's own undocumented default does when the field is omitted.
        let req = ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 256);
        let body = build_body(&req, false);
        assert_eq!(body["thinking"]["type"], "disabled");

        // claude-fable-5 has no "off" wire shape at all (pi: `thinkingLevelMap: {"off": null}") — the
        // `thinking` field must stay omitted entirely, not sent as `{"type":"disabled"}`.
        let req = ModelRequest::new("claude-fable-5", vec![Message::user("hi")], 256);
        let body = build_body(&req, false);
        assert!(body.get("thinking").is_none());

        // Legacy gen-3 models have no thinking support at all — same "field omitted" expectation.
        let req = ModelRequest::new("claude-3-5-sonnet-20241022", vec![Message::user("hi")], 256);
        let body = build_body(&req, false);
        assert!(body.get("thinking").is_none());
    }

    #[test]
    fn image_block_serializes_to_anthropic_source_shape() {
        use crate::message::ImageSource;
        let req = ModelRequest::new(
            "claude-opus-4-8",
            vec![Message::user_with_images(
                "what is this?",
                vec![ImageSource::base64("image/png", "AAAA")],
            )],
            256,
        );
        let body = build_body(&req, false);
        let content = &body["messages"][0]["content"];
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "image");
        assert_eq!(content[1]["source"]["type"], "base64");
        assert_eq!(content[1]["source"]["media_type"], "image/png");
        assert_eq!(content[1]["source"]["data"], "AAAA");
    }

    #[test]
    fn assistant_tool_use_round_trips_into_body() {
        let req = ModelRequest::new(
            "claude-opus-4-8",
            vec![
                Message::user("weather?"),
                Message::assistant(vec![ContentBlock::tool_use(
                    "toolu_1",
                    "get_weather",
                    json!({ "city": "SF" }),
                )]),
                Message::tool_result("toolu_1", "72F", false),
            ],
            256,
        );
        let body = build_body(&req, false);
        assert_eq!(body["messages"][1]["content"][0]["type"], "tool_use");
        assert_eq!(body["messages"][1]["content"][0]["id"], "toolu_1");
        assert_eq!(body["messages"][2]["content"][0]["type"], "tool_result");
        assert_eq!(body["messages"][2]["content"][0]["tool_use_id"], "toolu_1");
    }

    #[test]
    fn cross_model_tool_use_id_gets_disallowed_characters_normalized() {
        // pi-parity (Task #28): pi's `normalizeToolCallId` (`anthropic-messages.ts:1006-1009`)
        // replaces any character outside `[a-zA-Z0-9_-]` with `_`, applied by `transformMessages`
        // whenever the message wasn't produced by the model about to see it again. A foreign-model-
        // produced tool_use id containing a `|` (the OpenAI Responses combined-id separator) or a
        // space must be normalized before it reaches an Anthropic request — and the paired
        // tool_result's `tool_use_id` must follow so the pairing survives.
        let req = ModelRequest::new(
            "claude-opus-4-8",
            vec![
                Message::user("weather?"),
                Message::assistant(vec![ContentBlock::tool_use(
                    "call_1|weird id",
                    "get_weather",
                    json!({ "city": "SF" }),
                )])
                .with_model_id("gpt-4o"),
                Message::tool_result("call_1|weird id", "72F", false),
            ],
            256,
        );
        let body = build_body(&req, false);
        assert_eq!(body["messages"][1]["content"][0]["id"], "call_1_weird_id");
        assert_eq!(
            body["messages"][2]["content"][0]["tool_use_id"],
            "call_1_weird_id"
        );
    }

    #[test]
    fn cross_model_tool_use_id_over_64_chars_gets_truncated() {
        // OpenAI Responses' own combined ids can run to 450+ chars (GitHub Copilot in particular) —
        // Anthropic rejects anything past 64.
        let long_id = "x".repeat(450);
        let req = ModelRequest::new(
            "claude-opus-4-8",
            vec![
                Message::user("weather?"),
                Message::assistant(vec![ContentBlock::tool_use(
                    long_id.clone(),
                    "get_weather",
                    json!({ "city": "SF" }),
                )])
                .with_model_id("gpt-4o"),
                Message::tool_result(long_id, "72F", false),
            ],
            256,
        );
        let body = build_body(&req, false);
        let expected = "x".repeat(64);
        assert_eq!(body["messages"][1]["content"][0]["id"], expected);
        assert_eq!(body["messages"][2]["content"][0]["tool_use_id"], expected);
    }

    #[test]
    fn same_model_tool_use_id_is_left_untouched_even_if_it_would_otherwise_normalize() {
        // A message produced by the very model about to see it again is never "foreign" — its id must
        // not be rewritten, matching `Session::scrub_cross_model_state`'s identical same-model
        // exemption. Not a realistic id for Anthropic to have generated itself (it always produces
        // conformant ids), but proves the gate is keyed on `model_id`, not on the id's own shape.
        let req = ModelRequest::new(
            "claude-opus-4-8",
            vec![
                Message::user("weather?"),
                Message::assistant(vec![ContentBlock::tool_use(
                    "toolu 1",
                    "get_weather",
                    json!({ "city": "SF" }),
                )])
                .with_model_id("claude-opus-4-8"),
                Message::tool_result("toolu 1", "72F", false),
            ],
            256,
        );
        let body = build_body(&req, false);
        assert_eq!(body["messages"][1]["content"][0]["id"], "toolu 1");
        assert_eq!(body["messages"][2]["content"][0]["tool_use_id"], "toolu 1");
    }

    // A recorded text + tool_use streamed response.
    const FIXTURE: &str = r#"
event: message_start
data: {"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":24,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Let me check."}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_42","name":"get_weather","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\"SF\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":31}}

event: message_stop
data: {"type":"message_stop"}
"#;

    #[test]
    fn decodes_text_then_tool_use_stream() {
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, FIXTURE).unwrap();
        assert_eq!(
            events,
            vec![
                StreamEvent::MessageStart,
                StreamEvent::TextDelta {
                    index: 0,
                    text: "Let me check.".into()
                },
                StreamEvent::ContentBlockStop { index: 0 },
                StreamEvent::ToolUseStart {
                    index: 1,
                    id: "toolu_42".into(),
                    name: "get_weather".into()
                },
                StreamEvent::InputJsonDelta {
                    index: 1,
                    partial_json: "{\"city\":".into()
                },
                StreamEvent::InputJsonDelta {
                    index: 1,
                    partial_json: "\"SF\"}".into()
                },
                StreamEvent::ContentBlockStop { index: 1 },
                StreamEvent::Usage(TokenUsage {
                    input_tokens: 24,
                    output_tokens: 31,
                    ..Default::default()
                }),
                StreamEvent::MessageStop {
                    stop_reason: StopReason::ToolUse
                },
            ]
        );
    }

    #[test]
    fn content_block_delta_with_missing_index_is_dropped_not_misattributed_to_index_0() {
        // Anthropic always sends `index` on every content_block_delta — this simulates a malformed or
        // relay-corrupted event (a misbehaving gateway sitting in front of the real API) that drops
        // it. Before the fix, a missing/malformed index silently defaulted to 0, corrupting whichever
        // real block happened to occupy that slot; now the malformed event must be dropped instead.
        const SSE: &str = r#"
event: message_start
data: {"type":"message_start","message":{"usage":{"input_tokens":5,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"read"}}

event: content_block_delta
data: {"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"{\"a\":1}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_stop
data: {"type":"message_stop"}
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, SSE).unwrap();
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, StreamEvent::InputJsonDelta { .. })),
            "a delta with no index must be dropped, not misattributed to block 0: {events:?}"
        );
    }

    #[test]
    fn content_block_start_with_a_non_numeric_index_is_dropped() {
        const SSE: &str = r#"
event: message_start
data: {"type":"message_start","message":{"usage":{"input_tokens":5,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":"not-a-number","content_block":{"type":"tool_use","id":"toolu_1","name":"read"}}

event: message_stop
data: {"type":"message_stop"}
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, SSE).unwrap();
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, StreamEvent::ToolUseStart { .. })),
            "a content_block_start with a malformed index must be dropped: {events:?}"
        );
    }

    #[test]
    fn unrecognized_event_and_block_types_are_dropped_without_breaking_the_rest_of_the_stream() {
        // Anthropic has added new content-block/delta/event types before (`server_tool_use`,
        // `web_search_tool_result`, `mcp_tool_result`); this simulates a not-yet-supported one arriving
        // mid-stream — it must be dropped, not error out or corrupt decoding of the real content
        // around it.
        const SSE: &str = r#"
event: message_start
data: {"type":"message_start","message":{"usage":{"input_tokens":5,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"some_future_block_type"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"some_future_delta_type"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: some_future_event
data: {"type":"some_future_event_type","whatever":"payload"}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"still works"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: message_stop
data: {"type":"message_stop"}
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, SSE).unwrap();
        assert!(
            events.contains(&StreamEvent::TextDelta {
                index: 1,
                text: "still works".into()
            }),
            "real content around the unrecognized types must still decode: {events:?}"
        );
        assert!(matches!(
            events.last(),
            Some(StreamEvent::MessageStop { .. })
        ));
    }

    #[test]
    fn captures_cache_usage_from_message_start() {
        const CACHED: &str = r#"
event: message_start
data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":12,"cache_read_input_tokens":900,"cache_creation_input_tokens":40,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}

event: message_stop
data: {"type":"message_stop"}
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, CACHED).unwrap();
        let usage = events
            .iter()
            .find_map(|e| match e {
                StreamEvent::Usage(u) => Some(*u),
                _ => None,
            })
            .expect("a usage event");
        assert_eq!(usage.input_tokens, 12);
        assert_eq!(usage.output_tokens, 7);
        assert_eq!(usage.cache_read_tokens, 900);
        assert_eq!(usage.cache_write_tokens, 40);
    }

    #[test]
    fn captures_the_1h_cache_write_split_when_the_provider_breaks_it_out() {
        const SSE: &str = r#"
event: message_start
data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":12,"cache_read_input_tokens":900,"cache_creation_input_tokens":40,"cache_creation":{"ephemeral_5m_input_tokens":10,"ephemeral_1h_input_tokens":30},"output_tokens":1}}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}

event: message_stop
data: {"type":"message_stop"}
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, SSE).unwrap();
        let usage = events
            .iter()
            .find_map(|e| match e {
                StreamEvent::Usage(u) => Some(*u),
                _ => None,
            })
            .expect("a usage event");
        // The flat sum still includes both TTLs; the 1h-specific field breaks out just that share.
        assert_eq!(usage.cache_write_tokens, 40);
        assert_eq!(usage.cache_write_1h_tokens, 30);
    }

    #[test]
    fn message_delta_refreshes_cache_counts_when_a_proxy_reports_them_there() {
        // Real Anthropic only ever reports cache fields on `message_start` — this is a defensive
        // refresh for a proxy that might report updated figures mid-stream, not something the real API
        // does; the initial `message_start` value must still be a sane baseline either way.
        const SSE: &str = r#"
event: message_start
data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":12,"cache_read_input_tokens":100,"cache_creation_input_tokens":10,"output_tokens":1}}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5,"cache_read_input_tokens":150,"cache_creation_input_tokens":10,"cache_creation":{"ephemeral_1h_input_tokens":10}}}

event: message_stop
data: {"type":"message_stop"}
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, SSE).unwrap();
        let usage = events
            .iter()
            .find_map(|e| match e {
                StreamEvent::Usage(u) => Some(*u),
                _ => None,
            })
            .expect("a usage event");
        assert_eq!(usage.cache_read_tokens, 150); // refreshed from message_delta
        assert_eq!(usage.cache_write_1h_tokens, 10);
    }

    #[test]
    fn captures_reasoning_tokens_from_message_delta() {
        const SSE: &str = r#"event: message_start
data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":10,"output_tokens":1}}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":50,"output_tokens_details":{"thinking_tokens":32}}}

event: message_stop
data: {"type":"message_stop"}
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, SSE).unwrap();
        let usage = events
            .iter()
            .find_map(|e| match e {
                StreamEvent::Usage(u) => Some(*u),
                _ => None,
            })
            .expect("a usage event");
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.reasoning_tokens, 32);
    }

    #[test]
    fn long_retention_sets_1h_ttl_on_breakpoints() {
        let req = ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 256)
            .with_system("sys")
            .with_tools(vec![ToolDef {
                name: "read".into(),
                description: "d".into(),
                input_schema: json!({ "type": "object" }),
            }])
            .with_cache_long(true);
        let body = build_body(&req, false);
        assert_eq!(body["system"][0]["cache_control"]["ttl"], "1h");
        assert_eq!(body["tools"][0]["cache_control"]["ttl"], "1h");
        assert_eq!(
            body["messages"][0]["content"][0]["cache_control"]["ttl"],
            "1h"
        );
    }

    #[test]
    fn tool_result_images_become_content_array() {
        use crate::message::ImageSource;
        let req = ModelRequest::new(
            "claude-opus-4-8",
            vec![Message::tool_results(vec![ContentBlock::ToolResult {
                tool_use_id: "tu_1".into(),
                content: "here is the screenshot".into(),
                is_error: false,
                images: vec![ImageSource::base64("image/png", "AAAA")],
            }])],
            256,
        );
        let body = build_body(&req, false);
        let content = &body["messages"][0]["content"][0]["content"];
        // The string content was rewritten into an array: text block, then image block.
        assert_eq!(
            content[0],
            json!({ "type": "text", "text": "here is the screenshot" })
        );
        assert_eq!(content[1]["type"], "image");
        assert_eq!(content[1]["source"]["type"], "base64");
        assert_eq!(content[1]["source"]["media_type"], "image/png");
        assert_eq!(content[1]["source"]["data"], "AAAA");
        // The transient `images` field must not leak onto the wire.
        assert!(body["messages"][0]["content"][0].get("images").is_none());
    }

    #[test]
    fn a_tool_result_with_only_an_image_and_no_text_gets_the_see_attached_image_placeholder() {
        // A-L8 pi-parity test gap (fixed, `packages/ai/test/image-tool-result.test.ts`): a tool that
        // returns only an image (no text) — a screenshot tool, say — must not pad the wire content
        // array with a spurious empty `{"type":"text","text":""}` block ahead of the real image, but
        // it must also not omit the text block entirely — pi's own `convertContentBlocks`
        // (`anthropic-messages.ts`) unshifts a `"(see attached image)"` text block ahead of the image
        // in exactly this case, so a transcript/UI reader always has *some* text to show.
        use crate::message::ImageSource;
        let req = ModelRequest::new(
            "claude-opus-4-8",
            vec![Message::tool_results(vec![ContentBlock::ToolResult {
                tool_use_id: "tu_1".into(),
                content: String::new(),
                is_error: false,
                images: vec![ImageSource::base64("image/png", "AAAA")],
            }])],
            256,
        );
        let body = build_body(&req, false);
        let content = body["messages"][0]["content"][0]["content"]
            .as_array()
            .unwrap();
        assert_eq!(
            content.len(),
            2,
            "expected the placeholder text block, then the image block: {content:#?}"
        );
        assert_eq!(
            content[0],
            json!({ "type": "text", "text": "(see attached image)" })
        );
        assert_eq!(content[1]["type"], "image");
        assert_eq!(content[1]["source"]["data"], "AAAA");
    }

    #[test]
    fn images_are_downgraded_to_a_placeholder_for_a_non_vision_model() {
        use crate::message::ImageSource;
        // No current Anthropic id has `supports_vision: false`, so exercise the fallback
        // (`ModelCaps::unknown()`) via a genuinely unrecognized model id — the one real path that
        // reaches a non-vision Anthropic-wire model today, and the one a not-yet-catalogued future
        // model would take.
        let req = ModelRequest::new(
            "some-future-anthropic-model",
            vec![
                Message::user_with_images(
                    "what is this?",
                    vec![ImageSource::base64("image/png", "AAAA")],
                ),
                Message::tool_results(vec![ContentBlock::ToolResult {
                    tool_use_id: "tu_1".into(),
                    content: "here is the screenshot".into(),
                    is_error: false,
                    images: vec![ImageSource::base64("image/png", "BBBB")],
                }]),
            ],
            256,
        );
        let body = build_body(&req, false);

        // User-turn image → one placeholder text block, no image block.
        let user_content = &body["messages"][0]["content"];
        assert_eq!(
            user_content[0],
            json!({ "type": "text", "text": "what is this?" })
        );
        assert_eq!(
            user_content[1],
            json!({ "type": "text", "text": "(image omitted: model does not support images)" })
        );
        assert_eq!(user_content.as_array().unwrap().len(), 2);

        // Tool-result image → placeholder appended to the text content (still a plain string — no
        // images survive to make `encode_tool_result_images` promote it to a content array), and the
        // transient `images` field is gone.
        assert_eq!(
            body["messages"][1]["content"][0]["content"],
            "here is the screenshot\n(tool image omitted: model does not support images)"
        );
        assert!(body["messages"][1]["content"][0].get("images").is_none());

        // A vision-capable model is unaffected (existing image tests already cover the positive case,
        // but assert it here too for a direct before/after contrast in one place).
        let req = ModelRequest::new(
            "claude-opus-4-8",
            vec![Message::user_with_images(
                "what is this?",
                vec![ImageSource::base64("image/png", "AAAA")],
            )],
            256,
        );
        let body = build_body(&req, false);
        assert_eq!(body["messages"][0]["content"][1]["type"], "image");
    }

    /// pi-parity (models/dialects pass, Task E): `ModelRequest::block_images` must strip/downgrade
    /// images at the wire layer regardless of the active model's own vision support.
    #[test]
    fn block_images_downgrades_images_even_for_a_vision_capable_model() {
        use crate::message::ImageSource;
        let req = ModelRequest::new(
            "claude-opus-4-8", // vision-capable — would normally keep the real image block
            vec![Message::user_with_images(
                "what is this?",
                vec![ImageSource::base64("image/png", "AAAA")],
            )],
            256,
        )
        .with_block_images(true);
        let body = build_body(&req, false);
        let content = &body["messages"][0]["content"];
        assert_eq!(content[1]["type"], "text");
        assert_eq!(
            content[1]["text"], "(image omitted: model does not support images)",
            "block_images must downgrade the image even though claude-opus-4-8 supports vision"
        );
        assert!(
            !content
                .as_array()
                .unwrap()
                .iter()
                .any(|b| b["type"] == "image"),
            "no image block should reach the wire when block_images is set"
        );

        // Unset (the default), the same request keeps its real image.
        let req = ModelRequest::new(
            "claude-opus-4-8",
            vec![Message::user_with_images(
                "what is this?",
                vec![ImageSource::base64("image/png", "AAAA")],
            )],
            256,
        );
        let body = build_body(&req, false);
        assert_eq!(body["messages"][0]["content"][1]["type"], "image");
    }

    #[test]
    fn a_run_of_consecutive_images_collapses_into_one_placeholder_scoped_to_the_run() {
        // [A-M12] `downgrade_unsupported_images`'s collapse loop is a small state machine (`pending_placeholder`)
        // meant to fold a *run* of consecutive `{"type":"image"}` blocks into a single placeholder —
        // every existing regression here only ever fed it a single image, so the run-collapsing
        // behavior itself (as opposed to the single-image case, which a stateless per-block map would
        // also get right) was unexercised. Three consecutive images in one run, plus a second, separate
        // single-image run later in the same message, both bracketed by ordinary text: proves a run of
        // 3 collapses into exactly 1 placeholder (not 3, and not 0), and that the collapse is scoped to
        // each consecutive run rather than bleeding into the surrounding text or merging across runs.
        use crate::message::{ImageSource, Role};
        let req = ModelRequest::new(
            "some-future-anthropic-model",
            vec![Message {
                role: Role::User,
                content: vec![
                    ContentBlock::text("before"),
                    ContentBlock::Image {
                        source: ImageSource::base64("image/png", "AAAA"),
                    },
                    ContentBlock::Image {
                        source: ImageSource::base64("image/png", "BBBB"),
                    },
                    ContentBlock::Image {
                        source: ImageSource::base64("image/png", "CCCC"),
                    },
                    ContentBlock::text("between"),
                    ContentBlock::Image {
                        source: ImageSource::base64("image/png", "DDDD"),
                    },
                    ContentBlock::text("after"),
                ],
                model_id: None,
                error_message: None,
                aborted: false,
                usage: None,
                stop_reason: None,
            }],
            256,
        )
        .with_no_cache(true);
        let body = build_body(&req, false);
        let content = body["messages"][0]["content"].as_array().unwrap();
        let placeholder = json!({ "type": "text", "text": USER_IMAGE_PLACEHOLDER });
        assert_eq!(
            content,
            &vec![
                json!({ "type": "text", "text": "before" }),
                placeholder.clone(),
                json!({ "type": "text", "text": "between" }),
                placeholder,
                json!({ "type": "text", "text": "after" }),
            ],
            "a run of 3 images must collapse into exactly one placeholder, scoped to the run: {content:?}"
        );
    }

    #[test]
    fn long_retention_gated_off_for_unsupported_model() {
        // Even with `cache_long`, a model whose capabilities don't include long-cache retention must
        // get the default 5-minute TTL (no `ttl` field) — otherwise Anthropic 400s the turn.
        let req = ModelRequest::new("some-unknown-model", vec![Message::user("hi")], 256)
            .with_system("sys")
            .with_cache_long(true);
        let body = build_body(&req, false);
        assert!(
            body["system"][0]["cache_control"].get("ttl").is_none(),
            "unsupported model must not receive the 1h TTL"
        );
        assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn build_body_emits_thinking_config() {
        // claude-opus-4-5 predates the adaptive requirement — still the `Budget`/`enabled` shape.
        let req = ModelRequest::new("claude-opus-4-5", vec![Message::user("hi")], 8192)
            .with_thinking(4096);
        let body = build_body(&req, false);
        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["thinking"]["budget_tokens"], 4096);
        assert_eq!(body["thinking"]["display"], "summarized");
        assert!(body.get("output_config").is_none());
    }

    #[test]
    fn build_body_emits_adaptive_thinking_config() {
        // claude-opus-4-8 (our default model) requires the adaptive shape: `output_config.effort` is a
        // sibling top-level field, not nested under `thinking`, and `display` must be set explicitly or
        // Anthropic silently omits visible reasoning text.
        let req = ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 8192)
            .with_thinking(4096)
            .with_reasoning_effort(crate::transport::ReasoningEffort::High);
        let body = build_body(&req, false);
        assert_eq!(body["thinking"]["type"], "adaptive");
        assert_eq!(body["thinking"]["display"], "summarized");
        assert!(body["thinking"].get("budget_tokens").is_none());
        assert_eq!(body["output_config"]["effort"], "high");
    }

    #[test]
    fn build_body_honors_omitted_thinking_display_when_requested() {
        // pi: `thinkingDisplay: "omitted"` (anthropic-messages.ts) — skips rendering thinking text
        // (keeping the signature for replay) for faster time-to-first-text-token. Default stays
        // "summarized" unless a caller explicitly opts in, on both thinking shapes.
        let req = ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 8192)
            .with_thinking(4096)
            .with_thinking_display(crate::transport::ThinkingDisplay::Omitted)
            .with_reasoning_effort(crate::transport::ReasoningEffort::High);
        let body = build_body(&req, false);
        assert_eq!(body["thinking"]["type"], "adaptive");
        assert_eq!(body["thinking"]["display"], "omitted");

        // Same opt-in on a Budget-shape (pre-gen6) model.
        let req = ModelRequest::new("claude-opus-4-5", vec![Message::user("hi")], 8192)
            .with_thinking(4096)
            .with_thinking_display(crate::transport::ThinkingDisplay::Omitted);
        let body = build_body(&req, false);
        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["thinking"]["display"], "omitted");
    }

    #[test]
    fn with_thinking_display_is_a_no_op_without_a_prior_with_thinking_call() {
        // Setting the display mode before `thinking` is ever enabled must not panic or fabricate a
        // `thinking` config out of nothing.
        let req = ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 8192)
            .with_thinking_display(crate::transport::ThinkingDisplay::Omitted);
        assert!(req.thinking.is_none());
    }

    #[test]
    fn adaptive_effort_wire_value_is_clamped_per_model() {
        // opus-4-6 uniquely remaps xhigh to "max" (pi: "effort 'max' is only valid on Opus 4.6").
        let req = ModelRequest::new("claude-opus-4-6", vec![Message::user("hi")], 8192)
            .with_thinking(4096)
            .with_reasoning_effort(crate::transport::ReasoningEffort::XHigh);
        assert_eq!(build_body(&req, false)["output_config"]["effort"], "max");

        // sonnet-4-6 has no xhigh wire value at all — must degrade to "high", not send "xhigh" and get
        // rejected by Anthropic.
        let req = ModelRequest::new("claude-sonnet-4-6", vec![Message::user("hi")], 8192)
            .with_thinking(4096)
            .with_reasoning_effort(crate::transport::ReasoningEffort::XHigh);
        assert_eq!(build_body(&req, false)["output_config"]["effort"], "high");

        // No adaptive model has a "minimal" wire tier — always collapses to "low".
        let req = ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 8192)
            .with_thinking(4096)
            .with_reasoning_effort(crate::transport::ReasoningEffort::Minimal);
        assert_eq!(build_body(&req, false)["output_config"]["effort"], "low");
    }

    #[test]
    fn thinking_block_round_trips_into_body_for_replay() {
        // A prior assistant turn with a signed thinking block must replay verbatim — Anthropic rejects
        // a tool turn whose thinking block is missing or unsigned.
        let req = ModelRequest::new(
            "claude-opus-4-8",
            vec![
                Message::user("think then answer"),
                Message::assistant(vec![
                    ContentBlock::Thinking {
                        text: "let me reason".into(),
                        signature: "sig-abc".into(),
                    },
                    ContentBlock::text("answer"),
                ]),
                Message::user("again"),
            ],
            8192,
        );
        let body = build_body(&req, false);
        let block = &body["messages"][1]["content"][0];
        assert_eq!(block["type"], "thinking");
        assert_eq!(block["thinking"], "let me reason");
        assert_eq!(block["signature"], "sig-abc");
    }

    #[test]
    fn unsigned_thinking_block_downgrades_to_text_on_replay() {
        // A thinking block with no (or empty) signature — e.g. from an aborted stream that never
        // delivered its `signature_delta` — must not be replayed as a `thinking` block verbatim;
        // Anthropic requires a signature to accept one on a later turn.
        let req = ModelRequest::new(
            "claude-opus-4-8",
            vec![
                Message::user("think then answer"),
                Message::assistant(vec![
                    ContentBlock::Thinking {
                        text: "half-formed reasoning".into(),
                        signature: String::new(),
                    },
                    ContentBlock::text("answer"),
                ]),
                Message::user("again"),
            ],
            8192,
        );
        let body = build_body(&req, false);
        let block = &body["messages"][1]["content"][0];
        assert_eq!(block["type"], "text");
        assert_eq!(block["text"], "half-formed reasoning");
        assert!(block.get("signature").is_none());
        // The following real text block is untouched.
        assert_eq!(body["messages"][1]["content"][1]["type"], "text");
        assert_eq!(body["messages"][1]["content"][1]["text"], "answer");
    }

    #[test]
    fn unsigned_thinking_block_with_no_text_is_dropped_not_downgraded_to_an_empty_text_block() {
        // A stream aborted before any `thinking_delta` landed leaves both `signature` and `thinking`
        // empty. Downgrading that to `{"type": "text", "text": ""}` would just trade one 400 for
        // another — Anthropic's text block requires non-empty text — so it must be dropped instead.
        let req = ModelRequest::new(
            "claude-opus-4-8",
            vec![
                Message::user("think then answer"),
                Message::assistant(vec![
                    ContentBlock::Thinking {
                        text: String::new(),
                        signature: String::new(),
                    },
                    ContentBlock::text("answer"),
                ]),
                Message::user("again"),
            ],
            8192,
        );
        let body = build_body(&req, false);
        let content = body["messages"][1]["content"].as_array().unwrap();
        assert_eq!(content.len(), 1, "{content:?}");
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "answer");
    }

    #[test]
    fn a_signed_thinking_block_with_empty_or_whitespace_only_text_is_dropped_not_kept_or_downgraded()
     {
        // [A-M1] pi (`anthropic-messages.ts`'s `convertMessages`) checks
        // `block.thinking.trim().length === 0` *before* the signature check — an empty-text thinking
        // block is dropped outright regardless of whether it carries a signature. A prior version of
        // this function checked `signed` first and returned early, so a *signed* empty/whitespace-only
        // block would survive verbatim as `{"type": "thinking", "thinking": "", "signature": "sig"}` —
        // a shape Anthropic's `thinking` content block (which requires non-empty text, same as `text`)
        // rejects just as it would reject the unsigned case this file already covered. Two signed
        // blocks — genuinely empty and whitespace-only — must both be dropped, not kept and not
        // downgraded to an empty `text` block.
        let req = ModelRequest::new(
            "claude-opus-4-8",
            vec![
                Message::user("think then answer"),
                Message::assistant(vec![
                    ContentBlock::Thinking {
                        text: String::new(),
                        signature: "sig-empty".into(),
                    },
                    ContentBlock::Thinking {
                        text: "   ".into(),
                        signature: "sig-whitespace".into(),
                    },
                    ContentBlock::text("answer"),
                ]),
                Message::user("again"),
            ],
            8192,
        );
        let body = build_body(&req, false);
        let content = body["messages"][1]["content"].as_array().unwrap();
        assert_eq!(content.len(), 1, "{content:?}");
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "answer");
    }

    #[test]
    fn decodes_thinking_then_text_stream() {
        const THINKING: &str = r#"
event: message_start
data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":5,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"step one"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"SIG"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"answer"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":9}}

event: message_stop
data: {"type":"message_stop"}
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, THINKING).unwrap();
        assert!(events.contains(&StreamEvent::ThinkingDelta {
            index: 0,
            text: "step one".into()
        }));
        assert!(events.contains(&StreamEvent::SignatureDelta {
            index: 0,
            signature: "SIG".into()
        }));
    }

    #[test]
    fn refusal_stop_reason_is_distinct() {
        const REFUSED: &str = r#"
event: message_start
data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":5,"output_tokens":1}}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"refusal"},"usage":{"output_tokens":2}}

event: message_stop
data: {"type":"message_stop"}
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, REFUSED).unwrap();
        assert!(events.contains(&StreamEvent::MessageStop {
            stop_reason: StopReason::Refusal
        }));
    }

    #[test]
    fn refusal_explanation_surfaces_as_text() {
        // A refusal carrying `stop_details.explanation` must surface that text (as a text delta) so
        // the caller can see *why* the model declined, rather than getting an empty turn.
        const REFUSED: &str = r#"
event: message_start
data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":5,"output_tokens":1}}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"refusal","stop_details":{"type":"refusal","explanation":"I can't help with that."}},"usage":{"output_tokens":2}}

event: message_stop
data: {"type":"message_stop"}
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, REFUSED).unwrap();
        assert!(events.contains(&StreamEvent::TextDelta {
            index: REFUSAL_EXPLANATION_INDEX,
            text: "I can't help with that.".into()
        }));
        assert!(events.contains(&StreamEvent::MessageStop {
            stop_reason: StopReason::Refusal
        }));
    }

    #[test]
    fn pause_turn_is_not_end_turn() {
        // `pause_turn` must not read as a clean `EndTurn` (which would truncate a turn the model meant
        // to continue); it maps to the non-terminal `Other`.
        const PAUSED: &str = r#"
event: message_start
data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":5,"output_tokens":1}}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"pause_turn"},"usage":{"output_tokens":2}}

event: message_stop
data: {"type":"message_stop"}
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, PAUSED).unwrap();
        assert!(events.contains(&StreamEvent::MessageStop {
            stop_reason: StopReason::Other
        }));
        assert!(!events.contains(&StreamEvent::MessageStop {
            stop_reason: StopReason::EndTurn
        }));
    }

    #[test]
    fn sensitive_stop_reason_is_not_end_turn() {
        // Content flagged by safety filters must not read as success either — it shares `Refusal`'s
        // variant (no distinct explanation to surface, unlike an actual `refusal`) rather than
        // silently collapsing into `Other`/`EndTurn`.
        const FLAGGED: &str = r#"
event: message_start
data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":5,"output_tokens":1}}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"sensitive"},"usage":{"output_tokens":2}}

event: message_stop
data: {"type":"message_stop"}
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, FLAGGED).unwrap();
        assert!(events.contains(&StreamEvent::MessageStop {
            stop_reason: StopReason::Refusal
        }));
        assert!(!events.contains(&StreamEvent::MessageStop {
            stop_reason: StopReason::EndTurn
        }));
    }

    #[test]
    fn genuinely_unknown_stop_reason_falls_back_to_other_not_end_turn() {
        // A value Anthropic might add later that we don't recognize yet must not be misread as a clean
        // completion — it's conservatively `Other` (warn!-logged, not hard-failed; see `map_stop_reason`).
        const NOVEL: &str = r#"
event: message_start
data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":5,"output_tokens":1}}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"some_future_reason"},"usage":{"output_tokens":2}}

event: message_stop
data: {"type":"message_stop"}
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, NOVEL).unwrap();
        assert!(events.contains(&StreamEvent::MessageStop {
            stop_reason: StopReason::Other
        }));
    }

    #[test]
    fn tool_choice_emitted_only_when_set() {
        use crate::message::ToolDef;
        use crate::transport::ToolChoice;
        let tools = vec![ToolDef {
            name: "read".into(),
            description: "d".into(),
            input_schema: json!({ "type": "object" }),
        }];
        // Unset → no `tool_choice` on the wire (the default request shape is untouched).
        let req = ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 64)
            .with_tools(tools.clone());
        assert!(build_body(&req, false).get("tool_choice").is_none());

        // Each variant maps to Anthropic's vocabulary (`any` for required; `{type:"tool",name}`).
        let body = build_body(
            &ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 64)
                .with_tools(tools.clone())
                .with_tool_choice(ToolChoice::Auto),
            false,
        );
        assert_eq!(body["tool_choice"], json!({ "type": "auto" }));

        let body = build_body(
            &ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 64)
                .with_tools(tools.clone())
                .with_tool_choice(ToolChoice::None),
            false,
        );
        assert_eq!(body["tool_choice"], json!({ "type": "none" }));

        let body = build_body(
            &ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 64)
                .with_tools(tools.clone())
                .with_tool_choice(ToolChoice::Required),
            false,
        );
        assert_eq!(body["tool_choice"], json!({ "type": "any" }));

        let body = build_body(
            &ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 64)
                .with_tools(tools)
                .with_tool_choice(ToolChoice::Tool("read".into())),
            false,
        );
        assert_eq!(
            body["tool_choice"],
            json!({ "type": "tool", "name": "read" })
        );
    }

    #[test]
    fn user_id_is_sent_as_metadata_user_id_when_present_and_omitted_otherwise() {
        let req = ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 64);
        assert!(build_body(&req, false).get("metadata").is_none());

        let req = ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 64)
            .with_user_id("hashed-user-abc123");
        let body = build_body(&req, false);
        assert_eq!(body["metadata"], json!({ "user_id": "hashed-user-abc123" }));
    }

    #[test]
    fn truncated_stream_is_rejected() {
        // Opens but never delivers `message_stop`.
        const TRUNCATED: &str = r#"
event: message_start
data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":5,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}
"#;
        let mut dec = Decoder::default();
        let err = decode_sse(&mut dec, TRUNCATED).unwrap_err();
        assert!(matches!(err, Error::Transport(_)));
    }

    #[test]
    fn mid_stream_error_event_surfaces() {
        const ERRORED: &str = r#"
event: message_start
data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":5,"output_tokens":1}}}

event: error
data: {"type":"error","error":{"type":"overloaded_error","message":"server overloaded"}}
"#;
        let mut dec = Decoder::default();
        let err = decode_sse(&mut dec, ERRORED).unwrap_err();
        match err {
            Error::Transport(msg) => {
                assert!(msg.contains("overloaded"), "got: {msg}");
            }
            other => panic!("expected transport error, got {other:?}"),
        }
    }

    #[test]
    fn a_non_json_trailer_after_message_stop_is_ignored_not_a_transport_error() {
        // pi: anthropic-sse-parsing.test.ts (`event: done` / `event: proxy.stats` with `data: not
        // json` after `message_stop`, asserting a clean `stopReason: "stop"`, no error). A gateway or
        // proxy appending a keepalive/stats line after the real message has already completed must
        // not fail an otherwise-successful turn — the already-emitted `MessageStop`/`Usage` events are
        // unaffected either way.
        const TRAILING_NOISE: &str = r#"
event: message_start
data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":5,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}

event: message_stop
data: {"type":"message_stop"}

event: done
data: not-json-at-all

event: proxy.stats
data: {"malformed": true, "trailing brace missing"
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, TRAILING_NOISE).unwrap();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::MessageStop { stop_reason, .. } if *stop_reason == StopReason::EndTurn)),
            "the real message's own MessageStop must still be emitted: {events:?}"
        );
    }

    #[test]
    fn a_non_json_data_line_before_message_stop_is_still_a_hard_error() {
        // The flip side of the test above: garbage arriving *before* the decoder has seen its
        // terminal event is a genuine corrupted/tampered stream, not trailing proxy noise — must still
        // fail loudly rather than silently swallowing mid-turn data loss.
        const GARBAGE_MID_STREAM: &str = r#"
event: message_start
data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":5,"output_tokens":1}}}

event: mystery
data: not-json-at-all
"#;
        let mut dec = Decoder::default();
        let err = decode_sse(&mut dec, GARBAGE_MID_STREAM).unwrap_err();
        assert!(matches!(err, Error::Transport(_)));
    }

    #[test]
    fn a_bad_backslash_escape_in_an_sse_event_body_is_repaired_not_a_hard_error() {
        // pi: anthropic-sse-parsing.test.ts:82-167 (`repairJson`, `packages/ai/src/utils/json-parse.ts`)
        // tolerates a backslash that isn't a valid JSON escape — e.g. a Windows path streamed without
        // escaping its own backslashes — by doubling it before the first parse attempt, rather than
        // failing the whole event. `\U`/`\x` are not valid JSON escapes, so this must fail a raw parse
        // and only succeed via `StreamDecoder::repairs_json`'s repair pass.
        const BAD_ESCAPE: &str = r#"
event: message_start
data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":5,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"C:\Users\x"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}

event: message_stop
data: {"type":"message_stop"}
"#;
        assert!(
            serde_json::from_str::<Value>(
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"C:\Users\x"}}"#
            )
            .is_err(),
            "fixture must actually be invalid JSON first"
        );
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, BAD_ESCAPE).unwrap();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::TextDelta { text, .. } if text == r"C:\Users\x")),
            "the stray backslashes must survive as literal backslashes, not abort the turn: {events:?}"
        );
    }

    #[test]
    fn a_raw_control_character_in_an_sse_event_body_is_repaired_not_a_hard_error() {
        // pi: anthropic-sse-parsing.test.ts:82-167. A raw control byte inside a JSON string value
        // (here a literal tab — not a newline, which would split the SSE `data:` line itself) must be
        // escaped and recovered rather than failing the event outright.
        const RAW_CONTROL_CHAR: &str = "
event: message_start
data: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"usage\":{\"input_tokens\":5,\"output_tokens\":1}}}

event: content_block_start
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}

event: content_block_delta
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"col1\tcol2\"}}

event: content_block_stop
data: {\"type\":\"content_block_stop\",\"index\":0}

event: message_delta
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":3}}

event: message_stop
data: {\"type\":\"message_stop\"}
";
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, RAW_CONTROL_CHAR).unwrap();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::TextDelta { text, .. } if text == "col1\tcol2")),
            "the raw tab must survive as an escaped-then-recovered tab, not abort the turn: {events:?}"
        );
    }

    #[test]
    fn repairs_malformed_sse_json_and_malformed_streamed_tool_json() {
        // pi: anthropic-sse-parsing.test.ts:82-167, `malformedToolJsonDelta` — the exact compound
        // fixture: an `input_json_delta` whose `partial_json` string itself contains an escaped JSON
        // object with an invalid `\H` escape *and* a raw embedded tab. This is the two-layer case pi's
        // test exists for: `parseJsonWithRepair` must repair the *outer* SSE event body (this decoder's
        // job — the bug this test guards) before the resulting `partial_json` fragment ever reaches the
        // *inner* accumulated-tool-args repair pi's test also exercises (already ported on the Rust side
        // as `agent::repair_json`, applied to the fully accumulated buffer in `Accumulator::flush_block`,
        // not retested here).
        const MALFORMED_TOOL_JSON: &str = r#"
event: message_start
data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":12,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_test","name":"edit","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"A\H\",\"text\":\"col1	col2\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}

event: message_stop
data: {"type":"message_stop"}
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, MALFORMED_TOOL_JSON).unwrap();

        assert!(
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::ToolUseStart { name, .. } if name == "edit")),
            "got: {events:?}"
        );
        // One repair pass at the outer-event layer: the invalid `\H` escape survives as a literal
        // backslash + `H`, and the raw tab survives as an actual tab byte — exactly what a *second*,
        // already-ported repair pass (`agent::repair_json`, over the fully accumulated buffer) expects
        // to receive and itself resolve into `{"path": "A\\H", "text": "col1\tcol2"}`.
        let expected_partial_json = "{\"path\":\"A\\H\",\"text\":\"col1\tcol2\"}";
        assert!(
            events.iter().any(|e| matches!(
                e,
                StreamEvent::InputJsonDelta { partial_json, .. } if partial_json == expected_partial_json
            )),
            "got: {events:?}"
        );
        assert!(
            events.iter().any(|e| matches!(
                e,
                StreamEvent::MessageStop { stop_reason } if *stop_reason == StopReason::ToolUse
            )),
            "a malformed-but-recoverable event must not abort the turn: {events:?}"
        );
    }

    #[test]
    fn max_tokens_is_clamped_when_the_live_prompt_nears_the_context_window() {
        // claude-3-5-sonnet: 200_000 context window. A ~150_000-token prompt (600_000 chars, the
        // chars/4 estimator) leaves 200_000 - 150_000 - 4_096 (CONTEXT_CLAMP_MARGIN) = 45_904 tokens of
        // headroom — less than the request's own (artificially large, to force the clamp) max_tokens.
        let big_text = "x".repeat(600_000);
        let req = ModelRequest::new("claude-3-5-sonnet", vec![Message::user(big_text)], 50_000);
        let body = build_body(&req, false);
        assert_eq!(
            body["max_tokens"], 45_904,
            "max_tokens must be clamped down to the actual remaining context headroom, not sent as \
             the static 50_000 ceiling regardless of how much of the window the prompt already fills"
        );
    }

    #[test]
    fn max_tokens_is_unchanged_for_a_prompt_nowhere_near_the_context_window() {
        // The clamp must never *raise* max_tokens, and must not needlessly reduce it either — a short
        // prompt has ample headroom, so the request's own ceiling should reach the wire untouched.
        let req = ModelRequest::new("claude-3-5-sonnet", vec![Message::user("hi")], 8_192);
        let body = build_body(&req, false);
        assert_eq!(body["max_tokens"], 8_192);
    }

    #[test]
    fn max_tokens_clamp_never_drops_below_the_configured_thinking_budget() {
        // Anthropic requires max_tokens > thinking.budget_tokens. Even when the live prompt is so
        // close to the window that the naive headroom computation would fall below that, the clamp
        // must not introduce a *second*, different 400 by clamping under the budget it's itself
        // configured to reserve — a request this close to the window is a case compaction should have
        // already caught, not one this clamp should make worse.
        let huge_text = "x".repeat(796_000); // ~199_000 estimated tokens
        let req = ModelRequest::new("claude-opus-4-5", vec![Message::user(huge_text)], 50_000)
            .with_thinking(10_000);
        let body = build_body(&req, false);
        let max_tokens = body["max_tokens"].as_u64().unwrap();
        assert!(
            max_tokens > 10_000,
            "max_tokens ({max_tokens}) must stay above the thinking budget (10_000) even under \
             extreme context pressure"
        );
        assert_eq!(
            max_tokens, 10_001,
            "expected the exact thinking-budget-plus-one floor"
        );
    }

    #[test]
    fn oauth_build_body_substitutes_claude_code_identity_and_appends_the_real_system_prompt() {
        // pi-parity: an OAuth (Claude Pro/Max) request MUST present Claude Code's own identity as the
        // first system block (`anthropic-messages.ts:916-931`) — the caller's real system prompt is
        // appended as a *second* block, never substituted for it, and both get the same cache
        // breakpoint.
        let req = ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 256)
            .with_system("be a helpful assistant");
        let body = build_body(&req, true);
        assert_eq!(body["system"][0]["type"], "text");
        assert_eq!(
            body["system"][0]["text"],
            "You are Claude Code, Anthropic's official CLI for Claude."
        );
        assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
        assert_eq!(body["system"][1]["text"], "be a helpful assistant");
        assert_eq!(body["system"][1]["cache_control"]["type"], "ephemeral");
        assert_eq!(
            body["system"].as_array().unwrap().len(),
            2,
            "no third block: got {:#?}",
            body["system"]
        );
    }

    #[test]
    fn oauth_build_body_sends_only_the_identity_block_when_no_real_system_prompt_is_set() {
        let req = ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 256);
        let body = build_body(&req, true);
        assert_eq!(
            body["system"].as_array().unwrap().len(),
            1,
            "got: {:#?}",
            body["system"]
        );
        assert_eq!(
            body["system"][0]["text"],
            "You are Claude Code, Anthropic's official CLI for Claude."
        );
    }

    #[test]
    fn non_oauth_build_body_never_substitutes_the_claude_code_identity() {
        let req = ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 256)
            .with_system("be brief");
        let body = build_body(&req, false);
        assert_eq!(body["system"][0]["text"], "be brief");
        assert_eq!(body["system"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn oauth_build_body_canonicalizes_advertised_tool_names_to_claude_code_casing() {
        // pi's `claudeCodeTools` table (`anthropic-messages.ts:72-91`): our own lowercase `bash`
        // canonicalizes to Claude Code's `Bash`; a name with no match (`get_weather`) passes through.
        let req =
            ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 256).with_tools(vec![
                ToolDef {
                    name: "bash".into(),
                    description: "run a shell command".into(),
                    input_schema: json!({ "type": "object" }),
                },
                ToolDef {
                    name: "get_weather".into(),
                    description: "look up the weather".into(),
                    input_schema: json!({ "type": "object" }),
                },
            ]);
        let body = build_body(&req, true);
        assert_eq!(body["tools"][0]["name"], "Bash");
        assert_eq!(body["tools"][1]["name"], "get_weather");
    }

    #[test]
    fn non_oauth_build_body_never_renames_tools() {
        let req =
            ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 256).with_tools(vec![
                ToolDef {
                    name: "bash".into(),
                    description: "run a shell command".into(),
                    input_schema: json!({ "type": "object" }),
                },
            ]);
        let body = build_body(&req, false);
        assert_eq!(body["tools"][0]["name"], "bash");
    }

    #[test]
    fn oauth_build_body_canonicalizes_replayed_assistant_tool_use_names_in_history() {
        // A tool_use block already sitting in history (produced on some earlier turn, always stored
        // under our own real name — see `canonicalize_tool_use_names`'s doc comment) must be renamed
        // to Claude Code's casing on replay, matching pi's `convertMessages` (`anthropic-messages.ts:
        // 1111`), exactly like a freshly advertised tool definition.
        let req = ModelRequest::new(
            "claude-opus-4-8",
            vec![
                Message::user("run ls"),
                Message::assistant(vec![ContentBlock::tool_use(
                    "toolu_1",
                    "bash",
                    json!({ "cmd": "ls" }),
                )]),
                Message::tool_result("toolu_1", "ok", false),
            ],
            256,
        );
        let body = build_body(&req, true);
        assert_eq!(body["messages"][1]["content"][0]["name"], "Bash");
    }

    #[test]
    fn a_full_oauth_round_trip_canonicalizes_the_tool_name_out_and_reverses_it_back_in() {
        // The end-to-end contract this pi-parity fix restores: a tool advertised under our own real
        // name goes out renamed to Claude Code's canonical casing, and a live `tool_use` block the model
        // streams back under that same canonical casing is decoded back to our own real name — so
        // nothing above this dialect (the agent loop, tool dispatch) ever sees Claude Code's naming.
        let req =
            ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 256).with_tools(vec![
                ToolDef {
                    name: "bash".into(),
                    description: "run a shell command".into(),
                    input_schema: json!({ "type": "object" }),
                },
            ]);
        let body = build_body(&req, true);
        assert_eq!(body["tools"][0]["name"], "Bash");

        let mut dec = Decoder::new(true, req.tools.clone());
        const SSE: &str = r#"
event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"Bash","input":{}}}
"#;
        let events = decode_sse(&mut dec, SSE).unwrap();
        assert_eq!(
            events,
            vec![StreamEvent::ToolUseStart {
                index: 0,
                id: "toolu_1".into(),
                name: "bash".into(),
            }],
            "the live tool_use block's Claude-Code-cased name must decode back to our own real name"
        );
    }

    #[test]
    fn a_non_oauth_decoder_never_reverses_tool_use_names() {
        let tools: Arc<[ToolDef]> = Arc::from(vec![ToolDef {
            name: "bash".into(),
            description: "run a shell command".into(),
            input_schema: json!({ "type": "object" }),
        }]);
        let mut dec = Decoder::new(false, tools);
        const SSE: &str = r#"
event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"Bash","input":{}}}
"#;
        let events = decode_sse(&mut dec, SSE).unwrap();
        assert_eq!(
            events,
            vec![StreamEvent::ToolUseStart {
                index: 0,
                id: "toolu_1".into(),
                name: "Bash".into(),
            }],
            "a non-OAuth decoder must pass the wire name through untouched"
        );
    }

    #[test]
    fn a_tool_use_name_with_no_advertised_match_passes_through_unchanged_even_under_oauth() {
        // `from_claude_code_name` only reverses a name it can match against the tools this turn
        // actually advertised — a name for a tool this turn never offered has nothing to reverse to,
        // so it must reach the loop as the model sent it rather than being silently dropped or altered.
        let tools: Arc<[ToolDef]> = Arc::from(vec![ToolDef {
            name: "bash".into(),
            description: "run a shell command".into(),
            input_schema: json!({ "type": "object" }),
        }]);
        let mut dec = Decoder::new(true, tools);
        const SSE: &str = r#"
event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"Read","input":{}}}
"#;
        let events = decode_sse(&mut dec, SSE).unwrap();
        assert_eq!(
            events,
            vec![StreamEvent::ToolUseStart {
                index: 0,
                id: "toolu_1".into(),
                name: "Read".into(),
            }]
        );
    }

    #[test]
    fn fast_path_handles_every_content_block_delta_kind() {
        let mut dec = Decoder::default();
        assert_eq!(
            dec.try_fast_path(
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#
            ),
            Some(vec![StreamEvent::TextDelta {
                index: 0,
                text: "hi".into()
            }])
        );
        assert_eq!(
            dec.try_fast_path(
                r#"{"type":"content_block_delta","index":2,"delta":{"type":"thinking_delta","thinking":"hmm"}}"#
            ),
            Some(vec![StreamEvent::ThinkingDelta {
                index: 2,
                text: "hmm".into()
            }])
        );
        assert_eq!(
            dec.try_fast_path(
                r#"{"type":"content_block_delta","index":2,"delta":{"type":"signature_delta","signature":"sig"}}"#
            ),
            Some(vec![StreamEvent::SignatureDelta {
                index: 2,
                signature: "sig".into()
            }])
        );
        assert_eq!(
            dec.try_fast_path(
                r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"a\":"}}"#
            ),
            Some(vec![StreamEvent::InputJsonDelta {
                index: 1,
                partial_json: "{\"a\":".into()
            }])
        );
    }

    #[test]
    fn fast_path_declines_every_non_delta_event_and_malformed_deltas() {
        let mut dec = Decoder::default();
        // Every other real event type — must fall through to `push`'s general handling.
        for payload in [
            r#"{"type":"message_start","message":{"usage":{"input_tokens":1,"output_tokens":0}}}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}"#,
            r#"{"type":"message_stop"}"#,
            r#"{"type":"ping"}"#,
            // A missing index — the malformed-input case
            // `content_block_delta_with_missing_index_is_dropped_not_misattributed_to_index_0` exercises
            // end-to-end; confirms the fast lane itself declines rather than silently defaulting.
            r#"{"type":"content_block_delta","delta":{"type":"text_delta","text":"x"}}"#,
            // An unrecognized delta kind — must decline so `push`'s own warn-and-drop still applies.
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{}}}"#,
        ] {
            assert_eq!(
                dec.try_fast_path(payload),
                None,
                "expected fast path to decline: {payload}"
            );
        }
    }

    #[test]
    fn fast_path_declines_a_coincidentally_shaped_event_with_the_wrong_outer_type() {
        // Guards `FastContentBlockDelta::kind`: an event carrying both an `index` and a
        // `delta.type`-tagged object but whose *outer* `type` isn't `content_block_delta` must not be
        // misparsed as one, even though no real Anthropic event does this today.
        let mut dec = Decoder::default();
        assert_eq!(
            dec.try_fast_path(
                r#"{"type":"content_block_start","index":0,"delta":{"type":"text_delta","text":"x"}}"#
            ),
            None
        );
    }

    /// The cross-dialect resume bug. A session started on an OpenAI model and continued on a Claude one
    /// carries `Text::id`/`Text::phase` (OpenAI Responses item ids) and `ToolUse::thought_signature`
    /// through history. Anthropic rejects unknown block fields outright — this used to 400 the entire
    /// request with `messages.N.content.0.text.id: Extra inputs are not permitted`.
    #[test]
    fn a_transcript_that_crossed_dialects_carries_no_foreign_block_fields_to_anthropic() {
        let assistant = Message::assistant(vec![
            ContentBlock::Text {
                text: "A".to_string(),
                // What `openai_responses` stamps on every assistant text block.
                id: Some("msg_010ec72a89156dda".to_string()),
                phase: Some("final".to_string()),
            },
            ContentBlock::ToolUse {
                id: "call_1".to_string(),
                name: "read".to_string(),
                input: json!({"path": "a.txt"}),
                thought_signature: Some("sig".to_string()),
            },
        ]);
        let req = ModelRequest::new("claude-opus-4-8", vec![Message::user("hi"), assistant], 64);
        let body = build_body(&req, false);

        let blocks = body["messages"][1]["content"]
            .as_array()
            .expect("assistant content");
        let text = blocks[0].as_object().expect("text block");
        assert_eq!(text.get("text").and_then(Value::as_str), Some("A"));
        assert!(
            !text.contains_key("id") && !text.contains_key("phase"),
            "a text block must reach Anthropic with no OpenAI item id: {text:?}"
        );
        let tool_use = blocks[1].as_object().expect("tool_use block");
        assert_eq!(tool_use.get("name").and_then(Value::as_str), Some("read"));
        assert!(
            !tool_use.contains_key("thought_signature"),
            "tool_use carries no thought_signature on the Anthropic wire: {tool_use:?}"
        );
    }

    /// The allowlist must not eat fields Anthropic genuinely wants. Pairs with the test above: together
    /// they pin both directions, so a future `ContentBlock` field can't silently break either one.
    #[test]
    fn pruning_keeps_every_field_anthropic_actually_needs() {
        let assistant = Message::assistant(vec![
            ContentBlock::Thinking {
                text: "hmm".to_string(),
                signature: "sig-abc".to_string(),
            },
            ContentBlock::ToolUse {
                id: "call_1".to_string(),
                name: "read".to_string(),
                input: json!({"path": "a.txt"}),
                thought_signature: None,
            },
        ]);
        let result = Message::tool_results(vec![ContentBlock::ToolResult {
            tool_use_id: "call_1".to_string(),
            content: "file contents".to_string(),
            is_error: false,
            images: Vec::new(),
        }]);
        let req = ModelRequest::new(
            "claude-opus-4-8",
            vec![Message::user("hi"), assistant, result],
            64,
        );
        let body = build_body(&req, false);

        let thinking = &body["messages"][1]["content"][0];
        assert_eq!(thinking["thinking"], json!("hmm"));
        assert_eq!(
            thinking["signature"],
            json!("sig-abc"),
            "the thinking signature is load-bearing — Anthropic rejects a later tool turn without it"
        );
        let tool_use = &body["messages"][1]["content"][1];
        assert_eq!(tool_use["input"], json!({"path": "a.txt"}));
        assert_eq!(tool_use["id"], json!("call_1"));
        let tool_result = &body["messages"][2]["content"][0];
        assert_eq!(tool_result["tool_use_id"], json!("call_1"));
        assert_eq!(tool_result["is_error"], json!(false));
    }
}
