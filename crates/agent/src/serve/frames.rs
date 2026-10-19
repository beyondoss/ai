//! Builders for the frames a session writes that more than one command shares, and `response`/`ack`.

use crate::serve::*;

/// The `data` payload `get_messages` answers with — and the identical payload an attach-time catch-up
/// is seeded from, so a reconnecting client parses one shape, not two.
///
/// Deliberately takes the messages and ids **by reference** rather than reading `session`: during a run
/// `run_events_steered` holds `&mut session` for the whole duration, so anything needing `&session`
/// simply cannot answer. Taking a snapshot instead is what lets both a mid-run `get_messages` and the
/// catch-up work at all (the same escape hatch `LiveStats` uses for `get_state`).
///
/// Each message is tagged with its tree id when persistence tracks one (parallel to `session.messages`
/// — see `Persistence::active_ids`), so a client can pick a `switch_branch` `target_id` from anywhere in
/// the visible transcript, not only from a branch's leaf (which is all `list_branches` alone can offer).
/// A length mismatch (in-memory mode, or a transient inconsistency) leaves messages untagged rather than
/// mistagging them.
///
/// `since` returns only what's new since a tree id the client already has — pi's own
/// `get_entries({since})` — so a client polling for updates doesn't re-transfer the whole transcript.
/// An unmatched `since`, or one given while nothing is tagged, is an `Err`, never a silent fallback to
/// "everything": that would look like a working incremental fetch that is actually re-sending the full
/// history, masking the bug instead of surfacing it.
///
/// `leaf_id` is the active-tip id `get_tree`'s response also carries, so a client fetching the
/// transcript needn't issue a second `get_tree` just to learn the tip.
pub(super) fn messages_payload(
    messages: &[agent_core::Message],
    msg_ids: &[String],
    since: Option<&str>,
) -> Result<Value, String> {
    let mut messages = serde_json::to_value(messages).unwrap_or(Value::Null);
    if let Value::Array(arr) = &mut messages
        && arr.len() == msg_ids.len()
    {
        for (m, mid) in arr.iter_mut().zip(msg_ids) {
            if let Value::Object(obj) = m {
                obj.insert("id".into(), json!(mid));
            }
        }
    }
    if let Some(since) = since {
        // `idx` indexes `msg_ids` (the active *tree path*, which includes non-message `Custom` entries),
        // but the slice below applies to `messages` (LLM messages only). The two align 1:1 exactly when
        // the tagging above ran — i.e. `arr.len() == msg_ids.len()`. When they don't, an `idx` past
        // `messages`' end would make `split_off(idx + 1)` panic, so fold that into the unmatched branch
        // rather than slicing blindly.
        let tagged = matches!(&messages, Value::Array(arr) if arr.len() == msg_ids.len());
        match msg_ids
            .iter()
            .position(|mid| mid == since)
            .filter(|_| tagged)
        {
            Some(idx) => {
                if let Value::Array(arr) = &mut messages {
                    *arr = arr.split_off(idx + 1);
                }
            }
            None => return Err(format!("no message with id {since} in this session")),
        }
    }
    Ok(json!({ "messages": messages, "leaf_id": msg_ids.last() }))
}

/// Serialize one [`OutFrame`] to its final JSON line (no trailing newline), as [`bytes::Bytes`]. `Raw`
/// (the hot `event`-frame path) is already the final text and moves out with no copy; only `Value` pays
/// a `serde_json::to_string`. Returns `None` only if a frame we built ourselves fails to serialize — a
/// bug, skipped rather than tearing down the stream. Shared so every transport emits byte-identical
/// frame text; each converts the `Bytes` to its own wire form at the socket edge
/// (a validation-only `Utf8Bytes` for WebSocket, a buffered `\n`-terminated write for stdio, JSON
/// body for HTTP POST).
pub(crate) fn frame_to_line(frame: OutFrame) -> Option<Bytes> {
    match frame {
        OutFrame::Raw(line) => Some(line),
        OutFrame::Value(v) => match serde_json::to_string(&v) {
            Ok(line) => Some(Bytes::from(line)),
            Err(e) => {
                eprintln!("serve: failed to serialize output frame: {e}");
                None
            }
        },
    }
}

/// Build a `login_progress` frame — an unsolicited push update for an in-flight `login`, correlated
/// to the eventual terminal `response` via the same `id`. `step` is `"browser"` (open `url`),
/// `"device_code"` (show `user_code`/`verification_uri`), `"manual_code"` (the local callback
/// listener couldn't complete — paste a code via `submit_code`), or `"progress"` (a bare narration
/// `message`).
#[allow(clippy::too_many_arguments)]
pub(super) fn login_progress_frame(
    id: Option<String>,
    provider: &str,
    step: &str,
    url: Option<&str>,
    user_code: Option<&str>,
    verification_uri: Option<&str>,
    expires_in_secs: Option<u64>,
    message: Option<&str>,
) -> OutFrame {
    let mut m = Map::new();
    m.insert("type".into(), json!("login_progress"));
    if let Some(id) = id {
        m.insert("id".into(), json!(id));
    }
    m.insert("command".into(), json!("login"));
    m.insert("provider".into(), json!(provider));
    m.insert("step".into(), json!(step));
    if let Some(url) = url {
        m.insert("url".into(), json!(url));
    }
    if let Some(user_code) = user_code {
        m.insert("user_code".into(), json!(user_code));
    }
    if let Some(verification_uri) = verification_uri {
        m.insert("verification_uri".into(), json!(verification_uri));
    }
    if let Some(expires_in) = expires_in_secs {
        m.insert("expires_in".into(), json!(expires_in));
    }
    if let Some(message) = message {
        m.insert("message".into(), json!(message));
    }
    Value::Object(m).into()
}

/// A listing's default page size, when the caller does not ask for one.
pub(super) const DEFAULT_LIST_LIMIT: usize = 50;

/// The most a caller can ask for in one page, whatever it passes.
pub(super) const MAX_LIST_LIMIT: usize = 500;

/// The `data` payload of a listing response: `query` applied, then **one page** of the result.
///
/// Shared by the four listing arms (`list_sessions`/`list_all_sessions`, idle and busy) so the
/// paging rule is stated once rather than four times.
///
/// **A listing is always paged.** It used to be unbounded in cardinality *and* each entry carried up
/// to 50 KB of that session's conversation text, so a single call asked a replica to build the
/// tenant's whole transcript set in memory and write it to one frame — and in service mode the
/// caller is a tenant holding one session's grant. `limit` defaults to [`DEFAULT_LIST_LIMIT`] and is
/// capped at [`MAX_LIST_LIMIT`]; an explicit `0` is clamped to 1, since a page of nothing is never
/// what a caller meant. `total` is the size of the match set *before* paging, so a client can page
/// without guessing when to stop.
///
/// This bounds the **response**, not the scan: every session is still stat'd to build the match set
/// (cheap against the warm listing index, and the reason that index exists). Bounding the scan means
/// an index that can seek, which is a catalog — see `crates/agent/FLEET.md`.
pub(super) fn listing_page(sessions: Vec<SessionMeta>, cmd: &Value) -> Value {
    let matched = search_sessions(sessions, cmd.get("query").and_then(Value::as_str));
    let total = matched.len();
    let offset = cmd.get("offset").and_then(Value::as_u64).unwrap_or(0) as usize;
    let limit = cmd
        .get("limit")
        .and_then(Value::as_u64)
        .map_or(DEFAULT_LIST_LIMIT, |n| {
            (n as usize).clamp(1, MAX_LIST_LIMIT)
        });
    let page: Vec<Value> = matched
        .iter()
        .skip(offset)
        .take(limit)
        .map(SessionMeta::to_listing_json)
        .collect();
    json!({ "sessions": page, "total": total })
}

/// Build a `list_progress` frame — an unsolicited progress update for an in-flight `list_sessions`/
/// `list_all_sessions` scan, correlated to the eventual `response` frame via the same request `id`.
pub(super) fn list_progress_frame(
    id: Option<String>,
    command: &str,
    scanned: usize,
    total: usize,
) -> OutFrame {
    let mut m = Map::new();
    m.insert("type".into(), json!("list_progress"));
    if let Some(id) = id {
        m.insert("id".into(), json!(id));
    }
    m.insert("command".into(), json!(command));
    m.insert("scanned".into(), json!(scanned));
    m.insert("total".into(), json!(total));
    Value::Object(m).into()
}

/// Build an `auto_retry_start` frame — an unsolicited notice that a `prompt`'s run failed with what
/// looks like a transient error and is about to be automatically retried, correlated to the eventual
/// `response` frame via the same request `id`. Sent once per attempt, immediately before the backoff
/// sleep (not after — a client watching for retry activity shouldn't have to infer it from a gap).
/// Named `auto_retry_start` (not the bare `auto_retry` this crate used before) to match pi's own wire
/// discriminator (`agent-session.ts`'s `AgentSessionEvent`), the counterpart to `auto_retry_end` below.
pub(super) fn auto_retry_frame(
    id: Option<String>,
    attempt: u32,
    max_attempts: u32,
    delay_ms: u64,
    error: &str,
) -> OutFrame {
    let mut m = Map::new();
    m.insert("type".into(), json!("auto_retry_start"));
    if let Some(id) = id {
        m.insert("id".into(), json!(id));
    }
    m.insert("command".into(), json!("prompt"));
    m.insert("attempt".into(), json!(attempt));
    m.insert("max_attempts".into(), json!(max_attempts));
    m.insert("delay_ms".into(), json!(delay_ms));
    m.insert("error".into(), json!(error));
    Value::Object(m).into()
}

/// Build an `auto_retry_end` frame — the terminal notice for a whole-run retry sequence that made at
/// least one attempt: either a later attempt succeeded (`success: true`), or the sequence gave up
/// (`success: false`, `final_error` set) — including the backoff itself being interrupted by
/// `abort`/`abort_retry` (`final_error: "retry cancelled"`). Mirrors pi's own `auto_retry_end` event.
/// Never sent when no retry ever started (a first-attempt success, or a failure that was never
/// retryable in the first place, needs no "the retries ended" notice — nothing began to end).
pub(super) fn auto_retry_end_frame(
    id: Option<String>,
    success: bool,
    attempt: u32,
    final_error: Option<&str>,
) -> OutFrame {
    let mut m = Map::new();
    m.insert("type".into(), json!("auto_retry_end"));
    if let Some(id) = id {
        m.insert("id".into(), json!(id));
    }
    m.insert("command".into(), json!("prompt"));
    m.insert("success".into(), json!(success));
    m.insert("attempt".into(), json!(attempt));
    if let Some(err) = final_error {
        m.insert("final_error".into(), json!(err));
    }
    Value::Object(m).into()
}

/// Build a `session_info_changed` frame — an unsolicited push notice that the session's title changed,
/// correlated to the triggering `set_session_name` request via the same `id`. pi's own
/// `session_info_changed` event (`rpc-mode.ts:632-639`) lets a client learn the final *sanitized* title
/// without a follow-up `get_state`; `title` is `None` when the sanitized result was empty (a caller can
/// explicitly clear a title — see `sanitize_title`/`title_or_clear`).
pub(super) fn session_info_changed_frame(id: Option<String>, title: Option<String>) -> OutFrame {
    let mut m = Map::new();
    m.insert("type".into(), json!("session_info_changed"));
    if let Some(id) = id {
        m.insert("id".into(), json!(id));
    }
    m.insert("command".into(), json!("set_session_name"));
    m.insert("title".into(), json!(title));
    Value::Object(m).into()
}

/// Filter `nodes` (the full, chronologically ordered [`crate::session_store::TreeNode`] list — see
/// `SessionStore::tree`'s own doc comment: every node, every branch, sorted by `(timestamp, id)`) down
/// to only the entries appended after `since`, a tree id the caller already has. Task #48 (pi-parity
/// gap): pi's own `SessionManager.getEntries({since})` backs both its `get_tree` and its dedicated
/// `get_entries` RPC command (`rpc-types.ts:63`, `rpc-mode.ts:609-620`), letting a client fetch only
/// what's new across *every* entry type (message/branch_summary/compaction/custom — `tree()` already
/// folds all of these into one list, unlike `get_messages`'s own `since`, which only ever sees the
/// active path's plain LLM messages). Since `tree()` is already in append order, finding `since`'s own
/// position in that same list and returning everything after it is enough — no separate incremental
/// index needed. `Err` (not a silent full re-fetch) when `since` matches no node in `nodes` — the same
/// "surface the bug, don't mask it" contract `get_messages`'s own `since` already established, rather
/// than silently returning everything a naive fallback would.
pub(super) fn nodes_since(
    mut nodes: Vec<crate::session_store::TreeNode>,
    since: &str,
) -> Result<Vec<crate::session_store::TreeNode>, String> {
    match nodes.iter().position(|n| n.id == since) {
        Some(idx) => Ok(nodes.split_off(idx + 1)),
        None => Err(format!("no entry with id {since} in this session")),
    }
}

/// Build a `response` frame.
pub(super) fn response(
    id: Option<String>,
    command: &str,
    success: bool,
    data: Option<Value>,
    error: Option<&str>,
) -> OutFrame {
    let mut m = Map::new();
    m.insert("type".into(), json!("response"));
    if let Some(id) = id {
        m.insert("id".into(), json!(id));
    }
    m.insert("command".into(), json!(command));
    m.insert("success".into(), json!(success));
    if let Some(d) = data {
        m.insert("data".into(), d);
    }
    if let Some(e) = error {
        m.insert("error".into(), json!(e));
    }
    Value::Object(m).into()
}

/// A lightweight acknowledgement frame, emitted the moment a `prompt` is queued — before the model
/// turn(s) actually run — so a client can distinguish "received and starting" from the eventual
/// terminal `response` (which may be seconds away on a long tool-heavy run).
pub(super) fn ack(id: Option<String>, command: &str) -> OutFrame {
    let mut m = Map::new();
    m.insert("type".into(), json!("ack"));
    if let Some(id) = id {
        m.insert("id".into(), json!(id));
    }
    m.insert("command".into(), json!(command));
    Value::Object(m).into()
}

pub(super) fn event_frame(ev: AgentEvent) -> Option<OutFrame> {
    #[derive(serde::Serialize)]
    struct EventFrame<'a> {
        r#type: &'static str,
        event: &'a AgentEvent,
    }
    let line = serde_json::to_string(&EventFrame {
        r#type: "event",
        event: &ev,
    })
    .inspect_err(|e| eprintln!("serve: failed to serialize agent event: {e}"))
    .ok()?;
    // Hand the freshly-built `String`'s buffer to `Bytes` by move — no copy — so every downstream
    // clone (turn recording, each extra sink) is a refcount bump.
    Some(OutFrame::Raw(Bytes::from(line)))
}

/// Write one newline-delimited frame to stdout and flush it.
/// `line` must already carry its own trailing `\n` — folded into the one caller's buffer rather than a
/// second `write_all` here, so a frame goes out as a single write instead of two.
pub(super) async fn write_frame(out: &mut tokio::io::Stdout, line: &[u8]) -> std::io::Result<()> {
    out.write_all(line).await?;
    out.flush().await
}
