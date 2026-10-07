//! The transport every **stdio** MCP server is spoken to over — not only servers with MCP Events.
//!
//! rmcp's own `TokioChildProcess` was replaced for one reason: rmcp 3.x decodes every response into
//! an untagged union of typed results, and `CallToolResult` matches *any* object carrying `_meta`.
//! Every `2026-07-28` server attaches `_meta` (`serverInfo`) and `resultType` to every result, so a
//! custom request's result (`events/list`, `events/poll`, …) came back as an empty tool result. The
//! fix has to see the raw bytes, which only a transport of our own can. Everything else this module
//! does is what `TokioChildProcess` did, kept deliberately the same:
//!
//! - the child is spawned by the caller's `Command` (its process group, env, args) with piped
//!   stdin/stdout and inherited stderr;
//! - a line that is not UTF-8 is skipped, not fatal (rmcp's codec skipped it too);
//! - when the transport or the connection goes away, the server's stdin closes (the MCP shutdown
//!   signal) and it gets [`SHUTDOWN_GRACE`] to exit on its own; then whatever is left of its process
//!   group is killed, so anything it double-forked goes with it ([`retire`]) — on a reap, a registry
//!   rebuild, and process exit alike (every exit path of `run` and `serve` calls [`retire_all`] and
//!   waits for the sweeps).
//!
//! **One boundary.** [`rescue`] / [`unwrap_rescued`] are the whole workaround, for events and skills
//! results alike: this transport applies [`rescue`] to every stdio line, and `tools/mcp_wire.rs`'s
//! `HttpClient` applies it to `skills/*` responses over streamable HTTP; each custom-request caller
//! undoes it with [`unwrap_rescued`].

use serde_json::{Value, json};

/// How long a stdio server gets to exit after its stdin closes before it is killed.
pub const SHUTDOWN_GRACE: std::time::Duration = std::time::Duration::from_secs(3);

/// A spawned stdio server. The child and its stdin are shared — not owned by a task — so that a
/// synchronous caller (a `Drop`, a thread on the way out of the process) can close stdin and poll
/// for the exit itself, with no runtime needing to run anything.
pub struct ServerProcess {
    pgid: Option<u32>,
    child: std::sync::Mutex<tokio::process::Child>,
    stdin: std::sync::Mutex<Option<tokio::process::ChildStdin>>,
    retired: std::sync::atomic::AtomicBool,
}

/// Every stdio server spawned and not yet dropped, for [`retire_all`].
static LIVE: std::sync::Mutex<Vec<std::sync::Weak<ServerProcess>>> =
    std::sync::Mutex::new(Vec::new());

fn lock<T>(m: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl ServerProcess {
    /// Its process-group id (its pid: it leads its group).
    pub(crate) fn pgid(&self) -> Option<u32> {
        self.pgid
    }

    /// Whether the server has exited — a non-blocking wait on the child this process owns, so it
    /// is portable and can never be answered by a recycled pid. Reaps it when it has.
    fn exited(&self) -> bool {
        !matches!(lock(&self.child).try_wait(), Ok(None))
    }
}

/// End a stdio server: close its stdin now (its cue to exit), give it [`SHUTDOWN_GRACE`] to do so,
/// kill it if it has not, and kill whatever else is left in its process group — a grandchild it
/// double-forked away (a browser, say) went to init and survives a kill aimed at the server alone.
/// Idempotent.
///
/// The waiting and killing run on a tracked OS thread
/// ([`crate::tools::exec::spawn_tracked_cleanup`]), never a task: this is called from `Drop`, and
/// on the way out of the process, where no task may run again;
/// [`crate::tools::exec::wait_for_pending_group_kills`] waits for it.
pub(crate) fn retire(server: &std::sync::Arc<ServerProcess>) {
    if server
        .retired
        .swap(true, std::sync::atomic::Ordering::AcqRel)
    {
        return;
    }
    drop(lock(&server.stdin).take());
    #[cfg(unix)]
    {
        let server = server.clone();
        crate::tools::exec::spawn_tracked_cleanup(move || {
            let deadline = std::time::Instant::now() + SHUTDOWN_GRACE;
            let mut reaped = server.exited();
            while !reaped && std::time::Instant::now() < deadline {
                std::thread::sleep(std::time::Duration::from_millis(25));
                reaped = server.exited();
            }
            if !reaped {
                // Not reaped, so the pid is still this server's: killing it by pid is safe.
                let _ = lock(&server.child).start_kill();
            }
            if let Some(pgid) = server.pgid {
                // The rest of the group — by group only: once the leader is reaped its pid may be
                // reused, so there is no kill-by-pid fallback here.
                crate::tools::exec::kill_group_members(pgid);
            }
            if !reaped {
                for _ in 0..40 {
                    if server.exited() {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(25));
                }
            }
        });
    }
}

/// The way out of the process: [`retire_all`], then wait for those sweeps (and any `bash` group
/// kill still in flight). `process::exit`, a return from `main` and a panic unwinding out of it
/// would otherwise end the sweep threads mid-kill, orphaning exactly the grandchildren they exist
/// to reap. Bounded by a server's grace plus a margin for the sweep — so exiting takes a little
/// longer whenever a stdio server ran (it is given the chance to exit cleanly), and up to that
/// bound when one ignores its stdin closing.
///
/// Every streamable-HTTP MCP session still open is ended too (`mcp_http_exit`): its `DELETE`s go out
/// first, in parallel and bounded by [`CLOSE_DEADLINE`](crate::tools::mcp_http_exit::CLOSE_DEADLINE),
/// overlapping the stdio grace.
pub fn sweep_before_exit() {
    let http_closed =
        crate::tools::mcp_http_exit::begin_close_all(crate::tools::mcp_http_exit::CLOSE_DEADLINE);
    retire_all();
    http_closed();
    #[cfg(unix)]
    crate::tools::exec::wait_for_pending_group_kills(
        SHUTDOWN_GRACE + std::time::Duration::from_secs(2),
    );
}

/// Runs [`sweep_before_exit`] when dropped — at the end of `main`, and while a panic unwinds out of
/// it, where a `Drop` that only *started* a sweep thread would have it killed by the exit.
pub struct ExitSweep;

impl Drop for ExitSweep {
    fn drop(&mut self) {
        sweep_before_exit();
    }
}

/// Retire every stdio server still running — for the way out of the process, where a connection
/// some owner still holds would otherwise never be dropped. Follow it with
/// [`crate::tools::exec::wait_for_pending_group_kills`] ([`sweep_before_exit`] does both).
pub fn retire_all() {
    let live: Vec<_> = lock(&LIVE)
        .iter()
        .filter_map(std::sync::Weak::upgrade)
        .collect();
    for server in &live {
        retire(server);
    }
}

/// The key a rescued result is wrapped under on its way through rmcp; see [`rescue`].
const RESCUE_KEY: &str = "x-beyond-raw-result";

/// Undo [`rescue`]'s wrapping, if it happened.
pub(crate) fn unwrap_rescued(v: Value) -> Value {
    match v {
        Value::Object(mut m) if m.len() == 1 && m.contains_key(RESCUE_KEY) => {
            m.remove(RESCUE_KEY).unwrap_or(Value::Null)
        }
        other => other,
    }
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    hay.windows(needle.len()).any(|w| w == needle)
}

/// The cheap byte test that decides whether a line is parsed at all. A line can only be lossy if it
/// is a response carrying `_meta`; and an ordinary tool result (it always has a `content` array) is
/// skipped without parsing unless it also carries one of the events extension's own result keys.
pub(crate) fn needs_check(line: &[u8]) -> bool {
    if !contains(line, b"\"result\"") || !contains(line, b"\"_meta\"") {
        return false;
    }
    if !contains(line, b"\"content\":[") && !contains(line, b"\"content\": [") {
        return true;
    }
    [
        &b"\"events\""[..],
        b"\"nextPollMs\"",
        b"\"refreshBefore\"",
        b"\"hasMore\"",
    ]
    .iter()
    .any(|k| contains(line, k))
}

/// Keep rmcp 3.x from silently dropping a custom request's result: `Some(rewritten)` for exactly
/// the lossy case — a response whose result rmcp would read as a `CallToolResult` yet carries none
/// of that type's own fields and does carry others — wrapped as `{"x-beyond-raw-result": <result>}`,
/// which only rmcp's catch-all can match. `None` leaves the line untouched (the common case,
/// decided by [`needs_check`] without parsing).
pub(crate) fn rescue(line: &[u8]) -> Option<Vec<u8>> {
    if !needs_check(line) {
        return None;
    }
    let mut msg = serde_json::from_slice::<Value>(line).ok()?;
    let result = msg.get("result")?.as_object()?;
    msg.get("id")?;
    let lossy = !["content", "structuredContent", "isError"]
        .iter()
        .any(|k| result.contains_key(*k))
        && result.keys().any(|k| k != "_meta" && k != "resultType")
        && matches!(
            serde_json::from_value::<rmcp::model::ServerResult>(Value::Object(result.clone())),
            Ok(rmcp::model::ServerResult::CallToolResult(_))
        );
    if !lossy {
        return None;
    }
    let inner = msg["result"].take();
    msg["result"] = json!({ RESCUE_KEY: inner });
    Some(msg.to_string().into_bytes())
}

// ---- per-message caps -------------------------------------------------------------------------------

/// The most one inbound message from a stdio server may be before the host refuses it, in bytes
/// (`BEYOND_AI_AGENT_MCP_MAX_MESSAGE_BYTES` overrides). A line past it is discarded as it streams in —
/// never buffered whole — and the request it answered gets a JSON-RPC error instead, so one runaway
/// server response cannot take the host's memory with it.
pub const DEFAULT_MAX_MESSAGE_BYTES: usize = 64 * 1024 * 1024;

/// The per-message cap in force — the one cap every MCP transport applies: stdio lines here, every
/// streamable-HTTP body and SSE event (`mcp_wire::HttpClient`), and the MCP Events direct-HTTP wire.
pub(crate) fn max_message_bytes() -> usize {
    std::env::var("BEYOND_AI_AGENT_MCP_MAX_MESSAGE_BYTES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_MESSAGE_BYTES)
}

/// How much of a line's start is scanned for its top-level JSON-RPC members ([`scan_head`]). A
/// server's serializer puts `id` ahead of the `result` it answers with (serde_json, rmcp and the
/// Python and TypeScript SDKs all do); one that does not gets no synthesized answer.
const ID_WINDOW: usize = 4096;

/// Per-connection message caps: the global one, and a tighter one for the responses to requests the
/// host flagged as it wrote them — today an MCP App's `resources/read` of a `ui://` view, capped at
/// the largest view the host will show — so an oversized view is refused as it streams in rather
/// than read whole and refused after.
#[derive(Default)]
pub(crate) struct MessageCaps {
    /// Request id (its JSON text) → that response's cap.
    tight: std::sync::Mutex<std::collections::HashMap<String, usize>>,
    /// Outbound bytes not yet ended by a newline.
    outbound: std::sync::Mutex<Vec<u8>>,
}

impl MessageCaps {
    /// Note bytes written to the server; a completed request line that reads a `ui://` resource gets
    /// the view cap. Only lines that mention `resources/read` are parsed.
    fn wrote(&self, bytes: &[u8]) {
        let mut pending = lock(&self.outbound);
        pending.extend_from_slice(bytes);
        while let Some(end) = pending.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = pending.drain(..=end).collect();
            if !contains(&line, b"resources/read") {
                continue;
            }
            let Ok(msg) = serde_json::from_slice::<Value>(&line) else {
                continue;
            };
            let is_view = msg["method"] == "resources/read"
                && msg["params"]["uri"]
                    .as_str()
                    .is_some_and(crate::tools::mcp_apps::is_ui_uri);
            if is_view && let Some(id) = msg.get("id") {
                lock(&self.tight).insert(id.to_string(), crate::tools::mcp_apps::MAX_VIEW_BYTES);
            }
        }
    }

    /// The cap for the response with this id (its JSON text), and forget it.
    fn take(&self, id: Option<&str>) -> Option<usize> {
        id.and_then(|id| lock(&self.tight).remove(id))
    }

    fn peek(&self, id: &str) -> Option<usize> {
        lock(&self.tight).get(id).copied()
    }
}

/// What a message's first bytes say about its **top-level** object — read by walking that object's
/// members in order (strings, escapes and nested values skipped whole), never by searching for a
/// key's text, which a nested `"id":` inside `params` or `result` would match.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct Head {
    /// The top-level `id`, as its JSON text, if reached.
    id: Option<String>,
    /// A top-level `result` or `error` member was reached: the message is a response.
    response: bool,
    /// The top-level `method`, if reached: the message is a request or a notification.
    method: Option<String>,
}

impl Head {
    /// The id of the host request this message answers — only when the prefix proves it: a
    /// top-level `id` seen before a top-level `result`/`error`, and no `method`. Anything less (an
    /// id past the window, after the body, or on a server→client request, whose id is the
    /// *server's* and can equal one of the host's) is not answered on its behalf.
    fn reply_to(&self) -> Option<&str> {
        if self.response && self.method.is_none() {
            self.id.as_deref()
        } else {
            None
        }
    }
}

/// Scan `head` — a bounded prefix of one message — for its top-level `id`, `method` and whether it
/// is a response. Stops at the first `result`/`error` member (its value is the bulk of the message)
/// or wherever the prefix runs out or stops being a JSON object.
pub(crate) fn scan_head(head: &[u8]) -> Head {
    let mut out = Head::default();
    let ws = |mut i: usize| {
        while head.get(i).is_some_and(u8::is_ascii_whitespace) {
            i += 1;
        }
        i
    };
    let mut i = ws(0);
    if head.get(i) != Some(&b'{') {
        return out;
    }
    i += 1;
    loop {
        i = ws(i);
        if head.get(i) != Some(&b'"') {
            return out;
        }
        let Some(key_end) = string_end(head, i) else {
            return out;
        };
        let key = &head[i + 1..key_end - 1];
        i = ws(key_end);
        if head.get(i) != Some(&b':') {
            return out;
        }
        i = ws(i + 1);
        if key == b"result" || key == b"error" {
            out.response = true;
            return out;
        }
        let Some(end) = value_end(head, i) else {
            return out;
        };
        let value = || serde_json::from_slice::<Value>(&head[i..end]).ok();
        match key {
            b"id" => match value() {
                Some(v @ (Value::Number(_) | Value::String(_))) => out.id = Some(v.to_string()),
                _ => return out,
            },
            b"method" => match value() {
                Some(Value::String(m)) => out.method = Some(m),
                _ => return out,
            },
            _ => {}
        }
        i = ws(end);
        if head.get(i) != Some(&b',') {
            return out;
        }
        i += 1;
    }
}

/// A small stand-in for an over-cap **MCP Events notification** (`notifications/events/*`), built
/// from the bounded head of the message — the oversized one itself is never held. The head is read
/// **in any key order** (JSON fixes none): the top-level members are walked as far as the window
/// reaches, wherever `method`, `id`, `result`/`error` and `params` fall, and inside `params` every
/// member that appears whole is kept if it is a scalar (`cursor`, `eventId`, `name`, …) or the small
/// `_meta` (routing) — a nested payload is skipped. The stand-in is the notification with those
/// `params` plus `"$oversized": true`, so the events client skips that event, keeps what position it
/// can and tells the model it was dropped, rather than reconnecting into the same giant event.
///
/// When the head does not reach the `method` (a payload-first `params` fills the window), the
/// message is still taken for an event — `notifications/events/event` — unless the head proves
/// otherwise: a response (`result`/`error`) or a request (an `id`; notifications have none). An
/// over-cap message that is something else is far likelier to be an event than not on a connection
/// carrying events; the price of guessing wrong is one spurious gap notice, where guessing the
/// other way loses an event silently (stdio) or loops on it (HTTP). `None` for anything else (it
/// is dropped as before).
pub(crate) fn oversized_stand_in(head: &[u8]) -> Option<Value> {
    let ws = |mut i: usize| {
        while head.get(i).is_some_and(u8::is_ascii_whitespace) {
            i += 1;
        }
        i
    };
    let mut i = ws(0);
    if head.get(i) != Some(&b'{') {
        return None;
    }
    i += 1;
    let mut method: Option<String> = None;
    let mut params = serde_json::Map::new();
    loop {
        i = ws(i);
        if head.get(i) != Some(&b'"') {
            break;
        }
        let Some(key_end) = string_end(head, i) else {
            break;
        };
        let key = &head[i + 1..key_end - 1];
        i = ws(key_end);
        if head.get(i) != Some(&b':') {
            break;
        }
        i = ws(i + 1);
        match key {
            // A response, or a request: not an event, whatever else the head shows.
            b"result" | b"error" | b"id" => return None,
            b"params" => {
                let (members, end) = params_members(head, i);
                params = members;
                match end {
                    Some(end) => i = end,
                    None => break,
                }
            }
            _ => {
                let Some(end) = value_end(head, i) else {
                    break;
                };
                if key == b"method" {
                    method = Some(serde_json::from_slice::<String>(&head[i..end]).ok()?);
                }
                i = end;
            }
        }
        i = ws(i);
        if head.get(i) != Some(&b',') {
            break;
        }
        i += 1;
    }
    let method = match method {
        Some(m) if m.starts_with("notifications/events/") => m,
        Some(_) => return None,
        None => {
            // Taken for an event without seeing its method: the gap it becomes says so.
            params.insert("$ambiguous".into(), Value::Bool(true));
            "notifications/events/event".to_owned()
        }
    };
    params.insert("$oversized".into(), Value::Bool(true));
    params.insert("$host".into(), Value::from(host_mark()));
    Some(json!({ "jsonrpc": "2.0", "method": method, "params": params }))
}

/// The members of the `params` object starting at `head[at]` that appear whole within `head` and
/// are worth keeping in a stand-in — scalars, and `_meta` — in any order; and where the object
/// ends, if it ends within `head`.
fn params_members(head: &[u8], at: usize) -> (serde_json::Map<String, Value>, Option<usize>) {
    let ws = |mut i: usize| {
        while head.get(i).is_some_and(u8::is_ascii_whitespace) {
            i += 1;
        }
        i
    };
    let mut kept = serde_json::Map::new();
    if head.get(at) != Some(&b'{') {
        return (kept, value_end(head, at));
    }
    let mut j = at + 1;
    loop {
        j = ws(j);
        match head.get(j) {
            Some(b'}') => return (kept, Some(j + 1)),
            Some(b'"') => {}
            _ => return (kept, None),
        }
        let Some(k_end) = string_end(head, j) else {
            return (kept, None);
        };
        let Ok(k) = serde_json::from_slice::<String>(&head[j..k_end]) else {
            return (kept, None);
        };
        j = ws(k_end);
        if head.get(j) != Some(&b':') {
            return (kept, None);
        }
        j = ws(j + 1);
        let Some(v_end) = value_end(head, j) else {
            return (kept, None);
        };
        let value = &head[j..v_end];
        // Scalars, and the small `_meta` (routing); a whole nested payload is skipped.
        let keep = !matches!(value.first(), Some(b'{' | b'[')) || k == "_meta";
        if keep && let Ok(v) = serde_json::from_slice::<Value>(value) {
            kept.insert(k, v);
        }
        j = ws(v_end);
        match head.get(j) {
            Some(b',') => j += 1,
            Some(b'}') => return (kept, Some(j + 1)),
            _ => return (kept, None),
        }
    }
}

/// A per-process secret the host stamps into the notifications it makes up itself (an over-cap
/// event's stand-in, `$host`), so they can be told from a server's: a server never sees it, so its
/// own `$`-keys cannot pass for the host's. Seeded from the OS's randomness (std's `RandomState`).
pub(crate) fn host_mark() -> u64 {
    use std::hash::{BuildHasher as _, Hasher as _};
    static MARK: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MARK.get_or_init(|| {
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u64(0x6d63_705f_686f_7374);
        h.finish()
    })
}

/// A server notification's `params` with the host-reserved `$`-keys (`$oversized`, `$ambiguous`,
/// `$dropped_id`, …) stripped — unless it is one the host made up itself (its `$host` is
/// [`host_mark`]), which keeps them (`$host` itself is dropped either way). Applied wherever
/// events notifications come in, so a server cannot forge a gap, or pre-empt the notice of a real
/// drop, with keys of its own.
pub(crate) fn host_params(mut params: Value) -> Value {
    if let Value::Object(p) = &mut params {
        let ours = p.remove("$host").and_then(|v| v.as_u64()) == Some(host_mark());
        if !ours {
            p.retain(|k, _| !k.starts_with('$'));
        }
    }
    params
}

/// The index just past the JSON string opening at `buf[at]`, if it closes within `buf`.
fn string_end(buf: &[u8], at: usize) -> Option<usize> {
    let mut j = at + 1;
    while let Some(&b) = buf.get(j) {
        match b {
            b'\\' => j += 2,
            b'"' => return Some(j + 1),
            _ => j += 1,
        }
    }
    None
}

/// The index just past the JSON value starting at `buf[at]`, if it ends within `buf` (a scalar
/// running to the very end may continue past it, so it does not count).
fn value_end(buf: &[u8], at: usize) -> Option<usize> {
    match buf.get(at)? {
        b'"' => string_end(buf, at),
        b'{' | b'[' => {
            let mut depth = 0usize;
            let mut j = at;
            while let Some(&b) = buf.get(j) {
                match b {
                    b'"' => {
                        j = string_end(buf, j)?;
                        continue;
                    }
                    b'{' | b'[' => depth += 1,
                    b'}' | b']' => {
                        depth -= 1;
                        if depth == 0 {
                            return Some(j + 1);
                        }
                    }
                    _ => {}
                }
                j += 1;
            }
            None
        }
        _ => {
            let len = buf[at..]
                .iter()
                .position(|b| matches!(b, b',' | b'}' | b']') || b.is_ascii_whitespace())?;
            Some(at + len)
        }
    }
}

/// One inbound line, read without ever buffering more than its cap.
#[derive(Debug, PartialEq)]
pub(crate) enum Inbound {
    Line(Vec<u8>),
    /// Over its cap: discarded as it streamed in. `reply_to` is the host request it provably
    /// answered ([`Head::reply_to`]) — the only case the host answers in its place; `method` is
    /// set when it was a server→client request or notification, which is dropped.
    Refused {
        reply_to: Option<String>,
        method: Option<String>,
        cap: usize,
        /// For an over-cap MCP Events notification, its stand-in ([`oversized_stand_in`]).
        stand_in: Option<Value>,
    },
    Eof,
}

/// Read one `\n`-terminated line from `reader`, holding at most `max(cap, ID_WINDOW)` bytes of it.
/// The cap is the global one, or a tighter one once the line's first [`ID_WINDOW`] bytes prove it
/// answers a capped request ([`Head::reply_to`]). `peak` reports the most bytes held at once.
pub(crate) async fn read_capped<R: tokio::io::AsyncBufRead + Unpin>(
    reader: &mut R,
    caps: &MessageCaps,
    global: usize,
    peak: &mut usize,
) -> std::io::Result<Inbound> {
    use tokio::io::AsyncBufReadExt;
    let mut buf: Vec<u8> = Vec::new();
    let mut cap = global;
    // What the line's first bytes say, scanned once they are in (or the line ended sooner).
    let mut head: Option<Head> = None;
    // Set once the line is over its cap. The rest is discarded unread.
    let mut discarding: Option<Head> = None;
    // An over-cap events notification's stand-in, built from its head before that is let go.
    let mut stand_in: Option<Value> = None;
    let refused = |head: Head, cap: usize, stand_in: Option<Value>| {
        caps.take(head.reply_to());
        Inbound::Refused {
            reply_to: head.reply_to().map(str::to_owned),
            method: head.method,
            cap,
            stand_in,
        }
    };
    loop {
        let avail = reader.fill_buf().await?;
        if avail.is_empty() {
            return Ok(match discarding {
                Some(head) => refused(head, cap, stand_in.take()),
                None if buf.is_empty() => Inbound::Eof,
                None => Inbound::Line(buf),
            });
        }
        let (chunk, done) = match avail.iter().position(|&b| b == b'\n') {
            Some(i) => (&avail[..=i], true),
            None => (avail, false),
        };
        let n = chunk.len();
        if discarding.is_none() {
            buf.extend_from_slice(chunk);
            if head.is_none() && (buf.len() >= ID_WINDOW || done) {
                let scanned = scan_head(&buf[..buf.len().min(ID_WINDOW)]);
                if let Some(tight) = scanned.reply_to().and_then(|id| caps.peek(id)) {
                    cap = cap.min(tight);
                }
                head = Some(scanned);
            }
            // Over the window means scanned above, so `head` is always `Some` here.
            if buf.len() > cap.max(ID_WINDOW) || (done && buf.len() > cap) {
                let scanned = head.take().unwrap_or_default();
                if scanned.reply_to().is_none() {
                    stand_in = oversized_stand_in(&buf[..buf.len().min(ID_WINDOW)]);
                }
                discarding = Some(scanned);
                buf = Vec::new();
            }
        }
        *peak = (*peak).max(buf.len());
        reader.consume(n);
        if done {
            return Ok(match discarding {
                Some(head) => refused(head, cap, stand_in.take()),
                None => {
                    caps.take(head.as_ref().and_then(Head::reply_to));
                    Inbound::Line(buf)
                }
            });
        }
    }
}

/// The transport's write half: writes go to the server's stdin while it is open. Dropping it (rmcp
/// dropped the transport) retires the server.
struct StdinGuard {
    server: std::sync::Arc<ServerProcess>,
    caps: std::sync::Arc<MessageCaps>,
}

impl Drop for StdinGuard {
    fn drop(&mut self) {
        retire(&self.server);
    }
}

fn closed() -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::BrokenPipe,
        "the server's stdin is closed",
    )
}

impl tokio::io::AsyncWrite for StdinGuard {
    fn poll_write(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        let written = match lock(&self.server.stdin).as_mut() {
            Some(stdin) => std::pin::Pin::new(stdin).poll_write(cx, buf),
            None => std::task::Poll::Ready(Err(closed())),
        };
        if let std::task::Poll::Ready(Ok(n)) = &written {
            self.caps.wrote(&buf[..*n]);
        }
        written
    }

    fn poll_flush(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        match lock(&self.server.stdin).as_mut() {
            Some(stdin) => std::pin::Pin::new(stdin).poll_flush(cx),
            None => std::task::Poll::Ready(Ok(())),
        }
    }

    fn poll_shutdown(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        match lock(&self.server.stdin).as_mut() {
            Some(stdin) => std::pin::Pin::new(stdin).poll_shutdown(cx),
            None => std::task::Poll::Ready(Ok(())),
        }
    }
}

/// Spawn a stdio MCP server; return it and an rmcp transport whose inbound lines pass through
/// [`rescue`].
///
/// One small task per server pumps its stdout into an in-memory pipe rmcp reads; it ends when the
/// server closes stdout or rmcp stops reading. Ending the server is [`retire`]'s, called when the
/// transport is dropped and when the connection is.
pub(crate) fn stdio_transport(
    mut cmd: tokio::process::Command,
) -> std::io::Result<(
    std::sync::Arc<ServerProcess>,
    (tokio::io::DuplexStream, impl tokio::io::AsyncWrite),
)> {
    use tokio::io::AsyncWriteExt;
    cmd.stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit())
        .kill_on_drop(true);
    let mut child = cmd.spawn()?;
    let pgid = child.id();
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| std::io::Error::other("no stdout"))?;
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| std::io::Error::other("no stdin"))?;
    let server = std::sync::Arc::new(ServerProcess {
        pgid,
        child: std::sync::Mutex::new(child),
        stdin: std::sync::Mutex::new(Some(stdin)),
        retired: std::sync::atomic::AtomicBool::new(false),
    });
    {
        let mut live = lock(&LIVE);
        live.retain(|w| w.strong_count() > 0);
        live.push(std::sync::Arc::downgrade(&server));
    }
    let (rmcp_side, mut pump_side) = tokio::io::duplex(64 * 1024);
    let caps = std::sync::Arc::new(MessageCaps::default());
    let pump_caps = caps.clone();
    tokio::spawn(async move {
        let mut reader = tokio::io::BufReader::new(stdout);
        let global = max_message_bytes();
        let mut peak = 0;
        loop {
            let line = match read_capped(&mut reader, &pump_caps, global, &mut peak).await {
                Ok(Inbound::Line(line)) => line,
                Ok(Inbound::Refused {
                    reply_to,
                    method,
                    cap,
                    stand_in,
                }) => {
                    // An over-cap MCP Events notification goes on as its small stand-in, so the
                    // subscription can skip the event and report it dropped.
                    if let Some(stand_in) = stand_in {
                        let mut line = stand_in.to_string().into_bytes();
                        line.push(b'\n');
                        if pump_side.write_all(&line).await.is_err() {
                            break;
                        }
                        continue;
                    }
                    // Answer in the server's place only for a response that provably answers a host
                    // request. A server→client request or notification is dropped (its id is the
                    // server's, and an error under it could fail an unrelated host request), as is a
                    // message whose id the prefix does not prove — its request times out instead.
                    tracing::warn!(
                        reply_to = reply_to.as_deref(),
                        method = method.as_deref(),
                        cap,
                        "refused an MCP server message over its size cap"
                    );
                    let Some(id) = reply_to else { continue };
                    let error = format!(
                        "{{\"jsonrpc\":\"2.0\",\"id\":{id},\"error\":{{\"code\":-32000,\"message\":\"MCP message over {cap} bytes refused by the host\"}}}}\n"
                    );
                    if pump_side.write_all(error.as_bytes()).await.is_err() {
                        break;
                    }
                    continue;
                }
                Ok(Inbound::Eof) | Err(_) => break,
            };
            if std::str::from_utf8(&line).is_err() {
                tracing::debug!("skipping a non-UTF-8 line from an MCP server's stdout");
                continue;
            }
            let body = line.strip_suffix(b"\n").unwrap_or(&line);
            let body = body.strip_suffix(b"\r").unwrap_or(body);
            let mut out = rescue(body).unwrap_or_else(|| body.to_vec());
            out.push(b'\n');
            if pump_side.write_all(&out).await.is_err() {
                break;
            }
        }
    });
    let guard = StdinGuard {
        server: server.clone(),
        caps,
    };
    Ok((server, (rmcp_side, guard)))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    /// A reader that hands out at most `step` bytes per read, like a pipe would.
    struct Trickle {
        data: Vec<u8>,
        at: usize,
        step: usize,
    }

    impl tokio::io::AsyncRead for Trickle {
        fn poll_read(
            mut self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
            buf: &mut tokio::io::ReadBuf<'_>,
        ) -> std::task::Poll<std::io::Result<()>> {
            let n = self
                .step
                .min(self.data.len() - self.at)
                .min(buf.remaining());
            let at = self.at;
            buf.put_slice(&self.data[at..at + n]);
            self.at += n;
            std::task::Poll::Ready(Ok(()))
        }
    }

    fn reader(lines: &[String]) -> tokio::io::BufReader<Trickle> {
        let data = lines
            .iter()
            .map(|l| format!("{l}\n"))
            .collect::<String>()
            .into_bytes();
        tokio::io::BufReader::with_capacity(
            8192,
            Trickle {
                data,
                at: 0,
                step: 8192,
            },
        )
    }

    #[tokio::test]
    async fn an_oversized_view_response_is_refused_while_holding_no_more_than_its_cap() {
        let caps = MessageCaps::default();
        caps.wrote(
            json!({"jsonrpc":"2.0","id":7,"method":"resources/read","params":{"uri":"ui://x/v"}})
                .to_string()
                .as_bytes(),
        );
        caps.wrote(b"\n");
        let view_cap = crate::tools::mcp_apps::MAX_VIEW_BYTES;
        // id first (as serde_json, rmcp and the Python SDK write it), then a 20 MiB view.
        let huge = format!(
            "{{\"id\":7,\"jsonrpc\":\"2.0\",\"result\":{{\"contents\":[{{\"text\":\"{}\"}}]}}}}",
            "x".repeat(20 * 1024 * 1024)
        );
        let ok = json!({"jsonrpc":"2.0","id":8,"result":{"tools":[]}}).to_string();
        let mut r = reader(&[huge, ok.clone()]);
        let mut peak = 0;
        let got = read_capped(&mut r, &caps, DEFAULT_MAX_MESSAGE_BYTES, &mut peak)
            .await
            .unwrap();
        assert_eq!(
            got,
            Inbound::Refused {
                reply_to: Some("7".into()),
                method: None,
                cap: view_cap,
                stand_in: None,
            }
        );
        assert!(
            peak <= view_cap + 8192,
            "held {peak} bytes of a refused view"
        );
        // The stream stays in sync: the next message reads normally.
        let next = read_capped(&mut r, &caps, DEFAULT_MAX_MESSAGE_BYTES, &mut peak)
            .await
            .unwrap();
        assert!(
            next == Inbound::Line(format!("{ok}\n").into_bytes()),
            "{next:?}"
        );
    }

    #[tokio::test]
    async fn a_message_over_the_global_cap_is_refused_but_an_id_past_its_body_is_not_trusted() {
        let caps = MessageCaps::default();
        // id last: past the body, so the prefix cannot prove which request this answers (a tail
        // search could land on an `"id":` inside the body). Refused, and answered by no one.
        let big = format!(
            "{{\"result\":{{\"content\":[{{\"type\":\"text\",\"text\":\"{}\"}}]}},\"jsonrpc\":\"2.0\",\"id\":\"abc\"}}",
            "y".repeat(300_000)
        );
        let mut r = reader(&[big]);
        let mut peak = 0;
        let got = read_capped(&mut r, &caps, 100_000, &mut peak)
            .await
            .unwrap();
        assert_eq!(
            got,
            Inbound::Refused {
                reply_to: None,
                method: None,
                cap: 100_000,
                stand_in: None,
            }
        );
        assert!(peak <= 100_000 + 8192, "held {peak}");
        // A string id ahead of the body is proof.
        let big = format!(
            "{{\"jsonrpc\":\"2.0\",\"id\":\"abc\",\"result\":{{\"x\":\"{}\"}}}}",
            "y".repeat(300_000)
        );
        let got = read_capped(&mut reader(&[big]), &caps, 100_000, &mut peak)
            .await
            .unwrap();
        assert_eq!(
            got,
            Inbound::Refused {
                reply_to: Some("\"abc\"".into()),
                method: None,
                cap: 100_000,
                stand_in: None,
            }
        );
        // A plain response under the cap passes untouched, and an untracked large one is not
        // held to the view cap.
        let fine = format!(
            "{{\"id\":1,\"result\":{{\"x\":\"{}\"}}}}",
            "z".repeat(6 * 1024 * 1024)
        );
        let mut r = reader(std::slice::from_ref(&fine));
        let got = read_capped(&mut r, &caps, DEFAULT_MAX_MESSAGE_BYTES, &mut peak)
            .await
            .unwrap();
        assert!(
            matches!(&got, Inbound::Line(l) if l.len() == fine.len() + 1),
            "an untracked 6 MiB message must pass whole"
        );
    }

    #[tokio::test]
    async fn an_oversized_message_with_only_a_nested_id_is_answered_by_no_one() {
        // No top-level id: the `"id":5` inside `params` is not the message's. Answering it would
        // fail whatever host request has id 5.
        let caps = MessageCaps::default();
        let big = format!(
            "{{\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\",\"params\":{{\"id\":5,\"blob\":\"{}\"}}}}",
            "n".repeat(300_000)
        );
        let mut peak = 0;
        let got = read_capped(&mut reader(&[big]), &caps, 100_000, &mut peak)
            .await
            .unwrap();
        assert_eq!(
            got,
            Inbound::Refused {
                reply_to: None,
                method: Some("notifications/progress".into()),
                cap: 100_000,
                stand_in: None,
            }
        );
        // Nor when the nested id comes before the top-level members, in a response with no id.
        let big = format!(
            "{{\"jsonrpc\":\"2.0\",\"meta\":{{\"id\":5}},\"result\":{{\"x\":\"{}\"}}}}",
            "n".repeat(300_000)
        );
        let got = read_capped(&mut reader(&[big]), &caps, 100_000, &mut peak)
            .await
            .unwrap();
        assert!(
            matches!(&got, Inbound::Refused { reply_to: None, .. }),
            "{got:?}"
        );
    }

    #[tokio::test]
    async fn an_oversized_server_to_client_request_is_dropped_not_answered() {
        // Its id is the server's own numbering, which can equal one of the host's in-flight
        // request ids: an error under it would fail that unrelated request.
        let caps = MessageCaps::default();
        let big = format!(
            "{{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"sampling/createMessage\",\"params\":{{\"x\":\"{}\"}}}}",
            "r".repeat(300_000)
        );
        let mut peak = 0;
        let got = read_capped(&mut reader(&[big]), &caps, 100_000, &mut peak)
            .await
            .unwrap();
        assert_eq!(
            got,
            Inbound::Refused {
                reply_to: None,
                method: Some("sampling/createMessage".into()),
                cap: 100_000,
                stand_in: None,
            }
        );
        // And it never steals a tight cap registered for the host's own request with that id.
        caps.wrote(
            json!({"jsonrpc":"2.0","id":3,"method":"resources/read","params":{"uri":"ui://x/v"}})
                .to_string()
                .as_bytes(),
        );
        caps.wrote(b"\n");
        let request = json!({"jsonrpc":"2.0","id":3,"method":"ping"}).to_string();
        let got = read_capped(
            &mut reader(std::slice::from_ref(&request)),
            &caps,
            100_000,
            &mut peak,
        )
        .await
        .unwrap();
        assert!(matches!(got, Inbound::Line(_)), "{got:?}");
        assert_eq!(caps.peek("3"), Some(crate::tools::mcp_apps::MAX_VIEW_BYTES));
    }

    #[test]
    fn the_head_scan_reads_only_top_level_members() {
        let head = |s: &str| scan_head(s.as_bytes());
        assert_eq!(
            head(r#"{"jsonrpc":"2.0","id":7,"result":{"#).reply_to(),
            Some("7")
        );
        // Strings with escaped quotes and braces, and nested arrays, are skipped whole.
        assert_eq!(
            head(r#" { "x" : "a\"}{\\" , "y":[1,{"id":9},"]"], "id" : "q\"1" , "error":"#)
                .reply_to(),
            Some(r#""q\"1""#)
        );
        // A truncated prefix proves nothing past where it stops.
        assert_eq!(head(r#"{"jsonrpc":"2.0","id":12"#).reply_to(), None);
        assert_eq!(head(r#"{"params":{"id":1,"#).reply_to(), None);
        // An id that is not a number or a string is not an id.
        assert_eq!(head(r#"{"id":{"n":1},"result":"#).reply_to(), None);
        // Not an object at all.
        assert_eq!(head(r#"["id",1]"#), Head::default());
    }

    #[test]
    fn rescue_keeps_a_meta_carrying_custom_result_and_leaves_typed_results_alone() {
        // What a 2026-07-28 server sends for `events/list`: rmcp alone reads this as an empty
        // `CallToolResult` and the events are gone.
        let line = json!({"jsonrpc": "2.0", "id": 7, "result": {
            "events": [{"name": "a"}], "resultType": "complete",
            "_meta": {"io.modelcontextprotocol/serverInfo": {"name": "x", "version": ""}}
        }})
        .to_string();
        let lossy: rmcp::model::ServerJsonRpcMessage = serde_json::from_str(&line).unwrap();
        assert!(
            !serde_json::to_string(&lossy).unwrap().contains("events"),
            "the rmcp behaviour this works around has changed; revisit rescue"
        );
        let rescued: rmcp::model::ServerJsonRpcMessage =
            serde_json::from_slice(&rescue(line.as_bytes()).unwrap()).unwrap();
        let rmcp::model::ServerJsonRpcMessage::Response(r) = rescued else {
            panic!("not a response")
        };
        let back = unwrap_rescued(serde_json::to_value(r.result).unwrap());
        assert_eq!(back["events"][0]["name"], "a");

        for untouched in [
            json!({"jsonrpc": "2.0", "id": 1, "result": {"content": [], "_meta": {}}}),
            json!({"jsonrpc": "2.0", "id": 2, "result": {"tools": [], "_meta": {}}}),
            json!({"jsonrpc": "2.0", "id": 3, "result": {"resultType": "complete", "_meta": {}}}),
            json!({"jsonrpc": "2.0", "method": "notifications/x", "params": {"result": 1, "_meta": {}}}),
        ] {
            assert!(rescue(untouched.to_string().as_bytes()).is_none());
        }
    }

    #[test]
    fn an_ordinary_meta_carrying_tool_result_is_not_even_parsed() {
        let tool = json!({"jsonrpc": "2.0", "id": 1, "result": {
            "content": [{"type": "text", "text": "hi"}], "_meta": {"x": 1}
        }})
        .to_string();
        assert!(!needs_check(tool.as_bytes()));
        // No `_meta`: never parsed either.
        assert!(!needs_check(
            br#"{"jsonrpc":"2.0","id":1,"result":{"events":[]}}"#
        ));
        // An events result that happens to mention `content`: still checked.
        let events = json!({"jsonrpc": "2.0", "id": 1, "result": {
            "events": [{"data": {"content": [1]}}], "_meta": {}
        }})
        .to_string();
        assert!(needs_check(events.as_bytes()));
    }

    /// `retire_all` and the pending-kill registry are process-wide: tests that use them take turns.
    static PROCESS_WIDE: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    /// The sweep after a server's leader is reaped signals its group only. A process that is not
    /// a group leader stands in for an unrelated one handed the reaped server's pid: the sweep must
    /// leave it alone (no kill-by-pid fallback).
    #[cfg(unix)]
    #[test]
    fn the_sweep_after_a_reap_never_kills_by_pid() {
        let mut bystander = std::process::Command::new("sleep")
            .arg("30")
            .spawn()
            .unwrap();
        let pid = bystander.id();
        crate::tools::exec::kill_group_members(pid);
        std::thread::sleep(std::time::Duration::from_millis(200));
        let survived = bystander.try_wait().unwrap().is_none();
        // The hazard is real: with the fallback, the same call kills it.
        crate::tools::exec::kill_process_group(pid);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let mut killed = false;
        while std::time::Instant::now() < deadline {
            if bystander.try_wait().unwrap().is_some() {
                killed = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        let _ = bystander.kill();
        let _ = bystander.wait();
        assert!(
            survived,
            "a process merely holding the pid survives the group sweep"
        );
        assert!(
            killed,
            "the kill-by-pid fallback this sweep avoids would have killed it"
        );
    }

    /// A server that ignores stdin closing is killed once its grace is up — and only then: the
    /// grace is measured on the child itself (a non-blocking wait), not by reading `/proc`, so it
    /// holds on every unix.
    #[cfg(unix)]
    #[tokio::test]
    async fn retire_gives_the_grace_then_kills_a_server_that_ignores_stdin() {
        let _serial = PROCESS_WIDE.lock().await;
        let mut cmd = tokio::process::Command::new("sh");
        cmd.args(["-c", "trap '' TERM; exec sleep 60"])
            .process_group(0);
        let (server, transport) = stdio_transport(cmd).unwrap();
        let started = std::time::Instant::now();
        retire(&server);
        drop(transport);
        // Still running well inside the grace.
        tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
        assert!(!server.exited(), "killed before its grace was up");
        let deadline = std::time::Instant::now() + SHUTDOWN_GRACE * 2;
        while !server.exited() && std::time::Instant::now() < deadline {
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
        assert!(server.exited(), "killed once the grace was up");
        assert!(started.elapsed() >= SHUTDOWN_GRACE);
    }

    /// A panic unwinding out of `main` still sweeps: [`ExitSweep`]'s `Drop` retires every server
    /// and waits for the sweeps before the unwind goes on — a server's double-forked grandchild is
    /// gone by the time the panic surfaces, not left to a thread the exit would kill.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_panic_unwinding_past_the_exit_guard_still_sweeps() {
        let _serial = PROCESS_WIDE.lock().await;
        let dir = tempfile::tempdir().unwrap();
        let pidfile = dir.path().join("orphan.pid");
        let mut cmd = tokio::process::Command::new("sh");
        cmd.args([
            "-c",
            // The grandchild's stdio is detached, so a regression leaks it without also holding
            // the test harness's output pipe open.
            &format!(
                "sleep 600 </dev/null >/dev/null 2>&1 & echo $! > {}; exec sleep 600",
                pidfile.display()
            ),
        ])
        .process_group(0);
        let (server, transport) = stdio_transport(cmd).unwrap();
        let orphan: u32 = loop {
            if let Some(pid) = std::fs::read_to_string(&pidfile)
                .ok()
                .and_then(|s| s.trim().parse().ok())
            {
                break pid;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        };
        let alive = |pid: u32| {
            std::fs::read_to_string(format!("/proc/{pid}/stat")).is_ok_and(|s| {
                s.rsplit_once(") ")
                    .is_some_and(|(_, rest)| !rest.starts_with('Z'))
            })
        };
        assert!(alive(orphan));
        let held = (server, transport);
        let unwound = tokio::task::spawn_blocking(move || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                let _sweep = ExitSweep;
                let _held = held;
                std::panic::panic_any("boom after MCP connect");
            }))
            .is_err()
        })
        .await
        .unwrap();
        assert!(unwound);
        assert!(
            !alive(orphan),
            "the panic surfaced with the server's grandchild {orphan} still running"
        );
    }

    /// An over-cap events notification's stand-in keeps what its head shows whole — routing, cursor,
    /// id — and none of the payload; anything that is not an events notification gets none.
    #[test]
    fn an_oversized_events_notification_gets_a_stand_in_from_its_head() {
        let head = br#"{"jsonrpc":"2.0","method":"notifications/events/event","params":{"_meta":{"io.modelcontextprotocol/subscriptionId":"s1"},"cursor":"41","eventId":"e9","data":{"blob":"xxxxxxxxxx"#;
        let stand_in = oversized_stand_in(head).unwrap();
        assert_eq!(stand_in["method"], "notifications/events/event");
        assert_eq!(stand_in["params"]["cursor"], "41");
        assert_eq!(stand_in["params"]["eventId"], "e9");
        assert_eq!(stand_in["params"]["$oversized"], true);
        assert_eq!(
            stand_in["params"]["_meta"]["io.modelcontextprotocol/subscriptionId"],
            "s1"
        );
        assert!(stand_in["params"].get("data").is_none(), "no payload");
        // Members past the head are simply unknown.
        let late = br#"{"jsonrpc":"2.0","method":"notifications/events/event","params":{"data":{"blob":"xxxx"#;
        let stand_in = oversized_stand_in(late).unwrap();
        assert!(stand_in["params"].get("cursor").is_none());
        for other in [
            &br#"{"jsonrpc":"2.0","method":"notifications/progress","params":{"progress":1"#[..],
            br#"{"jsonrpc":"2.0","id":3,"result":{"x":"yyyy"#,
            b"not json",
        ] {
            assert_eq!(
                oversized_stand_in(other),
                None,
                "{}",
                String::from_utf8_lossy(other)
            );
        }
    }

    /// JSON fixes no key order: the head is read wherever `method`, `id` and `params` fall.
    #[test]
    fn an_oversized_events_notification_is_recognized_in_any_key_order() {
        // `params` (whole) before `method`.
        let stand_in = oversized_stand_in(
            br#"{"params":{"cursor":"7","eventId":"e1"},"jsonrpc":"2.0","method":"notifications/events/event","x":"yyyy"#,
        )
        .unwrap();
        assert_eq!(stand_in["method"], "notifications/events/event");
        assert_eq!(stand_in["params"]["cursor"], "7");
        assert_eq!(stand_in["params"]["eventId"], "e1");
        // `params` first and the payload in it: the method is past the window, the routing and
        // cursor ahead of the payload are kept, and it is taken for an event.
        let stand_in = oversized_stand_in(
            br#"{"params":{"_meta":{"io.modelcontextprotocol/subscriptionId":"s1"},"cursor":"9","data":{"blob":"xxxx"#,
        )
        .unwrap();
        assert_eq!(stand_in["method"], "notifications/events/event");
        assert_eq!(stand_in["params"]["cursor"], "9");
        assert_eq!(
            stand_in["params"]["_meta"]["io.modelcontextprotocol/subscriptionId"],
            "s1"
        );
        // Payload first: nothing identifying is visible, and it is still skipped as an event.
        let stand_in = oversized_stand_in(br#"{"params":{"data":{"blob":"xxxx"#).unwrap();
        assert_eq!(stand_in["params"]["$oversized"], true);
        assert!(stand_in["params"].get("cursor").is_none());
        // …and, its method unseen, it is marked as possibly not an event at all; one whose method
        // was seen is not.
        assert_eq!(stand_in["params"]["$ambiguous"], true);
        let seen = oversized_stand_in(
            br#"{"jsonrpc":"2.0","method":"notifications/events/event","params":{"data":"xx"#,
        )
        .unwrap();
        assert!(seen["params"].get("$ambiguous").is_none());
        // What the head proves is not an event is not stood in for, in any order: a request (an
        // id, which notifications never carry), a response, another method.
        for other in [
            &br#"{"params":{"data":"x"},"id":4,"method":"#[..],
            br#"{"id":4,"params":{"data":{"blob":"xxxx"#,
            br#"{"params":{"progress":1},"method":"notifications/progress","jsonrpc":"#,
            br#"{"error":{"code":1,"message":"xxxx"#,
        ] {
            assert_eq!(
                oversized_stand_in(other),
                None,
                "{}",
                String::from_utf8_lossy(other)
            );
        }
    }
}
