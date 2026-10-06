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
//! - when rmcp drops the transport, the server's stdin closes (the MCP shutdown signal) and it gets
//!   [`SHUTDOWN_GRACE`] to exit on its own before it is killed; [`retire`] then sweeps its process
//!   group, so anything it double-forked goes with it — on a reap, a registry rebuild, and process
//!   exit alike (`serve` calls [`settle`] and waits for the sweeps before it exits).
//!
//! **One boundary.** [`rescue`] / [`unwrap_rescued`] are the whole workaround. PR #131 (Skills) carries
//! its own rmcp `_meta` workaround in `tools/mcp_wire.rs`; unifying the two is meant to be a swap of
//! these two functions, nothing else.

use serde_json::{Value, json};
use tokio::sync::oneshot;

/// How long a stdio server gets to exit after its stdin closes before it is killed.
pub const SHUTDOWN_GRACE: std::time::Duration = std::time::Duration::from_secs(3);

/// Stdout pumps still running — servers whose shutdown has not finished. [`settle`] waits on it.
static LIVE_PUMPS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static PUMPS_CHANGED: tokio::sync::Notify = tokio::sync::Notify::const_new();

/// One running pump, counted in [`LIVE_PUMPS`] for exactly as long as it lives.
struct LivePump;

impl LivePump {
    fn new() -> Self {
        LIVE_PUMPS.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        LivePump
    }
}

impl Drop for LivePump {
    fn drop(&mut self) {
        LIVE_PUMPS.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
        PUMPS_CHANGED.notify_waiters();
    }
}

/// Wait (up to `timeout`) for every stdio server whose connection was dropped to finish shutting
/// down: stdin closed, its [`SHUTDOWN_GRACE`] given, exited or killed. Called on the way out of the
/// process, before `std::process::exit`, so the grace is real: on a current-thread runtime nothing
/// else would poll rmcp's tasks (which close stdin) or the pumps once the caller blocks.
pub async fn settle(timeout: std::time::Duration) {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let changed = PUMPS_CHANGED.notified();
        if LIVE_PUMPS.load(std::sync::atomic::Ordering::Acquire) == 0 {
            return;
        }
        if tokio::time::timeout_at(deadline, changed).await.is_err() {
            return;
        }
    }
}

/// Sweep a dropped stdio server's process group — anything it double-forked away (a browser, say)
/// went to init and survives a kill aimed at the server alone.
///
/// The sweep waits first, up to [`SHUTDOWN_GRACE`] (plus a beat), for the server itself to exit:
/// dropping its connection closed its stdin, which is its cue, and a well-behaved server closes its
/// browser on the way out. Whatever is left is then killed, leader included — so this is complete
/// even when the pump that would have killed the leader never runs again (process exit).
///
/// On a tracked OS thread ([`crate::tools::exec::spawn_tracked_cleanup`]), never a task: it is
/// called from `Drop`, and `std::process::exit` waits for it via
/// [`crate::tools::exec::wait_for_pending_group_kills`].
pub(crate) fn retire(pgid: Option<u32>) {
    #[cfg(unix)]
    if let Some(pgid) = pgid {
        crate::tools::exec::spawn_tracked_cleanup(move || {
            let deadline =
                std::time::Instant::now() + SHUTDOWN_GRACE + std::time::Duration::from_millis(500);
            while leader_running(pgid) && std::time::Instant::now() < deadline {
                std::thread::sleep(std::time::Duration::from_millis(25));
            }
            crate::tools::exec::kill_process_group(pgid);
        });
    }
}

/// Whether the group's leader (the server; its pid is the group id) is still running. A zombie —
/// exited, not yet reaped — is not.
fn leader_running(pid: u32) -> bool {
    std::fs::read(format!("/proc/{pid}/stat")).is_ok_and(|stat| {
        stat.iter()
            .rposition(|&b| b == b')')
            .and_then(|i| stat.get(i + 2))
            .is_some_and(|&state| state != b'Z' && state != b'X')
    })
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

/// Closes over the server's stdin. When rmcp drops its transport, dropping this closes stdin (the
/// server's cue to exit) and tells the pump to start the grace window.
struct StdinGuard {
    stdin: tokio::process::ChildStdin,
    _closed: oneshot::Sender<()>,
}

impl tokio::io::AsyncWrite for StdinGuard {
    fn poll_write(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        std::pin::Pin::new(&mut self.stdin).poll_write(cx, buf)
    }

    fn poll_flush(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::pin::Pin::new(&mut self.stdin).poll_flush(cx)
    }

    fn poll_shutdown(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::pin::Pin::new(&mut self.stdin).poll_shutdown(cx)
    }
}

/// Spawn a stdio MCP server; return its pid (its process-group id, since it leads one) and an rmcp
/// transport whose inbound lines pass through [`rescue`].
///
/// One small task per server pumps its stdout into an in-memory pipe rmcp reads. It owns the child:
/// when rmcp drops the transport (stdin closes) or the server closes stdout, it waits up to
/// [`SHUTDOWN_GRACE`] for a clean exit and kills the server if it is still there. The process-group
/// sweep is [`retire`]'s, called by whoever drops the connection.
pub(crate) fn stdio_transport(
    mut cmd: tokio::process::Command,
) -> std::io::Result<(
    Option<u32>,
    (tokio::io::DuplexStream, impl tokio::io::AsyncWrite),
)> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
    cmd.stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit())
        .kill_on_drop(true);
    let mut child = cmd.spawn()?;
    let pid = child.id();
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| std::io::Error::other("no stdout"))?;
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| std::io::Error::other("no stdin"))?;
    let (closed_tx, mut closed_rx) = oneshot::channel::<()>();
    let (rmcp_side, mut pump_side) = tokio::io::duplex(64 * 1024);
    let live = LivePump::new();
    tokio::spawn(async move {
        let mut child = child;
        let mut reader = tokio::io::BufReader::new(stdout);
        let mut line = Vec::new();
        loop {
            line.clear();
            tokio::select! {
                _ = &mut closed_rx => break,
                n = reader.read_until(b'\n', &mut line) => match n {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        if std::str::from_utf8(&line).is_err() {
                            tracing::debug!("skipping a non-UTF-8 line from an MCP server's stdout");
                            continue;
                        }
                        let body = line.strip_suffix(b"\n").unwrap_or(&line);
                        let body = body.strip_suffix(b"\r").unwrap_or(body);
                        let written = match rescue(body) {
                            Some(rewritten) => {
                                let mut out = rewritten;
                                out.push(b'\n');
                                pump_side.write_all(&out).await
                            }
                            None => {
                                let mut out = body.to_vec();
                                out.push(b'\n');
                                pump_side.write_all(&out).await
                            }
                        };
                        if written.is_err() {
                            break;
                        }
                    }
                },
            }
        }
        // The grace window: stdin is closed (rmcp dropped the transport) or the server closed its
        // own stdout. Either way, give it the time a well-behaved server needs to clean up.
        drop(pump_side);
        if tokio::time::timeout(SHUTDOWN_GRACE, child.wait())
            .await
            .is_err()
        {
            let _ = child.start_kill();
            let _ = child.wait().await;
        }
        // The group sweep is not done here: it belongs to whoever dropped the connection
        // ([`retire`]), which runs on a tracked OS thread and so also happens on process exit,
        // when this task may never be polled again.
        drop(live);
    });
    Ok((
        pid,
        (
            rmcp_side,
            StdinGuard {
                stdin,
                _closed: closed_tx,
            },
        ),
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

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
}
