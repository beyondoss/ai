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
//! **One boundary.** [`rescue`] / [`unwrap_rescued`] are the whole workaround. PR #131 (Skills) carries
//! its own rmcp `_meta` workaround in `tools/mcp_wire.rs`; unifying the two is meant to be a swap of
//! these two functions, nothing else.

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
pub fn sweep_before_exit() {
    retire_all();
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

/// The transport's write half: writes go to the server's stdin while it is open. Dropping it (rmcp
/// dropped the transport) retires the server.
struct StdinGuard {
    server: std::sync::Arc<ServerProcess>,
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
        match lock(&self.server.stdin).as_mut() {
            Some(stdin) => std::pin::Pin::new(stdin).poll_write(cx, buf),
            None => std::task::Poll::Ready(Err(closed())),
        }
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
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
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
    tokio::spawn(async move {
        let mut reader = tokio::io::BufReader::new(stdout);
        let mut line = Vec::new();
        loop {
            line.clear();
            match reader.read_until(b'\n', &mut line).await {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
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
    };
    Ok((server, (rmcp_side, guard)))
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
}
