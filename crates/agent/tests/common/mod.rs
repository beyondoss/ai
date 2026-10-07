//! Shared test helpers: a mock model server speaking Anthropic SSE, port helpers, a locator for
//! the gateway binary, and a process-lifetime JetStream for managed-gateway tests (allowance is
//! fail-closed until the watcher seeds).
// `dead_code` and `unused_imports` for the same reason: this module is compiled into *every* test
// binary, and no single binary uses all of it. That was already true of the helpers defined here; it
// is now also true of the re-exports from `beyond-ai-test-support`, which are `pub use` in a module
// that is private to each test crate and so are flagged per binary that happens not to need them.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    dead_code,
    unused_imports
)]

use std::io::{BufRead, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

/// The agent binary under test.
pub const BIN: &str = env!("CARGO_BIN_EXE_beyond-ai-agent");

/// A stand-in exec endpoint (the sandbox side of the exec protocol).
pub mod exec_mock;
/// A `bsg_v1` session-grant minter, written independently of `src/grant.rs`.
pub mod grant;
/// The MCP Apps suites' harness: fixture `$HOME`, daemon, renderer-side commands.
pub mod mcp_apps;
/// A streamable-HTTP MCP Apps server, for service mode's grant connectors.
pub mod mcp_apps_http;
/// The MCP Events fixture's control API and a bounded frame reader, for the `mcp_events_*` suites.
pub mod mcp_events_fixture;
/// A streamable-HTTP MCP server that records every request header it is sent.
pub mod mcp_fixture;
/// A real OAuth-protected MCP server with mid-session controls, and the `mcp-login` steps.
pub mod mcp_oauth_fixture;
/// A running `serve --service` replica, for the service-mode suites.
pub mod service;
/// The MCP Skills (SEP-2640) suites' isolated `$HOME` + skills-fixture harness.
pub mod skills_env;

/// Deterministic dev signing public key (standard base64), for a gateway `[signing_keys] 1 = …`.
pub const DEV_PUBKEY_B64: &str = "6kpsY+KcUgq+9VB7Ey7F+ZVHdq6+vnuSQh7qaRRG0iw=";
/// The matching dev `bai_v1` token (tenant 1 / vpc 1, kid 1).
pub const DEV_TOKEN: &str = "bai_v1.1.AQAAAAAAAAABAAAAAAAAAA.WrWcPbklu91PS-4WuR6GnBNF3h4nROpH0EQQlfJf06f7_lEnlQOCSBimhH2JMwXFJgw40BniTB7-yIdFnpldDw";

/// A `HOME` that deliberately doesn't exist on disk. `serve`/`run` read `~/.claude/skills` and
/// `~/.claude/trusted-projects.json` unconditionally (skill discovery is no longer gated on project
/// trust — an untrusted project must not blank out the user's own global skills), and every codepath
/// that reads under `HOME` already treats a missing file/directory as "nothing there" rather than an
/// error, so this keeps a test hermetic (never sees, and can't pollute, the actual developer's real
/// `~/.claude/`) without needing a `TempDir` guard kept alive for the spawned process's lifetime. A test
/// that specifically wants real HOME-relative behavior (trust store writes, seeded skills) overrides
/// this afterward via its own `.env("HOME", ...)`, which simply wins — `Command::env` is last-write.
pub const ISOLATED_HOME: &str = "/nonexistent-beyond-ai-agent-test-home";

/// A spawned child process that is killed and reaped when it goes out of scope — **including when the
/// enclosing test panics**.
///
/// `std::process::Child` deliberately does *not* kill on drop, so the usual test shape
///
/// ```ignore
/// let mut child = serve_cmd(..).spawn().unwrap();
/// assert_eq!(thing, other);   // <-- panics here
/// let _ = child.kill();       // <-- never runs
/// ```
///
/// orphans a real `serve` daemon on every failing assertion. Those orphans are reparented to init and
/// survive the whole `cargo test` run: they hold their listening port, their session directory (long
/// after the `TempDir` is gone), and tens of MB of RSS each. `--session-idle-timeout 0` tests pin
/// sessions for the daemon's lifetime by design, so those leaks never self-reap at all.
///
/// The compounding is what makes this worth a guard rather than more `kill()` calls: one genuine
/// failure leaks a daemon, the leak starves the *next* concurrent test of ports/memory, and that one
/// fails too — turning a single real bug into a cascade of unrelated red tests, which is exactly the
/// shape that makes a flaky suite impossible to read. Killing on drop makes a failure cost exactly one
/// test.
///
/// Derefs to [`Child`], so `child.stdin.take()`, `child.kill()`, and `child.wait()` all keep working
/// unchanged; an explicit `wait()` before the drop is fine (the drop's `kill` on an already-reaped pid
/// fails harmlessly and is ignored).
pub struct ChildGuard(Option<std::process::Child>);

impl ChildGuard {
    /// Spawn `cmd` under the guard. Panics with the command's name on failure, matching the
    /// `.spawn().unwrap()` this replaces.
    pub fn spawn(cmd: &mut Command) -> Self {
        Self(Some(cmd.spawn().unwrap_or_else(|e| {
            panic!("failed to spawn {:?}: {e}", cmd.get_program())
        })))
    }

    /// [`Child::wait_with_output`], which consumes the child and so can't come through `Deref`. Taking
    /// the child out disarms the guard — safe precisely because this call reaps the process itself.
    pub fn wait_with_output(mut self) -> std::io::Result<std::process::Output> {
        self.0.take().expect("child taken").wait_with_output()
    }
}

impl std::ops::Deref for ChildGuard {
    type Target = std::process::Child;
    fn deref(&self) -> &Self::Target {
        self.0.as_ref().expect("child taken")
    }
}

impl std::ops::DerefMut for ChildGuard {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut().expect("child taken")
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// `cmd.spawn_guarded()` — [`ChildGuard::spawn`] as a method, so an existing
/// `Command`-building chain only has to swap its trailing `.spawn().unwrap()`.
pub trait SpawnGuarded {
    fn spawn_guarded(&mut self) -> ChildGuard;
}

impl SpawnGuarded for Command {
    fn spawn_guarded(&mut self) -> ChildGuard {
        ChildGuard::spawn(self)
    }
}

/// Locate the gateway binary (built beside the agent binary); build it on demand if absent.
pub fn gateway_bin() -> PathBuf {
    let agent = PathBuf::from(env!("CARGO_BIN_EXE_beyond-ai-agent"));
    let dir = agent.parent().unwrap();
    let gw = dir.join("beyond-ai");
    if !gw.exists() {
        let mut args = vec!["build", "-q", "-p", "beyond-ai", "--bin", "beyond-ai"];
        if agent.to_string_lossy().contains("/release/") {
            args.push("--release");
        }
        let status = Command::new(env!("CARGO"))
            .args(&args)
            .status()
            .expect("build gateway");
        assert!(status.success(), "failed to build the gateway binary");
    }
    gw
}

/// An Anthropic SSE turn that calls one tool with the given JSON-argument string.
// The mock model server and its turn builders now live in `beyond-ai-test-support`, so the fleet
// simulator — a binary in its own crate — can use the same doubles these tests do. Re-exported
// rather than re-imported at each call site: every test that already says `common::turn_text`
// keeps working, and there is one implementation of the wire format rather than two.
pub use beyond_ai_test_support::{
    SCRIPTED_SESSION_TITLE, SESSION_TITLE_MARKER, spawn_model_server, spawn_model_server_routed,
    spawn_model_server_with_stalled_response, sse, title_calls, turn_refusal, turn_text,
    turn_text_responses, turn_tool_use,
};

/// A free localhost port (bind `:0`, read it back, release) — for a subprocess that can neither
/// report the port it bound nor adopt one it is handed (the gateway, `nats-server`). It has a window:
/// between the release and the child's bind any other process can take the port, and under parallel
/// suites one did, handing a test's connections to another test's server. A `serve` child never needs
/// this: use [`spawn_listening`], or [`HeldPort`] when the port must be known before it starts.
pub fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// The line `serve` prints on stderr once its `--listen` socket is bound (`serve_ws::serve_ws`).
const LISTENING: &str = "serve: websocket listening on ";

/// The port in `serve`'s listening announcement, if `line` is one.
fn announced_port(line: &str) -> Option<u16> {
    let addr = line.strip_prefix(LISTENING)?.split_whitespace().next()?;
    addr.parse::<std::net::SocketAddr>().ok().map(|a| a.port())
}

/// Spawn a `serve` command listening on a port the kernel picks, and return that port: appends
/// `--listen 127.0.0.1:0` and reads the bound address back from the line `serve` announces it with.
/// The port is `serve`'s from the moment it exists, so — unlike [`free_port`], which releases the
/// port it picked and hopes the child binds it first — no other process can take it. When this
/// returns the listener is up, so there is nothing to wait for.
///
/// Takes the child's stderr to read that line; everything it writes is forwarded line by line to the
/// test's (captured) stderr, so a child's panic or startup error is shown with a failing test instead
/// of being discarded. A child that exits without announcing fails here, with what it said.
pub fn spawn_listening(cmd: &mut Command) -> (ChildGuard, u16) {
    spawn_listening_logged(cmd, None)
}

/// [`spawn_listening`], also copying the child's stderr to `log` — for a test that reads it back.
pub fn spawn_listening_logged(cmd: &mut Command, log: Option<std::fs::File>) -> (ChildGuard, u16) {
    cmd.args(["--listen", "127.0.0.1:0"]).stderr(Stdio::piped());
    let mut child = ChildGuard::spawn(cmd);
    let stderr = child.stderr.take().expect("stderr is piped");
    let (tx, rx) = std::sync::mpsc::channel::<Result<u16, String>>();
    // Spawned threads inherit the test's output capture, so `eprintln!` here lands in it.
    thread::spawn(move || {
        let mut log = log;
        let mut pending = Some((tx, String::new()));
        for line in std::io::BufReader::new(stderr).lines() {
            let Ok(line) = line else { break };
            eprintln!("[serve] {line}");
            if let Some(f) = log.as_mut() {
                let _ = writeln!(f, "{line}");
            }
            if let Some((tx, said)) = pending.as_mut() {
                match announced_port(&line) {
                    Some(port) => {
                        let _ = tx.send(Ok(port));
                        pending = None;
                    }
                    None => {
                        said.push_str(&line);
                        said.push('\n');
                    }
                }
            }
        }
        if let Some((tx, said)) = pending {
            let _ = tx.send(Err(said));
        }
    });
    match rx.recv_timeout(Duration::from_secs(60)) {
        Ok(Ok(port)) => (child, port),
        Ok(Err(said)) => panic!("serve exited without listening; it said:\n{said}"),
        Err(e) => panic!("serve did not announce its listener within 60s ({e})"),
    }
}

/// A loopback port the test holds from before `serve` starts, for a test that has to name the port
/// in `serve`'s own arguments (an MCP Events callback URL pointing back at the daemon), so cannot
/// learn it afterwards. The test binds the listener and hands it to `serve` as a socket-activated
/// one (`LISTEN_FDS`, exactly as systemd does) — so the port is never released between being chosen
/// and being served, and no other process can take it. Held for as long as this value lives, so a
/// daemon restarted on the same port gets the same socket ([`HeldPort::down`] covers the gap).
///
/// It is passed as the child's stdin (inetd's convention, `LISTEN_FDS_FIRST_FD=0`): std's `dup2` onto
/// a standard stream is the one way to give a child a descriptor without `unsafe`, which the workspace
/// forbids, and `serve` never reads stdin once it serves a listener. Drop it once the daemon is up
/// unless it will be restarted: while the test holds the socket, a daemon that died leaves it
/// listening, and a client would wait on it instead of being refused.
pub struct HeldPort {
    listener: TcpListener,
    port: u16,
}

impl HeldPort {
    pub fn bind() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        Self { listener, port }
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    /// Make this socket `cmd`'s listener. Call it after anything else that sets stdin, and pass no
    /// `--listen`/`--listen-uds` (either turns socket activation off).
    pub fn hand_to(&self, cmd: &mut Command) {
        let fd = std::os::fd::OwnedFd::from(self.listener.try_clone().unwrap());
        cmd.stdin(Stdio::from(fd))
            .env("LISTEN_FDS", "1")
            .env("LISTEN_FDS_FIRST_FD", "0")
            .env_remove("LISTEN_PID");
    }

    /// Between a daemon's death and its restart: until the guard drops, every connection is accepted
    /// and closed at once, so a client fails promptly — as against a stopped daemon — rather than
    /// waiting in the backlog of a socket the test still holds.
    pub fn down(&self) -> PortDown {
        let listener = self.listener.try_clone().unwrap();
        listener.set_nonblocking(true).unwrap();
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stopping = stop.clone();
        let thread = thread::spawn(move || {
            while !stopping.load(std::sync::atomic::Ordering::Relaxed) {
                match listener.accept() {
                    Ok((conn, _)) => drop(conn),
                    Err(_) => thread::sleep(Duration::from_millis(5)),
                }
            }
        });
        PortDown {
            stop,
            thread: Some(thread),
        }
    }
}

/// See [`HeldPort::down`].
pub struct PortDown {
    stop: Arc<std::sync::atomic::AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl Drop for PortDown {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// A loopback port nothing listens on, held for as long as the value lives: bound but never
/// listening, so a connection is refused, and taken, so no other process can start listening there
/// in the meantime — which a released "dead" port from [`free_port`] cannot promise.
pub struct DeadPort {
    _socket: tokio::net::TcpSocket,
    port: u16,
}

impl DeadPort {
    pub fn bind() -> Self {
        let socket = tokio::net::TcpSocket::new_v4().unwrap();
        socket.bind("127.0.0.1:0".parse().unwrap()).unwrap();
        let port = socket.local_addr().unwrap().port();
        Self {
            _socket: socket,
            port,
        }
    }

    pub fn port(&self) -> u16 {
        self.port
    }
}

/// Block until `port` accepts a TCP connection, or panic after ~5s.
pub fn wait_for_port(port: u16) {
    for _ in 0..500 {
        if std::net::TcpStream::connect(("127.0.0.1", port)).is_ok() {
            return;
        }
        thread::sleep(Duration::from_millis(10));
    }
    panic!("port {port} never came up");
}

/// One JetStream server for this test process. Held until exit so many gateway boots can share it.
///
/// Allowance is fail-closed until the watcher stores a scan (empty = remaining-ok). A closed NATS
/// port 402s every managed request, which is why [`wait_for_allowance_ready`] exists alongside this.
struct SharedNats {
    port: u16,
    child: Child,
}

impl SharedNats {
    fn spawn() -> Self {
        let port = free_port();
        let store_dir = std::env::temp_dir().join(format!("beyond-ai-agent-nats-{port}"));
        let _ = std::fs::create_dir_all(&store_dir);
        let mut child = Command::new("nats-server")
            .args([
                "-js",
                "-a",
                "127.0.0.1",
                "-p",
                &port.to_string(),
                "-sd",
                store_dir.to_str().unwrap(),
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn nats-server (on PATH? run via mise)");
        let deadline = Instant::now() + Duration::from_secs(20);
        while Instant::now() < deadline {
            if TcpStream::connect(("127.0.0.1", port)).is_ok() {
                return Self { port, child };
            }
            thread::sleep(Duration::from_millis(50));
        }
        let _ = child.kill();
        let _ = child.wait();
        panic!("shared nats-server did not come up on port {port}");
    }
}

pub fn unused_nats_port() -> u16 {
    static SERVER: OnceLock<SharedNats> = OnceLock::new();
    SERVER.get_or_init(SharedNats::spawn).port
}

/// Block until the gateway's allowance watcher has seeded (`ai_allowance_ready==1`).
///
/// Listen-port readiness is not enough: managed traffic 402s until the first scan (including an
/// empty one).
pub fn wait_for_allowance_ready(metrics_port: u16) {
    wait_for_port(metrics_port);
    for _ in 0..200 {
        if scrape_gauge(&fetch_metrics(metrics_port), "ai_allowance_ready") >= 1.0 {
            return;
        }
        thread::sleep(Duration::from_millis(50));
    }
    panic!("ai_allowance_ready never reached 1 on metrics port {metrics_port}");
}

fn fetch_metrics(port: u16) -> String {
    let Ok(mut stream) = TcpStream::connect(("127.0.0.1", port)) else {
        return String::new();
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let _ = write!(
        stream,
        "GET /metrics HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n"
    );
    let mut buf = Vec::new();
    let _ = stream.read_to_end(&mut buf);
    String::from_utf8_lossy(&buf).into_owned()
}

fn scrape_gauge(metrics: &str, name: &str) -> f64 {
    metrics
        .lines()
        .find(|l| l.starts_with(name) && !l[name.len()..].starts_with('_'))
        .and_then(|l| l.rsplit(' ').next())
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0)
}

/// Read stdout frames from a `serve` child until the `response` frame for `command` arrives; return
/// all frames seen (including any `event`/progress frames along the way).
pub fn read_until_response(reader: &mut impl BufRead, command: &str) -> Vec<Value> {
    let mut frames = Vec::new();
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line).unwrap() == 0 {
            break;
        }
        let Ok(v) = serde_json::from_str::<Value>(line.trim()) else {
            continue;
        };
        let done = v.get("type").and_then(Value::as_str) == Some("response")
            && v.get("command").and_then(Value::as_str) == Some(command);
        frames.push(v);
        if done {
            break;
        }
    }
    frames
}

/// A `serve` child's stdout, read line by line on its own thread, that fails rather than hangs: a
/// read that sees no new line within `deadline` panics, naming the deadline. Hand it to
/// [`read_until_response`]/[`read_until_event`] like any `BufRead`. Without one, a run that stalls —
/// a scripted reply consumed by the wrong request, say — leaves the test blocked in `read_line` until
/// the runner's own timeout kills it, with nothing said about where it stopped.
pub struct Frames {
    lines: std::sync::mpsc::Receiver<String>,
    current: Vec<u8>,
    pos: usize,
    deadline: Duration,
}

/// [`Frames`] over `out`.
pub fn frames_with_deadline(out: impl Read + Send + 'static, deadline: Duration) -> Frames {
    let (tx, lines) = std::sync::mpsc::channel();
    thread::spawn(move || {
        let mut out = std::io::BufReader::new(out);
        loop {
            let mut line = String::new();
            match out.read_line(&mut line) {
                Ok(0) | Err(_) => break,
                Ok(_) if tx.send(line).is_err() => break,
                Ok(_) => {}
            }
        }
    });
    Frames {
        lines,
        current: Vec::new(),
        pos: 0,
        deadline,
    }
}

impl Read for Frames {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let available = self.fill_buf()?;
        let n = available.len().min(buf.len());
        buf[..n].copy_from_slice(&available[..n]);
        self.consume(n);
        Ok(n)
    }
}

impl BufRead for Frames {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        if self.pos >= self.current.len() {
            match self.lines.recv_timeout(self.deadline) {
                Ok(line) => {
                    self.current = line.into_bytes();
                    self.pos = 0;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return Ok(&[]),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => panic!(
                    "serve wrote nothing for {:?}: the run is stalled (a scripted reply taken by \
                     the wrong request — the session-title call, say — leaves it waiting forever)",
                    self.deadline
                ),
            }
        }
        Ok(&self.current[self.pos..])
    }

    fn consume(&mut self, n: usize) {
        self.pos += n;
    }
}

/// Read stdout frames from a `serve` child until an `event` frame whose body satisfies `matches`
/// arrives; return all frames seen, that event last.
///
/// This is the deterministic replacement for sleeping a fixed duration and hoping the run has reached
/// a particular point. A wall-clock guess ("150ms should land mid the tool call's own `sleep 0.5`")
/// holds only while the machine is idle. Under a loaded shard — every test spawning a real
/// `beyond-ai-agent` plus a mock model server, four at a time — the probe lands *before* the turn
/// reaches its `bash` call, or *after* that call finished, and the assertion fails for a reason with
/// nothing to do with the behaviour under test. Asking the run where it is has no such window.
///
/// `matches` receives the event body (the `kind`-tagged [`beyond_ai_agent_core::AgentEvent`]), because
/// what counts as "there yet" differs per test: `kind == "tool_start"` is "the call is running now",
/// the first `tool_progress` is "output has already streamed", and a turn that calls the same tool
/// twice has to key on the tool-use `id` the test itself authored.
///
/// Frames read on the way are returned rather than dropped, so a caller that still needs them can
/// chain: `frames.extend(read_until_response(&mut stdout, "prompt"))`.
pub fn read_until_event(
    reader: &mut impl BufRead,
    mut matches: impl FnMut(&Value) -> bool,
) -> Vec<Value> {
    let mut frames = Vec::new();
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line).unwrap() == 0 {
            break;
        }
        let Ok(v) = serde_json::from_str::<Value>(line.trim()) else {
            continue;
        };
        let done = v.get("type").and_then(Value::as_str) == Some("event")
            && v.get("event").is_some_and(&mut matches);
        frames.push(v);
        if done {
            break;
        }
    }
    frames
}

/// Strip ambient provider-routing env so a mock `--gateway-url` actually wins, and strip
/// run-lifecycle env so an unconfigured test never POSTs to a URL the developer happened to export.
///
/// Eval hosts (this one included) export `AI_DIRECT=1` / `AI_PROVIDER=openrouter` /
/// `OPENROUTER_API_KEY` for Harbor runs. `AI_DIRECT=1` makes the binary ignore `--gateway-url`
/// and dial OpenRouter; a hermetic test that inherited that would bill a live provider and fail
/// on `claude-test`. Tests that *want* those vars (the ignored live Code Mode smoke) re-set them
/// after [`run_cmd`] — `Command::env` is last-write.
fn isolate_provider_env(cmd: &mut Command) {
    for key in [
        "AI_DIRECT",
        "AI_PROVIDER",
        "AI_BASE_URL",
        "AI_API_KEY",
        "AI_GATEWAY_URL",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY",
        "AWS_BEARER_TOKEN_BEDROCK",
        // A developer/CI env that POSTs run lifecycle must not leak into hermetic tests; tests that
        // want the emitter set it after [`serve_cmd`]/[`run_cmd`] (last-write wins).
        "AI_AGENT_LIFECYCLE_URL",
        "AI_AGENT_LIFECYCLE_HEADER",
        "AI_AGENT_LIFECYCLE_HEARTBEAT_SECS",
        // Session-grant trust names a key file; an exported one must not make every test's startup
        // depend on it.
        "AI_AGENT_GRANT_KEY",
        "AI_AGENT_SEAL_KEY",
    ] {
        cmd.env_remove(key);
    }
}

/// A `serve` child bound to a single session file, talking to the mock gateway at `base`.
pub fn serve_cmd(bin: &str, base: &str, session_file: &str) -> Command {
    let mut c = Command::new(bin);
    isolate_provider_env(&mut c);
    c.args([
        "serve",
        "--gateway-url",
        base,
        "--key",
        "bai_v1.test",
        "--model",
        "claude-test",
        "--session-file",
        session_file,
    ])
    .env("HOME", ISOLATED_HOME)
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::null());
    c
}

/// Like `serve_cmd`, but bound to a session *directory* (`--session-dir`) rather than a single file —
/// exercises the multi-session-per-process repo mode (`list_sessions`, `switch`, fork).
pub fn serve_dir_cmd(bin: &str, base: &str, session_dir: &str) -> Command {
    let mut c = Command::new(bin);
    isolate_provider_env(&mut c);
    c.args([
        "serve",
        "--gateway-url",
        base,
        "--key",
        "bai_v1.test",
        "--model",
        "claude-test",
        "--session-dir",
        session_dir,
    ])
    .env("HOME", ISOLATED_HOME)
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::null());
    c
}

/// [`Command::new`] for the `run` binary, pre-isolated from the real machine's `HOME` — see
/// [`ISOLATED_HOME`]. A test that wants real HOME-relative behavior overrides it via its own
/// `.env("HOME", ...)`, which simply wins (`Command::env` is last-write).
pub fn run_cmd(bin: &str) -> Command {
    let mut c = Command::new(bin);
    isolate_provider_env(&mut c);
    c.env("HOME", ISOLATED_HOME);
    c
}

/// The fixed WebSocket path `serve --listen` accepts (see `serve_ws`).
pub const WS_PATH: &str = "/_beyond/agent";

/// A connected test WebSocket client (over plain `ws://`, so `MaybeTlsStream` is always the plain arm).
pub type TestWs =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// Connect with extra request headers — `x-beyond-grant`, for `serve --service`.
///
/// Returns the **HTTP status** on failure rather than a message: service mode's whole refusal
/// contract is a status table (401 / 400 / 403 / 421), and the statuses are answered *before* the
/// upgrade precisely so a client can read them. A test that asserted on an error string would not
/// be checking that.
pub async fn ws_connect_with_headers(
    port: u16,
    session_id: Option<&str>,
    headers: &[(&str, &str)],
) -> Result<TestWs, u16> {
    match tokio_tungstenite::connect_async(ws_request(port, session_id, headers)).await {
        Ok((ws, _resp)) => Ok(ws),
        Err(tokio_tungstenite::tungstenite::Error::Http(resp)) => Err(resp.status().as_u16()),
        Err(e) => panic!("websocket connect failed without an HTTP status: {e}"),
    }
}

/// Connect expecting to be **refused**, and return the status with its `Retry-After` value — the two
/// things a client or a proxy acts on for a 503. Panics if the connection is accepted.
pub async fn ws_refusal(
    port: u16,
    session_id: Option<&str>,
    headers: &[(&str, &str)],
) -> (u16, Option<String>) {
    match tokio_tungstenite::connect_async(ws_request(port, session_id, headers)).await {
        Ok(_) => panic!("expected the connection to be refused"),
        Err(tokio_tungstenite::tungstenite::Error::Http(resp)) => (
            resp.status().as_u16(),
            resp.headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .map(str::to_string),
        ),
        Err(e) => panic!("websocket connect failed without an HTTP status: {e}"),
    }
}

pub fn ws_request(
    port: u16,
    session_id: Option<&str>,
    headers: &[(&str, &str)],
) -> tokio_tungstenite::tungstenite::handshake::client::Request {
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;
    let url = match session_id {
        Some(id) => format!("ws://127.0.0.1:{port}{WS_PATH}?session_id={id}"),
        None => format!("ws://127.0.0.1:{port}{WS_PATH}"),
    };
    let mut request = url.into_client_request().expect("build ws request");
    for (name, value) in headers {
        request.headers_mut().insert(
            tokio_tungstenite::tungstenite::http::header::HeaderName::from_bytes(name.as_bytes())
                .expect("header name"),
            value.parse().expect("header value"),
        );
    }
    request
}

/// A `serve --service` child: a grant verifier and one or more `--shard <name>=<path>` mounts. Spawn
/// it with [`spawn_listening`], which adds the listener. Deliberately *not* built on [`serve_cmd`] —
/// that passes `--key` and `--session-file`, both of which service mode refuses at startup.
pub fn serve_service_cmd(
    bin: &str,
    base: &str,
    grant_key_flag: &str,
    seal_key: &std::path::Path,
    shards: &[(&str, &std::path::Path)],
) -> Command {
    let mut c = Command::new(bin);
    isolate_provider_env(&mut c);
    c.args([
        "serve",
        "--service",
        "--gateway-url",
        base,
        "--model",
        "claude-test",
        "--grant-key",
        grant_key_flag,
        "--seal-key",
        &seal_key.to_string_lossy(),
    ]);
    for (name, path) in shards {
        c.arg("--shard").arg(format!("{name}={}", path.display()));
    }
    c.env("HOME", ISOLATED_HOME)
        .stdin(Stdio::null())
        .stdout(Stdio::null());
    c
}

/// Connect a WebSocket client to a `serve --listen` port, optionally naming a session via the URL.
pub async fn ws_connect(port: u16, session_id: Option<&str>) -> TestWs {
    let url = match session_id {
        Some(id) => format!("ws://127.0.0.1:{port}{WS_PATH}?session_id={id}"),
        None => format!("ws://127.0.0.1:{port}{WS_PATH}"),
    };
    let (ws, _resp) = tokio_tungstenite::connect_async(url)
        .await
        .expect("websocket connect");
    ws
}

/// A test WebSocket client running over a Unix-domain socket (the `--listen-uds` transport). The
/// generic stream arm differs from [`TestWs`]'s TCP one, so UDS helpers take this type; the frame
/// helpers ([`ws_send`], [`ws_next_frame`], [`ws_read_until_response`]) are generic over the stream.
pub type TestWsUds = tokio_tungstenite::WebSocketStream<tokio::net::UnixStream>;

/// Dial a `serve --listen-uds` socket at `path`, optionally naming a session via the URL. The HTTP
/// `Host` is ignored over a UDS, so it's a synthetic `localhost`; only the path + `?session_id=` query
/// matter (the handshake validates the path and parses the session id).
pub async fn ws_connect_uds(path: &std::path::Path, session_id: Option<&str>) -> TestWsUds {
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;
    let url = match session_id {
        Some(id) => format!("ws://localhost{WS_PATH}?session_id={id}"),
        None => format!("ws://localhost{WS_PATH}"),
    };
    let request = url.into_client_request().expect("build ws request");
    let stream = tokio::net::UnixStream::connect(path)
        .await
        .expect("connect unix socket");
    let (ws, _resp) = tokio_tungstenite::client_async(request, stream)
        .await
        .expect("websocket handshake over uds");
    ws
}

/// Send one command object as a single WS text message. Generic over the underlying transport so the
/// same helper drives both the TCP ([`TestWs`]) and UDS ([`TestWsUds`]) clients.
pub async fn ws_send<T>(ws: &mut tokio_tungstenite::WebSocketStream<T>, v: Value)
where
    T: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    use futures::SinkExt as _;
    ws.send(tokio_tungstenite::tungstenite::Message::Text(
        v.to_string().into(),
    ))
    .await
    .expect("websocket send");
}

/// Read the next JSON frame (skipping ping/pong/binary), or `None` if the socket closed.
pub async fn ws_next_frame<T>(ws: &mut tokio_tungstenite::WebSocketStream<T>) -> Option<Value>
where
    T: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    use futures::StreamExt as _;
    use tokio_tungstenite::tungstenite::Message;
    while let Some(msg) = ws.next().await {
        match msg.expect("websocket recv") {
            Message::Text(t) => {
                if let Ok(v) = serde_json::from_str::<Value>(t.as_str()) {
                    return Some(v);
                }
            }
            Message::Close(_) => return None,
            _ => {}
        }
    }
    None
}

/// Collect WS frames until the `response` frame for `command` arrives (or the socket closes).
pub async fn ws_read_until_response<T>(
    ws: &mut tokio_tungstenite::WebSocketStream<T>,
    command: &str,
) -> Vec<Value>
where
    T: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let mut frames = Vec::new();
    while let Some(v) = ws_next_frame(ws).await {
        let done = v.get("type").and_then(Value::as_str) == Some("response")
            && v.get("command").and_then(Value::as_str) == Some(command);
        frames.push(v);
        if done {
            break;
        }
    }
    frames
}

/// The JSON body of a raw recorded request (as `spawn_model_server` records it: headers + body).
pub fn body_json(raw_request: &str) -> Value {
    let body = raw_request
        .split_once("\r\n\r\n")
        .expect("request must have a body")
        .1;
    serde_json::from_str(body).expect("request body must be JSON")
}

/// The names of the tools a request *advertised* to the model.
///
/// Not a substring search on the raw request: the body also carries the conversation history, whose
/// `tool_use`/`tool_result` blocks name every tool the model has ever called. A turn that no longer
/// offers a tool still mentions it, so only the `tools` array answers "what may the model call now?".
pub fn advertised_tools(raw_request: &str) -> Vec<String> {
    body_json(raw_request)["tools"]
        .as_array()
        .map(|tools| {
            tools
                .iter()
                .filter_map(|t| t["name"].as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// Every persisted `message` entry's id, in order, read straight off a session JSONL file.
pub fn message_ids(session_file: &str) -> Vec<String> {
    std::fs::read_to_string(session_file)
        .unwrap()
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|v| v["type"] == "message")
        .filter_map(|v| v["id"].as_str().map(str::to_string))
        .collect()
}
