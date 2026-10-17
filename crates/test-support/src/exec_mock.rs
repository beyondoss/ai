//! A stand-in exec endpoint: the other side of the exec protocol (v1.1, and v1.2 streaming),
//! running each command as a real process on this host.
//!
//! Shared rather than per-test because several suites need the same thing for different reasons.
//! `exec_endpoint_writes` needs a **v1** endpoint (`honors_stdin: false`) that ignores `stdin_base64`
//! like any unknown field, so the 128 KiB argv cap bites for real. `serve_service_mode` needs a
//! **sandbox**: a directory that stands in for the tenant's box, with its own `HOME`, so a test can
//! tell "the tool ran in the sandbox" from "the tool ran on the replica" by which marker file it
//! finds — and can assert, from [`ExecMock::requests`], exactly which headers the grant sent. The
//! streaming suites need a **v1.2** endpoint whose process groups they can watch die
//! ([`ExecMock::streams`]), and the endpoints that get v1.2 wrong in each way the probe and the
//! conformance verifier must catch ([`Protocol`], [`MockOptions::kills_on_disconnect`]).
//!
//! The default is a conforming v1.3 endpoint — v1.2 streaming plus native sandbox jobs
//! ([`MockOptions::jobs`]): what a real sandbox should be, so every suite that does not ask for
//! something else exercises the newest path. `jobs: false` is a v1.2-only endpoint, on which the
//! agent falls back to running jobs with plain exec commands (the baseline).

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use base64::Engine as _;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// One recorded request: its headers (lowercased names) and its JSON body.
#[derive(Clone, Debug)]
pub struct ExecRequest {
    pub headers: Vec<(String, String)>,
    pub body: Value,
}

impl ExecRequest {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    pub fn command(&self) -> &str {
        self.body["command"].as_str().unwrap_or("")
    }

    pub fn args(&self) -> Vec<String> {
        self.body["args"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// Which protocol the endpoint speaks when asked to stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Protocol {
    /// v1.2, conforming: `"stream": true` gets frames as the command produces them.
    Streaming,
    /// v1.1: `stream` is an unknown field and ignored; the answer is always one JSON object.
    V11,
    /// Answers with `application/x-ndjson` but holds everything until the command exits, then sends
    /// one frame per stream and the final frame — a lazy implementation, or a buffering proxy in
    /// front of a good one. Looks like v1.2 on the wire; the probe must refuse it.
    Buffered,
}

/// How the endpoint behaves. [`Default`] is a conforming v1.2 sandbox with stdin support.
#[derive(Clone, Debug)]
pub struct MockOptions {
    pub honors_stdin: bool,
    pub protocol: Protocol,
    /// The silence after which a stream gets a heartbeat frame. The spec's is 5 s; a suite that
    /// wants to prove heartbeats are not output makes it much shorter than its idle window.
    pub heartbeat: Duration,
    /// Kill the command's process group when the client disconnects mid-stream — the v1.2 MUST.
    /// `false` is the non-conforming endpoint the conformance verifier has to catch.
    pub kills_on_disconnect: bool,
    /// When the shell exits leaving processes of its group running, send `{"shell_exit": N}` and
    /// keep the stream open until the group is empty — the v1.2 MUST ("Leftover processes").
    /// `false` is an endpoint predating that rule: the stream ends with the shell and the leftovers
    /// run on, untracked.
    pub tracks_leftovers: bool,
    /// `HOME` for every command; `None` inherits this process's.
    pub home: Option<PathBuf>,
    /// Offer sandbox jobs natively (v1.3): answer requests carrying `"job"`. `false` is a v1.2-only
    /// endpoint, which ignores the field and runs the command.
    pub jobs: bool,
    /// Kill a held job whose stream closes before it was kept — the v1.3 MUST. `false` is the
    /// non-conforming endpoint the conformance verifier has to catch.
    pub jobs_hold_kills: bool,
    /// How much of a job's output is kept; older bytes are dropped and read back as a gap.
    pub job_output_cap: usize,
}

impl Default for MockOptions {
    fn default() -> Self {
        Self {
            honors_stdin: true,
            protocol: Protocol::Streaming,
            heartbeat: Duration::from_secs(5),
            kills_on_disconnect: true,
            tracks_leftovers: true,
            home: None,
            jobs: true,
            jobs_hold_kills: true,
            job_output_cap: 8 * 1024 * 1024,
        }
    }
}

/// How a streamed command's exchange ended, as the endpoint saw it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamEnd {
    /// The command exited (or was signalled) and its final frame was sent.
    Finished,
    /// The endpoint enforced `timeout_ms`.
    TimedOut,
    /// The client went away before the final frame. `killed` is whether the endpoint then killed
    /// the process group (always, unless [`MockOptions::kills_on_disconnect`] is off).
    Disconnected { killed: bool },
}

/// One streamed command, as the endpoint ran it.
#[derive(Clone, Debug)]
pub struct StreamRecord {
    /// The command's process group on this host (it leads it).
    pub pgid: u32,
    pub command: String,
    pub args: Vec<String>,
    pub started: Instant,
    /// `None` while it runs.
    pub end: Option<(StreamEnd, Instant)>,
}

/// `Clone` shares the endpoint rather than starting a second one — the request log is behind the
/// same `Arc`. That is what a driver hosting one exec double for a whole run wants: every scenario
/// holds a handle to the same server, at the one address its replicas were started with.
#[derive(Clone)]
pub struct ExecMock {
    pub url: String,
    requests: Arc<Mutex<Vec<ExecRequest>>>,
    streams: Arc<Mutex<Vec<StreamRecord>>>,
}

#[derive(Clone)]
struct Shared {
    root: PathBuf,
    opts: MockOptions,
    requests: Arc<Mutex<Vec<ExecRequest>>>,
    streams: Arc<Mutex<Vec<StreamRecord>>>,
    jobs: Jobs,
}

impl ExecMock {
    /// An endpoint rooted at `root`, with no `HOME` of its own.
    pub async fn start(root: &Path, honors_stdin: bool) -> Self {
        Self::start_with_home(root, None, honors_stdin).await
    }

    /// A sandbox: commands run with `cwd` defaulting to `root` and `HOME` set to `home`, so `$HOME`
    /// (what `serve --service`'s startup probe asks for) is the sandbox's, never the replica's.
    pub async fn start_with_home(root: &Path, home: Option<&Path>, honors_stdin: bool) -> Self {
        Self::start_with(
            root,
            MockOptions {
                honors_stdin,
                home: home.map(Path::to_path_buf),
                ..MockOptions::default()
            },
        )
        .await
    }

    /// An endpoint behaving as `opts` says.
    pub async fn start_with(root: &Path, opts: MockOptions) -> Self {
        Self::serve(crate::ports::tokio_listener().await, root, opts)
    }

    /// As [`Self::start_with_home`], bound where the caller says.
    ///
    /// A test wants an ephemeral loopback port. A **fleet the simulator attached to** needs the
    /// opposite: replicas carry the exec URL in their grant and are started before the driver, so
    /// the address has to be predictable and reachable from another host — which `127.0.0.1:0` is
    /// neither of.
    pub async fn start_on(
        bind: &str,
        root: &Path,
        home: Option<&Path>,
        honors_stdin: bool,
    ) -> Self {
        let listener = tokio::net::TcpListener::bind(bind).await.unwrap();
        Self::serve(
            listener,
            root,
            MockOptions {
                honors_stdin,
                home: home.map(Path::to_path_buf),
                ..MockOptions::default()
            },
        )
    }

    /// Serve on `listener`, which the endpoint holds for as long as it runs.
    fn serve(listener: tokio::net::TcpListener, root: &Path, opts: MockOptions) -> Self {
        let url = format!("http://{}/exec", listener.local_addr().unwrap());
        let shared = Shared {
            root: root.to_path_buf(),
            opts,
            requests: Arc::default(),
            streams: Arc::default(),
            jobs: Jobs::default(),
        };
        let mock = Self {
            url,
            requests: shared.requests.clone(),
            streams: shared.streams.clone(),
        };
        tokio::spawn(async move {
            while let Ok((sock, _)) = listener.accept().await {
                tokio::spawn(serve_one(sock, shared.clone()));
            }
        });
        mock
    }

    pub fn requests(&self) -> Vec<ExecRequest> {
        self.requests.lock().unwrap().clone()
    }

    /// Every request body, for suites that only care about the protocol.
    pub fn bodies(&self) -> Vec<Value> {
        self.requests().into_iter().map(|r| r.body).collect()
    }

    /// Every streamed command so far, in the order they started.
    pub fn streams(&self) -> Vec<StreamRecord> {
        self.streams.lock().unwrap().clone()
    }

    /// The streamed commands whose arguments mention `needle` — how a suite finds *its* command
    /// among the probes.
    pub fn streams_running(&self, needle: &str) -> Vec<StreamRecord> {
        self.streams()
            .into_iter()
            .filter(|s| s.args.iter().any(|a| a.contains(needle)))
            .collect()
    }

    /// Wait (bounded) until a streamed command mentioning `needle` has started; its record.
    pub async fn wait_for_stream(&self, needle: &str, within: Duration) -> Option<StreamRecord> {
        let deadline = Instant::now() + within;
        loop {
            if let Some(s) = self.streams_running(needle).into_iter().next() {
                return Some(s);
            }
            if Instant::now() >= deadline {
                return None;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// Wait (bounded) until the stream for `pgid` has ended; how it ended.
    pub async fn wait_for_end(&self, pgid: u32, within: Duration) -> Option<StreamEnd> {
        let deadline = Instant::now() + within;
        loop {
            let end = self
                .streams()
                .into_iter()
                .find(|s| s.pgid == pgid)
                .and_then(|s| s.end.map(|(e, _)| e));
            if end.is_some() || Instant::now() >= deadline {
                return end;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }
}

/// Whether any process is still in group `pgid` (`kill -0 -- -pgid`). A zombie counts until it is
/// reaped, so callers poll with [`wait_group_gone`].
pub fn group_alive(pgid: u32) -> bool {
    std::process::Command::new("kill")
        .args(["-0", "--", &format!("-{pgid}")])
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// Wait (bounded) for every process in group `pgid` to be gone.
pub async fn wait_group_gone(pgid: u32, within: Duration) -> bool {
    let deadline = Instant::now() + within;
    while group_alive(pgid) {
        if Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    true
}

fn kill_group(pgid: u32) {
    let _ = std::process::Command::new("kill")
        .args(["-KILL", "--", &format!("-{pgid}")])
        .stderr(std::process::Stdio::null())
        .status();
}

/// Read one request: headers, then `Content-Length` bytes of JSON body.
async fn read_request(sock: &mut tokio::net::TcpStream) -> Option<(Vec<(String, String)>, Value)> {
    let (mut buf, mut chunk) = (Vec::new(), [0u8; 64 * 1024]);
    let (mut need, mut head_end) = (0usize, None);
    let mut headers = Vec::new();
    loop {
        let n = sock.read(&mut chunk).await.ok()?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
        if head_end.is_none()
            && let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n")
        {
            head_end = Some(p + 4);
            let head = String::from_utf8_lossy(&buf[..p]).into_owned();
            for line in head.lines().skip(1) {
                if let Some((name, value)) = line.split_once(':') {
                    headers.push((name.trim().to_lowercase(), value.trim().to_string()));
                }
            }
            need = headers
                .iter()
                .find(|(n, _)| n == "content-length")
                .and_then(|(_, v)| v.parse().ok())
                .unwrap_or(0);
        }
        if head_end.is_some_and(|h| buf.len() >= h + need) {
            break;
        }
    }
    let req: Value =
        serde_json::from_slice(&buf[head_end.unwrap_or(buf.len())..]).unwrap_or(json!({}));
    Some((headers, req))
}

async fn serve_one(mut sock: tokio::net::TcpStream, shared: Shared) {
    let Some((headers, req)) = read_request(&mut sock).await else {
        return;
    };
    shared.requests.lock().unwrap().push(ExecRequest {
        headers,
        body: req.clone(),
    });
    if shared.opts.jobs && req.get("job").is_some_and(Value::is_object) {
        serve_job(sock, req, shared).await;
        return;
    }
    let wants_stream = req["stream"].as_bool() == Some(true);
    if wants_stream && shared.opts.protocol != Protocol::V11 {
        serve_stream(sock, req, shared).await;
    } else {
        serve_envelope(sock, req, shared).await;
    }
}

fn args_of(req: &Value) -> Vec<String> {
    req["args"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

fn command_for(req: &Value, shared: &Shared) -> tokio::process::Command {
    let mut cmd = tokio::process::Command::new(req["command"].as_str().unwrap_or("true"));
    cmd.args(args_of(req))
        .current_dir(
            req["cwd"]
                .as_str()
                .map_or_else(|| shared.root.clone(), Into::into),
        )
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    if let Some(home) = &shared.opts.home {
        cmd.env("HOME", home);
    }
    cmd
}

/// v1.1: run to completion, answer with one object.
async fn serve_envelope(mut sock: tokio::net::TcpStream, req: Value, shared: Shared) {
    let stdin = req["stdin_base64"]
        .as_str()
        .filter(|_| shared.opts.honors_stdin)
        .map(|b| base64::engine::general_purpose::STANDARD.decode(b).unwrap());
    let mut cmd = command_for(&req, &shared);
    cmd.stdin(if stdin.is_some() {
        std::process::Stdio::piped()
    } else {
        std::process::Stdio::null()
    });
    let payload = match cmd.spawn() {
        Ok(mut child) => {
            let pipe = child.stdin.take();
            let feed = async move {
                if let (Some(mut pipe), Some(bytes)) = (pipe, stdin) {
                    let _ = pipe.write_all(&bytes).await;
                }
            };
            let (out, ()) = tokio::join!(child.wait_with_output(), feed);
            let o = out.unwrap();
            json!({
                "exit_code": o.status.code().unwrap_or(-1),
                "stdout": String::from_utf8_lossy(&o.stdout),
                "stderr": String::from_utf8_lossy(&o.stderr),
            })
        }
        // `Argument list too long` lands here — the defect v1.1 exists to fix.
        Err(e) => json!({ "exit_code": 127, "stdout": "", "stderr": e.to_string() }),
    };
    let body = payload.to_string();
    let resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = sock.write_all(resp.as_bytes()).await;
    let _ = sock.shutdown().await;
}

/// One frame as an HTTP/1.1 chunk: the JSON line plus its newline.
fn chunk(frame: &Value) -> Vec<u8> {
    let line = format!("{frame}\n");
    let mut out = format!("{:x}\r\n", line.len()).into_bytes();
    out.extend_from_slice(line.as_bytes());
    out.extend_from_slice(b"\r\n");
    out
}

fn b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// The most raw output one frame carries: the spec's ceiling.
const FRAME_MAX: usize = 64 * 1024;

/// Read `pipe` into `key` frames of at most [`FRAME_MAX`] bytes as they arrive — or, `buffered`, into
/// one frame at EOF.
fn pump<R: tokio::io::AsyncRead + Unpin + Send + 'static>(
    pipe: Option<R>,
    key: &'static str,
    tx: tokio::sync::mpsc::UnboundedSender<Value>,
    buffered: bool,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let Some(mut pipe) = pipe else { return };
        let mut held = Vec::new();
        let mut buf = vec![0u8; FRAME_MAX];
        while let Ok(n) = pipe.read(&mut buf).await {
            if n == 0 {
                break;
            }
            if buffered {
                held.extend_from_slice(&buf[..n]);
            } else if tx.send(json!({ key: b64(&buf[..n]) })).is_err() {
                return;
            }
        }
        if buffered && !held.is_empty() {
            let _ = tx.send(json!({ key: b64(&held) }));
        }
    })
}

/// v1.2: frames as the command produces them; the process group dies with the connection.
async fn serve_stream(sock: tokio::net::TcpStream, req: Value, shared: Shared) {
    let (mut rd, mut wr) = sock.into_split();
    let head = "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nTransfer-Encoding: chunked\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n";
    if wr.write_all(head.as_bytes()).await.is_err() {
        return;
    }
    let mut cmd = command_for(&req, &shared);
    cmd.stdin(std::process::Stdio::null()).process_group(0);
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            let mut out = chunk(&json!({ "stderr": b64(e.to_string().as_bytes()) }));
            out.extend(chunk(&json!({ "exit_code": 127 })));
            out.extend_from_slice(b"0\r\n\r\n");
            let _ = wr.write_all(&out).await;
            let _ = wr.shutdown().await;
            return;
        }
    };
    let pgid = child.id().unwrap_or(0);
    let index = {
        let mut streams = shared.streams.lock().unwrap();
        streams.push(StreamRecord {
            pgid,
            command: req["command"].as_str().unwrap_or("").to_string(),
            args: args_of(&req),
            started: Instant::now(),
            end: None,
        });
        streams.len() - 1
    };
    let finish = |end: StreamEnd| {
        shared.streams.lock().unwrap()[index].end = Some((end, Instant::now()));
    };

    // Frames from the pipe readers and the heartbeat funnel through one writer, so each stream's
    // frames stay in order and a failed write (the client is gone) is noticed in one place.
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Value>();
    let buffered = shared.opts.protocol == Protocol::Buffered;
    let out_task = pump(child.stdout.take(), "stdout", tx.clone(), buffered);
    let err_task = pump(child.stderr.take(), "stderr", tx.clone(), buffered);

    let timeout = Duration::from_millis(req["timeout_ms"].as_u64().unwrap_or(120_000));
    let deadline = tokio::time::Instant::now() + timeout;
    let mut heartbeat = tokio::time::interval(shared.opts.heartbeat);
    heartbeat.tick().await;
    let mut probe = [0u8; 1];
    enum Why {
        Exited(std::process::ExitStatus),
        TimedOut,
        Gone,
    }
    // Set once the shell has exited leaving processes of its group running: the stream goes on
    // until they are gone.
    let mut shell: Option<std::process::ExitStatus> = None;
    let mut leftovers = tokio::time::interval(Duration::from_millis(50));
    let why = loop {
        tokio::select! {
            status = child.wait(), if shell.is_none() => match status {
                Ok(s) if shared.opts.tracks_leftovers && group_alive(pgid) => {
                    use std::os::unix::process::ExitStatusExt as _;
                    let code = s.code().unwrap_or_else(|| 128 + s.signal().unwrap_or(0));
                    // What the shell printed goes first: forward output until the pipes have been
                    // quiet for the 100 ms settle (at most a second: a chatty leftover never goes
                    // quiet), then say it exited.
                    let mut gone = false;
                    let settle_by = tokio::time::Instant::now() + Duration::from_secs(1);
                    while let Ok(Some(frame)) = tokio::time::timeout(
                        Duration::from_millis(100)
                            .min(settle_by.saturating_duration_since(tokio::time::Instant::now())),
                        rx.recv(),
                    )
                    .await
                    {
                        if wr.write_all(&chunk(&frame)).await.is_err() {
                            gone = true;
                            break;
                        }
                    }
                    if gone || wr.write_all(&chunk(&json!({ "shell_exit": code }))).await.is_err() {
                        break Why::Gone;
                    }
                    shell = Some(s);
                }
                Ok(s) => break Why::Exited(s),
                Err(_) => break Why::Gone,
            },
            _ = leftovers.tick(), if shell.is_some() => {
                if !group_alive(pgid) && let Some(s) = shell {
                    break Why::Exited(s);
                }
            }
            Some(frame) = rx.recv() => {
                if wr.write_all(&chunk(&frame)).await.is_err() {
                    break Why::Gone;
                }
            }
            _ = heartbeat.tick(), if !buffered => {
                if wr.write_all(&chunk(&json!({}))).await.is_err() {
                    break Why::Gone;
                }
            }
            // The request is fully read, so anything here — EOF above all — means the client left.
            _ = rd.read(&mut probe) => break Why::Gone,
            () = tokio::time::sleep_until(deadline) => break Why::TimedOut,
        }
    };

    let final_frame = match why {
        Why::Gone => {
            let killed = shared.opts.kills_on_disconnect;
            if killed {
                kill_group(pgid);
                let _ = child.wait().await;
            } else {
                // Leave it running, as a non-conforming endpoint would; drop must not reap it.
                std::mem::forget(child);
            }
            out_task.abort();
            err_task.abort();
            finish(StreamEnd::Disconnected { killed });
            return;
        }
        Why::TimedOut => {
            kill_group(pgid);
            let _ = child.wait().await;
            json!({ "timed_out": true })
        }
        Why::Exited(status) => {
            use std::os::unix::process::ExitStatusExt as _;
            match (status.code(), status.signal()) {
                (Some(code), _) => json!({ "exit_code": code }),
                (None, Some(sig)) => json!({ "signal": sig }),
                (None, None) => json!({ "exit_code": -1 }),
            }
        }
    };
    // Drain what the pipes still hold; a detached grandchild keeping them open is released after a
    // short grace, as the local runner does.
    drop(tx);
    let (mut out_task, mut err_task) = (out_task, err_task);
    let drained = async {
        let _ = (&mut out_task).await;
        let _ = (&mut err_task).await;
    };
    let _ = tokio::time::timeout(Duration::from_millis(200), drained).await;
    out_task.abort();
    err_task.abort();
    let mut tail = Vec::new();
    while let Ok(frame) = rx.try_recv() {
        tail.extend(chunk(&frame));
    }
    tail.extend(chunk(&final_frame));
    tail.extend_from_slice(b"0\r\n\r\n");
    let ok = wr.write_all(&tail).await.is_ok();
    let _ = wr.shutdown().await;
    finish(match (ok, &final_frame) {
        (_, f) if f.get("timed_out").is_some() => StreamEnd::TimedOut,
        _ => StreamEnd::Finished,
    });
}

// ---------------------------------------------------------------------------------------------
// v1.3: sandbox jobs, natively.
// ---------------------------------------------------------------------------------------------

/// One job, as the endpoint keeps it.
struct MockJob {
    pgid: u32,
    label: String,
    started: u64,
    ended: u64,
    /// Output bytes `[base, base + out.len())`.
    out: Vec<u8>,
    base: u64,
    shell_exit: Option<i32>,
    end: Option<i32>,
    kept: bool,
    num: Option<u64>,
    kind: Option<String>,
    seen: u64,
}

/// Every job, by `(scope, id)`, and a counter bumped on every change.
#[derive(Clone)]
struct Jobs {
    map: Arc<Mutex<std::collections::HashMap<(String, String), MockJob>>>,
    changed: Arc<tokio::sync::watch::Sender<u64>>,
}

impl Default for Jobs {
    fn default() -> Self {
        Self {
            map: Arc::default(),
            changed: Arc::new(tokio::sync::watch::Sender::new(0)),
        }
    }
}

impl Jobs {
    fn bump(&self) {
        self.changed.send_modify(|v| *v = v.wrapping_add(1));
    }
}

fn epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// The live (non-zombie) members of group `pgid`.
fn members(pgid: u32) -> Vec<u32> {
    let mut out = Vec::new();
    let Ok(dir) = std::fs::read_dir("/proc") else {
        return out;
    };
    for e in dir.flatten() {
        let Some(pid) = e.file_name().to_str().and_then(|s| s.parse::<u32>().ok()) else {
            continue;
        };
        let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) else {
            continue;
        };
        let Some((_, rest)) = stat.rsplit_once(')') else {
            continue;
        };
        let f: Vec<&str> = rest.split_whitespace().collect();
        if f.len() > 2 && f[0] != "Z" && f[2].parse() == Ok(pgid) {
            out.push(pid);
        }
    }
    out
}

fn signal_group(pgid: u32, sig: &str) {
    let _ = std::process::Command::new("kill")
        .args([sig, "--", &format!("-{pgid}")])
        .stderr(std::process::Stdio::null())
        .status();
}

/// Stop the jobs `pgids` name: `SIGTERM`, up to `grace` for them to go, `SIGKILL`, a second more.
async fn stop_groups(pgids: &[u32], grace: Duration) {
    for &g in pgids {
        signal_group(g, "-TERM");
    }
    let deadline = Instant::now() + grace;
    while pgids.iter().any(|&g| !members(g).is_empty()) && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    for &g in pgids {
        if !members(g).is_empty() {
            signal_group(g, "-KILL");
        }
    }
    let deadline = Instant::now() + Duration::from_secs(1);
    while pgids.iter().any(|&g| !members(g).is_empty()) && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

async fn respond_json(mut sock: tokio::net::TcpStream, status: u16, body: &Value) {
    let body = body.to_string();
    let reason = if status == 200 { "OK" } else { "Error" };
    let resp = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = sock.write_all(resp.as_bytes()).await;
    let _ = sock.shutdown().await;
}

fn state_of(j: &MockJob) -> (&'static str, Value) {
    match (j.end, j.shell_exit) {
        (Some(c), _) => ("ended", json!(c)),
        (None, Some(c)) => ("exited", json!(c)),
        (None, None) => ("running", Value::Null),
    }
}

async fn serve_job(sock: tokio::net::TcpStream, req: Value, shared: Shared) {
    let job = &req["job"];
    let op = job["op"].as_str().unwrap_or("");
    let scope = job["scope"].as_str().unwrap_or("").to_string();
    let key = |id: &str| (scope.clone(), id.to_string());
    let ids: Vec<String> = job["ids"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    let jobs = shared.jobs.clone();
    match op {
        "probe" => respond_json(sock, 200, &json!({ "jobs": "v1.3" })).await,
        "start" => start_job(sock, req.clone(), shared).await,
        "attach" => {
            let id = job["id"].as_str().unwrap_or("").to_string();
            if !jobs.map.lock().unwrap().contains_key(&key(&id)) {
                return respond_json(sock, 404, &json!({ "error": "no such job" })).await;
            }
            let offset = job["offset"].as_u64().unwrap_or(0);
            stream_job(sock, shared, key(&id), offset, false, None).await;
        }
        "read" => {
            let id = job["id"].as_str().unwrap_or("");
            let offset = job["offset"].as_u64().unwrap_or(0);
            let max = job["max"].as_u64().unwrap_or(0) as usize;
            let body = match jobs.map.lock().unwrap().get(&key(id)) {
                None => json!({ "state": "gone", "offset": 0, "end": 0, "data": "" }),
                Some(j) => {
                    let (state, code) = state_of(j);
                    let from = offset.max(j.base);
                    let at = (from - j.base) as usize;
                    let to = (at + max).min(j.out.len());
                    let data = if at < j.out.len() {
                        &j.out[at..to]
                    } else {
                        &[][..]
                    };
                    json!({
                        "state": state,
                        "code": code,
                        "offset": from,
                        "end": j.base + j.out.len() as u64,
                        "data": b64(data),
                    })
                }
            };
            respond_json(sock, 200, &body).await;
        }
        "keep" | "seen" => {
            let id = job["id"].as_str().unwrap_or("");
            let found = {
                let mut map = jobs.map.lock().unwrap();
                match map.get_mut(&key(id)) {
                    Some(j) => {
                        if op == "keep" {
                            j.kept = true;
                            j.num = job["num"].as_u64();
                            j.kind = job["kind"].as_str().map(str::to_string);
                            j.seen = job["seen"].as_u64().unwrap_or(j.seen);
                        } else {
                            j.seen = job["offset"].as_u64().unwrap_or(j.seen);
                        }
                        true
                    }
                    None => false,
                }
            };
            if found {
                respond_json(sock, 200, &json!({})).await;
            } else {
                respond_json(sock, 404, &json!({ "error": "no such job" })).await;
            }
        }
        "kill" | "wipe" => {
            let grace = Duration::from_millis(job["grace_ms"].as_u64().unwrap_or(300));
            let targets: Vec<(String, u32)> = {
                let map = jobs.map.lock().unwrap();
                map.iter()
                    .filter(|((sc, id), j)| {
                        *sc == scope && j.end.is_none() && (op == "wipe" || ids.contains(id))
                    })
                    .map(|((_, id), j)| (id.clone(), j.pgid))
                    .collect()
            };
            let pgids: Vec<u32> = targets.iter().map(|(_, g)| *g).collect();
            stop_groups(&pgids, grace).await;
            let mut survivors = Vec::new();
            for (id, g) in &targets {
                let left = members(*g);
                if !left.is_empty() {
                    survivors.push(json!({ "id": id, "pids": left }));
                }
            }
            let remove = op == "wipe" || job["remove"].as_bool() == Some(true);
            if remove {
                let mut map = jobs.map.lock().unwrap();
                map.retain(|(sc, id), j| {
                    let named = *sc == scope && (op == "wipe" || ids.contains(id));
                    !(named && members(j.pgid).is_empty())
                });
            }
            jobs.bump();
            respond_json(sock, 200, &json!({ "survivors": survivors })).await;
        }
        "remove" => {
            jobs.map.lock().unwrap().retain(|(sc, id), j| {
                !(*sc == scope
                    && ids.contains(id)
                    && (j.end.is_some() || members(j.pgid).is_empty()))
            });
            jobs.bump();
            respond_json(sock, 200, &json!({})).await;
        }
        "list" => {
            let list: Vec<Value> = jobs
                .map
                .lock()
                .unwrap()
                .iter()
                .filter(|((sc, _), _)| *sc == scope)
                .map(|((_, id), j)| {
                    let (state, code) = state_of(j);
                    json!({
                        "id": id,
                        "state": state,
                        "code": code,
                        "kept": j.kept,
                        "num": j.num,
                        "kind": j.kind,
                        "seen": j.seen,
                        "started": j.started,
                        "ended": j.ended,
                        "label": j.label,
                    })
                })
                .collect();
            respond_json(sock, 200, &json!({ "jobs": list })).await;
        }
        other => {
            respond_json(
                sock,
                400,
                &json!({ "error": format!("unknown job op {other:?}") }),
            )
            .await
        }
    }
}

/// Start a job: its own process group, its output kept in memory under the cap, its end recorded
/// once the group is empty.
async fn start_job(sock: tokio::net::TcpStream, req: Value, shared: Shared) {
    let job = &req["job"];
    let scope = job["scope"].as_str().unwrap_or("").to_string();
    let id = job["id"].as_str().unwrap_or("").to_string();
    let hold = job["hold"].as_bool() == Some(true);
    let key = (scope, id);
    if shared.jobs.map.lock().unwrap().contains_key(&key) {
        return respond_json(sock, 409, &json!({ "error": "the job exists" })).await;
    }
    let mut cmd = command_for(&req, &shared);
    cmd.stdin(std::process::Stdio::null())
        .process_group(0)
        .kill_on_drop(false);
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return respond_json(sock, 500, &json!({ "error": e.to_string() })).await,
    };
    let pgid = child.id().unwrap_or(0);
    shared.jobs.map.lock().unwrap().insert(
        key.clone(),
        MockJob {
            pgid,
            label: job["label"].as_str().unwrap_or("").to_string(),
            started: epoch(),
            ended: 0,
            out: Vec::new(),
            base: 0,
            shell_exit: None,
            end: None,
            kept: !hold,
            num: None,
            kind: (!hold).then(|| "background".to_string()),
            seen: 0,
        },
    );
    // Both pipes into one buffer, in arrival order.
    let pump = |pipe: Option<Box<dyn tokio::io::AsyncRead + Unpin + Send>>| {
        let jobs = shared.jobs.clone();
        let key = key.clone();
        let cap = shared.opts.job_output_cap;
        tokio::spawn(async move {
            let Some(mut pipe) = pipe else { return };
            let mut buf = vec![0u8; FRAME_MAX];
            while let Ok(n) = pipe.read(&mut buf).await {
                if n == 0 {
                    break;
                }
                if let Some(j) = jobs.map.lock().unwrap().get_mut(&key) {
                    j.out.extend_from_slice(&buf[..n]);
                    if j.out.len() > cap {
                        let drop_n = j.out.len() - cap;
                        j.out.drain(..drop_n);
                        j.base += drop_n as u64;
                    }
                }
                jobs.bump();
            }
        })
    };
    let out_task = pump(child.stdout.take().map(|p| Box::new(p) as Box<_>));
    let err_task = pump(child.stderr.take().map(|p| Box::new(p) as Box<_>));
    {
        let jobs = shared.jobs.clone();
        let key = key.clone();
        tokio::spawn(async move {
            use std::os::unix::process::ExitStatusExt as _;
            let status = child.wait().await;
            let code = status
                .map(|s| s.code().unwrap_or_else(|| 128 + s.signal().unwrap_or(0)))
                .unwrap_or(-1);
            if !members(pgid).is_empty() {
                if let Some(j) = jobs.map.lock().unwrap().get_mut(&key) {
                    j.shell_exit = Some(code);
                }
                jobs.bump();
                while !members(pgid).is_empty() {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
            }
            let drained = async {
                let _ = out_task.await;
                let _ = err_task.await;
            };
            let _ = tokio::time::timeout(Duration::from_millis(200), drained).await;
            if let Some(j) = jobs.map.lock().unwrap().get_mut(&key) {
                j.end = Some(code);
                j.ended = epoch();
            }
            jobs.bump();
        });
    }
    if hold {
        stream_job(sock, shared, key, 0, true, Some(pgid)).await;
    } else {
        respond_json(sock, 200, &json!({ "started": pgid })).await;
    }
}

/// Stream a job's output from `offset` to its end. `started`: a held start, which says so first —
/// and whose job dies with the connection unless it was kept by then.
async fn stream_job(
    sock: tokio::net::TcpStream,
    shared: Shared,
    key: (String, String),
    offset: u64,
    hold: bool,
    started: Option<u32>,
) {
    let (mut rd, mut wr) = sock.into_split();
    let head = "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nTransfer-Encoding: chunked\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n";
    let mut gone = wr.write_all(head.as_bytes()).await.is_err();
    if let (false, Some(pgid)) = (gone, started) {
        gone = wr
            .write_all(&chunk(&json!({ "started": pgid })))
            .await
            .is_err();
    }
    let mut changed = shared.jobs.changed.subscribe();
    let mut sent = offset;
    let mut shell_told = false;
    let mut probe = [0u8; 1];
    while !gone {
        changed.mark_unchanged();
        enum Next {
            Frame(Value),
            End(i32),
            Wait,
            Missing,
        }
        let next = {
            let map = shared.jobs.map.lock().unwrap();
            match map.get(&key) {
                None => Next::Missing,
                Some(j) => {
                    let total = j.base + j.out.len() as u64;
                    if sent < j.base {
                        let n = j.base - sent;
                        sent = j.base;
                        Next::Frame(json!({ "gap": n }))
                    } else if sent < total {
                        let at = (sent - j.base) as usize;
                        let to = (at + FRAME_MAX).min(j.out.len());
                        sent += (to - at) as u64;
                        Next::Frame(json!({ "stdout": b64(&j.out[at..to]) }))
                    } else if let (Some(c), false) = (j.shell_exit, shell_told) {
                        shell_told = true;
                        Next::Frame(json!({ "shell_exit": c }))
                    } else if let Some(c) = j.end {
                        Next::End(c)
                    } else {
                        Next::Wait
                    }
                }
            }
        };
        match next {
            Next::Frame(f) => gone = wr.write_all(&chunk(&f)).await.is_err(),
            Next::End(c) => {
                let mut tail = chunk(&json!({ "exit_code": c }));
                tail.extend_from_slice(b"0\r\n\r\n");
                let _ = wr.write_all(&tail).await;
                let _ = wr.shutdown().await;
                return;
            }
            Next::Missing => {
                let mut tail = chunk(&json!({ "error": "the job is gone" }));
                tail.extend_from_slice(b"0\r\n\r\n");
                let _ = wr.write_all(&tail).await;
                return;
            }
            Next::Wait => {
                tokio::select! {
                    _ = changed.changed() => {}
                    () = tokio::time::sleep(shared.opts.heartbeat) => {
                        gone = wr.write_all(&chunk(&json!({}))).await.is_err();
                    }
                    _ = rd.read(&mut probe) => gone = true,
                }
            }
        }
    }
    // The connection is gone before the job's end: a held job nobody kept dies with it.
    if hold && shared.opts.jobs_hold_kills {
        let pgid = {
            let map = shared.jobs.map.lock().unwrap();
            map.get(&key)
                .filter(|j| !j.kept && j.end.is_none())
                .map(|j| j.pgid)
        };
        if let Some(g) = pgid {
            signal_group(g, "-KILL");
        }
    }
}
