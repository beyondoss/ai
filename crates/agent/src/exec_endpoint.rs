//! Executing commands against a remote endpoint — any endpoint.
//!
//! The agent is handed a **URL** and it POSTs commands to it. That is the entire contract. It does not
//! know, and must never know, what is on the other side: Daytona, E2B, Modal, a container behind an
//! internal service, someone's laptop over a tunnel. Nothing here is specific to any vendor.
//!
//! ## Not the agent's business
//!
//! **Lifecycle.** Creating, warming, snapshotting, pausing or destroying a sandbox is the caller's job.
//! The agent receives an endpoint that already works and uses it until told otherwise. An agent that
//! provisions infrastructure has to model quotas, billing, placement and cleanup — none of which has
//! anything to do with running a coding task, and all of which differ per vendor.
//!
//! **Co-location.** The endpoint is a URL, so the agent runs wherever it likes and the sandbox runs
//! wherever it likes. Any design that shells out to a locally-installed control-plane binary quietly
//! requires the agent to sit on the compute node beside the workloads it is meant to be isolated from,
//! which is backwards.
//!
//! ## The protocol (v1.1)
//!
//! One `POST` to the configured URL:
//!
//! ```jsonc
//! // request
//! { "command": "rg", "args": ["--files"], "cwd": "/work", "timeout_ms": 120000 }
//! // request carrying stdin (v1.1, optional) — standard base64, fed to the command, then closed
//! { "command": "cat", "args": [], "timeout_ms": 10000, "stdin_base64": "c3RkaW4tcHJvYmU=" }
//! // response — 200
//! { "exit_code": 0, "stdout": "…", "stderr": "…" }
//! ```
//!
//! Deliberately the smallest thing that can carry a command and its result. Putting an adapter in
//! front of a vendor's SDK is a few dozen lines on their side, and that seam is where vendor
//! specifics belong — not in here.
//!
//! **v1.1 is backward compatible.** `stdin_base64` is the only addition, and only `ShellFs`'s file
//! writes send it — after a probe (`cat` must echo `stdin-probe`) has shown the endpoint honors it. A
//! v1 endpoint ignores the unknown field and answers the probe with empty stdout, so writes fall back
//! to argv-sized chunks. The field is never trusted blindly: writing through an endpoint that dropped
//! it would silently produce *empty files*.
//!
//! **The response is capped** ([`DEFAULT_MAX_RESPONSE_BYTES`], `--exec-max-response-bytes`). The body
//! is read incrementally and reading stops at the cap; an oversized response is an **error**, never a
//! partial result. The body is one JSON object, so a prefix of it cannot be parsed — there is no
//! honest partial `ExecResult` to return (not even the exit code, which may come after the output),
//! and presenting whatever fit as the command's output would be a confident wrong answer. The error
//! says the command *did run*, so the caller narrows its output instead of blindly repeating a
//! command that may not be idempotent. The cap applies to v1.1 objects only; a v1.2 stream is read
//! a frame at a time and never buffered whole.
//!
//! ## Streaming (v1.2)
//!
//! A tool that behaves differently depending on where it runs is a broken API. Over v1.1, `bash`
//! cannot see a remote command's output until it exits, so it cannot tell a silent build from a hung
//! one (the idle yield never fires: a hung command burns its whole timeout), cannot show progress,
//! and cannot kill a command it has given up on. v1.2 streams the output and makes **the connection
//! the command's lifetime**, so the endpoint stays stateless — no job ids, no poll or kill routes.
//!
//! The key words MUST, MUST NOT, SHOULD and MAY are used as in RFC 2119.
//!
//! **Request.** The v1.1 request plus `"stream": true`:
//!
//! ```jsonc
//! { "command": "sh", "args": ["-c", "cargo build"], "cwd": "/work", "timeout_ms": 2147483647, "stream": true }
//! ```
//!
//! **Response.** `200` with `Content-Type: application/x-ndjson`, a body of frames, one JSON object
//! per line, each terminated by `\n` (chunked transfer on HTTP/1.1, DATA frames on HTTP/2; no
//! `Content-Length`). A frame carries exactly one of:
//!
//! | frame | meaning |
//! |---|---|
//! | `{"stdout": "<base64>"}` | bytes the command wrote to stdout |
//! | `{"stderr": "<base64>"}` | bytes the command wrote to stderr |
//! | `{}` | heartbeat: the connection is alive; never output |
//! | `{"shell_exit": N}` | the command exited with code N, but processes of its group run on — see "Leftover processes"; not final, never output |
//! | `{"exit_code": N}` | **final**: the command exited with code N |
//! | `{"signal": S}` | **final**: the command was killed by signal S (9 for SIGKILL) |
//! | `{"timed_out": true}` | **final**: the endpoint killed the command at `timeout_ms` |
//! | `{"error": "<message>"}` | **final**: the endpoint failed after answering 200 (lost the sandbox) |
//!
//! - Output is **standard base64 with padding** (RFC 4648 §4), so arbitrary bytes — invalid UTF-8,
//!   NULs, terminal escapes — survive. An output frame MUST NOT be empty and MUST carry at most
//!   64 KiB of decoded bytes ([`stream::MAX_FRAME_PAYLOAD`]); split anything longer.
//! - **Exactly one final frame, last.** After it the endpoint MUST end the body. A body that ends
//!   without one is a broken stream, never a success. Failures *before* answering are non-2xx
//!   statuses exactly as in v1.1; a command that cannot be spawned is an ordinary result (`stderr`
//!   frame, then `{"exit_code": 127}`), as a shell reports it.
//! - **Ordering.** Frames of one stream are in the order the bytes were read; the relative order of
//!   stdout and stderr is best-effort, as between two local pipes. The final frame comes after all
//!   output read from both pipes. Once the command's process group is empty, the endpoint SHOULD
//!   stop waiting for pipes held open by a descendant that left the group after a short grace (the
//!   client uses 100 ms) rather than until they close.
//! - **Flushing.** Each frame MUST reach the connection without buffering: the endpoint MAY coalesce
//!   output read within 100 ms into one frame, and MUST NOT hold any longer. Anything between the
//!   endpoint and the client must not buffer either (send `Cache-Control: no-store`, and e.g.
//!   `X-Accel-Buffering: no` behind nginx) — the probe below refuses an endpoint whose frames arrive
//!   all at once.
//! - **Heartbeat.** When no frame has been written for [`stream::HEARTBEAT_INTERVAL`] (5 s), the
//!   endpoint MUST write `{}`. It keeps a silent command's connection alive through proxies (whose
//!   idle timeouts start around 60 s), and it bounds how long a vanished client can go unnoticed. A
//!   client MUST NOT treat a heartbeat as output — `bash` measures a command's silence on output
//!   frames only — and this client treats 30 s with no frame at all as a dead connection.
//! - **Unknown fields are ignored**, and a frame with no known field is a heartbeat, so a later
//!   version can add fields (a sequence number, timings) without breaking this client.
//! - **Timeouts.** `timeout_ms` is the endpoint's to enforce: at the deadline it MUST kill the
//!   command's whole process group and send `{"timed_out": true}`. The client's own deadline is
//!   `timeout_ms` + 30 s for the whole exchange, so the endpoint's verdict normally wins. `bash`
//!   jobs have no deadline — they end with the turn, the run or the session, or a `bash_job` kill,
//!   all of which close the stream — so a streamed job sends [`STREAM_DEADLINE`] (`i32::MAX` ms,
//!   ~24.8 days, the most a Node endpoint's `setTimeout` can hold): the connection is the lifetime,
//!   and the deadline only bounds one that outlives every session. Over v1.1 nothing streams and
//!   nothing can be killed, so a `bash` command there sends [`UNTRACKED_DEADLINE`] (30 minutes): the
//!   only thing that ends a command that hangs.
//!
//! **Disconnect kills — normative.** Every streamed command MUST run in its own process group (or an
//! equivalent the endpoint can kill as a unit, such as a cgroup). **If the connection closes before
//! the final frame, the endpoint MUST kill that whole group** — SIGKILL, grandchildren included —
//! within two heartbeat intervals (10 s), and SHOULD do so at once by watching the connection for
//! EOF or `RST_STREAM` rather than waiting for a heartbeat write to fail. This is how the agent stops
//! a command: `bash_job` kill, a cancelled call, a dropped job and the agent's own exit all close the
//! stream (see [`Kill`]), and nothing else is ever sent.
//!
//! **Leftover processes — normative.** A command often exits leaving processes it started running
//! (`server &`, `nohup … &`, a daemon that stays in its group). Locally those become a job the agent
//! tracks and stops with the session; remotely they must too, or `nohup server &` holds its port
//! until the sandbox dies. So when the command's own process exits **while its process group still
//! has a live (non-zombie) member**, the endpoint:
//!
//! - MUST send `{"shell_exit": N}` — N the command's exit code, `128 + S` if signal S killed it —
//!   after the output read from both pipes up to that exit (the same short grace as before a final
//!   frame);
//! - MUST then keep the stream open — output, heartbeats, `timeout_ms` and disconnect-kills all
//!   still apply — until the group has no live member, and only then send the final frame, which
//!   carries the command's own result (`{"exit_code": N}`, or `{"signal": S}`).
//!
//! A group that is empty when the command exits gets the final frame directly; `shell_exit` is
//! never sent for it. Zombies do not count: a sandbox whose pid 1 is not an init (`sleep infinity`)
//! never reaps the orphans it inherits. A process that leaves the group (`setsid`) is not covered,
//! unless the endpoint tracks a cgroup (it MAY). The client adopts the leftovers as a job at
//! `shell_exit` (`on_orphans`), and closes the stream to stop them, like any job. A call that is not
//! a job (nothing tracks it) returns at `shell_exit` and closes the stream, so the endpoint kills
//! the leftovers. Compatible both ways: an endpoint predating this rule ends the stream with the
//! shell (its leftovers run on, untracked), and a client predating it reads `shell_exit` as a
//! heartbeat (an unknown field) and keeps the job until the group ends.
//!
//! **Over v1.1 leftovers are untracked.** One request, one response: the endpoint answers when the
//! command exits, and nothing the agent can do afterwards reaches what it left running. A v1.1
//! endpoint SHOULD kill the command's process group once it has answered; one that does not leaves
//! `server &` running until the sandbox ends.
//!
//! **The probe.** Streaming is adopted only after the endpoint proves it, once per runner, on the
//! first streaming call ([`HttpExecRunner::streams`]): `sh -c 'printf a; sleep 0.3; printf b'` with
//! `"stream": true` must come back as `application/x-ndjson` with at least two output frames, `ab`,
//! `{"exit_code": 0}`, and the first output frame at least 100 ms before the final one. A v1.1
//! endpoint answers with one JSON object (it ignores the unknown field); a buffering endpoint or proxy
//! sends one frame, or every frame at once. Either way the runner stays on v1.1 for good, and `bash`
//! keeps v1.1's behavior exactly: no yield, no live progress, and its description says so. A probe
//! that could not reach the endpoint is not an answer and is retried on the next call. Only `bash`
//! streams: the filesystem backend's commands stay v1.1, whose single object they need anyway.
//!
//! **Conformance.** [`conformance::verify`] checks an endpoint against all of the above — framing,
//! the probe, exit/signal/timeout frames, heartbeats, the 64 KiB frame cap, kill on disconnect
//! measured against a grandchild's marker file, and leftover processes (`shell_exit`, the stream
//! held until they exit, killed on a disconnect after it) — and `tests/exec_conformance.rs` runs it against any
//! URL. `tests/fixtures/exec/v1.2-frames.json` holds the decoding vectors.
//!
//! For targets that have no HTTP surface at all (`ssh`, `docker exec`, `kubectl exec`, any CLI), see
//! [`TemplateRunner`], which is the same idea expressed as an argv template. Its transport is a local
//! process, so its output streams natively; killing it is the hard part — see `far_kill`.

use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine as _;
use serde::{Deserialize, Serialize};

use crate::tools::exec::{
    ChunkSink, CommandRunner, ExecResult, JobHooks, Kill, STREAM_HEAD, STREAM_TAIL, SpawnSink,
    UNTRACKED_DEADLINE, drain_capped, feed_stdin,
};

pub mod conformance;
mod far_kill;
pub mod stream;

use stream::{FRAME_SILENCE_LIMIT, Frame, FrameReader, NDJSON};

/// What the agent sends.
#[derive(Debug, Serialize)]
struct ExecRequest<'a> {
    command: &'a str,
    args: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    cwd: Option<&'a str>,
    timeout_ms: u64,
    /// v1.1: bytes for the command's stdin, standard base64. Absent means no stdin — v1 behavior.
    #[serde(skip_serializing_if = "Option::is_none")]
    stdin_base64: Option<String>,
    /// v1.2: answer with a stream of frames instead of one object. Absent means v1.1.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
}

/// What the agent expects back. Extra fields are ignored, so an endpoint may return more.
#[derive(Debug, Deserialize)]
struct ExecResponse {
    exit_code: i32,
    #[serde(default)]
    stdout: String,
    #[serde(default)]
    stderr: String,
}

/// The `timeout_ms` a streamed `bash` job sends: `i32::MAX` ms (~24.8 days), the longest deadline
/// every endpoint can honor — Node's `setTimeout` fires at once for anything longer. A job's lifetime
/// is the connection's (see "Disconnect kills"), so this only has to outlast any session; it is not a
/// limit anyone should meet.
pub const STREAM_DEADLINE: Duration = Duration::from_millis(i32::MAX as u64);

/// What a remote command left running, as `on_orphans` describes it: the runner cannot list the far
/// side's processes, only that its process group still has some.
pub(crate) const REMOTE_LEFTOVERS: &str = "processes on the exec target";

/// How long a template command's output may still be in flight once its shell-exit marker has
/// arrived — the local runner's 100 ms (`bash_proc::SETTLE`).
const SHELL_EXIT_SETTLE: Duration = Duration::from_millis(100);

/// Where a runner reports a shell that exited leaving processes running — [`JobHooks::on_orphans`].
type OrphanSink<'a> = &'a (dyn Fn(&ExecResult, String) + Send + Sync);

/// The default response-body cap for [`HttpExecRunner`]: 16 MiB.
///
/// Far above anything a tool consumes — `bash` keeps ~30 KB of output and the local runner keeps
/// 256 KiB per stream — so a legitimate response never meets it. What it bounds is the response that
/// isn't legitimate: a `cat` of a multi-gigabyte file must not be buffered whole into a process that
/// may be serving other sessions.
pub const DEFAULT_MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

/// A [`CommandRunner`] that POSTs each command to a URL.
///
/// Holds a `reqwest::Client` so connections are pooled across calls, but no session and no state — a
/// call is a request. There is nothing to reconnect, nothing to keep alive, and no lifecycle to get
/// wrong.
pub struct HttpExecRunner {
    url: String,
    client: reqwest::Client,
    headers: Vec<(String, String)>,
    max_response_bytes: usize,
    /// Whether the endpoint has proved it streams (v1.2) — probed once, on the first streaming call,
    /// and cached. A probe that could not reach the endpoint is not cached: that is an outage, not
    /// an answer.
    streams: tokio::sync::OnceCell<bool>,
}

impl HttpExecRunner {
    /// Point the runner at `url`.
    pub fn new(url: impl Into<String>) -> Result<Self, String> {
        let url = url.into();
        // Fail at construction rather than on the first tool call, so a typo is an immediate startup
        // error instead of a confusing mid-turn failure the model tries to work around.
        let parsed =
            reqwest::Url::parse(&url).map_err(|e| format!("invalid exec URL {url:?}: {e}"))?;
        if !matches!(parsed.scheme(), "http" | "https") {
            return Err(format!(
                "exec URL must be http or https, got {:?}",
                parsed.scheme()
            ));
        }
        Ok(Self {
            url,
            client: {
                agent_core::ensure_provider();
                reqwest::Client::new()
            },
            headers: Vec::new(),
            max_response_bytes: DEFAULT_MAX_RESPONSE_BYTES,
            streams: tokio::sync::OnceCell::new(),
        })
    }

    /// Cap how much of a response body is read before the call fails. See the module doc for why an
    /// oversized response is an error rather than a truncated result.
    pub fn with_max_response_bytes(mut self, max: usize) -> Self {
        self.max_response_bytes = max;
        self
    }

    /// Add a header sent with every request — an `Authorization`, an API key, a tenant id. Every real
    /// endpoint needs some form of auth, and which form is the endpoint's business, not the agent's.
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Parse a `Name: value` pair, the shape a CLI flag or env var carries.
    pub fn parse_header(raw: &str) -> Result<(String, String), String> {
        let (name, value) = raw
            .split_once(':')
            .ok_or_else(|| format!("header must be `Name: value`, got {raw:?}"))?;
        let name = name.trim();
        if name.is_empty() {
            return Err(format!("header name is empty in {raw:?}"));
        }
        Ok((name.to_string(), value.trim().to_string()))
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    /// Send one request and return the response once its status is known to be a success. `stream`
    /// asks for v1.2 frames; whether they came is the response's `Content-Type` to say.
    async fn send(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        stdin: Option<&[u8]>,
        stream: bool,
    ) -> std::io::Result<reqwest::Response> {
        let body = ExecRequest {
            command: program,
            args,
            cwd,
            timeout_ms: timeout.as_millis().min(u64::MAX as u128) as u64,
            stdin_base64: stdin.map(|b| base64::engine::general_purpose::STANDARD.encode(b)),
            stream,
        };
        let mut req = self.client.post(&self.url).json(&body);
        for (name, value) in &self.headers {
            req = req.header(name.as_str(), value.as_str());
        }
        // A little past the command's own budget, so the endpoint gets the chance to enforce and
        // report the timeout itself — its message is more useful than a bare client-side abort. It
        // bounds the whole exchange, a stream's body included.
        let resp = req
            .timeout(timeout.saturating_add(Duration::from_secs(30)))
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        format!("exec endpoint {} timed out: {e}", self.url),
                    )
                } else {
                    std::io::Error::other(format!("exec endpoint {} unreachable: {e}", self.url))
                }
            })?;
        let status = resp.status();
        if !status.is_success() {
            // The endpoint refused the *request* — a bad URL, auth, a dead sandbox. Distinct from the
            // command running and exiting non-zero, which is an ordinary result.
            let (body, _) = self.read_capped(resp).await?;
            return Err(std::io::Error::other(format!(
                "exec endpoint {} returned {}: {}",
                self.url,
                status.as_u16(),
                excerpt(&body)
            )));
        }
        Ok(resp)
    }

    /// One request, one response: the whole v1.1 protocol.
    async fn post(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        stdin: Option<&[u8]>,
    ) -> std::io::Result<ExecResponse> {
        let resp = self.send(program, args, cwd, timeout, stdin, false).await?;
        self.envelope(resp).await
    }

    /// Read and parse a v1.1 response body: one JSON object, capped.
    async fn envelope(&self, resp: reqwest::Response) -> std::io::Result<ExecResponse> {
        let (body, over_cap) = self.read_capped(resp).await?;
        if over_cap {
            return Err(std::io::Error::other(format!(
                "exec endpoint {}: the response exceeded the {}-byte cap and was discarded — the \
                 command ran, but its output is too large to return; narrow it (e.g. pipe through \
                 `head`) rather than re-running it unchanged",
                self.url, self.max_response_bytes
            )));
        }
        serde_json::from_slice(&body).map_err(|e| {
            std::io::Error::other(format!(
                "exec endpoint {} returned unparseable JSON ({e}): {}",
                self.url,
                excerpt(&body)
            ))
        })
    }

    /// Read the body incrementally, stopping at the cap instead of buffering whatever the endpoint
    /// sends. `true` means the cap was hit and the bytes returned are only a prefix.
    async fn read_capped(&self, mut resp: reqwest::Response) -> std::io::Result<(Vec<u8>, bool)> {
        let cap = self.max_response_bytes;
        // A declared length over the cap fails before a byte of it is read.
        let declared = resp.content_length().unwrap_or(0);
        if declared > cap as u64 {
            return Ok((Vec::new(), true));
        }
        let mut body = Vec::with_capacity(declared as usize);
        while let Some(chunk) = resp
            .chunk()
            .await
            .map_err(|e| std::io::Error::other(format!("exec endpoint {}: {e}", self.url)))?
        {
            if chunk.len() > cap - body.len() {
                // Dropping `resp` abandons the rest of the body along with its connection.
                return Ok((body, true));
            }
            body.extend_from_slice(&chunk);
        }
        Ok((body, false))
    }
}

#[async_trait]
impl CommandRunner for HttpExecRunner {
    async fn run_with_stdin(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        stdin: &[u8],
    ) -> std::io::Result<ExecResult> {
        // Sent as `stdin_base64`. A v1 endpoint ignores it and runs the command with no input, so an
        // `Ok` here is not proof of delivery — see the module doc on the probe.
        let parsed = self.post(program, args, cwd, timeout, Some(stdin)).await?;
        Ok(ExecResult {
            code: Some(parsed.exit_code),
            stdout: parsed.stdout,
            stderr: parsed.stderr,
            ..Default::default()
        })
    }

    async fn run(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
    ) -> std::io::Result<ExecResult> {
        let parsed = self.post(program, args, cwd, timeout, None).await?;
        Ok(ExecResult {
            code: Some(parsed.exit_code),
            signal: None,
            stdout: parsed.stdout,
            stderr: parsed.stderr,
            timed_out: false,
            truncated: false,
        })
    }

    /// v1.2 when the endpoint has proved it streams; otherwise v1.1, which never calls the sink.
    ///
    /// The sink gets *every* byte of *both* streams or none: a v1.1 answer (or a v1.2 endpoint that
    /// answers this request with one object anyway) goes back as `stdout`/`stderr` for `bash` to
    /// assemble. An earlier override fed the sink stdout alone, told `bash` the output had streamed,
    /// and every remote command that printed to stdout lost its stderr.
    async fn run_streaming(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        on_chunk: ChunkSink<'_>,
    ) -> std::io::Result<ExecResult> {
        self.run_stream(program, args, cwd, timeout, on_chunk, None, None)
            .await
    }

    /// As [`Self::run_streaming`], reporting a [`Kill`] once the stream is open — which is also the
    /// promise that the output is live. Killing it closes the stream; the endpoint MUST then kill the
    /// command's process group (see the module doc).
    ///
    /// The job has no deadline of its own, but the wire needs a `timeout_ms`: a stream sends
    /// [`STREAM_DEADLINE`], so the connection (closed by a kill, a dropped job or the agent's exit) is
    /// what ends it; a v1.1 endpoint gets [`UNTRACKED_DEADLINE`], the only thing that can end a
    /// command it cannot stream or stop. See the module doc's "Timeouts".
    async fn run_streaming_tracked(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        hooks: &JobHooks<'_>,
    ) -> std::io::Result<ExecResult> {
        let deadline = if self.streams().await {
            STREAM_DEADLINE
        } else {
            UNTRACKED_DEADLINE
        };
        // The endpoint owns the command's process group: it says when the shell exits leaving some
        // of it running (`shell_exit`), and keeps the stream — and so the job — open until the group
        // is empty. See the module doc's "Leftover processes".
        self.run_stream(
            program,
            args,
            cwd,
            deadline,
            hooks.on_chunk,
            Some(hooks.on_spawn),
            Some(hooks.on_orphans),
        )
        .await
    }

    async fn tracks_jobs(&self) -> bool {
        self.streams().await
    }
}

/// The start of a body, for an error message.
fn excerpt(body: &[u8]) -> String {
    String::from_utf8_lossy(body)
        .trim()
        .chars()
        .take(400)
        .collect()
}

/// How long a [`Kill`] of a stream waits for the runner to confirm it has let go of the connection.
/// The runner's future runs on the async runtime, which a process on its way out may no longer poll;
/// past this the exit itself closes the socket, which the endpoint sees the same way.
const STOP_WAIT: Duration = Duration::from_secs(2);

/// The shared state between a stream's runner future and the [`Kill`] handed out for it.
#[derive(Default)]
struct StreamStop {
    /// Asked to stop: the runner drops the response, which closes the connection.
    requested: tokio::sync::Notify,
    /// Set once the runner future has ended (by any path, a drop included) and the response is gone.
    released: std::sync::Mutex<bool>,
    released_cv: std::sync::Condvar,
}

impl StreamStop {
    /// Blocking: ask the runner to drop the stream and wait (bounded) until it has.
    fn stop(&self) {
        self.requested.notify_one();
        let released = self.released.lock().unwrap_or_else(|e| e.into_inner());
        let _ = self
            .released_cv
            .wait_timeout_while(released, STOP_WAIT, |r| !*r);
    }
}

/// Marks the stream released when the runner future ends — dropped by an abort, or returning. Must
/// be declared before the response in the future's scope, so it drops *after* the response does.
struct Released(Arc<StreamStop>);

impl Drop for Released {
    fn drop(&mut self) {
        *self.0.released.lock().unwrap_or_else(|e| e.into_inner()) = true;
        self.0.released_cv.notify_all();
    }
}

impl HttpExecRunner {
    /// Whether this endpoint streams (v1.2), probing it the first time. See the module doc's "The
    /// probe".
    pub async fn streams(&self) -> bool {
        self.streams
            .get_or_try_init(|| async {
                let resp = self
                    .send(
                        "sh",
                        &["-c".to_string(), stream::PROBE_SCRIPT.to_string()],
                        None,
                        Duration::from_secs(10),
                        None,
                        true,
                    )
                    .await?;
                if !is_ndjson(&resp) {
                    return Ok::<bool, std::io::Error>(false);
                }
                let seen = stream::observe_probe(resp)
                    .await
                    .map_err(std::io::Error::other)?;
                let streams = seen.streams();
                if !streams {
                    tracing::info!(url = %self.url, why = %seen.explain(), "exec endpoint does not stream; `bash` waits for each command to exit");
                }
                Ok(streams)
            })
            .await
            .copied()
            .unwrap_or(false)
    }

    /// The v1.2 exchange. Falls back to v1.1 when the endpoint has not proved it streams, or when it
    /// answers this request with one JSON object after all (a fleet mid-rollout).
    #[allow(clippy::too_many_arguments)]
    async fn run_stream(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        on_chunk: ChunkSink<'_>,
        on_spawn: Option<SpawnSink<'_>>,
        on_orphans: Option<OrphanSink<'_>>,
    ) -> std::io::Result<ExecResult> {
        if !self.streams().await {
            return self.run(program, args, cwd, timeout).await;
        }
        let stop = Arc::new(StreamStop::default());
        // Declared before the response: dropped after it, so "released" is only ever set once the
        // connection has been let go.
        let _released = Released(stop.clone());
        let resp = self.send(program, args, cwd, timeout, None, true).await?;
        if !is_ndjson(&resp) {
            let parsed = self.envelope(resp).await?;
            return Ok(ExecResult {
                code: Some(parsed.exit_code),
                stdout: parsed.stdout,
                stderr: parsed.stderr,
                ..Default::default()
            });
        }
        if let Some(report) = on_spawn {
            let stop = stop.clone();
            report(Kill::Hook(Arc::new(move || stop.stop())));
        }
        let mut frames = FrameReader::new(resp);
        let broken = |e: String| {
            std::io::Error::other(format!(
                "exec endpoint {}: {e} (the command may have been killed)",
                self.url
            ))
        };
        loop {
            let next = tokio::select! {
                biased;
                () = stop.requested.notified() => {
                    return Err(std::io::Error::other("stopped: the exec stream was closed"));
                }
                next = frames.next(FRAME_SILENCE_LIMIT) => next,
            };
            match next.map_err(broken)? {
                Some(Frame::Stdout(b) | Frame::Stderr(b)) => on_chunk(&b),
                Some(Frame::Heartbeat) => {}
                Some(Frame::ShellExit(code)) => {
                    let shell = ExecResult {
                        code: Some(code),
                        ..Default::default()
                    };
                    match on_orphans {
                        // A job: its leftovers are adopted, and the stream stays open until they
                        // have exited or a kill closes it.
                        Some(sink) => sink(&shell, REMOTE_LEFTOVERS.to_string()),
                        // A call that ends with its command: returning drops the stream, and the
                        // endpoint kills what is left.
                        None => return Ok(shell),
                    }
                }
                Some(Frame::Exit(code)) => {
                    frames.finish().await;
                    return Ok(ExecResult {
                        code: Some(code),
                        ..Default::default()
                    });
                }
                Some(Frame::Signal(signal)) => {
                    frames.finish().await;
                    return Ok(ExecResult {
                        signal: Some(signal),
                        ..Default::default()
                    });
                }
                Some(Frame::TimedOut) => {
                    frames.finish().await;
                    return Ok(ExecResult {
                        timed_out: true,
                        stderr: "command exceeded its timeout".into(),
                        ..Default::default()
                    });
                }
                Some(Frame::Error(e)) => {
                    return Err(broken(format!("the endpoint failed mid-command: {e}")));
                }
                None => {
                    return Err(broken(
                        "the stream ended before the command's final frame".to_string(),
                    ));
                }
            }
        }
    }
}

/// Whether a response is a v1.2 stream rather than a v1.1 object.
fn is_ndjson(resp: &reqwest::Response) -> bool {
    resp.headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| {
            v.split(';')
                .next()
                .is_some_and(|m| m.trim().eq_ignore_ascii_case(NDJSON))
        })
}

/// A [`CommandRunner`] that runs each command through a caller-supplied **argv template**, for targets
/// with no HTTP surface: `ssh`, `docker exec`, `kubectl exec`, `podman`, a vendor's own CLI.
///
/// The template is a list of arguments; the literal `{}` element is replaced by the command and its
/// arguments. For example `["ssh", "build-host", "--", "{}"]` or
/// `["docker", "exec", "my-container", "{}"]`.
///
/// **The command is never rendered into a string.** `{}` expands to *multiple argv entries*, one per
/// token, so a path containing spaces, quotes or `;` stays a single argument on the far side and
/// cannot be reparsed as syntax. A template that pasted the command into a shell string would make
/// every model-supplied path an injection vector.
pub struct TemplateRunner {
    template: Vec<String>,
}

impl TemplateRunner {
    /// Build from a template that must contain exactly one `{}` placeholder.
    pub fn new(template: Vec<String>) -> Result<Self, String> {
        let holes = template.iter().filter(|t| *t == "{}").count();
        if holes != 1 {
            return Err(format!(
                "exec template needs exactly one `{{}}` placeholder for the command, found {holes} \
                 in {template:?}"
            ));
        }
        if template.first().is_none_or(|p| p == "{}") {
            return Err("exec template must start with a program to run".to_string());
        }
        Ok(Self { template })
    }

    /// Parse a whitespace-separated template, the shape a CLI flag carries.
    pub fn parse(raw: &str) -> Result<Self, String> {
        Self::new(raw.split_whitespace().map(str::to_string).collect())
    }

    fn expand(&self, program: &str, args: &[String]) -> Vec<String> {
        expand(&self.template, program, args)
    }
}

#[async_trait]
impl CommandRunner for TemplateRunner {
    async fn run(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
    ) -> std::io::Result<ExecResult> {
        self.exec(program, args, cwd, Some(timeout), None, None, None, None)
            .await
    }

    /// The transport is a local process, so its output streams for free: every chunk the far side
    /// writes arrives on the transport's pipes as it happens.
    async fn run_streaming(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        on_chunk: ChunkSink<'_>,
    ) -> std::io::Result<ExecResult> {
        self.exec(
            program,
            args,
            cwd,
            Some(timeout),
            Some(on_chunk),
            None,
            None,
            None,
        )
        .await
    }

    /// As [`Self::run_streaming`], reporting a [`Kill`] that stops the command **on the far side**,
    /// not just the local transport — see `far_kill`. No deadline: that kill is how it ends early.
    /// What the command leaves running in its process group is reported to `on_orphans` and waited
    /// for, as a local shell's leftovers are (see `far_kill::WRAPPER`).
    async fn run_streaming_tracked(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        hooks: &JobHooks<'_>,
    ) -> std::io::Result<ExecResult> {
        self.exec(
            program,
            args,
            cwd,
            None,
            Some(hooks.on_chunk),
            None,
            Some(hooks.on_spawn),
            Some(hooks.on_orphans),
        )
        .await
    }

    async fn tracks_jobs(&self) -> bool {
        true
    }

    /// Piped into the *transport* process. Whether it reaches the far side is the transport's
    /// business: `ssh` forwards stdin, but `docker exec`/`kubectl exec` do so only with `-i`. Without
    /// it the far command reads nothing, which `ShellFs`'s probe detects — writes then use argv-sized
    /// chunks instead.
    async fn run_with_stdin(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        stdin: &[u8],
    ) -> std::io::Result<ExecResult> {
        self.exec(
            program,
            args,
            cwd,
            Some(timeout),
            None,
            Some(stdin),
            None,
            None,
        )
        .await
    }
}

/// How long the far-side kill may take before it is abandoned: one more round trip through the
/// transport, which for `docker exec` is a fraction of a second.
const FAR_KILL_LIMIT: Duration = Duration::from_secs(10);

/// Stops a template command wherever it is: on the far side through the transport (once its
/// marker has arrived), then the local transport's own process group. Blocking.
#[derive(Clone)]
struct TemplateKill {
    template: Arc<Vec<String>>,
    far: far_kill::FarIdCell,
    local_group: Option<crate::tools::exec::GroupId>,
}

impl TemplateKill {
    fn kill(&self) {
        let far = *self.far.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(id) = far {
            let (program, args) = far_kill::far_kill_argv(id);
            far_kill::run_bounded(&expand(&self.template, &program, &args), FAR_KILL_LIMIT);
        }
        #[cfg(unix)]
        if let Some(group) = self.local_group {
            crate::tools::exec::kill_process_group(group);
        }
    }
}

/// Runs a [`TemplateKill`] if the command's future is dropped before the command exits — a
/// cancelled call, an aborted job. Disarmed once the transport has exited on its own.
struct TemplateKillGuard(Option<TemplateKill>);

impl Drop for TemplateKillGuard {
    fn drop(&mut self) {
        #[cfg(unix)]
        if let Some(kill) = self.0.take() {
            crate::tools::exec::spawn_tracked_cleanup(move || kill.kill());
        }
    }
}

/// `{}` replaced by the command and its arguments, as argv entries.
fn expand(template: &[String], program: &str, args: &[String]) -> Vec<String> {
    let mut out = Vec::with_capacity(template.len() + args.len());
    for part in template {
        if part == "{}" {
            out.push(program.to_string());
            out.extend(args.iter().cloned());
        } else {
            out.push(part.clone());
        }
    }
    out
}

impl TemplateRunner {
    #[allow(clippy::too_many_arguments)]
    async fn exec(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Option<Duration>,
        on_chunk: Option<ChunkSink<'_>>,
        stdin: Option<&[u8]>,
        on_spawn: Option<SpawnSink<'_>>,
        on_orphans: Option<OrphanSink<'_>>,
    ) -> std::io::Result<ExecResult> {
        // `cwd` is expressed by wrapping the command, not by setting the *local* process's directory —
        // the directory belongs to the far side. Positional parameters to a fixed script, never
        // substituted text.
        //
        // Everything but a stdin-carrying call runs under `far_kill::WRAPPER`, which makes the
        // command killable on the far side (killing the transport does not reach it — see
        // `far_kill`). A stdin call cannot: the wrapper backgrounds the command, and a background
        // job's stdin is `/dev/null`.
        let killable = stdin.is_none();
        let (program, args): (String, Vec<String>) = if killable {
            // A tracked job lingers on the far side while its leftovers run; any other call ends
            // with its command, as before.
            let linger = if on_orphans.is_some() { "1" } else { "0" };
            let mut v = vec![
                "-c".to_string(),
                far_kill::WRAPPER.to_string(),
                "sh".to_string(),
                linger.to_string(),
                cwd.unwrap_or("").to_string(),
                program.to_string(),
            ];
            v.extend(args.iter().cloned());
            ("sh".to_string(), v)
        } else {
            match cwd {
                Some(dir) => {
                    let mut v = vec![
                        "-c".to_string(),
                        r#"cd "$1" || exit 1; shift; exec "$@""#.to_string(),
                        "sh".to_string(),
                        dir.to_string(),
                        program.to_string(),
                    ];
                    v.extend(args.iter().cloned());
                    ("sh".to_string(), v)
                }
                None => (program.to_string(), args.to_vec()),
            }
        };
        let argv = self.expand(&program, &args);

        let mut cmd = tokio::process::Command::new(&argv[0]);
        cmd.args(&argv[1..])
            .stdin(if stdin.is_some() {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        // Its own group, so the local half of a kill reaches whatever helpers the transport forks.
        #[cfg(unix)]
        cmd.process_group(0);
        let mut child = cmd.spawn()?;
        let far = far_kill::FarIdCell::default();
        let kill = TemplateKill {
            template: Arc::new(self.template.clone()),
            far: far.clone(),
            local_group: child.id().map(crate::tools::exec::GroupId::leader),
        };
        let mut guard = TemplateKillGuard(Some(kill.clone()));
        if let Some(report) = on_spawn {
            let kill = kill.clone();
            report(Kill::Hook(Arc::new(move || kill.kill())));
        }
        let stdout = child.stdout.take();
        // The wrapper's shell-exit marker, heard on stderr: the command exited leaving processes of
        // its group running.
        let (shell_exit_tx, mut shell_exit_rx) = tokio::sync::watch::channel(None::<i32>);
        let stderr = child.stderr.take().map(|e| {
            let on_shell_exit = on_orphans.map(|_| -> far_kill::ShellExitSink<'_> {
                Box::new(move |code| {
                    let _ = shell_exit_tx.send(Some(code));
                })
            });
            far_kill::MarkerStrip::new(e, far.clone(), on_shell_exit)
        });
        // Fed concurrently with the drains, never before them — see `feed_stdin`.
        let feed = feed_stdin(child.stdin.take(), stdin);
        // `drain_capped` is the same streaming head+tail accumulator the local runner uses: memory
        // is bounded at `STREAM_HEAD + STREAM_TAIL` per stream regardless of how much arrives, and
        // it says honestly whether a middle was dropped (`wait_with_output`, used here once, buffered
        // everything and reported `truncated: false` regardless). When streaming, every chunk goes to
        // the sink and nothing is captured, exactly as `RealRunner` does. Not
        // `HttpExecRunner::read_capped` — that *discards* an over-cap body and errors, which is right
        // for a response envelope that must parse as a whole, and wrong here: a long-running
        // command's output is still the answer.
        //
        // Both pipes are drained *concurrently* with the wait. A child that fills one pipe's OS
        // buffer while we read only the other deadlocks, and an unread pipe stalls its exit.
        let (exited_tx, exited_rx) = tokio::sync::watch::channel(false);
        let wait = async {
            let status = child.wait().await;
            let _ = exited_tx.send(true);
            status
        };
        // Report the leftovers once the marker arrives, after the same 100 ms the local runner gives
        // output already in flight on the other pipe (`bash_proc::SETTLE`), so what the shell
        // printed lands in the call's result, not the job's.
        let orphans = async {
            let Some(sink) = on_orphans else {
                return;
            };
            let mut exited = exited_rx.clone();
            tokio::select! {
                // Each guard is dropped inside its own arm: neither may live across the sleep.
                code = async { shell_exit_rx.wait_for(Option::is_some).await.ok().and_then(|c| *c) } => {
                    tokio::time::sleep(SHELL_EXIT_SETTLE).await;
                    let shell = ExecResult {
                        code,
                        ..Default::default()
                    };
                    sink(&shell, REMOTE_LEFTOVERS.to_string());
                }
                () = async { let _ = exited.wait_for(|e| *e).await; } => {}
            }
        };
        let collect = async {
            tokio::join!(
                wait,
                drain_capped(
                    stdout,
                    STREAM_HEAD,
                    STREAM_TAIL,
                    on_chunk,
                    exited_rx.clone()
                ),
                drain_capped(
                    stderr,
                    STREAM_HEAD,
                    STREAM_TAIL,
                    on_chunk,
                    exited_rx.clone()
                ),
                feed,
                orphans,
            )
        };
        // `None`: a tracked job, whose kill is how it ends early.
        let collected = match timeout {
            Some(limit) => tokio::time::timeout(limit, collect).await.ok(),
            None => Some(collect.await),
        };
        match collected {
            Some((status, (stdout, out_truncated), (stderr, err_truncated), fed, ())) => {
                guard.0 = None;
                let status = status?;
                fed?;
                Ok(ExecResult {
                    code: status.code(),
                    // The transport's own exit signal, not the far command's — `ssh` reports a
                    // remote signal as an exit code, and a local `docker exec` killed by one says
                    // nothing about what ran inside. Left `None` rather than guessing.
                    signal: None,
                    stdout: String::from_utf8_lossy(&stdout).into_owned(),
                    stderr: String::from_utf8_lossy(&stderr).into_owned(),
                    timed_out: false,
                    truncated: out_truncated || err_truncated,
                })
            }
            None => {
                // Reaping the transport is not enough: the far command outlives it. Kill both and
                // wait for it, so a caller that re-checks side effects sees the command gone.
                if let Some(kill) = guard.0.take() {
                    let _ = tokio::task::spawn_blocking(move || kill.kill()).await;
                }
                Ok(ExecResult {
                    code: None,
                    signal: None,
                    stdout: String::new(),
                    stderr: format!(
                        "exec template timed out after {:?}",
                        timeout.unwrap_or_default()
                    ),
                    timed_out: true,
                    truncated: false,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn a_url_must_be_http_or_https() {
        assert!(HttpExecRunner::new("https://exec.example/run").is_ok());
        assert!(HttpExecRunner::new("http://127.0.0.1:9000/exec").is_ok());
        assert!(HttpExecRunner::new("file:///etc/passwd").is_err());
        assert!(HttpExecRunner::new("not a url").is_err());
    }

    #[test]
    fn headers_parse_as_name_colon_value() {
        let (n, v) = HttpExecRunner::parse_header("Authorization: Bearer abc").unwrap();
        assert_eq!(n, "Authorization");
        assert_eq!(v, "Bearer abc");
        assert!(HttpExecRunner::parse_header("no-colon").is_err());
        assert!(HttpExecRunner::parse_header(": empty-name").is_err());
    }

    #[test]
    fn a_template_needs_exactly_one_placeholder() {
        assert!(TemplateRunner::parse("ssh host -- {}").is_ok());
        assert!(TemplateRunner::parse("ssh host").is_err());
        assert!(TemplateRunner::parse("ssh {} {}").is_err());
        assert!(TemplateRunner::parse("{} foo").is_err());
    }

    #[test]
    fn the_placeholder_expands_to_argv_entries_not_a_string() {
        // The injection invariant. A path containing `;` and quotes must stay ONE argument on the far
        // side; if this ever renders to a string, the far side reparses it as syntax.
        let r = TemplateRunner::parse("docker exec ctr {}").unwrap();
        let argv = r.expand(
            "grep",
            &["-e".into(), "'; rm -rf / #".into(), "/a b.txt".into()],
        );
        assert_eq!(
            argv,
            vec![
                "docker",
                "exec",
                "ctr",
                "grep",
                "-e",
                "'; rm -rf / #",
                "/a b.txt"
            ]
        );
    }

    #[tokio::test]
    async fn a_template_runner_actually_runs_the_command() {
        // `env` is a harmless stand-in for a real transport: it runs whatever follows, which is
        // exactly the shape `ssh host --` / `docker exec ctr` have.
        let r = TemplateRunner::parse("env {}").unwrap();
        let out = r
            .run(
                "printf",
                &["%s".into(), "hello".into()],
                None,
                Duration::from_secs(10),
            )
            .await
            .unwrap();
        assert_eq!(out.code, Some(0));
        assert_eq!(out.stdout, "hello");
    }

    #[tokio::test]
    async fn a_template_runner_honors_cwd_without_splicing_it() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("marker.txt"), "x").unwrap();
        let r = TemplateRunner::parse("env {}").unwrap();
        let out = r
            .run("ls", &[], dir.path().to_str(), Duration::from_secs(10))
            .await
            .unwrap();
        assert_eq!(out.code, Some(0));
        assert!(out.stdout.contains("marker.txt"), "{out:?}");
    }

    /// The transport's output is bounded by the capture window and truncation is reported honestly.
    /// `wait_with_output` buffered the whole thing and hardcoded `truncated: false`, so a command
    /// that printed a gigabyte sized this process — and said nothing about what it had lost.
    #[tokio::test]
    #[cfg(unix)]
    async fn a_template_runner_caps_its_output_and_says_when_it_truncated() {
        let r = TemplateRunner::parse("env {}").unwrap();
        let bytes = 2 * (STREAM_HEAD + STREAM_TAIL);
        let out = r
            .run(
                "sh",
                &[
                    "-c".into(),
                    format!("head -c {bytes} /dev/zero | tr '\\0' 'a'"),
                ],
                None,
                Duration::from_secs(30),
            )
            .await
            .unwrap();

        assert_eq!(out.code, Some(0), "{out:?}");
        assert!(out.truncated, "the middle was dropped, so say so: {out:?}");
        assert!(
            out.stdout.len() <= STREAM_HEAD + STREAM_TAIL,
            "kept {} bytes of a {bytes}-byte stream",
            out.stdout.len()
        );
    }

    /// …and output that fits is returned whole, still flagged untruncated.
    #[tokio::test]
    #[cfg(unix)]
    async fn a_template_runner_leaves_output_under_the_cap_alone() {
        let r = TemplateRunner::parse("env {}").unwrap();
        let out = r
            .run(
                "sh",
                &[
                    "-c".into(),
                    "printf 'hello\\n'; printf 'oops\\n' >&2".into(),
                ],
                None,
                Duration::from_secs(10),
            )
            .await
            .unwrap();
        assert_eq!(out.stdout, "hello\n");
        assert_eq!(out.stderr, "oops\n");
        assert!(!out.truncated, "{out:?}");
    }
}

// ---------------------------------------------------------------------------------------------
// Per-session targets, for a multi-tenant server.
// ---------------------------------------------------------------------------------------------

/// One configured target: the runner, and the filesystem backend built over it.
///
/// They are constructed together and handed out together because they must always name the *same*
/// machine — a `bash` on one and an `edit` on another is not a partial configuration, it is a bug.
#[derive(Clone)]
pub struct ExecTarget {
    runner: std::sync::Arc<dyn CommandRunner>,
    backend: std::sync::Arc<dyn crate::tools::fs::FsBackend>,
    caps: crate::tools::fs::shell::Capabilities,
}

impl ExecTarget {
    /// Build from an HTTP endpoint plus `Name: value` headers, reading at most `max_response_bytes` of
    /// any one response ([`DEFAULT_MAX_RESPONSE_BYTES`] unless configured).
    pub async fn http(
        url: &str,
        headers: &[String],
        max_response_bytes: usize,
    ) -> Result<Self, String> {
        let mut runner = HttpExecRunner::new(url)?.with_max_response_bytes(max_response_bytes);
        for raw in headers {
            let (name, value) = HttpExecRunner::parse_header(raw)?;
            runner = runner.with_header(name, value);
        }
        Ok(Self::over(std::sync::Arc::new(runner)).await)
    }

    /// Build from an argv template (`ssh host -- {}`, `docker exec ctr {}`, …).
    pub async fn template(template: &str) -> Result<Self, String> {
        Ok(Self::over(std::sync::Arc::new(TemplateRunner::parse(template)?)).await)
    }

    /// Wrap an already-built runner, probing it for the capability rung.
    pub async fn over(runner: std::sync::Arc<dyn CommandRunner>) -> Self {
        Self::over_with_home(runner, None).await
    }

    /// [`Self::over`], told the target's `$HOME` up front so a model-supplied `~/notes.md` expands
    /// against the sandbox's own home instead of being left alone. Service mode learns the home from
    /// its startup probe (`crate::service::ServiceSession::connect_exec`), which is the only caller
    /// that can know it before the first tool call.
    pub async fn over_with_home(
        runner: std::sync::Arc<dyn CommandRunner>,
        home: Option<String>,
    ) -> Self {
        let backend = crate::tools::fs::shell::ShellFs::connect(runner.clone()).await;
        let backend = match home {
            Some(home) => backend.with_home(home),
            None => backend,
        };
        let caps = backend.capabilities();
        Self {
            runner,
            backend: std::sync::Arc::new(backend),
            caps,
        }
    }

    pub fn runner(&self) -> std::sync::Arc<dyn CommandRunner> {
        self.runner.clone()
    }

    pub fn backend(&self) -> std::sync::Arc<dyn crate::tools::fs::FsBackend> {
        self.backend.clone()
    }

    /// Which search rung this target resolved to — reported to the client so a degraded box (no
    /// `rg`, or busybox) is visible at attach rather than inferred from bad results later.
    pub fn search_engine(&self) -> crate::tools::fs::shell::SearchEngine {
        self.caps.search_engine()
    }
}

/// A **per-session** exec target that can be re-pointed at runtime.
///
/// This exists because of a specific hazard. `serve`'s tool registry is rebuilt only when the model
/// or thinking level changes — a `switch_session` between two sessions on the same model does *not*
/// rebuild it. So a target bound at registry-construction time would leave the new session talking to
/// the previous session's machine. In a multi-tenant server that is not a stale-config bug, it is one
/// tenant's tools operating inside another tenant's sandbox.
///
/// The cell closes that by construction: the tools hold *this*, and read it on every call, so a
/// re-point takes effect immediately and cannot be missed by a skipped rebuild. It is the same shape
/// `memory::file::SessionDir` already uses for the per-session `/session` mount, for the same reason.
///
/// An empty cell means the host, so one registry serves both local and remote sessions — unless the
/// cell is [strict](Self::strict), in which case an empty cell is an error and there is no host to
/// fall back to.
#[derive(Clone, Default)]
pub struct ExecCell {
    route: CellRoute,
    /// The session's `bash` jobs. Here because they share the cell's lifetime and reach: one per
    /// session, handed to every registry rebuild (`build_tools` takes the cell), so a `set_model`
    /// keeps them running.
    ///
    /// Deliberately *not* in [`CellRoute`], the part [`Self::runner`] and [`Self::backend`] clone:
    /// every running job's task holds the runner, so a runner that held the table would make a
    /// cycle (task → runner → table → job → task), and a session that ended without stopping its
    /// jobs would leak them, unreachable, for the life of the process.
    bash_jobs: std::sync::Arc<crate::tools::bash_jobs::BashJobs>,
}

/// Where an [`ExecCell`] dispatches: the part its runner and filesystem backend share.
#[derive(Clone, Default)]
struct CellRoute {
    target: std::sync::Arc<std::sync::RwLock<Option<ExecTarget>>>,
    /// Fail closed: an empty cell errors instead of running on this host. Service mode
    /// (`serve --service`) sets it, because there the host is the *replica* — a tenant's `bash`
    /// landing on it is not a stale-config bug but a breach. Fixed at construction.
    strict: bool,
}

impl CellRoute {
    fn get(&self) -> Option<ExecTarget> {
        self.target
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// The target to dispatch to, or — in strict mode with an empty cell — the refusal.
    fn resolve(&self) -> Result<Option<ExecTarget>, &'static str> {
        match (self.get(), self.strict) {
            (Some(t), _) => Ok(Some(t)),
            (None, false) => Ok(None),
            (None, true) => Err(NO_TARGET),
        }
    }
}

/// What a strict cell answers when nothing is configured. Static text: it reaches the model as a
/// tool error, so it says what to do, not what went wrong internally.
const NO_TARGET: &str = "no sandbox is attached to this session";

impl ExecCell {
    pub fn new() -> Self {
        Self::default()
    }

    /// A cell with **no host fallback**: until a target is set, every command and every filesystem
    /// call fails, rather than running on this host. Service mode (`serve --service`) uses this,
    /// because there "this host" is the *replica* — a tenant's `bash` landing on it is not a
    /// stale-config bug but a breach.
    pub fn strict() -> Self {
        Self {
            route: CellRoute {
                strict: true,
                ..CellRoute::default()
            },
            ..Self::default()
        }
    }

    /// The session's `bash` job table — see [`crate::tools::bash_jobs`].
    pub fn bash_jobs(&self) -> &std::sync::Arc<crate::tools::bash_jobs::BashJobs> {
        &self.bash_jobs
    }

    /// Re-point at `target`, or back to the host with `None`. Takes effect on the next tool call.
    pub fn set(&self, target: Option<ExecTarget>) {
        // Recover a poisoned lock rather than panicking: the cell holds a single value with no
        // invariant a panicked writer could leave half-updated, and the workspace forbids `unwrap`.
        *self.route.target.write().unwrap_or_else(|e| e.into_inner()) = target;
    }

    pub fn get(&self) -> Option<ExecTarget> {
        self.route.get()
    }

    /// A [`CommandRunner`] that always dispatches to whatever this cell currently holds.
    pub fn runner(&self) -> std::sync::Arc<dyn CommandRunner> {
        std::sync::Arc::new(CellRunner(self.route.clone()))
    }

    /// An [`FsBackend`](crate::tools::fs::FsBackend) that always dispatches to whatever this cell
    /// currently holds.
    pub fn backend(&self) -> std::sync::Arc<dyn crate::tools::fs::FsBackend> {
        std::sync::Arc::new(CellFs(self.route.clone()))
    }
}

/// Dispatches each command to the cell's current target, or this host when it is empty.
struct CellRunner(CellRoute);

#[async_trait]
impl CommandRunner for CellRunner {
    async fn run(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
    ) -> std::io::Result<ExecResult> {
        match self.0.resolve().map_err(std::io::Error::other)? {
            Some(t) => t.runner().run(program, args, cwd, timeout).await,
            None => {
                crate::tools::exec::RealRunner
                    .run(program, args, cwd, timeout)
                    .await
            }
        }
    }

    async fn run_streaming(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        on_chunk: ChunkSink<'_>,
    ) -> std::io::Result<ExecResult> {
        match self.0.resolve().map_err(std::io::Error::other)? {
            Some(t) => {
                t.runner()
                    .run_streaming(program, args, cwd, timeout, on_chunk)
                    .await
            }
            None => {
                crate::tools::exec::RealRunner
                    .run_streaming(program, args, cwd, timeout, on_chunk)
                    .await
            }
        }
    }

    /// Forwarded, not left to the default: without it a `bash` job behind a cell — every `serve`
    /// session's — reported no [`Kill`], so the exit path could not reach it, a remote job never
    /// yielded, and on this host a leftover `cmd &` went untracked.
    async fn run_streaming_tracked(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        hooks: &crate::tools::exec::JobHooks<'_>,
    ) -> std::io::Result<ExecResult> {
        match self.0.resolve().map_err(std::io::Error::other)? {
            Some(t) => {
                t.runner()
                    .run_streaming_tracked(program, args, cwd, hooks)
                    .await
            }
            None => {
                crate::tools::exec::RealRunner
                    .run_streaming_tracked(program, args, cwd, hooks)
                    .await
            }
        }
    }

    /// Forwarded for the same reason: the target behind the cell decides whether `bash` can run a
    /// command in the background. A strict cell with no target answers `false`; running anything
    /// then fails with the cell's own refusal.
    async fn tracks_jobs(&self) -> bool {
        match self.0.resolve() {
            Ok(Some(t)) => t.runner().tracks_jobs().await,
            Ok(None) => true,
            Err(_) => false,
        }
    }

    async fn run_with_stdin(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        stdin: &[u8],
    ) -> std::io::Result<ExecResult> {
        match self.0.resolve().map_err(std::io::Error::other)? {
            Some(t) => {
                t.runner()
                    .run_with_stdin(program, args, cwd, timeout, stdin)
                    .await
            }
            None => {
                crate::tools::exec::RealRunner
                    .run_with_stdin(program, args, cwd, timeout, stdin)
                    .await
            }
        }
    }
}

/// The filesystem half of the same indirection.
struct CellFs(CellRoute);

impl CellFs {
    fn inner(
        &self,
    ) -> Result<std::sync::Arc<dyn crate::tools::fs::FsBackend>, crate::tools::fs::FsError> {
        match self.0.resolve() {
            Ok(Some(t)) => Ok(t.backend()),
            Ok(None) => Ok(std::sync::Arc::new(crate::tools::fs::local::LocalFs::new())),
            Err(e) => Err(crate::tools::fs::FsError::Backend(e.to_owned())),
        }
    }
}

#[async_trait]
impl crate::tools::fs::FsBackend for CellFs {
    /// Not fallible, so a strict empty cell answers with the world it *will* have: remote, with no
    /// home known yet. Never `Local` — a tenant's path must not resolve against the replica even in
    /// the window before its sandbox is attached.
    fn world(&self) -> crate::tools::fs::PathWorld {
        match self.inner() {
            Ok(b) => b.world(),
            Err(_) => crate::tools::fs::PathWorld::Remote { home: None },
        }
    }
    async fn search(
        &self,
        q: &crate::tools::fs::SearchQuery,
    ) -> Result<crate::tools::fs::SearchOutcome, crate::tools::fs::FsError> {
        self.inner()?.search(q).await
    }
    async fn stat(
        &self,
        path: &std::path::Path,
    ) -> Result<Option<crate::tools::fs::Meta>, crate::tools::fs::FsError> {
        self.inner()?.stat(path).await
    }
    async fn read_bytes(
        &self,
        path: &std::path::Path,
        offset: u64,
        max: usize,
    ) -> Result<Vec<u8>, crate::tools::fs::FsError> {
        self.inner()?.read_bytes(path, offset, max).await
    }
    async fn write_bytes(
        &self,
        path: &std::path::Path,
        bytes: Vec<u8>,
    ) -> Result<(), crate::tools::fs::FsError> {
        self.inner()?.write_bytes(path, bytes).await
    }
    async fn write_if_unchanged(
        &self,
        path: &std::path::Path,
        bytes: Vec<u8>,
        expected: Option<std::time::SystemTime>,
    ) -> Result<bool, crate::tools::fs::FsError> {
        self.inner()?
            .write_if_unchanged(path, bytes, expected)
            .await
    }
    async fn create_dir_all(
        &self,
        path: &std::path::Path,
    ) -> Result<(), crate::tools::fs::FsError> {
        self.inner()?.create_dir_all(path).await
    }
    async fn list_dir(
        &self,
        path: &std::path::Path,
        cap: usize,
        include_hidden: bool,
    ) -> Result<Vec<crate::tools::fs::DirEntry>, crate::tools::fs::FsError> {
        self.inner()?.list_dir(path, cap, include_hidden).await
    }
    async fn glob(
        &self,
        q: &crate::tools::fs::GlobQuery,
    ) -> Result<crate::tools::fs::GlobOutcome, crate::tools::fs::FsError> {
        self.inner()?.glob(q).await
    }
}
