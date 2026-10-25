//! The conformance verifier for exec endpoints: run it against an endpoint and it says, check by
//! check, whether the endpoint implements the protocol in the parent module's doc.
//!
//! This crate owns the contract, so it owns the verifier: an endpoint lives on the other side of a
//! repo boundary (a sandbox vendor's adapter, the platform's own sandbox service), and this is how
//! that side proves it conforms without this side learning anything about it. Run it with
//!
//! ```sh
//! AGENT_CONFORMANCE_EXEC_URL=https://exec.example/run \
//! AGENT_CONFORMANCE_EXEC_HEADER='Authorization: Bearer …' \
//!   cargo nextest run -p beyond-ai-agent -E 'test(/^exec_conformance::/)' --no-capture
//! ```
//!
//! Every check runs real commands through the endpoint (`sh`, `printf`, `sleep`, `kill`, `head`,
//! `tr`, `test`, `touch`, `rm` — what a busybox box has) and writes nothing but two marker files
//! under `/tmp`, which it removes. A full run against a conforming endpoint takes about fifteen
//! seconds, most of it waiting out a deliberately silent command and a grandchild's timer.
//!
//! The checks that decide whether `bash` can treat the endpoint like a local shell — incremental
//! delivery, heartbeats, and above all **kill on disconnect** — are measured, not taken on trust:
//! the disconnect check watches a backgrounded grandchild that would write a marker file if its
//! process group outlived the connection.
//!
//! An endpoint that advertises job notify (v1.4) is held to [`notify`]'s checks too, against a
//! callback receiver the verifier runs ([`Expectations::callback_listen`]): the proof's timing and
//! exact shape, a callback at each state change and for a change during an attempt, one url and
//! one sender, `keep` not waiting, the grammar, the latest keep governing, nothing without notify or
//! after removal, no redirects, a verified server, the schedule and the status table, the token kept
//! out of argv, environment and answers, and a kept job surviving the sandbox going idle (alone, at
//! the end, for [`Expectations::notify_idle`]). They add about 35 s to a run, the idle window
//! included.

use std::time::{Duration, Instant};

use super::HttpExecRunner;
use super::jobs::{JobEvent, JobId, JobScope, JobSpec, JobStatus, SandboxJobs as _};
use super::jobs_native::NativeJobs;
use super::stream::{self, Frame, FrameReader, HEARTBEAT_INTERVAL};

/// The v1.4 (job notify) checks; their names are this module's constants.
pub mod notify;

/// The exec front / edge kit: the front's behavior in front of the endpoints (single-flight create,
/// the tenant cap, tokens, wake), with the operator hook it needs.
pub mod front;

/// What the verifier holds the endpoint to. [`Default`] is the spec; a test of the verifier itself
/// shrinks the intervals so it runs in seconds.
#[derive(Debug, Clone)]
pub struct Expectations {
    /// The longest silence the endpoint may leave between frames ([`HEARTBEAT_INTERVAL`]).
    pub heartbeat: Duration,
    /// How soon after a disconnect the command's process group must be gone: two heartbeat
    /// intervals, the bound an endpoint that only notices a disconnect when a heartbeat write fails
    /// still meets.
    pub kill_within: Duration,
    /// Where the v1.4 checks' callback receiver listens. The default, an ephemeral loopback port,
    /// serves an endpoint whose sandbox shares this host's loopback (the in-repo mock, a container
    /// on the host network): `keep` takes plain `http` only to loopback.
    pub callback_listen: String,
    /// The origin the sandbox reaches the receiver at, when it is not the listen address: an
    /// `https` origin that forwards to [`Self::callback_listen`]. The TLS check is then skipped.
    pub callback_origin: Option<String>,
    /// How long a kept job must survive with nothing sent to the sandbox (v1.4 "a kept job
    /// survives idling"). An auto-stop slower than this goes unseen.
    pub notify_idle: Duration,
}

impl Default for Expectations {
    fn default() -> Self {
        Self {
            heartbeat: HEARTBEAT_INTERVAL,
            kill_within: HEARTBEAT_INTERVAL * 2,
            // port-0: the callback receiver binds it and holds the listener for the whole run.
            callback_listen: "127.0.0.1:0".to_string(),
            callback_origin: None,
            notify_idle: Duration::from_secs(30),
        }
    }
}

/// One check's outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Pass,
    Fail,
    /// Not applicable: a v1.1 endpoint is not held to the streaming checks.
    Skip,
}

#[derive(Debug, Clone)]
pub struct Check {
    pub name: &'static str,
    pub verdict: Verdict,
    pub detail: String,
}

/// The whole run.
#[derive(Debug, Clone)]
pub struct Report {
    /// The endpoint answered `"stream": true` with a v1.2 stream.
    pub streams: bool,
    /// The endpoint offers sandbox jobs natively (v1.3).
    pub jobs: bool,
    /// The endpoint offers job notify (v1.4), natively or for the baseline.
    pub notify: bool,
    pub checks: Vec<Check>,
}

impl Report {
    /// No check failed. A v1.1 endpoint passes with its streaming checks skipped — see
    /// [`Self::streams`] for whether it is v1.2.
    pub fn passed(&self) -> bool {
        self.checks.iter().all(|c| c.verdict != Verdict::Fail)
    }

    pub fn check(&self, name: &str) -> Option<&Check> {
        self.checks.iter().find(|c| c.name == name)
    }

    /// One line per check, for a terminal.
    pub fn render(&self) -> String {
        let mut out = format!(
            "exec endpoint: {}\n",
            if self.jobs && self.notify {
                "v1.4 (streams, runs sandbox jobs itself, and calls back when a kept job changes)"
            } else if self.jobs {
                "v1.3 (streams, and runs sandbox jobs itself)"
            } else if self.streams {
                "v1.2 (streams; sandbox jobs run on plain exec commands)"
            } else {
                "v1.1 (no streaming: `bash` waits for every command to exit, cannot yield a silent \
                 one, and cannot kill one)"
            }
        );
        for c in &self.checks {
            let mark = match c.verdict {
                Verdict::Pass => "PASS",
                Verdict::Fail => "FAIL",
                Verdict::Skip => "skip",
            };
            out.push_str(&format!("  {mark}  {}: {}\n", c.name, c.detail));
        }
        out
    }
}

pub const ENVELOPE: &str = "v1.1 envelope";
pub const CONTENT_TYPE: &str = "stream content type";
pub const INCREMENTAL: &str = "incremental delivery (the probe)";
pub const BYTES: &str = "bytes and streams survive exactly";
pub const FRAME_SIZE: &str = "frames carry at most 64 KiB";
pub const EXIT: &str = "exit code, and the body ends after the final frame";
pub const SIGNAL: &str = "signal";
pub const TIMEOUT: &str = "timeout kills the process group";
pub const HEARTBEAT: &str = "heartbeat during silence";
pub const DISCONNECT: &str = "disconnect kills the process group";
pub const LEFTOVERS: &str = "leftover processes keep the stream open";
pub const JOBS: &str = "sandbox jobs offered (v1.3)";
pub const JOBS_HELD: &str = "a held job streams its output and ends";
pub const JOBS_HOLD: &str = "a dropped hold kills a job nobody kept";
pub const JOBS_KEEP: &str = "a kept job outlives its hold and reads back from an offset";
pub const JOBS_KILL: &str = "kill stops a job's group and removes it";
pub const JOBS_SCOPE: &str = "list and wipe stay inside their scope";

/// Every v1.3 check after the capability itself.
const JOB_CHECKS: [&str; 5] = [JOBS_HELD, JOBS_HOLD, JOBS_KEEP, JOBS_KILL, JOBS_SCOPE];

/// Run every check against the endpoint at `url`.
pub async fn verify(url: &str, headers: &[(String, String)], expect: &Expectations) -> Report {
    let mut runner = match HttpExecRunner::new(url) {
        Ok(r) => r,
        Err(e) => {
            return Report {
                streams: false,
                jobs: false,
                notify: false,
                checks: vec![fail(ENVELOPE, e)],
            };
        }
    };
    for (n, v) in headers {
        runner = runner.with_header(n.clone(), v.clone());
    }
    let v = Verifier {
        runner,
        expect: expect.clone(),
        nonce: format!(
            "{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default()
        ),
    };

    let mut checks = vec![v.envelope().await];
    let streams = match v.run("printf a", Duration::from_secs(10)).await {
        Ok(Exchange::Stream { .. }) => {
            checks.push(pass(CONTENT_TYPE, format!("`{}`", stream::NDJSON)));
            true
        }
        Ok(Exchange::Envelope) => {
            checks.push(skip(
                CONTENT_TYPE,
                "answered `\"stream\": true` with one JSON object: a v1.1 endpoint",
            ));
            false
        }
        Err(e) => {
            checks.push(fail(CONTENT_TYPE, e));
            false
        }
    };
    if !streams {
        for name in [
            INCREMENTAL,
            BYTES,
            FRAME_SIZE,
            EXIT,
            SIGNAL,
            TIMEOUT,
            HEARTBEAT,
            DISCONNECT,
            LEFTOVERS,
        ] {
            checks.push(skip(name, "not a v1.2 endpoint"));
        }
        for name in std::iter::once(JOBS)
            .chain(JOB_CHECKS)
            .chain(notify::CHECKS)
        {
            checks.push(skip(name, "not a v1.2 endpoint"));
        }
        return Report {
            streams,
            jobs: false,
            notify: false,
            checks,
        };
    }
    checks.push(v.incremental().await);
    checks.push(v.bytes().await);
    checks.push(v.frame_size().await);
    checks.push(v.exit().await);
    checks.push(v.signal().await);
    // The slow checks wait on timers, not on each other.
    let (t, h, d, l) = tokio::join!(v.timeout(), v.heartbeat(), v.disconnect(), v.leftovers());
    checks.extend([t, h, d, l]);
    let jobs = v.runner.offers_jobs().await;
    if jobs {
        checks.push(pass(JOBS, "answered the probe with `{\"jobs\": \"v1.3\"}`"));
        let (held, hold, keep, kill, scope) = tokio::join!(
            v.jobs_held(),
            v.jobs_hold(),
            v.jobs_keep(),
            v.jobs_kill(),
            v.jobs_scope()
        );
        checks.extend([held, hold, keep, kill, scope]);
    } else {
        checks.push(skip(
            JOBS,
            "not offered: the agent runs sandbox jobs with plain exec commands instead",
        ));
        for name in JOB_CHECKS {
            checks.push(skip(name, "sandbox jobs are not offered"));
        }
    }
    let notify = notify::checks(&v, &mut checks).await;
    Report {
        streams,
        jobs,
        notify,
        checks,
    }
}

fn pass(name: &'static str, detail: impl Into<String>) -> Check {
    Check {
        name,
        verdict: Verdict::Pass,
        detail: detail.into(),
    }
}

fn fail(name: &'static str, detail: impl Into<String>) -> Check {
    Check {
        name,
        verdict: Verdict::Fail,
        detail: detail.into(),
    }
}

fn skip(name: &'static str, detail: impl Into<String>) -> Check {
    Check {
        name,
        verdict: Verdict::Skip,
        detail: detail.into(),
    }
}

/// What one streamed request produced.
enum Exchange {
    Stream {
        /// Every frame, with when it arrived, measured from just before the request was sent.
        frames: Vec<(Duration, Frame)>,
        /// What followed the final frame: `None` if the body ended, as it must.
        after_final: Option<String>,
    },
    Envelope,
}

impl Exchange {
    fn frames(&self) -> &[(Duration, Frame)] {
        match self {
            Exchange::Stream { frames, .. } => frames,
            Exchange::Envelope => &[],
        }
    }

    fn output(&self, stderr: bool) -> Vec<u8> {
        let mut out = Vec::new();
        for (_, f) in self.frames() {
            match (f, stderr) {
                (Frame::Stdout(b), false) | (Frame::Stderr(b), true) => out.extend_from_slice(b),
                _ => {}
            }
        }
        out
    }

    fn end(&self) -> Option<&Frame> {
        self.frames()
            .last()
            .map(|(_, f)| f)
            .filter(|f| f.is_final())
    }
}

struct Verifier {
    runner: HttpExecRunner,
    expect: Expectations,
    nonce: String,
}

impl Verifier {
    fn args(script: &str) -> Vec<String> {
        vec!["-c".to_string(), script.to_string()]
    }

    /// How long to wait for any frame before calling the stream dead: a conforming endpoint sends
    /// one at least every heartbeat, so a few missed heartbeats is a verdict, not a hang.
    fn silence(&self) -> Duration {
        self.expect.heartbeat * 3 + Duration::from_secs(2)
    }

    fn marker(&self, which: &str) -> String {
        format!("/tmp/exec-conformance-{}-{which}", self.nonce)
    }

    /// A v1.1 request, for setup and for observing the far side between streamed requests.
    async fn sh(&self, script: &str) -> Result<super::ExecResponse, String> {
        self.runner
            .post(
                "sh",
                &Self::args(script),
                None,
                Duration::from_secs(10),
                None,
            )
            .await
            .map_err(|e| e.to_string())
    }

    /// One streamed request, read to its final frame.
    async fn run(&self, script: &str, timeout: Duration) -> Result<Exchange, String> {
        let started = Instant::now();
        let resp = self
            .runner
            .send("sh", &Self::args(script), None, timeout, None, true)
            .await
            .map_err(|e| e.to_string())?;
        if !super::is_ndjson(&resp) {
            self.runner
                .envelope(resp)
                .await
                .map_err(|e| e.to_string())?;
            return Ok(Exchange::Envelope);
        }
        let mut reader = FrameReader::new(resp);
        let mut frames = Vec::new();
        loop {
            let Some(frame) = reader.next(self.silence()).await? else {
                return Err(format!(
                    "the body ended without a final frame after {} frames",
                    frames.len()
                ));
            };
            let last = frame.is_final();
            frames.push((started.elapsed(), frame));
            if last {
                break;
            }
        }
        let after_final = match reader.next(Duration::from_secs(1)).await {
            Ok(None) => None,
            Ok(Some(f)) => Some(format!("a frame followed the final frame: {f:?}")),
            Err(e) => Some(format!("the body did not end after the final frame ({e})")),
        };
        Ok(Exchange::Stream {
            frames,
            after_final,
        })
    }

    async fn envelope(&self) -> Check {
        match self.sh("printf out; printf err >&2; exit 3").await {
            Ok(r) if r.exit_code == 3 && r.stdout == "out" && r.stderr == "err" => pass(
                ENVELOPE,
                "a request without `stream` gets one JSON object, as in v1.1",
            ),
            Ok(r) => fail(
                ENVELOPE,
                format!(
                    "expected exit 3, stdout \"out\", stderr \"err\"; got exit {}, stdout {:?}, \
                     stderr {:?}",
                    r.exit_code, r.stdout, r.stderr
                ),
            ),
            Err(e) => fail(ENVELOPE, e),
        }
    }

    async fn incremental(&self) -> Check {
        let resp = match self
            .runner
            .send(
                "sh",
                &Self::args(stream::PROBE_SCRIPT),
                None,
                Duration::from_secs(10),
                None,
                true,
            )
            .await
        {
            Ok(r) => r,
            Err(e) => return fail(INCREMENTAL, e.to_string()),
        };
        match stream::observe_probe(resp).await {
            Ok(seen) if seen.streams() => pass(INCREMENTAL, seen.explain()),
            Ok(seen) => fail(INCREMENTAL, seen.explain()),
            Err(e) => fail(INCREMENTAL, e),
        }
    }

    async fn bytes(&self) -> Check {
        let script = r"printf '\377\000\001'; printf 'e\n' >&2; printf 'x'";
        match self.run(script, Duration::from_secs(10)).await {
            Ok(x) => {
                let (out, err) = (x.output(false), x.output(true));
                if out == [0xff, 0x00, 0x01, b'x']
                    && err == b"e\n"
                    && x.end() == Some(&Frame::Exit(0))
                {
                    pass(
                        BYTES,
                        "0xff 0x00 0x01 on stdout and a line on stderr, intact",
                    )
                } else {
                    fail(
                        BYTES,
                        format!(
                            "stdout {out:?} (want [255, 0, 1, 120]), stderr {:?}, end {:?}",
                            String::from_utf8_lossy(&err),
                            x.end()
                        ),
                    )
                }
            }
            Err(e) => fail(BYTES, e),
        }
    }

    async fn frame_size(&self) -> Check {
        const TOTAL: usize = 1024 * 1024;
        let script = format!("head -c {TOTAL} /dev/zero | tr '\\0' a");
        match self.run(&script, Duration::from_secs(30)).await {
            Ok(x) => {
                let biggest = x
                    .frames()
                    .iter()
                    .filter_map(|(_, f)| match f {
                        Frame::Stdout(b) | Frame::Stderr(b) => Some(b.len()),
                        _ => None,
                    })
                    .max()
                    .unwrap_or(0);
                let out = x.output(false);
                if biggest > stream::MAX_FRAME_PAYLOAD {
                    fail(
                        FRAME_SIZE,
                        format!(
                            "a frame carried {biggest} bytes (max {})",
                            stream::MAX_FRAME_PAYLOAD
                        ),
                    )
                } else if out.len() != TOTAL || out.iter().any(|&b| b != b'a') {
                    fail(
                        FRAME_SIZE,
                        format!("{} of {TOTAL} bytes arrived intact", out.len()),
                    )
                } else {
                    pass(
                        FRAME_SIZE,
                        format!("1 MiB in frames of at most {biggest} bytes"),
                    )
                }
            }
            Err(e) => fail(FRAME_SIZE, e),
        }
    }

    async fn exit(&self) -> Check {
        match self.run("exit 7", Duration::from_secs(10)).await {
            Ok(Exchange::Stream {
                after_final: Some(e),
                ..
            }) => fail(EXIT, e),
            Ok(x) if x.end() == Some(&Frame::Exit(7)) => {
                pass(EXIT, "`{\"exit_code\":7}`, then the body ended")
            }
            Ok(x) => fail(EXIT, format!("want exit 7, got {:?}", x.end())),
            Err(e) => fail(EXIT, e),
        }
    }

    async fn signal(&self) -> Check {
        match self.run("kill -KILL $$", Duration::from_secs(10)).await {
            Ok(x) if x.end() == Some(&Frame::Signal(9)) => pass(SIGNAL, "`{\"signal\":9}`"),
            Ok(x) => fail(
                SIGNAL,
                format!(
                    "want `{{\"signal\":9}}` for a command killed by SIGKILL, got {:?}",
                    x.end()
                ),
            ),
            Err(e) => fail(SIGNAL, e),
        }
    }

    /// `timeout_ms` is the endpoint's to enforce, on the whole group: a grandchild that would
    /// write a marker after the deadline must never write it.
    async fn timeout(&self) -> Check {
        let marker = self.marker("timeout");
        let grandchild = Duration::from_secs(2);
        let script = format!(
            "(sleep {}; touch {marker}) & sleep 60",
            grandchild.as_secs()
        );
        let limit = Duration::from_secs(1);
        let x = self.run(&script, limit).await;
        let check = match &x {
            Ok(x) if x.end() == Some(&Frame::TimedOut) => {
                let at = x.frames().last().map(|(t, _)| *t).unwrap_or_default();
                if at > limit + self.expect.kill_within {
                    fail(
                        TIMEOUT,
                        format!("`timed_out` came {} ms after a 1 s timeout", at.as_millis()),
                    )
                } else {
                    tokio::time::sleep(grandchild.saturating_sub(at) + Duration::from_secs(1))
                        .await;
                    match self.sh(&format!("test -e {marker}")).await {
                        Ok(r) if r.exit_code != 0 => pass(
                            TIMEOUT,
                            format!(
                                "`{{\"timed_out\":true}}` after {} ms; the backgrounded grandchild \
                                 died with its group",
                                at.as_millis()
                            ),
                        ),
                        Ok(_) => fail(
                            TIMEOUT,
                            "a backgrounded grandchild survived the timeout: the endpoint killed the \
                             command, not its process group",
                        ),
                        Err(e) => fail(TIMEOUT, e),
                    }
                }
            }
            Ok(x) => fail(TIMEOUT, format!("want `timed_out`, got {:?}", x.end())),
            Err(e) => fail(TIMEOUT, e.clone()),
        };
        let _ = self.sh(&format!("rm -f {marker}")).await;
        check
    }

    async fn heartbeat(&self) -> Check {
        let silence = (self.expect.heartbeat + Duration::from_secs(2))
            .as_secs_f64()
            .ceil() as u64;
        let slack = Duration::from_secs(1);
        match self
            .run(
                &format!("sleep {silence}"),
                Duration::from_secs(silence + 30),
            )
            .await
        {
            Ok(x) => {
                let mut prev = Duration::ZERO;
                let mut widest = Duration::ZERO;
                for (t, _) in x.frames() {
                    widest = widest.max(t.saturating_sub(prev));
                    prev = *t;
                }
                let beats = x
                    .frames()
                    .iter()
                    .filter(|(_, f)| *f == Frame::Heartbeat)
                    .count();
                let output = x
                    .frames()
                    .iter()
                    .any(|(_, f)| matches!(f, Frame::Stdout(_) | Frame::Stderr(_)));
                if output {
                    fail(HEARTBEAT, "a silent command produced output frames")
                } else if beats == 0 || widest > self.expect.heartbeat + slack {
                    fail(
                        HEARTBEAT,
                        format!(
                            "{beats} heartbeats over {silence} s of silence; the widest gap between \
                             frames was {} ms (max {} ms)",
                            widest.as_millis(),
                            self.expect.heartbeat.as_millis()
                        ),
                    )
                } else {
                    pass(
                        HEARTBEAT,
                        format!(
                            "{beats} heartbeats over {silence} s of silence, widest gap {} ms",
                            widest.as_millis()
                        ),
                    )
                }
            }
            Err(e) => fail(HEARTBEAT, e),
        }
    }

    /// The normative requirement: the connection is the command's lifetime. Start a command with a
    /// backgrounded grandchild, read both pids, hang up, and watch both die — then confirm, through
    /// a marker the grandchild would write if it lived, that the whole group did.
    async fn disconnect(&self) -> Check {
        let marker = self.marker("disconnect");
        let grandchild = self.expect.kill_within + Duration::from_secs(2);
        let script = format!(
            "echo $$; (sleep {}; touch {marker}) & echo $!; sleep 60",
            grandchild.as_secs()
        );
        let check = self.disconnect_inner(&script, &marker, grandchild).await;
        let _ = self.sh(&format!("rm -f {marker}")).await;
        check
    }

    async fn disconnect_inner(&self, script: &str, marker: &str, grandchild: Duration) -> Check {
        let started = Instant::now();
        let resp = match self
            .runner
            .send(
                "sh",
                &Self::args(script),
                None,
                Duration::from_secs(120),
                None,
                true,
            )
            .await
        {
            Ok(r) => r,
            Err(e) => return fail(DISCONNECT, e.to_string()),
        };
        let mut reader = FrameReader::new(resp);
        let mut out = Vec::new();
        let pids: Vec<String> = loop {
            let text = String::from_utf8_lossy(&out).into_owned();
            let lines: Vec<String> = text.lines().map(str::to_string).collect();
            if lines.len() >= 2 && text.ends_with('\n') {
                break lines;
            }
            match reader.next(self.silence()).await {
                Ok(Some(Frame::Stdout(b))) => out.extend_from_slice(&b),
                Ok(Some(f)) if f.is_final() => {
                    return fail(DISCONNECT, format!("the command ended early: {f:?}"));
                }
                Ok(Some(_)) => {}
                Ok(None) => return fail(DISCONNECT, "the body ended early"),
                Err(e) => return fail(DISCONNECT, e),
            }
        };
        // Hang up.
        drop(reader);
        let hung_up = Instant::now();
        // Not a bare `kill -0`: that succeeds on a zombie, and a sandbox whose pid 1 is not an init
        // (a container running `sleep infinity`) never reaps the orphaned grandchild it inherits —
        // a killed group would read as alive forever. `/proc` says `Z`.
        let alive = format!(
            "alive() {{ if [ -d /proc/self ]; then s=$(cut -d' ' -f3 \"/proc/$1/stat\" 2>/dev/null); \
             [ -n \"$s\" ] && [ \"$s\" != Z ]; else kill -0 \"$1\" 2>/dev/null; fi; }}; \
             alive {} || alive {}",
            pids[0], pids[1]
        );
        let gone_after = loop {
            match self.sh(&alive).await {
                Ok(r) if r.exit_code != 0 => break Some(hung_up.elapsed()),
                Ok(_) if hung_up.elapsed() > self.expect.kill_within => break None,
                Ok(_) => tokio::time::sleep(Duration::from_millis(100)).await,
                Err(e) => return fail(DISCONNECT, e),
            }
        };
        let Some(gone_after) = gone_after else {
            return fail(
                DISCONNECT,
                format!(
                    "the command was still running {} ms after the client disconnected (max {} \
                     ms): the endpoint must kill the process group when the connection closes",
                    hung_up.elapsed().as_millis(),
                    self.expect.kill_within.as_millis()
                ),
            );
        };
        // The pids could be invisible to a later request (a separate pid namespace), so the
        // grandchild's own marker is the proof that does not depend on seeing them.
        tokio::time::sleep((grandchild + Duration::from_secs(1)).saturating_sub(started.elapsed()))
            .await;
        match self.sh(&format!("test -e {marker}")).await {
            Ok(r) if r.exit_code != 0 => pass(
                DISCONNECT,
                format!(
                    "the process group was gone {} ms after the disconnect",
                    gone_after.as_millis()
                ),
            ),
            Ok(_) => fail(
                DISCONNECT,
                "a backgrounded grandchild survived the disconnect: the endpoint killed the \
                 command, not its process group",
            ),
            Err(e) => fail(DISCONNECT, e),
        }
    }

    /// "Leftover processes": a shell that exits while a process it started runs on gets
    /// `{"shell_exit": N}`, the stream stays open until that process exits, and then ends with the
    /// shell's final frame — and a disconnect in between kills the leftover, so it is a job the client
    /// can stop like any other.
    async fn leftovers(&self) -> Check {
        let ex = match self
            .run("(sleep 1) & exit 3", Duration::from_secs(30))
            .await
        {
            Ok(ex) => ex,
            Err(e) => return fail(LEFTOVERS, e),
        };
        let frames = ex.frames();
        let end = frames.last().filter(|(_, f)| f.is_final());
        let Some(&(shell_at, ref shell)) = frames
            .iter()
            .find(|(_, f)| matches!(f, Frame::ShellExit(_)))
        else {
            return fail(
                LEFTOVERS,
                format!(
                    "the stream ended with the shell ({:?}) while a process it started ran on: the \
                     endpoint must send `{{\"shell_exit\": N}}` and keep the stream open until the \
                     process group is empty",
                    end.map(|(_, f)| f)
                ),
            );
        };
        if shell != &Frame::ShellExit(3) {
            return fail(LEFTOVERS, format!("expected `shell_exit` 3, got {shell:?}"));
        }
        match end {
            Some((at, Frame::Exit(3)))
                if at.saturating_sub(shell_at) >= Duration::from_millis(500) => {}
            Some((at, f)) => {
                return fail(
                    LEFTOVERS,
                    format!(
                        "after `shell_exit` the stream must end with the shell's `{{\"exit_code\": 3}}` \
                         once the leftover (a 1 s sleep) has exited; got {f:?} {} ms after it",
                        at.saturating_sub(shell_at).as_millis()
                    ),
                );
            }
            None => return fail(LEFTOVERS, "no final frame"),
        }

        let marker = self.marker("leftover");
        let grandchild = self.expect.kill_within + Duration::from_secs(2);
        let check = self
            .leftover_disconnect(
                &format!("(sleep {}; touch {marker}) & exit 0", grandchild.as_secs()),
                &marker,
                grandchild,
            )
            .await;
        let _ = self.sh(&format!("rm -f {marker}")).await;
        check
    }

    async fn leftover_disconnect(&self, script: &str, marker: &str, grandchild: Duration) -> Check {
        let started = Instant::now();
        let resp = match self
            .runner
            .send(
                "sh",
                &Self::args(script),
                None,
                Duration::from_secs(120),
                None,
                true,
            )
            .await
        {
            Ok(r) => r,
            Err(e) => return fail(LEFTOVERS, e.to_string()),
        };
        let mut reader = FrameReader::new(resp);
        loop {
            match reader.next(self.silence()).await {
                Ok(Some(Frame::ShellExit(_))) => break,
                Ok(Some(f)) if f.is_final() => {
                    return fail(LEFTOVERS, format!("ended with the shell: {f:?}"));
                }
                Ok(Some(_)) => {}
                Ok(None) => return fail(LEFTOVERS, "the body ended early"),
                Err(e) => return fail(LEFTOVERS, e),
            }
        }
        // Hang up while only the leftover runs.
        drop(reader);
        tokio::time::sleep((grandchild + Duration::from_secs(1)).saturating_sub(started.elapsed()))
            .await;
        match self.sh(&format!("test -e {marker}")).await {
            Ok(r) if r.exit_code != 0 => pass(
                LEFTOVERS,
                "`shell_exit` sent, the stream held until the leftover exited, and a disconnect \
                 killed it",
            ),
            Ok(_) => fail(
                LEFTOVERS,
                "a leftover process survived a disconnect after `shell_exit`: the endpoint must kill \
                 the process group whenever the connection closes before the final frame",
            ),
            Err(e) => fail(LEFTOVERS, e),
        }
    }
}

/// The v1.3 checks: every operation through the agent's own client ([`NativeJobs`]), with the far
/// side's processes observed by v1.1 requests, never taken on trust.
impl Verifier {
    fn scope(&self, which: &str) -> JobScope {
        JobScope::under(
            super::jobs::JOBS_ROOT,
            format!("conformance-{}-{which}", self.nonce),
        )
    }

    fn spec(script: &str) -> JobSpec {
        JobSpec {
            program: "sh".into(),
            args: Self::args(script),
            cwd: None,
            label: script.into(),
        }
    }

    /// Whether group `pgid` has a live (non-zombie) member, asked of the far side.
    async fn group_alive(&self, pgid: u32) -> Result<bool, String> {
        let script = format!(
            "kill -0 -{pgid} 2>/dev/null || exit 1; [ -d /proc/self ] || exit 0; \
             for f in /proc/[0-9]*/stat; do read -r t <\"$f\" 2>/dev/null || continue; \
             t=${{t##*) }}; set -- $t; [ \"$3\" = {pgid} ] && [ \"$1\" != Z ] && exit 0; done; exit 1"
        );
        Ok(self.sh(&script).await?.exit_code == 0)
    }

    /// Wait (bounded) for group `pgid` to be gone; how long it took.
    async fn wait_gone(&self, pgid: u32, within: Duration) -> Result<Option<Duration>, String> {
        let started = Instant::now();
        loop {
            if !self.group_alive(pgid).await? {
                return Ok(Some(started.elapsed()));
            }
            if started.elapsed() > within {
                return Ok(None);
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    async fn jobs_held(&self) -> Check {
        let jobs = NativeJobs::over(&self.runner);
        let scope = self.scope("held");
        let id = JobId::fresh();
        let heard = std::sync::Mutex::new((None::<u32>, Vec::<u8>::new(), Option::<Instant>::None));
        let started = Instant::now();
        let events = |e: JobEvent<'_>| {
            let mut h = heard.lock().unwrap_or_else(|e| e.into_inner());
            match e {
                JobEvent::Started(p) => h.0 = Some(p),
                JobEvent::Output(b) => {
                    h.1.extend_from_slice(b);
                    h.2.get_or_insert_with(Instant::now);
                }
                _ => {}
            }
        };
        let r = jobs
            .run_held(&scope, &id, &Self::spec(stream::PROBE_SCRIPT), &events)
            .await;
        let _ = jobs.wipe(&scope).await;
        let (pgid, out, first) = heard.into_inner().unwrap_or_else(|e| e.into_inner());
        match r {
            Ok(0) if pgid.is_some() && out == b"ab" => {
                let lead =
                    first.map(|f| started.elapsed().saturating_sub(f.duration_since(started)));
                match lead {
                    Some(l) if l >= stream::PROBE_MIN_GAP => pass(
                        JOBS_HELD,
                        format!(
                            "`started`, then `ab` live ({} ms before the end), then exit 0",
                            l.as_millis()
                        ),
                    ),
                    _ => fail(
                        JOBS_HELD,
                        "the output arrived all at once with the end: not live",
                    ),
                }
            }
            Ok(code) => fail(
                JOBS_HELD,
                format!(
                    "want `started`, output `ab` and exit 0; got started {pgid:?}, output {:?}, exit {code}",
                    String::from_utf8_lossy(&out)
                ),
            ),
            Err(e) => fail(JOBS_HELD, e.to_string()),
        }
    }

    async fn jobs_hold(&self) -> Check {
        let jobs = NativeJobs::over(&self.runner);
        let scope = self.scope("hold");
        let id = JobId::fresh();
        let started = std::sync::Mutex::new(None::<u32>);
        let events = |e: JobEvent<'_>| {
            if let JobEvent::Started(p) = e {
                *started.lock().unwrap_or_else(|e| e.into_inner()) = Some(p);
            }
        };
        let spec = Self::spec("sleep 60 & sleep 60");
        let run = jobs.run_held(&scope, &id, &spec, &events);
        let seen = async {
            loop {
                if let Some(p) = *started.lock().unwrap_or_else(|e| e.into_inner()) {
                    return p;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        };
        let pgid = tokio::select! {
            r = run => return fail(JOBS_HOLD, format!("the job ended early: {r:?}")),
            p = tokio::time::timeout(Duration::from_secs(10), seen) => match p {
                Ok(p) => p,
                Err(_) => return fail(JOBS_HOLD, "no `started` frame within 10 s"),
            },
        };
        // The hold is dropped with `run`.
        let check = match self.wait_gone(pgid, self.expect.kill_within).await {
            Ok(Some(after)) => pass(
                JOBS_HOLD,
                format!(
                    "the group was gone {} ms after the hold closed",
                    after.as_millis()
                ),
            ),
            Ok(None) => fail(
                JOBS_HOLD,
                format!(
                    "the job still ran {} ms after its hold closed: a held job nobody kept must die \
                     with its connection",
                    self.expect.kill_within.as_millis()
                ),
            ),
            Err(e) => fail(JOBS_HOLD, e),
        };
        let _ = jobs.wipe(&scope).await;
        check
    }

    async fn jobs_keep(&self) -> Check {
        let check = self.jobs_keep_inner().await;
        let _ = NativeJobs::over(&self.runner)
            .wipe(&self.scope("keep"))
            .await;
        check
    }

    async fn jobs_keep_inner(&self) -> Check {
        let jobs = NativeJobs::over(&self.runner);
        let scope = self.scope("keep");
        let id = JobId::fresh();
        let heard = std::sync::Mutex::new((None::<u32>, Vec::<u8>::new()));
        let events = |e: JobEvent<'_>| {
            let mut h = heard.lock().unwrap_or_else(|e| e.into_inner());
            match e {
                JobEvent::Started(p) => h.0 = Some(p),
                JobEvent::Output(b) => h.1.extend_from_slice(b),
                _ => {}
            }
        };
        let spec = Self::spec("printf one; sleep 1; printf two; sleep 60");
        let mut run = Box::pin(jobs.run_held(&scope, &id, &spec, &events));
        // Boxed, not stack-pinned: dropping it below must really close the hold.
        let ready = async {
            loop {
                if heard.lock().unwrap_or_else(|e| e.into_inner()).1 == b"one" {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        };
        tokio::select! {
            r = &mut run => return fail(JOBS_KEEP, format!("the job ended early: {r:?}")),
            r = tokio::time::timeout(Duration::from_secs(10), ready) => if r.is_err() {
                return fail(JOBS_KEEP, "no `one` within 10 s");
            },
        }
        if let Err(e) = jobs.keep(&scope, &id, 7, "background", 3).await {
            return fail(JOBS_KEEP, format!("keep: {e}"));
        }
        drop(run);
        let Some(pgid) = heard.lock().unwrap_or_else(|e| e.into_inner()).0 else {
            return fail(JOBS_KEEP, "no `started` frame");
        };
        tokio::time::sleep(Duration::from_millis(1500)).await;
        match self.group_alive(pgid).await {
            Ok(true) => {}
            Ok(false) => return fail(JOBS_KEEP, "the kept job died when its hold closed"),
            Err(e) => return fail(JOBS_KEEP, e),
        }
        let listed = match jobs.list(&scope).await {
            Ok(l) => l,
            Err(e) => return fail(JOBS_KEEP, format!("list: {e}")),
        };
        match listed.iter().find(|j| j.id == id) {
            Some(j)
                if j.kept
                    && j.num == Some(7)
                    && j.kind.as_deref() == Some("background")
                    && j.seen == 3
                    && j.status == JobStatus::Running => {}
            other => {
                return fail(
                    JOBS_KEEP,
                    format!(
                        "list should show it running, kept as job 7 (background, seen 3): {other:?}"
                    ),
                );
            }
        }
        let mut later = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(5);
        while later != b"two" && Instant::now() < deadline {
            match jobs.read(&scope, &id, 3, 1024).await {
                Ok(c) => later = c.data,
                Err(e) => return fail(JOBS_KEEP, format!("read: {e}")),
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        if later != b"two" {
            return fail(
                JOBS_KEEP,
                format!(
                    "read from offset 3 gave {:?}, want \"two\"",
                    String::from_utf8_lossy(&later)
                ),
            );
        }
        let follow = std::sync::Mutex::new(Vec::new());
        let events = |e: JobEvent<'_>| {
            if let JobEvent::Output(b) = e {
                follow
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .extend_from_slice(b);
            }
        };
        let attach = jobs.follow(&scope, &id, 3, &events);
        let got_two = async {
            loop {
                if follow.lock().unwrap_or_else(|e| e.into_inner()).as_slice() == b"two" {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        };
        tokio::select! {
            r = attach => return fail(JOBS_KEEP, format!("attach ended early: {r:?}")),
            r = tokio::time::timeout(Duration::from_secs(5), got_two) => if r.is_err() {
                return fail(JOBS_KEEP, "attach from offset 3 did not stream \"two\"");
            },
        }
        // Closing an attach must not kill.
        tokio::time::sleep(Duration::from_millis(500)).await;
        match self.group_alive(pgid).await {
            Ok(true) => pass(
                JOBS_KEEP,
                "survived its hold, listed as kept job 7, read and attached from offset 3, and an \
                 attach that closed did not kill it",
            ),
            Ok(false) => fail(JOBS_KEEP, "closing an attach killed the job"),
            Err(e) => fail(JOBS_KEEP, e),
        }
    }

    async fn jobs_kill(&self) -> Check {
        let jobs = NativeJobs::over(&self.runner);
        let scope = self.scope("kill");
        let id = JobId::fresh();
        let check = async {
            let pgid = jobs
                .start_kept(
                    &scope,
                    &id,
                    &Self::spec("trap '' TERM; sleep 60 & sleep 60 & wait"),
                )
                .await
                .map_err(|e| format!("start: {e}"))?;
            tokio::time::sleep(Duration::from_millis(300)).await;
            if !self.group_alive(pgid).await? {
                return Err("the job was not running".to_string());
            }
            let survivors = jobs
                .kill(&scope, std::slice::from_ref(&id), true)
                .await
                .map_err(|e| format!("kill: {e}"))?;
            if !survivors.is_empty() {
                return Err(format!("survivors reported: {survivors:?}"));
            }
            if self.group_alive(pgid).await? {
                return Err(
                    "the group still runs after kill answered (a job that ignores SIGTERM \
                            must get SIGKILL)"
                        .to_string(),
                );
            }
            let gone = jobs
                .read(&scope, &id, 0, 1)
                .await
                .map_err(|e| format!("read: {e}"))?;
            if gone.status != JobStatus::Gone {
                return Err(format!(
                    "a removed job reads as {:?}, want gone",
                    gone.status
                ));
            }
            Ok(
                "SIGTERM ignored, SIGKILL followed; the group was gone when kill answered, and the \
                job then read as gone",
            )
        }
        .await;
        let _ = jobs.wipe(&scope).await;
        match check {
            Ok(d) => pass(JOBS_KILL, d),
            Err(e) => fail(JOBS_KILL, e),
        }
    }

    async fn jobs_scope(&self) -> Check {
        let jobs = NativeJobs::over(&self.runner);
        let (a, b) = (self.scope("scope-a"), self.scope("scope-b"));
        let check = async {
            let mut mine = Vec::new();
            for _ in 0..2 {
                mine.push(
                    jobs.start_kept(&a, &JobId::fresh(), &Self::spec("sleep 60"))
                        .await
                        .map_err(|e| format!("start: {e}"))?,
                );
            }
            let theirs = jobs
                .start_kept(&b, &JobId::fresh(), &Self::spec("sleep 60"))
                .await
                .map_err(|e| format!("start: {e}"))?;
            let (la, lb) = (
                jobs.list(&a).await.map_err(|e| e.to_string())?,
                jobs.list(&b).await.map_err(|e| e.to_string())?,
            );
            if la.len() != 2 || lb.len() != 1 {
                return Err(format!(
                    "list gave {} and {} jobs, want 2 and 1",
                    la.len(),
                    lb.len()
                ));
            }
            jobs.wipe(&a).await.map_err(|e| format!("wipe: {e}"))?;
            for g in mine {
                if self.group_alive(g).await? {
                    return Err("a wiped scope's job still runs".to_string());
                }
            }
            if !jobs.list(&a).await.map_err(|e| e.to_string())?.is_empty() {
                return Err("a wiped scope still lists jobs".to_string());
            }
            if !self.group_alive(theirs).await? {
                return Err("wiping one scope killed another's job".to_string());
            }
            Ok("two scopes listed apart; wiping one killed its jobs and left the other's running")
        }
        .await;
        let _ = jobs.wipe(&a).await;
        let _ = jobs.wipe(&b).await;
        match check {
            Ok(d) => pass(JOBS_SCOPE, d),
            Err(e) => fail(JOBS_SCOPE, e),
        }
    }
}
