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
//!   cargo test -p beyond-ai-agent --test exec_conformance -- --nocapture
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

use std::time::{Duration, Instant};

use super::HttpExecRunner;
use super::stream::{self, Frame, FrameReader, HEARTBEAT_INTERVAL};

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
}

impl Default for Expectations {
    fn default() -> Self {
        Self {
            heartbeat: HEARTBEAT_INTERVAL,
            kill_within: HEARTBEAT_INTERVAL * 2,
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
            if self.streams {
                "v1.2 (streams)"
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

/// Run every check against the endpoint at `url`.
pub async fn verify(url: &str, headers: &[(String, String)], expect: &Expectations) -> Report {
    let mut runner = match HttpExecRunner::new(url) {
        Ok(r) => r,
        Err(e) => {
            return Report {
                streams: false,
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
        ] {
            checks.push(skip(name, "not a v1.2 endpoint"));
        }
        return Report { streams, checks };
    }
    checks.push(v.incremental().await);
    checks.push(v.bytes().await);
    checks.push(v.frame_size().await);
    checks.push(v.exit().await);
    checks.push(v.signal().await);
    // The three slow checks wait on timers, not on each other.
    let (t, h, d) = tokio::join!(v.timeout(), v.heartbeat(), v.disconnect());
    checks.extend([t, h, d]);
    Report { streams, checks }
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
}
