//! Exec protocol **v1.2**: the streamed response, its frames, and the probe that adopts it.
//!
//! The normative spec is the parent module's doc ("Streaming (v1.2)"). This file is the client half:
//! `FrameReader` turns an `application/x-ndjson` body into [`Frame`]s ([`decode_line`] is one line),
//! and the probe (`observe_probe`) decides whether an endpoint has *proved* it streams before `bash`
//! is allowed to treat its silence as the command's.

use std::time::{Duration, Instant};

use base64::Engine as _;
use serde::Deserialize;

/// The media type of a v1.2 streamed response. Anything else in answer to `"stream": true` is a
/// v1.1 endpoint that ignored the field and sent the whole result as one JSON object.
pub const NDJSON: &str = "application/x-ndjson";

/// The longest gap the endpoint may leave between frames: it sends a heartbeat (`{}`) whenever the
/// command has been silent this long. Short enough to keep an idle connection open through every
/// common proxy (their idle timeouts start at 60 s), and to bound how long a vanished client goes
/// unnoticed by an endpoint that only learns of the disconnect when a write fails.
pub const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);

/// How long the client waits for *any* frame before it decides the connection is dead: six missed
/// heartbeats. Without it a stream whose far side vanished (a sandbox host lost power, a NAT entry
/// expired) would hold a `bash` job until the command's own timeout — the hang streaming exists to
/// end.
pub const FRAME_SILENCE_LIMIT: Duration = Duration::from_secs(30);

/// The most output one frame may carry, decoded. The endpoint MUST split anything longer.
pub const MAX_FRAME_PAYLOAD: usize = 64 * 1024;

/// The longest line the client accepts: a full-size frame base64-encodes to ~87 KiB, so this leaves
/// ample room for a lenient endpoint while still bounding what a broken one can make this process
/// buffer while it waits for a newline.
const MAX_LINE: usize = 1024 * 1024;

/// One decoded line of a v1.2 stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Frame {
    /// Bytes the command wrote to stdout.
    Stdout(Vec<u8>),
    /// Bytes the command wrote to stderr.
    Stderr(Vec<u8>),
    /// A keepalive (`{}`, or any frame with no field this version knows). Not output: it says the
    /// connection is alive, never that the command printed anything.
    Heartbeat,
    /// Not final: the command's own process (the shell) exited with this code, but processes of its
    /// group are still running. The stream stays open until they have all exited, then ends with the
    /// shell's final frame. See "Leftover processes" in the parent module's doc.
    ShellExit(i32),
    /// Final: the command exited with this code.
    Exit(i32),
    /// Final: the command was killed by this signal.
    Signal(i32),
    /// Final: the endpoint killed the command at its `timeout_ms`.
    TimedOut,
    /// Final: the endpoint failed after it had started answering (lost the sandbox, internal error).
    Error(String),
}

impl Frame {
    /// Whether this frame ends the stream. Nothing may follow it.
    pub fn is_final(&self) -> bool {
        matches!(
            self,
            Frame::Exit(_) | Frame::Signal(_) | Frame::TimedOut | Frame::Error(_)
        )
    }
}

/// A frame as it appears on the wire. Every field optional: which one is present says what the frame
/// is, and unknown fields are ignored so a later version can add some.
#[derive(Deserialize)]
struct WireFrame {
    stdout: Option<String>,
    stderr: Option<String>,
    shell_exit: Option<i32>,
    exit_code: Option<i32>,
    signal: Option<i32>,
    #[serde(default)]
    timed_out: bool,
    error: Option<String>,
}

/// Decode one line (without its `\n`). A frame names exactly one field; one that names more is
/// split rather than truncated — output first, then the final field — so a sloppy endpoint loses
/// nothing.
pub fn decode_line(line: &[u8]) -> Result<Vec<Frame>, String> {
    let line = line.strip_suffix(b"\r").unwrap_or(line);
    if line.iter().all(u8::is_ascii_whitespace) {
        // A blank line carries nothing; tolerated rather than failed, like an extra heartbeat.
        return Ok(vec![Frame::Heartbeat]);
    }
    let wire: WireFrame = serde_json::from_slice(line).map_err(|e| {
        format!(
            "unparseable frame ({e}): {}",
            String::from_utf8_lossy(&line[..line.len().min(200)])
        )
    })?;
    let b64 = |field: &str, s: &str| {
        base64::engine::general_purpose::STANDARD
            .decode(s)
            .map_err(|e| format!("`{field}` is not standard base64 ({e})"))
    };
    let mut frames = Vec::with_capacity(1);
    if let Some(s) = &wire.stdout {
        frames.push(Frame::Stdout(b64("stdout", s)?));
    }
    if let Some(s) = &wire.stderr {
        frames.push(Frame::Stderr(b64("stderr", s)?));
    }
    if let Some(code) = wire.shell_exit {
        frames.push(Frame::ShellExit(code));
    }
    if let Some(e) = wire.error {
        frames.push(Frame::Error(e));
    } else if let Some(code) = wire.exit_code {
        frames.push(Frame::Exit(code));
    } else if let Some(sig) = wire.signal {
        frames.push(Frame::Signal(sig));
    } else if wire.timed_out {
        frames.push(Frame::TimedOut);
    }
    if frames.is_empty() {
        frames.push(Frame::Heartbeat);
    }
    Ok(frames)
}

/// Reads frames off a streamed response body, one line at a time. Holds at most one partial line
/// (bounded by [`MAX_LINE`]) — the output itself goes straight to the caller, so a command that
/// prints gigabytes costs this reader nothing but time.
pub struct FrameReader {
    resp: reqwest::Response,
    buf: Vec<u8>,
    /// Bytes of `buf` already searched for a newline, so a long partial line is not rescanned on
    /// every chunk.
    scanned: usize,
    pending: std::collections::VecDeque<Frame>,
    ended: bool,
}

impl FrameReader {
    pub fn new(resp: reqwest::Response) -> Self {
        Self {
            resp,
            buf: Vec::new(),
            scanned: 0,
            pending: std::collections::VecDeque::new(),
            ended: false,
        }
    }

    /// The next frame, or `None` at the end of the body. An `Err` is a broken stream: the
    /// connection failed, a line did not parse, or nothing at all (not even a heartbeat) arrived for
    /// `silence`.
    pub async fn next(&mut self, silence: Duration) -> Result<Option<Frame>, String> {
        loop {
            if let Some(f) = self.pending.pop_front() {
                return Ok(Some(f));
            }
            match self.next_line(silence).await? {
                Some(line) => {
                    let frames = decode_line(&line)?;
                    self.pending.extend(frames);
                }
                None => return Ok(None),
            }
        }
    }

    /// The next line of the body (without its `\n`), or `None` at the end — for a decoder of its
    /// own (the v1.3 job streams, `jobs_native`). Errors as [`Self::next`].
    pub async fn next_line(&mut self, silence: Duration) -> Result<Option<Vec<u8>>, String> {
        loop {
            if let Some(nl) = self.buf[self.scanned..].iter().position(|&b| b == b'\n') {
                let end = self.scanned + nl;
                let line = self.buf[..end].to_vec();
                self.buf.drain(..=end);
                self.scanned = 0;
                return Ok(Some(line));
            }
            self.scanned = self.buf.len();
            if self.buf.len() > MAX_LINE {
                return Err(format!(
                    "a frame exceeded {MAX_LINE} bytes without a newline"
                ));
            }
            if self.ended {
                return if self.buf.is_empty() {
                    Ok(None)
                } else {
                    Err("the stream ended in the middle of a frame".to_string())
                };
            }
            match tokio::time::timeout(silence, self.resp.chunk()).await {
                Err(_) => {
                    return Err(format!(
                        "no frame for {} s — the endpoint stopped sending heartbeats, so the \
                         connection is presumed dead",
                        silence.as_secs()
                    ));
                }
                Ok(Err(e)) => {
                    return Err(if e.is_timeout() {
                        format!("timed out: {e}")
                    } else {
                        format!("the stream broke: {e}")
                    });
                }
                Ok(Ok(None)) => self.ended = true,
                Ok(Ok(Some(chunk))) => self.buf.extend_from_slice(&chunk),
            }
        }
    }

    /// After the final frame: give the body a moment to end so the connection can go back to the
    /// pool instead of being torn down. Bounded — a conforming endpoint ends the body right after
    /// the final frame, and one that does not costs a connection, never a hang.
    pub async fn finish(mut self) {
        if !self.ended {
            let _ = tokio::time::timeout(Duration::from_millis(500), self.resp.chunk()).await;
        }
    }
}

/// The probe's script: output, a pause, more output. Only a stream that delivers the first part a
/// clear interval before the end proves it is incremental; a buffering endpoint or proxy delivers
/// everything at once. `sleep 0.3` is not POSIX (busybox has it, some minimal `sleep`s do not), so
/// it falls back to a whole second rather than failing the probe.
pub const PROBE_SCRIPT: &str = "printf a; sleep 0.3 2>/dev/null || sleep 1; printf b";

/// The least gap between the probe's first output frame and its final frame that counts as
/// incremental delivery. A third of the script's pause: wide enough to absorb scheduling noise on a
/// loaded host, far above the ~0 a buffered body shows.
pub const PROBE_MIN_GAP: Duration = Duration::from_millis(100);

/// What a probe saw, for the verdict and for the conformance report.
#[derive(Debug, Default)]
pub struct ProbeSeen {
    pub ndjson: bool,
    pub output_frames: usize,
    pub stdout: Vec<u8>,
    pub end: Option<Frame>,
    /// From the first output frame to the final frame.
    pub gap: Option<Duration>,
}

impl ProbeSeen {
    /// The probe's verdict: streaming is adopted only on this.
    pub fn streams(&self) -> bool {
        self.ndjson
            && self.output_frames >= 2
            && self.stdout == b"ab"
            && self.end == Some(Frame::Exit(0))
            && self.gap.is_some_and(|g| g >= PROBE_MIN_GAP)
    }

    /// Why the verdict is what it is, in a sentence.
    pub fn explain(&self) -> String {
        if !self.ndjson {
            return "answered `\"stream\": true` with a single JSON object (a v1.1 endpoint)"
                .into();
        }
        if self.end != Some(Frame::Exit(0)) || self.stdout != b"ab" {
            return format!(
                "the probe did not complete as `ab` + exit 0 (stdout {:?}, end {:?})",
                String::from_utf8_lossy(&self.stdout),
                self.end
            );
        }
        if self.output_frames < 2 {
            return "sent the probe's output as one frame: it buffered the output until exit"
                .into();
        }
        match self.gap {
            Some(g) if g >= PROBE_MIN_GAP => format!(
                "streamed incrementally ({} output frames, first {} ms before the end)",
                self.output_frames,
                g.as_millis()
            ),
            g => format!(
                "delivered every frame at once ({} ms apart): something buffers the stream",
                g.unwrap_or_default().as_millis()
            ),
        }
    }
}

/// Read a probe's stream to its end, timing it.
pub async fn observe_probe(resp: reqwest::Response) -> Result<ProbeSeen, String> {
    let mut seen = ProbeSeen {
        ndjson: true,
        ..ProbeSeen::default()
    };
    let mut reader = FrameReader::new(resp);
    let mut first_output: Option<Instant> = None;
    while let Some(frame) = reader.next(FRAME_SILENCE_LIMIT).await? {
        match frame {
            Frame::Stdout(b) => {
                first_output.get_or_insert_with(Instant::now);
                seen.output_frames += 1;
                seen.stdout.extend_from_slice(&b);
            }
            Frame::Stderr(_) => {
                first_output.get_or_insert_with(Instant::now);
                seen.output_frames += 1;
            }
            Frame::Heartbeat => {}
            end => {
                seen.gap = first_output.map(|t| t.elapsed());
                seen.end = Some(end);
                reader.finish().await;
                return Ok(seen);
            }
        }
    }
    Ok(seen)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic)]
    use super::*;

    /// The golden vectors: what a conforming client makes of each body. An endpoint implementer can
    /// run their encoder's output through the same cases.
    #[test]
    fn the_frame_vectors_decode_as_specified() {
        let raw = include_str!("../../tests/fixtures/exec/v1.2-frames.json");
        let doc: serde_json::Value = serde_json::from_str(raw).unwrap();
        let b64 = |v: &serde_json::Value| {
            base64::engine::general_purpose::STANDARD
                .decode(v.as_str().unwrap_or(""))
                .unwrap()
        };
        for case in doc["cases"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let mut stdout = Vec::new();
            let mut stderr = Vec::new();
            let mut heartbeats = 0;
            let mut shell_exits = Vec::new();
            let mut end = None;
            let mut error = None;
            for line in case["lines"].as_array().unwrap() {
                match decode_line(line.as_str().unwrap().as_bytes()) {
                    Err(e) => {
                        error = Some(e);
                        break;
                    }
                    Ok(frames) => {
                        for f in frames {
                            assert!(end.is_none(), "{name}: a frame followed the final frame");
                            match f {
                                Frame::Stdout(b) => stdout.extend(b),
                                Frame::Stderr(b) => stderr.extend(b),
                                Frame::Heartbeat => heartbeats += 1,
                                Frame::ShellExit(c) => shell_exits.push(serde_json::json!(c)),
                                other => end = Some(other),
                            }
                        }
                    }
                }
            }
            let want = &case["expect"];
            if want["invalid"].as_bool() == Some(true) {
                assert!(error.is_some(), "{name}: must be rejected");
                continue;
            }
            assert!(error.is_none(), "{name}: {error:?}");
            assert_eq!(stdout, b64(&want["stdout_base64"]), "{name}: stdout");
            assert_eq!(stderr, b64(&want["stderr_base64"]), "{name}: stderr");
            assert_eq!(
                heartbeats,
                want["heartbeats"].as_u64().unwrap_or(0) as usize,
                "{name}: heartbeats"
            );
            let no_shell_exits = Vec::new();
            assert_eq!(
                &shell_exits,
                want["shell_exits"].as_array().unwrap_or(&no_shell_exits),
                "{name}: shell exits"
            );
            let want_end = match &want["end"] {
                v if v["exit_code"].is_i64() => {
                    Some(Frame::Exit(v["exit_code"].as_i64().unwrap() as i32))
                }
                v if v["signal"].is_i64() => {
                    Some(Frame::Signal(v["signal"].as_i64().unwrap() as i32))
                }
                v if v["timed_out"].as_bool() == Some(true) => Some(Frame::TimedOut),
                v if v["error"].is_string() => {
                    Some(Frame::Error(v["error"].as_str().unwrap().to_string()))
                }
                _ => None,
            };
            assert_eq!(end, want_end, "{name}: final frame");
        }
    }

    #[test]
    fn a_probe_is_adopted_only_on_incremental_delivery() {
        let good = ProbeSeen {
            ndjson: true,
            output_frames: 2,
            stdout: b"ab".to_vec(),
            end: Some(Frame::Exit(0)),
            gap: Some(Duration::from_millis(300)),
        };
        assert!(good.streams(), "{}", good.explain());
        let v11 = ProbeSeen::default();
        assert!(!v11.streams());
        assert!(v11.explain().contains("v1.1"));
        let one_frame = ProbeSeen {
            output_frames: 1,
            ..clone(&good)
        };
        assert!(!one_frame.streams());
        assert!(one_frame.explain().contains("one frame"));
        let buffered = ProbeSeen {
            gap: Some(Duration::from_millis(2)),
            ..clone(&good)
        };
        assert!(!buffered.streams());
        assert!(buffered.explain().contains("buffers"));
    }

    fn clone(p: &ProbeSeen) -> ProbeSeen {
        ProbeSeen {
            ndjson: p.ndjson,
            output_frames: p.output_frames,
            stdout: p.stdout.clone(),
            end: p.end.clone(),
            gap: p.gap,
        }
    }
}
