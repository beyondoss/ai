//! The streamed response and its frames.
//!
//! The normative spec is the parent module's doc ("Streaming"). This file is the client half:
//! `FrameReader` turns an `application/x-ndjson` body into [`Frame`]s ([`decode_line`] is one line).

use std::time::Duration;

use base64::Engine as _;
use serde::Deserialize;

/// The media type of a streamed response.
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

/// One decoded line of a stream.
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
    /// own (the job streams, `jobs_native`). Errors as [`Self::next`].
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic)]
    use super::*;

    /// The golden vectors: what a conforming client makes of each body. An endpoint implementer can
    /// run their encoder's output through the same cases.
    #[test]
    fn the_frame_vectors_decode_as_specified() {
        let raw = include_str!("../../tests/fixtures/exec/frames.json");
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
}
