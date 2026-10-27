//! Exact output for a one-shot call (v1.1 `output_base64`): the probe, and the three ways
//! [`HttpExecRunner::run_exact`](crate::Exec::run_exact) gets a command's bytes. The normative text
//! is the module doc's "Exact output".

use std::time::Duration;

use base64::Engine as _;

use super::stream::{FRAME_SILENCE_LIMIT, Frame, FrameReader};
use super::{ExecRequest, ExecResponse, HttpExecRunner, is_ndjson};
use crate::exec::ExecOutput;

/// The probe command: the bytes `ff 00 0a`, which no text round trip keeps (0xff is not UTF-8, NUL
/// does not survive every string type, and the newline proves nothing is trimmed).
pub const PROBE_SCRIPT: &str = r"printf '\377\000\n'";

/// What [`PROBE_SCRIPT`] writes.
pub const PROBE_BYTES: [u8; 3] = [0xff, 0x00, 0x0a];

/// What a v1.1 endpoint puts at the end of stderr for a command it killed at its deadline.
const TIMEOUT_NOTE: &str = "command exceeded its timeout";

/// Room for the JSON around two base64 strings (the keys, the exit code, an error's text).
const ENVELOPE_SLACK: usize = 4096;

/// The most body a client reads for an `output_base64` answer under response cap `cap`: the base64
/// of `cap` decoded bytes, plus the envelope. The cap itself is on the decoded bytes (the
/// module doc, "Exact output").
pub fn body_cap(cap: usize) -> usize {
    cap.div_ceil(3)
        .saturating_mul(4)
        .saturating_add(ENVELOPE_SLACK)
}

/// The error for an answer past the response cap: the command ran. `FileTooLarge`, the kind (and
/// the words) of the text answer's, so a caller handles one.
fn too_large(url: &str, cap: usize) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::FileTooLarge,
        format!(
            "exec endpoint {url}: the response exceeded the {cap}-byte cap and was discarded — the \
             command ran, but its output is too large to return; narrow it (e.g. pipe through \
             `head`) rather than re-running it unchanged"
        ),
    )
}

fn protocol(url: &str, what: &str) -> std::io::Error {
    std::io::Error::other(format!("exec endpoint {url}: {what}"))
}

impl HttpExecRunner {
    /// Whether the endpoint honours `output_base64`, probing it the first time: `printf '\377\000\n'`
    /// must come back as `stdout_base64` decoding to exactly `ff 00 0a`. A front that refuses the
    /// field (`400`) has answered: no. A probe that could not reach the endpoint is not an answer.
    pub async fn outputs_base64(&self) -> bool {
        self.try_outputs_base64().await.unwrap_or(false)
    }

    /// [`Self::outputs_base64`], but a probe that failed is an error, not "no". Not cached.
    pub async fn try_outputs_base64(&self) -> std::io::Result<bool> {
        self.output_probe
            .get_or_try_init(|| async {
                let args = ["-c".to_string(), PROBE_SCRIPT.to_string()];
                let resp = match self
                    .send_exact("sh", &args, None, Duration::from_secs(10), None)
                    .await
                {
                    Ok(resp) => resp,
                    Err(e) if e.kind() == std::io::ErrorKind::InvalidInput => return Ok(false),
                    Err(e) => return Err(e),
                };
                let seen = self.envelope_to(resp, body_cap(PROBE_BYTES.len())).await?;
                let honoured = probe_honoured(&seen);
                if !honoured {
                    tracing::info!(url = %self.url, "exec endpoint does not honour output_base64; raw exec output is exact only over a v1.2 stream");
                }
                Ok::<bool, std::io::Error>(honoured)
            })
            .await
            .copied()
    }

    /// One v1.1 request carrying `output_base64`; the answer is the caller's to read (the verifier
    /// inspects it field by field).
    #[doc(hidden)]
    pub async fn send_exact(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        stdin: Option<&[u8]>,
    ) -> std::io::Result<reqwest::Response> {
        let body = ExecRequest {
            command: program,
            args,
            cwd,
            timeout_ms: timeout.as_millis().min(u64::MAX as u128) as u64,
            stdin_base64: stdin.map(|b| base64::engine::general_purpose::STANDARD.encode(b)),
            stream: false,
            job: None,
            output_base64: true,
        };
        self.send_request(timeout, &body).await
    }

    /// `run_exact`: see the trait method.
    pub(super) async fn exact(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        stdin: Option<&[u8]>,
    ) -> std::io::Result<ExecOutput> {
        if self.try_outputs_base64().await? {
            let resp = self.send_exact(program, args, cwd, timeout, stdin).await?;
            let parsed = self
                .envelope_to(resp, body_cap(self.max_response_bytes))
                .await?;
            return self.output_of(parsed);
        }
        // Without a stdin on the wire for streams (the spec does not give a stream one), a call
        // with stdin is a v1.1 call, whose text is what it is.
        if stdin.is_none() && self.try_streams().await? {
            return self.exact_over_stream(program, args, cwd, timeout).await;
        }
        let parsed = self.post(program, args, cwd, timeout, stdin).await?;
        Ok(text_output(parsed))
    }

    /// An `output_base64` answer, taken apart. See [`decode_answer`].
    pub(crate) fn output_of(&self, parsed: ExecResponse) -> std::io::Result<ExecOutput> {
        decode_answer(&self.url, self.max_response_bytes, parsed)
    }

    /// One-shot over a v1.2 stream: its frames are base64 already. The v1.1 shape of the result
    /// (`128 + N`, `124` and the note) is built here so every path answers alike.
    async fn exact_over_stream(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
    ) -> std::io::Result<ExecOutput> {
        let resp = self.send(program, args, cwd, timeout, None, true).await?;
        if !is_ndjson(&resp) {
            // One object after all (a fleet mid-rollout): text.
            return Ok(text_output(self.envelope(resp).await?));
        }
        let mut frames = FrameReader::new(resp);
        let broken = |e: String| {
            protocol(
                &self.url,
                &format!("{e} (the command may have been killed)"),
            )
        };
        let mut fold = Fold::default();
        loop {
            let Some(frame) = frames.next(FRAME_SILENCE_LIMIT).await.map_err(broken)? else {
                return Err(broken(
                    "the stream ended before the command's final frame".to_string(),
                ));
            };
            let graceful = matches!(frame, Frame::Exit(_) | Frame::Signal(_));
            if let Some(out) = fold.feed(frame).map_err(broken)? {
                if graceful {
                    frames.finish().await;
                }
                return Ok(out);
            }
            if fold.len() > self.max_response_bytes {
                // Dropping the reader closes the stream, which stops the command.
                return Err(too_large(&self.url, self.max_response_bytes));
            }
        }
    }
}

/// Whether a probe's answer shows the endpoint honouring `output_base64`: `stdout_base64` decoding
/// to exactly [`PROBE_BYTES`].
fn probe_honoured(answer: &ExecResponse) -> bool {
    answer
        .stdout_base64
        .as_deref()
        .and_then(|b| base64::engine::general_purpose::STANDARD.decode(b).ok())
        .is_some_and(|b| b == PROBE_BYTES)
}

/// An `output_base64` answer. Both `stdout_base64` and `stderr_base64`: the bytes, exact (the text
/// fields, if the endpoint sent them too, are ignored). Neither: the endpoint did not honour the
/// field this time (one replica of a fleet in rollout), so it is a text answer and says so. One
/// without the other is malformed, and an error: reading the missing stream as empty would lose
/// output without a word.
fn decode_answer(url: &str, cap: usize, parsed: ExecResponse) -> std::io::Result<ExecOutput> {
    let (out, err) = match (&parsed.stdout_base64, &parsed.stderr_base64) {
        (Some(out), Some(err)) => (out, err),
        (None, None) => return Ok(text_output(parsed)),
        _ => {
            return Err(protocol(
                url,
                "the answer has only one of `stdout_base64` and `stderr_base64`",
            ));
        }
    };
    let decode = |field: &str, b: &str| {
        base64::engine::general_purpose::STANDARD
            .decode(b)
            .map_err(|e| protocol(url, &format!("`{field}` is not standard base64 ({e})")))
    };
    let (stdout, stderr) = (decode("stdout_base64", out)?, decode("stderr_base64", err)?);
    if stdout.len().saturating_add(stderr.len()) > cap {
        return Err(too_large(url, cap));
    }
    Ok(ExecOutput {
        code: parsed.exit_code,
        stdout,
        stderr,
        exact: true,
    })
}

/// Frames of a one-shot call made over a v1.2 stream, folded into the v1.1 shape of its result.
#[derive(Default)]
struct Fold {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

impl Fold {
    /// Output held so far.
    fn len(&self) -> usize {
        self.stdout.len().saturating_add(self.stderr.len())
    }

    /// One frame: the finished result at the frame that ends a one-shot call, `None` before it.
    /// A one-shot call is answered when the command exits, so `shell_exit` ends it (what the command
    /// left running is not its business; closing the stream has the endpoint kill it).
    fn feed(&mut self, frame: Frame) -> Result<Option<ExecOutput>, String> {
        let code = match frame {
            Frame::Stdout(b) => {
                self.stdout.extend_from_slice(&b);
                return Ok(None);
            }
            Frame::Stderr(b) => {
                self.stderr.extend_from_slice(&b);
                return Ok(None);
            }
            Frame::Heartbeat => return Ok(None),
            Frame::ShellExit(code) | Frame::Exit(code) => code,
            Frame::Signal(s) => 128 + s,
            Frame::TimedOut => {
                if !self.stderr.is_empty() && !self.stderr.ends_with(b"\n") {
                    self.stderr.push(b'\n');
                }
                self.stderr.extend_from_slice(TIMEOUT_NOTE.as_bytes());
                124
            }
            Frame::Error(e) => return Err(format!("the endpoint failed mid-command: {e}")),
        };
        Ok(Some(ExecOutput {
            code,
            stdout: std::mem::take(&mut self.stdout),
            stderr: std::mem::take(&mut self.stderr),
            exact: true,
        }))
    }
}

/// A text answer as bytes: never exact.
fn text_output(r: ExecResponse) -> ExecOutput {
    ExecOutput {
        code: r.exit_code,
        stdout: r.stdout.into_bytes(),
        stderr: r.stderr.into_bytes(),
        exact: false,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic)]
    use super::*;
    use crate::exec_endpoint::stream::decode_line;

    fn doc() -> serde_json::Value {
        serde_json::from_str(include_str!("../../tests/fixtures/exec/v1.1-output.json")).unwrap()
    }

    fn b64(v: &serde_json::Value) -> Vec<u8> {
        base64::engine::general_purpose::STANDARD
            .decode(v.as_str().unwrap())
            .unwrap()
    }

    fn check(name: &str, got: Result<ExecOutput, String>, expect: &serde_json::Value) {
        if expect["invalid"] == true {
            assert!(
                got.is_err(),
                "{name}: a conforming client reports an error, got {got:?}"
            );
            return;
        }
        let got = got.unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            i64::from(got.code),
            expect["exit_code"].as_i64().unwrap(),
            "{name}: exit code"
        );
        assert_eq!(got.stdout, b64(&expect["stdout_base64"]), "{name}: stdout");
        assert_eq!(got.stderr, b64(&expect["stderr_base64"]), "{name}: stderr");
        assert_eq!(got.exact, expect["exact"] == true, "{name}: exact");
    }

    #[test]
    fn the_probe_is_the_bytes_the_vectors_name() {
        let d = doc();
        assert_eq!(d["probe"]["script"], PROBE_SCRIPT);
        assert_eq!(b64(&d["probe"]["bytes_base64"]), PROBE_BYTES);
    }

    #[test]
    fn the_probe_vectors_are_read_as_specified() {
        for case in doc()["probes"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let parsed = serde_json::from_value::<ExecResponse>(case["answer"].clone());
            if case["error"] == true {
                assert!(parsed.is_err(), "{name}: not a v1.1 answer, so an error");
            } else {
                let honoured = probe_honoured(&parsed.unwrap());
                assert_eq!(honoured, case["supported"] == true, "{name}");
            }
        }
    }

    #[test]
    fn the_answer_vectors_decode_as_specified() {
        for case in doc()["answers"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let got = serde_json::from_value::<ExecResponse>(case["answer"].clone())
                .map_err(|e| e.to_string())
                .and_then(|p| decode_answer("http://x", 1 << 20, p).map_err(|e| e.to_string()));
            check(name, got, &case["expect"]);
        }
    }

    #[test]
    fn the_stream_vectors_fold_into_the_v11_shape() {
        for case in doc()["streams"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let mut fold = Fold::default();
            let mut result = Err("the body ended without a final frame".to_string());
            'lines: for line in case["lines"].as_array().unwrap() {
                for f in decode_line(line.as_str().unwrap().as_bytes()).unwrap() {
                    match fold.feed(f) {
                        Ok(None) => {}
                        Ok(Some(out)) => {
                            result = Ok(out);
                            break 'lines;
                        }
                        Err(e) => {
                            result = Err(e);
                            break 'lines;
                        }
                    }
                }
            }
            check(name, result, &case["expect"]);
        }
    }

    #[test]
    fn the_cap_is_on_the_decoded_bytes_of_both_streams_together() {
        let answer = |cap| {
            decode_answer(
                "http://x",
                cap,
                ExecResponse {
                    exit_code: 0,
                    stdout: String::new(),
                    stderr: String::new(),
                    stdout_base64: Some("AAAA".into()),
                    stderr_base64: Some("AAAA".into()),
                },
            )
        };
        assert!(answer(6).is_ok(), "3 + 3 decoded bytes fit a cap of 6");
        let over = answer(5).unwrap_err();
        assert_eq!(over.kind(), std::io::ErrorKind::FileTooLarge);
        assert!(over.to_string().contains("the command ran"), "{over}");
    }

    #[test]
    fn the_body_cap_holds_the_base64_of_a_cap_of_output() {
        // 16 MiB of output is 21_845_334 base64 characters (4 per 3 bytes, padded up).
        assert!(body_cap(16 * 1024 * 1024) >= 22_369_620);
        assert_eq!(body_cap(3), 4 + ENVELOPE_SLACK);
        assert_eq!(body_cap(4), 8 + ENVELOPE_SLACK);
    }
}
