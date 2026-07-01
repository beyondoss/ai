//! `bash` — run a shell command via `sh -c` and stream/return its combined output.
//!
//! Output handling mirrors pi: raw stdout+stderr feed one [`OutputAccumulator`] (tail-truncated for
//! display, full stream spilled to a temp file → `Full output: <path>`), and while the command runs
//! the tool emits an initial empty update then **throttled snapshots** (the whole output so far, every
//! [`UPDATE_THROTTLE`]) via its [`ToolProgress`] sink. Non-zero exit / timeout become errors carrying
//! the output — same as pi's throw.

use std::borrow::Cow;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use agent_core::tool::Tool;
use agent_core::{ToolError, ToolOutput, ToolProgress};
use async_trait::async_trait;
use regex::Regex;
use serde_json::{Value, json};

use super::exec::{ChunkSink, CommandRunner, RealRunner};
use super::output::{OutputAccumulator, OutputSnapshot, format_output};

/// Default command timeout (ms) when the model doesn't specify one.
const DEFAULT_TIMEOUT_MS: u64 = 120_000;
/// Minimum gap between streamed progress snapshots — pi's `BASH_UPDATE_THROTTLE_MS`. Keeps a chatty
/// command from flooding the event stream; the final snapshot is always emitted regardless.
const UPDATE_THROTTLE: Duration = Duration::from_millis(100);

pub struct Bash {
    runner: Arc<dyn CommandRunner>,
}

impl Bash {
    /// A `bash` tool that runs commands for real.
    pub fn real() -> Self {
        Self {
            runner: Arc::new(RealRunner),
        }
    }

    /// A `bash` tool over a custom runner (tests inject one to capture the invocation).
    #[cfg(test)]
    pub fn with_runner(runner: Arc<dyn CommandRunner>) -> Self {
        Self { runner }
    }

    /// Shared body for [`Tool::run`] and [`Tool::run_streaming`]. Raw output feeds one accumulator; when
    /// `progress` is set, snapshots stream live. The returned value is the cleaned, tail-truncated
    /// output with pi's `Full output:` marker; a non-zero exit or timeout is an error carrying it.
    async fn exec(
        &self,
        input: Value,
        progress: Option<&ToolProgress>,
    ) -> Result<ToolOutput, ToolError> {
        let command = input
            .get("command")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidInput("missing `command`".into()))?;
        let cwd = input.get("cwd").and_then(Value::as_str);
        let timeout_ms = input
            .get("timeout_ms")
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_TIMEOUT_MS);
        let args = vec!["-c".to_string(), command.to_string()];
        let dur = Duration::from_millis(timeout_ms);

        let acc = Arc::new(Mutex::new(OutputAccumulator::new()));
        let streamed = Arc::new(AtomicBool::new(false));
        let last_emit = Arc::new(Mutex::new(Instant::now()));

        // pi emits an initial empty update the moment execution starts.
        if let Some(p) = progress {
            p.emit("", None);
        }

        // Feed every raw chunk (both stdout and stderr, in arrival order) into the one accumulator, and
        // emit a throttled snapshot as it grows. Always via the streaming path so the temp-file spill
        // sees the *complete* output; a non-streaming runner (test double) delivers no chunks and is
        // handled by the fallback below.
        let result = {
            let acc = acc.clone();
            let streamed = streamed.clone();
            let last_emit = last_emit.clone();
            let sink = move |bytes: &[u8]| {
                streamed.store(true, Ordering::Relaxed);
                let mut a = lock(&acc);
                a.append(bytes);
                if let Some(p) = progress {
                    let mut le = lock(&last_emit);
                    if le.elapsed() >= UPDATE_THROTTLE {
                        *le = Instant::now();
                        drop(le);
                        let snap = a.snapshot(false);
                        emit_update(p, &snap);
                    }
                }
            };
            let sink: ChunkSink<'_> = &sink;
            self.runner.run_streaming("sh", &args, cwd, dur, sink).await
        }
        .map_err(|e| ToolError::Execution(format!("spawn failed: {e}")))?;

        // Fallback for a non-streaming runner (test double): feed its final captured output.
        if !streamed.load(Ordering::Relaxed) {
            let mut a = lock(&acc);
            a.append(result.stdout.as_bytes());
            if !result.stderr.is_empty() {
                if !result.stdout.is_empty() {
                    a.append(b"\n");
                }
                a.append(result.stderr.as_bytes());
            }
        }

        let snap = {
            let mut a = lock(&acc);
            a.finish();
            a.snapshot(true) // persist the full output to the temp file if truncated
        };

        // pi flushes a final snapshot before returning the result.
        if let Some(p) = progress {
            emit_update(p, &snap);
        }

        let text = clean(format_output(&snap, "(no output)"));

        if result.timed_out {
            let secs = timeout_ms / 1000;
            return Err(ToolError::Execution(append_status(
                &text,
                &format!("Command timed out after {secs} seconds"),
            )));
        }
        match result.code {
            Some(0) | None => Ok(text.into()),
            // Non-zero exit is an error result that still carries the output — pi throws here too.
            Some(code) => Err(ToolError::Execution(append_status(
                &text,
                &format!("Command exited with code {code}"),
            ))),
        }
    }
}

#[async_trait]
impl Tool for Bash {
    fn name(&self) -> &str {
        "bash"
    }
    fn description(&self) -> &str {
        "Run a shell command via `sh -c` and return its combined stdout/stderr. Supports an optional \
         `cwd` and `timeout_ms`."
    }
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "Shell command to run." },
                "cwd": { "type": "string", "description": "Working directory." },
                "timeout_ms": { "type": "integer", "description": "Timeout in milliseconds." }
            },
            "required": ["command"]
        })
    }

    async fn run(&self, input: Value) -> Result<ToolOutput, ToolError> {
        self.exec(input, None).await
    }

    async fn run_streaming(
        &self,
        input: Value,
        progress: &ToolProgress,
    ) -> Result<ToolOutput, ToolError> {
        self.exec(input, Some(progress)).await
    }
}

/// Lock a mutex, recovering the data on poison rather than panicking (the crate forbids `unwrap`).
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Emit one streamed progress snapshot: the cleaned output so far, plus truncation + full-output-path
/// details when the output has overflowed (pi's `onUpdate({content, details})`).
fn emit_update(p: &ToolProgress, snap: &OutputSnapshot) {
    let details = snap.truncation.truncated.then(|| {
        json!({
            "truncated": true,
            "total_lines": snap.truncation.total_lines,
            "total_bytes": snap.truncation.total_bytes,
            "output_lines": snap.truncation.output_lines,
            "output_bytes": snap.truncation.output_bytes,
            "full_output_path": snap.full_output_path,
        })
    });
    p.emit(clean(snap.content.clone()), details);
}

/// Append a status line (`exit code` / `timed out`) after the output, pi-style (`text\n\n<status>`).
fn append_status(text: &str, status: &str) -> String {
    if text.is_empty() {
        status.to_string()
    } else {
        format!("{text}\n\n{status}")
    }
}

/// ANSI escape + OSC sequences a terminal emits but the model can't use: CSI/SGR colour and cursor
/// moves (`ESC [ … final`) plus OSC strings (`ESC ] … BEL`). Built once; the pattern is a static
/// literal, so a build failure is impossible — we model it as `None` instead of unwrapping.
fn ansi_re() -> Option<&'static Regex> {
    static RE: OnceLock<Option<Regex>> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07").ok())
        .as_ref()
}

/// Whitespace the model actually needs (`\t`, `\n`, `\r`) survives; every other C0 control byte (NUL,
/// stray ESC, …) and the Unicode interlinear-annotation controls are dropped so binary output can't
/// corrupt the context.
fn is_keepable(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\r') || (c >= ' ' && !('\u{fff9}'..='\u{fffb}').contains(&c))
}

/// Strip terminal control noise from output. Two single passes: a regex removes ANSI escape sequences,
/// then we drop residual control bytes. When the text is already clean we reuse the string we have
/// rather than reallocating. (pi sanitizes binary output too; we also drop ANSI, which the model can't
/// use — a small, deliberate step past pi that only ever removes noise.)
fn clean(s: String) -> String {
    let stripped = match ansi_re() {
        Some(re) => re.replace_all(&s, ""),
        None => Cow::Borrowed(s.as_str()),
    };
    if stripped.chars().all(is_keepable) {
        return match stripped {
            Cow::Owned(owned) => owned,
            Cow::Borrowed(_) => s,
        };
    }
    stripped.chars().filter(|&c| is_keepable(c)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::exec::ExecResult;

    /// Records the last invocation and returns a canned result (a non-streaming runner: it never
    /// delivers chunks, so `bash` exercises its fallback-from-`ExecResult` path).
    struct RecordingRunner {
        last: std::sync::Mutex<Option<(String, Vec<String>)>>,
        result: ExecResult,
    }

    #[async_trait]
    impl CommandRunner for RecordingRunner {
        async fn run(
            &self,
            program: &str,
            args: &[String],
            _cwd: Option<&str>,
            _t: Duration,
        ) -> std::io::Result<ExecResult> {
            *self.last.lock().unwrap() = Some((program.to_string(), args.to_vec()));
            Ok(self.result.clone())
        }
    }

    fn recording(result: ExecResult) -> Arc<RecordingRunner> {
        Arc::new(RecordingRunner {
            last: std::sync::Mutex::new(None),
            result,
        })
    }

    #[tokio::test]
    async fn invokes_sh_dash_c() {
        let runner = recording(ExecResult {
            code: Some(0),
            stdout: "hi\n".into(),
            ..Default::default()
        });
        let out = Bash::with_runner(runner.clone())
            .run(json!({ "command": "echo hi" }))
            .await
            .unwrap()
            .text;
        // pi does not trim: the command's output is shown as-is (trailing newline kept).
        assert_eq!(out, "hi\n");
        let (prog, args) = runner.last.lock().unwrap().clone().unwrap();
        assert_eq!(prog, "sh");
        assert_eq!(args, vec!["-c".to_string(), "echo hi".to_string()]);
    }

    #[tokio::test]
    async fn nonzero_exit_is_an_error_carrying_the_output() {
        let runner = recording(ExecResult {
            code: Some(2),
            stderr: "boom".into(),
            ..Default::default()
        });
        let err = Bash::with_runner(runner)
            .run(json!({ "command": "false" }))
            .await
            .unwrap_err();
        let ToolError::Execution(msg) = err else {
            panic!("expected Execution error")
        };
        assert!(msg.contains("boom"));
        assert!(msg.contains("Command exited with code 2"));
    }

    #[tokio::test]
    async fn real_runner_executes() {
        let out = Bash::real()
            .run(json!({ "command": "printf done" }))
            .await
            .unwrap()
            .text;
        assert_eq!(out, "done");
    }

    #[tokio::test]
    async fn strips_ansi_escape_sequences() {
        let runner = recording(ExecResult {
            code: Some(0),
            stdout: "\x1b[31mred\x1b[0m \x1b[2Aup\x1b]0;title\x07 done".into(),
            ..Default::default()
        });
        let out = Bash::with_runner(runner)
            .run(json!({ "command": "x" }))
            .await
            .unwrap()
            .text;
        assert_eq!(out, "red up done");
    }

    #[tokio::test]
    async fn sanitizes_control_characters() {
        let runner = recording(ExecResult {
            code: Some(0),
            stdout: "a\x00b\x07c\td\ne".into(),
            ..Default::default()
        });
        let out = Bash::with_runner(runner)
            .run(json!({ "command": "x" }))
            .await
            .unwrap()
            .text;
        assert_eq!(out, "abc\td\ne");
    }

    #[tokio::test]
    async fn large_output_keeps_the_tail_and_points_at_the_full_file() {
        // 5000 lines exceed the 2000-line cap: the accumulator keeps the tail, reports the range, and
        // spills the complete output to a temp file the marker names (pi's `Full output:`).
        let body = (0..5000)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let runner = recording(ExecResult {
            code: Some(0),
            stdout: body,
            ..Default::default()
        });
        let out = Bash::with_runner(runner)
            .run(json!({ "command": "x" }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("[Showing lines"), "range marker present");
        assert!(out.contains("Full output:"), "full-output path present");
        assert!(out.contains("4999"), "tail kept (the last line is present)");
        assert!(
            !out.starts_with("0\n1\n"),
            "head dropped (tail truncation, not head+tail)"
        );
    }

    #[tokio::test]
    async fn timeout_is_an_error() {
        let runner = recording(ExecResult {
            timed_out: true,
            ..Default::default()
        });
        let err = Bash::with_runner(runner)
            .run(json!({ "command": "sleep 10", "timeout_ms": 5000 }))
            .await
            .unwrap_err();
        let ToolError::Execution(msg) = err else {
            panic!("expected Execution error")
        };
        assert!(msg.contains("timed out"));
    }
}
