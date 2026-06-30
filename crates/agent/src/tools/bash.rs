//! `bash` — run a shell command via `sh -c` and return its combined output.

use std::borrow::Cow;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use agent_core::tool::Tool;
use agent_core::{ToolError, ToolOutput, ToolProgress};
use async_trait::async_trait;
use regex::Regex;
use serde_json::{Value, json};

use super::exec::{ChunkSink, CommandRunner, RealRunner};

/// Default command timeout (ms) when the model doesn't specify one.
const DEFAULT_TIMEOUT_MS: u64 = 120_000;
/// Cap on returned output bytes (head + tail kept) to protect the model's context.
const MAX_OUTPUT: usize = 30_000;
/// Cap on returned output lines (head + tail kept). Pairs with the byte cap — whichever bites first
/// wins — so a flood of short lines can't bury the model even when it stays under `MAX_OUTPUT`.
const MAX_LINES: usize = 2000;

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

    /// Shared body for [`Tool::run`] and [`Tool::run_streaming`]. When `on_chunk` is set, raw output
    /// is streamed through it as it arrives; either way the returned value is the fully cleaned +
    /// capped output.
    async fn exec(
        &self,
        input: Value,
        on_chunk: Option<ChunkSink<'_>>,
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
        let result = match on_chunk {
            Some(sink) => self.runner.run_streaming("sh", &args, cwd, dur, sink).await,
            None => self.runner.run("sh", &args, cwd, dur).await,
        }
        .map_err(|e| ToolError::Execution(format!("spawn failed: {e}")))?;

        if result.timed_out {
            return Err(ToolError::Execution(format!(
                "command timed out after {timeout_ms}ms"
            )));
        }

        let mut out = String::new();
        out.push_str(result.stdout.trim_end());
        if !result.stderr.trim().is_empty() {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(result.stderr.trim_end());
        }
        match result.code {
            Some(0) => {}
            Some(code) => out.push_str(&format!("\n[exit code {code}]")),
            None => out.push_str("\n[killed]"),
        }
        // Hygiene first (strip terminal control noise so binary/coloured output can't corrupt the
        // model's context), then the line cap, then the byte cap — so each cap measures clean text.
        // `result.truncated` is the exec layer's signal that capture already dropped the middle in
        // memory; we thread it through so the marker can say the elision happened at the source.
        Ok(truncate(clean(out), result.truncated).into())
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
        // Forward each raw output chunk as a progress event the moment it arrives; the returned
        // result is still the fully cleaned + capped output.
        let sink = move |bytes: &[u8]| progress.emit(String::from_utf8_lossy(bytes));
        self.exec(input, Some(&sink)).await
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

/// Strip terminal control noise from combined output. Two single passes: a regex removes ANSI escape
/// sequences, then we drop residual control bytes. When the text is already clean we reuse the string
/// we have rather than reallocating.
fn clean(s: String) -> String {
    let stripped = match ansi_re() {
        Some(re) => re.replace_all(&s, ""),
        None => Cow::Borrowed(s.as_str()),
    };
    if stripped.chars().all(is_keepable) {
        return match stripped {
            // Regex rewrote the text (matches found) — keep its owned buffer.
            Cow::Owned(owned) => owned,
            // No ANSI, no control bytes: the original is already what we want.
            Cow::Borrowed(_) => s,
        };
    }
    stripped.chars().filter(|&c| is_keepable(c)).collect()
}

/// Apply the line cap then the byte cap, surfacing `source_truncated` exactly once. Both caps keep the
/// head and tail — the middle is least useful to the model — mirroring each other so output stays
/// readable whichever cap bites.
fn truncate(s: String, source_truncated: bool) -> String {
    let s = cap_lines(s);
    let (s, byte_capped) = cap_bytes(s, source_truncated);
    // If capture dropped the middle in memory but neither display cap fired, the model would see no
    // sign of the elision — append the marker so the source truncation is always surfaced.
    if source_truncated && !byte_capped {
        format!("{s}\n… (output truncated at source) …")
    } else {
        s
    }
}

/// Keep the first and last `MAX_LINES / 2` lines; the marker reports how many of the middle lines were
/// hidden. The cheap newline count avoids splitting the common (uncapped) case.
fn cap_lines(s: String) -> String {
    // `count` is newlines, so lines == count + 1; `count < MAX_LINES` is the same uncapped test.
    if s.bytes().filter(|&b| b == b'\n').count() < MAX_LINES {
        return s;
    }
    let half = MAX_LINES / 2;
    let lines: Vec<&str> = s.split('\n').collect();
    let hidden = lines.len() - half * 2;
    format!(
        "{}\n… ({hidden} lines hidden) …\n{}",
        lines[..half].join("\n"),
        lines[lines.len() - half..].join("\n")
    )
}

/// Keep the head and tail of oversized output. The `half` budget is in bytes (matching the
/// `MAX_OUTPUT` byte check), snapped to char boundaries so we never slice mid-codepoint — one pass per
/// end, no intermediate allocations. Returns whether the cap fired so `truncate` can decide if the
/// source-truncation marker still needs appending.
fn cap_bytes(s: String, source_truncated: bool) -> (String, bool) {
    if s.len() <= MAX_OUTPUT {
        return (s, false);
    }
    let half = MAX_OUTPUT / 2;
    // Head: largest char boundary <= half.
    let mut head_end = half.min(s.len());
    while !s.is_char_boundary(head_end) {
        head_end -= 1;
    }
    // Tail: smallest char boundary >= len - half.
    let mut tail_start = s.len() - half;
    while !s.is_char_boundary(tail_start) {
        tail_start += 1;
    }
    // The marker tells the model whether the middle was dropped at the capture source or just by our
    // display cap.
    let marker = if source_truncated {
        "… (output truncated at source) …"
    } else {
        "… (output truncated) …"
    };
    (
        format!("{}\n{marker}\n{}", &s[..head_end], &s[tail_start..]),
        true,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::exec::ExecResult;

    /// Records the last invocation and returns a canned result.
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

    #[tokio::test]
    async fn invokes_sh_dash_c() {
        let runner = Arc::new(RecordingRunner {
            last: std::sync::Mutex::new(None),
            result: ExecResult {
                code: Some(0),
                stdout: "hi\n".into(),
                stderr: String::new(),
                timed_out: false,
                ..Default::default()
            },
        });
        let bash = Bash::with_runner(runner.clone());
        let out = bash
            .run(json!({ "command": "echo hi" }))
            .await
            .unwrap()
            .text;
        assert_eq!(out, "hi");
        let (prog, args) = runner.last.lock().unwrap().clone().unwrap();
        assert_eq!(prog, "sh");
        assert_eq!(args, vec!["-c".to_string(), "echo hi".to_string()]);
    }

    #[tokio::test]
    async fn appends_nonzero_exit_code() {
        let runner = Arc::new(RecordingRunner {
            last: std::sync::Mutex::new(None),
            result: ExecResult {
                code: Some(2),
                stdout: String::new(),
                stderr: "boom".into(),
                timed_out: false,
                ..Default::default()
            },
        });
        let out = Bash::with_runner(runner)
            .run(json!({ "command": "false" }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("boom"));
        assert!(out.contains("[exit code 2]"));
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
        // SGR colour codes, a cursor move, and an OSC title set should all vanish, leaving the text.
        let runner = Arc::new(RecordingRunner {
            last: std::sync::Mutex::new(None),
            result: ExecResult {
                code: Some(0),
                stdout: "\x1b[31mred\x1b[0m \x1b[2Aup\x1b]0;title\x07 done".into(),
                stderr: String::new(),
                timed_out: false,
                ..Default::default()
            },
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
        // NUL and other C0 controls are dropped; tab/newline survive as the only allowed whitespace.
        let runner = Arc::new(RecordingRunner {
            last: std::sync::Mutex::new(None),
            result: ExecResult {
                code: Some(0),
                stdout: "a\x00b\x07c\td\ne".into(),
                stderr: String::new(),
                timed_out: false,
                ..Default::default()
            },
        });
        let out = Bash::with_runner(runner)
            .run(json!({ "command": "x" }))
            .await
            .unwrap()
            .text;
        assert_eq!(out, "abc\td\ne");
    }

    #[tokio::test]
    async fn caps_lines_with_hidden_count_marker() {
        // 5000 short lines exceed the 2000-line cap but stay under the byte cap, so only the line cap
        // fires: head + tail kept, marker reports the 3000 hidden middle lines.
        let body = (0..5000)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let runner = Arc::new(RecordingRunner {
            last: std::sync::Mutex::new(None),
            result: ExecResult {
                code: Some(0),
                stdout: body,
                stderr: String::new(),
                timed_out: false,
                ..Default::default()
            },
        });
        let out = Bash::with_runner(runner)
            .run(json!({ "command": "x" }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("… (3000 lines hidden) …"));
        assert!(out.starts_with("0\n1\n"));
        assert!(out.trim_end().ends_with("4999"));
        assert!(out.lines().count() <= MAX_LINES + 1);
    }

    #[tokio::test]
    async fn surfaces_source_truncation_in_marker() {
        // Capture dropped the middle in memory; even though the final text is tiny (no display cap
        // fires), the marker must tell the model the elision happened at the source.
        let runner = Arc::new(RecordingRunner {
            last: std::sync::Mutex::new(None),
            result: ExecResult {
                code: Some(0),
                stdout: "head … tail".into(),
                stderr: String::new(),
                timed_out: false,
                truncated: true,
            },
        });
        let out = Bash::with_runner(runner)
            .run(json!({ "command": "x" }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("… (output truncated at source) …"));
    }

    #[tokio::test]
    async fn timeout_is_reported() {
        let runner = Arc::new(RecordingRunner {
            last: std::sync::Mutex::new(None),
            result: ExecResult {
                code: None,
                stdout: String::new(),
                stderr: String::new(),
                timed_out: true,
                ..Default::default()
            },
        });
        let err = Bash::with_runner(runner)
            .run(json!({ "command": "sleep 10", "timeout_ms": 5 }))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Execution(_)));
    }
}
