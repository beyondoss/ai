//! `bash` — run a shell command via `sh -c` and return its combined output.

use std::sync::Arc;
use std::time::Duration;

use agent_core::ToolError;
use agent_core::tool::Tool;
use async_trait::async_trait;
use serde_json::{Value, json};

use super::exec::{CommandRunner, RealRunner};

/// Default command timeout (ms) when the model doesn't specify one.
const DEFAULT_TIMEOUT_MS: u64 = 120_000;
/// Cap on returned output bytes (head + tail kept) to protect the model's context.
const MAX_OUTPUT: usize = 30_000;

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

    async fn run(&self, input: Value) -> Result<String, ToolError> {
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
        let result = self
            .runner
            .run("sh", &args, cwd, Duration::from_millis(timeout_ms))
            .await
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
        Ok(truncate(out))
    }
}

/// Keep the head and tail of oversized output; the middle is what's least useful to the model.
fn truncate(s: String) -> String {
    if s.len() <= MAX_OUTPUT {
        return s;
    }
    let half = MAX_OUTPUT / 2;
    let head: String = s.chars().take(half).collect();
    let tail: String = s
        .chars()
        .rev()
        .take(half)
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    format!("{head}\n… (output truncated) …\n{tail}")
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
            },
        });
        let bash = Bash::with_runner(runner.clone());
        let out = bash.run(json!({ "command": "echo hi" })).await.unwrap();
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
            },
        });
        let out = Bash::with_runner(runner)
            .run(json!({ "command": "false" }))
            .await
            .unwrap();
        assert!(out.contains("boom"));
        assert!(out.contains("[exit code 2]"));
    }

    #[tokio::test]
    async fn real_runner_executes() {
        let out = Bash::real()
            .run(json!({ "command": "printf done" }))
            .await
            .unwrap();
        assert_eq!(out, "done");
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
            },
        });
        let err = Bash::with_runner(runner)
            .run(json!({ "command": "sleep 10", "timeout_ms": 5 }))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Execution(_)));
    }
}
