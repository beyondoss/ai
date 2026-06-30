//! Beyond platform tools — the agent's hooks into the platform from `../beyond/PLATFORM.md`.
//!
//! `fork` clones a running app (its real data) into an isolated branch, `sync` pushes the working
//! tree to a running instance, and `logs` queries instance logs. Each shells out to the `beyond` CLI
//! through the shared [`CommandRunner`] seam, so tests assert the exact argv without a control plane.
//!
//! **Scope boundary:** the `beyond` CLI lives in the `beyond` repo, not here. These tools are
//! verified at the argv level; real fork/sync/logs behavior is validated against a live control
//! plane, which isn't available in this repo's tests.

use std::sync::Arc;
use std::time::Duration;

use agent_core::ToolError;
use agent_core::tool::Tool;
use async_trait::async_trait;
use serde_json::{Value, json};

use super::exec::{CommandRunner, RealRunner};

/// Wall-clock ceiling for a `beyond` invocation.
const TIMEOUT: Duration = Duration::from_secs(120);

/// Run `beyond <args…>` through the runner, mapping the result to a tool output/error.
async fn run_beyond(
    runner: &Arc<dyn CommandRunner>,
    args: Vec<String>,
) -> Result<String, ToolError> {
    let res = runner
        .run("beyond", &args, None, TIMEOUT)
        .await
        .map_err(|e| ToolError::Execution(format!("spawn `beyond`: {e}")))?;
    if res.timed_out {
        return Err(ToolError::Execution("`beyond` timed out".into()));
    }
    let mut out = res.stdout.trim_end().to_string();
    if !res.stderr.trim().is_empty() {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(res.stderr.trim_end());
    }
    match res.code {
        Some(0) => Ok(out),
        Some(code) => Err(ToolError::Execution(format!(
            "`beyond` exited {code}: {out}"
        ))),
        None => Err(ToolError::Execution("`beyond` was killed".into())),
    }
}

macro_rules! runner_ctor {
    () => {
        /// A tool that runs the real `beyond` CLI.
        pub fn real() -> Self {
            Self {
                runner: Arc::new(RealRunner),
            }
        }
        /// A tool over a custom runner (tests inject one to capture the argv).
        #[cfg(test)]
        pub fn with_runner(runner: Arc<dyn CommandRunner>) -> Self {
            Self { runner }
        }
    };
}

/// `fork` — branch a Beyond app, getting an isolated copy of its real data (the agent sandbox).
pub struct Fork {
    runner: Arc<dyn CommandRunner>,
}
impl Fork {
    runner_ctor!();
}

#[async_trait]
impl Tool for Fork {
    fn name(&self) -> &str {
        "fork"
    }
    fn description(&self) -> &str {
        "Fork a Beyond app into an isolated branch with a copy of its real data — a sandbox to make \
         and verify changes before promoting or destroying."
    }
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "app": { "type": "string", "description": "App to fork." },
                "name": { "type": "string", "description": "Optional name for the branch." }
            },
            "required": ["app"]
        })
    }
    async fn run(&self, input: Value) -> Result<String, ToolError> {
        let app = input
            .get("app")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidInput("missing `app`".into()))?;
        let mut args = vec!["fork".to_string(), app.to_string()];
        if let Some(name) = input.get("name").and_then(Value::as_str) {
            args.push("--name".into());
            args.push(name.to_string());
        }
        run_beyond(&self.runner, args).await
    }
}

/// `sync` — push the working tree to the running instance (the tight edit→run loop).
pub struct Sync {
    runner: Arc<dyn CommandRunner>,
}
impl Sync {
    runner_ctor!();
}

#[async_trait]
impl Tool for Sync {
    fn name(&self) -> &str {
        "sync"
    }
    fn description(&self) -> &str {
        "Push the working tree to the running Beyond instance so changes take effect immediately."
    }
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Directory to sync (default: project root)." }
            }
        })
    }
    async fn run(&self, input: Value) -> Result<String, ToolError> {
        let mut args = vec!["sync".to_string()];
        if let Some(path) = input.get("path").and_then(Value::as_str) {
            args.push(path.to_string());
        }
        run_beyond(&self.runner, args).await
    }
}

/// `logs` — query a Beyond instance's logs (optionally a Lucene query).
pub struct Logs {
    runner: Arc<dyn CommandRunner>,
}
impl Logs {
    runner_ctor!();
}

#[async_trait]
impl Tool for Logs {
    fn name(&self) -> &str {
        "logs"
    }
    fn description(&self) -> &str {
        "Read or search a Beyond instance's logs. Optionally filter by `app` and a Lucene `query`."
    }
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "app": { "type": "string", "description": "App whose logs to read." },
                "query": { "type": "string", "description": "Lucene query to filter log lines." }
            }
        })
    }
    async fn run(&self, input: Value) -> Result<String, ToolError> {
        let mut args = vec!["logs".to_string()];
        if let Some(app) = input.get("app").and_then(Value::as_str) {
            args.push("--app".into());
            args.push(app.to_string());
        }
        if let Some(query) = input.get("query").and_then(Value::as_str) {
            args.push("--query".into());
            args.push(query.to_string());
        }
        run_beyond(&self.runner, args).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::exec::ExecResult;

    struct RecordingRunner {
        last: std::sync::Mutex<Option<(String, Vec<String>)>>,
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
            Ok(ExecResult {
                code: Some(0),
                stdout: "ok".into(),
                stderr: String::new(),
                timed_out: false,
            })
        }
    }

    fn recorder() -> Arc<RecordingRunner> {
        Arc::new(RecordingRunner {
            last: std::sync::Mutex::new(None),
        })
    }

    #[tokio::test]
    async fn fork_builds_argv() {
        let r = recorder();
        let out = Fork::with_runner(r.clone())
            .run(json!({ "app": "myapp", "name": "branch1" }))
            .await
            .unwrap();
        assert_eq!(out, "ok");
        let (prog, args) = r.last.lock().unwrap().clone().unwrap();
        assert_eq!(prog, "beyond");
        assert_eq!(args, vec!["fork", "myapp", "--name", "branch1"]);
    }

    #[tokio::test]
    async fn sync_builds_argv() {
        let r = recorder();
        Sync::with_runner(r.clone())
            .run(json!({ "path": "./web" }))
            .await
            .unwrap();
        assert_eq!(
            r.last.lock().unwrap().clone().unwrap().1,
            vec!["sync", "./web"]
        );
    }

    #[tokio::test]
    async fn logs_builds_argv_with_query() {
        let r = recorder();
        Logs::with_runner(r.clone())
            .run(json!({ "app": "api", "query": "level:error" }))
            .await
            .unwrap();
        assert_eq!(
            r.last.lock().unwrap().clone().unwrap().1,
            vec!["logs", "--app", "api", "--query", "level:error"]
        );
    }

    #[tokio::test]
    async fn nonzero_exit_is_error() {
        struct Fail;
        #[async_trait]
        impl CommandRunner for Fail {
            async fn run(
                &self,
                _p: &str,
                _a: &[String],
                _c: Option<&str>,
                _t: Duration,
            ) -> std::io::Result<ExecResult> {
                Ok(ExecResult {
                    code: Some(1),
                    stdout: String::new(),
                    stderr: "no such app".into(),
                    timed_out: false,
                })
            }
        }
        let err = Fork::with_runner(Arc::new(Fail))
            .run(json!({ "app": "ghost" }))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Execution(_)));
    }
}
