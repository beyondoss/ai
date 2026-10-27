//! `bash` and `bash_job` pointed at a remote target, the way the registry pairs them: commands go to
//! the runner, the filesystem (and so the [`PathWorld`]) is the target's.
//!
//! The capability probe is skipped (`Capabilities::default()`) so a suite's request log holds only
//! what `bash` itself sent — the stream probe and the commands.

use std::sync::Arc;
use std::time::Duration;

use agent_core::tool::Tool;
use beyond_agent::tools::bash::{Bash, BashJob};
use beyond_agent::tools::exec::CommandRunner;
use beyond_agent::tools::fs::FsBackend;
use beyond_agent::tools::fs::shell::{Capabilities, ShellFs};
use serde_json::{Value, json};

/// A remote `bash` over `runner` that yields after `idle` of silence, and the `bash_job` that
/// shares its job table.
pub fn remote_bash(runner: Arc<dyn CommandRunner>, idle: Duration) -> (Arc<Bash>, BashJob) {
    let backend: Arc<dyn FsBackend> = Arc::new(ShellFs::with_capabilities(
        runner.clone(),
        Capabilities::default(),
    ));
    let bash = Arc::new(
        Bash::with_runner(runner)
            .with_backend(backend)
            .with_idle_yield(Some(idle)),
    );
    let job = BashJob::new(bash.clone());
    (bash, job)
}

/// Run a tool, flattening its result to text either way.
pub async fn call(tool: &dyn Tool, input: Value) -> Result<String, String> {
    tool.run(input)
        .await
        .map(|o| o.text)
        .map_err(|e| e.to_string())
}

/// `bash_job` kill of job `n`.
pub async fn kill_job(job: &BashJob, n: u64) -> Result<String, String> {
    call(job, json!({ "job": n, "kill": true })).await
}

/// The job number a yielded `bash` result names ("... as job 3: ...").
pub fn job_number(text: &str) -> Option<u64> {
    let at = text.find("as job ")? + "as job ".len();
    text[at..]
        .split(|c: char| !c.is_ascii_digit())
        .next()?
        .parse()
        .ok()
}
