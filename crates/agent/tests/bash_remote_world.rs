//! `bash` when its commands land somewhere other than this host.
//!
//! `bash` is the tool with the most host-shaped assumptions baked into it, and each one fails
//! silently rather than loudly once the target is a sandbox: a `cwd` validated with a host `is_dir`
//! rejects every perfectly good sandbox path, a shell resolved from this machine's `/bin` and `$PATH`
//! hands an Alpine target a binary it does not have, and a spill file written to this machine's temp
//! dir is a path the model can never read — on a shared replica, one tenant's command output left on
//! the host besides.
//!
//! Each test makes the host and the target disagree and asserts the target won. The local half of
//! every behavior is asserted alongside it, because "keep local byte-identical" is the other half of
//! the contract and is exactly what a world check is easy to break.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use beyond_ai_agent::tools::exec::{CommandRunner, ExecResult};
use beyond_ai_agent::tools::fs::FsBackend;
use beyond_ai_agent::tools::fs::shell::{Capabilities, ShellFs};
use beyond_ai_agent::tools::{ToolConfig, default_registry_with_config};
use serde_json::{Value, json};

/// One command the tool issued.
#[derive(Clone, Debug)]
struct Call {
    program: String,
    args: Vec<String>,
    cwd: Option<String>,
}

/// A `CommandRunner` that records instead of running. Canned output, because what is under test is
/// the *decision* — which shell, which directory — not what a command would print.
struct Recorder {
    calls: Mutex<Vec<Call>>,
    stdout: String,
}

impl Recorder {
    fn new() -> Arc<Self> {
        Self::emitting(String::new())
    }

    fn emitting(stdout: String) -> Arc<Self> {
        Arc::new(Self {
            calls: Mutex::new(Vec::new()),
            stdout,
        })
    }

    fn calls(&self) -> Vec<Call> {
        self.calls.lock().unwrap().clone()
    }

    /// The one invocation of the model's own command — the last one, since a filesystem probe would
    /// come first.
    fn shell_call(&self) -> Call {
        self.calls()
            .into_iter()
            .find(|c| c.args.first().is_some_and(|a| a == "-c"))
            .expect("`bash` must have invoked a shell")
    }
}

#[async_trait]
impl CommandRunner for Recorder {
    async fn run(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        _timeout: Duration,
    ) -> std::io::Result<ExecResult> {
        self.calls.lock().unwrap().push(Call {
            program: program.to_string(),
            args: args.to_vec(),
            cwd: cwd.map(str::to_string),
        });
        Ok(ExecResult {
            code: Some(0),
            signal: None,
            stdout: self.stdout.clone(),
            stderr: String::new(),
            timed_out: false,
            truncated: false,
        })
    }
}

/// `bash` pointed at `runner`, with the filesystem tools on the same (remote) target — the pairing
/// `default_registry_with_config` exists to guarantee, and the only configuration in which `bash`
/// reports a remote world.
fn remote_registry(
    runner: Arc<Recorder>,
    root: &str,
    remote_shell: Option<&str>,
) -> agent_core::ToolRegistry {
    let backend: Arc<dyn FsBackend> = Arc::new(
        ShellFs::with_capabilities(runner.clone(), Capabilities::default()).with_home("/sbx"),
    );
    default_registry_with_config(&ToolConfig {
        root: root.into(),
        remote_shell,
        fs_backend: Some(backend),
        command_runner: Some(runner),
        ..ToolConfig::new()
    })
}

/// `bash` over `runner` with **no** filesystem backend: commands go to the runner, but the world is
/// this host, exactly as every existing caller has it.
fn local_registry(runner: Arc<Recorder>, root: &str) -> agent_core::ToolRegistry {
    default_registry_with_config(&ToolConfig {
        root: root.into(),
        command_runner: Some(runner),
        ..ToolConfig::new()
    })
}

async fn run_bash(reg: &agent_core::ToolRegistry, input: Value) -> Result<String, String> {
    match reg.get("bash").expect("bash registered").run(input).await {
        Ok(out) => Ok(out.text),
        Err(e) => Err(e.to_string()),
    }
}

// ---- the working directory ---------------------------------------------------------------------

#[tokio::test]
async fn a_relative_cwd_resolves_under_the_tool_root_not_this_process_cwd() {
    let runner = Recorder::new();
    let reg = remote_registry(runner.clone(), "/workspace", None);
    run_bash(&reg, json!({ "command": "true", "cwd": "sub" }))
        .await
        .expect("a relative cwd is not an error");

    let call = runner.shell_call();
    assert_eq!(call.cwd.as_deref(), Some("/workspace/sub"));
    let here = std::env::current_dir().unwrap().display().to_string();
    assert!(
        !call.cwd.as_deref().unwrap_or_default().starts_with(&here),
        "the replica's cwd leaked into a sandbox path: {call:?}"
    );
}

#[tokio::test]
async fn an_omitted_cwd_is_the_tool_root_in_both_worlds() {
    let runner = Recorder::new();
    let reg = remote_registry(runner.clone(), "/workspace", None);
    run_bash(&reg, json!({ "command": "true" })).await.unwrap();
    assert_eq!(runner.shell_call().cwd.as_deref(), Some("/workspace"));

    // Unchanged locally: an empty root still means "inherit this process's cwd", which is `None`.
    let local = Recorder::new();
    let reg = local_registry(local.clone(), "");
    run_bash(&reg, json!({ "command": "true" })).await.unwrap();
    assert_eq!(local.shell_call().cwd, None);
}

#[tokio::test]
async fn a_cwd_that_exists_only_on_the_target_is_not_rejected() {
    // The host check's failure mode, exactly: `/sbx/proj` is a real directory in the sandbox and
    // absent here, so `is_dir()` answers `false` and the command never runs.
    assert!(
        !std::path::Path::new("/sbx/proj").exists(),
        "this test needs a path that does not exist on this host"
    );
    let runner = Recorder::new();
    let reg = remote_registry(runner.clone(), "", None);
    run_bash(&reg, json!({ "command": "true", "cwd": "/sbx/proj" }))
        .await
        .expect("a sandbox-only cwd must reach the target, not be pre-rejected here");
    assert_eq!(runner.shell_call().cwd.as_deref(), Some("/sbx/proj"));
}

#[tokio::test]
async fn a_missing_cwd_is_still_rejected_up_front_on_this_host() {
    // The other half: locally the pre-check is a *better* error than the raw spawn failure, and both
    // of its two messages stay exactly as they were.
    let runner = Recorder::new();
    let reg = local_registry(runner.clone(), "");
    let err = run_bash(&reg, json!({ "command": "true", "cwd": "/sbx/proj" }))
        .await
        .expect_err("a missing local cwd is an error");
    assert!(
        err.contains("Working directory does not exist: /sbx/proj")
            && err.contains("Cannot execute bash commands."),
        "got: {err}"
    );
    assert!(
        runner.calls().is_empty(),
        "the command must not have been dispatched"
    );

    let file = tempfile::NamedTempFile::new().unwrap();
    let err = run_bash(
        &reg,
        json!({ "command": "true", "cwd": file.path().to_str().unwrap() }),
    )
    .await
    .expect_err("a cwd that is a file is an error");
    assert!(
        err.contains("Working directory is not a directory:"),
        "got: {err}"
    );
}

// ---- which shell ------------------------------------------------------------------------------

#[tokio::test]
async fn a_remote_command_runs_through_the_targets_shell_not_this_hosts() {
    let runner = Recorder::new();
    let reg = remote_registry(runner.clone(), "", None);
    run_bash(&reg, json!({ "command": "true" })).await.unwrap();
    assert_eq!(
        runner.shell_call().program,
        "sh",
        "with no probed shell the fallback must be the `sh` every POSIX target has — never a \
         `/bin/bash` path that only exists here"
    );

    let runner = Recorder::new();
    let reg = remote_registry(runner.clone(), "", Some("/bin/bash"));
    run_bash(&reg, json!({ "command": "true" })).await.unwrap();
    assert_eq!(runner.shell_call().program, "/bin/bash");
}

#[tokio::test]
async fn an_explicit_shell_path_outranks_both_worlds() {
    let runner = Recorder::new();
    let backend: Arc<dyn FsBackend> = Arc::new(ShellFs::with_capabilities(
        runner.clone(),
        Capabilities::default(),
    ));
    let reg = default_registry_with_config(&ToolConfig {
        bash_shell_path: Some("/opt/hardened/sh"),
        remote_shell: Some("sh"),
        fs_backend: Some(backend),
        command_runner: Some(runner.clone()),
        ..ToolConfig::new()
    });
    run_bash(&reg, json!({ "command": "true" })).await.unwrap();
    assert_eq!(runner.shell_call().program, "/opt/hardened/sh");
}

#[tokio::test]
async fn a_local_command_still_resolves_this_hosts_shell() {
    // The local resolution order is unchanged: `/bin/bash` when this host has one.
    let runner = Recorder::new();
    let reg = local_registry(runner.clone(), "");
    run_bash(&reg, json!({ "command": "true" })).await.unwrap();
    let program = runner.shell_call().program;
    if std::path::Path::new("/bin/bash").exists() {
        assert_eq!(program, "/bin/bash");
    } else {
        assert!(
            program.ends_with("bash") || program == "sh",
            "got: {program}"
        );
    }
}

// ---- oversized output ---------------------------------------------------------------------------

/// Enough output to blow past the 50 KiB display budget and trigger the spill decision.
fn firehose() -> String {
    "a line of remote command output that is long enough to add up\n".repeat(2_000)
}

#[tokio::test]
async fn oversized_remote_output_names_no_host_file_and_says_what_to_do_instead() {
    let runner = Recorder::emitting(firehose());
    let reg = remote_registry(runner, "", None);
    let out = run_bash(&reg, json!({ "command": "cat big.log" }))
        .await
        .unwrap();

    assert!(
        out.contains("Full output not saved — re-run redirecting it to a file you can read"),
        "the truncation marker must be actionable from inside the sandbox:\n{}",
        tail(&out)
    );
    assert!(
        !out.contains("Full output: "),
        "a path on the replica is one the model can never read:\n{}",
        tail(&out)
    );
    assert!(
        !out.contains(&std::env::temp_dir().display().to_string()),
        "no host temp path may appear in a remote command's result:\n{}",
        tail(&out)
    );
}

#[tokio::test]
async fn oversized_local_output_still_spills_to_a_readable_file() {
    // The behavior that must not regress: on this host the spill file is the whole point, and the
    // marker still names a path that really exists and really holds the full stream.
    let runner = Recorder::emitting(firehose());
    let reg = local_registry(runner, "");
    let out = run_bash(&reg, json!({ "command": "cat big.log" }))
        .await
        .unwrap();

    let marker = out
        .lines()
        .last()
        .expect("a truncated result ends with the marker");
    let path = marker
        .split("Full output: ")
        .nth(1)
        .map(|rest| rest.trim_end_matches(']'))
        .expect("the local marker names a path");
    let spilled = std::fs::read_to_string(path).expect("the spill file exists and is readable");
    assert_eq!(spilled.len(), firehose().len());
    let _ = std::fs::remove_file(path);
}

/// The last few lines, which is where every marker lives — printing a 120 KB assertion failure helps
/// nobody.
fn tail(out: &str) -> String {
    out.lines()
        .rev()
        .take(4)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("\n")
}
