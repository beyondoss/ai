//! Running the agent's tools against a sandbox VM.
//!
//! This is the whole of the "remote" half, and it is deliberately tiny: a
//! [`CommandRunner`](crate::tools::exec::CommandRunner) that executes inside a Firecracker instance
//! instead of on this host. Everything above it — the six filesystem tools, the shell translations in
//! [`ShellFs`](crate::tools::fs::shell::ShellFs), the path-world handling — is unchanged and already
//! tested. Pointing the toolset at a VM is *only* a matter of substituting the runner.
//!
//! That is the payoff of the [`FsBackend`](crate::tools::fs::FsBackend) seam. The remote support here
//! is ~150 lines because the seam did the work; without it, every filesystem tool would have needed
//! its own remote path.
//!
//! ## Why the CLI rather than the framed socket
//!
//! `instd` exposes exec three ways: a loopback admin socket speaking a framed
//! `[len][type][msgpack]` protocol on `127.0.0.1:9445`, an mTLS gateway, and the `instd instance exec`
//! CLI (which is itself a thin client of the first). This uses the CLI.
//!
//! The framed client would save a process spawn per call (~5 ms against a ~40 ms round trip) and is
//! the right destination. But it is ~300 lines of protocol code whose only advantage is latency that
//! [the cost measurement](../../tests/fs_backend_cost.rs) already showed is not the constraint — a
//! 6-call turn issues 14 backend operations, so the spawn overhead is well under a tenth of a second
//! against a multi-second turn. Shipping the CLI-backed runner first means the *whole stack* is
//! exercised against real VMs now, and swapping in a framed client later changes one struct behind
//! this same trait with the test suite already in place to catch a regression.

use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;

use crate::tools::exec::{ChunkSink, CommandRunner, ExecResult};

/// `instd instance exec`'s JSON envelope.
#[derive(Deserialize)]
struct ExecEnvelope {
    output: ExecOutput,
}

#[derive(Deserialize)]
struct ExecOutput {
    exit_code: i32,
    stdout: String,
    stderr: String,
}

/// A [`CommandRunner`] that runs commands inside an instd-managed VM.
///
/// Holds no connection: each call is one `instd instance exec`, which is stateless and therefore has
/// no reconnect, no keepalive, and no sleep/wake reattach to get wrong. A destroyed instance surfaces
/// as a failed CLI invocation rather than a hang, because instd's own teardown fence synthesizes a
/// terminal close for any exec in flight.
pub struct InstdRunner {
    instance_id: String,
    program: String,
    /// Prefix arguments before the `instd` invocation — `["sudo", "-n"]` when the daemon's admin
    /// channel needs privilege this process doesn't have.
    prefix: Vec<String>,
}

impl InstdRunner {
    /// Run commands inside `instance_id` via the `instd` on `PATH`.
    pub fn new(instance_id: impl Into<String>) -> Self {
        Self {
            instance_id: instance_id.into(),
            program: "instd".to_string(),
            prefix: Vec::new(),
        }
    }

    /// Invoke `instd` through `sudo -n` (non-interactive: fails rather than prompting, since there is
    /// no terminal to prompt on inside a tool call).
    pub fn with_sudo(mut self) -> Self {
        self.prefix = vec!["sudo".to_string(), "-n".to_string()];
        self
    }

    /// Override the `instd` binary path.
    pub fn with_cli(mut self, path: impl Into<String>) -> Self {
        self.program = path.into();
        self
    }

    pub fn instance_id(&self) -> &str {
        &self.instance_id
    }

    /// Build the argv for one exec.
    ///
    /// `cwd` has no flag on the CLI, so it is expressed by wrapping the command in a **fixed** `sh`
    /// script that `cd`s and then `exec`s the original argv from positional parameters — the same
    /// discipline `ShellFs` uses. The directory and the command are values of `$1`/`$@`, never text
    /// spliced into a script, so a path containing shell metacharacters stays inert.
    fn argv(&self, program: &str, args: &[String], cwd: Option<&str>) -> Vec<String> {
        let mut argv = self.prefix.clone();
        argv.push(self.program.clone());
        argv.push("instance".into());
        argv.push("exec".into());
        argv.push(self.instance_id.clone());
        // Everything after `--` is the guest command, never the CLI's own flags. Without it, a
        // perfectly ordinary `mkdir -p` has its `-p` claimed by the argument parser and the exec
        // fails — a real bug this hit on the first live run.
        argv.push("--".into());
        match cwd {
            Some(dir) => {
                argv.push("sh".into());
                argv.push("-c".into());
                argv.push(r#"cd "$1" || exit 1; shift; exec "$@""#.into());
                argv.push("sh".into());
                argv.push(dir.to_string());
                argv.push(program.to_string());
                argv.extend(args.iter().cloned());
            }
            None => {
                argv.push(program.to_string());
                argv.extend(args.iter().cloned());
            }
        }
        argv
    }
}

#[async_trait]
impl CommandRunner for InstdRunner {
    async fn run(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
    ) -> std::io::Result<ExecResult> {
        let argv = self.argv(program, args, cwd);
        let mut cmd = tokio::process::Command::new(&argv[0]);
        cmd.args(&argv[1..])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let out = match tokio::time::timeout(timeout, cmd.output()).await {
            Ok(r) => r?,
            Err(_) => {
                return Ok(ExecResult {
                    code: None,
                    signal: None,
                    stdout: String::new(),
                    stderr: format!("instd exec timed out after {:?}", timeout),
                    timed_out: true,
                    truncated: false,
                });
            }
        };

        let raw = String::from_utf8_lossy(&out.stdout);
        // The CLI prints one JSON envelope carrying the *guest command's* result. Its own exit status
        // reflects whether the exec could be delivered, not what the command did — so a failure to
        // parse means the exec itself failed (instance gone, daemon down, no privilege), and that is
        // reported rather than silently read as an empty successful command.
        match serde_json::from_str::<ExecEnvelope>(raw.trim()) {
            Ok(env) => Ok(ExecResult {
                code: Some(env.output.exit_code),
                signal: None,
                stdout: env.output.stdout,
                stderr: env.output.stderr,
                timed_out: false,
                truncated: false,
            }),
            Err(_) => {
                let detail = String::from_utf8_lossy(&out.stderr);
                let detail = detail.trim();
                Err(std::io::Error::other(format!(
                    "instd exec on {} failed: {}",
                    self.instance_id,
                    if detail.is_empty() {
                        raw.trim().to_string()
                    } else {
                        detail.to_string()
                    }
                )))
            }
        }
    }

    async fn run_streaming(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        on_chunk: ChunkSink<'_>,
    ) -> std::io::Result<ExecResult> {
        // The CLI buffers the guest's output into one JSON envelope, so there is nothing to stream
        // incrementally; the whole result arrives at once and is handed to the sink in one piece.
        // A framed-socket runner would stream genuinely — another reason it is the eventual
        // destination, and the reason this override exists rather than silently inheriting the
        // default and appearing to stream.
        let result = self.run(program, args, cwd, timeout).await?;
        if !result.stdout.is_empty() {
            on_chunk(result.stdout.as_bytes());
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn argv_puts_the_command_after_the_instance_id() {
        let r = InstdRunner::new("vm-1");
        let argv = r.argv("grep", &["-n".into(), "x".into()], None);
        assert_eq!(
            argv,
            vec!["instd", "instance", "exec", "vm-1", "--", "grep", "-n", "x"]
        );
    }

    #[test]
    fn a_dash_dash_separates_the_cli_flags_from_the_guest_command() {
        // Regression from the first live run: without `--`, `mkdir -p` fails because the CLI's own
        // argument parser claims `-p`.
        let r = InstdRunner::new("vm-1");
        let argv = r.argv("mkdir", &["-p".into(), "/tmp/x".into()], None);
        let dd = argv.iter().position(|a| a == "--").unwrap();
        assert_eq!(argv[dd + 1], "mkdir");
        assert_eq!(argv[dd + 2], "-p");
    }

    #[test]
    fn sudo_prefixes_the_invocation() {
        let r = InstdRunner::new("vm-1").with_sudo();
        let argv = r.argv("ls", &[], None);
        assert_eq!(&argv[..3], &["sudo", "-n", "instd"]);
    }

    #[test]
    fn a_cwd_becomes_positional_parameters_never_spliced_script_text() {
        // The invariant: a directory containing shell metacharacters must ride as a *value*, so it
        // cannot be parsed as syntax. If this ever regresses into `format!("cd {dir} && ...")`, this
        // test is what catches it.
        let r = InstdRunner::new("vm-1");
        let argv = r.argv("echo", &["hi".into()], Some("/tmp/'; rm -rf / #"));
        let script = argv.iter().find(|a| a.contains("cd \"$1\"")).unwrap();
        assert!(
            !script.contains("rm -rf"),
            "the script must not contain the caller's path: {script}"
        );
        assert!(
            argv.contains(&"/tmp/'; rm -rf / #".to_string()),
            "the path must appear as its own argv entry: {argv:?}"
        );
        // and the original command still follows it, in order
        let i = argv.iter().position(|a| a == "echo").unwrap();
        assert_eq!(argv[i + 1], "hi");
    }
}
