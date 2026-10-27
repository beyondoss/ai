//! What a remote command leaves running (`nohup server &`, `cmd &`) becomes a job, exactly as a local
//! shell's leftovers do: adopted the moment the shell exits, waited for, and stopped with the session.
//! Before, the far side's leftovers were never tracked: a streaming endpoint ended the stream with the
//! shell, a template transport's wrapper stopped watching after `wait $!`, and `nohup server &` held
//! its port until the sandbox died.
//!
//! Both transports, each on this host: the conforming streaming mock (`{"shell_exit": N}`, then the stream
//! stays open until the group is empty), and the template stand-in for `docker exec`, whose far
//! command lives in its own session, out of reach of any local kill. The live suite
//! (`exec_endpoint_live`) repeats this against a real container.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::sync::Arc;
use std::time::Duration;

use beyond_agent::exec_endpoint::{HttpExecRunner, TemplateRunner};
use beyond_agent::tools::exec::CommandRunner;
use common::exec_mock::{ExecMock, MockOptions, StreamEnd};
use common::remote_bash::{call, job_number, remote_bash};
use serde_json::json;

/// `sh -c 'setsid "$@" & wait $!' sh {}`: a transport whose far side lives in its own session, as
/// `docker exec`'s does (see `exec_template_kill.rs`).
fn detaching_transport() -> Arc<dyn CommandRunner> {
    Arc::new(
        TemplateRunner::new(vec![
            "sh".into(),
            "-c".into(),
            r#"setsid "$@" & wait $!"#.into(),
            "sh".into(),
            "{}".into(),
        ])
        .unwrap(),
    )
}

/// Live (non-zombie) processes whose command line is exactly `cmd`.
fn running(cmd: &str) -> Vec<u32> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir("/proc").unwrap().flatten() {
        let Some(pid) = entry.file_name().to_str().and_then(|s| s.parse().ok()) else {
            continue;
        };
        let line: Vec<String> = std::fs::read(format!("/proc/{pid}/cmdline"))
            .unwrap_or_default()
            .split(|&b| b == 0)
            .filter(|s| !s.is_empty())
            .map(|s| String::from_utf8_lossy(s).into_owned())
            .collect();
        let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).unwrap_or_default();
        let zombie = stat
            .rsplit_once(')')
            .is_some_and(|(_, rest)| rest.trim_start().starts_with('Z'));
        if line.join(" ") == cmd && !zombie {
            out.push(pid);
        }
    }
    out
}

async fn wait_gone(cmd: &str) -> bool {
    for _ in 0..100 {
        if running(cmd).is_empty() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    false
}

/// Kills what a failed assertion would otherwise leave running.
struct Reap(&'static str);

impl Drop for Reap {
    fn drop(&mut self) {
        for pid in running(self.0) {
            let _ = std::process::Command::new("kill")
                .args(["-KILL", &pid.to_string()])
                .status();
        }
    }
}

/// `nohup <cmd> &` on `runner`: adopted as a job at once, running, and gone once the session's
/// jobs are stopped.
async fn adopted_then_stopped(runner: Arc<dyn CommandRunner>, cmd: &'static str) {
    let _reap = Reap(cmd);
    // The idle window is far longer than the test: an adoption here is the shell's exit, never a
    // yield on silence.
    let (bash, _jobs) = remote_bash(runner, Duration::from_secs(60));
    let out = tokio::time::timeout(
        Duration::from_secs(20),
        call(
            bash.as_ref(),
            json!({ "command": format!("nohup {cmd} >/dev/null 2>&1 & echo started") }),
        ),
    )
    .await
    .expect("the call returns when the shell exits, not on the idle window")
    .unwrap();
    assert!(out.contains("started"), "{out}");
    assert!(
        out.contains("left processes on the exec target running") && out.contains("tracked as job"),
        "{out}"
    );
    assert!(
        !out.contains('\u{1e}'),
        "no marker reaches the model: {out:?}"
    );
    assert!(!running(cmd).is_empty(), "the leftover runs");
    let stopped = tokio::task::spawn_blocking({
        let bash = bash.clone();
        move || bash.jobs().stop_all()
    })
    .await
    .unwrap();
    assert_eq!(stopped.len(), 1, "{stopped:?}");
    assert!(wait_gone(cmd).await, "the leftover outlived the session");
}

#[tokio::test]
async fn a_v12_leftover_is_adopted_and_stopped_with_the_session() {
    let dir = tempfile::tempdir().unwrap();
    let mock = ExecMock::start(dir.path()).await;
    let runner = Arc::new(HttpExecRunner::new(&mock.url).unwrap());
    adopted_then_stopped(runner, "sleep 63.11").await;
    let s = mock.streams().into_iter().last().unwrap();
    assert_eq!(
        mock.wait_for_end(s.pgid, Duration::from_secs(5)).await,
        Some(StreamEnd::Disconnected { killed: true }),
        "the stop closed the stream, and the endpoint killed the group"
    );
}

#[tokio::test]
async fn a_template_leftover_is_adopted_and_stopped_with_the_session() {
    adopted_then_stopped(detaching_transport(), "sleep 63.12").await;
}

/// A leftover that keeps the shell's stdout open used to look like a silent command: the call
/// yielded on the idle window, as a *yielded* job the end of the turn kills. It is adopted at the
/// shell's exit, like a local one.
#[tokio::test]
async fn a_leftover_holding_stdout_is_adopted_not_yielded() {
    for (runner, cmd) in [
        (detaching_transport(), "sleep 63.13"),
        (
            Arc::new(
                HttpExecRunner::new(
                    &ExecMock::start(tempfile::tempdir().unwrap().path())
                        .await
                        .url,
                )
                .unwrap(),
            ) as Arc<dyn CommandRunner>,
            "sleep 63.14",
        ),
    ] {
        let _reap = Reap(cmd);
        let (bash, _jobs) = remote_bash(runner, Duration::from_secs(60));
        let out = tokio::time::timeout(
            Duration::from_secs(20),
            call(
                bash.as_ref(),
                json!({ "command": format!("{cmd} & echo started") }),
            ),
        )
        .await
        .expect("adopted at the shell's exit")
        .unwrap();
        assert!(out.contains("tracked as job"), "{cmd}: {out}");
        assert!(
            bash.jobs().end_turn().is_empty(),
            "{cmd}: an adopted job outlives the turn"
        );
        bash.jobs().stop_all();
        assert!(wait_gone(cmd).await, "{cmd}");
    }
}

/// A leftover that exits on its own ends the job, with its output and the shell's status.
#[tokio::test]
async fn a_leftover_that_exits_ends_its_job() {
    let dir = tempfile::tempdir().unwrap();
    let mock = ExecMock::start(dir.path()).await;
    for runner in [
        detaching_transport(),
        Arc::new(HttpExecRunner::new(&mock.url).unwrap()) as Arc<dyn CommandRunner>,
    ] {
        let (bash, jobs) = remote_bash(runner, Duration::from_secs(60));
        let out = call(
            bash.as_ref(),
            json!({ "command": "(sleep 1; echo late) & echo now" }),
        )
        .await
        .unwrap();
        assert!(
            out.contains("now") && out.contains("tracked as job"),
            "{out}"
        );
        let n = job_number(&out.replace("tracked as job", "as job")).unwrap();
        let waited = call(&jobs, json!({ "job": n })).await.unwrap();
        assert!(
            waited.contains("late") && waited.contains("have exited"),
            "{waited}"
        );
    }
}

/// An endpoint predating the rule ends the stream with the shell: the leftover is not adopted (and
/// runs on — the gap this closes). Pins that the client still works against one.
#[tokio::test]
async fn an_endpoint_without_shell_exit_still_ends_the_call_with_the_shell() {
    let cmd = "sleep 63.15";
    let _reap = Reap(cmd);
    let dir = tempfile::tempdir().unwrap();
    let mock = ExecMock::start_with(
        dir.path(),
        MockOptions {
            tracks_leftovers: false,
            ..MockOptions::default()
        },
    )
    .await;
    let runner = Arc::new(HttpExecRunner::new(&mock.url).unwrap());
    let (bash, _jobs) = remote_bash(runner, Duration::from_secs(60));
    let out = call(
        bash.as_ref(),
        json!({ "command": format!("nohup {cmd} >/dev/null 2>&1 & echo started") }),
    )
    .await
    .unwrap();
    assert!(
        out.contains("started") && !out.contains("tracked as job"),
        "{out}"
    );
}
