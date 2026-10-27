//! Command-template mode (`docker exec ctr {}`, `kubectl exec …`, `ssh host --`): killing the local
//! transport does not kill the command on the far side, so every kill — `bash_job`, a cancelled
//! call, a timeout — must reach the far side through the transport itself.
//!
//! The transport here is a faithful stand-in for `docker exec`: it starts the far command in a new
//! session, so — exactly as measured against a real `docker exec` (see `exec_endpoint::far_kill`) —
//! signalling the transport's own process group leaves the far command running. The first test
//! proves the stand-in has that property; the rest prove the runner kills through it anyway. The
//! live suite (`exec_endpoint_live`) repeats the bash kill against a real container.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use beyond_agent::exec_endpoint::TemplateRunner;
use beyond_agent::tools::exec::CommandRunner;
use common::exec_mock::{group_alive, wait_group_gone};
use common::remote_bash::{call, job_number, kill_job, remote_bash};
use serde_json::json;

/// `sh -c 'setsid "$@" & wait $!' sh {}`: a transport whose far side lives in its own session.
fn detaching_transport() -> TemplateRunner {
    TemplateRunner::new(vec![
        "sh".into(),
        "-c".into(),
        r#"setsid "$@" & wait $!"#.into(),
        "sh".into(),
        "{}".into(),
    ])
    .unwrap()
}

/// Wait (bounded) for `path` to hold a pid.
async fn read_pid(path: &Path) -> u32 {
    for _ in 0..200 {
        if let Ok(s) = std::fs::read_to_string(path)
            && let Ok(pid) = s.trim().parse()
        {
            return pid;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("no pid in {}", path.display());
}

fn kill_group(pgid: u32) {
    let _ = std::process::Command::new("kill")
        .args(["-KILL", "--", &format!("-{pgid}")])
        .status();
}

#[tokio::test]
async fn the_stand_in_transport_like_docker_exec_does_not_propagate_a_kill() {
    let dir = tempfile::tempdir().unwrap();
    let pidfile = dir.path().join("pid");
    let mut transport = tokio::process::Command::new("sh")
        .args([
            "-c",
            r#"setsid "$@" & wait $!"#,
            "sh",
            "sh",
            "-c",
            &format!("echo $$ > {}; sleep 60", pidfile.display()),
        ])
        .process_group(0)
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let far = read_pid(&pidfile).await;
    kill_group(transport.id().unwrap());
    let _ = transport.wait().await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let survived = group_alive(far);
    kill_group(far);
    assert!(
        survived,
        "the stand-in must leave the far command running when the transport dies, as docker exec does"
    );
    assert!(wait_group_gone(far, Duration::from_secs(5)).await);
}

#[tokio::test]
async fn bash_job_kill_reaches_the_far_side() {
    let dir = tempfile::tempdir().unwrap();
    let pidfile = dir.path().join("pid");
    let (bash, jobs) = remote_bash(Arc::new(detaching_transport()), Duration::from_millis(500));
    let out = call(
        bash.as_ref(),
        json!({ "command": format!(
            "echo $$ > {}; echo template-tag; echo to-stderr >&2; (sleep 60) & sleep 60",
            pidfile.display()
        ) }),
    )
    .await
    .unwrap();
    let n = job_number(&out).unwrap_or_else(|| panic!("a template command yields too: {out}"));
    assert!(
        out.contains("template-tag") && out.contains("to-stderr"),
        "{out}"
    );
    assert!(
        !out.contains("exec-pgid") && !out.contains('\u{1e}'),
        "the far-side marker must never reach the model: {out:?}"
    );
    let far = read_pid(&pidfile).await;
    assert!(group_alive(far));

    let killed = kill_job(&jobs, n).await.unwrap();
    assert!(
        killed.contains(&format!("Job {n} (`")) && killed.contains(") killed after"),
        "{killed}"
    );
    let gone = wait_group_gone(far, Duration::from_secs(5)).await;
    if !gone {
        kill_group(far);
    }
    assert!(
        gone,
        "the far command and its grandchild outlived bash_job kill"
    );
}

#[tokio::test]
async fn a_template_timeout_kills_the_far_command() {
    let dir = tempfile::tempdir().unwrap();
    let pidfile = dir.path().join("pid");
    let res = detaching_transport()
        .run(
            "sh",
            &[
                "-c".into(),
                format!("echo $$ > {}; (sleep 60) & sleep 60", pidfile.display()),
            ],
            None,
            Duration::from_millis(800),
        )
        .await
        .unwrap();
    assert!(res.timed_out, "{res:?}");
    let far = read_pid(&pidfile).await;
    let gone = wait_group_gone(far, Duration::from_secs(5)).await;
    if !gone {
        kill_group(far);
    }
    assert!(
        gone,
        "a timed-out template command kept running on the far side"
    );
}

#[tokio::test]
async fn cancelling_a_template_call_kills_the_far_command() {
    let dir = tempfile::tempdir().unwrap();
    let pidfile = dir.path().join("pid");
    let runner = detaching_transport();
    let args = vec![
        "-c".to_string(),
        format!("echo $$ > {}; sleep 60", pidfile.display()),
    ];
    tokio::select! {
        r = runner.run_streaming("sh", &args, None, Duration::from_secs(120), &|_: &[u8]| {}) => {
            panic!("should have been dropped: {r:?}")
        }
        _ = read_pid(&pidfile) => {}
    }
    let far = read_pid(&pidfile).await;
    // The marker is read concurrently with the pid file being written; give the drop's far kill a
    // moment to land.
    let gone = wait_group_gone(far, Duration::from_secs(5)).await;
    if !gone {
        kill_group(far);
    }
    assert!(
        gone,
        "a cancelled template call left its far command running"
    );
}

#[tokio::test]
async fn a_template_command_still_reports_its_exit_and_cwd() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("here.txt"), "x").unwrap();
    let res = detaching_transport()
        .run(
            "sh",
            &["-c".into(), "ls; echo oops >&2; exit 5".into()],
            dir.path().to_str(),
            Duration::from_secs(10),
        )
        .await
        .unwrap();
    assert_eq!(res.code, Some(5));
    assert!(res.stdout_text().contains("here.txt"), "{res:?}");
    assert_eq!(
        res.stderr_text(),
        "oops\n",
        "marker stripped, the rest intact"
    );
    let missing = detaching_transport()
        .run(
            "true",
            &[],
            Some("/definitely/not/here"),
            Duration::from_secs(10),
        )
        .await
        .unwrap();
    assert_ne!(missing.code, Some(0), "{missing:?}");
}
