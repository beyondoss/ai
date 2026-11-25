//! A real E2B sandbox deleted while a job is followed: the handle ends with the backend's own answer
//! (`SandboxGone`) and not before, and a front that is merely refusing stays `Reconnecting`.
//!
//! Opt-in and billed (one VM, deleted by the test): needs `E2B_API_KEY` and `E2B_EXECD_BIN` (a static
//! execd), and `cargo test -p beyond-sandbox-conformance --test e2b_sandbox_gone -- --ignored --nocapture`.
//! The test fails (it never skips) when either is missing. It prints the backend's real answer.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use beyond_sandbox::exec_endpoint::HttpExecRunner;
use beyond_sandbox::exec_endpoint::jobs::{JobError, JobId, JobScope, JobSpec, SandboxJobs as _};
use beyond_sandbox::exec_endpoint::jobs_native::NativeJobs;
use sandbox_conformance::e2b;

fn sleeper() -> JobSpec {
    JobSpec {
        program: "sh".into(),
        args: vec!["-c".into(), "echo up; sleep 600".into()],
        cwd: None,
        label: "gone".into(),
        env: Vec::new(),
        stdin_open: false,
    }
}

/// `curl -i` of an exec request to `url`: the status line and the body, as the backend sent them.
async fn raw_answer(url: &str, bearer: &str) -> String {
    let out = tokio::process::Command::new("curl")
        .args(["-sS", "-i", "-m", "30", "-X", "POST", url, "-d", "{}", "-H"])
        .arg(format!("Authorization: Bearer {bearer}"))
        .output()
        .await
        .unwrap();
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[tokio::test]
#[ignore = "billed: one E2B VM; needs E2B_API_KEY and E2B_EXECD_BIN"]
async fn deleting_the_sandbox_mid_follow_ends_the_handle_with_the_backends_answer() {
    let key = e2b::key().expect("E2B_API_KEY is not set");
    assert!(
        std::env::var("E2B_EXECD_BIN").is_ok_and(|v| !v.is_empty()),
        "E2B_EXECD_BIN is not set"
    );
    let started = e2b::start("gone").await.unwrap();
    let bearer = started.bearer.clone().unwrap();
    // `https://8081-<id>.e2b.app/exec`
    let id = started
        .url
        .split("//")
        .nth(1)
        .and_then(|h| h.split('.').next())
        .and_then(|h| h.split_once('-'))
        .map(|(_, id)| id.to_string())
        .unwrap();
    let runner = Arc::new(
        HttpExecRunner::new(&started.url)
            .unwrap()
            .with_header("Authorization", format!("Bearer {bearer}")),
    );
    let (scope, job) = (
        JobScope::named("gone").unwrap(),
        JobId::parse("j1").unwrap(),
    );
    NativeJobs::over(&runner)
        .start_kept(&scope, &job, &sleeper())
        .await
        .unwrap();
    let follow = {
        let runner = runner.clone();
        let (scope, job) = (scope.clone(), job.clone());
        tokio::spawn(async move {
            let t = Instant::now();
            let r = NativeJobs::over(&runner)
                .follow(&scope, &job, 0, &|_| {})
                .await;
            (r, t.elapsed())
        })
    };
    // Followed and alive: it has not ended.
    tokio::time::sleep(Duration::from_secs(5)).await;
    assert!(
        !follow.is_finished(),
        "the handle ended with the sandbox alive"
    );
    e2b::delete_sandbox(&key, &id);
    let deleted = Instant::now();
    tokio::time::sleep(Duration::from_secs(8)).await;
    eprintln!("REAL ANSWER:\n{}", raw_answer(&started.url, &bearer).await);
    let (r, _) = tokio::time::timeout(Duration::from_secs(120), follow)
        .await
        .expect("the handle never ended after the sandbox was deleted")
        .unwrap();
    let e = r.expect_err("a followed job of a deleted sandbox ended Ok");
    eprintln!(
        "ended {:?} after the delete: kind {:?}: {e}",
        deleted.elapsed(),
        e.kind()
    );
    assert_eq!(e.kind(), std::io::ErrorKind::NotFound, "{e}");
    assert!(e.to_string().contains("sandbox no longer exists"), "{e}");
    assert!(JobError::of(&e).is_none(), "{e}");
    assert_eq!(
        runner.sandbox_id(),
        Some(id.as_str()),
        "the endpoint's id comes from its host"
    );
}
