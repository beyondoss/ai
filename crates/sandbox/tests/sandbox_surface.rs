//! Layer 3, and a helper a user writes on layer 1 alone.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

#[path = "sdk_conformance/local_sh.rs"]
mod local_sh;

use std::sync::Arc;
use std::time::Duration;

use beyond_sandbox::{Exec, FsBackend, Sandbox, TokenSource};
use beyond_test_support::exec_mock::{ExecMock, MockOptions};

/// A user's own helper: `git status --porcelain` as (code, path) pairs. It takes any [`Exec`] and
/// uses nothing but `run`: no files, no jobs, no HTTP.
async fn git_status(exec: &dyn Exec, repo: &str) -> std::io::Result<Vec<(String, String)>> {
    let out = exec
        .run(
            "git",
            &["status".into(), "--porcelain".into()],
            Some(repo),
            Duration::from_secs(30),
        )
        .await?;
    if out.code != Some(0) {
        return Err(std::io::Error::other(out.stderr_text().into_owned()));
    }
    Ok(out
        .stdout_text()
        .lines()
        .filter_map(|l| l.split_once(' '))
        .map(|(code, path)| (code.to_string(), path.trim().to_string()))
        .collect())
}

fn sh(script: &str) -> Vec<String> {
    vec!["-c".into(), script.into()]
}

#[tokio::test]
async fn a_helper_written_on_raw_exec_runs_over_any_transport() {
    let dir = tempfile::tempdir().unwrap();
    let repo = dir.path().to_str().unwrap();
    let exec = local_sh::LocalSh;
    let init = sh("git init -q . && echo a > tracked && git add tracked && echo b > loose");
    assert_eq!(
        exec.run("sh", &init, Some(repo), Duration::from_secs(30))
            .await
            .unwrap()
            .code,
        Some(0)
    );
    let status = git_status(&exec, repo).await.unwrap();
    assert_eq!(
        status,
        [
            ("A".to_string(), "tracked".to_string()),
            ("??".to_string(), "loose".to_string())
        ]
    );
}

#[tokio::test]
async fn a_sandbox_over_a_test_written_transport_has_all_three_layers() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    let sandbox = Sandbox::over(Arc::new(local_sh::LocalSh)).await.unwrap();

    let out = sandbox
        .exec("sh", ["-c", "echo $0; pwd"])
        .cwd(root.to_str().unwrap())
        .run()
        .await
        .unwrap();
    assert_eq!(out.stdout_text(), format!("sh\n{}\n", root.display()));

    let piped = sandbox
        .exec("cat", [] as [&str; 0])
        .stdin("in")
        .run()
        .await
        .unwrap();
    assert_eq!(piped.stdout, b"in");

    let file = root.join("notes.txt");
    sandbox
        .files()
        .write_bytes(&file, b"hello\n".to_vec())
        .await
        .unwrap();
    assert_eq!(
        sandbox.files().read_bytes(&file, 0, 1024).await.unwrap(),
        b"hello\n"
    );

    assert!(
        sandbox.jobs().available().await,
        "the baseline jobs are built from plain exec"
    );
    let raw = sandbox
        .raw()
        .run("true", &[], None, Duration::from_secs(5))
        .await
        .unwrap();
    assert_eq!(raw.code, Some(0));
}

#[tokio::test]
async fn connect_reaches_an_endpoint_with_a_bearer_and_streams() {
    let dir = tempfile::tempdir().unwrap();
    let work = dir.path().canonicalize().unwrap();
    let mock = ExecMock::start_with(
        &work,
        MockOptions {
            require_bearer: Some("secret".into()),
            ..MockOptions::default()
        },
    )
    .await;

    // Connecting sends nothing; the first request is where a wrong bearer is refused.
    let wrong = Sandbox::connect(mock.url.clone(), "nope").await.unwrap();
    let refused = wrong.exec("true", [] as [&str; 0]).run().await;
    assert!(refused.unwrap_err().to_string().contains("401"));

    let sandbox = Sandbox::connect(mock.url.clone(), TokenSource::fixed("secret"))
        .await
        .unwrap();
    let chunks = std::sync::Mutex::new(Vec::new());
    let out = sandbox
        .exec("sh", ["-c", "printf one; sleep 0.2; printf two"])
        .stream(&|b| {
            chunks
                .lock()
                .unwrap()
                .push(String::from_utf8_lossy(b).to_string())
        })
        .await
        .unwrap();
    assert_eq!(out.code, Some(0));
    assert_eq!(chunks.lock().unwrap().concat(), "onetwo");
    assert!(
        chunks.lock().unwrap().len() >= 2,
        "output arrived as it was produced"
    );

    let file = work.join("f");
    sandbox
        .files()
        .write_bytes(&file, b"x".to_vec())
        .await
        .unwrap();
    assert_eq!(sandbox.files().stat(&file).await.unwrap().unwrap().len, 1);
}

/// The test transport itself must not leak: a command past its deadline is killed with every process
/// it started, and so is one whose call is dropped.
#[tokio::test]
async fn the_test_transport_leaves_no_process_behind() {
    fn gone(pid: i32) -> bool {
        match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
            Err(_) => true,
            Ok(stat) => stat.contains(") Z"),
        }
    }
    async fn wait_gone(pidfile: &std::path::Path) {
        let pid: i32 = std::fs::read_to_string(pidfile)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        for _ in 0..100 {
            if gone(pid) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("process {pid} outlived its command");
    }
    let dir = tempfile::tempdir().unwrap();
    let exec = local_sh::LocalSh;
    let script = |name: &str| {
        let pidfile = dir.path().join(name);
        let args = sh(&format!(
            "sleep 300 & echo $! > {}; wait",
            pidfile.display()
        ));
        (pidfile, args)
    };

    let (pidfile, args) = script("timeout.pid");
    let err = exec
        .run("sh", &args, None, Duration::from_millis(500))
        .await
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::TimedOut);
    wait_gone(&pidfile).await;

    let (pidfile, args) = script("dropped.pid");
    let call = exec.run_attached("sh", &args, None, &|_| {}, &|_| {});
    let _ = tokio::time::timeout(Duration::from_millis(500), call).await;
    wait_gone(&pidfile).await;
}
