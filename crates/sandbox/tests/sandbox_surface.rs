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
        return Err(std::io::Error::other(out.stderr));
    }
    Ok(out
        .stdout
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
    assert_eq!(out.stdout, format!("sh\n{}\n", root.display()));

    let piped = sandbox
        .exec("cat", [] as [&str; 0])
        .stdin("in")
        .run()
        .await
        .unwrap();
    assert_eq!(piped.stdout, "in");

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

    let wrong = Sandbox::connect(mock.url.clone(), "nope").await;
    assert!(wrong.is_err(), "the probe is refused");

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
