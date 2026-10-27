//! `run` e2e: a failed run prints one plain `error: …` line that names what went wrong (and where),
//! never Rust's `Debug` dump of the error value.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::process::Stdio;

use common::run_cmd;

fn run(args: &[&str]) -> (bool, String) {
    let dir = tempfile::tempdir().unwrap();
    let out = run_cmd(env!("CARGO_BIN_EXE_agent"))
        .args(["run", "--key", "bai_v1.test", "--model", "claude-test"])
        .args(args)
        .arg("hi")
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .unwrap();
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn assert_plain(stderr: &str) {
    for debug in ["Os {", "kind:", "Transport(", "\\\"", "Error: "] {
        assert!(
            !stderr.contains(debug),
            "a Debug rendering leaked ({debug}): {stderr}"
        );
    }
}

#[test]
fn an_unusable_session_dir_names_the_path() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("a-file");
    std::fs::write(&file, "").unwrap();
    // A directory can't be created under a regular file, whoever runs the test.
    let session_dir = file.join("sessions");
    let session_dir = session_dir.to_str().unwrap();
    let (ok, stderr) = run(&[
        "--gateway-url",
        "http://127.0.0.1:9",
        "--session-dir",
        session_dir,
    ]);
    assert!(!ok);
    assert!(
        stderr.contains(&format!(
            "error: cannot create session directory {session_dir}: "
        )),
        "{stderr}"
    );
    assert_plain(&stderr);
}

#[test]
fn a_transport_failure_prints_as_plain_text() {
    let port = {
        let l = beyond_test_support::ports::listener();
        l.local_addr().unwrap().port()
    };
    let gateway = format!("http://127.0.0.1:{port}");
    let (ok, stderr) = run(&[
        "--gateway-url",
        &gateway,
        "--no-session-persistence",
        "--retry-max-retries",
        "0",
        "--no-retry",
    ]);
    assert!(!ok);
    assert!(stderr.contains("error: transport error: "), "{stderr}");
    assert_plain(&stderr);
}
