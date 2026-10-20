//! The sandbox the demo deploys passes the repo's exec conformance suite, job notify included.
//!
//! It runs the verifier against the exact options the `exec-stub` binary serves
//! (`stub::options`), on a loopback port the test holds, so the sandbox's callback receiver and the
//! verifier's share a host. A full run is about 50 s: the suite waits out a deliberately silent
//! command, a grandchild's timer, and the idle window a kept job must survive.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::time::Duration;

use beyond_ai_agent::exec_endpoint::conformance::{Expectations, Verdict, notify, verify};
use beyond_ai_demo::stub;
use beyond_ai_test_support::exec_mock::ExecMock;

#[tokio::test]
async fn the_demo_sandbox_passes_every_conformance_check() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let mock = ExecMock::start_with(dir.path(), stub::options(home, None)).await;

    let report = verify(
        &mock.url,
        &[],
        &Expectations {
            notify_idle: Duration::from_secs(5),
            ..Expectations::default()
        },
    )
    .await;
    println!("{}", report.render());

    assert!(report.streams, "{}", report.render());
    assert!(report.jobs, "{}", report.render());
    assert!(
        report.notify,
        "the sandbox must advertise job notify (v1.4)"
    );
    assert!(
        report.checks.iter().all(|c| c.verdict == Verdict::Pass),
        "{}",
        report.render()
    );
    for name in notify::CHECKS {
        assert!(
            report.check(name).is_some(),
            "the v1.4 check {name} did not run"
        );
    }
}

/// A sandbox on the public internet requires the bearer its grant carries: the suite passes with it,
/// and a request without it (or with another) is answered 401 before anything runs.
#[tokio::test]
async fn a_bearer_protected_sandbox_conforms_and_refuses_everyone_else() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let mock = ExecMock::start_with(dir.path(), stub::options(home, Some("tok-123".into()))).await;

    let addr = mock
        .url
        .trim_start_matches("http://")
        .split('/')
        .next()
        .unwrap()
        .to_string();
    let status = |auth: Option<&'static str>| {
        let addr = addr.clone();
        async move {
            let body = r#"{"command":"true"}"#;
            let auth = auth
                .map(|a| format!("Authorization: {a}\r\n"))
                .unwrap_or_default();
            let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
            s.write_all(
                format!(
                    "POST /exec HTTP/1.1\r\nHost: x\r\n{auth}Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .as_bytes(),
            )
            .await
            .unwrap();
            let mut out = String::new();
            s.read_to_string(&mut out).await.unwrap();
            out.split_whitespace().nth(1).unwrap().to_string()
        }
    };
    assert_eq!(status(None).await, "401");
    assert_eq!(status(Some("Bearer wrong")).await, "401");
    assert_eq!(status(Some("Bearer tok-123")).await, "200");

    let report = verify(
        &mock.url,
        &[("Authorization".into(), "Bearer tok-123".into())],
        &Expectations {
            notify_idle: Duration::from_secs(5),
            ..Expectations::default()
        },
    )
    .await;
    assert!(report.notify, "{}", report.render());
    assert!(
        report.checks.iter().all(|c| c.verdict == Verdict::Pass),
        "{}",
        report.render()
    );
}
