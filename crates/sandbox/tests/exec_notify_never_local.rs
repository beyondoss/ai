//! With an exec endpoint configured, commands and jobs run on the endpoint, whatever it answers
//! about v1.4 notify: no probe answer, notify setting, refused or failed `notify` makes anything run
//! on this host, and an endpoint that cannot be reached fails closed (an error), never locally.
//!
//! The mock runs commands on this machine too, so "on the endpoint" is told by its own `HOME`: a
//! directory only the mock hands its commands.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use beyond_sandbox::exec::CommandRunner;
use beyond_sandbox::exec_endpoint::HttpExecRunner;
use beyond_sandbox::exec_endpoint::jobs::{JobId, JobNotify, JobScope, JobSpec};
use beyond_test_support::exec_mock::{ExecMock, MockOptions, NotifyFaults, Protocol};

fn spec(script: &str) -> JobSpec {
    JobSpec {
        program: "sh".into(),
        args: vec!["-c".into(), script.into()],
        cwd: None,
        label: script.into(),
    }
}

/// Every way an endpoint can answer about notify, and some it should not.
fn variants() -> Vec<(&'static str, MockOptions)> {
    let notifying = MockOptions::notifying();
    let probe = |v: serde_json::Value| MockOptions {
        faults: NotifyFaults {
            probe_notify: Some(v),
            ..NotifyFaults::default()
        },
        ..MockOptions::notifying()
    };
    vec![
        ("v1.3, the setting off", MockOptions::default()),
        ("v1.4 natively", notifying.clone()),
        (
            "notify for the baseline",
            MockOptions {
                jobs: false,
                ..notifying.clone()
            },
        ),
        (
            "no native jobs, the setting off",
            MockOptions {
                jobs: false,
                ..MockOptions::default()
            },
        ),
        (
            "no egress",
            MockOptions {
                egress: false,
                ..notifying.clone()
            },
        ),
        ("notify as a boolean", probe(serde_json::json!(true))),
        ("a later notify version", probe(serde_json::json!("v9"))),
        (
            "notify for the baseline, without stdin",
            MockOptions {
                jobs: false,
                honors_stdin: false,
                ..notifying.clone()
            },
        ),
        (
            "notify for the baseline over v1.1",
            MockOptions {
                jobs: false,
                protocol: Protocol::V11,
                ..notifying.clone()
            },
        ),
        (
            "an idle-killing sandbox that advertises",
            MockOptions {
                idle_kill: Some(Duration::from_secs(60)),
                ..notifying.clone()
            },
        ),
        (
            "a notify accepted whatever it says",
            MockOptions {
                faults: NotifyFaults {
                    accepts_any_notify: true,
                    ..NotifyFaults::default()
                },
                ..notifying
            },
        ),
    ]
}

#[tokio::test]
async fn no_notify_answer_runs_anything_on_this_host() {
    for (name, opts) in variants() {
        let (root, home) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let home_path = home.path().to_string_lossy().into_owned();
        let mock = ExecMock::start_with(
            root.path(),
            MockOptions {
                home: Some(home.path().to_path_buf()),
                ..opts
            },
        )
        .await;
        let runner = Arc::new(HttpExecRunner::new(&mock.url).unwrap());
        let jobs = runner.clone().sandbox_jobs().unwrap();
        let _ = jobs.offers_notify().await;
        let r = runner
            .run(
                "sh",
                &["-c".into(), "printf %s \"$HOME\"".into()],
                None,
                Duration::from_secs(10),
            )
            .await
            .unwrap();
        assert_eq!(
            r.stdout, home_path,
            "{name}: a command ran off the endpoint"
        );
        if !jobs.available().await {
            continue;
        }
        let scope = JobScope::under(root.path().to_string_lossy(), "s");
        let id = JobId::fresh();
        jobs.start_kept(&scope, &id, &spec("printf %s \"$HOME\"; sleep 3"))
            .await
            .unwrap();
        // Keep it with notify, whether or not it is offered: refused or not, the job stays put.
        let _ = jobs
            .keep_notify(
                &scope,
                &id,
                1,
                "background",
                0,
                &JobNotify {
                    url: "http://127.0.0.1:9/_beyond/agent?session_id=s".into(),
                    token: "bjn_v1.x.y".into(),
                },
            )
            .await;
        let deadline = Instant::now() + Duration::from_secs(10);
        let out = loop {
            let c = jobs.read(&scope, &id, 0, 4096).await.unwrap();
            if !c.data.is_empty() || Instant::now() > deadline {
                break String::from_utf8_lossy(&c.data).into_owned();
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        };
        assert_eq!(out, home_path, "{name}: a job ran off the endpoint");
        let _ = jobs.wipe(&scope).await;
        assert!(
            !mock.requests().is_empty(),
            "{name}: nothing reached the endpoint"
        );
    }
}

/// An endpoint that cannot be reached is an error for every operation, never a local run.
#[tokio::test]
async fn an_unreachable_endpoint_fails_closed() {
    let port = beyond_test_support::ports::DeadPort::bind();
    let runner =
        Arc::new(HttpExecRunner::new(format!("http://127.0.0.1:{}/exec", port.port())).unwrap());
    let marker = tempfile::tempdir().unwrap();
    let file = marker.path().join("ran-here");
    let script = format!("touch {}", file.display());
    assert!(
        runner
            .run(
                "sh",
                &["-c".into(), script.clone()],
                None,
                Duration::from_secs(5)
            )
            .await
            .is_err()
    );
    let jobs = runner.clone().sandbox_jobs().unwrap();
    assert!(!jobs.offers_notify().await);
    let scope = JobScope::under(marker.path().to_string_lossy(), "s");
    assert!(
        jobs.start_kept(&scope, &JobId::fresh(), &spec(&script))
            .await
            .is_err()
    );
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(!file.exists(), "a command ran on this host");
}
