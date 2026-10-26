//! The exec endpoint conformance verifier (`exec_endpoint::conformance`), two ways.
//!
//! **Against an endpoint you are building** — the reason it exists. The other side of the exec
//! contract lives outside this repo; this is how it proves it conforms:
//!
//! ```sh
//! AGENT_CONFORMANCE_EXEC_URL=https://exec.example/run \
//! AGENT_CONFORMANCE_EXEC_HEADER='Authorization: Bearer …' \
//!   cargo nextest run -p beyond-ai-agent -E 'test(/^exec_conformance::/)' --no-capture
//! ```
//!
//! It requires v1.2; set `AGENT_CONFORMANCE_ALLOW_V11=1` to accept a v1.1 endpoint (whose streaming
//! checks are then skipped). Skipped when the URL is unset.
//!
//! The v1.4 (job notify) checks run a callback receiver here. By default it listens on an
//! ephemeral loopback port, which a sandbox on this host's loopback reaches. For a sandbox
//! elsewhere, set `AGENT_CONFORMANCE_CALLBACK_LISTEN` (an address to listen on) and
//! `AGENT_CONFORMANCE_CALLBACK_ORIGIN` (an `https` origin that forwards to it; the TLS check is then
//! skipped). `AGENT_CONFORMANCE_NOTIFY_IDLE_SECS` sets how long a kept job must survive idling
//! (default 30).
//!
//! **Against the in-repo mock endpoints**, always: the verifier must pass a conforming endpoint and
//! fail each of the ways an endpoint gets v1.2, v1.3 or v1.4 wrong (one broken mock per v1.4 check)
//! — otherwise a green run against a real endpoint would mean nothing.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::time::Duration;

use beyond_ai_agent::exec_endpoint::conformance::{self, Expectations, Verdict, verify};
use common::exec_mock::{ExecMock, MockOptions, NotifyFaults, Protocol};

/// The mocks heartbeat fast so the verifier's silent command need not wait out the spec's 5 s.
fn quick() -> (MockOptions, Expectations) {
    (
        MockOptions {
            heartbeat: Duration::from_millis(300),
            ..MockOptions::default()
        },
        Expectations {
            heartbeat: Duration::from_millis(300),
            kill_within: Duration::from_secs(1),
            notify_idle: Duration::from_secs(1),
            ..Expectations::default()
        },
    )
}

async fn verify_mock(opts: MockOptions, expect: &Expectations) -> conformance::Report {
    let dir = tempfile::tempdir().unwrap();
    let mock = ExecMock::start_with(dir.path(), opts).await;
    let report = verify(&mock.url, &[], expect).await;
    println!("{}", report.render());
    // A non-conforming endpoint leaves its command running; this suite must not.
    for s in mock.streams() {
        let _ = std::process::Command::new("kill")
            .args(["-KILL", "--", &format!("-{}", s.pgid)])
            .stderr(std::process::Stdio::null())
            .status();
    }
    report
}

#[tokio::test]
async fn a_conforming_endpoint_passes_every_check() {
    let (opts, expect) = quick();
    let report = verify_mock(opts, &expect).await;
    assert!(report.streams);
    assert!(report.jobs, "the default mock offers v1.3 jobs");
    assert!(!report.notify, "its deployment setting is off by default");
    assert!(report.passed(), "{}", report.render());
    let skipped: Vec<&str> = report
        .checks
        .iter()
        .filter(|c| c.verdict != Verdict::Pass)
        .map(|c| c.name)
        .collect();
    assert_eq!(skipped, conformance::notify::CHECKS, "{}", report.render());
    assert_eq!(report.checks.len(), 18 + conformance::notify::CHECKS.len());
}

/// The reference v1.4 endpoint passes every check, the v1.4 ones included.
#[tokio::test]
async fn a_conforming_v14_endpoint_passes_every_check() {
    let (opts, expect) = quick();
    let report = verify_mock(
        MockOptions {
            kept_jobs_survive_idle: true,
            ..opts
        },
        &expect,
    )
    .await;
    assert!(report.jobs && report.notify);
    assert!(
        report.checks.iter().all(|c| c.verdict == Verdict::Pass),
        "{}",
        report.render()
    );
    assert_eq!(report.checks.len(), 18 + conformance::notify::CHECKS.len());
    assert!(report.render().contains("v1.4"), "{}", report.render());
}

/// The endpoint modes that do not offer notify natively pass, their v1.4 checks skipped: one that
/// knows it has no egress, an idle-killing sandbox deployed honestly, and one that offers notify
/// for the baseline.
#[tokio::test]
async fn endpoints_without_native_notify_pass_with_its_checks_skipped() {
    let (opts, expect) = quick();
    for (name, o, notify) in [
        (
            "no egress",
            MockOptions {
                kept_jobs_survive_idle: true,
                egress: false,
                ..opts.clone()
            },
            false,
        ),
        (
            "an idle-killing sandbox",
            MockOptions {
                idle_kill: Some(Duration::from_millis(2500)),
                ..opts.clone()
            },
            false,
        ),
        (
            "notify for the baseline",
            MockOptions {
                jobs: false,
                kept_jobs_survive_idle: true,
                ..opts.clone()
            },
            true,
        ),
    ] {
        let report = verify_mock(o, &expect).await;
        assert!(report.passed(), "{name}: {}", report.render());
        assert_eq!(report.notify, notify, "{name}");
        for check in &conformance::notify::CHECKS[1..] {
            assert_eq!(
                report.check(check).unwrap().verdict,
                Verdict::Skip,
                "{name}: {check}"
            );
        }
    }
}

/// A conforming v1.4 mock with one fault: of the v1.4 checks, exactly `check` fails. (The v1.2 and
/// v1.3 checks are other tests' business: a loaded host blurs their timings, not this rule.)
async fn fails_only(check: &str, opts: impl FnOnce(MockOptions) -> MockOptions) {
    let (base, expect) = quick();
    let opts = opts(MockOptions {
        kept_jobs_survive_idle: true,
        ..base
    });
    // The idle window outlasts the idle-killing variant's 2.5 s.
    let expect = Expectations {
        notify_idle: Duration::from_millis(3500),
        ..expect
    };
    let report = verify_mock(opts, &expect).await;
    let failed: Vec<&str> = report
        .checks
        .iter()
        .filter(|c| c.verdict == Verdict::Fail && conformance::notify::CHECKS.contains(&c.name))
        .map(|c| c.name)
        .collect();
    assert_eq!(failed, vec![check], "{}", report.render());
}

fn with(f: NotifyFaults) -> impl FnOnce(MockOptions) -> MockOptions {
    move |o| MockOptions { faults: f, ..o }
}

/// One broken endpoint per v1.4 check, each failing exactly that check.
macro_rules! broken {
    ($($name:ident: $check:ident <= $opts:expr;)*) => {
        $(
            #[tokio::test]
            async fn $name() {
                fails_only(conformance::notify::$check, $opts).await;
            }
        )*
        /// Every v1.4 check has its broken endpoint above.
        #[test]
        fn every_v14_check_has_a_broken_endpoint() {
            let mut named = vec![$(conformance::notify::$check),*];
            named.sort_unstable();
            let mut all = conformance::notify::CHECKS.to_vec();
            all.sort_unstable();
            assert_eq!(named, all);
        }
    };
}

broken! {
    notify_answered_as_a_boolean_fails_the_offer: NOTIFY <= with(NotifyFaults {
        probe_notify: Some(serde_json::json!(true)),
        ..NotifyFaults::default()
    });
    an_idle_killing_sandbox_that_advertises_fails_idling: IDLE <= |o| MockOptions {
        idle_kill: Some(Duration::from_millis(2500)),
        ..o
    };
    accepting_any_notify_fails_the_grammar: KEEP_INVALID <= with(NotifyFaults {
        accepts_any_notify: true,
        ..NotifyFaults::default()
    });
    keeping_the_first_notify_fails_the_latest: LATEST <= with(NotifyFaults {
        keeps_first_notify: true,
        ..NotifyFaults::default()
    });
    an_extra_body_member_fails_the_shape: SHAPE <= with(NotifyFaults {
        body_extra_member: true,
        ..NotifyFaults::default()
    });
    ends_sent_elsewhere_fail_the_same_url: SAME_URL <= with(NotifyFaults {
        end_url_suffix: Some("&moved=1".into()),
        ..NotifyFaults::default()
    });
    a_keep_that_waits_for_its_proof_fails_no_wait: NO_WAIT <= with(NotifyFaults {
        keep_waits_for_proof: true,
        ..NotifyFaults::default()
    });
    a_late_proof_fails_the_first_callback: FIRST <= with(NotifyFaults {
        proof_delay: Duration::from_secs(3),
        ..NotifyFaults::default()
    });
    no_shell_exit_callback_fails_after_the_end: AFTER_END <= with(NotifyFaults {
        no_shell_exit_callback: true,
        ..NotifyFaults::default()
    });
    dropping_a_change_in_flight_fails_during: DURING <= with(NotifyFaults {
        drops_changes_in_flight: true,
        ..NotifyFaults::default()
    });
    a_plain_keep_that_keeps_notify_fails_no_callback: NO_CALLBACK <= with(NotifyFaults {
        plain_keep_keeps_notify: true,
        ..NotifyFaults::default()
    });
    following_redirects_fails_redirects: REDIRECTS <= with(NotifyFaults {
        follows_redirects: true,
        ..NotifyFaults::default()
    });
    accepting_any_certificate_fails_tls: TLS <= with(NotifyFaults {
        accepts_invalid_certs: true,
        ..NotifyFaults::default()
    });
    retrying_a_404_fails_retry: RETRY <= with(NotifyFaults {
        retries_refusals: true,
        ..NotifyFaults::default()
    });
    the_token_in_list_fails_the_token: TOKEN <= with(NotifyFaults {
        token_in_list: true,
        ..NotifyFaults::default()
    });
}

#[tokio::test]
async fn a_v12_endpoint_passes_with_its_job_checks_skipped() {
    let (opts, expect) = quick();
    let report = verify_mock(
        MockOptions {
            jobs: false,
            ..opts
        },
        &expect,
    )
    .await;
    assert!(report.streams && !report.jobs);
    assert!(report.passed(), "{}", report.render());
    for name in [
        conformance::JOBS,
        conformance::JOBS_HOLD,
        conformance::JOBS_KEEP,
    ] {
        assert_eq!(
            report.check(name).unwrap().verdict,
            Verdict::Skip,
            "{name}: {}",
            report.render()
        );
    }
    assert!(
        report.render().contains("plain exec commands"),
        "{}",
        report.render()
    );
}

#[tokio::test]
async fn an_endpoint_whose_dropped_hold_leaves_the_job_running_fails_the_hold_check() {
    let (opts, expect) = quick();
    let report = verify_mock(
        MockOptions {
            jobs_hold_kills: false,
            ..opts
        },
        &expect,
    )
    .await;
    let failed: Vec<&str> = report
        .checks
        .iter()
        .filter(|c| c.verdict == Verdict::Fail)
        .map(|c| c.name)
        .collect();
    assert_eq!(failed, vec![conformance::JOBS_HOLD], "{}", report.render());
    assert!(
        report
            .check(conformance::JOBS_HOLD)
            .unwrap()
            .detail
            .contains("die with its connection"),
        "{}",
        report.render()
    );
}

#[tokio::test]
async fn a_v11_endpoint_passes_with_its_streaming_checks_skipped() {
    let (opts, expect) = quick();
    let report = verify_mock(
        MockOptions {
            protocol: Protocol::V11,
            ..opts
        },
        &expect,
    )
    .await;
    assert!(!report.streams);
    assert!(report.passed(), "{}", report.render());
    assert_eq!(
        report.check(conformance::ENVELOPE).unwrap().verdict,
        Verdict::Pass
    );
    assert_eq!(
        report.check(conformance::DISCONNECT).unwrap().verdict,
        Verdict::Skip
    );
}

#[tokio::test]
async fn a_buffering_endpoint_fails_incremental_delivery_and_heartbeats() {
    let (opts, expect) = quick();
    let report = verify_mock(
        MockOptions {
            protocol: Protocol::Buffered,
            ..opts
        },
        &expect,
    )
    .await;
    assert!(!report.passed());
    for name in [conformance::INCREMENTAL, conformance::HEARTBEAT] {
        assert_eq!(
            report.check(name).unwrap().verdict,
            Verdict::Fail,
            "{name}: {}",
            report.render()
        );
    }
}

#[tokio::test]
async fn an_endpoint_that_ends_the_stream_with_the_shell_fails_the_leftover_check() {
    let (opts, expect) = quick();
    let report = verify_mock(
        MockOptions {
            tracks_leftovers: false,
            ..opts
        },
        &expect,
    )
    .await;
    let check = report.check(conformance::LEFTOVERS).unwrap();
    assert_eq!(check.verdict, Verdict::Fail, "{}", report.render());
    assert!(check.detail.contains("shell_exit"), "{}", check.detail);
    assert_eq!(
        report
            .checks
            .iter()
            .filter(|c| c.verdict == Verdict::Fail)
            .count(),
        1,
        "{}",
        report.render()
    );
}

#[tokio::test]
async fn an_endpoint_that_survives_a_disconnect_fails_the_kill_check() {
    let (opts, expect) = quick();
    let report = verify_mock(
        MockOptions {
            kills_on_disconnect: false,
            ..opts
        },
        &expect,
    )
    .await;
    let check = report.check(conformance::DISCONNECT).unwrap();
    assert_eq!(check.verdict, Verdict::Fail, "{}", report.render());
    assert!(check.detail.contains("still running"), "{}", check.detail);
    // Everything else about it is fine: the failures are pinned on the one requirement it breaks —
    // kill on disconnect, which the leftover check measures a second time, after `shell_exit`.
    let failed: Vec<&str> = report
        .checks
        .iter()
        .filter(|c| c.verdict == Verdict::Fail)
        .map(|c| c.name)
        .collect();
    assert_eq!(
        failed,
        vec![conformance::DISCONNECT, conformance::LEFTOVERS],
        "{}",
        report.render()
    );
}

/// An endpoint that ignores `timeout_ms` over v1.1 runs a hung command until the client leaves; the
/// verifier must catch it, and only it.
#[tokio::test]
async fn an_endpoint_that_ignores_the_v11_timeout_fails_that_check_alone() {
    let (opts, expect) = quick();
    let report = verify_mock(
        MockOptions {
            v11_timeout: false,
            ..opts
        },
        &expect,
    )
    .await;
    let failed: Vec<&str> = report
        .checks
        .iter()
        .filter(|c| c.verdict == Verdict::Fail)
        .map(|c| c.name)
        .collect();
    assert_eq!(
        failed,
        vec![conformance::ENVELOPE_TIMEOUT],
        "{}",
        report.render()
    );
    assert!(
        report
            .check(conformance::ENVELOPE_TIMEOUT)
            .unwrap()
            .detail
            .contains("ignores the deadline"),
        "{}",
        report.render()
    );
}

#[tokio::test]
async fn a_live_endpoint_conforms() {
    let Ok(url) = std::env::var("AGENT_CONFORMANCE_EXEC_URL") else {
        eprintln!("skipping: set AGENT_CONFORMANCE_EXEC_URL to verify an endpoint");
        return;
    };
    let headers: Vec<(String, String)> = std::env::var("AGENT_CONFORMANCE_EXEC_HEADER")
        .ok()
        .filter(|h| !h.is_empty())
        .map(|h| beyond_ai_agent::exec_endpoint::HttpExecRunner::parse_header(&h).unwrap())
        .into_iter()
        .collect();
    let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
    let mut expect = Expectations::default();
    if let Some(listen) = env("AGENT_CONFORMANCE_CALLBACK_LISTEN") {
        expect.callback_listen = listen;
    }
    expect.callback_origin = env("AGENT_CONFORMANCE_CALLBACK_ORIGIN");
    if let Some(secs) = env("AGENT_CONFORMANCE_NOTIFY_IDLE_SECS") {
        expect.notify_idle = Duration::from_secs(secs.parse().unwrap());
    }
    let report = verify(&url, &headers, &expect).await;
    println!("{}", report.render());
    assert!(report.passed(), "{}", report.render());
    if std::env::var("AGENT_CONFORMANCE_ALLOW_V11").as_deref() != Ok("1") {
        assert!(
            report.streams,
            "a v1.1 endpoint: `bash` cannot yield or kill its commands (set \
             AGENT_CONFORMANCE_ALLOW_V11=1 to accept it)"
        );
    }
}
