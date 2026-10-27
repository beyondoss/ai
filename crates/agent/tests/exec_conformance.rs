//! The exec endpoint conformance verifier (`exec_endpoint::conformance`), two ways.
//!
//! **Against an endpoint you are building** — the reason it exists. The other side of the exec
//! contract lives outside this repo; this is how it proves it conforms:
//!
//! ```sh
//! AGENT_CONFORMANCE_EXEC_URL=https://exec.example/run \
//! AGENT_CONFORMANCE_EXEC_HEADER='Authorization: Bearer …' \
//!   cargo nextest run -p beyond-agent -E 'test(/^exec_conformance::/)' --no-capture
//! ```
//!
//! Skipped when the URL is unset.
//!
//! The job notify checks run a callback receiver here. By default it listens on an
//! ephemeral loopback port, which a sandbox on this host's loopback reaches. For a sandbox
//! elsewhere, set `AGENT_CONFORMANCE_CALLBACK_LISTEN` (an address to listen on) and
//! `AGENT_CONFORMANCE_CALLBACK_ORIGIN` (an `https` origin that forwards to it; the TLS check is then
//! skipped). `AGENT_CONFORMANCE_NOTIFY_IDLE_SECS` sets how long a kept job must survive idling
//! (default 30).
//!
//! **Against the in-repo mock endpoints**, always: the verifier must pass a conforming endpoint and
//! fail each of the ways an endpoint gets the protocol wrong (one broken mock per notify check)
//! — otherwise a green run against a real endpoint would mean nothing.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::time::Duration;

use beyond_agent::exec_endpoint::conformance::{self, Expectations, Verdict, verify};
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
    assert!(
        report.checks.iter().all(|c| c.verdict == Verdict::Pass),
        "{}",
        report.render()
    );
    assert_eq!(report.checks.len(), 18 + conformance::notify::CHECKS.len());
}

/// A conforming mock with one fault: of the notify checks, exactly `check` fails. (The other checks
/// are other tests' business: a loaded host blurs their timings, not this rule.)
async fn fails_only(check: &str, opts: impl FnOnce(MockOptions) -> MockOptions) {
    let (base, expect) = quick();
    let opts = opts(MockOptions { ..base });
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

/// One broken endpoint per notify check, each failing exactly that check.
macro_rules! broken {
    ($($name:ident: $check:ident <= $opts:expr;)*) => {
        $(
            #[tokio::test]
            async fn $name() {
                fails_only(conformance::notify::$check, $opts).await;
            }
        )*
        /// Every notify check has its broken endpoint above.
        #[test]
        fn every_notify_check_has_a_broken_endpoint() {
            let mut named = vec![$(conformance::notify::$check),*];
            named.sort_unstable();
            let mut all = conformance::notify::CHECKS.to_vec();
            all.sort_unstable();
            assert_eq!(named, all);
        }
    };
}

broken! {
    an_idle_killing_sandbox_fails_idling: IDLE <= |o| MockOptions {
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

/// An endpoint that ignores `timeout_ms` on a one-shot request runs a hung command until the client leaves; the
/// verifier must catch it, and only it.
#[tokio::test]
async fn an_endpoint_that_ignores_the_one_shot_timeout_fails_that_check_alone() {
    let (opts, expect) = quick();
    let report = verify_mock(
        MockOptions {
            enforces_timeout: false,
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

/// An endpoint whose base64 is not base64 delivers nothing the client can decode: the verifier must
/// catch it, and only it.
#[tokio::test]
async fn an_endpoint_with_corrupt_base64_fails_the_byte_check() {
    let (opts, expect) = quick();
    let report = verify_mock(
        MockOptions {
            corrupts_base64: true,
            ..opts
        },
        &expect,
    )
    .await;
    let check = report.check(conformance::OUTPUT_BYTES).unwrap();
    assert_eq!(check.verdict, Verdict::Fail, "{}", report.render());
    assert!(
        check.detail.contains("not standard base64"),
        "{}",
        check.detail
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
        .map(|h| beyond_agent::exec_endpoint::HttpExecRunner::parse_header(&h).unwrap())
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
}
