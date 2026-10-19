//! exec v1.4 end to end, the release: a model starts a background job through a real `serve
//! --service` replica behind the fleet's front; once `keep` and the job's proof are in, nothing
//! crosses a replica-to-endpoint connection until the job ends (counted in bytes at the socket),
//! and the end then produces exactly the notice, and the wake, that the held stream produces
//! today. Each case runs in every endpoint mode (`common::notify_e2e::Mode`): with no egress the
//! stream stays held and the end comes through it.
#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::time::{Duration, Instant};

use common::notify_e2e::{E2e, Mode, Setup, Verdict, in_every_mode, notice, unique};

/// A job that prints its marker only at its end, so the command line never contains it.
fn job(secs: u32, marker: &str) -> String {
    format!("echo up; sleep {secs}; printf 'fin-%s\\n' {marker}")
}

/// Cases 1–3: the model starts the job; after `keep` and its proof, zero bytes cross any
/// replica-to-endpoint connection until the end's callback arrives (N35, N39); with no egress
/// the stream stays held, and its heartbeats cross. The end produces one notice, carrying the
/// output and the exit code, identical to the one the held stream produces with notify off, and
/// starts the same one wake turn (N33, N34).
async fn a_released_job_is_silent_until_its_end(mode: Mode) {
    let marker = "m1";
    let f = E2e::start(Setup::new(mode)).await;
    let session = &unique("alpha");
    let mut ws = f.start_job(session, &job(7, marker)).await;
    let pgid = f.stream(&job(7, marker)).await;
    let quiet_from;
    if mode.releases() {
        let proof = f.proof(session).await;
        assert_eq!(proof.status, 200, "{mode:?}: the proof was taken");
        f.released(pgid).await;
        // The release's own reconcile lists once; then nothing.
        tokio::time::sleep(Duration::from_millis(1500)).await;
        quiet_from = Instant::now();
    } else {
        tokio::time::sleep(Duration::from_millis(1500)).await;
        quiet_from = Instant::now();
        assert!(f.front.seen().is_empty(), "no callback without egress");
    }
    let wake = f
        .wake_with(&format!("fin-{marker}"), Duration::from_secs(30))
        .await
        .unwrap_or_else(|| panic!("{mode:?}: the end never reached the model"));
    if mode.releases() {
        let end = f
            .front
            .seen()
            .into_iter()
            .find(|c| c.nth >= 2)
            .expect("the end's callback");
        assert_eq!(end.status, 200, "{mode:?}: {end:?}");
        assert!(end.at > quiet_from + Duration::from_secs(2), "{end:?}");
        let crossed: usize = f
            .meter
            .chunks()
            .iter()
            .filter(|(t, _)| *t >= quiet_from && *t < end.at)
            .map(|(_, n)| n)
            .sum();
        assert_eq!(
            crossed, 0,
            "{mode:?}: bytes crossed to the endpoint while the job was released"
        );
    } else {
        assert!(
            f.meter.bytes_since(quiet_from) > 0,
            "the held stream's heartbeats"
        );
    }
    let text = notice(&wake, &format!("fin-{marker}"));
    assert!(
        text.contains("exited with code 0") && text.contains(&format!("fin-{marker}")),
        "{mode:?}: {text}"
    );
    assert_eq!(f.turns_with(&format!("fin-{marker}")).len(), 1, "one wake");
    assert_eq!(f.reports(&mut ws, &format!("fin-{marker}")).await, 1);

    // The same job with notify off is today's held stream: the same notice, the same one wake.
    let today = E2e::start(Setup {
        notify: false,
        ..Setup::new(mode)
    })
    .await;
    let mut ws = today.start_job(&unique("today"), &job(7, marker)).await;
    let held = today
        .wake_with(&format!("fin-{marker}"), Duration::from_secs(30))
        .await
        .expect("the held end");
    assert!(today.front.seen().is_empty(), "notify off: no callbacks");
    assert_eq!(
        notice(&held, &format!("fin-{marker}")),
        text,
        "{mode:?}: the notified end's notice is today's"
    );
    assert_eq!(today.turns_with(&format!("fin-{marker}")).len(), 1);
    assert_eq!(today.reports(&mut ws, &format!("fin-{marker}")).await, 1);
}

/// Case 7: the proof never reaches the replica (the path refuses it), so the stream stays held
/// for good and the end comes through it, once (N38). The end's own callback finds the job held
/// and is answered without contacting the sandbox (N32).
async fn without_its_proof_the_stream_stays_held(mode: Mode) {
    let marker = "m7";
    let f = E2e::start(Setup::new(mode)).await;
    f.front
        .set_policy(Some(std::sync::Arc::new(|_job: &str, nth: usize| {
            if nth == 1 {
                Verdict::Answer(404)
            } else {
                Verdict::Route
            }
        })));
    let session = &unique("alpha");
    let mut ws = f.start_job(session, &job(6, marker)).await;
    let pgid = f.stream(&job(6, marker)).await;
    if mode.releases() {
        let proof = f.proof(session).await;
        assert_eq!(proof.status, 404, "the path refused the proof");
    }
    assert!(
        f.svc
            .exec
            .wait_for_end(pgid, Duration::from_secs(3))
            .await
            .is_none(),
        "{mode:?}: no proof, no release"
    );
    let wake = f
        .wake_with(&format!("fin-{marker}"), Duration::from_secs(30))
        .await
        .expect("the end through the held stream");
    assert!(notice(&wake, &format!("fin-{marker}")).contains("exited with code 0"));
    assert_eq!(f.reports(&mut ws, &format!("fin-{marker}")).await, 1);
}

in_every_mode!(
    a_released_job_is_silent_until_its_end,
    without_its_proof_the_stream_stays_held
);
