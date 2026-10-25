//! exec v1.4 end to end, ends no callback delivers: the one accepted difference. After a job is
//! let go of, an end whose callback is lost is noticed at the session's next real use of its
//! sandbox or its next open, and reported once; a sandbox that dies is noticed the same way, and
//! then takes today's error path. Native and baseline endpoints only: with no egress nothing is
//! let go of, so nothing can be lost.
#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use common::exec_mock::LoseCallbacks;
use common::notify_e2e::{
    E2e, Mode, SeenCallback, Setup, Verdict, holds, in_notify_modes, notice, unique, use_sandbox,
    wait_reaped,
};

fn job(secs: u32, marker: &str) -> String {
    format!("echo up; sleep {secs}; printf 'fin-%s\\n' {marker}")
}

/// The spacing between a session's reconciles below which a real use starts none (N29).
const USE_SPACING: Duration =
    Duration::from_millis(beyond_ai_agent::tools::job_notify::kernel::USE_SPACING);

/// A fleet whose path loses every callback after the job's proof: the native endpoint's own
/// sender drops them, and the baseline's are refused at the front (the supervisor stops).
async fn losing(mode: Mode, idle_timeout: Option<u64>) -> E2e {
    let f = E2e::start(Setup {
        idle_timeout,
        tweak: if mode == Mode::Native {
            |m| m.lose_callbacks = LoseCallbacks::Changes
        } else {
            |_| {}
        },
        ..Setup::new(mode)
    })
    .await;
    f.front.set_policy(Some(Arc::new(|c: &SeenCallback| {
        if c.nth == 1 {
            Verdict::Route
        } else {
            Verdict::Answer(404)
        }
    })));
    f
}

/// Start the job, see it released; the moment its release reconcile is over.
async fn released(f: &E2e, session: &str, command: &str) -> Instant {
    let pgid = f.stream(command).await;
    assert_eq!(f.proof(session).await.status, 200);
    f.released(pgid).await;
    tokio::time::sleep(Duration::from_millis(1500)).await;
    Instant::now()
}

/// Case 6, by use: the end's callback is lost, so nothing reports it, until the session's next
/// real use of its sandbox, which reports it once (N29, N34).
async fn a_lost_end_is_reported_at_the_next_use(mode: Mode) {
    let marker = "m6u";
    let session = &unique("alpha");
    let f = losing(mode, None).await;
    let mut ws = f.start_job(session, &job(5, marker)).await;
    let quiet = released(&f, session, &job(5, marker)).await;
    // The job ends; nothing tells the replica.
    assert!(
        f.wake_with(&format!("fin-{marker}"), Duration::from_secs(8))
            .await
            .is_none(),
        "{mode:?}: reported without a callback, a use or an open"
    );
    // A use within the spacing of the last reconcile starts none; one after it does.
    tokio::time::sleep(
        (quiet + USE_SPACING + Duration::from_millis(500))
            .saturating_duration_since(Instant::now()),
    )
    .await;
    use_sandbox(&mut ws).await;
    f.wake_with(&format!("fin-{marker}"), Duration::from_secs(15))
        .await
        .unwrap_or_else(|| panic!("{mode:?}: the use never noticed the end"));
    assert_eq!(f.reports(&mut ws, &format!("fin-{marker}")).await, 1);
}

/// Case 6, by open: the end's callback is lost and the session closes before any use; its next
/// open reads the end back and reports it once, as today.
async fn a_lost_end_is_reported_at_the_next_open(mode: Mode) {
    let marker = "m6o";
    let session = &unique("alpha");
    let f = losing(mode, Some(1)).await;
    let ws = f.start_job(session, &job(5, marker)).await;
    released(&f, session, &job(5, marker)).await;
    assert!(
        f.wake_with(&format!("fin-{marker}"), Duration::from_secs(8))
            .await
            .is_none(),
        "{mode:?}: reported without a callback, a use or an open"
    );
    drop(ws);
    wait_reaped(&f.svc, &[f.svc.port], session).await;
    let mut ws = f.open(session).await;
    use_sandbox(&mut ws).await;
    f.wake_with(&format!("fin-{marker}"), Duration::from_secs(15))
        .await
        .unwrap_or_else(|| panic!("{mode:?}: the open never read the end back"));
    assert_eq!(f.reports(&mut ws, &format!("fin-{marker}")).await, 1);
    assert!(holds(&f.svc, f.svc.port, session).await);
}

/// The one sanctioned exception: an unheld job's sandbox dies (unreachable, and its callbacks with
/// it). Nothing notices until the session's next real use; then the job takes today's error path,
/// with the notice a held job gets when its sandbox dies today.
async fn a_dead_sandbox_is_noticed_at_the_next_use(mode: Mode) {
    let session = &unique("alpha");
    let command = "echo up; sleep 60";
    let f = E2e::start(Setup::new(mode)).await;
    let mut ws = f.start_job(session, command).await;
    let quiet = released(&f, session, command).await;
    let model_before = f.model_requests();
    f.front
        .set_policy(Some(Arc::new(|_: &SeenCallback| Verdict::Drop)));
    f.meter.go_down();
    tokio::time::sleep(Duration::from_secs(5)).await;
    assert_eq!(
        f.model_requests(),
        model_before,
        "{mode:?}: noticed with no use"
    );
    tokio::time::sleep(
        (quiet + USE_SPACING + Duration::from_millis(500))
            .saturating_duration_since(Instant::now()),
    )
    .await;
    use_sandbox(&mut ws).await;
    let wake = f
        .wake_with("failed after", Duration::from_secs(90))
        .await
        .unwrap_or_else(|| panic!("{mode:?}: the use never noticed the dead sandbox"));
    let text = notice(&wake, "\n").replace(&f.meter.url, "<endpoint>");

    // Today: the same job held, and its sandbox dies the same way.
    let today = E2e::start(Setup {
        notify: false,
        ..Setup::new(mode)
    })
    .await;
    let _ws = today.start_job(&unique("today"), command).await;
    today.stream(command).await;
    tokio::time::sleep(Duration::from_millis(1500)).await;
    today.meter.go_down();
    let held = today
        .wake_with("failed after", Duration::from_secs(90))
        .await
        .expect("today's error notice");
    let held = notice(&held, "\n").replace(&today.meter.url, "<endpoint>");
    eprintln!("notified: {text:?}\ntoday:    {held:?}");
    assert_eq!(text, held, "{mode:?}: today's error path");
}

in_notify_modes!(
    a_lost_end_is_reported_at_the_next_use,
    a_lost_end_is_reported_at_the_next_open,
    a_dead_sandbox_is_noticed_at_the_next_use
);
