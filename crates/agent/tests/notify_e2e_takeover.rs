//! exec v1.4 end to end, a session that moves or closes while its job runs. Two real `serve
//! --service` replicas on a hash ring behind the fleet's front, which routes clients and callbacks
//! alike by session id, retrying and walking the ring on `503`. A callback never opens a session:
//! one that reaches a replica without the session is answered `503` and the front walks on; one
//! that finds no replica with it changes nothing, and the next open reads the end back, as today.
//! Every case runs in every endpoint mode.
#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::time::{Duration, Instant};

use common::notify_e2e::{
    E2e, Mode, SeenCallback, Setup, holds, in_every_mode, post_to, scope, session_on, use_sandbox,
    wait_reaped,
};

fn job(secs: u32, marker: &str) -> String {
    format!("echo up; sleep {secs}; printf 'fin-%s\\n' {marker}")
}

/// Wait (bounded) for a callback at the front that `pick` accepts.
async fn callback_where(
    f: &E2e,
    within: Duration,
    pick: impl Fn(&SeenCallback) -> bool,
) -> SeenCallback {
    let deadline = Instant::now() + within;
    loop {
        if let Some(c) = f.front.seen().into_iter().find(|c| pick(c)) {
            return c;
        }
        assert!(
            Instant::now() < deadline,
            "no such callback: {:?}",
            f.front.seen()
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// The job's sandbox id: its proof at the front, or (no egress) its first `keep`.
async fn job_id(f: &E2e, session: &str) -> String {
    if f.mode.releases() {
        return f.proof(session).await.job;
    }
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        if let Some(id) = f
            .svc
            .exec
            .bodies()
            .iter()
            .find(|b| b["job"]["op"] == "keep")
            .and_then(|b| b["job"]["id"].as_str().map(str::to_string))
        {
            return id;
        }
        assert!(Instant::now() < deadline, "no keep");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// Case 5: the replica that has the session is killed mid-job. The session reopens on the other
/// replica (the front walks past the dead one), whose open reconcile adopts the job and keeps it
/// again, proving the path anew (N42). The dead replica's slot comes back as a fresh replica
/// without the session: a callback reaching it gets `503`, and the front walks on to the owner.
/// The end is reported once (N34).
async fn a_killed_replica_hands_the_job_on(mode: Mode) {
    let marker = "m5";
    let session = session_on("five", 0, 2);
    let mut f = E2e::start(Setup {
        replicas: 2,
        ..Setup::new(mode)
    })
    .await;
    let ws = f.start_job(&session, &job(20, marker)).await;
    let pgid = f.stream(&job(20, marker)).await;
    let id = job_id(&f, &session).await;
    if mode.releases() {
        let proof = f.proof(&session).await;
        assert_eq!((proof.status, proof.tried.clone()), (200, vec![(0, 200)]));
        f.released(pgid).await;
    }
    tokio::time::sleep(Duration::from_millis(1500)).await;

    // The owner dies with nothing unwound; the client reconnects through the front.
    drop(ws);
    f.kill(0);
    let mut ws = f.open(&session).await;
    use_sandbox(&mut ws).await;
    assert!(
        holds(&f.svc, f.svc_port(1), &session).await,
        "it reopened on the other replica"
    );
    if mode.releases() {
        // Kept again with notify, and proved again, through the walk to the new owner.
        let reproof = callback_where(&f, Duration::from_secs(20), |c| {
            c.nth >= 2 && c.status == 200
        })
        .await;
        assert_eq!(reproof.tried.last(), Some(&(1, 200)), "{reproof:?}");
    }

    // The dead slot comes back without the session.
    let old = f.restart(0);
    let token = f.token(&session, &id);
    assert_eq!(
        post_to(old, &session, &token, &scope(&session), &id).await,
        503,
        "a replica without the session answers 503"
    );
    if mode.releases() {
        let end = callback_where(&f, Duration::from_secs(40), |c| {
            c.status == 200 && c.tried.first().is_some_and(|t| *t == (0, 503))
        })
        .await;
        assert_eq!(
            end.tried.last(),
            Some(&(1, 200)),
            "the front walked on: {end:?}"
        );
    }
    f.wake_with(&format!("fin-{marker}"), Duration::from_secs(40))
        .await
        .unwrap_or_else(|| panic!("{mode:?}: the end never reached the new owner's model"));
    assert_eq!(f.reports(&mut ws, &format!("fin-{marker}")).await, 1);
}

/// Case 9: the session is reaped before its job ends. The end's callbacks find no replica with
/// the session (`503` from every one), nothing opens it, and nothing reaches the sandbox (N24);
/// the next open reads the end back and reports it once, as today.
async fn a_closed_session_learns_of_the_end_on_reopen(mode: Mode) {
    let marker = "m9";
    let session = session_on("nine", 0, 2);
    let f = E2e::start(Setup {
        replicas: 2,
        idle_timeout: Some(1),
        ..Setup::new(mode)
    })
    .await;
    let ws = f.start_job(&session, &job(8, marker)).await;
    let pgid = f.stream(&job(8, marker)).await;
    if mode.releases() {
        assert_eq!(f.proof(&session).await.status, 200);
        f.released(pgid).await;
    }
    drop(ws);
    let ports = [f.svc_port(0), f.svc_port(1)];
    wait_reaped(&f.svc, &ports, &session).await;
    let (exec_before, model_before) = (f.exec_requests(), f.model_requests());

    if mode.releases() {
        // The end's callback walks the whole ring and finds no one.
        let end = callback_where(&f, Duration::from_secs(30), |c| c.nth >= 2).await;
        assert_eq!(end.status, 503, "{end:?}");
        assert!(end.tried.iter().all(|(_, s)| *s == 503), "{end:?}");
        assert!(
            end.tried.iter().any(|(slot, _)| *slot == 1),
            "walked: {end:?}"
        );
    } else {
        tokio::time::sleep(Duration::from_secs(9)).await;
    }
    tokio::time::sleep(Duration::from_secs(2)).await;
    for p in ports {
        assert!(
            !holds(&f.svc, p, &session).await,
            "{mode:?}: a callback opened it"
        );
    }
    assert_eq!(
        f.exec_requests(),
        exec_before,
        "{mode:?}: the sandbox was contacted"
    );
    assert_eq!(f.model_requests(), model_before, "{mode:?}: a turn ran");

    let mut ws = f.open(&session).await;
    use_sandbox(&mut ws).await;
    f.wake_with(&format!("fin-{marker}"), Duration::from_secs(20))
        .await
        .unwrap_or_else(|| panic!("{mode:?}: the reopen never read the end back"));
    assert_eq!(f.reports(&mut ws, &format!("fin-{marker}")).await, 1);
}

in_every_mode!(
    a_killed_replica_hands_the_job_on,
    a_closed_session_learns_of_the_end_on_reopen
);
