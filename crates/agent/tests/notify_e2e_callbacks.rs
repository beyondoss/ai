//! exec v1.4 end to end, callbacks that are not the sandbox's honest report: forged ones, repeats
//! and a loop. A callback is a hint: a forged one is refused before the replica looks for the
//! session and changes nothing; a repeat costs at most one reconcile a second and starts no model
//! turn; and however many arrive, the end is reported once. Every case runs in every endpoint mode,
//! through the fleet's front to a real `serve --service` replica.
#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::time::{Duration, Instant};

use beyond_ai_agent::exec_endpoint::jobs::JobId;
use common::notify_e2e::{E2e, Mode, Setup, TENANT, in_every_mode, key, scope, unique};

fn job(secs: u32, marker: &str) -> String {
    format!("echo up; sleep {secs}; printf 'fin-%s\\n' {marker}")
}

/// The job's sandbox id: from its proof at the front, or (no egress) its `keep` at the endpoint.
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

/// The job is under way: released (after its release reconcile) or, with no egress, held.
async fn under_way(f: &E2e, session: &str, command: &str) -> String {
    let pgid = f.stream(command).await;
    let id = job_id(f, session).await;
    if f.mode.releases() {
        f.released(pgid).await;
    }
    tokio::time::sleep(Duration::from_millis(1500)).await;
    id
}

/// Case 4: a forged callback (a token that is not the fleet's, another session's token, another
/// session's job) is refused `401`, the spec's status for a token that does not verify for what it names, and changes nothing: nothing reaches the
/// sandbox and no model turn starts (N23). After the end, a repeat of the real callback adds no
/// notice: the end is reported once (N32, N34).
async fn a_forged_callback_changes_nothing(mode: Mode) {
    let marker = "m4";
    let session = &unique("alpha");
    let other = &unique("other");
    let f = E2e::start(Setup::new(mode)).await;
    let mut ws = f.start_job(session, &job(8, marker)).await;
    let id = under_way(&f, session, &job(8, marker)).await;
    let job = JobId::parse(&id).unwrap();
    let k = key(&f.svc);

    let since = Instant::now();
    let (exec_before, model_before) = (f.exec_requests(), f.model_requests());
    // A token with a MAC that is not the fleet's.
    let real = k.mint(TENANT, session, &job);
    let mut forged = real.clone();
    let last = forged.pop().unwrap();
    forged.push(if last == 'A' { 'B' } else { 'A' });
    assert_eq!(f.post(session, &forged, &scope(session), &id).await, 401);
    // Another session's token for this job, sent to this session.
    let theirs = k.mint(TENANT, other, &job);
    assert_eq!(f.post(session, &theirs, &scope(session), &id).await, 401);
    // Another session's job, with its real token, naming this session's scope: the MAC covers
    // the body's scope (N44), so it does not verify.
    let their_job = JobId::fresh();
    let theirs = k.mint(TENANT, other, &their_job);
    assert_eq!(
        f.post(other, &theirs, &scope(session), their_job.as_str())
            .await,
        401
    );
    // This job's real token, naming another session's scope.
    assert_eq!(f.post(session, &real, &scope(other), &id).await, 401);
    tokio::time::sleep(Duration::from_millis(1500)).await;
    assert_eq!(
        f.exec_requests(),
        exec_before,
        "{mode:?}: reached the sandbox"
    );
    assert_eq!(f.model_requests(), model_before, "{mode:?}: started a turn");
    if mode.releases() {
        assert_eq!(f.meter.bytes_since(since), 0, "{mode:?}: bytes crossed");
    }

    // The end, then the real callback again: no second notice.
    f.wake_with(&format!("fin-{marker}"), Duration::from_secs(30))
        .await
        .expect("the end");
    for _ in 0..3 {
        let status = f.callback(session, &id).await;
        assert!(
            status == 200 || status == 410,
            "{mode:?}: a repeat after the end got {status}"
        );
    }
    tokio::time::sleep(Duration::from_millis(1500)).await;
    assert_eq!(f.turns_with(&format!("fin-{marker}")).len(), 1, "one wake");
    assert_eq!(f.reports(&mut ws, &format!("fin-{marker}")).await, 1);
}

/// Case 8: sandbox code sends `job_ended` in a loop with the job's real token while the job runs.
/// The replica runs at most one reconcile a second for it (N30) and no model turn of its own; the
/// end is still reported once.
async fn a_looped_callback_costs_one_reconcile_a_second(mode: Mode) {
    let marker = "m8";
    let session = &unique("alpha");
    let f = E2e::start(Setup::new(mode)).await;
    let mut ws = f.start_job(session, &job(12, marker)).await;
    let id = under_way(&f, session, &job(12, marker)).await;
    let token = f.token(session, &id);

    let t0 = Instant::now();
    let (exec_before, model_before) = (f.exec_requests(), f.model_requests());
    let mut sends = Vec::new();
    for i in 0..30u64 {
        let (port, token, scope, id) = (f.front.port, token.clone(), scope(session), id.clone());
        let session = session.clone();
        sends.push(tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100 * i)).await;
            common::notify_e2e::post_to(port, &session, &token, &scope, &id).await
        }));
    }
    let mut statuses = Vec::new();
    for s in sends {
        statuses.push(s.await.unwrap());
    }
    let elapsed = t0.elapsed();
    let reconciles = f.exec_requests() - exec_before;
    let bound = elapsed.as_secs() as usize + 2;
    assert!(
        reconciles <= bound,
        "{mode:?}: {reconciles} requests to the sandbox in {elapsed:?} from a loop"
    );
    assert!(
        statuses.iter().all(|s| *s == 200 || *s == 503),
        "{mode:?}: {statuses:?}"
    );
    assert!(statuses.contains(&200), "{mode:?}: {statuses:?}");
    assert_eq!(
        f.model_requests(),
        model_before,
        "{mode:?}: a turn of its own"
    );
    if !mode.releases() {
        assert_eq!(
            reconciles, 0,
            "a held job's callback never contacts the sandbox"
        );
    }

    f.wake_with(&format!("fin-{marker}"), Duration::from_secs(30))
        .await
        .expect("the end");
    assert_eq!(f.reports(&mut ws, &format!("fin-{marker}")).await, 1);
}

in_every_mode!(
    a_forged_callback_changes_nothing,
    a_looped_callback_costs_one_reconcile_a_second
);
