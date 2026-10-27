//! exec v1.4's release in `serve --service`: a kept job's stream is let go only after `keep` with
//! `notify` succeeded and the job's proof arrived inside its window, and after that nothing reaches
//! the sandbox about the job but what a callback, a use or a call causes. Each test names the
//! `job_notify::` check of `notify-checks.json` it is.
//!
//! The fleet here is real: a `serve --service` replica, the reference v1.4 endpoint (`exec_mock`,
//! which makes the callbacks itself), and a stand-in for the edge that forwards them to the
//! replica, as the fleet's edge routes a session's requests (`--notify-origin` names it). The
//! rules that need no fleet (the reconcile schedule, what a live reconcile may touch) are held to
//! the pure kernel, which is the function production calls.
#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use beyond_agent::tools::job_notify::kernel::{
    Action, Event, Facts, JobState, KeepOutcome, Kernel, Listed, RECONCILE_SPACING, TableJob,
    USE_SPACING,
};
use common::exec_mock::{LoseCallbacks, MockOptions, StreamEnd};
use common::notify_fleet::{
    Reap, SESSION, TENANT, bash_bg, fleet, fleet_with, post, seal_key, still_held,
};
use common::service::{Options, Service};
use common::{spawn_model_server, turn_text, turn_tool_use, ws_connect_with_headers, ws_send};
use serde_json::{Value, json};

/// N35: `keep` with `notify` alone releases nothing; the proof arriving inside the window does,
/// and the stream is closed with the job still running. The job's end then reaches the model by
/// a callback, as a streamed end would.
#[tokio::test(flavor = "multi_thread")]
async fn release_waits_for_the_first_callback() {
    // No proof ever arrives: kept with notify, still held.
    let f = fleet(
        MockOptions {
            lose_callbacks: LoseCallbacks::Proof,
            ..MockOptions::default()
        },
        "echo up; sleep 6; echo done-a1",
        None,
    )
    .await;
    let _ws = f.start_job().await;
    let pgid = f.stream("done-a1").await;
    assert!(
        f.keeps().iter().any(|k| k.get("notify").is_some()),
        "kept with notify"
    );
    assert!(
        still_held(&f, pgid, Duration::from_secs(3)).await,
        "no proof, no release"
    );

    // The proof arrives: released at once, the job still running.
    let f = fleet(
        MockOptions::default(),
        "echo up; sleep 6; echo done-a2",
        None,
    )
    .await;
    let _ws = f.start_job().await;
    let pgid = f.stream("done-a2").await;
    let end = f.svc.exec.wait_for_end(pgid, Duration::from_secs(5)).await;
    assert_eq!(
        end,
        Some(StreamEnd::Disconnected { killed: false }),
        "released"
    );
    assert!(common::exec_mock::group_alive(pgid), "the job runs on");
    assert!(
        f.svc.exec.callbacks().iter().any(|c| c.status == Some(200)),
        "the proof was answered 200: {:?}",
        f.svc.exec.callbacks()
    );
    // Its end comes back by the callback, through today's follow, to the model.
    assert!(
        f.model_saw("done-a2", Duration::from_secs(20)).await,
        "the end's notice reached the model"
    );
}

/// N38: a proof that never arrives leaves the job held for good: its end comes through its
/// stream, as today.
#[tokio::test(flavor = "multi_thread")]
async fn no_callback_keeps_the_stream() {
    let f = fleet(
        MockOptions {
            lose_callbacks: LoseCallbacks::All,
            ..MockOptions::default()
        },
        "echo up; sleep 3; echo done-b1",
        None,
    )
    .await;
    let _ws = f.start_job().await;
    let pgid = f.stream("done-b1").await;
    let end = f.svc.exec.wait_for_end(pgid, Duration::from_secs(15)).await;
    assert_eq!(end, Some(StreamEnd::Finished), "held to the end");
    assert!(f.model_saw("done-b1", Duration::from_secs(15)).await);
}

/// N36: no notify without the edge's origin, or from an endpoint that does not offer it: `keep` is
/// sent without `notify`, exactly as today.
#[tokio::test(flavor = "multi_thread")]
async fn keep_without_notify_when_unavailable() {
    // The endpoint does not advertise (its deployment does not say kept jobs survive idling).
    let f = fleet(
        MockOptions::default(),
        "echo up; sleep 30; echo done-c1",
        None,
    )
    .await;
    let _ws = f.start_job().await;
    let pgid = f.stream("done-c1").await;
    let _reap = Reap(pgid);
    assert!(still_held(&f, pgid, Duration::from_secs(1)).await);
    let keeps = f.keeps();
    assert!(!keeps.is_empty());
    assert!(keeps.iter().all(|k| k.get("notify").is_none()), "{keeps:?}");

    // No origin configured: nothing to name, so no notify, whatever the endpoint offers.
    let mut turns = vec![bash_bg("echo up; sleep 3; echo done-c2"), turn_text("ok")];
    turns.extend((0..4).map(|_| turn_text("noted")));
    let (base, _r) = spawn_model_server(turns);
    let svc = Service::start_with(
        &base,
        &["s1"],
        Options {
            exec: Some(MockOptions::default()),
            ..Default::default()
        },
    )
    .await;
    let token = svc.token(TENANT, SESSION);
    let mut ws = ws_connect_with_headers(svc.port, Some(SESSION), &svc.header(&token))
        .await
        .unwrap();
    ws_send(&mut ws, json!({"type":"prompt","id":"p","message":"go"})).await;
    common::ws_read_until_response(&mut ws, "prompt").await;
    svc.exec
        .wait_for_stream("done-c2", Duration::from_secs(20))
        .await
        .unwrap();
    let keeps: Vec<Value> = svc
        .exec
        .bodies()
        .into_iter()
        .filter(|b| b["job"]["op"] == "keep")
        .collect();
    assert!(!keeps.is_empty());
    assert!(
        keeps.iter().all(|k| k["job"].get("notify").is_none()),
        "{keeps:?}"
    );
    assert!(svc.exec.callbacks().is_empty());
}

/// N37: the window opens before `keep` is sent, so a proof that overtakes `keep`'s answer counts.
#[tokio::test(flavor = "multi_thread")]
async fn proof_window_opens_before_keep() {
    let f = fleet(
        MockOptions {
            keep_delay: Duration::from_millis(3000),
            ..MockOptions::default()
        },
        "echo up; sleep 8; echo done-d1",
        None,
    )
    .await;
    let _ws = f.start_job().await;
    let pgid = f.stream("done-d1").await;
    let end = f.svc.exec.wait_for_end(pgid, Duration::from_secs(8)).await;
    assert_eq!(end, Some(StreamEnd::Disconnected { killed: false }));
    // The proof was answered (before `keep` was: the endpoint held `keep`'s answer back), and on
    // its first attempt.
    let proof = f
        .svc
        .exec
        .callbacks()
        .into_iter()
        .find(|c| c.status == Some(200))
        .unwrap();
    assert_eq!(proof.n, 1);
}

/// N38: a refused `notify` (`400`) is answered with a plain `keep`, and the job stays held.
#[tokio::test(flavor = "multi_thread")]
async fn a_refused_notify_falls_back() {
    // An origin the endpoint refuses: plain http to a host that is not loopback.
    let f = fleet(
        MockOptions::default(),
        "echo up; sleep 3; echo done-e1",
        Some("http://edge.example"),
    )
    .await;
    let _ws = f.start_job().await;
    let pgid = f.stream("done-e1").await;
    let keeps = f.keeps();
    assert!(keeps.len() >= 2, "{keeps:?}");
    assert!(keeps[0].get("notify").is_some());
    assert!(keeps[1].get("notify").is_none(), "kept again without it");
    let end = f.svc.exec.wait_for_end(pgid, Duration::from_secs(15)).await;
    assert_eq!(end, Some(StreamEnd::Finished), "held to the end");
}

/// N39: once a job is unheld nothing is sent about it on any timer: no heartbeat, no poll, no
/// list, until a callback or a use.
#[tokio::test(flavor = "multi_thread")]
async fn silent_after_release() {
    let f = fleet(MockOptions::default(), "echo up; sleep 300", None).await;
    let _ws = f.start_job().await;
    let pgid = f.stream("sleep 300").await;
    let _reap = Reap(pgid);
    f.svc
        .exec
        .wait_for_end(pgid, Duration::from_secs(5))
        .await
        .unwrap();
    // The release's own reconcile.
    tokio::time::sleep(Duration::from_millis(500)).await;
    let before = f.svc.exec.requests().len();
    tokio::time::sleep(Duration::from_secs(7)).await;
    let after = f.svc.exec.requests();
    assert_eq!(
        after.len(),
        before,
        "sent after release: {:?}",
        &after[before..]
    );
}

/// N42: a live reconcile never keeps; only an open reconcile does.
#[tokio::test(flavor = "multi_thread")]
async fn a_live_reconcile_never_keeps() {
    let f = fleet(MockOptions::default(), "echo up; sleep 300", None).await;
    let _ws = f.start_job().await;
    let pgid = f.stream("sleep 300").await;
    let _reap = Reap(pgid);
    f.svc
        .exec
        .wait_for_end(pgid, Duration::from_secs(5))
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(1500)).await;
    let keeps = f.keeps().len();
    let lists = f.ops("list");
    // A callback for the unheld job, with its real token, through the edge.
    let job = f.keeps()[0]["id"].as_str().unwrap().to_string();
    let scope = beyond_agent::exec_endpoint::jobs::JobScope::for_session(Some(TENANT), SESSION);
    let key = seal_key(&f.svc);
    let token = key.mint(
        TENANT,
        SESSION,
        &beyond_agent::exec_endpoint::jobs::JobId::parse(&job).unwrap(),
    );
    let body = format!(
        r#"{{"type":"job_ended","scope":"{}","id":"{job}"}}"#,
        scope.name()
    );
    let status = post(
        &f.edge.origin,
        &format!("/_beyond/agent?session_id={SESSION}&tenant={TENANT}"),
        &token,
        &body,
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(f.ops("list"), lists + 1, "one live reconcile, list only");
    assert_eq!(f.keeps().len(), keeps, "and no keep");
}

/// N42: a session that moves keeps its running job again, with notify, and proves the path anew.
#[tokio::test(flavor = "multi_thread")]
async fn takeover_proves_again() {
    let mut f = fleet(MockOptions::default(), "echo up; sleep 300", None).await;
    let ws = f.start_job().await;
    let pgid = f.stream("sleep 300").await;
    let _reap = Reap(pgid);
    f.svc
        .exec
        .wait_for_end(pgid, Duration::from_secs(5))
        .await
        .unwrap();
    // The replica dies; the session opens on another, which the edge now routes to.
    drop(ws);
    f.svc.child.kill().unwrap();
    let _ = f.svc.child.wait();
    let peer = f
        .svc
        .start_peer(&["--notify-origin", &f.edge.origin.clone()]);
    f.edge.target.store(peer.port, Ordering::SeqCst);
    let keeps_before = f.keeps().len();
    let token = f.svc.token(TENANT, SESSION);
    let mut ws = loop {
        match ws_connect_with_headers(peer.port, Some(SESSION), &f.svc.header(&token)).await {
            Ok(ws) => break ws,
            Err(_) => tokio::time::sleep(Duration::from_millis(500)).await,
        }
    };
    // A real use of the sandbox runs the open reconcile.
    ws_send(&mut ws, json!({"type":"bash","id":"b","command":"true"})).await;
    common::ws_read_until_response(&mut ws, "bash").await;
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while f.keeps().len() == keeps_before && std::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let keeps = f.keeps();
    assert!(keeps.len() > keeps_before, "kept again on the new owner");
    assert!(keeps.last().unwrap().get("notify").is_some(), "with notify");
    // And proved again: a second proof answered 200.
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        let ok = f
            .svc
            .exec
            .callbacks()
            .iter()
            .filter(|c| c.status == Some(200))
            .count();
        if ok >= 2 {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "{:?}",
            f.svc.exec.callbacks()
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    drop(peer);
}

/// N40: the exec runner's HTTP client closes a connection idle for 30 s.
#[tokio::test(flavor = "multi_thread")]
async fn no_idle_connection_to_the_endpoint() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = beyond_test_support::ports::tokio_listener().await;
    let url = format!("http://{}/exec", listener.local_addr().unwrap());
    let closed = Arc::new(tokio::sync::Notify::new());
    let seen = closed.clone();
    tokio::spawn(async move {
        let (mut sock, _) = listener.accept().await.unwrap();
        let mut buf = vec![0u8; 8192];
        let _ = sock.read(&mut buf).await.unwrap();
        let body = r#"{"exit_code":0,"stdout":"","stderr":""}"#;
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        sock.write_all(resp.as_bytes()).await.unwrap();
        // Keep-alive: the connection stays pooled until the client closes it.
        loop {
            match sock.read(&mut buf).await {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
        }
        seen.notify_one();
    });
    let runner = beyond_agent::exec_endpoint::HttpExecRunner::new(url).unwrap();
    use beyond_agent::tools::exec::CommandRunner as _;
    runner
        .run("true", &[], None, Duration::from_secs(5))
        .await
        .unwrap();
    let t0 = std::time::Instant::now();
    tokio::time::timeout(Duration::from_secs(45), closed.notified())
        .await
        .expect("the idle connection was never closed");
    let idle = t0.elapsed();
    assert!(
        idle >= Duration::from_secs(25) && idle < Duration::from_secs(40),
        "{idle:?}"
    );
}

/// Send `message` as a prompt and wait for its response.
async fn prompt(ws: &mut common::TestWs, message: &str) {
    ws_send(ws, json!({"type":"prompt","id":"p2","message":message})).await;
    let frames = common::ws_read_until_response(ws, "prompt").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:?}");
}

/// Wait (bounded) until the endpoint's stream for `pgid` was let go of (released).
async fn released(f: &common::notify_fleet::Fleet, pgid: u32) {
    let end = f.svc.exec.wait_for_end(pgid, Duration::from_secs(8)).await;
    assert_eq!(
        end,
        Some(StreamEnd::Disconnected { killed: false }),
        "released"
    );
}

/// N39: a `bash_job` call about an unheld job reads in what the sandbox holds (so even a peek
/// shows what it printed while unheld), follows it for the call, and lets go again when the call
/// puts it back running.
#[tokio::test(flavor = "multi_thread")]
async fn a_call_lets_go_again() {
    let mut turns = vec![
        bash_bg("echo up; sleep 2; echo while-unheld-q7; sleep 300"),
        turn_text("started"),
        turn_tool_use(
            "t2",
            "bash_job",
            &json!({"job": 1, "timeout_ms": 0}).to_string(),
        ),
        turn_text("peeked"),
    ];
    turns.extend((0..6).map(|i| turn_text(&format!("noted {i}"))));
    let f = fleet_with(MockOptions::default(), turns, None).await;
    let mut ws = f.start_job().await;
    let pgid = f.stream("while-unheld-q7").await;
    let _reap = Reap(pgid);
    released(&f, pgid).await;
    // It prints while unheld; nothing reads it.
    tokio::time::sleep(Duration::from_secs(4)).await;
    let reads_before = f.ops("read");
    prompt(&mut ws, "peek at it").await;
    assert!(f.ops("read") > reads_before, "the call read it in");
    assert!(
        f.model_saw("while-unheld-q7", Duration::from_secs(5)).await,
        "the peek showed what it printed while unheld"
    );
    // The call returned with the job running: the follow is closed, and nothing is sent.
    tokio::time::sleep(Duration::from_millis(1500)).await;
    let before = f.svc.exec.requests().len();
    tokio::time::sleep(Duration::from_secs(6)).await;
    let after = f.svc.exec.requests();
    assert_eq!(
        after.len(),
        before,
        "sent after the call let go: {:?}",
        &after[before..]
    );
    assert!(common::exec_mock::group_alive(pgid), "and the job runs on");
}

/// A session reaped while its job is unheld behaves as today: callbacks find no session (`503`)
/// and nothing opens; the session reads the job's end back when it next uses its sandbox, once.
#[tokio::test(flavor = "multi_thread")]
async fn a_reaped_session_reads_its_jobs_back_on_reopen() {
    let mut turns = vec![
        bash_bg("echo up; sleep 5; printf 'fin-%s\\n' r4"),
        turn_text("started"),
        turn_tool_use("t2", "bash", &json!({"command": "true"}).to_string()),
        turn_text("checked"),
    ];
    turns.extend((0..6).map(|i| turn_text(&format!("noted {i}"))));
    let (base, requests) = spawn_model_server(turns);
    let (edge, listener) = common::notify_fleet::Edge::bind().await;
    let svc = Service::start_with(
        &base,
        &["s1"],
        Options {
            extra_args: vec![
                "--notify-origin".into(),
                edge.origin.clone(),
                "--session-idle-timeout".into(),
                "1".into(),
            ],
            exec: Some(MockOptions::default()),
            ..Default::default()
        },
    )
    .await;
    edge.target
        .store(svc.port, std::sync::atomic::Ordering::SeqCst);
    edge.serve(listener);
    let f = common::notify_fleet::Fleet {
        svc,
        edge,
        requests,
    };
    let ws = f.start_job().await;
    let pgid = f.stream("fin-%s").await;
    released(&f, pgid).await;
    drop(ws);
    // The job ends with no session open: its callbacks are answered 503.
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    while !f.svc.exec.callbacks().iter().any(|c| c.status == Some(503)) {
        assert!(
            std::time::Instant::now() < deadline,
            "{:?}",
            f.svc.exec.callbacks()
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    // Reopen, and use the sandbox: the end is read back and reported once.
    let token = f.svc.token(TENANT, SESSION);
    let mut ws = ws_connect_with_headers(f.svc.port, Some(SESSION), &f.svc.header(&token))
        .await
        .unwrap();
    prompt(&mut ws, "check").await;
    assert!(
        f.model_saw("fin-r4", Duration::from_secs(10)).await,
        "the end was reported"
    );
    let last = f.requests.lock().unwrap().last().cloned().unwrap();
    assert_eq!(last.matches("fin-r4").count(), 1, "once: {last}");
}

/// `new_session` while a job is unheld ends it with its transcript, as it ends any job today; its
/// callbacks then name a job the session does not know (`410`).
#[tokio::test(flavor = "multi_thread")]
async fn new_session_ends_the_unheld_job() {
    let f = fleet(MockOptions::default(), "echo up; sleep 300", None).await;
    let mut ws = f.start_job().await;
    let pgid = f.stream("sleep 300").await;
    let _reap = Reap(pgid);
    released(&f, pgid).await;
    let job = f.keeps()[0]["id"].as_str().unwrap().to_string();
    ws_send(&mut ws, json!({"type":"new_session","id":"n"})).await;
    common::ws_read_until_response(&mut ws, "new_session").await;
    assert!(
        common::exec_mock::wait_group_gone(pgid, Duration::from_secs(15)).await,
        "the old transcript's job was ended"
    );
    // The job's group being gone is not the switch being done: the session stops the old jobs,
    // then tells the kernel the transcript changed, then wipes the old scope. The wipe is the
    // sandbox seeing that last step, so a callback sent after it meets a finished switch. Before
    // it the session is mid-switch and may answer 503, which the sandbox retries (N24), as for
    // any callback that arrives while its job is being taken.
    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    while !f
        .svc
        .exec
        .requests()
        .iter()
        .any(|r| r.body["job"]["op"] == "wipe")
    {
        assert!(
            std::time::Instant::now() < deadline,
            "the old scope was never wiped"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let key = seal_key(&f.svc);
    let id = beyond_agent::exec_endpoint::jobs::JobId::parse(&job).unwrap();
    let scope = beyond_agent::exec_endpoint::jobs::JobScope::for_session(Some(TENANT), SESSION);
    let body = format!(
        r#"{{"type":"job_ended","scope":"{}","id":"{job}"}}"#,
        scope.name()
    );
    let status = post(
        &f.edge.origin,
        &format!("/_beyond/agent?session_id={SESSION}&tenant={TENANT}"),
        &key.mint(TENANT, SESSION, &id),
        &body,
    )
    .await;
    assert_eq!(status, 410);
}

// ---- the kernel's rules, which need no fleet

const J: &str = "0123456789abcdef";
const K: &str = "fedcba9876543210";

fn running() -> Facts {
    Facts {
        open_reconcile_done: true,
        job: TableJob::Running,
    }
}

/// A kernel with `J` unheld (through the release row) and its release reconcile run out at `t`.
fn unheld_at(t: u64) -> Kernel {
    let mut k = Kernel::new();
    k.step(Event::Kept {
        job: J.into(),
        notify_available: true,
        adopted: false,
    });
    k.step(Event::KeepAnswered {
        job: J.into(),
        outcome: KeepOutcome::Succeeded,
        now: 0,
    });
    k.step(Event::Callback {
        call: 0,
        job: J.into(),
        facts: running(),
        now: 0,
    });
    let a = k.step(Event::StreamClosed {
        job: J.into(),
        offset: 9,
        shell_exit_seen: false,
        now: 0,
    });
    assert_eq!(a, [Action::StartReconcile { reconcile: 0 }]);
    k.step(Event::Listed {
        reconcile: 0,
        jobs: Some(BTreeMap::from([(J.to_string(), Listed::Running)])),
        now: t,
    });
    k
}

/// N28: a live reconcile acts only on unheld jobs, and only on the transcript it began on.
#[test]
fn the_live_reconcile_changes_only_unheld_jobs() {
    let mut k = unheld_at(0);
    // `K` is proving (held): a listing that says it ended changes nothing about it.
    k.step(Event::Kept {
        job: K.into(),
        notify_available: true,
        adopted: false,
    });
    let a = k.step(Event::Use { now: USE_SPACING });
    assert_eq!(a, [Action::StartReconcile { reconcile: 1 }]);
    let ended = BTreeMap::from([
        (J.to_string(), Listed::Ended { code: 3, at: 100 }),
        (K.to_string(), Listed::Ended { code: 0, at: 100 }),
    ]);
    let a = k.step(Event::Listed {
        reconcile: 1,
        jobs: Some(ended.clone()),
        now: USE_SPACING + 10,
    });
    assert_eq!(
        a,
        [Action::Follow {
            job: J.into(),
            from: 9,
            end: Some(beyond_agent::tools::job_notify::kernel::ListedEnd {
                code: Some(3),
                at: 100
            })
        }]
    );
    assert!(
        matches!(k.job(K), Some(JobState::Proving { .. })),
        "untouched"
    );
    assert_eq!(k.job(J), None, "held again, by its follow");

    // A transcript switch while a reconcile runs: it hands nothing over.
    let mut k = unheld_at(0);
    k.step(Event::Use { now: USE_SPACING });
    k.step(Event::TranscriptSwitched);
    let a = k.step(Event::Listed {
        reconcile: 1,
        jobs: Some(ended),
        now: USE_SPACING + 10,
    });
    assert!(
        a.iter().all(|x| !matches!(x, Action::Follow { .. })),
        "{a:?}"
    );

    // A taken job is a call's: a listing leaves it alone.
    let mut k = unheld_at(0);
    k.step(Event::Taken { job: J.into() });
    k.step(Event::Use { now: USE_SPACING });
    let a = k.step(Event::Listed {
        reconcile: 1,
        jobs: Some(BTreeMap::new()),
        now: USE_SPACING + 10,
    });
    assert!(a.is_empty(), "{a:?}");
    assert!(matches!(k.job(J), Some(JobState::Taken { .. })));
}

/// N29: a live reconcile starts only for a callback about an unheld job, a release, or a use (at
/// once, or at the 30 s mark when one ended less than 30 s before); never on a timer alone.
#[test]
fn reconciles_only_on_a_callback_or_a_use() {
    let mut k = unheld_at(1000);
    // Time alone starts nothing.
    for now in (1000..200_000).step_by(997) {
        assert!(k.step(Event::Tick { now }).is_empty(), "at {now}");
    }
    // A use 30 s or more after the last reconcile's end starts one at once.
    let mut k = unheld_at(1000);
    assert_eq!(
        k.step(Event::Use {
            now: 1000 + USE_SPACING
        }),
        [Action::StartReconcile { reconcile: 1 }]
    );
    // A use inside the 30 s starts nothing now, and asks for one at the 30 s mark; more uses
    // meanwhile share it.
    let mut k = unheld_at(1000);
    assert!(
        k.step(Event::Use {
            now: 1000 + USE_SPACING - 1
        })
        .is_empty()
    );
    assert!(
        k.step(Event::Use {
            now: 1000 + USE_SPACING - 1
        })
        .is_empty()
    );
    assert_eq!(k.next_wake(), Some(1000 + USE_SPACING));
    assert_eq!(
        k.step(Event::Tick {
            now: 1000 + USE_SPACING
        }),
        [Action::StartReconcile { reconcile: 1 }]
    );
    // No unheld job: a use starts nothing.
    let mut k = Kernel::new();
    assert!(
        k.step(Event::Use {
            now: 10 * USE_SPACING
        })
        .is_empty()
    );
    // A callback about a held job contacts nothing.
    let mut k = unheld_at(1000);
    let a = k.step(Event::Callback {
        call: 7,
        job: K.into(),
        facts: running(),
        now: 5000,
    });
    assert_eq!(
        a,
        [Action::Answer {
            call: 7,
            status: 200,
            why: beyond_agent::tools::job_notify::kernel::Why::Hint
        }]
    );
    // One about the unheld job does, after the spacing.
    let a = k.step(Event::Callback {
        call: 8,
        job: J.into(),
        facts: running(),
        now: 1000 + RECONCILE_SPACING - 1,
    });
    assert!(a.is_empty());
    let a = k.step(Event::Tick {
        now: 1000 + RECONCILE_SPACING,
    });
    assert_eq!(a, [Action::StartReconcile { reconcile: 1 }]);
}
