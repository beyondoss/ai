//! exec v1.4's release in `serve --service`: a kept job's stream is let go only after `keep` with
//! `notify` succeeded and the job's proof arrived inside its window, and after that nothing reaches
//! the sandbox about the job but what a callback, a use or a call causes. Each test names the
//! `job_notify::` check of `v1.4-checks.json` it is.
//!
//! The fleet here is real: a `serve --service` replica, the reference v1.4 endpoint (`exec_mock`,
//! which makes the callbacks itself), and a stand-in for the edge that forwards them to the
//! replica, as the fleet's edge routes a session's requests (`--notify-origin` names it). The
//! rules that need no fleet (the reconcile schedule, what a live reconcile may touch) are held to
//! the pure kernel, which is the function production calls.
#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicUsize, Ordering};
use std::time::Duration;

use beyond_ai_agent::tools::job_notify::kernel::{
    Action, Event, Facts, JobState, KeepOutcome, Kernel, Listed, RECONCILE_SPACING, TableJob,
    USE_SPACING,
};
use common::exec_mock::{LoseCallbacks, MockOptions, StreamEnd};
use common::service::{Options, Service};
use common::{spawn_model_server, turn_text, turn_tool_use, ws_connect_with_headers, ws_send};
use serde_json::{Value, json};

const TENANT: &str = "00tenant1";
const SESSION: &str = "s1.alpha";

/// The edge's stand-in: forwards each connection to the replica it currently points at, and can
/// drop every callback instead (a broken path).
struct Edge {
    origin: String,
    target: Arc<AtomicU16>,
    drop_all: Arc<AtomicBool>,
    forwarded: Arc<AtomicUsize>,
}

impl Edge {
    async fn bind() -> (Self, tokio::net::TcpListener) {
        let listener = beyond_ai_test_support::ports::tokio_listener().await;
        let origin = format!("http://{}", listener.local_addr().unwrap());
        (
            Self {
                origin,
                target: Arc::new(AtomicU16::new(0)),
                drop_all: Arc::new(AtomicBool::new(false)),
                forwarded: Arc::new(AtomicUsize::new(0)),
            },
            listener,
        )
    }

    fn serve(&self, listener: tokio::net::TcpListener) {
        let (target, drop_all, forwarded) = (
            self.target.clone(),
            self.drop_all.clone(),
            self.forwarded.clone(),
        );
        tokio::spawn(async move {
            while let Ok((mut inbound, _)) = listener.accept().await {
                if drop_all.load(Ordering::SeqCst) {
                    drop(inbound);
                    continue;
                }
                forwarded.fetch_add(1, Ordering::SeqCst);
                let port = target.load(Ordering::SeqCst);
                tokio::spawn(async move {
                    if let Ok(mut out) = tokio::net::TcpStream::connect(("127.0.0.1", port)).await {
                        let _ = tokio::io::copy_bidirectional(&mut inbound, &mut out).await;
                    }
                });
            }
        });
    }
}

/// A replica, its endpoint, and the edge before it.
struct Fleet {
    svc: Service,
    edge: Edge,
    requests: Arc<std::sync::Mutex<Vec<String>>>,
}

fn bash_bg(command: &str) -> String {
    turn_tool_use(
        "t1",
        "bash",
        &json!({"command": command, "run_in_background": true}).to_string(),
    )
}

/// A fleet whose model starts `command` in the background on the first prompt, then answers every
/// later request with text.
async fn fleet(mock: MockOptions, command: &str, origin: Option<&str>) -> Fleet {
    let mut turns = vec![bash_bg(command), turn_text("started")];
    turns.extend((0..8).map(|i| turn_text(&format!("noted {i}"))));
    let (base, requests) = spawn_model_server(turns);
    let (edge, listener) = Edge::bind().await;
    let origin = origin.map_or_else(|| edge.origin.clone(), str::to_string);
    let svc = Service::start_with(
        &base,
        &["s1"],
        Options {
            extra_args: vec!["--notify-origin".into(), origin],
            exec: Some(mock),
            ..Default::default()
        },
    )
    .await;
    edge.target.store(svc.port, Ordering::SeqCst);
    edge.serve(listener);
    Fleet {
        svc,
        edge,
        requests,
    }
}

impl Fleet {
    /// Open the session and run the prompt that starts the job; the client stays attached.
    async fn start_job(&self) -> common::TestWs {
        let token = self.svc.token(TENANT, SESSION);
        let mut ws =
            ws_connect_with_headers(self.svc.port, Some(SESSION), &self.svc.header(&token))
                .await
                .expect("a valid grant opens its session");
        ws_send(
            &mut ws,
            json!({"type":"prompt","id":"p","message":"start it"}),
        )
        .await;
        let frames = common::ws_read_until_response(&mut ws, "prompt").await;
        assert_eq!(frames.last().unwrap()["success"], true, "{frames:?}");
        ws
    }

    /// The job's held stream, by a needle of its command.
    async fn stream(&self, needle: &str) -> u32 {
        self.svc
            .exec
            .wait_for_stream(needle, Duration::from_secs(20))
            .await
            .expect("the job's stream")
            .pgid
    }

    /// Every `keep` the endpoint received.
    fn keeps(&self) -> Vec<Value> {
        self.svc
            .exec
            .bodies()
            .into_iter()
            .filter(|b| b["job"]["op"] == "keep")
            .map(|b| b["job"].clone())
            .collect()
    }

    fn ops(&self, op: &str) -> usize {
        self.svc
            .exec
            .bodies()
            .iter()
            .filter(|b| b["job"]["op"] == op)
            .count()
    }

    /// Wait (bounded) for a model request that mentions `needle`.
    async fn model_saw(&self, needle: &str, within: Duration) -> bool {
        let deadline = std::time::Instant::now() + within;
        while std::time::Instant::now() < deadline {
            if self
                .requests
                .lock()
                .unwrap()
                .iter()
                .any(|r| r.contains(needle))
            {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        false
    }
}

/// Whether the endpoint's stream for `pgid` is still open after `within`.
async fn still_held(f: &Fleet, pgid: u32, within: Duration) -> bool {
    f.svc.exec.wait_for_end(pgid, within).await.is_none()
}

/// N35: `keep` with `notify` alone releases nothing; the proof arriving inside the window does,
/// and the stream is closed with the job still running. The job's end then reaches the model by
/// a callback, as a streamed end would.
#[tokio::test(flavor = "multi_thread")]
async fn release_waits_for_the_first_callback() {
    // No proof ever arrives: kept with notify, still held.
    let f = fleet(
        MockOptions {
            lose_callbacks: LoseCallbacks::Proof,
            ..MockOptions::notifying()
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
        MockOptions::notifying(),
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
            ..MockOptions::notifying()
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
        "echo up; sleep 3; echo done-c1",
        None,
    )
    .await;
    let _ws = f.start_job().await;
    let pgid = f.stream("done-c1").await;
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
            exec: Some(MockOptions::notifying()),
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
            ..MockOptions::notifying()
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
        MockOptions::notifying(),
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
    let f = fleet(MockOptions::notifying(), "echo up; sleep 300", None).await;
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
    let f = fleet(MockOptions::notifying(), "echo up; sleep 300", None).await;
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
    let scope = beyond_ai_agent::exec_endpoint::jobs::JobScope::for_session(Some(TENANT), SESSION);
    let key = seal_key(&f.svc);
    let token = key.mint(
        TENANT,
        SESSION,
        &beyond_ai_agent::exec_endpoint::jobs::JobId::parse(&job).unwrap(),
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
    let mut f = fleet(MockOptions::notifying(), "echo up; sleep 300", None).await;
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
    let listener = beyond_ai_test_support::ports::tokio_listener().await;
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
    let runner = beyond_ai_agent::exec_endpoint::HttpExecRunner::new(url).unwrap();
    use beyond_ai_agent::tools::exec::CommandRunner as _;
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
            end: Some(beyond_ai_agent::tools::job_notify::kernel::ListedEnd {
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

/// N29: a live reconcile starts only for a callback about an unheld job, a release, or a use
/// with none in the last 30 s; never on a timer.
#[test]
fn reconciles_only_on_a_callback_or_a_use() {
    let mut k = unheld_at(1000);
    // Time alone starts nothing.
    for now in (1000..200_000).step_by(997) {
        assert!(k.step(Event::Tick { now }).is_empty(), "at {now}");
    }
    // A use within 30 s of the last reconcile's end starts nothing; one after does.
    let mut k = unheld_at(1000);
    assert!(
        k.step(Event::Use {
            now: 1000 + USE_SPACING - 1
        })
        .is_empty()
    );
    assert_eq!(
        k.step(Event::Use {
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
            status: 200
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

// ---- helpers

/// Kills a job's group a failed assertion would leave running.
struct Reap(u32);

impl Drop for Reap {
    fn drop(&mut self) {
        let _ = std::process::Command::new("kill")
            .args(["-KILL", &format!("-{}", self.0)])
            .status();
    }
}

fn seal_key(svc: &Service) -> beyond_ai_agent::notify_token::NotifyKey {
    use base64::Engine as _;
    let b64 = std::fs::read_to_string(svc.minter.seal_key()).unwrap();
    let bytes: [u8; 32] = base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .unwrap()
        .try_into()
        .unwrap();
    beyond_ai_agent::notify_token::NotifyKey::from_secret_bytes(&bytes)
}

/// POST a callback to `origin` + `target`; the status.
async fn post(origin: &str, target: &str, token: &str, body: &str) -> u16 {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let addr = origin.trim_start_matches("http://");
    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
    let req = format!(
        "POST {target} HTTP/1.1\r\nHost: {addr}\r\nAuthorization: Bearer {token}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).await.unwrap();
    let mut out = Vec::new();
    let _ = tokio::time::timeout(Duration::from_secs(30), s.read_to_end(&mut out)).await;
    String::from_utf8_lossy(&out)
        .split(' ')
        .nth(1)
        .and_then(|c| c.parse().ok())
        .unwrap_or(0)
}
