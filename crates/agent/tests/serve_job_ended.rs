//! exec v1.4's `job_ended` receiver in `serve --service` ("The receiver" in the `exec_endpoint`
//! module doc): how a sandbox's callback is told apart, checked, refused, and answered, and that it
//! never opens a session, never counts as a client, and never waits behind the session's run.
//!
//! Two layers. The end-to-end tests drive a real replica over raw TCP (the head classification,
//! the refusals, the front door's refusal of `job_ended`, a run in flight, an idle reap). The rest
//! drive the receiver ([`job_ended::serve`]) and the session driver over an in-memory stream with
//! a scripted job table, which is how every case of `v1.4-callback.json` is reached, released jobs
//! included.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use base64::Engine as _;
use beyond_agent::exec_endpoint::jobs::{JobId, JobScope};
use beyond_agent::notify_token::NotifyKey;
use beyond_agent::serve_ws::job_ended;
use beyond_agent::tools::job_notify::kernel::{
    Action, Event, Facts, JobState, KeepOutcome, Listed, TableJob,
};
use beyond_agent::tools::job_notify::{JobTracker, NotifyHub, SessionNotify};
use common::service::Service;
use common::{spawn_model_server, ws_connect_with_headers, ws_read_until_response, ws_send};
use futures::future::BoxFuture;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::time::Instant;

const TENANT: &str = "00tenant1";
const SESSION: &str = "s1.alpha";
const JOB: &str = "0123456789abcdef";

// ---- raw HTTP against a replica

/// One raw request; the status and the body of the answer.
async fn raw(port: u16, request: &[u8]) -> (u16, String) {
    let mut s = tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .unwrap();
    s.write_all(request).await.unwrap();
    let mut out = Vec::new();
    let _ = tokio::time::timeout(Duration::from_secs(30), s.read_to_end(&mut out)).await;
    let text = String::from_utf8_lossy(&out).into_owned();
    let status = text
        .split(' ')
        .nth(1)
        .and_then(|c| c.parse().ok())
        .unwrap_or(0);
    let body = text
        .split_once("\r\n\r\n")
        .map(|(_, b)| b.to_string())
        .unwrap_or_default();
    (status, body)
}

/// A callback request: `headers` are whole header lines; the body's length is not added.
fn request(method: &str, target: &str, headers: &[String], body: &str) -> Vec<u8> {
    let mut r = format!("{method} {target} HTTP/1.1\r\nHost: 127.0.0.1\r\n");
    for h in headers {
        r.push_str(h);
        r.push_str("\r\n");
    }
    r.push_str("\r\n");
    r.push_str(body);
    r.into_bytes()
}

fn body_for(scope: &str, id: &str) -> String {
    format!(r#"{{"type":"job_ended","scope":"{scope}","id":"{id}"}}"#)
}

fn target(tenant: &str, session: &str) -> String {
    format!("/_beyond/agent?session_id={session}&tenant={tenant}")
}

fn seal_key(svc: &Service) -> NotifyKey {
    let b64 = std::fs::read_to_string(svc.minter.seal_key()).unwrap();
    let bytes: [u8; 32] = base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .unwrap()
        .try_into()
        .unwrap();
    NotifyKey::from_secret_bytes(&bytes)
}

/// A well-formed callback for `JOB` of `session`, with a real token.
fn callback(key: &NotifyKey, tenant: &str, session: &str) -> Vec<u8> {
    let scope = JobScope::for_session(Some(tenant), session);
    let token = key.mint(tenant, session, &JobId::parse(JOB).unwrap());
    let body = body_for(scope.name(), JOB);
    request(
        "POST",
        &target(tenant, session),
        &[
            format!("Authorization: Bearer {token}"),
            "Content-Type: application/json".into(),
            format!("Content-Length: {}", body.len()),
        ],
        &body,
    )
}

async fn service() -> Service {
    let (base, _requests) = spawn_model_server(vec![]);
    Service::start(&base, &["s1"]).await
}

/// Open the session on the replica with an ordinary grant, and leave it attached.
async fn open(svc: &Service) -> common::TestWs {
    let token = svc.token(TENANT, SESSION);
    let mut ws = ws_connect_with_headers(svc.port, Some(SESSION), &svc.header(&token))
        .await
        .expect("a valid grant opens its session");
    ws_send(&mut ws, json!({"type":"get_state","id":"g"})).await;
    ws_read_until_response(&mut ws, "get_state").await;
    ws
}

fn shard_files(svc: &Service) -> Vec<std::path::PathBuf> {
    common::service::files_under(svc.shard("s1"))
}

/// N21: a request is a callback by its `Authorization` header alone, before its `Content-Length`
/// is read; a look-alike is refused `400`, however broken the rest of its head.
#[tokio::test]
async fn the_token_header_is_read_exactly() {
    let svc = service().await;
    let key = seal_key(&svc);
    let scope = JobScope::for_session(Some(TENANT), SESSION);
    let token = key.mint(TENANT, SESSION, &JobId::parse(JOB).unwrap());
    let body = body_for(scope.name(), JOB);
    let len = format!("Content-Length: {}", body.len());
    let t = target(TENANT, SESSION);
    let (_, fixed_400_body) = job_ended::refusal_body(400);

    // Callbacks, in every spelling the rule allows: no session here, so 503.
    for auth in [
        format!("Authorization: Bearer {token}"),
        format!("Authorization: bearer {token}"),
        format!("Authorization: BEARER   {token}"),
        format!("Authorization:   Bearer {token} \t"),
    ] {
        let (status, _) = raw(
            svc.port,
            &request("POST", &t, &[auth.clone(), len.clone()], &body),
        )
        .await;
        assert_eq!(status, 503, "{auth}");
    }
    // Look-alikes: 400, with the fixed body, before anything else is looked at.
    for headers in [
        vec![format!("Authorization: Bearer\t{token}"), len.clone()],
        vec![format!("Authorization: Bearer{token}"), len.clone()],
        vec![
            format!("Authorization: Bearer {token}"),
            format!("Authorization: Bearer {token}"),
            len.clone(),
        ],
        // A Content-Length today's front door would refuse with its own message.
        vec![
            format!("Authorization: Bearer\t{token}"),
            "Content-Length: not-a-number".into(),
        ],
    ] {
        let (status, got) = raw(svc.port, &request("POST", &t, &headers, &body)).await;
        assert_eq!(status, 400, "{headers:?}");
        assert_eq!(got, fixed_400_body, "{headers:?}");
    }
    // The rule reads bytes: an Authorization that is not UTF-8 but reads as a callback's is
    // refused as a look-alike, with the fixed body, never handed to the front door.
    let mut raw_req = format!("POST {t} HTTP/1.1\r\nHost: 127.0.0.1\r\n").into_bytes();
    raw_req.extend_from_slice(b"Authorization: Bearer bjn_v1.\xff\xfe\r\n");
    raw_req.extend_from_slice(format!("{len}\r\n\r\n{body}").as_bytes());
    let (status, got) = raw(svc.port, &raw_req).await;
    assert_eq!((status, got.as_str()), (400, fixed_400_body), "non-UTF-8");
    // A callback whose Content-Length is not a number is a callback first: refused by the
    // callback's own rules, with its fixed body, never the front door's message.
    let (status, got) = raw(
        svc.port,
        &request(
            "POST",
            &t,
            &[
                format!("Authorization: Bearer {token}"),
                "Content-Length: 1x".into(),
            ],
            &body,
        ),
    )
    .await;
    assert_eq!((status, got.as_str()), (400, fixed_400_body));
    // Not a callback: the front door's, which refuses a request with no grant.
    let (status, _) = raw(
        svc.port,
        &request(
            "POST",
            &t,
            &["Authorization: Basic x".into(), len.clone()],
            &body,
        ),
    )
    .await;
    assert_eq!(status, 401);
}

/// N22: `job_ended` sent any other way than as a callback is refused: `403` under a grant over
/// HTTP, a failed `response` over a WebSocket.
#[tokio::test]
async fn job_ended_is_only_a_callback() {
    let svc = service().await;
    let token = svc.token(TENANT, SESSION);
    let body = json!({"type":"job_ended","scope":"x","id":JOB}).to_string();
    let (status, _) = raw(
        svc.port,
        &request(
            "POST",
            &format!("/_beyond/agent?session_id={SESSION}"),
            &[
                format!("x-beyond-grant: {token}"),
                "Content-Type: application/json".into(),
                format!("Content-Length: {}", body.len()),
            ],
            &body,
        ),
    )
    .await;
    assert_eq!(status, 403);

    let mut ws = open(&svc).await;
    ws_send(
        &mut ws,
        json!({"type":"job_ended","id":"j","scope":"x","id2":JOB}),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "job_ended").await;
    let last = frames.last().unwrap();
    assert_eq!(last["success"], false, "{last}");
}

/// N23: every check before the session lookup, in order; a forged callback reaches nothing, and
/// leaves nothing on disk.
#[tokio::test]
async fn a_forged_callback_reaches_nothing() {
    let svc = service().await;
    let key = seal_key(&svc);
    let before = shard_files(&svc);
    let id = JobId::parse(JOB).unwrap();
    let scope = JobScope::for_session(Some(TENANT), SESSION);
    let send = |token: String, tenant: &str, session: &str, scope: &str, id: &str| {
        let body = body_for(scope, id);
        request(
            "POST",
            &target(tenant, session),
            &[
                format!("Authorization: Bearer {token}"),
                format!("Content-Length: {}", body.len()),
            ],
            &body,
        )
    };
    let good = key.mint(TENANT, SESSION, &id);
    let mut tampered = good.clone().into_bytes();
    let n = tampered.len();
    tampered[n - 2] = if tampered[n - 2] == b'A' { b'B' } else { b'A' };
    let tampered = String::from_utf8(tampered).unwrap();
    let other_secret = NotifyKey::from_secret_bytes(&[9; 32]).mint(TENANT, SESSION, &id);
    let other_session = key.mint(TENANT, "s1.beta", &id);
    let cases = [
        (send(tampered, TENANT, SESSION, scope.name(), JOB), 401),
        (
            send(good.clone(), "00tenant2", SESSION, scope.name(), JOB),
            401,
        ),
        (
            send(
                good.clone(),
                TENANT,
                SESSION,
                scope.name(),
                "fedcba9876543210",
            ),
            401,
        ),
        (send(other_secret, TENANT, SESSION, scope.name(), JOB), 503),
        (send(other_session, TENANT, SESSION, scope.name(), JOB), 401),
        // Another session's scope: the token's tag covers the scope, so it is a bad tag.
        (
            send(
                good.clone(),
                TENANT,
                SESSION,
                JobScope::for_session(Some(TENANT), "s1.beta").name(),
                JOB,
            ),
            401,
        ),
    ];
    for (req, want) in cases {
        let (status, _) = raw(svc.port, &req).await;
        assert_eq!(status, want);
    }
    assert_eq!(
        shard_files(&svc),
        before,
        "a refused callback writes nothing"
    );
}

/// N24: a callback that passes every check for a session no task here holds is answered `503`,
/// and nothing is opened, created or written.
#[tokio::test]
async fn a_callback_never_opens_a_session() {
    let svc = service().await;
    let key = seal_key(&svc);
    let before = shard_files(&svc);
    for _ in 0..3 {
        let (status, body) = raw(svc.port, &callback(&key, TENANT, SESSION)).await;
        assert_eq!(status, 503, "{body}");
    }
    assert_eq!(
        shard_files(&svc),
        before,
        "no session was opened or written"
    );
    assert!(
        !svc.sessions_dir("s1", TENANT).exists(),
        "not even the tenant's directory"
    );

    // Open it, then the same callback reaches it (the job is not the session's: 410).
    let ws = open(&svc).await;
    let (status, _) = raw(svc.port, &callback(&key, TENANT, SESSION)).await;
    assert!(
        status == 410 || status == 200,
        "the session is held now: {status}"
    );
    drop(ws);
}

/// N25: a refusal's body is fixed by its status alone.
#[tokio::test]
async fn refusals_say_nothing() {
    let svc = service().await;
    let key = seal_key(&svc);
    let id = JobId::parse(JOB).unwrap();
    let token = key.mint(TENANT, SESSION, &id);
    let scope = JobScope::for_session(Some(TENANT), SESSION);
    let body = body_for(scope.name(), JOB);
    let auth = format!("Authorization: Bearer {token}");
    let len = format!("Content-Length: {}", body.len());
    let t = target(TENANT, SESSION);
    let mut by_status: BTreeMap<u16, Vec<String>> = BTreeMap::new();
    let reqs = vec![
        request("POST", "/elsewhere", &[auth.clone(), len.clone()], &body),
        request("GET", &t, &[auth.clone(), len.clone()], &body),
        request(
            "POST",
            &format!("/_beyond/agent?session_id={SESSION}"),
            &[auth.clone(), len.clone()],
            &body,
        ),
        request("POST", &t, std::slice::from_ref(&auth), &body),
        request("POST", &t, &[auth.clone(), len.clone(), len.clone()], &body),
        request(
            "POST",
            &t,
            &[
                auth.clone(),
                len.clone(),
                "Transfer-Encoding: chunked".into(),
            ],
            &body,
        ),
        request(
            "POST",
            &t,
            &[auth.clone(), "Content-Length: 5000".into()],
            &"x".repeat(5000),
        ),
        request(
            "POST",
            &t,
            &[auth.clone(), "Content-Length: 2".into()],
            "[]",
        ),
        request(
            "POST",
            &t,
            &[format!("Authorization: Bearer\t{token}"), len.clone()],
            &body,
        ),
        callback(&NotifyKey::from_secret_bytes(&[3; 32]), TENANT, SESSION),
        request(
            "POST",
            &t,
            &[format!("Authorization: Bearer {token}x"), len.clone()],
            &body,
        ),
        callback(&key, TENANT, SESSION),
    ];
    for r in reqs {
        let (status, body) = raw(svc.port, &r).await;
        assert_ne!(status, 200);
        by_status.entry(status).or_default().push(body);
    }
    assert!(by_status.len() >= 6, "{by_status:?}");
    for (status, bodies) in by_status {
        let (_, fixed) = job_ended::refusal_body(status);
        for b in bodies {
            assert_eq!(b, fixed, "{status}");
        }
    }
}

/// N20 for the receiver, and FLEET.md's L4 pass-through note: the replica never records a
/// callback's token. Every way a callback is answered (accepted, bad tag, look-alike, not open
/// here, wrong path) is sent, then the replica's captured stderr is searched for each token.
#[tokio::test]
async fn the_replica_never_records_a_callbacks_token() {
    let svc = service().await;
    let key = seal_key(&svc);
    let id = JobId::parse(JOB).unwrap();
    let scope = JobScope::for_session(Some(TENANT), SESSION);
    let body = body_for(scope.name(), JOB);
    let len = format!("Content-Length: {}", body.len());
    let t = target(TENANT, SESSION);
    let good = key.mint(TENANT, SESSION, &id);
    let forged = NotifyKey::from_secret_bytes(&[5; 32]).mint(TENANT, SESSION, &id);
    let send = |method: &str, target: &str, auth: String| {
        request(method, target, &[auth, len.clone()], &body)
    };

    let tokens = vec![good.clone(), forged.clone(), format!("{good}x")];
    let sent = vec![
        // Not open here: 503.
        (
            send("POST", &t, format!("Authorization: Bearer {good}")),
            503,
        ),
        // A bad tag: 401.
        (
            send("POST", &t, format!("Authorization: Bearer {good}x")),
            401,
        ),
        // A look-alike: 400.
        (
            send("POST", &t, format!("Authorization: Bearer\t{good}")),
            400,
        ),
        // Not a POST: 403.
        (
            send("GET", &t, format!("Authorization: Bearer {good}")),
            403,
        ),
        // Wrong path: 404.
        (
            send(
                "POST",
                "/elsewhere",
                format!("Authorization: Bearer {good}"),
            ),
            404,
        ),
        // A key this replica does not hold: its kid is unknown here.
        (
            send("POST", &t, format!("Authorization: Bearer {forged}")),
            503,
        ),
    ];
    for (req, want) in sent {
        let (status, _) = raw(svc.port, &req).await;
        assert_eq!(status, want);
    }
    // Accepted: the session is open here, so the token is checked and the callback answered.
    let ws = open(&svc).await;
    let (status, _) = raw(svc.port, &callback(&key, TENANT, SESSION)).await;
    assert!(status == 200 || status == 410, "{status}");
    drop(ws);

    let logged =
        std::fs::read_to_string(&svc.stderr_log).expect("the replica's stderr was captured");
    for token in tokens {
        let secret = token.strip_prefix("bjn_v1.").unwrap_or(&token);
        assert!(
            !logged.contains(secret),
            "a token reached the replica's stderr:\n{logged}"
        );
    }
}

/// N26: served while a run is in flight, with a status of its own, never the busy `200`.
#[tokio::test]
async fn served_during_a_run() {
    let base =
        common::spawn_model_server_with_stalled_response(vec![], Duration::from_secs(15), vec![]);
    let svc = Service::start(&base, &["s1"]).await;
    let key = seal_key(&svc);
    let mut ws = open(&svc).await;
    ws_send(&mut ws, json!({"type":"prompt","id":"p","message":"go"})).await;
    // Let the run reach the model.
    tokio::time::sleep(Duration::from_millis(500)).await;
    let t0 = std::time::Instant::now();
    let (status, body) = raw(svc.port, &callback(&key, TENANT, SESSION)).await;
    assert_eq!(status, 410, "the job is not the session's: {body}");
    assert!(!body.contains("busy"), "{body}");
    assert!(
        t0.elapsed() < Duration::from_secs(5),
        "not queued behind the run"
    );
    ws_send(&mut ws, json!({"type":"abort","id":"a"})).await;
}

/// N27: a callback is not a client: it never pins the session, so a detached session is reaped
/// on its idle clock however many callbacks arrive, and after that they find no session.
#[tokio::test]
async fn the_callback_is_not_a_client() {
    let (base, _r) = spawn_model_server(vec![]);
    let svc = Service::start_with(
        &base,
        &["s1"],
        common::service::Options {
            extra_args: vec!["--session-idle-timeout".into(), "2".into()],
            ..Default::default()
        },
    )
    .await;
    let key = seal_key(&svc);
    let ws = open(&svc).await;
    drop(ws);
    let start = std::time::Instant::now();
    let mut reaped = false;
    while start.elapsed() < Duration::from_secs(15) {
        let (status, _) = raw(svc.port, &callback(&key, TENANT, SESSION)).await;
        if status == 503 {
            reaped = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    assert!(
        reaped,
        "callbacks every 200 ms kept a detached session alive"
    );
    assert!(start.elapsed() >= Duration::from_secs(1));
}

// ---- the receiver and the driver over an in-memory stream

/// A scripted job table: its facts, and each `list` in turn (the last repeats).
struct Table {
    facts: Mutex<Facts>,
    lists: Mutex<Vec<Option<BTreeMap<String, Listed>>>>,
    listed: AtomicU32,
    performed: Mutex<Vec<Action>>,
    hold: Duration,
}

impl Table {
    fn new(facts: Facts, lists: Vec<Option<BTreeMap<String, Listed>>>) -> Arc<Self> {
        Arc::new(Self {
            facts: Mutex::new(facts),
            lists: Mutex::new(lists),
            listed: AtomicU32::new(0),
            performed: Mutex::new(Vec::new()),
            hold: Duration::from_millis(20),
        })
    }
    fn performed(&self) -> Vec<Action> {
        self.performed.lock().unwrap().clone()
    }
}

impl JobTracker for Table {
    fn facts(&self, _id: &JobId) -> Facts {
        *self.facts.lock().unwrap()
    }
    fn list(&self) -> BoxFuture<'static, Option<BTreeMap<String, Listed>>> {
        self.listed.fetch_add(1, Ordering::SeqCst);
        let mut lists = self.lists.lock().unwrap();
        let next = if lists.len() > 1 {
            lists.remove(0)
        } else {
            lists[0].clone()
        };
        let hold = self.hold;
        Box::pin(async move {
            tokio::time::sleep(hold).await;
            next
        })
    }
    fn perform(&self, action: Action) {
        self.performed.lock().unwrap().push(action);
    }
}

fn fixture_key() -> NotifyKey {
    let fx: Value = serde_json::from_str(include_str!(
        "../../sandbox/tests/fixtures/exec/v1.4-token.json"
    ))
    .unwrap();
    let bytes: [u8; 32] = base64::engine::general_purpose::STANDARD
        .decode(fx["secrets"]["current"].as_str().unwrap())
        .unwrap()
        .try_into()
        .unwrap();
    NotifyKey::from_secret_bytes(&bytes)
}

/// Serve one request through the real receiver over an in-memory stream.
#[allow(clippy::too_many_arguments)]
async fn serve_one(
    hub: Option<&NotifyHub>,
    service_mode: bool,
    method: &str,
    path: &str,
    query: Option<&str>,
    headers: &[(String, String)],
    body: &[u8],
    arrived: Instant,
) -> (u16, String) {
    let (mut client, mut server) = tokio::io::duplex(64 * 1024);
    // Outside service mode a replica has no hub, which is what makes it refuse a callback.
    let rx = job_ended::Receiver {
        hub: hub.filter(|_| service_mode),
    };
    let head = job_ended::Head {
        method,
        path,
        query,
        headers,
    };
    let status = match job_ended::classify(headers) {
        job_ended::Class::Callback(token) => {
            let token = token.to_string();
            client.write_all(body).await.unwrap();
            job_ended::serve(&mut server, &head, &token, &[], arrived, &rx).await;
            drop(server);
            None
        }
        job_ended::Class::LooksLike => Some(400),
        // The front door: a `job_ended` without a callback's credential is refused 403 there
        // (end to end in `job_ended_is_only_a_callback`).
        job_ended::Class::Other => Some(403),
    };
    if let Some(s) = status {
        return (s, String::new());
    }
    let mut out = Vec::new();
    client.read_to_end(&mut out).await.unwrap();
    let text = String::from_utf8_lossy(&out).into_owned();
    let status = text.split(' ').nth(1).unwrap().parse().unwrap();
    let body = text.split_once("\r\n\r\n").unwrap().1.to_string();
    (status, body)
}

fn listing(sandbox: &str) -> Option<BTreeMap<String, Listed>> {
    let one = |l| Some(BTreeMap::from([(JOB.to_string(), l)]));
    match sandbox {
        "running" => one(Listed::Running),
        "exited" => one(Listed::Exited { code: 0 }),
        "ended" => one(Listed::Ended { code: 0, at: 1 }),
        "lost" => one(Listed::Lost { at: 1 }),
        "missing" => Some(BTreeMap::new()),
        "unreachable" => None,
        other => panic!("sandbox {other}"),
    }
}

/// Wait until the session has no reconcile running.
async fn settle(s: &Arc<SessionNotify>) {
    for _ in 0..200 {
        if !s.kernel().reconciling() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("a reconcile never finished");
}

/// Put the session's job in the case's state, through the kernel's own events.
async fn prepare(s: &Arc<SessionNotify>, state: &Value) {
    let job = state["job"].as_str().unwrap();
    match job {
        "proving" | "unheld" => {
            s.feed(Event::Kept {
                job: JOB.into(),
                notify_available: true,
                adopted: false,
            });
            if state["keep_succeeded"] == true || job == "unheld" {
                let now = s.now();
                s.feed(Event::KeepAnswered {
                    job: JOB.into(),
                    outcome: KeepOutcome::Succeeded,
                    now,
                });
            }
        }
        _ => {}
    }
    if job == "unheld" {
        // The proof, then the stream closed, and the release's own reconcile run out.
        let now = s.now();
        s.feed(Event::Callback {
            call: u64::MAX,
            job: JOB.into(),
            facts: Facts {
                open_reconcile_done: true,
                job: TableJob::Running,
            },
            now,
        });
        let now = s.now();
        s.feed(Event::StreamClosed {
            job: JOB.into(),
            offset: 11,
            shell_exit_seen: false,
            now,
        });
        settle(s).await;
        assert!(matches!(s.kernel().job(JOB), Some(JobState::Unheld { .. })));
    }
}

/// Every receiver case of `v1.4-callback.json`, through the real receiver, driver and kernel.
#[tokio::test]
async fn every_receiver_case_of_the_fixture() {
    let fx: Value = serde_json::from_str(include_str!(
        "../../sandbox/tests/fixtures/exec/v1.4-callback.json"
    ))
    .unwrap();
    let cases = fx["receiver"].as_array().unwrap().clone();
    let mut runs = Vec::new();
    for case in cases {
        runs.push(tokio::spawn(async move { run_case(&case).await }));
    }
    for r in runs {
        r.await.unwrap();
    }
}

async fn run_case(case: &Value) {
    let name = case["name"].as_str().unwrap();
    let rq = &case["request"];
    let st = &case["state"];
    let ex = &case["expect"];
    let hub = NotifyHub::new(fixture_key());
    let table_job = match st["job"].as_str().unwrap() {
        "held" | "proving" | "unheld" => TableJob::Running,
        "end_recorded" => TableJob::Ended,
        "taken" => TableJob::Taken,
        "untracked" => TableJob::Absent,
        other => panic!("{other}"),
    };
    let facts = Facts {
        open_reconcile_done: st["open_reconcile_done"] == true,
        job: table_job,
    };
    let sandbox = st["sandbox"].as_str().unwrap();
    // The release's own reconcile lists the job running; the callback's sees the case's sandbox.
    let table = Table::new(facts, vec![listing("running"), listing(sandbox)]);
    let url = &fx_url_session();
    let session = (st["open_here"] == true).then(|| {
        let s = SessionNotify::register(&hub, &url.0, &url.1);
        let t: Arc<dyn JobTracker> = table.clone();
        s.set_tracker(Arc::downgrade(&t));
        (s, t)
    });
    if let Some((s, _)) = &session {
        // Prepare with the open reconcile done; the callback's facts are the case's.
        let real = *table.facts.lock().unwrap();
        *table.facts.lock().unwrap() = Facts {
            open_reconcile_done: true,
            job: TableJob::Running,
        };
        prepare(s, st).await;
        *table.facts.lock().unwrap() = real;
    }
    let lists_before = table.listed.load(Ordering::SeqCst);
    let performed_before = table.performed().len();

    let headers: Vec<(String, String)> = rq["authorization"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| ("Authorization".to_string(), v.as_str().unwrap().to_string()))
        .chain(rq["content_length"].as_array().unwrap().iter().map(|v| {
            (
                "Content-Length".to_string(),
                v.as_str().unwrap().to_string(),
            )
        }))
        .chain(rq["transfer_encoding"].as_array().unwrap().iter().map(|v| {
            (
                "Transfer-Encoding".to_string(),
                v.as_str().unwrap().to_string(),
            )
        }))
        .collect();
    // Arriving after the proof window, for a case that says so.
    let mut arrived = Instant::now();
    if st["job"] == "proving" && st["keep_succeeded"] == true && st["in_window"] != true {
        arrived += Duration::from_secs(31);
    }
    let query = rq["query"].as_str().unwrap();
    let (status, _body) = serve_one(
        Some(&hub),
        st["service_mode"] == true,
        rq["method"].as_str().unwrap(),
        rq["path"].as_str().unwrap(),
        (!query.is_empty()).then_some(query),
        &headers,
        rq["body"].as_str().unwrap().as_bytes(),
        arrived,
    )
    .await;
    assert_eq!(u64::from(status), ex["status"].as_u64().unwrap(), "{name}");
    let waited = table.listed.load(Ordering::SeqCst) > lists_before;
    assert_eq!(
        waited,
        ex["answer_waits_for_list"] == true,
        "{name}: waits for a list"
    );
    let new: Vec<Action> = table.performed()[performed_before..].to_vec();
    let effect = if new.iter().any(|a| matches!(a, Action::CloseStream { .. })) {
        "release"
    } else if new
        .iter()
        .any(|a| matches!(a, Action::ApplyShellExit { .. }))
    {
        "shell_exit"
    } else if new.iter().any(|a| matches!(a, Action::Follow { .. })) {
        "follow"
    } else {
        "none"
    };
    assert_eq!(effect, ex["effect"].as_str().unwrap(), "{name}: effect");
    if effect == "follow" {
        assert!(
            new.iter()
                .any(|a| matches!(a, Action::Follow { from: 11, .. })),
            "{name}: followed from the offset its stream reached"
        );
    }
    drop(session);
}

fn fx_url_session() -> (String, String) {
    let fx: Value = serde_json::from_str(include_str!(
        "../../sandbox/tests/fixtures/exec/v1.4-callback.json"
    ))
    .unwrap();
    (
        fx["url_session"]["tenant"].as_str().unwrap().to_string(),
        fx["url_session"]["session_id"]
            .as_str()
            .unwrap()
            .to_string(),
    )
}

/// A session with an unheld job, its release reconcile run out.
async fn unheld_session(
    lists: Vec<Option<BTreeMap<String, Listed>>>,
    hold: Duration,
) -> (
    Arc<NotifyHub>,
    Arc<SessionNotify>,
    Arc<Table>,
    Arc<dyn JobTracker>,
) {
    let hub = NotifyHub::new(fixture_key());
    let mut table = Table::new(
        Facts {
            open_reconcile_done: true,
            job: TableJob::Running,
        },
        lists,
    );
    Arc::get_mut(&mut table).unwrap().hold = hold;
    let (tenant, session) = fx_url_session();
    let s = SessionNotify::register(&hub, &tenant, &session);
    let t: Arc<dyn JobTracker> = table.clone();
    s.set_tracker(Arc::downgrade(&t));
    prepare(&s, &json!({"job":"unheld","keep_succeeded":true})).await;
    (hub, s, table, t)
}

fn job_id() -> JobId {
    JobId::parse(JOB).unwrap()
}

/// N30: one reconcile at a time; every callback that arrives during one shares the next, which
/// begins no sooner than a second after the last ended.
#[tokio::test]
async fn one_reconcile_at_a_time() {
    let (_hub, s, table, _t) =
        unheld_session(vec![listing("running")], Duration::from_millis(300)).await;
    let after_release = table.listed.load(Ordering::SeqCst);
    let t0 = std::time::Instant::now();
    let mut calls = Vec::new();
    for _ in 0..10 {
        let s = s.clone();
        calls.push(tokio::spawn(async move {
            s.callback(&job_id(), Instant::now()).await
        }));
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    for c in calls {
        assert_eq!(c.await.unwrap(), 200);
    }
    // The first callback's reconcile (after the spacing from the release's), and one shared by
    // every callback that arrived while it ran.
    let ran = table.listed.load(Ordering::SeqCst) - after_release;
    assert!((1..=2).contains(&ran), "{ran} reconciles for 10 callbacks");
    assert!(t0.elapsed() >= Duration::from_millis(900));
}

/// N31: an unheld job's callback is answered once a reconcile that began after it arrived has
/// listed; `503` if that list failed.
#[tokio::test]
async fn answers_after_the_list() {
    let (_hub, s, table, _t) = unheld_session(
        vec![listing("running"), listing("running"), None],
        Duration::from_millis(50),
    )
    .await;
    assert_eq!(s.callback(&job_id(), Instant::now()).await, 200);
    assert_eq!(table.listed.load(Ordering::SeqCst), 2);
    assert_eq!(s.callback(&job_id(), Instant::now()).await, 503);
    assert!(
        table.performed().iter().any(|a| matches!(
            a,
            Action::Follow {
                from: 11,
                end: None,
                ..
            }
        )),
        "a failed list follows the unheld job"
    );
    // Arrived 20 s ago: 503 at once, without waiting.
    let (_hub, s, _table, _t) =
        unheld_session(vec![listing("running")], Duration::from_millis(50)).await;
    let t = std::time::Instant::now();
    let late = Instant::now() - Duration::from_secs(20);
    assert_eq!(s.callback(&job_id(), late).await, 503);
    assert!(t.elapsed() < Duration::from_secs(5));
}

/// N32: a callback for anything but an unheld job is answered without contacting the sandbox.
#[tokio::test]
async fn repeats_are_free() {
    let hub = NotifyHub::new(fixture_key());
    for (open_done, job, want) in [
        (false, TableJob::Absent, 200),
        (true, TableJob::Running, 200),
        (true, TableJob::Ended, 200),
        (true, TableJob::Taken, 503),
        (true, TableJob::Absent, 410),
    ] {
        let table = Table::new(
            Facts {
                open_reconcile_done: open_done,
                job,
            },
            vec![listing("running")],
        );
        let s = SessionNotify::register(&hub, TENANT, SESSION);
        let t: Arc<dyn JobTracker> = table.clone();
        s.set_tracker(Arc::downgrade(&t));
        for _ in 0..20 {
            assert_eq!(s.callback(&job_id(), Instant::now()).await, want, "{job:?}");
        }
        assert_eq!(table.listed.load(Ordering::SeqCst), 0, "{job:?}");
    }
}

/// N27: each callback is answered through its own reply path, a closed connection stops its
/// wait, and a session holds at most 16 waiting.
#[tokio::test]
async fn each_callback_gets_its_own_answer() {
    let (_hub, s, _table, _t) =
        unheld_session(vec![listing("running")], Duration::from_millis(1500)).await;
    let mut calls = Vec::new();
    for _ in 0..16 {
        let s = s.clone();
        calls.push(tokio::spawn(async move {
            s.callback(&job_id(), Instant::now()).await
        }));
    }
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(s.kernel().waiting(), 16);
    assert_eq!(s.callback(&job_id(), Instant::now()).await, 503, "a 17th");
    // One whose connection closes stops waiting; the others still get theirs.
    calls.pop().unwrap().abort();
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(s.kernel().waiting(), 15);
    for c in calls {
        assert_eq!(c.await.unwrap(), 200);
    }
    assert_eq!(s.kernel().waiting(), 0);
}

/// N34: a job's end reaches the session once, however many callbacks and reconciles see it.
#[tokio::test(flavor = "multi_thread")]
async fn one_report_per_end() {
    use common::exec_mock::{MockOptions, StreamEnd};
    use common::notify_fleet::{bash_bg, fleet_with, post};
    use common::turn_text;
    let mut turns = vec![
        bash_bg("echo up; sleep 3; printf 'fin-%s\\n' q9"),
        turn_text("started"),
    ];
    turns.extend((0..8).map(|i| turn_text(&format!("noted {i}"))));
    let f = fleet_with(MockOptions::notifying(), turns, None).await;
    let _ws = f.start_job().await;
    let pgid = f.stream("fin-%s").await;
    let end = f.svc.exec.wait_for_end(pgid, Duration::from_secs(8)).await;
    assert_eq!(
        end,
        Some(StreamEnd::Disconnected { killed: false }),
        "released"
    );
    assert!(
        f.model_saw("fin-q9", Duration::from_secs(20)).await,
        "the end was reported"
    );
    // The same end, called back again and again: each answered, none reported again.
    let job = f.keeps()[0]["id"].as_str().unwrap().to_string();
    let key = common::notify_fleet::seal_key(&f.svc);
    let id = JobId::parse(&job).unwrap();
    let scope = JobScope::for_session(Some(TENANT), SESSION);
    let body = format!(
        r#"{{"type":"job_ended","scope":"{}","id":"{job}"}}"#,
        scope.name()
    );
    for _ in 0..5 {
        let status = post(
            &f.edge.origin,
            &target(TENANT, SESSION),
            &key.mint(TENANT, SESSION, &id),
            &body,
        )
        .await;
        assert!(status == 200 || status == 410, "{status}");
    }
    tokio::time::sleep(Duration::from_secs(2)).await;
    let all = f.requests.lock().unwrap().clone();
    let last = all.last().unwrap();
    assert_eq!(
        last.matches("exited with code 0").count(),
        1,
        "one notice for one end: {last}"
    );
    let wakes = all.iter().filter(|r| r.contains("fin-q9")).count();
    let first = all.iter().position(|r| r.contains("fin-q9")).unwrap();
    assert_eq!(
        wakes,
        all.len() - first,
        "no request without it after the first, and no extra turn"
    );
}

/// The receiver counts what it answers itself: refusals before the session lookup, and callbacks
/// for a session not held here.
#[tokio::test]
async fn the_receiver_counts_what_it_answers() {
    let metrics = beyond_agent::metrics::Metrics::new().unwrap();
    let hub = NotifyHub::with_metrics(fixture_key(), Some(metrics.clone()));
    let (tenant, session) = fx_url_session();
    let scope = JobScope::for_session(Some(&tenant), &session);
    let body = body_for(scope.name(), JOB);
    let query = format!("session_id={session}&tenant={tenant}");
    let headers = |token: &str| {
        vec![
            ("Authorization".to_string(), format!("Bearer {token}")),
            ("Content-Length".to_string(), body.len().to_string()),
        ]
    };
    let value = |series: &str| -> f64 {
        String::from_utf8(metrics.encode())
            .unwrap()
            .lines()
            .find_map(|l| l.strip_prefix(series).map(|v| v.trim().parse().unwrap()))
            .unwrap()
    };
    let (status, _) = serve_one(
        Some(&hub),
        true,
        "POST",
        "/_beyond/agent",
        Some(&query),
        &headers("bjn_v1.00000000.x"),
        body.as_bytes(),
        Instant::now(),
    )
    .await;
    assert_eq!(status, 401);
    assert_eq!(
        value(r#"agent_job_callbacks_total{outcome="refused"}"#),
        1.0
    );
    let token = fixture_key().mint(&tenant, &session, &JobId::parse(JOB).unwrap());
    let (status, _) = serve_one(
        Some(&hub),
        true,
        "POST",
        "/_beyond/agent",
        Some(&query),
        &headers(&token),
        body.as_bytes(),
        Instant::now(),
    )
    .await;
    assert_eq!(status, 503);
    assert_eq!(
        value(r#"agent_job_callbacks_total{outcome="not_open"}"#),
        1.0
    );
}
