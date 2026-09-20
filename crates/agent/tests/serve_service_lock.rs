//! Two replicas, one mount: who owns a session, and what the other one is told.
//!
//! This is the deployment the segmented layout exists for — several `serve --service` tasks with the
//! same shard mounted, any of which an edge may route a reconnect to. Two properties have to hold
//! together. A session that is genuinely *live* somewhere else must be refused with a status a
//! client can act on (503 + `Retry-After`), not accepted into a second writer. And once its owner is
//! gone — cleanly, or by `kill -9` with nothing unwound — the next replica must take it over,
//! seal what the predecessor wrote, and replay it.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::path::Path;
use std::time::{Duration, Instant};

use agent_core::Message;
use beyond_ai_agent::session_store::SessionMeta;
use common::service::{Options, Service};
use common::{
    TestWs, spawn_model_server, turn_text, ws_connect_with_headers, ws_read_until_response,
    ws_refusal, ws_send,
};
use serde_json::{Value, json};

/// Open `id` for `tenant` on `port`, warm it with one `get_state` so its task is provably live (and
/// so it provably holds its lock), and hand back the attached socket.
async fn open(svc: &Service, port: u16, tenant: &str, id: &str) -> TestWs {
    let token = svc.token(tenant, id);
    let mut ws = ws_connect_with_headers(port, Some(id), &svc.header(&token))
        .await
        .unwrap_or_else(|status| panic!("{tenant}/{id} on :{port} refused with {status}"));
    ws_send(&mut ws, json!({"type":"get_state","id":"warm"})).await;
    let frames = ws_read_until_response(&mut ws, "get_state").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    ws
}

/// Keep asking until `port` lets this session in, or give up. A takeover is bounded by how long the
/// previous owner takes to let go, not by anything this test controls.
async fn attach_within(
    svc: &Service,
    port: u16,
    tenant: &str,
    id: &str,
    within: Duration,
) -> TestWs {
    let deadline = Instant::now() + within;
    loop {
        let token = svc.token(tenant, id);
        let refused = match ws_connect_with_headers(port, Some(id), &svc.header(&token)).await {
            Ok(ws) => return ws,
            Err(status) => status,
        };
        assert!(
            Instant::now() < deadline,
            "{id} was still refused with {refused} after {within:?}"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// The epoch headers on disk: `(epoch, sealed map)` per segment, oldest first. Plaintext by design —
/// a reader has to be able to bound a segment it holds no key for.
fn segment_headers(session_dir: &Path) -> Vec<(u64, Value)> {
    let mut entries: Vec<_> = std::fs::read_dir(session_dir)
        .unwrap_or_else(|e| panic!("no session dir {}: {e}", session_dir.display()))
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("jsonl"))
        .collect();
    entries.sort();
    entries
        .iter()
        .map(|p| {
            let first = std::fs::read_to_string(p).unwrap();
            let first = first.lines().next().unwrap_or_default().to_string();
            let header: Value = serde_json::from_str(&first)
                .unwrap_or_else(|e| panic!("{} has no plaintext header: {e}", p.display()));
            assert_eq!(header["type"], "segment", "{header}");
            (header["epoch"].as_u64().unwrap(), header["sealed"].clone())
        })
        .collect()
}

async fn title(ws: &mut TestWs) -> String {
    ws_send(ws, json!({"type":"get_state","id":"g"})).await;
    let frames = ws_read_until_response(ws, "get_state").await;
    frames.last().unwrap()["data"]["title"]
        .as_str()
        .unwrap_or_default()
        .to_string()
}

async fn set_title(ws: &mut TestWs, title: &str) {
    ws_send(
        ws,
        json!({"type":"set_session_name","id":"n1","title": title}),
    )
    .await;
    let frames = ws_read_until_response(ws, "set_session_name").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
}

/// A live session is owned by exactly one replica. The other answers 503 with a `Retry-After`,
/// **before** the upgrade — past it the only vocabulary left is a close code, which no client, proxy
/// or retry policy reads as "try again in a second".
#[tokio::test]
async fn a_session_live_on_another_replica_is_refused_until_its_owner_lets_go() {
    let (base, _requests) = spawn_model_server(vec![]);
    // A short idle timeout so the first replica's session actually ends once nothing is attached —
    // that release is the other half of what this test is about.
    let svc = Service::start_with(
        &base,
        &["s1"],
        Options {
            extra_args: vec!["--session-idle-timeout".into(), "1".into()],
            ..Default::default()
        },
    )
    .await;
    let peer = svc.start_peer(&[]);

    let mut owner = open(&svc, svc.port, "tenant-a", "s1.alpha").await;
    set_title(&mut owner, "written by the first replica").await;

    let token = svc.token("tenant-a", "s1.alpha");
    let (status, retry_after) = ws_refusal(peer.port, Some("s1.alpha"), &svc.header(&token)).await;
    assert_eq!(
        status, 503,
        "a session held elsewhere is unavailable, not 4xx"
    );
    assert_eq!(
        retry_after.as_deref(),
        Some("1"),
        "a 503 has to say when to come back"
    );
    // The refusal started nothing: the peer holds no session for that id.
    assert_eq!(
        segment_headers(&svc.sessions_dir("s1", "tenant-a").join("s1.alpha")).len(),
        1
    );

    // Let go: detach, and let the first replica's reaper end the session.
    drop(owner);
    let mut taken_over = attach_within(
        &svc,
        peer.port,
        "tenant-a",
        "s1.alpha",
        Duration::from_secs(30),
    )
    .await;
    assert_eq!(
        title(&mut taken_over).await,
        "written by the first replica",
        "the new owner replays what the old one wrote"
    );

    // Nothing is created until the new owner actually writes — and then it fences: a second epoch,
    // whose header seals the first at the offset the replay consumed.
    assert_eq!(
        segment_headers(&svc.sessions_dir("s1", "tenant-a").join("s1.alpha")).len(),
        1,
        "a replay alone must not roll a segment"
    );
    set_title(&mut taken_over, "written by the second replica").await;
    let headers = segment_headers(&svc.sessions_dir("s1", "tenant-a").join("s1.alpha"));
    assert_eq!(headers.len(), 2, "{headers:?}");
    assert_eq!(headers[1].0, 2);
    assert!(
        headers[1].1["1"].as_u64().is_some_and(|n| n > 0),
        "epoch 2 must seal epoch 1 at the offset it replayed: {:?}",
        headers[1].1
    );
}

/// `kill -9` is the case a lease would get wrong: nothing unwinds, nothing is flushed, and the lock
/// is released only because the kernel closed the descriptor. The next replica takes the session
/// over from what is on disk.
#[tokio::test]
async fn after_a_kill_the_next_replica_takes_the_session_over() {
    let (base, _requests) = spawn_model_server(vec![turn_text("the first replica answered")]);
    let mut svc = Service::start(&base, &["s1"]).await;
    let peer = svc.start_peer(&[]);

    let mut owner = open(&svc, svc.port, "tenant-a", "s1.alpha").await;
    ws_send(
        &mut owner,
        json!({"type":"prompt","id":"p1","message":"hello"}),
    )
    .await;
    let frames = ws_read_until_response(&mut owner, "prompt").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    set_title(&mut owner, "survived a kill").await;

    // No shutdown, no persist on the way out, no lock release of its own.
    svc.child.kill().unwrap();
    let _ = svc.child.wait();
    drop(owner);

    let mut taken_over = attach_within(
        &svc,
        peer.port,
        "tenant-a",
        "s1.alpha",
        Duration::from_secs(30),
    )
    .await;
    assert_eq!(title(&mut taken_over).await, "survived a kill");
    ws_send(&mut taken_over, json!({"type":"get_messages","id":"m1"})).await;
    let frames = ws_read_until_response(&mut taken_over, "get_messages").await;
    let messages = serde_json::to_string(&frames.last().unwrap()["data"]).unwrap();
    assert!(
        messages.contains("the first replica answered"),
        "the killed replica's turn must replay: {messages}"
    );

    set_title(&mut taken_over, "and carried on").await;
    let headers = segment_headers(&svc.sessions_dir("s1", "tenant-a").join("s1.alpha"));
    assert_eq!(headers.len(), 2, "{headers:?}");
    assert!(
        headers[1].1["1"].as_u64().is_some_and(|n| n > 0),
        "{headers:?}"
    );
}

/// The lock is liveness only — lose it to a partition or a stuck client and correctness still holds,
/// because the epoch fence does not need it. What the fenced owner must not do is carry on taking
/// turns it can no longer persist: its next write reports the takeover, and the session ends with an
/// event that tells the client to reconnect (which lands it on whoever owns the session now).
#[tokio::test]
async fn a_session_fenced_by_a_newer_epoch_ends_with_session_superseded() {
    let (base, _requests) = spawn_model_server(vec![]);
    let svc = Service::start(&base, &["s1"]).await;
    let repo = svc.tenant_repo("s1", "tenant-a");
    // Already on the mount, so the replica *opens* this session rather than creating it — an owner
    // that created the newest epoch itself is the one writer nobody can have raced.
    {
        let mut store = repo
            .create(SessionMeta::with_id("s1.alpha", "/w", "claude-test"))
            .unwrap();
        store
            .append_new(&[Message::user("written before any replica")])
            .unwrap();
    }
    let mut ws = open(&svc, svc.port, "tenant-a", "s1.alpha").await;

    // Another owner takes it over, exactly as a second replica would: opening and writing creates
    // the next epoch, sealing this one where its reader stopped.
    {
        let (mut taken, _session) = repo.open_id("s1.alpha").unwrap();
        taken.set_title("taken over").unwrap();
    }

    ws_send(
        &mut ws,
        json!({"type":"set_session_name","id":"n1","title":"too late"}),
    )
    .await;
    let mut superseded = None;
    while let Some(frame) = common::ws_next_frame(&mut ws).await {
        if frame["type"] == "session_superseded" {
            superseded = Some(frame);
            break;
        }
    }
    let frame = superseded.expect("a fenced session must say so rather than going quiet");
    assert_eq!(frame["session_id"], "s1.alpha", "{frame}");
    assert_eq!(frame["tenant"], "tenant-a", "{frame}");

    // What the other owner wrote is what is on disk; the fenced writer changed nothing.
    let (store, _session) = repo.open_id_read_only("s1.alpha").unwrap();
    assert_eq!(store.meta().title.as_deref(), Some("taken over"));
}

/// A replica holds two descriptors per live session — its newest segment and its lock — against a
/// network filesystem's per-instance ceiling on both. Past the cap a new session is refused with a
/// status, rather than failing later as an unexplained I/O error inside a tenant's turn.
#[tokio::test]
async fn the_live_session_limit_refuses_a_second_session() {
    let (base, _requests) = spawn_model_server(vec![]);
    let svc = Service::start_with(
        &base,
        &["s1"],
        Options {
            extra_args: vec!["--max-live-sessions".into(), "1".into()],
            ..Default::default()
        },
    )
    .await;

    let mut first = open(&svc, svc.port, "tenant-a", "s1.alpha").await;

    let token = svc.token("tenant-a", "s1.beta");
    let (status, retry_after) = ws_refusal(svc.port, Some("s1.beta"), &svc.header(&token)).await;
    assert_eq!(status, 503);
    assert_eq!(retry_after.as_deref(), Some("1"));
    assert!(
        !svc.sessions_dir("s1", "tenant-a").join("s1.beta").exists(),
        "a refused session must not leave a directory behind"
    );

    // The cap is on *live* sessions, not on the replica: a second connection to the one that is
    // already live still attaches.
    let _second = open(&svc, svc.port, "tenant-a", "s1.alpha").await;
    assert_eq!(title(&mut first).await, "");
}
