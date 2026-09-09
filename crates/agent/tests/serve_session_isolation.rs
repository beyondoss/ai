//! Cross-session leakage on the multi-tenant `serve --listen` daemon.
//!
//! The daemon runs every session as a task on **one** shared Tokio runtime and (by default) one
//! shared HTTP pool. That is a density win, not a license to share tenant state: credentials stay
//! per-request on each session's `GatewayClient`, and transcripts, `/session` memory, persistence,
//! tools/`ExecCell`, and approvals are built inside `serve_session`. These tests are the proof —
//! two live sessions on the same daemon, asserted not to see each other's private state.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::{Command, Stdio};

use common::{
    ChildGuard, ISOLATED_HOME, SpawnGuarded, free_port, spawn_model_server,
    spawn_model_server_routed, turn_text, turn_tool_use, wait_for_port, ws_connect,
    ws_read_until_response, ws_send,
};
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_beyond-ai-agent");

fn serve_ws(base: &str, session_dir: &str, port: u16, extra: &[&str]) -> ChildGuard {
    Command::new(BIN)
        .args([
            "serve",
            "--listen",
            &format!("127.0.0.1:{port}"),
            "--gateway-url",
            base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session-dir",
            session_dir,
        ])
        .args(extra)
        .env("HOME", ISOLATED_HOME)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn_guarded()
}

fn dump(frames: &[Value]) -> String {
    frames
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("\n")
}

fn messages_blob(frames: &[Value]) -> String {
    frames
        .last()
        .and_then(|f| f.get("data"))
        .map(ToString::to_string)
        .unwrap_or_default()
}

fn tenant(secret: &str) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("secret.txt"), format!("{secret}\n")).unwrap();
    dir
}

fn target_cmd(dir: &std::path::Path) -> String {
    format!("env -C {} {{}}", dir.display())
}

/// Two concurrent one-turn prompts on distinct session ids: each transcript (and each live
/// `get_messages`) must contain only that session's marker. The mock is routed by the marker so
/// arrival order cannot hand session A's reply to B.
#[tokio::test]
async fn concurrent_sessions_do_not_leak_transcripts() {
    let (base, requests) = spawn_model_server_routed(
        vec![
            ("MARKER-ALPHA".to_string(), turn_text("alpha-only-reply")),
            ("MARKER-BRAVO".to_string(), turn_text("bravo-only-reply")),
        ],
        turn_text("unexpected-fallback"),
    );
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let mut child = serve_ws(&base, dir.path().to_str().unwrap(), port, &[]);
    wait_for_port(port);

    let mut a = ws_connect(port, Some("iso-alpha")).await;
    let mut b = ws_connect(port, Some("iso-bravo")).await;

    ws_send(
        &mut a,
        json!({ "type": "prompt", "id": "pa", "message": "say MARKER-ALPHA" }),
    )
    .await;
    ws_send(
        &mut b,
        json!({ "type": "prompt", "id": "pb", "message": "say MARKER-BRAVO" }),
    )
    .await;

    let fa = ws_read_until_response(&mut a, "prompt").await;
    let fb = ws_read_until_response(&mut b, "prompt").await;
    let da = dump(&fa);
    let db = dump(&fb);
    assert!(
        da.contains("alpha-only-reply"),
        "alpha's own reply missing: {da}"
    );
    assert!(
        db.contains("bravo-only-reply"),
        "bravo's own reply missing: {db}"
    );
    assert!(
        !da.contains("bravo-only-reply") && !da.contains("MARKER-BRAVO"),
        "bravo leaked into alpha's live stream: {da}"
    );
    assert!(
        !db.contains("alpha-only-reply") && !db.contains("MARKER-ALPHA"),
        "alpha leaked into bravo's live stream: {db}"
    );

    ws_send(&mut a, json!({ "type": "get_messages", "id": "ga" })).await;
    ws_send(&mut b, json!({ "type": "get_messages", "id": "gb" })).await;
    let ma = messages_blob(&ws_read_until_response(&mut a, "get_messages").await);
    let mb = messages_blob(&ws_read_until_response(&mut b, "get_messages").await);
    assert!(
        ma.contains("MARKER-ALPHA"),
        "alpha's prompt missing from its transcript: {ma}"
    );
    assert!(
        mb.contains("MARKER-BRAVO"),
        "bravo's prompt missing from its transcript: {mb}"
    );
    assert!(
        !ma.contains("MARKER-BRAVO"),
        "bravo's prompt leaked into alpha's get_messages: {ma}"
    );
    assert!(
        !mb.contains("MARKER-ALPHA"),
        "alpha's prompt leaked into bravo's get_messages: {mb}"
    );

    // Credentials stay per-request on GatewayClient, not on the shared pool: every upstream call
    // still carries this daemon's key, and neither session's body is served with the other's.
    let rec = requests.lock().unwrap();
    assert!(
        rec.len() >= 2,
        "both sessions must have hit the gateway through the shared pool: {rec:?}"
    );
    for raw in rec.iter() {
        assert!(
            raw.to_ascii_lowercase().contains("authorization"),
            "shared pool must not strip per-request credentials: {raw}"
        );
        let a_hit = raw.contains("MARKER-ALPHA");
        let b_hit = raw.contains("MARKER-BRAVO");
        assert!(
            !(a_hit && b_hit),
            "one gateway request mixed both sessions' prompts — pool/header bleed: {raw}"
        );
    }
    drop(rec);

    let _ = child.kill();
    let _ = child.wait();
}

/// Two live sessions, two sandboxes. A's `read` must not see B's secret (and vice versa), even
/// though both tasks share a runtime and (by default) one HTTP client. This is the concurrent
/// counterpart of `serve_exec_endpoint`'s switch_session test.
#[tokio::test]
async fn concurrent_sessions_do_not_leak_exec_endpoints() {
    let a_dir = tenant("SECRET-AAAA");
    let b_dir = tenant("SECRET-BBBB");
    let (base, _bodies) = spawn_model_server(vec![
        turn_tool_use("t1", "read", &json!({ "path": "secret.txt" }).to_string()),
        turn_text("first"),
        turn_tool_use("t2", "read", &json!({ "path": "secret.txt" }).to_string()),
        turn_text("second"),
    ]);
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let mut child = serve_ws(&base, dir.path().to_str().unwrap(), port, &[]);
    wait_for_port(port);

    let mut a = ws_connect(port, Some("exec-alpha")).await;
    let mut b = ws_connect(port, Some("exec-bravo")).await;

    ws_send(
        &mut a,
        json!({ "type": "set_exec_endpoint", "id": "ea", "command": target_cmd(a_dir.path()) }),
    )
    .await;
    let ea = ws_read_until_response(&mut a, "set_exec_endpoint").await;
    assert_eq!(ea.last().unwrap()["success"], true, "{:?}", ea.last());

    ws_send(
        &mut b,
        json!({ "type": "set_exec_endpoint", "id": "eb", "command": target_cmd(b_dir.path()) }),
    )
    .await;
    let eb = ws_read_until_response(&mut b, "set_exec_endpoint").await;
    assert_eq!(eb.last().unwrap()["success"], true, "{:?}", eb.last());

    // Serialized prompts so the FIFO mock stays aligned; both sessions are live on the shared
    // runtime the whole time — that's the isolation claim, not overlapping tool futures.
    ws_send(
        &mut a,
        json!({ "type": "prompt", "id": "pa", "message": "read it" }),
    )
    .await;
    let fa = dump(&ws_read_until_response(&mut a, "prompt").await);
    assert!(
        fa.contains("SECRET-AAAA"),
        "alpha did not reach its sandbox: {fa}"
    );
    assert!(
        !fa.contains("SECRET-BBBB"),
        "bravo's secret leaked into alpha: {fa}"
    );

    ws_send(
        &mut b,
        json!({ "type": "prompt", "id": "pb", "message": "read it" }),
    )
    .await;
    let fb = dump(&ws_read_until_response(&mut b, "prompt").await);
    assert!(
        fb.contains("SECRET-BBBB"),
        "bravo did not reach its sandbox: {fb}"
    );
    assert!(
        !fb.contains("SECRET-AAAA"),
        "alpha's secret leaked into bravo: {fb}"
    );

    let _ = child.kill();
    let _ = child.wait();
}

/// `/session` working memory is a per-session mount. A fact written by alpha must not appear in
/// bravo's store (and the on-disk sibling dirs must stay distinct).
#[tokio::test]
async fn concurrent_sessions_do_not_leak_session_memory() {
    let (base, _bodies) = spawn_model_server(vec![
        turn_tool_use(
            "t1",
            "memory",
            &json!({
                "command": "create",
                "path": "/session/who.md",
                "file_text": "alice-only\n"
            })
            .to_string(),
        ),
        turn_text("noted-a"),
        turn_tool_use(
            "t2",
            "memory",
            &json!({
                "command": "create",
                "path": "/session/who.md",
                "file_text": "bob-only\n"
            })
            .to_string(),
        ),
        turn_text("noted-b"),
        turn_tool_use(
            "t3",
            "memory",
            &json!({ "command": "view", "path": "/session/who.md" }).to_string(),
        ),
        turn_text("alice-read"),
        turn_tool_use(
            "t4",
            "memory",
            &json!({ "command": "view", "path": "/session/who.md" }).to_string(),
        ),
        turn_text("bob-read"),
    ]);
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let mut child = serve_ws(&base, dir.path().to_str().unwrap(), port, &[]);
    wait_for_port(port);

    let mut a = ws_connect(port, Some("mem-alpha")).await;
    let mut b = ws_connect(port, Some("mem-bravo")).await;

    ws_send(
        &mut a,
        json!({ "type": "prompt", "id": "wa", "message": "remember who" }),
    )
    .await;
    let wa = dump(&ws_read_until_response(&mut a, "prompt").await);
    assert!(
        wa.contains("alice-only") || wa.contains("noted-a"),
        "alpha write failed: {wa}"
    );

    ws_send(
        &mut b,
        json!({ "type": "prompt", "id": "wb", "message": "remember who" }),
    )
    .await;
    let wb = dump(&ws_read_until_response(&mut b, "prompt").await);
    assert!(
        wb.contains("bob-only") || wb.contains("noted-b"),
        "bravo write failed: {wb}"
    );
    assert!(
        !wb.contains("alice-only"),
        "alice leaked into bravo's write turn: {wb}"
    );

    ws_send(
        &mut a,
        json!({ "type": "prompt", "id": "ra", "message": "who am i" }),
    )
    .await;
    let ra = dump(&ws_read_until_response(&mut a, "prompt").await);
    assert!(
        ra.contains("alice-only"),
        "alpha cannot read its own /session: {ra}"
    );
    assert!(
        !ra.contains("bob-only"),
        "bob leaked into alpha's /session: {ra}"
    );

    ws_send(
        &mut b,
        json!({ "type": "prompt", "id": "rb", "message": "who am i" }),
    )
    .await;
    let rb = dump(&ws_read_until_response(&mut b, "prompt").await);
    assert!(
        rb.contains("bob-only"),
        "bravo cannot read its own /session: {rb}"
    );
    assert!(
        !rb.contains("alice-only"),
        "alice leaked into bravo's /session: {rb}"
    );

    let _ = child.kill();
    let _ = child.wait();
}

/// An `allow_session` on alpha's write must not satisfy bravo's later write of a different file.
/// Approvals are session-scoped memory, not daemon-scoped.
#[tokio::test]
async fn concurrent_sessions_do_not_leak_approvals() {
    let dir = tempfile::tempdir().unwrap();
    let a_file = dir.path().join("alpha.txt");
    let b_file = dir.path().join("bravo.txt");
    let (base, _b) = spawn_model_server(vec![
        turn_tool_use(
            "t1",
            "write",
            &json!({ "path": a_file.to_str().unwrap(), "content": "from-alpha\n" }).to_string(),
        ),
        turn_text("wrote-a"),
        turn_tool_use(
            "t2",
            "write",
            &json!({ "path": b_file.to_str().unwrap(), "content": "from-bravo\n" }).to_string(),
        ),
        turn_text("wrote-b"),
    ]);
    let port = free_port();
    let mut child = serve_ws(
        &base,
        dir.path().to_str().unwrap(),
        port,
        &["--approve", "writes"],
    );
    wait_for_port(port);

    let mut a = ws_connect(port, Some("appr-alpha")).await;
    let mut b = ws_connect(port, Some("appr-bravo")).await;

    ws_send(
        &mut a,
        json!({ "type": "prompt", "id": "pa", "message": "write alpha" }),
    )
    .await;
    let mut fa = Vec::new();
    let req_a = loop {
        let Some(f) = common::ws_next_frame(&mut a).await else {
            panic!("alpha socket closed before approval_request; saw: {fa:#?}");
        };
        let is_req = f["type"] == "approval_request";
        let is_resp = f["type"] == "response" && f["command"] == "prompt";
        fa.push(f);
        if is_req {
            break fa.last().unwrap().clone();
        }
        if is_resp {
            panic!("alpha prompt finished without asking; saw: {fa:#?}");
        }
    };
    ws_send(
        &mut a,
        json!({
            "type": "approve",
            "id": "aa",
            "request_id": req_a["request_id"],
            "decision": "allow",
            "scope": "session"
        }),
    )
    .await;
    let done_a = dump(&ws_read_until_response(&mut a, "prompt").await);
    assert!(
        done_a.contains("wrote-a") || a_file.exists(),
        "alpha's approved write should have run: {done_a}"
    );

    ws_send(
        &mut b,
        json!({ "type": "prompt", "id": "pb", "message": "write bravo" }),
    )
    .await;
    let mut saw_b_request = false;
    let mut fb = Vec::new();
    loop {
        let Some(f) = common::ws_next_frame(&mut b).await else {
            panic!("bravo socket closed; saw: {fb:#?}");
        };
        let is_req = f["type"] == "approval_request";
        let is_resp = f["type"] == "response" && f["command"] == "prompt";
        fb.push(f);
        if is_req {
            saw_b_request = true;
            break;
        }
        if is_resp {
            break;
        }
    }
    assert!(
        saw_b_request,
        "alpha's session-scoped allow leaked onto bravo — bravo wrote without being asked: {fb:#?}"
    );
    assert!(
        !b_file.exists(),
        "bravo's write must not have run before its own approval"
    );

    let _ = child.kill();
    let _ = child.wait();
}

/// Persistence files are addressed by session id. Two live sessions must not append into one jsonl.
#[tokio::test]
async fn concurrent_sessions_persist_to_distinct_files() {
    let (base, _r) = spawn_model_server_routed(
        vec![
            ("PERSIST-A".to_string(), turn_text("persist-a-reply")),
            ("PERSIST-B".to_string(), turn_text("persist-b-reply")),
        ],
        turn_text("unexpected-fallback"),
    );
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let mut child = serve_ws(&base, dir.path().to_str().unwrap(), port, &[]);
    wait_for_port(port);

    let mut a = ws_connect(port, Some("persist-alpha")).await;
    let mut b = ws_connect(port, Some("persist-bravo")).await;
    ws_send(
        &mut a,
        json!({ "type": "prompt", "id": "pa", "message": "PERSIST-A" }),
    )
    .await;
    ws_send(
        &mut b,
        json!({ "type": "prompt", "id": "pb", "message": "PERSIST-B" }),
    )
    .await;
    assert_eq!(
        ws_read_until_response(&mut a, "prompt")
            .await
            .last()
            .unwrap()["success"],
        true
    );
    assert_eq!(
        ws_read_until_response(&mut b, "prompt")
            .await
            .last()
            .unwrap()["success"],
        true
    );

    // Dropping the sockets detaches; persist already happened at turn end. Scan the repo dir.
    drop(a);
    drop(b);
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir.path()).unwrap() {
        let p = entry.unwrap().path();
        if p.extension().and_then(|e| e.to_str()) == Some("jsonl") {
            files.push(p);
        }
    }
    assert!(
        files.len() >= 2,
        "each session must own a jsonl, got {files:?}"
    );
    let bodies: Vec<String> = files
        .iter()
        .map(|p| std::fs::read_to_string(p).unwrap_or_default())
        .collect();
    let a_file = bodies.iter().find(|b| b.contains("PERSIST-A"));
    let b_file = bodies.iter().find(|b| b.contains("PERSIST-B"));
    assert!(
        a_file.is_some(),
        "alpha's transcript not on disk: {bodies:?}"
    );
    assert!(
        b_file.is_some(),
        "bravo's transcript not on disk: {bodies:?}"
    );
    assert!(
        a_file.map(|s| !s.contains("PERSIST-B")).unwrap_or(false),
        "bravo's prompt leaked into alpha's jsonl: {bodies:?}"
    );
    assert!(
        b_file.map(|s| !s.contains("PERSIST-A")).unwrap_or(false),
        "alpha's prompt leaked into bravo's jsonl: {bodies:?}"
    );

    let _ = child.kill();
    let _ = child.wait();
}
