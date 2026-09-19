//! The no-host rule, from both directions: nothing of the replica's reaches a tenant, and nothing
//! of a tenant's is written where the replica (or another tenant) can read it.
//!
//! These are the tests that would go quiet if a *fallback* crept back in — a credential ladder that
//! finds an ambient key, an exec cell that falls through to `RealRunner`, a prompt that picks up
//! `~/.claude/SYSTEM.md`. Each of those is a one-line change that breaks nothing else.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::path::Path;

use beyond_ai_agent::memory::{MemPath, MemoryBackend, View};
use common::service::{EXEC_HEADER, GATEWAY_KEY, Options, Service, files_under};
use common::{
    spawn_model_server, turn_text, turn_tool_use, ws_connect_with_headers, ws_next_frame,
    ws_read_until_response, ws_send,
};
use serde_json::json;

/// A model turn that runs one `bash` command in the sandbox, then answers.
fn bash_turn(command: &str, cwd: &Path) -> Vec<String> {
    let args = json!({ "command": command, "cwd": cwd.to_string_lossy() }).to_string();
    vec![turn_tool_use("t1", "bash", &args), turn_text("done")]
}

/// The credential is the grant's, full stop. An ambient `ANTHROPIC_API_KEY` on the replica — which
/// the old resolution ladder would happily have found — must not reach the wire, because on a
/// replica it is the operator's key and every tenant would be spending it.
#[tokio::test]
async fn the_grants_gateway_key_is_used_and_an_ambient_host_key_is_ignored() {
    const HOST_KEY: &str = "sk-ant-host-key-must-never-be-used";
    let (base, requests) = spawn_model_server(vec![turn_text("hi")]);
    let svc = Service::start_with(
        &base,
        &["s1"],
        Options {
            // `AI_AGENT_KEY` is deliberately absent: service mode refuses it at *startup*
            // (`serve_service_mode::an_ambient_agent_key_refuses_startup`), so it can never get as
            // far as the wire. A provider key is the interesting case — nothing refuses it, and the
            // old credential ladder would have found it.
            env: vec![("ANTHROPIC_API_KEY".into(), HOST_KEY.into())],
            ..Default::default()
        },
    )
    .await;
    let token = svc.token("t1", "s1.alpha");
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .unwrap();
    ws_send(
        &mut ws,
        json!({"type":"prompt","id":"p1","message":"hello"}),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "prompt").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");

    let recorded = requests.lock().unwrap().clone();
    assert_eq!(recorded.len(), 1, "one model call");
    let raw = &recorded[0];
    assert!(
        raw.contains(GATEWAY_KEY),
        "the grant's gateway key must be on the wire: {raw}"
    );
    assert!(
        !raw.contains(HOST_KEY),
        "the replica's ambient key reached the wire: {raw}"
    );
}

/// The replica's own `~/.claude` is the operator's machine. A `SYSTEM.md` there would rewrite every
/// tenant's identity; a skill there would be advertised to every tenant.
#[tokio::test]
async fn the_replicas_own_claude_directory_never_reaches_a_tenants_prompt() {
    const SYSTEM_MARKER: &str = "HOST-SYSTEM-MD-MARKER";
    const SKILL_MARKER: &str = "host-only-skill";
    let host_home = tempfile::tempdir().unwrap();
    let claude = host_home.path().join(".claude");
    std::fs::create_dir_all(claude.join("skills").join(SKILL_MARKER)).unwrap();
    std::fs::write(
        claude.join("SYSTEM.md"),
        format!("You are {SYSTEM_MARKER}, the replica's own agent."),
    )
    .unwrap();
    std::fs::write(
        claude.join("APPEND_SYSTEM.md"),
        "HOST-APPEND-MARKER always applies.",
    )
    .unwrap();
    std::fs::write(
        claude.join("skills").join(SKILL_MARKER).join("SKILL.md"),
        format!("---\nname: {SKILL_MARKER}\ndescription: the replica's own skill\n---\nbody\n"),
    )
    .unwrap();
    // A CLAUDE.md in the replica's cwd would likewise be picked up by context-file discovery.
    std::fs::write(claude.join("CLAUDE.md"), "HOST-CONTEXT-MARKER").unwrap();

    let (base, requests) = spawn_model_server(vec![turn_text("hi")]);
    let svc = Service::start_with(
        &base,
        &["s1"],
        Options {
            host_home: Some(host_home.path().to_path_buf()),
            ..Default::default()
        },
    )
    .await;
    let token = svc.token("t1", "s1.alpha");
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .unwrap();

    // No skill or template is advertised at all.
    ws_send(&mut ws, json!({"type":"get_commands","id":"c1"})).await;
    let frames = ws_read_until_response(&mut ws, "get_commands").await;
    let commands = frames.last().unwrap()["data"]["commands"]
        .as_array()
        .unwrap()
        .clone();
    assert!(commands.is_empty(), "{commands:?}");

    ws_send(
        &mut ws,
        json!({"type":"prompt","id":"p1","message":"hello"}),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "prompt").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    let raw = requests.lock().unwrap()[0].clone();
    for marker in [
        SYSTEM_MARKER,
        SKILL_MARKER,
        "HOST-APPEND-MARKER",
        "HOST-CONTEXT-MARKER",
    ] {
        assert!(!raw.contains(marker), "{marker} reached the prompt: {raw}");
    }
}

/// A session whose sandbox never answers has no host to fall back to, so no tool of any kind runs.
/// The marker file is the proof: it sits on the replica's filesystem, and nothing ever opens it.
#[tokio::test]
async fn a_failed_probe_means_no_tool_ever_runs_on_the_replica() {
    const MARKER: &str = "HOST-FILE-CONTENTS-NOBODY-MAY-READ";
    let marker_dir = tempfile::tempdir().unwrap();
    let marker = marker_dir.path().join("secret.txt");
    std::fs::write(&marker, MARKER).unwrap();

    // The model would read the marker, if it were ever asked anything.
    let (base, requests) = spawn_model_server(vec![
        turn_tool_use(
            "t1",
            "read",
            &json!({ "path": marker.to_string_lossy() }).to_string(),
        ),
        turn_text("done"),
    ]);
    let svc = Service::start(&base, &["s1"]).await;
    let token = svc.token_with("t1", "s1.alpha", |c| {
        c.exec_url = "http://127.0.0.1:1/exec".into()
    });
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .unwrap();
    ws_send(
        &mut ws,
        json!({"type":"prompt","id":"p1","message":"read it"}),
    )
    .await;

    let mut frames = Vec::new();
    while let Some(frame) = ws_next_frame(&mut ws).await {
        let done = frame["type"] == "error";
        frames.push(frame);
        if done {
            break;
        }
    }
    assert!(
        frames.iter().any(|f| f["type"] == "error"),
        "the session must end rather than run anything: {frames:#?}"
    );
    assert!(
        requests.lock().unwrap().is_empty(),
        "the session died before it ever called the model"
    );
    let text = serde_json::to_string(&frames).unwrap();
    assert!(!text.contains(MARKER), "the host file was read: {text}");
    assert_eq!(std::fs::read_to_string(&marker).unwrap(), MARKER);
}

/// The sandbox's `$HOME` — not the replica's — is what a session's tools see, which is what the
/// startup probe exists to establish.
#[tokio::test]
async fn tools_run_in_the_sandbox_with_the_sandboxs_own_home() {
    let host_home = tempfile::tempdir().unwrap();
    let (base, _requests) = spawn_model_server(vec![]);
    let svc = Service::start_with(
        &base,
        &["s1"],
        Options {
            host_home: Some(host_home.path().to_path_buf()),
            ..Default::default()
        },
    )
    .await;
    let token = svc.token("t1", "s1.alpha");
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .unwrap();

    // The host `bash` RPC command runs through the same exec cell the model's tools do.
    ws_send(
        &mut ws,
        json!({
            "type": "bash",
            "id": "b1",
            "command": "printf %s \"$HOME\" > \"$HOME/from-the-sandbox\"; printf %s \"$HOME\"",
            "cwd": svc.workspace.to_string_lossy(),
        }),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "bash").await;
    let response = frames.last().unwrap();
    assert_eq!(response["success"], true, "{response}");

    let landed = svc.sandbox_home.join("from-the-sandbox");
    assert!(
        landed.is_file(),
        "the command must have run with the sandbox's HOME: {}",
        landed.display()
    );
    assert_eq!(
        std::fs::read_to_string(&landed).unwrap(),
        svc.sandbox_home.to_string_lossy()
    );
    assert!(
        !host_home.path().join("from-the-sandbox").exists(),
        "nothing may be written under the replica's HOME"
    );
}

/// The grant's exec headers are credentials: they must reach the sandbox and appear nowhere else.
#[tokio::test]
async fn exec_headers_reach_the_sandbox_and_never_a_response_or_a_file() {
    let (base, _requests) = spawn_model_server({
        let mut turns = Vec::new();
        turns.extend(bash_turn("echo in-the-sandbox", Path::new(".")));
        turns
    });
    let svc = Service::start(&base, &["s1"]).await;
    let token = svc.token("00tenant1", "s1.alpha");
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .unwrap();

    let mut frames = Vec::new();
    ws_send(
        &mut ws,
        json!({"type":"prompt","id":"p1","message":"run it"}),
    )
    .await;
    while let Some(frame) = ws_next_frame(&mut ws).await {
        let done = frame["type"] == "response" && frame["command"] == "prompt";
        frames.push(frame);
        if done {
            break;
        }
    }
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");

    // The header did go out — otherwise this test would pass for the wrong reason.
    let sent = svc.exec.requests();
    assert!(!sent.is_empty(), "the sandbox was never called");
    assert!(
        sent.iter()
            .all(|r| r.header(EXEC_HEADER.0) == Some(EXEC_HEADER.1)),
        "every exec request carries the grant's header"
    );

    // …and it is in no frame the tenant saw.
    let text = serde_json::to_string(&frames).unwrap();
    assert!(
        !text.contains(EXEC_HEADER.1),
        "exec header in a frame: {text}"
    );
    assert!(
        !text.contains(&token),
        "the grant itself in a frame: {text}"
    );
    assert!(
        !text.contains(&svc.shard("s1").to_string_lossy().into_owned()),
        "a replica mount path in a frame: {text}"
    );

    // …nor in anything written to disk. Exec headers are never persisted: `set_exec_endpoint` (the
    // only writer of that record) is refused in service mode.
    for file in files_under(svc.shard("s1")) {
        let bytes = std::fs::read(&file).unwrap();
        let body = String::from_utf8_lossy(&bytes);
        assert!(
            !body.contains(EXEC_HEADER.1),
            "exec header persisted in {}",
            file.display()
        );
        assert!(
            !body.contains(&token),
            "the grant persisted in {}",
            file.display()
        );
    }
}

/// The transcript records the tenant's workspace as its cwd, so a restart reopens the session
/// against the sandbox rather than against whatever directory the replica happened to be started in.
#[tokio::test]
async fn the_persisted_session_records_the_sandbox_workspace_as_its_cwd() {
    let (base, _requests) = spawn_model_server(vec![]);
    let svc = Service::start(&base, &["s1"]).await;
    let token = svc.token("00tenant1", "s1.alpha");
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .unwrap();
    ws_send(&mut ws, json!({"type":"get_state","id":"g1"})).await;
    let _ = ws_read_until_response(&mut ws, "get_state").await;

    // Read back the way the next owner of this shard would: the tenant's own segmented, sealed repo.
    // The transcript itself is unreadable without that key, which is the point.
    let (store, _session) = svc
        .tenant_repo("s1", "00tenant1")
        .open_id_read_only("s1.alpha")
        .unwrap();
    assert_eq!(
        store.meta().cwd,
        svc.workspace.to_string_lossy().into_owned()
    );
}

/// Durable memory is the tenant's, on its home shard — not a directory derived from the replica's
/// cwd, which every tenant on the box would share.
#[tokio::test]
async fn durable_memory_lands_in_the_tenants_own_directory() {
    // One turn that writes a durable memory document, so the backend has to resolve its directory.
    let (base, _requests) = spawn_model_server(vec![
        turn_tool_use(
            "m1",
            "memory",
            &json!({"command":"create","path":"/memories/note.md","file_text":"kept\n"})
                .to_string(),
        ),
        turn_text("noted"),
    ]);
    let svc = Service::start(&base, &["s1", "s2"]).await;
    let token = svc.token("00tenant1", "s2.alpha");
    let mut ws = ws_connect_with_headers(svc.port, Some("s2.alpha"), &svc.header(&token))
        .await
        .unwrap();
    ws_send(
        &mut ws,
        json!({"type":"prompt","id":"p1","message":"remember this"}),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "prompt").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");

    // The session lives on its id's shard …
    assert!(svc.sessions_dir("s2", "00tenant1").is_dir());
    // … and durable memory on the grant's *home* shard, which is `s1` here, sealed with the tenant's
    // own key (the mount is shared with every other tenant on the shard).
    let doc = svc.memory_dir("s1", "00tenant1").join("note.md");
    assert!(
        !std::fs::read_to_string(&doc).unwrap().contains("kept"),
        "a memory document must not sit in plaintext on a shared mount"
    );
    let View::Document(text) = svc
        .tenant_memory("s1", "00tenant1")
        .view(&MemPath::parse("/memories/note.md").unwrap(), None)
        .await
        .unwrap()
    else {
        panic!("expected a document at {}", doc.display());
    };
    assert_eq!(
        text, "kept\n",
        "durable memory must be rooted at <home shard>/<tenant>/memory"
    );
    assert!(
        !svc.memory_dir("s2", "00tenant1").exists(),
        "memory must not follow the session's shard"
    );
}
