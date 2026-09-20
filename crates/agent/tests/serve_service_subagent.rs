//! Subagents in service mode: a child runs in the same sandbox its parent does, sees the same
//! project context, and cannot reach around the isolation by asking for a git worktree.
//!
//! The delegation path is where a containment mistake hides best — the parent's own tools can be
//! perfectly sandboxed while a child it spawns quietly runs on the replica.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::service::{Options, Service, host_claude_home};
use common::{
    spawn_model_server, turn_text, turn_tool_use, ws_connect_with_headers, ws_read_until_response,
    ws_send,
};
use serde_json::json;

/// An agent definition planted only in the sandbox is discoverable, delegable, and its child's
/// `write` lands in the sandbox — not on the replica.
#[tokio::test]
async fn a_sandbox_defined_subagent_writes_into_the_sandbox() {
    let (host_home, markers) = host_claude_home();
    let (base, requests) = spawn_model_server(vec![
        // The parent delegates.
        turn_tool_use(
            "call-1",
            "subagent",
            &json!({ "agent": "scribe", "task": "leave a note" }).to_string(),
        ),
        // The child writes, with a relative path — resolved against the tenant's workspace root.
        turn_tool_use(
            "call-2",
            "write",
            &json!({ "path": "child-note.md", "content": "WRITTEN-BY-THE-CHILD" }).to_string(),
        ),
        turn_text("note left"),
        turn_text("the scribe left a note"),
    ]);
    let svc = Service::start_with(
        &base,
        &["s1"],
        Options {
            host_home: Some(host_home.path().to_path_buf()),
            ..Default::default()
        },
    )
    .await;
    svc.write_in_workspace(
        ".claude/agents/scribe.md",
        "---\nname: scribe\ndescription: leaves notes\ntools: write\n---\nYou are SANDBOX-CHILD-MARKER.",
    );
    svc.write_in_workspace("AGENTS.md", "SANDBOX-CONTEXT-FOR-THE-CHILD");

    let token = svc.token("t1", "s1.alpha");
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .unwrap();
    ws_send(
        &mut ws,
        json!({"type":"prompt","id":"p1","message":"delegate"}),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "prompt").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");

    let landed = svc.workspace.join("child-note.md");
    assert_eq!(
        std::fs::read_to_string(&landed).unwrap(),
        "WRITTEN-BY-THE-CHILD",
        "the child's relative write must land in the tenant's workspace: {}",
        landed.display()
    );

    // The child's own request (the second one) carries the sandbox definition's body and the
    // parent's already-fetched project context — and none of the replica's `~/.claude`.
    let raw = requests.lock().unwrap();
    assert!(raw.len() >= 2, "the child must have called the model");
    let child = &raw[1];
    assert!(child.contains("SANDBOX-CHILD-MARKER"), "{child}");
    assert!(
        child.contains("SANDBOX-CONTEXT-FOR-THE-CHILD"),
        "the child reuses the parent's fetched context rather than walking the replica: {child}"
    );
    for marker in &markers {
        assert!(
            !child.contains(marker),
            "{marker} reached the child: {child}"
        );
    }
}

/// `isolation: worktree` is a host `git worktree` against the parent's cwd. In service mode that cwd
/// is inside the sandbox, so the call is refused with the reason — never silently downgraded to the
/// shared root, which would hand a write-capable child exactly what the field exists to prevent.
#[tokio::test]
async fn worktree_isolation_is_refused_when_the_childs_filesystem_is_remote() {
    let (base, _requests) = spawn_model_server(vec![
        turn_tool_use(
            "call-1",
            "subagent",
            &json!({ "agent": "isolated", "task": "do work" }).to_string(),
        ),
        turn_text("gave up"),
    ]);
    let svc = Service::start(&base, &["s1"]).await;
    svc.write_in_workspace(
        ".claude/agents/isolated.md",
        "---\nname: isolated\ndescription: wants its own tree\ntools: write\nisolation: worktree\n---\nBody.",
    );

    let token = svc.token("t1", "s1.alpha");
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .unwrap();
    ws_send(
        &mut ws,
        json!({"type":"prompt","id":"p1","message":"delegate"}),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "prompt").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");

    let text = serde_json::to_string(&frames).unwrap();
    assert!(
        text.contains("isolation: worktree"),
        "the refusal must name the reason: {text}"
    );
    assert!(
        text.contains("remote"),
        "the refusal must say why it is unavailable: {text}"
    );
}
