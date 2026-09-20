//! `--deny-path` and the tool root, in the remote world.
//!
//! A deny-list that resolves paths on the *replica* while the tool writes them in the *sandbox* is
//! not a cosmetic mismatch: it compares a string nothing is going to write, so it silently stops
//! firing. These tests write with a relative path and with a `~` path — the two spellings whose
//! resolution differs between the two worlds — and require the deny to still land.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::service::{Options, Service};
use common::{
    spawn_model_server, turn_text, turn_tool_use, ws_connect_with_headers, ws_read_until_response,
    ws_send,
};
use serde_json::json;

async fn service_denying(base: &str, glob: &str) -> Service {
    Service::start_with(
        base,
        &["s1"],
        Options {
            extra_args: vec!["--deny-path".into(), glob.into()],
            ..Default::default()
        },
    )
    .await
}

/// A relative `write` resolves against the tenant's `workspace_root` — the tool root service mode
/// sets — so a glob written against the sandbox's layout matches.
#[tokio::test]
async fn deny_path_blocks_a_relative_write_resolved_against_the_sandbox_workspace() {
    let (base, _requests) = spawn_model_server(vec![
        turn_tool_use(
            "t1",
            "write",
            &json!({"path": "config/secrets.env", "content": "KEY=1"}).to_string(),
        ),
        turn_text("blocked"),
    ]);
    let svc = service_denying(&base, "**/secrets.env").await;
    let token = svc.token("t1", "s1.alpha");
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .unwrap();
    ws_send(
        &mut ws,
        json!({"type":"prompt","id":"p1","message":"write it"}),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "prompt").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");

    let target = svc.workspace.join("config/secrets.env");
    assert!(
        !target.exists(),
        "the write must have been blocked before it ran: {}",
        target.display()
    );
    let text = serde_json::to_string(&frames).unwrap();
    assert!(
        text.contains("denied by policy"),
        "the model must be told why: {text}"
    );
    assert!(
        text.contains(&target.to_string_lossy().into_owned()),
        "the message must name the sandbox path actually about to be written: {text}"
    );
}

/// The same for `edit`, and for a `~`-rooted path: `~` expands against the **sandbox's** `$HOME`,
/// which the startup probe learned, not this process's.
#[tokio::test]
async fn deny_path_blocks_an_edit_under_the_sandboxs_own_home() {
    let secret = "~/.ssh/id_ed25519";
    let (base, _requests) = spawn_model_server(vec![
        turn_tool_use(
            "t1",
            "edit",
            &json!({"path": secret, "old_string": "a", "new_string": "b"}).to_string(),
        ),
        turn_text("blocked"),
    ]);
    let svc = service_denying(&base, "**/.ssh/**").await;
    // The file exists in the sandbox, so only the policy can be what stops the edit.
    svc.write_in_sandbox_home(".ssh/id_ed25519", "a");

    let token = svc.token("t1", "s1.alpha");
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .unwrap();
    ws_send(
        &mut ws,
        json!({"type":"prompt","id":"p1","message":"edit it"}),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "prompt").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");

    assert_eq!(
        std::fs::read_to_string(svc.sandbox_home.join(".ssh/id_ed25519")).unwrap(),
        "a",
        "the edit must have been blocked"
    );
    let text = serde_json::to_string(&frames).unwrap();
    assert!(text.contains("denied by policy"), "{text}");
    assert!(
        text.contains(
            &svc.sandbox_home
                .join(".ssh/id_ed25519")
                .to_string_lossy()
                .into_owned()
        ),
        "`~` must have expanded against the sandbox's own home: {text}"
    );
}

/// A path the glob does not cover still writes — proof the deny is the reason above, not a
/// blanket failure of relative writes in the remote world.
#[tokio::test]
async fn an_unmatched_relative_write_still_lands_in_the_sandbox_workspace() {
    let (base, _requests) = spawn_model_server(vec![
        turn_tool_use(
            "t1",
            "write",
            &json!({"path": "notes.md", "content": "fine"}).to_string(),
        ),
        turn_text("wrote it"),
    ]);
    let svc = service_denying(&base, "**/secrets.env").await;
    let token = svc.token("t1", "s1.alpha");
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .unwrap();
    ws_send(
        &mut ws,
        json!({"type":"prompt","id":"p1","message":"write it"}),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "prompt").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");

    assert_eq!(
        std::fs::read_to_string(svc.workspace.join("notes.md")).unwrap(),
        "fine",
        "a relative path must resolve under the tenant's workspace root"
    );
}
