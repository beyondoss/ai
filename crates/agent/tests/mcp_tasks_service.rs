//! SEP-2663 tasks in service mode: a session's journal lives in its sealed segments, which already
//! authenticate every line under the tenant's key, so whichever replica owns the session next
//! resumes its tasks (see `crate::mcp_resume::JournalAuth`).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;
mod mcp_tasks_env;

use std::time::{Duration, Instant};

use common::service::{Options, Service};
use common::{
    TestWs, spawn_model_server, turn_text, turn_tool_use, ws_connect_with_headers, ws_next_frame,
    ws_read_until_response, ws_send,
};
use mcp_tasks_env::Env;
use serde_json::{Value, json};

/// Attach to `session_id` on `port` with a grant naming the tasks fixture as connector `t`, retrying
/// while another replica still holds it.
async fn attach(svc: &Service, port: u16, session_id: &str, url: &str) -> TestWs {
    let mut claims = svc.claims("tenant-a", session_id, &svc.shards[0].0);
    claims.mcp = vec![("t".to_string(), url.to_string())];
    let token = svc.minter.mint(&claims, &svc.secrets());
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match ws_connect_with_headers(port, Some(session_id), &svc.header(&token)).await {
            Ok(ws) => return ws,
            Err(status) => assert!(
                Instant::now() < deadline,
                "{session_id} still refused with HTTP {status}"
            ),
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Frames until `stop` matches, failing after 30 s however busy the socket is (its keepalives
/// included).
async fn read_until(ws: &mut TestWs, what: &str, stop: impl Fn(&Value) -> bool) -> Vec<Value> {
    let mut frames = Vec::new();
    let read = tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let Some(frame) = ws_next_frame(ws).await else {
                return false;
            };
            let done = stop(&frame);
            frames.push(frame);
            if done {
                return true;
            }
        }
    })
    .await;
    match read {
        Ok(true) => frames,
        Ok(false) => panic!("socket closed before {what}: {frames:#?}"),
        Err(_) => panic!("{what}: not within 30s: {frames:#?}"),
    }
}

fn event(frame: &Value, kind: &str) -> bool {
    frame["type"] == "event" && frame["event"]["kind"] == kind && frame["event"]["id"] == "toolu_g"
}

/// A task left in flight by one replica (killed outright) is resumed by the next owner of the
/// session, another process on the same store: polled in the background on attach, its result
/// journaled, and the next prompt answered with it.
#[tokio::test]
async fn a_task_left_in_flight_on_one_replica_resumes_on_another() {
    let env = Env::new();
    let (_server, entry) = env.http_server();
    let url = entry["url"].as_str().unwrap().to_owned();
    let (base, requests) = spawn_model_server(vec![
        turn_tool_use("toolu_g", "mcp__t__gated_task", "{}"),
        turn_text("carrying on"),
        turn_text("t"),
        turn_text("t"),
    ]);
    let opts = Options {
        extra_args: vec!["--mcp-allow-private".into()],
        ..Default::default()
    };
    let mut svc = Service::start_with(&base, &["s1"], opts).await;
    let peer = svc.start_peer(&["--mcp-allow-private"]);

    let mut ws = attach(&svc, svc.port, "s1.alpha", &url).await;
    ws_send(
        &mut ws,
        json!({ "type": "prompt", "id": "p1", "message": "start the job" }),
    )
    .await;
    read_until(&mut ws, "the task's record", |f| {
        event(f, "tool_progress") && f["event"]["details"]["mcpTask"].is_object()
    })
    .await;
    // The record is journaled as it is emitted; give the append a moment, then kill the owner.
    let gets = env.methods("tasks/get").len();
    let deadline = Instant::now() + Duration::from_secs(20);
    while env.methods("tasks/get").len() == gets {
        assert!(Instant::now() < deadline, "the task was never polled");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    svc.child.kill().unwrap();
    let _ = svc.child.wait();
    drop(ws);

    let mut ws = attach(&svc, peer.port, "s1.alpha", &url).await;
    read_until(&mut ws, "the task resumed on the other replica", |f| {
        event(f, "tool_progress")
    })
    .await;
    std::fs::write(&env.gate, b"open").unwrap();
    ws_send(
        &mut ws,
        json!({ "type": "prompt", "id": "p2", "message": "what happened?" }),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "prompt").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    let end = frames
        .iter()
        .find(|f| event(f, "tool_end"))
        .unwrap_or_else(|| panic!("no tool_end for the resumed call: {frames:#?}"));
    assert_eq!(end["event"]["is_error"], false, "{end}");
    assert!(
        end["event"]["result"]
            .as_str()
            .unwrap()
            .contains("gated-done"),
        "{end}"
    );
    let sent = requests
        .lock()
        .unwrap()
        .iter()
        .rfind(|r| !r.contains("You write short titles"))
        .cloned()
        .unwrap();
    mcp_tasks_env::assert_alternates(&sent);
    let result = mcp_tasks_env::tool_result_sent(&sent, "toolu_g");
    assert!(result.to_string().contains("gated-done"), "{result}");
    assert_eq!(
        env.methods("tools/call").len(),
        1,
        "resumed, not re-invoked"
    );
}
