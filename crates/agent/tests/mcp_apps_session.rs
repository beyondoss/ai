//! MCP Apps across a session's life — the host duties that outlive one tool call, each proved
//! against the real `serve` daemon and `mcp_apps_fixture_server`:
//! - a view's `tools/call` waits on the same `--approve` gate (and session memory) as the model's;
//! - a client that reconnects — after the run, or in the middle of it — reopens its views;
//! - `ui/update-model-context` sent mid-run reaches the model at that run's next turn, beside a
//!   steer, and is not sent again;
//! - view HTML is read once per connection and re-read after `resources/list_changed`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::time::Duration;

use common::mcp_apps::{
    Ws, app_request, command, daemon, daemon_with, declare_apps, home_with_fixture, of_type,
    prompt, tool_end_text, view_id, views_model,
};
use common::{
    body_json, spawn_model_server_routed, turn_text, turn_tool_use, ws_next_frame, ws_send,
};
use serde_json::{Value, json};

const VIEW_URI: &str = "ui://apps-fixture/weather";

async fn next_frame(ws: &mut Ws) -> Value {
    tokio::time::timeout(Duration::from_secs(30), ws_next_frame(ws))
        .await
        .expect("a frame in time")
        .expect("socket open")
}

/// Send one bridge `tools/call` from view `app_id`, answering any approval question with
/// `decision`/`scope`. Returns the question (if one was asked) and the bridge's response.
async fn app_call_answering(
    ws: &mut Ws,
    app_id: &str,
    tool: &str,
    decision: &str,
    scope: &str,
) -> (Option<Value>, Value) {
    ws_send(
        ws,
        json!({
            "type": "mcp_app_request",
            "id": "bridge",
            "server": "wx",
            "app_id": app_id,
            "request": { "method": "tools/call",
                         "params": { "name": tool, "arguments": { "city": "Oslo", "text": "hi" } } },
        }),
    )
    .await;
    let mut asked = None;
    loop {
        let f = next_frame(ws).await;
        if f["type"] == "approval_request" {
            ws_send(
                ws,
                json!({ "type": "approve", "id": "a", "request_id": f["request_id"],
                        "decision": decision, "scope": scope }),
            )
            .await;
            asked = Some(f);
        } else if f["type"] == "response" && f["command"] == "mcp_app_request" {
            return (asked, f);
        }
    }
}

/// Run one prompt, answering every approval question with `decision`/`scope`; returns the
/// questions asked and every frame.
async fn prompt_answering(
    ws: &mut Ws,
    message: &str,
    decision: &str,
    scope: &str,
) -> (Vec<Value>, Vec<Value>) {
    ws_send(ws, json!({ "type": "prompt", "message": message })).await;
    let (mut asked, mut frames) = (Vec::new(), Vec::new());
    loop {
        let f = next_frame(ws).await;
        if f["type"] == "approval_request" {
            ws_send(
                ws,
                json!({ "type": "approve", "id": "a", "request_id": f["request_id"],
                        "decision": decision, "scope": scope }),
            )
            .await;
            asked.push(f.clone());
        }
        let done = f["type"] == "response" && f["command"] == "prompt";
        frames.push(f);
        if done {
            return (asked, frames);
        }
    }
}

#[tokio::test]
async fn a_views_tool_call_waits_on_the_same_approval_gate_and_memory_as_the_model() {
    let home = home_with_fixture(json!({}));
    let (base, _) = views_model(
        1,
        vec![
            ("toolu_echo_x".into(), turn_text("done")),
            (
                "zqechoz".into(),
                turn_tool_use(
                    "toolu_echo_x",
                    "mcp__wx__plain_echo",
                    &json!({ "text": "hi" }).to_string(),
                ),
            ),
        ],
    );
    let sessions = tempfile::tempdir().unwrap();
    let daemon = daemon_with(
        &home.home,
        &base,
        sessions.path(),
        "0",
        &["--approve", "all"],
    )
    .await;
    let mut ws = daemon.connect("gate").await;
    declare_apps(&mut ws).await;
    let calls = |h: &common::mcp_apps::Home| {
        h.requests()
            .iter()
            .filter(|(m, _)| m == "tools/call")
            .count()
    };

    // The model's `show_weather` asks; "always allow" it for the session.
    let (asked, frames) = prompt_answering(&mut ws, "zqopen1z", "allow", "session").await;
    assert_eq!(asked.len(), 1, "{asked:?}");
    assert_eq!(asked[0]["origin"], "main");
    assert_eq!(of_type(&frames, "mcp_app_open").len(), 1, "{frames:#?}");
    let view = view_id(1);

    // Model → view: the view's call of the same tool is answered by that memory, not asked again.
    let (asked, r) = app_call_answering(&mut ws, &view, "show_weather", "deny", "once").await;
    assert!(
        asked.is_none(),
        "the model's session allow covers the view: {asked:?}"
    );
    assert_eq!(r["success"], true, "{r}");

    // Asked with the ordinary approval frame, naming the tool and the view that asked.
    let (asked, r) = app_call_answering(&mut ws, &view, "refresh_weather", "allow", "once").await;
    let asked = asked.expect("an app's tools/call must ask under --approve all");
    assert_eq!(asked["tool"], "mcp__wx__refresh_weather", "{asked}");
    assert_eq!(
        asked["origin"],
        json!({ "app": { "server": "wx", "app_id": view } })
    );
    assert_eq!(r["success"], true, "{r}");
    assert_eq!(r["data"]["result"]["content"][0]["text"], "refreshed:Oslo");

    // A denial never reaches the server.
    let before = calls(&home);
    let (asked, r) = app_call_answering(&mut ws, &view, "refresh_weather", "deny", "once").await;
    assert!(asked.is_some());
    assert_eq!(r["success"], false);
    assert!(r["error"].as_str().unwrap().contains("denied"), "{r}");
    assert_eq!(calls(&home), before, "{:?}", home.requests());

    // View → model: a session allow given to the view's call answers the model's call.
    let (asked, r) = app_call_answering(&mut ws, &view, "plain_echo", "allow", "session").await;
    assert!(asked.is_some());
    assert_eq!(r["success"], true, "{r}");
    let (asked, frames) = prompt_answering(&mut ws, "zqechoz", "deny", "once").await;
    assert!(
        asked.is_empty(),
        "the view's session allow covers the model: {asked:?}"
    );
    assert_eq!(tool_end_text(&frames), "hi", "{frames:#?}");
}

#[tokio::test]
async fn a_reconnecting_client_reopens_its_views_after_and_during_a_run() {
    let home = home_with_fixture(json!({}));
    // Routed by body: first match wins, and a later prompt's body carries earlier tool ids.
    let (base, _) = spawn_model_server_routed(
        vec![
            ("Title:".into(), turn_text("title")),
            ("toolu_s".into(), turn_text("slow done")),
            (
                "zqslowx".into(),
                turn_tool_use(
                    "toolu_s",
                    "mcp__wx__slow_view",
                    &json!({ "ms": 2500 }).to_string(),
                ),
            ),
            ("toolu_r".into(), turn_text("done")),
        ],
        turn_tool_use(
            "toolu_r",
            "mcp__wx__show_weather",
            &json!({ "city": "Oslo" }).to_string(),
        ),
    );
    let sessions = tempfile::tempdir().unwrap();
    let daemon = daemon(&home.home, &base, sessions.path(), "0").await;

    let mut first = daemon.connect("replay").await;
    declare_apps(&mut first).await;
    let frames = prompt(&mut first, "weather please").await;
    assert_eq!(of_type(&frames, "mcp_app_result").len(), 1);
    drop(first);

    // After the run: the view and its result come back, marked as replays.
    let mut second = daemon.connect("replay").await;
    let (mut open, mut result) = (None, None);
    while open.is_none() || result.is_none() {
        let f = next_frame(&mut second).await;
        match f["type"].as_str() {
            Some("mcp_app_open") => open = Some(f),
            Some("mcp_app_result") => result = Some(f),
            _ => {}
        }
    }
    let (open, result) = (open.unwrap(), result.unwrap());
    assert_eq!(open["app_id"], "toolu_r");
    assert_eq!(open["replay"], true);
    assert!(
        open["resource"]["text"]
            .as_str()
            .unwrap()
            .contains("ui/initialize")
    );
    assert_eq!(result["replay"], true);
    assert_eq!(
        result["result"]["structuredContent"]["secret"],
        "STRUCTURED-ONLY-Oslo"
    );

    // During a run: a client attaching while the view's tool runs gets the open view replayed,
    // then its result live.
    ws_send(
        &mut second,
        json!({ "type": "prompt", "message": "zqslowx" }),
    )
    .await;
    loop {
        let f = next_frame(&mut second).await;
        if f["type"] == "mcp_app_open" && f["app_id"] == "toolu_s" {
            break;
        }
    }
    let mut third = daemon.connect("replay").await;
    let mut replayed = Vec::new();
    let mut saw_tool_start = false;
    let live_result = loop {
        let f = next_frame(&mut third).await;
        if f["event"]["kind"] == "tool_start" && f["event"]["id"] == "toolu_s" {
            saw_tool_start = true;
        }
        if f["type"] == "mcp_app_open" {
            assert_eq!(f["replay"], true, "{f}");
            // A view of the turn in flight is replayed right after the call it belongs to — a
            // renderer never meets a view before its `tool_start`.
            if f["app_id"] == "toolu_s" {
                assert!(
                    saw_tool_start,
                    "the view was replayed before its tool_start"
                );
            }
            replayed.push(f["app_id"].as_str().unwrap().to_owned());
        }
        if f["type"] == "mcp_app_result" && f["app_id"] == "toolu_s" {
            break f;
        }
    };
    assert!(replayed.contains(&"toolu_s".to_owned()), "{replayed:?}");
    assert!(replayed.contains(&"toolu_r".to_owned()), "{replayed:?}");
    assert!(live_result.get("replay").is_none(), "{live_result}");
    assert_eq!(live_result["result"]["content"][0]["text"], "slow-done");
}

#[tokio::test]
async fn model_context_sent_mid_run_reaches_that_runs_next_turn_beside_a_steer() {
    let home = home_with_fixture(json!({}));
    let (base, bodies) = spawn_model_server_routed(
        vec![
            ("Title:".into(), turn_text("title")),
            ("zqafterx".into(), turn_text("ok")),
            ("toolu_c".into(), turn_text("done")),
        ],
        turn_tool_use(
            "toolu_c",
            "mcp__wx__slow_view",
            &json!({ "ms": 2500 }).to_string(),
        ),
    );
    let sessions = tempfile::tempdir().unwrap();
    let daemon = daemon(&home.home, &base, sessions.path(), "0").await;
    let mut ws = daemon.connect("midrun").await;
    declare_apps(&mut ws).await;

    ws_send(&mut ws, json!({ "type": "prompt", "message": "slow work" })).await;
    loop {
        let f = next_frame(&mut ws).await;
        if f["type"] == "mcp_app_open" {
            break;
        }
    }
    // While the tool runs: the view updates the model's context, and the user steers.
    ws_send(
        &mut ws,
        json!({
            "type": "mcp_app_request", "id": "ctx", "server": "wx", "app_id": "toolu_c",
            "request": { "method": "ui/update-model-context",
                         "params": { "content": [{ "type": "text", "text": "ctx-midrun" }] } },
        }),
    )
    .await;
    ws_send(&mut ws, json!({ "type": "steer", "message": "steer-text" })).await;
    loop {
        let f = next_frame(&mut ws).await;
        if f["type"] == "response" && f["command"] == "mcp_app_request" {
            assert_eq!(f["success"], true, "{f}");
        }
        if f["type"] == "response" && f["command"] == "prompt" {
            break;
        }
    }
    prompt(&mut ws, "zqafterx").await;

    let bodies = bodies.lock().unwrap().clone();
    let last_user = |needle: &str| {
        bodies
            .iter()
            .filter(|b| !b.contains("Title:"))
            .map(|b| body_json(b)["messages"].as_array().unwrap().clone())
            .filter_map(|m| m.last().cloned())
            .map(|m| m.to_string())
            .find(|m| m.contains(needle))
            .unwrap_or_else(|| panic!("no turn carried `{needle}`"))
    };
    // The run's next turn (its tool results) carries the context and the steer together.
    let turn = last_user("toolu_c");
    assert!(turn.contains("ctx-midrun"), "{turn}");
    assert!(turn.contains("mcp_app_context"), "{turn}");
    assert!(turn.contains("steer-text"), "{turn}");
    // And it was taken: the next prompt does not carry it again.
    let after = last_user("zqafterx");
    assert!(!after.contains("ctx-midrun"), "{after}");
}

#[tokio::test]
async fn view_html_is_cached_per_connection_until_the_server_says_resources_changed() {
    let home = home_with_fixture(json!({}));
    let call = |id: &str| {
        turn_tool_use(
            id,
            "mcp__wx__show_weather",
            &json!({ "city": "Oslo" }).to_string(),
        )
    };
    let (base, _) = spawn_model_server_routed(
        vec![
            ("Title:".into(), turn_text("title")),
            ("toolu_3".into(), turn_text("done")),
            ("zq3x".into(), call("toolu_3")),
            ("toolu_2".into(), turn_text("done")),
            ("zq2x".into(), call("toolu_2")),
            ("toolu_1".into(), turn_text("done")),
        ],
        call("toolu_1"),
    );
    let sessions = tempfile::tempdir().unwrap();
    let daemon = daemon(&home.home, &base, sessions.path(), "0").await;
    let mut ws = daemon.connect("cache").await;
    declare_apps(&mut ws).await;

    let html = |frames: &[Value]| {
        of_type(frames, "mcp_app_open")
            .first()
            .unwrap_or_else(|| panic!("no view: {frames:#?}"))["resource"]["text"]
            .as_str()
            .unwrap()
            .to_owned()
    };
    let one = prompt(&mut ws, "zq1x").await;
    let two = prompt(&mut ws, "zq2x").await;
    assert!(html(&one).contains("weather v1") && html(&two).contains("weather v1"));
    assert_eq!(
        home.reads_of(VIEW_URI),
        1,
        "the second call is served from the cache"
    );

    // The server publishes a new view and says so; the next call reads it again.
    let r = app_request(
        &mut ws,
        "wx",
        "toolu_2",
        "tools/call",
        json!({ "name": "change_view" }),
    )
    .await;
    assert_eq!(r["data"]["result"]["content"][0]["text"], "view-v2", "{r}");
    // `list_changed` is handled on the client's own task; give it a moment to land.
    tokio::time::sleep(Duration::from_millis(200)).await;
    let three = prompt(&mut ws, "zq3x").await;
    assert!(html(&three).contains("weather v2"), "{}", html(&three));
    assert_eq!(home.reads_of(VIEW_URI), 2);

    // The bridge's own `resources/read` is never cached: a view asking reads what is there.
    let r = command(
        &mut ws,
        json!({ "type": "mcp_app_request", "id": "r", "server": "wx", "app_id": "toolu_3",
                "request": { "method": "resources/read", "params": { "uri": VIEW_URI } } }),
    )
    .await;
    assert_eq!(r["success"], true, "{r}");
    assert_eq!(home.reads_of(VIEW_URI), 3);
}

#[tokio::test]
async fn a_views_context_rides_its_turn_across_a_restart_and_a_session_switch() {
    let home = home_with_fixture(json!({}));
    let sessions = tempfile::tempdir().unwrap();
    let request_for = |bodies: &[String], needle: &str| {
        common::mcp_apps::request_with(bodies, needle)["messages"].clone()
    };
    let strip = common::mcp_apps::strip_cache_control;

    // First daemon: open a view, report context, and let one turn carry it.
    let first_turn = {
        let (base, bodies) = views_model(1, vec![]);
        let daemon = daemon(&home.home, &base, sessions.path(), "0").await;
        let mut ws = daemon.connect("persist").await;
        declare_apps(&mut ws).await;
        let view = common::mcp_apps::open_view(&mut ws, 1).await;
        let r = app_request(
            &mut ws,
            "wx",
            &view,
            "ui/update-model-context",
            json!({ "content": [{ "type": "text", "text": "kept-across-restart" }] }),
        )
        .await;
        assert_eq!(r["success"], true, "{r}");
        prompt(&mut ws, "zqfirstz").await;
        // One more command: the idle loop has written the sidecar before reading it.
        let _ = command(&mut ws, json!({ "type": "get_state" })).await;
        let bodies = bodies.lock().unwrap().clone();
        strip(request_for(&bodies, "zqfirstz"))
    };
    assert!(first_turn.to_string().contains("kept-across-restart"));
    let n = first_turn.as_array().unwrap().len();

    // Second daemon, same sessions: the next request carries that turn byte-identically.
    let (base, bodies) = views_model(0, vec![]);
    let daemon = daemon(&home.home, &base, sessions.path(), "0").await;
    let mut ws = daemon.connect("persist").await;
    prompt(&mut ws, "zqsecondz").await;
    let after_restart = strip(request_for(&bodies.lock().unwrap().clone(), "zqsecondz"));
    assert_eq!(
        after_restart.as_array().unwrap()[..n],
        first_turn.as_array().unwrap()[..],
        "the restarted request must reproduce the earlier history exactly"
    );

    // Switch away and back: still there.
    let mut elsewhere = daemon.connect("elsewhere").await;
    prompt(&mut elsewhere, "zqelsewherez").await;
    drop(elsewhere);
    for target in ["elsewhere", "persist"] {
        let r = command(
            &mut ws,
            json!({ "type": "switch_session", "session_id": target }),
        )
        .await;
        assert_eq!(r["success"], true, "{r}");
    }
    prompt(&mut ws, "zqthirdz").await;
    let after_switch = strip(request_for(&bodies.lock().unwrap().clone(), "zqthirdz"));
    assert_eq!(
        after_switch.as_array().unwrap()[..n],
        first_turn.as_array().unwrap()[..],
    );
    // Never in the transcript a client reads.
    let r = command(&mut ws, json!({ "type": "get_messages" })).await;
    assert!(
        !r["data"].to_string().contains("kept-across-restart"),
        "{r}"
    );
}

#[tokio::test]
async fn a_question_the_server_asks_during_a_views_call_reaches_that_views_session() {
    let home = home_with_fixture(json!({}));
    let (base, _) = views_model(1, vec![]);
    let sessions = tempfile::tempdir().unwrap();
    let daemon = daemon(&home.home, &base, sessions.path(), "0").await;
    // A bystander session shares the same (process-wide) apps connection.
    let mut bystander = daemon.connect("bystander").await;
    declare_apps(&mut bystander).await;
    let mut ws = daemon.connect("asker").await;
    declare_apps(&mut ws).await;
    let view = common::mcp_apps::open_view(&mut ws, 1).await;

    // The view calls a tool whose server asks the client a question mid-call.
    ws_send(
        &mut ws,
        json!({
            "type": "mcp_app_request", "id": "ask", "server": "wx", "app_id": view,
            "request": { "method": "tools/call", "params": { "name": "ask_view" } },
        }),
    )
    .await;
    let asked = loop {
        let f = next_frame(&mut ws).await;
        if f["type"] == "elicitation_request" {
            break f;
        }
        assert!(
            !(f["type"] == "response" && f["command"] == "mcp_app_request"),
            "the call finished without asking this session: {f}"
        );
    };
    assert_eq!(asked["server"], "wx", "{asked}");
    ws_send(
        &mut ws,
        json!({ "type": "elicit", "request_id": asked["request_id"], "action": "accept",
                "content": { "name": "Ferris" } }),
    )
    .await;
    let r = loop {
        let f = next_frame(&mut ws).await;
        if f["type"] == "response" && f["command"] == "mcp_app_request" {
            break f;
        }
    };
    assert_eq!(r["success"], true, "{r}");
    assert_eq!(
        r["data"]["result"]["content"][0]["text"], "asked:Ferris",
        "{r}"
    );
    // The other session was never asked.
    let stray =
        tokio::time::timeout(Duration::from_millis(500), ws_next_frame(&mut bystander)).await;
    assert!(
        !matches!(&stray, Ok(Some(f)) if f["type"] == "elicitation_request"),
        "{stray:?}"
    );
}
