//! MCP Apps (SEP-1865, `io.modelcontextprotocol/ui`) end to end: the real `serve` daemon over its
//! WebSocket, a real MCP Apps server process (`mcp_apps_fixture_server`), and a test client playing
//! the renderer — the part of the host the headless agent hands off.
//!
//! What is proved, each against the processes rather than internal state:
//! - the extension is advertised only to connections dialed for a session whose client declared it
//!   renders apps; a session that did not, and `run`, never advertise it;
//! - a view-bearing tool call produces `mcp_app_open` (view HTML + CSP meta + tool input) then
//!   `mcp_app_result` (the whole result), while the model gets only the text content;
//! - app-only tools are hidden from the model, `ui://` resources are never model tools;
//! - the view's bridge calls (`mcp_app_request`) obey visibility, `set_mcp_enabled` and the
//!   same-server rule, and `ui/update-model-context` reaches the next user turn once;
//! - the apps manifest lets a restarted daemon declare apps without starting the server, and a
//!   reaped apps connection redials with the extension still advertised.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::time::{Duration, Instant};

use common::mcp_apps::{
    Home, app_request, command, daemon, declare_apps, home_with_fixture, of_type, open_view,
    prompt, request_with, strip_cache_control, tool_end_text, views_model,
};
use common::{
    advertised_tools, body_json, run_cmd, spawn_model_server, turn_text, turn_tool_use,
    ws_next_frame, ws_read_until_response, ws_send,
};
use serde_json::{Value, json};

#[tokio::test]
async fn a_declared_session_renders_the_view_and_the_model_gets_only_text() {
    let home = home_with_fixture(json!({}));
    let (base, bodies) = spawn_model_server(vec![
        turn_tool_use(
            "toolu_wx",
            "mcp__wx__show_weather",
            &json!({ "city": "Oslo" }).to_string(),
        ),
        turn_text("done"),
    ]);
    let sessions = tempfile::tempdir().unwrap();
    let daemon = daemon(&home.home, &base, sessions.path(), "0").await;
    let mut ws = daemon.connect("apps").await;

    let declared = declare_apps(&mut ws).await;
    assert_eq!(declared["data"]["servers"], json!(["wx"]), "{declared}");
    assert_eq!(
        declared["data"]["mime_types"],
        json!(["text/html;profile=mcp-app"])
    );
    assert_eq!(declared["data"]["spec"], "2026-01-26");
    assert!(
        home.saw_ui(),
        "the apps connection must advertise the extension"
    );

    let frames = prompt(&mut ws, "weather please").await;
    let open = of_type(&frames, "mcp_app_open");
    let result = of_type(&frames, "mcp_app_result");
    assert_eq!(open.len(), 1, "{frames:#?}");
    assert_eq!(result.len(), 1, "{frames:#?}");
    let (open, result) = (open[0], result[0]);
    let pos = |f: &Value| frames.iter().position(|x| x == f).unwrap();
    assert!(pos(open) < pos(result), "the view opens before its result");

    // The view, as the renderer needs it: same id as the tool call, HTML + CSP meta, tool input.
    assert_eq!(open["app_id"], "toolu_wx");
    assert_eq!(open["server"], "wx");
    assert_eq!(open["resource"]["uri"], "ui://apps-fixture/weather");
    assert_eq!(open["resource"]["mimeType"], "text/html;profile=mcp-app");
    assert!(
        open["resource"]["text"]
            .as_str()
            .unwrap()
            .contains("ui/initialize")
    );
    assert_eq!(
        open["resource"]["_meta"]["ui"]["csp"]["connectDomains"],
        json!(["https://api.example.test"])
    );
    assert_eq!(open["input"], json!({ "city": "Oslo" }));
    assert_eq!(open["tool"]["name"], "show_weather");
    assert_eq!(
        open["tool"]["_meta"]["ui"]["resourceUri"],
        "ui://apps-fixture/weather"
    );
    // The whole result reaches the view…
    assert_eq!(result["app_id"], "toolu_wx");
    assert_eq!(
        result["result"]["structuredContent"]["secret"],
        "STRUCTURED-ONLY-Oslo"
    );
    assert_eq!(result["result"]["_meta"]["fixture"], "result-meta-only");
    // …and only its text reaches the model.
    assert_eq!(tool_end_text(&frames), "weather-text:Oslo");

    let bodies = bodies.lock().unwrap().clone();
    assert_eq!(bodies.len(), 2);
    let tools = advertised_tools(&bodies[0]);
    for visible in [
        "mcp__wx__show_weather",
        "mcp__wx__model_only_tool",
        "mcp__wx__plain_echo",
        "mcp__wx__resource__notes",
    ] {
        assert!(tools.iter().any(|t| t == visible), "{visible} in {tools:?}");
    }
    assert!(
        !tools.iter().any(|t| t == "mcp__wx__refresh_weather"),
        "an app-only tool must not reach the model: {tools:?}"
    );
    assert!(
        !tools.iter().any(|t| t == "mcp__wx__app_only_always"),
        "{tools:?}"
    );
    assert!(
        !tools.iter().any(|t| t.contains("weather_view")),
        "a ui:// resource must not be a model tool: {tools:?}"
    );
    let second = body_json(&bodies[1]).to_string();
    assert!(second.contains("weather-text:Oslo"));
    assert!(
        !second.contains("STRUCTURED-ONLY") && !second.contains("result-meta-only"),
        "structuredContent/_meta must stay out of model context"
    );
}

#[tokio::test]
async fn an_undeclared_session_on_the_same_daemon_stays_plain() {
    let home = home_with_fixture(json!({}));
    let (base, bodies) = spawn_model_server(vec![
        turn_tool_use(
            "toolu_plain",
            "mcp__wx__show_weather",
            &json!({ "city": "Rome" }).to_string(),
        ),
        turn_text("done"),
    ]);
    let sessions = tempfile::tempdir().unwrap();
    let daemon = daemon(&home.home, &base, sessions.path(), "0").await;

    // Another session declares apps first, so the apps-flavored connection exists in this process.
    let mut renderer = daemon.connect("renderer").await;
    declare_apps(&mut renderer).await;
    assert!(home.saw_ui());
    assert!(
        home.saw_plain_tools_list(),
        "startup dialed the plain flavor"
    );

    let mut headless = daemon.connect("headless").await;
    let frames = prompt(&mut headless, "weather please").await;
    assert!(of_type(&frames, "mcp_app_open").is_empty(), "{frames:#?}");
    assert!(of_type(&frames, "mcp_app_result").is_empty());
    assert_eq!(tool_end_text(&frames), "weather-text:Rome");
    // The renderer session must not receive the headless session's view either.
    let stray =
        tokio::time::timeout(Duration::from_millis(300), ws_next_frame(&mut renderer)).await;
    assert!(
        !matches!(&stray, Ok(Some(f)) if f["type"] == "mcp_app_open"),
        "{stray:?}"
    );

    let tools = advertised_tools(&bodies.lock().unwrap()[0]);
    assert!(tools.iter().any(|t| t == "mcp__wx__show_weather"));
    assert!(!tools.iter().any(|t| t == "mcp__wx__refresh_weather"));
    // An app-only tool the server lists unconditionally is hidden in a plain session too.
    assert!(
        !tools.iter().any(|t| t == "mcp__wx__app_only_always"),
        "{tools:?}"
    );
    // The headless session's bridge is closed.
    let r = app_request(
        &mut headless,
        "wx",
        "toolu_plain",
        "tools/call",
        json!({ "name": "show_weather", "arguments": {} }),
    )
    .await;
    assert_eq!(r["success"], false);
    assert!(r["error"].as_str().unwrap().contains("set_mcp_apps"), "{r}");
}

#[tokio::test]
async fn the_bridge_proxies_a_views_calls_under_the_spec_rules() {
    // The legacy `initialize` lifecycle, so both negotiation paths are covered across the suite.
    let home = home_with_fixture(json!({ "MCP_APPS_FIXTURE_LEGACY": "1" }));
    let (base, _bodies) = views_model(1, vec![]);
    let sessions = tempfile::tempdir().unwrap();
    let daemon = daemon(&home.home, &base, sessions.path(), "0").await;
    let mut ws = daemon.connect("bridge").await;
    declare_apps(&mut ws).await;
    assert!(
        home.requests()
            .iter()
            .any(|(m, ui)| m == "initialize" && *ui),
        "{:?}",
        home.requests()
    );
    let view = open_view(&mut ws, 1).await;

    // An app-only tool: callable from the view, with its whole result.
    let r = app_request(
        &mut ws,
        "wx",
        &view,
        "tools/call",
        json!({ "name": "refresh_weather", "arguments": { "city": "Oslo" } }),
    )
    .await;
    assert_eq!(r["success"], true, "{r}");
    assert_eq!(r["data"]["result"]["content"][0]["text"], "refreshed:Oslo");
    assert_eq!(r["data"]["result"]["structuredContent"]["refreshed"], true);

    // Visibility `["model"]`: the host MUST reject it from an app.
    let r = app_request(
        &mut ws,
        "wx",
        &view,
        "tools/call",
        json!({ "name": "model_only_tool" }),
    )
    .await;
    assert_eq!(r["success"], false);
    assert!(r["error"].as_str().unwrap().contains("visibility"), "{r}");

    // A tool the server never listed is refused.
    let r = app_request(
        &mut ws,
        "wx",
        &view,
        "tools/call",
        json!({ "name": "nope" }),
    )
    .await;
    assert_eq!(r["success"], false, "{r}");

    // resources/read and ping proxy through.
    let r = app_request(
        &mut ws,
        "wx",
        &view,
        "resources/read",
        json!({ "uri": "fixture://notes" }),
    )
    .await;
    assert_eq!(
        r["data"]["result"]["contents"][0]["text"], "notes-body",
        "{r}"
    );
    let r = app_request(&mut ws, "wx", &view, "ping", json!({})).await;
    assert_eq!(r["data"]["result"], json!({}), "{r}");

    // `ui/message` is the renderer's to send as a prompt; renderer-only methods are refused.
    let r = app_request(&mut ws, "wx", &view, "ui/message", json!({})).await;
    assert!(r["error"].as_str().unwrap().contains("prompt"), "{r}");
    let r = app_request(
        &mut ws,
        "wx",
        &view,
        "ui/open-link",
        json!({ "url": "https://x" }),
    )
    .await;
    assert_eq!(r["success"], false, "{r}");

    // Session gating: a server disabled for this session is closed to its views too.
    let r = command(&mut ws, json!({ "type": "set_mcp_enabled", "servers": [] })).await;
    assert_eq!(r["success"], true, "{r}");
    let r = app_request(
        &mut ws,
        "wx",
        &view,
        "tools/call",
        json!({ "name": "refresh_weather" }),
    )
    .await;
    assert_eq!(r["success"], false);
    assert!(r["error"].as_str().unwrap().contains("disabled"), "{r}");

    // Withdrawing the declaration tears the view down and closes the bridge.
    command(
        &mut ws,
        json!({ "type": "set_mcp_enabled", "servers": null }),
    )
    .await;
    ws_send(&mut ws, json!({ "type": "set_mcp_apps", "enabled": false })).await;
    let frames = tokio::time::timeout(
        Duration::from_secs(30),
        ws_read_until_response(&mut ws, "set_mcp_apps"),
    )
    .await
    .unwrap();
    let teardown = of_type(&frames, "mcp_app_teardown");
    assert_eq!(teardown.len(), 1, "{frames:#?}");
    assert_eq!(teardown[0]["app_id"], view.as_str());
    assert_eq!(frames.last().unwrap()["data"]["enabled"], false);
    let r = app_request(&mut ws, "wx", &view, "ping", json!({})).await;
    assert_eq!(r["success"], false, "{r}");
    // …and empties the replay store: a client attaching now reopens nothing.
    drop(ws);
    let mut again = daemon.connect("bridge").await;
    let replayed =
        tokio::time::timeout(Duration::from_millis(500), ws_next_frame(&mut again)).await;
    assert!(
        !matches!(&replayed, Ok(Some(f)) if f["type"] == "mcp_app_open"),
        "{replayed:?}"
    );
}

#[tokio::test]
async fn a_views_model_context_is_a_request_only_block_and_the_cached_prefix_stays_byte_stable() {
    let home = home_with_fixture(json!({}));
    let (base, bodies) = views_model(1, vec![]);
    let sessions = tempfile::tempdir().unwrap();
    let daemon = daemon(&home.home, &base, sessions.path(), "0").await;
    let mut ws = daemon.connect("ctx").await;
    declare_apps(&mut ws).await;
    let view = open_view(&mut ws, 1).await;
    prompt(&mut ws, "zqbeforez").await;

    for text in ["stale-selection", "selected-city=Oslo"] {
        let r = app_request(
            &mut ws,
            "wx",
            &view,
            "ui/update-model-context",
            json!({ "content": [{ "type": "text", "text": text }] }),
        )
        .await;
        assert_eq!(r["success"], true, "{r}");
    }
    prompt(&mut ws, "zqfirstz").await;
    prompt(&mut ws, "zqsecondz").await;

    let bodies = bodies.lock().unwrap().clone();
    let before = request_with(&bodies, "zqbeforez");
    let first = request_with(&bodies, "zqfirstz");
    let second = request_with(&bodies, "zqsecondz");

    // The system prompt is byte-identical across the update: the prompt cache's prefix holds.
    assert_eq!(before["system"], first["system"]);
    assert_eq!(first["system"], second["system"]);
    assert!(!first["system"].to_string().contains("selected-city"));

    // Every message before the update is byte-identical in the request after it…
    let messages = |b: &Value| strip_cache_control(b["messages"].clone());
    let (m0, m1, m2) = (messages(&before), messages(&first), messages(&second));
    let prior = m0.as_array().unwrap().len();
    assert_eq!(m1.as_array().unwrap()[..prior], m0.as_array().unwrap()[..]);
    // …and the context appears only on the newest user turn, as a block of its own.
    let newest = m1.as_array().unwrap().last().unwrap();
    let blocks = newest["content"].as_array().unwrap();
    assert_eq!(
        blocks[0]["text"], "zqfirstz",
        "the user's text is untouched: {newest}"
    );
    let ctx: Vec<&Value> = blocks
        .iter()
        .filter(|b| {
            b["text"]
                .as_str()
                .is_some_and(|t| t.contains("mcp_app_context"))
        })
        .collect();
    assert_eq!(ctx.len(), 1, "{newest}");
    let ctx = ctx[0]["text"].as_str().unwrap();
    assert!(ctx.starts_with("<mcp_app_context>"), "{ctx}");
    assert!(ctx.contains("Untrusted data from MCP App views"), "{ctx}");
    assert!(ctx.contains("selected-city=Oslo") && !ctx.contains("stale-selection"));
    let others = m1.as_array().unwrap()[..m1.as_array().unwrap().len() - 1].to_vec();
    assert!(!Value::Array(others).to_string().contains("selected-city"));

    // The next turn sends the whole previous request's history byte-identical — the block still on
    // its turn (so the model keeps it) — and carries no new context.
    let first_len = m1.as_array().unwrap().len();
    assert_eq!(
        m2.as_array().unwrap()[..first_len],
        m1.as_array().unwrap()[..]
    );
    let newest = m2.as_array().unwrap().last().unwrap();
    assert_eq!(newest["content"].as_array().unwrap().len(), 1, "{newest}");

    // Never persisted as the user's words: the transcript has no trace of it.
    let r = command(&mut ws, json!({ "type": "get_messages" })).await;
    assert!(!r["data"].to_string().contains("selected-city"), "{r}");
}

#[tokio::test]
async fn an_aborted_view_call_is_reported_cancelled_and_a_missing_view_falls_back_to_text() {
    let home = home_with_fixture(json!({}));
    let (base, _bodies) = spawn_model_server(vec![
        turn_tool_use(
            "toolu_slow",
            "mcp__wx__slow_view",
            &json!({ "ms": 30_000 }).to_string(),
        ),
        turn_tool_use("toolu_broken", "mcp__wx__broken_view", "{}"),
        turn_text("done"),
    ]);
    let sessions = tempfile::tempdir().unwrap();
    let daemon = daemon(&home.home, &base, sessions.path(), "0").await;
    let mut ws = daemon.connect("abort").await;
    declare_apps(&mut ws).await;

    ws_send(&mut ws, json!({ "type": "prompt", "message": "slow" })).await;
    // The view opens while the tool is still running…
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        assert!(
            Instant::now() < deadline,
            "no mcp_app_open while the tool ran"
        );
        let f = tokio::time::timeout(Duration::from_secs(30), ws_next_frame(&mut ws))
            .await
            .unwrap()
            .unwrap();
        if f["type"] == "mcp_app_open" {
            assert_eq!(f["app_id"], "toolu_slow");
            break;
        }
    }
    // …and an abort tells the renderer the call will never produce a result.
    ws_send(&mut ws, json!({ "type": "abort" })).await;
    let frames = tokio::time::timeout(
        Duration::from_secs(30),
        ws_read_until_response(&mut ws, "prompt"),
    )
    .await
    .unwrap();
    let results = of_type(&frames, "mcp_app_result");
    assert_eq!(results.len(), 1, "{frames:#?}");
    assert_eq!(results[0]["app_id"], "toolu_slow");
    assert!(results[0]["cancelled"]["reason"].is_string(), "{frames:#?}");

    // A view whose resource is missing: no frames, the tool's text still reaches the model.
    let frames = prompt(&mut ws, "broken").await;
    assert!(of_type(&frames, "mcp_app_open").is_empty(), "{frames:#?}");
    assert_eq!(tool_end_text(&frames), "broken-view-text");
}

#[tokio::test]
async fn a_cached_apps_manifest_starts_nothing_and_a_reaped_connection_redials_with_the_extension()
{
    let home = home_with_fixture(json!({}));
    let sessions = tempfile::tempdir().unwrap();

    // First boot discovers both flavors and records both manifests.
    {
        let (base, _) = spawn_model_server(vec![]);
        let daemon = daemon(&home.home, &base, sessions.path(), "0").await;
        let mut ws = daemon.connect("m1").await;
        declare_apps(&mut ws).await;
    }
    let apps_manifest = home.home.join(".claude/mcp-manifest-apps.json");
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(&apps_manifest).unwrap()).unwrap();
    let cached = manifest["wx"]["tools"].as_array().unwrap();
    assert!(
        cached
            .iter()
            .any(|t| t["remote_name"] == "refresh_weather"
                && t["ui"]["visibility"] == json!(["app"])),
        "{manifest:#}"
    );
    std::fs::remove_file(&home.log).unwrap();

    // Second boot: declaring apps is answered from the manifest — no server process at all.
    let (base, _) = views_model(1, vec![]);
    let daemon = daemon(&home.home, &base, sessions.path(), "1").await;
    let mut ws = daemon.connect("m2").await;
    let declared = declare_apps(&mut ws).await;
    assert_eq!(declared["data"]["servers"], json!(["wx"]));
    assert!(home.requests().is_empty(), "{:?}", home.requests());

    // The first view-bearing call dials — with the extension advertised.
    let view = open_view(&mut ws, 1).await;
    assert!(home.saw_ui());
    let call = json!({ "name": "refresh_weather", "arguments": { "city": "Bergen" } });
    let r = app_request(&mut ws, "wx", &view, "tools/call", call.clone()).await;
    assert_eq!(
        r["data"]["result"]["content"][0]["text"], "refreshed:Bergen",
        "{r}"
    );
    let handshakes = |home: &Home| {
        home.requests()
            .iter()
            .filter(|(m, _)| m == "server/discover" || m == "initialize")
            .count()
    };
    let before = handshakes(&home);

    // Idle past the 1s window, the apps connection is reaped; the next call redials, still apps.
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        tokio::time::sleep(Duration::from_millis(1500)).await;
        let r = app_request(&mut ws, "wx", &view, "tools/call", call.clone()).await;
        assert_eq!(r["success"], true, "{r}");
        if handshakes(&home) > before {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the apps connection was never reaped"
        );
    }
    assert!(
        home.requests()
            .iter()
            .filter(|(m, _)| m == "tools/call")
            .all(|(_, ui)| *ui),
        "every call on the apps connection carries the extension: {:?}",
        home.requests()
    );
}

#[test]
fn run_never_advertises_the_extension() {
    let home = home_with_fixture(json!({}));
    let cwd = tempfile::tempdir().unwrap();
    let (base, bodies) = spawn_model_server(vec![
        turn_tool_use(
            "toolu_run",
            "mcp__wx__show_weather",
            &json!({ "city": "Lima" }).to_string(),
        ),
        turn_text("done"),
    ]);
    let output = run_cmd(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .env("HOME", &home.home)
        .env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0")
        .args([
            "run",
            "weather",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--max-steps",
            "4",
            "--no-session-persistence",
        ])
        .current_dir(cwd.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!home.requests().is_empty());
    assert!(!home.saw_ui(), "{:?}", home.requests());
    let bodies = bodies.lock().unwrap().clone();
    // Text fallback: the plain tool list, and the tool's text in the next turn.
    let tools = advertised_tools(&bodies[0]);
    assert!(!tools.iter().any(|t| t == "mcp__wx__refresh_weather"));
    assert!(
        !tools.iter().any(|t| t.contains("weather_view")),
        "{tools:?}"
    );
    // `run` hides an app-only tool the server lists without negotiation, too.
    assert!(
        tools.iter().any(|t| t == "mcp__wx__show_weather"),
        "{tools:?}"
    );
    assert!(
        !tools.iter().any(|t| t == "mcp__wx__app_only_always"),
        "{tools:?}"
    );
    assert!(
        body_json(&bodies[1])
            .to_string()
            .contains("weather-text:Lima")
    );
}
