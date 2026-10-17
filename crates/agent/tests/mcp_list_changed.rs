//! A server's tool list is not fixed at its first boot.
//!
//! A server may add tools after the agent connected — one that enables toolsets on request and says
//! so with `notifications/tools/list_changed`, or one upgraded since the manifest cache recorded it.
//! Both used to stay invisible to the model for the life of the cache: the manifest was written at
//! first boot and never re-listed, and the notification was ignored. Real fixture server throughout.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::time::{Duration, Instant};

use beyond_ai_agent::settings::{McpServerConfig, McpTransport};
use beyond_ai_agent::tools::{mcp, mcp_manifest};
use common::{body_json, run_cmd, spawn_model_server, turn_text, turn_tool_use};
use serde_json::{Value, json};

const FIXTURE: &str = env!("CARGO_BIN_EXE_mcp_fixture_stdio_server");

fn fixture(list_changes: bool) -> McpServerConfig {
    let mut env = std::collections::BTreeMap::new();
    if list_changes {
        env.insert("MCP_FIXTURE_LIST_CHANGES".to_string(), "1".to_string());
    }
    McpServerConfig {
        name: "fixture".into(),
        events: Vec::new(),
        transport: McpTransport::Stdio {
            command: FIXTURE.into(),
            args: vec![],
            env,
        },
    }
}

fn names(live: &mcp::McpTools) -> Vec<String> {
    live.snapshot()
        .iter()
        .map(|t| t.name().to_string())
        .collect()
}

fn tool(live: &mcp::McpTools, name: &str) -> std::sync::Arc<dyn agent_core::Tool> {
    live.snapshot()
        .iter()
        .find(|t| t.name() == name)
        .unwrap_or_else(|| panic!("{name} is not offered: {:?}", names(live)))
        .clone()
}

/// `notifications/tools/list_changed` re-lists the server: the tool it added is offered and works,
/// and the call that caused the change returns only once it is — so the next request has it.
#[tokio::test(flavor = "multi_thread")]
async fn a_tool_the_server_adds_after_connect_becomes_callable() {
    let (live, _catalog, warnings) = mcp::connect_all(&[fixture(true)], Duration::ZERO, None).await;
    assert!(warnings.is_empty(), "{warnings:?}");
    assert!(!names(&live).contains(&"mcp__fixture__secret".to_string()));
    let before = live.generation();

    let unlocked = tool(&live, "mcp__fixture__unlock")
        .run(json!({}))
        .await
        .unwrap();
    assert_eq!(unlocked.text, "unlocked");

    // No waiting: the re-listing landed before `unlock` returned.
    assert!(live.generation() > before, "the change bumped the set");
    let secret = tool(&live, "mcp__fixture__secret")
        .run(json!({}))
        .await
        .unwrap();
    assert_eq!(secret.text, "secret-unlocked");
    assert!(
        names(&live).contains(&"mcp__fixture__echo".to_string()),
        "the server's other tools are still offered: {:?}",
        names(&live)
    );
    assert_eq!(
        names(&live)
            .iter()
            .filter(|n| *n == "mcp__fixture__unlock")
            .count(),
        1,
        "re-listing replaces the server's tools rather than adding a second copy"
    );
}

/// Poll until `f` holds or 10 s pass.
async fn eventually(mut f: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if f() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    f()
}

/// A manifest that no longer matches the server — a tool it lacks, one it has missing — is
/// corrected by the server's first real start: the advertised tools follow the live listing, and
/// the manifest is rewritten so the next boot starts right.
#[tokio::test(flavor = "multi_thread")]
async fn a_stale_manifest_is_rewritten_once_the_server_starts() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = mcp_manifest::ManifestDir::at(dir.path());
    let config = fixture(false);

    // A first boot records the real listing...
    let (live, _catalog, warnings) = mcp::connect_all(
        std::slice::from_ref(&config),
        Duration::ZERO,
        Some(&manifest),
    )
    .await;
    assert!(warnings.is_empty(), "{warnings:?}");
    drop(live);
    let recorded = mcp_manifest::load(&manifest, &config).expect("the first boot recorded it");
    assert!(recorded.tools.iter().any(|t| t.remote_name == "ping"));

    // ...which then goes stale: `ping` is missing, and `ghost` is a tool the server does not have.
    let mut stale = recorded.tools.clone();
    stale.retain(|t| t.remote_name != "ping");
    stale.push(mcp_manifest::CachedTool {
        remote_name: "ghost".into(),
        description: "Not on the server.".into(),
        input_schema: json!({ "type": "object", "properties": {} }),
        ui: None,
    });
    mcp_manifest::store(
        &manifest,
        &config,
        stale,
        recorded.resources.clone(),
        recorded.prompts.clone(),
        recorded.skills.clone(),
        recorded.skill_diagnostics.clone(),
    )
    .await;

    // A boot from the stale manifest advertises it, starting nothing.
    let (live, _catalog, warnings) = mcp::connect_all(
        std::slice::from_ref(&config),
        Duration::ZERO,
        Some(&manifest),
    )
    .await;
    assert!(warnings.is_empty(), "{warnings:?}");
    assert!(names(&live).contains(&"mcp__fixture__ghost".to_string()));
    assert!(!names(&live).contains(&"mcp__fixture__ping".to_string()));

    // The first call starts the server, and its listing wins.
    let echoed = tool(&live, "mcp__fixture__echo")
        .run(json!({ "text": "hi" }))
        .await
        .unwrap();
    assert_eq!(echoed.text, "hi");
    assert!(
        eventually(|| {
            let offered = names(&live);
            offered.contains(&"mcp__fixture__ping".to_string())
                && !offered.contains(&"mcp__fixture__ghost".to_string())
        })
        .await,
        "the advertised tools follow the started server: {:?}",
        names(&live)
    );
    assert!(
        eventually(|| {
            mcp_manifest::load(&manifest, &config).is_some_and(|m| {
                m.tools.iter().any(|t| t.remote_name == "ping")
                    && !m.tools.iter().any(|t| t.remote_name == "ghost")
            })
        })
        .await,
        "the manifest was rewritten with the live listing"
    );
    let rewritten = mcp_manifest::load(&manifest, &config).unwrap();
    assert_eq!(
        rewritten.tools, recorded.tools,
        "exactly what the server lists"
    );
    assert_eq!(rewritten.resources, recorded.resources, "the rest is kept");
}

fn tool_names(body: &Value) -> Vec<String> {
    body["tools"]
        .as_array()
        .map(|tools| {
            tools
                .iter()
                .filter_map(|t| t["name"].as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// End to end through the model loop: a tool the server adds during a run is in the very next
/// request, the model can call it, and its result reaches the model.
#[test]
fn a_tool_added_mid_run_is_offered_on_the_next_request() {
    let home = tempfile::tempdir().unwrap();
    let config = home.path().join(".beyond");
    std::fs::create_dir_all(&config).unwrap();
    std::fs::write(
        config.join("settings.json"),
        json!({ "mcp_servers": [{
            "name": "fixture",
            "transport": "stdio",
            "command": FIXTURE,
            "args": [],
            "env": { "MCP_FIXTURE_LIST_CHANGES": "1" },
        }] })
        .to_string(),
    )
    .unwrap();
    let (base, bodies) = spawn_model_server(vec![
        turn_tool_use("c1", "mcp__fixture__unlock", "{}"),
        turn_tool_use("c2", "mcp__fixture__secret", "{}"),
        turn_text("done"),
    ]);
    let out = run_cmd(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .env("HOME", home.path())
        .current_dir(home.path())
        .args([
            "run",
            "go",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-sonnet-5-5",
            "--no-session-persistence",
            "--mcp-tool-loading",
            "eager",
        ])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    let bodies = bodies.lock().unwrap();
    assert!(bodies.len() >= 3, "three model requests: {}", bodies.len());
    let first = tool_names(&body_json(&bodies[0]));
    let second = tool_names(&body_json(&bodies[1]));
    assert!(
        first.contains(&"mcp__fixture__unlock".to_string()),
        "{first:?}"
    );
    assert!(
        !first.contains(&"mcp__fixture__secret".to_string()),
        "{first:?}"
    );
    assert!(
        second.contains(&"mcp__fixture__secret".to_string()),
        "the next request offers the new tool: {second:?}"
    );
    assert!(
        bodies[2].contains("secret-unlocked"),
        "the new tool ran and its result reached the model: {}",
        bodies[2]
    );
}

/// `serve`: the change reaches the run in flight, and the session's next prompt — whose agent was
/// built before the change — offers it too.
#[test]
fn serve_offers_a_tool_added_mid_run_now_and_on_the_next_prompt() {
    use common::{SpawnGuarded, read_until_response, serve_cmd};
    use std::io::Write;

    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let config = home.join(".beyond");
    std::fs::create_dir_all(&config).unwrap();
    std::fs::write(
        config.join("settings.json"),
        json!({ "mcp_servers": [{
            "name": "fixture",
            "transport": "stdio",
            "command": FIXTURE,
            "args": [],
            "env": { "MCP_FIXTURE_LIST_CHANGES": "1" },
        }] })
        .to_string(),
    )
    .unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    // The fourth turn covers the one-off title call a session's first run makes.
    let (base, bodies) = spawn_model_server(vec![
        turn_tool_use("c1", "mcp__fixture__unlock", "{}"),
        turn_tool_use("c2", "mcp__fixture__secret", "{}"),
        turn_text("done"),
        turn_text("a title"),
        turn_text("again"),
    ]);
    let mut cmd = serve_cmd(env!("CARGO_BIN_EXE_beyond-ai-agent"), &base, &session_file);
    cmd.env("HOME", &home).args(["--mcp-tool-loading", "eager"]);
    let mut child = cmd.spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = common::child_frames(&mut child);

    for (id, message) in [("p1", "go"), ("p2", "again")] {
        writeln!(
            stdin,
            "{}",
            json!({ "id": id, "type": "prompt", "message": message })
        )
        .unwrap();
        stdin.flush().unwrap();
        let frames = read_until_response(&mut stdout, "prompt");
        let done = frames.last().unwrap();
        assert_eq!(done["success"], true, "{done}");
    }

    let recorded = bodies.lock().unwrap();
    let with_tools: Vec<Vec<String>> = recorded
        .iter()
        .map(|r| tool_names(&body_json(r)))
        .filter(|names| !names.is_empty())
        .collect();
    assert_eq!(with_tools.len(), 4, "four tool-carrying requests");
    assert!(!with_tools[0].contains(&"mcp__fixture__secret".to_string()));
    assert!(
        with_tools[1].contains(&"mcp__fixture__secret".to_string()),
        "the run in flight: {:?}",
        with_tools[1]
    );
    assert!(
        with_tools[3].contains(&"mcp__fixture__secret".to_string()),
        "the next prompt: {:?}",
        with_tools[3]
    );
    assert!(recorded.iter().any(|r| r.contains("secret-unlocked")));
    drop(recorded);
    drop(stdin);
    child.wait().unwrap();
}
