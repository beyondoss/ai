//! A stdio MCP server that dies is dialed again on the next call. Servers crash — an OOM kill, an
//! unhandled exception, a `kill -9` from an operator — and a connection that keeps handing out the
//! dead client fails every later call with "transport closed" until the whole agent restarts.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::time::{Duration, Instant};

use beyond_ai_agent::settings::{McpServerConfig, McpTransport};
use beyond_ai_agent::tools::mcp;
use serde_json::json;

const FIXTURE: &str = env!("CARGO_BIN_EXE_mcp_fixture_stdio_server");

/// The pids of fixture servers carrying `tag`, read from `/proc`.
fn fixture_pids(tag: &str) -> Vec<i32> {
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return Vec::new();
    };
    entries
        .filter_map(Result::ok)
        .filter_map(|e| {
            let pid: i32 = e.file_name().to_str()?.parse().ok()?;
            let cmdline = std::fs::read(format!("/proc/{pid}/cmdline")).ok()?;
            let c = String::from_utf8_lossy(&cmdline);
            (c.contains(FIXTURE) && c.contains(tag)).then_some(pid)
        })
        .collect()
}

async fn within(limit: Duration, mut f: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + limit;
    while Instant::now() < deadline {
        if f() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    f()
}

fn fixture_config(tag: &str) -> McpServerConfig {
    McpServerConfig {
        name: "fixture".into(),
        events: Vec::new(),
        transport: McpTransport::Stdio {
            command: FIXTURE.into(),
            // The fixture ignores argv; the tag exists to be found in /proc.
            args: vec!["--redial-test-tag".into(), tag.into()],
            env: Default::default(),
        },
    }
}

fn kill9(pid: i32) {
    let status = std::process::Command::new("kill")
        .args(["-9", &pid.to_string()])
        .status()
        .unwrap();
    assert!(status.success());
}

/// `kill -9` the server between calls: the next call — a tool, then a resource read — redials once
/// and succeeds, with exactly one fresh process behind it.
#[tokio::test(flavor = "multi_thread")]
async fn a_killed_server_is_redialed_on_the_next_call() {
    let tag = &format!("redial-kill-{}", std::process::id());
    // Never reaped: the redial must come from noticing the death, not from an idle reap.
    let (tools, _catalog, warnings) =
        mcp::connect_all(&[fixture_config(tag)], Duration::ZERO, None).await;
    assert!(warnings.is_empty(), "{warnings:?}");
    let tool = |name: &str| {
        tools
            .snapshot()
            .iter()
            .find(|t| t.name() == name)
            .unwrap_or_else(|| panic!("{name} not advertised"))
            .clone()
    };
    let echo = tool("mcp__fixture__echo");
    let doc = tool("mcp__fixture__resource__doc");

    let first = echo.run(json!({ "text": "before" })).await.unwrap();
    assert!(first.text.contains("before"), "{first:?}");
    let pids = fixture_pids(tag);
    assert_eq!(pids.len(), 1, "{pids:?}");
    kill9(pids[0]);
    assert!(
        within(Duration::from_secs(5), || fixture_pids(tag).is_empty()).await,
        "the killed server should be gone"
    );
    // Let the client's read loop see the EOF, as it would in the model round trip between calls.
    tokio::time::sleep(Duration::from_millis(200)).await;

    let after = tokio::time::timeout(
        Duration::from_secs(20),
        echo.run(json!({ "text": "after" })),
    )
    .await
    .expect("the call after a crash hung");
    assert!(
        after.as_ref().is_ok_and(|o| o.text.contains("after")),
        "the call after a crash should redial and succeed: {after:?}"
    );
    let fresh = fixture_pids(tag);
    assert_eq!(fresh.len(), 1, "one redial, one process: {fresh:?}");
    assert_ne!(fresh[0], pids[0]);

    // Every tool of the server shares the connection: a second crash, and the resource read redials.
    kill9(fresh[0]);
    assert!(within(Duration::from_secs(5), || fixture_pids(tag).is_empty()).await);
    tokio::time::sleep(Duration::from_millis(200)).await;
    let read = tokio::time::timeout(Duration::from_secs(20), doc.run(json!({})))
        .await
        .expect("the resource read after a crash hung");
    assert!(
        read.as_ref()
            .is_ok_and(|o| o.text.contains("fixture-resource-body")),
        "{read:?}"
    );
    assert_eq!(fixture_pids(tag).len(), 1);
}
