//! An idle MCP server's process is reaped, and comes back when it is needed again.
//!
//! Why this earns its own suite: an MCP server is a whole language runtime sitting on a box waiting
//! to be asked something. Measured on the vps primitive, `@playwright/mcp` holds **63.9 MB of
//! anonymous memory** while completely idle — 87% of everything anonymous on a guest whose actual job
//! is the agent — and 66.7 MB of that is a single `require("playwright-core")`, loading a browser API
//! before any browser exists.
//!
//! These assert on **process count**, from the operating system, not on internal state. A reap that
//! leaves the child alive reclaims nothing, and a reconnect that reports success without a working
//! call is worse than never reaping, so both are checked against reality.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::time::{Duration, Instant};

use beyond_ai_agent::settings::{McpServerConfig, McpTransport};
use beyond_ai_agent::tools::mcp;
use serde_json::json;

const FIXTURE: &str = env!("CARGO_BIN_EXE_mcp_fixture_stdio_server");

/// How many fixture servers carrying `tag` are alive, read straight from `/proc`.
///
/// Not `pgrep`: matching the absolute binary path against a full cmdline turned out not to work here,
/// and matching the bare name would count the *other* test in this file, which runs concurrently in
/// the same process. Each test tags its servers with a unique argument instead, so the count is
/// exactly its own — no cross-test interference, no shelling out.
fn fixture_processes(tag: &str) -> usize {
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return 0;
    };
    entries
        .filter_map(Result::ok)
        .filter(|e| {
            let name = e.file_name();
            let Some(name) = name.to_str() else {
                return false;
            };
            if !name.bytes().all(|b| b.is_ascii_digit()) {
                return false;
            }
            let Ok(cmdline) = std::fs::read(format!("/proc/{name}/cmdline")) else {
                return false;
            };
            // cmdline is NUL-separated argv; the tag is its own argument.
            let c = String::from_utf8_lossy(&cmdline);
            c.contains(FIXTURE) && c.contains(tag)
        })
        .count()
}

/// Poll until `f` holds or `limit` elapses. Reaping is driven by a timer, so the alternative is a
/// fixed sleep long enough for the slowest machine — which is either flaky or slow.
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

/// A fixture server tagged so only this test counts its own processes.
fn fixture_config(tag: &str) -> McpServerConfig {
    McpServerConfig {
        name: "fixture".into(),
        transport: McpTransport::Stdio {
            command: FIXTURE.into(),
            // The fixture ignores argv entirely; this exists purely to be visible in /proc.
            args: vec!["--reap-test-tag".into(), tag.into()],
            env: Default::default(),
        },
    }
}

/// The whole point, end to end: the tool works, the idle process goes away, and the next call brings
/// it back and still works. Reaping without a working reconnect would just be breakage.
#[tokio::test(flavor = "multi_thread")]
async fn an_idle_server_is_reaped_and_reconnects_on_the_next_call() {
    let tag = "reap-idle-abc123";
    assert_eq!(fixture_processes(tag), 0, "tag must start unused");

    // A one-second window: the production default is minutes, which no test can wait out.
    let (tools, warnings) =
        mcp::connect_all(&[fixture_config(tag)], Duration::from_secs(1), None).await;
    assert!(
        warnings.is_empty(),
        "fixture failed to connect: {warnings:?}"
    );
    let tool = tools
        .iter()
        .find(|t| t.name().starts_with("mcp__fixture__"))
        .expect("fixture advertised no tools")
        .clone();

    assert!(
        within(Duration::from_secs(5), || fixture_processes(tag) == 1).await,
        "connecting should have started a fixture server"
    );

    let first = tool.run(json!({})).await;
    assert!(first.is_ok(), "first call failed: {first:?}");

    // Reaped purely by going idle — nothing here asks for it.
    assert!(
        within(Duration::from_secs(20), || fixture_processes(tag) == 0).await,
        "an idle server's process should have been reaped within the idle window"
    );

    // ...and the tool still works, which is what makes reaping safe rather than merely cheap.
    let second = tool.run(json!({})).await;
    assert!(
        second.is_ok(),
        "call after a reap should have reconnected: {second:?}"
    );
    assert!(
        within(Duration::from_secs(5), || fixture_processes(tag) == 1).await,
        "reconnecting should have started a fresh fixture server"
    );

    // Dropping the tools drops the connection, which kills the child: no reaper needed for cleanup.
    drop(tool);
    drop(tools);
    assert!(
        within(Duration::from_secs(10), || fixture_processes(tag) == 0).await,
        "dropping the last tool should have taken the server process with it"
    );
}

/// A window of zero disables reaping entirely — the escape hatch for an operator who would rather pay
/// the memory than ever re-spawn. Asserted because "0 means off" is exactly the kind of contract that
/// silently becomes "0 means reap instantly".
#[tokio::test(flavor = "multi_thread")]
async fn a_zero_window_keeps_the_server_resident() {
    let tag = "reap-zero-def456";
    assert_eq!(fixture_processes(tag), 0, "tag must start unused");
    let (tools, warnings) = mcp::connect_all(&[fixture_config(tag)], Duration::ZERO, None).await;
    assert!(
        warnings.is_empty(),
        "fixture failed to connect: {warnings:?}"
    );

    assert!(
        within(Duration::from_secs(5), || fixture_processes(tag) == 1).await,
        "connecting should have started a fixture server"
    );
    tokio::time::sleep(Duration::from_secs(3)).await;
    assert!(
        fixture_processes(tag) == 1,
        "a zero window must keep the server resident, not reap it immediately"
    );
    drop(tools);
}

/// The point of the manifest: once a server's tools are known, a later start advertises them and
/// runs **nothing**. This is the difference between "reaped after 120s" and "never started" — a guest
/// that boots and is never asked to browse should spawn no browser server at any moment.
///
/// Asserted on process count, because a cache that still spawns is just reaping with extra steps.
#[tokio::test(flavor = "multi_thread")]
async fn a_cached_manifest_advertises_tools_without_starting_the_server() {
    let tag = "reap-manifest-ghi789";
    let dir = std::env::temp_dir().join(format!("mcp-manifest-test-{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    let manifest = beyond_ai_agent::tools::mcp_manifest::ManifestDir::at(&dir);

    // First start: nothing cached, so it connects, discovers, and records what it found.
    let (tools, warnings) = mcp::connect_all(
        &[fixture_config(tag)],
        Duration::from_secs(60),
        Some(&manifest),
    )
    .await;
    assert!(
        warnings.is_empty(),
        "fixture failed to connect: {warnings:?}"
    );
    let discovered: Vec<String> = tools.iter().map(|t| t.name().to_string()).collect();
    assert!(!discovered.is_empty(), "first start discovered no tools");
    assert!(
        within(Duration::from_secs(5), || fixture_processes(tag) == 1).await,
        "the first start should have run the server in order to discover"
    );

    drop(tools);
    assert!(
        within(Duration::from_secs(10), || fixture_processes(tag) == 0).await,
        "dropping the tools should have taken the server with it"
    );

    // Second start: same invocation, so the manifest answers and no process appears.
    let (cached, warnings) = mcp::connect_all(
        &[fixture_config(tag)],
        Duration::from_secs(60),
        Some(&manifest),
    )
    .await;
    assert!(warnings.is_empty(), "cached start warned: {warnings:?}");
    let from_cache: Vec<String> = cached.iter().map(|t| t.name().to_string()).collect();
    assert_eq!(
        from_cache, discovered,
        "the cached start must advertise exactly what discovery found"
    );

    // The assertion this test exists for. Given a moment to be wrong, then checked.
    tokio::time::sleep(Duration::from_secs(3)).await;
    assert_eq!(
        fixture_processes(tag),
        0,
        "a cached start must run no server at all"
    );

    // ...and calling one still works, by dialing on demand.
    let tool = cached
        .iter()
        .find(|t| t.name().starts_with("mcp__fixture__"))
        .expect("cached tools");
    let called = tool.run(json!({})).await;
    assert!(
        called.is_ok(),
        "a tool from the cache must still be callable: {called:?}"
    );
    assert!(
        within(Duration::from_secs(5), || fixture_processes(tag) == 1).await,
        "the call should have started the server on demand"
    );

    drop(cached);
    let _ = within(Duration::from_secs(10), || fixture_processes(tag) == 0).await;
    let _ = std::fs::remove_dir_all(&dir);
}
