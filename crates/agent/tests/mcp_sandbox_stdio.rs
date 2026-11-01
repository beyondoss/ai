//! A `command` server of `.mcp.json` runs as a job in the session's sandbox and is spoken to over
//! its stdin and stdout, through nothing but the exec protocol: a sandbox job that reads a fifo, a
//! followed output stream, and one plain exec per write.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;
use std::time::Duration;

use agent_core::tool::Tool;
use beyond_agent::exec_endpoint::ExecTarget;
use beyond_agent::exec_endpoint::jobs::JobScope;
use beyond_agent::grant::{McpConnector, SandboxCommand};
use beyond_agent::tools::mcp::{self, McpEgress, SandboxServers};
use beyond_agent::tools::web::ssrf::EgressPolicy;
use beyond_test_support::exec_mock::{ExecMock, group_alive, wait_group_gone};
use serde_json::json;

const SERVER: &str = env!("CARGO_BIN_EXE_mcp_fixture_stdio_server");

struct Box {
    exec: ExecMock,
    dir: tempfile::TempDir,
    servers: SandboxServers,
}

async fn sandbox() -> Box {
    let dir = tempfile::tempdir().unwrap();
    let exec = ExecMock::start(dir.path()).await;
    let target = ExecTarget::http(&exec.url, &[], 8 * 1024 * 1024)
        .await
        .unwrap();
    let servers = SandboxServers {
        runner: target.runner(),
        // A session of its own: the scope and the fifo under /tmp are shared by the host.
        scope: JobScope::for_session_mcp(Some("tenant"), &dir.path().to_string_lossy()),
        cwd: dir.path().to_string_lossy().into_owned(),
    };
    Box { exec, dir, servers }
}

/// The fixture, started by `sh` so it writes its pid first (`exec` keeps the pid: it is the job's
/// process-group leader).
fn connector(pidfile: &std::path::Path, env: &[(&str, &str)]) -> McpConnector {
    McpConnector {
        name: "box".into(),
        url: String::new(),
        sandbox: true,
        managed: false,
        command: Some(SandboxCommand {
            command: "sh".into(),
            args: vec![
                "-c".into(),
                format!("echo $$ > '{}'; exec \"$0\"", pidfile.display()),
                SERVER.into(),
            ],
            env: env
                .iter()
                .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
                .collect(),
        }),
    }
}

async fn connect(b: &Box, c: McpConnector) -> (mcp::McpTools, mcp::McpCatalog, Vec<String>) {
    let egress = McpEgress::new(EgressPolicy::new(true, &[])).unwrap();
    mcp::connect_granted(
        &[c],
        &BTreeMap::new(),
        &egress,
        "",
        mcp::host(),
        Duration::ZERO,
        &BTreeMap::new(),
        Some(&b.servers),
    )
    .await
}

fn tool(tools: &mcp::McpTools, name: &str) -> std::sync::Arc<dyn Tool> {
    tools
        .snapshot()
        .iter()
        .find(|t| t.name() == name)
        .unwrap_or_else(|| panic!("{name} not advertised"))
        .clone()
}

fn pid_of(file: &std::path::Path) -> u32 {
    std::fs::read_to_string(file)
        .unwrap()
        .trim()
        .parse()
        .unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_command_server_is_one_long_lived_job_in_the_sandbox_and_ends_with_its_connection() {
    let b = sandbox().await;
    let pidfile = b.dir.path().join("server.pid");
    let (tools, catalog, warnings) = connect(
        &b,
        connector(&pidfile, &[("I62_VAR", "literal-${NOT_EXPANDED}")]),
    )
    .await;
    assert!(warnings.is_empty(), "{warnings:?}");

    let echo = tool(&tools, "mcp__box__echo");
    for i in 0..3 {
        let out = echo
            .run(json!({ "text": format!("call {i}") }))
            .await
            .unwrap();
        assert!(out.text.contains(&format!("call {i}")), "{out:?}");
    }
    // The entry's env reaches the server, as written: nothing was resolved on the replica.
    let env = tool(&tools, "mcp__box__echo_env")
        .run(json!({ "var": "I62_VAR" }))
        .await
        .unwrap();
    assert!(env.text.contains("literal-${NOT_EXPANDED}"), "{env:?}");

    // One server for all of those calls: a single start, and its process the one that answered.
    let starts = b
        .exec
        .bodies()
        .iter()
        .filter(|body| body["job"]["op"] == "start")
        .count();
    assert_eq!(starts, 1, "the server must be started once per connection");
    let pid = pid_of(&pidfile);
    assert!(group_alive(pid), "the server is running in the sandbox");

    // Ending the connection stops the job: nothing is left running in the sandbox.
    drop(echo);
    drop(tools);
    drop(catalog);
    assert!(
        wait_group_gone(pid, Duration::from_secs(15)).await,
        "the server outlived its connection"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn concurrent_calls_with_large_arguments_never_interleave_on_the_fifo() {
    let b = sandbox().await;
    let pidfile = b.dir.path().join("server.pid");
    let (tools, _catalog, warnings) = connect(&b, connector(&pidfile, &[])).await;
    assert!(warnings.is_empty(), "{warnings:?}");
    let echo = tool(&tools, "mcp__box__echo");

    // Each argument is far past PIPE_BUF (4096): two unserialized `cat >> fifo` would interleave.
    for round in 0..5 {
        let calls = (0..4).map(|i| {
            let echo = echo.clone();
            let text = format!(
                "{round}-{i}-{}",
                char::from(b'a' + i as u8).to_string().repeat(200_000)
            );
            async move {
                let out = echo.run(json!({ "text": text.clone() })).await.unwrap();
                (text, out.text)
            }
        });
        for (sent, got) in futures::future::join_all(calls).await {
            assert!(
                got.contains(&sent),
                "a reply lost or mixed its argument ({} bytes)",
                got.len()
            );
        }
    }
}
