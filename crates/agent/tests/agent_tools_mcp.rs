//! Gate 1: `@beyond.dev/tools` is a real MCP server for **this** agent's client.
//!
//! Spawns the package CLI (Bun), connects with `sandbox: true` via `mcp::connect_granted`
//! (same path as production sandbox connectors), and runs `get_weather`.
//!
//! Needs `bun` on PATH and the workspace package at `sdk/ts/agent-tools-mcp`.
//! Skips when bun is missing or `BEYOND_AI_SKIP_AGENT_TOOLS_MCP=1`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use beyond_agent::grant::McpConnector;
use beyond_agent::tools::mcp::{self, McpEgress};
use beyond_agent::tools::web::ssrf::EgressPolicy;
use serde_json::json;

fn skip() -> bool {
    std::env::var_os("BEYOND_AI_SKIP_AGENT_TOOLS_MCP").is_some()
        || Command::new("bun").arg("--version").output().is_err()
}

fn package_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("sdk/ts/agent-tools-mcp")
        .canonicalize()
        .expect("sdk/ts/agent-tools-mcp")
}

struct ToolsMcp {
    child: Child,
    pub url: String,
}

impl Drop for ToolsMcp {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn spawn_tools_mcp(tools_dir: &Path) -> ToolsMcp {
    let pkg = package_root();
    let cli = pkg.join("src/cli.ts");
    assert!(cli.is_file(), "missing {}", cli.display());

    // Port from beyond_test_support::ports (ports_lint); drop so bun can bind it.
    let listener = beyond_test_support::ports::listener();
    let port = listener.local_addr().expect("local_addr").port();
    drop(listener);
    let bind = format!("127.0.0.1:{port}");

    let mut child = Command::new("bun")
        .arg("run")
        .arg(&cli)
        .args([
            "serve",
            "--dir",
            tools_dir.to_str().unwrap(),
            "--http",
            &bind,
        ])
        .current_dir(&pkg)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn bun tools-mcp");

    let stderr = child.stderr.take().expect("stderr");
    let mut lines = BufReader::new(stderr).lines();
    let ready = std::thread::spawn(move || {
        for line in lines.by_ref().flatten() {
            if let Some(rest) = line.strip_prefix("READY ") {
                return rest.trim().to_string();
            }
            if line.contains("READY") {
                return port.to_string();
            }
        }
        panic!("tools-mcp never printed READY on stderr");
    });
    let _ready_port = ready
        .join()
        .unwrap_or_else(|_| panic!("READY waiter failed"));
    std::thread::sleep(Duration::from_millis(50));
    ToolsMcp {
        child,
        url: format!("http://127.0.0.1:{port}/mcp"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn connect_granted_sandbox_calls_get_weather() {
    if skip() {
        eprintln!("skip: bun not on PATH (or BEYOND_AI_SKIP_AGENT_TOOLS_MCP=1)");
        return;
    }
    let tools = package_root().join("examples/weather/tools");
    assert!(tools.is_dir(), "{}", tools.display());
    let mcp = spawn_tools_mcp(&tools);

    let egress = McpEgress::new(EgressPolicy::new(true, &[])).unwrap();
    let connector = McpConnector {
        name: "tools".into(),
        url: mcp.url.clone(),
        sandbox: true,
        managed: false,
    };
    let (live, _catalog, warnings) = mcp::connect_granted(
        &[connector],
        &BTreeMap::new(),
        &egress,
        "",
        mcp::host(),
        Duration::from_secs(30),
        &BTreeMap::new(),
    )
    .await;
    assert!(
        warnings.is_empty(),
        "connect_granted warnings: {warnings:?}"
    );

    let tool = live
        .snapshot()
        .iter()
        .find(|t| t.name() == "mcp__tools__get_weather")
        .cloned()
        .unwrap_or_else(|| {
            let names: Vec<_> = live
                .snapshot()
                .iter()
                .map(|t| t.name().to_string())
                .collect();
            panic!("get_weather not advertised; have {names:?}");
        });

    let out = tool.run(json!({ "city": "SF" })).await.expect("tools/call");
    assert!(
        out.text.contains("SF") && (out.text.contains("sunny") || out.text.contains("72")),
        "unexpected tool result: {out:?}"
    );
}
