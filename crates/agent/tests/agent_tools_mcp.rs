//! `@beyond.dev/tools` against this agent's MCP client.
//!
//! Spawns the package CLI (Bun) and connects with `sandbox: true` via `mcp::connect_granted`
//! (same path as production sandbox connectors).
//!
//! - `connect_granted_sandbox_calls_get_weather` dials the live server and calls `get_weather`.
//! - `saved_menu_honors_ends_turn_from_the_live_call` stamps `.mcp-tools.json` with the guest,
//!   advertises that menu without a handshake, then `tools/call`s the live server. `endsTurn`
//!   sets `ToolOutput.terminate` and `availableInSubagents: false` hides the tool for a sandbox
//!   connector, whether or not a menu was saved. A connector that is not a sandbox one gets neither.
//! - The `serve --service` tests run the agent binary: a sandbox connector's successful `endsTurn`
//!   call ends the run without another model turn, with or without a saved menu and in an
//!   apps-capable session. A connector that is not `mcp_sandbox` does not end it.
//!
//! Needs `bun` on PATH and the workspace package at `sdk/ts/agent-tools-mcp` (with the workspace's
//! `npm ci` done), or `TOOLS_MCP_BIN=<path to the compiled tools-mcp>` to run the same tests against
//! the `npm run compile` binary instead of `bun src/cli.ts`.
//! Skips when neither is available or `BEYOND_AI_SKIP_AGENT_TOOLS_MCP=1`; under `CI` a missing
//! runtime fails the test instead, so a misconfigured job cannot pass without running them.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

use agent_core::Tool;
use beyond_agent::grant::McpConnector;
use beyond_agent::tools::mcp::{self, McpEgress, McpTools};
use beyond_agent::tools::mcp_manifest::CachedTool;
use beyond_agent::tools::web::ssrf::EgressPolicy;
use serde_json::json;

/// The compiled binary under test, when `TOOLS_MCP_BIN` names one.
fn compiled_bin() -> Option<PathBuf> {
    std::env::var_os("TOOLS_MCP_BIN")
        .map(|p| PathBuf::from(p).canonicalize().expect("TOOLS_MCP_BIN"))
}

fn skip() -> bool {
    if std::env::var_os("BEYOND_AI_SKIP_AGENT_TOOLS_MCP").is_some() {
        return true;
    }
    // The guest modules are TypeScript either way, but the compiled binary carries its own runtime.
    if compiled_bin().is_some() || Command::new("bun").arg("--version").output().is_ok() {
        return false;
    }
    assert!(
        std::env::var_os("CI").is_none(),
        "bun is not on PATH and CI is set: these tests would pass without running"
    );
    true
}

/// `tools-mcp <args>`: the compiled binary, else the package CLI under Bun.
fn tools_mcp_cmd() -> Command {
    match compiled_bin() {
        Some(tools_mcp) => Command::new(tools_mcp),
        None => {
            let mut cmd = Command::new("bun");
            cmd.arg("run").arg(package_root().join("src/cli.ts"));
            cmd
        }
    }
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
    assert!(pkg.join("src/cli.ts").is_file());

    // Port from beyond_test_support::ports (ports_lint); drop so bun can bind it.
    let listener = beyond_test_support::ports::listener();
    let port = listener.local_addr().expect("local_addr").port();
    drop(listener);
    let bind = format!("127.0.0.1:{port}");

    let mut child = tools_mcp_cmd()
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
    assert!(!out.terminate, "get_weather does not end the run: {out:?}");
}

const ENDS_TURN_SENTENCE: &str = "Calling this tool ends your turn once it succeeds: do not write a reply or call other tools in the same step. If it fails, you will see the error and can continue.";

/// Guest modules plus a `node_modules` link so Bun resolves `zod` from the package.
struct TurnTools {
    root: PathBuf,
}

impl TurnTools {
    fn new() -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("tools-mcp-e2e-{nanos}"));
        let tools = root.join("tools");
        std::fs::create_dir_all(&tools).unwrap();
        // Zod is hoisted to the workspace `sdk/ts/node_modules`, not this package's.
        // A link at the temp root is where Bun stops walking.
        let zod_modules = package_root()
            .join("../node_modules")
            .canonicalize()
            .expect("sdk/ts/node_modules");
        std::os::unix::fs::symlink(zod_modules, root.join("node_modules")).unwrap();
        let eve = package_root().join("src/eve-tools.ts");
        let eve = eve.display();
        std::fs::write(
            tools.join("finish.ts"),
            format!(
                r#"import {{ z }} from "zod";
import {{ defineTool }} from "{eve}";
export default defineTool({{
  description: "Finish the turn.",
  inputSchema: z.object({{}}),
  endsTurn: true,
  async execute() {{ return {{ done: true }}; }},
}});
"#
            ),
        )
        .unwrap();
        std::fs::write(
            tools.join("stay.ts"),
            format!(
                r#"import {{ z }} from "zod";
import {{ defineTool }} from "{eve}";
export default defineTool({{
  description: "Keep going.",
  inputSchema: z.object({{}}),
  async execute() {{ return {{ stayed: true }}; }},
}});
"#
            ),
        )
        .unwrap();
        std::fs::write(
            tools.join("decide.ts"),
            format!(
                r#"import {{ z }} from "zod";
import {{ defineTool }} from "{eve}";
export default defineTool({{
  description: "Stop only when asked.",
  inputSchema: z.object({{ stop: z.boolean() }}),
  endsTurn: (output) => output.stop === true,
  async execute({{ stop }}) {{ return {{ stop }}; }},
}});
"#
            ),
        )
        .unwrap();
        std::fs::write(
            tools.join("boom.ts"),
            format!(
                r#"import {{ z }} from "zod";
import {{ defineTool }} from "{eve}";
export default defineTool({{
  description: "Fail.",
  inputSchema: z.object({{}}),
  endsTurn: true,
  async execute() {{ throw new Error("nope"); }},
}});
"#
            ),
        )
        .unwrap();
        std::fs::write(
            tools.join("quiet.ts"),
            format!(
                r#"import {{ z }} from "zod";
import {{ defineTool }} from "{eve}";
export default defineTool({{
  description: "Hidden from subagents.",
  inputSchema: z.object({{}}),
  availableInSubagents: false,
  async execute() {{ return {{ quiet: true }}; }},
}});
"#
            ),
        )
        .unwrap();
        Self { root }
    }

    fn tools_dir(&self) -> PathBuf {
        self.root.join("tools")
    }

    /// The guest's own manifest writer. The host parses this file as `Dial::saved`.
    fn manifest(&self) -> BTreeMap<String, Vec<CachedTool>> {
        let out = self.root.join(".mcp-tools.json");
        let output = tools_mcp_cmd()
            .arg("manifest")
            .arg("--dir")
            .arg(self.tools_dir())
            .args(["--name", "tools", "-o"])
            .arg(&out)
            .current_dir(package_root())
            .output()
            .expect("spawn bun manifest");
        assert!(
            output.status.success(),
            "manifest failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap()
    }
}

impl Drop for TurnTools {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn pick(live: &McpTools, name: &str) -> Arc<dyn Tool> {
    live.snapshot()
        .iter()
        .find(|t| t.name() == name)
        .cloned()
        .unwrap_or_else(|| {
            let names: Vec<_> = live
                .snapshot()
                .iter()
                .map(|t| t.name().to_string())
                .collect();
            panic!("{name} not advertised; have {names:?}");
        })
}

async fn connect(url: &str, saved: &BTreeMap<String, Vec<CachedTool>>) -> McpTools {
    connect_as(url, saved, true).await
}

async fn connect_as(
    url: &str,
    saved: &BTreeMap<String, Vec<CachedTool>>,
    sandbox: bool,
) -> McpTools {
    let egress = McpEgress::new(EgressPolicy::new(true, &[])).unwrap();
    let connector = McpConnector {
        name: "tools".into(),
        url: url.to_string(),
        sandbox,
        managed: false,
    };
    let (live, _catalog, warnings) = mcp::connect_granted(
        &[connector],
        &BTreeMap::new(),
        &egress,
        "",
        mcp::host(),
        Duration::from_secs(30),
        saved,
    )
    .await;
    assert!(
        warnings.is_empty(),
        "connect_granted warnings: {warnings:?}"
    );
    live
}

#[tokio::test(flavor = "multi_thread")]
async fn saved_menu_honors_ends_turn_from_the_live_call() {
    if skip() {
        eprintln!("skip: bun not on PATH (or BEYOND_AI_SKIP_AGENT_TOOLS_MCP=1)");
        return;
    }
    let guest = TurnTools::new();
    let saved = guest.manifest();
    let finish_menu = &saved["tools"]
        .iter()
        .find(|t| t.remote_name == "finish")
        .unwrap();
    assert!(
        finish_menu.description.contains(ENDS_TURN_SENTENCE),
        "manifest description: {}",
        finish_menu.description
    );
    assert!(
        saved["tools"]
            .iter()
            .any(|t| t.remote_name == "quiet" && t.available_in_subagents == Some(false))
    );

    let mcp = spawn_tools_mcp(&guest.tools_dir());
    let live = connect(&mcp.url, &saved).await;

    let finish = pick(&live, "mcp__tools__finish");
    assert!(finish.description().contains(ENDS_TURN_SENTENCE));
    let done = finish.run(json!({})).await.expect("finish");
    assert!(
        done.terminate,
        "a sandbox connector's endsTurn meta must end the run: {done:?}"
    );
    assert!(done.text.contains("done"), "{done:?}");

    let stay = pick(&live, "mcp__tools__stay")
        .run(json!({}))
        .await
        .expect("stay");
    assert!(
        !stay.terminate,
        "a tool that did not set endsTurn must not end the run: {stay:?}"
    );
    assert!(stay.text.contains("stayed"), "{stay:?}");

    let decide = pick(&live, "mcp__tools__decide");
    assert!(
        !decide.description().contains(ENDS_TURN_SENTENCE),
        "a function endsTurn is decided after execute, not written into the description"
    );
    let stopping = decide
        .run(json!({ "stop": true }))
        .await
        .expect("decide stop");
    assert!(stopping.terminate, "{stopping:?}");
    let going = decide
        .run(json!({ "stop": false }))
        .await
        .expect("decide go");
    assert!(!going.terminate, "{going:?}");

    let boom = pick(&live, "mcp__tools__boom").run(json!({})).await;
    assert!(
        boom.is_err(),
        "a failed endsTurn call is an error, not a stop: {boom:?}"
    );

    assert!(!pick(&live, "mcp__tools__quiet").available_in_subagents());

    // No saved menu: the connector dials live, and is still the sandbox's own server.
    let live = connect(&mcp.url, &BTreeMap::new()).await;
    let finish = pick(&live, "mcp__tools__finish")
        .run(json!({}))
        .await
        .expect("finish");
    assert!(
        finish.terminate,
        "a sandbox connector ends the run without a saved menu too: {finish:?}"
    );
    assert!(!pick(&live, "mcp__tools__quiet").available_in_subagents());
}

/// The same server reached as an ordinary connector: its `_meta` is only metadata.
#[tokio::test(flavor = "multi_thread")]
async fn a_connector_that_is_not_a_sandbox_one_ignores_the_tool_flags() {
    if skip() {
        eprintln!("skip: bun not on PATH (or BEYOND_AI_SKIP_AGENT_TOOLS_MCP=1)");
        return;
    }
    let guest = TurnTools::new();
    let mcp = spawn_tools_mcp(&guest.tools_dir());
    let plain = connect_as(&mcp.url, &BTreeMap::new(), false).await;
    let finish = pick(&plain, "mcp__tools__finish")
        .run(json!({}))
        .await
        .expect("finish");
    assert!(
        !finish.terminate,
        "only a sandbox connector's endsTurn ends the run: {finish:?}"
    );
    assert!(
        pick(&plain, "mcp__tools__quiet").available_in_subagents(),
        "availableInSubagents is honored only for a sandbox connector"
    );
}

/// The production bundle shape: tools import `eve/tools` and `zod` by bare name, resolved from the
/// bundle's own `node_modules` (a copy of `eve-shim` and of this package's `package.json` + `src`).
/// Only the compiled binary can fail this: `bun src/cli.ts` resolves packages on its own, while a
/// `bun build --compile` executable needs `--compile-autoload-package-json` to read the `exports` maps.
#[test]
fn compiled_binary_resolves_eve_tools_from_the_bundles_node_modules() {
    let Some(tools_mcp) = compiled_bin() else {
        eprintln!("skip: set TOOLS_MCP_BIN to the compiled tools-mcp");
        return;
    };
    let pkg = package_root();
    let root = tempfile::tempdir().unwrap();
    let nm = root.path().join("node_modules");
    let copy = |from: &Path, to: &Path| {
        std::fs::create_dir_all(to.parent().unwrap()).unwrap();
        let ok = Command::new("cp")
            .arg("-a")
            .arg(from)
            .arg(to)
            .status()
            .unwrap()
            .success();
        assert!(ok, "cp {} {}", from.display(), to.display());
    };
    copy(&pkg.join("eve-shim"), &nm.join("eve"));
    copy(
        &pkg.join("package.json"),
        &nm.join("@beyond.dev/tools/package.json"),
    );
    copy(&pkg.join("src"), &nm.join("@beyond.dev/tools/src"));
    copy(&package_root().join("../node_modules/zod"), &nm.join("zod"));
    copy(
        &package_root().join("../node_modules/@standard-schema"),
        &nm.join("@standard-schema"),
    );
    std::fs::write(root.path().join("package.json"), r#"{"type":"module"}"#).unwrap();
    let tools = root.path().join("tools");
    std::fs::create_dir_all(&tools).unwrap();
    std::fs::write(
        tools.join("ping.ts"),
        r#"import { defineTool } from "eve/tools";
import { z } from "zod";
export default defineTool({
  description: "Ping.",
  inputSchema: z.object({}),
  async execute() { return { pong: true }; },
});
"#,
    )
    .unwrap();
    let out = Command::new(tools_mcp)
        .args(["manifest", "--dir"])
        .arg(&tools)
        .args(["--name", "tools", "-o"])
        .arg(root.path().join(".mcp-tools.json"))
        .current_dir(root.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "the compiled binary could not load a bundle's tools: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[cfg(target_os = "linux")]
mod service {
    use super::*;

    use beyond_test_support::bundle::{BundleServer, file, sha256_hex, tar};
    use common::grant::Bundle;
    use common::service::{Options, Service};
    use common::{
        TestWs, spawn_model_server, turn_text, turn_tool_use, ws_connect_with_headers,
        ws_read_until_response, ws_send,
    };

    use serde_json::{Value, json};

    use crate::common;

    async fn open(svc: &Service, server: &BundleServer, bytes: &[u8], mcp_url: &str) -> TestWs {
        open_as(svc, server, bytes, mcp_url, true).await
    }

    async fn open_as(
        svc: &Service,
        server: &BundleServer,
        bytes: &[u8],
        mcp_url: &str,
        sandbox: bool,
    ) -> TestWs {
        let token = svc.token_with("t1", "s1.ends", |c| {
            c.mcp = vec![("tools".into(), mcp_url.to_string())];
            if sandbox {
                c.mcp_sandbox = vec!["tools".into()];
            }
            c.bundle = Some(Bundle {
                url: server.url.clone(),
                sha256: sha256_hex(bytes),
            });
        });
        ws_connect_with_headers(svc.port, Some("s1.ends"), &svc.header(&token))
            .await
            .unwrap()
    }

    async fn service(base: &str) -> (Service, tempfile::TempDir) {
        let cache = tempfile::tempdir().unwrap();
        let svc = Service::start_with(
            base,
            &["s1"],
            Options {
                extra_args: vec!["--mcp-allow-private".into()],
                env: vec![("TMPDIR".into(), cache.path().to_string_lossy().into_owned())],
                ..Default::default()
            },
        )
        .await;
        (svc, cache)
    }

    async fn prompt(svc: &Service, ws: &mut TestWs, id: &str, message: &str) -> Vec<Value> {
        ws_send(ws, json!({"type": "prompt", "id": id, "message": message})).await;
        let frames = ws_read_until_response(ws, "prompt").await;
        assert_eq!(
            frames.last().unwrap()["success"],
            true,
            "{frames:#?}\n--- replica stderr ---\n{}",
            std::fs::read_to_string(&svc.stderr_log).unwrap_or_default()
        );
        frames
    }

    /// Saved menu, one scripted model turn that calls `finish`. A second model turn is not
    /// scripted: the run stays successful only when `endsTurn` stops it after the tool.
    #[tokio::test(flavor = "multi_thread")]
    async fn saved_ends_turn_stops_the_agent_run() {
        if skip() {
            eprintln!("skip: bun not on PATH (or BEYOND_AI_SKIP_AGENT_TOOLS_MCP=1)");
            return;
        }
        let guest = TurnTools::new();
        let _saved = guest.manifest();
        let manifest = std::fs::read_to_string(guest.root.join(".mcp-tools.json")).unwrap();
        let bytes = tar(&[
            file("AGENTS.md", "be brief\n"),
            file(".mcp-tools.json", &manifest),
        ]);
        let bundle = BundleServer::start(bytes.clone());
        let mcp = spawn_tools_mcp(&guest.tools_dir());
        let (base, requests) =
            spawn_model_server(vec![turn_tool_use("t1", "mcp__tools__finish", "{}")]);
        let (svc, _cache) = service(&base).await;
        let mut ws = open(&svc, &bundle, &bytes, &mcp.url).await;
        let frames = prompt(&svc, &mut ws, "p1", "finish").await;
        let blob = serde_json::to_string(&frames).unwrap();
        assert!(
            blob.contains("done"),
            "the agent never recorded the tool result: {blob}"
        );
        assert_eq!(
            requests.lock().unwrap().len(),
            1,
            "endsTurn must stop the run before another model turn: {:?}",
            requests.lock().unwrap()
        );
    }

    /// No `.mcp-tools.json`: the connector dials at open. It is still a sandbox connector, so the
    /// run ends after the tool with no second model turn scripted.
    #[tokio::test(flavor = "multi_thread")]
    async fn ends_turn_without_a_saved_menu_stops_the_agent_run() {
        if skip() {
            eprintln!("skip: bun not on PATH (or BEYOND_AI_SKIP_AGENT_TOOLS_MCP=1)");
            return;
        }
        let guest = TurnTools::new();
        let bytes = tar(&[file("AGENTS.md", "be brief\n")]);
        let bundle = BundleServer::start(bytes.clone());
        let mcp = spawn_tools_mcp(&guest.tools_dir());
        let (base, requests) =
            spawn_model_server(vec![turn_tool_use("t1", "mcp__tools__finish", "{}")]);
        let (svc, _cache) = service(&base).await;
        let mut ws = open(&svc, &bundle, &bytes, &mcp.url).await;
        let frames = prompt(&svc, &mut ws, "p1", "finish").await;
        assert!(serde_json::to_string(&frames).unwrap().contains("done"));
        assert_eq!(requests.lock().unwrap().len(), 1);
    }

    /// An apps-capable session dials its own (apps) connection with no saved menu, so this is the
    /// gate that must not depend on one.
    #[tokio::test(flavor = "multi_thread")]
    async fn ends_turn_in_an_apps_session_stops_the_agent_run() {
        if skip() {
            eprintln!("skip: bun not on PATH (or BEYOND_AI_SKIP_AGENT_TOOLS_MCP=1)");
            return;
        }
        let guest = TurnTools::new();
        let _saved = guest.manifest();
        let manifest = std::fs::read_to_string(guest.root.join(".mcp-tools.json")).unwrap();
        let bytes = tar(&[
            file("AGENTS.md", "be brief\n"),
            file(".mcp-tools.json", &manifest),
        ]);
        let bundle = BundleServer::start(bytes.clone());
        let mcp = spawn_tools_mcp(&guest.tools_dir());
        let (base, requests) =
            spawn_model_server(vec![turn_tool_use("t1", "mcp__tools__finish", "{}")]);
        let (svc, _cache) = service(&base).await;
        let mut ws = open(&svc, &bundle, &bytes, &mcp.url).await;
        ws_send(&mut ws, json!({"type": "set_mcp_apps", "enabled": true})).await;
        let declared = ws_read_until_response(&mut ws, "set_mcp_apps").await;
        assert_eq!(declared.last().unwrap()["success"], true, "{declared:?}");
        prompt(&svc, &mut ws, "p1", "finish").await;
        assert_eq!(
            requests.lock().unwrap().len(),
            1,
            "endsTurn must stop an apps session's run too: {:?}",
            requests.lock().unwrap()
        );
    }

    /// The same server granted as an ordinary connector (not `mcp_sandbox`): `_meta` is only
    /// metadata, so the second scripted turn is reached.
    #[tokio::test(flavor = "multi_thread")]
    async fn ends_turn_from_a_non_sandbox_connector_reaches_the_model_again() {
        if skip() {
            eprintln!("skip: bun not on PATH (or BEYOND_AI_SKIP_AGENT_TOOLS_MCP=1)");
            return;
        }
        let guest = TurnTools::new();
        let bytes = tar(&[file("AGENTS.md", "be brief\n")]);
        let bundle = BundleServer::start(bytes.clone());
        let mcp = spawn_tools_mcp(&guest.tools_dir());
        let (base, requests) = spawn_model_server(vec![
            turn_tool_use("t1", "mcp__tools__finish", "{}"),
            turn_text("after"),
        ]);
        let (svc, _cache) = service(&base).await;
        let mut ws = open_as(&svc, &bundle, &bytes, &mcp.url, false).await;
        prompt(&svc, &mut ws, "p1", "finish").await;
        let seen = requests.lock().unwrap().clone();
        assert!(
            seen.len() >= 2,
            "the run stopped before the model saw the tool result: {seen:?}"
        );
        assert!(
            seen.last().unwrap().contains("done"),
            "tool result never reached the next model turn: {}",
            &seen.last().unwrap()[..seen.last().unwrap().len().min(800)]
        );
    }
}
