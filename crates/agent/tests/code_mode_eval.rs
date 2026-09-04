//! Code Mode integration smokes — not an eval (no hidden verifier, no $/pass).
//!
//! Harness evals live in [`eval/`](../eval/) (Harbor + FrontierHarness Terminal-Bench
//! subset). These tests only check that `execute` is wired: mocked-model catalog A/B and
//! a scripted `Promise.all` against the MCP fixture.
//!
//! 1. **Catalog tokens** — a fat MCP catalog advertised as flattened `mcp__…` tools vs deferred
//!    behind `execute`. The win is schema bytes on the first model request.
//! 2. **Composition** — a scripted `execute` program `Promise.all`s real MCP tools; the nested
//!    calls must actually run and the results must reach the next model turn.
//!
//! A live Grok 4.6 / OpenRouter smoke is `#[ignore]` and skips when `OPENROUTER_API_KEY` is unset:
//!
//! ```sh
//! AI_PROVIDER=openrouter OPENROUTER_API_KEY=… \
//!   cargo test -p beyond-ai-agent --features code-mode --test code_mode_eval -- --ignored --nocapture
//! ```
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::path::Path;
use std::process::Command;

use common::{run_cmd, spawn_model_server, turn_text, turn_tool_use};
use serde_json::{Value, json};

fn fixture_bin() -> &'static str {
    env!("CARGO_BIN_EXE_mcp_fixture_stdio_server")
}

fn stdio_server_config(name: &str) -> Value {
    json!({
        "name": name,
        "transport": "stdio",
        "command": fixture_bin(),
        "args": [],
        "env": {},
    })
}

fn write_global_settings(home: &Path, servers: &[Value]) {
    let claude_dir = home.join(".claude");
    std::fs::create_dir_all(&claude_dir).unwrap();
    std::fs::write(
        claude_dir.join("settings.json"),
        serde_json::to_string_pretty(&json!({ "mcp_servers": servers })).unwrap(),
    )
    .unwrap();
}

fn run_against(
    home: &Path,
    cwd: &Path,
    task: &str,
    extra_args: &[&str],
    responses: Vec<String>,
) -> (bool, String, String, Vec<String>) {
    let (base, bodies) = spawn_model_server(responses);
    let mut args = vec![
        "run",
        task,
        "--gateway-url",
        base.as_str(),
        "--key",
        "bai_v1.test",
        "--model",
        "claude-test",
        "--max-steps",
        "6",
        "--no-session-persistence",
    ];
    args.extend_from_slice(extra_args);
    let output = run_cmd(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .env("HOME", home)
        .args(&args)
        .current_dir(cwd)
        .output()
        .expect("spawn binary");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let bodies = bodies.lock().unwrap().clone();
    (output.status.success(), stdout, stderr, bodies)
}

fn request_json(raw: &str) -> Value {
    let body = raw
        .split("\r\n\r\n")
        .nth(1)
        .unwrap_or_else(|| panic!("no HTTP body in recorded request: {raw}"));
    serde_json::from_str(body).unwrap_or_else(|e| panic!("request JSON: {e}: {body}"))
}

fn advertised_tool_names(raw: &str) -> Vec<String> {
    request_json(raw)["tools"]
        .as_array()
        .unwrap_or_else(|| panic!("tools array missing: {raw}"))
        .iter()
        .filter_map(|t| t["name"].as_str().map(str::to_string))
        .collect()
}

fn advertised_tools_bytes(raw: &str) -> usize {
    request_json(raw)["tools"].to_string().len()
}

fn tools_named_bytes(raw: &str, keep: impl Fn(&str) -> bool) -> usize {
    request_json(raw)["tools"]
        .as_array()
        .unwrap_or_else(|| panic!("tools array missing: {raw}"))
        .iter()
        .filter(|t| t["name"].as_str().is_some_and(&keep))
        .map(|t| t.to_string().len())
        .sum()
}

/// Four copies of the fixture — 14 tools each — is a small stand-in for a real MCP-heavy guest
/// (github + jira + slack + linear). One server is not enough: `execute`'s catalog description can
/// match a tiny flattened set, which would hide the context win this eval exists to measure.
const FAT_SERVERS: &[&str] = &["github", "jira", "slack", "linear"];

#[test]
fn eval_fat_mcp_catalog_drops_advertised_schema_bytes() {
    let home = tempfile::tempdir().unwrap();
    let cwd = tempfile::tempdir().unwrap();
    let servers: Vec<Value> = FAT_SERVERS
        .iter()
        .copied()
        .map(stdio_server_config)
        .collect();
    write_global_settings(home.path(), &servers);

    let (ok, _stdout, stderr, flat_bodies) = run_against(
        home.path(),
        cwd.path(),
        "just say hi",
        &[],
        vec![turn_text("hi")],
    );
    assert!(ok, "flattened run failed: {stderr}");
    assert!(
        !flat_bodies.is_empty(),
        "flattened run recorded no model request"
    );

    let (ok, _stdout, stderr, code_bodies) = run_against(
        home.path(),
        cwd.path(),
        "just say hi",
        &["--code-mode"],
        vec![turn_text("hi")],
    );
    assert!(ok, "code-mode run failed: {stderr}");
    assert!(
        !code_bodies.is_empty(),
        "code-mode run recorded no model request"
    );

    let flat_names = advertised_tool_names(&flat_bodies[0]);
    let code_names = advertised_tool_names(&code_bodies[0]);
    let flat_bytes = advertised_tools_bytes(&flat_bodies[0]);
    let code_bytes = advertised_tools_bytes(&code_bodies[0]);

    let mcp_flat = flat_names.iter().filter(|n| n.starts_with("mcp__")).count();
    let mcp_code = code_names.iter().filter(|n| n.starts_with("mcp__")).count();
    let mcp_bytes = tools_named_bytes(&flat_bodies[0], |n| n.starts_with("mcp__"));
    let execute_bytes = tools_named_bytes(&code_bodies[0], |n| n == "execute");

    eprintln!("code_mode_eval catalog:");
    eprintln!(
        "  flattened: {mcp_flat} mcp tools, {mcp_bytes} mcp schema bytes, {flat_bytes} total tool-JSON bytes"
    );
    eprintln!(
        "  code-mode: {mcp_code} mcp tools, {execute_bytes} execute-tool bytes, {code_bytes} total tool-JSON bytes"
    );
    eprintln!(
        "  mcp schema reduction: {:.1}%  (total advertised: {:.1}%)",
        (1.0 - (execute_bytes as f64 / mcp_bytes as f64)) * 100.0,
        (1.0 - (code_bytes as f64 / flat_bytes as f64)) * 100.0
    );

    assert!(
        mcp_flat >= FAT_SERVERS.len() * 10,
        "fixture catalog too small to be an eval ({mcp_flat} mcp tools): {flat_names:?}"
    );
    assert_eq!(
        mcp_code, 0,
        "code-mode must not advertise mcp__ tools: {code_names:?}"
    );
    assert!(
        code_names.iter().any(|n| n == "execute"),
        "code-mode must advertise execute: {code_names:?}"
    );
    assert!(
        code_bytes < flat_bytes,
        "code-mode advertised tool JSON ({code_bytes} bytes) must be smaller than flattened \
         ({flat_bytes} bytes)"
    );
    // The Code Mode win is the MCP schemas, not the built-ins (those stay direct). Execute's
    // compact catalog listing must be well under the flattened JSON-Schema dump.
    assert!(
        execute_bytes * 2 < mcp_bytes,
        "execute ({execute_bytes} bytes) should be under half the flattened MCP schemas \
         ({mcp_bytes} bytes)"
    );
}

#[test]
fn eval_execute_composes_real_mcp_calls() {
    let home = tempfile::tempdir().unwrap();
    let cwd = tempfile::tempdir().unwrap();
    write_global_settings(home.path(), &[stdio_server_config("github")]);

    let code = r#"
        const [echoed, sum, pong] = await Promise.all([
          tools.github.echo({text: "CODEMODE-EVAL"}),
          tools.github.add({a: 2, b: 3}),
          tools.github.ping({}),
        ]);
        return {echo: echoed, sum, ping: pong};
    "#;
    let turn1 = turn_tool_use("toolu_1", "execute", &json!({ "code": code }).to_string());
    let (ok, stdout, stderr, bodies) = run_against(
        home.path(),
        cwd.path(),
        "compose the github tools",
        &["--code-mode"],
        vec![turn1, turn_text("done")],
    );
    assert!(ok, "run failed: {stderr}");
    assert!(
        stdout.contains("[tool: execute]"),
        "should show the execute call: {stdout}"
    );
    assert_eq!(bodies.len(), 2, "expected tool round-trip then final turn");
    let result = &bodies[1];
    assert!(
        result.contains("CODEMODE-EVAL"),
        "echo must round-trip through execute into the next model request: {result}"
    );
    assert!(
        result.contains('5') || result.contains("5.0"),
        "add(2,3) must round-trip through execute: {result}"
    );
    assert!(
        result.contains("pong"),
        "ping must round-trip through execute: {result}"
    );
    assert!(
        !result.contains("mcp__github__"),
        "the model should not see flattened MCP names in the tool_result path: {result}"
    );
}

/// Live Grok 4.6 over OpenRouter: the model, not a script, has to choose `execute` and compose.
/// Bills a real request. Skip when the key is unset rather than failing CI.
const LIVE_PROMPT: &str = "You have MCP tools from github, jira, slack, and linear. \
     Compute 2+3 with add, echo the exact string CODEMODE-EVAL, and ping. \
     If you have an `execute` tool, do all three inside one JavaScript program using \
     Promise.all (tools.github.add / tools.github.echo / tools.github.ping) and return \
     JSON {sum, echo, ping}. Otherwise call the MCP tools directly. \
     Reply with only that JSON object.";

fn live_grok(code_mode: bool) -> (bool, String, String) {
    let home = tempfile::tempdir().unwrap();
    let cwd = tempfile::tempdir().unwrap();
    write_global_settings(
        home.path(),
        &[
            stdio_server_config("github"),
            stdio_server_config("jira"),
            stdio_server_config("slack"),
            stdio_server_config("linear"),
        ],
    );
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_beyond-ai-agent"));
    cmd.env("HOME", home.path())
        .env("AI_PROVIDER", "openrouter")
        .env("AI_DIRECT", "1")
        .args([
            "run",
            LIVE_PROMPT,
            "--model",
            "x-ai/grok-4.6",
            "--reasoning-effort",
            "low",
            "--max-steps",
            "12",
            "--no-session-persistence",
        ])
        .current_dir(cwd.path());
    if code_mode {
        cmd.arg("--code-mode");
    }
    let output = cmd.output().expect("spawn live agent");
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

fn require_openrouter_key() -> bool {
    std::env::var("OPENROUTER_API_KEY")
        .ok()
        .is_some_and(|s| !s.trim().is_empty())
}

#[test]
#[ignore = "live OpenRouter/Grok; needs OPENROUTER_API_KEY"]
fn eval_live_grok_composes_via_execute() {
    if !require_openrouter_key() {
        eprintln!("OPENROUTER_API_KEY unset — skipping live Code Mode eval");
        return;
    }
    let (ok, stdout, stderr) = live_grok(true);
    eprintln!("live grok --code-mode stdout:\n{stdout}");
    if !stderr.is_empty() {
        eprintln!("live grok --code-mode stderr:\n{stderr}");
    }
    assert!(
        ok,
        "live grok --code-mode failed.\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stdout.contains("[tool: execute]"),
        "Grok 4.6 should compose via execute, not flattened mcp__ calls.\nstdout: {stdout}"
    );
    assert!(
        stdout.contains("CODEMODE-EVAL"),
        "echo result missing.\nstdout: {stdout}"
    );
    assert!(
        stdout.contains("pong"),
        "ping result missing.\nstdout: {stdout}"
    );
}

#[test]
#[ignore = "live OpenRouter/Grok; needs OPENROUTER_API_KEY"]
fn eval_live_grok_flattened_mcp_baseline() {
    if !require_openrouter_key() {
        eprintln!("OPENROUTER_API_KEY unset — skipping live flattened baseline");
        return;
    }
    let (ok, stdout, stderr) = live_grok(false);
    eprintln!("live grok flattened stdout:\n{stdout}");
    if !stderr.is_empty() {
        eprintln!("live grok flattened stderr:\n{stderr}");
    }
    assert!(
        ok,
        "live grok flattened failed.\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stdout.contains("CODEMODE-EVAL"),
        "echo result missing.\nstdout: {stdout}"
    );
    assert!(
        stdout.contains("pong"),
        "ping result missing.\nstdout: {stdout}"
    );
    assert!(
        !stdout.contains("[tool: execute]"),
        "flattened baseline must not have an execute tool.\nstdout: {stdout}"
    );
}
