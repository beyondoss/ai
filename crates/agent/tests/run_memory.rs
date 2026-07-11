//! `run` e2e: persistent memory driven through the **real binary** against a real (local) HTTP model
//! server — no mock, no in-process `Agent`. Proves the whole stack end to end: the `memory` tool is
//! registered and advertised, a `create` call lands a file on real disk under the per-project memory
//! dir, a *later* run injects that `MEMORY.md` index back into the system prompt it sends to the model,
//! and the model can recall a stored document. Also covers `--no-memory` and the not-yet-implemented
//! backend seam.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::path::{Path, PathBuf};

use common::{advertised_tools, run_cmd, spawn_model_server, turn_text, turn_tool_use};
use serde_json::Value;

const BIN: &str = env!("CARGO_BIN_EXE_beyond-ai-agent");

fn mem_turn(id: &str, args: Value) -> String {
    turn_tool_use(id, "memory", &args.to_string())
}

/// Drive the real `run --json` binary with `HOME`/cwd pinned so memory persists across invocations that
/// share them. Returns (event lines, raw request bodies, exit-ok).
fn run_in(
    home: &Path,
    work: &Path,
    turns: Vec<String>,
    extra: &[&str],
) -> (Vec<Value>, Vec<String>, bool) {
    let (base, bodies) = spawn_model_server(turns);
    let mut cmd = run_cmd(BIN);
    cmd.env("HOME", home); // last-write wins over run_cmd's ISOLATED_HOME
    cmd.args([
        "run",
        "do the work",
        "--gateway-url",
        &base,
        "--key",
        "bai_v1.test",
        "--model",
        "claude-test",
        "--max-steps",
        "6",
        "--no-session-persistence",
        "--json",
    ])
    .args(extra)
    .current_dir(work);
    let output = cmd.output().expect("spawn binary");
    let events: Vec<Value> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    let bodies = bodies.lock().unwrap().clone();
    (events, bodies, output.status.success())
}

/// The single per-project memory directory the binary created under `HOME/.claude/projects/*/memory/`.
fn memory_dir(home: &Path) -> PathBuf {
    let projects = home.join(".claude/projects");
    let entry = std::fs::read_dir(&projects)
        .unwrap_or_else(|e| panic!("no projects dir at {}: {e}", projects.display()))
        .next()
        .expect("exactly one project memory dir")
        .unwrap();
    entry.path().join("memory")
}

fn tool_ends<'a>(events: &'a [Value], name: &str) -> Vec<&'a Value> {
    events
        .iter()
        .filter(|e| e["kind"] == "tool_end" && e["name"] == name)
        .collect()
}

#[test]
fn persists_to_disk_then_injects_the_index_and_recalls_it() {
    let home = tempfile::tempdir().unwrap();
    let work = tempfile::tempdir().unwrap();

    // --- Run A: the model writes two memories to disk. ------------------------------------------------
    let (events, _bodies, ok) = run_in(
        home.path(),
        work.path(),
        vec![
            mem_turn(
                "toolu_1",
                serde_json::json!({
                    "command": "create",
                    "path": "/memories/notes.md",
                    "file_text": "build: mise run test:local\n"
                }),
            ),
            mem_turn(
                "toolu_2",
                serde_json::json!({
                    "command": "create",
                    "path": "/memories/MEMORY.md",
                    "file_text": "- [notes](notes.md) — how to build & test\n"
                }),
            ),
            turn_text("saved to memory"),
        ],
        &[],
    );
    assert!(ok, "run A should exit 0");
    let creates = tool_ends(&events, "memory");
    assert_eq!(creates.len(), 2, "two memory writes");
    assert_eq!(creates[0]["is_error"], false);
    assert!(
        creates[0]["result"]
            .as_str()
            .unwrap()
            .contains("Created /memories/notes.md")
    );

    // The file is really on disk, with the content the model wrote.
    let dir = memory_dir(home.path());
    assert_eq!(
        std::fs::read_to_string(dir.join("notes.md")).unwrap(),
        "build: mise run test:local\n"
    );
    assert!(dir.join("MEMORY.md").exists());

    // --- Run B: a fresh run in the same project injects the MEMORY.md index into the system prompt. ---
    let (_events, bodies, ok) = run_in(
        home.path(),
        work.path(),
        vec![turn_text("nothing to do")],
        &[],
    );
    assert!(ok, "run B should exit 0");
    let req = &bodies[0];
    assert!(
        advertised_tools(req).iter().any(|t| t == "memory"),
        "the memory tool must be advertised"
    );
    assert!(
        req.contains("## Memory"),
        "the memory guidance must be injected"
    );
    assert!(
        req.contains("[notes](notes.md)"),
        "the current MEMORY.md index must be injected into the system prompt"
    );

    // --- Run C: the model recalls a stored document via `view`. --------------------------------------
    let (events, _bodies, ok) = run_in(
        home.path(),
        work.path(),
        vec![
            mem_turn(
                "toolu_1",
                serde_json::json!({ "command": "view", "path": "/memories/notes.md" }),
            ),
            turn_text("recalled it"),
        ],
        &[],
    );
    assert!(ok, "run C should exit 0");
    let views = tool_ends(&events, "memory");
    assert_eq!(views[0]["is_error"], false);
    assert!(
        views[0]["result"]
            .as_str()
            .unwrap()
            .contains("mise run test:local"),
        "view must return the stored document: {:?}",
        views[0]["result"]
    );
}

#[test]
fn no_memory_omits_the_tool_and_the_section() {
    let home = tempfile::tempdir().unwrap();
    let work = tempfile::tempdir().unwrap();
    let (_events, bodies, ok) = run_in(
        home.path(),
        work.path(),
        vec![turn_text("nothing to do")],
        &["--no-memory"],
    );
    assert!(ok);
    let req = &bodies[0];
    assert!(
        !advertised_tools(req).iter().any(|t| t == "memory"),
        "memory must not be advertised under --no-memory"
    );
    assert!(
        !req.contains("## Memory"),
        "no memory guidance without the tool"
    );
    // And nothing was written to disk.
    assert!(!home.path().join(".claude/projects").exists());
}

#[test]
fn an_unsupported_backend_scheme_fails_fast() {
    let home = tempfile::tempdir().unwrap();
    let work = tempfile::tempdir().unwrap();
    // A redis backend is recognized but not implemented — the run must refuse before any model call.
    let (base, _bodies) = spawn_model_server(vec![turn_text("unused")]);
    let mut cmd = run_cmd(BIN);
    cmd.env("HOME", home.path());
    cmd.args([
        "run",
        "do the work",
        "--gateway-url",
        &base,
        "--key",
        "bai_v1.test",
        "--model",
        "claude-test",
        "--no-session-persistence",
        "--memory",
        "redis://localhost:6379",
    ])
    .current_dir(work.path());
    let output = cmd.output().expect("spawn binary");
    assert!(
        !output.status.success(),
        "an unsupported backend must fail the run"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("redis://") && stderr.contains("not yet supported"),
        "stderr should explain the unsupported backend: {stderr}"
    );
}
