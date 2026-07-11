//! `run` e2e: the dual-root memory tool through the **real binary**. One `run` mounts *both* roots —
//! durable `/memories` (persists across runs, under HOME) and per-session `/session` (working memory,
//! scoped to this run) — and the model can write to each independently. Also covers `--no-session-memory`
//! (drops only the `/session` root) and that a persisted run's `/session` store lives beside its session
//! file. The compaction-survival + reminder path is proven on the `serve` surface (serve_session_memory).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::path::Path;

use common::{run_cmd, spawn_model_server, turn_text, turn_tool_use};
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_beyond-ai-agent");

fn mem_turn(id: &str, args: Value) -> String {
    turn_tool_use(id, "memory", &args.to_string())
}

/// Drive the real `run --json` binary, returning (event lines, raw request bodies, exit-ok).
fn run_in(
    home: &Path,
    work: &Path,
    turns: Vec<String>,
    extra: &[&str],
) -> (Vec<Value>, Vec<String>, bool) {
    let (base, bodies) = spawn_model_server(turns);
    let mut cmd = run_cmd(BIN);
    cmd.env("HOME", home);
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

fn tool_ends<'a>(events: &'a [Value], name: &str) -> Vec<&'a Value> {
    events
        .iter()
        .filter(|e| e["kind"] == "tool_end" && e["name"] == name)
        .collect()
}

#[test]
fn one_run_mounts_both_roots_and_writes_to_each() {
    let home = tempfile::tempdir().unwrap();
    let work = tempfile::tempdir().unwrap();
    let session = work.path().join("s.jsonl");
    let session_str = session.to_string_lossy().into_owned();

    let (events, bodies, ok) = run_in(
        home.path(),
        work.path(),
        vec![
            mem_turn(
                "toolu_1",
                json!({ "command": "create", "path": "/memories/durable.md", "file_text": "project fact\n" }),
            ),
            mem_turn(
                "toolu_2",
                json!({ "command": "create", "path": "/session/scratch.md", "file_text": "task scratch\n" }),
            ),
            turn_text("wrote both"),
        ],
        &["--session", &session_str],
    );
    assert!(ok, "run should exit 0");

    // Both roots' guidance was injected — the model was told about durable and working memory distinctly.
    let req = &bodies[0];
    assert!(
        req.contains("Memory (durable, cross-session)"),
        "durable guidance must be injected"
    );
    assert!(
        req.contains("Working memory (this session)"),
        "session working-memory guidance must be injected"
    );

    // Both writes succeeded, each routed to its own store.
    let writes = tool_ends(&events, "memory");
    assert_eq!(writes.len(), 2);
    assert!(
        writes.iter().all(|w| w["is_error"] == false),
        "both writes ok: {writes:?}"
    );

    // Durable lands under HOME's per-project dir; session lands in the `<...>.memory/` sibling.
    let projects = home.path().join(".claude/projects");
    let durable_dir = std::fs::read_dir(&projects)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path()
        .join("memory");
    assert_eq!(
        std::fs::read_to_string(durable_dir.join("durable.md")).unwrap(),
        "project fact\n"
    );
    let session_mem = session.with_extension("memory");
    assert_eq!(
        std::fs::read_to_string(session_mem.join("scratch.md")).unwrap(),
        "task scratch\n"
    );
}

#[test]
fn durable_persists_across_runs_while_session_does_not() {
    let home = tempfile::tempdir().unwrap();
    let work = tempfile::tempdir().unwrap();

    // Run A (ephemeral session): write to both roots.
    let (_e, _b, ok) = run_in(
        home.path(),
        work.path(),
        vec![
            mem_turn(
                "toolu_1",
                json!({ "command": "create", "path": "/memories/k.md", "file_text": "kept\n" }),
            ),
            mem_turn(
                "toolu_2",
                json!({ "command": "create", "path": "/session/e.md", "file_text": "ephemeral\n" }),
            ),
            turn_text("done"),
        ],
        &["--no-session-persistence"],
    );
    assert!(ok);

    // Run B (a fresh ephemeral session): the durable index is injected, but /session starts empty — a
    // fresh, per-run working memory (different session id → different ephemeral dir).
    let (events, bodies, ok) = run_in(
        home.path(),
        work.path(),
        vec![
            mem_turn(
                "toolu_1",
                json!({ "command": "view", "path": "/memories/k.md" }),
            ),
            mem_turn(
                "toolu_2",
                json!({ "command": "view", "path": "/session/e.md" }),
            ),
            turn_text("checked"),
        ],
        &["--no-session-persistence"],
    );
    assert!(ok);
    // The durable memory carried over; its index is back in the prompt.
    assert!(bodies[0].contains("/memories/MEMORY.md") || bodies[0].contains("Memory (durable"));

    let views = tool_ends(&events, "memory");
    assert!(
        views[0]["result"].as_str().unwrap().contains("kept"),
        "durable memory must persist across runs: {:?}",
        views[0]["result"]
    );
    assert_eq!(
        views[1]["is_error"], true,
        "the previous run's /session working memory must NOT be visible to a new session: {:?}",
        views[1]["result"]
    );
}

#[test]
fn no_session_memory_drops_only_the_session_root() {
    let home = tempfile::tempdir().unwrap();
    let work = tempfile::tempdir().unwrap();
    let (_events, bodies, ok) = run_in(
        home.path(),
        work.path(),
        vec![turn_text("nothing to do")],
        &["--no-session-persistence", "--no-session-memory"],
    );
    assert!(ok);
    let req = &bodies[0];
    assert!(
        req.contains("Memory (durable, cross-session)"),
        "durable memory must remain under --no-session-memory"
    );
    assert!(
        !req.contains("Working memory (this session)"),
        "--no-session-memory must drop the /session guidance"
    );
}
