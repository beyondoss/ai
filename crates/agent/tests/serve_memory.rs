//! `serve` e2e: persistent memory driven through the **real `serve` binary** over its stdio protocol —
//! the same persist → restart-inject → recall loop as `run_memory`, but on the daemon path, so memory is
//! covered on both surfaces (every other tool has a `run_*` and a `serve_*` e2e).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{ChildStdin, Command, Stdio};
use std::sync::{Arc, Mutex};

use common::{
    ChildGuard, SpawnGuarded, read_until_response, spawn_model_server, turn_text, turn_tool_use,
};
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_beyond-ai-agent");

fn mem_turn(id: &str, args: Value) -> String {
    turn_tool_use(id, "memory", &args.to_string())
}

fn send(stdin: &mut ChildStdin, cmd: Value) {
    writeln!(stdin, "{cmd}").unwrap();
    stdin.flush().unwrap();
}

fn kill(mut child: ChildGuard) {
    let _ = child.kill();
    let _ = child.wait();
}

/// A `serve` process with `HOME` and cwd pinned so memory persists to a controlled per-project dir, and
/// optional extra flags (e.g. `--no-memory`).
fn serve_mem_cmd(
    base: &str,
    session_file: &str,
    home: &Path,
    work: &Path,
    extra: &[&str],
) -> Command {
    let mut c = Command::new(BIN);
    c.args([
        "serve",
        "--gateway-url",
        base,
        "--key",
        "bai_v1.test",
        "--model",
        "claude-test",
        "--session-file",
        session_file,
    ])
    .args(extra)
    .env("HOME", home)
    .current_dir(work)
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::null());
    c
}

fn memory_dir(home: &Path) -> PathBuf {
    let projects = home.join(".claude/projects");
    let entry = std::fs::read_dir(&projects)
        .unwrap_or_else(|e| panic!("no projects dir at {}: {e}", projects.display()))
        .next()
        .expect("exactly one project memory dir")
        .unwrap();
    entry.path().join("memory")
}

/// Run one `prompt` against a fresh serve process (scripted upstream), returning the recorded request
/// bodies plus the terminal frames.
fn one_prompt(
    base: &str,
    bodies: &Arc<Mutex<Vec<String>>>,
    session_file: &str,
    home: &Path,
    work: &Path,
    extra: &[&str],
    message: &str,
) -> (Vec<String>, Vec<Value>) {
    let mut child = serve_mem_cmd(base, session_file, home, work, extra).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    send(&mut stdin, json!({ "type": "prompt", "message": message }));
    let frames = read_until_response(&mut stdout, "prompt");
    let recorded = bodies.lock().unwrap().clone();
    kill(child);
    (recorded, frames)
}

fn tool_result_texts(frames: &[Value], name: &str) -> Vec<String> {
    // serve wraps each agent event as {"type":"event","event":{"kind":"tool_end", ...}}.
    frames
        .iter()
        .filter(|f| {
            f["type"] == "event" && f["event"]["kind"] == "tool_end" && f["event"]["name"] == name
        })
        .filter_map(|f| f["event"]["result"].as_str().map(str::to_string))
        .collect()
}

#[test]
fn persists_across_a_restart_then_injects_the_index_and_recalls() {
    let home = tempfile::tempdir().unwrap();
    let work = tempfile::tempdir().unwrap();
    let session_a = work.path().join("a.jsonl").to_string_lossy().into_owned();

    // --- Process A: write two memories. --------------------------------------------------------------
    let (base_a, _b) = spawn_model_server(vec![
        mem_turn(
            "toolu_1",
            json!({ "command": "create", "path": "/memories/notes.md", "file_text": "deploy: mise run deploy:compute:local\n" }),
        ),
        mem_turn(
            "toolu_2",
            json!({ "command": "create", "path": "/memories/MEMORY.md", "file_text": "- [notes](notes.md) — deploy command\n" }),
        ),
        turn_text("saved"),
    ]);
    let mut child =
        serve_mem_cmd(&base_a, &session_a, home.path(), work.path(), &[]).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    send(
        &mut stdin,
        json!({ "type": "prompt", "message": "remember the deploy command" }),
    );
    assert_eq!(
        read_until_response(&mut stdout, "prompt").last().unwrap()["success"],
        true
    );
    kill(child);

    // Really on disk.
    let dir = memory_dir(home.path());
    assert_eq!(
        std::fs::read_to_string(dir.join("notes.md")).unwrap(),
        "deploy: mise run deploy:compute:local\n"
    );

    // --- Process B: a fresh serve, same HOME+cwd, injects the index and recalls the doc. -------------
    let session_b = work.path().join("b.jsonl").to_string_lossy().into_owned();
    let (base_b, bodies_b) = spawn_model_server(vec![
        mem_turn(
            "toolu_1",
            json!({ "command": "view", "path": "/memories/notes.md" }),
        ),
        turn_text("recalled"),
    ]);
    let (recorded, frames) = one_prompt(
        &base_b,
        &bodies_b,
        &session_b,
        home.path(),
        work.path(),
        &[],
        "what's the deploy command?",
    );

    // The MEMORY.md index this process read at startup was injected into the model request.
    let first = &recorded[0];
    assert!(
        first.contains("## Memory"),
        "memory guidance must be injected"
    );
    assert!(
        first.contains("[notes](notes.md)"),
        "the persisted MEMORY.md index must be injected after restart"
    );
    // And the model recalled the stored document.
    let recalled = tool_result_texts(&frames, "memory");
    assert!(
        recalled
            .iter()
            .any(|r| r.contains("mise run deploy:compute:local")),
        "view must return the stored document: {recalled:?}"
    );
}

#[test]
fn a_memory_written_mid_session_refreshes_the_injected_index_on_the_next_rebuild() {
    // The injected index is re-read whenever the system prompt is rebuilt (here, a `set_model`), so a
    // long-lived session sees memories it wrote earlier — not just the (empty) startup snapshot.
    let home = tempfile::tempdir().unwrap();
    let work = tempfile::tempdir().unwrap();
    let session = work.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, bodies) = spawn_model_server(vec![
        // Prompt 1: the model writes the index (startup injection was empty).
        mem_turn(
            "toolu_1",
            json!({ "command": "create", "path": "/memories/MEMORY.md", "file_text": "- [db](db.md) — schema notes\n" }),
        ),
        turn_text("saved"),
        // Prompt 2 (after set_model): plain reply.
        turn_text("ok"),
    ]);

    let mut child = serve_mem_cmd(&base, &session, home.path(), work.path(), &[]).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    send(
        &mut stdin,
        json!({ "type": "prompt", "message": "record the schema notes" }),
    );
    read_until_response(&mut stdout, "prompt");

    // The startup request (prompt 1) had no index yet.
    {
        let recorded = bodies.lock().unwrap();
        assert!(
            !recorded[0].contains("[db](db.md)"),
            "prompt 1's prompt predates the write — no index yet"
        );
    }

    // Rebuild the prompt via set_model, then prompt again.
    send(
        &mut stdin,
        json!({ "type": "set_model", "model": "claude-test-2" }),
    );
    read_until_response(&mut stdout, "set_model");
    send(
        &mut stdin,
        json!({ "type": "prompt", "message": "what were the schema notes?" }),
    );
    read_until_response(&mut stdout, "prompt");

    let recorded = bodies.lock().unwrap();
    let last = recorded.last().unwrap();
    assert!(
        last.contains("[db](db.md)"),
        "after a mid-session write + rebuild, the fresh index must be injected: {last}"
    );
    kill(child);
}

#[test]
fn no_memory_omits_the_tool_and_the_section() {
    let home = tempfile::tempdir().unwrap();
    let work = tempfile::tempdir().unwrap();
    let session = work.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, bodies) = spawn_model_server(vec![turn_text("nothing to do")]);
    let (recorded, _frames) = one_prompt(
        &base,
        &bodies,
        &session,
        home.path(),
        work.path(),
        &["--no-memory"],
        "hello",
    );
    let first = &recorded[0];
    assert!(
        !common::advertised_tools(first)
            .iter()
            .any(|t| t == "memory"),
        "memory must not be advertised under --no-memory"
    );
    assert!(
        !first.contains("## Memory"),
        "no memory section without the tool"
    );
    assert!(!home.path().join(".claude/projects").exists());
}
