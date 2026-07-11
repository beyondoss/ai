//! `run` e2e: subagents share the parent's persistent memory. Driven through the **real binary** against
//! a local scripted model server. Proves the whole sharing contract end to end: a scoped child still
//! gets the `memory` tool, a child's write lands in the *parent's* on-disk store, and a *second* child
//! spawned later sees that write through the shared store's auto-injected index.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::path::{Path, PathBuf};
use std::process::Stdio;

use common::{advertised_tools, run_cmd, spawn_model_server, turn_text, turn_tool_use};
use serde_json::json;

const BIN: &str = env!("CARGO_BIN_EXE_beyond-ai-agent");

fn write_agent(dir: &Path, name: &str, frontmatter: &str, body: &str) {
    let agents = dir.join(".claude/agents");
    std::fs::create_dir_all(&agents).unwrap();
    std::fs::write(
        agents.join(format!("{name}.md")),
        format!("---\nname: {name}\n{frontmatter}---\n{body}\n"),
    )
    .unwrap();
}

fn mem_turn(id: &str, args: serde_json::Value) -> String {
    turn_tool_use(id, "memory", &args.to_string())
}

fn memory_dir(home: &Path) -> PathBuf {
    let projects = home.join(".claude/projects");
    let entry = std::fs::read_dir(&projects)
        .unwrap_or_else(|e| panic!("no projects dir at {}: {e}", projects.display()))
        .next()
        .expect("a per-project memory dir")
        .unwrap();
    entry.path().join("memory")
}

#[test]
fn a_child_writes_shared_memory_a_later_child_reads_it_and_it_lands_in_the_parent_store() {
    let home = tempfile::tempdir().unwrap();
    let work = tempfile::tempdir().unwrap();

    // `scribe` is deliberately scoped to `tools: read` (no `memory` listed) — it must still get the
    // shared memory tool. `recaller` needs no tools at all; it recalls from the injected index.
    write_agent(
        work.path(),
        "scribe",
        "description: records facts\ntools: read\n",
        "You are the scribe.",
    );
    write_agent(
        work.path(),
        "recaller",
        "description: recalls facts\ntools: read\n",
        "You are the recaller.",
    );

    // FIFO request order (single child at a time): parent → scribe×3 → parent → recaller → parent.
    let (base, bodies) = spawn_model_server(vec![
        // 1. Parent delegates the write.
        turn_tool_use(
            "c1",
            "subagent",
            &json!({ "agent": "scribe", "task": "save the deploy command to memory" }).to_string(),
        ),
        // 2-4. Scribe child writes a topic file + a MEMORY.md pointer, then finishes.
        mem_turn(
            "m1",
            json!({ "command": "create", "path": "/memories/deploy.md", "file_text": "deploy: zig build test\n" }),
        ),
        mem_turn(
            "m2",
            json!({ "command": "create", "path": "/memories/MEMORY.md", "file_text": "- [deploy](deploy.md) — deploy command\n" }),
        ),
        turn_text("saved to memory"),
        // 5. Parent delegates the recall.
        turn_tool_use(
            "c2",
            "subagent",
            &json!({ "agent": "recaller", "task": "what is the deploy command?" }).to_string(),
        ),
        // 6. Recaller child answers — no tool call; it recalls from the injected shared index.
        turn_text("The deploy command is zig build test."),
        // 7. Parent finishes.
        turn_text("Done."),
    ]);

    let output = run_cmd(BIN)
        .env("HOME", home.path())
        .args([
            "run",
            "remember then recall the deploy command via subagents",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--trust-project",
            "--no-session-persistence",
        ])
        .current_dir(work.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");
    assert!(
        output.status.success(),
        "run failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let bodies = bodies.lock().unwrap();
    assert_eq!(
        bodies.len(),
        7,
        "parent → scribe×3 → parent → recaller → parent"
    );

    // The scoped `scribe` child (tools: read) still had the shared `memory` tool advertised.
    let scribe_req = &bodies[1];
    assert!(
        advertised_tools(scribe_req).iter().any(|t| t == "memory"),
        "a scoped subagent must still carry the shared memory tool: {scribe_req}"
    );

    // Shared WRITE: the child's create landed in the *parent's* project store on disk.
    let dir = memory_dir(home.path());
    assert_eq!(
        std::fs::read_to_string(dir.join("deploy.md")).unwrap(),
        "deploy: zig build test\n",
        "the subagent's write must persist to the parent's shared store"
    );

    // Shared READ: the *later* `recaller` child (request #6) sees the earlier child's write through the
    // shared store's auto-injected index — the whole point of sharing.
    let recaller_req = &bodies[5];
    assert!(
        recaller_req.contains("## Memory"),
        "the recaller child must carry the memory section: {recaller_req}"
    );
    assert!(
        recaller_req.contains("[deploy](deploy.md)"),
        "a later child must see an earlier child's write via the shared, injected index: {recaller_req}"
    );
}
