//! `serve` e2e: the `subagent` tool works over the serve protocol, and a child's activity surfaces as
//! progress frames on the parent's event stream (coalesced, not one frame per child token).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufReader, Write};

use common::{SpawnGuarded, read_until_response, spawn_model_server, turn_text, turn_tool_use};
use serde_json::{Value, json};

fn write_scout(dir: &std::path::Path) {
    let agents = dir.join(".claude/agents");
    std::fs::create_dir_all(&agents).unwrap();
    std::fs::write(
        agents.join("scout.md"),
        "---\nname: scout\ndescription: recon\ntools: read,grep,find,ls\n---\nYou are SCOUT-MARKER.\n",
    )
    .unwrap();
}

#[test]
fn serve_runs_a_subagent_and_streams_its_progress() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    // A project-local agent (trust-gated — the test passes --trust-project below).
    write_scout(dir.path());

    let (base, bodies) = spawn_model_server(vec![
        turn_tool_use(
            "call-1",
            "subagent",
            &json!({ "agent": "scout", "task": "look around" }).to_string(),
        ),
        turn_text("CHILD-FOUND-IT"),
        turn_text("The scout found it."),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = common::serve_cmd(bin, &base, &session_file)
        .arg("--trust-project")
        .current_dir(dir.path())
        .stderr(std::process::Stdio::piped())
        .spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "explore" })
    )
    .unwrap();
    stdin.flush().unwrap();

    let frames = read_until_response(&mut stdout, "prompt");
    let _ = child.kill();
    let _ = child.wait();

    // The run completed and the child's output reached the parent's final turn.
    let child_req = &bodies.lock().unwrap()[1];
    assert!(
        child_req.contains("SCOUT-MARKER"),
        "the child request must use the def body: {child_req}"
    );

    // Progress frames for the subagent call must have been emitted. `subagent` reports its status via
    // `ToolProgress` (`emit`), which the loop turns into a tool-progress event carrying a snapshot that
    // names the agent.
    let progress_frames: Vec<&Value> = frames
        .iter()
        .filter(|f| {
            let s = f.to_string();
            s.contains("subagent") && (s.contains("progress") || s.contains("scout"))
        })
        .collect();
    assert!(
        !progress_frames.is_empty(),
        "expected at least one subagent progress frame; frames were:\n{}",
        frames
            .iter()
            .map(|f| f.to_string())
            .collect::<Vec<_>>()
            .join("\n")
    );

    // Coalesced, not one-per-token: a single child producing one short message must not flood the stream
    // with dozens of subagent frames.
    assert!(
        progress_frames.len() < 20,
        "subagent progress must be coalesced, got {} frames",
        progress_frames.len()
    );
}

#[test]
fn reload_makes_a_newly_added_agent_delegable_without_a_restart() {
    // Serve starts with NO agent definitions, so no `subagent` tool. A definition is added on disk and
    // `reload` is sent — after which the tool must be registered and the agent delegable, proving reload
    // rebuilds the agent (not just its system prompt).
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    // Model turns, in order: (1) answers the first prompt (no subagent yet); (2) after reload, calls
    // subagent; (3) the child's result; (4) the parent's final answer.
    let (base, bodies) = spawn_model_server(vec![
        turn_text("nothing to delegate yet"),
        turn_tool_use(
            "call-1",
            "subagent",
            &json!({ "agent": "scout", "task": "look" }).to_string(),
        ),
        turn_text("CHILD-RAN-AFTER-RELOAD"),
        turn_text("the scout reported back"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = common::serve_cmd(bin, &base, &session_file)
        .arg("--trust-project")
        .current_dir(dir.path())
        .stderr(std::process::Stdio::piped())
        .spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // First prompt: no agents defined yet.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "anything to delegate?" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let _ = read_until_response(&mut stdout, "prompt");

    // Add an agent definition on disk, then reload.
    write_scout(dir.path());
    writeln!(stdin, "{}", json!({ "type": "reload" })).unwrap();
    stdin.flush().unwrap();
    let _ = read_until_response(&mut stdout, "reload");

    // Second prompt: the model can now call subagent.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "now delegate" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let _ = read_until_response(&mut stdout, "prompt");

    let _ = child.kill();
    let _ = child.wait();

    let bodies = bodies.lock().unwrap();
    // First prompt request: no subagent tool.
    assert!(
        !bodies[0].contains("\"name\":\"subagent\"")
            && !bodies[0].contains("\"name\": \"subagent\""),
        "before reload there are no agents, so no subagent tool: {}",
        bodies[0]
    );
    // Second prompt request (after reload): the subagent tool and the agent are now advertised.
    assert!(
        bodies[1].contains("subagent"),
        "reload must register the subagent tool: {}",
        bodies[1]
    );
    assert!(
        bodies[1].contains("available_agents"),
        "reload must advertise the new agent: {}",
        bodies[1]
    );
    // And the child actually ran, using the newly-added definition's body.
    assert!(
        bodies.iter().any(|b| b.contains("SCOUT-MARKER")),
        "the newly-added agent's child must have run after reload"
    );
}
