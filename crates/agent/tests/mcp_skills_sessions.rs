//! SEP-2640 skill state belongs to a session, and to that session's subagents.
//!
//! - A `serve` daemon's sessions share MCP connections, so what one session loaded must not stand for
//!   another: session B cannot read a file of a skill only session A loaded.
//! - A subagent of a session is told about the session's MCP skills (`<available_skills
//!   origin="mcp">` in its own system prompt), and is under the same code-execution gate.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::{Command, Stdio};

use common::skills_env::{BIN, Env, bash, last_message_text, read_skill, system_text};
use common::{
    ChildGuard, SpawnGuarded, spawn_listening, spawn_model_server, spawn_model_server_routed,
    turn_text, turn_tool_use, ws_connect, ws_read_until_response, ws_send,
};
use serde_json::json;

const GIT: &str = "skill://git-workflow/SKILL.md";
const GUIDE: &str = "skill://git-workflow/references/GUIDE.md";

fn daemon(env: &Env, base: &str) -> (ChildGuard, u16) {
    let sessions = env.dir.path().join("sessions");
    spawn_listening(
        Command::new(BIN)
            .args([
                "serve",
                "--gateway-url",
                base,
                "--key",
                "bai_v1.test",
                "--model",
                "claude-test",
                "--session-dir",
                sessions.to_str().unwrap(),
                // Approvals are not what this proves; the loaded-skill state is.
                "--approve-mcp-skills",
            ])
            .env("HOME", &env.home)
            .env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0")
            .current_dir(&env.cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::null()),
    )
}

#[tokio::test(flavor = "multi_thread")]
async fn a_daemon_session_cannot_read_a_skill_another_session_loaded() {
    let env = Env::new(json!({}));
    // Keyed on what each conversation holds, so the two sessions (and any title request) can't take
    // each other's answers. Earlier routes win.
    let (base, bodies) = spawn_model_server_routed(
        vec![
            ("GUIDE-BODY".into(), turn_text("alpha read the guide")),
            ("alpha-reads-guide".into(), read_skill("a2", GUIDE)),
            (
                "load the skill first".into(),
                turn_text("bravo was refused"),
            ),
            ("bravo-reads-guide".into(), read_skill("b1", GUIDE)),
            ("GIT-WORKFLOW-BODY-v1".into(), turn_text("alpha loaded it")),
            ("alpha-loads".into(), read_skill("a1", GIT)),
        ],
        turn_text("fallback"),
    );
    let (_daemon, port) = daemon(&env, &base);
    let mut a = ws_connect(port, Some("skills-alpha")).await;
    let mut b = ws_connect(port, Some("skills-bravo")).await;

    ws_send(
        &mut a,
        json!({ "type": "prompt", "id": "a1", "message": "alpha-loads" }),
    )
    .await;
    ws_read_until_response(&mut a, "prompt").await;

    // Same process, same connection to `docs` — but a different session.
    ws_send(
        &mut b,
        json!({ "type": "prompt", "id": "b1", "message": "bravo-reads-guide" }),
    )
    .await;
    ws_read_until_response(&mut b, "prompt").await;
    assert!(
        env.log_of(&format!("resources/read {GUIDE}")).is_empty(),
        "session B's read must not reach the server: {:?}",
        env.log()
    );

    // Session A, which did load the skill, can.
    ws_send(
        &mut a,
        json!({ "type": "prompt", "id": "a2", "message": "alpha-reads-guide" }),
    )
    .await;
    ws_read_until_response(&mut a, "prompt").await;
    assert_eq!(env.log_of(&format!("resources/read {GUIDE}")).len(), 1);

    let bodies = bodies.lock().unwrap().clone();
    let refused = bodies
        .iter()
        .find(|b| b.contains("bravo-reads-guide") && b.contains("load the skill first"))
        .map(|b| last_message_text(b))
        .unwrap_or_else(|| panic!("no refusal reached the model: {bodies:#?}"));
    assert!(refused.contains("loaded in this session"), "{refused}");
    let read = bodies
        .iter()
        .find(|b| b.contains("alpha-reads-guide") && b.contains("GUIDE-BODY"))
        .map(|b| last_message_text(b))
        .unwrap();
    assert!(read.contains("GUIDE-BODY: squash before merge."), "{read}");
}

#[test]
fn a_subagent_gets_the_mcp_skills_listing_and_the_code_execution_gate() {
    let env = Env::new(json!({}));
    let agents = env.home.join(".claude/agents");
    std::fs::create_dir_all(&agents).unwrap();
    std::fs::write(
        agents.join("helper.md"),
        "---\nname: helper\ndescription: helps\n---\nYou are HELPER-MARKER.\n",
    )
    .unwrap();
    // The user loads a skill (their consent), the parent delegates, the child tries to run code.
    let (base, bodies) = spawn_model_server(vec![
        turn_tool_use(
            "d1",
            "subagent",
            &json!({ "agent": "helper", "task": "commit the change" }).to_string(),
        ),
        bash("c1", "echo CHILD-RAN-$((40+2))"),
        turn_text("child done"),
        turn_text("parent done"),
    ]);
    let output = common::run_cmd(BIN)
        .env("HOME", &env.home)
        .env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0")
        .args([
            "run",
            "/skill:docs:git-workflow delegate it",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--max-steps",
            "8",
            "--no-session-persistence",
        ])
        .current_dir(&env.cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn_guarded()
        .wait_with_output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let bodies = bodies.lock().unwrap().clone();
    let child = bodies
        .iter()
        .find(|b| system_text(b).contains("HELPER-MARKER"))
        .unwrap_or_else(|| panic!("the child never asked the model: {bodies:#?}"));
    let child_system = system_text(child);
    assert!(
        child_system.contains("<available_skills origin=\"mcp\">")
            && child_system.contains("<name>docs:git-workflow</name>"),
        "the child is told about its session's MCP skills: {child_system}"
    );
    // The parent session is acting on an MCP skill, so the child's `bash` is gated too — and in
    // `run`, with nobody to ask, denied.
    let child_bash = bodies
        .iter()
        .filter(|b| system_text(b).contains("HELPER-MARKER"))
        .map(|b| last_message_text(b))
        .find(|t| t.contains("'bash' was denied"))
        .unwrap_or_else(|| panic!("the child's bash was not gated: {bodies:#?}"));
    assert!(!child_bash.contains("CHILD-RAN-42"), "{child_bash}");
}
