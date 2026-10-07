//! The SEP-2640 acting window survives everything that brings a session back.
//!
//! A session whose transcript holds an MCP skill's `SKILL.md` is acting on that skill — however it
//! got here: resumed by a later `run --session-id`, switched back to in `serve`, forked, or reopened
//! after `serve` restarted. The code-execution gate must be on for it each time; the spec lets the
//! window be longer than the skill's time in context, never shorter.
//!
//! The user's approvals come back with it: persisted with the session, bound to the content they
//! were given for, so a reopened session is not asked again about an unchanged skill — and is about
//! a changed one.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::skills_env::{Env, Serve, bash, is_response, last_message_text, read_skill};
use common::{spawn_model_server_routed, turn_text};
use serde_json::{Value, json};

/// A model that answers a `/skill:` turn with text, runs `bash` on "run-it", and stops once told no.
fn model() -> (String, common::skills_env::Bodies) {
    spawn_model_server_routed(
        vec![
            ("was denied".into(), turn_text("ok, not running it")),
            ("RAN-42".into(), turn_text("ran it")),
            ("run-it".into(), bash("b1", "echo RAN-$((40+2))")),
        ],
        turn_text("loaded"),
    )
}

/// Send `run-it` and expect the code-execution question, naming the skill; deny it.
fn expect_gated(serve: &mut Serve) {
    serve.send(json!({ "type": "prompt", "message": "run-it" }));
    let frames = serve.read_until(|f| f["type"] == "approval_request" || is_response(f, "prompt"));
    let q = frames.last().unwrap();
    assert_eq!(
        q["type"], "approval_request",
        "bash ran ungated in a session acting on an MCP skill: {frames:#?}"
    );
    assert_eq!(q["mcp_skill"]["purpose"], "execute", "{q}");
    assert_eq!(
        q["mcp_skill"]["active_skills"][0]["uri"], "skill://git-workflow/SKILL.md",
        "{q}"
    );
    serve.approve(q, "deny", "once");
    serve.read_until(|f| is_response(f, "prompt"));
}

fn load_by_slash(serve: &mut Serve) {
    let frames = serve.call(
        json!({ "type": "prompt", "message": "/skill:docs:git-workflow load it" }),
        "prompt",
    );
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
}

fn session_id(serve: &mut Serve) -> String {
    let frames = serve.call(json!({ "type": "get_state" }), "get_state");
    frames.last().unwrap()["data"]["session_id"]
        .as_str()
        .unwrap()
        .to_string()
}

#[test]
fn a_resumed_run_is_still_gated() {
    let env = Env::new(json!({}));
    let sessions = env.dir.path().join("sessions");
    let dir = sessions.to_str().unwrap();
    let first = env.run_raw(
        "/skill:docs:git-workflow load it",
        vec![turn_text("loaded")],
        &["--session-id", "skilled", "--session-dir", dir],
    );
    assert!(first.ok, "{}", first.stderr);

    // A later process, the same session: SKILL.md is in its transcript, so `bash` is gated — and
    // `run`, with no one to ask, denies.
    let resumed = env.run_raw(
        "now commit",
        vec![bash("b1", "echo RAN-$((40+2))"), turn_text("done")],
        &["--session-id", "skilled", "--session-dir", dir],
    );
    assert!(resumed.ok, "{}", resumed.stderr);
    let result = last_message_text(&resumed.bodies[1]);
    assert!(
        result.contains("'bash' was denied") && result.contains("docs:git-workflow"),
        "{result}"
    );

    // The control: a session that never loaded a skill runs it.
    let fresh = env.run_raw(
        "now commit",
        vec![bash("b1", "echo RAN-$((40+2))"), turn_text("done")],
        &["--session-id", "plain", "--session-dir", dir],
    );
    assert!(last_message_text(&fresh.bodies[1]).contains("RAN-42"));
}

#[test]
fn switching_back_to_a_session_that_loaded_a_skill_restores_the_gate() {
    let env = Env::new(json!({}));
    let (base, _bodies) = model();
    let mut serve = env.serve_dir(&base, &[]);
    load_by_slash(&mut serve);
    let skilled = session_id(&mut serve);

    let frames = serve.call(json!({ "type": "new_session" }), "new_session");
    assert_eq!(frames.last().unwrap()["success"], true);
    let frames = serve.call(
        json!({ "type": "switch_session", "session_id": skilled }),
        "switch_session",
    );
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    expect_gated(&mut serve);
    serve.finish();
}

#[test]
fn a_fork_of_a_session_that_loaded_a_skill_is_gated() {
    let env = Env::new(json!({}));
    let (base, _bodies) = model();
    let mut serve = env.serve_dir(&base, &[]);
    load_by_slash(&mut serve);
    let before = session_id(&mut serve);
    let frames = serve.call(json!({ "type": "clone" }), "clone");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    assert_ne!(
        session_id(&mut serve),
        before,
        "clone moves to a new session"
    );
    expect_gated(&mut serve);
    serve.finish();

    // `fork` the same way, from a fresh session (the mock keys on the conversation, which after the
    // check above already holds a denial).
    let env = Env::new(json!({}));
    let (base, _bodies) = model();
    let mut serve = env.serve_dir(&base, &[]);
    load_by_slash(&mut serve);
    let before = session_id(&mut serve);
    let frames = serve.call(json!({ "type": "fork" }), "fork");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    assert_ne!(
        session_id(&mut serve),
        before,
        "fork moves to a new session"
    );
    expect_gated(&mut serve);
    serve.finish();
}

#[test]
fn a_restarted_serve_reopening_the_session_is_gated() {
    let env = Env::new(json!({}));
    let (base, _bodies) = model();
    let session_file = env.dir.path().join("restart.jsonl");
    let mut serve = env.serve_on(&base, &session_file, &[]);
    load_by_slash(&mut serve);
    serve.finish();

    let mut serve = env.serve_on(&base, &session_file, &[]);
    let state = serve.call(json!({ "type": "get_state" }), "get_state");
    let count: &Value = &state.last().unwrap()["data"]["message_count"];
    assert!(
        count.as_u64().unwrap() >= 2,
        "the session was reopened: {state:#?}"
    );
    expect_gated(&mut serve);
    serve.finish();
}

/// The model's next activation question, or `None` if the prompt finished without one.
fn activation_question(serve: &mut Serve) -> Option<Value> {
    let frames = serve.read_until(|f| f["type"] == "approval_request" || is_response(f, "prompt"));
    let last = frames.last().unwrap().clone();
    (last["type"] == "approval_request").then_some(last)
}

#[test]
fn a_skill_approval_survives_a_restart_for_unchanged_content_only() {
    let env = Env::new(json!({ "MCP_SKILLS_FIXTURE_VERSION_FILE": "V2" }));
    const GIT: &str = "skill://git-workflow/SKILL.md";
    // Each prompt has the model load the skill once, under its own tool-call id.
    let (base, _bodies) = spawn_model_server_routed(
        vec![
            ("\"tool_use_id\":\"t3\"".into(), turn_text("done three")),
            ("load-three".into(), read_skill("t3", GIT)),
            ("\"tool_use_id\":\"t2\"".into(), turn_text("done two")),
            ("load-two".into(), read_skill("t2", GIT)),
            ("\"tool_use_id\":\"t1\"".into(), turn_text("done one")),
            ("load-one".into(), read_skill("t1", GIT)),
        ],
        turn_text("ok"),
    );
    let session_file = env.dir.path().join("approvals.jsonl");

    // 1. Asked once; approved for the session.
    let mut serve = env.serve_on(&base, &session_file, &[]);
    serve.send(json!({ "type": "prompt", "message": "load-one" }));
    let q = activation_question(&mut serve).expect("the first load is asked about");
    assert_eq!(q["mcp_skill"]["purpose"], "activate", "{q}");
    serve.approve(&q, "allow", "session");
    serve.read_until(|f| is_response(f, "prompt"));
    serve.finish();

    // 2. A new process reopening the session: the same content is not asked about again.
    let mut serve = env.serve_on(&base, &session_file, &[]);
    serve.send(json!({ "type": "prompt", "message": "load-two" }));
    assert_eq!(
        activation_question(&mut serve),
        None,
        "a restored approval for unchanged content must not ask again"
    );

    // 3. The skill changes (an honest new manifest): the old approval does not cover it.
    std::fs::write(env.cwd.join("V2"), "v2").unwrap();
    serve.send(json!({ "type": "prompt", "message": "load-three" }));
    let q = activation_question(&mut serve).expect("a changed skill is asked about again");
    assert_eq!(q["mcp_skill"]["purpose"], "activate", "{q}");
    serve.approve(&q, "deny", "once");
    serve.read_until(|f| is_response(f, "prompt"));
    serve.finish();
}
