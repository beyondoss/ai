//! A `/skill:` steered into a running prompt must not freeze `serve`.
//!
//! Expanding an MCP-served skill fetches its `SKILL.md` from the server. If that happened inside the
//! busy loop's command arm, the loop would stop polling the run, reading commands (an `abort`, an
//! `approve`) and sending frames for as long as the server took. With a server that answers reads
//! slowly, a command sent right after the steer must still be answered promptly.
//!
//! The expansion is bound to the run it was sent to: if that run ends first (finished, aborted, or
//! cancelled by a session change) the steer is acked as not queued, and nothing of it — no body, no
//! active skill — reaches a later prompt or another session. Steers stay in arrival order: a plain
//! steer sent after a `/skill:` one never overtakes it, and the acks come back in order.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::time::{Duration, Instant};

use common::skills_env::{Env, Serve, bash, is_response, last_message_text};
use common::{spawn_model_server_routed, turn_text};
use serde_json::json;

#[test]
fn a_skill_steered_mid_run_does_not_stall_the_command_loop() {
    let env = Env::new(json!({ "MCP_SKILLS_FIXTURE_SLOW_READ_MS": "6000" }));
    let (base, _bodies) = spawn_model_server_routed(
        vec![
            ("SLEPT".into(), turn_text("done")),
            ("freeze-test".into(), bash("b1", "sleep 3; echo SLEPT")),
        ],
        turn_text("ok"),
    );
    let mut serve = env.serve(&base, &[]);
    serve.send(json!({ "type": "prompt", "message": "freeze-test" }));
    serve.read_until(|f| f["type"] == "event" && f["event"]["kind"] == "tool_start");

    serve.send(json!({ "type": "steer", "id": "s1", "message": "/skill:docs:git-workflow now" }));
    serve.send(json!({ "type": "get_state", "id": "g1" }));
    let asked = Instant::now();
    let mut seen = serve.read_until_within(Duration::from_secs(3), |f| is_response(f, "get_state"));
    assert!(
        asked.elapsed() < Duration::from_secs(3),
        "the loop answered while the skill was still being fetched"
    );

    // The run finishes and the steer is acknowledged — the latter only once the slow read lands.
    let mut prompt_done = seen.iter().any(|f| is_response(f, "prompt"));
    let mut steer_ack = seen
        .iter()
        .any(|f| is_response(f, "steer"))
        .then(|| seen.clone());
    while !(prompt_done && steer_ack.is_some()) {
        seen = serve.read_until(|f| is_response(f, "prompt") || is_response(f, "steer"));
        let last = seen.last().unwrap();
        prompt_done |= is_response(last, "prompt");
        if is_response(last, "steer") {
            steer_ack = Some(seen.clone());
        }
    }
    // The run (a 3s `bash`) ended before the 6s read: the steer is answered as not queued, saying why.
    let ack = steer_ack.unwrap();
    let ack = ack.last().unwrap();
    assert_eq!(ack["success"], false, "{ack}");
    assert!(
        ack["error"]
            .as_str()
            .unwrap()
            .contains("ended before its skill finished loading"),
        "{ack}"
    );
    assert!(
        env.log()
            .contains(&"resources/read skill://git-workflow/SKILL.md".to_string()),
        "{:?}",
        env.log()
    );
    serve.finish();
}

/// The audit's `steer_leak` sequence: a `/skill:` steer whose skill is slow to arrive, then `abort`,
/// then `new_session`, then a prompt in the new session. Nothing of the steer may reach it.
#[test]
fn a_skill_steer_from_an_aborted_run_never_reaches_the_next_session() {
    let env = Env::new(json!({ "MCP_SKILLS_FIXTURE_SLOW_READ_MS": "4000" }));
    let (base, bodies) = spawn_model_server_routed(
        vec![
            ("RAN-42".into(), turn_text("bravo done")),
            ("BRAVO".into(), bash("b2", "echo RAN-$((40+2))")),
            ("leak-test".into(), bash("b1", "sleep 3; echo SLEPT")),
        ],
        turn_text("ok"),
    );
    let mut serve = env.serve(&base, &[]);
    serve.send(json!({ "type": "prompt", "message": "leak-test" }));
    serve.read_until(|f| f["type"] == "event" && f["event"]["kind"] == "tool_start");
    serve.send(json!({ "type": "steer", "id": "s1", "message": "/skill:docs:git-workflow now" }));
    serve.send(json!({ "type": "abort", "id": "a1" }));
    let frames = serve.read_until(|f| is_response(f, "steer"));
    let ack = frames.last().unwrap();
    assert_eq!(
        ack["success"], false,
        "an aborted run's steer is not queued: {ack}"
    );
    serve.read_until(|f| is_response(f, "prompt"));
    serve.call(json!({ "type": "new_session" }), "new_session");
    // Long enough for the slow read to have landed, had the expansion survived.
    std::thread::sleep(Duration::from_millis(4500));
    // In the new session `bash` runs ungated (no skill is active there) and the model sees no body.
    serve.send(json!({ "type": "prompt", "message": "BRAVO hello" }));
    let frames = serve.read_until(|f| f["type"] == "approval_request" || is_response(f, "prompt"));
    assert!(
        is_response(frames.last().unwrap(), "prompt"),
        "a skill from the aborted run gated the new session: {frames:#?}"
    );
    serve.finish();
    let bodies = bodies.lock().unwrap().clone();
    let bravo: Vec<&String> = bodies.iter().filter(|b| b.contains("BRAVO")).collect();
    assert!(!bravo.is_empty());
    assert!(
        bravo.iter().all(|b| !b.contains("GIT-WORKFLOW-BODY")),
        "the steer's skill body leaked into the new session"
    );
    assert!(
        bravo
            .iter()
            .any(|b| last_message_text(b).contains("RAN-42")),
        "bash ran in the new session"
    );
}

/// The same, without a session change: the next prompt after the abort gets nothing of the steer.
#[test]
fn a_skill_steer_from_an_aborted_run_never_reaches_the_next_prompt() {
    let env = Env::new(json!({ "MCP_SKILLS_FIXTURE_SLOW_READ_MS": "4000" }));
    let (base, bodies) = spawn_model_server_routed(
        vec![
            ("AFTER".into(), turn_text("after done")),
            ("leak-test".into(), bash("b1", "sleep 3; echo SLEPT")),
        ],
        turn_text("ok"),
    );
    let mut serve: Serve = env.serve(&base, &[]);
    serve.send(json!({ "type": "prompt", "message": "leak-test" }));
    serve.read_until(|f| f["type"] == "event" && f["event"]["kind"] == "tool_start");
    serve.send(json!({ "type": "steer", "id": "s1", "message": "/skill:docs:git-workflow now" }));
    serve.send(json!({ "type": "abort", "id": "a1" }));
    serve.read_until(|f| is_response(f, "prompt"));
    std::thread::sleep(Duration::from_millis(4500));
    serve.call(
        json!({ "type": "prompt", "message": "AFTER abort" }),
        "prompt",
    );
    serve.finish();
    let bodies = bodies.lock().unwrap().clone();
    let after = bodies.iter().find(|b| b.contains("AFTER abort")).unwrap();
    assert!(
        !after.contains("GIT-WORKFLOW-BODY"),
        "the steer leaked into the next prompt"
    );
}

/// The audit's `steer_order` sequence: a `/skill:` steer (slow), then a plain steer. The plain one
/// must not overtake it, and the acks come back in order.
#[test]
fn a_plain_steer_does_not_overtake_an_earlier_skill_steer() {
    let env = Env::new(json!({ "MCP_SKILLS_FIXTURE_SLOW_READ_MS": "2000" }));
    let (base, bodies) = spawn_model_server_routed(
        vec![
            ("SLEPT".into(), turn_text("done")),
            ("order-test".into(), bash("b1", "sleep 5; echo SLEPT")),
        ],
        turn_text("ok"),
    );
    let mut serve = env.serve(&base, &[]);
    serve.send(json!({ "type": "prompt", "message": "order-test" }));
    serve.read_until(|f| f["type"] == "event" && f["event"]["kind"] == "tool_start");
    serve.send(json!({ "type": "steer", "id": "s1", "message": "/skill:docs:git-workflow now" }));
    serve.send(json!({ "type": "steer", "id": "s2", "message": "PLAIN-SECOND-STEER" }));
    let mut acks = Vec::new();
    let frames = serve.read_until(|f| {
        if is_response(f, "steer") {
            acks.push(f["id"].as_str().unwrap_or_default().to_string());
        }
        acks.len() == 2
    });
    assert_eq!(acks, ["s1", "s2"], "acks out of order: {frames:#?}");
    serve.read_until(|f| is_response(f, "prompt"));
    serve.finish();
    let bodies = bodies.lock().unwrap().clone();
    let after = bodies.iter().rev().find(|b| b.contains("SLEPT")).unwrap();
    let skill = after
        .find("GIT-WORKFLOW-BODY")
        .expect("the skill steer was delivered");
    let plain = after
        .find("PLAIN-SECOND-STEER")
        .expect("the plain steer was delivered");
    assert!(skill < plain, "the plain steer overtook the skill steer");
}

/// A nested request a server raises while it answers a skill read for a `/skill:` steered into a
/// running prompt belongs to the steering session: its elicitation reaches this session's client,
/// whose command loop is live, and the answer goes back — never to the shared connection's own host.
#[test]
fn a_nested_request_during_a_steered_skill_expansion_reaches_the_session() {
    let env = Env::new(json!({ "MCP_SKILLS_FIXTURE_ELICIT_ON_READ": "1" }));
    let (base, _bodies) = spawn_model_server_routed(
        vec![
            ("SLEPT".into(), turn_text("done")),
            ("nested-test".into(), bash("b1", "sleep 3; echo SLEPT")),
        ],
        turn_text("ok"),
    );
    let mut serve = env.serve(&base, &[]);
    serve.send(json!({ "type": "prompt", "message": "nested-test" }));
    serve.read_until(|f| f["type"] == "event" && f["event"]["kind"] == "tool_start");
    serve.send(json!({ "type": "steer", "id": "s1", "message": "/skill:docs:git-workflow now" }));
    let frames = serve.read_until(|f| {
        f["type"] == "elicitation_request" || is_response(f, "steer") || is_response(f, "prompt")
    });
    let ask = frames.last().unwrap();
    assert_eq!(
        ask["type"], "elicitation_request",
        "the skill read's nested elicitation never reached the session: {frames:#?}"
    );
    serve.send(json!({
        "type": "elicit",
        "id": "e1",
        "request_id": ask["request_id"],
        "action": "accept",
        "content": { "name": "ada" },
    }));
    let frames = serve.read_until(|f| is_response(f, "steer"));
    let ack = frames.last().unwrap();
    assert_eq!(ack["success"], true, "{ack}");
    serve.read_until(|f| is_response(f, "prompt"));
    serve.finish();
    assert!(
        env.log().contains(&"elicit accept".to_string()),
        "{:?}",
        env.log()
    );
}

/// Between runs nothing can answer a nested request: `serve`'s command loop is waiting on the very
/// read that raised it. So one raised during a `/skill:` expansion before a prompt starts is refused
/// at once — not routed to the session (which would stall the prompt until the expansion timed out)
/// and not to the connection's own host — and the skill still loads.
#[test]
fn a_nested_request_during_an_expansion_between_runs_is_refused_at_once() {
    let env = Env::new(json!({ "MCP_SKILLS_FIXTURE_ELICIT_ON_READ": "1" }));
    let (base, bodies) = spawn_model_server_routed(vec![], turn_text("ok"));
    let mut serve = env.serve(&base, &[]);
    let started = Instant::now();
    serve.send(json!({ "type": "prompt", "message": "/skill:docs:git-workflow go" }));
    let frames =
        serve.read_until(|f| f["type"] == "elicitation_request" || is_response(f, "prompt"));
    assert!(
        is_response(frames.last().unwrap(), "prompt"),
        "a nested request between runs was routed to a session that cannot answer it: {frames:#?}"
    );
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "the prompt stalled on the nested request: {:?}",
        started.elapsed()
    );
    serve.finish();
    assert!(
        env.log().contains(&"elicit decline".to_string()),
        "{:?}",
        env.log()
    );
    let bodies = bodies.lock().unwrap().clone();
    assert!(
        bodies.iter().any(|b| b.contains("GIT-WORKFLOW-BODY")),
        "the skill still loaded"
    );
}

/// A plain steer queued behind a `/skill:` whose skill is still loading belongs to the same run. When
/// that run is aborted it is dropped like every steer the abort cleared — acked as not queued — and
/// never reaches the next prompt.
#[test]
fn a_plain_steer_behind_a_skill_steer_dies_with_the_aborted_run() {
    let env = Env::new(json!({ "MCP_SKILLS_FIXTURE_SLOW_READ_MS": "4000" }));
    let (base, bodies) = spawn_model_server_routed(
        vec![
            ("AFTER".into(), turn_text("after done")),
            ("leak-test".into(), bash("b1", "sleep 3; echo SLEPT")),
        ],
        turn_text("ok"),
    );
    let mut serve = env.serve(&base, &[]);
    serve.send(json!({ "type": "prompt", "message": "leak-test" }));
    serve.read_until(|f| f["type"] == "event" && f["event"]["kind"] == "tool_start");
    serve.send(json!({ "type": "steer", "id": "s1", "message": "/skill:docs:git-workflow now" }));
    serve.send(json!({ "type": "steer", "id": "s2", "message": "PLAIN-BEHIND-SKILL" }));
    serve.send(json!({ "type": "abort", "id": "a1" }));
    let mut acks = Vec::new();
    let frames = serve.read_until(|f| {
        if is_response(f, "steer") {
            acks.push(f.clone());
        }
        acks.len() == 2
    });
    assert_eq!(acks[1]["id"], "s2", "{frames:#?}");
    assert_eq!(
        acks[1]["success"], false,
        "a steer from the aborted run is not queued: {}",
        acks[1]
    );
    assert!(
        acks[1]["error"].as_str().unwrap().contains("cancelled"),
        "{}",
        acks[1]
    );
    serve.read_until(|f| is_response(f, "prompt"));
    serve.call(
        json!({ "type": "prompt", "message": "AFTER abort" }),
        "prompt",
    );
    serve.finish();
    let bodies = bodies.lock().unwrap().clone();
    let after = bodies.iter().find(|b| b.contains("AFTER abort")).unwrap();
    assert!(
        !after.contains("PLAIN-BEHIND-SKILL"),
        "the aborted run's steer leaked into the next prompt"
    );
}

/// A `/skill:` expansion that has finished but is still waiting its turn (behind a slower one) when
/// the run is aborted never reached the model, so it must not activate: the session stays ungated,
/// and the next prompt's `bash` runs without a skill approval.
#[test]
fn a_finished_expansion_that_was_never_queued_does_not_activate_its_skill() {
    let env = Env::new(json!({
        "MCP_SKILLS_FIXTURE_SLOW_READ_MS": "4000",
        "MCP_SKILLS_FIXTURE_SLOW_READ_URI": "git-workflow",
    }));
    let (base, bodies) = spawn_model_server_routed(
        vec![
            ("RAN-42".into(), turn_text("bravo done")),
            ("BRAVO".into(), bash("b2", "echo RAN-$((40+2))")),
            ("defer-test".into(), bash("b1", "sleep 3; echo SLEPT")),
        ],
        turn_text("ok"),
    );
    let mut serve = env.serve(&base, &[]);
    serve.send(json!({ "type": "prompt", "message": "defer-test" }));
    serve.read_until(|f| f["type"] == "event" && f["event"]["kind"] == "tool_start");
    serve.send(json!({ "type": "steer", "id": "s1", "message": "/skill:docs:git-workflow now" }));
    serve.send(json!({ "type": "steer", "id": "s2", "message": "/skill:docs:yamlish too" }));
    // Until the fast skill has been fetched: its expansion is done, waiting behind `s1`'s.
    let deadline = Instant::now() + Duration::from_secs(10);
    while !env
        .log()
        .contains(&"resources/read skill://yamlish/SKILL.md".to_string())
    {
        assert!(Instant::now() < deadline, "{:?}", env.log());
        std::thread::sleep(Duration::from_millis(20));
    }
    std::thread::sleep(Duration::from_millis(300));
    serve.send(json!({ "type": "abort", "id": "a1" }));
    let mut acks = Vec::new();
    serve.read_until(|f| {
        if is_response(f, "steer") {
            acks.push(f.clone());
        }
        acks.len() == 2
    });
    assert!(
        acks.iter().all(|a| a["success"] == false),
        "neither steer was queued: {acks:#?}"
    );
    serve.read_until(|f| is_response(f, "prompt"));
    serve.send(json!({ "type": "prompt", "message": "BRAVO hello" }));
    let frames = serve.read_until(|f| f["type"] == "approval_request" || is_response(f, "prompt"));
    assert!(
        is_response(frames.last().unwrap(), "prompt"),
        "a skill whose text never reached the model gated the session: {frames:#?}"
    );
    serve.finish();
    let bodies = bodies.lock().unwrap().clone();
    let bravo: Vec<&String> = bodies.iter().filter(|b| b.contains("BRAVO")).collect();
    assert!(!bravo.is_empty());
    assert!(
        bravo.iter().all(|b| !b.contains("YAMLISH-BODY")),
        "the never-queued skill's body reached the model"
    );
    assert!(
        bravo
            .iter()
            .any(|b| last_message_text(b).contains("RAN-42")),
        "bash ran ungated"
    );
}
