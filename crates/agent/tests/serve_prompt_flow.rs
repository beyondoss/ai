//! `serve` e2e: The live `prompt`/`steer`/`follow_up` request path: queuing, busy semantics, refusal, abort.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufRead, BufReader, Write};

use common::{
    read_until_response, serve_cmd, spawn_model_server, turn_refusal, turn_text, turn_tool_use,
};
use serde_json::{Value, json};

#[test]
fn serve_follow_up_steers_an_in_flight_run() {
    use std::time::Duration;

    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    // turn 1 runs a 1s sleep (keeps the run in flight long enough to steer), turn 2 ends the turn —
    // at which point the queued follow-up is injected — and turn 3 answers the follow-up.
    let turn1 = turn_tool_use(
        "toolu_s",
        "bash",
        &json!({ "command": "sleep 1" }).to_string(),
    );
    let (base, _bodies) = spawn_model_server(vec![
        turn1,
        turn_text("done with the first part"),
        turn_text("done with the follow-up"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "start" })).unwrap();
    stdin.flush().unwrap();
    // Queue a follow-up while the first turn's sleep is running.
    std::thread::sleep(Duration::from_millis(300));
    writeln!(
        stdin,
        "{}",
        json!({ "type": "follow_up", "id": "f1", "message": "now the second thing" })
    )
    .unwrap();
    stdin.flush().unwrap();

    let frames = read_until_response(&mut stdout, "prompt");
    // The follow-up was acknowledged...
    assert!(
        frames
            .iter()
            .any(|f| f["command"] == "follow_up" && f["success"] == true),
        "follow_up should be acknowledged: {frames:#?}"
    );
    // ...and a `steered` event fired as it was injected.
    assert!(
        frames
            .iter()
            .any(|f| f["type"] == "event" && f["event"]["kind"] == "steered"),
        "a steered event should appear: {frames:#?}"
    );

    // The transcript holds the follow-up text and the second answer.
    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let dump = frames.last().unwrap()["data"]["messages"].to_string();
    assert!(dump.contains("now the second thing"));
    assert!(dump.contains("done with the follow-up"));

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_stop_after_turn_ends_the_run_after_the_current_tool_call_completes() {
    use std::time::Duration;

    // turn 1 runs a 1s sleep (keeps the run in flight long enough to send `stop_after_turn`); turn 2
    // and turn 3 would answer if the run continued — they must never be reached.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let turn1 = turn_tool_use(
        "toolu_stop",
        "bash",
        &json!({ "command": "sleep 1" }).to_string(),
    );
    let (base, bodies) = spawn_model_server(vec![
        turn1,
        turn_text("should never be reached"),
        turn_text("also never reached"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "start" })).unwrap();
    stdin.flush().unwrap();
    // Request a graceful stop while the first turn's bash sleep is still running.
    std::thread::sleep(Duration::from_millis(300));
    writeln!(
        stdin,
        "{}",
        json!({ "type": "stop_after_turn", "id": "s1" })
    )
    .unwrap();
    stdin.flush().unwrap();

    let frames = read_until_response(&mut stdout, "prompt");
    assert!(
        frames
            .iter()
            .any(|f| f["command"] == "stop_after_turn" && f["success"] == true),
        "stop_after_turn should be acknowledged: {frames:#?}"
    );
    // No `steered` event: the run ended, it wasn't redirected.
    assert!(
        !frames
            .iter()
            .any(|f| f["type"] == "event" && f["event"]["kind"] == "steered"),
        "a stop request must not be reported as steering: {frames:#?}"
    );

    // Exactly one model call happened — the run stopped after the first turn's tool call, never
    // asking the model to react to the tool result.
    assert_eq!(
        bodies.lock().unwrap().len(),
        1,
        "the run must not start a second model call after the stop request"
    );

    // The transcript holds the tool call and its result, but neither of the never-reached replies.
    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let dump = frames.last().unwrap()["data"]["messages"].to_string();
    assert!(dump.contains("toolu_stop"), "got: {dump}");
    assert!(!dump.contains("should never be reached"));
    assert!(!dump.contains("also never reached"));

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_stop_after_turn_is_a_no_op_ack_when_idle() {
    // Sent with no `prompt` in flight, `stop_after_turn` must not silently sabotage the *next*
    // prompt (which would only run one turn instead of the two the model script provides).
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, bodies) = spawn_model_server(vec![
        turn_tool_use(
            "toolu_idle",
            "bash",
            &json!({ "command": "echo hi" }).to_string(),
        ),
        turn_text("done"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "stop_after_turn", "id": "s0" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "stop_after_turn");
    assert!(
        frames
            .iter()
            .any(|f| f["command"] == "stop_after_turn" && f["success"] == true),
        "an idle stop_after_turn should still ack: {frames:#?}"
    );

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "start" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    assert_eq!(
        bodies.lock().unwrap().len(),
        2,
        "the prompt sent after an idle stop_after_turn must run to its natural completion"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_prompt_ack_arrives_before_the_first_event_frame() {
    // The lightweight `ack` frame is emitted the moment the turn is queued — before the model call
    // even starts — so it must arrive strictly before any `event` frame in the same `prompt` run.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![turn_text("done")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "id": "p1", "message": "hi" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");

    let ack_pos = frames
        .iter()
        .position(|f| f["type"] == "ack" && f["command"] == "prompt")
        .unwrap_or_else(|| panic!("no ack frame seen: {frames:#?}"));
    let first_event_pos = frames.iter().position(|f| f["type"] == "event");
    if let Some(event_pos) = first_event_pos {
        assert!(
            ack_pos < event_pos,
            "ack must precede the first event frame: {frames:#?}"
        );
    }
    // The ack carries the same id the client sent, so it can be correlated.
    assert_eq!(frames[ack_pos]["id"], "p1");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_busy_prompt_with_streaming_behavior_is_accepted_not_rejected() {
    // A `prompt` sent while another is in flight is normally rejected as busy — unless it carries
    // `streaming_behavior: "steer"|"follow_up"`, in which case it's accepted and routed through the
    // same `Steering` queue as an explicit `steer`/`follow_up` command.
    use std::time::Duration;

    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let turn1 = turn_tool_use(
        "toolu_s",
        "bash",
        &json!({ "command": "sleep 1" }).to_string(),
    );
    let (base, _bodies) = spawn_model_server(vec![
        turn1,
        turn_text("done with the first part"),
        turn_text("done with the steered part"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "start" })).unwrap();
    stdin.flush().unwrap();
    std::thread::sleep(Duration::from_millis(300));
    writeln!(
        stdin,
        "{}",
        json!({
            "type": "prompt",
            "id": "p2",
            "message": "also handle this",
            "streaming_behavior": "steer",
        })
    )
    .unwrap();
    stdin.flush().unwrap();

    // Two distinct "prompt"-command `response` frames arrive here: p2's immediate accept, and the
    // original (id-less) prompt's eventual terminal response — `read_until_response` (which matches by
    // command alone) would stop at whichever comes first, so read manually until both are seen.
    let mut frames = Vec::new();
    let mut prompt_responses_seen = 0;
    let mut line = String::new();
    while prompt_responses_seen < 2 {
        line.clear();
        if stdout.read_line(&mut line).unwrap() == 0 {
            break;
        }
        let Ok(v) = serde_json::from_str::<Value>(line.trim()) else {
            continue;
        };
        if v["type"] == "response" && v["command"] == "prompt" {
            prompt_responses_seen += 1;
        }
        frames.push(v);
    }
    // The busy `prompt` with `streaming_behavior` was accepted (not the "busy" rejection)...
    assert!(
        frames
            .iter()
            .any(|f| f["id"] == "p2" && f["command"] == "prompt" && f["success"] == true),
        "a busy prompt with streaming_behavior must be accepted: {frames:#?}"
    );
    assert!(
        !frames
            .iter()
            .any(|f| f["id"] == "p2" && f["error"].as_str().is_some_and(|e| e.contains("busy"))),
        "must not be rejected as busy: {frames:#?}"
    );
    // ...and was actually injected as a steer.
    assert!(
        frames
            .iter()
            .any(|f| f["type"] == "event" && f["event"]["kind"] == "steered"),
        "a steered event should appear: {frames:#?}"
    );

    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let dump = frames.last().unwrap()["data"]["messages"].to_string();
    assert!(dump.contains("also handle this"));

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_follow_up_queued_while_idle_is_picked_up_by_next_prompt() {
    // No `prompt` is in flight at all yet — `follow_up` must still be accepted (not rejected as an
    // unknown command) and queue against the persistent `Steering` handle, picked up the moment the
    // next `prompt`'s first turn reaches a stop boundary.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, _bodies) = spawn_model_server(vec![
        turn_text("first answer"),
        turn_text("answered the queued follow-up"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // Queue the follow-up first, while genuinely idle.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "follow_up", "id": "f0", "message": "the queued question" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "follow_up");
    assert!(
        frames
            .iter()
            .any(|f| f["command"] == "follow_up" && f["success"] == true),
        "follow_up while idle must be acknowledged, not rejected as unknown: {frames:#?}"
    );

    // Now prompt: turn 1 ends with no tool calls, so the queued follow-up is injected at that stop
    // boundary and turn 2 answers it — all within this one `prompt` call.
    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "start" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    assert!(
        frames.last().unwrap()["success"] == true,
        "prompt should succeed: {frames:#?}"
    );

    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let dump = frames.last().unwrap()["data"]["messages"].to_string();
    assert!(dump.contains("the queued question"));
    assert!(dump.contains("answered the queued follow-up"));

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_follow_up_expands_a_skill_invocation_while_idle() {
    // MEDIUM pi-parity gap (fixed): `follow_up`/`steer` (and `prompt` with `streaming_behavior`) used
    // to push the raw message straight into the steering queue with no `/skill:name`/`/name`
    // expansion — only a fresh top-level `prompt` got that. A `/skill:name` sent through `follow_up`
    // must reach the model as the skill's expanded body, exactly like a fresh `prompt` would, not as
    // the literal unexpanded string.
    let dir = tempfile::tempdir().unwrap();
    let skill_dir = dir.path().join(".claude/skills/foo");
    std::fs::create_dir_all(&skill_dir).unwrap();
    std::fs::write(
        skill_dir.join("SKILL.md"),
        "---\nname: foo\ndescription: a test skill\n---\nSKILL-BODY-MARKER-456",
    )
    .unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, bodies) = spawn_model_server(vec![
        turn_text("first answer"),
        turn_text("answered the skill"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.arg("--trust-project").current_dir(dir.path());
    let mut child = cmd.spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // Queue the skill invocation while genuinely idle, via `follow_up`.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "follow_up", "id": "f0", "message": "/skill:foo" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "follow_up");
    assert!(
        frames
            .iter()
            .any(|f| f["command"] == "follow_up" && f["success"] == true),
        "follow_up while idle must be acknowledged: {frames:#?}"
    );

    // Turn 1 ends with no tool calls, so the queued follow-up is injected at that stop boundary and
    // turn 2 sees it — all within this one `prompt` call.
    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "start" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    assert!(
        frames.last().unwrap()["success"] == true,
        "prompt should succeed: {frames:#?}"
    );
    drop(stdin);
    child.wait().unwrap();

    let recorded = bodies.lock().unwrap();
    assert!(
        recorded.iter().any(|b| b.contains("SKILL-BODY-MARKER-456")),
        "the skill's body must be expanded into the follow-up message before it reaches the model: \
         {recorded:#?}"
    );
    assert!(
        recorded.iter().all(|b| !b.contains("/skill:foo")),
        "the raw, unexpanded invocation must never reach the model: {recorded:#?}"
    );
}

#[test]
fn serve_mid_run_steer_expands_a_skill_invocation() {
    // Same gap as `serve_follow_up_expands_a_skill_invocation_while_idle`, but for the *other* code
    // path: a `steer` sent while a run is genuinely in flight (the busy-loop's own command handler,
    // architecturally distinct from the idle handler above) must expand a `/skill:name` invocation too.
    use std::time::Duration;

    let dir = tempfile::tempdir().unwrap();
    let skill_dir = dir.path().join(".claude/skills/foo");
    std::fs::create_dir_all(&skill_dir).unwrap();
    std::fs::write(
        skill_dir.join("SKILL.md"),
        "---\nname: foo\ndescription: a test skill\n---\nSKILL-BODY-MARKER-789",
    )
    .unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    // turn 1 runs a 1s sleep (keeps the run in flight long enough to steer), turn 2 ends the turn —
    // at which point the steered skill invocation is injected and answered.
    let turn1 = turn_tool_use(
        "toolu_s",
        "bash",
        &json!({ "command": "sleep 1" }).to_string(),
    );
    let (base, bodies) = spawn_model_server(vec![turn1, turn_text("answered the skill")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.arg("--trust-project").current_dir(dir.path());
    let mut child = cmd.spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "start" })).unwrap();
    stdin.flush().unwrap();
    std::thread::sleep(Duration::from_millis(300));
    writeln!(
        stdin,
        "{}",
        json!({ "type": "steer", "id": "s1", "message": "/skill:foo" })
    )
    .unwrap();
    stdin.flush().unwrap();

    let frames = read_until_response(&mut stdout, "prompt");
    assert!(
        frames
            .iter()
            .any(|f| f["command"] == "steer" && f["success"] == true),
        "steer should be acknowledged: {frames:#?}"
    );
    drop(stdin);
    child.wait().unwrap();

    let recorded = bodies.lock().unwrap();
    assert!(
        recorded.iter().any(|b| b.contains("SKILL-BODY-MARKER-789")),
        "the skill's body must be expanded into the steered message before it reaches the model: \
         {recorded:#?}"
    );
    assert!(
        recorded.iter().all(|b| !b.contains("/skill:foo")),
        "the raw, unexpanded invocation must never reach the model: {recorded:#?}"
    );
}

#[test]
fn serve_follow_up_carries_image_attachments_to_the_model() {
    // MEDIUM pi-parity gap (fixed): `follow_up`/`steer` used to have nowhere to put an `images` field
    // at all — a client attaching a screenshot to a queued follow-up had it silently dropped, unlike a
    // fresh `prompt`, which has always supported `images`.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, bodies) =
        spawn_model_server(vec![turn_text("first answer"), turn_text("saw the image")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({
            "type": "follow_up",
            "id": "f0",
            "message": "look at this",
            "images": [{ "media_type": "image/png", "data": "aGVsbG8taW1hZ2UtZGF0YQ==" }],
        })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "follow_up");
    assert!(
        frames
            .iter()
            .any(|f| f["command"] == "follow_up" && f["success"] == true),
        "follow_up with images should be acknowledged: {frames:#?}"
    );

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "start" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    assert!(
        frames.last().unwrap()["success"] == true,
        "prompt should succeed: {frames:#?}"
    );
    drop(stdin);
    child.wait().unwrap();

    let recorded = bodies.lock().unwrap();
    assert!(
        recorded
            .iter()
            .any(|b| b.contains("aGVsbG8taW1hZ2UtZGF0YQ==")),
        "the follow-up's image data must reach the model, not be silently dropped: {recorded:#?}"
    );
}

#[test]
fn serve_no_skills_prevents_discovery_and_leaves_an_invocation_unexpanded() {
    // MEDIUM pi-parity gap (fixed): `serve` had no `--no-skills`/`--no-prompt-templates` at all — only
    // `run` did — so an operator wanting a hardened, no-custom-content `serve` deployment had no way to
    // refuse project-supplied skills. Same fixture/assertions as `run`'s own `--no-skills` test.
    let dir = tempfile::tempdir().unwrap();
    let skill_dir = dir.path().join(".claude/skills/foo");
    std::fs::create_dir_all(&skill_dir).unwrap();
    std::fs::write(
        skill_dir.join("SKILL.md"),
        "---\nname: foo\ndescription: a test skill\n---\nSKILL-BODY-MARKER-999",
    )
    .unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, bodies) = spawn_model_server(vec![turn_text("done")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.args(["--trust-project", "--no-skills"])
        .current_dir(dir.path());
    let mut child = cmd.spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "get_commands" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_commands");
    let commands = frames.last().unwrap()["data"]["commands"]
        .as_array()
        .unwrap();
    assert!(
        commands.is_empty(),
        "--no-skills must prevent the skill from being discovered/advertised at all: {commands:?}"
    );

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "/skill:foo" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    let recorded = bodies.lock().unwrap();
    assert!(
        recorded
            .iter()
            .all(|b| !b.contains("SKILL-BODY-MARKER-999")),
        "the skill's body must never reach the model when --no-skills is set: {recorded:#?}"
    );
    assert!(
        recorded.iter().any(|b| b.contains("/skill:foo")),
        "the raw invocation must reach the model unexpanded: {recorded:#?}"
    );
}

#[test]
fn serve_no_prompt_templates_prevents_discovery_and_leaves_an_invocation_unexpanded() {
    let dir = tempfile::tempdir().unwrap();
    let prompt_dir = dir.path().join(".claude/prompts");
    std::fs::create_dir_all(&prompt_dir).unwrap();
    std::fs::write(prompt_dir.join("bar.md"), "TEMPLATE-BODY-MARKER-999: $1").unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, bodies) = spawn_model_server(vec![turn_text("done")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.args(["--trust-project", "--no-prompt-templates"])
        .current_dir(dir.path());
    let mut child = cmd.spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "get_commands" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_commands");
    let commands = frames.last().unwrap()["data"]["commands"]
        .as_array()
        .unwrap();
    assert!(
        commands.is_empty(),
        "--no-prompt-templates must prevent the template from being discovered/advertised at all: \
         {commands:?}"
    );

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "/bar arg" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    let recorded = bodies.lock().unwrap();
    assert!(
        recorded
            .iter()
            .all(|b| !b.contains("TEMPLATE-BODY-MARKER-999")),
        "the template's body must never reach the model when --no-prompt-templates is set: \
         {recorded:#?}"
    );
    assert!(
        recorded.iter().any(|b| b.contains("/bar arg")),
        "the raw invocation must reach the model unexpanded: {recorded:#?}"
    );
}

#[test]
fn serve_default_queue_mode_drains_queued_follow_ups_one_at_a_time() {
    // pi's `PendingMessageQueue` default: several messages queued in quick succession land as
    // *separate* turns, one at a time — not folded into a single injection. Two follow-ups queued
    // while idle must reach the model server as two distinct requests (three total, including the
    // initial prompt), each carrying exactly one of the two follow-up texts as its newest user turn.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, bodies) = spawn_model_server(vec![
        turn_text("first answer"),
        turn_text("answered f1"),
        turn_text("answered f2"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    for (id, msg) in [("f1", "first follow-up"), ("f2", "second follow-up")] {
        writeln!(
            stdin,
            "{}",
            json!({ "type": "follow_up", "id": id, "message": msg })
        )
        .unwrap();
        stdin.flush().unwrap();
        read_until_response(&mut stdout, "follow_up");
    }

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "start" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    assert_eq!(frames.last().unwrap()["success"], true, "got: {frames:#?}");

    let bodies = bodies.lock().unwrap();
    assert_eq!(
        bodies.len(),
        3,
        "initial prompt + one request per follow-up, not one merged request"
    );
    assert!(
        bodies[1].contains("first follow-up") && !bodies[1].contains("second follow-up"),
        "the second request should carry only the first follow-up: {}",
        bodies[1]
    );
    assert!(
        bodies[2].contains("first follow-up") && bodies[2].contains("second follow-up"),
        "the third request replays history, so it sees both by then, but as separate turns: {}",
        bodies[2]
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_set_queue_mode_all_folds_queued_follow_ups_into_one_injection() {
    // The opt-in `"all"` mode (this crate's original behavior, before pi-parity default flipped it):
    // both queued follow-ups are folded into the *same* next request, not drained one at a time.
    // Follow-ups are governed by `set_follow_up_mode`, not `set_steering_mode` — the two lanes have
    // independent settings.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, bodies) =
        spawn_model_server(vec![turn_text("first answer"), turn_text("answered both")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_follow_up_mode", "mode": "all" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_follow_up_mode");
    assert_eq!(frames.last().unwrap()["success"], true, "got: {frames:#?}");
    assert_eq!(frames.last().unwrap()["data"]["mode"], "all");

    for (id, msg) in [("f1", "first follow-up"), ("f2", "second follow-up")] {
        writeln!(
            stdin,
            "{}",
            json!({ "type": "follow_up", "id": id, "message": msg })
        )
        .unwrap();
        stdin.flush().unwrap();
        read_until_response(&mut stdout, "follow_up");
    }

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "start" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    assert_eq!(frames.last().unwrap()["success"], true, "got: {frames:#?}");

    let bodies = bodies.lock().unwrap();
    assert_eq!(
        bodies.len(),
        2,
        "initial prompt + one request carrying BOTH follow-ups folded together"
    );
    assert!(
        bodies[1].contains("first follow-up") && bodies[1].contains("second follow-up"),
        "both follow-ups should reach the model in the same request: {}",
        bodies[1]
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_steering_mode_and_follow_up_mode_are_independent_rpc_settings() {
    // Track M12: `set_queue_mode` split into `set_steering_mode`/`set_follow_up_mode` — setting one
    // via RPC must not clobber the other, and `get_state` must report both independently.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_steering_mode", "mode": "all" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "set_steering_mode");

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    let data = &frames.last().unwrap()["data"];
    assert_eq!(data["steering_mode"], "all", "{data:#?}");
    assert_eq!(
        data["follow_up_mode"], "one_at_a_time",
        "follow_up_mode must be untouched by a steering-mode change: {data:#?}"
    );

    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_follow_up_mode", "mode": "all" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "set_follow_up_mode");
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_steering_mode", "mode": "one_at_a_time" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "set_steering_mode");

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    let data = &frames.last().unwrap()["data"];
    assert_eq!(
        data["steering_mode"], "one_at_a_time",
        "steering_mode must be untouched by a follow-up-mode change: {data:#?}"
    );
    assert_eq!(data["follow_up_mode"], "all", "{data:#?}");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_refusal_ends_the_run_without_draining_steering() {
    // A refusal must be a distinct terminal condition: the `prompt` response reports `refused: true`,
    // no second model call happens (a queued follow-up is NOT drained/injected right after a refusal),
    // and the queued message survives untouched for a later `prompt` to pick up.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, _bodies) = spawn_model_server(vec![
        turn_refusal("I can't help with that."),
        turn_text("second prompt's normal answer"),
        turn_text("answered the queued follow-up"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // Queue a follow-up while idle, before the refusal even happens.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "follow_up", "id": "f0", "message": "should stay queued" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "follow_up");

    // First prompt: the model refuses.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "please do something disallowed" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    let response = frames.last().unwrap();
    assert_eq!(response["success"], true);
    assert_eq!(
        response["data"]["refused"], true,
        "refused must be reported: {response:#?}"
    );
    assert!(
        !frames
            .iter()
            .any(|f| f["type"] == "event" && f["event"]["kind"] == "steered"),
        "a refusal must not drain/inject the queued follow-up: {frames:#?}"
    );

    // Second prompt: an ordinary stop — the queued follow-up (still intact) is drained/injected now.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "a normal message" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    assert!(
        frames
            .iter()
            .any(|f| f["type"] == "event" && f["event"]["kind"] == "steered"),
        "the queued follow-up must survive the refusal and be injected on the next stop: {frames:#?}"
    );

    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let dump = frames.last().unwrap()["data"]["messages"].to_string();
    assert!(dump.contains("should stay queued"));
    assert!(dump.contains("answered the queued follow-up"));

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_abort_cancels_an_in_flight_prompt() {
    use std::time::{Duration, Instant};

    let dir = tempfile::tempdir().unwrap();
    let session_file = dir
        .path()
        .join("session.json")
        .to_string_lossy()
        .into_owned();

    // The model asks to run a 30s shell sleep; the run will be aborted mid-tool, so a second turn is
    // never requested.
    let turn1 = turn_tool_use(
        "toolu_b",
        "bash",
        &json!({ "command": "sleep 30" }).to_string(),
    );
    let (base, _bodies) = spawn_model_server(vec![turn1]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "run a long sleep" })
    )
    .unwrap();
    stdin.flush().unwrap();

    // Give the run time to reach the tool, then abort.
    std::thread::sleep(Duration::from_millis(500));
    writeln!(stdin, "{}", json!({ "type": "abort", "id": "a1" })).unwrap();
    stdin.flush().unwrap();

    // The prompt response must come back promptly (well under the 30s sleep) and report failure.
    let start = Instant::now();
    let frames = read_until_response(&mut stdout, "prompt");
    assert!(
        start.elapsed() < Duration::from_secs(15),
        "abort must cancel the in-flight prompt promptly, took {:?}",
        start.elapsed()
    );
    let resp = frames.last().unwrap();
    assert_eq!(resp["command"], "prompt");
    assert_eq!(resp["success"], false, "an aborted prompt reports failure");
    assert!(
        frames
            .iter()
            .any(|f| f["type"] == "response" && f["command"] == "abort" && f["success"] == true),
        "the abort command should have been acknowledged: {frames:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_bare_prompt_while_busy_is_rejected_not_queued() {
    // pi: agent-session-concurrent.test.ts / agent-session-prompt.test.ts — a bare `prompt` (no
    // `streaming_behavior`) sent while one is already in flight must be rejected as busy, distinct
    // from the accepted case (`serve_busy_prompt_with_streaming_behavior_is_accepted_not_rejected`
    // above, which carries `streaming_behavior`).
    use std::time::Duration;

    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let turn1 = turn_tool_use(
        "toolu_busy",
        "bash",
        &json!({ "command": "sleep 1" }).to_string(),
    );
    let (base, _bodies) = spawn_model_server(vec![turn1, turn_text("done")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "start" })).unwrap();
    stdin.flush().unwrap();
    std::thread::sleep(Duration::from_millis(300));
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "id": "p2", "message": "also handle this" })
    )
    .unwrap();
    stdin.flush().unwrap();

    let rejection = read_until_response(&mut stdout, "prompt");
    let p2 = rejection
        .iter()
        .find(|f| f["id"] == "p2")
        .expect("p2's own response frame");
    assert_eq!(p2["success"], false, "got: {p2:#?}");
    assert!(
        p2["error"].as_str().unwrap_or_default().contains("busy"),
        "got: {p2:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
}
