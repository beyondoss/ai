//! `run` e2e: One-shot `run` execution: tool round trips, refusal exit codes, text vs json mode, multi-message turns, stdin/@file input.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::Write;
use std::process::Stdio;

use common::{
    SpawnGuarded, run_cmd, spawn_model_server, sse, turn_refusal, turn_text, turn_tool_use,
};
use serde_json::{Value, json};

/// Build an Anthropic SSE turn that calls one tool with its JSON arguments split across multiple
/// `input_json_delta` fragments, the way a real streaming response actually arrives (not the single
/// whole-string delta `turn_tool_use` uses).
fn turn_tool_use_chunked(id: &str, name: &str, arg_fragments: &[&str]) -> String {
    let mut events = vec![
        json!({ "type": "message_start", "message": { "usage": { "input_tokens": 10, "output_tokens": 1 } } }),
        json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "tool_use", "id": id, "name": name, "input": {} } }),
    ];
    for fragment in arg_fragments {
        events.push(json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": { "type": "input_json_delta", "partial_json": fragment }
        }));
    }
    events.push(json!({ "type": "content_block_stop", "index": 0 }));
    events.push(
        json!({ "type": "message_delta", "delta": { "stop_reason": "tool_use" }, "usage": { "output_tokens": 8 } }),
    );
    events.push(json!({ "type": "message_stop" }));
    sse(&events)
}

#[test]
fn run_binary_performs_tool_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("hello.txt"), "secret-marker-42\n").unwrap();
    let abs = dir.path().join("hello.txt").to_string_lossy().into_owned();

    let turn1 = turn_tool_use("toolu_1", "read", &json!({ "path": abs }).to_string());
    let turn2 = turn_text("I read the file.");
    let (base, bodies) = spawn_model_server(vec![turn1, turn2]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "read hello.txt and report it",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--max-steps",
            "4",
            "--no-session-persistence",
        ])
        .current_dir(dir.path())
        .output()
        .expect("spawn binary");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "binary failed.\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stdout.contains("[tool: read]"),
        "should show the tool call.\nstdout: {stdout}"
    );
    assert!(
        stdout.contains(&abs),
        "the streamed tool-call arguments (a live preview of the call, not just its name) should \
         appear in stdout.\nstdout: {stdout}"
    );
    assert!(
        stdout.contains("I read the file."),
        "should print the final turn.\nstdout: {stdout}"
    );

    let bodies = bodies.lock().unwrap();
    assert_eq!(bodies.len(), 2, "expected two model requests");
    assert!(
        bodies[1].contains("secret-marker-42"),
        "the 2nd request must feed the file contents back as a tool_result.\nbody: {}",
        bodies[1]
    );
}

#[test]
fn run_binary_exits_nonzero_on_a_refusal_in_text_mode() {
    // Track L18: a refusal in text mode previously still exited 0, indistinguishable from a normal
    // completion — a script/CI caller has no other signal to key off of in that mode.
    let (base, _bodies) = spawn_model_server(vec![turn_refusal("I can't help with that.")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "do something the model refuses",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--no-session-persistence",
        ])
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        !output.status.success(),
        "a refusal in text mode must exit non-zero.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.status.code(), Some(1));
}

#[test]
fn run_binary_a_normal_completion_in_text_mode_still_exits_zero() {
    // The exit-code check must not misfire on an ordinary end-of-turn — only an actual refusal.
    let (base, _bodies) = spawn_model_server(vec![turn_text("all done")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "hi",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--no-session-persistence",
        ])
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn run_binary_refusal_exit_code_check_is_scoped_to_text_mode() {
    // `--json` mode already carries `stop_reason` on every `TurnEnd` event in its own output stream —
    // a caller there is expected to inspect it programmatically, not rely on the process exit code.
    let (base, _bodies) = spawn_model_server(vec![turn_refusal("I can't help with that.")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "do something the model refuses",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--json",
            "--no-session-persistence",
        ])
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "--json mode must not apply the text-mode-only refusal exit code.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("\"refusal\"") || stdout.contains("refusal"),
        "the refusal must still be observable in the JSON event stream itself: {stdout}"
    );
}

#[test]
fn run_binary_json_mode_streams_structured_agent_events_not_text() {
    // `--json` must emit a leading session header, then one `AgentEvent` object per line — the full
    // observation surface (tool_start/tool_end included, not just raw text deltas) — instead of the
    // human-readable `[tool: name]`/plain-text output the default text mode prints.
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("hello.txt"), "secret-marker-42\n").unwrap();
    let abs = dir.path().join("hello.txt").to_string_lossy().into_owned();

    let turn1 = turn_tool_use("toolu_1", "read", &json!({ "path": abs }).to_string());
    let turn2 = turn_text("I read the file.");
    let (base, _bodies) = spawn_model_server(vec![turn1, turn2]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "read hello.txt and report it",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--max-steps",
            "4",
            "--json",
            "--no-session-persistence",
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "binary failed.\nstdout: {stdout}\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !stdout.contains("[tool:"),
        "json mode must not print the text-mode tool marker: {stdout}"
    );

    let lines: Vec<Value> = stdout
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("non-JSON line {l:?}: {e}")))
        .collect();
    assert!(!lines.is_empty(), "must emit at least the header line");

    let header = &lines[0];
    assert_eq!(header["kind"], "session");
    assert!(header["id"].as_str().is_some_and(|s| !s.is_empty()));
    assert_eq!(header["model"], "claude-test");

    let kinds: Vec<&str> = lines[1..]
        .iter()
        .filter_map(|f| f["kind"].as_str())
        .collect();
    assert_eq!(
        kinds.first(),
        Some(&"agent_start"),
        "kinds: {kinds:?}\nstdout: {stdout}"
    );
    assert!(kinds.contains(&"tool_start"), "kinds: {kinds:?}");
    assert!(kinds.contains(&"tool_end"), "kinds: {kinds:?}");
    assert!(
        kinds.iter().filter(|k| **k == "turn_end").count() >= 2,
        "two turns ran (tool call, then final text): {kinds:?}"
    );
    assert_eq!(
        kinds.last(),
        Some(&"agent_end"),
        "kinds: {kinds:?}\nstdout: {stdout}"
    );

    // The final assistant text must still be present, just carried as a `stream`/`text_delta` event
    // rather than printed as bare text.
    let carries_final_text = lines.iter().any(|f| {
        f["kind"] == "stream"
            && f["type"] == "text_delta"
            && f["text"]
                .as_str()
                .is_some_and(|t| t.contains("I read the file."))
    });
    assert!(
        carries_final_text,
        "final assistant text must appear in a stream/text_delta event: {stdout}"
    );
}

#[test]
fn run_binary_session_id_flag_gives_a_deterministic_id_for_an_ephemeral_run() {
    // Track L6: `--session-id` (matching pi's own flag) lets a script/test harness pick a known,
    // predictable id instead of parsing a randomly-generated one back out of the run's own output —
    // for a plain ephemeral run (no `--session`/`--continue`), the case that previously had no way to
    // override the id at all.
    let dir = tempfile::tempdir().unwrap();
    let (base, _bodies) = spawn_model_server(vec![turn_text("ok")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "hi",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session-id",
            "deterministic-test-id-42",
            "--json",
            "--no-session-persistence",
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "binary failed.\nstdout: {stdout}\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let header: Value = serde_json::from_str(stdout.lines().next().unwrap()).unwrap();
    assert_eq!(header["kind"], "session");
    assert_eq!(
        header["id"], "deterministic-test-id-42",
        "the ephemeral run must report exactly the requested id, not a generated one: {header:#?}"
    );
}

#[test]
fn run_binary_prints_a_live_preview_of_streamed_tool_arguments() {
    // A real streaming response delivers a tool call's JSON arguments as several fragments, not one
    // whole string. Assembled in order right after the `[tool: name]` marker, they must read as the
    // complete, valid argument JSON on stdout — proving the loop actually renders each
    // `InputJsonDelta` live rather than only ever seeing (and rendering) a single final fragment.
    let dir = tempfile::tempdir().unwrap();
    let turn1 = turn_tool_use_chunked(
        "toolu_1",
        "unknown_preview_tool",
        &[r#"{"comman"#, r#"d":"echo hi"}"#],
    );
    let (base, _bodies) = spawn_model_server(vec![turn1, turn_text("done")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "do the thing",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--max-steps",
            "4",
            "--no-session-persistence",
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "binary failed.\nstdout: {stdout}\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("[tool: unknown_preview_tool] {\"command\":\"echo hi\"}"),
        "the two fragments must render adjacently, in order, as the complete argument JSON: {stdout}"
    );
}

#[test]
fn run_binary_prints_startup_timings_only_when_the_env_var_is_set() {
    // Track L10: `AI_AGENT_TIMING=1` (pi's own `PI_TIMING=1`) turns on a startup-timing breakdown to
    // stderr; unset, it must add nothing at all — and even enabled, it must never touch stdout, since
    // that's the streamed-text protocol surface.
    let (base, _bodies) = spawn_model_server(vec![turn_text("done")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let args = [
        "run",
        "hi",
        "--gateway-url",
        &base,
        "--key",
        "bai_v1.test",
        "--model",
        "claude-test",
        "--no-session-persistence",
    ];

    let off = run_cmd(bin)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");
    assert!(off.status.success());
    assert!(
        !String::from_utf8_lossy(&off.stderr).contains("Startup Timings"),
        "must print nothing when unset: {}",
        String::from_utf8_lossy(&off.stderr)
    );

    let (base2, _bodies2) = spawn_model_server(vec![turn_text("done")]);
    let on = run_cmd(bin)
        .args([
            "run",
            "hi",
            "--gateway-url",
            &base2,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--no-session-persistence",
        ])
        .env("AI_AGENT_TIMING", "1")
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");
    assert!(on.status.success());
    let stderr = String::from_utf8_lossy(&on.stderr);
    assert!(stderr.contains("Startup Timings"), "stderr: {stderr}");
    assert!(stderr.contains("TOTAL:"), "stderr: {stderr}");
    let stdout = String::from_utf8_lossy(&on.stdout);
    assert!(
        !stdout.contains("Startup Timings"),
        "timing output must never reach stdout: {stdout}"
    );
    assert!(stdout.contains("done"), "stdout: {stdout}");
}

#[test]
fn run_binary_sends_multiple_messages_as_sequential_turns() {
    // Two positional messages must run as two separate turns — the second only sent after the first
    // fully completes — not concatenated into one prompt.
    let dir = tempfile::tempdir().unwrap();
    let (base, bodies) =
        spawn_model_server(vec![turn_text("first reply"), turn_text("second reply")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "first message",
            "second message",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--no-session-persistence",
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "binary failed.\nstdout: {stdout}\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("first reply"));
    assert!(stdout.contains("second reply"));

    let bodies = bodies.lock().unwrap();
    assert_eq!(bodies.len(), 2, "each message must be its own model call");
    assert!(
        bodies[0].contains("first message") && !bodies[0].contains("second message"),
        "the first request must not already include the second message: {}",
        bodies[0]
    );
    assert!(
        bodies[1].contains("first message") && bodies[1].contains("second message"),
        "the second request must carry the accumulated history: {}",
        bodies[1]
    );
}

#[test]
fn run_binary_composes_an_at_file_reference_into_the_first_message() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("notes.txt"), "the-file-marker-77").unwrap();
    let (base, bodies) = spawn_model_server(vec![turn_text("got it")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "@notes.txt",
            "summarize this",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--no-session-persistence",
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "binary failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let bodies = bodies.lock().unwrap();
    assert!(
        bodies[0].contains("the-file-marker-77") && bodies[0].contains("summarize this"),
        "the request must carry both the file contents and the message: {}",
        bodies[0]
    );
}

#[test]
fn run_binary_attaches_an_at_referenced_image_instead_of_erroring() {
    // Track L20 (pi-parity fix): `run @screenshot.png "..."` used to crash — `read_file_refs` plain
    // `std::fs::read_to_string`'d every `@file` ref, which errors outright on binary image data. A
    // real (magic-byte-detectable) image reference must now reach the model as an image content block
    // instead of failing the whole invocation.
    use base64::Engine as _;
    // Same fixture `smoke.rs`'s real-provider vision test uses: a tiny, deterministically-generated,
    // genuinely decodable PNG (not just a magic-byte prefix) — a 48x48 solid-red swatch.
    const RED_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAIAAADYYG7QAAAANklEQVR42u3OQQ0AAAgAoetfWls4H2wEoKlXEhISEhISEhISEhISEhISEhISEhISEhISEhK6s98T93mKDkyKAAAAAElFTkSuQmCC";
    let png = base64::engine::general_purpose::STANDARD
        .decode(RED_PNG_B64)
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("swatch.png"), &png).unwrap();
    let (base, bodies) = spawn_model_server(vec![turn_text("I see a red swatch.")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "@swatch.png",
            "what color is this?",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--no-session-persistence",
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "binary failed instead of attaching the image.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let bodies = bodies.lock().unwrap();
    assert!(
        bodies[0].contains(RED_PNG_B64),
        "the actual image bytes (base64) must reach the model, not a read error: {}",
        bodies[0]
    );
    assert!(
        bodies[0].contains("image/png"),
        "the request must carry an image content block: {}",
        bodies[0]
    );
}

#[test]
fn run_binary_reads_piped_stdin_into_the_first_message() {
    let dir = tempfile::tempdir().unwrap();
    let (base, bodies) = spawn_model_server(vec![turn_text("noted")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = run_cmd(bin)
        .args([
            "run",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--no-session-persistence",
        ])
        .current_dir(dir.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn_guarded();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"piped-content-marker-99")
        .unwrap();
    let output = child.wait_with_output().unwrap();

    assert!(
        output.status.success(),
        "binary failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let bodies = bodies.lock().unwrap();
    assert!(
        bodies[0].contains("piped-content-marker-99"),
        "piped stdin must reach the model as the message: {}",
        bodies[0]
    );
}

#[test]
fn run_binary_errors_when_no_task_stdin_or_file_is_given() {
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "--gateway-url",
            "http://127.0.0.1:1",
            "--key",
            "bai_v1.test",
            "--no-session-persistence",
        ])
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        !output.status.success(),
        "an empty invocation must fail, not silently prompt with nothing"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("no task given"),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
