//! `run` e2e: `--output-schema` turning a run into a callable function, driven through the real binary.
//!
//! Covers the whole contract: the tool and its prompt section are only present when asked for; a valid
//! payload is the process's stdout and its exit code; a schema violation is fed back and retried; a run
//! that never answers exits non-zero; a bad schema fails before a single model call is billed; and the
//! `--json` stream gains exactly one line, and only when the flag is set.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::{advertised_tools, run_cmd, spawn_model_server, turn_text, turn_tool_use};
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_beyond-ai-agent");

/// A strict schema — `additionalProperties: false` is the right thing for one describing an exact
/// payload, and is exactly what the loop's injected `_model_supports_vision` key would otherwise break.
fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "files_changed": { "type": "array", "items": { "type": "string" } },
            "summary": { "type": "string" }
        },
        "required": ["files_changed", "summary"],
        "additionalProperties": false
    })
}

fn answer(files: &[&str], summary: &str) -> String {
    json!({ "files_changed": files, "summary": summary }).to_string()
}

fn so_turn(id: &str, payload: &str) -> String {
    turn_tool_use(id, "structured_output", payload)
}

struct Run {
    stdout: String,
    stderr: String,
    code: Option<i32>,
    bodies: Vec<String>,
}

impl Run {
    fn ok(&self) -> bool {
        self.code == Some(0)
    }
    /// The last non-empty stdout line — where the payload lands.
    fn last_line(&self) -> &str {
        self.stdout
            .lines()
            .rfind(|l| !l.trim().is_empty())
            .unwrap_or("")
    }
}

fn run(turns: Vec<String>, extra: &[&str]) -> Run {
    let dir = tempfile::tempdir().unwrap();
    let (base, bodies) = spawn_model_server(turns);
    let output = run_cmd(BIN)
        .args([
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
        ])
        .args(extra)
        .current_dir(dir.path())
        .output()
        .expect("spawn binary");
    let bodies = bodies.lock().unwrap().clone();
    Run {
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        code: output.status.code(),
        bodies,
    }
}

fn with_schema(turns: Vec<String>, extra: &[&str]) -> Run {
    let s = schema().to_string();
    let mut args = vec!["--output-schema", s.as_str()];
    args.extend_from_slice(extra);
    run(turns, &args)
}

#[test]
fn a_valid_payload_is_the_last_stdout_line_and_the_run_exits_zero() {
    let r = with_schema(vec![so_turn("tu_1", &answer(&["a.rs"], "did it"))], &[]);
    assert!(r.ok(), "exit {:?}\nstderr: {}", r.code, r.stderr);

    // The contract: the payload is the last stdout line, so `... | tail -1 | jq` works.
    let value: Value =
        serde_json::from_str(r.last_line()).expect("last stdout line must be the payload");
    assert_eq!(value["summary"], "did it");
    assert_eq!(value["files_changed"][0], "a.rs");
    // The loop's schema-undocumented injected key must never reach the caller.
    assert!(
        value.get("_model_supports_vision").is_none(),
        "got: {value}"
    );

    // Exactly one model call: `terminate` ended the run, no wrap-up turn was requested.
    assert_eq!(r.bodies.len(), 1, "the tool must terminate the run");
}

#[test]
fn the_tool_and_its_protocol_reach_the_model_only_when_a_schema_is_given() {
    let with = with_schema(vec![so_turn("tu_1", &answer(&[], "x"))], &[]);
    let body = &with.bodies[0];
    assert!(
        advertised_tools(body)
            .iter()
            .any(|t| t == "structured_output"),
        "tool must be advertised"
    );
    assert!(
        body.contains("structured_output_protocol"),
        "prompt must carry the contract"
    );
    assert!(body.contains("Do not describe the result in prose"));
    // The caller's schema is what the model is shown, verbatim.
    assert!(
        body.contains("files_changed"),
        "the schema must be the tool's input_schema"
    );

    let without = run(vec![turn_text("prose")], &[]);
    let body = &without.bodies[0];
    assert!(
        !body.contains("structured_output"),
        "no tool, no protocol: {body}"
    );
}

#[test]
fn an_explicit_tools_allow_list_cannot_strip_the_tool_the_flag_exists_to_add() {
    // `--tools read` scopes what the agent may *do*. Silently dropping `structured_output` would leave a
    // run that can never satisfy the contract it was started with.
    let r = with_schema(
        vec![so_turn("tu_1", &answer(&["a.rs"], "scoped"))],
        &["--tools", "read"],
    );
    assert!(r.ok(), "exit {:?}\nstderr: {}", r.code, r.stderr);
    let tools = advertised_tools(&r.bodies[0]);
    assert!(tools.iter().any(|t| t == "structured_output"));
    assert!(tools.iter().any(|t| t == "read"));
    assert!(
        !tools.iter().any(|t| t == "bash"),
        "the allow-list must still apply to everything else: {tools:?}"
    );
}

#[test]
fn a_schema_violation_is_fed_back_and_the_model_corrects_itself() {
    let r = with_schema(
        vec![
            so_turn(
                "tu_1",
                &json!({ "summary": "forgot the array" }).to_string(),
            ),
            so_turn("tu_2", &answer(&["b.rs"], "corrected")),
        ],
        &[],
    );
    assert!(
        r.ok(),
        "a rejected payload must not fail the run: {}",
        r.stderr
    );
    assert_eq!(
        r.bodies.len(),
        2,
        "the model must get a second turn to correct itself"
    );

    // The retry prompt carries the complaint, naming the field that was wrong.
    assert!(
        r.bodies[1].contains("files_changed"),
        "the model must be told what was wrong: {}",
        r.bodies[1]
    );
    let value: Value = serde_json::from_str(r.last_line()).expect("payload");
    assert_eq!(value["summary"], "corrected");
}

#[test]
fn a_run_that_never_calls_the_tool_exits_non_zero_with_a_diagnostic() {
    // Exiting 0 here would be indistinguishable from success to a script piping stdout into `jq`.
    let r = with_schema(vec![turn_text("here is a prose answer instead")], &[]);
    assert_eq!(
        r.code,
        Some(1),
        "stdout: {}\nstderr: {}",
        r.stdout,
        r.stderr
    );
    assert!(
        r.stderr.contains("no structured output"),
        "stderr must say why: {}",
        r.stderr
    );
}

#[test]
fn an_invalid_schema_fails_before_any_model_call_is_billed() {
    let r = run(
        vec![turn_text("never reached")],
        &["--output-schema", r#"{"type":"array"}"#],
    );
    assert_eq!(r.code, Some(2), "stderr: {}", r.stderr);
    assert!(
        r.stderr.contains("\"type\": \"object\""),
        "stderr: {}",
        r.stderr
    );
    assert!(
        r.bodies.is_empty(),
        "no model call may be made for an unusable schema"
    );

    let r = run(
        vec![turn_text("never reached")],
        &["--output-schema", "{not json"],
    );
    assert_eq!(r.code, Some(2));
    assert!(r.stderr.contains("not valid JSON"), "stderr: {}", r.stderr);
    assert!(r.bodies.is_empty());
}

#[test]
fn a_schema_can_be_read_from_a_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("schema.json");
    std::fs::write(&path, schema().to_string()).unwrap();

    let (base, _bodies) =
        spawn_model_server(vec![so_turn("tu_1", &answer(&["c.rs"], "from a file"))]);
    let output = run_cmd(BIN)
        .args([
            "run",
            "do the work",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--no-session-persistence",
            "--output-schema",
            path.to_str().unwrap(),
        ])
        .current_dir(dir.path())
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let last = stdout.lines().rfind(|l| !l.trim().is_empty()).unwrap_or("");
    let value: Value = serde_json::from_str(last).expect("payload");
    assert_eq!(value["summary"], "from a file");
}

#[test]
fn output_description_overrides_what_the_model_is_told_the_payload_is_for() {
    let r = with_schema(
        vec![so_turn("tu_1", &answer(&[], "x"))],
        &["--output-description", "Return the review verdict."],
    );
    assert!(r.ok());
    assert!(
        r.bodies[0].contains("Return the review verdict."),
        "{}",
        r.bodies[0]
    );
}

#[test]
fn json_mode_emits_exactly_one_extra_line_and_only_with_a_schema() {
    let r = with_schema(
        vec![so_turn("tu_1", &answer(&["d.rs"], "streamed"))],
        &["--json"],
    );
    assert!(r.ok(), "stderr: {}", r.stderr);

    let lines: Vec<Value> = r
        .stdout
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    let payloads: Vec<&Value> = lines
        .iter()
        .filter(|v| v["kind"] == "structured_output")
        .collect();
    assert_eq!(payloads.len(), 1, "exactly one payload line");
    assert_eq!(payloads[0]["value"]["summary"], "streamed");
    // It is the last line: everything a consumer needs arrives before it.
    assert_eq!(r.last_line(), payloads[0].to_string());

    // Without the flag, an existing `--json` consumer sees no new line at all.
    let plain = run(vec![turn_text("prose")], &["--json"]);
    assert!(plain.ok());
    assert!(
        !plain.stdout.contains("structured_output"),
        "no schema, no extra line: {}",
        plain.stdout
    );
}

/// One assistant turn dispatching several tool calls at once — the shape `terminate`'s batch-AND rule
/// exists for. `turn_tool_use` only ever emits a single block.
fn turn_tool_uses(calls: &[(&str, &str, String)]) -> String {
    let mut events = vec![
        json!({ "type": "message_start", "message": { "usage": { "input_tokens": 10, "output_tokens": 1 } } }),
    ];
    for (i, (id, name, args)) in calls.iter().enumerate() {
        events.push(json!({ "type": "content_block_start", "index": i, "content_block": { "type": "tool_use", "id": id, "name": name, "input": {} } }));
        events.push(json!({ "type": "content_block_delta", "index": i, "delta": { "type": "input_json_delta", "partial_json": args } }));
        events.push(json!({ "type": "content_block_stop", "index": i }));
    }
    events.push(
        json!({ "type": "message_delta", "delta": { "stop_reason": "tool_use" }, "usage": { "output_tokens": 8 } }),
    );
    events.push(json!({ "type": "message_stop" }));
    common::sse(&events)
}

#[test]
fn a_mixed_batch_stages_the_payload_and_lets_the_sibling_call_finish() {
    // `terminate` is ANDed across the batch, so a `structured_output` call must not cut off a `read` the
    // model dispatched alongside it. The payload is staged, the run continues, and a later solo call
    // revises it — which is the value the caller actually receives.
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("f.txt"), "SENTINEL-CONTENT\n").unwrap();
    let read_path = dir.path().join("f.txt").to_string_lossy().into_owned();

    let (base, bodies) = spawn_model_server(vec![
        turn_tool_uses(&[
            ("tu_1", "structured_output", answer(&["stale.rs"], "staged")),
            ("tu_2", "read", json!({ "path": read_path }).to_string()),
        ]),
        so_turn("tu_3", &answer(&["final.rs"], "revised")),
    ]);

    let s = schema().to_string();
    let output = run_cmd(BIN)
        .args([
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
            "--output-schema",
            &s,
        ])
        .current_dir(dir.path())
        .output()
        .expect("spawn binary");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "stderr: {stderr}");
    let bodies = bodies.lock().unwrap().clone();
    assert_eq!(bodies.len(), 2, "the mixed batch must not end the run");

    // The sibling `read` really ran, and its result was fed back.
    assert!(
        bodies[1].contains("SENTINEL-CONTENT"),
        "the sibling tool call must not have been cut off: {}",
        bodies[1]
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let last = stdout.lines().rfind(|l| !l.trim().is_empty()).unwrap_or("");
    let value: Value = serde_json::from_str(last).expect("payload");
    assert_eq!(
        value["summary"], "revised",
        "the host reads the slot after the run drains, so it sees the final value"
    );
}
