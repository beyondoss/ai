//! `serve` e2e: Model and thinking-level/reasoning-effort selection, cycling, and `--models` scoping.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufReader, Write};
use std::process::{Command, Stdio};

use common::{ISOLATED_HOME, read_until_response, serve_cmd, spawn_model_server};
use serde_json::{Value, json};

/// Like `serve_cmd`, but with an explicit `--model` instead of the hardcoded `"claude-test"` — for
/// tests exercising model-specific reasoning-effort clamping, where the test model itself matters.
fn serve_cmd_with_model(bin: &str, base: &str, session_file: &str, model: &str) -> Command {
    let mut c = Command::new(bin);
    c.args([
        "serve",
        "--gateway-url",
        base,
        "--key",
        "bai_v1.test",
        "--model",
        model,
        "--session-file",
        session_file,
    ])
    .env("HOME", ISOLATED_HOME)
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::null());
    c
}

#[test]
fn serve_switches_model_and_thinking_at_runtime() {
    // These are pure control commands — no model call — so the mock server is never hit.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // The known-model list is returned and non-empty, each entry a structured capability object (F-M2:
    // pi's `Model<any>` shape — `id`/`contextWindow`/`reasoning`, minus pricing — not a bare id string).
    writeln!(stdin, "{}", json!({ "type": "get_available_models" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_available_models");
    let models = frames.last().unwrap()["data"]["models"].as_array().unwrap();
    let opus = models
        .iter()
        .find(|m| m["id"] == "claude-opus-4-8")
        .unwrap_or_else(|| panic!("model list should include the default opus id: {models:#?}"));
    assert!(
        opus["context_window"].as_u64().unwrap() > 0,
        "got: {opus:#?}"
    );
    assert!(opus["reasoning"].is_boolean(), "got: {opus:#?}");
    assert_eq!(opus["provider"], "anthropic", "got: {opus:#?}");

    // Switch the model; the response echoes it and `get_state` reflects it.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_model", "model": "gpt-4o" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_model");
    assert_eq!(frames.last().unwrap()["data"]["model"], "gpt-4o");

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    assert_eq!(
        frames.last().unwrap()["data"]["model"],
        "gpt-4o",
        "get_state must reflect the switched model"
    );

    // A missing `model` is rejected.
    writeln!(stdin, "{}", json!({ "type": "set_model" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_model");
    assert_eq!(frames.last().unwrap()["success"], false);

    // Track L19: an empty/whitespace-only `model` is rejected too — the narrow, unambiguous mistake
    // this process CAN catch on its own (unlike a merely-unrecognized-but-otherwise-well-formed id,
    // which it can't: every id is forwarded verbatim through the gateway, with no local registry to
    // validate a real one against — see the RPC handler's own doc comment). The live model must be
    // left exactly as `gpt-4o` set it above, not reset to empty.
    writeln!(stdin, "{}", json!({ "type": "set_model", "model": "   " })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_model");
    assert_eq!(frames.last().unwrap()["success"], false);
    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    assert_eq!(
        frames.last().unwrap()["data"]["model"],
        "gpt-4o",
        "a rejected empty model must not disturb the live model: {frames:#?}"
    );

    // Set then clear the thinking budget.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_thinking", "budget": 4096 })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_thinking");
    assert_eq!(frames.last().unwrap()["success"], true);
    assert_eq!(frames.last().unwrap()["data"]["thinking"], 4096);

    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_thinking", "budget": Value::Null })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_thinking");
    assert_eq!(frames.last().unwrap()["success"], true);
    assert!(frames.last().unwrap()["data"]["thinking"].is_null());

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_cycle_model_advances_and_wraps() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "get_available_models" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_available_models");
    let models: Vec<String> = frames.last().unwrap()["data"]["models"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_string())
        .collect();

    // Pin the model to the list's *last* entry first, so cycling from a known position is
    // unambiguous (the server's own default id, "claude-test", isn't in `available_models()` at all,
    // and would otherwise wrap to index 0 on the very first cycle regardless of direction).
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_model", "model": models[models.len() - 1] })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "set_model");

    // Cycling past the last entry wraps to the first...
    writeln!(stdin, "{}", json!({ "type": "cycle_model" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "cycle_model");
    assert_eq!(frames.last().unwrap()["data"]["model"], models[0]);

    // ...and cycling again advances normally to the second.
    writeln!(stdin, "{}", json!({ "type": "cycle_model" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "cycle_model");
    let last = frames.last().unwrap();
    assert_eq!(last["data"]["model"], models[1]);
    assert_eq!(
        last["data"]["scoped"], false,
        "no --models flag was given, so cycling the full list must report scoped: false"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_cycle_model_scoped_by_the_models_flag_cycles_only_the_scope_but_lists_the_full_catalog() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = Command::new(bin)
        .args([
            "serve",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session-file",
            &session_file,
            "--models",
            "claude-opus-4-8,gpt-4o",
        ])
        .env("HOME", ISOLATED_HOME)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // `get_available_models` is deliberately NOT scoped by `--models` — that flag only narrows
    // `cycle_model`'s own candidate list (asserted below). A client's model *picker* still needs to
    // see the full catalog to offer a "show everything" view, same as pi's own `/model` selector can
    // Tab out of its scope-defaulted view.
    writeln!(stdin, "{}", json!({ "type": "get_available_models" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_available_models");
    let all_models: Vec<String> = frames.last().unwrap()["data"]["models"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(
        all_models,
        beyond_ai_agent::serve::available_models()
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>(),
        "get_available_models must report the full catalog, unaffected by --models scoping"
    );

    // Pin to the scoped list's first entry so cycling from a known position is unambiguous.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_model", "model": "claude-opus-4-8" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "set_model");

    writeln!(stdin, "{}", json!({ "type": "cycle_model" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "cycle_model");
    let last = frames.last().unwrap();
    assert_eq!(last["data"]["model"], "gpt-4o");
    assert_eq!(last["data"]["scoped"], true);

    // Cycling again must wrap back to the scoped list's *first* entry, not fall through to whatever
    // comes third in the full unscoped list.
    writeln!(stdin, "{}", json!({ "type": "cycle_model" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "cycle_model");
    assert_eq!(frames.last().unwrap()["data"]["model"], "claude-opus-4-8");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_models_flag_expands_a_glob_against_the_known_catalog() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = Command::new(bin)
        .args([
            "serve",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session-file",
            &session_file,
            "--models",
            "claude-*",
        ])
        .env("HOME", ISOLATED_HOME)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // Pin to the first claude-* entry in catalog order so cycling from a known position is
    // unambiguous, then walk the whole scoped cycle and confirm it's exactly the claude-* subset of
    // `available_models()`, in catalog order, wrapping back to the start — never a gpt-*/o*-series id.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_model", "model": "claude-opus-4-8" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "set_model");

    let expected: Vec<&str> = beyond_ai_agent::serve::available_models()
        .iter()
        .copied()
        .filter(|id| id.starts_with("claude-"))
        .collect();
    assert!(
        expected.len() >= 2,
        "fixture assumption: the known catalog has multiple claude-* ids"
    );

    let cycle_order: Vec<&str> = expected[1..]
        .iter()
        .chain(expected[..1].iter())
        .copied()
        .collect();
    for want in &cycle_order {
        writeln!(stdin, "{}", json!({ "type": "cycle_model" })).unwrap();
        stdin.flush().unwrap();
        let frames = read_until_response(&mut stdout, "cycle_model");
        let last = frames.last().unwrap();
        assert_eq!(last["data"]["model"], *want);
        assert_eq!(last["data"]["scoped"], true);
    }

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_models_flag_pattern_level_suffix_pins_that_models_thinking_level_on_cycle() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = Command::new(bin)
        .args([
            "serve",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session-file",
            &session_file,
            "--models",
            "claude-opus-4-8:high,gpt-4o",
        ])
        .env("HOME", ISOLATED_HOME)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // Pin to the unpinned scoped entry first so cycling onto the pinned one is unambiguous.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_model", "model": "gpt-4o" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "set_model");

    // Cycling onto "claude-opus-4-8:high" must land with reasoning_effort "high", the level its
    // `--models` pattern pinned — not whatever level happened to be active before.
    writeln!(stdin, "{}", json!({ "type": "cycle_model" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "cycle_model");
    let last = frames.last().unwrap();
    assert_eq!(last["data"]["model"], "claude-opus-4-8");
    assert_eq!(last["data"]["reasoning_effort"], "high");

    // Cycling onto the unpinned "gpt-4o" must not carry the pinned level along — it keeps whatever
    // was already active (still "high" here, since nothing unpins it), matching pi's own
    // "unpinned entries inherit the session's current level" rule rather than resetting to off.
    writeln!(stdin, "{}", json!({ "type": "cycle_model" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "cycle_model");
    let last = frames.last().unwrap();
    assert_eq!(last["data"]["model"], "gpt-4o");
    assert_eq!(last["data"]["reasoning_effort"], "high");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_models_flag_rejects_an_invalid_thinking_level_suffix_as_part_of_the_literal_id() {
    // "claude-opus-4-8:bogus" has no valid thinking-level suffix, so the whole string is kept as a
    // literal id (pi's own scope-mode fallback) rather than silently dropping ":bogus" — since our
    // catalog match is glob-only, a literal that doesn't equal any catalog entry is still forwarded
    // verbatim (see `available_models`'s "hint, not an allowlist" contract), so cycling reaches it too.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = Command::new(bin)
        .args([
            "serve",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session-file",
            &session_file,
            "--models",
            "claude-opus-4-8:bogus",
        ])
        .env("HOME", ISOLATED_HOME)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "cycle_model" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "cycle_model");
    assert_eq!(
        frames.last().unwrap()["data"]["model"],
        "claude-opus-4-8:bogus"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_cycle_thinking_level_advances_through_the_ladder_and_wraps() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // Starting Off, each cycle advances one rung on the portable Off/Minimal/Low/Medium/High/XHigh
    // ladder, wrapping back to Off. `claude-test` (this test's model) resolves to `ThinkingShape::Budget`
    // with a 32_000 max_output, so `reasoning_effort` stays null throughout (that dialect arm never
    // reads it) and `thinking` is the level's derived, clamped budget.
    let expected = [
        ("minimal", json!(1024)),
        ("low", json!(2048)),
        ("medium", json!(8192)),
        ("high", json!(24000)),
        ("xhigh", json!(31999)),
        ("off", Value::Null),
    ];
    for (level, thinking) in expected {
        writeln!(stdin, "{}", json!({ "type": "cycle_thinking_level" })).unwrap();
        stdin.flush().unwrap();
        let frames = read_until_response(&mut stdout, "cycle_thinking_level");
        let data = &frames.last().unwrap()["data"];
        assert_eq!(frames.last().unwrap()["success"], true, "got: {data:#?}");
        assert_eq!(data["level"], level, "got: {data:#?}");
        assert_eq!(data["thinking"], thinking, "got: {data:#?}");
        assert!(data["reasoning_effort"].is_null(), "got: {data:#?}");
    }

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_set_reasoning_effort_sets_the_portable_level_directly() {
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
        json!({ "type": "set_reasoning_effort", "effort": "high" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_reasoning_effort");
    let data = &frames.last().unwrap()["data"];
    assert_eq!(frames.last().unwrap()["success"], true, "got: {data:#?}");
    assert_eq!(data["level"], "high");
    assert_eq!(data["thinking"], 24000);

    // A subsequent cycle starts from "high", advancing to "xhigh" — proving `set_reasoning_effort`
    // really did move `current_level`, not just a one-off override.
    writeln!(stdin, "{}", json!({ "type": "cycle_thinking_level" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "cycle_thinking_level");
    assert_eq!(frames.last().unwrap()["data"]["level"], "xhigh");

    // `null` clears it back to off.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_reasoning_effort", "effort": Value::Null })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_reasoning_effort");
    let data = &frames.last().unwrap()["data"];
    assert_eq!(data["level"], "off");
    assert!(data["thinking"].is_null());

    // An unrecognized effort name is rejected, not silently ignored.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_reasoning_effort", "effort": "extreme" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_reasoning_effort");
    assert_eq!(frames.last().unwrap()["success"], false);

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_set_reasoning_effort_wins_over_a_stale_set_thinking_override() {
    // `set_thinking` sets an explicit raw-budget override; `set_reasoning_effort` (like
    // `cycle_thinking_level`) must clear it so the newly-requested level takes visible effect
    // immediately rather than being masked by the leftover override.
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
        json!({ "type": "set_thinking", "budget": 4096 })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "set_thinking");

    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_reasoning_effort", "effort": "low" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_reasoning_effort");
    let data = &frames.last().unwrap()["data"];
    assert_eq!(
        data["thinking"], 2048,
        "the level's own budget must win, not the stale 4096 override: {data:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_starts_clamped_not_off_for_a_model_that_cannot_disable_reasoning() {
    // The CRITICAL bug this closes: a session on a model with a reasoning mechanism it can't
    // explicitly disable (`gpt-5-codex`: `reasoning_disableable == false`) must never start at the
    // stored level `Off` — that would silently omit the `reasoning` field from every request and let
    // the provider apply its own hidden default effort, with the operator believing reasoning is off.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd_with_model(bin, &base, &session_file, "gpt-5-codex")
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    let data = &frames.last().unwrap()["data"];
    assert_eq!(
        data["thinking_level"], "minimal",
        "gpt-5-codex's floor is minimal; a fresh session with no --reasoning-effort must start \
         there, not at off: got {data:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_set_model_reclamps_off_when_switching_onto_a_non_disableable_model() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    // `claude-test` is disable-capable, so the session starts at a legal "off".
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    assert_eq!(frames.last().unwrap()["data"]["thinking_level"], "off");

    // Switching to a model that can't disable reasoning must bump the still-stored "off" up to that
    // model's own floor, not silently carry an illegal level across the switch.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_model", "model": "gpt-5-codex" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_model");
    let data = &frames.last().unwrap()["data"];
    assert_eq!(data["reasoning_effort"], "minimal", "got: {data:#?}");

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    assert_eq!(
        frames.last().unwrap()["data"]["thinking_level"],
        "minimal",
        "get_state must reflect the re-clamped level too, not just the set_model response"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_cycle_thinking_level_never_gets_stuck_for_a_model_without_xhigh_or_off() {
    // Regression guard for a bug the naive fix would have introduced: a plain `level.next()` then
    // re-clamp bounces forever between `high` and a re-clamped `xhigh` for a model lacking xhigh
    // support, since `xhigh` always clamps back down to the very `high` it started from. This model
    // additionally can't reach `off` at all (`reasoning_disableable == false`), so the full available
    // ladder is exactly minimal/low/medium/high, wrapping.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd_with_model(bin, &base, &session_file, "gpt-5-codex")
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // Starts clamped at "minimal" (see the dedicated startup-clamp test above).
    let expected = ["low", "medium", "high", "minimal", "low", "medium", "high"];
    for level in expected {
        writeln!(stdin, "{}", json!({ "type": "cycle_thinking_level" })).unwrap();
        stdin.flush().unwrap();
        let frames = read_until_response(&mut stdout, "cycle_thinking_level");
        let data = &frames.last().unwrap()["data"];
        assert_eq!(data["level"], level, "got: {data:#?}");
    }

    drop(stdin);
    child.wait().unwrap();
}
