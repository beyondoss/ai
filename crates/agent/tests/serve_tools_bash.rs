//! `serve` e2e: Tool registration/exclusion and the host `bash` RPC command.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufReader, Read, Write};
use std::process::Stdio;

use common::{read_until_response, serve_cmd, spawn_model_server, sse, turn_text, turn_tool_use};
use serde_json::json;

#[test]
fn serve_streams_tool_progress_from_a_running_bash() {
    // The full streaming chain, deterministically (mock model + real bash, no network): the model
    // calls `bash` with a command that emits output over time; the run must surface those chunks as
    // `tool_progress` event frames *before* the tool's result.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let cmd = "printf 'chunk-a\\n'; sleep 0.15; printf 'chunk-b\\n'";
    let (base, _bodies) = spawn_model_server(vec![
        turn_tool_use("toolu_b", "bash", &json!({ "command": cmd }).to_string()),
        turn_text("done"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "run it" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    // Collect tool_progress chunks in arrival order, and prove they precede the tool's end.
    let kinds: Vec<&str> = frames
        .iter()
        .filter(|f| f["type"] == "event")
        .filter_map(|f| f["event"]["kind"].as_str())
        .collect();
    let progress_chunks: String = frames
        .iter()
        .filter(|f| f["type"] == "event" && f["event"]["kind"] == "tool_progress")
        .filter_map(|f| f["event"]["snapshot"].as_str())
        .collect();

    assert!(
        kinds.contains(&"tool_progress"),
        "a running bash must stream tool_progress frames: {kinds:?}"
    );
    assert!(
        progress_chunks.contains("chunk-a") && progress_chunks.contains("chunk-b"),
        "streamed chunks should carry the live output, got: {progress_chunks:?}"
    );
    let first_progress = kinds.iter().position(|k| *k == "tool_progress").unwrap();
    let tool_end = kinds.iter().position(|k| *k == "tool_end").unwrap();
    assert!(
        first_progress < tool_end,
        "progress must arrive before tool_end: {kinds:?}"
    );
}

#[test]
fn serve_exclude_tools_removes_a_tool_from_the_advertised_set() {
    // `--exclude-tools bash` must remove it from both the tool definitions sent to the model and the
    // default system prompt's tool list — an excluded tool should be invisible to the model, not just
    // rejected after the fact if it tries to call it.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, bodies) = spawn_model_server(vec![turn_text("ok")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.args(["--exclude-tools", "bash"]);
    let mut child = cmd.spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    let recorded = bodies.lock().unwrap();
    assert!(
        !recorded[0].contains("\"bash\""),
        "excluded tool must not appear in the request body (tool defs or system prompt): {:?}",
        recorded[0]
    );
    assert!(
        recorded[0].contains("\"read\""),
        "other tools must remain advertised: {:?}",
        recorded[0]
    );
}

#[test]
fn serve_no_tools_sends_no_tools_field_at_all() {
    // `--no-tools` must leave the agent with an empty registry — the Anthropic dialect omits the
    // `tools` key entirely from the wire body when there are none to advertise (see
    // `dialect::anthropic::build_body`), so its absence is the precise signal to check for.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, bodies) = spawn_model_server(vec![turn_text("ok")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.args(["--no-tools"]);
    let mut child = cmd.spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    let recorded = bodies.lock().unwrap();
    assert!(
        !recorded[0].contains("\"tools\":"),
        "no-tools mode must omit the tools field entirely: {:?}",
        recorded[0]
    );
}

#[test]
fn serve_bash_runs_a_host_command_independent_of_the_model() {
    // A `bash` RPC command must run without ever touching the model — no scripted response is queued,
    // so if `serve` mistakenly routed it through a model turn the mock server would have nothing to
    // answer with and the test would hang/fail on `read_until_response`.
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
        json!({ "type": "bash", "command": "printf host-bash-ran" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "bash");
    drop(stdin);
    child.wait().unwrap();

    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], true, "frames: {frames:#?}");
    assert_eq!(resp["data"]["result"], "host-bash-ran");
    assert_eq!(resp["data"]["is_error"], false);

    // tool_start/tool_end events fire exactly like a model-invoked bash call, for a client that
    // renders both cases through the same code path.
    let kinds: Vec<&str> = frames
        .iter()
        .filter(|f| f["type"] == "event")
        .filter_map(|f| f["event"]["kind"].as_str())
        .collect();
    assert!(kinds.contains(&"tool_start"), "{kinds:?}");
    assert!(kinds.contains(&"tool_end"), "{kinds:?}");
}

#[test]
fn serve_bash_records_its_result_into_session_context_by_default() {
    // pi-parity fix (M13): the host `bash` RPC command never touched `session` at all — the calling
    // client saw the result, but the model never would on a later turn. Matches pi's own
    // `recordBashResult`, which records by default.
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
        json!({ "type": "bash", "command": "printf host-bash-context-marker" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "bash");

    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let dump = frames.last().unwrap()["data"]["messages"].to_string();
    assert!(
        dump.contains("host-bash-context-marker"),
        "the host bash command's own output must reach session context: {dump}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_bash_exclude_from_context_is_recorded_but_hidden_from_the_model() {
    // Fix 9 (pi-parity gap, HIGH PRIORITY — silent data loss): `exclude_from_context: true` used to
    // skip *recording* the command/output entirely — it never showed up in `session.messages`,
    // persisted storage, or an `export`, unlike pi's own two-layer split: `recordBashResult`
    // (`agent-session.ts`) unconditionally records into history; `excludeFromContext` is consulted only
    // later, by the separate `convertToLlm` transform (`messages.ts`) that builds what's actually sent
    // to the model. Proves all four: present in `session.messages` (`get_messages`), present in the
    // persisted `.jsonl`, present in an HTML export — but absent from the next turn's actual wire
    // request.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let marker = "should-not-reach-the-model-but-must-be-recorded";
    let (base, bodies) = spawn_model_server(vec![turn_text("ack")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({
            "type": "bash",
            "command": format!("printf {marker}"),
            "exclude_from_context": true,
        })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "bash");
    let resp = frames.last().unwrap();
    assert_eq!(resp["data"]["result"], marker);

    // 1. Present in session.messages, via get_messages.
    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let dump = frames.last().unwrap()["data"]["messages"].to_string();
    assert!(
        dump.contains(marker),
        "exclude_from_context must NOT skip recording into session.messages: {dump}"
    );

    // 2. Absent from the next real turn's actual wire request to the model.
    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "go" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    {
        let bodies = bodies.lock().unwrap();
        assert!(
            !bodies[0].contains(marker),
            "exclude_from_context must keep the command/output out of what's sent to the model: {}",
            bodies[0]
        );
    }

    drop(stdin);
    child.wait().unwrap();

    // 3. Present in persisted storage on disk.
    let persisted = std::fs::read_to_string(&session_file).unwrap();
    assert!(
        persisted.contains(marker),
        "exclude_from_context must still persist to disk: {persisted}"
    );

    // 4. Present in export.rs's rendered output.
    let export_path = dir.path().join("transcript.html");
    let output = std::process::Command::new(bin)
        .args(["export", &session_file, export_path.to_str().unwrap()])
        .env("HOME", dir.path())
        .output()
        .expect("spawn binary");
    assert!(
        output.status.success(),
        "export subcommand failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let html = std::fs::read_to_string(&export_path).unwrap();
    assert!(
        html.contains(marker),
        "exclude_from_context must still render in an html export: {html}"
    );
}

#[test]
fn serve_bash_is_rejected_when_the_tool_is_excluded() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.args(["--exclude-tools", "bash"]);
    let mut child = cmd.spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "bash", "command": "echo should-not-run" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "bash");
    drop(stdin);
    child.wait().unwrap();

    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], false, "frames: {frames:#?}");
    assert!(
        resp["error"]
            .as_str()
            .unwrap_or_default()
            .contains("not registered"),
        "frames: {frames:#?}"
    );
}

#[test]
fn serve_bash_rpc_command_is_blocked_by_deny_bash_pattern() {
    // Track L23 (pi-parity fix): the host `{"type":"bash"}` RPC command used to call
    // `tool.run_streaming` directly, never consulting the `ToolPolicy` `build_agent` installs for the
    // model's own tool calls — only `--exclude-tools bash` (proven by
    // `serve_bash_is_rejected_when_the_tool_is_excluded`, above) actually gated it. Proves
    // `--deny-bash-pattern` now blocks this RPC path too, the same way it already blocks a
    // model-invoked `bash` call.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let marker = dir.path().join("should-not-exist.txt");
    let (base, _bodies) = spawn_model_server(vec![]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.args(["--deny-bash-pattern", "touch"]);
    let mut child = cmd.spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "bash", "command": format!("touch {}", marker.to_str().unwrap()) })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "bash");
    drop(stdin);
    child.wait().unwrap();

    assert!(
        !marker.exists(),
        "--deny-bash-pattern must block the host `bash` RPC command before it ever runs"
    );
    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], false, "frames: {frames:#?}");
    assert!(
        resp["error"]
            .as_str()
            .unwrap_or_default()
            .contains("blocked by policy"),
        "frames: {frames:#?}"
    );
}

#[test]
#[cfg(unix)]
fn serve_bash_shell_path_overrides_the_auto_resolved_shell() {
    if !std::path::Path::new("/bin/sh").exists() {
        return; // no alternate shell on this host to prove the override actually took effect
    }
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.args(["--bash-shell-path", "/bin/sh"]);
    let mut child = cmd.spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // `$BASH_VERSION` is set only by bash itself; `/bin/sh` on this host is dash, which leaves it
    // unset — the POSIX `${VAR:-default}` expansion below works identically under both, so the
    // *value* it prints is the only thing that can differ.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "bash", "command": "echo ${BASH_VERSION:-no-bash}" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "bash");
    drop(stdin);
    child.wait().unwrap();

    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], true, "frames: {frames:#?}");
    assert_eq!(resp["data"]["result"], "no-bash\n", "frames: {frames:#?}");
}

#[test]
fn serve_bash_command_prefix_flag_runs_before_the_given_command() {
    // Pi-parity fix: `Bash::with_command_prefix` (pi's own `shellCommandPrefix` setting) was fully
    // built and tested at the tool level but had zero call sites reachable from either binary.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.args(["--bash-command-prefix", "export PREFIX_VAR=from-prefix"]);
    let mut child = cmd.spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "bash", "command": "echo $PREFIX_VAR" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "bash");
    drop(stdin);
    child.wait().unwrap();

    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], true, "frames: {frames:#?}");
    assert_eq!(
        resp["data"]["result"], "from-prefix\n",
        "the prefix must run before the command, in the same shell invocation: {frames:#?}"
    );
}

#[test]
fn serve_sequential_tools_flag_is_accepted_and_both_calls_still_run() {
    // Pi-parity fix: `agent_core::Agent` had no way to force fully-sequential tool dispatch at all —
    // see `agent_core::agent::tests::with_sequential_tools_forces_one_group_in_flight_at_a_time` for the
    // mechanism itself (proven there with a non-exclusive counting tool, since `bash` is already
    // `conservative_exclusive` and so always serializes regardless of this flag). This is the `serve`
    // wiring smoke test: `--sequential-tools` parses, reaches `build_agent`'s
    // `Agent::with_sequential_tools`, and a turn batching two independent `bash` calls still dispatches
    // both correctly instead of the flag silently breaking multi-call turns.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let marker_a = dir.path().join("a.txt");
    let marker_b = dir.path().join("b.txt");

    let two_bash_calls = sse(&[
        json!({ "type": "message_start", "message": { "usage": { "input_tokens": 10, "output_tokens": 1 } } }),
        json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "tool_use", "id": "toolu_a", "name": "bash", "input": {} } }),
        json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "input_json_delta", "partial_json": json!({ "command": format!("touch {}", marker_a.to_str().unwrap()) }).to_string() } }),
        json!({ "type": "content_block_stop", "index": 0 }),
        json!({ "type": "content_block_start", "index": 1, "content_block": { "type": "tool_use", "id": "toolu_b", "name": "bash", "input": {} } }),
        json!({ "type": "content_block_delta", "index": 1, "delta": { "type": "input_json_delta", "partial_json": json!({ "command": format!("touch {}", marker_b.to_str().unwrap()) }).to_string() } }),
        json!({ "type": "content_block_stop", "index": 1 }),
        json!({ "type": "message_delta", "delta": { "stop_reason": "tool_use" }, "usage": { "output_tokens": 8 } }),
        json!({ "type": "message_stop" }),
    ]);
    let (base, _bodies) = spawn_model_server(vec![two_bash_calls, turn_text("done")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.args(["--sequential-tools"]);
    let mut child = cmd.spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "run two commands" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], true, "frames: {frames:#?}");
    assert!(marker_a.exists(), "the first bash call must still run");
    assert!(marker_b.exists(), "the second bash call must still run");
}

#[test]
fn serve_fails_fast_when_bash_shell_path_does_not_exist() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.args(["--bash-shell-path", "/no/such/shell-binary"]);
    cmd.stderr(Stdio::piped());
    let mut child = cmd.spawn().unwrap();
    drop(child.stdin.take()); // the process must exit before ever trying to read a command line
    let mut stderr = String::new();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut stderr)
        .unwrap();
    let status = child.wait().unwrap();
    assert!(!status.success(), "stderr: {stderr}");
    assert!(stderr.contains("--bash-shell-path"), "stderr: {stderr}");
}

#[test]
fn serve_abort_bash_cancels_a_running_host_command() {
    use std::time::{Duration, Instant};

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
        json!({ "type": "bash", "command": "sleep 30" })
    )
    .unwrap();
    stdin.flush().unwrap();
    // Give the command a moment to actually start, then abort it — a real 30s sleep would fail the
    // test on timeout if cancellation didn't work.
    std::thread::sleep(Duration::from_millis(200));
    writeln!(stdin, "{}", json!({ "type": "abort_bash" })).unwrap();
    stdin.flush().unwrap();

    let start = Instant::now();
    let frames = read_until_response(&mut stdout, "bash");
    assert!(
        start.elapsed() < Duration::from_secs(10),
        "abort_bash should cancel promptly, not wait out the full sleep"
    );
    drop(stdin);
    child.wait().unwrap();

    assert!(
        frames
            .iter()
            .any(|f| f["command"] == "abort_bash" && f["success"] == true),
        "abort_bash should be acknowledged: {frames:#?}"
    );
    let resp = frames.last().unwrap();
    assert_eq!(
        resp["data"]["is_error"], true,
        "a cancelled command must be reported as an error result: {frames:#?}"
    );
}

#[test]
fn serve_abort_bash_returns_the_partial_output_streamed_before_cancellation() {
    // Pi-parity: pi's `bash-executor.ts` returns whatever output already streamed when a command is
    // cancelled (`BashResult { output, cancelled: true, .. }`), not a bare "cancelled" placeholder.
    // Previously `abort_bash`'s result discarded every `ToolProgress` snapshot that had already been
    // forwarded to the client, replacing it with the literal string "cancelled".
    use std::time::{Duration, Instant};

    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    let cmd = "printf 'partial-output-line\\n'; sleep 30";
    writeln!(stdin, "{}", json!({ "type": "bash", "command": cmd })).unwrap();
    stdin.flush().unwrap();
    // Give the printed output time to cross the `UPDATE_THROTTLE` (100ms) into at least one streamed
    // `tool_progress` snapshot before cancelling.
    std::thread::sleep(Duration::from_millis(300));
    writeln!(stdin, "{}", json!({ "type": "abort_bash" })).unwrap();
    stdin.flush().unwrap();

    let start = Instant::now();
    let frames = read_until_response(&mut stdout, "bash");
    assert!(
        start.elapsed() < Duration::from_secs(10),
        "abort_bash should cancel promptly, not wait out the full sleep"
    );
    drop(stdin);
    child.wait().unwrap();

    assert!(
        frames
            .iter()
            .any(|f| f["type"] == "event" && f["event"]["kind"] == "tool_progress"),
        "the command must have streamed at least one progress snapshot before cancellation: {frames:#?}"
    );
    let resp = frames.last().unwrap();
    assert_eq!(
        resp["data"]["is_error"], true,
        "a cancelled command must be reported as an error result: {frames:#?}"
    );
    let result_text = resp["data"]["result"].as_str().unwrap();
    assert!(
        result_text.contains("partial-output-line"),
        "cancellation must preserve output that already streamed, not discard it: {result_text:?}"
    );
    assert!(
        result_text.to_lowercase().contains("cancel"),
        "cancellation result should still say it was cancelled: {result_text:?}"
    );
    // Fix 7 (pi-parity gap): pi's own structured `BashResult{cancelled, exitCode, ...}` alongside the
    // text, not flattened into it as the only signal.
    assert_eq!(
        resp["data"]["cancelled"], true,
        "the terminal response must carry a structured cancelled field, not just embed it in the text: {frames:#?}"
    );
    assert!(
        resp["data"]["exit_code"].is_null(),
        "a cancelled run has no exit code: {frames:#?}"
    );
}

#[test]
fn serve_bash_terminal_response_carries_structured_exit_code_and_truncation_fields() {
    // Fix 7 (pi-parity gap): a non-zero exit and a truncated/full-output-path result must reach the
    // terminal RPC response as their own structured fields, not only ever as a status line embedded in
    // `result`'s text (exit code) or on an interim `tool_progress` event a polling-only client never
    // sees (truncation/full_output_path).
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
        json!({ "type": "bash", "command": "exit 7" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "bash");
    let resp = frames.last().unwrap();
    assert_eq!(resp["data"]["is_error"], true, "got: {resp:#?}");
    assert_eq!(resp["data"]["exit_code"], 7, "got: {resp:#?}");
    assert_eq!(resp["data"]["cancelled"], false, "got: {resp:#?}");
    assert_eq!(resp["data"]["truncated"], false, "got: {resp:#?}");
    assert!(resp["data"]["full_output_path"].is_null(), "got: {resp:#?}");

    writeln!(
        stdin,
        "{}",
        json!({ "type": "bash", "command": "yes | head -c 200000" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "bash");
    let resp = frames.last().unwrap();
    assert_eq!(resp["data"]["is_error"], false, "got: {resp:#?}");
    assert_eq!(resp["data"]["exit_code"], 0, "got: {resp:#?}");
    assert_eq!(
        resp["data"]["truncated"], true,
        "200KB of output must trip truncation: {resp:#?}"
    );
    assert!(
        resp["data"]["full_output_path"].as_str().is_some(),
        "a truncated result must report where the full output was saved: {resp:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_deny_tool_flag_blocks_a_model_invoked_call_end_to_end() {
    // Pi-parity Critical fix: `agent_core::AgentHooks`/`with_hooks` was fully built and tested but had
    // zero call sites outside its own unit test — every real `serve` process built its `Agent` with
    // the no-op `NoHooks`. Proves the real compiled binary blocks a denied tool over the actual NDJSON
    // protocol, not just in isolated unit tests.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let marker = dir.path().join("should-not-exist.txt");

    let (base, _bodies) = spawn_model_server(vec![
        turn_tool_use(
            "toolu_d",
            "bash",
            &json!({ "command": format!("touch {}", marker.to_str().unwrap()) }).to_string(),
        ),
        turn_text("done"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file)
        .args(["--deny-tool", "bash"])
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "run it" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    assert!(
        !marker.exists(),
        "--deny-tool bash must block the call before it ever runs"
    );
    let tool_end = frames
        .iter()
        .find(|f| f["type"] == "event" && f["event"]["kind"] == "tool_end")
        .expect("a tool_end event must still fire for a blocked call");
    assert!(
        tool_end["event"]["result"]
            .as_str()
            .unwrap()
            .contains("denied by policy"),
        "the blocked call's result must explain it was policy-denied: {tool_end:#?}"
    );
    assert_eq!(tool_end["event"]["is_error"], true);
}

#[test]
fn serve_deny_path_flag_blocks_a_model_invoked_write_end_to_end() {
    // Track L22: `ToolPolicy` previously had no way to gate `write`/`edit` by their `path` argument —
    // only whole-tool-name (`--deny-tool`) or `bash`-substring (`--deny-bash-pattern`) denial. Proves
    // the real compiled `serve` binary blocks a `write` whose path matches `--deny-path`'s glob over
    // the actual NDJSON protocol, the same way `serve_deny_tool_flag_blocks_a_model_invoked_call_end_to_end`
    // proves it for `--deny-tool`.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let secret = dir.path().join("secrets.env");

    let (base, _bodies) = spawn_model_server(vec![
        turn_tool_use(
            "toolu_d",
            "write",
            &json!({ "path": secret.to_str().unwrap(), "content": "TOKEN=abc" }).to_string(),
        ),
        turn_text("done"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file)
        .args(["--deny-path", "*.env"])
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "write the secret" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    assert!(
        !secret.exists(),
        "--deny-path '*.env' must block the write before it ever runs"
    );
    let tool_end = frames
        .iter()
        .find(|f| f["type"] == "event" && f["event"]["kind"] == "tool_end")
        .expect("a tool_end event must still fire for a blocked call");
    assert!(
        tool_end["event"]["result"]
            .as_str()
            .unwrap()
            .contains("denied by policy"),
        "the blocked call's result must explain it was policy-denied: {tool_end:#?}"
    );
    assert_eq!(tool_end["event"]["is_error"], true);
}
