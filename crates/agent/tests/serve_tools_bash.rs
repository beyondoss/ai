//! `serve` e2e: Tool registration/exclusion and the host `bash` RPC command.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufReader, Read, Write};
use std::process::Stdio;

use common::{read_until_response, serve_cmd, spawn_model_server, turn_text, turn_tool_use};
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
fn serve_bash_exclude_from_context_keeps_the_session_untouched() {
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
        json!({
            "type": "bash",
            "command": "printf should-not-reach-context",
            "exclude_from_context": true,
        })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "bash");
    let resp = frames.last().unwrap();
    assert_eq!(resp["data"]["result"], "should-not-reach-context");

    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let dump = frames.last().unwrap()["data"]["messages"].to_string();
    assert!(
        !dump.contains("should-not-reach-context"),
        "exclude_from_context: true must keep the session untouched: {dump}"
    );

    drop(stdin);
    child.wait().unwrap();
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
