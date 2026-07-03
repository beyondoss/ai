//! `serve` e2e: Process/session-file lifecycle: startup defaults, resume/reattach, crash recovery, jsonl framing, HTML export.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Command, Stdio};

use beyond_ai_agent::session_store::{SessionMeta, SessionRepo};
use common::{
    ISOLATED_HOME, message_ids, read_until_response, serve_cmd, serve_dir_cmd, spawn_model_server,
    turn_text, turn_tool_use,
};
use serde_json::{Value, json};

/// Neither `--session-file` nor `--session-dir` — exercises `Persistence::open`'s default directory
/// (`~/.claude/sessions/<encoded-cwd>/`) rather than in-memory-only.
fn serve_default_persistence_cmd(bin: &str, base: &str) -> Command {
    let mut c = Command::new(bin);
    c.args([
        "serve",
        "--gateway-url",
        base,
        "--key",
        "bai_v1.test",
        "--model",
        "claude-test",
    ])
    .env("HOME", ISOLATED_HOME)
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::null());
    c
}

#[test]
fn serve_export_html_writes_a_self_contained_transcript() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![turn_text("hello there")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "say hi" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    let output_path = dir.path().join("out.html").to_string_lossy().into_owned();
    writeln!(
        stdin,
        "{}",
        json!({ "type": "export_html", "output_path": output_path })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "export_html");
    let response = frames.last().unwrap();
    assert_eq!(response["success"], true, "got: {response:#?}");
    assert_eq!(response["data"]["path"], output_path);

    let html = std::fs::read_to_string(&output_path).unwrap();
    assert!(html.starts_with("<!DOCTYPE html>"));
    assert!(html.contains("say hi"));
    assert!(html.contains("hello there"));

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_export_html_includes_abandoned_branches_not_just_the_active_path() {
    // Track M19: an abandoned branch (created by rewinding via `switch_branch`) must still show up in
    // the export — the whole point being that the old flat `export_html` silently dropped anything not
    // on the active path.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) =
        spawn_model_server(vec![turn_text("first answer"), turn_text("second answer")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "first" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "second" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    // Rewind to the first turn's assistant reply (message index 1), abandoning the second turn
    // (indices 2, 3) without a summary, so its original text survives verbatim on disk.
    let ids = message_ids(&session_file);
    assert_eq!(ids.len(), 4, "expected 4 persisted messages: {ids:?}");
    writeln!(
        stdin,
        "{}",
        json!({ "type": "switch_branch", "target_id": ids[1], "summarize": false })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "switch_branch");

    let output_path = dir.path().join("out.html").to_string_lossy().into_owned();
    writeln!(
        stdin,
        "{}",
        json!({ "type": "export_html", "output_path": output_path })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "export_html");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");

    let html = std::fs::read_to_string(&output_path).unwrap();
    // The abandoned branch renders inline, as a collapsible <details> block positioned right after
    // the message it diverged from — not a separate flat "Other branches" section.
    let split = html
        .find("<details class=\"branch\">")
        .unwrap_or_else(|| panic!("the abandoned branch must get its own <details> block: {html}"));
    let (active_section, branches_section) = html.split_at(split);
    assert!(active_section.contains("first"), "the active path: {html}");
    assert!(
        active_section.contains("first answer"),
        "the active path: {html}"
    );
    assert!(
        !active_section.contains("second"),
        "abandoned by the rewind, must not be on the active path: {active_section}"
    );
    assert!(
        branches_section.contains("second answer"),
        "the abandoned branch's own content must still appear, in its section: {branches_section}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_defaults_to_home_claude_sessions_when_no_session_flag_given() {
    // Neither --session-file nor --session-dir: must default to a real, cwd-encoded directory under
    // HOME rather than silently running in-memory-only. HOME is overridden to a tempdir so this
    // neither sees nor pollutes the real developer's `~/.claude/sessions`.
    let home = tempfile::tempdir().unwrap();
    let project = tempfile::tempdir().unwrap();
    let (base, _bodies) = spawn_model_server(vec![turn_text("ok")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_default_persistence_cmd(bin, &base);
    cmd.current_dir(project.path()).env("HOME", home.path());
    let mut child = cmd.spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    let sessions_root = home.path().join(".claude/sessions");
    assert!(
        sessions_root.is_dir(),
        "expected a default sessions directory under HOME at {}",
        sessions_root.display()
    );
    // Exactly one project subdirectory (the encoded cwd), containing exactly one session file.
    let project_dirs: Vec<_> = std::fs::read_dir(&sessions_root)
        .unwrap()
        .filter_map(|e| e.ok())
        .collect();
    assert_eq!(
        project_dirs.len(),
        1,
        "expected one encoded-cwd subdirectory: {project_dirs:?}"
    );
    let session_files: Vec<_> = std::fs::read_dir(project_dirs[0].path())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "jsonl"))
        .collect();
    assert_eq!(
        session_files.len(),
        1,
        "expected one persisted session file: {session_files:?}"
    );
}

#[test]
fn serve_resumes_newest_session_matching_cwd_not_globally_newest() {
    // A --session-dir shared across two different project directories (the case the new default
    // avoids by cwd-encoding its own path, but still possible with an explicit shared directory):
    // reattaching from project A must resume A's own session, not B's more-recently-updated one.
    let session_dir_tmp = tempfile::tempdir().unwrap();
    let session_dir = session_dir_tmp.path().to_string_lossy().into_owned();
    let project_a = tempfile::tempdir().unwrap();
    let project_b = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    // Session A, from project_a.
    {
        let (base, _bodies) = spawn_model_server(vec![turn_text("answer from A")]);
        let mut cmd = serve_dir_cmd(bin, &base, &session_dir);
        cmd.current_dir(project_a.path());
        let mut child = cmd.spawn().unwrap();
        let mut stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        writeln!(
            stdin,
            "{}",
            json!({ "type": "prompt", "message": "hi from A" })
        )
        .unwrap();
        stdin.flush().unwrap();
        read_until_response(&mut stdout, "prompt");
        drop(stdin);
        child.wait().unwrap();
    }

    // Session B, from project_b — created and updated *after* A, so a globally-newest-first pick
    // would wrongly resume this one from project_a.
    {
        let (base, _bodies) = spawn_model_server(vec![turn_text("answer from B")]);
        let mut cmd = serve_dir_cmd(bin, &base, &session_dir);
        cmd.current_dir(project_b.path());
        let mut child = cmd.spawn().unwrap();
        let mut stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        writeln!(
            stdin,
            "{}",
            json!({ "type": "prompt", "message": "hi from B" })
        )
        .unwrap();
        stdin.flush().unwrap();
        read_until_response(&mut stdout, "prompt");
        drop(stdin);
        child.wait().unwrap();
    }

    // Reattach from project_a again — must resume A's transcript, not B's.
    {
        let (base, _bodies) = spawn_model_server(vec![]);
        let mut cmd = serve_dir_cmd(bin, &base, &session_dir);
        cmd.current_dir(project_a.path());
        let mut child = cmd.spawn().unwrap();
        let mut stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
        stdin.flush().unwrap();
        let frames = read_until_response(&mut stdout, "get_messages");
        let dump = frames.last().unwrap()["data"]["messages"].to_string();
        assert!(
            dump.contains("hi from A") && dump.contains("answer from A"),
            "expected to resume project_a's own session: {dump}"
        );
        assert!(
            !dump.contains("from B"),
            "must not resume project_b's newer-but-different-cwd session: {dump}"
        );
        drop(stdin);
        child.wait().unwrap();
    }
}

#[test]
fn serve_reattaches_through_a_symlinked_cwd_to_the_session_recorded_under_its_real_path() {
    // A project reached through a symlink one time and its real path another must resolve to the
    // same session (`session_store::canonical_cwd`), not silently fork into two.
    let session_dir_tmp = tempfile::tempdir().unwrap();
    let session_dir = session_dir_tmp.path().to_string_lossy().into_owned();
    let projects = tempfile::tempdir().unwrap();
    let real = projects.path().join("real-project");
    std::fs::create_dir(&real).unwrap();
    let link = projects.path().join("project-link");
    std::os::unix::fs::symlink(&real, &link).unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    // Start (and prompt) from the real path.
    {
        let (base, _bodies) = spawn_model_server(vec![turn_text("answer via real path")]);
        let mut cmd = serve_dir_cmd(bin, &base, &session_dir);
        cmd.current_dir(&real);
        let mut child = cmd.spawn().unwrap();
        let mut stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        writeln!(
            stdin,
            "{}",
            json!({ "type": "prompt", "message": "hi via real path" })
        )
        .unwrap();
        stdin.flush().unwrap();
        read_until_response(&mut stdout, "prompt");
        drop(stdin);
        child.wait().unwrap();
    }

    // Reattach from the symlinked path — must resume the same session, not mint a new one.
    {
        let (base, _bodies) = spawn_model_server(vec![]);
        let mut cmd = serve_dir_cmd(bin, &base, &session_dir);
        cmd.current_dir(&link);
        let mut child = cmd.spawn().unwrap();
        let mut stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
        stdin.flush().unwrap();
        let frames = read_until_response(&mut stdout, "get_messages");
        let dump = frames.last().unwrap()["data"]["messages"].to_string();
        assert!(
            dump.contains("hi via real path") && dump.contains("answer via real path"),
            "a symlinked cwd must reattach to the session recorded under its real path: {dump}"
        );
        drop(stdin);
        child.wait().unwrap();
    }
}

#[test]
fn serve_list_sessions_streams_progress_frames_correlated_to_the_request_id() {
    let dir = tempfile::tempdir().unwrap();
    let session_dir = dir.path().join("sessions").to_string_lossy().into_owned();

    // Pre-seed sessions directly on disk (no need to drive a `prompt` per session) so the scan
    // `list_sessions` performs has more than one file to report progress across.
    let repo = SessionRepo::open(&session_dir).unwrap();
    for i in 0..6 {
        repo.create(SessionMeta::new(format!("/w{i}"), "m"))
            .unwrap();
    }

    let (base, _bodies) = spawn_model_server(vec![]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_dir_cmd(bin, &base, &session_dir).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    // Drain the `ready` banner.
    let mut ready = String::new();
    stdout.read_line(&mut ready).unwrap();

    writeln!(
        stdin,
        "{}",
        json!({ "type": "list_sessions", "id": "req-1" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "list_sessions");

    let progress: Vec<&Value> = frames
        .iter()
        .filter(|f| f["type"] == "list_progress")
        .collect();
    assert!(
        !progress.is_empty(),
        "expected at least one list_progress frame: {frames:#?}"
    );
    for p in &progress {
        assert_eq!(p["command"], "list_sessions");
        assert_eq!(
            p["id"], "req-1",
            "progress must correlate to the request id"
        );
        assert!(p["scanned"].as_u64().unwrap() >= 1);
        assert!(p["total"].as_u64().unwrap() >= p["scanned"].as_u64().unwrap());
    }
    // The last progress frame observed must reach the full total. Since the scan is parallel, frames
    // may not arrive in strictly increasing `scanned` order, but the maximum reported must still be
    // the total, and the total must match the response's own session count — `serve`'s own startup
    // reattach mints one more session for its actual cwd (which matches none of the 6 seeded here),
    // so the total is 7, not 6.
    let max_scanned = progress
        .iter()
        .map(|p| p["scanned"].as_u64().unwrap())
        .max()
        .unwrap();
    let total = progress[0]["total"].as_u64().unwrap();
    assert!(
        total >= 6,
        "must cover at least the 6 pre-seeded sessions: {progress:#?}"
    );
    assert_eq!(
        max_scanned, total,
        "the last progress frame must reach 100%"
    );

    let response = frames.last().unwrap();
    assert_eq!(response["success"], true);
    assert_eq!(
        response["data"]["sessions"].as_array().unwrap().len() as u64,
        total,
        "progress total must match the number of sessions actually returned"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
#[cfg(unix)]
fn serve_exits_gracefully_on_sigterm_mid_run() {
    use std::time::{Duration, Instant};

    // A SIGTERM (what `systemctl restart`/`docker stop`/a pod eviction sends) mid-run must be
    // treated like `abort`/stdin-closing: cancel the in-flight turn, persist what's there, and exit
    // on its own — not Rust's default disposition of immediate termination with no destructors run,
    // which would orphan the sleeping child process and lose the turn's unpersisted messages.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir
        .path()
        .join("session.json")
        .to_string_lossy()
        .into_owned();

    let turn1 = turn_tool_use(
        "toolu_b",
        "bash",
        &json!({ "command": "sleep 30" }).to_string(),
    );
    let (base, _bodies) = spawn_model_server(vec![turn1]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let pid = child.id();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "run a long sleep" })
    )
    .unwrap();
    stdin.flush().unwrap();

    // Give the run time to reach the tool before signaling, so this exercises the mid-run
    // cancellation path (the harder one) rather than racing the idle-between-commands one.
    std::thread::sleep(Duration::from_millis(500));

    let status = Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .status()
        .unwrap();
    assert!(status.success(), "failed to send SIGTERM to serve");

    // Must exit on its own well under the 30s sleep the in-flight tool call was running — not need
    // a hard `child.kill()` to reap it.
    let deadline = Instant::now() + Duration::from_secs(10);
    let exit = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        assert!(
            Instant::now() < deadline,
            "serve did not exit within 10s of SIGTERM"
        );
        std::thread::sleep(Duration::from_millis(50));
    };
    assert!(
        exit.success(),
        "serve should exit cleanly on SIGTERM, got {exit:?}"
    );

    // The stdout writer flushes on shutdown; whatever is left in the pipe just needs to not panic
    // when drained, not match any particular frame.
    let mut trailing = String::new();
    let _ = stdout.read_to_string(&mut trailing);

    // What was persisted before the cancel (at least the user's turn) must be a valid, non-empty,
    // readable session file — not lost by the abrupt-looking exit.
    let contents = std::fs::read_to_string(&session_file).unwrap();
    assert!(
        !contents.trim().is_empty(),
        "nothing was persisted before SIGTERM shutdown"
    );
}

#[test]
fn serve_streams_events_and_reattaches() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("hello.txt"), "secret-marker-77\n").unwrap();
    let abs = dir.path().join("hello.txt").to_string_lossy().into_owned();
    let session_file = dir
        .path()
        .join("session.json")
        .to_string_lossy()
        .into_owned();

    // One prompt drives two model turns: read tool, then text.
    let turn1 = turn_tool_use("toolu_1", "read", &json!({ "path": abs }).to_string());
    let turn2 = turn_text("Read complete.");
    let (base, _bodies) = spawn_model_server(vec![turn1, turn2]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    // --- First session: prompt, observe streamed events, read transcript ---
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "read hello.txt" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");

    // A `ready` frame was emitted on startup.
    assert!(
        frames
            .iter()
            .any(|f| f.get("type").and_then(Value::as_str) == Some("ready"))
    );
    // Tool-call boundaries streamed as events.
    let events: Vec<&Value> = frames.iter().filter(|f| f["type"] == "event").collect();
    assert!(
        events
            .iter()
            .any(|e| e["event"]["kind"] == "tool_start" && e["event"]["name"] == "read"),
        "expected a tool_start event for `read`; frames: {frames:#?}"
    );
    assert!(
        events
            .iter()
            .any(|e| e["event"]["kind"] == "tool_end" && e["event"]["name"] == "read")
    );
    // Final response is a success.
    let resp = frames.last().unwrap();
    assert_eq!(resp["type"], "response");
    assert_eq!(resp["success"], true);

    // get_messages returns the transcript including the tool result and final text.
    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames2 = read_until_response(&mut stdout, "get_messages");
    let dump = frames2.last().unwrap()["data"]["messages"].to_string();
    assert!(
        dump.contains("secret-marker-77"),
        "transcript should hold the tool result: {dump}"
    );
    assert!(dump.contains("Read complete."));

    drop(stdin); // close stdin → server exits
    assert!(child.wait().unwrap().success());

    // --- Reattach: a fresh process over the same session file sees the prior transcript ---
    let mut child2 = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin2 = child2.stdin.take().unwrap();
    let mut stdout2 = BufReader::new(child2.stdout.take().unwrap());
    writeln!(stdin2, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin2.flush().unwrap();
    let frames3 = read_until_response(&mut stdout2, "get_messages");
    let dump3 = frames3.last().unwrap()["data"]["messages"].to_string();
    assert!(
        dump3.contains("secret-marker-77"),
        "reattached session must restore the transcript: {dump3}"
    );

    drop(stdin2);
    child2.wait().unwrap();
}

#[test]
fn serve_survives_a_hard_crash_mid_run_with_the_first_round_trip_already_durable() {
    use std::time::Duration;

    // A genuine crash (SIGKILL — no signal handler, no graceful drain, nothing like the SIGTERM path
    // above) partway through a *second* tool round-trip must still leave the *first* round-trip's
    // messages durable on disk: proof that incremental mid-run persistence (H-6), not the final
    // post-run persist or the graceful-shutdown path, is what saved them.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let turn1 = turn_tool_use(
        "toolu_1",
        "bash",
        &json!({ "command": "printf round-one-marker" }).to_string(),
    );
    let turn2 = turn_tool_use(
        "toolu_2",
        "bash",
        &json!({ "command": "sleep 5" }).to_string(),
    );
    let (base, _bodies) = spawn_model_server(vec![turn1, turn2, turn_text("done")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "go" })).unwrap();
    stdin.flush().unwrap();

    // The first round-trip (a fast `printf`) should complete and checkpoint well within this window;
    // the second turn's `sleep 5` is still running when we kill the process.
    std::thread::sleep(Duration::from_millis(800));
    child.kill().unwrap(); // SIGKILL — no destructors, no signal handler, an actual hard crash
    child.wait().unwrap();

    // Reattach with a fresh process and check what survived.
    let (base2, _bodies2) = spawn_model_server(vec![]);
    let mut child2 = serve_cmd(bin, &base2, &session_file).spawn().unwrap();
    let mut stdin2 = child2.stdin.take().unwrap();
    let mut stdout2 = BufReader::new(child2.stdout.take().unwrap());
    writeln!(stdin2, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin2.flush().unwrap();
    let frames = read_until_response(&mut stdout2, "get_messages");
    drop(stdin2);
    child2.wait().unwrap();

    let dump = frames.last().unwrap()["data"]["messages"].to_string();
    assert!(
        dump.contains("round-one-marker"),
        "the first tool round-trip must have been checkpointed before the crash: {dump}"
    );
    assert!(
        !dump.contains("toolu_2"),
        "the second (interrupted) round-trip must not appear as a completed pair: {dump}"
    );
}

#[test]
fn serve_reports_success_when_a_failed_checkpoint_is_superseded_by_a_successful_final_persist() {
    // LOW pi-parity gap (fixed): `persist_error` used to be set the moment any mid-run checkpoint
    // failed and never cleared, so a checkpoint hiccup early in a run made the terminal `prompt`
    // response report failure even when the run's actual final state was later persisted just fine.
    // Root-only environments can't exercise this (permission bits don't restrict root), so skip there.
    if std::env::var("USER").as_deref() == Ok("root") {
        return;
    }

    use std::os::unix::fs::PermissionsExt;
    use std::time::Duration;

    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let turn1 = turn_tool_use(
        "toolu_1",
        "bash",
        &json!({ "command": "printf first-round-marker" }).to_string(),
    );
    let turn2 = turn_tool_use(
        "toolu_2",
        "bash",
        &json!({ "command": "sleep 1" }).to_string(),
    );
    let (base, _bodies) = spawn_model_server(vec![turn1, turn2, turn_text("done")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // Consume the ready frame — by now `Persistence::open` has already created the file with normal
    // permissions, so this doesn't race the file's own creation.
    let mut ready = String::new();
    stdout.read_line(&mut ready).unwrap();

    // Make the session file read-only *before* sending the prompt, so the first round-trip's
    // mid-run checkpoint (fired right after `toolu_1` completes) fails to append to it.
    std::fs::set_permissions(&session_file, std::fs::Permissions::from_mode(0o444)).unwrap();

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "go" })).unwrap();
    stdin.flush().unwrap();

    // The first round-trip (a fast `printf`) completes and its checkpoint attempt fails well within
    // this window; the second turn's `sleep 1` is still running when permissions are restored.
    std::thread::sleep(Duration::from_millis(500));
    std::fs::set_permissions(&session_file, std::fs::Permissions::from_mode(0o644)).unwrap();

    // The run ends once `sleep 1` finishes and the model's concluding "done" turn is emitted; the
    // unconditional final persist right after that must now succeed against the writable-again file.
    let frames = read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    let response = frames.last().unwrap();
    assert_eq!(
        response["success"], true,
        "the run's true final state was persisted successfully — an earlier, superseded checkpoint \
         failure must not surface as a false failure: {response:#?}"
    );
    assert!(
        response.get("error").is_none() || response["error"].is_null(),
        "got: {response:#?}"
    );

    // And the final persist genuinely did land on disk, not just "no error was reported."
    let on_disk = std::fs::read_to_string(&session_file).unwrap();
    assert!(
        on_disk.contains("first-round-marker"),
        "the final persist must have actually written the transcript: {on_disk}"
    );
}

#[test]
fn serve_stdout_stays_valid_json_even_when_a_load_warning_fires() {
    // A prior bug (output-guard gap): the tracing subscriber defaulted to stdout, the same stream
    // `serve`'s NDJSON protocol writes to. A `tracing::warn!` on a live path (here: `session_store.rs`
    // skipping an unparseable line while loading) would interleave raw log text into the protocol
    // stream, breaking any line-based JSON parser reading it. Plant a session file with exactly that
    // kind of corrupt line, run with `RUST_LOG=warn` so the warning is guaranteed to fire, and assert
    // every single stdout line still parses as JSON — while independently confirming (via captured
    // stderr) that the warning really did fire, so the assertion isn't vacuously true.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl");
    let header = json!({
        "type": "session",
        "id": "log-purity-session",
        "created_at": 1,
        "cwd": dir.path().to_string_lossy(),
        "model": "claude-test",
    });
    std::fs::write(
        &session_file,
        format!("{header}\nthis line is not valid JSON at all\n"),
    )
    .unwrap();

    let (base, _bodies) = spawn_model_server(vec![]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file.to_string_lossy())
        .env("RUST_LOG", "warn")
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    let mut stderr = BufReader::new(child.stderr.take().unwrap());

    // Deliberately NOT `read_until_response` — it silently skips a line that fails to parse as JSON,
    // which would make this test vacuously pass even with the bug reintroduced. Every line must parse.
    let mut line = String::new();
    let mut lines_seen = 0;
    loop {
        line.clear();
        if stdout.read_line(&mut line).unwrap() == 0 {
            panic!("stdout closed before a ready frame arrived");
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(trimmed)
            .unwrap_or_else(|e| panic!("stdout line was not valid JSON ({e}): {trimmed:?}"));
        lines_seen += 1;
        if v.get("type").and_then(Value::as_str) == Some("ready") {
            break;
        }
    }
    assert!(lines_seen >= 1);

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    loop {
        line.clear();
        if stdout.read_line(&mut line).unwrap() == 0 {
            panic!("stdout closed before the get_state response arrived");
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(trimmed)
            .unwrap_or_else(|e| panic!("stdout line was not valid JSON ({e}): {trimmed:?}"));
        let done = v.get("type").and_then(Value::as_str) == Some("response")
            && v.get("command").and_then(Value::as_str) == Some("get_state");
        if done {
            break;
        }
    }

    drop(stdin);
    child.wait().unwrap();

    let mut stderr_text = String::new();
    stderr.read_to_string(&mut stderr_text).unwrap();
    assert!(
        stderr_text.contains("skipping unparseable session entry line"),
        "the load warning must actually have fired (on stderr, not stdout) for this test to mean \
         anything: stderr was {stderr_text:?}"
    );
}

#[test]
fn new_session_reports_failure_and_leaves_the_old_session_active_when_persist_fails() {
    // `Persistence::new_session` used to unconditionally return a fresh, empty in-memory `Session`
    // even when the on-disk reset actually failed — reporting RPC success on a session that was never
    // really created, while `SessionStore`'s own persisted-message-count bookkeeping (correctly)
    // stayed untouched. A subsequent real `persist` would then see the small "new" in-memory message
    // count against that stale-large persisted count and silently no-op via `append_new`'s own dedup
    // guard, discarding every message of the "new" session with no error at all. The fix: report
    // failure over RPC, and leave the *old* session active rather than switching to a phantom one.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, _bodies) = spawn_model_server(vec![turn_text("hello-reply")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hello" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let before = read_until_response(&mut stdout, "get_messages");
    let before_dump = before.last().unwrap()["data"]["messages"].to_string();
    assert!(
        before_dump.contains("hello-reply"),
        "the first prompt must have actually persisted: {before_dump}"
    );

    // Force the *next* write to `session_file` to fail: replace it with a directory of the same name.
    // `new_session`'s single-file-mode reset (`SessionStore::rewrite`) writes a temp file then renames
    // it onto this path — a rename can never land a file onto an existing directory, so this reliably
    // reproduces a real write failure without needing root or a special filesystem.
    std::fs::remove_file(&session_file).unwrap();
    std::fs::create_dir(&session_file).unwrap();

    writeln!(stdin, "{}", json!({ "type": "new_session" })).unwrap();
    stdin.flush().unwrap();
    let new_session_response = read_until_response(&mut stdout, "new_session");
    let resp = new_session_response.last().unwrap();
    assert_eq!(
        resp["success"], false,
        "a failed on-disk reset must be reported as failure, not silent success: {resp:#?}"
    );
    assert!(
        resp.get("error").is_some(),
        "failure must carry an error message: {resp:#?}"
    );

    // The old session must still be the active one — not silently swapped for an empty one that was
    // never actually persisted.
    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let after = read_until_response(&mut stdout, "get_messages");
    let after_dump = after.last().unwrap()["data"]["messages"].to_string();
    assert_eq!(
        after_dump, before_dump,
        "a failed new_session must leave the previous session's transcript untouched: {after_dump}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_unknown_command_type_reports_a_clear_failure_frame() {
    // pi: suite/regressions/5868-rpc-unknown-command-id.test.ts — an unrecognized `type` must still
    // produce a well-formed response frame (echoing both `id` and the unrecognized `command`), not a
    // dropped connection or a malformed/missing reply.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![turn_text("hi")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "foobar", "id": "test" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "foobar");
    let resp = frames.last().unwrap();
    assert_eq!(resp["id"], "test");
    assert_eq!(resp["command"], "foobar");
    assert_eq!(resp["success"], false, "got: {resp:#?}");
    assert!(resp.get("error").is_some(), "got: {resp:#?}");

    // The connection must still be alive afterward — a genuinely recognized command works normally.
    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    let ok = read_until_response(&mut stdout, "prompt");
    assert_eq!(ok.last().unwrap()["success"], true);

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_jsonl_framing_preserves_u2028_and_u2029_inside_a_payload() {
    // F-L1 (pi: rpc-jsonl.test.ts "splits on LF only and preserves U+2028/U+2029 inside payloads"):
    // U+2028 (LINE SEPARATOR) and U+2029 (PARAGRAPH SEPARATOR) are not ASCII `\n`, so a byte-oriented
    // NDJSON reader that only splits on `0x0A` must pass them straight through as ordinary payload
    // bytes rather than treating them as line breaks. `tokio::io::AsyncBufReadExt::lines()` (this
    // server's stdin reader — see `serve()`) only ever splits on `0x0A`/strips a trailing `\r`, so this
    // is expected to already be correct; proven here rather than assumed.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    let title = "a\u{2028}b\u{2029}c";
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_session_name", "title": title })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_session_name");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    // No embedded `\r`/`\n`, so `sanitize_title` leaves it untouched — the separators round-trip
    // byte-for-byte, proving neither the client-side serializer nor this server's line reader treated
    // them as a line break mid-payload.
    assert_eq!(
        frames.last().unwrap()["data"]["title"],
        title,
        "got: {frames:#?}"
    );

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let state = read_until_response(&mut stdout, "get_state");
    assert_eq!(state.last().unwrap()["data"]["title"], title, "{state:#?}");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_jsonl_framing_handles_crlf_delimited_commands() {
    // F-L1 (pi: rpc-jsonl.test.ts "handles CRLF-delimited input"): a client on Windows, or one that
    // simply writes `\r\n`, must have each command recognized as its own line — `tokio::io::
    // AsyncBufReadExt::lines()` strips a trailing `\r` after splitting on `\n` (see `lines.rs`'s
    // `poll_next_line`), so two `\r\n`-terminated commands in the same write must parse as two clean
    // JSON lines, not one line with a stray `\r` corrupting the trailing `}` or the two commands
    // merging into one.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    let cmd_a = json!({ "id": "a", "type": "set_session_name", "title": "first" }).to_string();
    let cmd_b = json!({ "id": "b", "type": "set_session_name", "title": "second" }).to_string();
    write!(stdin, "{cmd_a}\r\n{cmd_b}\r\n").unwrap();
    stdin.flush().unwrap();

    let frames_a = read_until_response(&mut stdout, "set_session_name");
    let resp_a = frames_a
        .iter()
        .find(|f| f["type"] == "response" && f["id"] == "a")
        .unwrap_or_else(|| panic!("expected a response to id \"a\": {frames_a:#?}"));
    assert_eq!(resp_a["success"], true, "{resp_a:#?}");
    assert_eq!(resp_a["data"]["title"], "first", "{resp_a:#?}");

    let frames_b = read_until_response(&mut stdout, "set_session_name");
    let resp_b = frames_b
        .iter()
        .find(|f| f["type"] == "response" && f["id"] == "b")
        .unwrap_or_else(|| panic!("expected a response to id \"b\": {frames_b:#?}"));
    assert_eq!(resp_b["success"], true, "{resp_b:#?}");
    assert_eq!(resp_b["data"]["title"], "second", "{resp_b:#?}");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_jsonl_framing_handles_a_final_command_with_no_trailing_newline() {
    // F-L1 (pi: rpc-jsonl.test.ts "emits a final line without trailing LF"): a client that closes
    // stdin right after its last command, with no trailing `\n`, must still have that command
    // processed — `AsyncBufReadExt::lines()` yields the trailing partial line once at EOF (`n == 0`
    // with a non-empty buffer still returns `Some(buf)`; only a truly empty read returns `None`), so
    // this server's `lines.next_line()` loop sees and processes it before observing stdin's close.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // No trailing `\n` at all — `write!`, not `writeln!` — then drop `stdin` to close the pipe.
    write!(
        stdin,
        "{}",
        json!({ "id": "last", "type": "set_session_name", "title": "no newline" })
    )
    .unwrap();
    stdin.flush().unwrap();
    drop(stdin);

    let frames = read_until_response(&mut stdout, "set_session_name");
    let resp = frames.last().unwrap();
    assert_eq!(resp["id"], "last", "{resp:#?}");
    assert_eq!(resp["success"], true, "{resp:#?}");
    assert_eq!(resp["data"]["title"], "no newline", "{resp:#?}");

    child.wait().unwrap();
}

#[test]
fn serve_name_flag_sets_the_initial_session_title() {
    // Companion to `run`'s own `--name` e2e coverage (`run_e2e.rs`) — `serve`'s version only applies
    // to a genuinely fresh session (see `ServeConfig::name`'s doc comment), which a brand-new
    // `--session-file` always is.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![turn_text("hi")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file)
        .args(["--name", "my-serve-session"])
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let state = read_until_response(&mut stdout, "get_state");
    assert_eq!(
        state.last().unwrap()["data"]["title"],
        "my-serve-session",
        "got: {state:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
}
