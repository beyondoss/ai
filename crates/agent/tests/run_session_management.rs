//! `run` e2e: `--session`/`--continue`/`--name` flag semantics for a one-shot `run`: persistence, resume, naming, validation.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::{Command, Stdio};

use common::{run_cmd, spawn_model_server, turn_text};
use serde_json::json;

#[test]
fn run_binary_session_flag_persists_and_resumes_across_invocations() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl");

    let (base1, bodies1) = spawn_model_server(vec![turn_text("first answer")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output1 = run_cmd(bin)
        .args([
            "run",
            "remember the marker: xyzzy-42",
            "--gateway-url",
            &base1,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session",
            session_file.to_str().unwrap(),
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");
    assert!(
        output1.status.success(),
        "first run failed.\nstderr: {}",
        String::from_utf8_lossy(&output1.stderr)
    );
    drop(bodies1);
    assert!(session_file.exists(), "the session file must be created");

    let (base2, bodies2) = spawn_model_server(vec![turn_text("second answer")]);
    let output2 = run_cmd(bin)
        .args([
            "run",
            "what was the marker?",
            "--gateway-url",
            &base2,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session",
            session_file.to_str().unwrap(),
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");
    assert!(
        output2.status.success(),
        "second run failed.\nstderr: {}",
        String::from_utf8_lossy(&output2.stderr)
    );

    let bodies2 = bodies2.lock().unwrap();
    assert!(
        bodies2[0].contains("xyzzy-42"),
        "the second run must see the first run's history: {}",
        bodies2[0]
    );
    assert!(bodies2[0].contains("first answer"));
    assert!(bodies2[0].contains("what was the marker?"));
}

#[test]
fn run_binary_initializes_an_existing_empty_session_file_instead_of_hard_failing() {
    // Track L8: `--session <path>` pointing at a zero-byte file (e.g. `touch`'d ahead of time by a
    // caller that wants the path to already exist) must initialize it in place, not hard-fail with
    // "session file has no header."
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl");
    std::fs::write(&session_file, b"").unwrap(); // pre-create as an empty file
    assert_eq!(std::fs::metadata(&session_file).unwrap().len(), 0);

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
            "--session",
            session_file.to_str().unwrap(),
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
    assert!(
        std::fs::metadata(&session_file).unwrap().len() > 0,
        "the session file must actually have a real header now"
    );
}

#[test]
fn run_binary_reports_a_clear_error_for_an_invalid_session_file_and_preserves_its_content() {
    // pi-parity fix (C-M6): pi's `session-file-invalid.test.ts` — pointing `--session` at a non-empty
    // file that isn't a valid session (no header) must fail with exit 1, a clear error naming the
    // path (not a raw `Debug`-formatted `io::Error` with no path at all — see the fix at this call
    // site in `run_task`), and must leave the original file's bytes completely untouched.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("not-a-session.log");
    let original_content = "{\"type\":\"event\",\"data\":\"not a session\"}\n";
    std::fs::write(&session_file, original_content).unwrap();

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "hi",
            "--gateway-url",
            "http://127.0.0.1:1",
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session",
            session_file.to_str().unwrap(),
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        !output.status.success(),
        "an invalid session file must fail the run"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(session_file.to_str().unwrap()),
        "stderr must name the offending path: {stderr}"
    );
    assert!(
        stderr.contains("not a valid session"),
        "stderr must carry a clear, human-readable message: {stderr}"
    );
    assert!(
        !stderr.contains("Custom {") && !stderr.contains("kind:") && !stderr.contains("at "),
        "stderr must not leak the raw internal error representation or stack frames: {stderr}"
    );
    assert_eq!(
        std::fs::read_to_string(&session_file).unwrap(),
        original_content,
        "the original (invalid) file content must be byte-for-byte unchanged"
    );
}

#[test]
fn run_binary_name_sets_the_session_title() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl");
    let (base, _bodies) = spawn_model_server(vec![turn_text("hi")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "hello",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session",
            session_file.to_str().unwrap(),
            "--name",
            "my-named-run",
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
    let on_disk = std::fs::read_to_string(&session_file).unwrap();
    assert!(
        on_disk.contains("my-named-run"),
        "the session's title must be persisted: {on_disk}"
    );
}

#[test]
fn run_binary_short_n_flag_is_an_alias_for_name() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl");
    let (base, _bodies) = spawn_model_server(vec![turn_text("hi")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "hello",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session",
            session_file.to_str().unwrap(),
            "-n",
            "my-short-named-run",
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
    let on_disk = std::fs::read_to_string(&session_file).unwrap();
    assert!(
        on_disk.contains("my-short-named-run"),
        "the session's title must be persisted via the -n short alias: {on_disk}"
    );
}

#[test]
fn run_binary_name_is_a_noop_when_resuming_an_existing_session() {
    // pi-parity (C-M5): pi's `startup-session-name.test.ts:111-121` renames an existing `--session
    // <path>` unconditionally, even before runtime model validation fails. This project deliberately
    // diverges (see the fresh-only check documented at the `--name` application site in `run_task`):
    // a startup flag shouldn't silently rename an already-running session just because it was passed
    // again on a later invocation. This proves the documented no-op actually holds: a second `--session
    // <path>` run against a session that already has messages/a title must leave the original title
    // alone, even though `--name` was passed again with a different value.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl");
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let (base1, _bodies1) = spawn_model_server(vec![turn_text("first")]);
    let output1 = run_cmd(bin)
        .args([
            "run",
            "hello",
            "--gateway-url",
            &base1,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session",
            session_file.to_str().unwrap(),
            "--name",
            "original-name",
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");
    assert!(
        output1.status.success(),
        "first run failed.\nstderr: {}",
        String::from_utf8_lossy(&output1.stderr)
    );
    let after_first = std::fs::read_to_string(&session_file).unwrap();
    assert!(after_first.contains("original-name"));

    let (base2, _bodies2) = spawn_model_server(vec![turn_text("second")]);
    let output2 = run_cmd(bin)
        .args([
            "run",
            "hello again",
            "--gateway-url",
            &base2,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session",
            session_file.to_str().unwrap(),
            "--name",
            "renamed-attempt",
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");
    assert!(
        output2.status.success(),
        "second run failed.\nstderr: {}",
        String::from_utf8_lossy(&output2.stderr)
    );

    let after_second = std::fs::read_to_string(&session_file).unwrap();
    assert!(
        after_second.contains("original-name"),
        "the original title must survive a resumed --session run with a different --name: \
         {after_second}"
    );
    assert!(
        !after_second.contains("renamed-attempt"),
        "resuming an existing --session must not rename it, matching this project's documented \
         divergence from pi's unconditional rename: {after_second}"
    );
}

#[test]
fn run_binary_rejects_a_whitespace_only_name_and_persists_nothing() {
    // pi: startup-session-name.test.ts — a whitespace-only `--name` is rejected with a clear error
    // ("--name requires a non-empty value") rather than silently producing a blank/meaningless title,
    // and must fail before any session file is created at all.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl");
    let (base, _bodies) = spawn_model_server(vec![turn_text("hi")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "hello",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session",
            session_file.to_str().unwrap(),
            "--name",
            "   ",
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        !output.status.success(),
        "a whitespace-only --name must fail the run"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("--name requires a non-empty value"),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !session_file.exists(),
        "nothing must be persisted when --name is rejected"
    );
}

#[test]
fn run_binary_rejects_a_path_traversal_session_id_and_persists_nothing() {
    // pi-parity fix (M4): `--session-id` is embedded directly into a filename with no other
    // sanitization — must be rejected with a clear error before touching any files, matching pi's own
    // `assertValidSessionId`.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl");
    let (base, _bodies) = spawn_model_server(vec![turn_text("hi")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "hello",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session",
            session_file.to_str().unwrap(),
            "--session-id",
            "../../../tmp/pwned/evil",
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        !output.status.success(),
        "a path-traversal --session-id must fail the run"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("--session-id"),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !session_file.exists(),
        "nothing must be persisted when --session-id is rejected"
    );
    assert!(
        !dir.path().join("tmp").exists(),
        "no directory must be created outside the intended session file's own parent"
    );
}

#[test]
fn run_session_flag_restores_the_sessions_own_model_when_no_explicit_model_flag_given() {
    // pi-parity fix: reopening `--session <path>` used to always build the `Agent` on the CLI-resolved
    // model (the flag if given, else `DEFAULT_MODEL`) rather than the session's own persisted model —
    // reattaching to a session last driven on a different model silently switched it with no warning.
    // The session file's header alone (a `SessionMeta` with no messages yet) is enough to exercise this
    // without needing to fully replay prior history.
    let dir = tempfile::tempdir().unwrap();
    let session_path = dir.path().join("s.jsonl");
    let header = json!({
        "type": "session",
        "id": "restored-model-test",
        "created_at": 1,
        "cwd": dir.path().to_string_lossy(),
        "model": "claude-test-restored",
    });
    std::fs::write(&session_path, format!("{header}\n")).unwrap();

    let (base, bodies) = spawn_model_server(vec![turn_text("ok")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "hi",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            // Deliberately no `--model` flag — the session's own recorded model must win over
            // `DEFAULT_MODEL`.
            "--session",
            &session_path.to_string_lossy(),
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
        bodies[0].contains(r#""model":"claude-test-restored""#),
        "the request must use the session's own persisted model, not DEFAULT_MODEL: {}",
        bodies[0]
    );
}

#[test]
fn run_session_flag_prefers_an_explicit_model_flag_over_the_sessions_own_model() {
    // The other half of the fix above: an operator who *does* pass `--model` is deliberately
    // overriding, and that must still win over the session's own persisted model.
    let dir = tempfile::tempdir().unwrap();
    let session_path = dir.path().join("s.jsonl");
    let header = json!({
        "type": "session",
        "id": "explicit-override-test",
        "created_at": 1,
        "cwd": dir.path().to_string_lossy(),
        "model": "claude-test-restored",
    });
    std::fs::write(&session_path, format!("{header}\n")).unwrap();

    let (base, bodies) = spawn_model_server(vec![turn_text("ok")]);

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
            "claude-test-explicit",
            "--session",
            &session_path.to_string_lossy(),
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
        bodies[0].contains(r#""model":"claude-test-explicit""#),
        "an explicit --model must win over the session's own persisted model: {}",
        bodies[0]
    );
}

#[test]
fn run_continue_with_name_titles_a_genuinely_fresh_session() {
    // pi-parity fix (H6): `--continue`'s fresh-session path (`resume_or_create` finding no existing
    // session for this cwd) minted its own `SessionMeta` internally, bypassing the `fresh_meta` closure
    // every other startup path used to apply `--name` — a brand-new session created this way was
    // silently left untitled no matter what `--name` was passed, with no error.
    let home_dir = tempfile::tempdir().unwrap();
    let project_dir = tempfile::tempdir().unwrap();

    let (base, _bodies) = spawn_model_server(vec![turn_text("ok")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = Command::new(bin)
        .env("HOME", home_dir.path())
        .args([
            "run",
            "hi",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--continue",
            "--name",
            "My Session",
        ])
        .current_dir(project_dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "binary failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    // `--continue` persists under `$HOME/.claude/sessions/<encoded-cwd>/` — one project subdirectory,
    // one session file inside it, for this brand-new project.
    let project_subdir = std::fs::read_dir(home_dir.path().join(".claude/sessions"))
        .expect("sessions root must exist")
        .next()
        .expect("one project subdirectory")
        .unwrap()
        .path();
    let session_file = std::fs::read_dir(&project_subdir)
        .expect("project subdirectory must exist")
        .next()
        .expect("one session file")
        .unwrap()
        .path();
    let content = std::fs::read_to_string(&session_file).unwrap();
    assert!(
        content.contains("My Session"),
        "--name must reach the session --continue creates when no prior session for this cwd \
         exists: {content}"
    );
}

#[test]
fn run_continue_with_name_does_not_rename_an_existing_session_for_this_cwd() {
    // pi-parity (C-M5): the complement of `run_continue_with_name_titles_a_genuinely_fresh_session`
    // above — once a session already exists for this cwd, a later `--continue --name` must leave it
    // alone (the same fresh-only check documented at the `--name` application site in `run_task`;
    // pi itself renames unconditionally here, see `startup-session-name.test.ts:111-121` — a
    // deliberate, already-decided divergence, not an oversight).
    let home_dir = tempfile::tempdir().unwrap();
    let project_dir = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let (base1, _bodies1) = spawn_model_server(vec![turn_text("first")]);
    let output1 = Command::new(bin)
        .env("HOME", home_dir.path())
        .args([
            "run",
            "hi",
            "--gateway-url",
            &base1,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--continue",
            "--name",
            "Original Name",
        ])
        .current_dir(project_dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");
    assert!(
        output1.status.success(),
        "first run failed.\nstderr: {}",
        String::from_utf8_lossy(&output1.stderr)
    );

    let (base2, _bodies2) = spawn_model_server(vec![turn_text("second")]);
    let output2 = Command::new(bin)
        .env("HOME", home_dir.path())
        .args([
            "run",
            "hi again",
            "--gateway-url",
            &base2,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--continue",
            "--name",
            "Renamed Attempt",
        ])
        .current_dir(project_dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");
    assert!(
        output2.status.success(),
        "second run failed.\nstderr: {}",
        String::from_utf8_lossy(&output2.stderr)
    );

    let project_subdir = std::fs::read_dir(home_dir.path().join(".claude/sessions"))
        .expect("sessions root must exist")
        .next()
        .expect("one project subdirectory")
        .unwrap()
        .path();
    let session_file = std::fs::read_dir(&project_subdir)
        .expect("project subdirectory must exist")
        .next()
        .expect("one session file")
        .unwrap()
        .path();
    let content = std::fs::read_to_string(&session_file).unwrap();
    assert!(
        content.contains("Original Name"),
        "the original title must survive a --continue run against the same cwd: {content}"
    );
    assert!(
        !content.contains("Renamed Attempt"),
        "--continue reopening an existing session for this cwd must not rename it: {content}"
    );
}

#[test]
fn run_session_flag_warns_when_the_sessions_recorded_cwd_no_longer_matches() {
    // pi-parity fix (M5): pi's `MissingSessionCwdError` surfaces a clear warning when a resumed
    // session's recorded `cwd` doesn't match reality; `run` previously had no equivalent at all —
    // silently proceeding with no signal to the operator that tools would run somewhere different
    // from where the session was originally created.
    let session_dir = tempfile::tempdir().unwrap();
    let actual_dir = tempfile::tempdir().unwrap();
    let session_path = session_dir.path().join("s.jsonl");
    let header = json!({
        "type": "session",
        "id": "stale-cwd-test",
        "created_at": 1,
        "cwd": "/definitely/does/not/exist/beyond-ai-agent-test-fixture",
        "model": "claude-test",
    });
    std::fs::write(&session_path, format!("{header}\n")).unwrap();

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
            "--session",
            &session_path.to_string_lossy(),
        ])
        .current_dir(actual_dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "a stale cwd must warn, not fail the run.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("warning") && stderr.contains("working directory"),
        "stderr: {stderr}"
    );
}
