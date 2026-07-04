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
fn run_binary_persists_and_resumes_a_session_by_default_with_no_session_flags_given() {
    // pi-parity fix: a plain `run` with none of `--fork`/`--session`/`--continue` previously stayed
    // in-memory-only — pi's own default (every mode, including one-shot print-mode) is a persisted,
    // disk-backed session, matching `serve`'s own default repo-mode persistence. No `--continue` and no
    // `--session` here at all: the second invocation must still pick up the first's history from the
    // same per-cwd default repo `--continue` itself resolves against.
    let home_dir = tempfile::tempdir().unwrap();
    let project_dir = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let (base1, bodies1) = spawn_model_server(vec![turn_text("first answer")]);
    let output1 = Command::new(bin)
        .env("HOME", home_dir.path())
        .args([
            "run",
            "remember the marker: default-persist-99",
            "--gateway-url",
            &base1,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
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
    drop(bodies1);
    assert!(
        home_dir.path().join(".claude/sessions").is_dir(),
        "a plain no-flag run must persist under the default per-cwd sessions repo"
    );

    let (base2, bodies2) = spawn_model_server(vec![turn_text("second answer")]);
    let output2 = Command::new(bin)
        .env("HOME", home_dir.path())
        .args([
            "run",
            "what was the marker?",
            "--gateway-url",
            &base2,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
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

    let bodies2 = bodies2.lock().unwrap();
    assert!(
        bodies2[0].contains("default-persist-99"),
        "the second no-flag run must see the first no-flag run's history: {}",
        bodies2[0]
    );
}

#[test]
fn run_binary_no_session_persistence_flag_opts_out_of_the_default_persistence() {
    // The escape hatch for the fix above: `--no-session-persistence` restores the old ephemeral,
    // in-memory-only behavior — no file on disk at all, and a second invocation has no history to see.
    let home_dir = tempfile::tempdir().unwrap();
    let project_dir = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let (base1, bodies1) = spawn_model_server(vec![turn_text("first answer")]);
    let output1 = Command::new(bin)
        .env("HOME", home_dir.path())
        .args([
            "run",
            "remember the marker: should-not-persist-77",
            "--gateway-url",
            &base1,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--no-session-persistence",
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
    drop(bodies1);
    assert!(
        !home_dir.path().join(".claude/sessions").exists(),
        "--no-session-persistence must not create the default sessions repo at all"
    );

    let (base2, bodies2) = spawn_model_server(vec![turn_text("second answer")]);
    let output2 = Command::new(bin)
        .env("HOME", home_dir.path())
        .args([
            "run",
            "what was the marker?",
            "--gateway-url",
            &base2,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--no-session-persistence",
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

    let bodies2 = bodies2.lock().unwrap();
    assert!(
        !bodies2[0].contains("should-not-persist-77"),
        "with --no-session-persistence the second run must not see the first run's history: {}",
        bodies2[0]
    );
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

/// Reads a session `.jsonl` file's header line and returns its `id` field.
fn session_id_of(path: &std::path::Path) -> String {
    let content = std::fs::read_to_string(path).unwrap();
    let header: serde_json::Value = serde_json::from_str(content.lines().next().unwrap()).unwrap();
    header["id"].as_str().unwrap().to_string()
}

#[test]
fn run_fork_by_path_copies_a_session_file_into_a_new_session_and_continues_from_it() {
    // Pi-parity fix: `--fork <path>` (pi's own cross-project `--fork`) previously didn't exist at all —
    // `run` had no fork surface of any kind. A direct path to a `.jsonl` file must resolve regardless of
    // which project it belongs to, copy its transcript into a brand-new session under the *current*
    // project, and continue the run from there — leaving the original source file untouched.
    let source_dir = tempfile::tempdir().unwrap();
    let source_path = source_dir.path().join("source.jsonl");

    let (base1, _bodies1) = spawn_model_server(vec![turn_text("first answer")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output1 = run_cmd(bin)
        .args([
            "run",
            "remember the marker: fork-by-path-99",
            "--gateway-url",
            &base1,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session",
            source_path.to_str().unwrap(),
        ])
        .current_dir(source_dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");
    assert!(
        output1.status.success(),
        "source run failed.\nstderr: {}",
        String::from_utf8_lossy(&output1.stderr)
    );
    let source_content_before = std::fs::read_to_string(&source_path).unwrap();

    let target_home = tempfile::tempdir().unwrap();
    let target_project = tempfile::tempdir().unwrap();
    let (base2, bodies2) = spawn_model_server(vec![turn_text("second answer")]);
    let output2 = Command::new(bin)
        .env("HOME", target_home.path())
        .args([
            "run",
            "what was the marker?",
            "--gateway-url",
            &base2,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--fork",
            source_path.to_str().unwrap(),
        ])
        .current_dir(target_project.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");
    assert!(
        output2.status.success(),
        "forked run failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output2.stdout),
        String::from_utf8_lossy(&output2.stderr)
    );

    let bodies2 = bodies2.lock().unwrap();
    assert!(
        bodies2[0].contains("fork-by-path-99"),
        "the forked run must see the source session's history: {}",
        bodies2[0]
    );

    // The fork lands under the *current* project's own session directory, not the source's.
    let project_subdir = std::fs::read_dir(target_home.path().join(".claude/sessions"))
        .expect("sessions root must exist")
        .next()
        .expect("one project subdirectory")
        .unwrap()
        .path();
    let forked_path = std::fs::read_dir(&project_subdir)
        .expect("project subdirectory must exist")
        .next()
        .expect("one forked session file")
        .unwrap()
        .path();
    let forked_content: String = std::fs::read_to_string(&forked_path).unwrap();
    let header: serde_json::Value =
        serde_json::from_str(forked_content.lines().next().unwrap()).unwrap();
    assert_eq!(
        header["parent"].as_str().unwrap(),
        session_id_of(&source_path),
        "the forked session must record the source session as its parent"
    );
    assert_eq!(
        header["cwd"].as_str().unwrap(),
        target_project
            .path()
            .canonicalize()
            .unwrap()
            .to_str()
            .unwrap(),
        "the forked session's cwd must be the current project, not the source's"
    );

    // The original source file is untouched — forking copies, it doesn't move or mutate.
    assert_eq!(
        std::fs::read_to_string(&source_path).unwrap(),
        source_content_before,
        "the source session must be left exactly as it was"
    );
}

#[test]
fn run_fork_by_id_finds_a_session_in_a_different_projects_own_directory() {
    // The cross-project half of `--fork <path|id>`: an id not found in the current project's own
    // session directory must fall back to searching every other project's directory under
    // `~/.claude/sessions/` — matching pi's own `SessionManager.listAll`/`resolveSessionPath` fallback.
    let home_dir = tempfile::tempdir().unwrap();
    let project_a = tempfile::tempdir().unwrap();
    let project_b = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let (base1, _bodies1) = spawn_model_server(vec![turn_text("first answer")]);
    let output1 = Command::new(bin)
        .env("HOME", home_dir.path())
        .args([
            "run",
            "remember the marker: cross-project-77",
            "--gateway-url",
            &base1,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--continue",
        ])
        .current_dir(project_a.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");
    assert!(
        output1.status.success(),
        "project A run failed.\nstderr: {}",
        String::from_utf8_lossy(&output1.stderr)
    );

    let project_a_subdir = std::fs::read_dir(home_dir.path().join(".claude/sessions"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let source_path = std::fs::read_dir(&project_a_subdir)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let source_id = session_id_of(&source_path);

    let (base2, bodies2) = spawn_model_server(vec![turn_text("second answer")]);
    let output2 = Command::new(bin)
        .env("HOME", home_dir.path())
        .args([
            "run",
            "what was the marker?",
            "--gateway-url",
            &base2,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--fork",
            &source_id,
        ])
        .current_dir(project_b.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");
    assert!(
        output2.status.success(),
        "forked run in project B failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output2.stdout),
        String::from_utf8_lossy(&output2.stderr)
    );

    let bodies2 = bodies2.lock().unwrap();
    assert!(
        bodies2[0].contains("cross-project-77"),
        "forking by id across projects must carry the source transcript over: {}",
        bodies2[0]
    );

    // The fork must land under project B's own subdirectory, distinct from project A's.
    let sessions_root = home_dir.path().join(".claude/sessions");
    let subdirs: Vec<_> = std::fs::read_dir(&sessions_root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    assert_eq!(
        subdirs.len(),
        2,
        "expected one subdirectory per project, got: {subdirs:?}"
    );
    let project_b_subdir = subdirs
        .iter()
        .find(|p| p != &&project_a_subdir)
        .expect("a second, distinct project subdirectory for project B");
    let forked_path = std::fs::read_dir(project_b_subdir)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let header: serde_json::Value = serde_json::from_str(
        std::fs::read_to_string(&forked_path)
            .unwrap()
            .lines()
            .next()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(header["parent"].as_str().unwrap(), source_id);
}

#[test]
fn run_fork_with_an_unknown_id_fails_clearly_instead_of_silently_starting_fresh() {
    let home_dir = tempfile::tempdir().unwrap();
    let project_dir = tempfile::tempdir().unwrap();
    let (base, _bodies) = spawn_model_server(vec![]);
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
            "--fork",
            "no-such-session-id-anywhere",
        ])
        .current_dir(project_dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        !output.status.success(),
        "forking an unknown id must fail the run"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--fork") && stderr.contains("no-such-session-id-anywhere"),
        "stderr must name the flag and the id, not a leaked internal error shape: {stderr}"
    );
}

#[test]
fn run_binary_session_dir_flag_redirects_the_continue_repo() {
    // Pi-parity fix: `run` had no `--session-dir` equivalent at all — `--continue` was pinned to the
    // hardcoded default `~/.claude/sessions/<encoded-cwd>/` with no override, unlike `serve`'s own
    // `--session-dir`. The given directory must become the repo root directly (matching `serve`'s own
    // semantics), not a further per-cwd subdirectory nested under it.
    let repo_dir = tempfile::tempdir().unwrap();
    let project_dir = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let (base1, _bodies1) = spawn_model_server(vec![turn_text("first answer")]);
    let output1 = run_cmd(bin)
        .args([
            "run",
            "remember the marker: session-dir-55",
            "--gateway-url",
            &base1,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--continue",
            "--session-dir",
            repo_dir.path().to_str().unwrap(),
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

    // The session file must land directly inside `repo_dir` — not `repo_dir/.claude/sessions/...`.
    let entries: Vec<_> = std::fs::read_dir(repo_dir.path())
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("jsonl"))
        .collect();
    assert_eq!(
        entries.len(),
        1,
        "the session file must be created directly under --session-dir: {entries:?}"
    );

    // A second `--continue --session-dir` run must reopen the same session (proving `resume_or_create`
    // actually used the overridden directory, not the default one).
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
            "--continue",
            "--session-dir",
            repo_dir.path().to_str().unwrap(),
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
    let bodies2 = bodies2.lock().unwrap();
    assert!(
        bodies2[0].contains("session-dir-55"),
        "the second run must see the first run's history: {}",
        bodies2[0]
    );
}

#[test]
fn run_binary_ai_agent_session_dir_env_var_redirects_the_continue_repo() {
    let repo_dir = tempfile::tempdir().unwrap();
    let (base, _bodies) = spawn_model_server(vec![turn_text("ok")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let output = run_cmd(bin)
        .env("AI_AGENT_SESSION_DIR", repo_dir.path())
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
        ])
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "binary failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let entries: Vec<_> = std::fs::read_dir(repo_dir.path())
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("jsonl"))
        .collect();
    assert_eq!(
        entries.len(),
        1,
        "AI_AGENT_SESSION_DIR must redirect the repo the same way --session-dir does: {entries:?}"
    );
}

#[test]
fn run_binary_fork_with_session_dir_scopes_cross_project_search_to_its_parent() {
    // `--session-dir`'s cross-project search root (for `--fork <id>` when the id isn't in the current
    // repo) is that directory's own *parent* — matching how `serve`'s `list_all_sessions` scopes its
    // cross-project scan off `--session-dir`'s parent, so both binaries agree on what "every project"
    // means once a custom root is in play.
    let shared_root = tempfile::tempdir().unwrap();
    let repo_a = shared_root.path().join("repo-a");
    let repo_b = shared_root.path().join("repo-b");
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let (base1, _bodies1) = spawn_model_server(vec![turn_text("first answer")]);
    let output1 = run_cmd(bin)
        .args([
            "run",
            "remember the marker: cross-repo-88",
            "--gateway-url",
            &base1,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--continue",
            "--session-dir",
            repo_a.to_str().unwrap(),
        ])
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");
    assert!(
        output1.status.success(),
        "seeding run into repo-a failed.\nstderr: {}",
        String::from_utf8_lossy(&output1.stderr)
    );
    let source_path = std::fs::read_dir(&repo_a)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let source_id = session_id_of(&source_path);

    // `--fork <id> --session-dir repo-b` must find the session seeded into the *sibling* repo-a via
    // repo-b's parent (the shared root), not fail with "no session matching".
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
            "--fork",
            &source_id,
            "--session-dir",
            repo_b.to_str().unwrap(),
        ])
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");
    assert!(
        output2.status.success(),
        "forking across --session-dir siblings failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output2.stdout),
        String::from_utf8_lossy(&output2.stderr)
    );
    let bodies2 = bodies2.lock().unwrap();
    assert!(
        bodies2[0].contains("cross-repo-88"),
        "the forked run must carry over repo-a's history: {}",
        bodies2[0]
    );
}
