//! `run` e2e: Standalone CLI surface: `--version`/`--help`/`list-models`/`export`, and flags that reach the wire request untouched by a live turn.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::{Command, Stdio};

use common::{run_cmd, spawn_model_server, sse, turn_text, turn_text_responses, turn_tool_use};
use serde_json::json;

#[test]
fn run_binary_exports_the_transcript_when_asked() {
    let dir = tempfile::tempdir().unwrap();
    let (base, _bodies) = spawn_model_server(vec![turn_text("all done")]);
    let export_path = dir.path().join("transcript.html");

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "say hi",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--export",
            export_path.to_str().unwrap(),
            "--no-session-persistence",
        ])
        .current_dir(dir.path())
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "binary failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let html = std::fs::read_to_string(&export_path).expect("exported file must exist");
    assert!(html.starts_with("<!DOCTYPE html>"));
    assert!(html.contains("say hi"));
    assert!(html.contains("all done"));
    // Task #44 integration: `run --export` now calls `export_html_full` with the actually-running
    // agent's real system prompt and tool set, not the plainer entries-only form — both sections must
    // be present, not just the transcript.
    assert!(
        html.contains("System Prompt"),
        "expected a rendered system-prompt section: {html}"
    );
    assert!(
        html.contains("Available Tools"),
        "expected a rendered tools section: {html}"
    );
    assert!(
        html.contains("bash") && html.contains("read") && html.contains("write"),
        "the tools section should list the default registry's own tools: {html}"
    );
}

#[test]
fn run_binary_list_models_prints_known_model_ids_with_no_gateway_or_key() {
    // A pure informational query — no `--gateway-url`/`--key` needed, matching `tools`'s own shape.
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args(["list-models"])
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("claude-opus-4-8"), "stdout: {stdout}");
    assert!(stdout.contains("gpt-5"), "stdout: {stdout}");
}

#[test]
fn run_binary_list_models_search_argument_filters_to_matching_model_ids() {
    // pi-parity fix: `list-models` had no way to narrow a long table down to models matching a search
    // term (pi's own `--list-models <search>`) — an optional positional argument, case-insensitive
    // substring match against the model id.
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args(["list-models", "GPT"])
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("gpt-5"),
        "a case-insensitive match on 'GPT' must still find gpt-5: {stdout}"
    );
    assert!(
        !stdout.contains("claude-opus-4-8"),
        "a model id not matching the search term must be filtered out: {stdout}"
    );
}

#[test]
fn run_binary_list_models_search_argument_fuzzy_matches_a_non_contiguous_subsequence() {
    // Task #51 (pi-parity fix): `--list-models <search>` used to be a plain case-insensitive substring
    // check, which "sn5" would never match against "claude-sonnet-4-5" (no such literal substring
    // exists). pi's own `--list-models` uses a fuzzy, order-preserving subsequence match instead.
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args(["list-models", "sn5"])
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("claude-sonnet-4-5"),
        "a fuzzy subsequence match on \"sn5\" must find claude-sonnet-4-5: {stdout}"
    );
    assert!(
        !stdout.contains("gpt-5-mini"),
        "a model id whose characters don't appear in \"sn5\" order must still be filtered out: {stdout}"
    );
}

#[test]
fn run_binary_list_models_prints_capability_columns_not_just_bare_ids() {
    // Pi-parity fix: previously a bare list of model ids — pi's own `--list-models` prints a table
    // with context/max-out/thinking/images columns, all already computed by `agent_core::capabilities`
    // but never surfaced. This pins the enriched shape down so it can't silently regress back to a
    // bare list.
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args(["list-models"])
        .output()
        .expect("spawn binary");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Header row names the columns.
    assert!(stdout.contains("context"), "stdout: {stdout}");
    assert!(stdout.contains("max-out"), "stdout: {stdout}");
    assert!(stdout.contains("thinking"), "stdout: {stdout}");
    assert!(stdout.contains("vision"), "stdout: {stdout}");

    // `claude-opus-4-8`'s own row carries its actual capability numbers, not just its id.
    let opus_line = stdout
        .lines()
        .find(|l| l.contains("claude-opus-4-8"))
        .unwrap_or_else(|| panic!("no claude-opus-4-8 row: {stdout}"));
    assert!(
        opus_line.contains("1000000"),
        "expected the real context window on claude-opus-4-8's row: {opus_line}"
    );
    assert!(
        opus_line.contains("yes"),
        "claude-opus-4-8 supports thinking and vision, expected at least one yes: {opus_line}"
    );

    // `gpt-4o` doesn't support thinking/reasoning at all — its row must say so, not just "yes"
    // everywhere regardless of the actual model.
    let gpt4o_line = stdout
        .lines()
        .find(|l| l.contains("gpt-4o"))
        .unwrap_or_else(|| panic!("no gpt-4o row: {stdout}"));
    assert!(
        gpt4o_line.contains("no"),
        "gpt-4o has no thinking mechanism, expected a real \"no\" on its row: {gpt4o_line}"
    );
}

#[test]
fn run_binary_version_flag_prints_only_the_version_to_stdout() {
    // F-L3 (pi: stdout-cleanliness.test.ts "prints --version to stdout when stdout is redirected"):
    // `--version` is clap's own generated flag (`#[command(version, ...)]` on `Cli` — see `main.rs`),
    // and this module's `#![deny(clippy::print_stdout)]` already makes a stray application `println!`
    // a compile error, but neither of those is *proof* that `--version` itself stays clean — this pins
    // it down directly rather than assuming.
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .arg("--version")
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.trim() == format!("beyond-ai-agent {}", env!("CARGO_PKG_VERSION")),
        "stdout should be exactly the binary name and version, nothing else: {stdout:?}"
    );
    assert_eq!(
        output.stderr,
        b"",
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn run_binary_lowercase_v_alias_prints_the_same_version_as_capital_v() {
    // Task #43: clap's auto-generated version flag only binds the capital `-V`; pi documents a
    // lowercase `-v` alias too. `expand_short_aliases` (`main.rs`) rewrites it to `--version` before
    // clap ever sees it — this pins down that the two really produce identical output, not just that
    // the process happens to exit 0.
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let lower = run_cmd(bin).arg("-v").output().expect("spawn binary");
    let upper = run_cmd(bin).arg("-V").output().expect("spawn binary");

    assert!(
        lower.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&lower.stderr)
    );
    assert!(
        upper.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&upper.stderr)
    );
    assert_eq!(lower.stdout, upper.stdout);
    assert_eq!(lower.stderr, b"");
}

#[test]
fn run_binary_help_flag_prints_usage_to_stdout_with_empty_stderr() {
    // F-L3 (pi: stdout-cleanliness.test.ts "prints plain --help to stdout when stdout is redirected"):
    // same idea as the `--version` case above, for clap's generated `--help`.
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin).arg("--help").output().expect("spawn binary");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Usage:"), "stdout: {stdout}");
    assert_eq!(
        output.stderr,
        b"",
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn run_binary_json_help_flag_routes_usage_to_stderr_with_empty_stdout() {
    // F-L3 (pi: stdout-cleanliness.test.ts "keeps stdout empty for --mode json --help ..." / "keeps
    // stdout empty for -p --help ..."): unlike plain `run --help` above, `--json` marks `run`'s stdout
    // as the NDJSON `AgentEvent` stream (see `main.rs`'s `run_turn_once`) — the same one-frame-per-line
    // invariant `serve`'s NDJSON protocol depends on. clap's own `--help` short-circuit runs before any
    // application code, so it can't consult that fact on its own; `main.rs`'s `cli()` helper scans argv
    // for `run` + `--json` + a help flag and redirects clap's rendered help to stderr instead of
    // stdout when it does. No `--gateway-url`/`--key` needed: clap's help short-circuit fires before
    // any of that is ever read.
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args(["run", "--json", "--help",
            "--no-session-persistence",
        ])
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        output.stdout,
        b"",
        "stdout must stay empty when --json --help combine: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Usage:"), "stderr: {stderr}");
}

#[test]
fn export_subcommand_renders_an_existing_session_file_with_no_gateway_or_key() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl");

    // Create a session file the ordinary way, with a real (fake) model server — this part still
    // needs a gateway/key, exactly like any other `run`.
    let (base, _bodies) = spawn_model_server(vec![turn_text("the answer is 42")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let setup = run_cmd(bin)
        .args([
            "run",
            "what is the answer?",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session",
            session_file.to_str().unwrap(),
            "--no-session-persistence",
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");
    assert!(
        setup.status.success(),
        "session setup run failed.\nstderr: {}",
        String::from_utf8_lossy(&setup.stderr)
    );
    assert!(session_file.exists());

    // Now export that already-persisted session file directly — no --gateway-url/--key/--model at
    // all, proving the export subcommand is pure offline rendering of what's on disk, unlike `run
    // --export` (which only exports after a live model run completes).
    let export_path = dir.path().join("transcript.html");
    let output = Command::new(bin)
        .args([
            "export",
            session_file.to_str().unwrap(),
            export_path.to_str().unwrap(),
        ])
        .env("HOME", dir.path())
        .output()
        .expect("spawn binary");
    assert!(
        output.status.success(),
        "export subcommand failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let html = std::fs::read_to_string(&export_path).expect("exported file must exist");
    assert!(html.starts_with("<!DOCTYPE html>"));
    assert!(html.contains("what is the answer?"));
    assert!(html.contains("the answer is 42"));
    // Task #44 integration: the standalone subcommand has no live `Agent`/`ToolRegistry` to pull a
    // system prompt/tool set from (no gateway/key/model given at all, proven above) — it correctly
    // passes `None` for both rather than fabricating data, so neither section should render, unlike
    // `run --export`'s own live-run equivalent (see `run_binary_exports_the_transcript_when_asked`).
    assert!(
        !html.contains("System Prompt"),
        "the standalone export has no live system prompt to render: {html}"
    );
    assert!(
        !html.contains("Available Tools"),
        "the standalone export has no live tool registry to render: {html}"
    );
}

#[test]
fn run_binary_thinking_flag_reaches_the_wire_request() {
    // pi-parity fix (M12): `--thinking` existed on `serve` but not `run` — there was no way to enable
    // extended thinking for a one-shot task at all.
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
            "claude-test",
            "--thinking",
            "2048",
            "--no-session-persistence",
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
    let bodies = bodies.lock().unwrap();
    assert!(
        bodies[0].contains(r#""budget_tokens":2048"#),
        "--thinking must set the wire request's thinking budget: {}",
        bodies[0]
    );
}

#[test]
fn run_binary_temperature_flag_reaches_the_wire_request() {
    // Pi-parity gap: none of the three dialects exposed a temperature knob at all.
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
            "claude-test",
            "--temperature",
            "0.25",
            "--no-session-persistence",
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
    let bodies = bodies.lock().unwrap();
    assert!(
        bodies[0].contains(r#""temperature":0.25"#),
        "--temperature must set the wire request's temperature: {}",
        bodies[0]
    );
}

#[test]
fn run_binary_max_tokens_flag_reaches_the_wire_request() {
    // Pi-parity audit: `Agent::with_max_tokens` existed but had no CLI/RPC override anywhere —
    // every `run`/`serve` process was locked to the model-derived default.
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
            "claude-test",
            "--max-tokens",
            "555",
            "--no-session-persistence",
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
    let bodies = bodies.lock().unwrap();
    assert!(
        bodies[0].contains(r#""max_tokens":555"#),
        "--max-tokens must set the wire request's max_tokens: {}",
        bodies[0]
    );
}

#[test]
fn run_binary_carries_a_stable_prompt_cache_affinity_key_on_the_wire() {
    // Pi-parity audit: `serve` pins every request to a stable per-session `prompt_cache_key` (and
    // OpenAI session-affinity headers) via `Agent::with_cache_key`, so repeated turns keep landing on
    // the same cache-warmed backend node — but `run_task` never called `with_cache_key` at all, so a
    // `run` invocation's requests carried no affinity key whatsoever. Anthropic's dialect has no wire
    // manifestation of `cache_key` at all (it's an OpenAI-only concept), so this needs an OpenAI
    // Responses-dialect turn (`gpt-4o`) to actually observe the field on the wire.
    let (base, bodies) = spawn_model_server(vec![turn_text_responses("ok")]);

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
            "gpt-4o",
            "--no-session-persistence",
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
    let bodies = bodies.lock().unwrap();
    assert!(
        bodies[0].contains(r#""prompt_cache_key""#),
        "run's request must carry a stable prompt_cache_key: {}",
        bodies[0]
    );
    assert!(
        bodies[0]
            .to_ascii_lowercase()
            .contains("x-client-request-id:"),
        "run's request must carry the x-client-request-id session-affinity header: {}",
        bodies[0]
    );
}

#[test]
fn run_binary_context_window_flag_forces_proactive_compaction_on_a_resumed_session() {
    // Pi-parity fix: `run` had no `--context-window` override at all — `serve`'s own identical flag
    // already existed, but `run_task` hardcoded the model's own capability-table window with no way
    // to pin a smaller one. Mirrors `serve_proactively_compacts_a_resumed_large_session_on_its_very_
    // next_prompt`: seed a session already comfortably over a tiny pinned threshold, then confirm the
    // very first prompt against it triggers compaction (the summarization call) before the real
    // answer — proof `--context-window` actually reached the `Agent`'s `CompactionConfig`, not just
    // that the flag parses.
    let dir = tempfile::tempdir().unwrap();

    let session_file = {
        use agent_core::{ContentBlock, Message, Session};
        use beyond_ai_agent::session_store::{SessionMeta, SessionRepo};

        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "claude-test")).unwrap();
        let mut seed = Session::new();
        seed.user("u".repeat(400));
        seed.push(Message::assistant(vec![ContentBlock::text(
            "a".repeat(400),
        )]));
        seed.user("u".repeat(400));
        seed.push(Message::assistant(vec![ContentBlock::text(
            "a".repeat(400),
        )]));
        store.append_new(&seed.messages).unwrap();
        store.path().to_string_lossy().into_owned()
    };

    let (base, bodies) = spawn_model_server(vec![turn_text("SUMMARY"), turn_text("answered")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "continue please",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session",
            &session_file,
            "--context-window",
            "200",
            "--compaction-reserve-tokens",
            "50",
            "--compaction-keep-recent-tokens",
            "1",
            "--no-session-persistence",
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
    let bodies = bodies.lock().unwrap();
    assert_eq!(
        bodies.len(),
        2,
        "expected a summarization call before the real answer: {bodies:#?}"
    );
}

#[test]
fn run_binary_no_compaction_flag_suppresses_proactive_compaction_on_a_resumed_session() {
    // pi-parity fix: no way to disable automatic compaction entirely — `--no-compaction` must suppress
    // the *soft*, proactive threshold (`compaction::should_compact`) the sibling test above forces.
    //
    // Deliberately NOT the sibling's own `--context-window 200`/`--compaction-reserve-tokens 50`: this
    // test's ~400-estimated-token seed (4 messages × 400 chars ÷ 4) would then *also* cross the raw
    // `context_window` itself, tripping `compaction::is_hard_overflow`'s hard-overflow backstop — which
    // `agent.rs`'s turn loop deliberately runs regardless of `--no-compaction` (disabling proactive
    // compaction isn't license to keep sending requests already guaranteed to overflow — see
    // `is_hard_overflow`'s own doc comment). That would make this test pass for the wrong reason (the
    // unsuppressible hard backstop just happening not to fire) or fail outright (it does fire), neither
    // of which says anything about whether `--no-compaction` actually suppressed the *soft* trigger.
    // `--context-window 1000`/`--compaction-reserve-tokens 700` instead: the soft threshold
    // (1000 − 700 = 300) is still comfortably crossed by ~400 estimated tokens (proving there's a real
    // soft-threshold trigger to suppress), while the raw window (1000) is not (keeping the hard backstop
    // out of play), so the only thing standing between this session and a compaction call is
    // `--no-compaction` itself.
    let dir = tempfile::tempdir().unwrap();

    let session_file = {
        use agent_core::{ContentBlock, Message, Session};
        use beyond_ai_agent::session_store::{SessionMeta, SessionRepo};

        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "claude-test")).unwrap();
        let mut seed = Session::new();
        seed.user("u".repeat(400));
        seed.push(Message::assistant(vec![ContentBlock::text(
            "a".repeat(400),
        )]));
        seed.user("u".repeat(400));
        seed.push(Message::assistant(vec![ContentBlock::text(
            "a".repeat(400),
        )]));
        store.append_new(&seed.messages).unwrap();
        store.path().to_string_lossy().into_owned()
    };

    let (base, bodies) = spawn_model_server(vec![turn_text("answered")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "continue please",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session",
            &session_file,
            "--context-window",
            "1000",
            "--compaction-reserve-tokens",
            "700",
            "--compaction-keep-recent-tokens",
            "1",
            "--no-compaction",
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
    let bodies = bodies.lock().unwrap();
    assert_eq!(
        bodies.len(),
        1,
        "--no-compaction must suppress the proactive summarization call entirely: {bodies:#?}"
    );
}

#[test]
fn run_binary_cache_long_flag_reaches_the_wire_request() {
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
            "claude-test",
            "--cache-long",
            "--no-session-persistence",
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
    let bodies = bodies.lock().unwrap();
    assert!(
        bodies[0].contains(r#""ttl":"1h""#),
        "--cache-long must select the 1-hour cache TTL on the wire request: {}",
        bodies[0]
    );
}

#[test]
fn run_binary_bash_shell_path_flag_is_used_for_bash_calls() {
    let dir = tempfile::tempdir().unwrap();
    // A fake "shell" that just echoes a distinctive marker, proving the real command never ran through
    // the auto-resolved shell — only through this one.
    let fake_shell = dir.path().join("fake-shell.sh");
    std::fs::write(
        &fake_shell,
        "#!/bin/sh\necho ran-through-fake-shell-marker\n",
    )
    .unwrap();
    let mut perms = std::fs::metadata(&fake_shell).unwrap().permissions();
    std::os::unix::fs::PermissionsExt::set_mode(&mut perms, 0o755);
    std::fs::set_permissions(&fake_shell, perms).unwrap();

    let turn1 = turn_tool_use(
        "toolu_1",
        "bash",
        &json!({ "command": "echo real-command" }).to_string(),
    );
    let turn2 = turn_text("done");
    let (base, bodies) = spawn_model_server(vec![turn1, turn2]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "run a command",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--bash-shell-path",
            fake_shell.to_str().unwrap(),
            "--max-steps",
            "4",
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
        bodies[1].contains("ran-through-fake-shell-marker"),
        "--bash-shell-path must route the command through the given shell, not the real command's \
         own output: {}",
        bodies[1]
    );
}

#[test]
fn run_binary_bash_command_prefix_flag_runs_before_the_model_s_command() {
    // Pi-parity fix: `Bash::with_command_prefix` (pi's own `shellCommandPrefix` setting) was fully
    // built and tested at the tool level but had zero call sites reachable from either binary — no
    // CLI flag or env var ever set it.
    let dir = tempfile::tempdir().unwrap();
    let marker_file = dir.path().join("prefix-ran.txt");

    let turn1 = turn_tool_use(
        "toolu_1",
        "bash",
        &json!({ "command": "cat prefix-ran.txt" }).to_string(),
    );
    let turn2 = turn_text("done");
    let (base, bodies) = spawn_model_server(vec![turn1, turn2]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "run a command",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--bash-command-prefix",
            &format!("echo prefix-marker > {}", marker_file.to_str().unwrap()),
            "--max-steps",
            "4",
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
        bodies[1].contains("prefix-marker"),
        "--bash-command-prefix must run before the model's own command, in the same shell \
         invocation, so the command can see its effects: {}",
        bodies[1]
    );
}

#[test]
fn run_binary_ai_agent_bash_command_prefix_env_var_is_used_for_bash_calls() {
    let turn1 = turn_tool_use(
        "toolu_1",
        "bash",
        &json!({ "command": "echo real-command" }).to_string(),
    );
    let turn2 = turn_text("done");
    let (base, bodies) = spawn_model_server(vec![turn1, turn2]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .env("AI_AGENT_BASH_COMMAND_PREFIX", "echo env-prefix-marker")
        .args([
            "run",
            "run a command",
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
    assert!(bodies[1].contains("env-prefix-marker"), "{}", bodies[1]);
}

#[test]
fn run_binary_sequential_tools_flag_is_accepted_and_both_calls_still_run() {
    // Pi-parity fix: `agent_core::Agent` had no way to force fully-sequential tool dispatch at all —
    // see `agent_core::agent::tests::with_sequential_tools_forces_one_group_in_flight_at_a_time` for the
    // mechanism itself (proven there with a non-exclusive counting tool, since `bash` is already
    // `conservative_exclusive` and so always serializes regardless of this flag — it can't demonstrate
    // a *timing* difference end to end). This is the CLI wiring smoke test: `--sequential-tools`
    // parses, reaches `run_task`'s `Agent::with_sequential_tools`, and a turn batching two independent
    // `bash` calls still dispatches both correctly (right result mapped to the right `tool_use_id`, both
    // real commands actually ran) instead of the flag silently breaking multi-call turns.
    let dir = tempfile::tempdir().unwrap();
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
    let output = run_cmd(bin)
        .args([
            "run",
            "run two commands",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--sequential-tools",
            "--max-steps",
            "4",
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
    assert!(marker_a.exists(), "the first bash call must still run");
    assert!(marker_b.exists(), "the second bash call must still run");
}

#[test]
fn run_binary_deny_tool_flag_blocks_the_named_tool_end_to_end() {
    // Pi-parity Critical fix: `agent_core::AgentHooks`/`with_hooks` was fully built and tested but had
    // zero call sites outside its own unit test — every real `run`/`serve` process built its `Agent`
    // with the no-op `NoHooks`, so `bash`/`write`/`edit` ran completely unconstrained. Proves the real
    // compiled binary, not just `ToolPolicy`'s own unit tests, actually blocks a denied tool: the marker
    // file must never be created, and the model must see a policy-blocked `tool_result`, not the real
    // command's output.
    let dir = tempfile::tempdir().unwrap();
    let marker = dir.path().join("should-not-exist.txt");

    let turn1 = turn_tool_use(
        "toolu_1",
        "bash",
        &json!({ "command": format!("touch {}", marker.to_str().unwrap()) }).to_string(),
    );
    let turn2 = turn_text("done");
    let (base, bodies) = spawn_model_server(vec![turn1, turn2]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "run a command",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--deny-tool",
            "bash",
            "--max-steps",
            "4",
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
    assert!(
        !marker.exists(),
        "--deny-tool bash must block the call before it ever runs — the marker file must not exist"
    );
    let bodies = bodies.lock().unwrap();
    assert!(
        bodies[1].contains("denied by policy"),
        "the model must see a policy-blocked tool_result, not the real command's output: {}",
        bodies[1]
    );
}

#[test]
fn run_binary_deny_tool_env_var_blocks_the_named_tool_end_to_end() {
    // pi-parity fix: `deny_tool`/`deny_bash_pattern` had no `env = "..."` attrs on `Run` even though
    // the identical `Serve` fields did — so `AI_AGENT_DENY_TOOL`/`AI_AGENT_DENY_BASH_PATTERN` silently
    // had no effect on `run`, only on `serve`. Proves the env var alone (no `--deny-tool` flag at all)
    // blocks the call end-to-end, the same way the sibling flag-based test above does.
    let dir = tempfile::tempdir().unwrap();
    let marker = dir.path().join("should-not-exist.txt");

    let turn1 = turn_tool_use(
        "toolu_1",
        "bash",
        &json!({ "command": format!("touch {}", marker.to_str().unwrap()) }).to_string(),
    );
    let turn2 = turn_text("done");
    let (base, bodies) = spawn_model_server(vec![turn1, turn2]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "run a command",
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
        .env("AI_AGENT_DENY_TOOL", "bash")
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
        !marker.exists(),
        "AI_AGENT_DENY_TOOL=bash must block the call before it ever runs — the marker file must not \
         exist"
    );
    let bodies = bodies.lock().unwrap();
    assert!(
        bodies[1].contains("denied by policy"),
        "the model must see a policy-blocked tool_result, not the real command's output: {}",
        bodies[1]
    );
}

#[test]
fn run_binary_deny_bash_pattern_flag_blocks_matching_commands_end_to_end() {
    let dir = tempfile::tempdir().unwrap();
    let marker = dir.path().join("should-not-exist.txt");

    let turn1 = turn_tool_use(
        "toolu_1",
        "bash",
        &json!({ "command": format!("touch {}", marker.to_str().unwrap()) }).to_string(),
    );
    let turn2 = turn_text("done");
    let (base, bodies) = spawn_model_server(vec![turn1, turn2]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "run a command",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--deny-bash-pattern",
            "TOUCH", // uppercase, proving the match is case-insensitive against the lowercase command
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

    assert!(
        output.status.success(),
        "binary failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !marker.exists(),
        "--deny-bash-pattern must block a matching command before it ever runs"
    );
    let bodies = bodies.lock().unwrap();
    assert!(
        bodies[1].contains("blocked by policy"),
        "the model must see a policy-blocked tool_result: {}",
        bodies[1]
    );
}

#[test]
fn run_binary_deny_path_flag_blocks_a_write_to_the_matching_path_end_to_end() {
    // Track L22: `ToolPolicy` previously had no way to gate `write`/`edit` by their `path` argument at
    // all — only whole-tool-name (`--deny-tool`) or `bash`-substring (`--deny-bash-pattern`) denial.
    // Proves the real compiled binary blocks a `write` whose path matches `--deny-path`'s glob, before
    // the file is ever created.
    let dir = tempfile::tempdir().unwrap();
    let secret = dir.path().join("secrets.env");

    let turn1 = turn_tool_use(
        "toolu_1",
        "write",
        &json!({ "path": secret.to_str().unwrap(), "content": "TOKEN=abc" }).to_string(),
    );
    let turn2 = turn_text("done");
    let (base, bodies) = spawn_model_server(vec![turn1, turn2]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "write the secret",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--deny-path",
            "*.env",
            "--max-steps",
            "4",
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
    assert!(
        !secret.exists(),
        "--deny-path '*.env' must block the write before it ever runs — the file must not exist"
    );
    let bodies = bodies.lock().unwrap();
    assert!(
        bodies[1].contains("denied by policy"),
        "the model must see a policy-blocked tool_result, not the real write's output: {}",
        bodies[1]
    );
}

#[test]
fn run_binary_ai_agent_exclude_tools_env_var_restricts_the_registry() {
    // Pi-parity audit: `Run`'s `tools`/`exclude_tools` clap fields carried no `env = ...` attribute at
    // all, unlike every other shared flag and unlike `serve`'s identical flags — a deployment
    // convention setting `AI_AGENT_EXCLUDE_TOOLS` to sandbox an agent (e.g. a read-only reviewer) had
    // silently no effect on `run` invocations specifically, a security-relevant gap, not just an
    // inconsistency.
    let (base, bodies) = spawn_model_server(vec![turn_text("ok")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .env("AI_AGENT_EXCLUDE_TOOLS", "bash")
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
        "binary failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let bodies = bodies.lock().unwrap();
    assert!(
        !bodies[0].contains("\"bash\""),
        "AI_AGENT_EXCLUDE_TOOLS=bash must remove it from the request (tool defs or system prompt): {}",
        bodies[0]
    );
    assert!(
        bodies[0].contains("\"read\""),
        "other tools must remain advertised: {}",
        bodies[0]
    );
}

#[test]
fn run_binary_ai_agent_tools_env_var_restricts_the_registry_to_an_allow_list() {
    let (base, bodies) = spawn_model_server(vec![turn_text("ok")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .env("AI_AGENT_TOOLS", "read,write")
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
        "binary failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let bodies = bodies.lock().unwrap();
    assert!(
        bodies[0].contains("\"read\"") && bodies[0].contains("\"write\""),
        "the allow-listed tools must be advertised: {}",
        bodies[0]
    );
    assert!(
        !bodies[0].contains("\"bash\""),
        "AI_AGENT_TOOLS=read,write must exclude everything outside the allow-list: {}",
        bodies[0]
    );
}

#[test]
fn run_binary_short_t_flag_is_an_alias_for_tools() {
    // pi-parity fix: `--tools`/`--exclude-tools`/`--no-tools`/`--no-skills`/`--no-prompt-templates`/
    // `--no-context-files`/`--trust-project`/`--force-untrusted` had no short-flag aliases at all,
    // unlike pi's own CLI (`cli/args.ts`). `-t` matches pi's `--tools`/`-t`.
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
            "claude-test",
            "-t",
            "read,write",
            "--no-session-persistence",
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
    let bodies = bodies.lock().unwrap();
    assert!(
        bodies[0].contains("\"read\"") && bodies[0].contains("\"write\""),
        "-t must behave exactly like --tools: {}",
        bodies[0]
    );
    assert!(!bodies[0].contains("\"bash\""), "{}", bodies[0]);
}

#[test]
fn run_binary_short_xt_flag_is_an_alias_for_exclude_tools() {
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
            "claude-test",
            "-xt",
            "bash",
            "--no-session-persistence",
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
    let bodies = bodies.lock().unwrap();
    assert!(
        !bodies[0].contains("\"bash\""),
        "-xt must behave exactly like --exclude-tools: {}",
        bodies[0]
    );
    assert!(bodies[0].contains("\"read\""), "{}", bodies[0]);
}

#[test]
fn run_binary_short_nt_flag_is_an_alias_for_no_tools() {
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
            "claude-test",
            "-nt",
            "--no-session-persistence",
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
    let bodies = bodies.lock().unwrap();
    assert!(
        !bodies[0].contains("\"tools\":"),
        "-nt must behave exactly like --no-tools (no tools field on the wire at all): {}",
        bodies[0]
    );
}

#[test]
fn run_binary_short_a_and_na_flags_are_aliases_for_trust_project_and_force_untrusted() {
    // `-a` matches pi's own `--approve`/`-a`; `-na` matches pi's own `--no-approve`/`-na`. Proven
    // together, mirroring `run_binary_force_untrusted_overrides_trust_project`: `-na` must still win
    // even when `-a` is also given.
    let dir = tempfile::tempdir().unwrap();
    let skill_dir = dir.path().join(".claude/skills/greet");
    std::fs::create_dir_all(&skill_dir).unwrap();
    std::fs::write(
        skill_dir.join("SKILL.md"),
        "---\nname: greet\ndescription: a test skill\n---\nSKILL-BODY-MARKER-123",
    )
    .unwrap();
    let (base, bodies) = spawn_model_server(vec![turn_text("done")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "/skill:greet",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "-a",
            "-na",
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
        !bodies[0].contains("SKILL-BODY-MARKER-123"),
        "-na must override -a, so the skill must not expand: {}",
        bodies[0]
    );
}

#[test]
fn run_binary_short_ns_flag_is_an_alias_for_no_skills() {
    let dir = tempfile::tempdir().unwrap();
    let skill_dir = dir.path().join(".claude/skills/greet");
    std::fs::create_dir_all(&skill_dir).unwrap();
    std::fs::write(
        skill_dir.join("SKILL.md"),
        "---\nname: greet\ndescription: a test skill\n---\nSKILL-BODY-MARKER-123",
    )
    .unwrap();
    let (base, bodies) = spawn_model_server(vec![turn_text("done")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "/skill:greet",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "-a",
            "-ns",
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
        !bodies[0].contains("SKILL-BODY-MARKER-123"),
        "-ns must behave exactly like --no-skills: {}",
        bodies[0]
    );
    assert!(bodies[0].contains("/skill:greet"), "{}", bodies[0]);
}

#[test]
fn run_binary_short_np_flag_is_an_alias_for_no_prompt_templates() {
    let dir = tempfile::tempdir().unwrap();
    let prompt_dir = dir.path().join(".claude/prompts");
    std::fs::create_dir_all(&prompt_dir).unwrap();
    std::fs::write(
        prompt_dir.join("fix.md"),
        "Fix the bug in $1 — TEMPLATE-BODY-MARKER-456",
    )
    .unwrap();
    let (base, bodies) = spawn_model_server(vec![turn_text("done")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "/fix foo.rs",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "-a",
            "-np",
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
        !bodies[0].contains("TEMPLATE-BODY-MARKER-456"),
        "-np must behave exactly like --no-prompt-templates: {}",
        bodies[0]
    );
}

#[test]
fn run_binary_short_nc_flag_is_an_alias_for_no_context_files() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("CLAUDE.md"), "PROJECT-MARKER-777").unwrap();
    let (base, bodies) = spawn_model_server(vec![turn_text("done")]);

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
            "-nc",
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
        !bodies[0].contains("PROJECT-MARKER-777"),
        "-nc must behave exactly like --no-context-files: {}",
        bodies[0]
    );
}
