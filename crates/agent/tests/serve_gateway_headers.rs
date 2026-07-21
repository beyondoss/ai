//! `serve` e2e: a `models.json` `ModelOverride.headers` override must reach every gateway client
//! `serve` builds — startup, and every runtime rebuild (`set_model`/`cycle_model`/`fork`/`clone`/
//! `switch_session`/`switch_branch`), not just `run`'s one-shot `run_task` construction.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufReader, Write};

use common::{SpawnGuarded, read_until_response, serve_cmd, spawn_model_server, turn_text};
use serde_json::json;

#[test]
fn serve_startup_model_override_header_reaches_the_wire_request() {
    // Task 2 (pi-parity fix, serve pass 19): `main.rs::run_task` chains
    // `.with_extra_headers(model_override_extra_headers(&model))` onto its gateway client; `serve.rs`'s
    // own `build_gateway_client` never did, silently defeating any `models.json`
    // `ModelOverride.headers` (and the auto-seeded NVIDIA/Kimi-Coding default headers) whenever the
    // model is used via `serve` instead of `run`. `claude-test` is the model `serve_cmd` always starts
    // with.
    let dir = tempfile::tempdir().unwrap();
    let config_dir = dir.path().join("config");
    std::fs::create_dir_all(&config_dir).unwrap();
    std::fs::write(
        config_dir.join("models.json"),
        r#"{"claude-test": {"headers": {"X-Custom-Auth": "literal-header-value"}}}"#,
    )
    .unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, bodies) = spawn_model_server(vec![turn_text("done")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file)
        .env("AI_AGENT_CONFIG_DIR", &config_dir)
        .spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    drop(stdin);
    child.wait().unwrap();

    let bodies = bodies.lock().unwrap();
    assert!(
        bodies[0]
            .to_ascii_lowercase()
            .contains("x-custom-auth: literal-header-value"),
        "the configured header must reach the startup gateway client's request, matching `run`'s \
         identical `model_override_extra_headers` wiring: {}",
        bodies[0]
    );
}

#[test]
fn serve_set_model_to_an_overridden_model_carries_that_models_own_header_on_the_next_request() {
    // The header wiring must survive a runtime `set_model` rebuild too — `build_gateway_client` is
    // re-invoked from `set_model`'s own arm with the *new* model id, so the header lookup must be
    // re-resolved fresh each time, not just fixed at startup.
    let dir = tempfile::tempdir().unwrap();
    let config_dir = dir.path().join("config");
    std::fs::create_dir_all(&config_dir).unwrap();
    std::fs::write(
        config_dir.join("models.json"),
        r#"{"gpt-4o": {"headers": {"X-Other-Model-Header": "gpt4o-value"}}}"#,
    )
    .unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    // First request is on `claude-test` (no override configured for it); second is after switching to
    // `gpt-4o`, which does have one.
    let (base, bodies) = spawn_model_server(vec![turn_text("first"), turn_text("second")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file)
        .env("AI_AGENT_CONFIG_DIR", &config_dir)
        .spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_model", "model": "gpt-4o" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_model");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "again" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    let bodies = bodies.lock().unwrap();
    assert!(
        !bodies[0]
            .to_ascii_lowercase()
            .contains("x-other-model-header"),
        "the gpt-4o-only header must not leak onto the first (claude-test) request: {}",
        bodies[0]
    );
    assert!(
        bodies[1]
            .to_ascii_lowercase()
            .contains("x-other-model-header: gpt4o-value"),
        "set_model's own rebuilt gateway client must carry the newly-active model's configured \
         header: {}",
        bodies[1]
    );
}
