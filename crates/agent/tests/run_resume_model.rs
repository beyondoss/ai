//! `run` e2e: a resumed session talks to its own model, over that model's route. `--continue` without
//! `--model` resolved the credential, headers and route for the CLI default model, then sent the
//! session's model over them; and `run` never recorded a `--model` switch, so the next `--continue`
//! went back to the model the session was created with.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::Stdio;

use common::{run_cmd, spawn_model_server, turn_text};

fn run(
    config_dir: &std::path::Path,
    sessions: &std::path::Path,
    gateway: &str,
    extra: &[&str],
) -> std::process::Output {
    let out = run_cmd(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .env("AI_AGENT_CONFIG_DIR", config_dir)
        .env_remove("AI_AGENT_MODEL")
        .args(["run", "--gateway-url", gateway, "--key", "bai_v1.test"])
        .arg("--session-dir")
        .arg(sessions)
        .args(extra)
        .current_dir(sessions.parent().unwrap())
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "run {extra:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out
}

fn models(bodies: &[String]) -> Vec<String> {
    bodies
        .iter()
        .map(|b| {
            common::body_json(b)["model"]
                .as_str()
                .unwrap_or("?")
                .to_string()
        })
        .collect()
}

#[test]
fn continue_routes_over_the_resumed_models_own_base_url() {
    let dir = tempfile::tempdir().unwrap();
    let config_dir = dir.path().join("config");
    std::fs::create_dir_all(&config_dir).unwrap();
    let sessions = dir.path().join("sessions");
    let (gateway, gateway_bodies) = spawn_model_server(vec![turn_text("from-gateway")]);
    let (direct, direct_bodies) =
        spawn_model_server(vec![turn_text("direct-1"), turn_text("direct-2")]);
    std::fs::write(
        config_dir.join("models.json"),
        format!(r#"{{"claude-custom-local": {{"base_url": "{direct}", "dialect": "anthropic"}}}}"#),
    )
    .unwrap();

    run(
        &config_dir,
        &sessions,
        &gateway,
        &["first", "--model", "claude-custom-local"],
    );
    run(&config_dir, &sessions, &gateway, &["second", "--continue"]);

    assert!(
        gateway_bodies.lock().unwrap().is_empty(),
        "the resumed model has its own base_url; nothing goes to the gateway"
    );
    let direct = direct_bodies.lock().unwrap().clone();
    assert_eq!(
        models(&direct),
        ["claude-custom-local", "claude-custom-local"]
    );
    assert!(
        direct[1].contains("first"),
        "the resume carries the history"
    );
}

#[test]
fn continue_after_a_model_switch_stays_on_the_switched_model() {
    let dir = tempfile::tempdir().unwrap();
    let config_dir = dir.path().join("config");
    std::fs::create_dir_all(&config_dir).unwrap();
    let sessions = dir.path().join("sessions");
    let (gateway, bodies) =
        spawn_model_server(vec![turn_text("1"), turn_text("2"), turn_text("3")]);

    run(
        &config_dir,
        &sessions,
        &gateway,
        &["one", "--model", "claude-model-a"],
    );
    run(
        &config_dir,
        &sessions,
        &gateway,
        &["two", "--continue", "--model", "claude-model-b"],
    );
    let third = run(
        &config_dir,
        &sessions,
        &gateway,
        &["three", "--continue", "--json"],
    );

    let bodies = bodies.lock().unwrap().clone();
    assert_eq!(
        models(&bodies),
        ["claude-model-a", "claude-model-b", "claude-model-b"]
    );
    let header: serde_json::Value = serde_json::from_str(
        String::from_utf8_lossy(&third.stdout)
            .lines()
            .next()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(header["kind"], "session");
    assert_eq!(
        header["model"], "claude-model-b",
        "the session header names the model in use"
    );
}
