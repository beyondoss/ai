//! `serve --service` with the seal key in `AI_AGENT_SEAL_KEY_B64` rather than a file.
//!
//! The variable carries the key itself so a platform that injects secrets as environment (ECS) need
//! not write a file first. It is read from the environment and never declared as a flag (a flag is
//! argv, which every process on the host can read). Giving both forms is refused, and no refusal
//! may print the value.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::{Command, Stdio};

use common::service::{Options, Service};
use common::{
    BIN, SpawnGuarded, spawn_model_server, turn_text, ws_connect_with_headers,
    ws_read_until_response, ws_send,
};
use serde_json::json;

#[tokio::test]
async fn the_env_form_serves_a_real_turn() {
    let (base, _requests) = spawn_model_server(vec![turn_text("hi")]);
    let svc = Service::start_with(
        &base,
        &["s1"],
        Options {
            seal_key_from_env: true,
            ..Options::default()
        },
    )
    .await;
    let token = svc.token("t1", "s1.alpha");
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .expect("a grant sealed to the env-provided key must open");
    ws_send(
        &mut ws,
        json!({"type":"prompt","id":"p1","message":"hello"}),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "prompt").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
}

struct Keys {
    dir: tempfile::TempDir,
    flag: String,
    seal: std::path::PathBuf,
    shard: std::path::PathBuf,
}

fn keys() -> Keys {
    let dir = tempfile::tempdir().unwrap();
    let minter = common::grant::Minter::new(dir.path());
    let shard = dir.path().join("shard");
    std::fs::create_dir_all(&shard).unwrap();
    Keys {
        flag: minter.grant_key_flag(),
        seal: minter.seal_key().to_path_buf(),
        shard,
        dir,
    }
}

/// `serve --service` with no seal flag; the caller adds the file flag and/or the env. Returns stderr
/// of a startup that must fail.
fn refused(k: &Keys, seal_flag: bool, env: Option<&str>) -> String {
    // Startup must fail before it binds; a held, never-listening port keeps a regression from
    // ever serving on it.
    let dead = beyond_ai_test_support::ports::DeadPort::bind();
    let listen = format!("127.0.0.1:{}", dead.port());
    let mut c = Command::new(BIN);
    c.args([
        "serve",
        "--service",
        "--gateway-url",
        "http://127.0.0.1:1",
        "--model",
        "claude-test",
        "--listen",
        &listen,
        "--grant-key",
        &k.flag,
        "--shard",
        &format!("a={}", k.shard.display()),
    ]);
    if seal_flag {
        c.arg("--seal-key").arg(&k.seal);
    }
    if let Some(v) = env {
        c.env("AI_AGENT_SEAL_KEY_B64", v);
    }
    c.env("HOME", common::ISOLATED_HOME);
    let output = c
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn_guarded()
        .wait_with_output()
        .unwrap();
    let all = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !output.status.success(),
        "expected a startup failure: {all}"
    );
    all
}

#[test]
fn bad_base64_is_refused_without_printing_it() {
    let k = keys();
    let secret = "!!not-base64-SECRETMARKER!!";
    let err = refused(&k, false, Some(secret));
    assert!(err.contains("AI_AGENT_SEAL_KEY_B64"), "{err}");
    assert!(!err.contains("SECRETMARKER"), "value echoed: {err}");
}

#[test]
fn a_wrong_length_key_is_refused_without_printing_it() {
    use base64::Engine as _;
    let k = keys();
    for len in [31usize, 33, 64] {
        let value = base64::engine::general_purpose::STANDARD.encode(vec![0x5au8; len]);
        let err = refused(&k, false, Some(&value));
        assert!(err.contains("32-byte"), "{len}: {err}");
        assert!(!err.contains(&value), "{len}: value echoed: {err}");
    }
}

#[test]
fn giving_both_forms_is_refused() {
    let k = keys();
    // A perfectly valid key in the env, and a valid file: the refusal is about ambiguity alone.
    let valid = std::fs::read_to_string(&k.seal).unwrap();
    let err = refused(&k, true, Some(valid.trim()));
    assert!(
        err.contains("AI_AGENT_SEAL_KEY_B64") && err.contains("both"),
        "{err}"
    );
    assert!(!err.contains(valid.trim()), "value echoed: {err}");
    drop(k.dir);
}
