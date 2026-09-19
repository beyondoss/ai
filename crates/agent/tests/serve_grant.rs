//! `serve --grant-key`/`--seal-key`: startup validation of session-grant (`bsg_v1`) trust.
//!
//! The wire format, the crypto, and the golden vectors are unit-tested in `src/grant.rs`. This file
//! is the CLI wiring against the real binary: valid keys start `serve` exactly as before, and every
//! bad combination stops it at startup with a message that says what's wrong — and, for the seal key,
//! never what's in the file.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use common::{SpawnGuarded, read_until_response, serve_cmd, spawn_model_server};
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_beyond-ai-agent");

/// The golden vectors' own flag-format keys, so this file and `grant.rs` agree on what valid means.
fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/grant/v1.json")).unwrap()
}

fn grant_key_flag() -> String {
    fixture()["grant_key_flag"].as_str().unwrap().to_owned()
}

fn write_seal_key(dir: &Path, contents: &str) -> PathBuf {
    let path = dir.join("seal.key");
    std::fs::write(&path, contents).unwrap();
    path
}

fn serve(dir: &Path) -> Command {
    let (base, _requests) = spawn_model_server(vec![]);
    let session_file = dir.join("s.jsonl").to_string_lossy().into_owned();
    serve_cmd(BIN, &base, &session_file)
}

/// Drive one `get_state` round trip, then close stdin: the server started and serves as usual.
fn assert_serves(cmd: &mut Command) {
    let mut child = cmd.spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    writeln!(stdin, "{}", json!({ "type": "get_state", "id": "g1" })).unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    drop(stdin);
    assert!(child.wait().unwrap().success());
}

/// Run to exit (stdin closed first, so a server that wrongly started exits too) and return stderr.
fn startup_error(cmd: &mut Command) -> String {
    let output = cmd
        .stderr(Stdio::piped())
        .spawn_guarded()
        .wait_with_output()
        .unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        !output.status.success(),
        "expected a startup failure; stderr: {stderr}"
    );
    stderr
}

#[test]
fn serve_starts_as_usual_with_valid_grant_flags() {
    let dir = tempfile::tempdir().unwrap();
    let seal_key = write_seal_key(dir.path(), fixture()["seal_key_file"].as_str().unwrap());
    assert_serves(
        serve(dir.path())
            .arg("--grant-key")
            .arg(grant_key_flag())
            .arg("--seal-key")
            .arg(&seal_key),
    );
}

#[test]
fn serve_takes_several_grant_keys_from_the_environment() {
    // Rotation: two kids trusted at once, comma-separated in the env var.
    let dir = tempfile::tempdir().unwrap();
    let seal_key = write_seal_key(dir.path(), fixture()["seal_key_file"].as_str().unwrap());
    let old = grant_key_flag();
    let new = format!("2={}", old.split_once('=').unwrap().1);
    assert_serves(
        serve(dir.path())
            .env("AI_AGENT_GRANT_KEY", format!("{old},{new}"))
            .env("AI_AGENT_SEAL_KEY", &seal_key),
    );
}

#[test]
fn serve_fails_fast_on_a_half_configured_grant_trust() {
    let dir = tempfile::tempdir().unwrap();
    let seal_key = write_seal_key(dir.path(), fixture()["seal_key_file"].as_str().unwrap());

    let stderr = startup_error(serve(dir.path()).arg("--grant-key").arg(grant_key_flag()));
    assert!(stderr.contains("--grant-key needs --seal-key"), "{stderr}");

    let stderr = startup_error(serve(dir.path()).arg("--seal-key").arg(&seal_key));
    assert!(
        stderr.contains("--seal-key needs at least one --grant-key"),
        "{stderr}"
    );
}

#[test]
fn serve_fails_fast_on_a_bad_grant_key() {
    let dir = tempfile::tempdir().unwrap();
    let seal_key = write_seal_key(dir.path(), fixture()["seal_key_file"].as_str().unwrap());
    let key = grant_key_flag();
    for (grant_keys, expected) in [
        (
            vec!["1:not-a-key".to_owned()],
            "expected <kid>=<base64 Ed25519 public key>",
        ),
        (vec!["one=AAAA".to_owned()], "the kid must be a decimal u32"),
        (
            vec!["1=AAAA".to_owned()],
            "not base64 of a 32-byte Ed25519 public key",
        ),
        (vec![key.clone(), key], "kid 1 is given more than once"),
    ] {
        let mut cmd = serve(dir.path());
        for k in &grant_keys {
            cmd.arg("--grant-key").arg(k);
        }
        let stderr = startup_error(cmd.arg("--seal-key").arg(&seal_key));
        assert!(stderr.contains(expected), "{grant_keys:?}: {stderr}");
    }
}

#[test]
fn serve_fails_fast_on_a_bad_seal_key_without_echoing_it() {
    let dir = tempfile::tempdir().unwrap();
    let not_a_key = "c2VjcmV0LWJ1dC13cm9uZy1sZW5ndGg=";
    let seal_key = write_seal_key(dir.path(), not_a_key);
    let stderr = startup_error(
        serve(dir.path())
            .arg("--grant-key")
            .arg(grant_key_flag())
            .arg("--seal-key")
            .arg(&seal_key),
    );
    assert!(
        stderr.contains("expected base64 of a 32-byte X25519 secret"),
        "{stderr}"
    );
    assert!(stderr.contains("seal.key"), "{stderr}");
    assert!(
        !stderr.contains(not_a_key),
        "the seal key's contents leaked: {stderr}"
    );

    let missing = dir.path().join("missing.key");
    let stderr = startup_error(
        serve(dir.path())
            .arg("--grant-key")
            .arg(grant_key_flag())
            .arg("--seal-key")
            .arg(&missing),
    );
    assert!(stderr.contains("missing.key"), "{stderr}");
}
