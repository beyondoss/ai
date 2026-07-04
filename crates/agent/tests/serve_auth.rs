//! `serve` e2e: the `login`/`logout`/`auth_status`/`submit_code`/`abort_login` RPC commands — the
//! headless counterpart to `auth_cli.rs`'s CLI coverage. None of these touch the model/session at
//! all, so no mock gateway is ever actually contacted (a bogus, unreachable `--gateway-url` is enough
//! — these tests never send a `prompt`).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufReader, Write};
use std::process::Command;

use common::{read_until_response, serve_cmd};
use serde_json::{Value, json};

/// A bogus, never-actually-contacted gateway URL — every test here only exercises the auth-store
/// commands, which never touch `GatewayClient` at all.
const DUMMY_GATEWAY: &str = "http://127.0.0.1:1";

/// `serve_cmd` bakes in `HOME=ISOLATED_HOME` (a deliberately nonexistent path, so any accidental
/// real-HOME-relative access fails loudly) — but the auth store's cross-process `FileLock` needs to
/// actually create a real directory to acquire its lock, unlike a plain settings/trust read. These
/// tests need a real, writable HOME, so `.env("HOME", ...)` here wins over `serve_cmd`'s own
/// (`Command::env` is last-write, per that helper's own doc comment).
fn serve_cmd_with_real_home(bin: &str, session_file: &str, home: &std::path::Path) -> Command {
    let mut c = serve_cmd(bin, DUMMY_GATEWAY, session_file);
    c.env("HOME", home);
    c
}

fn write_auth_json(home: &std::path::Path, body: &str) {
    let dir = home.join(".claude");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("auth.json"), body).unwrap();
}

#[test]
fn auth_status_with_nothing_stored_reports_every_provider_logged_out() {
    let home = tempfile::tempdir().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let mut child = serve_cmd_with_real_home(bin, &session_file, home.path())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "id": "1", "type": "auth_status" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "auth_status");
    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], true, "{frames:#?}");
    let providers = resp["data"]["providers"].as_array().unwrap();
    assert_eq!(providers.len(), 3);
    assert!(
        providers
            .iter()
            .all(|p| p["status"] == "logged_out"),
        "{providers:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn auth_status_for_a_single_provider_reports_logged_in_when_a_non_expired_credential_is_stored() {
    let home = tempfile::tempdir().unwrap();
    write_auth_json(
        home.path(),
        r#"{"anthropic":{"type":"oauth","access":"tok","refresh":"ref","expires":4000000000000}}"#,
    );
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let mut child = serve_cmd_with_real_home(bin, &session_file, home.path())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "id": "1", "type": "auth_status", "provider": "anthropic" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "auth_status");
    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], true, "{frames:#?}");
    assert_eq!(resp["data"]["provider"], "anthropic");
    assert_eq!(resp["data"]["status"], "logged_in");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn auth_status_for_an_unknown_provider_is_a_clean_error_not_a_crash() {
    let home = tempfile::tempdir().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let mut child = serve_cmd_with_real_home(bin, &session_file, home.path())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "id": "1", "type": "auth_status", "provider": "not-a-real-provider" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "auth_status");
    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], false, "{frames:#?}");

    // The server must still be alive and answer a follow-up command normally.
    writeln!(stdin, "{}", json!({ "id": "2", "type": "auth_status" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "auth_status");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn logout_removes_a_stored_credential_and_is_idempotent() {
    let home = tempfile::tempdir().unwrap();
    write_auth_json(
        home.path(),
        r#"{"anthropic":{"type":"oauth","access":"tok","refresh":"ref","expires":4000000000000}}"#,
    );
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let mut child = serve_cmd_with_real_home(bin, &session_file, home.path())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "id": "1", "type": "logout", "provider": "anthropic" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "logout");
    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], true, "{frames:#?}");
    assert_eq!(resp["data"]["was_logged_in"], true);

    writeln!(
        stdin,
        "{}",
        json!({ "id": "2", "type": "logout", "provider": "anthropic" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "logout");
    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], true, "{frames:#?}");
    assert_eq!(
        resp["data"]["was_logged_in"], false,
        "a second logout must be an idempotent no-op, not an error"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn login_acks_then_a_second_concurrent_login_is_rejected_then_abort_login_cancels_it() {
    let home = tempfile::tempdir().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let mut child = serve_cmd_with_real_home(bin, &session_file, home.path())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "id": "1", "type": "login", "provider": "anthropic" })
    )
    .unwrap();
    stdin.flush().unwrap();

    // Read frames until we've seen the `ack` for id 1 — the login itself keeps running in the
    // background (it needs a real local port + network to ever complete), so we don't wait for its
    // terminal `response` here.
    let ack = read_one_frame_matching(&mut stdout, |v| {
        v["type"] == "ack" && v["command"] == "login"
    });
    assert_eq!(ack["id"], "1", "{ack:#?}");

    // A second `login` while the first is in flight must be rejected outright, not queued.
    writeln!(
        stdin,
        "{}",
        json!({ "id": "2", "type": "login", "provider": "anthropic" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "login");
    let rejection = frames
        .iter()
        .find(|v| v["type"] == "response" && v["id"] == "2")
        .unwrap();
    assert_eq!(rejection["success"], false);
    assert!(
        rejection["error"]
            .as_str()
            .unwrap()
            .contains("already in flight"),
        "{rejection:#?}"
    );

    // Cancel the first login; it must resolve with a failure response, not hang forever.
    writeln!(stdin, "{}", json!({ "id": "3", "type": "abort_login" })).unwrap();
    stdin.flush().unwrap();
    let abort_ack = read_one_frame_matching(&mut stdout, |v| {
        v["type"] == "response" && v["id"] == "3"
    });
    assert_eq!(abort_ack["success"], true, "{abort_ack:#?}");

    let login_result = read_one_frame_matching(&mut stdout, |v| {
        v["type"] == "response" && v["command"] == "login" && v["id"] == "1"
    });
    assert_eq!(login_result["success"], false, "{login_result:#?}");

    // The slot must be cleared once the first login resolves — a fresh `login` is accepted again
    // immediately, not left rejected by a stale in-flight marker. Anthropic again (not a different
    // provider): its flow makes no network call until a code is actually obtained, so aborting it
    // immediately after the `ack` — as done here — never reaches out to a real OAuth endpoint at all,
    // keeping this test hermetic.
    writeln!(
        stdin,
        "{}",
        json!({ "id": "4", "type": "login", "provider": "anthropic" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let ack2 = read_one_frame_matching(&mut stdout, |v| {
        v["type"] == "ack" && v["command"] == "login"
    });
    assert_eq!(ack2["id"], "4", "{ack2:#?}");

    writeln!(stdin, "{}", json!({ "id": "5", "type": "abort_login" })).unwrap();
    stdin.flush().unwrap();
    let _ = read_one_frame_matching(&mut stdout, |v| {
        v["type"] == "response" && v["command"] == "login" && v["id"] == "4"
    });

    drop(stdin);
    child.wait().unwrap();
}

/// Read frames until one matches `pred`, returning it. Unlike `read_until_response`, doesn't stop at
/// the first `response` for a given command — needed here since several distinct commands' frames
/// interleave (`login`'s own eventual `response` arrives well after `abort_login`'s).
fn read_one_frame_matching(
    reader: &mut impl std::io::BufRead,
    pred: impl Fn(&Value) -> bool,
) -> Value {
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line).unwrap();
        assert!(n > 0, "stream ended before a matching frame arrived");
        let Ok(v) = serde_json::from_str::<Value>(line.trim()) else {
            continue;
        };
        if pred(&v) {
            return v;
        }
    }
}
