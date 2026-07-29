//! `serve` e2e: per-session exec endpoints over the real control protocol.
//!
//! The unit-level proof lives in `exec_multi_tenant.rs`. This is the same claim asserted through the
//! actual server — a real `serve` process, real NDJSON commands — because a mechanism that works in
//! isolation and is wired up wrongly is indistinguishable from one that does not work.
//!
//! Two things are checked that only the server can show:
//!
//! 1. `set_exec_endpoint` points *this session's* tools at a target, and every tool follows.
//! 2. **Switching sessions detaches it.** A tenant's sandbox must not be inherited by whoever gets
//!    the session next. Failing open to "runs on the host" is recoverable; failing open to "runs in
//!    the previous tenant's box" is a breach.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::{
    SpawnGuarded, read_until_response, serve_dir_cmd, spawn_model_server, turn_text, turn_tool_use,
};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};

const BIN: &str = env!("CARGO_BIN_EXE_beyond-ai-agent");

/// The `response` frame from a command's frame batch. `read_until_response` returns everything it saw
/// up to and including it; the response is always last.
fn response(frames: &[Value]) -> &Value {
    frames.last().expect("at least one frame")
}

/// Everything the server emitted for a command, as one searchable string — the tool results a turn
/// produced are in here, which is what the secret assertions look at.
fn all(frames: &[Value]) -> String {
    frames
        .iter()
        .map(|f| f.to_string())
        .collect::<Vec<_>>()
        .join("\n")
}

/// A tenant's sandbox: a directory reachable only through that tenant's exec target.
fn tenant(secret: &str) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("secret.txt"), format!("{secret}\n")).unwrap();
    dir
}

/// `env -C <dir> {}` stands in for `docker exec`/`ssh` — it runs what follows, somewhere specific.
fn target_cmd(dir: &std::path::Path) -> String {
    format!("env -C {} {{}}", dir.display())
}

#[test]
fn a_session_endpoint_is_set_over_the_protocol_and_cleared_on_session_switch() {
    let a = tenant("SECRET-AAAA");
    let b = tenant("SECRET-BBBB");
    let sessions = tempfile::tempdir().unwrap();

    // Two scripted turns, each reading the same relative path. Which tenant answers is entirely
    // decided by the endpoint in force at the time.
    let (base, _bodies) = spawn_model_server(vec![
        turn_tool_use("t1", "read", &json!({ "path": "secret.txt" }).to_string()),
        turn_text("first"),
        turn_tool_use("t2", "read", &json!({ "path": "secret.txt" }).to_string()),
        turn_text("second"),
        turn_tool_use("t3", "read", &json!({ "path": "secret.txt" }).to_string()),
        turn_text("third"),
    ]);

    let mut child = serve_dir_cmd(BIN, &base, sessions.path().to_str().unwrap()).spawn_guarded();

    let mut stdin = child.stdin.take().unwrap();
    let mut out = BufReader::new(child.stdout.take().unwrap());
    let mut send = |v: Value| {
        writeln!(stdin, "{v}").unwrap();
        stdin.flush().unwrap();
    };

    // 1. Attach tenant A's sandbox to this session.
    send(json!({ "id": "1", "type": "set_exec_endpoint", "command": target_cmd(a.path()) }));
    let f = read_until_response(&mut out, "set_exec_endpoint");
    let resp = response(&f);
    assert_eq!(
        resp["success"],
        json!(true),
        "set_exec_endpoint failed: {resp}"
    );
    assert_eq!(resp["data"]["attached"], json!(true), "{resp}");

    // 2. A turn now reads tenant A's file.
    send(json!({ "id": "2", "type": "prompt", "message": "read it" }));
    let transcript = all(&read_until_response(&mut out, "prompt"));
    assert!(
        transcript.contains("SECRET-AAAA"),
        "the session's tools did not reach tenant A: {transcript}"
    );

    // 3. Re-point at tenant B — the same session, a different sandbox.
    send(json!({ "id": "3", "type": "set_exec_endpoint", "command": target_cmd(b.path()) }));
    assert_eq!(
        response(&read_until_response(&mut out, "set_exec_endpoint"))["success"],
        json!(true)
    );
    send(json!({ "id": "4", "type": "prompt", "message": "read it" }));
    let done = all(&read_until_response(&mut out, "prompt"));
    assert!(
        done.contains("SECRET-BBBB"),
        "re-pointing did not take effect: {done}"
    );
    assert!(
        !done.contains("SECRET-AAAA"),
        "tenant A's secret survived the re-point: {done}"
    );

    // 4. **The security-critical one.** A new session must NOT inherit the endpoint. Note this is a
    //    same-model switch, which does not rebuild the tool registry — the exact case where a target
    //    captured at construction time would silently persist.
    send(json!({ "id": "5", "type": "new_session" }));
    assert_eq!(
        response(&read_until_response(&mut out, "new_session"))["success"],
        json!(true)
    );
    send(json!({ "id": "6", "type": "prompt", "message": "read it" }));
    let done = all(&read_until_response(&mut out, "prompt"));
    assert!(
        !done.contains("SECRET-BBBB") && !done.contains("SECRET-AAAA"),
        "a new session inherited the previous session's sandbox — cross-tenant leak: {done}"
    );

    send(json!({ "id": "7", "type": "shutdown" }));
    let _ = out.read_line(&mut String::new());
}

#[test]
fn clearing_the_endpoint_returns_the_session_to_the_host() {
    // Mixed deployment: a session can be handed back to local execution without being torn down, and
    // "cleared" must mean *this host*, not "whatever was attached before".
    let a = tenant("SECRET-AAAA");
    let host_dir = tempfile::tempdir().unwrap();
    std::fs::write(host_dir.path().join("host.txt"), "ON-THE-HOST\n").unwrap();
    let sessions = tempfile::tempdir().unwrap();

    let (base, _bodies) = spawn_model_server(vec![
        turn_tool_use(
            "t1",
            "read",
            &json!({ "path": host_dir.path().join("host.txt").to_str().unwrap() }).to_string(),
        ),
        turn_text("done"),
    ]);

    let mut child = serve_dir_cmd(BIN, &base, sessions.path().to_str().unwrap()).spawn_guarded();

    let mut stdin = child.stdin.take().unwrap();
    let mut out = BufReader::new(child.stdout.take().unwrap());
    let mut send = |v: Value| {
        writeln!(stdin, "{v}").unwrap();
        stdin.flush().unwrap();
    };

    send(json!({ "id": "1", "type": "set_exec_endpoint", "command": target_cmd(a.path()) }));
    assert_eq!(
        response(&read_until_response(&mut out, "set_exec_endpoint"))["success"],
        json!(true)
    );

    // No `url` and no `command` detaches, explicitly.
    send(json!({ "id": "2", "type": "set_exec_endpoint" }));
    let f = read_until_response(&mut out, "set_exec_endpoint");
    let resp = response(&f);
    assert_eq!(resp["success"], json!(true), "{resp}");
    assert_eq!(resp["data"]["attached"], json!(false), "{resp}");

    send(json!({ "id": "3", "type": "prompt", "message": "read the host file" }));
    let done = all(&read_until_response(&mut out, "prompt"));
    assert!(
        done.contains("ON-THE-HOST"),
        "after clearing, the tools should read this host: {done}"
    );

    send(json!({ "id": "4", "type": "shutdown" }));
    let _ = out.read_line(&mut String::new());
}

#[test]
fn a_malformed_endpoint_is_rejected_without_attaching_anything() {
    // A bad URL must fail loudly at attach, not silently leave the session pointed at the host while
    // the caller believes it is sandboxed — that is how a tenant's work ends up on a shared box.
    let sessions = tempfile::tempdir().unwrap();
    let (base, _bodies) = spawn_model_server(vec![turn_text("unused")]);

    let mut child = serve_dir_cmd(BIN, &base, sessions.path().to_str().unwrap()).spawn_guarded();

    let mut stdin = child.stdin.take().unwrap();
    let mut out = BufReader::new(child.stdout.take().unwrap());
    let mut send = |v: Value| {
        writeln!(stdin, "{v}").unwrap();
        stdin.flush().unwrap();
    };

    send(json!({ "id": "1", "type": "set_exec_endpoint", "url": "file:///etc/passwd" }));
    let f = read_until_response(&mut out, "set_exec_endpoint");
    assert_eq!(
        response(&f)["success"],
        json!(false),
        "a non-http URL must be refused: {:?}",
        response(&f)
    );

    send(
        json!({ "id": "2", "type": "set_exec_endpoint", "url": "https://x/", "command": "env {}" }),
    );
    let f = read_until_response(&mut out, "set_exec_endpoint");
    assert_eq!(
        response(&f)["success"],
        json!(false),
        "url+command together must be refused: {:?}",
        response(&f)
    );

    send(json!({ "id": "3", "type": "shutdown" }));
    let _ = out.read_line(&mut String::new());
}
