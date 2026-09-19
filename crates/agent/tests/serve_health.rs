//! `/livez` and `/readyz` — the two endpoints that make a `serve --service` replica deployable.
//!
//! They are the one part of the listener that is **not** a tenant's: an orchestrator deciding whether
//! this replica may have sessions at all has no session grant to present, and could never obtain one.
//! So these assert the whole contract from outside the process — over TCP and over the Unix socket,
//! with a grant and without one — against a real `serve --service` child.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::path::Path;
use std::process::{Command, Stdio};

use common::service::{Options, Service};
use common::{
    BIN, ChildGuard, ISOLATED_HOME, SpawnGuarded, free_port, spawn_model_server, turn_text,
    wait_for_port, ws_connect_with_headers, ws_read_until_response, ws_send,
};
use serde_json::{Value, json};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::TcpStream;

/// One raw HTTP/1.1 exchange, returning the status and the **entire** response text.
///
/// Raw bytes rather than a client library on purpose: a health probe is whatever the orchestrator
/// sends, and the assertion "no grant is echoed" has to see every byte that came back — headers
/// included, since a reflected header is exactly the leak being ruled out.
async fn exchange<S>(
    mut stream: S,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
) -> (u16, String)
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let mut req = format!("{method} {path} HTTP/1.1\r\nHost: health.test\r\nConnection: close\r\n");
    for (name, value) in headers {
        req.push_str(&format!("{name}: {value}\r\n"));
    }
    req.push_str("\r\n");
    stream.write_all(req.as_bytes()).await.expect("write");
    stream.flush().await.expect("flush");
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).await.expect("read");
    let text = String::from_utf8_lossy(&buf).into_owned();
    let status: u16 = text
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("no status line in {text:?}"));
    (status, text)
}

async fn get(port: u16, path: &str, headers: &[(&str, &str)]) -> (u16, String) {
    let stream = TcpStream::connect(("127.0.0.1", port))
        .await
        .expect("connect to the replica");
    exchange(stream, "GET", path, headers).await
}

fn body_of(response: &str) -> Value {
    let body = response
        .split_once("\r\n\r\n")
        .map(|(_, b)| b)
        .unwrap_or_default();
    serde_json::from_str(body).unwrap_or_else(|e| panic!("body {body:?} is not JSON: {e}"))
}

/// `/livez` answers with no grant, no session, and no query string — and the same replica still
/// refuses an ungranted *session* attach, so "no grant" is a property of the health path, not of a
/// replica that forgot to check.
#[tokio::test]
async fn livez_is_200_without_a_grant() {
    let (base, _requests) = spawn_model_server(vec![]);
    let svc = Service::start(&base, &["s1"]).await;

    let (status, response) = get(svc.port, "/livez", &[]).await;
    assert_eq!(status, 200, "{response}");
    assert_eq!(body_of(&response)["status"], "alive");

    // Contrast: the agent path with no grant is a 401 on the same listener.
    let refused = ws_connect_with_headers(svc.port, Some("s1.alpha"), &[]).await;
    assert_eq!(refused.err(), Some(401));

    // Anything but GET on a health path is a 405, not a 411/400 from the POST body reader.
    let stream = TcpStream::connect(("127.0.0.1", svc.port)).await.unwrap();
    let (status, response) = exchange(stream, "POST", "/livez", &[]).await;
    assert_eq!(status, 405, "{response}");
}

/// The happy path: every `--shard` is a directory the replica can write to.
#[tokio::test]
async fn readyz_is_200_when_every_shard_is_writable() {
    let (base, _requests) = spawn_model_server(vec![]);
    let svc = Service::start(&base, &["s1", "s2"]).await;

    let (status, response) = get(svc.port, "/readyz", &[]).await;
    assert_eq!(status, 200, "{response}");
    assert_eq!(body_of(&response)["status"], "ready");

    // Repeated probes are served from the memo and stay correct.
    for _ in 0..3 {
        let (status, _) = get(svc.port, "/readyz", &[]).await;
        assert_eq!(status, 200);
    }

    // The probe leaves nothing behind in the shard it wrote to.
    for (name, path) in &svc.shards {
        let leftovers: Vec<_> = std::fs::read_dir(path)
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert!(leftovers.is_empty(), "{name}: {leftovers:?}");
    }
}

/// A mount that went away is a 503 naming **which** shard — and naming only the shard. The reason is
/// readable by anything that can reach the port, so it must not hand out the replica's mount layout.
#[tokio::test]
async fn readyz_is_503_and_names_the_missing_shard() {
    let (base, _requests) = spawn_model_server(vec![]);
    let svc = Service::start(&base, &["s1", "s2"]).await;
    let gone = svc.shard("s2").to_path_buf();
    std::fs::remove_dir_all(&gone).unwrap();

    let (status, response) = get(svc.port, "/readyz", &[]).await;
    assert_eq!(status, 503, "{response}");
    let reason = body_of(&response)["reason"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    assert!(reason.contains("shard s2"), "{reason}");
    assert!(reason.contains("not mounted"), "{reason}");
    assert!(
        !response.contains(&gone.to_string_lossy().into_owned()),
        "the mount path must not be in the body: {response}"
    );

    // Liveness is unaffected: killing and restarting the process does not put a mount back.
    let (status, _) = get(svc.port, "/livez", &[]).await;
    assert_eq!(status, 200);
}

/// A read-only remount is the failure `stat` alone would miss, which is why the probe writes.
#[tokio::test]
#[cfg(unix)]
async fn readyz_is_503_when_a_shard_is_read_only() {
    use std::os::unix::fs::PermissionsExt;

    let probe = tempfile::tempdir().unwrap();
    if !write_bit_is_enforced(probe.path()) {
        eprintln!("skipping: running as root, which ignores the write bit");
        return;
    }

    let (base, _requests) = spawn_model_server(vec![]);
    let svc = Service::start(&base, &["s1", "s2"]).await;
    let ro = svc.shard("s2").to_path_buf();
    std::fs::set_permissions(&ro, std::fs::Permissions::from_mode(0o555)).unwrap();

    let (status, response) = get(svc.port, "/readyz", &[]).await;
    // Restore before asserting, so a failure still lets the temp dir clean itself up.
    std::fs::set_permissions(&ro, std::fs::Permissions::from_mode(0o755)).unwrap();

    assert_eq!(status, 503, "{response}");
    let reason = body_of(&response)["reason"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    assert!(reason.contains("shard s2"), "{reason}");
    assert!(reason.contains("not writable"), "{reason}");
}

/// Root ignores the write bit, so the read-only case is not testable there. Detect it by trying,
/// rather than by reading a uid — the question is the behaviour, not the identity.
#[cfg(unix)]
fn write_bit_is_enforced(root: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    let dir = root.join("ro-probe");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o555)).unwrap();
    let enforced = std::fs::File::create(dir.join("x")).is_err();
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();
    enforced
}

/// Both endpoints work on the Unix-socket transport, which is the one a sidecar probe uses when the
/// TCP listener is reserved for the edge.
#[tokio::test]
#[cfg(unix)]
async fn health_answers_over_a_unix_socket() {
    let sock_dir = tempfile::tempdir().unwrap();
    let sock = sock_dir.path().join("agent.sock");
    let (base, _requests) = spawn_model_server(vec![]);
    let svc = Service::start_with(
        &base,
        &["s1"],
        Options {
            extra_args: vec![
                "--listen-uds".to_string(),
                sock.to_string_lossy().into_owned(),
            ],
            ..Default::default()
        },
    )
    .await;

    for _ in 0..50 {
        if sock.exists() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }

    let stream = tokio::net::UnixStream::connect(&sock)
        .await
        .expect("connect to the unix socket");
    let (status, response) = exchange(stream, "GET", "/livez", &[]).await;
    assert_eq!(status, 200, "{response}");
    assert_eq!(body_of(&response)["status"], "alive");

    let stream = tokio::net::UnixStream::connect(&sock).await.unwrap();
    let (status, response) = exchange(stream, "GET", "/readyz", &[]).await;
    assert_eq!(status, 200, "{response}");
    assert_eq!(body_of(&response)["status"], "ready");

    // Same process, same shards: the TCP listener agrees.
    let (status, _) = get(svc.port, "/readyz", &[]).await;
    assert_eq!(status, 200);
}

/// A grant is neither required nor echoed. `HttpHead::take_grant` strips it before anything else
/// reads the head, and the health responses are built from scratch rather than from request headers.
#[tokio::test]
async fn a_grant_is_never_required_and_never_echoed() {
    let (base, _requests) = spawn_model_server(vec![]);
    let svc = Service::start(&base, &["s1"]).await;
    let token = svc.token("t1", "s1.alpha");

    for path in ["/livez", "/readyz"] {
        let (status, response) = get(svc.port, path, &svc.header(&token)).await;
        assert_eq!(status, 200, "{path}: {response}");
        assert!(
            !response.contains(&token),
            "{path} echoed the grant: {response}"
        );
        assert!(
            !response.to_ascii_lowercase().contains("x-beyond-grant"),
            "{path}: {response}"
        );
    }

    // A grant far past the header cap is dropped unread on this path too — still a 200, not a 401.
    let huge = "A".repeat(13 * 1024);
    let (status, response) = get(svc.port, "/readyz", &[("x-beyond-grant", &huge)]).await;
    assert_eq!(status, 200, "{response}");
    assert!(!response.contains("AAAA"), "{response}");
}

/// Probing must not disturb the thing being probed: the session table is never touched, and a run
/// before and after a probe behaves identically.
#[tokio::test]
async fn readyz_does_not_interfere_with_a_session_attach() {
    let (base, _requests) = spawn_model_server(vec![turn_text("hi")]);
    let svc = Service::start(&base, &["s1"]).await;

    let (status, _) = get(svc.port, "/readyz", &[]).await;
    assert_eq!(status, 200);

    let token = svc.token("t1", "s1.alpha");
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .expect("attach with a valid grant");

    let (status, _) = get(svc.port, "/readyz", &[]).await;
    assert_eq!(status, 200, "a probe mid-session must not fail");

    ws_send(
        &mut ws,
        json!({"type":"prompt","id":"p1","message":"hello"}),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "prompt").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");

    ws_send(&mut ws, json!({"type":"get_state","id":"g1"})).await;
    let frames = ws_read_until_response(&mut ws, "get_state").await;
    assert_eq!(frames.last().unwrap()["success"], true);

    // And the probe still only ever reports on shards — no session directory is walked, so a live
    // session leaves no trace in the answer.
    let (status, response) = get(svc.port, "/readyz", &[]).await;
    assert_eq!(status, 200);
    assert!(!response.contains("s1.alpha"), "{response}");
}

/// The single-tenant daemon has them too: `/livez` always, and `/readyz` reporting ready once the
/// listener is up, since with no shards and no verifier there is nothing else readiness could mean.
#[tokio::test]
async fn health_exists_on_a_non_service_daemon() {
    let (base, _requests) = spawn_model_server(vec![]);
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let _child: ChildGuard = Command::new(BIN)
        .args([
            "serve",
            "--listen",
            &format!("127.0.0.1:{port}"),
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session-dir",
            &dir.path().to_string_lossy(),
        ])
        .env("HOME", ISOLATED_HOME)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn_guarded();
    wait_for_port(port);

    let (status, response) = get(port, "/livez", &[]).await;
    assert_eq!(status, 200, "{response}");
    assert_eq!(body_of(&response)["status"], "alive");

    let (status, response) = get(port, "/readyz", &[]).await;
    assert_eq!(status, 200, "{response}");
    assert_eq!(body_of(&response)["status"], "ready");

    // An unknown path is still a 404 — health did not widen the router.
    let (status, _) = get(port, "/nope", &[]).await;
    assert_eq!(status, 404);
}
