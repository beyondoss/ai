//! `serve --service` at the door: which grants get in, and what a refusal looks like on the wire.
//!
//! Every case here asserts an **HTTP status**, not a close code or an error frame. That is the whole
//! point of verifying before the method branch and pinning before the upgrade: an edge, a proxy, or
//! a retry policy has to be able to tell "your grant expired" (401) from "that session isn't yours"
//! (403) from "wrong replica, try another" (421) without speaking the agent protocol.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::grant::{Claims, Minter, Secrets};
use common::service::{Options, Service};
use common::{spawn_model_server, ws_connect_with_headers};
use serde_json::Value;

/// The minter in `tests/common/grant.rs` is written from the spec, not from `src/grant.rs`. This is
/// what keeps that honest: given the golden vectors' own inputs it must produce the golden token,
/// byte for byte — the same bar the control plane's Go minter has to clear. Without it, a mistake
/// shared by the test minter and the verifier would make every status test below pass vacuously.
#[test]
fn the_test_minter_reproduces_the_golden_vector_byte_for_byte() {
    let fx: Value = serde_json::from_str(include_str!("fixtures/grant/v1.json")).unwrap();
    let hex32 = |key: &str| -> [u8; 32] {
        hex::decode(fx[key].as_str().unwrap())
            .unwrap()
            .try_into()
            .unwrap()
    };
    let dir = tempfile::tempdir().unwrap();
    let minter = Minter::with_keys(
        dir.path(),
        fx["kid"].as_u64().unwrap() as u32,
        hex32("signing_seed_hex"),
        hex32("fleet_secret_hex"),
    );
    assert_eq!(
        minter.grant_key_flag(),
        fx["grant_key_flag"].as_str().unwrap(),
        "the flag form of the signing key must match"
    );

    let c = &fx["claims"];
    let claims = Claims {
        tenant: c["tenant"].as_str().unwrap().into(),
        session_id: c["session_id"].as_str().unwrap().into(),
        home_shard: c["home_shard"].as_str().unwrap().into(),
        workspace_root: c["workspace_root"].as_str().unwrap().into(),
        exec_url: c["exec_url"].as_str().unwrap().into(),
        mcp: c["mcp"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| {
                (
                    m["name"].as_str().unwrap().to_string(),
                    m["url"].as_str().unwrap().to_string(),
                )
            })
            .collect(),
        exp: c["exp"].as_u64().unwrap(),
    };
    let s = &fx["secrets"];
    let headers = |v: &Value| -> Vec<(String, String)> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|h| {
                (
                    h["name"].as_str().unwrap().to_string(),
                    h["value"].as_str().unwrap().to_string(),
                )
            })
            .collect()
    };
    let secrets = Secrets {
        exec_headers: headers(&s["exec_headers"]),
        mcp_headers: s["mcp_headers"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), headers(v)))
            .collect(),
        gateway_key: s["gateway_key"].as_str().unwrap().into(),
        dek: base64::Engine::decode(
            &base64::engine::general_purpose::STANDARD,
            s["dek"].as_str().unwrap(),
        )
        .unwrap()
        .try_into()
        .unwrap(),
    };
    let nonce: [u8; 24] = hex::decode(fx["nonce_hex"].as_str().unwrap())
        .unwrap()
        .try_into()
        .unwrap();
    let token = minter.mint_with(&claims, &secrets, hex32("ephemeral_secret_hex"), nonce);
    assert_eq!(token, fx["token"].as_str().unwrap());
}

/// One replica, two shards, so an unmounted-shard case is a real routing miss rather than an
/// empty-config artifact.
async fn service() -> Service {
    let (base, _requests) = spawn_model_server(vec![]);
    Service::start(&base, &["s1", "s2"]).await
}

#[tokio::test]
async fn a_connection_with_no_grant_is_unauthorized() {
    let svc = service().await;
    let status = ws_connect_with_headers(svc.port, Some("s1.alpha"), &[])
        .await
        .expect_err("no grant must be refused");
    assert_eq!(status, 401);
}

#[tokio::test]
async fn a_grant_that_does_not_verify_is_unauthorized() {
    let svc = service().await;
    let valid = svc.token("t1", "s1.alpha");
    // Garbage, a foreign signing key, and an expired-but-otherwise-perfect grant are all 401: the
    // client is not authenticated, and the replica says nothing about which stage failed.
    let other = Minter::with_keys(svc.dir.path(), 1, [0x99; 32], [0x98; 32]);
    let forged = other.mint(&svc.claims("t1", "s1.alpha", "s1"), &svc.secrets());
    let expired = svc.token_with("t1", "s1.alpha", |c| c.exp = 1);
    let oversize = "x".repeat(13 * 1024);
    for (what, token) in [
        ("garbage", "not-a-grant".to_string()),
        ("truncated", valid[..valid.len() - 4].to_string()),
        ("foreign signing key", forged),
        ("expired", expired),
        ("over the 12 KiB cap", oversize),
    ] {
        let status = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
            .await
            .expect_err(what);
        assert_eq!(status, 401, "{what}");
    }
}

#[tokio::test]
async fn a_grant_with_no_session_id_in_the_url_is_a_bad_request() {
    let svc = service().await;
    let token = svc.token("t1", "s1.alpha");
    // A grant names exactly one session; without `?session_id=` the daemon would mint a fresh id and
    // run it under a grant that never authorized it.
    let status = ws_connect_with_headers(svc.port, None, &svc.header(&token))
        .await
        .expect_err("a grant needs an explicit session id");
    assert_eq!(status, 400);
}

#[tokio::test]
async fn a_grant_for_another_session_is_forbidden() {
    let svc = service().await;
    let token = svc.token("t1", "s1.alpha");
    let status = ws_connect_with_headers(svc.port, Some("s1.beta"), &svc.header(&token))
        .await
        .expect_err("a grant for alpha must not open beta");
    assert_eq!(status, 403);
}

/// The one that has to be a status rather than a close code: tenant B holds a perfectly valid grant
/// — for its own session with the same id — and the refusal comes from the *slot*, which only exists
/// once A's session is live. Pinning before the upgrade is what makes this a 403 instead of an
/// accepted socket that hangs up.
#[tokio::test]
async fn a_live_session_is_forbidden_to_another_tenant_before_the_upgrade() {
    let (base, _requests) = spawn_model_server(vec![]);
    let svc = Service::start(&base, &["s1"]).await;
    let a = svc.token("tenant-a", "s1.shared");
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.shared"), &svc.header(&a))
        .await
        .expect("tenant A opens its own session");
    // Drive one command so the session is genuinely live, not merely pinned.
    common::ws_send(&mut ws, serde_json::json!({"type":"get_state","id":"a1"})).await;
    let frames = common::ws_read_until_response(&mut ws, "get_state").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");

    let b = svc.token("tenant-b", "s1.shared");
    let status = ws_connect_with_headers(svc.port, Some("s1.shared"), &svc.header(&b))
        .await
        .expect_err("a foreign tenant must not attach to a live session");
    assert_eq!(status, 403);
}

#[tokio::test]
async fn a_shard_this_replica_does_not_mount_is_misdirected() {
    let svc = service().await;
    // The id's own shard is unmounted …
    let elsewhere = svc.token_with("t1", "s9.alpha", |_| {});
    let status = ws_connect_with_headers(svc.port, Some("s9.alpha"), &svc.header(&elsewhere))
        .await
        .expect_err("an unmounted shard must be misdirected");
    assert_eq!(status, 421);

    // … and so is the home shard, even when the session id's own shard is served: the tenant's
    // memory lives on the home shard, so serving this would silently run without it.
    let homeless = svc.token_with("t1", "s1.alpha", |c| c.home_shard = "s9".into());
    let status = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&homeless))
        .await
        .expect_err("an unmounted home shard must be misdirected");
    assert_eq!(status, 421);
}

#[tokio::test]
async fn claims_that_are_not_the_right_shape_are_a_bad_request() {
    let svc = service().await;
    /// One spoiled-claims case: what is wrong, and the edit that makes it so.
    type Case = (&'static str, Box<dyn FnOnce(&mut Claims)>);
    let cases: Vec<Case> = vec![
        (
            "a tenant that is not a path segment",
            Box::new(|c: &mut Claims| c.tenant = "../escape".into()),
        ),
        (
            "an empty tenant",
            Box::new(|c: &mut Claims| c.tenant = String::new()),
        ),
        (
            "a relative workspace root",
            Box::new(|c: &mut Claims| c.workspace_root = "relative/path".into()),
        ),
        (
            "a home shard with a dot",
            Box::new(|c: &mut Claims| c.home_shard = "s1.evil".into()),
        ),
        (
            "a connector name that is not a path segment",
            Box::new(|c: &mut Claims| {
                c.mcp = vec![("../evil".into(), "https://example.invalid/mcp".into())]
            }),
        ),
    ];
    for (what, edit) in cases {
        let token = svc.token_with("t1", "s1.alpha", edit);
        let status = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
            .await
            .expect_err(what);
        assert_eq!(status, 400, "{what}");
    }
}

/// The grant must not survive the read: `create_response` echoes the request's headers into the
/// handshake, so a grant left on the head would be reflected straight back to whoever sent it.
#[tokio::test]
async fn the_grant_header_is_not_echoed_in_the_handshake_response() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let (base, _requests) = spawn_model_server(vec![]);
    let svc = Service::start(&base, &["s1"]).await;
    let token = svc.token("t1", "s1.alpha");

    let mut sock = tokio::net::TcpStream::connect(("127.0.0.1", svc.port))
        .await
        .unwrap();
    let req = format!(
        "GET /_beyond/agent?session_id=s1.alpha HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\n\
         Connection: Upgrade\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\
         x-beyond-grant: {token}\r\n\r\n"
    );
    sock.write_all(req.as_bytes()).await.unwrap();
    let mut buf = vec![0u8; 4096];
    let n = sock.read(&mut buf).await.unwrap();
    let response = String::from_utf8_lossy(&buf[..n]).into_owned();
    assert!(
        response.starts_with("HTTP/1.1 101"),
        "expected an upgrade, got: {response}"
    );
    assert!(
        !response.contains(&token),
        "the grant was reflected into the handshake response: {response}"
    );
    assert!(
        !response.to_ascii_lowercase().contains("x-beyond-grant"),
        "the grant header survived into the handshake response: {response}"
    );
}

/// Without `--service` the header is meaningless and nothing about the daemon changes.
#[tokio::test]
async fn a_daemon_without_service_mode_ignores_the_grant_header() {
    let (base, _requests) = spawn_model_server(vec![]);
    let dir = tempfile::tempdir().unwrap();
    let port = common::free_port();
    let mut cmd = common::serve_dir_cmd(
        common::BIN,
        &base,
        &dir.path().join("sessions").to_string_lossy(),
    );
    cmd.arg("--listen").arg(format!("127.0.0.1:{port}"));
    let _child = {
        use common::SpawnGuarded as _;
        cmd.spawn_guarded()
    };
    common::wait_for_port(port);

    let mut ws = ws_connect_with_headers(port, Some("plain"), &[("x-beyond-grant", "nonsense")])
        .await
        .expect("a non-service daemon accepts any connection");
    common::ws_send(&mut ws, serde_json::json!({"type":"get_state","id":"g1"})).await;
    let frames = common::ws_read_until_response(&mut ws, "get_state").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
}

/// `Options` exists for the suites that need it; keep it exercised here so an unused-field change is
/// caught by this file rather than by a distant one.
#[tokio::test]
async fn a_replica_starts_with_extra_arguments() {
    let (base, _requests) = spawn_model_server(vec![]);
    let svc = Service::start_with(
        &base,
        &["s1"],
        Options {
            extra_args: vec!["--session-idle-timeout".into(), "0".into()],
            ..Options::default()
        },
    )
    .await;
    let token = svc.token("t1", "s1.alpha");
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .expect("a valid grant opens its session");
    common::ws_send(&mut ws, serde_json::json!({"type":"get_state","id":"g1"})).await;
    let frames = common::ws_read_until_response(&mut ws, "get_state").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
}
