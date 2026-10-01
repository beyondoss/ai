//! Claim tests: the request id on every response joins the billing row, and the operator surface
//! (`doctor`, `/livez`, `/readyz`, `/metrics`) answers as documented.
//!
//! Run via `mise run test:integration:rs`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::*;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const CHAT: &str = r#"{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}"#;

fn request_id(resp: &reqwest::Response) -> Option<String> {
    resp.headers()
        .get("x-beyond-request-id")
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
}

async fn post(gw: &Gateway, path: &str, auth: Option<&str>, body: &str) -> reqwest::Response {
    let mut req = test_client()
        .post(format!("{}{path}", gw.url()))
        .header("content-type", "application/json");
    if let Some(key) = auth {
        req = req.header("authorization", format!("Bearer {key}"));
    }
    req.body(body.to_owned()).send().await.unwrap()
}

/// A served call's `x-beyond-request-id` is the `request_id` on its billing row, and every
/// gateway-made rejection carries one too.
/// claim: O1
#[tokio::test]
async fn every_response_carries_a_request_id_and_a_served_one_joins_its_row() {
    let (pubkey, sk) = test_keypair(55);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .rate_limit_rps(2)
        .start()
        .await;
    let key = billing_vkey(&sk, 55);

    let served = post(&gw, "/v1/chat/completions", Some(&key), CHAT).await;
    assert_eq!(served.status().as_u16(), 200);
    let id = request_id(&served).expect("a served response carries its id");
    let row = usage_row_of(&gw).await;
    assert_eq!(row["request_id"].as_str(), Some(id.as_str()), "{row}");

    // A key each, so the per-credential limit below trips only where it is meant to.
    let (k404, k400, kpath) = (
        billing_vkey(&sk, 551),
        billing_vkey(&sk, 552),
        billing_vkey(&sk, 553),
    );
    let rejects = [
        (401, "/v1/chat/completions", None, CHAT.to_owned()),
        (
            401,
            "/v1/chat/completions",
            Some("bai_v1.1.bogus.bogus"),
            CHAT.to_owned(),
        ),
        (
            404,
            "/v1/chat/completions",
            Some(k404.as_str()),
            r#"{"model":"no-such-model","messages":[]}"#.to_owned(),
        ),
        (
            400,
            "/v1/chat/completions",
            Some(k400.as_str()),
            r#"{"model":"text-embedding-3-small","messages":[]}"#.to_owned(),
        ),
        (
            404,
            "/bogus/v1/chat/completions",
            Some(kpath.as_str()),
            CHAT.to_owned(),
        ),
    ];
    for (want, path, auth, body) in rejects {
        let resp = post(&gw, path, auth, &body).await;
        assert_eq!(resp.status().as_u16(), want, "{path} {auth:?} {body}");
        assert!(
            request_id(&resp).is_some(),
            "a {want} on {path} carries no x-beyond-request-id"
        );
    }

    // A rate-limit 429.
    let mut limited = None;
    for _ in 0..20 {
        let resp = post(&gw, "/v1/chat/completions", Some("sk-byo-flood"), CHAT).await;
        if resp.status().as_u16() == 429 {
            limited = Some(resp);
            break;
        }
    }
    let limited = limited.expect("the per-credential limit tripped");
    assert!(request_id(&limited).is_some(), "the 429 carries an id");

    // A 413, declared on the header and refused before any body is read.
    let mut s = tokio::net::TcpStream::connect(("127.0.0.1", gw.port))
        .await
        .unwrap();
    s.write_all(
        format!(
            "POST /v1/chat/completions HTTP/1.1\r\nhost: x\r\nauthorization: Bearer {key}\r\n\
             content-type: application/json\r\ncontent-length: 209715201\r\nconnection: close\r\n\r\n"
        )
        .as_bytes(),
    )
    .await
    .unwrap();
    let mut buf = vec![0u8; 4096];
    let n = tokio::time::timeout(Duration::from_secs(5), s.read(&mut buf))
        .await
        .unwrap()
        .unwrap();
    let resp = parse_raw_response(&buf[..n]);
    assert_eq!(resp.status, 413);
    assert!(
        resp.header("x-beyond-request-id").is_some(),
        "the 413 carries an id: {resp:?}"
    );
}

/// `beyond-ai doctor` runs its checks and exits non-zero naming the one that failed (here NATS,
/// which nothing listens for).
/// claim: O3
#[tokio::test]
async fn doctor_names_a_failed_check_and_exits_one() {
    let path = std::env::temp_dir().join(format!("beyond-ai-doctor-{}.toml", free_port()));
    std::fs::write(
        &path,
        format!(
            "listen = \"127.0.0.1:{}\"\nmetrics_listen = \"127.0.0.1:{}\"\n\
             nats_url = \"nats://127.0.0.1:{}\"\nupstream_tls = false\n",
            free_port(),
            free_port(),
            closed_port()
        ),
    )
    .unwrap();
    let cmd_path = path.clone();
    let out = tokio::time::timeout(
        Duration::from_secs(60),
        tokio::task::spawn_blocking(move || {
            std::process::Command::new(env!("CARGO_BIN_EXE_beyond-ai"))
                .arg("doctor")
                .arg("-c")
                .arg(&cmd_path)
                .output()
                .unwrap()
        }),
    )
    .await
    .expect("doctor must finish promptly")
    .unwrap();
    let _ = std::fs::remove_file(&path);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(1), "{stdout}");
    assert!(
        stdout.contains("== Beyond AI Gateway Doctor =="),
        "{stdout}"
    );
    assert!(stdout.contains("[FAIL] nats"), "{stdout}");
    assert!(
        stdout.contains("[ok]") || stdout.contains("[FAIL] signing_keys"),
        "every check reports a line: {stdout}"
    );
}

/// `/livez`, `/readyz` and `/metrics` answer on the admin listener; `/metrics` is Prometheus text
/// that counts the traffic just served.
/// claim: O3
#[tokio::test]
async fn health_and_metrics_answer_on_the_admin_listener() {
    let (pubkey, sk) = test_keypair(56);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let (live, body) = gw.admin_get("/livez").await;
    assert_eq!(live, 200, "{body}");
    assert!(body.contains("\"status\":\"ok\""), "{body}");
    let (ready, body) = gw.admin_get("/readyz").await;
    assert_eq!(ready, 200, "{body}");
    assert!(body.contains("\"status\""), "{body}");

    let served = post(
        &gw,
        "/v1/chat/completions",
        Some(&billing_vkey(&sk, 56)),
        CHAT,
    )
    .await;
    assert_eq!(served.status().as_u16(), 200);
    let resp = reqwest::get(format!("http://127.0.0.1:{}/metrics", gw.metrics_port))
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_owned();
    assert!(ct.starts_with("text/plain"), "{ct}");
    let text = resp.text().await.unwrap();
    assert!(text.contains("# TYPE ai_tokens_total"), "{text}");
    wait_for_metric(&gw, "ai_tokens_total", "input", 11.0).await;
}
