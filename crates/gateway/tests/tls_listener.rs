//! The optional TLS listener (`tls_listen` + `tls_cert` + `tls_key`): one completion over HTTPS
//! behaves like one over the plaintext listener, the plaintext listener is unaffected, and a bad
//! certificate setup stops the boot with a message.

// Test target: `.unwrap()`/`.expect()`/`panic!` are assertions, not production code. See e2e.rs.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use beyond_ai::__test::key::{VirtualKey, mint};
use beyond_ai_test_support::ports::{GATEWAY_TLS_LISTEN, gateway_tls_port};
use common::*;
use std::path::PathBuf;
use std::time::Duration;

const BODY: &str = r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#;

/// A throwaway self-signed certificate for `127.0.0.3` / `localhost`, as PEM files.
struct Pems {
    dir: PathBuf,
    cert: PathBuf,
    key: PathBuf,
    cert_pem: String,
}

impl Pems {
    fn new(tag: &str) -> Self {
        let ck = rcgen::generate_simple_self_signed(vec![
            "127.0.0.3".to_string(),
            "localhost".to_string(),
        ])
        .unwrap();
        let dir = std::env::temp_dir().join(format!(
            "beyond-ai-tls-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let (cert, key) = (dir.join("cert.pem"), dir.join("key.pem"));
        let cert_pem = ck.cert.pem();
        std::fs::write(&cert, &cert_pem).unwrap();
        std::fs::write(&key, ck.key_pair.serialize_pem()).unwrap();
        Pems {
            dir,
            cert,
            key,
            cert_pem,
        }
    }

    fn lines(&self) -> String {
        format!(
            "{GATEWAY_TLS_LISTEN}tls_cert = \"{}\"\ntls_key = \"{}\"",
            self.cert.display(),
            self.key.display()
        )
    }
}

impl Drop for Pems {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

async fn tls_port(gw: &Gateway) -> u16 {
    for _ in 0..250 {
        if let Some(p) = gateway_tls_port(gw.pid()) {
            return p;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("the TLS listener never came up; log:\n{}", gw.log());
}

fn trusting(pems: &Pems, http1_only: bool) -> reqwest::Client {
    let root = reqwest::Certificate::from_pem(pems.cert_pem.as_bytes()).unwrap();
    let mut b = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .add_root_certificate(root);
    if http1_only {
        b = b.http1_only();
    }
    b.build().unwrap()
}

async fn complete(client: &reqwest::Client, base: &str, key: &str) -> reqwest::Response {
    client
        .post(format!("{base}/v1/chat/completions"))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .body(BODY)
        .send()
        .await
        .unwrap()
}

fn tenant_key(sk: &ed25519_dalek::SigningKey, tenant_id: u64) -> String {
    mint(
        &VirtualKey {
            tenant_id,
            vpc_id: 7,
            key_id: None,
        },
        1,
        sk,
    )
}

#[tokio::test]
async fn a_completion_over_https_matches_one_over_plaintext() {
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(3);
    let mock = MockUpstream::start(Mode::Json).await;
    let pems = Pems::new("ok");
    let gw = Gateway::builder(nats.port, &mock.authority(), &b64(&pubkey))
        .config_line(&pems.lines())
        .start()
        .await;
    let port = tls_port(&gw).await;
    let vkey = tenant_key(&sk, 42);
    let https = format!("https://127.0.0.3:{port}");

    // h2 by ALPN.
    let resp = complete(&trusting(&pems, false), &https, &vkey).await;
    assert_eq!(resp.version(), reqwest::Version::HTTP_2);
    assert_eq!(resp.status(), 200);
    assert!(resp.text().await.unwrap().contains("\"chatcmpl-mock\""));
    // The managed key swap happened exactly as on the plaintext listener.
    let cap = mock.captured().expect("mock received a request");
    assert_eq!(cap.authorization.as_deref(), Some("Bearer sk-pool-secret"));

    // http/1.1 by ALPN.
    let resp = complete(&trusting(&pems, true), &https, &vkey).await;
    assert_eq!(resp.version(), reqwest::Version::HTTP_11);
    assert_eq!(resp.status(), 200);
    resp.text().await.unwrap();

    // Metrics and the billing row come from the same service.
    wait_for_metric(&gw, "ai_tokens_total", "input", 22.0).await;
    gw.wait_for_log_line(&["ai.usage", "\"tenant_id\":42"])
        .await;

    // A client that does not trust the certificate is refused.
    let untrusted = test_client()
        .post(format!("{https}/v1/chat/completions"))
        .header("authorization", format!("Bearer {vkey}"))
        .body(BODY)
        .send()
        .await;
    assert!(
        untrusted.is_err(),
        "an untrusted client must fail the handshake"
    );

    // The plaintext listener is unchanged.
    let resp = complete(&test_client(), &gw.url(), &vkey).await;
    assert_eq!(resp.status(), 200);
    assert_eq!(mock.hits(), 3);
}

#[tokio::test]
async fn http1_only_when_downstream_h2c_is_off() {
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(4);
    let mock = MockUpstream::start(Mode::Json).await;
    let pems = Pems::new("h1");
    let gw = Gateway::builder(nats.port, &mock.authority(), &b64(&pubkey))
        .config_line("downstream_h2c = false")
        .config_line(&pems.lines())
        .start()
        .await;
    let port = tls_port(&gw).await;
    let resp = complete(
        &trusting(&pems, false),
        &format!("https://127.0.0.3:{port}"),
        &tenant_key(&sk, 1),
    )
    .await;
    assert_eq!(resp.version(), reqwest::Version::HTTP_11);
    assert_eq!(resp.status(), 200);
}

#[test]
fn a_bad_certificate_setup_stops_the_boot() {
    let good = Pems::new("good");
    let other = Pems::new("other");
    let pair = |cert: &std::path::Path, key: &std::path::Path| {
        format!(
            "{GATEWAY_TLS_LISTEN}tls_cert = \"{}\"\ntls_key = \"{}\"\n",
            cert.display(),
            key.display()
        )
    };
    let cases: Vec<(&str, String, &str)> = vec![
        (
            "missing cert",
            pair(std::path::Path::new("/nonexistent/cert.pem"), &good.key),
            "tls_cert",
        ),
        (
            "key that is not a key",
            pair(&good.cert, &good.cert),
            "tls_key",
        ),
        (
            "key from another cert",
            pair(&good.cert, &other.key),
            "do not work together",
        ),
        (
            "listener without files",
            GATEWAY_TLS_LISTEN.to_string(),
            "must be set together",
        ),
    ];
    for (what, lines, needle) in cases {
        let err = boot_refuses(&lines, Duration::from_secs(10))
            .unwrap_or_else(|| panic!("{what}: the gateway should have refused to start"));
        assert!(err.contains(needle), "{what}: {err}");
    }
}
