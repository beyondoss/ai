//! `serve --tls-listen`: the control protocol over rustls, behind every guard `--listen` has.
//!
//! The TLS listener hands each handshaken stream to the same `handle_connection` the plain listener
//! feeds, so what these tests prove is that the seam holds: a service-mode grant is still demanded,
//! the browser guard still runs, a client that does not speak TLS is dropped without an HTTP reply,
//! and a bad certificate setup stops the daemon before it serves anything.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::service::{Options, Service};
use common::{
    BIN, SpawnGuarded, serve_cmd, serve_service_cmd, spawn_model_server, turn_text,
    ws_read_until_response, ws_send,
};
use serde_json::json;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::{Connector, connect_async_tls_with_config};

// port-0: a child that serves keeps the socket and announces the port it got; one that is meant to
// fail to start never binds it.
const TLS_ADDR: &str = "127.0.0.1:0";

/// A throwaway self-signed certificate for `127.0.0.1` / `localhost`, as PEM files.
struct Pems {
    dir: tempfile::TempDir,
    cert: PathBuf,
    key: PathBuf,
    cert_der: Vec<u8>,
}

impl Pems {
    fn new() -> Self {
        let ck = rcgen::generate_simple_self_signed(vec![
            "127.0.0.1".to_string(),
            "localhost".to_string(),
        ])
        .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let (cert, key) = (dir.path().join("cert.pem"), dir.path().join("key.pem"));
        std::fs::write(&cert, ck.cert.pem()).unwrap();
        std::fs::write(&key, ck.key_pair.serialize_pem()).unwrap();
        Pems {
            dir,
            cert,
            key,
            cert_der: ck.cert.der().to_vec(),
        }
    }

    /// The `serve` arguments that turn the TLS listener on, at a port the kernel picks.
    fn args(&self) -> Vec<String> {
        vec![
            "--tls-listen".into(),
            TLS_ADDR.into(),
            "--tls-cert".into(),
            self.cert.to_string_lossy().into_owned(),
            "--tls-key".into(),
            self.key.to_string_lossy().into_owned(),
        ]
    }

    fn client_config(&self) -> Arc<rustls::ClientConfig> {
        let mut roots = rustls::RootCertStore::empty();
        roots
            .add(rustls::pki_types::CertificateDer::from(
                self.cert_der.clone(),
            ))
            .unwrap();
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        Arc::new(
            rustls::ClientConfig::builder_with_provider(provider)
                .with_safe_default_protocol_versions()
                .unwrap()
                .with_root_certificates(roots)
                .with_no_client_auth(),
        )
    }
}

/// The port of the TLS listener, from the line `serve` announces it with. It is printed just after
/// the plain listener's, which is what `spawn_listening` waits for, so poll the log for it.
fn tls_port(log: &Path) -> u16 {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let text = std::fs::read_to_string(log).unwrap_or_default();
        if let Some(addr) = text
            .lines()
            .find_map(|l| l.strip_prefix("serve: websocket tls listening on "))
            .and_then(|r| r.split_whitespace().next())
            && let Ok(a) = addr.parse::<std::net::SocketAddr>()
        {
            return a.port();
        }
        assert!(Instant::now() < deadline, "no TLS announcement in:\n{text}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

type Wss =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// Connect over `wss://`, trusting `pems`. Returns the HTTP status on a refusal, as the plain helper does.
async fn wss_connect(
    port: u16,
    pems: &Pems,
    session_id: &str,
    headers: &[(&str, &str)],
) -> Result<Wss, u16> {
    let mut req = format!("wss://127.0.0.1:{port}/_beyond/agent?session_id={session_id}")
        .into_client_request()
        .unwrap();
    for (k, v) in headers {
        req.headers_mut().insert(
            tokio_tungstenite::tungstenite::http::header::HeaderName::from_bytes(k.as_bytes())
                .unwrap(),
            v.parse().unwrap(),
        );
    }
    match connect_async_tls_with_config(
        req,
        None,
        false,
        Some(Connector::Rustls(pems.client_config())),
    )
    .await
    {
        Ok((ws, _)) => Ok(ws),
        Err(tokio_tungstenite::tungstenite::Error::Http(resp)) => Err(resp.status().as_u16()),
        Err(e) => panic!("wss connect failed without an HTTP status: {e}"),
    }
}

async fn replica(base: &str, pems: &Pems) -> (Service, u16) {
    let svc = Service::start_with(
        base,
        &["s1"],
        Options {
            extra_args: pems.args(),
            ..Options::default()
        },
    )
    .await;
    let port = tls_port(&svc.stderr_log);
    (svc, port)
}

#[tokio::test]
async fn a_wss_client_completes_a_turn_in_service_mode() {
    let pems = Pems::new();
    let (base, requests) = spawn_model_server(vec![turn_text("hello over tls")]);
    let (svc, port) = replica(&base, &pems).await;

    let token = svc.token("t1", "s1.alpha");
    let mut ws = wss_connect(port, &pems, "s1.alpha", &svc.header(&token))
        .await
        .unwrap();
    ws_send(&mut ws, json!({"type":"prompt","id":"p1","message":"hi"})).await;
    let frames = ws_read_until_response(&mut ws, "prompt").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    assert!(
        frames
            .iter()
            .any(|f| f.to_string().contains("hello over tls")),
        "{frames:#?}"
    );
    assert_eq!(
        requests.lock().unwrap().len(),
        1,
        "the model was called once"
    );

    // `--listen` is untouched: the same replica still serves a plain ws:// client.
    let mut plain = common::ws_connect_with_headers(
        svc.port,
        Some("s1.beta"),
        &svc.header(&svc.token("t1", "s1.beta")),
    )
    .await
    .unwrap();
    ws_send(&mut plain, json!({"type":"get_state","id":"g"})).await;
    let frames = ws_read_until_response(&mut plain, "get_state").await;
    assert_eq!(frames.last().unwrap()["success"], true);
}

#[tokio::test]
async fn the_grant_check_applies_over_tls() {
    let pems = Pems::new();
    let (base, _requests) = spawn_model_server(vec![]);
    let (svc, port) = replica(&base, &pems).await;

    // No grant: the same refusal as the plain listener gives.
    assert_eq!(
        wss_connect(port, &pems, "s1.alpha", &[]).await.err(),
        Some(401)
    );
    // A grant for another session id is refused too.
    let token = svc.token("t1", "s1.other");
    assert_eq!(
        wss_connect(port, &pems, "s1.alpha", &svc.header(&token))
            .await
            .err(),
        Some(
            common::ws_refusal(svc.port, Some("s1.alpha"), &svc.header(&token))
                .await
                .0
        )
    );
}

#[tokio::test]
async fn the_browser_guard_applies_over_tls() {
    let pems = Pems::new();
    let (base, _requests) = spawn_model_server(vec![]);
    let session_file = tempfile::NamedTempFile::new().unwrap();
    // The TLS port is on the child's stderr; the log file is how the test reads it back.
    let log = tempfile::NamedTempFile::new().unwrap();
    let mut cmd = serve_cmd(BIN, &base, &session_file.path().to_string_lossy());
    cmd.args(pems.args());
    let (_child, _plain) = common::spawn_listening_logged(&mut cmd, Some(log.reopen().unwrap()));
    let port = tls_port(log.path());

    // A foreign Origin is refused on the TLS listener exactly as on `--listen` (403)...
    let status = wss_connect(port, &pems, "web", &[("origin", "https://evil.example")])
        .await
        .err();
    assert_eq!(status, Some(403));
    // ...and a request with no Origin is a non-browser client and is let in.
    assert!(wss_connect(port, &pems, "web", &[]).await.is_ok());
}

#[tokio::test]
async fn a_plain_request_to_the_tls_port_is_dropped_without_a_reply() {
    use std::io::Write as _;
    let pems = Pems::new();
    let (base, _requests) = spawn_model_server(vec![]);
    let (svc, port) = replica(&base, &pems).await;

    let mut s = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    s.write_all(b"GET /_beyond/agent?session_id=x HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        .unwrap();
    let mut got = Vec::new();
    // EOF or a reset: both are a refusal. A timeout would mean the connection was left hanging.
    match s.read_to_end(&mut got) {
        Ok(_) => {}
        Err(e) => assert_ne!(
            e.kind(),
            std::io::ErrorKind::WouldBlock,
            "the connection was left open"
        ),
    }
    assert!(
        !String::from_utf8_lossy(&got).contains("HTTP/1.1"),
        "no HTTP reply on the TLS port: {got:?}"
    );

    // The listener is still serving.
    let token = svc.token("t1", "s1.alpha");
    assert!(
        wss_connect(port, &pems, "s1.alpha", &svc.header(&token))
            .await
            .is_ok()
    );
}

/// Run `serve` with a bad TLS setup and return its stderr, asserting it refused to start.
fn refused(extra: &[String]) -> String {
    let dir = tempfile::tempdir().unwrap();
    let shard = dir.path().join("shard");
    std::fs::create_dir_all(&shard).unwrap();
    let minter = common::grant::Minter::new(dir.path());
    let mut cmd = serve_service_cmd(
        BIN,
        "http://127.0.0.1:1",
        &minter.grant_key_flag(),
        Some(minter.seal_key()),
        &[("a", shard.as_path())],
    );
    cmd.args(extra).stderr(Stdio::piped());
    let out = cmd.spawn_guarded().wait_with_output().unwrap();
    let err = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        !out.status.success(),
        "expected a startup failure; stderr: {err}"
    );
    err
}

#[test]
fn a_bad_certificate_setup_stops_the_daemon_before_it_serves() {
    let good = Pems::new();
    let other = Pems::new();
    let args = |cert: &Path, key: &Path| {
        vec![
            "--tls-listen".to_string(),
            TLS_ADDR.to_string(),
            "--tls-cert".to_string(),
            cert.to_string_lossy().into_owned(),
            "--tls-key".to_string(),
            key.to_string_lossy().into_owned(),
        ]
    };
    let missing = Path::new("/nonexistent/cert.pem");
    assert!(refused(&args(missing, &good.key)).contains("--tls-cert"));
    assert!(refused(&args(&good.cert, missing)).contains("--tls-key"));
    // A certificate where the key should be.
    assert!(refused(&args(&good.cert, &good.cert)).contains("--tls-key"));
    // A key that belongs to another certificate.
    assert!(refused(&args(&good.cert, &other.key)).contains("do not work together"));
    // Flags given only in part.
    let partial = refused(&["--tls-listen".to_string(), TLS_ADDR.to_string()]);
    assert!(partial.contains("must be given together"), "{partial}");
    let _keep = (&good.dir, &other.dir);
}
