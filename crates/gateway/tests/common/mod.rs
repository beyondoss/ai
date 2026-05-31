//! e2e harness: a real `beyond-ai` binary, a real `nats-server` (JetStream KV backing the deny-set),
//! and a mock HTTP upstream that records what the gateway forwarded.
//!
//! Requires `nats-server` on PATH — run via `mise run test:integration:rs`.
//! Signing keys + pool keys are passed via the gateway's *config* (not NATS); NATS carries only the
//! deny-set. Every component picks a free port and cleans up on drop, so tests run in parallel.

#![allow(dead_code)]
// Test harness: `.unwrap()`/`.expect()`/`panic!` are assertions, not production code. See e2e.rs.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::io::Write;
use std::net::TcpListener as StdTcpListener;
use std::process::{Child, Command};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use base64::Engine;
use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::service::service_fn;
use hyper::{Request, Response};
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto;
use store::Connection;
use tokio::net::TcpListener;
use tokio::time::{sleep, timeout};
use tokio_rustls::TlsAcceptor;

/// Hand out a TCP port no other `free_port()` call in this test binary has returned.
///
/// Tests run as concurrent threads in **one** process, and closing a `bind(:0)` listener lets the OS
/// immediately re-hand that ephemeral port to the next `bind(:0)` — so two `free_port()` calls (a
/// gateway's `listen` + `metrics` ports, or two tests at once) can collide, and a component then
/// fails to bind → a *different* test flakes. A process-global reservation set makes every returned
/// port distinct within the run; binding fresh listeners on collision forces the OS off the just-used
/// port (it can't re-hand a port still held open) so the loop makes progress.
///
/// A residual TOCTOU window remains between returning a port and a *subprocess* (nats/gateway) binding
/// it, vs. other OS processes — unavoidable when the bind happens in another process. In-process
/// servers must instead bind `:0` and read the port back (see `MockUpstream`), which has no window.
pub fn free_port() -> u16 {
    use std::collections::HashSet;
    use std::sync::OnceLock;
    static USED: OnceLock<Mutex<HashSet<u16>>> = OnceLock::new();
    let used = USED.get_or_init(|| Mutex::new(HashSet::new()));

    let mut held = Vec::new();
    for _ in 0..1000 {
        let listener = StdTcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        if used.lock().unwrap_or_else(|p| p.into_inner()).insert(port) {
            return port; // `listener` drops here, freeing the port for the (sub)process to bind.
        }
        // Already handed out: keep this listener open so the next bind gets a different port, then
        // try again. The held listeners all drop at return, releasing those ports back to the OS.
        held.push(listener);
    }
    panic!("could not find an unused free port after 1000 attempts");
}

/// Base64 (standard) — used to put an Ed25519 public key into the gateway's `signing_keys` config.
pub fn b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Deterministic Ed25519 keypair: (raw 32-byte public key, signing key).
pub fn test_keypair(seed: u8) -> (Vec<u8>, ed25519_dalek::SigningKey) {
    let sk = ed25519_dalek::SigningKey::from_bytes(&[seed; 32]);
    (sk.verifying_key().to_bytes().to_vec(), sk)
}

async fn wait_for_port(port: u16, what: &str) {
    timeout(Duration::from_secs(20), async {
        loop {
            if tokio::net::TcpStream::connect(("127.0.0.1", port))
                .await
                .is_ok()
            {
                return;
            }
            sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("{what} did not come up on port {port}"));
}

// --- nats-server (JetStream) ------------------------------------------------

pub struct Nats {
    child: Child,
    pub port: u16,
    store_dir: std::path::PathBuf,
}

impl Nats {
    pub async fn start() -> Self {
        let port = free_port();
        let store_dir = std::env::temp_dir().join(format!("beyond-ai-nats-{port}"));
        let _ = std::fs::create_dir_all(&store_dir);
        let child = Command::new("nats-server")
            .args([
                "-js",
                "-a",
                "127.0.0.1",
                "-p",
                &port.to_string(),
                "-sd",
                store_dir.to_str().unwrap(),
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn nats-server (on PATH? run via mise)");
        let nats = Nats {
            child,
            port,
            store_dir,
        };
        wait_for_port(port, "nats-server").await;
        nats
    }
}

impl Nats {
    /// Kill the server mid-test (for fail-open coverage). Idempotent with `Drop`.
    pub fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for Nats {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = std::fs::remove_dir_all(&self.store_dir);
    }
}

pub async fn put_kv(nats_port: u16, key: &str, value: &[u8]) {
    open_writer(nats_port).await.put(key, value).await.unwrap();
}

pub async fn del_kv(nats_port: u16, key: &str) {
    open_writer(nats_port).await.delete(key).await.unwrap();
}

async fn open_writer(nats_port: u16) -> std::sync::Arc<dyn store::KvWriter> {
    let conn = store::NatsConnection::new(store::NatsConnectionConfig {
        url: format!("nats://127.0.0.1:{nats_port}"),
        creds: None,
        creds_file: None,
    });
    conn.connect().await.unwrap();
    let kv = conn
        .store_with_config(store::StoreConfig {
            name: "ai-gateway".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    kv.writer().expect("bucket is writable")
}

// --- mock upstream provider -------------------------------------------------

#[derive(Clone, Copy)]
pub enum Mode {
    /// OpenAI-shaped non-streaming JSON body.
    Json,
    /// OpenAI-shaped SSE stream with a terminal usage chunk.
    Sse,
    /// Anthropic-shaped non-streaming JSON body (`usage.input_tokens`).
    AnthropicJson,
    /// OpenAI-shaped SSE stream with >128 KiB of content *before* the usage chunk — forces the
    /// proxy's response-tail compaction path.
    SseLarge,
}

#[derive(Default, Clone)]
pub struct Captured {
    pub path: String,
    pub authorization: Option<String>,
    pub x_api_key: Option<String>,
    pub host: Option<String>,
    pub body: Vec<u8>,
}

pub struct MockUpstream {
    pub port: u16,
    captured: Arc<Mutex<Option<Captured>>>,
    task: tokio::task::JoinHandle<()>,
}

const CANNED_JSON: &str = r#"{"id":"chatcmpl-mock","object":"chat.completion","model":"gpt-4o-2024-08-06","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}}"#;

const CANNED_SSE: &str = "data: {\"id\":\"chatcmpl-mock\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":9}}\n\ndata: [DONE]\n\n";

const CANNED_ANTHROPIC_JSON: &str = r#"{"id":"msg_mock","type":"message","model":"claude-opus-4-8","content":[{"type":"text","text":"hi"}],"usage":{"input_tokens":13,"output_tokens":7}}"#;

/// An OpenAI SSE stream whose first chunk carries ~130 KiB of content, pushing the proxy's response
/// tail past `2 × USAGE_TAIL_CAP` (128 KiB) so it compacts at least once before the trailing usage
/// chunk arrives. The usage event must survive in the retained 64 KiB tail.
fn large_sse() -> String {
    let filler = "x".repeat(130 * 1024);
    format!(
        "data: {{\"id\":\"chatcmpl-mock\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{{\"delta\":{{\"content\":\"{filler}\"}}}}]}}\n\n\
         data: {{\"choices\":[],\"usage\":{{\"prompt_tokens\":5,\"completion_tokens\":9}}}}\n\n\
         data: [DONE]\n\n"
    )
}

/// The canned `(content-type, body)` for a mode. `SseLarge` allocates; the rest are static.
fn canned_body(mode: Mode) -> (&'static str, Bytes) {
    match mode {
        Mode::Json => (
            "application/json",
            Bytes::from_static(CANNED_JSON.as_bytes()),
        ),
        Mode::Sse => (
            "text/event-stream",
            Bytes::from_static(CANNED_SSE.as_bytes()),
        ),
        Mode::AnthropicJson => (
            "application/json",
            Bytes::from_static(CANNED_ANTHROPIC_JSON.as_bytes()),
        ),
        Mode::SseLarge => ("text/event-stream", Bytes::from(large_sse())),
    }
}

/// The protocol the gateway used to *reach the mock* — derived from the version hyper parsed off the
/// wire. Echoed back in `x-mock-proto`; since the gateway relays response headers untouched, the bench
/// client reads this to prove which protocol the gateway→upstream hop negotiated (H2 vs H1).
fn proto_label(version: hyper::Version) -> &'static str {
    match version {
        hyper::Version::HTTP_2 => "h2",
        _ => "http/1.1",
    }
}

/// Shared request handler for both the plaintext and TLS listeners: record what the gateway forwarded,
/// then return the canned body tagged with the negotiated protocol.
async fn mock_handle(
    req: Request<hyper::body::Incoming>,
    cap: Arc<Mutex<Option<Captured>>>,
    mode: Mode,
) -> Result<Response<Full<Bytes>>, std::convert::Infallible> {
    let version = req.version();
    let path = req.uri().path().to_string();
    // Pull the headers we record before consuming the body (which moves `req`).
    let (authorization, x_api_key, host) = {
        let h = req.headers();
        let get = |k: &str| h.get(k).and_then(|v| v.to_str().ok()).map(String::from);
        (get("authorization"), get("x-api-key"), get("host"))
    };
    let body = req
        .into_body()
        .collect()
        .await
        .map(|b| b.to_bytes().to_vec())
        .unwrap_or_default();
    *cap.lock().unwrap() = Some(Captured {
        path,
        authorization,
        x_api_key,
        host,
        body,
    });
    let (ct, payload) = canned_body(mode);
    Ok(Response::builder()
        .status(200)
        .header("content-type", ct)
        .header("x-mock-proto", proto_label(version))
        .body(Full::new(payload))
        .unwrap())
}

impl MockUpstream {
    pub async fn start(mode: Mode) -> Self {
        // Bind `:0` and read the port back, keeping the listener open the whole time — no
        // free_port()→rebind window for another test to slip into (this is an in-process server).
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let captured: Arc<Mutex<Option<Captured>>> = Arc::new(Mutex::new(None));
        let cap = captured.clone();
        let task = tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                let io = TokioIo::new(stream);
                let cap = cap.clone();
                tokio::spawn(async move {
                    let svc = service_fn(move |req| mock_handle(req, cap.clone(), mode));
                    let _ = hyper::server::conn::http1::Builder::new()
                        .serve_connection(io, svc)
                        .await;
                });
            }
        });
        MockUpstream {
            port,
            captured,
            task,
        }
    }

    /// Like [`start`], but terminates **TLS** and serves H1 *and* H2 on the one listener (protocol
    /// chosen by ALPN, via hyper-util's auto builder). Presents a throwaway self-signed cert, so the
    /// gateway must be pointed at it with `upstream_tls = true` and `upstream_verify_cert = false`.
    /// This is what lets the concurrency bench drive the gateway's real TLS+ALPN+H2 path against a
    /// local mock. Returns the mock; reach it at `authority()` (host `127.0.0.1`).
    pub async fn start_tls(mode: Mode) -> Self {
        // rustls 0.23 needs a process crypto provider; both ring and aws-lc are compiled in (so there's
        // no default), pick ring to match the gateway. Idempotent across multiple mocks in one process.
        let _ = rustls::crypto::ring::default_provider().install_default();

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        let ck = rcgen::generate_simple_self_signed(vec![
            "127.0.0.1".to_string(),
            "localhost".to_string(),
        ])
        .expect("self-signed cert");
        let certs = vec![ck.cert.der().clone()];
        let key = rustls::pki_types::PrivateKeyDer::Pkcs8(ck.key_pair.serialize_der().into());
        let mut tls = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(certs, key)
            .expect("server tls config");
        // Offer both so the gateway's ALPN preference decides: H2H1 → h2, H1 → http/1.1.
        tls.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
        let acceptor = TlsAcceptor::from(Arc::new(tls));

        let captured: Arc<Mutex<Option<Captured>>> = Arc::new(Mutex::new(None));
        let cap = captured.clone();
        let task = tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                let acceptor = acceptor.clone();
                let cap = cap.clone();
                tokio::spawn(async move {
                    let Ok(tls_stream) = acceptor.accept(stream).await else {
                        return;
                    };
                    let io = TokioIo::new(tls_stream);
                    let svc = service_fn(move |req| mock_handle(req, cap.clone(), mode));
                    // Auto builder: serves H2 or H1 per the negotiated ALPN.
                    let _ = auto::Builder::new(TokioExecutor::new())
                        .serve_connection(io, svc)
                        .await;
                });
            }
        });
        MockUpstream {
            port,
            captured,
            task,
        }
    }

    pub fn authority(&self) -> String {
        format!("127.0.0.1:{}", self.port)
    }

    pub fn captured(&self) -> Option<Captured> {
        self.captured.lock().unwrap().clone()
    }
}

impl Drop for MockUpstream {
    fn drop(&mut self) {
        self.task.abort();
    }
}

// --- the real beyond-ai binary ----------------------------------------------

pub struct Gateway {
    child: Child,
    pub port: u16,
    pub metrics_port: u16,
    config_path: std::path::PathBuf,
}

/// The managed pool key configured for a provider. Each provider gets a distinct value so a test
/// can assert the gateway swapped in the *right* one.
fn pool_key(provider: &str) -> &'static str {
    match provider {
        "openai" => "sk-pool-secret",
        "anthropic" => "sk-anthropic-pool",
        "fireworks" => "sk-fireworks-pool",
        "openrouter" => "sk-openrouter-pool",
        _ => "sk-unknown-pool",
    }
}

/// Builds a gateway config, choosing which providers are *configured* (authority → the mock + a
/// pool key). A managed request to a provider absent from this list has no pool key → 503.
pub struct GatewayBuilder {
    nats_port: u16,
    authority: String,
    signkey_b64: String,
    providers: Vec<&'static str>,
    snapshot_path: Option<String>,
    real_upstreams: bool,
    pool_key_overrides: Vec<(String, String)>,
    rate_limit_rps: Option<u32>,
    byo_rate_limit_rps: Option<u32>,
    /// Point at a TLS mock (`MockUpstream::start_tls`): `upstream_tls = true` + skip cert verification
    /// (the mock is self-signed), while still routing via `provider_authorities`. For the H2 bench.
    tls_upstream: bool,
    /// Override the gateway's `upstream_http2` (H2H1 vs H1 ALPN). `None` ⇒ leave the gateway default.
    upstream_http2: Option<bool>,
}

impl GatewayBuilder {
    /// Set which providers are configured. Defaults to `["openai", "fireworks"]`.
    pub fn providers(mut self, providers: &[&'static str]) -> Self {
        self.providers = providers.to_vec();
        self
    }

    /// Point the gateway at the **real** provider hosts over TLS (the `route::KNOWN_PROVIDERS`
    /// defaults), instead of the plaintext mock. Used by the live smoke tests (`tests/smoke.rs`):
    /// no authority overrides, no pool keys, no signing keys — smoke traffic is BYO (the caller's
    /// real provider token, passed through), so none of that is needed.
    pub fn real_upstreams(mut self) -> Self {
        self.real_upstreams = true;
        self
    }

    /// Set the managed pool key for a provider by name — in `real_upstreams` mode this is the *real*
    /// provider key the gateway swaps in for a managed (`bai_…`) request. Combine with a signing key
    /// (the `signkey_b64` passed to `builder`) to smoke-test the full managed path against the real
    /// provider.
    pub fn pool_key(mut self, provider: &str, key: &str) -> Self {
        self.pool_key_overrides
            .push((provider.to_string(), key.to_string()));
        self
    }

    /// Point the gateway at an on-disk deny-set snapshot. Pass the same path to two `start()` calls
    /// to model a restart that reloads from disk.
    pub fn snapshot_path(mut self, path: impl Into<String>) -> Self {
        self.snapshot_path = Some(path.into());
        self
    }

    /// Override the per-credential request-rate ceiling (requests/sec). The harness default leaves
    /// the gateway's own generous default (100) in place; set a small value to exercise the 429 path.
    pub fn rate_limit_rps(mut self, rps: u32) -> Self {
        self.rate_limit_rps = Some(rps);
        self
    }

    /// Override the aggregate BYO request-rate ceiling (requests/sec). `0` disables that tier so a
    /// per-credential 429 test isn't perturbed by the shared BYO bucket.
    pub fn byo_rate_limit_rps(mut self, rps: u32) -> Self {
        self.byo_rate_limit_rps = Some(rps);
        self
    }

    /// Talk to the upstream over TLS without verifying its cert — for a `MockUpstream::start_tls`
    /// target (self-signed). The gateway still routes via `provider_authorities` (the mock), but with
    /// real TLS + ALPN, so the H2 path is exercised. Used by the concurrency bench.
    pub fn tls_upstream(mut self) -> Self {
        self.tls_upstream = true;
        self
    }

    /// Force the gateway's upstream ALPN: `true` ⇒ H2H1 (prefer H2), `false` ⇒ H1 only. The bench
    /// starts one gateway each way against the same TLS mock to compare them.
    pub fn upstream_http2(mut self, on: bool) -> Self {
        self.upstream_http2 = Some(on);
        self
    }

    pub async fn start(self) -> Gateway {
        let port = free_port();
        let metrics_port = free_port();
        let config_path = std::env::temp_dir().join(format!("beyond-ai-config-{port}.toml"));
        let nats_port = self.nats_port;
        // Scalars first, `[…]` tables last (TOML ordering).
        let tls = self.real_upstreams || self.tls_upstream;
        let mut cfg = format!(
            "listen = \"127.0.0.1:{port}\"\n\
             metrics_listen = \"127.0.0.1:{metrics_port}\"\n\
             nats_url = \"nats://127.0.0.1:{nats_port}\"\n\
             config_bucket = \"ai-gateway\"\n\
             upstream_tls = {tls}\n"
        );
        // TLS mock is self-signed → don't verify its cert (production always verifies).
        if self.tls_upstream {
            cfg.push_str("upstream_verify_cert = false\n");
        }
        if let Some(h2) = self.upstream_http2 {
            cfg.push_str(&format!("upstream_http2 = {h2}\n"));
        }
        if let Some(path) = &self.snapshot_path {
            cfg.push_str(&format!("snapshot_path = \"{path}\"\n"));
        }
        if let Some(rps) = self.rate_limit_rps {
            cfg.push_str(&format!("rate_limit_rps = {rps}\n"));
        }
        if let Some(rps) = self.byo_rate_limit_rps {
            cfg.push_str(&format!("byo_rate_limit_rps = {rps}\n"));
        }
        if self.real_upstreams {
            // Real-host smoke mode: built-in provider defaults (no authority overrides). For a
            // *managed* smoke we still write the caller-supplied pool key(s) — the real provider key
            // the gateway swaps in — and the signing key the minted virtual key verifies against.
            // With neither set, this is a BYO smoke (the caller's token passes through).
            if !self.pool_key_overrides.is_empty() {
                cfg.push_str("\n[pool_keys]\n");
                for (p, k) in &self.pool_key_overrides {
                    cfg.push_str(&format!("{p} = \"{k}\"\n"));
                }
            }
            if !self.signkey_b64.is_empty() {
                cfg.push_str(&format!("\n[signing_keys]\n1 = \"{}\"\n", self.signkey_b64));
            }
        } else {
            // Every configured provider points at the one mock upstream...
            cfg.push_str("\n[provider_authorities]\n");
            for p in &self.providers {
                cfg.push_str(&format!("{p} = \"{}\"\n", self.authority));
            }
            // ...with a distinct pool key per provider so key-swap assertions can tell them apart.
            cfg.push_str("\n[pool_keys]\n");
            for p in &self.providers {
                cfg.push_str(&format!("{p} = \"{}\"\n", pool_key(p)));
            }
            cfg.push_str(&format!("\n[signing_keys]\n1 = \"{}\"\n", self.signkey_b64));
        }
        std::fs::File::create(&config_path)
            .unwrap()
            .write_all(cfg.as_bytes())
            .unwrap();

        let child = Command::new(env!("CARGO_BIN_EXE_beyond-ai"))
            .arg("run")
            .arg("-c")
            .arg(&config_path)
            .env(
                "AI_LOG",
                std::env::var("AI_LOG").unwrap_or_else(|_| "warn".into()),
            )
            .spawn()
            .expect("spawn beyond-ai");
        let gw = Gateway {
            child,
            port,
            metrics_port,
            config_path,
        };
        wait_for_port(port, "beyond-ai").await;
        gw
    }
}

impl Gateway {
    /// Start the gateway pointed at `nats` (deny-set) + the mock upstream, configuring the OpenAI
    /// and Fireworks providers. Signing key + pool key come from config (mirrors production: NATS
    /// holds only the deny-set). For other provider sets use [`Gateway::builder`].
    pub async fn start(nats_port: u16, openai_authority: &str, signkey_b64: &str) -> Self {
        Gateway::builder(nats_port, openai_authority, signkey_b64)
            .start()
            .await
    }

    /// A configurable gateway (which providers exist, etc.). Defaults match [`Gateway::start`].
    pub fn builder(nats_port: u16, authority: &str, signkey_b64: &str) -> GatewayBuilder {
        GatewayBuilder {
            nats_port,
            authority: authority.to_string(),
            signkey_b64: signkey_b64.to_string(),
            providers: vec!["openai", "fireworks"],
            snapshot_path: None,
            real_upstreams: false,
            pool_key_overrides: Vec::new(),
            rate_limit_rps: None,
            byo_rate_limit_rps: None,
            tls_upstream: false,
            upstream_http2: None,
        }
    }

    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    pub async fn metrics(&self) -> String {
        reqwest::get(format!("http://127.0.0.1:{}/metrics", self.metrics_port))
            .await
            .unwrap()
            .text()
            .await
            .unwrap()
    }

    /// GET a path on the admin/metrics listener, returning `(status, body)`. Used to probe
    /// `/livez` and `/readyz` (which live on `metrics_port`, alongside `/metrics`).
    pub async fn admin_get(&self, path: &str) -> (u16, String) {
        let resp = reqwest::get(format!("http://127.0.0.1:{}{path}", self.metrics_port))
            .await
            .unwrap();
        let status = resp.status().as_u16();
        (status, resp.text().await.unwrap())
    }
}

impl Drop for Gateway {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = std::fs::remove_file(&self.config_path);
    }
}

// --- assertions -------------------------------------------------------------

pub fn parse_metric(metrics: &str, name: &str, label_value: &str) -> f64 {
    metrics
        .lines()
        .find(|l| l.starts_with(name) && l.contains(label_value))
        .and_then(|l| l.rsplit(' ').next())
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0)
}

pub async fn wait_for_metric(gw: &Gateway, name: &str, label: &str, min: f64) {
    let r = timeout(Duration::from_secs(5), async {
        loop {
            if parse_metric(&gw.metrics().await, name, label) >= min {
                return;
            }
            sleep(Duration::from_millis(150)).await;
        }
    })
    .await;
    assert!(r.is_ok(), "metric {name}{{{label}}} never reached {min}");
}

pub async fn wait_for_status<F, Fut>(want: u16, mut f: F)
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = u16>,
{
    let r = timeout(Duration::from_secs(10), async {
        loop {
            if f().await == want {
                return;
            }
            sleep(Duration::from_millis(150)).await;
        }
    })
    .await;
    assert!(r.is_ok(), "status never became {want}");
}
