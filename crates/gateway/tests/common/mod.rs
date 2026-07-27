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
use std::process::{Child, Command, Stdio};
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

/// The directory this test *run* reserves ports in, shared by every process in the run.
///
/// Keyed on `NEXTEST_RUN_ID` so concurrent runs never see each other's reservations and a finished
/// run leaves nothing that shrinks the next one's pool. Falls back to the pid under plain
/// `cargo test`, where a single process makes the in-memory set sufficient anyway.
fn port_reservation_dir() -> Option<std::path::PathBuf> {
    let run =
        std::env::var("NEXTEST_RUN_ID").unwrap_or_else(|_| format!("pid-{}", std::process::id()));
    let dir = std::env::temp_dir().join(format!("beyond-ai-ports-{run}"));
    let made = std::fs::create_dir_all(&dir).ok().map(|()| dir);
    // Sweep reservations from runs that ended (killed job, panicked harness, plain `cargo test` pid
    // reuse). Best-effort and non-fatal: the only cost of a missed sweep is a stale directory, and
    // the only cost of a failed removal is that it stays one round longer.
    sweep_stale_port_dirs();
    made
}

/// Remove `beyond-ai-ports-*` directories older than an hour, skipping this run's own.
fn sweep_stale_port_dirs() {
    let Ok(entries) = std::fs::read_dir(std::env::temp_dir()) else {
        return;
    };
    let hour_ago = std::time::SystemTime::now() - Duration::from_secs(3600);
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.starts_with("beyond-ai-ports-") {
            continue;
        }
        let stale = entry
            .metadata()
            .and_then(|m| m.modified())
            .is_ok_and(|t| t < hour_ago);
        if stale {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
}

/// Hand out a TCP port no other `free_port()` call in this test **run** has returned.
///
/// Two layers, because there are two ways to collide:
///
/// - **Within a process**, closing a `bind(:0)` listener lets the OS immediately re-hand that
///   ephemeral port to the next `bind(:0)`, so a gateway's `listen` and `metrics` ports can come
///   back identical. A process-global set makes every returned port distinct, and holding the
///   colliding listeners open forces the OS onto a different port so the loop makes progress.
/// - **Across processes**, which is the case that actually bites in CI. The comment here used to say
///   "tests run as concurrent threads in one process" — true under `cargo test`, and false under
///   `cargo nextest`, which CI uses and which reports `NEXTEST_EXECUTION_MODE=process-per-test`. Each
///   test therefore had its *own* empty reservation set, so the guard coordinated nothing between
///   them: two tests could be handed the same port inside their TOCTOU windows, and whichever
///   subprocess bound second failed to start — surfacing as a 20s `wait_for_port` timeout in a test
///   that had done nothing wrong. A reservation file created with `create_new` in a run-scoped
///   directory closes that: the create is atomic, so exactly one process wins a given port.
///
/// The residual window — between returning a port and the *subprocess* binding it — is unavoidable
/// when the bind happens elsewhere, but it is now only racing processes outside this test run.
/// In-process servers should still bind `:0` and read the port back (see `MockUpstream`), which has
/// no window at all.
pub fn free_port() -> u16 {
    use std::collections::HashSet;
    use std::sync::OnceLock;
    static USED: OnceLock<Mutex<HashSet<u16>>> = OnceLock::new();
    static DIR: OnceLock<Option<std::path::PathBuf>> = OnceLock::new();
    let used = USED.get_or_init(|| Mutex::new(HashSet::new()));
    let dir = DIR.get_or_init(port_reservation_dir);

    let mut held = Vec::new();
    for _ in 0..1000 {
        let listener = StdTcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        if !used.lock().unwrap_or_else(|p| p.into_inner()).insert(port) {
            // Already handed out in this process: keep this listener open so the next bind gets a
            // different port. The held listeners all drop at return, releasing those ports.
            held.push(listener);
            continue;
        }
        // Claim it for the whole run. `create_new` is atomic, so a concurrent test process racing
        // for the same port loses here rather than at `bind` time in a subprocess.
        let claimed = match dir {
            Some(d) => std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(d.join(port.to_string()))
                .is_ok(),
            // No shared directory (read-only tmp, say): fall back to the in-process guard, which is
            // what this always used to be.
            None => true,
        };
        if claimed {
            return port; // `listener` drops here, freeing the port for the (sub)process to bind.
        }
        held.push(listener);
    }
    panic!("could not find an unused free port after 1000 attempts");
}

/// How much of a gateway's log to retain. Half is dropped when it fills, so the buffer stays within
/// this bound while always holding the most recent lines — which is what every assertion reads.
const LOG_CAPTURE_CAP: usize = 512 * 1024;

/// A NATS port for a gateway that never touches the deny-set.
///
/// The deny-set is the *only* thing the gateway reads from NATS, and it fails open — an unreachable
/// server means an empty deny-set and a retrying background watcher, which is exactly right for a
/// test that denies nobody. Auth, pool keys and routing all come from config.
///
/// Worth having because the alternative is not free: `Nats::start()` spawns a real JetStream server
/// per test, and a suite that starts twenty of them it never queries is spending a CI runner's
/// memory and disk on nothing. Tests that *do* exercise the deny-set still use `Nats::start()`.
pub fn unused_nats_port() -> u16 {
    free_port()
}

/// An HTTP client that cannot hang.
///
/// `reqwest::Client::new()` has **no** timeout, so a request that stalls stalls the test, and under
/// nextest that costs the per-test terminate budget (180s) plus a retry before anyone learns which
/// test it was — and a job killed that way uploads no log at all. A bounded client turns the same
/// stall into a named failure in seconds.
///
/// 30s is far above any legitimate local round-trip here (the slowest fixtures are ~600 KiB streams
/// over loopback) while still being an order of magnitude below the harness's own patience.
pub fn test_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .expect("build test client")
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

/// Connect to the test NATS and open the deny-set bucket, **bounded**.
///
/// Both steps are unbounded on their own and can wait forever rather than fail. `wait_for_port` only
/// proves the TCP listener is accepting; JetStream may still be initialising, and creating the KV
/// bucket then blocks until it is ready — on a fast local disk that is instant, which is exactly the
/// kind of difference that turns into an unkillable CI job and no log. async-nats also retries
/// reconnects indefinitely by design, so a server that dies mid-test would hang here too.
///
/// 20s is generous for a local JetStream that has already opened its port; the point is that the
/// failure is named and finite rather than silent and infinite.
async fn open_writer(nats_port: u16) -> std::sync::Arc<dyn store::KvWriter> {
    let conn = store::NatsConnection::new(store::NatsConnectionConfig {
        url: format!("nats://127.0.0.1:{nats_port}"),
        creds: None,
        creds_file: None,
    });
    timeout(Duration::from_secs(20), conn.connect())
        .await
        .unwrap_or_else(|_| {
            panic!(
                "nats-server on {nats_port} accepted a connection but never completed the handshake"
            )
        })
        .unwrap();
    let kv = timeout(
        Duration::from_secs(20),
        conn.store_with_config(store::StoreConfig {
            name: "ai-gateway".into(),
            ..Default::default()
        }),
    )
    .await
    .unwrap_or_else(|_| panic!("JetStream on {nats_port} never produced the ai-gateway bucket"))
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
    /// Anthropic-shaped SSE stream long enough to push `message_start` out of the retained tail.
    /// The input and cache token counts live on that first event, so this is the only fixture that
    /// can prove the proxy still meters them — see [`anthropic_sse_large`].
    AnthropicSseLarge,
    /// Always reply with this HTTP status and a small JSON error body — for circuit-breaker tests
    /// (5xx trips the breaker; 4xx/429 do not).
    Status(u16),
    /// Kill any request that is **not the first on its connection**, without answering it.
    ///
    /// This is pingora's *reused-connection* failure, produced deterministically. Keying on
    /// position-within-the-connection rather than a global request counter is what makes it
    /// reliable: how many connections pingora opens, and which request lands on which, varies with
    /// load. A global "kill request 2" fired on a *fresh* connection roughly half the time once the
    /// suite's other tests were running alongside — and a fresh-connection failure is not retryable,
    /// so the test failed for a reason that had nothing to do with what it was testing.
    ///
    /// The retry this provokes is one pingora decides on by itself, via its default
    /// `error_while_proxy`, without ever calling `fail_to_connect` — which is precisely why the
    /// gateway's breaker ledger lives in `upstream_peer` rather than there.
    CloseOnReusedConnection,
    /// Hold the request open for this many milliseconds before answering — long enough for a client
    /// to give up first. The only way to produce a *downstream* abort while the upstream is still
    /// healthy, which is the distinction the breaker has to draw.
    Slow(u64),
}

#[derive(Default, Clone)]
pub struct Captured {
    /// The forwarded path **including** any query string, exactly as the upstream received it.
    pub path: String,
    pub authorization: Option<String>,
    pub x_api_key: Option<String>,
    pub host: Option<String>,
    /// The gateway's model-routing header. Must always be `None`: it is ours, and stripping it is
    /// asserted rather than assumed, because a leaked internal header is the kind of thing nobody
    /// notices until a provider starts rejecting it.
    pub beyond_model: Option<String>,
    pub body: Vec<u8>,
}

pub struct MockUpstream {
    pub port: u16,
    captured: Arc<Mutex<Option<Captured>>>,
    hits: Arc<std::sync::atomic::AtomicUsize>,
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

/// Input + cache tokens the [`Mode::AnthropicSseLarge`] fixture reports on `message_start`. Asserted
/// by the e2e test, which is the only thing that can prove they survived the tail compaction.
pub const ANTHROPIC_LARGE_INPUT_TOKENS: u64 = 5000;
pub const ANTHROPIC_LARGE_CACHE_READ_TOKENS: u64 = 4000;
pub const ANTHROPIC_LARGE_OUTPUT_TOKENS: u64 = 2500;

/// A realistic Anthropic SSE stream: `message_start` carrying input + cache tokens, then enough
/// `content_block_delta` events to exceed `2 × USAGE_TAIL_CAP`, then the terminal `message_delta`
/// carrying the output count.
///
/// The shape matters. Anthropic splits the usage facts across the **first** and **last** events, so
/// a tail-only tap keeps the output count and silently drops input and cache — the fixture is built
/// to be long enough for exactly that to happen (~600 KiB, i.e. a routine 2500-token reply, since
/// Anthropic spends ~110 bytes of framing per delta).
fn anthropic_sse_large() -> String {
    let mut s = format!(
        "event: message_start\n\
         data: {{\"type\":\"message_start\",\"message\":{{\"id\":\"msg_mock\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-opus-4-8\",\"content\":[],\"usage\":{{\"input_tokens\":{ANTHROPIC_LARGE_INPUT_TOKENS},\"output_tokens\":1,\"cache_read_input_tokens\":{ANTHROPIC_LARGE_CACHE_READ_TOKENS},\"cache_creation_input_tokens\":100}}}}}}\n\n"
    );
    while s.len() < 600 * 1024 {
        s.push_str(
            "event: content_block_delta\n\
             data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" token\"}}\n\n",
        );
    }
    s.push_str(&format!(
        "event: message_delta\n\
         data: {{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":\"end_turn\"}},\"usage\":{{\"output_tokens\":{ANTHROPIC_LARGE_OUTPUT_TOKENS}}}}}\n\n\
         event: message_stop\ndata: {{\"type\":\"message_stop\"}}\n\n"
    ));
    s
}

/// The canned `(content-type, body)` for a mode. The `*Large` modes allocate; the rest are static.
fn canned_body(mode: Mode) -> (&'static str, Bytes) {
    match mode {
        // A slow reply, and the surviving requests of a close-on-Nth mock, are ordinary successes.
        Mode::Json | Mode::Slow(_) | Mode::CloseOnReusedConnection => (
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
        Mode::AnthropicSseLarge => ("text/event-stream", Bytes::from(anthropic_sse_large())),
        // The status is applied by `mock_handle`; the body is a stock error shape.
        Mode::Status(_) => (
            "application/json",
            Bytes::from_static(br#"{"error":{"message":"mock"}}"#),
        ),
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
    hits: Arc<std::sync::atomic::AtomicUsize>,
    mode: Mode,
    // `on_conn`: how many requests this **connection** has already served. 0 ⇒ fresh connection.
    on_conn: usize,
) -> Result<Response<Full<Bytes>>, std::io::Error> {
    hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let version = req.version();
    // Path **and query**. Recording only `uri.path()` meant the harness was structurally blind to the
    // query string: no test could tell whether the gateway forwarded `?api-version=…` (which Azure
    // OpenAI requires on every call), dropped it, or mangled it. Existing assertions compare against
    // query-less paths and are unaffected.
    let path = req
        .uri()
        .path_and_query()
        .map(|pq| pq.as_str())
        .unwrap_or_else(|| req.uri().path())
        .to_string();
    // Pull the headers we record before consuming the body (which moves `req`).
    let (authorization, x_api_key, host, beyond_model) = {
        let h = req.headers();
        let get = |k: &str| h.get(k).and_then(|v| v.to_str().ok()).map(String::from);
        (
            get("authorization"),
            get("x-api-key"),
            get("host"),
            get("x-beyond-model"),
        )
    };
    let body = req
        .into_body()
        .collect()
        .await
        .map(|b| b.to_bytes().to_vec())
        .unwrap_or_default();
    // Kill only *after* the request body has been fully read. Killing on the head instead raced the
    // gateway's body write and surfaced as `Upstream WriteError … Broken pipe`, which pingora does
    // not retry — correctly, since it cannot know how much the upstream consumed. Draining first
    // puts the failure squarely on the response-header read, which is the `ReusedOnly` shape this
    // mode exists to produce.
    if matches!(mode, Mode::CloseOnReusedConnection) && on_conn > 0 {
        return Err(std::io::Error::other("mock closing a reused connection"));
    }
    *cap.lock().unwrap() = Some(Captured {
        path,
        authorization,
        x_api_key,
        host,
        beyond_model,
        body,
    });
    // A slow upstream is still a *working* upstream; the point is to be slower than the client's
    // patience, so the client hangs up first.
    if let Mode::Slow(ms) = mode {
        sleep(Duration::from_millis(ms)).await;
    }
    let (ct, payload) = canned_body(mode);
    let status = match mode {
        Mode::Status(s) => s,
        _ => 200,
    };
    Ok(Response::builder()
        .status(status)
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
        let hits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let cap = captured.clone();
        let hit_counter = hits.clone();
        let task = tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                let io = TokioIo::new(stream);
                let cap = cap.clone();
                let hit_counter = hit_counter.clone();
                tokio::spawn(async move {
                    // Per-connection request index: `fetch_add` returns how many this connection has
                    // already served, so `0` means "first on a fresh connection".
                    let on_conn = Arc::new(std::sync::atomic::AtomicUsize::new(0));
                    let svc = service_fn(move |req| {
                        let n = on_conn.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        mock_handle(req, cap.clone(), hit_counter.clone(), mode, n)
                    });
                    let _ = hyper::server::conn::http1::Builder::new()
                        .serve_connection(io, svc)
                        .await;
                });
            }
        });
        MockUpstream {
            port,
            captured,
            hits,
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
        let hits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let cap = captured.clone();
        let hit_counter = hits.clone();
        let task = tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                let acceptor = acceptor.clone();
                let cap = cap.clone();
                let hit_counter = hit_counter.clone();
                tokio::spawn(async move {
                    let Ok(tls_stream) = acceptor.accept(stream).await else {
                        return;
                    };
                    let io = TokioIo::new(tls_stream);
                    let on_conn = Arc::new(std::sync::atomic::AtomicUsize::new(0));
                    let svc = service_fn(move |req| {
                        let n = on_conn.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        mock_handle(req, cap.clone(), hit_counter.clone(), mode, n)
                    });
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
            hits,
            task,
        }
    }

    pub fn authority(&self) -> String {
        format!("127.0.0.1:{}", self.port)
    }

    pub fn captured(&self) -> Option<Captured> {
        self.captured.lock().unwrap().clone()
    }

    /// Total requests the mock has received — used to prove an open circuit breaker stops requests
    /// from reaching the upstream at all.
    pub fn hits(&self) -> usize {
        self.hits.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Drop for MockUpstream {
    fn drop(&mut self) {
        self.task.abort();
    }
}

// --- the real beyond-ai binary ----------------------------------------------

pub struct Gateway {
    /// The child's stderr, drained by a background thread. Structured JSON, one object per line —
    /// including the `ai.usage` billing rows, which exist nowhere else.
    log: Arc<Mutex<String>>,
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
    /// Override the per-provider circuit-breaker threshold (failures in the window before opening).
    /// `None` ⇒ leave the gateway default; `Some(0)` disables the breaker.
    circuit_breaker_threshold: Option<u32>,
    /// Per-provider authority overrides, for a topology with more than one upstream — a failover
    /// test needs a live mock and a dead port at the same time, which the single `authority` cannot
    /// express. Falls back to `authority` for any provider not named here.
    authority_overrides: Vec<(String, String)>,
    /// Override the proxy's tokio worker-thread count. `None` ⇒ the gateway default (one per core);
    /// `Some(1)` reproduces Pingora's single-threaded default, which is what the scaling bench
    /// compares against.
    worker_threads: Option<usize>,
}

impl GatewayBuilder {
    /// Set which providers are configured. Defaults to `["openai", "fireworks"]`.
    pub fn providers(mut self, providers: &[&'static str]) -> Self {
        self.providers = providers.to_vec();
        self
    }

    /// Point one provider at its own authority, instead of the shared mock. Use for model-routing
    /// tests, where candidates must resolve to *different* upstreams — typically one live mock and
    /// one unbound port, which refuses instantly and so makes failover deterministic and fast.
    pub fn provider_authority(mut self, provider: &str, authority: &str) -> Self {
        self.authority_overrides
            .push((provider.to_string(), authority.to_string()));
        self
    }

    /// An authority nothing is listening on: connecting gets ECONNREFUSED immediately, with no
    /// timeout to wait out. The port is leased and dropped by `free_port`, so it is unbound.
    pub fn dead_authority() -> String {
        format!("127.0.0.1:{}", free_port())
    }

    /// Pin the proxy's worker-thread count. Used by the scaling bench to stand a single-threaded
    /// gateway (Pingora's own default) next to a per-core one.
    pub fn worker_threads(mut self, threads: usize) -> Self {
        self.worker_threads = Some(threads);
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

    /// Set the per-provider circuit-breaker failure threshold (a tight window/reset are written too,
    /// so the breaker trips fast in-test). `0` disables it.
    pub fn circuit_breaker_threshold(mut self, threshold: u32) -> Self {
        self.circuit_breaker_threshold = Some(threshold);
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
        if let Some(threads) = self.worker_threads {
            cfg.push_str(&format!("worker_threads = {threads}\n"));
        }
        if let Some(threshold) = self.circuit_breaker_threshold {
            // Tight window + reset so the test trips and recovers quickly.
            cfg.push_str(&format!(
                "circuit_breaker_threshold = {threshold}\n\
                 circuit_breaker_window_secs = 60\n\
                 circuit_breaker_reset_secs = 1\n"
            ));
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
            // Authority overrides still apply in real-upstream mode, so a smoke test can point one
            // provider at a dead port while the rest stay real — which is what proves a *live*
            // failover rather than a mocked one.
            if !self.authority_overrides.is_empty() {
                cfg.push_str("\n[provider_authorities]\n");
                for (name, authority) in &self.authority_overrides {
                    cfg.push_str(&format!("{name} = \"{authority}\"\n"));
                }
            }
        } else {
            // Every configured provider points at the one mock upstream...
            cfg.push_str("\n[provider_authorities]\n");
            for p in &self.providers {
                let authority = self
                    .authority_overrides
                    .iter()
                    .find(|(name, _)| name == p)
                    .map(|(_, a)| a.as_str())
                    .unwrap_or(&self.authority);
                cfg.push_str(&format!("{p} = \"{authority}\"\n"));
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

        let mut child = Command::new(env!("CARGO_BIN_EXE_beyond-ai"))
            .arg("run")
            .arg("-c")
            .arg(&config_path)
            .env(
                "AI_LOG",
                // `info` so the `ai.usage` billing rows reach the captured log. Still overridable.
                // `warn` for everything, `info` only for the billing target — the `ai.usage` rows
                // are the reason this is captured at all. Turning the whole gateway (and pingora
                // under it) up to `info` for every test produced far more output than any assertion
                // reads, on a path where each line is then held in memory below.
                std::env::var("AI_LOG").unwrap_or_else(|_| "warn,ai.usage=info".into()),
            )
            // Capture the child's output instead of letting it inherit ours. Two reasons: the
            // `ai.usage` rows are only observable this way (they are a log target, not a metric),
            // and a child that dies on a panic otherwise takes its own diagnosis with it — a
            // gateway crash then surfaces as a garbled assertion in whatever request raced it.
            //
            // **Both** streams: `init_tracing` installs `fmt::layer().json()`, whose default writer
            // is stdout, so that is where every structured log line (including `ai.usage`) goes.
            // Panics and pre-tracing boot failures go to stderr. Capturing only one loses half the
            // picture, and it is not the half you would guess.
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn beyond-ai");
        let log = Arc::new(Mutex::new(String::new()));
        let drain = |stream: Option<Box<dyn std::io::Read + Send>>| {
            let Some(stream) = stream else { return };
            let sink = Arc::clone(&log);
            std::thread::spawn(move || {
                use std::io::{BufRead, BufReader};
                for line in BufReader::new(stream).lines().map_while(Result::ok) {
                    if let Ok(mut buf) = sink.lock() {
                        // Bounded. A test binary holds one of these per gateway it starts, several
                        // run concurrently, and a chatty or wedged gateway would otherwise grow this
                        // without limit for the life of the test — memory pressure on a CI runner
                        // being a spectacularly unhelpful failure, since a reaped runner uploads no
                        // logs to explain itself. Assertions only ever read recent lines.
                        if buf.len() > LOG_CAPTURE_CAP {
                            let keep = buf.len() - LOG_CAPTURE_CAP / 2;
                            buf.drain(..keep);
                        }
                        buf.push_str(&line);
                        buf.push('\n');
                    }
                }
            });
        };
        drain(
            child
                .stdout
                .take()
                .map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
        );
        drain(
            child
                .stderr
                .take()
                .map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
        );
        let gw = Gateway {
            child,
            port,
            metrics_port,
            config_path,
            log,
        };
        wait_for_port(port, "beyond-ai").await;
        // The metrics/admin listener (`/livez`, `/readyz`, `/metrics`) binds on a *separate* port from
        // the proxy; wait for it too, or a test that probes it right after `start()` races the bind
        // (pre-existing flake in `health_endpoints_report_ready_on_the_metrics_listener`).
        wait_for_port(metrics_port, "beyond-ai-metrics").await;
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
            authority_overrides: Vec::new(),
            snapshot_path: None,
            real_upstreams: false,
            pool_key_overrides: Vec::new(),
            rate_limit_rps: None,
            byo_rate_limit_rps: None,
            tls_upstream: false,
            upstream_http2: None,
            circuit_breaker_threshold: None,
            worker_threads: None,
        }
    }

    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    /// Everything the gateway has logged so far.
    pub fn log(&self) -> String {
        self.log.lock().map(|l| l.clone()).unwrap_or_default()
    }

    /// Wait for a log line containing every one of `needles`, and return it.
    ///
    /// Takes a slice rather than one string because the interesting assertions are conjunctions —
    /// "an `ai.usage` row that names *this* provider" — and a single substring cannot express that
    /// without depending on field order.
    pub async fn wait_for_log_line(&self, needles: &[&str]) -> String {
        for _ in 0..200 {
            let log = self.log();
            if let Some(line) = log.lines().find(|l| needles.iter().all(|n| l.contains(n))) {
                return line.to_string();
            }
            sleep(Duration::from_millis(25)).await;
        }
        panic!(
            "no log line matched {needles:?} within 5s; captured log was:\n{}",
            self.log(),
        );
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
        // Retry briefly: the listener is bound (we waited for the port), but the app can answer a
        // connection with a transient non-200 for a few ms right after startup. Retry a handful of
        // times before giving up, so a startup-timing blip doesn't flake the probe.
        let url = format!("http://127.0.0.1:{}{path}", self.metrics_port);
        for attempt in 0..20 {
            match reqwest::get(&url).await {
                Ok(resp) => {
                    let status = resp.status().as_u16();
                    let body = resp.text().await.unwrap_or_default();
                    if status == 200 || attempt == 19 {
                        return (status, body);
                    }
                }
                Err(_) if attempt < 19 => {}
                Err(e) => panic!("admin_get {url} failed: {e}"),
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
        unreachable!("admin_get loop always returns")
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
