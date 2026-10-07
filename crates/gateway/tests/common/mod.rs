//! e2e harness: a real `beyond-ai` binary, a real `nats-server` (JetStream KV backing the deny-set
//! and allowance-set), and a mock HTTP upstream that records what the gateway forwarded.
//!
//! Requires `nats-server` on PATH — run via `mise run test:integration:rs`.
//! Signing keys + pool keys are passed via the gateway's *config* (not NATS); NATS carries the
//! deny-set, allowance-set, and capture-set. Every component binds a port the kernel picks and the
//! harness reads it back (`beyond_ai_test_support::ports`) — none is picked free, released and
//! rebound, so no other process can take one in between — and cleans up on drop, so tests run in
//! parallel.

#![allow(dead_code)]
// Test harness: `.unwrap()`/`.expect()`/`panic!` are assertions, not production code. See e2e.rs.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::io::Write;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use base64::Engine;
use bytes::Bytes;
use http_body_util::{BodyExt, Either, Full};
use hyper::service::service_fn;
use hyper::{Request, Response};
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto;
use store::Connection;
use tokio::net::TcpListener;
use tokio::time::{sleep, timeout};
use tokio_rustls::TlsAcceptor;

/// A per-process, per-call temporary file name: `<prefix>-<pid>-<n>.<ext>`.
fn unique_temp(prefix: &str, ext: &str) -> std::path::PathBuf {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!("{prefix}-{}-{n}.{ext}", std::process::id()))
}

/// How much of a gateway's log to retain. Half is dropped when it fills, so the buffer stays within
/// this bound while always holding the most recent lines — which is what every assertion reads.
const LOG_CAPTURE_CAP: usize = 512 * 1024;

/// A stated latency bound (a deny lands within 2 s), held exactly on an idle host and stretched on
/// a loaded one, where nothing local can meet a wall-clock promise. `round_trip` is
/// [`median_round_trip`] of an ordinary request through the same gateway, measured in the same run:
/// the bound is the larger of the stated one and [`LOAD_STRETCH`] such round trips. On an idle host
/// a round trip is ~35 ms (debug gateway, fresh client connection), so the stated bound is what
/// binds; under a load that makes every hop slow (600 ms round trips at a load average of 160), the
/// bound grows with it. A regression that ignores the event altogether (a deny that never
/// lands) still fails, at whichever bound is in force.
pub fn stretched(stated: Duration, round_trip: Duration) -> Duration {
    stated.max(round_trip * LOAD_STRETCH)
}

/// Round trips a stretched bound may take. A deny lands within one to three round trips at every
/// load measured; 40 leaves an idle host's bound at the stated one (40 x 35 ms is under 2 s), so a
/// deny that lands a second late there still fails.
pub const LOAD_STRETCH: u32 = 40;

/// The median of five timed runs of `op` (an ordinary request, for [`stretched`]).
pub async fn median_round_trip<F, Fut, T>(mut op: F) -> Duration
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = T>,
{
    let mut took = Vec::with_capacity(5);
    for _ in 0..5 {
        let start = std::time::Instant::now();
        let _ = op().await;
        took.push(start.elapsed());
    }
    took.sort();
    took[2]
}

/// How long a wait on a *condition* (a metric reaching a value, a log line appearing) may take
/// before it fails the test.
///
/// This is a failure detector, not a pass condition: a wait returns the moment its condition holds,
/// so on a passing run the bound costs nothing, and a short bound buys nothing but flakes. Under the
/// `gateway-stress` CI job every core runs a `yes` hog, and a gateway subprocess that must be
/// scheduled several times (a NATS round-trip, a watcher apply, a scrape) can take seconds to get
/// there — a 5 s bound failed there on unchanged code. The ceiling is the harness: nextest's `ci`
/// profile terminates a test after 180 s, and a test makes a few such waits, so 30 s each still
/// fails a genuinely stuck wait *by name* well before nextest kills it anonymously. It matches
/// [`test_client`]'s request timeout, for the same reason.
pub const CONDITION_BUDGET: Duration = Duration::from_secs(30);

/// How long a spawned server (`beyond-ai`, `nats-server`) may take to listen on its port.
///
/// A stall guard, not a performance bound: an idle host boots the gateway in ~0.1 s and 200 busy
/// loops on 16 cores slow that to ~1.3 s, but a mutation run on the same box (load 60-230, disk
/// writeback pinned at ~99% IO pressure while freshly linked 120 MB test binaries are exec'd)
/// stalled whole batches of starts past the old 20 s — every test starting a gateway in that
/// window failed at once. The gateway does no network or disk work before it binds (config read,
/// state build, `add_tcp`), so a start this slow is the host, not the code. A port another process
/// took is not waited out against this budget: it is detected and retried on fresh ports.
pub const STARTUP_BUDGET: Duration = Duration::from_secs(60);

/// A NATS port for a gateway that does not write deny/allowance keys of its own. (The name is
/// historical: it is the shared server's port, which the server chose itself.)
///
/// Allowance is fail-closed until the watcher stores a scan (empty = remaining-ok), so a closed
/// port would 402 every managed request. This starts **one** JetStream server per test process
/// (held in a `OnceLock` until exit) and returns its port. Tests that *write* KV still use
/// [`Nats::start()`] so they cannot see each other's `blackhole.*` / `allowance.*` keys.
pub fn unused_nats_port() -> u16 {
    static SERVER: OnceLock<Nats> = OnceLock::new();
    SERVER
        .get_or_init(|| Nats::spawn_ready(Nats::spawn_reaped, "beyond-ai-nats-shared"))
        .port
}

/// A TCP port nothing is listening on. The fail-closed allowance test uses this so the watcher
/// never seeds.
///
/// Held, not merely unbound: the port is bound by a socket that never listens, kept for the life
/// of this test process. A connect to it is refused at once (no listener), and no other process can
/// bind it meanwhile (the holder sets neither `SO_REUSEADDR` nor `SO_REUSEPORT`). A port picked free
/// and released was only dead until something else on the host took it — under a concurrent
/// cargo-mutants run, another run's mock or gateway — and then the "dead" candidate answered: the
/// failover tests' "a dead primary must be invisible" and "tried once" assertions failed.
pub fn closed_port() -> u16 {
    static HELD: Mutex<Vec<beyond_ai_test_support::ports::DeadPort>> = Mutex::new(Vec::new());
    let dead = beyond_ai_test_support::ports::DeadPort::bind();
    let port = dead.port();
    HELD.lock().unwrap_or_else(|p| p.into_inner()).push(dead);
    port
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

/// The `[id_signing_keys]` table, when there is one.
fn write_id_signing_toml(cfg: &mut String, keys: Option<&IdSigning>) {
    if let Some((keys, _)) = keys {
        cfg.push_str("\n[id_signing_keys]\n");
        for (kid, secret) in keys {
            cfg.push_str(&format!("{kid} = \"{}\"\n", b64(secret)));
        }
    }
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

// --- nats-server (JetStream) ------------------------------------------------

pub struct Nats {
    child: Child,
    /// The port it chose (`-p -1`) and reported (`--ports_file_dir`); `0` until [`Nats::spawn_ready`]
    /// has read it.
    pub port: u16,
    store_dir: std::path::PathBuf,
    /// The `server_name` it was started with, unique to this test process.
    name: String,
}

impl Nats {
    /// A store directory (holding the ports file directory) and a server name for one `nats-server`.
    fn layout(store_prefix: &str) -> (std::path::PathBuf, String) {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let name = format!("{store_prefix}-{}-{n}", std::process::id());
        let store_dir = std::env::temp_dir().join(&name);
        let _ = std::fs::remove_dir_all(&store_dir);
        let _ = std::fs::create_dir_all(Self::ports_dir_of(&store_dir));
        (store_dir, name)
    }

    /// Where `--ports_file_dir` writes the port the server chose.
    fn ports_dir_of(store_dir: &std::path::Path) -> std::path::PathBuf {
        store_dir.join("ports")
    }

    fn spawn(store_prefix: &str) -> Self {
        let (store_dir, name) = Self::layout(store_prefix);
        let child = Command::new("nats-server")
            .args(["-js", "-a", "127.0.0.1", "-p", "-1", "-n", &name])
            .arg("-sd")
            .arg(&store_dir)
            .arg("--ports_file_dir")
            .arg(Self::ports_dir_of(&store_dir))
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn nats-server (on PATH? run via mise)");
        Nats {
            child,
            port: 0,
            store_dir,
            name,
        }
    }

    /// Like [`Self::spawn`], for a server that outlives every `Drop`: the per-process shared one
    /// in [`unused_nats_port`], held in a `static`, whose destructor never runs. Under nextest every
    /// test is its own process, so each run leaked one `nats-server` per test (thousands a day on a
    /// busy machine). A shell watchdog polls this test process and stops the server, and removes its
    /// store, within a second of the process exiting, however it exits. The shell exits as soon as
    /// the server does, so a server that failed to start shows as an exited `child` here too.
    fn spawn_reaped(store_prefix: &str) -> Self {
        const WATCHDOG: &str = r#"nats-server -js -a 127.0.0.1 -p -1 --ports_file_dir "$2" -sd "$3" -n "$4" >/dev/null 2>&1 &
server=$!
echo "$server" > "$3/server.pid"
while kill -0 "$1" 2>/dev/null && kill -0 "$server" 2>/dev/null; do sleep 1; done
kill "$server" 2>/dev/null
wait "$server" 2>/dev/null
rm -rf "$3""#;
        let (store_dir, name) = Self::layout(store_prefix);
        let child = Command::new("sh")
            .args([
                "-c",
                WATCHDOG,
                "nats-watchdog",
                &std::process::id().to_string(),
            ])
            .arg(Self::ports_dir_of(&store_dir))
            .arg(&store_dir)
            .arg(&name)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn nats-server (on PATH? run via mise)");
        Nats {
            child,
            port: 0,
            store_dir,
            name,
        }
    }

    /// Spawn with `spawn` and return the server once it has reported the port it chose — which it
    /// writes only after it is listening. The port was never anyone else's, so there is nothing to
    /// lose to another process and nothing to respawn: a server that exits, or that is not up within
    /// [`STARTUP_BUDGET`], fails the test.
    fn spawn_ready(spawn: fn(&str) -> Self, store_prefix: &str) -> Self {
        let mut nats = spawn(store_prefix);
        let ports = Self::ports_dir_of(&nats.store_dir);
        let deadline = std::time::Instant::now() + STARTUP_BUDGET;
        loop {
            // The server's own pid: the child itself, or — under the watchdog shell — the one it
            // recorded.
            let server_pid = std::fs::read_to_string(nats.store_dir.join("server.pid"))
                .ok()
                .and_then(|p| p.trim().parse().ok())
                .unwrap_or_else(|| nats.child.id());
            if let Some(port) = beyond_ai_test_support::ports::nats_port_from(&ports, server_pid) {
                nats.port = port;
                return nats;
            }
            if let Ok(Some(status)) = nats.child.try_wait() {
                panic!(
                    "nats-server `{}` exited ({status}) before listening",
                    nats.name
                );
            }
            assert!(
                std::time::Instant::now() < deadline,
                "nats-server `{}` did not report a port within {STARTUP_BUDGET:?}",
                nats.name
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    pub async fn start() -> Self {
        tokio::task::spawn_blocking(|| Self::spawn_ready(Self::spawn, "beyond-ai-nats"))
            .await
            .expect("nats-server start task")
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
    /// Anthropic-shaped SSE stream (message_start / content_block_delta / message_delta).
    AnthropicSse,
    /// OpenAI-shaped SSE stream with >128 KiB of content *before* the usage chunk — forces the
    /// proxy's response-tail compaction path.
    SseLarge,
    /// Anthropic-shaped SSE stream long enough to push `message_start` out of the retained tail.
    /// The input and cache token counts live on that first event, so this is the only fixture that
    /// can prove the proxy still meters them — see [`anthropic_sse_large`].
    AnthropicSseLarge,
    /// Always reply with this HTTP status and a small JSON error body — for circuit-breaker tests
    /// (5xx trips the breaker; 4xx/429 do not). A 429 also carries `Retry-After` so relay tests can
    /// assert the gateway forwards it. OpenAI error envelope.
    Status(u16),
    /// Like [`Status`], but the body is an Anthropic error envelope (`type: error`).
    AnthropicStatus(u16),
    /// Anthropic-shaped non-streaming `tool_use` JSON (`stop_reason: tool_use`).
    AnthropicToolJson,
    /// Anthropic-shaped SSE `tool_use` stream (`input_json_delta` + `stop_reason: tool_use`).
    AnthropicToolSse,
    /// OpenAI-shaped non-streaming `tool_calls` JSON.
    OpenAiToolJson,
    /// OpenAI-shaped SSE `tool_calls` stream.
    OpenAiToolSse,
    /// Anthropic SSE `event: error`.
    AnthropicErrorSse,
    /// Anthropic SSE with a `thinking` block, then text, plus cache + thinking token counts.
    AnthropicThinkingSse,
    /// OpenAI SSE error chunk.
    OpenAiErrorSse,
    /// 429 (with `Retry-After`) when the presented credential contains this secret; 200 otherwise.
    /// Proves a key-walk actually sent the second pool key.
    ThrottleKey(&'static str),
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
    /// An OpenAI chat stream that sends [`STALL_DELTAS`] content deltas and then never sends another
    /// byte — no usage chunk, no `[DONE]`. The client reads what arrived and hangs up: a cancelled
    /// agent turn, with the upstream still healthy and still (in reality) billing.
    StallSse,
    /// The Anthropic twin: `message_start` (exact input/cache counts), [`STALL_DELTAS`] text deltas,
    /// then silence — no `message_delta`, so no output count.
    AnthropicStallSse,
    /// An OpenAI embeddings response: vectors first, then `model`, then `usage` (input only).
    Embeddings,
    /// Reply with exactly this status, content type, and body — for fixtures that belong to one
    /// test file (a provider's real stream shape, an error body with no `error` key).
    Raw(u16, &'static str, &'static str),
    /// [`Raw`](Mode::Raw) for a request whose body contains the marker (the first field), the
    /// [`Json`](Mode::Json) 200 otherwise: one upstream that refuses some requests and serves the
    /// rest, as OpenRouter does a request too costly for the balance left.
    RawWhenBody(&'static str, u16, &'static str, &'static str),
    /// Answer with this status, then never send the body: an upstream that fails and hangs. The
    /// gateway must not wait on an attempt it abandoned.
    StatusThenStall(u16),
}

/// Content deltas a `*StallSse` mode sends before it stalls. Each carries one short token, so an
/// estimate of one token per delta event is exact against these fixtures.
pub const STALL_DELTAS: u64 = 40;
/// Input tokens `Mode::AnthropicStallSse` reports on `message_start`.
pub const STALL_ANTHROPIC_INPUT_TOKENS: u64 = 1234;

fn stall_sse(anthropic: bool) -> String {
    let mut s = String::new();
    if anthropic {
        s.push_str(&format!(
            "event: message_start\n\
             data: {{\"type\":\"message_start\",\"message\":{{\"id\":\"msg_mock\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-opus-4-8\",\"content\":[],\"usage\":{{\"input_tokens\":{STALL_ANTHROPIC_INPUT_TOKENS},\"output_tokens\":1}}}}}}\n\n"
        ));
    }
    for _ in 0..STALL_DELTAS {
        if anthropic {
            s.push_str("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" tok\"}}\n\n");
        } else {
            s.push_str("data: {\"id\":\"chatcmpl-mock\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" tok\"}}]}\n\n");
        }
    }
    s
}

/// A response body that yields one chunk and then never completes — the upstream half of a stream
/// the client abandons. It never wakes after the first frame; hyper drops it when the gateway closes
/// the connection.
pub struct StallingBody(Option<Bytes>);

impl StallingBody {
    /// `first`, then nothing ever again.
    #[allow(dead_code)]
    pub fn new(first: Bytes) -> Self {
        StallingBody(Some(first))
    }
}

impl hyper::body::Body for StallingBody {
    type Data = Bytes;
    type Error = std::io::Error;

    fn poll_frame(
        mut self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Result<hyper::body::Frame<Bytes>, Self::Error>>> {
        match self.0.take() {
            Some(b) => std::task::Poll::Ready(Some(Ok(hyper::body::Frame::data(b)))),
            None => std::task::Poll::Pending,
        }
    }
}

type MockBody = Either<Full<Bytes>, StallingBody>;

#[derive(Default, Clone, Debug)]
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
    /// The `x-beyond-*` control headers, recorded for the same reason as `beyond_model` — they are
    /// ours, they are meaningless upstream, and a provider that rejects unknown headers would turn
    /// an observability opt-in into their 400. Both must always be `None` at the upstream.
    pub beyond_metadata: Option<String>,
    pub beyond_capture: Option<String>,
    pub beyond_order: Option<String>,
    pub beyond_only: Option<String>,
    pub beyond_split: Option<String>,
    /// Anthropic (and Bedrock Messages) require this; a stock OpenAI SDK never sends it. Recorded
    /// so a Chat Completions → Messages translate walk can prove the gateway injected it.
    pub anthropic_version: Option<String>,
    /// What the gateway asked the provider to compress with. Managed traffic must ask for
    /// `identity`: the usage tail, the cache and translation all read the body as plain bytes.
    pub accept_encoding: Option<String>,
    /// Recorded so a translate walk onto a conversation-binding Claude model can prove the gateway
    /// sent the beta its `thinking.block_binding` needs (and that no other walk gets it).
    pub anthropic_beta: Option<String>,
    /// Every header the upstream received, for assertions about what must *not* be forwarded.
    pub headers: hyper::HeaderMap,
    pub body: Vec<u8>,
}

pub struct MockUpstream {
    pub port: u16,
    captured: Arc<Mutex<Option<Captured>>>,
    hits: Arc<std::sync::atomic::AtomicUsize>,
    task: tokio::task::JoinHandle<()>,
}

const CANNED_EMBEDDINGS: &str = r#"{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.0023,-0.0093,0.0158]}],"model":"text-embedding-3-small","usage":{"prompt_tokens":5,"total_tokens":5}}"#;
const CANNED_JSON: &str = r#"{"id":"chatcmpl-mock","object":"chat.completion","model":"gpt-4o-2024-08-06","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}}"#;

const CANNED_SSE: &str = "data: {\"id\":\"chatcmpl-mock\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n\
data: {\"id\":\"chatcmpl-mock\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":9}}\n\n\
data: [DONE]\n\n";

const CANNED_ANTHROPIC_JSON: &str = r#"{"id":"msg_mock","type":"message","model":"claude-opus-4-8","content":[{"type":"text","text":"hi"}],"usage":{"input_tokens":13,"output_tokens":7}}"#;

const CANNED_ANTHROPIC_SSE: &str = "event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_mock\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-opus-4-8\",\"content\":[],\"usage\":{\"input_tokens\":13,\"output_tokens\":1}}}\n\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":7}}\n\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\n";

const CANNED_ANTHROPIC_TOOL_JSON: &str = r#"{"id":"msg_mock","type":"message","role":"assistant","model":"claude-opus-4-8","content":[{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"SF"}}],"stop_reason":"tool_use","usage":{"input_tokens":13,"output_tokens":7}}"#;

const CANNED_OPENAI_TOOL_JSON: &str = r#"{"id":"chatcmpl-mock","object":"chat.completion","model":"gpt-4o-2024-08-06","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"SF\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}}"#;

const CANNED_ANTHROPIC_TOOL_SSE: &str = "event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_mock\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-opus-4-8\",\"content\":[],\"usage\":{\"input_tokens\":13,\"output_tokens\":1}}}\n\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get_weather\",\"input\":{}}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\\\"SF\\\"}\"}}\n\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":7}}\n\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\n";

const CANNED_OPENAI_TOOL_SSE: &str = "data: {\"id\":\"chatcmpl-mock\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]}}]}\n\n\
data: {\"id\":\"chatcmpl-mock\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\":\\\"SF\\\"}\"}}]}}]}\n\n\
data: {\"id\":\"chatcmpl-mock\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n\
data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":9}}\n\n\
data: [DONE]\n\n";

const CANNED_ANTHROPIC_ERROR_SSE: &str = "event: error\n\
data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"try again\"}}\n\n";

const CANNED_ANTHROPIC_THINKING_SSE: &str = "event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_mock\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-opus-4-8\",\"content\":[],\"usage\":{\"input_tokens\":13,\"output_tokens\":1,\"cache_read_input_tokens\":4,\"cache_creation_input_tokens\":2}}}\n\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"plan\"}}\n\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":1}\n\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":7,\"output_tokens_details\":{\"thinking_tokens\":3}}}\n\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\n";

const CANNED_OPENAI_ERROR_SSE: &str =
    "data: {\"error\":{\"message\":\"try again\",\"type\":\"server_error\"}}\n\n";

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
        Mode::StallSse => ("text/event-stream", Bytes::from(stall_sse(false))),
        Mode::AnthropicStallSse => ("text/event-stream", Bytes::from(stall_sse(true))),
        Mode::Json
        | Mode::Slow(_)
        | Mode::CloseOnReusedConnection
        | Mode::ThrottleKey(_)
        | Mode::RawWhenBody(..) => (
            "application/json",
            Bytes::from_static(CANNED_JSON.as_bytes()),
        ),
        Mode::Sse => (
            "text/event-stream",
            Bytes::from_static(CANNED_SSE.as_bytes()),
        ),
        Mode::Embeddings => (
            "application/json",
            Bytes::from_static(CANNED_EMBEDDINGS.as_bytes()),
        ),
        Mode::AnthropicJson => (
            "application/json",
            Bytes::from_static(CANNED_ANTHROPIC_JSON.as_bytes()),
        ),
        Mode::AnthropicSse => (
            "text/event-stream",
            Bytes::from_static(CANNED_ANTHROPIC_SSE.as_bytes()),
        ),
        Mode::AnthropicToolJson => (
            "application/json",
            Bytes::from_static(CANNED_ANTHROPIC_TOOL_JSON.as_bytes()),
        ),
        Mode::OpenAiToolJson => (
            "application/json",
            Bytes::from_static(CANNED_OPENAI_TOOL_JSON.as_bytes()),
        ),
        Mode::AnthropicToolSse => (
            "text/event-stream",
            Bytes::from_static(CANNED_ANTHROPIC_TOOL_SSE.as_bytes()),
        ),
        Mode::OpenAiToolSse => (
            "text/event-stream",
            Bytes::from_static(CANNED_OPENAI_TOOL_SSE.as_bytes()),
        ),
        Mode::AnthropicErrorSse => (
            "text/event-stream",
            Bytes::from_static(CANNED_ANTHROPIC_ERROR_SSE.as_bytes()),
        ),
        Mode::AnthropicThinkingSse => (
            "text/event-stream",
            Bytes::from_static(CANNED_ANTHROPIC_THINKING_SSE.as_bytes()),
        ),
        Mode::OpenAiErrorSse => (
            "text/event-stream",
            Bytes::from_static(CANNED_OPENAI_ERROR_SSE.as_bytes()),
        ),
        Mode::SseLarge => ("text/event-stream", Bytes::from(large_sse())),
        Mode::AnthropicSseLarge => ("text/event-stream", Bytes::from(anthropic_sse_large())),
        // The status is applied by `mock_handle`; the body is a stock error shape.
        Mode::Status(_) | Mode::StatusThenStall(_) => (
            "application/json",
            Bytes::from_static(br#"{"error":{"message":"mock"}}"#),
        ),
        Mode::AnthropicStatus(_) => (
            "application/json",
            Bytes::from_static(
                br#"{"type":"error","error":{"type":"api_error","message":"mock"}}"#,
            ),
        ),
        Mode::Raw(_, content_type, body) => (content_type, Bytes::from_static(body.as_bytes())),
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
) -> Result<Response<MockBody>, std::io::Error> {
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
    let (
        authorization,
        x_api_key,
        host,
        beyond_model,
        beyond_metadata,
        beyond_capture,
        beyond_order,
        beyond_only,
        beyond_split,
        anthropic_version,
        accept_encoding,
        anthropic_beta,
    ) = {
        let h = req.headers();
        let get = |k: &str| h.get(k).and_then(|v| v.to_str().ok()).map(String::from);
        (
            get("authorization"),
            get("x-api-key"),
            get("host"),
            get("x-beyond-model"),
            get("x-beyond-metadata"),
            get("x-beyond-capture"),
            get("x-beyond-order"),
            get("x-beyond-only"),
            get("x-beyond-split"),
            get("anthropic-version"),
            get("accept-encoding"),
            get("anthropic-beta"),
        )
    };
    let headers = req.headers().clone();
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
    let mode = match mode {
        Mode::RawWhenBody(marker, status, ct, raw) => {
            if memchr::memmem::find(&body, marker.as_bytes()).is_some() {
                Mode::Raw(status, ct, raw)
            } else {
                Mode::Json
            }
        }
        m => m,
    };
    if matches!(mode, Mode::CloseOnReusedConnection) && on_conn > 0 {
        return Err(std::io::Error::other("mock closing a reused connection"));
    }
    let throttled = match mode {
        Mode::ThrottleKey(key) => authorization
            .as_deref()
            .into_iter()
            .chain(x_api_key.as_deref())
            .any(|v| v.contains(key)),
        _ => false,
    };
    *cap.lock().unwrap() = Some(Captured {
        path,
        authorization,
        x_api_key,
        host,
        beyond_model,
        beyond_metadata,
        beyond_capture,
        beyond_order,
        beyond_only,
        beyond_split,
        anthropic_version,
        accept_encoding,
        anthropic_beta,
        headers,
        body,
    });
    // A slow upstream is still a *working* upstream; the point is to be slower than the client's
    // patience, so the client hangs up first.
    if let Mode::Slow(ms) = mode {
        sleep(Duration::from_millis(ms)).await;
    }
    let status = match mode {
        Mode::Status(s)
        | Mode::AnthropicStatus(s)
        | Mode::Raw(s, _, _)
        | Mode::StatusThenStall(s) => s,
        Mode::ThrottleKey(_) if throttled => 429,
        _ => 200,
    };
    let (ct, payload) = if status == 429 && !matches!(mode, Mode::Raw(..)) {
        (
            "application/json",
            Bytes::from_static(br#"{"error":{"message":"mock"}}"#),
        )
    } else {
        canned_body(mode)
    };
    let mut builder = Response::builder()
        .status(status)
        .header("content-type", ct)
        .header("x-mock-proto", proto_label(version));
    if status == 429 {
        builder = builder.header("retry-after", "7");
    }
    let body = if matches!(mode, Mode::StatusThenStall(_)) {
        Either::Right(StallingBody(None))
    } else if matches!(mode, Mode::StallSse | Mode::AnthropicStallSse) {
        Either::Right(StallingBody(Some(payload)))
    } else {
        Either::Left(Full::new(payload))
    };
    Ok(builder.body(body).unwrap())
}

impl MockUpstream {
    pub async fn start(mode: Mode) -> Self {
        // Bind `:0` and read the port back, keeping the listener open the whole time — no
        // pick-release-rebind window for another test to slip into (this is an in-process server).
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
    /// The metrics/admin listener: `/metrics`, `/livez`, `/readyz` (on `127.0.0.2`; see
    /// `GATEWAY_LISTENERS`).
    pub metrics: std::net::SocketAddr,
    config_path: std::path::PathBuf,
    /// The read end of a stdout pipe nobody drains ([`GatewayBuilder::stall_stdout`]): held, so the
    /// pipe stays open and full rather than broken.
    _stalled_stdout: Option<std::process::ChildStdout>,
}

/// The managed pool key configured for a provider. Each provider gets a distinct value so a test
/// can assert the gateway swapped in the *right* one.
fn pool_key(provider: &str) -> &'static str {
    match provider {
        "openai" => "sk-pool-secret",
        "anthropic" => "sk-anthropic-pool",
        "bedrock" => "sk-bedrock-pool",
        "fireworks" => "sk-fireworks-pool",
        "openrouter" => "sk-openrouter-pool",
        _ => "sk-unknown-pool",
    }
}

fn write_pool_keys_toml(cfg: &mut String, provider: &str, keys: &[String]) {
    cfg.push_str(provider);
    cfg.push_str(" = [");
    for (i, k) in keys.iter().enumerate() {
        if i > 0 {
            cfg.push_str(", ");
        }
        cfg.push('"');
        cfg.push_str(k);
        cfg.push('"');
    }
    cfg.push_str("]\n");
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
    pool_key_overrides: Vec<(String, Vec<String>)>,
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
    /// Override the per-direction payload capture cap, so truncation is testable with a small body.
    capture_max_bytes: Option<u32>,
    /// Override the default capture sampling (1 in N).
    capture_default_sample_n: Option<u32>,
    /// Exact-match response cache TTL. `None` ⇒ gateway default (off). `Some(0)` disables.
    cache_ttl_secs: Option<u64>,
    /// Per-tenant in-flight cap. `None` ⇒ gateway default (off).
    tenant_max_in_flight: Option<u32>,
    /// Wait until `ai_allowance_ready==1` after the listeners bind. Default on: allowance is
    /// fail-closed until the watcher seeds, and a first request that races the scan 402s. Tests that
    /// prove the unready path skip this.
    wait_allowance_ready: bool,
    /// Per-provider authority overrides, for a topology with more than one upstream — a failover
    /// test needs a live mock and a dead port at the same time, which the single `authority` cannot
    /// express. Falls back to `authority` for any provider not named here.
    authority_overrides: Vec<(String, String)>,
    /// Override the proxy's tokio worker-thread count. `None` ⇒ the gateway default (one per core);
    /// `Some(1)` reproduces Pingora's single-threaded default, which is what the scaling bench
    /// compares against.
    worker_threads: Option<usize>,
    /// verify phase 0: billing — raw `key = value` scalars and child env (see the marked block).
    extra_config: Vec<String>,
    env_overrides: Vec<(String, String)>,
    /// Leave stdout undrained (see [`GatewayBuilder::stall_stdout`]).
    stall_stdout: bool,
    /// `[id_signing_keys]` (kid → raw secret) and `id_signing_kid`. Defaults to [`DEV_ID_SECRET`]
    /// under kid `1`; `None` writes no table (managed Responses on a GPT row then 503s).
    id_signing: Option<IdSigning>,
}

/// `[id_signing_keys]` (kid → raw secret) and the `id_signing_kid` that signs, if named.
type IdSigning = (Vec<(char, Vec<u8>)>, Option<char>);

/// The test gateway's default id signing secret (kid `1`): what [`dev_id_signer`] signs with.
pub const DEV_ID_SECRET: [u8; 32] = [7; 32];

/// A signer holding the test gateway's default id signing key, to mint and read the signed ids
/// a test gateway issues (`signed_id.rs`).
pub fn dev_id_signer() -> beyond_ai::signed_id::Signer {
    beyond_ai::signed_id::Signer::new(&[(b'1', &DEV_ID_SECRET)], b'1').unwrap()
}

impl GatewayBuilder {
    /// Replace the id signing keys (kid → raw secret) and name the one that signs new ids.
    pub fn id_signing_keys(mut self, keys: &[(char, &[u8])], current: char) -> Self {
        self.id_signing = Some((
            keys.iter().map(|(k, s)| (*k, s.to_vec())).collect(),
            Some(current),
        ));
        self
    }

    /// Configure no id signing key at all.
    pub fn without_id_signing_keys(mut self) -> Self {
        self.id_signing = None;
        self
    }

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
    /// timeout to wait out. The port is held bound, unlistened, for the test's life ([`closed_port`]).
    pub fn dead_authority() -> String {
        format!("127.0.0.1:{}", closed_port())
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
        if let Some((_, keys)) = self
            .pool_key_overrides
            .iter_mut()
            .find(|(p, _)| p == provider)
        {
            keys.push(key.to_string());
        } else {
            self.pool_key_overrides
                .push((provider.to_string(), vec![key.to_string()]));
        }
        self
    }

    /// Set every managed pool key for a provider (TOML array). Replaces any earlier override.
    pub fn pool_keys(mut self, provider: &str, keys: &[&str]) -> Self {
        self.pool_key_overrides.retain(|(p, _)| p != provider);
        self.pool_key_overrides.push((
            provider.to_string(),
            keys.iter().map(|k| (*k).to_string()).collect(),
        ));
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

    /// Per-direction capture byte cap. Set it small to exercise truncation without having to send a
    /// 256 KiB body.
    pub fn capture_max_bytes(mut self, bytes: u32) -> Self {
        self.capture_max_bytes = Some(bytes);
        self
    }

    /// Default capture sampling (1 in N) for control-plane-enabled tenants.
    pub fn capture_default_sample_n(mut self, n: u32) -> Self {
        self.capture_default_sample_n = Some(n);
        self
    }

    /// Enable the exact-match response cache for this gateway (`0` disables).
    pub fn cache_ttl_secs(mut self, secs: u64) -> Self {
        self.cache_ttl_secs = Some(secs);
        self
    }

    pub fn tenant_max_in_flight(mut self, n: u32) -> Self {
        self.tenant_max_in_flight = Some(n);
        self
    }

    /// Do not wait for the allowance watcher to seed. Only for the fail-closed-unready test —
    /// every other managed test needs remaining-ok before the first request.
    pub fn skip_allowance_ready(mut self) -> Self {
        self.wait_allowance_ready = false;
        self
    }

    /// Spawn the gateway on ports the kernel picks, and wait until both listeners are up.
    async fn spawn(&self) -> Gateway {
        let wait_allowance_ready = self.wait_allowance_ready;
        let stall_stdout = self.stall_stdout;
        let config_path = unique_temp("beyond-ai-config", "toml");
        let nats_port = self.nats_port;
        // Scalars first, `[…]` tables last (TOML ordering).
        let tls = self.real_upstreams || self.tls_upstream;
        let listeners = beyond_ai_test_support::ports::GATEWAY_LISTENERS;
        let mut cfg = format!(
            "{listeners}\
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
        if let Some(bytes) = self.capture_max_bytes {
            cfg.push_str(&format!("capture_max_bytes = {bytes}\n"));
        }
        if let Some(n) = self.capture_default_sample_n {
            cfg.push_str(&format!("capture_default_sample_n = {n}\n"));
        }
        if let Some(secs) = self.cache_ttl_secs {
            cfg.push_str(&format!("cache_ttl_secs = {secs}\n"));
        }
        if let Some(n) = self.tenant_max_in_flight {
            cfg.push_str(&format!("tenant_max_in_flight = {n}\n"));
        }
        for line in &self.extra_config {
            cfg.push_str(line);
            cfg.push('\n');
        }
        if let Some((_, Some(kid))) = &self.id_signing {
            cfg.push_str(&format!("id_signing_kid = \"{kid}\"\n"));
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
                for (p, keys) in &self.pool_key_overrides {
                    write_pool_keys_toml(&mut cfg, p, keys);
                }
            }
            if !self.signkey_b64.is_empty() {
                cfg.push_str(&format!("\n[signing_keys]\n1 = \"{}\"\n", self.signkey_b64));
                write_id_signing_toml(&mut cfg, self.id_signing.as_ref());
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
                let keys = self
                    .pool_key_overrides
                    .iter()
                    .find(|(name, _)| name == p)
                    .map(|(_, ks)| ks.clone())
                    .unwrap_or_else(|| vec![pool_key(p).to_string()]);
                write_pool_keys_toml(&mut cfg, p, &keys);
            }
            cfg.push_str(&format!("\n[signing_keys]\n1 = \"{}\"\n", self.signkey_b64));
            write_id_signing_toml(&mut cfg, self.id_signing.as_ref());
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
                // `warn` for everything, `info` only for the two targets tests assert on — the
                // `ai.usage` rows are the reason this is captured at all, and `ai.payload` carries
                // captured request/response bodies. Turning the whole gateway (and pingora under it)
                // up to `info` for every test produced far more output than any assertion reads, on
                // a path where each line is then held in memory below.
                //
                // Note `ai.payload` needs naming here even though it has its own `tracing` layer:
                // the layer split routes events to different *writers*, while this `EnvFilter` still
                // gates them all. Without this the payload layer is installed and silently starved.
                std::env::var("AI_LOG")
                    .unwrap_or_else(|_| "warn,ai.usage=info,ai.payload=info".into()),
            )
            .envs(
                self.env_overrides
                    .iter()
                    .map(|(k, v)| (k.as_str(), v.as_str())),
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
        let stalled_stdout = if stall_stdout {
            child.stdout.take()
        } else {
            drain(
                child
                    .stdout
                    .take()
                    .map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
            );
            None
        };
        drain(
            child
                .stderr
                .take()
                .map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
        );
        // Both listeners up — the metrics/admin one (`/livez`, `/readyz`, `/metrics`) too, or a test
        // that probes it right after `start()` races its bind — read from the child's own LISTEN
        // sockets, so the ports are certainly this gateway's.
        let pid = child.id();
        let deadline = std::time::Instant::now() + STARTUP_BUDGET;
        let ports = loop {
            if let Some(ports) = beyond_ai_test_support::ports::gateway_ports(pid) {
                break ports;
            }
            if let Ok(Some(status)) = child.try_wait() {
                panic!(
                    "beyond-ai (pid {pid}) exited ({status}) before listening; log:\n{}",
                    log_tail(&log.lock().unwrap_or_else(|p| p.into_inner()))
                );
            }
            if std::time::Instant::now() >= deadline {
                panic!(
                    "beyond-ai (pid {pid}) did not come up within {STARTUP_BUDGET:?}; log:\n{}",
                    log_tail(&log.lock().unwrap_or_else(|p| p.into_inner()))
                );
            }
            sleep(Duration::from_millis(20)).await;
        };
        let gw = Gateway {
            child,
            port: ports.proxy,
            metrics: ports.metrics,
            config_path,
            log,
            _stalled_stdout: stalled_stdout,
        };
        if wait_allowance_ready {
            wait_for_metric(&gw, "ai_allowance_ready", "", 1.0).await;
        }
        gw
    }

    /// Start the gateway and wait until it serves (and, by default, until allowance is ready).
    pub async fn start(self) -> Gateway {
        self.spawn().await
    }
}

impl Gateway {
    /// Start the gateway pointed at `nats` (deny-set + allowance-set) + the mock upstream, configuring the OpenAI
    /// and Fireworks providers. Signing key + pool key come from config (mirrors production: NATS
    /// holds the watched sets). For other provider sets use [`Gateway::builder`].
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
            capture_max_bytes: None,
            capture_default_sample_n: None,
            cache_ttl_secs: None,
            tenant_max_in_flight: None,
            wait_allowance_ready: true,
            extra_config: Vec::new(),
            env_overrides: Vec::new(),
            stall_stdout: false,
            id_signing: Some((vec![('1', DEV_ID_SECRET.to_vec())], None)),
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
        let deadline = std::time::Instant::now() + CONDITION_BUDGET;
        while std::time::Instant::now() < deadline {
            let log = self.log();
            if let Some(line) = log.lines().find(|l| needles.iter().all(|n| l.contains(n))) {
                return line.to_string();
            }
            sleep(Duration::from_millis(25)).await;
        }
        panic!(
            "no log line matched {needles:?} within {CONDITION_BUDGET:?}; captured log was:\n{}",
            self.log(),
        );
    }

    pub async fn metrics(&self) -> String {
        reqwest::get(format!("http://{}/metrics", self.metrics))
            .await
            .unwrap()
            .text()
            .await
            .unwrap()
    }

    /// GET a path on the admin/metrics listener, returning `(status, body)`. Used to probe
    /// `/livez` and `/readyz` (which live on the metrics listener, alongside `/metrics`).
    pub async fn admin_get(&self, path: &str) -> (u16, String) {
        // Retry briefly: the listener is bound (we waited for the port), but the app can answer a
        // connection with a transient non-200 for a few ms right after startup. Retry a handful of
        // times before giving up, so a startup-timing blip doesn't flake the probe.
        let url = format!("http://{}{path}", self.metrics);
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

/// The last ~16 KiB of a captured gateway log, cut on a line boundary: enough to name why a wait
/// failed (a watcher's retry errors, a bind that keeps finding its port in use) without flooding
/// the failure output.
pub fn log_tail(log: &str) -> &str {
    let mut start = log.len().saturating_sub(16 * 1024);
    while !log.is_char_boundary(start) {
        start += 1;
    }
    let tail = &log[start..];
    match tail.find('\n') {
        Some(nl) if start > 0 => &tail[nl + 1..],
        _ => tail,
    }
}

pub fn parse_metric(metrics: &str, name: &str, label_value: &str) -> f64 {
    metrics
        .lines()
        .find(|l| l.starts_with(name) && l.contains(label_value))
        .and_then(|l| l.rsplit(' ').next())
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0)
}

pub async fn wait_for_metric(gw: &Gateway, name: &str, label: &str, min: f64) {
    let r = timeout(CONDITION_BUDGET, async {
        loop {
            if parse_metric(&gw.metrics().await, name, label) >= min {
                return;
            }
            sleep(Duration::from_millis(150)).await;
        }
    })
    .await;
    if r.is_err() {
        let log = gw.log();
        let tail = log_tail(&log);
        panic!("metric {name}{{{label}}} never reached {min}; gateway log tail:\n{tail}");
    }
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

// --- verify phase 0: billing ---
//
// What the billing reproductions need that `MockUpstream` cannot express: an upstream that sees the
// request body before deciding what to send (a provider honoring `stream_options.include_usage`),
// that stalls before the response head, that sends half a body and closes, or that drains the body
// and never answers. `ScriptedUpstream` is a raw HTTP/1.1 server driven by a per-request script, so
// every byte and every pause is the test's to choose. Plus two `GatewayBuilder` knobs: raw config
// scalars and child env overrides (`AI_LOG`).

impl GatewayBuilder {
    /// Append a raw top-level `key = value` line to the gateway config (e.g. `read_timeout_secs = 2`).
    pub fn config_line(mut self, line: &str) -> Self {
        self.extra_config.push(line.to_string());
        self
    }

    /// Never read the gateway's stdout: a log pipeline that stopped draining. The pipe fills and
    /// every later `write(2)` to it blocks. Only stderr reaches [`Gateway::log`].
    pub fn stall_stdout(mut self) -> Self {
        self.stall_stdout = true;
        self
    }

    /// Set an env var on the gateway child, overriding the harness's own (including `AI_LOG`).
    pub fn env(mut self, key: &str, value: &str) -> Self {
        self.env_overrides
            .push((key.to_string(), value.to_string()));
        self
    }
}

/// One step of a [`ScriptedUpstream`] reply. After the last step the connection is closed.
pub enum Step {
    Write(Vec<u8>),
    Sleep(Duration),
    /// Wait, however long it takes, until the test says so: a reply whose next bytes must follow
    /// something the client saw, as an ordering rather than a delay a loaded host could outrun.
    /// A gateway that never lets the client see it hangs the test at its own read bound.
    Until(Arc<dyn Fn() -> bool + Send + Sync>),
}

/// A complete HTTP/1.1 response with `content-length` and `connection: close`.
pub fn http_response(status: u16, content_type: &str, body: &[u8]) -> Vec<u8> {
    let mut out = http_head(status, content_type, Some(body.len()));
    out.extend_from_slice(body);
    out
}

/// A response head. `content_length: None` ⇒ the body runs to connection close.
pub fn http_head(status: u16, content_type: &str, content_length: Option<usize>) -> Vec<u8> {
    let mut head =
        format!("HTTP/1.1 {status} X\r\ncontent-type: {content_type}\r\nconnection: close\r\n");
    if let Some(n) = content_length {
        head.push_str(&format!("content-length: {n}\r\n"));
    }
    head.push_str("\r\n");
    head.into_bytes()
}

type Script = Arc<dyn Fn(&[u8], usize) -> Vec<Step> + Send + Sync>;

pub struct ScriptedUpstream {
    pub port: u16,
    /// Requests whose body arrived in full, in arrival order.
    bodies: Arc<Mutex<Vec<Vec<u8>>>>,
    task: tokio::task::JoinHandle<()>,
}

impl ScriptedUpstream {
    /// `script(body, n)` decides the reply to the `n`th (0-based) fully received request.
    pub async fn start(script: impl Fn(&[u8], usize) -> Vec<Step> + Send + Sync + 'static) -> Self {
        let script: Script = Arc::new(script);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let bodies: Arc<Mutex<Vec<Vec<u8>>>> = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&bodies);
        let task = tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let script = Arc::clone(&script);
                let seen = Arc::clone(&seen);
                tokio::spawn(scripted_conn(stream, script, seen));
            }
        });
        ScriptedUpstream { port, bodies, task }
    }

    /// Always answer with this complete response.
    pub async fn reply(status: u16, content_type: &'static str, body: String) -> Self {
        let bytes = http_response(status, content_type, body.as_bytes());
        Self::start(move |_, _| vec![Step::Write(bytes.clone())]).await
    }

    pub fn authority(&self) -> String {
        format!("127.0.0.1:{}", self.port)
    }

    /// How many requests arrived with their whole body.
    pub fn hits(&self) -> usize {
        self.bodies.lock().unwrap().len()
    }

    pub fn bodies(&self) -> Vec<Vec<u8>> {
        self.bodies.lock().unwrap().clone()
    }
}

impl Drop for ScriptedUpstream {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// Read one request (head, then a `content-length` or chunked body), record it, run the script.
async fn scripted_conn(
    mut stream: tokio::net::TcpStream,
    script: Script,
    seen: Arc<Mutex<Vec<Vec<u8>>>>,
) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut buf = Vec::new();
    let mut chunk = [0u8; 16 * 1024];
    let head_end = loop {
        if let Some(i) = find_bytes(&buf, b"\r\n\r\n") {
            break i + 4;
        }
        match stream.read(&mut chunk).await {
            Ok(0) | Err(_) => return,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
        }
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).to_ascii_lowercase();
    let header = |name: &str| {
        head.lines()
            .find_map(|l| l.strip_prefix(name).map(|v| v.trim().to_string()))
    };
    let mut rest = buf.split_off(head_end);
    let body = if let Some(n) = header("content-length:").and_then(|v| v.parse::<usize>().ok()) {
        while rest.len() < n {
            match stream.read(&mut chunk).await {
                Ok(0) | Err(_) => return,
                Ok(k) => rest.extend_from_slice(&chunk[..k]),
            }
        }
        rest.truncate(n);
        rest
    } else if header("transfer-encoding:").is_some_and(|v| v.contains("chunked")) {
        while find_bytes(&rest, b"\r\n0\r\n\r\n").is_none() && !rest.starts_with(b"0\r\n\r\n") {
            match stream.read(&mut chunk).await {
                Ok(0) | Err(_) => return,
                Ok(k) => rest.extend_from_slice(&chunk[..k]),
            }
        }
        let mut out = Vec::new();
        let mut at = 0;
        while let Some(eol) = find_bytes(&rest[at..], b"\r\n") {
            let size = std::str::from_utf8(&rest[at..at + eol])
                .ok()
                .and_then(|s| {
                    usize::from_str_radix(s.split(';').next().unwrap_or("").trim(), 16).ok()
                })
                .unwrap_or(0);
            at += eol + 2;
            if size == 0 {
                break;
            }
            out.extend_from_slice(&rest[at..at + size]);
            at += size + 2;
        }
        out
    } else {
        Vec::new()
    };
    let n = {
        let mut seen = seen.lock().unwrap();
        seen.push(body.clone());
        seen.len() - 1
    };
    for step in script(&body, n) {
        match step {
            Step::Write(bytes) => {
                if stream.write_all(&bytes).await.is_err() {
                    return;
                }
                let _ = stream.flush().await;
            }
            Step::Sleep(d) => sleep(d).await,
            Step::Until(ready) => {
                while !ready() {
                    sleep(Duration::from_millis(5)).await;
                }
            }
        }
    }
    let _ = stream.shutdown().await;
}

fn find_bytes(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// The first `ai.usage` row's fields. `tracing`'s JSON layer nests event fields under `fields`.
pub async fn usage_row_of(gw: &Gateway) -> serde_json::Value {
    let line = gw.wait_for_log_line(&[r#""target":"ai.usage""#]).await;
    let v: serde_json::Value = serde_json::from_str(&line).expect("usage line is JSON");
    v.get("fields").cloned().unwrap_or(v)
}

/// Every `ai.usage` row logged so far, fields only.
pub fn usage_rows_of(gw: &Gateway) -> Vec<serde_json::Value> {
    gw.log()
        .lines()
        .filter(|l| l.contains(r#""target":"ai.usage""#))
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .map(|v| v.get("fields").cloned().unwrap_or(v))
        .collect()
}

/// Wait up to `secs` for at least `n` usage rows; returns whatever exists at the deadline.
pub async fn wait_usage_rows(gw: &Gateway, n: usize, secs: u64) -> Vec<serde_json::Value> {
    let deadline = std::time::Instant::now() + Duration::from_secs(secs);
    loop {
        let rows = usage_rows_of(gw);
        if rows.len() >= n || std::time::Instant::now() >= deadline {
            return rows;
        }
        sleep(Duration::from_millis(25)).await;
    }
}

/// A `bai_` virtual key for `tenant_id` signed with `sk`.
pub fn billing_vkey(sk: &ed25519_dalek::SigningKey, tenant_id: u64) -> String {
    beyond_ai::key::mint(
        &beyond_ai::key::VirtualKey {
            tenant_id,
            vpc_id: 1,
            key_id: None,
        },
        1,
        sk,
    )
}

// --- verify phase 0: reliability ---
//
// Additive helpers for the `reliability_*` test files: process lifecycle on a running gateway, an
// upstream whose reply is chosen per request (`ReplyUpstream`), and raw HTTP/1.1 response parsing for the cases reqwest cannot express (chunked uploads, a reader that
// stops reading).

impl Gateway {
    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    /// Send SIGTERM to the gateway process.
    pub fn sigterm(&self) {
        let _ = Command::new("kill")
            .args(["-TERM", &self.child.id().to_string()])
            .status();
    }

    /// Wait up to `limit` for the process to exit. `Some(elapsed)` when it did.
    pub async fn wait_exit(&mut self, limit: Duration) -> Option<Duration> {
        let start = std::time::Instant::now();
        while start.elapsed() < limit {
            if let Ok(Some(_)) = self.child.try_wait() {
                return Some(start.elapsed());
            }
            sleep(Duration::from_millis(50)).await;
        }
        None
    }

    /// The current value of a metric (0 when absent).
    pub async fn metric(&self, name: &str, label: &str) -> f64 {
        parse_metric(&self.metrics().await, name, label)
    }

    /// Resident set size of the gateway process, in KiB, from `/proc/<pid>/status`.
    pub fn rss_kib(&self) -> u64 {
        self.proc_status_kib("VmRSS:")
    }

    /// The gateway process's peak resident set size so far (`VmHWM`), in KiB: a transient spike
    /// that a later `rss_kib` sample would miss still shows here.
    pub fn peak_rss_kib(&self) -> u64 {
        self.proc_status_kib("VmHWM:")
    }

    fn proc_status_kib(&self, field: &str) -> u64 {
        let status = std::fs::read_to_string(format!("/proc/{}/status", self.child.id()))
            .unwrap_or_default();
        status
            .lines()
            .find(|l| l.starts_with(field))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    }
}

/// Boot the gateway binary on a minimal config plus `extra` scalar lines and report whether it
/// refuses to start: `Some(stderr)` if it exited within `limit`, `None` if it is still running.
pub fn boot_refuses(extra: &str, limit: Duration) -> Option<String> {
    let path = unique_temp("beyond-ai-bootcheck", "toml");
    let cfg = format!(
        "{}nats_url = \"nats://127.0.0.1:{}\"\nupstream_tls = false\n{extra}\n",
        beyond_ai_test_support::ports::GATEWAY_LISTENERS,
        closed_port()
    );
    std::fs::write(&path, cfg).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_beyond-ai"))
        .args(["run", "-c"])
        .arg(&path)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn beyond-ai");
    let start = std::time::Instant::now();
    let out = loop {
        if let Ok(Some(_)) = child.try_wait() {
            let mut err = String::new();
            if let Some(mut s) = child.stderr.take() {
                let _ = std::io::Read::read_to_string(&mut s, &mut err);
            }
            break Some(err);
        }
        if start.elapsed() > limit {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    let _ = std::fs::remove_file(&path);
    out
}

/// What a [`ReplyUpstream`] does with one request.
#[derive(Clone)]
pub enum Reply {
    /// A complete response.
    Full {
        status: u16,
        content_type: &'static str,
        body: Bytes,
    },
    /// The response head and `first`, then nothing ever again.
    Stall {
        status: u16,
        content_type: &'static str,
        first: Bytes,
    },
    /// Never send a response head.
    HeaderStall,
    /// Wait this long, then send the inner reply.
    Delayed(Duration, Box<Reply>),
    /// Wait until the test releases it (`notify_one`), then send the inner reply. For an ordering
    /// a test must observe — "the others were answered while this one was still in flight" — that a
    /// delay only makes likely, and a starved runner breaks.
    Held(Arc<tokio::sync::Notify>, Box<Reply>),
    /// Drop the connection without answering, after reading the body.
    Reset,
}

impl Reply {
    pub fn json(status: u16, body: &'static str) -> Self {
        Reply::Full {
            status,
            content_type: "application/json",
            body: Bytes::from_static(body.as_bytes()),
        }
    }

    /// The stock OpenAI chat completion the plain mock serves.
    pub fn ok() -> Self {
        Reply::json(200, CANNED_JSON)
    }

    /// The stock OpenAI SSE stream the plain mock serves.
    pub fn sse() -> Self {
        Reply::Full {
            status: 200,
            content_type: "text/event-stream",
            body: Bytes::from_static(CANNED_SSE.as_bytes()),
        }
    }
}

/// The request facts a script can branch on.
pub struct ScriptReq {
    pub authorization: Option<String>,
    pub body_len: usize,
    /// The forwarded path, query included.
    pub path: String,
    pub body: Bytes,
}

type ReplyScript = Arc<dyn Fn(usize, &ScriptReq) -> Reply + Send + Sync>;

/// A plaintext HTTP/1.1 upstream whose reply to request `n` (0-based, global) is `script(n, req)`.
pub struct ReplyUpstream {
    pub port: u16,
    hits: Arc<std::sync::atomic::AtomicUsize>,
    task: tokio::task::JoinHandle<()>,
}

async fn scripted_reply(mut reply: Reply) -> Result<Response<MockBody>, std::io::Error> {
    loop {
        match reply {
            Reply::Delayed(d, inner) => {
                sleep(d).await;
                reply = *inner;
            }
            Reply::Held(release, inner) => {
                release.notified().await;
                reply = *inner;
            }
            Reply::HeaderStall => std::future::pending::<()>().await,
            Reply::Reset => return Err(std::io::Error::other("scripted reset")),
            Reply::Full {
                status,
                content_type,
                body,
            } => {
                return Ok(Response::builder()
                    .status(status)
                    .header("content-type", content_type)
                    .body(Either::Left(Full::new(body)))
                    .unwrap());
            }
            Reply::Stall {
                status,
                content_type,
                first,
            } => {
                return Ok(Response::builder()
                    .status(status)
                    .header("content-type", content_type)
                    .body(Either::Right(StallingBody(Some(first))))
                    .unwrap());
            }
        }
    }
}

impl ReplyUpstream {
    pub async fn start(
        script: impl Fn(usize, &ScriptReq) -> Reply + Send + Sync + 'static,
    ) -> Self {
        let script: ReplyScript = Arc::new(script);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let hits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = hits.clone();
        let task = tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                let io = TokioIo::new(stream);
                let (script, counter) = (script.clone(), counter.clone());
                tokio::spawn(async move {
                    let svc = service_fn(move |req: Request<hyper::body::Incoming>| {
                        let (script, counter) = (script.clone(), counter.clone());
                        async move {
                            let n = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            let authorization = req
                                .headers()
                                .get("authorization")
                                .and_then(|v| v.to_str().ok())
                                .map(String::from);
                            let path = req
                                .uri()
                                .path_and_query()
                                .map_or_else(|| req.uri().path().to_owned(), |pq| pq.to_string());
                            let body = req
                                .into_body()
                                .collect()
                                .await
                                .map(|b| b.to_bytes())
                                .unwrap_or_default();
                            let reply = script(
                                n,
                                &ScriptReq {
                                    authorization,
                                    body_len: body.len(),
                                    path,
                                    body,
                                },
                            );
                            scripted_reply(reply).await
                        }
                    });
                    let _ = hyper::server::conn::http1::Builder::new()
                        .serve_connection(io, svc)
                        .await;
                });
            }
        });
        ReplyUpstream { port, hits, task }
    }

    pub fn authority(&self) -> String {
        format!("127.0.0.1:{}", self.port)
    }

    pub fn hits(&self) -> usize {
        self.hits.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl Drop for ReplyUpstream {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// A raw HTTP/1.1 response: status, lower-cased headers, body (as far as it was read).
#[derive(Debug, Default)]
pub struct RawResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl RawResponse {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }
}

/// Parse whatever HTTP/1.1 response bytes arrived. Status `0` when no status line came back.
pub fn parse_raw_response(buf: &[u8]) -> RawResponse {
    let Some(end) = buf.windows(4).position(|w| w == b"\r\n\r\n") else {
        return RawResponse::default();
    };
    let head = String::from_utf8_lossy(&buf[..end]);
    let mut lines = head.split("\r\n");
    let status = lines
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let headers = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
        .collect();
    RawResponse {
        status,
        headers,
        body: buf[end + 4..].to_vec(),
    }
}
