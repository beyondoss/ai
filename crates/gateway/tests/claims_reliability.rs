//! Reliability claims whose subject is the claim itself, not one defect: client aborts at every
//! phase, connection reuse after rejects and cancels, `Expect: 100-continue`, NATS outages, key
//! rotation, drain on SIGTERM, slow-drip streams, mid-stream deaths and upstream H2. Each test
//! asserts the CORRECT behavior; one that reproduces an open defect is `#[ignore]`d and named in
//! `verify/defects.toml`.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::collections::HashSet;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use beyond_ai::key::{VirtualKey, mint};
use bytes::Bytes;
use common::*;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

const CHAT: &str = r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#;
const OK_JSON: &str = r#"{"id":"chatcmpl-ok","object":"chat.completion","model":"gpt-4o-2024-08-06","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}}"#;

async fn post(gw: &Gateway, path: &str, key: &str, body: &str) -> reqwest::Response {
    test_client()
        .post(format!("{}{path}", gw.url()))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .body(body.to_owned())
        .send()
        .await
        .unwrap()
}

async fn status_of(gw: &Gateway, path: &str, key: &str, body: &str) -> u16 {
    let resp = post(gw, path, key, body).await;
    let s = resp.status().as_u16();
    let _ = resp.bytes().await;
    s
}

/// Poll `f` until it returns `want` or `limit` passes. Longer than `wait_for_status`'s 10s,
/// because a NATS reconnect can sit in a doubling backoff.
async fn eventually<F, Fut>(limit: Duration, want: u16, mut f: F) -> u16
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = u16>,
{
    let start = Instant::now();
    loop {
        let got = f().await;
        if got == want || start.elapsed() > limit {
            return got;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

// --- an upstream that reports when the gateway lets go of it -----------------------------------

/// A raw HTTP/1.1 upstream. A request whose body contains `STALL_HEAD` never gets a response; one
/// with `STALL_BODY` gets a chunked SSE head and one event, then nothing. Either way the server
/// waits for the gateway to close the connection and records that it did. A request whose body
/// never arrives in full records that too. Everything else gets [`OK_JSON`].
struct RawUpstream {
    port: u16,
    events: Arc<Mutex<Vec<&'static str>>>,
    conns: Arc<AtomicUsize>,
    task: tokio::task::JoinHandle<()>,
}

impl RawUpstream {
    async fn start() -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let events = Arc::new(Mutex::new(Vec::new()));
        let conns = Arc::new(AtomicUsize::new(0));
        let (ev, cn) = (events.clone(), conns.clone());
        let task = tokio::spawn(async move {
            while let Ok((s, _)) = listener.accept().await {
                cn.fetch_add(1, Ordering::SeqCst);
                tokio::spawn(raw_conn(s, ev.clone()));
            }
        });
        RawUpstream {
            port,
            events,
            conns,
            task,
        }
    }

    fn authority(&self) -> String {
        format!("127.0.0.1:{}", self.port)
    }

    async fn wait_event(&self, what: &str, limit: Duration) -> bool {
        let start = Instant::now();
        while start.elapsed() < limit {
            if self.events.lock().unwrap().contains(&what) {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        false
    }
}

impl Drop for RawUpstream {
    fn drop(&mut self) {
        self.task.abort();
    }
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

async fn drain_to_eof(s: &mut TcpStream) {
    let mut buf = [0u8; 4096];
    while let Ok(n) = s.read(&mut buf).await {
        if n == 0 {
            return;
        }
    }
}

async fn raw_conn(mut s: TcpStream, events: Arc<Mutex<Vec<&'static str>>>) {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 16 * 1024];
    let head_end = loop {
        if let Some(i) = find(&buf, b"\r\n\r\n") {
            break i + 4;
        }
        match s.read(&mut chunk).await {
            Ok(0) | Err(_) => return,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
        }
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).to_ascii_lowercase();
    let len = head
        .lines()
        .find_map(|l| l.strip_prefix("content-length:"))
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(0);
    let mut body = buf.split_off(head_end);
    if head.contains("transfer-encoding: chunked") {
        // The framing stays in `body`; only the markers are looked for.
        while find(&body, b"0\r\n\r\n").is_none() {
            match s.read(&mut chunk).await {
                Ok(0) | Err(_) => {
                    events.lock().unwrap().push("upload-eof");
                    return;
                }
                Ok(n) => body.extend_from_slice(&chunk[..n]),
            }
        }
    }
    while body.len() < len {
        match s.read(&mut chunk).await {
            Ok(0) | Err(_) => {
                events.lock().unwrap().push("upload-eof");
                return;
            }
            Ok(n) => body.extend_from_slice(&chunk[..n]),
        }
    }
    if find(&body, b"STALL_HEAD").is_some() {
        drain_to_eof(&mut s).await;
        events.lock().unwrap().push("head-eof");
    } else if find(&body, b"STALL_BODY").is_some() {
        let event = "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n";
        let out = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n{:x}\r\n{event}\r\n",
            event.len()
        );
        let _ = s.write_all(out.as_bytes()).await;
        drain_to_eof(&mut s).await;
        events.lock().unwrap().push("stream-eof");
    } else {
        let _ = s
            .write_all(&http_response(200, "application/json", OK_JSON.as_bytes()))
            .await;
        let _ = s.shutdown().await;
    }
}

/// The in-flight and stream gauges are back to zero.
async fn wait_gauges_idle(gw: &Gateway) -> (f64, f64) {
    let start = Instant::now();
    loop {
        let m = gw.metrics().await;
        let got = (
            parse_metric(&m, "ai_requests_in_flight", ""),
            parse_metric(&m, "ai_active_streams", ""),
        );
        if got == (0.0, 0.0) || start.elapsed() > Duration::from_secs(5) {
            return got;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// A gateway where any leak shows on the next request: one tenant slot, and a breaker that opens
/// on a single failure.
async fn abort_gateway(up: &RawUpstream, seed: u8) -> (Gateway, String) {
    let (pubkey, sk) = test_keypair(seed);
    let gw = Gateway::builder(unused_nats_port(), &up.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .tenant_max_in_flight(1)
        .circuit_breaker_threshold(1)
        .start()
        .await;
    (gw, billing_vkey(&sk, u64::from(seed)))
}

/// After an abort: gauges idle, then the tenant's next request is admitted (its slot came back)
/// and reaches the provider (the breaker did not count the abort as a provider failure).
async fn assert_everything_released(gw: &Gateway, up: &RawUpstream, key: &str, phase: &str) {
    assert_eq!(
        wait_gauges_idle(gw).await,
        (0.0, 0.0),
        "{phase}: in-flight/stream gauges did not return to zero"
    );
    let before = up.conns.load(Ordering::SeqCst);
    let status = status_of(gw, "/openai/v1/chat/completions", key, CHAT).await;
    assert_eq!(
        status, 200,
        "{phase}: the next request was refused (slot or breaker leaked)"
    );
    assert!(
        up.conns.load(Ordering::SeqCst) > before,
        "{phase}: the next request did not reach the provider"
    );
}

/// The client hangs up halfway through uploading its body.
/// claim: REL-9
#[tokio::test]
async fn a_client_abort_during_upload_releases_everything() {
    let up = RawUpstream::start().await;
    let (gw, key) = abort_gateway(&up, 91).await;
    let mut s = TcpStream::connect(("127.0.0.1", gw.port)).await.unwrap();
    let head = format!(
        "POST /openai/v1/chat/completions HTTP/1.1\r\nhost: x\r\nauthorization: Bearer {key}\r\ncontent-type: application/json\r\ncontent-length: 100000\r\n\r\n"
    );
    s.write_all(head.as_bytes()).await.unwrap();
    s.write_all(&[b' '; 2000]).await.unwrap();
    tokio::time::sleep(Duration::from_millis(300)).await;
    drop(s);
    tokio::time::sleep(Duration::from_millis(200)).await;
    if up.conns.load(Ordering::SeqCst) > 0 {
        assert!(
            up.wait_event("upload-eof", Duration::from_secs(5)).await,
            "the gateway kept the upstream connection of an abandoned upload open"
        );
    }
    assert_everything_released(&gw, &up, &key, "upload").await;
}

/// The client gives up while the provider is still thinking (before the response head).
/// claim: REL-9
#[tokio::test]
async fn a_client_abort_before_the_response_head_releases_everything() {
    let up = RawUpstream::start().await;
    let (gw, key) = abort_gateway(&up, 92).await;
    let impatient = reqwest::Client::builder()
        .timeout(Duration::from_millis(400))
        .build()
        .unwrap();
    let r = impatient
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .body(r#"{"model":"gpt-4o","messages":[{"role":"user","content":"STALL_HEAD"}]}"#)
        .send()
        .await;
    assert!(
        r.is_err(),
        "the upstream never answers; the client timed out"
    );
    assert!(
        up.wait_event("head-eof", Duration::from_secs(5)).await,
        "the gateway kept waiting on the provider for a client that left"
    );
    assert_everything_released(&gw, &up, &key, "before head").await;
}

/// The client hangs up mid-stream (an agent's ESC).
/// claim: REL-9
#[tokio::test]
async fn a_client_abort_mid_stream_releases_everything() {
    let up = RawUpstream::start().await;
    let (gw, key) = abort_gateway(&up, 93).await;
    let mut resp = post(
        &gw,
        "/openai/v1/chat/completions",
        &key,
        r#"{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"STALL_BODY"}]}"#,
    )
    .await;
    assert_eq!(resp.status(), 200);
    let first = tokio::time::timeout(Duration::from_secs(10), resp.chunk())
        .await
        .expect("first event")
        .unwrap();
    assert!(first.is_some());
    drop(resp);
    assert!(
        up.wait_event("stream-eof", Duration::from_secs(5)).await,
        "the gateway kept the provider's stream open after the client left"
    );
    assert_everything_released(&gw, &up, &key, "mid-stream").await;
}

// --- connection reuse ----------------------------------------------------------------------------

/// Read one HTTP/1.1 response (content-length or chunked) off a kept-alive connection.
async fn read_response(s: &mut TcpStream) -> Option<RawResponse> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 8192];
    let fut = async {
        let head_end = loop {
            if let Some(i) = find(&buf, b"\r\n\r\n") {
                break i + 4;
            }
            match s.read(&mut chunk).await {
                Ok(0) | Err(_) => return None,
                Ok(n) => buf.extend_from_slice(&chunk[..n]),
            }
        };
        let head = String::from_utf8_lossy(&buf[..head_end]).to_ascii_lowercase();
        if let Some(len) = head
            .lines()
            .find_map(|l| l.strip_prefix("content-length:"))
            .and_then(|v| v.trim().parse::<usize>().ok())
        {
            while buf.len() < head_end + len {
                match s.read(&mut chunk).await {
                    Ok(0) | Err(_) => return None,
                    Ok(n) => buf.extend_from_slice(&chunk[..n]),
                }
            }
        } else if head.contains("transfer-encoding: chunked") {
            while find(&buf[head_end..], b"0\r\n\r\n").is_none() {
                match s.read(&mut chunk).await {
                    Ok(0) | Err(_) => return None,
                    Ok(n) => buf.extend_from_slice(&chunk[..n]),
                }
            }
        }
        Some(parse_raw_response(&buf))
    };
    tokio::time::timeout(Duration::from_secs(10), fut)
        .await
        .ok()
        .flatten()
}

/// A rejected request whose body the gateway never needed must not desync the connection: the
/// next request on it is answered as itself, or the reject said `connection: close` and the
/// gateway closed. Never a 400 from parsing the leftover body as a request line.
/// claim: REL-11
#[tokio::test]
async fn h1_keep_alive_stays_in_sync_after_rejects() {
    let (pubkey, sk) = test_keypair(111);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let key = billing_vkey(&sk, 111);
    let pad = "x".repeat(3000);
    let body = format!(r#"{{"model":"gpt-4o","messages":[{{"role":"user","content":"{pad}"}}]}}"#);
    let request = |auth: Option<&str>, path: &str| {
        let auth = auth
            .map(|k| format!("authorization: Bearer {k}\r\n"))
            .unwrap_or_default();
        format!(
            "POST {path} HTTP/1.1\r\nhost: x\r\n{auth}content-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
            body.len()
        )
    };
    // reject reasons: no credential, bad credential, unknown provider, managed non-generation path
    let rejects = [
        request(None, "/openai/v1/chat/completions"),
        request(Some("bai_v1.1.bogus.bogus"), "/openai/v1/chat/completions"),
        request(Some(&key), "/nope/v1/chat/completions"),
        request(Some(&key), "/openai/v1/files"),
    ];
    for reject in rejects {
        let mut s = TcpStream::connect(("127.0.0.1", gw.port)).await.unwrap();
        s.write_all(reject.as_bytes()).await.unwrap();
        let first = read_response(&mut s).await.expect("the reject is answered");
        assert!(
            (400..500).contains(&first.status),
            "{}: {first:?}",
            reject.lines().next().unwrap()
        );
        let closing = first
            .header("connection")
            .is_some_and(|v| v.eq_ignore_ascii_case("close"));
        let ok = request(Some(&key), "/openai/v1/chat/completions");
        if s.write_all(ok.as_bytes()).await.is_err() {
            assert!(closing, "closed without saying so: {first:?}");
            continue;
        }
        match read_response(&mut s).await {
            Some(second) => assert_eq!(
                second.status,
                200,
                "after `{}` the next request on the connection was misread: {second:?}",
                reject.lines().next().unwrap()
            ),
            None => assert!(
                closing,
                "after `{}` the connection died without `connection: close`: {first:?}",
                reject.lines().next().unwrap()
            ),
        }
    }
}

/// One h2c connection carries a stream that is cancelled, a request that is rejected, a slow
/// request in flight the whole time, and a fresh request afterwards. Only the cancelled stream is
/// affected.
/// claim: REL-11
#[tokio::test]
async fn h2c_multiplexing_survives_a_cancel_and_a_reject() {
    let (pubkey, sk) = test_keypair(112);
    // Classified by body size: slow (≥ 1000 bytes), stalling stream (≥ 500), otherwise ok.
    let up = ReplyUpstream::start(|_, req| {
        if req.body_len >= 1000 {
            Reply::Delayed(Duration::from_millis(1500), Box::new(Reply::ok()))
        } else if req.body_len >= 500 {
            Reply::Stall {
                status: 200,
                content_type: "text/event-stream",
                first: Bytes::from_static(b"data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n"),
            }
        } else {
            Reply::ok()
        }
    })
    .await;
    let gw = Gateway::builder(unused_nats_port(), &up.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let key = billing_vkey(&sk, 112);
    let h2 = reqwest::Client::builder()
        .http2_prior_knowledge()
        .timeout(Duration::from_secs(15))
        .build()
        .unwrap();
    let url = format!("{}/openai/v1/chat/completions", gw.url());
    let req = |body: String, auth: &str| {
        h2.post(&url)
            .header("authorization", format!("Bearer {auth}"))
            .header("content-type", "application/json")
            .body(body)
    };
    let chat = |pad: usize, stream: bool| {
        format!(
            r#"{{"model":"gpt-4o","stream":{stream},"messages":[{{"role":"user","content":"{}"}}]}}"#,
            "p".repeat(pad)
        )
    };
    let slow = tokio::spawn(req(chat(1200, false), &key).send());
    tokio::time::sleep(Duration::from_millis(100)).await;
    // Cancel a stream mid-way.
    let mut stream = req(chat(600, true), &key).send().await.unwrap();
    assert_eq!(stream.version(), reqwest::Version::HTTP_2);
    assert_eq!(stream.status(), 200);
    let _ = tokio::time::timeout(Duration::from_secs(5), stream.chunk()).await;
    drop(stream);
    // A reject on the same connection.
    let rejected = req(chat(10, false), "bai_v1.1.bogus.bogus")
        .send()
        .await
        .unwrap();
    assert_eq!(rejected.status(), 401);
    let _ = rejected.bytes().await;
    // A fresh request.
    let fresh = req(chat(10, false), &key).send().await.unwrap();
    assert_eq!(fresh.status(), 200);
    assert!(fresh.text().await.unwrap().contains("chatcmpl"));
    // The slow one, in flight throughout.
    let slow = slow.await.unwrap().expect("the slow request survived");
    assert_eq!(slow.status(), 200);
    assert!(slow.text().await.unwrap().contains("chatcmpl"));
}

/// `Expect: 100-continue` (curl does this for large bodies): the gateway answers with `100
/// Continue` or a final status promptly, so the client never sits out its own expect timeout,
/// and a request it rejects is refused without waiting for the body.
/// claim: REL-12
/// defect: D73
#[tokio::test]
async fn expect_100_continue_never_stalls() {
    let (pubkey, sk) = test_keypair(113);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let key = billing_vkey(&sk, 113);
    for (auth, want) in [(Some(key.as_str()), 200u16), (None, 401)] {
        let mut s = TcpStream::connect(("127.0.0.1", gw.port)).await.unwrap();
        let auth_line = auth
            .map(|k| format!("authorization: Bearer {k}\r\n"))
            .unwrap_or_default();
        let head = format!(
            "POST /openai/v1/chat/completions HTTP/1.1\r\nhost: x\r\n{auth_line}content-type: application/json\r\nexpect: 100-continue\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
            CHAT.len()
        );
        s.write_all(head.as_bytes()).await.unwrap();
        // curl waits 1s for the interim response; answer well inside that.
        let mut buf = vec![0u8; 8192];
        let first = tokio::time::timeout(Duration::from_millis(900), s.read(&mut buf)).await;
        let got = match first {
            Ok(Ok(n)) if n > 0 => String::from_utf8_lossy(&buf[..n]).to_string(),
            _ => panic!(
                "auth={}: nothing within 900ms of the expect head",
                auth.is_some()
            ),
        };
        let mut all = got.clone().into_bytes();
        if got.starts_with("HTTP/1.1 100") {
            s.write_all(CHAT.as_bytes()).await.unwrap();
        }
        let _ = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                match s.read(&mut buf).await {
                    Ok(0) | Err(_) => break,
                    Ok(n) => all.extend_from_slice(&buf[..n]),
                }
            }
        })
        .await;
        let text = String::from_utf8_lossy(&all).to_string();
        // Skip the interim response (whatever headers it carries).
        let final_part = if text.starts_with("HTTP/1.1 100") {
            text.split_once("\r\n\r\n").map_or("", |(_, rest)| rest)
        } else {
            &text
        };
        let resp = parse_raw_response(final_part.as_bytes());
        assert_eq!(resp.status, want, "auth={}: {text}", auth.is_some());
    }
}

// --- NATS outages --------------------------------------------------------------------------------

/// A `nats-server` that can be stopped and started again on the same port and JetStream store,
/// so its KV contents survive the outage.
struct RestartableNats {
    child: Option<Child>,
    port: u16,
    dir: std::path::PathBuf,
}

impl RestartableNats {
    fn new() -> Self {
        let port = free_port();
        let dir = std::env::temp_dir().join(format!("beyond-ai-rel13-{port}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        RestartableNats {
            child: None,
            port,
            dir,
        }
    }

    async fn up(&mut self) {
        let child = Command::new("nats-server")
            .args([
                "-js",
                "-a",
                "127.0.0.1",
                "-p",
                &self.port.to_string(),
                "-sd",
            ])
            .arg(&self.dir)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn nats-server");
        self.child = Some(child);
        let start = Instant::now();
        while TcpStream::connect(("127.0.0.1", self.port)).await.is_err() {
            assert!(
                start.elapsed() < Duration::from_secs(20),
                "nats-server did not come up"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    fn down(&mut self) {
        if let Some(mut c) = self.child.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

impl Drop for RestartableNats {
    fn drop(&mut self) {
        self.down();
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn tenant_key(sk: &ed25519_dalek::SigningKey, tenant: u64) -> String {
    billing_vkey(sk, tenant)
}

/// NATS is down when the gateway boots: it must come up anyway, and once NATS returns it must seed
/// the deny-set (including a hold written before boot) and the allowance-set without a restart.
/// claim: REL-13
#[tokio::test]
async fn a_nats_outage_at_boot_recovers_and_applies_what_it_missed() {
    let mut nats = RestartableNats::new();
    nats.up().await;
    put_kv(nats.port, "blackhole.1301", b"spend").await;
    nats.down();
    let (pubkey, sk) = test_keypair(131);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats.port, &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .skip_allowance_ready()
        .start()
        .await;
    nats.up().await;
    let (held, free) = (tenant_key(&sk, 1301), tenant_key(&sk, 1302));
    let path = "/openai/v1/chat/completions";
    let got = eventually(Duration::from_secs(45), 200, || {
        status_of(&gw, path, &free, CHAT)
    })
    .await;
    assert_eq!(
        got, 200,
        "the allowance-set never seeded after NATS returned"
    );
    let got = eventually(Duration::from_secs(10), 402, || {
        status_of(&gw, path, &held, CHAT)
    })
    .await;
    assert_eq!(got, 402, "the hold written before boot was never applied");
}

/// NATS bounces mid-run. A hold written the moment it is back (before the gateway's watch has
/// necessarily re-attached) must still land: the watch resumes or re-seeds, never silently skips.
/// claim: REL-13
#[tokio::test]
async fn a_mid_run_nats_outage_recovers_and_applies_missed_updates() {
    let mut nats = RestartableNats::new();
    nats.up().await;
    let (pubkey, sk) = test_keypair(132);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats.port, &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let key = tenant_key(&sk, 1303);
    let path = "/openai/v1/chat/completions";
    assert_eq!(status_of(&gw, path, &key, CHAT).await, 200);
    nats.down();
    // Let the watchers notice the disconnect.
    tokio::time::sleep(Duration::from_secs(2)).await;
    assert_eq!(
        status_of(&gw, path, &key, CHAT).await,
        200,
        "fail-open while NATS is down"
    );
    nats.up().await;
    put_kv(nats.port, "blackhole.1303", b"spend").await;
    let got = eventually(Duration::from_secs(45), 402, || {
        status_of(&gw, path, &key, CHAT)
    })
    .await;
    assert_eq!(
        got, 402,
        "a hold written after the outage was never applied"
    );
    del_kv(nats.port, "blackhole.1303").await;
    let got = eventually(Duration::from_secs(10), 200, || {
        status_of(&gw, path, &key, CHAT)
    })
    .await;
    assert_eq!(got, 200, "the release after the outage was never applied");
}

/// A gateway that boots from an on-disk snapshot holding a hold that was lifted while it was down
/// must replace the stale snapshot from NATS, not enforce it forever.
/// claim: REL-13
#[tokio::test]
async fn a_stale_snapshot_is_replaced_once_nats_is_reachable() {
    let mut nats = RestartableNats::new();
    nats.up().await;
    let (pubkey, sk) = test_keypair(133);
    let mock = MockUpstream::start(Mode::Json).await;
    let snap = std::env::temp_dir().join(format!("beyond-ai-rel13-snap-{}.log", nats.port));
    let _ = std::fs::remove_file(&snap);
    let snap_str = snap.to_str().unwrap().to_string();
    let key = tenant_key(&sk, 1304);
    let path = "/openai/v1/chat/completions";
    {
        let gw = Gateway::builder(nats.port, &mock.authority(), &b64(&pubkey))
            .providers(&["openai"])
            .snapshot_path(&snap_str)
            .start()
            .await;
        put_kv(nats.port, "blackhole.1304", b"fraud").await;
        let got = eventually(Duration::from_secs(10), 403, || {
            status_of(&gw, path, &key, CHAT)
        })
        .await;
        assert_eq!(got, 403);
        tokio::time::sleep(Duration::from_millis(400)).await;
    }
    del_kv(nats.port, "blackhole.1304").await;
    let gw = Gateway::builder(nats.port, &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .snapshot_path(&snap_str)
        .start()
        .await;
    let got = eventually(Duration::from_secs(45), 200, || {
        status_of(&gw, path, &key, CHAT)
    })
    .await;
    let _ = std::fs::remove_file(&snap);
    assert_eq!(
        got, 200,
        "the lifted hold from the stale snapshot is still enforced"
    );
}

// --- rotation ------------------------------------------------------------------------------------

/// Two signing kids verify side by side, each only with its own key; retiring kid 1 (pointing it
/// at the new key) refuses old-kid tokens and keeps new-kid tokens working.
/// claim: REL-14
#[tokio::test]
async fn two_signing_kids_rotate_cleanly() {
    let (pub_a, sk_a) = test_keypair(141);
    let (pub_b, sk_b) = test_keypair(142);
    let mock = MockUpstream::start(Mode::Json).await;
    let vk = VirtualKey {
        tenant_id: 1401,
        vpc_id: 1,
        key_id: None,
    };
    let path = "/openai/v1/chat/completions";
    let both = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pub_a))
        .providers(&["openai"])
        .env("AI_SIGNING_KEY_2", &b64(&pub_b))
        .start()
        .await;
    assert_eq!(
        status_of(&both, path, &mint(&vk, 1, &sk_a), CHAT).await,
        200
    );
    assert_eq!(
        status_of(&both, path, &mint(&vk, 2, &sk_b), CHAT).await,
        200
    );
    assert_eq!(
        status_of(&both, path, &mint(&vk, 2, &sk_a), CHAT).await,
        401,
        "kid 2 verifies only with kid 2's key"
    );
    assert_eq!(
        status_of(&both, path, &mint(&vk, 3, &sk_b), CHAT).await,
        401,
        "an unknown kid is refused"
    );
    let retired = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pub_a))
        .providers(&["openai"])
        .env("AI_SIGNING_KEY_1", &b64(&pub_b))
        .env("AI_SIGNING_KEY_2", &b64(&pub_b))
        .start()
        .await;
    assert_eq!(
        status_of(&retired, path, &mint(&vk, 1, &sk_a), CHAT).await,
        401,
        "the retired key no longer verifies"
    );
    assert_eq!(
        status_of(&retired, path, &mint(&vk, 2, &sk_b), CHAT).await,
        200
    );
}

/// An upstream that accepts only the pool keys in `valid`, and 401s the rest.
async fn keyed_upstream(valid: Arc<Mutex<HashSet<&'static str>>>) -> ReplyUpstream {
    ReplyUpstream::start(move |_, req| {
        let auth = req.authorization.clone().unwrap_or_default();
        if valid.lock().unwrap().iter().any(|k| auth.ends_with(k)) {
            Reply::ok()
        } else {
            Reply::json(
                401,
                r#"{"error":{"message":"invalid api key","type":"invalid_request_error"}}"#,
            )
        }
    })
    .await
}

/// A rolling pool-key rotation — old pods on `[old]`, new pods on `[new, old]`, then the provider
/// revokes `old` once the old pods are gone — drops no request at any step.
/// claim: REL-14
#[tokio::test]
async fn a_rolling_pool_key_rotation_drops_nothing() {
    let valid = Arc::new(Mutex::new(HashSet::from(["sk-old", "sk-new"])));
    let up = keyed_upstream(valid.clone()).await;
    let (pubkey, sk) = test_keypair(143);
    let key = billing_vkey(&sk, 1403);
    let path = "/openai/v1/chat/completions";
    let old_pod = Gateway::builder(unused_nats_port(), &up.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .pool_keys("openai", &["sk-old"])
        .start()
        .await;
    let new_pod = Gateway::builder(unused_nats_port(), &up.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .pool_keys("openai", &["sk-new", "sk-old"])
        .start()
        .await;
    for _ in 0..5 {
        assert_eq!(status_of(&old_pod, path, &key, CHAT).await, 200);
        assert_eq!(status_of(&new_pod, path, &key, CHAT).await, 200);
    }
    drop(old_pod);
    valid.lock().unwrap().remove("sk-old");
    for _ in 0..5 {
        assert_eq!(status_of(&new_pod, path, &key, CHAT).await, 200);
    }
}

/// The natural rotation order — append the new key, revoke the old one at the provider, then
/// remove it from config — must not drop traffic in between: a 401 on a managed pool key is our
/// credential's fault, never the client's, so the gateway tries the next key.
/// claim: REL-14
/// defect: D71
#[tokio::test]
async fn a_revoked_first_pool_key_walks_to_the_next() {
    let valid = Arc::new(Mutex::new(HashSet::from(["sk-new"])));
    let up = keyed_upstream(valid).await;
    let (pubkey, sk) = test_keypair(144);
    let gw = Gateway::builder(unused_nats_port(), &up.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .pool_keys("openai", &["sk-old", "sk-new"])
        .start()
        .await;
    let key = billing_vkey(&sk, 1404);
    for _ in 0..3 {
        assert_eq!(
            status_of(&gw, "/openai/v1/chat/completions", &key, CHAT).await,
            200,
            "a revoked first pool key black-holed the request"
        );
    }
}

/// A revoked pool key is paid for once, not on every request: its 401 cools it off, so later
/// requests start on the next key. With every key revoked, the last key's 401 is relayed after
/// each key was tried once — never a vendor switch on a provider route.
/// claim: REL-14
#[tokio::test]
async fn a_revoked_pool_key_cools_off_and_the_last_401_is_relayed() {
    let valid = Arc::new(Mutex::new(HashSet::from(["sk-new"])));
    let up = keyed_upstream(valid.clone()).await;
    let (pubkey, sk) = test_keypair(145);
    let gw = Gateway::builder(unused_nats_port(), &up.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .pool_keys("openai", &["sk-old", "sk-new"])
        .start()
        .await;
    let key = billing_vkey(&sk, 1405);
    let path = "/openai/v1/chat/completions";
    for _ in 0..4 {
        assert_eq!(status_of(&gw, path, &key, CHAT).await, 200);
    }
    assert_eq!(up.hits(), 5, "sk-old is tried by the first request only");
    assert_eq!(gw.metric("ai_key_auth_failures_total", "").await, 1.0);

    valid.lock().unwrap().clear();
    let before = up.hits();
    assert_eq!(status_of(&gw, path, &key, CHAT).await, 401);
    assert_eq!(
        up.hits() - before,
        1,
        "started on sk-new, the last key: nothing to walk to"
    );
    let resp = post(&gw, path, &key, CHAT).await;
    assert_eq!(resp.status(), 401);
    assert!(resp.text().await.unwrap().contains("invalid api key"));
}

// --- drain ---------------------------------------------------------------------------------------

/// SIGTERM with a request in flight: the request finishes, the client gets the whole answer, and
/// its billing row is written before the process exits.
/// claim: REL-16, BIL-21
#[tokio::test]
async fn sigterm_drains_an_in_flight_request_and_bills_it() {
    let up = ReplyUpstream::start(|_, _| {
        Reply::Delayed(Duration::from_millis(1500), Box::new(Reply::ok()))
    })
    .await;
    let (pubkey, sk) = test_keypair(161);
    let gw = Gateway::builder(unused_nats_port(), &up.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .config_line("shutdown_grace_period_secs = 10")
        .start()
        .await;
    let key = billing_vkey(&sk, 1601);
    let url = format!("{}/openai/v1/chat/completions", gw.url());
    let inflight = tokio::spawn(async move {
        let resp = test_client()
            .post(url)
            .header("authorization", format!("Bearer {key}"))
            .header("content-type", "application/json")
            .body(CHAT)
            .send()
            .await
            .unwrap();
        (
            resp.status().as_u16(),
            resp.text().await.unwrap_or_default(),
        )
    });
    tokio::time::sleep(Duration::from_millis(400)).await;
    assert_eq!(
        up.hits(),
        1,
        "the request reached the provider before SIGTERM"
    );
    gw.sigterm();
    let (status, text) = inflight.await.unwrap();
    assert_eq!(status, 200, "{text}");
    assert!(text.contains("chatcmpl"), "{text}");
    let rows = wait_usage_rows(&gw, 1, 5).await;
    assert_eq!(rows.len(), 1, "the drained request's row: {}", gw.log());
    assert_eq!(rows[0]["input_tokens"].as_u64(), Some(11), "{}", rows[0]);
}

// --- streams -------------------------------------------------------------------------------------

/// A healthy stream that drips one event every 1.2s survives a 2s read timeout end to end — the
/// stall detector measures gaps between bytes, not total duration — and is billed exactly.
/// claim: REL-7
#[tokio::test]
async fn a_slow_drip_stream_survives_a_short_read_timeout() {
    let up = ScriptedUpstream::start(|_, _| {
        let mut steps = vec![Step::Write(http_head(200, "text/event-stream", None))];
        for _ in 0..5 {
            steps.push(Step::Sleep(Duration::from_millis(1200)));
            steps.push(Step::Write(
                b"data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" tok\"}}]}\n\n".to_vec(),
            ));
        }
        steps.push(Step::Write(
            b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":5}}\n\ndata: [DONE]\n\n".to_vec(),
        ));
        steps
    })
    .await;
    let (pubkey, sk) = test_keypair(171);
    let gw = Gateway::builder(unused_nats_port(), &up.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .config_line("read_timeout_secs = 2")
        .start()
        .await;
    let resp = post(
        &gw,
        "/openai/v1/chat/completions",
        &billing_vkey(&sk, 1701),
        r#"{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"hi"}]}"#,
    )
    .await;
    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("the whole stream arrived");
    assert_eq!(text.matches(" tok").count(), 5, "{text}");
    assert!(text.contains("[DONE]"), "{text}");
    let row = usage_row_of(&gw).await;
    assert_eq!(row["output_tokens"].as_u64(), Some(5), "{row}");
    assert_eq!(row["usage_estimated"], false, "{row}");
}

/// A chunked SSE head and one event, then the connection closes without the terminating chunk.
fn dies_mid_stream(_: &[u8], _: usize) -> Vec<Step> {
    let event = "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n";
    vec![Step::Write(
        format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n{:x}\r\n{event}\r\n",
            event.len()
        )
        .into_bytes(),
    )]
}

/// A JSON body that promises 400 bytes, sends 100, and closes.
fn dies_mid_body(_: &[u8], _: usize) -> Vec<Step> {
    let mut out = http_head(200, "application/json", Some(400));
    out.extend_from_slice(&OK_JSON.as_bytes()[..100]);
    vec![Step::Write(out)]
}

/// Once the first byte reaches the client nothing fails over (the fallback is never asked), and a
/// response the provider abandoned mid-way reaches the client as an error — never as a clean end
/// the client would take for a complete answer. Streamed and not.
/// claim: REL-3
#[tokio::test]
async fn a_response_that_dies_mid_way_is_an_error_and_never_fails_over() {
    for (script, stream) in [
        (dies_mid_stream as fn(&[u8], usize) -> Vec<Step>, true),
        (dies_mid_body, false),
    ] {
        let primary = ScriptedUpstream::start(script).await;
        let fallback = MockUpstream::start(Mode::Json).await;
        let (pubkey, sk) = test_keypair(31);
        let gw = Gateway::builder(unused_nats_port(), &primary.authority(), &b64(&pubkey))
            .providers(&["openai", "openrouter"])
            .provider_authority("openrouter", &fallback.authority())
            .start()
            .await;
        let resp = post(
            &gw,
            "/v1/chat/completions",
            &billing_vkey(&sk, 31),
            &format!(
                r#"{{"model":"gpt-4o-mini","stream":{stream},"messages":[{{"role":"user","content":"hi"}}]}}"#
            ),
        )
        .await;
        assert_eq!(resp.status(), 200, "stream={stream}");
        let body = resp.bytes().await;
        assert!(
            body.is_err(),
            "stream={stream}: an abandoned response ended cleanly: {:?}",
            body.map(|b| String::from_utf8_lossy(&b).to_string())
        );
        assert_eq!(primary.hits(), 1, "stream={stream}");
        assert_eq!(
            fallback.hits(),
            0,
            "stream={stream}: failed over after the first byte"
        );
    }
}

// --- failure classes -----------------------------------------------------------------------------

/// Every upstream failure class ends within a bound, with a documented status: the provider's own
/// 429/5xx is relayed, and a failure the gateway itself reports (refused, reset, stalled) is a
/// retryable 502/503/504 — never a hang, a 2xx, or a non-retryable 4xx.
/// claim: REL-1, REL-19
#[tokio::test]
async fn every_upstream_failure_class_has_a_bounded_outcome() {
    let (pubkey, sk) = test_keypair(11);
    let key = billing_vkey(&sk, 11);
    let classes: [(&str, Option<Reply>, &[u16]); 5] = [
        ("connect refused", None, &[502, 503]),
        ("reset after the body", Some(Reply::Reset), &[502, 503]),
        ("header stall", Some(Reply::HeaderStall), &[502, 504]),
        (
            "provider 503",
            Some(Reply::json(503, r#"{"error":{"message":"overloaded"}}"#)),
            &[503],
        ),
        (
            "provider 429",
            Some(Reply::json(429, r#"{"error":{"message":"slow down"}}"#)),
            &[429],
        ),
    ];
    for (what, reply, allowed) in classes {
        let up = match reply {
            Some(r) => Some(ReplyUpstream::start(move |_, _| r.clone()).await),
            None => None,
        };
        let authority = up
            .as_ref()
            .map(ReplyUpstream::authority)
            .unwrap_or_else(GatewayBuilder::dead_authority);
        let gw = Gateway::builder(unused_nats_port(), &authority, &b64(&pubkey))
            .providers(&["openai"])
            .config_line("read_timeout_secs = 2")
            .config_line("connect_timeout_secs = 2")
            .start()
            .await;
        let start = Instant::now();
        let resp = post(&gw, "/openai/v1/chat/completions", &key, CHAT).await;
        let status = resp.status().as_u16();
        let _ = resp.bytes().await;
        let took = start.elapsed();
        assert!(
            took < Duration::from_secs(10),
            "{what}: took {took:?} (unbounded)"
        );
        assert!(
            allowed.contains(&status),
            "{what}: status {status}, want one of {allowed:?}"
        );
    }
}

/// Every candidate of a catalog walk 5xxes: the client gets a JSON error body, the request id it
/// can quote, and a 5xx — the last provider's own answer.
/// claim: REL-2
#[tokio::test]
async fn every_candidate_5xx_ends_in_a_json_error_with_a_request_id() {
    let primary = MockUpstream::start(Mode::Status(500)).await;
    let fallback = MockUpstream::start(Mode::Status(503)).await;
    let (pubkey, sk) = test_keypair(21);
    let gw = Gateway::builder(unused_nats_port(), &primary.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .start()
        .await;
    let resp = post(
        &gw,
        "/v1/chat/completions",
        &billing_vkey(&sk, 21),
        r#"{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}"#,
    )
    .await;
    let status = resp.status().as_u16();
    assert!(resp.headers().contains_key("x-beyond-request-id"));
    let text = resp.text().await.unwrap();
    assert_eq!(status, 503, "the last candidate's status: {text}");
    let json: serde_json::Value = serde_json::from_str(&text).expect("a JSON body");
    assert!(json["error"]["message"].is_string(), "{text}");
    assert_eq!((primary.hits(), fallback.hits()), (1, 1));
}

// --- upstream H2 ---------------------------------------------------------------------------------

/// Many concurrent requests over the gateway's H2 upstream connection all succeed, negotiated as
/// h2, and run concurrently rather than queueing behind one another.
/// claim: REL-22
#[tokio::test]
async fn upstream_h2_multiplexes_concurrent_requests() {
    let mock = MockUpstream::start_tls(Mode::Slow(400)).await;
    let (pubkey, sk) = test_keypair(221);
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .tls_upstream()
        .upstream_http2(true)
        .start()
        .await;
    let key = billing_vkey(&sk, 2201);
    let start = Instant::now();
    let mut tasks = Vec::new();
    for _ in 0..32 {
        let (url, key) = (gw.url(), key.clone());
        tasks.push(tokio::spawn(async move {
            let resp = test_client()
                .post(format!("{url}/openai/v1/chat/completions"))
                .header("authorization", format!("Bearer {key}"))
                .header("content-type", "application/json")
                .body(CHAT)
                .send()
                .await
                .unwrap();
            let proto = resp
                .headers()
                .get("x-mock-proto")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("")
                .to_owned();
            (resp.status().as_u16(), proto)
        }));
    }
    for t in tasks {
        let (status, proto) = t.await.unwrap();
        assert_eq!(status, 200);
        assert_eq!(proto, "h2");
    }
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "32 × 400ms requests took {:?}: serialized",
        start.elapsed()
    );
}

/// A TLS H2 upstream that sends GOAWAY (graceful) on every connection right after its first
/// request — what a provider's load balancer does when it drains a node.
async fn goaway_upstream() -> (u16, Arc<AtomicUsize>, tokio::task::JoinHandle<()>) {
    use http_body_util::{BodyExt, Full};
    use hyper::service::service_fn;
    use hyper_util::rt::{TokioExecutor, TokioIo};
    let _ = rustls::crypto::ring::default_provider().install_default();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let ck = rcgen::generate_simple_self_signed(vec!["127.0.0.1".to_string()]).unwrap();
    let key = rustls::pki_types::PrivateKeyDer::Pkcs8(ck.key_pair.serialize_der().into());
    let mut tls = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(vec![ck.cert.der().clone()], key)
        .unwrap();
    tls.alpn_protocols = vec![b"h2".to_vec()];
    let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(tls));
    let served = Arc::new(AtomicUsize::new(0));
    let counter = served.clone();
    let task = tokio::spawn(async move {
        while let Ok((s, _)) = listener.accept().await {
            let (acceptor, counter) = (acceptor.clone(), counter.clone());
            tokio::spawn(async move {
                let Ok(tls) = acceptor.accept(s).await else {
                    return;
                };
                let first = Arc::new(tokio::sync::Notify::new());
                let notify = first.clone();
                let svc = service_fn(move |req: hyper::Request<hyper::body::Incoming>| {
                    let (notify, counter) = (notify.clone(), counter.clone());
                    async move {
                        let _ = req.into_body().collect().await;
                        counter.fetch_add(1, Ordering::SeqCst);
                        notify.notify_one();
                        Ok::<_, std::convert::Infallible>(
                            hyper::Response::builder()
                                .header("content-type", "application/json")
                                .body(Full::new(Bytes::from_static(OK_JSON.as_bytes())))
                                .unwrap(),
                        )
                    }
                });
                let conn = hyper::server::conn::http2::Builder::new(TokioExecutor::new())
                    .serve_connection(TokioIo::new(tls), svc);
                tokio::pin!(conn);
                tokio::select! {
                    _ = conn.as_mut() => {}
                    _ = first.notified() => {
                        conn.as_mut().graceful_shutdown();
                        let _ = conn.await;
                    }
                }
            });
        }
    });
    (port, served, task)
}

/// How [`refusing_h2_upstream`] refuses every stream after a connection's first.
#[derive(Clone, Copy, Debug)]
enum Refuse {
    /// `GOAWAY(last_stream_id = the stream it served, NO_ERROR)`: the new stream is above it, so
    /// RFC 9113 §6.8 guarantees it was not processed.
    GoAway,
    /// `RST_STREAM(REFUSED_STREAM)`: §8.7, closed before any processing.
    RefusedStream,
}

/// Write one H2 frame.
async fn h2_frame<W: tokio::io::AsyncWrite + Unpin>(
    w: &mut W,
    kind: u8,
    flags: u8,
    stream: u32,
    payload: &[u8],
) -> std::io::Result<()> {
    let len = u32::try_from(payload.len()).unwrap().to_be_bytes();
    let mut buf = vec![len[1], len[2], len[3], kind, flags];
    buf.extend_from_slice(&stream.to_be_bytes());
    buf.extend_from_slice(payload);
    w.write_all(&buf).await?;
    w.flush().await
}

/// A hand-rolled TLS H2 upstream that refuses the second request on each connection, and only
/// once its whole body has arrived — the moment the gateway counts the body as delivered. Every
/// other request is answered (a GOAWAY ends the connection, so it has no third). Hand-rolled because no server library lets a test choose the GOAWAY's
/// `last_stream_id`. Returns (port, requests served, streams refused, task).
async fn refusing_h2_upstream(
    mode: Refuse,
) -> (
    u16,
    Arc<AtomicUsize>,
    Arc<AtomicUsize>,
    tokio::task::JoinHandle<()>,
) {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let ck = rcgen::generate_simple_self_signed(vec!["127.0.0.1".to_string()]).unwrap();
    let key = rustls::pki_types::PrivateKeyDer::Pkcs8(ck.key_pair.serialize_der().into());
    let mut tls = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(vec![ck.cert.der().clone()], key)
        .unwrap();
    tls.alpn_protocols = vec![b"h2".to_vec()];
    let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(tls));
    let (served, refused) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let (s2, r2) = (served.clone(), refused.clone());
    let task = tokio::spawn(async move {
        while let Ok((s, _)) = listener.accept().await {
            let (acceptor, served, refused) = (acceptor.clone(), s2.clone(), r2.clone());
            tokio::spawn(async move {
                let Ok(mut io) = acceptor.accept(s).await else {
                    return;
                };
                let mut preface = [0u8; 24];
                if io.read_exact(&mut preface).await.is_err() {
                    return;
                }
                if h2_frame(&mut io, 0x4, 0, 0, &[]).await.is_err() {
                    return;
                }
                // `:status: 200` (static index 8), then `content-type` (static name 31) as a
                // literal without indexing.
                let mut head = vec![0x88, 0x0f, 0x10, 16];
                head.extend_from_slice(b"application/json");
                let (mut seen, mut last_served) = (0u32, 0u32);
                loop {
                    let mut h = [0u8; 9];
                    if io.read_exact(&mut h).await.is_err() {
                        return;
                    }
                    let len = usize::from(h[0]) << 16 | usize::from(h[1]) << 8 | usize::from(h[2]);
                    let (kind, flags) = (h[3], h[4]);
                    let stream = u32::from_be_bytes([h[5], h[6], h[7], h[8]]) & 0x7fff_ffff;
                    let mut payload = vec![0u8; len];
                    if io.read_exact(&mut payload).await.is_err() {
                        return;
                    }
                    let ok = match (kind, flags & 0x1) {
                        // SETTINGS → ACK; PING → PONG.
                        (0x4, 0) => h2_frame(&mut io, 0x4, 0x1, 0, &[]).await,
                        (0x6, 0) => h2_frame(&mut io, 0x6, 0x1, 0, &payload).await,
                        // HEADERS or DATA ending the request body.
                        (0x0 | 0x1, 0x1) if seen != 1 => {
                            seen += 1;
                            last_served = stream;
                            served.fetch_add(1, Ordering::SeqCst);
                            let r = h2_frame(&mut io, 0x1, 0x4, stream, &head).await;
                            match r {
                                Ok(()) => {
                                    h2_frame(&mut io, 0x0, 0x1, stream, OK_JSON.as_bytes()).await
                                }
                                e => e,
                            }
                        }
                        (0x0 | 0x1, 0x1) => {
                            seen += 1;
                            refused.fetch_add(1, Ordering::SeqCst);
                            match mode {
                                Refuse::GoAway => {
                                    let mut p = last_served.to_be_bytes().to_vec();
                                    p.extend_from_slice(&0u32.to_be_bytes());
                                    let _ = h2_frame(&mut io, 0x7, 0, 0, &p).await;
                                    // Hold the connection open a moment, as a draining server
                                    // does, then close it.
                                    tokio::time::sleep(Duration::from_millis(200)).await;
                                    let _ = io.shutdown().await;
                                    return;
                                }
                                Refuse::RefusedStream => {
                                    h2_frame(&mut io, 0x3, 0, stream, &7u32.to_be_bytes()).await
                                }
                            }
                        }
                        _ => Ok(()),
                    };
                    if ok.is_err() {
                        return;
                    }
                }
            });
        }
    });
    (port, served, refused, task)
}

/// A request whose stream the upstream refused — GOAWAY with a `last_stream_id` below it, or
/// RST_STREAM(REFUSED_STREAM) — was not processed, so it is retried on a new connection even
/// though its whole body went out. The "never resend a delivered body" rule (D09) is about a
/// provider that may be generating; a refused stream is the provider's guarantee it is not.
/// claim: REL-22
/// defect: D72
#[tokio::test]
async fn a_stream_the_upstream_refused_is_retried() {
    let (pubkey, sk) = test_keypair(223);
    let key = billing_vkey(&sk, 2203);
    let path = "/openai/v1/chat/completions";
    for mode in [Refuse::GoAway, Refuse::RefusedStream] {
        let (port, served, refused, task) = refusing_h2_upstream(mode).await;
        let gw = Gateway::builder(
            unused_nats_port(),
            &format!("127.0.0.1:{port}"),
            &b64(&pubkey),
        )
        .providers(&["openai"])
        .tls_upstream()
        .upstream_http2(true)
        .start()
        .await;
        for i in 0..4 {
            let resp = post(&gw, path, &key, CHAT).await;
            let status = resp.status().as_u16();
            let text = resp.text().await.unwrap_or_default();
            assert_eq!(status, 200, "{mode:?} #{i}: {text}");
        }
        task.abort();
        assert!(
            refused.load(Ordering::SeqCst) >= 1,
            "{mode:?}: no stream was refused, so the test proved nothing"
        );
        assert_eq!(served.load(Ordering::SeqCst), 4, "{mode:?}");
    }
}

/// Every upstream H2 connection is drained with GOAWAY after one request. Sequential and
/// concurrent requests all still succeed: the gateway opens a new connection rather than failing
/// the request that met the GOAWAY.
/// claim: REL-22
#[tokio::test]
async fn upstream_goaway_is_handled() {
    let (port, served, task) = goaway_upstream().await;
    let (pubkey, sk) = test_keypair(222);
    let gw = Gateway::builder(
        unused_nats_port(),
        &format!("127.0.0.1:{port}"),
        &b64(&pubkey),
    )
    .providers(&["openai"])
    .tls_upstream()
    .upstream_http2(true)
    .start()
    .await;
    let key = billing_vkey(&sk, 2202);
    let path = "/openai/v1/chat/completions";
    for i in 0..10 {
        assert_eq!(
            status_of(&gw, path, &key, CHAT).await,
            200,
            "sequential #{i}"
        );
    }
    let mut tasks = Vec::new();
    for _ in 0..16 {
        let (url, key) = (gw.url(), key.clone());
        tasks.push(tokio::spawn(async move {
            test_client()
                .post(format!("{url}{path}"))
                .header("authorization", format!("Bearer {key}"))
                .header("content-type", "application/json")
                .body(CHAT)
                .send()
                .await
                .map(|r| r.status().as_u16())
                .unwrap_or(0)
        }));
    }
    let mut statuses = Vec::new();
    for t in tasks {
        statuses.push(t.await.unwrap());
    }
    task.abort();
    assert!(
        statuses.iter().all(|s| *s == 200),
        "concurrent requests across GOAWAYs: {statuses:?}"
    );
    assert!(served.load(Ordering::SeqCst) >= 26);
}
