//! Reliability, verify phase 0: readiness, shutdown, stalls, slow readers, body memory and DNS.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::time::{Duration, Instant};

use beyond_ai::key::{VirtualKey, mint};
use common::*;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

fn vkey(sk: &ed25519_dalek::SigningKey, tenant_id: u64) -> String {
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

const STREAM_BODY: &str =
    r#"{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"hi"}]}"#;

/// NATS is down at boot, so the allowance set never seeds and every managed request is refused
/// with "allowance unavailable". Readiness must say so, or the load balancer keeps sending traffic
/// to a pod that refuses all of it.
/// claim: REL-5
/// defect: D11
#[tokio::test]
async fn readyz_is_not_ready_while_allowance_is_unseeded() {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(closed_port(), &mock.authority(), &b64(&pubkey))
        .skip_allowance_ready()
        .start()
        .await;
    let resp = test_client()
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk, 1)))
        .header("content-type", "application/json")
        .body(r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#)
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    let text = resp.text().await.unwrap();
    assert!(
        text.contains("allowance unavailable"),
        "precondition: managed traffic is refused ({status} {text})"
    );
    let (ready, body) = gw.admin_get("/readyz").await;
    assert_ne!(
        ready, 200,
        "readyz {ready} {body} while managed requests get {status} allowance unavailable"
    );
}

/// An idle gateway exits promptly on SIGTERM: there is nothing to drain. Default config (600s
/// grace).
/// claim: REL-16, BIL-21
/// defect: D38
#[tokio::test]
async fn an_idle_gateway_exits_promptly_on_sigterm() {
    let (pubkey, _sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let mut gw = Gateway::start(unused_nats_port(), &mock.authority(), &b64(&pubkey)).await;
    gw.sigterm();
    let exited = gw.wait_exit(Duration::from_secs(15)).await;
    assert!(
        exited.is_some_and(|t| t < Duration::from_secs(10)),
        "idle gateway exit after SIGTERM: {exited:?} (None = still running at 15s)"
    );
}

/// The same with a short configured grace: shutdown time tracks the grace, even when idle. The
/// drain's own line shows what ended the process (pingora's path out waits the grace, then the
/// runtime timeout), so the time bound need only be the grace, which a loaded host cannot stretch
/// a sub-second drain past.
/// claim: REL-16, BIL-21
/// defect: D38
#[tokio::test]
async fn an_idle_gateway_with_a_short_grace_exits_before_the_grace() {
    let (pubkey, _sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let mut gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .config_line("shutdown_grace_period_secs = 10")
        .config_line("shutdown_runtime_timeout_secs = 1")
        // The drain's line is `info`.
        .env("AI_LOG", "info")
        .start()
        .await;
    gw.sigterm();
    let exited = gw.wait_exit(CONDITION_BUDGET).await;
    assert!(
        exited.is_some_and(|t| t < Duration::from_secs(10)),
        "idle gateway (10s grace) exit after SIGTERM: {exited:?}"
    );
    gw.wait_for_log_line(&["drained: no request in flight; exiting"])
        .await;
}

/// `read_timeout_secs` is honored: a header stall ends at the configured bound.
/// claim: REL-7
/// defect: D36
#[tokio::test]
async fn a_header_stall_ends_at_the_configured_read_timeout() {
    let (pubkey, _sk) = test_keypair(1);
    let mock = ReplyUpstream::start(|_, _| Reply::HeaderStall).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .config_line("read_timeout_secs = 2")
        .start()
        .await;
    let start = Instant::now();
    let status = test_client()
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header("authorization", "Bearer sk-byo-test")
        .header("content-type", "application/json")
        .body(STREAM_BODY)
        .send()
        .await
        .map(|r| r.status().as_u16())
        .unwrap_or(0);
    let took = start.elapsed();
    // Unloaded it ends at the 2s bound; the bound it must not wait out instead is the default 600s
    // (and the client gives up at 30s, as status 0).
    assert!(
        status >= 500 && took < Duration::from_secs(20),
        "status {status} after {took:?}"
    );
}

/// A TLS HTTP/2 upstream. Each request waits `head_delay`, then gets a 200 SSE head and one event
/// and no more. With `freeze`, the connection stops being driven shortly after that event: the
/// socket stays open and the kernel still ACKs TCP, but HTTP/2 PINGs go unanswered — a wedged
/// peer, the case only H2 PING can see.
async fn h2_upstream(head_delay: Duration, freeze: bool) -> (u16, tokio::task::JoinHandle<()>) {
    use hyper::service::service_fn;
    use hyper_util::rt::{TokioExecutor, TokioIo};
    use std::sync::Arc;
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
    let task = tokio::spawn(async move {
        while let Ok((s, _)) = listener.accept().await {
            let acceptor = acceptor.clone();
            tokio::spawn(async move {
                let Ok(tls) = acceptor.accept(s).await else {
                    return;
                };
                let answered = Arc::new(tokio::sync::Notify::new());
                let notify = answered.clone();
                let svc = service_fn(move |_req: hyper::Request<hyper::body::Incoming>| {
                    let notify = notify.clone();
                    async move {
                        tokio::time::sleep(head_delay).await;
                        notify.notify_one();
                        Ok::<_, std::convert::Infallible>(
                            hyper::Response::builder()
                                .header("content-type", "text/event-stream")
                                .body(StallingBody::new(bytes::Bytes::from_static(FIRST_EVENT)))
                                .unwrap(),
                        )
                    }
                });
                let conn = hyper::server::conn::http2::Builder::new(TokioExecutor::new())
                    .serve_connection(TokioIo::new(tls), svc);
                tokio::pin!(conn);
                if !freeze {
                    let _ = conn.await;
                    return;
                }
                tokio::select! {
                    _ = conn.as_mut() => return,
                    _ = answered.notified() => {}
                }
                // Let the head and the event go out, then stop driving the connection while
                // keeping it (and its socket) open.
                let until = tokio::time::Instant::now() + Duration::from_millis(300);
                tokio::select! {
                    _ = conn.as_mut() => return,
                    _ = tokio::time::sleep_until(until) => {}
                }
                std::future::pending::<()>().await;
            });
        }
    });
    (port, task)
}

const FIRST_EVENT: &[u8] = b"data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n";

fn long_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(40))
        .build()
        .unwrap()
}

/// Post a stream request and read the response to its end: `(status, body or error, elapsed)`.
async fn stream_to_end(gw: &Gateway) -> (u16, Result<String, String>, Duration) {
    let start = Instant::now();
    let resp = long_client()
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header("authorization", "Bearer sk-byo-test")
        .header("content-type", "application/json")
        .body(STREAM_BODY)
        .send()
        .await;
    let resp = match resp {
        Ok(r) => r,
        Err(e) => return (0, Err(e.to_string()), start.elapsed()),
    };
    let status = resp.status().as_u16();
    let body = resp.text().await.map_err(|e| e.to_string());
    (status, body, start.elapsed())
}

/// A provider connection that goes dead mid-stream (the peer stops answering HTTP/2 PINGs while
/// its socket stays open) is detected by transport liveness and failed promptly — not held until
/// `read_timeout_secs` (600s). Measured with a 1s PING interval: detection within the interval plus
/// pingora's 5s PING ACK deadline. Silence alone would never end it: the read timeout is the
/// default 600s.
/// claim: REL-7
/// defect: D36
#[tokio::test]
async fn a_dead_h2_upstream_is_detected_by_ping() {
    let defaults = beyond_ai::config::AiConfig::default();
    assert!(
        defaults.h2_ping_interval_secs > 0 && defaults.h2_ping_interval_secs + 5 < 60,
        "upstream H2 PING is on by default, with a bound far under any client timeout: {}s",
        defaults.h2_ping_interval_secs
    );
    let (port, _task) = h2_upstream(Duration::ZERO, true).await;
    let (pubkey, _sk) = test_keypair(1);
    let gw = Gateway::builder(
        unused_nats_port(),
        &format!("127.0.0.1:{port}"),
        &b64(&pubkey),
    )
    .tls_upstream()
    .upstream_http2(true)
    .config_line("h2_ping_interval_secs = 1")
    .start()
    .await;
    let (status, body, took) = stream_to_end(&gw).await;
    assert_eq!(status, 200, "{body:?}");
    // Unloaded: about 6s. Under the client's own 40s timeout, which would end it as an error too.
    assert!(
        took < Duration::from_secs(30),
        "a dead upstream was held for {took:?}: {body:?}"
    );
    assert!(
        body.as_ref().is_err() || body.as_ref().is_ok_and(|b| !b.contains("[DONE]")),
        "a dead upstream must end as an error, never a clean finish: {body:?}"
    );
}

/// The other half of the contract: a provider that is alive but silent — a model thinking without
/// emitting — is never cut by the gateway before `read_timeout_secs`, the clients' own 600s
/// default. Silence is not a failure signal; only a dead transport is. Each upstream here holds
/// its response head for 8s, several times the configured liveness bound (H2 PING every 1s with a
/// 5s ACK deadline; TCP keepalive probing after 1s idle, every 1s, 2 probes), and the stream
/// arrives: the PINGs and probes are answered by the peer's HTTP/2 stack and kernel, not the
/// model. The old silence cut (`stream_idle_timeout_secs`, 120s by default) would have ended these
/// at its bound, configured to 2s in the tests it replaced.
/// claim: REL-7
/// defect: D36
#[tokio::test]
async fn an_alive_but_silent_upstream_is_not_cut() {
    let defaults = beyond_ai::config::AiConfig::default();
    assert_eq!(
        defaults.read_timeout_secs, 600,
        "the silence bound is the OpenAI and Anthropic SDKs' 600s default request timeout"
    );
    let liveness = [
        "h2_ping_interval_secs = 1",
        "tcp_keepalive_idle_secs = 1",
        "tcp_keepalive_interval_secs = 1",
        "tcp_keepalive_count = 2",
    ];
    let (pubkey, _sk) = test_keypair(1);
    let silence = Duration::from_secs(8);
    // HTTP/2: the peer keeps answering PINGs while the model is silent.
    let (port, _task) = h2_upstream(silence, false).await;
    let mut h2 = Gateway::builder(
        unused_nats_port(),
        &format!("127.0.0.1:{port}"),
        &b64(&pubkey),
    )
    .tls_upstream()
    .upstream_http2(true);
    // HTTP/1.1: only TCP keepalive watches the connection.
    let h1_up =
        ReplyUpstream::start(move |_, _| Reply::Delayed(silence, Box::new(Reply::sse()))).await;
    let mut h1 = Gateway::builder(unused_nats_port(), &h1_up.authority(), &b64(&pubkey));
    for line in liveness {
        h2 = h2.config_line(line);
        h1 = h1.config_line(line);
    }
    let (h2, h1) = (h2.start().await, h1.start().await);
    let h2_run = async {
        // This upstream stalls after its first event by design: the head and event arriving
        // after the silence is what is asserted.
        let start = Instant::now();
        let mut resp = long_client()
            .post(format!("{}/openai/v1/chat/completions", h2.url()))
            .header("authorization", "Bearer sk-byo-test")
            .header("content-type", "application/json")
            .body(STREAM_BODY)
            .send()
            .await
            .unwrap();
        let status = resp.status().as_u16();
        let first = resp.chunk().await.ok().flatten().unwrap_or_default();
        let took = start.elapsed();
        (status, String::from_utf8_lossy(&first).into_owned(), took)
    };
    let ((status, first, took), (h1_status, body, h1_took)) =
        tokio::join!(h2_run, stream_to_end(&h1));
    assert!(
        status == 200 && first.contains("\"hi\"") && took >= silence,
        "h2: an alive, silent upstream was cut: {status} after {took:?}: {first}"
    );
    assert!(
        h1_status == 200 && body.as_ref().is_ok_and(|b| b.contains("[DONE]")) && h1_took >= silence,
        "h1: an alive, silent upstream was cut: {h1_status} after {h1_took:?}: {body:?}"
    );
}

/// The kernel half of liveness is armed: the gateway's upstream socket and its accepted client
/// socket both carry TCP keepalive, read back from `/proc/net/tcp` (timer 2 = keepalive). A
/// vanished peer cannot be simulated without dropping packets, so this pins the configuration the
/// kernel acts on; the probes themselves are the kernel's.
/// claim: REL-7
#[cfg(target_os = "linux")]
#[tokio::test]
async fn upstream_and_client_sockets_arm_tcp_keepalive() {
    let mock = ReplyUpstream::start(|_, _| Reply::HeaderStall).await;
    let (pubkey, _sk) = test_keypair(1);
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .config_line("tcp_keepalive_idle_secs = 7")
        .start()
        .await;
    let mut s = tokio::net::TcpStream::connect(("127.0.0.1", gw.port))
        .await
        .unwrap();
    let req = format!(
        "POST /openai/v1/chat/completions HTTP/1.1\r\nhost: 127.0.0.1\r\n\
         authorization: Bearer sk-byo-test\r\ncontent-type: application/json\r\n\
         content-length: {}\r\n\r\n{STREAM_BODY}",
        STREAM_BODY.len()
    );
    s.write_all(req.as_bytes()).await.unwrap();
    // `(local port, remote port, timer)` of every established IPv4 socket.
    let sockets = || -> Vec<(u16, u16, u8)> {
        let table = std::fs::read_to_string("/proc/net/tcp").unwrap();
        let port = |addr: &str| u16::from_str_radix(addr.rsplit(':').next().unwrap(), 16).unwrap();
        table
            .lines()
            .skip(1)
            .filter_map(|l| {
                let f: Vec<&str> = l.split_whitespace().collect();
                let timer = f.get(5)?.split(':').next()?.parse().ok()?;
                (f.get(3) == Some(&"01")).then(|| (port(f[1]), port(f[2]), timer))
            })
            .collect()
    };
    let start = Instant::now();
    let (mut upstream, mut client) = (None, None);
    while start.elapsed() < Duration::from_secs(10) && (upstream != Some(2) || client != Some(2)) {
        for (local, remote, timer) in sockets() {
            if remote == mock.port {
                upstream = Some(timer);
            }
            if local == gw.port {
                client = Some(timer);
            }
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    drop(s);
    assert_eq!(upstream, Some(2), "upstream socket keepalive timer");
    assert_eq!(client, Some(2), "client socket keepalive timer");
}

/// A client that sends a request for a large stream and then never reads must be released (its
/// in-flight slot, upstream connection, gauge) within a bound. The bound is the downstream write
/// timeout: 60s by default (asserted below), configured to 3s here so the release is observed
/// inside the test's 20s window rather than after a minute.
/// claim: REL-10
/// defect: D42
#[tokio::test]
async fn a_client_that_stops_reading_is_released() {
    assert_eq!(
        beyond_ai::config::AiConfig::default().client_write_timeout_secs,
        60,
        "a default downstream write timeout exists"
    );
    let (pubkey, _sk) = test_keypair(1);
    let big = {
        let event = "data: {\"choices\":[{\"delta\":{\"content\":\"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\"}}]}\n\n";
        bytes::Bytes::from(event.repeat((64 << 20) / event.len()))
    };
    let mock = ReplyUpstream::start(move |_, _| Reply::Full {
        status: 200,
        content_type: "text/event-stream",
        body: big.clone(),
    })
    .await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .config_line("client_write_timeout_secs = 3")
        .start()
        .await;
    let mut s = tokio::net::TcpStream::connect(("127.0.0.1", gw.port))
        .await
        .unwrap();
    let req = format!(
        "POST /openai/v1/chat/completions HTTP/1.1\r\nhost: 127.0.0.1\r\n\
         authorization: Bearer sk-byo-test\r\ncontent-type: application/json\r\n\
         content-length: {}\r\n\r\n{STREAM_BODY}",
        STREAM_BODY.len()
    );
    s.write_all(req.as_bytes()).await.unwrap();
    // Read nothing, ever. Wait for the request to be in flight, then for it to be released.
    wait_for_metric(&gw, "ai_requests_in_flight", "", 1.0).await;
    let start = Instant::now();
    let mut in_flight = 1.0;
    while start.elapsed() < Duration::from_secs(20) {
        in_flight = gw.metric("ai_requests_in_flight", "").await;
        if in_flight == 0.0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    drop(s);
    assert_eq!(
        in_flight,
        0.0,
        "a non-reading client still holds its slot after {:?}",
        start.elapsed()
    );
}

/// Concurrent large uploads must not grow the gateway's memory without bound. Eight managed
/// catalog requests with ~90 MiB bodies, held while the upstream stalls, may not cost more than
/// 512 MiB of gateway RSS between them (the bodies alone are 720 MiB).
/// claim: SEC-19
/// defect: D35
#[tokio::test]
async fn concurrent_large_uploads_have_bounded_memory() {
    const N: u64 = 8;
    const BODY_MIB: usize = 90;
    let (pubkey, sk) = test_keypair(1);
    let mock = ReplyUpstream::start(|_, _| Reply::HeaderStall).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .start()
        .await;
    let base = gw.rss_kib();
    let body = std::sync::Arc::new({
        let mut b = br#"{"model":"gpt-4o-mini","messages":[{"role":"user","content":""#.to_vec();
        b.extend(std::iter::repeat_n(b'x', BODY_MIB << 20));
        b.extend_from_slice(br#""}]}"#);
        b
    });
    let mut conns = Vec::new();
    for i in 0..N {
        let (body, port, key) = (body.clone(), gw.port, vkey(&sk, 100 + i));
        conns.push(tokio::spawn(async move {
            let mut s = tokio::net::TcpStream::connect(("127.0.0.1", port))
                .await
                .unwrap();
            let head = format!(
                "POST /v1/chat/completions HTTP/1.1\r\nhost: 127.0.0.1\r\n\
                 authorization: Bearer {key}\r\ncontent-type: application/json\r\n\
                 content-length: {}\r\n\r\n",
                body.len()
            );
            let _ = s.write_all(head.as_bytes()).await;
            let _ = s.write_all(&body).await;
            // Keep the connection open while the test samples memory.
            let mut buf = [0u8; 1024];
            let _ = tokio::time::timeout(Duration::from_secs(20), s.read(&mut buf)).await;
        }));
    }
    let start = Instant::now();
    let mut peak = base;
    while start.elapsed() < Duration::from_secs(12) {
        peak = peak.max(gw.rss_kib());
        if mock.hits() >= N as usize && start.elapsed() > Duration::from_secs(2) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    peak = peak.max(gw.rss_kib());
    let grew_mib = (peak.saturating_sub(base)) / 1024;
    for c in conns {
        c.abort();
    }
    assert!(
        grew_mib < 512,
        "gateway RSS grew {grew_mib} MiB for {N} concurrent {BODY_MIB} MiB uploads (upstream saw {})",
        mock.hits()
    );
}

/// A provider authority whose name resolves to more than one address must be reachable when the
/// first one is dead. `localhost` resolves to `::1` first here and the mock listens only on
/// `127.0.0.1`.
/// claim: REL-20
/// defect: D40
#[tokio::test]
async fn dns_tries_the_next_address_when_the_first_is_dead() {
    let addrs: Vec<_> = tokio::net::lookup_host("localhost:80")
        .await
        .map(|a| a.collect())
        .unwrap_or_default();
    let v6_first = addrs.first().is_some_and(|a| a.is_ipv6());
    if !(v6_first && addrs.iter().any(|a| a.is_ipv4())) {
        eprintln!("SKIP: localhost does not resolve ::1 before 127.0.0.1 here ({addrs:?})");
        return;
    }
    let (pubkey, _sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .provider_authority("openai", &format!("localhost:{}", mock.port))
        .start()
        .await;
    let status = test_client()
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header("authorization", "Bearer sk-byo-test")
        .header("content-type", "application/json")
        .body(r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#)
        .send()
        .await
        .map(|r| r.status().as_u16())
        .unwrap_or(0);
    assert_eq!(
        status,
        200,
        "localhost -> {addrs:?}; upstream hits {}",
        mock.hits()
    );
}

/// A panic inside a proxy phase skips `logging`, which is where a request gives back what it holds.
/// The request context's drop must give it back instead: the in-flight and SSE gauges return to 0
/// and the tenant's only concurrency slot is free for its next request. `AI_FAULT_PANIC` (debug
/// builds only) panics the first request to reach `response_body_filter`, mid-stream.
/// claim: REL-17
/// defect: D39
#[tokio::test]
async fn a_panic_in_a_proxy_phase_releases_what_the_request_held() {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Sse).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .tenant_max_in_flight(1)
        .env("AI_FAULT_PANIC", "response_body_filter")
        .start()
        .await;
    let key = vkey(&sk, 39);
    let send = || {
        test_client()
            .post(format!("{}/openai/v1/chat/completions", gw.url()))
            .header("authorization", format!("Bearer {key}"))
            .header("content-type", "application/json")
            .body(STREAM_BODY)
            .send()
    };
    // The panicking request: the connection dies mid-response, whatever the client makes of it.
    let first = match send().await {
        Ok(r) => r.text().await.map(|_| ()).map_err(|e| e.to_string()),
        Err(e) => Err(e.to_string()),
    };
    // The panic message reaches the captured log through a reader thread, which can trail the
    // dropped connection under load: wait for it rather than read the log once.
    eprintln!("first request: {first:?}");
    gw.wait_for_log_line(&["AI_FAULT_PANIC"]).await;
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(5)
        && (gw.metric("ai_requests_in_flight", "").await != 0.0
            || gw.metric("ai_active_streams", "").await != 0.0)
    {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert_eq!(gw.metric("ai_requests_in_flight", "").await, 0.0);
    assert_eq!(gw.metric("ai_active_streams", "").await, 0.0);
    // The tenant's single slot came back: its next request is served, not 429.
    let second = send().await.unwrap();
    assert_eq!(second.status().as_u16(), 200);
}

/// The drain waits for in-flight work: a request already running when SIGTERM lands finishes with
/// its answer and its billing row, and the process exits right after it, not at the grace's end.
/// claim: REL-16, BIL-21
/// defect: D38
#[tokio::test]
async fn sigterm_drains_an_in_flight_request_then_exits() {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Slow(1500)).await;
    // `info`: the drain's line, and the billing row.
    let mut gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .env("AI_LOG", "info")
        .start()
        .await;
    let (url, key) = (gw.url(), vkey(&sk, 38));
    let held = tokio::spawn(async move {
        test_client()
            .post(format!("{url}/openai/v1/chat/completions"))
            .header("authorization", format!("Bearer {key}"))
            .header("content-type", "application/json")
            .body(r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#)
            .send()
            .await
            .map(|r| r.status().as_u16())
    });
    wait_for_metric(&gw, "ai_requests_in_flight", "", 1.0).await;
    gw.sigterm();
    // The grace is the default 600s: an exit at all within the budget is the drain's, and its own
    // line says so.
    let exited = gw.wait_exit(CONDITION_BUDGET).await;
    let status = held.await.unwrap();
    assert_eq!(status.ok(), Some(200), "the in-flight request was cut");
    assert!(exited.is_some(), "no exit after the drain: {exited:?}");
    gw.wait_for_log_line(&["drained: no request in flight; exiting"])
        .await;
    assert!(
        gw.log().contains("\"target\":\"ai.usage\""),
        "the drained request's billing row was not written"
    );
}

/// The catalog walk does the same: a candidate whose first address is dead is tried on its next
/// address before the walk gives up on it.
/// claim: REL-20
/// defect: D40
#[tokio::test]
async fn a_catalog_candidate_is_tried_on_its_next_address() {
    let addrs: Vec<_> = tokio::net::lookup_host("localhost:80")
        .await
        .map(|a| a.collect())
        .unwrap_or_default();
    let v6_first = addrs.first().is_some_and(|a| a.is_ipv6());
    if !(v6_first && addrs.iter().any(|a| a.is_ipv4())) {
        eprintln!("SKIP: localhost does not resolve ::1 before 127.0.0.1 here ({addrs:?})");
        return;
    }
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openai", &format!("localhost:{}", mock.port))
        .provider_authority("openrouter", &GatewayBuilder::dead_authority())
        .start()
        .await;
    let resp = test_client()
        .post(format!("{}/auto/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk, 40)))
        .header("content-type", "application/json")
        .header("x-beyond-model", "gpt-4o-mini")
        .header("x-beyond-order", "openai,openrouter")
        .body(r#"{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(
        (
            resp.status().as_u16(),
            resp.headers()
                .get("x-beyond-provider")
                .and_then(|v| v.to_str().ok())
                .map(String::from)
        ),
        (200, Some("openai".to_string())),
        "upstream hits {}",
        mock.hits()
    );
}

/// A body the process budget cannot hold is refused before it is read: a 503 with `Retry-After`
/// and a JSON error, counted on `ai_rejections_total{reason="body_memory"}`. A body within the
/// budget is still served.
/// claim: SEC-19
/// defect: D35
#[tokio::test]
async fn a_body_past_the_memory_budget_is_a_retryable_503() {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .config_line("max_buffered_body_bytes = 1048576")
        .start()
        .await;
    let chat = |pad: usize| {
        format!(
            r#"{{"model":"gpt-4o-mini","messages":[{{"role":"user","content":"{}"}}]}}"#,
            "x".repeat(pad)
        )
    };
    let send = |body: String| {
        test_client()
            .post(format!("{}/v1/chat/completions", gw.url()))
            .header("authorization", format!("Bearer {}", vkey(&sk, 35)))
            .header("content-type", "application/json")
            .body(body)
            .send()
    };
    // 1 MiB declared, held twice: past the 1 MiB budget.
    let resp = send(chat(1 << 20)).await.unwrap();
    let status = resp.status().as_u16();
    let retry_after = resp.headers().get("retry-after").is_some();
    let text = resp.text().await.unwrap_or_default();
    assert!(
        status == 503 && retry_after && text.contains("too many large request bodies"),
        "status {status} retry-after {retry_after}: {text}"
    );
    assert_eq!(mock.hits(), 0, "refused before any upstream attempt");
    wait_for_metric(&gw, "ai_rejections_total", "body_memory", 1.0).await;
    // 200 KiB, held twice: within it.
    let ok = send(chat(200 * 1024)).await.unwrap();
    assert_eq!(ok.status().as_u16(), 200);
}

/// The up-front reservation refuses, before reading a byte of it, a body the budget can hold once
/// but not twice: a catalog walk holds a large body twice (the peek and its full-body re-run),
/// whether the body or the `x-beyond-model` header names the row. A buffered `/{provider}` body
/// is held once, and is refused before the upstream sees the request.
/// claim: SEC-19
/// defect: D35
#[tokio::test]
async fn a_large_body_reserves_every_copy_it_will_hold_before_it_is_read() {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .config_line("max_buffered_body_bytes = 1048576")
        .start()
        .await;
    let chat = |pad: usize| {
        format!(
            r#"{{"model":"gpt-4o-mini","messages":[{{"role":"user","content":"{}"}}]}}"#,
            "x".repeat(pad)
        )
    };
    let send = |path: &'static str, header: Option<&'static str>, body: String| {
        let mut req = test_client()
            .post(format!("{}{path}", gw.url()))
            .header("authorization", format!("Bearer {}", vkey(&sk, 35)))
            .header("content-type", "application/json");
        if let Some(model) = header {
            req = req.header("x-beyond-model", model);
        }
        req.body(body).send()
    };
    // 600 KiB fits the 1 MiB budget once, not twice. Declared and never sent: the refusal comes
    // from the declared length, before a byte of the body is read.
    for (what, header) in [
        ("body-named", ""),
        ("header-named", "x-beyond-model: gpt-4o-mini\r\n"),
    ] {
        let addr = gw.url().trim_start_matches("http://").to_owned();
        let mut s = tokio::net::TcpStream::connect(&addr).await.unwrap();
        let head = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nhost: gw\r\nauthorization: Bearer {}\r\n\
             content-type: application/json\r\n{header}content-length: {}\r\n\r\n",
            vkey(&sk, 35),
            600 * 1024
        );
        s.write_all(head.as_bytes()).await.unwrap();
        let mut buf = vec![0u8; 4096];
        let n = tokio::time::timeout(Duration::from_secs(5), s.read(&mut buf))
            .await
            .unwrap_or_else(|_| panic!("{what}: no answer until the body arrives"))
            .unwrap();
        let text = String::from_utf8_lossy(&buf[..n]);
        assert!(
            text.starts_with("HTTP/1.1 503") && text.contains("too many large request bodies"),
            "{what}: {text}"
        );
    }
    // 1.5 MiB declared on `/openai/…`, buffered for `stream_options`: past the budget even once,
    // and refused from the declared length too, before a byte of it is read.
    {
        let addr = gw.url().trim_start_matches("http://").to_owned();
        let mut s = tokio::net::TcpStream::connect(&addr).await.unwrap();
        let head = format!(
            "POST /openai/v1/chat/completions HTTP/1.1\r\nhost: gw\r\nauthorization: Bearer {}\r\n\
             content-type: application/json\r\ncontent-length: {}\r\n\r\n",
            vkey(&sk, 35),
            1536 * 1024
        );
        s.write_all(head.as_bytes()).await.unwrap();
        let mut buf = vec![0u8; 4096];
        let n = tokio::time::timeout(Duration::from_secs(5), s.read(&mut buf))
            .await
            .expect("provider route: no answer until the body arrives")
            .unwrap();
        let text = String::from_utf8_lossy(&buf[..n]);
        assert!(
            text.starts_with("HTTP/1.1 503") && text.contains("too many large request bodies"),
            "provider route, declared: {text}"
        );
    }
    let resp = send("/openai/v1/chat/completions", None, chat(1536 * 1024))
        .await
        .unwrap();
    let status = resp.status().as_u16();
    let text = resp.text().await.unwrap_or_default();
    assert!(
        status == 503 && text.contains("too many large request bodies"),
        "provider route: status {status}: {text}"
    );
    assert_eq!(
        mock.hits(),
        0,
        "every refusal came before an upstream attempt"
    );
}

/// A body pingora can replay from its own 64 KiB buffer is bounded by concurrency, not charged to
/// the body budget: 60 KiB and exactly 64 KiB are served under a budget smaller than two copies of
/// either. Only a body past the buffer reserves (the test above).
/// claim: SEC-19
/// defect: D35
#[tokio::test]
async fn a_body_within_the_replay_buffer_is_not_charged_to_the_budget() {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .config_line("max_buffered_body_bytes = 100000")
        .start()
        .await;
    let chat = |len: usize| {
        let empty = r#"{"model":"gpt-4o-mini","messages":[{"role":"user","content":""}]}"#;
        let body = format!(
            r#"{{"model":"gpt-4o-mini","messages":[{{"role":"user","content":"{}"}}]}}"#,
            "x".repeat(len - empty.len())
        );
        assert_eq!(body.len(), len);
        body
    };
    for len in [60 * 1024, 64 * 1024] {
        let resp = test_client()
            .post(format!("{}/v1/chat/completions", gw.url()))
            .header("authorization", format!("Bearer {}", vkey(&sk, 35)))
            .header("content-type", "application/json")
            .body(chat(len))
            .send()
            .await
            .unwrap();
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        assert_eq!(status, 200, "{len} B: {text}");
    }
    assert_eq!(mock.hits(), 2);
}

/// A chunked body (no `Content-Length`, so nothing to reserve up front) that outgrows the budget
/// while the walk reads it is the budget's retryable 503, not "request body too large": it is far
/// below the size cap.
/// claim: SEC-19
/// defect: D35
#[tokio::test]
async fn a_chunked_body_that_outgrows_the_budget_is_a_503_not_a_413() {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .config_line("max_buffered_body_bytes = 1048576")
        .start()
        .await;
    let mut s = tokio::net::TcpStream::connect(("127.0.0.1", gw.port))
        .await
        .unwrap();
    let head = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nhost: gw\r\nauthorization: Bearer {}\r\n\
         content-type: application/json\r\ntransfer-encoding: chunked\r\n\r\n",
        vkey(&sk, 35)
    );
    s.write_all(head.as_bytes()).await.unwrap();
    // 700 KiB in 64 KiB chunks: held twice, past the 1 MiB budget half way through.
    let open = br#"{"model":"gpt-4o-mini","messages":[{"role":"user","content":""#;
    let mut body = open.to_vec();
    body.resize(700 * 1024, b'x');
    body.extend_from_slice(br#""}]}"#);
    for chunk in body.chunks(64 * 1024) {
        let frame = [format!("{:x}\r\n", chunk.len()).as_bytes(), chunk, b"\r\n"].concat();
        // The gateway may answer, and stop reading, before the last chunk.
        if s.write_all(&frame).await.is_err() {
            break;
        }
    }
    let _ = s.write_all(b"0\r\n\r\n").await;
    let mut buf = vec![0u8; 4096];
    let n = tokio::time::timeout(Duration::from_secs(10), s.read(&mut buf))
        .await
        .expect("an answer")
        .unwrap();
    let text = String::from_utf8_lossy(&buf[..n]);
    assert!(
        text.starts_with("HTTP/1.1 503") && text.contains("too many large request bodies"),
        "{text}"
    );
    assert_eq!(mock.hits(), 0);
}

/// The size cap is inclusive: a body declared at exactly `MAX_REQUEST_BODY` (100 MiB) is read, not
/// refused up front the way one byte more is.
/// claim: SEC-19
#[tokio::test]
async fn a_body_declared_at_the_size_cap_is_not_refused_up_front() {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .start()
        .await;
    const CAP: usize = 100 * 1024 * 1024;
    for (len, refused) in [(CAP + 1, true), (CAP, false)] {
        let mut s = tokio::net::TcpStream::connect(("127.0.0.1", gw.port))
            .await
            .unwrap();
        let head = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nhost: gw\r\nauthorization: Bearer {}\r\n\
             content-type: application/json\r\ncontent-length: {len}\r\n\r\n{{\"model\":\"gpt-4o-mini\"",
            vkey(&sk, 35)
        );
        s.write_all(head.as_bytes()).await.unwrap();
        // Refused up front, the answer comes at once; read, the walk waits for the rest of it.
        let mut buf = vec![0u8; 4096];
        let answer = tokio::time::timeout(Duration::from_secs(2), s.read(&mut buf)).await;
        let text = match answer {
            Ok(Ok(n)) => String::from_utf8_lossy(&buf[..n]).into_owned(),
            _ => String::new(),
        };
        assert_eq!(
            text.starts_with("HTTP/1.1 413"),
            refused,
            "{len} B declared: {text:?}"
        );
    }
}

/// The size cap is inclusive on the bytes that stream through too: a body of exactly
/// `MAX_REQUEST_BODY` (100 MiB) relayed to a `/{provider}` route is answered, not aborted at its
/// last byte.
/// claim: SEC-19
#[tokio::test]
async fn a_body_of_exactly_the_size_cap_streams_through() {
    let (pubkey, _sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .start()
        .await;
    const CAP: usize = 100 * 1024 * 1024;
    let close = br#""}]}"#;
    let mut body = br#"{"model":"gpt-4o-mini","messages":[{"role":"user","content":""#.to_vec();
    body.resize(CAP - close.len(), b'x');
    body.extend_from_slice(close);
    let resp = test_client()
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header("authorization", "Bearer sk-byo-test")
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200, "{}", gw.log());
}

/// `client_write_timeout_secs = 0` disables the downstream write timeout rather than making it
/// zero: a client that is slow to start reading a large stream still gets every byte of it.
/// claim: REL-10
/// defect: D42
#[tokio::test]
async fn a_zero_client_write_timeout_disables_it() {
    let (pubkey, _sk) = test_keypair(1);
    let big = {
        let event = "data: {\"choices\":[{\"delta\":{\"content\":\"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\"}}]}\n\n";
        bytes::Bytes::from(event.repeat((32 << 20) / event.len()))
    };
    let want = big.len();
    let mock = ReplyUpstream::start(move |_, _| Reply::Full {
        status: 200,
        content_type: "text/event-stream",
        body: big.clone(),
    })
    .await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .config_line("client_write_timeout_secs = 0")
        .start()
        .await;
    let resp = test_client()
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header("authorization", "Bearer sk-byo-test")
        .header("content-type", "application/json")
        .body(STREAM_BODY)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    // Let the socket buffers fill before reading: every write after that has to wait.
    tokio::time::sleep(Duration::from_millis(500)).await;
    let got = resp.bytes().await.map(|b| b.len());
    assert_eq!(got.ok(), Some(want), "{}", gw.log());
}
