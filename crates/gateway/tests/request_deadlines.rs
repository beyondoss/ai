//! End-to-end: every request ends. `request_max_secs` caps a request's whole life, and
//! `client_read_timeout_secs` caps a client that stalls while the gateway reads its body up front.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).
//!
//! Every other bound is per read or per write, so a stream that keeps moving, or a silent one an
//! HTTP/2 PING keeps alive (OpenAI's `-pro` rows have no silence bound at all, D252), held its
//! tenant slot for as long as nobody hung up; and pingora's HTTP/2 server has no body read timeout,
//! so an h2c client that stopped mid-body held one forever with nothing upstream open. What must
//! hold: each ends within its bound, gives its tenant slot back, and charges no provider for it.

// Test target: `.unwrap()`/`.expect()`/`panic!` are assertions, not production code — allow the
// panic-surface restriction lints denied workspace-wide in `[workspace.lints.clippy]`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::time::{Duration, Instant};

const CHAT: &str = r#"{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}"#;
const CHAT_STREAM: &str =
    r#"{"model":"gpt-4o-mini","stream":true,"messages":[{"role":"user","content":"hi"}]}"#;
const OK_JSON: &str = r#"{"id":"chatcmpl-1","object":"chat.completion","model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}"#;

/// An SSE head and then an event every 200 ms for a minute: a stream that never stops moving.
fn endless_stream() -> Vec<Step> {
    let mut steps = vec![Step::Write(http_head(200, "text/event-stream", None))];
    for _ in 0..300 {
        steps.push(Step::Write(
            b"data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" tok\"}}]}\n\n".to_vec(),
        ));
        steps.push(Step::Sleep(Duration::from_millis(200)));
    }
    steps
}

fn ok_reply() -> Vec<Step> {
    vec![Step::Write(http_response(
        200,
        "application/json",
        OK_JSON.as_bytes(),
    ))]
}

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

/// A stream that never stops sending is cut at `request_max_secs`, the way any stream that dies
/// mid-way is: the client sees an error, never a clean end, and the row is a cut-short stream
/// billed an estimate. Its tenant slot comes back (the next request under a ceiling of 1 is
/// admitted), and the provider is charged nothing: with a breaker threshold of 1 that next request
/// still reaches it.
/// claim: REL-1, BIL-3
/// defect: D264
#[tokio::test]
async fn a_stream_that_never_ends_is_cut_at_request_max_secs() {
    let up =
        ScriptedUpstream::start(|_, n| if n == 0 { endless_stream() } else { ok_reply() }).await;
    let (pubkey, sk) = test_keypair(64);
    let gw = Gateway::builder(unused_nats_port(), &up.authority(), &b64(&pubkey))
        .tenant_max_in_flight(1)
        .circuit_breaker_threshold(1)
        .config_line("request_max_secs = 2")
        .start()
        .await;
    let key = billing_vkey(&sk, 264);
    let started = Instant::now();
    let resp = post(&gw, "/openai/v1/chat/completions", &key, CHAT_STREAM).await;
    assert_eq!(resp.status(), 200);
    let body = tokio::time::timeout(Duration::from_secs(15), resp.bytes())
        .await
        .expect("the stream ended");
    let took = started.elapsed();
    assert!(
        body.is_err(),
        "a cut stream ended cleanly: {:?}",
        body.map(|b| String::from_utf8_lossy(&b).into_owned())
    );
    // Two seconds, plus up to one tick of the coarse clock that checks a moving stream.
    assert!(
        took >= Duration::from_millis(1500) && took < Duration::from_secs(6),
        "{took:?}"
    );
    let row = usage_row_of(&gw).await;
    assert_eq!(row["outcome"], "cut_short", "{row}");
    assert_eq!(row["usage_estimated"], true, "{row}");
    assert!(row["output_tokens"].as_u64().unwrap_or(0) > 0, "{row}");

    let resp = post(&gw, "/openai/v1/chat/completions", &key, CHAT).await;
    let status = resp.status().as_u16();
    let text = resp.text().await.unwrap_or_default();
    assert_eq!(
        status,
        200,
        "slot or breaker still held: {text}\n{}",
        gw.log()
    );
    assert_eq!(up.hits(), 2);
    let metrics = gw.metrics().await;
    assert_eq!(
        parse_metric(&metrics, "ai_rejections_total", "request_deadline"),
        1.0
    );
}

/// A provider that takes the request and never answers is a 504 at `request_max_secs` when that
/// comes before the silence bound, on an ordinary row (below `read_timeout_secs`) and on a `-pro`
/// row (which has no silence bound at all, D252). Not a provider failure: with a breaker threshold
/// of 1 the next request still reaches the provider, no pool key is cooled, and the walk does not
/// fail over. The tenant slot comes back. The prompt is billed as an estimate, as any wait the
/// gateway gives up on after delivery is (D130).
/// claim: REL-1, SEC-16, BIL-3
/// defect: D264
#[tokio::test]
async fn a_silent_provider_is_a_504_at_request_max_secs() {
    const ANSWER: &str = r#"{"id":"resp_1","object":"response","created_at":1,"status":"completed","model":"gpt-5-pro","output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"DONE","annotations":[]}]}],"usage":{"input_tokens":5,"input_tokens_details":{"cached_tokens":0},"output_tokens":2,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":7}}"#;
    for (path, body, answer) in [
        ("/openai/v1/chat/completions", CHAT, OK_JSON),
        (
            "/v1/responses",
            r#"{"model":"gpt-5-pro","input":"hi"}"#,
            ANSWER,
        ),
    ] {
        let up = ScriptedUpstream::start(move |_, n| {
            if n == 0 {
                vec![Step::Sleep(Duration::from_secs(60))]
            } else {
                vec![Step::Write(http_response(
                    200,
                    "application/json",
                    answer.as_bytes(),
                ))]
            }
        })
        .await;
        let (pubkey, sk) = test_keypair(65);
        let gw = Gateway::builder(unused_nats_port(), &up.authority(), &b64(&pubkey))
            .providers(&["openai", "openrouter"])
            .tenant_max_in_flight(1)
            .circuit_breaker_threshold(1)
            .config_line("request_max_secs = 2")
            .start()
            .await;
        let key = billing_vkey(&sk, 265);
        let started = Instant::now();
        let resp = post(&gw, path, &key, body).await;
        let took = started.elapsed();
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        assert_eq!(status, 504, "{path}: {text}\n{}", gw.log());
        assert!(text.contains("maximum duration"), "{path}: {text}");
        assert!(
            took >= Duration::from_millis(1900) && took < Duration::from_secs(5),
            "{path}: {took:?}"
        );
        let row = usage_row_of(&gw).await;
        assert_eq!(row["usage_estimated"], true, "{path}: {row}");
        assert!(
            row["input_tokens"].as_u64().unwrap_or(0) > 0,
            "{path}: {row}"
        );

        let resp = post(&gw, path, &key, body).await;
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        assert_eq!(status, 200, "{path}: {text}\n{}", gw.log());
        // Both requests went to the one upstream (it backs every provider here); a failover would
        // have sent the first to the fallback as well.
        assert_eq!(up.hits(), 2, "{path}");
        let metrics = gw.metrics().await;
        assert_eq!(
            parse_metric(&metrics, "ai_candidate_failovers_total", ""),
            0.0,
            "{path}"
        );
        assert_eq!(
            parse_metric(&metrics, "ai_rejections_total", "circuit_open"),
            0.0,
            "{path}"
        );
        for reason in ["revoked", "unfunded", "key_named_403"] {
            assert_eq!(
                parse_metric(&metrics, "ai_key_auth_failures_total", reason),
                0.0,
                "{path}: {reason}"
            );
        }
    }
}

/// `request_max_secs = 0` turns the ceiling off: a stream longer than any small ceiling ends
/// cleanly on its own.
/// claim: REL-1
/// defect: D264
#[tokio::test]
async fn request_max_secs_zero_disables_the_ceiling() {
    let up = ScriptedUpstream::start(|_, _| {
        let mut steps = vec![Step::Write(http_head(200, "text/event-stream", None))];
        for _ in 0..12 {
            steps.push(Step::Sleep(Duration::from_millis(200)));
            steps.push(Step::Write(
                b"data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" tok\"}}]}\n\n".to_vec(),
            ));
        }
        steps.push(Step::Write(
            b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":12}}\n\ndata: [DONE]\n\n".to_vec(),
        ));
        steps
    })
    .await;
    let (pubkey, sk) = test_keypair(66);
    let gw = Gateway::builder(unused_nats_port(), &up.authority(), &b64(&pubkey))
        .config_line("request_max_secs = 0")
        .start()
        .await;
    let resp = post(
        &gw,
        "/openai/v1/chat/completions",
        &billing_vkey(&sk, 266),
        CHAT_STREAM,
    )
    .await;
    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("the whole stream arrived");
    assert_eq!(text.matches(" tok").count(), 12, "{text}");
    assert!(text.contains("[DONE]"), "{text}");
}

/// A listener that accepts connections, reads whatever arrives and never answers; returns its port
/// and how many connections it has accepted.
async fn silent_upstream() -> (u16, Arc<AtomicUsize>) {
    use tokio::io::AsyncReadExt;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let accepted = Arc::new(AtomicUsize::new(0));
    let counter = accepted.clone();
    tokio::spawn(async move {
        while let Ok((mut s, _)) = listener.accept().await {
            counter.fetch_add(1, SeqCst);
            tokio::spawn(async move {
                let mut buf = [0u8; 4096];
                while matches!(s.read(&mut buf).await, Ok(n) if n > 0) {}
            });
        }
    });
    (port, accepted)
}

/// Over h2c: one stream whose headers declare `content-length` and whose single DATA frame carries
/// only `sent`, the stream left open. Returns the status and the response body.
async fn h2_stalled_upload(port: u16, headers: &[(&str, String)], sent: &str) -> (u16, String) {
    let tcp = tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .unwrap();
    let (mut client, conn) = h2::client::handshake(tcp).await.unwrap();
    tokio::spawn(conn);
    let mut req = http::Request::post("http://gw/auto/chat/completions");
    for (k, v) in headers {
        req = req.header(*k, v.as_str());
    }
    let (resp, mut send) = client.send_request(req.body(()).unwrap(), false).unwrap();
    send.send_data(bytes::Bytes::copy_from_slice(sent.as_bytes()), false)
        .unwrap();
    let resp = tokio::time::timeout(Duration::from_secs(10), resp)
        .await
        .expect("an answer")
        .unwrap();
    let status = resp.status().as_u16();
    let mut body = resp.into_body();
    let mut text = Vec::new();
    while let Some(chunk) = body.data().await {
        let Ok(chunk) = chunk else { break };
        let _ = body.flow_control().release_capacity(chunk.len());
        text.extend_from_slice(&chunk);
    }
    drop(send);
    (status, String::from_utf8_lossy(&text).into_owned())
}

/// The same over HTTP/1.1, on a raw connection that stays open. Returns the status and the answer.
async fn h1_stalled_upload(port: u16, headers: &[(&str, String)], sent: &str) -> (u16, String) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut head = "POST /auto/chat/completions HTTP/1.1\r\nhost: gw\r\n".to_owned();
    for (k, v) in headers {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str("\r\n");
    let mut s = tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .unwrap();
    s.write_all(head.as_bytes()).await.unwrap();
    s.write_all(sent.as_bytes()).await.unwrap();
    let mut buf = [0u8; 4096];
    let n = tokio::time::timeout(Duration::from_secs(10), s.read(&mut buf))
        .await
        .expect("an answer")
        .unwrap_or(0);
    let text = String::from_utf8_lossy(&buf[..n]).into_owned();
    let status = text
        .strip_prefix("HTTP/1.1 ")
        .and_then(|t| t.get(..3))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    (status, text)
}

/// A client that stalls while the gateway reads its body up front (a headerless `/auto` request
/// reads the whole body to find `model` before it picks a provider) ends its own request at
/// `client_read_timeout_secs`: a 408, no upstream connection ever made, and its tenant slot back
/// (the next request under a ceiling of 1 is admitted). Over h2c, where pingora's server has no
/// body read timeout and this used to wait forever, and over HTTP/1.1, where pingora's own timeout
/// fired but its untagged error was answered as a 504 blaming the provider.
/// claim: REL-1, SEC-16, A3
/// defect: D265
#[tokio::test]
async fn a_client_stalled_in_an_up_front_body_read_gets_a_408() {
    let payload = CHAT;
    let half = &payload[..payload.len() / 2];
    for h2 in [true, false] {
        let what = if h2 { "h2c" } else { "http/1.1" };
        let (upstream, conns) = silent_upstream().await;
        let (pubkey, sk) = test_keypair(67);
        let gw = Gateway::builder(
            unused_nats_port(),
            &format!("127.0.0.1:{upstream}"),
            &b64(&pubkey),
        )
        .tenant_max_in_flight(1)
        .config_line("client_read_timeout_secs = 1")
        .start()
        .await;
        let key = billing_vkey(&sk, 267);
        let headers = [
            ("authorization", format!("Bearer {key}")),
            ("content-type", "application/json".to_owned()),
            ("content-length", payload.len().to_string()),
        ];
        let started = Instant::now();
        let (status, text) = if h2 {
            h2_stalled_upload(gw.port, &headers, half).await
        } else {
            h1_stalled_upload(gw.port, &headers, half).await
        };
        let took = started.elapsed();
        assert_eq!(status, 408, "{what}: {text}\nlog:\n{}", gw.log());
        assert!(text.contains("request body timed out"), "{what}: {text}");
        assert!(took < Duration::from_secs(5), "{what}: {took:?}");
        assert_eq!(conns.load(SeqCst), 0, "{what}: reached the upstream");

        // The slot is back: under a ceiling of 1 the next request is admitted, which the silent
        // upstream shows by accepting its connection (a 429 would never reach it).
        let next = tokio::spawn({
            let url = format!("{}/auto/chat/completions", gw.url());
            async move {
                let _ = test_client()
                    .post(url)
                    .header("authorization", format!("Bearer {key}"))
                    .header("content-type", "application/json")
                    .body(CHAT)
                    .send()
                    .await;
            }
        });
        tokio::time::timeout(Duration::from_secs(10), async {
            while conns.load(SeqCst) == 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("{what}: the next request was not admitted\n{}", gw.log()));
        next.abort();
        let metrics = gw.metrics().await;
        assert_eq!(
            parse_metric(&metrics, "ai_rejections_total", "tenant_concurrency"),
            0.0,
            "{what}"
        );
    }
}

/// An h2c upload that trickles in up front, one byte at a time and never silent long enough for
/// `client_read_timeout_secs`, still ends at `request_max_secs`: a 408, nothing upstream.
/// claim: REL-1, SEC-16
/// defect: D264
#[tokio::test]
async fn an_up_front_body_still_trickling_at_request_max_secs_gets_a_408() {
    let (upstream, conns) = silent_upstream().await;
    let (pubkey, sk) = test_keypair(68);
    let gw = Gateway::builder(
        unused_nats_port(),
        &format!("127.0.0.1:{upstream}"),
        &b64(&pubkey),
    )
    .config_line("request_max_secs = 2")
    .start()
    .await;
    let key = billing_vkey(&sk, 268);
    let tcp = tokio::net::TcpStream::connect(("127.0.0.1", gw.port))
        .await
        .unwrap();
    let (mut client, conn) = h2::client::handshake(tcp).await.unwrap();
    tokio::spawn(conn);
    let req = http::Request::post("http://gw/auto/chat/completions")
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .header("content-length", CHAT.len().to_string())
        .body(())
        .unwrap();
    let (resp, mut send) = client.send_request(req, false).unwrap();
    let trickle = tokio::spawn(async move {
        for b in CHAT.bytes() {
            if send.send_data(bytes::Bytes::from(vec![b]), false).is_err() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(300)).await;
        }
    });
    let started = Instant::now();
    let resp = tokio::time::timeout(Duration::from_secs(10), resp)
        .await
        .expect("an answer")
        .unwrap();
    let took = started.elapsed();
    trickle.abort();
    assert_eq!(resp.status().as_u16(), 408, "{}", gw.log());
    assert!(took < Duration::from_secs(5), "{took:?}");
    assert_eq!(conns.load(SeqCst), 0, "reached the upstream");
}

/// An upload a provider route streams through (chunked BYO, so nothing reads or holds it first)
/// that is still trickling at `request_max_secs` ends there with the deadline's 504. Each chunk
/// that goes upstream re-arms its read timeout, so only the per-chunk check ends it.
/// claim: REL-1
/// defect: D264
#[tokio::test]
async fn a_streamed_upload_still_trickling_at_request_max_secs_ends() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let (upstream, conns) = silent_upstream().await;
    let (pubkey, _sk) = test_keypair(69);
    let gw = Gateway::builder(
        unused_nats_port(),
        &format!("127.0.0.1:{upstream}"),
        &b64(&pubkey),
    )
    .config_line("request_max_secs = 2")
    .start()
    .await;
    // BYO: nothing buffers the body (a managed chat body may be held for usage injection), so
    // every chunk goes upstream and re-arms its read timeout.
    let key = "sk-byo-test";
    let s = tokio::net::TcpStream::connect(("127.0.0.1", gw.port))
        .await
        .unwrap();
    let (mut rd, mut wr) = s.into_split();
    let head = format!(
        "POST /openai/v1/chat/completions HTTP/1.1\r\nhost: gw\r\nauthorization: Bearer {key}\r\n\
         content-type: application/json\r\ntransfer-encoding: chunked\r\n\r\n",
    );
    wr.write_all(head.as_bytes()).await.unwrap();
    let trickle = tokio::spawn(async move {
        for b in CHAT.bytes() {
            let chunk = format!("1\r\n{}\r\n", b as char);
            if wr.write_all(chunk.as_bytes()).await.is_err() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(300)).await;
        }
    });
    let started = Instant::now();
    let mut buf = [0u8; 4096];
    let n = tokio::time::timeout(Duration::from_secs(10), rd.read(&mut buf))
        .await
        .expect("an answer")
        .unwrap_or(0);
    let took = started.elapsed();
    trickle.abort();
    let text = String::from_utf8_lossy(&buf[..n]);
    assert!(text.starts_with("HTTP/1.1 504"), "{text}\n{}", gw.log());
    assert!(text.contains("maximum duration"), "{text}");
    assert!(took < Duration::from_secs(5), "{took:?}");
    assert_eq!(conns.load(SeqCst), 1);
}

/// A body past the replay buffer runs as a re-run that carries it (`FullBody`): the re-run keeps
/// the request's deadline rather than starting its own clock, so a silent provider is still the
/// deadline's 504.
/// claim: REL-1
/// defect: D264
#[tokio::test]
async fn a_large_body_re_run_keeps_the_requests_deadline() {
    let (upstream, conns) = silent_upstream().await;
    let (pubkey, sk) = test_keypair(70);
    let gw = Gateway::builder(
        unused_nats_port(),
        &format!("127.0.0.1:{upstream}"),
        &b64(&pubkey),
    )
    .config_line("request_max_secs = 2")
    .start()
    .await;
    let big = format!(
        r#"{{"model":"gpt-4o-mini","messages":[{{"role":"user","content":"{}"}}]}}"#,
        "x".repeat(100 * 1024)
    );
    let started = Instant::now();
    let resp = post(&gw, "/auto/chat/completions", &billing_vkey(&sk, 270), &big).await;
    let took = started.elapsed();
    let status = resp.status().as_u16();
    let text = resp.text().await.unwrap_or_default();
    assert_eq!(status, 504, "{text}\n{}", gw.log());
    assert!(text.contains("maximum duration"), "{text}");
    assert!(took < Duration::from_secs(5), "{took:?}");
    assert_eq!(conns.load(SeqCst), 1);
}

/// A listener whose accept queue is full: a connect to it hangs (the kernel drops the SYN) until
/// the connector's own timeout. Returns its authority and the connections that filled it.
async fn unaccepting_upstream() -> (String, Vec<tokio::net::TcpStream>) {
    let sock = tokio::net::TcpSocket::new_v4().unwrap();
    sock.bind("127.0.0.1:0".parse().unwrap()).unwrap();
    let addr = sock.local_addr().unwrap();
    let listener = sock.listen(0).unwrap();
    let mut held = Vec::new();
    for _ in 0..4 {
        if let Ok(Ok(s)) = tokio::time::timeout(
            Duration::from_millis(200),
            tokio::net::TcpStream::connect(addr),
        )
        .await
        {
            held.push(s);
        }
    }
    // Leaked so the listener stays open for the test without ever accepting.
    std::mem::forget(listener);
    (addr.to_string(), held)
}

/// A walk that reaches its next candidate after the deadline passed (here the first candidate's
/// connect hung past it) ends there with the deadline's 504, without opening a connection to that
/// candidate that the provider would bill.
/// claim: REL-1, BIL-3
/// defect: D264
#[tokio::test]
async fn a_candidate_reached_after_the_deadline_is_not_contacted() {
    let (primary, _held) = unaccepting_upstream().await;
    let (fallback, fallback_conns) = silent_upstream().await;
    let (pubkey, sk) = test_keypair(71);
    let gw = Gateway::builder(unused_nats_port(), &primary, &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &format!("127.0.0.1:{fallback}"))
        .config_line("request_max_secs = 2")
        .config_line("connect_timeout_secs = 3")
        .start()
        .await;
    let resp = test_client()
        .post(format!("{}/auto/chat/completions", gw.url()))
        .header(
            "authorization",
            format!("Bearer {}", billing_vkey(&sk, 271)),
        )
        .header("content-type", "application/json")
        .header("x-beyond-model", "gpt-4o-mini")
        .header("x-beyond-order", "openai,openrouter")
        .body(CHAT)
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    let text = resp.text().await.unwrap_or_default();
    assert_eq!(status, 504, "{text}\n{}", gw.log());
    assert!(text.contains("maximum duration"), "{text}");
    assert_eq!(fallback_conns.load(SeqCst), 0, "{}", gw.log());
}

/// `client_read_timeout_secs = 0` turns the up-front read bound off, on both transports: a client
/// that pauses mid-body longer than any small bound still has its request read and sent on (the
/// silent upstream accepts its connection) rather than answered 408.
/// claim: REL-1
/// defect: D265
#[tokio::test]
async fn client_read_timeout_secs_zero_disables_the_up_front_bound() {
    use tokio::io::AsyncWriteExt;
    let payload = CHAT;
    let (first, rest) = payload.split_at(payload.len() / 2);
    for h2 in [true, false] {
        let what = if h2 { "h2c" } else { "http/1.1" };
        let (upstream, conns) = silent_upstream().await;
        let (pubkey, sk) = test_keypair(68);
        let gw = Gateway::builder(
            unused_nats_port(),
            &format!("127.0.0.1:{upstream}"),
            &b64(&pubkey),
        )
        .config_line("client_read_timeout_secs = 0")
        .start()
        .await;
        let key = billing_vkey(&sk, 268);
        let headers = [
            ("authorization", format!("Bearer {key}")),
            ("content-type", "application/json".to_owned()),
            ("content-length", payload.len().to_string()),
        ];
        let tcp = tokio::net::TcpStream::connect(("127.0.0.1", gw.port))
            .await
            .unwrap();
        // Held so the connection stays open while the upstream is waited on.
        let mut _keep: Vec<Box<dyn std::any::Any + Send>> = Vec::new();
        if h2 {
            let (mut client, conn) = h2::client::handshake(tcp).await.unwrap();
            tokio::spawn(conn);
            let mut req = http::Request::post("http://gw/auto/chat/completions");
            for (k, v) in &headers {
                req = req.header(*k, v.as_str());
            }
            let (resp, mut send) = client.send_request(req.body(()).unwrap(), false).unwrap();
            send.send_data(bytes::Bytes::copy_from_slice(first.as_bytes()), false)
                .unwrap();
            tokio::time::sleep(Duration::from_millis(1500)).await;
            send.send_data(bytes::Bytes::copy_from_slice(rest.as_bytes()), true)
                .unwrap();
            _keep.push(Box::new((client, resp, send)));
        } else {
            let mut s = tcp;
            let mut head = "POST /auto/chat/completions HTTP/1.1\r\nhost: gw\r\n".to_owned();
            for (k, v) in &headers {
                head.push_str(&format!("{k}: {v}\r\n"));
            }
            head.push_str("\r\n");
            s.write_all(head.as_bytes()).await.unwrap();
            s.write_all(first.as_bytes()).await.unwrap();
            tokio::time::sleep(Duration::from_millis(1500)).await;
            s.write_all(rest.as_bytes()).await.unwrap();
            _keep.push(Box::new(s));
        }
        tokio::time::timeout(Duration::from_secs(10), async {
            while conns.load(SeqCst) == 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap_or_else(|_| {
            panic!(
                "{what}: the paused upload never reached the upstream\n{}",
                gw.log()
            )
        });
    }
}
