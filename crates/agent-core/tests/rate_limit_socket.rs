//! Rate-limit retry end to end: an `Agent` over a real `GatewayClient` over a TCP socket, where the
//! server answers with rate limits — as an HTTP 429 (the gateway path, Anthropic's body) and as an
//! error event inside a 200 stream (OpenAI Responses, the shape that killed an eval run) — before it
//! finally succeeds. Waits are kept short with small `retry-after-ms` hints and a 1ms backoff base, so
//! the hint is what the elapsed time measures.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::io::{Read, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use agent_core::rate_limit::RateLimitRetry;
use agent_core::{Agent, AgentEvent, GatewayClient, Session};

/// Serve one canned HTTP response per connection, in order; returns the base URL and a connection
/// counter.
fn serve(responses: Vec<String>) -> (String, Arc<AtomicUsize>) {
    let listener = beyond_ai_test_support::ports::listener();
    let addr = listener.local_addr().expect("addr");
    let count = Arc::new(AtomicUsize::new(0));
    let seen = count.clone();
    thread::spawn(move || {
        for resp in responses {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            seen.fetch_add(1, Ordering::SeqCst);
            let mut buf = [0u8; 65536];
            let _ = stream.read(&mut buf);
            let _ = stream.write_all(resp.as_bytes());
            let _ = stream.flush();
        }
    });
    (format!("http://{addr}"), count)
}

fn sse(body: &str) -> String {
    format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{body}")
}

const ANTHROPIC_OK: &str = "event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":7,\"output_tokens\":1}}}\n\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"made it\"}}\n\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}\n\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\n";

const OPENAI_OK: &str = "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_1\"}}\n\n\
data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"made it\"}\n\n\
data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_1\",\"content\":[{\"type\":\"output_text\",\"text\":\"made it\"}]}}\n\n\
data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n";

/// The in-stream TPM error from the eval run, its hint shortened to 40ms.
const OPENAI_RATE_LIMITED: &str = "event: error\n\
data: {\"type\":\"error\",\"error\":{\"type\":\"tokens\",\"code\":\"rate_limit_exceeded\",\"headers\":{\"retry-after\":\"1\",\"retry-after-ms\":\"40\",\"x-ratelimit-reset-tokens\":\"59.032s\"},\"message\":\"Rate limit reached for gpt-5 on tokens per min (TPM): Limit 10000000, Used 9838806, Requested 201732. Please try again in 40ms.\",\"param\":null},\"sequence_number\":2}\n\n";

/// Fast backoff, roomy budget: the server's hint is what each wait measures.
fn fast() -> RateLimitRetry {
    RateLimitRetry {
        budget: Duration::from_secs(30),
        base_backoff: Duration::from_millis(1),
        max_wait: Duration::from_secs(5),
    }
}

/// Run one prompt, collecting the `Retry` events and the final text.
async fn run(agent: &Agent) -> (agent_core::Result<()>, Vec<(u32, u64, bool)>, Session) {
    let mut session = Session::new();
    session.user("hi");
    let mut retries = Vec::new();
    let result = agent
        .run_events(&mut session, |ev| {
            if let AgentEvent::Retry {
                attempt,
                delay_ms,
                rate_limited,
                ..
            } = ev
            {
                retries.push((attempt, delay_ms, rate_limited));
            }
        })
        .await;
    (result, retries, session)
}

fn last_text(session: &Session) -> String {
    format!("{:?}", session.messages.last().expect("a reply").content)
}

/// Five 429s — more than the 3 retries every transient error used to get — then success. The client's
/// own pre-first-byte retry is off so each 429 reaches the turn's patient retry, which reads the
/// `retry-after-ms` header back out of the client's error.
#[tokio::test]
async fn http_429s_with_retry_after_are_waited_out_until_the_run_completes() {
    let body = r#"{"type":"error","error":{"type":"rate_limit_error","message":"This request would exceed the rate limit for your organization"}}"#;
    let limited = format!(
        "HTTP/1.1 429 Too Many Requests\r\nretry-after-ms: 40\r\nretry-after: 1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let mut responses = vec![limited; 5];
    responses.push(sse(ANTHROPIC_OK));
    let (base, connections) = serve(responses);
    let client = GatewayClient::new(base, "bai_v1.test")
        .unwrap()
        .with_retry(0, Duration::from_millis(1));
    let agent = Agent::new(Arc::new(client), "claude-opus-4-8").with_rate_limit_retry(fast());

    let started = Instant::now();
    let (result, retries, session) = run(&agent).await;
    result.expect("the run completes once the rate limit clears");
    assert_eq!(connections.load(Ordering::SeqCst), 6);
    assert!(last_text(&session).contains("made it"));
    assert_eq!(retries.len(), 5, "{retries:?}");
    for (i, (attempt, delay_ms, rate_limited)) in retries.iter().enumerate() {
        assert_eq!(*attempt as usize, i + 1);
        assert!(*rate_limited);
        assert!(*delay_ms >= 40, "the 40ms hint is a floor: {retries:?}");
    }
    assert!(started.elapsed() >= Duration::from_millis(200));
}

/// The eval's failure: OpenAI Responses answers 200 and then an `error` event carrying the rate-limit
/// headers in its body. Five of those, then success.
#[tokio::test]
async fn openai_in_stream_rate_limits_are_waited_out_honouring_the_hint_in_the_body() {
    let mut responses = vec![sse(OPENAI_RATE_LIMITED); 5];
    responses.push(sse(OPENAI_OK));
    let (base, connections) = serve(responses);
    let client = GatewayClient::new(base, "bai_v1.test").unwrap();
    let agent = Agent::new(Arc::new(client), "gpt-5").with_rate_limit_retry(fast());

    let started = Instant::now();
    let (result, retries, session) = run(&agent).await;
    result.expect("the run completes once the rate limit clears");
    assert_eq!(connections.load(Ordering::SeqCst), 6);
    assert!(last_text(&session).contains("made it"));
    assert_eq!(retries.len(), 5, "{retries:?}");
    assert!(
        retries.iter().all(|&(_, d, r)| r && d >= 40),
        "the body's retry-after-ms (not the 59s reset) sets each wait: {retries:?}"
    );
    assert!(started.elapsed() >= Duration::from_millis(200));
    // Only the successful attempt's reply is in the transcript.
    let assistants = session
        .messages
        .iter()
        .filter(|m| m.role == agent_core::message::Role::Assistant)
        .count();
    assert_eq!(assistants, 1);
}
