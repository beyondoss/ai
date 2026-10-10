//! Drain-on-cancel: a managed stream the client hangs up on, from a provider that keeps generating
//! (and billing) after a disconnect, is read to its end so its row carries the vendor's own final
//! usage. See `crate::drain` and ARCHITECTURE.md, "Streams cut short".
//!
//! Every reply is scripted so the upstream's next bytes follow the client's hang-up as an
//! ordering, not a delay: the drain is proved by usage the client could never have seen.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Output tokens the scripted vendor reports in its final usage, which only a drained row sees.
const FULL_OUTPUT: u64 = 603;
const INPUT: u64 = 25;

/// A one-shot gate a script waits on with [`Step::Until`].
#[derive(Clone, Default)]
struct Gate(Arc<AtomicBool>);

impl Gate {
    fn open(&self) {
        self.0.store(true, Ordering::SeqCst);
    }
    fn step(&self) -> Step {
        let g = Arc::clone(&self.0);
        Step::Until(Arc::new(move || g.load(Ordering::SeqCst)))
    }
}

fn sse_head() -> Vec<u8> {
    http_head(200, "text/event-stream", None)
}

/// An Anthropic Messages stream: `message_start` and some deltas, then (after `gone`) one delta
/// that starts the drain, then (after `finish`) the final `message_delta` with the full usage.
fn anthropic_script(gone: &Gate, finish: &Gate) -> Vec<Step> {
    let mut start = format!(
        "event: message_start\ndata: {{\"type\":\"message_start\",\"message\":{{\"id\":\"msg_drain\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-haiku-4-5-20251001\",\"content\":[],\"usage\":{{\"input_tokens\":{INPUT},\"output_tokens\":1}}}}}}\n\n\
         event: content_block_start\ndata: {{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{{\"type\":\"text\",\"text\":\"\"}}}}\n\n"
    );
    let delta = "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" 1\\n\"}}\n\n";
    for _ in 0..5 {
        start.push_str(delta);
    }
    let end = format!(
        "event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":0}}\n\n\
         event: message_delta\ndata: {{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":\"end_turn\"}},\"usage\":{{\"output_tokens\":{FULL_OUTPUT}}}}}\n\n\
         event: message_stop\ndata: {{\"type\":\"message_stop\"}}\n\n"
    );
    vec![
        Step::Write(sse_head()),
        Step::Write(start.into_bytes()),
        gone.step(),
        Step::Write(delta.as_bytes().to_vec()),
        finish.step(),
        Step::Write(end.into_bytes()),
    ]
}

/// An OpenAI-wire stream as OpenRouter sends it: content chunks, then (after `gone`) one more,
/// then (after `finish`) the usage chunk carrying `usage.cost`, and `[DONE]`.
fn openrouter_script(gone: &Gate, finish: &Gate) -> Vec<Step> {
    let chunk = "data: {\"id\":\"gen-drain\",\"object\":\"chat.completion.chunk\",\"model\":\"anthropic/claude-haiku-4.5\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" 1\\n\"}}]}\n\n";
    let end = format!(
        "data: {{\"id\":\"gen-drain\",\"object\":\"chat.completion.chunk\",\"model\":\"anthropic/claude-haiku-4.5\",\"choices\":[],\"usage\":{{\"prompt_tokens\":{INPUT},\"completion_tokens\":{FULL_OUTPUT},\"total_tokens\":{},\"cost\":0.003040}}}}\n\ndata: [DONE]\n\n",
        INPUT + FULL_OUTPUT
    );
    vec![
        Step::Write(sse_head()),
        Step::Write(chunk.repeat(5).into_bytes()),
        gone.step(),
        Step::Write(chunk.as_bytes().to_vec()),
        finish.step(),
        Step::Write(end.into_bytes()),
    ]
}

/// Send `body` to `path`, read the first chunk, and hang up. The gate opens once the hang-up has
/// had time to reach the gateway (its proxy loop polls the client every turn).
async fn read_one_chunk_then_hang_up(
    gw: &Gateway,
    path: &str,
    headers: &[(&str, String)],
    body: &str,
    gone: &Gate,
) {
    let mut req = test_client()
        .post(format!("{}{path}", gw.url()))
        .header("content-type", "application/json");
    for (k, v) in headers {
        req = req.header(*k, v);
    }
    let mut resp = req.body(body.to_owned()).send().await.unwrap();
    assert_eq!(resp.status(), 200, "log:\n{}", gw.log());
    let first = tokio::time::timeout(Duration::from_secs(10), resp.chunk())
        .await
        .unwrap()
        .unwrap();
    assert!(first.is_some());
    drop(resp);
    tokio::time::sleep(Duration::from_millis(300)).await;
    gone.open();
}

fn messages_headers(sk: &ed25519_dalek::SigningKey, tenant: u64) -> Vec<(&'static str, String)> {
    vec![
        ("x-api-key", billing_vkey(sk, tenant)),
        ("anthropic-version", "2023-06-01".to_owned()),
    ]
}

const MESSAGES_BODY: &str = r#"{"model":"claude-haiku-4-5","max_tokens":1500,"stream":true,"messages":[{"role":"user","content":"Count from 1 to 300."}]}"#;

/// Wait until `ai_requests_in_flight` is back to 0: the request, and any drain, has ended.
async fn wait_idle(gw: &Gateway, within: Duration) -> bool {
    let end = tokio::time::Instant::now() + within;
    while tokio::time::Instant::now() < end {
        if parse_metric(&gw.metrics().await, "ai_requests_in_flight", "") == 0.0 {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    false
}

/// Bedrock keeps generating after a disconnect: a stream cut after its first chunk is read to its
/// end, and the row carries the vendor's final usage, exact and priced, not an estimate of what
/// was relayed. The drain is counted while it runs and when it settles.
/// claim: BIL-26
#[tokio::test]
async fn a_cancelled_bedrock_stream_is_drained_to_the_vendors_final_usage() {
    let (pubkey, sk) = test_keypair(181);
    let (gone, finish) = (Gate::default(), Gate::default());
    let (g, f) = (gone.clone(), finish.clone());
    let mock = ScriptedUpstream::start(move |_, _| anthropic_script(&g, &f)).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["bedrock"])
        .start()
        .await;
    read_one_chunk_then_hang_up(
        &gw,
        "/v1/messages",
        &messages_headers(&sk, 181),
        MESSAGES_BODY,
        &gone,
    )
    .await;
    // The delta after the hang-up started the drain; it is in flight until the vendor finishes.
    wait_for_metric(&gw, "ai_usage_drains_in_flight", "", 1.0).await;
    assert!(
        usage_rows_of(&gw).is_empty(),
        "no row until the drain ends: {:?}",
        usage_rows_of(&gw)
    );
    finish.open();

    let row = usage_row_of(&gw).await;
    assert_eq!(row["provider"], "bedrock", "{row}");
    assert_eq!(row["outcome"], "client_cancelled", "{row}");
    assert_eq!(row["output_tokens"].as_u64(), Some(FULL_OUTPUT), "{row}");
    assert_eq!(row["input_tokens"].as_u64(), Some(INPUT), "{row}");
    assert_eq!(row["usage_estimated"], false, "{row}");
    assert!(row.get("usage_estimated_parts").is_none(), "{row}");
    assert_eq!(row["usage_settled"], "drained", "{row}");
    assert_eq!(row["upstream_may_continue"], false, "{row}");
    assert_eq!(row["price_status"], "priced", "{row}");
    wait_for_metric(&gw, "ai_usage_drains_total", "result=\"settled\"", 1.0).await;
    let m = gw.metrics().await;
    assert_eq!(parse_metric(&m, "ai_usage_drains_in_flight", ""), 0.0);
    assert_eq!(
        parse_metric(&m, "ai_usage_drains_total", "result=\"deadline\""),
        0.0
    );
    assert_eq!(
        parse_metric(&m, "ai_usage_drains_total", "result=\"error\""),
        0.0
    );
}

/// The client got part of a drained answer: it is never stored in the response cache (the same
/// request again goes upstream), and its capture says it is not complete.
/// claim: BIL-26
#[tokio::test]
async fn a_drained_answer_is_not_cached_and_its_capture_is_incomplete() {
    let (pubkey, sk) = test_keypair(186);
    let (gone, finish) = (Gate::default(), Gate::default());
    let (g, f) = (gone.clone(), finish.clone());
    let open = Gate::default();
    open.open();
    let mock = ScriptedUpstream::start(move |_, n| {
        if n == 0 {
            anthropic_script(&g, &f)
        } else {
            anthropic_script(&open, &open)
        }
    })
    .await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["bedrock"])
        .cache_ttl_secs(60)
        .start()
        .await;
    let mut headers = messages_headers(&sk, 186);
    headers.push(("x-beyond-capture", "on".to_owned()));
    read_one_chunk_then_hang_up(&gw, "/v1/messages", &headers, MESSAGES_BODY, &gone).await;
    finish.open();
    let row = usage_row_of(&gw).await;
    assert_eq!(row["usage_settled"], "drained", "{row}");
    let log = gw.log();
    let payload: serde_json::Value = log
        .lines()
        .find(|l| l.contains(r#""target":"ai.payload""#))
        .and_then(|l| serde_json::from_str(l).ok())
        .unwrap_or_else(|| panic!("no ai.payload line in:\n{log}"));
    assert_eq!(payload["fields"]["complete"], false, "{payload}");

    // The same request again: not a cache hit, the upstream answers it.
    let mut req = test_client()
        .post(format!("{}/v1/messages", gw.url()))
        .header("content-type", "application/json");
    for (k, v) in messages_headers(&sk, 186) {
        req = req.header(k, v);
    }
    let resp = req.body(MESSAGES_BODY).send().await.unwrap();
    assert_eq!(resp.status(), 200);
    let _ = resp.bytes().await;
    assert_eq!(mock.hits(), 2, "a drained answer was served from the cache");
}

/// OpenRouter on its provider route: the drained row carries the vendor's `usage.cost`, so it is
/// priced from the reported cost, not the dearest endpoint.
/// claim: BIL-26
#[tokio::test]
async fn a_cancelled_openrouter_stream_is_drained_to_its_reported_cost() {
    let (pubkey, sk) = test_keypair(182);
    let (gone, finish) = (Gate::default(), Gate::default());
    let (g, f) = (gone.clone(), finish.clone());
    let mock = ScriptedUpstream::start(move |_, _| openrouter_script(&g, &f)).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openrouter"])
        .start()
        .await;
    read_one_chunk_then_hang_up(
        &gw,
        "/openrouter/api/v1/chat/completions",
        &[("authorization", format!("Bearer {}", billing_vkey(&sk, 182)))],
        r#"{"model":"anthropic/claude-haiku-4.5","stream":true,"messages":[{"role":"user","content":"Count from 1 to 300."}]}"#,
        &gone,
    )
    .await;
    finish.open();
    let row = usage_row_of(&gw).await;
    assert_eq!(row["outcome"], "client_cancelled", "{row}");
    assert_eq!(row["output_tokens"].as_u64(), Some(FULL_OUTPUT), "{row}");
    assert_eq!(row["usage_estimated"], false, "{row}");
    assert_eq!(row["usage_settled"], "drained", "{row}");
    assert_eq!(row["upstream_may_continue"], false, "{row}");
    assert_eq!(row["upstream_cost_usd"], "0.00304", "{row}");
    assert_eq!(row["cost_basis"], "reported", "{row}");
    assert_eq!(row["price_status"], "priced", "{row}");
}

/// The drain is bounded by the request's deadline: a vendor still streaming at `request_max_secs`
/// is cut there, and the row is the estimate it always was, flagged.
/// claim: BIL-26
#[tokio::test]
async fn a_drain_that_reaches_the_deadline_is_estimated_and_flagged() {
    let (pubkey, sk) = test_keypair(183);
    let gone = Gate::default();
    let g = gone.clone();
    // After the hang-up: one delta (the drain starts), then silence past the deadline.
    let mock = ScriptedUpstream::start(move |_, _| {
        let mut s = anthropic_script(&g, &Gate::default());
        s.truncate(4);
        s.push(Step::Sleep(Duration::from_secs(30)));
        s
    })
    .await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["bedrock"])
        .config_line("request_max_secs = 4")
        .start()
        .await;
    read_one_chunk_then_hang_up(
        &gw,
        "/v1/messages",
        &messages_headers(&sk, 183),
        MESSAGES_BODY,
        &gone,
    )
    .await;
    let rows = wait_usage_rows(&gw, 1, 20).await;
    let row = rows
        .first()
        .unwrap_or_else(|| panic!("no ai.usage row; log:\n{}", gw.log()));
    assert_eq!(row["outcome"], "client_cancelled", "{row}");
    assert_eq!(row["usage_estimated"], true, "{row}");
    assert!(row.get("usage_settled").is_none(), "{row}");
    assert_eq!(row["upstream_may_continue"], true, "{row}");
    assert_eq!(row["price_status"], "estimated", "{row}");
    assert!(
        row["output_tokens"].as_u64().unwrap_or(0) < FULL_OUTPUT,
        "{row}"
    );
    wait_for_metric(&gw, "ai_usage_drains_total", "result=\"deadline\"", 1.0).await;
    assert_eq!(
        parse_metric(&gw.metrics().await, "ai_usage_drains_in_flight", ""),
        0.0
    );
}

/// Anthropic stops generating when the client leaves and bills what it generated: its cancelled
/// stream ends at once, as before, with no drain.
/// claim: BIL-26
#[tokio::test]
async fn a_provider_that_stops_on_cancel_is_not_drained() {
    let (pubkey, sk) = test_keypair(184);
    let (gone, finish) = (Gate::default(), Gate::default());
    let (g, f) = (gone.clone(), finish.clone());
    let mock = ScriptedUpstream::start(move |_, _| anthropic_script(&g, &f)).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic"])
        .start()
        .await;
    read_one_chunk_then_hang_up(
        &gw,
        "/anthropic/v1/messages",
        &messages_headers(&sk, 184),
        MESSAGES_BODY,
        &gone,
    )
    .await;
    // The vendor is never let finish: a drain would wait forever and write no row.
    let rows = wait_usage_rows(&gw, 1, 10).await;
    let row = rows
        .first()
        .unwrap_or_else(|| panic!("no ai.usage row; log:\n{}", gw.log()));
    assert_eq!(row["outcome"], "client_cancelled", "{row}");
    assert_eq!(row["usage_estimated"], true, "{row}");
    assert!(row.get("usage_settled").is_none(), "{row}");
    assert_eq!(row["upstream_may_continue"], false, "{row}");
    let m = gw.metrics().await;
    assert_eq!(parse_metric(&m, "ai_usage_drains_in_flight", ""), 0.0);
    assert_eq!(
        parse_metric(&m, "ai_usage_drains_total", "result=\"settled\""),
        0.0
    );
    drop(finish);
}

/// A BYO request bills nothing through the gateway (no row), so its cancel is never drained: the
/// request ends when the client leaves, though the provider would keep generating.
/// claim: BIL-26
#[tokio::test]
async fn a_byo_stream_is_not_drained() {
    let (pubkey, _sk) = test_keypair(185);
    let (gone, finish) = (Gate::default(), Gate::default());
    let (g, f) = (gone.clone(), finish.clone());
    let mock = ScriptedUpstream::start(move |_, _| openrouter_script(&g, &f)).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openrouter"])
        .start()
        .await;
    read_one_chunk_then_hang_up(
        &gw,
        "/openrouter/api/v1/chat/completions",
        &[("authorization", "Bearer sk-or-v1-byo-key".to_owned())],
        r#"{"model":"anthropic/claude-haiku-4.5","stream":true,"messages":[{"role":"user","content":"hi"}]}"#,
        &gone,
    )
    .await;
    // Never let the vendor finish: a drained request would stay in flight.
    assert!(
        wait_idle(&gw, Duration::from_secs(10)).await,
        "a BYO cancel is still in flight; log:\n{}",
        gw.log()
    );
    let m = gw.metrics().await;
    assert_eq!(parse_metric(&m, "ai_usage_drains_in_flight", ""), 0.0);
    assert!(usage_rows_of(&gw).is_empty());
    drop(finish);
}
