//! Claim tests: streams arrive as the provider sends them, survive a deploy signal, and a stream cut
//! short is an error for the client and an estimate on the ledger. Plus the automatic cache
//! breakpoints an OpenAI-SDK caller gets on a Claude row.
//!
//! Run via `mise run test:integration:rs`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::*;
use serde_json::Value;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst};
use std::time::Duration;

/// How long the scripted provider pauses between its first event and the rest of the stream.
const PAUSE: Duration = Duration::from_millis(1500);

const OPENAI_FIRST: &str = "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"FIRST\"}}]}\n\n";
const OPENAI_REST: &str = "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"LAST\"},\"finish_reason\":\"stop\"}]}\n\n\
data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-mini\",\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n\n\
data: [DONE]\n\n";

const CLAUDE_FIRST: &str = "event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-opus-4-8\",\"content\":[],\"usage\":{\"input_tokens\":5,\"output_tokens\":1}}}\n\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"FIRST\"}}\n\n";
const CLAUDE_REST: &str = "event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"LAST\"}}\n\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\n";

/// A provider that sends `first`, pauses, then sends `rest` and closes.
async fn pausing_provider(first: &'static str, rest: &'static str) -> ScriptedUpstream {
    ScriptedUpstream::start(move |_, _| {
        let mut head = http_head(200, "text/event-stream", None);
        head.extend_from_slice(first.as_bytes());
        vec![
            Step::Write(head),
            Step::Sleep(PAUSE),
            Step::Write(rest.as_bytes().to_vec()),
        ]
    })
    .await
}

/// A provider that sends `first`, then waits for `gate` before it sends `rest` and closes.
async fn gated_provider(
    first: &'static str,
    rest: &'static str,
    gate: Arc<AtomicBool>,
) -> ScriptedUpstream {
    ScriptedUpstream::start(move |_, _| {
        let mut head = http_head(200, "text/event-stream", None);
        head.extend_from_slice(first.as_bytes());
        let gate = gate.clone();
        vec![
            Step::Write(head),
            Step::Until(Arc::new(move || gate.load(SeqCst))),
            Step::Write(rest.as_bytes().to_vec()),
        ]
    })
    .await
}

/// Read a streamed response to its end. Runs `on_first` the moment the first event is in hand.
async fn read_stream(resp: reqwest::Response, mut on_first: impl FnMut()) -> String {
    let mut resp = resp;
    let mut text = String::new();
    let mut seen_first = false;
    while let Some(chunk) = tokio::time::timeout(CONDITION_BUDGET, resp.chunk())
        .await
        .expect("the stream must not hang")
        .unwrap()
    {
        text.push_str(&String::from_utf8_lossy(&chunk));
        if !seen_first && text.contains("FIRST") {
            seen_first = true;
            on_first();
        }
    }
    assert!(seen_first, "the first event arrived: {text}");
    text
}

async fn post_stream(gw: &Gateway, key: &str, path: &str, body: &str) -> reqwest::Response {
    test_client()
        .post(format!("{}{path}", gw.url()))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .body(body.to_owned())
        .send()
        .await
        .unwrap()
}

/// The first event reaches the client while the provider is still mid-stream, on a byte relay and
/// on a translated walk alike — the gateway never holds a stream back to the end. The provider
/// sends the rest only once the client has the first event, so the stream completing at all is
/// the proof: an ordering, which no host load can reorder, where a timing would be.
/// claim: S1
#[tokio::test]
async fn streams_reach_the_client_as_the_provider_sends_them() {
    for (first, rest, providers, model) in [
        (OPENAI_FIRST, OPENAI_REST, &["openai"][..], "gpt-4o-mini"),
        (
            CLAUDE_FIRST,
            CLAUDE_REST,
            &["anthropic", "openai", "openrouter"][..],
            "claude-opus-4-8",
        ),
    ] {
        let (pubkey, sk) = test_keypair(70);
        let gate = Arc::new(AtomicBool::new(false));
        let provider = gated_provider(first, rest, gate.clone()).await;
        let gw = Gateway::builder(unused_nats_port(), &provider.authority(), &b64(&pubkey))
            .providers(providers)
            .start()
            .await;
        let body = format!(
            r#"{{"model":"{model}","stream":true,"messages":[{{"role":"user","content":"hi"}}]}}"#
        );
        let resp = post_stream(&gw, &billing_vkey(&sk, 70), "/v1/chat/completions", &body).await;
        assert_eq!(resp.status().as_u16(), 200, "{model}");
        let text = read_stream(resp, || gate.store(true, SeqCst)).await;
        let (first, last) = (text.find("FIRST"), text.find("LAST"));
        assert!(
            first.is_some() && last.is_some() && first < last,
            "{model}: {text}"
        );
    }
}

/// A same-wire Chat Completions stream from a host other than OpenAI goes through `SseBridge`'s
/// relay (it drops the `role` OpenRouter repeats on every chunk). Each event still leaves the
/// gateway as it arrives: an OpenRouter stream of a Claude row, written one event at a time with
/// OpenRouter's keep-alive comments between them, reaches the client one event at a time: the
/// provider writes each event only once the client holds the one before it.
/// claim: S1
/// defect: D244
#[tokio::test]
async fn a_same_wire_openrouter_chat_relay_streams_each_event_as_it_arrives() {
    let chunk = |text: &str, finish: &str| {
        format!(
            "data: {{\"id\":\"gen-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\
             \"model\":\"anthropic/claude-haiku-4.5\",\"provider\":\"Amazon Bedrock\",\"choices\":\
             [{{\"index\":0,\"delta\":{{\"content\":\"{text}\",\"role\":\"assistant\"}},\
             \"finish_reason\":{finish},\"native_finish_reason\":{finish}}}]}}\n\n"
        )
    };
    let words = ["W0", "W1", "W2", "W3", "W4"];
    // How many of `words` the client holds.
    let seen = Arc::new(AtomicUsize::new(0));
    let client_has = {
        let seen = seen.clone();
        move |n: usize| {
            let seen = seen.clone();
            Step::Until(Arc::new(move || seen.load(SeqCst) >= n))
        }
    };
    let provider = ScriptedUpstream::start(move |_, _| {
        let mut head = http_head(200, "text/event-stream", None);
        head.extend_from_slice(chunk(words[0], "null").as_bytes());
        let mut steps = vec![Step::Write(head)];
        for (i, w) in words.iter().enumerate().skip(1) {
            steps.push(client_has(i));
            steps.push(Step::Write(b": OPENROUTER PROCESSING\n\n".to_vec()));
            steps.push(Step::Write(chunk(w, "null").into_bytes()));
        }
        steps.push(client_has(words.len()));
        let mut tail = chunk("", "\"stop\"");
        tail.push_str(
            "data: {\"id\":\"gen-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\
             \"model\":\"anthropic/claude-haiku-4.5\",\"choices\":[],\"usage\":\
             {\"prompt_tokens\":5,\"completion_tokens\":5,\"total_tokens\":10}}\n\ndata: [DONE]\n\n",
        );
        steps.push(Step::Write(tail.into_bytes()));
        steps
    })
    .await;
    let (pubkey, sk) = test_keypair(72);
    let gw = Gateway::builder(unused_nats_port(), &provider.authority(), &b64(&pubkey))
        .providers(&["openrouter"])
        .start()
        .await;
    let body = r#"{"model":"claude-haiku-4-5","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"hi"}]}"#;
    let mut resp = post_stream(&gw, &billing_vkey(&sk, 72), "/v1/chat/completions", body).await;
    assert_eq!(resp.status().as_u16(), 200);
    let mut text = String::new();
    // A word the provider has not written yet cannot be in hand: each one arriving is the gateway
    // having relayed the one before it on its own, not held it for the next.
    while let Some(c) = tokio::time::timeout(CONDITION_BUDGET, resp.chunk())
        .await
        .expect("the stream must not hang: an event was held back for the next")
        .unwrap()
    {
        text.push_str(&String::from_utf8_lossy(&c));
        let mut n = seen.load(SeqCst);
        while n < words.len() && text.contains(words[n]) {
            n += 1;
        }
        seen.store(n, SeqCst);
    }
    assert_eq!(seen.load(SeqCst), words.len(), "every word arrived: {text}");
    assert!(text.contains("[DONE]"), "{text}");
    assert_eq!(
        text.matches("\"role\"").count(),
        1,
        "the repeated role is dropped: {text}"
    );
}

/// A translated stream whose provider sends only keep-alives while the model thinks (Anthropic's
/// `event: ping`, here to a Chat Completions client) still hands its client a byte per ping: an
/// SSE comment, which every SSE parser ignores, so a load balancer's idle timeout does not cut a
/// stream that is alive. The answer that follows, and the row, are as without the pings. Each ping
/// is sent only once the client holds the comment for the one before, so each comment is shown to
/// leave as its ping arrives, whatever the host's load.
/// claim: S1
/// claim: REL-1
/// defect: D251
#[tokio::test]
async fn a_translated_stream_turns_provider_pings_into_keep_alive_comments() {
    const PINGS: usize = 4;
    // Keep-alive comments the client holds; `usize::MAX` until it holds the first event.
    let alive = Arc::new(AtomicUsize::new(usize::MAX));
    let client_has = {
        let alive = alive.clone();
        move |n: usize| {
            let alive = alive.clone();
            Step::Until(Arc::new(move || alive.load(SeqCst).wrapping_add(1) > n))
        }
    };
    let provider = ScriptedUpstream::start(move |_, _| {
        let mut head = http_head(200, "text/event-stream", None);
        head.extend_from_slice(CLAUDE_FIRST.as_bytes());
        let mut steps = vec![Step::Write(head)];
        for i in 0..PINGS {
            steps.push(client_has(i));
            steps.push(Step::Write(
                b"event: ping\ndata: {\"type\": \"ping\"}\n\n".to_vec(),
            ));
        }
        steps.push(client_has(PINGS));
        steps.push(Step::Write(CLAUDE_REST.as_bytes().to_vec()));
        steps
    })
    .await;
    let (pubkey, sk) = test_keypair(73);
    let gw = Gateway::builder(unused_nats_port(), &provider.authority(), &b64(&pubkey))
        .providers(&["anthropic"])
        .start()
        .await;
    let body = r#"{"model":"claude-opus-4-8","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"hi"}]}"#;
    let mut resp = post_stream(&gw, &billing_vkey(&sk, 73), "/v1/chat/completions", body).await;
    assert_eq!(resp.status().as_u16(), 200);
    let mut text = String::new();
    while let Some(c) = tokio::time::timeout(CONDITION_BUDGET, resp.chunk())
        .await
        .expect("the stream must not hang: a ping did not reach the client as a comment")
        .unwrap()
    {
        text.push_str(&String::from_utf8_lossy(&c));
        if text.contains("FIRST") {
            alive.store(text.matches(": keep-alive\n\n").count(), SeqCst);
        }
    }
    let before_last = &text[..text.find("LAST").unwrap_or(text.len())];
    let alive_before_last = before_last.matches(": keep-alive\n\n").count();
    assert_eq!(
        alive_before_last, PINGS,
        "one keep-alive comment per provider ping, before the answer: {text}"
    );
    assert!(
        !text.contains("ping"),
        "the ping itself is not relayed: {text}"
    );
    // The events, comments aside, are what a ping-free stream gives.
    let events: Vec<&str> = text
        .split("\n\n")
        .filter(|e| !e.is_empty() && !e.starts_with(':'))
        .collect();
    assert!(events.iter().any(|e| e.contains("FIRST")), "{text}");
    assert!(events.iter().any(|e| e.contains("LAST")), "{text}");
    assert_eq!(events.last(), Some(&"data: [DONE]"), "{text}");
    let row = usage_row_of(&gw).await;
    assert_eq!(row["input_tokens"].as_u64(), Some(5), "{row}");
    assert_eq!(row["output_tokens"].as_u64(), Some(2), "{row}");
    assert_ne!(row["usage_estimated"], true, "{row}");
}

/// A deploy signal lands while a stream is open. The stream finishes, whole, and is billed.
/// claim: S3
#[tokio::test]
async fn sigterm_drains_an_open_stream() {
    let (pubkey, sk) = test_keypair(71);
    let provider = pausing_provider(OPENAI_FIRST, OPENAI_REST).await;
    let gw = Gateway::builder(unused_nats_port(), &provider.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let resp = post_stream(
        &gw,
        &billing_vkey(&sk, 71),
        "/v1/chat/completions",
        r#"{"model":"gpt-4o-mini","stream":true,"messages":[{"role":"user","content":"hi"}]}"#,
    )
    .await;
    assert_eq!(resp.status().as_u16(), 200);
    let text = read_stream(resp, || gw.sigterm()).await;
    assert!(
        text.contains("LAST") && text.contains("[DONE]"),
        "the open stream must complete across SIGTERM: {text}"
    );
    let row = usage_row_of(&gw).await;
    assert_eq!(row["output_tokens"].as_u64(), Some(2), "{row}");
}

const CUT_CLAUDE: &str = "event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-opus-4-8\",\"content\":[],\"usage\":{\"input_tokens\":3,\"output_tokens\":1}}}\n\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"half an ans\"}}\n\n";

/// The provider's stream ends without its terminal event. The client gets `stream_truncated`, and
/// the ledger says its count is an estimate.
/// claim: T7
#[tokio::test]
async fn a_cut_stream_is_stream_truncated_for_the_client_and_estimated_on_the_row() {
    let (pubkey, sk) = test_keypair(72);
    let mock = MockUpstream::start(Mode::Raw(200, "text/event-stream", CUT_CLAUDE)).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic", "openai", "openrouter"])
        .start()
        .await;
    let resp = post_stream(
        &gw,
        &billing_vkey(&sk, 72),
        "/v1/chat/completions",
        r#"{"model":"claude-opus-4-8","stream":true,"messages":[{"role":"user","content":"hi"}]}"#,
    )
    .await;
    assert_eq!(resp.status().as_u16(), 200);
    let text = resp.text().await.unwrap();
    assert!(text.contains("stream_truncated"), "{text}");
    let row = usage_row_of(&gw).await;
    assert_eq!(row["usage_estimated"], true, "{row}");
    assert_eq!(row["input_tokens"].as_u64(), Some(3), "{row}");
}

const CLAUDE_CACHE_HIT: &str = r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-4-8","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":20,"output_tokens":5,"cache_read_input_tokens":4000,"cache_creation_input_tokens":0}}"#;

/// A stock OpenAI SDK's second turn on a Claude row, with no `cache_control` anywhere. The gateway
/// marks the prefix for Anthropic, and the cache hit comes back as `cached_tokens` on the client and
/// `cache_read_tokens` on the row.
/// claim: K1, B3
#[tokio::test]
async fn an_openai_sdk_turn_on_claude_gets_cache_breakpoints_and_sees_the_hit() {
    let (pubkey, sk) = test_keypair(73);
    let mock = MockUpstream::start(Mode::Raw(200, "application/json", CLAUDE_CACHE_HIT)).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic", "openai", "openrouter"])
        .start()
        .await;
    let body = r#"{"model":"claude-opus-4-8","messages":[
        {"role":"system","content":"You are a careful assistant."},
        {"role":"user","content":"first question"},
        {"role":"assistant","content":"first answer"},
        {"role":"user","content":"second question"}]}"#;
    let resp = post_stream(&gw, &billing_vkey(&sk, 73), "/v1/chat/completions", body).await;
    assert_eq!(resp.status().as_u16(), 200);
    let v: Value = resp.json().await.unwrap();

    let sent: Value = serde_json::from_slice(&mock.captured().unwrap().body).unwrap();
    let marked = sent.to_string().matches("\"cache_control\"").count();
    assert!(
        marked >= 1,
        "an unmarked request must reach Anthropic with breakpoints: {sent}"
    );
    assert_eq!(
        v["usage"]["prompt_tokens_details"]["cached_tokens"], 4000,
        "the client sees the hit: {v}"
    );
    let row = usage_row_of(&gw).await;
    assert_eq!(row["cache_read_tokens"].as_u64(), Some(4000), "{row}");
}
