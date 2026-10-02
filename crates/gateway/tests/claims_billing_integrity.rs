//! Billing claims whose subject is the claim itself, not one defect: what a row carries and when
//! one is written at all. Each test asserts the CORRECT behavior; one that reproduces an open
//! defect is `#[ignore]`d and named in `verify/defects.toml`.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::collections::HashSet;
use std::time::Duration;

use common::*;

const CHAT: &str = r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#;

struct Sent {
    status: u16,
    request_id: Option<String>,
    text: String,
}

async fn send(url: String, auth: (&str, String), body: &str, extra: &[(&str, &str)]) -> Sent {
    let mut req = test_client()
        .post(url)
        .header(auth.0, auth.1)
        .header("content-type", "application/json");
    for (k, v) in extra {
        req = req.header(*k, *v);
    }
    let resp = req.body(body.to_owned()).send().await.unwrap();
    let status = resp.status().as_u16();
    let request_id = resp
        .headers()
        .get("x-beyond-request-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let text = resp.text().await.unwrap_or_default();
    Sent {
        status,
        request_id,
        text,
    }
}

fn bearer(key: &str) -> (&'static str, String) {
    ("authorization", format!("Bearer {key}"))
}

fn rows_text(rows: &[serde_json::Value]) -> String {
    rows.iter()
        .map(|r| r.to_string())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Every served managed request writes exactly one row, and no two rows share a `request_id` — so
/// a consumer can dedupe on it without ever merging two billed calls. Concurrency and a mix of
/// stream and non-stream calls are the conditions under which a shared counter or a lost write
/// would show.
/// claim: BIL-19, BIL-4
#[tokio::test]
async fn every_served_request_writes_one_row_with_a_unique_request_id() {
    let (pubkey, sk) = test_keypair(101);
    let json = MockUpstream::start(Mode::Json).await;
    let sse = MockUpstream::start(Mode::Sse).await;
    let gw = Gateway::builder(unused_nats_port(), &json.authority(), &b64(&pubkey))
        .providers(&["openai", "fireworks"])
        .provider_authority("fireworks", &sse.authority())
        .start()
        .await;
    let key = billing_vkey(&sk, 101);
    const N: usize = 24;
    let mut tasks = Vec::new();
    for i in 0..N {
        let (url, key) = (gw.url(), key.clone());
        tasks.push(tokio::spawn(async move {
            if i % 2 == 0 {
                send(format!("{url}/openai/v1/chat/completions"), bearer(&key), CHAT, &[]).await
            } else {
                send(
                    format!("{url}/fireworks/inference/v1/chat/completions"),
                    bearer(&key),
                    r#"{"model":"accounts/fireworks/models/x","stream":true,"messages":[{"role":"user","content":"hi"}]}"#,
                    &[],
                )
                .await
            }
        }));
    }
    let mut header_ids = HashSet::new();
    for t in tasks {
        let sent = t.await.unwrap();
        assert_eq!(sent.status, 200, "{}", sent.text);
        assert!(
            header_ids.insert(sent.request_id.expect("x-beyond-request-id")),
            "two responses carried the same request id"
        );
    }
    let rows = wait_usage_rows(&gw, N, 10).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let rows = if rows.len() == N {
        usage_rows_of(&gw)
    } else {
        rows
    };
    assert_eq!(
        rows.len(),
        N,
        "one row per served call:\n{}",
        rows_text(&rows)
    );
    let row_ids: HashSet<String> = rows
        .iter()
        .map(|r| r["request_id"].as_str().unwrap_or_default().to_owned())
        .collect();
    assert_eq!(row_ids.len(), N, "duplicate request_id across rows");
    assert_eq!(
        row_ids, header_ids,
        "each row's request_id is the one its client was given"
    );
    assert!(
        rows.iter()
            .all(|r| r["input_tokens"].as_u64().unwrap_or(0) > 0),
        "every row metered: {}",
        rows_text(&rows)
    );
}

/// A dedupe key must stay unique across a restart: the second process must not re-issue the
/// first one's ids, or a deploy would merge two billed calls downstream.
/// claim: BIL-19
#[tokio::test]
async fn request_ids_do_not_repeat_across_a_restart() {
    let (pubkey, sk) = test_keypair(102);
    let mock = MockUpstream::start(Mode::Json).await;
    let key = billing_vkey(&sk, 102);
    let mut ids = Vec::new();
    for _ in 0..2 {
        let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
            .providers(&["openai"])
            .start()
            .await;
        for _ in 0..3 {
            let sent = send(
                format!("{}/openai/v1/chat/completions", gw.url()),
                bearer(&key),
                CHAT,
                &[],
            )
            .await;
            assert_eq!(sent.status, 200);
        }
        let rows = wait_usage_rows(&gw, 3, 5).await;
        ids.extend(
            rows.iter()
                .map(|r| r["request_id"].as_str().unwrap().to_owned()),
        );
    }
    let unique: HashSet<&String> = ids.iter().collect();
    assert_eq!(ids.len(), 6, "{ids:?}");
    assert_eq!(unique.len(), 6, "an id repeated across processes: {ids:?}");
}

/// A catalog walk whose primary 5xxes and whose fallback serves is one billed call: one row,
/// under the id the client was given, naming the candidate that served. A managed 429 key walk is
/// the same. Retries are invisible to the bill only because they cost nothing; the attempt that
/// generated is the one billed.
/// claim: BIL-19, BIL-14
#[tokio::test]
async fn a_failover_and_a_key_walk_each_write_one_row_for_the_serving_attempt() {
    let (pubkey, sk) = test_keypair(103);
    let primary = MockUpstream::start(Mode::Status(500)).await;
    let fallback = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &primary.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .start()
        .await;
    let key = billing_vkey(&sk, 103);
    let sent = send(
        format!("{}/v1/chat/completions", gw.url()),
        bearer(&key),
        r#"{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}"#,
        &[],
    )
    .await;
    assert_eq!(sent.status, 200, "{}", sent.text);
    assert_eq!((primary.hits(), fallback.hits()), (1, 1));
    let _ = wait_usage_rows(&gw, 1, 5).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let rows = usage_rows_of(&gw);
    assert_eq!(rows.len(), 1, "{}", rows_text(&rows));
    assert_eq!(rows[0]["provider"], "openrouter", "{}", rows[0]);
    assert_eq!(
        rows[0]["request_id"].as_str(),
        sent.request_id.as_deref(),
        "{}",
        rows[0]
    );
    assert_eq!(rows[0]["input_tokens"].as_u64(), Some(11), "{}", rows[0]);

    // A 429 on the first pool key walks to the second, on the same provider.
    let throttled = MockUpstream::start(Mode::ThrottleKey("sk-first")).await;
    let gw = Gateway::builder(unused_nats_port(), &throttled.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .pool_keys("openai", &["sk-first", "sk-second"])
        .start()
        .await;
    let sent = send(
        format!("{}/openai/v1/chat/completions", gw.url()),
        bearer(&key),
        CHAT,
        &[],
    )
    .await;
    assert_eq!(sent.status, 200, "{}", sent.text);
    assert_eq!(throttled.hits(), 2, "the walk reached the second key");
    let _ = wait_usage_rows(&gw, 1, 5).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let rows = usage_rows_of(&gw);
    assert_eq!(rows.len(), 1, "{}", rows_text(&rows));
    assert_eq!(
        rows[0]["request_id"].as_str(),
        sent.request_id.as_deref(),
        "{}",
        rows[0]
    );
}

/// Requests refused before any provider is called cost nothing and must not look like calls: no
/// row at all, so no row can name a provider that never served. BYO traffic is not ours to bill.
/// claim: BIL-12
#[tokio::test]
async fn refusals_before_the_upstream_write_no_row() {
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(104);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats.port, &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .start()
        .await;
    let ok = billing_vkey(&sk, 104);
    let denied = billing_vkey(&sk, 105);
    put_kv(nats.port, "blackhole.105", b"spend").await;
    // Wait for the deny delta.
    wait_for_status(402, || {
        let (url, k) = (gw.url(), denied.clone());
        async move {
            send(
                format!("{url}/openai/v1/chat/completions"),
                bearer(&k),
                CHAT,
                &[],
            )
            .await
            .status
        }
    })
    .await;
    let url = gw.url();
    let cases: Vec<(&str, Sent)> = vec![
        (
            "bad key",
            send(
                format!("{url}/openai/v1/chat/completions"),
                bearer("bai_v1.1.bogus.bogus"),
                CHAT,
                &[],
            )
            .await,
        ),
        (
            "denied tenant",
            send(
                format!("{url}/openai/v1/chat/completions"),
                bearer(&denied),
                CHAT,
                &[],
            )
            .await,
        ),
        (
            "unknown catalog model",
            send(
                format!("{url}/v1/chat/completions"),
                bearer(&ok),
                r#"{"model":"no-such-model","messages":[]}"#,
                &[],
            )
            .await,
        ),
        (
            "unconfigured provider",
            send(
                format!("{url}/anthropic/v1/messages"),
                bearer(&ok),
                r#"{"model":"claude-opus-4-8","max_tokens":8,"messages":[]}"#,
                &[],
            )
            .await,
        ),
        (
            "managed non-generation endpoint",
            send(format!("{url}/openai/v1/files"), bearer(&ok), "{}", &[]).await,
        ),
    ];
    for (what, sent) in &cases {
        assert!(sent.status >= 400, "{what}: {} {}", sent.status, sent.text);
    }
    assert_eq!(mock.hits(), 0, "nothing reached the provider");
    // A BYO call does reach the provider, and still writes no row.
    let byo = send(
        format!("{url}/openai/v1/chat/completions"),
        bearer("sk-user-own"),
        CHAT,
        &[],
    )
    .await;
    assert_eq!(byo.status, 200);
    tokio::time::sleep(Duration::from_millis(500)).await;
    let rows = usage_rows_of(&gw);
    assert!(
        rows.is_empty(),
        "statuses {:?}; refused and BYO calls wrote rows:\n{}",
        cases
            .iter()
            .map(|(w, s)| (*w, s.status))
            .collect::<Vec<_>>(),
        rows_text(&rows)
    );
}

/// A token count is free on every route, not only on the catalog walk: the provider-routed
/// `/anthropic/v1/messages/count_tokens` and `/openai/v1/responses/input_tokens` must write no
/// row either (they carry no usage block, so a row there is a fabricated zero-token call that also
/// trips the parse-error alarm).
/// claim: BIL-18
/// defect: D69
#[tokio::test]
async fn token_counts_write_no_row_on_a_provider_route() {
    let (pubkey, sk) = test_keypair(106);
    let anthropic =
        MockUpstream::start(Mode::Raw(200, "application/json", r#"{"input_tokens":9}"#)).await;
    let openai = MockUpstream::start(Mode::Raw(
        200,
        "application/json",
        r#"{"object":"response.input_tokens","input_tokens":9}"#,
    ))
    .await;
    let gw = Gateway::builder(unused_nats_port(), &anthropic.authority(), &b64(&pubkey))
        .providers(&["anthropic", "openai"])
        .provider_authority("openai", &openai.authority())
        .start()
        .await;
    let key = billing_vkey(&sk, 106);
    let a = send(
        format!("{}/anthropic/v1/messages/count_tokens", gw.url()),
        ("x-api-key", key.clone()),
        r#"{"model":"claude-opus-4-8","messages":[{"role":"user","content":"hi"}]}"#,
        &[("anthropic-version", "2023-06-01")],
    )
    .await;
    let o = send(
        format!("{}/openai/v1/responses/input_tokens", gw.url()),
        bearer(&key),
        r#"{"model":"gpt-4o-mini","input":"hi"}"#,
        &[],
    )
    .await;
    assert_eq!((a.status, o.status), (200, 200), "{} / {}", a.text, o.text);
    assert_eq!((anthropic.hits(), openai.hits()), (1, 1));
    tokio::time::sleep(Duration::from_millis(500)).await;
    let rows = usage_rows_of(&gw);
    assert!(
        rows.is_empty(),
        "a token count is free on every route:\n{}",
        rows_text(&rows)
    );
    assert_eq!(
        gw.metric("ai_usage_parse_errors_total", "").await,
        0.0,
        "a free sub-resource is not a usage-shape regression"
    );
}

/// A cache hit is defined as: one row per hit, `cache_hit=true`, the fill's provider, models and
/// exact token counts, `usage_estimated=false`, and its own `request_id` (the one the client got),
/// so a hit is billable like the call it replays yet distinguishable from it. Opting out
/// (`x-beyond-cache: off`) goes upstream and writes an ordinary row.
/// claim: BIL-17
#[tokio::test]
async fn a_cache_hit_row_replays_the_fill_under_its_own_id() {
    let (pubkey, sk) = test_keypair(107);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .cache_ttl_secs(60)
        .start()
        .await;
    let key = billing_vkey(&sk, 107);
    let url = format!("{}/v1/chat/completions", gw.url());
    let body = r#"{"model":"gpt-4o-mini","messages":[{"role":"user","content":"cache me"}]}"#;
    let fill = send(url.clone(), bearer(&key), body, &[]).await;
    assert_eq!(fill.status, 200);
    let _ = wait_usage_rows(&gw, 1, 5).await;
    let hit = send(url.clone(), bearer(&key), body, &[]).await;
    assert_eq!(hit.status, 200);
    assert_eq!(hit.text, fill.text, "replayed byte for byte");
    assert_eq!(mock.hits(), 1, "the hit did not reach the provider");
    let off = send(url, bearer(&key), body, &[("x-beyond-cache", "off")]).await;
    assert_eq!(off.status, 200);
    assert_eq!(mock.hits(), 2, "an opt-out goes upstream");
    let rows = wait_usage_rows(&gw, 3, 5).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let rows = if rows.len() == 3 {
        usage_rows_of(&gw)
    } else {
        rows
    };
    assert_eq!(rows.len(), 3, "{}", rows_text(&rows));
    let by_id = |id: &Option<String>| {
        rows.iter()
            .find(|r| r["request_id"].as_str() == id.as_deref())
            .unwrap_or_else(|| panic!("no row for {id:?}:\n{}", rows_text(&rows)))
    };
    let (f, h, o) = (
        by_id(&fill.request_id),
        by_id(&hit.request_id),
        by_id(&off.request_id),
    );
    assert_eq!(f["cache_hit"], false, "{f}");
    assert_eq!(h["cache_hit"], true, "{h}");
    assert_eq!(o["cache_hit"], false, "{o}");
    for field in [
        "provider",
        "model",
        "requested_model",
        "routed_model",
        "stream",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "tenant_id",
    ] {
        assert_eq!(
            h[field], f[field],
            "hit {field} differs from the fill:\n{f}\n{h}"
        );
    }
    assert_eq!(h["usage_estimated"], false, "{h}");
    assert_ne!(h["request_id"], f["request_id"]);
}

/// Server-sent text is not a protocol signal. An Anthropic stream cut short after the model wrote
/// the literal words `message_delta` (any coding agent working on SSE code will) must still be
/// billed an estimate for its output; the finished-stream check must look at event types, not at
/// a substring anywhere in the tail.
/// claim: BIL-22
/// defect: D70
#[tokio::test]
async fn generated_message_delta_text_does_not_hide_a_cut_short_anthropic_stream() {
    let (pubkey, sk) = test_keypair(108);
    let mut sse = String::from(
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-opus-4-8\",\"content\":[],\"usage\":{\"input_tokens\":500,\"output_tokens\":1}}}\n\n",
    );
    for _ in 0..40 {
        sse.push_str("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" handle the message_delta event\"}}\n\n");
    }
    let up = ReplyUpstream::start(move |_, _| Reply::Stall {
        status: 200,
        content_type: "text/event-stream",
        first: bytes::Bytes::from(sse.clone()),
    })
    .await;
    let gw = Gateway::builder(unused_nats_port(), &up.authority(), &b64(&pubkey))
        .providers(&["anthropic"])
        .start()
        .await;
    let mut resp = test_client()
        .post(format!("{}/anthropic/v1/messages", gw.url()))
        .header("x-api-key", billing_vkey(&sk, 108))
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .body(r#"{"model":"claude-opus-4-8","max_tokens":512,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let first = tokio::time::timeout(Duration::from_secs(10), resp.chunk())
        .await
        .expect("a first chunk")
        .unwrap();
    assert!(first.is_some());
    drop(resp);
    let row = usage_row_of(&gw).await;
    assert_eq!(row["input_tokens"].as_u64(), Some(500), "{row}");
    assert_eq!(row["usage_estimated"], true, "{row}");
    assert!(
        row["output_tokens"].as_u64().unwrap_or(0) >= 40,
        "40 generated deltas were relayed: {row}"
    );
}

/// The OpenAI twin: generated text that spells out a usage object is JSON-escaped inside the
/// delta, so it can never be read as the stream's usage. A cut-short stream stays estimated.
/// claim: BIL-22
#[tokio::test]
async fn generated_usage_text_does_not_hide_a_cut_short_openai_stream() {
    let (pubkey, sk) = test_keypair(109);
    let mut sse = String::new();
    for _ in 0..40 {
        sse.push_str("data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"data: {\\\"choices\\\":[],\\\"usage\\\":{\\\"prompt_tokens\\\":1,\\\"completion_tokens\\\":1}}\"}}]}\n\n");
    }
    let up = ReplyUpstream::start(move |_, _| Reply::Stall {
        status: 200,
        content_type: "text/event-stream",
        first: bytes::Bytes::from(sse.clone()),
    })
    .await;
    let gw = Gateway::builder(unused_nats_port(), &up.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let mut resp = test_client()
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header(
            "authorization",
            format!("Bearer {}", billing_vkey(&sk, 109)),
        )
        .header("content-type", "application/json")
        .body(r#"{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"hi"}]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let first = tokio::time::timeout(Duration::from_secs(10), resp.chunk())
        .await
        .expect("a first chunk")
        .unwrap();
    assert!(first.is_some());
    drop(resp);
    let row = usage_row_of(&gw).await;
    assert_eq!(row["usage_estimated"], true, "{row}");
    assert!(row["output_tokens"].as_u64().unwrap_or(0) >= 40, "{row}");
}

/// CRC-32 (IEEE), bitwise. Only for building the gzip fixture below.
fn crc32(data: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &b in data {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// A valid gzip member holding `data` in one stored (uncompressed) deflate block.
fn gzip_stored(data: &[u8]) -> Vec<u8> {
    assert!(data.len() <= 0xFFFF);
    let mut out = vec![0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 0xff];
    let len = data.len() as u16;
    out.push(1);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&(!len).to_le_bytes());
    out.extend_from_slice(data);
    out.extend_from_slice(&crc32(data).to_le_bytes());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out
}

/// A provider that ignores `Accept-Encoding: identity` and gzips anyway must not turn the call
/// into a free one: the row carries the real counts, or a flagged non-zero estimate.
/// claim: BIL-16
#[tokio::test]
async fn a_compressed_upstream_body_never_bills_zero() {
    let (pubkey, sk) = test_keypair(110);
    let json = br#"{"id":"chatcmpl-mock","object":"chat.completion","model":"gpt-4o-2024-08-06","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}}"#;
    let sse = b"data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":9}}\n\ndata: [DONE]\n\n";
    let reply = |ct: &str, body: &[u8]| {
        let gz = gzip_stored(body);
        let mut out = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: {ct}\r\ncontent-encoding: gzip\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
            gz.len()
        )
        .into_bytes();
        out.extend_from_slice(&gz);
        out
    };
    let (json_reply, sse_reply) = (
        reply("application/json", json),
        reply("text/event-stream", sse),
    );
    let up = ScriptedUpstream::start(move |body, _| {
        let streaming = String::from_utf8_lossy(body).contains("\"stream\":true");
        vec![Step::Write(if streaming {
            sse_reply.clone()
        } else {
            json_reply.clone()
        })]
    })
    .await;
    let gw = Gateway::builder(unused_nats_port(), &up.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let key = billing_vkey(&sk, 110);
    for body in [
        CHAT,
        r#"{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"hi"}]}"#,
    ] {
        let resp = test_client()
            .post(format!("{}/openai/v1/chat/completions", gw.url()))
            .header("authorization", format!("Bearer {key}"))
            .header("content-type", "application/json")
            .header("accept-encoding", "gzip")
            .body(body)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let _ = resp.bytes().await;
    }
    let rows = wait_usage_rows(&gw, 2, 5).await;
    assert_eq!(rows.len(), 2, "{}", gw.log());
    for row in &rows {
        let total =
            row["input_tokens"].as_u64().unwrap_or(0) + row["output_tokens"].as_u64().unwrap_or(0);
        assert!(total > 0, "a compressed response billed zero: {row}");
    }
    // The gateway decodes the body before metering: the counts are exact, not estimated.
    assert_eq!(rows[0]["input_tokens"].as_u64(), Some(11), "{}", rows[0]);
    assert_eq!(rows[1]["output_tokens"].as_u64(), Some(9), "{}", rows[1]);
    assert_eq!(rows[1]["usage_estimated"], false, "{}", rows[1]);
}

/// The billed model is the snapshot the provider echoed, on Chat Completions (body and stream)
/// and on Messages; the alias the client sent is `requested_model`. Every row's model resolves to
/// a price key only if it is the id the provider invoices.
/// claim: BIL-13
#[tokio::test]
async fn rows_bill_the_echoed_snapshot_on_chat_and_messages() {
    let (pubkey, sk) = test_keypair(111);
    let json = MockUpstream::start(Mode::Json).await;
    let sse = MockUpstream::start(Mode::Sse).await;
    let messages = MockUpstream::start(Mode::Raw(
        200,
        "application/json",
        r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-4-8-20260115","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":13,"output_tokens":7}}"#,
    ))
    .await;
    let key = billing_vkey(&sk, 111);
    let cases = [
        (
            &json,
            "openai",
            "/openai/v1/chat/completions",
            CHAT,
            "gpt-4o",
            "gpt-4o-2024-08-06",
        ),
        (
            &sse,
            "openai",
            "/openai/v1/chat/completions",
            r#"{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"hi"}]}"#,
            "gpt-4o",
            "gpt-4o-2024-08-06",
        ),
        (
            &messages,
            "anthropic",
            "/anthropic/v1/messages",
            r#"{"model":"claude-opus-4-8","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}"#,
            "claude-opus-4-8",
            "claude-opus-4-8-20260115",
        ),
    ];
    for (mock, provider, path, body, requested, billed) in cases {
        let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
            .providers(&[provider])
            .start()
            .await;
        let auth = if provider == "anthropic" {
            ("x-api-key", key.clone())
        } else {
            bearer(&key)
        };
        let sent = send(
            format!("{}{path}", gw.url()),
            auth,
            body,
            &[("anthropic-version", "2023-06-01")],
        )
        .await;
        assert_eq!(sent.status, 200, "{path}: {}", sent.text);
        let row = usage_row_of(&gw).await;
        assert_eq!(row["requested_model"], requested, "{path}: {row}");
        assert_eq!(row["model"], billed, "{path}: {row}");
    }
}

/// The client's usage, as `(total prompt incl. cache, cache_read, output)`.
fn client_usage(text: &str) -> (u64, u64, u64) {
    let mut found = None;
    let mut consider = |v: &serde_json::Value| {
        let u = if v["usage"].is_object() {
            &v["usage"]
        } else if v["message"]["usage"].is_object() {
            &v["message"]["usage"]
        } else {
            return;
        };
        let n = |k: &str| u[k].as_u64().unwrap_or(0);
        let (prompt, cache, out) = if u.get("prompt_tokens").is_some() {
            (
                n("prompt_tokens"),
                u["prompt_tokens_details"]["cached_tokens"]
                    .as_u64()
                    .unwrap_or(0),
                n("completion_tokens"),
            )
        } else {
            (
                n("input_tokens") + n("cache_read_input_tokens") + n("cache_creation_input_tokens"),
                n("cache_read_input_tokens"),
                n("output_tokens"),
            )
        };
        // Anthropic streams split usage across message_start and message_delta: keep the max of
        // each field over events (both are cumulative).
        let prev = found.unwrap_or((0, 0, 0));
        found = Some((prev.0.max(prompt), prev.1.max(cache), prev.2.max(out)));
    };
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(text) {
        consider(&v);
    } else {
        for line in text.lines() {
            if let Some(data) = line.strip_prefix("data: ")
                && let Ok(v) = serde_json::from_str::<serde_json::Value>(data)
            {
                consider(&v);
            }
        }
    }
    found.unwrap_or_else(|| panic!("no usage shown to the client: {text}"))
}

/// Row tokens equal what the client was shown, after normalizing each side's input semantics
/// (Chat Completions' `prompt_tokens` includes cache; Messages' `input_tokens` does not), for every
/// translation pair, streamed and not, with cache reads in play.
/// claim: BIL-6, BIL-7
#[tokio::test]
async fn row_tokens_match_the_client_shown_usage_on_every_translation_pair() {
    let (pubkey, sk) = test_keypair(112);
    let key = billing_vkey(&sk, 112);
    const CLAUDE_JSON: &str = r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-4-8","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":100,"cache_read_input_tokens":900,"cache_creation_input_tokens":0,"output_tokens":7}}"#;
    const GPT_JSON: &str = r#"{"id":"chatcmpl-1","object":"chat.completion","model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1000,"completion_tokens":7,"total_tokens":1007,"prompt_tokens_details":{"cached_tokens":900}}}"#;
    // (client path, client body, upstream mode, label)
    let cases: [(&str, &str, Mode, &str); 4] = [
        (
            "/v1/chat/completions",
            r#"{"model":"claude-opus-4-8","messages":[{"role":"user","content":"hi"}]}"#,
            Mode::Raw(200, "application/json", CLAUDE_JSON),
            "chat client -> claude, body",
        ),
        (
            "/v1/chat/completions",
            r#"{"model":"claude-opus-4-8","stream":true,"messages":[{"role":"user","content":"hi"}]}"#,
            Mode::AnthropicThinkingSse,
            "chat client -> claude, stream",
        ),
        (
            "/v1/messages",
            r#"{"model":"gpt-4o-mini","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}"#,
            Mode::Raw(200, "application/json", GPT_JSON),
            "messages client -> gpt, body",
        ),
        (
            "/v1/messages",
            r#"{"model":"gpt-4o-mini","max_tokens":16,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#,
            Mode::Sse,
            "messages client -> gpt, stream",
        ),
    ];
    for (path, body, mode, label) in cases {
        let mock = MockUpstream::start(mode).await;
        let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
            .providers(&["anthropic", "openai", "openrouter"])
            .start()
            .await;
        let auth = if path.ends_with("/messages") {
            ("x-api-key", key.clone())
        } else {
            bearer(&key)
        };
        let sent = send(
            format!("{}{path}", gw.url()),
            auth,
            body,
            &[("anthropic-version", "2023-06-01")],
        )
        .await;
        assert_eq!(sent.status, 200, "{label}: {}", sent.text);
        let (c_prompt, c_cache, c_out) = client_usage(&sent.text);
        let row = usage_row_of(&gw).await;
        let n = |k: &str| row[k].as_u64().unwrap_or(0);
        let row_total = n("input_tokens") + n("cache_read_tokens") + n("cache_write_tokens");
        assert!(
            row_total == c_prompt || n("input_tokens") == c_prompt,
            "{label}: client saw {c_prompt} prompt tokens, row: {row}\nclient: {}",
            sent.text
        );
        assert_eq!(n("cache_read_tokens"), c_cache, "{label}: {row}");
        assert_eq!(n("output_tokens"), c_out, "{label}: {row}");
        assert_eq!(row["usage_estimated"], false, "{label}: {row}");
    }
}
