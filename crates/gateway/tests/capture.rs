//! End-to-end: the `x-beyond-*` control surface and optional payload capture.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).
//!
//! The headline invariant is the **first** test: capture must not change a single relayed byte, in
//! either direction. Everything else here is a property of what gets *logged*; that one is a
//! property of what gets *proxied*, and it is the one that would make this feature unshippable.

// Test target: `.unwrap()`/`.expect()`/`panic!` are assertions, not production code — allow the
// panic-surface restriction lints denied workspace-wide in `[workspace.lints.clippy]`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use beyond_ai::key::{VirtualKey, mint};
use common::*;

const TENANT: u64 = 42;

fn vkey(sk: &ed25519_dalek::SigningKey, tenant_id: u64) -> String {
    mint(
        &VirtualKey {
            tenant_id,
            vpc_id: 7,
        },
        1,
        sk,
    )
}

fn body() -> &'static str {
    r#"{"model":"gpt-4o","messages":[{"role":"user","content":"who are you"}]}"#
}

/// Wait until `pred` holds over the gateway's captured log, or fail with the log for diagnosis.
///
/// Payload lines are written by a *separate thread* behind a bounded queue (see `capture_sink`), so
/// unlike `ai.usage` they are not guaranteed to have landed by the time the HTTP response does.
/// Polling here is not flake-papering — it is the asynchrony the design deliberately introduced.
async fn wait_for_log(gw: &Gateway, what: &str, pred: impl Fn(&str) -> bool) -> String {
    for _ in 0..200 {
        let log = gw.log();
        if pred(&log) {
            return log;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    panic!(
        "timed out waiting for {what}\n--- gateway log ---\n{}",
        gw.log()
    );
}

/// The one `ai.payload` line in the log, parsed.
fn payload_line(log: &str) -> serde_json::Value {
    let line = log
        .lines()
        .find(|l| l.contains(r#""target":"ai.payload""#))
        .unwrap_or_else(|| panic!("no ai.payload line in:\n{log}"));
    serde_json::from_str(line).expect("payload line is JSON")
}

/// `tracing`'s JSON layer nests event fields under `fields`.
fn field<'a>(v: &'a serde_json::Value, name: &str) -> &'a serde_json::Value {
    &v["fields"][name]
}

/// Switch capture on for `tenant` and **wait until the gateway has actually applied the delta**.
///
/// The `ai_capture_set_size` gauge is the synchronization point. Without it every test here would
/// have to send requests in a loop until one happened to be captured, which makes "how many payload
/// lines are there?" unanswerable — the surplus requests keep landing asynchronously behind the
/// bounded queue and race the assertion.
async fn enable_capture(gw: &Gateway, nats_port: u16, tenant: u64) {
    put_kv(nats_port, &format!("aicapture.{tenant}"), b"{}").await;
    wait_for_metric(gw, "ai_capture_set_size", "", 1.0).await;
}

#[tokio::test]
async fn capture_does_not_alter_a_single_relayed_byte() {
    // THE invariant. Capture is a passive tap; if it perturbs the proxied bytes in either direction
    // it is not shippable, no matter how good the logging is. Asserted against the same mock the
    // passthrough-fidelity tests in `e2e.rs` use, with capture forced on.
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(70);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::start(nats.port, &mock.authority(), &b64(&pubkey)).await;
    let client = test_client();

    // Baseline: same request, capture off.
    let before = client
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk, TENANT)))
        .header("content-type", "application/json")
        .body(body())
        .send()
        .await
        .unwrap();
    let before_status = before.status();
    let before_body = before.text().await.unwrap();
    let sent_without_capture = mock.captured().expect("mock saw the request").body;

    enable_capture(&gw, nats.port, TENANT).await;

    // `enable_capture` already waited for the delta, so this single request is definitely captured.
    let after = client
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk, TENANT)))
        .header("content-type", "application/json")
        .body(body())
        .send()
        .await
        .unwrap();
    let after_status = after.status();
    let after_body = after.text().await.unwrap();
    let sent_with_capture = mock.captured().expect("mock saw the request").body;

    // Proves the comparison was meaningful — without this the test would also pass if capture had
    // silently never engaged.
    wait_for_log(&gw, "proof that capture engaged", |l| {
        l.contains(r#""target":"ai.payload""#)
    })
    .await;

    assert_eq!(
        sent_without_capture, sent_with_capture,
        "capture altered the bytes forwarded upstream"
    );
    assert_eq!(before_status, after_status, "capture altered the status");
    assert_eq!(
        before_body, after_body,
        "capture altered the bytes relayed back downstream"
    );
}

#[tokio::test]
async fn control_plane_capture_emits_both_bodies_and_correlates_by_request_id() {
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(71);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::start(nats.port, &mock.authority(), &b64(&pubkey)).await;
    let client = test_client();

    enable_capture(&gw, nats.port, TENANT).await;

    // Exactly one request, so exactly one payload line is expected and the correlation assertion
    // below can't be satisfied by a different request's line.
    let r = client
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk, TENANT)))
        .header("content-type", "application/json")
        .body(body())
        .send()
        .await
        .unwrap();
    let request_id = r
        .headers()
        .get("x-beyond-request-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();

    let log = wait_for_log(&gw, "an ai.payload line", |l| {
        l.contains(r#""target":"ai.payload""#)
    })
    .await;
    let payload = payload_line(&log);

    // The whole user journey: "user quotes the request id from their response header, we pull the
    // conversation". That only works if the header and the payload line carry the same id.
    assert!(!request_id.is_empty(), "no x-beyond-request-id header");
    assert_eq!(
        field(&payload, "request_id").as_str(),
        Some(request_id.as_str()),
        "payload line must correlate with the id the client was handed"
    );
    assert_eq!(field(&payload, "tenant_id").as_u64(), Some(TENANT));

    let req = field(&payload, "request_body").as_str().unwrap();
    assert!(
        req.contains("who are you"),
        "the prompt is the point of capturing at all: {req}"
    );
    assert!(
        field(&payload, "response_body")
            .as_str()
            .is_some_and(|s| !s.is_empty()),
        "response body missing from {payload}"
    );
    assert_eq!(field(&payload, "complete").as_bool(), Some(true));
    assert_eq!(field(&payload, "request_truncated").as_bool(), Some(false));
}

#[tokio::test]
async fn a_tenant_without_a_rule_is_never_captured() {
    // Default-off is the whole safety posture of a feature that stores customer prompts. A tenant
    // nobody switched on must produce no payload line at all.
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(72);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::start(nats.port, &mock.authority(), &b64(&pubkey)).await;
    let client = test_client();

    // Switch on a *different* tenant, so the capture-set is non-empty and the watcher demonstrably
    // ran — otherwise this test would also pass if capture were simply broken.
    enable_capture(&gw, nats.port, 999).await;

    for _ in 0..8 {
        client
            .post(format!("{}/openai/v1/chat/completions", gw.url()))
            .header("authorization", format!("Bearer {}", vkey(&sk, TENANT)))
            .header("content-type", "application/json")
            .body(body())
            .send()
            .await
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    // The usage rows prove the requests were served and metered...
    let log = wait_for_log(&gw, "usage rows", |l| l.contains(r#""target":"ai.usage""#)).await;
    // ...while nothing was captured.
    assert!(
        !log.contains(r#""target":"ai.payload""#),
        "captured a tenant that was never enabled:\n{log}"
    );
}

#[tokio::test]
async fn the_header_enables_capture_for_an_unenabled_tenant_and_is_never_sampled_away() {
    // Two properties in one request, because they only matter together: a caller can ask for a
    // single trace with no control-plane entry at all, and that explicit ask must survive a sample
    // rate that would otherwise drop 999 of every 1000 requests. A caller who asks to log one trace
    // and silently gets nothing is the outcome that makes the whole feature useless.
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(73);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats.port, &mock.authority(), &b64(&pubkey))
        .capture_default_sample_n(1000)
        .start()
        .await;
    let client = test_client();

    let r = client
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk, TENANT)))
        .header("content-type", "application/json")
        .header("x-beyond-capture", "on")
        .body(body())
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);

    let log = wait_for_log(&gw, "a header-requested payload", |l| {
        l.contains(r#""target":"ai.payload""#)
    })
    .await;
    let payload = payload_line(&log);
    assert!(
        field(&payload, "request_body")
            .as_str()
            .is_some_and(|s| s.contains("who are you")),
        "{payload}"
    );
}

#[tokio::test]
async fn the_header_can_suppress_capture_for_an_enabled_tenant() {
    // The other direction, and the one that matters for privacy: a caller must be able to say "not
    // this request, it has PII" even while their tenant is switched on.
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(74);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::start(nats.port, &mock.authority(), &b64(&pubkey)).await;
    let client = test_client();

    enable_capture(&gw, nats.port, TENANT).await;

    // First establish that capture really is on for this tenant — one request, one payload line.
    client
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk, TENANT)))
        .header("content-type", "application/json")
        .body(body())
        .send()
        .await
        .unwrap();
    wait_for_log(&gw, "the baseline payload", |l| {
        l.contains(r#""target":"ai.payload""#)
    })
    .await;
    let before = gw.log().matches(r#""target":"ai.payload""#).count();
    assert_eq!(before, 1, "expected exactly one baseline capture");

    // ...then that an opted-out request adds no payload line, while still being served.
    let r = client
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk, TENANT)))
        .header("content-type", "application/json")
        .header("x-beyond-capture", "off")
        .header("x-beyond-metadata", r#"{"sensitive":"yes"}"#)
        .body(r#"{"model":"gpt-4o","messages":[{"role":"user","content":"SSN 000-00-0000"}]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);

    // Give any payload line time to land before concluding none did.
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    let log = gw.log();
    assert_eq!(
        log.matches(r#""target":"ai.payload""#).count(),
        before,
        "a suppressed request was captured anyway:\n{log}"
    );
    assert!(
        !log.contains("000-00-0000"),
        "suppressed payload leaked into the log:\n{log}"
    );
}

#[tokio::test]
async fn metadata_reaches_the_billing_row_but_never_the_provider() {
    // Tagging is useful with capture *off* — that's the point of shipping it alongside. It must
    // land on `ai.usage` (where the tokens are, so `GROUP BY metadata['feature']` answers "which
    // feature is burning money") and must not be forwarded to the provider.
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(75);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::start(nats.port, &mock.authority(), &b64(&pubkey)).await;
    let client = test_client();

    let r = client
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk, TENANT)))
        .header("content-type", "application/json")
        // Deliberately unsorted, to pin the canonicalization.
        .header(
            "x-beyond-metadata",
            r#"{"org":"acme","feature":"summarizer"}"#,
        )
        .body(body())
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);

    let log = wait_for_log(&gw, "a usage row", |l| l.contains(r#""target":"ai.usage""#)).await;
    let usage: serde_json::Value = serde_json::from_str(
        log.lines()
            .find(|l| l.contains(r#""target":"ai.usage""#))
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        field(&usage, "metadata").as_str(),
        Some(r#"{"feature":"summarizer","org":"acme"}"#),
        "metadata must arrive canonicalized and key-sorted: {usage}"
    );

    let captured = mock.captured().expect("mock saw the request");
    assert_eq!(
        captured.beyond_metadata, None,
        "our control header leaked upstream"
    );
}

#[tokio::test]
async fn a_malformed_control_header_is_dropped_not_rejected() {
    // An observability header that can 400 a customer's inference call is a worse bug than the
    // missing observability. Junk in both headers, and the request must still be served normally.
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(76);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::start(nats.port, &mock.authority(), &b64(&pubkey)).await;
    let client = test_client();

    let r = client
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk, TENANT)))
        .header("content-type", "application/json")
        .header("x-beyond-metadata", "not json at all")
        .header("x-beyond-capture", "maybe")
        .body(body())
        .send()
        .await
        .unwrap();
    assert_eq!(
        r.status(),
        200,
        "a junk control header rejected the request"
    );

    let log = wait_for_log(&gw, "a usage row", |l| l.contains(r#""target":"ai.usage""#)).await;
    let usage: serde_json::Value = serde_json::from_str(
        log.lines()
            .find(|l| l.contains(r#""target":"ai.usage""#))
            .unwrap(),
    )
    .unwrap();
    assert!(
        field(&usage, "metadata").is_null(),
        "unusable metadata must be absent, not passed through: {usage}"
    );

    // And the drop is visible, so a client whose tags never appear can find out why.
    let metrics = gw.metrics().await;
    assert!(
        parse_metric(&metrics, "ai_control_header_errors_total", "") >= 1.0,
        "malformed headers were dropped without being counted:\n{metrics}"
    );

    // The capture header was junk too, so it must not have enabled capture by accident.
    assert!(
        !gw.log().contains(r#""target":"ai.payload""#),
        "an unparseable capture header switched capture on"
    );
    assert_eq!(
        mock.captured().expect("mock saw it").beyond_capture,
        None,
        "our control header leaked upstream"
    );
}

#[tokio::test]
async fn an_oversize_body_is_truncated_at_the_head_and_flagged() {
    // Truncation keeps the *front* — the system prompt and opening messages, the part that explains
    // what the agent was told to do. And it must be flagged: a capture that reads as complete when
    // it isn't produces confident wrong conclusions mid-incident.
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(77);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats.port, &mock.authority(), &b64(&pubkey))
        .capture_max_bytes(64)
        .start()
        .await;
    let client = test_client();

    let big = format!(
        r#"{{"model":"gpt-4o","messages":[{{"role":"user","content":"{}"}}]}}"#,
        "z".repeat(4096)
    );

    for _ in 0..200 {
        client
            .post(format!("{}/openai/v1/chat/completions", gw.url()))
            .header("authorization", format!("Bearer {}", vkey(&sk, TENANT)))
            .header("content-type", "application/json")
            .header("x-beyond-capture", "on")
            .body(big.clone())
            .send()
            .await
            .unwrap();
        if gw.log().contains(r#""target":"ai.payload""#) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }

    let log = wait_for_log(&gw, "a truncated payload", |l| {
        l.contains(r#""target":"ai.payload""#)
    })
    .await;
    let payload = payload_line(&log);
    let req = field(&payload, "request_body").as_str().unwrap();
    assert!(req.len() <= 64, "cap not enforced: {} bytes", req.len());
    assert!(
        req.starts_with(r#"{"model":"gpt-4o""#),
        "kept the wrong end — truncation must preserve the head: {req}"
    );
    assert_eq!(field(&payload, "request_truncated").as_bool(), Some(true));
}
