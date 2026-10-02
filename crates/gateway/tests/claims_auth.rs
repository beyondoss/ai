//! Claim tests: deny-set rejections reach the client as a JSON error an SDK can raise, before any
//! upstream is contacted.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::*;
use serde_json::Value;

const CHAT: &str = r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#;

async fn status_of(gw: &Gateway, key: &str) -> u16 {
    test_client()
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .body(CHAT)
        .send()
        .await
        .map(|r| r.status().as_u16())
        .unwrap_or(0)
}

/// A spend deny is 402 and a fraud deny is 403. Each carries a JSON error with a message (what an
/// SDK raises its typed error from), and neither reaches the provider.
/// claim: A2
#[tokio::test]
async fn deny_rejections_are_json_errors_with_no_upstream_call() {
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(57);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::start(nats.port, &mock.authority(), &b64(&pubkey)).await;

    for (tenant, reason, want) in [(5701u64, "spend", 402u16), (5702, "fraud", 403)] {
        let key = billing_vkey(&sk, tenant);
        {
            let (gw, key) = (&gw, key.clone());
            wait_for_status(200, move || {
                let key = key.clone();
                async move { status_of(gw, &key).await }
            })
            .await;
        }
        put_kv(nats.port, &format!("blackhole.{tenant}"), reason.as_bytes()).await;
        {
            let (gw, key) = (&gw, key.clone());
            wait_for_status(want, move || {
                let key = key.clone();
                async move { status_of(gw, &key).await }
            })
            .await;
        }
        let hits = mock.hits();
        let resp = test_client()
            .post(format!("{}/v1/chat/completions", gw.url()))
            .header("authorization", format!("Bearer {key}"))
            .header("content-type", "application/json")
            .body(CHAT)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), want, "{reason}");
        assert_eq!(
            resp.headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok()),
            Some("application/json"),
            "{reason}"
        );
        let v: Value = resp.json().await.unwrap();
        assert!(
            v["error"]["message"]
                .as_str()
                .is_some_and(|m| !m.is_empty()),
            "{reason}: {v}"
        );
        assert_eq!(
            mock.hits(),
            hits,
            "{reason}: a deny never reaches the provider"
        );
    }
}
