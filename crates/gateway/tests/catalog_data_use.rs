//! End-to-end: a catalog row whose vendor may train on what it serves
//! (`providers::catalog::DataUse::TrainsOnInputs`, Meta's Muse Spark Contributor rows) is served
//! only to a caller who names it, and `GET /v1/models` labels it.
//!
//! - Named by its catalog id (body `model` or `x-beyond-model`) or its OpenRouter spelling, it is
//!   served.
//! - No other row's walk reaches its endpoint, whatever `x-beyond-only`, `x-beyond-order` or
//!   `x-beyond-split` asks: those reorder or filter a row's own candidates, and no other row lists
//!   a training endpoint.
//! - A name that is not a row (a newer version, a family stem) is a 404, never the nearest row.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::sync::{Arc, Mutex};

use common::*;
use providers::ProviderId;
use providers::catalog::MODEL_ROUTES;
use serde_json::{Value, json};

type Seen = Arc<Mutex<Vec<(String, Value)>>>;

/// The upstream ids of every candidate whose row trains on inputs.
fn training_endpoints() -> Vec<&'static str> {
    MODEL_ROUTES
        .iter()
        .filter(|r| r.card.data_use.trains_on_inputs())
        .flat_map(|r| r.candidates.iter().chain(r.responses))
        .map(|c| c.upstream_model)
        .collect()
}

async fn gateway(seen: &Seen, seed: u8) -> (ReplyUpstream, Gateway, String) {
    let seen = seen.clone();
    let up = ReplyUpstream::start(move |_, req: &ScriptReq| {
        let body: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
        seen.lock().unwrap().push((req.path.clone(), body));
        Reply::ok()
    })
    .await;
    let (pubkey, sk) = test_keypair(seed);
    let gw = Gateway::builder(unused_nats_port(), &up.authority(), &b64(&pubkey))
        .providers(&[
            "openai",
            "anthropic",
            "openrouter",
            "together",
            "bedrock",
            "xai",
        ])
        .provider_authority("openrouter", &up.authority())
        .start()
        .await;
    (up, gw, billing_vkey(&sk, seed.into()))
}

async fn chat(gw: &Gateway, key: &str, model: &str, headers: &[(&str, &str)]) -> u16 {
    let mut req = test_client()
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .body(
            json!({"model": model, "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}]})
            .to_string(),
        );
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    req.send().await.unwrap().status().as_u16()
}

/// Named, each training row is served, by its catalog id, by `x-beyond-model`, and by its
/// OpenRouter spelling; the upstream gets that row's endpoint.
/// claim: CAT-17
#[tokio::test]
async fn a_row_that_trains_on_inputs_is_served_when_named() {
    let seen = Seen::default();
    let (_up, gw, key) = gateway(&seen, 71).await;
    let rows: Vec<_> = MODEL_ROUTES
        .iter()
        .filter(|r| r.card.data_use.trains_on_inputs())
        .collect();
    assert_eq!(rows.len(), 2);
    for r in rows {
        let slug = r.candidates[0].upstream_model;
        for (model, headers) in [
            (r.model, vec![]),
            (slug, vec![]),
            ("ignored", vec![("x-beyond-model", r.model)]),
        ] {
            assert_eq!(chat(&gw, &key, model, &headers).await, 200, "{model}");
            let (path, sent) = seen.lock().unwrap().last().cloned().unwrap();
            assert_eq!(path, "/api/v1/chat/completions");
            assert_eq!(sent["model"], slug, "{model} {headers:?}");
        }
    }
}

/// No walk of a row the caller named reaches a training endpoint: every row with an OpenRouter
/// candidate, pinned to OpenRouter with `only`, `order` or `split`, is sent its own model. And a
/// name that is not a row is a 404 that reaches no upstream, not the nearest training row.
/// claim: CAT-17
#[tokio::test]
async fn no_other_row_or_unknown_name_reaches_a_row_that_trains_on_inputs() {
    let seen = Seen::default();
    let (_up, gw, key) = gateway(&seen, 72).await;
    let training = training_endpoints();
    let mut checked = 0;
    for r in MODEL_ROUTES
        .iter()
        .filter(|r| !r.card.data_use.trains_on_inputs())
        .filter(|r| !r.candidates[0].path.ends_with("/embeddings"))
    {
        let Some(c) = r
            .candidates
            .iter()
            .find(|c| c.provider == ProviderId::OpenRouter)
        else {
            continue;
        };
        for h in [
            ("x-beyond-only", "openrouter"),
            ("x-beyond-order", "openrouter"),
            ("x-beyond-split", "openrouter=100"),
        ] {
            seen.lock().unwrap().clear();
            chat(&gw, &key, r.model, &[h]).await;
            let sent_all = seen.lock().unwrap().clone();
            assert!(
                !sent_all.is_empty(),
                "{} with {h:?} reached no upstream",
                r.model
            );
            for (_, sent) in &sent_all {
                let m = sent["model"].as_str().unwrap_or("");
                assert!(!training.contains(&m), "{} with {h:?} sent {m}", r.model);
                assert_eq!(m, c.upstream_model, "{} with {h:?}", r.model);
            }
            checked += 1;
        }
    }
    assert!(checked > 100, "{checked}");

    seen.lock().unwrap().clear();
    for name in [
        "meta/muse-spark-1.4-contributor",
        "meta/muse-spark-contributor",
        "muse-spark-1.3-contributor-2026",
        "meta/muse-spark",
        "auto",
    ] {
        assert_eq!(chat(&gw, &key, name, &[]).await, 404, "{name}");
    }
    assert!(
        seen.lock().unwrap().is_empty(),
        "a catalog miss reached an upstream"
    );
}

/// `GET /v1/models` labels each training row `"data_use": "trains_on_inputs"` with its vendor's
/// terms page, and no other row carries the field.
/// claim: CAT-17
#[tokio::test]
async fn the_models_list_labels_rows_that_train_on_inputs() {
    let seen = Seen::default();
    let (_up, gw, key) = gateway(&seen, 73).await;
    let resp = test_client()
        .get(format!("{}/v1/models", gw.url()))
        .header("authorization", format!("Bearer {key}"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let v: Value = resp.json().await.unwrap();
    let labeled: Vec<(&str, &str)> = v["data"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|m| m.get("data_use").is_some())
        .map(|m| {
            assert_eq!(m["data_use"], "trains_on_inputs", "{m}");
            (
                m["id"].as_str().unwrap(),
                m["data_use_terms"].as_str().unwrap(),
            )
        })
        .collect();
    let terms = "https://llama.developer.meta.com/docs/pricing-rate-limits";
    assert_eq!(
        labeled,
        [
            ("meta/muse-spark-1.2-contributor", terms),
            ("meta/muse-spark-1.3-contributor", terms)
        ]
    );
    assert!(seen.lock().unwrap().is_empty());
}

/// Meta's Muse Spark accepts only `tool_choice: "auto"`: a forced choice goes as `auto` with a
/// closing system message naming the tool, `none` goes with no tools at all, and `auto` as sent.
/// claim: T1
/// defect: D270
#[tokio::test]
async fn muse_spark_gets_only_the_tool_choice_it_accepts() {
    let seen = Seen::default();
    let (_up, gw, key) = gateway(&seen, 74).await;
    let tools = json!([{"type": "function", "function": {"name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]);
    let post = |choice: Value| {
        test_client()
            .post(format!("{}/v1/chat/completions", gw.url()))
            .header("authorization", format!("Bearer {key}"))
            .header("content-type", "application/json")
            .body(
                json!({"model": "meta/muse-spark-1.3-contributor", "tools": tools,
                    "tool_choice": choice,
                    "messages": [{"role": "user", "content": "Weather in Paris?"}]})
                .to_string(),
            )
            .send()
    };
    for (choice, must) in [
        (
            json!({"type": "function", "function": {"name": "get_weather"}}),
            "Respond by calling the `get_weather` tool.",
        ),
        (
            json!("required"),
            "Respond by calling one of the provided tools.",
        ),
    ] {
        assert_eq!(post(choice).await.unwrap().status().as_u16(), 200);
        let (_, sent) = seen.lock().unwrap().last().cloned().unwrap();
        assert_eq!(sent["tool_choice"], "auto", "{sent}");
        assert_eq!(
            sent["messages"][1],
            json!({"role": "system", "content": must}),
            "{sent}"
        );
    }
    assert_eq!(post(json!("none")).await.unwrap().status().as_u16(), 200);
    let (_, sent) = seen.lock().unwrap().last().cloned().unwrap();
    assert!(
        sent.get("tools").is_none() && sent.get("tool_choice").is_none(),
        "{sent}"
    );
    assert_eq!(sent["messages"].as_array().unwrap().len(), 1, "{sent}");
    assert_eq!(post(json!("auto")).await.unwrap().status().as_u16(), 200);
    let (_, sent) = seen.lock().unwrap().last().cloned().unwrap();
    assert_eq!(sent["tool_choice"], "auto");
    assert_eq!(sent["tools"], tools);
    assert_eq!(sent["messages"].as_array().unwrap().len(), 1, "{sent}");
}
