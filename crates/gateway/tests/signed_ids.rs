//! End-to-end: tenant-bound ids for the Responses state OpenAI keeps under Beyond's pool key
//! (`signed_id.rs`, D230, D231).
//!
//! Every managed tenant shares one OpenAI organization through the pool key, and OpenAI stores
//! each response by default. So the ids in a managed Responses answer are signed for the tenant it
//! went to, and every id a request sends back (`previous_response_id`, `conversation`, each
//! `input` item's `id`) is checked against the caller: another tenant's, a raw provider id, or an
//! altered one is a 400 that names the field when it is a reference, and the upstream never sees
//! the request; on a full item it is cut, and the item is read from its own content. The upstream
//! always gets its own ids back. BYO keys and rows without a store are untouched.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::*;
use serde_json::{Value, json};
use std::time::Duration;

const A: u64 = 501;
const B: u64 = 502;

const RESP: &str = "resp_0750331520328311006abeeacfb62c87d0bdb6cbe1c41eea26";
const RS: &str = "rs_0750331520328311006abeead0270c87d0b8128a81f8474f00";
const MSG: &str = "msg_0750331520328311006abeead0270c87d0b8128a81f8474fea";

/// An OpenAI Responses answer: a reasoning item and a message, each with its stored id.
const RESPONSE: &str = r#"{"id":"resp_0750331520328311006abeeacfb62c87d0bdb6cbe1c41eea26","object":"response","created_at":1759300000,"status":"completed","model":"gpt-4o-2024-08-06","previous_response_id":null,"output":[{"type":"reasoning","id":"rs_0750331520328311006abeead0270c87d0b8128a81f8474f00","summary":[]},{"type":"message","id":"msg_0750331520328311006abeead0270c87d0b8128a81f8474fea","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Hello.","annotations":[]}]}],"usage":{"input_tokens":5,"input_tokens_details":{"cached_tokens":0},"output_tokens":2,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":7}}"#;

/// The same answer streamed, as OpenAI streams it: the response id on `created` and `completed`,
/// the item id on `output_item.*` (`item.id`) and on every content event (`item_id`).
const STREAM: &str = concat!(
    "event: response.created\n",
    r#"data: {"type":"response.created","sequence_number":0,"response":{"id":"resp_0750331520328311006abeeacfb62c87d0bdb6cbe1c41eea26","object":"response","status":"in_progress","model":"gpt-4o-2024-08-06","output":[],"usage":null}}"#,
    "\n\n",
    "event: response.output_item.added\n",
    r#"data: {"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"id":"msg_0750331520328311006abeead0270c87d0b8128a81f8474fea","type":"message","status":"in_progress","role":"assistant","content":[]}}"#,
    "\n\n",
    "event: response.content_part.added\n",
    r#"data: {"type":"response.content_part.added","sequence_number":2,"item_id":"msg_0750331520328311006abeead0270c87d0b8128a81f8474fea","output_index":0,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}"#,
    "\n\n",
    "event: response.output_text.delta\n",
    r#"data: {"type":"response.output_text.delta","sequence_number":3,"item_id":"msg_0750331520328311006abeead0270c87d0b8128a81f8474fea","output_index":0,"content_index":0,"delta":"Hel"}"#,
    "\n\n",
    "event: response.output_text.delta\n",
    r#"data: {"type":"response.output_text.delta","sequence_number":4,"item_id":"msg_0750331520328311006abeead0270c87d0b8128a81f8474fea","output_index":0,"content_index":0,"delta":"lo."}"#,
    "\n\n",
    "event: response.output_item.done\n",
    r#"data: {"type":"response.output_item.done","sequence_number":5,"output_index":0,"item":{"id":"msg_0750331520328311006abeead0270c87d0b8128a81f8474fea","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Hello.","annotations":[]}]}}"#,
    "\n\n",
    "event: response.completed\n",
    r#"data: {"type":"response.completed","sequence_number":6,"response":{"id":"resp_0750331520328311006abeeacfb62c87d0bdb6cbe1c41eea26","object":"response","status":"completed","model":"gpt-4o-2024-08-06","output":[{"id":"msg_0750331520328311006abeead0270c87d0b8128a81f8474fea","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Hello.","annotations":[]}]}],"usage":{"input_tokens":5,"output_tokens":2,"total_tokens":7}}}"#,
    "\n\n",
);

async fn gpt_gateway(mode: Mode) -> (MockUpstream, Gateway, ed25519_dalek::SigningKey) {
    gpt_gateway_with(mode, |b| b).await
}

async fn gpt_gateway_with(
    mode: Mode,
    f: impl FnOnce(GatewayBuilder) -> GatewayBuilder,
) -> (MockUpstream, Gateway, ed25519_dalek::SigningKey) {
    let (pubkey, sk) = test_keypair(31);
    let mock = MockUpstream::start(mode).await;
    let gw = f(
        Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey)).providers(&[
            "openai",
            "openrouter",
            "anthropic",
        ]),
    )
    .start()
    .await;
    (mock, gw, sk)
}

async fn post_as(
    gw: &Gateway,
    sk: &ed25519_dalek::SigningKey,
    tenant: u64,
    path: &str,
    body: &Value,
) -> (u16, String) {
    let resp = test_client()
        .post(format!("{}{path}", gw.url()))
        .header(
            "authorization",
            format!("Bearer {}", billing_vkey(sk, tenant)),
        )
        .header("content-type", "application/json")
        .body(serde_json::to_vec(body).unwrap())
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    (status, resp.text().await.unwrap())
}

/// Tenant A's first turn: the signed ids it was given for `RESP`, `RS` and `MSG`.
async fn first_turn(gw: &Gateway, sk: &ed25519_dalek::SigningKey) -> (String, String, String) {
    let (status, text) = post_as(
        gw,
        sk,
        A,
        "/v1/responses",
        &json!({"model": "gpt-4o", "input": "hi"}),
    )
    .await;
    assert_eq!(status, 200, "{text}");
    let v: Value = serde_json::from_str(&text).unwrap();
    let id = |x: &Value| x.as_str().unwrap().to_owned();
    (
        id(&v["id"]),
        id(&v["output"][0]["id"]),
        id(&v["output"][1]["id"]),
    )
}

/// Every way a request can name stored state, with `id` in each position.
fn references(id: &str) -> Vec<(&'static str, Value)> {
    vec![
        (
            "previous_response_id",
            json!({"model": "gpt-4o", "input": "go on", "previous_response_id": id}),
        ),
        (
            "conversation",
            json!({"model": "gpt-4o", "input": "go on", "conversation": id}),
        ),
        (
            "conversation.id",
            json!({"model": "gpt-4o", "input": "go on", "conversation": {"id": id}}),
        ),
        (
            "input[1].id",
            json!({"model": "gpt-4o", "input": [
                {"role": "user", "content": "hi"},
                {"type": "item_reference", "id": id},
                {"role": "user", "content": "repeat that"},
            ]}),
        ),
    ]
}

fn assert_refused(status: u16, text: &str, field: &str) {
    assert_eq!(status, 400, "{field}: {text}");
    assert!(
        text.contains(field) && text.contains("does not belong to this tenant"),
        "{field}: {text}"
    );
    assert!(text.contains("invalid_request_error"), "{text}");
}

/// Tenant B holds tenant A's response and item ids (a log, a ticket, a client bug). Sent back in
/// any position, signed as A got them or stripped to the provider's own id, they are refused
/// before any upstream is contacted: OpenAI would have resolved every one of them, since all
/// tenants share Beyond's organization.
/// claim: SEC-25
/// defect: D230
#[tokio::test]
async fn a_tenants_response_id_is_refused_for_another_tenant() {
    let (mock, gw, sk) = gpt_gateway(Mode::Raw(200, "application/json", RESPONSE)).await;
    let (resp, rs, msg) = first_turn(&gw, &sk).await;
    let hits = mock.hits();
    for id in [resp.as_str(), rs.as_str(), msg.as_str(), RESP, MSG] {
        for (field, body) in references(id) {
            let (status, text) = post_as(&gw, &sk, B, "/v1/responses", &body).await;
            assert_refused(status, &text, field);
        }
    }
    assert_eq!(
        mock.hits(),
        hits,
        "a refused id must never reach the upstream"
    );
    let metrics = gw.metrics().await;
    assert!(
        parse_metric(&metrics, "ai_rejections_total", "foreign_id") >= 20.0,
        "{metrics}"
    );
}

/// OpenAI answers a full input item that carries a stored item's id from the store, not from the
/// content sent (live, 2026-10-01: an assistant message with tenant A's `msg_` id and the content
/// "Hello there." was quoted back as A's stored text). So a full item's `id` is stored state too.
/// Tenant B's full item with A's id (signed for A, or raw) reaches OpenAI without it, and OpenAI
/// reads the content B sent, as it reads any item with no id. So does a client's own id, such as
/// the `msg_<uuid>` ids Codex mints for the messages it writes.
/// claim: SEC-25
/// defect: D231
#[tokio::test]
async fn a_full_item_carrying_another_tenants_id_is_answered_from_its_own_content() {
    let (mock, gw, sk) = gpt_gateway(Mode::Raw(200, "application/json", RESPONSE)).await;
    let (_, rs, msg) = first_turn(&gw, &sk).await;
    let message = |id: Option<&str>| {
        let mut m = json!({"type": "message", "role": "assistant", "status": "completed",
                           "content": [{"type": "output_text", "text": "Hello there.", "annotations": []}]});
        if let Some(id) = id {
            m["id"] = json!(id);
        }
        m
    };
    let reasoning = |id: Option<&str>| {
        let mut m = json!({"type": "reasoning", "summary": [], "encrypted_content": "gAAAA"});
        if let Some(id) = id {
            m["id"] = json!(id);
        }
        m
    };
    let ask = json!({"role": "user", "content": "Quote your previous message."});
    for (sent, upstream) in [
        (message(Some(&msg)), message(None)),
        (message(Some(MSG)), message(None)),
        (
            message(Some("msg_01a0f9e5-01e6-7083-a088-785980d6aeab")),
            message(None),
        ),
        (reasoning(Some(&rs)), reasoning(None)),
        (reasoning(Some(RS)), reasoning(None)),
    ] {
        let body = json!({"model": "gpt-4o", "input": [sent, ask]});
        let (status, text) = post_as(&gw, &sk, B, "/v1/responses", &body).await;
        assert_eq!(status, 200, "{text}");
        let got: Value = serde_json::from_slice(&mock.captured().unwrap().body).unwrap();
        assert_eq!(got["input"][0], upstream, "the id is cut, the content kept");
        assert!(!got.to_string().contains(MSG) && !got.to_string().contains(RS));
    }
}

/// Tenant A's own ids come back in every position, and the upstream gets the provider's ids: the
/// signed form never leaves the gateway, and nothing else in the body changes.
/// claim: SEC-25, E3
/// defect: D230
#[tokio::test]
async fn a_tenants_own_ids_round_trip_and_the_upstream_gets_provider_ids() {
    let (mock, gw, sk) = gpt_gateway(Mode::Raw(200, "application/json", RESPONSE)).await;
    let (resp, rs, msg) = first_turn(&gw, &sk).await;
    assert_ne!(resp, RESP, "the client never sees the provider id");
    assert!(resp.starts_with("resp_") && rs.starts_with("rs_") && msg.starts_with("msg_"));
    assert!(resp.len() <= 64 && rs.len() <= 64 && msg.len() <= 64);
    for (signed, raw) in [(&resp, RESP), (&rs, RS), (&msg, MSG)] {
        for ((field, client), (_, upstream)) in references(signed).into_iter().zip(references(raw))
        {
            let (status, text) = post_as(&gw, &sk, A, "/v1/responses", &client).await;
            assert_eq!(status, 200, "{field}: {text}");
            let cap = mock.captured().unwrap();
            assert_eq!(cap.path, "/v1/responses");
            assert_eq!(
                String::from_utf8(cap.body).unwrap(),
                serde_json::to_string(&upstream).unwrap(),
                "{field}: the upstream gets the body with its own id"
            );
        }
    }
    // A full item (the AI SDK with `store: false`, openai-python's `input += response.output`).
    let full = |id: &str| {
        json!({"model": "gpt-4o", "store": false, "input": [
            {"role": "user", "content": "hi"},
            {"type": "message", "role": "assistant", "id": id, "status": "completed",
             "content": [{"type": "output_text", "text": "Hello.", "annotations": []}]},
            {"role": "user", "content": "and?"},
        ]})
    };
    let (status, text) = post_as(&gw, &sk, A, "/v1/responses", &full(&msg)).await;
    assert_eq!(status, 200, "{text}");
    assert_eq!(
        String::from_utf8(mock.captured().unwrap().body).unwrap(),
        serde_json::to_string(&full(MSG)).unwrap()
    );
    // Another of the same tenant's keys: ids belong to the tenant, not the key.
    let other_key = beyond_ai::key::mint(
        &beyond_ai::key::VirtualKey {
            tenant_id: A,
            vpc_id: 9,
            key_id: Some(77),
        },
        1,
        &sk,
    );
    let resp2 = test_client()
        .post(format!("{}/v1/responses", gw.url()))
        .header("authorization", format!("Bearer {other_key}"))
        .header("content-type", "application/json")
        .body(
            serde_json::to_vec(
                &json!({"model": "gpt-4o", "input": "go on", "previous_response_id": resp}),
            )
            .unwrap(),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(resp2.status().as_u16(), 200);
}

/// A non-streaming answer's ids are all signed for the tenant it went to, consistently: the same
/// provider id is the same signed id on every response, and verifies only for that tenant.
/// claim: SEC-25
#[tokio::test]
async fn a_json_response_has_every_id_signed() {
    let (_mock, gw, sk) = gpt_gateway(Mode::Raw(200, "application/json", RESPONSE)).await;
    let (status, text) = post_as(
        &gw,
        &sk,
        A,
        "/v1/responses",
        &json!({"model": "gpt-4o", "input": "hi"}),
    )
    .await;
    assert_eq!(status, 200, "{text}");
    for raw in [RESP, RS, MSG] {
        assert!(
            !text.contains(raw),
            "a provider id reached the client: {text}"
        );
    }
    let signer = dev_id_signer();
    let v: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(
        signer.verify(A, v["id"].as_str().unwrap()).as_deref(),
        Some(RESP)
    );
    assert_eq!(
        signer
            .verify(A, v["output"][0]["id"].as_str().unwrap())
            .as_deref(),
        Some(RS)
    );
    assert_eq!(
        signer
            .verify(A, v["output"][1]["id"].as_str().unwrap())
            .as_deref(),
        Some(MSG)
    );
    assert_eq!(signer.verify(B, v["id"].as_str().unwrap()), None);
    // Everything but the ids is as the provider sent it.
    let mut want: Value = serde_json::from_str(RESPONSE).unwrap();
    want["id"] = v["id"].clone();
    want["output"][0]["id"] = v["output"][0]["id"].clone();
    want["output"][1]["id"] = v["output"][1]["id"].clone();
    assert_eq!(v, want);
    // The same provider id signs the same way next time.
    let (_, again) = post_as(
        &gw,
        &sk,
        A,
        "/v1/responses",
        &json!({"model": "gpt-4o", "input": "hi"}),
    )
    .await;
    assert_eq!(
        serde_json::from_str::<Value>(&again).unwrap()["id"],
        v["id"]
    );
}

/// A streamed answer signs every id on every event that carries one, and the same item is the same
/// signed id on all of them (`item.id` on `output_item.*`, `item_id` on each content event, the
/// output on `completed`), so a client that keys items by id still joins them.
/// claim: SEC-25
#[tokio::test]
async fn a_streamed_response_signs_every_id_consistently() {
    let (_mock, gw, sk) = gpt_gateway(Mode::Raw(200, "text/event-stream", STREAM)).await;
    let (status, text) = post_as(
        &gw,
        &sk,
        A,
        "/v1/responses",
        &json!({"model": "gpt-4o", "input": "hi", "stream": true}),
    )
    .await;
    assert_eq!(status, 200, "{text}");
    assert!(!text.contains(RESP) && !text.contains(MSG), "{text}");
    let signer = dev_id_signer();
    let (resp, msg) = (signer.sign(A, RESP), signer.sign(A, MSG));
    assert_eq!(text, STREAM.replace(RESP, &resp).replace(MSG, &msg));
    let mut item_ids = 0;
    for line in text.lines().filter_map(|l| l.strip_prefix("data: ")) {
        let ev: Value = serde_json::from_str(line).unwrap();
        for id in [
            &ev["item_id"],
            &ev["item"]["id"],
            &ev["response"]["output"][0]["id"],
        ] {
            if let Some(id) = id.as_str() {
                assert_eq!(id, msg, "{line}");
                item_ids += 1;
            }
        }
        if let Some(id) = ev["response"]["id"].as_str() {
            assert_eq!(id, resp);
        }
    }
    assert_eq!(item_ids, 6);
}

/// A BYO key is the caller's own OpenAI organization: its ids are relayed and accepted as sent.
/// claim: SEC-25, SEC-10
#[tokio::test]
async fn byo_responses_ids_are_untouched() {
    let (mock, gw, _sk) = gpt_gateway(Mode::Raw(200, "application/json", RESPONSE)).await;
    let body = json!({"model": "gpt-4o", "previous_response_id": RESP,
                      "input": [{"type": "item_reference", "id": MSG}]});
    let resp = test_client()
        .post(format!("{}/v1/responses", gw.url()))
        .header("authorization", "Bearer sk-byo-own-org")
        .header("content-type", "application/json")
        .body(serde_json::to_vec(&body).unwrap())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    assert_eq!(
        resp.text().await.unwrap(),
        RESPONSE,
        "relayed byte for byte"
    );
    let cap = mock.captured().unwrap();
    assert_eq!(cap.authorization.as_deref(), Some("Bearer sk-byo-own-org"));
    assert_eq!(cap.body, serde_json::to_vec(&body).unwrap());
}

/// Rotation: with a new key signing and the previous one still listed, ids issued before the
/// rotation keep verifying, and new ids carry the new kid. Once the previous key is removed its
/// ids are refused.
/// claim: SEC-25
#[tokio::test]
async fn ids_issued_before_a_rotation_still_verify() {
    let old = dev_id_signer();
    let issued = old.sign(A, RESP);
    let new_secret = [9u8; 32];
    let (mock, gw, sk) = gpt_gateway_with(Mode::Raw(200, "application/json", RESPONSE), |b| {
        b.id_signing_keys(&[('1', &DEV_ID_SECRET), ('2', &new_secret)], '2')
    })
    .await;
    let body = |id: &str| json!({"model": "gpt-4o", "input": "go on", "previous_response_id": id});
    let (status, text) = post_as(&gw, &sk, A, "/v1/responses", &body(&issued)).await;
    assert_eq!(status, 200, "{text}");
    assert_eq!(
        String::from_utf8(mock.captured().unwrap().body).unwrap(),
        serde_json::to_string(&body(RESP)).unwrap()
    );
    let id = serde_json::from_str::<Value>(&text).unwrap()["id"]
        .as_str()
        .unwrap()
        .to_owned();
    assert!(id.starts_with("resp_x2"), "new ids carry the new kid: {id}");
    let rotated = beyond_ai::signed_id::Signer::new(&[(b'2', &new_secret)], b'2').unwrap();
    assert_eq!(rotated.verify(A, &id).as_deref(), Some(RESP));

    let (mock2, retired, sk2) =
        gpt_gateway_with(Mode::Raw(200, "application/json", RESPONSE), |b| {
            b.id_signing_keys(&[('2', &new_secret)], '2')
        })
        .await;
    let _ = sk;
    let (status, text) = post_as(&retired, &sk2, A, "/v1/responses", &body(&issued)).await;
    assert_refused(status, &text, "previous_response_id");
    assert_eq!(mock2.hits(), 0);
}

/// Any change to a signed id (one character, a truncation, the tail of another id) is refused.
/// claim: SEC-25
#[tokio::test]
async fn a_tampered_id_is_refused() {
    let (mock, gw, sk) = gpt_gateway(Mode::Raw(200, "application/json", RESPONSE)).await;
    let (resp, _, msg) = first_turn(&gw, &sk).await;
    let hits = mock.hits();
    let flip = |s: &str, i: usize| {
        let mut b = s.as_bytes().to_vec();
        b[i] = if b[i] == b'A' { b'B' } else { b'A' };
        String::from_utf8(b).unwrap()
    };
    let spliced = format!("{}{}", &resp[..20], &msg[19..]);
    for bad in [
        flip(&resp, resp.len() - 1),
        flip(&resp, 10),
        flip(&resp, 6),
        resp[..resp.len() - 2].to_owned(),
        format!("{resp}A"),
        spliced,
    ] {
        let (status, text) = post_as(
            &gw,
            &sk,
            A,
            "/v1/responses",
            &json!({"model": "gpt-4o", "input": "go on", "previous_response_id": bad}),
        )
        .await;
        assert_refused(status, &text, "previous_response_id");
    }
    assert_eq!(mock.hits(), hits);
}

/// No id signing key on a deployment that serves managed traffic (it has `signing_keys`): the
/// gateway refuses to boot, naming the missing key, rather than handing out Responses ids every
/// tenant could resolve. A BYO-only deployment (no `signing_keys`) stores nothing on Beyond's
/// accounts and boots without one.
/// claim: SEC-25
#[tokio::test]
async fn a_missing_id_signing_key_refuses_to_boot() {
    let (pubkey, _sk) = test_keypair(1);
    let managed = format!("[signing_keys]\n1 = \"{}\"", b64(&pubkey));
    let err = boot_refuses(&managed, Duration::from_secs(10))
        .expect("a managed deployment without id_signing_keys must not boot");
    assert!(
        err.contains("id_signing_keys"),
        "names the missing key: {err}"
    );
    assert!(
        boot_refuses("", Duration::from_secs(3)).is_none(),
        "a BYO-only deployment boots without an id signing key"
    );
}

/// The managed `/{provider}` escape hatch reaches the same store, so it gets the same binding: the
/// answer's ids are signed, the tenant's own ids are stripped for the upstream, and another
/// tenant's id is refused before a byte of the body leaves the gateway.
/// claim: SEC-25, SEC-1
/// defect: D230
#[tokio::test]
async fn the_provider_route_is_bound_too() {
    let (mock, gw, sk) = gpt_gateway(Mode::Raw(200, "application/json", RESPONSE)).await;
    let (status, text) = post_as(
        &gw,
        &sk,
        A,
        "/openai/v1/responses",
        &json!({"model": "gpt-4o", "input": "hi"}),
    )
    .await;
    assert_eq!(status, 200, "{text}");
    let resp = serde_json::from_str::<Value>(&text).unwrap()["id"]
        .as_str()
        .unwrap()
        .to_owned();
    assert_eq!(dev_id_signer().verify(A, &resp).as_deref(), Some(RESP));

    let body = |id: &str| json!({"model": "gpt-4o", "input": "go on", "previous_response_id": id});
    let (status, text) = post_as(&gw, &sk, A, "/openai/v1/responses", &body(&resp)).await;
    assert_eq!(status, 200, "{text}");
    let cap = mock.captured().unwrap();
    assert_eq!(cap.body, serde_json::to_vec(&body(RESP)).unwrap());

    for id in [resp.as_str(), RESP] {
        let tenant = if id == RESP { A } else { B };
        let (status, text) = post_as(&gw, &sk, tenant, "/openai/v1/responses", &body(id)).await;
        assert_eq!(status, 400, "{text}");
        assert!(text.contains("does not belong to this tenant"), "{text}");
        // The body streams on this route, so its headers left before it was read; not one byte
        // of it did (the mock saw an empty, aborted request).
        assert!(
            mock.captured().unwrap().body.is_empty(),
            "a refused body reached the upstream"
        );
    }
}

/// A large body (past the 64 KiB replay buffer, relayed as a `FullBody`) is checked before its
/// first attempt connects, like any other catalog walk.
/// claim: SEC-25
/// defect: D230
#[tokio::test]
async fn a_large_body_with_another_tenants_id_is_refused_before_connecting() {
    let (mock, gw, sk) = gpt_gateway(Mode::Raw(200, "application/json", RESPONSE)).await;
    let (resp, _, _) = first_turn(&gw, &sk).await;
    let hits = mock.hits();
    let filler = "y".repeat(150 * 1024);
    let body = |id: &str| json!({"model": "gpt-4o", "input": filler, "previous_response_id": id});
    let (status, text) = post_as(&gw, &sk, B, "/v1/responses", &body(&resp)).await;
    assert_refused(status, &text, "previous_response_id");
    assert_eq!(mock.hits(), hits);
    let (status, text) = post_as(&gw, &sk, A, "/v1/responses", &body(&resp)).await;
    assert_eq!(status, 200, "{text}");
    assert_eq!(
        mock.captured().unwrap().body,
        serde_json::to_vec(&body(RESP)).unwrap()
    );
}

/// Codex's remote compaction (`/v1/responses/compact`) returns items it may later resolve: their
/// ids are signed, and its input is checked like a turn's.
/// claim: SEC-25
#[tokio::test]
async fn compaction_ids_are_bound_too() {
    let (mock, gw, sk) = gpt_gateway(Mode::Raw(200, "application/json", RESPONSE)).await;
    let (status, text) = post_as(
        &gw,
        &sk,
        A,
        "/v1/responses/compact",
        &json!({"model": "gpt-4o", "input": "hi"}),
    )
    .await;
    assert_eq!(status, 200, "{text}");
    let v: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(
        dev_id_signer()
            .verify(A, v["output"][1]["id"].as_str().unwrap())
            .as_deref(),
        Some(MSG)
    );
    let hits = mock.hits();
    let (status, text) = post_as(
        &gw,
        &sk,
        B,
        "/v1/responses/compact",
        &json!({"model": "gpt-4o", "input": [{"type": "item_reference", "id": v["output"][1]["id"]}]}),
    )
    .await;
    assert_refused(status, &text, "input[0].id");
    assert_eq!(mock.hits(), hits);
}

/// A row with no store is untouched: a Claude row's translated answer carries ids nothing upstream
/// can resolve and is not signed, a grok row's relay to xAI still goes as `store: false` (D145),
/// and D175's rule still drops a tool step's `item_reference` and refuses a turn's, whatever form
/// its id takes.
/// claim: SEC-25, E3
/// defect: D175
#[tokio::test]
async fn rows_without_a_store_are_untouched() {
    let (pubkey, sk) = test_keypair(32);
    let claude = MockUpstream::start(Mode::AnthropicJson).await;
    let gw = Gateway::builder(unused_nats_port(), &claude.authority(), &b64(&pubkey))
        .providers(&["anthropic", "openrouter"])
        .start()
        .await;
    let (status, text) = post_as(
        &gw,
        &sk,
        A,
        "/v1/responses",
        &json!({"model": "claude-opus-4-8", "input": "hi"}),
    )
    .await;
    assert_eq!(status, 200, "{text}");
    let v: Value = serde_json::from_str(&text).unwrap();
    let id = v["id"].as_str().unwrap();
    assert_eq!(
        dev_id_signer().verify(A, id),
        None,
        "translated ids are not signed: {id}"
    );
    let signed = dev_id_signer().sign(A, MSG);
    let tool_step = json!({"model": "claude-opus-4-8", "input": [
        {"role": "user", "content": "weather?"},
        {"type": "item_reference", "id": signed},
        {"type": "function_call", "call_id": "c1", "name": "weather", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "sunny"},
    ], "tools": [{"type": "function", "name": "weather", "parameters": {"type": "object", "properties": {}}}]});
    let (status, text) = post_as(&gw, &sk, A, "/v1/responses", &tool_step).await;
    assert_eq!(status, 200, "a tool step's reference is dropped: {text}");
    let turn = json!({"model": "claude-opus-4-8", "input": [
        {"role": "user", "content": "hi"},
        {"type": "item_reference", "id": signed},
        {"role": "user", "content": "repeat that"},
    ]});
    let (status, text) = post_as(&gw, &sk, A, "/v1/responses", &turn).await;
    assert_eq!(status, 400, "{text}");
    assert!(
        text.contains("item_reference") && text.contains("store: false"),
        "{text}"
    );

    let xai = MockUpstream::start(Mode::Raw(200, "application/json", RESPONSE)).await;
    let gw = Gateway::builder(unused_nats_port(), &xai.authority(), &b64(&pubkey))
        .providers(&["xai"])
        .start()
        .await;
    let (status, text) = post_as(
        &gw,
        &sk,
        A,
        "/v1/responses",
        &json!({"model": "grok-4.3", "input": "hi"}),
    )
    .await;
    assert_eq!(status, 200, "{text}");
    assert_eq!(
        text, RESPONSE,
        "xAI keeps nothing (store: false), so nothing is signed"
    );
    let sent: Value = serde_json::from_slice(&xai.captured().unwrap().body).unwrap();
    assert_eq!(sent["store"], false);
}
