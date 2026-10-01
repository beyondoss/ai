//! Property tests for `translate::request` and `translate::response_json_status`.
//!
//! Properties:
//! - never panics on any JSON (or any bytes); a translated body is always a JSON object, bounded
//!   by a constant factor of its input;
//! - a request translated onto another dialect, and back, still carries every text the model
//!   reads, every base64 payload, every offered tool and every tool call with its arguments
//!   (compared as JSON values), whatever the target model;
//! - a non-stream response reaches a client of another dialect with its text, tool calls, signed
//!   thinking and usage intact, and its usage equals what `usage.rs` bills for the upstream body;
//! - any non-2xx body becomes an error in the client's own envelope.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod props_support;

use beyond_ai::route::Endpoint;
use beyond_ai::translate;
use beyond_ai::usage;
use props_support::requests::{
    Carried, MODELS, chat_request, check_carried, messages_request, responses_request,
};
use props_support::script::{DIALECTS, Script, Seen, script, seen_body};
use props_support::*;
use proptest::prelude::*;
use proptest::test_runner::TestCaseResult;
use serde_json::{Value, json};

const GATEWAY_SIGNATURE_PREFIX: &str = "rs_gw:";

fn bounded(input: usize, output: usize) -> bool {
    output <= input.saturating_mul(8) + 16 * 1024
}

fn model() -> impl Strategy<Value = &'static str> {
    prop::sample::select(MODELS)
}

/// A (client, upstream) pair that translates.
fn cross() -> impl Strategy<Value = (Endpoint, Endpoint)> {
    prop::sample::select(DIALECTS.to_vec()).prop_flat_map(|c| (Just(c), other(c)))
}

fn other(d: Endpoint) -> impl Strategy<Value = Endpoint> {
    prop::sample::select(
        DIALECTS
            .iter()
            .copied()
            .filter(|x| *x != d)
            .collect::<Vec<_>>(),
    )
}

/// Translate and parse, failing the case on a non-object or oversized result.
fn translate_ok(
    from: Endpoint,
    to: Endpoint,
    body: &Value,
    m: &str,
) -> Result<Value, TestCaseError> {
    let bytes = serde_json::to_vec(body).unwrap();
    let out = translate::request(from, to, &bytes, m);
    prop_assert!(
        bounded(bytes.len(), out.len()),
        "{} bytes in, {} out",
        bytes.len(),
        out.len()
    );
    match serde_json::from_slice::<Value>(&out) {
        Ok(v) if v.is_object() => Ok(v),
        other => Err(TestCaseError::fail(format!(
            "{from:?} -> {to:?} gave {other:?}: {}",
            String::from_utf8_lossy(&out)
        ))),
    }
}

fn carry(from: Endpoint, to: Endpoint, body: &Value, c: &Carried, m: &str) -> TestCaseResult {
    let out = translate_ok(from, to, body, m)?;
    if let Err(e) = check_carried(c, &out) {
        prop_assert!(
            false,
            "{from:?} -> {to:?} on {m:?}: {e}\nin: {body}\nout: {out}"
        );
    }
    // And back again: the composition must carry it too.
    let back = translate_ok(to, from, &out, m)?;
    if let Err(e) = check_carried(c, &back) {
        prop_assert!(
            false,
            "{from:?} -> {to:?} -> {from:?} on {m:?}: {e}\nin: {body}\nmid: {out}\nback: {back}"
        );
    }
    Ok(())
}

fn billed_body(d: Endpoint, body: &[u8]) -> Option<usage::Usage> {
    match d {
        Endpoint::Messages => usage::anthropic_body(body),
        _ => usage::openai_body(body),
    }
}

fn check_content(
    s: &Script,
    client: Endpoint,
    upstream: Endpoint,
    seen: &Seen,
) -> Result<(), String> {
    if seen.text != s.text() {
        return Err(format!(
            "text: client saw {:?}, upstream sent {:?}",
            seen.text,
            s.text()
        ));
    }
    let want = s.tools();
    let got: Vec<(String, Option<Value>)> = seen
        .tools
        .iter()
        .map(|(n, a)| (n.clone(), serde_json::from_str(a).ok()))
        .collect();
    let want: Vec<(String, Option<Value>)> = want.into_iter().map(|(n, a)| (n, Some(a))).collect();
    if got != want {
        return Err(format!(
            "tool calls: client saw {:?}, upstream sent {want:?}",
            seen.tools
        ));
    }
    let signed = s.replayable_thinking();
    match (upstream, client) {
        (Endpoint::Messages, Endpoint::ChatCompletions)
        | (Endpoint::ChatCompletions, Endpoint::Messages) => {
            // A Chat client's `thinking` list holds the blocks exactly as Anthropic sent them,
            // unsigned ones included (their text is also in `reasoning_content`); a Messages
            // client only ever gets the replayable ones.
            let replayable: Vec<Value> = seen
                .thinking
                .iter()
                .filter(|b| {
                    b["type"] == "redacted_thinking"
                        || b["signature"].as_str().is_some_and(|s| !s.is_empty())
                })
                .cloned()
                .collect();
            if replayable != signed {
                return Err(format!(
                    "thinking: client saw {:?}, upstream sent {signed:?}",
                    seen.thinking
                ));
            }
        }
        (Endpoint::Messages | Endpoint::ChatCompletions, Endpoint::Responses) => {
            let want: Vec<String> = signed
                .iter()
                .filter_map(|b| b["signature"].as_str())
                .map(|sig| format!("{GATEWAY_SIGNATURE_PREFIX}{sig}"))
                .collect();
            if seen.encrypted != want {
                return Err(format!(
                    "signed reasoning: client saw {:?}, want {want:?}",
                    seen.encrypted
                ));
            }
        }
        _ => {}
    }
    Ok(())
}

/// Request-shaped garbage: the dialects' request keys with arbitrary values.
fn request_vocab() -> impl Strategy<Value = Value> {
    const REQ_KEYS: &[&str] = &[
        "model",
        "messages",
        "input",
        "instructions",
        "system",
        "tools",
        "tool_choice",
        "max_tokens",
        "max_completion_tokens",
        "max_output_tokens",
        "stream",
        "stream_options",
        "temperature",
        "top_p",
        "stop",
        "stop_sequences",
        "reasoning",
        "reasoning_effort",
        "thinking",
        "response_format",
        "text",
        "output_config",
        "metadata",
        "user",
        "functions",
        "function_call",
        "parallel_tool_calls",
        "store",
        "previous_response_id",
        "n",
        "logprobs",
        "role",
        "content",
        "tool_calls",
        "tool_call_id",
        "type",
        "name",
        "arguments",
        "call_id",
        "output",
        "image_url",
        "url",
        "source",
        "data",
        "cache_control",
        "id",
        "input_schema",
        "parameters",
        "function",
        "strict",
        "signature",
        "text",
        "file",
        "file_data",
    ];
    prop::collection::vec((prop::sample::select(REQ_KEYS), vocab_value(4)), 0..10)
        .prop_map(|kv| Value::Object(kv.into_iter().map(|(k, v)| (k.to_owned(), v)).collect()))
}

#[test]
fn prop_chat_requests_carry_their_content_onto_every_dialect() {
    check(
        "prop_chat_requests_carry_their_content_onto_every_dialect",
        (chat_request(), other(Endpoint::ChatCompletions), model()),
        |(r, to, m)| carry(Endpoint::ChatCompletions, to, &r.body, &r.carried, m),
    );
}

#[test]
fn prop_messages_requests_carry_their_content_onto_every_dialect() {
    check(
        "prop_messages_requests_carry_their_content_onto_every_dialect",
        (messages_request(), other(Endpoint::Messages), model()),
        |(r, to, m)| carry(Endpoint::Messages, to, &r.body, &r.carried, m),
    );
}

#[test]
fn prop_responses_requests_carry_their_content_onto_every_dialect() {
    check(
        "prop_responses_requests_carry_their_content_onto_every_dialect",
        (responses_request(), other(Endpoint::Responses), model()),
        |(r, to, m)| carry(Endpoint::Responses, to, &r.body, &r.carried, m),
    );
}

/// Any request-shaped JSON, every pairing, every model: no panic, an object out, bounded.
#[test]
fn prop_any_request_json_translates_to_a_bounded_object() {
    check(
        "prop_any_request_json_translates_to_a_bounded_object",
        (
            request_vocab(),
            prop::sample::select(DIALECTS.to_vec()),
            prop::sample::select(DIALECTS.to_vec()),
            model(),
        ),
        |(body, from, to, m)| {
            let out = translate_ok(from, to, &body, m)?;
            // Translating the translation again must not panic either.
            let _ = translate_ok(to, from, &out, m)?;
            Ok(())
        },
    );
}

/// Any bytes: no panic, and bytes that are not a JSON object pass through unchanged.
#[test]
fn prop_any_bytes_translate_without_panicking() {
    check(
        "prop_any_bytes_translate_without_panicking",
        (
            prop::collection::vec(any::<u8>(), 0..256),
            prop::sample::select(DIALECTS.to_vec()),
            prop::sample::select(DIALECTS.to_vec()),
            any::<u16>(),
        ),
        |(bytes, a, b, status)| {
            let out = translate::request(a, b, &bytes, "claude-opus-4-8");
            if !matches!(
                serde_json::from_slice::<Value>(&bytes),
                Ok(Value::Object(_))
            ) {
                prop_assert_eq!(&out, &bytes);
            }
            let _ = translate::response_json_status(a, b, status, &bytes);
            Ok(())
        },
    );
}

/// A complete non-stream response reaches a client of another dialect with its content and
/// usage intact; the usage shown equals what `usage.rs` bills for the upstream body.
#[test]
fn prop_translated_bodies_preserve_content_and_usage() {
    check(
        "prop_translated_bodies_preserve_content_and_usage",
        (cross(), script()),
        |((client, upstream), s)| {
            let body = s.body(upstream);
            let out = translate::response_json(upstream, client, &body);
            prop_assert!(bounded(body.len(), out.len()));
            let seen = match seen_body(client, &out) {
                Ok(seen) => seen,
                Err(e) => return Err(TestCaseError::fail(e)),
            };
            prop_assert!(
                !seen.error,
                "a success body became an error: {}",
                String::from_utf8_lossy(&out)
            );
            if let Err(e) = check_content(&s, client, upstream, &seen) {
                prop_assert!(
                    false,
                    "{e}\nin: {}\nout: {}",
                    String::from_utf8_lossy(&body),
                    String::from_utf8_lossy(&out)
                );
            }
            let up = billed_body(upstream, &body).map(|u| normalize(&u));
            let shown = billed_body(client, &out).map(|u| normalize(&u));
            prop_assert!(up.is_some());
            prop_assert_eq!(
                shown,
                up,
                "client was shown other usage than is billed\nout: {}",
                String::from_utf8_lossy(&out)
            );
            Ok(())
        },
    );
}

/// Every non-2xx JSON body is an error in the client's own envelope, with a message.
#[test]
fn prop_error_bodies_arrive_in_the_client_envelope() {
    check(
        "prop_error_bodies_arrive_in_the_client_envelope",
        (cross(), 400u16..600, vocab_value(4)),
        |((client, upstream), status, body)| {
            let bytes = serde_json::to_vec(&body).unwrap();
            let out = translate::response_json_status(upstream, client, status, &bytes);
            prop_assert!(bounded(bytes.len(), out.len()));
            let v: Value = serde_json::from_slice(&out).map_err(|e| {
                TestCaseError::fail(format!("not JSON ({e}): {}", String::from_utf8_lossy(&out)))
            })?;
            let err = &v["error"];
            prop_assert!(err.is_object(), "no error object: {v}");
            prop_assert!(err["message"].is_string(), "error without a message: {v}");
            prop_assert!(err["type"].is_string(), "error without a type: {v}");
            if client == Endpoint::Messages {
                prop_assert_eq!(&v["type"], &json!("error"));
            }
            Ok(())
        },
    );
}
