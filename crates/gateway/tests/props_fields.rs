//! Property test: no request field that changes what the client gets back is silently dropped.
//!
//! translate.rs's contract: hints with no equivalent and no effect on the response's shape may be
//! dropped (`seed`, penalties, `logit_bias`, …); everything that changes the answer either maps
//! onto the target or is forwarded for the provider to reject by name; Responses session state is
//! never stripped (it relays to a Responses arm or is refused). Here each such field, with a
//! realistic value, is added to an otherwise random request: translating with and without it must
//! give different bodies (it mapped, or it was forwarded), or the field must be session state the
//! catalog walk never translates.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod props_support;

use beyond_ai::route::Endpoint;
use beyond_ai::translate;
use props_support::requests::{chat_request, messages_request, responses_request};
use props_support::*;
use proptest::prelude::*;
use serde_json::{Value, json};

/// Fields of each dialect that change the response, with a value a client really sends.
fn answer_fields(d: Endpoint) -> Vec<(&'static str, Value)> {
    match d {
        Endpoint::ChatCompletions => vec![
            ("max_completion_tokens", json!(777)),
            ("stop", json!(["STOP-HERE"])),
            ("n", json!(2)),
            ("logprobs", json!(true)),
            ("modalities", json!(["text", "audio"])),
            ("audio", json!({ "voice": "alloy", "format": "wav" })),
            (
                "web_search_options",
                json!({ "search_context_size": "low" }),
            ),
            ("reasoning_effort", json!("high")),
            (
                "response_format",
                json!({ "type": "json_schema", "json_schema": { "name": "out", "schema": { "type": "object", "properties": { "a": { "type": "string" } } } } }),
            ),
            ("tool_choice", json!("required")),
            ("parallel_tool_calls", json!(false)),
        ],
        Endpoint::Messages => vec![
            ("stop_sequences", json!(["STOP-HERE"])),
            (
                "thinking",
                json!({ "type": "enabled", "budget_tokens": 3000 }),
            ),
            ("tool_choice", json!({ "type": "any" })),
            (
                "mcp_servers",
                json!([{ "type": "url", "url": "https://mcp.example.com", "name": "ex" }]),
            ),
            (
                "output_config",
                json!({ "format": { "type": "json_schema", "schema": { "type": "object", "properties": { "a": { "type": "string" } } } } }),
            ),
        ],
        _ => vec![
            ("instructions", json!("Answer in French.")),
            ("max_output_tokens", json!(777)),
            ("reasoning", json!({ "effort": "high" })),
            (
                "text",
                json!({ "format": { "type": "json_schema", "name": "out", "schema": { "type": "object", "properties": { "a": { "type": "string" } } } } }),
            ),
            ("tool_choice", json!("required")),
            ("parallel_tool_calls", json!(false)),
            (
                "prompt",
                json!({ "id": "pmpt_abc123", "variables": { "city": "Paris" } }),
            ),
            ("top_logprobs", json!(5)),
            ("conversation", json!("conv_abc123")),
            ("previous_response_id", json!("resp_abc123")),
        ],
    }
}

/// The upstream model each target is exercised with: one that takes reasoning and every option.
fn model_for(to: Endpoint) -> &'static str {
    match to {
        Endpoint::Messages => "claude-sonnet-4-5",
        _ => "gpt-5",
    }
}

/// Fields that may legitimately leave no trace in this request, and why (each a documented
/// outcome).
fn excused(body: &Value, field: &str) -> bool {
    let has = |k: &str| body.get(k).is_some_and(|v| !v.is_null());
    match field {
        // A tool choice or parallelism with no tools offered is dropped onto Chat Completions,
        // which 400s it (documented).
        "tool_choice" | "parallel_tool_calls"
            if body
                .get("tools")
                .and_then(Value::as_array)
                .is_none_or(Vec::is_empty) =>
        {
            true
        }
        // Reasoning effort is model-aware: a request that already sets one, or forces a tool,
        // may keep the other setting; thinking is turned off for a conversation whose tool turns
        // carry no thinking to replay (TRN-7); and a budget-thinking model gets none when the
        // output limit is 1024 or less (the budget must be at least 1024 and below it). All
        // documented.
        "thinking" | "reasoning" | "reasoning_effort" => {
            let text = body.to_string();
            ["thinking", "reasoning", "reasoning_effort", "tool_choice"]
                .iter()
                .any(|k| *k != field && has(k))
                || ["function_call", "tool_calls", "tool_use"]
                    .iter()
                    .any(|k| text.contains(k))
                || ["max_tokens", "max_completion_tokens", "max_output_tokens"]
                    .iter()
                    .any(|k| {
                        body.get(*k)
                            .and_then(Value::as_u64)
                            .is_some_and(|n| n <= 1024)
                    })
        }
        // Parallelism means nothing once tools are switched off.
        "parallel_tool_calls" if body.get("tool_choice") == Some(&json!("none")) => true,
        // Two spellings of one limit: with both set, one of them wins.
        "max_completion_tokens" => has("max_tokens"),
        _ => false,
    }
}

fn request_of(d: Endpoint) -> BoxedStrategy<Value> {
    match d {
        Endpoint::ChatCompletions => chat_request().prop_map(|r| r.body).boxed(),
        Endpoint::Messages => messages_request().prop_map(|r| r.body).boxed(),
        _ => responses_request()
            .prop_map(|r| {
                let mut b = r.body;
                // A one-shot (the only kind that is translated), unless a case adds session state.
                b["store"] = json!(false);
                b
            })
            .boxed(),
    }
}

fn case() -> impl Strategy<Value = (Endpoint, Endpoint, Value, usize)> {
    prop::sample::select(props_support::script::DIALECTS.to_vec()).prop_flat_map(|from| {
        let to = prop::sample::select(
            props_support::script::DIALECTS
                .iter()
                .copied()
                .filter(|d| *d != from)
                .collect::<Vec<_>>(),
        );
        (
            Just(from),
            to,
            request_of(from),
            0..answer_fields(from).len(),
        )
    })
}

#[test]
fn prop_answer_changing_fields_are_never_silently_dropped() {
    check(
        "prop_answer_changing_fields_are_never_silently_dropped",
        case(),
        |(from, to, body, which)| {
            let (field, value) = answer_fields(from)[which].clone();
            prop_assume!(!excused(&body, field));
            let mut without = body.clone();
            if let Some(m) = without.as_object_mut() {
                m.remove(field);
            }
            let mut with = without.clone();
            with[field] = value;
            let bytes = serde_json::to_vec(&with).unwrap();
            // Session state is not translated at all: the walk relays it to a Responses arm or
            // refuses it, so dropping it here would be unreachable.
            if from == Endpoint::Responses
                && translate::responses_session_field(&bytes, false).is_some()
            {
                return Ok(());
            }
            let m = model_for(to);
            let a = translate::request(from, to, &bytes, m);
            let b = translate::request(from, to, &serde_json::to_vec(&without).unwrap(), m);
            let (a, b): (Value, Value) = (
                serde_json::from_slice(&a).unwrap(),
                serde_json::from_slice(&b).unwrap(),
            );
            prop_assert!(
                a != b,
                "{from:?} -> {to:?}: `{field}` vanished without a trace (neither mapped nor forwarded, and not session state)\nwith it: {}",
                serde_json::to_string(&with).unwrap()
            );
            Ok(())
        },
    );
}
