//! Deterministic regressions for the defects the property suites (`props_*.rs`) found, each the
//! minimized counterexample its property shrank to. Each asserts the correct behavior and stays
//! `#[ignore]`d while its defect is open (see verify/README.md).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use beyond_ai::route::Endpoint;
use beyond_ai::translate::{self, SseBridge};
use beyond_ai::usage::{self, InputTally};
use serde_json::Value;

/// An OpenAI-shaped usage block whose `prompt + completion + reasoning` passes `u64::MAX` is a
/// count no parser may panic on: the reasoning-outside-completion check sums the three unchecked,
/// and the release profile has `overflow-checks = true`. Shrunk from
/// `prop_translated_streams_preserve_content_and_usage`.
/// claim: BIL-4, REL-17
/// defect: D87
#[test]
fn usage_with_boundary_counts_never_panics() {
    let body = br#"{"usage":{"prompt_tokens":18446744073709551615,"completion_tokens":1,"total_tokens":0,"completion_tokens_details":{"reasoning_tokens":1}}}"#;
    let billed = std::panic::catch_unwind(|| usage::openai_body(body));
    assert!(billed.is_ok(), "usage::openai_body panicked");
    assert_eq!(billed.unwrap().map(|u| u.output_tokens), Some(1));
    // The translated client's usage goes through `translate`'s own copy of the same arithmetic.
    let body = br#"{"id":"c","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":18446744073709551615,"completion_tokens":1,"total_tokens":0,"completion_tokens_details":{"reasoning_tokens":1}}}"#;
    let shown = std::panic::catch_unwind(|| {
        translate::response_json(Endpoint::ChatCompletions, Endpoint::Messages, body)
    });
    assert!(shown.is_ok(), "translate::response_json panicked");
}

/// Nothing follows a client's terminal event: a Chat client's `[DONE]`, a Messages client's
/// `message_stop`. An upstream that sends an error (or anything) after its own end — after a Chat
/// usage chunk, after `message_stop` — must not reach the client past the end it was already
/// given. Shrunk from `prop_errors_and_cuts_anywhere_leave_a_well_formed_failure` and
/// `prop_vocabulary_events_keep_the_client_lifecycle`.
/// claim: S2
/// defect: D124
#[test]
fn nothing_reaches_the_client_after_its_terminal_event() {
    // Chat upstream → Messages client: an error between the usage chunk and `[DONE]`.
    let chat = concat!(
        "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"model\":\"m\",\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n",
        "data: {\"error\":{\"message\":\"boom\",\"type\":\"server_error\",\"code\":\"server_error\"}}\n\n",
        "data: [DONE]\n\n",
    );
    let out =
        SseBridge::new(Endpoint::Messages, Endpoint::ChatCompletions).feed(chat.as_bytes(), true);
    let out = String::from_utf8(out).unwrap();
    let after = out.split("event: message_stop").nth(1).unwrap_or("");
    assert!(
        !after.contains("event:"),
        "events after message_stop:{after}\nwhole stream:\n{out}"
    );
    // Messages upstream → Chat client: an event after `message_stop`.
    let ant = concat!(
        "data: {\"type\":\"message_stop\"}\n\n",
        "data: {\"type\":\"error\"}\n\n",
    );
    let out =
        SseBridge::new(Endpoint::ChatCompletions, Endpoint::Messages).feed(ant.as_bytes(), true);
    let out = String::from_utf8(out).unwrap();
    let after = out.split("data: [DONE]\n\n").nth(1).unwrap_or("");
    assert!(
        after.is_empty(),
        "data after [DONE]: {after}\nwhole stream:\n{out}"
    );
}

/// A request body's binary payload is excluded from the cut-short input estimate however the body
/// is chunked. A `;base64,` marker spread over three chunks (any chunk shorter than the marker
/// holding its middle) was missed, so the payload counted as text and the estimate jumped.
/// Shrunk from `prop_input_tally_ignores_chunking` (fed byte by byte).
/// claim: BIL-20
/// defect: D125
#[test]
fn input_estimate_ignores_how_the_body_is_chunked() {
    let image = "A".repeat(100_000);
    let body = format!(
        r#"{{"model":"m","messages":[{{"role":"user","content":[{{"type":"image_url","image_url":{{"url":"data:image/png;base64,{image}"}}}}]}}]}}"#
    );
    let mut whole = InputTally::default();
    whole.feed(body.as_bytes());
    // The marker's middle (`se`) arrives as its own two-byte chunk.
    let at = body.find(";base64,").unwrap();
    let (a, rest) = body.split_at(at + 3);
    let (b, c) = rest.split_at(2);
    let mut chunked = InputTally::default();
    for part in [a, b, c] {
        chunked.feed(part.as_bytes());
    }
    assert_eq!(
        chunked.estimate_tokens(),
        whole.estimate_tokens(),
        "the same body chunked differently estimates differently"
    );
}

/// SSE lets one event carry its data on several `data:` lines, joined with `\n` (JSON whitespace).
/// The translator joins them; the usage parsers read each line alone, so a usage event spread over
/// two lines bills no exact usage while the client is shown it. Shrunk from
/// `prop_streams_bill_like_their_bodies`.
/// claim: B1
/// defect: D126
#[test]
#[ignore = "D126 reproduced: a usage event spread over two data: lines is not parsed"]
fn usage_spread_over_two_data_lines_is_billed() {
    let openai = concat!(
        "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: {\n",
        "data: \"id\":\"c\",\"object\":\"chat.completion.chunk\",\"model\":\"m\",\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":9,\"total_tokens\":14}}\n\n",
        "data: [DONE]\n\n",
    );
    let u = usage::openai_stream(openai.as_bytes());
    assert_eq!(
        u.map(|u| (u.input_tokens, u.output_tokens)),
        Some((5, 9)),
        "OpenAI stream"
    );
    // The client side of the same bytes does see it: the bridge joins the lines.
    let shown =
        SseBridge::new(Endpoint::Messages, Endpoint::ChatCompletions).feed(openai.as_bytes(), true);
    assert!(
        String::from_utf8(shown)
            .unwrap()
            .contains("\"output_tokens\":9")
    );
    let anthropic = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"m\",\"content\":[],\"usage\":{\"input_tokens\":5,\"output_tokens\":1}}}\n\n",
        "event: message_delta\n",
        "data: {\n",
        "data: \"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":9}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );
    let u = usage::anthropic_stream(anthropic.as_bytes());
    assert_eq!(
        u.map(|u| (u.input_tokens, u.output_tokens)),
        Some((5, 9)),
        "Anthropic stream"
    );
}

/// The number literal after `"n":` in a JSON text.
fn n_of(text: &str) -> String {
    let at = text.find("\"n\":").unwrap() + 4;
    let rest = text[at..].trim_start();
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || matches!(c, '-' | '+' | '.' | 'e' | 'E')))
        .unwrap_or(rest.len());
    rest[..end].to_owned()
}

/// A number in a tool call's arguments keeps its value through translation. Re-parsed with
/// serde_json's default (not correctly rounded) float parser, a 15-digit float comes out a
/// different number, and an integer past `u64` becomes an `f64`. Shrunk from
/// `prop_tool_argument_numbers_keep_their_value`.
/// claim: T1
/// defect: D127
#[test]
#[ignore = "D127 reproduced: tool-argument numbers change value when translation re-parses them"]
fn tool_argument_numbers_keep_their_value() {
    // A Chat history call onto Messages.
    let body = r#"{"model":"m","messages":[{"role":"user","content":"hi"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"f","arguments":"{\"n\":-1.32417719719006e-11}"}}]},{"role":"tool","tool_call_id":"call_1","content":"ok"}]}"#;
    let out = translate::request(
        Endpoint::ChatCompletions,
        Endpoint::Messages,
        body.as_bytes(),
        "claude-sonnet-4-5",
    );
    let got = n_of(std::str::from_utf8(&out).unwrap());
    assert_eq!(
        got.parse::<f64>().ok(),
        Some(-1.32417719719006e-11),
        "history arguments became {got}"
    );
    // A Messages tool_use onto a Chat client: a 30-digit integer (an id, an account number).
    let body = r#"{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[{"type":"tool_use","id":"toolu_1","name":"f","input":{"n":123456789012345678901234567890}}],"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}}"#;
    let out = translate::response_json(
        Endpoint::Messages,
        Endpoint::ChatCompletions,
        body.as_bytes(),
    );
    let v: Value = serde_json::from_slice(&out).unwrap();
    let args = v
        .pointer("/choices/0/message/tool_calls/0/function/arguments")
        .and_then(Value::as_str)
        .unwrap();
    assert_eq!(
        n_of(args),
        "123456789012345678901234567890",
        "arguments: {args}"
    );
}

/// A Responses `conversation` is session state, like `previous_response_id`: its items live on
/// OpenAI, so a catalog walk relays it to a Responses arm or refuses it. It was neither: the
/// request counted as a one-shot, was translated, and `conversation` was dropped, so the model
/// answered without the conversation. Shrunk from
/// `prop_answer_changing_fields_are_never_silently_dropped`.
/// claim: E3
/// defect: D128
#[test]
fn a_responses_conversation_is_session_state() {
    let body = br#"{"model":"m","input":"What did I just say?","store":false,"conversation":"conv_abc123"}"#;
    for arm in [true, false] {
        assert!(
            translate::responses_session_field(body, arm).is_some(),
            "conversation counted as a one-shot (responses_arm = {arm})"
        );
    }
}

/// Responses options that change the answer reach the target or are forwarded for the provider to
/// reject by name, never dropped: the `prompt` template (documented as forwarded, and it is onto
/// Chat Completions, but not onto Messages) and `top_logprobs` (dropped onto both). Shrunk from
/// `prop_answer_changing_fields_are_never_silently_dropped`.
/// claim: TRN-21
/// defect: D129
#[test]
fn responses_answer_options_are_never_dropped() {
    let cases = [
        (
            Endpoint::Messages,
            "claude-sonnet-4-5",
            r#"{"model":"m","input":"hi","store":false,"prompt":{"id":"pmpt_abc123"}}"#,
            "pmpt_abc123",
        ),
        (
            Endpoint::Messages,
            "claude-sonnet-4-5",
            r#"{"model":"m","input":"hi","store":false,"top_logprobs":5}"#,
            "top_logprobs",
        ),
        (
            Endpoint::ChatCompletions,
            "gpt-5",
            r#"{"model":"m","input":"hi","store":false,"top_logprobs":5}"#,
            "top_logprobs",
        ),
    ];
    let lost: Vec<String> = cases
        .iter()
        .filter_map(|(to, model, body, needle)| {
            let out = translate::request(Endpoint::Responses, *to, body.as_bytes(), model);
            let out = String::from_utf8(out).unwrap();
            (!out.contains(needle)).then(|| format!("{to:?}: {body} -> {out}"))
        })
        .collect();
    assert!(lost.is_empty(), "dropped:\n{}", lost.join("\n"));
}
