use super::*;

/// OpenRouter's `native_finish_reason` on a Claude model is an Anthropic stop reason, and it
/// reaches a Messages client exactly, whatever the mapped `finish_reason` beside it says —
/// including the two (`pause_turn`, `model_context_window_exceeded`) the mapping table would
/// otherwise flatten to `end_turn` and `max_tokens`.
/// claim: TRN-22
#[test]
fn a_native_anthropic_stop_reason_wins_over_the_mapped_finish_reason() {
    for (native, finish) in [
        ("end_turn", "length"),
        ("max_tokens", "stop"),
        ("stop_sequence", "stop"),
        ("tool_use", "stop"),
        ("pause_turn", "stop"),
        ("refusal", "stop"),
        ("model_context_window_exceeded", "length"),
    ] {
        let body = json!({
            "id": "gen-1",
            "model": "anthropic/claude-sonnet-4.5",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": finish,
                "native_finish_reason": native,
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        });
        let out = response_json(
            Endpoint::ChatCompletions,
            Endpoint::Messages,
            &serde_json::to_vec(&body).unwrap(),
        );
        let out: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(out["stop_reason"], native, "{finish}");
    }
}

/// Which reasoning-control values ask for reasoning, each documented "off" spelling included,
/// and an object that says off in any one of its keys.
/// claim: TRN-7
#[test]
fn reasoning_value_on_reads_every_off_spelling() {
    for (v, on) in [
        (Value::Null, false),
        (json!(false), false),
        (json!(true), true),
        (json!("none"), false),
        (json!("high"), true),
        (json!({}), true),
        (json!({"effort": "none"}), false),
        (json!({"effort": "high"}), true),
        (json!({"enabled": false}), false),
        (json!({"enabled": true}), true),
        (json!({"type": "disabled"}), false),
        (json!({"type": "enabled"}), true),
        (json!({"effort": "high", "enabled": false}), false),
        (json!({"effort": "high", "type": "disabled"}), false),
        (json!({"enabled": true, "type": "disabled"}), false),
        (json!(7), true),
    ] {
        assert_eq!(reasoning_value_on(&v), on, "{v}");
    }
}

/// The hashed ids (tool ids, email `user_id`s) repeat across builds and processes only if the
/// hash is the published FNV-1a, not merely deterministic.
/// claim: TRN-23
#[test]
fn fnv1a64_matches_the_published_vectors() {
    assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
    assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
    assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_f739_67e8);
}

/// The effort ↔ budget tables, every row and boundary, and the effort a native `thinking`
/// object stands for.
/// claim: T5
#[test]
fn effort_and_budget_tables_hold_row_by_row() {
    for (effort, budget) in [
        ("minimal", 1024),
        ("low", 1024),
        ("medium", 4096),
        ("high", 8192),
        ("xhigh", 16384),
        ("max", 16384),
        ("other", 4096),
    ] {
        assert_eq!(budget_for_effort(effort), budget, "{effort}");
    }
    for (budget, effort) in [
        (0, "low"),
        (1024, "low"),
        (1025, "medium"),
        (4096, "medium"),
        (4097, "high"),
        (8192, "high"),
        (8193, "xhigh"),
    ] {
        assert_eq!(effort_from_budget(budget), effort, "{budget}");
    }
    for (thinking, effort) in [
        (json!({"type": "disabled"}), Some("none")),
        (json!({"type": "adaptive"}), Some("high")),
        (json!({"type": "adaptive", "effort": "max"}), Some("xhigh")),
        (
            json!({"type": "enabled", "budget_tokens": 2000}),
            Some("medium"),
        ),
        (json!({"type": "enabled"}), Some("medium")),
        (json!({"type": "between_tools"}), None),
    ] {
        assert_eq!(effort_from_thinking(&thinking), effort, "{thinking}");
    }
}

/// A data URI is base64 inline data only with a media type and a payload.
/// claim: T3
#[test]
fn a_data_uri_needs_both_a_media_type_and_data() {
    assert_eq!(
        parse_data_uri("data:image/png;base64,AAAA"),
        Some(("image/png", "AAAA"))
    );
    assert_eq!(parse_data_uri("data:;base64,AAAA"), None);
    assert_eq!(parse_data_uri("data:image/png;base64,"), None);
    assert_eq!(parse_data_uri("data:image/svg+xml;utf8,<svg/>"), None);
}

/// Tool-call arguments as a JSON string or an object become the input object; anything else
/// is `{}` (Messages' `input` must be an object).
/// claim: T1
#[test]
fn tool_arguments_become_an_input_object() {
    assert_eq!(parse_arguments(Some(&json!(r#"{"a":1}"#))), json!({"a": 1}));
    assert_eq!(parse_arguments(Some(&json!({"a": 1}))), json!({"a": 1}));
    assert_eq!(parse_arguments(Some(&json!([1]))), json!({}));
    assert_eq!(parse_arguments(Some(&json!(3))), json!({}));
    assert_eq!(parse_arguments(None), json!({}));
}

/// OpenAI strict mode: every object, at any depth (properties, items, combinators), closes
/// `additionalProperties` and requires every property; `type: object` alone or `properties`
/// alone makes a schema an object.
/// claim: TRN-11, T4
#[test]
fn openai_strict_schema_checks_every_nested_object() {
    let strict = json!({"type": "object", "properties": {"a": {"type": "string"}},
        "required": ["a"], "additionalProperties": false});
    let open = json!({"type": "object", "properties": {"a": {"type": "string"}}});
    assert!(openai_strict_schema(&strict));
    assert!(!openai_strict_schema(&open));
    assert!(!openai_strict_schema(&json!({"type": "object"})));
    assert!(!openai_strict_schema(&json!({"properties": {}})));
    assert!(openai_strict_schema(&json!({"type": "string"})));
    for key in ["items", "anyOf", "oneOf", "allOf", "prefixItems"] {
        let one = if key == "items" {
            open.clone()
        } else {
            json!([open])
        };
        assert!(
            !openai_strict_schema(&json!({"type": "array", key: one})),
            "{key}"
        );
        let ok = if key == "items" {
            strict.clone()
        } else {
            json!([strict])
        };
        assert!(
            openai_strict_schema(&json!({"type": "array", key: ok})),
            "{key}"
        );
    }
    for key in ["$defs", "definitions"] {
        assert!(!openai_strict_schema(&json!({key: {"x": open}})), "{key}");
    }
}

/// Each bound is crossed one byte past its documented size, not at it: an unterminated event
/// at 32 MiB, and gathered output at the catalog's largest answer times the widest token.
/// claim: SEC-19
#[test]
fn a_bridge_overflows_one_byte_past_each_bound() {
    let largest = providers::catalog::MODEL_ROUTES
        .iter()
        .map(|r| r.card.max_output_tokens as usize)
        .max()
        .unwrap();
    assert_eq!(MAX_STREAM_OUTPUT, largest * 128);
    let mut b = SseBridge::new(Endpoint::Messages, Endpoint::ChatCompletions);
    b.buf = vec![b' '; 32 << 20];
    assert_eq!(b.overflow(), None);
    b.buf.push(b' ');
    assert_eq!(b.overflow(), Some("sse_event"));
    b.buf = Vec::new();
    b.oai_to_resp.held = MAX_STREAM_OUTPUT;
    assert_eq!(b.overflow(), None);
    b.oai_to_resp.held += 1;
    assert_eq!(b.overflow(), Some("sse_output"));
}

/// A thinking block is held whole up to 8 MiB and dropped one byte past it, in both the
/// Messages-stream holder and the OpenRouter gatherer.
/// claim: T2
#[test]
fn thinking_is_held_up_to_8_mib_and_dropped_past_it() {
    let cap = 8 << 20;
    let at_cap = "a".repeat(cap);
    let mut held = HeldThinking::default();
    held.push(&at_cap[..cap - 1]);
    held.push("a");
    assert!(!held.overflow);
    assert_eq!(held.text.len(), cap);
    held.push("a");
    assert!(held.overflow && held.text.is_empty());
    let mut held = HeldThinking::default();
    held.push(&at_cap);
    held.push("a");
    assert!(held.overflow, "one push past the cap");

    let mut done = Vec::new();
    let mut g = Gather::default();
    g.text(None, &at_cap, &mut done);
    g.sign("S", &mut done);
    assert_eq!(done.len(), 1);
    assert_eq!(done[0]["thinking"].as_str().map(str::len), Some(cap));
    let mut done = Vec::new();
    let mut g = Gather::default();
    g.text(None, &at_cap, &mut done);
    g.text(None, "a", &mut done);
    g.sign("S", &mut done);
    g.text(None, &at_cap, &mut done);
    g.text(None, "a", &mut done);
    g.close(&mut done);
    assert!(done.is_empty(), "{done:?}");
}

/// An unsigned gathered block closes as a `thinking` block; with nothing gathered, closing
/// adds nothing.
/// claim: T2
#[test]
fn closing_a_gather_emits_only_a_block_it_holds() {
    let mut done = Vec::new();
    Gather::default().close(&mut done);
    assert!(done.is_empty());
    let mut g = Gather::default();
    g.text(Some(0), "hm", &mut done);
    g.close(&mut done);
    assert_eq!(done, [json!({"type": "thinking", "thinking": "hm"})]);
}

/// Only a number a `Value` keeps as an `f64`, at any depth, makes tool input keep its text; an
/// object stands for kept text only when that is its one member.
/// claim: T1
#[test]
fn only_a_float_keeps_tool_input_text() {
    assert!(has_float(&json!([1.5])));
    assert!(has_float(&json!({"a": [1, {"b": [2.5]}]})));
    assert!(!has_float(&json!({"a": [1, -2, "1.5"]})));
    assert_eq!(exact_value(r#"{"a": 1}"#, json!({"a": 1})), json!({"a": 1}));
    let kept = exact_value(r#"{"a": [1.10]}"#, json!({"a": [1.1]}));
    assert_eq!(raw_json(&kept), Some(r#"{"a": [1.10]}"#));
    let mut two = kept.as_object().unwrap().clone();
    two.insert("b".into(), json!(2));
    assert_eq!(raw_json(&Value::Object(two)), None);
    // Negative integers survive the input-keeping parse as integers.
    let v = parse_body(
        Endpoint::Messages,
        br#"{"temperature": -1, "input": {"n": -5}}"#,
    )
    .unwrap();
    assert_eq!(v, json!({"temperature": -1, "input": {"n": -5}}));
}

/// What is and is not an upstream error body.
/// claim: T6
#[test]
fn looks_like_error_needs_the_whole_signal() {
    for (v, err) in [
        // A Responses object is an error only when it failed *and* carries the error.
        (
            json!({"object": "response", "status": "failed", "error": {"code": "x"}}),
            true,
        ),
        (
            json!({"object": "response", "status": "failed", "error": null}),
            false,
        ),
        (
            json!({"object": "response", "status": "completed", "error": {"code": "x"}}),
            false,
        ),
        // A string `error` beside a success field is not an error.
        (json!({"error": "warn", "choices": []}), false),
        (json!({"error": "warn", "content": []}), false),
        (json!({"error": "warn", "output": []}), false),
        (json!({"error": "warn", "type": "message"}), false),
        (json!({"error": "boom"}), true),
        (json!({"error": {"message": "boom"}, "choices": []}), true),
    ] {
        assert_eq!(looks_like_error(&v), err, "{v}");
    }
}

/// Anthropic's error envelope needs `type: "error"` and both an error `type` and `message`.
/// claim: T6
#[test]
fn the_messages_envelope_needs_every_field() {
    let full = json!({"type": "error", "error": {"type": "x", "message": "m"}});
    assert!(in_client_envelope(&full, Endpoint::Messages));
    for v in [
        json!({"error": {"type": "x", "message": "m"}}),
        json!({"type": "error", "error": {"message": "m"}}),
        json!({"type": "error", "error": {"type": "x"}}),
    ] {
        assert!(!in_client_envelope(&v, Endpoint::Messages), "{v}");
    }
    assert!(in_client_envelope(
        &json!({"error": {"message": "m"}}),
        Endpoint::ChatCompletions
    ));
}

/// The error type a status means when the body names none, row by row.
/// claim: T6
#[test]
fn status_error_types_hold_row_by_row() {
    use Endpoint::{ChatCompletions as Chat, Messages};
    for (client, status, typ) in [
        (Messages, 400, "invalid_request_error"),
        (Messages, 401, "authentication_error"),
        (Messages, 402, "billing_error"),
        (Messages, 403, "permission_error"),
        (Messages, 404, "not_found_error"),
        (Messages, 413, "request_too_large"),
        (Messages, 429, "rate_limit_error"),
        (Messages, 500, "api_error"),
        (Messages, 504, "timeout_error"),
        (Messages, 529, "overloaded_error"),
        (Chat, 401, "invalid_request_error"),
        (Chat, 429, "rate_limit_error"),
        (Chat, 504, "api_error"),
    ] {
        assert_eq!(
            status_error_type(status, client),
            typ,
            "{client:?} {status}"
        );
    }
}

/// OpenAI error types and codes onto Anthropic's closed set, row by row.
/// claim: T6
#[test]
fn anthropic_error_types_hold_row_by_row() {
    for (typ, code, want) in [
        ("server_error", None, "api_error"),
        ("insufficient_quota", None, "billing_error"),
        (
            "invalid_request_error",
            Some("insufficient_quota"),
            "billing_error",
        ),
        ("requests", None, "rate_limit_error"),
        ("x", Some("rate_limit_exceeded"), "rate_limit_error"),
        (
            "invalid_request_error",
            Some("invalid_api_key"),
            "authentication_error",
        ),
        (
            "invalid_request_error",
            Some("model_not_found"),
            "not_found_error",
        ),
        (
            "invalid_request_error",
            Some("other"),
            "invalid_request_error",
        ),
    ] {
        let code = code.map(|c| json!(c));
        assert_eq!(
            anthropic_error_type(typ, code.as_ref()),
            want,
            "{typ} {code:?}"
        );
    }
}

/// An error message past 4096 bytes is cut at the last character boundary at or before it.
/// claim: T6
#[test]
fn clip_cuts_at_a_character_boundary() {
    assert_eq!(clip(&"a".repeat(4096)), "a".repeat(4096));
    assert_eq!(clip(&"a".repeat(5000)), format!("{}…", "a".repeat(4096)));
    let wide = format!("{}é{}", "a".repeat(4095), "b".repeat(10));
    assert_eq!(clip(&wide), format!("{}…", "a".repeat(4095)));
}

/// An error body with no message anywhere is quoted whole (a choice that finished with an
/// error says so instead); OpenRouter's provider error as an object is quoted after its
/// wrapper's message; and a message that does not say overflow gets no overflow code.
/// claim: T6, TRN-18
#[test]
fn error_messages_come_from_wherever_the_body_keeps_them() {
    let chat = Endpoint::ChatCompletions;
    let out = map_error(&json!({"foo": "bar"}), chat, 500);
    assert_eq!(out["error"]["message"], r#"{"foo":"bar"}"#, "{out}");
    let ended = json!({"choices": [{"finish_reason": "error"}]});
    assert_eq!(
        map_error(&ended, chat, 200)["error"]["message"],
        "The upstream ended the response with an error."
    );
    let wrapped = json!({"error": {"message": "Provider returned error", "code": 400,
        "metadata": {"provider_name": "Acme", "raw": {"error": {"message": "inner"}}}}});
    assert_eq!(
        map_error(&wrapped, chat, 400)["error"]["message"],
        "Provider returned error (Acme): inner"
    );
    let plain = map_error(&json!({"error": {"message": "bad"}}), chat, 400);
    assert_eq!(plain["error"]["code"], Value::Null, "{plain}");
}
