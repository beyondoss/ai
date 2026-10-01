//! Request-side translation: what each wire's body becomes on the upstream this attempt targets.
//!
//! Every test here pins a defect a stock SDK hit in production shape (a 400 from the provider, or a
//! field that vanished). The upstream model id is the candidate's, since several mappings depend on
//! what that model accepts.

use super::*;

fn req(from: Endpoint, to: Endpoint, body: &Value, model: &str) -> Value {
    serde_json::from_slice(&request(
        from,
        to,
        &serde_json::to_vec(body).unwrap(),
        model,
    ))
    .unwrap()
}

/// Anthropic SDK → a Chat Completions row.
fn m2c(body: &Value, model: &str) -> Value {
    req(Endpoint::Messages, Endpoint::ChatCompletions, body, model)
}

/// OpenAI SDK → a Messages row.
fn c2m(body: &Value, model: &str) -> Value {
    req(Endpoint::ChatCompletions, Endpoint::Messages, body, model)
}

/// Responses client → a Chat Completions row.
fn r2c(body: &Value, model: &str) -> Value {
    req(Endpoint::Responses, Endpoint::ChatCompletions, body, model)
}

/// OpenAI SDK → a Responses-only row.
fn c2r(body: &Value, model: &str) -> Value {
    req(Endpoint::ChatCompletions, Endpoint::Responses, body, model)
}

fn with(base: Value, extra: Value) -> Value {
    let mut b = base;
    if let (Some(o), Some(e)) = (b.as_object_mut(), extra.as_object()) {
        for (k, v) in e {
            o.insert(k.clone(), v.clone());
        }
    }
    b
}

fn anth(extra: Value) -> Value {
    with(
        json!({"model": "m", "max_tokens": 300, "messages": [{"role": "user", "content": "hi"}]}),
        extra,
    )
}

fn chat(extra: Value) -> Value {
    with(
        json!({"model": "m", "max_tokens": 4000, "messages": [{"role": "user", "content": "hi"}]}),
        extra,
    )
}

fn roles(v: &Value) -> Vec<&str> {
    v["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["role"].as_str().unwrap())
        .collect()
}

const WEATHER: &str =
    r#"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"#;

fn weather_schema() -> Value {
    serde_json::from_str(WEATHER).unwrap()
}

// ---- OpenAI model capabilities onto Chat Completions (audit 2/12) ----

#[test]
fn openai_model_parses_families_and_hosts() {
    use OpenAiReasoning::*;
    for (id, native, reasoning) in [
        ("gpt-4.1-nano", true, Off),
        ("gpt-4o-mini", true, Off),
        ("gpt-4-turbo", true, Off),
        ("chatgpt-4o-latest", true, Off),
        (
            "gpt-5-nano",
            true,
            Efforts(&["minimal", "low", "medium", "high"]),
        ),
        ("gpt-5-pro", true, Efforts(&["high"])),
        ("gpt-5.1", true, Efforts(&["none", "low", "medium", "high"])),
        (
            "gpt-5.2",
            true,
            Efforts(&["none", "low", "medium", "high", "xhigh"]),
        ),
        (
            "gpt-5.3-codex",
            true,
            Efforts(&["none", "low", "medium", "high", "xhigh"]),
        ),
        (
            "gpt-5.6-luna",
            true,
            Efforts(&["none", "low", "medium", "high", "xhigh"]),
        ),
        ("gpt-5.5-pro", true, Efforts(&["medium", "high", "xhigh"])),
        (
            "gpt-6-astra",
            true,
            Efforts(&["low", "medium", "high", "xhigh"]),
        ),
        ("o3", true, Efforts(&["low", "medium", "high"])),
        ("o4-mini", true, Efforts(&["low", "medium", "high"])),
        ("gpt-5.2-chat", true, Unknown),
        (
            "openai/gpt-5-nano",
            false,
            Efforts(&["minimal", "low", "medium", "high"]),
        ),
        ("openai/o4-mini", false, Efforts(&["low", "medium", "high"])),
        // gpt-oss on any host: only the effort is fit (reasoning is mandatory there).
        (
            "openai/gpt-oss-120b",
            false,
            Efforts(&["low", "medium", "high"]),
        ),
        (
            "accounts/fireworks/models/gpt-oss-120b",
            false,
            Efforts(&["low", "medium", "high"]),
        ),
        // Not OpenAI's own model, or not OpenAI at all: nothing is reshaped.
        ("grok-4.6", false, Unknown),
        ("deepseek-chat", false, Unknown),
        ("x-ai/grok-4.6", false, Unknown),
        ("claude-opus-4-8", false, Unknown),
    ] {
        assert_eq!(
            OpenAiModel::of(id),
            OpenAiModel { native, reasoning },
            "{id}"
        );
    }
}

/// OpenAI 400s `max_tokens` ("Use 'max_completion_tokens'") and non-default sampling on every
/// reasoning model: the Anthropic SDK's everyday body used to fail on every GPT-5 row.
#[test]
fn anthropic_sdk_onto_gpt5_sends_max_completion_tokens_and_no_sampling() {
    let v = m2c(
        &anth(json!({"temperature": 0.2, "top_p": 0.9})),
        "gpt-5-nano",
    );
    assert_eq!(v["max_completion_tokens"], 300, "{v}");
    assert!(v.get("max_tokens").is_none(), "{v}");
    assert!(
        v.get("temperature").is_none() && v.get("top_p").is_none(),
        "{v}"
    );
    // Non-reasoning models keep sampling; OpenAI's own API takes `max_completion_tokens` there too.
    let v = m2c(&anth(json!({"temperature": 0.2})), "gpt-4.1-nano");
    assert_eq!(v["temperature"], 0.2);
    assert_eq!(v["max_completion_tokens"], 300);
}

/// `thinking: disabled` → `reasoning_effort: "none"` was "Unrecognized request argument" on
/// gpt-4.1 and "does not support 'none'" on gpt-5.
#[test]
fn thinking_disabled_fits_each_family() {
    let off = anth(json!({"thinking": {"type": "disabled"}}));
    let effort = |model: &str| m2c(&off, model).get("reasoning_effort").cloned();
    assert_eq!(
        effort("gpt-4.1-nano"),
        None,
        "no reasoning model takes the field"
    );
    assert_eq!(effort("gpt-5-nano"), Some(json!("minimal")), "its lowest");
    assert_eq!(effort("o4-mini"), Some(json!("low")));
    assert_eq!(effort("gpt-5.1"), Some(json!("none")));
    assert_eq!(effort("gpt-6-astra"), Some(json!("low")));
    assert_eq!(effort("gpt-5-pro"), Some(json!("high")), "its only value");
}

/// `effort: max` → `xhigh` is outside gpt-5's and the o-series' accepted sets.
#[test]
fn effort_is_clamped_to_what_each_family_accepts() {
    let max = anth(json!({"output_config": {"effort": "max"}}));
    for (model, want) in [
        ("gpt-5-nano", "high"),
        ("gpt-5.1", "high"),
        ("gpt-5.4-nano", "xhigh"),
        ("gpt-6-astra", "xhigh"),
        ("o3", "high"),
        ("openai/o4-mini", "high"),
    ] {
        assert_eq!(m2c(&max, model)["reasoning_effort"], want, "{model}");
    }
    // A gap in the ladder takes the next value up: `minimal` on 5.1 is `low`.
    let r =
        json!({"model": "m", "store": false, "input": "hi", "reasoning": {"effort": "minimal"}});
    assert_eq!(r2c(&r, "gpt-5.1")["reasoning_effort"], "low");
}

/// Sampling stays wherever the model accepts it: reasoning off, a non-reasoning model, or a host
/// that normalizes it itself.
#[test]
fn sampling_survives_where_the_upstream_accepts_it() {
    let body = anth(json!({"temperature": 0.2, "thinking": {"type": "disabled"}}));
    let v = m2c(&body, "gpt-5.1");
    assert_eq!(v["reasoning_effort"], "none");
    assert_eq!(
        v["temperature"], 0.2,
        "5.1 takes sampling at effort none: {v}"
    );
    let v = m2c(&body, "gpt-5-nano");
    assert!(v.get("temperature").is_none(), "gpt-5 always reasons: {v}");
}

/// OpenRouter's `openai/…` normalizes `max_tokens` and sampling itself (measured), but passes the
/// effort through, where `none` on gpt-5 is "Reasoning is mandatory".
#[test]
fn openrouter_openai_ids_get_only_the_effort_clamp() {
    let v = m2c(
        &anth(json!({"temperature": 0.2, "thinking": {"type": "disabled"}})),
        "openai/gpt-5-nano",
    );
    assert_eq!(v["max_tokens"], 300);
    assert_eq!(v["temperature"], 0.2);
    assert_eq!(v["reasoning_effort"], "minimal");
}

#[test]
fn other_chat_hosts_keep_the_body_as_sent() {
    let v = m2c(
        &anth(json!({"temperature": 0.2, "thinking": {"type": "disabled"}})),
        "grok-4.6",
    );
    assert_eq!(v["max_tokens"], 300);
    assert_eq!(v["temperature"], 0.2);
    assert_eq!(v["reasoning_effort"], "none");
}

/// A Responses client on a GPT row goes through the same rules.
#[test]
fn responses_onto_gpt_follows_the_same_rules() {
    let body = json!({
        "model": "m", "store": false, "max_output_tokens": 500, "temperature": 0.3,
        "reasoning": {"effort": "xhigh"}, "input": "hi"
    });
    let v = r2c(&body, "gpt-5-nano");
    assert_eq!(v["max_completion_tokens"], 500);
    assert!(
        v.get("max_tokens").is_none() && v.get("temperature").is_none(),
        "{v}"
    );
    assert_eq!(v["reasoning_effort"], "high");
    let v = r2c(&body, "gpt-4.1-nano");
    assert!(v.get("reasoning_effort").is_none(), "{v}");
    assert_eq!(v["temperature"], 0.3);
}

/// `reasoning` is "Unsupported parameter" on Responses for a non-reasoning model.
#[test]
fn chat_onto_responses_fits_reasoning_to_the_model() {
    let body = chat(json!({"reasoning_effort": "high", "temperature": 0.3}));
    let v = c2r(&body, "gpt-4.1-nano");
    assert!(v.get("reasoning").is_none(), "{v}");
    assert_eq!(v["temperature"], 0.3);
    let v = c2r(&body, "gpt-5-pro");
    assert_eq!(v["reasoning"]["effort"], "high");
    assert!(v.get("temperature").is_none(), "{v}");
    let v = c2r(&chat(json!({"reasoning_effort": "low"})), "gpt-5.5-pro");
    assert_eq!(v["reasoning"]["effort"], "medium", "pro's lowest");
}

// ---- Responses → Chat Completions (audit 6/7) ----

fn fc(id: &str, city: &str) -> Value {
    json!({"type": "function_call", "call_id": id, "name": "get_weather", "arguments": format!("{{\"city\":\"{city}\"}}")})
}

/// Parallel calls are consecutive `function_call` items; one assistant message per call is a 400
/// on OpenAI ("tool messages must follow the assistant message with their tool_calls").
#[test]
fn responses_parallel_function_calls_become_one_assistant_message() {
    let body = json!({"model": "m", "store": false, "input": [
        {"role": "user", "content": "Paris and Rome?"},
        {"type": "reasoning", "id": "rs_1", "summary": []},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Checking."}]},
        fc("call_a", "Paris"),
        fc("call_b", "Rome"),
        {"type": "function_call_output", "call_id": "call_a", "output": "sunny"},
        {"type": "function_call_output", "call_id": "call_b", "output": "rain"},
        fc("call_c", "Oslo"),
        {"type": "function_call_output", "call_id": "call_c", "output": "snow"},
    ]});
    let v = r2c(&body, "gpt-4.1-nano");
    assert_eq!(
        roles(&v),
        ["user", "assistant", "tool", "tool", "assistant", "tool"],
        "{v}"
    );
    let turn = &v["messages"][1];
    assert_eq!(
        turn["content"], "Checking.",
        "the calls join the turn's text"
    );
    let ids: Vec<&str> = turn["tool_calls"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["call_a", "call_b"]);
    assert_eq!(v["messages"][4]["tool_calls"][0]["id"], "call_c");
}

/// Hosted and custom tools used to vanish: the model answered without a tool the client offered.
#[test]
fn responses_hosted_tools_are_forwarded_and_custom_tools_mapped() {
    let body = json!({"model": "m", "store": false, "input": "x", "tools": [
        {"type": "web_search"},
        {"type": "file_search", "vector_store_ids": ["vs_1"]},
        {"type": "custom", "name": "apply_patch", "description": "patch", "format": {"type": "text"}},
        {"type": "function", "name": "f", "parameters": {"type": "object", "properties": {}}, "strict": true}
    ]});
    let v = r2c(&body, "gpt-5-nano");
    let tools = v["tools"].as_array().unwrap();
    assert_eq!(tools[0], json!({"type": "web_search"}));
    assert_eq!(tools[1]["type"], "file_search");
    assert_eq!(
        tools[2],
        json!({"type": "custom", "custom": {"name": "apply_patch", "description": "patch", "format": {"type": "text"}}})
    );
    assert_eq!(tools[3]["function"]["strict"], true);
}

#[test]
fn responses_tool_choice_shapes_become_chat_shapes() {
    let choice = |c: Value| {
        r2c(
            &json!({"model": "m", "store": false, "input": "x", "tool_choice": c,
                "tools": [{"type": "function", "name": "get_weather", "parameters": {"type": "object"}}]}),
            "gpt-5",
        )["tool_choice"]
            .clone()
    };
    assert_eq!(
        choice(json!({"type": "function", "name": "get_weather"})),
        json!({"type": "function", "function": {"name": "get_weather"}})
    );
    assert_eq!(
        choice(json!({"type": "custom", "name": "apply_patch"})),
        json!({"type": "custom", "custom": {"name": "apply_patch"}})
    );
    assert_eq!(
        choice(
            json!({"type": "allowed_tools", "mode": "required", "tools": [{"type": "function", "name": "a"}]})
        ),
        json!({"type": "allowed_tools", "allowed_tools": {"mode": "required", "tools": [{"type": "function", "function": {"name": "a"}}]}})
    );
    assert_eq!(choice(json!("required")), json!("required"));
    assert_eq!(
        choice(json!({"type": "web_search"})),
        json!({"type": "web_search"})
    );
}

/// An array `function_call_output` was passed through as Responses parts, which Chat rejects.
#[test]
fn responses_array_tool_output_is_flattened_and_images_follow() {
    let body = json!({"model": "m", "store": false, "input": [
        {"role": "user", "content": "screenshot"},
        {"type": "function_call", "call_id": "call_a", "name": "shot", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_a", "output": [
            {"type": "input_text", "text": "ok"},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA", "detail": "low"}
        ]},
        {"role": "user", "content": "what is on it?"}
    ]});
    let v = r2c(&body, "gpt-4.1-nano");
    assert_eq!(
        roles(&v),
        ["user", "assistant", "tool", "user", "user"],
        "{v}"
    );
    assert_eq!(v["messages"][2]["content"], "ok");
    assert_eq!(
        v["messages"][3]["content"],
        json!([{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": "low"}}])
    );
}

// ---- Messages → Chat Completions (audit 8/9/13/15/16) ----

/// A server tool became an empty-schema function the model would call and nothing would run.
#[test]
fn anthropic_server_tools_are_forwarded_not_emptied() {
    let tools = json!([
        {"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
        {"type": "bash_20250124", "name": "bash"},
        {"type": "custom", "name": "f", "input_schema": weather_schema()},
        {"name": "g", "input_schema": weather_schema()}
    ]);
    let v = m2c(&anth(json!({"tools": tools.clone()})), "gpt-5");
    let out = v["tools"].as_array().unwrap();
    assert_eq!(out[0], tools[0]);
    assert_eq!(out[1], tools[1]);
    assert_eq!(out[2]["function"]["name"], "f");
    assert_eq!(out[3]["function"]["parameters"], weather_schema());
}

/// Images in a `tool_result` (a screenshot tool) were dropped: the model answered about output it
/// never saw. A tool message holds text, so they follow as a user message.
#[test]
fn tool_result_images_follow_the_tool_messages() {
    let img = json!({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}});
    let body = anth(json!({"messages": [
        {"role": "user", "content": "look"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "shot", "input": {}},
            {"type": "tool_use", "id": "toolu_2", "name": "shot", "input": {}}
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "Image:"}, img]},
            {"type": "tool_result", "tool_use_id": "toolu_2", "content": "none"},
            {"type": "text", "text": "compare them"}
        ]}
    ]}));
    let v = m2c(&body, "gpt-5");
    assert_eq!(
        roles(&v),
        ["user", "assistant", "tool", "tool", "user"],
        "{v}"
    );
    assert_eq!(v["messages"][2]["content"], "Image:");
    assert_eq!(v["messages"][3]["content"], "none");
    assert_eq!(
        v["messages"][4]["content"],
        json!([
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
            {"type": "text", "text": "compare them"}
        ])
    );
}

#[test]
fn tool_result_is_error_is_said_in_text() {
    let body = anth(json!({"messages": [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "t", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "is_error": true, "content": "permission denied"}]}
    ]}));
    assert_eq!(
        m2c(&body, "gpt-5")["messages"][2]["content"],
        "Error: permission denied"
    );
}

/// `strict: true` with an optional property is a 400 on OpenAI.
#[test]
fn structured_output_is_strict_only_when_openai_can_be() {
    let optional = json!({"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}, "required": ["a"], "additionalProperties": false});
    let v = m2c(
        &anth(json!({"output_config": {"format": {"type": "json_schema", "schema": optional}}})),
        "gpt-5",
    );
    assert_eq!(v["response_format"]["json_schema"]["strict"], false);
    let nested_open = json!({"type": "object", "properties": {"a": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}}, "required": ["a"], "additionalProperties": false});
    let v = m2c(
        &anth(json!({"output_config": {"format": {"type": "json_schema", "schema": nested_open}}})),
        "gpt-5",
    );
    assert_eq!(
        v["response_format"]["json_schema"]["strict"], false,
        "nested object lacks additionalProperties"
    );
    let closed = json!({"type": "object", "properties": {"a": {"type": "array", "items": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"], "additionalProperties": false}}}, "required": ["a"], "additionalProperties": false});
    let v = m2c(
        &anth(json!({"output_config": {"format": {"type": "json_schema", "schema": closed}}})),
        "gpt-5",
    );
    assert_eq!(v["response_format"]["json_schema"]["strict"], true);
}

/// Tool `strict` was dropped both ways, losing the schema-valid-arguments guarantee.
#[test]
fn tool_strict_crosses_both_ways() {
    let closed = json!({"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"], "additionalProperties": false});
    let v = c2m(
        &chat(
            json!({"tools": [{"type": "function", "function": {"name": "t", "strict": true, "parameters": closed.clone()}}]}),
        ),
        "claude-opus-4-8",
    );
    assert_eq!(v["tools"][0]["strict"], true);
    let v = m2c(
        &anth(json!({"tools": [{"name": "t", "strict": true, "input_schema": closed}]})),
        "gpt-5",
    );
    assert_eq!(v["tools"][0]["function"]["strict"], true);
    // Anthropic's strict allows optional properties; OpenAI's does not.
    let v = m2c(
        &anth(
            json!({"tools": [{"name": "t", "strict": true, "input_schema": {"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": false}}]}),
        ),
        "gpt-5",
    );
    assert_eq!(v["tools"][0]["function"]["strict"], false);
}

/// A mid-conversation `role: "system"` became a user message: a user saying "answer in French".
#[test]
fn messages_mid_conversation_system_stays_a_system_message_on_chat() {
    let body = anth(json!({"system": "base", "messages": [
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "From now on answer in French."},
        {"role": "assistant", "content": "Bonjour"},
        {"role": "user", "content": "again"},
        {"role": "system", "content": [], "output_config": {"effort": "low"}}
    ]}));
    let v = m2c(&body, "gpt-5");
    assert_eq!(
        roles(&v),
        ["system", "user", "system", "assistant", "user"],
        "{v}"
    );
    assert_eq!(v["messages"][2]["content"], "From now on answer in French.");
}

#[test]
fn mcp_servers_and_file_images_are_forwarded() {
    let file_img = json!({"type": "image", "source": {"type": "file", "file_id": "file_abc"}});
    let v = m2c(
        &anth(json!({
            "mcp_servers": [{"type": "url", "url": "https://mcp.example.com", "name": "x"}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "describe"}, file_img.clone()]}]
        })),
        "gpt-5",
    );
    assert_eq!(v["mcp_servers"][0]["name"], "x");
    assert_eq!(v["messages"][0]["content"][1], file_img);
}

// ---- Chat Completions → Messages (audit 10/11/14/17/20) ----

/// Pre-4.6 Claude 400s `temperature` above 1 and `temperature` with `top_p`.
#[test]
fn sampling_onto_older_claude_is_clamped_and_top_p_yields() {
    let v = c2m(
        &chat(json!({"temperature": 1.5, "top_p": 0.9})),
        "claude-haiku-4-5",
    );
    assert_eq!(v["temperature"], 1.0);
    assert!(v.get("top_p").is_none(), "{v}");
    let v = c2m(&chat(json!({"top_p": 0.9})), "claude-haiku-4-5");
    assert_eq!(v["top_p"], 0.9, "alone it stays");
}

/// "user_id appears to contain an email address" is a 400; the hash is stable per user.
#[test]
fn an_email_user_id_is_hashed_stably() {
    let id =
        |u: &str| c2m(&chat(json!({"user": u})), "claude-opus-4-8")["metadata"]["user_id"].clone();
    let a = id("jared@example.com");
    assert!(!a.as_str().unwrap().contains('@'), "{a}");
    assert_eq!(a.as_str().unwrap().len(), 16);
    assert_eq!(a, id("jared@example.com"), "stable");
    assert_ne!(a, id("other@example.com"));
    assert_eq!(id("user_123"), "user_123", "an opaque id passes");
}

/// "Thinking may not be enabled when tool_choice forces tool use" on budget-thinking models.
#[test]
fn forced_tool_use_wins_over_budget_thinking() {
    let body = chat(json!({
        "reasoning_effort": "low", "tool_choice": "required",
        "tools": [{"type": "function", "function": {"name": "t", "parameters": {"type": "object", "properties": {}}}}]
    }));
    let v = c2m(&body, "claude-haiku-4-5");
    assert_eq!(v["tool_choice"]["type"], "any");
    assert!(v.get("thinking").is_none(), "{v}");
    // Adaptive thinking takes forced tool use (measured on 4.6, 4.8, Opus 5).
    let v = c2m(&body, "claude-opus-4-8");
    assert_eq!(v["thinking"]["type"], "adaptive");
    assert_eq!(v["tool_choice"]["type"], "any");
}

/// A client's own `thinking` object bypassed the model mapping: `budget_tokens` is a 400 on Opus
/// 4.8, and `{type: enabled}` without a budget is "budget_tokens: Field required".
#[test]
fn a_native_thinking_object_is_fit_to_the_model() {
    let t = |thinking: Value, model: &str| c2m(&chat(json!({"thinking": thinking})), model);
    let v = t(
        json!({"type": "enabled", "budget_tokens": 1024}),
        "claude-opus-4-8",
    );
    assert_eq!(v["thinking"], json!({"type": "adaptive"}));
    assert_eq!(v["output_config"]["effort"], "low");
    let v = t(json!({"type": "enabled"}), "claude-opus-4-8");
    assert_eq!(v["thinking"]["type"], "adaptive");
    let v = t(json!({"type": "enabled"}), "claude-haiku-4-5");
    assert_eq!(
        v["thinking"],
        json!({"type": "enabled", "budget_tokens": 2000})
    );
    let v = t(
        json!({"type": "enabled", "budget_tokens": 99999}),
        "claude-haiku-4-5",
    );
    assert_eq!(
        v["thinking"]["budget_tokens"], 3999,
        "held below max_tokens"
    );
    let v = t(
        json!({"type": "adaptive", "display": "summarized"}),
        "claude-haiku-4-5",
    );
    assert_eq!(
        v["thinking"]["type"], "enabled",
        "no adaptive before 4.6: {v}"
    );
    assert_eq!(v["thinking"]["display"], "summarized");
    let v = t(json!({"type": "disabled"}), "claude-opus-5");
    assert!(v.get("thinking").is_none(), "{v}");
    assert_eq!(v["output_config"]["effort"], "low");
    let v = t(json!({"type": "between_tools"}), "claude-sonnet-5-5");
    assert_eq!(
        v["thinking"],
        json!({"type": "between_tools"}),
        "no block_binding with it"
    );
    let v = t(json!({"type": "between_tools"}), "claude-opus-4-8");
    assert!(v.get("thinking").is_none(), "only Sonnet 5.5 has it: {v}");
}

/// A Kimi-style id (`functions.get_weather:0`) is a 400 on Anthropic's id pattern.
#[test]
fn foreign_tool_ids_are_sanitized_identically_on_both_sides() {
    let body = chat(json!({"messages": [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": null, "tool_calls": [
            {"id": "functions.get_weather:0", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}},
            {"id": "functions_get_weather_0", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "functions.get_weather:0", "content": "sunny"},
        {"role": "tool", "tool_call_id": "functions_get_weather_0", "content": "rain"}
    ]}));
    let v = c2m(&body, "claude-haiku-4-5");
    let uses = v["messages"][1]["content"].as_array().unwrap();
    let results = v["messages"][2]["content"].as_array().unwrap();
    let valid = |s: &str| {
        s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    };
    assert!(valid(uses[0]["id"].as_str().unwrap()), "{v}");
    assert_eq!(uses[0]["id"], results[0]["tool_use_id"]);
    assert_eq!(
        uses[1]["id"], "functions_get_weather_0",
        "valid ids are untouched"
    );
    assert_eq!(results[1]["tool_use_id"], "functions_get_weather_0");
    assert_ne!(uses[0]["id"], uses[1]["id"], "no collision");
    assert_eq!(v, c2m(&body, "claude-haiku-4-5"), "deterministic");
}

#[test]
fn legacy_functions_become_tools() {
    let body = chat(json!({
        "functions": [{"name": "get_weather", "parameters": weather_schema()}],
        "function_call": {"name": "get_weather"},
        "messages": [
            {"role": "user", "content": "Paris?"},
            {"role": "assistant", "content": null, "function_call": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}},
            {"role": "function", "name": "get_weather", "content": "sunny"},
            {"role": "user", "content": "Rome?"}
        ]
    }));
    let v = c2m(&body, "claude-haiku-4-5");
    assert_eq!(v["tools"][0]["name"], "get_weather");
    assert_eq!(
        v["tool_choice"],
        json!({"type": "tool", "name": "get_weather"})
    );
    let call = &v["messages"][1]["content"][0];
    assert_eq!(call["type"], "tool_use");
    assert_eq!(call["input"]["city"], "Paris");
    assert_eq!(v["messages"][2]["content"][0]["tool_use_id"], call["id"]);
    // The same body onto a Responses row.
    let v = c2r(&body, "gpt-5-pro");
    assert_eq!(v["tools"][0]["name"], "get_weather");
    assert_eq!(
        v["tool_choice"],
        json!({"type": "function", "name": "get_weather"})
    );
}

#[test]
fn allowed_tools_required_forces() {
    let tools = json!([
        {"type": "function", "function": {"name": "a", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "b", "parameters": {"type": "object", "properties": {}}}}
    ]);
    let choice = |mode: &str, names: &[&str]| {
        let refs: Vec<Value> = names
            .iter()
            .map(|n| json!({"type": "function", "function": {"name": n}}))
            .collect();
        c2m(
            &chat(json!({"tools": tools.clone(), "tool_choice": {"type": "allowed_tools", "allowed_tools": {"mode": mode, "tools": refs}}})),
            "claude-haiku-4-5",
        )["tool_choice"]
            .clone()
    };
    assert_eq!(
        choice("required", &["a"]),
        json!({"type": "tool", "name": "a"})
    );
    assert_eq!(choice("required", &["a", "b"]), json!({"type": "any"}));
    assert_eq!(choice("auto", &["a"]), json!({"type": "auto"}));
}

#[test]
fn a_non_base64_data_uri_is_forwarded_and_other_schemes_are_not() {
    let svg = json!({"type": "image_url", "image_url": {"url": "data:image/svg+xml;utf8,<svg/>"}});
    let file = json!({"type": "image_url", "image_url": {"url": "file:///etc/passwd"}});
    let v = c2m(
        &chat(
            json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "what?"}, svg.clone(), file]}]}),
        ),
        "claude-haiku-4-5",
    );
    assert_eq!(v["messages"][0]["content"][1], svg);
    assert_eq!(
        v["messages"][0]["content"].as_array().unwrap().len(),
        2,
        "{v}"
    );
}

#[test]
fn an_assistant_refusal_is_kept_as_text() {
    let v = c2m(
        &chat(json!({"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "refusal", "refusal": "I can't help with that."}]},
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": null, "refusal": "No."},
            {"role": "user", "content": "fine"}
        ]})),
        "claude-haiku-4-5",
    );
    assert_eq!(v["messages"][1]["content"], "I can't help with that.");
    assert_eq!(v["messages"][3]["content"], "No.");
}

#[test]
fn a_tool_message_image_stays_in_the_tool_result() {
    let v = c2m(
        &chat(json!({"messages": [
            {"role": "user", "content": "shot"},
            {"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "s", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": [
                {"type": "text", "text": "ok"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
            ]}
        ]})),
        "claude-haiku-4-5",
    );
    let result = &v["messages"][2]["content"][0];
    assert_eq!(result["content"][0]["text"], "ok");
    assert_eq!(result["content"][1]["source"]["type"], "base64");
}

#[test]
fn custom_and_hosted_tools_are_forwarded_onto_messages() {
    let custom =
        json!({"type": "custom", "custom": {"name": "apply_patch", "format": {"type": "text"}}});
    let v = c2m(&chat(json!({"tools": [custom.clone()]})), "claude-opus-4-8");
    assert_eq!(v["tools"][0]["type"], "custom");
    assert_eq!(v["tools"][0]["custom"], custom["custom"]);
}

/// Responses `reasoning.summary` asks for readable reasoning; current Claude defaults to `omitted`.
#[test]
fn a_reasoning_summary_asks_for_summarized_thinking() {
    let body = json!({"model": "m", "store": false, "input": "x", "reasoning": {"effort": "high", "summary": "auto"}});
    let v = req(
        Endpoint::Responses,
        Endpoint::Messages,
        &body,
        "claude-opus-4-8",
    );
    assert_eq!(
        v["thinking"],
        json!({"type": "adaptive", "display": "summarized"})
    );
    // 4.6 and older summarize by default; nothing to add.
    let v = req(
        Endpoint::Responses,
        Endpoint::Messages,
        &body,
        "claude-sonnet-4-6",
    );
    assert_eq!(v["thinking"], json!({"type": "adaptive"}));
}

// ---- Preserved thinking (audit 14) ----

fn convo(tail: Value) -> Value {
    let mut msgs = vec![
        json!({"role": "system", "content": "You are helpful."}),
        json!({"role": "user", "content": "hi"}),
        json!({"role": "assistant", "content": "hello", "thinking": [{"type": "thinking", "thinking": "", "signature": "sig1"}]}),
    ];
    msgs.extend(tail.as_array().unwrap().iter().cloned());
    chat(json!({"messages": msgs}))
}

/// A mid-conversation system message was hoisted into top-level `system`, rewriting the prefix
/// every earlier thinking block is bound to (and the prompt cache) each time one was appended.
#[test]
fn mid_conversation_system_stays_in_place_where_supported() {
    let body = convo(json!([
        {"role": "user", "content": "x"},
        {"role": "system", "content": "Reminder: 3 files changed."}
    ]));
    let v = c2m(&body, "claude-opus-4-8");
    assert_eq!(v["system"][0]["text"], "You are helpful.");
    assert_eq!(v["system"].as_array().unwrap().len(), 1, "{v}");
    assert_eq!(roles(&v), ["user", "assistant", "user", "system"]);
    let last = v["messages"].as_array().unwrap().last().unwrap();
    assert_eq!(last["content"][0]["text"], "Reminder: 3 files changed.");
    // The next turn replays the same bytes in the same place.
    let next = convo(json!([
        {"role": "user", "content": "x"},
        {"role": "system", "content": "Reminder: 3 files changed."},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "y"}
    ]));
    let w = c2m(&next, "claude-opus-4-8");
    assert_eq!(w["system"], v["system"]);
    assert_eq!(w["messages"][3]["content"], "Reminder: 3 files changed.");
}

/// Messages accepts a system message only after a user turn and before an assistant turn (or last).
#[test]
fn a_misplaced_system_message_moves_past_the_next_user_turn() {
    let v = c2m(
        &convo(json!([
            {"role": "developer", "content": "Answer in French."},
            {"role": "user", "content": "again"},
            {"role": "user", "content": "and again"}
        ])),
        "claude-fable-5-1",
    );
    assert_eq!(roles(&v), ["user", "assistant", "user", "system"], "{v}");
    // Ending on an assistant turn leaves no valid place: the top-level prompt keeps its authority.
    let v = c2m(
        &chat(json!({"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "system", "content": "Answer in French."}
        ]})),
        "claude-opus-4-8",
    );
    assert_eq!(roles(&v), ["user", "assistant"]);
    assert_eq!(v["system"][0]["text"], "Answer in French.");
}

#[test]
fn mid_conversation_system_is_hoisted_where_unsupported() {
    let body = convo(
        json!([{"role": "user", "content": "x"}, {"role": "system", "content": "Be terse."}]),
    );
    for model in [
        "claude-sonnet-5",
        "claude-haiku-4-5",
        "anthropic/claude-sonnet-4.6",
        "kimi-k3",
    ] {
        let v = c2m(&body, model);
        assert!(roles(&v).iter().all(|r| *r != "system"), "{model}: {v}");
        assert!(
            v["system"].to_string().contains("Be terse."),
            "{model}: {v}"
        );
    }
}

/// The forced-tool instruction exists only in the gateway's request, so the client's next turn
/// replays that turn's thinking without it: "block is bound to a different conversation" (400)
/// on enforced accounts. `drop_block` degrades that to a dropped block.
#[test]
fn binding_models_get_drop_block() {
    let body = convo(json!([{"role": "user", "content": "x"}]));
    for model in ["claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-1"] {
        let v = c2m(&body, model);
        assert_eq!(
            v["thinking"]["block_binding"],
            json!({"prefix_mismatch_behavior": "drop_block"}),
            "{model}: {v}"
        );
        assert_eq!(v["thinking"]["type"], "adaptive");
        assert_eq!(messages_beta(model), Some(THINKING_BINDING_BETA));
    }
    // An explicit effort keeps its thinking object and gains the binding.
    let v = c2m(
        &with(body.clone(), json!({"reasoning_effort": "high"})),
        "claude-opus-5-5",
    );
    assert_eq!(
        v["thinking"]["block_binding"]["prefix_mismatch_behavior"],
        "drop_block"
    );
    assert_eq!(v["output_config"]["effort"], "high");
    // Also set on a first turn with nothing to replay: a thinking parameter that changed between
    // turns would restart the prompt cache.
    let v = c2m(&chat(json!({})), "claude-sonnet-5-5");
    assert_eq!(
        v["thinking"]["block_binding"]["prefix_mismatch_behavior"],
        "drop_block"
    );
}

#[test]
fn drop_block_only_where_the_field_is_accepted() {
    let body = convo(json!([{"role": "user", "content": "x"}]));
    for model in [
        "claude-opus-4-8",                   // no conversation check
        "claude-sonnet-5",                   // no conversation check
        "claude-mythos-5-1",                 // skips the conversation check
        "anthropic/claude-sonnet-5.5",       // OpenRouter: beta header unverified
        "us.anthropic.claude-opus-5-5-v1:0", // Bedrock: beta header unverified
    ] {
        let v = c2m(&body, model);
        assert!(
            v.pointer("/thinking/block_binding").is_none(),
            "{model}: {v}"
        );
        assert_eq!(messages_beta(model), None, "{model}");
    }
    // Never with `between_tools` (a 400).
    let v = c2m(
        &with(body, json!({"thinking": {"type": "between_tools"}})),
        "claude-sonnet-5-5",
    );
    assert_eq!(v["thinking"], json!({"type": "between_tools"}));
}

/// The instruction still lands after a client's trailing system message.
#[test]
fn the_forced_tool_instruction_follows_a_trailing_system_message() {
    let body = with(
        convo(
            json!([{"role": "user", "content": "x"}, {"role": "system", "content": "Be terse."}]),
        ),
        json!({"tool_choice": "required", "tools": [{"type": "function", "function": {"name": "t", "parameters": {"type": "object", "properties": {}}}}]}),
    );
    let v = c2m(&body, "claude-sonnet-5-5");
    assert_eq!(
        roles(&v),
        ["user", "assistant", "user", "system", "system"],
        "{v}"
    );
    assert!(
        v["messages"][4]["content"]
            .as_str()
            .unwrap()
            .contains("provided tools")
    );
}

// ---- Responses prompt fields (audit 20) ----

#[test]
fn responses_cache_tier_and_verbosity_reach_openai_chat() {
    let body = json!({"model": "m", "store": false, "input": "x", "prompt_cache_key": "k", "service_tier": "flex", "text": {"verbosity": "low"}});
    let v = r2c(&body, "gpt-5");
    assert_eq!(v["prompt_cache_key"], "k");
    assert_eq!(v["service_tier"], "flex");
    assert_eq!(v["verbosity"], "low");
    // Hints elsewhere: dropped, not a 400 on a host that lacks them.
    let v = r2c(&body, "grok-4.6");
    assert!(
        v.get("prompt_cache_key").is_none() && v.get("verbosity").is_none(),
        "{v}"
    );
}

#[test]
fn a_responses_prompt_template_and_file_image_are_forwarded() {
    let v = r2c(
        &json!({"model": "m", "store": false, "prompt": {"id": "pmpt_123", "version": "2"}, "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "describe"}, {"type": "input_image", "file_id": "file_img"}]}
        ]}),
        "gpt-5",
    );
    assert_eq!(v["prompt"]["id"], "pmpt_123");
    assert_eq!(
        v["messages"][0]["content"][1],
        json!({"type": "input_image", "file_id": "file_img"})
    );
}

// ---- Chat Completions / Messages → Responses (Responses-only rows) ----

#[test]
fn chat_onto_responses_maps_tools_choice_and_history_in_order() {
    let body = chat(json!({
        "tools": [
            {"type": "function", "function": {"name": "get_weather", "parameters": weather_schema(), "strict": true}, "cache_control": {"type": "ephemeral"}},
            {"type": "custom", "custom": {"name": "apply_patch", "description": "patch"}}
        ],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        "verbosity": "low",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "developer", "content": [{"type": "text", "text": "Use metric."}]},
            {"role": "user", "content": [{"type": "text", "text": "Paris?", "cache_control": {"type": "ephemeral"}}, {"type": "image_url", "image_url": {"url": "https://x/a.png", "detail": "high"}}]},
            {"role": "assistant", "content": "Checking.", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}},
                {"id": "call_2", "type": "custom", "custom": {"name": "apply_patch", "input": "*** patch"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "text", "text": "18C"}]},
            {"role": "tool", "tool_call_id": "call_2", "content": "applied"},
            {"role": "system", "content": "Answer in French."},
            {"role": "user", "content": "Rome?"}
        ]
    }));
    let v = c2r(&body, "gpt-5-pro");
    assert_eq!(v["instructions"], "Be brief.\n\nUse metric.");
    assert_eq!(
        v["tools"],
        json!([
            {"type": "function", "name": "get_weather", "parameters": weather_schema(), "strict": true},
            {"type": "custom", "name": "apply_patch", "description": "patch"}
        ])
    );
    assert_eq!(
        v["tool_choice"],
        json!({"type": "function", "name": "get_weather"})
    );
    assert_eq!(v["text"]["verbosity"], "low");
    let items = v["input"].as_array().unwrap();
    let kinds: Vec<String> = items
        .iter()
        .map(|i| {
            format!(
                "{}:{}",
                i["type"].as_str().unwrap(),
                i["role"].as_str().unwrap_or("")
            )
        })
        .collect();
    assert_eq!(
        kinds,
        [
            "message:user",
            "message:assistant",
            "function_call:",
            "custom_tool_call:",
            "function_call_output:",
            "custom_tool_call_output:",
            "message:system",
            "message:user"
        ]
    );
    assert_eq!(
        items[0]["content"],
        json!([{"type": "input_text", "text": "Paris?"}, {"type": "input_image", "image_url": "https://x/a.png", "detail": "high"}])
    );
    assert_eq!(items[3]["input"], "*** patch");
    assert_eq!(
        items[4]["output"],
        json!([{"type": "input_text", "text": "18C"}])
    );
}

/// Responses stores every response unless told not to; a Chat client never asked for that.
#[test]
fn chat_onto_responses_does_not_store_by_default() {
    assert_eq!(c2r(&chat(json!({})), "gpt-5-pro")["store"], false);
    assert_eq!(
        c2r(&chat(json!({"store": true})), "gpt-5-pro")["store"],
        true
    );
}

/// A Claude Code-shaped body (thinking in history, `cache_control` everywhere) onto a
/// Responses-only GPT row: another vendor's thinking is not input, and markers are not fields.
#[test]
fn messages_onto_responses_drops_thinking_and_cache_control() {
    let body = anth(json!({
        "system": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
        "tools": [{"name": "t", "input_schema": {"type": "object", "properties": {}}, "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm", "signature": "s"}, {"type": "text", "text": "hello"}]},
            {"role": "user", "content": "again"}
        ]
    }));
    let v = req(Endpoint::Messages, Endpoint::Responses, &body, "gpt-5-pro");
    let s = v.to_string();
    assert!(
        !s.contains("cache_control") && !s.contains("thinking"),
        "{s}"
    );
    assert_eq!(v["instructions"], "sys");
    assert_eq!(
        v["input"][1]["content"],
        json!([{"type": "output_text", "text": "hello"}])
    );
    assert_eq!(v["max_output_tokens"], 300);
}

#[test]
fn chat_only_fields_are_forwarded_onto_responses() {
    let v = c2r(
        &chat(json!({"stop": ["END"], "n": 2, "logprobs": true, "seed": 1})),
        "gpt-5-pro",
    );
    assert_eq!(v["stop"], json!(["END"]));
    assert_eq!(v["n"], 2);
    assert_eq!(v["logprobs"], true);
    assert!(v.get("seed").is_none(), "a hint: {v}");
}

/// The Responses API 400s on `max_output_tokens` under 16; a Chat client may send 1.
#[test]
fn a_tiny_limit_is_raised_to_the_responses_floor() {
    let body = json!({"model": "gpt-5-pro", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]});
    let v: Value = serde_json::from_slice(&request(
        Endpoint::ChatCompletions,
        Endpoint::Responses,
        &serde_json::to_vec(&body).unwrap(),
        "gpt-5-pro",
    ))
    .unwrap();
    assert_eq!(v["max_output_tokens"], 16);
    let body = json!({"model": "gpt-5-pro", "max_tokens": 900, "messages": [{"role": "user", "content": "hi"}]});
    let v: Value = serde_json::from_slice(&request(
        Endpoint::ChatCompletions,
        Endpoint::Responses,
        &serde_json::to_vec(&body).unwrap(),
        "gpt-5-pro",
    ))
    .unwrap();
    assert_eq!(v["max_output_tokens"], 900);
}

// ---- Thinking replay, tool fields, effort (second audit) ----

/// A Chat client that keeps only `reasoning_content` (LiteLLM and most frameworks do) or
/// `reasoning`: that text was never signed, and sent as a thinking block it is a 400 on every later
/// turn ("messages.1.content.0.thinking.signature: Field required", measured). Anthropic takes the
/// turn without its thinking, so the text goes; so does an unsigned thinking part.
#[test]
fn bare_reasoning_text_never_becomes_an_unsigned_thinking_block() {
    for assistant in [
        json!({"role": "assistant", "content": "4", "reasoning_content": "simple"}),
        json!({"role": "assistant", "content": "4", "reasoning": "simple"}),
        json!({"role": "assistant", "content": [
            {"type": "thinking", "thinking": "simple"},
            {"type": "text", "text": "4"},
        ]}),
        json!({"role": "assistant", "content": "4", "thinking": [{"type": "thinking", "thinking": "simple"}]}),
    ] {
        let v = c2m(
            &chat(json!({"messages": [
                {"role": "user", "content": "2+2?"}, assistant.clone(), {"role": "user", "content": "3+3?"},
            ]})),
            "claude-sonnet-4-5",
        );
        let turn = &v["messages"][1]["content"];
        assert!(
            !turn.to_string().contains("\"thinking\""),
            "{assistant} became {turn}"
        );
        assert!(turn.to_string().contains('4'), "{turn}");
    }
}

/// Signed blocks cross whichever way the client carried them: our `thinking` array (what the
/// gateway's Chat responses return), thinking parts, or OpenRouter's `reasoning_details` — a turn a
/// Chat client got relayed from OpenRouter (failover) and echoes onto the Anthropic primary.
#[test]
fn signed_thinking_crosses_from_every_chat_shape() {
    let want = json!([
        {"type": "thinking", "thinking": "simple", "signature": "EqQBSIG"},
        {"type": "redacted_thinking", "data": "RD"},
        {"type": "text", "text": "4"},
    ]);
    for assistant in [
        json!({"role": "assistant", "content": "4", "reasoning_content": "simple", "thinking": [
            {"index": 0, "type": "thinking", "thinking": "simple", "signature": "EqQBSIG"},
            {"index": 1, "type": "redacted_thinking", "data": "RD"},
        ]}),
        json!({"role": "assistant", "content": [
            {"type": "thinking", "thinking": "simple", "signature": "EqQBSIG"},
            {"type": "redacted_thinking", "data": "RD"},
            {"type": "text", "text": "4"},
        ]}),
        json!({"role": "assistant", "content": "4", "reasoning": "simple", "reasoning_details": [
            {"type": "reasoning.text", "text": "simple", "signature": "EqQBSIG", "format": "anthropic-claude-v1", "index": 0},
            {"type": "reasoning.encrypted", "data": "RD", "format": "anthropic-claude-v1", "index": 1},
        ]}),
    ] {
        let v = c2m(
            &chat(json!({"messages": [
                {"role": "user", "content": "2+2?"}, assistant.clone(), {"role": "user", "content": "3+3?"},
            ]})),
            "claude-sonnet-4-5",
        );
        let mut turn = v["messages"][1]["content"].clone();
        for b in turn.as_array_mut().unwrap() {
            b.as_object_mut().unwrap().remove("cache_control");
        }
        assert_eq!(turn, want, "{assistant}");
    }
}

/// OpenRouter replays Claude's thinking only from `reasoning_details`: with the gateway's
/// `thinking` array and `reasoning_content` alone, a thinking + tool loop on an OpenRouter-only
/// Claude row (or any Claude row's OpenRouter failover) 400ed on turn 2 ("a final `assistant`
/// message must start with a thinking block", measured on claude-sonnet-4).
#[test]
fn signed_thinking_rides_reasoning_details_to_claude_on_openrouter() {
    let body = anth(json!({
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "max_tokens": 2000,
        "tools": [{"name": "get_weather", "input_schema": weather_schema()}],
        "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "t", "signature": "SIG"},
                {"type": "redacted_thinking", "data": "RD"},
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}},
            ]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny"}]},
        ],
    }));
    let v = m2c(&body, "anthropic/claude-sonnet-4");
    assert_eq!(
        v["messages"][1]["reasoning_details"],
        json!([
            {"type": "reasoning.text", "text": "t", "signature": "SIG", "format": "anthropic-claude-v1", "index": 0},
            {"type": "reasoning.encrypted", "data": "RD", "format": "anthropic-claude-v1", "index": 1},
        ])
    );
    // Only a Claude model can verify the signature; nobody else is sent it.
    let v = m2c(&body, "gpt-5-nano");
    assert!(v["messages"][1].get("reasoning_details").is_none(), "{v}");
}

/// A Responses client on a Claude row gets each signed thinking block as a `reasoning` item
/// (`rs_gw…`, signature in `encrypted_content`) and sends it back as it came. Dropped, a thinking +
/// tool loop on a model that needs its thinking back (claude-sonnet-4 via OpenRouter) 400ed.
/// OpenAI's own reasoning items (and unsigned ones) mean nothing to Claude and stay dropped.
#[test]
fn gateway_reasoning_items_come_back_as_signed_thinking() {
    let body = json!({"model": "claude-sonnet-4", "store": false, "reasoning": {"effort": "low"},
    "tools": [{"type": "function", "name": "get_weather", "parameters": weather_schema()}],
    "input": [
        {"role": "user", "content": "weather?"},
        {"type": "reasoning", "id": "rs_gw18f2a0001", "summary": [{"type": "summary_text", "text": "t"}], "encrypted_content": "SIG"},
        {"type": "reasoning", "id": "rs_68ab01", "summary": [], "encrypted_content": "gAAAAopenai"},
        {"type": "reasoning", "id": "rs_gw18f2a0002", "summary": [{"type": "summary_text", "text": "unsigned"}]},
        {"type": "function_call", "call_id": "toolu_1", "name": "get_weather", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "toolu_1", "output": "sunny"},
    ]});
    let chat = r2c(&body, "anthropic/claude-sonnet-4");
    let turn = &chat["messages"][1];
    assert_eq!(turn["tool_calls"][0]["id"], "toolu_1");
    assert_eq!(
        turn["reasoning_details"],
        json!([{"type": "reasoning.text", "text": "t", "signature": "SIG", "format": "anthropic-claude-v1", "index": 0}]),
        "{chat}"
    );
    assert!(!chat.to_string().contains("gAAAA"), "{chat}");

    let msgs = req(
        Endpoint::Responses,
        Endpoint::Messages,
        &body,
        "claude-sonnet-4-5",
    );
    let content = &msgs["messages"][1]["content"];
    assert_eq!(
        content[0],
        json!({"type": "thinking", "thinking": "t", "signature": "SIG"}),
        "{msgs}"
    );
    assert_eq!(content[1]["type"], "tool_use");
    assert!(!msgs.to_string().contains("gAAAA") && !msgs.to_string().contains("unsigned"));

    // An OpenAI row gets none of it.
    let v = r2c(&body, "gpt-5-nano");
    assert!(!v.to_string().contains("SIG"), "{v}");
}

/// OpenAI's Chat Completions 400s `tool_choice` and `parallel_tool_calls` without `tools` ("only
/// allowed when 'tools' are specified", measured); Messages and Responses accept both.
#[test]
fn tool_choice_and_parallel_tool_calls_go_to_chat_only_with_tools() {
    let a = m2c(
        &anth(json!({"tool_choice": {"type": "auto", "disable_parallel_tool_use": true}})),
        "gpt-5-mini",
    );
    assert!(
        a.get("tool_choice").is_none() && a.get("parallel_tool_calls").is_none(),
        "{a}"
    );
    let r = r2c(
        &json!({"model": "m", "input": "hi", "store": false, "tool_choice": "auto", "parallel_tool_calls": false}),
        "gpt-5-mini",
    );
    assert!(
        r.get("tool_choice").is_none() && r.get("parallel_tool_calls").is_none(),
        "{r}"
    );
    // With tools, both still cross.
    let tools = json!([{"name": "get_weather", "input_schema": weather_schema()}]);
    let a = m2c(
        &anth(
            json!({"tools": tools, "tool_choice": {"type": "auto", "disable_parallel_tool_use": true}}),
        ),
        "gpt-5-mini",
    );
    assert_eq!(a["tool_choice"], "auto");
    assert_eq!(a["parallel_tool_calls"], false);
    let r = r2c(
        &json!({"model": "m", "input": "hi", "store": false, "tool_choice": "required", "parallel_tool_calls": false,
            "tools": [{"type": "function", "name": "get_weather", "parameters": weather_schema()}]}),
        "gpt-5-mini",
    );
    assert_eq!(r["tool_choice"], "required");
    assert_eq!(r["parallel_tool_calls"], false);
}

/// Sonnet 5.5 rejects `thinking: disabled`, and an omitted `thinking` is adaptive thinking, so
/// `reasoning_effort: "none"` thought anyway. Its "off" is `between_tools` at effort `high` or
/// below, alone in the `thinking` object — unless the history holds thinking on a request that
/// carries `block_binding`, which `between_tools` rejects: that one stays adaptive at `low`.
#[test]
fn reasoning_none_on_sonnet_5_5_is_between_tools() {
    for model in ["claude-sonnet-5-5", "us.anthropic.claude-sonnet-5-5-v1:0"] {
        let v = c2m(&chat(json!({"reasoning_effort": "none"})), model);
        assert_eq!(
            v["thinking"],
            json!({"type": "between_tools"}),
            "{model}: {v}"
        );
        assert_eq!(v["output_config"]["effort"], "low", "{model}: {v}");
        let v = c2m(&chat(json!({"thinking": {"type": "disabled"}})), model);
        assert_eq!(
            v["thinking"],
            json!({"type": "between_tools"}),
            "{model}: {v}"
        );
    }
    let history = chat(json!({"reasoning_effort": "none", "messages": [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "y", "thinking": [{"type": "thinking", "thinking": "", "signature": "SIG"}]},
        {"role": "user", "content": "z"},
    ]}));
    let v = c2m(&history, "claude-sonnet-5-5");
    assert_eq!(v["thinking"]["type"], "adaptive", "{v}");
    assert_eq!(
        v["thinking"]["block_binding"]["prefix_mismatch_behavior"],
        "drop_block"
    );
    assert_eq!(v["output_config"]["effort"], "low");
    // Bedrock carries no `block_binding`, so nothing stands in the way there.
    let v = c2m(&history, "us.anthropic.claude-sonnet-5-5-v1:0");
    assert_eq!(v["thinking"], json!({"type": "between_tools"}), "{v}");
    // Models without `between_tools` keep the omitted-thinking shape.
    let v = c2m(
        &chat(json!({"reasoning_effort": "none"})),
        "claude-opus-4-8",
    );
    assert!(v.get("thinking").is_none(), "{v}");
    assert_eq!(v["output_config"]["effort"], "low");
}

/// A large Anthropic budget (Claude Code's 31999) or `effort: max` maps to `xhigh`, which only
/// OpenAI's newer families define; a host the table does not know gets `high`. gpt-oss, on any
/// host, always reasons (`none` is a 400 there, measured on OpenRouter).
#[test]
fn unknown_hosts_get_the_classic_efforts() {
    let big =
        anth(json!({"max_tokens": 32000, "thinking": {"type": "enabled", "budget_tokens": 31999}}));
    for model in [
        "x-ai/grok-4.6",
        "grok-4.6",
        "deepseek-chat",
        "moonshotai/kimi-k3",
    ] {
        assert_eq!(m2c(&big, model)["reasoning_effort"], "high", "{model}");
    }
    let max = anth(json!({"output_config": {"effort": "max"}}));
    assert_eq!(m2c(&max, "deepseek-chat")["reasoning_effort"], "high");
    let off = anth(json!({"thinking": {"type": "disabled"}}));
    for model in [
        "openai/gpt-oss-120b",
        "accounts/fireworks/models/gpt-oss-120b",
    ] {
        assert_eq!(m2c(&off, model)["reasoning_effort"], "low", "{model}");
        assert_eq!(m2c(&big, model)["reasoning_effort"], "high", "{model}");
        let v = m2c(&anth(json!({"temperature": 0.3})), model);
        assert_eq!(v["max_tokens"], 300, "not OpenAI's own API: {v}");
        assert_eq!(v["temperature"], 0.3, "{v}");
    }
    let r =
        json!({"model": "m", "store": false, "input": "hi", "reasoning": {"effort": "minimal"}});
    assert_eq!(r2c(&r, "x-ai/grok-4.6")["reasoning_effort"], "low");
}

// ---- verification phase 0: thinking + tools for clients that drop thinking ---------------------

/// Anthropic's rule for a tool loop with thinking on: "a final `assistant` message must start with
/// a thinking block (preceding the lastmost set of `tool_use` and `tool_result` blocks)". A body
/// passes when thinking is off (absent or `disabled`) or the last assistant turn opens with a
/// `thinking` / `redacted_thinking` block. `None` means Anthropic accepts it; `Some` says why not.
fn tool_loop_thinking_violation(v: &Value) -> Option<String> {
    let thinking_on = matches!(
        v.pointer("/thinking/type").and_then(Value::as_str),
        Some("enabled" | "adaptive")
    );
    if !thinking_on {
        return None;
    }
    let last = v["messages"]
        .as_array()?
        .iter()
        .rev()
        .find(|m| m["role"] == "assistant")?;
    let blocks = last["content"].as_array()?;
    let has_tool_use = blocks.iter().any(|b| b["type"] == "tool_use");
    let opens_with_thinking = matches!(
        blocks.first().and_then(|b| b["type"].as_str()),
        Some("thinking" | "redacted_thinking")
    );
    (has_tool_use && !opens_with_thinking).then(|| {
        format!(
            "thinking is {} but the final assistant turn starts with {}: {v}",
            v["thinking"], blocks[0]["type"]
        )
    })
}

/// Turn 2 of a thinking + tool loop on budget-thinking Claude, from clients that do not send our
/// thinking back: Vercel / LangChain on Chat Completions (the assistant echo carries the tool call,
/// maybe `reasoning_content`, never our `thinking` list), a Responses client that drops reasoning
/// items, and Codex / the Agents SDK replaying the reasoning item without its `id` (the gateway
/// only recognises its own `rs_gw…` ids). Each must become a body Anthropic accepts: replay a
/// verifiable block, or turn thinking off for the request.
/// claim: TRN-7, TRN-8
/// defect: D14
#[test]
#[ignore = "D14 reproduced: thinking stays enabled while the final assistant tool turn has no thinking block"]
fn a_tool_turn_without_echoed_thinking_is_still_accepted_on_budget_claude() {
    const MODEL: &str = "claude-sonnet-4-5";
    assert_eq!(ClaudeModel::of(MODEL).reasoning, ClaudeGen::Budget);
    let mut bad = Vec::new();
    let turn2 = |assistant: Value| {
        chat(json!({
            "reasoning_effort": "high",
            "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": weather_schema()}}],
            "messages": [
                {"role": "user", "content": "weather in Paris?"},
                assistant,
                {"role": "tool", "tool_call_id": "toolu_01", "content": "sunny"},
            ],
        }))
    };

    // Control: a client that echoes our signed `thinking` list already produces a valid body, so
    // the check below is not vacuous.
    let echoed = c2m(
        &turn2(json!({"role": "assistant", "content": null,
            "thinking": [{"type": "thinking", "thinking": "Need weather.", "signature": "SIG"}],
            "tool_calls": [{"id": "toolu_01", "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}]})),
        MODEL,
    );
    assert_eq!(echoed["thinking"]["type"], "enabled", "{echoed}");
    assert_eq!(tool_loop_thinking_violation(&echoed), None);

    // Chat Completions: the echo a stock SDK rebuilds (tool call, reasoning text, no `thinking`).
    for assistant in [
        json!({"role": "assistant", "content": null, "tool_calls": [{"id": "toolu_01", "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}]}),
        json!({"role": "assistant", "content": "", "reasoning_content": "Need weather.",
            "tool_calls": [{"id": "toolu_01", "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}]}),
    ] {
        if let Some(why) = tool_loop_thinking_violation(&c2m(&turn2(assistant), MODEL)) {
            bad.push(format!("Chat→Messages: {why}"));
        }
    }

    // Responses: reasoning dropped, and reasoning replayed without its id.
    let call = json!({"type": "function_call", "call_id": "toolu_01", "name": "get_weather",
        "arguments": "{\"city\":\"Paris\"}"});
    let output = json!({"type": "function_call_output", "call_id": "toolu_01", "output": "sunny"});
    let id_less = json!({"type": "reasoning", "encrypted_content": "EqQBsig==",
        "summary": [{"type": "summary_text", "text": "Need weather."}]});
    for (what, input) in [
        (
            "no reasoning item",
            json!([{"role": "user", "content": "weather in Paris?"}, call, output]),
        ),
        (
            "id-less reasoning item",
            json!([{"role": "user", "content": "weather in Paris?"}, id_less, call, output]),
        ),
    ] {
        let body = json!({
            "model": "m", "store": false, "max_output_tokens": 4000, "reasoning": {"effort": "high"},
            "tools": [{"type": "function", "name": "get_weather", "parameters": weather_schema()}],
            "input": input,
        });
        let v = req(Endpoint::Responses, Endpoint::Messages, &body, MODEL);
        if let Some(why) = tool_loop_thinking_violation(&v) {
            bad.push(format!("Responses→Messages ({what}): {why}"));
        }
    }

    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

// ---- verification phase 0: translation detail (D44-D50) ----------------------------------------

/// Two consecutive user messages onto Messages, which wants one user turn: the turn may merge, but
/// each message keeps its own text block (or a separator), never `"Hello" + "World"` fused into one
/// word.
/// claim: TRN-3
/// defect: D44
#[test]
fn consecutive_user_messages_keep_their_text_apart() {
    let v = c2m(
        &chat(json!({"messages": [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "World"},
        ]})),
        "claude-haiku-4-5",
    );
    let texts: Vec<String> = v["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|m| m["role"] == "user")
        .flat_map(|m| match &m["content"] {
            Value::String(s) => vec![s.clone()],
            Value::Array(bs) => bs
                .iter()
                .filter_map(|b| b["text"].as_str().map(str::to_owned))
                .collect(),
            _ => Vec::new(),
        })
        .collect();
    assert!(
        !texts.iter().any(|t| t.contains("HelloWorld")),
        "the two messages fused: {v}"
    );
    assert!(
        texts.iter().any(|t| t.contains("Hello")) && texts.iter().any(|t| t.contains("World")),
        "{v}"
    );
}

/// A Chat client's `cache_control` on a whole assistant or tool message (the same message-level
/// marker the gateway honours on user and system messages) must reach the Messages block it
/// becomes. Placed mid-history, so the automatic last-message breakpoint cannot stand in for it.
/// claim: TRN-4
/// defect: D47
#[test]
fn message_level_cache_control_survives_on_assistant_and_tool_messages() {
    let cc = json!({"type": "ephemeral"});
    let v = c2m(
        &chat(json!({
            "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": weather_schema()}}],
            "messages": [
                {"role": "user", "content": "weather in Paris?"},
                {"role": "assistant", "content": "Let me check.", "cache_control": cc,
                 "tool_calls": [{"id": "toolu_01", "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}]},
                {"role": "tool", "tool_call_id": "toolu_01", "content": "sunny", "cache_control": cc},
                {"role": "assistant", "content": "It is sunny."},
                {"role": "user", "content": "thanks"},
            ],
        })),
        "claude-haiku-4-5",
    );
    let msgs = v["messages"].as_array().unwrap();
    let marked = |m: &Value| {
        m["content"].as_array().is_some_and(|bs| {
            bs.iter().any(|b| {
                b.get("cache_control") == Some(&cc)
                    || b["content"].as_array().is_some_and(|inner| {
                        inner.iter().any(|x| x.get("cache_control") == Some(&cc))
                    })
            })
        })
    };
    assert!(
        marked(&msgs[1]),
        "assistant message lost its cache_control: {v}"
    );
    assert!(marked(&msgs[2]), "tool message lost its cache_control: {v}");
}

/// A custom (free-form) tool call in Chat or Responses history, onto Messages: the `tool_use`
/// keeps the tool's name and its input. `name: ""` is a 400, and `input: {}` erases what the model
/// wrote.
/// claim: TRN-14
/// defect: D48
#[test]
fn a_custom_tool_call_in_history_keeps_its_name_and_input_on_messages() {
    let patch = "*** Begin Patch\n*** End Patch";
    let chat_body = chat(json!({
        "tools": [{"type": "custom", "custom": {"name": "apply_patch", "format": {"type": "text"}}}],
        "messages": [
            {"role": "user", "content": "fix it"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "custom", "custom": {"name": "apply_patch", "input": patch}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "applied"},
        ],
    }));
    let resp_body = json!({
        "model": "m", "store": false, "max_output_tokens": 4000,
        "tools": [{"type": "custom", "name": "apply_patch", "format": {"type": "text"}}],
        "input": [
            {"role": "user", "content": "fix it"},
            {"type": "custom_tool_call", "call_id": "call_1", "name": "apply_patch", "input": patch},
            {"type": "custom_tool_call_output", "call_id": "call_1", "output": "applied"},
        ],
    });
    let mut bad = Vec::new();
    for (what, v) in [
        ("Chat→Messages", c2m(&chat_body, "claude-haiku-4-5")),
        (
            "Responses→Messages",
            req(
                Endpoint::Responses,
                Endpoint::Messages,
                &resp_body,
                "claude-haiku-4-5",
            ),
        ),
    ] {
        let tool_use = v["messages"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|m| m["content"].as_array())
            .flatten()
            .find(|b| b["type"] == "tool_use")
            .cloned()
            .unwrap_or(Value::Null);
        if tool_use["name"] != "apply_patch"
            || !tool_use["input"].to_string().contains("Begin Patch")
        {
            bad.push(format!("{what}: {tool_use}"));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// The tool message answering a custom call, onto a Responses row, is a
/// `custom_tool_call_output`: OpenAI rejects a `function_call_output` whose call was a
/// `custom_tool_call`.
/// claim: TRN-14
/// defect: D48
#[test]
fn a_custom_tool_result_reaches_responses_as_custom_tool_call_output() {
    let v = c2r(
        &chat(json!({"messages": [
            {"role": "user", "content": "fix it"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "custom", "custom": {"name": "apply_patch", "input": "*** Begin Patch"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "applied"},
        ]})),
        "gpt-5",
    );
    let types: Vec<&str> = v["input"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|i| i["type"].as_str())
        .collect();
    assert!(types.contains(&"custom_tool_call"), "{v}");
    assert!(
        types.contains(&"custom_tool_call_output") && !types.contains(&"function_call_output"),
        "{types:?}: {v}"
    );
}

/// OpenAI SDKs and frameworks send unset optional fields as explicit `null`. A null means "not
/// set", so it must not reach an upstream that has no such field (Messages has no `audio`,
/// Responses has no `stop`): forwarding it trades a working request for a 400 by name.
/// claim: TRN-15
/// defect: D49
#[test]
fn explicit_nulls_are_not_forwarded_upstream() {
    let nulls = json!({"stop": null, "audio": null, "top_logprobs": null,
        "web_search_options": null});
    let mut bad = Vec::new();
    for (what, v) in [
        (
            "Chat→Messages",
            c2m(&chat(nulls.clone()), "claude-haiku-4-5"),
        ),
        ("Chat→Responses", c2r(&chat(nulls), "gpt-5")),
    ] {
        for key in ["stop", "audio", "top_logprobs", "web_search_options"] {
            if v.get(key).is_some_and(Value::is_null) {
                bad.push(format!("{what}: {key}: null forwarded"));
            }
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// A Responses `developer` message onto a non-OpenAI Chat Completions host (DeepSeek, Mistral,
/// most OpenAI-compatible servers) becomes `system`: those hosts only know system / user /
/// assistant / tool.
/// claim: TRN-16
/// defect: D49
#[test]
fn a_responses_developer_message_is_system_on_a_non_openai_chat_host() {
    let v = r2c(
        &json!({"model": "m", "store": false, "input": [
            {"role": "user", "content": "hi"},
            {"role": "developer", "content": "Answer in French."},
            {"role": "user", "content": "how are you?"},
        ]}),
        "deepseek-chat",
    );
    let r = roles(&v);
    assert!(!r.contains(&"developer"), "{r:?}: {v}");
}

/// Responses input items the gateway has no mapping for must reach the upstream so it rejects
/// them by name, never silently vanish: an `item_reference` is the turn's content, and a dropped
/// `computer_call_output` / `local_shell_call` leaves the model answering a different history.
/// claim: TRN-17
/// defect: D49
#[test]
fn unknown_responses_input_items_are_forwarded_not_dropped() {
    let mut dropped = Vec::new();
    for item in [
        json!({"type": "item_reference", "id": "msg_abc123"}),
        json!({"type": "local_shell_call", "id": "lsh_1", "call_id": "call_1", "status": "completed",
            "action": {"type": "exec", "command": ["ls"], "env": {}}}),
        json!({"type": "computer_call_output", "call_id": "call_2",
            "output": {"type": "computer_screenshot", "image_url": "data:image/png;base64,AAAA"}}),
        json!({"type": "compaction", "id": "cmp_1", "encrypted_content": "opaque"}),
    ] {
        let typ = item["type"].as_str().unwrap().to_owned();
        let v = r2c(
            &json!({"model": "m", "store": false, "input": [
                {"role": "user", "content": "hi"}, item,
            ]}),
            "gpt-4o-mini",
        );
        if !v.to_string().contains(&typ) {
            dropped.push(typ);
        }
    }
    assert!(dropped.is_empty(), "silently dropped: {dropped:?}");
}
