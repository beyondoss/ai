//! Request generators, one per client dialect, and what a translation must carry across.

use super::*;

/// The upstream model ids translation decides per-model behavior on (thinking, sampling, limits).
pub const MODELS: &[&str] = &[
    "claude-opus-4-8",
    "claude-sonnet-4-5",
    "claude-sonnet-4-5-20250929",
    "claude-3-5-haiku-20241022",
    "claude-haiku-4-5",
    "anthropic/claude-sonnet-4.5",
    "gpt-5",
    "gpt-4o",
    "gpt-4.1-nano",
    "o3",
    "grok-4",
    "deepseek-chat",
    "",
];

/// What every translation of a request must still say: the texts the model reads, the tools it
/// was offered, the calls it made and the base64 payloads it was shown.
#[derive(Debug, Default, Clone)]
pub struct Carried {
    pub texts: Vec<String>,
    pub tools: Vec<String>,
    pub calls: Vec<(String, Value)>,
    pub payloads: Vec<String>,
}

fn opt_null(v: Value, null: bool) -> Value {
    if null { Value::Null } else { v }
}

fn cache_control() -> impl Strategy<Value = Option<Value>> {
    prop::option::weighted(
        0.2,
        prop_oneof![
            Just(json!({ "type": "ephemeral" })),
            Just(json!({ "type": "ephemeral", "ttl": "1h" })),
        ],
    )
}

fn b64() -> impl Strategy<Value = String> {
    "[A-Za-z0-9+/]{4,40}={0,2}"
}

fn schema() -> impl Strategy<Value = Value> {
    prop_oneof![
        Just(json!({ "type": "object", "properties": {} })),
        Just(
            json!({ "type": "object", "properties": { "q": { "type": "string" } }, "required": ["q"] })
        ),
        json_object(2).prop_map(|props| json!({ "type": "object", "properties": props })),
    ]
}

// --- Chat Completions -----------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ChatReq {
    pub body: Value,
    pub carried: Carried,
}

fn chat_user_part() -> impl Strategy<Value = (Value, Option<String>, Option<String>)> {
    prop_oneof![
        5 => (text1(6), cache_control()).prop_map(|(t, cc)| {
            let mut p = json!({ "type": "text", "text": t });
            if let Some(cc) = cc { p["cache_control"] = cc; }
            (p, Some(t), None)
        }),
        1 => b64().prop_map(|d| (json!({ "type": "image_url", "image_url": { "url": format!("data:image/png;base64,{d}"), "detail": "auto" } }), None, Some(d))),
        1 => Just((json!({ "type": "image_url", "image_url": { "url": "https://example.com/cat.png" } }), None, None)),
        1 => b64().prop_map(|d| (json!({ "type": "file", "file": { "filename": "a.pdf", "file_data": format!("data:application/pdf;base64,{d}") } }), None, Some(d))),
        1 => b64().prop_map(|d| (json!({ "type": "input_audio", "input_audio": { "data": d, "format": "wav" } }), None, None)),
    ]
}

/// One Chat message and what it carries.
fn chat_message() -> impl Strategy<Value = (Value, Carried)> {
    prop_oneof![
        // system / developer
        2 => (prop::sample::select(vec!["system", "developer"]), text1(6), any::<bool>(), cache_control()).prop_map(|(role, t, parts, cc)| {
            let content = if parts {
                let mut p = json!({ "type": "text", "text": t });
                if let Some(cc) = cc { p["cache_control"] = cc; }
                json!([p])
            } else { json!(t) };
            (json!({ "role": role, "content": content }), Carried { texts: vec![t], ..Carried::default() })
        }),
        // user, string
        3 => (text1(8), cache_control()).prop_map(|(t, cc)| {
            let mut m = json!({ "role": "user", "content": t });
            if let Some(cc) = cc { m["cache_control"] = cc; }
            (m, Carried { texts: vec![t], ..Carried::default() })
        }),
        // user, parts
        3 => prop::collection::vec(chat_user_part(), 1..4).prop_map(|parts| {
            let mut c = Carried::default();
            let mut ps = Vec::new();
            for (p, t, d) in parts {
                ps.push(p);
                c.texts.extend(t);
                c.payloads.extend(d);
            }
            (json!({ "role": "user", "content": ps }), c)
        }),
        // assistant: text and/or tool calls
        3 => (
            prop::option::of(text1(6)),
            prop::collection::vec((up_id("call_"), tool_name(), json_object(2)), 0..3),
            any::<bool>(),
            prop::option::of(("[A-Za-z0-9]{8,20}", text1(4))),
        ).prop_map(|(t, calls, null_content, thinking)| {
            let mut c = Carried::default();
            let content = match &t {
                Some(t) => { c.texts.push(t.clone()); json!(t) }
                None => opt_null(json!(""), null_content),
            };
            let mut m = json!({ "role": "assistant", "content": content });
            if !calls.is_empty() {
                m["tool_calls"] = Value::Array(calls.iter().map(|(id, name, args)| json!({
                    "id": id, "type": "function", "function": { "name": name, "arguments": args.to_string() },
                })).collect());
                c.calls = calls.into_iter().map(|(_, n, a)| (n, a)).collect();
            }
            if let Some((sig, text)) = thinking {
                m["thinking"] = json!([{ "type": "thinking", "thinking": text, "signature": sig }]);
            }
            (m, c)
        }),
        // tool result
        2 => (up_id("call_"), text1(6), any::<bool>()).prop_map(|(id, t, parts)| {
            let content = if parts { json!([{ "type": "text", "text": t }]) } else { json!(t) };
            (json!({ "role": "tool", "tool_call_id": id, "content": content }), Carried { texts: vec![t], ..Carried::default() })
        }),
    ]
}

pub fn chat_request() -> impl Strategy<Value = ChatReq> {
    (
        prop::collection::vec(chat_message(), 1..7),
        prop::collection::vec((tool_name(), prop::option::of(text(4)), schema(), prop::option::of(any::<bool>())), 0..4),
        prop::option::of(prop_oneof![
            Just(json!("auto")), Just(json!("none")), Just(json!("required")), Just(Value::Null),
        ]),
        prop::option::of(prop_oneof![Just(json!(null)), (1u64..300_000).prop_map(|n| json!(n))]),
        prop::option::of(any::<bool>()),
        prop::option::of(prop_oneof![
            Just(json!("low")), Just(json!("medium")), Just(json!("high")), Just(json!("minimal")), Just(Value::Null),
        ]),
        prop::option::of(prop_oneof![Just(json!(0.7)), Just(json!(2.0)), Just(Value::Null)]),
        prop::option::of(prop_oneof![
            Just(json!("STOP")), Just(json!(["a", "b"])), Just(Value::Null),
        ]),
        prop::option::of(prop_oneof![
            Just(json!({ "type": "json_schema", "json_schema": { "name": "out", "schema": { "type": "object", "properties": {} } } })),
            Just(json!({ "type": "json_object" })),
            Just(Value::Null),
        ]),
        prop::option::of(prop_oneof![Just(json!("user-1")), Just(json!("a@b.co")), Just(Value::Null)]),
    )
        .prop_map(|(msgs, tools, choice, max, stream, effort, temp, stop, rf, user)| {
            let mut carried = Carried::default();
            let mut messages = Vec::new();
            for (m, c) in msgs {
                messages.push(m);
                carried.texts.extend(c.texts);
                carried.calls.extend(c.calls);
                carried.payloads.extend(c.payloads);
            }
            let mut body = json!({ "model": "client-model", "messages": messages });
            let mut seen = std::collections::HashSet::new();
            let tools: Vec<_> = tools.into_iter().filter(|t| seen.insert(t.0.clone())).collect();
            if !tools.is_empty() {
                body["tools"] = Value::Array(tools.iter().map(|(name, desc, params, strict)| {
                    let mut f = json!({ "name": name, "parameters": params });
                    if let Some(d) = desc { f["description"] = json!(d); }
                    if let Some(s) = strict { f["strict"] = json!(s); }
                    json!({ "type": "function", "function": f })
                }).collect());
                carried.tools = tools.into_iter().map(|t| t.0).collect();
            }
            let mut set = |k: &str, v: Option<Value>| {
                if let Some(v) = v { body[k] = v; }
            };
            set("tool_choice", choice);
            set("max_tokens", max);
            set("stream", stream.map(Value::Bool));
            set("reasoning_effort", effort);
            set("temperature", temp);
            set("stop", stop);
            set("response_format", rf);
            set("user", user);
            ChatReq { body, carried }
        })
}

// --- Messages -------------------------------------------------------------------------------

fn ant_user_block() -> impl Strategy<Value = (Value, Carried)> {
    prop_oneof![
        4 => (text1(6), cache_control()).prop_map(|(t, cc)| {
            let mut b = json!({ "type": "text", "text": t });
            if let Some(cc) = cc { b["cache_control"] = cc; }
            (b, Carried { texts: vec![t], ..Carried::default() })
        }),
        1 => b64().prop_map(|d| (json!({ "type": "image", "source": { "type": "base64", "media_type": "image/png", "data": d } }), Carried { payloads: vec![d], ..Carried::default() })),
        1 => b64().prop_map(|d| (json!({ "type": "document", "source": { "type": "base64", "media_type": "application/pdf", "data": d } }), Carried { payloads: vec![d], ..Carried::default() })),
        2 => (up_id("toolu_"), text1(6), any::<bool>(), any::<bool>()).prop_map(|(id, t, blocks, is_error)| {
            let content = if blocks { json!([{ "type": "text", "text": t }]) } else { json!(t) };
            let mut b = json!({ "type": "tool_result", "tool_use_id": id, "content": content });
            if is_error { b["is_error"] = json!(true); }
            (b, Carried { texts: vec![t], ..Carried::default() })
        }),
    ]
}

fn ant_assistant_block() -> impl Strategy<Value = (Value, Carried)> {
    prop_oneof![
        3 => text1(6).prop_map(|t| (json!({ "type": "text", "text": t }), Carried { texts: vec![t], ..Carried::default() })),
        1 => (text1(4), "[A-Za-z0-9]{8,20}").prop_map(|(t, sig)| (json!({ "type": "thinking", "thinking": t, "signature": sig }), Carried::default())),
        1 => "[A-Za-z0-9]{8,20}".prop_map(|d| (json!({ "type": "redacted_thinking", "data": d }), Carried::default())),
        2 => (up_id("toolu_"), tool_name(), json_object(2)).prop_map(|(id, name, input)| {
            (json!({ "type": "tool_use", "id": id, "name": name, "input": input }), Carried { calls: vec![(name, input)], ..Carried::default() })
        }),
    ]
}

#[derive(Debug, Clone)]
pub struct MessagesReq {
    pub body: Value,
    pub carried: Carried,
}

pub fn messages_request() -> impl Strategy<Value = MessagesReq> {
    let turn = prop_oneof![
        (text1(8), any::<bool>()).prop_map(|(t, _)| (
            json!({ "role": "user", "content": t }),
            Carried {
                texts: vec![t],
                ..Carried::default()
            }
        )),
        prop::collection::vec(ant_user_block(), 1..4).prop_map(|bs| merge("user", bs)),
        prop::collection::vec(ant_assistant_block(), 1..4).prop_map(|bs| merge("assistant", bs)),
    ];
    (
        prop::option::of((text1(6), any::<bool>(), cache_control())),
        prop::collection::vec(turn, 1..6),
        prop::collection::vec((tool_name(), prop::option::of(text(4)), schema()), 0..4),
        1u64..200_000,
        prop::option::of(any::<bool>()),
        prop::option::of(prop_oneof![
            Just(json!({ "type": "enabled", "budget_tokens": 2048 })),
            Just(json!({ "type": "adaptive" })),
            Just(json!({ "type": "disabled" })),
        ]),
        prop::option::of(prop_oneof![
            Just(json!({ "type": "auto" })),
            Just(json!({ "type": "any" })),
            Just(json!({ "type": "none" })),
        ]),
    )
        .prop_map(|(system, turns, tools, max, stream, thinking, choice)| {
            let mut carried = Carried::default();
            let mut body = json!({ "model": "client-model", "max_tokens": max });
            if let Some((t, blocks, cc)) = system {
                body["system"] = if blocks {
                    let mut b = json!({ "type": "text", "text": t });
                    if let Some(cc) = cc {
                        b["cache_control"] = cc;
                    }
                    json!([b])
                } else {
                    json!(t)
                };
                carried.texts.push(t);
            }
            let mut messages = Vec::new();
            for (m, c) in turns {
                messages.push(m);
                carried.texts.extend(c.texts);
                carried.calls.extend(c.calls);
                carried.payloads.extend(c.payloads);
            }
            body["messages"] = Value::Array(messages);
            let mut seen = std::collections::HashSet::new();
            let tools: Vec<_> = tools
                .into_iter()
                .filter(|t| seen.insert(t.0.clone()))
                .collect();
            if !tools.is_empty() {
                body["tools"] = Value::Array(
                    tools
                        .iter()
                        .map(|(name, desc, schema)| {
                            let mut t = json!({ "name": name, "input_schema": schema });
                            if let Some(d) = desc {
                                t["description"] = json!(d);
                            }
                            t
                        })
                        .collect(),
                );
                carried.tools = tools.into_iter().map(|t| t.0).collect();
            }
            if let Some(s) = stream {
                body["stream"] = json!(s);
            }
            if let Some(t) = thinking {
                body["thinking"] = t;
            }
            if let Some(c) = choice {
                body["tool_choice"] = c;
            }
            MessagesReq { body, carried }
        })
}

fn merge(role: &str, blocks: Vec<(Value, Carried)>) -> (Value, Carried) {
    let mut c = Carried::default();
    let mut bs = Vec::new();
    for (b, x) in blocks {
        bs.push(b);
        c.texts.extend(x.texts);
        c.calls.extend(x.calls);
        c.payloads.extend(x.payloads);
    }
    (json!({ "role": role, "content": bs }), c)
}

// --- Responses ------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ResponsesReq {
    pub body: Value,
    pub carried: Carried,
}

fn responses_item() -> impl Strategy<Value = (Value, Carried)> {
    prop_oneof![
        3 => (prop::sample::select(vec!["user", "system", "developer"]), text1(6), any::<bool>()).prop_map(|(role, t, parts)| {
            let content = if parts { json!([{ "type": "input_text", "text": t }]) } else { json!(t) };
            (json!({ "type": "message", "role": role, "content": content }), Carried { texts: vec![t], ..Carried::default() })
        }),
        1 => b64().prop_map(|d| (json!({ "role": "user", "content": [{ "type": "input_image", "image_url": format!("data:image/png;base64,{d}") }] }), Carried { payloads: vec![d], ..Carried::default() })),
        2 => text1(6).prop_map(|t| (json!({ "type": "message", "role": "assistant", "content": [{ "type": "output_text", "text": t }] }), Carried { texts: vec![t], ..Carried::default() })),
        2 => (up_id("call_"), tool_name(), json_object(2)).prop_map(|(id, name, args)| {
            (json!({ "type": "function_call", "call_id": id, "name": name, "arguments": args.to_string() }), Carried { calls: vec![(name, args)], ..Carried::default() })
        }),
        2 => (up_id("call_"), text1(6)).prop_map(|(id, t)| {
            (json!({ "type": "function_call_output", "call_id": id, "output": t }), Carried { texts: vec![t], ..Carried::default() })
        }),
    ]
}

pub fn responses_request() -> impl Strategy<Value = ResponsesReq> {
    (
        prop::option::of(text1(6)),
        prop_oneof![
            1 => text1(8).prop_map(|t| (json!(t), Carried { texts: vec![t], ..Carried::default() })),
            4 => prop::collection::vec(responses_item(), 1..6).prop_map(|items| {
                let mut c = Carried::default();
                let mut v = Vec::new();
                for (i, x) in items { v.push(i); c.texts.extend(x.texts); c.calls.extend(x.calls); c.payloads.extend(x.payloads); }
                (Value::Array(v), c)
            }),
        ],
        prop::collection::vec((tool_name(), prop::option::of(text(4)), schema()), 0..4),
        prop::option::of(1u64..200_000),
        prop::option::of(any::<bool>()),
        prop::option::of(prop_oneof![Just(json!({ "effort": "high" })), Just(json!({ "effort": "low", "summary": "auto" }))]),
    )
        .prop_map(|(instructions, (input, c), tools, max, stream, reasoning)| {
            let mut carried = c;
            let mut body = json!({ "model": "client-model", "input": input });
            if let Some(i) = instructions {
                body["instructions"] = json!(i);
                carried.texts.push(i);
            }
            let mut seen = std::collections::HashSet::new();
            let tools: Vec<_> = tools.into_iter().filter(|t| seen.insert(t.0.clone())).collect();
            if !tools.is_empty() {
                body["tools"] = Value::Array(tools.iter().map(|(name, desc, schema)| {
                    let mut t = json!({ "type": "function", "name": name, "parameters": schema });
                    if let Some(d) = desc { t["description"] = json!(d); }
                    t
                }).collect());
                carried.tools = tools.into_iter().map(|t| t.0).collect();
            }
            if let Some(m) = max { body["max_output_tokens"] = json!(m); }
            if let Some(s) = stream { body["stream"] = json!(s); }
            if let Some(r) = reasoning { body["reasoning"] = r; }
            ResponsesReq { body, carried }
        })
}

// --- reading a translated request -----------------------------------------------------------

/// Every string value in `v`, recursively.
pub fn strings(v: &Value, out: &mut Vec<String>) {
    match v {
        Value::String(s) => out.push(s.clone()),
        Value::Array(a) => a.iter().for_each(|x| strings(x, out)),
        Value::Object(m) => m.values().for_each(|x| strings(x, out)),
        _ => {}
    }
}

/// The tool calls a translated request carries, in order: (name, arguments as JSON).
pub fn calls_in(v: &Value, out: &mut Vec<(String, Value)>) {
    match v {
        Value::Object(m) => {
            match m.get("type").and_then(Value::as_str) {
                Some("tool_use") => out.push((
                    m.get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_owned(),
                    m.get("input").cloned().unwrap_or(Value::Null),
                )),
                Some("function_call") => out.push((
                    m.get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_owned(),
                    m.get("arguments")
                        .and_then(Value::as_str)
                        .and_then(|a| serde_json::from_str(a).ok())
                        .unwrap_or(Value::Null),
                )),
                Some("function")
                    if m.contains_key("id")
                        || m.get("function")
                            .is_some_and(|f| f.get("arguments").is_some()) =>
                {
                    let f = m.get("function").unwrap_or(&Value::Null);
                    out.push((
                        f.get("name")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_owned(),
                        f.get("arguments")
                            .and_then(Value::as_str)
                            .and_then(|a| serde_json::from_str(a).ok())
                            .unwrap_or(Value::Null),
                    ));
                    return;
                }
                _ => {}
            }
            for (k, x) in m {
                if k != "tools" {
                    calls_in(x, out);
                }
            }
        }
        Value::Array(a) => a.iter().for_each(|x| calls_in(x, out)),
        _ => {}
    }
}

/// The tool names a translated request offers, in order.
pub fn tool_names(v: &Value) -> Vec<String> {
    v.get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|t| {
            t.get("name")
                .or_else(|| t.pointer("/function/name"))
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .collect()
}

/// Check that `out` (a translated request) still carries everything in `c`.
pub fn check_carried(c: &Carried, out: &Value) -> Result<(), String> {
    let mut all = Vec::new();
    strings(out, &mut all);
    for t in &c.texts {
        if !all.iter().any(|s| s.contains(t.as_str())) {
            return Err(format!("text {t:?} was lost"));
        }
    }
    for p in &c.payloads {
        if !all.iter().any(|s| s.contains(p.as_str())) {
            return Err(format!("payload {p:?} was lost"));
        }
    }
    let mut calls = Vec::new();
    calls_in(out, &mut calls);
    if calls != c.calls {
        return Err(format!(
            "tool calls: sent {:?}, translated {calls:?}",
            c.calls
        ));
    }
    let names = tool_names(out);
    if names != c.tools {
        return Err(format!(
            "tools: offered {:?}, translated {names:?}",
            c.tools
        ));
    }
    Ok(())
}
