//! Response- and stream-side translation: what a client receives back.
//!
//! Every stream test goes through [`stream`], which feeds the fixture whole, one byte at a time,
//! and in 7-byte pieces and requires the same events from all three — so each also proves the
//! bridge holds an event split across chunk boundaries.

use super::*;
use Endpoint::{ChatCompletions as Chat, Messages, Responses};

// ---- helpers -----------------------------------------------------------------------------------

fn json_resp(up: Endpoint, client: Endpoint, v: &Value) -> Value {
    let out = response_json(up, client, &serde_json::to_vec(v).unwrap());
    serde_json::from_slice(&out).unwrap()
}

fn json_status(up: Endpoint, client: Endpoint, status: u16, v: &Value) -> Value {
    let out = response_json_status(up, client, status, &serde_json::to_vec(v).unwrap());
    serde_json::from_slice(&out).unwrap()
}

/// One SSE event: its `event:` name (empty when unnamed) and its data (`"[DONE]"` as a string).
type Event = (String, Value);

fn events(out: &str) -> Vec<Event> {
    out.split("\n\n")
        .filter(|e| !e.trim().is_empty())
        .map(|e| {
            let mut name = String::new();
            let mut data = String::new();
            for line in e.lines() {
                if let Some(n) = line.strip_prefix("event: ") {
                    name = n.to_owned();
                } else if let Some(d) = line.strip_prefix("data: ") {
                    data = d.to_owned();
                }
            }
            let v = if data == "[DONE]" {
                json!("[DONE]")
            } else {
                serde_json::from_str(&data).unwrap_or_else(|e| panic!("{e}: {data}"))
            };
            (name, v)
        })
        .collect()
}

fn run(up: Endpoint, client: Endpoint, chunks: &[&[u8]]) -> String {
    let mut b = SseBridge::new(client, up);
    let mut out = Vec::new();
    for c in chunks {
        out.extend(b.feed(c, false));
    }
    out.extend(b.feed(b"", true));
    String::from_utf8(out).unwrap()
}

/// Gateway-minted ids differ per bridge and timestamps per second; replace both so feeds of the
/// same stream compare equal.
fn normalize(evs: &[Event]) -> Vec<Event> {
    fn walk(v: &Value, ids: &mut Vec<String>) -> Value {
        match v {
            Value::String(s) if s.contains("_gw") => {
                let n = ids.iter().position(|x| x == s).unwrap_or_else(|| {
                    ids.push(s.clone());
                    ids.len() - 1
                });
                json!(format!("ID{n}"))
            }
            Value::Array(a) => Value::Array(a.iter().map(|x| walk(x, ids)).collect()),
            Value::Object(m) => Value::Object(
                m.iter()
                    .map(|(k, x)| {
                        let x = if k == "created" || k == "created_at" {
                            json!(0)
                        } else {
                            walk(x, ids)
                        };
                        (k.clone(), x)
                    })
                    .collect(),
            ),
            other => other.clone(),
        }
    }
    let mut ids = Vec::new();
    evs.iter()
        .map(|(n, v)| (n.clone(), walk(v, &mut ids)))
        .collect()
}

/// Translate `src` fed whole, byte by byte, and in 7-byte chunks; all three must agree.
fn stream(up: Endpoint, client: Endpoint, src: &str) -> Vec<Event> {
    let whole = events(&run(up, client, &[src.as_bytes()]));
    let bytes: Vec<&[u8]> = src.as_bytes().chunks(1).collect();
    let sevens: Vec<&[u8]> = src.as_bytes().chunks(7).collect();
    let want = normalize(&whole);
    assert_eq!(
        normalize(&events(&run(up, client, &bytes))),
        want,
        "byte-by-byte feed differs"
    );
    assert_eq!(
        normalize(&events(&run(up, client, &sevens))),
        want,
        "7-byte feed differs"
    );
    whole
}

fn named<'a>(evs: &'a [Event], name: &str) -> Vec<&'a Value> {
    evs.iter()
        .filter(|(n, _)| n == name)
        .map(|(_, v)| v)
        .collect()
}

fn one<'a>(evs: &'a [Event], name: &str) -> &'a Value {
    let found = named(evs, name);
    assert_eq!(found.len(), 1, "want exactly one {name}: {evs:#?}");
    found[0]
}

/// Chat chunks only (no `[DONE]`).
fn chunks(evs: &[Event]) -> Vec<&Value> {
    evs.iter()
        .map(|(_, v)| v)
        .filter(|v| v.is_object())
        .collect()
}

/// A Chat client's tool calls, rebuilt the way the OpenAI SDK accumulates them: by `index`.
fn chat_tool_calls(evs: &[Event]) -> Vec<(String, String, String)> {
    let mut calls: Vec<(String, String, String)> = Vec::new();
    for c in chunks(evs) {
        for tc in c
            .pointer("/choices/0/delta/tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let i = usize::try_from(tc["index"].as_u64().unwrap()).unwrap();
            if calls.len() <= i {
                calls.resize(i + 1, Default::default());
            }
            if let Some(id) = tc["id"].as_str() {
                calls[i].0 = id.to_owned();
            }
            if let Some(n) = tc.pointer("/function/name").and_then(Value::as_str) {
                calls[i].1.push_str(n);
            }
            if let Some(a) = tc.pointer("/function/arguments").and_then(Value::as_str) {
                calls[i].2.push_str(a);
            }
        }
    }
    calls
}

/// An Anthropic client's content, rebuilt the way the Anthropic SDK accumulates it:
/// `content_block_start` appends, deltas land on `content[index]`.
fn anthropic_content(evs: &[Event]) -> Vec<Value> {
    let mut content: Vec<Value> = Vec::new();
    let mut json_bufs: Vec<String> = Vec::new();
    for (name, v) in evs {
        match name.as_str() {
            "content_block_start" => {
                assert_eq!(
                    v["index"].as_u64().unwrap() as usize,
                    content.len(),
                    "blocks must start in order: {evs:#?}"
                );
                content.push(v["content_block"].clone());
                json_bufs.push(String::new());
            }
            "content_block_delta" => {
                let i = v["index"].as_u64().unwrap() as usize;
                let d = &v["delta"];
                let block = &mut content[i];
                match d["type"].as_str().unwrap() {
                    "text_delta" => {
                        let t = format!(
                            "{}{}",
                            block["text"].as_str().unwrap(),
                            d["text"].as_str().unwrap()
                        );
                        block["text"] = json!(t);
                    }
                    "thinking_delta" => {
                        let t = format!(
                            "{}{}",
                            block["thinking"].as_str().unwrap(),
                            d["thinking"].as_str().unwrap()
                        );
                        block["thinking"] = json!(t);
                    }
                    "signature_delta" => block["signature"] = d["signature"].clone(),
                    "input_json_delta" => {
                        json_bufs[i].push_str(d["partial_json"].as_str().unwrap());
                        block["input"] = serde_json::from_str(&json_bufs[i]).unwrap_or(json!({}));
                    }
                    other => panic!("unexpected delta {other}"),
                }
            }
            _ => {}
        }
    }
    content
}

fn stop_reason(evs: &[Event]) -> Value {
    one(evs, "message_delta")["delta"]["stop_reason"].clone()
}

fn finish_reason(evs: &[Event]) -> Value {
    chunks(evs)
        .iter()
        .find_map(|c| {
            c.pointer("/choices/0/finish_reason")
                .filter(|f| !f.is_null())
                .cloned()
        })
        .unwrap_or(Value::Null)
}

fn chat_usage(evs: &[Event]) -> Value {
    chunks(evs)
        .iter()
        .find_map(|c| c.get("usage").filter(|u| u.is_object()).cloned())
        .expect("a usage chunk")
}

fn anthropic_msg(content: Value, stop: &str, usage: Value) -> Value {
    json!({
        "id": "msg_01", "type": "message", "role": "assistant", "model": "claude-haiku-4-5",
        "content": content, "stop_reason": stop, "stop_sequence": null, "usage": usage,
    })
}

fn chat_completion(message: Value, finish: &str, usage: Value) -> Value {
    json!({
        "id": "chatcmpl-1", "object": "chat.completion", "created": 1_790_000_000, "model": "gpt-5",
        "choices": [{ "index": 0, "message": message, "finish_reason": finish }],
        "usage": usage,
    })
}

// ---- fixtures ----------------------------------------------------------------------------------

/// Claude with thinking (signed), text, and two parallel tool calls; cache reads and writes.
const CLAUDE_TOOLS_SSE: &str = r#"event: message_start
data: {"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","model":"claude-haiku-4-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"cache_read_input_tokens":5000,"cache_creation_input_tokens":2000,"output_tokens":1}}}

event: ping
data: {"type":"ping"}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Two cities, "}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"two calls."}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"EqQBsig=="}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Checking both."}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: content_block_start
data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_a","name":"get_weather","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}

event: content_block_delta
data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"\"Paris\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":2}

event: content_block_start
data: {"type":"content_block_start","index":3,"content_block":{"type":"tool_use","id":"toolu_b","name":"get_weather","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"Rome\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":3}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":42}}

event: message_stop
data: {"type":"message_stop"}

"#;

/// OpenAI streaming two parallel tool calls after some text, `include_usage` on.
const OPENAI_PARALLEL_SSE: &str = r#"data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1790000000,"model":"gpt-5","choices":[{"index":0,"delta":{"role":"assistant","content":"Checking "},"finish_reason":null}]}

data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1790000000,"model":"gpt-5","choices":[{"index":0,"delta":{"content":"both."},"finish_reason":null}]}

data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1790000000,"model":"gpt-5","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1790000000,"model":"gpt-5","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1790000000,"model":"gpt-5","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Paris\"}"}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1790000000,"model":"gpt-5","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_b","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1790000000,"model":"gpt-5","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\"city\":\"Rome\"}"}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1790000000,"model":"gpt-5","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1790000000,"model":"gpt-5","choices":[],"usage":{"prompt_tokens":6000,"completion_tokens":40,"total_tokens":6040,"prompt_tokens_details":{"cached_tokens":5000},"completion_tokens_details":{"reasoning_tokens":12}}}

data: [DONE]

"#;

/// OpenRouter serving Claude over Chat Completions with thinking: the text rides `reasoning` and
/// `reasoning_details`, the signature only a later `reasoning_details` entry. Shape copied from a
/// live capture (2026-09-30), trimmed.
const OPENROUTER_CLAUDE_SSE: &str = r#"data: {"id":"gen-1","object":"chat.completion.chunk","created":1790802605,"model":"anthropic/claude-haiku-4.5","provider":"Amazon Bedrock","choices":[{"index":0,"delta":{"content":"","role":"assistant","reasoning":"The user wants ","reasoning_details":[{"type":"reasoning.text","text":"The user wants ","format":"anthropic-claude-v1","index":0}]},"finish_reason":null,"native_finish_reason":null}]}

data: {"id":"gen-1","object":"chat.completion.chunk","created":1790802605,"model":"anthropic/claude-haiku-4.5","provider":"Amazon Bedrock","choices":[{"index":0,"delta":{"content":"","role":"assistant","reasoning":"two lookups.","reasoning_details":[{"type":"reasoning.text","text":"two lookups.","format":"anthropic-claude-v1","index":0}]},"finish_reason":null,"native_finish_reason":null}]}

data: {"id":"gen-1","object":"chat.completion.chunk","created":1790802605,"model":"anthropic/claude-haiku-4.5","provider":"Amazon Bedrock","choices":[{"index":0,"delta":{"content":"","role":"assistant","reasoning_details":[{"type":"reasoning.text","signature":"EqIFCpwBsig","format":"anthropic-claude-v1","index":0}]},"finish_reason":null,"native_finish_reason":null}]}

data: {"id":"gen-1","object":"chat.completion.chunk","created":1790802605,"model":"anthropic/claude-haiku-4.5","provider":"Amazon Bedrock","choices":[{"index":0,"delta":{"content":null,"role":"assistant","tool_calls":[{"index":0,"id":"toolu_bdrk_01","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null,"native_finish_reason":null}]}

data: {"id":"gen-1","object":"chat.completion.chunk","created":1790802605,"model":"anthropic/claude-haiku-4.5","provider":"Amazon Bedrock","choices":[{"index":0,"delta":{"content":null,"role":"assistant","tool_calls":[{"index":0,"function":{"arguments":"{\"city\": \"Paris"}}]},"finish_reason":null,"native_finish_reason":null}]}

data: {"id":"gen-1","object":"chat.completion.chunk","created":1790802605,"model":"anthropic/claude-haiku-4.5","provider":"Amazon Bedrock","choices":[{"index":0,"delta":{"content":null,"role":"assistant","tool_calls":[{"index":0,"function":{"arguments":"\"}"}}]},"finish_reason":null,"native_finish_reason":null}]}

data: {"id":"gen-1","object":"chat.completion.chunk","created":1790802605,"model":"anthropic/claude-haiku-4.5","provider":"Amazon Bedrock","choices":[{"index":0,"delta":{"content":null,"role":"assistant","tool_calls":[{"index":1,"id":"toolu_bdrk_02","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null,"native_finish_reason":null}]}

data: {"id":"gen-1","object":"chat.completion.chunk","created":1790802605,"model":"anthropic/claude-haiku-4.5","provider":"Amazon Bedrock","choices":[{"index":0,"delta":{"content":null,"role":"assistant","tool_calls":[{"index":1,"function":{"arguments":"{\"city\": \"Rome\"}"}}]},"finish_reason":null,"native_finish_reason":null}]}

data: {"id":"gen-1","object":"chat.completion.chunk","created":1790802605,"model":"anthropic/claude-haiku-4.5","provider":"Amazon Bedrock","choices":[{"index":0,"delta":{"content":"","role":"assistant","reasoning":null},"finish_reason":"tool_calls","native_finish_reason":"tool_use"}]}

data: {"id":"gen-1","object":"chat.completion.chunk","created":1790802605,"model":"anthropic/claude-haiku-4.5","provider":"Amazon Bedrock","choices":[{"index":0,"delta":{"content":"","role":"assistant"},"finish_reason":"tool_calls","native_finish_reason":"tool_use"}],"usage":{"prompt_tokens":597,"completion_tokens":190,"total_tokens":787,"prompt_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},"completion_tokens_details":{"reasoning_tokens":98}}}

data: [DONE]

"#;

/// An OpenAI Responses stream: a reasoning summary, text, and two function calls.
const RESPONSES_TOOLS_SSE: &str = r#"event: response.created
data: {"type":"response.created","sequence_number":0,"response":{"id":"resp_1","object":"response","created_at":1790000000,"status":"in_progress","model":"gpt-5-pro","output":[],"usage":null}}

event: response.in_progress
data: {"type":"response.in_progress","sequence_number":1,"response":{"id":"resp_1","object":"response","created_at":1790000000,"status":"in_progress","model":"gpt-5-pro","output":[],"usage":null}}

event: response.output_item.added
data: {"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"id":"rs_1","type":"reasoning","summary":[]}}

event: response.reasoning_summary_part.added
data: {"type":"response.reasoning_summary_part.added","sequence_number":3,"item_id":"rs_1","output_index":0,"summary_index":0,"part":{"type":"summary_text","text":""}}

event: response.reasoning_summary_text.delta
data: {"type":"response.reasoning_summary_text.delta","sequence_number":4,"item_id":"rs_1","output_index":0,"summary_index":0,"delta":"Two cities."}

event: response.reasoning_summary_text.done
data: {"type":"response.reasoning_summary_text.done","sequence_number":5,"item_id":"rs_1","output_index":0,"summary_index":0,"text":"Two cities."}

event: response.output_item.done
data: {"type":"response.output_item.done","sequence_number":6,"output_index":0,"item":{"id":"rs_1","type":"reasoning","summary":[{"type":"summary_text","text":"Two cities."}],"encrypted_content":"gAAAA"}}

event: response.output_item.added
data: {"type":"response.output_item.added","sequence_number":7,"output_index":1,"item":{"id":"msg_1","type":"message","status":"in_progress","role":"assistant","content":[]}}

event: response.content_part.added
data: {"type":"response.content_part.added","sequence_number":8,"item_id":"msg_1","output_index":1,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":9,"item_id":"msg_1","output_index":1,"content_index":0,"delta":"Checking both.","logprobs":[]}

event: response.output_text.done
data: {"type":"response.output_text.done","sequence_number":10,"item_id":"msg_1","output_index":1,"content_index":0,"text":"Checking both.","logprobs":[]}

event: response.output_item.done
data: {"type":"response.output_item.done","sequence_number":11,"output_index":1,"item":{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Checking both.","annotations":[]}]}}

event: response.output_item.added
data: {"type":"response.output_item.added","sequence_number":12,"output_index":2,"item":{"id":"fc_1","type":"function_call","status":"in_progress","arguments":"","call_id":"call_a","name":"get_weather"}}

event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","sequence_number":13,"item_id":"fc_1","output_index":2,"delta":"{\"city\":"}

event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","sequence_number":14,"item_id":"fc_1","output_index":2,"delta":"\"Paris\"}"}

event: response.function_call_arguments.done
data: {"type":"response.function_call_arguments.done","sequence_number":15,"item_id":"fc_1","output_index":2,"arguments":"{\"city\":\"Paris\"}"}

event: response.output_item.done
data: {"type":"response.output_item.done","sequence_number":16,"output_index":2,"item":{"id":"fc_1","type":"function_call","status":"completed","arguments":"{\"city\":\"Paris\"}","call_id":"call_a","name":"get_weather"}}

event: response.output_item.added
data: {"type":"response.output_item.added","sequence_number":17,"output_index":3,"item":{"id":"fc_2","type":"function_call","status":"in_progress","arguments":"","call_id":"call_b","name":"get_weather"}}

event: response.function_call_arguments.done
data: {"type":"response.function_call_arguments.done","sequence_number":18,"item_id":"fc_2","output_index":3,"arguments":"{\"city\":\"Rome\"}"}

event: response.output_item.done
data: {"type":"response.output_item.done","sequence_number":19,"output_index":3,"item":{"id":"fc_2","type":"function_call","status":"completed","arguments":"{\"city\":\"Rome\"}","call_id":"call_b","name":"get_weather"}}

event: response.completed
data: {"type":"response.completed","sequence_number":20,"response":{"id":"resp_1","object":"response","created_at":1790000000,"status":"completed","model":"gpt-5-pro","output":[],"usage":{"input_tokens":6000,"input_tokens_details":{"cached_tokens":5000},"output_tokens":300,"output_tokens_details":{"reasoning_tokens":200},"total_tokens":6300}}}

"#;

// ---- 3. usage: cache tokens both ways ----------------------------------------------------------

/// claim: TRN-23
#[test]
fn anthropic_cache_counts_join_prompt_tokens_for_an_openai_client() {
    let usage = json!({
        "input_tokens": 10, "cache_read_input_tokens": 5000,
        "cache_creation_input_tokens": 2000, "output_tokens": 5,
    });
    let oai = json_resp(
        Messages,
        Chat,
        &anthropic_msg(json!([{"type":"text","text":"hi"}]), "end_turn", usage),
    );
    let u = &oai["usage"];
    assert_eq!(u["prompt_tokens"], 7010, "{u}");
    assert_eq!(u["completion_tokens"], 5);
    assert_eq!(u["total_tokens"], 7015);
    assert_eq!(u["prompt_tokens_details"]["cached_tokens"], 5000);
    assert_eq!(u["prompt_tokens_details"]["cache_write_tokens"], 2000);

    // Streamed: the counts ride `message_start`, the output `message_delta`.
    let evs = stream(Messages, Chat, CLAUDE_TOOLS_SSE);
    let u = chat_usage(&evs);
    assert_eq!(u["prompt_tokens"], 7010, "{u}");
    assert_eq!(u["completion_tokens"], 42);
    assert_eq!(u["prompt_tokens_details"]["cached_tokens"], 5000);
    assert_eq!(u["prompt_tokens_details"]["cache_write_tokens"], 2000);
}

/// claim: TRN-23
#[test]
fn anthropic_message_delta_usage_is_cumulative() {
    let src = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"model\":\"c\",\"usage\":{\"input_tokens\":10,\"cache_read_input_tokens\":5,\"output_tokens\":1}}}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"input_tokens\":12,\"output_tokens\":9}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );
    let u = chat_usage(&stream(Messages, Chat, src));
    assert_eq!(
        u["prompt_tokens"], 17,
        "a message_delta total replaces message_start's: {u}"
    );
    assert_eq!(
        u["prompt_tokens_details"]["cached_tokens"], 5,
        "an absent field keeps its value"
    );
    assert_eq!(u["completion_tokens"], 9);
}

/// claim: TRN-23
#[test]
fn openai_cached_tokens_are_not_anthropic_input_tokens() {
    let usage = json!({
        "prompt_tokens": 6000, "completion_tokens": 5, "total_tokens": 6005,
        "prompt_tokens_details": {"cached_tokens": 5000, "cache_write_tokens": 100},
        "completion_tokens_details": {"reasoning_tokens": 3},
    });
    let anth = json_resp(
        Chat,
        Messages,
        &chat_completion(json!({"role":"assistant","content":"hi"}), "stop", usage),
    );
    let u = &anth["usage"];
    assert_eq!(u["input_tokens"], 900, "{u}");
    assert_eq!(u["cache_read_input_tokens"], 5000);
    assert_eq!(u["cache_creation_input_tokens"], 100);
    assert_eq!(u["output_tokens"], 5);
    assert_eq!(u["output_tokens_details"]["thinking_tokens"], 3);

    let evs = stream(Chat, Messages, OPENAI_PARALLEL_SSE);
    let u = &one(&evs, "message_delta")["usage"];
    assert_eq!(u["input_tokens"], 1000, "{u}");
    assert_eq!(u["cache_read_input_tokens"], 5000);
    assert_eq!(u["output_tokens"], 40);
}

/// claim: TRN-23
#[test]
fn responses_usage_carries_cache_and_reasoning_both_ways() {
    let usage = json!({
        "input_tokens": 10, "cache_read_input_tokens": 5000,
        "cache_creation_input_tokens": 2000, "output_tokens": 5,
        "output_tokens_details": {"thinking_tokens": 4},
    });
    let r = json_resp(
        Messages,
        Responses,
        &anthropic_msg(json!([{"type":"text","text":"hi"}]), "end_turn", usage),
    );
    assert_eq!(
        r["usage"],
        json!({
            "input_tokens": 7010,
            "input_tokens_details": {"cached_tokens": 5000, "cache_write_tokens": 2000},
            "output_tokens": 5,
            "output_tokens_details": {"reasoning_tokens": 4},
            "total_tokens": 7015,
        })
    );

    let resp = json!({
        "id": "resp_1", "object": "response", "created_at": 1_790_000_000, "status": "completed",
        "model": "gpt-5-pro",
        "output": [{"type":"message","id":"msg_1","role":"assistant","status":"completed",
                    "content":[{"type":"output_text","text":"hi","annotations":[]}]}],
        "usage": {"input_tokens": 6000, "input_tokens_details": {"cached_tokens": 5000},
                  "output_tokens": 300, "output_tokens_details": {"reasoning_tokens": 200},
                  "total_tokens": 6300},
    });
    let chat = json_resp(Responses, Chat, &resp);
    assert_eq!(chat["usage"]["prompt_tokens"], 6000);
    assert_eq!(
        chat["usage"]["prompt_tokens_details"]["cached_tokens"],
        5000
    );
    assert_eq!(
        chat["usage"]["completion_tokens_details"]["reasoning_tokens"],
        200
    );
    let anth = json_resp(Responses, Messages, &resp);
    assert_eq!(anth["usage"]["input_tokens"], 1000);
    assert_eq!(anth["usage"]["cache_read_input_tokens"], 5000);

    // Streamed onto Responses the required detail objects are always present.
    let evs = stream(Chat, Responses, OPENAI_PARALLEL_SSE);
    let u = &one(&evs, "response.completed")["response"]["usage"];
    assert_eq!(u["input_tokens"], 6000, "{u}");
    assert_eq!(u["input_tokens_details"]["cached_tokens"], 5000);
    assert_eq!(u["input_tokens_details"]["cache_write_tokens"], 0);
    assert_eq!(u["output_tokens_details"]["reasoning_tokens"], 12);
}

// ---- 1. the Responses stream lifecycle ---------------------------------------------------------

/// Checks the invariants every Responses client relies on and returns the completed response.
fn assert_responses_lifecycle(evs: &[Event], terminal: &str) -> Value {
    assert_eq!(evs[0].0, "response.created", "{evs:#?}");
    assert_eq!(evs[1].0, "response.in_progress", "{evs:#?}");
    assert_eq!(evs.last().unwrap().0, terminal, "{evs:#?}");
    for (i, (name, v)) in evs.iter().enumerate() {
        assert_eq!(v["type"], json!(name), "data type matches the event name");
        assert_eq!(
            v["sequence_number"],
            json!(i),
            "sequence numbers count from 0: {v}"
        );
        assert!(
            v.get("error").is_none(),
            "an OpenAI SDK raises on a top-level error: {v}"
        );
    }
    // Items are numbered in the order they are added; every item event names an item that was
    // added and is not yet done.
    let mut open: Vec<(u64, String)> = Vec::new();
    let mut done: Vec<u64> = Vec::new();
    for (name, v) in evs {
        match name.as_str() {
            "response.output_item.added" => {
                let oi = v["output_index"].as_u64().unwrap();
                assert_eq!(oi as usize, open.len(), "{v}");
                let id = v["item"]["id"].as_str().unwrap().to_owned();
                assert!(!id.is_empty());
                open.push((oi, id));
            }
            "response.output_item.done" => {
                let oi = v["output_index"].as_u64().unwrap();
                let (_, id) = open
                    .iter()
                    .find(|(o, _)| *o == oi)
                    .expect("done for an added item");
                assert_eq!(v["item"]["id"].as_str().unwrap(), id);
                assert!(!done.contains(&oi), "item done twice");
                done.push(oi);
            }
            n if n.starts_with("response.") && v.get("output_index").is_some() => {
                let oi = v["output_index"].as_u64().unwrap();
                let (_, id) = open
                    .iter()
                    .find(|(o, _)| *o == oi)
                    .unwrap_or_else(|| panic!("{n} before its item: {v}"));
                assert!(!done.contains(&oi), "{n} after its item was done");
                assert_eq!(v["item_id"].as_str(), Some(id.as_str()), "{v}");
            }
            _ => {}
        }
    }
    assert_eq!(open.len(), done.len(), "every added item is done");
    let resp = evs.last().unwrap().1["response"].clone();
    assert_eq!(resp["object"], "response");
    assert!(resp["created_at"].as_u64().unwrap() > 0);
    assert!(resp["id"].as_str().is_some_and(|s| !s.is_empty()));
    // The completed output is the done items, in output order.
    let items: Vec<Value> = {
        let mut by_index: Vec<(u64, Value)> = named(evs, "response.output_item.done")
            .into_iter()
            .map(|v| (v["output_index"].as_u64().unwrap(), v["item"].clone()))
            .collect();
        by_index.sort_by_key(|(i, _)| *i);
        by_index.into_iter().map(|(_, i)| i).collect()
    };
    assert_eq!(resp["output"], json!(items));
    resp
}

#[test]
fn chat_stream_reaches_responses_with_text_and_parallel_calls() {
    let evs = stream(Chat, Responses, OPENAI_PARALLEL_SSE);
    let resp = assert_responses_lifecycle(&evs, "response.completed");
    assert_eq!(resp["status"], "completed");
    let out = resp["output"].as_array().unwrap();
    assert_eq!(out.len(), 3, "{resp:#}");
    assert_eq!(out[0]["type"], "message");
    assert_eq!(
        out[0]["content"][0],
        json!({"type":"output_text","text":"Checking both.","annotations":[]})
    );
    for (item, (call_id, city)) in out[1..]
        .iter()
        .zip([("call_a", "Paris"), ("call_b", "Rome")])
    {
        assert_eq!(item["type"], "function_call");
        assert_eq!(item["call_id"], call_id);
        assert!(item["id"].as_str().unwrap().starts_with("fc_"), "{item}");
        assert_eq!(item["name"], "get_weather");
        assert_eq!(item["status"], "completed");
        let args: Value = serde_json::from_str(item["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["city"], city);
    }
    // The text arrived as deltas on content part 0 of the message item.
    let deltas: String = named(&evs, "response.output_text.delta")
        .iter()
        .map(|v| {
            assert_eq!(v["content_index"], 0);
            v["delta"].as_str().unwrap()
        })
        .collect();
    assert_eq!(deltas, "Checking both.");
    let done = one(&evs, "response.output_text.done");
    assert_eq!(done["text"], "Checking both.");
    let args_done = named(&evs, "response.function_call_arguments.done");
    assert_eq!(args_done.len(), 2);
    assert_eq!(args_done[1]["arguments"], "{\"city\":\"Rome\"}");
}

#[test]
fn claude_stream_reaches_responses_with_reasoning_text_and_calls() {
    let evs = stream(Messages, Responses, CLAUDE_TOOLS_SSE);
    let resp = assert_responses_lifecycle(&evs, "response.completed");
    let out = resp["output"].as_array().unwrap();
    let types: Vec<&str> = out.iter().map(|i| i["type"].as_str().unwrap()).collect();
    assert_eq!(
        types,
        ["reasoning", "message", "function_call", "function_call"],
        "{resp:#}"
    );
    assert_eq!(
        out[0]["summary"],
        json!([{"type":"summary_text","text":"Two cities, two calls."}])
    );
    assert_eq!(
        out[0]["encrypted_content"], "rs_gw:EqQBsig==",
        "the signature rides encrypted_content, behind the gateway marker"
    );
    assert_eq!(out[1]["content"][0]["text"], "Checking both.");
    assert_eq!(out[2]["call_id"], "toolu_a");
    assert_eq!(out[2]["arguments"], "{\"city\":\"Paris\"}");
    assert_eq!(out[3]["call_id"], "toolu_b");
    assert_eq!(out[3]["arguments"], "{\"city\":\"Rome\"}");
    let summary: String = named(&evs, "response.reasoning_summary_text.delta")
        .iter()
        .map(|v| v["delta"].as_str().unwrap())
        .collect();
    assert_eq!(summary, "Two cities, two calls.");
    assert_eq!(
        one(&evs, "response.reasoning_summary_part.added")["summary_index"],
        0
    );
    assert_eq!(resp["usage"]["input_tokens"], 7010);
    assert_eq!(resp["model"], "claude-haiku-4-5");
}

#[test]
fn a_truncated_stream_ends_responses_incomplete() {
    let src = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"model\":\"c\",\"usage\":{\"input_tokens\":3,\"output_tokens\":1}}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"cut of\"}}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"max_tokens\"},\"usage\":{\"output_tokens\":5}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );
    let evs = stream(Messages, Responses, src);
    let resp = assert_responses_lifecycle(&evs, "response.incomplete");
    assert_eq!(resp["status"], "incomplete");
    assert_eq!(
        resp["incomplete_details"],
        json!({"reason": "max_output_tokens"})
    );
    assert_eq!(resp["output"][0]["status"], "incomplete");
    assert_eq!(resp["output"][0]["content"][0]["text"], "cut of");

    let refused = src.replace("max_tokens", "refusal");
    let resp = assert_responses_lifecycle(
        &stream(Messages, Responses, &refused),
        "response.incomplete",
    );
    assert_eq!(
        resp["incomplete_details"],
        json!({"reason": "content_filter"})
    );
}

#[test]
fn a_truncated_json_response_is_incomplete_on_responses() {
    let r = json_resp(
        Messages,
        Responses,
        &anthropic_msg(
            json!([{"type":"text","text":"cut of"},{"type":"tool_use","id":"toolu_1","name":"f","input":{"a":1}}]),
            "max_tokens",
            json!({"input_tokens": 1, "output_tokens": 1}),
        ),
    );
    assert_eq!(r["status"], "incomplete");
    assert_eq!(
        r["incomplete_details"],
        json!({"reason": "max_output_tokens"})
    );
    assert_eq!(r["output"][0]["status"], "incomplete");
    assert_eq!(r["output"][1]["type"], "function_call");
    assert_eq!(r["output"][1]["call_id"], "toolu_1");
    assert!(r["output"][1]["id"].as_str().unwrap().starts_with("fc_"));

    let r = json_resp(
        Chat,
        Responses,
        &chat_completion(
            json!({"role":"assistant","content":"cut"}),
            "length",
            json!({"prompt_tokens": 6, "completion_tokens": 5}),
        ),
    );
    assert_eq!(r["status"], "incomplete");
    assert_eq!(r["incomplete_details"]["reason"], "max_output_tokens");
    let r = json_resp(
        Chat,
        Responses,
        &chat_completion(
            json!({"role":"assistant","content":"x"}),
            "content_filter",
            json!({}),
        ),
    );
    assert_eq!(r["incomplete_details"]["reason"], "content_filter");
    let r = json_resp(
        Chat,
        Responses,
        &chat_completion(json!({"role":"assistant","content":"x"}), "stop", json!({})),
    );
    assert_eq!(r["status"], "completed");
    assert!(r["incomplete_details"].is_null());
}

#[test]
fn a_stream_error_reaches_responses_as_error_then_failed() {
    let src = concat!(
        "data: {\"id\":\"c\",\"model\":\"gpt-5\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"par\"}}]}\n\n",
        "data: {\"error\":{\"message\":\"overloaded, retry\",\"type\":\"server_error\",\"code\":\"overloaded\"}}\n\n",
        "data: [DONE]\n\n",
    );
    let evs = stream(Chat, Responses, src);
    let names: Vec<&str> = evs.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names.last(), Some(&"response.failed"), "{names:?}");
    assert!(!names.contains(&"response.completed"), "{names:?}");
    let err = one(&evs, "error");
    assert_eq!(err["message"], "overloaded, retry");
    assert_eq!(err["code"], "overloaded");
    assert_eq!(
        err["error"]["message"], "overloaded, retry",
        "an OpenAI SDK raises this as APIError"
    );
    let failed = &one(&evs, "response.failed")["response"];
    assert_eq!(failed["status"], "failed");
    assert_eq!(failed["error"]["message"], "overloaded, retry");
}

// ---- 4. thinking signatures ------------------------------------------------------------------

#[test]
fn openrouter_signature_reaches_the_anthropic_thinking_block() {
    let evs = stream(Chat, Messages, OPENROUTER_CLAUDE_SSE);
    let content = anthropic_content(&evs);
    assert_eq!(content.len(), 3, "{content:#?}");
    assert_eq!(content[0]["type"], "thinking");
    assert_eq!(content[0]["thinking"], "The user wants two lookups.");
    assert_eq!(content[0]["signature"], "EqIFCpwBsig", "{content:#?}");
    assert_eq!(content[1]["type"], "tool_use");
    assert_eq!(content[1]["id"], "toolu_bdrk_01");
    assert_eq!(content[1]["input"], json!({"city": "Paris"}));
    assert_eq!(content[2]["id"], "toolu_bdrk_02");
    assert_eq!(content[2]["input"], json!({"city": "Rome"}));
    assert_eq!(stop_reason(&evs), "tool_use");
    let u = &one(&evs, "message_delta")["usage"];
    assert_eq!(u["input_tokens"], 597);
    assert_eq!(u["output_tokens_details"]["thinking_tokens"], 98);
}

#[test]
fn openrouter_nonstream_reasoning_details_become_signed_thinking() {
    let message = json!({
        "role": "assistant", "content": "4",
        "reasoning": "simple arithmetic",
        "reasoning_details": [
            {"type": "reasoning.text", "text": "simple arithmetic", "signature": "sigA", "format": "anthropic-claude-v1", "index": 0},
            {"type": "reasoning.encrypted", "data": "redactedB", "format": "anthropic-claude-v1", "index": 1},
        ],
    });
    let anth = json_resp(Chat, Messages, &chat_completion(message, "stop", json!({})));
    assert_eq!(
        anth["content"],
        json!([
            {"type": "thinking", "thinking": "simple arithmetic", "signature": "sigA"},
            {"type": "redacted_thinking", "data": "redactedB"},
            {"type": "text", "text": "4"},
        ])
    );
}

/// claim: TRN-21
#[test]
fn unsigned_thinking_never_reaches_a_messages_client() {
    // DeepSeek / gpt-oss style: reasoning text, never a signature.
    let message = json!({"role": "assistant", "content": "4", "reasoning_content": "hmm"});
    let anth = json_resp(Chat, Messages, &chat_completion(message, "stop", json!({})));
    assert_eq!(anth["content"], json!([{"type": "text", "text": "4"}]));

    let src = concat!(
        "data: {\"id\":\"c\",\"model\":\"deepseek-reasoner\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"hmm\"}}]}\n\n",
        "data: {\"id\":\"c\",\"model\":\"deepseek-reasoner\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"4\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2}}\n\n",
    );
    let evs = stream(Chat, Messages, src);
    assert_eq!(
        anthropic_content(&evs),
        vec![json!({"type": "text", "text": "4"})]
    );
    assert!(
        !evs.iter().any(|(_, v)| v.to_string().contains("hmm")),
        "{evs:#?}"
    );

    // A signature in another provider's format would not verify at Anthropic either.
    let foreign = OPENROUTER_CLAUDE_SSE.replace("anthropic-claude-v1", "openai-responses-v1");
    let content = anthropic_content(&stream(Chat, Messages, &foreign));
    assert!(
        content.iter().all(|b| b["type"] == "tool_use"),
        "{content:#?}"
    );
}

#[test]
fn each_openrouter_reasoning_index_is_its_own_block() {
    let src = concat!(
        "data: {\"id\":\"g\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_details\":[{\"type\":\"reasoning.text\",\"text\":\"first\",\"format\":\"anthropic-claude-v1\",\"index\":0}]}}]}\n\n",
        "data: {\"id\":\"g\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_details\":[{\"type\":\"reasoning.text\",\"signature\":\"s0\",\"format\":\"anthropic-claude-v1\",\"index\":0}]}}]}\n\n",
        "data: {\"id\":\"g\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_details\":[{\"type\":\"reasoning.text\",\"text\":\"unsigned\",\"format\":\"anthropic-claude-v1\",\"index\":1}]}}]}\n\n",
        "data: {\"id\":\"g\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_details\":[{\"type\":\"reasoning.text\",\"text\":\"second\",\"format\":\"anthropic-claude-v1\",\"index\":2}]}}]}\n\n",
        "data: {\"id\":\"g\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_details\":[{\"type\":\"reasoning.text\",\"signature\":\"s2\",\"format\":\"anthropic-claude-v1\",\"index\":2}]}}]}\n\n",
        "data: {\"id\":\"g\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_details\":[{\"type\":\"reasoning.encrypted\",\"data\":\"opaque\",\"format\":\"anthropic-claude-v1\",\"index\":3}]}}]}\n\n",
        "data: {\"id\":\"g\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n",
    );
    let content = anthropic_content(&stream(Chat, Messages, src));
    assert_eq!(
        content,
        vec![
            json!({"type": "thinking", "thinking": "first", "signature": "s0"}),
            json!({"type": "thinking", "thinking": "second", "signature": "s2"}),
            json!({"type": "redacted_thinking", "data": "opaque"}),
            json!({"type": "text", "text": "done"}),
        ]
    );
}

#[test]
fn a_claude_signature_crosses_chat_into_anthropic_json_and_back() {
    let anth = anthropic_msg(
        json!([
            {"type": "thinking", "thinking": "plan", "signature": "sigX"},
            {"type": "redacted_thinking", "data": "blob"},
            {"type": "text", "text": "hi"},
        ]),
        "end_turn",
        json!({"input_tokens": 1, "output_tokens": 1}),
    );
    let chat = json_resp(Messages, Chat, &anth);
    assert_eq!(chat["choices"][0]["message"]["reasoning_content"], "plan");
    let back = json_resp(Chat, Messages, &chat);
    assert_eq!(back["content"], anth["content"]);
}

// ---- 5. stop reasons ---------------------------------------------------------------------------

/// claim: TRN-22
#[test]
fn every_anthropic_stop_reason_has_an_openai_finish_reason() {
    for (stop, finish) in [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("tool_use", "tool_calls"),
        ("max_tokens", "length"),
        ("model_context_window_exceeded", "length"),
        ("pause_turn", "length"),
        ("refusal", "content_filter"),
    ] {
        let oai = json_resp(
            Messages,
            Chat,
            &anthropic_msg(json!([{"type":"text","text":"partial"}]), stop, json!({})),
        );
        assert_eq!(oai["choices"][0]["finish_reason"], finish, "{stop}");
        let src = format!(
            "event: message_start\ndata: {{\"type\":\"message_start\",\"message\":{{\"id\":\"m\",\"model\":\"c\",\"usage\":{{\"input_tokens\":1}}}}}}\n\n\
             event: message_delta\ndata: {{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":\"{stop}\"}},\"usage\":{{\"output_tokens\":5}}}}\n\n\
             event: message_stop\ndata: {{\"type\":\"message_stop\"}}\n\n"
        );
        assert_eq!(
            finish_reason(&stream(Messages, Chat, &src)),
            finish,
            "{stop} streamed"
        );
    }
}

/// claim: TRN-22
#[test]
fn an_anthropic_refusal_explanation_reaches_openai_refusal() {
    let mut msg = anthropic_msg(json!([]), "refusal", json!({}));
    msg["stop_details"] =
        json!({"type": "refusal", "category": "cyber", "explanation": "Declined: cyber."});
    let oai = json_resp(Messages, Chat, &msg);
    let m = &oai["choices"][0]["message"];
    assert_eq!(m["content"], Value::Null, "no fake empty text: {m}");
    assert_eq!(m["refusal"], "Declined: cyber.");
    assert_eq!(oai["choices"][0]["finish_reason"], "content_filter");
}

/// claim: TRN-22
#[test]
fn an_openai_refusal_reaches_an_anthropic_client_as_a_refusal() {
    let oai = chat_completion(
        json!({"role": "assistant", "content": null, "refusal": "I can't help with that."}),
        "stop",
        json!({}),
    );
    let anth = json_resp(Chat, Messages, &oai);
    assert_eq!(anth["stop_reason"], "refusal");
    assert_eq!(
        anth["content"],
        json!([{"type": "text", "text": "I can't help with that."}])
    );

    // A filtered answer with no words has no content, not an empty text block.
    let filtered = chat_completion(
        json!({"role": "assistant", "content": null}),
        "content_filter",
        json!({}),
    );
    let anth = json_resp(Chat, Messages, &filtered);
    assert_eq!(anth["stop_reason"], "refusal");
    assert_eq!(anth["content"], json!([]));

    let src = concat!(
        "data: {\"id\":\"c\",\"model\":\"gpt-5\",\"choices\":[{\"index\":0,\"delta\":{\"refusal\":\"I can't \"}}]}\n\n",
        "data: {\"id\":\"c\",\"model\":\"gpt-5\",\"choices\":[{\"index\":0,\"delta\":{\"refusal\":\"help.\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2}}\n\n",
    );
    let evs = stream(Chat, Messages, src);
    assert_eq!(stop_reason(&evs), "refusal");
    assert_eq!(
        anthropic_content(&evs),
        vec![json!({"type": "text", "text": "I can't help."})]
    );
}

/// claim: TRN-22
#[test]
fn every_chat_finish_reason_has_an_anthropic_stop_reason() {
    for (finish, stop) in [
        ("stop", "end_turn"),
        ("length", "max_tokens"),
        ("tool_calls", "tool_use"),
        ("function_call", "tool_use"),
        ("content_filter", "refusal"),
    ] {
        let anth = json_resp(
            Chat,
            Messages,
            &chat_completion(json!({"role":"assistant","content":"x"}), finish, json!({})),
        );
        assert_eq!(anth["stop_reason"], stop, "{finish}");
    }
    // OpenRouter names Claude's own reason; it survives exactly.
    let mut c = chat_completion(json!({"role":"assistant","content":"x"}), "stop", json!({}));
    c["choices"][0]["native_finish_reason"] = json!("stop_sequence");
    assert_eq!(
        json_resp(Chat, Messages, &c)["stop_reason"],
        "stop_sequence"
    );
    // Tool calls under a plain `stop` (some OpenAI-compatible servers) are still a tool turn.
    let c = chat_completion(
        json!({"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"f","arguments":"{}"}}]}),
        "stop",
        json!({}),
    );
    assert_eq!(json_resp(Chat, Messages, &c)["stop_reason"], "tool_use");
}

/// claim: TRN-22
#[test]
fn an_openai_refusal_becomes_a_responses_refusal_part() {
    let r = json_resp(
        Chat,
        Responses,
        &chat_completion(
            json!({"role":"assistant","content":null,"refusal":"No."}),
            "stop",
            json!({}),
        ),
    );
    assert_eq!(
        r["output"][0]["content"],
        json!([{"type": "refusal", "refusal": "No."}])
    );

    let src = concat!(
        "data: {\"id\":\"c\",\"model\":\"gpt-5\",\"choices\":[{\"index\":0,\"delta\":{\"refusal\":\"No\"}}]}\n\n",
        "data: {\"id\":\"c\",\"model\":\"gpt-5\",\"choices\":[{\"index\":0,\"delta\":{\"refusal\":\".\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2}}\n\n",
    );
    let evs = stream(Chat, Responses, src);
    let resp = assert_responses_lifecycle(&evs, "response.completed");
    assert_eq!(
        resp["output"][0]["content"],
        json!([{"type": "refusal", "refusal": "No."}])
    );
    assert_eq!(one(&evs, "response.refusal.done")["refusal"], "No.");
    assert_eq!(
        one(&evs, "response.content_part.added")["part"]["type"],
        "refusal"
    );
}

// ---- 18. errors ------------------------------------------------------------------------------

#[test]
fn a_string_error_body_keeps_its_message() {
    let v = json_resp(
        Chat,
        Messages,
        &json!({"error": "Model gpt-x does not exist"}),
    );
    assert_eq!(v["type"], "error");
    assert_eq!(v["error"]["message"], "Model gpt-x does not exist");
    let v = json_resp(Messages, Chat, &json!({"error": "overloaded"}));
    assert_eq!(v["error"]["message"], "overloaded");
}

#[test]
fn error_code_and_param_reach_the_client() {
    let body = json!({"error": {
        "message": "Invalid 'messages[1].content': too long.",
        "type": "invalid_request_error", "param": "messages[1].content", "code": "string_above_max_length",
    }});
    let anth = json_resp(Chat, Messages, &body);
    assert_eq!(anth["error"]["type"], "invalid_request_error");
    assert_eq!(anth["error"]["param"], "messages[1].content");
    assert_eq!(anth["error"]["code"], "string_above_max_length");
    let resp = json_resp(Chat, Responses, &body);
    assert_eq!(resp["error"], body["error"]);

    // OpenAI types that have an Anthropic name get it.
    let rate = json!({"error": {"message": "slow down", "type": "requests", "code": "rate_limit_exceeded"}});
    assert_eq!(
        json_resp(Chat, Messages, &rate)["error"]["type"],
        "rate_limit_error"
    );
    let server = json!({"error": {"message": "boom", "type": "server_error"}});
    assert_eq!(
        json_resp(Chat, Messages, &server)["error"]["type"],
        "api_error"
    );
}

#[test]
fn openrouter_quotes_the_provider_error() {
    let raw = r#"{"type":"error","error":{"type":"invalid_request_error","message":"messages.1.content.0.thinking.signature: Field required"}}"#;
    let body = json!({"error": {
        "message": "Provider returned error", "code": 400,
        "metadata": {"raw": raw, "provider_name": "Anthropic"},
    }, "user_id": "u"});
    let anth = json_resp(Chat, Messages, &body);
    assert_eq!(
        anth["error"]["message"],
        "Provider returned error (Anthropic): messages.1.content.0.thinking.signature: Field required"
    );
    assert_eq!(
        anth["error"]["type"], "invalid_request_error",
        "the provider's type fills ours"
    );
    let resp = json_resp(Chat, Responses, &body);
    assert_eq!(
        resp["error"]["code"], "400",
        "a numeric code is a string on the OpenAI wire"
    );
    assert_eq!(resp["error"]["metadata"]["provider_name"], "Anthropic");

    let moderation = json!({"error": {
        "message": "Input flagged", "code": 403,
        "metadata": {"reasons": ["violence", "hate"], "provider_name": "OpenAI"},
    }});
    assert_eq!(
        json_resp(Chat, Messages, &moderation)["error"]["message"],
        "Input flagged (OpenAI): violence, hate"
    );
    let plain = json!({"error": {"message": "Provider returned error", "metadata": {"raw": "upstream timeout"}}});
    assert_eq!(
        json_resp(Chat, Messages, &plain)["error"]["message"],
        "Provider returned error: upstream timeout"
    );
}

#[test]
fn a_non_2xx_body_is_an_error_whatever_its_shape() {
    // Bedrock's error body has no `error` key.
    let bedrock = json!({"message": "The provided model identifier is invalid."});
    let v = json_status(Messages, Chat, 400, &bedrock);
    assert_eq!(
        v["error"]["message"],
        "The provided model identifier is invalid."
    );
    assert!(v.get("choices").is_none(), "{v}");
    let v = json_status(Chat, Messages, 404, &json!({"detail": "Not Found"}));
    assert_eq!(
        v,
        json!({"type": "error", "error": {"type": "api_error", "message": "Not Found"}})
    );
    // A 2xx with the same body is not second-guessed.
    let v = json_status(
        Messages,
        Chat,
        200,
        &anthropic_msg(json!([{"type":"text","text":"ok"}]), "end_turn", json!({})),
    );
    assert_eq!(v["choices"][0]["message"]["content"], "ok");
    // Not JSON: passed as it came.
    assert_eq!(
        response_json_status(Messages, Chat, 502, b"<html>bad gateway</html>"),
        b"<html>bad gateway</html>"
    );
}

#[test]
fn a_failed_responses_object_is_an_error() {
    let failed = json!({
        "id": "resp_1", "object": "response", "status": "failed", "output": [],
        "error": {"code": "server_error", "message": "The model failed."},
    });
    let chat = json_resp(Responses, Chat, &failed);
    assert_eq!(chat["error"]["message"], "The model failed.");
    assert_eq!(chat["error"]["code"], "server_error");
    let anth = json_resp(Responses, Messages, &failed);
    assert_eq!(anth["error"]["type"], "api_error");
    // A completed one with `"error": null` is not.
    let ok = json!({"id": "resp_1", "object": "response", "status": "completed", "error": null, "output": []});
    assert_eq!(json_resp(Responses, Chat, &ok)["object"], "chat.completion");
}

#[test]
fn stream_errors_keep_their_detail() {
    let src = "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n";
    let evs = stream(Messages, Chat, src);
    assert_eq!(chunks(&evs)[0]["error"]["message"], "Overloaded");
    assert_eq!(chunks(&evs)[0]["error"]["type"], "overloaded_error");
    assert_eq!(evs.last().unwrap().1, "[DONE]");

    let src = "data: {\"error\":{\"message\":\"Provider returned error\",\"code\":502,\"metadata\":{\"raw\":\"upstream reset\",\"provider_name\":\"Anthropic\"}}}\n\n";
    let evs = stream(Chat, Messages, src);
    assert_eq!(
        evs.len(),
        1,
        "the error alone, with no invented start or close: {evs:#?}"
    );
    assert_eq!(
        one(&evs, "error")["error"]["message"],
        "Provider returned error (Anthropic): upstream reset"
    );
}

// ---- 19. required fields ---------------------------------------------------------------------

/// claim: TRN-23
#[test]
fn chat_completions_and_every_chunk_carry_created_and_one_id() {
    let oai = json_resp(
        Messages,
        Chat,
        &anthropic_msg(json!([{"type":"text","text":"x"}]), "end_turn", json!({})),
    );
    assert!(oai["created"].as_u64().unwrap() > 1_700_000_000, "{oai}");
    assert_eq!(oai["object"], "chat.completion");
    for up in [Messages, Responses] {
        let src = if up == Messages {
            CLAUDE_TOOLS_SSE
        } else {
            RESPONSES_TOOLS_SSE
        };
        let evs = stream(up, Chat, src);
        let cs = chunks(&evs);
        assert!(cs.len() > 3);
        for c in &cs {
            assert_eq!(c["object"], "chat.completion.chunk", "{c}");
            assert!(c["created"].as_u64().unwrap() > 1_700_000_000, "{c}");
            assert_eq!(c["id"], cs[0]["id"], "one id per response: {c}");
            assert!(!c["id"].as_str().unwrap().is_empty());
        }
    }
    let evs = stream(Responses, Chat, RESPONSES_TOOLS_SSE);
    assert_eq!(
        chunks(&evs)[0]["created"],
        1_790_000_000,
        "the upstream's created_at"
    );
}

/// claim: TRN-23
#[test]
fn responses_objects_carry_created_at_and_item_ids() {
    let r = json_resp(
        Chat,
        Responses,
        &chat_completion(
            json!({"role":"assistant","content":"hi","tool_calls":[{"id":"call_1","type":"function","function":{"name":"f","arguments":"{}"}}]}),
            "tool_calls",
            json!({}),
        ),
    );
    assert_eq!(r["object"], "response");
    assert_eq!(r["created_at"], 1_790_000_000);
    for key in [
        "parallel_tool_calls",
        "tool_choice",
        "tools",
        "error",
        "incomplete_details",
    ] {
        assert!(r.get(key).is_some(), "{key} is required: {r}");
    }
    assert!(r["output"][0]["id"].as_str().unwrap().starts_with("msg_"));
    assert_eq!(r["output"][1]["call_id"], "call_1");
    assert!(r["output"][1]["id"].as_str().unwrap().starts_with("fc_"));
    // Two responses never share an item id.
    let again = json_resp(
        Chat,
        Responses,
        &chat_completion(
            json!({"role":"assistant","content":"hi"}),
            "stop",
            json!({}),
        ),
    );
    assert_ne!(again["output"][0]["id"], r["output"][0]["id"]);
}

// ---- tool-call deltas keyed by index ----------------------------------------------------------

/// claim: T1
#[test]
fn interleaved_parallel_tool_deltas_follow_their_index() {
    let src = concat!(
        "data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"function\":{\"name\":\"f\",\"arguments\":\"\"}},{\"index\":1,\"id\":\"call_b\",\"function\":{\"name\":\"g\",\"arguments\":\"\"}}]}}]}\n\n",
        "data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"a\\\":\"}}]}}]}\n\n",
        "data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"{\\\"b\\\":2}\"}}]}}]}\n\n",
        "data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"1}\"}}]}}]}\n\n",
        "data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2}}\n\n",
    );
    let content = anthropic_content(&stream(Chat, Messages, src));
    assert_eq!(content.len(), 2, "{content:#?}");
    assert_eq!(
        (&content[0]["id"], &content[0]["input"]),
        (&json!("call_a"), &json!({"a": 1}))
    );
    assert_eq!(
        (&content[1]["id"], &content[1]["input"]),
        (&json!("call_b"), &json!({"b": 2}))
    );

    let resp = assert_responses_lifecycle(&stream(Chat, Responses, src), "response.completed");
    assert_eq!(resp["output"][0]["arguments"], "{\"a\":1}");
    assert_eq!(resp["output"][1]["arguments"], "{\"b\":2}");
}

#[test]
fn a_call_id_repeated_on_every_delta_is_one_call() {
    let src = concat!(
        "data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]}}]}\n\n",
        "data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"arguments\":\"{\\\"city\\\":\"}}]}}]}\n\n",
        "data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"arguments\":\"\\\"Paris\\\"}\"}}]}}]}\n\n",
        "data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2}}\n\n",
        "data: [DONE]\n\n",
    );
    let content = anthropic_content(&stream(Chat, Messages, src));
    assert_eq!(content.len(), 1, "{content:#?}");
    assert_eq!(content[0]["input"], json!({"city": "Paris"}));
}

#[test]
fn index_less_and_whole_chunk_tool_calls_are_kept_apart() {
    // Gemini-compatible style: no `index`, each call whole in one chunk.
    let src = concat!(
        "data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"id\":\"call_a\",\"type\":\"function\",\"function\":{\"name\":\"f\",\"arguments\":\"{\\\"x\\\":1}\"}},{\"id\":\"call_b\",\"type\":\"function\",\"function\":{\"name\":\"f\",\"arguments\":\"{\\\"x\\\":2}\"}}]}}]}\n\n",
        "data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2}}\n\n",
    );
    let content = anthropic_content(&stream(Chat, Messages, src));
    assert_eq!(content.len(), 2, "{content:#?}");
    assert_eq!(content[0]["input"], json!({"x": 1}));
    assert_eq!(content[1]["input"], json!({"x": 2}));

    // One `index` reused for two calls with different ids.
    let reused = src.replace("{\"id\":\"call_", "{\"index\":0,\"id\":\"call_");
    let content = anthropic_content(&stream(Chat, Messages, &reused));
    assert_eq!(content.len(), 2, "{content:#?}");
    assert_eq!(content[1]["id"], "call_b");

    // A call whose name comes late opens once named, never nameless.
    let late = concat!(
        "data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"function\":{\"arguments\":\"{\\\"x\\\"\"}}]}}]}\n\n",
        "data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"f\",\"arguments\":\":1}\"}}]}}]}\n\n",
        "data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2}}\n\n",
    );
    let evs = stream(Chat, Messages, late);
    let content = anthropic_content(&evs);
    assert_eq!(
        content,
        vec![json!({"type": "tool_use", "id": "call_a", "name": "f", "input": {"x": 1}})]
    );
}

#[test]
fn a_server_tool_input_never_lands_on_a_client_call() {
    let src = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"model\":\"c\",\"usage\":{\"input_tokens\":1}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_a\",\"name\":\"f\",\"input\":{}}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"x\\\":1}\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"server_tool_use\",\"id\":\"srvtoolu_1\",\"name\":\"web_search\",\"input\":{}}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"query\\\":\\\"q\\\"}\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":5}}\n\n",
    );
    let evs = stream(Messages, Chat, src);
    assert_eq!(
        chat_tool_calls(&evs),
        vec![("toolu_a".to_owned(), "f".to_owned(), "{\"x\":1}".to_owned())]
    );
}

#[test]
fn an_empty_reasoning_details_does_not_hide_reasoning_text() {
    let src = concat!(
        "data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning\":\"hmm\",\"reasoning_details\":[]}}]}\n\n",
        "data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2}}\n\n",
    );
    let resp = assert_responses_lifecycle(&stream(Chat, Responses, src), "response.completed");
    assert_eq!(resp["output"][0]["summary"][0]["text"], "hmm", "{resp:#}");
}

// ---- Responses upstream ------------------------------------------------------------------------

#[test]
fn a_responses_stream_reaches_chat_with_tool_calls() {
    let evs = stream(Responses, Chat, RESPONSES_TOOLS_SSE);
    assert_eq!(evs.last().unwrap().1, "[DONE]");
    assert_eq!(
        chat_tool_calls(&evs),
        vec![
            (
                "call_a".to_owned(),
                "get_weather".to_owned(),
                "{\"city\":\"Paris\"}".to_owned()
            ),
            (
                "call_b".to_owned(),
                "get_weather".to_owned(),
                "{\"city\":\"Rome\"}".to_owned()
            ),
        ]
    );
    assert_eq!(finish_reason(&evs), "tool_calls");
    let text: String = chunks(&evs)
        .iter()
        .filter_map(|c| {
            c.pointer("/choices/0/delta/content")
                .and_then(Value::as_str)
        })
        .collect();
    assert_eq!(text, "Checking both.");
    let reasoning: String = chunks(&evs)
        .iter()
        .filter_map(|c| {
            c.pointer("/choices/0/delta/reasoning_content")
                .and_then(Value::as_str)
        })
        .collect();
    assert_eq!(reasoning, "Two cities.");
    let u = chat_usage(&evs);
    assert_eq!(u["prompt_tokens"], 6000);
    assert_eq!(u["prompt_tokens_details"]["cached_tokens"], 5000);
    assert_eq!(u["completion_tokens_details"]["reasoning_tokens"], 200);
    assert_eq!(chunks(&evs)[0]["model"], "gpt-5-pro");
}

#[test]
fn a_responses_stream_reaches_messages_with_tool_use() {
    let evs = stream(Responses, Messages, RESPONSES_TOOLS_SSE);
    let content = anthropic_content(&evs);
    // The OpenAI reasoning summary has no Anthropic signature, so it is not shown as thinking.
    assert_eq!(
        content,
        vec![
            json!({"type": "text", "text": "Checking both."}),
            json!({"type": "tool_use", "id": "call_a", "name": "get_weather", "input": {"city": "Paris"}}),
            json!({"type": "tool_use", "id": "call_b", "name": "get_weather", "input": {"city": "Rome"}}),
        ]
    );
    assert_eq!(stop_reason(&evs), "tool_use");
    assert_eq!(one(&evs, "message_delta")["usage"]["input_tokens"], 1000);
    assert_eq!(evs.last().unwrap().0, "message_stop");
}

/// claim: TRN-22
#[test]
fn a_responses_stream_ending_incomplete_or_failed_says_so() {
    let incomplete = RESPONSES_TOOLS_SSE
        .replace("\"status\":\"completed\",\"model\"", "\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"max_output_tokens\"},\"model\"")
        .replace("event: response.completed\ndata: {\"type\":\"response.completed\"", "event: response.incomplete\ndata: {\"type\":\"response.incomplete\"");
    assert_eq!(
        finish_reason(&stream(Responses, Chat, &incomplete)),
        "length"
    );
    assert_eq!(
        stop_reason(&stream(Responses, Messages, &incomplete)),
        "max_tokens"
    );
    let filtered = incomplete.replace("max_output_tokens", "content_filter");
    assert_eq!(
        finish_reason(&stream(Responses, Chat, &filtered)),
        "content_filter"
    );

    let failed = concat!(
        "event: response.created\n",
        "data: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"created_at\":1790000000,\"status\":\"in_progress\",\"model\":\"gpt-5-pro\",\"output\":[]}}\n\n",
        "event: response.failed\n",
        "data: {\"type\":\"response.failed\",\"sequence_number\":1,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"status\":\"failed\",\"error\":{\"code\":\"server_error\",\"message\":\"The model failed.\"}}}\n\n",
    );
    let evs = stream(Responses, Chat, failed);
    let err = chunks(&evs)
        .into_iter()
        .find(|c| c.get("error").is_some())
        .expect("an error chunk");
    assert_eq!(err["error"]["message"], "The model failed.");
    let evs = stream(Responses, Messages, failed);
    assert_eq!(one(&evs, "error")["error"]["message"], "The model failed.");
    assert!(
        named(&evs, "message_stop").is_empty(),
        "no invented success: {evs:#?}"
    );

    let error_event = "event: error\ndata: {\"type\":\"error\",\"sequence_number\":0,\"code\":\"rate_limit_exceeded\",\"message\":\"Slow down.\",\"param\":null}\n\n";
    let evs = stream(Responses, Messages, error_event);
    assert_eq!(one(&evs, "error")["error"]["message"], "Slow down.");
    assert_eq!(one(&evs, "error")["error"]["type"], "rate_limit_error");
}

#[test]
fn a_responses_json_reaches_chat_with_reasoning_refusal_and_calls() {
    let resp = json!({
        "id": "resp_1", "object": "response", "created_at": 1_790_000_000, "status": "completed",
        "model": "gpt-5-pro",
        "output": [
            {"type": "reasoning", "id": "rs_1", "summary": [{"type":"summary_text","text":"a"},{"type":"summary_text","text":"b"}]},
            {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": "hi", "annotations": []}]},
            {"type": "function_call", "id": "fc_1", "call_id": "call_a", "name": "f", "arguments": "{\"x\":1}", "status": "completed"},
        ],
        "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
    });
    let chat = json_resp(Responses, Chat, &resp);
    let m = &chat["choices"][0]["message"];
    assert_eq!(m["content"], "hi");
    assert_eq!(m["reasoning_content"], "a\n\nb");
    assert_eq!(m["tool_calls"][0]["id"], "call_a");
    assert_eq!(m["tool_calls"][0]["function"]["arguments"], "{\"x\":1}");
    assert_eq!(chat["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(chat["created"], 1_790_000_000);

    let anth = json_resp(Responses, Messages, &resp);
    assert_eq!(anth["stop_reason"], "tool_use");
    assert_eq!(
        anth["content"],
        json!([{"type":"text","text":"hi"},{"type":"tool_use","id":"call_a","name":"f","input":{"x":1}}])
    );

    let refused = json!({
        "id": "resp_2", "object": "response", "status": "completed", "model": "gpt-5-pro",
        "output": [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                    "content": [{"type": "refusal", "refusal": "No."}]}],
    });
    let chat = json_resp(Responses, Chat, &refused);
    assert_eq!(chat["choices"][0]["message"]["refusal"], "No.");
    assert_eq!(chat["choices"][0]["message"]["content"], Value::Null);
    let anth = json_resp(Responses, Messages, &refused);
    assert_eq!(anth["stop_reason"], "refusal");
    assert_eq!(anth["content"], json!([{"type": "text", "text": "No."}]));

    let truncated = json!({
        "id": "resp_3", "object": "response", "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"}, "model": "gpt-5-pro",
        "output": [{"type": "message", "id": "m", "role": "assistant", "status": "incomplete",
                    "content": [{"type": "output_text", "text": "cut", "annotations": []}]}],
    });
    assert_eq!(
        json_resp(Responses, Chat, &truncated)["choices"][0]["finish_reason"],
        "length"
    );
}

// ---- thinking a Chat client sends back (second audit) ----------------------------------------

/// openai-python's `accumulate_delta` (`lib/streaming/chat/_completions.py`), faithfully: how
/// `chat.completions.stream()` builds the message a client appends to its history. Strings
/// concatenate, `index` / `type` are replaced, and every dict entry of a list delta must carry an
/// `index` (a `RuntimeError` inside `.stream()` otherwise).
fn openai_accumulate(
    acc: &mut Map<String, Value>,
    delta: &Map<String, Value>,
) -> Result<(), String> {
    for (k, dv) in delta {
        let Some(av) = acc.get_mut(k) else {
            acc.insert(k.clone(), dv.clone());
            continue;
        };
        if av.is_null() || k == "index" || k == "type" {
            *av = dv.clone();
            continue;
        }
        match (av, dv) {
            (Value::String(a), Value::String(d)) => a.push_str(d),
            (Value::Object(a), Value::Object(d)) => openai_accumulate(a, d)?,
            (Value::Array(a), Value::Array(d)) => {
                if a.iter().all(|x| x.is_string() || x.is_number()) {
                    a.extend(d.iter().cloned());
                    continue;
                }
                for e in d {
                    let eo = e.as_object().ok_or(format!("not a dict: {e}"))?;
                    let i = eo.get("index").and_then(Value::as_u64).ok_or(format!(
                        "Expected list delta entry to have an `index` key; {e}"
                    ))?;
                    let i = usize::try_from(i).unwrap();
                    if i >= a.len() {
                        a.push(e.clone());
                    } else {
                        openai_accumulate(a[i].as_object_mut().unwrap(), eo)?;
                    }
                }
            }
            (av, dv) => *av = dv.clone(),
        }
    }
    Ok(())
}

/// The assistant message an openai-python `.stream()` consumer ends up with.
fn openai_stream_message(evs: &[Event]) -> Result<Value, String> {
    let mut msg = Map::new();
    for c in chunks(evs) {
        if let Some(d) = c.pointer("/choices/0/delta").and_then(Value::as_object) {
            openai_accumulate(&mut msg, d)?;
        }
    }
    Ok(Value::Object(msg))
}

fn ant_sse(events: &[Value]) -> String {
    events
        .iter()
        .map(|e| format!("event: {}\ndata: {e}\n\n", e["type"].as_str().unwrap()))
        .collect()
}

/// A Claude turn: `blocks` (each a whole content block, its text streamed as one delta), then
/// `stop`.
fn claude_turn(blocks: &[Value], stop: &str) -> String {
    let mut evs = vec![
        json!({"type": "message_start", "message": {"id": "msg_1", "type": "message",
        "role": "assistant", "model": "claude-haiku-4-5", "content": [],
        "usage": {"input_tokens": 10, "output_tokens": 1}}}),
    ];
    for (index, b) in blocks.iter().enumerate() {
        let (start, delta) = match b["type"].as_str().unwrap() {
            "thinking" => (
                json!({"type": "thinking", "thinking": "", "signature": ""}),
                vec![
                    json!({"type": "thinking_delta", "thinking": b["thinking"]}),
                    json!({"type": "signature_delta", "signature": b["signature"]}),
                ],
            ),
            "text" => (
                json!({"type": "text", "text": ""}),
                vec![json!({"type": "text_delta", "text": b["text"]})],
            ),
            "tool_use" => (
                json!({"type": "tool_use", "id": b["id"], "name": b["name"], "input": {}}),
                vec![json!({"type": "input_json_delta", "partial_json": b["input"].to_string()})],
            ),
            _ => (b.clone(), vec![]),
        };
        evs.push(json!({"type": "content_block_start", "index": index, "content_block": start}));
        for d in delta {
            evs.push(json!({"type": "content_block_delta", "index": index, "delta": d}));
        }
        evs.push(json!({"type": "content_block_stop", "index": index}));
    }
    evs.push(json!({"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {"output_tokens": 30}}));
    evs.push(json!({"type": "message_stop"}));
    ant_sse(&evs)
}

/// The Chat client's next turn onto Messages: `assistant` echoed as the SDK accumulated it.
fn next_turn(assistant: Value) -> Value {
    let body = json!({
        "model": "claude-haiku-4-5", "reasoning_effort": "low", "max_tokens": 2000,
        "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}],
        "messages": [
            {"role": "user", "content": "weather in Paris?"},
            assistant,
            {"role": "tool", "tool_call_id": "toolu_01", "content": "sunny"},
        ],
    });
    serde_json::from_slice(&request(
        Chat,
        Messages,
        &serde_json::to_vec(&body).unwrap(),
        "claude-haiku-4-5",
    ))
    .unwrap()
}

/// A streaming Chat client on a Claude row with reasoning: turn 2 was a 400 ("thinking.signature:
/// Field required"). The signature rode a `thinking_signature` string the request side never read,
/// and the echoed `reasoning_content` became an unsigned block. Now each finished block arrives
/// whole on the `thinking` list — what the non-stream body returns — and `messages.append(final
/// .choices[0].message)` sends back exactly the signed blocks.
/// claim: T2
#[test]
fn a_streamed_claude_turn_goes_back_signed() {
    let blocks = [
        json!({"type": "thinking", "thinking": "Need weather.", "signature": "SIG_ONE"}),
        json!({"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {"city": "Paris"}}),
    ];
    let evs = stream(Messages, Chat, &claude_turn(&blocks, "tool_use"));
    let msg = openai_stream_message(&evs).unwrap();
    assert_eq!(
        msg["reasoning_content"], "Need weather.",
        "still streamed as text: {msg}"
    );
    assert_eq!(
        msg["thinking"],
        json!([{"index": 0, "type": "thinking", "thinking": "Need weather.", "signature": "SIG_ONE"}])
    );
    assert!(msg.get("thinking_signature").is_none(), "{msg}");
    // The stream and the non-stream body carry the same blocks.
    let body = json_resp(
        Messages,
        Chat,
        &anthropic_msg(
            json!(blocks),
            "tool_use",
            json!({"input_tokens": 1, "output_tokens": 1}),
        ),
    );
    assert_eq!(
        body["choices"][0]["message"]["thinking"][0]["signature"],
        msg["thinking"][0]["signature"]
    );

    let up = next_turn(msg);
    let turn = &up["messages"][1]["content"];
    assert_eq!(
        turn[0],
        json!({"type": "thinking", "thinking": "Need weather.", "signature": "SIG_ONE"}),
        "{up}"
    );
    assert_eq!(turn[1]["type"], "tool_use");
    assert_eq!(turn.as_array().unwrap().len(), 2, "{turn}");
}

/// Interleaved thinking signs each block; two blocks' signatures on one string concatenated
/// ("S1S2") beyond repair. Two `redacted_thinking` blocks had no `index`, which openai-python's
/// accumulator requires on the second list delta: `.stream()` itself raised.
#[test]
fn every_thinking_block_streams_whole_under_its_own_index() {
    let blocks = [
        json!({"type": "redacted_thinking", "data": "RD1"}),
        json!({"type": "thinking", "thinking": "A", "signature": "S1"}),
        json!({"type": "text", "text": "x"}),
        json!({"type": "thinking", "thinking": "B", "signature": "S2"}),
        json!({"type": "redacted_thinking", "data": "RD2"}),
        json!({"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {}}),
    ];
    let src = claude_turn(&blocks, "tool_use");
    let msg = openai_stream_message(&stream(Messages, Chat, &src)).unwrap();
    assert_eq!(
        msg["thinking"],
        json!([
            {"index": 0, "type": "redacted_thinking", "data": "RD1"},
            {"index": 1, "type": "thinking", "thinking": "A", "signature": "S1"},
            {"index": 2, "type": "thinking", "thinking": "B", "signature": "S2"},
            {"index": 3, "type": "redacted_thinking", "data": "RD2"},
        ])
    );
    let turn = next_turn(msg)["messages"][1]["content"].clone();
    let kinds: Vec<(&str, &str)> = turn
        .as_array()
        .unwrap()
        .iter()
        .map(|b| {
            (
                b["type"].as_str().unwrap(),
                b["signature"]
                    .as_str()
                    .or_else(|| b["data"].as_str())
                    .unwrap_or(""),
            )
        })
        .collect();
    assert_eq!(
        kinds,
        [
            ("redacted_thinking", "RD1"),
            ("thinking", "S1"),
            ("thinking", "S2"),
            ("redacted_thinking", "RD2"),
            ("text", ""),
            ("tool_use", ""),
        ]
    );

    // A Responses client gets one signed `reasoning` item per thinking block.
    let evs = stream(Messages, Responses, &src);
    let resp = assert_responses_lifecycle(&evs, "response.completed");
    let sigs: Vec<&Value> = resp["output"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|i| i["type"] == "reasoning")
        .map(|i| &i["encrypted_content"])
        .collect();
    assert_eq!(sigs, [&json!("rs_gw:S1"), &json!("rs_gw:S2")], "{resp:#}");

    // A block that never got its signature is never offered for replay.
    let mut unsigned = blocks.to_vec();
    unsigned[1]["signature"] = json!("");
    let msg = openai_stream_message(&stream(Messages, Chat, &claude_turn(&unsigned, "tool_use")))
        .unwrap();
    assert_eq!(msg["reasoning_content"], "AB");
    assert!(!msg["thinking"].to_string().contains("\"A\""), "{msg}");
}

// ---- custom (free-form) tool calls (second audit) ---------------------------------------------

/// OpenAI streaming a custom tool call (shape from a live gpt-5-nano capture, 2026-09-30).
const OPENAI_CUSTOM_SSE: &str = r#"data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1790808721,"model":"gpt-5-nano","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_c1","type":"custom","custom":{"name":"run_code","input":""}}],"refusal":null},"finish_reason":null}]}

data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1790808721,"model":"gpt-5-nano","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"custom":{"input":"print"}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1790808721,"model":"gpt-5-nano","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"custom":{"input":"(1+1)"}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1790808721,"model":"gpt-5-nano","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]

"#;

/// A Responses client (Codex's `apply_patch`) on a Chat Completions row: the custom call became a
/// `function_call` with an empty name and `{}` arguments — the patch was lost.
#[test]
fn a_chat_custom_tool_call_reaches_responses_as_a_custom_tool_call() {
    let body = chat_completion(
        json!({"role": "assistant", "content": null, "tool_calls": [
            {"id": "call_1", "type": "custom", "custom": {"name": "apply_patch", "input": "*** Begin Patch"}},
            {"id": "call_2", "type": "function", "function": {"name": "f", "arguments": "{}"}},
        ]}),
        "tool_calls",
        json!({}),
    );
    let r = json_resp(Chat, Responses, &body);
    let out = r["output"].as_array().unwrap();
    assert_eq!(out[0]["type"], "custom_tool_call", "{r}");
    assert_eq!(out[0]["call_id"], "call_1");
    assert_eq!(out[0]["name"], "apply_patch");
    assert_eq!(out[0]["input"], "*** Begin Patch");
    assert!(out[0]["id"].as_str().unwrap().starts_with("ctc_"), "{r}");
    assert_eq!(out[1]["type"], "function_call");

    let evs = stream(Chat, Responses, OPENAI_CUSTOM_SSE);
    let resp = assert_responses_lifecycle(&evs, "response.completed");
    let item = &resp["output"][0];
    assert_eq!(item["type"], "custom_tool_call", "{resp:#}");
    assert_eq!(item["call_id"], "call_c1");
    assert_eq!(item["name"], "run_code");
    assert_eq!(item["input"], "print(1+1)");
    let deltas: String = named(&evs, "response.custom_tool_call_input.delta")
        .iter()
        .map(|v| v["delta"].as_str().unwrap())
        .collect();
    assert_eq!(deltas, "print(1+1)");
    assert_eq!(
        one(&evs, "response.custom_tool_call_input.done")["input"],
        "print(1+1)"
    );
    assert!(named(&evs, "response.function_call_arguments.delta").is_empty());
}

/// A Chat client on a Responses-only row (the `-codex` models) with a custom tool: the call was
/// dropped, and the turn ended `stop` with nothing to run.
#[test]
fn a_responses_custom_tool_call_reaches_chat_as_a_custom_tool_call() {
    let body = json!({"id": "resp_1", "object": "response", "created_at": 1, "status": "completed", "model": "gpt-5.3-codex",
        "output": [{"type": "custom_tool_call", "id": "ctc_1", "call_id": "call_1", "name": "run_code", "input": "print(1)", "status": "completed"}]});
    let c = json_resp(Responses, Chat, &body);
    assert_eq!(c["choices"][0]["finish_reason"], "tool_calls", "{c}");
    assert_eq!(
        c["choices"][0]["message"]["tool_calls"],
        json!([{"id": "call_1", "type": "custom", "custom": {"name": "run_code", "input": "print(1)"}}])
    );

    let src = concat!(
        "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"created_at\":1,\"model\":\"gpt-5.3-codex\"}}\n\n",
        "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"custom_tool_call\",\"id\":\"ctc_1\",\"call_id\":\"call_1\",\"name\":\"run_code\",\"input\":\"\"}}\n\n",
        "event: response.custom_tool_call_input.delta\ndata: {\"type\":\"response.custom_tool_call_input.delta\",\"output_index\":0,\"item_id\":\"ctc_1\",\"delta\":\"print\"}\n\n",
        "event: response.custom_tool_call_input.delta\ndata: {\"type\":\"response.custom_tool_call_input.delta\",\"output_index\":0,\"item_id\":\"ctc_1\",\"delta\":\"(1)\"}\n\n",
        "event: response.custom_tool_call_input.done\ndata: {\"type\":\"response.custom_tool_call_input.done\",\"output_index\":0,\"item_id\":\"ctc_1\",\"input\":\"print(1)\"}\n\n",
        "event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"custom_tool_call\",\"id\":\"ctc_1\",\"call_id\":\"call_1\",\"name\":\"run_code\",\"input\":\"print(1)\"}}\n\n",
        "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"usage\":{\"input_tokens\":5,\"output_tokens\":3}}}\n\n",
    );
    let evs = stream(Responses, Chat, src);
    let msg = openai_stream_message(&evs).unwrap();
    assert_eq!(
        msg["tool_calls"],
        json!([{"index": 0, "id": "call_1", "type": "custom", "custom": {"name": "run_code", "input": "print(1)"}}]),
        "{msg}"
    );
    assert_eq!(finish_reason(&evs), "tool_calls");
    // A call announced only by its closing item still arrives whole.
    let late = src
        .lines()
        .filter(|l| !l.contains("output_item.added") && !l.contains("input.delta"))
        .collect::<Vec<_>>()
        .join("\n");
    let msg = openai_stream_message(&stream(Responses, Chat, &late)).unwrap();
    assert_eq!(msg["tool_calls"][0]["custom"]["input"], "print(1)", "{msg}");
    assert_eq!(msg["tool_calls"][0]["custom"]["name"], "run_code");
}

// ---- streams that fail or stop mid-way (second audit) -----------------------------------------

/// OpenRouter reports a provider that died mid-generation as a chunk with `choices` *and* an
/// `error`, and `finish_reason: "error"`. A Messages client got `end_turn` and a Responses client
/// `completed` on a half-written answer.
#[test]
fn an_openrouter_mid_stream_error_is_an_error() {
    let src = concat!(
        "data: {\"id\":\"gen-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"moonshotai/kimi-k3\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello, the answer is\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"gen-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"moonshotai/kimi-k3\",\"provider\":\"Fireworks\",\"error\":{\"code\":\"server_error\",\"message\":\"Provider disconnected unexpectedly\"},\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\"},\"finish_reason\":\"error\"}]}\n\n",
        "data: [DONE]\n\n",
    );
    let evs = stream(Chat, Messages, src);
    let names: Vec<&str> = evs.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names.last(), Some(&"error"), "{names:?}");
    assert!(!names.contains(&"message_stop") && !names.contains(&"message_delta"));
    assert_eq!(
        one(&evs, "error")["error"]["message"],
        "Provider disconnected unexpectedly"
    );

    let evs = stream(Chat, Responses, src);
    let names: Vec<&str> = evs.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names.last(), Some(&"response.failed"), "{names:?}");
    assert!(!names.contains(&"response.completed"), "{names:?}");
    assert_eq!(
        one(&evs, "error")["message"],
        "Provider disconnected unexpectedly"
    );

    // `finish_reason: "error"` alone, and the same failure in a non-stream body.
    let bare = src.replace(
        ",\"error\":{\"code\":\"server_error\",\"message\":\"Provider disconnected unexpectedly\"}",
        "",
    );
    assert_ne!(bare, src);
    let evs = stream(Chat, Messages, &bare);
    assert_eq!(evs.last().unwrap().0, "error", "{evs:#?}");
    let body = json!({"id": "gen-1", "object": "chat.completion", "model": "m",
        "error": {"code": 502, "message": "Provider disconnected unexpectedly"},
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "error"}]});
    let anth = json_resp(Chat, Messages, &body);
    assert_eq!(anth["type"], "error", "{anth}");
    assert_eq!(
        anth["error"]["message"],
        "Provider disconnected unexpectedly"
    );
    let resp = json_resp(Chat, Responses, &body);
    assert!(resp.get("output").is_none(), "{resp}");
    assert_eq!(
        resp["error"]["message"],
        "Provider disconnected unexpectedly"
    );
}

/// An upstream stream that ends cleanly without saying how the response ended (no stop reason, no
/// end marker) was cut short. The bridge used to close it as a success — `end_turn`,
/// `response.completed`, a bare `[DONE]` — on a half-written answer.
/// claim: T7
#[test]
fn a_stream_that_ends_without_saying_how_is_an_error_for_every_client() {
    let cut_claude = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"model\":\"c\",\"usage\":{\"input_tokens\":3,\"output_tokens\":1}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"half an ans\"}}\n\n",
    );
    let evs = stream(Messages, Chat, cut_claude);
    assert_eq!(evs.last().unwrap().1, "[DONE]");
    let err = &evs[evs.len() - 2].1;
    assert_eq!(err["error"]["code"], "stream_truncated", "{evs:#?}");
    assert_eq!(
        finish_reason(&evs),
        Value::Null,
        "no finish reason is invented"
    );

    let evs = stream(Messages, Responses, cut_claude);
    let names: Vec<&str> = evs.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names.last(), Some(&"response.failed"), "{names:?}");
    assert!(!names.contains(&"response.completed"), "{names:?}");
    assert_eq!(one(&evs, "error")["code"], "stream_truncated");
    assert_eq!(
        one(&evs, "response.failed")["response"]["error"]["code"],
        "server_error",
        "a failed Response's code is a closed set; a typed client rejects any other"
    );

    let cut_chat = "data: {\"id\":\"c\",\"model\":\"gpt-5\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"half\"}}]}\n\n";
    let evs = stream(Chat, Messages, cut_chat);
    let names: Vec<&str> = evs.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names.last(), Some(&"error"), "{names:?}");
    assert!(!names.contains(&"message_stop"), "{names:?}");
    assert_eq!(
        one(&evs, "error")["error"]["type"],
        "api_error",
        "an Anthropic SDK raises it"
    );
    // Nothing at all is no answer either.
    assert_eq!(stream(Chat, Messages, "").last().unwrap().0, "error");

    // Each upstream's own end marker still closes cleanly.
    let ended = format!("{cut_chat}data: [DONE]\n\n");
    assert_eq!(
        stream(Chat, Messages, &ended).last().unwrap().0,
        "message_stop"
    );
    let stopped = cut_chat.replace(
        "\"content\":\"half\"}}",
        "\"content\":\"half\"},\"finish_reason\":\"stop\"}",
    );
    assert_eq!(
        stream(Chat, Responses, &stopped).last().unwrap().0,
        "response.completed"
    );
}

/// openai-python's `accumulate_delta`, which builds `.stream()`'s final message: a string adds
/// to the one before it unless its key is `index` or `type`, objects merge, and a list of objects
/// merges by each entry's `index`.
fn sdk_accumulate(acc: &mut Map<String, Value>, delta: &Map<String, Value>) {
    for (k, d) in delta {
        let Some(a) = acc.get_mut(k) else {
            acc.insert(k.clone(), d.clone());
            continue;
        };
        if a.is_null() || k == "index" || k == "type" {
            *a = d.clone();
            continue;
        }
        match (a, d) {
            (Value::String(a), Value::String(d)) => a.push_str(d),
            (Value::Object(a), Value::Object(d)) => sdk_accumulate(a, d),
            (Value::Array(a), Value::Array(d)) => {
                for e in d {
                    let i = e["index"].as_u64().unwrap() as usize;
                    match a.get_mut(i) {
                        Some(x) => {
                            sdk_accumulate(x.as_object_mut().unwrap(), e.as_object().unwrap())
                        }
                        None => a.insert(i, e.clone()),
                    }
                }
            }
            _ => {}
        }
    }
}

fn sdk_message(evs: &[Event]) -> Value {
    let mut msg = Map::new();
    for c in chunks(evs) {
        if let Some(d) = c.pointer("/choices/0/delta").and_then(Value::as_object) {
            sdk_accumulate(&mut msg, d);
        }
    }
    Value::Object(msg)
}

/// claim: S2
#[test]
fn a_chat_relay_sends_each_identity_field_once() {
    // As OpenRouter sent it, the SDK's message is unusable on the next turn.
    let raw = sdk_message(&events(OPENROUTER_CLAUDE_SSE));
    assert!(
        raw["role"]
            .as_str()
            .unwrap()
            .starts_with("assistantassistant"),
        "{raw}"
    );

    let evs = stream(Chat, Chat, OPENROUTER_CLAUDE_SSE);
    let msg = sdk_message(&evs);
    assert_eq!(msg["role"], "assistant", "{msg}");
    let rd = &msg["reasoning_details"][0];
    assert_eq!(rd["format"], "anthropic-claude-v1", "{msg}");
    assert_eq!(rd["text"], "The user wants two lookups.");
    assert_eq!(rd["signature"], "EqIFCpwBsig");
    let calls = msg["tool_calls"].as_array().unwrap();
    assert_eq!(calls.len(), 2, "{msg}");
    assert_eq!(calls[0]["id"], "toolu_bdrk_01");
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    assert_eq!(calls[0]["function"]["arguments"], r#"{"city": "Paris"}"#);
    assert_eq!(calls[1]["id"], "toolu_bdrk_02");
    assert_eq!(calls[1]["function"]["arguments"], r#"{"city": "Rome"}"#);

    // Everything but the repeats arrives as sent: usage, finish reasons, `[DONE]`.
    assert_eq!(chat_usage(&evs)["prompt_tokens"], 597);
    assert_eq!(finish_reason(&evs), "tool_calls");
    assert_eq!(evs.last().unwrap().1, "[DONE]");
    assert_eq!(
        chunks(&evs).len(),
        chunks(&events(OPENROUTER_CLAUDE_SSE)).len()
    );
}

#[test]
fn a_chat_relay_forwards_untouched_events_byte_for_byte() {
    // OpenAI's own stream sends each identity field once: nothing is re-written, and a comment
    // (OpenRouter's keep-alive) passes through too.
    let src = format!(": OPENROUTER PROCESSING\n\n{OPENAI_PARALLEL_SSE}");
    assert_eq!(run(Chat, Chat, &[src.as_bytes()]), src);
}

#[test]
fn identity_fields_are_tracked_per_choice_and_entry() {
    let mut id = ChatIdentity::default();
    let mut first = json!({"choices":[
        {"index":0,"delta":{"role":"assistant","reasoning_details":[{"index":0,"format":"f"}]}},
        {"index":1,"delta":{"role":"assistant"}}]});
    assert!(!id.strip(&mut first), "a first sighting is kept: {first}");
    let mut again = json!({"choices":[
        {"index":1,"delta":{"role":"assistant","content":"x"}},
        {"index":0,"delta":{"reasoning_details":[{"index":0,"format":"f"},{"index":1,"format":"f"}]}}]});
    assert!(id.strip(&mut again));
    assert_eq!(
        again,
        json!({"choices":[
            {"index":1,"delta":{"content":"x"}},
            {"index":0,"delta":{"reasoning_details":[{"index":0},{"index":1,"format":"f"}]}}]})
    );
}

// ---- verification phase 0: zero-argument tool calls --------------------------------------------

/// A Claude turn calling a tool that takes no arguments, as Anthropic streams it: the block opens
/// with `input: {}` and its one `input_json_delta` carries an empty `partial_json`.
fn claude_zero_arg_call_sse() -> String {
    ant_sse(&[
        json!({"type": "message_start", "message": {"id": "msg_1", "type": "message",
        "role": "assistant", "model": "claude-haiku-4-5", "content": [],
        "usage": {"input_tokens": 10, "output_tokens": 1}}}),
        json!({"type": "content_block_start", "index": 0,
        "content_block": {"type": "tool_use", "id": "toolu_01", "name": "get_time", "input": {}}}),
        json!({"type": "content_block_delta", "index": 0,
        "delta": {"type": "input_json_delta", "partial_json": ""}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 12}}),
        json!({"type": "message_stop"}),
    ])
}

/// A zero-argument tool call streamed from Claude must reach a Chat or Responses client as
/// `arguments: "{}"`, the same string the non-stream body carries. `""` is not JSON: the Agents
/// SDK `json.loads` it and Vercel's AI SDK rejects it as invalid tool input.
/// claim: TRN-9
/// defect: D13
#[test]
fn a_streamed_zero_argument_call_has_json_arguments() {
    // The non-stream body is the baseline: it already says "{}".
    let body = json_resp(
        Messages,
        Chat,
        &anthropic_msg(
            json!([{"type": "tool_use", "id": "toolu_01", "name": "get_time", "input": {}}]),
            "tool_use",
            json!({"input_tokens": 10, "output_tokens": 12}),
        ),
    );
    assert_eq!(
        body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
        "{}"
    );

    let src = claude_zero_arg_call_sse();

    // Messages → Chat: the arguments an OpenAI SDK accumulates.
    let chat = stream(Messages, Chat, &src);
    let calls = chat_tool_calls(&chat);
    assert_eq!(calls.len(), 1, "{chat:#?}");
    assert_eq!(calls[0].1, "get_time");
    assert_eq!(calls[0].2, "{}", "Messages→Chat stream: {chat:#?}");

    // Messages → Responses: the deltas, the `.done` event and the completed item all agree.
    let evs = stream(Messages, Responses, &src);
    let resp = assert_responses_lifecycle(&evs, "response.completed");
    let deltas: String = named(&evs, "response.function_call_arguments.delta")
        .iter()
        .map(|v| v["delta"].as_str().unwrap())
        .collect();
    let done = one(&evs, "response.function_call_arguments.done");
    let item = &resp["output"][0];
    assert_eq!(item["type"], "function_call", "{resp:#}");
    assert_eq!(
        (deltas.as_str(), &done["arguments"], &item["arguments"]),
        ("{}", &json!("{}"), &json!("{}")),
        "Messages→Responses stream (deltas, .done, completed item): {evs:#?}"
    );
}

// ---- verification phase 0: translation detail (D44-D50) ----------------------------------------

/// Anthropic streams one content block at a time: each `content_block_start` follows the previous
/// block's `content_block_stop`, and deltas land only on the open block. Returns the first breach.
fn messages_block_sequencing_violation(evs: &[Event]) -> Option<String> {
    let mut open: Option<u64> = None;
    for (name, v) in evs {
        let idx = v["index"].as_u64();
        match name.as_str() {
            "content_block_start" => {
                if let Some(o) = open {
                    return Some(format!("block {idx:?} started while block {o} is open"));
                }
                open = idx;
            }
            "content_block_delta" if idx != open => {
                return Some(format!("delta for block {idx:?} while {open:?} is open"));
            }
            "content_block_stop" => {
                if idx != open {
                    return Some(format!("stop for block {idx:?} while {open:?} is open"));
                }
                open = None;
            }
            _ => {}
        }
    }
    None
}

/// Parallel tool calls from a Chat upstream reach a Messages client as sequential blocks
/// (start, deltas, stop, then the next start), whether the upstream sends the calls one after
/// another or interleaves their argument deltas. The Anthropic SDK accumulates either way, but
/// harnesses that act on `content_block_stop` (or assert one open block) see two open at once.
/// claim: TRN-12
/// defect: D45
#[test]
fn parallel_tool_use_blocks_are_sequential_on_a_messages_stream() {
    let sequential = concat!(
        "data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"function\":{\"name\":\"f\",\"arguments\":\"{\\\"a\\\":1}\"}}]}}]}\n\n",
        "data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"call_b\",\"function\":{\"name\":\"g\",\"arguments\":\"{\\\"b\\\":2}\"}}]}}]}\n\n",
        "data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2}}\n\n",
    );
    let interleaved = concat!(
        "data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"function\":{\"name\":\"f\",\"arguments\":\"\"}},{\"index\":1,\"id\":\"call_b\",\"function\":{\"name\":\"g\",\"arguments\":\"\"}}]}}]}\n\n",
        "data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"a\\\":\"}}]}}]}\n\n",
        "data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"{\\\"b\\\":2}\"}}]}}]}\n\n",
        "data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"1}\"}}]}}]}\n\n",
        "data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2}}\n\n",
    );
    let mut bad = Vec::new();
    for (what, src) in [("sequential", sequential), ("interleaved", interleaved)] {
        let evs = stream(Chat, Messages, src);
        // The calls themselves still arrive whole; only the sequencing is in question.
        let content = anthropic_content(&evs);
        assert_eq!(content.len(), 2, "{what}: {content:#?}");
        assert_eq!(content[0]["input"], json!({"a": 1}), "{what}");
        assert_eq!(content[1]["input"], json!({"b": 2}), "{what}");
        if let Some(why) = messages_block_sequencing_violation(&evs) {
            let order: Vec<String> = evs
                .iter()
                .filter(|(n, _)| n.starts_with("content_block_"))
                .map(|(n, v)| format!("{}{}", &n["content_block_".len()..], v["index"]))
                .collect();
            bad.push(format!("{what}: {why} ({})", order.join(" ")));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// An upstream context-overflow error, translated for each client, carries what that client's
/// harness compacts on: Claude Code looks for "prompt is too long" in a Messages error, Codex and
/// the Agents SDK for code `context_length_exceeded` in a Chat or Responses one.
/// claim: TRN-18
/// defect: D46
#[test]
fn a_context_overflow_error_carries_what_each_harness_compacts_on() {
    let openai = json!({"error": {
        "message": "This model's maximum context length is 128000 tokens. However, your messages resulted in 130512 tokens. Please reduce the length of the messages.",
        "type": "invalid_request_error", "param": "messages", "code": "context_length_exceeded"}});
    let anthropic = json!({"type": "error", "error": {"type": "invalid_request_error",
        "message": "prompt is too long: 210345 tokens > 200000 maximum"}});
    let mut bad = Vec::new();

    let m = json_status(Chat, Messages, 400, &openai);
    let msg = m["error"]["message"].as_str().unwrap_or("").to_lowercase();
    if !msg.contains("prompt is too long") {
        bad.push(format!("Chat upstream → Messages client: {m}"));
    }
    for client in [Chat, Responses] {
        let c = json_status(Messages, client, 400, &anthropic);
        if c["error"]["code"] != "context_length_exceeded" {
            bad.push(format!("Messages upstream → {client:?} client: {c}"));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// A mid-stream error reaches a Responses client's `response.failed` with its code intact when it
/// is one a harness acts on: Codex compacts on `context_length_exceeded` and stops retrying on
/// `insufficient_quota`, both read from `response.failed`'s `error.code`, which real OpenAI fills
/// with those values though the published enum omits them.
/// claim: TRN-19
/// defect: D46
#[test]
fn a_mid_stream_overflow_or_quota_code_survives_response_failed() {
    let mut bad = Vec::new();
    for (code, typ) in [
        ("context_length_exceeded", "invalid_request_error"),
        ("insufficient_quota", "insufficient_quota"),
    ] {
        let src = format!(
            concat!(
                "data: {{\"id\":\"c\",\"model\":\"gpt-5\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"par\"}}}}]}}\n\n",
                "data: {{\"error\":{{\"message\":\"boom\",\"type\":\"{typ}\",\"code\":\"{code}\"}}}}\n\n",
                "data: [DONE]\n\n",
            ),
            typ = typ,
            code = code,
        );
        let evs = stream(Chat, Responses, &src);
        assert_eq!(one(&evs, "error")["code"], code, "the error event keeps it");
        let failed = &one(&evs, "response.failed")["response"];
        if failed["error"]["code"] != code {
            bad.push(format!(
                "{code}: response.failed carries {}",
                failed["error"]
            ));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// A Messages-compatible host may send a tool call's whole input on `content_block_start` and no
/// `input_json_delta` after it. The input is the call's arguments, so a Chat or Responses client
/// must receive it rather than `""` / `{}`.
/// claim: TRN-13
/// defect: D50
#[test]
fn a_tool_input_sent_in_the_start_block_reaches_the_client() {
    let src = ant_sse(&[
        json!({"type": "message_start", "message": {"id": "msg_1", "type": "message",
        "role": "assistant", "model": "claude-haiku-4-5", "content": [],
        "usage": {"input_tokens": 10, "output_tokens": 1}}}),
        json!({"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use",
        "id": "toolu_01", "name": "get_weather", "input": {"city": "Paris"}}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 12}}),
        json!({"type": "message_stop"}),
    ]);
    let mut bad = Vec::new();

    let chat = stream(Messages, Chat, &src);
    let calls = chat_tool_calls(&chat);
    let args = calls.first().map(|c| c.2.clone()).unwrap_or_default();
    if serde_json::from_str::<Value>(&args).ok() != Some(json!({"city": "Paris"})) {
        bad.push(format!("Messages→Chat arguments: {args:?}"));
    }

    let evs = stream(Messages, Responses, &src);
    let resp = assert_responses_lifecycle(&evs, "response.completed");
    let item_args = resp["output"][0]["arguments"].as_str().unwrap_or("");
    if serde_json::from_str::<Value>(item_args).ok() != Some(json!({"city": "Paris"})) {
        bad.push(format!("Messages→Responses item arguments: {item_args:?}"));
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// xAI puts reasoning beside `completion_tokens`; a Messages or Responses client is shown the output
/// it is billed (visible + reasoning), the same count `usage::openai_body` bills.
///
/// claim: BIL-6, BIL-9
/// defect: D64
#[test]
fn translated_usage_counts_reasoning_reported_beside_completion_tokens() {
    let xai = json!({
        "id": "x", "object": "chat.completion", "created": 1, "model": "grok-4.3",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "391"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 202, "completion_tokens": 7, "total_tokens": 376,
                  "completion_tokens_details": {"reasoning_tokens": 167}},
    });
    let ant = json_resp(Chat, Messages, &xai);
    assert_eq!(ant["usage"]["output_tokens"], 174, "{ant}");
    let resp = json_resp(Chat, Responses, &xai);
    assert_eq!(resp["usage"]["output_tokens"], 174, "{resp}");
    assert_eq!(
        resp["usage"]["output_tokens_details"]["reasoning_tokens"], 167,
        "{resp}"
    );
}
