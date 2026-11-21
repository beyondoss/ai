use super::*;

/// A current (adaptive-generation) Claude id, what most catalog walks land on.
const OPUS: &str = "claude-opus-4-8";

#[test]
fn json_buffer_refuses_past_the_cap_and_keeps_what_it_has() {
    let mut t = TranslateState::new(Endpoint::ChatCompletions);
    assert!(t.push_json(&vec![b' '; MAX_TRANSLATE_BUFFER - 1]));
    assert!(t.push_json(b"{"), "exactly at the cap is allowed");
    assert!(!t.push_json(b"}"), "one byte past is refused");
    assert_eq!(
        t.json_buf.len(),
        MAX_TRANSLATE_BUFFER,
        "a refused chunk appends nothing"
    );
}

#[test]
fn an_unterminated_event_is_held_and_reported_as_pending() {
    let mut b = SseBridge::new(Endpoint::ChatCompletions, Endpoint::Messages);
    let big = format!("event: content_block_delta\ndata: {}", "x".repeat(10_000));
    for chunk in big.as_bytes().chunks(1000) {
        assert!(b.feed(chunk, false).is_empty());
    }
    assert_eq!(b.pending_len(), big.len());
}

/// A blank line ends an event whatever line endings frame it; a lone `\r\n` (one line ending)
/// does not.
#[test]
fn an_event_ends_at_a_blank_line_of_either_ending() {
    assert_eq!(find_event_end(b"data: a\n\nb", 0), Some(9));
    assert_eq!(find_event_end(b"data: a\r\n\r\nb", 0), Some(11));
    assert_eq!(find_event_end(b"data: a\n\r\nb", 0), Some(10));
    assert_eq!(find_event_end(b"data: a\r\n\nb", 0), Some(10));
    assert_eq!(find_event_end(b"\n\r\nb", 0), Some(3));
    assert_eq!(find_event_end(b"data: a\r\ndata: b\r\n", 0), None);
    assert_eq!(find_event_end(b"data: a\n\r", 0), None, "undecided");
}

/// The resumable search must find exactly what a from-scratch search finds, including a
/// terminator split across chunks at every possible point, for both line endings.
#[test]
fn resumable_event_scan_matches_a_fresh_scan_at_every_split() {
    for stream in [
        &b"data: a\n\ndata: bb\n\n: c\n\n"[..],
        &b"data: a\r\n\r\ndata: bb\r\n\r\n"[..],
        &b"data: a\r\ndata: b\n\ndata: c\r\n\r\n"[..],
        // A `\n` line ending then a `\r\n` blank line, and the reverse: SSE allows both.
        &b"data: a\n\r\ndata: b\r\n\ndata: c\n\n"[..],
    ] {
        // As `SseBridge::feed` holds it: events taken by cursor, the buffer compacted per part.
        let events = |parts: &[&[u8]]| {
            let (mut buf, mut scanned, mut got) = (Vec::new(), 0, Vec::new());
            for part in parts {
                buf.extend_from_slice(part);
                let mut start = 0;
                while let Some((s, e)) = next_event(&buf, &mut start, &mut scanned) {
                    got.push(buf[s..e].to_vec());
                }
                buf.drain(..start);
                scanned -= start;
            }
            got
        };
        let want = events(&[stream]);
        assert!(want.len() >= 2, "{want:?}");
        for split in 0..=stream.len() {
            let got = events(&[&stream[..split], &stream[split..]]);
            assert_eq!(
                got,
                want,
                "split at {split} of {:?}",
                String::from_utf8_lossy(stream)
            );
        }
    }
}

fn oai_req() -> Value {
    json!({
        "model": "claude-opus-4-8",
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"}
        ],
        "max_tokens": 16,
        "temperature": 0.2,
        "stream": true,
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
            }
        }],
        "tool_choice": "auto",
        "stream_options": {"include_usage": true}
    })
}

fn to_messages(body: &Value) -> Value {
    let out = request(
        Endpoint::ChatCompletions,
        Endpoint::Messages,
        &serde_json::to_vec(body).unwrap(),
        OPUS,
    );
    serde_json::from_slice(&out).unwrap()
}

fn count_markers(v: &Value) -> usize {
    match v {
        Value::Object(m) => {
            usize::from(m.contains_key("cache_control"))
                + m.values().map(count_markers).sum::<usize>()
        }
        Value::Array(a) => a.iter().map(count_markers).sum(),
        _ => 0,
    }
}

/// A stock OpenAI SDK sends no `cache_control`. Without a marker Anthropic caches nothing, so
/// the static prefix gets one.
/// claim: K1
#[test]
fn an_unmarked_request_caches_its_system_prompt() {
    let v = to_messages(&oai_req());
    assert_eq!(v["system"][0]["text"], "be brief");
    assert_eq!(v["system"][0]["cache_control"]["type"], "ephemeral");
    assert!(
        v["tools"][0].get("cache_control").is_none(),
        "one marker covers tools + system"
    );
    // Single turn: no conversation marker (a write costs 1.25x and a one-shot never reads it).
    assert_eq!(v["messages"][0]["content"], "hi");
    assert_eq!(count_markers(&v), 1, "{v}");
}

/// claim: K1
#[test]
fn without_a_system_prompt_the_last_tool_is_marked() {
    let mut req = oai_req();
    req["messages"] = json!([{"role": "user", "content": "hi"}]);
    req["tools"]
        .as_array_mut()
        .unwrap()
        .push(json!({"type": "function", "function": {"name": "second"}}));
    let v = to_messages(&req);
    assert!(v.get("system").is_none());
    assert!(v["tools"][0].get("cache_control").is_none());
    assert_eq!(v["tools"][1]["cache_control"]["type"], "ephemeral");
    assert_eq!(count_markers(&v), 1, "{v}");
}

/// Once the request is a conversation, the whole prefix is marked so the next turn reads it.
/// claim: K1
#[test]
fn a_conversation_marks_the_end_of_its_last_message() {
    let mut req = oai_req();
    req["messages"] = json!([
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": null, "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
        }]},
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        {"role": "user", "content": "and tomorrow?"}
    ]);
    let v = to_messages(&req);
    let last = v["messages"].as_array().unwrap().last().unwrap().clone();
    assert_eq!(last["role"], "user");
    let blocks = last["content"].as_array().unwrap();
    assert!(blocks[0].get("cache_control").is_none(), "{last}");
    assert_eq!(blocks.last().unwrap()["cache_control"]["type"], "ephemeral");
    assert_eq!(count_markers(&v), 2, "system + conversation: {v}");
}

#[test]
fn a_string_last_message_becomes_a_marked_text_block() {
    let mut req = oai_req();
    req["messages"] = json!([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "again"}
    ]);
    let v = to_messages(&req);
    assert_eq!(
        v["messages"][2]["content"],
        json!([{"type": "text", "text": "again", "cache_control": {"type": "ephemeral"}}])
    );
}

/// Anthropic rejects `cache_control` on a thinking block.
#[test]
fn the_conversation_marker_skips_trailing_thinking_blocks() {
    let mut req = oai_req();
    req["messages"] = json!([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "hello"},
            {"type": "thinking", "thinking": "hm", "signature": "sig"}
        ]}
    ]);
    let v = to_messages(&req);
    let blocks = v["messages"][1]["content"].as_array().unwrap();
    assert!(
        blocks.iter().any(|b| b["type"] == "thinking"),
        "fixture must carry a thinking block: {blocks:?}"
    );
    for b in blocks {
        if matches!(b["type"].as_str(), Some("thinking" | "redacted_thinking")) {
            assert!(b.get("cache_control").is_none(), "{b}");
        }
    }
    assert!(
        blocks.iter().any(|b| b.get("cache_control").is_some()),
        "a non-thinking block carries the marker: {blocks:?}"
    );
}

/// A client that manages its own caching keeps full control.
/// claim: K1
#[test]
fn a_client_marker_anywhere_disables_the_defaults() {
    let mut req = oai_req();
    req["messages"] = json!([
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": [
            {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral", "ttl": "1h"}}
        ]},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "again"}
    ]);
    let v = to_messages(&req);
    assert_eq!(count_markers(&v), 1, "{v}");
    assert_eq!(v["messages"][0]["content"][0]["cache_control"]["ttl"], "1h");
}

/// claim: K1
#[test]
fn responses_onto_messages_gets_the_same_defaults() {
    let body = json!({
        "model": "claude-opus-4-8",
        "instructions": "be brief",
        "input": "hi"
    });
    let out = request(
        Endpoint::Responses,
        Endpoint::Messages,
        &serde_json::to_vec(&body).unwrap(),
        OPUS,
    );
    let v: Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(v["system"][0]["cache_control"]["type"], "ephemeral", "{v}");
}

#[test]
fn openai_request_maps_system_tools_and_drops_stream_options() {
    let body = serde_json::to_vec(&oai_req()).unwrap();
    let out = request(Endpoint::ChatCompletions, Endpoint::Messages, &body, OPUS);
    let v: Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(v["system"][0]["text"], "be brief");
    assert_eq!(v["max_tokens"], 16);
    assert_eq!(v["stream"], true);
    assert!(v.get("stream_options").is_none(), "{v}");
    assert_eq!(v["messages"][0]["role"], "user");
    assert_eq!(v["messages"][0]["content"], "hi");
    assert_eq!(v["tools"][0]["name"], "get_weather");
    assert_eq!(
        v["tools"][0]["input_schema"]["properties"]["city"]["type"],
        "string"
    );
    assert_eq!(v["tool_choice"]["type"], "auto");
    assert_eq!(v["model"], "claude-opus-4-8");
}

/// Chat Completions nests a tool's description under `function`; the Responses shape is flat.
/// Every field has to come from the nested object, or the model loses what the tool is for.
#[test]
fn openai_tools_keep_their_description_on_responses() {
    let body = serde_json::to_vec(&oai_req()).unwrap();
    for from in [Endpoint::ChatCompletions, Endpoint::Messages] {
        let src = if from == Endpoint::Messages {
            request(Endpoint::ChatCompletions, Endpoint::Messages, &body, OPUS)
        } else {
            body.clone()
        };
        let v: Value =
            serde_json::from_slice(&request(from, Endpoint::Responses, &src, OPUS)).unwrap();
        let tool = &v["tools"][0];
        assert_eq!(tool["name"], "get_weather", "{from:?}: {v}");
        assert_eq!(tool["description"], "weather", "{from:?}: {v}");
        assert_eq!(tool["parameters"]["properties"]["city"]["type"], "string");
    }
}

#[test]
fn openai_request_defaults_max_tokens() {
    let body = br#"{"model":"x","messages":[{"role":"user","content":"hi"}]}"#;
    let v: Value = serde_json::from_slice(&request(
        Endpoint::ChatCompletions,
        Endpoint::Messages,
        body,
        OPUS,
    ))
    .unwrap();
    assert_eq!(v["max_tokens"], DEFAULT_MAX_TOKENS);
}

/// claim: T1
#[test]
fn tool_loop_round_trips_openai_through_anthropic() {
    let oai = json!({
        "model": "m",
        "max_tokens": 8,
        "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"}
            }]},
            {"role": "tool", "tool_call_id": "call_1", "content": "64F"}
        ],
        "tools": [{"type": "function", "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {}}
        }}]
    });
    let anth_bytes = request(
        Endpoint::ChatCompletions,
        Endpoint::Messages,
        &serde_json::to_vec(&oai).unwrap(),
        OPUS,
    );
    let anth: Value = serde_json::from_slice(&anth_bytes).unwrap();
    assert_eq!(anth["messages"][1]["role"], "assistant");
    assert_eq!(anth["messages"][1]["content"][0]["type"], "tool_use");
    assert_eq!(anth["messages"][1]["content"][0]["input"]["city"], "SF");
    assert_eq!(anth["messages"][2]["role"], "user");
    assert_eq!(anth["messages"][2]["content"][0]["type"], "tool_result");
    assert_eq!(anth["messages"][2]["content"][0]["tool_use_id"], "call_1");

    let back_bytes = request(
        Endpoint::Messages,
        Endpoint::ChatCompletions,
        &anth_bytes,
        OPUS,
    );
    let back: Value = serde_json::from_slice(&back_bytes).unwrap();
    assert_eq!(back["messages"][1]["tool_calls"][0]["id"], "call_1");
    assert_eq!(back["messages"][2]["role"], "tool");
    assert_eq!(back["messages"][2]["content"], "64F");
    assert_eq!(back["tools"][0]["function"]["name"], "get_weather");
}

#[test]
fn anthropic_stream_true_does_not_inject_stream_options_here() {
    // include_usage is spliced by the proxy onto the *translated OpenAI* body, not here.
    let body = br#"{"model":"gpt-4o-mini","max_tokens":8,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#;
    let v: Value = serde_json::from_slice(&request(
        Endpoint::Messages,
        Endpoint::ChatCompletions,
        body,
        OPUS,
    ))
    .unwrap();
    assert_eq!(v["stream"], true);
    assert!(v.get("stream_options").is_none(), "{v}");
    assert_eq!(v["messages"][0]["role"], "user");
    assert_eq!(v["max_tokens"], 8);
}

#[test]
fn response_json_maps_usage_and_text() {
    let anth = json!({
        "id": "msg_mock",
        "type": "message",
        "model": "claude-opus-4-8",
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 13, "output_tokens": 7}
    });
    let oai: Value = serde_json::from_slice(&response_json(
        Endpoint::Messages,
        Endpoint::ChatCompletions,
        &serde_json::to_vec(&anth).unwrap(),
    ))
    .unwrap();
    assert_eq!(oai["object"], "chat.completion");
    assert_eq!(oai["choices"][0]["message"]["content"], "hi");
    assert_eq!(oai["usage"]["prompt_tokens"], 13);
    assert_eq!(oai["usage"]["completion_tokens"], 7);

    let back: Value = serde_json::from_slice(&response_json(
        Endpoint::ChatCompletions,
        Endpoint::Messages,
        &serde_json::to_vec(&oai).unwrap(),
    ))
    .unwrap();
    assert_eq!(back["type"], "message");
    assert_eq!(back["content"][0]["text"], "hi");
    assert_eq!(back["usage"]["input_tokens"], 13);
}

/// claim: T6
#[test]
fn error_bodies_map_into_the_client_envelope() {
    let oai_err = br#"{"error":{"message":"nope","type":"invalid_request_error"}}"#;
    let anth: Value = serde_json::from_slice(&response_json(
        Endpoint::ChatCompletions,
        Endpoint::Messages,
        oai_err,
    ))
    .unwrap();
    assert_eq!(anth["type"], "error");
    assert_eq!(anth["error"]["message"], "nope");

    let back: Value = serde_json::from_slice(&response_json(
        Endpoint::Messages,
        Endpoint::ChatCompletions,
        &serde_json::to_vec(&anth).unwrap(),
    ))
    .unwrap();
    assert_eq!(back["error"]["type"], "invalid_request_error");
}

/// claim: S1
#[test]
fn anthropic_sse_becomes_chat_completion_chunk_without_waiting_for_stop() {
    let mut b = SseBridge::new(Endpoint::ChatCompletions, Endpoint::Messages);
    let start = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_x\",\"model\":\"claude-opus-4-8\",\"usage\":{\"input_tokens\":13,\"output_tokens\":1}}}\n\n",
    );
    let delta = concat!(
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
    );
    let out = String::from_utf8(b.feed(start.as_bytes(), false)).unwrap();
    assert!(out.contains("chat.completion.chunk"), "{out}");
    let out = String::from_utf8(b.feed(delta.as_bytes(), false)).unwrap();
    assert!(out.contains("chat.completion.chunk"), "{out}");
    assert!(out.contains("\"hi\""), "{out}");
    assert!(
        !out.contains("[DONE]"),
        "must not wait for message_stop: {out}"
    );
}

#[test]
fn sse_event_split_across_chunks_is_held() {
    let mut b = SseBridge::new(Endpoint::ChatCompletions, Endpoint::Messages);
    let first = b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\"";
    let out = b.feed(first, false);
    assert!(out.is_empty(), "incomplete event must be withheld");
    let rest = b",\"delta\":{\"type\":\"text_delta\",\"text\":\"z\"}}\n\n";
    let out = String::from_utf8(b.feed(rest, false)).unwrap();
    assert!(out.contains("\"z\""), "{out}");
}

/// claim: S1
#[test]
fn openai_sse_becomes_anthropic_events_and_does_not_wait_for_done() {
    let mut b = SseBridge::new(Endpoint::Messages, Endpoint::ChatCompletions);
    let chunk = "data: {\"id\":\"chatcmpl-x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n";
    let out = String::from_utf8(b.feed(chunk.as_bytes(), false)).unwrap();
    assert!(out.contains("event: message_start"), "{out}");
    assert!(out.contains("text_delta"), "{out}");
    assert!(out.contains("hi"), "{out}");
    assert!(
        !out.contains("message_stop"),
        "must not wait for [DONE]: {out}"
    );
}

#[test]
fn same_wire_is_a_byte_copy() {
    let body = br#"{"model":"x"}"#;
    assert_eq!(
        request(
            Endpoint::ChatCompletions,
            Endpoint::ChatCompletions,
            body,
            OPUS
        ),
        body
    );
    assert_eq!(
        response_json(Endpoint::Messages, Endpoint::Messages, body),
        body
    );
}

#[test]
fn openai_sse_flush_without_deltas_still_emits_message_start() {
    // A `[DONE]`-only stream ended as the upstream said; a bodiless one is covered by
    // `a_stream_that_ends_without_saying_how_is_an_error_for_every_client`.
    let mut b = SseBridge::new(Endpoint::Messages, Endpoint::ChatCompletions);
    let out = String::from_utf8(b.feed(b"data: [DONE]\n\n", true)).unwrap();
    assert!(out.contains("event: message_start"), "{out}");
    assert!(out.contains("event: message_stop"), "{out}");
}

/// claim: T1
#[test]
fn anthropic_tool_sse_becomes_openai_tool_calls() {
    let mut b = SseBridge::new(Endpoint::ChatCompletions, Endpoint::Messages);
    let src = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_x\",\"model\":\"claude-opus-4-8\",\"usage\":{\"input_tokens\":13,\"output_tokens\":1}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get_weather\",\"input\":{}}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\\\"SF\\\"}\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":7}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );
    let mut out = Vec::new();
    for byte in src.as_bytes() {
        out.extend(b.feed(&[*byte], false));
    }
    out.extend(b.feed(b"", true));
    let text = String::from_utf8(out).unwrap();
    assert!(text.contains("\"tool_calls\""), "{text}");
    assert!(text.contains("get_weather"), "{text}");
    assert!(text.contains("toolu_1"), "{text}");
    assert!(text.contains("city"), "{text}");
    assert!(text.contains("\"finish_reason\":\"tool_calls\""), "{text}");
    assert!(text.contains("\"prompt_tokens\":13"), "{text}");
    assert!(text.contains("[DONE]"), "{text}");
}

/// claim: T1
#[test]
fn openai_tool_sse_becomes_anthropic_tool_use() {
    let mut b = SseBridge::new(Endpoint::Messages, Endpoint::ChatCompletions);
    let src = concat!(
        "data: {\"id\":\"chatcmpl-x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]}}]}\n\n",
        "data: {\"id\":\"chatcmpl-x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\":\\\"SF\\\"}\"}}]}}]}\n\n",
        "data: {\"id\":\"chatcmpl-x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
        "data: {\"id\":\"chatcmpl-x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o\",\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":9}}\n\n",
        "data: [DONE]\n\n",
    );
    let out = String::from_utf8(b.feed(src.as_bytes(), true)).unwrap();
    assert!(out.contains("event: message_start"), "{out}");
    assert!(out.contains("\"type\":\"tool_use\""), "{out}");
    assert!(out.contains("get_weather"), "{out}");
    assert!(out.contains("call_1"), "{out}");
    assert!(out.contains("input_json_delta"), "{out}");
    assert!(out.contains("\"stop_reason\":\"tool_use\""), "{out}");
    assert!(out.contains("event: message_stop"), "{out}");
    assert!(!out.contains("chat.completion.chunk"), "{out}");
}

/// claim: T6
#[test]
fn sse_error_events_map_into_the_client_envelope() {
    let mut oai = SseBridge::new(Endpoint::ChatCompletions, Endpoint::Messages);
    let anth_err = concat!(
        "event: error\n",
        "data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"try again\"}}\n\n",
    );
    let out = String::from_utf8(oai.feed(anth_err.as_bytes(), true)).unwrap();
    assert!(out.contains("\"message\":\"try again\""), "{out}");
    assert!(out.contains("overloaded_error"), "{out}");
    assert!(!out.contains("event: error"), "{out}");

    let mut anth = SseBridge::new(Endpoint::Messages, Endpoint::ChatCompletions);
    let oai_err = "data: {\"error\":{\"message\":\"try again\",\"type\":\"server_error\"}}\n\n";
    let out = String::from_utf8(anth.feed(oai_err.as_bytes(), true)).unwrap();
    assert!(out.contains("event: error"), "{out}");
    assert!(out.contains("\"type\":\"error\""), "{out}");
    assert!(out.contains("try again"), "{out}");
    assert!(
        !out.contains("event: message_start"),
        "error stream must not invent a success close: {out}"
    );
}

/// claim: T1
#[test]
fn nonstream_tool_json_round_trips() {
    let anth = json!({
        "id": "msg_mock",
        "type": "message",
        "model": "claude-opus-4-8",
        "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "SF"}}],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 13, "output_tokens": 7}
    });
    let oai: Value = serde_json::from_slice(&response_json(
        Endpoint::Messages,
        Endpoint::ChatCompletions,
        &serde_json::to_vec(&anth).unwrap(),
    ))
    .unwrap();
    assert_eq!(oai["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(
        oai["choices"][0]["message"]["tool_calls"][0]["id"],
        "toolu_1"
    );
    assert_eq!(
        oai["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "get_weather"
    );
    let args = oai["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        .as_str()
        .unwrap();
    assert!(args.contains("SF"), "{args}");

    let back: Value = serde_json::from_slice(&response_json(
        Endpoint::ChatCompletions,
        Endpoint::Messages,
        &serde_json::to_vec(&oai).unwrap(),
    ))
    .unwrap();
    assert_eq!(back["stop_reason"], "tool_use");
    assert_eq!(back["content"][0]["type"], "tool_use");
    assert_eq!(back["content"][0]["input"]["city"], "SF");
}

/// A Messages history call's `input` reaches a Chat upstream with its numbers as written,
/// next to an integer-only call that takes the ordinary path.
#[test]
fn messages_history_tool_input_keeps_its_number_text() {
    let body = br#"{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"f","input":{"n":0.30000000000000001,"big":123456789012345678901234567890}},{"type":"tool_use","id":"toolu_2","name":"g","input":{"k":7}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"ok"},{"type":"tool_result","tool_use_id":"toolu_2","content":"ok"}]}]}"#;
    let out = request(
        Endpoint::Messages,
        Endpoint::ChatCompletions,
        body,
        "gpt-4o",
    );
    let v: Value = serde_json::from_slice(&out).unwrap();
    let calls = v["messages"][1]["tool_calls"].as_array().unwrap();
    assert_eq!(
        calls[0]["function"]["arguments"],
        r#"{"n":0.30000000000000001,"big":123456789012345678901234567890}"#
    );
    assert_eq!(calls[1]["function"]["arguments"], r#"{"k":7}"#);
}

/// The verbatim stand-in is written as its text only when that text is one JSON value: a client
/// object that happens to carry the key cannot splice arbitrary bytes into a body.
#[test]
fn a_verbatim_stand_in_is_written_only_as_valid_json() {
    let valid = json!({ "a": { RAW_JSON_KEY: " {\"n\":1.50} " } });
    assert_eq!(encode(&valid), br#"{"a":{"n":1.50}}"#);
    let spliced = json!({ "a": { RAW_JSON_KEY: "1, \"model\": \"other\"" } });
    let out: Value = serde_json::from_slice(&encode(&spliced)).unwrap();
    assert_eq!(out["a"][RAW_JSON_KEY], "1, \"model\": \"other\"");
    assert!(out.get("model").is_none());
}

/// claim: E3
/// defect: D145
#[test]
fn store_false_is_added_once_and_nothing_else_moves() {
    let mut omitted = br#" {"model":"grok-4.3", "input": "hi"}"#.to_vec();
    assert!(store_false(&mut omitted));
    assert_eq!(
        omitted,
        br#" {"store":false,"model":"grok-4.3", "input": "hi"}"#
    );
    let mut null = br#"{"store":null,"model":"m","input":"hi"}"#.to_vec();
    assert!(store_false(&mut null));
    assert_eq!(null, br#"{"store":false,"model":"m","input":"hi"}"#);
    let mut twice = br#"{"model":"m","store":false,"store":null}"#.to_vec();
    assert!(store_false(&mut twice));
    let v: Value = serde_json::from_slice(&twice).unwrap();
    assert_eq!(v["store"], false);
    assert_eq!(
        String::from_utf8(twice).unwrap().matches("store").count(),
        1
    );
    let mut already = br#"{"model":"m","store":false}"#.to_vec();
    assert!(!store_false(&mut already));
    let mut empty = b"{}".to_vec();
    assert!(store_false(&mut empty));
    assert_eq!(empty, br#"{"store":false}"#);
    let mut not_json = b"[1]".to_vec();
    assert!(!store_false(&mut not_json));
}

/// Only `developer` role values change; spacing, key order, a `developer` string elsewhere and
/// numbers as written stay.
/// claim: TRN-16
/// defect: D173
#[test]
fn developer_as_system_changes_only_the_roles() {
    let mut body = br#"{"temperature": 1.50, "messages": [ {"content":"x", "role" : "developer"}, {"role":"user","content":"developer"} ]}"#.to_vec();
    assert!(developer_as_system(&mut body));
    assert_eq!(
        String::from_utf8(body).unwrap(),
        r#"{"temperature": 1.50, "messages": [ {"content":"x", "role" : "system"}, {"role":"user","content":"developer"} ]}"#
    );
    let mut none = br#"{"messages":[{"role":"user","content":"developer"}]}"#.to_vec();
    assert!(!developer_as_system(&mut none));
    let mut not_json = b"developer {".to_vec();
    assert!(!developer_as_system(&mut not_json));
}

/// A forced tool turns thinking off under either rule (every `reasoning` / `reasoning_effort`
/// cut, `"reasoning":{"enabled":false}` first); unforced tools only under `OffWithTools` and
/// only when the client asked for no reasoning; no tools, or `Free`, leaves the body as sent.
/// claim: T1
/// defect: D171, D172
#[test]
fn fit_tools_to_thinking_follows_the_rule() {
    use providers::catalog::ToolThinking::{Free, OffWhenForced, OffWithTools};
    let forced = br#"{"model":"m","tools":[{"type":"function"}],"tool_choice":"required","reasoning_effort":"high","n":1.50}"#;
    let mut b = forced.to_vec();
    assert!(fit_tools_to_thinking(&mut b, OffWhenForced));
    assert_eq!(
        String::from_utf8(b).unwrap(),
        r#"{"reasoning":{"enabled":false},"model":"m","tools":[{"type":"function"}],"tool_choice":"required","n":1.50}"#
    );
    let named = br#"{"tools":[{}],"tool_choice":{"type":"function","function":{"name":"f"}}}"#;
    let mut b = named.to_vec();
    assert!(fit_tools_to_thinking(&mut b, OffWithTools));
    assert!(b.starts_with(br#"{"reasoning":{"enabled":false},"tools""#));

    let auto = br#"{"tools":[{}],"tool_choice":"auto"}"#;
    let mut b = auto.to_vec();
    assert!(!fit_tools_to_thinking(&mut b, OffWhenForced));
    assert!(fit_tools_to_thinking(&mut b, OffWithTools));
    let asked = br#"{"tools":[{}],"reasoning_effort":"low"}"#;
    let mut b = asked.to_vec();
    assert!(!fit_tools_to_thinking(&mut b, OffWithTools));
    let asked_null = br#"{"tools":[{}],"reasoning_effort":null}"#;
    let mut b = asked_null.to_vec();
    assert!(fit_tools_to_thinking(&mut b, OffWithTools));

    for (body, rule) in [
        (&br#"{"tool_choice":"required"}"#[..], OffWithTools),
        (
            &br#"{"tools":[],"tool_choice":"required"}"#[..],
            OffWithTools,
        ),
        (&forced[..], Free),
    ] {
        let mut b = body.to_vec();
        assert!(!fit_tools_to_thinking(&mut b, rule));
        assert_eq!(b, body);
    }
}

/// A candidate that accepts only `auto` gets a forced tool as an instruction, and `none` as no
/// tools at all; `auto` is sent as it came.
/// claim: T1
/// defect: D270
#[test]
fn an_auto_only_candidate_gets_none_as_no_tools_and_forced_as_an_instruction() {
    use providers::catalog::ToolThinking::AutoOnly;
    let run = |body: &str| {
        let mut b = body.as_bytes().to_vec();
        let changed = fit_tools_to_thinking(&mut b, AutoOnly);
        (changed, String::from_utf8(b).unwrap())
    };
    assert_eq!(
        run(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{}],"tool_choice":"none","n":1}"#
        ),
        (
            true,
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"n":1}"#.to_owned()
        )
    );
    let (changed, out) = run(
        r#"{"messages":[{"role":"user","content":"hi"}],"tools":[{}],"tool_choice":"required"}"#,
    );
    assert!(changed);
    let v: Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v["tool_choice"], "auto");
    assert_eq!(
        v["messages"][1],
        json!({"role": "system", "content": "Respond by calling one of the provided tools."})
    );
    let auto = r#"{"messages":[],"tools":[{}],"tool_choice":"auto"}"#;
    assert_eq!(run(auto), (false, auto.to_owned()));
}

/// A candidate that always thinks gets a forced tool as `auto` plus a closing system message
/// naming the tool (or any tool, for `required`); its reasoning stays as asked. Unforced, no
/// tools, or no `messages` array: as sent.
/// claim: T1
/// defect: D269
#[test]
fn a_forced_tool_goes_as_an_instruction_where_thinking_cannot_go_off() {
    use providers::catalog::ToolThinking::ForcedAsInstruction;
    let run = |body: &str| {
        let mut b = body.as_bytes().to_vec();
        let changed = fit_tools_to_thinking(&mut b, ForcedAsInstruction);
        (changed, String::from_utf8(b).unwrap())
    };
    assert_eq!(
        run(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{}],"tool_choice":{"type":"function","function":{"name":"get_weather"}},"reasoning_effort":"high"}"#
        ),
        (
            true,
            r#"{"tool_choice":"auto","model":"m","messages":[{"role":"user","content":"hi"},{"content":"Respond by calling the `get_weather` tool.","role":"system"}],"tools":[{}],"reasoning_effort":"high"}"#
                .to_owned()
        )
    );
    let (changed, out) = run(r#"{"messages":[ ],"tools":[{}],"tool_choice":"required"}"#);
    assert!(changed);
    let v: Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v["tool_choice"], "auto");
    assert_eq!(
        v["messages"],
        json!([{"role": "system", "content": "Respond by calling one of the provided tools."}])
    );
    for body in [
        r#"{"messages":[],"tools":[{}],"tool_choice":"auto"}"#,
        r#"{"messages":[],"tools":[{}]}"#,
        r#"{"messages":[],"tools":[],"tool_choice":"required"}"#,
        r#"{"messages":"x","tools":[{}],"tool_choice":"required"}"#,
    ] {
        assert_eq!(run(body), (false, body.to_owned()));
    }
}

/// claim: E1
/// defect: D147
#[test]
fn force_stream_asks_for_one_stream_and_moves_nothing_else() {
    const ON: &str = r#""stream":true,"stream_options":{"include_usage":true}"#;
    let mut omitted = br#" {"model":"m", "messages": []}"#.to_vec();
    assert!(force_stream(&mut omitted, true));
    assert_eq!(
        String::from_utf8(omitted).unwrap(),
        format!(r#" {{{ON},"model":"m", "messages": []}}"#)
    );
    let mut off =
        br#"{"stream":false,"model":"m","stream_options":{"include_usage":false}}"#.to_vec();
    assert!(force_stream(&mut off, true));
    let v: Value = serde_json::from_slice(&off).unwrap();
    assert_eq!(v["stream"], true);
    assert_eq!(v["stream_options"]["include_usage"], true);
    let text = String::from_utf8(off).unwrap();
    assert_eq!(text.matches("\"stream\"").count(), 1, "{text}");
    assert_eq!(text.matches("stream_options").count(), 1, "{text}");
    let mut messages = br#"{"model":"m","stream":null}"#.to_vec();
    assert!(force_stream(&mut messages, false));
    assert_eq!(messages, br#"{"stream":true,"model":"m"}"#);
    // The provider reads the last `stream`: a body that streams is the client's own stream.
    let mut streams = br#"{"stream":false,"model":"m","stream":true}"#.to_vec();
    assert!(!force_stream(&mut streams, true));
    assert_eq!(streams, br#"{"stream":false,"model":"m","stream":true}"#);
    let mut empty = b"{}".to_vec();
    assert!(force_stream(&mut empty, true));
    assert_eq!(String::from_utf8(empty).unwrap(), format!("{{{ON}}}"));
    let mut not_json = b"[1]".to_vec();
    assert!(!force_stream(&mut not_json, true));
}

/// An assembling bridge writes nothing while the stream runs and hands the client its whole
/// body at the end: a refusal and an `incomplete` finish (the output limit) map as a
/// non-streaming answer carries them.
/// claim: E1, E2
/// defect: D147
#[test]
fn an_assembling_bridge_writes_one_body_at_the_end() {
    let sse = concat!(
        "data: {\"id\":\"c1\",\"created\":5,\"model\":\"q\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Par\"}}]}\n\n",
        "data: {\"id\":\"c1\",\"created\":5,\"model\":\"q\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"tial\"},\"finish_reason\":\"length\"}]}\n\n",
        "data: {\"id\":\"c1\",\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2}}\n\n",
        "data: [DONE]\n\n",
    );
    for (client, check) in [
        (
            Endpoint::ChatCompletions,
            (|v: &Value| {
                assert_eq!(v["choices"][0]["message"]["content"], "Partial", "{v}");
                assert_eq!(v["choices"][0]["finish_reason"], "length", "{v}");
                assert_eq!(v["usage"]["completion_tokens"], 2, "{v}");
                assert_eq!(v["model"], "q", "{v}");
            }) as fn(&Value),
        ),
        (Endpoint::Messages, |v: &Value| {
            assert_eq!(v["content"][0]["text"], "Partial", "{v}");
            assert_eq!(v["stop_reason"], "max_tokens", "{v}");
            assert_eq!(v["usage"]["output_tokens"], 2, "{v}");
        }),
        (Endpoint::Responses, |v: &Value| {
            assert_eq!(v["status"], "incomplete", "{v}");
            assert_eq!(v["output"][0]["content"][0]["text"], "Partial", "{v}");
            assert_eq!(v["usage"]["input_tokens"], 3, "{v}");
        }),
    ] {
        let mut bridge = SseBridge::assembling(Endpoint::ChatCompletions);
        let (head, tail) = sse.as_bytes().split_at(70);
        assert!(bridge.feed(head, false).is_empty());
        assert!(bridge.feed(tail, true).is_empty());
        let v: Value = serde_json::from_slice(&bridge.assembled(client)).unwrap();
        check(&v);
    }
    // An error event mid-stream is the client's error, in its own envelope.
    let mut bridge = SseBridge::assembling(Endpoint::ChatCompletions);
    let err = "data: {\"error\":{\"message\":\"overloaded\",\"type\":\"server_error\"}}\n\n";
    assert!(bridge.feed(err.as_bytes(), true).is_empty());
    let v: Value = serde_json::from_slice(&bridge.assembled(Endpoint::Messages)).unwrap();
    assert_eq!(v["type"], "error", "{v}");
    assert_eq!(v["error"]["message"], "overloaded", "{v}");
}

#[test]
fn responses_session_field_names_previous_response_id_first() {
    let body = br#"{"model":"gpt-4o","previous_response_id":"resp_1","store":false}"#;
    assert_eq!(
        responses_session_field(body, false),
        Some("previous_response_id")
    );
    // An omitted store is session state only where a Responses arm can keep it.
    let omitted = br#"{"model":"gpt-4o","input":"hi"}"#;
    assert_eq!(responses_session_field(omitted, true), Some("store"));
    assert_eq!(responses_session_field(omitted, false), None);
    let null = br#"{"model":"gpt-4o","store":null}"#;
    assert_eq!(responses_session_field(null, false), None);
    let stored = br#"{"model":"gpt-4o","store":true}"#;
    assert_eq!(responses_session_field(stored, false), Some("store"));
    let one_shot = br#"{"model":"gpt-4o","store":false,"input":"hi"}"#;
    assert_eq!(responses_session_field(one_shot, true), None);
    let empty_prev = br#"{"model":"gpt-4o","previous_response_id":"","store":false}"#;
    assert_eq!(responses_session_field(empty_prev, true), None);
    assert_eq!(responses_session_field(b"not-json", false), Some("store"));
    // A reference-only input stands for an earlier turn, on every row.
    let reference =
        br#"{"model":"m","store":false,"input":[{"type":"item_reference","id":"msg_1"}]}"#;
    assert_eq!(
        responses_session_field(reference, false),
        Some("item_reference")
    );
    assert_eq!(
        responses_session_field(reference, true),
        Some("item_reference")
    );
    // A tool step's reference is not session state: a store: false one-shot stays one.
    let step = br#"{"model":"m","store":false,"input":[{"role":"user","content":"hi"},
        {"type":"item_reference","id":"msg_1"},
        {"type":"function_call","call_id":"c","name":"f","arguments":"{}"},
        {"type":"function_call_output","call_id":"c","output":"ok"}]}"#;
    assert_eq!(responses_session_field(step, false), None);
    assert_eq!(responses_session_field(step, true), None);
}

/// The `item_reference` rule over the Responses input item sequence: a reference inside a step (a run of
/// references, reasoning and tool calls) that holds a call is dropped; one in a step without a
/// call is an earlier turn.
#[test]
fn an_item_reference_is_a_turn_unless_its_step_made_a_call() {
    let r = || json!({"type": "item_reference", "id": "msg_1"});
    let reasoning = || json!({"type": "reasoning", "id": "rs_1", "summary": []});
    let call =
        |id: &str| json!({"type": "function_call", "call_id": id, "name": "f", "arguments": "{}"});
    let out = |id: &str| json!({"type": "function_call_output", "call_id": id, "output": "ok"});
    let user = |t: &str| json!({"role": "user", "content": t});
    let refused = |v: Value| {
        responses_session_field(&serde_json::to_vec(&v).unwrap(), false) == Some("item_reference")
    };
    let turn = |input: Vec<Value>| refused(json!({"store": false, "input": input}));
    // Turns: followed by a user message, at the end of input, alone, or before an output with
    // no call in its step.
    assert!(turn(vec![user("a"), r(), user("b")]));
    assert!(turn(vec![user("a"), r()]));
    assert!(turn(vec![r()]));
    assert!(turn(vec![r(), r()]));
    assert!(turn(vec![user("a"), r(), reasoning(), user("b")]));
    assert!(turn(vec![user("a"), r(), out("c")]));
    // A tool loop that then answered and was asked again: the answer is a turn.
    assert!(turn(vec![
        user("a"),
        r(),
        call("c"),
        out("c"),
        r(),
        user("b")
    ]));
    // Tool steps: preamble, several references, reasoning between, parallel calls, text after
    // the call before its output, a custom tool call.
    assert!(!turn(vec![user("a"), r(), call("c"), out("c")]));
    assert!(!turn(vec![user("a"), r(), r(), call("c"), out("c")]));
    assert!(!turn(vec![
        user("a"),
        r(),
        reasoning(),
        r(),
        call("c"),
        out("c")
    ]));
    assert!(!turn(vec![
        user("a"),
        reasoning(),
        r(),
        call("c"),
        out("c")
    ]));
    assert!(!turn(vec![
        user("a"),
        r(),
        call("c1"),
        call("c2"),
        out("c1"),
        out("c2")
    ]));
    assert!(!turn(vec![user("a"), call("c"), r(), out("c")]));
    assert!(!turn(vec![
        user("a"),
        r(),
        json!({"type": "custom_tool_call", "call_id": "c", "name": "f", "input": ""}),
    ]));
    // Two steps of one loop, each with its preamble.
    assert!(!turn(vec![
        user("a"),
        r(),
        call("c1"),
        out("c1"),
        r(),
        call("c2"),
        out("c2")
    ]));
    // No reference, or no input array: nothing to refuse.
    assert!(!turn(vec![user("a"), call("c"), out("c")]));
    assert!(!refused(json!({"store": false, "input": "hi"})));
    assert!(!refused(json!({"store": false})));
}

/// A Responses relay off a row with no Responses arm loses every `item_reference` and nothing
/// else, byte for byte; a body without one is untouched.
#[test]
fn strip_item_references_cuts_only_references() {
    let mut body = br#"{"model":"m","input":[{"role":"user","content":"item_reference?"},{"type":"item_reference","id":"msg_1"},{"type":"function_call","call_id":"c","name":"f","arguments":"{}"},{"type":"item_reference","id":"msg_2"}]}"#.to_vec();
    assert!(strip_item_references(&mut body));
    assert_eq!(
        String::from_utf8(body).unwrap(),
        r#"{"model":"m","input":[{"role":"user","content":"item_reference?"},{"type":"function_call","call_id":"c","name":"f","arguments":"{}"}]}"#
    );
    let plain = br#"{"model":"m","input":[{"role":"user","content":"hi"}]}"#;
    let mut body = plain.to_vec();
    assert!(!strip_item_references(&mut body));
    assert_eq!(body, plain);
    let mut only = br#"{"input":[{"type":"item_reference","id":"msg_1"}]}"#.to_vec();
    assert!(strip_item_references(&mut only));
    assert_eq!(only, br#"{"input":[]}"#);
}

/// A Bedrock-bound Messages body loses each tool's `eager_input_streaming` — first, middle or
/// last member, any value, an escaped key spelling — and nothing else, byte for byte, with the
/// spacing as sent. The string elsewhere (a prompt, a tool's description) is not a member of a
/// tool and stays; a body without it is untouched.
/// defect: D272
#[test]
fn strip_eager_input_streaming_cuts_only_the_tool_field() {
    let mut body = br#"{"model":"m","tools":[{"eager_input_streaming":true,"name":"a","input_schema":{"type":"object"}}, {"name":"b", "eager_input_streaming" : false, "description":"eager_input_streaming"}, {"name":"c","input_schema":{"properties":{"eager_input_streaming":{"type":"boolean"}}},"eager_input_streaming":true},{"name":"d"}],"messages":[{"role":"user","content":"\"eager_input_streaming\":true"}]}"#.to_vec();
    assert!(strip_eager_input_streaming(&mut body));
    assert_eq!(
        String::from_utf8(body).unwrap(),
        r#"{"model":"m","tools":[{"name":"a","input_schema":{"type":"object"}}, {"name":"b", "description":"eager_input_streaming"}, {"name":"c","input_schema":{"properties":{"eager_input_streaming":{"type":"boolean"}}}},{"name":"d"}],"messages":[{"role":"user","content":"\"eager_input_streaming\":true"}]}"#
    );
    // Mentioned, but on no tool: untouched.
    let mentioned = br#"{"model":"m","tools":[{"name":"a"}],"messages":[{"role":"user","content":"eager_input_streaming"}]}"#;
    let mut body = mentioned.to_vec();
    assert!(!strip_eager_input_streaming(&mut body));
    assert_eq!(body, mentioned);
    // A tool whose only member is the field is left an empty object, still valid JSON.
    let mut only = br#"{"tools":[{"eager_input_streaming":true}]}"#.to_vec();
    assert!(strip_eager_input_streaming(&mut only));
    assert_eq!(only, br#"{"tools":[{}]}"#);
}

#[test]
fn responses_to_chat_drops_session_fields_and_maps_input() {
    let body = serde_json::to_vec(&json!({
        "model": "gpt-4o",
        "input": [{"role":"user","content":[{"type":"input_text","text":"hi"}]}],
        "max_output_tokens": 16,
        "store": false,
        "include": ["reasoning.encrypted_content"],
        "truncation": "auto",
        "previous_response_id": "resp_x",
    }))
    .unwrap();
    let v: Value = serde_json::from_slice(&responses_to_chat(&body)).unwrap();
    assert!(v.get("store").is_none(), "{v}");
    assert!(v.get("previous_response_id").is_none(), "{v}");
    assert!(v.get("include").is_none(), "{v}");
    assert!(v.get("truncation").is_none(), "{v}");
    assert!(v.get("input").is_none(), "{v}");
    assert_eq!(v["max_tokens"], 16);
    assert_eq!(v["messages"][0]["role"], "user");
    assert_eq!(v["messages"][0]["content"], "hi");
    assert_eq!(v["model"], "gpt-4o");
}

#[test]
fn openai_cache_control_and_reasoning_reach_anthropic_fields() {
    let oai = json!({
        "model": "claude-opus-4-8",
        "reasoning_effort": "high",
        "max_tokens": 16,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
            ]
        }],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {}}
            },
            "cache_control": {"type": "ephemeral"}
        }]
    });
    let v: Value = serde_json::from_slice(&request(
        Endpoint::ChatCompletions,
        Endpoint::Messages,
        &serde_json::to_vec(&oai).unwrap(),
        OPUS,
    ))
    .unwrap();
    // Opus 4.8 rejects `budget_tokens`: effort goes to `output_config`.
    assert_eq!(v["thinking"]["type"], "adaptive");
    assert_eq!(v["output_config"]["effort"], "high");
    assert!(v["thinking"].get("budget_tokens").is_none(), "{v}");
    assert_eq!(
        v["messages"][0]["content"][0]["cache_control"]["type"],
        "ephemeral"
    );
    assert_eq!(v["tools"][0]["cache_control"]["type"], "ephemeral");
    assert!(
        v.get("reasoning_effort").is_none(),
        "Anthropic takes thinking, not reasoning_effort: {v}"
    );
    assert!(
        v.get("max_output_tokens").is_none() && v.get("input").is_none(),
        "Responses-only fields must stay dropped: {v}"
    );
}

#[test]
fn anthropic_thinking_becomes_openai_reasoning_effort() {
    let anth = json!({
        "model": "m",
        "max_tokens": 8,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "messages": [{"role": "user", "content": "hi"}]
    });
    let v: Value = serde_json::from_slice(&request(
        Endpoint::Messages,
        Endpoint::ChatCompletions,
        &serde_json::to_vec(&anth).unwrap(),
        OPUS,
    ))
    .unwrap();
    assert_eq!(v["reasoning_effort"], "low");
    assert!(v.get("thinking").is_none(), "{v}");
}

/// claim: T2
#[test]
fn thinking_and_redacted_thinking_round_trip_in_history() {
    let anth = json!({
        "model": "m",
        "max_tokens": 8,
        "messages": [{
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "plan", "signature": "sig"},
                {"type": "redacted_thinking", "data": "redacted"},
                {"type": "text", "text": "hi"}
            ]
        }]
    });
    let oai: Value = serde_json::from_slice(&request(
        Endpoint::Messages,
        Endpoint::ChatCompletions,
        &serde_json::to_vec(&anth).unwrap(),
        OPUS,
    ))
    .unwrap();
    assert_eq!(oai["messages"][0]["content"], "hi");
    assert_eq!(oai["messages"][0]["reasoning_content"], "plan");
    assert_eq!(oai["messages"][0]["thinking"][0]["type"], "thinking");
    assert_eq!(
        oai["messages"][0]["thinking"][1]["type"],
        "redacted_thinking"
    );

    let back: Value = serde_json::from_slice(&request(
        Endpoint::ChatCompletions,
        Endpoint::Messages,
        &serde_json::to_vec(&oai).unwrap(),
        OPUS,
    ))
    .unwrap();
    let content = &back["messages"][0]["content"];
    assert_eq!(content[0]["type"], "thinking");
    assert_eq!(content[0]["thinking"], "plan");
    assert_eq!(content[0]["signature"], "sig");
    assert_eq!(content[1]["type"], "redacted_thinking");
    assert_eq!(content[2]["type"], "text");
    assert_eq!(content[2]["text"], "hi");
}

/// claim: T3
#[test]
fn http_image_urls_become_anthropic_url_sources() {
    let oai = json!({
        "model": "m",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "see"},
                {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}
            ]
        }]
    });
    let v: Value = serde_json::from_slice(&request(
        Endpoint::ChatCompletions,
        Endpoint::Messages,
        &serde_json::to_vec(&oai).unwrap(),
        OPUS,
    ))
    .unwrap();
    let content = &v["messages"][0]["content"];
    assert_eq!(content[0]["text"], "see");
    assert_eq!(content[1]["type"], "image");
    assert_eq!(content[1]["source"]["type"], "url");
    assert_eq!(content[1]["source"]["url"], "https://example.com/x.png");
}

/// claim: T3
#[test]
fn anthropic_url_sources_become_openai_image_urls() {
    let msg = json!({
        "model": "m",
        "max_tokens": 16,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "url", "url": "https://example.com/x.png"}},
                {"type": "text", "text": "see"}
            ]
        }]
    });
    let v: Value = serde_json::from_slice(&request(
        Endpoint::Messages,
        Endpoint::ChatCompletions,
        &serde_json::to_vec(&msg).unwrap(),
        OPUS,
    ))
    .unwrap();
    let content = &v["messages"][0]["content"];
    assert_eq!(content[0]["type"], "image_url");
    assert_eq!(content[0]["image_url"]["url"], "https://example.com/x.png");
    assert_eq!(content[1]["text"], "see");
}

/// A `url` source that is not http(s) (`file:`, `ftp:`, garbage) is not forwarded as an
/// OpenAI `image_url` the upstream would try, and fail, to fetch.
#[test]
fn non_http_url_sources_are_not_forwarded() {
    let msg = json!({
        "model": "m",
        "max_tokens": 16,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "url", "url": "file:///etc/passwd"}},
                {"type": "text", "text": "see"}
            ]
        }]
    });
    let v: Value = serde_json::from_slice(&request(
        Endpoint::Messages,
        Endpoint::ChatCompletions,
        &serde_json::to_vec(&msg).unwrap(),
        OPUS,
    ))
    .unwrap();
    assert_eq!(v["messages"][0]["content"], "see");
}

/// The same between Responses and Chat Completions: a `file://` image or `file_url` document
/// is dropped, an https one is kept.
#[test]
fn local_file_urls_are_not_forwarded_from_responses() {
    let resp = json!({
        "model": "m",
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "see"},
                {"type": "input_image", "image_url": "file:///etc/passwd"},
                {"type": "input_file", "file_url": "file:///etc/shadow"},
                {"type": "input_file", "file_url": "https://example.com/q3.pdf"}
            ]
        }]
    });
    let chat: Value = serde_json::from_slice(&request(
        Endpoint::Responses,
        Endpoint::ChatCompletions,
        &serde_json::to_vec(&resp).unwrap(),
        OPUS,
    ))
    .unwrap();
    let text = chat.to_string();
    assert!(!text.contains("file://"), "{chat}");
    assert!(text.contains("https://example.com/q3.pdf"), "{chat}");
}

#[test]
fn http_image_urls_pass_between_responses_and_chat() {
    let resp = json!({
        "model": "m",
        "store": false,
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "see"},
                {"type": "input_image", "image_url": "https://example.com/x.png"}
            ]
        }]
    });
    let chat: Value = serde_json::from_slice(&request(
        Endpoint::Responses,
        Endpoint::ChatCompletions,
        &serde_json::to_vec(&resp).unwrap(),
        OPUS,
    ))
    .unwrap();
    let content = &chat["messages"][0]["content"];
    assert_eq!(content[1]["type"], "image_url");
    assert_eq!(content[1]["image_url"]["url"], "https://example.com/x.png");

    let back: Value = serde_json::from_slice(&request(
        Endpoint::ChatCompletions,
        Endpoint::Responses,
        &serde_json::to_vec(&chat).unwrap(),
        OPUS,
    ))
    .unwrap();
    let content = &back["input"][0]["content"];
    assert_eq!(content[1]["type"], "input_image");
    assert_eq!(content[1]["image_url"], "https://example.com/x.png");

    // And on through Messages: Responses → Messages is composed through Chat Completions.
    let msgs: Value = serde_json::from_slice(&request(
        Endpoint::Responses,
        Endpoint::Messages,
        &serde_json::to_vec(&resp).unwrap(),
        OPUS,
    ))
    .unwrap();
    let content = &msgs["messages"][0]["content"];
    assert_eq!(content[1]["source"]["type"], "url");
    assert_eq!(content[1]["source"]["url"], "https://example.com/x.png");
}

#[test]
fn anthropic_thinking_sse_reappears_on_openai_stream() {
    let mut b = SseBridge::new(Endpoint::ChatCompletions, Endpoint::Messages);
    let src = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_x\",\"model\":\"claude-opus-4-8\",\"usage\":{\"input_tokens\":13,\"cache_read_input_tokens\":4}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"plan\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":7,\"output_tokens_details\":{\"thinking_tokens\":3}}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );
    let out = String::from_utf8(b.feed(src.as_bytes(), true)).unwrap();
    assert!(out.contains("\"reasoning_content\":\"plan\""), "{out}");
    assert!(out.contains("\"hi\""), "{out}");
    assert!(out.contains("\"reasoning_tokens\":3"), "{out}");
    assert!(out.contains("\"cached_tokens\":4"), "{out}");
}

/// Unsigned reasoning never reaches a Messages client: echoed back, it would 400 the next
/// turn Anthropic serves. Signed reasoning is covered in `translate/response_tests.rs`.
#[test]
fn openai_reasoning_sse_becomes_anthropic_thinking() {
    let mut b = SseBridge::new(Endpoint::Messages, Endpoint::ChatCompletions);
    let src = concat!(
        "data: {\"id\":\"chatcmpl-x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"reasoning_content\":\"plan\"}}]}\n\n",
        "data: {\"id\":\"chatcmpl-x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
        "data: [DONE]\n\n",
    );
    let out = String::from_utf8(b.feed(src.as_bytes(), true)).unwrap();
    assert!(!out.contains("\"type\":\"thinking\""), "{out}");
    assert!(!out.contains("plan"), "{out}");
    assert!(out.contains("text_delta"), "{out}");
    assert!(out.contains("hi"), "{out}");
}

#[test]
fn client_usage_maps_cache_and_reasoning_without_changing_shape_of_tools() {
    let anth = json!({
        "id": "msg_mock",
        "type": "message",
        "model": "claude-opus-4-8",
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 13,
            "output_tokens": 7,
            "cache_read_input_tokens": 4,
            "output_tokens_details": {"thinking_tokens": 3}
        }
    });
    let oai: Value = serde_json::from_slice(&response_json(
        Endpoint::Messages,
        Endpoint::ChatCompletions,
        &serde_json::to_vec(&anth).unwrap(),
    ))
    .unwrap();
    assert_eq!(oai["usage"]["prompt_tokens_details"]["cached_tokens"], 4);
    assert_eq!(
        oai["usage"]["completion_tokens_details"]["reasoning_tokens"],
        3
    );
}

fn stock_responses(model: &str) -> Value {
    json!({
        "model": model,
        "input": [{
            "role": "user",
            "content": [{"type": "input_text", "text": "hi"}]
        }],
        "max_output_tokens": 16,
        "stream": true,
        "store": false,
        "reasoning": { "effort": "high" },
    })
}

#[test]
fn stock_responses_body_becomes_chat_completions_and_drops_store() {
    let v: Value = serde_json::from_slice(&request(
        Endpoint::Responses,
        Endpoint::ChatCompletions,
        &serde_json::to_vec(&stock_responses("gpt-4o-mini")).unwrap(),
        OPUS,
    ))
    .unwrap();
    assert_eq!(v["messages"][0]["role"], "user");
    assert_eq!(v["messages"][0]["content"], "hi");
    assert_eq!(v["max_tokens"], 16);
    assert_eq!(v["stream"], true);
    assert_eq!(v["reasoning_effort"], "high");
    assert!(v.get("store").is_none(), "{v}");
    assert!(v.get("input").is_none(), "{v}");
    assert!(v.get("max_output_tokens").is_none(), "{v}");
}

#[test]
fn responses_body_with_a_claude_id_becomes_messages() {
    let v: Value = serde_json::from_slice(&request(
        Endpoint::Responses,
        Endpoint::Messages,
        &serde_json::to_vec(&stock_responses("claude-opus-4-8")).unwrap(),
        OPUS,
    ))
    .unwrap();
    assert_eq!(v["messages"][0]["role"], "user");
    assert_eq!(v["messages"][0]["content"], "hi");
    assert_eq!(v["max_tokens"], 16);
    assert_eq!(v["thinking"]["type"], "adaptive");
    assert!(v.get("store").is_none(), "{v}");
    assert!(v.get("input").is_none(), "{v}");
}

#[test]
fn openai_sse_becomes_responses_events_without_waiting_for_done() {
    let mut b = SseBridge::new(Endpoint::Responses, Endpoint::ChatCompletions);
    let chunk = "data: {\"id\":\"chatcmpl-x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n";
    let out = String::from_utf8(b.feed(chunk.as_bytes(), false)).unwrap();
    assert!(out.contains("response.output_text.delta"), "{out}");
    assert!(out.contains("\"hi\""), "{out}");
    assert!(
        !out.contains("response.completed"),
        "must not wait for [DONE]: {out}"
    );
    let rest = "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":9}}\n\ndata: [DONE]\n\n";
    let out = String::from_utf8(b.feed(rest.as_bytes(), true)).unwrap();
    assert!(out.contains("response.completed"), "{out}");
    assert!(out.contains("\"input_tokens\":5"), "{out}");
    assert!(out.contains("\"output_tokens\":9"), "{out}");
}

#[test]
fn anthropic_sse_becomes_responses_via_chat() {
    let mut b = SseBridge::new(Endpoint::Responses, Endpoint::Messages);
    let src = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_x\",\"model\":\"claude-opus-4-8\",\"usage\":{\"input_tokens\":13,\"output_tokens\":1}}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":7}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );
    let out = String::from_utf8(b.feed(src.as_bytes(), true)).unwrap();
    assert!(out.contains("response.output_text.delta"), "{out}");
    assert!(out.contains("\"hi\""), "{out}");
    assert!(out.contains("response.completed"), "{out}");
    assert!(out.contains("\"input_tokens\":13"), "{out}");
}

#[test]
fn responses_json_round_trips_text_and_usage() {
    let chat = json!({
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "hi" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18 }
    });
    let resp: Value = serde_json::from_slice(&response_json(
        Endpoint::ChatCompletions,
        Endpoint::Responses,
        &serde_json::to_vec(&chat).unwrap(),
    ))
    .unwrap();
    assert_eq!(resp["object"], "response");
    assert_eq!(resp["output"][0]["content"][0]["text"], "hi");
    assert_eq!(resp["usage"]["input_tokens"], 11);
    assert_eq!(resp["usage"]["output_tokens"], 7);
    assert!(resp.get("choices").is_none(), "{resp}");
}

// ---- Model-aware reasoning and sampling ----

#[test]
fn claude_generation_parses_every_candidate_spelling() {
    use ClaudeGen::*;
    // (id, reasoning, forced tool_choice accepted)
    for (id, reasoning, forced) in [
        ("claude-opus-4-8", Adaptive, true),
        ("anthropic/claude-opus-4.8", Adaptive, true),
        ("global.anthropic.claude-opus-4-7-v1:0", Adaptive, true),
        ("claude-opus-5", Adaptive, true),
        ("claude-opus-5-5", Adaptive, false),
        ("anthropic/claude-opus-5.5", Adaptive, false),
        ("claude-sonnet-5", Adaptive, true),
        ("claude-sonnet-5-5", Adaptive, false),
        ("claude-fable-5", Adaptive, true),
        ("claude-fable-5-1", Adaptive, false),
        ("claude-mythos-5-1", Adaptive, false),
        ("claude-opus-4-6", Adaptive46, true),
        ("claude-sonnet-4-6", Adaptive46, true),
        ("claude-haiku-5-5", Adaptive, true),
        ("us.anthropic.claude-haiku-5-5", Adaptive, true),
        ("claude-haiku-4-5", Budget, true),
        ("anthropic.claude-haiku-4-5-20251001-v1:0", Budget, true),
        ("claude-sonnet-4-20250514", Budget, true),
        ("claude-opus-4-1", Budget, true),
        ("claude-3-haiku", Budget, true),
        ("claude-3-5-sonnet-latest", Budget, true),
        ("claude-nova", Adaptive, false),
        ("kimi-k3", Budget, true),
        ("", Budget, true),
    ] {
        let m = ClaudeModel::of(id);
        assert_eq!(
            (m.reasoning, m.forced_tool_choice),
            (reasoning, forced),
            "{id}"
        );
    }
}

fn chat(extra: Value) -> Value {
    let mut req = json!({
        "model": "m",
        "max_tokens": 16000,
        "messages": [{"role": "user", "content": "hi"}]
    });
    if let (Some(r), Some(e)) = (req.as_object_mut(), extra.as_object()) {
        for (k, v) in e {
            r.insert(k.clone(), v.clone());
        }
    }
    req
}

fn to_claude(body: &Value, model: &str) -> Value {
    serde_json::from_slice(&request(
        Endpoint::ChatCompletions,
        Endpoint::Messages,
        &serde_json::to_vec(body).unwrap(),
        model,
    ))
    .unwrap()
}

/// Current Claude models 400 on `budget_tokens` and on `temperature` / `top_p`.
/// claim: T5
#[test]
fn current_claude_gets_adaptive_thinking_and_no_sampling() {
    let v = to_claude(
        &chat(json!({"reasoning_effort": "xhigh", "temperature": 0.2, "top_p": 0.9})),
        "claude-opus-4-8",
    );
    assert_eq!(v["thinking"], json!({"type": "adaptive"}));
    assert_eq!(v["output_config"]["effort"], "xhigh");
    assert!(v.get("temperature").is_none(), "{v}");
    assert!(v.get("top_p").is_none(), "{v}");
}

/// Several current models reject `thinking: disabled`; "no reasoning" is the lowest effort.
#[test]
fn reasoning_none_on_current_claude_omits_thinking() {
    let v = to_claude(
        &chat(json!({"reasoning_effort": "none"})),
        "claude-opus-5-5",
    );
    // Opus 5.5 always thinks; the explicit `adaptive` only carries `block_binding`.
    assert_eq!(v["thinking"]["type"], "adaptive", "{v}");
    assert_eq!(v["output_config"]["effort"], "low");
}

#[test]
fn claude_4_6_has_no_xhigh() {
    let v = to_claude(
        &chat(json!({"reasoning_effort": "xhigh"})),
        "claude-sonnet-4-6",
    );
    assert_eq!(v["output_config"]["effort"], "max");
}

/// claim: T5
#[test]
fn older_claude_gets_a_budget_below_max_tokens() {
    let v = to_claude(
        &chat(json!({"reasoning_effort": "high"})),
        "claude-haiku-4-5",
    );
    assert_eq!(v["thinking"]["type"], "enabled");
    assert_eq!(
        v["thinking"]["budget_tokens"], 8000,
        "half of max_tokens 16000: {v}"
    );
    assert!(
        v.get("output_config").is_none(),
        "Haiku 4.5 rejects effort: {v}"
    );

    let mut small = chat(json!({"reasoning_effort": "high"}));
    small["max_tokens"] = json!(1000);
    let v = to_claude(&small, "claude-haiku-4-5");
    assert!(
        v.get("thinking").is_none(),
        "no room to think under 1024: {v}"
    );
}

/// Sampling is fine on older models, except alongside thinking.
/// claim: T5
#[test]
fn older_claude_keeps_sampling_unless_thinking() {
    let v = to_claude(&chat(json!({"temperature": 0.2})), "claude-haiku-4-5");
    assert_eq!(v["temperature"], 0.2);
    let v = to_claude(
        &chat(json!({"temperature": 0.2, "reasoning_effort": "low"})),
        "claude-haiku-4-5",
    );
    assert!(v.get("temperature").is_none(), "{v}");
}

// ---- Fields that must not vanish ----

fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": false
    })
}

/// claim: T4
#[test]
fn json_schema_response_format_becomes_output_config_format() {
    let v = to_claude(
        &chat(json!({"response_format": {
            "type": "json_schema",
            "json_schema": {"name": "a", "strict": true, "schema": schema()}
        }})),
        OPUS,
    );
    assert_eq!(
        v["output_config"]["format"],
        json!({"type": "json_schema", "schema": schema()})
    );
    assert!(v.get("response_format").is_none(), "{v}");
}

#[test]
fn format_and_effort_share_one_output_config() {
    let v = to_claude(
        &chat(json!({
            "reasoning_effort": "low",
            "response_format": {"type": "json_schema", "json_schema": {"name": "a", "schema": schema()}}
        })),
        OPUS,
    );
    assert_eq!(v["output_config"]["effort"], "low");
    assert_eq!(v["output_config"]["format"]["type"], "json_schema");
}

/// claim: T4
#[test]
fn output_config_format_becomes_response_format() {
    let anth = json!({
        "model": "claude-opus-4-8",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
        "output_config": {"format": {"type": "json_schema", "schema": schema()}, "effort": "medium"}
    });
    let v: Value = serde_json::from_slice(&request(
        Endpoint::Messages,
        Endpoint::ChatCompletions,
        &serde_json::to_vec(&anth).unwrap(),
        "gpt-5",
    ))
    .unwrap();
    assert_eq!(v["response_format"]["type"], "json_schema");
    assert_eq!(v["response_format"]["json_schema"]["schema"], schema());
    assert_eq!(v["reasoning_effort"], "medium");
}

#[test]
fn parallel_tool_calls_false_disables_parallel_tool_use_both_ways() {
    let v = to_claude(
        &chat(json!({
            "parallel_tool_calls": false,
            "tools": [{"type": "function", "function": {"name": "f"}}]
        })),
        OPUS,
    );
    assert_eq!(
        v["tool_choice"],
        json!({"type": "auto", "disable_parallel_tool_use": true})
    );
    let back: Value = serde_json::from_slice(&request(
        Endpoint::Messages,
        Endpoint::ChatCompletions,
        &serde_json::to_vec(&v).unwrap(),
        "gpt-5",
    ))
    .unwrap();
    assert_eq!(back["parallel_tool_calls"], false);
}

#[test]
fn user_becomes_metadata_user_id_both_ways() {
    let v = to_claude(&chat(json!({"user": "u-123"})), OPUS);
    assert_eq!(v["metadata"], json!({"user_id": "u-123"}));
    let back: Value = serde_json::from_slice(&request(
        Endpoint::Messages,
        Endpoint::ChatCompletions,
        &serde_json::to_vec(&v).unwrap(),
        "gpt-5",
    ))
    .unwrap();
    assert_eq!(back["user"], "u-123");
}

/// claim: T3
#[test]
fn an_inline_pdf_becomes_a_document_and_back() {
    let v = to_claude(
        &chat(json!({"messages": [{"role": "user", "content": [
            {"type": "file", "file": {"filename": "q3.pdf", "file_data": "data:application/pdf;base64,JVBERi0="}},
            {"type": "text", "text": "summarize"}
        ]}]})),
        OPUS,
    );
    let doc = &v["messages"][0]["content"][0];
    assert_eq!(doc["type"], "document");
    assert_eq!(
        doc["source"],
        json!({"type": "base64", "media_type": "application/pdf", "data": "JVBERi0="})
    );
    assert_eq!(doc["title"], "q3.pdf");
    let back: Value = serde_json::from_slice(&request(
        Endpoint::Messages,
        Endpoint::ChatCompletions,
        &serde_json::to_vec(&v).unwrap(),
        "gpt-5",
    ))
    .unwrap();
    assert_eq!(
        back["messages"][0]["content"][0],
        json!({"type": "file", "file": {"filename": "q3.pdf", "file_data": "data:application/pdf;base64,JVBERi0="}})
    );
}

/// No Messages equivalent: forwarded so the provider names the problem. Dropping them would
/// answer a question about input the model never saw.
#[test]
fn responses_input_audio_is_forwarded_to_chat_completions() {
    let audio = json!({"type": "input_audio", "input_audio": {"data": "UklG", "format": "wav"}});
    let resp = json!({"model": "m", "store": false, "input": [{"role": "user", "content": [
        {"type": "input_text", "text": "what is this"}, audio.clone()]}]});
    let chat: Value = serde_json::from_slice(&request(
        Endpoint::Responses,
        Endpoint::ChatCompletions,
        &serde_json::to_vec(&resp).unwrap(),
        OPUS,
    ))
    .unwrap();
    assert_eq!(chat["messages"][0]["content"][1], audio, "{chat}");
}

#[test]
fn unmappable_input_is_forwarded_for_the_provider_to_reject() {
    let audio = json!({"type": "input_audio", "input_audio": {"data": "UklG", "format": "wav"}});
    let by_id = json!({"type": "file", "file": {"file_id": "file-abc"}});
    let v = to_claude(
        &chat(
            json!({"n": 2, "logprobs": true, "messages": [{"role": "user", "content": [
                audio.clone(), by_id.clone(), {"type": "text", "text": "what is this"}
            ]}]}),
        ),
        OPUS,
    );
    assert_eq!(v["n"], 2);
    assert_eq!(v["logprobs"], true);
    assert_eq!(v["messages"][0]["content"][0], audio);
    assert_eq!(v["messages"][0]["content"][1], by_id);
}

#[test]
fn harmless_defaults_are_not_forwarded() {
    let v = to_claude(&chat(json!({"n": 1, "logprobs": false, "seed": 7})), OPUS);
    for key in ["n", "logprobs", "seed"] {
        assert!(v.get(key).is_none(), "{key}: {v}");
    }
}

#[test]
fn responses_text_format_and_input_file_reach_messages() {
    let body = json!({
        "model": "claude-opus-4-8",
        "store": false,
        "text": {"format": {"type": "json_schema", "name": "a", "strict": true, "schema": schema()}},
        "input": [{"role": "user", "content": [
            {"type": "input_file", "filename": "q3.pdf", "file_data": "data:application/pdf;base64,JVBERi0="},
            {"type": "input_text", "text": "summarize"}
        ]}]
    });
    let v: Value = serde_json::from_slice(&request(
        Endpoint::Responses,
        Endpoint::Messages,
        &serde_json::to_vec(&body).unwrap(),
        OPUS,
    ))
    .unwrap();
    assert_eq!(v["output_config"]["format"]["schema"], schema());
    assert_eq!(v["messages"][0]["content"][0]["type"], "document");
}

/// claim: T4
#[test]
fn chat_response_format_becomes_responses_text_format() {
    let v: Value = serde_json::from_slice(&request(
        Endpoint::ChatCompletions,
        Endpoint::Responses,
        &serde_json::to_vec(&chat(json!({"response_format": {
            "type": "json_schema",
            "json_schema": {"name": "a", "strict": true, "schema": schema()}
        }})))
        .unwrap(),
        "gpt-5",
    ))
    .unwrap();
    assert_eq!(
        v["text"]["format"],
        json!({"type": "json_schema", "name": "a", "strict": true, "schema": schema()})
    );
}

// ---- Forced tool use on models that reject it ----

fn tools_req(choice: Value) -> Value {
    chat(json!({
        "tool_choice": choice,
        "tools": [
            {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}}
        ]
    }))
}

#[test]
fn required_is_forced_where_the_model_allows_it() {
    let v = to_claude(&tools_req(json!("required")), "claude-opus-4-8");
    assert_eq!(v["tool_choice"], json!({"type": "any"}));
    assert!(
        v["messages"]
            .as_array()
            .unwrap()
            .iter()
            .all(|m| m["role"] != "system"),
        "{v}"
    );
}

/// Opus 5.5 / Sonnet 5.5 / Fable 5.1 400 on `any` and `tool`: `auto` plus a closing
/// instruction, which is Anthropic's documented migration for these models.
#[test]
fn required_becomes_auto_plus_an_instruction_where_forcing_is_rejected() {
    for model in ["claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1"] {
        let v = to_claude(&tools_req(json!("required")), model);
        assert_eq!(v["tool_choice"], json!({"type": "auto"}), "{model}");
        let last = v["messages"].as_array().unwrap().last().unwrap().clone();
        assert_eq!(last["role"], "system", "{model}: {v}");
        assert!(last["content"].as_str().unwrap().contains("provided tools"));
    }
}

#[test]
fn a_named_tool_choice_names_the_tool_in_the_instruction() {
    let v = to_claude(
        &tools_req(json!({"type": "function", "function": {"name": "get_time"}})),
        "claude-opus-5-5",
    );
    assert_eq!(v["tool_choice"], json!({"type": "auto"}));
    let last = v["messages"].as_array().unwrap().last().unwrap().clone();
    assert_eq!(last["role"], "system");
    assert!(
        last["content"].as_str().unwrap().contains("`get_time`"),
        "{last}"
    );
}

#[test]
fn unforced_choice_keeps_disable_parallel_tool_use() {
    let mut req = tools_req(json!("required"));
    req["parallel_tool_calls"] = json!(false);
    let v = to_claude(&req, "claude-opus-5-5");
    assert_eq!(
        v["tool_choice"],
        json!({"type": "auto", "disable_parallel_tool_use": true})
    );
}

/// The instruction goes after the cache marker, so it never invalidates the cached prefix.
#[test]
fn the_instruction_follows_the_conversation_cache_marker() {
    let mut req = tools_req(json!("required"));
    req["messages"] = json!([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "weather?"}
    ]);
    let v = to_claude(&req, "claude-opus-5-5");
    let msgs = v["messages"].as_array().unwrap();
    assert_eq!(msgs.last().unwrap()["role"], "system");
    assert!(msgs.last().unwrap().get("cache_control").is_none());
    let user = &msgs[msgs.len() - 2];
    assert_eq!(
        user["content"][0]["cache_control"]["type"], "ephemeral",
        "{v}"
    );
}

#[test]
fn auto_and_none_are_untouched_everywhere() {
    for choice in ["auto", "none"] {
        let v = to_claude(&tools_req(json!(choice)), "claude-opus-5-5");
        assert_eq!(v["tool_choice"]["type"], choice);
        assert!(
            v["messages"]
                .as_array()
                .unwrap()
                .iter()
                .all(|m| m["role"] != "system")
        );
    }
}
