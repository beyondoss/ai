//! `ModelScanner::feed_keeping`: the request's price knobs and the response's id and host.

use super::*;

fn knobs(body: &[u8], chunk: usize) -> (Option<String>, Kept) {
    let mut s = ModelScanner::new();
    let mut k = Kept::request_knobs();
    for c in body.chunks(chunk.max(1)) {
        s.feed_keeping(c, &mut k);
    }
    (s.take_model(), k)
}

/// Every knob, string or raw JSON, before or after `model`, at every chunking; nested look-alikes
/// (a message's `speed`, a tool's `container`) are not root knobs.
#[test]
fn request_knobs_are_read_wherever_the_client_put_them() {
    let body = br#"{"service_tier":"priority","messages":[{"role":"user","content":"\"speed\":\"fast\"","speed":"slow","container":"nested"}],
        "model":"claude-opus-4-8","speed":"fast","inference_geo":"us",
        "provider":{"order":["anthropic","amazon-bedrock"],"allow_fallbacks":false,"max_price":{"prompt":1}},
        "plugins":[{"id":"web","max_results":3}],"container":"container_011x"}"#;
    for chunk in [1, 2, 7, 64, body.len()] {
        let (model, k) = knobs(body, chunk);
        assert_eq!(model.as_deref(), Some("claude-opus-4-8"), "chunk {chunk}");
        assert_eq!(k.get(0), Some("priority"), "chunk {chunk}");
        assert_eq!(k.get(1), Some("fast"), "chunk {chunk}");
        assert_eq!(k.get(2), Some("us"), "chunk {chunk}");
        assert_eq!(
            k.get(3),
            Some(
                r#"{"order":["anthropic","amazon-bedrock"],"allow_fallbacks":false,"max_price":{"prompt":1}}"#
            ),
            "chunk {chunk}"
        );
        assert_eq!(
            k.get(4),
            Some(r#"[{"id":"web","max_results":3}]"#),
            "chunk {chunk}"
        );
        assert_eq!(k.get(5), Some("container_011x"), "chunk {chunk}");
    }
}

/// A body with no knobs keeps nothing, and the model is still found.
#[test]
fn a_body_without_knobs_keeps_nothing() {
    let (model, k) = knobs(br#"{"model":"gpt-5","messages":[]}"#, 3);
    assert_eq!(model.as_deref(), Some("gpt-5"));
    assert!((0..6).all(|i| k.get(i).is_none()));
}

/// A raw value past `RAW_CAPTURE` is cut there, at a character boundary.
#[test]
fn a_raw_knob_is_capped() {
    let long = format!(
        r#"{{"model":"m","provider":{{"order":["{}"]}}}}"#,
        "é".repeat(600)
    );
    let (_, k) = knobs(long.as_bytes(), 5);
    let v = k.get(3).unwrap();
    assert!(
        v.len() <= RAW_CAPTURE && v.len() >= RAW_CAPTURE - 1,
        "{}",
        v.len()
    );
    assert!(v.starts_with(r#"{"order":[""#));
}

/// The response scan keeps the id and the serving host that precede `model`, on each wire's
/// shape, and still stops at the model.
#[test]
fn a_response_keeps_its_id_and_host_before_the_model() {
    let cases: [(&[u8], Option<&str>, Option<&str>); 4] = [
        (
            br#"data: {"id":"gen-1","provider":"Amazon Bedrock","model":"anthropic/claude-opus-4.8","choices":[]}"#,
            Some("gen-1"),
            Some("Amazon Bedrock"),
        ),
        (
            b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"model\":\"claude-opus-4-8\"}}",
            Some("msg_1"),
            None,
        ),
        (
            br#"data: {"type":"response.created","sequence_number":0,"response":{"id":"resp_1","object":"response","model":"gpt-5"}}"#,
            Some("resp_1"),
            None,
        ),
        (
            br#"{"id": "chatcmpl-1", "object": "chat.completion", "model": "gpt-4o", "choices": [{"message": {"tool_calls": [{"id": "call_1"}]}}]}"#,
            Some("chatcmpl-1"),
            None,
        ),
    ];
    for (body, id, host) in cases {
        for chunk in [1, 3, body.len()] {
            let mut s = ModelScanner::for_response();
            let mut k = Kept::response();
            for c in body.chunks(chunk) {
                s.feed_keeping(c, &mut k);
            }
            assert!(s.found(), "{}", String::from_utf8_lossy(body));
            assert_eq!(k.get(0), id, "{}", String::from_utf8_lossy(body));
            assert_eq!(k.get(1), host, "{}", String::from_utf8_lossy(body));
        }
    }
}
