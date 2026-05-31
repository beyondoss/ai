//! Token-usage extraction — the "passive tap" the gateway emits as billing *facts*.
//!
//! We never compute price here (pricing is a closed downstream consumer); we only extract raw
//! token counts. Two shapes per provider: the non-streaming JSON body, and the terminal event of
//! an SSE stream. For streaming we scan the relayed bytes for the usage event but never block the
//! relay on it (see `proxy`).

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
}

fn u64_at(v: &serde_json::Value, ptr: &str) -> u64 {
    v.pointer(ptr).and_then(|x| x.as_u64()).unwrap_or(0)
}

/// OpenAI non-streaming: `usage.{prompt_tokens, completion_tokens}` (+ cached details).
pub fn openai_body(body: &[u8]) -> Option<Usage> {
    let v: serde_json::Value = serde_json::from_slice(body).ok()?;
    let u = v.get("usage")?;
    Some(Usage {
        input_tokens: u64_at(u, "/prompt_tokens"),
        output_tokens: u64_at(u, "/completion_tokens"),
        cache_read_tokens: u64_at(u, "/prompt_tokens_details/cached_tokens"),
        cache_write_tokens: 0,
    })
}

/// Anthropic non-streaming: `usage.{input_tokens, output_tokens, cache_*}`.
pub fn anthropic_body(body: &[u8]) -> Option<Usage> {
    let v: serde_json::Value = serde_json::from_slice(body).ok()?;
    let u = v.get("usage")?;
    Some(Usage {
        input_tokens: u64_at(u, "/input_tokens"),
        output_tokens: u64_at(u, "/output_tokens"),
        cache_read_tokens: u64_at(u, "/cache_read_input_tokens"),
        cache_write_tokens: u64_at(u, "/cache_creation_input_tokens"),
    })
}

/// Iterate the JSON objects carried on `data:` lines of an SSE byte stream. `[DONE]` and
/// non-JSON payloads are skipped. Used by both stream parsers below.
fn sse_data_objects(sse: &[u8]) -> impl Iterator<Item = serde_json::Value> + '_ {
    sse.split(|&b| b == b'\n').filter_map(|line| {
        let line = line.strip_prefix(b"data:")?;
        let line = line.strip_prefix(b" ").unwrap_or(line);
        if line == b"[DONE]" {
            return None;
        }
        serde_json::from_slice::<serde_json::Value>(line).ok()
    })
}

/// OpenAI streaming (requires `stream_options.include_usage`): the penultimate chunk carries a
/// top-level `usage` object. Last one with usage wins.
pub fn openai_stream(sse: &[u8]) -> Option<Usage> {
    let mut found = None;
    for v in sse_data_objects(sse) {
        if let Some(u) = v.get("usage").filter(|u| !u.is_null()) {
            found = Some(Usage {
                input_tokens: u64_at(u, "/prompt_tokens"),
                output_tokens: u64_at(u, "/completion_tokens"),
                cache_read_tokens: u64_at(u, "/prompt_tokens_details/cached_tokens"),
                cache_write_tokens: 0,
            });
        }
    }
    found
}

/// Anthropic streaming: input + cache tokens arrive in `message_start.message.usage`; output
/// accumulates in `message_delta.usage.output_tokens` (last delta is the cumulative total).
pub fn anthropic_stream(sse: &[u8]) -> Option<Usage> {
    let mut usage = Usage::default();
    let mut saw_any = false;
    for v in sse_data_objects(sse) {
        if let Some(u) = v.pointer("/message/usage") {
            usage.input_tokens = u64_at(u, "/input_tokens");
            usage.cache_read_tokens = u64_at(u, "/cache_read_input_tokens");
            usage.cache_write_tokens = u64_at(u, "/cache_creation_input_tokens");
            saw_any = true;
        }
        if let Some(u) = v.get("usage") {
            // message_delta carries the running output token count.
            let out = u64_at(u, "/output_tokens");
            if out > 0 {
                usage.output_tokens = out;
            }
            saw_any = true;
        }
    }
    saw_any.then_some(usage)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_nonstreaming() {
        let body = br#"{"usage":{"prompt_tokens":12,"completion_tokens":34,
            "prompt_tokens_details":{"cached_tokens":4}}}"#;
        assert_eq!(
            openai_body(body).unwrap(),
            Usage {
                input_tokens: 12,
                output_tokens: 34,
                cache_read_tokens: 4,
                cache_write_tokens: 0
            }
        );
    }

    #[test]
    fn anthropic_nonstreaming() {
        let body = br#"{"usage":{"input_tokens":100,"output_tokens":50,
            "cache_read_input_tokens":10,"cache_creation_input_tokens":7}}"#;
        assert_eq!(
            anthropic_body(body).unwrap(),
            Usage {
                input_tokens: 100,
                output_tokens: 50,
                cache_read_tokens: 10,
                cache_write_tokens: 7
            }
        );
    }

    #[test]
    fn openai_streaming_terminal_usage() {
        let sse = b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n\
                    data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":9}}\n\n\
                    data: [DONE]\n\n";
        assert_eq!(
            openai_stream(sse).unwrap(),
            Usage {
                input_tokens: 5,
                output_tokens: 9,
                cache_read_tokens: 0,
                cache_write_tokens: 0
            }
        );
    }

    #[test]
    fn anthropic_streaming_accumulates() {
        let sse = b"event: message_start\n\
                    data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":20,\"output_tokens\":0}}}\n\n\
                    event: message_delta\n\
                    data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":15}}\n\n";
        assert_eq!(
            anthropic_stream(sse).unwrap(),
            Usage {
                input_tokens: 20,
                output_tokens: 15,
                cache_read_tokens: 0,
                cache_write_tokens: 0
            }
        );
    }

    #[test]
    fn anthropic_streaming_includes_cache_tokens() {
        // Cache tokens ride in `message_start.message.usage` alongside input_tokens. The earlier
        // accumulation test omits them; this guards the `cache_read`/`cache_creation` pointers so a
        // regression can't silently zero cache billing.
        let sse = b"event: message_start\n\
                    data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":20,\"output_tokens\":0,\"cache_read_input_tokens\":12,\"cache_creation_input_tokens\":8}}}\n\n\
                    event: message_delta\n\
                    data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":15}}\n\n";
        assert_eq!(
            anthropic_stream(sse).unwrap(),
            Usage {
                input_tokens: 20,
                output_tokens: 15,
                cache_read_tokens: 12,
                cache_write_tokens: 8
            }
        );
    }

    #[test]
    fn no_usage_returns_none() {
        assert!(openai_stream(b"data: {\"choices\":[]}\n\n").is_none());
        assert!(anthropic_body(b"{}").map(|u| u.input_tokens).unwrap_or(0) == 0);
    }
}
