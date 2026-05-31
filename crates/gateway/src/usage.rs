//! Token-usage extraction — the "passive tap" the gateway emits as billing *facts*.
//!
//! We never compute price here (pricing is a closed downstream consumer); we only extract raw
//! token counts. Two shapes per provider: the non-streaming JSON body, and the terminal event of
//! an SSE stream. For streaming we scan the relayed bytes for the usage event but never block the
//! relay on it (see `proxy`).

use serde::Deserialize;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
}

// Typed views of just the fields we meter. Deserializing into these (rather than a
// `serde_json::Value` DOM) lets serde skip every field we don't read without allocating a node for
// it — no `Map`/`String`/`Number` tree to build and drop per body or per SSE line. Every field is
// `#[serde(default)]` so a missing or partial `usage` block reads as zeros, matching the prior
// pointer-with-`unwrap_or(0)` behavior.

/// OpenAI `usage` block (chat/completions + responses). `prompt`/`completion` map to in/out; cached
/// input rides in `prompt_tokens_details.cached_tokens`. No cache-write concept on the OpenAI wire.
#[derive(Deserialize, Default)]
struct OpenAiUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
    #[serde(default)]
    prompt_tokens_details: OpenAiPromptDetails,
}

#[derive(Deserialize, Default)]
struct OpenAiPromptDetails {
    #[serde(default)]
    cached_tokens: u64,
}

impl From<OpenAiUsage> for Usage {
    fn from(u: OpenAiUsage) -> Self {
        Usage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
            cache_read_tokens: u.prompt_tokens_details.cached_tokens,
            cache_write_tokens: 0,
        }
    }
}

/// Anthropic `usage` block (`/v1/messages` body + streaming events).
#[derive(Deserialize, Default)]
struct AnthropicUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    cache_read_input_tokens: u64,
    #[serde(default)]
    cache_creation_input_tokens: u64,
}

/// OpenAI non-streaming: top-level `usage`. `None` (absent/`null`) ⇒ no usage to meter.
pub fn openai_body(body: &[u8]) -> Option<Usage> {
    #[derive(Deserialize)]
    struct Body {
        usage: Option<OpenAiUsage>,
    }
    serde_json::from_slice::<Body>(body)
        .ok()?
        .usage
        .map(Usage::from)
}

/// Anthropic non-streaming: top-level `usage.{input,output,cache_*}`.
pub fn anthropic_body(body: &[u8]) -> Option<Usage> {
    #[derive(Deserialize)]
    struct Body {
        usage: Option<AnthropicUsage>,
    }
    let u = serde_json::from_slice::<Body>(body).ok()?.usage?;
    Some(Usage {
        input_tokens: u.input_tokens,
        output_tokens: u.output_tokens,
        cache_read_tokens: u.cache_read_input_tokens,
        cache_write_tokens: u.cache_creation_input_tokens,
    })
}

/// Iterate the raw JSON payloads carried on `data:` lines of an SSE byte stream. `[DONE]` and the
/// `data:` framing are stripped; each caller deserializes the payload into its own typed view.
fn sse_data_lines(sse: &[u8]) -> impl Iterator<Item = &[u8]> + '_ {
    sse.split(|&b| b == b'\n').filter_map(|line| {
        let line = line.strip_prefix(b"data:")?;
        let line = line.strip_prefix(b" ").unwrap_or(line);
        (line != b"[DONE]").then_some(line)
    })
}

/// OpenAI streaming (requires `stream_options.include_usage`): the penultimate chunk carries a
/// top-level `usage` object. Last one with usage wins.
pub fn openai_stream(sse: &[u8]) -> Option<Usage> {
    #[derive(Deserialize)]
    struct Chunk {
        usage: Option<OpenAiUsage>,
    }
    let mut found = None;
    for line in sse_data_lines(sse) {
        if let Ok(chunk) = serde_json::from_slice::<Chunk>(line) {
            if let Some(u) = chunk.usage {
                found = Some(Usage::from(u));
            }
        }
    }
    found
}

/// Anthropic streaming: input + cache tokens arrive in `message_start.message.usage`; output
/// accumulates in `message_delta.usage.output_tokens` (last delta is the cumulative total).
pub fn anthropic_stream(sse: &[u8]) -> Option<Usage> {
    #[derive(Deserialize)]
    struct Message {
        usage: Option<AnthropicUsage>,
    }
    #[derive(Deserialize)]
    struct Chunk {
        // `message_start` nests usage under `message`; `message_delta` carries it top-level.
        message: Option<Message>,
        usage: Option<AnthropicUsage>,
    }
    let mut usage = Usage::default();
    let mut saw_any = false;
    for line in sse_data_lines(sse) {
        let Ok(chunk) = serde_json::from_slice::<Chunk>(line) else {
            continue;
        };
        if let Some(u) = chunk.message.and_then(|m| m.usage) {
            usage.input_tokens = u.input_tokens;
            usage.cache_read_tokens = u.cache_read_input_tokens;
            usage.cache_write_tokens = u.cache_creation_input_tokens;
            saw_any = true;
        }
        if let Some(u) = chunk.usage {
            // message_delta carries the running output token count.
            if u.output_tokens > 0 {
                usage.output_tokens = u.output_tokens;
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
