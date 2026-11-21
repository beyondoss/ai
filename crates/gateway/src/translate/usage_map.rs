//! Usage counts as each wire reports them.

use super::*;

fn u64_at(v: &Value, ptr: &str) -> Option<u64> {
    v.pointer(ptr).and_then(Value::as_u64)
}

/// Token counts in one shape, whichever dialect reported them.
///
/// Anthropic's `input_tokens` counts only the prompt tokens that were neither read from nor
/// written to the cache. OpenAI's `prompt_tokens` / Responses `input_tokens` count the whole
/// prompt, with the cached share broken out beneath it. Passing either number through as the
/// other told an OpenAI client a 7,000-token prompt was 10 tokens, and an Anthropic client that a
/// mostly-cached prompt was uncached.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(super) struct Usage {
    uncached: u64,
    cache_read: u64,
    cache_write: u64,
    output: u64,
    reasoning: Option<u64>,
}

impl Usage {
    pub(super) fn from_anthropic(u: &Value) -> Self {
        let mut usage = Self::default();
        usage.merge_anthropic(u);
        usage
    }

    /// Take what `u` reports. A `message_delta` repeats cumulative totals and omits those that do
    /// not apply, so an absent field keeps the `message_start` value.
    pub(super) fn merge_anthropic(&mut self, u: &Value) {
        if let Some(n) = u64_at(u, "/input_tokens") {
            self.uncached = n;
        }
        if let Some(n) = u64_at(u, "/cache_read_input_tokens") {
            self.cache_read = n;
        }
        if let Some(n) = u64_at(u, "/cache_creation_input_tokens") {
            self.cache_write = n;
        }
        if let Some(n) = u64_at(u, "/output_tokens") {
            self.output = n;
        }
        if let Some(n) = u64_at(u, "/output_tokens_details/thinking_tokens") {
            self.reasoning = Some(n);
        }
    }

    pub(super) fn from_chat(u: &Value) -> Self {
        let cache_read = u64_at(u, "/prompt_tokens_details/cached_tokens").unwrap_or(0);
        // OpenAI and OpenRouter spell cache writes `cache_write_tokens`; LiteLLM keeps Anthropic's.
        let cache_write = u64_at(u, "/prompt_tokens_details/cache_write_tokens")
            .or_else(|| u64_at(u, "/cache_creation_input_tokens"))
            .unwrap_or(0);
        let prompt = u64_at(u, "/prompt_tokens").unwrap_or(0);
        let completion = u64_at(u, "/completion_tokens").unwrap_or(0);
        let reasoning = u64_at(u, "/completion_tokens_details/reasoning_tokens");
        // xAI reports reasoning beside `completion_tokens` (total = prompt + completion +
        // reasoning) and bills it as output, like `usage::openai_body` does; a translated client
        // must be shown the output it is billed.
        let r = reasoning.unwrap_or(0);
        // Saturating: provider numbers are untrusted and overflow-checks are on.
        let outside =
            u64_at(u, "/total_tokens") == Some(prompt.saturating_add(completion).saturating_add(r));
        Self {
            uncached: prompt
                .saturating_sub(cache_read)
                .saturating_sub(cache_write),
            cache_read,
            cache_write,
            output: completion.saturating_add(if outside { r } else { 0 }),
            reasoning,
        }
    }

    pub(super) fn from_responses(u: &Value) -> Self {
        let cache_read = u64_at(u, "/input_tokens_details/cached_tokens").unwrap_or(0);
        let cache_write = u64_at(u, "/input_tokens_details/cache_write_tokens").unwrap_or(0);
        let input = u64_at(u, "/input_tokens").unwrap_or(0);
        Self {
            uncached: input.saturating_sub(cache_read).saturating_sub(cache_write),
            cache_read,
            cache_write,
            output: u64_at(u, "/output_tokens").unwrap_or(0),
            reasoning: u64_at(u, "/output_tokens_details/reasoning_tokens"),
        }
    }

    /// When `on`, cache writes become uncached input: the gateway added the breakpoints that caused
    /// them, and records them as `gateway_cache_write_tokens` (see [`request_with_tools`]). The whole prompt is
    /// unchanged; only the cache-write share moves.
    pub(super) fn writes_as_input(mut self, on: bool) -> Self {
        if on {
            self.uncached = self.uncached.saturating_add(self.cache_write);
            self.cache_write = 0;
        }
        self
    }

    fn prompt(&self) -> u64 {
        self.uncached
            .saturating_add(self.cache_read)
            .saturating_add(self.cache_write)
    }

    pub(super) fn to_anthropic(self) -> Value {
        let mut m = Map::new();
        m.insert("input_tokens".into(), json!(self.uncached));
        m.insert(
            "cache_creation_input_tokens".into(),
            json!(self.cache_write),
        );
        m.insert("cache_read_input_tokens".into(), json!(self.cache_read));
        m.insert("output_tokens".into(), json!(self.output));
        if let Some(r) = self.reasoning {
            m.insert(
                "output_tokens_details".into(),
                json!({ "thinking_tokens": r }),
            );
        }
        Value::Object(m)
    }

    /// Cache writes ride `prompt_tokens_details.cache_write_tokens`, the field OpenAI and
    /// OpenRouter both use; `prompt_tokens` includes them, as it includes cache reads.
    pub(super) fn to_chat(self) -> Value {
        let mut m = Map::new();
        m.insert("prompt_tokens".into(), json!(self.prompt()));
        m.insert("completion_tokens".into(), json!(self.output));
        m.insert(
            "total_tokens".into(),
            json!(self.prompt().saturating_add(self.output)),
        );
        m.insert(
            "prompt_tokens_details".into(),
            json!({ "cached_tokens": self.cache_read, "cache_write_tokens": self.cache_write }),
        );
        if let Some(r) = self.reasoning {
            m.insert(
                "completion_tokens_details".into(),
                json!({ "reasoning_tokens": r }),
            );
        }
        Value::Object(m)
    }

    /// Every detail field is required by the Responses schema, so unknown counts are 0.
    pub(super) fn to_responses(self) -> Value {
        json!({
            "input_tokens": self.prompt(),
            "input_tokens_details": {
                "cached_tokens": self.cache_read,
                "cache_write_tokens": self.cache_write,
            },
            "output_tokens": self.output,
            "output_tokens_details": { "reasoning_tokens": self.reasoning.unwrap_or(0) },
            "total_tokens": self.prompt().saturating_add(self.output),
        })
    }
}
