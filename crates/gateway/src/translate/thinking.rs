//! Thinking and reasoning: effort and budget mappings, and the thinking blocks that cross the wires.

use super::*;

/// A thinking request, whichever shape the client used, before it is fit to the model.
enum ThinkingIntent<'a> {
    Off,
    BetweenTools,
    Budget(u64),
    Effort(&'a str),
}

/// Returns whether the client asked for no reasoning on an adaptive-generation model, where that
/// is an omitted `thinking` at effort `low` (see `thinking_off_between_tools` for the exception).
pub(super) fn openai_reasoning_to_anthropic(
    v: &Value,
    out: &mut Map<String, Value>,
    claude: ClaudeModel,
    max_tokens: u64,
) -> bool {
    let effort = v
        .get("reasoning_effort")
        .and_then(Value::as_str)
        .or_else(|| v.pointer("/reasoning/effort").and_then(Value::as_str));
    // A client may send Anthropic's own `thinking` object on the OpenAI wire. It is a request for
    // thinking, not a finished shape: `budget_tokens` is a 400 on current models and adaptive a
    // 400 before 4.6, so it goes through the same model-aware mapping as `reasoning_effort`.
    let native = v.get("thinking").filter(|t| t.is_object());
    let intent = match native.and_then(|t| t.get("type")).and_then(Value::as_str) {
        Some("disabled") => ThinkingIntent::Off,
        Some("between_tools") => ThinkingIntent::BetweenTools,
        Some("enabled") => match native
            .and_then(|t| t.get("budget_tokens"))
            .and_then(Value::as_u64)
        {
            Some(n) if effort.is_none() => ThinkingIntent::Budget(n),
            _ => ThinkingIntent::Effort(effort.unwrap_or("medium")),
        },
        Some("adaptive") => ThinkingIntent::Effort(effort.unwrap_or("high")),
        Some(_) => {
            // An unknown thinking type: the provider names it.
            if let Some(t) = native {
                out.insert("thinking".into(), t.clone());
            }
            return false;
        }
        None => match effort {
            Some("none" | "off" | "disabled") => ThinkingIntent::Off,
            Some(e) => ThinkingIntent::Effort(e),
            None => return false,
        },
    };
    let intent = match intent {
        ThinkingIntent::BetweenTools if !claude.between_tools => ThinkingIntent::Off,
        other => other,
    };
    match claude.reasoning {
        ClaudeGen::Budget => {
            let budget = match intent {
                ThinkingIntent::Off | ThinkingIntent::BetweenTools => {
                    out.insert("thinking".into(), json!({ "type": "disabled" }));
                    return false;
                }
                ThinkingIntent::Budget(n) => n,
                ThinkingIntent::Effort(e) => budget_for_effort(e).min(max_tokens / 2),
            };
            // `budget_tokens` must be at least 1024 and below `max_tokens`, which also has to hold
            // the answer. Leave the answer at least half; too small a request thinks not at all.
            if max_tokens <= MIN_THINKING_BUDGET {
                return false;
            }
            let budget = budget.clamp(MIN_THINKING_BUDGET, max_tokens - 1);
            let mut t = json!({ "type": "enabled", "budget_tokens": budget });
            copy_display(&mut t, native);
            out.insert("thinking".into(), t);
            false
        }
        ClaudeGen::Adaptive46 | ClaudeGen::Adaptive => {
            let requested = match intent {
                ThinkingIntent::BetweenTools => {
                    out.insert("thinking".into(), json!({ "type": "between_tools" }));
                    return false;
                }
                ThinkingIntent::Off => "none",
                ThinkingIntent::Budget(n) => effort_from_budget(n),
                ThinkingIntent::Effort(e) => e,
            };
            let level = match effort_alias(requested) {
                "none" => "low",
                "xhigh" if claude.reasoning == ClaudeGen::Adaptive46 => "max",
                "xhigh" if requested == "max" => "max",
                other => other,
            };
            if !matches!(intent, ThinkingIntent::Off) {
                let mut t = json!({ "type": "adaptive" });
                copy_display(&mut t, native);
                // A Responses client asking for a reasoning summary: current models default to
                // `omitted`, which streams empty thinking blocks.
                if claude.reasoning == ClaudeGen::Adaptive
                    && t.get("display").is_none()
                    && v.pointer("/reasoning/summary")
                        .and_then(Value::as_str)
                        .is_some_and(|s| s != "none")
                    && let Some(obj) = t.as_object_mut()
                {
                    obj.insert("display".into(), json!("summarized"));
                }
                out.insert("thinking".into(), t);
            }
            set_output_config(out, "effort", json!(level));
            matches!(intent, ThinkingIntent::Off)
        }
    }
}

/// Carry a client's `thinking.display` (`summarized` / `omitted` / …) onto the mapped object.
fn copy_display(t: &mut Value, native: Option<&Value>) {
    if let Some(d) = native.and_then(|n| n.get("display"))
        && let Some(obj) = t.as_object_mut()
    {
        obj.insert("display".into(), d.clone());
    }
}

/// Anthropic's floor for `thinking.budget_tokens`.
const MIN_THINKING_BUDGET: u64 = 1024;

pub(super) fn budget_for_effort(effort: &str) -> u64 {
    match effort {
        "minimal" | "low" => 1024,
        "high" => 8192,
        "xhigh" | "max" => 16384,
        _ => 4096,
    }
}

pub(super) fn effort_from_thinking(thinking: &Value) -> Option<&'static str> {
    match thinking.get("type").and_then(Value::as_str) {
        Some("disabled") => Some("none"),
        Some("adaptive") => Some(
            thinking
                .get("effort")
                .and_then(Value::as_str)
                .map(effort_alias)
                .unwrap_or("high"),
        ),
        Some("enabled") => {
            let budget = thinking
                .get("budget_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(4096);
            Some(effort_from_budget(budget))
        }
        _ => None,
    }
}

pub(super) fn effort_alias(s: &str) -> &'static str {
    match s {
        "none" | "off" | "disabled" => "none",
        "minimal" | "low" => "low",
        "high" => "high",
        "xhigh" | "max" => "xhigh",
        _ => "medium",
    }
}

pub(super) fn effort_from_budget(budget: u64) -> &'static str {
    if budget <= 1024 {
        "low"
    } else if budget <= 4096 {
        "medium"
    } else if budget <= 8192 {
        "high"
    } else {
        "xhigh"
    }
}

/// Longest thinking text held back waiting for its signature. Past it the block is dropped: a
/// signature covers the whole text, so a truncated block could never be sent back.
pub(super) const MAX_HELD_THINKING: usize = 8 * 1024 * 1024;

/// Whether an OpenRouter `reasoning_details` entry's opaque payload is Anthropic's. Only those
/// verify at Anthropic; an OpenAI or Gemini blob presented as a signature would 400 there.
pub(super) fn anthropic_format(detail: &Value) -> bool {
    detail
        .get("format")
        .and_then(Value::as_str)
        .is_none_or(|f| f.starts_with("anthropic"))
}

/// Gathers one thinking block until it is whole. OpenRouter streams a Claude block as
/// `reasoning_details` `reasoning.text` pieces sharing an `index`, then one more carrying only its
/// `signature`; a non-stream message carries both in one entry. Finished blocks come out in
/// Anthropic's shape — `thinking` (with `signature` when there was one) or `redacted_thinking`.
#[derive(Default)]
pub(super) struct Gather {
    index: Option<u64>,
    text: String,
    active: bool,
    overflow: bool,
}

impl Gather {
    /// One OpenRouter `reasoning_details` entry.
    pub(super) fn detail(&mut self, d: &Value, done: &mut Vec<Value>) {
        let index = d.get("index").and_then(Value::as_u64);
        match d.get("type").and_then(Value::as_str) {
            Some("reasoning.text") => {
                self.text(
                    index,
                    d.get("text").and_then(Value::as_str).unwrap_or(""),
                    done,
                );
                if let Some(sig) = non_empty_str(d, "signature") {
                    if anthropic_format(d) {
                        self.sign(sig, done);
                    } else {
                        self.close(done);
                    }
                }
            }
            Some("reasoning.summary") => {
                self.text(
                    index,
                    d.get("summary").and_then(Value::as_str).unwrap_or(""),
                    done,
                );
            }
            Some("reasoning.encrypted") => {
                self.close(done);
                if anthropic_format(d)
                    && let Some(data) = non_empty_str(d, "data")
                {
                    done.push(json!({ "type": "redacted_thinking", "data": data }));
                }
            }
            _ => {}
        }
    }

    pub(super) fn text(&mut self, index: Option<u64>, text: &str, done: &mut Vec<Value>) {
        if self.active && index.is_some() && self.index.is_some() && index != self.index {
            self.close(done);
        }
        if !self.active {
            self.active = true;
            self.index = index;
        }
        if self.overflow {
            return;
        }
        if self.text.len().saturating_add(text.len()) > MAX_HELD_THINKING {
            self.overflow = true;
            self.text = String::new();
            return;
        }
        self.text.push_str(text);
    }

    /// The block is whole and signed.
    pub(super) fn sign(&mut self, signature: &str, done: &mut Vec<Value>) {
        if !self.overflow {
            done.push(json!({
                "type": "thinking",
                "thinking": std::mem::take(&mut self.text),
                "signature": signature,
            }));
        }
        *self = Self::default();
    }

    /// The block ended without a signature.
    pub(super) fn close(&mut self, done: &mut Vec<Value>) {
        if self.active && !self.overflow {
            done.push(json!({ "type": "thinking", "thinking": std::mem::take(&mut self.text) }));
        }
        *self = Self::default();
    }

    /// Forget the block being gathered: it ended, and will never be signed.
    pub(super) fn discard(&mut self) {
        *self = Self::default();
    }
}

/// The thinking blocks of a Chat Completions message, in order: our own `thinking` array (what
/// this module writes for a Messages upstream), else OpenRouter's `reasoning_details`, else the
/// plain `reasoning_content` / `reasoning` text, which is never signed.
pub(super) fn chat_thinking_blocks(m: &Value) -> Vec<Value> {
    if let Some(blocks) = m.get("thinking").and_then(Value::as_array) {
        return blocks.iter().map(anthropic_thinking_block).collect();
    }
    let mut out = Vec::new();
    if let Some(details) = m.get("reasoning_details").and_then(Value::as_array) {
        let mut gather = Gather::default();
        for d in details {
            gather.detail(d, &mut out);
        }
        gather.close(&mut out);
        return out;
    }
    if let Some(text) =
        non_empty_str(m, "reasoning_content").or_else(|| non_empty_str(m, "reasoning"))
    {
        out.push(json!({ "type": "thinking", "thinking": text }));
    }
    out
}

/// One entry of our `thinking` array (or a thinking content part) in exactly Anthropic's block
/// shape. A streamed entry also carries the `index` it was accumulated under, which goes.
pub(super) fn anthropic_thinking_block(b: &Value) -> Value {
    if b.get("type").and_then(Value::as_str) == Some("redacted_thinking") {
        return json!({
            "type": "redacted_thinking",
            "data": b.get("data").cloned().unwrap_or(json!("")),
        });
    }
    let mut t = Map::new();
    t.insert("type".into(), json!("thinking"));
    let text = b
        .get("thinking")
        .or_else(|| b.get("text"))
        .and_then(Value::as_str)
        .unwrap_or("");
    t.insert("thinking".into(), json!(text));
    if let Some(sig) = non_empty_str(b, "signature") {
        t.insert("signature".into(), json!(sig));
    }
    Value::Object(t)
}

/// The `format` OpenRouter stamps on Claude's reasoning, and requires to replay it to Anthropic.
const ANTHROPIC_REASONING_FORMAT: &str = "anthropic-claude-v1";

/// Signed Anthropic thinking as OpenRouter `reasoning_details`: `reasoning.text` with its
/// signature, `reasoning.encrypted` for a redacted block. OpenRouter replays nothing else — not
/// `reasoning_content`, not our `thinking` array — so without these a thinking + tool loop on a
/// Claude model OpenRouter serves 400s on turn 2 ("a final `assistant` message must start with a
/// thinking block"). Unsigned blocks are left out: they would not verify.
pub(super) fn reasoning_details(blocks: &[Value]) -> Vec<Value> {
    blocks
        .iter()
        .filter(|b| is_replayable_thinking(b))
        .enumerate()
        .map(|(index, b)| match b.get("type").and_then(Value::as_str) {
            Some("redacted_thinking") => json!({
                "type": "reasoning.encrypted",
                "data": b.get("data").cloned().unwrap_or(json!("")),
                "format": ANTHROPIC_REASONING_FORMAT,
                "index": index,
            }),
            _ => json!({
                "type": "reasoning.text",
                "text": b.get("thinking").cloned().unwrap_or(json!("")),
                "signature": b.get("signature").cloned().unwrap_or(json!("")),
                "format": ANTHROPIC_REASONING_FORMAT,
                "index": index,
            }),
        })
        .collect()
}

/// Add thinking blocks to a Chat Completions assistant message: our `thinking` array (read back by
/// `chat_thinking_blocks` onto Messages), plus `reasoning_details` when the upstream is a Claude
/// model (see [`reasoning_details`]).
pub(super) fn attach_thinking(
    msg: &mut Map<String, Value>,
    add: Vec<Value>,
    claude_thinking: bool,
) {
    if add.is_empty() {
        return;
    }
    let mut blocks = msg
        .get("thinking")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    blocks.extend(add);
    if claude_thinking {
        let details = reasoning_details(&blocks);
        if !details.is_empty() {
            msg.insert("reasoning_details".into(), Value::Array(details));
        }
    }
    msg.insert("thinking".into(), Value::Array(blocks));
}

/// A thinking block an Anthropic client may hold: signed, or redacted. An unsigned `thinking`
/// block would be echoed back on the next turn and a turn served by Anthropic would 400 on it
/// ("thinking.signature: Field required") for the rest of the conversation — so it is dropped.
pub(super) fn is_replayable_thinking(block: &Value) -> bool {
    match block.get("type").and_then(Value::as_str) {
        Some("thinking") => non_empty_str(block, "signature").is_some(),
        Some("redacted_thinking") => true,
        _ => false,
    }
}

pub(super) fn is_thinking_block(block: &Value) -> bool {
    matches!(
        block.get("type").and_then(Value::as_str),
        Some("thinking" | "redacted_thinking")
    )
}
