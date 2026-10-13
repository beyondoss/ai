//! How many real prompt tokens a stretch of the conversation costs — the size compaction and
//! masking reason about, measured the way the provider bills it rather than in characters.
//!
//! Two sources, the first wherever it applies:
//!
//! 1. **Anchors.** Every request reports its real prompt size (`input + cache_read + cache_write`,
//!    [`TokenUsage::prompt_tokens`]). [`Session::record_usage`](crate::session::Session::record_usage)
//!    keys it to the message boundary the request saw, and two consecutive requests' sizes differ by
//!    exactly the real tokens of the turn between them (the previous assistant message and the user
//!    message answering it). [`Session::push`](crate::session::Session::push) stamps that difference
//!    on the next assistant message as [`Message::prior_turn_tokens`], which is persisted with it, so
//!    a reopened session keeps every measured turn. The real size of any suffix that starts at an
//!    assistant boundary is then exact: the live size minus the measured turns before it. Exact
//!    for everything a character count cannot see: Claude's summarized thinking (the stored text is
//!    a summary; the prompt carries the full thinking, 5.7–6.4× as long, and one Haiku turn carried
//!    56K hidden tokens behind about 2K chars), the tokenizer, and per-block framing.
//! 2. **The calibrated estimate** ([`message_tokens`]), where no anchor applies: the first message,
//!    the turn whose request has not happened yet, a turn masking rewrote, a turn measured on a
//!    different model, and every turn of a session persisted before anchors existed.
//!
//! Character counts divided by four are used nowhere a size spans several messages, except as the
//! [`TokenFamily::Other`] fallback for model families never measured.
//!
//! ## Evidence
//!
//! Measured on every longctx eval run (793 runs, about 73K requests; Claude Haiku/Sonnet/Opus 5.5,
//! GPT-6 Luna and GPT-6.1 Sol), against the prompt sizes the providers reported. chars/4 was 1.9×
//! low on Sonnet and Opus and 2.3–2.5× low on Haiku for the whole conversation, and 1.11–1.14× low
//! on GPT. The calibrated estimate alone (no anchors) is 0.96–1.02× real at the median, p10–p90
//! 0.93–1.08, for every family; with anchors, the `keep_recent` tail is 0.998–1.000× real at the
//! median, p10–p90 0.987–1.014, and a compaction's `tokens_after` 0.99–1.01× the next request's
//! real prompt. `crates/agent/eval/longctx/token_sizing.py` reproduces every number here from the
//! eval output.

use crate::compaction::MASKED_PREFIX;
use crate::message::{ContentBlock, Message, Role, TokenUsage};

/// A model family whose tokenizer and prompt shape the estimate is calibrated for. Picked from the
/// model id by [`TokenFamily::for_model`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TokenFamily {
    /// Claude (and Fable). Measured on Haiku, Sonnet and Opus 5.5.
    Claude,
    /// OpenAI's GPT models. Measured on GPT-6 Luna and GPT-6.1 Sol.
    Gpt,
    /// Every family never measured: chars/4 everywhere, the estimate this crate used before
    /// calibration, and the 20K `keep_recent` it shipped with.
    #[default]
    Other,
}

/// The per-family constants of the calibrated estimate. A rate is in tenths of a character per
/// token (`25` = 2.5 chars/token), so the arithmetic stays integral.
#[derive(Debug, Clone, Copy)]
struct Calibration {
    /// Tool-result content.
    tool_tenths: u32,
    /// Tokens of framing each tool-result block costs beyond its content.
    framing: u32,
    /// Prose: user turns, a compaction summary, an assistant message with no recorded usage.
    text_tenths: u32,
    /// Percent of an assistant message's recorded reasoning tokens that do *not* come back into
    /// the prompt.
    reasoning_dropped_pct: u32,
}

/// Claude 5.x.
///
/// - Tool results at 2.5 chars/token: a regression over 7.5K clean request pairs (Haiku 4,547,
///   Sonnet 2,407, Opus 566) fits code at 2.5–2.7, test logs and listings at 2.3–2.5, JSON at 2.3.
/// - 80 tokens of framing per tool result: the same regression puts it at 70–120 with an
///   intercept and 28–77 without one (the two are confounded with the tool call's own encoding);
///   80 centres the whole-conversation check below.
/// - Prose at 2.3 chars/token: 77 compaction summaries measured against the next request's real
///   prompt, median 2.01 (Haiku), 2.25 (Sonnet), 2.40 (Opus).
/// - An assistant message costs its recorded output tokens: the prompt carries the full thinking
///   (regression coefficient on reasoning tokens 1.00 on all three), which the stored summary of
///   it cannot show.
///
/// Whole-conversation check (`token_sizing.py`'s `whole`: real growth over the first request's,
/// against the estimate, at every request with 20K+ of content): median 1.02 (Haiku), 0.96
/// (Sonnet), 0.97 (Opus), p10–p90 0.93–1.04; chars/4 gave 2.20, 1.83 and 1.83.
const CLAUDE: Calibration = Calibration {
    tool_tenths: 25,
    framing: 80,
    text_tenths: 23,
    reasoning_dropped_pct: 0,
};

/// GPT-6.
///
/// - Tool results at 3.5 chars/token: code 3.5, test logs 3.8–4.1, listings 3.6, JSON 3.25 over
///   30.8K clean pairs; framing per result fits at 20–24 tokens and is left at 0, which centres the
///   whole-conversation check.
/// - Prose at 4.0 chars/token: 33 compaction summaries, median 4.37 (Luna) and 3.72 (Sol).
/// - An assistant message costs its output tokens less 70% of its reasoning tokens: only about 30%
///   of reasoning comes back into the prompt (regression coefficient 0.29 on Luna, 0.35 on Sol).
///
/// Whole-conversation check: median 1.00 (Luna), 1.02 (Sol), p10–p90 0.95–1.08; chars/4 gave 1.12
/// and 1.14.
const GPT: Calibration = Calibration {
    tool_tenths: 35,
    framing: 0,
    text_tenths: 40,
    reasoning_dropped_pct: 70,
};

/// The flat estimate a dropped image costs (Anthropic bills vision at about this much).
const IMAGE_TOKENS: u32 = 1_200;

/// The `keep_recent` tail the estimate-based cut shipped with, before anchors.
const LEGACY_KEEP_RECENT: u32 = 20_000;

impl TokenFamily {
    /// The family of `model` (any case; a vendor slug like `anthropic/` and a Bedrock
    /// `[region.]anthropic.` prefix are ignored). `gpt-oss` is an open-weight model, not a GPT.
    pub fn for_model(model: &str) -> Self {
        let id = model.rsplit('/').next().unwrap_or(model);
        let id = match find_ci(id, "anthropic.") {
            Some(at) => &id[at + "anthropic.".len()..],
            None => id,
        };
        if starts_with_ci(id, "claude") || starts_with_ci(id, "fable") {
            TokenFamily::Claude
        } else if starts_with_ci(id, "gpt-") && !starts_with_ci(id, "gpt-oss") {
            TokenFamily::Gpt
        } else {
            TokenFamily::Other
        }
    }

    fn calibration(self) -> Option<Calibration> {
        match self {
            TokenFamily::Claude => Some(CLAUDE),
            TokenFamily::Gpt => Some(GPT),
            TokenFamily::Other => None,
        }
    }

    /// The default `keep_recent` tail, in real tokens: the value that reproduces the tail the
    /// threshold sweep validated, which ran on the chars/4 cut with a nominal 20K budget.
    ///
    /// Measured (`token_sizing.py --budgets`, the cut simulated at every request of the clean eval
    /// runs): the old cut kept a real median of 45.0K on Haiku, 37.0K on Sonnet and 35.5K on Opus
    /// (37.9K pooled over 1,002 cuts), and 20.7K on both GPT models (12,503 cuts). The anchored cut,
    /// which keeps at most its budget, reproduces those pooled medians at 40.5K (37.8K) and 22K
    /// (20.7K). Per model that is 36.2K on Haiku (its old tail, heavy with thinking, was also the
    /// noisiest: p10–p90 33–64K, now 24–40K), 38.2K on Sonnet, 37.9K on Opus, 20.7K on Luna and
    /// 19.7K on Sol. The tail size itself is for a later A/B to choose; this keeps the validated
    /// behaviour.
    pub const fn keep_recent_tokens(self) -> u32 {
        match self {
            TokenFamily::Claude => 40_500,
            TokenFamily::Gpt => 22_000,
            TokenFamily::Other => LEGACY_KEEP_RECENT,
        }
    }

    /// Scale a budget the old chars/4 window walk was tuned in (masking's kept window, a quarter of
    /// the trigger) to real tokens, by the ratio [`Self::keep_recent_tokens`] measured for the same
    /// walk: 2.025 on Claude, 1.1 on GPT, 1 otherwise. On masking's own window the old walk kept
    /// 2.20× its nominal budget on Haiku, 1.84× on Sonnet, 1.80× on Opus and 1.06–1.07× on GPT
    /// (`token_sizing.py`'s mask window).
    pub fn scale_span(self, legacy: u32) -> u32 {
        scale(legacy, self.keep_recent_tokens(), LEGACY_KEEP_RECENT)
    }

    /// Scale a count of tool-output tokens the old estimate measured (chars/4) to the calibrated
    /// one: × 1.6 on Claude, × 40/35 on GPT, unchanged otherwise. Masking's least-it-must-free bar
    /// was set in chars/4 tokens of freed tool output; scaled, the same passes qualify.
    pub fn scale_tool(self, legacy: u32) -> u32 {
        match self.calibration() {
            Some(c) => scale(legacy, 40, c.tool_tenths),
            None => legacy,
        }
    }

    /// The tokens one assistant message costs in later prompts, from the usage the provider reported
    /// for the turn that produced it.
    pub fn assistant_tokens(self, usage: &TokenUsage) -> u32 {
        match self.calibration() {
            Some(c) => usage.output_tokens.saturating_sub(
                (u64::from(usage.reasoning_tokens) * u64::from(c.reasoning_dropped_pct) / 100)
                    as u32,
            ),
            None => usage.output_tokens,
        }
    }
}

fn scale(n: u32, num: u32, den: u32) -> u32 {
    (u64::from(n) * u64::from(num) / u64::from(den.max(1))).min(u64::from(u32::MAX)) as u32
}

fn starts_with_ci(s: &str, prefix: &str) -> bool {
    s.len() >= prefix.len() && s.as_bytes()[..prefix.len()].eq_ignore_ascii_case(prefix.as_bytes())
}

fn find_ci(s: &str, needle: &str) -> Option<usize> {
    let (s, n) = (s.as_bytes(), needle.as_bytes());
    s.windows(n.len()).rposition(|w| w.eq_ignore_ascii_case(n))
}

fn tokens(chars: usize, tenths: u32) -> u32 {
    (chars as u64 * 10)
        .div_ceil(u64::from(tenths))
        .min(u64::from(u32::MAX)) as u32
}

/// Estimated tokens of a stretch of prose (a summary, an instruction).
pub fn text_tokens(text: &str, family: TokenFamily) -> u32 {
    let chars = text.chars().count();
    match family.calibration() {
        Some(c) => tokens(chars, c.text_tenths),
        None => chars.div_ceil(4) as u32,
    }
}

/// Estimated tokens of a tool result's content alone — not its framing, which stays when masking
/// replaces the content.
pub fn tool_content_tokens(chars: usize, family: TokenFamily) -> u32 {
    match family.calibration() {
        Some(c) => tokens(chars, c.tool_tenths),
        None => chars.div_ceil(4) as u32,
    }
}

/// The calibrated estimate of what `m` costs in a prompt. An assistant message with recorded usage
/// costs [`TokenFamily::assistant_tokens`]; anything else is sized from its characters at the
/// family's rates, plus 1,200 per image. [`TokenFamily::Other`] is chars/4 throughout.
pub fn message_tokens(m: &Message, family: TokenFamily) -> u32 {
    if family != TokenFamily::Other
        && m.role == Role::Assistant
        && let Some(usage) = m.usage.filter(|u| u.output_tokens > 0)
    {
        return family.assistant_tokens(&usage);
    }
    content_tokens(m, family)
}

/// [`message_tokens`] from the characters alone, ignoring recorded usage: what the stored text
/// costs once rendered (a summary request's transcript, which shows a summary of the thinking).
pub fn content_tokens(m: &Message, family: TokenFamily) -> u32 {
    let Some(c) = family.calibration() else {
        return legacy_message_tokens(m);
    };
    let (mut tool_chars, mut text_chars, mut results, mut images) = (0usize, 0usize, 0u32, 0u32);
    for b in &m.content {
        match b {
            ContentBlock::ToolResult {
                content, images: i, ..
            } => {
                results += 1;
                images += i.len() as u32;
                tool_chars += content.chars().count();
            }
            ContentBlock::Image { .. } => images += 1,
            b => text_chars += block_chars(b),
        }
    }
    let chars = (tool_chars as u64 * 10).div_ceil(u64::from(c.tool_tenths))
        + (text_chars as u64 * 10).div_ceil(u64::from(c.text_tenths));
    (chars.min(u64::from(u32::MAX)) as u32)
        .saturating_add(results.saturating_mul(c.framing))
        .saturating_add(images.saturating_mul(IMAGE_TOKENS))
}

/// The pre-calibration estimate: one `ceil(chars / 4)` over the whole message (matching pi's
/// `estimateTokens`, which sums a message's characters before its single division), an image at
/// 4,800 chars. [`TokenFamily::Other`]'s estimate.
fn legacy_message_tokens(m: &Message) -> u32 {
    let chars: usize = m
        .content
        .iter()
        .map(|b| match b {
            ContentBlock::ToolResult { content, .. } => content.chars().count(),
            ContentBlock::Image { .. } => 4_800,
            b => block_chars(b),
        })
        .sum();
    chars.div_ceil(4) as u32
}

fn block_chars(b: &ContentBlock) -> usize {
    match b {
        ContentBlock::Text { text, .. } | ContentBlock::Thinking { text, .. } => {
            text.chars().count()
        }
        ContentBlock::RedactedThinking { data } => data.chars().count(),
        ContentBlock::ToolUse { name, input, .. } => {
            name.chars().count() + count_value_chars(input)
        }
        ContentBlock::ToolResult { content, .. } => content.chars().count(),
        ContentBlock::Image { .. } => 4_800,
    }
}

/// `value`'s characters as its JSON would render, without building the string.
fn count_value_chars(value: &serde_json::Value) -> usize {
    struct CharCount(usize);
    impl std::fmt::Write for CharCount {
        fn write_str(&mut self, s: &str) -> std::fmt::Result {
            self.0 += s.chars().count();
            Ok(())
        }
    }
    let mut counter = CharCount(0);
    let _ = std::fmt::Write::write_fmt(&mut counter, format_args!("{value}"));
    counter.0
}

/// The real size of every message in `messages`, as the prompt of a request on `model` would carry
/// it: each measured turn ([`Message::prior_turn_tokens`]) exactly, split between its assistant
/// message (its [`message_tokens`], capped at the turn) and the user message after it (the rest);
/// every other message by [`message_tokens`].
///
/// A measured turn is used only while it still describes what the prompt carries: both its
/// messages come from the same model, that model is `model` (when given — a turn measured on
/// another tokenizer, or whose thinking a model switch downgraded to text, says nothing about this
/// one), and masking has not since replaced any of its tool output.
///
/// Sums over a range that starts and ends at an assistant boundary are exact wherever every turn in
/// it was measured; the split inside a turn is the one estimated part.
pub fn message_sizes(messages: &[Message], family: TokenFamily, model: Option<&str>) -> Vec<u32> {
    let mut sizes: Vec<u32> = messages.iter().map(|m| message_tokens(m, family)).collect();
    for i in 2..messages.len() {
        if let Some(turn) = measured_turn(messages, i, model) {
            let assistant = sizes[i - 2].min(turn);
            sizes[i - 2] = assistant;
            sizes[i - 1] = turn - assistant;
        }
    }
    sizes
}

fn measured_turn(messages: &[Message], i: usize, model: Option<&str>) -> Option<u32> {
    let m = &messages[i];
    let turn = m.prior_turn_tokens?;
    let (assistant, user) = (&messages[i - 2], &messages[i - 1]);
    let valid = m.role == Role::Assistant
        && assistant.role == Role::Assistant
        && user.role == Role::User
        && assistant.model_id == m.model_id
        && model.is_none_or(|model| m.model_id.as_deref() == Some(model))
        && !user.content.iter().any(|b| {
            matches!(b, ContentBlock::ToolResult { content, .. } if content.starts_with(MASKED_PREFIX))
        });
    valid.then_some(turn)
}

/// The sum of `sizes`, saturating.
pub fn total(sizes: &[u32]) -> u32 {
    sizes.iter().fold(0u32, |acc, &n| acc.saturating_add(n))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asst(model: &str, output: u32, reasoning: u32) -> Message {
        Message::assistant(vec![ContentBlock::text("ok")])
            .with_model_id(model)
            .with_usage(TokenUsage {
                output_tokens: output,
                reasoning_tokens: reasoning,
                ..Default::default()
            })
    }

    #[test]
    fn families_follow_the_model_id_through_vendor_and_region_prefixes() {
        for id in [
            "claude-haiku-5-5",
            "anthropic/claude-sonnet-5",
            "us.anthropic.claude-opus-5-5-v1:0",
            "CLAUDE-OPUS-4-8",
            "claude-fable-5-1",
        ] {
            assert_eq!(TokenFamily::for_model(id), TokenFamily::Claude, "{id}");
        }
        for id in ["gpt-6-luna", "openai/gpt-5.2", "gpt-6.1-sol"] {
            assert_eq!(TokenFamily::for_model(id), TokenFamily::Gpt, "{id}");
        }
        for id in ["gpt-oss-120b", "gemini-3-pro", "kimi-k3", ""] {
            assert_eq!(TokenFamily::for_model(id), TokenFamily::Other, "{id}");
        }
    }

    #[test]
    fn the_fallback_uses_each_familys_measured_rates() {
        // 1,000 chars of tool output: Claude 2.5 chars/token + 80 framing, GPT 3.5, Other chars/4.
        let result = Message::tool_result("t", "x".repeat(1_000), false);
        assert_eq!(message_tokens(&result, TokenFamily::Claude), 400 + 80);
        assert_eq!(message_tokens(&result, TokenFamily::Gpt), 286);
        assert_eq!(message_tokens(&result, TokenFamily::Other), 250);
        // Prose: Claude 2.3, GPT 4.0.
        assert_eq!(text_tokens(&"y".repeat(230), TokenFamily::Claude), 100);
        assert_eq!(text_tokens(&"y".repeat(400), TokenFamily::Gpt), 100);
        // An image is a flat 1,200 on every family.
        let shot = Message::user_with_images(
            "",
            vec![crate::message::ImageSource::base64("image/png", "AA")],
        );
        assert_eq!(message_tokens(&shot, TokenFamily::Claude), 1_200);
        assert_eq!(message_tokens(&shot, TokenFamily::Other), 1_200);
    }

    #[test]
    fn an_assistant_message_costs_its_recorded_output_not_its_stored_text() {
        // The stored text is two chars; the turn generated 56K tokens of (summarized) thinking,
        // all of which the prompt carries on Claude, and 30% of the reasoning on GPT.
        let m = asst("claude-haiku-5-5", 56_000, 55_000);
        assert_eq!(message_tokens(&m, TokenFamily::Claude), 56_000);
        assert_eq!(message_tokens(&m, TokenFamily::Gpt), 56_000 - 38_500);
        // Other keeps chars/4 for multi-message sizes.
        assert_eq!(message_tokens(&m, TokenFamily::Other), 1);
        // No usage (a synthetic error record): its characters.
        let bare = Message::assistant(vec![ContentBlock::text("z".repeat(23))]);
        assert_eq!(message_tokens(&bare, TokenFamily::Claude), 10);
    }

    #[test]
    fn a_measured_turn_replaces_the_estimate_exactly_and_splits_at_the_assistant() {
        let model = "claude-haiku-5-5";
        let mut second = asst(model, 10, 0);
        second.prior_turn_tokens = Some(5_000);
        let messages = vec![
            Message::user("go"),
            asst(model, 300, 0),
            Message::tool_result("t", "x".repeat(100), false),
            second,
        ];
        let sizes = message_sizes(&messages, TokenFamily::Claude, Some(model));
        assert_eq!(sizes[1], 300, "the assistant keeps its recorded output");
        assert_eq!(
            sizes[2], 4_700,
            "the user message gets the rest of the measured turn"
        );
        assert_eq!(sizes[1] + sizes[2], 5_000);
        // Another model's request (or a tokenizer we are not sizing for) does not count.
        let other = message_sizes(&messages, TokenFamily::Claude, Some("claude-sonnet-5-5"));
        assert_eq!(other[2], message_tokens(&messages[2], TokenFamily::Claude));
        // Nor does a turn whose tool output masking has since replaced.
        let mut masked = messages.clone();
        masked[2] = Message::tool_result(
            "t",
            format!("{MASKED_PREFIX} read(path=a), 100 chars.]"),
            false,
        );
        let sizes = message_sizes(&masked, TokenFamily::Claude, Some(model));
        assert_eq!(sizes[2], message_tokens(&masked[2], TokenFamily::Claude));
        // Nor one whose two assistant messages came from different models.
        let mut mixed = messages.clone();
        mixed[1].model_id = Some("gpt-6-luna".into());
        let sizes = message_sizes(&mixed, TokenFamily::Claude, None);
        assert_eq!(sizes[2], message_tokens(&mixed[2], TokenFamily::Claude));
    }

    #[test]
    fn preserved_defaults_scale_by_the_measured_ratios() {
        assert_eq!(TokenFamily::Claude.scale_span(22_500), 45_562);
        assert_eq!(TokenFamily::Gpt.scale_span(60_000), 66_000);
        assert_eq!(TokenFamily::Other.scale_span(22_500), 22_500);
        assert_eq!(TokenFamily::Claude.scale_tool(9_000), 14_400);
        assert_eq!(TokenFamily::Gpt.scale_tool(24_000), 27_428);
        assert_eq!(TokenFamily::Other.scale_tool(9_000), 9_000);
    }
}
