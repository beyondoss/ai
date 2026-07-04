//! Branch summarization — an LLM-generated recap of an abandoned tree branch's activity, generated
//! when navigating away from it (the storage-layer navigation primitive is
//! `crate::session_store::SessionStore::switch_active` in the `agent` crate; this module builds the
//! network-free summarization request for whatever calls it — the headless server's branch-navigation
//! RPC handler).
//!
//! This reuses [`crate::compaction`]'s `SUMMARY_SYSTEM`/`extract_file_ops` unchanged, and its
//! `render_prefix` in spirit but not verbatim (see
//! [`crate::compaction::render_prefix_without_tool_results`]) — a branch summary and a compaction
//! summary are the same kind of "condense this transcript into a structured recap" operation, just
//! triggered by different events (navigating away from a branch vs. the context window filling up),
//! framed by a different instruction, and — since a branch summary is read once on return rather than
//! carried forward as live context — rendered tersely enough to drop tool-result content entirely
//! rather than truncating it. No incremental-update path (unlike compaction's
//! [`crate::compaction::previous_summary`]): a branch is summarized once, when it's abandoned, not
//! repeatedly re-summarized forward.

use std::sync::Arc;

use crate::compaction::{
    SUMMARY_MARKER, SUMMARY_SYSTEM, estimate_message_tokens, extract_file_ops,
    render_prefix_without_tool_results,
};
use crate::message::{ContentBlock, Message};
use crate::transport::ModelRequest;

/// The branch-summarization call's output budget — a small fixed constant, unlike compaction's own
/// `CompactionConfig::summary_max_tokens` (scaled off `reserve_tokens`). Matches pi's own
/// `generateBranchSummary` (`branch-summarization.ts`), which hardcodes `maxTokens: 2048` for this call
/// independent of `reserveTokens`: a branch recap is inherently a short, single-shot summary (no
/// incremental-update path to budget for), so it doesn't need — and shouldn't scale with — the same
/// headroom-proportional budget a live compaction summary does.
pub const BRANCH_SUMMARY_MAX_TOKENS: u32 = 2_048;

/// The instruction appended after the rendered branch transcript. Matches pi's own
/// `BRANCH_SUMMARY_PROMPT` structurally — the same bracketed-exemplar/"(none)"-fallback/checkbox/
/// numbered-list granularity [`crate::compaction::SUMMARY_INSTRUCTION`]'s doc comment describes — so a
/// downstream reader sees a consistent shape whether a section came from compaction or branch
/// summarization. Deliberately has no `## Critical Context` section (pi's own `BRANCH_SUMMARY_PROMPT`
/// doesn't either) — an abandoned branch's recap is read once on return, not carried forward turn after
/// turn the way a live compaction summary is, so it doesn't need the same "what would a fresh context
/// need to keep working" section.
pub const BRANCH_SUMMARY_INSTRUCTION: &str = "Create a structured summary of this conversation branch \
for context when returning later.\n\nUse this EXACT format:\n\n## Goal\n[What was the user trying to \
accomplish in this branch?]\n\n## Constraints & Preferences\n- [Any constraints, preferences, or \
requirements mentioned]\n- [Or \"(none)\" if none were mentioned]\n\n## Progress\n### Done\n- [x] \
[Completed tasks/changes]\n\n### In Progress\n- [ ] [Work that was started but not finished]\n\n### \
Blocked\n- [Issues preventing progress, if any]\n\n## Key Decisions\n- **[Decision]**: [Brief \
rationale]\n\n## Next Steps\n1. [What should happen next to continue this work]\n\nKeep each section \
concise. Preserve exact file paths, function names, and error messages.";

/// Prefix marking a message as a branch summary, mirroring [`crate::compaction::SUMMARY_MARKER`] — an
/// explicit signal (for a human or a later, nested summarization) that this text recaps an abandoned
/// branch rather than being organic conversation content.
pub const BRANCH_SUMMARY_MARKER: &str = "[Explored a different branch before returning here]";

/// Whether `m` is itself a nested summary — a prior compaction or branch-summary message (tagged with
/// [`SUMMARY_MARKER`]/[`BRANCH_SUMMARY_MARKER`]) folded into this branch's own history.
fn is_nested_summary(m: &Message) -> bool {
    matches!(
        m.content.first(),
        Some(ContentBlock::Text { text, .. })
            if text.starts_with(SUMMARY_MARKER) || text.starts_with(BRANCH_SUMMARY_MARKER)
    )
}

/// Keep only as much of the *tail* of `messages` as fits within `budget_tokens`, walking newest-to-
/// oldest and stopping once the accumulated estimate would exceed it — pi's `prepareBranchEntries`
/// windowing, so an oversized abandoned branch can't blow out the summarization call's own context
/// window. Unlike compaction's `find_cut` (whose kept suffix re-enters the live conversation and so
/// must land on a clean role-alternation boundary), a branch summary's rendered transcript is a
/// one-off prompt discarded right after the call — no alternation constraint, so a plain budget walk
/// suffices. Always keeps at least the single most recent message, even alone over budget, so a
/// pathologically large final message doesn't window down to nothing.
///
/// A nested compaction/branch-summary message — already a dense recap of a lot of history, not raw
/// conversation — gets one exception: if it's the entry that would tip the budget over, it's kept
/// anyway as long as the accumulated tail is still under 90% of budget (matching pi's
/// `totalTokens < tokenBudget * 0.9`), rather than being dropped like an ordinary message would be.
/// Losing a compaction/branch-summary's condensed recap of everything *behind* it is a much bigger
/// loss of context per token than trimming a few more raw messages, so it's worth stretching for when
/// there's still reasonable room.
fn windowed_by_budget(messages: &[Message], budget_tokens: u32) -> &[Message] {
    let mut acc = 0u32;
    let mut start = messages.len();
    for (i, m) in messages.iter().enumerate().rev() {
        let cost = estimate_message_tokens(m);
        if start != messages.len() && acc.saturating_add(cost) > budget_tokens {
            if is_nested_summary(m) && acc < budget_tokens.saturating_mul(9) / 10 {
                start = i;
            }
            break;
        }
        acc = acc.saturating_add(cost);
        start = i;
    }
    &messages[start..]
}

/// Build the (network-free) branch-summarization request for `messages` — an abandoned branch's
/// materialized messages, root to its old tip (e.g. the slice `SessionStore::switch_active` returns,
/// captured *before* navigating away from it). [`extract_file_ops`] tags the files the *whole* branch
/// touched (cheap metadata, kept regardless of size); the rendered transcript itself is windowed to
/// `input_token_budget` (typically the model's `context_window - reserve_tokens`, mirroring pi) via
/// [`windowed_by_budget`], with a note when that actually dropped older activity — otherwise a long
/// branch could overflow the summarization model's own context window, the one failure mode this
/// windowing exists to prevent.
///
/// `custom_instructions`, when given, is appended after [`BRANCH_SUMMARY_INSTRUCTION`] as "Additional
/// focus: {custom_instructions}" — the same framing [`crate::compaction::summary_request`] uses for a
/// manual compaction's client-supplied steering, matching pi's own `generateSummary` — so a client
/// navigating away from a branch can steer what the recap emphasizes without replacing the structured
/// Markdown format itself. Set `replace_instructions` to use `custom_instructions` as the *entire*
/// instruction section instead — pi's own `replaceInstructions` option
/// (`branch-summarization.ts`'s `generateBranchSummary`: `if (replaceInstructions && customInstructions)
/// instructions = customInstructions`) — for a caller that wants full control over the summarization
/// task rather than steering the default one. Has no effect when `custom_instructions` is `None`: there's
/// nothing to replace the base instruction *with*, so it's used as-is either way, matching pi's own
/// `replaceInstructions && customInstructions` guard (a `true` with no instructions given falls through
/// to the plain-`BRANCH_SUMMARY_PROMPT` case, not an empty prompt).
pub fn branch_summary_request(
    model: &str,
    messages: &[Message],
    max_tokens: u32,
    input_token_budget: u32,
    custom_instructions: Option<&str>,
    replace_instructions: bool,
) -> ModelRequest {
    let (read, modified) = extract_file_ops(messages);
    let windowed = windowed_by_budget(messages, input_token_budget);

    let mut prompt = String::new();
    if windowed.len() < messages.len() {
        prompt.push_str(&format!(
            "[{} earlier message(s) from this branch omitted to fit the summarization budget]\n\n",
            messages.len() - windowed.len()
        ));
    }
    // `<conversation>` wraps the rendered transcript, matching pi's own `generateBranchSummary`
    // (`branch-summarization.ts`) and this crate's own compaction summary-request path (see
    // `crate::compaction::summary_request`'s fresh/split-turn path).
    prompt.push_str("<conversation>\n");
    prompt.push_str(&render_prefix_without_tool_results(windowed));
    prompt.push_str("\n</conversation>\n\n");
    if !read.is_empty() {
        prompt.push_str(&format!(
            "<read-files>\n{}\n</read-files>\n",
            read.join("\n")
        ));
    }
    if !modified.is_empty() {
        prompt.push_str(&format!(
            "<modified-files>\n{}\n</modified-files>\n",
            modified.join("\n")
        ));
    }
    // Matches pi's own `if (replaceInstructions && customInstructions) ... else if (customInstructions)
    // ... else ...`: replacing only actually applies once there's something to replace the base
    // instruction *with* — `replace_instructions` alone, with no custom instructions given, falls
    // through to the plain base-instruction case rather than an empty instruction section.
    match custom_instructions {
        Some(custom) if replace_instructions => prompt.push_str(custom),
        Some(custom) => {
            prompt.push_str(BRANCH_SUMMARY_INSTRUCTION);
            prompt.push_str("\n\nAdditional focus: ");
            prompt.push_str(custom);
        }
        None => prompt.push_str(BRANCH_SUMMARY_INSTRUCTION),
    }

    ModelRequest::new(model, Arc::new(vec![Message::user(prompt)]), max_tokens)
        .with_system(SUMMARY_SYSTEM)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::ContentBlock;
    use serde_json::json;

    fn branch() -> Vec<Message> {
        vec![
            Message::user("try approach X"),
            Message::assistant(vec![ContentBlock::tool_use(
                "1",
                "read",
                json!({ "path": "src/x.rs" }),
            )]),
            Message::tool_result("1", "fn x() {}", false),
            Message::assistant(vec![ContentBlock::tool_use(
                "2",
                "edit",
                json!({ "path": "src/x.rs" }),
            )]),
            Message::tool_result("2", "edited", false),
            Message::assistant(vec![ContentBlock::text("approach X didn't pan out")]),
        ]
    }

    #[test]
    fn branch_summary_max_tokens_is_a_small_fixed_constant_independent_of_reserve_tokens() {
        // pi-parity gap (fixed): pi's `generateBranchSummary` hardcodes `maxTokens: 2048` for this call,
        // independent of `reserveTokens` — unlike compaction's own scaled `summary_max_tokens`
        // (0.8 * reserve_tokens, which at typical defaults is an order of magnitude larger).
        assert_eq!(BRANCH_SUMMARY_MAX_TOKENS, 2_048);
    }

    #[test]
    fn branch_summary_instruction_gives_pi_parity_structural_guidance() {
        // Same pi-parity gap as `crate::compaction::SUMMARY_INSTRUCTION` — matches pi's own
        // `BRANCH_SUMMARY_PROMPT` (`branch-summarization.ts`) structurally.
        assert!(
            BRANCH_SUMMARY_INSTRUCTION.contains("- [x]")
                && BRANCH_SUMMARY_INSTRUCTION.contains("- [ ]"),
            "must use checkbox conventions for Done/In Progress: {BRANCH_SUMMARY_INSTRUCTION}"
        );
        assert!(
            BRANCH_SUMMARY_INSTRUCTION.contains("\"(none)\""),
            "must give an explicit \"(none)\" fallback: {BRANCH_SUMMARY_INSTRUCTION}"
        );
        assert!(
            BRANCH_SUMMARY_INSTRUCTION.contains("1. ["),
            "Next Steps must be a numbered (not bulleted) list: {BRANCH_SUMMARY_INSTRUCTION}"
        );
        assert!(
            BRANCH_SUMMARY_INSTRUCTION.contains("## Goal\n["),
            "must give a bracketed exemplar right after the Goal heading: {BRANCH_SUMMARY_INSTRUCTION}"
        );
    }

    #[test]
    fn branch_summary_request_renders_transcript_and_tags_files() {
        let req = branch_summary_request("claude-test", &branch(), 512, 100_000, None, false);
        let ContentBlock::Text { text, .. } = &req.messages[0].content[0] else {
            panic!("expected text");
        };
        assert!(text.contains("try approach X"));
        assert!(text.contains("<read-files>"));
        assert!(text.contains("src/x.rs"));
        assert!(text.contains("<modified-files>"));
        assert!(text.contains("## Goal"));
        assert!(text.contains("conversation branch for context when returning later"));
        assert_eq!(req.system.as_deref(), Some(SUMMARY_SYSTEM));
        assert_eq!(req.model, "claude-test");
        assert_eq!(req.max_tokens, 512);
    }

    #[test]
    fn branch_summary_request_wraps_the_transcript_in_conversation_tags() {
        let req = branch_summary_request("claude-test", &branch(), 512, 100_000, None, false);
        let ContentBlock::Text { text, .. } = &req.messages[0].content[0] else {
            panic!("expected text");
        };
        assert!(text.contains("<conversation>"));
        assert!(text.contains("</conversation>"));
        let open = text.find("<conversation>").unwrap();
        let close = text.find("</conversation>").unwrap();
        assert!(
            text[open..close].contains("try approach X"),
            "the rendered transcript must live inside the conversation tags: {text}"
        );
    }

    #[test]
    fn branch_summary_request_excludes_tool_result_content_from_the_rendered_transcript() {
        // pi-parity gap (fixed): pi's `getMessageFromEntry` (branch-summarization.ts) drops every
        // tool-result-role entry entirely, so a branch summary's rendered transcript never includes
        // tool output — only user/assistant text, thinking, and tool-call names.
        let messages = vec![
            Message::user("try approach X"),
            Message::assistant(vec![ContentBlock::tool_use(
                "1",
                "read",
                json!({ "path": "src/x.rs" }),
            )]),
            Message::tool_result("1", "SECRET FILE CONTENTS should not appear", false),
            Message::assistant(vec![ContentBlock::text("approach X didn't pan out")]),
        ];
        let req = branch_summary_request("claude-test", &messages, 512, 100_000, None, false);
        let ContentBlock::Text { text, .. } = &req.messages[0].content[0] else {
            panic!("expected text");
        };
        assert!(text.contains("try approach X"));
        assert!(text.contains("Assistant called tool `read`"));
        assert!(
            !text.contains("SECRET FILE CONTENTS"),
            "tool result content must not survive into the rendered transcript: {text}"
        );
    }

    #[test]
    fn windowed_by_budget_keeps_only_the_newest_tail_within_budget() {
        // Each message here costs a few tokens (estimate_tokens is chars/4); a tiny budget should keep
        // only the last one or two, never the whole branch.
        let messages = branch();
        let windowed = windowed_by_budget(&messages, 1);
        assert!(
            windowed.len() < messages.len(),
            "must actually drop older messages"
        );
        assert_eq!(
            windowed.last().unwrap().content,
            messages.last().unwrap().content,
            "the most recent message must always survive"
        );
    }

    #[test]
    fn windowed_by_budget_never_drops_to_empty_on_one_oversized_message() {
        let messages = vec![Message::user("x".repeat(10_000))];
        let windowed = windowed_by_budget(&messages, 1);
        assert_eq!(
            windowed.len(),
            1,
            "a single message, however large, must survive"
        );
    }

    #[test]
    fn windowed_by_budget_keeps_everything_when_the_whole_branch_fits() {
        let messages = branch();
        let windowed = windowed_by_budget(&messages, 1_000_000);
        assert_eq!(windowed.len(), messages.len());
    }

    #[test]
    fn windowed_by_budget_privileges_a_nested_compaction_summary_over_the_ordinary_cutoff() {
        // The recent tail alone (1 token) leaves plenty of room under the budget (10 tokens, so the
        // 90% privilege threshold is 9) — pi keeps a nested summary in that case even though it, on
        // its own, blows well past the remaining budget: losing its condensed recap of everything
        // *behind* it is a bigger loss than trimming a bit more of the raw tail.
        let messages = vec![
            Message::user(format!("{SUMMARY_MARKER}\n\n{}", "x".repeat(400))),
            Message::user("abcd"),
        ];
        let windowed = windowed_by_budget(&messages, 10);
        assert_eq!(
            windowed.len(),
            2,
            "the nested compaction summary must survive alongside the recent tail"
        );
    }

    #[test]
    fn windowed_by_budget_does_not_privilege_an_ordinary_oversized_message() {
        // Same shape as the privileged case, but the oversized old message carries no summary marker
        // — it's dropped like any other message that would blow the budget.
        let messages = vec![Message::user("y".repeat(400)), Message::user("abcd")];
        let windowed = windowed_by_budget(&messages, 10);
        assert_eq!(
            windowed.len(),
            1,
            "an ordinary oversized message must be dropped"
        );
    }

    #[test]
    fn windowed_by_budget_still_drops_a_nested_summary_once_the_tail_already_fills_the_budget() {
        // The privilege only applies with "reasonable room" left (< 90% of budget already spent).
        // Two recent messages already accumulate to the full 10-token budget, so a nested summary
        // behind them gets dropped exactly like the unprivileged case above.
        let messages = vec![
            Message::user(format!("{SUMMARY_MARKER}\n\n{}", "x".repeat(400))),
            Message::user("a".repeat(20)), // 5 tokens
            Message::user("b".repeat(20)), // 5 tokens
        ];
        let windowed = windowed_by_budget(&messages, 10);
        assert_eq!(
            windowed.len(),
            2,
            "no room left for the privilege exception once the tail already spends ~all the budget"
        );
    }

    #[test]
    fn branch_summary_request_notes_when_windowing_dropped_older_activity() {
        let req = branch_summary_request("claude-test", &branch(), 512, 1, None, false);
        let ContentBlock::Text { text, .. } = &req.messages[0].content[0] else {
            panic!("expected text");
        };
        assert!(
            text.contains("omitted to fit the summarization budget"),
            "must disclose that older branch activity was windowed out: {text}"
        );
    }

    #[test]
    fn branch_summary_request_extracts_file_ops_from_the_full_branch_even_when_windowed() {
        // File awareness is cheap metadata and should span the whole branch — a tiny budget that
        // windows out the early `read`/`edit` calls from the rendered transcript must still surface
        // them in `<read-files>`/`<modified-files>`.
        let req = branch_summary_request("claude-test", &branch(), 512, 1, None, false);
        let ContentBlock::Text { text, .. } = &req.messages[0].content[0] else {
            panic!("expected text");
        };
        assert!(text.contains("<read-files>"));
        assert!(text.contains("src/x.rs"));
        assert!(text.contains("<modified-files>"));
    }

    #[test]
    fn branch_summary_request_omits_file_tags_when_no_tool_calls() {
        let messages = vec![
            Message::user("just chatting"),
            Message::assistant(vec![ContentBlock::text("sure")]),
        ];
        let req = branch_summary_request("claude-test", &messages, 512, 100_000, None, false);
        let ContentBlock::Text { text, .. } = &req.messages[0].content[0] else {
            panic!("expected text");
        };
        assert!(!text.contains("<read-files>"));
        assert!(!text.contains("<modified-files>"));
    }

    #[test]
    fn branch_summary_request_appends_custom_instructions_as_additional_focus() {
        // Matches pi's own `generateSummary`/`compaction::summary_request`: "Additional focus:
        // {customInstructions}" appended after the structured instruction template, not replacing it.
        let req = branch_summary_request(
            "claude-test",
            &branch(),
            512,
            100_000,
            Some("keep every detail about the auth refactor"),
            false,
        );
        let ContentBlock::Text { text, .. } = &req.messages[0].content[0] else {
            panic!("expected text");
        };
        assert!(
            text.contains("conversation branch for context when returning later"),
            "the structured template survives: {text}"
        );
        assert!(
            text.contains("Additional focus: keep every detail about the auth refactor"),
            "the custom instructions must be appended: {text}"
        );
    }

    #[test]
    fn branch_summary_request_omits_additional_focus_when_no_custom_instructions_given() {
        let req = branch_summary_request("claude-test", &branch(), 512, 100_000, None, false);
        let ContentBlock::Text { text, .. } = &req.messages[0].content[0] else {
            panic!("expected text");
        };
        assert!(!text.contains("Additional focus"));
    }

    #[test]
    fn branch_summary_request_replace_instructions_uses_custom_instructions_as_the_whole_prompt() {
        // pi-parity gap (task 44): pi's `replaceInstructions` option lets custom instructions fully
        // replace `BRANCH_SUMMARY_PROMPT` instead of being appended after it — for a caller that wants
        // full control over the summarization task, not just to steer the default one.
        let req = branch_summary_request(
            "claude-test",
            &branch(),
            512,
            100_000,
            Some("Summarize only the auth-related changes, in one sentence."),
            true,
        );
        let ContentBlock::Text { text, .. } = &req.messages[0].content[0] else {
            panic!("expected text");
        };
        assert!(
            text.contains("Summarize only the auth-related changes, in one sentence."),
            "the custom instructions must be present: {text}"
        );
        assert!(
            !text.contains("conversation branch for context when returning later"),
            "the base BRANCH_SUMMARY_INSTRUCTION must be replaced, not appended alongside: {text}"
        );
        assert!(
            !text.contains("Additional focus"),
            "replace mode doesn't use the additive framing at all: {text}"
        );
        // The rendered transcript and file tags are still built the normal way — only the instruction
        // section itself is replaced.
        assert!(text.contains("try approach X"));
        assert!(text.contains("<read-files>"));
    }

    #[test]
    fn replace_instructions_without_custom_instructions_falls_back_to_the_base_prompt() {
        // `replace_instructions: true` with no `custom_instructions` has nothing to replace the base
        // prompt with — matches pi's own `replaceInstructions && customInstructions` guard, which falls
        // through to plain `BRANCH_SUMMARY_PROMPT` rather than an empty instruction section.
        let req = branch_summary_request("claude-test", &branch(), 512, 100_000, None, true);
        let ContentBlock::Text { text, .. } = &req.messages[0].content[0] else {
            panic!("expected text");
        };
        assert!(
            text.contains("conversation branch for context when returning later"),
            "must fall back to the base instruction: {text}"
        );
    }
}
