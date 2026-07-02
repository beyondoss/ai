//! Branch summarization — an LLM-generated recap of an abandoned tree branch's activity, generated
//! when navigating away from it (the storage-layer navigation primitive is
//! `crate::session_store::SessionStore::switch_active` in the `agent` crate; this module builds the
//! network-free summarization request for whatever calls it — the headless server's branch-navigation
//! RPC handler).
//!
//! This reuses [`crate::compaction`]'s `render_prefix`/`SUMMARY_SYSTEM`/`extract_file_ops` unchanged —
//! a branch summary and a compaction summary are the same kind of "condense this transcript into a
//! structured recap" operation, just triggered by different events (navigating away from a branch vs.
//! the context window filling up) and framed by a different instruction. No incremental-update path
//! (unlike compaction's [`crate::compaction::previous_summary`]): a branch is summarized once, when
//! it's abandoned, not repeatedly re-summarized forward.

use std::sync::Arc;

use crate::compaction::{
    SUMMARY_MARKER, SUMMARY_SYSTEM, estimate_message_tokens, extract_file_ops, render_prefix,
};
use crate::message::{ContentBlock, Message};
use crate::transport::ModelRequest;

/// The instruction appended after the rendered branch transcript. Adapted from pi's
/// `BRANCH_SUMMARY_PROMPT` to this project's existing Markdown-heading convention (matching
/// [`crate::compaction::SUMMARY_INSTRUCTION`]'s structure), so a downstream reader sees a consistent
/// shape whether a section came from compaction or branch summarization.
pub const BRANCH_SUMMARY_INSTRUCTION: &str = "The user explored a different conversation branch \
before returning here. Summarize that branch into this exact Markdown structure, keeping concrete \
identifiers (file paths, function names, commands) verbatim and omitting pleasantries:\n\n## Goal\n\
## Constraints\n## Progress\n### Done\n### In Progress\n### Blocked\n## Key Decisions\n## Next Steps";

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
/// Markdown format itself.
pub fn branch_summary_request(
    model: &str,
    messages: &[Message],
    max_tokens: u32,
    input_token_budget: u32,
    custom_instructions: Option<&str>,
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
    prompt.push_str(&render_prefix(windowed));
    prompt.push_str("\n\n");
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
    prompt.push_str(BRANCH_SUMMARY_INSTRUCTION);
    if let Some(custom) = custom_instructions {
        prompt.push_str("\n\nAdditional focus: ");
        prompt.push_str(custom);
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
    fn branch_summary_request_renders_transcript_and_tags_files() {
        let req = branch_summary_request("claude-test", &branch(), 512, 100_000, None);
        let ContentBlock::Text { text, .. } = &req.messages[0].content[0] else {
            panic!("expected text");
        };
        assert!(text.contains("try approach X"));
        assert!(text.contains("<read-files>"));
        assert!(text.contains("src/x.rs"));
        assert!(text.contains("<modified-files>"));
        assert!(text.contains("## Goal"));
        assert!(text.contains("different conversation branch"));
        assert_eq!(req.system.as_deref(), Some(SUMMARY_SYSTEM));
        assert_eq!(req.model, "claude-test");
        assert_eq!(req.max_tokens, 512);
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
        let req = branch_summary_request("claude-test", &branch(), 512, 1, None);
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
        let req = branch_summary_request("claude-test", &branch(), 512, 1, None);
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
        let req = branch_summary_request("claude-test", &messages, 512, 100_000, None);
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
        );
        let ContentBlock::Text { text, .. } = &req.messages[0].content[0] else {
            panic!("expected text");
        };
        assert!(
            text.contains("different conversation branch"),
            "the structured template survives: {text}"
        );
        assert!(
            text.contains("Additional focus: keep every detail about the auth refactor"),
            "the custom instructions must be appended: {text}"
        );
    }

    #[test]
    fn branch_summary_request_omits_additional_focus_when_no_custom_instructions_given() {
        let req = branch_summary_request("claude-test", &branch(), 512, 100_000, None);
        let ContentBlock::Text { text, .. } = &req.messages[0].content[0] else {
            panic!("expected text");
        };
        assert!(!text.contains("Additional focus"));
    }
}
