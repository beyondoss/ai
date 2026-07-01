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

use crate::compaction::{SUMMARY_SYSTEM, extract_file_ops, render_prefix};
use crate::message::Message;
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

/// Build the (network-free) branch-summarization request for `messages` — an abandoned branch's
/// materialized messages, root to its old tip (e.g. the slice `SessionStore::switch_active` returns,
/// captured *before* navigating away from it). Reuses [`render_prefix`] to render the transcript and
/// [`extract_file_ops`] to tag the files the branch touched — exactly like a compaction summary, since
/// both operations condense a transcript the same way for a different trigger.
pub fn branch_summary_request(model: &str, messages: &[Message], max_tokens: u32) -> ModelRequest {
    let (read, modified) = extract_file_ops(messages);

    let mut prompt = String::new();
    prompt.push_str(&render_prefix(messages));
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
            Message::assistant(vec![ContentBlock::ToolUse {
                id: "1".into(),
                name: "read".into(),
                input: json!({ "path": "src/x.rs" }),
            }]),
            Message::tool_result("1", "fn x() {}", false),
            Message::assistant(vec![ContentBlock::ToolUse {
                id: "2".into(),
                name: "edit".into(),
                input: json!({ "path": "src/x.rs" }),
            }]),
            Message::tool_result("2", "edited", false),
            Message::assistant(vec![ContentBlock::Text {
                text: "approach X didn't pan out".into(),
            }]),
        ]
    }

    #[test]
    fn branch_summary_request_renders_transcript_and_tags_files() {
        let req = branch_summary_request("claude-test", &branch(), 512);
        let ContentBlock::Text { text } = &req.messages[0].content[0] else {
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
    fn branch_summary_request_omits_file_tags_when_no_tool_calls() {
        let messages = vec![
            Message::user("just chatting"),
            Message::assistant(vec![ContentBlock::Text {
                text: "sure".into(),
            }]),
        ];
        let req = branch_summary_request("claude-test", &messages, 512);
        let ContentBlock::Text { text } = &req.messages[0].content[0] else {
            panic!("expected text");
        };
        assert!(!text.contains("<read-files>"));
        assert!(!text.contains("<modified-files>"));
    }
}
