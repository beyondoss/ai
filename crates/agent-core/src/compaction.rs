//! Context compaction — keeping a long run under the model's context window.
//!
//! A loop re-sends an ever-growing transcript every turn. Left unbounded it eventually exceeds the
//! model's context window and every further request fails — the session effectively dies. Compaction
//! replaces the older prefix of the conversation with an LLM-generated summary while keeping the most
//! recent turns verbatim, so the run can continue indefinitely.
//!
//! This module is the pure, network-free half: the trigger ([`should_compact`]), the cut-point
//! search ([`find_cut`]), the summary-prompt construction ([`render_prefix`]/[`summary_request`]),
//! and file-op extraction. [`Agent`](crate::Agent) owns the one piece that needs the network — making
//! the summarization model call — and stitches the result back into the [`Session`].

use std::sync::Arc;

use crate::message::{ContentBlock, Message, Role};
use crate::session::Session;
use crate::transport::ModelRequest;

/// Tunables for automatic compaction.
#[derive(Debug, Clone, Copy)]
pub struct CompactionConfig {
    /// The model's context window, in tokens. The trigger compares the live prompt size against it.
    pub context_window: u32,
    /// Headroom left below the window: compaction fires once the prompt crosses
    /// `context_window - reserve_tokens`, leaving room for the next turn's output + slack.
    pub reserve_tokens: u32,
    /// Roughly how many tokens of recent conversation to keep verbatim after a compaction.
    pub keep_recent_tokens: u32,
    /// Output ceiling for the summarization call itself.
    pub summary_max_tokens: u32,
    /// Whether automatic (threshold-triggered) compaction is on. Manual/overflow compaction ignores it.
    pub enabled: bool,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        // Defaults sized for a 200k-context Claude model; override per deployment.
        Self {
            context_window: 200_000,
            reserve_tokens: 24_000,
            keep_recent_tokens: 40_000,
            summary_max_tokens: 4_096,
            enabled: true,
        }
    }
}

/// System prompt for the summarization call.
pub const SUMMARY_SYSTEM: &str = "You compact a long agent transcript so the agent can keep working \
with far fewer tokens but no loss of essential context. Be precise, concrete, and information-dense; \
preserve file paths, identifiers, commands, and decisions exactly.";

/// The instruction appended after the rendered transcript in the summarization call.
pub const SUMMARY_INSTRUCTION: &str = "Summarize the conversation above into this exact Markdown \
structure, keeping concrete identifiers (file paths, function names, commands) verbatim and omitting \
pleasantries:\n\n## Goal\n## Constraints\n## Progress\n### Done\n### In Progress\n### Blocked\n## Key \
Decisions\n## Next Steps\n## Critical Context";

/// Tool-result content is truncated to this many characters when rendered into the summary prompt —
/// tool output is usually the bulk of the transcript and the least useful to re-summarize in full.
const TOOL_RESULT_MAX_CHARS: usize = 2_000;

/// Estimate a text's token count. The cheap, dependency-free char/4 heuristic — good enough for the
/// trigger and the cut-point search, which only need to be in the right ballpark.
pub fn estimate_tokens(text: &str) -> u32 {
    (text.chars().count() / 4) as u32
}

/// Estimate one message's token cost from its blocks.
fn estimate_message_tokens(m: &Message) -> u32 {
    m.content
        .iter()
        .map(|b| match b {
            ContentBlock::Text { text } => estimate_tokens(text),
            ContentBlock::Thinking { text, .. } => estimate_tokens(text),
            ContentBlock::RedactedThinking { data } => estimate_tokens(data),
            ContentBlock::ToolUse { name, input, .. } => {
                estimate_tokens(name) + estimate_tokens(&input.to_string())
            }
            ContentBlock::ToolResult { content, .. } => estimate_tokens(content),
            // A flat ~1.2k-token estimate for an image (Anthropic bills vision roughly there); the
            // exact figure depends on resolution, but the trigger only needs the right ballpark.
            ContentBlock::Image { .. } => 1_200,
        })
        .sum()
}

/// Whether the live prompt has crossed the compaction threshold. Uses the *last turn's* input size
/// ([`Session::last_input_tokens`]) — the real, current context size — not the cumulative totals.
pub fn should_compact(session: &Session, cfg: &CompactionConfig) -> bool {
    session.last_input_tokens > 0
        && session.last_input_tokens >= cfg.context_window.saturating_sub(cfg.reserve_tokens)
}

/// Choose the first message index to keep verbatim. Walks back from the end accumulating estimated
/// tokens until ~`keep_recent_tokens` is covered, then snaps **back** to the nearest assistant
/// message so the post-compaction history is `[summary(user), assistant, user, …]` — valid
/// alternation, and never cutting between an assistant's `tool_use` and its `tool_result`.
///
/// Returns `None` when there isn't a worthwhile prefix to summarize (too short, or no clean
/// boundary) — the caller then leaves the conversation untouched.
pub fn find_cut(messages: &[Message], keep_recent_tokens: u32) -> Option<usize> {
    let n = messages.len();
    // Need at least a couple of turns to bother — a summary plus a kept suffix.
    if n < 4 {
        return None;
    }
    let mut kept = 0u32;
    let mut idx = n - 1;
    while idx > 0 {
        kept = kept.saturating_add(estimate_message_tokens(&messages[idx]));
        if kept >= keep_recent_tokens {
            break;
        }
        idx -= 1;
    }
    // Snap back to the nearest assistant message — the suffix must start with one so it follows the
    // summary (a user message) cleanly, and so a kept assistant `tool_use` keeps its `tool_result`.
    let mut first_kept = idx;
    while first_kept > 0 && messages[first_kept].role != Role::Assistant {
        first_kept -= 1;
    }
    // Leave a non-trivial prefix to summarize, and require a real assistant boundary.
    if first_kept == 0 || messages[first_kept].role != Role::Assistant {
        return None;
    }
    Some(first_kept)
}

/// Render the prefix `messages` into a plain-text transcript for the summarization prompt. Tool
/// results — usually the bulk — are truncated; thinking blocks are dropped (not worth re-summarizing).
pub fn render_prefix(messages: &[Message]) -> String {
    let mut out = String::new();
    for m in messages {
        for b in &m.content {
            match b {
                ContentBlock::Text { text } => {
                    let who = match m.role {
                        Role::User => "User",
                        Role::Assistant => "Assistant",
                        Role::System => "System",
                    };
                    out.push_str(who);
                    out.push_str(": ");
                    out.push_str(text);
                    out.push('\n');
                }
                ContentBlock::ToolUse { name, input, .. } => {
                    out.push_str(&format!("Assistant called tool `{name}`({input})\n"));
                }
                ContentBlock::ToolResult {
                    content, is_error, ..
                } => {
                    let body = truncate_chars(content, TOOL_RESULT_MAX_CHARS);
                    let tag = if *is_error {
                        "tool error"
                    } else {
                        "tool result"
                    };
                    out.push_str(&format!("[{tag}] {body}\n"));
                }
                ContentBlock::Image { .. } => out.push_str("[image]\n"),
                // Thinking is internal scratch — omit from the summary input.
                ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. } => {}
            }
        }
    }
    out
}

/// Truncate a string to at most `max` characters, appending a marker that notes the elision.
fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let kept: String = s.chars().take(max).collect();
    let dropped = s.chars().count() - max;
    format!("{kept}… [{dropped} more characters truncated]")
}

/// Files the prefix read vs. modified, extracted from `read`/`write`/`edit` tool calls. Appended to
/// the summary so the model keeps file awareness across the cut.
pub fn extract_file_ops(messages: &[Message]) -> (Vec<String>, Vec<String>) {
    let mut read = Vec::new();
    let mut modified = Vec::new();
    for m in messages {
        for b in &m.content {
            if let ContentBlock::ToolUse { name, input, .. } = b {
                let Some(path) = input.get("path").and_then(|p| p.as_str()) else {
                    continue;
                };
                let bucket = match name.as_str() {
                    "read" | "ls" => &mut read,
                    "write" | "edit" => &mut modified,
                    _ => continue,
                };
                let path = path.to_string();
                if !bucket.contains(&path) {
                    bucket.push(path);
                }
            }
        }
    }
    (read, modified)
}

/// Build the (network-free) summarization request: the rendered prefix plus the structured
/// instruction and file-op tags, as a single user turn. One self-contained message sidesteps any
/// role-alternation constraints, and capping tool results keeps the prompt well under the window.
pub fn summary_request(model: &str, prefix: &[Message], max_tokens: u32) -> ModelRequest {
    let transcript = render_prefix(prefix);
    let (read, modified) = extract_file_ops(prefix);

    let mut prompt = String::new();
    prompt.push_str(&transcript);
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
    prompt.push_str(SUMMARY_INSTRUCTION);

    ModelRequest::new(model, Arc::new(vec![Message::user(prompt)]), max_tokens)
        .with_system(SUMMARY_SYSTEM)
}

/// Splice a summary into `session`, replacing the prefix before `first_kept` with one summary user
/// message. Resets [`Session::last_input_tokens`] so the freshly-shrunk context doesn't immediately
/// re-trigger the threshold (the true size is recomputed after the next turn).
pub fn apply_summary(session: &mut Session, first_kept: usize, summary: &str) {
    let kept = &session.messages[first_kept..];
    let mut new_messages = Vec::with_capacity(1 + kept.len());
    new_messages.push(Message::user(format!(
        "[Earlier conversation compacted to save context]\n\n{summary}"
    )));
    new_messages.extend_from_slice(kept);
    session.messages = Arc::new(new_messages);
    session.last_input_tokens = 0;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn convo() -> Vec<Message> {
        vec![
            Message::user("the original task: refactor foo"),
            Message::assistant(vec![ContentBlock::ToolUse {
                id: "1".into(),
                name: "read".into(),
                input: json!({ "path": "src/foo.rs" }),
            }]),
            Message::tool_result("1", "fn foo() {}", false),
            Message::assistant(vec![ContentBlock::ToolUse {
                id: "2".into(),
                name: "edit".into(),
                input: json!({ "path": "src/foo.rs" }),
            }]),
            Message::tool_result("2", "edited", false),
            Message::assistant(vec![ContentBlock::Text {
                text: "done".into(),
            }]),
        ]
    }

    #[test]
    fn should_compact_tracks_last_input_against_window() {
        let cfg = CompactionConfig {
            context_window: 1000,
            reserve_tokens: 200,
            ..Default::default()
        };
        let mut s = Session::new();
        assert!(!should_compact(&s, &cfg)); // no usage yet
        s.last_input_tokens = 700;
        assert!(!should_compact(&s, &cfg)); // below 1000-200
        s.last_input_tokens = 850;
        assert!(should_compact(&s, &cfg));
    }

    #[test]
    fn find_cut_snaps_to_assistant_and_leaves_both_sides() {
        let messages = convo();
        // Keep a tiny budget so the cut lands late, then snaps to an assistant boundary.
        let cut = find_cut(&messages, 1).expect("a cut");
        assert_eq!(messages[cut].role, Role::Assistant);
        assert!(cut > 0 && cut < messages.len());
    }

    #[test]
    fn find_cut_declines_short_conversations() {
        assert!(find_cut(&convo()[..3], 1).is_none());
    }

    #[test]
    fn extract_file_ops_buckets_reads_and_writes() {
        let (read, modified) = extract_file_ops(&convo());
        assert_eq!(read, vec!["src/foo.rs"]);
        assert_eq!(modified, vec!["src/foo.rs"]);
    }

    #[test]
    fn summary_request_truncates_tool_results_and_tags_files() {
        let big = "x".repeat(5000);
        let messages = vec![
            Message::user("task"),
            Message::assistant(vec![ContentBlock::ToolUse {
                id: "1".into(),
                name: "read".into(),
                input: json!({ "path": "a.rs" }),
            }]),
            Message::tool_result("1", &big, false),
        ];
        let req = summary_request("claude-test", &messages, 512);
        let text = match &req.messages[0].content[0] {
            ContentBlock::Text { text } => text,
            other => panic!("expected text, got {other:?}"),
        };
        assert!(text.contains("characters truncated"));
        assert!(text.contains("<read-files>"));
        assert!(text.contains("a.rs"));
        assert!(text.contains("## Goal"));
        assert_eq!(req.system.as_deref(), Some(SUMMARY_SYSTEM));
    }

    #[test]
    fn apply_summary_replaces_prefix_and_resets_size() {
        let mut s = Session::new();
        s.messages = Arc::new(convo());
        s.last_input_tokens = 9999;
        let n_before = s.messages.len();
        apply_summary(&mut s, 3, "SUMMARY");

        // summary(user) + kept suffix [3..]
        assert_eq!(s.messages.len(), 1 + (n_before - 3));
        assert_eq!(s.messages[0].role, Role::User);
        assert!(matches!(
            &s.messages[0].content[0],
            ContentBlock::Text { text } if text.contains("SUMMARY")
        ));
        // Alternation: summary(user) then the kept assistant message.
        assert_eq!(s.messages[1].role, Role::Assistant);
        assert_eq!(s.last_input_tokens, 0);
    }
}
