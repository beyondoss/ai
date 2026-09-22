//! A short human-readable name for a session, generated from its opening exchange.
//!
//! **Why this is the agent's job and not the control plane's.** A title is derived from conversation
//! content, and in a fleet the replica is the only component that holds both the transcript and the
//! tenant's key at the same time. Everything else about a session catalog — filtering, sorting,
//! paging, facets — belongs on the other side of the lifecycle contract; a title does not, because
//! nothing over there can read the session to write one.
//!
//! **Why it is worth a model call.** The title is the highest-priority field
//! `session_store::search_rank` checks and, until this existed, it was `None` for every session that
//! nobody had explicitly named — so the thing ranked first was empty almost always. It is also what
//! the rest of the field leans on: ChatGPT's conversation search matches mostly on titles, and Devin
//! added an explicit rename command to make session histories findable.
//!
//! A title is generated **once**, from the first exchange, and never regenerated. A session is about
//! what it was opened for; re-titling it later would make the same session change names underneath
//! whoever was looking for it.

use std::sync::Arc;

use crate::message::{ContentBlock, Message, Role};
use crate::transport::ModelRequest;

/// Output budget. A title is a handful of words, and the cap is what stops a model that decided to
/// explain itself from being charged for a paragraph.
pub const SESSION_TITLE_MAX_TOKENS: u32 = 32;

/// Longest title kept. Past this the model has misunderstood the instruction rather than produced a
/// long title, so the result is discarded rather than truncated into a broken phrase.
pub const SESSION_TITLE_MAX_CHARS: usize = 60;

/// How much of the opening exchange is shown. Enough to see what was asked, short enough that the
/// call stays cheap on a session whose first message is an entire pasted file.
const TITLE_INPUT_MAX_CHARS: usize = 2_000;

/// Framed to suppress the two things a chat model does by default here: answering the question, and
/// prefacing the answer. Anything but the bare title is unusable, because it goes straight into a
/// listing.
pub const SESSION_TITLE_SYSTEM: &str = "You write short titles for saved conversations. Reply with \
the title and nothing else — no quotes, no trailing period, no preamble, and never an answer to the \
conversation's question. Use three to six words in sentence case naming the concrete subject, like \
\"Fixing the NFS lease timeout\" or \"Refactoring the billing webhook\". If the opening is too vague \
to name a subject, reply with exactly: none";

/// The sentinel the model is told to use when the opening names no subject. Matched case-insensitively
/// by [`clean_title`], which then yields `None` — better an untitled session than one called "none".
const NO_TITLE: &str = "none";

/// Build the title request from a session's opening messages.
///
/// Only text blocks, and only the first user turn plus the first assistant reply: tool calls and
/// their results say what the agent *did*, which is a poor description of what the session is
/// *about*, and they are the bulk of the tokens.
pub fn session_title_request(model: &str, messages: &[Message]) -> ModelRequest {
    let mut prompt = String::from("<conversation>\n");
    let mut budget = TITLE_INPUT_MAX_CHARS;
    for msg in messages
        .iter()
        .filter(|m| matches!(m.role, Role::User | Role::Assistant))
    {
        let text: String = msg
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text, .. } => Some(&**text),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" ");
        if text.trim().is_empty() {
            continue;
        }
        let role = if msg.role == Role::User {
            "user"
        } else {
            "assistant"
        };
        let taken: String = text.chars().take(budget).collect();
        budget -= taken.chars().count();
        prompt.push_str(role);
        prompt.push_str(": ");
        prompt.push_str(&taken);
        prompt.push('\n');
        if budget == 0 {
            break;
        }
    }
    prompt.push_str("</conversation>\n\nTitle:");

    ModelRequest::new(
        model,
        Arc::new(vec![Message::user(prompt)]),
        SESSION_TITLE_MAX_TOKENS,
    )
    .with_system(SESSION_TITLE_SYSTEM)
}

/// Normalize whatever the model returned into a title worth storing, or `None`.
///
/// Rejects rather than repairs. A title is decoration on a listing: a wrong one is worse than none,
/// because it is what a person scans for and what `search_rank` weighs most heavily.
pub fn clean_title(raw: &str) -> Option<String> {
    let mut t = raw.trim();
    // A model that ignored "no preamble" usually produces exactly one line that is the title, after
    // one that is not. Take the last non-empty line rather than the first.
    if let Some(last) = t.lines().map(str::trim).rfind(|l| !l.is_empty()) {
        t = last;
    }
    // Surrounding quotes are the single most common deviation, and the only one worth repairing
    // because it leaves the title itself intact.
    let t = t
        .trim_matches(|c| c == '"' || c == '\'' || c == '`')
        .trim_end_matches('.')
        .trim();
    if t.is_empty() || t.eq_ignore_ascii_case(NO_TITLE) {
        return None;
    }
    if t.chars().count() > SESSION_TITLE_MAX_CHARS || t.contains('\n') {
        return None;
    }
    Some(t.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plain_title_survives_untouched() {
        assert_eq!(
            clean_title("Fixing the NFS lease timeout"),
            Some("Fixing the NFS lease timeout".to_string())
        );
    }

    #[test]
    fn the_decorations_a_model_adds_are_stripped() {
        for raw in [
            "\"Fixing the NFS lease timeout\"",
            "'Fixing the NFS lease timeout'",
            "`Fixing the NFS lease timeout`",
            "Fixing the NFS lease timeout.",
            "  Fixing the NFS lease timeout  ",
        ] {
            assert_eq!(
                clean_title(raw).as_deref(),
                Some("Fixing the NFS lease timeout"),
                "{raw:?}"
            );
        }
    }

    #[test]
    fn a_preamble_line_is_dropped_in_favour_of_the_title_line() {
        // The failure this guards is a listing full of "Sure, here's a title:".
        assert_eq!(
            clean_title("Sure, here's a title:\nFixing the NFS lease timeout"),
            Some("Fixing the NFS lease timeout".to_string())
        );
    }

    #[test]
    fn the_no_subject_sentinel_yields_no_title() {
        for raw in ["none", "None", "  NONE  ", "\"none\""] {
            assert_eq!(clean_title(raw), None, "{raw:?}");
        }
    }

    #[test]
    fn an_overlong_answer_is_rejected_rather_than_truncated() {
        // A model that answered the question instead of naming it produces prose. Half a sentence is
        // a worse title than no title, so this rejects instead of cutting.
        let prose = "x".repeat(SESSION_TITLE_MAX_CHARS + 1);
        assert_eq!(clean_title(&prose), None);
        assert_eq!(clean_title(""), None);
        assert_eq!(clean_title("   "), None);
    }

    #[test]
    fn the_request_carries_only_the_opening_text_not_tool_traffic() {
        let messages = vec![
            Message::user("help me fix the lease timeout"),
            Message::assistant(vec![ContentBlock::tool_use(
                "1",
                "read",
                serde_json::json!({ "path": "src/x.rs" }),
            )]),
            Message::tool_result("1", "fn x() {}", false),
            Message::assistant(vec![ContentBlock::text("Looking at the NFS client now")]),
        ];
        let req = session_title_request("claude-test", &messages);
        let rendered = format!("{:?}", req.messages);
        assert!(
            rendered.contains("help me fix the lease timeout"),
            "{rendered}"
        );
        assert!(
            rendered.contains("Looking at the NFS client now"),
            "{rendered}"
        );
        assert!(
            !rendered.contains("fn x() {}"),
            "tool results describe what the agent did, not what the session is about: {rendered}"
        );
        assert_eq!(req.max_tokens, SESSION_TITLE_MAX_TOKENS);
    }

    #[test]
    fn a_huge_opening_message_is_bounded() {
        let messages = vec![Message::user("y".repeat(TITLE_INPUT_MAX_CHARS * 10))];
        let req = session_title_request("claude-test", &messages);
        let rendered = format!("{:?}", req.messages);
        assert!(
            rendered.matches('y').count() <= TITLE_INPUT_MAX_CHARS,
            "a pasted file must not become the title call's input"
        );
    }
}
