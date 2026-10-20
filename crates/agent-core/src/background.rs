//! Work a tool leaves running past its own call, reported to the model without it asking.
//!
//! A tool that hands work to the background (`bash`'s jobs, in the `agent` crate) exposes a
//! [`Background`] through [`crate::Tool::background`]. The loop then does four things with it, so
//! the model never has to poll and never ends a run blind:
//!
//! - **Finish notices** ([`Background::take_notices`]) join the next tool-results turn as trailing
//!   text, the way a mid-run steer does ([`crate::steering`]). At a would-stop boundary, pending
//!   notices are delivered first, as a user turn, and the run continues: a result that arrived while
//!   the model was writing its last reply is never lost.
//! - **One stop reminder per run** ([`Background::stop_reminder`]): at a would-stop boundary with
//!   nothing to deliver, work still running that the model has not been reminded of is listed, and
//!   the run continues; the model may wait for a result it still needs, or end again. Like the
//!   step-budget wrap-up ([`crate::step_budget`]), the loop asks at most once per run — work started
//!   or waited on after it does not earn a second — so a model that ends again ends; and a source
//!   never lists the same work twice, so a later run (a `serve` prompt) is reminded only of new work. A
//!   source skips work the model already knows outlives the turn and will announce its own end.
//! - **Compaction carry** ([`Background::carry`]): the work in flight at each compaction rides the
//!   summary message as a `<background-jobs>` block, outside anything the summarizer can drop, so
//!   a compacted model does not relaunch a job it forgot. Rendered from live state at each
//!   compaction and attached as a request-only block ([`crate::Session::attach_request_block_at`]),
//!   never written into the summary text or the persisted [`crate::CompactionProvenance`]: the work
//!   is process state, and a persisted copy would describe dead jobs after a restart. With nothing
//!   in flight the block says so ([`NO_JOBS`]), and a run on a summarized session whose block is
//!   gone (request blocks are never persisted, so every reopen) attaches it afresh: a summary's own
//!   prose can still say "the server is running as job 1", and only the live table corrects it.
//!   The summarizer is told not to record job state ([`crate::compaction::LIVENESS_NOTE`]).
//!
//! The texts are the source's own; the loop adds nothing to them. Each begins with one of
//! [`NOTE_PREFIXES`], which is how compaction's carried user requests
//! ([`crate::compaction::merge_user_requests`]) tell these harness notes from the user's words.
//! Delivery is reported as [`crate::AgentEvent::Background`].

use std::sync::Arc;

/// How every notice text starts: `[job 3 exited with code 0 after 31 s: …]`.
pub const NOTICE_PREFIX: &str = "[job ";
/// How every stop reminder starts.
pub const REMINDER_PREFIX: &str = "[background jobs]";
/// Harness-authored notes the loop folds into user turns: background notices and reminders, and the
/// step-budget wrap-up. Not the user's words.
pub const NOTE_PREFIXES: [&str; 3] = [
    NOTICE_PREFIX,
    REMINDER_PREFIX,
    crate::step_budget::NOTE_PREFIX,
];

/// Whether `text` is a harness note (see [`NOTE_PREFIXES`]) rather than something a user wrote.
pub fn is_note(text: &str) -> bool {
    let text = text.trim_start();
    NOTE_PREFIXES.iter().any(|p| text.starts_with(p))
}

/// The tag around [`Background::carry`] text in a compaction summary.
pub const CARRY_OPEN: &str = "<background-jobs>";
/// See [`CARRY_OPEN`].
pub const CARRY_CLOSE: &str = "</background-jobs>";

/// The [`CARRY_OPEN`] block's text when nothing is in flight.
pub const NO_JOBS: &str =
    "No background jobs are running. Any earlier mention of a running job is out of date.";

/// Whether a request block is a compaction's [`CARRY_OPEN`] block: process state, which a host that
/// saves request blocks across a restart (`serve`'s MCP App context sidecar) must not save.
pub fn is_carry_block(text: &str) -> bool {
    text.starts_with(CARRY_OPEN)
}

/// Whether a request block describes processes — a compaction's [`CARRY_OPEN`] block, or a
/// [`Background::resume`] note (a [`REMINDER_PREFIX`] text, which otherwise only ever rides a
/// persisted user turn) — so a host that saves request blocks across a restart must not save it.
pub fn is_process_state(text: &str) -> bool {
    is_carry_block(text) || text.starts_with(REMINDER_PREFIX)
}

/// What the loop asks a tool's background work. Every method is cheap and non-blocking: they run
/// on the loop's own task at turn boundaries.
pub trait Background: Send + Sync {
    /// Notices for work that ended unattended since the last call, folded into one text (starting
    /// with [`NOTICE_PREFIX`]). Each piece of work is noticed at most once: a call consumes them.
    fn take_notices(&self) -> Option<String>;

    /// At a would-stop boundary with no notices pending: the work still running that no earlier
    /// reminder covered, as one text (starting with [`REMINDER_PREFIX`]), or `None`. Consumes: the
    /// same work is never listed twice. The loop asks at most once per run.
    fn stop_reminder(&self) -> Option<String>;

    /// The work in flight right now (running, or ended with its result unread), for a compaction
    /// summary. `None` when there is none.
    fn carry(&self) -> Option<String>;

    /// A run is starting on `messages`. A source whose live state does not cover this transcript yet
    /// — a fresh one on a reopened session, or one a transcript switch reset — reads it, once: it
    /// numbers new work past every id the transcript already used, so one id never names two pieces
    /// of work in a conversation, and returns a note (starting with [`REMINDER_PREFIX`]) listing the
    /// work the transcript left running, which is gone now. `None` otherwise, and on every later
    /// call until the source is reset. The loop attaches the note to the run's prompt as a
    /// request-only block: a later reopen reads the same transcript and says it again.
    fn resume(&self, _messages: &[crate::Message]) -> Option<String> {
        None
    }
}

/// Every distinct [`Background`] the tools of a registry expose, in a stable order.
pub(crate) fn collect(tools: &crate::ToolRegistry) -> Vec<Arc<dyn Background>> {
    let mut names = tools.names();
    names.sort();
    let mut out: Vec<Arc<dyn Background>> = Vec::new();
    for name in names {
        let Some(bg) = tools.get(&name).and_then(|t| t.background()) else {
            continue;
        };
        // Two tools over one table (`bash` and `bash_job`) must not report it twice.
        if !out.iter().any(|b| Arc::ptr_eq(b, &bg)) {
            out.push(bg);
        }
    }
    out
}

/// [`Background::take_notices`] over several sources, joined.
pub(crate) fn take_notices(sources: &[Arc<dyn Background>]) -> Option<String> {
    join(sources.iter().filter_map(|b| b.take_notices()))
}

/// [`Background::stop_reminder`] over several sources, joined.
pub(crate) fn stop_reminder(sources: &[Arc<dyn Background>]) -> Option<String> {
    join(sources.iter().filter_map(|b| b.stop_reminder()))
}

/// [`Background::resume`] over several sources, joined.
pub(crate) fn resume(
    sources: &[Arc<dyn Background>],
    messages: &[crate::Message],
) -> Option<String> {
    join(sources.iter().filter_map(|b| b.resume(messages)))
}

/// The `<background-jobs>` block for a compaction summary, or `None` when nothing is in flight.
pub(crate) fn format_carry(sources: &[Arc<dyn Background>]) -> Option<String> {
    join(sources.iter().filter_map(|b| b.carry()))
        .map(|text| format!("{CARRY_OPEN}\n{text}\n{CARRY_CLOSE}"))
}

/// The `<background-jobs>` block a summary rides with: [`format_carry`], or [`NO_JOBS`] when
/// nothing is in flight. `None` only when no tool can leave work running at all.
pub(crate) fn carry_block(sources: &[Arc<dyn Background>]) -> Option<String> {
    if sources.is_empty() {
        return None;
    }
    format_carry(sources).or_else(|| Some(format!("{CARRY_OPEN}\n{NO_JOBS}\n{CARRY_CLOSE}")))
}

fn join(parts: impl Iterator<Item = String>) -> Option<String> {
    let parts: Vec<String> = parts.collect();
    (!parts.is_empty()).then(|| parts.join("\n\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notes_are_told_apart_from_user_words() {
        assert!(is_note("[job 3 exited with code 0 after 31 s: `make`]\nok"));
        assert!(is_note("[background jobs] Still running:"));
        assert!(is_note("[step budget] This run's 10 steps are used up."));
        assert!(!is_note("[urgent] fix the build"));
        assert!(!is_note("please look at job 3"));
    }
}
