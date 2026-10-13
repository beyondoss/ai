//! Work a tool leaves running past its own call, reported to the model without it asking.
//!
//! A tool that hands work to the background (`bash`'s jobs, in the `agent` crate) exposes a
//! [`Background`] through [`crate::Tool::background`]. The loop then does three things with it, so
//! the model never has to poll and never ends a run blind:
//!
//! - **Finish notices** ([`Background::take_notices`]) join the next tool-results turn as trailing
//!   text, the way a mid-run steer does ([`crate::steering`]). At a would-stop boundary, pending
//!   notices are delivered first, as a user turn, and the run continues: a result that arrived while
//!   the model was writing its last reply is never lost.
//! - **One stop reminder** ([`Background::stop_reminder`]): at a would-stop boundary with nothing
//!   to deliver, work still running that the model has not been reminded of is listed once, and the
//!   run continues; the model may wait, stop it, or end again. Like the step-budget wrap-up
//!   ([`crate::step_budget`]), it never repeats for the same work, so a model that ends again ends.
//! - **Compaction carry** ([`Background::carry`]): the work in flight is appended to every
//!   compaction summary as a `<background-jobs>` block, outside anything the summarizer can drop, so
//!   a compacted model does not relaunch a job it forgot. Rendered from live state at each
//!   compaction rather than folded into the persisted [`crate::CompactionProvenance`]: the work is
//!   process state, and a persisted copy would describe dead jobs after a restart.
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

/// What the loop asks a tool's background work. Every method is cheap and non-blocking: they run
/// on the loop's own task at turn boundaries.
pub trait Background: Send + Sync {
    /// Notices for work that ended unattended since the last call, folded into one text (starting
    /// with [`NOTICE_PREFIX`]). Each piece of work is noticed at most once: a call consumes them.
    fn take_notices(&self) -> Option<String>;

    /// At a would-stop boundary with no notices pending: the work still running that no earlier
    /// reminder covered, as one text (starting with [`REMINDER_PREFIX`]), or `None`. Consumes: the
    /// same work is never listed twice.
    fn stop_reminder(&self) -> Option<String>;

    /// The work in flight right now (running, or ended with its result unread), for a compaction
    /// summary. `None` when there is none.
    fn carry(&self) -> Option<String>;
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

/// The `<background-jobs>` block for a compaction summary, or `""`.
pub(crate) fn format_carry(sources: &[Arc<dyn Background>]) -> String {
    match join(sources.iter().filter_map(|b| b.carry())) {
        Some(text) => format!("\n\n{CARRY_OPEN}\n{text}\n{CARRY_CLOSE}"),
        None => String::new(),
    }
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
