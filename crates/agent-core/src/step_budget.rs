//! Experimental step-budget behaviors for a run with a `max_steps` ceiling, all off by default.
//!
//! In the longctx eval, gpt-6-luna hit `MaxSteps(150)` in 7 of 22 code runs. Every one stopped
//! within two steps of an edit, with no test run since. Two of them scored 0% with 220+ regressions,
//! though their sibling attempts scored ~90%: one edit had left an `IndentationError` in place, and
//! the cap cut off the fix. Separately, Luna quit 4 text runs with 88–201 of 250 steps left,
//! reporting "I did not complete the task". These knobs exist so an A/B can measure three
//! responses. Each is inert unless [`crate::Agent::with_max_steps`] set a ceiling:
//!
//! - [`StepBudget::notice_at`]: tell the model once how many steps remain, near the end.
//! - [`StepBudget::wrap_up_steps`]: past the ceiling, grant a few steps to leave the tree consistent.
//! - [`StepBudget::continue_nudge`]: when the model stops early with budget left, ask it once
//!   whether it is really done.
//!
//! The text rides the existing steering paths: a notice or wrap-up joins the tool-results turn as a
//! trailing text block, as a mid-run steer does, and the nudge is a user turn, as a follow-up is.
//! Each fires at most once per `run` call and is reported as [`crate::AgentEvent::StepBudget`].

use serde::Serialize;

/// See the module doc. `Default` turns everything off.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StepBudget {
    /// Once this many steps or fewer remain, fold a one-time "N of M steps remain" notice into the
    /// next tool-results turn.
    pub notice_at: Option<u32>,
    /// When the last allowed step's tools have run, grant this many more steps with an instruction
    /// to leave the workspace consistent and summarize. The run still ends in `Error::MaxSteps`
    /// (reporting the original ceiling) if the model uses them all.
    pub wrap_up_steps: Option<u32>,
    /// When the model ends its turn with at least [`NUDGE_MIN_REMAINING_PCT`] of its steps left,
    /// ask it once to continue if the task is not complete.
    pub continue_nudge: bool,
}

/// The least share of the ceiling, in percent, that must remain for [`StepBudget::continue_nudge`]
/// to fire. Luna's four early quits had 35–80% left. A model stopping near the cap has a better
/// reason to stop than to be asked again.
pub const NUDGE_MIN_REMAINING_PCT: u32 = 20;

/// Which step-budget message was injected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StepBudgetKind {
    Notice,
    WrapUp,
    Nudge,
}

pub(crate) fn notice_text(remaining: u32, max: u32) -> String {
    format!(
        "[step budget] {remaining} of this run's {max} steps remain. Plan to end with the work in a \
         consistent state: finish the change in progress and re-run its tests before starting \
         anything new."
    )
}

pub(crate) fn wrap_up_text(max: u32, extra: u32) -> String {
    format!(
        "[step budget] This run's {max} steps are used up. You have {extra} more to leave the \
         workspace consistent: if your recent edits left the code broken (a syntax error, a failing \
         import, a half-applied change), fix or revert them and confirm with a quick check. Start no \
         new work. Then reply with a short summary of what is done and what is not."
    )
}

pub(crate) fn nudge_text(remaining: u32, max: u32) -> String {
    format!(
        "[step budget] You ended your turn with {remaining} of {max} steps left. If the task is not \
         complete (tests still fail, or parts are unimplemented), keep working on it now. If it is \
         complete, reply with your final summary only."
    )
}

/// Per-call state: what has fired, and the ceiling as extended by a wrap-up grant.
#[derive(Debug)]
pub(crate) struct BudgetState {
    pub policy: StepBudget,
    /// The configured ceiling. `None` makes every method here a no-op.
    pub max: Option<u32>,
    /// `max` plus any wrap-up grant: what the loop's ceiling check compares against.
    pub limit: Option<u32>,
    noticed: bool,
    wrapped: bool,
    nudged: bool,
}

impl BudgetState {
    pub fn new(policy: StepBudget, max: Option<u32>) -> Self {
        Self {
            policy,
            max,
            limit: max,
            noticed: false,
            wrapped: false,
            nudged: false,
        }
    }

    /// After a tool-results turn has used `steps` steps: the text to fold into it, if any. A
    /// wrap-up grant also raises [`Self::limit`].
    pub fn after_tools(&mut self, steps: u32) -> Option<(StepBudgetKind, u32, String)> {
        let (max, limit) = (self.max?, self.limit?);
        let remaining = limit.saturating_sub(steps);
        if remaining == 0
            && !self.wrapped
            && let Some(extra) = self.policy.wrap_up_steps.filter(|&n| n > 0)
        {
            self.wrapped = true;
            self.limit = Some(limit.saturating_add(extra));
            return Some((StepBudgetKind::WrapUp, extra, wrap_up_text(max, extra)));
        }
        if remaining > 0
            && !self.noticed
            && !self.wrapped
            && self.policy.notice_at.is_some_and(|n| remaining <= n)
        {
            self.noticed = true;
            return Some((
                StepBudgetKind::Notice,
                remaining,
                notice_text(remaining, max),
            ));
        }
        None
    }

    /// The model ended its turn after `steps` steps with nothing queued: the nudge to send, if any.
    pub fn at_stop(&mut self, steps: u32) -> Option<(u32, String)> {
        let max = self.max?;
        if !self.policy.continue_nudge || self.nudged || self.wrapped || max == 0 {
            return None;
        }
        let remaining = max.saturating_sub(steps);
        if u64::from(remaining) * 100 < u64::from(max) * u64::from(NUDGE_MIN_REMAINING_PCT) {
            return None;
        }
        self.nudged = true;
        Some((remaining, nudge_text(remaining, max)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(policy: StepBudget, max: u32) -> BudgetState {
        BudgetState::new(policy, Some(max))
    }

    #[test]
    fn everything_is_inert_without_a_ceiling_or_a_policy() {
        let all = StepBudget {
            notice_at: Some(5),
            wrap_up_steps: Some(3),
            continue_nudge: true,
        };
        let mut s = BudgetState::new(all, None);
        assert!(s.after_tools(1).is_none());
        assert!(s.at_stop(1).is_none());
        let mut s = state(StepBudget::default(), 10);
        for step in 1..=10 {
            assert!(s.after_tools(step).is_none());
        }
        assert!(s.at_stop(1).is_none());
        assert_eq!(s.limit, Some(10));
    }

    #[test]
    fn the_notice_fires_once_at_the_threshold() {
        let mut s = state(
            StepBudget {
                notice_at: Some(3),
                ..StepBudget::default()
            },
            10,
        );
        assert!(s.after_tools(6).is_none());
        let (kind, remaining, text) = s.after_tools(7).expect("3 remain");
        assert_eq!((kind, remaining), (StepBudgetKind::Notice, 3));
        assert!(text.contains("3 of this run's 10 steps remain"), "{text}");
        assert!(s.after_tools(8).is_none(), "once");
    }

    #[test]
    fn the_wrap_up_extends_the_limit_once_at_the_ceiling() {
        let mut s = state(
            StepBudget {
                notice_at: Some(2),
                wrap_up_steps: Some(3),
                ..StepBudget::default()
            },
            10,
        );
        assert_eq!(s.after_tools(8).map(|t| t.0), Some(StepBudgetKind::Notice));
        assert!(s.after_tools(9).is_none());
        let (kind, extra, text) = s.after_tools(10).expect("ceiling reached");
        assert_eq!((kind, extra), (StepBudgetKind::WrapUp, 3));
        assert!(text.contains("10 steps are used up"), "{text}");
        assert_eq!(s.limit, Some(13));
        assert!(s.after_tools(13).is_none(), "one grant per call");
    }

    #[test]
    fn the_nudge_fires_once_and_only_with_enough_budget_left() {
        let nudging = StepBudget {
            continue_nudge: true,
            ..StepBudget::default()
        };
        let mut s = state(nudging, 100);
        assert!(s.at_stop(81).is_none(), "19% left is below the floor");
        let (remaining, text) = s.at_stop(80).expect("20% left");
        assert_eq!(remaining, 20);
        assert!(text.contains("20 of 100 steps left"), "{text}");
        assert!(s.at_stop(10).is_none(), "once");

        // Never during a wrap-up: the model was told to stop.
        let mut s = state(
            StepBudget {
                wrap_up_steps: Some(2),
                continue_nudge: true,
                ..StepBudget::default()
            },
            10,
        );
        assert!(s.after_tools(10).is_some());
        assert!(s.at_stop(11).is_none());
    }
}
