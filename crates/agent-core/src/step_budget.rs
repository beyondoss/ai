//! An experimental step-budget behavior for a run with a `max_steps` ceiling, off by default.
//!
//! In the longctx eval, gpt-6-luna hit `MaxSteps(150)` in 7 of 22 code runs. Every one stopped
//! within two steps of an edit, with no test run since. Two of them scored 0% with 220+ regressions,
//! though their sibling attempts scored ~90%: one edit had left an `IndentationError` in place, and
//! the cap cut off the fix. [`StepBudget::wrap_up_steps`] exists so an A/B can measure granting a few
//! steps past the ceiling to leave the tree consistent. It is inert unless
//! [`crate::Agent::with_max_steps`] set a ceiling.
//!
//! The text rides the existing steering path: it joins the tool-results turn as a trailing text
//! block, as a mid-run steer does. It fires at most once per `run` call and is reported as
//! [`crate::AgentEvent::StepBudget`].

use serde::Serialize;

/// See the module doc. `Default` turns it off.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StepBudget {
    /// When the last allowed step's tools have run, grant this many more steps with an instruction
    /// to leave the workspace consistent and summarize. The run still ends in `Error::MaxSteps`
    /// (reporting the original ceiling) if the model uses them all.
    pub wrap_up_steps: Option<u32>,
}

/// Which step-budget message was injected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StepBudgetKind {
    WrapUp,
}

pub(crate) fn wrap_up_text(max: u32, extra: u32) -> String {
    format!(
        "[step budget] This run's {max} steps are used up. You have {extra} more to leave the \
         workspace consistent: if your recent edits left the code broken (a syntax error, a failing \
         import, a half-applied change), fix or revert them and confirm with a quick check. Start no \
         new work. Then reply with a short summary of what is done and what is not."
    )
}

/// Per-call state: whether the wrap-up has fired, and the ceiling as extended by its grant.
#[derive(Debug)]
pub(crate) struct BudgetState {
    pub policy: StepBudget,
    /// The configured ceiling. `None` makes every method here a no-op.
    pub max: Option<u32>,
    /// `max` plus any wrap-up grant: what the loop's ceiling check compares against.
    pub limit: Option<u32>,
    wrapped: bool,
}

impl BudgetState {
    pub fn new(policy: StepBudget, max: Option<u32>) -> Self {
        Self {
            policy,
            max,
            limit: max,
            wrapped: false,
        }
    }

    /// After a tool-results turn has used `steps` steps: the text to fold into it, if any. A
    /// wrap-up grant also raises [`Self::limit`].
    pub fn after_tools(&mut self, steps: u32) -> Option<(StepBudgetKind, u32, String)> {
        let (max, limit) = (self.max?, self.limit?);
        if limit.saturating_sub(steps) > 0 || self.wrapped {
            return None;
        }
        let extra = self.policy.wrap_up_steps.filter(|&n| n > 0)?;
        self.wrapped = true;
        self.limit = Some(limit.saturating_add(extra));
        Some((StepBudgetKind::WrapUp, extra, wrap_up_text(max, extra)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(policy: StepBudget, max: u32) -> BudgetState {
        BudgetState::new(policy, Some(max))
    }

    #[test]
    fn it_is_inert_without_a_ceiling_or_a_policy() {
        let on = StepBudget {
            wrap_up_steps: Some(3),
        };
        let mut s = BudgetState::new(on, None);
        assert!(s.after_tools(1).is_none());
        let mut s = state(StepBudget::default(), 10);
        for step in 1..=10 {
            assert!(s.after_tools(step).is_none());
        }
        assert_eq!(s.limit, Some(10));
    }

    #[test]
    fn the_wrap_up_extends_the_limit_once_at_the_ceiling() {
        let mut s = state(
            StepBudget {
                wrap_up_steps: Some(3),
            },
            10,
        );
        assert!(s.after_tools(9).is_none());
        let (kind, extra, text) = s.after_tools(10).expect("ceiling reached");
        assert_eq!((kind, extra), (StepBudgetKind::WrapUp, 3));
        assert!(text.contains("10 steps are used up"), "{text}");
        assert_eq!(s.limit, Some(13));
        assert!(s.after_tools(13).is_none(), "one grant per call");
    }
}
