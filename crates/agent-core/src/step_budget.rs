//! The wrap-up at a `max_steps` ceiling: when the last allowed step's tools have run, the model gets
//! [`WRAP_UP_STEPS`] more steps to stop cleanly. Inert unless [`crate::Agent::with_max_steps`] set a
//! ceiling.
//!
//! Why: in the longctx eval, gpt-6-luna hit `MaxSteps(150)` in 7 of 22 code runs, each within two
//! steps of an untested edit. Two of those scored 0% with 220+ regressions while their sibling
//! attempts scored ~90%: an edit had left an `IndentationError` in place, and the cap cut off the
//! fix. With the wrap-up (Luna, sqlparse and marshmallow at `--max-steps 60`, 3 runs a side), all 6
//! runs ended with an honest summary of what was done and what was not, against 6 silent cut-offs,
//! no tree ended broken on either side, and the cost was the same. The first wording ("fix or revert
//! anything broken, start no new work") let one run keep implementing during the grant and end with
//! 171 regressions, so the text forbids continuing the task outright.
//!
//! The text rides the steering path: it joins the tool-results turn as a trailing text block, as a
//! mid-run steer does. It fires at most once per `run` call and is reported as
//! [`crate::AgentEvent::WrapUp`].

/// Steps granted past the ceiling. The runs above used 1–3 of them: a compile check, one full test
/// run, a summary.
pub const WRAP_UP_STEPS: u32 = 5;

pub(crate) fn wrap_up_text(max: u32, extra: u32) -> String {
    format!(
        "[step budget] This run's {max} steps are used up. You have {extra} more, only to stop \
         cleanly. Do not continue the task: write no new code and do not finish a change in \
         progress. Only if a file you edited now fails to parse or import, revert or fix that one \
         problem. Run the test suite once to confirm. Then reply with a short summary of what is \
         done and what is not."
    )
}

/// Per-call state: whether the wrap-up has fired, and the ceiling as extended by its grant.
#[derive(Debug)]
pub(crate) struct BudgetState {
    /// The configured ceiling. `None` makes every method here a no-op.
    max: Option<u32>,
    /// `max` plus the wrap-up grant once it fires: what the loop's ceiling check compares against.
    pub limit: Option<u32>,
    wrapped: bool,
}

impl BudgetState {
    pub fn new(max: Option<u32>) -> Self {
        Self {
            max,
            limit: max,
            wrapped: false,
        }
    }

    /// After a tool-results turn has used `steps` steps: the steps granted and the text to fold into
    /// that turn, the first time the ceiling is reached. Also raises [`Self::limit`].
    pub fn after_tools(&mut self, steps: u32) -> Option<(u32, String)> {
        let (max, limit) = (self.max?, self.limit?);
        if steps < limit || self.wrapped {
            return None;
        }
        self.wrapped = true;
        self.limit = Some(limit.saturating_add(WRAP_UP_STEPS));
        Some((WRAP_UP_STEPS, wrap_up_text(max, WRAP_UP_STEPS)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_is_inert_without_a_ceiling() {
        let mut s = BudgetState::new(None);
        assert!(s.after_tools(1).is_none());
        assert_eq!(s.limit, None);
    }

    #[test]
    fn the_wrap_up_extends_the_limit_once_at_the_ceiling() {
        let mut s = BudgetState::new(Some(10));
        assert!(s.after_tools(9).is_none());
        let (extra, text) = s.after_tools(10).expect("ceiling reached");
        assert_eq!(extra, WRAP_UP_STEPS);
        assert!(text.contains("10 steps are used up"), "{text}");
        assert!(text.contains("Do not continue the task"), "{text}");
        assert_eq!(s.limit, Some(10 + WRAP_UP_STEPS));
        assert!(
            s.after_tools(10 + WRAP_UP_STEPS).is_none(),
            "one grant per call"
        );
    }
}
