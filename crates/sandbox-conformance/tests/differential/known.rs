//! Divergences the differential run has found and that have their own scenario in
//! `sdk/conformance/scenarios/differential-found.json` (marked `xfail` for the drivers that fail
//! it). They are tolerated here by id, with the drivers that must be the odd ones out, so that one
//! known defect does not hide every other: the same arrangement as the property test's `known(..)` calls. A fix deletes its entry here and its `xfail` mark there.
use serde_json::Value;

/// `Some(id)` when `step` is one an open defect explains and exactly the drivers it is known to
/// afflict (`differing`) are the ones that answered unlike the rest or against the filesystem.
/// There is none open: a defect found by the run gets an entry here and a scenario in
/// `differential-found.json`, and the fix deletes both.
pub fn divergence(_step: &Value, _differing: &[&str], _answers: &[Value]) -> Option<&'static str> {
    None
}
