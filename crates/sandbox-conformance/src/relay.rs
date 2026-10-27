//! Relay scenarios: steps run through different drivers on the same sandbox (DRIVER.md section 5,
//! "Relay"). A step with `"via": "A"` or no `via` goes to the first driver of a pair, `"via": "B"`
//! to the second; the harness binds A and B to every ordered pair of registered drivers.
use serde_json::Value;

use crate::run::{Backend, Check, Status, run_scenario_via};

/// Any step (setup included) says `via`.
pub fn is_relay(sc: &Value) -> bool {
    sc["setup"]
        .as_array()
        .into_iter()
        .flatten()
        .chain(sc["steps"].as_array().into_iter().flatten())
        .any(|s| s.get("via").is_some())
}

/// The backend (driver) a step goes to.
pub fn side<'a>(step: &Value, a: &'a Backend, b: Option<&'a Backend>) -> &'a Backend {
    match (step["via"].as_str(), b) {
        (Some("B"), Some(b)) => b,
        _ => a,
    }
}

/// Every relay scenario of `files` (those whose suite `suites` allows and whose id `keep` accepts),
/// run with `a` as A and `b` as B. `same_driver` pairs run only the scenarios that ask for them
/// (`"same_driver": true`).
pub async fn run_pair(
    files: &[(String, Value)],
    a: &Backend,
    b: &Backend,
    same_driver: bool,
    suites: Option<&[&str]>,
    keep: &(dyn Fn(&str) -> bool + Sync),
) -> Vec<Check> {
    let mut checks = Vec::new();
    for (_, file) in files {
        if suites.is_some_and(|s| !s.contains(&file["suite"].as_str().unwrap_or(""))) {
            continue;
        }
        for sc in file["scenarios"].as_array().into_iter().flatten() {
            if !is_relay(sc) || !keep(sc["id"].as_str().unwrap_or("")) {
                continue;
            }
            if same_driver && sc["same_driver"] != true {
                continue;
            }
            checks.extend(run_scenario_via(sc, a, Some(b)).await);
        }
    }
    checks
}

/// Whether a pair produced anything worth a cell.
pub fn ran(checks: &[Check]) -> bool {
    checks.iter().any(|c| c.status != Status::Skipped)
}
