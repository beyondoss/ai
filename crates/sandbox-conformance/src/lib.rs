//! The one evaluator of `sdk/conformance`: replays the shared scenario files through any driver
//! (see `sdk/conformance/DRIVER.md`) against any backend, and reports each check as pass, fail,
//! known defect, unsupported or skipped.
//!
//! Two callers share it: the reference test in `crates/agent/tests` (an in-process driver) and the
//! `sandbox-matrix` binary (driver processes). Both reach a driver through [`Link`].
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub mod backends;
pub mod e2b;
pub mod eval;
pub mod link;
pub mod matrix;
pub mod reaper;
pub mod relay;
pub mod report;
pub mod run;

pub use run::{
    Backend, Check, Connector, Link, Status, Tally, load_scenarios, run_scenario, run_scenario_via,
};
