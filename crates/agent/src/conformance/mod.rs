//! The conformance kits for the parts of Beyond that sit behind this repo's contracts (`BEYOND.md`,
//! "Conformance"): [`broker`] for the connector broker. (The exec protocol's kit is
//! `exec_endpoint::conformance`.)
//!
//! A kit is a list of named checks, each a short function that says which `BEYOND.md` line it
//! proves, and a verdict per check. A kit that needs to see or provoke state only the implementer can
//! reach asks an **operator hook**: one command the implementer supplies, called as
//! `<hook> <op> <args…>`, that prints its answer on stdout and exits non-zero when it cannot do the
//! op. The kit stays ignorant of the implementation, and the hook is the only door into it.

use crate::exec_endpoint::conformance::{Check, Verdict};

pub mod broker;

pub(crate) fn pass(name: &'static str, detail: impl Into<String>) -> Check {
    Check {
        name,
        verdict: Verdict::Pass,
        detail: detail.into(),
    }
}

pub(crate) fn fail(name: &'static str, detail: impl Into<String>) -> Check {
    Check {
        name,
        verdict: Verdict::Fail,
        detail: detail.into(),
    }
}

pub(crate) fn skip(name: &'static str, detail: impl Into<String>) -> Check {
    Check {
        name,
        verdict: Verdict::Skip,
        detail: detail.into(),
    }
}

/// A kit's verdict: one line per check.
#[derive(Debug, Clone)]
pub struct Report {
    pub kit: &'static str,
    pub checks: Vec<Check>,
}

impl Report {
    /// No check failed (a skipped one is not a pass, but it is not a failure either).
    pub fn passed(&self) -> bool {
        self.checks.iter().all(|c| c.verdict != Verdict::Fail)
    }

    pub fn check(&self, name: &str) -> Option<&Check> {
        self.checks.iter().find(|c| c.name == name)
    }

    pub fn failed(&self) -> Vec<&'static str> {
        self.checks
            .iter()
            .filter(|c| c.verdict == Verdict::Fail)
            .map(|c| c.name)
            .collect()
    }

    pub fn render(&self) -> String {
        let mut out = format!("{}:\n", self.kit);
        for c in &self.checks {
            let mark = match c.verdict {
                Verdict::Pass => "PASS",
                Verdict::Fail => "FAIL",
                Verdict::Skip => "skip",
            };
            out.push_str(&format!("  {mark}  {}: {}\n", c.name, c.detail));
        }
        out
    }
}

/// An operator hook run as a command: `<command> <op> <args…>` through `sh -c`.
pub struct CommandHook(pub String);

impl CommandHook {
    pub async fn run(&self, op: &str, args: &[&str]) -> Result<String, String> {
        let out = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(format!("{} \"$@\"", self.0))
            .arg("hook")
            .arg(op)
            .args(args)
            .output()
            .await
            .map_err(|e| format!("hook `{op}` did not start: {e}"))?;
        if !out.status.success() {
            return Err(format!(
                "hook `{op}` exited {}: {}",
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    }
}
