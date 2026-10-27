//! One cell of the matrix per `driver__backend`, as a human line and as JSON.
use serde_json::{Value, json};

use crate::run::{Check, Status, Tally};

pub struct Cell {
    pub driver: String,
    pub backend: String,
    pub tally: Tally,
    /// The failing checks with their reasons, for the human report.
    pub failures: Vec<Check>,
    /// Known defects, unsupported and skipped checks, with why: shown by `--verbose`.
    pub others: Vec<Check>,
}

impl Cell {
    pub fn name(&self) -> String {
        format!("{}__{}", self.driver, self.backend)
    }

    pub fn passes(&self) -> bool {
        self.tally.failed == 0
    }

    pub fn from_checks(driver: &str, backend: &str, checks: &[Check]) -> Self {
        let mut tally = Tally::default();
        tally.add(checks);
        Cell {
            driver: driver.into(),
            backend: backend.into(),
            failures: checks
                .iter()
                .filter(|c| c.status == Status::Fail)
                .cloned()
                .collect(),
            others: checks
                .iter()
                .filter(|c| {
                    matches!(
                        c.status,
                        Status::KnownDefect | Status::Unsupported | Status::Skipped
                    )
                })
                .cloned()
                .collect(),
            tally,
        }
    }

    pub fn failed_to_run(driver: &str, backend: &str, id: &str, why: String) -> Self {
        Self::from_checks(
            driver,
            backend,
            &[Check {
                id: id.into(),
                status: Status::Fail,
                detail: why,
            }],
        )
    }

    pub fn json(&self, commit: &str, updated: &str) -> Value {
        let t = &self.tally;
        json!({
            "status": if self.passes() { "pass" } else { "fail" },
            "checks": t.total(),
            "passed": t.passed,
            "failed": t.failed,
            "known_defect": t.known_defect,
            "unsupported": t.unsupported,
            "skipped": t.skipped,
            "failing": t.failing,
            "commit": commit,
            "updated": updated,
        })
    }
}

/// The table: one line per cell.
pub fn table(cells: &[Cell], verbose: bool) -> String {
    let w = cells.iter().map(|c| c.name().len()).max().unwrap_or(0);
    let mut out = format!(
        "{:<w$}  {:<5}  {:>6} {:>6} {:>12} {:>11} {:>7}\n",
        "driver__backend", "", "pass", "fail", "known-defect", "unsupported", "skipped"
    );
    for c in cells {
        let t = &c.tally;
        out.push_str(&format!(
            "{:<w$}  {:<5}  {:>6} {:>6} {:>12} {:>11} {:>7}\n",
            c.name(),
            if c.passes() { "PASS" } else { "FAIL" },
            t.passed,
            t.failed,
            t.known_defect,
            t.unsupported,
            t.skipped
        ));
    }
    for c in cells {
        for f in &c.failures {
            out.push_str(&format!("\nFAIL {} {}: {}\n", c.name(), f.id, f.detail));
        }
    }
    if verbose {
        for c in cells {
            for o in &c.others {
                out.push_str(&format!(
                    "\n{:?} {} {}: {}",
                    o.status,
                    c.name(),
                    o.id,
                    o.detail
                ));
            }
        }
        out.push('\n');
    }
    out
}

/// UTC `YYYY-MM-DDTHH:MM:SSZ` for the report, without a date crate.
pub fn now_rfc3339() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let (days, rem) = (secs.div_euclid(86400), secs.rem_euclid(86400));
    // Civil-from-days (Howard Hinnant).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}
