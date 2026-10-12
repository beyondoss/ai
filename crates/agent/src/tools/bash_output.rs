//! How `bash` shapes a finished command's output for the model — the `--tool-tuning` knobs
//! `bash-max-bytes` and `bash-test-digest` (see [`super::tuning`]). Both default to today's behavior:
//! the last 50 KiB, verbatim.
//!
//! The evidence (longctx eval): test output was 18.9% of GPT-6 Luna's prompt tokens against 2.7–3.1%
//! for every other model, because Luna runs the bare test command and pytest's default long
//! tracebacks fill the cap (median result 13 KB, 29% at the 50 KiB cap). Tracebacks were 74% of those
//! bytes, the short test summary 24%, and passing-test noise under 1%.

use super::output::{OutputAccumulator, OutputSnapshot, format_size, marker};

/// Output above this many bytes is a digest candidate. Below it the whole thing is cheap enough to
/// keep, and no model's typical (filtered) test run reaches it: Haiku/Sonnet/Opus never exceeded it.
const DIGEST_MIN_BYTES: usize = 16 * 1024;
/// How much of the FAILURES/ERRORS section to keep: its start, i.e. the first failures in run order —
/// the part tail truncation drops first.
const FAILURES_BUDGET: usize = 6 * 1024;
/// How much of the "short test summary info" block to keep. Luna's median block is ~4.4 KB.
const SUMMARY_BUDGET: usize = 4 * 1024;

/// `bash`'s output knobs, carried by [`super::ToolTuning`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BashOutput {
    /// The display cap in bytes (`bash-max-bytes`); the tail is kept. Default 50 KiB.
    pub max_bytes: usize,
    /// Condense large pytest output to its first failures, short summary and final counts
    /// (`bash-test-digest=pytest`). Default off.
    pub pytest_digest: bool,
}

impl Default for BashOutput {
    fn default() -> Self {
        Self {
            max_bytes: super::output::DEFAULT_MAX_BYTES,
            pytest_digest: false,
        }
    }
}

impl BashOutput {
    /// A fresh accumulator honoring [`Self::max_bytes`].
    pub fn accumulator(&self, spill: bool) -> OutputAccumulator {
        let acc = OutputAccumulator::new().with_max_bytes(self.max_bytes);
        if spill { acc } else { acc.without_spill() }
    }

    /// The finished command's snapshot, plus the digested display text when the digest applies.
    ///
    /// A digest drops lines from output that may not have been truncated at all, so it forces the
    /// spill file the plain path only writes past the cap: whatever is dropped must stay readable.
    pub fn finish(&self, acc: &mut OutputAccumulator) -> (OutputSnapshot, Option<String>) {
        let mut snap = acc.snapshot(true);
        let Some(body) = self
            .pytest_digest
            .then(|| pytest_digest(&snap.content))
            .flatten()
        else {
            return (snap, None);
        };
        if snap.full_output_path.is_none() && !snap.spill_disabled {
            snap.full_output_path = acc.persist();
        }
        let full = if snap.spill_disabled || snap.full_output_path.is_some() {
            format!(" {}", super::output::full_output_segment(&snap))
        } else {
            String::new()
        };
        let note = marker(format!(
            "pytest output condensed from {} to its first failures, short test summary and final \
             counts.{full}",
            format_size(snap.truncation.total_bytes),
        ));
        (snap, Some(format!("{body}\n\n{note}")))
    }
}

/// A pytest banner line — `===== title =====` — and its title.
fn banner(line: &str) -> Option<&str> {
    let l = line.trim_end();
    (l.len() >= 7 && l.starts_with("===") && l.ends_with("===")).then(|| l.trim_matches('=').trim())
}

/// The final counts banner: `=== 3 failed, 10 passed, 1 warning in 1.23s ===`.
fn is_counts(title: &str) -> bool {
    let Some((counts, time)) = title.rsplit_once(" in ") else {
        return false;
    };
    time.starts_with(|c: char| c.is_ascii_digit())
        && [
            "passed",
            "failed",
            "error",
            "skipped",
            "xfailed",
            "xpassed",
            "deselected",
        ]
        .iter()
        .any(|w| counts.contains(w))
}

/// Append `lines` to `out` until `budget` bytes, returning how many lines did not fit.
fn take_budgeted(out: &mut String, lines: &[&str], budget: usize) -> usize {
    let start = out.len();
    for (i, l) in lines.iter().enumerate() {
        if out.len() - start + l.len() > budget && i > 0 {
            return lines.len() - i;
        }
        out.push_str(l);
    }
    0
}

/// Condense pytest output to (1) the start of the FAILURES/ERRORS section, (2) the short test
/// summary, and (3) the final counts line, dropping the session header, progress lines, the warnings
/// summary and later tracebacks. `None` when the output is small, is not pytest's, or would not shrink.
pub(crate) fn pytest_digest(content: &str) -> Option<String> {
    if content.len() <= DIGEST_MIN_BYTES {
        return None;
    }
    let lines: Vec<&str> = content.split_inclusive('\n').collect();
    let titles: Vec<Option<&str>> = lines.iter().map(|l| banner(l)).collect();
    let failures = titles
        .iter()
        .position(|t| matches!(t, Some("FAILURES" | "ERRORS")));
    let summary = titles
        .iter()
        .position(|t| *t == Some("short test summary info"));
    let counts = titles.iter().rposition(|t| t.is_some_and(is_counts));
    if failures.is_none() && summary.is_none() && counts.is_none() {
        return None;
    }

    let mut out = String::new();
    if let Some(f) = failures {
        // The section runs until the next banner that is not another FAILURES/ERRORS header.
        let end = (f + 1..lines.len())
            .find(|&i| titles[i].is_some_and(|t| t != "FAILURES" && t != "ERRORS"))
            .unwrap_or(lines.len());
        let dropped = take_budgeted(&mut out, &lines[f..end], FAILURES_BUDGET);
        if dropped > 0 {
            out.push_str(&format!("… {dropped} more lines of tracebacks\n"));
        }
    }
    if let Some(s) = summary {
        let end = counts.filter(|&c| c > s).unwrap_or(lines.len());
        let dropped = take_budgeted(&mut out, &lines[s..end], SUMMARY_BUDGET);
        if dropped > 0 {
            out.push_str(&format!("… {dropped} more summary lines\n"));
        }
    }
    // Never already in either block: both stop at the first banner that is not their own.
    if let Some(c) = counts {
        out.push_str(lines[c]);
    }
    let out = out.trim_end().to_string();
    (out.len() < content.len()).then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pytest_run(failures: usize, tb_lines: usize) -> String {
        let mut s = String::from(
            "============================= test session starts ==============================\n\
             platform linux -- Python 3.12.3, pytest-8.3.4\nrootdir: /repo\ncollected 500 items\n\n",
        );
        s.push_str(&format!("tests/test_a.py {}\n", ".".repeat(400)));
        s.push_str(
            "=================================== FAILURES ===================================\n",
        );
        for i in 0..failures {
            s.push_str(&format!("_________________________________ test_case_{i} _________________________________\n"));
            for j in 0..tb_lines {
                s.push_str(&format!(
                    "    assert value_{i}_{j} == expected  # traceback line {j}\n"
                ));
            }
            s.push_str(&format!("E   AssertionError: case {i}\n"));
        }
        s.push_str(
            "=============================== warnings summary ===============================\n",
        );
        s.push_str("tests/test_a.py::test_x\n  DeprecationWarning: old\n");
        s.push_str(
            "=========================== short test summary info ============================\n",
        );
        for i in 0..failures {
            s.push_str(&format!(
                "FAILED tests/test_a.py::test_case_{i} - AssertionError: case {i}\n"
            ));
        }
        s.push_str(&format!(
            "======================= {failures} failed, {} passed in 3.21s ========================\n",
            500 - failures
        ));
        s
    }

    #[test]
    fn small_output_is_left_alone() {
        assert_eq!(pytest_digest(&pytest_run(2, 5)), None);
    }

    #[test]
    fn non_pytest_output_is_left_alone() {
        let big = "make: building target\n".repeat(2000);
        assert_eq!(pytest_digest(&big), None);
    }

    #[test]
    fn large_pytest_output_keeps_first_failures_summary_and_counts() {
        let run = pytest_run(40, 30);
        assert!(run.len() > 3 * DIGEST_MIN_BYTES);
        let d = pytest_digest(&run).expect("must digest");
        assert!(
            d.len() < FAILURES_BUDGET + SUMMARY_BUDGET + 1024,
            "{} bytes",
            d.len()
        );
        // The first failure in run order survives — the part tail truncation loses.
        assert!(d.contains("test_case_0 ___"), "{d}");
        assert!(d.contains("more lines of tracebacks"), "{d}");
        // Every FAILED line fits here, and the final counts line ends it.
        assert!(d.contains("FAILED tests/test_a.py::test_case_39"), "{d}");
        assert!(d.trim_end().ends_with("========================"), "{d}");
        assert!(d.contains("40 failed, 460 passed in 3.21s"), "{d}");
        // Dropped: the session header, the progress dots and the warnings summary.
        assert!(!d.contains("test session starts"), "{d}");
        assert!(!d.contains("....."), "{d}");
        assert!(!d.contains("DeprecationWarning"), "{d}");
    }

    #[test]
    fn an_oversized_summary_is_capped_with_a_count_of_what_was_dropped() {
        let d = pytest_digest(&pytest_run(400, 1)).expect("must digest");
        assert!(d.contains("more summary lines"), "{d}");
        assert!(d.contains("400 failed, 100 passed"), "{d}");
    }

    #[test]
    fn a_tail_truncated_run_with_no_failures_header_still_keeps_summary_and_counts() {
        // What the 50 KiB tail looks like when the head (and the FAILURES banner) was cut.
        let run = pytest_run(60, 40);
        let tail = &run[run.len() - 40 * 1024..];
        let tail = &tail[tail.find('\n').unwrap() + 1..];
        let d = pytest_digest(tail).expect("must digest");
        assert!(
            d.starts_with("=========================== short test summary info"),
            "{d}"
        );
        assert!(d.contains("60 failed"), "{d}");
    }

    #[test]
    fn counts_banner_recognition() {
        assert!(is_counts("3 failed, 10 passed in 1.23s"));
        assert!(is_counts("1 passed, 1 warning in 0.50s (0:00:00)"));
        assert!(is_counts("no tests ran, 2 deselected in 0.01s"));
        assert!(!is_counts("short test summary info"));
        assert!(!is_counts("warnings summary"));
        assert!(!is_counts("FAILURES"));
    }

    #[tokio::test]
    async fn finish_persists_output_the_digest_drops_even_below_the_cap() {
        let run = pytest_run(20, 20);
        assert!(run.len() < super::super::output::DEFAULT_MAX_BYTES);
        let policy = BashOutput {
            pytest_digest: true,
            ..BashOutput::default()
        };
        let mut acc = policy.accumulator(true);
        acc.append(run.as_bytes());
        acc.finish();
        let (snap, text) = policy.finish(&mut acc);
        let text = text.expect("must digest");
        let path = snap
            .full_output_path
            .expect("digest must leave the full output readable");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), run);
        assert!(text.contains(&path), "{text}");
        assert!(text.contains("pytest output condensed"), "{text}");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn the_default_policy_never_digests() {
        let policy = BashOutput::default();
        let mut acc = policy.accumulator(true);
        acc.append(pytest_run(40, 30).as_bytes());
        acc.finish();
        let (snap, text) = policy.finish(&mut acc);
        assert!(text.is_none());
        if let Some(p) = snap.full_output_path {
            let _ = std::fs::remove_file(p);
        }
    }

    #[test]
    fn a_lower_cap_keeps_a_shorter_tail_and_says_which_limit() {
        let policy = BashOutput {
            max_bytes: 16 * 1024,
            ..BashOutput::default()
        };
        let mut acc = policy.accumulator(false);
        // 300 lines of 100 bytes: over the byte cap, well under the 2000-line cap.
        acc.append(format!("{}\n", "x".repeat(99)).repeat(300).as_bytes());
        acc.finish();
        let (snap, _) = policy.finish(&mut acc);
        assert!(snap.truncation.truncated);
        assert!(snap.content.len() <= 16 * 1024);
        let text = super::super::output::format_output(&snap, "");
        assert!(
            text.contains("(16.0KB limit)"),
            "{}",
            &text[text.len() - 200..]
        );
    }
}
