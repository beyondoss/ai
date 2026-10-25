//! Startup timing instrumentation — pi's own `PI_TIMING=1`/`timings.ts`, for profiling where `run`/
//! `serve` actually spend time before the first turn (resource discovery, system-prompt assembly,
//! session open). Zero-cost when unset: every method short-circuits on the disabled flag before
//! touching the clock, so this is safe to sprinkle through a hot startup path without a config knob
//! per call site.

use std::time::{Duration, Instant};

/// Env var that turns this on. Unset (or anything but `"1"`) means every method here is a no-op.
const ENV_VAR: &str = "AI_AGENT_TIMING";

/// A running log of named checkpoints since the last one, printed to stderr on request.
pub struct StartupTiming {
    enabled: bool,
    /// Record marks even when [`ENV_VAR`] is unset, for a caller that reports them itself
    /// ([`Self::summary`]). Printing stays gated on `enabled`.
    record: bool,
    last: Instant,
    entries: Vec<(&'static str, Duration)>,
}

impl StartupTiming {
    /// Start a new timing run, checking [`ENV_VAR`] once. The first [`mark`](Self::mark) is measured
    /// from this call, not from process start — a caller wanting to include argument parsing/process
    /// setup should construct this as early as possible.
    pub fn new() -> Self {
        Self::with_enabled(std::env::var(ENV_VAR).as_deref() == Ok("1"))
    }

    /// Like [`new`](Self::new), but with the enabled flag given directly rather than read from the
    /// environment — split out so tests can exercise both branches without mutating process-global
    /// env state (unsafe to do from a test that may run in parallel with others reading the same var).
    fn with_enabled(enabled: bool) -> Self {
        Self {
            enabled,
            record: false,
            last: Instant::now(),
            entries: Vec::new(),
        }
    }

    /// Like [`new`](Self::new), but always recording: a service replica logs one line per session open
    /// from [`summary`](Self::summary), whether or not [`ENV_VAR`] is set. A mark is one clock read and
    /// one push onto a short vector.
    pub fn recording() -> Self {
        let mut t = Self::new();
        t.record = true;
        t
    }

    /// Record `label` as the elapsed time since the previous `mark` (or since [`new`](Self::new)).
    /// A no-op when disabled — doesn't even read the clock.
    pub fn mark(&mut self, label: &'static str) {
        if !self.enabled && !self.record {
            return;
        }
        let now = Instant::now();
        self.entries.push((label, now.duration_since(self.last)));
        self.last = now;
    }

    /// The marks as `label=ms` pairs, labels with underscores for spaces, for a log line.
    pub fn summary(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        for (label, dur) in &self.entries {
            if !out.is_empty() {
                out.push(' ');
            }
            let _ = write!(
                out,
                "{}={:.1}",
                label.replace(' ', "_"),
                dur.as_secs_f64() * 1e3
            );
        }
        out
    }

    /// Print every recorded checkpoint (plus a total) to stderr — never stdout, so this can't corrupt
    /// `run`'s streamed text or `serve`'s NDJSON protocol. A no-op when disabled or nothing was marked.
    pub fn print(&self) {
        if !self.enabled || self.entries.is_empty() {
            return;
        }
        eprintln!("\n--- Startup Timings ---");
        let mut total = Duration::ZERO;
        for (label, dur) in &self.entries {
            eprintln!("  {label}: {}ms", dur.as_millis());
            total += *dur;
        }
        eprintln!("  TOTAL: {}ms", total.as_millis());
        eprintln!("-----------------------\n");
    }
}

/// One line on stderr for a service session's open: `OPEN <kind> sid=<id> t=<epoch ms> <fields>
/// blk=<jobs in flight>/<of them queued>`. The last two numbers are the blocking pool's state at the
/// moment the line is written ([`crate::blocking::inflight`]); `fields` are `name=ms` phase times.
/// Joined on `sid` with what a client saw, they say which phase a slow open spent its time in.
pub fn open_line(kind: &str, sid: &str, fields: &str) {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    let (inflight, queued) = crate::blocking::inflight();
    eprintln!("OPEN {kind} sid={sid} t={t} {fields} blk={inflight}/{queued}");
}

impl Default for StartupTiming {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_records_nothing() {
        let mut t = StartupTiming::with_enabled(false);
        t.mark("a");
        t.mark("b");
        assert!(t.entries.is_empty());
    }

    #[test]
    fn enabled_records_marks_in_order() {
        let mut t = StartupTiming::with_enabled(true);
        t.mark("a");
        t.mark("b");
        assert_eq!(t.entries.len(), 2);
        assert_eq!(t.entries[0].0, "a");
        assert_eq!(t.entries[1].0, "b");
    }

    #[test]
    fn a_recording_timer_keeps_marks_for_its_summary() {
        let mut t = StartupTiming::with_enabled(false);
        t.record = true;
        t.mark("open persistence");
        t.mark("build agent");
        let s = t.summary();
        assert!(s.starts_with("open_persistence="), "{s}");
        assert!(s.contains(" build_agent="), "{s}");
    }

    #[test]
    fn print_is_a_no_op_when_disabled_or_empty() {
        // Nothing meaningful to assert on stderr output directly here, but this at least proves
        // `print` doesn't panic in either short-circuit branch.
        StartupTiming::with_enabled(false).print();
        StartupTiming::with_enabled(true).print(); // enabled, but no marks recorded yet
    }
}
