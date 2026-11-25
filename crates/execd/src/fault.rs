//! Fault injection for the tests of a keyed start's claim (K10, K11), read in this one place.
//!
//! `EXECD_TEST_FAULT` names a file; while the file exists, each line `point=what` makes that point fail as
//! `what` says, and removing the file clears the fault, so a test can retry once it has cleared. Only a build with
//! `debug_assertions` reads the variable: a release build (what ships) has no knob.
//!
//! Points: `claim_save=full` (the claim's write fails with a full disk), `claim_save=eio` (it fails otherwise),
//! `after_spawn=exit` (execd exits between the spawn and the group's save, leaving the job's process running with
//! only its intent on disk).

use std::path::PathBuf;
use std::sync::OnceLock;

/// What the fault file says for `point`, if a fault is set there.
pub(crate) fn at(point: &str) -> Option<String> {
    if !cfg!(debug_assertions) {
        return None;
    }
    static FILE: OnceLock<Option<PathBuf>> = OnceLock::new();
    let file = FILE
        .get_or_init(|| std::env::var_os("EXECD_TEST_FAULT").map(PathBuf::from))
        .as_ref()?;
    let text = std::fs::read_to_string(file).ok()?;
    text.lines()
        .find_map(|l| l.strip_prefix(point)?.strip_prefix('='))
        .map(str::to_string)
}
