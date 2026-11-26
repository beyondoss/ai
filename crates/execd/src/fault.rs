//! Fault injection for the tests of a keyed start's claim (K10, K11), read in this one place.
//!
//! `EXECD_TEST_FAULT` names a file; while the file exists, each line `point=what` makes that point fail as
//! `what` says, and removing the file clears the fault, so a test can retry once it has cleared. Only a build with
//! `debug_assertions` reads the variable: a release build (what ships) has no knob.
//!
//! Points: `claim_save=full` (the claim's write fails with a full disk), `claim_save=eio` (it fails otherwise),
//! `after_spawn=exit` (execd exits between the spawn and the group's save, leaving the job's process running with
//! only its intent on disk), `attach_after_decide=<file>` ([`park`]: an attach that has chosen its offset and
//! registered its stream waits until the file goes), `attach_before_decide=<file>` (an attach that has looked its job
//! up waits before it locks it), `sweep_after_save=exit` (the retention exits between its ledger write and its
//! unlinks), `ledger_slow=<ms>` (every write of the jobs ledger costs this long, the way a slow disk does).

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

/// The longest [`park`] holds a request: longer than any test needs, short enough that a test that
/// forgets to clear the fault cannot hang a server.
pub(crate) const PARK_MAX: std::time::Duration = std::time::Duration::from_secs(5);

/// Hold this step while the fault file at `point` exists (bounded by [`PARK_MAX`]): a test's way to
/// stop a request between two steps it must not be split across, so the race the two steps could
/// lose is *forced* rather than hoped for. With no fault file this is one branch.
pub(crate) async fn park(point: &str) {
    let Some(hold) = at(point) else { return };
    crate::log_debug!("fault parked point={point}");
    let until = tokio::time::Instant::now() + PARK_MAX;
    while std::path::Path::new(&hold).exists() && tokio::time::Instant::now() < until {
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
    }
}
