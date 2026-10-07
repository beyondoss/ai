//! Shared helpers for this crate's unit tests.

use std::path::Path;

/// What makes a directory "inside a project" to the code under test: an enclosing git repo, a
/// context file, or a project skills root anywhere above it.
const PROJECT_MARKERS: [&str; 4] = [".git", "CLAUDE.md", "AGENTS.md", ".agents"];

/// Whether `dir` or any ancestor carries a [project marker](PROJECT_MARKERS).
fn inside_a_project(dir: &Path) -> bool {
    dir.ancestors()
        .any(|a| PROJECT_MARKERS.iter().any(|m| a.join(m).exists()))
}

/// A temp directory that no ancestor walk can escape into a real project.
///
/// Tests of "no enclosing repo" / "no context file above" behaviour walk to the filesystem root, so
/// a plain `tempfile::tempdir()` makes them depend on where `TMPDIR` points: under a git checkout or
/// a `CLAUDE.md` (a worktree's own `target/tmp`, say) they find the surrounding project and fail.
/// This picks the first candidate root with no project above it — `TMPDIR` itself when it is clean.
///
/// # Panics
/// When every candidate root sits inside a project, naming them, rather than letting the test
/// pass or fail for a reason that has nothing to do with it.
pub(crate) fn isolated_tempdir() -> tempfile::TempDir {
    let candidates = [
        std::env::temp_dir(),
        "/tmp".into(),
        "/var/tmp".into(),
        "/dev/shm".into(),
    ];
    for root in &candidates {
        let Ok(root) = root.canonicalize() else {
            continue;
        };
        if inside_a_project(&root) {
            continue;
        }
        if let Ok(dir) = tempfile::tempdir_in(&root) {
            return dir;
        }
    }
    panic!(
        "no temp root outside a project (git repo, CLAUDE.md, AGENTS.md or .agents above it) among \
         {candidates:?}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_isolated_tempdir_has_no_project_above_it() {
        let dir = isolated_tempdir();
        assert!(!inside_a_project(dir.path()), "{}", dir.path().display());
    }

    #[test]
    fn a_marker_anywhere_above_counts() {
        let dir = isolated_tempdir();
        let deep = dir.path().join("a/b");
        std::fs::create_dir_all(&deep).unwrap();
        std::fs::write(dir.path().join("CLAUDE.md"), "x").unwrap();
        assert!(inside_a_project(&deep));
    }
}
