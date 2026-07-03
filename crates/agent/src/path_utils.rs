//! Shared path-normalization helpers: canonicalize when the filesystem lets you, fall back to a
//! lexically-normalized absolute path when it can't (a not-yet-created directory, or a discovery root
//! that's simply absent this run) — never fails, unlike a bare [`Path::canonicalize`].

use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Absolutize-and-lexically-normalize `path` first (no filesystem access beyond reading the cwd, so
/// this never fails even for a path that doesn't exist yet), then try to canonicalize on top of that —
/// falling back to the already-absolute, already-normalized form if *that* fails.
///
/// Matches pi's own `resolvePath` → `canonicalizePath` order (`trust-manager.ts`'s `normalizeCwd`):
/// never canonicalize a path that might still be relative or carry a trailing separator, because the
/// fallback on failure needs to already be a clean, consistent key. Without the absolutize-first step,
/// a path passed before the directory exists (a pre-trust grant, or a discovery root not yet created)
/// would key on the literal relative/trailing-slash string, since `canonicalize()` fails outright on a
/// nonexistent path (`ENOENT`).
pub(crate) fn resolved_path(path: &Path) -> PathBuf {
    let absolutized = absolutize(path);
    absolutized.canonicalize().unwrap_or(absolutized)
}

/// Push `dir` onto `roots` only if its canonical form hasn't already been added (to `roots` or any
/// other discovery root tracked in `seen`) — a root reached by two different paths (a symlink, a
/// relative-vs-absolute spelling, `cwd` itself equaling a global root in some deployment) must only be
/// scanned once, or shared by `skills.rs`/`prompts.rs`'s discovery-root building: scanning the same real
/// directory twice would double-count its contents and self-collide every name in it against a
/// phantom duplicate rather than a genuine collision. Pi itself doesn't dedupe at the root-list level
/// either (only file-level, in `skills.ts`'s own `loadSkills`/`realPathSet`); this is a Beyond-specific
/// hardening.
pub(crate) fn push_unique_root(
    roots: &mut Vec<PathBuf>,
    seen: &mut HashSet<PathBuf>,
    dir: PathBuf,
) {
    if seen.insert(resolved_path(&dir)) {
        roots.push(dir);
    }
}

/// Make `path` absolute (joining it onto the current working directory if it's relative) and
/// lexically normalize it (collapse `.`/`..` components and a trailing separator) — no filesystem
/// access beyond reading the cwd itself, so this works even when `path` doesn't exist yet. Matches
/// Node's `path.resolve` semantics pi's own `resolvePath` builds on.
pub(crate) fn absolutize(path: &Path) -> PathBuf {
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    };
    lexically_normalize(&joined)
}

/// Collapse `.`/`..` path components without touching the filesystem — unlike [`Path::canonicalize`],
/// this never fails and never follows symlinks. A `..` at (or above) the root is dropped rather than
/// pushed literally, matching Node's `path.resolve` (which never lets `..` escape above the root) —
/// correct here because [`absolutize`] only ever calls this on an already-absolute path.
fn lexically_normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if matches!(
                    out.components().next_back(),
                    Some(std::path::Component::Normal(_))
                ) {
                    out.pop();
                }
            }
            other => out.push(other),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lexically_normalize_collapses_dot_and_dot_dot_components() {
        assert_eq!(
            lexically_normalize(Path::new("/a/b/../c")),
            PathBuf::from("/a/c")
        );
        assert_eq!(
            lexically_normalize(Path::new("/a/./b")),
            PathBuf::from("/a/b")
        );
        assert_eq!(
            lexically_normalize(Path::new("/a/b/")),
            PathBuf::from("/a/b")
        );
        // A `..` at (or above) the root is dropped, not pushed literally.
        assert_eq!(
            lexically_normalize(Path::new("/../../a")),
            PathBuf::from("/a")
        );
    }

    #[test]
    fn absolutize_joins_a_relative_path_onto_the_real_current_directory() {
        let real_cwd = std::env::current_dir().unwrap();
        let expected = lexically_normalize(&real_cwd.join("some/relative/path"));
        assert_eq!(absolutize(Path::new("some/relative/path")), expected);
    }

    #[test]
    fn absolutize_leaves_an_already_absolute_path_alone_but_still_normalizes_it() {
        assert_eq!(absolutize(Path::new("/a/b/../c/")), PathBuf::from("/a/c"));
    }

    #[test]
    fn resolved_path_falls_back_to_the_absolutized_form_for_a_nonexistent_path() {
        // `canonicalize` fails outright (ENOENT) on a path that doesn't exist yet — the
        // absolutize-and-normalize step first is what makes the fallback still a clean, consistent key
        // rather than a literal relative/trailing-slash string.
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("does-not-exist-yet");
        assert_eq!(resolved_path(&missing), absolutize(&missing));
    }

    #[test]
    fn resolved_path_canonicalizes_a_path_that_does_exist() {
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().canonicalize().unwrap();
        assert_eq!(resolved_path(dir.path()), real);
    }
}
