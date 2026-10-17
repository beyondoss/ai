//! Shared path-normalization helpers: canonicalize as much of a path as exists and append the rest
//! ([`resolve_nearest`]) — never fails, unlike a bare [`Path::canonicalize`], and never leaves a
//! symlinked parent of a not-yet-created path unresolved.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// The one key a path resolves to, whether or not it exists yet — see [`resolve_nearest`].
///
/// A path passed before the directory exists (a pre-trust grant, or a discovery root not yet
/// created) still resolves its **nearest existing ancestor**: pre-trusting `/tmp/newproj` on macOS,
/// where `/tmp` is a symlink to `/private/tmp`, must store the key the directory will have once it is
/// created, not the spelling it was typed with — or the grant silently never takes effect.
pub(crate) fn resolved_path(path: &Path) -> PathBuf {
    resolve_nearest(path)
}

/// How many dangling symlinks [`resolve_nearest`] follows before giving up — the kernel's own
/// `ELOOP` bound, so a link cycle terminates rather than recursing forever.
const MAX_LINK_HOPS: usize = 40;

/// Resolve `path` against the real filesystem as far as it exists, then append the components that
/// don't exist yet.
///
/// `canonicalize` refuses a path that doesn't exist — which is precisely the path a `write`, a
/// pre-trust grant or a deny-list check is usually about. Falling back to the path *as typed* leaves
/// every symlinked parent unresolved: `sl/new.env` (where `sl -> secrets`) and `secrets/new.env`
/// become two keys for one file, so a `--deny-path '**/secrets/**'` misses the first, a remembered
/// approval doesn't recognize it, and two writes to it don't serialize.
///
/// This walks the path one component at a time. While components exist, each step is canonicalized,
/// so every symlink along the way is resolved by the kernel. From the first missing component on,
/// nothing below it can exist, so the rest is appended verbatim. `..` is folded **after**
/// resolution, never before: `sl/../x` is the parent of `sl`'s *target*, which a lexical fold (`./x`)
/// gets wrong. A `..` over a missing component simply drops it — a component that doesn't exist
/// can't be a symlink. A dangling symlink is followed to its target (the file a write through it
/// creates), bounded by [`MAX_LINK_HOPS`].
///
/// Relative paths are joined onto the process cwd first. Never fails.
pub(crate) fn resolve_nearest(path: &Path) -> PathBuf {
    resolve_nearest_hops(path, 0)
}

fn resolve_nearest_hops(path: &Path, hops: usize) -> PathBuf {
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        match std::env::current_dir() {
            Ok(cwd) => cwd.join(path),
            Err(_) => path.to_path_buf(),
        }
    };
    // Fast path: the whole thing exists.
    if let Ok(real) = joined.canonicalize() {
        return real;
    }
    let mut cur = PathBuf::new();
    // How many trailing components of `cur` are known not to exist.
    let mut missing = 0usize;
    for component in joined.components() {
        match component {
            std::path::Component::Prefix(_) | std::path::Component::RootDir => {
                cur.push(component.as_os_str());
            }
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                // `cur`'s existing part is canonical, so its lexical parent is its real parent.
                cur.pop();
                missing = missing.saturating_sub(1);
            }
            std::path::Component::Normal(name) => {
                cur.push(name);
                if missing > 0 {
                    missing += 1;
                    continue;
                }
                let Ok(meta) = std::fs::symlink_metadata(&cur) else {
                    missing = 1;
                    continue;
                };
                if let Ok(real) = cur.canonicalize() {
                    cur = real;
                } else if meta.file_type().is_symlink()
                    && hops < MAX_LINK_HOPS
                    && let Ok(target) = std::fs::read_link(&cur)
                {
                    // A dangling link: what a write through it creates is its target.
                    let base = cur.parent().map(Path::to_path_buf).unwrap_or_default();
                    cur = resolve_nearest_hops(&base.join(target), hops + 1);
                    missing = missing_tail(&cur);
                }
            }
        }
    }
    cur
}

/// How many trailing components of `path` don't exist.
fn missing_tail(path: &Path) -> usize {
    path.ancestors()
        .take_while(|a| !a.as_os_str().is_empty() && std::fs::symlink_metadata(a).is_err())
        .count()
}

/// Whether name lookups on the volume holding `path` ignore letter case (macOS APFS/HFS+ by default,
/// Windows NTFS, an ext4 `casefold` directory) — **probed**, never assumed from the OS: a Mac can
/// mount a case-sensitive volume, and a Linux box can have a case-insensitive one.
///
/// The probe needs no writes: take the nearest existing ancestor of `path`, find an ancestor whose
/// name has a letter, flip that name's case and look it up in the same directory. The same file back
/// means the directory ignores case. Only ancestors on the same device as the nearest existing entry
/// are probed, so a lookup on the root volume can't answer for a mounted one. With nothing
/// probe-able (`/` alone), the answer is `false` — case-sensitive, which never merges two distinct
/// files into one key.
pub(crate) fn is_case_insensitive_at(path: &Path) -> bool {
    let Some(existing) = path
        .ancestors()
        .find(|a| !a.as_os_str().is_empty() && std::fs::symlink_metadata(a).is_ok())
    else {
        return false;
    };
    let Some(device) = device_of(existing) else {
        return false;
    };
    for candidate in existing.ancestors() {
        let (Some(parent), Some(name)) = (candidate.parent(), candidate.file_name()) else {
            break;
        };
        if device_of(parent) != Some(device) {
            break;
        }
        let Some(name) = name.to_str() else { continue };
        let flipped: String = name
            .chars()
            .map(|c| {
                if c.is_lowercase() {
                    c.to_uppercase().next().unwrap_or(c)
                } else {
                    c.to_lowercase().next().unwrap_or(c)
                }
            })
            .collect();
        if flipped == name {
            continue;
        }
        return same_file(candidate, &parent.join(flipped));
    }
    false
}

/// Fold a resolved path into the string every spelling of one file agrees on: as-is on a
/// case-sensitive volume, lowercased on a case-insensitive one.
pub(crate) fn comparison_key(path: &Path, case_insensitive: bool) -> String {
    let s = path.display().to_string();
    if case_insensitive {
        s.to_lowercase()
    } else {
        s
    }
}

#[cfg(unix)]
fn device_of(path: &Path) -> Option<u64> {
    use std::os::unix::fs::MetadataExt;
    std::fs::symlink_metadata(path).ok().map(|m| m.dev())
}

#[cfg(not(unix))]
fn device_of(_path: &Path) -> Option<u64> {
    Some(0)
}

#[cfg(unix)]
fn same_file(a: &Path, b: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    match (std::fs::symlink_metadata(a), std::fs::symlink_metadata(b)) {
        (Ok(x), Ok(y)) => x.dev() == y.dev() && x.ino() == y.ino(),
        _ => false,
    }
}

#[cfg(not(unix))]
fn same_file(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(x), Ok(y)) => x == y,
        _ => false,
    }
}

/// Push `(dir, tag)` onto `roots` only if `dir`'s canonical form hasn't already been added (to `roots`
/// or any other discovery root tracked in `seen`) — a root reached by two different paths (a symlink, a
/// relative-vs-absolute spelling, `cwd` itself equaling a global root in some deployment) must only be
/// scanned once, or shared by `skills.rs`/`prompts.rs`'s discovery-root building: scanning the same real
/// directory twice would double-count its contents and self-collide every name in it against a
/// phantom duplicate rather than a genuine collision. Pi itself doesn't dedupe at the root-list level
/// either (only file-level, in `skills.ts`'s own `loadSkills`/`realPathSet`); this is a Beyond-specific
/// hardening. `tag` rides along uninspected — `skills.rs`/`prompts.rs` both pass their own discovery-root
/// category (`"user"`/`"project"`) here, for `Skill::scope`/`PromptTemplate::scope` (Task #39) — so each
/// root's own scope travels with it instead of needing a second, separately-indexed data structure.
pub(crate) fn push_unique_scoped_root<T>(
    roots: &mut Vec<(PathBuf, T)>,
    seen: &mut HashSet<PathBuf>,
    dir: PathBuf,
    tag: T,
) {
    if seen.insert(resolved_path(&dir)) {
        roots.push((dir, tag));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_nearest_folds_dot_components_and_trailing_separators_of_a_missing_path() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().canonicalize().unwrap();
        let want = base.join("new/c");
        for spelling in ["new/b/../c", "new/./c", "new/c/", "./new/c"] {
            assert_eq!(resolve_nearest(&base.join(spelling)), want, "{spelling}");
        }
        // A `..` at (or above) the root is dropped, not pushed literally.
        assert_eq!(
            resolve_nearest(Path::new("/../../does-not-exist-nearest")),
            PathBuf::from("/does-not-exist-nearest")
        );
    }

    #[test]
    fn resolve_nearest_joins_a_relative_path_onto_the_real_current_directory() {
        let real_cwd = std::env::current_dir().unwrap().canonicalize().unwrap();
        assert_eq!(
            resolve_nearest(Path::new("some/relative/path-nearest")),
            real_cwd.join("some/relative/path-nearest")
        );
    }

    #[test]
    fn resolved_path_keys_a_nonexistent_path_under_its_resolved_parent() {
        // `canonicalize` fails outright (ENOENT) on a path that doesn't exist yet; the key is still
        // the one it will have once created.
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("does-not-exist-yet");
        assert_eq!(
            resolved_path(&missing),
            dir.path()
                .canonicalize()
                .unwrap()
                .join("does-not-exist-yet")
        );
    }

    #[test]
    #[cfg(unix)]
    fn resolve_nearest_resolves_a_symlinked_parent_of_a_path_that_does_not_exist_yet() {
        // macOS's `/tmp -> /private/tmp`, in miniature: a new file's key must be the one it will have
        // once it exists, so a deny glob, an approval key and a trust grant all see one file.
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().canonicalize().unwrap().join("secrets");
        std::fs::create_dir(&real).unwrap();
        std::os::unix::fs::symlink(&real, dir.path().join("sl")).unwrap();
        assert_eq!(
            resolve_nearest(&dir.path().join("sl/new/deeper.env")),
            real.join("new/deeper.env")
        );
        // Once it exists, the key is unchanged.
        std::fs::create_dir_all(real.join("new")).unwrap();
        std::fs::write(real.join("new/deeper.env"), "x").unwrap();
        assert_eq!(
            resolve_nearest(&dir.path().join("sl/new/deeper.env")),
            real.join("new/deeper.env")
        );
    }

    #[test]
    #[cfg(unix)]
    fn resolve_nearest_folds_dot_dot_after_resolving_symlinks_not_before() {
        // `sl/../x` is the parent of `sl`'s *target*; a lexical fold would say `./x`.
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().canonicalize().unwrap();
        std::fs::create_dir_all(base.join("a/inner")).unwrap();
        std::os::unix::fs::symlink(base.join("a/inner"), base.join("sl")).unwrap();
        assert_eq!(
            resolve_nearest(&base.join("sl/../new.txt")),
            base.join("a/new.txt")
        );
        // A `..` over a missing component just drops it.
        assert_eq!(
            resolve_nearest(&base.join("missing/../sl/new.txt")),
            base.join("a/inner/new.txt")
        );
    }

    #[test]
    #[cfg(unix)]
    fn resolve_nearest_follows_a_dangling_symlink_to_the_file_a_write_would_create() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().canonicalize().unwrap();
        std::fs::create_dir(base.join("secrets")).unwrap();
        std::os::unix::fs::symlink(base.join("secrets/x.env"), base.join("alias")).unwrap();
        assert_eq!(
            resolve_nearest(&base.join("alias")),
            base.join("secrets/x.env")
        );
        // A cycle terminates.
        std::os::unix::fs::symlink(base.join("loop2"), base.join("loop1")).unwrap();
        std::os::unix::fs::symlink(base.join("loop1"), base.join("loop2")).unwrap();
        let _ = resolve_nearest(&base.join("loop1/x"));
    }

    #[test]
    fn comparison_key_folds_case_only_on_a_case_insensitive_volume() {
        let p = Path::new("/Vol/SECRETS/New2.env");
        assert_eq!(comparison_key(p, true), "/vol/secrets/new2.env");
        assert_eq!(
            comparison_key(p, true),
            comparison_key(Path::new("/vol/Secrets/new2.ENV"), true)
        );
        assert_eq!(comparison_key(p, false), "/Vol/SECRETS/New2.env");
    }

    #[test]
    fn case_probe_agrees_with_what_the_volume_actually_does() {
        // Ground truth from the filesystem itself: create a name, look up its case-flipped twin.
        let dir = tempfile::tempdir().unwrap();
        let probe = dir.path().join("CaseProbe");
        std::fs::create_dir(&probe).unwrap();
        let actually_insensitive = dir.path().join("cASEpROBE").exists();
        assert_eq!(
            is_case_insensitive_at(&probe.join("not-yet.txt")),
            actually_insensitive
        );
    }

    #[test]
    fn resolved_path_canonicalizes_a_path_that_does_exist() {
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().canonicalize().unwrap();
        assert_eq!(resolved_path(dir.path()), real);
    }
}
