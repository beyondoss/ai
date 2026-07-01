//! The agent's coding tools — pi's tool set, ported.
//!
//! Each tool implements [`agent_core::Tool`]; [`default_registry`] assembles the set the agent
//! advertises to the model. The Beyond platform tools (fork/sync/logs) register here too once added.

use std::sync::Arc;

use agent_core::ToolRegistry;

pub mod bash;
pub mod beyond;
pub mod edit;
pub mod exec;
pub mod find;
pub mod grep;
pub mod ls;
pub mod output;
pub mod read;
pub mod write;

/// Overwrite `path` atomically: write a sibling temp file, then `rename` it over the target.
/// `rename(2)` is atomic within one filesystem, so a concurrent reader — or a crash mid-write — sees
/// either the original file or the fully-written one, never a half-written file. The temp file is a
/// sibling (same directory) so the rename stays on one filesystem. A bare `std::fs::write` truncates
/// in place and would leave a partial file if the process died between truncation and the last byte.
///
/// Shared by `write` and `edit`: both replace whole files and must not leave a corrupt intermediate
/// state that a later read (or `serve` reattach) would observe.
pub(crate) fn write_atomic(path: &str, content: &[u8]) -> std::io::Result<()> {
    let p = std::path::Path::new(path);
    let tmp = match p.file_name() {
        Some(name) => p.with_file_name(format!(".{}.tmp", name.to_string_lossy())),
        None => return Err(std::io::Error::other(format!("invalid path: {path}"))),
    };
    std::fs::write(&tmp, content)?;
    match std::fs::rename(&tmp, p) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = std::fs::remove_file(&tmp); // don't leave the temp behind on failure
            Err(e)
        }
    }
}

/// Canonicalize a path for use as a same-file grouping key (`Tool::write_target`): two same-turn calls
/// naming the same underlying file — `./foo.rs`, `foo.rs`, or a symlink alias — must land on one key so
/// the loop serializes them instead of racing two concurrent read-modify-writes on one file.
///
/// Real canonicalization (`std::fs::canonicalize`) resolves symlinks and `.`/`..` against the actual
/// filesystem, but fails with `NotFound` for a path that doesn't exist yet — exactly `write`'s common
/// case (creating a new file). The fallback below only resolves `.`/`..` *lexically* and makes the
/// result absolute (joined against the process cwd), so differently-spelled references to a
/// not-yet-created file still normalize to the same key; it just can't resolve a symlink it can't stat.
///
/// Never fails outright: a pathological input (e.g. one that can't even be joined) falls back to the
/// original string unchanged, which only degrades grouping (calls that should serialize might not) —
/// strictly no worse than today's un-canonicalized behavior, never a new correctness hazard.
pub(crate) fn canonical_write_target(path: &str) -> String {
    if let Ok(real) = std::fs::canonicalize(path) {
        return real.display().to_string();
    }
    let p = std::path::Path::new(path);
    let mut normalized = if p.is_absolute() {
        std::path::PathBuf::new()
    } else {
        std::env::current_dir().unwrap_or_default()
    };
    for component in p.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized.display().to_string()
}

/// The default tool set: pi's seven coding tools (read, write, edit, bash, ls, grep, find) plus the
/// Beyond platform tools (fork, sync, logs).
pub fn default_registry() -> ToolRegistry {
    default_registry_with(None)
}

/// Like [`default_registry`], overriding `bash`'s default timeout (applied when the model omits
/// `timeout_ms`) when `bash_timeout_ms` is `Some` — an operator-tunable knob (`--bash-timeout-ms`),
/// distinct from `default_registry`'s fixed default so callers that don't need the override (the
/// `tools` listing command, tests) don't have to pass one.
pub fn default_registry_with(bash_timeout_ms: Option<u64>) -> ToolRegistry {
    let mut reg = ToolRegistry::new();
    reg.register(Arc::new(read::Read));
    reg.register(Arc::new(write::Write));
    reg.register(Arc::new(edit::Edit));
    reg.register(Arc::new(ls::Ls));
    reg.register(Arc::new(grep::Grep));
    reg.register(Arc::new(find::Find));
    let bash = match bash_timeout_ms {
        Some(ms) => bash::Bash::real().with_default_timeout_ms(ms),
        None => bash::Bash::real(),
    };
    reg.register(Arc::new(bash));
    reg.register(Arc::new(beyond::Fork::real()));
    reg.register(Arc::new(beyond::Sync::real()));
    reg.register(Arc::new(beyond::Logs::real()));
    reg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_atomic_cleans_up_temp_on_rename_failure() {
        let dir = tempfile::tempdir().unwrap();
        // A regular file can't be `rename`d over an existing directory (EISDIR) — forces the rename
        // failure path without relying on permissions or a second filesystem.
        let target = dir.path().join("target");
        std::fs::create_dir(&target).unwrap();

        assert!(write_atomic(target.to_str().unwrap(), b"new content").is_err());

        // The sibling temp file must not survive a failed rename — `write_atomic` removes it
        // explicitly on the failure path rather than leaving litter behind.
        let temps: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(temps.is_empty(), "failed rename left a temp file behind");
        // The original target is untouched by the failed write.
        assert!(target.is_dir());
    }

    #[test]
    fn canonical_write_target_unifies_spellings_of_an_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().canonicalize().unwrap().join("foo.rs");
        std::fs::write(&real, b"fn main() {}").unwrap();

        let dotted = dir.path().join("./foo.rs");
        let via_parent = dir.path().join("sub/../foo.rs");
        std::fs::create_dir_all(dir.path().join("sub")).unwrap();

        let a = canonical_write_target(real.to_str().unwrap());
        let b = canonical_write_target(dotted.to_str().unwrap());
        let c = canonical_write_target(via_parent.to_str().unwrap());
        assert_eq!(a, b, "absolute vs `./`-prefixed spelling must match");
        assert_eq!(a, c, "a path through `..` must resolve to the same key");
    }

    #[test]
    fn canonical_write_target_falls_back_to_lexical_normalization_when_the_file_is_new() {
        // `write` routinely targets a path that doesn't exist yet — `canonicalize` fails (NotFound),
        // so the grouping key must still unify spellings without touching the filesystem. Uses
        // absolute paths (rather than `std::env::set_current_dir`, which is process-global and unsafe
        // to mutate from a test that may run in parallel with others) so the relative-path branch of
        // `canonical_write_target` isn't exercised here, only the `.`/`..` lexical resolution.
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().canonicalize().unwrap();

        let a = canonical_write_target(base.join("brand-new.rs").to_str().unwrap());
        let b = canonical_write_target(base.join("./brand-new.rs").to_str().unwrap());
        let c = canonical_write_target(base.join("sub/../brand-new.rs").to_str().unwrap());

        assert_eq!(a, b);
        assert_eq!(a, c);
        assert!(
            std::path::Path::new(&a).is_absolute(),
            "fallback must still produce an absolute key: {a}"
        );
    }

    #[test]
    fn default_registry_has_coding_and_beyond_tools() {
        let reg = default_registry();
        // pi's coding tools …
        for name in ["read", "write", "edit", "bash", "ls", "grep", "find"] {
            assert!(reg.get(name).is_some(), "missing coding tool: {name}");
        }
        // … plus the Beyond platform tools.
        for name in ["fork", "sync", "logs"] {
            assert!(reg.get(name).is_some(), "missing beyond tool: {name}");
        }
        assert_eq!(reg.len(), 10);
    }

    #[test]
    fn default_registry_with_none_matches_default_registry() {
        // `default_registry()` must genuinely delegate to `default_registry_with(None)`, not just
        // happen to look similar — same tool set either way.
        assert_eq!(default_registry().len(), default_registry_with(None).len());
        for name in [
            "read", "write", "edit", "bash", "ls", "grep", "find", "fork", "sync", "logs",
        ] {
            assert!(default_registry_with(None).get(name).is_some());
        }
    }
}
