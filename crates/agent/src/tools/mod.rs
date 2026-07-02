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

/// Normalize a filesystem tool's `path` argument the way `bash` gets for free from the real shell it
/// spawns: expand a leading `~` or `~/` to the user's home directory, fold non-ASCII Unicode space
/// characters (a pasted non-breaking space, common when a path is copied from a terminal or rich-text
/// source) to a plain ASCII one, and strip a leading `@` (pi's own `@file` CLI convention, occasionally
/// forwarded verbatim by a caller that composed the argument from a `run --json`-style transcript).
///
/// Every filesystem tool (`read`/`write`/`edit`/`grep`/`find`/`ls`) calls this on its path/root
/// argument before any `std::fs`/walk call. Without it, `~/notes.md` behaved inconsistently within the
/// very same turn: `bash: cat ~/notes.md` succeeds (the spawned shell expands `~` itself), while
/// `read({"path": "~/notes.md"})` failed with a confusing `ENOENT` — `std::fs` has no shell and treats
/// `~` as a literal directory name.
///
/// Falls back to returning `path` unchanged if `$HOME` isn't set (or is empty) rather than erroring —
/// consistent with every other tool here, which surface a filesystem error at the point of actual use
/// rather than pre-validating input that might still resolve to something real.
pub(crate) fn normalize_path(path: &str) -> String {
    let path = path.strip_prefix('@').unwrap_or(path);
    let folded: String = path
        .chars()
        .map(|c| {
            if c != ' ' && c != '\t' && c != '\n' && c != '\r' && c.is_whitespace() {
                ' '
            } else {
                c
            }
        })
        .collect();
    expand_tilde(&folded, home_dir().as_deref())
}

fn home_dir() -> Option<String> {
    std::env::var("HOME").ok().filter(|h| !h.is_empty())
}

/// Split from [`normalize_path`] so tilde-expansion logic is unit-testable without mutating the
/// process's `$HOME` — unsafe to do from a test that may run in parallel with others reading it (same
/// concern as `resources::tz_string_offset`). `home: None` (unset or empty `$HOME`) leaves `path`
/// unchanged rather than erroring, consistent with every tool here surfacing a filesystem error at the
/// point of actual use rather than pre-validating input that might still resolve to something real.
fn expand_tilde(path: &str, home: Option<&str>) -> String {
    let Some(home) = home else {
        return path.to_string();
    };
    if path == "~" {
        return home.to_string();
    }
    match path.strip_prefix("~/") {
        Some(rest) => format!("{}/{rest}", home.trim_end_matches('/')),
        None => path.to_string(),
    }
}

/// Overwrite `path` atomically: write a sibling temp file, then `rename` it over the target.
/// `rename(2)` is atomic within one filesystem, so a concurrent reader — or a crash mid-write — sees
/// either the original file or the fully-written one, never a half-written file. The temp file is a
/// sibling (same directory) so the rename stays on one filesystem. A bare `std::fs::write` truncates
/// in place and would leave a partial file if the process died between truncation and the last byte.
///
/// The temp file's name carries an unpredictable suffix ([`temp_suffix`]) and is opened with
/// `create_new` rather than plain `create`: a deterministic `.foo.tmp` name would let anything able to
/// plant a symlink in the same directory (another tool call, a prior turn) redirect this write to an
/// arbitrary path — plant `.foo.tmp -> ~/.ssh/authorized_keys`, then wait for a `write`/`edit` on
/// `foo`. `create_new` refuses to open through an existing path (including a symlink) instead of
/// following it, closing that TOCTOU window.
///
/// If `path` already exists, the temp file is created with its permission bits rather than the process
/// umask default — `rename(2)` swaps the whole directory entry, mode included, so a freshly-created
/// temp file would otherwise silently downgrade e.g. a `chmod 600` file to world-readable on every
/// edit.
///
/// Shared by `write` and `edit`: both replace whole files and must not leave a corrupt intermediate
/// state that a later read (or `serve` reattach) would observe.
pub(crate) fn write_atomic(path: &str, content: &[u8]) -> std::io::Result<()> {
    use std::io::Write as _;

    let p = std::path::Path::new(path);
    let name = match p.file_name() {
        Some(name) => name,
        None => return Err(std::io::Error::other(format!("invalid path: {path}"))),
    };

    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
        if let Ok(meta) = std::fs::metadata(p) {
            opts.mode(meta.permissions().mode());
        }
    }

    let tmp = p.with_file_name(format!(".{}.tmp.{}", name.to_string_lossy(), temp_suffix()));
    let mut f = opts.open(&tmp)?;
    if let Err(e) = f.write_all(content) {
        drop(f);
        let _ = std::fs::remove_file(&tmp); // don't leave the temp behind on a mid-write failure
        return Err(e);
    }
    drop(f);

    match std::fs::rename(&tmp, p) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = std::fs::remove_file(&tmp); // don't leave the temp behind on failure
            Err(e)
        }
    }
}

/// An unpredictable per-call suffix for [`write_atomic`]'s temp file name, so it can't be preplanted
/// under a symlink before the call happens. Same salt-plus-counter construction as
/// `session_store::new_id`: `RandomState` draws OS entropy once per process, and a monotonic counter
/// keeps same-process calls distinct — no need for a `rand`/`uuid` dependency just for this.
fn temp_suffix() -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::BuildHasher;
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering};

    static SALT: OnceLock<u64> = OnceLock::new();
    let salt = *SALT.get_or_init(|| RandomState::new().hash_one(0xA70A1Cu64));
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    format!("{salt:x}{seq:x}")
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
    default_registry_with(None, None)
}

/// Like [`default_registry`], overriding `bash`'s default timeout (applied when the model omits
/// `timeout_ms`) when `bash_timeout_ms` is `Some`, and/or which shell it runs commands through when
/// `bash_shell_path` is `Some` — operator-tunable knobs (`--bash-timeout-ms`/`--bash-shell-path`),
/// distinct from `default_registry`'s fixed defaults so callers that don't need either override (the
/// `tools` listing command, tests) don't have to pass one. `bash_shell_path` is trusted as already
/// validated (see `--bash-shell-path` in `main.rs`, checked once at CLI-argument time) rather than
/// re-checked on every rebuild this feeds (`set_model`/`set_thinking` each rebuild the registry).
pub fn default_registry_with(
    bash_timeout_ms: Option<u64>,
    bash_shell_path: Option<&str>,
) -> ToolRegistry {
    let mut reg = ToolRegistry::new();
    reg.register(Arc::new(read::Read));
    reg.register(Arc::new(write::Write));
    reg.register(Arc::new(edit::Edit));
    reg.register(Arc::new(ls::Ls));
    reg.register(Arc::new(grep::Grep));
    reg.register(Arc::new(find::Find));
    let mut bash = match bash_timeout_ms {
        Some(ms) => bash::Bash::real().with_default_timeout_ms(ms),
        None => bash::Bash::real(),
    };
    if let Some(path) = bash_shell_path {
        bash = bash.with_shell_path(path);
    }
    reg.register(Arc::new(bash));
    reg.register(Arc::new(beyond::Fork::real()));
    reg.register(Arc::new(beyond::Sync::real()));
    reg.register(Arc::new(beyond::Logs::real()));
    reg
}

/// Restrict a registry to an allow-list, a deny-list, or nothing at all — the CLI/RPC surface for
/// scoping an agent's capabilities (e.g. a read-only reviewer with no `bash`/`edit`/`write`), which
/// otherwise has no way to run with less than the full default tool set. `no_tools` wins outright
/// (checked first); otherwise `tools` (if given) narrows to exactly those names, and `exclude` (if
/// given) drops names from whatever remains — so `tools` + `exclude` can combine to carve one tool out
/// of an allow-list, though picking just one of the three is the common case. Unknown names in either
/// list are silently ignored (there's nothing to remove/keep that isn't already there), matching
/// `ToolRegistry::retain`'s set semantics rather than erroring on a typo.
pub fn apply_filter(
    reg: &mut ToolRegistry,
    tools: Option<&[String]>,
    exclude: Option<&[String]>,
    no_tools: bool,
) {
    if no_tools {
        reg.retain(|_| false);
        return;
    }
    if let Some(allow) = tools {
        reg.retain(|name| allow.iter().any(|a| a == name));
    }
    if let Some(deny) = exclude {
        reg.retain(|name| !deny.iter().any(|d| d == name));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_tilde_expands_bare_tilde_to_home() {
        assert_eq!(expand_tilde("~", Some("/home/jared")), "/home/jared");
    }

    #[test]
    fn expand_tilde_expands_tilde_slash_prefix() {
        assert_eq!(
            expand_tilde("~/notes.md", Some("/home/jared")),
            "/home/jared/notes.md"
        );
    }

    #[test]
    fn expand_tilde_does_not_touch_a_tilde_in_the_middle_of_a_path() {
        // Only a *leading* `~`/`~/` is a home-directory reference — `foo/~bar` is a literal filename.
        assert_eq!(expand_tilde("foo/~bar", Some("/home/jared")), "foo/~bar");
    }

    #[test]
    fn expand_tilde_leaves_path_unchanged_when_home_is_unavailable() {
        assert_eq!(expand_tilde("~/notes.md", None), "~/notes.md");
    }

    #[test]
    fn expand_tilde_trims_a_trailing_slash_on_home_before_joining() {
        assert_eq!(
            expand_tilde("~/notes.md", Some("/home/jared/")),
            "/home/jared/notes.md"
        );
    }

    #[test]
    fn normalize_path_strips_a_leading_at_prefix() {
        assert_eq!(normalize_path("@/etc/hosts"), "/etc/hosts");
    }

    #[test]
    fn normalize_path_folds_a_non_breaking_space_to_a_plain_space() {
        // U+00A0 NBSP — a common artifact of copy-pasting a path from a terminal or rich-text source.
        assert_eq!(normalize_path("foo\u{00A0}bar.txt"), "foo bar.txt");
    }

    #[test]
    fn normalize_path_is_a_no_op_for_an_ordinary_absolute_path() {
        assert_eq!(normalize_path("/etc/hosts"), "/etc/hosts");
    }

    #[test]
    fn write_atomic_cleans_up_temp_on_rename_failure() {
        let dir = tempfile::tempdir().unwrap();
        // A regular file can't be `rename`d over an existing directory (EISDIR) — forces the rename
        // failure path without relying on permissions or a second filesystem.
        let target = dir.path().join("target");
        std::fs::create_dir(&target).unwrap();

        assert!(write_atomic(target.to_str().unwrap(), b"new content").is_err());

        // The sibling temp file must not survive a failed rename — `write_atomic` removes it
        // explicitly on the failure path rather than leaving litter behind. `target` (the directory)
        // should be the only entry left.
        let entries: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(
            entries.len(),
            1,
            "failed rename left extra files behind: {:?}",
            entries.iter().map(|e| e.file_name()).collect::<Vec<_>>()
        );
        // The original target is untouched by the failed write.
        assert!(target.is_dir());
    }

    #[test]
    #[cfg(unix)]
    fn write_atomic_does_not_follow_a_symlink_planted_at_the_old_deterministic_temp_name() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("secret.txt");
        std::fs::write(&target, b"original").unwrap();

        // Simulate a symlink preplanted at the old, guessable `.{name}.tmp` path, pointing
        // somewhere the write must never reach.
        let decoy = dir.path().join("decoy-target");
        let planted = dir.path().join(".secret.txt.tmp");
        std::os::unix::fs::symlink(&decoy, &planted).unwrap();

        write_atomic(target.to_str().unwrap(), b"new content").unwrap();

        assert!(!decoy.exists(), "write_atomic followed the planted symlink");
        assert_eq!(std::fs::read_link(&planted).unwrap(), decoy);
        assert_eq!(std::fs::read(&target).unwrap(), b"new content");
    }

    #[test]
    #[cfg(unix)]
    fn write_atomic_preserves_existing_file_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("secret.env");
        std::fs::write(&target, b"SECRET=1").unwrap();
        std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o600)).unwrap();

        write_atomic(target.to_str().unwrap(), b"SECRET=2").unwrap();

        let mode = std::fs::metadata(&target).unwrap().permissions().mode() & 0o777;
        assert_eq!(
            mode, 0o600,
            "write_atomic silently changed the file's permissions"
        );
        assert_eq!(std::fs::read(&target).unwrap(), b"SECRET=2");
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
    fn apply_filter_no_tools_wins_outright() {
        let mut reg = default_registry();
        apply_filter(
            &mut reg,
            Some(&["read".to_string()]),
            Some(&["write".to_string()]),
            true,
        );
        assert!(reg.is_empty());
    }

    #[test]
    fn apply_filter_allow_list_keeps_only_named_tools() {
        let mut reg = default_registry();
        apply_filter(
            &mut reg,
            Some(&["read".to_string(), "ls".to_string()]),
            None,
            false,
        );
        assert_eq!(reg.len(), 2);
        assert!(reg.get("read").is_some());
        assert!(reg.get("ls").is_some());
        assert!(reg.get("bash").is_none());
    }

    #[test]
    fn apply_filter_deny_list_drops_named_tools() {
        let mut reg = default_registry();
        apply_filter(
            &mut reg,
            None,
            Some(&["bash".to_string(), "edit".to_string(), "write".to_string()]),
            false,
        );
        assert!(reg.get("bash").is_none());
        assert!(reg.get("edit").is_none());
        assert!(reg.get("write").is_none());
        assert!(reg.get("read").is_some());
        assert_eq!(reg.len(), default_registry().len() - 3);
    }

    #[test]
    fn apply_filter_allow_and_deny_combine() {
        let mut reg = default_registry();
        // Allow read/write/edit, then carve `write` back out — the intersection.
        apply_filter(
            &mut reg,
            Some(&["read".to_string(), "write".to_string(), "edit".to_string()]),
            Some(&["write".to_string()]),
            false,
        );
        assert_eq!(reg.len(), 2);
        assert!(reg.get("read").is_some());
        assert!(reg.get("edit").is_some());
        assert!(reg.get("write").is_none());
    }

    #[test]
    fn apply_filter_unknown_names_are_silently_ignored() {
        let mut reg = default_registry();
        let before = reg.len();
        apply_filter(&mut reg, None, Some(&["does-not-exist".to_string()]), false);
        assert_eq!(reg.len(), before);
    }

    #[test]
    fn default_registry_with_none_matches_default_registry() {
        // `default_registry()` must genuinely delegate to `default_registry_with(None, None)`, not
        // just happen to look similar — same tool set either way.
        assert_eq!(
            default_registry().len(),
            default_registry_with(None, None).len()
        );
        for name in [
            "read", "write", "edit", "bash", "ls", "grep", "find", "fork", "sync", "logs",
        ] {
            assert!(default_registry_with(None, None).get(name).is_some());
        }
    }
}
