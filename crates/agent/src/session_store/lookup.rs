use std::fs;
use std::path::{Path, PathBuf};

use agent_core::Session;

use super::*;

/// The `<id>` component of a `<created_at>_<id>.jsonl` session filename, or `None` for anything that
/// isn't shaped like one. `split_once` on the *first* underscore only: `<created_at>` is always plain
/// digits and so never contains one, while a caller-supplied `--session-id` legally can — so everything
/// after that first separator is the id, `_` and all.
pub(super) fn file_id(path: &Path) -> Option<&str> {
    let name = path.file_name()?.to_str()?;
    let (_, id) = name.strip_suffix(".jsonl")?.split_once('_')?;
    Some(id)
}

/// Canonicalize `cwd` for session-matching purposes: resolves symlinks and `.`/`..` components (and, as
/// a side effect, any trailing separator) so two different-but-equivalent spellings of the same real
/// directory — a project reached through a symlink one time and its real path another, or a caller that
/// happens to include a trailing `/` — match the same session instead of silently fragmenting into two.
/// Falls back to `cwd` unchanged if it can't be resolved (removed out from under the process, a
/// permission error): matching degrades to today's exact-string comparison, never a new failure mode.
/// Every path this module ever records into or matches against a [`SessionMeta::cwd`] should be passed
/// through this first — `serve`'s own startup cwd and `run --continue`'s both do.
pub fn canonical_cwd(cwd: &Path) -> PathBuf {
    cwd.canonicalize().unwrap_or_else(|_| cwd.to_path_buf())
}

/// Encode `cwd` into a filesystem-safe directory-name component: every path separator becomes `-`, so
/// `/home/jared/ai` becomes `-home-jared-ai` — the same convention this repo's other per-project state
/// already uses (`trust_store.rs`'s `trusted-projects.json`, `prompts.rs`'s
/// `prompts/` directory), extended here to give each project its own session subdirectory.
pub fn encode_cwd(cwd: &str) -> String {
    cwd.chars()
        .map(|c| if c == '/' || c == '\\' { '-' } else { c })
        .collect()
}

/// The root every project's own session directory lives under: `~/.beyond/sessions/` (the agent's own
/// state directory, so `$AI_AGENT_CONFIG_DIR/sessions/` when that is set — see
/// `settings::config_dir_root`). Each immediate subdirectory is one project's `<encoded-cwd>/` (see
/// [`default_session_dir`]) — this is the root [`SessionRepo::list_all`] scans for pi's cross-project
/// session search (`--fork <id>` when the id isn't found in the current project — see [`fork_by_arg`]).
/// Never under `~/.claude`, whose `sessions/` holds Claude Code's own `<pid>.json` files.
pub fn sessions_root() -> PathBuf {
    crate::settings::config_dir_root().join("sessions")
}

/// The default session directory for `cwd` when nothing more specific was given: `~/.beyond/sessions/
/// <encoded-cwd>/`, one subdirectory per project so unrelated projects' sessions never mix in the same
/// listing. Shared by `serve`'s own default (`--session-file`/`--session-dir` absent) and `run
/// --continue`.
pub fn default_session_dir(cwd: &str) -> PathBuf {
    sessions_root().join(encode_cwd(cwd))
}

/// Whether a `--fork`/`--session` argument looks like a path rather than a bare session id — pi's own
/// `resolveSessionPath` treats any of these as "don't search, just resolve this as a file": a path
/// separator, a leading `.`/`~`, or a `.jsonl` extension. A session id (`new_id()`'s shape, or a
/// caller-supplied `--session-id`) never looks like this, so there's no realistic ambiguity. `pub`
/// (crosses the `main.rs` binary/lib boundary): shared with `run --session <arg>`'s own resolution
/// (Task #24), which needs the identical literal-path-or-bare-id classification [`fork_by_arg`] already
/// applies for `--fork <arg>`.
pub fn is_path_like(arg: &str) -> bool {
    arg.contains('/') || arg.starts_with('.') || arg.starts_with('~') || arg.ends_with(".jsonl")
}

/// Resolve a path-like `--fork`/`--session` argument against `cwd`: `~`/`~/...` expands to the home
/// directory (reusing the same convention `--skill`/`--prompt-template` extra paths already get via
/// `tools::expand_tilde`), and a relative path resolves against `cwd` — matching pi's own
/// `resolvePath(arg, cwd)`. An already-absolute path (post tilde-expansion) is returned unchanged.
/// Existence is deliberately not checked here — the caller's eventual `SessionStore::open` surfaces a
/// clear not-found error if it doesn't exist, consistent with every other path this crate resolves
/// lazily rather than pre-validating.
pub(super) fn resolve_path_like(arg: &str, cwd: &str) -> PathBuf {
    let expanded = crate::tools::expand_tilde(arg, std::env::var("HOME").ok().as_deref());
    let path = PathBuf::from(expanded);
    if path.is_absolute() {
        path
    } else {
        Path::new(cwd).join(path)
    }
}

/// Resolve a `--fork <arg>` argument and fork it into a brand-new session under `target`'s own repo —
/// pi's own cross-project `--fork <path|id>` (`SessionManager.forkFrom`/`resolveSessionPath`). `arg` may
/// be:
/// - a direct path to a `.jsonl` file (any project — see [`is_path_like`]/[`resolve_path_like`]),
/// - a session id that exists in `target`'s own project (forked in place via [`SessionRepo::fork`],
///   identical to today's same-project fork),
/// - a session id that exists in some *other* project's own directory under `sessions_root` (found via
///   [`SessionRepo::list_all`], then forked cross-project via [`SessionRepo::fork_from_path`], with
///   `target_cwd` — the project being forked *into* — becoming the new session's own `cwd`, not
///   wherever the source was originally recorded against).
///
/// Prefix matching is intentionally out of scope here (a bare id must match exactly) — see the
/// partial-session-id-resolution finding this crate tracks separately; this only ever resolves an exact
/// id or an explicit path.
pub fn fork_by_arg(
    arg: &str,
    target: &SessionRepo,
    target_cwd: &str,
    sessions_root: &Path,
    upto: usize,
) -> std::io::Result<(SessionStore, Session)> {
    if is_path_like(arg) {
        let path = resolve_path_like(arg, target_cwd);
        return target.fork_from_path(&path, target_cwd, upto);
    }
    // Try the current project first — `fork`'s own `open_id` is exact-then-unique-prefix aware (see
    // `SessionRepo::find_path`), so this alone already covers a prefix that resolves within this
    // project. An ambiguous prefix here is a real error to surface immediately, not a reason to widen
    // the search to other projects (that would risk silently reinterpreting the same prefix against a
    // *different* set of candidates instead of just reporting the one ambiguity the caller already hit).
    match target.fork(arg, upto) {
        Ok(result) => return Ok(result),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }
    match find_session_path_under(sessions_root, arg)? {
        Some(source_path) => target.fork_from_path(&source_path, target_cwd, upto),
        None => Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("no session matching \"{arg}\" in this project or any other"),
        )),
    }
}

/// Resolve a bare session id (or unique prefix) for `run --session <arg>` (Task #24) and reopen it *in
/// place* — pi's own `resolveSessionPath`'s "local"/"global" branches (`main.ts`), except a cross-project
/// match is reopened directly rather than routed through pi's own interactive fork-confirmation prompt:
/// this crate's `run` is headless, with no TTY-prompt path anywhere else to hang that on. Only ever
/// consulted once the caller has already ruled out a literal path (`is_path_like`, or one that already
/// exists on disk as-is) — unlike [`fork_by_arg`], which this otherwise mirrors tier-for-tier, `arg`
/// here is never treated as a path at all. Tries `target`'s own project first (`SessionRepo::open_id`,
/// exact-then-unique-prefix — the same first tier `--fork <id>` uses), then, on a genuine not-found,
/// every other project's own directory under `sessions_root` (`find_session_path_under`, `--fork`'s
/// identical second tier). `Err(NotFound)` naming `arg` when nothing matches anywhere — unlike a literal
/// path (which `--session` creates fresh when absent), a bare identifier that resolves to nothing is a
/// likely typo, not a request to start a new session named after it.
pub fn open_session_by_id(
    arg: &str,
    target: &SessionRepo,
    sessions_root: &Path,
) -> std::io::Result<(SessionStore, Session)> {
    match target.open_id(arg) {
        Ok(result) => Ok(result),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            match find_session_path_under(sessions_root, arg)? {
                Some(path) => SessionStore::open(path),
                None => Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("no session matching \"{arg}\" found as a path or session id"),
                )),
            }
        }
        Err(e) => Err(e),
    }
}

/// Locate a session matching `id` somewhere under `sessions_root`, by walking each immediate
/// subdirectory the same way [`SessionRepo::list_all_with_progress`] does and checking filenames
/// directly (`<created_at>_<id>.jsonl`) — deliberately *not* by recomputing a path from the session's
/// own recorded `cwd` via `default_session_dir`/`encode_cwd`. A caller-supplied `--session-dir`/
/// `AI_AGENT_SESSION_DIR` override need not follow that naming convention at all (it may be an
/// arbitrarily-named shared directory), so reconstructing a path from `cwd` would silently miss a real
/// session sitting right there under `sessions_root`. Cheaper than a full [`SessionRepo::list_all`] scan
/// too: this only ever inspects filenames, never a file's contents. `pub` (crosses the `main.rs`
/// binary/lib boundary): shared with [`open_session_by_id`]'s own cross-project fallback (Task #24).
pub fn find_session_path_under(sessions_root: &Path, id: &str) -> std::io::Result<Option<PathBuf>> {
    let project_dirs: Vec<PathBuf> = match fs::read_dir(sessions_root) {
        Ok(entries) => entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    let exact_suffix = format!("_{id}.jsonl");
    let all_files: Vec<PathBuf> = project_dirs
        .iter()
        .filter_map(|d| fs::read_dir(d).ok())
        .flat_map(|entries| entries.flatten().map(|e| e.path()))
        .collect();
    if let Some(path) = all_files.iter().find(|p| {
        p.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with(&exact_suffix))
    }) {
        return Ok(Some(path.clone()));
    }
    // No exact match anywhere: fall back to a unique-prefix match spanning every project — see
    // `SessionRepo::find_path`'s identical reasoning (ambiguous is an error naming every candidate, not
    // pi's own silent most-recent-wins).
    let matches: Vec<(&str, &PathBuf)> = all_files
        .iter()
        .filter_map(|path| {
            let name = path.file_name()?.to_str()?;
            let rest = name.strip_suffix(".jsonl")?;
            let (_, file_id) = rest.split_once('_')?;
            file_id.starts_with(id).then_some((file_id, path))
        })
        .collect();
    match matches.as_slice() {
        [] => Ok(None),
        [(_, path)] => Ok(Some((*path).clone())),
        _ => {
            let mut ids: Vec<&str> = matches.iter().map(|(id, _)| *id).collect();
            ids.sort_unstable();
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("\"{id}\" matches more than one session: {}", ids.join(", ")),
            ))
        }
    }
}

/// Validate — and, in a future format, migrate — a header to the current [`VERSION`]. Today there is
/// exactly one readable shape; a header whose version is *newer* than this build understands means we'd
/// be guessing at fields we don't know, so we refuse it loudly rather than mis-parse. The `match` is the
/// extension point: a later bump keeps the newer-version guard and adds an arm per older version that
/// upgrades it in place.
pub(super) fn migrate(meta: SessionMeta, path: &Path) -> std::io::Result<SessionMeta> {
    match meta.version {
        // `0` is a pre-versioning header (the field was absent and defaulted); it is wire-compatible
        // with v1. `VERSION` is the current shape.
        0 | VERSION => Ok(meta),
        v => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "session {} is format version {v}, which this build (version {VERSION}) cannot read",
                path.display()
            ),
        )),
    }
}

/// Gather the sessions directly under `dir` — the flat candidate set the daemon's
/// `list_daemon_sessions` feeds to [`scan_listings_in`]: `*.jsonl` files in the single-file layout,
/// or the subdirectories holding at least one segment (which skips `.trash/` and any half-created id)
/// in the segmented one. A single non-recursive `read_dir`; an unreadable directory (or a missing
/// one) yields an empty list rather than erroring, matching the skip-and-continue semantics of the
/// listing scans that consume it.
pub(crate) fn scan_session_dir_in(dir: &Path, layout: &Layout) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let segmented = layout.is_segmented();
    entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            if segmented {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| !n.starts_with('.'))
                    && segment_dir_is_session(p)
            } else {
                p.extension().and_then(|e| e.to_str()) == Some("jsonl")
            }
        })
        .collect()
}
