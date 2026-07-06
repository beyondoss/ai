//! Project trust: gates the `SYSTEM.md` project override (and, per-project, skill/prompt-template
//! discovery) on an explicit allowlist.
//!
//! `resources::system_prompt_override` lets a project pin its own agent identity via
//! `<cwd>/.claude/SYSTEM.md` — but unconditionally honoring that let any repo checkout hijack the
//! agent's base instructions just by shipping the file. pi gates the same override (and its whole
//! project-local config bundle: skills, prompt templates, extensions) on an interactive "trust this
//! folder?" TUI prompt; this agent is headless (one-shot `run`, or `serve` driven by a client over
//! stdio) with no human at a terminal to ask, so trust here is either operator-asserted up front
//! (`--trust-project` / a `serve` init param) or recorded ahead of time in a persisted allowlist an
//! operator manages out-of-band (`agent trust <path>` / `agent untrust <path>`) — never granted
//! through an interactive prompt.
//!
//! Trust is tri-state and ancestor-aware: trusting `~/code` implies `~/code/anything` is trusted too,
//! unless a subdirectory was explicitly `untrust`ed, which shadows the inherited grant for that
//! subtree without touching the parent's entry.

use std::collections::BTreeSet;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};

use crate::path_utils::resolved_path as resolved_key_path;

/// Whether `cwd` has anything project trust actually gates: a `SYSTEM.md`/`APPEND_SYSTEM.md`
/// override, project-local skills (`.claude/skills`, or an `.agents/skills` at any ancestor level up
/// to the enclosing git-repo root), project-local prompt templates (`.claude/prompts`), or a
/// project-level settings tier (`.claude/settings.json` — Round 3, pi-parity feature: see
/// `settings::effective_settings_for_cwd`) — see `resources.rs`/`skills.rs`/`prompts.rs`/`settings.rs`
/// for where each of these is actually read. When none exist, there's nothing an untrusted checkout
/// could do differently by being trusted, so a caller can treat the directory as trusted without ever
/// consulting the allowlist — matching pi's own `hasTrustRequiringProjectResources`/
/// `resolveProjectTrusted` fast path (`trust-manager.ts`), which skips the trust store (and its own
/// interactive prompt) entirely for the same reason. Nothing here is cached or sticky: a directory that
/// gains one of these later (e.g. a fresh `git init` adding `.claude/skills`) simply starts requiring
/// trust from that point on, the next time this is checked.
pub fn has_trust_gated_resources(cwd: &Path) -> bool {
    cwd.join(".claude/SYSTEM.md").is_file()
        || cwd.join(".claude/APPEND_SYSTEM.md").is_file()
        || cwd.join(".claude/skills").is_dir()
        || cwd.join(".claude/prompts").is_dir()
        || cwd.join(".claude/settings.json").is_file()
        // pi-parity fix (Task #45): the operator's own `~/.agents/skills` (pi's documented,
        // always-trusted personal-skills convention) must not itself count as a project-controlled,
        // trust-gated resource — `collect_ancestor_agents_skill_dirs` alone has no notion of "this
        // ancestor happens to be the user's home skills dir, not the project's", so for a `cwd` with no
        // enclosing git repo (the ancestor walk then reaches all the way to `/`) an operator who has
        // ever created `~/.agents/skills` would otherwise always trip this, even though the project
        // itself defines nothing that actually needs gating. Matches the same exclusion the actual
        // skill-*loading* path already applies (`skills::discover_with_diagnostics_impl`) and pi's own
        // `hasTrustRequiringProjectResources` (`trust-manager.ts:184-206`).
        || crate::skills::collect_ancestor_agents_skill_dirs_excluding_user_dir(cwd)
            .iter()
            .any(|dir| dir.is_dir())
}

/// The trust decision for a directory, after walking up to the nearest ancestor with an explicit
/// entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trust {
    Trusted,
    Untrusted,
    Unknown,
}

/// A persisted, tri-state, ancestor-aware allowlist of project paths.
pub struct TrustStore {
    path: PathBuf,
    trusted: BTreeSet<String>,
    untrusted: BTreeSet<String>,
}

/// On-disk shape once the store has been written by this version. Older files are a bare
/// `Vec<String>` (trusted-only) — see [`TrustStore::open`].
#[derive(Serialize, Deserialize, Default)]
struct StoreFile {
    #[serde(default)]
    trusted: BTreeSet<String>,
    #[serde(default)]
    untrusted: BTreeSet<String>,
}

impl TrustStore {
    /// Open the default trust store (`~/.claude/trusted-projects.json`). A missing or unparsable file
    /// is treated as an empty allowlist — nothing is trusted until explicitly recorded.
    pub fn open_default() -> Self {
        Self::open(default_path())
    }

    /// Open the trust store at `path`. Tries the current tri-state object shape first, falling back
    /// to the legacy bare `Vec<String>` shape (trusted-only) so files written by an older binary keep
    /// working with no migration step — the new shape is written the next time `trust`/`distrust` is
    /// called.
    pub fn open(path: PathBuf) -> Self {
        let (trusted, untrusted) = read_store_file(&path);
        Self {
            path,
            trusted,
            untrusted,
        }
    }

    /// The trust decision for `dir`, walking from `dir` up through its ancestors and returning the
    /// first explicit entry found (checking `untrusted` before `trusted` at each level, so a
    /// same-level conflict favors denying). No entry anywhere in the chain means [`Trust::Unknown`].
    pub fn lookup(&self, dir: &Path) -> Trust {
        let canonical = resolved_key_path(dir);
        for ancestor in canonical.ancestors() {
            let key = ancestor.display().to_string();
            if self.untrusted.contains(&key) {
                return Trust::Untrusted;
            }
            if self.trusted.contains(&key) {
                return Trust::Trusted;
            }
        }
        Trust::Unknown
    }

    /// Whether `dir` is trusted (directly, or by inheriting an ancestor's trust grant).
    pub fn is_trusted(&self, dir: &Path) -> bool {
        matches!(self.lookup(dir), Trust::Trusted)
    }

    /// Idempotently trust `dir`, persisting atomically. Clears any exact-path `untrusted` entry for
    /// `dir` first, so re-trusting a directory that was previously explicitly distrusted actually
    /// takes effect rather than being shadowed by its own stale entry.
    pub fn trust(&mut self, dir: &Path) -> std::io::Result<()> {
        let key = canonical_key(dir);
        self.mutate_locked(|trusted, untrusted| {
            let removed_untrusted = untrusted.remove(&key);
            let inserted = trusted.insert(key.clone());
            inserted || removed_untrusted
        })
    }

    /// Idempotently mark `dir` as explicitly untrusted, persisting atomically. Clears any exact-path
    /// `trusted` entry for `dir` first (a directory can't be both).
    pub fn distrust(&mut self, dir: &Path) -> std::io::Result<()> {
        let key = canonical_key(dir);
        self.mutate_locked(|trusted, untrusted| {
            let removed_trusted = trusted.remove(&key);
            let inserted = untrusted.insert(key.clone());
            inserted || removed_trusted
        })
    }

    /// Remove any exact-path entry for `dir` — trusted *or* untrusted — without recording a new one,
    /// so `dir` reverts to inheriting whatever its nearest ancestor decides (or [`Trust::Unknown`] if
    /// none does), rather than staying pinned to its own explicit grant/denial. Neither `trust` nor
    /// `distrust` can express this: both always leave `dir` with *some* exact-path entry. Idempotent —
    /// clearing an already-unset directory is a no-op, not an error. Mirrors pi's own
    /// `ProjectTrustStore::setMany` accepting a `null` decision to delete an entry.
    pub fn clear(&mut self, dir: &Path) -> std::io::Result<()> {
        let key = canonical_key(dir);
        self.mutate_locked(|trusted, untrusted| {
            let removed_trusted = trusted.remove(&key);
            let removed_untrusted = untrusted.remove(&key);
            removed_trusted || removed_untrusted
        })
    }

    /// Acquire the cross-process lock (see [`FileLock`]), re-read the store's *current* on-disk state
    /// — not `self`'s possibly-stale in-memory copy, which another process may have moved past since
    /// this handle's `open()` or its own last mutation — let `mutate` apply its change against that
    /// fresh state, and persist only if something actually changed. `self` is updated to match
    /// afterward either way, so this handle's own `lookup()` stays consistent with what's now on disk.
    ///
    /// This closes the read-modify-write race the plain "mutate `self`, then write" version had: two
    /// processes calling `trust`/`distrust`/`clear` concurrently could each start from their own stale
    /// in-memory snapshot and silently clobber the other's update on write — e.g. process A trusts
    /// `~/proj1` while process B trusts `~/proj2` at nearly the same moment; without a lock, whichever
    /// writes last wins outright and the other's grant is gone from disk despite both calls returning
    /// `Ok`. With the lock, B's write waits for A's to land, then B re-reads and re-applies its own
    /// change on top of A's — both grants survive.
    fn mutate_locked(
        &mut self,
        mutate: impl FnOnce(&mut BTreeSet<String>, &mut BTreeSet<String>) -> bool,
    ) -> std::io::Result<()> {
        let _lock = FileLock::acquire(&self.path)?;
        let (mut trusted, mut untrusted) = read_store_file(&self.path);
        let changed = mutate(&mut trusted, &mut untrusted);
        if changed {
            write_store_file(&self.path, &trusted, &untrusted)?;
        }
        self.trusted = trusted;
        self.untrusted = untrusted;
        Ok(())
    }
}

/// Read and parse the store file at `path`, tolerating a missing/unparsable file (empty allowlist) and
/// the legacy bare `Vec<String>` shape (trusted-only) — shared by [`TrustStore::open`] and
/// [`TrustStore::mutate_locked`], which both need "current on-disk state, gracefully defaulted."
///
/// A missing file is the ordinary, silent case (nothing has ever been trusted yet). Anything else that
/// falls back to an empty allowlist — an existing-but-unreadable file (permission denied, a directory
/// where a file was expected) or one that parses as neither the current tri-state shape nor the legacy
/// bare-array shape (corrupted/hand-edited JSON) — `warn!`s instead of silently discarding every
/// persisted trust/untrust decision, matching `session_store.rs`'s own "skip and warn, don't silently
/// lose data" precedent for a corrupt entry.
fn read_store_file(path: &Path) -> (BTreeSet<String>, BTreeSet<String>) {
    let contents = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) if e.kind() == ErrorKind::NotFound => return Default::default(),
        Err(e) => {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "could not read trust store file, treating it as an empty allowlist"
            );
            return Default::default();
        }
    };
    if let Ok(f) = serde_json::from_str::<StoreFile>(&contents) {
        return (f.trusted, f.untrusted);
    }
    if let Ok(v) = serde_json::from_str::<Vec<String>>(&contents) {
        return (v.into_iter().collect(), BTreeSet::new());
    }
    tracing::warn!(
        path = %path.display(),
        "trust store file is not valid JSON in either the current or legacy shape, treating it as an \
         empty allowlist"
    );
    Default::default()
}

fn write_store_file(
    path: &Path,
    trusted: &BTreeSet<String>,
    untrusted: &BTreeSet<String>,
) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = StoreFile {
        trusted: trusted.clone(),
        untrusted: untrusted.clone(),
    };
    let body = serde_json::to_string_pretty(&file).map_err(std::io::Error::other)?;
    let path_str = path
        .to_str()
        .ok_or_else(|| std::io::Error::other(format!("non-UTF-8 path: {}", path.display())))?;
    crate::tools::write_atomic(path_str, body.as_bytes())
}

/// How long [`FileLock::acquire`] retries before giving up.
const LOCK_TIMEOUT: Duration = Duration::from_secs(5);
/// How long between retries while waiting for another process's lock.
const LOCK_RETRY_INTERVAL: Duration = Duration::from_millis(20);
/// A lock file older than this is assumed abandoned (its owner crashed without cleaning up, rather
/// than genuinely being mid-operation for this long — a trust-file read/parse/write is a handful of
/// milliseconds) and is forcibly reclaimed instead of blocking every future trust/untrust indefinitely.
const STALE_LOCK_AGE: Duration = Duration::from_secs(10);

/// A cross-process advisory lock via atomic lockfile creation (`O_EXCL`-equivalent,
/// `create_new(true)`) — the same primitive [`crate::tools::write_atomic`]'s temp file and
/// `session_store.rs`'s collision-safe id generation already rely on elsewhere in this codebase, so no
/// new dependency (`flock`/`fs2`/etc.) is needed for this narrower single-file case. Released by
/// deleting the lock file on `Drop`, so a panicked or early-returning holder still releases it.
struct FileLock {
    path: PathBuf,
}

impl FileLock {
    /// Acquire the lock guarding `store_path` (a sibling `.lock` file), retrying while another process
    /// holds it, up to [`LOCK_TIMEOUT`]. A lock older than [`STALE_LOCK_AGE`] is reclaimed immediately
    /// rather than waited out.
    fn acquire(store_path: &Path) -> std::io::Result<Self> {
        let lock_path = lock_path_for(store_path);
        if let Some(parent) = lock_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let deadline = Instant::now() + LOCK_TIMEOUT;
        loop {
            match fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&lock_path)
            {
                Ok(_) => return Ok(Self { path: lock_path }),
                Err(e) if e.kind() == ErrorKind::AlreadyExists => {
                    if is_stale(&lock_path) {
                        // Best-effort: another process reclaiming it at the same instant just means
                        // our own retry loop's next `create_new` fails normally and we wait it out.
                        let _ = fs::remove_file(&lock_path);
                        continue;
                    }
                    if Instant::now() >= deadline {
                        return Err(std::io::Error::new(
                            ErrorKind::TimedOut,
                            format!(
                                "timed out waiting for trust store lock at {}",
                                lock_path.display()
                            ),
                        ));
                    }
                    std::thread::sleep(LOCK_RETRY_INTERVAL);
                }
                Err(e) => return Err(e),
            }
        }
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn is_stale(lock_path: &Path) -> bool {
    fs::metadata(lock_path)
        .and_then(|m| m.modified())
        .and_then(|modified| {
            SystemTime::now()
                .duration_since(modified)
                .map_err(std::io::Error::other)
        })
        .is_ok_and(|age| age > STALE_LOCK_AGE)
}

fn lock_path_for(store_path: &Path) -> PathBuf {
    let mut os = store_path.as_os_str().to_owned();
    os.push(".lock");
    PathBuf::from(os)
}

/// Canonicalize `path` for allowlist comparison, falling back to the path as given when it doesn't
/// exist yet or can't be resolved (canonicalization needs the path to exist; a project directory
/// always does by the time trust is checked, but degrading gracefully here is cheap insurance).
fn canonical_key(path: &Path) -> String {
    resolved_key_path(path).display().to_string()
}

/// pi-parity fix: previously hardcoded `home.join(".claude/...")` directly, with no way to redirect it
/// short of overriding `$HOME` itself (which also moves every other HOME-relative thing). Shares
/// `settings.rs::config_dir_root`'s `AI_AGENT_CONFIG_DIR`-aware resolution rather than duplicating it —
/// both stores' config directory must agree on the same root.
fn default_path() -> PathBuf {
    crate::settings::config_dir_root().join("trusted-projects.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_trust_gated_resources_is_false_for_a_bare_directory() {
        // Pi-parity fix: a directory with nothing project trust actually gates has no functional
        // difference between trusted and untrusted — matching pi's own
        // `hasTrustRequiringProjectResources` fast path.
        let dir = tempfile::tempdir().unwrap();
        assert!(!has_trust_gated_resources(dir.path()));
    }

    #[test]
    fn has_trust_gated_resources_is_true_for_a_project_system_md() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join(".claude")).unwrap();
        fs::write(dir.path().join(".claude/SYSTEM.md"), "custom identity").unwrap();
        assert!(has_trust_gated_resources(dir.path()));
    }

    #[test]
    fn has_trust_gated_resources_is_true_for_a_project_append_system_md() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join(".claude")).unwrap();
        fs::write(dir.path().join(".claude/APPEND_SYSTEM.md"), "extra rules").unwrap();
        assert!(has_trust_gated_resources(dir.path()));
    }

    #[test]
    fn has_trust_gated_resources_is_true_for_a_project_skills_directory() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join(".claude/skills")).unwrap();
        assert!(has_trust_gated_resources(dir.path()));
    }

    #[test]
    fn has_trust_gated_resources_is_true_for_a_project_prompts_directory() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join(".claude/prompts")).unwrap();
        assert!(has_trust_gated_resources(dir.path()));
    }

    #[test]
    fn has_trust_gated_resources_is_true_for_a_project_settings_json() {
        // Round 3 (pi-parity feature): a project-level settings tier (`settings::
        // effective_settings_for_cwd`) is itself a trust-gated resource, same as SYSTEM.md/skills/
        // prompts above.
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join(".claude")).unwrap();
        fs::write(dir.path().join(".claude/settings.json"), r#"{"default_model":"m"}"#).unwrap();
        assert!(has_trust_gated_resources(dir.path()));
    }

    #[test]
    fn has_trust_gated_resources_is_true_for_an_ancestor_agents_skills_directory() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("a/b/c");
        fs::create_dir_all(&nested).unwrap();
        fs::create_dir_all(dir.path().join("a/.agents/skills")).unwrap();
        assert!(
            has_trust_gated_resources(&nested),
            "an ancestor's .agents/skills must count, not just cwd's own"
        );
    }

    #[test]
    fn pre_trusting_a_not_yet_created_directory_via_a_noisy_path_takes_effect_once_it_exists() {
        // Pi-parity audit H68: the old `canonical_key`/`lookup` called `path.canonicalize()` with no
        // absolutize-and-normalize step first — for a directory that doesn't exist yet, `canonicalize`
        // fails outright (`ENOENT`), and the fallback was the literal path as given. A trailing slash
        // or a redundant `./` component then landed in the stored key verbatim, which could never
        // match the fully-canonical absolute path a later `lookup()` (once the directory actually
        // exists) compares against — `agent trust newproj/`, run before `newproj` exists, silently
        // never took effect.
        let store_dir = tempfile::tempdir().unwrap();
        let mut trust_store = TrustStore::open(store_dir.path().join("trusted-projects.json"));

        let parent = tempfile::tempdir().unwrap();
        let not_yet_created = parent.path().join("newproj");
        // Trailing slash and a redundant `./` component — realistic shapes a hand-typed path or shell
        // completion could produce, and exactly the noise `canonicalize()` itself would collapse if
        // the directory existed (it doesn't yet).
        let noisy_path = parent.path().join("./newproj/");

        trust_store.trust(&noisy_path).unwrap();

        // The directory now actually gets created — the workflow this store's own module doc comment
        // says it must support: pre-authorizing a project out-of-band, ahead of its first real
        // `run`/`serve` invocation.
        fs::create_dir_all(&not_yet_created).unwrap();

        // The real runtime lookup always uses a canonicalized, absolute path (see `canonical_cwd` in
        // `session_store.rs`) — reproduced directly here rather than via `set_current_dir`.
        let real_lookup_path = not_yet_created.canonicalize().unwrap();
        assert_eq!(
            trust_store.lookup(&real_lookup_path),
            Trust::Trusted,
            "a directory pre-trusted via a noisy (trailing-slash/`./`) path, before it existed, must \
             still be recognized as trusted once it exists and is looked up via its canonical form"
        );
    }

    #[test]
    fn untrusted_directory_is_not_trusted() {
        let dir = tempfile::tempdir().unwrap();
        let store = TrustStore::open(dir.path().join("trusted-projects.json"));
        assert!(!store.is_trusted(dir.path()));
        assert_eq!(store.lookup(dir.path()), Trust::Unknown);
    }

    #[test]
    fn trust_then_is_trusted() {
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path().join("project");
        fs::create_dir_all(&project).unwrap();
        let store_path = dir.path().join("trusted-projects.json");

        let mut store = TrustStore::open(store_path.clone());
        assert!(!store.is_trusted(&project));
        store.trust(&project).unwrap();
        assert!(store.is_trusted(&project));

        // Persisted: a fresh handle on the same file sees the same trust.
        let reopened = TrustStore::open(store_path);
        assert!(reopened.is_trusted(&project));
    }

    #[test]
    fn trust_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path().join("project");
        fs::create_dir_all(&project).unwrap();
        let store_path = dir.path().join("trusted-projects.json");

        let mut store = TrustStore::open(store_path.clone());
        store.trust(&project).unwrap();
        store.trust(&project).unwrap(); // must not error or duplicate

        let body = fs::read_to_string(&store_path).unwrap();
        let file: StoreFile = serde_json::from_str(&body).unwrap();
        assert_eq!(file.trusted.len(), 1);
    }

    #[test]
    fn missing_store_file_is_an_empty_allowlist() {
        let dir = tempfile::tempdir().unwrap();
        let store = TrustStore::open(dir.path().join("does-not-exist.json"));
        assert!(!store.is_trusted(dir.path()));
    }

    #[test]
    fn trust_creates_parent_directories() {
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path().join("project");
        fs::create_dir_all(&project).unwrap();
        // A nested store path whose parent directory doesn't exist yet — matches the real
        // `~/.claude/trusted-projects.json` layout on a fresh `~/.claude`.
        let store_path = dir.path().join("nested/.claude/trusted-projects.json");

        let mut store = TrustStore::open(store_path.clone());
        store.trust(&project).unwrap();
        assert!(store_path.exists());
    }

    #[test]
    fn trusting_a_parent_implies_an_untrusted_child_dir() {
        let dir = tempfile::tempdir().unwrap();
        let code = dir.path().join("code");
        let sub = code.join("sub");
        let deeper = sub.join("deeper");
        let other = dir.path().join("other");
        fs::create_dir_all(&deeper).unwrap();
        fs::create_dir_all(&other).unwrap();
        let store_path = dir.path().join("trusted-projects.json");

        let mut store = TrustStore::open(store_path);
        store.trust(&code).unwrap();
        assert!(
            store.is_trusted(&sub),
            "child of a trusted dir inherits trust"
        );
        assert!(store.is_trusted(&deeper), "grandchild inherits too");
        assert!(!store.is_trusted(&other), "an unrelated dir stays unknown");

        store.distrust(&sub).unwrap();
        assert_eq!(store.lookup(&code), Trust::Trusted);
        assert_eq!(
            store.lookup(&sub),
            Trust::Untrusted,
            "an explicit distrust shadows the inherited parent grant"
        );
        assert_eq!(
            store.lookup(&deeper),
            Trust::Untrusted,
            "the shadow applies to the whole distrusted subtree"
        );
    }

    #[test]
    fn legacy_bare_array_file_still_parses_as_trusted_only() {
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path().join("project");
        fs::create_dir_all(&project).unwrap();
        let store_path = dir.path().join("trusted-projects.json");
        let legacy = serde_json::to_string(&vec![canonical_key(&project)]).unwrap();
        fs::write(&store_path, legacy).unwrap();

        let store = TrustStore::open(store_path);
        assert!(store.is_trusted(&project));
    }

    #[test]
    fn a_missing_store_file_is_silent_and_treated_as_an_empty_allowlist() {
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("does-not-exist.json");

        let capture = crate::tracing_test::capture(|| {
            let store = TrustStore::open(store_path);
            assert!(matches!(store.lookup(dir.path()), Trust::Unknown));
        });
        assert!(
            capture.messages().is_empty(),
            "a missing file is the ordinary case, not worth warning about: {:?}",
            capture.messages()
        );
    }

    #[test]
    fn a_corrupt_store_file_warns_and_is_treated_as_an_empty_allowlist() {
        // Pi-parity fix: a corrupted/unreadable store file previously fell back to an empty allowlist
        // exactly like a missing one — silently discarding every persisted trust/untrust decision with
        // no signal at all that anything was wrong.
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("trusted-projects.json");
        fs::write(&store_path, "not valid json at all { [ }").unwrap();

        let capture = crate::tracing_test::capture(|| {
            let store = TrustStore::open(store_path.clone());
            assert!(matches!(store.lookup(dir.path()), Trust::Unknown));
        });
        let messages = capture.messages();
        assert!(
            messages.iter().any(|m| m.contains("trust store")),
            "a corrupt store file must be logged, not silently swallowed: {messages:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn an_unreadable_store_file_warns_and_is_treated_as_an_empty_allowlist() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("trusted-projects.json");
        fs::write(&store_path, "{}").unwrap();
        fs::set_permissions(&store_path, fs::Permissions::from_mode(0o000)).unwrap();
        // Some environments (root, certain sandboxes) don't actually enforce permission bits — skip
        // rather than assert a false failure if the mode change didn't block the read at all.
        let mode_actually_blocks_reads = fs::read_to_string(&store_path).is_err();

        let capture = crate::tracing_test::capture(|| {
            let store = TrustStore::open(store_path.clone());
            assert!(matches!(store.lookup(dir.path()), Trust::Unknown));
        });
        let _ = fs::set_permissions(&store_path, fs::Permissions::from_mode(0o644));

        if !mode_actually_blocks_reads {
            eprintln!("skipping: this environment doesn't enforce file permission bits");
            return;
        }
        let messages = capture.messages();
        assert!(
            messages.iter().any(|m| m.contains("trust store")),
            "an unreadable store file must be logged, not silently swallowed: {messages:?}"
        );
    }

    #[test]
    fn retrusting_a_distrusted_dir_clears_the_distrust() {
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path().join("project");
        fs::create_dir_all(&project).unwrap();
        let store_path = dir.path().join("trusted-projects.json");

        let mut store = TrustStore::open(store_path);
        store.distrust(&project).unwrap();
        assert_eq!(store.lookup(&project), Trust::Untrusted);
        store.trust(&project).unwrap();
        assert_eq!(store.lookup(&project), Trust::Trusted);
    }

    #[test]
    fn clear_reverts_to_inheriting_from_the_nearest_ancestor() {
        // Neither `trust` nor `distrust` can express "no opinion, inherit from the parent" — both
        // always leave an exact-path entry behind. `clear` removes it, so a subdirectory explicitly
        // distrusted earlier goes back to following its trusted parent, rather than needing a
        // `trust()` call that would instead pin it with its *own* explicit grant.
        let dir = tempfile::tempdir().unwrap();
        let code = dir.path().join("code");
        let sub = code.join("sub");
        fs::create_dir_all(&sub).unwrap();
        let store_path = dir.path().join("trusted-projects.json");

        let mut store = TrustStore::open(store_path);
        store.trust(&code).unwrap();
        store.distrust(&sub).unwrap();
        assert_eq!(store.lookup(&sub), Trust::Untrusted);

        store.clear(&sub).unwrap();
        assert_eq!(
            store.lookup(&sub),
            Trust::Trusted,
            "clearing the shadow reveals the inherited parent grant"
        );

        // Clearing the parent's own entry drops it all the way to Unknown (no ancestor left to
        // inherit from).
        store.clear(&code).unwrap();
        assert_eq!(store.lookup(&code), Trust::Unknown);
        assert_eq!(store.lookup(&sub), Trust::Unknown);
    }

    #[test]
    fn clear_on_an_already_unset_directory_is_a_no_op() {
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path().join("project");
        fs::create_dir_all(&project).unwrap();
        let store_path = dir.path().join("trusted-projects.json");

        let mut store = TrustStore::open(store_path.clone());
        store.clear(&project).unwrap();
        assert_eq!(store.lookup(&project), Trust::Unknown);
        assert!(
            !store_path.exists(),
            "a no-op clear must not create the store file"
        );
    }

    #[test]
    fn a_stale_in_memory_snapshot_is_refreshed_before_mutating_not_blindly_overwritten() {
        // pi-parity fix: two `TrustStore` handles on the same file (matching what actually happens
        // across two separate `agent trust` invocations) each trusting a *different* directory used to
        // be able to silently lose one grant — the second handle to write would persist straight from
        // its own in-memory snapshot, taken before the first handle's write ever landed, discarding it.
        // Deterministic (no thread timing): `handle_b` is opened *before* `handle_a`'s write, so its
        // in-memory state is genuinely stale by the time it mutates — proving `mutate_locked` re-reads
        // fresh state under the lock rather than trusting `self`.
        let dir = tempfile::tempdir().unwrap();
        let project_a = dir.path().join("a");
        let project_b = dir.path().join("b");
        fs::create_dir_all(&project_a).unwrap();
        fs::create_dir_all(&project_b).unwrap();
        let store_path = dir.path().join("trusted-projects.json");

        let mut handle_a = TrustStore::open(store_path.clone());
        let mut handle_b = TrustStore::open(store_path.clone());

        handle_a.trust(&project_a).unwrap();
        handle_b.trust(&project_b).unwrap();

        let reopened = TrustStore::open(store_path);
        assert!(
            reopened.is_trusted(&project_a),
            "project a's grant must survive project b's later write from a stale handle"
        );
        assert!(reopened.is_trusted(&project_b));
    }

    #[test]
    fn concurrent_trust_calls_from_separate_handles_do_not_lose_an_update() {
        // Realistic-scenario stress test for the same fix: many independent handles (simulating
        // separate `agent trust` processes) racing to trust different directories at once. A
        // `Barrier` lines up every handle's `open()` before any of them mutate, maximizing the odds
        // every handle's in-memory snapshot is stale by the time it writes — exactly the shape of race
        // the deterministic test above proves in isolation.
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("trusted-projects.json");
        const N: usize = 8;
        let projects: Vec<PathBuf> = (0..N)
            .map(|i| {
                let p = dir.path().join(format!("project-{i}"));
                fs::create_dir_all(&p).unwrap();
                p
            })
            .collect();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(N));

        let handles: Vec<_> = projects
            .iter()
            .map(|project| {
                let project = project.clone();
                let store_path = store_path.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    let mut store = TrustStore::open(store_path);
                    barrier.wait();
                    store.trust(&project).unwrap();
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }

        let reopened = TrustStore::open(store_path);
        for (i, project) in projects.iter().enumerate() {
            assert!(
                reopened.is_trusted(project),
                "project-{i} lost its trust grant to a concurrent write"
            );
        }
    }

    #[test]
    fn a_stale_lock_file_is_reclaimed_rather_than_blocking_forever() {
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path().join("project");
        fs::create_dir_all(&project).unwrap();
        let store_path = dir.path().join("trusted-projects.json");
        let lock_path = lock_path_for(&store_path);
        fs::write(&lock_path, b"").unwrap();
        // Back-date the lock file's mtime past `STALE_LOCK_AGE` so `acquire` treats it as abandoned
        // (a crashed holder that never cleaned up) instead of waiting out `LOCK_TIMEOUT`.
        let stale = SystemTime::now() - (STALE_LOCK_AGE + Duration::from_secs(1));
        fs::File::open(&lock_path)
            .unwrap()
            .set_modified(stale)
            .unwrap();

        let mut store = TrustStore::open(store_path.clone());
        store.trust(&project).unwrap();
        assert!(store.is_trusted(&project));
    }
}
