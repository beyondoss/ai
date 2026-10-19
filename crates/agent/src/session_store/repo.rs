use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use agent_core::{Message, Role, Session};

use super::*;

/// A directory of session files. `Clone` is just a `PathBuf` copy — cheap, and lets a caller move an
/// owned handle into a `spawn_blocking` closure (see `serve/persistence.rs`'s `Persistence::list_with_progress`)
/// without holding a borrow of the original across the `.await`.
#[derive(Clone)]
pub struct SessionRepo {
    pub(super) dir: PathBuf,
    pub(super) opts: RepoOptions,
}

/// `e`, with what was being done and to which path in front of the OS's message — an
/// `io::Error` alone says "Permission denied (os error 13)" and nothing about where. Keeps the kind,
/// so callers that branch on it still can.
pub(super) fn at_path(action: &str, path: &Path, e: std::io::Error) -> std::io::Error {
    std::io::Error::new(e.kind(), format!("{action} {}: {e}", path.display()))
}

impl SessionRepo {
    /// Open (creating if needed) a repository rooted at `dir`, in the single-file layout with no id
    /// prefix — what every local invocation uses.
    pub fn open(dir: impl Into<PathBuf>) -> std::io::Result<Self> {
        Self::open_with(dir, RepoOptions::default())
    }

    /// [`open`](Self::open) with an explicit [`Layout`] and id prefix — the service-mode entry point.
    pub fn open_with(dir: impl Into<PathBuf>, opts: RepoOptions) -> std::io::Result<Self> {
        let dir = dir.into();
        fs::create_dir_all(&dir)
            .map_err(|e| at_path("cannot create session directory", &dir, e))?;
        Ok(Self { dir, opts })
    }

    /// The directory this repo is rooted at.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub(super) fn layout(&self) -> &Layout {
        &self.opts.layout
    }

    pub(super) fn segmented(&self) -> bool {
        self.opts.layout.is_segmented()
    }

    /// A fresh id for a session this repo *derives* — a fork, a clone, an archive. With an
    /// `id_prefix` configured it comes out as `<prefix>.<opaque>`, so a derived session stays on the
    /// shard its parent lives on and every path built from it is still rooted at the same tenant.
    pub fn mint_id(&self) -> String {
        match &self.opts.id_prefix {
            Some(prefix) => format!("{prefix}.{}", new_id()),
            None => new_id(),
        }
    }

    /// Fresh metadata carrying an id this repo minted (see [`mint_id`](Self::mint_id)).
    pub(super) fn derived_meta(
        &self,
        cwd: impl Into<String>,
        model: impl Into<String>,
    ) -> SessionMeta {
        SessionMeta::with_id(self.mint_id(), cwd, model)
    }

    /// Open a session this repo owns, for writing.
    pub(super) fn open_path(
        &self,
        path: PathBuf,
        id: &str,
    ) -> std::io::Result<(SessionStore, Session)> {
        SessionStore::open_with(path, Some(id), self.layout(), false)
    }

    /// Open a session as a *source* — a fork's origin, a preview, an export. Read-only: none of these
    /// owns the session, and a stray write from one would take an epoch off whoever does.
    pub(super) fn open_source(
        &self,
        path: PathBuf,
        id: &str,
    ) -> std::io::Result<(SessionStore, Session)> {
        SessionStore::open_with(path, Some(id), self.layout(), true)
    }

    /// Open the session named `id`, or create it under exactly that id if it doesn't exist yet.
    ///
    /// **A session id is an address.** This resolves to the session named `id` or to a brand-new one
    /// carrying that name — never to some *other* session, and never to a unique-prefix neighbour (see
    /// [`Self::find_path_exact`]). That total, unambiguous mapping is what makes an id usable as a
    /// routing key: a caller minting one id per tenant, per task, or per connection gets one session
    /// apiece, and re-running with the same id is idempotent, landing back on the same conversation.
    ///
    /// Deliberately does **not** consult `cwd`. It's recorded on a freshly created session (and is what
    /// [`Self::resume_latest_or_create`] matches on), but it never overrides an explicit name: an
    /// earlier version of this API checked `cwd` first and silently discarded the caller's id, which
    /// collapsed every distinct id in a shared directory onto one shared session.
    pub fn open_or_create_id(
        &self,
        id: &str,
        cwd: &str,
        model: &str,
    ) -> std::io::Result<(SessionStore, Session)> {
        let create = || -> std::io::Result<(SessionStore, Session)> {
            Ok((
                self.create(SessionMeta::with_id(id.to_string(), cwd, model))?,
                Session::new(),
            ))
        };
        let Some(path) = self.find_path_exact(id)? else {
            return create();
        };
        match self.open_path(path.clone(), id) {
            Ok(opened) => Ok(opened),
            // A create that crashed before its header landed (under a build that did not create
            // files whole) left a file that is neither openable nor creatable over: this id was
            // bricked for good. Nothing was ever written to it, so it is treated as absent.
            Err(e) => match self.headerless_first_file(&path, id) {
                Some((first, expected)) if clear_headerless(&first, &expected)? => create(),
                _ => Err(e),
            },
        }
    }

    /// The first file of the session at `path` — the `.jsonl` itself, or a segmented session's
    /// `000001.jsonl` when it is the only segment — and the bytes its first line begins with, if that
    /// file is headerless ([`is_headerless`]).
    pub(super) fn headerless_first_file(
        &self,
        path: &Path,
        id: &str,
    ) -> Option<(PathBuf, Vec<u8>)> {
        let (first, expected) = if self.segmented() {
            let segments = scan_segments(path).ok()?;
            let [only] = segments.as_slice() else {
                return None;
            };
            (
                only.path.clone(),
                SegmentHeader::line(1, &BTreeMap::new(), true).ok()?,
            )
        } else {
            let mut header = Vec::new();
            write_line(
                &mut header,
                &Entry::Session(SessionMeta::with_id(id, "", "")),
            )
            .ok()?;
            (path.to_path_buf(), header)
        };
        // Only the stable start of that first line: the rest carries per-session values.
        let expected = expected[..expected.len().min(16)].to_vec();
        is_headerless(&first, &expected).then_some((first, expected))
    }

    /// Reopen the most recent session whose recorded `cwd` matches, or create a fresh one — "my last
    /// session in this directory", which is exactly what `--continue` means on both `run` and `serve`.
    ///
    /// Not just the globally newest session, so a shared repo directory spanning multiple projects
    /// doesn't resume a stranger's unrelated session. Matching is an exact string comparison — callers
    /// are expected to have already passed `cwd` through [`canonical_cwd`], so a symlinked or
    /// trailing-slashed spelling of the same real directory still matches.
    ///
    /// Only ever reached when the caller explicitly asked to continue. It is deliberately **not** what a
    /// bare invocation does: two shells running in one repo would both land on this same store and
    /// interleave their transcripts into it, and since [`SessionStore::append_new`] is count-keyed
    /// neither process can even observe the other's writes. Callers wanting a brand-new session call
    /// [`Self::create`]; callers wanting a *named* one call [`Self::open_or_create_id`].
    ///
    /// A header-only session (no messages yet) is passed over for an older one that has a conversation:
    /// an invocation that failed after creating its session (a bad flag, an unreadable schema) must
    /// not make the next `--continue` resume nothing while the real conversation sits one entry down.
    /// Only when every match is empty is the newest empty one reused, rather than minting yet another.
    pub fn resume_latest_or_create(
        &self,
        cwd: &str,
        model: &str,
    ) -> std::io::Result<(SessionStore, Session)> {
        match self.resume_latest(cwd)? {
            Some(opened) => Ok(opened),
            None => Ok((self.create(SessionMeta::new(cwd, model))?, Session::new())),
        }
    }

    /// [`Self::resume_latest_or_create`]'s selection without the create: `None` when no session here
    /// matches `cwd`. For a caller (`run`) that must read the resumed session's model before it
    /// resolves anything model-keyed, and creates a fresh session only once the invocation has passed
    /// every check.
    pub fn resume_latest(&self, cwd: &str) -> std::io::Result<Option<(SessionStore, Session)>> {
        let matching: Vec<SessionMeta> =
            self.list()?.into_iter().filter(|m| m.cwd == cwd).collect();
        let latest = matching
            .iter()
            .find(|m| m.message_count > 0)
            .or_else(|| matching.first());
        latest.map(|meta| self.open_id(&meta.id)).transpose()
    }

    /// [`Self::open_or_create_id`]'s open half: the session named exactly `id`, or `None` if there is
    /// none (no prefix match — see [`Self::find_path_exact`]).
    pub fn open_id_if_exists(&self, id: &str) -> std::io::Result<Option<(SessionStore, Session)>> {
        self.find_path_exact(id)?
            .map(|path| self.open_path(path, id))
            .transpose()
    }

    /// List every session across every project's own repo directory — each immediate subdirectory of
    /// `sessions_root` is treated as one project's [`SessionRepo`] (the convention `serve`'s default
    /// session directory follows: `<sessions_root>/<encoded-cwd>/`). Unlike [`Self::list`], which is
    /// scoped to one project, this is pi's cross-project `listAll`: each session's own `cwd` field
    /// (recorded at creation) already identifies which project it belongs to, so callers don't need the
    /// subdirectory name itself. A missing root is an empty list, not an error (nothing has ever been
    /// persisted there yet); a subdirectory that isn't a valid repo, or can't be read, contributes
    /// nothing rather than failing the whole scan — the same per-entry skip semantics [`Self::list`]
    /// already applies one level down, one level up.
    pub fn list_all(sessions_root: &Path) -> std::io::Result<Vec<SessionMeta>> {
        Self::list_all_with_progress(sessions_root, |_, _| {})
    }

    /// Same as [`Self::list_all`], but reports scan progress — see [`Self::list_with_progress`], whose
    /// parallel-scan strategy this shares: every project directory's `.jsonl` files are gathered up
    /// front into one flat list, then scanned together across one worker pool instead of one project
    /// (and its own pool) at a time, so `on_progress`'s `total` — and the parallelism — spans the whole
    /// root, not just whichever project happens to be scanning at a given moment.
    pub fn list_all_with_progress(
        sessions_root: &Path,
        on_progress: impl Fn(usize, usize) + Send + Sync,
    ) -> std::io::Result<Vec<SessionMeta>> {
        let entries = match fs::read_dir(sessions_root) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };
        let mut paths = Vec::new();
        for entry in entries {
            let Ok(entry) = entry else { continue };
            let project_dir = entry.path();
            if !project_dir.is_dir() {
                continue;
            }
            let Ok(project_entries) = fs::read_dir(&project_dir) else {
                continue;
            };
            paths.extend(
                project_entries
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("jsonl")),
            );
        }
        let mut metas = scan_listings(paths, &on_progress);
        metas.sort_by(by_recency);
        Ok(metas)
    }

    /// The on-disk path a session with this metadata would live at within this repo. `pub(crate)`: also
    /// used by [`fork_by_arg`] to locate a source session discovered via [`Self::list_all`] in some
    /// *other* project's own repo (a `SessionRepo` opened just long enough to compute the path, not to
    /// hold onto).
    pub(crate) fn path_for(&self, meta: &SessionMeta) -> PathBuf {
        self.path_for_id(&meta.id, meta.created_at)
    }

    /// The path an id resolves to. The segmented layout needs no creation timestamp in the name — the
    /// directory *is* the id — which is also what makes a lookup a single `stat` rather than a scan.
    pub(super) fn path_for_id(&self, id: &str, created_at: u64) -> PathBuf {
        if self.segmented() {
            self.dir.join(id)
        } else {
            self.dir.join(format!("{created_at}_{id}.jsonl"))
        }
    }

    /// Create a new, empty session and return its store.
    pub fn create(&self, meta: SessionMeta) -> std::io::Result<SessionStore> {
        SessionStore::create_with(self.path_for(&meta), meta, self.layout())
    }

    /// All sessions' metadata, most-recently-active first (by `updated_at` — matches pi's own session
    /// list, sorted by `modified`). Each entry carries the derived listing fields (`updated_at`,
    /// `message_count`, `preview`). Files that fail to read, lack a header, or carry an
    /// unreadable version are skipped.
    pub fn list(&self) -> std::io::Result<Vec<SessionMeta>> {
        self.list_with_progress(|_, _| {})
    }

    /// Same as [`Self::list`], but invokes `on_progress(scanned, total)` once per file as the scan
    /// completes it (`total` known up front from the directory listing; `scanned` counts up
    /// monotonically to it, including files that turn out unreadable — those still count as scanned,
    /// just contribute nothing to the result). Each file's `read_listing` is pure disk I/O plus parsing
    /// with no cross-file dependency, so the scan fans out across a small worker pool instead of running
    /// one file at a time — a repo directory with hundreds of sessions (or, via
    /// [`Self::list_all_with_progress`], hundreds spread across many projects) scans in wall-clock time
    /// bounded by how many can run concurrently, not by their sum. A client can surface `on_progress` as
    /// a live "scanning…" indicator for a listing large enough to take a moment.
    pub fn list_with_progress(
        &self,
        on_progress: impl Fn(usize, usize) + Send + Sync,
    ) -> std::io::Result<Vec<SessionMeta>> {
        let mut paths = Vec::new();
        for entry in fs::read_dir(&self.dir)? {
            let path = entry?.path();
            if self.is_session_path(&path) {
                paths.push(path);
            }
        }
        let mut metas = scan_listings_in(paths, self.layout(), &on_progress);
        metas.sort_by(by_recency);
        Ok(metas)
    }

    /// Whether a directory entry names a session in this repo's layout: a `*.jsonl` file, or a
    /// directory holding at least one segment (which excludes `.trash/` and a half-created id).
    pub(super) fn is_session_path(&self, path: &Path) -> bool {
        if self.segmented() {
            path.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| !n.starts_with('.'))
                && segment_dir_is_session(path)
        } else {
            path.extension().and_then(|e| e.to_str()) == Some("jsonl")
        }
    }

    /// Whether `id` names a session here, matched **exactly** — no unique-prefix fallback (see
    /// [`Self::find_path_exact`]). What a caller addressing a minted id asks before acting on it, so
    /// "that session is gone" cannot resolve to a neighbour of it.
    pub fn has_exact(&self, id: &str) -> std::io::Result<bool> {
        Ok(self.find_path_exact(id)?.is_some())
    }

    /// Open a session by id.
    pub fn open_id(&self, id: &str) -> std::io::Result<(SessionStore, Session)> {
        let path = self.find_path(id)?.ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, format!("no session {id}"))
        })?;
        self.open_path(path, id)
    }

    /// Open a session by id as a read-only *source* — a fork's origin, a preview, an export. None
    /// of these owns the session, and in the segmented layout a stray write from one would take an
    /// epoch off whoever does.
    pub fn open_id_read_only(&self, id: &str) -> std::io::Result<(SessionStore, Session)> {
        let path = self.find_path(id)?.ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, format!("no session {id}"))
        })?;
        self.open_source(path, id)
    }

    /// Delete a session by id. Idempotent per the repo invariant "check before destroy; don't error if
    /// it's gone": deleting an absent (or already-deleted) session is a successful no-op.
    ///
    /// Prefers a soft delete: moves the file into a `.trash` subdirectory rather than removing it
    /// outright, so a session deleted by mistake is still recoverable. `.trash` is itself a directory
    /// (no `.jsonl` extension), so [`Self::list`]'s flat, extension-filtered scan already excludes it
    /// with no further filtering needed. Falls back to a hard delete if the trash directory can't be
    /// created or the move fails for any reason (a read-only filesystem, a cross-device rename) —
    /// losing the undo, not the delete itself.
    pub fn delete(&self, id: &str) -> std::io::Result<()> {
        // An ambiguous prefix is a real error (surfaced to the caller so they can be more specific), not
        // treated as "nothing to delete" — only a genuine zero-match lookup is a no-op.
        let Some(path) = self.find_path(id)? else {
            return Ok(());
        };
        if self.segmented() {
            return self.delete_segmented(&path, id);
        }
        if let Some(file_name) = path.file_name() {
            let trash_dir = self.dir.join(".trash");
            if fs::create_dir_all(&trash_dir).is_ok()
                && fs::rename(&path, trash_dir.join(file_name)).is_ok()
            {
                // The `/session` working-memory dir trashes alongside its session (best-effort).
                move_sibling_memory(&path, &trash_dir);
                return Ok(());
            }
        }
        match fs::remove_file(&path) {
            // Hard-delete fallback: reap the sibling working-memory dir too so it doesn't leak.
            Ok(()) => {
                remove_sibling_memory(&path);
                Ok(())
            }
            // Raced with another deleter: the file vanished between `find_path` and `remove`. Still a
            // no-op success — the post-condition ("no session with this id") holds either way.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e),
        }
    }

    /// Soft-delete a segmented session: take its lock (so a live owner's session is never yanked out
    /// from under it), then rename the whole directory — segments, lock and `memory/` together — to
    /// `.trash/<id>.<unix_ns>`. The timestamp suffix is what makes repeated deletes of the same id
    /// work: each lands on its own name instead of colliding with the last one.
    pub(super) fn delete_segmented(&self, path: &Path, id: &str) -> std::io::Result<()> {
        let Some(_lock) = acquire_session_lock(path)? else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                format!("session {id} is in use and cannot be deleted"),
            ));
        };
        let trash_dir = self.dir.join(".trash");
        fs::create_dir_all(&trash_dir)?;
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        for ext in MCP_EVENTS_SIDECARS {
            let events = path.with_extension(ext);
            if events.is_file() {
                let _ = fs::rename(&events, trash_dir.join(format!("{id}.{stamp}.{ext}")));
            }
        }
        match fs::rename(path, trash_dir.join(format!("{id}.{stamp}"))) {
            Ok(()) => Ok(()),
            // Raced with another deleter: it is already gone, which is the post-condition.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            // Losing the undo is better than losing the delete (a read-only `.trash`, a cross-device
            // rename) — same fallback the single-file layout takes.
            Err(_) => fs::remove_dir_all(path).or_else(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    Ok(())
                } else {
                    Err(e)
                }
            }),
        }
    }

    /// List every session currently sitting in this repo's `.trash/` subdirectory (see [`Self::delete`]),
    /// most-recently-deleted first — pi-parity gap (Fix 7): `delete` has always soft-deleted into
    /// `.trash/`, but nothing anywhere read, listed, restored from, or pruned it, so a mistaken delete
    /// was recoverable only by reaching for a shell. Deliberately minimal (id, when it was trashed, and
    /// where it would be restored to) rather than a full trash-management surface — a client wanting more
    /// (title/preview/message count) can always resolve `id` back through `--session`/`switch_session`'s
    /// own machinery once restored.
    ///
    /// Returns an empty list (not an error) when `.trash/` doesn't exist yet — nothing has ever been
    /// deleted here.
    pub fn list_trash(&self) -> std::io::Result<Vec<TrashEntry>> {
        let trash_dir = self.dir.join(".trash");
        let read_dir = match fs::read_dir(&trash_dir) {
            Ok(rd) => rd,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };
        let mut entries = Vec::new();
        for entry in read_dir {
            let entry = entry?;
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            // The segmented layout trashes a whole directory as `<id>.<unix_ns>`; the single-file one
            // keeps the original `<created_at>_<id>.jsonl` name.
            let (id, original) = if self.segmented() {
                let Some((id, stamp)) = name.rsplit_once('.') else {
                    continue;
                };
                if id.is_empty() || !stamp.bytes().all(|b| b.is_ascii_digit()) {
                    continue;
                }
                (id.to_string(), self.dir.join(id))
            } else {
                if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                    continue;
                }
                let Some(id) = file_id(&path) else { continue };
                (id.to_string(), self.dir.join(name))
            };
            // Best-effort: a filesystem that can't report an mtime (rare) just leaves this `None` rather
            // than failing the whole listing over one unreadable timestamp.
            let deleted_at = entry
                .metadata()
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs());
            entries.push(TrashEntry {
                id,
                deleted_at,
                original_path: original.to_string_lossy().into_owned(),
            });
        }
        entries.sort_by(|a, b| b.deleted_at.cmp(&a.deleted_at));
        Ok(entries)
    }

    /// Move `id` back out of this repo's `.trash/` subdirectory to its original location (see
    /// [`Self::delete`]/[`Self::list_trash`]). Exact-id match only (unlike `delete`'s own
    /// prefix-fallback via `find_path`) — a client restoring an id almost always has the exact one
    /// straight from `list_trash`'s own output, so the extra ambiguity surface isn't worth it here.
    ///
    /// `Ok(false)` when nothing in `.trash/` matches `id` — not found is not an error, matching
    /// `delete`'s own idempotent convention; unlike `delete`, though, there's no meaningful "already
    /// restored" no-op to collapse into `Ok(())`, so the caller must check this to know whether
    /// anything actually happened. Fails clearly (rather than silently overwriting) if a session already
    /// occupies the destination path — e.g. a new session was created with the same id after the delete.
    pub fn restore_session(&self, id: &str) -> std::io::Result<bool> {
        let trash_dir = self.dir.join(".trash");
        let read_dir = match fs::read_dir(&trash_dir) {
            Ok(rd) => rd,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(e) => return Err(e),
        };
        // Match the id *component*, never an `_<id>.jsonl` suffix: an id may legally contain `_`
        // ([`is_valid_session_id`]), so a suffix test would let a restore of `b` pick up the session
        // actually named `a_b`. The segmented layout's `<id>.<unix_ns>` names are matched the same
        // way, newest first, so restoring an id twice-deleted brings back the most recent copy.
        let mut matched: Option<(PathBuf, u128)> = None;
        for entry in read_dir {
            let path = entry?.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let stamp = if self.segmented() {
                match name.rsplit_once('.') {
                    Some((found, stamp)) if found == id => stamp.parse().unwrap_or(0),
                    _ => continue,
                }
            } else {
                if file_id(&path) != Some(id) {
                    continue;
                }
                0
            };
            if matched.as_ref().is_none_or(|(_, best)| stamp >= *best) {
                matched = Some((path, stamp));
            }
        }
        let Some((path, _)) = matched else {
            return Ok(false);
        };
        let dest = if self.segmented() {
            self.dir.join(id)
        } else {
            // `path` was just built from a real directory entry's own file name, so this is always
            // `Some` in practice — but production code here stays panic-free regardless (workspace
            // lint), so a `None` (which should never happen) is a clear error instead of a panic.
            let Some(file_name) = path.file_name() else {
                return Err(std::io::Error::other(format!(
                    "trash entry for {id} has no file name: {}",
                    path.display()
                )));
            };
            self.dir.join(file_name)
        };
        if dest.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!(
                    "cannot restore session {id}: {} already exists",
                    dest.display()
                ),
            ));
        }
        fs::rename(&path, &dest)?;
        if !self.segmented() {
            // Bring the session's `/session` working-memory dir back out of `.trash/` too (best-effort
            // — a missing dir just means the session had no working memory, or an older delete predates
            // this). A segmented session's memory lives *inside* its directory and already moved.
            move_sibling_memory(&path, &self.dir);
        }
        Ok(true)
    }

    /// Fork session `id` at `upto` messages: a new session whose transcript is the first `upto`
    /// messages of the original, linked back via `parent`. `upto` is clamped to the source length, so
    /// `usize::MAX` clones the whole session. Returns the new store and its restored session.
    pub fn fork(&self, id: &str, upto: usize) -> std::io::Result<(SessionStore, Session)> {
        let (src, src_session) = self.open_id_read_only(id)?;
        let upto = upto.min(src_session.messages.len());
        // Task #18 (pi-parity fix): the model actually active at the copied prefix's own last message
        // — not `src.meta.model` (the source's creation-time value, blindly copied here previously) —
        // see `model_at_or_created`'s own doc comment for why that matters once a `set_model` has
        // happened anywhere relative to the fork point.
        let target_id = (upto > 0)
            .then(|| src.active_ids().get(upto - 1))
            .flatten()
            .map(String::as_str);
        let model = src.model_at_or_created(target_id).to_string();
        let mut meta = self.derived_meta(src.meta.cwd.clone(), model);
        // The source's own resolved id, not the caller's raw `id` argument — since `open_id` now accepts
        // a unique prefix, blindly echoing `id` back would persist the *prefix* as `parent` instead of
        // the real full id it resolved to.
        meta.parent = Some(src.meta.id.clone());
        // Pass 15 (pi-parity fix): the title actually in effect at that same copied prefix's own last
        // message — not `src.meta.title` (the source's whole-file-latest rename, blindly copied here
        // previously) — see `title_at_or_root`'s own doc comment for why that matters once a rename has
        // happened anywhere relative to the fork point.
        meta.title = src.title_at_or_root(target_id).map(str::to_string);
        // Thinking-level analogue of the model/title fixes just above — see
        // `thinking_level_at_or_created`'s own doc comment.
        let thinking_level = src
            .thinking_level_at_or_created(target_id)
            .map(str::to_string);
        meta.thinking_level = thinking_level.clone();

        let mut store = self.create(meta)?;
        let prefix: Vec<Message> = src_session.messages[..upto].to_vec();
        store.rewrite(&prefix)?;
        // Re-anchor the resolved level onto the new session's own tree root — *after* `rewrite` above,
        // never before: `rewrite` is a full temp-file-then-rename swap of the whole file (see its own
        // doc comment), not an append, so anything appended earlier would just be silently discarded
        // by it. `seed_thinking_level_at_root` anchors at `None` (the tree's own root) rather than
        // `record_thinking_level_change`'s usual "wherever the tip currently is" — see that method's
        // own doc comment for why: anchoring at the new chain's own tip would put it exactly where
        // `change_at`'s "anchored-at, not before" exclusion never sees it once queried at that same
        // tip, the precise query `Persistence::model_and_level_at_active` (`serve/persistence.rs`) makes on every
        // reopen. Without this, a fresh reopen of the forked file (`run --continue`, a `serve`
        // restart, `switch_session` back to it) finds nothing recorded on the fork's brand-new chain
        // at all — the source's own `ThinkingLevelChange` entries are anchored to the *source*'s old
        // message ids, which don't exist in this file — and silently falls back to the process's own
        // starting level instead of the level actually in effect at the fork point.
        if let Some(level) = &thinking_level {
            store.seed_thinking_level_at_root(level)?;
        }
        let mut session = Session::new();
        session.messages = Arc::new(prefix);
        Ok((store, session))
    }

    /// Fork an arbitrary source session — one that need not live in `self.dir` at all — into a
    /// brand-new session under `self`. This is [`fork`](Self::fork) generalized for pi's cross-project
    /// `--fork <path|id>`: `source_path` is a fully resolved on-disk path (see [`fork_by_arg`], which
    /// finds it by searching the current project and then every other project's own session directory),
    /// and `target_cwd` becomes the new session's own `cwd` — the project being forked *into*, not
    /// wherever the source was originally recorded against — matching pi's own
    /// `SessionManager.forkFrom(path, targetCwd, …)`.
    pub fn fork_from_path(
        &self,
        source_path: &Path,
        target_cwd: &str,
        upto: usize,
    ) -> std::io::Result<(SessionStore, Session)> {
        let (src, src_session) =
            SessionStore::open_with(source_path.to_path_buf(), None, self.layout(), true)?;
        let upto = upto.min(src_session.messages.len());
        // Task #18 (pi-parity fix): same reasoning as `fork`'s identical resolution just above.
        let target_id = (upto > 0)
            .then(|| src.active_ids().get(upto - 1))
            .flatten()
            .map(String::as_str);
        let model = src.model_at_or_created(target_id).to_string();
        let mut meta = self.derived_meta(target_cwd.to_string(), model);
        meta.parent = Some(src.meta.id.clone());
        // Pass 15 (pi-parity fix): same reasoning as `fork`'s identical resolution just above.
        meta.title = src.title_at_or_root(target_id).map(str::to_string);
        // Same reasoning as `fork`'s identical resolution just above — see
        // `thinking_level_at_or_created`'s own doc comment.
        let thinking_level = src
            .thinking_level_at_or_created(target_id)
            .map(str::to_string);
        meta.thinking_level = thinking_level.clone();

        let mut store = self.create(meta)?;
        let prefix: Vec<Message> = src_session.messages[..upto].to_vec();
        store.rewrite(&prefix)?;
        // See `fork`'s identical re-anchoring for why this must happen after `rewrite` above, not
        // before.
        if let Some(level) = &thinking_level {
            store.seed_thinking_level_at_root(level)?;
        }
        let mut session = Session::new();
        session.messages = Arc::new(prefix);
        Ok((store, session))
    }

    /// Fork session `id` at an arbitrary tree entry `entry_id` — anywhere in the *whole* tree, on or
    /// off whatever the active path currently is, unlike [`fork`](Self::fork)'s active-path-only
    /// `upto` count. Mirrors pi's `createBranchedSession(leafId)`: `before` (pi's `position:"before"`)
    /// excludes `entry_id` itself from the forked prefix — forking right before a message the caller
    /// wants to redo, rather than keep; `false` (pi's `"at"`, the default) includes it. Errors
    /// (`NotFound`) if `entry_id` names no known message in `id`'s tree.
    ///
    /// Labels carry forward (pi-parity C-M1, `session-manager/labels.test.ts`'s "labels are preserved
    /// in createBranchedSession"): any label recorded in the source session against a message that
    /// survives into the forked prefix is re-applied in the new store, against that message's *new* id
    /// — [`rewrite`](Self::rewrite) mints fresh ids for a forked copy (it has to: see its own doc
    /// comment on why a rewrite can't in general reuse originals), so `fork_at_entry_prefix`'s
    /// `original_ids` (positionally parallel to the prefix `rewrite` just wrote) is what maps each old
    /// id to its replacement. A label whose target fell *outside* the forked prefix (off `entry_id`'s
    /// path, or past it when `before` excludes it) has no surviving id to re-attach to and is simply
    /// dropped, matching pi's "labels not on path are not preserved."
    ///
    /// Custom entries (Track C-M2, [`Entry::Custom`]) on the forked path are **not** carried into the
    /// new session — the copied prefix is a plain `Vec<Message>` fed through the ordinary
    /// [`rewrite`](Self::rewrite), which only ever produces message nodes; a custom entry's opaque
    /// `data` has no message representation to carry forward. This matches this method's existing
    /// precedent for every *other* side-channel entry type (`ModelChange`/`ThinkingLevelChange`/
    /// `BranchSummary` provenance): forking already only ever preserves the message content itself,
    /// never the surrounding bookkeeping — labels are carried forward as a deliberate, narrower
    /// exception (see above), not a general rule this extends.
    pub fn fork_at_entry(
        &self,
        id: &str,
        entry_id: &str,
        before: bool,
    ) -> std::io::Result<(SessionStore, Session)> {
        let prefix = self.fork_at_entry_prefix(id, entry_id, before)?;
        let mut store = self.create(prefix.meta)?;
        store.rewrite(&prefix.messages)?;
        // See `fork`'s identical re-anchoring for why this must happen after `rewrite` above, not
        // before — anchors at the tree's own root rather than the source's now-nonexistent message
        // ids.
        if let Some(level) = &prefix.thinking_level {
            store.seed_thinking_level_at_root(level)?;
        }
        if !prefix.labels.is_empty() {
            let new_ids: Vec<String> = store.active_ids().to_vec();
            debug_assert_eq!(new_ids.len(), prefix.original_ids.len());
            let old_to_new: HashMap<&str, &str> = prefix
                .original_ids
                .iter()
                .map(String::as_str)
                .zip(new_ids.iter().map(String::as_str))
                .collect();
            for (target_id, label) in &prefix.labels {
                if let Some(&new_id) = old_to_new.get(target_id.as_str()) {
                    store.set_label(new_id, Some(label))?;
                }
            }
        }
        let mut session = Session::new();
        session.messages = Arc::new(prefix.messages);
        Ok((store, session))
    }

    /// Compute what [`fork_at_entry`](Self::fork_at_entry) would write, without writing it: the
    /// would-be child's metadata, message prefix, the source's own ids for that prefix (positionally
    /// parallel — see [`ForkPrefix::original_ids`]), and any labels found within it. Pure/in-memory
    /// beyond the initial `open_id` read — no file is created. Used both by `fork_at_entry` itself (to
    /// avoid duplicating the prefix logic) and by [`fork_at_entry_messages`](Self::fork_at_entry_messages)
    /// for side-effect-free previews.
    ///
    /// Investigated (Round 2 pi-parity remediation, Low severity, left unfixed): `entry_id` must name a
    /// real tree node (a key in `nodes` — a `Message`/`BranchSummary`/`Custom` entry), so it 404s for
    /// one of the anchored side-channel entries (`Entry::Label`/`ModelChange`/`ThinkingLevelChange`/
    /// `TitleChange`) below `nodes.contains_key` — those deliberately never occupy a tree slot at all
    /// (see [`Entry::Label`]'s own doc comment on why). Pi's own uniform tree model allows forking at
    /// any entry id, including these; beyond's narrower one doesn't.
    ///
    /// Assessed as confirmed-real but disproportionate to fix, the same judgment call already made for
    /// the abandoned-branch export-nesting edge case (see `orphaned_off_branch_survives_compaction_of_its_ancestor`'s
    /// sibling test, "confirms Task #34"): no known workflow needs this, and — more than
    /// "unconfirmed" — it's provably unreachable through any surface this crate currently exposes.
    /// [`Self::tree`] (a client's only way to discover an entry id worth forking at) synthesizes
    /// `TreeNode`s solely from `nodes` and `self.compactions`; a label is surfaced only as the `label`
    /// *field* of the node it targets, never as its own addressable entry, and `ModelChange`/
    /// `ThinkingLevelChange`/`TitleChange` aren't surfaced as tree nodes at all. So no client can ever
    /// have actually *seen* one of these entries' own ids to pass back in here in the first place —
    /// there is no live caller this gap could affect today. Properly supporting it would mean first
    /// giving each of these four variants its own discoverable identity in `tree()` (a real new surface,
    /// not a bug fix) and then reworking this method's resolution to translate such an id to the
    /// nearest real tree position — a meaningful reshaping of the entry model for a target no code path
    /// can currently reach. Left as documented, deliberately unfixed.
    pub(super) fn fork_at_entry_prefix(
        &self,
        id: &str,
        entry_id: &str,
        before: bool,
    ) -> std::io::Result<ForkPrefix> {
        let (src, _src_session) = self.open_id_read_only(id)?;
        if !src.nodes.contains_key(entry_id) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("no message with id {entry_id} in session {id}"),
            ));
        }
        // `before` (now the default — see `serve/commands/`'s `fork`/`preview_fork` handlers) means "fork
        // right before this entry", which only makes sense anchored to a user turn: popping a
        // non-user entry (an assistant reply, a materialized branch-summary recap, or a custom entry)
        // would silently land the fork one entry earlier than the caller actually asked for, with no
        // way for them to tell. pi's own `getEntriesToFork` (`repo-utils.ts`) rejects the same case as
        // `SessionError("invalid_fork_target")` — mirrored here rather than guessing.
        if before {
            let is_user_message = src
                .nodes
                .get(entry_id)
                .and_then(Node::as_message)
                .is_some_and(|m| m.role == Role::User);
            if !is_user_message {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("invalid_fork_target: entry {entry_id} is not a user message"),
                ));
            }
        }

        let mut path = path_from_root(&src.nodes, Some(entry_id));
        if before {
            path.pop();
        }
        // Task #18 (pi-parity fix): the model actually active at the resolved fork point (`path`'s own
        // last id post-`before`-adjustment, or the tree's root if `before` popped the very first entry)
        // — not `src.meta.model` (the source's creation-time value) — see `model_at_or_created`'s own
        // doc comment. Computed *after* the `before` adjustment above, since that's what determines the
        // actual point being forked at.
        let model = src
            .model_at_or_created(path.last().map(String::as_str))
            .to_string();
        let mut meta = self.derived_meta(src.meta.cwd.clone(), model);
        // The source's own resolved id, not the caller's raw `id` argument — see `fork`'s identical fix
        // for why: `open_id` now accepts a unique prefix, so `id` itself may not be the real full id.
        meta.parent = Some(src.meta.id.clone());
        // Pass 15 (pi-parity fix): same reasoning as `fork`'s identical resolution — the title actually
        // in effect at the resolved fork point, not `src.meta.title`'s whole-file-latest rename.
        meta.title = src
            .title_at_or_root(path.last().map(String::as_str))
            .map(str::to_string);
        // Same reasoning as `fork`'s identical resolution — the thinking level actually in effect at
        // the resolved fork point. See `thinking_level_at_or_created`'s own doc comment.
        let thinking_level = src
            .thinking_level_at_or_created(path.last().map(String::as_str))
            .map(str::to_string);
        meta.thinking_level = thinking_level.clone();

        // Labels are looked up against the *full* path (including any custom entry's id) — a label
        // whose target is a custom entry that then gets filtered out below simply won't be found in
        // `fork_at_entry`'s old-id → new-id map, and is correctly dropped the same way an
        // off-path label already is. `original_ids`/`messages` below, by contrast, must stay message-
        // only and positionally parallel to each other: `rewrite` (called next, on `messages`) only
        // ever produces message nodes, so a custom entry has no counterpart id in the destination
        // store to carry a label forward to anyway (Track C-M2 forks don't carry custom entries
        // themselves — see `SessionRepo::fork_at_entry`'s doc comment).
        let labels = src.labels_within(&path);
        let mut original_ids = Vec::with_capacity(path.len());
        let mut messages = Vec::with_capacity(path.len());
        for id in &path {
            if let Some(m) = src.nodes[id].as_message() {
                original_ids.push(id.clone());
                messages.push(m.clone());
            }
        }
        Ok(ForkPrefix {
            meta,
            messages,
            original_ids,
            labels,
            thinking_level,
        })
    }

    /// Preview what [`fork_at_entry`](Self::fork_at_entry) would produce — the exact message prefix —
    /// without creating a new session file. The read-only counterpart to `fork_at_entry`, for a client
    /// browsing a fork point before committing to it (see `fork_messages` in `serve/persistence.rs`).
    pub fn fork_at_entry_messages(
        &self,
        id: &str,
        entry_id: &str,
        before: bool,
    ) -> std::io::Result<Vec<Message>> {
        Ok(self.fork_at_entry_prefix(id, entry_id, before)?.messages)
    }

    /// Every `.jsonl`-or-otherwise path directly under this repo's directory, unfiltered. A missing
    /// directory is an empty list, not an error — nothing has been persisted here yet, which every
    /// lookup below already treats as "no match" rather than a failure.
    pub(super) fn session_paths(&self) -> std::io::Result<Vec<PathBuf>> {
        match fs::read_dir(&self.dir) {
            Ok(entries) => Ok(entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| !self.segmented() || self.is_session_path(p))
                .collect()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(e) => Err(e),
        }
    }

    /// The id a directory entry names, in this repo's layout: the `<id>` component of a
    /// `<created_at>_<id>.jsonl`, or the session directory's own name.
    pub(super) fn entry_id<'a>(&self, path: &'a Path) -> Option<&'a str> {
        if self.segmented() {
            path.file_name()?.to_str()
        } else {
            file_id(path)
        }
    }

    /// The path whose `<id>` component is exactly `id`, from an already-gathered listing. Filename-only
    /// — no file is opened, so this is a string compare per directory entry rather than a parse.
    ///
    /// Compares the parsed component, not an `_<id>.jsonl` *suffix*: ids may legally contain `_`
    /// ([`is_valid_session_id`]), so a suffix test lets a lookup for `b` match a session actually named
    /// `a_b`. Harmless while this only backed a convenience lookup; not harmless now that it decides
    /// which session an address resolves to.
    pub(super) fn exact_match(&self, entries: &[PathBuf], id: &str) -> Option<PathBuf> {
        entries
            .iter()
            .find(|p| self.entry_id(p) == Some(id))
            .cloned()
    }

    /// Resolve `id` to its on-disk path by **exact** match only, with no unique-prefix fallback.
    ///
    /// This is the lookup an *addressing* caller needs — `--session-id`, `serve`'s `?session_id=` — as
    /// opposed to [`Self::find_path`]'s human-convenience resolution. The distinction is not cosmetic:
    /// under prefix matching, asking for `abc` when `abcdef` already exists would both hand back the
    /// wrong session *and* suppress creating the one actually asked for, so a caller minting ids that
    /// happen to share a prefix would silently collapse onto whichever landed first. Exact-or-absent
    /// keeps "open it, else create it under this exact name" total and unambiguous.
    pub(super) fn find_path_exact(&self, id: &str) -> std::io::Result<Option<PathBuf>> {
        if self.segmented() {
            // The directory *is* the id, so an address resolves with one `stat` — no directory scan,
            // however many sessions the shard holds.
            let path = self.dir.join(id);
            return Ok(segment_dir_is_session(&path).then_some(path));
        }
        Ok(self.exact_match(&self.session_paths()?, id))
    }

    /// Resolve `id` to its on-disk path in this repo: an exact match first (cheap, unambiguous), then a
    /// unique-prefix match — pi's own convenience for typing a shortened id (`main.ts`'s
    /// `resolveSessionPath`), but *not* pi's own silent "pick whichever sorts first" when a prefix
    /// matches more than one session: that's a real footgun (operating on session B when the caller
    /// typed a prefix meaning to reach session A), so an ambiguous prefix is an error naming every
    /// candidate instead of a guess. `Ok(None)` when nothing matches at all — not found is not an error
    /// here, matching every existing caller's own "session may not exist" handling.
    pub(super) fn find_path(&self, id: &str) -> std::io::Result<Option<PathBuf>> {
        if let Some(path) = self.find_path_exact(id)? {
            return Ok(Some(path));
        }
        let entries = self.session_paths()?;
        // No exact match: fall back to a unique-prefix match over each file's own `<id>` component.
        let matches: Vec<(&str, &PathBuf)> = entries
            .iter()
            .filter_map(|path| {
                let file_id = self.entry_id(path)?;
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
}
