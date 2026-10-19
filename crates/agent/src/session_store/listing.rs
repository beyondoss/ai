use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::UNIX_EPOCH;

use agent_core::{ContentBlock, Message, Role};
use serde::{Deserialize, Serialize};

use super::*;

/// The sidecar listing index: one per session directory, next to the `.jsonl` files it describes.
///
/// A listing needs a handful of derived fields (preview, message count, `updated_at`, search text), and
/// the only way to recover them from a transcript is to read it end to end — `read_listing` is a
/// streaming parse, but it is still O(bytes), and a session file grows without bound as its
/// conversation does. Measured: listing a 13 MB session costs ~14 ms, of which only ~15% is reading the
/// bytes; the other ~85% is `serde_json` *tokenizing* payload it then throws away. That's why no parser
/// trick fixes this — skipping materialization doesn't skip tokenization (both were tried and both
/// bought 0% wall time). The only way to make the scan cheaper is to not do it.
///
/// So each file's computed `SessionMeta` is cached here against the `(size, mtime)` it was computed
/// from. A session file is append-only, so any change to it moves both — a stale entry can't survive a
/// write. On a warm index a listing costs one `stat` per file instead of a full parse, which turns the
/// scan from O(total transcript bytes) into O(sessions).
///
/// It is a **cache, and only a cache**: every read is best-effort, and any failure (missing, corrupt,
/// unknown version, unreadable, unwritable directory) simply falls back to scanning the file, which is
/// exactly what the code did before this existed. Nothing here is a source of truth, so a lost or
/// clobbered index costs time, never correctness — which is also why concurrent writers need no locking
/// beyond the atomic rename below.
pub(super) const LISTING_INDEX_FILE: &str = ".listings.json";

/// Bumped whenever a change to the derived fields **or to [`Stamp`]** would make previously-cached
/// entries wrong (a new field, a different preview rule, a stamp that means something else). An index
/// at any other version is ignored wholesale and rebuilt, so an upgrade can never serve stale-shaped
/// metadata — or, worse, validate a v1 stamp against a v2 one and serve a listing for a session that
/// has since moved on.
///
/// v2: [`Stamp`] became the newest segment's `(len, mtime, epoch)` plus the segment count, where v1
/// held the directory's total bytes, newest mtime, newest epoch and base epoch.
///
/// v3: `search_text` is gone. A v2 index holds up to 50 KB of every session's conversation text, so
/// the bump is not only about shape — it is what gets that text deleted from the shard on the next
/// listing, rather than left sitting in a cache nothing reads any more.
pub(super) const LISTING_INDEX_VERSION: u32 = 3;

#[derive(Deserialize)]
pub(super) struct ListingIndex {
    pub(super) version: u32,
    /// Keyed by file *name*, not full path — the index lives in the directory it describes, so the
    /// directory component is implied, and the file stays valid if the whole tree is moved.
    pub(super) entries: HashMap<String, ListingIndexEntry>,
}

/// Write-side mirror of [`ListingIndex`] that **borrows** the entries map instead of owning it — so
/// [`store_listing_index`] can serialize the index directly out of the caller's `HashMap` without
/// cloning every entry just to wrap it in a `version` field. Serialize
/// only; the read path still deserializes the owned [`ListingIndex`].
#[derive(Serialize)]
pub(super) struct ListingIndexRef<'a> {
    pub(super) version: u32,
    pub(super) entries: &'a HashMap<String, ListingIndexEntry>,
}

/// One file's cached listing.
///
/// The four derived fields are carried **explicitly** rather than riding along inside `meta`, because
/// `SessionMeta` marks every one of them `#[serde(skip)]` — deliberately, so they can never go stale
/// inside the on-disk *header* (see their doc comments). That is the right call there and exactly the
/// wrong one here: serializing a `SessionMeta` into this index would silently drop the only fields the
/// index exists to remember, and hand back a listing with `updated_at: 0` and no preview. Naming them
/// here keeps the header's invariant intact while letting the cache do its job, and means adding a fifth
/// derived field is a compile error in this struct rather than a silent hole in the cache.
#[derive(Clone, Serialize, Deserialize)]
pub(super) struct ListingIndexEntry {
    /// The stamp the cached listing was computed from. Every field must match for it to be used.
    pub(super) size: u64,
    pub(super) mtime_ns: u128,
    /// The newest segment's epoch and how many segments there were, for the segmented layout only —
    /// absent (and so `0`) for a single-file entry, which keeps that layout's index byte-identical to
    /// what it always wrote.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub(super) epoch: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub(super) segments: u64,
    /// The persisted header fields (everything `SessionMeta` does serialize).
    pub(super) meta: SessionMeta,
    pub(super) updated_at: u64,
    pub(super) message_count: usize,
    pub(super) preview: Option<String>,
}

pub(super) fn is_zero(v: &u64) -> bool {
    *v == 0
}

impl ListingIndexEntry {
    pub(super) fn new(stamp: Stamp, meta: &SessionMeta) -> Self {
        Self {
            size: stamp.size,
            mtime_ns: stamp.mtime_ns,
            epoch: stamp.epoch,
            segments: stamp.segments,
            meta: meta.clone(),
            updated_at: meta.updated_at,
            message_count: meta.message_count,
            preview: meta.preview.clone(),
        }
    }

    pub(super) fn stamp(&self) -> Stamp {
        Stamp {
            size: self.size,
            mtime_ns: self.mtime_ns,
            epoch: self.epoch,
            segments: self.segments,
        }
    }

    /// Reassemble the full `SessionMeta` a scan would have produced — header fields from `meta`, derived
    /// fields from the ones stored beside it.
    pub(super) fn to_meta(&self) -> SessionMeta {
        SessionMeta {
            updated_at: self.updated_at,
            message_count: self.message_count,
            preview: self.preview.clone(),
            ..self.meta.clone()
        }
    }
}

/// Best-effort load. Any problem at all yields an empty index and a full rescan.
pub(super) fn load_listing_index(
    dir: &Path,
    codec: Option<&TenantCodec>,
) -> HashMap<String, ListingIndexEntry> {
    let raw = match fs::read(dir.join(LISTING_INDEX_FILE)) {
        Ok(raw) => raw,
        Err(_) => return HashMap::new(),
    };
    // The index holds titles, previews and up to 50 KiB of every session's text, so it is sealed
    // exactly like the transcripts it summarizes. A blob that won't open is treated as no cache at
    // all (the same as a corrupt one) — it is only ever a cache.
    let raw = match codec {
        Some(c) => match c.open_listing(&raw) {
            Ok(plain) => plain,
            Err(_) => return HashMap::new(),
        },
        None => raw,
    };
    match serde_json::from_slice::<ListingIndex>(&raw) {
        Ok(idx) if idx.version == LISTING_INDEX_VERSION => idx.entries,
        _ => HashMap::new(),
    }
}

/// Write the index atomically (temp file + rename), so a reader never observes a half-written one and a
/// crash mid-write leaves the previous index intact rather than a corrupt one. No `fsync`: this is a
/// cache, and paying durability costs to protect data we can always recompute would defeat the purpose.
/// Concurrent writers race to rename and the last one wins — both wrote a valid index, so either is
/// correct.
///
/// Every error is swallowed deliberately. A read-only or unwritable session directory must still *list*;
/// it just doesn't get to keep a cache.
pub(super) fn store_listing_index(
    dir: &Path,
    entries: &HashMap<String, ListingIndexEntry>,
    codec: Option<&TenantCodec>,
) {
    let idx = ListingIndexRef {
        version: LISTING_INDEX_VERSION,
        entries,
    };
    let Ok(bytes) = serde_json::to_vec(&idx) else {
        return;
    };
    let bytes = match codec {
        Some(c) => match c.seal_listing(&bytes) {
            Ok(sealed) => sealed,
            Err(_) => return,
        },
        None => bytes,
    };
    // Unpredictable per call, not just per process: a deterministic sibling temp name is something
    // anything able to plant a symlink in this directory could aim elsewhere, and two listers inside
    // one process would tear each other's file.
    let tmp = dir.join(format!(".listings.{}.tmp", crate::tools::temp_suffix()));
    let write = (|| -> std::io::Result<()> {
        let mut f = create_private(&tmp)?;
        f.write_all(&bytes)?;
        f.flush()?;
        drop(f);
        fs::rename(&tmp, dir.join(LISTING_INDEX_FILE))
    })();
    if write.is_err() {
        let _ = fs::remove_file(&tmp);
    }
}

/// Scan every path in `paths` for its listing metadata, calling `on_progress(scanned, total)` once per
/// path — including ones that turn out unreadable, which still count as scanned and just contribute
/// nothing.
///
/// Files whose `(size, mtime)` still match the sidecar [`ListingIndex`] are served from it without being
/// opened; only the rest are actually parsed. Those misses are pure disk I/O plus parsing with no
/// dependency between files, so they fan out across a small worker pool
/// (`std::thread::available_parallelism`, capped at one thread per file) rather than running strictly one
/// at a time; below two candidate workers it just runs inline, there being no thread pool worth its setup
/// cost for a one- or two-file scan. Returned in arbitrary order — every caller sorts the result itself.
pub(crate) fn scan_listings(
    paths: Vec<PathBuf>,
    on_progress: &(impl Fn(usize, usize) + Send + Sync),
) -> Vec<SessionMeta> {
    scan_listings_in(paths, &Layout::File, on_progress)
}

/// [`scan_listings`] for a specific [`Layout`] — the segmented paths are session *directories*, and
/// the sidecar index is sealed with the same tenant codec its transcripts are.
pub(crate) fn scan_listings_in(
    paths: Vec<PathBuf>,
    layout: &Layout,
    on_progress: &(impl Fn(usize, usize) + Send + Sync),
) -> Vec<SessionMeta> {
    let codec = layout.codec().map(Arc::as_ref);
    let segmented = layout.is_segmented();
    let total = paths.len();
    if total == 0 {
        return Vec::new();
    }

    // One index per directory — `list_all` spans many project directories in a single flat scan, and
    // each keeps its own sidecar next to its own files.
    let mut indexes: HashMap<PathBuf, HashMap<String, ListingIndexEntry>> = HashMap::new();
    for p in &paths {
        if let Some(dir) = p.parent()
            && !indexes.contains_key(dir)
        {
            indexes.insert(dir.to_path_buf(), load_listing_index(dir, codec));
        }
    }

    // Split into cache hits (served from the index) and misses (must be parsed). A hit still counts as
    // scanned, so `on_progress` stays monotonic and reaches `total` exactly as before.
    let mut hits: Vec<SessionMeta> = Vec::new();
    let mut misses: Vec<PathBuf> = Vec::new();
    // The stamp each miss was read at, so the rebuilt index records what it was *actually* computed
    // from — and so `read_listing` can reuse the mtime this scan already `stat`'d instead of
    // re-`stat`ing the same file. Stamping after the parse instead would race an append landing
    // mid-scan and cache a listing against a file state it never saw.
    let mut miss_stamps: HashMap<PathBuf, Stamp> = HashMap::new();
    // The file names this scan actually produced a listing for, per directory — the oracle for dropping
    // an index entry whose session file has since been deleted. Names only, never the meta: a cache
    // hit reuses the entry already sitting in `indexes` rather than rebuilding it, so the warm path
    // clones each entry once (the `to_meta()` it must return), not twice as it used to (`to_meta()`
    // *and* `ListingIndexEntry::new`).
    let mut seen: HashMap<PathBuf, HashSet<String>> = HashMap::new();
    // Directory index sizes as loaded, so a shrink (a deleted session, whose entry must be forgotten)
    // is distinguishable from an unchanged scan without holding a copy of the old entries.
    let orig_lens: HashMap<PathBuf, usize> =
        indexes.iter().map(|(d, e)| (d.clone(), e.len())).collect();
    let mut scanned = 0usize;

    for p in &paths {
        let stamp = log_stamp(p, segmented);
        let cached = (|| {
            let stamp = stamp?;
            let dir = p.parent()?;
            let name = p.file_name()?.to_str()?;
            let e = indexes.get(dir)?.get(name)?;
            (e.stamp() == stamp).then(|| e.to_meta())
        })();
        match cached {
            Some(meta) => {
                if let (Some(dir), Some(name)) =
                    (p.parent(), p.file_name().and_then(|n| n.to_str()))
                {
                    seen.entry(dir.to_path_buf())
                        .or_default()
                        .insert(name.to_string());
                }
                hits.push(meta);
                scanned += 1;
                on_progress(scanned, total);
            }
            None => {
                if let Some(s) = stamp {
                    miss_stamps.insert(p.clone(), s);
                }
                misses.push(p.clone());
            }
        }
    }

    let mut metas = hits;
    metas.reserve(misses.len());
    for (path, meta) in scan_uncached(&misses, &miss_stamps, layout, scanned, total, on_progress) {
        // Only a file we managed to stamp *before* reading can be cached — otherwise there's nothing to
        // validate a future hit against. The stamped entry lands straight in `indexes` (owned, so the
        // meta is moved in, not re-cloned), which *is* the index this scan will persist.
        if let (Some(dir), Some(name), Some(&stamp)) = (
            path.parent(),
            path.file_name().and_then(|n| n.to_str()),
            miss_stamps.get(&path),
        ) {
            let dir = dir.to_path_buf();
            seen.entry(dir.clone())
                .or_default()
                .insert(name.to_string());
            indexes
                .entry(dir)
                .or_default()
                .insert(name.to_string(), ListingIndexEntry::new(stamp, &meta));
        }
        metas.push(meta);
    }

    // Persist the index only where this scan changed it. `indexes` already holds every hit's entry
    // untouched and every miss's freshly-built one, so all that's left is to forget entries for files
    // this scan no longer saw (a deleted session), then write back the directories that moved. The
    // write gate matches the old one exactly — rewrite when any file was parsed (`!misses.is_empty()`)
    // or when a directory's entry count shrank — so a listing that was pure cache hits with nothing
    // deleted does no write at all.
    for (dir, names) in &seen {
        let Some(entries) = indexes.get_mut(dir) else {
            continue;
        };
        entries.retain(|name, _| names.contains(name));
        let shrank = orig_lens.get(dir).is_some_and(|&old| old != entries.len());
        if !misses.is_empty() || shrank {
            store_listing_index(dir, entries, codec);
        }
    }
    metas
}

/// The parse-every-byte path, for the files the index couldn't answer for. `base` is how many paths were
/// already served from cache, so `on_progress` keeps counting up to `total` across both halves.
pub(super) fn scan_uncached(
    paths: &[PathBuf],
    stamps: &HashMap<PathBuf, Stamp>,
    layout: &Layout,
    base: usize,
    total: usize,
    on_progress: &(impl Fn(usize, usize) + Send + Sync),
) -> Vec<(PathBuf, SessionMeta)> {
    let total_uncached = paths.len();
    if total_uncached == 0 {
        return Vec::new();
    }
    let workers = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1)
        .min(total_uncached);
    if workers < 2 {
        return paths
            .iter()
            .enumerate()
            .filter_map(|(i, path)| {
                let meta = read_listing(path, mtime_from_stamps(stamps, path), layout);
                on_progress(base + i + 1, total);
                meta.map(|m| (path.clone(), m))
            })
            .collect();
    }

    let scanned = AtomicUsize::new(0);
    let metas = Mutex::new(Vec::with_capacity(total_uncached));
    let scanned_ref = &scanned;
    let metas_ref = &metas;
    let chunk_size = total_uncached.div_ceil(workers);
    std::thread::scope(|scope| {
        for chunk in paths.chunks(chunk_size) {
            scope.spawn(move || {
                for path in chunk {
                    if let Some(meta) = read_listing(path, mtime_from_stamps(stamps, path), layout)
                    {
                        metas_ref
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .push((path.clone(), meta));
                    }
                    on_progress(
                        base + scanned_ref.fetch_add(1, Ordering::Relaxed) + 1,
                        total,
                    );
                }
            });
        }
    });
    metas
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Read a session file's listing metadata: its (version-checked) header with the derived `updated_at` /
/// `message_count` / `preview` fields filled in. One streaming pass — lines are read and
/// parsed individually, never collected — so this stays light even for long transcripts (the header
/// alone gives id/title/etc.; only the count and preview need the scan). Returns `None` for
/// a file that isn't a readable session (no/invalid header, or an unreadable version), matching `list`'s
/// skip semantics.
/// Whole-second mtime derived from a `(size, mtime_ns)` stamp the caller already `stat`'d — nanos → the
/// same floor-to-seconds `mtime_secs` computes — so [`read_listing`] can skip re-`stat`ing a file the
/// listing scan just stamped. `None` when the file wasn't stamped, leaving the read path to `stat` it.
pub(super) fn mtime_from_stamps(stamps: &HashMap<PathBuf, Stamp>, path: &Path) -> Option<u64> {
    stamps
        .get(path)
        .map(|s| (s.mtime_ns / 1_000_000_000) as u64)
}

pub(crate) fn read_listing(
    path: &Path,
    mtime: Option<u64>,
    layout: &Layout,
) -> Option<SessionMeta> {
    // `mtime` is the file's whole-second modified time when a caller already `stat`'d it (a cache-miss
    // scan stamps every file it parses — see [`mtime_from_stamps`]), so threading it here avoids the
    // second `stat` `mtime_secs` would otherwise do. It's only ever consulted as the `updated_at`
    // fallback for a legacy file with no message timestamp, so a supplied value and a fresh `stat` are
    // interchangeable — the supplied one is if anything *more* consistent, being the state the rest of
    // this scan was computed against. `None` falls back to a `stat` here.
    let mtime = mtime.unwrap_or_else(|| mtime_secs(path));
    let mut reader = Log::reader_at(path, layout).ok()?;
    let mut raw = Vec::new();

    // The header is the first line of the effective view.
    reader.next_line(&mut raw).ok()??;
    let header = std::str::from_utf8(&raw).ok()?;
    let mut meta = match serde_json::from_str::<Entry>(header.trim()).ok()? {
        Entry::Session(m) => migrate(m, path).ok()?,
        Entry::Message { .. } | Entry::Leaf { .. } | Entry::BranchSummary { .. } => return None,
        Entry::Compaction { .. } => return None,
        Entry::ModelChange { .. } | Entry::ThinkingLevelChange { .. } => return None,
        Entry::TitleChange { .. } => return None,
        Entry::Label { .. } => return None,
        Entry::Custom { .. } | Entry::MessageEdit { .. } => return None,
    };

    // A streaming line count, not a tree walk: it counts every `Message` line in the file, which for a
    // branched session (Track L3, once wired) can exceed the *active* path's length by however many
    // off-branch entries exist. A display convenience, not a correctness input — accurate for every
    // session today, since nothing yet writes an off-branch entry.
    let mut message_count = 0usize;
    let mut preview = None;
    // The most recent stamped `Entry::Message` timestamp seen — preferred over `mtime` below when
    // any message actually carries one (see `Entry::Message::timestamp`'s doc comment). `0` means
    // "no stamped message seen" (an all-legacy file, or one with no message lines at all), in which
    // case `mtime` is the only signal available.
    let mut max_message_timestamp = 0u64;
    loop {
        // Same lenient, skip-just-this-line recovery as `SessionStore::open` (see its comment):
        // `read_capped_line` lets an oversized or invalid-UTF-8 line be skipped without losing the
        // count/preview derived from every good line after it, and only a genuine I/O failure stops
        // the scan early.
        let oversized = match reader.next_line(&mut raw) {
            Ok(None) => break,
            Ok(Some(oversized)) => oversized,
            Err(_) => break,
        };
        if oversized {
            tracing::warn!(path = %path.display(), "skipping oversized session entry line while listing");
            continue;
        }
        let Ok(line) = std::str::from_utf8(&raw) else {
            tracing::warn!(path = %path.display(), "skipping non-UTF-8 session entry line while listing");
            continue;
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match parse_entry_lenient(path, line) {
            Ok(Entry::Message {
                message, timestamp, ..
            }) => {
                message_count += 1;
                max_message_timestamp = max_message_timestamp.max(timestamp);
                if let Some(text) = first_user_text(&message)
                    && preview.is_none()
                {
                    preview = Some(preview_of(text));
                }
            }
            // A stray header mid-file (or a branch-navigation/summary/compaction-provenance/label/custom
            // marker) is ignored — a custom entry contributes no message text of its own, matching
            // `message_count`'s "real conversation messages only" semantics.
            Ok(Entry::Session(_))
            | Ok(Entry::Leaf { .. })
            | Ok(Entry::BranchSummary { .. })
            | Ok(Entry::Compaction { .. })
            | Ok(Entry::Label { .. })
            | Ok(Entry::Custom { .. })
            | Ok(Entry::MessageEdit { .. }) => {}
            // Whole-session-scoped: the most recent one anywhere in the file wins (Task #18, pi-parity
            // fix, for `model`/`thinking_level`) — so a `list_sessions`/`list_all_sessions` listing shows
            // the session's actual last-used model rather than its frozen creation-time one. Safe here
            // specifically because `read_listing`'s own `SessionMeta` is a display-only, throwaway
            // value (never fed into `Persistence::model_and_level_at`'s tree-fallback resolution the way
            // a real `SessionStore::open` would be) — see `SessionStore::record_model_change`'s doc
            // comment for why the *real*, operative `meta.model` (built by `open`, not this listing scan)
            // must NOT be mutated the same way.
            Ok(Entry::ModelChange { model, .. }) => {
                meta.model = model;
            }
            Ok(Entry::ThinkingLevelChange { level, .. }) => {
                meta.thinking_level = Some(level);
            }
            Ok(Entry::TitleChange { title, .. }) => {
                meta.title = title_or_clear(title);
            }
            // A fully-read line that failed to deserialize — skip just this one and keep scanning,
            // same relaxed recovery as `SessionStore::open` (see its comment): we know this line's
            // exact boundaries, so one bad line (anywhere in the file, not only a torn tail) no longer
            // truncates the count/preview derived from every good line after it.
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "skipping unparseable session entry line");
            }
        }
    }

    // Content-derived timestamp preferred over the file's OS mtime; mtime is only a fallback for a
    // legacy file with no stamped message (see `Entry::Message::timestamp`'s doc comment).
    meta.updated_at = if max_message_timestamp > 0 {
        max_message_timestamp
    } else {
        mtime
    };
    meta.message_count = message_count;
    meta.preview = preview;
    Some(meta)
}

/// The first plain-text block of a user message — what a preview shows. Tool-result user turns carry no
/// `Text` block, so they yield `None` and are skipped; assistant turns aren't user input.
pub(super) fn first_user_text(msg: &Message) -> Option<&str> {
    if msg.role != Role::User {
        return None;
    }
    msg.content.iter().find_map(|b| match b {
        ContentBlock::Text { text, .. } => Some(&**text),
        _ => None,
    })
}

/// A one-line, length-bounded preview: the first non-blank line of `text`, trimmed, truncated on a char
/// boundary to [`PREVIEW_MAX`] with an ellipsis when it overran.
pub(super) fn preview_of(text: &str) -> String {
    let line = text
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .trim();
    let mut out: String = line.chars().take(PREVIEW_MAX).collect();
    if line.chars().count() > PREVIEW_MAX {
        out.push('…');
    }
    out
}

/// A preview of any message's own text content (unlike [`first_user_text`], not restricted to the
/// `User` role) — [`SessionStore::tree`]'s per-node preview, one message at a time rather than a
/// branch's first user turn. `None` for a message with no plain-text block (a pure tool-use/tool-result/
/// thinking/image turn).
pub(super) fn message_text_preview(msg: &Message) -> Option<String> {
    msg.content.iter().find_map(|b| match b {
        ContentBlock::Text { text, .. } if !text.trim().is_empty() => Some(preview_of(text)),
        _ => None,
    })
}

/// The file's last-modified time as Unix seconds, read from `fs` metadata (no file content). Falls back
/// to 0 if the platform/clock can't supply it — a listing field, not a correctness input.
pub(super) fn mtime_secs(path: &Path) -> u64 {
    fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
