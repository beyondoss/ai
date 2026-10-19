use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::*;

// ---- The log seam -------------------------------------------------------------------------------

/// Where a [`SessionStore`]'s entries live. Replaces the bare `PathBuf` the store used to hold, so
/// every write goes through one place and the tree logic above never learns which layout it is on.
pub(super) enum Log {
    File {
        path: PathBuf,
        read_only: bool,
        /// The file's length when the open found its last line torn: the first write cuts that
        /// line off (`trim_torn_tail`) if the file is still exactly this long, then clears it.
        torn_at: Option<u64>,
    },
    Segmented(Box<SegLog>),
}

/// The writer state for one segmented session: what this process observed when it opened, and what it
/// has created since.
pub(super) struct SegLog {
    /// `<repo>/<id>`.
    pub(super) dir: PathBuf,
    pub(super) id: String,
    pub(super) codec: Option<Arc<TenantCodec>>,
    pub(super) read_only: bool,
    /// The newest segment at open, and how many of its bytes this process consumed. The pair is the
    /// fence: the first write refuses unless the file is still exactly `consumed` bytes long.
    pub(super) newest: u64,
    pub(super) consumed: u64,
    /// Seals to carry into the next header — every seal since the last base.
    pub(super) seals: BTreeMap<u64, u64>,
    /// The newest complete base. Its predecessors become deletable the moment a newer base lands.
    pub(super) base: Option<u64>,
    /// Segments above the last base. Past [`CONSOLIDATION_K`], the next persist writes a base.
    pub(super) since_base: usize,
    /// The segment this store appends to, once it has created one, and its current length.
    pub(super) target: Option<u64>,
    pub(super) target_len: u64,
    /// Set after an append fails: the target is sealed at its last good offset and the next write
    /// rolls, without re-checking a predecessor this process already owns.
    pub(super) force_roll: bool,
    pub(super) superseded: bool,
}

impl Log {
    /// The path this session is addressed by: the `.jsonl` file, or the session directory.
    pub(super) fn path(&self) -> &Path {
        match self {
            Log::File { path, .. } => path,
            Log::Segmented(s) => &s.dir,
        }
    }

    /// Where this session's MCP task journal key lives (see `crate::mcp_resume::JournalAuth`):
    /// beside a single-file session, inside a segmented session's directory, and nowhere for one
    /// sealed under a tenant codec, whose storage already authenticates every line.
    pub(super) fn journal_key_path(&self) -> Option<PathBuf> {
        match self {
            Log::File { path, .. } => Some(journal_key_beside(path)),
            Log::Segmented(s) if s.codec.is_some() => None,
            Log::Segmented(s) => Some(s.dir.join(crate::mcp_resume::JournalAuth::SIDECAR)),
        }
    }

    /// The name a single-file session's journal key had before it became a suffix:
    /// `with_extension`, which gave `work.1` and `work.2` one key file. Read, and carried over by the
    /// first journal write, never written.
    pub(super) fn legacy_journal_key_path(&self) -> Option<PathBuf> {
        match self {
            Log::File { path, .. } => {
                Some(path.with_extension(crate::mcp_resume::JournalAuth::SIDECAR))
            }
            Log::Segmented(_) => None,
        }
    }

    /// Where this session's `/session` working memory lives.
    ///
    /// For the segmented layout that is `<id>/memory`, *inside* the session directory — not a
    /// sibling derived with `with_extension`, which would turn a dotted id (`<shard>.<opaque>`, what
    /// service mode mints) into `<shard>.memory` and hand every session on a shard the same
    /// directory.
    pub(super) fn memory_dir(&self) -> PathBuf {
        match self {
            Log::File { path, .. } => path.with_extension("memory"),
            Log::Segmented(s) => s.dir.join("memory"),
        }
    }

    pub(super) fn read_only(&self) -> bool {
        match self {
            Log::File { read_only, .. } => *read_only,
            Log::Segmented(s) => s.read_only,
        }
    }

    pub(super) fn superseded(&self) -> bool {
        match self {
            // There is no takeover to lose: a single-file session has no epoch to be outbid for.
            Log::File { .. } => false,
            Log::Segmented(s) => s.superseded,
        }
    }

    /// Create a new session, writing `first_line` (the header entry) as its first content.
    pub(super) fn create(
        path: PathBuf,
        id: &str,
        layout: &Layout,
        first_line: &[u8],
    ) -> std::io::Result<Self> {
        match layout {
            Layout::File => {
                Self::create_file(&path, first_line)?;
                Ok(Log::File {
                    path,
                    read_only: false,
                    torn_at: None,
                })
            }
            Layout::Segmented { codec } => Ok(Log::Segmented(Box::new(SegLog::create(
                path,
                id,
                codec.clone(),
                first_line,
            )?))),
        }
    }

    /// The single-file create: the header written whole ([`create_whole`], mode 0600, the file and
    /// its directory fsynced), a zero-byte file already at `path` (`touch`ed ahead of time) initialized
    /// rather than refused.
    pub(super) fn create_file(path: &Path, first_line: &[u8]) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| at_path("cannot create session directory", parent, e))?;
        }
        create_whole(path, first_line, true)
    }

    /// Open an existing session for reading, and (unless `read_only`) for writing.
    pub(super) fn open(
        path: PathBuf,
        id_hint: Option<&str>,
        layout: &Layout,
        read_only: bool,
    ) -> std::io::Result<Self> {
        // `is_dir` is the layout probe: it lets `main.rs`, `fork_from_path` and export keep handing
        // this a bare path without knowing which shape it names.
        if layout.is_segmented() || path.is_dir() {
            let id = id_hint
                .map(str::to_string)
                .or_else(|| path.file_name()?.to_str().map(str::to_string))
                .unwrap_or_default();
            let seg = SegLog::open(path, id, layout.codec().cloned(), read_only)?;
            Ok(Log::Segmented(Box::new(seg)))
        } else {
            // A missing file must surface as `NotFound` here, exactly as `File::open` used to.
            drop(File::open(&path)?);
            Ok(Log::File {
                path,
                read_only,
                torn_at: None,
            })
        }
    }

    /// The effective view, line by line — the same contract as [`read_capped_line`]: `Ok(None)` at the
    /// end, `Ok(Some(oversized))` per line, the bytes in `buf`.
    pub(super) fn lines(&self) -> std::io::Result<LogReader> {
        match self {
            Log::File { path, .. } => Ok(LogReader::File(BufReader::new(File::open(path)?))),
            Log::Segmented(s) => Ok(LogReader::Seg(s.open_reader()?)),
        }
    }

    /// A reader over a session nobody has opened a [`Log`] for — the listing scan, which only ever
    /// reads and would otherwise pay for a store's worth of setup per session.
    pub(super) fn reader_at(path: &Path, layout: &Layout) -> std::io::Result<LogReader> {
        match layout {
            Layout::File => Ok(LogReader::File(BufReader::new(File::open(path)?))),
            Layout::Segmented { codec } => {
                let id = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or_default()
                    .to_string();
                let seg = SegLog::open(path.to_path_buf(), id, codec.clone(), true)?;
                Ok(LogReader::Seg(seg.open_reader()?))
            }
        }
    }

    /// Adopt the writer state a just-finished read discovered: which segment is newest, how far it
    /// was consumed (the other half of the fence), which seals carry forward, and where the last base
    /// is. Nothing is written before this runs.
    pub(super) fn observe(&mut self, reader: &LogReader) {
        if let (Log::Segmented(s), LogReader::Seg(r)) = (self, reader) {
            s.newest = r.newest;
            s.consumed = r.consumed;
            s.seals = r.seals.clone();
            s.base = r.base;
            s.since_base = r.since_base;
        }
    }

    pub(super) fn guard_writable(&self) -> std::io::Result<()> {
        if self.read_only() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "session was opened read-only",
            ));
        }
        if self.superseded() {
            return Err(superseded_error());
        }
        Ok(())
    }

    /// Append already-serialized entry lines. O(bytes written), never O(transcript).
    pub(super) fn append(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        self.guard_writable()?;
        match self {
            Log::File { path, torn_at, .. } => {
                use std::io::{Seek, SeekFrom};
                // The torn tail the open found, cut now that this store is writing (see `open_with`).
                // Only if the file is still exactly as the open saw it: anything appended since is
                // someone else's, and the newline below keeps this entry clear of it regardless.
                if let Some(len) = torn_at.take()
                    && fs::metadata(&*path)?.len() == len
                {
                    trim_torn_tail(path)?;
                }
                let mut f = OpenOptions::new().read(true).append(true).open(&*path)?;
                keep_private(&f)?;
                let pre = f.metadata()?.len();
                // Never glue an entry onto a line that did not end: a tail left unterminated since
                // the open (a rollback below that itself failed) gets its newline first, so this
                // entry reads back as its own line whatever the fragment before it is.
                let mut last = [b'\n'];
                if pre > 0 {
                    f.seek(SeekFrom::Start(pre - 1))?;
                    f.read_exact(&mut last)?;
                }
                let written = (|| -> std::io::Result<()> {
                    if last[0] != b'\n' {
                        f.write_all(b"\n")?;
                    }
                    f.write_all(bytes)?;
                    // `flush` only pushes past our buffer into the OS; `sync_all` forces the bytes
                    // to disk, which is what the module's crash-safety claim actually requires. The
                    // parent dir is unchanged on an append (same inode, same dentry), so no
                    // directory fsync.
                    f.flush()?;
                    f.sync_all()
                })();
                if written.is_err() {
                    // A failed append (a full disk, a file-size limit, a failed sync) may have left
                    // part of its bytes behind. The caller's state did not advance, so put the file
                    // back exactly as long as it was: the retry, or the next turn, must never land
                    // after half a line.
                    if let Err(e) = f.set_len(pre).and_then(|()| f.sync_all()) {
                        tracing::warn!(path = %path.display(), error = %e, "could not roll back a failed session append");
                    }
                }
                written
            }
            Log::Segmented(s) => s.append(bytes),
        }
    }

    /// Replace the whole log with `bytes`. Atomic: a reader sees the old content or the new one.
    pub(super) fn replace_all(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        self.guard_writable()?;
        match self {
            Log::File { path, torn_at, .. } => {
                // The whole file is replaced, torn tail and all.
                *torn_at = None;
                let tmp = path.with_extension("jsonl.tmp");
                let mut f = create_private(&tmp)?;
                let write = (|| -> std::io::Result<()> {
                    f.write_all(bytes)?;
                    // Sync the temp file's contents, then rename (atomic), then fsync the parent
                    // directory so the rename itself is durable: without the dir fsync a crash could
                    // surface the old file — or, in the window between, neither — even though the new
                    // bytes had reached disk.
                    f.flush()?;
                    f.sync_all()?;
                    fs::rename(&tmp, &*path)?;
                    fsync_dir(path)
                })();
                if write.is_err() {
                    // A genuine in-process error (disk full, a permission error mid-write), not a hard
                    // crash: the process is still alive and can just remove the temp file.
                    let _ = fs::remove_file(&tmp);
                }
                write
            }
            Log::Segmented(s) => {
                // A base's content is transcript, so it is sealed line by line exactly like an
                // append. Only the header and the trailer stay plaintext.
                let (codec, id) = (s.codec.clone(), s.id.clone());
                s.write_base(|f| {
                    let mut out = Vec::with_capacity(bytes.len() + 64);
                    append_sealed(&mut out, codec.as_deref(), &id, bytes)?;
                    f.write_all(&out)
                })
            }
        }
    }
}

impl SegLog {
    /// O_EXCL-create `000001.jsonl` — **the fence**. Two processes racing to create the same id both
    /// reach this; exactly one gets the file, the other gets `AlreadyExists` and never writes a byte.
    ///
    /// Epoch 1 is written as a complete base (header, the session header entry, trailer), so a replay
    /// always has a base to start from and `replace_all` has an append target to roll off.
    pub(super) fn create(
        dir: PathBuf,
        id: &str,
        codec: Option<Arc<TenantCodec>>,
        first_line: &[u8],
    ) -> std::io::Result<Self> {
        // Idempotent, so the lock file (and a retried create) can live alongside the segments.
        fs::create_dir_all(&dir)?;
        if segment_dir_is_session(&dir) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!("session {id} already exists"),
            ));
        }
        let path = dir.join(segment_name(1));
        let mut buf = SegmentHeader::line(1, &BTreeMap::new(), true)?;
        append_sealed(&mut buf, codec.as_deref(), id, first_line)?;
        buf.extend_from_slice(BASE_TRAILER);
        buf.push(b'\n');
        // Still the fence — `link` refuses an existing name exactly as `create_new` did — but the
        // segment appears whole: a crash between creating it and writing it used to leave an empty
        // `000001.jsonl` that made the id neither openable (no header) nor creatable (it exists).
        create_whole(&path, &buf, false)?;
        // The repo directory the new session directory appeared in.
        fsync_dir(&dir)?;
        Ok(Self {
            dir,
            id: id.to_string(),
            codec,
            read_only: false,
            newest: 1,
            consumed: buf.len() as u64,
            seals: BTreeMap::new(),
            base: Some(1),
            since_base: 0,
            target: Some(1),
            target_len: buf.len() as u64,
            force_roll: false,
            superseded: false,
        })
    }

    pub(super) fn open(
        dir: PathBuf,
        id: String,
        codec: Option<Arc<TenantCodec>>,
        read_only: bool,
    ) -> std::io::Result<Self> {
        let segments = scan_segments(&dir)?;
        if segments.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("no session segments in {}", dir.display()),
            ));
        }
        Ok(Self {
            dir,
            id,
            codec,
            read_only,
            // Filled in for real by `Log::observe` once the caller's replay says how far it got; these
            // are the pre-read defaults, and nothing writes before a read.
            newest: segments.last().map(|s| s.epoch).unwrap_or(1),
            consumed: segments.last().map(|s| s.len).unwrap_or(0),
            seals: BTreeMap::new(),
            base: None,
            since_base: segments.len(),
            target: None,
            target_len: 0,
            force_roll: false,
            superseded: false,
        })
    }

    pub(super) fn poison(&mut self) -> std::io::Error {
        self.superseded = true;
        superseded_error()
    }

    /// Build the effective view: open every segment up front, fold the seals, find the last complete
    /// base, and hand back a reader bounded to each segment's sealed range.
    pub(super) fn open_reader(&self) -> std::io::Result<SegReader> {
        let mut last: Option<std::io::Error> = None;
        for _ in 0..SEG_READ_RETRIES {
            match self.try_open_reader() {
                Ok(r) => return Ok(r),
                // The segment set moved under us — a consolidation unlinked an older segment, or an
                // NFS handle went stale. Rescan and read it again from scratch.
                Err(e) if is_transient_read_error(&e) => last = Some(e),
                Err(e) => return Err(e),
            }
        }
        Err(last.unwrap_or_else(|| {
            std::io::Error::other(format!(
                "session {} kept changing while being read",
                self.dir.display()
            ))
        }))
    }

    pub(super) fn try_open_reader(&self) -> std::io::Result<SegReader> {
        let stats = scan_segments(&self.dir)?;
        if stats.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("no session segments in {}", self.dir.display()),
            ));
        }
        // Every descriptor up front: once open, an unlink underneath us is harmless on POSIX.
        struct Opened {
            epoch: u64,
            file: File,
            len: u64,
            header: Option<SegmentHeader>,
            header_len: u64,
        }
        let mut opened = Vec::with_capacity(stats.len());
        for s in &stats {
            let mut file = File::open(&s.path)?;
            let (header, header_len) = match read_segment_header(&mut file)? {
                Some((h, n)) => (Some(h), n),
                // Torn or missing header: the segment counts as empty, sealed at 0.
                None => (None, s.len),
            };
            opened.push(Opened {
                epoch: s.epoch,
                file,
                len: s.len,
                header,
                header_len,
            });
        }

        // Every seal any header declares; a newer header wins, since it saw more.
        let mut seals: BTreeMap<u64, u64> = BTreeMap::new();
        for o in &opened {
            if let Some(h) = &o.header {
                seals.extend(h.seals());
            }
        }

        // The start: the newest base whose trailer actually landed, else the oldest segment present.
        let mut start = opened.first().map(|o| o.epoch).unwrap_or(1);
        let mut base = None;
        for o in opened.iter_mut().rev() {
            let Some(h) = &o.header else { continue };
            if !h.base {
                continue;
            }
            let end = seals.get(&o.epoch).copied().unwrap_or(o.len).min(o.len);
            if !segment_has_trailer(&mut o.file, o.header_len, end)? {
                continue;
            }
            base = Some(o.epoch);
            start = o.epoch;
            break;
        }

        let newest = opened.last().map(|o| o.epoch).unwrap_or(1);
        let newest_len = opened.last().map(|o| o.len).unwrap_or(0);
        let since_base = opened.iter().filter(|o| o.epoch > start).count();
        // What the next header carries forward: every seal recorded above the starting base. Seals
        // for segments the view no longer reaches are dropped — that is what keeps the header bounded.
        let carry: BTreeMap<u64, u64> = opened
            .iter()
            .filter(|o| o.epoch > start)
            .filter_map(|o| o.header.as_ref())
            .flat_map(|h| h.seals())
            .collect();

        let mut segs = Vec::with_capacity(opened.len());
        for o in opened {
            if o.epoch < start {
                continue;
            }
            // A base newer than the start is one whose trailer never landed — a crash mid-rewrite.
            // Its content is a partial replacement of the whole transcript and must never be replayed;
            // its header's seals still count, and were folded in above.
            let is_torn_base = o.header.as_ref().is_some_and(|h| h.base) && o.epoch != start;
            if o.header.is_none() || is_torn_base {
                continue;
            }
            let end = seals.get(&o.epoch).copied().unwrap_or(o.len).min(o.len);
            if end <= o.header_len {
                continue;
            }
            let mut file = o.file;
            use std::io::Seek;
            file.seek(std::io::SeekFrom::Start(o.header_len))?;
            segs.push(SegOpen {
                reader: BufReader::new(file),
                remaining: Some(end - o.header_len),
            });
        }

        Ok(SegReader {
            segs,
            idx: 0,
            codec: self.codec.clone(),
            id: self.id.clone(),
            newest,
            consumed: newest_len,
            seals: carry,
            base,
            since_base,
        })
    }

    /// The segment this store appends to, creating it if this is its first write.
    ///
    /// Creating it is the takeover: the predecessor must still be exactly as long as this process
    /// read it, and the new epoch must not already exist. Either check failing means another owner
    /// got there first.
    pub(super) fn ensure_target(&mut self) -> std::io::Result<PathBuf> {
        if let Some(epoch) = self.target {
            return Ok(self.dir.join(segment_name(epoch)));
        }
        let predecessor = self.dir.join(segment_name(self.newest));
        if !self.force_roll {
            let len = fs::metadata(&predecessor).map(|m| m.len()).unwrap_or(0);
            if len != self.consumed {
                return Err(self.poison());
            }
        }
        let mut seals = std::mem::take(&mut self.seals);
        seals.insert(self.newest, self.consumed);
        let epoch = self.newest + 1;
        let path = self.dir.join(segment_name(epoch));
        let mut opts = OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let mut f = match opts.open(&path) {
            Ok(f) => f,
            // Someone else already claimed this epoch. That is the whole fence: they are the owner
            // now, and this store never writes again.
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                self.seals = seals;
                return Err(self.poison());
            }
            Err(e) => {
                self.seals = seals;
                return Err(e);
            }
        };
        let header = SegmentHeader::line(epoch, &seals, false)?;
        let written = (|| -> std::io::Result<()> {
            f.write_all(&header)?;
            f.flush()?;
            f.sync_all()?;
            fsync_dir(&path)
        })();
        self.seals = seals;
        if let Err(e) = written {
            // The epoch is ours but its header never landed, so a reader counts it as empty. Step
            // over it on the next attempt rather than retrying a name that now exists (which would
            // read as a takeover and poison a store that is in fact still the rightful owner).
            self.newest = epoch;
            self.consumed = 0;
            self.force_roll = true;
            self.since_base += 1;
            return Err(e);
        }
        self.target = Some(epoch);
        self.target_len = header.len() as u64;
        self.since_base += 1;
        self.force_roll = false;
        Ok(path)
    }

    pub(super) fn consolidation_due(&self) -> bool {
        self.since_base > CONSOLIDATION_K
    }

    pub(super) fn append(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        // Past the segment budget, the next persist writes a base instead of rolling yet another
        // segment — one base replaces the whole chain and resets the count.
        if self.consolidation_due() {
            self.consolidate()?;
        }
        let path = self.ensure_target()?;
        let mut buf = Vec::with_capacity(bytes.len() + 64);
        append_sealed(&mut buf, self.codec.as_deref(), &self.id, bytes)?;
        let before = self.target_len;
        let write = (|| -> std::io::Result<()> {
            let mut f = OpenOptions::new().append(true).open(&path)?;
            keep_private(&f)?;
            f.write_all(&buf)?;
            f.flush()?;
            f.sync_all()
        })();
        match write {
            Ok(()) => {
                self.target_len += buf.len() as u64;
                // Before the caller may acknowledge this batch: has anyone claimed the epoch after
                // ours while we were writing?
                self.guard_not_superseded()?;
                Ok(())
            }
            Err(e) => {
                // Seal at the last offset known good and roll on the next write, so one ENOSPC can't
                // brick the session with a torn line in the middle of a segment nobody can bound.
                if let Some(epoch) = self.target.take() {
                    self.seals.insert(epoch, before);
                    self.newest = epoch;
                    self.consumed = before;
                    self.force_roll = true;
                }
                Err(e)
            }
        }
    }

    /// Has another owner claimed the epoch after this one?
    ///
    /// The gap this closes. [`Self::ensure_target`] checks for a rival *when it rolls* — the
    /// predecessor's length, then an `O_EXCL` create — and that is a complete fence for an owner
    /// that is starting a segment. But an owner that already **has** one open appends straight to
    /// it and re-checks nothing, so an owner fenced while it holds a target never finds out. Its
    /// appends keep succeeding, because they are ordinary appends to a file it still has permission
    /// to write; the successor sealed that segment at the offset it read, so every reader stops
    /// there and ignores the rest. The owner goes on acknowledging turns that no reader will ever
    /// replay, which is the "no acknowledged line is lost" guarantee failing in the one situation
    /// the whole segmented layout exists for.
    ///
    /// Measured before this existed, with a partitioned owner and a real NFSv4 lease: the successor
    /// took the session over after 102 s, and the old owner then committed a turn, told its client
    /// `success`, and that turn was absent from the transcript the successor replayed.
    ///
    /// One `stat` per committed batch — not per line: [`SessionStore::append_new`] buffers a whole
    /// turn and appends it once, so this is paid once per turn, next to an `fsync` that costs far
    /// more.
    ///
    /// **What it does not close.** A successor that reads this segment's length and creates its own
    /// epoch *after* this check has passed still seals away a line this owner has acknowledged. The
    /// window shrinks from unbounded to one round trip. On a network filesystem the `exists()` below
    /// can also be answered from the client's cache: a negative lookup for the successor's name (a
    /// cached negative dentry) is trusted until the directory's cached attributes expire, which
    /// takes up to `acdirmax` (with the default `lookupcache=all`). That is safe only while
    /// `acdirmax` is shorter than the NFSv4 lease a successor must outwait before it can take the
    /// session from a partitioned owner — service mode refuses to start on a mount where it is not,
    /// or with `nocto` ([`crate::shard_mount`]). Closing the round-trip window completely needs a
    /// fencing token carried in the write itself, which the storage format has no room for today.
    pub(super) fn guard_not_superseded(&mut self) -> std::io::Result<()> {
        let Some(epoch) = self.target else {
            return Ok(());
        };
        if self.dir.join(segment_name(epoch + 1)).exists() {
            return Err(self.poison());
        }
        Ok(())
    }

    /// Write a new base segment, whose content `fill` produces, and retire what it replaces.
    pub(super) fn write_base(
        &mut self,
        fill: impl FnOnce(&mut File) -> std::io::Result<()>,
    ) -> std::io::Result<()> {
        // A base supersedes whatever this process was appending to, so seal that too.
        let (pred, pred_len) = match self.target {
            Some(epoch) => (epoch, self.target_len),
            None => {
                if !self.force_roll {
                    let len = fs::metadata(self.dir.join(segment_name(self.newest)))
                        .map(|m| m.len())
                        .unwrap_or(0);
                    if len != self.consumed {
                        return Err(self.poison());
                    }
                }
                (self.newest, self.consumed)
            }
        };
        let mut seals = std::mem::take(&mut self.seals);
        seals.insert(pred, pred_len);
        let epoch = pred + 1;
        let path = self.dir.join(segment_name(epoch));
        let mut opts = OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let mut f = match opts.open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                self.seals = seals;
                return Err(self.poison());
            }
            Err(e) => {
                self.seals = seals;
                return Err(e);
            }
        };
        let written = (|| -> std::io::Result<u64> {
            let header = SegmentHeader::line(epoch, &seals, true)?;
            f.write_all(&header)?;
            fill(&mut f)?;
            // The trailer is what makes this base count. Until it is on disk a reader ignores this
            // segment's content entirely and replays from the previous base.
            f.write_all(BASE_TRAILER)?;
            f.write_all(b"\n")?;
            f.flush()?;
            f.sync_all()?;
            f.metadata().map(|m| m.len())
        })();
        let len = match written {
            Ok(len) => len,
            Err(e) => {
                // Nothing committed: the trailer never landed, so a reader ignores this segment's
                // content entirely and replays from the previous base. It is ours, though, so the
                // next attempt must step *over* it rather than retry the same name — retrying would
                // hit `EEXIST` against our own debris and read as a takeover.
                self.seals = seals;
                self.target = None;
                self.newest = epoch;
                self.consumed = 0;
                self.since_base += 1;
                self.force_roll = true;
                return Err(e);
            }
        };
        fsync_dir(&path)?;
        // Everything below the base this one supersedes. That base itself survives, so a reader that
        // already started there keeps a complete view.
        self.prune(self.base);
        self.base = Some(epoch);
        self.seals = BTreeMap::new();
        self.target = Some(epoch);
        self.target_len = len;
        self.newest = epoch;
        self.consumed = len;
        self.since_base = 0;
        self.force_roll = false;
        Ok(())
    }

    /// Rewrite the effective view into a fresh base, then retire the segments it replaced.
    ///
    /// The content is a **byte copy of the current view**, not a re-serialization of the in-memory
    /// tree: a base must reproduce the log exactly, including the model/title/label/leaf records that
    /// live only on disk, and copying is the only way to be sure it does.
    pub(super) fn consolidate(&mut self) -> std::io::Result<()> {
        let mut reader = self.open_reader()?;
        let codec = self.codec.clone();
        let id = self.id.clone();
        self.write_base(move |f| {
            let mut line = Vec::new();
            let mut out = Vec::new();
            loop {
                match reader.next_line(&mut line)? {
                    None => break,
                    Some(true) => continue,
                    Some(false) => {}
                }
                out.clear();
                append_sealed(&mut out, codec.as_deref(), &id, &line)?;
                f.write_all(&out)?;
            }
            Ok(())
        })
    }

    /// Delete segments older than `floor` — the base a new base just superseded. Nothing a live
    /// reader could still be starting from is ever removed, and epoch 1 stays forever: it is the
    /// O_EXCL fence that says this session exists at all.
    pub(super) fn prune(&self, floor: Option<u64>) {
        let Some(floor) = floor else { return };
        let Ok(segments) = scan_segments(&self.dir) else {
            return;
        };
        for s in segments {
            if s.epoch > 1 && s.epoch < floor {
                let _ = fs::remove_file(&s.path);
            }
        }
    }
}

/// Serialize `bytes` (one or more `\n`-terminated entry lines) into `out`, sealing each line when a
/// codec is present. Without one this is a single `extend_from_slice` — the unsealed path costs
/// nothing.
pub(super) fn append_sealed(
    out: &mut Vec<u8>,
    codec: Option<&TenantCodec>,
    id: &str,
    bytes: &[u8],
) -> std::io::Result<()> {
    let Some(codec) = codec else {
        out.extend_from_slice(bytes);
        if !bytes.is_empty() && !bytes.ends_with(b"\n") {
            out.push(b'\n');
        }
        return Ok(());
    };
    for line in bytes.split_inclusive(|b| *b == b'\n') {
        let body = line.strip_suffix(b"\n").unwrap_or(line);
        if body.is_empty() {
            continue;
        }
        out.extend_from_slice(&codec.seal_line(id, body)?);
        out.push(b'\n');
    }
    Ok(())
}
