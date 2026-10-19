use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use serde::{Deserialize, Serialize};

use super::*;

// =====================================================================================================
// Storage layout — one session as a single file, or as a directory of epoch segments
// =====================================================================================================
//
// The single-file layout ([`Layout::File`]) is what every local invocation uses and is the default:
// one `<created_at>_<id>.jsonl` per session, appended to in place. It is unchanged, byte for byte,
// by everything below — `Log::File` is a path and the same `OpenOptions::append` write.
//
// The segmented layout ([`Layout::Segmented`]) exists for a multi-replica service on shared storage
// (EFS), where "this process owns this session" cannot be assumed and a lock is at best a liveness
// hint. A session becomes a **directory of epoch segments**:
//
// ```text
// <dir>/<id>/
//   lock              advisory, liveness only (see `acquire_session_lock`)
//   000001.jsonl      epoch 1 — created with O_EXCL; this is the session's existence fence
//   000002.jsonl      epoch 2 — created by the next owner, sealing epoch 1 at the offset it read
//   ...
//   memory/           this session's `/session` working memory
// ```
//
// Ownership is established by **creating a file nobody else could have created**: each new owner
// O_EXCL-creates the next epoch, and its header records how far it consumed each predecessor. A
// losing writer finds the epoch already taken (`EEXIST`) or its predecessor longer than it read, and
// is [`Superseded`] — permanently, for that store. No lease, no clock, no heartbeat.

/// How many segments may pile up above the last base before the next persist writes a new base.
/// Bounds the growth a reap-and-reattach cycle causes: each attach rolls one segment, so without
/// this a long-lived session would accumulate one file per attach forever.
pub(super) const CONSOLIDATION_K: usize = 8;

/// Closes a base segment's own content. A base counts as complete — and so as a valid starting point
/// for a replay — only once this line is on disk; lines after it are ordinary appends.
pub(super) const BASE_TRAILER: &[u8] = br#"{"type":"base_complete"}"#;

/// Sealed-blob framing: `e1.<kfp8>:` then base64url of `nonce(24) ‖ ciphertext ‖ tag(16)`.
pub(super) const SEAL_FRAME: &str = "e1.";
pub(super) const SEAL_NONCE_LEN: usize = 24;
pub(super) const SEAL_TAG_LEN: usize = 16;

/// How many times a segmented read restarts when the segment set moves under it — a consolidation
/// unlinking an older segment, or an NFS file handle going stale.
pub(super) const SEG_READ_RETRIES: usize = 5;

/// `ESTALE`. Not an `io::ErrorKind` variant, so it is matched on the raw errno.
pub(super) const ESTALE: i32 = 116;

/// One epoch's file name: `000001.jsonl`. Zero-padded so a plain directory listing sorts by epoch.
pub(super) fn segment_name(epoch: u64) -> String {
    format!("{epoch:06}.jsonl")
}

/// The epoch a file name encodes, or `None` for anything that isn't a segment.
pub(super) fn segment_epoch(name: &str) -> Option<u64> {
    let digits = name.strip_suffix(".jsonl")?;
    (digits.len() >= 6 && digits.bytes().all(|b| b.is_ascii_digit()))
        .then(|| digits.parse().ok())
        .flatten()
}

/// Whether an error means "the segment set moved under us" rather than a real failure.
pub(super) fn is_transient_read_error(e: &std::io::Error) -> bool {
    e.kind() == std::io::ErrorKind::NotFound || e.raw_os_error() == Some(ESTALE)
}

// ---- Segment metadata ---------------------------------------------------------------------------

/// A segment's first line: which epoch it is, how far each predecessor was consumed before this one
/// was created, and whether its own content is a full rewrite.
///
/// The seal map carries **every** seal since the last base, not just the immediate predecessor's, so
/// a reader that starts at that base can bound every segment between without walking headers it may
/// no longer be able to open.
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename = "segment")]
pub(super) struct SegmentHeader {
    pub(super) epoch: u64,
    /// `epoch → byte offset`, as decimal-string keys so the map round-trips through JSON.
    pub(super) sealed: BTreeMap<String, u64>,
    pub(super) base: bool,
}

impl SegmentHeader {
    pub(super) fn seals(&self) -> BTreeMap<u64, u64> {
        self.sealed
            .iter()
            .filter_map(|(k, v)| k.parse().ok().map(|k| (k, *v)))
            .collect()
    }

    pub(super) fn line(
        epoch: u64,
        seals: &BTreeMap<u64, u64>,
        base: bool,
    ) -> std::io::Result<Vec<u8>> {
        let header = SegmentHeader {
            epoch,
            sealed: seals.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            base,
        };
        let mut out = serde_json::to_vec(&header).map_err(std::io::Error::other)?;
        out.push(b'\n');
        Ok(out)
    }
}

/// One segment as the directory reports it, before anything is read from it.
pub(super) struct SegmentStat {
    pub(super) epoch: u64,
    pub(super) path: PathBuf,
    pub(super) len: u64,
    pub(super) mtime_ns: u128,
}

/// Every segment in `dir`, oldest first. A directory with none counts as no session at all.
pub(super) fn scan_segments(dir: &Path) -> std::io::Result<Vec<SegmentStat>> {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(epoch) = name.to_str().and_then(segment_epoch) else {
            continue;
        };
        let Ok(meta) = entry.metadata() else { continue };
        if !meta.is_file() {
            continue;
        }
        let mtime_ns = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        out.push(SegmentStat {
            epoch,
            path: dir.join(name),
            len: meta.len(),
            mtime_ns,
        });
    }
    out.sort_by_key(|s| s.epoch);
    Ok(out)
}

/// Whether `dir` holds a session at all. Fast path: `000001.jsonl` is created with O_EXCL before
/// anything else and is never deleted (see [`SegLog::prune`]), so its presence answers the question
/// without a directory read in every ordinary case.
pub(super) fn segment_dir_is_session(dir: &Path) -> bool {
    if dir.join(segment_name(1)).is_file() {
        return true;
    }
    scan_segments(dir).is_ok_and(|s| !s.is_empty())
}

/// Read a segment's header line off an already-open descriptor, leaving the cursor just past it.
/// A torn or unparseable header means the segment counts as empty and sealed at 0 — the crash-mid-
/// create case, where the file exists but says nothing trustworthy about itself.
pub(super) fn read_segment_header(
    file: &mut File,
) -> std::io::Result<Option<(SegmentHeader, u64)>> {
    use std::io::Seek;
    file.seek(std::io::SeekFrom::Start(0))?;
    let mut reader = BufReader::new(&mut *file);
    let mut raw = Vec::new();
    let mut limit = None;
    let Some(read) = read_capped_line_limited(&mut reader, &mut raw, &mut limit)? else {
        return Ok(None);
    };
    if read.oversized || !read.terminated {
        return Ok(None);
    }
    Ok(serde_json::from_slice::<SegmentHeader>(&raw)
        .ok()
        .map(|h| (h, read.consumed)))
}

/// Whether a base segment's trailer is on disk — the only thing that makes it a valid replay start.
/// A raw byte scan of the line, not a parse: nothing here needs to decode content, and a base can be
/// the whole transcript.
pub(super) fn segment_has_trailer(file: &mut File, start: u64, end: u64) -> std::io::Result<bool> {
    use std::io::Seek;
    if end <= start {
        return Ok(false);
    }
    file.seek(std::io::SeekFrom::Start(start))?;
    let mut reader = BufReader::new(file.take(end - start));
    let mut raw = Vec::new();
    let mut limit = None;
    while let Some(read) = read_capped_line_limited(&mut reader, &mut raw, &mut limit)? {
        if read.terminated && !read.oversized && raw == BASE_TRAILER {
            return Ok(true);
        }
    }
    Ok(false)
}
