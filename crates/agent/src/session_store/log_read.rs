use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;
use std::time::UNIX_EPOCH;

use super::*;

// ---- Reading --------------------------------------------------------------------------------

/// One open segment, bounded to the range the effective view reads from it.
pub(super) struct SegOpen {
    pub(super) reader: BufReader<File>,
    /// Bytes left in this segment's range.
    pub(super) remaining: Option<u64>,
}

/// The effective view of a segmented session, as a line source.
pub(super) struct SegReader {
    pub(super) segs: Vec<SegOpen>,
    pub(super) idx: usize,
    pub(super) codec: Option<Arc<TenantCodec>>,
    pub(super) id: String,
    pub(super) newest: u64,
    /// How far the newest segment was consumed — the offset the next header seals it at. It is the
    /// newest segment's length as the scan observed it, which is also exactly where the bounded read
    /// stops: bytes a stale writer appends after that point are outside the range and never read, and
    /// sealing there is what makes them unreachable on every later replay too.
    pub(super) consumed: u64,
    pub(super) seals: BTreeMap<u64, u64>,
    pub(super) base: Option<u64>,
    pub(super) since_base: usize,
}

/// A line source over either layout. `next_line` keeps [`read_capped_line`]'s contract exactly, so the
/// replay loops above are identical for both.
pub(super) enum LogReader {
    File(BufReader<File>),
    Seg(SegReader),
}

impl LogReader {
    pub(super) fn next_line(&mut self, buf: &mut Vec<u8>) -> std::io::Result<Option<bool>> {
        match self {
            LogReader::File(r) => read_capped_line(r, buf),
            LogReader::Seg(r) => r.next_line(buf),
        }
    }
}

impl SegReader {
    pub(super) fn next_line(&mut self, buf: &mut Vec<u8>) -> std::io::Result<Option<bool>> {
        loop {
            let Some(cur) = self.segs.get_mut(self.idx) else {
                buf.clear();
                return Ok(None);
            };
            // Read straight into the caller's buffer: a plaintext line — the whole unsealed layout,
            // and every framing line — is then already where it needs to be.
            let Some(read) = read_capped_line_limited(&mut cur.reader, buf, &mut cur.remaining)?
            else {
                self.idx += 1;
                continue;
            };
            if !read.terminated {
                // A torn final line — a crash mid-append, or a stale writer's bytes past the seal.
                // Tolerated only here, at the very end of a segment's range; anywhere else the range
                // would have ended on a newline.
                self.idx += 1;
                continue;
            }
            if read.oversized {
                buf.clear();
                return Ok(Some(true));
            }
            // Framing belongs to the log, not the session.
            if buf.as_slice() == BASE_TRAILER {
                continue;
            }
            // A plaintext line starts with `{`; a sealed one never does. That is the whole test — no
            // per-store mode to get wrong, and a store written before sealing still reads.
            let Some(codec) = self.codec.as_ref().filter(|_| buf.first() != Some(&b'{')) else {
                return Ok(Some(false));
            };
            *buf = codec.open_line(&self.id, buf)?;
            return Ok(Some(false));
        }
    }
}

// ---- Listing stamps --------------------------------------------------------------------------

/// A session's cache-validity stamp. For the single-file layout that is the file's `(size, mtime)`;
/// for the segmented layout it is the **newest segment's** `(len, mtime, epoch)` plus how many
/// segments there are — so a roll, an append, a base or a prune all invalidate a cached listing.
///
/// Every field is a `readdir` away. That is the point: this is the validator [`scan_listings_in`]
/// runs against *every* candidate path on a cache hit as well as a miss, so its cost is the floor on
/// listing a shard. It used to `File::open` and header-parse **every segment of every session** just
/// to learn which one was the base, making a 100%-warm cache pay O(shards × sessions × segments)
/// opens for an answer it then threw away.
///
/// Dropping `base_epoch` is safe because a base is never written in place: [`SegLog::write_base`]
/// creates `segment_name(pred + 1)` with `create_new`, so a new base *always* advances the newest
/// epoch (which this stamp carries), and the `prune` that retires what it superseded changes the
/// segment count (which it also carries). There is no rewrite that leaves both untouched, so there
/// is nothing left for a base epoch to catch.
///
/// The `size` narrowing is deliberate too. Total bytes could only detect a change in a segment that
/// is *not* the newest, and an older segment only ever changes by a stale writer appending past its
/// seal — bytes every reader ignores. So the listing those bytes would invalidate is still correct.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(super) struct Stamp {
    pub(super) size: u64,
    pub(super) mtime_ns: u128,
    pub(super) epoch: u64,
    pub(super) segments: u64,
}

pub(super) fn log_stamp(path: &Path, segmented: bool) -> Option<Stamp> {
    if !segmented {
        let m = fs::metadata(path).ok()?;
        let mtime = m.modified().ok()?.duration_since(UNIX_EPOCH).ok()?;
        return Some(Stamp {
            size: m.len(),
            mtime_ns: mtime.as_nanos(),
            epoch: 0,
            segments: 0,
        });
    }
    let segments = scan_segments(path).ok()?;
    // Sorted by epoch, so the last is the newest — the only one being appended to.
    let newest = segments.last()?;
    Some(Stamp {
        size: newest.len,
        mtime_ns: newest.mtime_ns,
        epoch: newest.epoch,
        segments: segments.len() as u64,
    })
}
