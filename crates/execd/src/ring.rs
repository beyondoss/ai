//! The job output ring: the last bytes of a job's output, by offset from its first byte, kept as
//! chunk files (see [`crate::spool`]) plus an unfinished chunk in memory, one stderr bit per byte.
//!
//! Every decision (which offsets are held, where a read starts and how its pieces tile the window,
//! which chunks retention drops, what a refused chunk costs, which byte ranges are stderr) is the
//! pure kernel [`execd_ring`], proven in Verus (R1..R6, see its docs); this file keeps the I/O:
//! files, the copy of bytes, and the bits' storage. A differential test holds it to a plain
//! `Vec<(u8, bool)>` over random pushes, reads and refused chunks.

use std::path::PathBuf;
use std::sync::atomic::Ordering;

use execd_ring as kernel;

use crate::frame::FRAME_MAX;
use crate::spool;

pub(crate) use kernel::RANGES_MAX;

// The kernel's chunk is execd's frame, and its bits are one per byte.
const _: () = assert!(kernel::CHUNK == FRAME_MAX && kernel::CHUNK == 8 * kernel::CHUNK_BITS);

/// The last bytes of a job's output, by offset from its first byte, kept as chunk files (see
/// [`crate::spool`]). A chunk is the 64 KiB at a multiple of 64 KiB; whole chunks are evicted from
/// the front, so the retention is a floor. In memory: a path and the unfinished chunk.
pub(crate) struct Ring {
    /// This job's directory, created when the first chunk is written.
    dir: PathBuf,
    made: bool,
    /// Full chunks on disk.
    files: usize,
    /// The bytes after them, fewer than a chunk.
    tail: Vec<u8>,
    /// The offset of the first byte held (a multiple of a chunk).
    pub(crate) base: u64,
    retain: usize,
    /// One bit per byte of `tail`, set for a byte that came from stderr. Allocated at the first
    /// stderr byte, so a job that never writes stderr pays nothing; may be shorter than `tail`
    /// (a missing bit is stdout). A full chunk's bits go to `<chunk>.e` when any is set.
    tail_e: Option<Vec<u8>>,
    /// The directory is not this ring's to delete: its owner (a job's spool) removes it.
    keep: bool,
    /// Keeps open the descriptor `dir` names, when it is a path through one (a job's spool).
    _hold: Option<std::sync::Arc<std::os::fd::OwnedFd>>,
}

impl Ring {
    pub(crate) fn new(retain: usize, dir: PathBuf) -> Self {
        Self {
            dir,
            made: false,
            files: 0,
            tail: Vec::new(),
            base: 0,
            retain,
            tail_e: None,
            keep: false,
            _hold: None,
        }
    }

    /// This ring's `dir` is a path through `fd`: it is kept open as long as the ring lives.
    pub fn holding(mut self, fd: std::sync::Arc<std::os::fd::OwnedFd>) -> Self {
        self._hold = Some(fd);
        self
    }

    /// The ring's directory outlives this value (a job's spool directory, which a restarted execd
    /// reads again, and which its owner removes with the job).
    pub fn kept(mut self) -> Self {
        self.keep = true;
        self
    }

    /// A ring over the chunk files a previous execd left in `dir`: the longest run of whole chunks
    /// that ends at the highest one. Anything else there (a chunk the crash cut short, one cut off
    /// from the run by a refused write) is deleted. Its unfinished tail is not here: the journal
    /// has it.
    pub fn recover(retain: usize, dir: PathBuf, floor: u64) -> Self {
        let mut have = std::collections::BTreeSet::new();
        let mut junk = Vec::new();
        for e in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            let name = e.file_name();
            let Some(name) = name.to_str() else { continue };
            let stem = name.strip_suffix(".e").unwrap_or(name);
            let Ok(idx) = u64::from_str_radix(stem, 16) else {
                junk.push(e.path());
                continue;
            };
            let whole = e
                .metadata()
                .is_ok_and(|m| m.is_file() && (stem != name || m.len() == FRAME_MAX as u64));
            if stem == name {
                if whole {
                    have.insert(idx);
                } else {
                    junk.push(e.path());
                }
            }
        }
        let mut ring = Self::new(retain, dir.clone()).kept();
        if let Some(&top) = have.last() {
            let mut low = top;
            while low > 0 && have.contains(&(low - 1)) {
                low -= 1;
            }
            ring.base = low * FRAME_MAX as u64;
            ring.files = (top - low + 1) as usize;
            ring.made = true;
            junk.extend(have.iter().filter(|&&i| i < low).flat_map(|&i| {
                [
                    dir.join(format!("{i:010x}")),
                    dir.join(format!("{i:010x}.e")),
                ]
            }));
        }
        // The chunks end before the journal's first byte: a chunk the disk refused, so the offsets
        // between are a gap, as they were before the crash. The ring starts at the journal.
        if ring.end() < floor && floor.is_multiple_of(FRAME_MAX as u64) {
            junk.extend((0..ring.files).flat_map(|i| {
                let at = ring.base / FRAME_MAX as u64 + i as u64;
                [
                    dir.join(format!("{at:010x}")),
                    dir.join(format!("{at:010x}.e")),
                ]
            }));
            ring.base = floor;
            ring.files = 0;
        }
        for j in junk {
            let _ = std::fs::remove_file(j);
        }
        // Sidecars of chunks that are not in the run.
        if let Ok(rd) = std::fs::read_dir(&dir) {
            for e in rd.flatten() {
                let name = e.file_name();
                let Some(stem) = name.to_str().and_then(|n| n.strip_suffix(".e")) else {
                    continue;
                };
                let held = u64::from_str_radix(stem, 16).is_ok_and(|i| {
                    i * FRAME_MAX as u64 >= ring.base
                        && i * (FRAME_MAX as u64) < ring.base + (ring.files * FRAME_MAX) as u64
                });
                if !held {
                    let _ = std::fs::remove_file(e.path());
                }
            }
        }
        ring
    }

    /// The bytes of the unfinished chunk.
    pub fn tail_len(&self) -> usize {
        self.tail.len()
    }

    fn held(&self) -> usize {
        self.files * FRAME_MAX + self.tail.len()
    }

    /// The offset of the next byte.
    pub fn end(&self) -> u64 {
        self.base + self.held() as u64
    }

    fn chunk_path(&self, offset: u64) -> PathBuf {
        self.dir.join(format!("{:010x}", offset / FRAME_MAX as u64))
    }

    /// The bits of the chunk that starts at `at`: the file beside it, if it has stderr.
    fn chunk_bits_path(&self, at: u64) -> PathBuf {
        self.dir.join(format!("{:010x}.e", at / FRAME_MAX as u64))
    }

    /// Append `data`, which came from stderr when `stderr`.
    pub fn push(&mut self, data: &[u8], stderr: bool) {
        if stderr && !data.is_empty() {
            let at = self.tail.len();
            kernel::set_bits(
                self.tail_e.get_or_insert_with(Vec::new),
                at,
                at + data.len(),
            );
        }
        self.tail.extend_from_slice(data);
        while self.tail.len() >= FRAME_MAX {
            let at = self.base + (self.files * FRAME_MAX) as u64;
            // This chunk's bits, taken off the front of the tail's.
            let bits: Option<Vec<u8>> = match self.tail_e.take() {
                Some(e) => {
                    let (chunk, rest) = kernel::take_chunk_bits(e);
                    self.tail_e = Some(rest);
                    chunk.iter().any(|&b| b != 0).then_some(chunk)
                }
                None => None,
            };
            let made = self.made || spool::make_job_dir(&self.dir).is_ok();
            let wrote = made
                && spool::write_chunk(&self.chunk_path(at), &self.tail[..FRAME_MAX]).is_ok()
                && bits.as_ref().is_none_or(|b| {
                    let ok = spool::write_chunk(&self.chunk_bits_path(at), b).is_ok();
                    if !ok {
                        // Bytes without their labels would read back as stdout: a gap instead.
                        let _ = std::fs::remove_file(self.chunk_path(at));
                    }
                    ok
                });
            self.made = made;
            self.tail.drain(..FRAME_MAX);
            if !wrote {
                // The disk refused it: what was held and this chunk read back as a gap.
                self.forget_files();
            }
            (self.base, self.files) = kernel::after_flush(self.base, self.files, wrote);
        }
        for _ in 0..kernel::evict(self.files, self.tail.len(), self.retain) {
            let _ = std::fs::remove_file(self.chunk_path(self.base));
            let _ = std::fs::remove_file(self.chunk_bits_path(self.base));
            self.base += FRAME_MAX as u64;
            self.files -= 1;
        }
    }

    /// Delete every chunk file held (the caller moves `base` past them).
    fn forget_files(&self) {
        for i in 0..self.files {
            let at = self.base + (i * FRAME_MAX) as u64;
            let _ = std::fs::remove_file(self.chunk_path(at));
            let _ = std::fs::remove_file(self.chunk_bits_path(at));
        }
    }

    /// Up to `max` bytes from `offset` (or from the oldest byte held, when that is later); where
    /// they start, and the absolute `(from, len)` ranges of them that came from stderr: ascending,
    /// disjoint, at most [`RANGES_MAX`] (the window is cut before the range that would exceed it).
    pub fn read(&self, offset: u64, max: usize) -> (u64, Vec<u8>, Vec<(u64, u64)>) {
        let (from, want) = kernel::window(offset, self.base, self.end(), max);
        let mut out = vec![0u8; want];
        let mut ranges: Vec<(u64, u64)> = Vec::new();
        let mut done = 0;
        while done < want {
            let at = from + done as u64;
            let seg = kernel::seg(at, want - done, self.base, self.files, self.tail.len());
            let (n, bits): (usize, Option<std::borrow::Cow<[u8]>>) = if seg.disk {
                let dst = &mut out[done..done + seg.n];
                match spool::read_chunk(&self.chunk_path(at), seg.pos as u64, dst) {
                    Ok(k) if k > 0 => {
                        let bits = std::fs::read(self.chunk_bits_path(at)).ok();
                        (k, bits.map(std::borrow::Cow::Owned))
                    }
                    _ => break,
                }
            } else {
                out[done..done + seg.n].copy_from_slice(&self.tail[seg.pos..seg.pos + seg.n]);
                (
                    seg.n,
                    self.tail_e.as_deref().map(std::borrow::Cow::Borrowed),
                )
            };
            if let Some(bits) = bits {
                // `bits` is indexed from the chunk's (or the tail's) first byte, as `seg.pos` is.
                let kept = kernel::scan(&bits, seg.pos, n, at, &mut ranges);
                if kept < n {
                    done += kept;
                    break;
                }
            }
            done += n;
        }
        out.truncate(done);
        (from, out, ranges)
    }
}

impl Drop for Ring {
    fn drop(&mut self) {
        if self.made && !self.keep {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }
}

/// Fuzz seam: drive a [`Ring`] with the pushes and reads `bytes` describe, and hold it to a plain
/// `Vec` of everything pushed: every read returns the bytes at its offsets, from the oldest kept
/// byte when the offset is older, and the ring keeps at least its retention.
pub(crate) fn fuzz_ring(bytes: &[u8]) {
    let mut it = bytes.iter().copied();
    let retain = 1 + usize::from(it.next().unwrap_or(0)) * 1024;
    // The ring spools to files: give this run a directory of its own, removed when it ends.
    static RUNS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "execd-fuzz-ring-{}-{}",
        std::process::id(),
        RUNS.fetch_add(1, Ordering::Relaxed)
    ));
    let _cleanup = scopeguard_remove(dir.clone());
    let _ = std::fs::create_dir_all(&dir);
    let mut ring = Ring::new(retain, dir.join("ring"));
    let mut all: Vec<(u8, bool)> = Vec::new();
    while let Some(op) = it.next() {
        let a = u16::from_le_bytes([it.next().unwrap_or(0), it.next().unwrap_or(0)]);
        if op % 2 == 0 {
            let seed = op;
            let data: Vec<u8> = (0..usize::from(a))
                .map(|i| (i % 251) as u8 ^ seed)
                .collect();
            ring.push(&data, op % 4 == 0);
            all.extend(data.iter().map(|&b| (b, op % 4 == 0)));
            continue;
        }
        let end = ring.end();
        assert_eq!(end as usize, all.len());
        // Offsets: near zero, around the end, and absurd. Sizes: small, huge and unbounded.
        let offset = match op % 8 {
            1 => u64::from(a),
            3 => end.saturating_sub(u64::from(a)),
            5 => end + u64::from(a),
            _ => u64::MAX - u64::from(a),
        };
        let max = match (op / 8) % 4 {
            0 => usize::from(a),
            1 => usize::MAX,
            2 => usize::MAX - usize::from(a),
            _ => 1 << 20,
        };
        let (from, data, ranges) = ring.read(offset, max);
        assert_eq!(from, offset.max(ring.base));
        let avail = end.saturating_sub(from).min(max as u64);
        // A read with too many stderr runs is cut short, never longer.
        assert!(data.len() as u64 <= avail);
        assert!(ranges.len() <= RANGES_MAX);
        if ranges.len() < RANGES_MAX {
            assert_eq!(data.len() as u64, avail);
        }
        for (i, b) in data.iter().enumerate() {
            let (want, e) = all[from as usize + i];
            assert_eq!(*b, want);
            let off = from + i as u64;
            let in_range = ranges.iter().any(|&(a, n)| off >= a && off < a + n);
            assert_eq!(in_range, e, "stderr flag at {off}");
        }
        let mut prev_end = from;
        for &(a, n) in &ranges {
            assert!(n > 0 && a >= prev_end && a + n <= from + data.len() as u64);
            prev_end = a + n;
        }
        let kept = end - ring.base;
        assert!(kept >= (all.len().min(retain)) as u64);
        assert!(kept < (retain + FRAME_MAX) as u64);
    }
}

/// Removes a directory tree when dropped.
struct RemoveOnDrop(PathBuf);

impl Drop for RemoveOnDrop {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn scopeguard_remove(p: PathBuf) -> RemoveOnDrop {
    RemoveOnDrop(p)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn ring(retain: usize) -> (Ring, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let r = Ring::new(retain, dir.path().join("job"));
        (r, dir)
    }

    /// Write `n` bytes whose value at stream offset `i` is `i % 251`.
    fn fill(ring: &mut Ring, n: usize) {
        let mut at = 0usize;
        while at < n {
            let k = (n - at).min(10_000);
            let piece: Vec<u8> = (at..at + k).map(|i| (i % 251) as u8).collect();
            ring.push(&piece, false);
            at += k;
        }
    }

    #[test]
    fn it_keeps_at_least_the_retention_and_never_more_than_a_chunk_over() {
        let (mut ring, _d) = ring(200_000);
        fill(&mut ring, 1_000_000);
        assert_eq!(ring.end(), 1_000_000);
        let kept = ring.end() - ring.base;
        assert!(
            (200_000..200_000 + FRAME_MAX as u64).contains(&kept),
            "kept {kept}"
        );
        assert_eq!(ring.held() as u64, kept);
        // The oldest bytes are gone: a read from 0 starts where the ring does.
        let (from, data, _) = ring.read(0, 5);
        assert_eq!(from, ring.base);
        assert_eq!(data.len(), 5);
        // Evicted chunks are deleted: only what is held is on disk.
        let n = std::fs::read_dir(&ring.dir).unwrap().count();
        assert_eq!(n, ring.files);
    }

    #[test]
    fn stderr_bits_survive_chunks_and_go_with_their_chunk_on_eviction() {
        let (mut ring, _d) = ring(100_000);
        // A stderr byte early in every chunk, stdout elsewhere, over many chunks.
        for _ in 0..8 {
            ring.push(b"e", true);
            ring.push(&vec![b'o'; FRAME_MAX - 1], false);
        }
        let (from, data, ranges) = ring.read(0, 1 << 20);
        assert_eq!(from, ring.base);
        assert!(
            ranges
                .iter()
                .all(|&(a, n)| n == 1 && a % FRAME_MAX as u64 == 0)
        );
        for (i, b) in data.iter().enumerate() {
            let off = from + i as u64;
            assert_eq!(*b == b'e', ranges.iter().any(|&(a, _)| a == off), "{off}");
        }
        // Only the chunks held have sidecars: evicted ones took theirs along.
        let sidecars = std::fs::read_dir(&ring.dir)
            .unwrap()
            .filter(|e| {
                e.as_ref()
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .ends_with(".e")
            })
            .count();
        assert!(
            ring.files > 0 && sidecars == ring.files,
            "{sidecars} sidecars, {} chunks",
            ring.files
        );
    }

    #[test]
    fn a_read_spans_chunks_and_matches_what_was_written() {
        let (mut ring, _d) = ring(10_000_000);
        fill(&mut ring, 300_000);
        let (from, data, _) = ring.read(70_000, 200_000);
        assert_eq!(from, 70_000);
        let expect: Vec<u8> = (70_000..270_000usize).map(|i| (i % 251) as u8).collect();
        assert_eq!(data, expect);
        assert!(ring.read(300_000, 10).1.is_empty());
    }

    /// A `max` the client chose (`u64::MAX` is a valid JSON number) must not overflow the arithmetic
    /// of a read from inside a chunk: it panicked while holding the job's lock.
    #[test]
    fn a_read_of_any_size_from_any_offset_does_not_overflow() {
        let (mut ring, _d) = ring(10_000_000);
        fill(&mut ring, 1000);
        for offset in [0, 1, 500, 999, 1000, 5000, u64::MAX] {
            let (from, data, _) = ring.read(offset, usize::MAX);
            assert_eq!(from, offset, "{offset}");
            assert_eq!(
                data.len() as u64,
                1000u64.saturating_sub(offset),
                "{offset}"
            );
        }
    }

    #[test]
    fn tiny_writes_do_not_make_tiny_chunks_and_a_small_job_touches_no_disk() {
        let (mut ring, d) = ring(10_000_000);
        for _ in 0..100_000 {
            ring.push(b"x", false);
        }
        assert_eq!(ring.files, 1);
        assert_eq!(ring.tail.len(), 100_000 - FRAME_MAX);
        let (mut small, d2) = self::ring(10_000_000);
        small.push(b"hello", false);
        assert!(!d2.path().join("job").exists());
        drop((small, d2));
        // Dropping the ring removes its directory.
        let path = ring.dir.clone();
        assert!(path.exists());
        drop(ring);
        assert!(!path.exists());
        drop(d);
    }

    #[test]
    fn files_are_private_and_never_follow_a_link() {
        use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};
        let (mut ring, _d) = ring(10_000_000);
        ring.push(&vec![7u8; FRAME_MAX], false);
        let dir_mode = std::fs::metadata(&ring.dir).unwrap().mode() & 0o777;
        let file_mode = std::fs::metadata(ring.chunk_path(0))
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!((dir_mode, file_mode), (0o700, 0o600));
        // A link planted where the next chunk goes is refused, and the bytes read back as a gap.
        let target = ring.dir.join("elsewhere");
        std::os::unix::fs::symlink(&target, ring.chunk_path(FRAME_MAX as u64)).unwrap();
        ring.push(&vec![8u8; FRAME_MAX], false);
        assert!(!target.exists());
        assert_eq!(ring.base, 2 * FRAME_MAX as u64);
    }

    // -- The differential test: the ring against a plain `Vec<(u8, bool)>` ---------------------------

    /// A small deterministic generator (splitmix64): a failing case is reproduced by its seed.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n.max(1)
        }
        /// A length: often a boundary of a chunk, else small, else a few chunks.
        fn len(&mut self) -> usize {
            let c = FRAME_MAX;
            match self.below(8) {
                0 => 0,
                1 => 1,
                2 => c - 1,
                3 => c,
                4 => c + 1,
                5 => self.below(100) as usize,
                6 => self.below(c as u64) as usize,
                _ => self.below(3 * c as u64) as usize,
            }
        }
    }

    /// The model: every byte ever pushed with its stream, the first offset a refused chunk left
    /// readable, and the chunk (and file) a fault was planted for.
    struct Model {
        all: Vec<(u8, bool)>,
        floor: u64,
        planted: Option<(u64, bool)>,
        /// Names of the directories planted where a chunk or its bits go (a directory refuses
        /// `create_new` and reads back as no file).
        blockers: std::collections::HashSet<String>,
    }

    impl Model {
        /// The oldest offset held: whole chunks, the last one that still leaves `retain` bytes (the
        /// retention), but never before the first chunk a refused write left.
        fn base(&self, retain: usize) -> u64 {
            let c = FRAME_MAX as u64;
            let kept = (self.all.len() as u64).saturating_sub(retain as u64) / c * c;
            kept.max(self.floor)
        }

        /// The maximal runs of stderr in `[from, to)`, as `(start, len)`.
        fn runs(&self, from: u64, to: u64) -> Vec<(u64, u64)> {
            let mut runs: Vec<(u64, u64)> = Vec::new();
            for off in from..to {
                if self.all[off as usize].1 {
                    match runs.last_mut() {
                        Some(l) if l.0 + l.1 == off => l.1 += 1,
                        _ => runs.push((off, 1)),
                    }
                }
            }
            runs
        }
    }

    /// Push `data`, then check what the planted fault (a file where a chunk or its bits go) did.
    fn push(ring: &mut Ring, m: &mut Model, data: &[u8], stderr: bool) {
        let before = m.all.len() as u64;
        ring.push(data, stderr);
        m.all.extend(data.iter().map(|&b| (b, stderr)));
        let c = FRAME_MAX as u64;
        if let Some((chunk, on_bits)) = m.planted
            && m.all.len() as u64 >= (chunk + 1) * c
        {
            {
                let (a, z) = ((chunk * c) as usize, ((chunk + 1) * c) as usize);
                // A bits file only refuses a chunk that has a stderr byte.
                if !on_bits || m.all[a..z].iter().any(|b| b.1) {
                    m.floor = m.floor.max((chunk + 1) * c);
                }
                m.planted = None;
            }
        }
        assert!(m.all.len() as u64 >= before);
    }

    fn check(ring: &Ring, m: &Model, retain: usize, rng: &mut Rng, what: &str) {
        let end = m.all.len() as u64;
        assert_eq!(ring.end(), end, "{what}: end");
        assert_eq!(ring.base, m.base(retain), "{what}: base (retain {retain})");
        // On disk: exactly the chunks held, and a sidecar only beside a chunk.
        let mut chunks = 0;
        for e in std::fs::read_dir(&ring.dir).into_iter().flatten() {
            let name = e.unwrap().file_name().to_string_lossy().into_owned();
            if m.blockers.contains(&name) {
                continue;
            }
            if let Some(stem) = name.strip_suffix(".e") {
                assert!(
                    ring.dir.join(stem).exists(),
                    "{what}: sidecar {name} without its chunk"
                );
            } else {
                chunks += 1;
            }
        }
        assert_eq!(chunks, ring.files, "{what}: chunk files");
        let c = FRAME_MAX as u64;
        for _ in 0..6 {
            let offset = match rng.below(6) {
                0 => 0,
                1 => ring.base.saturating_sub(rng.below(3)),
                2 => ring.base + rng.below(2 * c),
                3 => end.saturating_sub(rng.below(2 * c)),
                4 => end + rng.below(3),
                _ => u64::MAX - rng.below(3),
            };
            let max = match rng.below(5) {
                0 => rng.below(10) as usize,
                1 => rng.below(2 * c) as usize,
                2 => usize::MAX,
                3 => c as usize,
                _ => rng.below(8 * c) as usize,
            };
            let (from, data, ranges) = ring.read(offset, max);
            assert_eq!(
                from,
                offset.max(ring.base),
                "{what}: from of read({offset}, {max})"
            );
            let to = end.max(from).min(from.saturating_add(max as u64));
            let runs = m.runs(from, to.max(from));
            // The window is cut before the first run past RANGES_MAX, and only then.
            let (len, want_runs) = if runs.len() > RANGES_MAX {
                (runs[RANGES_MAX].0 - from, runs[..RANGES_MAX].to_vec())
            } else {
                (to.max(from) - from, runs)
            };
            assert_eq!(
                data.len() as u64,
                len,
                "{what}: length of read({offset}, {max})"
            );
            let expect: Vec<u8> = (from..from + len).map(|o| m.all[o as usize].0).collect();
            assert!(
                data == expect,
                "{what}: bytes of read({offset}, {max}) from {from}"
            );
            assert_eq!(
                ranges, want_runs,
                "{what}: stderr ranges of read({offset}, {max})"
            );
        }
    }

    /// The ring against a plain `Vec<(u8, bool)>` over random pushes, reads and refused chunks: every read
    /// returns exactly the bytes of its window, in order, with exactly its stderr ranges; the ring holds
    /// the oldest chunk boundary at or after `end - retain`; a refused chunk reads back as a gap.
    #[test]
    fn the_ring_matches_a_plain_buffer_under_random_pushes_reads_and_refused_chunks() {
        for seed in 0..60u64 {
            let mut rng = Rng(seed);
            let retain = [
                0,
                1,
                1000,
                FRAME_MAX - 1,
                FRAME_MAX,
                FRAME_MAX + 1,
                150_000,
                300_000,
            ][rng.below(8) as usize];
            let (mut ring, _d) = ring(retain);
            let mut m = Model {
                all: Vec::new(),
                floor: 0,
                planted: None,
                blockers: Default::default(),
            };
            for step in 0..30 {
                match rng.below(10) {
                    // Plant a file where the next chunk (or its bits) will go: the write is refused.
                    0 if m.planted.is_none() && step > 0 => {
                        let chunk = m.all.len() as u64 / FRAME_MAX as u64;
                        let on_bits = rng.below(2) == 0;
                        let at = chunk * FRAME_MAX as u64;
                        let path = if on_bits {
                            ring.chunk_bits_path(at)
                        } else {
                            ring.chunk_path(at)
                        };
                        let _ = spool::make_job_dir(&ring.dir);
                        if std::fs::create_dir(&path).is_ok() {
                            m.blockers
                                .insert(path.file_name().unwrap().to_string_lossy().into_owned());
                            m.planted = Some((chunk, on_bits));
                        }
                    }
                    // Many one-byte alternations of stderr and stdout: more runs than a read answers.
                    1 => {
                        for i in 0..rng.below(3000) {
                            push(&mut ring, &mut m, &[i as u8], i % 2 == 0);
                        }
                    }
                    2..=6 => {
                        let n = rng.len();
                        let data: Vec<u8> =
                            (0..n).map(|i| (i as u64 ^ rng.below(251)) as u8).collect();
                        let stderr = rng.below(3) == 0;
                        push(&mut ring, &mut m, &data, stderr);
                    }
                    _ => {}
                }
                // A planted file is not a held chunk: it sits where a chunk has not been written.
                check(
                    &ring,
                    &m,
                    retain,
                    &mut rng,
                    &format!("seed {seed} step {step}"),
                );
            }
        }
    }

    /// Production decides through the kernel and keeps no copy of its arithmetic.
    #[test]
    fn production_calls_the_kernel_for_every_decision() {
        let src = include_str!("ring.rs");
        let prod = src.split("#[cfg(test)]").next().unwrap_or(src);
        for call in [
            "kernel::window(",
            "kernel::seg(",
            "kernel::evict(",
            "kernel::after_flush(",
            "kernel::set_bits(",
            "kernel::take_chunk_bits(",
            "kernel::scan(",
        ] {
            assert!(prod.contains(call), "ring.rs does not call {call}");
        }
        for copy in [
            "fn set_bits",
            "fn bit(",
            "const RANGES_MAX",
            "const CHUNK_BITS",
        ] {
            assert!(!prod.contains(copy), "ring.rs restates the kernel: {copy}");
        }
    }

    /// A job whose first chunk the disk refuses leaves its directory behind when the ring is dropped:
    /// the directory exists (it was made for that chunk), so `made` must say so.
    #[test]
    fn a_ring_whose_first_chunk_was_refused_still_removes_its_directory() {
        let (mut ring, _d) = ring(10_000_000);
        spool::make_job_dir(&ring.dir).unwrap();
        std::fs::create_dir(ring.chunk_path(0)).unwrap();
        ring.push(&vec![1u8; FRAME_MAX], false);
        assert_eq!(
            ring.base, FRAME_MAX as u64,
            "the refused chunk reads back as a gap"
        );
        let dir = ring.dir.clone();
        drop(ring);
        assert!(!dir.exists(), "the job's directory outlives its ring");
    }
}
