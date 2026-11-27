//! The job output ring: the last bytes of a job's output, by offset from its first byte, kept as
//! whole 64 KiB chunks (one file at a stride a chunk, or a file a chunk: [`Layout`], see
//! [`crate::spool`]) plus an unfinished chunk in memory, one stderr bit per byte.
//!
//! Every decision (which offsets are held, where a read starts and how its pieces tile the window,
//! which chunks retention drops, what a refused chunk costs, which byte ranges are stderr) is the
//! pure kernel [`execd_ring`], proven in Verus (R1..R6, see its docs); this file keeps the I/O:
//! files, the copy of bytes, and the bits' storage. A differential test holds it to a plain
//! `Vec<(u8, bool)>` over random pushes, reads and refused chunks.

use std::fs::File;
use std::os::fd::AsFd as _;
use std::path::PathBuf;
use std::sync::atomic::Ordering;

use execd_ring as kernel;

use crate::frame::FRAME_MAX;
use crate::spool;

pub(crate) use kernel::RANGES_MAX;

// The kernel's chunk is execd's frame, and its bits are one per byte.
const _: () = assert!(kernel::CHUNK == FRAME_MAX && kernel::CHUNK == 8 * kernel::CHUNK_BITS);

/// The pack's name in a job's ring directory.
const PACK: &str = "chunks";

/// The pack's stride: chunk `k`'s bytes at `k * stride()`, its stderr bits at `+ FRAME_MAX`.
///
/// A stride rather than a packing, so a chunk's file offset is arithmetic on its index: no table to
/// keep, nothing a client can name, and eviction is one punch of one stride.
const fn stride() -> u64 {
    if cfg!(feature = "mut_pack_stride_overlap") {
        // The mutant: a chunk's bits start where the next chunk's bytes do, so its labels land on
        // its successor's data. The differential test and the stderr tests go red.
        FRAME_MAX as u64
    } else {
        (FRAME_MAX + kernel::CHUNK_BITS) as u64
    }
}

/// How a ring keeps the chunks it holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Layout {
    /// One file per job ([`PACK`]), a [`stride`] per chunk: one inode for a job's whole output, one
    /// `pwrite` a chunk, one `pread` a chunk served and one open a read. Eviction is a
    /// `FALLOC_FL_PUNCH_HOLE` of the chunk's stride, so the filesystem must punch
    /// (`spool::punch_hole_supported`).
    ///
    /// Its length is the record of which chunks are whole: a chunk's bytes go down first and its
    /// bits after them, so a length that stops inside a payload is a chunk that is not all there,
    /// and a chunk the disk refused takes the file back to where the ring's window started.
    Pack,
    /// One file per chunk, and one `.e` sidecar per chunk that has stderr: the layout for a
    /// filesystem that cannot punch, where eviction is an `unlink`.
    PerChunk,
}

/// The test seam for a pack write whose tail never reached the disk (`EXECD_TEST_FAULT`,
/// `pack_torn=<bytes>`, debug builds only): after a chunk lands, the file keeps that many bytes
/// fewer and execd exits at once, as if the journal's advance had reached the disk and the chunk it
/// names had not. `tests/pack.rs` restarts from there and holds the answer to the bytes written.
fn lost_pack_tail(pack: Option<&File>) {
    let Some(bytes) = crate::fault::at("pack_torn").and_then(|v| v.parse::<u64>().ok()) else {
        return;
    };
    if let Some(pack) = pack
        && let Ok(len) = pack.metadata().map(|m| m.len())
    {
        let _ = pack.set_len(len.saturating_sub(bytes));
    }
    std::process::exit(86);
}

/// The last bytes of a job's output, by offset from its first byte, kept as whole chunks under a
/// directory (see [`crate::spool`]). A chunk is the 64 KiB at a multiple of 64 KiB; whole chunks are
/// evicted from the front, so the retention is a floor. In memory: a path, the unfinished chunk,
/// and one "has stderr bits" byte per full chunk.
pub(crate) struct Ring {
    /// This job's directory, created when the first chunk is written.
    dir: PathBuf,
    made: bool,
    /// Full chunks on disk.
    files: usize,
    /// Chunks written to disk since the ring was made (a refusal does not count): a test's way to
    /// see that a push flushed one.
    flushed: usize,
    /// The bytes after them, fewer than a chunk.
    tail: Vec<u8>,
    /// The offset of the first byte held (a multiple of a chunk).
    pub(crate) base: u64,
    retain: usize,
    /// One bit per byte of `tail`, set for a byte that came from stderr. Allocated at the first
    /// stderr byte, so a job that never writes stderr pays nothing; may be shorter than `tail`
    /// (a missing bit is stdout).
    tail_e: Option<Vec<u8>>,
    /// How the full chunks are kept.
    layout: Layout,
    /// The pack, open while the ring writes to it: every chunk is a `pwrite` on it, not a file of
    /// its own. `None` for [`Layout::PerChunk`], and at rest — an ended job holds no descriptor.
    pack: Option<File>,
    /// One byte per full chunk held, in order from `base`: whether the chunk's stride carries
    /// stderr bits. Only the pack's regions are worth asking about: a per-chunk layout learns it
    /// from the sidecar's existence.
    bits: Vec<u8>,
    /// The punch failure was reported once.
    warned_punch: bool,
    /// The directory is not this ring's to delete: its owner (a job's spool) removes it.
    keep: bool,
    /// Keeps open the descriptor `dir` names, when it is a path through one (a job's spool).
    _hold: Option<std::sync::Arc<std::os::fd::OwnedFd>>,
}

impl Ring {
    pub(crate) fn new(retain: usize, dir: PathBuf, layout: Layout) -> Self {
        Self {
            dir,
            made: false,
            files: 0,
            flushed: 0,
            tail: Vec::new(),
            base: 0,
            retain,
            tail_e: None,
            layout,
            pack: None,
            bits: Vec::new(),
            warned_punch: false,
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

    /// A ring over the chunks a previous execd left in `dir`, in the layout they are in: one file
    /// named [`PACK`] is the pack, anything else is a run of per-chunk files. `prefer` is the
    /// layout of a ring that has nothing there yet. `floor` is the journal's first byte (see
    /// [`Self::recover_pack`] and the per-chunk pass below).
    pub fn recover(retain: usize, dir: PathBuf, floor: u64, prefer: Layout) -> Self {
        match std::fs::symlink_metadata(dir.join(PACK)) {
            Ok(m) if m.is_file() => return Self::recover_pack(retain, dir, floor),
            // A link or a directory where the pack's name goes is not one of ours.
            Ok(_) => {
                let _ = std::fs::remove_file(dir.join(PACK));
            }
            Err(_) => {}
        }
        // The longest run of whole chunk files that ends at the highest one. Anything else there (a
        // chunk the crash cut short, one cut off from the run by a refused write) is deleted. The
        // unfinished tail is not here: the journal has it.
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
        // A run of chunk files is the layout this ring keeps, whatever this run prefers: the
        // capability probe answers per directory per run, so a directory whose filesystem changed
        // between two runs (a remount, a different mount's options) recovers the chunks it has and
        // goes on writing them, rather than reading a pack that does not hold them. An empty
        // directory takes this run's layout.
        let layout = if have.is_empty() {
            prefer
        } else {
            Layout::PerChunk
        };
        let mut ring = Self::new(retain, dir.clone(), layout).kept();
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

    /// [`Self::recover`] for the pack. Chunk `k` is held when the file's length reaches the end of
    /// its payload, and the first chunk held is at the file's first data byte: eviction punched
    /// every stride before it, and a hole reads as zeros, so the length alone would name chunks
    /// whose bytes are gone.
    ///
    /// **A stride the crash cut short is not a chunk.** A chunk's bytes are written first and its
    /// bits after them, so a length that stops inside a payload is a chunk whose bytes are not all
    /// there: it is not counted, and the next chunk's write covers it. `floor`, the journal's first byte, is where
    /// the recovered ring has to reach: a pack that stops short of it lost a chunk the journal had
    /// advanced past, and everything before the journal's first byte is a gap then — the same gap a
    /// per-chunk file cut short leaves, and the same one a disk that refuses a chunk leaves.
    fn recover_pack(retain: usize, dir: PathBuf, floor: u64) -> Self {
        let mut ring = Self::new(retain, dir.clone(), Layout::Pack).kept();
        // Everything in the ring's directory that is not the pack is not this ring's: a chunk file
        // of a build that wrote them, a sidecar, anything planted. The pack names its own chunks.
        if let Ok(rd) = std::fs::read_dir(&dir) {
            for e in rd.flatten() {
                if e.file_name() != PACK {
                    let _ = std::fs::remove_file(e.path());
                }
            }
        }
        // Read and write: the tail writes the chunks, and a read serves them from the same
        // descriptor while the job runs.
        let Ok(pack) = spool::open_pack(&dir.join(PACK)) else {
            return ring;
        };
        let frame = FRAME_MAX as u64;
        let len = pack.metadata().map(|m| m.len()).unwrap_or(0);
        let whole = if cfg!(feature = "mut_pack_keeps_short_tail") {
            // The mutant: the length is rounded up to a stride, so a stride the crash cut short is
            // named as a whole chunk and its missing bytes read back as zeros. `ring.rs`'s torn
            // test and `tests/pack.rs` go red.
            len.div_ceil(stride())
        } else if len >= frame {
            (len - frame) / stride() + 1
        } else {
            0
        };
        let end = whole * frame;
        // The front the pack no longer holds, from the file's own blocks: eviction punched it. The
        // first data byte is rounded up to a stride boundary, so a stride with a hole anywhere in
        // it is never a stride of bytes.
        let mut base = match nix::unistd::lseek(pack.as_fd(), 0, nix::unistd::Whence::SeekData) {
            Ok(off) => u64::try_from(off).unwrap_or(0).div_ceil(stride()) * frame,
            // Nothing but a hole: nothing is held.
            Err(_) => end,
        }
        .min(end);
        let outran = if cfg!(feature = "mut_pack_ignores_journal_floor") {
            // The mutant: a pack that stops short of the journal's first byte is kept as it is, so
            // the chunk the journal had advanced past reads back as the hole it is (zeros) at an
            // offset the answer still names. `tests/pack.rs` goes red.
            false
        } else {
            end < floor && floor.is_multiple_of(frame)
        };
        if outran {
            // The pack stops short of the journal's first byte: a chunk the journal had advanced
            // past and the disk did not keep. Everything in the pack is behind it, so all of it is
            // the gap it was before the crash — the same one a per-chunk file cut short leaves —
            // and the file starts again at the journal's byte, with nothing of it on disk.
            let _ = pack.set_len(0);
            base = floor;
        }
        let files = end.saturating_sub(base) / frame;
        ring.base = base;
        ring.files = files as usize;
        // A recovered chunk's bits are read: a hole is no bits, which is the same answer a chunk
        // that never had one gives. Recovering which strides have any would cost a seek each.
        ring.bits = vec![1; files as usize];
        ring.made = true;
        ring.pack = Some(pack);
        ring
    }

    /// [`Self::recover`] for a job whose output the previous execd dropped and whose entry it kept
    /// (a tombstone, `kept` here): nothing is held, and the offsets continue from `end`, so a
    /// client's reads never go backwards. It touches no file: the job's spool is gone.
    pub fn evicted(retain: usize, dir: PathBuf, end: u64) -> Self {
        let mut ring = Self::new(retain, dir, Layout::PerChunk);
        ring.base = end;
        ring
    }

    /// Drop every byte the ring holds, in memory and on disk, keeping only the offset: `base` moves
    /// to `end`, so a read answers from there and a stream sends one gap. The files that held the
    /// bytes are returned for the caller to delete **off the runtime thread**; so is the spool,
    /// which held them (`Job::spool`), and the ring's own directory is not its to delete (`kept`).
    /// Idempotent: a second call finds nothing held and returns nothing.
    pub fn evict_all(&mut self) -> Vec<PathBuf> {
        let mut files = Vec::new();
        match self.layout {
            Layout::PerChunk => {
                files.reserve(self.files * 2);
                for i in 0..self.files {
                    let at = self.base + (i * FRAME_MAX) as u64;
                    files.push(self.chunk_path(at));
                    files.push(self.chunk_bits_path(at));
                }
            }
            // One file holds every chunk: it goes as one.
            Layout::Pack if self.made => files.push(self.dir.join(PACK)),
            Layout::Pack => {}
        }
        self.base = self.end();
        self.files = 0;
        self.bits.clear();
        self.tail.clear();
        self.tail_e = None;
        self.made = false;
        // An ended job holds no descriptor (ARCHITECTURE.md, Bounds): the ring's handle on the spool
        // directory goes with the spool, and so does its handle on the pack.
        self.pack = None;
        self._hold = None;
        files
    }

    /// Close the pack's descriptor. A running job holds it, so its flushes are one `pwrite` each; an
    /// ended job holds none (ARCHITECTURE.md, Bounds), so its reads open the file once a read.
    pub fn close_pack(&mut self) {
        self.pack = None;
    }

    /// The test seam for a journal write whose chunk's bytes never reached the disk
    /// (`EXECD_TEST_FAULT`, `pack_torn_after_journal=<bytes>`, debug builds only): the pack keeps
    /// that many bytes fewer and execd exits at once. Called after the journal has recorded a push,
    /// so the journal's advance is on disk and the chunk it names is not — the ordering a machine
    /// crash can produce and a process crash cannot. `tests/pack.rs` restarts from there.
    pub fn lose_pack_tail(&self) {
        let Some(bytes) =
            crate::fault::at("pack_torn_after_journal").and_then(|v| v.parse::<u64>().ok())
        else {
            return;
        };
        let Some(pack) = self.pack.as_ref() else {
            return;
        };
        if let Ok(len) = pack.metadata().map(|m| m.len()) {
            let _ = pack.set_len(len.saturating_sub(bytes));
        }
        std::process::exit(86);
    }

    /// How many chunks this ring has written to disk (a refused one does not count).
    pub fn chunks_flushed(&self) -> usize {
        self.flushed
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

    /// Where the chunk that holds `at` starts in the pack.
    fn slot(&self, at: u64) -> u64 {
        (at / FRAME_MAX as u64) * stride()
    }

    /// Whether the chunk that starts at `at` carries stderr bits (a missing answer is no).
    fn chunk_has_bits(&self, at: u64) -> bool {
        let i = ((at / FRAME_MAX as u64) - (self.base / FRAME_MAX as u64)) as usize;
        self.bits.get(i).is_some_and(|&b| b != 0)
    }

    /// Open the pack to write the next chunk, if it is not open: one descriptor for a job's whole
    /// output, made the first time a chunk is written.
    fn open_pack(&mut self) -> bool {
        if self.pack.is_none() {
            self.pack = spool::open_pack(&self.dir.join(PACK)).ok();
        }
        self.pack.is_some()
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
            self.made = made;
            let wrote = made && self.write_chunk(at, bits.as_deref());
            self.tail.drain(..FRAME_MAX);
            if !wrote {
                // The disk refused it: what was held and this chunk read back as a gap.
                self.forget_files();
            }
            if wrote {
                self.flushed += 1;
            }
            (self.base, self.files) = kernel::after_flush(self.base, self.files, wrote);
        }
        for _ in 0..kernel::evict(self.files, self.tail.len(), self.retain) {
            self.evict_front();
            self.base += FRAME_MAX as u64;
            self.files -= 1;
        }
    }

    /// Write the chunk the tail starts with, and its bits when it has any.
    fn write_chunk(&mut self, at: u64, bits: Option<&[u8]>) -> bool {
        if self.layout == Layout::PerChunk {
            return spool::write_chunk(&self.chunk_path(at), &self.tail[..FRAME_MAX]).is_ok()
                && bits.is_none_or(|b| {
                    let ok = spool::write_chunk(&self.chunk_bits_path(at), b).is_ok();
                    if !ok {
                        // Bytes without their labels would read back as stdout: a gap instead.
                        let _ = std::fs::remove_file(self.chunk_path(at));
                    }
                    ok
                });
        }
        // The pack: the bytes at the stride, then the bits after them, so the file's length says
        // exactly how much of the stride is there — a chunk whose payload is complete is a chunk,
        // and one the crash cut short is not, whatever its bits region holds. (A crash between the
        // two writes is the window a chunk file and its sidecar have today: the bytes are there and
        // the labels are missing, which reads as stdout.)
        if !self.open_pack() {
            return false;
        }
        let slot = self.slot(at);
        let bits_at = slot + FRAME_MAX as u64;
        let wrote = match self.pack.as_ref() {
            Some(pack) => {
                spool::write_at(pack, slot, &self.tail[..FRAME_MAX]).is_ok()
                    && bits.is_none_or(|b| spool::write_at(pack, bits_at, b).is_ok())
            }
            None => false,
        };
        if wrote {
            self.bits.push(u8::from(bits.is_some()));
            // A test's write whose tail the kernel lost, and the crash that follows it.
            lost_pack_tail(self.pack.as_ref());
        }
        wrote
    }

    /// The disk refused a chunk: every chunk held reads back as a gap, so nothing below the next
    /// chunk is served again. The per-chunk layout deletes the files; the pack punches the same
    /// strides to holes, so that a restart finds no data where the ring holds nothing.
    fn forget_files(&mut self) {
        match self.layout {
            Layout::PerChunk => {
                for i in 0..self.files {
                    let at = self.base + (i * FRAME_MAX) as u64;
                    let _ = std::fs::remove_file(self.chunk_path(at));
                    let _ = std::fs::remove_file(self.chunk_bits_path(at));
                }
            }
            Layout::Pack => {
                if let Some(pack) = self.pack.as_ref() {
                    // What was held, and the stride the disk refused, are not the ring's any more,
                    // and a half-written stride must never be counted as a chunk: the file goes
                    // back to where it started, so its length names no chunk of its own. What the
                    // ring writes next lands beyond it, and the hole between is the same gap.
                    let _ = pack.set_len(self.slot(self.base));
                }
                self.bits.clear();
            }
        }
    }

    /// Drop the chunk at `base`.
    fn evict_front(&mut self) {
        match self.layout {
            Layout::PerChunk => {
                let _ = std::fs::remove_file(self.chunk_path(self.base));
                let _ = std::fs::remove_file(self.chunk_bits_path(self.base));
            }
            Layout::Pack => {
                // A punch: the stride's blocks go, its offset stays (the pack's length is the
                // stream's byte count), and a hole reads as zeros — which is why recovery takes
                // the base from the first data byte and never from the length.
                if !self.warned_punch
                    && let Some(pack) = self.pack.as_ref()
                    && let Err(e) = spool::punch(pack, self.slot(self.base), stride())
                {
                    self.warned_punch = true;
                    crate::log_warn!(
                        "cannot return an evicted chunk to the filesystem: {e}; its blocks stay \
                         until the job is removed"
                    );
                }
                if !self.bits.is_empty() {
                    self.bits.remove(0);
                }
            }
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
        // An ended job holds no descriptor: one open serves the whole read, not one a chunk.
        let idle: Option<File> = match self.layout {
            Layout::Pack if self.pack.is_none() && self.files > 0 => {
                spool::open_pack_ro(&self.dir.join(PACK)).ok()
            }
            _ => None,
        };
        let pack = self.pack.as_ref().or(idle.as_ref());
        // The bits of one chunk: the buffer a read reuses for every piece of it, so a chunk's
        // labels cost no allocation (a hole is zeros, which is "no stderr").
        let mut buf = [0u8; kernel::CHUNK_BITS];
        while done < want {
            let at = from + done as u64;
            let seg = kernel::seg(at, want - done, self.base, self.files, self.tail.len());
            // The per-chunk layout's sidecar, when the chunk has one: allocated by this piece and
            // dropped with it; the pack's bits need no allocation at all.
            #[allow(unused_assignments)]
            let mut side: Option<Vec<u8>> = None;
            let (n, bits): (usize, Option<&[u8]>) = if seg.disk {
                let dst = &mut out[done..done + seg.n];
                match self.layout {
                    Layout::Pack => {
                        let Some(p) = pack else { break };
                        let n = spool::read_at(p, self.slot(at) + seg.pos as u64, dst).unwrap_or(0);
                        if n == 0 {
                            break;
                        }
                        if !self.chunk_has_bits(at) {
                            (n, None)
                        } else {
                            // Only what the file answered: a bits region the file does not reach
                            // leaves the buffer's tail as it was, and the missing bits are zeros.
                            let labels =
                                match spool::read_at(p, self.slot(at) + FRAME_MAX as u64, &mut buf)
                                {
                                    Ok(labels) => labels,
                                    // Bytes without their labels would read back as stdout: a gap.
                                    Err(_) => break,
                                };
                            (n, Some(&buf[..labels]))
                        }
                    }
                    Layout::PerChunk => {
                        let Ok(n) = spool::read_chunk(&self.chunk_path(at), seg.pos as u64, dst)
                        else {
                            break;
                        };
                        if n == 0 {
                            break;
                        }
                        side = std::fs::read(self.chunk_bits_path(at)).ok();
                        (n, side.as_deref())
                    }
                }
            } else {
                out[done..done + seg.n].copy_from_slice(&self.tail[seg.pos..seg.pos + seg.n]);
                (seg.n, self.tail_e.as_deref())
            };
            if let Some(bits) = bits {
                // `bits` is indexed from the chunk's (or the tail's) first byte, as `seg.pos` is.
                let kept = kernel::scan(bits, seg.pos, n, at, &mut ranges);
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
        self.pack = None;
        if self.made && !self.keep {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }
}

/// Fuzz seam: drive a [`Ring`] with the pushes and reads `bytes` describe, and hold it to a plain
/// `Vec` of everything pushed: every read returns the bytes at its offsets, from the oldest kept
/// byte when the offset is older, and the ring keeps at least its retention. Both layouts: the
/// pack's offsets are arithmetic on a chunk's index and a file's are its name, and nothing a client
/// can see may tell them apart.
pub(crate) fn fuzz_ring(bytes: &[u8]) {
    for layout in [Layout::Pack, Layout::PerChunk] {
        fuzz_ring_with(bytes, layout);
    }
}

fn fuzz_ring_with(bytes: &[u8], layout: Layout) {
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
    let mut ring = Ring::new(retain, dir.join("ring"), layout);
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
        ring_in(retain, Layout::PerChunk)
    }

    /// A ring in the pack layout, in a directory of its own.
    fn packed(retain: usize) -> (Ring, tempfile::TempDir) {
        ring_in(retain, Layout::Pack)
    }

    /// [`packed`] for a job's spool: its directory outlives the ring, so a test can recover from
    /// what the disk holds, as a restarted execd does.
    fn packed_kept(retain: usize) -> (Ring, tempfile::TempDir) {
        let (r, d) = ring_in(retain, Layout::Pack);
        (r.kept(), d)
    }

    fn ring_in(retain: usize, layout: Layout) -> (Ring, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let r = Ring::new(retain, dir.path().join("job"), layout);
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
        for layout in [Layout::Pack, Layout::PerChunk] {
            let (mut ring, _d) = ring_in(200_000, layout);
            fill(&mut ring, 1_000_000);
            assert_eq!(ring.end(), 1_000_000);
            let kept = ring.end() - ring.base;
            assert!(
                (200_000..200_000 + FRAME_MAX as u64).contains(&kept),
                "{layout:?}: kept {kept}"
            );
            assert_eq!(ring.held() as u64, kept);
            // The oldest bytes are gone: a read from 0 starts where the ring does.
            let (from, data, _) = ring.read(0, 5);
            assert_eq!(from, ring.base);
            assert_eq!(data.len(), 5);
            // Evicted chunks are gone from disk: one file holds the pack, and a per-chunk ring
            // keeps exactly the chunks it holds.
            let n = std::fs::read_dir(&ring.dir).unwrap().count();
            assert_eq!(
                n,
                if layout == Layout::Pack {
                    1
                } else {
                    ring.files
                }
            );
        }
    }

    #[test]
    fn stderr_bits_survive_chunks_and_go_with_their_chunk_on_eviction() {
        for layout in [Layout::Pack, Layout::PerChunk] {
            stderr_bits(layout);
        }
    }

    fn stderr_bits(layout: Layout) {
        let (mut ring, _d) = ring_in(100_000, layout);
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
        if layout == Layout::PerChunk {
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
    }

    #[test]
    fn a_read_spans_chunks_and_matches_what_was_written() {
        for layout in [Layout::Pack, Layout::PerChunk] {
            let (mut ring, _d) = ring_in(10_000_000, layout);
            fill(&mut ring, 300_000);
            let (from, data, _) = ring.read(70_000, 200_000);
            assert_eq!(from, 70_000);
            let expect: Vec<u8> = (70_000..270_000usize).map(|i| (i % 251) as u8).collect();
            assert_eq!(data, expect, "{layout:?}");
            assert!(ring.read(300_000, 10).1.is_empty());
        }
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
        let (mut ring, d) = ring_in(10_000_000, Layout::Pack);
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

    // -- The pack: one file a job, at a stride a chunk -------------------------------------------------

    /// Every full chunk of `n` bytes is in the pack, at its stride, and its bits after it.
    #[test]
    fn a_chunk_goes_at_its_stride_in_the_one_file() {
        let (mut ring, _d) = packed(10_000_000);
        // Three chunks, the middle one with a stderr byte.
        fill(&mut ring, FRAME_MAX);
        ring.push(b"e", true);
        ring.push(&vec![b'o'; FRAME_MAX - 1], false);
        let third: Vec<u8> = (2 * FRAME_MAX..3 * FRAME_MAX)
            .map(|i| (i % 251) as u8)
            .collect();
        ring.push(&third, false);
        assert_eq!(ring.files, 3);
        assert_eq!(
            std::fs::read_dir(&ring.dir).unwrap().count(),
            1,
            "a pack is one file"
        );
        let f = spool::open_pack_ro(&ring.dir.join(PACK)).unwrap();
        use std::os::unix::fs::FileExt as _;
        // Chunk 0's first bytes are chunk 0's, at offset 0; chunk 1's at one stride; chunk 2's at
        // two, and chunk 1's bits in the 8 KiB after chunk 1's payload.
        let mut b = [0u8; 1];
        let want2 = ((2 * FRAME_MAX + 5) % 251) as u8;
        for (i, want) in [(0u64, 5u8), (1, b'o'), (2, want2)] {
            f.read_exact_at(&mut b, i * stride() + 5).unwrap();
            assert_eq!(b[0], want, "chunk {i} at its stride");
        }
        f.read_exact_at(&mut b, stride() + FRAME_MAX as u64)
            .unwrap();
        assert_eq!(b[0] & 1, 1, "chunk 1's first stderr bit");
        let (from, data, ranges) = ring.read(0, 3 * FRAME_MAX);
        assert_eq!((from, data.len(), ranges.len()), (0, 3 * FRAME_MAX, 1));
        assert_eq!(ranges, vec![(FRAME_MAX as u64, 1)]);
    }

    /// A job's whole output is one inode, however many chunks it holds: 8 MiB, four files.
    #[test]
    fn no_file_is_created_per_chunk() {
        let (mut ring, d) = packed(16 * 1024 * 1024);
        fill(&mut ring, 8 * 1024 * 1024);
        assert_eq!(ring.files, 128);
        let names: Vec<String> = std::fs::read_dir(&ring.dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            names,
            vec![PACK.to_string()],
            "8 MiB, {} files",
            names.len()
        );
        // The same job in the per-chunk layout is 128 files plus 128 sidecars; that is what this
        // layout exists to remove.
        let (mut files, d2) = ring_in(16 * 1024 * 1024, Layout::PerChunk);
        fill(&mut files, 8 * 1024 * 1024);
        assert_eq!(std::fs::read_dir(&files.dir).unwrap().count(), 128);
        drop((files, d2));
        drop(d);
    }

    /// A chunk's bytes are written before its bits, so a stride the crash cut short is not a chunk:
    /// the length stops inside the payload, and what it names is every chunk before it.
    ///
    /// Red under `mut_pack_keeps_short_tail`.
    #[test]
    fn a_stride_the_crash_cut_short_is_not_a_chunk() {
        let (mut ring, d) = packed_kept(10_000_000);
        fill(&mut ring, 2 * FRAME_MAX);
        let path = ring.dir.join(PACK);
        // The third chunk's payload is half there: the length is inside its stride.
        let f = spool::open_pack(&path).unwrap();
        spool::write_at(&f, 2 * stride(), &vec![9u8; FRAME_MAX / 2]).unwrap();
        drop(f);
        drop(ring);
        let ring = Ring::recover(
            10_000_000,
            d.path().join("job"),
            2 * FRAME_MAX as u64,
            Layout::Pack,
        );
        assert_eq!(
            ring.end(),
            2 * FRAME_MAX as u64,
            "the torn chunk is no chunk"
        );
        assert_eq!(ring.base, 0);
        let (from, data, _) = ring.read(0, 3 * FRAME_MAX);
        assert_eq!((from, data.len()), (0, 2 * FRAME_MAX));
        let expect: Vec<u8> = (0..2 * FRAME_MAX).map(|i| (i % 251) as u8).collect();
        assert_eq!(data, expect, "no zeros from the torn stride");
    }

    /// The same, with the journal ahead of the pack: a chunk the journal had advanced past and the
    /// disk did not keep. Nothing the pack holds is before the journal's byte any more, so all of
    /// it is the gap it was before the crash, and no byte of it is served as data.
    ///
    /// Red under `mut_pack_ignores_journal_floor`.
    #[test]
    fn a_pack_short_of_the_journals_byte_is_a_gap() {
        let (mut ring, d) = packed_kept(10_000_000);
        fill(&mut ring, 2 * FRAME_MAX);
        drop(ring);
        let ring = Ring::recover(
            10_000_000,
            d.path().join("job"),
            3 * FRAME_MAX as u64,
            Layout::Pack,
        );
        assert_eq!(ring.end(), 3 * FRAME_MAX as u64);
        assert_eq!(ring.base, 3 * FRAME_MAX as u64, "the window is the gap");
        assert_eq!(ring.read(0, 1000).1.len(), 0, "nothing below the gap");
        // A pack whose length says nothing holds nothing, so the bytes read back as the gap.
        assert_eq!(std::fs::metadata(ring.dir.join(PACK)).unwrap().len(), 0);
    }

    /// Eviction is a punch: the evicted chunk's blocks go, the file's length stays (it is the
    /// stream's byte count), and the bytes read back as zeros — which is why the base comes from
    /// the first data byte, never from the length.
    #[test]
    fn an_evicted_chunk_frees_its_blocks() {
        use std::os::unix::fs::MetadataExt as _;
        let (mut ring, _d) = packed(FRAME_MAX);
        fill(&mut ring, 4 * FRAME_MAX);
        let f = spool::open_pack_ro(&ring.dir.join(PACK)).unwrap();
        let held = f.metadata().unwrap();
        if !is_sparse(&f) {
            eprintln!("SKIPPED: this filesystem does not punch a hole (no sparse file)");
            return;
        }
        let blocks = held.blocks();
        // The ring holds one chunk (of four written): its stride is the only one with blocks.
        assert_eq!(ring.files, 1);
        assert!(
            blocks <= (2 * stride()) / 512,
            "{blocks} blocks held for {} chunks",
            ring.files
        );
    }

    /// A pack opened again from what is on disk: the base is where the first data byte is, so a
    /// punched front is never served as zeros, and the bytes and the labels are the ones written.
    #[test]
    fn a_recovered_pack_serves_from_its_first_data_byte() {
        let (mut ring, d) = packed_kept(FRAME_MAX);
        for _ in 0..6 {
            ring.push(&vec![b'e'; FRAME_MAX], true);
        }
        let base = ring.base;
        assert_eq!(base, 5 * FRAME_MAX as u64);
        let dir = ring.dir.clone();
        drop(ring);
        let ring = Ring::recover(FRAME_MAX, dir, 6 * FRAME_MAX as u64, Layout::Pack);
        assert_eq!(ring.base, base, "the punched front is not held");
        assert_eq!(ring.end(), 6 * FRAME_MAX as u64);
        let (from, data, ranges) = ring.read(0, 2 * FRAME_MAX);
        assert_eq!(from, base);
        assert_eq!(data, vec![b'e'; data.len()]);
        assert_eq!(
            ranges,
            vec![(base, data.len() as u64)],
            "the labels are kept"
        );
        drop(d);
    }

    /// **A ring's layout is what its files are.** A job's ring directory that holds chunk files is
    /// recovered as the per-chunk layout even where this run's probe says the filesystem punches
    /// (the answer is per directory per run, and a remount can change it): the bytes and labels are
    /// the ones written, the next chunk is a chunk file too, and no pack appears beside them to
    /// split the window across two layouts.
    #[test]
    fn a_ring_of_chunk_files_recovers_as_one() {
        let (ring, _d) = ring(10_000_000);
        let mut ring = ring.kept();
        ring.push(&vec![b'x'; FRAME_MAX], false);
        ring.push(&vec![b'e'; FRAME_MAX], true);
        assert_eq!(ring.files, 2);
        let dir = ring.dir.clone();
        drop(ring);
        let mut ring = Ring::recover(10_000_000, dir.clone(), 2 * FRAME_MAX as u64, Layout::Pack);
        assert!(ring.layout == Layout::PerChunk, "the layout on disk wins");
        assert_eq!((ring.base, ring.end()), (0, 2 * FRAME_MAX as u64));
        let (_, data, ranges) = ring.read(0, 2 * FRAME_MAX);
        assert_eq!(data[..FRAME_MAX], vec![b'x'; FRAME_MAX]);
        assert!(data[FRAME_MAX..].iter().all(|&b| b == b'e'));
        assert_eq!(
            ranges,
            vec![(FRAME_MAX as u64, FRAME_MAX as u64)],
            "the second chunk's label is kept"
        );
        ring.push(&vec![b'y'; FRAME_MAX], false);
        assert_eq!(ring.files, 3);
        assert!(
            dir.join(format!("{:010x}", 2)).is_file(),
            "the next chunk is a chunk file"
        );
        assert!(!dir.join(PACK).exists(), "no pack beside the chunk files");
    }

    /// An ended job holds no descriptor: the pack is closed, and a read opens it once, serving the
    /// same bytes.
    #[test]
    fn an_ended_ring_closes_its_pack_and_still_reads() {
        let (mut ring, _d) = packed(10_000_000);
        fill(&mut ring, 3 * FRAME_MAX);
        assert!(ring.pack.is_some(), "a running ring holds its pack");
        ring.close_pack();
        assert!(ring.pack.is_none());
        let (from, data, _) = ring.read(0, 3 * FRAME_MAX);
        assert_eq!((from, data.len()), (0, 3 * FRAME_MAX));
        assert_eq!(
            data,
            (0..3 * FRAME_MAX)
                .map(|i| (i % 251) as u8)
                .collect::<Vec<_>>()
        );
    }

    /// A link planted where the pack goes is refused: nothing is written through it, and every
    /// chunk reads back as the gap a refused chunk costs.
    #[test]
    fn a_planted_link_at_the_pack_is_refused() {
        use std::os::unix::fs::PermissionsExt as _;
        let (mut ring, _d) = packed(10_000_000);
        spool::make_job_dir(&ring.dir).unwrap();
        let target = ring.dir.join("elsewhere");
        std::os::unix::fs::symlink(&target, ring.dir.join(PACK)).unwrap();
        ring.push(&vec![7u8; FRAME_MAX], false);
        assert!(!target.exists(), "the link was followed");
        assert_eq!(ring.base, FRAME_MAX as u64, "the chunk read back as a gap");
        assert_eq!(ring.read(0, 10).1.len(), 0);
        assert_eq!(
            std::fs::metadata(&ring.dir).unwrap().permissions().mode() & 0o777,
            0o700
        );
    }

    /// Recovers a fixture written by hand at the stride, to prove the offsets are the format, not
    /// an implementation detail: chunk k at `k * 72 KiB`, bits at `+ 64 KiB`.
    #[test]
    fn the_stride_is_the_format() {
        let d = tempfile::tempdir().unwrap();
        let dir = d.path().join("job");
        spool::make_job_dir(&dir).unwrap();
        let f = spool::open_pack(&dir.join(PACK)).unwrap();
        for k in 0..3u64 {
            spool::write_at(&f, k * stride(), &vec![k as u8 + 1; FRAME_MAX]).unwrap();
        }
        // Chunk 1's first byte is stderr.
        let mut bits = vec![0u8; kernel::CHUNK_BITS];
        bits[0] = 1;
        spool::write_at(&f, stride() + FRAME_MAX as u64, &bits).unwrap();
        drop(f);
        let ring = Ring::recover(10_000_000, dir, 3 * FRAME_MAX as u64, Layout::Pack);
        assert_eq!((ring.base, ring.end()), (0, 3 * FRAME_MAX as u64));
        let (_, data, ranges) = ring.read(0, 3 * FRAME_MAX);
        assert_eq!(data.len(), 3 * FRAME_MAX);
        for k in 0..3usize {
            assert!(
                data[k * FRAME_MAX..(k + 1) * FRAME_MAX]
                    .iter()
                    .all(|&b| b == k as u8 + 1),
                "chunk {k}"
            );
        }
        assert_eq!(ranges, vec![(FRAME_MAX as u64, 1)]);
    }

    /// Whether `f` is sparse enough that a punched stride is a hole: the seek skips the front.
    fn is_sparse(f: &std::fs::File) -> bool {
        let len = f.metadata().map(|m| m.len()).unwrap_or(0);
        len > 0
            && nix::unistd::lseek(f.as_fd(), 0, nix::unistd::Whence::SeekData)
                .map(|off| off > 0)
                .unwrap_or(false)
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
        // On disk: exactly the chunks held, in the layout this ring keeps them in.
        let mut chunks = 0;
        let mut pack = false;
        for e in std::fs::read_dir(&ring.dir).into_iter().flatten() {
            let name = e.unwrap().file_name().to_string_lossy().into_owned();
            if m.blockers.contains(&name) {
                continue;
            }
            if name == PACK {
                pack = true;
            } else if let Some(stem) = name.strip_suffix(".e") {
                assert!(
                    ring.dir.join(stem).exists(),
                    "{what}: sidecar {name} without its chunk"
                );
            } else {
                chunks += 1;
            }
        }
        if ring.layout == Layout::Pack {
            let len = std::fs::metadata(ring.dir.join(PACK)).map_or(0, |m| m.len());
            assert_eq!(
                chunks, 0,
                "{what}: a pack holds the chunks; no file a chunk"
            );
            // The length names the top chunk the ring holds, padded to its stride: every chunk the
            // ring holds is whole, and a length that stops inside a payload is no chunk at all.
            let named = if len >= FRAME_MAX as u64 {
                (len - FRAME_MAX as u64) / stride() + 1
            } else {
                0
            };
            let top = ring.base / FRAME_MAX as u64 + ring.files as u64;
            assert_eq!(
                named, top,
                "{what}: pack length {len} names {named} chunks, the ring holds {} from {}",
                ring.files, ring.base
            );
            assert!(pack || ring.files == 0, "{what}: no pack file");
            // No hole inside the window: every chunk the ring holds has its bytes, and a hole
            // reads as zeros, so a held chunk over one would be served as data that never existed.
            if ring.files > 0 {
                let f = spool::open_pack_ro(&ring.dir.join(PACK)).unwrap();
                for i in 0..ring.files as u64 {
                    let at = ring.base / FRAME_MAX as u64 + i;
                    let want = at * stride();
                    let got =
                        nix::unistd::lseek(f.as_fd(), want as i64, nix::unistd::Whence::SeekData)
                            .map(|o| o as u64)
                            .unwrap_or(u64::MAX);
                    assert_eq!(got, want, "{what}: chunk {at} is a hole");
                }
            }
        } else {
            assert!(!pack, "{what}: a pack in a per-chunk ring");
            assert_eq!(chunks, ring.files, "{what}: chunk files");
        }
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
        // Both layouts answer the same bytes, the same labels and the same retention window. Only
        // the per-chunk layout can be refused a chunk by a planted name; the pack's own refusals (a
        // link or a disk that fails) have their own tests.
        for layout in [Layout::Pack, Layout::PerChunk] {
            the_ring_matches(layout);
        }
    }

    fn the_ring_matches(layout: Layout) {
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
            let (mut ring, _d) = ring_in(retain, layout);
            let mut m = Model {
                all: Vec::new(),
                floor: 0,
                planted: None,
                blockers: Default::default(),
            };
            for step in 0..30 {
                match rng.below(10) {
                    // Plant a file where the next chunk (or its bits) will go: the write is refused.
                    0 if layout == Layout::PerChunk && m.planted.is_none() && step > 0 => {
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
                    &format!("{layout:?} seed {seed} step {step}"),
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
