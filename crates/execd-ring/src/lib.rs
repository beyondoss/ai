//! execd's job-output ring, as pure decisions. A job's output is kept by byte offset as whole chunks
//! of [`CHUNK`] bytes (files, owned by the shell) plus an unfinished tail in memory; one stderr bit per
//! byte rides along (`tail` bits in memory, a sidecar file per chunk). Which offsets are held, where a
//! read starts and in which piece of which chunk it continues, which chunks retention drops, what a
//! refused chunk costs, and which byte ranges are stderr are decided here, inside `verus!`, so the
//! function production runs is the one Verus checks, with no bound on offsets, chunks or reads. The
//! shell (`crates/execd/src/ring.rs`) keeps the I/O: it opens files, copies bytes and removes chunks.
//!
//! The ring holds `[base, end)` where `end = base + files * CHUNK + tail`, `base` a multiple of
//! [`CHUNK`], `tail < CHUNK`; `end` is the number of bytes ever pushed. What is proven (R1..R6):
//!
//! | | property | function |
//! |---|---|---|
//! | R1 | a read starts at `max(offset, base)` and answers at most `max` bytes, none past `end` | [`window`] |
//! | R2 | each piece of a read is `n >= 1` bytes inside one chunk file (or the tail), at the file position its offset names, within `[at, window end)`: pieces tile the window with no loss, no repeat, no gap | [`seg`] |
//! | R3 | retention drops whole chunks only, never leaves fewer than `retain` bytes, and drops every chunk it can: the ring holds `[end - retain, end)` and less than `retain + CHUNK` bytes | [`evict`] |
//! | R4 | a chunk the disk refused costs the offsets of what was held, never the offset arithmetic: `end` is unchanged and `base` lands on a chunk boundary | [`after_flush`] |
//! | R5 | the stderr bit of a byte is set exactly when it was pushed from stderr, in the tail and across the chunk split | [`set_bits`], [`take_chunk_bits`] |
//! | R6 | the ranges of a read are exactly the stderr bytes of the window it answers: ascending, disjoint and not adjacent, at most [`RANGES_MAX`]; the window is cut only when one more range would exceed it | [`scan`] |
//!
//! Not proven here: that a chunk file holds the bytes written to it (the filesystem), and the shell's loop
//! around [`seg`] and [`scan`] (a differential test, `ring.rs`, holds it to a plain `Vec`). Offsets
//! cannot overflow: every function requires `end <= u64::MAX` (16 EiB of output) and the shell keeps it.
//! Mutations (cargo features `mut_*`) each remove one guard; the proof and the differential test both
//! go red under each.

#![allow(
    clippy::match_like_matches_macro,
    clippy::if_same_then_else,
    clippy::implicit_saturating_sub,
    clippy::needless_return,
    unused_braces,
    clippy::let_and_return,
    clippy::manual_div_ceil,
    clippy::manual_is_multiple_of
)]

use vstd::prelude::*;

verus! {

global size_of usize == 8;

/// Bytes per chunk: execd's frame size (checked there with a const assertion).
pub const CHUNK: usize = 65536;
/// Bytes of stderr bits for one chunk.
pub const CHUNK_BITS: usize = 8192;
/// The most stderr ranges one read answers.
pub const RANGES_MAX: usize = 1024;

// ---------------------------------------------------------------------------------------------
// Offsets and retention.
// ---------------------------------------------------------------------------------------------

/// R1. Where a read from `offset` starts, for a ring holding `[base, end)`, and how many bytes it
/// may answer of at most `max`: from the oldest byte held when `offset` is older.
pub fn window(offset: u64, base: u64, end: u64, max: usize) -> (r: (u64, usize))
    ensures
        r.0 == (if offset >= base { offset } else { base }),
        r.1 <= max,
        r.1 as int == (if end > r.0 { if (end - r.0) as int > max as int { max as int } else { (end - r.0) as int } } else { 0int }),
        r.0 < end && max > 0 ==> r.1 > 0,
        r.0 as int + r.1 as int <= if end > r.0 { end as int } else { r.0 as int },
{
    let from = if cfg!(feature = "mut_window_before_base") {
        offset
    } else if offset > base {
        offset
    } else {
        base
    };
    let avail: u64 = if end > from { end - from } else { 0 };
    let want: usize = if avail > max as u64 { max } else { avail as usize };
    (from, want)
}

/// One piece of a read: `n` bytes at `pos` of a chunk file (`disk`) or of the tail.
pub struct Seg {
    pub disk: bool,
    pub pos: usize,
    pub n: usize,
}

/// The first offset the tail holds.
pub open spec fn on_disk(base: u64, files: usize) -> int {
    base as int + files as int * CHUNK as int
}

/// The offset of the next byte.
pub open spec fn end_of(base: u64, files: usize, tail: usize) -> int {
    base as int + files as int * CHUNK as int + tail as int
}

/// The ring's shape: the first byte held is on a chunk boundary, the tail is less than a chunk, and
/// no offset passes `u64::MAX`.
pub open spec fn shape(base: u64, files: usize, tail: usize) -> bool {
    base as int % CHUNK as int == 0 && tail < CHUNK && end_of(base, files, tail) <= u64::MAX as int
}

/// R2. The piece of a read that starts at `at`, with `left` bytes still wanted: in the chunk file
/// that holds `at` (its index is `at / CHUNK`, the position `at % CHUNK`) up to that file's end, or in
/// the tail up to its end. `n >= 1` is the progress that lets the shell's loop reach the end of the
/// window; `at + n <= end` and `n <= left` keep it inside.
pub fn seg(at: u64, left: usize, base: u64, files: usize, tail: usize) -> (s: Seg)
    requires
        shape(base, files, tail),
        base <= at,
        at as int <= end_of(base, files, tail),
        at as int != end_of(base, files, tail),
        left > 0,
    ensures
        s.n >= 1,
        s.n <= left,
        at as int + s.n as int <= end_of(base, files, tail),
        s.disk <==> (at as int) < on_disk(base, files),
        s.disk ==> {
            &&& s.pos as int == at as int % CHUNK as int
            &&& s.pos + s.n <= CHUNK
            &&& at as int + s.n as int <= on_disk(base, files)
            &&& base as int <= (at as int / CHUNK as int) * CHUNK as int
            &&& (at as int / CHUNK as int) * CHUNK as int + CHUNK as int <= on_disk(base, files)
        },
        !s.disk ==> {
            &&& s.pos as int == at as int - on_disk(base, files)
            &&& s.pos + s.n <= tail
        },
{
    let disk_end: u64 = base + (files as u64) * (CHUNK as u64);
    let disk = if cfg!(feature = "mut_seg_boundary_off_by_one") { at <= disk_end && files > 0 } else { at < disk_end };
    if disk {
        let pos = (at % (CHUNK as u64)) as usize;
        let room = CHUNK - pos;
        let n = if left < room { left } else { room };
        Seg { disk: true, pos, n }
    } else {
        let t = (at - disk_end) as usize;
        let room = tail - t;
        let n = if left < room { left } else { room };
        Seg { disk: false, pos: t, n }
    }
}

/// R3. After a push, how many chunks retention drops from the front: whole chunks while the rest still
/// holds `retain` bytes. The ring then holds `[end - retain, end)` and, past one chunk, nothing more
/// than `retain + CHUNK - 1` bytes; with no whole chunk to drop it drops none.
pub fn evict(files: usize, tail: usize, retain: usize) -> (k: usize)
    requires
        tail < CHUNK,
        files as int * CHUNK as int + tail as int <= u64::MAX as int,
    ensures
        k <= files,
        k > 0 ==> files as int * CHUNK as int + tail as int - k as int * CHUNK as int >= retain as int,
        k < files ==> (files as int * CHUNK as int + tail as int - (k as int + 1) * CHUNK as int) < (retain as int),
{
    if files == 0 {
        return 0;
    }
    let held: usize = files * CHUNK + tail;
    if held - CHUNK < retain {
        return 0;
    }
    let by_retain = (held - retain) / CHUNK;
    let k = if cfg!(feature = "mut_evict_one_extra") && by_retain < files { by_retain + 1 } else if by_retain < files { by_retain } else { files };
    k
}

/// R4. What a full tail chunk becomes: written (`wrote`) it is one more file; refused by the disk,
/// what was held and this chunk read back as a gap, so `base` moves past all of it. Either way `end`
/// is unchanged. `files` is the count before the chunk.
pub fn after_flush(base: u64, files: usize, wrote: bool) -> (r: (u64, usize))
    requires
        base as int % CHUNK as int == 0,
        end_of(base, files, CHUNK) <= u64::MAX as int,
    ensures
        wrote ==> r.0 == base && r.1 == files + 1,
        !wrote ==> r.0 as int == base as int + (files as int + 1) * CHUNK as int && r.1 == 0,
        r.0 as int % CHUNK as int == 0,
        end_of(r.0, r.1, 0) == end_of(base, files, CHUNK),
{
    assert((files as int + 1) * CHUNK as int == files as int * CHUNK as int + CHUNK as int) by (nonlinear_arith);
    if wrote {
        (base, files + 1)
    } else if cfg!(feature = "mut_flush_fail_base") {
        (base + (files as u64) * (CHUNK as u64), 0)
    } else {
        (base + ((files + 1) as u64) * (CHUNK as u64), 0)
    }
}

// ---------------------------------------------------------------------------------------------
// Stderr bits: bit `i` of a byte string is `bits[i / 8] >> (i % 8) & 1`; a missing byte is zeros.
// ---------------------------------------------------------------------------------------------

pub open spec fn byte_bit(b: u8, k: int) -> bool {
    (b >> (k as u8)) & 1u8 == 1u8
}

pub open spec fn bit_at(bits: Seq<u8>, i: int) -> bool {
    0 <= i && i / 8 < bits.len() && byte_bit(bits[i / 8], i % 8)
}

proof fn lemma_or_bit(b: u8, s: int, k: int)
    requires
        0 <= s < 8,
        0 <= k < 8,
    ensures
        byte_bit(b | (1u8 << (s as u8)), k) == (byte_bit(b, k) || k == s),
{
    let s8 = s as u8;
    let k8 = k as u8;
    assert(((b | (1u8 << s8)) >> k8) & 1u8 == 1u8 <==> (((b >> k8) & 1u8 == 1u8) || k8 == s8)) by (bit_vector)
        requires s8 < 8 && k8 < 8;
}

proof fn lemma_ff_bit(k: int)
    requires
        0 <= k < 8,
    ensures
        byte_bit(0xFFu8, k),
{
    let k8 = k as u8;
    assert((0xFFu8 >> k8) & 1u8 == 1u8) by (bit_vector)
        requires k8 < 8;
}

proof fn lemma_zero_bit(k: int)
    requires
        0 <= k < 8,
    ensures
        !byte_bit(0u8, k),
{
    let k8 = k as u8;
    assert(!((0u8 >> k8) & 1u8 == 1u8)) by (bit_vector)
        requires k8 < 8;
}

/// A byte of zeros has no bits set.
proof fn lemma_zero_byte(bits: Seq<u8>, p: int)
    requires
        0 <= p / 8 < bits.len(),
        bits[p / 8] == 0u8,
        0 <= p,
    ensures
        !bit_at(bits, p),
{
    lemma_zero_bit(p % 8);
}

/// Changing byte `idx` changes only the bits of that byte.
proof fn lemma_other_byte(bits: Seq<u8>, idx: int, v: u8, j: int)
    requires
        0 <= idx < bits.len(),
        j / 8 != idx,
    ensures
        bit_at(bits.update(idx, v), j) == bit_at(bits, j),
{
}

/// Whether bit `i` of `bits` is set; bits past the end are clear.
pub fn bit(bits: &[u8], i: usize) -> (r: bool)
    ensures
        r == bit_at(bits@, i as int),
{
    if i / 8 < bits.len() {
        let k = (i % 8) as u8;
        (bits[i / 8] >> k) & 1u8 == 1u8
    } else {
        false
    }
}

/// R5. Set bits `[a, b)` of `bits`, growing it with zeros as needed; every other bit is as it was.
pub fn set_bits(bits: &mut Vec<u8>, a: usize, b: usize)
    requires
        a <= b,
        b + 8 <= usize::MAX,
    ensures
        forall|j: int| #[trigger] bit_at(final(bits)@, j) == (bit_at(old(bits)@, j) || (a as int <= j && j < b as int)),
{
    let need = (b + 7) / 8;
    let ghost orig = bits@;
    if bits.len() < need {
        bits.resize(need, 0u8);
    }
    proof {
        assert forall|j: int| #[trigger] bit_at(bits@, j) == bit_at(orig, j) by {
            if 0 <= j && j / 8 < bits@.len() && j / 8 >= orig.len() {
                lemma_zero_bit(j % 8);
            }
        }
    }
    let ghost grown = bits@;
    let mut i = a;
    while i < b
        invariant
            a <= i <= b,
            b + 8 <= usize::MAX,
            bits.len() >= need,
            need == (b + 7) / 8,
            forall|j: int| #[trigger] bit_at(bits@, j) == (bit_at(grown, j) || (a as int <= j && j < i as int)),
            forall|j: int| #[trigger] bit_at(grown, j) == bit_at(orig, j),
        decreases b - i,
    {
        let ghost before = bits@;
        if i % 8 == 0 && i + 8 <= b {
            bits.set(i / 8, 0xFFu8);
            proof {
                assert forall|j: int| #[trigger] bit_at(bits@, j) == (bit_at(grown, j) || (a as int <= j && j < (i + 8) as int)) by {
                    if j / 8 == (i / 8) as int {
                        if 0 <= j {
                            lemma_ff_bit(j % 8);
                        }
                    } else {
                        lemma_other_byte(before, (i / 8) as int, 0xFFu8, j);
                    }
                }
            }
            i += 8;
        } else {
            let idx = i / 8;
            let k = (i % 8) as u8;
            let old_byte = bits[idx];
            assert(k as int == i as int % 8);
            bits.set(idx, old_byte | (1u8 << k));
            proof {
                assert forall|j: int| #[trigger] bit_at(bits@, j) == (bit_at(grown, j) || (a as int <= j && j < (i + 1) as int)) by {
                    assert(bit_at(before, j) == (bit_at(grown, j) || (a as int <= j && j < i as int)));
                    if j / 8 == idx as int {
                        if 0 <= j {
                            assert(bits@[idx as int] == old_byte | (1u8 << k));
                            assert(before[idx as int] == old_byte);
                            lemma_or_bit(old_byte, i as int % 8, j % 8);
                            assert(j % 8 == i as int % 8 ==> j == i as int);
                        }
                    } else {
                        lemma_other_byte(before, idx as int, old_byte | (1u8 << k), j);
                    }
                }
            }
            i += 1;
        }
    }
}

/// R5. A full chunk's bits, taken off the front of the tail's `e`: `chunk` is the first
/// [`CHUNK_BITS`] bytes (the bits of the chunk's [`CHUNK`] bytes, zeros if `e` was shorter) and `rest`
/// the tail's bits from the next byte on, bit for bit.
pub fn take_chunk_bits(e: Vec<u8>) -> (r: (Vec<u8>, Vec<u8>))
    ensures
        r.0.len() == CHUNK_BITS,
        forall|j: int| 0 <= j && j < CHUNK as int ==> #[trigger] bit_at(r.0@, j) == bit_at(e@, j),
        forall|j: int| 0 <= j ==> #[trigger] bit_at(r.1@, j) == bit_at(e@, j + CHUNK as int),
{
    let ghost orig = e@;
    let mut chunk = e;
    if chunk.len() < CHUNK_BITS {
        chunk.resize(CHUNK_BITS, 0u8);
    }
    let cut = if cfg!(feature = "mut_bits_shifted") { CHUNK_BITS - 1 } else { CHUNK_BITS };
    let ghost whole = chunk@;
    let rest = chunk.split_off(cut);
    proof {
        // `whole` is `orig` zero-extended to CHUNK_BITS bytes (or `orig` itself, if longer).
        assert forall|j: int| 0 <= j && j < CHUNK as int implies #[trigger] bit_at(chunk@, j) == bit_at(orig, j) by {
            if j / 8 >= orig.len() && j / 8 < chunk@.len() {
                lemma_zero_bit(j % 8);
            }
        }
        assert forall|j: int| 0 <= j implies #[trigger] bit_at(rest@, j) == bit_at(orig, j + CHUNK as int) by {
            let q = j + CHUNK as int;
            if q / 8 >= orig.len() && j / 8 < rest@.len() {
                lemma_zero_bit(q % 8);
            }
        }
    }
    (chunk, rest)
}

// ---------------------------------------------------------------------------------------------
// Stderr ranges.
// ---------------------------------------------------------------------------------------------

/// The label of the byte at offset `off`, when bit `bi` is the label of the byte at `start`.
pub open spec fn label(bits: Seq<u8>, bi: int, start: int, off: int) -> bool {
    bit_at(bits, bi + off - start)
}

/// Whether some range of `r` covers offset `off`.
pub open spec fn cov(r: Seq<(u64, u64)>, off: int) -> bool {
    exists|k: int| 0 <= k < r.len() && r[k].0 as int <= off && off < r[k].0 as int + r[k].1 as int
}

/// Ranges are non-empty, ascending, pairwise separated by at least one byte (so none touch), and at
/// most [`RANGES_MAX`].
pub open spec fn ranges_wf(r: Seq<(u64, u64)>) -> bool {
    &&& r.len() <= RANGES_MAX
    &&& forall|k: int| 0 <= k < r.len() ==> r[k].1 > 0 && r[k].0 as int + r[k].1 as int <= u64::MAX as int
    &&& forall|k: int, l: int| 0 <= k < l < r.len() ==> (r[k].0 as int + r[k].1 as int) < (r[l].0 as int)
}

/// Every range ends by `limit`.
pub open spec fn ends_by(r: Seq<(u64, u64)>, limit: int) -> bool {
    forall|k: int| 0 <= k < r.len() ==> r[k].0 as int + r[k].1 as int <= limit
}

/// Adding a range at `st` (past every range, not touching the last) keeps ranges well formed and covers
/// exactly what was covered plus the new range.
proof fn lemma_push(before: Seq<(u64, u64)>, st: u64, len: u64)
    requires
        ranges_wf(before),
        before.len() < RANGES_MAX,
        len > 0,
        st as int + len as int <= u64::MAX as int,
        ends_by(before, st as int),
        before.len() > 0 ==> before[before.len() - 1].0 as int + before[before.len() - 1].1 as int != st as int,
    ensures
        ranges_wf(before.push((st, len))),
        forall|off: int| #[trigger] cov(before.push((st, len)), off) == (cov(before, off) || (st as int <= off && off < st as int + len as int)),
{
    let after = before.push((st, len));
    assert forall|k: int| 0 <= k < before.len() implies (before[k].0 as int + before[k].1 as int) < st as int by {
        if k == before.len() - 1 {
        } else {
            assert((before[k].0 as int + before[k].1 as int) < (before[before.len() - 1].0 as int));
        }
    }
    assert forall|off: int| cov(after, off) == (cov(before, off) || (st as int <= off && off < st as int + len as int)) by {
        if cov(after, off) && !(st as int <= off && off < st as int + len as int) {
            let k = choose|k: int| 0 <= k < after.len() && after[k].0 as int <= off && off < after[k].0 as int + after[k].1 as int;
            assert(k < before.len());
            assert(before[k] == after[k]);
        }
        if cov(before, off) {
            let k = choose|k: int| 0 <= k < before.len() && before[k].0 as int <= off && off < before[k].0 as int + before[k].1 as int;
            assert(after[k] == before[k]);
        }
        if st as int <= off && off < st as int + len as int {
            assert(after[before.len() as int] == (st, len));
        }
    }
}

/// Lengthening the last range (which ends at `st`) by `len` keeps ranges well formed and covers exactly
/// what was covered plus `[st, st + len)`.
proof fn lemma_extend(before: Seq<(u64, u64)>, st: u64, len: u64)
    requires
        ranges_wf(before),
        before.len() > 0,
        before[before.len() - 1].0 as int + before[before.len() - 1].1 as int == st as int,
        len > 0,
        st as int + len as int <= u64::MAX as int,
    ensures
        ranges_wf(before.update(before.len() - 1, (before[before.len() - 1].0, (before[before.len() - 1].1 + len) as u64))),
        forall|off: int| #[trigger] cov(before.update(before.len() - 1, (before[before.len() - 1].0, (before[before.len() - 1].1 + len) as u64)), off)
            == (cov(before, off) || (st as int <= off && off < st as int + len as int)),
{
    let l = before.len() - 1;
    let after = before.update(l, (before[l].0, (before[l].1 + len) as u64));
    assert forall|off: int| cov(after, off) == (cov(before, off) || (st as int <= off && off < st as int + len as int)) by {
        if cov(after, off) && !(st as int <= off && off < st as int + len as int) {
            let k = choose|k: int| 0 <= k < after.len() && after[k].0 as int <= off && off < after[k].0 as int + after[k].1 as int;
            if k == l {
                assert(before[l].0 as int <= off && off < before[l].0 as int + before[l].1 as int);
            } else {
                assert(before[k] == after[k]);
            }
        }
        if cov(before, off) {
            let k = choose|k: int| 0 <= k < before.len() && before[k].0 as int <= off && off < before[k].0 as int + before[k].1 as int;
            if k == l {
                assert(after[l].0 == before[l].0);
            } else {
                assert(after[k] == before[k]);
            }
        }
        if st as int <= off && off < st as int + len as int {
            assert(after[l].0 as int <= off && off < after[l].0 as int + after[l].1 as int);
        }
    }
}

/// R6. Extend `ranges` with the stderr runs of `n` bytes that start at offset `start`, whose labels are
/// bits `bi..bi + n` of `bits`; returns how many of the `n` bytes the answer keeps (`m == n` unless
/// one more range would pass [`RANGES_MAX`], when it is the bytes before that run). The ranges then
/// cover exactly the stderr bytes of `[start, start + m)` and of what they covered before, and
/// nothing after. A run that touches the last range extends it, even across pieces.
pub fn scan(bits: &[u8], bi: usize, n: usize, start: u64, ranges: &mut Vec<(u64, u64)>) -> (m: usize)
    requires
        ranges_wf(old(ranges)@),
        ends_by(old(ranges)@, start as int),
        start as int + n as int <= u64::MAX as int,
        bi + n + 8 <= usize::MAX,
    ensures
        m <= n,
        ranges_wf(final(ranges)@),
        forall|off: int| start as int <= off < start as int + m as int ==> #[trigger] cov(final(ranges)@, off) == label(bits@, bi as int, start as int, off),
        forall|off: int| off < start as int ==> #[trigger] cov(final(ranges)@, off) == cov(old(ranges)@, off),
        forall|off: int| off >= start as int + m as int ==> !(#[trigger] cov(final(ranges)@, off)),
        m < n ==> {
            &&& final(ranges)@.len() == RANGES_MAX
            &&& bit_at(bits@, bi as int + m as int)
            &&& final(ranges)@[RANGES_MAX - 1].0 as int + final(ranges)@[RANGES_MAX - 1].1 as int != start as int + m as int
        },
{
    let mut i: usize = 0;
    while i < n
        invariant
            i <= n,
            start as int + n as int <= u64::MAX as int,
            bi + n + 8 <= usize::MAX,
            ranges_wf(ranges@),
            ends_by(ranges@, start as int + i as int),
            forall|off: int| start as int <= off < start as int + i as int ==> #[trigger] cov(ranges@, off) == label(bits@, bi as int, start as int, off),
            forall|off: int| off < start as int ==> #[trigger] cov(ranges@, off) == cov(old(ranges)@, off),
            forall|off: int| off >= start as int + i as int ==> !(#[trigger] cov(ranges@, off)),
        decreases n - i,
    {
        if !bit(bits, bi + i) {
            if (bi + i) % 8 == 0 && (bi + i) / 8 < bits.len() && bits[(bi + i) / 8] == 0u8 {
                let step = if i + 8 < n { i + 8 } else { n };
                proof {
                    lemma_zero_byte(bits@, (bi + i) as int);
                    assert forall|off: int| start as int + i as int <= off < start as int + step as int implies #[trigger] cov(ranges@, off) == label(bits@, bi as int, start as int, off) by {
                        let p = bi as int + off - start as int;
                        assert(p / 8 == (bi + i) as int / 8) by {
                            assert((bi + i) as int % 8 == 0);
                            assert((bi + i) as int <= p < (bi + i) as int + 8);
                        }
                        lemma_zero_bit(p % 8);
                        assert(bits@[p / 8] == 0u8);
                    }
                }
                i = step;
            } else {
                i += 1;
            }
            continue;
        }
        let st: u64 = start + i as u64;
        let mut j = i;
        proof {
            assert(bit_at(bits@, (bi + i) as int));
        }
        while j < n && bit(bits, bi + j)
            invariant
                i <= j <= n,
                bi + n + 8 <= usize::MAX,
                forall|p: int| (bi + i) as int <= p < (bi + j) as int ==> #[trigger] bit_at(bits@, p),
            decreases n - j,
        {
            j += 1;
        }
        let len: u64 = (j - i) as u64;
        let l = ranges.len();
        let adjacent = l > 0 && ranges[l - 1].0 + ranges[l - 1].1 == st;
        let ghost before = ranges@;
        if adjacent {
            let last = ranges[l - 1];
            ranges.set(l - 1, (last.0, last.1 + len));
            proof {
                lemma_extend(before, st, len);
                assert(ranges@ == before.update(before.len() - 1, (before[before.len() - 1].0, (before[before.len() - 1].1 + len) as u64)));
                assert forall|off: int| #[trigger] cov(ranges@, off) == (cov(before, off) || (st as int <= off && off < st as int + len as int)) by {
                    assert(cov(ranges@, off) == cov(before.update(before.len() - 1, (before[before.len() - 1].0, (before[before.len() - 1].1 + len) as u64)), off));
                }
            }
        } else if ranges.len() >= RANGES_MAX && !cfg!(feature = "mut_ranges_unbounded") {
            return i;
        } else {
            ranges.push((st, len));
            proof {
                lemma_push(before, st, len);
                assert(ranges@ == before.push((st, len)));
                assert forall|off: int| #[trigger] cov(ranges@, off) == (cov(before, off) || (st as int <= off && off < st as int + len as int)) by {
                    assert(cov(ranges@, off) == cov(before.push((st, len)), off));
                    assert(cov(before.push((st, len)), off) == (cov(before, off) || (st as int <= off && off < st as int + len as int)));
                }
            }
        }
        proof {
            assert forall|off: int| start as int <= off < start as int + j as int implies #[trigger] cov(ranges@, off) == label(bits@, bi as int, start as int, off) by {
                if off >= st as int {
                    assert(bit_at(bits@, bi as int + off - start as int));
                }
            }
            assert forall|k: int| 0 <= k < ranges@.len() implies ranges@[k].0 as int + ranges@[k].1 as int <= start as int + j as int by {
                if adjacent && k == l as int - 1 {
                } else if k < before.len() {
                    assert(ranges@[k] == before[k]);
                }
            }
        }
        i = j;
    }
    n
}

} // verus!
