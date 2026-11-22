# beyond-execd-ring

The decisions of execd's job output ring, as pure functions: where a read starts and how its pieces tile
the window, which chunks retention drops, what a refused chunk costs, and which byte ranges are stderr.
execd's `src/ring.rs` calls them and keeps only the I/O (chunk files, the tail, the bits' storage).

## Proven

Written inside `verus!`, so the function production runs is the one [Verus](https://github.com/verus-lang/verus)
checks, with no bound on offsets, chunks or reads. `mise run verify:kernels` checks them
(`.github/scripts/verify-kernels.sh`, the release pinned in `.github/scripts/verus.env`).

| Property | Where |
| --- | --- |
| R1 a read starts at `max(offset, base)` and answers at most `max` bytes, none past `end` | `window` |
| R2 each piece of a read is at least one byte, inside one chunk file (or the tail), at the position its offset names: pieces tile the window with no loss, repeat or gap | `seg` |
| R3 retention drops whole chunks only, never leaves fewer than `retain` bytes, and drops every chunk it can | `evict` |
| R4 a refused chunk keeps `end` and puts `base` on a chunk boundary | `after_flush` |
| R5 a byte's stderr bit is set exactly when it came from stderr, in the tail and across the chunk split | `set_bits`, `take_chunk_bits` |
| R6 a read's ranges are exactly the stderr bytes of its window: ascending, disjoint, not touching, at most `RANGES_MAX`; the window is cut only when one more range would exceed it | `scan` |

Not proven: that a chunk file holds the bytes written to it (the filesystem), and the shell's loop around
`seg` and `scan`; `execd`'s `ring::tests` holds both to a `Vec<(u8, bool)>` under random pushes, reads and
refused chunks. Offsets cannot overflow under the stated `end <= u64::MAX`.

## Mutations

Cargo features `mut_*` each remove one guard: `mut_window_before_base`, `mut_seg_boundary_off_by_one`,
`mut_evict_one_extra`, `mut_bits_shifted`, `mut_ranges_unbounded`, `mut_flush_fail_base`. Each turns
`mise run verify:kernels -- --features <mut>` red and
`cargo test -p beyond-execd --features <mut> --lib ring::tests::the_ring` red.
