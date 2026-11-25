# beyond-job-client-ffi

The job client's kernel (`crates/job-client`) behind a C ABI, for the SDKs that cannot link Rust: one WASM module for
TypeScript (`sdk/ts/sandbox/src/kernel/wasm.ts`, inlined as base64) and one native library for Python's `ctypes`.

It holds **no logic**. Each export decodes its bytes, makes one call into the proven function, and encodes the answer.
`tests/no_decision.rs` fails if a source here grows a branch, a loop, a comparison, arithmetic, a three-digit number or a
constant of the kernel, if an export is anything but one `io(.., codec::<fn>)` call, or if the crate reaches into the kernel
by any path but its three proven functions and its types. It is outside the Verus run (`.github/scripts/verify-kernels.sh`).

## ABI (version 2)

All exports take `(in_ptr, in_len, out_ptr, out_cap)` and return the answer's length, or a negative code: `-1` bad input
(cut short, unknown tag, not a mark), `-2` the output buffer is too small, `-3` the cursor's fields are not a cursor. A
native caller passes its own buffers (`ctypes.create_string_buffer`); a WASM host writes into memory from `jc_alloc(n)` /
`jc_free(p, n)`. Nothing is kept between calls: the state is a value the caller holds and passes back. All integers are
little endian; a mark is 33 bytes (its length, then 32); a state is 85 opaque bytes that only the kernel makes.

| Export                | Input                                             | Answer                               |
| --------------------- | ------------------------------------------------- | ------------------------------------ |
| `jc_abi_version`      | none (returns `u32`)                              | the layout version of this table     |
| `jc_state_for_start`  | `key u64, mode u8, remove_on_exit u8, beat_s u32` | a state                              |
| `jc_state_for_attach` | `mark, offset u64, mode, remove_on_exit, beat_s`  | a state                              |
| `jc_state_for_cursor` | `mode, remove_on_exit, beat_s u32, cursor bytes`  | a state                              |
| `jc_step`             | `state, event, now_ms u64, draw u32`              | `state, n u8, action * n`            |
| `jc_classify`         | `op u8, status u16`                               | `0` go on, `1` retry, `2 fault` stop |
| `jc_silence_ms`       | `beat_s u32`                                      | `u64` ms                             |
| `jc_cursor_of`        | `state, scope_len u8, scope, id`                  | the cursor string                    |

Events, frames and actions are tagged in the order `job_client::step` declares them; `src/codec.rs` is the layout.

## Build

`build.sh` builds `wasm32-unknown-unknown` (release, the compiler pinned in `toolchain`, `opt-level = s`, LTO, `panic = abort`, paths remapped) and writes
`sdk/ts/sandbox/src/kernel/wasm.ts` and `kernel.src.sha256` (the hash of this crate's and the kernel's sources). The
TypeScript test `kernel.test.ts` fails when a source changed and `build.sh` did not run; CI runs `build.sh --check`, which
rebuilds and fails when the committed bytes differ. Python's wheel build uses `cargo build --release` of the same crate
as a `cdylib`.

The module is about 35 KB (14 KB gzipped), has no imports, and loads (decode, compile, instantiate) in under 1 ms on Node 24.
