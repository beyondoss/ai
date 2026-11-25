//! The exports. Each is one call to [`io`], which hands the codec function its input and output as slices.
//! `jc_alloc` and `jc_free` are the memory a WASM host writes its arguments into; a native caller (ctypes)
//! passes its own buffers and never calls them.
//!
//! Every function returns the length of its answer, or a negative code (`codec::BAD_INPUT`, `SHORT_OUT`,
//! `BAD_CURSOR`). Nothing is kept between calls: the state is a value the caller holds and passes back.

#![allow(unsafe_code)]

use std::ptr::NonNull;

use crate::codec;

/// The ABI's version: the layout of every record in `codec.rs`. A change to a layout bumps it.
pub const ABI_VERSION: u32 = 3;

/// The most input one call takes. A host that passes more has a wrong length: no record here is near it.
const MAX_IN: usize = 0x1_0000;

type Call = fn(&[u8], &mut [u8]) -> Result<usize, i32>;

/// Runs `call` on the buffers `(inp, n)` and `(out, cap)`. A length past [`MAX_IN`] is `BAD_INPUT`,
/// before any byte is read.
fn io(inp: *const u8, n: usize, out: *mut u8, cap: usize, call: Call) -> i32 {
    MAX_IN
        .checked_sub(n)
        .map_or(codec::BAD_INPUT, |_| run(inp, n, out, cap, call))
}

fn run(inp: *const u8, n: usize, out: *mut u8, cap: usize, call: Call) -> i32 {
    // A null pointer is an empty buffer: the slice is then the empty one, never a null slice.
    let input = NonNull::new(inp.cast_mut()).map_or(&[][..], |p| {
        // SAFETY: the caller promises `n` readable bytes at `inp` for this call.
        unsafe { std::slice::from_raw_parts(p.as_ptr(), n) }
    });
    let output = NonNull::new(out).map_or(&mut [][..], |p| {
        // SAFETY: the caller promises `cap` writable bytes at `out`, not overlapping `inp`.
        unsafe { std::slice::from_raw_parts_mut(p.as_ptr(), cap) }
    });
    call(input, output).map_or_else(
        |code| code,
        |len| i32::try_from(len).unwrap_or(codec::SHORT_OUT),
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn jc_abi_version() -> u32 {
    ABI_VERSION
}

#[unsafe(no_mangle)]
pub extern "C" fn jc_state_for_start(inp: *const u8, n: usize, out: *mut u8, cap: usize) -> i32 {
    io(inp, n, out, cap, codec::state_for_start)
}

#[unsafe(no_mangle)]
pub extern "C" fn jc_state_for_attach(inp: *const u8, n: usize, out: *mut u8, cap: usize) -> i32 {
    io(inp, n, out, cap, codec::state_for_attach)
}

#[unsafe(no_mangle)]
pub extern "C" fn jc_state_for_cursor(inp: *const u8, n: usize, out: *mut u8, cap: usize) -> i32 {
    io(inp, n, out, cap, codec::state_for_cursor)
}

#[unsafe(no_mangle)]
pub extern "C" fn jc_step(inp: *const u8, n: usize, out: *mut u8, cap: usize) -> i32 {
    io(inp, n, out, cap, codec::step)
}

#[unsafe(no_mangle)]
pub extern "C" fn jc_classify(inp: *const u8, n: usize, out: *mut u8, cap: usize) -> i32 {
    io(inp, n, out, cap, codec::classify)
}

#[unsafe(no_mangle)]
pub extern "C" fn jc_silence_ms(inp: *const u8, n: usize, out: *mut u8, cap: usize) -> i32 {
    io(inp, n, out, cap, codec::silence)
}

#[unsafe(no_mangle)]
pub extern "C" fn jc_cursor_of(inp: *const u8, n: usize, out: *mut u8, cap: usize) -> i32 {
    io(inp, n, out, cap, codec::cursor_of)
}

/// `n` bytes of this module's memory for the host to write into (WASM); 0 for none.
#[unsafe(no_mangle)]
pub extern "C" fn jc_alloc(n: usize) -> *mut u8 {
    let mut v = Vec::<u8>::with_capacity(n);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

/// Returns what `jc_alloc(n)` gave. (An exported C function cannot be `unsafe fn`-shaped for its caller.)
#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn jc_free(p: *mut u8, n: usize) {
    // SAFETY: `p` came from `jc_alloc(n)`, which made a `Vec<u8>` of capacity `n`, and is not used again.
    drop(unsafe { Vec::from_raw_parts(p, 0, n) });
}
