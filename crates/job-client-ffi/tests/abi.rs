//! The exports as a host calls them: a length past the bound is refused before a byte is read.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use job_client_ffi::{jc_abi_version, jc_state_for_cursor, jc_step};

#[test]
fn an_input_past_the_bound_is_refused_not_read() {
    // A length no buffer holds, over a pointer that points at one byte: were it read, this would fault.
    let one = [0u8; 1];
    let mut out = [0u8; 16];
    let n = jc_state_for_cursor(one.as_ptr(), 0x1_0001, out.as_mut_ptr(), out.len());
    assert_eq!(n, -1, "BAD_INPUT");
    assert_eq!(
        jc_step(one.as_ptr(), usize::MAX, out.as_mut_ptr(), out.len()),
        -1
    );
}

#[test]
fn a_long_cursor_within_the_bound_is_a_bad_cursor_not_a_crash() {
    let mut input = vec![0u8, 0, 0, 0, 0, 0];
    input.extend(std::iter::repeat_n(b'a', 60_000));
    let mut out = [0u8; 128];
    assert_eq!(
        jc_state_for_cursor(input.as_ptr(), input.len(), out.as_mut_ptr(), out.len()),
        -3
    );
}

#[test]
fn the_abi_version_is_the_one_the_codecs_speak() {
    assert_eq!(jc_abi_version(), 3);
}
