//! The job output ring against a plain buffer: pushes of any size and reads at any offset and any
//! `max` (including `u64::MAX`, which is a valid JSON number a client can send) must return the
//! bytes at those offsets, never panic and never overflow.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    execd::__fuzz::ring(data);
});
