//! The request body parser: the envelope, a `job` object's operation and `notify`, `stdin_base64`
//! and the frame encoding it feeds. Nothing may panic, whatever the bytes are; the server answers
//! `400` for a body this rejects (`tests/parsers.rs` holds it to that against the real binary).
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = execd::__fuzz::request(data);
});
