//! `Keyring::verify_at` on arbitrary token strings against the contract-vector ring
//! (`verify/key_vectors.json`: kid 1, seed byte 7): no panic, and the verdict is deterministic at a
//! fixed clock. The seed corpus is that file's tokens, so mutations start from well-formed
//! `bai_v1`/`bai_v2`/`bai_v3` shapes and reach the base64, length and signature checks.
//!
//! Input: 8 bytes of clock (unix seconds), the rest is the token.
#![no_main]

use beyond_ai::__test::key::Keyring;
use libfuzzer_sys::fuzz_target;
use std::sync::OnceLock;

fn ring() -> &'static Keyring {
    static RING: OnceLock<Keyring> = OnceLock::new();
    RING.get_or_init(|| {
        let sk = ed25519_dalek::SigningKey::from_bytes(&[7u8; 32]);
        [(1u32, sk.verifying_key())].into_iter().collect()
    })
}

fuzz_target!(|data: &[u8]| {
    let Some((now, token)) = data.split_first_chunk::<8>() else {
        return;
    };
    // A token reaches `verify` as a `&str`; only valid UTF-8 is a case.
    let Ok(token) = std::str::from_utf8(token) else {
        return;
    };
    let now = u64::from_le_bytes(*now);
    let first = ring().verify_at(token, now);
    let second = ring().verify_at(token, now);
    assert_eq!(first.is_ok(), second.is_ok());
});
