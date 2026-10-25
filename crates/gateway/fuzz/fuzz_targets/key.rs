//! `Keyring::verify_at` on arbitrary token strings against the contract-vector ring
//! (`verify/key_vectors.json`: kid 1, seed byte 7): no panic, and no forgery.
//!
//! The forgery property: any input the ring accepts must be byte-for-byte the token the ring's own
//! signer mints for the identity it decodes to. Ed25519 signing is deterministic and the token
//! encoding is canonical (decimal kid, unpadded base64url with no stray bits, strict signatures), so
//! an accepted token with a second spelling, or one a different signer produced, is a verifier bug.
//! The seed is public, so the target can sign what the verifier should accept and compare.
//!
//! The assertion is checked against the vectors' own tokens before fuzzing starts, so it cannot fire
//! on a valid token. The seed corpus is those tokens, so mutations start from well-formed
//! `bai_v1`/`bai_v2`/`bai_v3` shapes and reach the base64, length and signature checks.
//!
//! Input: 8 bytes of clock (unix seconds), the rest is the token.
#![no_main]

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use beyond_ai::__test::key::{Keyring, VirtualKey, mint, mint_v2, mint_v3};
use libfuzzer_sys::fuzz_target;
use std::sync::OnceLock;

const KID: u32 = 1;
const VECTORS: &str = include_str!("../../../../verify/key_vectors.json");

fn signing_key() -> ed25519_dalek::SigningKey {
    ed25519_dalek::SigningKey::from_bytes(&[7u8; 32])
}

/// The ring, built once; the first call also runs the property over the vectors' own tokens.
fn ring() -> &'static Keyring {
    static RING: OnceLock<Keyring> = OnceLock::new();
    RING.get_or_init(|| {
        let ring: Keyring = [(KID, signing_key().verifying_key())].into_iter().collect();
        let doc: serde_json::Value = serde_json::from_str(VECTORS).expect("vectors parse");
        for row in doc["mint"].as_array().expect("mint rows") {
            let token = row["token"].as_str().expect("token");
            assert!(
                ring.verify_at(token, 0).is_ok(),
                "vector {} must verify",
                row["name"]
            );
            assert_eq!(remint(&ring, token, 0).as_deref(), Some(token));
        }
        ring
    })
}

/// The token the ring's signer mints for the identity `token` decodes to; `None` if `token` does
/// not verify at `now`.
fn remint(ring: &Keyring, token: &str, now: u64) -> Option<String> {
    let id = ring.verify_at(token, now).ok()?;
    let mut parts = token.splitn(4, '.');
    let prefix = parts.next()?;
    let kid: u32 = parts.next()?.parse().ok()?;
    let payload = URL_SAFE_NO_PAD.decode(parts.next()?).ok()?;
    let vk = VirtualKey {
        tenant_id: id.tenant_id,
        vpc_id: id.vpc_id,
        key_id: id.key_id,
    };
    let sk = signing_key();
    Some(match prefix {
        "bai_v1" => mint(&vk, kid, &sk),
        "bai_v2" => mint_v2(&vk, id.key_id?, kid, &sk),
        "bai_v3" => {
            let exp = u64::from_le_bytes(payload.get(24..32)?.try_into().ok()?);
            mint_v3(&vk, id.key_id?, exp, kid, &sk)
        }
        _ => return None,
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
    let ring = ring();
    if ring.verify_at(token, now).is_ok() {
        assert_eq!(
            remint(ring, token, now).as_deref(),
            Some(token),
            "the ring accepted a token it would not mint"
        );
    }
});
