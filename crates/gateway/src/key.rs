//! Stateless virtual API key: `bai_v1.{kid}.{payload}.{sig}`.
//!
//! The gateway authenticates every request from a `{payload}` it can verify **without a
//! lookup**: tenant/app identity lives *inside* the token, signed with Ed25519. We hold only
//! the *public* keys (by `kid`), so a compromised — or third-party / OSS — gateway can verify
//! but **cannot mint** new tenant keys; the private signing key lives only in the control plane.
//!
//! Why signed-token instead of opaque-token + registry lookup: at millions of tenants we don't
//! want a per-request lookup (latency + a state dependency) just to learn *who* is calling.
//! Identity is stateless here; the only per-request state is the sparse deny-set (see `deny`),
//! which is a membership check, not an identity lookup.
//!
//! Why deterministic (no nonce/timestamp in the payload): `mint(tenant, app)` is reproducible,
//! so the control plane can re-derive a tenant's key on demand and store nothing. Revocation is
//! handled out-of-band by the deny-set, not by per-key expiry.

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use std::collections::HashMap;

/// Wire prefix + version. Bumping the version is a breaking change to the token format;
/// the version is inside the signed bytes so it cannot be downgraded by an attacker.
pub const PREFIX: &str = "bai_v1";

/// Signing-key identifier. Lets the control plane rotate signing keys: new tokens are minted
/// under a new `kid` while the gateway still trusts the public keys of older, un-retired `kid`s.
pub type Kid = u32;

/// The identity carried by (and the entire contents of) a virtual key.
///
/// `tenant_id`/`vpc_id` are `u64` to match the platform's id width (cf. ClickHouse
/// `tenant_id UInt64` / `vpc_id UInt64`) and to keep the payload a fixed 16 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VirtualKey {
    pub tenant_id: u64,
    pub vpc_id: u64,
}

impl VirtualKey {
    /// Fixed 16-byte little-endian payload: `tenant_id ++ vpc_id`. Fixed layout (not JSON) so
    /// the encoding is deterministic byte-for-byte — required for `mint` to be reproducible.
    fn encode_payload(&self) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[..8].copy_from_slice(&self.tenant_id.to_le_bytes());
        out[8..].copy_from_slice(&self.vpc_id.to_le_bytes());
        out
    }

    fn decode_payload(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != 16 {
            return None;
        }
        Some(Self {
            tenant_id: u64::from_le_bytes(bytes[..8].try_into().ok()?),
            vpc_id: u64::from_le_bytes(bytes[8..].try_into().ok()?),
        })
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum KeyError {
    #[error("malformed virtual key")]
    Malformed,
    #[error("unsupported key version")]
    BadVersion,
    #[error("unknown signing key id {0}")]
    UnknownKid(Kid),
    #[error("signature verification failed")]
    BadSignature,
}

/// The set of trusted Ed25519 public keys, indexed by `kid`. Built once at boot from config
/// (`signing_keys`); multiple kids may be trusted at once for zero-downtime rotation via redeploy.
#[derive(Debug, Default, Clone)]
pub struct Keyring {
    keys: HashMap<Kid, VerifyingKey>,
}

impl Keyring {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, kid: Kid, key: VerifyingKey) {
        self.keys.insert(kid, key);
    }

    pub fn get(&self, kid: Kid) -> Option<&VerifyingKey> {
        self.keys.get(&kid)
    }

    pub fn remove(&mut self, kid: Kid) {
        self.keys.remove(&kid);
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Verify a virtual key string and extract its identity. Stateless: the only input besides
    /// the token is the public keyring.
    pub fn verify(&self, token: &str) -> Result<VirtualKey, KeyError> {
        // Split into exactly 4 parts: `bai_v1`, kid, payload, sig. `splitn(4, '.')` rejects any
        // token with fewer separators; a payload/sig never contains '.' (base64url has none).
        let mut parts = token.splitn(4, '.');
        let prefix = parts.next().ok_or(KeyError::Malformed)?;
        let kid_str = parts.next().ok_or(KeyError::Malformed)?;
        let payload_b64 = parts.next().ok_or(KeyError::Malformed)?;
        let sig_b64 = parts.next().ok_or(KeyError::Malformed)?;

        if prefix != PREFIX {
            // Distinguish "wrong version of our token" from "not our token at all" only loosely;
            // both are unauthenticated. A `bai_vN` with N != 1 reports BadVersion for clarity.
            return if prefix.starts_with("bai_v") {
                Err(KeyError::BadVersion)
            } else {
                Err(KeyError::Malformed)
            };
        }

        let kid: Kid = kid_str.parse().map_err(|_| KeyError::Malformed)?;

        // Decode the fixed-size fields straight onto the stack — no per-request heap allocation on
        // the verify hot path. The payload is always 16 bytes, the signature 64. `decode_slice`
        // sizes its bounds check against a (ceil) estimate, so the buffers are a few bytes larger
        // than the exact decoded length; we slice to what was actually written and the fixed-size
        // checks below reject anything off (an oversized field overruns the estimate → Malformed).
        let mut payload_buf = [0u8; 24]; // ≥ estimate for a 22-char (16-byte) payload
        let plen = URL_SAFE_NO_PAD
            .decode_slice(payload_b64, &mut payload_buf)
            .map_err(|_| KeyError::Malformed)?;
        let payload = &payload_buf[..plen];

        let mut sig_buf = [0u8; 72]; // ≥ estimate for an 86-char (64-byte) signature
        let slen = URL_SAFE_NO_PAD
            .decode_slice(sig_b64, &mut sig_buf)
            .map_err(|_| KeyError::Malformed)?;
        let sig_arr: [u8; 64] = sig_buf[..slen]
            .try_into()
            .map_err(|_| KeyError::Malformed)?;
        let signature = Signature::from_bytes(&sig_arr);

        // Resolve the public key *before* the cryptographic check so an unknown kid is a distinct,
        // cheap rejection (no signature math on keys we don't trust).
        let vk = self.get(kid).ok_or(KeyError::UnknownKid(kid))?;

        // The signed message binds version + kid + payload, so none can be swapped independently.
        // Build it into a stack buffer (≤ 40 bytes) — no allocation per verify. A payload longer
        // than the buffer can hold can't be a valid 16-byte payload anyway, so it's `Malformed`
        // rather than a panic on this per-request hot path.
        let mut signed_buf = [0u8; SIGNED_BYTES_CAP];
        let signed =
            write_signed_bytes(&mut signed_buf, kid, payload_b64).ok_or(KeyError::Malformed)?;
        vk.verify(signed, &signature)
            .map_err(|_| KeyError::BadSignature)?;

        VirtualKey::decode_payload(payload).ok_or(KeyError::Malformed)
    }
}

/// Upper bound on `bai_v1.{kid}.{payload}`: `PREFIX` (6) + `.` + a `u32` kid (≤ 10 digits) + `.`
/// + a 16-byte base64url payload (22 chars) = 40 bytes. 64 leaves headroom.
const SIGNED_BYTES_CAP: usize = 64;

/// Write the signature-covered bytes `bai_v1.{kid}.{payload}` into `buf`, returning the written
/// slice — or `None` if they don't fit in `SIGNED_BYTES_CAP`. Binding kid + payload here is what
/// stops an attacker from re-pointing a valid signature at a different kid or a tampered payload.
/// For a well-formed key the length is bounded (≤ 40 bytes; see `SIGNED_BYTES_CAP`), so `None`
/// means the input was malformed — `write!` returns `WriteZero` rather than panicking or
/// truncating, keeping the verify hot path allocation- *and* panic-free.
fn write_signed_bytes<'a>(
    buf: &'a mut [u8; SIGNED_BYTES_CAP],
    kid: Kid,
    payload_b64: &str,
) -> Option<&'a [u8]> {
    use std::io::Write;
    let mut cur = std::io::Cursor::new(&mut buf[..]);
    write!(cur, "{PREFIX}.{kid}.{payload_b64}").ok()?;
    let n = cur.position() as usize;
    Some(&buf[..n])
}

/// Parse an Ed25519 public key from a slipstream `signkey.*` value: accept raw 32 bytes or
/// base64 (standard or url-safe) of 32 bytes, so the control plane can store whichever form.
pub fn verifying_key_from_value(bytes: &[u8]) -> Option<VerifyingKey> {
    if let Ok(arr) = <[u8; 32]>::try_from(bytes) {
        return VerifyingKey::from_bytes(&arr).ok();
    }
    let s = std::str::from_utf8(bytes).ok()?.trim();
    for decoded in [
        base64::engine::general_purpose::STANDARD.decode(s).ok(),
        URL_SAFE_NO_PAD.decode(s).ok(),
    ]
    .into_iter()
    .flatten()
    {
        if let Ok(arr) = <[u8; 32]>::try_from(decoded.as_slice()) {
            return VerifyingKey::from_bytes(&arr).ok();
        }
    }
    None
}

/// Mint a virtual key. Lives here for tests + determinism checks and as the reference
/// implementation; production minting is the Go control plane (`crypto/ed25519`), which must
/// produce byte-identical output for the same inputs.
pub fn mint(vk: &VirtualKey, kid: Kid, signing_key: &SigningKey) -> String {
    let payload_b64 = URL_SAFE_NO_PAD.encode(vk.encode_payload());
    let mut signed_buf = [0u8; SIGNED_BYTES_CAP];
    // mint builds the payload itself (a fixed 22-char base64 of 16 bytes) from controlled inputs,
    // so it always fits; this `expect` is a true invariant assertion, not a fallible runtime path.
    let signed = write_signed_bytes(&mut signed_buf, kid, &payload_b64)
        .expect("minted signed bytes fit in SIGNED_BYTES_CAP");
    let sig: Signature = signing_key.sign(signed);
    let sig_b64 = URL_SAFE_NO_PAD.encode(sig.to_bytes());
    format!("{PREFIX}.{kid}.{payload_b64}.{sig_b64}")
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic test keypair from a fixed seed — avoids an RNG dep and keeps tests reproducible.
    fn test_keypair(seed: u8) -> (SigningKey, VerifyingKey) {
        let sk = SigningKey::from_bytes(&[seed; 32]);
        let vk = sk.verifying_key();
        (sk, vk)
    }

    fn ring_with(kid: Kid, vk: VerifyingKey) -> Keyring {
        let mut r = Keyring::new();
        r.insert(kid, vk);
        r
    }

    #[test]
    fn mint_then_verify_roundtrips_identity() {
        let (sk, vk) = test_keypair(1);
        let ring = ring_with(7, vk);
        let id = VirtualKey {
            tenant_id: 42,
            vpc_id: 99,
        };

        let token = mint(&id, 7, &sk);
        assert_eq!(ring.verify(&token).unwrap(), id);
    }

    #[test]
    fn mint_is_deterministic() {
        let (sk, _) = test_keypair(2);
        let id = VirtualKey {
            tenant_id: 1,
            vpc_id: 2,
        };
        // Ed25519 is deterministic (RFC 8032) and the payload has no nonce, so two mints match.
        assert_eq!(mint(&id, 1, &sk), mint(&id, 1, &sk));
    }

    #[test]
    fn tampered_payload_is_rejected() {
        let (sk, vk) = test_keypair(3);
        let ring = ring_with(1, vk);
        let token = mint(
            &VirtualKey {
                tenant_id: 10,
                vpc_id: 20,
            },
            1,
            &sk,
        );

        // Flip a byte in the payload segment; the signature no longer covers it.
        let mut parts: Vec<&str> = token.split('.').collect();
        let mut payload = URL_SAFE_NO_PAD.decode(parts[2]).unwrap();
        payload[0] ^= 0xff;
        let tampered_payload = URL_SAFE_NO_PAD.encode(&payload);
        parts[2] = &tampered_payload;
        let tampered = parts.join(".");

        assert_eq!(ring.verify(&tampered), Err(KeyError::BadSignature));
    }

    #[test]
    fn tampered_signature_is_rejected() {
        let (sk, vk) = test_keypair(4);
        let ring = ring_with(1, vk);
        let token = mint(
            &VirtualKey {
                tenant_id: 5,
                vpc_id: 6,
            },
            1,
            &sk,
        );

        let mut sig = URL_SAFE_NO_PAD
            .decode(token.rsplit('.').next().unwrap())
            .unwrap();
        sig[0] ^= 0xff;
        let bad_sig = URL_SAFE_NO_PAD.encode(&sig);
        let base = &token[..token.rfind('.').unwrap()];
        let tampered = format!("{base}.{bad_sig}");

        assert_eq!(ring.verify(&tampered), Err(KeyError::BadSignature));
    }

    #[test]
    fn unknown_kid_is_rejected_without_crypto() {
        let (sk, vk) = test_keypair(5);
        let ring = ring_with(1, vk); // trusts kid=1 only
        let token = mint(
            &VirtualKey {
                tenant_id: 1,
                vpc_id: 1,
            },
            2,
            &sk,
        ); // minted under kid=2
        assert_eq!(ring.verify(&token), Err(KeyError::UnknownKid(2)));
    }

    #[test]
    fn signature_from_a_different_kid_is_rejected() {
        // A valid signature minted under kid=2 must not verify when presented as kid=1, even if
        // the gateway trusts both — because kid is part of the signed bytes.
        let (sk1, vk1) = test_keypair(6);
        let (sk2, vk2) = test_keypair(7);
        let mut ring = Keyring::new();
        ring.insert(1, vk1);
        ring.insert(2, vk2);

        let id = VirtualKey {
            tenant_id: 3,
            vpc_id: 4,
        };
        let token2 = mint(&id, 2, &sk2);
        // Re-label the kid segment as 1 while keeping kid=2's signature.
        let parts: Vec<&str> = token2.split('.').collect();
        let relabeled = format!("{}.1.{}.{}", parts[0], parts[2], parts[3]);
        assert_eq!(ring.verify(&relabeled), Err(KeyError::BadSignature));
        let _ = sk1;
    }

    #[test]
    fn malformed_and_version_errors() {
        let (_, vk) = test_keypair(8);
        let ring = ring_with(1, vk);
        assert_eq!(ring.verify("garbage"), Err(KeyError::Malformed));
        assert_eq!(ring.verify("bai_v1.1.only-three"), Err(KeyError::Malformed));
        assert_eq!(ring.verify("bai_v2.1.aaaa.bbbb"), Err(KeyError::BadVersion));
        assert_eq!(
            ring.verify("sk-openai.1.aaaa.bbbb"),
            Err(KeyError::Malformed)
        );
    }
}
