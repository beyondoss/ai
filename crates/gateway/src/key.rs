//! Stateless virtual API key: `bai_v1` / `bai_v2`.
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
//! `bai_v1` is tenant+vpc only and `mint` is deterministic for those two fields — the control
//! plane can re-derive one key per (tenant, vpc). That cannot name a *credential*, so it cannot
//! be cut off without cutting off the tenant. `bai_v2` adds an explicit `key_id` the caller
//! supplies (not derived from tenant+vpc); mint is still deterministic for a given key_id.
//! Revocation is out-of-band via the deny-set (`blackhole.{tenant}` and `blackhole.key.{id}`).

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use std::collections::HashMap;

/// Wire prefix + version. Inside the signed bytes, so it cannot be downgraded by an attacker.
///
/// `request_filter` fail-closes on a managed prefix: a match that then fails verify is 401, never BYO.
pub const PREFIX_V1: &str = "bai_v1";
pub const PREFIX_V2: &str = "bai_v2";
/// Alias for the original version — existing call sites and benches.
pub const PREFIX: &str = PREFIX_V1;

/// Whether a presented credential claims to be a managed virtual key (`bai_v1…` or `bai_v2…`).
///
/// Used by the identity branch and the rate-guard managed flag — they must agree. A prefix match
/// is fail-closed at verify (401), so classifying it as managed here also exempts a forged flood
/// from the BYO aggregate without ever forwarding it upstream.
#[inline]
pub fn is_managed_prefix(token: &str) -> bool {
    token.starts_with(PREFIX_V1) || token.starts_with(PREFIX_V2)
}

/// Signing-key identifier. Lets the control plane rotate signing keys: new tokens are minted
/// under a new `kid` while the gateway still trusts the public keys of older, un-retired `kid`s.
pub type Kid = u32;

/// The identity carried by a virtual key.
///
/// `tenant_id`/`vpc_id`/`key_id` are `u64` to match the platform's id width (cf. ClickHouse
/// `tenant_id UInt64` / `vpc_id UInt64`). `key_id` is `Some` on `bai_v2` only — v1 has no
/// per-credential identity, so a key-level deny cannot apply to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VirtualKey {
    pub tenant_id: u64,
    pub vpc_id: u64,
    pub key_id: Option<u64>,
}

impl VirtualKey {
    /// v1: fixed 16-byte little-endian payload `tenant_id ++ vpc_id`.
    fn encode_payload_v1(&self) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[..8].copy_from_slice(&self.tenant_id.to_le_bytes());
        out[8..].copy_from_slice(&self.vpc_id.to_le_bytes());
        out
    }

    /// v2: fixed 24-byte little-endian payload `tenant_id ++ vpc_id ++ key_id`.
    /// `key_id` is an argument, not derived from tenant+vpc — two credentials for the
    /// same tenant are distinct tokens and can be denied independently.
    fn encode_payload_v2(&self, key_id: u64) -> [u8; 24] {
        let mut out = [0u8; 24];
        out[..8].copy_from_slice(&self.tenant_id.to_le_bytes());
        out[8..16].copy_from_slice(&self.vpc_id.to_le_bytes());
        out[16..].copy_from_slice(&key_id.to_le_bytes());
        out
    }

    fn decode_payload_v1(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != 16 {
            return None;
        }
        Some(Self {
            tenant_id: u64::from_le_bytes(bytes[..8].try_into().ok()?),
            vpc_id: u64::from_le_bytes(bytes[8..].try_into().ok()?),
            key_id: None,
        })
    }

    fn decode_payload_v2(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != 24 {
            return None;
        }
        Some(Self {
            tenant_id: u64::from_le_bytes(bytes[..8].try_into().ok()?),
            vpc_id: u64::from_le_bytes(bytes[8..16].try_into().ok()?),
            key_id: Some(u64::from_le_bytes(bytes[16..].try_into().ok()?)),
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
    /// the token is the public keyring. Accepts `bai_v1` and `bai_v2`.
    pub fn verify(&self, token: &str) -> Result<VirtualKey, KeyError> {
        // Split into exactly 4 parts: prefix, kid, payload, sig. `splitn(4, '.')` rejects any
        // token with fewer separators; a payload/sig never contains '.' (base64url has none).
        let mut parts = token.splitn(4, '.');
        let prefix = parts.next().ok_or(KeyError::Malformed)?;
        let kid_str = parts.next().ok_or(KeyError::Malformed)?;
        let payload_b64 = parts.next().ok_or(KeyError::Malformed)?;
        let sig_b64 = parts.next().ok_or(KeyError::Malformed)?;

        let version = match prefix {
            PREFIX_V1 => 1u8,
            PREFIX_V2 => 2,
            p if p.starts_with("bai_v") => return Err(KeyError::BadVersion),
            _ => return Err(KeyError::Malformed),
        };

        let kid: Kid = kid_str.parse().map_err(|_| KeyError::Malformed)?;

        // Decode the fixed-size fields straight onto the stack — no per-request heap allocation on
        // the verify hot path. v1 payload is 16 bytes (22-char b64), v2 is 24 bytes (32-char b64);
        // the signature is 64. `decode_slice` sizes its bounds check against a (ceil) estimate, so
        // the buffers are a few bytes larger than the exact decoded length.
        let mut payload_buf = [0u8; 36];
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

        // The signed message binds version + kid + payload, so none can be swapped independently
        // (including a v1↔v2 downgrade). Stack buffer, no allocation per verify.
        let mut signed_buf = [0u8; SIGNED_BYTES_CAP];
        let signed = write_signed_bytes(&mut signed_buf, prefix, kid, payload_b64)
            .ok_or(KeyError::Malformed)?;
        vk.verify(signed, &signature)
            .map_err(|_| KeyError::BadSignature)?;

        match version {
            1 => VirtualKey::decode_payload_v1(payload).ok_or(KeyError::Malformed),
            2 => VirtualKey::decode_payload_v2(payload).ok_or(KeyError::Malformed),
            _ => Err(KeyError::BadVersion),
        }
    }
}

/// Upper bound on `{prefix}.{kid}.{payload}`: prefix (6) + `.` + a `u32` kid (≤ 10 digits) + `.`
/// + a 24-byte base64url payload (32 chars, v2) = 50 bytes. 64 leaves headroom. v1 is 40.
const SIGNED_BYTES_CAP: usize = 64;

/// Write the signature-covered bytes `{prefix}.{kid}.{payload}` into `buf`, returning the written
/// slice — or `None` if they don't fit in `SIGNED_BYTES_CAP`. Binding version + kid + payload here
/// is what stops an attacker from re-pointing a valid signature at a different kid, a tampered
/// payload, or the other version. For a well-formed key the length is bounded (≤ 50 bytes), so
/// `None` means the input was malformed — `write!` returns `WriteZero` rather than panicking or
/// truncating, keeping the verify hot path allocation- *and* panic-free.
fn write_signed_bytes<'a>(
    buf: &'a mut [u8; SIGNED_BYTES_CAP],
    prefix: &str,
    kid: Kid,
    payload_b64: &str,
) -> Option<&'a [u8]> {
    use std::io::Write;
    let mut cur = std::io::Cursor::new(&mut buf[..]);
    write!(cur, "{prefix}.{kid}.{payload_b64}").ok()?;
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
    // Standard first, url-safe only if that didn't yield a key. Chained rather than collected into an
    // array of both candidates: an array is built *before* it's iterated, so the url-safe decode —
    // and its allocation — ran even when the standard one had already succeeded.
    let key32 = |decoded: Vec<u8>| <[u8; 32]>::try_from(decoded.as_slice()).ok();
    let arr = base64::engine::general_purpose::STANDARD
        .decode(s)
        .ok()
        .and_then(key32)
        .or_else(|| URL_SAFE_NO_PAD.decode(s).ok().and_then(key32))?;
    VerifyingKey::from_bytes(&arr).ok()
}

/// Mint a `bai_v1` virtual key (tenant+vpc only). Lives here for tests + determinism checks
/// and as the reference implementation; production minting is the Go control plane
/// (`crypto/ed25519`), which must produce byte-identical output for the same inputs.
#[allow(clippy::expect_used)] // payload is a fixed 22-char base64 of 16 bytes; always fits the cap
pub fn mint(vk: &VirtualKey, kid: Kid, signing_key: &SigningKey) -> String {
    let payload_b64 = URL_SAFE_NO_PAD.encode(vk.encode_payload_v1());
    let mut signed_buf = [0u8; SIGNED_BYTES_CAP];
    let signed = write_signed_bytes(&mut signed_buf, PREFIX_V1, kid, &payload_b64)
        .expect("minted signed bytes fit in SIGNED_BYTES_CAP");
    let sig: Signature = signing_key.sign(signed);
    let sig_b64 = URL_SAFE_NO_PAD.encode(sig.to_bytes());
    format!("{PREFIX_V1}.{kid}.{payload_b64}.{sig_b64}")
}

/// Mint a `bai_v2` virtual key. `key_id` is explicit — not derived from tenant+vpc — so two
/// credentials for the same tenant are distinct and can be denied independently.
///
/// Byte layout (document this for the Go control plane; do not invent a second encoding):
/// `bai_v2.{kid}.{payload_b64}.{sig_b64}` where `payload` is 24 bytes little-endian
/// `tenant_id u64 || vpc_id u64 || key_id u64`, `payload_b64`/`sig_b64` are base64url (no pad),
/// and the signed bytes are `bai_v2.{kid}.{payload_b64}`. Same Ed25519 keyring as v1.
#[allow(clippy::expect_used)] // payload is a fixed 32-char base64 of 24 bytes; always fits the cap
pub fn mint_v2(vk: &VirtualKey, key_id: u64, kid: Kid, signing_key: &SigningKey) -> String {
    let payload_b64 = URL_SAFE_NO_PAD.encode(vk.encode_payload_v2(key_id));
    let mut signed_buf = [0u8; SIGNED_BYTES_CAP];
    let signed = write_signed_bytes(&mut signed_buf, PREFIX_V2, kid, &payload_b64)
        .expect("minted v2 signed bytes fit in SIGNED_BYTES_CAP");
    let sig: Signature = signing_key.sign(signed);
    let sig_b64 = URL_SAFE_NO_PAD.encode(sig.to_bytes());
    format!("{PREFIX_V2}.{kid}.{payload_b64}.{sig_b64}")
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
            key_id: None,
        };

        let token = mint(&id, 7, &sk);
        assert_eq!(ring.verify(&token).unwrap(), id);
        assert!(token.starts_with("bai_v1."));
    }

    #[test]
    fn mint_v2_roundtrips_explicit_key_id() {
        let (sk, vk) = test_keypair(1);
        let ring = ring_with(7, vk);
        let id = VirtualKey {
            tenant_id: 42,
            vpc_id: 99,
            key_id: None,
        };

        let token = mint_v2(&id, 1_000_042, 7, &sk);
        let got = ring.verify(&token).unwrap();
        assert_eq!(got.tenant_id, 42);
        assert_eq!(got.vpc_id, 99);
        assert_eq!(got.key_id, Some(1_000_042));
        assert!(token.starts_with("bai_v2."));
        // Same tenant+vpc, different key_id → a different credential.
        let other = mint_v2(&id, 1_000_043, 7, &sk);
        assert_ne!(token, other);
        assert_eq!(ring.verify(&other).unwrap().key_id, Some(1_000_043));
        // key_id is not derived from tenant+vpc: mint is deterministic for the explicit id.
        assert_eq!(token, mint_v2(&id, 1_000_042, 7, &sk));
    }

    #[test]
    fn mint_is_deterministic() {
        let (sk, _) = test_keypair(2);
        let id = VirtualKey {
            tenant_id: 1,
            vpc_id: 2,
            key_id: None,
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
                key_id: None,
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
                key_id: None,
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
                key_id: None,
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
            key_id: None,
        };
        let token2 = mint(&id, 2, &sk2);
        // Re-label the kid segment as 1 while keeping kid=2's signature.
        let parts: Vec<&str> = token2.split('.').collect();
        let relabeled = format!("{}.1.{}.{}", parts[0], parts[2], parts[3]);
        assert_eq!(ring.verify(&relabeled), Err(KeyError::BadSignature));
        let _ = sk1;
    }

    #[test]
    fn verifying_key_accepts_every_stored_encoding() {
        // The three shapes the control plane may store a `signkey.*` value in. A rejection here is a
        // silently empty keyring — every managed token 401s — so pin all of them, plus the negatives.
        let (_, vk) = test_keypair(9);
        let raw = vk.to_bytes();
        let std_b64 = base64::engine::general_purpose::STANDARD.encode(raw);
        let url_b64 = URL_SAFE_NO_PAD.encode(raw);

        assert_eq!(verifying_key_from_value(&raw), Some(vk));
        assert_eq!(verifying_key_from_value(std_b64.as_bytes()), Some(vk));
        assert_eq!(verifying_key_from_value(url_b64.as_bytes()), Some(vk));
        // Surrounding whitespace (a trailing newline from a file or a KV write) is trimmed.
        assert_eq!(
            verifying_key_from_value(format!(" {url_b64}\n").as_bytes()),
            Some(vk)
        );

        assert_eq!(verifying_key_from_value(b"not base64 at all"), None);
        // Decodes fine, but isn't 32 bytes — not a key.
        let short = base64::engine::general_purpose::STANDARD.encode([0u8; 16]);
        assert_eq!(verifying_key_from_value(short.as_bytes()), None);
    }

    #[test]
    fn malformed_and_version_errors() {
        let (_, vk) = test_keypair(8);
        let ring = ring_with(1, vk);
        assert_eq!(ring.verify("garbage"), Err(KeyError::Malformed));
        assert_eq!(ring.verify("bai_v1.1.only-three"), Err(KeyError::Malformed));
        assert_eq!(ring.verify("bai_v3.1.aaaa.bbbb"), Err(KeyError::BadVersion));
        // A v2-shaped token with junk payload/sig is this version, just malformed — not BadVersion.
        assert_eq!(ring.verify("bai_v2.1.aaaa.bbbb"), Err(KeyError::Malformed));
        assert_eq!(
            ring.verify("sk-openai.1.aaaa.bbbb"),
            Err(KeyError::Malformed)
        );
    }

    #[test]
    fn managed_prefix_is_v1_or_v2() {
        assert!(is_managed_prefix("bai_v1.1.payload.sig"));
        assert!(is_managed_prefix("bai_v2.1.payload.sig"));
        // Unknown versions stay off the managed branch (verify would BadVersion if we entered).
        assert!(!is_managed_prefix("bai_v3.1.aaaa.bbbb"));
        assert!(!is_managed_prefix("bai_"));
        assert!(!is_managed_prefix("sk-openai"));
    }
}
