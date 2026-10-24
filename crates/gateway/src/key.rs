//! Stateless virtual API key: `bai_v1` / `bai_v2` / `bai_v3`.
//!
//! The gateway authenticates every request from a `{payload}` it can verify **without a
//! lookup**: tenant/app identity lives *inside* the token, signed with Ed25519. We hold only
//! the *public* keys (by `kid`), so a compromised — or third-party / OSS — gateway can verify
//! but **cannot mint** new tenant keys; the private signing key lives only in the control plane.
//!
//! Why signed-token instead of opaque-token + registry lookup: at millions of tenants we don't
//! want a per-request lookup (latency + a state dependency) just to learn *who* is calling.
//! Identity is stateless here; the only per-request state is the sparse deny-set (see `deny`) and
//! the sparse allowance-set (see `allowance`) — membership checks, not an identity lookup.
//!
//! `bai_v1` is tenant+vpc only and `mint` is deterministic for those two fields — the control
//! plane can re-derive one key per (tenant, vpc). That cannot name a *credential*, so it cannot
//! be cut off without cutting off the tenant. `bai_v2` adds an explicit `key_id` the caller
//! supplies (not derived from tenant+vpc); mint is still deterministic for a given key_id.
//! Revocation is out-of-band via the deny-set (`blackhole.{tenant}` and `blackhole.key.{id}`).
//! Quota exhaust is the same shape on the allowance-set (`allowance.{tenant}` / `allowance.key.{id}`).
//!
//! `bai_v3` is `bai_v2` plus a mandatory `exp` (unix seconds): a short-lived credential the control
//! plane mints in exchange for a workload identity. Long-lived keys stay v2. `exp` is checked at
//! admission only — a stream already admitted is not cut when it passes, the same rule as a deny.
//!
//! The trusted public keys are the config ring (this module's [`Keyring`]) plus the live ring the
//! control plane publishes at runtime (`live_keys`); a `kid` named in the deny-set
//! (`blackhole.kid.{kid}`) is distrusted whichever source holds it, before any signature math.

use crate::deny::DenySet;
use crate::live_keys::LiveKeys;
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use ed25519_dalek::{Signature, VerifyingKey};
#[cfg(any(test, feature = "mint"))]
use ed25519_dalek::{Signer, SigningKey};
use std::collections::HashMap;

/// Wire prefix + version. Inside the signed bytes, so it cannot be downgraded by an attacker.
///
/// `request_filter` fail-closes on a managed prefix: a match that then fails verify is 401, never BYO.
pub const PREFIX_V1: &str = "bai_v1";
pub const PREFIX_V2: &str = "bai_v2";
pub const PREFIX_V3: &str = "bai_v3";

/// Clock skew a `bai_v3` key is forgiven past its `exp`: a key is expired when `now > exp + LEEWAY_SECS`.
/// Covers drift between the control plane that minted `exp` and the gateway replica reading it.
pub const LEEWAY_SECS: u64 = 30;

/// Whether a presented credential claims to be a managed virtual key (`bai_v1…`, `bai_v2…` or `bai_v3…`).
///
/// Used by the identity branch and the rate-guard managed flag — they must agree. A prefix match
/// is fail-closed at verify (401), so classifying it as managed here also exempts a forged flood
/// from the BYO aggregate without ever forwarding it upstream.
#[inline]
pub fn is_managed_prefix(token: &str) -> bool {
    token.starts_with(PREFIX_V1) || token.starts_with(PREFIX_V2) || token.starts_with(PREFIX_V3)
}

/// Signing-key identifier. Lets the control plane rotate signing keys: new tokens are minted
/// under a new `kid` while the gateway still trusts the public keys of older, un-retired `kid`s.
pub type Kid = u32;

/// The identity carried by a virtual key.
///
/// `tenant_id`/`vpc_id`/`key_id` are `u64` to match the platform's id width (cf. ClickHouse
/// `tenant_id UInt64` / `vpc_id UInt64`). `key_id` is `Some` on `bai_v2` and `bai_v3` — v1 has no
/// per-credential identity, so a key-level deny cannot apply to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VirtualKey {
    pub tenant_id: u64,
    pub vpc_id: u64,
    pub key_id: Option<u64>,
}

impl VirtualKey {
    /// v1: fixed 16-byte little-endian payload `tenant_id ++ vpc_id`.
    #[cfg(any(test, feature = "mint"))]
    fn encode_payload_v1(&self) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[..8].copy_from_slice(&self.tenant_id.to_le_bytes());
        out[8..].copy_from_slice(&self.vpc_id.to_le_bytes());
        out
    }

    /// v2: fixed 24-byte little-endian payload `tenant_id ++ vpc_id ++ key_id`.
    /// `key_id` is an argument, not derived from tenant+vpc — two credentials for the
    /// same tenant are distinct tokens and can be denied independently.
    #[cfg(any(test, feature = "mint"))]
    fn encode_payload_v2(&self, key_id: u64) -> [u8; 24] {
        let mut out = [0u8; 24];
        out[..8].copy_from_slice(&self.tenant_id.to_le_bytes());
        out[8..16].copy_from_slice(&self.vpc_id.to_le_bytes());
        out[16..].copy_from_slice(&key_id.to_le_bytes());
        out
    }

    /// v3: fixed 32-byte little-endian payload `tenant_id ++ vpc_id ++ key_id ++ exp`, `exp` in
    /// unix seconds. Mandatory — a v3 key with no expiry is just a v2 key.
    #[cfg(any(test, feature = "mint"))]
    fn encode_payload_v3(&self, key_id: u64, exp: u64) -> [u8; 32] {
        let mut out = [0u8; 32];
        out[..24].copy_from_slice(&self.encode_payload_v2(key_id));
        out[24..].copy_from_slice(&exp.to_le_bytes());
        out
    }

    /// The identity and `exp` of a v3 payload: the v2 fields, then `exp`.
    fn decode_payload_v3(bytes: &[u8]) -> Option<(Self, u64)> {
        if bytes.len() != 32 {
            return None;
        }
        let vk = Self::decode_payload_v2(&bytes[..24])?;
        Some((vk, u64::from_le_bytes(bytes[24..].try_into().ok()?)))
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

/// A [`VirtualKey`] whose signature [`Keyring::verify`] checked: the only way to make one, so a
/// tenant id held as an `Identity` was proven, never parsed from something a client sent. Reads as
/// the claims it carries (`identity.tenant_id`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Identity(VirtualKey);

impl std::ops::Deref for Identity {
    type Target = VirtualKey;
    fn deref(&self) -> &VirtualKey {
        &self.0
    }
}

impl PartialEq<VirtualKey> for Identity {
    fn eq(&self, other: &VirtualKey) -> bool {
        self.0 == *other
    }
}

impl Identity {
    /// An identity no key proved, for a unit test that needs a managed request without minting one.
    #[cfg(test)]
    pub(crate) fn unverified_for_test(tenant_id: u64, vpc_id: u64, key_id: Option<u64>) -> Self {
        Self(VirtualKey {
            tenant_id,
            vpc_id,
            key_id,
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
    /// The `kid` is in the deny-set (`blackhole.kid.{kid}`): distrusted, whichever source holds it.
    #[error("signing key id {0} is distrusted")]
    Distrusted(Kid),
    /// A `bai_v3` key whose signature verified but whose `exp` has passed (past [`LEEWAY_SECS`]).
    /// Only ever returned for an authentic token, so it tells a forger nothing.
    #[error("key expired")]
    Expired,
}

/// The Ed25519 public keys trusted from config, indexed by `kid`. Built once at boot from
/// `signing_keys`; multiple kids may be trusted at once for zero-downtime rotation via redeploy.
/// Keys the control plane adds at runtime live in [`LiveKeys`] and are consulted after these.
#[derive(Debug, Default, Clone)]
pub struct Keyring {
    keys: HashMap<Kid, VerifyingKey>,
}

/// A whole ring from its `(kid, key)` pairs: how a caller outside the crate (the dev-key example,
/// the benches) builds one, since it cannot add to one.
impl FromIterator<(Kid, VerifyingKey)> for Keyring {
    fn from_iter<I: IntoIterator<Item = (Kid, VerifyingKey)>>(pairs: I) -> Self {
        Self {
            keys: pairs.into_iter().collect(),
        }
    }
}

impl Keyring {
    pub fn new() -> Self {
        Self::default()
    }

    /// Crate-private: the trust store is built from config (`AiConfig::build_keyring`), never
    /// changed by a caller holding one. Outside the crate a ring is collected whole (`FromIterator`).
    pub(crate) fn insert(&mut self, kid: Kid, key: VerifyingKey) {
        self.keys.insert(kid, key);
    }

    pub fn get(&self, kid: Kid) -> Option<&VerifyingKey> {
        self.keys.get(&kid)
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Verify a virtual key against this ring alone, on the system clock. Accepts `bai_v1`,
    /// `bai_v2` and `bai_v3`. The gateway itself calls [`Self::verify_with`], which also consults
    /// the live ring and the deny-set; this is the config-only form for callers holding just a ring.
    pub fn verify(&self, token: &str) -> Result<Identity, KeyError> {
        self.verify_with(token, &LiveKeys::default(), &DenySet::default(), unix_now)
    }

    /// [`Self::verify`] with the clock stated: unix seconds `now`. How the contract vectors and
    /// any caller that owns its time check a `bai_v3` key.
    pub fn verify_at(&self, token: &str, now: u64) -> Result<Identity, KeyError> {
        self.verify_with(token, &LiveKeys::default(), &DenySet::default(), || {
            Some(now)
        })
    }

    /// Verify a virtual key string and extract its identity. Stateless: the inputs besides the
    /// token are the public keys (this ring, then `live`), the set of distrusted kids in `deny`,
    /// and — for `bai_v3` only — `now`, unix seconds, or `None` when the clock cannot be read (a v3 key
    /// is then refused as expired: a clock that says nothing must not make every `exp` pass). `now` is a parameter, not a read of the
    /// clock, so a test sets the time without a seam in the release binary; it is called at most
    /// once, and only for a v3 token that already verified.
    ///
    /// Order is cheapest refusal first: shape, distrusted kid, unknown kid, signature, and only
    /// then `exp` — an expiry verdict is never given for a token the signature has not vouched for.
    pub fn verify_with(
        &self,
        token: &str,
        live: &LiveKeys,
        deny: &DenySet,
        now: impl FnOnce() -> Option<u64>,
    ) -> Result<Identity, KeyError> {
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
            PREFIX_V3 => 3,
            p if p.starts_with("bai_v") => return Err(KeyError::BadVersion),
            _ => return Err(KeyError::Malformed),
        };

        // Canonical decimal only: `u32::from_str` also accepts `+1` and `01`, and the signed bytes
        // are rebuilt from the *parsed* kid, so every such spelling would verify as the same key
        // while the rate guard (keyed on the raw token) gave each its own bucket.
        if !is_canonical_kid(kid_str) {
            return Err(KeyError::Malformed);
        }
        let kid: Kid = kid_str.parse().map_err(|_| KeyError::Malformed)?;

        // Decode the fixed-size fields straight onto the stack — no per-request heap allocation on
        // the verify hot path. v1 payload is 16 bytes (22-char b64), v2 is 24 (32-char), v3 is 32
        // (43-char); the signature is 64. `decode_slice` sizes its bounds check against a (ceil)
        // estimate, so the buffers are a few bytes larger than the exact decoded length.
        let mut payload_buf = [0u8; 40];
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

        // Distrust beats both key sources and costs no signature math: a revoked kid is refused
        // before we even look for its key.
        if deny.kid_denied(kid) {
            return Err(KeyError::Distrusted(kid));
        }
        // Resolve the public key *before* the cryptographic check so an unknown kid is a distinct,
        // cheap rejection (no signature math on keys we don't trust). Config first, then the live
        // ring the control plane feeds.
        let vk = self
            .get(kid)
            .or_else(|| live.get(kid))
            .ok_or(KeyError::UnknownKid(kid))?;

        // The signed message binds version + kid + payload, so none can be swapped independently
        // (including a v1↔v2↔v3 relabel). Stack buffer, no allocation per verify. `verify_strict`,
        // not `verify`: it also refuses a small-order public key or `R`, which the lax check admits.
        let mut signed_buf = [0u8; SIGNED_BYTES_CAP];
        let signed = write_signed_bytes(&mut signed_buf, prefix, kid, payload_b64)
            .ok_or(KeyError::Malformed)?;
        vk.verify_strict(signed, &signature)
            .map_err(|_| KeyError::BadSignature)?;

        match version {
            1 => VirtualKey::decode_payload_v1(payload)
                .map(Identity)
                .ok_or(KeyError::Malformed),
            2 => VirtualKey::decode_payload_v2(payload)
                .map(Identity)
                .ok_or(KeyError::Malformed),
            3 => {
                let (vk, exp) =
                    VirtualKey::decode_payload_v3(payload).ok_or(KeyError::Malformed)?;
                // `exp + LEEWAY_SECS` saturates: an `exp` of `u64::MAX` is "never", not a wrap to 29.
                if now().is_none_or(|now| now > exp.saturating_add(LEEWAY_SECS)) {
                    return Err(KeyError::Expired);
                }
                Ok(Identity(vk))
            }
            _ => Err(KeyError::BadVersion),
        }
    }
}

/// Unix seconds now, or `None` when the clock reads before the epoch. Not 0: a 0 would pass every
/// `bai_v3` `exp`, so [`Keyring::verify_with`] refuses a v3 key on `None`.
pub fn unix_now() -> Option<u64> {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_secs())
}

/// Whether `s` is the one decimal spelling `mint` writes for a kid: ASCII digits, no sign, and no
/// leading zero (except `0` itself).
pub(crate) fn is_canonical_kid(s: &str) -> bool {
    let b = s.as_bytes();
    !b.is_empty() && b.iter().all(u8::is_ascii_digit) && (b.len() == 1 || b[0] != b'0')
}

/// Upper bound on `{prefix}.{kid}.{payload}`: prefix (6) + `.` + a `u32` kid (≤ 10 digits) + `.`
/// + a 32-byte base64url payload (43 chars, v3) = 61 bytes. 64 leaves headroom. v2 is 50, v1 40.
const SIGNED_BYTES_CAP: usize = 64;

/// Write the signature-covered bytes `{prefix}.{kid}.{payload}` into `buf`, returning the written
/// slice — or `None` if they don't fit in `SIGNED_BYTES_CAP`. Binding version + kid + payload here
/// is what stops an attacker from re-pointing a valid signature at a different kid, a tampered
/// payload, or the other version. For a well-formed key the length is bounded (≤ 61 bytes), so
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

/// The public half of the dev signing seed (`[7u8; 32]`, `examples/mint_dev_key.rs`, also in
/// `verify/key_vectors.json`). The seed is public, so anyone can mint any tenant under this key; a
/// release build refuses it as a trust anchor, from config or from the KV `signing_key.*` channel.
/// A test derives it from the seed, so it cannot drift.
const DEV_PUBLIC_KEY: [u8; 32] = [
    234, 74, 108, 99, 226, 156, 82, 10, 190, 245, 80, 123, 19, 46, 197, 249, 149, 71, 118, 174,
    190, 190, 123, 146, 66, 30, 234, 105, 20, 70, 210, 44,
];

/// Whether `vk` is the public dev signing key ([`DEV_PUBLIC_KEY`]).
pub(crate) fn is_public_dev_key(vk: &VerifyingKey) -> bool {
    vk.to_bytes() == DEV_PUBLIC_KEY
}

/// Mint a `bai_v1` virtual key (tenant+vpc only). Lives here for tests + determinism checks
/// and as the reference implementation; production minting is the Go control plane
/// (`crypto/ed25519`), which must produce byte-identical output for the same inputs.
///
/// Never in the production binary (the gateway holds only public keys and cannot mint): built for
/// the unit tests, and under the `mint` feature, which only this crate's dev targets enable.
#[cfg(any(test, feature = "mint"))]
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
/// and the signed bytes are `bai_v2.{kid}.{payload_b64}`. Same Ed25519 keyring as v1. Gated as
/// [`mint`] is.
#[cfg(any(test, feature = "mint"))]
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

/// Mint a `bai_v3` virtual key: `bai_v2` plus `exp`, unix seconds, after which (and
/// [`LEEWAY_SECS`]) the gateway refuses it. `exp` is mandatory — a key that should not expire is a
/// `bai_v2` key.
///
/// Byte layout (the Go control plane must match it byte for byte):
/// `bai_v3.{kid}.{payload_b64}.{sig_b64}` where `payload` is 32 bytes little-endian
/// `tenant_id u64 || vpc_id u64 || key_id u64 || exp u64`, `payload_b64`/`sig_b64` are base64url
/// (no pad), and the signed bytes are `bai_v3.{kid}.{payload_b64}`. Same Ed25519 keyring as v1/v2.
/// Gated as [`mint`] is.
#[cfg(any(test, feature = "mint"))]
#[allow(clippy::expect_used)] // payload is a fixed 43-char base64 of 32 bytes; always fits the cap
pub fn mint_v3(
    vk: &VirtualKey,
    key_id: u64,
    exp: u64,
    kid: Kid,
    signing_key: &SigningKey,
) -> String {
    let payload_b64 = URL_SAFE_NO_PAD.encode(vk.encode_payload_v3(key_id, exp));
    let mut signed_buf = [0u8; SIGNED_BYTES_CAP];
    let signed = write_signed_bytes(&mut signed_buf, PREFIX_V3, kid, &payload_b64)
        .expect("minted v3 signed bytes fit in SIGNED_BYTES_CAP");
    let sig: Signature = signing_key.sign(signed);
    let sig_b64 = URL_SAFE_NO_PAD.encode(sig.to_bytes());
    format!("{PREFIX_V3}.{kid}.{payload_b64}.{sig_b64}")
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
    fn the_dev_public_key_is_what_the_dev_seed_derives() {
        let (_, vk) = test_keypair(7);
        assert!(is_public_dev_key(&vk));
        assert_eq!(DEV_PUBLIC_KEY, vk.to_bytes());
        assert!(!is_public_dev_key(&test_keypair(8).1));
    }

    /// claim: A1
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

    /// claim: A1
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
        assert_eq!(ring.verify("bai_v4.1.aaaa.bbbb"), Err(KeyError::BadVersion));
        // A v2-shaped token with junk payload/sig is this version, just malformed — not BadVersion.
        assert_eq!(ring.verify("bai_v2.1.aaaa.bbbb"), Err(KeyError::Malformed));
        assert_eq!(
            ring.verify("sk-openai.1.aaaa.bbbb"),
            Err(KeyError::Malformed)
        );
    }

    /// claim: SEC-12
    /// defect: D28
    #[test]
    fn only_the_canonical_kid_spelling_verifies() {
        let (sk, vk) = test_keypair(9);
        let ring = ring_with(1, vk);
        let id = VirtualKey {
            tenant_id: 5,
            vpc_id: 6,
            key_id: None,
        };
        let token = mint(&id, 1, &sk);
        assert_eq!(ring.verify(&token).as_deref(), Ok(&id));
        let rest = token.strip_prefix("bai_v1.1.").unwrap();
        for kid in ["01", "+1", "001", "1 ", " 1", ""] {
            let respelled = format!("bai_v1.{kid}.{rest}");
            assert_eq!(
                ring.verify(&respelled),
                Err(KeyError::Malformed),
                "kid {kid:?} must not verify as kid 1"
            );
        }
        let (sk0, vk0) = test_keypair(10);
        let ring0 = ring_with(0, vk0);
        assert!(
            ring0.verify(&mint(&id, 0, &sk0)).is_ok(),
            "kid 0 is canonical"
        );
    }

    const EXP: u64 = 2_000_000_000;

    fn who() -> VirtualKey {
        VirtualKey {
            tenant_id: 42,
            vpc_id: 99,
            key_id: None,
        }
    }

    /// Verify at a fixed `now` against only the config ring.
    fn verify_at(ring: &Keyring, token: &str, now: u64) -> Result<Identity, KeyError> {
        ring.verify_with(token, &LiveKeys::default(), &DenySet::default(), || {
            Some(now)
        })
    }

    /// claim: A5
    #[test]
    fn v3_roundtrips_identity_and_key_id() {
        let (sk, vk) = test_keypair(1);
        let ring = ring_with(7, vk);
        let token = mint_v3(&who(), 1_000_042, EXP, 7, &sk);
        assert!(token.starts_with("bai_v3."));
        let got = verify_at(&ring, &token, EXP - 10).unwrap();
        assert_eq!(
            (got.tenant_id, got.vpc_id, got.key_id),
            (42, 99, Some(1_000_042))
        );
        // Deterministic like v2: same inputs, same token; a different exp is a different token.
        assert_eq!(token, mint_v3(&who(), 1_000_042, EXP, 7, &sk));
        assert_ne!(token, mint_v3(&who(), 1_000_042, EXP + 1, 7, &sk));
    }

    /// claim: A5
    #[test]
    fn v3_is_expired_only_past_exp_plus_leeway() {
        let (sk, vk) = test_keypair(1);
        let ring = ring_with(7, vk);
        let token = mint_v3(&who(), 5, EXP, 7, &sk);
        for now in [
            0,
            EXP - 1,
            EXP,
            EXP + 1,
            EXP + LEEWAY_SECS - 1,
            EXP + LEEWAY_SECS,
        ] {
            assert!(verify_at(&ring, &token, now).is_ok(), "now={now}");
        }
        for now in [EXP + LEEWAY_SECS + 1, EXP + 3600, u64::MAX] {
            assert_eq!(
                verify_at(&ring, &token, now),
                Err(KeyError::Expired),
                "now={now}"
            );
        }
    }

    #[test]
    fn v3_exp_at_the_top_of_the_range_does_not_wrap() {
        let (sk, vk) = test_keypair(1);
        let ring = ring_with(7, vk);
        let token = mint_v3(&who(), 5, u64::MAX, 7, &sk);
        assert!(verify_at(&ring, &token, u64::MAX).is_ok());
    }

    #[test]
    fn expiry_is_judged_after_the_signature() {
        // An expired token that was tampered with is a bad signature, not "expired": the verdict
        // on `exp` is only given to a token the signature vouches for.
        let (sk, vk) = test_keypair(1);
        let ring = ring_with(7, vk);
        let token = mint_v3(&who(), 5, 1, 7, &sk);
        let mut parts: Vec<&str> = token.split('.').collect();
        let mut payload = URL_SAFE_NO_PAD.decode(parts[2]).unwrap();
        payload[24] ^= 1;
        let forged = URL_SAFE_NO_PAD.encode(&payload);
        parts[2] = &forged;
        assert_eq!(
            verify_at(&ring, &parts.join("."), EXP),
            Err(KeyError::BadSignature)
        );
    }

    #[test]
    fn exp_is_read_only_for_v3() {
        // `now` is a clock read: v1/v2 must not take it.
        let (sk, vk) = test_keypair(1);
        let ring = ring_with(7, vk);
        let v2 = mint_v2(&who(), 5, 7, &sk);
        let got = ring.verify_with(&v2, &LiveKeys::default(), &DenySet::default(), || {
            unreachable!("v2 has no expiry")
        });
        assert!(got.is_ok());
    }

    /// A clock read that fails (before the epoch) refuses a v3 key instead of reading as 0, which
    /// would pass every `exp`; v1 and v2 never read the clock.
    /// claim: A5
    /// defect: D293
    #[test]
    fn an_unreadable_clock_refuses_a_v3_key() {
        let (sk, vk) = test_keypair(1);
        let ring = ring_with(7, vk);
        let v3 = mint_v3(&who(), 5, EXP, 7, &sk);
        let none = LiveKeys::default();
        let deny = DenySet::default();
        assert_eq!(
            ring.verify_with(&v3, &none, &deny, || None),
            Err(KeyError::Expired)
        );
        assert!(
            ring.verify_with(&v3, &none, &deny, || Some(EXP - 10))
                .is_ok()
        );
        let never = mint_v3(&who(), 5, u64::MAX, 7, &sk);
        assert_eq!(
            ring.verify_with(&never, &none, &deny, || None),
            Err(KeyError::Expired)
        );
        let v2 = mint_v2(&who(), 5, 7, &sk);
        assert!(ring.verify_with(&v2, &none, &deny, || None).is_ok());
    }

    /// claim: A5
    #[test]
    fn a_v2_token_relabelled_as_v3_is_rejected() {
        let (sk, vk) = test_keypair(1);
        let ring = ring_with(7, vk);
        let v2 = mint_v2(&who(), 5, 7, &sk);
        let relabelled = v2.replacen("bai_v2", "bai_v3", 1);
        assert_eq!(
            verify_at(&ring, &relabelled, 0),
            Err(KeyError::BadSignature)
        );
        // And a v3 relabelled down to v2 drops its exp, which the signature forbids too.
        let v3 = mint_v3(&who(), 5, EXP, 7, &sk);
        let down = v3.replacen("bai_v3", "bai_v2", 1);
        assert_eq!(verify_at(&ring, &down, 0), Err(KeyError::BadSignature));
    }

    #[test]
    fn a_correctly_signed_v3_with_the_wrong_payload_length_is_malformed() {
        let (sk, vk) = test_keypair(1);
        let ring = ring_with(7, vk);
        for len in [16usize, 24, 31, 33] {
            let payload = URL_SAFE_NO_PAD.encode(vec![1u8; len]);
            let mut buf = [0u8; SIGNED_BYTES_CAP];
            let signed = write_signed_bytes(&mut buf, PREFIX_V3, 7, &payload).unwrap();
            let sig = URL_SAFE_NO_PAD.encode(sk.sign(signed).to_bytes());
            let token = format!("{PREFIX_V3}.7.{payload}.{sig}");
            assert_eq!(
                verify_at(&ring, &token, 0),
                Err(KeyError::Malformed),
                "len={len}"
            );
        }
    }

    #[test]
    fn a_small_order_public_key_does_not_verify_strictly() {
        // The identity point as the public key with the identity as R and S = 0 verifies under the
        // lax check for any message; `verify_strict` refuses it.
        let mut ident = [0u8; 32];
        ident[0] = 1;
        let weak = VerifyingKey::from_bytes(&ident).unwrap();
        let ring = ring_with(1, weak);
        let payload = URL_SAFE_NO_PAD.encode(who().encode_payload_v2(5));
        let mut sig = [0u8; 64];
        sig[0] = 1;
        let token = format!("{PREFIX_V2}.1.{payload}.{}", URL_SAFE_NO_PAD.encode(sig));
        assert_eq!(ring.verify(&token), Err(KeyError::BadSignature));
    }

    /// claim: A6
    #[test]
    fn a_distrusted_kid_is_refused_from_either_source_before_any_signature_math() {
        let (sk, vk) = test_keypair(1);
        let ring = ring_with(7, vk);
        let mut deny = DenySet::new();
        deny.insert_target(
            crate::deny::DenyTarget::Kid(7),
            crate::deny::DenyReason::Fraud,
        );
        let token = mint_v2(&who(), 5, 7, &sk);
        let none = LiveKeys::default();
        assert_eq!(
            ring.verify_with(&token, &none, &deny, || Some(0)),
            Err(KeyError::Distrusted(7))
        );
        // Same for a live-only kid.
        let mut live = LiveKeys::new();
        live.put(
            8,
            base64::engine::general_purpose::STANDARD
                .encode(vk.to_bytes())
                .as_bytes(),
        );
        let t8 = mint_v2(&who(), 5, 8, &sk);
        assert!(
            Keyring::new()
                .verify_with(&t8, &live, &DenySet::default(), || Some(0))
                .is_ok()
        );
        deny.insert_target(
            crate::deny::DenyTarget::Kid(8),
            crate::deny::DenyReason::Fraud,
        );
        assert_eq!(
            Keyring::new().verify_with(&t8, &live, &deny, || Some(0)),
            Err(KeyError::Distrusted(8))
        );
        // The garbage token with a distrusted kid is still refused for its kid, with no sig to check.
        assert_eq!(
            ring.verify_with("bai_v2.7.AAAA.AAAA", &none, &deny, || Some(0)),
            Err(KeyError::Malformed)
        );
    }

    #[test]
    fn config_key_wins_when_both_sources_name_a_kid() {
        let (sk_cfg, vk_cfg) = test_keypair(1);
        let (sk_live, vk_live) = test_keypair(2);
        let ring = ring_with(7, vk_cfg);
        let mut live = LiveKeys::new();
        live.put(
            7,
            base64::engine::general_purpose::STANDARD
                .encode(vk_live.to_bytes())
                .as_bytes(),
        );
        let d = DenySet::default();
        assert!(
            ring.verify_with(&mint_v2(&who(), 5, 7, &sk_cfg), &live, &d, || Some(0))
                .is_ok()
        );
        assert_eq!(
            ring.verify_with(&mint_v2(&who(), 5, 7, &sk_live), &live, &d, || Some(0)),
            Err(KeyError::BadSignature)
        );
    }

    #[test]
    fn managed_prefix_is_v1_v2_or_v3() {
        assert!(is_managed_prefix("bai_v1.1.payload.sig"));
        assert!(is_managed_prefix("bai_v2.1.payload.sig"));
        assert!(is_managed_prefix("bai_v3.1.aaaa.bbbb"));
        // Unknown versions stay off the managed branch (verify would BadVersion if we entered).
        assert!(!is_managed_prefix("bai_v4.1.aaaa.bbbb"));
        assert!(!is_managed_prefix("bai_"));
        assert!(!is_managed_prefix("sk-openai"));
    }
}
