//! Session grant: `bsg_v1`. What a `serve` replica learns about a connection, and the secrets it needs
//! to run that session, from one token the edge presents.
//!
//! The control plane mints; a replica only **verifies and unseals**. Two asymmetric keys make that
//! split real rather than a convention:
//!
//! - **Ed25519** signs the claims (tenant, session, shard, workspace, exec/MCP endpoints, expiry). A
//!   replica holds only the *public* key, by `kid`, so it can check a grant offline — no lookup, no
//!   control-plane round trip — but a compromised replica still cannot mint one. Same shape as the
//!   gateway's `bai_v1` virtual key and the control plane's `bagt_v1`.
//! - **X25519** seals the secrets (exec/MCP headers, the gateway key, the tenant's DEK) to the
//!   fleet's public key: an ephemeral key agreement, HKDF-SHA256, then XChaCha20-Poly1305. The edge
//!   that relays a grant never sees what's inside it, and a grant captured in transit or in a log
//!   is useless without the fleet secret.
//!
//! This crate owns the spec (ARCHITECTURE.md, "Session grant (`bsg_v1`)") and the golden vectors
//! (`tests/fixtures/grant/v1.json`); the control plane's Go minter must reproduce those vectors byte
//! for byte. The only minter here is test-only, and exists to pin those vectors.
//!
//! Every rejection is its own [`GrantError`] variant, and none carries secret material: error text is
//! static apart from a kid or a timestamp, and every secret type's `Debug` redacts.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use base64::Engine;
use base64::engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD};
use chacha20poly1305::aead::{AeadInPlace, KeyInit};
use chacha20poly1305::{Key, Tag, XChaCha20Poly1305, XNonce};
use ed25519_dalek::{Signature, VerifyingKey};
use hkdf::Hkdf;
use serde::Deserialize;
use sha2::Sha256;
use x25519_dalek::{PublicKey, SharedSecret, StaticSecret};
use zeroize::{Zeroize, Zeroizing};

use crate::auth_store::Secret;

/// Wire prefix + version. Inside the signed bytes (and the seal's AAD), so it cannot be downgraded.
pub const PREFIX: &str = "bsg_v1";

/// Signing-key identifier: which control-plane key signed a grant. Several can be trusted at once,
/// so a key rotates without a flag day.
pub type Kid = u32;

/// HKDF `info` — domain separation, so this key agreement can never be replayed as another's.
const HKDF_INFO: &[u8] = b"bsg_v1 seal";
/// `sealed` = `eph_pk (32) ‖ nonce (24) ‖ ciphertext ‖ tag (16)`.
const EPH_LEN: usize = 32;
const NONCE_LEN: usize = 24;
const TAG_LEN: usize = 16;

/// Why a grant was rejected. One variant per stage of [`GrantVerifier::verify`], in the order they
/// run, so an operator can tell a rotation gap from a clock problem from a forgery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum GrantError {
    /// Not `bsg_v1.{kid}.{payload}.{sig}`: wrong prefix or field count, a kid that isn't canonical
    /// decimal, or a signature that isn't base64url of 64 bytes. Nothing was verified.
    #[error("malformed session grant")]
    Malformed,
    /// Signed under a kid this replica doesn't trust — typically a rotation gap: the control plane
    /// is minting under a key not yet passed as `--grant-key`.
    #[error("session grant signed by unknown key id {0}")]
    UnknownKid(Kid),
    /// The Ed25519 signature doesn't cover these bytes: tampered, or signed by another key.
    #[error("session grant signature verification failed")]
    BadSignature,
    /// Authentic, but `now >= exp`.
    #[error("session grant expired at {exp} (now {now})")]
    Expired { exp: u64, now: u64 },
    /// `sealed` isn't base64url of `eph_pk ‖ nonce ‖ ciphertext ‖ tag`, is too short to hold them, or
    /// carries a low-order ephemeral key (an all-zero shared secret).
    #[error("session grant's sealed secrets are malformed")]
    SealMalformed,
    /// The AEAD refused to open: sealed to a different fleet key than `--seal-key`, or altered.
    #[error("session grant's sealed secrets failed to open")]
    SealOpenFailed,
    /// Authentic, but the signed claims or the opened secrets aren't the `bsg_v1` shape (a missing or
    /// unknown field, a DEK that isn't 32 bytes). A minter bug, never an attacker.
    #[error("session grant claims are not the bsg_v1 shape")]
    BadClaims,
}

/// A verified, unsealed grant.
#[derive(Debug)]
pub struct Grant {
    pub tenant: String,
    pub session_id: String,
    pub home_shard: String,
    pub workspace_root: String,
    pub exec_url: String,
    pub mcp: Vec<McpConnector>,
    /// Unix seconds; the grant is valid strictly before this.
    pub exp: u64,
    pub secrets: GrantSecrets,
}

/// One MCP server the session may reach. Its credentials, if any, are in
/// [`GrantSecrets::mcp_headers`] under the same `name`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[cfg_attr(test, derive(serde::Serialize))]
#[serde(deny_unknown_fields)]
pub struct McpConnector {
    pub name: String,
    pub url: String,
}

/// What the grant carries sealed. Every value here is a credential; `Debug` shows only structure.
#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(serde::Serialize))]
#[serde(deny_unknown_fields)]
pub struct GrantSecrets {
    /// Sent with every exec request (the claims' `exec_url`).
    pub exec_headers: Vec<SecretHeader>,
    /// Per MCP connector, by [`McpConnector::name`].
    pub mcp_headers: BTreeMap<String, Vec<SecretHeader>>,
    /// The session's gateway credential.
    pub gateway_key: Secret,
    /// The tenant's data-encryption key: **per-tenant and stable**, not per-session. Every session a
    /// tenant owns is sealed under keys derived from this one
    /// ([`crate::session_store::TenantCodec`]), because a listing, a fork and a preview all open
    /// *other* sessions' files — the session id is mixed in as associated data instead.
    pub dek: Dek,
}

/// An HTTP header whose value is a credential: `Debug` shows the name, never the value.
#[derive(Debug, Deserialize)]
#[cfg_attr(test, derive(serde::Serialize))]
#[serde(deny_unknown_fields)]
pub struct SecretHeader {
    pub name: String,
    pub value: Secret,
}

/// A **tenant's** 32-byte data-encryption key — stable across that tenant's sessions, not minted per
/// session. `Debug` redacts it; it is zeroized on drop. On the wire it is standard base64 (padded),
/// and anything but exactly 32 bytes is [`GrantError::BadClaims`].
pub struct Dek([u8; 32]);

impl Dek {
    pub fn expose(&self) -> &[u8; 32] {
        &self.0
    }
}

impl std::fmt::Debug for Dek {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Dek(***)")
    }
}

impl Drop for Dek {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

impl<'de> Deserialize<'de> for Dek {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let b64 = Zeroizing::new(String::deserialize(deserializer)?);
        decode_exact(&STANDARD, &b64)
            .map(Dek)
            .ok_or_else(|| serde::de::Error::custom("dek is not standard base64 of 32 bytes"))
    }
}

#[cfg(test)]
impl serde::Serialize for Dek {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&STANDARD.encode(self.0))
    }
}

/// The signed payload, in wire order. `deny_unknown_fields`: a v1 replica refuses a claim it doesn't
/// understand rather than ignoring it — an ignored restriction would fail open.
#[derive(Deserialize)]
#[cfg_attr(test, derive(serde::Serialize))]
#[serde(deny_unknown_fields)]
struct Payload {
    tenant: String,
    session_id: String,
    home_shard: String,
    workspace_root: String,
    exec_url: String,
    mcp: Vec<McpConnector>,
    exp: u64,
    sealed: String,
}

/// Verifies and unseals `bsg_v1` grants: the trusted Ed25519 keys by kid, plus the fleet's X25519
/// secret. Built once at startup (`serve --grant-key … --seal-key …`) and shared by every session.
pub struct GrantVerifier {
    keys: HashMap<Kid, VerifyingKey>,
    fleet_secret: StaticSecret,
    /// `fleet_secret`'s public half — half of the HKDF salt, so derived once here.
    fleet_public: PublicKey,
}

impl GrantVerifier {
    pub fn new(keys: HashMap<Kid, VerifyingKey>, fleet_secret: StaticSecret) -> Self {
        let fleet_public = PublicKey::from(&fleet_secret);
        Self {
            keys,
            fleet_secret,
            fleet_public,
        }
    }

    /// Build from `serve`'s `--grant-key` (repeatable) and `--seal-key` flags. Neither ⇒ `Ok(None)`:
    /// grants are off and nothing else changes. Only one ⇒ an error, since a grant always carries
    /// sealed secrets — a keyring without the seal key (or the reverse) could accept nothing. So could
    /// a duplicated kid, which is refused rather than resolved last-wins.
    pub fn from_flags(
        grant_keys: &[String],
        seal_key: Option<&Path>,
    ) -> Result<Option<Self>, String> {
        let seal_key = match (grant_keys.is_empty(), seal_key) {
            (true, None) => return Ok(None),
            (false, Some(path)) => path,
            (true, Some(_)) => return Err("--seal-key needs at least one --grant-key".into()),
            (false, None) => return Err("--grant-key needs --seal-key".into()),
        };
        let mut keys = HashMap::with_capacity(grant_keys.len());
        for arg in grant_keys {
            let (kid, key) = parse_grant_key(arg)?;
            if keys.insert(kid, key).is_some() {
                return Err(format!("--grant-key: kid {kid} is given more than once"));
            }
        }
        Ok(Some(Self::new(keys, read_seal_key(seal_key)?)))
    }

    /// Verify `token` and open its secrets. `now` is unix seconds (a parameter so tests pin it).
    ///
    /// Order: parse → known kid → signature → claims → expiry → unseal. Nothing from the payload is
    /// read before the signature checks out, and no crypto runs for a kid this replica doesn't trust.
    pub fn verify(&self, token: &str, now: u64) -> Result<Grant, GrantError> {
        // Parse. The signed bytes and the seal's AAD are slices of the token itself — once the prefix
        // and kid are checked, `signed` is exactly `bsg_v1.{kid}.{payload_b64}` and `head` exactly
        // `bsg_v1.{kid}` — so neither is rebuilt or copied.
        let (signed, sig_b64) = token.rsplit_once('.').ok_or(GrantError::Malformed)?;
        let (head, payload_b64) = signed.rsplit_once('.').ok_or(GrantError::Malformed)?;
        let (prefix, kid) = head.split_once('.').ok_or(GrantError::Malformed)?;
        if prefix != PREFIX {
            return Err(GrantError::Malformed);
        }
        let kid = parse_kid(kid).ok_or(GrantError::Malformed)?;
        let sig: [u8; 64] = decode_exact(&URL_SAFE_NO_PAD, sig_b64).ok_or(GrantError::Malformed)?;

        let key = self.keys.get(&kid).ok_or(GrantError::UnknownKid(kid))?;
        // `verify_strict` also rejects small-order keys and non-canonical signatures; an honest
        // minter (Go's `crypto/ed25519`) never produces either.
        key.verify_strict(signed.as_bytes(), &Signature::from_bytes(&sig))
            .map_err(|_| GrantError::BadSignature)?;

        let payload = URL_SAFE_NO_PAD
            .decode(payload_b64)
            .map_err(|_| GrantError::BadClaims)?;
        let payload: Payload =
            serde_json::from_slice(&payload).map_err(|_| GrantError::BadClaims)?;
        if now >= payload.exp {
            return Err(GrantError::Expired {
                exp: payload.exp,
                now,
            });
        }
        let secrets = self.unseal(head.as_bytes(), &payload.sealed)?;
        Ok(Grant {
            tenant: payload.tenant,
            session_id: payload.session_id,
            home_shard: payload.home_shard,
            workspace_root: payload.workspace_root,
            exec_url: payload.exec_url,
            mcp: payload.mcp,
            exp: payload.exp,
            secrets,
        })
    }

    /// Open `sealed` (`eph_pk ‖ nonce ‖ ciphertext ‖ tag`) in place, inside the decoded blob — no
    /// second buffer — and parse the plaintext. The blob is zeroized on drop either way.
    fn unseal(&self, aad: &[u8], sealed: &str) -> Result<GrantSecrets, GrantError> {
        let mut blob = Zeroizing::new(
            URL_SAFE_NO_PAD
                .decode(sealed)
                .map_err(|_| GrantError::SealMalformed)?,
        );
        if blob.len() < EPH_LEN + NONCE_LEN + TAG_LEN {
            return Err(GrantError::SealMalformed);
        }
        let (head, body) = blob.split_at_mut(EPH_LEN + NONCE_LEN);
        let (eph, nonce) = head.split_at(EPH_LEN);
        let eph =
            PublicKey::from(<[u8; 32]>::try_from(eph).map_err(|_| GrantError::SealMalformed)?);
        let key = derive_seal_key(
            &self.fleet_secret.diffie_hellman(&eph),
            &eph,
            &self.fleet_public,
        )
        .ok_or(GrantError::SealMalformed)?;
        let (ciphertext, tag) = body.split_at_mut(body.len() - TAG_LEN);
        XChaCha20Poly1305::new(Key::from_slice(&key[..]))
            .decrypt_in_place_detached(
                XNonce::from_slice(nonce),
                aad,
                ciphertext,
                Tag::from_slice(tag),
            )
            .map_err(|_| GrantError::SealOpenFailed)?;
        serde_json::from_slice(ciphertext).map_err(|_| GrantError::BadClaims)
    }
}

/// HKDF-SHA256(ikm = shared, salt = eph_pk ‖ fleet_pk, info = `bsg_v1 seal`) → the AEAD key. `None`
/// for a non-contributory (all-zero) shared secret — a low-order `eph_pk`, which would make the key
/// independent of the fleet secret.
fn derive_seal_key(
    shared: &SharedSecret,
    eph_pk: &PublicKey,
    fleet_pk: &PublicKey,
) -> Option<Zeroizing<[u8; 32]>> {
    if !shared.was_contributory() {
        return None;
    }
    let mut salt = [0u8; 64];
    salt[..32].copy_from_slice(eph_pk.as_bytes());
    salt[32..].copy_from_slice(fleet_pk.as_bytes());
    let mut key = Zeroizing::new([0u8; 32]);
    Hkdf::<Sha256>::new(Some(&salt), shared.as_bytes())
        .expand(HKDF_INFO, key.as_mut())
        .ok()?;
    Some(key)
}

/// A kid is canonical decimal — ASCII digits, no sign, no leading zero — so each has one spelling.
fn parse_kid(s: &str) -> Option<Kid> {
    let canonical =
        !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()) && (s == "0" || !s.starts_with('0'));
    canonical.then(|| s.parse().ok()).flatten()
}

/// Decode base64 of exactly `N` bytes onto the stack; anything longer or shorter is `None`.
fn decode_exact<const N: usize>(engine: &impl Engine, s: &str) -> Option<[u8; N]> {
    let mut out = [0u8; N];
    match engine.decode_slice(s, &mut out) {
        Ok(n) if n == N => Some(out),
        _ => {
            out.zeroize();
            None
        }
    }
}

/// A 32-byte key as either standard (padded) or url-safe (unpadded) base64, surrounding whitespace
/// ignored — whichever form the operator's tooling wrote.
fn decode_key(s: &str) -> Option<[u8; 32]> {
    let s = s.trim();
    decode_exact(&STANDARD, s).or_else(|| decode_exact(&URL_SAFE_NO_PAD, s))
}

/// Parse one `--grant-key` value: `<kid>=<Ed25519 public key>`, the key as standard or url-safe
/// base64. Split at the first `=`, so standard base64's own padding is kept.
pub fn parse_grant_key(arg: &str) -> Result<(Kid, VerifyingKey), String> {
    let (kid, key) = arg.split_once('=').ok_or_else(|| {
        format!("--grant-key {arg:?}: expected <kid>=<base64 Ed25519 public key>")
    })?;
    let kid = parse_kid(kid.trim())
        .ok_or_else(|| format!("--grant-key {arg:?}: the kid must be a decimal u32"))?;
    let key = decode_key(key)
        .and_then(|bytes| VerifyingKey::from_bytes(&bytes).ok())
        .ok_or_else(|| {
            format!("--grant-key {arg:?}: not base64 of a 32-byte Ed25519 public key")
        })?;
    Ok((kid, key))
}

/// Read `--seal-key`: a file holding the fleet's X25519 secret as base64 of 32 bytes (standard or
/// url-safe; surrounding whitespace ignored — `wg genkey` writes exactly this). Errors name the path,
/// never the contents.
pub fn read_seal_key(path: &Path) -> Result<StaticSecret, String> {
    let contents = Zeroizing::new(
        std::fs::read_to_string(path).map_err(|e| format!("--seal-key {}: {e}", path.display()))?,
    );
    decode_key(&contents)
        .map(|mut bytes| {
            let secret = StaticSecret::from(bytes);
            bytes.zeroize();
            secret
        })
        .ok_or_else(|| {
            format!(
                "--seal-key {}: expected base64 of a 32-byte X25519 secret",
                path.display()
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};
    use serde::Serialize;

    /// The golden vectors. The control plane's Go minter reproduces `token` from these inputs.
    const FIXTURE: &str = include_str!("../tests/fixtures/grant/v1.json");
    const FIXTURE_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/grant/v1.json");

    #[derive(Serialize, Deserialize)]
    struct Fixture {
        #[serde(rename = "_comment")]
        comment: Vec<String>,
        kid: Kid,
        signing_seed_hex: String,
        verifying_key_hex: String,
        /// `kid=key` exactly as `--grant-key` takes it.
        grant_key_flag: String,
        fleet_secret_hex: String,
        fleet_public_hex: String,
        /// The `--seal-key` file's contents.
        seal_key_file: String,
        ephemeral_secret_hex: String,
        ephemeral_public_hex: String,
        nonce_hex: String,
        now: u64,
        claims: Claims,
        secrets: GrantSecrets,
        // Intermediates, so a second implementation can find the first step it disagrees on.
        shared_secret_hex: String,
        seal_key_hex: String,
        secrets_json: String,
        sealed: String,
        payload_json: String,
        token: String,
        negative: Vec<Negative>,
    }

    /// The claims, minus `sealed`.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct Claims {
        tenant: String,
        session_id: String,
        home_shard: String,
        workspace_root: String,
        exec_url: String,
        mcp: Vec<McpConnector>,
        exp: u64,
    }

    impl Claims {
        fn with_sealed(self, sealed: String) -> Payload {
            Payload {
                tenant: self.tenant,
                session_id: self.session_id,
                home_shard: self.home_shard,
                workspace_root: self.workspace_root,
                exec_url: self.exec_url,
                mcp: self.mcp,
                exp: self.exp,
                sealed,
            }
        }

        fn of(grant: &Grant) -> Self {
            Self {
                tenant: grant.tenant.clone(),
                session_id: grant.session_id.clone(),
                home_shard: grant.home_shard.clone(),
                workspace_root: grant.workspace_root.clone(),
                exec_url: grant.exec_url.clone(),
                mcp: grant.mcp.clone(),
                exp: grant.exp,
            }
        }
    }

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct Negative {
        name: String,
        how: String,
        token: String,
        now: u64,
        error: String,
    }

    fn hex32(s: &str) -> [u8; 32] {
        hex::decode(s).unwrap().try_into().unwrap()
    }

    /// The test-only minter: sign `payload_json` under `kid`.
    fn sign(signing_key: &SigningKey, kid: Kid, payload_json: &[u8]) -> String {
        let signed = format!("{PREFIX}.{kid}.{}", URL_SAFE_NO_PAD.encode(payload_json));
        let sig = signing_key.sign(signed.as_bytes());
        format!("{signed}.{}", URL_SAFE_NO_PAD.encode(sig.to_bytes()))
    }

    /// The test-only sealer, deterministic in its ephemeral secret and nonce: the raw blob
    /// (`eph_pk ‖ nonce ‖ ciphertext ‖ tag`), plus the shared secret and AEAD key for the fixture.
    fn seal(
        fleet_pk: &PublicKey,
        kid: Kid,
        eph_secret: [u8; 32],
        nonce: [u8; 24],
        plaintext: &[u8],
    ) -> (Vec<u8>, [u8; 32], [u8; 32]) {
        let eph = StaticSecret::from(eph_secret);
        let eph_pk = PublicKey::from(&eph);
        let shared = eph.diffie_hellman(fleet_pk);
        let key = derive_seal_key(&shared, &eph_pk, fleet_pk).unwrap();
        let mut ciphertext = plaintext.to_vec();
        let tag = XChaCha20Poly1305::new(Key::from_slice(&key[..]))
            .encrypt_in_place_detached(
                XNonce::from_slice(&nonce),
                format!("{PREFIX}.{kid}").as_bytes(),
                &mut ciphertext,
            )
            .unwrap();
        let blob = [
            &eph_pk.as_bytes()[..],
            &nonce[..],
            &ciphertext[..],
            &tag[..],
        ]
        .concat();
        (blob, shared.to_bytes(), *key)
    }

    /// Everything the fixture derives, recomputed from its inputs: the valid grant and every
    /// negative case. `mint_reproduces_the_golden_vectors_byte_for_byte` compares this to the file.
    fn build(inputs: &Fixture) -> Fixture {
        let kid = inputs.kid;
        let signing_key = SigningKey::from_bytes(&hex32(&inputs.signing_seed_hex));
        let verifying_key = signing_key.verifying_key();
        let fleet_secret = StaticSecret::from(hex32(&inputs.fleet_secret_hex));
        let fleet_pk = PublicKey::from(&fleet_secret);
        let eph_secret = hex32(&inputs.ephemeral_secret_hex);
        let nonce: [u8; 24] = hex::decode(&inputs.nonce_hex).unwrap().try_into().unwrap();
        let now = inputs.now;

        // The typed struct serializes in wire order, so this is the exact plaintext.
        let secrets_json = serde_json::to_string(&inputs.secrets).unwrap();
        let secrets = &inputs.secrets;
        let (blob, shared, seal_key) =
            seal(&fleet_pk, kid, eph_secret, nonce, secrets_json.as_bytes());
        let sealed = URL_SAFE_NO_PAD.encode(&blob);
        let payload_of = |claims: Claims, sealed: &str| {
            serde_json::to_string(&claims.with_sealed(sealed.to_owned())).unwrap()
        };
        let payload_json = payload_of(inputs.claims.clone(), &sealed);
        let token = sign(&signing_key, kid, payload_json.as_bytes());

        let negative = |name: &str, how: &str, token: String, now: u64, error: &str| Negative {
            name: name.into(),
            how: how.into(),
            token,
            now,
            error: error.into(),
        };
        let resign_with_sealed = |blob: &[u8]| {
            let payload = payload_of(inputs.claims.clone(), &URL_SAFE_NO_PAD.encode(blob));
            sign(&signing_key, kid, payload.as_bytes())
        };
        let resign_with_secrets = |secrets_json: &str| {
            let (blob, _, _) = seal(&fleet_pk, kid, eph_secret, nonce, secrets_json.as_bytes());
            resign_with_sealed(&blob)
        };
        let (_, sig_b64) = token.rsplit_once('.').unwrap();
        let (_, rest) = token.split_once('.').unwrap();

        let negatives = vec![
            negative(
                "wrong_prefix",
                "the valid token with its prefix changed to bsg_v2",
                format!("bsg_v2.{rest}"),
                now,
                "Malformed",
            ),
            negative(
                "tampered_payload",
                "the valid token's payload re-encoded with tenant 00tenant2, original signature kept",
                {
                    let mut claims = inputs.claims.clone();
                    claims.tenant = "00tenant2".into();
                    let payload = payload_of(claims, &sealed);
                    format!(
                        "{PREFIX}.{kid}.{}.{sig_b64}",
                        URL_SAFE_NO_PAD.encode(payload)
                    )
                },
                now,
                "BadSignature",
            ),
            negative(
                "tampered_signature",
                "the valid token with the first signature byte XORed with 0x01",
                {
                    let mut sig = URL_SAFE_NO_PAD.decode(sig_b64).unwrap();
                    sig[0] ^= 0x01;
                    let (signed, _) = token.rsplit_once('.').unwrap();
                    format!("{signed}.{}", URL_SAFE_NO_PAD.encode(sig))
                },
                now,
                "BadSignature",
            ),
            negative(
                "unknown_kid",
                "the valid payload signed under kid 2 by the key from signing seed 0x09 repeated, \
                 which this keyring does not trust",
                sign(
                    &SigningKey::from_bytes(&[0x09; 32]),
                    2,
                    payload_json.as_bytes(),
                ),
                now,
                "UnknownKid",
            ),
            negative(
                "expired",
                "the valid grant re-minted with exp = now: a grant is valid strictly before exp",
                {
                    let mut claims = inputs.claims.clone();
                    claims.exp = now;
                    sign(&signing_key, kid, payload_of(claims, &sealed).as_bytes())
                },
                now,
                "Expired",
            ),
            negative(
                "wrong_seal_key",
                "the valid secrets sealed to the fleet secret 0x44 repeated instead, re-signed",
                {
                    let other = PublicKey::from(&StaticSecret::from([0x44; 32]));
                    let (blob, _, _) =
                        seal(&other, kid, eph_secret, nonce, secrets_json.as_bytes());
                    resign_with_sealed(&blob)
                },
                now,
                "SealOpenFailed",
            ),
            negative(
                "truncated_sealed",
                "the valid sealed blob cut to 71 bytes, one short of eph_pk + nonce + tag, re-signed",
                resign_with_sealed(&blob[..EPH_LEN + NONCE_LEN + TAG_LEN - 1]),
                now,
                "SealMalformed",
            ),
            negative(
                "low_order_ephemeral",
                "the valid sealed blob with eph_pk replaced by 32 zero bytes (a low-order point), \
                 re-signed",
                {
                    let mut blob = blob.clone();
                    blob[..EPH_LEN].fill(0);
                    resign_with_sealed(&blob)
                },
                now,
                "SealMalformed",
            ),
            negative(
                "unknown_claim",
                "the valid payload with an extra \"scope\":\"admin\" claim appended, re-signed",
                {
                    let payload = format!(
                        "{},\"scope\":\"admin\"}}",
                        payload_json.strip_suffix('}').unwrap()
                    );
                    sign(&signing_key, kid, payload.as_bytes())
                },
                now,
                "BadClaims",
            ),
            negative(
                "short_dek",
                "the valid secrets with a 31-byte dek, sealed and signed as usual",
                resign_with_secrets(&secrets_json.replace(
                    &STANDARD.encode(secrets.dek.expose()),
                    &STANDARD.encode(&secrets.dek.expose()[..31]),
                )),
                now,
                "BadClaims",
            ),
        ];

        Fixture {
            comment: inputs.comment.clone(),
            kid,
            signing_seed_hex: inputs.signing_seed_hex.clone(),
            verifying_key_hex: hex::encode(verifying_key.as_bytes()),
            grant_key_flag: format!("{kid}={}", STANDARD.encode(verifying_key.as_bytes())),
            fleet_secret_hex: inputs.fleet_secret_hex.clone(),
            fleet_public_hex: hex::encode(fleet_pk.as_bytes()),
            seal_key_file: format!("{}\n", STANDARD.encode(fleet_secret.as_bytes())),
            ephemeral_secret_hex: inputs.ephemeral_secret_hex.clone(),
            ephemeral_public_hex: hex::encode(&blob[..EPH_LEN]),
            nonce_hex: inputs.nonce_hex.clone(),
            now,
            claims: inputs.claims.clone(),
            secrets: serde_json::from_str(&secrets_json).unwrap(),
            shared_secret_hex: hex::encode(shared),
            seal_key_hex: hex::encode(seal_key),
            secrets_json,
            sealed,
            payload_json,
            token,
            negative: negatives,
        }
    }

    fn fixture() -> Fixture {
        serde_json::from_str(FIXTURE).unwrap()
    }

    /// The verifier exactly as `serve` builds it: from the fixture's flag-format keys.
    fn verifier(fx: &Fixture) -> GrantVerifier {
        let dir = tempfile::tempdir().unwrap();
        let seal_key = dir.path().join("seal.key");
        std::fs::write(&seal_key, &fx.seal_key_file).unwrap();
        GrantVerifier::from_flags(std::slice::from_ref(&fx.grant_key_flag), Some(&seal_key))
            .unwrap()
            .unwrap()
    }

    fn variant(e: &GrantError) -> &'static str {
        match e {
            GrantError::Malformed => "Malformed",
            GrantError::UnknownKid(_) => "UnknownKid",
            GrantError::BadSignature => "BadSignature",
            GrantError::Expired { .. } => "Expired",
            GrantError::SealMalformed => "SealMalformed",
            GrantError::SealOpenFailed => "SealOpenFailed",
            GrantError::BadClaims => "BadClaims",
        }
    }

    /// `UPDATE_FIXTURES=1 cargo test -p beyond-ai-agent grant::` rewrites the file (then run
    /// `dprint fmt`). Only for a deliberate change to the vectors: the control plane's minter has to
    /// match whatever this writes.
    #[test]
    fn mint_reproduces_the_golden_vectors_byte_for_byte() {
        let fx = fixture();
        let built = build(&fx);
        if std::env::var_os("UPDATE_FIXTURES").is_some() {
            let json = serde_json::to_string_pretty(&built).unwrap();
            std::fs::write(FIXTURE_PATH, json + "\n").unwrap();
            return;
        }
        // Step by step, so a failure names the first stage that diverged.
        assert_eq!(built.verifying_key_hex, fx.verifying_key_hex);
        assert_eq!(built.fleet_public_hex, fx.fleet_public_hex);
        assert_eq!(built.ephemeral_public_hex, fx.ephemeral_public_hex);
        assert_eq!(built.shared_secret_hex, fx.shared_secret_hex);
        assert_eq!(built.seal_key_hex, fx.seal_key_hex);
        assert_eq!(built.secrets_json, fx.secrets_json);
        assert_eq!(built.sealed, fx.sealed);
        assert_eq!(built.payload_json, fx.payload_json);
        assert_eq!(built.token, fx.token);
        assert_eq!(built.negative, fx.negative);
        assert_eq!(
            serde_json::to_value(&built).unwrap(),
            serde_json::to_value(&fx).unwrap()
        );
    }

    #[test]
    fn verifier_accepts_the_golden_grant_with_exactly_its_claims_and_secrets() {
        let fx = fixture();
        let grant = verifier(&fx).verify(&fx.token, fx.now).unwrap();
        assert_eq!(Claims::of(&grant), fx.claims);
        assert_eq!(
            serde_json::to_value(&grant.secrets).unwrap(),
            serde_json::to_value(&fx.secrets).unwrap()
        );
        assert_eq!(grant.mcp.len(), 2);
        assert_eq!(grant.secrets.mcp_headers.len(), 2);
        // One second before expiry is still valid.
        assert!(verifier(&fx).verify(&fx.token, fx.claims.exp - 1).is_ok());
    }

    #[test]
    fn verifier_rejects_each_negative_vector_with_its_own_error() {
        let fx = fixture();
        let verifier = verifier(&fx);
        assert!(fx.negative.len() >= 10);
        for case in &fx.negative {
            let err = verifier.verify(&case.token, case.now).unwrap_err();
            assert_eq!(variant(&err), case.error, "case {}: got {err:?}", case.name);
        }
    }

    #[test]
    fn errors_carry_the_kid_and_times_operators_need() {
        let fx = fixture();
        let verifier = verifier(&fx);
        let case = |name: &str| fx.negative.iter().find(|c| c.name == name).unwrap();
        assert_eq!(
            verifier
                .verify(&case("unknown_kid").token, fx.now)
                .unwrap_err(),
            GrantError::UnknownKid(2)
        );
        assert_eq!(
            verifier.verify(&case("expired").token, fx.now).unwrap_err(),
            GrantError::Expired {
                exp: fx.now,
                now: fx.now
            }
        );
        // The real grant, verified late, reports its own exp.
        let late = fx.claims.exp + 60;
        assert_eq!(
            verifier.verify(&fx.token, late).unwrap_err(),
            GrantError::Expired {
                exp: fx.claims.exp,
                now: late
            }
        );
    }

    #[test]
    fn no_secret_reaches_debug_output() {
        let fx = fixture();
        let grant = verifier(&fx).verify(&fx.token, fx.now).unwrap();
        let debug = format!("{grant:?}");
        let secrets = &grant.secrets;
        let mut values: Vec<String> = secrets
            .exec_headers
            .iter()
            .chain(secrets.mcp_headers.values().flatten())
            .map(|h| h.value.expose().to_owned())
            .collect();
        values.push(secrets.gateway_key.expose().to_owned());
        values.push(STANDARD.encode(secrets.dek.expose()));
        for value in &values {
            assert!(
                !debug.contains(value.as_str()),
                "{value:?} leaked into {debug}"
            );
        }
        // Structure is still visible — header names and connector names aren't secrets.
        assert!(debug.contains("Authorization"));
        assert!(debug.contains("linear"));
    }

    #[test]
    fn parse_rejects_every_malformed_shape_before_any_crypto() {
        let fx = fixture();
        let verifier = verifier(&fx);
        let (signed, sig) = fx.token.rsplit_once('.').unwrap();
        let (_, payload_and_sig) = fx.token.split_once('.').unwrap();
        let (_, after_kid) = payload_and_sig.split_once('.').unwrap();
        for token in [
            String::new(),
            "bsg_v1".into(),
            "bsg_v1.1.payload".into(),
            format!("{}.extra", fx.token),
            format!("bsg_v1.01.{after_kid}"),
            format!("bsg_v1.+1.{after_kid}"),
            format!("bsg_v1.4294967296.{after_kid}"),
            format!("{signed}.{sig}="),
            format!("{signed}.{}", &sig[..sig.len() - 1]),
            format!("{signed}.{sig}AAAA"),
        ] {
            assert_eq!(
                verifier.verify(&token, fx.now).unwrap_err(),
                GrantError::Malformed,
                "{token:?}"
            );
        }
    }

    #[test]
    fn a_seal_at_the_length_floor_reaches_the_aead() {
        // The length floor is exactly eph_pk + nonce + tag: an empty plaintext is structurally fine,
        // so it fails at the AEAD, not the parser.
        let fx = fixture();
        let blob = URL_SAFE_NO_PAD.decode(&fx.sealed).unwrap();
        let err = verifier(&fx)
            .unseal(
                format!("{PREFIX}.{}", fx.kid).as_bytes(),
                &URL_SAFE_NO_PAD.encode(&blob[..72]),
            )
            .unwrap_err();
        assert_eq!(err, GrantError::SealOpenFailed);
    }

    #[test]
    fn the_seal_is_bound_to_the_kid() {
        // AAD = `bsg_v1.{kid}`: the same blob does not open under another kid's AAD.
        let fx = fixture();
        let err = verifier(&fx).unseal(b"bsg_v1.2", &fx.sealed).unwrap_err();
        assert_eq!(err, GrantError::SealOpenFailed);
    }

    #[test]
    fn grant_key_flag_accepts_standard_and_url_safe_base64() {
        let key = SigningKey::from_bytes(&[0x07; 32]).verifying_key();
        let std_b64 = STANDARD.encode(key.as_bytes());
        let url_b64 = URL_SAFE_NO_PAD.encode(key.as_bytes());
        assert!(
            std_b64.ends_with('='),
            "exercise padding surviving the `=` split"
        );
        for arg in [
            format!("7={std_b64}"),
            format!("7={url_b64}"),
            format!(" 7 = {std_b64} "),
        ] {
            assert_eq!(parse_grant_key(&arg).unwrap(), (7, key), "{arg:?}");
        }
        for bad in [
            std_b64.clone(),
            format!("x={std_b64}"),
            format!("07={std_b64}"),
            format!("7={}", &std_b64[..40]),
            "7=".into(),
        ] {
            assert!(parse_grant_key(&bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn flags_are_both_or_neither_and_kids_are_unique() {
        let dir = tempfile::tempdir().unwrap();
        let seal_key = dir.path().join("seal.key");
        std::fs::write(&seal_key, STANDARD.encode([0x11; 32])).unwrap();
        let key = format!(
            "1={}",
            STANDARD.encode(
                SigningKey::from_bytes(&[0x07; 32])
                    .verifying_key()
                    .as_bytes()
            )
        );

        assert!(GrantVerifier::from_flags(&[], None).unwrap().is_none());
        assert!(GrantVerifier::from_flags(&[], Some(&seal_key)).is_err());
        assert!(GrantVerifier::from_flags(std::slice::from_ref(&key), None).is_err());
        assert!(GrantVerifier::from_flags(&[key.clone(), key.clone()], Some(&seal_key)).is_err());
        let verifier = GrantVerifier::from_flags(&[key], Some(&seal_key))
            .unwrap()
            .unwrap();
        assert_eq!(verifier.keys.len(), 1);
    }

    #[test]
    fn seal_key_errors_name_the_path_never_the_contents() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("seal.key");
        let not_a_key = "c2VjcmV0LWJ1dC13cm9uZy1sZW5ndGg=";
        std::fs::write(&path, not_a_key).unwrap();
        let err = read_seal_key(&path).err().unwrap();
        assert!(err.contains("seal.key"), "{err}");
        assert!(!err.contains(not_a_key), "{err}");
        assert!(read_seal_key(&dir.path().join("missing")).is_err());
    }
}
