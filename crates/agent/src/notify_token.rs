//! The `bjn_v1` notify token: the bearer credential a sandbox presents when it tells a replica that
//! a kept job's state changed (exec v1.4, "The notify token (`bjn_v1`)" in [`crate::exec_endpoint`]).
//!
//! This module is the only way to make or check one. Nothing else in the crate builds the MAC
//! input, holds the derived key or compares a tag.
//!
//! ```text
//! token     "bjn_v1." kid "." mac
//! kid       8 lowercase hex digits = hex(SHA-256("bjn_v1 kid" ‖ K)[0..4])
//! mac       base64url without padding of the 32-byte tag (43 characters), canonical
//! K         HKDF-SHA256(ikm = S, salt = empty, info = "bjn_v1 notify"), 32 bytes
//! S         the fleet seal secret, exactly as given, before X25519 clamps it
//! tag       HMAC-SHA256(K, m)
//! m         lp("bjn_v1") ‖ lp(kid) ‖ lp(purpose) ‖ lp(tenant) ‖ lp(session id) ‖ lp(scope) ‖ lp(job id)
//! lp(x)     len(x) as a 4-byte big-endian unsigned integer, then x's bytes
//! purpose   "job_ended"
//! ```
//!
//! - **Mints under one key, verifies under at most two.** A replica mints under the key derived
//!   from its current seal secret only. During a rotation it may also hold the previous secret's
//!   key ([`NotifyKey::with_previous`]) and accepts tokens naming that kid, so a kept job's
//!   notification survives the switch; the kid picks the key, so the two are never tried in turn.
//!   A token naming any other kid is [`Verdict::UnknownKid`] (answered `503`, so a sandbox keeps
//!   retrying through a seal-secret rollout). A token has no time in it, so a job kept before the
//!   switch holds its old-kid token for as long as it runs. The operator drops the previous secret
//!   after 24 h (routine) or one grant life (5 min) after the minter switched (urgent); a job that
//!   ends after that gets `503`, retries for 723 s and gives up, and the session learns of the end
//!   when it next reads its jobs or reopens.
//! - **Its own scope only.** [`NotifyKey::mint`] takes the tenant, the session and the job, and
//!   computes the scope itself ([`JobScope::for_session`]), so a replica cannot mint for another
//!   session's jobs.
//! - **Nothing from the token.** [`NotifyKey::verify`] builds `m` from the caller's tenant,
//!   session, scope and job id (the URL's and the body's) and takes only the kid and the tag from
//!   the token.
//! - **Constant time, in full.** The tag is compared by `hmac::Mac::verify_slice`, never by `==`,
//!   and only a full 32-byte tag parses at all.
//! - **No time, nonce or replica.** A token is a function of the key and (tenant, session, scope,
//!   job id) alone, so every replica holding the seal secret mints the same token for the same job.

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use hkdf::Hkdf;
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};
use x25519_dalek::StaticSecret;
use zeroize::Zeroizing;

use crate::exec_endpoint::jobs::{JobId, JobScope};

type HmacSha256 = Hmac<Sha256>;

/// The token's prefix and version, and the first field of `m`.
const VERSION: &str = "bjn_v1";
/// The HKDF `info` that separates `K` from every other use of the seal secret.
const HKDF_INFO: &[u8] = b"bjn_v1 notify";
/// The domain label of the kid's fingerprint.
const KID_LABEL: &[u8] = b"bjn_v1 kid";
/// The one purpose a notify token is minted for.
const PURPOSE: &str = "job_ended";
/// Characters in a kid: 4 bytes in lowercase hex.
const KID_LEN: usize = 8;
/// Characters in a canonical unpadded base64url encoding of a 32-byte tag.
const MAC_LEN: usize = 43;

/// What [`NotifyKey::verify`] concluded about a token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Minted under this replica's key for exactly this tenant, session, scope and job.
    Valid,
    /// Not exactly the grammar: another prefix or version, a field count other than three, a kid
    /// that is not eight lowercase hex digits, or a mac that is not the canonical 43-character
    /// base64url of 32 bytes.
    Malformed,
    /// Well formed, under a kid this replica does not hold (a seal secret rotated away).
    UnknownKid,
    /// Under this replica's kid, with a tag that does not match.
    BadMac,
}

impl Verdict {
    /// The HTTP status a callback carrying a token with this verdict is refused with; `None` for
    /// [`Verdict::Valid`], which is not a refusal.
    pub fn refusal_status(self) -> Option<u16> {
        match self {
            Self::Valid => None,
            Self::Malformed | Self::BadMac => Some(401),
            Self::UnknownKid => Some(503),
        }
    }
}

/// The notify key `K` and its kid, derived once from the fleet seal secret, plus (during a
/// rotation) the same pair from the previous seal secret, which verifies but never mints.
pub struct NotifyKey {
    current: Derived,
    previous: Option<Derived>,
}

/// One seal secret's `K` and kid.
struct Derived {
    key: Zeroizing<[u8; 32]>,
    kid: String,
}

impl std::fmt::Debug for NotifyKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NotifyKey")
            .field("kid", &self.kid())
            .finish_non_exhaustive()
    }
}

impl NotifyKey {
    /// Derive from the fleet seal secret: the 32 bytes `--seal-key` decodes to, as given. X25519
    /// clamps a scalar when it is used, never when it is stored, so `to_bytes` is exactly them.
    pub fn from_seal_secret(secret: &StaticSecret) -> Self {
        Self::from_secret_bytes(&Zeroizing::new(secret.to_bytes()))
    }

    /// Derive from the seal secret's raw bytes.
    pub fn from_secret_bytes(secret: &[u8; 32]) -> Self {
        Self {
            current: Derived::from_secret_bytes(secret),
            previous: None,
        }
    }

    /// Also accept tokens minted under the previous seal secret's key. Never used to mint. A
    /// previous secret equal to the current one adds nothing.
    #[must_use]
    pub fn with_previous(mut self, previous: &StaticSecret) -> Self {
        let previous = Derived::from_secret_bytes(&Zeroizing::new(previous.to_bytes()));
        self.previous = (previous.kid != self.current.kid).then_some(previous);
        self
    }
}

impl Derived {
    // SAFETY (expect): HKDF-SHA256 fails only for an output longer than 255 × 32 bytes.
    #[allow(clippy::expect_used)]
    fn from_secret_bytes(secret: &[u8; 32]) -> Self {
        let mut key = Zeroizing::new([0u8; 32]);
        Hkdf::<Sha256>::new(None, secret)
            .expand(HKDF_INFO, key.as_mut())
            .expect("32 bytes is a valid HKDF-SHA256 output length");
        let mut h = Sha256::new();
        h.update(KID_LABEL);
        h.update(key.as_ref());
        let kid = hex::encode(&h.finalize()[..KID_LEN / 2]);
        Self { key, kid }
    }
}

impl NotifyKey {
    /// The kid tokens are minted under: eight lowercase hex digits.
    pub fn kid(&self) -> &str {
        &self.current.kid
    }

    /// The token for job `id` of `session_id` in `tenant`, bound to that session's own scope.
    pub fn mint(&self, tenant: &str, session_id: &str, id: &JobId) -> String {
        let scope = JobScope::for_session(Some(tenant), session_id);
        let tag = self
            .current
            .tag(tenant, session_id, scope.name(), id.as_str());
        let mut token = String::with_capacity(VERSION.len() + 2 + KID_LEN + MAC_LEN);
        token.push_str(VERSION);
        token.push('.');
        token.push_str(&self.current.kid);
        token.push('.');
        URL_SAFE_NO_PAD.encode_string(tag, &mut token);
        token
    }

    /// Check `token` against the request it came with: the URL's `tenant` and `session_id`, and
    /// the body's `scope` and `id`. Nothing but the kid and the tag is taken from the token.
    pub fn verify(
        &self,
        token: &str,
        tenant: &str,
        session_id: &str,
        scope: &str,
        id: &str,
    ) -> Verdict {
        let Some((kid, tag)) = parse(token) else {
            return Verdict::Malformed;
        };
        let Some(key) = [Some(&self.current), self.previous.as_ref()]
            .into_iter()
            .flatten()
            .find(|d| d.kid == kid)
        else {
            return Verdict::UnknownKid;
        };
        // `m` carries the key's kid, which the lookup above made equal to the token's.
        match key.mac(tenant, session_id, scope, id).verify_slice(&tag) {
            Ok(()) => Verdict::Valid,
            Err(_) => Verdict::BadMac,
        }
    }
}

impl Derived {
    /// HMAC-SHA256(K, m), finalized.
    fn tag(&self, tenant: &str, session_id: &str, scope: &str, id: &str) -> [u8; 32] {
        self.mac(tenant, session_id, scope, id)
            .finalize()
            .into_bytes()
            .into()
    }

    /// An HMAC over `m`, not yet finalized.
    // SAFETY (expect): HMAC accepts a key of any length.
    #[allow(clippy::expect_used)]
    fn mac(&self, tenant: &str, session_id: &str, scope: &str, id: &str) -> HmacSha256 {
        let mut mac = HmacSha256::new_from_slice(self.key.as_ref()).expect("HMAC takes any key");
        for field in [
            VERSION,
            self.kid.as_str(),
            PURPOSE,
            tenant,
            session_id,
            scope,
            id,
        ] {
            let len = u32::try_from(field.len()).unwrap_or(u32::MAX);
            mac.update(&len.to_be_bytes());
            mac.update(field.as_bytes());
        }
        mac
    }
}

/// The kid and the tag of a token that is exactly the grammar, else `None` (malformed).
fn parse(token: &str) -> Option<(&str, [u8; 32])> {
    let mut fields = token.split('.');
    let (Some(version), Some(kid), Some(mac), None) =
        (fields.next(), fields.next(), fields.next(), fields.next())
    else {
        return None;
    };
    if version != VERSION
        || kid.len() != KID_LEN
        || !kid.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
        || mac.len() != MAC_LEN
        || !mac
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return None;
    }
    let mut tag = [0u8; 32];
    match URL_SAFE_NO_PAD.decode_slice(mac, &mut tag) {
        Ok(32) => {}
        _ => return None,
    }
    // Canonical: the last character's padding bits are zero, so the tag re-encodes to itself.
    (URL_SAFE_NO_PAD.encode(tag) == mac).then_some((kid, tag))
}
