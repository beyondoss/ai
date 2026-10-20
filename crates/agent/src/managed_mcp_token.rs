//! The `bmc_v1` managed-connector token: the only credential a hosted session holds for a *managed*
//! MCP connector (`McpConnector::managed` in [`crate::grant`]).
//!
//! A managed connector's URL points at Beyond's managed-MCP endpoint, not at the third party. The
//! endpoint holds the tenant's real third-party credential and injects it upstream; the agent only
//! ever presents this token, so a third-party key is never in a grant, a transcript, a log or the
//! replica's memory. This crate owns the format and the golden vectors
//! (`tests/fixtures/managed_mcp/v1.json`); the control plane mints and the managed endpoint verifies,
//! and both must reproduce the vectors byte for byte.
//!
//! ```text
//! token     "bmc_v1." kid "." claims "." mac
//! kid       8 lowercase hex digits = hex(SHA-256("bmc_v1 kid" ‖ K)[0..4])
//! claims    base64url without padding of  lp(tenant) ‖ lp(session id) ‖ lp(connector) ‖ exp
//! exp       unix seconds, 8 bytes big-endian; the token is valid while now < exp
//! mac       base64url without padding of the 32-byte tag (43 characters), canonical
//! K         HKDF-SHA256(ikm = S, salt = empty, info = "bmc_v1 managed"), 32 bytes
//! S         the managed-MCP secret: 32 bytes shared by the minter and the managed endpoint
//! tag       HMAC-SHA256(K, lp("bmc_v1") ‖ lp(kid) ‖ claims bytes)
//! lp(x)     len(x) as a 4-byte big-endian unsigned integer, then x's bytes
//! ```
//!
//! - **Bound, in the clear.** The tenant, session and connector name are inside the token, so the
//!   endpoint learns them from a token whose MAC checks out, and a replica — which holds no key —
//!   can still check that a token names *its* grant ([`Parsed::claims`]). Nothing in `claims` is
//!   secret: the agent already knows all of it.
//! - **Short-lived.** `exp` is inside the MAC; there is no way to extend a token.
//! - **Replicas cannot mint.** `S` lives with the control plane and the endpoint, never on a
//!   replica, which is why [`ManagedKey::mint`] is not reachable from `serve` and a compromised
//!   replica cannot widen its own session's reach.
//! - **Constant time, in full.** The tag is compared by `hmac::Mac::verify_slice`, and only a full
//!   32-byte canonical tag parses at all.

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use hkdf::Hkdf;
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

type HmacSha256 = Hmac<Sha256>;

/// The token's prefix and version, and the first field of the MAC input.
pub const PREFIX: &str = "bmc_v1";
/// The HKDF `info` that separates `K` from every other use of a secret.
const HKDF_INFO: &[u8] = b"bmc_v1 managed";
/// The domain label of the kid's fingerprint.
const KID_LABEL: &[u8] = b"bmc_v1 kid";
/// Characters in a kid: 4 bytes in lowercase hex.
const KID_LEN: usize = 8;
/// Characters in a canonical unpadded base64url encoding of a 32-byte tag.
const MAC_LEN: usize = 43;
/// Longest `claims` segment accepted; a tenant, session and connector name are short.
const MAX_CLAIMS_LEN: usize = 1024;

/// What a token binds: who it is for and until when. Only trustworthy once the MAC has verified.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Claims {
    pub tenant: String,
    pub session_id: String,
    pub connector: String,
    /// Unix seconds; valid strictly before this.
    pub exp: u64,
}

impl Claims {
    /// The bytes the MAC covers (after the version and kid), and the `claims` segment's content.
    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for field in [&self.tenant, &self.session_id, &self.connector] {
            lp(&mut out, field.as_bytes());
        }
        out.extend_from_slice(&self.exp.to_be_bytes());
        out
    }

    fn decode(bytes: &[u8]) -> Option<Self> {
        let mut rest = bytes;
        let mut field = || -> Option<String> {
            let (len, tail) = rest.split_first_chunk::<4>()?;
            let len = u32::from_be_bytes(*len) as usize;
            if tail.len() < len {
                return None;
            }
            let (value, tail) = tail.split_at(len);
            rest = tail;
            String::from_utf8(value.to_vec()).ok()
        };
        let (tenant, session_id, connector) = (field()?, field()?, field()?);
        let exp = u64::from_be_bytes(*rest.first_chunk::<8>()?);
        (rest.len() == 8).then_some(Self {
            tenant,
            session_id,
            connector,
            exp,
        })
    }
}

fn lp(out: &mut Vec<u8>, field: &[u8]) {
    let len = u32::try_from(field.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(field);
}

/// A token that is exactly the grammar, not yet checked against any key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Parsed<'a> {
    pub kid: &'a str,
    pub claims: Claims,
    claims_bytes: Vec<u8>,
    tag: [u8; 32],
}

/// Why a token was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reject {
    /// Not exactly the grammar: another prefix or version, a field count other than four, a kid that
    /// is not eight lowercase hex digits, claims that do not decode, or a mac that is not the
    /// canonical 43-character base64url of 32 bytes.
    Malformed,
    /// Well formed, under a kid this key does not have (a secret rotated away).
    UnknownKid,
    /// Under this key's kid, with a tag that does not match.
    BadMac,
    /// Authentic, but `now >= exp`.
    Expired,
}

/// Parse `token` without a key. A replica uses this to check what a token *says*; only a holder of
/// the key can say whether it is genuine ([`ManagedKey::verify`]).
pub fn parse(token: &str) -> Option<Parsed<'_>> {
    let mut fields = token.split('.');
    let (Some(version), Some(kid), Some(claims), Some(mac), None) = (
        fields.next(),
        fields.next(),
        fields.next(),
        fields.next(),
        fields.next(),
    ) else {
        return None;
    };
    if version != PREFIX
        || kid.len() != KID_LEN
        || !kid.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
        || claims.len() > MAX_CLAIMS_LEN
        || mac.len() != MAC_LEN
    {
        return None;
    }
    let claims_bytes = URL_SAFE_NO_PAD.decode(claims).ok()?;
    // Canonical: both segments re-encode to themselves.
    if URL_SAFE_NO_PAD.encode(&claims_bytes) != claims {
        return None;
    }
    let mut tag = [0u8; 32];
    match URL_SAFE_NO_PAD.decode_slice(mac, &mut tag) {
        Ok(32) => {}
        _ => return None,
    }
    if URL_SAFE_NO_PAD.encode(tag) != mac {
        return None;
    }
    Some(Parsed {
        kid,
        claims: Claims::decode(&claims_bytes)?,
        claims_bytes,
        tag,
    })
}

/// The managed-MCP key `K` and its kid, derived once from the managed-MCP secret.
pub struct ManagedKey {
    key: Zeroizing<[u8; 32]>,
    kid: String,
}

impl std::fmt::Debug for ManagedKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ManagedKey")
            .field("kid", &self.kid)
            .finish_non_exhaustive()
    }
}

impl ManagedKey {
    /// Derive from the managed-MCP secret's raw bytes.
    // SAFETY (expect): HKDF-SHA256 fails only for an output longer than 255 × 32 bytes.
    #[allow(clippy::expect_used)]
    pub fn from_secret_bytes(secret: &[u8; 32]) -> Self {
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

    /// This key's kid: eight lowercase hex digits.
    pub fn kid(&self) -> &str {
        &self.kid
    }

    /// The token for `claims`.
    pub fn mint(&self, claims: &Claims) -> String {
        let claims_bytes = claims.encode();
        let tag = self.mac(&claims_bytes).finalize().into_bytes();
        let mut token = String::new();
        token.push_str(PREFIX);
        token.push('.');
        token.push_str(&self.kid);
        token.push('.');
        URL_SAFE_NO_PAD.encode_string(&claims_bytes, &mut token);
        token.push('.');
        URL_SAFE_NO_PAD.encode_string(tag, &mut token);
        token
    }

    /// Check `token` and return what it binds. `now` is unix seconds.
    pub fn verify(&self, token: &str, now: u64) -> Result<Claims, Reject> {
        let parsed = parse(token).ok_or(Reject::Malformed)?;
        if parsed.kid != self.kid {
            return Err(Reject::UnknownKid);
        }
        self.mac(&parsed.claims_bytes)
            .verify_slice(&parsed.tag)
            .map_err(|_| Reject::BadMac)?;
        if now >= parsed.claims.exp {
            return Err(Reject::Expired);
        }
        Ok(parsed.claims)
    }

    /// HMAC-SHA256(K, lp(version) ‖ lp(kid) ‖ claims), not yet finalized.
    // SAFETY (expect): HMAC accepts a key of any length.
    #[allow(clippy::expect_used)]
    fn mac(&self, claims_bytes: &[u8]) -> HmacSha256 {
        let mut mac = HmacSha256::new_from_slice(self.key.as_ref()).expect("HMAC takes any key");
        for field in [PREFIX, &self.kid] {
            let len = u32::try_from(field.len()).unwrap_or(u32::MAX);
            mac.update(&len.to_be_bytes());
            mac.update(field.as_bytes());
        }
        mac.update(claims_bytes);
        mac
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE_PATH: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/managed_mcp/v1.json"
    );

    fn claims() -> Claims {
        Claims {
            tenant: "00tenant1".into(),
            session_id: "sess_01J8ZK3Q9V".into(),
            connector: "github".into(),
            exp: 1_900_000_300,
        }
    }

    fn key() -> ManagedKey {
        ManagedKey::from_secret_bytes(&[0x5a; 32])
    }

    /// The golden vectors, regenerated from their inputs: `UPDATE_FIXTURES=1 cargo test -p
    /// beyond-ai-agent managed_mcp_token::` rewrites the file (then run `dprint fmt`). Only for a
    /// deliberate change: the control plane's minter and the managed endpoint have to match.
    fn build() -> serde_json::Value {
        let k = key();
        let c = claims();
        let token = k.mint(&c);
        let parts: Vec<&str> = token.split('.').collect();
        let wrong_kid = format!("{}.{}.{}.{}", parts[0], "00000000", parts[2], parts[3]);
        let other_session = Claims {
            session_id: "sess_OTHER".into(),
            ..c.clone()
        };
        let swapped = format!(
            "{}.{}.{}.{}",
            parts[0],
            parts[1],
            k.mint(&other_session).split('.').nth(2).unwrap(),
            parts[3]
        );
        serde_json::json!({
            "secret_hex": hex::encode([0x5a; 32]),
            "kid": k.kid(),
            "now": 1_900_000_000u64,
            "claims": {
                "tenant": c.tenant, "session_id": c.session_id, "connector": c.connector, "exp": c.exp,
            },
            "claims_hex": hex::encode(c.encode()),
            "token": token,
            "authorization_header": format!("Bearer {token}"),
            "negative": [
                { "name": "expired", "token": token, "now": c.exp, "error": "Expired" },
                { "name": "unknown kid", "token": wrong_kid, "now": 1_900_000_000u64, "error": "UnknownKid" },
                { "name": "claims swapped under another session's mac", "token": swapped, "now": 1_900_000_000u64, "error": "BadMac" },
                { "name": "wrong version", "token": token.replacen("bmc_v1", "bmc_v2", 1), "now": 1_900_000_000u64, "error": "Malformed" },
                { "name": "missing mac", "token": format!("{}.{}.{}", parts[0], parts[1], parts[2]), "now": 1_900_000_000u64, "error": "Malformed" },
            ],
        })
    }

    #[test]
    fn mint_reproduces_the_golden_vectors_byte_for_byte() {
        let built = build();
        if std::env::var_os("UPDATE_FIXTURES").is_some() {
            std::fs::create_dir_all(std::path::Path::new(FIXTURE_PATH).parent().unwrap()).unwrap();
            std::fs::write(
                FIXTURE_PATH,
                serde_json::to_string_pretty(&built).unwrap() + "\n",
            )
            .unwrap();
            return;
        }
        let on_disk: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(FIXTURE_PATH).unwrap()).unwrap();
        assert_eq!(built, on_disk);
    }

    #[test]
    fn every_golden_vector_verifies_to_its_verdict() {
        let fx = build();
        let k = key();
        let token = fx["token"].as_str().unwrap();
        assert_eq!(k.verify(token, 1_900_000_000), Ok(claims()));
        for n in fx["negative"].as_array().unwrap() {
            let want = match n["error"].as_str().unwrap() {
                "Expired" => Reject::Expired,
                "UnknownKid" => Reject::UnknownKid,
                "BadMac" => Reject::BadMac,
                "Malformed" => Reject::Malformed,
                other => panic!("{other}"),
            };
            assert_eq!(
                k.verify(n["token"].as_str().unwrap(), n["now"].as_u64().unwrap()),
                Err(want),
                "{}",
                n["name"]
            );
        }
    }

    #[test]
    fn a_keyless_parse_reads_what_the_token_binds() {
        let token = key().mint(&claims());
        assert_eq!(parse(&token).unwrap().claims, claims());
        assert!(parse(&format!("{token}.x")).is_none());
        assert!(parse(&token.replace("bmc_v1.", "bmc_v1.0")).is_none());
        // A mac whose last character carries padding bits is not canonical.
        // (43 characters carry 258 bits for 256, so a canonical last character ends in two zero bits.)
        let head = &token[..token.len() - 1];
        assert!(parse(&format!("{head}B")).is_none());
    }

    #[test]
    fn a_mac_from_one_key_is_refused_by_another() {
        let other = ManagedKey::from_secret_bytes(&[0x5b; 32]);
        assert_eq!(
            other.verify(&key().mint(&claims()), 0),
            Err(Reject::UnknownKid)
        );
    }
}
