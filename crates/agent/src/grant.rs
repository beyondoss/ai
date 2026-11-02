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
    /// unknown field, a DEK that isn't 32 bytes, a managed connector holding anything but its own
    /// `bmc_v1` token). A minter bug, never an attacker.
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
    /// May this session use Code Mode?
    ///
    /// Off unless the grant says otherwise, so a grant minted before the claim existed — or by an
    /// issuer that doesn't set it — gets today's behaviour. Outside service mode this is the
    /// `--code-mode` flag, which service mode refuses: it is a per-process switch, and on a replica
    /// one tenant's setting must not be everybody's.
    ///
    /// **A replica must accept this claim before any edge emits it.** [`Payload`] is
    /// `deny_unknown_fields`, so a grant carrying a claim a replica doesn't know is refused outright
    /// — agents roll first, the control plane follows. The claim also only means something in an
    /// image built with the `code-mode` feature; without it the interpreter isn't compiled in and the
    /// tool simply isn't registered.
    pub code_mode: bool,
    /// Ceilings on what the session may be configured to do. Absent claim = no ceiling. See
    /// [`Limits`].
    pub limits: Limits,
    /// Where this session's agent definition comes from, when it is a deployment rather than a folder
    /// in the sandbox. See [`Bundle`].
    pub bundle: Option<Bundle>,
    pub secrets: GrantSecrets,
}

/// The agent as a deployment: an immutable plain tar of the agent folder, named by the SHA-256 of its
/// bytes the way an image is named by its digest. The replica fetches `url`, refuses bytes whose hash
/// is not `sha256`, and reads the session's definition (`AGENTS.md`, `.mcp.json`, skills, subagents)
/// from the unpacked copy instead of through the sandbox; see [`crate::bundle`].
///
/// With a bundle, `exec_url` may be empty: the session then has no sandbox and offers no sandbox tools.
///
/// **A replica must accept this claim before any edge emits it** ([`Payload`] is
/// `deny_unknown_fields`). Omitted from the wire when absent, so a grant without it mints byte-for-byte
/// as before and the golden vectors do not move.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[cfg_attr(test, derive(serde::Serialize))]
#[serde(deny_unknown_fields)]
pub struct Bundle {
    pub url: String,
    /// 64 lowercase hex digits.
    pub sha256: String,
}

impl Bundle {
    /// Whether `sha256` is the shape of a digest: it names a cache directory, so nothing else may.
    pub fn is_well_formed(&self) -> bool {
        !self.url.is_empty()
            && self.sha256.len() == 64
            && self
                .sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    }
}

/// Ceilings the grant puts on a session, whatever its `AGENTS.md` or its client asks for. Every field is
/// optional and absent means no ceiling, so a grant minted before the claim existed — or by an issuer
/// that doesn't set it — behaves exactly as it did.
///
/// **A replica must accept this claim before any edge emits it** ([`Payload`] is
/// `deny_unknown_fields`): agents roll first, the control plane follows. Omitted from the wire when
/// absent, so a grant without it mints byte-for-byte as before and the golden vectors do not move.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[cfg_attr(test, derive(serde::Serialize))]
#[serde(deny_unknown_fields)]
pub struct Limits {
    /// Models the session may run: exact ids, or patterns where `*` matches any run of characters
    /// (`claude-sonnet-*`). Checked against the effective model at session start (which covers
    /// `AGENTS.md`, the replica's default and a reattached session's last model), on `set_model`, and
    /// before every prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub models: Option<Vec<String>>,
    /// Tools the session may call. Intersected with whatever the session already allows; a session that
    /// allows everything gets exactly this list.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<String>>,
    /// The most model turns one prompt may take; the lower of this and the session's own cap applies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_steps: Option<u32>,
    /// The most servers a session may add from the workspace or bundle `.mcp.json` (see
    /// [`crate::mcp_json`]); connectors the grant itself carries are not counted. The agent holds a
    /// hard ceiling of [`crate::mcp_json::MAX_HOSTED_CONNECTORS`] whatever this says, and applies it
    /// when this is absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_mcp_json_servers: Option<u32>,
}

impl Limits {
    /// `Ok` if `model` is allowed, else the refusal to show the client. Never swaps the model.
    pub fn check_model(&self, model: &str) -> Result<(), String> {
        match &self.models {
            Some(allowed) if !allowed.iter().any(|p| glob_match(p, model)) => Err(format!(
                "model {model:?} is not allowed by this session's grant limits (allowed: {})",
                allowed.join(", ")
            )),
            _ => Ok(()),
        }
    }

    /// `tools` (the session's allow-list, `None` for every tool) narrowed to the ceiling.
    pub fn narrow_tools(&self, tools: Option<Vec<String>>) -> Option<Vec<String>> {
        match (&self.tools, tools) {
            (None, tools) => tools,
            (Some(ceiling), None) => Some(ceiling.clone()),
            (Some(ceiling), Some(tools)) => {
                Some(tools.into_iter().filter(|t| ceiling.contains(t)).collect())
            }
        }
    }

    /// `steps` (the session's own cap, `None` for unbounded) lowered to the ceiling.
    pub fn cap_steps(&self, steps: Option<u32>) -> Option<u32> {
        match (self.max_steps, steps) {
            (Some(ceiling), Some(steps)) => Some(ceiling.min(steps)),
            (ceiling, steps) => ceiling.or(steps),
        }
    }
}

/// `pattern` against `text`, where `*` matches any run of characters (including none) and everything
/// else matches itself.
fn glob_match(pattern: &str, text: &str) -> bool {
    let mut parts = pattern.split('*');
    let first = parts.next().unwrap_or_default();
    let Some(mut rest) = text.strip_prefix(first) else {
        return false;
    };
    let mut parts = parts.peekable();
    if parts.peek().is_none() {
        return rest.is_empty();
    }
    while let Some(part) = parts.next() {
        if parts.peek().is_none() {
            return rest.ends_with(part);
        }
        match rest.find(part) {
            Some(i) => rest = &rest[i + part.len()..],
            None => return false,
        }
    }
    true
}

/// One MCP server the session may reach. Its credentials, if any, are in
/// [`GrantSecrets::mcp_headers`] under the same `name`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[cfg_attr(test, derive(serde::Serialize))]
#[serde(deny_unknown_fields)]
pub struct McpConnector {
    pub name: String,
    pub url: String,
    /// The server runs inside a sandbox that sleeps when idle. Any connection the agent holds open
    /// to it keeps the VM awake, so the replica never opens the standalone event stream for this
    /// connector and keeps no idle connection pooled between calls (the cost: it hears no
    /// tools-changed notifications). Absent means false, and is omitted when false, so a grant that
    /// does not ask for it mints byte-for-byte as before. **A replica must accept this field before
    /// any minter emits it** ([`Payload`] is `deny_unknown_fields`).
    #[serde(default, skip_serializing_if = "is_false")]
    pub sandbox: bool,
    /// The URL is Beyond's managed-MCP endpoint, which holds the real third-party credential and
    /// injects it upstream, so this session never does. The connector's only credential is a
    /// session-scoped [`bmc_v1`](crate::managed_mcp_token) token, and [`GrantVerifier::verify`]
    /// refuses a grant that gives it anything else. Absent means false; omitted when false. Like
    /// `sandbox`, **a replica must accept it before any minter emits it**.
    #[serde(default, skip_serializing_if = "is_false")]
    pub managed: bool,
    /// A `command` server of the workspace's `.mcp.json`, run as a job in the session's own
    /// sandbox and spoken to over its stdin and stdout ([`crate::tools::mcp_sandbox_stdio`]); `url`
    /// is empty and `sandbox` true. **Never in a grant**: `serde(skip)` leaves it out of what a
    /// minter may send and out of what the verifier reads, so only [`crate::mcp_json`] sets it.
    #[serde(skip)]
    pub command: Option<SandboxCommand>,
}

/// What a `command` entry of `.mcp.json` runs: the program and its arguments, in the sandbox's
/// workspace, with the entry's `env` as the literal strings the file holds (no `$VAR` or `!command`
/// lookup: nothing on the replica is read for it).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxCommand {
    pub command: String,
    pub args: Vec<String>,
    pub env: std::collections::BTreeMap<String, String>,
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
    /// Absent in a grant minted before this claim existed, which must keep verifying — so `default`,
    /// not a required field. The `deny_unknown_fields` reasoning above is about a claim a replica
    /// does not *know*; this is one it knows and the issuer simply didn't set.
    ///
    /// **Omitted when false, never emitted as `false`.** The signature covers the exact claim bytes,
    /// so a grant that does not ask for the capability has to mint byte-for-byte as it did before
    /// this field existed — otherwise every golden vector, and the Go twin they were cross-checked
    /// against, would have to change to add a field that means "as before".
    #[serde(default, skip_serializing_if = "is_false")]
    code_mode: bool,
    /// Absent in a grant minted before this claim existed: `default`, and omitted when absent so such a
    /// grant mints byte-for-byte as before (see [`Limits`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    limits: Option<Limits>,
    /// Absent unless the agent is a deployment; omitted when absent (see [`Bundle`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    bundle: Option<Bundle>,
    sealed: String,
}

/// Whether a managed connector's credentials are exactly one `Authorization: Bearer bmc_v1.…` header
/// whose token names this grant's tenant, session and the connector, and has not expired. Anything
/// else is refused, so a third-party key can't ride along. A replica holds no key for the token's
/// MAC, so this checks what it can: the grammar and the binding; the managed endpoint checks the MAC.
fn holds_only_its_bmc_token(
    payload: &Payload,
    connector: &McpConnector,
    headers: &[SecretHeader],
    now: u64,
) -> bool {
    let [header] = headers else { return false };
    if !header.name.eq_ignore_ascii_case("authorization") {
        return false;
    }
    let Some(token) = header.value.expose().strip_prefix("Bearer ") else {
        return false;
    };
    crate::managed_mcp_token::parse(token).is_some_and(|t| {
        t.claims.tenant == payload.tenant
            && t.claims.session_id == payload.session_id
            && t.claims.connector == connector.name
            && now < t.claims.exp
    })
}

/// `true` for the default of a boolean claim, so `skip_serializing_if` can leave it off the wire.
///
/// Only the test-side `Serialize` uses it — a replica verifies grants, it never mints them — so it is
/// compiled only where it is referenced.
#[cfg(test)]
#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_false(b: &bool) -> bool {
    !*b
}

/// Verifies and unseals `bsg_v1` grants: the trusted Ed25519 keys by kid, plus the fleet's X25519
/// secret. Built once at startup (`serve --grant-key … --seal-key …`) and shared by every session.
pub struct GrantVerifier {
    keys: HashMap<Kid, VerifyingKey>,
    fleet_secret: StaticSecret,
    /// `fleet_secret`'s public half — half of the HKDF salt, so derived once here.
    fleet_public: PublicKey,
    /// The seal secret this fleet rotated away from, during a rotation: a grant's seal and a
    /// notify token's kid are each tried against it after the current secret.
    previous_seal: Option<StaticSecret>,
}

impl GrantVerifier {
    pub fn new(keys: HashMap<Kid, VerifyingKey>, fleet_secret: StaticSecret) -> Self {
        let fleet_public = PublicKey::from(&fleet_secret);
        Self {
            keys,
            fleet_secret,
            fleet_public,
            previous_seal: None,
        }
    }

    /// Also accept grants sealed to, and notify tokens minted under, `previous`: the seal secret
    /// being rotated out.
    #[must_use]
    pub fn with_previous_seal(mut self, previous: StaticSecret) -> Self {
        self.previous_seal = Some(previous);
        self
    }

    /// Build from `serve`'s `--grant-key` (repeatable) and `--seal-key` flags. Neither ⇒ `Ok(None)`:
    /// grants are off and nothing else changes. Only one ⇒ an error, since a grant always carries
    /// sealed secrets — a keyring without the seal key (or the reverse) could accept nothing. So could
    /// a duplicated kid, which is refused rather than resolved last-wins.
    pub fn from_flags(
        grant_keys: &[String],
        seal_key: Option<SealKeySource<'_>>,
    ) -> Result<Option<Self>, String> {
        let seal_key = match (grant_keys.is_empty(), seal_key) {
            (true, None) => return Ok(None),
            (false, Some(source)) => source,
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
        Ok(Some(Self::new(keys, seal_key.load()?)))
    }

    /// The exec job notify key, derived from the fleet seal secret this verifier holds
    /// ([`crate::notify_token`]); it also verifies under the previous seal secret, if one is set.
    pub fn notify_key(&self) -> crate::notify_token::NotifyKey {
        let key = crate::notify_token::NotifyKey::from_seal_secret(&self.fleet_secret);
        match &self.previous_seal {
            Some(previous) => key.with_previous(previous),
            None => key,
        }
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
        for connector in payload.mcp.iter().filter(|c| c.managed) {
            let headers = secrets.mcp_headers.get(&connector.name).map(Vec::as_slice);
            if !holds_only_its_bmc_token(&payload, connector, headers.unwrap_or_default(), now) {
                return Err(GrantError::BadClaims);
            }
        }
        Ok(Grant {
            tenant: payload.tenant,
            session_id: payload.session_id,
            home_shard: payload.home_shard,
            workspace_root: payload.workspace_root,
            exec_url: payload.exec_url,
            mcp: payload.mcp,
            exp: payload.exp,
            code_mode: payload.code_mode,
            limits: payload.limits.unwrap_or_default(),
            bundle: payload.bundle,
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
        // The seal names no key, but the AEAD refuses a wrong one cleanly: during a rotation a
        // grant sealed to the previous secret opens on a second try, from a copy of the blob.
        let Some(previous) = &self.previous_seal else {
            return open_sealed(&self.fleet_secret, &self.fleet_public, aad, &mut blob);
        };
        let mut copy = Zeroizing::new(blob.to_vec());
        match open_sealed(&self.fleet_secret, &self.fleet_public, aad, &mut blob) {
            Err(GrantError::SealOpenFailed) => {
                let public = PublicKey::from(previous);
                open_sealed(previous, &public, aad, &mut copy)
            }
            other => other,
        }
    }
}

/// Open a sealed blob (`eph_pk ‖ nonce ‖ ciphertext ‖ tag`, at least that long) in place under
/// `secret` and parse the plaintext. `public` is `secret`'s public half.
fn open_sealed(
    secret: &StaticSecret,
    public: &PublicKey,
    aad: &[u8],
    blob: &mut [u8],
) -> Result<GrantSecrets, GrantError> {
    let (head, body) = blob.split_at_mut(EPH_LEN + NONCE_LEN);
    let (eph, nonce) = head.split_at(EPH_LEN);
    let eph = PublicKey::from(<[u8; 32]>::try_from(eph).map_err(|_| GrantError::SealMalformed)?);
    let key = derive_seal_key(&secret.diffie_hellman(&eph), &eph, public)
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

/// The environment variable that carries the seal key itself (base64), for a platform that injects a
/// secret as an env var and would otherwise have to write it to a file first.
pub const SEAL_KEY_B64_ENV: &str = "AI_AGENT_SEAL_KEY_B64";

/// Where the fleet's X25519 secret comes from. Exactly one: [`SealKeySource::pick`] refuses both.
#[derive(Clone, Copy)]
pub enum SealKeySource<'a> {
    /// `--seal-key <path>` / `AI_AGENT_SEAL_KEY`.
    File(&'a Path),
    /// `AI_AGENT_SEAL_KEY_B64`: the key itself.
    Base64(&'a str),
}

impl<'a> SealKeySource<'a> {
    /// Choose between the file form and the inline form. Giving both is an error rather than a
    /// precedence rule: two keys that disagree would silently pick which fleet secret this replica
    /// opens grants with, and a stale file left beside a rotated secret is exactly how that happens.
    pub fn pick(file: Option<&'a Path>, b64: Option<&'a str>) -> Result<Option<Self>, String> {
        match (file, b64) {
            (Some(_), Some(_)) => Err(format!(
                "--seal-key (or AI_AGENT_SEAL_KEY) and {SEAL_KEY_B64_ENV} are both set: \
                 give the seal key one way"
            )),
            (Some(p), None) => Ok(Some(Self::File(p))),
            (None, Some(b)) => Ok(Some(Self::Base64(b))),
            (None, None) => Ok(None),
        }
    }

    fn load(self) -> Result<StaticSecret, String> {
        match self {
            Self::File(path) => read_seal_key(path),
            Self::Base64(b64) => parse_seal_key_b64(b64),
        }
    }
}

/// The environment variable that carries the previous seal key (base64), as [`SEAL_KEY_B64_ENV`]
/// carries the current one.
pub const SEAL_KEY_PREVIOUS_B64_ENV: &str = "AI_AGENT_SEAL_KEY_PREVIOUS_B64";

/// Load the previous seal key, given as a file (`--seal-key-previous`) or inline base64
/// ([`SEAL_KEY_PREVIOUS_B64_ENV`]); neither ⇒ `None`, both ⇒ an error, as for the current key.
pub fn load_previous_seal_key(
    file: Option<&Path>,
    b64: Option<&str>,
) -> Result<Option<StaticSecret>, String> {
    match (file, b64) {
        (Some(_), Some(_)) => Err(format!(
            "--seal-key-previous (or AI_AGENT_SEAL_KEY_PREVIOUS) and {SEAL_KEY_PREVIOUS_B64_ENV} \
             are both set: give the previous seal key one way"
        )),
        (Some(path), None) => {
            let contents = Zeroizing::new(
                std::fs::read_to_string(path)
                    .map_err(|e| format!("--seal-key-previous {}: {e}", path.display()))?,
            );
            secret_from(&contents).map(Some).ok_or_else(|| {
                format!(
                    "--seal-key-previous {}: expected base64 of a 32-byte X25519 secret",
                    path.display()
                )
            })
        }
        (None, Some(b64)) => secret_from(b64).map(Some).ok_or_else(|| {
            format!("{SEAL_KEY_PREVIOUS_B64_ENV}: expected base64 of a 32-byte X25519 secret")
        }),
        (None, None) => Ok(None),
    }
}

/// Parse `AI_AGENT_SEAL_KEY_B64`: base64 of the 32-byte X25519 secret, standard or url-safe. The
/// error names the variable and never the value.
pub fn parse_seal_key_b64(value: &str) -> Result<StaticSecret, String> {
    secret_from(value)
        .ok_or_else(|| format!("{SEAL_KEY_B64_ENV}: expected base64 of a 32-byte X25519 secret"))
}

/// Decode into a `StaticSecret`, wiping the intermediate bytes (the secret itself zeroizes on drop).
fn secret_from(encoded: &str) -> Option<StaticSecret> {
    decode_key(encoded).map(|mut bytes| {
        let secret = StaticSecret::from(bytes);
        bytes.zeroize();
        secret
    })
}

/// Read `--seal-key`: a file holding the fleet's X25519 secret as base64 of 32 bytes (standard or
/// url-safe; surrounding whitespace ignored — `wg genkey` writes exactly this). Errors name the path,
/// never the contents.
pub fn read_seal_key(path: &Path) -> Result<StaticSecret, String> {
    let contents = Zeroizing::new(
        std::fs::read_to_string(path).map_err(|e| format!("--seal-key {}: {e}", path.display()))?,
    );
    secret_from(&contents).ok_or_else(|| {
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
        #[serde(default, skip_serializing_if = "is_false")]
        code_mode: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        limits: Option<Limits>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bundle: Option<Bundle>,
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
                code_mode: self.code_mode,
                limits: self.limits,
                bundle: self.bundle,
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
                code_mode: grant.code_mode,
                limits: (grant.limits != Limits::default()).then(|| grant.limits.clone()),
                bundle: grant.bundle.clone(),
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
        GrantVerifier::from_flags(
            std::slice::from_ref(&fx.grant_key_flag),
            Some(SealKeySource::File(&seal_key)),
        )
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

    /// `UPDATE_FIXTURES=1 cargo test -p beyond-agent grant::` rewrites the file (then run
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
    fn a_grant_minted_before_the_code_mode_claim_existed_still_verifies_with_it_off() {
        // The golden vectors predate the claim. They must keep verifying unchanged, and the
        // capability they never asked for must come back off — a new claim that flipped an old
        // grant's behaviour would be a silent capability grant to every tenant already in flight.
        let fx = fixture();
        let grant = verifier(&fx).verify(&fx.token, fx.now).unwrap();
        assert!(
            !grant.code_mode,
            "a grant with no code_mode claim must not confer it"
        );
    }

    #[test]
    fn a_grant_that_asks_for_code_mode_carries_it_and_changes_only_that() {
        // And the claim is load-bearing when it *is* set: same fixture, one field, verified through
        // the real signing path rather than by constructing a `Grant` directly.
        let mut inputs = fixture();
        inputs.claims.code_mode = true;
        let built = build(&inputs);
        let grant = verifier(&built).verify(&built.token, built.now).unwrap();
        assert!(grant.code_mode, "the claim must reach the session");
        // Nothing else moved.
        assert_eq!(grant.tenant, fixture().claims.tenant);
        assert_eq!(grant.session_id, fixture().claims.session_id);
        // And it is genuinely on the wire, not defaulted back in.
        assert!(
            built.payload_json.contains("code_mode"),
            "the signed payload must carry the claim: {}",
            built.payload_json
        );
    }

    #[test]
    fn a_sandbox_connector_flag_defaults_off_and_rides_the_signed_claims_when_set() {
        // The golden vectors predate the flag: their connectors verify with it off.
        let fx = fixture();
        let grant = verifier(&fx).verify(&fx.token, fx.now).unwrap();
        assert!(grant.mcp.iter().all(|c| !c.sandbox));
        assert!(!fx.payload_json.contains("\"sandbox\""));

        let mut inputs = fixture();
        inputs.claims.mcp[1].sandbox = true;
        let built = build(&inputs);
        let grant = verifier(&built).verify(&built.token, built.now).unwrap();
        assert_eq!(
            grant.mcp.iter().map(|c| c.sandbox).collect::<Vec<_>>(),
            [false, true]
        );
        assert!(built.payload_json.contains("\"sandbox\":true"));
    }

    /// The golden grant with its `linear` connector managed and `headers` as that connector's whole
    /// credential set.
    fn managed_inputs(headers: &[(&str, String)]) -> Fixture {
        let mut inputs = fixture();
        inputs.claims.mcp[1].managed = true;
        inputs.secrets.mcp_headers.insert(
            "linear".into(),
            headers
                .iter()
                .map(|(name, value)| SecretHeader {
                    name: (*name).into(),
                    value: Secret::new(value.clone()),
                })
                .collect(),
        );
        inputs
    }

    fn bmc(tenant: &str, session: &str, connector: &str, exp: u64) -> String {
        let key = crate::managed_mcp_token::ManagedKey::from_secret_bytes(&[0x33; 32]);
        key.mint(&crate::managed_mcp_token::Claims {
            tenant: tenant.into(),
            session_id: session.into(),
            connector: connector.into(),
            exp,
        })
    }

    /// A token bound to the golden grant's own tenant and session, for `connector`.
    fn own_bmc(connector: &str) -> String {
        let c = fixture().claims;
        bmc(&c.tenant, &c.session_id, connector, c.exp)
    }

    fn verdict(inputs: &Fixture) -> Result<Grant, GrantError> {
        let built = build(inputs);
        verifier(&built).verify(&built.token, built.now)
    }

    #[test]
    fn a_managed_connector_holding_only_its_own_bmc_token_is_accepted() {
        let inputs = managed_inputs(&[("Authorization", format!("Bearer {}", own_bmc("linear")))]);
        let grant = verdict(&inputs).unwrap();
        assert_eq!(
            grant.mcp.iter().map(|c| c.managed).collect::<Vec<_>>(),
            [false, true]
        );
        // The header name is case-insensitive, as HTTP's is.
        let inputs = managed_inputs(&[("authorization", format!("Bearer {}", own_bmc("linear")))]);
        assert!(verdict(&inputs).is_ok());
    }

    #[test]
    fn a_managed_connector_holding_anything_but_its_own_bmc_token_is_refused() {
        let own = own_bmc("linear");
        let c = fixture().claims;
        let cases: Vec<(&str, Vec<(&str, String)>)> = vec![
            ("no credential at all", vec![]),
            (
                "a third-party token",
                vec![("Authorization", "Bearer lin_real_0000".into())],
            ),
            (
                "the token plus a header that could carry a key",
                vec![
                    ("Authorization", format!("Bearer {own}")),
                    ("X-Workspace", "ws-eng-7f3a".into()),
                ],
            ),
            (
                "two Authorization headers",
                vec![
                    ("Authorization", format!("Bearer {own}")),
                    ("Authorization", "Bearer lin_real_0000".into()),
                ],
            ),
            ("a token in another header", vec![("X-Token", own.clone())]),
            (
                "a non-bearer scheme",
                vec![("Authorization", format!("Basic {own}"))],
            ),
            (
                "a token followed by a key",
                vec![("Authorization", format!("Bearer {own} lin_real_0000"))],
            ),
            (
                "another connector's token",
                vec![("Authorization", format!("Bearer {}", own_bmc("github")))],
            ),
            (
                "another session's token",
                vec![(
                    "Authorization",
                    format!("Bearer {}", bmc(&c.tenant, "sess_other", "linear", c.exp)),
                )],
            ),
            (
                "another tenant's token",
                vec![(
                    "Authorization",
                    format!("Bearer {}", bmc("tenant2", &c.session_id, "linear", c.exp)),
                )],
            ),
            (
                "an expired token",
                vec![(
                    "Authorization",
                    format!("Bearer {}", bmc(&c.tenant, &c.session_id, "linear", 1_000)),
                )],
            ),
        ];
        for (what, headers) in cases {
            assert_eq!(
                verdict(&managed_inputs(&headers)).err(),
                Some(GrantError::BadClaims),
                "{what} must refuse the grant"
            );
        }
    }

    #[test]
    fn a_connector_that_is_not_managed_may_still_carry_a_third_party_key() {
        // Managed is opt-in: the golden grant's own connectors keep verifying as before.
        let fx = fixture();
        assert!(verifier(&fx).verify(&fx.token, fx.now).is_ok());
        assert!(!fx.payload_json.contains("\"managed\""));
    }

    #[test]
    fn a_grant_minted_before_the_limits_claim_existed_has_no_ceiling() {
        let fx = fixture();
        assert!(!fx.payload_json.contains("limits"));
        let grant = verifier(&fx).verify(&fx.token, fx.now).unwrap();
        assert_eq!(grant.limits, Limits::default());
    }

    #[test]
    fn a_grant_that_sets_limits_carries_them_and_changes_only_that() {
        let mut inputs = fixture();
        inputs.claims.limits = Some(Limits {
            models: Some(vec!["claude-sonnet-*".into()]),
            tools: Some(vec!["read".into()]),
            max_steps: Some(7),
            max_mcp_json_servers: Some(3),
        });
        let built = build(&inputs);
        let grant = verifier(&built).verify(&built.token, built.now).unwrap();
        assert_eq!(
            grant.limits.models,
            Some(vec!["claude-sonnet-*".to_string()])
        );
        assert_eq!(grant.limits.tools, Some(vec!["read".to_string()]));
        assert_eq!(grant.limits.max_steps, Some(7));
        assert_eq!(grant.limits.max_mcp_json_servers, Some(3));
        assert!(
            built.payload_json.contains("\"max_mcp_json_servers\":3"),
            "{}",
            built.payload_json
        );
        assert!(
            built.payload_json.contains("\"limits\":{"),
            "{}",
            built.payload_json
        );
        // A field the replica doesn't know inside `limits` is refused like any unknown claim.
        assert!(!built.payload_json.contains("allow_all"));
    }

    #[test]
    fn a_grant_minted_before_the_bundle_claim_existed_has_no_bundle() {
        let fx = fixture();
        assert!(!fx.payload_json.contains("bundle"));
        assert_eq!(
            verifier(&fx).verify(&fx.token, fx.now).unwrap().bundle,
            None
        );
    }

    #[test]
    fn a_bundle_rides_the_signed_claims_and_refuses_unknown_fields() {
        let bundle = Bundle {
            url: "https://bundles.example/agent.tar".into(),
            sha256: "ab".repeat(32),
        };
        let mut inputs = fixture();
        inputs.claims.bundle = Some(bundle.clone());
        let built = build(&inputs);
        let grant = verifier(&built).verify(&built.token, built.now).unwrap();
        assert_eq!(grant.bundle, Some(bundle.clone()));
        assert!(bundle.is_well_formed());
        assert!(
            built.payload_json.contains(
                "\"bundle\":{\"url\":\"https://bundles.example/agent.tar\",\"sha256\":\"abab"
            ),
            "{}",
            built.payload_json
        );
        // A field the replica doesn't know inside `bundle` is refused like any unknown claim.
        let unknown = built
            .payload_json
            .replace("\"sha256\"", "\"mirror\":\"x\",\"sha256\"");
        let signing_key = SigningKey::from_bytes(&hex32(&built.signing_seed_hex));
        let token = sign(&signing_key, built.kid, unknown.as_bytes());
        assert!(matches!(
            verifier(&built).verify(&token, built.now),
            Err(GrantError::BadClaims)
        ));
    }

    #[test]
    fn only_a_lowercase_sha256_digest_names_a_bundle() {
        let ok = |sha256: &str| {
            Bundle {
                url: "https://x.test/a.tar".into(),
                sha256: sha256.into(),
            }
            .is_well_formed()
        };
        assert!(ok(&"0f".repeat(32)));
        for bad in [
            "",
            "abc",
            &"0F".repeat(32),
            &"0g".repeat(32),
            &"0f".repeat(33),
            "../../etc",
        ] {
            assert!(!ok(bad), "{bad}");
        }
    }

    #[test]
    fn a_ceiling_checks_models_narrows_tools_and_lowers_steps() {
        let none = Limits::default();
        assert!(none.check_model("anything").is_ok());
        assert_eq!(none.narrow_tools(None), None);
        assert_eq!(none.cap_steps(Some(9)), Some(9));
        assert_eq!(none.cap_steps(None), None);

        let l = Limits {
            models: Some(vec![
                "claude-test".into(),
                "gpt-*-mini".into(),
                "*-fast".into(),
            ]),
            tools: Some(vec!["read".into(), "bash".into()]),
            max_steps: Some(5),
            max_mcp_json_servers: None,
        };
        for ok in ["claude-test", "gpt-5-mini", "gpt--mini", "x-fast"] {
            assert!(l.check_model(ok).is_ok(), "{ok}");
        }
        for no in ["claude-test2", "gpt-5-mini-x", "gpt-mini", "fast"] {
            let err = l.check_model(no).unwrap_err();
            assert!(err.contains(no) && err.contains("limits"), "{err}");
        }
        assert_eq!(
            l.narrow_tools(None),
            Some(vec!["read".into(), "bash".into()])
        );
        assert_eq!(
            l.narrow_tools(Some(vec!["read".into(), "grep".into()])),
            Some(vec!["read".into()])
        );
        assert_eq!(l.narrow_tools(Some(vec!["grep".into()])), Some(vec![]));
        assert_eq!(l.cap_steps(None), Some(5));
        assert_eq!(l.cap_steps(Some(50)), Some(5));
        assert_eq!(l.cap_steps(Some(2)), Some(2));
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
        assert!(GrantVerifier::from_flags(&[], Some(SealKeySource::File(&seal_key))).is_err());
        assert!(GrantVerifier::from_flags(std::slice::from_ref(&key), None).is_err());
        assert!(
            GrantVerifier::from_flags(
                &[key.clone(), key.clone()],
                Some(SealKeySource::File(&seal_key))
            )
            .is_err()
        );
        let verifier = GrantVerifier::from_flags(&[key], Some(SealKeySource::File(&seal_key)))
            .unwrap()
            .unwrap();
        assert_eq!(verifier.keys.len(), 1);
    }

    #[test]
    fn grant_sealed_to_the_previous_secret_opens_only_with_it_configured() {
        let fx = fixture();
        let base = verifier(&fx);
        let old = secret_from(&fx.seal_key_file).unwrap();
        let rotated = || GrantVerifier::new(base.keys.clone(), StaticSecret::from([0x44; 32]));
        assert_eq!(
            rotated().verify(&fx.token, fx.now).unwrap_err(),
            GrantError::SealOpenFailed
        );
        let both = rotated().with_previous_seal(old);
        let grant = both.verify(&fx.token, fx.now).unwrap();
        assert_eq!(
            serde_json::to_value(&grant.secrets).unwrap(),
            serde_json::to_value(&fx.secrets).unwrap()
        );
        // A grant sealed to neither still fails as SealOpenFailed.
        let other = fx
            .negative
            .iter()
            .find(|c| c.name == "wrong_seal_key")
            .unwrap();
        let neither = GrantVerifier::new(base.keys.clone(), StaticSecret::from([0x55; 32]))
            .with_previous_seal(StaticSecret::from([0x66; 32]));
        assert_eq!(
            neither.verify(&other.token, other.now).unwrap_err(),
            GrantError::SealOpenFailed
        );
    }

    #[test]
    fn previous_seal_key_loads_one_way_only_and_reaches_the_notify_key() {
        let b64 = STANDARD.encode([0x22; 32]);
        assert!(load_previous_seal_key(None, None).unwrap().is_none());
        assert!(load_previous_seal_key(None, Some("nope")).is_err());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prev.key");
        std::fs::write(&path, &b64).unwrap();
        assert!(load_previous_seal_key(Some(&path), Some(&b64)).is_err());
        let prev = load_previous_seal_key(Some(&path), None).unwrap().unwrap();
        assert_eq!(prev.to_bytes(), [0x22; 32]);
        let v = GrantVerifier::new(HashMap::new(), StaticSecret::from([0x11; 32]))
            .with_previous_seal(prev);
        let old = crate::notify_token::NotifyKey::from_secret_bytes(&[0x22; 32]);
        let job = crate::exec_endpoint::jobs::JobId::parse("0123456789abcdef").unwrap();
        let scope = crate::exec_endpoint::jobs::JobScope::for_session(Some("t"), "s");
        let token = old.mint("t", "s", &job);
        assert_eq!(
            v.notify_key()
                .verify(&token, "t", "s", scope.name(), job.as_str()),
            crate::notify_token::Verdict::Valid
        );
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
