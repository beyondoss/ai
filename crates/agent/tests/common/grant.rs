//! A `bsg_v1` session-grant minter for tests — **independent of `src/grant.rs`**.
//!
//! The verifier's own unit tests already pin the golden vectors, but they do it with a minter that
//! lives beside the code it checks: a mistake shared by both halves would cancel out. This one is
//! written from the spec in ARCHITECTURE.md ("Session grant (`bsg_v1`)") and proves itself the same
//! way the control plane's Go minter has to —
//! [`mint_reproduces_the_golden_vector`](../serve_service_grant.rs) re-mints the fixture's inputs and
//! compares the token byte for byte. So a grant this module produces is a grant the shipped verifier
//! must accept, and the integration tests below it are testing the server rather than a shared
//! misunderstanding.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use base64::Engine as _;
use base64::engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD};
use chacha20poly1305::aead::{AeadInPlace, KeyInit};
use chacha20poly1305::{Key, XChaCha20Poly1305, XNonce};
use ed25519_dalek::{Signer, SigningKey};
use hkdf::Hkdf;
use sha2::Sha256;
use x25519_dalek::{PublicKey, StaticSecret};

/// The signed claims, in wire order.
pub struct Claims {
    pub tenant: String,
    pub session_id: String,
    pub home_shard: String,
    pub workspace_root: String,
    pub exec_url: String,
    pub mcp: Vec<(String, String)>,
    pub exp: u64,
}

/// The sealed secrets, in wire order. `dek` is 32 raw bytes.
pub struct Secrets {
    pub exec_headers: Vec<(String, String)>,
    pub mcp_headers: BTreeMap<String, Vec<(String, String)>>,
    pub gateway_key: String,
    pub dek: [u8; 32],
}

impl Default for Secrets {
    fn default() -> Self {
        Self {
            exec_headers: Vec::new(),
            mcp_headers: BTreeMap::new(),
            gateway_key: "bai_v1.test".into(),
            dek: [7u8; 32],
        }
    }
}

/// A control plane: one Ed25519 signing key under `kid`, and one X25519 fleet key whose secret is
/// written to a `--seal-key` file.
pub struct Minter {
    kid: u32,
    signing: SigningKey,
    fleet_pk: PublicKey,
    seal_key_path: PathBuf,
}

impl Minter {
    /// A deterministic minter (seeds fixed, so a failure is reproducible), with its seal-key file
    /// written under `dir`.
    pub fn new(dir: &Path) -> Self {
        Self::with_keys(dir, 1, [0x21; 32], [0x22; 32])
    }

    pub fn with_keys(dir: &Path, kid: u32, signing_seed: [u8; 32], fleet_secret: [u8; 32]) -> Self {
        let fleet = StaticSecret::from(fleet_secret);
        let seal_key_path = dir.join(format!("seal-{kid}.key"));
        std::fs::write(&seal_key_path, STANDARD.encode(fleet.to_bytes())).unwrap();
        Self {
            kid,
            signing: SigningKey::from_bytes(&signing_seed),
            fleet_pk: PublicKey::from(&fleet),
            seal_key_path,
        }
    }

    /// `--grant-key <kid>=<base64 public key>`.
    pub fn grant_key_flag(&self) -> String {
        format!(
            "{}={}",
            self.kid,
            STANDARD.encode(self.signing.verifying_key().as_bytes())
        )
    }

    /// `--seal-key <path>`.
    pub fn seal_key(&self) -> &Path {
        &self.seal_key_path
    }

    /// Mint a token, with the ephemeral key and nonce fixed by the caller so a vector reproduces.
    pub fn mint_with(
        &self,
        claims: &Claims,
        secrets: &Secrets,
        eph_secret: [u8; 32],
        nonce: [u8; 24],
    ) -> String {
        let plaintext = secrets_json(secrets);
        let eph = StaticSecret::from(eph_secret);
        let eph_pk = PublicKey::from(&eph);
        let mut key = [0u8; 32];
        let mut salt = [0u8; 64];
        salt[..32].copy_from_slice(eph_pk.as_bytes());
        salt[32..].copy_from_slice(self.fleet_pk.as_bytes());
        Hkdf::<Sha256>::new(Some(&salt), eph.diffie_hellman(&self.fleet_pk).as_bytes())
            .expand(b"bsg_v1 seal", &mut key)
            .unwrap();
        let mut ct = plaintext.into_bytes();
        let tag = XChaCha20Poly1305::new(Key::from_slice(&key))
            .encrypt_in_place_detached(
                XNonce::from_slice(&nonce),
                format!("bsg_v1.{}", self.kid).as_bytes(),
                &mut ct,
            )
            .unwrap();
        let blob = [eph_pk.as_bytes(), &nonce[..], &ct, &tag].concat();
        let payload = payload_json(claims, &URL_SAFE_NO_PAD.encode(blob));
        let signed = format!(
            "bsg_v1.{}.{}",
            self.kid,
            URL_SAFE_NO_PAD.encode(payload.as_bytes())
        );
        let sig = self.signing.sign(signed.as_bytes());
        format!("{signed}.{}", URL_SAFE_NO_PAD.encode(sig.to_bytes()))
    }

    /// Mint with a fresh ephemeral key and nonce — what a real minter does per grant.
    pub fn mint(&self, claims: &Claims, secrets: &Secrets) -> String {
        let (mut eph, mut nonce) = ([0u8; 32], [0u8; 24]);
        getrandom::fill(&mut eph).unwrap();
        getrandom::fill(&mut nonce).unwrap();
        self.mint_with(claims, secrets, eph, nonce)
    }
}

/// Compact JSON, keys in wire order (`serde_json::json!` sorts, which would break the vector).
fn payload_json(c: &Claims, sealed: &str) -> String {
    let mcp: Vec<String> = c
        .mcp
        .iter()
        .map(|(name, url)| format!("{{\"name\":{},\"url\":{}}}", q(name), q(url)))
        .collect();
    format!(
        "{{\"tenant\":{},\"session_id\":{},\"home_shard\":{},\"workspace_root\":{},\"exec_url\":{},\
         \"mcp\":[{}],\"exp\":{},\"sealed\":{}}}",
        q(&c.tenant),
        q(&c.session_id),
        q(&c.home_shard),
        q(&c.workspace_root),
        q(&c.exec_url),
        mcp.join(","),
        c.exp,
        q(sealed),
    )
}

fn secrets_json(s: &Secrets) -> String {
    let headers = |hs: &[(String, String)]| {
        hs.iter()
            .map(|(n, v)| format!("{{\"name\":{},\"value\":{}}}", q(n), q(v)))
            .collect::<Vec<_>>()
            .join(",")
    };
    let mcp: Vec<String> = s
        .mcp_headers
        .iter()
        .map(|(name, hs)| format!("{}:[{}]", q(name), headers(hs)))
        .collect();
    format!(
        "{{\"exec_headers\":[{}],\"mcp_headers\":{{{}}},\"gateway_key\":{},\"dek\":{}}}",
        headers(&s.exec_headers),
        mcp.join(","),
        q(&s.gateway_key),
        q(&STANDARD.encode(s.dek)),
    )
}

/// A JSON string literal, escaped the way the spec requires (no HTML escaping).
fn q(s: &str) -> String {
    serde_json::to_string(s).unwrap()
}
