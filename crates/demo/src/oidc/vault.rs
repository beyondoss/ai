//! Refresh tokens at rest: sealed under a key that belongs to one tenant.
//!
//! One broker master key (from the environment, never on disk) yields a key per tenant,
//! `HKDF-SHA256(master, info = "broker tenant key v1" ‖ tenant)`. A refresh token is sealed with
//! XChaCha20-Poly1305 under that key, with the connection's id as associated data, so a sealed token
//! cannot be opened under another tenant's key, nor moved to another connection's record. What reaches
//! disk is the nonce and ciphertext, base64url; no key ever does.

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use chacha20poly1305::aead::{Aead, KeyInit, Payload};
use chacha20poly1305::{XChaCha20Poly1305, XNonce};
use hkdf::Hkdf;
use sha2::Sha256;

pub struct Vault {
    master: [u8; 32],
}

const NONCE: usize = 24;

impl Vault {
    pub fn new(master: [u8; 32]) -> Self {
        Self { master }
    }

    fn cipher(&self, tenant: &str) -> Option<XChaCha20Poly1305> {
        let mut info = b"broker tenant key v1".to_vec();
        info.extend_from_slice(tenant.as_bytes());
        let mut key = [0u8; 32];
        Hkdf::<Sha256>::new(None, &self.master)
            .expand(&info, &mut key)
            .ok()?;
        XChaCha20Poly1305::new_from_slice(&key).ok()
    }

    /// Seal `secret` for `tenant`, bound to `aad` (the connection's id).
    pub fn seal(&self, tenant: &str, aad: &str, secret: &[u8]) -> Option<String> {
        let mut nonce = [0u8; NONCE];
        getrandom::fill(&mut nonce).ok()?;
        let ct = self
            .cipher(tenant)?
            .encrypt(
                XNonce::from_slice(&nonce),
                Payload {
                    msg: secret,
                    aad: aad.as_bytes(),
                },
            )
            .ok()?;
        let mut out = nonce.to_vec();
        out.extend(ct);
        Some(URL_SAFE_NO_PAD.encode(out))
    }

    /// Open what `seal` made; `None` for another tenant, another `aad`, or any damage.
    pub fn open(&self, tenant: &str, aad: &str, sealed: &str) -> Option<Vec<u8>> {
        let raw = URL_SAFE_NO_PAD.decode(sealed).ok()?;
        if raw.len() <= NONCE {
            return None;
        }
        let (nonce, ct) = raw.split_at(NONCE);
        self.cipher(tenant)?
            .decrypt(
                XNonce::from_slice(nonce),
                Payload {
                    msg: ct,
                    aad: aad.as_bytes(),
                },
            )
            .ok()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use super::*;

    #[test]
    fn a_sealed_token_opens_only_for_its_tenant_and_its_connection() {
        let v = Vault::new([7; 32]);
        let sealed = v.seal("acme", "conn-1", b"refresh-token-value").unwrap();
        assert!(!sealed.contains("refresh-token-value"));
        assert_eq!(
            v.open("acme", "conn-1", &sealed).unwrap(),
            b"refresh-token-value"
        );
        assert!(
            v.open("globex", "conn-1", &sealed).is_none(),
            "another tenant's key"
        );
        assert!(
            v.open("acme", "conn-2", &sealed).is_none(),
            "another connection's record"
        );
        assert!(
            Vault::new([8; 32])
                .open("acme", "conn-1", &sealed)
                .is_none(),
            "another master"
        );
        let mut damaged = sealed.into_bytes();
        let last = damaged.len() - 2;
        damaged[last] = if damaged[last] == b'A' { b'B' } else { b'A' };
        assert!(
            v.open("acme", "conn-1", &String::from_utf8(damaged).unwrap())
                .is_none()
        );
    }

    #[test]
    fn sealing_twice_gives_different_bytes() {
        let v = Vault::new([7; 32]);
        assert_ne!(
            v.seal("t", "c", b"x").unwrap(),
            v.seal("t", "c", b"x").unwrap()
        );
    }
}
