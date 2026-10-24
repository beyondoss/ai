//! The live signing keyring: Ed25519 public keys the control plane publishes at runtime.
//!
//! Config (`signing_keys`) fixes the trusted keys for the life of a process; rotating one that way
//! is a redeploy. This is the other source: KV key `signing_key.{kid}` holds the public key, in the
//! encoding config uses (`verifying_key_from_value`), and `store_watch` feeds this set the way it
//! feeds the deny-set. The trusted set is config ∪ live — `Keyring::verify_with` asks config first
//! and falls through to here, so a kid both define is verified under the config key. A kid in the
//! deny-set (`blackhole.kid.{kid}`) is distrusted from either source.
//!
//! Bounded at [`MAX_SIGNING_KEYS`]: signing keys rotate a few times a year, so a table past that is
//! a control-plane bug, not growth to serve. A `Put` that would exceed it is logged and ignored
//! (a replace of a kid already held is always taken). A KV `Delete`/`Purge` of `signing_key.{kid}`
//! is what frees an entry. A bad value is logged and ignored, and the previous key for that kid, if
//! any, stays.
//!
//! Fail direction: tokens under a live-only kid 401 until the set is read (NATS down at a cold start
//! with no snapshot). Config keys are unaffected. Distrust is the opposite and fail-open like all
//! deny entries, which is why revoking a kid is a deny entry and not a deletion here alone: a delete
//! that the gateway misses (NATS blip) leaves the key trusted, a deny entry is replayed on resume.

use crate::key::{Kid, is_canonical_kid, verifying_key_from_value};
use ed25519_dalek::VerifyingKey;
use std::collections::HashMap;
use tracing::warn;

/// KV key prefix the live signing keys are scanned and watched under.
pub const PREFIX: &str = "signing_key.";

/// Most live signing keys held at once. See the module doc for what frees one.
pub const MAX_SIGNING_KEYS: usize = 64;

#[derive(Debug, Default, Clone)]
pub struct LiveKeys {
    keys: HashMap<Kid, VerifyingKey>,
}

impl LiveKeys {
    pub fn new() -> Self {
        Self::default()
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

    /// Trust `value` as kid `kid`'s public key. Logged and ignored if `value` is not a key or the
    /// set is full and `kid` is new.
    pub fn put(&mut self, kid: Kid, value: &[u8]) {
        self.put_with(kid, value, !cfg!(debug_assertions));
    }

    /// [`Self::put`], with whether the public dev key is refused (release builds) passed in.
    fn put_with(&mut self, kid: Kid, value: &[u8], refuse_dev: bool) {
        let Some(vk) = verifying_key_from_value(value) else {
            warn!(
                kid,
                "ignoring signing_key entry: not a 32-byte Ed25519 public key"
            );
            return;
        };
        if refuse_dev && crate::key::is_public_dev_key(&vk) {
            warn!(
                kid,
                "ignoring signing_key entry: it is the public dev key, which anyone can mint under"
            );
            return;
        }
        if self.keys.len() >= MAX_SIGNING_KEYS && !self.keys.contains_key(&kid) {
            warn!(
                kid,
                max = MAX_SIGNING_KEYS,
                "ignoring signing_key entry: live keyring is full; delete a retired kid"
            );
            return;
        }
        self.keys.insert(kid, vk);
    }

    /// Stop trusting kid `kid` from this source (config, if it names the kid, still does).
    pub fn remove(&mut self, kid: Kid) {
        self.keys.remove(&kid);
    }
}

/// `signing_key.{kid}` → the kid. `None` for any other key, so an unrelated watched key never
/// touches the set; the kid must be spelled as the tokens spell it (see `is_canonical_kid`), or the
/// entry could never match one.
pub fn parse_key(key: &str) -> Option<Kid> {
    let kid = key.strip_prefix(PREFIX)?;
    if !is_canonical_kid(kid) {
        return None;
    }
    kid.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;

    fn pubkey(seed: u8) -> Vec<u8> {
        let sk = ed25519_dalek::SigningKey::from_bytes(&[seed; 32]);
        base64::engine::general_purpose::STANDARD
            .encode(sk.verifying_key().to_bytes())
            .into_bytes()
    }

    #[test]
    fn a_release_ring_ignores_the_public_dev_key() {
        let dev = pubkey(7);
        let mut live = LiveKeys::new();
        live.put_with(1, &dev, true);
        assert!(live.get(1).is_none());
        live.put_with(1, &dev, false);
        assert!(live.get(1).is_some());
        live.put_with(2, &pubkey(8), true);
        assert!(live.get(2).is_some());
    }

    #[test]
    fn parse_key_takes_only_canonical_kids() {
        assert_eq!(parse_key("signing_key.7"), Some(7));
        assert_eq!(parse_key("signing_key.0"), Some(0));
        assert_eq!(parse_key("signing_key.07"), None);
        assert_eq!(parse_key("signing_key.+7"), None);
        assert_eq!(parse_key("signing_key."), None);
        assert_eq!(parse_key("signing_key.x"), None);
        assert_eq!(parse_key("blackhole.7"), None);
    }

    #[test]
    fn put_replaces_and_a_bad_value_keeps_the_old_key() {
        let mut live = LiveKeys::new();
        live.put(1, &pubkey(1));
        let first = *live.get(1).unwrap();
        live.put(1, b"garbage");
        assert_eq!(live.get(1), Some(&first));
        live.put(1, &pubkey(2));
        assert_ne!(live.get(1), Some(&first));
        assert_eq!(live.len(), 1);
        live.remove(1);
        assert!(live.is_empty());
    }

    #[test]
    fn the_cap_refuses_a_new_kid_but_not_a_replace() {
        let mut live = LiveKeys::new();
        for kid in 0..MAX_SIGNING_KEYS as Kid {
            live.put(kid, &pubkey(1));
        }
        live.put(MAX_SIGNING_KEYS as Kid, &pubkey(1));
        assert_eq!(live.len(), MAX_SIGNING_KEYS);
        assert!(live.get(MAX_SIGNING_KEYS as Kid).is_none());
        live.put(0, &pubkey(2));
        assert_eq!(live.len(), MAX_SIGNING_KEYS);
        live.remove(0);
        live.put(MAX_SIGNING_KEYS as Kid, &pubkey(1));
        assert!(live.get(MAX_SIGNING_KEYS as Kid).is_some());
    }
}
