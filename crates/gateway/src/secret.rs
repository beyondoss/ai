//! A string secret that won't leak into logs and is scrubbed on drop.
//!
//! Hygiene, not a hard control: provider keys are necessarily long-lived in RAM (held for the
//! process life, copied into Pingora's request headers we don't own), so zeroize-on-drop only
//! helps at rotation/shutdown. The real protections are SSM-at-rest + never logging + rotation.
//! What this newtype buys: a redacting `Debug` (so a stray `{:?}` or `tracing` field can't print a
//! key) and a best-effort scrub when the value is dropped.

use std::fmt;
use zeroize::Zeroize;

#[derive(Clone)]
pub struct Secret(String);

impl Secret {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Borrow the plaintext. Call sites should keep the exposure as narrow as possible.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl From<String> for Secret {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Secret(***)")
    }
}

// Deserialize transparently from a plain string so config (`AI_POOL_KEY_*`, `nats_creds`) can load a
// secret straight into `Option<Secret>`.
impl<'de> serde::Deserialize<'de> for Secret {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Ok(Self(String::deserialize(d)?))
    }
}

// Serialize **redacting** — same threat model as `Debug`: a stray `serde_json::to_string(&config)`
// in a log line must not leak the key. This is sound for our only serialize path (figment's
// `Serialized::defaults`, where every secret field defaults to `None` and is skipped); a `Secret`
// is for holding a key, never for round-tripping config back out. Read the plaintext via `expose`.
impl serde::Serialize for Secret {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str("***")
    }
}

impl Drop for Secret {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_redacts() {
        let s = Secret::new("sk-supersecret");
        assert_eq!(format!("{s:?}"), "Secret(***)");
        assert!(!format!("{s:?}").contains("supersecret"));
    }

    #[test]
    fn expose_returns_plaintext() {
        assert_eq!(Secret::new("abc").expose(), "abc");
    }
}
