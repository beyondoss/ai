//! Shared PKCE (RFC 7636) verifier/challenge generation. Used by Anthropic's flow and OpenAI Codex's
//! *browser* flow — Codex's device-code flow does not call this: the token server issues its own
//! verifier for that path (see `openai_codex.rs`).

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use sha2::{Digest, Sha256};

use super::error::Result;

pub struct Pkce {
    pub verifier: String,
    pub challenge: String,
}

/// 32 random bytes, base64url-encoded (no padding), as the verifier; SHA-256 of the verifier's ASCII
/// bytes, base64url-encoded, as the challenge (`code_challenge_method=S256`).
pub fn generate() -> Result<Pkce> {
    let mut raw = [0u8; 32];
    getrandom::fill(&mut raw)?;
    let verifier = URL_SAFE_NO_PAD.encode(raw);
    let challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
    Ok(Pkce { verifier, challenge })
}

/// An independent random CSRF token, hex-encoded — OpenAI Codex's browser flow uses this instead of
/// reusing the PKCE verifier as `state` (unlike Anthropic, which deliberately does reuse it; see
/// `anthropic.rs`'s doc comment on that quirk).
pub fn random_state_hex() -> Result<String> {
    let mut raw = [0u8; 16];
    getrandom::fill(&mut raw)?;
    Ok(hex::encode(raw))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_produces_url_safe_unpadded_strings_of_the_expected_length() {
        let pkce = generate().unwrap();
        // 32 random bytes / SHA-256 digest (32 bytes) base64url-no-pad-encoded is always 43 chars.
        assert_eq!(pkce.verifier.len(), 43);
        assert_eq!(pkce.challenge.len(), 43);
        assert!(
            pkce.verifier
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
            "verifier must be URL-safe: {}",
            pkce.verifier
        );
        assert_ne!(pkce.verifier, pkce.challenge);
    }

    #[test]
    fn generate_is_random_across_calls() {
        let a = generate().unwrap();
        let b = generate().unwrap();
        assert_ne!(a.verifier, b.verifier);
    }

    #[test]
    fn random_state_hex_is_32_hex_chars() {
        let state = random_state_hex().unwrap();
        assert_eq!(state.len(), 32); // 16 bytes hex-encoded
        assert!(state.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
