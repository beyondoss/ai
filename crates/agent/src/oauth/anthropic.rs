//! Anthropic (Claude Pro/Max subscription) — PKCE + local callback login, matching
//! `platform.claude.com`'s OAuth client exactly (endpoints, client id, and the `state = verifier`
//! quirk below all come from reading pi's own implementation, not public docs — this flow is gated
//! to pi/Claude Code's registered OAuth client).

use std::time::Duration;

use reqwest::Client;
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

use super::callback_server::CallbackServer;
use super::callbacks::{LoginCallbacks, TextPrompt};
use super::error::{OAuthError, Result};
use super::pkce;

const CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
const AUTHORIZE_URL: &str = "https://claude.ai/oauth/authorize";
const TOKEN_URL: &str = "https://platform.claude.com/v1/oauth/token";
const SCOPES: &str = "org:create_api_key user:profile user:inference user:sessions:claude_code \
     user:mcp_servers user:file_upload";
const CALLBACK_PORT: u16 = 53692;
const CALLBACK_PATH: &str = "/callback";
/// The `redirect_uri` sent to Anthropic is always this literal value, regardless of what
/// `callback_host()` actually binds to — the OAuth client is registered against `localhost`
/// specifically, not whatever address the local listener happens to bind.
const REDIRECT_URI: &str = "http://localhost:53692/callback";
const TOKEN_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
/// Treat a token as due for refresh 5 minutes before the server's own `expires_in` — so a request
/// never races the real expiry.
const EARLY_REFRESH_BUFFER_MS: i64 = 5 * 60 * 1000;

#[derive(Clone)]
pub struct AnthropicCredential {
    pub access: String,
    pub refresh: String,
    pub expires_at_ms: i64,
}

/// Hand-written, not derived: `access`/`refresh` are live OAuth bearer tokens — a derived `Debug`
/// would print them verbatim into any future `tracing::debug!(?credential)` or error-context wrap.
/// Matches `auth_store.rs`'s `Secret::fmt` redaction convention (the on-disk representation), just
/// applied to this in-memory struct's own fields instead.
impl std::fmt::Debug for AnthropicCredential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicCredential")
            .field("access", &"<redacted>")
            .field("refresh", &"<redacted>")
            .field("expires_at_ms", &self.expires_at_ms)
            .finish()
    }
}

/// Bind address for the local callback listener — overridable for a remote/container setup where
/// the browser and this process aren't on the same host loopback. Only affects the *bind* address;
/// the `redirect_uri` sent to Anthropic is always `localhost` (see [`REDIRECT_URI`]).
fn callback_host() -> String {
    std::env::var("AI_AGENT_OAUTH_CALLBACK_HOST").unwrap_or_else(|_| "127.0.0.1".to_string())
}

/// Log in via PKCE + a local callback listener, racing a manual paste-the-code/URL fallback.
/// Anthropic has no bind-failure fallback (unlike OpenAI Codex's browser flow) — a port-in-use error
/// propagates as a hard failure, matching pi's own behavior.
pub async fn login(
    callbacks: &dyn LoginCallbacks,
    cancel: &CancellationToken,
) -> Result<AnthropicCredential> {
    let pkce = pkce::generate()?;
    // Anthropic's own quirk: `state` is the raw PKCE verifier, not an independently generated CSRF
    // token (contrast OpenAI Codex's browser flow, which does generate one — see `openai_codex.rs`).
    // Do not "fix" this into a shared helper that assumes one or the other; it's provider-specific.
    let state = pkce.verifier.clone();

    let authorize_url = build_authorize_url(&pkce.challenge, &state);
    callbacks.show_auth_url(&authorize_url, None).await;

    let server = CallbackServer::bind(&callback_host(), CALLBACK_PORT)?;
    let server_cancel = cancel.child_token();
    let code = tokio::select! {
        server_code = server.wait_for_callback(CALLBACK_PATH, state.clone(), server_cancel.clone()) => {
            server_code.ok_or(OAuthError::LoginCancelled)?
        }
        pasted = callbacks.prompt_text(&TextPrompt {
            message: "Paste the code or the full redirect URL from the browser:",
            placeholder: None,
        }) => {
            server_cancel.cancel(); // stop listening, free the port
            let parsed = parse_authorization_input(&pasted?)?;
            // Unlike the callback server (which silently keeps listening on a mismatch), the manual
            // path fails immediately on a state mismatch — a deliberate asymmetry, not a bug.
            check_state(parsed.state.as_deref(), &state)?;
            parsed.code
        }
    };

    exchange_code(&pkce.verifier, &code, &state).await
}

/// Refresh an existing credential. Anthropic's refresh request omits `scope` entirely — sending one
/// is a real regression (pi's own tests assert its absence).
pub async fn refresh(refresh_token: &str) -> Result<AnthropicCredential> {
    let http = Client::new();
    let body = serde_json::json!({
        "grant_type": "refresh_token",
        "client_id": CLIENT_ID,
        "refresh_token": refresh_token,
    });
    normalize(post_token(&http, &body).await?)
}

async fn exchange_code(code_verifier: &str, code: &str, state: &str) -> Result<AnthropicCredential> {
    let http = Client::new();
    let body = serde_json::json!({
        "grant_type": "authorization_code",
        "client_id": CLIENT_ID,
        "code": code,
        "state": state,
        "redirect_uri": REDIRECT_URI,
        "code_verifier": code_verifier,
    });
    normalize(post_token(&http, &body).await?)
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: String,
    expires_in: i64,
}

async fn post_token(http: &Client, body: &serde_json::Value) -> Result<TokenResponse> {
    let resp = http
        .post(TOKEN_URL)
        .timeout(TOKEN_REQUEST_TIMEOUT)
        .header("accept", "application/json")
        .json(body)
        .send()
        .await
        .map_err(|source| OAuthError::Network {
            url: TOKEN_URL.to_string(),
            source,
        })?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(OAuthError::Http {
            provider: "anthropic",
            status: status.as_u16(),
            body,
        });
    }
    resp.json::<TokenResponse>().await.map_err(|source| OAuthError::Network {
        url: TOKEN_URL.to_string(),
        source,
    })
}

fn normalize(tokens: TokenResponse) -> Result<AnthropicCredential> {
    let now = now_ms();
    // Checked arithmetic throughout — this workspace runs with `overflow-checks = true`, and a
    // pathological `expires_in` from a buggy/malicious endpoint must degrade to "already expired,"
    // never panic.
    let expires_at_ms = now
        .checked_add(tokens.expires_in.saturating_mul(1000))
        .and_then(|ms| ms.checked_sub(EARLY_REFRESH_BUFFER_MS))
        .unwrap_or(now);
    Ok(AnthropicCredential {
        access: tokens.access_token,
        refresh: tokens.refresh_token,
        expires_at_ms,
    })
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn build_authorize_url(challenge: &str, state: &str) -> String {
    #[allow(clippy::expect_used)] // AUTHORIZE_URL is a fixed, valid constant; parsing cannot fail
    let mut url = url::Url::parse(AUTHORIZE_URL).expect("static URL is valid");
    url.query_pairs_mut()
        .append_pair("code", "true")
        .append_pair("client_id", CLIENT_ID)
        .append_pair("response_type", "code")
        .append_pair("redirect_uri", REDIRECT_URI)
        .append_pair("scope", SCOPES)
        .append_pair("code_challenge", challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("state", state);
    url.to_string()
}

struct ParsedAuth {
    code: String,
    state: Option<String>,
}

/// Accepts a bare code, `code=...&state=...`, `code#state`, or a full redirect URL — matching pi's
/// own manual-paste input shapes exactly. Duplicated (not shared) in `openai_codex.rs`, which needs
/// the identical parsing logic for its own browser flow's manual-paste fallback — pi itself keeps
/// this per-file rather than factored out, and this port follows suit.
fn parse_authorization_input(input: &str) -> Result<ParsedAuth> {
    let trimmed = input.trim();
    if let Ok(url) = url::Url::parse(trimmed) {
        return from_pairs(url.query_pairs().map(|(k, v)| (k.into_owned(), v.into_owned())));
    }
    if let Some((code, state)) = trimmed.split_once('#') {
        return Ok(ParsedAuth {
            code: code.to_string(),
            state: Some(state.to_string()),
        });
    }
    if trimmed.contains('=') {
        return from_pairs(
            url::form_urlencoded::parse(trimmed.as_bytes())
                .map(|(k, v)| (k.into_owned(), v.into_owned())),
        );
    }
    Ok(ParsedAuth {
        code: trimmed.to_string(),
        state: None,
    })
}

fn from_pairs(pairs: impl Iterator<Item = (String, String)>) -> Result<ParsedAuth> {
    let mut code = None;
    let mut state = None;
    for (k, v) in pairs {
        match k.as_str() {
            "code" => code = Some(v),
            "state" => state = Some(v),
            _ => {}
        }
    }
    Ok(ParsedAuth {
        code: code.ok_or(OAuthError::MissingAuthorizationCode)?,
        state,
    })
}

fn check_state(state: Option<&str>, expected: &str) -> Result<()> {
    match state {
        Some(state) if state != expected => Err(OAuthError::StateMismatch),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_authorize_url_includes_all_required_params() {
        let url = build_authorize_url("chal123", "state456");
        assert!(url.starts_with(AUTHORIZE_URL));
        assert!(url.contains("client_id=9d1c250a-e61b-44d9-88ed-5944d1962f5e"));
        assert!(url.contains("response_type=code"));
        assert!(url.contains("code_challenge=chal123"));
        assert!(url.contains("code_challenge_method=S256"));
        assert!(url.contains("state=state456"));
        assert!(url.contains("redirect_uri=http%3A%2F%2Flocalhost%3A53692%2Fcallback"));
    }

    #[test]
    fn parses_a_bare_code() {
        let parsed = parse_authorization_input("  abc123  ").unwrap();
        assert_eq!(parsed.code, "abc123");
        assert_eq!(parsed.state, None);
    }

    #[test]
    fn parses_code_hash_state_shorthand() {
        let parsed = parse_authorization_input("abc123#state456").unwrap();
        assert_eq!(parsed.code, "abc123");
        assert_eq!(parsed.state.as_deref(), Some("state456"));
    }

    #[test]
    fn parses_a_query_string_shorthand() {
        let parsed = parse_authorization_input("code=abc123&state=state456").unwrap();
        assert_eq!(parsed.code, "abc123");
        assert_eq!(parsed.state.as_deref(), Some("state456"));
    }

    #[test]
    fn parses_a_full_redirect_url() {
        let parsed =
            parse_authorization_input("http://localhost:53692/callback?code=abc123&state=state456")
                .unwrap();
        assert_eq!(parsed.code, "abc123");
        assert_eq!(parsed.state.as_deref(), Some("state456"));
    }

    #[test]
    fn missing_code_is_an_error() {
        assert!(matches!(
            parse_authorization_input("state=onlystate"),
            Err(OAuthError::MissingAuthorizationCode)
        ));
    }

    #[test]
    fn check_state_passes_when_state_is_absent() {
        // A bare code carries no state at all — nothing to check, must not fail.
        check_state(None, "expected").unwrap();
    }

    #[test]
    fn check_state_fails_fast_on_a_mismatch() {
        assert!(matches!(
            check_state(Some("wrong"), "expected"),
            Err(OAuthError::StateMismatch)
        ));
    }

    #[test]
    fn normalize_applies_the_five_minute_early_refresh_buffer() {
        let now = now_ms();
        let cred = normalize(TokenResponse {
            access_token: "a".to_string(),
            refresh_token: "r".to_string(),
            expires_in: 3600,
        })
        .unwrap();
        let expected = now + 3600 * 1000 - EARLY_REFRESH_BUFFER_MS;
        assert!(
            (cred.expires_at_ms - expected).abs() < 1000,
            "expected ~{expected}, got {}",
            cred.expires_at_ms
        );
    }

    #[test]
    fn normalize_degrades_to_already_expired_on_a_pathological_expires_in_instead_of_panicking() {
        let cred = normalize(TokenResponse {
            access_token: "a".to_string(),
            refresh_token: "r".to_string(),
            expires_in: i64::MAX,
        })
        .unwrap();
        assert!(cred.expires_at_ms <= now_ms());
    }

    #[test]
    fn debug_redacts_the_access_and_refresh_tokens_but_keeps_expires_at_ms() {
        let cred = AnthropicCredential {
            access: "sk-live-access-secret".to_string(),
            refresh: "sk-live-refresh-secret".to_string(),
            expires_at_ms: 123456,
        };
        let debug = format!("{cred:?}");
        assert!(!debug.contains("sk-live-access-secret"));
        assert!(!debug.contains("sk-live-refresh-secret"));
        assert!(debug.contains("<redacted>"));
        assert!(debug.contains("123456"));
    }
}
