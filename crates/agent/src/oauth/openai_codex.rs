//! OpenAI Codex (ChatGPT Plus/Pro subscription) — dual-mode login: the user picks between a browser
//! PKCE flow (fixed local callback port `1455`) and a custom (non-RFC-8628) device-code flow. Both
//! converge on the same token-exchange endpoint.

use std::time::Duration;

use base64::Engine;
use reqwest::Client;
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

use super::callback_server::CallbackServer;
use super::callbacks::{DeviceCodeInfo, LoginCallbacks, SelectOption, SelectPrompt, TextPrompt};
use super::device_code::{self, DevicePollStep};
use super::error::{OAuthError, Result};
use super::pkce;

const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const AUTHORIZE_URL: &str = "https://auth.openai.com/oauth/authorize";
const TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const BROWSER_CALLBACK_PORT: u16 = 1455;
const BROWSER_CALLBACK_PATH: &str = "/auth/callback";
const BROWSER_REDIRECT_URI: &str = "http://localhost:1455/auth/callback";
const DEVICE_USER_CODE_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/usercode";
const DEVICE_TOKEN_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/token";
const DEVICE_VERIFICATION_URI: &str = "https://auth.openai.com/codex/device";
/// Only used as the `redirect_uri` param on the device flow's *token exchange* call — distinct from
/// the browser flow's [`BROWSER_REDIRECT_URI`].
const DEVICE_REDIRECT_URI: &str = "https://auth.openai.com/deviceauth/callback";
/// Fixed regardless of what the server returns — the device-flow's user-code response carries no
/// expiry field for this provider (unlike Copilot's).
const DEVICE_CODE_TIMEOUT: Duration = Duration::from_secs(900);
const SCOPE: &str = "openid profile email offline_access";
const ACCOUNT_ID_CLAIM_PATH: &str = "https://api.openai.com/auth";

#[derive(Clone)]
pub struct OpenaiCodexCredential {
    pub access: String,
    pub refresh: String,
    pub expires_at_ms: i64,
    pub account_id: String,
}

/// Hand-written, not derived: `access`/`refresh` are live OAuth bearer tokens — a derived `Debug`
/// would print them verbatim into any future `tracing::debug!(?credential)` or error-context wrap.
/// Matches `auth_store.rs`'s `Secret::fmt` redaction convention (the on-disk representation), just
/// applied to this in-memory struct's own fields instead. `account_id` isn't a secret (it's a claim
/// re-derived from the token, not the token itself — see `extract_chatgpt_account_id`'s doc comment)
/// so it prints verbatim.
impl std::fmt::Debug for OpenaiCodexCredential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenaiCodexCredential")
            .field("access", &"<redacted>")
            .field("refresh", &"<redacted>")
            .field("expires_at_ms", &self.expires_at_ms)
            .field("account_id", &self.account_id)
            .finish()
    }
}

fn callback_host() -> String {
    std::env::var("AI_AGENT_OAUTH_CALLBACK_HOST").unwrap_or_else(|_| "127.0.0.1".to_string())
}

/// Log in, letting the user pick browser vs. device-code via `callbacks.select`.
pub async fn login(
    callbacks: &dyn LoginCallbacks,
    cancel: &CancellationToken,
    originator: &str,
) -> Result<OpenaiCodexCredential> {
    let method = callbacks
        .select(&SelectPrompt {
            message: "How would you like to log in to ChatGPT?",
            options: &[
                SelectOption {
                    id: "browser".to_string(),
                    label: "Browser (recommended)".to_string(),
                },
                SelectOption {
                    id: "device_code".to_string(),
                    label: "Device code (headless/remote sessions)".to_string(),
                },
            ],
        })
        .await?
        .ok_or(OAuthError::LoginCancelled)?;

    match method.as_str() {
        "device_code" => login_via_device_code(callbacks, cancel).await,
        _ => login_via_browser(callbacks, cancel, originator).await,
    }
}

async fn login_via_browser(
    callbacks: &dyn LoginCallbacks,
    cancel: &CancellationToken,
    originator: &str,
) -> Result<OpenaiCodexCredential> {
    let pkce = pkce::generate()?;
    // Unlike Anthropic, `state` here is an *independently* random CSRF token, never the PKCE
    // verifier — do not "unify" this with `anthropic.rs`'s authorize-URL building, that would be a
    // real regression for this provider.
    let state = pkce::random_state_hex()?;
    let authorize_url = build_authorize_url(&pkce.challenge, &state, originator);
    callbacks.show_auth_url(&authorize_url, None).await;

    let manual_prompt = manual_paste_prompt();
    let code = match CallbackServer::bind(&callback_host(), BROWSER_CALLBACK_PORT) {
        Ok(server) => {
            let server_cancel = cancel.child_token();
            tokio::select! {
                server_code = server.wait_for_callback(BROWSER_CALLBACK_PATH, state.clone(), server_cancel.clone()) => {
                    server_code.ok_or(OAuthError::LoginCancelled)?
                }
                pasted = callbacks.prompt_text(&manual_prompt) => {
                    server_cancel.cancel();
                    let parsed = parse_authorization_input(&pasted?)?;
                    check_state(parsed.state.as_deref(), &state)?;
                    parsed.code
                }
            }
        }
        // Port already in use (or otherwise unbindable): degrade to manual-paste-only rather than
        // failing the whole login — matches pi's own no-op-fallback-server behavior for this
        // provider. Contrast Anthropic, which has no such fallback and propagates a bind failure.
        Err(_bind_error) => {
            let pasted = callbacks.prompt_text(&manual_prompt).await?;
            let parsed = parse_authorization_input(&pasted)?;
            check_state(parsed.state.as_deref(), &state)?;
            parsed.code
        }
    };

    exchange_code(&pkce.verifier, &code, BROWSER_REDIRECT_URI).await
}

async fn login_via_device_code(
    callbacks: &dyn LoginCallbacks,
    cancel: &CancellationToken,
) -> Result<OpenaiCodexCredential> {
    let http = Client::new();
    let device = request_device_user_code(&http).await?;

    callbacks
        .show_device_code(&DeviceCodeInfo {
            user_code: device.user_code.clone(),
            verification_uri: DEVICE_VERIFICATION_URI.to_string(),
            interval: Some(Duration::from_secs(device.interval.max(1))),
            expires_in: Some(DEVICE_CODE_TIMEOUT),
        })
        .await;

    let interval = Duration::from_secs(device.interval.max(1));
    let (code, code_verifier) =
        device_code::poll_device_code(interval, Some(DEVICE_CODE_TIMEOUT), false, cancel, {
            let http = http.clone();
            let device_auth_id = device.device_auth_id.clone();
            let user_code = device.user_code.clone();
            move || {
                let http = http.clone();
                let device_auth_id = device_auth_id.clone();
                let user_code = user_code.clone();
                async move { poll_device_token_once(&http, &device_auth_id, &user_code).await }
            }
        })
        .await?;

    exchange_code(&code_verifier, &code, DEVICE_REDIRECT_URI).await
}

/// Refresh. OpenAI Codex's own refresh failures must never print/log directly — return the error and
/// let the caller decide (a real regression pi's own tests specifically guard against for this
/// provider).
pub async fn refresh(refresh_token: &str) -> Result<OpenaiCodexCredential> {
    let http = Client::new();
    let params = [
        ("grant_type", "refresh_token"),
        ("refresh_token", refresh_token),
        ("client_id", CLIENT_ID),
    ];
    credential_from_tokens(post_token_form(&http, &params).await?)
}

async fn exchange_code(
    code_verifier: &str,
    code: &str,
    redirect_uri: &str,
) -> Result<OpenaiCodexCredential> {
    let http = Client::new();
    let params = [
        ("grant_type", "authorization_code"),
        ("client_id", CLIENT_ID),
        ("code", code),
        ("code_verifier", code_verifier),
        ("redirect_uri", redirect_uri),
    ];
    credential_from_tokens(post_token_form(&http, &params).await?)
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: String,
    expires_in: i64,
}

async fn post_token_form(http: &Client, params: &[(&str, &str)]) -> Result<TokenResponse> {
    let resp = http
        .post(TOKEN_URL)
        .form(params)
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
            provider: "openai-codex",
            status: status.as_u16(),
            body,
        });
    }
    resp.json().await.map_err(|source| OAuthError::Network {
        url: TOKEN_URL.to_string(),
        source,
    })
}

fn credential_from_tokens(tokens: TokenResponse) -> Result<OpenaiCodexCredential> {
    let account_id = extract_chatgpt_account_id(&tokens.access_token)?;
    // No early-refresh buffer here — unlike Anthropic/Copilot, Codex's `expires_in` is used exactly
    // as given. Checked arithmetic: a pathological `expires_in` degrades to "already expired," never
    // panics (this workspace runs with `overflow-checks = true`).
    let expires_at_ms = now_ms()
        .checked_add(tokens.expires_in.saturating_mul(1000))
        .unwrap_or_else(now_ms);
    Ok(OpenaiCodexCredential {
        access: tokens.access_token,
        refresh: tokens.refresh_token,
        expires_at_ms,
        account_id,
    })
}

/// Re-derive the account id from a *live* access token — the request-time consumer must call this
/// directly rather than trusting a stored `account_id` field, which is login-time convenience only
/// (a token can be refreshed after storage, and the claim must reflect whichever token is actually
/// about to be sent).
pub fn extract_chatgpt_account_id(access_token: &str) -> Result<String> {
    let payload = decode_jwt_payload(access_token)?;
    payload
        .get(ACCOUNT_ID_CLAIM_PATH)
        .and_then(|v| v.get("chatgpt_account_id"))
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .ok_or(OAuthError::MissingJwtClaim {
            claim: "chatgpt_account_id",
        })
}

/// Base64-decodes the JWT's middle (payload) segment and parses it as JSON — no signature
/// verification, since this is the client reading a claim out of its own freshly-issued token, not
/// validating a third party's.
fn decode_jwt_payload(token: &str) -> Result<serde_json::Value> {
    let claim_missing = || OAuthError::MissingJwtClaim {
        claim: "chatgpt_account_id",
    };
    let payload_b64 = token.split('.').nth(1).ok_or_else(claim_missing)?;
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(payload_b64)
        .or_else(|_| base64::engine::general_purpose::STANDARD_NO_PAD.decode(payload_b64))
        .map_err(|_| claim_missing())?;
    serde_json::from_slice(&bytes).map_err(|_| claim_missing())
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn build_authorize_url(challenge: &str, state: &str, originator: &str) -> String {
    #[allow(clippy::expect_used)] // AUTHORIZE_URL is a fixed, valid constant; parsing cannot fail
    let mut url = url::Url::parse(AUTHORIZE_URL).expect("static URL is valid");
    url.query_pairs_mut()
        .append_pair("response_type", "code")
        .append_pair("client_id", CLIENT_ID)
        .append_pair("redirect_uri", BROWSER_REDIRECT_URI)
        .append_pair("scope", SCOPE)
        .append_pair("code_challenge", challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("state", state)
        .append_pair("id_token_add_organizations", "true")
        .append_pair("codex_cli_simplified_flow", "true")
        .append_pair("originator", originator);
    url.to_string()
}

fn manual_paste_prompt() -> TextPrompt<'static> {
    TextPrompt {
        message: "Paste the code or the full redirect URL from the browser:",
        placeholder: None,
    }
}

struct ParsedAuth {
    code: String,
    state: Option<String>,
}

/// Identical logic to `anthropic.rs`'s helper of the same shape — duplicated rather than shared,
/// matching pi's own per-file duplication of this exact parsing (see that module's doc comment).
fn parse_authorization_input(input: &str) -> Result<ParsedAuth> {
    let trimmed = input.trim();
    if let Ok(url) = url::Url::parse(trimmed) {
        return from_pairs(
            url.query_pairs()
                .map(|(k, v)| (k.into_owned(), v.into_owned())),
        );
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

#[derive(Deserialize)]
struct DeviceUserCodeResponse {
    device_auth_id: String,
    user_code: String,
    #[serde(deserialize_with = "deserialize_interval")]
    interval: u64,
}

/// `interval` may arrive as a JSON number or a numeric string — coerce either shape.
fn deserialize_interval<'de, D>(deserializer: D) -> std::result::Result<u64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize as _;
    match serde_json::Value::deserialize(deserializer)? {
        serde_json::Value::Number(n) => n
            .as_u64()
            .ok_or_else(|| serde::de::Error::custom("interval is not a valid unsigned integer")),
        serde_json::Value::String(s) => s.parse().map_err(|_| {
            serde::de::Error::custom("interval string is not a valid unsigned integer")
        }),
        _ => Err(serde::de::Error::custom(
            "interval must be a number or numeric string",
        )),
    }
}

async fn request_device_user_code(http: &Client) -> Result<DeviceUserCodeResponse> {
    let resp = http
        .post(DEVICE_USER_CODE_URL)
        .json(&serde_json::json!({ "client_id": CLIENT_ID }))
        .send()
        .await
        .map_err(|source| OAuthError::Network {
            url: DEVICE_USER_CODE_URL.to_string(),
            source,
        })?;
    let status = resp.status();
    if status.as_u16() == 404 {
        return Err(OAuthError::InvalidResponse {
            provider: "openai-codex",
            detail:
                "device code login is not enabled for this server. Use browser login or verify \
                      the server URL."
                    .to_string(),
        });
    }
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(OAuthError::Http {
            provider: "openai-codex",
            status: status.as_u16(),
            body,
        });
    }
    resp.json().await.map_err(|source| OAuthError::Network {
        url: DEVICE_USER_CODE_URL.to_string(),
        source,
    })
}

async fn poll_device_token_once(
    http: &Client,
    device_auth_id: &str,
    user_code: &str,
) -> Result<DevicePollStep<(String, String)>> {
    #[derive(Deserialize)]
    struct DeviceTokenSuccess {
        authorization_code: String,
        code_verifier: String,
    }

    let resp = http
        .post(DEVICE_TOKEN_URL)
        .json(&serde_json::json!({ "device_auth_id": device_auth_id, "user_code": user_code }))
        .send()
        .await
        .map_err(|source| OAuthError::Network {
            url: DEVICE_TOKEN_URL.to_string(),
            source,
        })?;
    let status = resp.status();
    if status.is_success() {
        let success: DeviceTokenSuccess =
            resp.json().await.map_err(|source| OAuthError::Network {
                url: DEVICE_TOKEN_URL.to_string(),
                source,
            })?;
        return Ok(DevicePollStep::Complete((
            success.authorization_code,
            success.code_verifier,
        )));
    }
    // 403/404 both mean "pending" for this flow — non-standard vs. RFC 8628, but this is the real,
    // tested behavior (see module doc comment).
    if status.as_u16() == 403 || status.as_u16() == 404 {
        return Ok(DevicePollStep::Pending);
    }
    let body: serde_json::Value = resp.json().await.unwrap_or(serde_json::Value::Null);
    let error_code = body
        .get("error")
        .and_then(|e| {
            e.as_str()
                .map(str::to_string)
                .or_else(|| e.get("code").and_then(|c| c.as_str()).map(str::to_string))
        })
        .unwrap_or_default();
    match error_code.as_str() {
        "deviceauth_authorization_pending" => Ok(DevicePollStep::Pending),
        "slow_down" => Ok(DevicePollStep::SlowDown),
        _ => Err(OAuthError::DeviceFlowFailed(body.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_authorize_url_includes_all_required_params() {
        let url = build_authorize_url("chal123", "state456", "beyond-ai-agent");
        assert!(url.starts_with(AUTHORIZE_URL));
        assert!(url.contains("client_id=app_EMoamEEZ73f0CkXaXp7hrann"));
        assert!(url.contains("code_challenge=chal123"));
        assert!(url.contains("state=state456"));
        assert!(url.contains("id_token_add_organizations=true"));
        assert!(url.contains("codex_cli_simplified_flow=true"));
        assert!(url.contains("originator=beyond-ai-agent"));
        assert!(url.contains("redirect_uri=http%3A%2F%2Flocalhost%3A1455%2Fauth%2Fcallback"));
    }

    #[test]
    fn parses_a_full_redirect_url_and_a_bare_code_the_same_as_anthropics_helper() {
        let parsed =
            parse_authorization_input("http://localhost:1455/auth/callback?code=abc&state=s")
                .unwrap();
        assert_eq!(parsed.code, "abc");
        assert_eq!(parsed.state.as_deref(), Some("s"));

        let bare = parse_authorization_input("just-a-code").unwrap();
        assert_eq!(bare.code, "just-a-code");
        assert_eq!(bare.state, None);
    }

    #[test]
    fn check_state_fails_fast_on_mismatch_but_allows_absent_state() {
        check_state(None, "expected").unwrap();
        assert!(matches!(
            check_state(Some("wrong"), "expected"),
            Err(OAuthError::StateMismatch)
        ));
    }

    /// A hand-built (unsigned) JWT: header.payload.signature, base64url-no-pad each segment.
    fn fake_jwt(payload_json: &str) -> String {
        use base64::engine::general_purpose::URL_SAFE_NO_PAD;
        let header = URL_SAFE_NO_PAD.encode(b"{\"alg\":\"none\"}");
        let payload = URL_SAFE_NO_PAD.encode(payload_json.as_bytes());
        format!("{header}.{payload}.sig")
    }

    #[test]
    fn extracts_the_chatgpt_account_id_from_a_live_token() {
        let token =
            fake_jwt(r#"{"https://api.openai.com/auth":{"chatgpt_account_id":"acct_123"}}"#);
        assert_eq!(extract_chatgpt_account_id(&token).unwrap(), "acct_123");
    }

    #[test]
    fn missing_claim_is_an_error_not_a_panic() {
        let token = fake_jwt(r#"{"sub":"user_1"}"#);
        assert!(matches!(
            extract_chatgpt_account_id(&token),
            Err(OAuthError::MissingJwtClaim { .. })
        ));
    }

    #[test]
    fn malformed_token_is_an_error_not_a_panic() {
        assert!(matches!(
            extract_chatgpt_account_id("not-a-jwt"),
            Err(OAuthError::MissingJwtClaim { .. })
        ));
    }

    #[test]
    fn credential_from_tokens_applies_no_early_refresh_buffer() {
        let account_id_token =
            fake_jwt(r#"{"https://api.openai.com/auth":{"chatgpt_account_id":"acct_1"}}"#);
        let now = now_ms();
        let cred = credential_from_tokens(TokenResponse {
            access_token: account_id_token,
            refresh_token: "r".to_string(),
            expires_in: 3600,
        })
        .unwrap();
        let expected = now + 3600 * 1000;
        assert!(
            (cred.expires_at_ms - expected).abs() < 1000,
            "expected ~{expected}, got {}",
            cred.expires_at_ms
        );
    }

    #[test]
    fn deserialize_interval_accepts_a_number_or_a_numeric_string() {
        #[derive(Deserialize)]
        struct Wrapper {
            #[serde(deserialize_with = "deserialize_interval")]
            interval: u64,
        }
        let from_number: Wrapper = serde_json::from_str(r#"{"interval": 5}"#).unwrap();
        assert_eq!(from_number.interval, 5);
        let from_string: Wrapper = serde_json::from_str(r#"{"interval": "7"}"#).unwrap();
        assert_eq!(from_string.interval, 7);
    }

    #[test]
    fn debug_redacts_the_access_and_refresh_tokens_but_keeps_non_secret_fields() {
        let cred = OpenaiCodexCredential {
            access: "sk-live-access-secret".to_string(),
            refresh: "sk-live-refresh-secret".to_string(),
            expires_at_ms: 123456,
            account_id: "acct_visible".to_string(),
        };
        let debug = format!("{cred:?}");
        assert!(!debug.contains("sk-live-access-secret"));
        assert!(!debug.contains("sk-live-refresh-secret"));
        assert!(debug.contains("<redacted>"));
        assert!(debug.contains("123456"));
        assert!(debug.contains("acct_visible"));
    }
}
