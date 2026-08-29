//! GitHub Copilot — device-code login. `login()` prompts for an optional GitHub Enterprise domain
//! (Task #4, pi-parity fix — mirrors pi's own `loginGitHubCopilot`'s identical prompt): a blank answer
//! defaults to `github.com`; any other answer normalizes to a bare hostname and threads through every
//! URL the device/token exchange builds, ending up recorded on the stored credential's own
//! `enterprise_url` — the same field `refresh()`/`base_url_from_token()` already accepted for a
//! credential that had one recorded some other way (e.g. hand-edited into `auth.json`) before this fix.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use agent_core::client::{Credential, CredentialSource, DirectRouting, RouteOverride};
use futures::future::join_all;
use reqwest::{Client, RequestBuilder};
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

use super::callbacks::{DeviceCodeInfo, LoginCallbacks};
use super::device_code::{self, DevicePollStep};
use super::error::{OAuthError, Result};

const CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";
const DEVICE_SCOPE: &str = "read:user";
const COPILOT_USER_AGENT: &str = "GitHubCopilotChat/0.35.0";
const COPILOT_EDITOR_VERSION: &str = "vscode/1.107.0";
const COPILOT_EDITOR_PLUGIN_VERSION: &str = "copilot-chat/0.35.0";
const COPILOT_INTEGRATION_ID: &str = "vscode-chat";
const COPILOT_API_VERSION: &str = "2026-06-01";
const EARLY_REFRESH_BUFFER_MS: i64 = 5 * 60 * 1000;

/// Every model id GitHub Copilot may require an explicit per-model "enable" policy call for before
/// it's usable from this account — ported from pi's `GITHUB_COPILOT_MODELS` catalog
/// (`packages/ai/src/providers/github-copilot.models.ts`). Only the ids matter for the login-time
/// enable fan-out (`enable_models`); none of that catalog's other per-model metadata (pricing,
/// context window, headers, …) has a use here. Not consulted for anything past login — actual
/// model *availability* always comes from `fetch_available_models`'s live `/models` response, never
/// from this static list.
pub const KNOWN_MODEL_IDS: &[&str] = &[
    "claude-fable-5",
    "claude-haiku-4.5",
    "claude-opus-4.5",
    "claude-opus-4.6",
    "claude-opus-4.7",
    "claude-opus-4.8",
    "claude-sonnet-4",
    "claude-sonnet-4.5",
    "claude-sonnet-4.6",
    "claude-sonnet-5",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "gemini-3.5-flash",
    "gpt-4.1",
    "gpt-5-mini",
    "gpt-5.2",
    "gpt-5.2-codex",
    "gpt-5.3-codex",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.5",
    "kimi-k2.7-code",
    "mai-code-1-flash-picker",
];

#[derive(Clone)]
pub struct GithubCopilotCredential {
    pub access: String,
    /// The long-lived GitHub device-flow token (`ghu_...`) — unchanged across refreshes; the
    /// short-lived Copilot-internal proxy token (`access`) is re-derived from this every refresh.
    pub refresh: String,
    pub expires_at_ms: i64,
    pub enterprise_url: Option<String>,
    pub available_model_ids: Vec<String>,
}

/// Hand-written, not derived: `access`/`refresh` are live OAuth bearer tokens (the short-lived
/// Copilot-internal proxy token and the long-lived GitHub device-flow token, respectively) — a
/// derived `Debug` would print them verbatim into any future `tracing::debug!(?credential)` or
/// error-context wrap. Matches `auth_store.rs`'s `Secret::fmt` redaction convention (the on-disk
/// representation), just applied to this in-memory struct's own fields instead. `enterprise_url`/
/// `available_model_ids` aren't secrets, so they print verbatim.
impl std::fmt::Debug for GithubCopilotCredential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GithubCopilotCredential")
            .field("access", &"<redacted>")
            .field("refresh", &"<redacted>")
            .field("expires_at_ms", &self.expires_at_ms)
            .field("enterprise_url", &self.enterprise_url)
            .field("available_model_ids", &self.available_model_ids)
            .finish()
    }
}

/// Log in via GitHub's device-code flow, then exchange for a Copilot-internal token, enable every
/// model in `known_model_ids` (best-effort, login-only), and record which ones are actually
/// available.
pub async fn login(
    callbacks: &dyn LoginCallbacks,
    cancel: &CancellationToken,
    known_model_ids: &[&str],
) -> Result<GithubCopilotCredential> {
    let http = Client::new();
    // Task #4 (pi-parity fix): prompt for an optional GitHub Enterprise domain before starting the
    // device flow — mirrors pi's own `loginGitHubCopilot`'s identical `onPrompt`/`normalizeDomain`
    // sequence. `TextPrompt` already exists for exactly this purpose (see its own doc comment naming
    // "a GitHub Enterprise domain" as a use case); this was previously the one login flow that never
    // actually asked, so `enterprise_url` on the stored credential was always `None` regardless of
    // account type.
    let domain_input = callbacks
        .prompt_text(&super::callbacks::TextPrompt {
            message: "GitHub Enterprise URL/domain (blank for github.com)",
            placeholder: Some("company.ghe.com"),
        })
        .await?;
    let (enterprise_url, domain) = resolve_login_domain(&domain_input)?;
    let device = request_device_code(&http, &domain).await?;

    callbacks
        .show_device_code(&DeviceCodeInfo {
            user_code: device.user_code.clone(),
            verification_uri: device.verification_uri.clone(),
            interval: device.interval.map(Duration::from_secs),
            expires_in: Some(Duration::from_secs(device.expires_in)),
        })
        .await;

    let access_token_url = format!("https://{domain}/login/oauth/access_token");
    let interval = Duration::from_secs(device.interval.unwrap_or(5));
    let expires_in = Duration::from_secs(device.expires_in);
    let device_code_value = device.device_code.clone();
    let github_token = device_code::poll_device_code(interval, Some(expires_in), true, cancel, {
        let http = http.clone();
        let access_token_url = access_token_url.clone();
        move || {
            let http = http.clone();
            let access_token_url = access_token_url.clone();
            let device_code_value = device_code_value.clone();
            async move { poll_access_token_once(&http, &access_token_url, &device_code_value).await }
        }
    })
    .await?;

    callbacks.progress("Fetching Copilot token...").await;
    let (copilot_token, expires_at_ms) =
        fetch_copilot_token(&http, &github_token, enterprise_url.as_deref()).await?;
    let base_url = base_url_from_token(Some(&copilot_token), enterprise_url.as_deref());

    callbacks.progress("Enabling available models...").await;
    enable_models(&http, &base_url, &copilot_token, known_model_ids).await;
    let available_model_ids = fetch_available_models(&http, &base_url, &copilot_token)
        .await
        .unwrap_or_default();

    Ok(GithubCopilotCredential {
        access: copilot_token,
        refresh: github_token,
        expires_at_ms,
        enterprise_url,
        available_model_ids,
    })
}

/// Resolve `login()`'s prompted domain input into `(stored_enterprise_url, device_flow_domain)` — pi's
/// own `loginGitHubCopilot` does the identical thing inline (`normalizeDomain` + `enterpriseDomain ||
/// "github.com"`). A blank/whitespace-only answer means plain `github.com`, nothing enterprise-specific
/// to record (`stored_enterprise_url: None`); anything else must normalize to a real hostname (see
/// [`normalize_enterprise_domain`]) or the whole login fails outright — matching pi, which throws
/// rather than silently falling back to `github.com` on a typo'd domain.
fn resolve_login_domain(input: &str) -> Result<(Option<String>, String)> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok((None, "github.com".to_string()));
    }
    match normalize_enterprise_domain(trimmed) {
        Some(host) => Ok((Some(host.clone()), host)),
        None => Err(OAuthError::InvalidInput(
            "Invalid GitHub Enterprise URL/domain".to_string(),
        )),
    }
}

/// Normalize a raw GitHub Enterprise URL/domain into a bare hostname (`"company.ghe.com"`) — accepts
/// either a bare host or a full URL with an explicit scheme, mirrors pi's own `normalizeDomain`
/// (`packages/ai/src/utils/oauth/github-copilot.ts`). `None` for input that doesn't parse as a URL/host
/// at all; the caller ([`resolve_login_domain`]) turns that into a hard login error rather than a
/// silent fallback to `github.com`.
fn normalize_enterprise_domain(trimmed: &str) -> Option<String> {
    let candidate = if trimmed.contains("://") {
        trimmed.to_string()
    } else {
        format!("https://{trimmed}")
    };
    let parsed = url::Url::parse(&candidate).ok()?;
    parsed.host_str().map(str::to_string)
}

/// Refresh the Copilot-internal token from the long-lived GitHub device-flow token. Unlike `login`,
/// this never re-runs the best-effort model-enable fan-out — only `login` does that — but it does
/// re-fetch model availability, since newly-enabled/disabled models should still be reflected.
pub async fn refresh(
    refresh_token: &str,
    enterprise_domain: Option<&str>,
) -> Result<GithubCopilotCredential> {
    let http = Client::new();
    let (access, expires_at_ms) =
        fetch_copilot_token(&http, refresh_token, enterprise_domain).await?;
    let base_url = base_url_from_token(Some(&access), enterprise_domain);
    let available_model_ids = fetch_available_models(&http, &base_url, &access)
        .await
        .unwrap_or_default();
    Ok(GithubCopilotCredential {
        access,
        refresh: refresh_token.to_string(),
        expires_at_ms,
        enterprise_url: enterprise_domain.map(str::to_string),
        available_model_ids,
    })
}

#[derive(Deserialize)]
struct DeviceCodeResponse {
    device_code: String,
    user_code: String,
    verification_uri: String,
    expires_in: u64,
    #[serde(default)]
    interval: Option<u64>,
}

async fn request_device_code(http: &Client, domain: &str) -> Result<DeviceCodeResponse> {
    let url = format!("https://{domain}/login/device/code");
    let resp = http
        .post(&url)
        .header("Accept", "application/json")
        .header("User-Agent", COPILOT_USER_AGENT)
        .form(&[("client_id", CLIENT_ID), ("scope", DEVICE_SCOPE)])
        .send()
        .await
        .map_err(|source| OAuthError::Network {
            url: url.clone(),
            source,
        })?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(OAuthError::Http {
            provider: "github",
            status: status.as_u16(),
            body,
        });
    }
    let device: DeviceCodeResponse = resp.json().await.map_err(|source| OAuthError::Network {
        url: url.clone(),
        source,
    })?;
    validate_verification_uri(&device.verification_uri)?;
    Ok(device)
}

/// Rejects anything but `http`/`https` — a real security check against a malicious/compromised
/// device-flow server smuggling a shell-executable string into whatever opens this URI for the user.
fn validate_verification_uri(uri: &str) -> Result<()> {
    let parsed = url::Url::parse(uri).map_err(|_| OAuthError::InvalidResponse {
        provider: "github",
        detail: format!("verification_uri is not a valid URL: {uri}"),
    })?;
    if parsed.scheme() != "http" && parsed.scheme() != "https" {
        return Err(OAuthError::InvalidResponse {
            provider: "github",
            detail: format!("untrusted verification_uri scheme: {}", parsed.scheme()),
        });
    }
    Ok(())
}

async fn poll_access_token_once(
    http: &Client,
    url: &str,
    device_code: &str,
) -> Result<DevicePollStep<String>> {
    let resp = http
        .post(url)
        .header("Accept", "application/json")
        .header("User-Agent", COPILOT_USER_AGENT)
        .form(&[
            ("client_id", CLIENT_ID),
            ("device_code", device_code),
            ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
        ])
        .send()
        .await
        .map_err(|source| OAuthError::Network {
            url: url.to_string(),
            source,
        })?;
    let body: serde_json::Value = resp.json().await.map_err(|source| OAuthError::Network {
        url: url.to_string(),
        source,
    })?;
    parse_poll_response(&body)
}

/// Parse GitHub's device-token poll response body into a [`DevicePollStep`] — split out from
/// [`poll_access_token_once`] purely so the `slow_down`/`interval` handling is unit-testable without
/// a live HTTP server.
///
/// GitHub's `slow_down` response reports its own required minimum poll interval in an `interval`
/// field (seconds) — the same field the initial device-code response carries. A client that only
/// tracks its own locally-incremented interval can fall behind that requirement under clock drift
/// (WSL/VM guests), keep polling too early, and re-trigger `slow_down` on every subsequent attempt
/// until the whole login times out. When present (and sane — a finite positive number of seconds),
/// it's threaded through as [`DevicePollStep::SlowDownWithInterval`] so the shared poll loop
/// (`device_code::poll_device_code`) can prefer it outright over the local +5s increment.
fn parse_poll_response(body: &serde_json::Value) -> Result<DevicePollStep<String>> {
    if let Some(token) = body.get("access_token").and_then(|v| v.as_str()) {
        return Ok(DevicePollStep::Complete(token.to_string()));
    }
    let error = body
        .get("error")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    match error {
        "authorization_pending" => Ok(DevicePollStep::Pending),
        "slow_down" => {
            let server_interval = body
                .get("interval")
                .and_then(|v| v.as_u64())
                .filter(|&seconds| seconds > 0)
                .map(Duration::from_secs);
            Ok(match server_interval {
                Some(server_interval) => DevicePollStep::SlowDownWithInterval(server_interval),
                None => DevicePollStep::SlowDown,
            })
        }
        other => {
            let description = body
                .get("error_description")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            Err(OAuthError::DeviceFlowFailed(format!(
                "{other}: {description}"
            )))
        }
    }
}

#[derive(Deserialize)]
struct CopilotTokenResponse {
    token: String,
    /// Unix seconds, not milliseconds.
    expires_at: i64,
}

async fn fetch_copilot_token(
    http: &Client,
    refresh_token: &str,
    enterprise_domain: Option<&str>,
) -> Result<(String, i64)> {
    let domain = enterprise_domain.unwrap_or("github.com");
    let url = format!("https://api.{domain}/copilot_internal/v2/token");
    let req = apply_copilot_headers(
        http.get(&url)
            .header("Accept", "application/json")
            .header("Authorization", format!("Bearer {refresh_token}")),
    );
    let resp = req.send().await.map_err(|source| OAuthError::Network {
        url: url.clone(),
        source,
    })?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(OAuthError::Http {
            provider: "github-copilot",
            status: status.as_u16(),
            body,
        });
    }
    let parsed: CopilotTokenResponse = resp.json().await.map_err(|source| OAuthError::Network {
        url: url.clone(),
        source,
    })?;
    let expires_at_ms = parsed
        .expires_at
        .saturating_mul(1000)
        .checked_sub(EARLY_REFRESH_BUFFER_MS)
        .unwrap_or(0);
    Ok((parsed.token, expires_at_ms))
}

fn apply_copilot_headers(req: RequestBuilder) -> RequestBuilder {
    req.header("User-Agent", COPILOT_USER_AGENT)
        .header("Editor-Version", COPILOT_EDITOR_VERSION)
        .header("Editor-Plugin-Version", COPILOT_EDITOR_PLUGIN_VERSION)
        .header("Copilot-Integration-Id", COPILOT_INTEGRATION_ID)
}

/// Parses the access token itself for `proxy-ep=([^;]+)` and rewrites a leading `proxy.` to `api.`;
/// falls back to `https://copilot-api.{enterprise_domain}` if Enterprise and no proxy-ep found, else
/// `https://api.individual.githubcopilot.com`. Pure string logic, no network — the *only*
/// token-dependent request-time detail besides the bearer value itself.
pub fn base_url_from_token(access_token: Option<&str>, enterprise_domain: Option<&str>) -> String {
    if let Some(token) = access_token
        && let Some(proxy_ep) = extract_proxy_ep(token)
    {
        let api_host = match proxy_ep.strip_prefix("proxy.") {
            Some(rest) => format!("api.{rest}"),
            None => proxy_ep,
        };
        return format!("https://{api_host}");
    }
    if let Some(domain) = enterprise_domain {
        return format!("https://copilot-api.{domain}");
    }
    "https://api.individual.githubcopilot.com".to_string()
}

fn extract_proxy_ep(token: &str) -> Option<String> {
    token
        .split(';')
        .find_map(|part| part.trim().strip_prefix("proxy-ep=").map(str::to_string))
}

/// GitHub Copilot's real endpoint path per dialect — **not** the dialect's own default
/// `endpoint_path()`: Copilot's OpenAI-wire endpoints omit the `/v1` prefix pi's official SDKs bake
/// into their default `baseURL` (so the SDK's own `/chat/completions`/`/responses` relative paths land
/// bare on Copilot's host), while its Anthropic-wire endpoint matches the dialect default verbatim
/// (the Anthropic SDK's default `baseURL` carries no version segment, so it *always* appends
/// `/v1/messages` itself). See `packages/ai/src/api/{anthropic-messages,openai-completions,
/// openai-responses}.ts` in pi-mono (each sets `baseURL: model.baseUrl` on the vendor SDK, which then
/// appends its own fixed relative path).
pub fn copilot_endpoint_path(dialect: agent_core::dialect::Dialect) -> &'static str {
    match dialect {
        agent_core::dialect::Dialect::Anthropic => "/v1/messages",
        agent_core::dialect::Dialect::OpenAi => "/chat/completions",
        agent_core::dialect::Dialect::OpenAiResponses => "/responses",
    }
}

/// GitHub Copilot's [`CredentialSource`] wrapper. Unlike OpenAI Codex's fully-static routing
/// (`crate::gateway_credential`'s private `DirectRoutedCredentialSource`, whose `chatgpt-account-id`
/// claim never changes across a refresh of the same login), Copilot's proxy host is embedded in the
/// access token itself (`proxy-ep=…`) and CAN change across a refresh — GitHub migrating an account to
/// a different Copilot proxy pool mid-session is a real, previously observed occurrence. `base_url`
/// is therefore re-derived via [`base_url_from_token`] fresh on every [`credential`](Self::credential)
/// call, from whichever access token is CURRENTLY stored on disk — re-read directly from `store_path`
/// rather than through `inner`'s own opaque [`Credential`] (which never exposes a constructed
/// credential's plaintext token to any caller outside `agent_core::client` by design — see that
/// type's own doc comment) — instead of a value computed once at construction and frozen for this
/// source's lifetime.
pub struct CopilotRoutedCredentialSource {
    pub inner: Arc<dyn CredentialSource>,
    /// Same `auth.json` path `inner`'s own `OAuthCredentialSource` refreshes against — re-read here,
    /// after `inner.credential()` returns, so the routing this call computes reflects whatever that
    /// refresh (if any) just persisted, not whatever was on disk before it ran.
    pub store_path: PathBuf,
    /// GitHub Enterprise domain, if any — an account-level property that doesn't change across a
    /// token refresh, unlike `proxy-ep`, so (unlike `base_url`) this one *is* fine to fix at
    /// construction.
    pub enterprise_url: Option<String>,
    pub path: &'static str,
    /// Memoized `(expires_at_ms, routing)` from the last disk read (T9-F1). The proxy host in
    /// `access` only changes when the token is refreshed, and a refresh always mints a fresh
    /// `expires_at_ms`; so while `now < expires_at_ms` the stored token is provably unchanged and the
    /// derived routing can be reused without re-opening+re-parsing `auth.json` on every model request
    /// (which otherwise defeats `inner`'s own in-memory credential cache). Past that expiry `inner`
    /// may have just refreshed, so the disk read runs again and re-keys on the new expiry.
    #[allow(clippy::type_complexity)]
    pub cached_routing: Mutex<Option<(i64, DirectRouting)>>,
}

impl CopilotRoutedCredentialSource {
    /// The routing this call should use, derived from whichever `github-copilot` access token is
    /// CURRENTLY on disk. Factored out from `credential()` so it's independently testable without
    /// needing to see inside `agent_core::client::Credential`'s deliberately opaque internals.
    fn current_routing(&self) -> DirectRouting {
        // Fast path (T9-F1): while the cached token has not yet expired, no refresh can have run, so
        // the on-disk access token — and thus the derived routing — is unchanged. Skip the disk read.
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        {
            let guard = self
                .cached_routing
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if let Some((expires_at_ms, routing)) = guard.as_ref()
                && now_ms < *expires_at_ms
            {
                return routing.clone();
            }
        }
        // Cache empty or the cached token has expired: re-read whatever `inner`'s `credential()` just
        // (possibly) refreshed onto disk, and re-key on its expiry.
        let (access, expires_at_ms) = crate::auth_store::AuthStore::open(self.store_path.clone())
            .get("github-copilot")
            .and_then(|stored| match &stored.credential {
                crate::oauth::OAuthCredential::GithubCopilot(c) => {
                    Some((c.access.clone(), c.expires_at_ms))
                }
                _ => None,
            })
            .map_or((None, 0), |(access, expires_at_ms)| {
                (Some(access), expires_at_ms)
            });
        let routing = DirectRouting {
            route: RouteOverride::Direct {
                base_url: base_url_from_token(access.as_deref(), self.enterprise_url.as_deref()),
                path: self.path,
            },
            static_headers: vec![
                ("User-Agent", COPILOT_USER_AGENT.to_string()),
                ("Editor-Version", COPILOT_EDITOR_VERSION.to_string()),
                (
                    "Editor-Plugin-Version",
                    COPILOT_EDITOR_PLUGIN_VERSION.to_string(),
                ),
                ("Copilot-Integration-Id", COPILOT_INTEGRATION_ID.to_string()),
            ],
            copilot_dynamic_headers: true,
            auth_header: None,
            auth_header_prefix: None,
            dialect_override: None,
            deployment_name: None,
            query: None,
            // GitHub Copilot is reached via its own dynamically-resolved proxy host, matched by a
            // stored credential's `available_model_ids`, never a BYO `base_url` override — no
            // aggregator host of its own to report (see `DirectRouting::aggregator_host`'s own doc
            // comment).
            aggregator_host: None,
        };
        *self
            .cached_routing
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some((expires_at_ms, routing.clone()));
        routing
    }
}

#[async_trait::async_trait]
impl CredentialSource for CopilotRoutedCredentialSource {
    async fn credential(&self) -> agent_core::Result<Credential> {
        let credential = self.inner.credential().await?;
        Ok(credential.with_direct_routing(self.current_routing()))
    }
}

#[derive(Deserialize)]
struct ModelsResponse {
    data: Vec<ModelEntry>,
}

#[derive(Deserialize)]
struct ModelEntry {
    id: String,
    #[serde(default)]
    model_picker_enabled: bool,
    #[serde(default)]
    policy: Option<ModelPolicy>,
    #[serde(default)]
    capabilities: Option<ModelCapabilities>,
}

#[derive(Deserialize)]
struct ModelPolicy {
    state: String,
}

#[derive(Deserialize)]
struct ModelCapabilities {
    supports: ModelSupports,
}

#[derive(Deserialize)]
struct ModelSupports {
    #[serde(default = "default_true")]
    tool_calls: bool,
}

fn default_true() -> bool {
    true
}

/// Selectable iff `model_picker_enabled && policy?.state != "disabled" && capabilities.supports
/// .tool_calls != false` — a model with no `policy`/`capabilities` block at all defaults open (not
/// disabled, tool calls assumed supported), matching pi's own optional-chaining semantics.
fn is_model_selectable(entry: &ModelEntry) -> bool {
    entry.model_picker_enabled
        && entry.policy.as_ref().map(|p| p.state.as_str()) != Some("disabled")
        && entry
            .capabilities
            .as_ref()
            .map(|c| c.supports.tool_calls)
            .unwrap_or(true)
}

async fn fetch_available_models(http: &Client, base_url: &str, token: &str) -> Result<Vec<String>> {
    let url = format!("{base_url}/models");
    let req = apply_copilot_headers(
        http.get(&url)
            .timeout(Duration::from_secs(5))
            .header("Authorization", format!("Bearer {token}"))
            .header("X-GitHub-Api-Version", COPILOT_API_VERSION),
    );
    let resp = req.send().await.map_err(|source| OAuthError::Network {
        url: url.clone(),
        source,
    })?;
    if !resp.status().is_success() {
        // Best-effort: a failure here shouldn't fail the whole login/refresh — an empty list just
        // means "unknown," not "nothing available."
        return Ok(Vec::new());
    }
    let parsed: ModelsResponse = resp.json().await.map_err(|source| OAuthError::Network {
        url: url.clone(),
        source,
    })?;
    Ok(parsed
        .data
        .into_iter()
        .filter(is_model_selectable)
        .map(|m| m.id)
        .collect())
}

/// Best-effort, parallel, login-only: enable every model in `model_ids` for this account. Failures
/// are swallowed — matching pi, which only reports them via a progress callback, never fails login
/// over a policy-enable rejection.
async fn enable_models(http: &Client, base_url: &str, token: &str, model_ids: &[&str]) {
    let calls = model_ids.iter().map(|id| {
        let url = format!("{base_url}/models/{id}/policy");
        let http = http.clone();
        let token = token.to_string();
        async move {
            let req = apply_copilot_headers(
                http.post(&url)
                    .header("Authorization", format!("Bearer {token}"))
                    .header("openai-intent", "chat-policy")
                    .header("x-interaction-type", "chat-policy")
                    .json(&serde_json::json!({ "state": "enabled" })),
            );
            let _ = req.send().await;
        }
    });
    join_all(calls).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_model_ids_is_non_empty_and_has_no_duplicates() {
        assert!(!KNOWN_MODEL_IDS.is_empty());
        let mut sorted = KNOWN_MODEL_IDS.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            KNOWN_MODEL_IDS.len(),
            "KNOWN_MODEL_IDS must not contain duplicate ids"
        );
    }

    #[test]
    fn parse_poll_response_returns_complete_when_an_access_token_is_present() {
        let body = serde_json::json!({ "access_token": "ghu_abc123" });
        let step = parse_poll_response(&body).unwrap();
        assert!(matches!(step, DevicePollStep::Complete(token) if token == "ghu_abc123"));
    }

    #[test]
    fn parse_poll_response_returns_pending_on_authorization_pending() {
        let body = serde_json::json!({ "error": "authorization_pending" });
        let step = parse_poll_response(&body).unwrap();
        assert!(matches!(step, DevicePollStep::Pending));
    }

    #[test]
    fn parse_poll_response_returns_plain_slow_down_when_no_interval_is_given() {
        let body = serde_json::json!({ "error": "slow_down" });
        let step = parse_poll_response(&body).unwrap();
        assert!(matches!(step, DevicePollStep::SlowDown));
    }

    #[test]
    fn parse_poll_response_carries_the_servers_interval_when_slow_down_names_one() {
        // A server interval (7s) larger than the local +5s increment would ever produce on its own
        // — the whole point of threading this through is that this larger, server-named value must
        // be what the poll loop actually uses.
        let body = serde_json::json!({ "error": "slow_down", "interval": 7 });
        let step = parse_poll_response(&body).unwrap();
        match step {
            DevicePollStep::SlowDownWithInterval(interval) => {
                assert_eq!(interval, Duration::from_secs(7));
            }
            _ => panic!("expected SlowDownWithInterval"),
        }
    }

    #[test]
    fn parse_poll_response_ignores_a_zero_interval_and_falls_back_to_plain_slow_down() {
        let body = serde_json::json!({ "error": "slow_down", "interval": 0 });
        let step = parse_poll_response(&body).unwrap();
        assert!(matches!(step, DevicePollStep::SlowDown));
    }

    #[test]
    fn parse_poll_response_fails_on_an_unrecognized_error_with_its_description() {
        let body =
            serde_json::json!({ "error": "access_denied", "error_description": "user said no" });
        let err = parse_poll_response(&body).unwrap_err();
        match err {
            OAuthError::DeviceFlowFailed(msg) => {
                assert!(msg.contains("access_denied"));
                assert!(msg.contains("user said no"));
            }
            other => panic!("expected DeviceFlowFailed, got {other:?}"),
        }
    }

    #[test]
    fn base_url_extracts_proxy_ep_and_rewrites_proxy_prefix_to_api() {
        let token = "tid=abc;proxy-ep=proxy.business.githubcopilot.com;exp=123";
        assert_eq!(
            base_url_from_token(Some(token), None),
            "https://api.business.githubcopilot.com"
        );
    }

    #[test]
    fn base_url_keeps_a_proxy_ep_without_a_proxy_prefix_as_is() {
        let token = "tid=abc;proxy-ep=api.already-correct.com";
        assert_eq!(
            base_url_from_token(Some(token), None),
            "https://api.already-correct.com"
        );
    }

    #[test]
    fn base_url_falls_back_to_enterprise_when_no_proxy_ep_found() {
        assert_eq!(
            base_url_from_token(Some("tid=abc;exp=123"), Some("ghe.example.com")),
            "https://copilot-api.ghe.example.com"
        );
    }

    #[test]
    fn base_url_falls_back_to_individual_when_nothing_else_applies() {
        assert_eq!(
            base_url_from_token(None, None),
            "https://api.individual.githubcopilot.com"
        );
        assert_eq!(
            base_url_from_token(Some("tid=abc;exp=123"), None),
            "https://api.individual.githubcopilot.com"
        );
    }

    #[test]
    fn copilot_endpoint_path_strips_v1_for_openai_wire_dialects_but_not_anthropic() {
        // pi-parity fix: GitHub Copilot's OpenAI-wire endpoints omit the `/v1` prefix the dialect's
        // own default `endpoint_path()` carries (pi's own SDKs set `baseURL: model.baseUrl` with no
        // `/v1` in it, then the vendor SDK appends its fixed relative path) — only the Anthropic-wire
        // endpoint matches the dialect default verbatim (the Anthropic SDK's default `baseURL` has no
        // version segment at all, so it always appends `/v1/messages` itself).
        assert_eq!(
            copilot_endpoint_path(agent_core::dialect::Dialect::Anthropic),
            "/v1/messages"
        );
        assert_eq!(
            copilot_endpoint_path(agent_core::dialect::Dialect::OpenAi),
            "/chat/completions"
        );
        assert_eq!(
            copilot_endpoint_path(agent_core::dialect::Dialect::OpenAiResponses),
            "/responses"
        );
    }

    #[test]
    fn resolve_login_domain_a_blank_answer_defaults_to_github_com_with_no_stored_enterprise_url() {
        let (enterprise_url, domain) = resolve_login_domain("").unwrap();
        assert_eq!(enterprise_url, None);
        assert_eq!(domain, "github.com");
    }

    #[test]
    fn resolve_login_domain_whitespace_only_is_treated_as_blank() {
        let (enterprise_url, domain) = resolve_login_domain("   \n  ").unwrap();
        assert_eq!(enterprise_url, None);
        assert_eq!(domain, "github.com");
    }

    #[test]
    fn resolve_login_domain_a_bare_hostname_is_used_verbatim_and_stored_as_the_enterprise_url() {
        let (enterprise_url, domain) = resolve_login_domain("company.ghe.com").unwrap();
        assert_eq!(enterprise_url.as_deref(), Some("company.ghe.com"));
        assert_eq!(domain, "company.ghe.com");
        // The device-flow URLs this feeds `request_device_code`/`fetch_copilot_token` are built
        // directly from `domain` — confirming the resolved value is exactly the hostname the device
        // flow (and the stored credential's `enterprise_url`) must both use.
        assert_eq!(
            format!("https://{domain}/login/device/code"),
            "https://company.ghe.com/login/device/code"
        );
    }

    #[test]
    fn resolve_login_domain_a_full_url_normalizes_down_to_just_the_hostname() {
        let (enterprise_url, domain) =
            resolve_login_domain("https://company.ghe.com/some/path").unwrap();
        assert_eq!(enterprise_url.as_deref(), Some("company.ghe.com"));
        assert_eq!(domain, "company.ghe.com");
    }

    #[test]
    fn resolve_login_domain_trims_surrounding_whitespace_before_normalizing() {
        let (enterprise_url, domain) = resolve_login_domain("  company.ghe.com  ").unwrap();
        assert_eq!(enterprise_url.as_deref(), Some("company.ghe.com"));
        assert_eq!(domain, "company.ghe.com");
    }

    #[test]
    fn resolve_login_domain_rejects_unparsable_non_blank_input() {
        let err = resolve_login_domain("not a valid domain!! ??").unwrap_err();
        assert!(matches!(err, OAuthError::InvalidInput(_)));
    }

    #[test]
    fn normalize_enterprise_domain_accepts_a_bare_host_with_no_scheme() {
        assert_eq!(
            normalize_enterprise_domain("company.ghe.com").as_deref(),
            Some("company.ghe.com")
        );
    }

    #[test]
    fn normalize_enterprise_domain_accepts_a_url_with_a_scheme_and_path() {
        assert_eq!(
            normalize_enterprise_domain("https://company.ghe.com/x/y").as_deref(),
            Some("company.ghe.com")
        );
    }

    #[test]
    fn normalize_enterprise_domain_rejects_garbage() {
        assert_eq!(normalize_enterprise_domain("not a valid domain!! ??"), None);
    }

    #[test]
    fn validate_verification_uri_accepts_https() {
        validate_verification_uri("https://github.com/login/device").unwrap();
    }

    #[test]
    fn validate_verification_uri_rejects_a_non_http_scheme() {
        // A malicious/compromised device-flow server smuggling a shell-executable string.
        let err = validate_verification_uri("javascript:alert(1)").unwrap_err();
        assert!(matches!(err, OAuthError::InvalidResponse { .. }));
    }

    #[test]
    fn validate_verification_uri_rejects_garbage() {
        let err = validate_verification_uri("$(id>/tmp/pwned)").unwrap_err();
        assert!(matches!(err, OAuthError::InvalidResponse { .. }));
    }

    fn entry(
        model_picker_enabled: bool,
        state: Option<&str>,
        tool_calls: Option<bool>,
    ) -> ModelEntry {
        ModelEntry {
            id: "m".to_string(),
            model_picker_enabled,
            policy: state.map(|s| ModelPolicy {
                state: s.to_string(),
            }),
            capabilities: tool_calls.map(|t| ModelCapabilities {
                supports: ModelSupports { tool_calls: t },
            }),
        }
    }

    #[test]
    fn a_fully_enabled_model_with_no_policy_or_capabilities_block_is_selectable() {
        assert!(is_model_selectable(&entry(true, None, None)));
    }

    #[test]
    fn model_picker_disabled_is_never_selectable() {
        assert!(!is_model_selectable(&entry(false, None, None)));
    }

    #[test]
    fn an_explicitly_disabled_policy_is_not_selectable() {
        assert!(!is_model_selectable(&entry(true, Some("disabled"), None)));
    }

    #[test]
    fn a_non_disabled_policy_state_is_still_selectable() {
        assert!(is_model_selectable(&entry(true, Some("enabled"), None)));
    }

    #[test]
    fn tool_calls_false_is_not_selectable() {
        assert!(!is_model_selectable(&entry(true, None, Some(false))));
    }

    #[test]
    fn debug_redacts_the_access_and_refresh_tokens_but_keeps_non_secret_fields() {
        let cred = GithubCopilotCredential {
            access: "ghu_live_access_secret".to_string(),
            refresh: "ghu_live_refresh_secret".to_string(),
            expires_at_ms: 123456,
            enterprise_url: Some("company.ghe.com".to_string()),
            available_model_ids: vec!["gpt-5-codex".to_string()],
        };
        let debug = format!("{cred:?}");
        assert!(!debug.contains("ghu_live_access_secret"));
        assert!(!debug.contains("ghu_live_refresh_secret"));
        assert!(debug.contains("<redacted>"));
        assert!(debug.contains("123456"));
        assert!(debug.contains("company.ghe.com"));
        assert!(debug.contains("gpt-5-codex"));
    }

    struct DummyInner;

    #[async_trait::async_trait]
    impl CredentialSource for DummyInner {
        async fn credential(&self) -> agent_core::Result<Credential> {
            Ok(Credential::new("dummy-bearer", true))
        }
    }

    fn stored_copilot_credential(access: &str) -> crate::oauth::OAuthCredential {
        // A far-past expiry by default, so the T9-F1 routing memoization never short-circuits and each
        // `current_routing()` genuinely re-reads the store — the behavior every re-derivation test here
        // relies on. The dedicated cache-hit test below supplies its own future expiry instead.
        stored_copilot_credential_expiring(access, 0)
    }

    fn stored_copilot_credential_expiring(
        access: &str,
        expires_at_ms: i64,
    ) -> crate::oauth::OAuthCredential {
        crate::oauth::OAuthCredential::GithubCopilot(GithubCopilotCredential {
            access: access.to_string(),
            refresh: "refresh-token".to_string(),
            expires_at_ms,
            enterprise_url: None,
            available_model_ids: vec!["gpt-5-codex".to_string()],
        })
    }

    fn direct_base_url(routing: &DirectRouting) -> &str {
        match &routing.route {
            RouteOverride::Direct { base_url, .. } => base_url,
            other => panic!("expected a Direct route, got {other:?}"),
        }
    }

    #[test]
    fn copilot_routed_credential_source_rederives_base_url_from_the_current_token() {
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("auth.json");
        crate::auth_store::AuthStore::open(store_path.clone())
            .set(
                "github-copilot",
                stored_copilot_credential(
                    "tid=abc;proxy-ep=proxy.pool-a.githubcopilot.com;exp=123",
                ),
            )
            .unwrap();

        let source = CopilotRoutedCredentialSource {
            inner: Arc::new(DummyInner),
            store_path: store_path.clone(),
            enterprise_url: None,
            path: "/chat/completions",
            cached_routing: Mutex::new(None),
        };

        assert_eq!(
            direct_base_url(&source.current_routing()),
            "https://api.pool-a.githubcopilot.com"
        );

        // Simulate a live refresh migrating the account to a different Copilot proxy pool — the same
        // thing `AuthStore::refreshed` does by writing a fresh access token back to `store_path`, a
        // real, previously observed occurrence this crate must not silently misroute after.
        crate::auth_store::AuthStore::open(store_path.clone())
            .set(
                "github-copilot",
                stored_copilot_credential(
                    "tid=abc;proxy-ep=proxy.pool-b.githubcopilot.com;exp=456",
                ),
            )
            .unwrap();

        assert_eq!(
            direct_base_url(&source.current_routing()),
            "https://api.pool-b.githubcopilot.com",
            "must reflect the new proxy-ep claim on the second call, not the one frozen at construction"
        );
    }

    #[test]
    fn copilot_routing_is_memoized_while_the_token_has_not_expired() {
        // T9-F1: while the cached token is still valid (future expiry), a subsequent on-disk token
        // change must NOT be re-read — the whole point is to skip the per-request `auth.json` parse.
        // (A real proxy-pool migration only lands via a refresh, which mints a fresh `expires_at_ms`
        // and so re-keys the cache; that path is covered by the re-derivation test above.)
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("auth.json");
        let far_future = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64
            + 3_600_000;
        crate::auth_store::AuthStore::open(store_path.clone())
            .set(
                "github-copilot",
                stored_copilot_credential_expiring(
                    "tid=abc;proxy-ep=proxy.pool-a.githubcopilot.com;exp=123",
                    far_future,
                ),
            )
            .unwrap();

        let source = CopilotRoutedCredentialSource {
            inner: Arc::new(DummyInner),
            store_path: store_path.clone(),
            enterprise_url: None,
            path: "/chat/completions",
            cached_routing: Mutex::new(None),
        };

        assert_eq!(
            direct_base_url(&source.current_routing()),
            "https://api.pool-a.githubcopilot.com"
        );

        // Change the on-disk token WITHOUT changing the expiry — not a real refresh, so the cache is
        // entitled to keep serving the still-valid routing without touching disk.
        crate::auth_store::AuthStore::open(store_path.clone())
            .set(
                "github-copilot",
                stored_copilot_credential_expiring(
                    "tid=abc;proxy-ep=proxy.pool-b.githubcopilot.com;exp=456",
                    far_future,
                ),
            )
            .unwrap();

        assert_eq!(
            direct_base_url(&source.current_routing()),
            "https://api.pool-a.githubcopilot.com",
            "an unexpired cache must serve the memoized routing, not re-read auth.json"
        );
    }

    #[tokio::test]
    async fn copilot_routed_credential_source_credential_call_succeeds_end_to_end() {
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("auth.json");
        crate::auth_store::AuthStore::open(store_path.clone())
            .set(
                "github-copilot",
                stored_copilot_credential(
                    "tid=abc;proxy-ep=proxy.pool-a.githubcopilot.com;exp=123",
                ),
            )
            .unwrap();

        let source = CopilotRoutedCredentialSource {
            inner: Arc::new(DummyInner),
            store_path,
            enterprise_url: None,
            path: "/chat/completions",
            cached_routing: Mutex::new(None),
        };

        // `Credential`'s fields are private (see `agent_core::client::Credential`'s own doc comment);
        // the only externally observable proof available here is that the wrapper's own plumbing
        // (calling `inner`, re-reading the store, building the routing) completes without error.
        source.credential().await.unwrap();
    }
}
