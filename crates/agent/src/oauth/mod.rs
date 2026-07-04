//! OAuth / subscription-credential authentication — a Rust port of pi's support for logging in with
//! a Claude Pro/Max, GitHub Copilot, or ChatGPT Plus/Pro subscription instead of a metered API key.
//!
//! This module owns only the three providers' *protocol* flows (PKCE/device-code login, token
//! refresh) — not where the resulting credential is persisted (`crate::auth_store`), how login is
//! triggered from the CLI/RPC surface (`main.rs`/`serve.rs`), or how a live token gets attached to an
//! actual model request (`agent_core::client::CredentialSource`, `crate::auth_credential_source`).
//! Lives in `agent`, not `agent_core`: this crate's own doc comment already draws the line —
//! `agent_core`'s only network surface is `GatewayClient`, which never talks to a provider directly,
//! while OAuth login talks *directly* to `claude.ai`/`github.com`/`auth.openai.com`, a second,
//! fundamentally different network surface `agent_core` explicitly disclaims owning. `agent_core` is
//! also documented as runtime-agnostic (no hard `tokio` dependency); device-code polling and bridging
//! a blocking local callback listener into async both need a live tokio runtime as a real, non-dev
//! dependency, which would break that guarantee.
//!
//! **Deliberate divergence from pi**: pi's credential store can also hold a plain `api_key` per
//! provider (an alternative to OAuth). This crate already has a working static-credential mechanism
//! — the `--key`/`AI_AGENT_KEY` flag — so a second, parallel place to store a static key would be two
//! mechanisms for one job. `OAuthCredential` below is OAuth-only by design, not an oversight.
//!
//! **Deliberate, confirmed choice on identity headers**: Anthropic's and GitHub Copilot's
//! subscription-OAuth endpoints are gated to their own first-party clients via identity headers
//! (`user-agent: claude-cli/…`/`x-app: cli` for Anthropic, `Editor-Version: vscode/…`/
//! `Copilot-Integration-Id: vscode-chat` for Copilot). This port sends the real headers — full pi
//! parity, matching pi/opencode/other OSS coding agents — a user-confirmed tradeoff, not an
//! oversight; see `crates/agent/ARCHITECTURE.md`'s OAuth section for the full discussion.

pub mod anthropic;
pub mod callback_server;
pub mod callbacks;
pub mod device_code;
pub mod error;
pub mod github_copilot;
pub mod openai_codex;
pub mod pkce;

pub use callbacks::{DeviceCodeInfo, LoginCallbacks, SelectOption, SelectPrompt, TextPrompt};
pub use error::{OAuthError, Result};

use tokio_util::sync::CancellationToken;

/// The three subscription providers this module supports logging into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OAuthProviderId {
    Anthropic,
    GithubCopilot,
    OpenaiCodex,
}

impl OAuthProviderId {
    /// The key this provider's credential is stored/looked up under in `auth_store::AuthStore` —
    /// matches pi's own provider-id strings so `auth.json` stays recognizable across ports.
    pub fn store_key(self) -> &'static str {
        match self {
            OAuthProviderId::Anthropic => "anthropic",
            OAuthProviderId::GithubCopilot => "github-copilot",
            OAuthProviderId::OpenaiCodex => "openai-codex",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "anthropic" => Some(OAuthProviderId::Anthropic),
            "github-copilot" => Some(OAuthProviderId::GithubCopilot),
            "openai-codex" => Some(OAuthProviderId::OpenaiCodex),
            _ => None,
        }
    }

    pub fn all() -> [Self; 3] {
        [
            OAuthProviderId::Anthropic,
            OAuthProviderId::GithubCopilot,
            OAuthProviderId::OpenaiCodex,
        ]
    }
}

impl std::fmt::Display for OAuthProviderId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.store_key())
    }
}

/// A tagged enum, not one struct with an open-ended `extra` bag (pi's TS `[key: string]: unknown`) —
/// GitHub Copilot's `enterprise_url`/`available_model_ids` and OpenAI Codex's `account_id` are
/// meaningless on the other two providers; the type system should make that unrepresentable.
#[derive(Debug, Clone)]
pub enum OAuthCredential {
    Anthropic(anthropic::AnthropicCredential),
    GithubCopilot(github_copilot::GithubCopilotCredential),
    OpenaiCodex(openai_codex::OpenaiCodexCredential),
}

impl OAuthCredential {
    pub fn provider(&self) -> OAuthProviderId {
        match self {
            OAuthCredential::Anthropic(_) => OAuthProviderId::Anthropic,
            OAuthCredential::GithubCopilot(_) => OAuthProviderId::GithubCopilot,
            OAuthCredential::OpenaiCodex(_) => OAuthProviderId::OpenaiCodex,
        }
    }

    pub fn access(&self) -> &str {
        match self {
            OAuthCredential::Anthropic(c) => &c.access,
            OAuthCredential::GithubCopilot(c) => &c.access,
            OAuthCredential::OpenaiCodex(c) => &c.access,
        }
    }

    pub fn refresh_token(&self) -> &str {
        match self {
            OAuthCredential::Anthropic(c) => &c.refresh,
            OAuthCredential::GithubCopilot(c) => &c.refresh,
            OAuthCredential::OpenaiCodex(c) => &c.refresh,
        }
    }

    pub fn expires_at_ms(&self) -> i64 {
        match self {
            OAuthCredential::Anthropic(c) => c.expires_at_ms,
            OAuthCredential::GithubCopilot(c) => c.expires_at_ms,
            OAuthCredential::OpenaiCodex(c) => c.expires_at_ms,
        }
    }

    pub fn is_expired(&self, now_ms: i64) -> bool {
        now_ms >= self.expires_at_ms()
    }
}

/// Log into `provider`, driving `callbacks` for whatever UI the flow needs (a URL to open, a device
/// code, a manual-paste prompt). Blocks until the flow completes, is cancelled, or times out.
pub async fn login(
    provider: OAuthProviderId,
    callbacks: &dyn LoginCallbacks,
    cancel: &CancellationToken,
) -> Result<OAuthCredential> {
    match provider {
        OAuthProviderId::Anthropic => {
            anthropic::login(callbacks, cancel).await.map(OAuthCredential::Anthropic)
        }
        OAuthProviderId::GithubCopilot => {
            github_copilot::login(callbacks, cancel, github_copilot::KNOWN_MODEL_IDS)
                .await
                .map(OAuthCredential::GithubCopilot)
        }
        OAuthProviderId::OpenaiCodex => openai_codex::login(callbacks, cancel, "beyond-ai-agent")
            .await
            .map(OAuthCredential::OpenaiCodex),
    }
}

/// Refresh `cred`, returning the new credential to persist. Never logs a failure itself — the caller
/// (`auth_store::AuthStore::refreshed`) decides how to surface it, per this crate's own "must not
/// print/log directly" rule ported from OpenAI Codex's TS tests.
pub async fn refresh(cred: &OAuthCredential) -> Result<OAuthCredential> {
    match cred {
        OAuthCredential::Anthropic(c) => {
            anthropic::refresh(&c.refresh).await.map(OAuthCredential::Anthropic)
        }
        OAuthCredential::GithubCopilot(c) => {
            github_copilot::refresh(&c.refresh, c.enterprise_url.as_deref())
                .await
                .map(OAuthCredential::GithubCopilot)
        }
        OAuthCredential::OpenaiCodex(c) => {
            openai_codex::refresh(&c.refresh).await.map(OAuthCredential::OpenaiCodex)
        }
    }
}
