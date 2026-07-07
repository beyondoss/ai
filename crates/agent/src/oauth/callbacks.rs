//! [`LoginCallbacks`] — the interface a login-driving caller (CLI, `serve` RPC) implements so the
//! provider-specific flows in this module can show a URL/device code, ask for pasted input, and
//! report progress without knowing whether they're talking to a terminal, a headless RPC client, or
//! a test double. Deliberately thinner than pi's own `OAuthLoginCallbacks`: no separate
//! manual-code-input callback (folded into `prompt_text` — see `callback_server.rs` for why racing it
//! against the local listener needs no third fallback tier the way pi's does), and cancellation is a
//! `CancellationToken` parameter to `login`/`refresh` rather than a callback-embedded signal, matching
//! how `agent_core` already models run/turn/tool abort.

use async_trait::async_trait;

use super::error::OAuthError;

/// A free-text prompt — a GitHub Enterprise domain, or "paste the code/redirect URL" (also doubles
/// as the manual-code channel raced against the local callback server).
pub struct TextPrompt<'a> {
    pub message: &'a str,
    pub placeholder: Option<&'a str>,
}

/// A fixed choice — currently only OpenAI Codex's "browser vs. device-code" login-method pick.
pub struct SelectPrompt<'a> {
    pub message: &'a str,
    pub options: &'a [SelectOption],
}

pub struct SelectOption {
    pub id: String,
    pub label: String,
}

/// A device code + verification URL to display (GitHub Copilot; OpenAI Codex's device-code path).
pub struct DeviceCodeInfo {
    pub user_code: String,
    pub verification_uri: String,
    pub interval: Option<std::time::Duration>,
    pub expires_in: Option<std::time::Duration>,
}

#[async_trait]
pub trait LoginCallbacks: Send + Sync {
    /// A URL the user should open in a browser (Anthropic; Codex's browser path).
    async fn show_auth_url(&self, url: &str, instructions: Option<&str>);
    /// A device code + verification URL (Copilot; Codex's device-code path).
    async fn show_device_code(&self, info: &DeviceCodeInfo);
    /// Non-blocking narration ("Exchanging code for tokens...", "Enabling models...").
    async fn progress(&self, message: &str);
    /// Ask a free-text question. A caller with no paste UI (e.g. a headless RPC client that never
    /// offers manual-code entry) should return a future that never resolves
    /// (`std::future::pending().await`) rather than an early error — `tokio::select!` then simply
    /// never picks this arm, and the callback-server-only path still works on its own.
    async fn prompt_text(&self, prompt: &TextPrompt<'_>) -> Result<String, OAuthError>;
    /// Offer a fixed choice (Codex: browser vs. device-code). `Ok(None)` means the user explicitly
    /// backed out (→ `OAuthError::LoginCancelled`). A non-interactive caller should return a sensible
    /// default id instead of `None`.
    async fn select(&self, prompt: &SelectPrompt<'_>) -> Result<Option<String>, OAuthError>;
}
