//! [`OAuthError`] — failures logging into or refreshing an OAuth/subscription-credential provider.
//!
//! Kept separate from `agent_core::Error`, which is scoped to the agent loop and its gateway
//! transport (see that type's own doc comment): this module's HTTP calls are a wholly different
//! network surface — talking directly to `claude.ai`/`github.com`/`auth.openai.com`, never through
//! the gateway — so their failure modes don't belong mixed into that type. Converts to this crate's
//! `main.rs`/`serve.rs` `Box<dyn std::error::Error>` boundary via `?` like any other error here.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum OAuthError {
    #[error("network request to {url} failed: {source}")]
    Network {
        url: String,
        #[source]
        source: reqwest::Error,
    },

    #[error("{provider} returned {status}: {body}")]
    Http {
        provider: &'static str,
        status: u16,
        body: String,
    },

    #[error("OAuth state mismatch")]
    StateMismatch,

    #[error("missing authorization code")]
    MissingAuthorizationCode,

    #[error("could not extract required claim `{claim}` from access token")]
    MissingJwtClaim { claim: &'static str },

    /// The full, already-composed message — see `device_code::poll_device_code`'s doc comment for
    /// why the "did we ever see a slow_down" distinction lives there rather than as a bare bool here
    /// (keeping the exact wording, including the WSL/VM clock-drift hint, next to the one place that
    /// decides which message applies).
    #[error("{0}")]
    DeviceFlowTimedOut(String),

    #[error("Device flow failed: {0}")]
    DeviceFlowFailed(String),

    #[error("login cancelled")]
    LoginCancelled,

    #[error("failed to bind local OAuth callback listener on {host}:{port}: {detail}")]
    PortBindFailed {
        host: String,
        port: u16,
        detail: String,
    },

    #[error("random number generation failed: {0}")]
    Random(#[from] getrandom::Error),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("invalid response from {provider}: {detail}")]
    InvalidResponse {
        provider: &'static str,
        detail: String,
    },
}

pub type Result<T> = std::result::Result<T, OAuthError>;
