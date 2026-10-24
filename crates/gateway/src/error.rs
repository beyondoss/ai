//! Structured error types (PATTERNS.md convention: `thiserror` enums, `From` for foreign errors).
//!
//! Every `#[error]` string is the text an operator sees in a boot failure, `--doctor` output or a
//! log line, and greps for: changing one is a user-visible change, not a refactor. The structured
//! data behind each message (the path, provider, value or key) is in the variant for code to match.

use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use crate::key::Kid;
use crate::route::AUTO_SEGMENT;
use crate::signed_id::SignerError;

/// A gateway failure, with its cause.
///
/// Each Display is the whole operator-facing message, cause included (`dns resolution error:
/// {authority}: {cause}`), because that text is what boot output, `--doctor` and log lines show
/// and what operators grep. `source()` still returns the cause, for code that inspects it. So a
/// reporter that walks the chain prints the cause twice: print an error's Display alone, and never
/// format its `source()` chain after it. The same holds for [`ConfigError`].
#[derive(Debug, thiserror::Error)]
pub enum GatewayError {
    #[error("configuration error: {0}")]
    Config(#[from] ConfigError),

    #[error("store error: {0}")]
    Store(#[from] store::KvError),

    /// One lookup's failure is shared by every request that waited on it (single flight, see
    /// `state::DnsCache`), so the cause is behind an `Arc`.
    #[error("dns resolution error: {authority}: {source}")]
    Dns {
        authority: String,
        #[source]
        source: Arc<io::Error>,
    },
}

/// Why a config cannot boot a gateway: one variant per check in `config.rs` and `state.rs`.
///
/// Display already includes the cause where there is one, as it does on [`GatewayError`]:
/// `source()` is there for code, not for printing after the message.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// figment could not read the defaults or the environment, or extract the merged config.
    #[error("{0}")]
    Figment(#[source] Box<figment::Error>),

    #[error("failed to parse {}: {source}", path.display())]
    ParseFile {
        path: PathBuf,
        source: Box<figment::Error>,
    },

    #[error(
        "unknown key(s) in {}: {} — check for a typo (known keys: {})",
        path.display(),
        unknown.join(", "),
        known.join(", ")
    )]
    UnknownKeys {
        path: PathBuf,
        unknown: Vec<String>,
        known: Vec<String>,
    },

    #[error("[{table}] provider name {name:?} must be letters, digits, '-' or '_'")]
    ProviderNameCharset { table: &'static str, name: String },

    #[error(
        "[{table}] provider name {name:?} is {lower:?} once lowercased, which another name already is"
    )]
    ProviderNameCollision {
        table: &'static str,
        name: String,
        lower: String,
    },

    #[error("connect_timeout_secs must be > 0 (a 0 connect timeout fails every upstream connect)")]
    ConnectTimeoutZero,

    #[error(
        "read_timeout_secs must be > 0 (a 0 read timeout aborts every response before it arrives)"
    )]
    ReadTimeoutZero,

    // The run of spaces mid-message is in the text operators have always been shown; it stays
    // until the message is changed on purpose.
    #[error(
        "tcp_keepalive_interval_secs and tcp_keepalive_count must be > 0 when                  tcp_keepalive_idle_secs is set; set tcp_keepalive_idle_secs = 0 to disable keepalive"
    )]
    KeepaliveProbesZero,

    #[error(
        "circuit_breaker_window_secs must be > 0 (a 0 window never accrues a second failure, so \
         the breaker never opens); set circuit_breaker_threshold = 0 to disable it"
    )]
    BreakerWindowZero,

    #[error(
        "circuit_breaker_reset_secs must be > 0 (a 0 reset half-opens the breaker the instant it \
         opens); set circuit_breaker_threshold = 0 to disable it"
    )]
    BreakerResetZero,

    #[error(
        "circuit_breaker_threshold = {threshold} exceeds the maximum of {max} (the breaker's packed \
         failure count cannot hold more)",
        max = crate::circuit_breaker::MAX_FAILURE_THRESHOLD
    )]
    BreakerThresholdTooHigh { threshold: u32 },

    #[error(
        "tls_listen, tls_cert and tls_key must be set together (set none of them to serve plain \
         HTTP only)"
    )]
    TlsPartial,

    #[error("tls_listen is empty")]
    TlsListenEmpty,

    #[error("tls_cert {path}: {source}")]
    TlsCert {
        path: String,
        source: rustls::pki_types::pem::Error,
    },

    #[error("tls_cert {path}: no certificate found")]
    TlsNoCertificate { path: String },

    #[error("tls_key {path}: {source}")]
    TlsKey {
        path: String,
        source: rustls::pki_types::pem::Error,
    },

    #[error("tls: {0}")]
    TlsProtocols(#[source] rustls::Error),

    #[error("tls_cert {cert} and tls_key {key} do not work together: {source}")]
    TlsMismatch {
        cert: String,
        key: String,
        source: rustls::Error,
    },

    #[error("invalid signing key id {kid}")]
    SigningKeyId { kid: String },

    #[error("invalid signing public key for kid {kid}")]
    SigningPublicKey { kid: Kid },

    #[error(transparent)]
    IdSigner(#[from] SignerError),

    #[error(
        "signing_keys are configured (managed traffic) but no id_signing_keys are — refusing to \
         boot: managed Responses ids would be resolvable by every tenant. Set AI_ID_SIGNING_KEY_1 \
         to the base64 of 32 random bytes (e.g. `openssl rand -base64 32`)."
    )]
    IdSigningKeysMissing,

    #[error(
        "nats_url names a plaintext connection to {host:?}, which is not loopback — refusing to \
         boot: the NATS channel carries signing keys and deny entries, and async-nats upgrades to \
         TLS only if the server demands it. Use a tls:// URL (or nats://localhost for a local server)."
    )]
    NatsPlaintext { host: String },

    #[error(
        "require_signing_keys is set but no signing_keys are configured — refusing to boot into a \
         mode where every bai_v1 token 401s. Check the signing_keys config / SSM param."
    )]
    SigningKeysRequired,

    #[error(
        "provider_authorities.{AUTO_SEGMENT} = {authority:?} uses a reserved name: {AUTO_SEGMENT}/… \
         is the model-routed route, which a provider of that name would shadow"
    )]
    ReservedProviderName { authority: String },

    #[error(
        "provider_dialects.{provider} = {value:?} is not a recognized dialect (expected \"openai\" \
         or \"anthropic\")"
    )]
    UnknownDialect { provider: String, value: String },

    #[error(
        "provider_auth_schemes.{provider} = {value:?} is not a recognized auth scheme (expected \
         \"bearer\", \"x-api-key\", or \"api-key\")"
    )]
    UnknownAuthScheme { provider: String, value: String },
}

impl From<figment::Error> for ConfigError {
    fn from(e: figment::Error) -> Self {
        Self::Figment(Box::new(e))
    }
}

pub type Result<T> = std::result::Result<T, GatewayError>;
