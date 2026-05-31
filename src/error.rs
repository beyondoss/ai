//! Structured error type (PATTERNS.md convention: `thiserror` enum, `From` for foreign errors).

#[derive(Debug, thiserror::Error)]
pub enum GatewayError {
    #[error("configuration error: {0}")]
    Config(String),

    #[error("store error: {0}")]
    Store(#[from] store::KvError),

    #[error("dns resolution error: {0}")]
    Dns(String),
}

pub type Result<T> = std::result::Result<T, GatewayError>;
