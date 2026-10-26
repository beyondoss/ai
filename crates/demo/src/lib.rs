//! The code behind `deploy/demo`: the key and grant minters, the
//! chat client, the stand-in managed-MCP endpoint, and the broker that acts as a signed-in user.
//! `main.rs` is only argument parsing.

pub mod chat;
pub mod http;
pub mod keys;
pub mod load;
pub mod load_host;
#[cfg(feature = "agent-lib")]
pub mod managed;
#[cfg(feature = "agent-lib")]
pub mod oidc;
