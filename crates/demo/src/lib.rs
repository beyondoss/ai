//! The code behind `deploy/demo`: the sandbox's exec endpoint, the key and grant minters, the
//! chat client, and the stand-in managed-MCP endpoint. `main.rs` is only argument parsing.

pub mod chat;
pub mod keys;
#[cfg(feature = "agent-lib")]
pub mod managed;
pub mod stub;
