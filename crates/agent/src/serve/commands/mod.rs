//! The idle command table's handlers: one `impl SessionLoop` block per concern, one `cmd_<type>`
//! method per protocol command. `SessionLoop::run` reads a line, and its dispatch `match` calls these.
//! Each returns whether the session goes on (`false` ends the loop, as a closed writer does).

mod auth;
mod bash;
mod export;
mod mcp;
mod model;
mod sessions;
mod settings;
mod state;
mod tree;
