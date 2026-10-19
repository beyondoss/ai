//! The idle command table's handlers: one `impl SessionLoop` block per concern, one `cmd_<type>`
//! method per protocol command. `SessionLoop::run` reads a line, and `SessionLoop::dispatch` (in `control.rs`) calls these.
//! Each returns whether the session goes on (`false` ends the loop, as a closed writer does).
//!
//! A reply that a busy loop (a running `prompt` or `compact`) also gives is built by a `<type>_frame`
//! function over explicit fields, since the run holds `&mut session`; the `cmd_` method `emit`s it, the busy
//! loop sends it. `Activity` names where the two replies differ.

mod auth;
mod bash;
mod export;
mod mcp;
mod model;
mod sessions;
mod settings;
mod state;
mod tree;

pub(super) use sessions::{list_all_sessions_frame, list_sessions_frame};
pub(super) use state::{
    Activity, StateView, get_available_models_frame, get_commands_frame, get_messages_frame,
    get_session_stats_frame, get_state_frame,
};
pub(super) use tree::{get_tree_frame, list_branches_frame};
