//! Subcommand bodies, one module per group. `main.rs::run` parses the command line and dispatches
//! here; each function takes the flags of its subcommand and owns that subcommand's whole setup.

pub(crate) mod auth;
pub(crate) mod inspect;
pub(crate) mod serve;
pub(crate) mod settings;
pub(crate) mod trust;
