//! The job client's kernel (`job_client::step`) behind a C ABI. See `src/codec.rs` for the byte layouts
//! and `src/abi.rs` for the exports. This crate holds no decision: every function decodes its input,
//! makes one call into `job_client`, and encodes the answer.

mod abi;
pub mod codec;

pub use abi::*;
