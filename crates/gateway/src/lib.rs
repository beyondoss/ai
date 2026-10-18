//! Beyond AI gateway library.
//!
//! `src/main.rs` wires these modules into a Pingora `ProxyHttp` service. The load-bearing logic
//! (virtual-key verification, deny-set, usage parsing, routing, request peek) lives in modules
//! free of Pingora/IO so it is unit-tested without a running proxy or live providers.
//!
//! Every module is crate-private. The public surface is what the binary wires together (the
//! re-exports below, and `doctor`); the integration tests, examples and benches reach the rest
//! through [`__test`] and [`__bench`], which are not an API.

// Lint gates (`unsafe_code = "forbid"`, `unused_must_use = "deny"`) live in `[workspace.lints]` so
// they apply to *both* crate roots — this lib and the `main.rs` binary — not just whichever unit
// carries a crate-level `#![deny]`. A dropped `Result` (e.g. an unchecked `write_response_*`) is
// therefore a hard error, and `unsafe` is forbidden, everywhere in the crate.
//
// `unwrap_used`/`expect_used`/`panic` are denied in production code (see `[workspace.lints.clippy]`)
// but a unit test's whole job is to assert a precondition holds — `.unwrap()` *is* the assertion — so
// allow them in `#[cfg(test)]` modules.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub(crate) mod admin;
pub(crate) mod allowance;
pub(crate) mod body_rules;
pub(crate) mod cache;
pub(crate) mod capture;
pub(crate) mod capture_sink;
pub(crate) mod circuit_breaker;
pub(crate) mod concurrency;
pub(crate) mod config;
pub(crate) mod control;
pub(crate) mod deadline;
pub(crate) mod deny;
pub mod doctor;
pub(crate) mod drain;
pub(crate) mod error;
pub(crate) mod key;
pub(crate) mod lex;
pub(crate) mod metrics;
pub(crate) mod peek;
pub(crate) mod pin;
pub(crate) mod proxy;
pub(crate) mod ratelimit;
pub(crate) mod refusal;
pub(crate) mod remedy;
pub(crate) mod retry;
mod rewrite;
pub(crate) mod route;
pub(crate) mod secret;
pub(crate) mod signed_id;
pub(crate) mod state;
pub(crate) mod store_watch;
pub(crate) mod terminal;
pub(crate) mod translate;
pub(crate) mod unpriced;
pub(crate) mod usage;
pub(crate) mod usage_file;

// The binary's surface: what `main.rs` wires into the Pingora server.
pub use admin::AdminApp;
pub use capture_sink::{CaptureDrain, CaptureSink};
pub use config::{AiConfig, ValidConfig};
pub use metrics::Metrics;
pub use proxy::AiProxy;
pub use state::GatewayState;
pub use store_watch::{Allowance, Capture, Deny, WatcherService};
pub use usage::{USAGE_TARGET, usage_log_filter};
pub use usage_file::{UsageFiles, UsageRow};

/// What the integration tests (`tests/`) and `examples/` reach past the binary's surface, by
/// module. Not an API: hidden from docs, and changed with the tests that use it.
#[doc(hidden)]
pub mod __test {
    pub mod control {
        pub use crate::control::Walk;
    }
    pub mod key {
        pub use crate::key::{Keyring, VirtualKey};
        #[cfg(feature = "mint")]
        pub use crate::key::{mint, mint_v2};
    }
    pub mod lex {
        pub use crate::lex::{unescape, unescapes_to};
    }
    pub mod pin {
        pub use crate::pin::{affinity, order};
    }
    pub mod peek {
        pub use crate::peek::{
            BufferedScan, Kept, MAX_CAPTURE, Member, ModelScanner, OUTPUT_LIMIT_KEYS, Structure,
            elements, has_typed_member, last_member, members, remove_element_members, remove_items,
            remove_root_members, remove_root_nulls, root_members, root_open, scan_buffered,
            set_root_member, str_is, str_value, structure, value_end,
        };
    }
    pub mod route {
        pub use crate::route::{
            Dialect, Endpoint, OPENROUTER_402_STRIKES, SubResource, WireAction,
            catalog_wire_action, implied_endpoint, is_default_prefix, is_responses_path,
        };
    }
    pub mod signed_id {
        pub use crate::signed_id::{Relay, Signer};
    }
    pub mod translate {
        pub use crate::translate::{
            SseBridge, request, response_json, response_json_status, responses_session_field,
            translation_heap,
        };
    }
    pub mod usage {
        pub use crate::usage::{
            InputTally, Usage, anthropic_body, anthropic_stream, anthropic_stream_finished,
            anthropic_stream_parts, body, estimate_body_output, estimate_stream_output,
            openai_body, openai_stream, stream, stream_carried_error,
        };
    }
    /// The body with its scan, through the methods `request_body_filter` calls on it.
    pub use crate::rewrite::ScannedBody;
}

/// What `benches/` measure directly, by module. Not an API: hidden from docs, and changed with the
/// benches that use it.
#[doc(hidden)]
pub mod __bench {
    pub mod admin {
        pub use crate::admin::{AdminApp, HEALTH_OK};
    }
    pub mod allowance {
        pub use crate::allowance::{AllowanceSet, AllowanceTarget};
    }
    pub mod cache {
        pub use crate::cache::{CacheKey, CachedResponse, KeyParts, ResponseCache, key};
    }
    pub mod capture {
        pub use crate::capture::{CaptureBufs, CaptureRule, CaptureSet};
    }
    pub mod circuit_breaker {
        pub use crate::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
    }
    pub mod control {
        pub use crate::control::{Control, Walk};
    }
    pub mod deny {
        pub use crate::deny::{DenyReason, DenySet, DenyTarget, parse_key, parse_reason};
    }
    pub mod key {
        pub use crate::key::{Keyring, VirtualKey};
        #[cfg(feature = "mint")]
        pub use crate::key::{mint, mint_v2};
    }
    pub mod metrics {
        pub use crate::metrics::{Metrics, ProviderMetrics, Rejection};
    }
    pub mod peek {
        pub use crate::peek::{ModelScanner, scan_buffered};
    }
    pub mod pin {
        pub use crate::pin::{affinity, order};
    }
    pub mod ratelimit {
        pub use crate::ratelimit::RateLimit;
    }
    pub mod refusal {
        pub use crate::refusal::Refusal;
    }
    pub mod route {
        pub use crate::route::{
            Dialect, Endpoint, ModelRoute, dialect_default, known_providers, refused_input,
            unserved,
        };
    }
    pub mod signed_id {
        pub use crate::signed_id::{Relay, Signer};
    }
    pub mod store_watch {
        pub use crate::store_watch::apply_batch;
    }
    pub mod translate {
        pub use crate::translate::{
            SseBridge, claude_chat_relay_reasoning, request, response_json,
            responses_session_field, translation_heap,
        };
    }
    pub mod usage {
        pub use crate::usage::{
            InputTally, Usage, anthropic_body, anthropic_stream, openai_body, openai_stream,
        };
    }
    /// The body with its scan, through the methods `request_body_filter` calls on it.
    pub use crate::rewrite::ScannedBody;
}
