//! The `bsg_v1` minter now lives in `beyond-ai-test-support`, because the fleet simulator's edge
//! double mints grants too and a second implementation would be a second thing to get wrong. Its
//! independence from `src/grant.rs` — the property that makes these tests worth anything — is
//! unchanged: it is still written from the spec, and `serve_service_grant.rs` still re-mints the
//! golden fixture and compares byte for byte.
pub use beyond_ai_test_support::grant::*;
