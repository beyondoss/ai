//! The process-wide rustls crypto provider.
//!
//! This workspace takes reqwest's `rustls-no-provider` feature rather than its plain `rustls` one.
//! The plain feature is hard-wired to **aws-lc-rs** (`__rustls-aws-lc-rs`), which links the
//! `aws-lc-sys` C blob — a second crypto implementation alongside the `ring` that `crates/gateway`
//! already pins, ~1.3 MiB of `.text` in a binary that ships into a memory-tight guest, and a C
//! toolchain dependency that every cross-build (notably the static musl one) then has to satisfy.
//!
//! The cost of `-no-provider` is that reqwest installs nothing itself and **panics** inside
//! `Client::builder().build()` when no process default exists. That makes provider installation a
//! process-global precondition rather than a link-time fact, so it cannot live only in `main()`:
//! a unit test, a bench, or any library consumer constructing a client never runs `main()`.
//! [`ensure_provider`] is therefore called at every client-construction site in this workspace. It is
//! idempotent and cheap after the first call (a `Once`), so calling it defensively costs nothing.
//!
//! `install_default` returns `Err` only when a provider is already installed — including one another
//! caller installed first, which is a benign race, not a failure. Nothing here can fail in a way worth
//! propagating, so the return is deliberately `()`.

use std::sync::Once;

static INIT: Once = Once::new();

/// Install `ring` as the process-wide rustls crypto provider, once.
///
/// Call before constructing any `reqwest::Client` (or any other rustls user). Safe to call from
/// multiple threads and any number of times.
pub fn ensure_provider() {
    INIT.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}
