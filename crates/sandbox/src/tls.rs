//! The rustls crypto provider for the HTTP client.
//!
//! `reqwest` is taken with `rustls-no-provider` (its plain `rustls` feature is hard-wired to
//! aws-lc-rs, a second crypto stack and a C toolchain dependency), and with no provider installed
//! `Client::builder().build()` panics. A library cannot count on the program's `main` having
//! installed one, so the client calls [`ensure_provider`] before building. `ring`, once; an
//! already-installed provider (the program's own) is left alone.

use std::sync::Once;

static INIT: Once = Once::new();

/// Install `ring` as the process-wide provider unless one is already installed.
pub(crate) fn ensure_provider() {
    INIT.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}
