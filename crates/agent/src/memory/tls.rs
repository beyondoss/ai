//! Shared rustls client config for the networked memory backends.
//!
//! Both Redis (`rediss://`) and Postgres (`sslmode=require`/`prefer`) terminate TLS with the
//! workspace's rustls/`ring` provider — the same stack `reqwest` already uses. Native roots come
//! from the OS trust store (`rustls-native-certs`); we do not bundle Mozilla's webpki-roots, which
//! would be a second, drifting set of CAs.

use std::sync::Arc;

use rustls::ClientConfig;
use rustls::RootCertStore;

/// A rustls client config using the OS trust store and the process-wide `ring` provider.
pub fn client_config() -> Result<Arc<ClientConfig>, String> {
    agent_core::ensure_provider();
    let native = rustls_native_certs::load_native_certs();
    if native.certs.is_empty() {
        let detail = if native.errors.is_empty() {
            "no certificates returned".to_string()
        } else {
            native
                .errors
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("; ")
        };
        return Err(format!(
            "failed to load native TLS root certificates: {detail}"
        ));
    }
    let mut roots = RootCertStore::empty();
    for cert in native.certs {
        if roots.add(cert).is_err() {
            // A single malformed CA must not take down the whole store; skip it.
            continue;
        }
    }
    if roots.is_empty() {
        return Err("native TLS root store contained no usable certificates".into());
    }
    Ok(Arc::new(
        ClientConfig::builder()
            .with_root_certificates(roots)
            .with_no_client_auth(),
    ))
}
