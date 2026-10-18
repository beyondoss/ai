//! TLS for `serve --tls-listen`: load the certificate chain and key once at startup, and finish each
//! connection's handshake under a deadline before it reaches the same `handle_connection` the plain
//! `--listen` listener feeds.
//!
//! The provider is the process default (`agent_core::ensure_provider`, `ring`), the one every
//! `reqwest` client in the workspace already needs. TLS 1.2 and 1.3; ALPN offers only `http/1.1`,
//! because the control protocol is HTTP/1.1 with a WebSocket upgrade. The certificate is read once:
//! rotating it takes a restart.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use rustls::pki_types::{CertificateDer, PrivateKeyDer, pem::PemObject};
use tokio::net::TcpStream;
use tokio_rustls::{TlsAcceptor, server::TlsStream};

/// How long a client gets to finish the TLS handshake. Past it the connection is dropped, so a peer
/// that connects and says nothing (or trickles a ClientHello) cannot hold a task and a descriptor.
pub(super) const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

/// An acceptor for `cert` (PEM chain, leaf first) and `key` (PKCS#8, PKCS#1 or SEC1 PEM). Every
/// failure names the flag and the path: a missing or empty file, a file that holds no key, and a key
/// that does not belong to the leaf certificate.
pub fn load_acceptor(cert: &Path, key: &Path) -> Result<TlsAcceptor, String> {
    agent_core::ensure_provider();
    let certs = CertificateDer::pem_file_iter(cert)
        .and_then(|it| it.collect::<Result<Vec<_>, _>>())
        .map_err(|e| format!("--tls-cert {}: {e}", cert.display()))?;
    if certs.is_empty() {
        return Err(format!(
            "--tls-cert {}: no certificate found",
            cert.display()
        ));
    }
    let key_der = PrivateKeyDer::from_pem_file(key)
        .map_err(|e| format!("--tls-key {}: {e}", key.display()))?;
    let mut config = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key_der)
        .map_err(|e| {
            format!(
                "--tls-cert {} and --tls-key {} do not work together: {e}",
                cert.display(),
                key.display()
            )
        })?;
    config.alpn_protocols = vec![b"http/1.1".to_vec()];
    Ok(TlsAcceptor::from(Arc::new(config)))
}

/// Finish the handshake on an accepted socket within [`HANDSHAKE_TIMEOUT`].
pub(super) async fn handshake(
    acceptor: &TlsAcceptor,
    stream: TcpStream,
) -> Result<TlsStream<TcpStream>, String> {
    handshake_within(acceptor, stream, HANDSHAKE_TIMEOUT).await
}

async fn handshake_within(
    acceptor: &TlsAcceptor,
    stream: TcpStream,
    limit: Duration,
) -> Result<TlsStream<TcpStream>, String> {
    match tokio::time::timeout(limit, acceptor.accept(stream)).await {
        Ok(Ok(tls)) => Ok(tls),
        Ok(Err(e)) => Err(e.to_string()),
        Err(_) => Err(format!("no handshake within {}ms", limit.as_millis())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A client that connects and says nothing is dropped at the deadline, not held forever.
    #[tokio::test]
    async fn a_silent_client_is_dropped_at_the_handshake_deadline() {
        let dir = tempfile::tempdir().unwrap();
        let ck = rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).unwrap();
        let (cert, key) = (dir.path().join("c.pem"), dir.path().join("k.pem"));
        std::fs::write(&cert, ck.cert.pem()).unwrap();
        std::fs::write(&key, ck.key_pair.serialize_pem()).unwrap();
        let acceptor = load_acceptor(&cert, &key).unwrap();

        let listener = beyond_ai_test_support::ports::tokio_listener().await;
        let _silent = TcpStream::connect(listener.local_addr().unwrap())
            .await
            .unwrap();
        let (accepted, _) = listener.accept().await.unwrap();

        let err = handshake_within(&acceptor, accepted, Duration::from_millis(100))
            .await
            .unwrap_err();
        assert!(err.contains("no handshake within"), "{err}");
    }
}
