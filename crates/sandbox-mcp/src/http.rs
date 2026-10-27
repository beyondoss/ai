//! Streamable HTTP: `SANDBOX_MCP_LISTEN`.
//!
//! The server holds the sandbox's token, so whoever can reach this port can run commands in the
//! sandbox. It therefore listens on loopback only (and rmcp refuses a `Host` that is not loopback,
//! which stops DNS rebinding from a browser) unless `SANDBOX_MCP_INSECURE_NO_AUTH` says the
//! network in front of it is the authentication.

use std::net::SocketAddr;
use std::sync::Arc;

use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder;
use hyper_util::service::TowerToHyperService;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

use crate::server::SandboxMcp;

/// Bind `listen`, refusing a non-loopback address unless `insecure_no_auth`. Returns the listener,
/// so a caller (and a test) can read the port it got.
pub async fn bind(listen: &str, insecure_no_auth: bool) -> Result<TcpListener, String> {
    let addrs = tokio::net::lookup_host(listen)
        .await
        .map_err(|e| format!("SANDBOX_MCP_LISTEN={listen:?}: {e}"))?
        .collect::<Vec<SocketAddr>>();
    let first = addrs
        .first()
        .ok_or_else(|| format!("SANDBOX_MCP_LISTEN={listen:?} resolves to nothing"))?;
    if !insecure_no_auth && addrs.iter().any(|a| !a.ip().is_loopback()) {
        return Err(format!(
            "SANDBOX_MCP_LISTEN={listen:?} is not a loopback address, and anyone who can reach it \
             could run commands in the sandbox. Put an authenticating proxy in front and set \
             SANDBOX_MCP_INSECURE_NO_AUTH=1, or listen on 127.0.0.1"
        ));
    }
    TcpListener::bind(first)
        .await
        .map_err(|e| format!("cannot listen on {first}: {e}"))
}

/// Serve MCP over streamable HTTP on `listener` until `stop` is cancelled.
pub async fn serve(
    listener: TcpListener,
    server: SandboxMcp,
    insecure_no_auth: bool,
    stop: CancellationToken,
) {
    let mut config =
        StreamableHttpServerConfig::default().with_cancellation_token(stop.child_token());
    if insecure_no_auth {
        config = config.disable_allowed_hosts();
    }
    let service = StreamableHttpService::new(
        move || Ok(server.clone()),
        Arc::new(LocalSessionManager::default()),
        config,
    );
    loop {
        let (stream, _) = tokio::select! {
            _ = stop.cancelled() => return,
            accepted = listener.accept() => match accepted {
                Ok(a) => a,
                Err(_) => continue,
            },
        };
        let service = TowerToHyperService::new(service.clone());
        let stop = stop.clone();
        tokio::spawn(async move {
            let builder = Builder::new(TokioExecutor::new());
            let conn = builder.serve_connection(TokioIo::new(stream), service);
            tokio::select! {
                _ = stop.cancelled() => {}
                _ = conn => {}
            }
        });
    }
}
