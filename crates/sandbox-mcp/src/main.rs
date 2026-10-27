//! `sandbox-mcp`: stdio by default; `SANDBOX_MCP_LISTEN=127.0.0.1:PORT` for streamable HTTP.

use rmcp::ServiceExt as _;
use sandbox_mcp::{Config, SandboxMcp, config, http};
use tokio_util::sync::CancellationToken;

fn main() {
    if std::env::args().len() > 1 {
        eprint!("{}", config::USAGE);
        std::process::exit(2);
    }
    let config = match Config::from_env(|k| std::env::var(k).ok()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("sandbox-mcp: {e}\n\n{}", config::USAGE);
            std::process::exit(2);
        }
    };
    let runtime = match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("sandbox-mcp: cannot start the runtime: {e}");
            std::process::exit(1);
        }
    };
    if let Err(e) = runtime.block_on(run(config)) {
        eprintln!("sandbox-mcp: {e}");
        std::process::exit(1);
    }
}

async fn run(config: Config) -> Result<(), String> {
    let listen = config.listen.clone();
    let insecure = config.insecure_no_auth;
    let server = SandboxMcp::new(config);
    let stop = CancellationToken::new();
    tokio::spawn({
        let stop = stop.clone();
        async move {
            wait_for_signal().await;
            stop.cancel();
        }
    });
    match listen {
        Some(listen) => {
            let listener = http::bind(&listen, insecure).await?;
            if let Ok(addr) = listener.local_addr() {
                eprintln!("sandbox-mcp: listening on http://{addr}/mcp");
            }
            http::serve(listener, server, insecure, stop).await;
            Ok(())
        }
        None => {
            let service = server
                .serve(rmcp::transport::stdio())
                .await
                .map_err(|e| format!("MCP handshake failed: {e}"))?;
            tokio::select! {
                _ = service.waiting() => {}
                _ = stop.cancelled() => {}
            }
            Ok(())
        }
    }
}

async fn wait_for_signal() {
    use tokio::signal::unix::{SignalKind, signal};
    let (Ok(mut term), Ok(mut int)) = (
        signal(SignalKind::terminate()),
        signal(SignalKind::interrupt()),
    ) else {
        return std::future::pending().await;
    };
    tokio::select! {
        _ = term.recv() => {}
        _ = int.recv() => {}
    }
}
