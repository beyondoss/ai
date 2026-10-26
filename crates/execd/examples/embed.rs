//! A host that embeds execd with its own authorizer: the minimal correct `main()` that
//! ARCHITECTURE.md's "Embedding" section shows. `EMBED_KEY=secret cargo run -p beyond-ai-execd
//! --example embed`, then send `X-Host-Key: secret`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

fn main() -> ExitCode {
    // First, before anything else, if commands run as another user (`Options::run_as`): as the
    // switch helper this process becomes a command and never returns here.
    if let Some(code) = execd::helper_main() {
        return code;
    }
    // The library logs nothing until told a level.
    execd::log::set_level(execd::log::Level::Info);
    let key = std::env::var("EMBED_KEY").expect("EMBED_KEY");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let result: std::io::Result<()> = runtime.block_on(async move {
        // Orphans reparent to this process and are collected at once. Without it execd still
        // works, polling a command's group every 50 ms until it is empty.
        if let Err(e) = execd::become_subreaper() {
            execd::log_warn!("cannot become a subreaper: {e}");
        }
        let mut opts = execd::Options::new("/tmp");
        opts.authorizer = Some(Arc::new(move |headers: &execd::HeaderMap| {
            headers
                .get("x-host-key")
                .is_some_and(|v| v.as_bytes() == key.as_bytes())
        }));
        opts.idle_timeout = Some(Duration::from_secs(30));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:8080").await?;
        let server = execd::serve(listener, opts)?;
        tokio::signal::ctrl_c().await?;
        server.shutdown().await;
        Ok(())
    });
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            execd::log_error!("embedded execd failed: {e}");
            ExitCode::FAILURE
        }
    }
}
