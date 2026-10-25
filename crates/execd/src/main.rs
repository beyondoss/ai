//! The `execd` binary: `execd --listen 0.0.0.0:8080`.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;

/// An exec endpoint for a sandbox: runs the agent's commands and jobs (exec protocol v1.1 to v1.4).
#[derive(Parser, Debug)]
#[command(version)]
struct Cli {
    /// Address to listen on.
    #[arg(long, env = "EXECD_LISTEN", default_value = "127.0.0.1:8080")]
    listen: String,

    /// Where a command runs when its request names no directory (default: the current directory).
    #[arg(long, env = "EXECD_ROOT")]
    root: Option<PathBuf>,

    /// `HOME` for every command (default: execd's own).
    #[arg(long, env = "EXECD_HOME")]
    home: Option<PathBuf>,

    /// A file holding the bearer token every request must present. The environment variable
    /// `EXECD_BEARER` holds the token itself. Never a command-line argument: argv is visible in
    /// `ps`. With neither, execd accepts anyone who can reach it.
    #[arg(long, env = "EXECD_BEARER_FILE")]
    bearer_file: Option<PathBuf>,

    /// Say that a kept job survives this sandbox going idle, which turns on advertising v1.4
    /// job-end notify. Leave it off for a sandbox a vendor stops or kills when idle.
    #[arg(long, env = "EXECD_KEPT_JOBS_SURVIVE_IDLE")]
    kept_jobs_survive_idle: bool,

    /// Log level: error, warn, info, debug or trace (or a tracing filter).
    #[arg(long, env = "EXECD_LOG", default_value = "info")]
    log_level: String,
}

fn bearer(cli: &Cli) -> std::io::Result<Option<String>> {
    let raw = match (&cli.bearer_file, std::env::var("EXECD_BEARER")) {
        (Some(path), _) => Some(std::fs::read_to_string(path)?),
        (None, Ok(v)) => Some(v),
        (None, Err(_)) => None,
    };
    Ok(raw
        .map(|s| s.trim_end_matches(['\n', '\r']).to_string())
        .filter(|s| !s.is_empty()))
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new(&cli.log_level))
        .with_target(false)
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()))
        .with_writer(std::io::stderr)
        .init();
    match run(cli).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            tracing::error!(error = %e, "execd failed");
            ExitCode::FAILURE
        }
    }
}

async fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    let bearer = bearer(&cli)?;
    let root = match &cli.root {
        Some(r) => r.clone(),
        None => std::env::current_dir()?,
    };
    // Orphans reparent to execd and are reaped here, so a PID 1 (a container, E2B) never fills the
    // sandbox with zombies, and a zombie never counts as a live member of a group.
    if let Err(e) = execd::become_subreaper() {
        tracing::warn!(error = %e, "cannot become a subreaper; orphans go to the host's init");
    }
    let reaper_stop = tokio_util::sync::CancellationToken::new();
    tokio::spawn(execd::reap_orphans(reaper_stop.clone()));

    let mut opts = execd::Options::new(root);
    opts.home = cli.home;
    opts.kept_jobs_survive_idle = cli.kept_jobs_survive_idle;
    let secured = bearer.is_some();
    opts.bearer = bearer;

    let listener = tokio::net::TcpListener::bind(&cli.listen).await?;
    let server = execd::serve(listener, opts)?;
    tracing::info!(
        listen = %server.local_addr(),
        bearer = secured,
        notify = cli.kept_jobs_survive_idle,
        "execd listening"
    );
    if !secured {
        tracing::warn!(
            "NO BEARER CONFIGURED: anyone who can reach {} can run commands in this sandbox. \
             Set EXECD_BEARER_FILE (or EXECD_BEARER) to require one.",
            server.local_addr()
        );
    }

    wait_for_signal().await;
    let killed = server.shutdown().await;
    reaper_stop.cancel();
    tracing::info!(killed_groups = killed, "execd stopped");
    Ok(())
}

async fn wait_for_signal() {
    use tokio::signal::unix::{SignalKind, signal};
    let (Ok(mut term), Ok(mut int)) = (
        signal(SignalKind::terminate()),
        signal(SignalKind::interrupt()),
    ) else {
        // No signal handling to wait on: serve until killed.
        std::future::pending::<()>().await;
        return;
    };
    tokio::select! {
        _ = term.recv() => tracing::info!("SIGTERM"),
        _ = int.recv() => tracing::info!("SIGINT"),
    }
}
