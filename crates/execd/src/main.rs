//! The `execd` binary: `execd --listen 0.0.0.0:8080`.

use std::os::fd::IntoRawFd as _;
use std::os::unix::fs::PermissionsExt as _;
use std::os::unix::process::CommandExt as _;
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

    /// `HOME` for every command (default: execd's own, or with `--run-as` the user's own).
    #[arg(long, env = "EXECD_HOME")]
    home: Option<PathBuf>,

    /// A file holding the bearer token every request must present (mode 0600 is best). The
    /// environment variable `EXECD_BEARER` holds the token itself. Never a command-line argument:
    /// argv is visible in `ps`. With neither, execd refuses to start unless it listens on a
    /// loopback address or `--insecure-no-auth` says so.
    #[arg(long, env = "EXECD_BEARER_FILE")]
    bearer_file: Option<PathBuf>,

    /// Serve without a bearer on a non-loopback address: anyone who can reach execd can run
    /// commands in this sandbox. Only for a network that is itself the access control.
    #[arg(long, env = "EXECD_INSECURE_NO_AUTH")]
    insecure_no_auth: bool,

    /// Run every command as this user, `user`, `uid`, `user:group` or `uid:gid`, with the user's
    /// groups, instead of execd's own uid. Then a command cannot read execd's memory, its
    /// environment or a bearer file execd owns. execd must be root (or hold CAP_SETUID, CAP_SETGID
    /// and CAP_KILL) and the user must not be execd's own.
    #[arg(long, env = "EXECD_RUN_AS")]
    run_as: Option<String>,

    /// Say that a kept job survives this sandbox going idle, which turns on advertising v1.4
    /// job-end notify. Leave it off for a sandbox a vendor stops or kills when idle.
    #[arg(long, env = "EXECD_KEPT_JOBS_SURVIVE_IDLE")]
    kept_jobs_survive_idle: bool,

    /// The most output one v1.1 answer carries, in bytes; past it the answer is an error saying
    /// the command ran. Match the agent's `--exec-max-response-bytes`.
    #[arg(long, env = "EXECD_MAX_RESPONSE_BYTES", default_value_t = execd::MAX_RESPONSE_BYTES)]
    max_response_bytes: usize,

    /// Where kept jobs' retained output (the last 8 MiB of each) is spooled, in a directory of
    /// execd's own. Default: `$TMPDIR`, else `/tmp`. Use a disk-backed directory: on tmpfs the
    /// output is RAM.
    #[arg(long, env = "EXECD_JOBS_DIR")]
    jobs_dir: Option<PathBuf>,

    /// Log level: error, warn, info, debug or trace (or a tracing filter).
    #[arg(long, env = "EXECD_LOG", default_value = "info")]
    log_level: String,

    /// Log format: `text` (for a person) or `json` (one object per line, the event's fields beside
    /// `timestamp`, `level` and `message`, for a log pipeline).
    #[arg(long, env = "EXECD_LOG_FORMAT", default_value = "text")]
    log_format: LogFormat,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
enum LogFormat {
    Text,
    Json,
}

/// The variable that carries the bearer across the re-exec that scrubs it from the environment: the
/// number of an inherited descriptor holding it.
const BEARER_FD: &str = "EXECD_BEARER_FD";

/// If the bearer was handed over in `EXECD_BEARER`, move it out of the environment. A process's
/// `/proc/<pid>/environ` keeps showing the environment it was *started* with whatever the process
/// later does to its variables, and a command that runs as execd's uid can read it. So the token is
/// written to a memfd and execd re-executes itself without the variable; the new image's environ
/// never held the token. The descriptor is read and closed at once ([`take_inherited_bearer`]).
fn move_bearer_out_of_environ() {
    let Ok(token) = std::env::var("EXECD_BEARER") else {
        return;
    };
    let stash = || -> Result<i32, String> {
        let fd = nix::sys::memfd::memfd_create(c"execd-bearer", nix::sys::memfd::MFdFlags::empty())
            .map_err(|e| format!("memfd_create: {e}"))?;
        nix::unistd::write(&fd, token.as_bytes()).map_err(|e| format!("write: {e}"))?;
        Ok(fd.into_raw_fd())
    };
    let fd = match stash() {
        Ok(fd) => fd,
        Err(e) => {
            eprintln!(
                "execd: cannot move the bearer out of the environment ({e}); it stays in /proc/<execd>/environ"
            );
            return;
        }
    };
    let mut args = std::env::args_os();
    let mut cmd = std::process::Command::new("/proc/self/exe");
    if let Some(arg0) = args.next() {
        cmd.arg0(arg0);
    }
    let err = cmd
        .args(args)
        .env_remove("EXECD_BEARER")
        .env(BEARER_FD, fd.to_string())
        .exec();
    // Not replaced: carry on with the environment as it was.
    eprintln!("execd: cannot re-execute to scrub the environment: {err}");
    let _ = nix::unistd::close(fd);
}

/// `mimalloc` is a build option for measuring against the platform's malloc (a static musl
/// binary's is slow under threads); see ARCHITECTURE.md "Memory".
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// `EXECD_WORKER_THREADS`: unset or `1` is the single-threaded scheduler, `0` is one worker per
/// CPU, `N` is N workers. execd only moves bytes between sockets and pipes, so one thread carries
/// more than any sandbox asks of it, and a thread per CPU costs a stack and a heap each (see
/// ARCHITECTURE.md "Memory").
fn runtime() -> std::io::Result<tokio::runtime::Runtime> {
    let mut b = match std::env::var("EXECD_WORKER_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        None | Some(1) => tokio::runtime::Builder::new_current_thread(),
        Some(0) => tokio::runtime::Builder::new_multi_thread(),
        Some(n) => {
            let mut b = tokio::runtime::Builder::new_multi_thread();
            b.worker_threads(n);
            b
        }
    };
    b.enable_all().build()
}

/// The bearer a previous image of this process left in a descriptor, read and closed.
fn take_inherited_bearer() -> Option<String> {
    let fd: i32 = std::env::var(BEARER_FD).ok()?.parse().ok()?;
    let token = std::fs::read_to_string(format!("/proc/self/fd/{fd}")).ok();
    let _ = nix::unistd::close(fd);
    token
}

/// What `--bearer-file` / the environment hold, with trailing newlines trimmed. `Err` for a file
/// that is named but unreadable or empty: a bearer that was asked for must not silently vanish.
fn bearer(cli: &Cli, inherited: Option<String>) -> Result<Option<String>, String> {
    let trim = |s: String| s.trim_end_matches(['\n', '\r']).to_string();
    if let Some(path) = &cli.bearer_file {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read the bearer file {}: {e}", path.display()))?;
        let token = trim(text);
        if token.is_empty() {
            return Err(format!("the bearer file {} is empty", path.display()));
        }
        if let Ok(meta) = std::fs::metadata(path)
            && meta.permissions().mode() & 0o077 != 0
        {
            tracing::warn!(
                path = %path.display(),
                mode = format!("{:o}", meta.permissions().mode() & 0o777),
                "the bearer file is readable by others: chmod 600 it"
            );
        }
        return Ok(Some(token));
    }
    let raw = inherited.or_else(|| std::env::var("EXECD_BEARER").ok());
    Ok(raw.map(trim).filter(|s| !s.is_empty()))
}

fn main() -> ExitCode {
    // First, before anything is parsed or started: as the switch helper this process becomes a
    // command (see `execd::RunAs`).
    if let Some(code) = execd::helper_main() {
        return code;
    }
    move_bearer_out_of_environ();
    let inherited = take_inherited_bearer();
    let cli = Cli::parse();
    let fmt = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new(&cli.log_level))
        .with_target(false)
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()))
        .with_writer(std::io::stderr);
    match cli.log_format {
        LogFormat::Text => fmt.init(),
        LogFormat::Json => fmt.json().flatten_event(true).init(),
    }
    let runtime = match runtime() {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(error = %e, "execd failed");
            return ExitCode::FAILURE;
        }
    };
    match runtime.block_on(run(cli, inherited)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            tracing::error!(error = %e, "execd failed");
            ExitCode::FAILURE
        }
    }
}

/// Whether every address `listen` names is loopback.
async fn listens_on_loopback(listen: &str) -> std::io::Result<bool> {
    let mut any = false;
    for addr in tokio::net::lookup_host(listen).await? {
        if !addr.ip().to_canonical().is_loopback() {
            return Ok(false);
        }
        any = true;
    }
    Ok(any)
}

async fn run(cli: Cli, inherited: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    let bearer = bearer(&cli, inherited)?;
    let root = match &cli.root {
        Some(r) => r.clone(),
        None => std::env::current_dir()?,
    };
    // Fail closed: no bearer means anyone who can connect runs code, so that is allowed only where
    // "anyone who can connect" is this host, or when the operator said so out loud.
    if bearer.is_none() && !cli.insecure_no_auth && !listens_on_loopback(&cli.listen).await? {
        return Err(format!(
            "refusing to start: no bearer is configured and {} is not a loopback address, so \
             anyone who can reach it could run commands. Set EXECD_BEARER_FILE (or EXECD_BEARER), \
             listen on 127.0.0.1, or pass --insecure-no-auth to run open on purpose",
            cli.listen
        )
        .into());
    }
    let mut home = cli.home.clone();
    let run_as = match &cli.run_as {
        None => None,
        Some(spec) => {
            let resolved = execd::RunAs::resolve(spec)?;
            if let Some(warning) = resolved.run_as.check(&root)? {
                tracing::warn!("{warning}");
            }
            home = home.or(resolved.home);
            Some(resolved.run_as)
        }
    };
    // Orphans reparent to execd and are reaped here, so a PID 1 (a container, E2B) never fills the
    // sandbox with zombies, and a zombie never counts as a live member of a group.
    if let Err(e) = execd::become_subreaper() {
        tracing::warn!(error = %e, "cannot become a subreaper; orphans go to the host's init");
    }

    let mut opts = execd::Options::new(root);
    opts.home = home;
    opts.max_response_bytes = cli.max_response_bytes;
    opts.kept_jobs_survive_idle = cli.kept_jobs_survive_idle;
    opts.jobs_dir = cli.jobs_dir.clone();
    let secured = bearer.is_some();
    opts.bearer = bearer;
    opts.insecure_no_auth = cli.insecure_no_auth;
    let switched = run_as.is_some();
    opts.run_as = run_as;

    // Before the listener is bound: it reads the whole of /tmp, and a client that finds the port
    // open expects the server to be ready.
    execd::remove_stale_jobs_dirs(cli.jobs_dir.as_deref());
    let listener = tokio::net::TcpListener::bind(&cli.listen).await?;
    let server = execd::serve(listener, opts)?;
    tracing::info!(
        listen = %server.local_addr(),
        bearer = secured,
        run_as = switched,
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::listens_on_loopback;

    #[tokio::test]
    async fn only_loopback_addresses_count_as_loopback() {
        for yes in [
            "127.0.0.1:80",
            "127.9.9.9:80",
            "[::1]:80",
            "[::ffff:127.0.0.1]:80",
        ] {
            assert!(listens_on_loopback(yes).await.unwrap(), "{yes}");
        }
        for no in [
            "0.0.0.0:80",
            "[::]:80",
            "10.0.0.5:80",
            "192.168.1.1:80",
            "[::ffff:10.0.0.1]:80",
        ] {
            assert!(!listens_on_loopback(no).await.unwrap(), "{no}");
        }
        assert!(listens_on_loopback("not an address").await.is_err());
    }
}
