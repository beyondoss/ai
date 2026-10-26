//! The `execd` binary: `EXECD_LISTEN=0.0.0.0:8080 execd`. It takes no arguments.

use std::os::fd::IntoRawFd as _;
use std::os::unix::fs::PermissionsExt as _;
use std::os::unix::process::CommandExt as _;
use std::process::ExitCode;

mod config;

use config::Config;

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

/// `EXECD_WORKER_THREADS`: `1` (the default) is the single-threaded scheduler, `0` is one worker
/// per CPU, `N` is N workers. execd only moves bytes between sockets and pipes, so one thread
/// carries more than any sandbox asks of it, and a thread per CPU costs a stack and a heap each
/// (see ARCHITECTURE.md "Memory").
fn runtime(worker_threads: usize) -> std::io::Result<tokio::runtime::Runtime> {
    let mut b = match worker_threads {
        1 => tokio::runtime::Builder::new_current_thread(),
        0 => tokio::runtime::Builder::new_multi_thread(),
        n => {
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

/// What `EXECD_BEARER_FILE` / `EXECD_BEARER` hold, with trailing newlines trimmed. `Err` for a file
/// that is named but unreadable or empty: a bearer that was asked for must not silently vanish.
fn bearer(cli: &Config, inherited: Option<String>) -> Result<Option<String>, String> {
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
            execd::log_warn!(
                "the bearer file {} is readable by others (mode {:o}): chmod 600 it",
                path.display(),
                meta.permissions().mode() & 0o777
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
    // Configuration is the environment alone: any argument is a mistake, and the usage says so.
    if std::env::args_os().nth(1).is_some() {
        eprint!("{}", config::USAGE);
        return ExitCode::from(2);
    }
    let cli = match config::parse(|k| std::env::var(k).ok()) {
        Ok(cli) => cli,
        Err(e) => {
            eprintln!("execd: {e}");
            return ExitCode::from(2);
        }
    };
    move_bearer_out_of_environ();
    let inherited = take_inherited_bearer();
    execd::log::set_level(cli.log_level);
    execd::log::set_format(cli.log_format);
    let runtime = match runtime(cli.worker_threads) {
        Ok(r) => r,
        Err(e) => {
            execd::log_error!("execd failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    match runtime.block_on(run(cli, inherited)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            execd::log_error!("execd failed: {e}");
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

async fn run(cli: Config, inherited: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
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
             listen on 127.0.0.1, or set EXECD_INSECURE_NO_AUTH=1 to run open on purpose",
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
                execd::log_warn!("{warning}");
            }
            home = home.or(resolved.home);
            Some(resolved.run_as)
        }
    };
    // Orphans reparent to execd and are reaped here, so a PID 1 (a container, E2B) never fills the
    // sandbox with zombies, and a zombie never counts as a live member of a group.
    if let Err(e) = execd::become_subreaper() {
        execd::log_warn!("cannot become a subreaper; orphans go to the host's init: {e}");
    }
    let reaper_stop = tokio_util::sync::CancellationToken::new();
    tokio::spawn(execd::reap_orphans(reaper_stop.clone()));

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

    let listener = tokio::net::TcpListener::bind(&cli.listen).await?;
    let server = execd::serve(listener, opts)?;
    execd::log_info!(
        "execd {} listening listen={} bearer={secured} run_as={switched} notify={}",
        env!("CARGO_PKG_VERSION"),
        server.local_addr(),
        cli.kept_jobs_survive_idle
    );
    // Off the serving path: it reads the whole of the jobs directory's parent (39 ms for a /tmp of
    // 9,500 entries), and a client that finds the port open expects the server to be ready.
    let jobs_dir = cli.jobs_dir.clone();
    tokio::task::spawn_blocking(move || execd::remove_stale_jobs_dirs(jobs_dir.as_deref()));
    if !secured {
        execd::log_warn!(
            "NO BEARER CONFIGURED: anyone who can reach {} can run commands in this sandbox. \
             Set EXECD_BEARER_FILE (or EXECD_BEARER) to require one.",
            server.local_addr()
        );
    }

    wait_for_signal().await;
    let killed = server.shutdown().await;
    reaper_stop.cancel();
    execd::log_info!("execd stopped killed_groups={killed}");
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
        _ = term.recv() => execd::log_info!("SIGTERM"),
        _ = int.recv() => execd::log_info!("SIGINT"),
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
