//! The agent as the last parent of what it starts (Linux; a no-op elsewhere).
//!
//! A process the agent starts can outlive it two ways: it leaves the agent's tree (a `setsid`, a
//! daemon's double fork — a browser MCP server's Chromium does both), or the agent dies without
//! running any cleanup (SIGKILL, the OOM killer). Three pieces close them:
//!
//! 1. **A child subreaper.** [`run_main`] makes the agent a child subreaper
//!    (`PR_SET_CHILD_SUBREAPER`) before anything is spawned, so an orphaned descendant re-parents to
//!    the agent instead of to init: it stays in the agent's tree, where the exit sweeps (which match
//!    by environment tag, process group and parentage — `tools::bash_proc::ProcGroup`) find it.
//! 2. **The main thread reaps those orphans, and only those.** A subreaper must reap the zombies it
//!    inherits, but a `waitpid(-1)` would steal the exit status of a child some tokio or std
//!    `Child` handle is waiting on. So `main`'s work runs on a thread of its own (named `main`, with
//!    the main thread's stack size) and the main thread does nothing but
//!    `waitpid(-1, __WNOTHREAD | __WALL)`: `__WNOTHREAD` limits it to children *of the main thread*,
//!    and the kernel hands every orphan to the subreaper's first live thread — the main thread —
//!    while every child the agent spawns is a child of the thread that spawned it. The one way an
//!    own child reaches the main thread is its spawning thread exiting first, so a thread that
//!    spawns a child must outlive it or wait for it (every spawn site does: runtime threads live as
//!    long as the process, and blocking-pool closures wait for what they start).
//! 3. **A parent-death signal on every stdio MCP server.** [`server_command`] starts the server
//!    through [`TRAMPOLINE`] — this binary again, which sets `PR_SET_PDEATHSIG(SIGKILL)` and execs
//!    the server in place (same pid). The workspace forbids `unsafe`, so a `pre_exec` hook is not an
//!    option; the trampoline is the safe equivalent, as `tools::web::isolate` is for seccomp. The
//!    signal fires when the *thread* that forked the child exits, not the process, so
//!    [`spawn_lasting`] forks every server from one thread that lives as long as the process —
//!    never from a tokio blocking thread, which exits after 10 s idle.
//!
//! What still escapes: a SIGKILLed agent takes its direct servers down, but a grandchild that left
//! the server's tree and ignores its parent dying is out of any in-process reach — that needs a
//! boundary outside the agent (a cgroup, the sandbox VM).

use std::sync::atomic::{AtomicBool, Ordering};

/// The trampoline's subcommand, dispatched by `main` before anything else exists:
/// `<agent> __exec-with-parent-death-signal <agent pid> <program> <argv0> [args…]`.
pub const TRAMPOLINE: &str = "__exec-with-parent-death-signal";

/// Set by [`run_main`] once the agent is a subreaper with the main thread reaping: the only state in
/// which [`TRAMPOLINE`] is answered (a test harness or a library consumer is not this binary).
static INSTALLED: AtomicBool = AtomicBool::new(false);

/// Whether this process is the agent binary running under [`run_main`].
pub fn installed() -> bool {
    INSTALLED.load(Ordering::Acquire)
}

/// Run `body` — the rest of `main`, returning its exit status — with the agent as a child subreaper
/// and the main thread reaping orphans; see the module doc. On Linux this does not return: `body`'s
/// thread exits the process with `body`'s status (101 after a panic, Rust's own). Elsewhere, or if
/// the setup fails, `body` runs here as before.
pub fn run_main(body: fn() -> u8) -> std::process::ExitCode {
    #[cfg(target_os = "linux")]
    {
        linux::run_main(body)
    }
    #[cfg(not(target_os = "linux"))]
    {
        std::process::ExitCode::from(body())
    }
}

/// The command that starts the stdio MCP server `program`: through [`TRAMPOLINE`] when
/// [`installed`] (resolving `program` on `path` — the server's configured `PATH`, else this
/// process's — first, so a missing program still fails here, as a spawn does), else `program` itself.
pub(crate) fn server_command(
    program: &str,
    path: Option<&std::ffi::OsStr>,
) -> std::io::Result<tokio::process::Command> {
    #[cfg(target_os = "linux")]
    if installed() {
        let resolved = linux::resolve(program, path)?;
        let mut cmd = tokio::process::Command::new("/proc/self/exe");
        cmd.arg(TRAMPOLINE)
            .arg(std::process::id().to_string())
            .arg(resolved)
            .arg(program);
        return Ok(cmd);
    }
    let _ = path;
    Ok(tokio::process::Command::new(program))
}

/// Spawn `cmd` — from the spawner thread when [`installed`], so a parent-death signal set by the
/// trampoline is tied to a thread that lives as long as the process; inline otherwise.
pub(crate) fn spawn_lasting(
    mut cmd: tokio::process::Command,
) -> std::io::Result<tokio::process::Child> {
    #[cfg(target_os = "linux")]
    if installed() {
        return linux::spawn_lasting(cmd);
    }
    cmd.spawn()
}

/// The trampoline itself: arm the parent-death signal, make sure the agent did not die before it was
/// armed, and exec the server in place. Returns an exit code only on failure.
pub fn trampoline_main() -> i32 {
    #[cfg(target_os = "linux")]
    {
        linux::trampoline_main()
    }
    #[cfg(not(target_os = "linux"))]
    {
        127
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use super::{INSTALLED, Ordering, TRAMPOLINE};
    use std::ffi::{OsStr, OsString};
    use std::path::PathBuf;
    use std::sync::mpsc;

    /// The main thread's own default (`ulimit -s`), so moving `main`'s work off it changes nothing
    /// a deep future could notice. Committed lazily, like any stack.
    const MAIN_STACK: usize = 8 << 20;

    /// How long the main thread waits before looking again when it has no orphan to wait on. An
    /// orphan's arrival wakes nothing, so this bounds how long a zombie can sit before it is reaped
    /// when it is the first orphan in a while; once one exists the wait blocks in `waitpid`.
    const IDLE_LOOK: std::time::Duration = std::time::Duration::from_secs(1);

    pub(super) fn run_main(body: fn() -> u8) -> std::process::ExitCode {
        use rustix::process::{getpid, set_child_subreaper};
        if set_child_subreaper(Some(getpid())).is_err() {
            return std::process::ExitCode::from(body());
        }
        INSTALLED.store(true, Ordering::Release);
        let spawned = std::thread::Builder::new()
            .name("main".into())
            .stack_size(MAIN_STACK)
            .spawn(move || {
                let code = match std::panic::catch_unwind(body) {
                    Ok(code) => i32::from(code),
                    // The panic hook has reported it; Rust's own panic status.
                    Err(_) => 101,
                };
                std::process::exit(code)
            });
        if spawned.is_err() {
            INSTALLED.store(false, Ordering::Release);
            let _ = set_child_subreaper(None);
            return std::process::ExitCode::from(body());
        }
        reap_orphans()
    }

    /// The main thread from here on: reap every child of *this thread* — the orphans re-parented to
    /// the agent — as it exits. See the module doc for why this never takes another thread's child.
    fn reap_orphans() -> ! {
        use rustix::process::{WaitOptions, wait};
        // `__WNOTHREAD`: this thread's children only. `__WALL`: whatever signal a child was cloned
        // to report its exit with.
        let options = WaitOptions::from_bits_retain((libc::__WNOTHREAD | libc::__WALL) as u32);
        loop {
            // `wait`, not `waitpid(None, …)`: rustix maps `None` to pid 0 — this process group only,
            // which a `setsid` orphan has left.
            match wait(options) {
                Ok(_) | Err(rustix::io::Errno::INTR) => {}
                // `ECHILD`: nothing to wait on yet.
                Err(_) => std::thread::park_timeout(IDLE_LOOK),
            }
        }
    }

    /// `program` as `execvp` would find it: as given when it names a path, else the first executable
    /// file of that name on `path` (or this process's `PATH`).
    pub(super) fn resolve(program: &str, path: Option<&OsStr>) -> std::io::Result<PathBuf> {
        let not_found = || std::io::Error::from(rustix::io::Errno::NOENT);
        if program.contains('/') {
            return if std::path::Path::new(program).exists() {
                Ok(PathBuf::from(program))
            } else {
                Err(not_found())
            };
        }
        let path: OsString = path
            .map(OsString::from)
            .or_else(|| std::env::var_os("PATH"))
            .unwrap_or_else(|| "/bin:/usr/bin".into());
        std::env::split_paths(&path)
            .map(|dir| {
                if dir.as_os_str().is_empty() {
                    PathBuf::from(".").join(program)
                } else {
                    dir.join(program)
                }
            })
            .find(|candidate| {
                use std::os::unix::fs::PermissionsExt as _;
                std::fs::metadata(candidate)
                    .is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
            })
            .ok_or_else(not_found)
    }

    type Job = Box<dyn FnOnce() + Send>;

    /// The thread every stdio MCP server is forked from. Started on first use, never ends: it waits
    /// on a channel whose sender is static.
    fn spawner() -> Option<&'static mpsc::Sender<Job>> {
        static SPAWNER: std::sync::OnceLock<Option<mpsc::Sender<Job>>> = std::sync::OnceLock::new();
        SPAWNER
            .get_or_init(|| {
                let (tx, rx) = mpsc::channel::<Job>();
                std::thread::Builder::new()
                    .name("mcp-spawner".into())
                    .stack_size(256 << 10)
                    .spawn(move || {
                        for job in rx {
                            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
                        }
                    })
                    .ok()
                    .map(|_| tx)
            })
            .as_ref()
    }

    pub(super) fn spawn_lasting(
        mut cmd: tokio::process::Command,
    ) -> std::io::Result<tokio::process::Child> {
        // No fallback to spawning here: a server whose parent-death signal is tied to a thread that
        // may exit would be killed at a random moment.
        let spawner = spawner()
            .ok_or_else(|| std::io::Error::other("could not start the MCP spawner thread"))?;
        // A tokio child registers with the runtime it is spawned in, so the spawner enters ours.
        let runtime = tokio::runtime::Handle::try_current().ok();
        let (tx, rx) = mpsc::sync_channel(1);
        let job: Job = Box::new(move || {
            let _entered = runtime.as_ref().map(tokio::runtime::Handle::enter);
            let _ = tx.send(cmd.spawn());
        });
        spawner
            .send(job)
            .map_err(|_| std::io::Error::other("the MCP spawner thread is gone"))?;
        rx.recv()
            .map_err(|_| std::io::Error::other("the MCP spawner thread dropped a spawn"))?
    }

    pub(super) fn trampoline_main() -> i32 {
        use std::os::unix::process::CommandExt as _;
        let mut args = std::env::args_os().skip(2);
        let parent = args
            .next()
            .and_then(|a| a.to_str().and_then(|s| s.parse::<i32>().ok()));
        let (Some(parent), Some(program), Some(arg0)) = (parent, args.next(), args.next()) else {
            eprintln!("{TRAMPOLINE}: expected <parent pid> <program> <argv0> [args…]");
            return 127;
        };
        // Sent when the thread that forked this process exits — the agent's spawner thread, which
        // lives as long as the agent. It survives the exec below.
        let _ =
            rustix::process::set_parent_process_death_signal(Some(rustix::process::Signal::KILL));
        // The agent may have died before that was armed: then no signal is coming (and this process
        // has already been re-parented), so go now rather than run unowned.
        if rustix::process::getppid().map(|p| p.as_raw_nonzero().get()) != Some(parent) {
            return 1;
        }
        let err = std::process::Command::new(&program)
            .arg0(&arg0)
            .args(args)
            .exec();
        eprintln!("failed to start `{}`: {err}", arg0.to_string_lossy());
        127
    }

    #[cfg(test)]
    mod tests {
        use super::resolve;

        #[test]
        fn resolve_finds_a_program_on_the_given_path_like_execvp() {
            let dir = tempfile::tempdir().unwrap();
            let not_exec = dir.path().join("tool");
            std::fs::write(&not_exec, "").unwrap();
            let bin = tempfile::tempdir().unwrap();
            let exec = bin.path().join("tool");
            std::fs::write(&exec, "#!/bin/sh\n").unwrap();
            {
                use std::os::unix::fs::PermissionsExt as _;
                std::fs::set_permissions(&exec, std::fs::Permissions::from_mode(0o755)).unwrap();
            }
            let path = std::env::join_paths([dir.path(), bin.path()]).unwrap();
            // The non-executable file of the same name earlier on the path is skipped.
            assert_eq!(resolve("tool", Some(&path)).unwrap(), exec);
            let missing = resolve("no-such-tool", Some(&path)).unwrap_err();
            assert_eq!(missing.kind(), std::io::ErrorKind::NotFound);
            assert_eq!(
                missing.to_string(),
                std::io::Error::from_raw_os_error(2).to_string(),
                "the same error a spawn reports"
            );
            assert!(resolve("/no/such/server", None).is_err());
            assert_eq!(
                resolve("/bin/sh", None).unwrap(),
                std::path::Path::new("/bin/sh")
            );
        }
    }
}
