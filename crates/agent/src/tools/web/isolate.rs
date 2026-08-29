//! Run the HTML parsers in a locked-down child process.
//!
//! ## Why this exists
//!
//! The `web` tool fetches URLs the *model* chose and parses whatever comes back. Every other tool
//! processes data the operator or the model authored; this one processes bytes an arbitrary web
//! server sent. Those bytes then reach `scraper`/`html5ever` and `htmd`, and that stack carries real
//! `unsafe`: ~190 blocks across the tree, 92% of it in `tendril` (128, the refcounted string buffers)
//! and `ego-tree` (47, the arena). This crate forbids `unsafe_code`, but a dependency's memory-safety
//! bug is still remote code execution *in the agent* — a process holding `AI_AGENT_KEY`, the
//! workload's sealed environment, and a `bash` tool.
//!
//! So the parse does not run here. It runs in a child that holds none of that and can barely
//! syscall, and the only thing crossing back is a `String`.
//!
//! ## Two layers
//!
//! **Nothing to steal.** The child is spawned with a cleared environment (so the gateway credential
//! and the workload's secrets are simply absent), rooted at `/`, and wired to three pipes and
//! nothing else.
//!
//! **Nothing to do.** Before it touches a single attacker-controlled byte it installs a seccomp
//! filter that permits only the syscalls a parse actually needs — memory, the pipes, exit. `openat`,
//! `socket`, `connect`, `execve`, `clone`, `ptrace` are all gone. Code execution in `tendril` gets an
//! attacker a process that cannot open a file, dial a socket, or start a program, and that exits in
//! milliseconds.
//!
//! Layer one alone would not be worth much: an attacker who cannot read the key can still read the
//! filesystem and reach the network. It is the pair that makes this a boundary rather than a speed
//! bump.
//!
//! ## Cost
//!
//! One `fork`/`exec` and a pipe round-trip per parse — single-digit milliseconds against a network
//! fetch that already cost hundreds. The `fetch` mode does not parse and never pays it.

use std::io::{Read as _, Write as _};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

use agent_core::ToolError;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::Mode;

/// The hidden subcommand the parent re-execs itself with. Not a public interface: it exists so the
/// parse has its own address space, and it is `hide`den from help and completions accordingly.
pub const SUBCOMMAND: &str = "__web-parse";

/// How long a parse may take before the child is killed. A parser that does not terminate is
/// indistinguishable from one that is very slow, and this path is already behind a network fetch, so
/// the bound is generous — it exists to stop a hostile page pinning a session thread forever, not to
/// police normal work.
const PARSE_TIMEOUT: Duration = Duration::from_secs(30);

/// What the parent sends. `html` is the untrusted part; everything else the agent authored.
///
/// Two shapes for one message: the parent borrows (a page is megabytes, and there is no reason to
/// copy it just to serialize it), while the child owns what it decodes. `&Value` cannot implement
/// `Deserialize` at all, so this is not merely an optimization — it is the only way to keep the send
/// side zero-copy.
#[derive(Serialize)]
struct RequestOut<'a> {
    mode: &'a str,
    input: &'a Value,
    html: &'a str,
}

#[derive(Deserialize)]
struct RequestIn {
    mode: String,
    input: Value,
    html: String,
}

/// What the child sends back — the parser's own `Result`, flattened so a parse *error* (a bad
/// selector, say) stays distinguishable from the child dying.
#[derive(Serialize, Deserialize)]
enum Response {
    Ok(String),
    Err(String),
}

/// Parse `html` in an isolated child and return the rendered output.
///
/// Errors from the parser itself come back as [`ToolError::InvalidInput`]/[`ToolError::Execution`]
/// exactly as they would in-process. A child that crashes, is killed by its seccomp filter, or times
/// out is reported as an execution error naming the cause — never silently retried in-process, which
/// would defeat the point of the isolation.
pub(super) fn parse(mode: Mode, input: &Value, html: &str) -> Result<String, ToolError> {
    let exe = parser_binary()?;

    let mut cmd = Command::new(exe);
    cmd.arg(SUBCOMMAND)
        // Layer one. `env_clear` is the point: `AI_AGENT_KEY`, the sealed workload environment, and
        // anything else this process holds are simply not present in the child.
        .env_clear()
        // …except the allocator's own tuning, which is not a secret and which the deployment set
        // deliberately (a guest passes `MIMALLOC_DISALLOW_ARENA_ALLOC=1` to stop mimalloc taking
        // transparent huge pages). Dropping it would hand every parse a 1 GiB arena.
        .envs(std::env::vars().filter(|(k, _)| k.starts_with("MIMALLOC_")))
        // The child parses on the thread that dispatched the subcommand; a runtime sized to the host's
        // core count would spawn workers that do nothing but sit inside the sandbox.
        .env("TOKIO_WORKER_THREADS", "1")
        // Not the agent's cwd: a relative path in a payload should not resolve anywhere interesting.
        // `/` always exists, unlike a temp dir we would then have to create and clean up.
        .current_dir("/")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd
        .spawn()
        .map_err(|e| ToolError::Execution(format!("web: cannot spawn parser process: {e}")))?;

    let request = serde_json::to_vec(&RequestOut {
        mode: mode.as_str(),
        input,
        html,
    })
    .map_err(|e| ToolError::Execution(format!("web: cannot encode parse request: {e}")))?;

    let (Some(mut stdin), Some(mut stdout), Some(mut stderr)) =
        (child.stdin.take(), child.stdout.take(), child.stderr.take())
    else {
        let _ = child.kill();
        return Err(ToolError::Execution(
            "web: parser process pipes unavailable".into(),
        ));
    };

    // A page can be megabytes, which is far more than a pipe buffer holds, so the write and the read
    // must not be serialized: writing the whole request before reading any output deadlocks as soon
    // as the child's reply fills its own pipe. Writer on its own thread, reader here.
    let writer = std::thread::spawn(move || {
        let r = stdin.write_all(&request);
        // Dropping `stdin` closes it, which is what tells the child the request is complete.
        drop(stdin);
        r
    });

    let (tx, rx) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        let mut out = Vec::new();
        let r = stdout.read_to_end(&mut out);
        let _ = tx.send(());
        r.map(|_| out)
    });

    if rx.recv_timeout(PARSE_TIMEOUT).is_err() {
        let _ = child.kill();
        let _ = child.wait();
        return Err(ToolError::Execution(format!(
            "web: parsing timed out after {}s",
            PARSE_TIMEOUT.as_secs()
        )));
    }

    let out = match reader.join() {
        Ok(Ok(out)) => out,
        Ok(Err(e)) => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(ToolError::Execution(format!(
                "web: cannot read parser output: {e}"
            )));
        }
        Err(_) => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(ToolError::Execution("web: parser reader panicked".into()));
        }
    };
    // The writer's error is only interesting when the child produced nothing useful; a child that
    // answered before consuming the whole request (an early parse error) legitimately breaks the pipe.
    let write_failed = !matches!(writer.join(), Ok(Ok(())));

    let status = child
        .wait()
        .map_err(|e| ToolError::Execution(format!("web: cannot reap parser process: {e}")))?;

    // Checked before the output is even looked at. A child can write a complete, correct answer and
    // *then* trip the filter on the way out (that is precisely what a missing `sigaltstack` did), and
    // treating that as success would mean shipping a misconfigured sandbox that nothing ever reports.
    // The filter firing is a bug in the allowlist or an attack; either way it is not a parse result.
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt as _;
        if status.signal() == Some(libc::SIGSYS) {
            return Err(ToolError::Execution(describe_death(
                status,
                "",
                write_failed,
            )));
        }
    }

    if out.is_empty() {
        let mut err = String::new();
        let _ = stderr.read_to_string(&mut err);
        return Err(ToolError::Execution(describe_death(
            status,
            &err,
            write_failed,
        )));
    }

    match serde_json::from_slice::<Response>(&out) {
        Ok(Response::Ok(text)) => Ok(text),
        // The parser's own error text, preserved verbatim so the model sees the same wording it would
        // have seen from an in-process parse.
        Ok(Response::Err(msg)) => Err(ToolError::InvalidInput(msg)),
        Err(e) => Err(ToolError::Execution(format!(
            "web: malformed parser response: {e}"
        ))),
    }
}

/// Which binary hosts the parse.
///
/// Normally this process re-execs itself, which is what makes the child a *known* program rather
/// than whatever happens to be on `PATH`. But `current_exe` is only the agent when the agent is what
/// is running: under `cargo test` it is the test harness, and for anything embedding this crate as a
/// library it is that consumer's binary — neither of which answers to `__web-parse`. Both need a way
/// to name the real one, hence the override.
///
/// It is read from the environment rather than plumbed through `Web::new` because the parse is an
/// implementation detail of one mode of one tool; threading a path through the tool's constructor,
/// its registry entry, and every call site would put a testing concern in the production API.
const PARSER_BINARY_ENV: &str = "BEYOND_AI_AGENT_WEB_PARSER";

fn parser_binary() -> Result<std::path::PathBuf, ToolError> {
    if let Some(path) = std::env::var_os(PARSER_BINARY_ENV) {
        return Ok(path.into());
    }
    std::env::current_exe()
        .map_err(|e| ToolError::Execution(format!("web: cannot locate agent binary: {e}")))
}

/// Turn a dead child into a message that says *why*, because the three causes want different fixes.
fn describe_death(status: std::process::ExitStatus, stderr: &str, write_failed: bool) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt as _;
        // SIGSYS is the seccomp filter firing. That is either an exploit attempt or — far more
        // likely, and the reason this is called out by name — a syscall the allowlist should have
        // included. Say so, so it is diagnosable rather than a mystery crash.
        if status.signal() == Some(libc::SIGSYS) {
            return "web: parser process was killed by its seccomp filter (SIGSYS) — a syscall \
                    outside the parse allowlist was attempted"
                .into();
        }
        if let Some(sig) = status.signal() {
            return format!("web: parser process died on signal {sig}");
        }
    }
    let trailer = if stderr.trim().is_empty() {
        String::new()
    } else {
        format!(": {}", stderr.trim())
    };
    if write_failed {
        return format!("web: parser process exited before reading the page{trailer}");
    }
    format!("web: parser process produced no output{trailer}")
}

/// The child half. Reads one [`Request`], locks itself down, parses, writes one [`Response`].
///
/// Returns the process exit code rather than exiting, so `main` owns the exit as it does for every
/// other subcommand.
pub fn child_main() -> i32 {
    let mut buf = Vec::new();
    if let Err(e) = std::io::stdin().read_to_end(&mut buf) {
        eprintln!("web-parse: cannot read request: {e}");
        return 2;
    }

    // Lock down *before* the untrusted bytes are decoded, not merely before they are parsed. From
    // here on this process can allocate, use its pipes, and exit — nothing else.
    if let Err(e) = sandbox::lock_down() {
        eprintln!("web-parse: cannot install seccomp filter: {e}");
        return 2;
    }

    let response = match serde_json::from_slice::<RequestIn>(&buf) {
        Ok(req) => match Mode::parse(&req.mode) {
            Ok(Mode::Markdown) => match super::markdown::to_markdown(&req.html) {
                Ok(text) => Response::Ok(text),
                Err(e) => Response::Err(e),
            },
            Ok(mode @ (Mode::Outline | Mode::Locate | Mode::Extract | Mode::Table)) => {
                match super::extract::run(mode, &req.input, &req.html) {
                    Ok(text) => Response::Ok(text),
                    Err(e) => Response::Err(e.to_string()),
                }
            }
            // `fetch` never parses, so it never reaches the child; anything else is a parent bug.
            Ok(Mode::Fetch) => Response::Err("web: fetch mode does not parse".into()),
            Err(e) => Response::Err(e.to_string()),
        },
        Err(e) => Response::Err(format!("web: malformed parse request: {e}")),
    };

    match serde_json::to_vec(&response) {
        Ok(bytes) => {
            let mut out = std::io::stdout();
            if out.write_all(&bytes).and_then(|()| out.flush()).is_err() {
                return 2;
            }
            0
        }
        Err(e) => {
            eprintln!("web-parse: cannot encode response: {e}");
            2
        }
    }
}

mod sandbox {
    //! The seccomp allowlist for a parse.
    //!
    //! Deliberately an *allowlist* with `KillProcess` on a miss, not a denylist: a denylist has to
    //! anticipate what an exploit wants, and this has to anticipate only what our own parser does.
    //! The list is drawn from what remains after the request has already been read — allocate,
    //! decode, parse, write the answer, exit — plus the handful the Rust runtime and mimalloc reach
    //! for lazily (`getrandom` seeds `HashMap`; `futex` and `membarrier` back the allocator's
    //! synchronization; the signal calls are the panic/abort path).
    //!
    //! Being slightly generous here is safe. The syscalls that matter for an exploit — `openat`,
    //! `socket`, `connect`, `execve`, `clone`, `ptrace`, `prctl` — are absent, and no addition below
    //! reintroduces one.

    use std::collections::BTreeMap;

    use seccompiler::{SeccompAction, SeccompFilter, TargetArch};

    /// Syscalls a parse needs. Every entry exists on both release targets (x86_64 and aarch64), so
    /// this list is shared rather than `cfg`-split.
    fn allowed() -> Vec<i64> {
        vec![
            // The pipes.
            libc::SYS_read,
            libc::SYS_write,
            libc::SYS_readv,
            libc::SYS_writev,
            libc::SYS_close,
            libc::SYS_lseek,
            libc::SYS_fstat,
            // Memory. mimalloc leans on mmap/munmap/madvise; the rest is std's allocator paths.
            libc::SYS_mmap,
            libc::SYS_munmap,
            libc::SYS_mremap,
            libc::SYS_mprotect,
            libc::SYS_madvise,
            libc::SYS_brk,
            // Runtime bookkeeping.
            libc::SYS_futex,
            libc::SYS_getrandom,
            libc::SYS_sched_getaffinity,
            libc::SYS_sched_yield,
            libc::SYS_clock_gettime,
            libc::SYS_membarrier,
            libc::SYS_set_robust_list,
            libc::SYS_rseq,
            // Exit, including the abort path a panic takes.
            libc::SYS_exit,
            libc::SYS_exit_group,
            libc::SYS_getpid,
            libc::SYS_gettid,
            libc::SYS_tgkill,
            libc::SYS_rt_sigreturn,
            libc::SYS_rt_sigprocmask,
            libc::SYS_rt_sigaction,
            // Teardown, not the parse: the Rust runtime tears down its signal alternate stack on the
            // way out. Omitting it killed the child *after* it had written a correct answer — which
            // is exactly the sort of near-miss the `Ok`-with-SIGSYS check below refuses to paper over.
            libc::SYS_sigaltstack,
        ]
    }

    /// Install the filter across the whole process.
    ///
    /// `apply_filter` would cover only the calling thread, which is not enough: `main` is
    /// `#[tokio::main]`, so a runtime with worker threads is already up by the time the subcommand
    /// dispatches, and an unfiltered sibling thread is an unfiltered process. `apply_filter_all_threads`
    /// uses seccomp's `TSYNC` to cover every thread or fail — no partial application.
    pub fn lock_down() -> Result<(), String> {
        // Syscall numbers are per-architecture, so the filter has to be built for the one we are
        // actually running on. An unknown arch is a hard error rather than a silent pass-through:
        // shipping an unconfined parser because a target was unrecognized is exactly the failure
        // this module exists to prevent.
        let arch = match std::env::consts::ARCH {
            "x86_64" => TargetArch::x86_64,
            "aarch64" => TargetArch::aarch64,
            "riscv64" => TargetArch::riscv64,
            other => return Err(format!("no seccomp filter for architecture {other}")),
        };

        let rules: BTreeMap<i64, Vec<seccompiler::SeccompRule>> =
            allowed().into_iter().map(|nr| (nr, Vec::new())).collect();

        let filter = SeccompFilter::new(
            rules,
            // Anything not on the list kills the process. `Errno` would be quieter but wrong: a
            // parser that keeps running with `openat` failing is a parser whose behavior nobody has
            // reasoned about.
            SeccompAction::KillProcess,
            SeccompAction::Allow,
            arch,
        )
        .map_err(|e| e.to_string())?;

        let program: seccompiler::BpfProgram = filter
            .try_into()
            .map_err(|e: seccompiler::BackendError| e.to_string())?;
        seccompiler::apply_filter_all_threads(&program).map_err(|e| e.to_string())
    }
}
