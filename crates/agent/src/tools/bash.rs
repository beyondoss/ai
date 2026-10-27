//! `bash` — run a shell command via a resolved `bash` (falling back to `sh`) and stream/return its
//! combined output.
//!
//! Output handling mirrors pi: raw stdout+stderr feed one
//! [`OutputAccumulator`](super::output::OutputAccumulator) (tail-truncated for
//! display, full stream spilled to a temp file → `Full output: <path>`, except when the command ran
//! on a machine this process cannot write a readable file to — see [`describe`]), and while the
//! command runs the tool emits an initial empty update then **throttled snapshots** (the whole
//! output so far, every [`UPDATE_THROTTLE`]) via its [`ToolProgress`] sink. A non-zero exit becomes an
//! error carrying the output — same as pi's throw.
//!
//! Nothing here kills a command on a timer. `timeout_ms` only decides when the call returns; a
//! command still running then keeps running as a job (see [`super::bash_jobs`]), and every job ends
//! with the turn, the run or the session, or a `bash_job` kill. See "`bash` timeouts" in
//! `ARCHITECTURE.md`.

use std::borrow::Cow;
use std::path::Path;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use agent_core::tool::Tool;
use agent_core::{ToolError, ToolOutput, ToolProgress};
use async_trait::async_trait;
use regex::Regex;
use serde_json::{Value, json};

use super::bash_jobs::{
    BashJobs, COMMAND_CHARS, Job, JobEnd, JobKind, JobShared, MAX_INTENDED_JOBS, MAX_YIELDED_JOBS,
};
use super::bash_proc::truncate_command;
use super::exec::{CommandRunner, ExecResult, JobHooks, Kill, RealRunner};
use super::fs::local::LocalFs;
use super::fs::{FsBackend, PathWorld};
use super::output::{OutputSnapshot, TruncatedBy, format_output};

/// Ceiling on a model-supplied `timeout_ms` — pi's own `MAX_TIMEOUT_MS` (`i32::MAX` milliseconds,
/// ~24.8 days). `timeout_ms` only ever makes a call return *sooner* than the idle window and the
/// output ceiling would, so a large value is harmless; one past this is a typo or garbage, and saying
/// so beats silently accepting it.
const MAX_TIMEOUT_MS: u64 = 2_147_483_647;

/// The model's `timeout_ms`, the same field on `bash` and `bash_job`: how long the call waits before
/// returning with the output so far. `None` when absent ("no timeout of your own": the call returns
/// on exit, silence or its ceiling). A present one must be an integer from `min` to
/// [`MAX_TIMEOUT_MS`]; a negative value or a fraction is the caller mistake it looks like, and saying
/// so beats coercing it.
fn parse_timeout_ms(input: &Value, min: u64) -> Result<Option<u64>, ToolError> {
    let v = match input.get("timeout_ms") {
        None | Some(Value::Null) => return Ok(None),
        Some(v) => v,
    };
    match v.as_u64() {
        Some(ms) if (min..=MAX_TIMEOUT_MS).contains(&ms) => Ok(Some(ms)),
        Some(ms) if ms > MAX_TIMEOUT_MS => Err(ToolError::InvalidInput(format!(
            "`timeout_ms` must be at most {MAX_TIMEOUT_MS}, got {ms}"
        ))),
        _ => Err(ToolError::InvalidInput(format!(
            "`timeout_ms` must be {}, got {v}",
            if min == 0 {
                "a non-negative integer"
            } else {
                "a positive integer"
            }
        ))),
    }
}

/// How long a `run_in_background` call waits before returning with the job number: long enough that
/// a command which dies at once — a bad flag, a port in use, a missing script — reports its exit in
/// this result instead of as a notice a turn later; it returns sooner when the command exits. Measured
/// on the dev box (three runs each): a bad flag or a missing command exits in 1–3 ms, Python and Node
/// servers on a port in use in 24–90 ms, `npm run`/`npm start` with a missing script in 76–112 ms,
/// `npx` of a missing package in 330–587 ms, and `docker run` of a missing image in 907–979 ms — the
/// slowest. 1.5 s covers all of them with half again in hand, and costs a server that starts fine
/// that much latency once, against a model turn of several seconds.
pub const BACKGROUND_SETTLE: Duration = Duration::from_millis(1500);

/// Minimum gap between streamed progress snapshots — pi's `BASH_UPDATE_THROTTLE_MS`. Keeps a chatty
/// command from flooding the event stream; the final snapshot is always emitted regardless.
const UPDATE_THROTTLE: Duration = Duration::from_millis(100);

/// The shell used when the commands run somewhere other than this host and nobody said which shell
/// that target has (see [`Bash::with_remote_shell`]).
///
/// `sh`, not `bash`: the whole point of not running [`resolve_shell`] remotely is that this host's
/// filesystem answers a question about the wrong machine, and guessing `bash` would be the same
/// mistake in a different spelling — a busybox sandbox (Alpine, which is a large share of real ones)
/// has no `bash` at all, so every command would die on the shell rather than on its own merits.
/// Losing `[[` and `pipefail` degrades a command; losing the shell fails all of them.
const DEFAULT_REMOTE_SHELL: &str = "sh";

/// Resolve the shell commands run through: `/bin/bash` if present, else `bash` on `$PATH`, else `sh` —
/// pi's `shell.ts` resolution order (minus the Windows/WSL branches, which don't apply on this
/// platform). Bash's associative arrays, `[[`, `pipefail`, and process substitution are common enough
/// in model-generated commands that silently falling back to a POSIX `sh` (which may be `dash`, and
/// rejects all of the above) is a real correctness gap, not just a style preference. Cached: the
/// filesystem/PATH answer can't change mid-process.
///
/// **Host-only.** Every probe here reads *this* machine's filesystem and `$PATH`, so it is reached
/// only in [`PathWorld::Local`]; see [`DEFAULT_REMOTE_SHELL`] for the other side.
fn resolve_shell() -> &'static str {
    static SHELL: OnceLock<String> = OnceLock::new();
    SHELL.get_or_init(|| {
        if Path::new("/bin/bash").exists() {
            return "/bin/bash".to_string();
        }
        if let Some(path) = std::env::var_os("PATH") {
            for dir in std::env::split_paths(&path) {
                let candidate = dir.join("bash");
                if candidate.exists() {
                    return candidate.to_string_lossy().into_owned();
                }
            }
        }
        "sh".to_string()
    })
}

/// How long a running command may print nothing before its call returns to the model with the output
/// so far. The command is not stopped: it keeps running as a numbered job that `bash_job` waits on or
/// kills. Silence is grounds to ask the model, never to kill: a clean release build of this repo goes
/// 204 s without output. See "`bash` and hung commands" in `ARCHITECTURE.md` for the eval behind 60 s.
pub const IDLE_YIELD: Duration = Duration::from_secs(60);

/// How long the call that starts a command may stream the output of one that never goes quiet (a
/// server logging requests, a watcher) before it returns with the output so far and leaves the
/// command running as a job. Without it such a command held its call until the 30-minute timeout.
/// 10 minutes is well above the 3 min 44 s a clean release build of this repo takes, so a build or a
/// test suite that prints as it goes still finishes inside the call that started it.
pub const OUTPUT_CEILING: Duration = Duration::from_secs(600);

/// The longest one `bash_job` wait lasts, quiet or not, before it reports the job still running: a
/// job that never goes quiet (a server logging requests) cannot hold a wait past it. 5 minutes is
/// above the 3 min 44 s a clean release build of this repo takes, so a build that prints as it goes
/// finishes inside one wait. Silence still ends a wait at [`IDLE_YIELD`], and the model's own
/// `timeout_ms` sooner.
pub const WAIT_CEILING: Duration = Duration::from_secs(300);

pub struct Bash {
    runner: Arc<dyn CommandRunner>,
    /// The silence after which a call returns and leaves its command running as a job: [`IDLE_YIELD`],
    /// or `None` to wait for every command to exit (see [`Self::with_idle_yield`]).
    idle_yield: Option<Duration>,
    /// The call that starts a command returns by [`OUTPUT_CEILING`] however much it prints.
    call_ceiling: Duration,
    /// A `bash_job` wait returns by [`WAIT_CEILING`].
    wait_ceiling: Duration,
    /// A `run_in_background` call returns by [`BACKGROUND_SETTLE`].
    settle: Duration,
    /// Commands still running after their call returned, and ended ones not yet reported. Shared
    /// with every `bash`/`bash_job` built over the same table (see [`Self::with_jobs`]); dropping
    /// the last holder kills every job.
    jobs: Arc<BashJobs>,
    /// Overrides `resolve_shell()`'s auto-detection when set — see [`with_shell_path`](Self::with_shell_path).
    shell_path: Option<String>,
    /// The shell to use when [`Self::backend`] reports a remote world — see
    /// [`with_remote_shell`](Self::with_remote_shell).
    remote_shell: Option<String>,
    /// Prepended to every command, on its own line, when set — see
    /// [`with_command_prefix`](Self::with_command_prefix).
    command_prefix: Option<String>,
    /// Rendered at construction (and again by [`Self::with_idle_yield`]) — see [`describe`] — so the
    /// windows it states are the ones this instance uses.
    description: String,
    /// The working directory a command runs in when the model omits `cwd`, and what a *relative* `cwd`
    /// resolves against. Empty = the process cwd (the pre-root behavior: `run_streaming` is handed
    /// `None` and the spawned child inherits ours).
    ///
    /// This is the whole reason `bash` can honor a subagent's `isolation: worktree` at all. Unlike the
    /// other filesystem tools, `bash`'s blast radius is opaque — it reports no `write_target`, so a
    /// shared `WriteLockRegistry` provably cannot serialize two children's shell commands (see
    /// [`Tool::conservative_exclusive`]). Pointing its cwd at a per-child worktree is what keeps those
    /// commands from colliding.
    root: std::path::PathBuf,
    /// Which filesystem this tool's `cwd`, `~` and spilled output belong to.
    ///
    /// Held as the **backend itself**, not as a captured [`PathWorld`], and asked per call: an
    /// `ExecCell` re-pointed by `set_exec_endpoint` takes effect on the next tool call with no
    /// registry rebuild, so a snapshot taken at construction would keep answering for the machine the
    /// session just left — exactly the silent, plausible-looking wrong answer [`PathWorld`] exists to
    /// prevent. It is the same `Arc` the filesystem tools hold, which is what makes "`bash` and `edit`
    /// are on one machine" structural here rather than a comment; see
    /// [`ToolConfig::command_runner`](super::ToolConfig::command_runner).
    backend: Arc<dyn FsBackend>,
}

/// Build the model-facing tool description: the output truncation budget (matching pi's own bash tool
/// description), when a call returns while its command keeps running, and `run_in_background` — the
/// model shouldn't have to learn any of them from a surprising result. The `run_in_background`
/// sentence is the wording measured in the background-shell probe (1,500 trials on four models).
///
/// The spill file is **conditional**, and the description says so rather than promising one. When the
/// command runs somewhere else — an `ExecCell` pointed at a sandbox — spilling is off (see
/// `OutputAccumulator::without_spill`), because the only filesystem this process could write to is one
/// the model cannot read and, on a shared replica, belongs to everybody.
///
/// Deliberately *not* resolved at construction from the world. This tool holds the **backend** and
/// asks it per call, so an `ExecCell` re-pointed by `set_exec_endpoint` takes effect on the very next
/// tool call with no registry rebuild — and `Tool::description` returns `&str`, so a flag captured in
/// `new()` would be a stale snapshot for exactly the session that attaches a sandbox mid-life. The
/// runtime marker (`output.rs`'s `full_output_segment`) already tells the model what to do in each
/// case; this only has to stop promising the case that may not happen.
fn describe(idle_yield: Option<Duration>) -> String {
    let mut text = format!(
        "Run a shell command via a resolved bash (falls back to sh) and return its combined \
         stdout/stderr. Supports an optional `cwd`. Output is truncated to the last {} lines or {}, \
         whichever is hit first; when it is truncated, the output says whether the complete output \
         was saved to a temp file you can read, or has to be re-run redirected to a file.",
        super::output::DEFAULT_MAX_LINES,
        super::output::format_size(super::output::DEFAULT_MAX_BYTES as u64),
    );
    if let Some(idle) = idle_yield {
        text.push_str(&format!(
            " A command still running after {} s without new output, after {} minutes, or after its \
             `timeout_ms` returns early with its output so far and keeps running as a numbered job; \
             so does a process a command leaves running (`cmd &`). Set `run_in_background: true` for \
             a command you don't need to wait on (a server, a watcher): the call returns at once with \
             a job number and you are notified when it exits; don't poll or sleep. You are told when \
             every job exits, without asking. Use `bash_job` with the number to see what a job has \
             printed, to wait for one whose result you need, or to stop one. (Where a remote sandbox \
             cannot stream output as it happens, a call waits for its command to exit.)",
            idle.as_secs(),
            OUTPUT_CEILING.as_secs() / 60,
        ));
    }
    text
}

impl Bash {
    /// A `bash` tool that runs commands for real.
    pub fn real() -> Self {
        Self::with_runner(Arc::new(RealRunner))
    }

    /// Builder-style: keep jobs in `jobs` rather than a table of this tool's own — the session's,
    /// so a registry rebuild (`set_model`) keeps them, and so the host can stop them or wake on them.
    pub fn with_jobs(mut self, jobs: Arc<BashJobs>) -> Self {
        self.jobs = jobs;
        self
    }

    /// Builder-style: different ceilings than [`OUTPUT_CEILING`] (the call that starts a command)
    /// and [`WAIT_CEILING`] (a `bash_job` wait). For tests.
    pub fn with_ceilings(mut self, call: Duration, wait: Duration) -> Self {
        self.call_ceiling = call;
        self.wait_ceiling = wait;
        self
    }

    /// Builder-style: a different settle than [`BACKGROUND_SETTLE`]. For tests.
    pub fn with_settle(mut self, settle: Duration) -> Self {
        self.settle = settle;
        self
    }

    /// The job table this tool parks commands in.
    pub fn jobs(&self) -> &Arc<BashJobs> {
        &self.jobs
    }

    /// Builder-style: a different idle window than [`IDLE_YIELD`], or `None` to wait for every command
    /// to exit — for a caller with no `bash_job` to come back with (`serve`'s host `bash` command).
    pub fn with_idle_yield(mut self, idle_yield: Option<Duration>) -> Self {
        self.idle_yield = idle_yield;
        self.description = describe(idle_yield);
        self
    }

    /// Builder-style: run commands in `root` when the model omits `cwd`, and resolve a relative `cwd`
    /// against it. An empty `root` leaves the pre-root behavior (inherit the process cwd) untouched.
    pub fn with_root(mut self, root: impl Into<std::path::PathBuf>) -> Self {
        self.root = root.into();
        self
    }

    /// The directory this call runs in: the model's `cwd` resolved against [`Self::root`], else `root`
    /// itself, else `None` (inherit the process cwd). Returned as an owned `String` because both the
    /// validation below and the runner need it, and it may be a freshly-joined path rather than a
    /// borrow of the input.
    ///
    /// `world` decides only what a leading `~` expands against: this process's `$HOME` locally, the
    /// target's remotely (and nothing at all when the target's home isn't known, rather than a guess).
    /// The join onto `root` is pure string work and identical either way.
    fn resolve_cwd(&self, cwd: Option<&str>, world: &PathWorld) -> Option<String> {
        match cwd {
            Some(dir) => Some(super::resolve_against_in(&self.root, dir, world)),
            None if !self.root.as_os_str().is_empty() => Some(self.root.display().to_string()),
            None => None,
        }
    }

    /// Builder-style: run commands through this shell instead of the auto-resolved one
    /// (`resolve_shell()`: `/bin/bash`, else `bash` on `$PATH`, else `sh`) — for a non-standard
    /// environment (Cygwin, a container without `/bin/bash` at the expected path, a hardened/audited
    /// shell wrapper) where auto-detection would pick the wrong binary. Matches pi's own `shellPath`
    /// setting (`getShellConfig(customShellPath)`). Existence is the caller's responsibility to check
    /// up front (see `--bash-shell-path` in `main.rs`) — failing fast at CLI-argument time, once, is
    /// simpler than threading a `Result` through every tool-registry rebuild this would otherwise
    /// touch (`set_model`/`set_thinking` each rebuild the registry from `ServeConfig`).
    pub fn with_shell_path(mut self, path: impl Into<String>) -> Self {
        self.shell_path = Some(path.into());
        self
    }

    /// Builder-style: prepend `prefix` (its own line, before the model's command) to every command run
    /// through this tool — matches pi's `BashToolOptions.commandPrefix` (e.g. sourcing a project's env
    /// setup, activating a venv, or setting a variable every command should see). Both the prefix's and
    /// the command's output land in the same combined stdout/stderr stream, in order — this is just
    /// script concatenation, not a separate execution.
    pub fn with_command_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.command_prefix = Some(prefix.into());
        self
    }

    /// A `bash` tool over a caller-supplied runner.
    ///
    /// This is how `bash` runs somewhere other than this host: hand it the same
    /// [`CommandRunner`] the filesystem tools are using, and the whole toolset acts on one machine.
    /// Tests also use it to capture invocations without spawning anything.
    pub fn with_runner(runner: Arc<dyn CommandRunner>) -> Self {
        Self {
            runner,
            idle_yield: Some(IDLE_YIELD),
            call_ceiling: OUTPUT_CEILING,
            wait_ceiling: WAIT_CEILING,
            settle: BACKGROUND_SETTLE,
            jobs: Arc::default(),
            shell_path: None,
            remote_shell: None,
            command_prefix: None,
            description: describe(Some(IDLE_YIELD)),
            root: std::path::PathBuf::new(),
            backend: Arc::new(LocalFs::new()),
        }
    }

    /// Builder-style: take this tool's [`PathWorld`] from `backend`, which must be the backend the
    /// filesystem tools were built with. See [`Bash::backend`].
    pub fn with_backend(mut self, backend: Arc<dyn FsBackend>) -> Self {
        self.backend = backend;
        self
    }

    /// Builder-style: the shell to run through when the target is remote — see
    /// [`ToolConfig::remote_shell`](super::ToolConfig::remote_shell).
    pub fn with_remote_shell(mut self, shell: impl Into<String>) -> Self {
        self.remote_shell = Some(shell.into());
        self
    }

    /// The shell `exec()` invokes: the override from [`with_shell_path`](Self::with_shell_path) if
    /// set, else — on this host — the auto-resolved default, and otherwise the target's shell
    /// ([`with_remote_shell`](Self::with_remote_shell), defaulting to [`DEFAULT_REMOTE_SHELL`]).
    ///
    /// The world matters because [`resolve_shell`] answers by probing *this* machine's filesystem and
    /// `$PATH`. Off-host that is not merely unhelpful, it is wrong in the most expensive direction: it
    /// hands the sandbox a `/bin/bash` path that exists here, and every single command fails to spawn.
    fn shell(&self, world: &PathWorld) -> &str {
        if let Some(explicit) = self.shell_path.as_deref() {
            return explicit;
        }
        if world.is_remote() {
            return self.remote_shell.as_deref().unwrap_or(DEFAULT_REMOTE_SHELL);
        }
        resolve_shell()
    }

    /// Shared body for [`Tool::run`] and [`Tool::run_streaming`]. Raw output feeds one accumulator; when
    /// `progress` is set, snapshots stream live. The returned value is the cleaned, tail-truncated
    /// output with pi's `Full output:` marker; a non-zero exit is an error carrying it.
    async fn exec(
        &self,
        input: Value,
        progress: Option<&ToolProgress>,
    ) -> Result<ToolOutput, ToolError> {
        let command = input
            .get("command")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidInput("missing `command`".into()))?;
        // Pi-parity note: `cwd` being a per-call, model-settable field here is a deliberate DIVERGENCE,
        // not a gap to close. Pi's own bash tool schema has no `cwd` at all — its working directory is
        // fixed once at construction (the shell config baked in at startup), not choosable per
        // invocation. Letting the model pick `cwd` per call is a real capability beyond's own, kept as
        // a plausible enhancement rather than narrowed to match.
        //
        // Resolved against this tool's `root` (see `Bash::root`): an omitted `cwd` becomes `root`, and a
        // relative one is joined onto it. Everything below — the validation, the error text, and what
        // the runner is handed — uses the *resolved* directory, since that is the one the child will
        // actually chdir into and the one whose non-existence would fail the spawn.
        //
        // Asked once per call and threaded through everything below that could otherwise consult the
        // wrong machine: tilde expansion, the `cwd` pre-check, the shell, and the spill file.
        let world = self.backend.world();
        let cwd = self.resolve_cwd(input.get("cwd").and_then(Value::as_str), &world);
        let cwd = cwd.as_deref();
        // Fail with a clear message instead of the raw spawn-error wrapping ("spawn failed: No such
        // file or directory") a bad `cwd` would otherwise surface as — matches pi's own pre-check, plus
        // a stricter, more accurate rejection: pi only checks existence, so it would happily hand a
        // *file* path to the shell as `cwd` and let the spawn itself fail with a confusing "Not a
        // directory" OS error. `.is_dir()` rejects that up front too — but that means "does not exist"
        // and "exists but isn't a directory" must be told apart explicitly (pi-parity task 51): a path
        // that exists (a file, a broken permission on an ancestor notwithstanding) must not be reported
        // as nonexistent.
        //
        // Host-only, and deliberately not re-expressed as a remote `stat`. Off-host, `is_dir`/`exists`
        // answer for the replica — where a sandbox path is typically *absent*, so the check would
        // reject every perfectly good `cwd` — and paying a round trip to ask the target the same
        // question buys nothing the shell doesn't already tell us: `run_streaming` with a bad `cwd`
        // comes back as a failed spawn naming the directory.
        if let Some(dir) = cwd
            && !world.is_remote()
        {
            let p = Path::new(dir);
            if !p.is_dir() {
                let msg = if p.exists() {
                    format!("Working directory is not a directory: {dir}")
                } else {
                    // pi-parity fix: pi's own bash.ts message is two lines — the plain fact, plus this
                    // explanatory second line stating the consequence — not just the first on its own.
                    format!(
                        "Working directory does not exist: {dir}\nCannot execute bash commands."
                    )
                };
                return Err(ToolError::InvalidInput(msg));
            }
        }
        let start = match input.get("run_in_background") {
            // A present `timeout_ms` must be a positive integer (pi's `resolveTimeoutMs` bounds): 0
            // would return before the command had a chance to print anything.
            None | Some(Value::Null) | Some(Value::Bool(false)) => {
                Start::Foreground(parse_timeout_ms(&input, 1)?)
            }
            // A background start returns by itself, so `timeout_ms` has nothing to bound: ignored,
            // not validated (models send `0` alongside it).
            Some(Value::Bool(true)) => Start::Background,
            Some(v) => {
                return Err(ToolError::InvalidInput(format!(
                    "`run_in_background` must be true or false, got {v}"
                )));
            }
        };
        // Off this host, a session's commands run as jobs of the sandbox when it can host them
        // (see `bash_sandbox`): that is what lets an intended job outlive this replica.
        let sandbox = if world.is_remote() {
            // A reopened session attaches its sandbox lazily. Connect it first — the probe, whose
            // failure is this call's — and read back the jobs it kept before anything new starts.
            self.runner
                .ready()
                .await
                .map_err(|e| ToolError::Execution(e.to_string()))?;
            if let Err(e) = super::bash_sandbox::reconcile(&self.jobs).await {
                tracing::warn!(error = %e, "could not read back the session's sandbox jobs");
            }
            super::bash_sandbox::store_for(&self.jobs, &self.runner).await
        } else {
            None
        };
        if start == Start::Background {
            // Refused before anything runs: starting a server only to stop it helps nobody.
            if sandbox.is_none() && !self.runner.tracks_jobs().await {
                return Err(ToolError::InvalidInput(
                    "`run_in_background` is unavailable here: this sandbox's exec endpoint (protocol \
                     v1.1) cannot stream a command's output or stop it, so nothing can keep running \
                     past the call. Run the command without it; the call then waits for the command \
                     to exit."
                        .into(),
                ));
            }
            if let Some(running) = self.jobs.intended_full() {
                return Err(ToolError::Execution(format!(
                    "Not started: {MAX_INTENDED_JOBS} {INTENDED_JOBS} are already running \
                     ({running}). Stop one with bash_job {{\"job\": N, \"kill\": true}}, then run \
                     this again."
                )));
            }
        }
        // Prefix and command run as one script in one shell invocation — matches pi's
        // `${commandPrefix}\n${command}` — so both land in the same combined output stream, in order,
        // rather than looking like two separate calls.
        let resolved_command = match &self.command_prefix {
            Some(prefix) => format!("{prefix}\n{command}"),
            None => command.to_string(),
        };
        let args = vec!["-c".to_string(), resolved_command];
        self.exec_detachable(args, cwd, &world, progress, command, start, sandbox)
            .await
    }

    /// The command runs on its own task, and this call waits on it until it exits, it has been quiet
    /// for the idle window ([`Self::idle_window`]), it has kept printing for the call's ceiling
    /// ([`OUTPUT_CEILING`]), the model's `timeout_ms` passes, or its shell exits leaving processes
    /// running — or, for `run_in_background`, until [`BACKGROUND_SETTLE`] passes. In every case but
    /// the first the command keeps running as a numbered job (see [`super::bash_jobs`]): the model
    /// gets the output so far plus how to wait on or stop it, and is told without asking when it ends.
    ///
    /// Raw stdout+stderr feed one accumulator in arrival order, so the temp-file spill sees the
    /// complete output. The spill file is written on *this* host: when the command ran somewhere
    /// else, `Full output: /tmp/pi-bash-….log` would name a path the model can never read (and, on a
    /// replica serving other tenants, put a tenant's output on the host), so none is created and the
    /// marker says to redirect into the sandbox instead (see [`super::bash_jobs::accumulator`]).
    ///
    /// No deadline kills the command: a job lives until it exits, is killed, or its kind's owner ends
    /// it (the turn for a yielded job; the run or the session for an intended one).
    #[allow(clippy::too_many_arguments)]
    async fn exec_detachable(
        &self,
        args: Vec<String>,
        cwd: Option<&str>,
        world: &PathWorld,
        progress: Option<&ToolProgress>,
        command: &str,
        start: Start,
        sandbox: Option<(
            Arc<dyn crate::exec_endpoint::jobs::SandboxJobs>,
            crate::exec_endpoint::jobs::JobScope,
        )>,
    ) -> Result<ToolOutput, ToolError> {
        let shared = Arc::new(JobShared::new(world.is_remote(), &self.jobs));
        shared.set_command(truncate_command(command, COMMAND_CHARS));
        // A sandbox job when the sandbox can host it: held when its start can stream, kept from the
        // start otherwise — which only a background job needs; a foreground command over an endpoint
        // that cannot stream waits for its exit as before.
        let sandbox = match sandbox {
            Some((store, scope)) => {
                let held = store.holds().await;
                (held || start == Start::Background).then(|| {
                    let link = super::bash_sandbox::SandboxLink::new(
                        store,
                        scope,
                        crate::exec_endpoint::jobs::JobId::fresh(),
                        !held,
                    );
                    shared.set_sandbox(link.clone());
                    (link, held)
                })
            }
            None => None,
        };
        if let Some(p) = progress {
            p.emit("", None);
        }
        let handle = {
            let runner = self.runner.clone();
            let shell = self.shell(world).to_string();
            let cwd = cwd.map(str::to_string);
            let shared = shared.clone();
            let label = truncate_command(command, COMMAND_CHARS);
            tokio::spawn(async move {
                // Records an end even if this task is aborted or panics, so a waiting call never
                // waits on a task that is gone.
                let mut finished = FinishOnDrop(Some(shared.clone()));
                let on_chunk = |bytes: &[u8]| shared.append(bytes);
                let on_spawn = |kill: Kill| shared.track(kill);
                let on_orphans = |shell: &ExecResult, left: String| shared.orphaned(shell, left);
                let hooks = JobHooks {
                    on_chunk: &on_chunk,
                    on_spawn: &on_spawn,
                    on_orphans: &on_orphans,
                };
                let result = match sandbox {
                    Some((link, held)) => {
                        let spec = crate::exec_endpoint::jobs::JobSpec {
                            program: shell,
                            args,
                            cwd,
                            label,
                        };
                        super::bash_sandbox::run(&shared, &link, &spec, held).await
                    }
                    None => runner
                        .run_streaming_tracked(&shell, &args, cwd.as_deref(), &hooks)
                        .await
                        .map_err(|e| format!("spawn failed: {e}")),
                };
                // Exited: nothing left for anyone to kill.
                shared.untrack();
                finished.0 = None;
                shared.finish(result);
            })
        };
        let (kind, hold) = match start {
            Start::Foreground(timeout_ms) => (
                JobKind::Yielded,
                Hold::Foreground {
                    ceiling: self.call_ceiling,
                    timeout_ms,
                },
            ),
            Start::Background => (JobKind::Background, Hold::Settle(self.settle)),
        };
        let job = Job {
            id: None,
            kind,
            shared: shared.clone(),
            handle,
            started: Instant::now(),
            reminded: false,
            command: truncate_command(command, COMMAND_CHARS),
        };
        let started = job.started;
        let result = self.attach(job, progress, hold).await;
        // A runner that streams nothing (an exec endpoint on protocol v1.1) cannot hand control
        // back before the command exits; a `timeout_ms` it outlived must not look honored.
        match timeout_unheard(&shared, &start, started) {
            Some(note) => match result {
                Ok(out) => Ok(append_status(&out.text, &note).into()),
                Err(ToolError::Execution(text)) => {
                    Err(ToolError::Execution(append_status(&text, &note)))
                }
                Err(e) => Err(e),
            },
            None => result,
        }
    }

    /// Wait on `job` until it ends (its final result), its shell exits leaving processes running
    /// (adopt them as a job, if the call was a foreground one), `hold` says to hand control back
    /// (park it and report the output since the last report), or the call is cancelled (kill a fresh
    /// command and return what it printed, tagged "Command cancelled"; park a numbered job again).
    async fn attach(
        &self,
        job: Job,
        progress: Option<&ToolProgress>,
        hold: Hold,
    ) -> Result<ToolOutput, ToolError> {
        enum Woke {
            Changed,
            Deadline,
            Cancelled,
        }
        let attached = Instant::now();
        // pi emits the first throttled snapshot one `UPDATE_THROTTLE` after the start, not on the
        // first chunk.
        let mut last_emit = attached;
        loop {
            if let Some(end) = job.shared.take_end() {
                return self.finish_job(job, end, progress);
            }
            if job.kind == JobKind::Yielded
                && let Some((shell, left)) = job.shared.orphans()
            {
                return self.adopt(job, &shell, &left, progress).await;
            }
            let deadline = self.yield_deadline(&job, attached, hold);
            let cancelled = async {
                match progress {
                    Some(p) => p.cancelled().await,
                    None => std::future::pending().await,
                }
            };
            let woke = tokio::select! {
                biased;
                () = cancelled => Woke::Cancelled,
                () = job.shared.notify.notified() => Woke::Changed,
                () = tokio::time::sleep_until(deadline.into()) => Woke::Deadline,
            };
            match woke {
                Woke::Changed => {
                    // New output gets its live update even when the command ended right behind it
                    // (the final snapshot follows from `finish_job`).
                    if let Some(p) = progress
                        && last_emit.elapsed() >= UPDATE_THROTTLE
                    {
                        let mut out = job.shared.out();
                        if out.last_output > last_emit {
                            last_emit = Instant::now();
                            // `true`: once the output is past the display cap, every live update
                            // carries a spill path to point at (pi forces the spill on each emit).
                            let snap = out.acc.snapshot(true);
                            drop(out);
                            emit_update(p, &snap);
                        }
                    }
                }
                Woke::Deadline => {
                    if let Some(reason) = self.yield_reason(&job, attached, hold) {
                        return self.detach(job, reason).await;
                    }
                }
                Woke::Cancelled => {
                    let snap = job.shared.take_segment();
                    if let Some(p) = progress {
                        emit_update(p, &snap);
                    }
                    let text = clean(format_output(&snap, ""));
                    // A job with a number is a `bash_job` wait being cancelled (an abort, or a
                    // command that ends the run): the model was told how long that job lives, and
                    // cancelling a wait on it must not end it. It goes back in the table; only a
                    // call's own fresh command dies with the call.
                    if let Some(id) = job.id {
                        // An evicted yielded job is dropped (killed) here, outside the table lock.
                        let _evicted = self.jobs.park(job);
                        return Err(ToolError::Execution(append_status(
                            &text,
                            &format!("Wait cancelled; job {id} is still running."),
                        )));
                    }
                    // Dropping the job kills its processes (see `Job`'s `Drop`).
                    drop(job);
                    return Err(ToolError::Execution(append_status(
                        &text,
                        "Command cancelled",
                    )));
                }
            }
        }
    }

    /// When this attach hands control back. A foreground attach (the call that starts a command, or
    /// a `bash_job` wait) yields after the idle window of silence (measured from the later of the
    /// last output and this attach, so a re-attached job always gets a full window), at its
    /// ceiling, or at the model's `timeout_ms` (0: at once), whichever is first — and never, when
    /// the command's silence says nothing (see [`Self::idle_window`]). A background start returns
    /// at its settle.
    fn yield_deadline(&self, job: &Job, attached: Instant, hold: Hold) -> Instant {
        let (ceiling, timeout_ms) = match hold {
            Hold::Settle(settle) => return attached + settle,
            Hold::Foreground {
                ceiling,
                timeout_ms,
            } => (ceiling, timeout_ms),
        };
        let Some(idle) = self.idle_window(job) else {
            // Never yield: a far deadline is the honest answer.
            return attached + Duration::from_secs(u64::from(u32::MAX));
        };
        let quiet_since = job.shared.out().last_output.max(attached);
        let deadline = (quiet_since + idle).min(attached + ceiling);
        match timeout_ms {
            Some(ms) => deadline.min(attached + Duration::from_millis(ms)),
            None => deadline,
        }
    }

    /// The silence any attach allows before reporting: [`Self::idle_yield`], the same for the call
    /// that started the command and for every wait after it. It used to grow ×4 per report (60 s,
    /// 240 s, 960 s), so a silent build was reported once rather than every minute; that predates
    /// finish notices. With the exit pushed to the model, an early return costs one model turn
    /// (median 2.5 s on Luna), while a long window costs its whole length on a job that never exits
    /// — in the background A/B the one grown window was a 240 s wait on a watcher. See "`bash` and
    /// hung commands" in `ARCHITECTURE.md`.
    ///
    /// `None` (never yield) when the command runs somewhere this host sees no output from as it
    /// happens: a v1.1 request/response exec endpoint streams nothing, so its silence says nothing
    /// about the command. A remote runner whose output is live (exec protocol v1.2, a template
    /// transport) says so by reporting a [`Kill`], and from then on yields exactly as local does.
    fn idle_window(&self, job: &Job) -> Option<Duration> {
        if job.shared.remote && !job.shared.live.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        self.idle_yield
    }

    /// Why this attach should return now, if it should. The model's own `timeout_ms` is named first
    /// when it is due: it is the reason the model can act on.
    fn yield_reason(&self, job: &Job, attached: Instant, hold: Hold) -> Option<Yield> {
        let waited = attached.elapsed();
        let (ceiling, timeout_ms) = match hold {
            Hold::Settle(settle) => return (waited >= settle).then_some(Yield::Settled),
            Hold::Foreground {
                ceiling,
                timeout_ms,
            } => (ceiling, timeout_ms),
        };
        let window = self.idle_window(job)?;
        if let Some(ms) = timeout_ms
            && waited >= Duration::from_millis(ms)
        {
            return Some(Yield::Timeout(ms));
        }
        let quiet = job.shared.out().last_output.max(attached).elapsed();
        if quiet >= window {
            return Some(Yield::Quiet(quiet));
        }
        (waited >= ceiling).then_some(Yield::Ceiling(waited))
    }

    /// Park `job` in the table and tell the model what it has so far, why the call returned, and
    /// how to come back to it. A job that already has a number is a `bash_job` wait returning: the
    /// model was told how to use it when it started, so the report is short.
    async fn detach(&self, mut job: Job, reason: Yield) -> Result<ToolOutput, ToolError> {
        let why = match reason {
            Yield::Settled => return self.start_background(job).await,
            Yield::Quiet(quiet) => format!("No output for {}. ", super::format_duration(quiet)),
            Yield::Ceiling(waited) => format!(
                "Still printing after {}, so this call returned with the output so far. ",
                super::format_duration(waited)
            ),
            // A peek: nothing to explain.
            Yield::Timeout(0) => String::new(),
            Yield::Timeout(ms) => format!("Returned after your timeout_ms of {ms} ms. "),
        };
        let reattached = job.id.is_some();
        let id = self.jobs.assign_id(&mut job);
        let snap = job.shared.take_segment();
        let text = clean(format_output(&snap, ""));
        let running = super::format_duration(job.started.elapsed());
        let lifetime = self.jobs.lifetime(job.kind, job.shared.sandbox().is_some());
        // What the model has seen of a job the sandbox owns: where a replica that reattaches it
        // picks up.
        super::bash_sandbox::spawn_seen(&job.shared);
        let status = if reattached {
            format!(
                "{why}Job {id} is still running ({running} so far; {lifetime}). You will be told \
                 when it exits, so do not poll it. bash_job {{\"job\": {id}, \"kill\": true}} stops \
                 it."
            )
        } else {
            format!(
                "{why}The command is still running as job {id} ({running} so far; {lifetime}). \
                 You will be told when it exits, so do not poll it or sleep: keep working, or, if \
                 you need its result, wait for it with bash_job {{\"job\": {id}}}. bash_job \
                 {{\"job\": {id}, \"kill\": true}} stops it (not `pkill`, which can match unrelated \
                 processes)."
            )
        };
        let status = match self.jobs.park(job) {
            // Dropped at the end of this arm, outside the table lock; dropping kills it.
            Some((old, _evicted)) => format!(
                "{status} Job {old} was stopped: at most {MAX_YIELDED_JOBS} jobs whose calls \
                 returned early are kept running, the oldest stopped first."
            ),
            None => status,
        };
        Ok(append_status(&text, &status).into())
    }

    /// `run_in_background`, settled and still running: park it as an intended job and return its
    /// first output with the job number. The cap was checked before the command started; a race
    /// past it (only another session sharing the table could) stops this one.
    async fn start_background(&self, job: Job) -> Result<ToolOutput, ToolError> {
        let snap = job.shared.take_segment();
        let text = clean(format_output(&snap, "(no output yet)"));
        let shared = job.shared.clone();
        let lifetime = self.jobs.lifetime(job.kind, shared.sandbox().is_some());
        match self.jobs.park_intended(job) {
            Ok(id) => Ok(append_status(
                &text,
                // Three uses, each spelled out: a server's or watcher's new output is a peek, not
                // a wait (it never exits; waiting on one blocked for the whole silence window in
                // the A/B), and only a command that finishes is worth waiting for.
                &format!(
                    "Running in the background as job {id} ({lifetime}). You will be told when it \
                     exits, so do not poll it or sleep. To see what it has printed since, bash_job \
                     {{\"job\": {id}, \"timeout_ms\": 0}}; to wait for it to finish (a build or a \
                     test run, not a server), bash_job {{\"job\": {id}}}; to stop it, bash_job \
                     {{\"job\": {id}, \"kill\": true}}.{}",
                    self.hand_over(&shared, id, JobKind::Background).await
                ),
            )
            .into()),
            Err((job, running)) => {
                if let Some(kill) = job.shared.untrack() {
                    let _ = tokio::task::spawn_blocking(move || kill.kill()).await;
                }
                drop(job);
                Err(ToolError::Execution(append_status(
                    &text,
                    &format!(
                        "It was stopped: {MAX_INTENDED_JOBS} {INTENDED_JOBS} are already running \
                         ({running}). Stop one with bash_job {{\"job\": N, \"kill\": true}}, then \
                         run this again."
                    ),
                )))
            }
        }
    }

    /// The command's shell exited but left processes running (`cmd &`): they are tracked as a job
    /// of their own — never evicted, since the command meant them — unless [`MAX_INTENDED_JOBS`] are
    /// already running, in which case they are stopped and the model is told which jobs to stop.
    async fn adopt(
        &self,
        mut job: Job,
        shell: &ExecResult,
        left: &str,
        progress: Option<&ToolProgress>,
    ) -> Result<ToolOutput, ToolError> {
        let snap = job.shared.take_segment();
        if let Some(p) = progress {
            emit_update(p, &snap);
        }
        let text = clean(format_output(&snap, ""));
        let shell_status = exit_status_line(shell, job.started.elapsed());
        job.kind = JobKind::Adopted;
        let shared = job.shared.clone();
        let lifetime = self.jobs.lifetime(job.kind, shared.sandbox().is_some());
        // An instance that never detaches has no `bash_job` beside it and nobody to tell when the
        // processes exit: `serve`'s host `bash`, whose caller is a client, not the model.
        let note = match self.jobs.park_intended(job) {
            Ok(_) if !self.detaches() => {
                format!(
                    "The command exited but left {left} running; it runs until the session ends."
                )
            }
            Err((job, _)) if !self.detaches() => {
                if let Some(group) = job.shared.untrack() {
                    let _ = tokio::task::spawn_blocking(move || group.kill()).await;
                }
                drop(job);
                format!(
                    "The command exited but left {left} running. It was stopped: {MAX_INTENDED_JOBS} \
                     commands' leftover processes are already running in this session."
                )
            }
            Ok(id) => format!(
                "The command exited but left {left} running; it is tracked as job {id} \
                 ({lifetime}). You will be told when it exits. bash_job {{\"job\": {id}, \"kill\": \
                 true}} stops it.{}",
                self.hand_over(&shared, id, JobKind::Adopted).await
            ),
            Err((job, running)) => {
                if let Some(group) = job.shared.untrack() {
                    let _ = tokio::task::spawn_blocking(move || group.kill()).await;
                }
                drop(job);
                format!(
                    "The command exited but left {left} running. It was stopped: \
                     {MAX_INTENDED_JOBS} {INTENDED_JOBS} are already running ({running}). Stop one \
                     with bash_job {{\"job\": N, \"kill\": true}}, then run this again."
                )
            }
        };
        let text = match &shell_status {
            Some(line) => append_status(&text, line),
            None => text,
        };
        let text = append_status(&text, &note);
        if shell_status.is_some() {
            Err(ToolError::Execution(text))
        } else {
            Ok(text.into())
        }
    }

    /// Hand a parked intended job to the sandbox, if the sandbox runs it ([`super::bash_sandbox::keep`]).
    /// Empty when it did, or nothing needed doing; otherwise the note the model needs, since the job
    /// then lives only as long as this replica's stream.
    async fn hand_over(&self, shared: &JobShared, id: u64, kind: JobKind) -> String {
        match super::bash_sandbox::keep(shared, id, kind, self.jobs.policy()).await {
            Ok(()) => String::new(),
            Err(e) => {
                tracing::warn!(job = id, error = %e, "the sandbox could not take over a job");
                " (The sandbox could not take it over, so it stops if this session closes.)"
                    .to_string()
            }
        }
    }

    /// A job has ended: its final result over the output produced since the last report.
    fn finish_job(
        &self,
        job: Job,
        end: JobEnd,
        progress: Option<&ToolProgress>,
    ) -> Result<ToolOutput, ToolError> {
        // Reported now: its directory in the sandbox can go once this is persisted.
        if let Some(link) = job.shared.sandbox() {
            self.jobs.discard_later(link);
        }
        let result = end.result.map_err(ToolError::Execution)?;
        let snap = {
            let mut out = job.shared.out();
            // A non-streaming runner (a test double, a remote endpoint) delivered nothing through the
            // sink; its captured output is the whole story.
            if !out.streamed {
                out.acc.append(&result.stdout);
                if !result.stderr.is_empty() {
                    if !result.stdout.is_empty() {
                        out.acc.append(b"\n");
                    }
                    out.acc.append(&result.stderr);
                }
            }
            out.acc.finish();
            out.acc.snapshot(true)
        };
        if let Some(p) = progress {
            emit_update(p, &snap);
        }
        let secs = end.at.duration_since(job.started);
        // The shell's own exit was reported when its leftover processes were adopted; this is
        // their end.
        if job.shared.orphans().is_some() && !result.timed_out {
            let text = clean(format_output(&snap, "(no new output)"));
            let id = job.id.unwrap_or_default();
            return Ok(append_status(
                &text,
                &format!(
                    "The processes job {id} left running have exited ({} after it started).",
                    super::format_duration(secs)
                ),
            )
            .into());
        }
        finish_result(&snap, &result, secs)
    }

    /// `bash_job`'s body: `{"job": N}` waits on job N (or returns its result, if it already
    /// ended) — for at most `timeout_ms`, if given (0: a peek) — and `{"job": N, "kill": true}`
    /// stops it.
    async fn resume_job(
        &self,
        job: &Value,
        input: &Value,
        progress: Option<&ToolProgress>,
    ) -> Result<ToolOutput, ToolError> {
        let id = job
            .as_u64()
            .or_else(|| job.as_str().and_then(|s| s.trim().parse().ok()))
            .ok_or_else(|| {
                ToolError::InvalidInput(format!("`job` must be a job number, got {job}"))
            })?;
        // Checked before the job leaves the table: a bad input must not drop (and so kill) it.
        // 0 is allowed here, unlike on `bash`: a peek at a job that is already running is the
        // cheap way to see a server's latest output, where on `bash` 0 would return before the
        // command could print anything.
        let timeout_ms = parse_timeout_ms(input, 0)?;
        // A reopened session reads its sandbox's jobs back the first time one is asked about.
        let checked = super::bash_sandbox::reconcile(&self.jobs).await;
        let Some(mut job) = self.jobs.take(id) else {
            if let Err(e) = checked {
                return Err(ToolError::Execution(format!(
                    "job {id} could not be checked: {e}"
                )));
            }
            if let Some(command) = self.jobs.take_checked_gone(id) {
                return Err(ToolError::InvalidInput(format!(
                    "job {id} (`{command}`) is no longer running: the sandbox, checked just now, no \
                     longer has it, and how it ended was not recorded. Start it again if you still \
                     need it."
                )));
            }
            return Err(ToolError::InvalidInput(format!(
                "no job {id}: it was already reported, was stopped, or never existed (jobs do not \
                 survive a restart of the agent)"
            )));
        };
        // exec v1.4: the call has the job now; an unheld one is read in and followed for it.
        super::bash_sandbox::taken(&job.shared).await;
        let shared = job.shared.clone();
        if input.get("kill").and_then(Value::as_bool) != Some(true) || job.shared.ended() {
            let hold = Hold::Foreground {
                ceiling: self.wait_ceiling,
                timeout_ms,
            };
            let result = self.attach(job, progress, hold).await;
            super::bash_sandbox::put_back(&shared, false);
            return result;
        }
        // Kill it — every process it started, or its remote stream — and wait for the kill, so a
        // caller that checks side effects right after sees the command gone.
        let survived = match job.shared.untrack() {
            Some(kill) => tokio::task::spawn_blocking(move || kill.kill())
                .await
                .unwrap_or_default(),
            None => None,
        };
        job.handle.abort();
        let _ = tokio::time::timeout(Duration::from_secs(5), &mut job.handle).await;
        let snap = job.shared.take_segment();
        let running = super::format_duration(job.started.elapsed());
        // Named, so a kill is never ambiguous about which process it hit.
        let command = std::mem::take(&mut job.command);
        drop(job);
        super::bash_sandbox::put_back(&shared, true);
        let text = clean(format_output(&snap, ""));
        match survived {
            None => Ok(append_status(
                &text,
                &format!("Job {id} (`{command}`) killed after {running}."),
            )
            .into()),
            // Not "killed": they still run, and nothing tracks them any more.
            Some(left) => Err(ToolError::Execution(append_status(
                &text,
                &format!(
                    "Job {id} (`{command}`) was sent SIGTERM and then SIGKILL after {running}, \
                     but {}",
                    survived_note(&left)
                ),
            ))),
        }
    }
}

/// What to say about processes a kill could not stop.
pub(crate) fn survived_note(left: &str) -> String {
    format!(
        "{left} could not be killed (permission denied: it runs as another user, e.g. under sudo) \
         and is still running, no longer tracked as a job. Stop it with the privilege that started \
         it."
    )
}

/// How a `bash` call starts its command.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Start {
    /// Wait for it, returning early at the model's `timeout_ms` (if any) or a yield.
    Foreground(Option<u64>),
    /// `run_in_background`: return after the settle with the job number.
    Background,
}

/// When an attached call hands control back while its command keeps running.
#[derive(Clone, Copy)]
enum Hold {
    /// On silence (the idle window), at `ceiling`, or at the model's `timeout_ms`.
    Foreground {
        ceiling: Duration,
        timeout_ms: Option<u64>,
    },
    /// After this long, whatever the command does: a background start's settle.
    Settle(Duration),
}

/// Why an attach returned before its command ended.
enum Yield {
    /// No output for this long: at least the idle window.
    Quiet(Duration),
    /// Still printing after this long: the call's ceiling.
    Ceiling(Duration),
    /// The model's `timeout_ms`.
    Timeout(u64),
    /// A background start settled with its command still running.
    Settled,
}

/// What the cap on intended jobs counts, as the model is told.
const INTENDED_JOBS: &str = "jobs started in the background or left running by a command";

/// The note for a foreground call whose `timeout_ms` passed while a runner that streams nothing (an
/// exec endpoint on protocol v1.1) held it until the command exited — see [`Bash::idle_window`].
fn timeout_unheard(shared: &JobShared, start: &Start, started: Instant) -> Option<String> {
    let Start::Foreground(Some(ms)) = *start else {
        return None;
    };
    let unheard = shared.remote && !shared.live.load(std::sync::atomic::Ordering::Relaxed);
    (unheard && started.elapsed() >= Duration::from_millis(ms)).then(|| {
        "This sandbox's exec endpoint cannot return before a command exits, so `timeout_ms` had \
         no effect."
            .to_string()
    })
}

/// Records a job's end when its runner task is dropped (aborted) or panics before recording one.
struct FinishOnDrop(Option<Arc<JobShared>>);

impl Drop for FinishOnDrop {
    fn drop(&mut self) {
        if let Some(shared) = self.0.take() {
            shared.untrack();
            shared.finish(Err("the command's task ended before it did".into()));
        }
    }
}

pub use super::bash_jobs::kill_live_jobs;

/// The model-facing result for a command that has exited (or that its runner timed out): the cleaned
/// output plus the status line pi appends. `secs` is how long it ran.
fn finish_result(
    snap: &OutputSnapshot,
    result: &ExecResult,
    secs: Duration,
) -> Result<ToolOutput, ToolError> {
    // No "(no output)" placeholder on a timeout: an empty-output timeout should read as just the
    // status line ("Command timed out after Ns"), not "(no output)" glued in front of it — matching
    // the reference agent's abort/timeout formatting, which substitutes nothing for empty content on
    // this path specifically (the success/exit-code path keeps the placeholder, since there the empty
    // case is a genuinely silent command, not an interrupt).
    let empty = if result.timed_out { "" } else { "(no output)" };
    let text = clean(format_output(snap, empty));
    match exit_status_line(result, secs) {
        None => Ok(text.into()),
        Some(status) => Err(ToolError::Execution(append_status(&text, &status))),
    }
}

/// The status line for a command that did not succeed, or `None` for one that did. Only an exit code
/// of literally `0` is success. `secs` is how long it ran.
fn exit_status_line(result: &ExecResult, secs: Duration) -> Option<String> {
    // `bash` sets no deadline of its own, so a timeout is the runner's: an exec endpoint enforcing
    // the `timeout_ms` on the wire (see `exec_endpoint`'s "Timeouts").
    if result.timed_out {
        return Some(format!(
            "Command timed out after {}: the sandbox's exec endpoint stopped it.",
            super::format_duration(secs)
        ));
    }
    match (result.code, result.signal) {
        (Some(0), _) => None,
        // Non-zero exit is an error result that still carries the output — pi throws here too.
        (Some(code), _) => Some(format!("Command exited with code {code}")),
        // Killed by a signal: `ExitStatus::code()` is `None` in exactly this case, and this arm
        // used to fall in with `Some(0)` and report **success**. An OOM-killed build, a SIGSEGV, a
        // SIGTERM'd server — every one of them came back to the model as a clean run, silently, and
        // with `"(no output)"` if the command hadn't printed anything before dying. The model then
        // proceeds as though the step succeeded. The timeout case never reaches here (it returns
        // above), so a `None` code with a signal is unambiguous.
        (None, Some(signal)) => {
            let hint = if signal == 9 {
                " — typically the OOM killer or an external `kill -9`"
            } else {
                ""
            };
            Some(format!(
                "Command was killed by {}{hint}",
                super::bash_jobs::signal_text(signal)
            ))
        }
        // No code and no signal: not reachable on Unix (the two are exhaustive there), and on a
        // non-Unix host there is no signal concept to report. Genuinely unknown rather than known-
        // bad, so this keeps the original benefit-of-the-doubt behavior instead of inventing a
        // failure.
        (None, None) => None,
    }
}

#[async_trait]
impl Tool for Bash {
    fn name(&self) -> &str {
        "bash"
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn input_schema(&self) -> Value {
        // No `job` field: waiting on or killing a job is `bash_job`'s, not optional
        // fields here. A model that fills every property (gpt-6-luna sends `"job": 0` on every
        // call) would otherwise turn each command into a job lookup — seen in a live probe.
        //
        // `timeout_ms` says what it does because models read a bare "timeout" as "hand control back
        // after this long" (Codex's `yield_time_ms`): that is now exactly what it is, and it must
        // not read as a kill. `run_in_background` is a boolean on `bash`, not a tool of its own: it
        // beat a separate start tool on every model in the background-shell probe, and no model set
        // it on a short command.
        json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "Shell command to run." },
                "cwd": { "type": "string", "description": "Working directory." },
                "timeout_ms": {
                    "type": "integer",
                    "description": format!(
                        "How long to wait, in milliseconds, before returning with the output so \
                         far. The command is not stopped: it keeps running as a job and you are \
                         told when it exits. Without it, the call returns when the command exits, \
                         after {} s without output, or after {} minutes.",
                        self.idle_yield.unwrap_or(IDLE_YIELD).as_secs(),
                        self.call_ceiling.as_secs() / 60,
                    )
                },
                "run_in_background": {
                    "type": "boolean",
                    "description": "Start the command and return immediately with a job number instead of waiting for it. `timeout_ms` is ignored with it."
                }
            },
            "required": ["command"]
        })
    }

    // An opaque shell command can mutate anything reachable from its `cwd` — there's no single path
    // to report via `write_target`, so without this a same-turn `edit`/`write` it happens to race
    // (e.g. `edit foo.py` batched with `bash: black foo.py`) would run fully concurrently against it.
    fn conservative_exclusive(&self) -> bool {
        true
    }

    // The job table: finish notices, the stop reminder and the compaction carry (see `bash_jobs`).
    fn background(&self) -> Option<Arc<dyn agent_core::Background>> {
        Some(self.jobs.clone())
    }

    async fn run(&self, input: Value) -> Result<ToolOutput, ToolError> {
        self.exec(input, None).await
    }

    async fn run_streaming(
        &self,
        input: Value,
        progress: &ToolProgress,
    ) -> Result<ToolOutput, ToolError> {
        self.exec(input, Some(progress)).await
    }
}

impl Bash {
    /// Whether this tool can leave a command running past its call — the condition for registering
    /// [`BashJob`] beside it.
    pub fn detaches(&self) -> bool {
        self.idle_yield.is_some()
    }
}

/// [`BashJob`]'s name. It follows `bash` through every tool filter (`tools::filter_allows`): a `bash`
/// whose quiet command became a job must always have the tool that waits on it.
pub const JOB_TOOL: &str = "bash_job";

/// `bash_job`: wait on or kill a command `bash` left running (see [`IDLE_YIELD`]). Registered only
/// beside a detaching `bash`, and over that same instance, so the two share one job table.
///
/// A tool of its own rather than optional `job`/`kill` fields on `bash`, because some models fill
/// every property of a schema: a live gpt-6-luna probe sent `"job": 0, "kill": false` on every
/// `bash` call, and an optional `job` field turned each command into a failed job lookup.
pub struct BashJob {
    bash: Arc<Bash>,
}

impl BashJob {
    pub fn new(bash: Arc<Bash>) -> Self {
        Self { bash }
    }
}

#[async_trait]
impl Tool for BashJob {
    fn name(&self) -> &str {
        JOB_TOOL
    }
    fn description(&self) -> &str {
        "Wait for, look at, or stop a job: a command `bash` left running (its result or a notice \
         said \"job N\"). Returns what the job printed since the last report: its final result once \
         it exits (at once, if it already has), or the output so far if it is still running when \
         the call returns. `timeout_ms: 0` returns at once: how to see a server's or watcher's \
         latest output. `kill: true` stops the job and every process it started. You are told when \
         a job exits without calling this: wait only for a job whose result you need, never for a \
         server or watcher to exit, and do not poll."
    }
    fn input_schema(&self) -> Value {
        // `timeout_ms` means what it means on `bash` (and says so in the same words), plus 0: a
        // peek. The defaults it names are this instance's.
        json!({
            "type": "object",
            "properties": {
                "job": { "type": "integer", "description": "The job number from the `bash` result." },
                "kill": { "type": "boolean", "description": "Stop the job instead of waiting for it." },
                "timeout_ms": {
                    "type": "integer",
                    "description": format!(
                        "How long to wait, in milliseconds, before returning with the output so \
                         far; 0 returns at once. The job is not stopped: it keeps running and you \
                         are told when it exits. Without it, the call returns when the job exits, \
                         after {} s without output, or after {} minutes.",
                        self.bash.idle_yield.unwrap_or(IDLE_YIELD).as_secs(),
                        self.bash.wait_ceiling.as_secs() / 60,
                    )
                }
            },
            "required": ["job"]
        })
    }
    // Waiting reads nothing, but the job it waits on mutates as freely as any `bash` command.
    fn conservative_exclusive(&self) -> bool {
        true
    }
    fn background(&self) -> Option<Arc<dyn agent_core::Background>> {
        Some(self.bash.jobs.clone())
    }
    async fn run(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let job = input.get("job").cloned().unwrap_or(Value::Null);
        self.bash.resume_job(&job, &input, None).await
    }
    async fn run_streaming(
        &self,
        input: Value,
        progress: &ToolProgress,
    ) -> Result<ToolOutput, ToolError> {
        let job = input.get("job").cloned().unwrap_or(Value::Null);
        self.bash.resume_job(&job, &input, Some(progress)).await
    }
}

/// Output hygiene for text quoted outside a `bash` result (a job's finish notice): see [`clean`].
pub(crate) fn clean_output(s: &str) -> String {
    clean_str(s).into_owned()
}

/// Emit one streamed progress snapshot: the cleaned output so far, plus truncation + full-output-path
/// details when the output has overflowed (pi's `onUpdate({content, details})`).
///
/// Uses [`clean_str`] (borrowing) rather than [`clean`] (owning): `snap` only ever arrives by reference
/// here, so `clean` would need a `.clone()` of the whole accumulated content on every throttled tick just
/// to have something to hand it — `clean_str` reads `&snap.content` directly and only allocates when the
/// content actually needs stripping, same as `clean` already did internally.
fn emit_update(p: &ToolProgress, snap: &OutputSnapshot) {
    p.emit(clean_str(&snap.content), truncation_details(snap));
}

/// The `details` payload for a progress update: nested `{truncation: {...}, full_output_path}` —
/// matching the reference agent's `{truncation, fullOutputPath}` shape structurally (field names stay
/// snake_case: this is the only place in the whole wire protocol that would otherwise mix camelCase
/// into an all-snake_case convention, and nothing here needs byte-for-byte pi compatibility, just the
/// same information). The full `Truncation` record — not just the four fields this used to send — so a
/// future client doesn't have to guess `truncated_by`/`last_line_partial`/`max_lines`/`max_bytes` from
/// the others. `None` when the output was never truncated (nothing extra to report).
fn truncation_details(snap: &OutputSnapshot) -> Option<Value> {
    snap.truncation.truncated.then(|| {
        let by = match snap.truncation.truncated_by {
            Some(TruncatedBy::Lines) => Some("lines"),
            Some(TruncatedBy::Bytes) => Some("bytes"),
            None => None,
        };
        json!({
            "truncation": {
                "truncated": true,
                "truncated_by": by,
                "total_lines": snap.truncation.total_lines,
                "total_bytes": snap.truncation.total_bytes,
                "output_lines": snap.truncation.output_lines,
                "output_bytes": snap.truncation.output_bytes,
                "last_line_partial": snap.truncation.last_line_partial,
                "max_lines": snap.truncation.max_lines,
                "max_bytes": snap.truncation.max_bytes,
            },
            "full_output_path": snap.full_output_path,
            // A spill file that hit its byte budget holds only a prefix of the stream (see
            // `output::MAX_SPILL_BYTES`). A client rendering `full_output_path` as "the complete
            // output, click to open" would otherwise present that prefix as the whole thing — the
            // same dishonesty `format_output`'s marker guards against for the model's own text.
            "full_output_capped": snap.full_output_capped,
            "full_output_bytes": snap.full_output_bytes,
        })
    })
}

/// Append a status line (`exit code` / `timed out` / `cancelled`) after the output, pi-style
/// (`text\n\n<status>`).
fn append_status(text: &str, status: &str) -> String {
    if text.is_empty() {
        status.to_string()
    } else {
        format!("{text}\n\n{status}")
    }
}

/// ANSI escape + OSC sequences a terminal emits but the model can't use: CSI/SGR colour and cursor
/// moves (`ESC [ … final`, or the 8-bit C1 CSI introducer `\x9b`) plus OSC strings (`ESC ] … `,
/// terminated by BEL, ST (`ESC \`), or the 8-bit C1 ST `\x9c`). A direct port of pi's own
/// `ansi.ts::ansiRegex` (itself derived from the `ansi-regex`/`strip-ansi` npm packages — see that
/// file's license header) rather than a from-scratch pattern, so it inherits the exact same coverage:
/// the prior version only recognized a BEL-terminated OSC (missing an OSC-8 hyperlink or any other
/// modern tool that ST-terminates instead) and excluded `:` from CSI parameters (missing 24-bit
/// truecolor SGR, e.g. `\x1b[38:2:255:0:0m`, which uses colon-separated sub-parameters) — both leaked
/// raw escape-sequence noise into the model's context instead of being stripped. Built once; the
/// pattern is a static literal, so a build failure is impossible — we model it as `None` instead of
/// unwrapping.
fn ansi_re() -> Option<&'static Regex> {
    static RE: OnceLock<Option<Regex>> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?:\x1b\][\s\S]*?(?:\x07|\x1b\x5c|\x9c))|[\x1b\x9b][\[\]()#;?]*(?:\d{1,4}(?:[;:]\d{0,4})*)?[\dA-PR-TZcf-nq-uy=><~]",
        )
        .ok()
    })
    .as_ref()
}

/// Whitespace the model actually needs (`\t`, `\n`, `\r`) survives; every other C0 control byte (NUL,
/// stray ESC, …) and the Unicode interlinear-annotation controls are dropped so binary output can't
/// corrupt the context.
fn is_keepable(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\r') || (c >= ' ' && !('\u{fff9}'..='\u{fffb}').contains(&c))
}

/// Strip terminal control noise from output. Two single passes: a regex removes ANSI escape sequences,
/// then we drop residual control bytes. When the text is already clean we reuse the string we have
/// rather than reallocating. (pi sanitizes binary output too; we also drop ANSI, which the model can't
/// use — a small, deliberate step past pi that only ever removes noise.)
fn clean(s: String) -> String {
    match clean_str(&s) {
        Cow::Borrowed(_) => s,
        Cow::Owned(owned) => owned,
    }
}

/// [`clean`]'s own logic, borrowing: returns `s` itself (via `Cow::Borrowed`) when nothing needed
/// stripping, so a caller that only has `&str` (e.g. [`emit_update`], reading a snapshot it doesn't own)
/// isn't forced to clone the whole input up front just to satisfy an owned-`String` signature.
fn clean_str(s: &str) -> Cow<'_, str> {
    let stripped = match ansi_re() {
        Some(re) => re.replace_all(s, ""),
        None => Cow::Borrowed(s),
    };
    if stripped.chars().all(is_keepable) {
        return stripped;
    }
    Cow::Owned(stripped.chars().filter(|&c| is_keepable(c)).collect())
}

#[cfg(test)]
#[path = "bash_idle_yield_tests.rs"]
mod idle_yield_tests;

#[cfg(all(test, target_os = "linux"))]
#[path = "bash_jobs_tests.rs"]
mod jobs_tests;

#[cfg(all(test, target_os = "linux"))]
#[path = "bash_background_tests.rs"]
mod background_tests;

#[cfg(all(test, target_os = "linux"))]
#[path = "bash_kill_tests.rs"]
mod kill_tests;

#[cfg(all(test, target_os = "linux"))]
#[path = "bash_jobs_reopen_tests.rs"]
mod reopen_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::exec::{ChunkSink, ExecResult};
    use crate::tools::output::{DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, Truncation, format_size};

    #[test]
    fn truncation_details_nests_the_full_record_under_a_truncation_key() {
        let snap = OutputSnapshot {
            content: "tail".into(),
            truncation: Truncation {
                truncated: true,
                truncated_by: Some(TruncatedBy::Lines),
                total_lines: 2500,
                total_bytes: 12345,
                output_lines: 2000,
                output_bytes: 9000,
                last_line_partial: true,
                max_lines: DEFAULT_MAX_LINES,
                max_bytes: DEFAULT_MAX_BYTES,
            },
            full_output_path: Some("/tmp/pi-bash-abc.log".into()),
            full_output_capped: false,
            full_output_bytes: 12345,
            spill_disabled: false,
            last_line_bytes: 4,
        };
        let details = truncation_details(&snap).expect("truncated snapshot must carry details");
        assert_eq!(details["truncation"]["truncated"], true);
        assert_eq!(details["truncation"]["truncated_by"], "lines");
        assert_eq!(details["truncation"]["total_lines"], 2500);
        assert_eq!(details["truncation"]["total_bytes"], 12345);
        assert_eq!(details["truncation"]["output_lines"], 2000);
        assert_eq!(details["truncation"]["output_bytes"], 9000);
        assert_eq!(details["truncation"]["last_line_partial"], true);
        assert_eq!(details["truncation"]["max_lines"], DEFAULT_MAX_LINES);
        assert_eq!(details["truncation"]["max_bytes"], DEFAULT_MAX_BYTES);
        assert_eq!(details["full_output_path"], "/tmp/pi-bash-abc.log");
        assert_eq!(details["full_output_capped"], false);
        assert_eq!(details["full_output_bytes"], 12345);
        // `full_output_path` is a sibling of `truncation`, not nested inside it.
        assert!(details["truncation"].get("full_output_path").is_none());
    }

    #[test]
    fn truncation_details_tells_a_client_when_the_spill_file_holds_only_a_prefix() {
        // A spill file that hit `output::MAX_SPILL_BYTES` stops short of the stream. A client that
        // renders `full_output_path` as "the complete output" would otherwise show a firehose's first
        // 128 MiB as if it were the whole thing — the wire payload has to carry the same admission the
        // model's own marker text does.
        let snap = OutputSnapshot {
            content: "tail".into(),
            truncation: Truncation {
                truncated: true,
                truncated_by: Some(TruncatedBy::Bytes),
                total_lines: 900_000,
                total_bytes: 4_000_000_000,
                output_lines: 2000,
                output_bytes: 51_200,
                last_line_partial: false,
                max_lines: DEFAULT_MAX_LINES,
                max_bytes: DEFAULT_MAX_BYTES,
            },
            full_output_path: Some("/tmp/pi-bash-abc.log".into()),
            full_output_capped: true,
            full_output_bytes: 134_217_728,
            spill_disabled: false,
            last_line_bytes: 4,
        };
        let details = truncation_details(&snap).expect("truncated snapshot must carry details");
        assert_eq!(details["full_output_capped"], true);
        assert_eq!(details["full_output_bytes"], 134_217_728u64);
        assert_eq!(details["truncation"]["total_bytes"], 4_000_000_000u64);
    }

    #[test]
    fn truncation_details_is_none_when_output_was_not_truncated() {
        let snap = OutputSnapshot {
            content: "all of it".into(),
            truncation: Truncation {
                truncated: false,
                truncated_by: None,
                total_lines: 3,
                total_bytes: 9,
                output_lines: 3,
                output_bytes: 9,
                last_line_partial: false,
                max_lines: DEFAULT_MAX_LINES,
                max_bytes: DEFAULT_MAX_BYTES,
            },
            full_output_path: None,
            full_output_capped: false,
            full_output_bytes: 0,
            spill_disabled: false,
            last_line_bytes: 3,
        };
        assert!(truncation_details(&snap).is_none());
    }

    /// Records the last invocation and returns a canned result (a non-streaming runner: it never
    /// delivers chunks, so `bash` exercises its fallback-from-`ExecResult` path).
    struct RecordingRunner {
        last: std::sync::Mutex<Option<(String, Vec<String>, Duration)>>,
        result: ExecResult,
    }

    #[async_trait]
    impl CommandRunner for RecordingRunner {
        async fn run(
            &self,
            program: &str,
            args: &[String],
            _cwd: Option<&str>,
            timeout: Duration,
        ) -> std::io::Result<ExecResult> {
            *self.last.lock().unwrap() = Some((program.to_string(), args.to_vec(), timeout));
            Ok(self.result.clone())
        }
    }

    fn recording(result: ExecResult) -> Arc<RecordingRunner> {
        Arc::new(RecordingRunner {
            last: std::sync::Mutex::new(None),
            result,
        })
    }

    /// A streaming runner that replays a fixed sequence of raw byte chunks through `on_chunk`, then
    /// returns a canned final result — for exercising the live-streaming path (`RecordingRunner` above
    /// never streams at all).
    struct ChunkedRunner {
        chunks: Vec<Vec<u8>>,
    }

    #[async_trait]
    impl CommandRunner for ChunkedRunner {
        async fn run(
            &self,
            _program: &str,
            _args: &[String],
            _cwd: Option<&str>,
            _timeout: Duration,
        ) -> std::io::Result<ExecResult> {
            unreachable!("test only exercises the streaming path")
        }

        async fn run_streaming(
            &self,
            _program: &str,
            _args: &[String],
            _cwd: Option<&str>,
            _timeout: Duration,
            on_chunk: ChunkSink<'_>,
        ) -> std::io::Result<ExecResult> {
            for chunk in &self.chunks {
                on_chunk(chunk);
            }
            Ok(ExecResult {
                code: Some(0),
                ..Default::default()
            })
        }
    }

    #[tokio::test]
    async fn invokes_the_resolved_shell_dash_c() {
        let runner = recording(ExecResult {
            code: Some(0),
            stdout: "hi\n".into(),
            ..Default::default()
        });
        let out = Bash::with_runner(runner.clone())
            .run(json!({ "command": "echo hi" }))
            .await
            .unwrap()
            .text;
        // pi does not trim: the command's output is shown as-is (trailing newline kept).
        assert_eq!(out, "hi\n");
        let (prog, args, _timeout) = runner.last.lock().unwrap().clone().unwrap();
        assert_eq!(prog, resolve_shell());
        assert_eq!(args, vec!["-c".to_string(), "echo hi".to_string()]);
    }

    #[tokio::test]
    async fn with_shell_path_overrides_the_auto_resolved_shell() {
        let runner = recording(ExecResult {
            code: Some(0),
            stdout: "hi\n".into(),
            ..Default::default()
        });
        Bash::with_runner(runner.clone())
            .with_shell_path("/bin/dash")
            .run(json!({ "command": "echo hi" }))
            .await
            .unwrap();
        let (prog, _, _) = runner.last.lock().unwrap().clone().unwrap();
        assert_eq!(prog, "/bin/dash");
        // The auto-resolution path is unaffected when no override is set.
        assert_ne!(resolve_shell(), "/bin/dash");
    }

    #[test]
    fn resolve_shell_never_panics_and_returns_a_plausible_path() {
        let shell = resolve_shell();
        assert!(!shell.is_empty());
        assert!(shell == "sh" || shell.contains("bash"));
        // Cached: calling twice returns the same answer.
        assert_eq!(resolve_shell(), shell);
    }

    #[tokio::test]
    async fn timeout_ms_never_reaches_the_runner_as_a_kill_deadline() {
        // `timeout_ms` only decides when the call returns. A runner that cannot track a job (no
        // live output, no kill: here a test double, in production a v1.1 exec endpoint) gets the
        // one internal deadline that ends a hung command, whatever the model asked for.
        for input in [
            json!({ "command": "echo hi" }),
            json!({ "command": "echo hi", "timeout_ms": 5_000 }),
        ] {
            let runner = recording(ExecResult {
                code: Some(0),
                stdout: "hi\n".into(),
                ..Default::default()
            });
            Bash::with_runner(runner.clone()).run(input).await.unwrap();
            let (_, _, timeout) = runner.last.lock().unwrap().clone().unwrap();
            assert_eq!(timeout, crate::tools::exec::UNTRACKED_DEADLINE);
        }
    }

    #[test]
    fn the_schema_says_timeout_ms_returns_and_offers_run_in_background() {
        let schema = Bash::real().input_schema();
        let timeout = schema["properties"]["timeout_ms"]["description"]
            .as_str()
            .unwrap();
        assert!(timeout.contains("before returning"), "{timeout}");
        assert!(timeout.contains("The command is not stopped"), "{timeout}");
        assert!(
            timeout.contains("after 60 s without output, or after 10 minutes"),
            "{timeout}"
        );
        assert_eq!(
            schema["properties"]["run_in_background"],
            json!({
                "type": "boolean",
                "description": "Start the command and return immediately with a job number instead of waiting for it. `timeout_ms` is ignored with it."
            })
        );
        assert_eq!(schema["required"], json!(["command"]));
    }

    #[test]
    fn description_documents_the_truncation_budget_and_when_a_call_returns() {
        let desc = Bash::real().description().to_string();
        assert!(
            !desc.contains("defaults to") && !desc.contains("Timeout"),
            "no kill timeout to document: {desc}"
        );
        assert!(
            desc.contains(
                "Set `run_in_background: true` for a command you don't need to wait on (a server, \
                 a watcher): the call returns at once with a job number and you are notified when \
                 it exits; don't poll or sleep."
            ),
            "the measured wording: {desc}"
        );
        assert!(
            desc.contains("after its `timeout_ms` returns early"),
            "{desc}"
        );
        // Over v1.1 a background job is the sandbox's (it starts kept and is polled); only a
        // foreground call still has to wait for its command.
        assert!(
            desc.contains("a call waits for its command to exit")
                && !desc.contains("`run_in_background` is unavailable"),
            "the v1.1 exception: {desc}"
        );
        assert!(
            desc.contains(&DEFAULT_MAX_LINES.to_string())
                && desc.contains(&format_size(DEFAULT_MAX_BYTES as u64)),
            "description should state the truncation budget, got: {desc}"
        );

        // The spill file is conditional — it is not written when the command ran somewhere this
        // process cannot write a readable file to (`OutputAccumulator::without_spill`). Promising one
        // unconditionally sent the model looking for a path that was never going to be there.
        assert!(
            !desc.contains("the complete output is saved to a temp file"),
            "description must not promise a temp file the remote path never writes, got: {desc}"
        );
        assert!(
            desc.contains("whether the complete output was saved to a temp file"),
            "description should state the condition, got: {desc}"
        );
    }

    #[tokio::test]
    async fn zero_or_negative_timeout_ms_is_rejected_not_silently_coerced() {
        let runner = recording(ExecResult {
            code: Some(0),
            ..Default::default()
        });
        let bash = Bash::with_runner(runner.clone());

        // 0 would return before the command could print anything.
        let err = bash
            .run(json!({ "command": "echo hi", "timeout_ms": 0 }))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)), "got: {err:?}");

        // A negative value previously fell silently back to the default instead of being rejected.
        let err = bash
            .run(json!({ "command": "echo hi", "timeout_ms": -100 }))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)), "got: {err:?}");

        // Neither invalid call should have reached the runner at all.
        assert!(runner.last.lock().unwrap().is_none());

        // A missing `timeout_ms` means "no timeout of your own" — not an error.
        bash.run(json!({ "command": "echo hi" })).await.unwrap();
        assert!(runner.last.lock().unwrap().is_some());
    }

    #[tokio::test]
    async fn a_timeout_ms_past_the_max_is_rejected_not_treated_as_effectively_unbounded() {
        // Matches pi's `resolveTimeoutMs` upper-bound check: a value past it is a typo or garbage,
        // and saying so beats silently accepting it.
        let runner = recording(ExecResult {
            code: Some(0),
            ..Default::default()
        });
        let bash = Bash::with_runner(runner.clone());

        let err = bash
            .run(json!({ "command": "echo hi", "timeout_ms": MAX_TIMEOUT_MS + 1 }))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)), "got: {err:?}");
        assert!(runner.last.lock().unwrap().is_none());

        // The max itself is still accepted (an inclusive bound, not exclusive).
        bash.run(json!({ "command": "echo hi", "timeout_ms": MAX_TIMEOUT_MS }))
            .await
            .unwrap();
        assert!(runner.last.lock().unwrap().is_some());
    }

    #[tokio::test]
    async fn nonzero_exit_is_an_error_carrying_the_output() {
        let runner = recording(ExecResult {
            code: Some(2),
            stderr: "boom".into(),
            ..Default::default()
        });
        let err = Bash::with_runner(runner)
            .run(json!({ "command": "false" }))
            .await
            .unwrap_err();
        let ToolError::Execution(msg) = err else {
            panic!("expected Execution error")
        };
        assert!(msg.contains("boom"));
        assert!(msg.contains("Command exited with code 2"));
    }

    #[tokio::test]
    async fn a_signal_killed_command_is_an_error_not_a_silent_success() {
        // The regression: `ExitStatus::code()` is `None` for a signal-terminated process, and that
        // `None` used to share an arm with `Some(0)` — so an OOM-killed build reported **success** to
        // the model, with `"(no output)"` standing in for the output it never got to produce.
        let runner = recording(ExecResult {
            code: None,
            signal: Some(9),
            stdout: "compiling...".into(),
            ..Default::default()
        });
        let err = Bash::with_runner(runner)
            .run(json!({ "command": "make -j64" }))
            .await
            .unwrap_err();
        let ToolError::Execution(msg) = err else {
            panic!("expected Execution error, got: {err:?}")
        };
        assert!(msg.contains("compiling..."), "output must survive: {msg}");
        assert!(msg.contains("SIGKILL"), "the signal must be named: {msg}");
        assert!(
            msg.contains("OOM"),
            "SIGKILL's most likely cause is worth stating: {msg}"
        );
    }

    #[tokio::test]
    async fn an_unnamed_signal_is_still_reported_as_a_failure_by_number() {
        let runner = recording(ExecResult {
            code: None,
            signal: Some(31),
            ..Default::default()
        });
        let err = Bash::with_runner(runner)
            .run(json!({ "command": "weird" }))
            .await
            .unwrap_err();
        let ToolError::Execution(msg) = err else {
            panic!("expected Execution error, got: {err:?}")
        };
        assert!(msg.contains("signal 31"), "got: {msg}");
    }

    #[tokio::test]
    async fn a_real_signal_killed_command_reports_failure_end_to_end() {
        // Through the real runner and a real process, not a double: proves `exit_signal` actually
        // populates `ExecResult::signal` off a genuine `ExitStatus` rather than the field only ever
        // being set by test fixtures.
        let err = Bash::real()
            .run(json!({ "command": "kill -9 $$" }))
            .await
            .unwrap_err();
        let ToolError::Execution(msg) = err else {
            panic!("expected Execution error, got: {err:?}")
        };
        assert!(msg.contains("SIGKILL"), "got: {msg}");
    }

    #[tokio::test]
    async fn real_runner_executes() {
        let out = Bash::real()
            .run(json!({ "command": "printf done" }))
            .await
            .unwrap()
            .text;
        assert_eq!(out, "done");
    }

    #[tokio::test]
    async fn command_prefix_runs_before_the_command_in_the_same_shell() {
        // pi: tools.test.ts, "should prepend command prefix when configured" — the prefix must run in
        // the *same* shell invocation as the command (a variable it sets is visible to the command).
        let out = Bash::real()
            .with_command_prefix("export TEST_VAR=hello")
            .run(json!({ "command": "echo $TEST_VAR" }))
            .await
            .unwrap()
            .text;
        assert_eq!(out.trim(), "hello");
    }

    #[tokio::test]
    async fn command_prefix_and_command_output_both_appear_in_order() {
        // pi: tools.test.ts, "should include output from both prefix and command".
        let out = Bash::real()
            .with_command_prefix("echo prefix-output")
            .run(json!({ "command": "echo command-output" }))
            .await
            .unwrap()
            .text;
        assert_eq!(out.trim(), "prefix-output\ncommand-output");
    }

    #[tokio::test]
    async fn no_command_prefix_runs_the_command_unmodified() {
        // pi: tools.test.ts, "should work without command prefix".
        let out = Bash::real()
            .run(json!({ "command": "echo no-prefix" }))
            .await
            .unwrap()
            .text;
        assert_eq!(out.trim(), "no-prefix");
    }

    #[tokio::test]
    async fn strips_ansi_escape_sequences() {
        let runner = recording(ExecResult {
            code: Some(0),
            stdout: "\x1b[31mred\x1b[0m \x1b[2Aup\x1b]0;title\x07 done".into(),
            ..Default::default()
        });
        let out = Bash::with_runner(runner)
            .run(json!({ "command": "x" }))
            .await
            .unwrap()
            .text;
        assert_eq!(out, "red up done");
    }

    #[tokio::test]
    async fn strips_an_st_terminated_osc_sequence_not_just_bel_terminated() {
        // Pi-parity audit M9: the prior regex only recognized `ESC ] … BEL` — an OSC-8 hyperlink (or
        // any modern tool) terminated with ST (`ESC \`) instead leaked raw escape-sequence noise into
        // the model's context. Matches pi's own `ansi.ts`, which accepts either terminator.
        let runner = recording(ExecResult {
            code: Some(0),
            stdout: "\x1b]8;;http://example.com\x1b\\link text\x1b]8;;\x1b\\ done".into(),
            ..Default::default()
        });
        let out = Bash::with_runner(runner)
            .run(json!({ "command": "x" }))
            .await
            .unwrap()
            .text;
        assert_eq!(out, "link text done");
    }

    #[tokio::test]
    async fn strips_a_colon_separated_truecolor_sgr_sequence() {
        // Pi-parity audit M9: the prior regex excluded `:` from CSI parameters entirely, so a 24-bit
        // truecolor SGR sequence (`\x1b[38:2:255:0:0m`, colon-separated sub-parameters) wasn't
        // recognized at all and leaked through verbatim. Matches pi's own `ansi.ts`.
        let runner = recording(ExecResult {
            code: Some(0),
            stdout: "\x1b[38:2:255:0:0mRED\x1b[0m done".into(),
            ..Default::default()
        });
        let out = Bash::with_runner(runner)
            .run(json!({ "command": "x" }))
            .await
            .unwrap()
            .text;
        assert_eq!(out, "RED done");
    }

    #[tokio::test]
    async fn sanitizes_control_characters() {
        let runner = recording(ExecResult {
            code: Some(0),
            stdout: "a\x00b\x07c\td\ne".into(),
            ..Default::default()
        });
        let out = Bash::with_runner(runner)
            .run(json!({ "command": "x" }))
            .await
            .unwrap()
            .text;
        assert_eq!(out, "abc\td\ne");
    }

    #[tokio::test]
    async fn large_output_keeps_the_tail_and_points_at_the_full_file() {
        // 5000 lines exceed the 2000-line cap: the accumulator keeps the tail, reports the range, and
        // spills the complete output to a temp file the marker names (pi's `Full output:`).
        let body = (0..5000)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let runner = recording(ExecResult {
            code: Some(0),
            stdout: body.into(),
            ..Default::default()
        });
        let out = Bash::with_runner(runner)
            .run(json!({ "command": "x" }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("[Showing lines"), "range marker present");
        assert!(out.contains("Full output:"), "full-output path present");
        assert!(out.contains("4999"), "tail kept (the last line is present)");
        assert!(
            !out.starts_with("0\n1\n"),
            "head dropped (tail truncation, not head+tail)"
        );
    }

    /// `bash` sets no deadline, but a runner can enforce one: an exec endpoint stopping a command
    /// at the `timeout_ms` on the wire.
    #[tokio::test]
    async fn a_runner_reported_timeout_is_an_error() {
        let runner = recording(ExecResult {
            timed_out: true,
            ..Default::default()
        });
        let err = Bash::with_runner(runner)
            .run(json!({ "command": "sleep 10", "timeout_ms": 5000 }))
            .await
            .unwrap_err();
        let ToolError::Execution(msg) = err else {
            panic!("expected Execution error")
        };
        assert!(msg.contains("timed out"));
    }

    #[tokio::test]
    async fn a_timed_out_command_with_truncated_output_still_reports_the_full_output_path() {
        // pi: tools.test.ts, "should include full output path for truncated timeout and abort errors"
        // — iterates the timeout and aborted cases against 3000 lines of output. `truncation` and
        // `timeout` each have their own dedicated test above, but neither proves they compose: a
        // command that was both chatty *and* killed for running too long must still carry the
        // `[Showing lines …]`/`Full output: <path>` markers on its timeout error, with the complete
        // (untruncated) output actually readable from that path — not just an empty "timed out" message
        // that quietly drops everything the command had already printed.
        let body = (1..=3000)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let runner = recording(ExecResult {
            stdout: body.clone().into(),
            timed_out: true,
            ..Default::default()
        });
        let err = Bash::with_runner(runner)
            .run(json!({ "command": "chatty-and-slow", "timeout_ms": 5000 }))
            .await
            .unwrap_err();
        let ToolError::Execution(msg) = err else {
            panic!("expected Execution error")
        };

        assert!(msg.contains("Command timed out after"), "got: {msg}");
        assert!(
            msg.contains("[Showing lines"),
            "range marker missing: {msg}"
        );
        assert!(msg.contains("2999"), "tail must be kept: {msg}");
        assert!(
            !msg.contains("1\n2\n3\n4"),
            "head must be dropped (tail truncation, not head+tail): {msg}"
        );

        let path = msg
            .split("Full output: ")
            .nth(1)
            .and_then(|s| s.split(']').next())
            .expect("must carry a full-output path");
        assert!(
            std::path::Path::new(path).exists(),
            "full-output file must actually exist: {path}"
        );
        let full = std::fs::read_to_string(path).unwrap();
        assert_eq!(
            full, body,
            "the complete, untruncated output must be recoverable from the full-output file even \
             though the command timed out"
        );
    }

    #[tokio::test]
    async fn nonexistent_cwd_is_a_clear_invalid_input_error_not_a_raw_spawn_failure() {
        let runner = recording(ExecResult {
            code: Some(0),
            ..Default::default()
        });
        let err = Bash::with_runner(runner.clone())
            .run(json!({ "command": "echo hi", "cwd": "/definitely/not/a/real/path/xyz" }))
            .await
            .unwrap_err();
        match err {
            ToolError::InvalidInput(msg) => {
                assert!(
                    msg.contains("Working directory does not exist"),
                    "got: {msg}"
                );
                // pi-parity fix: pi's own bash.ts message is two lines — the plain fact, plus this
                // explanatory second line stating the consequence ("Cannot execute bash commands.") —
                // not just the first line on its own.
                assert!(msg.contains("Cannot execute bash commands."), "got: {msg}");
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
        // Must fail before ever reaching the runner.
        assert!(runner.last.lock().unwrap().is_none());
    }

    #[tokio::test]
    async fn a_cwd_that_is_a_file_not_a_directory_gets_its_own_distinct_message() {
        // Pi-parity fix (task 51): `.is_dir()` correctly rejects a non-directory `cwd` up front (this
        // is actually stricter than pi's own existence-only check), but the error message previously
        // always said "does not exist" even when the path exists and is simply a file. This must be
        // told apart from the genuinely-missing case exercised by
        // `nonexistent_cwd_is_a_clear_invalid_input_error_not_a_raw_spawn_failure` above.
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("not-a-directory.txt");
        std::fs::write(&file_path, "x").unwrap();

        let runner = recording(ExecResult {
            code: Some(0),
            ..Default::default()
        });
        let err = Bash::with_runner(runner.clone())
            .run(json!({ "command": "echo hi", "cwd": file_path.to_str().unwrap() }))
            .await
            .unwrap_err();
        match err {
            ToolError::InvalidInput(msg) => {
                assert!(msg.contains("is not a directory"), "got: {msg}");
                assert!(!msg.contains("does not exist"), "got: {msg}");
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
        // Must fail before ever reaching the runner.
        assert!(runner.last.lock().unwrap().is_none());
    }

    #[tokio::test]
    async fn carriage_returns_in_output_are_preserved_not_stripped() {
        // Deliberate divergence from pi (which strips every `\r`): `is_keepable` explicitly keeps
        // `\r` alongside `\t`/`\n` so progress-bar-style output (`pip`, `npm`, `curl -#`) survives with
        // its real structure intact rather than being silently collapsed. Pinning this so a future
        // change to `is_keepable` can't accidentally start stripping `\r` without a test noticing.
        let runner = recording(ExecResult {
            code: Some(0),
            stdout: "\x1b[31mred\x1b[0m\r\n".into(),
            ..Default::default()
        });
        let out = Bash::with_runner(runner)
            .run(json!({ "command": "x" }))
            .await
            .unwrap()
            .text;
        assert_eq!(out, "red\r\n");
    }

    #[tokio::test]
    async fn a_multibyte_utf8_character_split_across_stream_chunks_decodes_correctly() {
        // pi: tools.test.ts, "should decode UTF-8 characters split across output chunks". `é` is the
        // 2-byte UTF-8 sequence 0xC3 0xA9 — split it across two separate `on_chunk` deliveries and
        // confirm the final text still decodes correctly (raw bytes accumulate before any UTF-8 decode
        // happens, so this should hold by construction; pinning it so a future rewrite can't regress it).
        let runner = Arc::new(ChunkedRunner {
            chunks: vec![b"h\xC3".to_vec(), b"\xA9llo".to_vec()],
        });
        let out = Bash::with_runner(runner)
            .run(json!({ "command": "x" }))
            .await
            .unwrap()
            .text;
        assert_eq!(out, "héllo");
    }

    #[tokio::test]
    async fn streaming_updates_are_throttled_for_chatty_output() {
        // pi: tools.test.ts, "should coalesce streaming updates for chatty output". 5000 tiny chunks
        // arriving back-to-back (no real time passing between them) must not produce 5000 progress
        // events — `UPDATE_THROTTLE` should bound the emitted count to a small number, plus one final
        // flush after the run completes.
        let chunks: Vec<Vec<u8>> = (0..5000).map(|_| b"x".to_vec()).collect();
        let runner = Arc::new(ChunkedRunner { chunks });

        let (tx, mut rx) = futures::channel::mpsc::unbounded();
        let progress = ToolProgress::new(
            tx,
            "id".into(),
            "bash".into(),
            agent_core::CancellationToken::new(),
        );
        Bash::with_runner(runner)
            .run_streaming(json!({ "command": "x" }), &progress)
            .await
            .unwrap();
        drop(progress);

        let mut count = 0usize;
        while rx.try_recv().is_ok() {
            count += 1;
        }
        assert!(
            count < 100,
            "expected a throttled handful of updates for 5000 rapid chunks, got {count}"
        );
    }

    /// A streaming runner that delivers a fixed sequence of chunks with a real delay between each —
    /// unlike `ChunkedRunner` (which fires every chunk back-to-back with no elapsed time, so
    /// `UPDATE_THROTTLE` never trips before the run completes), this lets a test force a live throttled
    /// emit to land at a chosen point mid-stream.
    struct DelayedChunksRunner {
        chunks: Vec<Vec<u8>>,
        delay: Duration,
    }

    #[async_trait]
    impl CommandRunner for DelayedChunksRunner {
        async fn run(
            &self,
            _program: &str,
            _args: &[String],
            _cwd: Option<&str>,
            _timeout: Duration,
        ) -> std::io::Result<ExecResult> {
            unreachable!("test only exercises the streaming path")
        }

        async fn run_streaming(
            &self,
            _program: &str,
            _args: &[String],
            _cwd: Option<&str>,
            _timeout: Duration,
            on_chunk: ChunkSink<'_>,
        ) -> std::io::Result<ExecResult> {
            for (i, chunk) in self.chunks.iter().enumerate() {
                if i > 0 {
                    tokio::time::sleep(self.delay).await;
                }
                on_chunk(chunk);
            }
            Ok(ExecResult {
                code: Some(0),
                ..Default::default()
            })
        }
    }

    #[tokio::test]
    async fn a_live_progress_emit_between_50kb_and_100kb_carries_a_full_output_path() {
        // pi-parity fix (task 58): pi spills to a temp file at 50KB and forces this on *every* live
        // progress emit, not just the final one. This crate's auto-spill during `append` didn't trip
        // until the output outgrew `rolling_cap` (2x the 50KB display budget, i.e. ~100KB), and the
        // live-emit path passed `persist_if_truncated: false` — so a live snapshot taken while the
        // output sat between 50KB and 100KB reported `truncated: true` with no `full_output_path` to
        // point at, even though the final result would have one. Two chunks with a real delay between
        // them (well past `UPDATE_THROTTLE`) force a live emit to land after the first 60KB chunk
        // (over pi's 50KB mark, under the old ~100KB auto-spill mark) and before the run completes.
        let first = vec![b'a'; 60 * 1024];
        let second = b"tail\n".to_vec();
        let runner = Arc::new(DelayedChunksRunner {
            chunks: vec![first, second],
            delay: UPDATE_THROTTLE * 3,
        });

        let (tx, mut rx) = futures::channel::mpsc::unbounded();
        let progress = ToolProgress::new(
            tx,
            "id".into(),
            "bash".into(),
            agent_core::CancellationToken::new(),
        );
        Bash::with_runner(runner)
            .run_streaming(json!({ "command": "x" }), &progress)
            .await
            .unwrap();
        drop(progress);

        // Extract each update's `full_output_path` (or `None` for the initial content-less update),
        // dropping straight to `Option<String>` since `ToolUpdate` itself carries no `Debug` impl to
        // print on assertion failure.
        let mut full_paths = Vec::new();
        while let Ok(update) = rx.try_recv() {
            let agent_core::ToolUpdate::Progress { details, .. } = update else {
                continue;
            };
            let path = details
                .as_ref()
                .and_then(|d| d.get("full_output_path"))
                .and_then(|p| p.as_str())
                .map(str::to_string);
            full_paths.push(path);
        }
        assert!(
            full_paths.len() >= 2,
            "expected at least one live emit plus the final one, got {}",
            full_paths.len()
        );
        // The *last* update is the final snapshot, which was already forced to persist before this
        // fix — the bug is specifically about every update *before* it.
        let (last, live) = full_paths.split_last().expect("checked non-empty above");
        assert!(
            live.iter().any(Option::is_some),
            "a live (pre-final) progress update in the 50-100KB range must already carry a \
             full_output_path, not just the final one: {full_paths:?}"
        );
        assert!(
            last.is_some(),
            "sanity: the final update must still carry a full_output_path too: {full_paths:?}"
        );
    }

    #[tokio::test]
    async fn cancelling_mid_stream_returns_the_output_already_captured_instead_of_discarding_it() {
        // Pi-parity: pi's `bash-executor.ts` returns whatever output already streamed when a command is
        // cancelled (`BashResult { output, cancelled: true }`), not a bare placeholder. Previously this
        // tool's `exec` had no cancellation awareness of its own — a caller could only cancel by
        // dropping the whole future, which discarded everything captured so far, accumulator included.
        let (tx, rx) = futures::channel::mpsc::unbounded();
        let cancel = agent_core::CancellationToken::new();
        let progress = ToolProgress::new(tx, "id".into(), "bash".into(), cancel.clone());

        let bash = Bash::real();
        let run = bash.run_streaming(
            json!({ "command": "printf 'partial-output\\n'; sleep 30" }),
            &progress,
        );
        tokio::pin!(run);
        // Give the `printf` time to actually run and land in the accumulator before cancelling — the
        // command then goes silent for 30s, so nothing further would ever stream via `ToolProgress`
        // (throttled snapshots only fire on new chunk arrival): this is exactly the case an
        // external-drop-only cancellation would have lost entirely.
        tokio::select! {
            _ = &mut run => panic!("must not complete on its own within this window"),
            () = tokio::time::sleep(std::time::Duration::from_millis(300)) => {}
        }
        cancel.cancel();
        let err = tokio::time::timeout(std::time::Duration::from_secs(5), run)
            .await
            .expect("cancellation must resolve promptly, not wait out the sleep")
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("partial-output"),
            "cancellation must preserve output already captured: {msg:?}"
        );
        assert!(
            msg.to_lowercase().contains("cancel"),
            "cancellation result should say so: {msg:?}"
        );
        drop(rx);
    }

    #[tokio::test]
    async fn a_nonexistent_shell_path_produces_a_clear_spawn_error() {
        // pi: tools.test.ts, "should handle process spawn errors". `with_shell_path` pointed at a
        // genuinely nonexistent binary must surface a clear error, not panic or hang.
        let err = Bash::real()
            .with_shell_path("/definitely/not/a/real/shell/binary")
            .run(json!({ "command": "echo hi" }))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Execution(_)), "got: {err:?}");
    }

    #[tokio::test]
    async fn empty_output_timeout_has_no_placeholder_glued_on() {
        // A silent command that times out should read as just the status line, not
        // "(no output)\n\nCommand timed out after Ns" — the "(no output)" placeholder is for the
        // successful/exit-code paths, where an empty result is a genuinely silent command, not an
        // interrupted one.
        let runner = recording(ExecResult {
            timed_out: true,
            stdout: String::new().into(),
            stderr: String::new().into(),
            ..Default::default()
        });
        let err = Bash::with_runner(runner)
            .run(json!({ "command": "sleep 10", "timeout_ms": 5000 }))
            .await
            .unwrap_err();
        let ToolError::Execution(msg) = err else {
            panic!("expected Execution error")
        };
        assert_eq!(
            msg,
            "Command timed out after 0 seconds: the sandbox's exec endpoint stopped it."
        );
        assert!(!msg.contains("no output"));
    }

    /// Records the `cwd` the runner was actually handed — the one thing `RecordingRunner` throws away,
    /// and the only observable that proves `Bash::root` reached the spawned child.
    struct CwdRecordingRunner {
        cwd: std::sync::Mutex<Option<Option<String>>>,
    }

    #[async_trait]
    impl CommandRunner for CwdRecordingRunner {
        async fn run(
            &self,
            _program: &str,
            _args: &[String],
            cwd: Option<&str>,
            _timeout: Duration,
        ) -> std::io::Result<ExecResult> {
            *self.cwd.lock().unwrap() = Some(cwd.map(str::to_string));
            Ok(ExecResult {
                code: Some(0),
                signal: None,
                stdout: String::new().into(),
                stderr: String::new().into(),
                timed_out: false,
                truncated: false,
            })
        }
    }

    /// Run `input` through a `Bash` rooted at `root` and report the `cwd` its runner received.
    async fn captured_cwd(root: &Path, input: Value) -> Option<String> {
        let runner = Arc::new(CwdRecordingRunner {
            cwd: std::sync::Mutex::new(None),
        });
        Bash::with_runner(runner.clone())
            .with_root(root)
            .run(input)
            .await
            .unwrap();
        let captured = runner.cwd.lock().unwrap().clone();
        captured.expect("runner was never invoked")
    }

    #[tokio::test]
    async fn an_omitted_cwd_defaults_to_the_root() {
        // This is what makes `isolation: worktree` work at all: `bash` reports no `write_target`, so a
        // shared `WriteLockRegistry` can't serialize two children's shell commands. Pointing each
        // child's `bash` at its own worktree is the only thing keeping them apart.
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(
            captured_cwd(tmp.path(), json!({ "command": "echo hi" })).await,
            Some(tmp.path().display().to_string())
        );
    }

    #[tokio::test]
    async fn an_omitted_cwd_with_an_empty_root_still_inherits_the_process_cwd() {
        // The pre-root behavior, unchanged: no root and no `cwd` means the child inherits ours, so
        // `default_registry()` and every existing caller behave exactly as before.
        assert_eq!(
            captured_cwd(Path::new(""), json!({ "command": "echo hi" })).await,
            None
        );
    }

    #[tokio::test]
    async fn a_relative_cwd_resolves_against_the_root() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("sub")).unwrap();
        assert_eq!(
            captured_cwd(tmp.path(), json!({ "command": "echo hi", "cwd": "sub" })).await,
            Some(tmp.path().join("sub").display().to_string())
        );
    }

    #[tokio::test]
    async fn an_absolute_cwd_escapes_the_root_by_design() {
        // A worktree is conflict avoidance, not containment — same contract as Claude Code's
        // `isolation: "worktree"`. A child naming an absolute `cwd` reaches it. Asserted so that if
        // someone later decides to jail this, they do it deliberately and update this test.
        let tmp = tempfile::tempdir().unwrap();
        let elsewhere = tempfile::tempdir().unwrap();
        let abs = elsewhere.path().display().to_string();
        assert_eq!(
            captured_cwd(tmp.path(), json!({ "command": "echo hi", "cwd": &abs })).await,
            Some(abs)
        );
    }

    #[tokio::test]
    async fn a_relative_cwd_that_does_not_exist_under_the_root_is_reported_with_the_resolved_path()
    {
        // The error must name the directory the child would actually have chdir'd into, not the bare
        // relative spelling the model wrote — otherwise "does not exist: sub" is unactionable.
        let tmp = tempfile::tempdir().unwrap();
        let runner = Arc::new(CwdRecordingRunner {
            cwd: std::sync::Mutex::new(None),
        });
        let err = Bash::with_runner(runner)
            .with_root(tmp.path())
            .run(json!({ "command": "echo hi", "cwd": "nope" }))
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains(&tmp.path().join("nope").display().to_string()),
            "{msg}"
        );
        assert!(msg.contains("does not exist"), "{msg}");
    }
}
