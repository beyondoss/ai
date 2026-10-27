//! The agent's side of the exec protocol: per-session targets and the lazily attached cell.
//!
//! The client (`HttpExecRunner`, `TemplateRunner`, the stream and job clients) and the normative
//! protocol spec live in the `beyond-sandbox` crate (`crates/sandbox/src/exec_endpoint.rs`). What
//! stays here is agent policy: which target a session runs on, when it connects, the v1.4 job-end
//! receiver (`tools::job_notify`) and the server-conformance verifier ([`conformance`]).

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use beyond_sandbox::exec::{ChunkSink, CommandRunner, ExecResult};
pub use beyond_sandbox::exec_endpoint::*;

pub mod conformance;
/// When an endpoint calls back about a kept job (v1.4), as one pure function every endpoint shares.
pub use exec_notify as notify_kernel;

// ---------------------------------------------------------------------------------------------
// Per-session targets, for a multi-tenant server.
// ---------------------------------------------------------------------------------------------

/// One configured target: the runner, and the filesystem backend built over it.
///
/// They are constructed together and handed out together because they must always name the *same*
/// machine — a `bash` on one and an `edit` on another is not a partial configuration, it is a bug.
#[derive(Clone)]
pub struct ExecTarget {
    runner: std::sync::Arc<dyn CommandRunner>,
    backend: std::sync::Arc<dyn crate::tools::fs::FsBackend>,
    caps: crate::tools::fs::shell::Capabilities,
    /// Shared with the runner ([`HttpExecRunner::with_suspect_flag`]): raised by a request the
    /// endpoint did not answer or refused. A lazily attached [`ExecCell`] then probes again before
    /// its next call instead of trusting what this target learned.
    suspect: Option<Arc<std::sync::atomic::AtomicBool>>,
}

impl ExecTarget {
    /// Build from an HTTP endpoint plus `Name: value` headers, reading at most `max_response_bytes` of
    /// any one response ([`DEFAULT_MAX_RESPONSE_BYTES`] unless configured).
    pub async fn http(
        url: &str,
        headers: &[String],
        max_response_bytes: usize,
    ) -> Result<Self, String> {
        let mut runner = HttpExecRunner::new(url)?.with_max_response_bytes(max_response_bytes);
        for raw in headers {
            let (name, value) = HttpExecRunner::parse_header(raw)?;
            runner = runner.with_header(name, value);
        }
        Self::over(std::sync::Arc::new(runner)).await
    }

    /// Build from an argv template (`ssh host -- {}`, `docker exec ctr {}`, …).
    pub async fn template(template: &str) -> Result<Self, String> {
        Self::over(std::sync::Arc::new(TemplateRunner::parse(template)?)).await
    }

    /// Wrap an already-built runner, probing it for the capability rung. A probe request that fails
    /// (unreachable, `401`) fails the build with that error.
    pub async fn over(runner: std::sync::Arc<dyn CommandRunner>) -> Result<Self, String> {
        Self::over_with_home(runner, None).await
    }

    /// [`Self::over`], told the target's `$HOME` up front so a model-supplied `~/notes.md` expands
    /// against the sandbox's own home instead of being left alone. Service mode learns the home from
    /// its probe (`crate::service::ServiceSession::connect_exec`), which runs before the first tool
    /// call is dispatched, and is the only caller that can know it then.
    pub async fn over_with_home(
        runner: std::sync::Arc<dyn CommandRunner>,
        home: Option<String>,
    ) -> Result<Self, String> {
        let backend = crate::tools::fs::shell::ShellFs::connect(runner.clone())
            .await
            .map_err(|e| format!("the sandbox did not answer: {e}"))?;
        let backend = match home {
            Some(home) => backend.with_home(home),
            None => backend,
        };
        // The one probe of the sandbox's tools, here so an unreachable or refused endpoint fails the
        // attach with its own error rather than the first tool call.
        let caps = backend
            .capabilities()
            .await
            .map_err(|e| format!("the sandbox did not answer: {e}"))?;
        Ok(Self {
            runner,
            backend: std::sync::Arc::new(backend),
            caps,
            suspect: None,
        })
    }

    /// Watch `flag` — the one its runner raises on a failed request — so a cell holding this target
    /// knows to probe the endpoint again ([`ExecCell::set_lazy`]).
    pub fn watching(mut self, flag: Arc<std::sync::atomic::AtomicBool>) -> Self {
        self.suspect = Some(flag);
        self
    }

    /// Whether a request through this target failed in a way that suggests the sandbox behind it
    /// was replaced (unreachable, or refused).
    fn suspect(&self) -> bool {
        self.suspect
            .as_ref()
            .is_some_and(|f| f.load(std::sync::atomic::Ordering::Relaxed))
    }

    pub fn runner(&self) -> std::sync::Arc<dyn CommandRunner> {
        self.runner.clone()
    }

    pub fn backend(&self) -> std::sync::Arc<dyn crate::tools::fs::FsBackend> {
        self.backend.clone()
    }

    /// Which search rung this target resolved to — reported to the client so a degraded box (no
    /// `rg`, or busybox) is visible at attach rather than inferred from bad results later.
    pub fn search_engine(&self) -> crate::tools::fs::shell::SearchEngine {
        self.caps.search_engine()
    }
}

/// A **per-session** exec target that can be re-pointed at runtime.
///
/// This exists because of a specific hazard. `serve`'s tool registry is rebuilt only when the model
/// or thinking level changes — a `switch_session` between two sessions on the same model does *not*
/// rebuild it. So a target bound at registry-construction time would leave the new session talking to
/// the previous session's machine. In a multi-tenant server that is not a stale-config bug, it is one
/// tenant's tools operating inside another tenant's sandbox.
///
/// The cell closes that by construction: the tools hold *this*, and read it on every call, so a
/// re-point takes effect immediately and cannot be missed by a skipped rebuild. It is the same shape
/// `memory::file::SessionDir` already uses for the per-session `/session` mount, for the same reason.
///
/// An empty cell means the host, so one registry serves both local and remote sessions — unless the
/// cell is [strict](Self::strict), in which case an empty cell is an error and there is no host to
/// fall back to.
#[derive(Clone, Default)]
pub struct ExecCell {
    route: CellRoute,
    /// The session's `bash` jobs. Here because they share the cell's lifetime and reach: one per
    /// session, handed to every registry rebuild (`build_tools` takes the cell), so a `set_model`
    /// keeps them running.
    ///
    /// Deliberately *not* in [`CellRoute`], the part [`Self::runner`] and [`Self::backend`] clone:
    /// every running job's task holds the runner, so a runner that held the table would make a
    /// cycle (task → runner → table → job → task), and a session that ended without stopping its
    /// jobs would leak them, unreachable, for the life of the process.
    bash_jobs: std::sync::Arc<crate::tools::bash_jobs::BashJobs>,
}

/// Where an [`ExecCell`] dispatches: the part its runner and filesystem backend share.
#[derive(Clone, Default)]
struct CellRoute {
    target: std::sync::Arc<std::sync::RwLock<Option<ExecTarget>>>,
    /// How to build the target on first use, when it was attached lazily ([`ExecCell::set_lazy`]).
    /// Kept after the first connect, so a target that turns [suspect](ExecTarget::suspect) is
    /// built again — probe included — rather than trusted.
    lazy: std::sync::Arc<std::sync::RwLock<Option<std::sync::Arc<Lazy>>>>,
    /// Fail closed: an empty cell errors instead of running on this host. Service mode
    /// (`serve --service`) sets it, because there the host is the *replica* — a tenant's `bash`
    /// landing on it is not a stale-config bug but a breach. Fixed at construction.
    strict: bool,
    /// Told of every real use of the sandbox: each command and filesystem call dispatched through
    /// the cell. Never told of a sandbox job operation, which goes to the job store directly (exec
    /// v1.4's live reconcile starts on a use; its own `list` must not count as one).
    on_use: std::sync::Arc<std::sync::RwLock<Option<UseHook>>>,
}

/// What [`ExecCell::set_use_hook`] calls on each real use of the sandbox. Must not block.
pub type UseHook = std::sync::Arc<dyn Fn() + Send + Sync>;

/// Builds a lazily attached target: the probe and everything it derives, run on first use. Its
/// error reaches the tool call that needed the target, word for word.
pub type Connect = std::sync::Arc<
    dyn Fn() -> futures::future::BoxFuture<'static, Result<ExecTarget, String>> + Send + Sync,
>;

/// A lazily attached target, not built yet (or to be built again).
struct Lazy {
    connect: Connect,
    /// The target's `$HOME` as last seen, so a path with a `~` resolves the same way before the
    /// probe as after it ([`CellFs::world`] is not async and cannot wait for one).
    home: Option<String>,
    /// Single flight: concurrent first calls wait on one connect rather than each probing.
    gate: tokio::sync::Mutex<()>,
}

/// What a cell dispatches to right now.
enum Route {
    Target(ExecTarget),
    /// Nothing configured, and the cell is not strict: this host.
    Host,
    /// A lazily attached target not built yet (or suspect): connect before dispatching.
    Pending(std::sync::Arc<Lazy>),
    /// Nothing configured, and the cell is strict.
    Refused,
}

impl CellRoute {
    fn target(&self) -> Option<ExecTarget> {
        self.target
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    fn lazy(&self) -> Option<std::sync::Arc<Lazy>> {
        self.lazy.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Where a call goes, without waiting for anything. A built target wins unless it is suspect
    /// and there is a way to build it again.
    fn route(&self) -> Route {
        match (self.target(), self.lazy()) {
            (Some(t), Some(lazy)) if t.suspect() => Route::Pending(lazy),
            (Some(t), _) => Route::Target(t),
            (None, Some(lazy)) => Route::Pending(lazy),
            (None, None) if self.strict => Route::Refused,
            (None, None) => Route::Host,
        }
    }

    /// The target to dispatch to — connecting a lazily attached one first, once for every caller
    /// waiting on it — or `None` for this host, or the refusal.
    async fn resolve(&self) -> Result<Option<ExecTarget>, String> {
        let hook = self
            .on_use
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        if let Some(hook) = hook {
            hook();
        }
        match self.route() {
            Route::Target(t) => Ok(Some(t)),
            Route::Host => Ok(None),
            Route::Refused => Err(NO_TARGET.to_string()),
            Route::Pending(lazy) => self.connect(&lazy).await.map(Some),
        }
    }

    async fn connect(&self, lazy: &std::sync::Arc<Lazy>) -> Result<ExecTarget, String> {
        let _flight = lazy.gate.lock().await;
        // Whoever held the gate before may have connected already.
        if let Route::Target(t) = self.route() {
            return Ok(t);
        }
        let target = (lazy.connect)().await?;
        // Only if this is still the cell's connector: a re-point while the probe ran wins.
        if self
            .lazy()
            .is_some_and(|c| std::sync::Arc::ptr_eq(&c, lazy))
        {
            *self.target.write().unwrap_or_else(|e| e.into_inner()) = Some(target.clone());
        }
        Ok(target)
    }
}

/// What a strict cell answers when nothing is configured. Static text: it reaches the model as a
/// tool error, so it says what to do, not what went wrong internally.
const NO_TARGET: &str = "no sandbox is attached to this session";

impl ExecCell {
    pub fn new() -> Self {
        Self::default()
    }

    /// A cell with **no host fallback**: until a target is set, every command and every filesystem
    /// call fails, rather than running on this host. Service mode (`serve --service`) uses this,
    /// because there "this host" is the *replica* — a tenant's `bash` landing on it is not a
    /// stale-config bug but a breach.
    pub fn strict() -> Self {
        Self {
            route: CellRoute {
                strict: true,
                ..CellRoute::default()
            },
            ..Self::default()
        }
    }

    /// The session's `bash` job table — see [`crate::tools::bash_jobs`].
    pub fn bash_jobs(&self) -> &std::sync::Arc<crate::tools::bash_jobs::BashJobs> {
        &self.bash_jobs
    }

    /// Re-point at `target`, or back to the host with `None`. Takes effect on the next tool call.
    /// Replaces a lazily attached target too.
    pub fn set(&self, target: Option<ExecTarget>) {
        // Recover a poisoned lock rather than panicking: the cell holds a single value with no
        // invariant a panicked writer could leave half-updated, and the workspace forbids `unwrap`.
        *self.route.lazy.write().unwrap_or_else(|e| e.into_inner()) = None;
        *self.route.target.write().unwrap_or_else(|e| e.into_inner()) = target;
    }

    /// Attach a target **lazily**: nothing is sent until a call needs it, and then `connect` runs
    /// once — concurrent first calls share it — before that call is dispatched. A failure is that
    /// call's error and is not remembered, so the next call tries again. `connect` stays with the
    /// cell: a target whose requests start failing as though its sandbox was replaced (unreachable,
    /// or refused — see [`ExecTarget::watching`]) is built again on the call after.
    ///
    /// `home` is the target's `$HOME` as last seen, answered by the filesystem backend's
    /// [`world`](crate::tools::fs::FsBackend::world) until the probe has run.
    pub fn set_lazy(&self, connect: Connect, home: Option<String>) {
        let lazy = std::sync::Arc::new(Lazy {
            connect,
            home,
            gate: tokio::sync::Mutex::new(()),
        });
        *self.route.target.write().unwrap_or_else(|e| e.into_inner()) = None;
        *self.route.lazy.write().unwrap_or_else(|e| e.into_inner()) = Some(lazy);
    }

    /// Call `hook` on every real use of the sandbox from now on (`None`: stop).
    pub fn set_use_hook(&self, hook: Option<UseHook>) {
        *self.route.on_use.write().unwrap_or_else(|e| e.into_inner()) = hook;
    }

    /// Connect a lazily attached target now, as a call would. A no-op for any other cell.
    pub async fn connect(&self) -> Result<(), String> {
        self.route.resolve().await.map(|_| ())
    }

    /// Whether a lazily attached target is still to be built: nothing has needed it yet, or the
    /// one built turned suspect. A caller that must not wake the target checks this first.
    pub fn pending(&self) -> bool {
        matches!(self.route.route(), Route::Pending(_))
    }

    /// The target, if one is built and usable. Never connects.
    pub fn get(&self) -> Option<ExecTarget> {
        match self.route.route() {
            Route::Target(t) => Some(t),
            _ => None,
        }
    }

    /// A [`CommandRunner`] that always dispatches to whatever this cell currently holds.
    pub fn runner(&self) -> std::sync::Arc<dyn CommandRunner> {
        std::sync::Arc::new(CellRunner(self.route.clone()))
    }

    /// An [`FsBackend`](crate::tools::fs::FsBackend) that always dispatches to whatever this cell
    /// currently holds.
    pub fn backend(&self) -> std::sync::Arc<dyn crate::tools::fs::FsBackend> {
        std::sync::Arc::new(CellFs(self.route.clone()))
    }
}

/// Dispatches each command to the cell's current target, or this host when it is empty.
struct CellRunner(CellRoute);

impl CellRunner {
    async fn target(&self) -> std::io::Result<Option<ExecTarget>> {
        self.0.resolve().await.map_err(std::io::Error::other)
    }
}

#[async_trait]
impl CommandRunner for CellRunner {
    async fn run(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
    ) -> std::io::Result<ExecResult> {
        match self.target().await? {
            Some(t) => t.runner().run(program, args, cwd, timeout).await,
            None => {
                crate::tools::exec::RealRunner
                    .run(program, args, cwd, timeout)
                    .await
            }
        }
    }

    async fn run_streaming(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        on_chunk: ChunkSink<'_>,
    ) -> std::io::Result<ExecResult> {
        match self.target().await? {
            Some(t) => {
                t.runner()
                    .run_streaming(program, args, cwd, timeout, on_chunk)
                    .await
            }
            None => {
                crate::tools::exec::RealRunner
                    .run_streaming(program, args, cwd, timeout, on_chunk)
                    .await
            }
        }
    }

    /// Forwarded, not left to the default: without it a `bash` job behind a cell — every `serve`
    /// session's — reported no [`Kill`], so the exit path could not reach it, a remote job never
    /// yielded, and on this host a leftover `cmd &` went untracked.
    async fn run_streaming_tracked(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        hooks: &crate::tools::exec::JobHooks<'_>,
    ) -> std::io::Result<ExecResult> {
        match self.target().await? {
            Some(t) => {
                t.runner()
                    .run_streaming_tracked(program, args, cwd, hooks)
                    .await
            }
            None => {
                crate::tools::exec::RealRunner
                    .run_streaming_tracked(program, args, cwd, hooks)
                    .await
            }
        }
    }

    /// Forwarded for the same reason: the target behind the cell decides whether `bash` can run a
    /// command in the background. A strict cell with no target answers `false`; running anything
    /// then fails with the cell's own refusal.
    async fn tracks_jobs(&self) -> bool {
        match self.0.resolve().await {
            Ok(Some(t)) => t.runner().tracks_jobs().await,
            Ok(None) => true,
            Err(_) => false,
        }
    }

    async fn run_attached(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        on_stdout: ChunkSink<'_>,
        on_stderr: ChunkSink<'_>,
    ) -> std::io::Result<ExecResult> {
        match self.target().await? {
            Some(t) => {
                t.runner()
                    .run_attached(program, args, cwd, on_stdout, on_stderr)
                    .await
            }
            None => {
                crate::tools::exec::RealRunner
                    .run_attached(program, args, cwd, on_stdout, on_stderr)
                    .await
            }
        }
    }

    async fn attaches(&self) -> bool {
        match self.0.resolve().await {
            Ok(Some(t)) => t.runner().attaches().await,
            Ok(None) => true,
            Err(_) => false,
        }
    }

    /// Connects a lazily attached target, so the synchronous [`Self::sandbox_jobs`] after it sees
    /// one. Its error is the probe's, for the call that needed it.
    async fn ready(&self) -> std::io::Result<()> {
        self.target().await.map(|_| ())
    }

    /// The jobs of the sandbox the cell points at **now**. A job keeps the store it started with,
    /// so a later re-point never sends its kill to another machine. `None` while a lazily attached
    /// target is unbuilt: [`Self::ready`] first.
    fn sandbox_jobs(self: Arc<Self>) -> Option<Arc<dyn jobs::SandboxJobs>> {
        match self.0.route() {
            Route::Target(t) => t.runner().sandbox_jobs(),
            _ => None,
        }
    }

    async fn run_with_stdin(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        stdin: &[u8],
    ) -> std::io::Result<ExecResult> {
        match self.target().await? {
            Some(t) => {
                t.runner()
                    .run_with_stdin(program, args, cwd, timeout, stdin)
                    .await
            }
            None => {
                crate::tools::exec::RealRunner
                    .run_with_stdin(program, args, cwd, timeout, stdin)
                    .await
            }
        }
    }
}

/// The filesystem half of the same indirection.
struct CellFs(CellRoute);

impl CellFs {
    async fn inner(
        &self,
    ) -> Result<std::sync::Arc<dyn crate::tools::fs::FsBackend>, crate::tools::fs::FsError> {
        match self.0.resolve().await {
            Ok(Some(t)) => Ok(t.backend()),
            Ok(None) => Ok(std::sync::Arc::new(crate::tools::fs::local::LocalFs::new())),
            Err(e) => Err(crate::tools::fs::FsError::Backend(e)),
        }
    }
}

#[async_trait]
impl crate::tools::fs::FsBackend for CellFs {
    /// Not fallible, so a strict empty cell answers with the world it *will* have: remote, with no
    /// home known yet. Never `Local` — a tenant's path must not resolve against the replica even in
    /// the window before its sandbox is attached. A lazily attached target that has not been built
    /// answers with the home it was last seen with, so `~` means the same before the probe as after.
    fn world(&self) -> crate::tools::fs::PathWorld {
        match self.0.route() {
            Route::Target(t) => t.backend().world(),
            Route::Host => crate::tools::fs::PathWorld::Local,
            Route::Pending(lazy) => crate::tools::fs::PathWorld::Remote {
                home: lazy.home.clone(),
            },
            Route::Refused => crate::tools::fs::PathWorld::Remote { home: None },
        }
    }
    async fn search(
        &self,
        q: &crate::tools::fs::SearchQuery,
    ) -> Result<crate::tools::fs::SearchOutcome, crate::tools::fs::FsError> {
        self.inner().await?.search(q).await
    }
    async fn stat(
        &self,
        path: &std::path::Path,
    ) -> Result<Option<crate::tools::fs::Meta>, crate::tools::fs::FsError> {
        self.inner().await?.stat(path).await
    }
    async fn read_bytes(
        &self,
        path: &std::path::Path,
        offset: u64,
        max: usize,
    ) -> Result<Vec<u8>, crate::tools::fs::FsError> {
        self.inner().await?.read_bytes(path, offset, max).await
    }
    async fn write_bytes(
        &self,
        path: &std::path::Path,
        bytes: Vec<u8>,
    ) -> Result<(), crate::tools::fs::FsError> {
        self.inner().await?.write_bytes(path, bytes).await
    }
    async fn write_if_unchanged(
        &self,
        path: &std::path::Path,
        bytes: Vec<u8>,
        expected: Option<std::time::SystemTime>,
    ) -> Result<bool, crate::tools::fs::FsError> {
        self.inner()
            .await?
            .write_if_unchanged(path, bytes, expected)
            .await
    }
    async fn create_dir_all(
        &self,
        path: &std::path::Path,
    ) -> Result<(), crate::tools::fs::FsError> {
        self.inner().await?.create_dir_all(path).await
    }
    async fn list_dir(
        &self,
        path: &std::path::Path,
        cap: usize,
        include_hidden: bool,
    ) -> Result<Vec<crate::tools::fs::DirEntry>, crate::tools::fs::FsError> {
        self.inner()
            .await?
            .list_dir(path, cap, include_hidden)
            .await
    }
    async fn glob(
        &self,
        q: &crate::tools::fs::GlobQuery,
    ) -> Result<crate::tools::fs::GlobOutcome, crate::tools::fs::FsError> {
        self.inner().await?.glob(q).await
    }
}

#[cfg(test)]
mod lazy_cell_tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use super::*;
    use crate::tools::fs::PathWorld;

    /// What a counting connector shares with its test.
    #[derive(Default)]
    struct Probe {
        runs: AtomicUsize,
        fail: AtomicBool,
        /// Each built target's suspect flag, in build order.
        flags: Mutex<Vec<Arc<AtomicBool>>>,
    }

    /// A connector that counts its runs, takes a while (so first callers overlap), fails while
    /// `fail` is set, and builds a target over this host whose home is `/sandbox/home`.
    fn counting(probe: Arc<Probe>) -> Connect {
        Arc::new(move || {
            let probe = probe.clone();
            Box::pin(async move {
                probe.runs.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(50)).await;
                if probe.fail.load(Ordering::SeqCst) {
                    return Err("the sandbox did not answer".to_string());
                }
                let flag = Arc::new(AtomicBool::new(false));
                probe.flags.lock().unwrap().push(flag.clone());
                let runner: Arc<dyn CommandRunner> =
                    Arc::new(TemplateRunner::parse("env {}").map_err(|e| e.to_string())?);
                Ok(
                    ExecTarget::over_with_home(runner, Some("/sandbox/home".into()))
                        .await?
                        .watching(flag),
                )
            })
        })
    }

    async fn run_true(cell: &ExecCell) -> std::io::Result<Option<i32>> {
        cell.runner()
            .run("true", &[], None, Duration::from_secs(10))
            .await
            .map(|r| r.code)
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn a_lazy_cell_sends_nothing_until_used_and_concurrent_first_calls_share_one_connect() {
        let probe = Arc::new(Probe::default());
        let cell = ExecCell::strict();
        cell.set_lazy(counting(probe.clone()), Some("/seen/home".into()));
        assert!(cell.pending());
        // The home it was last seen with answers `world` before any probe.
        assert_eq!(
            cell.backend().world(),
            PathWorld::Remote {
                home: Some("/seen/home".into())
            }
        );
        assert_eq!(
            probe.runs.load(Ordering::SeqCst),
            0,
            "attaching sent nothing"
        );

        let calls = (0..8).map(|_| {
            let cell = cell.clone();
            tokio::spawn(async move { run_true(&cell).await })
        });
        for call in futures::future::join_all(calls).await {
            assert_eq!(call.unwrap().unwrap(), Some(0));
        }
        assert_eq!(
            probe.runs.load(Ordering::SeqCst),
            1,
            "one connect for eight first calls"
        );
        assert!(!cell.pending());
        assert_eq!(
            cell.backend().world(),
            PathWorld::Remote {
                home: Some("/sandbox/home".into())
            },
            "the probe's answer replaces the one last seen"
        );
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn a_failed_connect_is_that_calls_error_and_the_next_call_tries_again() {
        let probe = Arc::new(Probe::default());
        probe.fail.store(true, Ordering::SeqCst);
        let cell = ExecCell::strict();
        cell.set_lazy(counting(probe.clone()), None);

        let err = run_true(&cell).await.unwrap_err();
        assert!(
            err.to_string().contains("the sandbox did not answer"),
            "{err}"
        );
        assert!(
            cell.backend()
                .stat(std::path::Path::new("/"))
                .await
                .is_err(),
            "never the host: the filesystem half waits on the same connect"
        );
        assert!(cell.pending(), "a failure is not remembered");

        probe.fail.store(false, Ordering::SeqCst);
        assert_eq!(run_true(&cell).await.unwrap(), Some(0));
        assert_eq!(probe.runs.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn a_suspect_target_is_probed_again_and_an_explicit_target_ends_the_laziness() {
        let probe = Arc::new(Probe::default());
        let cell = ExecCell::strict();
        cell.set_lazy(counting(probe.clone()), None);
        cell.connect().await.unwrap();
        cell.connect().await.unwrap();
        assert_eq!(
            probe.runs.load(Ordering::SeqCst),
            1,
            "a good target is kept"
        );

        // A request was refused or never answered: the sandbox may have been replaced.
        probe.flags.lock().unwrap()[0].store(true, Ordering::SeqCst);
        assert!(cell.pending());
        assert_eq!(run_true(&cell).await.unwrap(), Some(0));
        assert_eq!(probe.runs.load(Ordering::SeqCst), 2, "probed again");
        assert!(!cell.pending());

        // An explicit target (here: none, on a strict cell) replaces the lazy one outright.
        cell.set(None);
        assert!(!cell.pending());
        assert!(run_true(&cell).await.is_err(), "strict and empty: refused");
        assert_eq!(probe.runs.load(Ordering::SeqCst), 2);
    }
}
