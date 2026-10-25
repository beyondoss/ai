//! `Persistence`: a session's store and metadata, and the blocking-pool wrappers around its writes.

use crate::serve::*;

/// Runs [`Persistence::persist`]'s blocking file I/O (`sync_all`, directory `fsync`) on tokio's
/// blocking thread pool instead of the runtime thread the session task is running on. `persist` runs
/// after every turn (and every manual `compact`), so on a slow or network-backed session directory a
/// multi-ms-to-100ms `sync_all` would otherwise pin that thread — and **every other session's task,
/// the accept loop and the idle reaper are on it too**: the process runtime is `current_thread` by
/// default (`main.rs::build_runtime`) and `serve_ws` `tokio::spawn`s every session onto it. Awaiting
/// the blocking task suspends *this* session at the await point either way, so the win is never its
/// own responsiveness — it is that the rest of the replica keeps running while this one writes.
///
/// **The ownership invariant, for anything added here.** There is exactly one [`Persistence`] per
/// session, on that session task's own stack. It is moved **by value** into the blocking task and
/// handed back out, so while the task holds it the session cannot reach it — that ownership *is* the
/// per-session serialization of session writes. So: never `tokio::spawn` a persist (its handle
/// outlives the await, which is exactly what lets a second write start), and never put `Persistence`
/// behind a shared lock. [`SessionStore::append_new`]'s dedup guard is count-keyed
/// (`messages.len() <= self.persisted`), so it cannot see a *shrink* — two writers that desync it
/// silently drop history rather than erroring. `session` is moved in for the same mechanical reason
/// and is cheap to clone at the call site (its `messages` field is an `Arc`, so a caller's clone is a
/// pointer copy, not a deep copy of the transcript).
pub(super) async fn persist_blocking(
    mut persistence: Persistence,
    session: Session,
) -> (Persistence, std::io::Result<()>) {
    // `persist` never panics itself (it handles its own I/O errors internally — see the `Err` arm
    // below for what a caller does with the returned `Result`), and this task is never cancelled
    // (always awaited directly, never `.abort()`ed) — so a `JoinError` here can only mean the closure
    // panicked. Re-raise that panic rather than `.expect()`ing (denied by the workspace's
    // panic-surface lints) on what would otherwise look like an ordinary recoverable error.
    match tokio::task::spawn_blocking(move || {
        let r = persistence.persist(&session);
        (persistence, r)
    })
    .await
    {
        Ok(result) => result,
        Err(e) => std::panic::resume_unwind(e.into_panic()),
    }
}

/// Like [`persist_blocking`], but for one incremental mid-run checkpoint (see [`ChannelCheckpoint`]):
/// takes just the snapshot a checkpoint carries rather than a whole `Session` (see
/// [`Persistence::persist_messages`]).
pub(super) async fn persist_messages_blocking(
    mut persistence: Persistence,
    messages: Arc<Vec<agent_core::Message>>,
    provenance: agent_core::CompactionProvenance,
    edited: Vec<usize>,
    cuts: agent_core::Cuts,
) -> (Persistence, std::io::Result<()>) {
    match tokio::task::spawn_blocking(move || {
        let r = simulated_checkpoint_write(&persistence.meta.id)
            .and_then(|()| persistence.persist_messages(&messages, &provenance, &edited, &cuts));
        (persistence, r)
    })
    .await
    {
        Ok(result) => result,
        Err(e) => std::panic::resume_unwind(e.into_panic()),
    }
}

/// [`persist_blocking`]'s move-in/move-out arrangement applied to [`Persistence::open`] — and this is
/// the one that matters most, because it is the *only* session write path whose cost is unbounded by
/// anything the client did: opening a session resolves a directory and then replays its whole
/// transcript, so it is O(the conversation so far) before the session has served a single command.
///
/// It is also the reconnect path, and reconnects are **correlated**: a deploy or a load-balancer
/// rehash lands N simultaneous opens on one replica at once. Run inline they serialize on the single
/// runtime thread, each session's replay added to every other session's latency and to the accept
/// loop's. On a network-backed session directory (the EFS mount service mode persists to) the
/// directory resolve alone can take tens of milliseconds before any transcript bytes are read.
///
/// `cfg` is moved in and handed back rather than cloned: `open` only borrows it, but the blocking
/// task has to own everything it touches, and the caller needs it for the rest of session start-up.
/// The [`Persistence`] that comes back out obeys the same ownership invariant `persist_blocking`
/// documents — it lands on the session task's stack and stays there.
pub(super) async fn open_persistence_blocking(
    cfg: ServeConfig,
) -> (ServeConfig, std::io::Result<(Persistence, Session)>) {
    match tokio::task::spawn_blocking(move || {
        let r = Persistence::open(&cfg);
        (cfg, r)
    })
    .await
    {
        // `open` reports every failure as an `Err` and this task is never cancelled, so a `JoinError`
        // can only be a panic — same re-raise, for the same reason, as `persist_blocking`'s.
        Ok(result) => result,
        Err(e) => std::panic::resume_unwind(e.into_panic()),
    }
}

/// A [`agent_core::CheckpointHook`] that forwards each mid-run checkpoint's message snapshot through an
/// unbounded channel rather than persisting it directly — sending is instant (never blocks on I/O), so
/// it's safe to call from deep inside `Agent::run_events_steered`'s hot loop without stalling it. The
/// `"prompt"` arm's busy-loop (the only place a checkpoint can fire from) drains the receiving half and
/// does the actual (blocking) append.
pub(super) struct ChannelCheckpoint(pub(super) mpsc::UnboundedSender<Checkpoint>);

/// One mid-run checkpoint on its way to the session loop: the transcript snapshot, its compaction
/// provenance (whose generation tells the store a compaction happened since its last write; see
/// [`SessionStore::persist`]), the messages the last masking pass rewrote in place
/// (`Session::edited`), and the acknowledgement the loop sends once it is on disk.
///
/// The edits have to ride along: each masking pass *replaces* `Session::edited`, so a run that masks
/// twice keeps the first pass's edits only until the second. The checkpoints in between are where
/// they reach disk — as they do in `run` — and without them only the last pass survived the run, and
/// every reopen re-sent the first pass's results unmasked, missing the prompt cache.
///
/// The cuts ride along for the same reason (`Session::cuts`): a checkpoint after the agent popped a
/// reply an earlier checkpoint wrote is where the store learns to rewind past it.
pub(super) type Checkpoint = (
    Arc<Vec<agent_core::Message>>,
    agent_core::CompactionProvenance,
    Vec<usize>,
    agent_core::Cuts,
    tokio::sync::oneshot::Sender<Result<(), String>>,
);

#[async_trait::async_trait]
impl agent_core::CheckpointHook for ChannelCheckpoint {
    /// Hand the snapshot to the session loop and wait until it is **on disk**. The agent awaits a
    /// checkpoint before it dispatches a tool, so that wait is what makes "the `tool_use` is durable
    /// before the tool runs" true: returning as soon as the snapshot was queued let the tool start (and
    /// do whatever it does) while the write was still pending, and a crash in that window lost the
    /// record that the model ever asked for it. Every path that drives the agent drains this channel
    /// while it waits (the prompt loop, a manual `compact`).
    ///
    /// `Err` when the snapshot is **not** on disk — the write failed, the loop is gone, or the snapshot
    /// was swept undrained when a run ended. Before a tool dispatch the agent then fails closed and the
    /// tool never runs (see `agent_core::CheckpointHook`); a write that failed leaves the same hole as
    /// one that never happened.
    async fn checkpoint(&self, session: &Session) -> Result<(), String> {
        let (ack, persisted) = tokio::sync::oneshot::channel();
        if self
            .0
            .send((
                session.messages.clone(),
                session.compaction.clone(),
                session.edited.clone(),
                session.cuts.clone(),
                ack,
            ))
            .is_err()
        {
            return Err("the session loop is gone".into());
        }
        persisted
            .await
            .unwrap_or_else(|_| Err("the checkpoint was dropped unwritten".into()))
    }
}

/// Perform a checkpoint write, then — only then — acknowledge it with the write's result. The one
/// place that ordering lives (both checkpoint arms go through it), so the agent can never be told a
/// checkpoint is durable before the write finished. In the session loop as it stands the order is
/// also forced by the loop itself — the run is not polled while an arm awaits its write — but that
/// is an accident of structure a future change could undo; this is the contract.
pub(super) async fn write_then_ack<T>(
    write: impl std::future::Future<Output = (T, std::io::Result<()>)>,
    ack: tokio::sync::oneshot::Sender<Result<(), String>>,
) -> (T, std::io::Result<()>) {
    let (state, result) = write.await;
    let _ = ack.send(result.as_ref().map(|_| ()).map_err(|e| e.to_string()));
    (state, result)
}

/// A slow or failing disk under a checkpoint write, simulated, **inside** the write itself: sleeps
/// `BEYOND_AI_AGENT_TEST_SLOW_CHECKPOINT_MS` milliseconds, then fails the write when
/// `BEYOND_AI_AGENT_TEST_FAIL_CHECKPOINT` says so — `all` fails every checkpoint write, a number `n`
/// only the `n`th (1-based) **of each session**, counted by session id, so on a daemon one session's
/// writes never shift which write of another's fails. Inside the write, not before it, so an
/// acknowledgement sent ahead of the write would precede the slow part — which is the mistake the
/// slow-disk test is there to catch. `#[cfg]`-gated like [`simulated_slow_open`], names and all: a
/// release binary has no fault-injection path and carries none of these variable names.
#[cfg(debug_assertions)]
pub(super) fn simulated_checkpoint_write(session: &str) -> std::io::Result<()> {
    if let Ok(ms) = std::env::var("BEYOND_AI_AGENT_TEST_SLOW_CHECKPOINT_MS")
        && let Ok(ms) = ms.parse::<u64>()
    {
        std::thread::sleep(std::time::Duration::from_millis(ms));
    }
    static WRITES: std::sync::Mutex<std::collections::BTreeMap<String, usize>> =
        std::sync::Mutex::new(std::collections::BTreeMap::new());
    let n = {
        let mut writes = WRITES
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let n = writes.entry(session.to_owned()).or_default();
        *n += 1;
        *n
    };
    let fail = match std::env::var("BEYOND_AI_AGENT_TEST_FAIL_CHECKPOINT").as_deref() {
        Ok("all") => true,
        Ok(which) => which.parse::<usize>().is_ok_and(|which| which == n),
        Err(_) => false,
    };
    if fail {
        return Err(std::io::Error::other("simulated checkpoint write failure"));
    }
    Ok(())
}

#[cfg(not(debug_assertions))]
pub(super) fn simulated_checkpoint_write(_session: &str) -> std::io::Result<()> {
    Ok(())
}

/// Mint a brand-new `SessionMeta`, using `session_id` in place of a freshly generated one when given
/// (`ServeConfig::session_id`, already validated by `main.rs` before this is ever called).
pub(super) fn fresh_meta(cwd: &str, model: &str, session_id: Option<&str>) -> SessionMeta {
    match session_id {
        Some(id) => SessionMeta::with_id(id.to_string(), cwd, model),
        None => SessionMeta::new(cwd, model),
    }
}

/// How [`Persistence`] picks which session to open in repo mode. Spelled out as a choice rather than
/// inferred from a pile of `Option`s at the point of use, because the precedence between these is the
/// whole contract: an explicit id is the most specific selector and always wins, `--continue` is the
/// only thing that reattaches, and everything else starts clean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SessionSelect<'a> {
    /// Address one specific session: open it, or create it under exactly this id
    /// ([`SessionRepo::open_or_create_id`]). `--session-id`, and the daemon's `?session_id=`.
    Id(&'a str),
    /// `--continue`: this cwd's most recent session, else a fresh one.
    Latest,
    /// The default: always a brand-new session.
    Fresh,
}

impl<'a> SessionSelect<'a> {
    /// Resolve the selector from config. An explicit id beats `--continue` because it names exactly one
    /// session while `--continue` only describes one, and a caller who supplied both has already been
    /// more specific than "whatever ran here last".
    pub(super) fn from_cfg(cfg: &'a ServeConfig) -> Self {
        match (&cfg.session_id, cfg.continue_session) {
            (Some(id), _) => Self::Id(id),
            (None, true) => Self::Latest,
            (None, false) => Self::Fresh,
        }
    }
}

/// The sibling session `new_session` archived a pinned slot's conversation into.
pub(super) struct Archived {
    pub(super) id: String,
    pub(super) title: Option<String>,
}

/// Where the server persists sessions: a multi-session [`SessionRepo`] (`--session-dir`), a single
/// JSONL file (`--session-file`), or nowhere (in-memory). It always carries the current session's
/// [`SessionMeta`] so the session id is stable across reattaches.
pub(super) struct Persistence {
    pub(super) repo: Option<SessionRepo>,
    pub(super) store: Option<SessionStore>,
    /// The writer lock on `store`'s session (see [`crate::session_store::lock_session_writer`]):
    /// taken before the store's first write, replaced with every store swap, so no other process can
    /// write the session this one is writing. `None` with no store, and for the segmented layout.
    pub(super) lock: Option<crate::session_store::SessionLock>,
    /// Whether this process takes writer locks at all: stdio `serve` does, as `run` does. A daemon
    /// session (`ServeConfig::supervised`) does not — its sessions share one process, where the
    /// supervisor's id table already keeps one task per session, and service mode locks in
    /// `serve_ws` itself.
    pub(super) writer_lock: bool,
    pub(super) meta: SessionMeta,
    /// Set when the caller *named* this session ([`ServeConfig::session_id`]), which makes the id a
    /// routing key rather than an incidental label — the daemon reconnects `?session_id=<id>` to it, and
    /// a supervised `serve --session-id` expects the same id back after a restart. [`Self::new_session`]
    /// honors it by blanking the session in place instead of minting a new id, so the address a client
    /// holds never goes stale underneath it. `None` leaves `new_session` free to mint (repo mode's
    /// natural behavior).
    pub(super) pinned_id: Option<String>,
    /// Bumped by every command that replaces the active transcript (`new_session`, `switch`,
    /// `fork`) — see [`Self::transcript`].
    pub(super) transcript: u64,
}

/// The latency a network-backed session directory adds to [`Persistence::open`], simulated: sleeps
/// `BEYOND_AI_AGENT_TEST_SLOW_SESSION_OPEN_MS` milliseconds if that variable is set.
///
/// A seam, because the property worth pinning here is not how long an open takes but *where* it
/// takes it: `tests/serve_startup_stall.rs` stalls one session's open for seconds and asserts that a
/// different session's RPC still answers in milliseconds — which is only a meaningful assertion if
/// the stall is large and deterministic, and a real transcript big enough to cost seconds is neither
/// cheap to build nor stable across machines. `#[cfg]`-gated rather than merely undocumented, so a
/// release binary contains no latency-injection path at all.
#[cfg(debug_assertions)]
pub(super) fn simulated_slow_open() {
    if let Ok(ms) = std::env::var("BEYOND_AI_AGENT_TEST_SLOW_SESSION_OPEN_MS")
        && let Ok(ms) = ms.parse::<u64>()
    {
        std::thread::sleep(std::time::Duration::from_millis(ms));
    }
}

#[cfg(not(debug_assertions))]
pub(super) fn simulated_slow_open() {}

/// Latency before a session installs its elicitation and sampling gates, simulated (debug builds
/// only, like [`simulated_slow_open`]): holds open the window in which a server could ask a question
/// before the session can take it, so a test can show nothing is asked in it.
#[cfg(debug_assertions)]
pub(super) async fn simulated_slow_gate_install() {
    if let Ok(ms) = std::env::var("BEYOND_AI_AGENT_TEST_SLOW_GATE_INSTALL_MS")
        && let Ok(ms) = ms.parse::<u64>()
    {
        tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
    }
}

#[cfg(not(debug_assertions))]
pub(super) async fn simulated_slow_gate_install() {}

impl Persistence {
    /// Hands the live persistence to a blocking-pool write and leaves an inert stand-in behind until
    /// the write returns it (`*self = returned`). Nothing reads the stand-in: the session task is
    /// parked on that very write.
    pub(super) fn take_for_blocking(&mut self) -> Self {
        std::mem::replace(
            self,
            Self {
                repo: None,
                store: None,
                lock: None,
                writer_lock: false,
                meta: fresh_meta("", "", None),
                pinned_id: None,
                transcript: 0,
            },
        )
    }

    /// Open persistence and select the active session: in repo mode per [`SessionSelect`], in file mode
    /// by opening the named file (or creating it).
    ///
    /// Synchronous, and deliberately so — it is the thing [`open_persistence_blocking`] wraps, not the
    /// wrapper. Every caller on a session task must go through that.
    pub(super) fn open(cfg: &ServeConfig) -> std::io::Result<(Self, Session)> {
        simulated_slow_open();
        // Service mode first, and with no fallback: the tenant's own directory or nothing. A
        // `service_mode` config that reached a session without a grant would otherwise land in the
        // per-cwd default repo below — the replica's own disk, shared across tenants.
        let mode = Mode::for_persistence(cfg)?;
        if let Some((session_dir, workspace_root, repo_options)) = mode.tenant_storage() {
            // Epoch segments, sealed with the tenant's own keys, and minting derived ids onto this
            // session's shard: the layout a mount shared by many replicas needs (see
            // `session_store`'s "Ownership: the epoch fence").
            return Self::open_repo_with(
                session_dir,
                workspace_root,
                &cfg.model,
                SessionSelect::from_cfg(cfg),
                repo_options,
                false,
            );
        }
        let writer_lock = !cfg.supervised;
        let cwd = crate::session_store::canonical_cwd(&std::env::current_dir().unwrap_or_default())
            .to_string_lossy()
            .into_owned();
        if let Some(dir) = &cfg.session_dir {
            return Self::open_repo_with(
                dir,
                &cwd,
                &cfg.model,
                SessionSelect::from_cfg(cfg),
                crate::session_store::RepoOptions::default(),
                writer_lock,
            );
        }
        if let Some(path) = &cfg.session_file {
            let path = std::path::PathBuf::from(path);
            // A zero-byte file at `path` (e.g. `touch`'d ahead of time, or left over from a crash
            // before the header write landed) has nothing to open — route it through `create`, which
            // now initializes an empty file in place rather than failing (see its own doc comment).
            let has_content = path.metadata().is_ok_and(|m| m.len() > 0);
            let (store, session) = if has_content {
                SessionStore::open(path)?
            } else {
                let meta = fresh_meta(&cwd, &cfg.model, cfg.session_id.as_deref());
                let store = SessionStore::create(path, meta)?;
                (store, Session::new())
            };
            let lock = if writer_lock {
                crate::session_store::lock_session_writer(store.path())?
            } else {
                None
            };
            let meta = store.meta().clone();
            return Ok((
                Self {
                    repo: None,
                    store: Some(store),
                    lock,
                    writer_lock,
                    meta,
                    // Recorded for consistency, but single-file mode never consults it: the operator
                    // named the *file*, and `new_session`'s file-mode arm already resets in place, so
                    // the id can't move regardless of whether one was pinned.
                    pinned_id: cfg.session_id.clone(),
                    transcript: 0,
                },
                session,
            ));
        }
        if cfg.no_session_persistence {
            return Ok((
                Self {
                    repo: None,
                    store: None,
                    lock: None,
                    writer_lock,
                    meta: fresh_meta(&cwd, &cfg.model, cfg.session_id.as_deref()),
                    pinned_id: cfg.session_id.clone(),
                    transcript: 0,
                },
                Session::new(),
            ));
        }
        // Neither flag was given and persistence wasn't explicitly opted out: default to a per-cwd
        // repo directory rather than silently running in-memory-only, so an operator who simply
        // forgot the flag doesn't lose all history on the next restart with no indication why.
        Self::open_repo_with(
            crate::session_store::default_session_dir(&cwd),
            &cwd,
            &cfg.model,
            SessionSelect::from_cfg(cfg),
            crate::session_store::RepoOptions::default(),
            writer_lock,
        )
    }

    /// Open (creating if needed) a multi-session repo at `dir` and pick a session out of it per
    /// `select` — see [`SessionSelect`] for the precedence, and the two `SessionRepo` methods it
    /// dispatches to for what each one guarantees.
    ///
    /// Note that [`SessionSelect::Fresh`] (the default) genuinely creates a new session on every start.
    /// A supervised `serve` that wants to pick its conversation back up across restarts should pin
    /// `--session-id`: that's deterministic and idempotent, whereas `--continue`'s "most recent for this
    /// cwd" silently depends on what else has touched the directory since.
    ///
    /// The tests' shorthand for [`Self::open_repo_with`] with the default layout and writer locks
    /// taken, as stdio `serve` takes them; [`Self::open`] calls that directly.
    #[cfg(test)]
    pub(super) fn open_repo(
        dir: impl Into<std::path::PathBuf>,
        cwd: &str,
        model: &str,
        select: SessionSelect<'_>,
    ) -> std::io::Result<(Self, Session)> {
        Self::open_repo_with(
            dir,
            cwd,
            model,
            select,
            crate::session_store::RepoOptions::default(),
            true,
        )
    }

    /// [`Self::open_repo`] with an explicit on-disk shape — the service-mode entry point, where the
    /// repo is segmented, sealed per tenant, and mints derived ids onto the session's own shard — and
    /// whether to take writer locks (see [`Self::writer_lock`]).
    pub(super) fn open_repo_with(
        dir: impl Into<std::path::PathBuf>,
        cwd: &str,
        model: &str,
        select: SessionSelect<'_>,
        opts: crate::session_store::RepoOptions,
        writer_lock: bool,
    ) -> std::io::Result<(Self, Session)> {
        let repo = SessionRepo::open_with(dir, opts)?;
        let (store, session) = match select {
            SessionSelect::Id(id) => repo.open_or_create_id(id, cwd, model)?,
            SessionSelect::Latest => repo.resume_latest_or_create(cwd, model)?,
            SessionSelect::Fresh => (repo.create(SessionMeta::new(cwd, model))?, Session::new()),
        };
        let lock = if writer_lock {
            crate::session_store::lock_session_writer(store.path())?
        } else {
            None
        };
        let meta = store.meta().clone();
        Ok((
            Self {
                lock,
                writer_lock,
                repo: Some(repo),
                // Only an explicitly addressed session is pinned. `Latest`/`Fresh` produce an id the
                // caller never asked for, so nothing is routing on it and `new_session` may mint freely.
                pinned_id: match select {
                    SessionSelect::Id(id) => Some(id.to_string()),
                    SessionSelect::Latest | SessionSelect::Fresh => None,
                },
                store: Some(store),
                meta,
                transcript: 0,
            },
            session,
        ))
    }

    pub(super) fn session_id(&self) -> &str {
        &self.meta.id
    }

    /// Which transcript is active: changes whenever a command replaces it, including the ones that
    /// keep the session id (`new_session` on a pinned session, which every daemon session is, and in
    /// single-file mode). What the active transcript's own state — its `bash` jobs — is keyed on;
    /// the id alone cannot tell a blanked session from the one it replaced.
    pub(super) fn transcript(&self) -> u64 {
        self.transcript
    }

    /// The active session's on-disk JSONL path — pi's `sessionFile` — or `None` when persistence is
    /// disabled entirely (`--no-session-persistence`, an in-memory-only run). Repo and single-file
    /// modes both populate `store`, so this covers either without needing to distinguish them.
    pub(super) fn session_file(&self) -> Option<&Path> {
        self.store.as_ref().map(SessionStore::path)
    }

    /// Where this session's `/session` working memory belongs. Always asked of the store, which is
    /// the only thing that knows the layout: deriving it from the path would hand every session on a
    /// shard one shared `<shard>.memory` directory (see [`crate::memory::session_dir`]). Without a
    /// store there is no session file to sit beside, so it falls back to the id-keyed tempdir.
    pub(super) fn memory_dir(&self) -> std::path::PathBuf {
        match &self.store {
            Some(store) => store.memory_dir(),
            None => crate::memory::session_dir(None, self.session_id()),
        }
    }

    /// Whether this session's storage has been taken over by another owner — a newer epoch on a
    /// shared mount. Once true nothing this process writes can ever land again, so the session ends
    /// rather than accepting turns it cannot persist.
    pub(super) fn superseded(&self) -> bool {
        self.store.as_ref().is_some_and(SessionStore::superseded)
    }

    /// Persist the session after a turn ([`SessionStore::persist`]): a compaction not yet on disk is
    /// written as one, non-destructively; then any tool results a masking pass rewrote in place; then
    /// the new messages. Mid-run checkpoints skip the masked results (they carry no `edited` list),
    /// so a masking pass reaches disk here, at the end of the turn; a crash before then reloads the
    /// unmasked transcript, which is valid, just larger.
    pub(super) fn persist(&mut self, session: &Session) -> std::io::Result<()> {
        let Some(store) = &mut self.store else {
            return Ok(());
        };
        let r = store.persist(
            &session.messages,
            &session.compaction,
            &session.edited,
            &session.cuts,
        );
        if let Err(e) = &r {
            eprintln!("serve: failed to persist session: {e}");
        }
        r
    }

    /// Incremental persist for a mid-run checkpoint (see [`ChannelCheckpoint`]). Keyed on the
    /// compaction generation, not just the message count: a checkpoint right after a mid-run
    /// compaction (`Agent::compact` takes one) writes the rewrite then and there, so a later
    /// checkpoint, once the list outgrows its old length, can never chain the post-compaction tail
    /// onto the uncompacted history — what a crash before the end of the turn used to leave behind.
    /// `edited` is the masking pass's in-place rewrites (see [`Checkpoint`]).
    pub(super) fn persist_messages(
        &mut self,
        messages: &[agent_core::Message],
        provenance: &agent_core::CompactionProvenance,
        edited: &[usize],
        cuts: &agent_core::Cuts,
    ) -> std::io::Result<()> {
        let Some(store) = &mut self.store else {
            return Ok(());
        };
        let r = store.persist(messages, provenance, edited, cuts);
        if let Err(e) = &r {
            eprintln!("serve: failed to persist checkpoint: {e}");
        }
        r
    }

    /// The working directory, for new session metadata — canonicalized, so a session created via
    /// `new_session` matches the same [`canonical_cwd`](crate::session_store::canonical_cwd) form
    /// every other session-cwd is recorded in.
    pub(super) fn cwd() -> String {
        crate::session_store::canonical_cwd(&std::env::current_dir().unwrap_or_default())
            .to_string_lossy()
            .into_owned()
    }

    /// Start a fresh session. In repo mode this creates a new file (new id); in single-file mode it
    /// resets the existing file (keeping its id); in-memory it just mints new metadata.
    ///
    /// **Pinned sessions are the exception** ([`Self::pinned_id`]): when the caller named this session,
    /// that name is an address something is routing on, so `new_session` keeps it and blanks the session
    /// in place. The outgoing conversation is *snapshotted into its own session first* — a real, listable
    /// sibling whose `parent` points back here — so "start a new session" never doubles as "destroy the
    /// old one". That archive-then-blank order matters: the copy is taken from what's on disk, so a
    /// failure to archive aborts before anything is cleared.
    ///
    /// In repo mode, the fresh session's `parent` records whatever session id was active immediately
    /// before this call, unless `parent_session` explicitly names a different one — pi's own
    /// `parentSession` lineage marker on a `/new`-equivalent reset (pi's own default, absent an
    /// explicit override, is no parent at all; this crate's default deliberately differs, always
    /// linking to the previously-active session, since that provenance is used elsewhere — e.g. tree
    /// navigation in the HTML export). It's provenance only (no shared history, unlike an actual
    /// `fork`): a client browsing sessions can still trace "this fresh one followed that one" in the
    /// same directory/conversation, or — with an explicit `parent_session` — "this one continues a
    /// *different* lineage than whatever happened to be active." Not recorded in single-file mode
    /// (there's no *new* id to link — the file/id just gets cleared in place) or in-memory mode
    /// (nothing persists a "previous" id anywhere a client could look it up later), matching pi's own
    /// `this.persist ? previousSessionFile : undefined`.
    ///
    /// The second half of the result is the sibling a pinned session archived into, if it did.
    ///
    /// On failure, nothing is mutated (no partial state) — the caller keeps whatever session was
    /// already active, matching the source of truth still on disk. This matters: silently returning a
    /// fresh in-memory `Session` while the on-disk reset actually failed would desync `SessionStore`'s
    /// persisted-message-count bookkeeping from what the caller believes is live, so a *subsequent*
    /// successful `persist` could look like a no-op append and silently drop every message of the
    /// "new" session (`SessionStore::append_new`'s own dedup guard, `messages.len() <= self.persisted`).
    pub(super) fn new_session(
        &mut self,
        model: &str,
        parent_session: Option<&str>,
    ) -> std::io::Result<(Session, Option<Archived>)> {
        let cwd = Self::cwd();
        if let Some(repo) = &self.repo {
            if let Some(pinned) = self.pinned_id.clone() {
                let blanked = self.blank_pinned_session(&pinned);
                if blanked.is_ok() {
                    self.transcript += 1;
                }
                return blanked;
            }
            let mut meta = SessionMeta::new(&cwd, model);
            meta.parent = Some(parent_session.unwrap_or(&self.meta.id).to_string());
            match repo.create(meta).and_then(|store| self.install(store)) {
                Ok(()) => {}
                Err(e) => {
                    eprintln!("serve: failed to create session: {e}");
                    return Err(e);
                }
            }
        } else if let Some(store) = &mut self.store {
            if let Err(e) = store.reset_for_new_session() {
                eprintln!("serve: failed to reset session: {e}");
                return Err(e);
            }
        } else {
            self.meta = SessionMeta::new(&cwd, model);
        }
        self.transcript += 1;
        Ok((Session::new(), None))
    }

    /// [`Self::new_session`] for a pinned (caller-addressed) session: archive the current conversation
    /// into a session of its own, then clear this one in place, keeping the id.
    ///
    /// The archive is a plain [`SessionRepo::fork`] of the whole active transcript, so it lands as an
    /// ordinary session — it appears in `list_sessions`, can be `switch_session`ed into, and records
    /// `parent = <pinned id>`, leaving the lineage back to this slot legible. An empty session has
    /// nothing worth archiving, so that's skipped and blanking an already-blank slot stays cheap.
    pub(super) fn blank_pinned_session(
        &mut self,
        pinned: &str,
    ) -> std::io::Result<(Session, Option<Archived>)> {
        let Self {
            repo: Some(repo),
            store: Some(store),
            ..
        } = self
        else {
            // Not repo mode with a live store: single-file and in-memory both keep their id anyway, so
            // the caller's own arms already do the right thing.
            return Ok((Session::new(), None));
        };
        let mut archived = None;
        // The slot's own title, read before anything is cleared: it is the archived conversation's.
        let title = store.meta().title.clone();
        if !store.active_ids().is_empty() {
            // Copied from what's on disk, *before* anything is cleared — a failure here leaves the
            // session exactly as it was rather than half-reset with its history already gone.
            match repo.fork(pinned, usize::MAX) {
                Ok((sibling, _)) => {
                    archived = Some(Archived {
                        id: sibling.meta().id.clone(),
                        title,
                    });
                }
                Err(e) => {
                    eprintln!("serve: failed to archive session {pinned} before reset: {e}");
                    return Err(e);
                }
            }
        }
        if let Err(e) = store.reset_for_new_session() {
            eprintln!("serve: failed to reset session: {e}");
            return Err(e);
        }
        let meta = store.meta().clone();
        self.meta = meta;
        Ok((Session::new(), archived))
    }

    /// Switch to another session by id (repo mode only).
    pub(super) fn switch(&mut self, id: &str) -> std::io::Result<Session> {
        let repo = self.repo.as_ref().ok_or_else(not_in_repo_mode)?;
        let (store, session) = repo.open_id(id)?;
        self.install(store)?;
        self.transcript += 1;
        Ok(session)
    }

    /// Make `store` the active session, taking its writer lock first (and releasing the old one): a
    /// session another process holds is refused, and nothing here changes. The same session reopened
    /// keeps the lock already held — a second acquire in this process would be refused.
    pub(super) fn install(&mut self, store: SessionStore) -> std::io::Result<()> {
        let same = self
            .store
            .as_ref()
            .is_some_and(|s| s.path() == store.path());
        let lock = if same {
            self.lock.take()
        } else if self.writer_lock {
            crate::session_store::lock_session_writer(store.path())?
        } else {
            None
        };
        self.meta = store.meta().clone();
        self.store = Some(store);
        self.lock = lock;
        Ok(())
    }

    /// Fork the current session into a new session and switch to it (repo mode). `entry_id`, when
    /// given, forks at that specific tree entry — anywhere in the whole tree, on or off the active
    /// path (`before` excludes the entry itself, matching pi's `position:"before"`/`"at"`) — otherwise
    /// falls back to `upto`, a message-count prefix of the active path (the original, narrower form).
    ///
    /// Also returns the model/thinking-level that was actually active at the forked-from point (Task #2,
    /// pi-parity fix) — resolved against *this* session's own tree (see
    /// [`Self::fork_target_model_and_level`]) *before* the swap below, since `SessionRepo::fork`/
    /// `fork_at_entry` deliberately don't carry `ModelChange`/`ThinkingLevelChange` bookkeeping into the
    /// freshly forked session's own file (see their own doc comments — forking only ever preserves
    /// message content, never the surrounding per-branch bookkeeping). Concretely: a session that
    /// switched models mid-run, then forked back to an entry on the *old* model's branch, previously kept
    /// running on whatever the process was last set to — silently replaying that old provider's signed
    /// `Thinking` blocks to a foreign model on the fork's very next turn. The caller (the `fork`/`clone`
    /// RPC handlers) applies this exactly like `switch_session`/`switch_branch` already do: clamp,
    /// `scrub_cross_model_state` when the model actually changes, and rebuild `agent` only if needed.
    pub(super) fn fork(
        &mut self,
        upto: usize,
        entry_id: Option<&str>,
        before: bool,
        starting_level: agent_core::ThinkingLevel,
    ) -> std::io::Result<(Session, String, agent_core::ThinkingLevel)> {
        let id = self.meta.id.clone();
        let repo = self.repo.as_ref().ok_or_else(not_in_repo_mode)?;
        let (restored_model, restored_level) =
            self.fork_target_model_and_level(upto, entry_id, before, starting_level);
        let (store, session) = match entry_id {
            Some(entry_id) => repo.fork_at_entry(&id, entry_id, before)?,
            None => repo.fork(&id, upto)?,
        };
        self.install(store)?;
        self.transcript += 1;
        Ok((session, restored_model, restored_level))
    }

    /// Fork the current session and **stay on it** — service mode's shape for `fork`/`clone`.
    ///
    /// A daemon session is addressed by its id: the supervisor's slot, the session's advisory lock,
    /// and the grant every attached connection presents are all keyed on it. Swapping `self.store`
    /// onto a different session would leave this task writing a session nothing routes to, while
    /// holding the lock on the one it left. So the derived id — `<shard>.<opaque>`, minted by the
    /// repo — goes back to the client instead, which opens it with a grant of its own.
    ///
    /// Reads the source from disk, like [`SessionRepo::fork`](crate::session_store::SessionRepo::fork)
    /// always has, which is why service mode answers a `fork` arriving mid-run with "busy" rather
    /// than copying a prefix the running turn hasn't persisted yet.
    pub(super) fn fork_detached(
        &self,
        upto: usize,
        entry_id: Option<&str>,
        before: bool,
    ) -> std::io::Result<String> {
        let repo = self.repo.as_ref().ok_or_else(not_in_repo_mode)?;
        let (store, _session) = match entry_id {
            Some(entry_id) => repo.fork_at_entry(&self.meta.id, entry_id, before)?,
            None => repo.fork(&self.meta.id, upto)?,
        };
        Ok(store.meta().id.clone())
    }

    /// The model/thinking-level actually active at what a would-be [`Self::fork`] targets — see that
    /// method's own doc comment for why this must be computed against the *source* tree before the fork
    /// swaps `self.store` to the freshly created one. Mirrors `fork`'s own `entry_id`/`before`-vs-`upto`
    /// resolution exactly, so the point this reads model/level at is the same one the copied prefix
    /// actually ends on.
    pub(super) fn fork_target_model_and_level(
        &self,
        upto: usize,
        entry_id: Option<&str>,
        before: bool,
        process_starting_level: agent_core::ThinkingLevel,
    ) -> (String, agent_core::ThinkingLevel) {
        match entry_id {
            Some(entry_id) => {
                // Same `before` resolution as `switch_branch`: `parent_of` returns `None` only for an
                // unknown id, in which case there's nothing meaningful to resolve here anyway — the
                // actual fork call just below fails with a clear `NotFound` and this fallback value is
                // never returned to a caller.
                let target: Option<String> = if before {
                    self.store
                        .as_ref()
                        .and_then(|s| s.parent_of(entry_id))
                        .unwrap_or(None)
                } else {
                    Some(entry_id.to_string())
                };
                self.model_and_level_at_opt(target.as_deref(), process_starting_level)
            }
            None => {
                let active = self.active_ids();
                let upto = upto.min(active.len());
                match upto {
                    0 => self.model_and_level_at_opt(None, process_starting_level),
                    n => self.model_and_level_at(&active[n - 1], process_starting_level),
                }
            }
        }
    }

    /// Preview what forking `session_id` would produce — the exact prefix `fork` would copy — without
    /// creating a new session file or switching to it (repo mode only, like `switch`/`fork`). Same
    /// `entry_id`/`before`-vs-`upto` choice as `fork`. A client browsing `list_sessions` uses this to
    /// preview a fork point before committing to it.
    pub(super) fn fork_messages(
        &self,
        mode: &Mode,
        session_id: &str,
        upto: usize,
        entry_id: Option<&str>,
        before: bool,
    ) -> std::io::Result<Vec<agent_core::Message>> {
        let repo = self.repo_for(mode, session_id)?;
        Self::require_exactly(mode, &repo, session_id)?;
        if let Some(entry_id) = entry_id {
            return repo.fork_at_entry_messages(session_id, entry_id, before);
        }
        let (_, session) = repo.open_id(session_id)?;
        let upto = upto.min(session.messages.len());
        Ok(session.messages[..upto].to_vec())
    }

    /// The repo a command naming *another* session acts on.
    ///
    /// Outside service mode that is always this session's own. In service mode a session id carries
    /// its shard, so it is `<that id's shard>/<tenant>/sessions`: one tenant's sessions can live on
    /// several mounts, and a command must reach the one it actually named rather than quietly
    /// operating on whichever mount this connection happened to land on. A shard this replica
    /// doesn't serve is an error, not an empty result.
    pub(super) fn repo_for(&self, mode: &Mode, id: &str) -> std::io::Result<SessionRepo> {
        match mode.tenant_repo_for(id) {
            Some(repo) => repo,
            None => self.repo.clone().ok_or_else(not_in_repo_mode),
        }
    }

    /// Service mode only: refuse an id that doesn't name a session **exactly**.
    ///
    /// [`SessionRepo::find_path`](crate::session_store::SessionRepo::find_path)'s unique-prefix
    /// fallback is a convenience for an id a human typed. A tenant's ids are minted, never typed, so
    /// there a prefix match turns "that session is gone" into "here is a neighbour of it" — and, for
    /// `delete`, turns a miss into a silent `Ok`. Exact or error.
    pub(super) fn require_exactly(
        mode: &Mode,
        repo: &SessionRepo,
        id: &str,
    ) -> std::io::Result<()> {
        if !mode.requires_exact_ids() {
            return Ok(());
        }
        if repo.has_exact(id)? {
            Ok(())
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("no session {id}"),
            ))
        }
    }

    /// Delete a session by id (repo mode only) — [`SessionRepo::delete`](crate::session_store::SessionRepo::delete)'s
    /// soft-delete-to-`.trash`, idempotent whether or not `id` still exists. Refuses to delete the
    /// *currently active* session (the one this `Persistence` itself is pointed at): that would move the
    /// file out from under an in-memory `Session` a client is still actively using — with no session
    /// file left to persist the next turn into — a footgun no legitimate caller needs, since `list_
    /// sessions`/`list_all_sessions` responses report which id is current. A client that genuinely wants
    /// that must `new_session`/`switch_session` away first.
    pub(super) fn delete(&self, mode: &Mode, id: &str) -> std::io::Result<()> {
        let repo = self.repo_for(mode, id)?;
        if id == self.session_id() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "cannot delete the currently active session — switch to another session first",
            ));
        }
        Self::require_exactly(mode, &repo, id)?;
        repo.delete(id)
    }

    /// List every session sitting in this repo's `.trash/` subdirectory (repo mode only) — see
    /// [`crate::session_store::SessionRepo::list_trash`].
    ///
    /// In service mode the trash of **every** mounted shard's tenant directory is merged, so a
    /// session deleted on one mount is restorable from any connection this tenant holds.
    pub(super) fn list_trash(
        &self,
        mode: &Mode,
    ) -> std::io::Result<Vec<crate::session_store::TrashEntry>> {
        if let Some((dirs, layout)) = mode.tenant_scope() {
            let mut all = Vec::new();
            // Read-only across every mount, so no id prefix: nothing here mints a session.
            let opts = crate::session_store::RepoOptions {
                layout,
                id_prefix: None,
            };
            for dir in dirs {
                if dir.is_dir() {
                    all.extend(SessionRepo::open_with(dir, opts.clone())?.list_trash()?);
                }
            }
            all.sort_by(|a, b| b.deleted_at.cmp(&a.deleted_at));
            return Ok(all);
        }
        let repo = self.repo.as_ref().ok_or_else(not_in_repo_mode)?;
        repo.list_trash()
    }

    /// Restore a session by id out of `.trash/` back to its original location (repo mode only) — see
    /// [`crate::session_store::SessionRepo::restore_session`].
    pub(super) fn restore_session(&self, mode: &Mode, id: &str) -> std::io::Result<bool> {
        self.repo_for(mode, id)?.restore_session(id)
    }

    /// This process's own cwd's sessions, newest first (empty unless in repo mode).
    /// `on_progress(scanned, total)` is invoked once per file as the scan completes it — see
    /// [`SessionRepo::list_with_progress`](crate::session_store::SessionRepo::list_with_progress).
    ///
    /// Filtered to `self.meta.cwd` — the exact match [`SessionRepo::resume_latest_or_create`]
    /// already applies when reattaching at startup. A no-op under the default per-cwd repo
    /// directory (every session there already shares this cwd, by construction — see
    /// `default_session_dir`), but load-bearing under an explicit `--session-dir` shared across
    /// projects: Track L28 (pi-parity fix) — this used to return every session in the directory
    /// unfiltered, so a shared `--session-dir` leaked another project's sessions into this one's
    /// `list_sessions` response.
    ///
    /// Runs the scan on `spawn_blocking` (like [`persist_blocking`]) rather than directly on the
    /// caller's own task — `SessionRepo::list_with_progress`'s underlying `scan_listings` does
    /// synchronous file I/O across a worker pool, and this command is reachable from the busy-mode
    /// loop while a model turn is in flight (unlike `persist_blocking`'s dedicated task, this one
    /// would otherwise stall that same task's turn-event delivery, `abort` handling, and checkpoint
    /// persistence for however long a large, unpruned session directory takes to scan).
    ///
    /// **Service mode drops the cwd filter.** There the cwd is the tenant's sandbox `workspace_root`
    /// and the directory is already `<shard>/<tenant>/sessions` — scoped by construction — so
    /// filtering would only hide a tenant's own sessions whose workspace path happens to have moved.
    pub(super) async fn list_with_progress(
        &self,
        mode: &Mode,
        on_progress: impl Fn(usize, usize) + Send + Sync + 'static,
    ) -> Vec<SessionMeta> {
        if let Some((dirs, layout)) = mode.listing_scope() {
            return scan_session_dirs(dirs, layout, on_progress).await;
        }
        let Some(repo) = self.repo.clone() else {
            return Vec::new();
        };
        let cwd = self.meta.cwd.clone();
        let sessions =
            match tokio::task::spawn_blocking(move || repo.list_with_progress(on_progress)).await {
                Ok(Ok(sessions)) => sessions,
                Ok(Err(e)) => {
                    eprintln!("serve: failed to list sessions: {e}");
                    Vec::new()
                }
                // `list_with_progress` never panics itself and this task is never cancelled — same
                // reasoning as `persist_blocking`'s identical re-raise.
                Err(e) => std::panic::resume_unwind(e.into_panic()),
            };
        sessions.into_iter().filter(|m| m.cwd == cwd).collect()
    }

    /// Every session across every project's own repo directory (pi's cross-project `listAll`), not
    /// just this process's own — the parent of this repo's directory is treated as the shared sessions
    /// root, with one subdirectory per project (the convention
    /// [`session_store::default_session_dir`](crate::session_store::default_session_dir) follows).
    /// `Err` when not in repo mode, or the repo directory has no parent to scan siblings of.
    /// `on_progress(scanned, total)` is invoked once per file across every project combined — see
    /// [`SessionRepo::list_all_with_progress`](crate::session_store::SessionRepo::list_all_with_progress).
    ///
    /// Runs on `spawn_blocking` — see [`Self::list_with_progress`]'s identical doc comment for why a
    /// cross-project scan (bigger than a single repo's own) must not run directly on the caller's own
    /// task.
    ///
    /// **Service mode's "everything" is the tenant's own, across every mounted shard** — a session
    /// minted on another shard is still this tenant's, and still listable here. Never another
    /// tenant's, and never the sibling-project walk (that root is the replica's).
    pub(super) async fn list_all_with_progress(
        &self,
        mode: &Mode,
        on_progress: impl Fn(usize, usize) + Send + Sync + 'static,
    ) -> std::io::Result<Vec<SessionMeta>> {
        if let Some((dirs, layout)) = mode.tenant_scope() {
            return Ok(scan_session_dirs(dirs, layout, on_progress).await);
        }
        let repo = self.repo.as_ref().ok_or_else(not_in_repo_mode)?;
        let root = repo
            .dir()
            .parent()
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "session directory has no parent to list other projects from",
                )
            })?
            .to_path_buf();
        match tokio::task::spawn_blocking(move || {
            SessionRepo::list_all_with_progress(&root, on_progress)
        })
        .await
        {
            Ok(result) => result,
            // `list_all_with_progress` never panics itself and this task is never cancelled — same
            // reasoning as `persist_blocking`'s identical re-raise.
            Err(e) => std::panic::resume_unwind(e.into_panic()),
        }
    }

    /// Set the current session's title.
    /// Persist this session's exec endpoint so a resume reattaches it. Best-effort: a session
    /// running without persistence still gets the live target, it just won't survive a restart.
    pub(super) fn set_exec_endpoint(&mut self, endpoint: Option<Value>) {
        self.meta.exec_endpoint = endpoint.clone();
        if let Some(store) = &mut self.store
            && let Err(e) = store.set_exec_endpoint(endpoint)
        {
            tracing::warn!(error = %e, "failed to persist the session's exec endpoint");
        }
    }

    pub(super) fn set_title(&mut self, title: &str) -> std::io::Result<()> {
        if let Some(store) = &mut self.store {
            store.set_title(title)?;
            self.meta = store.meta().clone();
        }
        Ok(())
    }

    /// Set (`label: Some`) or clear (`label: None`) a bookmark on `target_id` — see
    /// `SessionStore::set_label`. Unlike `set_title` (meaningful even without persistence, since
    /// `self.meta` always exists), a label only makes sense against a persisted tree entry, so this
    /// errors rather than silently no-oping when persistence isn't configured — the same contract
    /// `switch_branch` already uses for a `target_id`-based command.
    pub(super) fn set_label(
        &mut self,
        target_id: &str,
        label: Option<&str>,
    ) -> std::io::Result<()> {
        let store = self.store.as_mut().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "no session persistence configured (start serve with --session-file or --session-dir)",
            )
        })?;
        store.set_label(target_id, label)
    }

    /// Append an opaque, caller-defined tree entry as a child of the active tip and advance the tip to
    /// it — see `SessionStore::append_custom`. Same persistence-required contract as `set_label` (a
    /// custom entry only makes sense against a persisted tree). Returns the new entry's id.
    pub(super) fn append_custom(
        &mut self,
        kind: impl Into<String>,
        data: serde_json::Value,
    ) -> std::io::Result<String> {
        let store = self.store.as_mut().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "no session persistence configured (start serve with --session-file or --session-dir)",
            )
        })?;
        store.append_custom(kind, data)
    }

    /// The label currently set on `target_id`, if any — see `SessionStore::get_label`. Same
    /// persistence-required contract as `set_label`.
    pub(super) fn get_label(&self, target_id: &str) -> std::io::Result<Option<&str>> {
        let store = self.store.as_ref().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "no session persistence configured (start serve with --session-file or --session-dir)",
            )
        })?;
        Ok(store.get_label(target_id))
    }

    /// Every branch in the current session's tree (empty unless persistence is configured — a
    /// pure in-memory session, with no `--session-file`/`--session-dir`, has no tree to report).
    pub(super) fn list_branches(&self) -> Vec<BranchInfo> {
        self.store
            .as_ref()
            .map(SessionStore::list_branches)
            .unwrap_or_default()
    }

    /// Every node in the current session's tree — every message on every branch, unlike
    /// `list_branches`' leaves-only view (empty unless persistence is configured).
    pub(super) fn tree(&self) -> Vec<crate::session_store::TreeNode> {
        self.store
            .as_ref()
            .map(SessionStore::tree)
            .unwrap_or_default()
    }

    /// Every abandoned branch's full message chain, for `export_html` — see
    /// `SessionStore::abandoned_branches` (empty unless persistence is configured).
    pub(super) fn abandoned_branches(&self) -> Vec<(usize, Vec<agent_core::Message>)> {
        self.store
            .as_ref()
            .map(SessionStore::abandoned_branches)
            .unwrap_or_default()
    }

    /// Every non-message session event recorded so far, for `export_html` — see
    /// `SessionStore::export_events` (Track L36; empty unless persistence is configured).
    pub(super) fn export_events(&self) -> &[crate::session_store::ExportEvent] {
        self.store
            .as_ref()
            .map(SessionStore::export_events)
            .unwrap_or_default()
    }

    /// Ids of the active path's messages, root-first — parallel to the live `Session.messages` when
    /// persistence is configured; empty in pure in-memory mode (nothing to id). What `get_messages`
    /// tags each message with, and what a client names as `switch_branch`'s `target_id`.
    pub(super) fn active_ids(&self) -> &[String] {
        self.store
            .as_ref()
            .map(SessionStore::active_ids)
            .unwrap_or(&[])
    }

    /// MCP task journal entries of `kind` on the active path that the session accepts (see
    /// `SessionStore::active_journal`); empty without persistence.
    pub(super) fn active_journal(&self, kind: &str) -> Vec<Value> {
        self.store
            .as_ref()
            .map(|store| store.active_journal(kind))
            .unwrap_or_default()
    }

    /// Append an MCP task journal entry, sealed for the session (see
    /// `SessionStore::append_journal`). Same persistence-required contract as `append_custom`.
    pub(super) fn append_journal(&mut self, kind: &str, data: Value) -> std::io::Result<String> {
        let store = self.store.as_mut().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "no session persistence configured (start serve with --session-file or --session-dir)",
            )
        })?;
        store.append_journal(kind, data)
    }

    /// Every user-turn message anywhere in the session's tree — every branch, not just the active path
    /// — as `(id, Message)` pairs (empty unless persistence is configured). What `get_fork_messages`
    /// surfaces, matching pi's own whole-tree `getUserMessagesForForking` rather than only the active
    /// path's own chain.
    pub(super) fn all_user_messages(&self) -> Vec<(String, agent_core::Message)> {
        self.store
            .as_ref()
            .map(SessionStore::all_user_messages)
            .unwrap_or_default()
    }

    /// Switch the active branch to `target_id` — or, when `before` is set, to `target_id`'s own
    /// *parent* instead, which is the tree's root (before any message) when `target_id` is the very
    /// first message. That lets a client redo the first message in place (pi's own
    /// `SessionManager::resetLeaf`) without a separate root sentinel on the wire: the client already
    /// has real entry ids from `get_fork_messages`, so `{target_id: <first message>, before: true}`
    /// means exactly that, mirroring `fork`'s identical `before` semantics.
    ///
    /// When `summarize` and the branch being left behind has unsummarized activity (see
    /// `SessionStore::abandoned_by_switch`/`abandoned_to_root`), generates a summary via `agent` and
    /// persists it *before* switching — mirroring pi's `navigateTree`. A client-requested abort
    /// (`Error::Cancelled`) leaves the session completely unchanged (see the `Err` match arm below); any
    /// *other* summarization failure (a network error, the model returning an error) is fatal to the
    /// whole navigation too — Fix 3 (pi-parity gap): this used to log the failure and switch anyway
    /// (`eprintln!("serve: branch summarization failed, switching anyway: ...")`), matching neither pi's
    /// own `packages/coding-agent/src/core/agent-session.ts`'s `navigateTree` (which `throw`s on any
    /// non-abort summarization error — see `generateBranchSummary` in
    /// `packages/coding-agent/src/core/compaction/branch-summarization.ts`, whose `{error}` result
    /// `navigateTree` turns into that throw, aborting the whole navigation before ever switching the
    /// active leaf) nor this RPC's own error-handling convention elsewhere (`switch_active_with_summary`
    /// failing to *persist* an already-generated summary, just below, still falls back to a plain switch —
    /// a different, later failure mode this fix does not touch: the summary text already exists there,
    /// only the durable record of it is what's missing, so blocking navigation over that would strand the
    /// client on the old branch to save a recap it already generated once and could just regenerate).
    /// Silently proceeding as if nothing happened gave a caller no way to know the recap never got made
    /// and no way to retry — returning `Err` here makes the RPC dispatch's own generic `Err` handling
    /// surface it as a normal failed response, so the client sees the failure and can choose to retry
    /// (with `summarize:false` if it wants to switch without one) rather than silently getting a switch
    /// it didn't ask for.
    ///
    /// `custom_instructions`, when given, steers *what* the branch recap emphasizes — the same
    /// "Additional focus" framing manual `compact` already supports — threaded straight through to
    /// [`Agent::summarize_branch`]; ignored (no summarization call happens at all) when `summarize` is
    /// `false`. `replace_instructions` (Task #17, pi-parity fix) forwards straight through to that same
    /// call's own parameter of the same name — `true` uses `custom_instructions` as the *entire*
    /// instruction section instead of appending it after the default structured template; a no-op when
    /// `custom_instructions` is `None`, same as `summarize_branch`'s own doc comment describes.
    ///
    /// Returns the resolved target alongside the switched-to `Session` — `None` when `before` resolved
    /// to the tree's own root — so the caller can restore the correct model/thinking-level for
    /// wherever the session actually landed (see [`Self::model_and_level_at_opt`]) instead of querying
    /// against the raw, pre-resolution `target_id` argument.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn switch_branch(
        &mut self,
        agent: &Agent,
        target_id: &str,
        before: bool,
        summarize: bool,
        custom_instructions: Option<&str>,
        replace_instructions: bool,
        cancel: &CancellationToken,
    ) -> std::io::Result<(Session, Option<String>)> {
        let store = self.store.as_mut().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "no session persistence configured (start serve with --session-file or --session-dir)",
            )
        })?;

        // Resolve `before` up front, rejecting an unknown `target_id` here rather than wasting a
        // summarization call on a switch that's about to fail anyway. `resolved: None` means "reset to
        // root" — everything else (including a normal, non-`before` switch) names a real node.
        let resolved: Option<String> = if before {
            match store.parent_of(target_id) {
                Some(parent) => parent,
                None => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("no message with id {target_id} in this session"),
                    ));
                }
            }
        } else {
            Some(target_id.to_string())
        };

        // A summary, once generated, is applied via `switch_active_with_summary`/
        // `switch_active_to_root_with_summary` — it both persists the recap *and* installs it as the
        // new active tip in one step, so it actually reaches the model on the next turn. Anything else
        // (nothing abandoned, `summarize` off, the model call failed or returned nothing) falls through
        // to a plain `switch_active`/`switch_active_to_root`.
        let mut summary_to_apply: Option<(String, String, BranchSummaryDetails)> = None;
        if summarize {
            let abandoned = match &resolved {
                Some(real_target) => store.abandoned_by_switch(real_target),
                None => store.abandoned_to_root(),
            };
            if !abandoned.is_empty()
                && let Some(from_id) = store.active_ids().last().cloned()
            {
                let (ids, messages): (Vec<String>, Vec<agent_core::Message>) =
                    abandoned.into_iter().unzip();
                match agent
                    .summarize_branch(&messages, cancel, custom_instructions, replace_instructions)
                    .await
                {
                    Ok(summary) if !summary.trim().is_empty() => {
                        let (mut read_files, mut modified_files) =
                            agent_core::compaction::extract_file_ops(&messages);
                        // Fold forward any nested branch summary's own file-tracking within this
                        // same abandoned range — otherwise a detour-off-a-detour loses the earlier
                        // summary's file awareness the moment only its prose survives to be scanned
                        // (see `SessionStore::branch_summary_details_within`).
                        let prior = store.branch_summary_details_within(&ids);
                        for f in prior.read_files {
                            if !read_files.contains(&f) {
                                read_files.push(f);
                            }
                        }
                        for f in prior.modified_files {
                            if !modified_files.contains(&f) {
                                modified_files.push(f);
                            }
                        }
                        let details = BranchSummaryDetails {
                            read_files,
                            modified_files,
                            summarized_messages: messages.len() as u64,
                        };
                        summary_to_apply = Some((summary, from_id, details));
                    }
                    Ok(_) => {} // empty summary — nothing worth recording
                    // A client-requested abort mid-summarization: unlike a genuine failure below,
                    // this must not fall through to switching anyway — pi's own contract
                    // (`navigateTree`'s `abortBranchSummary`) leaves the session completely
                    // unchanged on cancel, distinct from `cancelled: true` in the response the
                    // caller builds from this `Err`.
                    Err(agent_core::Error::Cancelled) => {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::Interrupted,
                            "branch summarization cancelled",
                        ));
                    }
                    // Fix 3: any other summarization failure is fatal to the whole navigation — the
                    // switch does not happen, and the RPC dispatch's generic `Err` handling reports
                    // this as a failed `switch_branch` response the client can see and retry.
                    Err(e) => {
                        return Err(std::io::Error::other(format!(
                            "branch summarization failed: {e}"
                        )));
                    }
                }
            }
        }

        let messages = match (&resolved, summary_to_apply) {
            (Some(real_target), Some((summary, from_id, details))) => {
                match store.switch_active_with_summary(real_target, summary, from_id, details) {
                    Ok(messages) => messages,
                    Err(e) => {
                        // Recording the summary failed (a disk error mid-rewrite) — the switch itself
                        // must still succeed rather than leaving the client stuck on the old branch.
                        eprintln!("serve: failed to persist branch summary, switching anyway: {e}");
                        store.switch_active(real_target)?
                    }
                }
            }
            (None, Some((summary, from_id, details))) => {
                match store.switch_active_to_root_with_summary(summary, from_id, details) {
                    Ok(messages) => messages,
                    Err(e) => {
                        eprintln!("serve: failed to persist branch summary, switching anyway: {e}");
                        store.switch_active_to_root()?
                    }
                }
            }
            (Some(real_target), None) => store.switch_active(real_target)?,
            (None, None) => store.switch_active_to_root()?,
        };
        self.meta = store.meta().clone();
        let mut session = Session::new();
        session.messages = Arc::new(messages);
        // The store's compaction provenance, generation included: the switched session is a
        // continuation of this store, not a fresh one. Starting from defaults reset the generation the
        // store remembers, so the next persist read the turn as a compaction, and lost the provenance
        // (read files, todos, user requests) the next real compaction carries forward.
        session.compaction = store.compaction_provenance();
        Ok((session, resolved))
    }

    /// Record that the active model changed — a no-op when persistence isn't configured (in-memory
    /// mode has nothing to append to). The caller (`set_model`/`cycle_model`) only calls this once it
    /// has already confirmed the model actually changed.
    pub(super) fn record_model_change(&mut self, model: &str) -> std::io::Result<()> {
        let Some(store) = &mut self.store else {
            return Ok(());
        };
        let r = store.record_model_change(model);
        if let Err(e) = &r {
            eprintln!("serve: failed to record model change: {e}");
        }
        r
    }

    /// Same idea as [`Self::record_model_change`], for the portable thinking level.
    pub(super) fn record_thinking_level_change(&mut self, level: &str) -> std::io::Result<()> {
        let Some(store) = &mut self.store else {
            return Ok(());
        };
        let r = store.record_thinking_level_change(level);
        if let Err(e) = &r {
            eprintln!("serve: failed to record thinking-level change: {e}");
        }
        r
    }

    /// The model/thinking-level to restore when switching to `target_id` (see
    /// `SessionStore::model_at`/`thinking_level_at`). Both always resolve to *something* concrete, never
    /// "leave whatever's currently active alone" — that would let a sibling branch's `set_model`/
    /// `cycle_thinking_level` bleed across a switch to a branch that never touched it, instead of
    /// landing on that branch's own actual state. The model falls back to this session's own
    /// creation-time model (`self.meta.model`) if nothing was ever recorded reaching that point — every
    /// session has a real starting model. The thinking level has no equivalent creation-time baseline
    /// in `SessionMeta`, so the caller passes `process_starting_level` (the level the process itself
    /// started at, from `--reasoning-effort`/`ThinkingLevel::Off`) as its fallback instead — the same
    /// "no change recorded yet" logic, just sourced from the process's own starting config rather than
    /// the session's.
    pub(super) fn model_and_level_at(
        &self,
        target_id: &str,
        process_starting_level: agent_core::ThinkingLevel,
    ) -> (String, agent_core::ThinkingLevel) {
        let Some(store) = &self.store else {
            return (self.meta.model.clone(), process_starting_level);
        };
        (
            store
                .model_at(target_id)
                .map(str::to_string)
                .unwrap_or_else(|| self.meta.model.clone()),
            store
                .thinking_level_at(target_id)
                .and_then(agent_core::ThinkingLevel::parse)
                .unwrap_or(process_starting_level),
        )
    }

    /// [`Self::model_and_level_at`], but for a `switch_branch` target that may have resolved to the
    /// tree's own root (`before: true` reaching the very first message) instead of a real node —
    /// `model_at`/`thinking_level_at` require an existing id and can't express "root" directly, so
    /// `None` here reads `SessionStore::model_at_root`/`thinking_level_at_root` instead.
    pub(super) fn model_and_level_at_opt(
        &self,
        target_id: Option<&str>,
        process_starting_level: agent_core::ThinkingLevel,
    ) -> (String, agent_core::ThinkingLevel) {
        let Some(target_id) = target_id else {
            let Some(store) = &self.store else {
                return (self.meta.model.clone(), process_starting_level);
            };
            return (
                store
                    .model_at_root()
                    .map(str::to_string)
                    .unwrap_or_else(|| self.meta.model.clone()),
                store
                    .thinking_level_at_root()
                    .and_then(agent_core::ThinkingLevel::parse)
                    .unwrap_or(process_starting_level),
            );
        };
        self.model_and_level_at(target_id, process_starting_level)
    }

    /// [`Self::model_and_level_at`] resolved at the *currently active* session's own tip — what a
    /// caller that just opened a different session entirely (`switch`), rather than a different branch
    /// within the same one (`switch_branch`), needs: `switch`'s `self.store = Some(store)` swap means
    /// `model_and_level_at`'s own tree lookups already operate on the newly-opened session, so only the
    /// target id needs to come from *this* session's own active tip instead of an RPC-supplied one.
    pub(super) fn model_and_level_at_active(
        &self,
        process_starting_level: agent_core::ThinkingLevel,
    ) -> (String, agent_core::ThinkingLevel) {
        let Some(target_id) = self
            .store
            .as_ref()
            .and_then(|s| s.active_ids().last())
            .cloned()
        else {
            return (self.meta.model.clone(), process_starting_level);
        };
        self.model_and_level_at(&target_id, process_starting_level)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn switch_branch_returns_a_clear_error_when_no_session_persistence_is_configured() {
        // B-M7 pi-parity gap: in `--no-session-persistence` (pure in-memory) mode there's no tree to
        // navigate at all — `self.store` is `None` by construction. `switch_branch` must fail with a
        // clear, documented error rather than panicking or returning a confusing failure; this
        // codebase has no in-memory tree fallback the way pi does (a deliberate architectural
        // difference, not a gap to close), so the contract this test protects is simply "fails
        // cleanly, doesn't crash."
        let mut persistence = Persistence {
            pinned_id: None,
            transcript: 0,
            repo: None,
            store: None,
            lock: None,
            writer_lock: false,
            meta: SessionMeta::new("/w", "claude-test"),
        };
        let agent = Agent::new(
            Arc::new(agent_core::MockTransport::new(vec![])),
            "claude-test",
        );
        let cancel = CancellationToken::new();

        let err = persistence
            .switch_branch(&agent, "some-id", false, true, None, false, &cancel)
            .await
            .expect_err("must fail clearly, not panic, when there's no session tree to navigate");

        assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
        assert!(
            err.to_string()
                .contains("no session persistence configured"),
            "got: {err}"
        );
    }

    #[tokio::test]
    async fn switch_branch_propagates_a_non_cancelled_summarization_failure_instead_of_switching_anyway()
     {
        // Fix 3 (pi-parity gap): a genuine (non-abort) summarization failure used to be logged and the
        // switch proceeded anyway with no summary attached — this test pins the corrected behavior:
        // the switch must not happen at all, and the caller must see it failed, matching pi's own
        // `packages/coding-agent/src/core/agent-session.ts`'s `navigateTree`, which `throw`s on any
        // non-abort summarization error (any non-abort summarization error is fatal to the whole
        // navigation).
        let dir = tempfile::tempdir().unwrap();
        let (mut persistence, _session) =
            Persistence::open_repo(dir.path(), "/w", "claude-test", SessionSelect::Latest).unwrap();
        let ids = {
            let store = persistence.store.as_mut().unwrap();
            let mut session = Session::new();
            session.user("a");
            session.user("b");
            session.user("c");
            session.user("d");
            store.append_new(&session.messages).unwrap();
            store.active_ids().to_vec()
        };

        // No scripted turns at all: the one model call `summarize_branch` makes fails outright — a
        // stand-in for a real network error, not a client-requested abort.
        let agent = Agent::new(
            Arc::new(agent_core::MockTransport::new(vec![])),
            "claude-test",
        );
        let cancel = CancellationToken::new();

        // Switching to `ids[1]` ("b") abandons "c"/"d" (non-empty), so `summarize:true` actually
        // attempts a summarization call rather than skipping it as a no-op.
        let err = persistence
            .switch_branch(&agent, &ids[1], false, true, None, false, &cancel)
            .await
            .expect_err("a genuine summarization failure must fail the whole switch");
        assert_ne!(
            err.kind(),
            std::io::ErrorKind::Interrupted,
            "must not be reported as a client-requested abort — that's a distinct, non-error outcome"
        );

        // The switch must not have happened: the active tip is still "d", not "b".
        assert_eq!(
            persistence.store.as_ref().unwrap().active_ids(),
            ids.as_slice(),
            "a failed summarization must leave the session on its original branch"
        );
    }

    #[tokio::test]
    async fn switch_branch_threads_replace_instructions_through_to_the_summarization_request() {
        // Task #17 (pi-parity fix): `replace_instructions:true` must actually reach
        // `Agent::summarize_branch`/`branch_summary::branch_summary_request` — previously `switch_branch`
        // had no such parameter at all and always passed a hardcoded `false`, so a caller asking to
        // fully replace the default structured template with its own instructions had no way to do so.
        let dir = tempfile::tempdir().unwrap();
        let (mut persistence, _session) =
            Persistence::open_repo(dir.path(), "/w", "claude-test", SessionSelect::Latest).unwrap();
        let ids = {
            let store = persistence.store.as_mut().unwrap();
            let mut session = Session::new();
            session.user("a");
            session.user("b");
            session.user("c");
            session.user("d");
            store.append_new(&session.messages).unwrap();
            store.active_ids().to_vec()
        };

        let transport = Arc::new(agent_core::MockTransport::new(vec![
            agent_core::mock::turn::text("recap"),
        ]));
        let agent = Agent::new(transport.clone(), "claude-test");
        let cancel = CancellationToken::new();

        // Switching to "b" abandons "c"/"d" (non-empty), so `summarize:true` actually attempts a
        // summarization call rather than skipping it as a no-op.
        let (_session, _resolved) = persistence
            .switch_branch(
                &agent,
                &ids[1],
                false,
                true,
                Some("only mention the file names"),
                true,
                &cancel,
            )
            .await
            .expect("summarization is scripted to succeed");

        let requests = transport.requests();
        assert_eq!(
            requests.len(),
            1,
            "exactly one summarization call must fire"
        );
        let prompt = match requests[0].messages.first().map(|m| &m.content[0]) {
            Some(agent_core::ContentBlock::Text { text, .. }) => text.clone(),
            other => panic!("expected a single text block, got {other:?}"),
        };
        assert!(
            prompt.contains("only mention the file names"),
            "custom_instructions must reach the request: {prompt}"
        );
        assert!(
            !prompt.contains(agent_core::branch_summary::BRANCH_SUMMARY_INSTRUCTION),
            "replace_instructions:true must drop the default structured template entirely: {prompt}"
        );
        assert!(
            !prompt.contains("Additional focus:"),
            "replace_instructions:true must not use the append-style framing: {prompt}"
        );
    }

    /// `switch_session` onto a session another writer holds is refused, and the current session —
    /// and the lock on it — stays exactly as it was.
    #[test]
    fn switch_refuses_a_session_another_writer_holds_and_keeps_the_current_one() {
        let dir = tempfile::tempdir().unwrap();
        let (mut persistence, _session) =
            Persistence::open_repo(dir.path(), "/w", "claude-test", SessionSelect::Fresh).unwrap();
        let current = persistence.session_id().to_owned();
        let other = persistence
            .repo
            .as_ref()
            .unwrap()
            .create(SessionMeta::new("/w", "claude-test"))
            .unwrap();
        let held = crate::session_store::lock_session_writer(other.path()).unwrap();
        let other_id = other.meta().id.clone();

        let err = persistence.switch(&other_id).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::WouldBlock, "{err}");
        assert_eq!(persistence.session_id(), current);
        let current_path = persistence.store.as_ref().unwrap().path().to_path_buf();
        assert!(
            crate::session_store::acquire_session_lock(&current_path)
                .unwrap()
                .is_none(),
            "the current session's lock was let go"
        );

        // Released, the same switch goes through.
        drop(held);
        persistence.switch(&other_id).unwrap();
        assert_eq!(persistence.session_id(), other_id);
    }

    /// A branch switch on a compacted session hands back a session carrying the store's compaction
    /// provenance, so the next persist appends. It used to hand back `Session::new()`'s defaults; the
    /// store read the reset generation as a compaction and rewrote the round — re-minting ids and
    /// replacing the provenance with the switched session's empty one.
    #[tokio::test]
    async fn switch_branch_on_a_compacted_session_keeps_its_provenance_and_appends() {
        let dir = tempfile::tempdir().unwrap();
        let (mut persistence, mut session) =
            Persistence::open_repo(dir.path(), "/w", "claude-test", SessionSelect::Latest).unwrap();
        for i in 0..4 {
            session.user(format!("u{i}"));
            session.push(agent_core::Message::assistant(vec![
                agent_core::ContentBlock::text(format!("a{i}")),
            ]));
        }
        persistence.persist(&session).unwrap();
        // A compaction as `Agent::compact` leaves it: a summary, the kept suffix, the provenance.
        agent_core::compaction::apply_summary(&mut session, 4, "earlier work", 1000);
        session.compaction.compactions = 1;
        session.compaction.read_files = vec!["src/important.rs".into()];
        persistence.persist(&session).unwrap();
        let summary_id = {
            let store = persistence.store.as_mut().unwrap();
            store.record_model_change("model-b").unwrap();
            store.active_ids()[0].clone()
        };

        let agent = Agent::new(
            Arc::new(agent_core::MockTransport::new(vec![])),
            "claude-test",
        );
        let (mut switched, _) = persistence
            .switch_branch(
                &agent,
                &summary_id,
                false,
                false,
                None,
                false,
                &CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(switched.compaction.compactions, 1);
        assert_eq!(switched.compaction.read_files, ["src/important.rs"]);

        let path = persistence.store.as_ref().unwrap().path().to_path_buf();
        let before = std::fs::read(&path).unwrap();
        switched.push(agent_core::Message::assistant(vec![
            agent_core::ContentBlock::text("new branch reply"),
        ]));
        switched.user("new branch question");
        persistence.persist(&switched).unwrap();
        let after = std::fs::read(&path).unwrap();
        assert!(
            after.starts_with(&before),
            "the persist rewrote the session"
        );
        let compaction_records = String::from_utf8_lossy(&after)
            .lines()
            .filter(|l| l.contains(r#""type":"compaction""#))
            .count();
        assert_eq!(
            compaction_records, 1,
            "a spurious compaction round was written"
        );

        let (_store, reopened) = SessionStore::open(path).unwrap();
        assert_eq!(reopened.messages.len(), 3);
        assert_eq!(reopened.compaction.read_files, ["src/important.rs"]);
    }

    #[test]
    fn set_label_and_get_label_return_a_clear_error_when_no_session_persistence_is_configured() {
        // Pi-parity audit H3: `SessionStore::set_label`/`get_label` were fully built, persisted, and
        // even carried across forks, but had no RPC surface at all — `Persistence::set_label`/
        // `get_label` are the RPC handlers' entry point. Same "no tree, no label" contract as
        // `switch_branch` above.
        let mut persistence = Persistence {
            pinned_id: None,
            transcript: 0,
            repo: None,
            store: None,
            lock: None,
            writer_lock: false,
            meta: SessionMeta::new("/w", "claude-test"),
        };
        let err = persistence
            .set_label("some-id", Some("checkpoint"))
            .expect_err("must fail clearly when there's no session tree to label");
        assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
        assert!(
            err.to_string()
                .contains("no session persistence configured"),
            "got: {err}"
        );

        let err = persistence
            .get_label("some-id")
            .expect_err("must fail clearly when there's no session tree to query");
        assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
    }

    #[test]
    fn list_trash_and_restore_session_return_a_clear_error_outside_repo_mode() {
        // Fix 7: `.trash/` is a `SessionRepo`-level concept (multiple sessions sharing one directory),
        // same as `delete`'s own repo-mode requirement — neither single-file nor in-memory-only mode has
        // a repo directory to consult.
        let persistence = Persistence {
            pinned_id: None,
            transcript: 0,
            repo: None,
            store: None,
            lock: None,
            writer_lock: false,
            meta: SessionMeta::new("/w", "claude-test"),
        };
        let err = persistence
            .list_trash(&Mode::Local)
            .expect_err("must fail clearly outside repo mode");
        assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
        assert!(err.to_string().contains("not in repo mode"), "got: {err}");

        let err = persistence
            .restore_session(&Mode::Local, "some-id")
            .expect_err("must fail clearly outside repo mode");
        assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
    }

    #[test]
    fn list_trash_and_restore_session_round_trip_through_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let (persistence, _session) =
            Persistence::open_repo(dir.path(), "/w", "claude-test", SessionSelect::Latest).unwrap();
        let repo = persistence.repo.as_ref().unwrap();
        let other = repo.create(SessionMeta::new("/w", "claude-test")).unwrap();
        let other_id = other.meta().id.clone();

        assert!(persistence.list_trash(&Mode::Local).unwrap().is_empty());
        persistence.delete(&Mode::Local, &other_id).unwrap();
        let trash = persistence.list_trash(&Mode::Local).unwrap();
        assert_eq!(trash.len(), 1);
        assert_eq!(trash[0].id, other_id);

        assert!(
            persistence
                .restore_session(&Mode::Local, &other_id)
                .unwrap()
        );
        assert!(persistence.list_trash(&Mode::Local).unwrap().is_empty());
        assert!(
            !persistence
                .restore_session(&Mode::Local, "never-existed")
                .unwrap()
        );
    }

    /// A checkpoint is acknowledged only once its write is done, and with the write's result: the
    /// agent must never be told a `tool_use` is durable while the write is still under way.
    #[tokio::test]
    async fn a_checkpoint_is_acknowledged_only_once_its_write_is_done() {
        let (ack, mut acked) = tokio::sync::oneshot::channel();
        let (finish, finished) = tokio::sync::oneshot::channel::<()>();
        let write = tokio::spawn(write_then_ack(
            async move {
                let _ = finished.await;
                ((), Ok(()))
            },
            ack,
        ));
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(
            matches!(
                acked.try_recv(),
                Err(tokio::sync::oneshot::error::TryRecvError::Empty)
            ),
            "acknowledged before the write finished"
        );
        finish.send(()).unwrap();
        let ((), written) = write.await.unwrap();
        assert!(written.is_ok());
        assert_eq!(acked.await.unwrap(), Ok(()));

        // A write that fails is acknowledged as a failure.
        let (ack, acked) = tokio::sync::oneshot::channel();
        let (_, result) =
            write_then_ack(async { ((), Err(std::io::Error::other("disk full"))) }, ack).await;
        assert!(result.is_err());
        assert_eq!(acked.await.unwrap(), Err("disk full".to_string()));
    }

    #[test]
    fn append_custom_returns_a_clear_error_when_no_session_persistence_is_configured() {
        // Pi-parity audit: `SessionStore::append_custom` was fully built and tested but had no RPC
        // surface at all — `Persistence::append_custom` is the RPC handler's entry point. Same
        // "no tree, nothing to append to" contract as `set_label`/`get_label` above.
        let mut persistence = Persistence {
            pinned_id: None,
            transcript: 0,
            repo: None,
            store: None,
            lock: None,
            writer_lock: false,
            meta: SessionMeta::new("/w", "claude-test"),
        };
        let err = persistence
            .append_custom("marker", json!({"k": "v"}))
            .expect_err("must fail clearly when there's no session tree to append to");
        assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
        assert!(
            err.to_string()
                .contains("no session persistence configured"),
            "got: {err}"
        );
    }

    #[test]
    fn fork_returns_a_clear_error_when_no_session_persistence_is_configured() {
        // Same contract as `switch_branch` above, for `Persistence::fork` — in-memory mode has no
        // `repo` to fork within either (`--session-file`/`--no-session-persistence` both leave `repo`
        // `None`; only `--session-dir` sets it).
        let mut persistence = Persistence {
            pinned_id: None,
            transcript: 0,
            repo: None,
            store: None,
            lock: None,
            writer_lock: false,
            meta: SessionMeta::new("/w", "claude-test"),
        };

        let err = persistence
            .fork(usize::MAX, None, false, agent_core::ThinkingLevel::Off)
            .expect_err("must fail clearly, not panic, without a repo to fork within");

        assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
    }

    #[test]
    fn fork_restores_the_model_and_level_recorded_at_the_forked_from_point() {
        // Task #2 (pi-parity fix): `fork`/`clone` previously swapped `session`/`store` but never
        // resolved model/level at all — the process's current global setting silently bled into the
        // forked session regardless of what was actually recorded on the branch being forked. Since
        // `SessionRepo::fork_at_entry` deliberately doesn't carry `ModelChange` bookkeeping into the new
        // session's own file (see its doc comment), this must be resolved against the *source* tree,
        // before the fork swaps `self.store`.
        let dir = tempfile::tempdir().unwrap();
        let (mut persistence, _session) =
            Persistence::open_repo(dir.path(), "/w", "claude-a", SessionSelect::Latest).unwrap();
        let ids = {
            let store = persistence.store.as_mut().unwrap();
            let mut session = Session::new();
            session.user("a");
            session.user("b");
            store.append_new(&session.messages).unwrap();
            store.active_ids().to_vec()
        };
        // Record a model change anchored at "a" (ids[0]) — everything from there on (including "b")
        // should resolve to "claude-b", even though the *source* session's `meta.model` is still
        // "claude-a" (its creation-time model) and the live process may be running neither.
        {
            let store = persistence.store.as_mut().unwrap();
            store.switch_active(&ids[0]).unwrap();
            store.record_model_change("claude-b").unwrap();
            store.switch_active(&ids[1]).unwrap();
        }

        let (_session, restored_model, restored_level) = persistence
            .fork(usize::MAX, None, false, agent_core::ThinkingLevel::Off)
            .expect("fork must succeed");

        assert_eq!(
            restored_model, "claude-b",
            "must resolve against the source tree's own recorded model change, not meta.model \
             (\"claude-a\") or a hardcoded default"
        );
        // No thinking-level change was ever recorded, so this falls back to the process's own starting
        // level, exactly like `model_and_level_at`'s doc comment describes.
        assert_eq!(restored_level, agent_core::ThinkingLevel::Off);

        // The new session's own header must NOT have silently inherited a stale model either — Task
        // #18 fixes `record_model_change` to also keep `meta.model` current, so the source's `meta`
        // (copied verbatim into the fork's own header by `SessionRepo::fork`) already reflects the
        // latest recorded model by the time the fork runs.
        assert_eq!(persistence.meta.model, "claude-b");
    }

    #[test]
    fn fork_at_entry_restores_the_model_recorded_at_the_target_not_the_active_tip() {
        // Same as the test above, but for the `target_id`-based `fork_at_entry` path — the resolved
        // point is `target_id` itself (or its parent, when `before` is set), not wherever the active
        // path currently ends.
        let dir = tempfile::tempdir().unwrap();
        let (mut persistence, _session) =
            Persistence::open_repo(dir.path(), "/w", "claude-a", SessionSelect::Latest).unwrap();
        let ids = {
            let store = persistence.store.as_mut().unwrap();
            let mut session = Session::new();
            session.user("a");
            session.user("b");
            store.append_new(&session.messages).unwrap();
            store.active_ids().to_vec()
        };
        {
            let store = persistence.store.as_mut().unwrap();
            store.record_model_change("claude-b").unwrap(); // anchored at the active tip, "b"
        }

        // Forking at "a" (before the model change ever took effect) must resolve to "claude-a", not
        // "claude-b" — the change was anchored at "b", not "a".
        let (_session, restored_model, _restored_level) = persistence
            .fork(
                usize::MAX,
                Some(&ids[0]),
                false,
                agent_core::ThinkingLevel::Off,
            )
            .expect("fork_at_entry must succeed");
        assert_eq!(restored_model, "claude-a");
    }
}
