//! `SessionLoop::new`: everything a session does before its first command, as one function per phase.
//!
//! Each phase returns the struct of what the later phases (and the final `SessionLoop`) need, and takes
//! what it reads from earlier phases by reference. The phases run in this order and no other: the exec cell
//! is attached before discovery reads through it, and the gates are installed after discovery and before
//! the system prompt is built.

use crate::serve::*;

type SetupError = Box<dyn std::error::Error + Send + Sync>;

/// Opens persistence, restores the history, applies `--name`.
struct Opened {
    cfg: ServeConfig,
    persistence: Persistence,
    session: Session,
    title_attempted_for: Option<String>,
}

/// Opens persistence, restores the history, applies `--name`.
async fn open_session(
    cfg: ServeConfig,
    timing: &mut crate::timing::StartupTiming,
) -> Result<Opened, SetupError> {
    // On the blocking pool, not here: this is the replica's worst head-of-line blocker (a full
    // transcript replay, on the reconnect path, correlated across sessions). See
    // [`open_persistence_blocking`].
    let (cfg, opened) = open_persistence_blocking(cfg).await;
    let (mut persistence, mut session) = opened?;
    // A previous process killed mid-turn left this session ending on its unanswered prompt.
    close_interrupted_prompt(&mut session);

    // The session whose title has already been attempted, so the attempt happens **at most once**
    // rather than once per turn until it succeeds. `title.is_none()` alone cannot express that: a
    // failed attempt leaves it `None`, so a model that reliably declines (or a gateway that is
    // down) would charge an extra call on every turn, forever.
    //
    // Keyed by session id rather than a bool, so a `new_session`/`switch_session` becomes eligible
    // again by construction — nothing has to remember to reset it.
    let title_attempted_for: Option<String> = None;
    timing.mark("open persistence");

    // `--name`: only for a genuinely fresh session (no messages, no title yet) — a resumed session
    // (repo mode reattaching to an existing cwd session, or a `--session-file` that already has
    // content) must not be silently renamed just because the process happened to be started with this
    // flag again. Deliberately diverges from pi, whose `--name` renames unconditionally on every
    // invocation (last-write-wins) — a script that resumes the same session on a schedule would
    // otherwise get its title rewritten to the same value every single run, and there's no way for an
    // operator to tell "rename this" from "just start it the way I always do" apart on pi's side.
    if let Some(name) = &cfg.name
        && session.messages.is_empty()
        && persistence.meta.title.is_none()
        && let Err(e) = persistence.set_title(name)
    {
        eprintln!("serve: failed to set initial session name: {e}");
    }

    Ok(Opened {
        cfg,
        persistence,
        session,
        title_attempted_for,
    })
}

/// The session's mode, trust, exec cell, bash job tables, MCP host/hub/skills and sandbox
/// attachment: everything the tools run on and talk to.
struct ExecSetup {
    input_rx: mpsc::Receiver<String>,
    mode: Mode,
    cwd: std::path::PathBuf,
    disk_overrides: bool,
    project_trusted: bool,
    exec_cell: crate::exec_endpoint::ExecCell,
    bash_jobs: Arc<crate::tools::bash_jobs::BashJobs>,
    bash_jobs_wake: tokio::sync::watch::Receiver<u64>,
    host_bash_jobs: Arc<crate::tools::bash_jobs::BashJobs>,
    bash_jobs_transcript: u64,
    mcp_enabled: crate::tools::mcp::McpEnabledSet,
    mcp_host: Arc<crate::tools::mcp_host::McpHost>,
    mcp_events: Option<crate::tools::mcp_events::McpEventsHub>,
    mcp_skills: crate::tools::mcp_skills::McpSkills,
    sandbox_snapshot: Option<crate::service::Snapshot>,
    job_notify: Option<Arc<crate::tools::job_notify::SessionNotify>>,
}

/// The session's mode, trust, exec cell, bash job tables, MCP host/hub/skills and sandbox
/// attachment: everything the tools run on and talk to.
async fn setup_exec(
    cfg: &mut ServeConfig,
    mut input_rx: mpsc::Receiver<String>,
    out_conn: &SharedOutConn,
    running: &Arc<std::sync::atomic::AtomicBool>,
    keep_alive: &Arc<std::sync::atomic::AtomicBool>,
    timing: &mut crate::timing::StartupTiming,
    persistence: &Persistence,
) -> Result<ExecSetup, SetupError> {
    // Assemble the system prompt from the base identity + this repo's project instructions + skills +
    // environment, so the agent behaves like it belongs in the working directory. Split into a static
    // half (this discovery-based block — expensive, rebuilt only on `set_model`/`set_thinking`/`reload`)
    // and a cheap dynamic footer (current date/cwd, recomputed before every `prompt` via `full_system`)
    // so a long-running `serve` process doesn't re-walk the filesystem every turn just for the date.
    //
    // In service mode the "working directory" is the tenant's own `workspace_root` **inside its
    // sandbox** — a path on the other side of the exec endpoint, which this process never stats.
    // Everything below that would otherwise walk it is either skipped (trust resolution, the
    // worktree sweep — the only repo they could see is the replica's) or re-aimed at the sandbox
    // through the session's own backend (`ServiceSession::resources` below). What must never
    // happen is the third option: walking the replica's cwd and calling the result a tenant's.
    let mode = cfg.mode();
    let cwd = mode.session_cwd();
    let has_gated_resources =
        mode.trust_gate_applies() && crate::trust_store::has_trust_gated_resources(&cwd);
    // `disk_overrides` stays off in service mode: it governs whether the **replica's** filesystem
    // may set a tenant's base prompt, and the answer to that is never yes. The tenant's own
    // `SYSTEM.md`/`APPEND_SYSTEM.md` arrive instead as `base`/`append` below, read from its sandbox.
    let disk_overrides = mode.reads_disk_overrides();
    // Sandbox content is the tenant's own, so it counts as trusted — there is no second party for
    // the trust gate to protect against inside one tenant's own box.
    let project_trusted = !mode.trust_gate_applies()
        || resolve_project_trust(
            cfg.trust_project,
            cfg.force_untrusted,
            cfg.default_project_trust,
            crate::trust_store::TrustStore::open_default().lookup(&cwd),
            has_gated_resources,
        );
    // Reap any subagent worktree orphaned by a previous crash of a process against this repo. Not
    // in service mode: the only repo this could sweep is the replica's own checkout.
    if mode.sweeps_worktrees() {
        crate::worktree::sweep(&cwd);
    }

    // Track L32 (pi-parity fix): mirrors `main.rs`'s identical warning for `run` — an untrusted
    // project with a `SYSTEM.md`/skills/prompts on disk silently skipped all of them with no signal at
    // all that anything was there. Re-checked (not just at startup) on every `reload`, below, since
    // trust can change mid-process (`agent trust`/`--trust-project` since startup) without a restart.
    if !project_trusted && has_gated_resources {
        eprintln!(
            "warning: {} has a project-local SYSTEM.md/APPEND_SYSTEM.md, skills, prompt templates, AGENTS.md frontmatter, or a \
         settings.json on disk, but the project isn't trusted, so they were skipped — pass \
         --trust-project or run `{program} trust {}` to enable them (a project's own settings.json \
         additionally requires a *persisted* `{program} trust`, not just a one-off --trust-project)",
            cwd.display(),
            cwd.display(),
            program = crate::PROGRAM
        );
    }

    // Skills are discovered by path, not inlined into the prompt — invoking one relies on the model
    // being able to open its `SKILL.md` itself, so advertising them at all when `read` isn't registered
    // (a restricted `--tools`/`--exclude-tools` invocation) just adds dead weight (pi-parity fix).
    // Tools are fixed for the whole process (see `build_agent`'s doc comment), so this one check is
    // reused verbatim by `reload`'s own rebuild below rather than re-deriving it.
    // Strict in service mode: an empty cell is an error, never this host. The fallback that makes a
    // local session convenient (`RealRunner`/`LocalFs`) is, on a replica, a tenant's tool running on
    // the machine every other tenant shares.
    let exec_cell = mode.new_exec_cell();
    // `bash` jobs outlive the run that started them here: ending a turn leaves them running, and the
    // idle loop below starts a turn when one ends (see the `bash_jobs_wake` arm). They end with the
    // session (stopped explicitly when this function returns), and a transcript switch stops them
    // (they belong to its history).
    let bash_jobs = exec_cell.bash_jobs().clone();
    bash_jobs.set_policy(crate::tools::bash_jobs::EndPolicy::KeepRunning);
    let bash_jobs_wake = bash_jobs.subscribe();
    // What the host `bash` command (the client's own, not the model's) leaves running (`cmd &`).
    // A table of its own, so those processes never reach the model: no finish notice, no wake (a
    // paid turn the client never asked for, quoting output it may have excluded from context), no
    // `<background-jobs>` carry. Nothing reads notices from it; they live until they exit or the
    // session ends, and a transcript switch leaves them alone (they belong to the client, not to a
    // history).
    let host_bash_jobs = Arc::new(crate::tools::bash_jobs::BashJobs::new(
        crate::tools::bash_jobs::EndPolicy::KeepRunning,
    ));
    let bash_jobs_transcript = persistence.transcript();
    // Per-session MCP kit gate (which configured servers' tools are advertised). Defaults to all
    // enabled — matching prior behavior — and is cleared on session switch so tenants cannot inherit
    // each other's enablement the way they must not inherit each other's exec endpoint.
    let mcp_enabled = crate::tools::mcp::McpEnabledSet::new();
    // Where this session's MCP calls send their *questions* (elicitation, sampling). Always this
    // session's own hub. In service mode the connectors below are dialed by this session alone and
    // consult it directly; the operator's servers, connected once and shared by every session of a
    // daemon, find it through the task-local the session's runs are scoped in
    // (`tools::mcp::with_session_host`), so a server's question reaches the session whose call
    // raised it rather than whichever session installed a gate last.
    let mcp_host = Arc::new(crate::tools::mcp_host::McpHost::new());
    // Where this session's tools run, resolved once at start. The precedence, most specific first:
    //
    //   1. **the grant**, in service mode. It *is* the sandbox; nothing else is consulted.
    //   2. **this session's own record** (`meta.exec_endpoint`, written by `set_exec_endpoint`), if
    //      it still resolves. A session that was pointed at a sandbox stays pointed at it across a
    //      process restart, exactly as `reset_exec_endpoint!` keeps it across a session switch —
    //      which is what makes a cold reconnect at the same `?session_id=` behave like a warm one
    //      instead of quietly moving that session's work onto the server.
    //   3. **`--exec-url`/`--exec-cmd`**, the process-wide default. "A starting point, not a floor":
    //      it is what a session with no record of its own gets, and it never overrides one that has.
    //   4. nothing — this host.
    //
    // Service mode has neither flag (both are refused at startup) and no "starting point" either:
    // the grant names the sandbox, and it is attached **lazily** — probed by the first call that
    // needs it, once however many ask at the same time (`ExecCell::set_lazy`). A sandbox sleeps when
    // idle and any exec request wakes it, so a session that reopens with a snapshot of what it last
    // learned (below) sends nothing at all until it is used, and a probe that fails then is that
    // tool call's error. A new session with a bundle does the same: its definition is a local read.
    // A new session without one — no snapshot — probes now, as it needs the
    // sandbox for its prompt anyway, and one that doesn't answer **ends the session** rather than
    // leaving it running with nowhere for its tools to go. That error propagates out of
    // `serve_session`, which the daemon turns into an error frame on every attached connection.
    // A bundle grant's definition is fetched, verified and unpacked before anything reads it; a
    // failure ends the open, as an unreachable sandbox does.
    mode.load_bundle(cfg.mcp_http.as_ref()).await.map_err(
        |e| -> Box<dyn std::error::Error + Send + Sync> { format!("service: {e}").into() },
    )?;
    let sandbox_snapshot = mode.load_sandbox_snapshot(
        persistence.session_file(),
        persistence.session_id(),
        cfg.context_files,
    );
    match mode.lazy_exec(cfg.exec_max_response_bytes) {
        Some(connect) => {
            exec_cell.set_lazy(
                connect,
                sandbox_snapshot.as_ref().map(|s| s.sandbox.home.clone()),
            );
            match &sandbox_snapshot {
                Some(snapshot) => mode.seed_sandbox(snapshot.sandbox.clone()),
                // A bundle holds the whole definition, so nothing at open needs the sandbox:
                // the first tool that does connects it, and a dead one is that call's error.
                None if mode.has_bundle() => {}
                None => {
                    exec_cell.connect().await.map_err(
                        |e| -> Box<dyn std::error::Error + Send + Sync> {
                            format!("service: {e}").into()
                        },
                    )?;
                    timing.mark("probe sandbox");
                }
            }
        }
        None => {
            let restored = match persistence.meta.exec_endpoint.clone() {
                Some(spec) => {
                    restore_persisted_exec_endpoint(&spec, cfg.exec_max_response_bytes).await
                }
                None => None,
            };
            match restored {
                Some(target) => exec_cell.set(Some(target)),
                // Either this session never recorded an endpoint, or the one it recorded is gone.
                // Fall through to the process-wide default rather than to this host: a flag the
                // operator passed is a better guess than nothing, and a session whose sandbox has
                // genuinely vanished can still be re-pointed with `set_exec_endpoint`.
                None => match (&cfg.exec_url, &cfg.exec_cmd) {
                    (Some(u), _) => match crate::exec_endpoint::ExecTarget::http(
                        u,
                        &cfg.exec_header,
                        cfg.exec_max_response_bytes,
                    )
                    .await
                    {
                        Ok(t) => exec_cell.set(Some(t)),
                        Err(e) => eprintln!("warning: --exec-url ignored: {e}"),
                    },
                    (None, Some(c)) => match crate::exec_endpoint::ExecTarget::template(c).await {
                        Ok(t) => exec_cell.set(Some(t)),
                        Err(e) => eprintln!("warning: --exec-cmd ignored: {e}"),
                    },
                    (None, None) => {}
                },
            }
        }
    }
    // This session's own MCP connectors, named and credentialed by its grant, dialed now — not at
    // process startup, which is where the *operator's* configured servers would have come from.
    // Fail-soft per connector (a dead server costs its own tools, not the session), and the
    // warnings stay on the replica: they name the tenant's own connectors, but the client asked for
    // a session, not a connect report.
    //
    // After the sandbox is attached: the workspace's `.mcp.json` adds connectors, read through it (a
    // reopened session takes them from its snapshot instead, so it still wakes nothing).
    let from_file = match &sandbox_snapshot {
        Some(snapshot) => snapshot.resources.mcp_connectors.clone(),
        None => {
            mode.read_workspace_mcp(&exec_cell, persistence.session_id())
                .await
        }
    };
    if mode
        .connect_session_mcp(cfg, &mcp_host, persistence.session_id(), &from_file)
        .await
    {
        timing.mark("connect session MCP connectors");
    }
    // This session's view of the shared catalog: its non-tool MCP requests are registered under
    // its own host (see `McpCatalog::for_session`).
    cfg.mcp_catalog = cfg.mcp_catalog.for_session(mcp_host.clone());
    // MCP Events (draft extension): only when a server is connected at all, and never in service
    // mode — a grant connector has no `events` configuration, and the `mcp_events_*` commands are
    // refused there outright (`service::refused_command`). The hub interposes on `input_rx` so an
    // arriving event can be injected as an ordinary `prompt` command; with no servers there is
    // nothing to interpose for, and the channel stays exactly as it was.
    let mut mcp_events = None;
    if mode.allows_mcp_events() && !cfg.mcp_catalog.is_empty() {
        let out = Arc::downgrade(out_conn);
        let emit: crate::tools::mcp_events::Emitter = Arc::new(move |frame: Value| {
            if let Some(out) = out.upgrade() {
                lock_ignoring_poison(&out).broadcast(OutFrame::Value(frame));
            }
        });
        let (rx, hub) = crate::tools::mcp_events::McpEventsHub::attach(
            input_rx,
            crate::tools::mcp_events::McpEventsConfig {
                catalog: cfg.mcp_catalog.clone(),
                callback_url: cfg.mcp_events_callback_url.clone(),
                emit,
                running: running.clone(),
                keep_alive: (!cfg.mcp_events_reapable).then(|| keep_alive.clone()),
                owns_configured: cfg.mcp_events_owner,
                // Beside the transcript, so cursors and undelivered events persist with the
                // session: a restart resumes from the cursor and delivers what was pending.
                state_path: persistence
                    .session_file()
                    .map(crate::tools::mcp_events::state_path_for),
                session_id: persistence.session_id().to_string(),
            },
        );
        input_rx = rx;
        // Started below, once this session's elicitation and sampling gates are installed: a
        // server may raise a nested request during the very first `events/poll`, and asked before
        // then it would be declined as having no client.
        mcp_events = Some(hub);
    }
    // SEP-2640 skills from every connected server, seen through this session's MCP gate.
    let mcp_skills =
        crate::tools::mcp_skills::McpSkills::new(&cfg.mcp_catalog, mcp_enabled.clone());
    // The jobs this session's sandbox kept running while no replica served it — a reap, a deploy,
    // a failover — are its jobs again, under their old numbers: still running, or ended with their
    // notices due (which wake the idle loop below). What nobody kept is stopped. See
    // `tools::bash_sandbox`. A sandbox not attached yet is not woken for it: its jobs are read back
    // at its first use (or when one is asked about), and until then the reopen note says they have
    // not been checked.
    bash_jobs.set_scope(Some(crate::exec_endpoint::jobs::JobScope::for_session(
        mode.tenant(),
        persistence.session_id(),
    )));
    if exec_cell.pending() {
        crate::tools::bash_sandbox::reattach_later(&bash_jobs, exec_cell.runner());
    } else {
        let reattached = crate::tools::bash_sandbox::reattach(&bash_jobs, exec_cell.runner()).await;
        if reattached != Default::default() {
            tracing::info!(session = %persistence.session_id(), ?reattached, "reattached the session's sandbox jobs");
        }
    }
    stop_survivors(&bash_jobs, persistence).await;
    // exec v1.4: a sandbox's `job_ended` callback reaches this session only while this task holds
    // it, registered here for the task's life (dropped, it is unregistered). The job table answers
    // it. Service mode only: `job_notify` is never set anywhere else.
    let job_notify =
        match mode.register_job_notify(cfg.job_notify.as_ref(), persistence.session_id()) {
            Some(reg) => {
                let tracker: Arc<dyn crate::tools::job_notify::JobTracker> = bash_jobs.clone();
                reg.set_tracker(Arc::downgrade(&tracker));
                // Notify is available only with the edge's origin to name (and an endpoint that
                // offers it, asked at each keep). Without it every job is held, as before.
                reg.set_origin(cfg.notify_origin.clone());
                bash_jobs.set_notify(Some(&reg));
                // A real use of the sandbox may start a live reconcile (never one this section's own
                // requests cause: those go to the job store, not through the cell).
                let weak = Arc::downgrade(&reg);
                exec_cell.set_use_hook(Some(Arc::new(move || {
                    if let Some(n) = weak.upgrade() {
                        let now = n.now();
                        n.feed(crate::tools::job_notify::kernel::Event::Use { now });
                    }
                })));
                Some(reg)
            }
            None => None,
        };

    Ok(ExecSetup {
        input_rx,
        mode,
        cwd,
        disk_overrides,
        project_trusted,
        exec_cell,
        bash_jobs,
        bash_jobs_wake,
        host_bash_jobs,
        bash_jobs_transcript,
        mcp_enabled,
        mcp_host,
        mcp_events,
        mcp_skills,
        sandbox_snapshot,
        job_notify,
    })
}

/// Discovers resources and skills, opens memory, installs the approval, elicitation and sampling
/// gates and the MCP Apps plumbing, and builds the static system prompt.
struct PromptSetup {
    sandbox_refresh: bool,
    sandbox_snapshot_dirty: bool,
    sandbox_resources: Option<crate::service::Resources>,
    skills: Vec<crate::skills::Skill>,
    skill_collisions: Vec<crate::skills::Collision>,
    prompt_templates: Vec<crate::prompts::PromptTemplate>,
    prompt_collisions: Vec<crate::skills::Collision>,
    context_files: Vec<(String, String)>,
    sandbox_system: Option<String>,
    sandbox_append: Option<String>,
    has_read: bool,
    has_todo: bool,
    has_memory: bool,
    session_dir_cell: crate::memory::file::SessionDir,
    mounts: Vec<crate::memory::Mount>,
    memory_tool: Option<Arc<dyn agent_core::Tool>>,
    session_memory_active: bool,
    memory_sections: Vec<(crate::memory::MountKind, String)>,
    apps_epoch: Arc<std::sync::Mutex<u64>>,
    pending_approvals: PendingApprovals,
    approval: Option<crate::approval::ApprovalRuntime>,
    pending_elicitations: PendingElicitations,
    pending_samplings: serve_sampling::PendingSamplings,
    app_sink: crate::tools::mcp_apps::AppSink,
    app_redact: crate::tools::mcp_apps::Redactor,
    app_detached: crate::tools::mcp_apps::Detached,
    app_filter: crate::tools::mcp_apps::ToolFilter,
    apps_generation: Option<u64>,
    mcp_tools_generation: u64,
    mcp_filter: crate::tools::McpFilter,
    output_slot: crate::tools::structured_output::OutputSlot,
    structured_output: Option<Arc<crate::tools::structured_output::StructuredOutput>>,
    current_output_spec: Option<(Value, Option<String>)>,
    static_system: String,
}

/// Discovers resources and skills, opens memory, installs the approval, elicitation and sampling
/// gates and the MCP Apps plumbing, and builds the static system prompt.
#[allow(clippy::too_many_arguments)]
async fn setup_prompt(
    cfg: &mut ServeConfig,
    out_conn: &SharedOutConn,
    timing: &mut crate::timing::StartupTiming,
    persistence: &Persistence,
    mode: &Mode,
    cwd: &std::path::PathBuf,
    disk_overrides: bool,
    project_trusted: bool,
    exec_cell: &crate::exec_endpoint::ExecCell,
    mcp_enabled: &crate::tools::mcp::McpEnabledSet,
    mcp_host: &Arc<crate::tools::mcp_host::McpHost>,
    mcp_events: &Option<crate::tools::mcp_events::McpEventsHub>,
    sandbox_snapshot: Option<crate::service::Snapshot>,
) -> Result<PromptSetup, SetupError> {
    // Slash-command prompt templates (`/name args`), discoverable skills, agent definitions and this
    // project's own instruction files — for `get_commands`, for expanding a `/name`/`/skill:name`
    // prompt before it reaches the model, and to advertise in the system prompt below. Discovered
    // *before* the system prompt is built (rather than after, as this and the `reload` arm's own
    // discovery used to be ordered) so `build_static_system_prompt_with_context` takes the
    // already-discovered lists instead of re-walking the same directories itself.
    //
    // Deliberately **after** the exec probe: in service mode every one of these lives in the
    // tenant's sandbox and is read through the backend the probe just set up
    // (`ServiceSession::resources`). A reopened service session takes them from its snapshot
    // instead, so the prompt it builds is the one it had (and the prompt cache still hits) without
    // waking the sandbox; `sandbox_refresh` re-runs discovery once the sandbox is awake anyway.
    //
    // `sandbox_resources` is the set the prompt is built from in service mode, and what the
    // snapshot saves; `sandbox_snapshot_dirty` says the saved one is behind it (written at the top
    // of the loop, once the session's directory exists).
    let sandbox_refresh = sandbox_snapshot.is_some();
    let sandbox_snapshot_dirty = mode.keeps_sandbox_snapshot() && sandbox_snapshot.is_none();
    let mut sandbox_resources: Option<crate::service::Resources> = None;
    let mut discovered = match sandbox_snapshot {
        Some(snapshot) => {
            sandbox_resources = Some(snapshot.resources.clone());
            Discovered::from(snapshot.resources)
        }
        None => match mode
            .discover_sandbox_resources(exec_cell, cfg.context_files)
            .await
        {
            Some(found) => {
                sandbox_resources = Some(found.clone());
                Discovered::from(found)
            }
            None => discover_resources(cfg, cwd, project_trusted).await,
        },
    };
    // Agent definitions are trust-gated exactly like skills (a project-local `.claude/agents/*.md`
    // body is injected verbatim as a child's system prompt), so they're discovered here — after trust
    // is resolved — not by `main.rs` at `ServeConfig`-construction time, where the interactive trust
    // grant hasn't happened yet. Stored on `cfg` so `build_agent`/`build_tools` (which take only
    // `&cfg`) can reach them on every rebuild without re-walking. The `reload` arm re-discovers
    // below, since trust (and the definitions themselves) can change mid-process.
    cfg.agents = std::mem::take(&mut discovered.agents);
    let skills = std::mem::take(&mut discovered.skills);
    let skill_collisions = std::mem::take(&mut discovered.skill_collisions);
    let prompt_templates = std::mem::take(&mut discovered.prompt_templates);
    let prompt_collisions = std::mem::take(&mut discovered.prompt_collisions);
    let context_files = std::mem::take(&mut discovered.context_files);
    // The workspace's `AGENTS.md` frontmatter says which model, tools and step cap this agent runs
    // with. Applied once, here, before anything reads `cfg.model`/`cfg.tools`/`cfg.max_steps`: in
    // service mode this is the tenant's own file read through its sandbox (trusted by construction),
    // on this host only a trusted project's.
    if project_trusted {
        cfg.apply_agent_settings(&crate::resources::agent_settings_in(&context_files, cwd));
    }
    // The grant's ceilings come last, so nothing above (a flag, the file) can exceed them.
    (cfg.tools, cfg.max_steps) = mode.apply_limits(cfg.tools.take(), cfg.max_steps);
    cfg.exclude_tools = mode.without_sandbox_tools(cfg.exclude_tools.take());
    let sandbox_system = discovered.system.take();
    let sandbox_append = discovered.append_system.take();
    timing.mark("discover prompt templates/skills");

    let startup_tools = build_tools(
        cfg,
        cfg.image_auto_resize,
        exec_cell,
        mcp_enabled,
        &skills,
        None,
    );
    let has_read = startup_tools.get("read").is_some();
    let has_todo = startup_tools.get(crate::tools::todo::NAME).is_some();

    // Persistent memory (mirrors `run`): resolve the backend once here where `cwd` is known, then wrap it
    // as a shared `Arc<dyn Tool>` that every `build_agent` rebuild registers (so a mid-run `set_model`
    // doesn't drop it). `--no-memory` disables it; a bad backend DSN is fatal at startup, before any
    // session runs. The `MEMORY.md` index is read once and injected into the system prompt — a session's
    // memory is surfaced from its start (Claude Code's auto-memory model).
    //
    // In service mode the backend is the tenant's own `<home shard>/<tenant>/memory` — never a DSN
    // (`--memory` is refused) and never a path derived from the replica's cwd.
    let memory_backend: Option<Arc<dyn crate::memory::MemoryBackend>> =
        if cfg.no_memory || cfg.no_tools {
            None
        } else {
            Some(match mode.tenant_memory_backend() {
                Some(backend) => backend,
                None => crate::memory::open(cfg.memory.as_deref(), cwd)
                    .map_err(Box::<dyn std::error::Error + Send + Sync>::from)?,
            })
        };
    let has_memory = memory_backend.is_some();
    // The per-session working-memory mount (`/session`) rides a *shared, swappable* directory cell: a
    // `switch_session`/`new_session`/`fork` re-points the cell (below) and every holder — this session's
    // `memory` tool and every subagent sharing it by `Arc` — follows with no tool/agent rebuild. Off when
    // memory is disabled or `--no-session-memory` was passed. See `crate::memory::session_dir`.
    let session_dir_cell = crate::memory::file::SessionDir::new(persistence.memory_dir());
    let mounts: Vec<crate::memory::Mount> = {
        let mut v = Vec::new();
        if let Some(backend) = &memory_backend {
            v.push(crate::memory::Mount {
                kind: crate::memory::MountKind::Durable,
                backend: backend.clone(),
            });
            if !cfg.no_session_memory {
                // Sealed with the tenant's key in service mode, exactly like the durable mount and
                // the transcript beside it: `<id>/memory/` lives on the same shared mount they do.
                let session_backend =
                    crate::memory::file::FileBackend::session_shared(session_dir_cell.clone());
                let session_backend = mode.seal_session_memory(session_backend);
                v.push(crate::memory::Mount {
                    kind: crate::memory::MountKind::Session,
                    backend: Arc::new(session_backend),
                });
            }
        }
        v
    };
    let memory_tool: Option<Arc<dyn agent_core::Tool>> = if mounts.is_empty() {
        None
    } else {
        Some(
            Arc::new(crate::tools::memory::Memory::mounted(mounts.clone()))
                as Arc<dyn agent_core::Tool>,
        )
    };
    // Whether a `/session` mount is live — gates the active compaction reminder pushed below. Stays valid
    // across session switches (the mount is always present; only its dir cell re-points).
    let session_memory_active = mounts
        .iter()
        .any(|m| m.kind == crate::memory::MountKind::Session);
    // Re-read at each `static_system` rebuild (a `set_model`/`set_thinking` switch, or a `prompt` whose
    // output-schema changed) so a session that has written new memories injects the *current* index per
    // mount, not a stale startup snapshot. Steady-state within one model, the `memory` tool's own `view`
    // is always live, so this need only refresh where the prompt is already being rebuilt.
    let memory_sections: Vec<(crate::memory::MountKind, String)> =
        crate::memory::mount_sections(&mounts).await;

    // Bumped by every `set_mcp_apps` and session switch. A declaration's deferred answer is sent
    // only if nothing bumped it since — checked and sent under this lock, which a withdrawal also
    // holds while it answers — so `enabled: true` can never arrive after the withdrawal that
    // superseded it.
    let apps_epoch: Arc<std::sync::Mutex<u64>> = Arc::new(std::sync::Mutex::new(0));

    // The interactive approval gate. One per session, shared by every `build_agent` rebuild and by every
    // subagent child, so a remembered decision isn't re-asked and eight parallel children can't spam
    // eight simultaneous questions. `pending_approvals` is the map the `approve` command resolves
    // against — held here rather than inside the gate so both the busy and idle command loops can reach
    // it while a run is blocked inside the gate itself.
    //
    // The gate itself always exists: MCP-served skills ask through it (activation, and code execution
    // while one is active) whatever `--approve` says. Only the per-tool runtime is `--approve`'s.
    let (approval_gate, pending) = ServeApprovalGate::new(
        out_conn.clone(),
        cfg.approval_timeout,
        persistence.session_id(),
    );
    mcp_enabled.skill_session().set_gate(approval_gate.clone());
    mcp_enabled
        .skill_session()
        .set_preapproved(cfg.approve_mcp_skills);
    // `--no-skills`: no MCP-served skills listed, offered or expanded either.
    mcp_enabled
        .skill_session()
        .set_listing_suppressed(cfg.no_skills);
    let pending_approvals = pending;
    let approval = (!cfg.approve.is_off())
        .then(|| crate::approval::ApprovalRuntime::new(approval_gate, cfg.approve.clone()));

    simulated_slow_gate_install().await;
    let (elicit_gate, pending_elicitations) = ServeElicitationGate::new(
        out_conn.clone(),
        cfg.approval_timeout,
        persistence.session_id(),
    );
    mcp_host.elicitation.install(elicit_gate);
    let (sampling_gate, pending_samplings) = ServeSamplingGate::new(
        out_conn.clone(),
        cfg.approval_timeout,
        persistence.session_id(),
    );
    mcp_host.sampling.install(sampling_gate);
    if let Some(hub) = &mcp_events {
        hub.start();
    }
    // Where an MCP App view's `mcp_app_open`/`mcp_app_result` frames go: this session's fanout,
    // held weakly like the elicitation gate's, so a view-bearing tool cannot keep it alive.
    let app_sink: crate::tools::mcp_apps::AppSink = {
        let out = Arc::downgrade(out_conn);
        Arc::new(move |frame: Value| {
            if let Some(out) = out.upgrade() {
                lock_ignoring_poison(&out).broadcast_app(frame);
            }
        })
    };
    // Service mode strips replica paths from what a tenant's client sees — a view's cancelled reason
    // included (it is an error string from the transport).
    let app_redact: crate::tools::mcp_apps::Redactor = {
        let mode = mode.clone();
        Arc::new(move |text: &str| mode.redact(text))
    };
    // A bridge request with nobody left to answer is cancelled (see `mcp_apps::handle_request`).
    let app_detached: crate::tools::mcp_apps::Detached = {
        let out = Arc::downgrade(out_conn);
        Arc::new(move || {
            out.upgrade()
                .is_none_or(|out| lock_ignoring_poison(&out).is_empty())
        })
    };
    // A view's `tools/call` passes exactly the model's tool filter.
    let app_filter = crate::tools::mcp_apps::ToolFilter {
        tools: cfg.tools.clone(),
        exclude: cfg.exclude_tools.clone(),
        no_tools: cfg.no_tools,
        deny: cfg.deny_tool.clone(),
    };
    // The pool generation this session's agent was last built against (`None`: not declared). A
    // prompt rebuilds when it moved — an apps dial landed, or went into backoff.
    let apps_generation: Option<u64> = None;
    // The MCP tool set's generation this session's agent was last built against. A prompt rebuilds
    // when it moved (a server re-listed its tools); a run already in flight picks the change up
    // through the tool source each prompt installs (`tools::mcp::tool_source`). Read before the first
    // build, so a change racing it is rebuilt for rather than missed.
    let mcp_tools_generation = cfg.mcp_tools.generation();
    let mcp_filter = crate::tools::McpFilter {
        loading: cfg.mcp_loading,
        tools: cfg.tools.clone(),
        exclude: cfg.exclude_tools.clone(),
        no_tools: cfg.no_tools,
        code_mode: cfg.code_mode,
    };

    // `structured_output` is installed per-`prompt` (see that arm's `output_schema` handling), not at
    // startup: one session can answer one request in prose and the next as typed JSON. The `OutputSlot`
    // outlives every rebuild — it is this session's return channel, and a `set_model` mid-run must not
    // drop the answer that landed in it. `current_output_spec` is what an incoming `prompt` is compared
    // against to decide whether anything actually has to be rebuilt.
    let output_slot = crate::tools::structured_output::OutputSlot::new();
    let structured_output: Option<Arc<crate::tools::structured_output::StructuredOutput>> = None;
    let current_output_spec: Option<(Value, Option<String>)> = None;

    // `_with_context`, not the walking `build_static_system_prompt`: the project-instruction files
    // were already read by discovery — from the sandbox in service mode, where re-walking
    // here would read the replica's own filesystem instead.
    //
    // `base`/`append` carry a service session's own `SYSTEM.md`/`APPEND_SYSTEM.md`, read from its
    // sandbox (`None` on this host, where `disk_overrides` finds them the way it always has). An
    // explicit `--append-system-prompt` still outranks the file, matching the on-disk precedence.
    let static_system = crate::resources::build_static_system_prompt_with_context(
        &crate::resources::PromptOptions {
            base: sandbox_system.as_deref(),
            default_base: &cfg.system,
            append: cfg.append_system.as_deref().or(sandbox_append.as_deref()),
            cwd,
            include_context_files: cfg.context_files,
            skills: &skills,
            // The skill listing's budget follows the configured `--model`, at every static rebuild:
            // a mid-session `set_model` keeps the listing (and the cached prefix) as it was.
            context_window: agent_core::capabilities(&cfg.model).context_window,
            has_read,
            has_todo,
            has_structured_output: structured_output.is_some(),
            has_memory,
            memory_sections: &memory_sections,
            project_trusted,
            agents: &cfg.agents,
            disk_overrides,
        },
        &context_files,
    );
    timing.mark("build static system prompt");

    Ok(PromptSetup {
        sandbox_refresh,
        sandbox_snapshot_dirty,
        sandbox_resources,
        skills,
        skill_collisions,
        prompt_templates,
        prompt_collisions,
        context_files,
        sandbox_system,
        sandbox_append,
        has_read,
        has_todo,
        has_memory,
        session_dir_cell,
        mounts,
        memory_tool,
        session_memory_active,
        memory_sections,
        apps_epoch,
        pending_approvals,
        approval,
        pending_elicitations,
        pending_samplings,
        app_sink,
        app_redact,
        app_detached,
        app_filter,
        apps_generation,
        mcp_tools_generation,
        mcp_filter,
        output_slot,
        structured_output,
        current_output_spec,
        static_system,
    })
}

/// Resolves the starting model and thinking level, the runtime toggles, and builds the gateway client
/// and the first `Agent`.
struct AgentSetup {
    run_retry_policy: crate::retry::RunRetryPolicy,
    current_model: String,
    scoped_models: Vec<ScopedModel>,
    cycle_models: Vec<String>,
    current_thinking: Option<u32>,
    starting_level: agent_core::ThinkingLevel,
    current_level: agent_core::ThinkingLevel,
    settings_store: Option<crate::settings::SettingsStore>,
    current_auto_compaction: bool,
    current_auto_retry: bool,
    current_block_images: bool,
    current_image_auto_resize: bool,
    write_locks: Arc<agent_core::WriteLockRegistry>,
    subagent_ctx: Option<Arc<crate::tools::subagent::SubagentCtx>>,
    checkpoint_rx: mpsc::UnboundedReceiver<Checkpoint>,
    checkpoint: Arc<dyn agent_core::CheckpointHook>,
    mcp_task_tx: mpsc::UnboundedSender<(String, Value)>,
    mcp_task_rx: mpsc::UnboundedReceiver<(String, Value)>,
    client: Arc<GatewayClient>,
    agent: Agent,
}

/// Resolves the starting model and thinking level, the runtime toggles, and builds the gateway client
/// and the first `Agent`.
#[allow(clippy::too_many_arguments)]
async fn setup_agent(
    cfg: &ServeConfig,
    timing: &mut crate::timing::StartupTiming,
    persistence: &Persistence,
    mode: &Mode,
    cwd: &Path,
    project_trusted: bool,
    exec_cell: &crate::exec_endpoint::ExecCell,
    bash_jobs: &Arc<crate::tools::bash_jobs::BashJobs>,
    mcp_enabled: &crate::tools::mcp::McpEnabledSet,
    mcp_skills: &crate::tools::mcp_skills::McpSkills,
    skills: &[crate::skills::Skill],
    context_files: &Vec<(String, String)>,
    mounts: &[crate::memory::Mount],
    memory_tool: &Option<Arc<dyn agent_core::Tool>>,
    approval: &Option<crate::approval::ApprovalRuntime>,
    structured_output: &Option<Arc<crate::tools::structured_output::StructuredOutput>>,
    static_system: &str,
) -> Result<AgentSetup, SetupError> {
    // Task #50: the same two operator-supplied overrides also drive the *whole-run* retry loop
    // (`crate::retry::RunRetryPolicy`) below, not just `client`'s own pre-connect/mid-stream layer just
    // above — previously `--retry-max-retries`/`--retry-base-delay-ms` had no effect on the `"prompt"`
    // command's own auto-retry loop.
    let run_retry_policy = crate::retry::RunRetryPolicy::from_overrides(
        cfg.retry_max_retries,
        cfg.retry_base_delay_ms,
    );

    // The model, thinking budget, and auto-compaction flag are runtime-switchable; everything else
    // (transport, tools, system prompt, loop bounds, cache settings) is fixed for the process.
    // `build_agent` folds the mutable trio into a fresh `Agent` whenever any of them changes.
    //
    // Task #5 (pi-parity fix): when the operator didn't explicitly pass `--model`/`--reasoning-effort`
    // for *this* invocation (`cfg.model_explicit`/`cfg.reasoning_effort_explicit` — see their own doc
    // comments), prefer whatever this reattached session's own active tip was actually last running on
    // over the CLI-resolved default, via the same `model_and_level_at_active` lookup `switch_session`
    // already uses. Without this, `set_model gpt-5` in a live session, disconnect, then a plain `serve`
    // in the same directory with no `--model` re-passed silently reverted to the global default for the
    // rest of the process — the exact "bleed" class `switch_session`/`switch_branch` are hardened
    // against (see their own doc comments), just left unguarded at ordinary process restart.
    // `model_and_level_at_active` already falls back to `cfg.model`/the CLI-resolved level (via
    // `self.meta.model`, which a fresh session's own `SessionMeta` is seeded from — see
    // `Persistence::open`) when nothing was ever recorded reaching the active tip, so this is a no-op
    // for a genuinely fresh session or pure in-memory mode. Model and level are resolved independently,
    // matching `--model`/`--reasoning-effort` being independent flags — an explicit `--model` with no
    // `--reasoning-effort` must still pick up the session's own last thinking level, not the process's
    // bare `Off` default, and vice versa.
    // Fix 1 (pi-parity gap): `cfg.reasoning_effort` is `None` both when the operator explicitly
    // requested no reasoning effort be set and when they simply never passed `--reasoning-effort`/had
    // no stored default — previously both collapsed to a bare `ThinkingLevel::Off` here, so a bare
    // `serve`/`run` invocation with no flags on a reasoning-capable model wire-disabled thinking
    // outright instead of picking any default depth. `default_reasoning_effort_for_model` supplies pi's
    // own "medium" default in that case (see its own doc comment); a model with no reasoning mechanism
    // at all still falls through to `Off`, same as before.
    //
    // Task #29 (pi-parity fix): that "medium" fallback must never fire when `cfg.reasoning_effort_explicit`
    // is `true` — an explicit `--model <pattern>:off` (or `--reasoning-effort` with no portable "off"
    // value of its own) resolves `cfg.reasoning_effort` to this same bare `None`, previously
    // indistinguishable here from "the operator never said anything", so it was silently promoted back
    // to the default depth instead of actually starting off. `main.rs` already computes
    // `reasoning_effort_explicit` correctly for both `run` and `serve` (see `ServeConfig::
    // reasoning_effort_explicit`'s own doc comment) — this is the one place that value wasn't consulted.
    let cfg_level = cfg
        .reasoning_effort
        .or_else(|| {
            if cfg.reasoning_effort_explicit {
                None
            } else {
                default_reasoning_effort_for_model(&cfg.model)
            }
        })
        .map(agent_core::ThinkingLevel::from)
        .unwrap_or(agent_core::ThinkingLevel::Off);
    let (session_model, session_level) = persistence.model_and_level_at_active(cfg_level);
    let (current_model, restored_level) = resolve_startup_model_and_level(
        &cfg.model,
        cfg_level,
        cfg.model_explicit,
        cfg.reasoning_effort_explicit,
        session_model,
        session_level,
    );
    // The grant's model ceiling covers whatever won above — the file's model, the replica's default
    // or a reattached session's last model. A session that cannot run on an allowed model does not
    // start; it is never quietly moved to another.
    if let Some(why) = mode.model_refusal(&current_model) {
        return Err(why.into());
    }
    // `cycle_model`'s candidate list: the operator-scoped `--models` set when given (parsed by
    // `resolve_model_scope` — comma-separated patterns, each an exact id, a glob against
    // `available_models()`, or either suffixed with `:<thinking-level>` to pin that entry's depth —
    // pi's own `--models`/`resolveModelScopeWithDiagnostics`), else the full known-model hint list.
    // Fixed for the process, like `tools` — there's no runtime RPC to change it. `get_available_models`
    // is deliberately NOT scoped by this — see that handler's own comment.
    let scoped_models: Vec<ScopedModel> = resolve_model_scope(&cfg.models, available_models());
    let cycle_models: Vec<String> = if scoped_models.is_empty() {
        available_models().iter().map(|s| s.to_string()).collect()
    } else {
        scoped_models.iter().map(|m| m.id.clone()).collect()
    };
    let cycle_models: Vec<String> = cycle_models
        .into_iter()
        .filter(|m| mode.model_refusal(m).is_none())
        .collect();
    let current_thinking = cfg.thinking;
    // The portable thinking-depth level (see `agent_core::ThinkingLevel`) — the runtime-mutable
    // counterpart to `cfg.reasoning_effort` (or, per the Task #5 restoration above, whatever this
    // reattached session's own active tip last recorded), seeded from it so a process started with
    // `--reasoning-effort` keeps that depth until `cycle_thinking_level`/`set_reasoning_effort` change
    // it. Unlike `current_thinking` (an explicit raw-budget override that wins when present), this
    // always takes effect: `build_agent` derives both the thinking budget *and* the reasoning effort
    // from it via `agent_core::thinking_for_level`, correctly for whichever mechanism `current_model`
    // actually uses.
    // Clamped against `current_model`'s capabilities immediately: a model that has a reasoning
    // mechanism but can't explicitly disable it (`reasoning_disableable == false`, e.g. most of the
    // OpenAI gpt-5 codex/pro line) has no legal `Off` state at all — leaving it there would silently
    // omit the reasoning field on every request and let the provider apply its own hidden default
    // effort, with the operator believing reasoning is off. See `agent_core::clamp_thinking_level`.
    let starting_level =
        agent_core::clamp_thinking_level(&agent_core::capabilities(&current_model), restored_level);
    // The runtime-mutable level starts at `starting_level`, but `switch_branch` needs the original
    // starting value too — as the fallback for a branch that never recorded its own thinking-level
    // change (see `Persistence::model_and_level_at`), so switching to it lands on the process's real
    // starting depth instead of silently keeping whatever a *different* branch last set.
    let current_level = starting_level;
    // Track L26 (pi-parity fix): `--no-compaction`/`AI_AGENT_NO_COMPACTION` (`cfg.no_compaction`) wins
    // outright when given; otherwise a persisted `agent settings` override survives across restarts
    // where the flag/env is absent (matching `default_project_trust`'s identical precedence tier —
    // `settings::Settings::compaction_enabled`'s own doc comment), finally defaulting to enabled.
    // `settings_store` is kept open for the rest of this process so `set_auto_compaction`, below, can
    // write a later runtime toggle straight back through to the same file.
    //
    // In service mode the store is **not opened at all**: it lives under the replica's `$HOME`, so
    // reading it would let the operator's machine configure a tenant, and writing it would let one
    // tenant's toggle reach every other session on the replica. Every `set_*` toggle below stays
    // session-local there (see `persist_setting!`), and the starting value is the built-in default.
    let settings_store = mode
        .reads_operator_settings()
        .then(crate::settings::SettingsStore::open_default);
    let current_auto_compaction = if cfg.no_compaction {
        false
    } else {
        settings_store
            .as_ref()
            .and_then(|s| s.get().compaction_enabled)
            .unwrap_or(true)
    };
    // Mid-stream transport-failure retry (`agent_core::Agent::with_auto_retry`) — on by default;
    // `set_auto_retry` lets an operator debugging a flaky network hop disable it to see the raw failure
    // on the very first hiccup instead of after several silent retries.
    let current_auto_retry = true;
    // Pi-parity fix (pass 20): `cfg.block_images`/`cfg.image_auto_resize` are already fully resolved by
    // `main.rs` (explicit flag, then a persisted `agent settings --block-images`/`--image-auto-resize`
    // default), but — unlike `current_auto_compaction`/`current_auto_retry` just above — `build_agent`
    // used to read them straight back off `cfg` on every rebuild instead of a runtime-mutable local,
    // so there was no `set_block_images`/`set_image_auto_resize` RPC arm that could ever change them for
    // a live process. These start at whatever `cfg` already resolved, and from here on are the sole
    // source `build_agent` consults for both.
    let current_block_images = cfg.block_images;
    let current_image_auto_resize = cfg.image_auto_resize;
    // Shared across every `build_agent` rebuild for this process's lifetime, so file-mutation
    // exclusivity (same-path `edit`/`write` calls) survives a `set_model`/`set_thinking` rebuild.
    let write_locks = Arc::new(agent_core::WriteLockRegistry::new());
    // The subagent context, built once and reused across every rebuild — its transport factory closes
    // over `cfg` and re-resolves per child model. The one parent-dependent piece, the model a child
    // inherits, is a live cell that `build_agent` rewrites on every rebuild (see `ParentLoop`).
    // `None` when no agent definitions were discovered: no delegable agents, no `subagent` tool.
    // `mut` because `reload` rebuilds it when trust or the on-disk definitions change and then rebuilds
    // the agent, so a mid-session-added agent becomes delegable immediately.
    let subagent_ctx: Option<Arc<crate::tools::subagent::SubagentCtx>> = if cfg.agents.is_empty() {
        None
    } else {
        Some(build_subagent_ctx(
            exec_cell,
            cfg,
            cwd,
            project_trusted,
            &current_model,
            &write_locks,
            skills,
            mode.context_files(context_files.as_slice()),
            mounts.to_vec(),
            approval.as_ref(),
            mcp_enabled,
        ))
    };
    // A multi-step run (several tool round-trips) is otherwise only ever durable once it *fully*
    // completes — a crash, OOM-kill, or panic mid-run loses everything back to the turn's start,
    // including the user's own prompt. `ChannelCheckpoint` streams a cheap `Arc` snapshot of
    // `session.messages` through this channel at each durable mid-run point (see
    // `agent_core::CheckpointHook`'s doc comment for exactly which points those are); the `"prompt"`
    // arm's own busy-loop below drains it and persists incrementally, off the async task via
    // `spawn_blocking` like every other write in this file. The channel — not a `Mutex` around
    // `Persistence` itself — is what lets the checkpoint hook (called deep inside `agent.
    // run_events_steered`, which holds `&mut session` for the run's whole duration) reach persistence
    // without ever needing to borrow `session` a second time.
    let (checkpoint_tx, checkpoint_rx) = mpsc::unbounded_channel::<Checkpoint>();
    let checkpoint: Arc<dyn agent_core::CheckpointHook> =
        Arc::new(ChannelCheckpoint(checkpoint_tx));
    // SEP-2663 journal writes, `(kind, data)` (see `crate::mcp_resume`): task records from the
    // run's event sink and from the background resumer, results from the resumer. Drained by the
    // idle loop, the busy loop and at run end; `data.sessionId` must match the session persisted to.
    let (mcp_task_tx, mcp_task_rx) = mpsc::unbounded_channel::<(String, Value)>();
    // Each local `bash` job's processes are journaled through the same channel, stamped with the
    // session they belong to (re-pointed on a transcript change below).
    bash_jobs.set_recorder(Some(procs_recorder(&mcp_task_tx, persistence.session_id())));
    // Resolved *here*, against `current_model` (the actual starting model, after the anti-bleed
    // session-restore check above may have overridden `cfg.model`) rather than earlier against
    // `cfg.model` directly — a credential/routing resolved for the wrong (pre-restore) model would
    // silently misroute the very first turn. Kept in an `Arc` we can clone into `build_agent`, and
    // swapped out (not just the `Agent` rebuilt) by `set_model`/`cycle_model` and any other command
    // that changes `current_model` at runtime — see `build_gateway_client`'s own doc comment.
    let client = Arc::new(build_gateway_client(cfg, &current_model)?);
    let agent = build_agent(
        exec_cell,
        client.clone(),
        &full_system(static_system, cwd, mcp_skills),
        cfg,
        &current_model,
        current_thinking,
        current_level,
        current_auto_compaction,
        current_auto_retry,
        current_block_images,
        current_image_auto_resize,
        persistence.session_id(),
        &write_locks,
        &checkpoint,
        subagent_ctx.as_ref(),
        structured_output.as_ref(),
        memory_tool.as_ref(),
        approval.as_ref(),
        mcp_enabled,
        skills,
    );
    timing.mark("build agent");

    Ok(AgentSetup {
        run_retry_policy,
        current_model,
        scoped_models,
        cycle_models,
        current_thinking,
        starting_level,
        current_level,
        settings_store,
        current_auto_compaction,
        current_auto_retry,
        current_block_images,
        current_image_auto_resize,
        write_locks,
        subagent_ctx,
        checkpoint_rx,
        checkpoint,
        mcp_task_tx,
        mcp_task_rx,
        client,
        agent,
    })
}

/// The runtime state around the agent: steering, the host `bash` tool, the writer task, the MCP
/// resumer and the loop's bookkeeping.
struct RuntimeSetup {
    steering: agent_core::Steering,
    steer_done_rx: mpsc::UnboundedReceiver<SteerExpanded>,
    pending_steers: PendingSteers,
    bash_tool: Option<Arc<dyn agent_core::Tool>>,
    bash_policy: crate::policy::ToolPolicy,
    host_bash_seq: u32,
    out_tx: OutTx,
    writer: tokio::task::JoinHandle<()>,
    mcp_resumer: crate::mcp_resume::Resumer,
    last_history: Arc<Vec<agent_core::Message>>,
    app_state_for: Option<std::path::PathBuf>,
    saved_app_context: String,
    saved_app_views: u64,
    shutdown_cause: Option<Signal>,
    deferred_events: std::collections::VecDeque<String>,
    events_state_file: Option<std::path::PathBuf>,
    events_cmds: Option<crate::tools::mcp_events::McpEventsCommands>,
    pending_login: Arc<std::sync::Mutex<Option<PendingLogin>>>,
    login_generation: u64,
    shutdown: ShutdownSignal,
}

/// The runtime state around the agent: steering, the host `bash` tool, the writer task, the MCP
/// resumer and the loop's bookkeeping.
#[allow(clippy::too_many_arguments)]
async fn setup_runtime(
    cfg: &ServeConfig,
    out_conn: &SharedOutConn,
    timing: &mut crate::timing::StartupTiming,
    persistence: &Persistence,
    session: &Session,
    mode: &Mode,
    exec_cell: &crate::exec_endpoint::ExecCell,
    host_bash_jobs: &Arc<crate::tools::bash_jobs::BashJobs>,
    mcp_enabled: &crate::tools::mcp::McpEnabledSet,
    mcp_host: &Arc<crate::tools::mcp_host::McpHost>,
    mcp_events: &Option<crate::tools::mcp_events::McpEventsHub>,
    skills: &[crate::skills::Skill],
    mcp_task_tx: &mpsc::UnboundedSender<(String, Value)>,
) -> RuntimeSetup {
    // Persistent across every `prompt` call (not just the one currently in flight), so `steer`/
    // `follow_up` sent while idle queue for the *next* `prompt` instead of being rejected as an unknown
    // command. Cleared on every session switch (`new_session`/`switch_session`/`fork`/`switch_branch`)
    // so a message meant for the old session's next turn can't leak into the newly switched-to one.
    let steering = agent_core::Steering::new();
    // An MCP App's `ui/update-model-context` sent while a run is in flight reaches the model at the
    // run's next turn boundary (and rides a steered/follow-up turn), not only the next idle prompt.
    {
        let enabled = mcp_enabled.clone();
        steering.set_turn_context(Some(Arc::new(move || {
            crate::tools::mcp_apps::take_undelivered_context(&enabled)
        })));
    }
    // Steers/follow-ups received mid-run, in order, until queued — see `PendingSteers`.
    let (steer_done_tx, steer_done_rx) = mpsc::unbounded_channel::<SteerExpanded>();
    let pending_steers = PendingSteers::new(steer_done_tx);
    // Task 1 (pi-parity fix, pass 19): the persisted `agent settings --steering-mode`/
    // `--follow-up-mode` defaults (or an explicit `serve --steering-mode`/`--follow-up-mode` flag),
    // already resolved by `main.rs` into `cfg.steering_mode`/`cfg.follow_up_mode` — previously only
    // `run_task` applied its own equivalent resolution to its one-shot `Steering`; a `serve` process
    // silently ignored both and always started at `QueueMode::default()` regardless of what was
    // configured, until `set_steering_mode`/`set_follow_up_mode` was called at runtime. The `set_*` RPC
    // commands still change this afterward, same as before.
    steering.set_steering_mode(cfg.steering_mode);
    steering.set_follow_up_mode(cfg.follow_up_mode);
    // The `bash` tool, looked up once from the same (possibly filtered) registry `build_agent` builds —
    // `None` when `bash` was excluded (`--exclude-tools bash` / `--no-tools`), which the `bash` RPC
    // command below reports as a clean error rather than a side door around that restriction. A direct
    // `Arc<dyn Tool>` handle, not routed through `agent`/the model loop: the host `bash` RPC command runs
    // independent of any conversation turn.
    let bash_tool = build_tools(
        cfg,
        cfg.image_auto_resize,
        exec_cell,
        mcp_enabled,
        skills,
        Some(host_bash_jobs),
    )
    .get("bash");
    // The same deny-list `build_agent` installs as an `AgentHooks` gate on the model's own tool calls —
    // built once here rather than inline in the `bash` RPC arm below, since `--deny-tool`/
    // `--deny-bash-pattern`/`--deny-path` are fixed for the whole process (`build_agent`'s doc comment).
    // Track L23 (pi-parity fix): the host `bash` RPC command used to call `tool.run_streaming` directly,
    // never consulting this policy at all — `--exclude-tools bash` already gated it (`bash_tool` above
    // is `None`), but `--deny-tool`/`--deny-bash-pattern`/`--deny-path` didn't, a side door around an
    // operator's own restriction for any client speaking the wire protocol directly instead of through
    // the model.
    // Rooted/worlded like `build_agent`'s own gate, for the same reason: the RPC `bash` runs through
    // the same registry the model's calls do, so the policy must resolve paths the same way.
    let bash_policy = crate::policy::ToolPolicy::from_lists(
        &cfg.deny_tool,
        &cfg.deny_bash_pattern,
        &cfg.deny_path,
    )
    .with_root(tool_root(cfg))
    .in_world(exec_cell.backend().world());
    // Distinguishes successive host `bash` calls in their `tool_start`/`tool_progress`/`tool_end` event
    // ids — only ever one in flight at a time (see the `bash` command arm), but a stable, incrementing
    // id per call still lets a client correlate a run's own three events without ambiguity.
    let host_bash_seq: u32 = 0;

    // One writer task drains the FIFO and forwards each frame to whatever connection is currently
    // attached (`out_conn`); every frame (events + responses) passes through it in order, so output
    // never interleaves. The actual serialize + socket/stdout write happens in the connection's own
    // consumer (the stdio stdout task, or a WebSocket send task) — this task only preserves ordering
    // and bridges to the swappable tail.
    //
    // The channel is intentionally unbounded. The event `sink` (see `Agent::run_events`) is a
    // synchronous `FnMut`, so the producer cannot `.await` to apply backpressure; a bounded channel
    // would force `try_send`, which silently drops frames and corrupts the event stream — unacceptable
    // for a protocol. In practice the backlog is bounded by one in-flight turn's events, drained as
    // fast as the attached connection accepts.
    //
    // Unlike the old stdout-owning writer, this task does NOT tear down when the connection breaks: a
    // detached session (no connection, or a dead one) simply drops frames and keeps running, which is
    // exactly what lets a WebSocket client reconnect and re-attach to a still-live run. The task ends
    // only when every `out_tx` clone is dropped at teardown (`drop(out_tx)` + the joined run).
    let (raw_tx, mut out_rx) = mpsc::unbounded_channel::<OutFrame>();
    let out_tx = OutTx::new(raw_tx, mode.clone());
    let writer = {
        let out_conn = out_conn.clone();
        tokio::spawn(async move {
            while let Some(frame) = out_rx.recv().await {
                // Broadcast to every attached connection (see `OutFanout`); if none are attached the
                // frame is dropped by design. Lock is held only across the non-`await` sends.
                lock_ignoring_poison(&out_conn).broadcast(frame);
            }
        })
    };

    let mcp_resumer = start_mcp_resumer(
        persistence,
        session,
        cfg,
        &out_tx,
        mcp_host,
        mcp_task_tx,
        &[],
    );

    // The base transcript the *next* connection to attach is caught up with (see
    // [`OutFanout::add_with_catchup`]), published from the top of the idle loop — which covers *every*
    // command that mutates it (a completed run, `compact`, `fork`, `switch_session`, a branch switch, …)
    // without having to find and annotate each one. Miss a site and a reconnecting client silently gets a
    // stale backlog, which is the very class of bug this exists to fix.
    //
    // Deliberately **not** republished mid-run, even though the checkpoint arm has a perfectly good
    // snapshot in hand: the catch-up's contract is `history` = the transcript as of the run's start, and
    // `OutFanout::turn` = every frame since. That is what reconstructs a never-dropped client's view
    // exactly. Advance `history` at each mid-run checkpoint and a message committed mid-turn would land
    // in the catch-up *and* again in the replayed frames, and the client would render it twice.
    //
    // Cheap enough to call every pass: `Session::messages` is an `Arc`, so an unchanged transcript costs
    // one pointer comparison. Only a genuine change pays for `active_ids()`.
    let last_history: Arc<Vec<agent_core::Message>> = Arc::new(Vec::new());
    // MCP App view state kept beside the session (see `mcp_apps::Sidecar`): the attached view
    // context and the view replay store. Loaded when the session here changes (startup, a switch, a
    // fork or clone) and rewritten when either moves, so both survive a daemon restart — sealed with
    // the tenant's key in service mode, never in the clear. `app_state_for` is the session file they
    // were last loaded for.
    let app_state_for: Option<std::path::PathBuf> = None;
    let saved_app_context = String::from("[]");
    let saved_app_views: u64 = 0;

    timing.print();

    // Task #41 (pi-parity fix): which signal (if any) actually triggered shutdown, so the caller can
    // exit with the matching POSIX code (`Signal::exit_code`) instead of always `0` — every graceful
    // path previously returned bare `Ok(())` with no way to tell a clean stdin-EOF apart from a real
    // SIGTERM/SIGHUP/Ctrl-C. Set from whichever of this function's several `shutdown.wait()` sites
    // actually fires (there's no concurrent access to guard against — this is all one async task's own
    // sequential control flow, just interleaved via `select!`, so a bare local suffices).
    let shutdown_cause: Option<Signal> = None;
    // MCP Events injections that arrived while the loop was busy with something that cannot take a
    // prompt (a host `bash`, a branch summary, a retry backoff): replayed, in order, the moment it is
    // idle again — never refused as busy. See `mcp_events::is_injection`.
    let deferred_events: std::collections::VecDeque<String> = std::collections::VecDeque::new();
    let events_state_file = persistence.session_file().map(std::path::Path::to_path_buf);
    // `mcp_events_*` commands run as spawned tasks, accepted from every loop, idle or busy; their
    // responses go out through a *weak* handle on the writer so a straggler cannot hold teardown.
    let events_cmds = mcp_events.as_ref().map(|hub| {
        let tx = out_tx.downgrade();
        hub.commands(Arc::new(move |frame: Value| {
            if let Some(tx) = tx.upgrade() {
                let _ = tx.send(OutFrame::Value(frame));
            }
        }))
    });
    // At most one `login` in flight at a time, tracked here so a concurrent second `login` is
    // rejected and `submit_code`/`abort_login` know what to reach. Shared with the detached task
    // `login` spawns (see that arm below) — cleared back to `None` by that task itself once the
    // flow resolves, success or failure, so a later `login` is accepted again.
    let pending_login: Arc<std::sync::Mutex<Option<PendingLogin>>> =
        Arc::new(std::sync::Mutex::new(None));
    // Monotonic tag stamped on each login that claims the slot, so a late-winding-down task's guard can
    // tell its own slot from a successor's (see `PendingLoginGuard`). Only ever touched from this
    // single-threaded command loop, so a plain counter suffices.
    let login_generation: u64 = 0;

    // Replaced by the real signal handler once readiness has been announced (see `serve_session`).
    let shutdown = ShutdownSignal::inert();

    RuntimeSetup {
        steering,
        steer_done_rx,
        pending_steers,
        bash_tool,
        bash_policy,
        host_bash_seq,
        out_tx,
        writer,
        mcp_resumer,
        last_history,
        app_state_for,
        saved_app_context,
        saved_app_views,
        shutdown_cause,
        deferred_events,
        events_state_file,
        events_cmds,
        pending_login,
        login_generation,
        shutdown,
    }
}

impl SessionLoop {
    /// Everything a session does before its first command: opens persistence, discovers resources,
    /// builds the system prompt, tools and agent, and starts the writer task.
    pub(super) async fn new(
        cfg: ServeConfig,
        input_rx: mpsc::Receiver<String>,
        out_conn: SharedOutConn,
        // Set `true` for the duration of a `prompt` run, `false` otherwise. The daemon supervisor's idle
        // reaper reads this to never reap a session with an in-flight background run (see
        // [`crate::serve_ws`]); the stdio wrapper passes a throwaway it never observes.
        running: Arc<std::sync::atomic::AtomicBool>,
        // Set by the MCP Events hub while this session holds a live subscription; the daemon's idle
        // reaper leaves such a session alone (unless `--mcp-events-reapable`). Inert over stdio.
        keep_alive: Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<Self, SetupError> {
        let mut timing = crate::timing::StartupTiming::new();
        let Opened {
            mut cfg,
            persistence,
            session,
            title_attempted_for,
        } = open_session(cfg, &mut timing).await?;
        let ExecSetup {
            input_rx,
            mode,
            cwd,
            disk_overrides,
            project_trusted,
            exec_cell,
            bash_jobs,
            bash_jobs_wake,
            host_bash_jobs,
            bash_jobs_transcript,
            mcp_enabled,
            mcp_host,
            mcp_events,
            mcp_skills,
            sandbox_snapshot,
            job_notify,
        } = setup_exec(
            &mut cfg,
            input_rx,
            &out_conn,
            &running,
            &keep_alive,
            &mut timing,
            &persistence,
        )
        .await?;
        let PromptSetup {
            sandbox_refresh,
            sandbox_snapshot_dirty,
            sandbox_resources,
            skills,
            skill_collisions,
            prompt_templates,
            prompt_collisions,
            context_files,
            sandbox_system,
            sandbox_append,
            has_read,
            has_todo,
            has_memory,
            session_dir_cell,
            mounts,
            memory_tool,
            session_memory_active,
            memory_sections,
            apps_epoch,
            pending_approvals,
            approval,
            pending_elicitations,
            pending_samplings,
            app_sink,
            app_redact,
            app_detached,
            app_filter,
            apps_generation,
            mcp_tools_generation,
            mcp_filter,
            output_slot,
            structured_output,
            current_output_spec,
            static_system,
        } = setup_prompt(
            &mut cfg,
            &out_conn,
            &mut timing,
            &persistence,
            &mode,
            &cwd,
            disk_overrides,
            project_trusted,
            &exec_cell,
            &mcp_enabled,
            &mcp_host,
            &mcp_events,
            sandbox_snapshot,
        )
        .await?;
        let AgentSetup {
            run_retry_policy,
            current_model,
            scoped_models,
            cycle_models,
            current_thinking,
            starting_level,
            current_level,
            settings_store,
            current_auto_compaction,
            current_auto_retry,
            current_block_images,
            current_image_auto_resize,
            write_locks,
            subagent_ctx,
            checkpoint_rx,
            checkpoint,
            mcp_task_tx,
            mcp_task_rx,
            client,
            agent,
        } = setup_agent(
            &cfg,
            &mut timing,
            &persistence,
            &mode,
            &cwd,
            project_trusted,
            &exec_cell,
            &bash_jobs,
            &mcp_enabled,
            &mcp_skills,
            &skills,
            &context_files,
            &mounts,
            &memory_tool,
            &approval,
            &structured_output,
            &static_system,
        )
        .await?;
        let RuntimeSetup {
            steering,
            steer_done_rx,
            pending_steers,
            bash_tool,
            bash_policy,
            host_bash_seq,
            out_tx,
            writer,
            mcp_resumer,
            last_history,
            app_state_for,
            saved_app_context,
            saved_app_views,
            shutdown_cause,
            deferred_events,
            events_state_file,
            events_cmds,
            pending_login,
            login_generation,
            shutdown,
        } = setup_runtime(
            &cfg,
            &out_conn,
            &mut timing,
            &persistence,
            &session,
            &mode,
            &exec_cell,
            &host_bash_jobs,
            &mcp_enabled,
            &mcp_host,
            &mcp_events,
            &skills,
            &mcp_task_tx,
        )
        .await;

        Ok(Self {
            cfg,
            persistence,
            session,
            title_attempted_for,
            mode,
            cwd,
            disk_overrides,
            project_trusted,
            exec_cell,
            bash_jobs,
            bash_jobs_wake,
            host_bash_jobs,
            bash_jobs_transcript,
            mcp_enabled,
            mcp_host,
            mcp_events,
            mcp_skills,
            job_notify,
            sandbox_refresh,
            sandbox_snapshot_dirty,
            sandbox_resources,
            skills,
            skill_collisions,
            prompt_templates,
            prompt_collisions,
            context_files,
            sandbox_system,
            sandbox_append,
            has_read,
            has_todo,
            has_memory,
            session_dir_cell,
            mounts,
            memory_tool,
            session_memory_active,
            memory_sections,
            apps_epoch,
            pending_approvals,
            approval,
            pending_elicitations,
            pending_samplings,
            app_sink,
            app_redact,
            app_detached,
            app_filter,
            apps_generation,
            mcp_tools_generation,
            mcp_filter,
            output_slot,
            structured_output,
            current_output_spec,
            static_system,
            run_retry_policy,
            current_model,
            scoped_models,
            cycle_models,
            current_thinking,
            starting_level,
            current_level,
            settings_store,
            current_auto_compaction,
            current_auto_retry,
            current_block_images,
            current_image_auto_resize,
            write_locks,
            subagent_ctx,
            checkpoint_rx,
            checkpoint,
            mcp_task_tx,
            mcp_task_rx,
            client,
            agent,
            steering,
            steer_done_rx,
            pending_steers,
            bash_tool,
            bash_policy,
            host_bash_seq,
            out_tx,
            writer,
            mcp_resumer,
            last_history,
            app_state_for,
            saved_app_context,
            saved_app_views,
            shutdown,
            shutdown_cause,
            deferred_events,
            events_state_file,
            events_cmds,
            pending_login,
            login_generation,
            input_rx,
            out_conn,
            running,
        })
    }
}
