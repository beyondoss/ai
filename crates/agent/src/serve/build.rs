//! Building the pieces a session runs on: the gateway client, the tool registry, the agent, the subagent context
//! and the discovered resources.

use crate::serve::*;

/// The extra per-request headers a `models.json` override configures for this model id, if any (Task
/// #11 pi-parity feature) — the lib-crate-side twin of `main.rs::model_override_extra_headers`. Not
/// literally shared with it: `main.rs` is a separate binary crate over this library (see
/// `beyond-ai-agent`'s `Cargo.toml` `[[bin]]` target), so a `pub(crate)` there still wouldn't be
/// visible here, and vice versa — the two must each call through to
/// `settings::ModelOverrides::open_default()`/`ModelOverride::resolved_headers` (the one real shared
/// primitive both live in this same library crate and already depend on) rather than one calling the
/// other. Kept to the same trivial one-lookup shape as `main.rs`'s copy so the two can't drift on
/// anything but this wrapper itself.
pub(crate) fn model_override_extra_headers(
    model: &str,
) -> std::collections::HashMap<String, String> {
    crate::settings::ModelOverrides::open_default()
        .get(model)
        .map(|over| over.resolved_headers())
        .unwrap_or_default()
}

/// Resolve `model`'s gateway credential/routing ([`resolve_gateway_credential`]) and build a
/// ready-to-use [`GatewayClient`] for it, applying the process's fixed retry policy. Called once at
/// `serve` startup and again every time the active model changes at runtime (`set_model`,
/// `cycle_model`, and `switch_session`/`fork`/`clone`/`switch_branch` restoring a session's own
/// recorded model) — never cached across a model switch, since the resolved provider/routing can be
/// completely different from one model to the next (an OAuth login resolved for an Anthropic model has
/// no bearing on a GitHub Copilot one). This is the fix for the credential/routing having previously
/// been resolved exactly once, before `serve` even started, and then silently reused — stale provider,
/// stale routing — by every later model switch for the rest of the process's life.
/// **Service mode takes the credential from the grant and nothing else.** `resolve_gateway_credential`
/// is a ladder of host sources — `--key`, `AI_AGENT_KEY`, the ambient provider env
/// (`ANTHROPIC_API_KEY`, …), a `models.json` override, the operator's OAuth store — and every rung
/// of it belongs to the replica, not to the tenant. So the whole ladder (and the `models.json`
/// headers below it) is skipped: the grant's sealed `gateway_key` is the session's only credential,
/// and if it is wrong the session fails rather than silently billing whoever the replica is logged
/// in as. The retry / shared-client / idle-timeout chain is unchanged either way.
pub(super) fn build_gateway_client(
    cfg: &ServeConfig,
    model: &str,
) -> Result<GatewayClient, String> {
    let mode = cfg.mode();
    let client = match mode.gateway_key() {
        Some(key) => GatewayClient::new(cfg.gateway.clone(), key).map_err(|e| e.to_string())?,
        None => match resolve_gateway_credential(cfg.key.clone(), model, &cfg.provider_env)? {
            GatewayCredential::Static(key) => {
                GatewayClient::new(cfg.gateway.clone(), key).map_err(|e| e.to_string())?
            }
            GatewayCredential::Oauth(source) => {
                GatewayClient::with_credential_source(cfg.gateway.clone(), source)
                    .map_err(|e| e.to_string())?
            }
        },
    }
    .with_retry(
        cfg.retry_max_retries
            .unwrap_or(agent_core::client::MAX_RETRIES),
        cfg.retry_base_delay_ms
            .unwrap_or(agent_core::client::BASE_BACKOFF),
    )
    // pi-parity (Task 2, serve pass 19): `commands/run.rs::run_task`'s identical `with_extra_headers` wiring
    // (Task #11) had no `serve` counterpart — every `serve` entrypoint that (re)builds a gateway client
    // (startup, `set_model`, `cycle_model`, `fork`, `clone`, `switch_session`, `switch_branch`) went
    // through this one function, so a `models.json` `ModelOverride.headers` override (and the
    // auto-seeded NVIDIA `NVCF-POLL-SECONDS` / Kimi-Coding `User-Agent` defaults —
    // `ModelOverride::resolved_headers`'s own doc comment) silently never reached the wire under
    // `serve`, even though it did under `run`. Fixed once, here, rather than at each of those 8+ call
    // sites: chained right after `with_retry`, matching `run_task`'s own ordering — harmless (a no-op)
    // when no override configures any headers, since an empty map is also `GatewayClient::new`'s own
    // default.
    // Empty in service mode: `models.json` is the replica operator's own file.
    .with_extra_headers(mode.extra_headers(model));
    // Task #30 (pi-parity feature): `run`'s identical `--retry-max-backoff-ms` wiring (`main.rs::
    // run_task`'s own `with_max_backoff` call site) previously had no `serve` counterpart — called on
    // every model switch, same as `with_retry` above, so the override survives a mid-run switch.
    let client = if let Some(max_backoff) = cfg.retry_max_backoff_ms {
        client.with_max_backoff(max_backoff)
    } else {
        client
    };
    // W3 (shared upstream pool): inject the daemon-wide `reqwest::Client` so every session multiplexes
    // over one connection pool instead of each opening its own. **Must** precede `with_idle_timeout`
    // below: `with_http_client` sets the `http_shared` guard, which makes that later call a no-op rather
    // than rebuilding (and thereby tearing down) the shared pool on a model switch or idle override. The
    // shared client's own read timeout is fixed when the supervisor builds it (see
    // `serve_ws::build_shared_h2_client`). `None` (stdio/`run`, or `--upstream-http2 off`) is unchanged.
    let client = match &cfg.shared_http {
        Some(h) => client.with_http_client(h.clone()),
        None => client,
    };
    // Task #38 (pi-parity fix): `run`'s identical `--idle-timeout-ms` wiring (`commands/run.rs::run_task`'s own
    // `with_idle_timeout` call site) previously had no `serve` counterpart — this is called on every
    // model switch (`set_model`/`cycle_model` rebuild the whole client), same as `with_retry` above, so
    // the override survives a mid-run switch rather than only applying at startup.
    let client = if let Some(ms) = cfg.idle_timeout_ms {
        client
            .with_idle_timeout(std::time::Duration::from_millis(ms))
            .map_err(|e| e.to_string())?
    } else {
        client
    };
    Ok(client)
}

/// Build the subagent context once, at startup. Its transport factory closes over a clone of `cfg` and
/// calls [`build_gateway_client`] per child model — the same model-switch-safe recipe the parent uses —
/// so the ctx is valid for the whole session regardless of parent model switches.
///
/// `skills` are the parent session's own discovered skills, so a worktree/read-only child sees the same
/// `<available_skills>` the parent does. `prompt_guidelines` is empty: `serve` renders its base prompt
/// once into `cfg.system` and doesn't carry the raw guideline list past that.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_subagent_ctx(
    exec: &crate::exec_endpoint::ExecCell,
    cfg: &ServeConfig,
    cwd: &Path,
    project_trusted: bool,
    parent_model: &str,
    write_locks: &Arc<agent_core::WriteLockRegistry>,
    skills: &[crate::skills::Skill],
    // The parent's own project-instruction files when it fetched them through a backend (service
    // mode), so a child reuses them instead of walking a `cwd` that isn't on this machine. `None`
    // leaves each fan-out walking the host `cwd`, as before. See `SubagentCtx::context_files`.
    context_files: Option<&[(String, String)]>,
    // The parent's memory mounts, shared by `Arc` clone so the whole subagent tree reads/writes the very
    // same stores — including the `/session` cell that re-points on a session switch. See
    // `SubagentCtx::memory_mounts`.
    memory_mounts: Vec<crate::memory::Mount>,
    // Shared with the parent, not rebuilt — see `SubagentCtx::approval`.
    approval: Option<&crate::approval::ApprovalRuntime>,
    // The parent's live MCP kit gate, shared rather than snapshotted — see `SubagentCtx::mcp_enabled`.
    mcp_enabled: &crate::tools::mcp::McpEnabledSet,
) -> Arc<crate::tools::subagent::SubagentCtx> {
    use crate::tools::subagent;
    let cfg_for_factory = cfg.clone();
    let factory: subagent::TransportFactory = Arc::new(move |m: &str| {
        build_gateway_client(&cfg_for_factory, m)
            .map(|c| Arc::new(c) as Arc<dyn agent_core::ModelTransport>)
    });
    // The parent's effective set (after `--tools`/`--exclude-tools`) minus `subagent`, so a child with
    // no `tools:` of its own inherits exactly what the parent may do — no more.
    let mut probe = build_tools(
        cfg,
        cfg.image_auto_resize,
        exec,
        &crate::tools::mcp::McpEnabledSet::new(),
        skills,
        None,
    );
    crate::tools::apply_filter(
        &mut probe,
        cfg.tools.as_deref(),
        cfg.exclude_tools.as_deref(),
        cfg.no_tools,
    );
    let parent_tools: Vec<String> = probe
        .definitions()
        .into_iter()
        .map(|d| d.name)
        .filter(|n| n != subagent::NAME)
        .collect();
    Arc::new(subagent::SubagentCtx {
        factory,
        agents: Arc::new(cfg.agents.clone()),
        skills: Arc::new(skills.to_vec()),
        write_locks: write_locks.clone(),
        mcp_tools: cfg.mcp_tools.clone(),
        mcp_enabled: mcp_enabled.clone(),
        memory_mounts,
        tool_cfg: subagent::ChildToolConfig {
            bash_shell_path: cfg.bash_shell_path.clone(),
            bash_command_prefix: cfg.bash_command_prefix.clone(),
            remote_shell: cfg.mode().remote_shell(),
            web_allow_private: cfg.web_allow_private,
            web_allow_hosts: cfg.web_allow_hosts.clone(),
            web_timeout_ms: cfg.web_timeout_ms,
            image_auto_resize: cfg.image_auto_resize,
            // The **cell**, not a snapshot: a child must act on whatever machine its parent is
            // pointed at *now*. Handing down `None` here was a sandbox escape — the parent's tools
            // ran in the sandbox while any `subagent` it spawned ran on the host, so the model could
            // reach around the isolation simply by delegating.
            fs_backend: Some(cfg.mode().bundle_fs().unwrap_or_else(|| exec.backend())),
            command_runner: Some(exec.runner()),
            code_mode: cfg.code_mode,
            mcp_loading: cfg.mcp_loading,
            exclude_tools: cfg.exclude_tools.clone().unwrap_or_default(),
        },
        cwd: cwd.to_path_buf(),
        context_files: context_files.map(<[(String, String)]>::to_vec),
        include_context_files: cfg.context_files,
        project_trusted,
        // A child is as fail-closed about the replica's filesystem as its parent (see
        // `PromptOptions::disk_overrides`) — otherwise delegating would be a way around it.
        disk_overrides: cfg.mode().reads_disk_overrides(),
        prompt_guidelines: Vec::new(),
        // `build_agent` writes the parent's model and loop settings here on every (re)build, the first
        // one included, before any child can be dispatched.
        parent: subagent::ParentCell::default(),
        parent_cache_key: parent_model.to_string(),
        parent_tools,
        deny_tool: cfg.deny_tool.clone(),
        deny_bash_pattern: cfg.deny_bash_pattern.clone(),
        deny_path: cfg.deny_path.clone(),
        max_depth: subagent::DEFAULT_MAX_DEPTH,
        approval: approval.cloned(),
    })
}

/// Build the [`Agent`] for the current model + thinking budget + auto-compaction/auto-retry flags.
/// Called once at startup and again on every `set_model`/`set_thinking`/`cycle_model`/
/// `cycle_thinking_level`/`set_auto_compaction`/`set_auto_retry`, so a client can re-tune the run
/// without restarting `serve`. The transport, tools, system prompt, loop bounds, and cache settings are
/// the same each time; only the model id, thinking budget, and the two flags vary. Compaction's
/// `context_window` defaults to `model`'s own capabilities; an explicit `--context-window` overrides
/// that and stays pinned across a model switch (the operator's compaction budget, not the dialect's) —
/// left unset, each switch picks up the *new* model's real window instead of a stale operator number.
/// `reserve_tokens` defaults to `CompactionConfig::default()`'s and `keep_recent_tokens` to the model
/// family's (in real tokens, `agent_core::TokenFamily::keep_recent_tokens`), both overridable
/// independently of `context_window` — all resolved by [`compaction_config`].
// 11 arguments, all independent inputs every call site already has on hand from `cfg`/local
// runtime-switchable state — bundling them into a struct would just be a second place those same
// fields live, not a reduction in what the function needs to know (see `client.rs::send_with_retry`
// for the same tradeoff). Private, single-purpose helper, not a public API shape.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_agent(
    exec: &crate::exec_endpoint::ExecCell,
    transport: Arc<GatewayClient>,
    system: &str,
    cfg: &ServeConfig,
    model: &str,
    thinking: Option<u32>,
    level: agent_core::ThinkingLevel,
    auto_compaction: bool,
    auto_retry: bool,
    // Pi-parity fix (pass 20): live counterparts of `cfg.block_images`/`cfg.image_auto_resize` — see
    // `current_block_images`/`current_image_auto_resize`'s own doc comment in `serve` for why these are
    // now explicit, runtime-switchable parameters (mirroring `auto_compaction`/`auto_retry` just above)
    // instead of this function reading `cfg.block_images`/`cfg.image_auto_resize` straight off the
    // static startup config on every rebuild.
    block_images: bool,
    image_auto_resize: bool,
    cache_key: &str,
    write_locks: &Arc<agent_core::WriteLockRegistry>,
    checkpoint: &Arc<dyn agent_core::CheckpointHook>,
    subagent_ctx: Option<&Arc<crate::tools::subagent::SubagentCtx>>,
    // The `structured_output` tool for the schema currently installed on this session, if any (see the
    // `prompt` command's `output_schema`). An `Arc` shared across every rebuild rather than
    // reconstructed here: it owns the compiled validator, and — more importantly — the `OutputSlot` the
    // model's answer lands in, which must survive a mid-run `set_model` rebuild.
    structured_output: Option<&Arc<crate::tools::structured_output::StructuredOutput>>,
    // The `memory` tool for this session, when memory is enabled. An `Arc<dyn Tool>` shared across every
    // rebuild (like `structured_output`) so a mid-run `set_model` doesn't drop it — it wraps the
    // host-owned `MemoryBackend`, whose store must persist across a model switch.
    memory_tool: Option<&Arc<dyn agent_core::Tool>>,
    // The interactive approval gate, when `--approve` asked for one. Shared (not rebuilt) across every
    // `set_model` rebuild so the session's remembered "always allow" decisions — and the one
    // prompt-serializing lock the whole subagent tree queues behind — survive a model switch.
    approval: Option<&crate::approval::ApprovalRuntime>,
    // Session-scoped MCP server enablement (see `set_mcp_enabled`). Same lifetime as `exec`: per
    // `serve_session`, not on `ServeConfig`, so multi-tenant WS sessions cannot inherit each other's kit.
    mcp_enabled: &crate::tools::mcp::McpEnabledSet,
    // The session's discovered skills, so `read` resolves the `skill:<name>` paths the system
    // prompt's listing names them by.
    skills: &[crate::skills::Skill],
) -> Agent {
    let compaction = agent_core::CompactionConfig {
        enabled: auto_compaction,
        ..compaction_config(cfg, model)
    };

    let mut registry = build_tools(cfg, image_auto_resize, exec, mcp_enabled, skills, None);
    // Registered here, not in `build_tools`, because this is where `write_locks` is in scope — the ctx
    // needs the same registry every child shares. Reused across rebuilds; the ctx's factory handles the
    // per-child model itself, so a `set_model` rebuild doesn't invalidate it.
    // `level` (the portable ladder) supplies the default thinking/reasoning-effort pair for whichever
    // mechanism `model`'s capabilities actually call for; `thinking`, when present, is an explicit raw
    // budget override that wins over the level's own derived budget (but never touches reasoning
    // effort, which always comes from `level` — see `current_thinking`'s doc comment in `serve`).
    let (level_thinking, reasoning_effort) =
        agent_core::thinking_for_level(&agent_core::capabilities(model), level);
    let thinking = thinking.or(level_thinking);
    if let Some(ctx) = subagent_ctx {
        // Every model switch, thinking/compaction/image toggle, resume and branch lands here, so this
        // is the one place that keeps the long-lived ctx's view of the parent current: a child
        // dispatched after `set_model` inherits the model — and loop settings — the parent has *now*.
        ctx.parent.set(crate::tools::subagent::ParentLoop {
            model: model.to_string(),
            max_steps: cfg.max_steps,
            compaction: cfg.compaction,
            compaction_enabled: auto_compaction,
            thinking,
            reasoning_effort,
            block_images,
            rate_limit: crate::retry::rate_limit_policy(cfg.retry_rate_limit_secs),
        });
        registry.register(Arc::new(crate::tools::subagent::Subagent::new(ctx.clone())));
    }
    // After `build_tools`'s `--tools`/`--exclude-tools` filter, like `subagent` above: an allow-list
    // scopes what the agent may *do*, and must not strip the one tool a `prompt {output_schema}` exists
    // to add — leaving a run that can never satisfy the contract it was started with.
    if let Some(tool) = structured_output {
        registry.register(tool.clone());
    }
    // Persistent memory, registered after the filter for the same reason as `structured_output`/
    // `subagent`: a `--tools` allow-list scopes what the agent may *do* and must not strip its memory.
    if let Some(tool) = memory_tool {
        registry.register(tool.clone());
    }
    let mut agent = Agent::new(transport, model.to_string())
        .with_tools(registry)
        .with_system(system.to_string())
        .with_max_steps(cfg.max_steps)
        .with_compaction(compaction)
        .with_auto_retry(auto_retry)
        .with_rate_limit_retry(crate::retry::rate_limit_policy(cfg.retry_rate_limit_secs))
        .with_sequential_tools(cfg.sequential_tools)
        // Pin this session to a warm prompt-cache node via its stable id.
        .with_cache_key(cache_key.to_string())
        .with_cache_long(cfg.cache_long)
        // Shared across every `build_agent` call for this process (including `set_model`/
        // `set_thinking` rebuilds), so file-mutation exclusivity survives a model switch and extends
        // to any other session sharing this same registry.
        .with_write_locks(write_locks.clone())
        // Streams a snapshot of `session.messages` through `checkpoint`'s channel at each durable
        // mid-run point (see `ChannelCheckpoint`), so a long multi-step turn is persisted incrementally
        // instead of only once it fully completes.
        .with_checkpoint_hook(checkpoint.clone());
    // Task #31 (pi-parity feature): `run`'s identical `--branch-summary-reserve-tokens` wiring
    // (`commands/run.rs::run_task`'s own `with_branch_summary_reserve_tokens` call site) previously had no
    // `serve` counterpart — `Agent::with_branch_summary_reserve_tokens` had zero callers in either
    // binary. Applied on every `build_agent` rebuild, like every other `cfg`-sourced override, so it
    // survives a model switch.
    if let Some(reserve) = cfg.branch_summary_reserve_tokens {
        agent = agent.with_branch_summary_reserve_tokens(reserve);
    }
    if let Some(budget) = thinking {
        agent = agent.with_thinking(budget);
    }
    if let Some(effort) = reasoning_effort {
        agent = agent.with_reasoning_effort(effort);
    }
    if let Some(temperature) = cfg.temperature {
        agent = agent.with_temperature(temperature);
    }
    if let Some(max_tokens) = cfg.max_tokens {
        agent = agent.with_max_tokens(max_tokens);
    }
    // Task #34 (pi-parity fix): forces every image down the vision-downgrade path regardless of the
    // active model's real `supports_vision` capability — `run`'s identical `--block-images` handling
    // (`commands/run.rs::run_task`'s own `with_block_images` call site), previously never threaded through here
    // at all. Pass 20 (pi-parity fix): reads the live `block_images` parameter (settable at runtime via
    // `set_block_images`), not `cfg.block_images` — see this function's own doc comment.
    if block_images {
        agent = agent.with_block_images(true);
    }
    // Rooted and worlded exactly like the registry just built (`build_tools`): the hook resolves a
    // `write`/`edit` path the same way the tool that runs it will, so `--deny-path` and a remembered
    // approval `scope_key` key on the path actually about to be written — a sandbox path, resolved
    // lexically against the sandbox's own `$HOME`, never `canonicalize`d on this host.
    let policy = crate::policy::ToolPolicy::from_lists(
        &cfg.deny_tool,
        &cfg.deny_bash_pattern,
        &cfg.deny_path,
    )
    .with_root(tool_root(cfg))
    .in_world(exec.backend().world());
    // Fix 9 (pi-parity gap): always installed now — unlike the bare `ToolPolicy` this used to install
    // only when non-empty, `ServeHooks::before_provider_request` (the `bash` RPC's
    // `exclude_from_context` filter) is needed unconditionally, and `Agent::with_hooks` only ever holds
    // one hook object. See `ServeHooks`'s own doc comment.
    agent = agent.with_hooks(Arc::new(ServeHooks {
        policy,
        approval: approval.cloned(),
        mcp: mcp_enabled.clone(),
    }));
    agent
}

/// The one [`AgentHooks`] implementation `build_agent` installs — composes the optional
/// `--deny-tool`/`--deny-bash-pattern`/`--deny-path` gate ([`crate::policy::ToolPolicy`]) with
/// [`before_provider_request`](AgentHooks::before_provider_request)'s own, unconditional job: filtering
/// [`HOST_BASH_EXCLUDED_LABEL`]-tagged messages out of the outgoing model request (Fix 9, pi-parity
/// gap). `session.messages`/persisted storage/`export.rs`'s rendered output all still carry the
/// message unchanged — only the wire payload a turn actually sends omits it, mirroring pi's
/// `convertToLlm` (`packages/coding-agent/src/core/messages.ts`), which performs the identical
/// late-and-separate filter rather than gating what `recordBashResult` commits to history in the first
/// place.
///
/// It also composes in the interactive approval gate ([`crate::approval`]) — `Agent::with_hooks` only
/// ever holds one hook object, so a second seam has to fold in here rather than replace this one.
pub(super) struct ServeHooks {
    pub(super) policy: crate::policy::ToolPolicy,
    /// `None` unless `--approve` asked for one. Static deny still wins first; see
    /// [`crate::approval::gated_before_tool_call`], whose decision order is load-bearing.
    pub(super) approval: Option<crate::approval::ApprovalRuntime>,
    /// The session's MCP state, for the SEP-2640 code-execution gate: while the session is acting on
    /// an MCP-served skill, `bash`/`execute` run only with the user's approval.
    pub(super) mcp: crate::tools::mcp::McpEnabledSet,
}

#[async_trait::async_trait]
impl AgentHooks for ServeHooks {
    async fn before_tool_call(
        &self,
        name: &str,
        input: &Value,
        session: &Session,
        cancel: &CancellationToken,
    ) -> Option<String> {
        let origin = crate::approval::ApprovalOrigin::Main;
        if let Some(reason) = crate::approval::gated_before_tool_call(
            &self.policy,
            self.approval.as_ref(),
            &origin,
            name,
            input,
            session,
            cancel,
        )
        .await
        {
            return Some(reason);
        }
        self.mcp
            .skill_session()
            .gate_tool_call(name, input, &origin, cancel)
            .await
    }

    async fn before_provider_request(&self, req: &mut agent_core::transport::ModelRequest) {
        if !req.messages.iter().any(is_excluded_from_model_context) {
            return;
        }
        let kept: Vec<agent_core::Message> = req
            .messages
            .iter()
            .filter(|m| !is_excluded_from_model_context(m))
            .cloned()
            .collect();
        req.messages = std::sync::Arc::new(kept);
    }
}

/// The tool registry after `--tools`/`--exclude-tools`/`--no-tools` filtering — shared by every
/// `build_agent` rebuild and by the host-level `bash` RPC command (see [`serve`]), so excluding `bash`
/// from the model's own tool set also disables the host command rather than leaving a side door open
/// around an operator's explicit restriction.
///
/// `image_auto_resize` is an explicit parameter, not read off `cfg` (pass 20, pi-parity fix): the only
/// call site where its value is actually load-bearing is `build_agent`'s own live-rebuildable registry
/// (see `set_image_auto_resize`), which passes the runtime-mutable `current_image_auto_resize` rather
/// than `cfg`'s frozen startup value; the other call sites here only ever check tool
/// presence/definitions, which don't depend on this flag either way, so they just pass
/// `cfg.image_auto_resize` straight through.
/// Everything a session's prompt is assembled from that has to be *found* rather than passed in.
/// One value, because session start and `reload` must produce exactly the same set or the two drift.
#[derive(Default)]
pub(super) struct Discovered {
    pub(super) agents: Vec<crate::agents::AgentDef>,
    pub(super) skills: Vec<crate::skills::Skill>,
    pub(super) skill_collisions: Vec<crate::skills::Collision>,
    pub(super) prompt_templates: Vec<crate::prompts::PromptTemplate>,
    pub(super) prompt_collisions: Vec<crate::skills::Collision>,
    /// `AGENTS.md`/`CLAUDE.md` as `(path, body)`, already read — empty when `--no-context-files`.
    pub(super) context_files: Vec<(String, String)>,
    /// A `SYSTEM.md`/`APPEND_SYSTEM.md` override, but only in service mode: on this host the two are
    /// still found inside `build_static_system_prompt` under `PromptOptions::disk_overrides`, where
    /// they have always been. See that field's doc comment for why service mode routes around it.
    pub(super) system: Option<String>,
    pub(super) append_system: Option<String>,
}

impl From<crate::service::Resources> for Discovered {
    fn from(r: crate::service::Resources) -> Self {
        Self {
            agents: r.agents,
            skills: r.skills,
            skill_collisions: r.skill_collisions,
            prompt_templates: r.prompt_templates,
            prompt_collisions: r.prompt_collisions,
            context_files: r.context_files,
            system: r.system,
            append_system: r.append_system,
        }
    }
}

/// Find this session's skills, prompt templates, agent definitions and project-instruction files.
///
/// The one seam startup and `reload` share, and the one place the two worlds diverge:
///
/// - **On this host** — the ordinary `serve` — each resource walks its own standard roots.
///   Prompt templates are gated on trust wholesale (an untrusted repo's `.claude/prompts` is
///   attacker-controlled instructions, and `/name` injects them straight into context). Skills are
///   gated only at the *project* root: `~/.claude/skills` is the operator's own machine, and an
///   untrusted checkout must not blank out the operator's own skills along with its own.
///   `--no-skills`/`--no-prompt-templates` skip standard-root discovery outright rather than
///   discovering and then discarding, while still honoring an explicit `--skill`/`--prompt-template`
///   path (pi's own `noSkills`/`noPromptTemplates` do the same — pi-parity fix, M2).
/// - **In service mode** — not here. Every one of them comes from the tenant's sandbox, through the
///   session's own exec backend ([`crate::service::ServiceSession::resources`], or the session's
///   saved [`crate::service::Snapshot`] of what that last found), which `serve_session` calls
///   directly; nothing here may read the replica — not its `~/.claude`, not its cwd, not a
///   `--skill` path pointing at either (both flags are refused at startup).
///
/// Collisions (a skill or `/name` shadowed across roots) come back either way, surfaced through
/// `get_commands` rather than silently resolved.
pub(super) async fn discover_resources(
    cfg: &ServeConfig,
    cwd: &Path,
    project_trusted: bool,
) -> Discovered {
    let (prompt_templates, prompt_collisions) = if cfg.no_prompt_templates {
        crate::prompts::discover_extra_only(&cfg.extra_prompt_template_paths)
    } else {
        crate::prompts::discover_with_diagnostics(
            cwd,
            project_trusted,
            &cfg.extra_prompt_template_paths,
        )
    };
    let (skills, skill_collisions) = if cfg.no_skills {
        crate::skills::discover_extra_only(&cfg.extra_skill_paths)
    } else {
        crate::skills::discover_with_diagnostics(cwd, project_trusted, &cfg.extra_skill_paths)
    };
    Discovered {
        agents: crate::agents::discover(cwd, project_trusted),
        skills,
        skill_collisions,
        prompt_templates,
        prompt_collisions,
        context_files: match cfg.context_files {
            true => crate::resources::load_context_files(cwd),
            false => Vec::new(),
        },
        system: None,
        append_system: None,
    }
}

/// The directory this session's filesystem tools resolve a relative path against, and its policy
/// checks a `write`/`edit` target against. The two must agree or the deny-list is evaluating a path
/// nothing is going to write, so both read it from here.
///
/// In service mode it is the grant's `workspace_root` — a path inside the sandbox this process never
/// stats. Empty otherwise, which [`tools::ToolConfig::root`] reads as "the process cwd": the
/// behavior every non-service caller had before roots existed.
pub(super) fn tool_root(cfg: &ServeConfig) -> std::path::PathBuf {
    cfg.mode().tool_root()
}

/// `host_bash_jobs`: `Some` only for the host `bash` command's own instance, which waits for every
/// command to exit (see [`tools::ToolConfig::bash_waits_for_exit`]) and parks what a command leaves
/// running in this table — the client's, not the model's (see `host_bash_jobs` in `serve_session`).
pub(super) fn build_tools(
    cfg: &ServeConfig,
    image_auto_resize: bool,
    exec: &crate::exec_endpoint::ExecCell,
    mcp_enabled: &crate::tools::mcp::McpEnabledSet,
    skills: &[crate::skills::Skill],
    host_bash_jobs: Option<&Arc<crate::tools::bash_jobs::BashJobs>>,
) -> agent_core::ToolRegistry {
    // `cfg.mcp_tools` (already-connected, namespaced `mcp__<server>__<tool>` entries — see
    // `tools::mcp`'s module doc comment) is merged in *before* `apply_filter` below, not after: an
    // operator's `--tools`/`--exclude-tools` allow/deny-list is meant to scope the *whole* tool set the
    // model sees, MCP-discovered tools included (e.g. excluding a misbehaving MCP tool by name), not
    // just the built-ins. Session-scoped [`McpEnabledSet`] further drops tools from disabled servers
    // before that merge — kit shaping without reconnecting.
    let mcp_tools = crate::tools::mcp::filter_by_enabled(&cfg.mcp_tools.snapshot(), mcp_enabled);
    let empty: &[String] = &[];
    let nested_exclude = cfg.exclude_tools.as_deref().unwrap_or(empty);
    let mut registry = tools::default_registry_with_config(&tools::ToolConfig {
        bash_waits_for_exit: host_bash_jobs.is_some(),
        bash_shell_path: cfg.bash_shell_path.as_deref(),
        bash_command_prefix: cfg.bash_command_prefix.as_deref(),
        web_allow_private: cfg.web_allow_private,
        web_allow_hosts: &cfg.web_allow_hosts,
        web_timeout_ms: cfg.web_timeout_ms,
        image_auto_resize,
        code_mode: cfg.code_mode,
        mcp_loading: cfg.mcp_loading,
        nested_exclude,
        nested_deny: &cfg.deny_tool,
        mcp_tools: &mcp_tools,
        // The cell, not a fixed target: `build_agent` is skipped on a same-model `switch_session`, so
        // a target captured here would leave the incoming session on the previous one's machine —
        // in a multi-tenant server, one tenant's tools inside another tenant's sandbox. The tools read
        // the cell per call, so a re-point cannot be missed by a rebuild that never happens.
        fs_backend: Some(cfg.mode().bundle_fs().unwrap_or_else(|| exec.backend())),
        command_runner: Some(exec.runner()),
        skills,
        // The session's, not this registry's: a rebuild (`set_model`) keeps the jobs, and the idle
        // session wakes on them. The host `bash` command's instance gets the client's table instead:
        // what a client's command leaves running is not the model's to be told about.
        bash_jobs: Some(
            host_bash_jobs
                .cloned()
                .unwrap_or_else(|| exec.bash_jobs().clone()),
        ),
        // In service mode a relative path the model writes belongs in the tenant's workspace, not
        // wherever this replica's process happens to have been started. Empty (= the process cwd)
        // for every other caller, which is the pre-service behavior.
        root: tool_root(cfg),
        // What the startup probe found on the far side (`bash` or `sh`). Ignored in the local world
        // and whenever `--bash-shell-path` is set; `None` falls back to `sh`, never a `bash` a
        // busybox sandbox would fail every command on.
        remote_shell: cfg.mode().remote_shell(),
    });
    tools::apply_filter(
        &mut registry,
        cfg.tools.as_deref(),
        cfg.exclude_tools.as_deref(),
        cfg.no_tools,
    );
    #[cfg(feature = "code-mode")]
    if cfg.code_mode {
        let nested = crate::tools::code_mode::select_deferred_tools(
            &mcp_tools,
            nested_exclude,
            &cfg.deny_tool,
        );
        crate::tools::code_mode::restore_execute(
            &mut registry,
            nested,
            cfg.exclude_tools.as_deref(),
            cfg.no_tools,
        );
    }
    registry
}
