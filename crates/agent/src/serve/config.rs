//! `ServeConfig`: everything a `serve` invocation is configured with, and the trust rule for a project.

use crate::serve::*;

/// How the daemon pools its upstream (agent→gateway) connections across sessions — the mode selected
/// by `--upstream-http2` (see [`crate::serve_ws::build_shared_h2_client`]).
///
/// The agent→gateway hop is plaintext, so `H2c` (HTTP/2 cleartext) is what actually collapses N
/// sessions onto ~one multiplexed connection — but it hard-requires a gateway that speaks h2c, so it
/// stays opt-in (see the variant docs). `Auto` is the daemon default: one shared `reqwest::Client`
/// (HTTP/1.1 pooling today), credentials still applied per-request on each session's
/// [`GatewayClient`]. `FromStr` backs the CLI flag's `value_parser`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UpstreamHttp2 {
    /// No shared client — every session (and every model switch) builds its own `reqwest::Client`.
    /// The opt-out for an operator who wants connection isolation at the cost of per-session pool RSS.
    Off,
    /// One shared client with **no** prior-knowledge: HTTP/1.1 connection pooling across all sessions
    /// today, transparently negotiating h2 if the hop later moves to `https://` with ALPN. Safe against
    /// an h1-only gateway (unlike [`Self::H2c`]). The daemon default.
    #[default]
    Auto,
    /// One shared client pinned to HTTP/2 cleartext (`http2_prior_knowledge`) — multiplexes every
    /// session over ~one TCP connection to the gateway. **Requires a gateway that accepts h2c**: against
    /// an h1-only gateway *every* request fails, which is why this is never the default.
    H2c,
}

impl std::str::FromStr for UpstreamHttp2 {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" => Ok(Self::Off),
            "auto" => Ok(Self::Auto),
            "h2c" => Ok(Self::H2c),
            other => Err(format!(
                "invalid --upstream-http2 {other:?} (expected one of: off, auto, h2c)"
            )),
        }
    }
}

/// Options for the headless server (mirrors `run`, plus persistence).
///
/// `Clone` so a WebSocket supervisor ([`crate::serve_ws`]) can hand each per-session task its own
/// copy with a distinct [`Self::session_id`] pinned in — `mcp_tools` clones the `Arc` (one live tool
/// set and its connections, shared), `shared_http` clones the daemon-wide pool, every other field is a plain owned value.
/// Tenant-private state (credentials on the per-session [`GatewayClient`], transcript, `/session`
/// memory, persistence, tools/`ExecCell`, approvals) is **not** in this struct; each `serve_session`
/// task builds those itself.
#[derive(Clone)]
pub struct ServeConfig {
    /// The gateway base URL. Still a plain `String` — `main.rs` resolves the `DEFAULT_GATEWAY` fallback
    /// into it as before — but in **direct** mode (see [`Self::provider_env`]) it is simply never dialed:
    /// every direct credential carries a `RouteOverride::Direct`, which replaces this outright at request
    /// time. Which is why making the gateway optional needed no new plumbing through here.
    pub gateway: String,
    /// The direct-routing environment (provider keys, `AI_PROVIDER`, `AI_BASE_URL`, and whether a gateway
    /// is configured at all), read once at the CLI boundary. Threaded alongside [`Self::key`] and for the
    /// same reason: [`resolve_gateway_credential`] is keyed on the *model*, so it is re-run on every model
    /// switch, and it needs this each time.
    pub provider_env: crate::gateway_credential::ProviderEnv,
    /// The raw `--key`/`AI_AGENT_KEY` value, if the operator gave one explicitly. Deliberately *not*
    /// pre-resolved into a [`GatewayCredential`] by `main.rs` before `serve` starts, unlike every other
    /// `ServeConfig` field: [`resolve_gateway_credential`] is keyed on the model, and this process's
    /// active model isn't fixed for its lifetime — a reattached `--session`/`--continue` session may
    /// override `model` with its own last-recorded one before the first turn even runs, and
    /// `set_model`/`cycle_model`/`switch_session`/`fork`/`clone`/`switch_branch` can all change it again
    /// at any point afterward. `crate::serve::build_gateway_client` calls `resolve_gateway_credential`
    /// fresh with this value every time the active model is (re)determined, rather than resolving once
    /// here and freezing whichever provider happened to be active at that moment.
    pub key: Option<String>,
    pub model: String,
    /// Whether the operator explicitly passed `--model` for *this* invocation, as opposed to `model`
    /// having fallen back to a stored `agent settings` default or this crate's own built-in default
    /// (`main.rs`'s `resolved_model` resolution) — Task #5 (pi-parity fix). `serve`'s own startup uses
    /// this to decide whether to prefer a reattached session's own last-recorded model
    /// (`Persistence::model_and_level_at_active`) over `model` when reopening an existing session: a
    /// merely-stored default must not override what that session was actually last running on, the same
    /// "bleed" `switch_session`/`switch_branch` are already hardened against (see their own doc
    /// comments) — previously unguarded at ordinary process restart. A no-op for a genuinely fresh
    /// session, since its own recorded model already equals `model` either way. Matches `run
    /// --continue`'s own identical `model_explicit` distinction (`commands/run.rs::run_task`).
    pub model_explicit: bool,
    /// Same idea as [`Self::model_explicit`], for `--reasoning-effort`/`starting_level`.
    pub reasoning_effort_explicit: bool,
    /// Opt-in cap on model turns per prompt (`--max-steps`). `None` (the default) runs unbounded —
    /// see `agent_core::Agent::with_max_steps` for the rationale.
    pub max_steps: Option<u32>,
    /// Base system prompt (agent identity). Project instructions, skills, and env are layered on top.
    pub system: String,
    /// Extra instructions appended after the base prompt (`--append-system-prompt`).
    pub append_system: Option<String>,
    /// Whether to discover and inject `AGENTS.md`/`CLAUDE.md` project-instruction files.
    pub context_files: bool,
    /// Persist a single session to this JSONL file (append-per-turn). Mutually exclusive with
    /// `session_dir`.
    pub session_file: Option<String>,
    /// Persist many sessions under this directory (a [`SessionRepo`]). Enables the multi-session
    /// commands (`list_sessions`, `switch_session`, `fork`, `set_session_name`).
    pub session_dir: Option<String>,
    /// The persistent-memory backend DSN (`--memory`/`AI_AGENT_MEMORY_URL`, with the stored
    /// `default_memory_backend` folded in by `main.rs`). `None` resolves to a per-project local-file
    /// store. Resolved into a backend by `serve_session` itself — after project trust — not here, so it
    /// shares the same resolved `cwd` (mirroring `agents`). See [`crate::memory`].
    pub memory: Option<String>,
    /// Disable persistent memory entirely (`--no-memory`): no `memory` tool, no injected index.
    pub no_memory: bool,
    /// Disable only the per-session `/session` working-memory mount (`--no-session-memory`), keeping the
    /// durable `/memories` store. On by default whenever memory is enabled. See [`crate::memory`].
    pub no_session_memory: bool,
    /// **Address** this exact session: open it if it already exists, else create it under exactly this
    /// id. Already-validated by `main.rs` (embedded directly into a persisted filename, so it must be
    /// sanitized before it ever reaches here). Matches `run`'s identical `--session-id` flag/contract.
    ///
    /// This is the most specific selector there is, so it wins over [`Self::continue_session`] and over
    /// the default — see [`SessionSelect`]. It never resolves to a *different* session, which is what
    /// makes it usable as a routing key for a multi-tenant `serve` (and is exactly what the daemon's
    /// `?session_id=` relies on). Repeating the same id is idempotent: same session, every time.
    pub session_id: Option<String>,
    /// Reattach to this `cwd`'s most recent session instead of starting a fresh one (`--continue`).
    /// Without it, `serve` starts a new session on every launch — two servers sharing a directory would
    /// otherwise silently drive the same on-disk transcript. Ignored when [`Self::session_id`] is set
    /// (that names one session outright) and in `session_file` mode (the path already names one).
    pub continue_session: bool,
    /// Skip persistence entirely, even though neither `session_file` nor `session_dir` was set —
    /// without this, `Persistence::open` defaults to a per-cwd directory under
    /// `~/.beyond/sessions/<encoded-cwd>/` rather than silently running in-memory-only (an operator
    /// who simply forgot the flag would otherwise lose all history on restart with no indication why).
    /// An explicit opt-out for the cases that genuinely want pure in-memory operation (e.g. a
    /// short-lived test harness).
    pub no_session_persistence: bool,
    /// The operator's compaction flags: `--context-window`, `--compact-at`, the reserve and keep-recent
    /// budgets, and the tool-masking knobs. Each `None` defers to the current model's own value (its
    /// window, its `ModelCaps::compact_at`), re-read on every `set_model`, so a switch gets
    /// the new model's real budget instead of a stale operator number; a `Some` pins it across
    /// switches. Resolved by [`compaction_config`].
    pub compaction: CompactionFlags,
    /// The per-turn output token ceiling (`Agent::with_max_tokens`). `None` defers to `Agent::new`'s
    /// own model-aware default (the model's own capability-table `max_output`, floored at
    /// `DEFAULT_MAX_TOKENS`) — matches `context_window`'s own "unset means model-derived" convention.
    /// Re-applied on every `build_agent` rebuild (`set_model`/`set_thinking`/…), same as every other
    /// `cfg`-sourced override, so it survives a model switch rather than resetting to that new model's
    /// own default.
    pub max_tokens: Option<u32>,
    /// Ask for the 1-hour prompt-cache TTL instead of 5 minutes. `main` sets it from `--cache-ttl`,
    /// which defaults to `1h` for `serve` (see "Prompt-cache TTL by mode" in ARCHITECTURE.md).
    pub cache_long: bool,
    /// Extended-thinking token budget, when enabled (`None` leaves thinking off). Must be below the
    /// per-turn `max_tokens`.
    pub thinking: Option<u32>,
    /// Reasoning effort for models driven by an effort level rather than a token budget (OpenAI
    /// reasoning models, Anthropic adaptive-thinking models). `None` leaves the provider default.
    /// Fixed for the process — unlike `thinking`, there's no `set_reasoning_effort` RPC command.
    pub reasoning_effort: Option<agent_core::ReasoningEffort>,
    /// Sampling temperature. `None` leaves the provider default. Fixed for the process — no RPC
    /// command to change it mid-run. See `agent_core::ModelRequest::temperature`'s doc comment for
    /// per-dialect gating (Anthropic omits it while thinking is enabled).
    pub temperature: Option<f64>,
    /// Trust the working directory for this run only, so a project-local `.claude/SYSTEM.md` is
    /// honored even if the directory isn't in the persisted allowlist (`agent trust <path>`). See
    /// `crate::trust_store`.
    pub trust_project: bool,
    /// Force the working directory *untrusted* for this run only, overriding both `trust_project` and
    /// the persisted allowlist — pi's own `--no-approve`/`-na`. For a directory the operator has
    /// permanently `agent trust`ed but wants to run against once as if it weren't (testing untrusted
    /// behavior, or extra caution on a checkout that happens to live under an already-trusted parent).
    /// Wins over `trust_project` if both are somehow set.
    pub force_untrusted: bool,
    /// Token budget reserved below the context window when summarizing an abandoned tree branch —
    /// independent of `compaction_reserve_tokens`. `None` keeps `Agent`'s prior hard-tied behavior
    /// (falls back to whatever `compaction.reserve_tokens` resolves to — see
    /// `agent_core::Agent::summarize_branch`'s doc comment). Task #31 (pi-parity feature):
    /// `agent_core::Agent::with_branch_summary_reserve_tokens` previously had no caller in either
    /// binary; `run`'s identical `--branch-summary-reserve-tokens` flag.
    pub branch_summary_reserve_tokens: Option<u32>,
    /// Disable automatic (threshold-triggered) compaction entirely for this process — `run`'s
    /// identical flag. Only ever a one-way "definitely off" here: when `false` (the flag/env's own
    /// default, indistinguishable from "not given" for a bare disable-only flag), `serve`'s startup
    /// falls back to the persisted `agent settings` `compaction_enabled` override (see
    /// `settings::Settings::compaction_enabled`'s doc comment) before finally defaulting to enabled —
    /// see [`serve`]'s own resolution of `current_auto_compaction`.
    pub no_compaction: bool,
    /// How many times to retry a gateway request that fails before the first response byte arrives.
    /// `None` keeps the client's built-in default.
    pub retry_max_retries: Option<u32>,
    /// Base of the exponential backoff between those retries. `None` keeps the client's built-in
    /// default.
    pub retry_base_delay_ms: Option<std::time::Duration>,
    /// Ceiling on that exponential backoff. `None` keeps the client's built-in default
    /// (`agent_core::client::MAX_BACKOFF`, 60s). Task #30 (pi-parity feature): the retry cluster's
    /// third knob, `agent_core::client::GatewayClient::with_max_backoff`, previously had no CLI flag or
    /// persisted override at all.
    pub retry_max_backoff_ms: Option<std::time::Duration>,
    /// How long one turn keeps retrying a rate limit or overload, in seconds (`--retry-rate-limit-secs`).
    /// `None` keeps `agent_core::rate_limit::DEFAULT_BUDGET`.
    pub retry_rate_limit_secs: Option<u64>,
    /// Idle-read timeout between response chunks on the gateway HTTP client. `None` keeps the client's
    /// built-in default. Task #38 (pi-parity fix): `run`'s identical `--idle-timeout-ms`/
    /// `AI_AGENT_IDLE_TIMEOUT_MS`/persisted `default_provider_timeout_ms` setting (see `main.rs::
    /// run_task`'s own `with_idle_timeout` call site) previously had no `serve` counterpart at all —
    /// `build_gateway_client` never called `GatewayClient::with_idle_timeout`.
    pub idle_timeout_ms: Option<u64>,
    /// Force every image down the same downgrade-to-text-placeholder path a vision-incapable model
    /// already gets, regardless of the active model's real `supports_vision` capability — `run`'s
    /// identical `--block-images`/`settings --block-images`. Threaded to `Agent::with_block_images`
    /// in `build_agent`.
    pub block_images: bool,
    /// Whether `read` downscales/re-encodes an oversized image to fit the inline size budget —
    /// `run`'s identical `--no-image-auto-resize`/`settings --image-auto-resize`. Threaded to
    /// `ToolConfig::image_auto_resize` in `build_tools`.
    pub image_auto_resize: bool,
    /// Run `bash` commands through this shell instead of the auto-resolved one (`/bin/bash`, else
    /// `bash` on `$PATH`, else `sh`). Already validated to exist by the time it reaches here (see
    /// `--bash-shell-path` in `main.rs`) — `build_tools` trusts it rather than re-checking on every
    /// rebuild.
    pub bash_shell_path: Option<String>,
    /// Prepend this line to every `bash` command, in the same shell invocation — matches pi's own
    /// `shellCommandPrefix` setting. Fixed for the process, like `bash_shell_path`.
    pub bash_command_prefix: Option<String>,
    /// `--web-allow-private`: let the `web` tool reach loopback/private/link-local addresses. Off by
    /// default. See `tools::web`.
    pub web_allow_private: bool,
    /// `--web-allow-host`: hostnames the `web` tool may reach even with private egress off.
    pub web_allow_hosts: Vec<String>,
    /// `--web-timeout-ms`: the `web` tool's per-request timeout (default 30 s).
    pub web_timeout_ms: Option<u64>,
    /// A process-wide default exec endpoint, applied to every session at startup.
    ///
    /// For the single-tenant case (one server, one sandbox) this saves an RPC. A multi-tenant server
    /// leaves it unset and calls `set_exec_endpoint` per session instead — the per-session value
    /// always wins, because the cell is what the tools read.
    pub exec_url: Option<String>,
    pub exec_header: Vec<String>,
    pub exec_cmd: Option<String>,
    /// `--exec-max-response-bytes`: the most any one HTTP exec response may be, for the default
    /// endpoint *and* every per-session `set_exec_endpoint` — the cap protects this process, which
    /// every tenant shares. [`crate::exec_endpoint::DEFAULT_MAX_RESPONSE_BYTES`] when unset.
    pub exec_max_response_bytes: usize,
    /// Run-lifecycle emitter. [`crate::lifecycle::NoLifecycle`] when unconfigured — zero I/O, no
    /// worker. Constructed once at process startup (`lifecycle::open`) so a malformed URL fails
    /// before any session runs, and cloned (the `Arc`) into every daemon session so they share one
    /// outbound worker. The URL is `http`/`https` or `unix:` (same machine, no host, no TLS).
    pub lifecycle: std::sync::Arc<dyn crate::lifecycle::RunLifecycle>,
    /// How often to re-POST the latest progress sample while a run is in flight. `Duration::ZERO`
    /// disables the timer; tool-edge / todo / turn-end samples still fire. Default
    /// [`crate::lifecycle::HEARTBEAT`] (10s). An unchanged sample is not re-POSTed.
    pub lifecycle_heartbeat: std::time::Duration,
    /// Restrict the tool set to exactly these names, dropping everything else. Combine with
    /// `exclude_tools` to carve one back out of the allow-list. Fixed for the process — like `system`,
    /// there's no runtime RPC to change it, but it does survive a `set_model`/`set_thinking` rebuild
    /// (`build_agent` reapplies it every time).
    pub tools: Option<Vec<String>>,
    /// Drop these tools from the default set — e.g. `["bash", "write"]` for a read-only reviewer.
    pub exclude_tools: Option<Vec<String>>,
    /// Register no tools at all. Wins over `tools`/`exclude_tools`.
    pub no_tools: bool,
    /// Tools discovered from every configured MCP server, already connected and namespaced
    /// (`mcp__<server>__<tool>`) — see `tools::mcp`'s module doc comment. Resolved exactly once, by
    /// `main.rs`'s `serve`-dispatch branch (`tools::mcp::connect_all`), *before* this `ServeConfig` is
    /// constructed — connecting is inherently async and this struct is a plain, synchronous value.
    /// `build_tools` merges these into every registry rebuild (`set_model`/`set_thinking`/...) without
    /// reconnecting to any server: the live connection (and its spawned child process, for a stdio
    /// server) lives as long as the set's `Arc<dyn Tool>` entries do, i.e. for the whole `serve`
    /// session. Live: a server that changes its tools (`notifications/tools/list_changed`, or a start
    /// that lists something new) changes this set, and every session picks the change up — mid-run
    /// through the loop's tool source, and by an agent rebuild before the next `prompt`.
    pub mcp_tools: crate::tools::mcp::McpTools,
    /// Catalog of connected MCP servers (resources/prompts metadata + live handles for
    /// `mcp_complete`). Built alongside `mcp_tools` by `connect_all`.
    pub mcp_catalog: crate::tools::mcp::McpCatalog,
    /// `--mcp-events-callback-url`: the externally reachable base URL that routes to this daemon's
    /// listener, used as the base of MCP Events webhook callback URLs (see
    /// [`crate::tools::mcp_events`]). `None` disables webhook delivery; poll and push still work.
    /// `main.rs` refuses it without `--listen`/`--listen-uds`, since nothing would receive them.
    pub mcp_events_callback_url: Option<String>,
    /// `--mcp-events-reapable`: let the daemon's idle reaper stop a detached session even while it
    /// holds live MCP Events subscriptions. Off by default — a subscription is a background trigger,
    /// and reaping its session would silently end it.
    pub mcp_events_reapable: bool,
    /// `--mcp-events-session`: the daemon session that owns the configured
    /// (`mcp_servers[].events`) subscriptions — one per process, so one event is one model run.
    pub mcp_events_session: String,
    /// Whether *this* session owns the configured subscriptions: `true` for the stdio `serve`
    /// (its only session); set per session by `serve_ws::session_cfg` in the daemon.
    pub mcp_events_owner: bool,
    /// The same servers dialed with MCP Apps advertised, for a session whose client declares it
    /// renders apps (`set_mcp_apps`) — see [`crate::tools::mcp_apps`]. Process-wide (shared through
    /// this `Clone`) for the operator's configured servers; replaced per session in service mode.
    /// `None` = this process cannot host apps.
    pub mcp_apps: Option<crate::tools::mcp_apps::McpAppsPool>,
    /// Agent definitions discovered at startup (see [`crate::agents`]) — the delegable personas the
    /// `subagent` tool accepts, advertised in `<available_agents>`. Discovered once, like `mcp_tools`,
    /// rather than re-walked on every registry rebuild. Empty when subagents aren't configured.
    pub agents: Vec<crate::agents::AgentDef>,
    /// Force every batch of tool calls in a turn to run one at a time instead of the default
    /// bounded-concurrent dispatch (`agent_core::Agent::with_sequential_tools`). Fixed for the process,
    /// like `tools`/`no_tools` — `build_agent` reapplies it every rebuild.
    pub sequential_tools: bool,
    /// Defer MCP tools behind the Code Mode `execute` tool. Fixed for the process; `build_tools`
    /// reapplies it every rebuild. A no-op unless the binary was built with `--features code-mode`;
    /// the CLI rejects `--code-mode` on a density-default binary before this is set.
    pub code_mode: bool,
    /// How MCP tools reach the model when Code Mode is off (`--mcp-tool-loading`). Fixed for the
    /// process; `build_tools` reapplies it every rebuild. See `tools::tool_search`.
    pub mcp_loading: crate::tools::tool_search::McpLoading,
    /// Block every call to a tool named here, even though it stays registered/visible to the model —
    /// unlike `exclude_tools` (the model never learns it exists), a denied call still surfaces as a
    /// normal error `tool_result` explaining it was blocked by policy. Installs an `AgentHooks` gate
    /// (`crate::policy::ToolPolicy`) on every `build_agent` rebuild; empty means no hook at all (the
    /// `NoHooks` default), same zero-cost as before this flag existed.
    pub deny_tool: Vec<String>,
    /// Block a `bash` call whenever its command contains this substring, case-insensitively. Combines
    /// with `deny_tool` under the same policy hook.
    pub deny_bash_pattern: Vec<String>,
    /// Block a `write`/`edit` call whenever its `path` argument matches this glob, with gitignore
    /// semantics anchored to the tool root (see `policy::anchored_globs`). Combines with
    /// `deny_tool`/`deny_bash_pattern` under the same policy hook.
    pub deny_path: Vec<String>,
    /// Which tools require a human's approval before they run (`--approve off|writes|all|tools:a,b`). The
    /// static deny-lists above still win first — they need no round trip, and asking a person to approve
    /// what the operator already forbade is a way to social-engineer past them. See [`crate::approval`].
    pub approve: crate::approval::GatedSet,
    /// `--approve-mcp-skills`: approve every MCP-served skill's activation, and code execution while
    /// one is active, in advance — for a headless daemon with no one to ask. Off by default: those
    /// questions go to the attached clients as `approval_request` frames carrying an `mcp_skill`
    /// object (see `tools::mcp_skills::SkillSession`).
    pub approve_mcp_skills: bool,
    /// How long an unanswered `approval_request` waits before it is denied (`--approval-timeout`).
    /// `None` waits forever, which is only safe with a reliably-attached client: `running` stays `true`
    /// for the whole prompt, so a question nobody answers pins the session until an `abort`.
    pub approval_timeout: Option<std::time::Duration>,
    /// Disable *standard-root* skills discovery/loading (`~/.claude/skills`, `<cwd>/.claude/skills`) —
    /// no `<available_skills>` listing in the system prompt from either, and a `/skill:name` invocation
    /// is sent through unexpanded unless it resolves against a `--skill` path instead. Matches pi's own
    /// `--no-skills`, and `run`'s identical flag (`main.rs`'s `Run::no_skills`) — `serve` previously had
    /// no equivalent, so an operator wanting a hardened, no-custom-content deployment had no way to
    /// refuse project-supplied skills the way a one-shot `run` could. Skips standard-root discovery
    /// outright (like `run`) rather than discovering and then discarding, applied on every `reload` too
    /// — but an explicit `--skill <path>` is still honored even so, matching pi's own `--no-skills` (a
    /// documented, tested combination: `--no-skills` means "nothing auto-discovered", not "no skills at
    /// all" — see `skills::discover_extra_only`'s doc comment).
    pub no_skills: bool,
    /// Disable *standard-root* prompt-template discovery/loading (`~/.claude/prompts`,
    /// `<cwd>/.claude/prompts`) — a `/name` invocation is sent through unexpanded unless it resolves
    /// against a `--prompt-template` path instead. Matches pi's own `--no-prompt-templates` and `run`'s
    /// identical flag; see `no_skills`'s doc comment for why `serve` needed this too, and for the
    /// explicit-extra-path carve-out that applies here identically.
    pub no_prompt_templates: bool,
    /// Additional, ad-hoc skill-discovery roots beyond the two standard ones — pi's own
    /// `--skill <path>` (repeatable). Applied on every `reload` too, same as the standard roots.
    pub extra_skill_paths: Vec<String>,
    /// Additional, ad-hoc prompt-template-discovery roots — pi's own `--prompt-template <path>`
    /// (repeatable). See `extra_skill_paths`'s doc comment; applies identically.
    pub extra_prompt_template_paths: Vec<String>,
    /// Set the session's title up front, if a *new* session is minted at startup (persistence
    /// configured, no existing session reattached) — pi's own `--name`. Never renames an existing
    /// session just because the process happened to be started with this flag again; already validated
    /// non-whitespace by `main.rs` before this struct is even constructed.
    pub name: Option<String>,
    /// Restrict `cycle_model`'s candidate list to exactly these patterns, resolved in this order
    /// (`--models`, comma-separated; see `resolve_model_scope` for the exact grammar — literal ids,
    /// globs against [`available_models`], and an optional `:<thinking-level>` suffix pinning that
    /// entry's depth) — pi's own `--models` flag (`resolveModelScopeWithDiagnostics`/`_scopedModels`).
    /// `set_model` is unaffected: it still accepts any id, scoped or not. `get_available_models` is
    /// unaffected too — unlike pi's own scope-defaulted `/model` picker, this RPC has no secondary
    /// "show everything" toggle, so it always reports the full list and leaves any scoped-vs-all UI
    /// distinction to the client. Empty or absent falls back to the full [`available_models`] list for
    /// cycling as well.
    pub models: Vec<String>,
    /// A persisted, blanket project-trust policy (`agent settings --default-project-trust`) — Fix 1
    /// (pi-parity gap): this field didn't exist at all before, so a stored policy had zero effect on
    /// `serve` sessions even though `run` already partially honored it (`commands/run.rs::run_task`). Consulted
    /// by [`resolve_project_trust`] at both startup and `reload`, with the same precedence `run` now
    /// uses: `trust_project`/`force_untrusted` win outright when set; otherwise an explicit per-path
    /// `TrustStore` entry (`agent trust`/`agent untrust <path>`) wins next; only when neither applies
    /// does this blanket policy take effect.
    pub default_project_trust: Option<crate::settings::TrustPolicy>,
    /// The initial steer-lane drain mode (`agent_core::QueueMode`) — pi's own `steeringMode`. Applied
    /// once, at startup, via `steering.set_steering_mode` right after `Steering::new()`; the
    /// `set_steering_mode` RPC command can still change it at runtime afterward. `main.rs` resolves
    /// this the same "explicit flag/env, then stored `agent settings --steering-mode`, then
    /// `QueueMode::default()`" precedence `run_task` uses for its own one-shot `Steering` (see that
    /// call site's doc comment) before ever constructing this config.
    pub steering_mode: agent_core::QueueMode,
    /// Same idea as [`Self::steering_mode`], for the follow-up lane (`set_follow_up_mode`) — pi's own
    /// `followUpMode`.
    pub follow_up_mode: agent_core::QueueMode,
    /// When set, `serve` offers its (byte-identical) control protocol over a WebSocket on this address
    /// instead of stdio — each accepted connection drives a session, and a session outlives its
    /// connection so a dropped mobile client can reconnect and re-attach to a still-running run (see
    /// [`crate::serve_ws`]). Bind loopback/internal only: the agent authenticates no caller, it trusts
    /// whatever the front door forwarded. `None` (the default) keeps the stdio transport.
    pub listen: Option<std::net::SocketAddr>,
    /// When set, `serve` also (or instead) offers its control protocol over a Unix-domain socket at
    /// this path — a same-VM client gets kernel-enforced local authz via filesystem permissions
    /// (unlike loopback TCP, which authenticates nothing). Bound on the *same* shared supervisor as
    /// [`Self::listen`], so a session created over either transport is reachable over the other by the
    /// same `?session_id=`. `None` (the default) means no UDS listener. See [`crate::serve_ws`].
    pub listen_uds: Option<std::path::PathBuf>,
    /// The octal permission mode to `chmod` the [`Self::listen_uds`] socket to after binding (default
    /// `0o600` — owner-only; use `0o660` for a shared group). Ignored when `listen_uds` is `None`.
    pub listen_uds_mode: Option<u32>,
    /// The session-grant verifier (`--grant-key` + `--seal-key`; see [`crate::grant`]), built and
    /// validated once at startup. `None` when neither flag is given. The connection path doesn't
    /// consult it yet. `Arc` so every per-session copy of this config shares one keyring and one
    /// fleet secret rather than duplicating the secret per session.
    pub grant_verifier: Option<Arc<crate::grant::GrantVerifier>>,
    /// exec v1.4's job-end callbacks: the replica's notify key and the sessions it holds, which a
    /// service session registers with ([`crate::tools::job_notify`]). Filled by the daemon in
    /// service mode from [`Self::grant_verifier`]'s seal secret; `None` everywhere else.
    pub job_notify: Option<Arc<crate::tools::job_notify::NotifyHub>>,
    /// `--notify-origin`: the origin the fleet's edge serves sessions at, which a `keep`'s
    /// `notify.url` names. Service mode only; `None` keeps every job held, as before v1.4.
    pub notify_origin: Option<String>,
    /// When set (daemon mode only), the supervisor reaps a session that has had **no attached
    /// connection** for at least this long and isn't mid-run — dropping its retained `input_tx` so it
    /// persists and exits, exactly like graceful shutdown does per-session (see [`crate::serve_ws`]).
    /// `None` (the default) keeps today's forever-lived behavior: a detached session lives until the
    /// daemon shuts down. Ignored outside the WebSocket/UDS daemon path.
    pub session_idle_timeout: Option<std::time::Duration>,
    /// How the daemon pools upstream (agent→gateway) connections across sessions — the mode from
    /// `--upstream-http2`. Only consulted in the WebSocket daemon path ([`crate::serve_ws::serve_ws`],
    /// which reads it once to build [`Self::shared_http`]); the stdio/`run` path never pools.
    pub upstream_http2: UpstreamHttp2,
    /// The one `reqwest::Client` every session in this daemon shares, so N concurrent sessions collapse
    /// onto one connection pool (HTTP/2-multiplexed under [`UpstreamHttp2::H2c`]) instead of N. Built
    /// once by [`crate::serve_ws::serve_ws`] from [`Self::upstream_http2`] and injected per session via
    /// [`agent_core::GatewayClient::with_http_client`] in `build_gateway_client`; **kept** (not nulled)
    /// by `session_cfg` so every session inherits it. `reqwest::Client` is `Arc`-backed, so cloning it
    /// into each session is a cheap pointer bump. `None` (the default, and always on the stdio/`run`
    /// path) preserves the per-session-client behavior.
    pub shared_http: Option<reqwest::Client>,
    /// Which world this config lives in. `serve --service` makes the daemon a **fail-closed
    /// multi-tenant service**: every connection must present a verified [`bsg_v1`
    /// grant](crate::grant), every session runs against that grant's sandbox and tenant-rooted
    /// storage, and nothing tenant-specific is read from or written to the replica host (see
    /// [`crate::service`]). `main.rs` sets it once from the flags (`Local` or `Replica`); the
    /// connection that spawns a session binds it to that session's verified grant
    /// (`serve_ws::session_cfg`), since at startup there is no connection and so no tenant. A
    /// session still `Replica` has no grant, which fails the session rather than silently falling
    /// back to a host default.
    pub mode: Mode,
    /// The tenant-data mounts (`--shard <name>=</abs/path>`, repeatable). Empty outside service
    /// mode; `Arc` so every per-session copy of this config shares one map.
    pub shards: Arc<crate::service::Shards>,
    /// Cap on live sessions this replica will hold at once; `0` turns it off. Two open descriptors
    /// per live session (its newest segment and its lock) keeps the default comfortably inside EFS's
    /// documented 65,536 open files and 65,536 locks per instance. Enforced where a session is
    /// spawned (`serve_ws::Supervisor::try_pin`), which answers 503 rather than starting one.
    pub max_live_sessions: usize,
    /// The process-wide client per-session MCP connectors dial through — ALPN and the `web` tool's
    /// SSRF resolver, and deliberately *not* the gateway's h2c pool (a tenant's connector is not the
    /// gateway). Built once by `main.rs` in service mode and shared by every session on the replica;
    /// `None` everywhere else, which makes a grant's connectors simply unavailable rather than
    /// dialed unchecked. See [`crate::tools::mcp::McpEgress`].
    pub mcp_http: Option<crate::tools::mcp::McpEgress>,
    /// This replica's instruments, when `--metrics-listen` gave it a scrape endpoint. `None`
    /// everywhere else, so every call site is an `if let` and a replica without metrics pays
    /// nothing. Never carries a tenant or session identifier into a label — see
    /// [`crate::metrics`].
    pub metrics: Option<Arc<crate::metrics::Metrics>>,
    /// True for a session `serve_ws` spawned, false for stdio `serve` and `run`. A supervised
    /// session leaves OS signals to its supervisor; see [`ShutdownSignal::inert`].
    pub supervised: bool,
    /// How long a drain waits for in-flight runs before stopping every session (`--drain-grace`).
    /// `Duration::ZERO` — the default — means a signal shuts the daemon down at once, exactly as it
    /// did before drain existed, so Ctrl-C on a laptop daemon is still instant.
    pub drain_grace: std::time::Duration,
}

impl ServeConfig {
    /// Which world this config's session lives in: its grant's, or this host's.
    pub(super) fn mode(&self) -> Mode {
        self.mode.clone()
    }

    /// Fold the workspace's `AGENTS.md` frontmatter in under the operator's flags: a field the flags
    /// already set (`--model`, `--tools`/`--no-tools`, `--max-steps`) is left alone, one they did not
    /// is filled from the file. The one exception is a replica's `--model`, which is a fleet-wide
    /// default and so yields to the tenant's own file. A local session's model is *not* marked
    /// explicit, so a reopened one keeps the model it was last on, as it does for a stored default.
    pub(super) fn apply_agent_settings(&mut self, settings: &crate::resources::AgentSettings) {
        if let Some(model) = &settings.model
            && (!self.model_explicit || self.mode.model_flag_is_a_default())
        {
            self.model = model.clone();
        }
        if self.tools.is_none() && !self.no_tools {
            self.tools = settings.tools.clone();
        }
        if self.max_steps.is_none() {
            self.max_steps = settings.max_steps;
        }
    }
}

/// Resolve whether a project is trusted for this session, from already-gathered inputs — shared by
/// `run` (`commands/run.rs::run_task`) and `serve` (both its startup and `reload` paths) so the two binaries
/// agree on trust for the same directory under the same settings, matching pi's own
/// `trust-manager.ts:46-96` precedence:
///
/// 1. `force_untrusted` (`--force-untrusted`/`-na`) always wins outright: untrusted, full stop.
/// 2. `trust_project` (`--trust-project`/`-a`) wins next: trusted, full stop.
/// 3. An explicit **per-path** entry in the persisted [`crate::trust_store::TrustStore`] allowlist
///    (`agent trust`/`agent untrust <path>`, already resolved into `trust_lookup` by the caller) — Fix 1
///    (pi-parity bug): this used to be checked *after* the blanket policy below, so an operator's
///    specific exception for one directory could be silently overridden by a coarser `never`/`always`
///    default; pi's own resolution always checks the nearest explicit per-path decision before falling
///    back to any blanket policy, in either direction (an explicit `never` policy must not override a
///    specific `agent trust <path>`, and vice versa for `always`/an explicit `agent untrust <path>`).
/// 4. Only when `trust_lookup` is [`crate::trust_store::Trust::Unknown`] does the persisted blanket
///    `default_project_trust` policy apply — `always`/`never` decide outright; `ask` (or no policy at
///    all) falls back to this crate's original heuristic: trusted whenever the project has nothing
///    project trust actually gates (`has_gated_resources`, from
///    `crate::trust_store::has_trust_gated_resources`). This binary is headless, so unlike pi's own
///    interactive "trust this folder?" prompt, `ask` can't actually ask anything — see
///    [`crate::settings::TrustPolicy::Ask`]'s own doc comment.
///
/// Takes `trust_lookup`/`has_gated_resources` as plain data (rather than resolving them itself from
/// `cwd`) so the decision logic is unit-testable with no filesystem/global-trust-store state to
/// sandbox — the same "split I/O gathering from pure decision logic" shape this crate already uses
/// elsewhere (e.g. `resources::tz_string_offset` vs `tz_env_offset`).
pub fn resolve_project_trust(
    trust_project: bool,
    force_untrusted: bool,
    default_project_trust: Option<crate::settings::TrustPolicy>,
    trust_lookup: crate::trust_store::Trust,
    has_gated_resources: bool,
) -> bool {
    use crate::settings::TrustPolicy;
    use crate::trust_store::Trust;
    if force_untrusted {
        return false;
    }
    if trust_project {
        return true;
    }
    // Task #35 (pi-parity fix): pi's own `resolveProjectTrusted` (`project-trust.ts`) runs this
    // "nothing here for project trust to actually gate" fast path *immediately* after the
    // override check above — before even consulting the persisted trust store, let alone a blanket
    // `default_project_trust` policy. Previously this crate only consulted `has_gated_resources` deep
    // inside the `Unknown` arm below, as the final fallback — meaning an explicit per-path
    // `Trust::Trusted`/`Trust::Untrusted` entry, or an explicit `always`/`never` blanket policy, would
    // win even when the project has nothing trust-gated at all to protect. Currently inert in practice
    // (no trust-gated resource type today actually depends on this ordering), but would silently
    // matter the moment a new one is added without this precedence already matching pi's.
    if !has_gated_resources {
        return true;
    }
    match trust_lookup {
        Trust::Trusted => true,
        Trust::Untrusted => false,
        Trust::Unknown => match default_project_trust {
            Some(TrustPolicy::Always) => true,
            Some(TrustPolicy::Never) => false,
            // `has_gated_resources` is always `true` by the time this arm is reached (the fast path
            // above already returned otherwise), so this is effectively always `false` now — kept as
            // `!has_gated_resources` rather than a bare `false` literal so the fallback stays correct
            // and self-documenting on its own terms if the fast path above is ever refactored away.
            Some(TrustPolicy::Ask) | None => !has_gated_resources,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_project_trust_force_untrusted_wins_over_everything() {
        use crate::settings::TrustPolicy;
        use crate::trust_store::Trust;
        assert!(!resolve_project_trust(
            true, // trust_project also set
            true, // force_untrusted
            Some(TrustPolicy::Always),
            Trust::Trusted,
            false,
        ));
    }

    #[test]
    fn resolve_project_trust_trust_project_wins_when_not_force_untrusted() {
        use crate::settings::TrustPolicy;
        use crate::trust_store::Trust;
        assert!(resolve_project_trust(
            true,
            false,
            Some(TrustPolicy::Never),
            Trust::Untrusted,
            true,
        ));
    }

    #[test]
    fn resolve_project_trust_an_explicit_trusted_entry_wins_over_a_never_policy() {
        // The core Fix 1 regression: an operator's specific `agent trust <path>` exception must win
        // over a coarser `never` blanket default, not be overridden by it.
        use crate::settings::TrustPolicy;
        use crate::trust_store::Trust;
        assert!(resolve_project_trust(
            false,
            false,
            Some(TrustPolicy::Never),
            Trust::Trusted,
            true,
        ));
    }

    #[test]
    fn resolve_project_trust_an_explicit_untrusted_entry_wins_over_an_always_policy() {
        // The mirror case: `agent untrust <path>` must win over a blanket `always` default too.
        // `has_gated_resources: true` — Task #35's own fast path (below) short-circuits to trusted
        // whenever there's nothing to gate at all, which would defeat the very precedence this test
        // means to isolate if it were `false` here instead.
        use crate::settings::TrustPolicy;
        use crate::trust_store::Trust;
        assert!(!resolve_project_trust(
            false,
            false,
            Some(TrustPolicy::Always),
            Trust::Untrusted,
            true,
        ));
    }

    #[test]
    fn resolve_project_trust_falls_back_to_the_blanket_policy_only_when_the_lookup_is_unknown() {
        // Both cases use `has_gated_resources: true` for the same reason as the test above — isolating
        // the `Unknown`-branch blanket-policy precedence from Task #35's separate "nothing to gate"
        // fast path, which is covered by its own dedicated test below.
        use crate::settings::TrustPolicy;
        use crate::trust_store::Trust;
        assert!(resolve_project_trust(
            false,
            false,
            Some(TrustPolicy::Always),
            Trust::Unknown,
            true,
        ));
        assert!(!resolve_project_trust(
            false,
            false,
            Some(TrustPolicy::Never),
            Trust::Unknown,
            true,
        ));
    }

    #[test]
    fn resolve_project_trust_nothing_to_gate_wins_outright_over_the_trust_store_and_any_policy() {
        // Task #35 (pi-parity fix): pi's own `resolveProjectTrusted` runs the "nothing here for
        // project trust to actually gate" fast path *before* consulting the trust store at all — so it
        // wins even over an explicit per-path `Trust::Untrusted` entry or a blanket `never` policy, not
        // just the `Unknown`+no-policy fallback this crate previously limited it to. Currently inert
        // (no gated resource type hits this today), but this pins the correct precedence so a future
        // gated resource type added without re-checking this ordering doesn't silently regress it.
        use crate::settings::TrustPolicy;
        use crate::trust_store::Trust;
        assert!(
            resolve_project_trust(false, false, None, Trust::Untrusted, false),
            "an explicit untrusted entry must not matter when there's nothing to gate"
        );
        assert!(
            resolve_project_trust(
                false,
                false,
                Some(TrustPolicy::Never),
                Trust::Unknown,
                false
            ),
            "a blanket never policy must not matter either, for the same reason"
        );
    }

    #[test]
    fn resolve_project_trust_ask_or_no_policy_falls_back_to_has_gated_resources() {
        use crate::trust_store::Trust;
        // No gated resources at all: nothing meaningfully differs between trusted/untrusted, so this
        // crate's headless "ask" fallback treats it as trusted.
        assert!(resolve_project_trust(
            false,
            false,
            None,
            Trust::Unknown,
            false
        ));
        assert!(!resolve_project_trust(
            false,
            false,
            None,
            Trust::Unknown,
            true
        ));
    }

    // pi-parity fix (L10): `--models` only ever accepted flat literal ids — no glob against the known
    // catalog, no `pattern:<level>` suffix to pin a scoped entry's thinking depth. `resolve_model_scope`
    // is the resolver pi calls `resolveModelScopeWithDiagnostics`.
}
