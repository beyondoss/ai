//! `run`: one task to completion, in phases.
//!
//! `run_task` threads four phases, each handing the next a struct of what it settled:
//! [`resolve_session`] (flag checks, session selection, the first message) -> [`resolve_defaults`]
//! (flag, then stored setting, then built-in default; trust and skill discovery) ->
//! [`assemble_run`] (MCP, tools, session, system prompt) -> [`build_agent`] (client and agent), then
//! [`drive_run`] runs the turns and reports.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use agent_core::{Agent, GatewayClient, Session, Tool};
use beyond_ai_agent::gateway_credential::GatewayCredential;
use beyond_ai_agent::session_store::{SessionMeta, SessionStore};
use beyond_ai_agent::{serve, tools};

use beyond_ai_agent::policy::ToolPolicy;
use beyond_ai_agent::session_store::{
    SessionRepo, canonical_cwd, default_session_dir, fork_by_arg, is_path_like, open_session_by_id,
    sessions_root,
};

use crate::{
    CacheTtl, CliTerminal, Command, DEFAULT_GATEWAY, DEFAULT_MODEL, DirectCheckpoint, FreshSession,
    RUN_JOURNAL, RUN_WRAPPED_UP, RunJournal, RunSession, RunStore, build_run_gateway_client,
    emit_cli_lifecycle, exit_cancelled, exit_process, expand_message, is_valid_session_id,
    mcp_manifest_dir, parse_reasoning_effort, partition_tasks, persist_run, read_file_refs,
    read_stdin_explicit, read_stdin_if_piped, resolve_gateway_credential_and_headers,
    resolve_prompt_input, resolve_thinking_budget_overrides, resume_mcp_tasks_for_run, run_turn,
    stop_jobs, text_mode_failure_message, unwrap_turn_result, write_stdout_or_exit,
};

/// What `resolve_session` settled: the model, the session selection and the first message.
struct Composed {
    block_images: bool,
    cwd: PathBuf,
    cwd_str: String,
    image_auto_resize: bool,
    initial_images: Vec<agent_core::ImageSource>,
    initial_message: String,
    messages: Vec<String>,
    model: String,
    model_thinking_level: Option<agent_core::ThinkingLevel>,
    resumed_model: Option<String>,
    selected: RunSession,
    stored_settings: beyond_ai_agent::settings::Settings,
    timing: beyond_ai_agent::timing::StartupTiming,
}

/// What `resolve_defaults` settled, on top of `Composed`: every setting, the credential, trust and skills.
struct Resolved {
    bash_command_prefix: Option<String>,
    bash_shell_path: Option<String>,
    block_images: bool,
    branch_summary_reserve_tokens: Option<u32>,
    compaction_keep_recent_tokens: Option<u32>,
    compaction_reserve_tokens: Option<u32>,
    cwd: PathBuf,
    cwd_str: String,
    gateway: String,
    idle_timeout_ms: Option<u64>,
    image_auto_resize: bool,
    initial_images: Vec<agent_core::ImageSource>,
    initial_message: String,
    key: GatewayCredential,
    messages: Vec<String>,
    model: String,
    model_extra_headers: std::collections::HashMap<String, String>,
    project_trusted: bool,
    prompt_templates: Vec<beyond_ai_agent::prompts::PromptTemplate>,
    provider_env: beyond_ai_agent::gateway_credential::ProviderEnv,
    raw_gateway_key: Option<String>,
    reasoning_effort: Option<agent_core::ReasoningEffort>,
    resumed_model: Option<String>,
    retry_base_delay_ms: Option<u64>,
    retry_max_backoff_ms: Option<u64>,
    retry_max_retries: Option<u32>,
    selected: RunSession,
    skills: Vec<beyond_ai_agent::skills::Skill>,
    steering: agent_core::Steering,
    stored_settings: beyond_ai_agent::settings::Settings,
    timing: beyond_ai_agent::timing::StartupTiming,
}

/// What `assemble_run` settled: the tools, the open session, the system prompt and session metadata.
struct Assembled {
    _session_lock: Option<beyond_ai_agent::session_store::SessionLock>,
    bash_jobs: Arc<tools::bash_jobs::BashJobs>,
    block_images: bool,
    branch_summary_reserve_tokens: Option<u32>,
    compaction_candidates: agent_core::CompactionCandidates,
    compaction_keep_recent_tokens: Option<u32>,
    compaction_reserve_tokens: Option<u32>,
    exec_runner: Option<Arc<dyn tools::exec::CommandRunner>>,
    gateway: String,
    idle_timeout_ms: Option<u64>,
    initial_images: Vec<agent_core::ImageSource>,
    initial_message: String,
    key: GatewayCredential,
    lifecycle: Arc<dyn beyond_ai_agent::lifecycle::RunLifecycle>,
    mcp_catalog: tools::mcp::McpCatalog,
    mcp_enabled: tools::mcp::McpEnabledSet,
    mcp_skills: tools::mcp_skills::McpSkills,
    messages: Vec<String>,
    meta: SessionMeta,
    model: String,
    model_extra_headers: std::collections::HashMap<String, String>,
    output_slot: tools::structured_output::OutputSlot,
    parent_loop: beyond_ai_agent::tools::subagent::ParentCell,
    /// Which filesystem the tools act on, so `--deny-path` resolves paths in the same one.
    path_world: tools::fs::PathWorld,
    prompt_templates: Vec<beyond_ai_agent::prompts::PromptTemplate>,
    reasoning_effort: Option<agent_core::ReasoningEffort>,
    registry: agent_core::ToolRegistry,
    retry_base_delay_ms: Option<u64>,
    retry_max_backoff_ms: Option<u64>,
    retry_max_retries: Option<u32>,
    session: Session,
    session_memory_active: bool,
    skills: Vec<beyond_ai_agent::skills::Skill>,
    steering: agent_core::Steering,
    store: Option<SessionStore>,
    stored_settings: beyond_ai_agent::settings::Settings,
    system: String,
    wants_structured_output: bool,
    write_locks: Arc<agent_core::WriteLockRegistry>,
}

/// What `build_agent` settled: the agent and everything `drive_run` needs to run and report.
struct RunSetup {
    _session_lock: Option<beyond_ai_agent::session_store::SessionLock>,
    agent: Agent,
    bash_jobs: Arc<tools::bash_jobs::BashJobs>,
    initial_images: Vec<agent_core::ImageSource>,
    initial_message: String,
    lifecycle: Arc<dyn beyond_ai_agent::lifecycle::RunLifecycle>,
    mcp_catalog: tools::mcp::McpCatalog,
    mcp_enabled: tools::mcp::McpEnabledSet,
    mcp_skills: tools::mcp_skills::McpSkills,
    messages: Vec<String>,
    meta: SessionMeta,
    output_slot: tools::structured_output::OutputSlot,
    pressure_point: u32,
    prompt_templates: Vec<beyond_ai_agent::prompts::PromptTemplate>,
    retry_policy: beyond_ai_agent::retry::RunRetryPolicy,
    run_journal: Option<&'static RunJournal>,
    session: Session,
    session_memory_active: bool,
    skills: Vec<beyond_ai_agent::skills::Skill>,
    steering: agent_core::Steering,
    store: RunStore,
    system_for_export: String,
    tool_defs: Vec<agent_core::ToolDef>,
    wants_structured_output: bool,
}

/// `run`: one task to completion.
pub(crate) async fn run_task(cmd: Command) -> Result<(), Box<dyn std::error::Error>> {
    let composed = resolve_session(cmd.clone()).await?;
    let resolved = resolve_defaults(cmd.clone(), composed)?;
    let assembled = assemble_run(cmd.clone(), resolved).await?;
    let setup = build_agent(cmd.clone(), assembled).await?;
    drive_run(cmd, setup).await
}

/// Validates the flags, selects the session and composes the first message.
async fn resolve_session(cmd: Command) -> Result<Composed, Box<dyn std::error::Error>> {
    let Command::Run {
        tasks,
        model,
        block_images,
        no_block_images,
        no_image_auto_resize,
        tools: tools_allow,
        exclude_tools: tools_exclude,
        no_tools,
        code_mode,
        deny_tool,
        deny_path,
        extra_skill_paths,
        extra_prompt_template_paths,
        name,
        fork,
        session: session_path,
        session_id,
        r#continue: continue_session,
        session_dir,
        no_session_persistence,
        ..
    } = cmd
    else {
        unreachable!("dispatched on Command::Run")
    };
    // Fail fast, before touching any files — matches pi's own `--name` validation. Whitespace-only is
    // rejected outright here (a startup argument the operator clearly meant to be meaningful), unlike
    // the RPC `set_session_name` command's "empty clears the title" convention (renaming an
    // already-running session to nothing is a deliberate, different action).
    if let Some(n) = &name
        && n.trim().is_empty()
    {
        return Err("--name requires a non-empty value".into());
    }
    if code_mode {
        tools::code_mode::require_compiled()?;
    }
    // A malformed `--deny-path` glob must never silently produce a no-op policy — see
    // `ToolPolicy::deny_path`'s doc comment for the fail-open this closes.
    if let Err(e) = ToolPolicy::validate_deny_path_patterns(&deny_path) {
        return Err(e.into());
    }
    tools::selection::check_tool_flags(
        tools_allow.as_deref(),
        tools_exclude.as_deref(),
        &deny_tool,
        no_tools,
    )?;
    beyond_ai_agent::skills::check_paths("--skill", &extra_skill_paths)?;
    beyond_ai_agent::skills::check_paths("--prompt-template", &extra_prompt_template_paths)?;
    // `--session-id` is embedded directly into a filename (`SessionMeta::with_id` →
    // `SessionRepo::path_for`'s `{created_at}_{id}.jsonl`) with no other sanitization — an id like
    // `../../../tmp/pwned/evil` would write (and `mkdir -p`, since `SessionStore::create` does that
    // too) outside the intended sessions directory. Matches pi's own `assertValidSessionId`.
    if let Some(id) = &session_id
        && !is_valid_session_id(id)
    {
        return Err(format!(
            "--session-id {id:?} is invalid: must contain only letters, digits, '.', '_', '-', \
                 and start/end with a letter or digit — it becomes part of a filesystem path"
        )
        .into());
    }
    let mut timing = beyond_ai_agent::timing::StartupTiming::new();
    let cwd = canonical_cwd(&std::env::current_dir().unwrap_or_default());

    // A stored `agent settings` default sits between an explicit flag/env var and this crate's own
    // built-in default — checked here, once, rather than threading `SettingsStore` through every
    // individual flag's own resolution site. Feature 2 (Round 3, pi-parity): merges a trusted project's
    // own `<cwd>/.beyond/settings.json` (and `.claude/settings.json`) tier on top of the global one first — see
    // `settings::effective_settings_for_cwd`'s own doc comment for the trust-gating rationale.
    //
    // Task #35/#36 (pi-parity fix): `stored_settings`, `block_images`, and `image_auto_resize` are
    // resolved here — before the `@file` composition block just below — rather than in their original
    // spot further down (right before `bash_shell_path`'s own resolution), because `read_file_refs`
    // needs both flags already resolved to correctly gate a CLI `@file.png` attachment. Previously
    // `read_file_refs` ran first, using a bare `tools::read::Read::default()` wholly unaware of either
    // flag — `--block-images`/`agent settings --block-images` had no effect at all on a CLI-attached
    // image (only on a model-issued `read` tool call, via `Agent::block_images`'s tool-dispatch gate),
    // and `--no-image-auto-resize` was silently ignored for the same reason. Every other
    // `stored_settings`-backed fallback stays resolved in its original place below.
    let stored_settings = beyond_ai_agent::settings::effective_settings_for_cwd(&cwd);
    // Task #26 (pi-parity feature): an explicit `--block-images` always wins; otherwise fall back to a
    // persisted `agent settings --block-images` default before finally defaulting to images allowed —
    // same "explicit flag, then stored setting, then built-in default" precedence every other
    // `stored_settings`-backed fallback here follows.
    //
    // Pass 20 (pi-parity fix): `--no-block-images` wins outright over all of that — the escape hatch
    // that was previously missing for forcing `block_images` off for one invocation when a persisted
    // `agent settings --block-images` default is `true`, mirroring `--no-image-auto-resize`'s identical
    // shape just below for the oppositely-defaulted `image_auto_resize`.
    let block_images =
        !no_block_images && (block_images || stored_settings.block_images.unwrap_or(false));
    // Task #4 (pi-parity feature): same "explicit flag, then stored setting, then built-in default"
    // precedence as `block_images` above, adapted for a negating `--no-x` flag: an explicit
    // `--no-image-auto-resize` always forces it off; otherwise fall back to the persisted
    // `agent settings --image-auto-resize` default; otherwise resize stays on (pi's own
    // `ImageSettings.autoResize` default).
    let image_auto_resize =
        !no_image_auto_resize && stored_settings.image_auto_resize.unwrap_or(true);
    // Whether the operator explicitly passed `--model`, as opposed to `run` falling back to a stored
    // default or `DEFAULT_MODEL` — the distinction a reopened `--session`/`--continue` needs below to
    // know whether to keep going on the model the session was actually last driven on instead of
    // quietly switching it, the same bug class `switch_session` had (see
    // `Persistence::model_and_level_at_active` in `serve/persistence.rs`). A merely-stored default counts as *not*
    // explicit here — same as an unset flag — since the operator didn't ask for this specific
    // invocation to use it.
    //
    // Task 3 (pi-parity fix, pass 19): resolved here, before the `@file` composition block just below —
    // same reasoning as `block_images`/`image_auto_resize` above — rather than in this block's original
    // spot further down (right before `key`'s own resolution): `read_file_refs` needs the model's real
    // `supports_vision` capability to correctly gate a CLI `@file.png` attachment's
    // `_model_supports_vision` field, exactly like the model-issued `read` tool-call dispatch path in
    // `agent_core::agent` already does (`capabilities(&self.model).supports_vision && !this.block_images`)
    // — previously this call site only ANDed in `!block_images`, omitting the model's real capability
    // entirely, so a non-vision model with `--block-images` left at its default (`false`) got a CLI
    // attachment dispatched with `_model_supports_vision: true`, and `read.rs` never appended its
    // non-vision-image explanatory note the way an equivalent model-issued call would have.
    let model_explicit = model.is_some();
    let model = model
        .or_else(|| stored_settings.default_model.clone())
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());
    // Fix 10 (pi-parity feature): resolve a partial/fuzzy `--model` id against the known-model hint list
    // before it's used for anything else — dialect inference, OAuth-provider inference, and the model
    // itself all key off the *resolved* id, so this must happen before `resolve_gateway_credential`
    // below. An ambiguous partial match fails the whole invocation clearly (naming every candidate)
    // rather than guessing; a genuinely unrecognized id (no partial match at all) is forwarded
    // unchanged — see `serve::resolve_model_id`'s own doc comment. Fix 2 (pi-parity gap): `model` may
    // also carry a trailing `:<level>` suffix (e.g. `sonnet:high`, pi's own `--model
    // <pattern>:<thinking-level>` shorthand) — `resolve_model_id` returns it separately, applied to
    // `reasoning_effort` further below.
    let (model, model_thinking_level) = serve::resolve_model_id(&model, serve::available_models())
        .map_err(|e| format!("--model {model:?}: {e}"))?;
    let (cwd_str, selected) = select_session(
        continue_session,
        &cwd,
        fork,
        no_session_persistence,
        session_dir,
        &session_id,
        session_path,
        &stored_settings,
    )?;
    // The model the resumed session is on at its active tip: the last recorded model change on that
    // branch (`serve`'s `model_and_level_at_active`), else the session's creation model. It wins
    // unless `--model` was given explicitly; a merely-stored default doesn't count.
    let resumed_model = match &selected {
        RunSession::Existing(opened) => Some(
            opened
                .0
                .active_ids()
                .last()
                .and_then(|id| opened.0.model_at(id))
                .unwrap_or(&opened.0.meta().model)
                .to_string(),
        ),
        RunSession::Fresh(_) | RunSession::Ephemeral => None,
    };
    let (model, model_thinking_level) = match &resumed_model {
        Some(resumed) if !model_explicit => (resumed.clone(), None),
        _ => (model, model_thinking_level),
    };
    // Task 3 (pi-parity fix, pass 19): the exact capability the model-issued `read` tool-call dispatch
    // path already ANDs into `_model_supports_vision` (see this block's own doc comment above) — computed
    // once here, now that `model` is fully resolved, and threaded down to `read_file_refs` below.
    let model_supports_vision = agent_core::capabilities(&model).supports_vision;

    // Compose the first message from (in order) piped stdin, `@file` contents, then the first
    // plain-text message argument — mirroring the reference agent's own composition order. At least
    // one source must contribute something; a typo'd invocation with none of the three fails loudly
    // here rather than sending the model an empty prompt. An `@file` reference that's actually an image
    // (see `read_file_refs`) contributes no text of its own but still counts as "something", so an
    // invocation like `run @screenshot.png` with no other text still proceeds.
    let (file_refs, mut messages) = partition_tasks(tasks);
    // Stdin is read only when asked for: no message argument (`some-cmd | agent run`, or
    // `agent run @prompt.md < diff`), or an explicit `-`. Reading it unconditionally made
    // `run "task"` wait for an EOF that never comes under `ssh host agent run …` without `-t`,
    // `docker exec -i`, or any parent holding a pipe open, and made `while read …; do agent run …`
    // swallow the rest of the loop's input as its prompt.
    let stdin_content = match messages.first().map(String::as_str) {
        Some("-") => {
            messages.remove(0);
            read_stdin_explicit()
        }
        Some(_) => None,
        None => read_stdin_if_piped(),
    };
    let file_refs = read_file_refs(
        &file_refs,
        &cwd,
        block_images,
        image_auto_resize,
        model_supports_vision,
    )
    .await?;
    let initial_images = file_refs.images;
    let mut parts = Vec::new();
    if let Some(s) = stdin_content {
        parts.push(s);
    }
    if !file_refs.text.is_empty() {
        parts.push(file_refs.text);
    }
    if !messages.is_empty() {
        parts.push(messages.remove(0));
    }
    if parts.is_empty() && initial_images.is_empty() {
        return Err("no task given: pass a message, an @file, or pipe input via stdin".into());
    }
    let initial_message = parts.join("");
    timing.mark("compose initial message");
    Ok(Composed {
        block_images,
        cwd,
        cwd_str,
        image_auto_resize,
        initial_images,
        initial_message,
        messages,
        model,
        model_thinking_level,
        resumed_model,
        selected,
        stored_settings,
        timing,
    })
}

/// Flag, then stored setting, then built-in default, for every setting `run` takes; trust and skill discovery.
fn resolve_defaults(
    cmd: Command,
    composed: Composed,
) -> Result<Resolved, Box<dyn std::error::Error>> {
    let Command::Run {
        gateway_url,
        key,
        reasoning_effort,
        steering_mode,
        follow_up_mode,
        trust_project,
        force_untrusted,
        compaction_reserve_tokens,
        compaction_keep_recent_tokens,
        branch_summary_reserve_tokens,
        retry_max_retries,
        retry_base_delay_ms,
        retry_max_backoff_ms,
        idle_timeout_ms,
        bash_shell_path,
        bash_command_prefix,
        no_skills,
        no_prompt_templates,
        extra_skill_paths,
        extra_prompt_template_paths,
        ..
    } = cmd
    else {
        unreachable!("dispatched on Command::Run")
    };
    let Composed {
        block_images,
        cwd,
        cwd_str,
        image_auto_resize,
        initial_images,
        initial_message,
        messages,
        model,
        model_thinking_level,
        resumed_model,
        selected,
        stored_settings,
        mut timing,
    } = composed;

    // A gateway is *configured* only if someone actually said so — `--gateway-url`/`AI_GATEWAY_URL`, a
    // stored `default_gateway_url`, or a `--key`/`AI_AGENT_KEY` to present to it. `DEFAULT_GATEWAY` is a
    // fallback, not configuration: treating a fallback as a configured gateway is precisely what made the
    // gateway mandatory, since the agent would then always believe it had one and never route direct.
    let configured_gateway = gateway_url
        .or_else(|| stored_settings.default_gateway_url.clone())
        .or_else(|| key.is_some().then(|| DEFAULT_GATEWAY.to_string()));
    let provider_env = beyond_ai_agent::gateway_credential::ProviderEnv::from_process_env(
        configured_gateway.is_some(),
    );
    // Still materialized: in direct mode every credential carries a `RouteOverride::Direct` that replaces
    // this base URL outright, so it is never dialed — no need to make the type `Option` all the way down.
    let gateway = configured_gateway.unwrap_or_else(|| DEFAULT_GATEWAY.to_string());
    // Round 3 (pi-parity fix): five more flag/env-only settings gain the same "explicit flag/env, then
    // stored setting, then built-in default" precedence — `serve`'s identical block, in its own command
    // handler (`commands/serve.rs`), for the full set (including `--models`, `serve`-only).
    let bash_shell_path =
        bash_shell_path.or_else(|| stored_settings.default_bash_shell_path.clone());
    // Task #49 (pi-parity fix): `Command::Serve`'s handler (in `commands/serve.rs`) checks this upfront and
    // fails fast — `run_task` had no equivalent at all, so a bad `--bash-shell-path`/stored default
    // surfaced only as a confusing spawn error on the first `bash` call, potentially well into a
    // multi-step run, rather than failing the invocation immediately. Validated once the whole
    // fallback chain (explicit flag, then stored setting) has resolved, matching `Command::Serve`'s
    // own identical placement/comment.
    if let Some(path) = &bash_shell_path
        && !std::path::Path::new(path).exists()
    {
        return Err(format!("--bash-shell-path not found: {path}").into());
    }
    let bash_command_prefix =
        bash_command_prefix.or_else(|| stored_settings.default_bash_command_prefix.clone());
    let compaction_reserve_tokens =
        compaction_reserve_tokens.or(stored_settings.default_compaction_reserve_tokens);
    let compaction_keep_recent_tokens =
        compaction_keep_recent_tokens.or(stored_settings.default_compaction_keep_recent_tokens);
    // Task #31 (pi-parity feature): independent of `compaction_reserve_tokens` — see
    // `agent_core::Agent::with_branch_summary_reserve_tokens`'s own doc comment.
    let branch_summary_reserve_tokens =
        branch_summary_reserve_tokens.or(stored_settings.default_branch_summary_reserve_tokens);
    let retry_max_retries = retry_max_retries.or(stored_settings.default_retry_max_retries);
    let retry_base_delay_ms = retry_base_delay_ms.or(stored_settings.default_retry_base_delay_ms);
    // Task #30 (pi-parity feature): the retry cluster's third knob — see
    // `agent_core::client::GatewayClient::with_max_backoff`'s own doc comment.
    let retry_max_backoff_ms =
        retry_max_backoff_ms.or(stored_settings.default_retry_max_backoff_ms);
    let idle_timeout_ms = idle_timeout_ms.or(stored_settings.default_provider_timeout_ms);
    let extra_skill_paths = if extra_skill_paths.is_empty() {
        stored_settings
            .default_skill_paths
            .clone()
            .unwrap_or_default()
    } else {
        extra_skill_paths
    };
    let extra_prompt_template_paths = if extra_prompt_template_paths.is_empty() {
        stored_settings
            .default_prompt_template_paths
            .clone()
            .unwrap_or_default()
    } else {
        extra_prompt_template_paths
    };
    // `model`/`model_thinking_level` were already fully resolved above (before the `@file` composition
    // block), so `read_file_refs` could see the model's real `supports_vision` capability — see that
    // resolution's own doc comment (Task 3, pi-parity fix, pass 19).
    // Captured before the shadowing below turns `key` into a resolved `GatewayCredential` — the
    // subagent factory needs the *raw* key to re-resolve for a child model (credentials are model-keyed).
    let raw_gateway_key = key.clone();
    // One `models.json` parse feeds both the credential and the `with_extra_headers` call far below
    // (T9-F3) — captured here rather than re-parsing the file a second time at the header site.
    let (key, model_extra_headers) =
        resolve_gateway_credential_and_headers(key, &model, &provider_env)?;
    // Task #29 (pi-parity fix): whether the operator explicitly requested a specific reasoning depth
    // for *this* invocation — an explicit `--reasoning-effort` flag, or a `--model <pattern>:<level>`
    // suffix (`model_thinking_level`, including `:off`) — as opposed to neither ever being given at
    // all. `ThinkingLevel::Off.reasoning_effort()` is `None`, the exact same value a bare,
    // nothing-requested invocation produces, so without this flag an explicit `:off` is
    // indistinguishable from "say nothing" by the time it reaches the `default_reasoning_effort_for_model`
    // fallback below — which silently overrode it back to a default depth (usually medium) instead of
    // actually turning reasoning off. Mirrors `Command::Serve`'s identical `reasoning_effort_explicit`
    // tracking, in `commands/serve.rs`.
    let reasoning_effort_explicit = reasoning_effort.is_some() || model_thinking_level.is_some();
    // Fix 2 (pi-parity gap): `--reasoning-effort` previously had no persisted stored-default fallback at
    // all, unlike `default_model`/`default_gateway_url`/`default_session_dir`. Precedence, in order: an
    // explicit `--reasoning-effort` flag; else a `--model <pattern>:<level>` suffix (this invocation's
    // own model-scoped request, same standing as the flag itself — see `model_thinking_level` above);
    // else the stored setting. Converted from the portable `ThinkingLevel` (off included) down to the
    // wire-level `ReasoningEffort` (`None` for `Off`) only at the very end, once every candidate source
    // has had its turn — Task 2 (pi-parity fix, pass 19): `off` is now a legal value at each of those
    // layers (flag, model suffix, stored setting), not just the `--model <pattern>:off` suffix.
    let reasoning_effort = reasoning_effort
        .or(model_thinking_level)
        .or_else(|| {
            stored_settings
                .default_reasoning_effort
                .as_deref()
                .and_then(|s| parse_reasoning_effort(s).ok())
        })
        .and_then(|level| level.reasoning_effort());
    // Fix 1 (pi-parity gap) — pi's own "medium" default (`DEFAULT_THINKING_LEVEL`,
    // `packages/coding-agent/src/core/defaults.ts`) whenever the model supports reasoning at all, so a
    // bare invocation with no flags doesn't silently wire-disable thinking the way leaving this `None`
    // does (see `serve::default_reasoning_effort_for_model`'s own doc comment) — finally `None` for a
    // model with no reasoning mechanism to default at all. Task #29: never consulted when the operator
    // explicitly asked for `:off` above — `reasoning_effort_explicit` guards this fallback so an
    // explicit "off" stays off instead of being silently promoted to the default depth.
    let reasoning_effort = if reasoning_effort_explicit {
        reasoning_effort
    } else {
        reasoning_effort.or_else(|| serve::default_reasoning_effort_for_model(&model))
    };

    // Task 1 (pi-parity fix, pass 19): `steering_mode`/`follow_up_mode` previously had no persisted
    // stored-default fallback at all — pi's own persisted `steeringMode`/`followUpMode` apply at
    // agent/session construction time in every mode, not just its TUI, but this crate only ever exposed
    // them as `serve`-only runtime RPC commands (`set_steering_mode`/`set_follow_up_mode`); an operator
    // wanting a standing default had to re-issue that RPC call at the start of every `serve` session, and
    // `run` had no way to request either at all. Same "explicit flag/env, then stored setting, then
    // built-in default" precedence as every other `stored_settings`-backed fallback above.
    let steering_mode = steering_mode
        .or_else(|| {
            stored_settings
                .steering_mode
                .as_deref()
                .and_then(|s| s.parse::<agent_core::QueueMode>().ok())
        })
        .unwrap_or_default();
    let follow_up_mode = follow_up_mode
        .or_else(|| {
            stored_settings
                .follow_up_mode
                .as_deref()
                .and_then(|s| s.parse::<agent_core::QueueMode>().ok())
        })
        .unwrap_or_default();
    // `run` has no live control channel to change either mode mid-invocation the way `serve`'s RPC
    // commands do, so this is applied once, up front, and never revisited — still enough to give an
    // operator-configured standing default (via `agent settings`'s persisted `steering_mode`/
    // `follow_up_mode`, written by `serve`'s own RPC handlers) actual effect on a `run` invocation, which
    // is the whole point of this fix. `run` currently has no mechanism that ever queues a steer/follow-up
    // message mid-invocation either (its `tasks` list runs as separate, sequential turns — see `tasks`'s
    // own doc comment — not steer injections), so this has no observable effect on today's behavior; it
    // establishes correct, real construction-time wiring — matching pi's own agent/session construction
    // in every mode — rather than leaving `run` on `Steering::new()`'s bare built-in defaults regardless
    // of what the operator configured.
    let steering = agent_core::Steering::new();
    steering.set_steering_mode(steering_mode);
    steering.set_follow_up_mode(follow_up_mode);

    // Computed once and reused below (rather than called again inside the warning check) — it's a
    // filesystem walk (`has_trust_gated_resources`'s own doc comment), not free.
    let has_gated_resources = beyond_ai_agent::trust_store::has_trust_gated_resources(&cwd);
    // `--trust-project`/`--force-untrusted` always win outright when explicitly given; failing that, an
    // explicit per-path `TrustStore` entry (`agent trust`/`agent untrust <path>`) wins next; only when
    // neither applies does a persisted `agent settings --default-project-trust` policy take effect —
    // Fix 1 (pi-parity bug): this used to check the blanket policy *before* the per-path entry, so an
    // operator's specific exception for one directory could be silently overridden by a coarser
    // `never`/`always` default. `serve::resolve_project_trust` is the one shared implementation of this
    // precedence — `run` and `serve` must agree on trust for the same directory under the same
    // settings, so it isn't duplicated here.
    let project_trusted = serve::resolve_project_trust(
        trust_project,
        force_untrusted,
        stored_settings.default_project_trust,
        beyond_ai_agent::trust_store::TrustStore::open_default().lookup(&cwd),
        has_gated_resources,
    );
    // pi-parity fix: an untrusted project with a `SYSTEM.md`/skills/prompts on disk silently skipped all
    // of them with no signal at all that anything was there — an operator debugging "why isn't my
    // SYSTEM.md taking effect" had nothing to go on. One line, matching this function's existing
    // `warning: ...` convention (see the `cwd_is_stale` check further down).
    if !project_trusted && has_gated_resources {
        eprintln!(
            "warning: {} has a project-local SYSTEM.md/APPEND_SYSTEM.md, skills, prompt templates, or a \
             settings.json on disk, but the project isn't trusted, so they were skipped — pass \
             --trust-project or run `{program} trust {}` to enable them (a project's own \
             settings.json additionally requires a *persisted* `{program} trust`, not just a one-off \
             --trust-project)",
            cwd.display(),
            cwd.display(),
            program = beyond_ai_agent::PROGRAM
        );
    }
    // Discovered once, up front: a one-shot `run` has no `reload` to re-discover mid-process, unlike
    // `serve`. `/skill:name` and `/name` prompt-template invocations are expanded here exactly like
    // `serve`'s own "prompt" handler does — this was previously silently skipped in `run`, so a message
    // starting with either was sent to the model as a literal, unexpanded string instead.
    // `--no-skills`/`--no-prompt-templates` skip *standard-root* discovery outright rather than
    // discovering and then discarding — matching pi's own flags, and avoiding a needless filesystem walk
    // when the operator has already said neither standard root is wanted. An explicit `--skill`/
    // `--prompt-template` extra path is still honored even so — pi's own `noSkills`/`noPromptTemplates`
    // do the same (a documented, tested combination; see `skills::discover_extra_only`'s doc comment —
    // pi-parity fix, M2), so `--no-skills --skill ./foo` isn't a self-contradicting no-op.
    let skills = if no_skills {
        beyond_ai_agent::skills::discover_extra_only(&extra_skill_paths).0
    } else {
        beyond_ai_agent::skills::discover(&cwd, project_trusted, &extra_skill_paths)
    };
    let prompt_templates = if no_prompt_templates {
        beyond_ai_agent::prompts::discover_extra_only(&extra_prompt_template_paths).0
    } else {
        beyond_ai_agent::prompts::discover(&cwd, project_trusted, &extra_prompt_template_paths)
    };
    timing.mark("discover skills/prompt templates");
    Ok(Resolved {
        bash_command_prefix,
        bash_shell_path,
        block_images,
        branch_summary_reserve_tokens,
        compaction_keep_recent_tokens,
        compaction_reserve_tokens,
        cwd,
        cwd_str,
        gateway,
        idle_timeout_ms,
        image_auto_resize,
        initial_images,
        initial_message,
        key,
        messages,
        model,
        model_extra_headers,
        project_trusted,
        prompt_templates,
        provider_env,
        raw_gateway_key,
        reasoning_effort,
        resumed_model,
        retry_base_delay_ms,
        retry_max_backoff_ms,
        retry_max_retries,
        selected,
        skills,
        steering,
        stored_settings,
        timing,
    })
}

/// Connects MCP, builds the tool registry, opens the session and builds the system prompt.
async fn assemble_run(
    cmd: Command,
    resolved: Resolved,
) -> Result<Assembled, Box<dyn std::error::Error>> {
    let Command::Run {
        system_prompt,
        append_system_prompt,
        compaction_candidates,
        web_allow_private,
        web_allow_host,
        web_timeout_ms,
        exec_url,
        exec_header,
        exec_max_response_bytes,
        exec_cmd,
        lifecycle_url,
        lifecycle_header,
        tools: tools_allow,
        exclude_tools: tools_exclude,
        no_tools,
        code_mode,
        mcp_tool_loading,
        deny_tool,
        deny_bash_pattern,
        deny_path,
        no_skills,
        approve_mcp_skills,
        no_context_files,
        name,
        prompt_guidelines,
        session_id,
        memory,
        no_memory,
        no_session_memory,
        output_schema,
        output_description,
        ..
    } = cmd
    else {
        unreachable!("dispatched on Command::Run")
    };
    let mcp_loading = mcp_tool_loading.unwrap_or_default();
    let Resolved {
        bash_command_prefix,
        bash_shell_path,
        block_images,
        branch_summary_reserve_tokens,
        compaction_keep_recent_tokens,
        compaction_reserve_tokens,
        cwd,
        cwd_str,
        gateway,
        idle_timeout_ms,
        image_auto_resize,
        initial_images,
        initial_message,
        key,
        messages,
        model,
        model_extra_headers,
        project_trusted,
        prompt_templates,
        provider_env,
        raw_gateway_key,
        reasoning_effort,
        resumed_model,
        retry_base_delay_ms,
        retry_max_backoff_ms,
        retry_max_retries,
        selected,
        skills,
        steering,
        stored_settings,
        mut timing,
    } = resolved;
    let (mcp_live, mcp_catalog, mcp_enabled, mcp_skills, mcp_built_at, mcp_tools) =
        connect_run_mcp(
            approve_mcp_skills,
            no_skills,
            &skills,
            &stored_settings,
            &mut timing,
        )
        .await;
    let (exec_runner, lifecycle, fs_backend) = open_exec_backend(
        exec_cmd,
        exec_header,
        exec_max_response_bytes,
        exec_url,
        lifecycle_header,
        lifecycle_url,
    )
    .await?;
    let path_world = fs_backend
        .as_ref()
        .map_or(tools::fs::PathWorld::Local, |backend| backend.world());
    let (bash_jobs, mut registry) = build_run_registry(
        &bash_command_prefix,
        &bash_shell_path,
        code_mode,
        &deny_tool,
        &exec_runner,
        &fs_backend,
        image_auto_resize,
        mcp_built_at,
        &mcp_enabled,
        &mcp_live,
        mcp_loading,
        &mcp_tools,
        no_tools,
        &skills,
        &steering,
        tools_allow,
        &tools_exclude,
        &web_allow_host,
        web_allow_private,
        web_timeout_ms,
    );
    // `--session-id`, when given, applies only where a *new* `SessionMeta` is actually minted below —
    // reopening an existing `--session <path>` or resuming via `--continue` already has a fixed id from
    // disk. Matches pi's own `--session-id`: a known, predictable id for a script/test harness to
    // correlate against, instead of parsing it back out of the run's own output.
    // `--name`: seeded here for the in-memory-only case (no store at all, so the post-hoc check below
    // never runs) and for a brand-new `--session <path>` file (already fresh, so that check is a
    // harmless no-op there). The `--continue` and reopened-`--session` cases are handled uniformly by
    // that check instead, since they don't go through this closure — see its comment below.
    let fresh_meta = || {
        let mut meta = match &session_id {
            Some(id) => SessionMeta::with_id(id.clone(), &cwd_str, &model),
            None => SessionMeta::new(&cwd_str, &model),
        };
        meta.title = name.clone();
        meta
    };
    // Persistent, cross-session memory. Resolved here (before the subagent context is built) so the same
    // backend `Arc` is shared with every child — the whole subagent tree reads/writes one durable store.
    // Registered *after* `apply_filter` (like `structured_output`/`subagent`) so a `--tools` allow-list
    // can't strip the agent's memory; `--no-memory`/`--no-tools` opt out. A bad backend fails fast before
    // a model call is billed, and before a session is created (see below). The `MEMORY.md` index is
    // injected into the system prompt below.
    let memory_backend: Option<Arc<dyn beyond_ai_agent::memory::MemoryBackend>> =
        if no_memory || no_tools {
            None
        } else {
            let dsn = memory.or_else(|| stored_settings.default_memory_backend.clone());
            Some(
                beyond_ai_agent::memory::open(dsn.as_deref(), &cwd).unwrap_or_else(|e| {
                    eprintln!("{e}");
                    exit_process(2);
                }),
            )
        };

    // Every check that can fail this invocation runs before the session is opened: a session created
    // and then abandoned by a failed invocation is what the next `--continue` would otherwise resume.
    // `--output-schema` is loaded here for that reason, and registered further down.
    let compaction_candidates =
        agent_core::CompactionCandidates::from_names(&compaction_candidates.unwrap_or_default())?;
    let output_slot = tools::structured_output::OutputSlot::new();
    let structured_output_tool = output_schema.as_ref().map(|arg| {
        let schema = tools::structured_output::load_schema(arg).unwrap_or_else(|e| {
            eprintln!("{e}");
            exit_process(2);
        });
        tools::structured_output::StructuredOutput::new(
            schema,
            output_description,
            output_slot.clone(),
        )
        .unwrap_or_else(|e| {
            eprintln!("{e}");
            exit_process(2);
        })
    });
    let (mut store, session, _session_lock) =
        open_run_session(&fresh_meta, &model, resumed_model, selected)?;
    let (mounts, memory_sections, session_memory_active) = mount_memory(
        &fresh_meta,
        memory_backend,
        no_session_memory,
        &mut registry,
        &store,
    )
    .await;
    // `--output-schema` turns this run into a callable function: the model must fill the schema in via
    // `structured_output` rather than ending in prose, and the validated payload is this process's real
    // return value (printed last, and the difference between exit 0 and exit 1 below).
    //
    // Registered *after* `apply_filter`, like `subagent` below: a `--tools read` allow-list is about
    // scoping what the agent may *do*, and must not silently strip the one tool this flag exists to add
    // — leaving a run that can never satisfy the contract it was started with.
    //
    // Fail fast on a bad schema, before a single model call is billed — the same discipline
    // `ToolPolicy::validate_deny_path_patterns` uses for a malformed `--deny-path` glob.
    // (The schema itself was loaded and checked before the session was opened — see above.)
    let wants_structured_output = structured_output_tool.is_some();
    if let Some(tool) = structured_output_tool {
        registry.register(Arc::new(tool));
    }

    let (write_locks, agent_defs, parent_loop) = register_subagent(
        bash_command_prefix,
        bash_shell_path,
        code_mode,
        &cwd,
        &deny_bash_pattern,
        &deny_path,
        &deny_tool,
        &exec_runner,
        fs_backend,
        &gateway,
        idle_timeout_ms,
        image_auto_resize,
        &mcp_enabled,
        mcp_live,
        mcp_loading,
        &model,
        mounts,
        no_context_files,
        project_trusted,
        &prompt_guidelines,
        provider_env,
        raw_gateway_key,
        &mut registry,
        retry_base_delay_ms,
        retry_max_backoff_ms,
        retry_max_retries,
        &skills,
        tools_exclude,
        web_allow_host,
        web_allow_private,
        web_timeout_ms,
    );
    let system = build_run_system(
        agent_defs,
        append_system_prompt,
        &cwd,
        &mcp_skills,
        memory_sections,
        &model,
        no_context_files,
        project_trusted,
        prompt_guidelines,
        &registry,
        &skills,
        system_prompt,
        &mut timing,
    );
    let meta = open_run_meta(
        &cwd,
        &fresh_meta,
        &model,
        &name,
        &session,
        &mut store,
        &mut timing,
    )?;
    Ok(Assembled {
        _session_lock,
        bash_jobs,
        block_images,
        branch_summary_reserve_tokens,
        compaction_candidates,
        compaction_keep_recent_tokens,
        compaction_reserve_tokens,
        exec_runner,
        gateway,
        idle_timeout_ms,
        initial_images,
        initial_message,
        key,
        lifecycle,
        mcp_catalog,
        mcp_enabled,
        mcp_skills,
        messages,
        meta,
        model,
        model_extra_headers,
        output_slot,
        parent_loop,
        path_world,
        prompt_templates,
        reasoning_effort,
        registry,
        retry_base_delay_ms,
        retry_max_backoff_ms,
        retry_max_retries,
        session,
        session_memory_active,
        skills,
        steering,
        store,
        stored_settings,
        system,
        wants_structured_output,
        write_locks,
    })
}

/// Builds the client and agent and attaches the job journal.
async fn build_agent(
    cmd: Command,
    assembled: Assembled,
) -> Result<RunSetup, Box<dyn std::error::Error>> {
    let Command::Run {
        max_steps,
        max_tokens,
        thinking,
        temperature,
        context_window,
        compact_at,
        no_tool_masking,
        tool_masking_keep_turns,
        tool_masking_keep_tokens,
        tool_masking_min_tokens,
        no_cached_summary,
        no_compaction,
        retry_rate_limit_secs,
        no_retry,
        sequential_tools,
        deny_tool,
        deny_bash_pattern,
        deny_path,
        json,
        cache_ttl,
        cache_long,
        ..
    } = cmd
    else {
        unreachable!("dispatched on Command::Run")
    };
    let cache_long = CacheTtl::resolve(cache_ttl, cache_long, CacheTtl::FiveMinutes);
    let Assembled {
        _session_lock,
        bash_jobs,
        block_images,
        branch_summary_reserve_tokens,
        compaction_candidates,
        compaction_keep_recent_tokens,
        compaction_reserve_tokens,
        exec_runner,
        gateway,
        idle_timeout_ms,
        initial_images,
        initial_message,
        key,
        lifecycle,
        mcp_catalog,
        mcp_enabled,
        mcp_skills,
        messages,
        meta,
        model,
        model_extra_headers,
        output_slot,
        parent_loop,
        path_world,
        prompt_templates,
        reasoning_effort,
        registry,
        retry_base_delay_ms,
        retry_max_backoff_ms,
        retry_max_retries,
        session,
        session_memory_active,
        skills,
        steering,
        store,
        stored_settings,
        system,
        wants_structured_output,
        write_locks,
    } = assembled;
    let client = build_run_client(
        gateway,
        idle_timeout_ms,
        key,
        model_extra_headers,
        retry_base_delay_ms,
        retry_max_backoff_ms,
        retry_max_retries,
    )?;
    // Task #50: the same two operator-supplied overrides also drive the *whole-run* retry layer
    // (`retry::RunRetryPolicy`), not just the pre-connect/mid-stream layer just above — previously
    // `--retry-max-retries`/`--retry-base-delay-ms` silently had no effect on `run_turn`'s own retry loop.
    // Task #52 (pi-parity fix): `--no-retry` wins outright over `retry_max_retries` here — but only for
    // this whole-run layer, not `client`'s own pre-connect/mid-stream retry configured just above, which
    // stays governed by `--retry-max-retries`/`--retry-base-delay-ms` alone (matches pi's own
    // `RetrySettings.enabled`, which gates only its equivalent whole-run loop).
    let retry_policy = beyond_ai_agent::retry::RunRetryPolicy::from_overrides_with_enabled(
        !no_retry,
        retry_max_retries,
        retry_base_delay_ms.map(std::time::Duration::from_millis),
    );
    let (store, run_journal) = attach_run_journal(&bash_jobs, exec_runner, &meta, store).await;
    // `serve`'s own resolution (`build_agent`): the model's window, input limit and compaction cap,
    // then the `--context-window`/`--compact-at`/reserve/keep-recent overrides.
    let compaction_flags = beyond_ai_agent::serve::CompactionFlags {
        context_window,
        compact_at,
        reserve_tokens: compaction_reserve_tokens,
        keep_recent_tokens: compaction_keep_recent_tokens,
        no_tool_masking,
        tool_masking_keep_turns,
        tool_masking_keep_tokens,
        tool_masking_min_tokens,
        no_cached_summary,
        candidates: compaction_candidates,
    };
    let mut compaction = beyond_ai_agent::serve::resolve_compaction(&model, &compaction_flags);
    if no_compaction {
        compaction.enabled = false;
    }
    let compaction_enabled = compaction.enabled;
    // The live-prompt size at which to warn the model that a compaction is approaching (see
    // `run_turn_once`), derived from the same trigger the agent will compact at.
    let pressure_point =
        beyond_ai_agent::memory::compaction_pressure_point(compaction.trigger_tokens());
    // Captured before each is moved into the builder chain below — `Agent` exposes no getter for
    // either back, and `run --export`'s own call to `export_html_full` further down needs the exact
    // system prompt/tool set this run actually used, not a recomputed (and possibly out-of-sync) guess.
    // Deferred MCP tools too: loadable all run long, so part of the tool set it had.
    let mut tool_defs = registry.definitions();
    tool_defs.extend(registry.deferred_definitions());
    let system_for_export = system.clone();
    let mut agent = Agent::new(Arc::new(client), model.clone())
        .with_tools(registry)
        .with_system(system)
        .with_max_steps(max_steps)
        .with_compaction(compaction)
        .with_cache_long(cache_long)
        .with_sequential_tools(sequential_tools)
        .with_rate_limit_retry(beyond_ai_agent::retry::rate_limit_policy(
            retry_rate_limit_secs,
        ))
        .with_cache_key(meta.id.clone())
        // Shared with every subagent (built in `register_subagent`) so the parent's own
        // edits and a child's serialize on the same path.
        .with_write_locks(write_locks.clone())
        .with_checkpoint_hook(Arc::new(DirectCheckpoint(store.clone())));
    // Task #31 (pi-parity feature): independent of `compaction_reserve_tokens` above — see
    // `agent_core::Agent::with_branch_summary_reserve_tokens`'s own doc comment.
    if let Some(reserve) = branch_summary_reserve_tokens {
        agent = agent.with_branch_summary_reserve_tokens(reserve);
    }
    // Unlike `serve`, `run` has no thinking-level cycling — these are applied as-is, with no per-model
    // default derivation when omitted (matching `run`'s prior behavior of not setting either at all).
    let mut effective_thinking = thinking;
    if let Some(budget) = thinking {
        agent = agent.with_thinking(budget);
    }
    if let Some(effort) = reasoning_effort {
        agent = agent.with_reasoning_effort(effort);
        // Task #36 (pi-parity feature): derive a numeric thinking budget from the effort level for
        // Budget/Adaptive-shape models when the operator didn't also pass an exact `--thinking` token
        // count (which still wins outright) — mirrors `serve::build_agent`'s identical
        // `thinking_for_level`-based derivation for its own initial `--reasoning-effort`, but consults
        // the operator's `agent settings --thinking-budget` override table first (falling back to
        // `agent_core`'s built-in ladder). Without this, `--reasoning-effort high` alone (no
        // `--thinking`) set no thinking budget at all for a token-budget-shape model (Claude 3.x/4.x):
        // that dialect's wire body only ever reads `thinking`, never `reasoning_effort`.
        if thinking.is_none() {
            let caps = agent_core::capabilities(&model);
            if matches!(
                caps.thinking,
                agent_core::models::ThinkingShape::Budget
                    | agent_core::models::ThinkingShape::Adaptive
            ) {
                let overrides = resolve_thinking_budget_overrides(&stored_settings);
                let budget = agent_core::models::budget_for_effort_with_override(
                    effort,
                    caps.max_output,
                    overrides.as_ref(),
                );
                effective_thinking = Some(budget);
                agent = agent.with_thinking(budget);
            }
        }
    }
    if let Some(temperature) = temperature {
        agent = agent.with_temperature(temperature);
    }
    if let Some(max_tokens) = max_tokens {
        agent = agent.with_max_tokens(max_tokens);
    }
    // Task #26 (pi-parity feature): forces every image down the vision-downgrade path regardless of
    // the active model's real `supports_vision` capability — see `--block-images`'s own doc comment.
    if block_images {
        agent = agent.with_block_images(true);
    }
    parent_loop.set(beyond_ai_agent::tools::subagent::ParentLoop {
        model: model.clone(),
        max_steps,
        compaction: compaction_flags,
        compaction_enabled,
        thinking: effective_thinking,
        reasoning_effort,
        block_images,
        rate_limit: beyond_ai_agent::retry::rate_limit_policy(retry_rate_limit_secs),
    });
    // Always installed: besides the deny-lists, the hooks carry the SEP-2640 code-execution gate,
    // which must hold even when no deny-list was given.
    let policy = run_policy(&deny_tool, &deny_bash_pattern, &deny_path, path_world);
    agent = agent.with_hooks(Arc::new(tools::mcp_skills::RunHooks {
        policy,
        mcp: mcp_enabled.clone(),
    }));

    if json {
        // A leading header line so a `--json` consumer can identify the session before any event
        // arrives — the same purpose `serve`'s persisted header line serves, just for a one-shot run
        // with no server/control-protocol involved. `"kind"` matches `AgentEvent`'s own tag field, so
        // every stdout line (header or event) discriminates on the same key.
        println!(
            "{}",
            serde_json::json!({ "kind": "session", "id": meta.id, "model": meta.model, "cwd": meta.cwd })
        );
        let _ = std::io::stdout().flush();
    }

    Ok(RunSetup {
        _session_lock,
        agent,
        bash_jobs,
        initial_images,
        initial_message,
        lifecycle,
        mcp_catalog,
        mcp_enabled,
        mcp_skills,
        messages,
        meta,
        output_slot,
        pressure_point,
        prompt_templates,
        retry_policy,
        run_journal,
        session,
        session_memory_active,
        skills,
        steering,
        store,
        system_for_export,
        tool_defs,
        wants_structured_output,
    })
}

/// `run`'s deny-lists, resolving `--deny-path` in the world the tools act on: with a remote exec
/// backend a `~` pattern means the target's home, not this host's. (`run`'s tools are rooted at the
/// process cwd, the policy's default root.)
fn run_policy(
    deny_tool: &[String],
    deny_bash_pattern: &[String],
    deny_path: &[String],
    world: tools::fs::PathWorld,
) -> ToolPolicy {
    ToolPolicy::from_lists(deny_tool, deny_bash_pattern, deny_path).in_world(world)
}

/// Runs the turns, then reports.
async fn drive_run(cmd: Command, runsetup: RunSetup) -> Result<(), Box<dyn std::error::Error>> {
    let Command::Run {
        max_steps,
        lifecycle_heartbeat_secs,
        export,
        json,
        ..
    } = cmd
    else {
        unreachable!("dispatched on Command::Run")
    };
    let RunSetup {
        _session_lock,
        agent,
        bash_jobs,
        initial_images,
        initial_message,
        lifecycle,
        mcp_catalog,
        mcp_enabled,
        mcp_skills,
        messages,
        meta,
        output_slot,
        pressure_point,
        prompt_templates,
        retry_policy,
        run_journal,
        mut session,
        session_memory_active,
        skills,
        steering,
        store,
        system_for_export,
        tool_defs,
        wants_structured_output,
    } = runsetup;
    // `run` previously registered no signal handler at all — Rust's default SIGTERM/SIGINT
    // disposition terminates the process immediately, running no destructors, so a bash tool's
    // `GroupKillGuard` (which only reaps on `Drop`) never gets to kill a still-running child's
    // process group, and any not-yet-persisted checkpoint from the current turn is lost outright.
    // Reusing `serve`'s own `ShutdownSignal` (rather than a second, subtly different
    // implementation) ties a shutdown request to the *same* `CancellationToken` plumbing
    // `run_events_cancellable` already drops tool futures through on an explicit `abort` — so a
    // `Ctrl-C`/`systemctl stop`/pod eviction now takes the identical clean-cancellation path
    // instead of a raw kill.
    let cancel = agent_core::CancellationToken::new();
    let shutdown_cancel = cancel.clone();
    // Task #41 (pi-parity fix): which signal (if any) actually triggered a cancellation, so
    // `unwrap_turn_result` below can exit with the matching POSIX code instead of the same bare
    // `exit(1)` every cancellation used to get regardless of cause. A real `Mutex` (not a bare local,
    // unlike `commands/serve.rs`'s own `shutdown_cause`) since this is genuinely shared across a task boundary:
    // the signal wait runs on its own spawned task, concurrently with the run this variable is read
    // from.
    let shutdown_cause: Arc<std::sync::Mutex<Option<serve::Signal>>> =
        Arc::new(std::sync::Mutex::new(None));
    let shutdown_cause_writer = shutdown_cause.clone();
    tokio::spawn(async move {
        if let Ok(mut shutdown) = serve::ShutdownSignal::new() {
            let sig = shutdown.wait().await;
            *shutdown_cause_writer
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(sig);
            shutdown_cancel.cancel();
            // The first signal asks the run to stop cooperatively. A wait that doesn't watch the
            // token (an MCP server that never answers a skill fetch, say) would otherwise leave a
            // second Ctrl-C doing nothing at all; a second signal means "stop now", with the usual
            // exit cleanup (job group kills, the stdio-server sweep) and its own code.
            let sig = shutdown.wait().await;
            eprintln!("\n[second signal: exiting now]");
            exit_process(sig.exit_code());
        }
    });
    // Set by `write_stdout_or_exit` on a broken stdout pipe (a downstream consumer that closed its
    // read end early, e.g. `agent run --json "task" | head -1`) — trips `cancel` the same as a real
    // shutdown signal so the turn stops cooperatively instead of `process::exit`ing from deep inside
    // a streaming callback, which used to skip the `persist_run` calls below entirely and
    // silently drop the turn's final (possibly-partial) assistant message. Checked right after each
    // `persist_run`, below, so the exit still happens (with the same code 0 as before) but only
    // once that write has actually landed.
    let broken_pipe = AtomicBool::new(false);

    // A resumed session whose last run was killed outright (OOM, `kill -9`, a CI cancel) ends on that
    // run's prompt, never answered: closed the way a failed run closes its own, as `serve` does on
    // open. Persisted with the new prompt by the first checkpoint.
    serve::close_interrupted_prompt(&mut session);
    // A resumed session (`--session-id`, `--continue`) whose transcript still holds an MCP skill's
    // SKILL.md is acting on that skill: the code-execution gate is restored with it.
    mcp_enabled
        .skill_session()
        .restore_from_transcript(&session.messages);
    let initial_message =
        expand_message(&initial_message, &skills, &prompt_templates, &mcp_skills).await;
    let (resumed, placeholders) =
        match resume_mcp_tasks_for_run(run_journal, &session, &mcp_catalog, &cancel).await {
            Ok(resumed) => resumed,
            // Cancelled (the only error): nothing of this run's turn exists yet, so there is
            // nothing to persist; exit as a cancelled turn does, with the signal's code.
            Err(_) => exit_cancelled(&shutdown_cause),
        };
    if initial_images.is_empty() {
        session.user(initial_message);
    } else {
        session.push(agent_core::Message::user_with_images(
            initial_message,
            initial_images,
        ));
    }
    if !resumed.is_empty() {
        // Into the turn just pushed: the call it answers is at the tip.
        let messages: &mut Vec<agent_core::Message> = Arc::make_mut(&mut session.messages);
        beyond_ai_agent::mcp_resume::splice(messages, &resumed, &placeholders);
    }
    // `--export` renders the transcript however the run ends: a failed or cancelled run's transcript
    // (everything up to the failure, its closing error record included) is the one most worth
    // reading, and the export used to sit after the `?`s that end those runs.
    let export_transcript = |session: &Session| {
        let Some(export) = &export else {
            return;
        };
        let (branches, events) = {
            let guard = store.lock().unwrap_or_else(|e| e.into_inner());
            match guard.as_ref() {
                Some(s) => (s.abandoned_branches(), s.export_events().to_vec()),
                None => (Vec::new(), Vec::new()),
            }
        };
        // `export_html_full` (Task #44 integration): the running agent's actual system prompt/tool
        // set, not the plainer `export_html_with_entries` this call site used before — so an exported
        // transcript's own System Prompt/Available Tools sections reflect what this run really used.
        // `session`'s own running token counters are right here too, so the stats section gets real
        // usage numbers rather than omitting that line entirely.
        match beyond_ai_agent::export::export_html_full(
            &meta,
            &session.messages,
            &branches,
            Some(beyond_ai_agent::export::UsageTotals {
                input_tokens: session.input_tokens,
                output_tokens: session.output_tokens,
                cache_read_tokens: session.cache_read_tokens,
                cache_write_tokens: session.cache_write_tokens,
            }),
            &events,
            Some(&system_for_export),
            Some(&tool_defs),
            Some(export),
        ) {
            Ok(path) => eprintln!("[exported transcript to {}]", path.display()),
            Err(e) => eprintln!("[failed to export transcript: {e}]"),
        }
    };
    let life = lifecycle.enabled().then(|| {
        beyond_ai_agent::lifecycle::Run::begin(
            std::sync::Arc::clone(&lifecycle),
            meta.id.clone(),
            // `run` is single-tenant by construction: no grant, so no tenant to report.
            None,
            None,
            meta.model.clone(),
        )
    });
    let turn_result = run_turn(
        &agent,
        &mut session,
        json,
        &cancel,
        &retry_policy,
        &broken_pipe,
        &steering,
        session_memory_active,
        pressure_point,
        life.as_ref(),
        std::time::Duration::from_secs(lifecycle_heartbeat_secs),
    )
    .await;
    // A yielded job never outlives the turn that started it (the stop reminder said so).
    let mut stopped_jobs = stop_jobs(&bash_jobs, tools::bash_jobs::BashJobs::end_turn).await;
    // Persist whatever's in `session` regardless of outcome: `run_events_cancellable` mutates
    // `session` in place as it streams, so a cancelled turn still leaves behind whatever
    // assistant/tool content had already landed — the same partial-content guarantee `serve` gives
    // every session, not just the happy path. `DirectCheckpoint` already covers most of this
    // incrementally, but the turn's own tail (its final, possibly-partial assistant message) is
    // only ever captured here.
    persist_run(&store, &session)?;
    if broken_pipe.load(Ordering::Relaxed) {
        export_transcript(&session);
        emit_cli_lifecycle(life.as_ref(), &session, CliTerminal::Aborted, None).await;
        exit_process(0);
    }
    if let Err(e) = &turn_result {
        export_transcript(&session);
        emit_cli_lifecycle(
            life.as_ref(),
            &session,
            match e {
                agent_core::Error::Cancelled => CliTerminal::Aborted,
                other => CliTerminal::Failed {
                    error: other.to_string(),
                    refused: false,
                },
            },
            None,
        )
        .await;
    }
    let mut stop_reason = unwrap_turn_result(turn_result, &shutdown_cause)?;
    for message in messages {
        session.user(expand_message(&message, &skills, &prompt_templates, &mcp_skills).await);
        let turn_result = run_turn(
            &agent,
            &mut session,
            json,
            &cancel,
            &retry_policy,
            &broken_pipe,
            &steering,
            session_memory_active,
            pressure_point,
            life.as_ref(),
            std::time::Duration::from_secs(lifecycle_heartbeat_secs),
        )
        .await;
        stopped_jobs.extend(stop_jobs(&bash_jobs, tools::bash_jobs::BashJobs::end_turn).await);
        persist_run(&store, &session)?;
        if broken_pipe.load(Ordering::Relaxed) {
            export_transcript(&session);
            emit_cli_lifecycle(life.as_ref(), &session, CliTerminal::Aborted, None).await;
            exit_process(0);
        }
        if let Err(e) = &turn_result {
            export_transcript(&session);
            emit_cli_lifecycle(
                life.as_ref(),
                &session,
                match e {
                    agent_core::Error::Cancelled => CliTerminal::Aborted,
                    other => CliTerminal::Failed {
                        error: other.to_string(),
                        refused: false,
                    },
                },
                None,
            )
            .await;
        }
        stop_reason = unwrap_turn_result(turn_result, &shutdown_cause)?;
    }

    export_transcript(&session);

    report_run_summary(
        bash_jobs,
        &broken_pipe,
        &cancel,
        json,
        max_steps,
        &session,
        stopped_jobs,
    )
    .await;
    report_run_outcome(
        broken_pipe,
        cancel,
        json,
        life,
        output_slot,
        session,
        stop_reason,
        wants_structured_output,
    )
    .await;
    // Text mode has no other failure signal a script/CI caller could key off of — a refusal would
    // otherwise still exit 0, indistinguishable from a normal completion, unless the last turn's
    // stop reason is checked explicitly here. JSON mode already carries `stop_reason` on every
    // `TurnEnd` event in its own output stream, so it's unaffected either way — see
    // `text_mode_failure_message`'s own doc comment for the exact contract (including why `Aborted` is
    // checked too, defensively, even though it's currently unreachable here).
    if let Some(message) = text_mode_failure_message(json, stop_reason) {
        eprintln!("{message}");
        exit_process(1);
    }
    Ok(())
}

/// Selects the session the flags name (`--fork`, `--session`, `--continue`, `--session-id`), or a fresh one.
#[allow(clippy::too_many_arguments)]
fn select_session(
    continue_session: bool,
    cwd: &Path,
    fork: Option<String>,
    no_session_persistence: bool,
    session_dir: Option<String>,
    session_id: &Option<String>,
    session_path: Option<String>,
    stored_settings: &beyond_ai_agent::settings::Settings,
) -> Result<(String, RunSession), Box<dyn std::error::Error>> {
    // `--session`/`--continue` persist this run (and load prior history to continue it) exactly like
    // `serve`'s own repo/file modes. pi-parity fix: neither given previously kept `run` in-memory-only —
    // pi's own default (no flags at all, including one-shot print-mode) is a persisted, disk-backed
    // session, matching `serve`'s own default repo-mode persistence; only an explicit
    // `--no-session-persistence` now opts back out to the old ephemeral behavior (see the final `None`
    // arm below).
    let cwd_str = cwd.to_string_lossy().into_owned();
    // `--session-dir` (matching `serve`'s own flag/env var exactly) redirects the repo root that
    // `--continue` and `--fork` both use, in place of the default `~/.beyond/sessions/<encoded-cwd>/`.
    // Its parent becomes `--fork`'s cross-project search root too — the same convention `serve`'s own
    // `list_all_sessions` already applies (`Persistence::list_all_with_progress`'s `repo.dir().parent()`)
    // when `--session-dir` is set there, so both binaries scope a cross-project scan identically.
    let (repo_dir, fork_search_root): (PathBuf, PathBuf) =
        match session_dir.or_else(|| stored_settings.default_session_dir.clone()) {
            Some(dir) => {
                let dir = PathBuf::from(dir);
                let search_root = dir
                    .parent()
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| dir.clone());
                (dir, search_root)
            }
            None => (default_session_dir(&cwd_str), sessions_root()),
        };
    // Select the session before anything model-keyed is resolved — the credential and route, the
    // `models.json` headers, vision, the reasoning default and the context window all key off the
    // model, and a resumed session's model is the one it is actually on. Existing sessions are opened
    // here; a fresh one is only *selected* (created after every check passes, see `RunSession`).
    let selected = if let Some(arg) = &fork {
        // `--fork` wins over `--session`/`--continue`: a fork always starts a fresh child session,
        // never reopens one in place, so there is no meaningful way to combine it with either.
        let target = SessionRepo::open(&repo_dir)?;
        let (store, session) = fork_by_arg(arg, &target, &cwd_str, &fork_search_root, usize::MAX)
            .map_err(|e| format!("--fork {arg:?}: {e}"))?;
        RunSession::Existing(Box::new((store, session)))
    } else {
        match session_path {
            Some(arg) => {
                let literal_path = PathBuf::from(&arg);
                // Task #24 (pi-parity fix): `--session <arg>` accepts either a literal path or a bare
                // session id, matching pi's own `resolveSessionPath` — previously this always treated
                // `arg` as a literal filesystem path, so a bare id (no `/`, no leading `.`/`~`, no
                // `.jsonl` suffix — almost certainly not an existing relative path) silently created an
                // empty, wrongly-named session file instead of reopening the one actually meant. A
                // path-like argument, or one that already exists as a literal file, is still used as-is
                // below (creating a fresh session there when absent, exactly as `--session` always has);
                // anything else is resolved as a session id against the current project's repo first,
                // then cross-project (`open_session_by_id`, the identical two-tier search `--fork <id>`
                // already does via `fork_by_arg`) and REOPENED in place — continuing that session, not
                // forking a new one, since that's what `--session` (unlike `--fork`) has always meant.
                if is_path_like(&arg) || literal_path.exists() {
                    // A zero-byte file at `literal_path` (e.g. `touch`'d ahead of time, or left over
                    // from a crash before the header write landed) has nothing to open — route it
                    // through `create`, which now initializes an empty file in place rather than failing
                    // (see its own doc comment).
                    let has_content = literal_path.metadata().is_ok_and(|m| m.len() > 0);
                    if has_content {
                        // pi-parity fix (C-M6): bare `?` here propagated the raw `std::io::Error`
                        // straight to `main`'s `Result`, which Rust's default `Termination` impl prints
                        // via `{:?}` — a Debug dump of the error's internal shape (`Custom { kind:
                        // InvalidData, error: "..." }`) with no file path at all, matching neither pi's
                        // own clear `"Error: Session file is not a valid pi session: <path>"` nor this
                        // project's own no-leaked-internals bar for user-facing errors. Wrapping in a
                        // plain `String` message (still `Error: "..."` once printed, but a human-readable
                        // sentence, not an internal struct shape) and naming the path fixes both: the
                        // operator now sees *which* file and *why*, instead of guessing.
                        let (store, session) =
                            SessionStore::open(literal_path.clone()).map_err(|e| {
                                format!(
                                    "session file is not a valid session: {}: {e}",
                                    literal_path.display()
                                )
                            })?;
                        RunSession::Existing(Box::new((store, session)))
                    } else {
                        RunSession::Fresh(FreshSession::AtPath(literal_path))
                    }
                } else {
                    let repo = SessionRepo::open(&repo_dir)?;
                    let (store, session) = open_session_by_id(&arg, &repo, &fork_search_root)
                        .map_err(|e| format!("--session {arg:?}: {e}"))?;
                    RunSession::Existing(Box::new((store, session)))
                }
            }
            // Pure in-memory. `--continue` still overrides `--no-session-persistence` (it always has:
            // asking to continue a persisted session is a direct contradiction of not persisting, and
            // the explicit verb wins), but `--session-id` does not — there it just names the ephemeral
            // session for correlation, exactly as it does in `serve`'s own no-persistence branch.
            None if no_session_persistence && !continue_session => RunSession::Ephemeral,
            // The selection ladder, most specific first — the same one `serve` applies via
            // `serve::SessionSelect`, so the flags mean identically the same thing in both binaries.
            None => {
                let repo = SessionRepo::open(&repo_dir)?;
                let existing = match (session_id.as_deref(), continue_session) {
                    // `--session-id` addresses one session outright: open it, or create it under
                    // exactly that id. It outranks `--continue`, which only *describes* a session
                    // ("whatever ran here last"). It used to be discarded outright whenever any session
                    // already existed for this cwd, which collapsed every distinct id in a shared
                    // directory onto one shared conversation.
                    (Some(id), _) => repo.open_id_if_exists(id)?,
                    (None, true) => repo.resume_latest(&cwd_str)?,
                    // A bare `run` starts a genuinely new session — persisted, so nothing is lost, but
                    // its own. It briefly reattached to this cwd's most recent session instead, which
                    // meant two shells in one repo drove the same store: `append_new` is count-keyed,
                    // so neither could observe the other's writes and the transcript interleaved into
                    // nonsense. `--continue` is how you ask for the old behavior, and it is now the
                    // only thing that reattaches implicitly.
                    (None, false) => None,
                };
                match existing {
                    Some((store, session)) => RunSession::Existing(Box::new((store, session))),
                    None => RunSession::Fresh(FreshSession::InRepo(repo)),
                }
            }
        }
    };
    Ok((cwd_str, selected))
}

/// Connects the configured MCP servers and opens the skill-session state they share.
async fn connect_run_mcp(
    approve_mcp_skills: bool,
    no_skills: bool,
    skills: &[beyond_ai_agent::skills::Skill],
    stored_settings: &beyond_ai_agent::settings::Settings,
    timing: &mut beyond_ai_agent::timing::StartupTiming,
) -> (
    tools::mcp::McpTools,
    tools::mcp::McpCatalog,
    tools::mcp::McpEnabledSet,
    tools::mcp_skills::McpSkills,
    u64,
    Vec<Arc<dyn Tool>>,
) {
    // Fail-soft (Task: MCP client support): a server that fails to connect is skipped with a warning,
    // not a fatal error — matches `has_gated_resources`'s own "warn, don't block the run" convention
    // in `assemble_run`. `stored_settings.mcp_servers` is already trust-gated (a project's own
    // `.beyond/settings.json` — where a project-tier `mcp_servers` entry would live — only merges in at
    // all when `effective_settings_for_cwd` found `cwd` trusted; the global tier always applies).
    let (mcp_live, mcp_catalog, mcp_warnings) = tools::mcp::connect_all(
        stored_settings.mcp_servers.as_deref().unwrap_or(&[]),
        tools::mcp::idle_reap_after_from_env(),
        mcp_manifest_dir().as_ref(),
    )
    .await;
    for warning in &mcp_warnings {
        eprintln!("warning: {warning}");
    }
    timing.mark("connect mcp servers");
    // SEP-2640 skills the connected servers publish. `run` has no `set_mcp_enabled`, so every
    // configured server stays enabled and the gate never moves. The one enabled set is shared by the
    // registry, the hooks and any subagent, so they agree on what this run loaded and approved. There
    // is no one to ask in `run`: without `--approve-mcp-skills`, approval questions are denied.
    let mcp_enabled = tools::mcp::McpEnabledSet::new();
    mcp_enabled
        .skill_session()
        .set_preapproved(approve_mcp_skills);
    // `--no-skills`: no MCP-served skills listed or expanded either.
    mcp_enabled
        .skill_session()
        .set_listing_suppressed(no_skills);
    let mcp_skills = tools::mcp_skills::McpSkills::new(&mcp_catalog, mcp_enabled.clone());
    // Skill-loading tools rebound to this run's skill state (see `filter_by_enabled`).
    // The generation the registry below is built from: a server that re-lists its tools after this
    // reaches the run through its tool source (installed on `steering` below).
    let mcp_built_at = mcp_live.generation();
    let mcp_tools = tools::mcp::filter_by_enabled(&mcp_live.snapshot(), &mcp_enabled);
    for diagnostic in mcp_skills.diagnostics(skills) {
        eprintln!("warning: {diagnostic}");
    }
    (
        mcp_live,
        mcp_catalog,
        mcp_enabled,
        mcp_skills,
        mcp_built_at,
        mcp_tools,
    )
}

/// The remote exec runner (if any), the lifecycle sink and the filesystem backend that follows the runner.
async fn open_exec_backend(
    exec_cmd: Option<String>,
    exec_header: Vec<String>,
    exec_max_response_bytes: Option<u64>,
    exec_url: Option<String>,
    lifecycle_header: Vec<String>,
    lifecycle_url: Option<String>,
) -> Result<
    (
        Option<Arc<dyn tools::exec::CommandRunner>>,
        Arc<dyn beyond_ai_agent::lifecycle::RunLifecycle>,
        Option<Arc<dyn tools::fs::FsBackend>>,
    ),
    Box<dyn std::error::Error>,
> {
    // Pointing at a remote endpoint swaps *one* thing: where the filesystem tools' I/O lands. The
    // tools, their names, descriptions and schemas are untouched, so the model cannot tell the
    // difference — see `tools::fs`.
    let exec_runner: Option<std::sync::Arc<dyn tools::exec::CommandRunner>> =
        match (&exec_url, &exec_cmd) {
            (Some(url), _) => {
                let mut runner = beyond_ai_agent::exec_endpoint::HttpExecRunner::new(url.clone())
                    .map_err(std::io::Error::other)?;
                if let Some(max) = exec_max_response_bytes {
                    runner =
                        runner.with_max_response_bytes(usize::try_from(max).unwrap_or(usize::MAX));
                }
                for raw in &exec_header {
                    let (name, value) =
                        beyond_ai_agent::exec_endpoint::HttpExecRunner::parse_header(raw)
                            .map_err(std::io::Error::other)?;
                    runner = runner.with_header(name, value);
                }
                Some(std::sync::Arc::new(runner))
            }
            (None, Some(template)) => Some(std::sync::Arc::new(
                beyond_ai_agent::exec_endpoint::TemplateRunner::parse(template)
                    .map_err(std::io::Error::other)?,
            )),
            (None, None) => None,
        };
    let lifecycle = beyond_ai_agent::lifecycle::open(lifecycle_url.as_deref(), &lifecycle_header)
        .map_err(std::io::Error::other)?;
    let fs_backend: Option<std::sync::Arc<dyn tools::fs::FsBackend>> = match &exec_runner {
        Some(runner) => {
            let backend = tools::fs::shell::ShellFs::connect(runner.clone()).await;
            // Report the rung, because the alternative is discovering it from a search that walked
            // `target/` — see `SearchEngine::PosixGrep`.
            eprintln!(
                "tools run on the remote exec endpoint, `bash` included (search engine: {:?})",
                backend.capabilities().search_engine()
            );
            Some(std::sync::Arc::new(backend))
        }
        None => None,
    };
    Ok((exec_runner, lifecycle, fs_backend))
}

/// The background-jobs table and the filtered tool registry (plus the MCP tool source on `steering`).
#[allow(clippy::too_many_arguments)]
fn build_run_registry(
    bash_command_prefix: &Option<String>,
    bash_shell_path: &Option<String>,
    code_mode: bool,
    deny_tool: &[String],
    exec_runner: &Option<Arc<dyn tools::exec::CommandRunner>>,
    fs_backend: &Option<Arc<dyn tools::fs::FsBackend>>,
    image_auto_resize: bool,
    mcp_built_at: u64,
    mcp_enabled: &tools::mcp::McpEnabledSet,
    mcp_live: &tools::mcp::McpTools,
    mcp_loading: tools::tool_search::McpLoading,
    mcp_tools: &[Arc<dyn Tool>],
    no_tools: bool,
    skills: &[beyond_ai_agent::skills::Skill],
    steering: &agent_core::Steering,
    tools_allow: Option<Vec<String>>,
    tools_exclude: &Option<Vec<String>>,
    web_allow_host: &[String],
    web_allow_private: bool,
    web_timeout_ms: Option<u64>,
) -> (Arc<tools::bash_jobs::BashJobs>, agent_core::ToolRegistry) {
    let empty: &[String] = &[];
    let nested_exclude = tools_exclude.as_deref().unwrap_or(empty);
    // `bash`'s jobs belong to the run: they outlive any one turn (a second `-m` message sees them),
    // and the end of the run stops them and says which (see the end of this function).
    let bash_jobs = std::sync::Arc::new(tools::bash_jobs::BashJobs::new(
        tools::bash_jobs::EndPolicy::StopAtEnd,
    ));
    let mut registry = tools::default_registry_with_config(&tools::ToolConfig {
        bash_jobs: Some(bash_jobs.clone()),
        bash_shell_path: bash_shell_path.as_deref(),
        bash_command_prefix: bash_command_prefix.as_deref(),
        web_allow_private,
        web_allow_hosts: web_allow_host,
        web_timeout_ms,
        image_auto_resize,
        code_mode,
        mcp_loading,
        nested_exclude,
        nested_deny: deny_tool,
        mcp_tools,
        fs_backend: fs_backend.clone(),
        command_runner: exec_runner.clone(),
        skills,
        ..tools::ToolConfig::new()
    });
    tools::apply_filter(
        &mut registry,
        tools_allow.as_deref(),
        tools_exclude.as_deref(),
        no_tools,
    );
    // A server that changes its tools during the run (`notifications/tools/list_changed`, or a start
    // that lists something new) has them offered from the next turn.
    steering.set_tool_source(Some(tools::mcp::tool_source(
        mcp_live.clone(),
        mcp_enabled.clone(),
        mcp_built_at,
        tools::McpFilter {
            loading: mcp_loading,
            tools: tools_allow.clone(),
            exclude: tools_exclude.clone(),
            no_tools,
            code_mode,
        },
    )));
    #[cfg(feature = "code-mode")]
    if code_mode {
        let nested = tools::code_mode::select_deferred_tools(mcp_tools, nested_exclude, deny_tool);
        tools::code_mode::restore_execute(
            &mut registry,
            nested,
            tools_exclude.as_deref(),
            no_tools,
        );
    }

    (bash_jobs, registry)
}

/// The store (none when not persisting), the session, and the writer lock held for the whole run.
type OpenedSession = (
    Option<SessionStore>,
    Session,
    Option<beyond_ai_agent::session_store::SessionLock>,
);

/// Opens, creates or goes without the session the flags select, and takes its writer lock.
fn open_run_session(
    fresh_meta: &dyn Fn() -> SessionMeta,
    model: &str,
    resumed_model: Option<String>,
    selected: RunSession,
) -> Result<OpenedSession, Box<dyn std::error::Error>> {
    // Materialize the selection made before anything model-keyed was resolved (`select_session`, in `resolve_session`):
    // a fresh session is created only now, once every check that can fail this invocation has passed.
    let (mut store, session) = match selected {
        RunSession::Existing(opened) => {
            let (store, session) = *opened;
            (Some(store), session)
        }
        RunSession::Fresh(FreshSession::AtPath(path)) => (
            Some(SessionStore::create(path, fresh_meta())?),
            Session::new(),
        ),
        RunSession::Fresh(FreshSession::InRepo(repo)) => {
            (Some(repo.create(fresh_meta())?), Session::new())
        }
        RunSession::Ephemeral => (None, Session::new()),
    };
    // One writer per session: held for the whole run, taken before the store's first write. A
    // session another process is writing (`run -c` in a second terminal, a `serve` that has it open)
    // is refused with the file named, not shared — two writers each chain their turns onto their own
    // stale tip. See `session_store::lock_session_writer`.
    let _session_lock = match &store {
        Some(store) => beyond_ai_agent::session_store::lock_session_writer(store.path())
            .map_err(|e| format!("cannot use this session: {e}"))?,
        None => None,
    };
    // (After the lock: this is a write.) An explicit `--model` that differs from the resumed session's own is a model change, recorded
    // the way `serve`'s `set_model` records one, so the next `--continue` stays on it instead of
    // reverting to the model the session was created with.
    if let (Some(store), Some(resumed)) = (&mut store, &resumed_model)
        && *resumed != model
    {
        store.record_model_change(model)?;
    }

    Ok((store, session, _session_lock))
}

/// Builds the durable and session memory mounts and registers the `memory` tool over them.
async fn mount_memory(
    fresh_meta: &dyn Fn() -> SessionMeta,
    memory_backend: Option<Arc<dyn beyond_ai_agent::memory::MemoryBackend>>,
    no_session_memory: bool,
    registry: &mut agent_core::ToolRegistry,
    store: &Option<SessionStore>,
) -> (
    Vec<beyond_ai_agent::memory::Mount>,
    Vec<(beyond_ai_agent::memory::MountKind, String)>,
    bool,
) {
    // Build the mount list: durable `/memories`, plus (unless `--no-session-memory`) a per-session
    // `/session` working-memory store — the compaction-surviving scratchpad. `run` is single-session, so
    // the session dir is fixed for the whole process: the `<...>.memory/` sibling of a persisted session
    // file (so `session_store` trashes/restores it together), or an id-keyed tempdir when ephemeral. The
    // `open_run_session` runs before this, so `store` is already resolved here. Shared with every
    // subagent below by `Arc`, so a child sees the very same `/session` as the parent.
    let mounts: Vec<beyond_ai_agent::memory::Mount> = {
        use beyond_ai_agent::memory::{Mount, MountKind};
        let mut v = Vec::new();
        if let Some(backend) = &memory_backend {
            v.push(Mount {
                kind: MountKind::Durable,
                backend: backend.clone(),
            });
            if !no_session_memory {
                let session_file = store.as_ref().map(|s| s.path().to_path_buf());
                let session_id = store
                    .as_ref()
                    .map(|s| s.meta().id.clone())
                    .unwrap_or_else(|| fresh_meta().id);
                let dir =
                    beyond_ai_agent::memory::session_dir(session_file.as_deref(), &session_id);
                v.push(Mount {
                    kind: MountKind::Session,
                    backend: Arc::new(beyond_ai_agent::memory::file::FileBackend::session_at(dir)),
                });
            }
        }
        v
    };
    let memory_sections: Vec<(beyond_ai_agent::memory::MountKind, String)> = if mounts.is_empty() {
        Vec::new()
    } else {
        let sections = beyond_ai_agent::memory::mount_sections(&mounts).await;
        registry.register(Arc::new(tools::memory::Memory::mounted(mounts.clone())));
        sections
    };
    // Gates the active compaction reminder pushed on each `Compacted` (see `run_turn_once`): only when a
    // `/session` working-memory mount is live.
    let session_memory_active = mounts
        .iter()
        .any(|m| m.kind == beyond_ai_agent::memory::MountKind::Session);

    (mounts, memory_sections, session_memory_active)
}

/// Reaps orphaned worktrees, discovers agent definitions and registers the `subagent` tool.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::ptr_arg)]
fn register_subagent(
    bash_command_prefix: Option<String>,
    bash_shell_path: Option<String>,
    code_mode: bool,
    cwd: &PathBuf,
    deny_bash_pattern: &Vec<String>,
    deny_path: &Vec<String>,
    deny_tool: &Vec<String>,
    exec_runner: &Option<Arc<dyn tools::exec::CommandRunner>>,
    fs_backend: Option<Arc<dyn tools::fs::FsBackend>>,
    gateway: &String,
    idle_timeout_ms: Option<u64>,
    image_auto_resize: bool,
    mcp_enabled: &tools::mcp::McpEnabledSet,
    mcp_live: tools::mcp::McpTools,
    mcp_loading: tools::tool_search::McpLoading,
    model: &String,
    mounts: Vec<beyond_ai_agent::memory::Mount>,
    no_context_files: bool,
    project_trusted: bool,
    prompt_guidelines: &Vec<String>,
    provider_env: beyond_ai_agent::gateway_credential::ProviderEnv,
    raw_gateway_key: Option<String>,
    registry: &mut agent_core::ToolRegistry,
    retry_base_delay_ms: Option<u64>,
    retry_max_backoff_ms: Option<u64>,
    retry_max_retries: Option<u32>,
    skills: &Vec<beyond_ai_agent::skills::Skill>,
    tools_exclude: Option<Vec<String>>,
    web_allow_host: Vec<String>,
    web_allow_private: bool,
    web_timeout_ms: Option<u64>,
) -> (
    Arc<agent_core::WriteLockRegistry>,
    Vec<beyond_ai_agent::agents::AgentDef>,
    beyond_ai_agent::tools::subagent::ParentCell,
) {
    // Subagents. A shared write-lock registry (rather than `Agent::new`'s private default) so the parent
    // and every child serialize same-path writes against each other. Registered only when at least one
    // agent definition exists — a `subagent` tool with no agents to call is dead weight in the prompt.
    let write_locks = Arc::new(agent_core::WriteLockRegistry::new());
    beyond_ai_agent::worktree::sweep(cwd); // reap any worktree orphaned by a previous crash
    let agent_defs = beyond_ai_agent::agents::discover(cwd, project_trusted);
    // What a child inherits of this run's loop configuration — written once the parent's agent is
    // built (see `ParentLoop`), read by each dispatch.
    let parent_loop = beyond_ai_agent::tools::subagent::ParentCell::default();
    if !agent_defs.is_empty() {
        use beyond_ai_agent::tools::subagent;
        // `parent_tools` is what a child with no `tools:` of its own inherits — the parent's effective
        // set minus `subagent` itself, so a restricted parent can't spawn a fully-armed child.
        let parent_tools: Vec<String> = registry
            .definitions()
            .into_iter()
            .map(|d| d.name)
            .filter(|n| n != subagent::NAME)
            .collect();
        // The factory re-resolves credentials per child model (they're model-keyed). Captures the raw
        // key and the parent's retry/timeout knobs, mirroring the parent client build below.
        let raw_key = raw_gateway_key.clone();
        let gateway_for_factory = gateway.clone();
        // Same direct/gateway resolution the parent uses — a subagent on a different model must reach it
        // the same way the parent would have, gateway or not.
        let provider_env_for_factory = provider_env.clone();
        let factory: subagent::TransportFactory = Arc::new(move |m: &str| {
            build_run_gateway_client(
                raw_key.clone(),
                &gateway_for_factory,
                m,
                &provider_env_for_factory,
                retry_max_retries,
                retry_base_delay_ms,
                retry_max_backoff_ms,
                idle_timeout_ms,
            )
            .map(|c| Arc::new(c) as Arc<dyn agent_core::ModelTransport>)
        });
        let ctx = Arc::new(subagent::SubagentCtx {
            factory,
            agents: Arc::new(agent_defs.clone()),
            skills: Arc::new(skills.clone()),
            write_locks: write_locks.clone(),
            mcp_tools: mcp_live.clone(),
            // `run` has no `set_mcp_enabled` command, so the gate never moves: every configured
            // server stays enabled for the life of the process, which is what the default means.
            // Shared, so a child sees (and is gated by) the skills this run loaded.
            mcp_enabled: mcp_enabled.clone(),
            memory_mounts: mounts.clone(),
            tool_cfg: subagent::ChildToolConfig {
                bash_shell_path: bash_shell_path.clone(),
                bash_command_prefix: bash_command_prefix.clone(),
                // `run` never probes a target for its shell (only `serve --service` does), so a
                // remote child falls back to the `sh` every POSIX target has.
                remote_shell: None,
                web_allow_private,
                web_allow_hosts: web_allow_host.clone(),
                web_timeout_ms,
                image_auto_resize,
                // A child acts on the same machine as its parent. Handing down `None` here was a
                // sandbox escape: the model could reach the host simply by delegating to a subagent.
                fs_backend: fs_backend.clone(),
                command_runner: exec_runner.clone(),
                code_mode,
                mcp_loading,
                exclude_tools: tools_exclude.clone().unwrap_or_default(),
            },
            cwd: cwd.clone(),
            // `run`'s cwd is this host's, so each fan-out walks it as it always has.
            context_files: None,
            include_context_files: !no_context_files,
            project_trusted,
            disk_overrides: true,
            prompt_guidelines: prompt_guidelines.clone(),
            // Filled in once the parent's own agent is configured, below.
            parent: parent_loop.clone(),
            parent_cache_key: model.clone(),
            parent_tools,
            deny_tool: deny_tool.clone(),
            deny_bash_pattern: deny_bash_pattern.clone(),
            deny_path: deny_path.clone(),
            max_depth: subagent::DEFAULT_MAX_DEPTH,
            // `run` has no client to ask, so it has no interactive gate — only the static `--deny-*`
            // lists, which `ChildHooks` still installs. See `crate::approval`'s module doc.
            approval: None,
        });
        registry.register(Arc::new(subagent::Subagent::new(ctx)));
    }

    (write_locks, agent_defs, parent_loop)
}

/// The system prompt: the base, `--append-system-prompt`, context files, skills, memory and agents.
#[allow(clippy::too_many_arguments)]
fn build_run_system(
    agent_defs: Vec<beyond_ai_agent::agents::AgentDef>,
    append_system_prompt: Vec<String>,
    cwd: &Path,
    mcp_skills: &tools::mcp_skills::McpSkills,
    memory_sections: Vec<(beyond_ai_agent::memory::MountKind, String)>,
    model: &str,
    no_context_files: bool,
    project_trusted: bool,
    prompt_guidelines: Vec<String>,
    registry: &agent_core::ToolRegistry,
    skills: &[beyond_ai_agent::skills::Skill],
    system_prompt: Option<String>,
    timing: &mut beyond_ai_agent::timing::StartupTiming,
) -> String {
    // `--system-prompt`/`--append-system-prompt` may each name an existing, readable file instead of
    // literal text (pi-parity fix — matches pi's own `resolvePromptInput`); resolved once, here, rather
    // than re-deriving it at each of the several places downstream that would otherwise need to repeat
    // the same file-vs-literal check. `--append-system-prompt` is repeatable (pi-parity fix: previously
    // a second occurrence silently clobbered the first instead of accumulating) — each occurrence is
    // resolved independently, then joined into one block.
    let system_prompt = system_prompt.as_deref().map(resolve_prompt_input);
    let append_system_prompt = {
        let resolved: Vec<String> = append_system_prompt
            .iter()
            .map(|s| resolve_prompt_input(s))
            .collect();
        (!resolved.is_empty()).then(|| resolved.join("\n\n"))
    };
    // `--system-prompt` replaces the built-in base entirely — matches `serve`'s identical flag. Threaded
    // through as `Some`/`None` (rather than pre-collapsed with the computed default here) so
    // `build_system_prompt` can tell "an explicit override was given" apart from "nothing given, use the
    // built-in default" — an explicit flag must win outright over a trusted project's on-disk
    // `SYSTEM.md`, which previously always won regardless (pi-parity fix).
    let default_base =
        beyond_ai_agent::resources::default_system_prompt(registry, &prompt_guidelines);
    // Skills are discovered by path, not inlined into the prompt — invoking one relies on the model
    // being able to open its `SKILL.md` itself, so advertising them at all when `read` isn't registered
    // (a restricted `--tools`/`--exclude-tools` invocation) just adds dead weight (pi-parity fix).
    let has_read = registry.get("read").is_some();
    let has_todo = registry.get("todo").is_some();
    let has_structured_output = registry.get(tools::structured_output::NAME).is_some();
    let has_memory = registry.get(tools::memory::NAME).is_some();
    let system = beyond_ai_agent::resources::build_system_prompt(
        &beyond_ai_agent::resources::PromptOptions {
            base: system_prompt.as_deref(),
            default_base: &default_base,
            append: append_system_prompt.as_deref(),
            cwd,
            include_context_files: !no_context_files,
            skills,
            context_window: agent_core::capabilities(model).context_window,
            has_read,
            has_todo,
            has_structured_output,
            has_memory,
            memory_sections: &memory_sections,
            project_trusted,
            agents: &agent_defs,
            disk_overrides: true,
        },
    );
    // MCP-served skills get their own origin-tagged `<available_skills origin="mcp">` block, loaded
    // through their server's `mcp__<server>__skill__read` tool rather than `read` (`serve` renders the
    // same block per turn, through its session gate).
    let system = match mcp_skills.format_available() {
        mcp if mcp.is_empty() => system,
        mcp => format!("{system}\n\n{mcp}"),
    };
    timing.mark("build system prompt");

    system
}

/// Applies `--name`, builds the session metadata and warns about a stale working directory.
#[allow(clippy::ptr_arg)]
fn open_run_meta(
    cwd: &PathBuf,
    fresh_meta: &dyn Fn() -> SessionMeta,
    model: &String,
    name: &Option<String>,
    session: &Session,
    mut store: &mut Option<SessionStore>,
    timing: &mut beyond_ai_agent::timing::StartupTiming,
) -> Result<SessionMeta, Box<dyn std::error::Error>> {
    // `--name`, applied uniformly across every session-open path (mirrors `serve`'s own startup check) —
    // only for a genuinely fresh session (no messages, no title yet). `--continue`'s
    // `resume_latest_or_create` branch in `select_session` mints its own fresh `SessionMeta` internally when no
    // cwd match exists, bypassing the `fresh_meta` closure other branches use, so this was
    // previously the one path `--name` silently never reached even when it *did* open a brand-new
    // session.
    if let Some(name) = &name
        && session.messages.is_empty()
        && let Some(store) = &mut store
        && store.meta().title.is_none()
    {
        store.set_title(name)?;
    }
    let mut meta = store
        .as_ref()
        .map(|s| s.meta().clone())
        .unwrap_or_else(fresh_meta);
    // The model this run is on (the resumed session's active model, or an explicit `--model`), not the
    // header's creation-time one: what the `session` header line, lifecycle and export report.
    meta.model = model.clone();
    // A genuinely fresh session's `cwd` always equals the current one (just stamped by `fresh_meta`),
    // so this only fires for a reopened `--session`/`--continue` session — the recorded directory was
    // moved/deleted, or this process simply isn't running where the session was created (e.g. a
    // `--session-dir` shared across projects). `serve` already surfaces the identical check as
    // `cwd_stale` on its RPC responses; `run` had no equivalent at all, matching pi's
    // `MissingSessionCwdError` guard. Informational, not fatal — the tools underneath will surface
    // their own, more specific errors if this actually matters for the task at hand.
    if serve::cwd_is_stale(&meta.cwd, cwd) {
        eprintln!(
            "warning: this session's recorded working directory ({}) does not match the current one \
             ({}); tools will operate against the current directory",
            meta.cwd,
            cwd.display()
        );
    }
    timing.mark("open session");
    timing.print();

    Ok(meta)
}

/// The gateway client with the run's retry, backoff and idle-timeout settings.
fn build_run_client(
    gateway: String,
    idle_timeout_ms: Option<u64>,
    key: GatewayCredential,
    model_extra_headers: std::collections::HashMap<String, String>,
    retry_base_delay_ms: Option<u64>,
    retry_max_backoff_ms: Option<u64>,
    retry_max_retries: Option<u32>,
) -> Result<GatewayClient, Box<dyn std::error::Error>> {
    let mut client = match key {
        GatewayCredential::Static(key) => GatewayClient::new(gateway, key)?,
        GatewayCredential::Oauth(source) => GatewayClient::with_credential_source(gateway, source)?,
    }
    .with_retry(
        retry_max_retries.unwrap_or(agent_core::client::MAX_RETRIES),
        retry_base_delay_ms
            .map(std::time::Duration::from_millis)
            .unwrap_or(agent_core::client::BASE_BACKOFF),
    )
    // Task #11 (pi-parity feature): a `models.json` override's `headers` (if any) merged onto every
    // outgoing request via the generic `with_extra_headers` mechanism — harmless (a no-op) when no
    // override configured any, since an empty map is also `GatewayClient::new`'s own default. Resolved
    // together with the credential above from a single `models.json` parse (T9-F3).
    .with_extra_headers(model_extra_headers);
    // Task #30 (pi-parity feature): `with_max_backoff` previously had no CLI flag or persisted override
    // reaching it at all, unlike its two siblings (`retry_max_retries`/`retry_base_delay_ms`) above —
    // see `agent_core::client::GatewayClient::with_max_backoff`'s own doc comment.
    if let Some(ms) = retry_max_backoff_ms {
        client = client.with_max_backoff(std::time::Duration::from_millis(ms));
    }
    // Task #19 (pi-parity feature): `with_idle_timeout` previously had zero callers anywhere in this
    // codebase. Consulted here for every routing path (proxied through the gateway, or a
    // direct-routed/custom `models.json` `base_url` override that bypasses the gateway's own ~600s
    // built-in assumption entirely — see `RouteOverride::Direct`) since a self-hosted/alternate-provider
    // endpoint's own slow/fast tail has no reason to match the gateway's.
    if let Some(ms) = idle_timeout_ms {
        client = client.with_idle_timeout(std::time::Duration::from_millis(ms))?;
    }
    Ok(client)
}

/// Shares the session store with the journal and recovers background jobs a previous run left behind.
async fn attach_run_journal(
    bash_jobs: &Arc<tools::bash_jobs::BashJobs>,
    exec_runner: Option<Arc<dyn tools::exec::CommandRunner>>,
    meta: &SessionMeta,
    store: Option<SessionStore>,
) -> (RunStore, Option<&'static RunJournal>) {
    // Shared with `DirectCheckpoint` below (built before `agent`, so the hook can be installed at
    // construction) so a long multi-step turn (many tool round-trips) is persisted incrementally —
    // the same guarantee `serve` gives every session. Without this, only the *end* of each whole
    // turn was ever persisted (the `persist_run` calls below, after `run_turn` returns), so a
    // crash mid-turn — after several tool round-trips already ran real commands/edited real files —
    // lost all record of them with no session trace at all.
    let store: RunStore = Arc::new(std::sync::Mutex::new(store));
    let run_journal = store
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .is_some()
        .then(|| {
            RUN_JOURNAL.get_or_init(|| RunJournal {
                store: store.clone(),
                session_id: meta.id.clone(),
            })
        });
    // Remote jobs run in the sandbox under this session's scope. A run stops its jobs when it ends,
    // so whatever an earlier run of this session left there (it was killed before it could stop
    // them) is stopped now — the reopen note says so.
    bash_jobs.set_scope(Some(
        beyond_ai_agent::exec_endpoint::jobs::JobScope::for_session(None, &meta.id),
    ));
    if let Some(runner) = &exec_runner {
        tools::bash_sandbox::reattach(bash_jobs, runner.clone()).await;
    }
    // Local jobs a killed earlier run left running (a SIGKILL runs no exit path to stop them) are
    // found by their journaled tags and stopped now; the reopen note says which. And this run's own
    // jobs are journaled as they spawn, for whoever opens the session after it.
    if let Some(journal) = run_journal {
        let records = journal
            .store
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map(|s| s.active_journal(tools::bash_jobs::PROCS_ENTRY_KIND))
            .unwrap_or_default();
        if !records.is_empty() {
            let jobs = bash_jobs.clone();
            let _ = tokio::task::spawn_blocking(move || jobs.stop_survivors(&records)).await;
        }
        bash_jobs.set_recorder(Some(std::sync::Arc::new(|data| {
            if let Some(journal) = RUN_JOURNAL.get() {
                journal.append(tools::bash_jobs::PROCS_ENTRY_KIND, data);
            }
        })));
    }
    (store, run_journal)
}

/// Stops the remaining background jobs and prints the end-of-run summary lines.
async fn report_run_summary(
    bash_jobs: Arc<tools::bash_jobs::BashJobs>,
    broken_pipe: &AtomicBool,
    cancel: &agent_core::CancellationToken,
    json: bool,
    max_steps: Option<u32>,
    session: &Session,
    stopped_jobs: Vec<tools::bash_jobs::StoppedJob>,
) -> () {
    // The run is over, so are its jobs: the yielded ones stopped at the end of their turn (above),
    // the intended ones stop now. The model was told once, at its stop, which were still running
    // and that ending stops them. Say which were stopped — a `--json` consumer gets one
    // `bash_jobs_stopped` line (before any structured output, which stays last), each job with its
    // `kind`.
    let mut stopped = stopped_jobs;
    stopped.extend(stop_jobs(&bash_jobs, tools::bash_jobs::BashJobs::stop_all).await);
    // The reports of jobs that ended are persisted with the run: their sandbox directories can go.
    bash_jobs.collect_garbage().await;
    if !stopped.is_empty() {
        if json {
            let line =
                serde_json::json!({ "kind": "bash_jobs_stopped", "jobs": stopped }).to_string();
            write_stdout_or_exit(&line, cancel, broken_pipe);
            write_stdout_or_exit("\n", cancel, broken_pipe);
        }
        let list: Vec<String> = stopped
            .iter()
            .map(|j| match &j.survived {
                None => format!("job {} `{}` ({} s)", j.job, j.command, j.running_secs),
                Some(left) => format!(
                    "job {} `{}` ({} s; {left} could not be killed and is still running)",
                    j.job, j.command, j.running_secs
                ),
            })
            .collect();
        eprintln!(
            "[stopped {} background job(s) still running at the end of the run: {}]",
            stopped.len(),
            list.join(", ")
        );
    }

    // A run that hit its `--max-steps` ceiling still ends normally (exit 0) on the model's own
    // wrap-up answer — that answer is the point of the wrap-up grant, and a run that couldn't stop
    // within it already fails with `MaxSteps`. But the task may well be unfinished, which text mode
    // otherwise never says (`--json` streams the `wrap_up` event).
    if RUN_WRAPPED_UP.load(Ordering::Relaxed)
        && let Some(max) = max_steps
    {
        eprintln!(
            "[stopped at the --max-steps ceiling ({max}): the reply above is the model's wrap-up, \
             so the task may be unfinished; --continue (or a higher --max-steps) keeps going]"
        );
    }
    eprintln!(
        "[done in {} step(s); {} in / {} out tokens]",
        session.steps, session.input_tokens, session.output_tokens
    );
}

/// Emits the run's structured output (or its absence) and the lifecycle terminal event.
#[allow(clippy::too_many_arguments)]
async fn report_run_outcome(
    broken_pipe: AtomicBool,
    cancel: agent_core::CancellationToken,
    json: bool,
    life: Option<Arc<beyond_ai_agent::lifecycle::Run>>,
    output_slot: tools::structured_output::OutputSlot,
    session: Session,
    stop_reason: agent_core::StopReason,
    wants_structured_output: bool,
) -> () {
    // `--output-schema` made this run a callable function, so the payload is its real return value: the
    // last thing on stdout, and the difference between exit 0 and exit 1. Read only now, once the run
    // has fully drained — a mixed batch (`structured_output` alongside an `edit` in one turn, which the
    // loop's unanimous-terminate rule lets continue) stages the value and may revise it later.
    //
    // Emitted on *one* line in both modes, so `... | tail -1 | jq` is the whole contract. Text mode
    // already echoed the call as `[tool: structured_output] {...}` above and prints live assistant text,
    // so its stdout was never a bare JSON document to begin with; pretty-printing the payload here would
    // only cost the caller the one property that makes it usable. In `--json` mode the payload is
    // wrapped as a final `kind` object so the stream stays strictly one-event-per-line. Nothing is
    // emitted at all without `--output-schema`, so existing `--json` consumers see no new line.
    if wants_structured_output {
        match output_slot.take() {
            Some(value) => {
                let line = if json {
                    serde_json::json!({ "kind": "structured_output", "value": value }).to_string()
                } else {
                    value.to_string()
                };
                write_stdout_or_exit(&line, &cancel, &broken_pipe);
                write_stdout_or_exit("\n", &cancel, &broken_pipe);
                emit_cli_lifecycle(
                    life.as_ref(),
                    &session,
                    if !json && stop_reason == agent_core::StopReason::Refusal {
                        CliTerminal::Failed {
                            error: "refused".into(),
                            refused: true,
                        }
                    } else {
                        CliTerminal::Succeeded {
                            refused: stop_reason == agent_core::StopReason::Refusal,
                        }
                    },
                    Some(value),
                )
                .await;
            }
            None => {
                emit_cli_lifecycle(
                    life.as_ref(),
                    &session,
                    CliTerminal::Failed {
                        error: format!(
                            "no structured output: the model ended the run without calling `{}`",
                            tools::structured_output::NAME
                        ),
                        refused: false,
                    },
                    None,
                )
                .await;
                eprintln!(
                    "[no structured output: the model ended the run without calling `{}`]",
                    tools::structured_output::NAME
                );
                exit_process(1);
            }
        }
    } else {
        emit_cli_lifecycle(
            life.as_ref(),
            &session,
            if text_mode_failure_message(json, stop_reason).is_some() {
                if stop_reason == agent_core::StopReason::Refusal {
                    CliTerminal::Failed {
                        error: "refused".into(),
                        refused: true,
                    }
                } else {
                    CliTerminal::Aborted
                }
            } else {
                CliTerminal::Succeeded {
                    refused: stop_reason == agent_core::StopReason::Refusal,
                }
            },
            None,
        )
        .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_core::AgentHooks;

    #[tokio::test]
    async fn a_tilde_target_is_checked_against_the_remote_targets_home() {
        let policy = run_policy(
            &[],
            &[],
            &["/home/agent/.ssh/*".to_string()],
            tools::fs::PathWorld::Remote {
                home: Some("/home/agent".into()),
            },
        );
        let denied = policy
            .before_tool_call(
                "write",
                &serde_json::json!({"path": "~/.ssh/id_rsa", "content": "x"}),
                &agent_core::Session::new(),
                &tokio_util::sync::CancellationToken::new(),
            )
            .await;
        assert!(denied.is_some(), "the remote home must be what `~` means");
    }
}
