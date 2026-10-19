//! `serve`: flag validation, setting resolution, `ServeConfig` assembly and transport choice.
//!
//! Four phases run in order, each failing before anything is bound or opened: [`validate`] (flag
//! combinations, service-mode refusals), [`resolve`] (flag, then stored setting, then built-in
//! default), [`build_config`] (the `ServeConfig` literal), [`run_transport`] (stdio or the
//! WebSocket listeners).

use std::sync::Arc;

use beyond_ai_agent::mode::Mode;
use beyond_ai_agent::policy::ToolPolicy;
use beyond_ai_agent::session_store::canonical_cwd;
use beyond_ai_agent::{serve, serve_ws, tools};

use crate::{
    CacheTtl, Command, DEFAULT_GATEWAY, DEFAULT_MODEL, exit_process, is_valid_session_id,
    mcp_manifest_dir, parse_reasoning_effort, resolve_prompt_input,
};

/// What `validate` hands to `build_config`: the process mode and the session-grant trust it checked.
struct ServeTrust {
    mode: Mode,
    grant_verifier: Option<Arc<beyond_ai_agent::grant::GrantVerifier>>,
    shards: Arc<beyond_ai_agent::service::Shards>,
}

/// What `validate` hands to `run_transport`: the listeners and scrape endpoint it parsed.
struct ServeTransport {
    tls: Option<serve_ws::TlsListen>,
    systemd_activated: bool,
    metrics_addr: Option<std::net::SocketAddr>,
    metrics: Option<Arc<beyond_ai_agent::metrics::Metrics>>,
}

/// Every setting `resolve` settled (flag, then stored default, then built-in default), for
/// `build_config` to put into the `ServeConfig`.
struct ServeResolved {
    system: String,
    append_system_prompt: Option<String>,
    stored_settings: beyond_ai_agent::settings::Settings,
    bash_shell_path: Option<String>,
    bash_command_prefix: Option<String>,
    compaction_reserve_tokens: Option<u32>,
    compaction_keep_recent_tokens: Option<u32>,
    branch_summary_reserve_tokens: Option<u32>,
    retry_max_retries: Option<u32>,
    retry_base_delay_ms: Option<u64>,
    retry_max_backoff_ms: Option<u64>,
    idle_timeout_ms: Option<u64>,
    block_images: bool,
    image_auto_resize: bool,
    models: Option<Vec<String>>,
    extra_skill_paths: Vec<String>,
    extra_prompt_template_paths: Vec<String>,
    model_explicit: bool,
    reasoning_effort_explicit: bool,
    resolved_model: String,
    reasoning_effort: Option<agent_core::ReasoningEffort>,
    steering_mode: agent_core::QueueMode,
    follow_up_mode: agent_core::QueueMode,
    resolved_session_dir: Option<String>,
    mcp_tools: tools::mcp::McpTools,
    mcp_catalog: tools::mcp::McpCatalog,
    mcp_apps: Option<tools::mcp_apps::McpAppsPool>,
    mcp_http: Option<tools::mcp::McpEgress>,
    listen_uds_mode: Option<u32>,
    configured_gateway: Option<String>,
}

/// `serve`: validate, resolve, assemble, run, then exit the process with the shutdown cause.
pub(crate) async fn serve(cmd: Command) -> Result<(), Box<dyn std::error::Error>> {
    let (trust, transport) = validate(cmd.clone())?;
    let resolved = resolve(cmd.clone(), &trust.mode).await?;
    let serve_cfg = build_config(cmd, trust, &transport, resolved)?;
    let shutdown_cause = run_transport(serve_cfg, transport).await?;

    // `serve` reads stdin via `tokio::io::stdin()`, which parks a dedicated blocking OS
    // thread doing a blocking read for the life of the process. If stdin is never closed
    // (a client that doesn't hang up, or — the case this matters for — a SIGTERM/SIGINT
    // whose handler cancels the run and returns without stdin ever reaching EOF), that
    // thread is still parked here even though all async work is done. Falling through to
    // `Runtime` drop would then hang indefinitely: dropping a `Runtime` waits for every
    // outstanding blocking task, and a parked stdin read never completes on its own. Exit
    // explicitly instead — `serve` has already drained, persisted, and flushed everything
    // before returning, so there's nothing left to lose.
    // Task #41 (pi-parity fix): `shutdown_cause` distinguishes a real signal-triggered
    // shutdown from a clean stdin-EOF one — previously every graceful path exited 0
    // unconditionally, matching neither pi's own `rpc-mode.ts` (143/129 for SIGTERM/SIGHUP)
    // nor a shell's own convention for reporting which signal actually stopped a process.
    //
    // A signal-triggered shutdown gets here by cancelling every live run, which drops any
    // in-flight bash tool future and with it its `GroupKillGuard` — whose reaping happens on a
    // detached thread that is *not* among the session threads `serve`/`serve_ws` joined on the
    // way out, and that `process::exit` would otherwise terminate mid-`kill`, orphaning exactly
    // the backgrounded grandchildren the guard exists to reap. Bounded, so a wedged `kill`/`ps`
    // shell-out can't hold the daemon's own shutdown open.
    //
    // Stdio MCP servers are in the same position — see `exit_process`.
    exit_process(shutdown_cause.map(serve::Signal::exit_code).unwrap_or(0));
}

/// Flag combinations and service-mode refusals, checked before anything is bound or opened.
fn validate(cmd: Command) -> Result<(ServeTrust, ServeTransport), Box<dyn std::error::Error>> {
    let Command::Serve {
        key,
        session_file,
        session_dir,
        listen,
        tls_listen,
        tls_cert,
        tls_key,
        listen_uds,
        grant_key,
        seal_key,
        service,
        shard,
        metrics_listen,
        notify_origin,
        mcp_events_callback_url,
        session_id,
        r#continue: continue_session,
        no_session_persistence,
        memory,
        trust_project,
        force_untrusted,
        bash_shell_path,
        web_allow_private,
        web_allow_host,
        exec_url,
        exec_header,
        exec_cmd,
        tools,
        exclude_tools,
        no_tools,
        code_mode,
        deny_tool,
        deny_path,
        extra_skill_paths,
        extra_prompt_template_paths,
        name,
        ..
    } = cmd
    else {
        unreachable!("dispatched on Command::Serve")
    };
    // Fail fast, before starting the server — see `run_task`'s identical check for why this
    // rejects rather than silently clearing (pi's own `--name` behavior).
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
        tools.as_deref(),
        exclude_tools.as_deref(),
        &deny_tool,
        no_tools,
    )?;
    beyond_ai_agent::skills::check_paths("--skill", &extra_skill_paths)?;
    beyond_ai_agent::skills::check_paths("--prompt-template", &extra_prompt_template_paths)?;
    // Same filesystem-path-injection concern as `run`'s identical check (`--session-id` becomes
    // part of a persisted session's filename) — see `is_valid_session_id`'s doc comment.
    if let Some(id) = &session_id
        && !is_valid_session_id(id)
    {
        return Err(format!(
                    "--session-id {id:?} is invalid: must contain only letters, digits, '.', '_', \
                     '-', and start/end with a letter or digit — it becomes part of a filesystem path"
                )
                .into());
    }
    // Session-grant trust: a bad key or seal-key file fails the process here, not on the first
    // connection. Neither flag ⇒ `None`, and `serve` behaves exactly as before.
    //
    // The inline seal key is read straight from the environment, not declared as a flag: a
    // flag would put the secret in argv, readable by every process on the host.
    let seal_b64 = match std::env::var_os(beyond_ai_agent::grant::SEAL_KEY_B64_ENV) {
        None => None,
        Some(v) => match v.into_string() {
            Ok(v) if v.is_empty() => None,
            Ok(v) => Some(zeroize::Zeroizing::new(v)),
            Err(_) => {
                return Err(format!(
                    "{}: not valid UTF-8",
                    beyond_ai_agent::grant::SEAL_KEY_B64_ENV
                )
                .into());
            }
        },
    };
    let seal_source = beyond_ai_agent::grant::SealKeySource::pick(
        seal_key.as_deref(),
        seal_b64.as_deref().map(String::as_str),
    )?;
    let grant_verifier =
        beyond_ai_agent::grant::GrantVerifier::from_flags(&grant_key, seal_source)?.map(Arc::new);
    drop(seal_b64);
    // Service mode: validated here, before anything is bound or opened, because every one
    // of these is a way for the replica to reach a tenant (or the reverse). A refusal names
    // the flag, so an operator fixes the deployment rather than discovering at runtime that
    // a flag was silently ignored.
    let shards = Arc::new(beyond_ai_agent::service::Shards::parse(&shard)?);
    // Parsed before anything is bound or opened, like the shard list: a refusal names the
    // flag so an operator fixes the deployment rather than discovering at runtime that the
    // scrape never came up.
    let metrics_addr = match metrics_listen.as_deref() {
        Some(s) => Some(beyond_ai_agent::metrics::parse_listen_addr(s)?),
        None => None,
    };
    let metrics = match metrics_addr {
        Some(_) => Some(beyond_ai_agent::metrics::Metrics::new()?),
        None => None,
    };
    // All three or none, before anything is bound; the files themselves are checked by
    // `serve_ws` ahead of its first bind.
    let tls = match (tls_listen, tls_cert, tls_key) {
        (None, None, None) => None,
        (Some(addr), Some(cert), Some(key)) => Some(serve_ws::TlsListen { addr, cert, key }),
        _ => {
            return Err("--tls-listen, --tls-cert and --tls-key must be given together".into());
        }
    };
    let systemd_activated = cfg!(unix)
        && listen.is_none()
        && tls.is_none()
        && listen_uds.is_none()
        && std::env::var_os("LISTEN_FDS").is_some();
    // Webhook callbacks arrive on the daemon's HTTP listener; over stdio nothing would
    // receive them, and a server would fail its verification challenge on every subscribe.
    if mcp_events_callback_url.is_some()
        && listen.is_none()
        && tls.is_none()
        && listen_uds.is_none()
        && !systemd_activated
    {
        return Err(
            "--mcp-events-callback-url requires --listen, --listen-uds, or systemd socket \
                 activation: webhook deliveries arrive on serve's HTTP listener"
                .into(),
        );
    }
    // Decided once, here; every later "does service mode change this?" asks it.
    let mode = Mode::process(service);
    if let Some(origin) = &notify_origin {
        if !mode.offers_job_notify() {
            return Err(
                "--notify-origin needs --service: job notify is a service-mode \
                            feature"
                    .into(),
            );
        }
        beyond_ai_agent::tools::job_notify::check_origin(origin)?;
    }
    if mode.refuses_host_resources() {
        if grant_verifier.is_none() {
            return Err(
                    "--service requires --grant-key and --seal-key (or AI_AGENT_SEAL_KEY_B64): every connection must \
                     present a verified session grant"
                        .into(),
                );
        }
        if shards.is_empty() {
            return Err(
                "--service requires at least one --shard <name>=</absolute/path>: tenant \
                     data has nowhere to live"
                    .into(),
            );
        }
        // The session epoch fence trusts the NFS client's caches to expire within one
        // lease; a mount that keeps them longer (`nocto`, a large `acdirmax`/`actimeo`)
        // could lose acknowledged writes, so it is refused before anything is served.
        beyond_ai_agent::shard_mount::check_shards(&shards)?;
        if listen.is_none() && tls.is_none() && listen_uds.is_none() && !systemd_activated {
            return Err(
                "--service requires --listen, --listen-uds, or systemd socket activation: \
                     stdio has no place to carry a per-connection session grant"
                    .into(),
            );
        }
        // Every flag here is either a *host* resource a tenant must not reach, or a
        // per-process setting a grant now supplies per session. Refused rather than
        // ignored: silently dropping `--key` would leave an operator believing the
        // replica's own credential was in use.
        //
        // Worktree isolation isn't in this table because it isn't a flag: a subagent asks
        // for it in its own `.claude/agents/*.md` (`isolation: worktree`). Service mode
        // discovers those definitions in the sandbox and runs the worktree's `git` there
        // (`worktree::Git::Remote`), so there is nothing to refuse.
        let refused: [(&str, bool); 19] = [
            ("--key", key.is_some()),
            ("--session-file", session_file.is_some()),
            ("--session-dir", session_dir.is_some()),
            ("--session-id", session_id.is_some()),
            ("--continue", continue_session),
            ("--name", name.is_some()),
            ("--no-session-persistence", no_session_persistence),
            ("--memory", memory.is_some()),
            ("--trust-project", trust_project),
            ("--force-untrusted", force_untrusted),
            ("--exec-url", exec_url.is_some()),
            ("--exec-cmd", exec_cmd.is_some()),
            ("--exec-header", !exec_header.is_empty()),
            ("--web-allow-private", web_allow_private),
            ("--web-allow-host", !web_allow_host.is_empty()),
            ("--code-mode", code_mode),
            ("--skill", !extra_skill_paths.is_empty()),
            ("--prompt-template", !extra_prompt_template_paths.is_empty()),
            ("--bash-shell-path", bash_shell_path.is_some()),
        ];
        if let Some((flag, _)) = refused.iter().find(|(_, given)| *given) {
            return Err(format!(
                "{flag} is refused with --service: in service mode the session grant \
                     supplies this, and the replica's own must never reach a tenant"
            )
            .into());
        }
    }
    Ok((
        ServeTrust {
            mode,
            grant_verifier,
            shards,
        },
        ServeTransport {
            tls,
            systemd_activated,
            metrics_addr,
            metrics,
        },
    ))
}

/// Flag, then stored setting, then built-in default, for every setting `serve` takes; connects the operator's MCP servers.
async fn resolve(cmd: Command, mode: &Mode) -> Result<ServeResolved, Box<dyn std::error::Error>> {
    let Command::Serve {
        model,
        gateway_url,
        key,
        session_file,
        session_dir,
        listen_uds_mode,
        mcp_allow_private,
        no_session_persistence,
        system_prompt,
        append_system_prompt,
        reasoning_effort,
        steering_mode,
        follow_up_mode,
        compaction_reserve_tokens,
        compaction_keep_recent_tokens,
        branch_summary_reserve_tokens,
        retry_max_retries,
        retry_base_delay_ms,
        retry_max_backoff_ms,
        idle_timeout_ms,
        block_images,
        no_block_images,
        no_image_auto_resize,
        bash_shell_path,
        bash_command_prefix,
        tools,
        exclude_tools,
        no_tools,
        models,
        extra_skill_paths,
        extra_prompt_template_paths,
        prompt_guidelines,
        ..
    } = cmd
    else {
        unreachable!("dispatched on Command::Serve")
    };
    // `--system-prompt`/`--append-system-prompt` may each name an existing, readable file
    // instead of literal text (pi-parity fix — matches pi's own `resolvePromptInput`). `run`'s
    // identical resolution (`main.rs::resolve_prompt_input`).
    let system = system_prompt
        .as_deref()
        .map(resolve_prompt_input)
        .unwrap_or_else(|| {
            // Shell-path override doesn't affect this registry's use (listing tool
            // names/descriptions for the default system prompt) — `describe()` doesn't mention it.
            let mut reg = tools::default_registry_with_config(&tools::ToolConfig::new());
            tools::apply_filter(
                &mut reg,
                tools.as_deref(),
                exclude_tools.as_deref(),
                no_tools,
            );
            beyond_ai_agent::resources::default_system_prompt(&reg, &prompt_guidelines)
        });
    // `--append-system-prompt` is repeatable (pi-parity fix: previously a second occurrence
    // silently clobbered the first instead of accumulating) — each occurrence is resolved
    // independently, then joined into one block. `run`'s identical handling.
    let append_system_prompt = {
        let resolved: Vec<String> = append_system_prompt
            .iter()
            .map(|s| resolve_prompt_input(s))
            .collect();
        (!resolved.is_empty()).then(|| resolved.join("\n\n"))
    };
    // A stored `agent settings` default sits between an explicit flag/env var and this crate's
    // own built-in default — same convention `run_task` applies (see its identical comment).
    // Feature 2 (Round 3, pi-parity): merges a trusted project's own
    // `<cwd>/.beyond/settings.json` (and `.claude/settings.json`) tier on top of the global one first — see
    // `settings::effective_settings_for_cwd`'s own doc comment for the trust-gating rationale.
    //
    // **Not in service mode.** These are the replica operator's own settings — its
    // `~/.beyond/settings.json` and whatever project happens to be its cwd — and every one
    // of the defaults below (a shell path, skill and template paths, the memory backend, a
    // session directory, a model, a gateway URL) would otherwise configure someone else's
    // tenant. An empty `Settings` makes every `or_else` fall through to this crate's own
    // built-in default, exactly as it does on a machine with no settings file.
    let cwd = canonical_cwd(&std::env::current_dir().unwrap_or_default());
    let stored_settings = if mode.reads_operator_settings() {
        beyond_ai_agent::settings::effective_settings_for_cwd(&cwd)
    } else {
        beyond_ai_agent::settings::Settings::default()
    };
    // Round 3 (pi-parity fix): same "explicit flag/env, then stored setting, then built-in
    // default" precedence as every other `stored_settings`-backed fallback here — see
    // `run_task`'s identical block, just below in this file, for the full set.
    let bash_shell_path =
        bash_shell_path.or_else(|| stored_settings.default_bash_shell_path.clone());
    let bash_command_prefix =
        bash_command_prefix.or_else(|| stored_settings.default_bash_command_prefix.clone());
    let compaction_reserve_tokens =
        compaction_reserve_tokens.or(stored_settings.default_compaction_reserve_tokens);
    let compaction_keep_recent_tokens =
        compaction_keep_recent_tokens.or(stored_settings.default_compaction_keep_recent_tokens);
    // Task #31 (pi-parity fix): `run_task`'s identical resolution, just below in this file.
    let branch_summary_reserve_tokens =
        branch_summary_reserve_tokens.or(stored_settings.default_branch_summary_reserve_tokens);
    let retry_max_retries = retry_max_retries.or(stored_settings.default_retry_max_retries);
    let retry_base_delay_ms = retry_base_delay_ms.or(stored_settings.default_retry_base_delay_ms);
    // Task #30 (pi-parity fix): `run_task`'s identical resolution, just below in this file.
    let retry_max_backoff_ms =
        retry_max_backoff_ms.or(stored_settings.default_retry_max_backoff_ms);
    // Task #38 (pi-parity fix): `serve` previously had no `--idle-timeout-ms` flag/persisted
    // fallback at all, unlike `run` — see `run_task`'s identical resolution, just below in this
    // file.
    let idle_timeout_ms = idle_timeout_ms.or(stored_settings.default_provider_timeout_ms);
    // Task #34 (pi-parity fix): `run_task`'s identical "explicit flag, then stored setting,
    // then built-in default" precedence for both flags — see its own doc comments — previously
    // had no `serve` counterpart at all: `build_tools`/`build_agent` never consulted either.
    // Pass 20 (pi-parity fix): `--no-block-images` wins outright over both an explicit
    // `--block-images` and the persisted default, the same "escape hatch" `--no-image-auto-resize`
    // already gave the opposite-defaulted `image_auto_resize` just below — previously
    // `block_images` had no way at all to be forced off for one invocation once a persisted
    // `agent settings --block-images` default was `true`.
    let block_images =
        !no_block_images && (block_images || stored_settings.block_images.unwrap_or(false));
    let image_auto_resize =
        !no_image_auto_resize && stored_settings.image_auto_resize.unwrap_or(true);
    let models = models.or_else(|| stored_settings.default_models_list.clone());
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
    // Validated once the whole fallback chain (explicit flag, then stored setting) has resolved
    // — moved below the resolution above so a value that only came from a stored default is
    // checked exactly like an explicit flag would be, not skipped.
    if let Some(path) = &bash_shell_path
        && !std::path::Path::new(path).exists()
    {
        return Err(format!("--bash-shell-path not found: {path}").into());
    }
    // Task #5 (pi-parity fix): whether the operator explicitly passed `--model`/
    // `--reasoning-effort` for *this* invocation, as opposed to falling back to a stored
    // `agent settings` default or this crate's own built-in default — the distinction `serve`'s
    // own startup anti-bleed check needs in order to prefer a reattached session's own
    // last-recorded model/level instead (see `ServeConfig::model_explicit`'s doc comment).
    // Captured before either variable is shadowed by its own fallback resolution below. Same
    // convention as `run_task`'s identical `model_explicit`, just below in this file.
    let model_explicit = model.is_some();
    let mut reasoning_effort_explicit = reasoning_effort.is_some();
    let resolved_model = model
        .or_else(|| stored_settings.default_model.clone())
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());
    // Fix 10 (pi-parity feature): `run`'s identical resolution — see that call site's doc
    // comment for why this must happen before `resolve_gateway_credential` below.
    let (resolved_model, model_thinking_level) =
        serve::resolve_model_id(&resolved_model, serve::available_models())
            .map_err(|e| format!("--model {resolved_model:?}: {e}"))?;
    // Unlike `run_task` below, `serve`'s `key` is NOT resolved into a `GatewayCredential` here:
    // `resolve_gateway_credential` is keyed on the model, and this process's *actual* starting
    // model isn't final yet — a reattached `--session`/`--continue` session may still override
    // `resolved_model` with its own last-recorded one (`ServeConfig::model_explicit`'s anti-bleed
    // check, applied inside `serve()`), and `set_model`/`cycle_model` can change it again at any
    // point after that. `serve()` itself calls `resolve_gateway_credential` — fresh, keyed off
    // whichever model is actually active — at startup and again on every runtime model switch
    // (see `serve::build_gateway_client`), so only the raw `--key`/`AI_AGENT_KEY` value is passed
    // through here, unresolved.
    // Fix 2 (pi-parity gap): a `:<level>` suffix on `--model` (e.g. `--model sonnet:high`) sets
    // the reasoning effort for this invocation exactly as if `--reasoning-effort` had been
    // passed directly — but only when the operator didn't already pass that flag explicitly,
    // which always wins outright. Counts as explicit from here on: the anti-bleed check above
    // this block's own doc comment mentions must prefer this operator-requested depth over a
    // reattached session's last-recorded one, same as an explicit `--reasoning-effort` would.
    let mut reasoning_effort = reasoning_effort;
    if !reasoning_effort_explicit && let Some(level) = model_thinking_level {
        reasoning_effort = Some(level);
        reasoning_effort_explicit = true;
    }
    // Fix 2 (pi-parity gap): `run`'s identical stored-default fallback for `--reasoning-effort`
    // — see that call site's doc comment. Converted from the portable `ThinkingLevel` (off
    // included) down to the wire-level `ReasoningEffort` (`None` for `Off`) only at the very
    // end, once every candidate source (flag, model suffix, stored setting) has had its turn —
    // Task 2 (pi-parity fix, pass 19): `off` is now a legal value at each of those layers, not
    // just the `--model <pattern>:off` suffix.
    let reasoning_effort = reasoning_effort
        .or_else(|| {
            stored_settings
                .default_reasoning_effort
                .as_deref()
                .and_then(|s| parse_reasoning_effort(s).ok())
        })
        .and_then(|level| level.reasoning_effort());
    // Task 1 (pi-parity fix, pass 19): same "explicit flag/env, then stored setting, then
    // built-in default" precedence as every other `stored_settings`-backed fallback here —
    // matches `run_task`'s identical resolution just below in this file (which does construct a
    // `Steering` with it, since `run` — unlike `serve` — has no live RPC to (re)apply it later).
    // `ServeConfig::steering_mode`/`follow_up_mode` carry these through to `serve()`, which
    // applies them via `steering.set_steering_mode`/`set_follow_up_mode` right after
    // `Steering::new()` — the `set_steering_mode`/`set_follow_up_mode` RPC commands can still
    // change them at runtime afterward, same as before.
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
    let resolved_session_dir = session_dir.or_else(|| {
        // Only synthesize a stored default when *neither* explicit flag was given —
        // `Persistence::open` checks `session_dir` before `session_file`, so filling in a
        // stored session-dir default even when the operator explicitly chose `--session-file`
        // would silently switch them into repo mode instead of the file mode they asked for.
        // Nor under `--no-session-persistence`: a stored default is not an explicit flag,
        // and the opt-out beats it, exactly as it does in `run`.
        if session_file.is_none() && !no_session_persistence {
            stored_settings.default_session_dir.clone()
        } else {
            None
        }
    });
    let (mcp_tools, mcp_catalog, mcp_apps) = connect_operator_mcp(mode, &stored_settings).await;
    // The one client every *session's own* connectors dial through, in service mode. Built
    // here, once: a `reqwest::Client` is a connection pool, and a replica holding many
    // thousands of sessions cannot afford one each. Deliberately its own client rather than
    // the gateway's shared h2c pool — that one is pinned to one known hop, and a tenant's
    // connector is an arbitrary third-party endpoint. Its egress policy is fixed at build
    // time, which is why it is a process flag rather than anything a grant can ask for.
    let mcp_http = mode.session_mcp_egress(mcp_allow_private)?;
    // Parse the UDS socket mode from its octal string (`0o660`, `660`, `0660` all work) up
    // front so a bad value fails fast with a clear message rather than deep in the listener.
    let listen_uds_mode: Option<u32> = match &listen_uds_mode {
        Some(s) => {
            let trimmed = s.trim().trim_start_matches("0o");
            Some(u32::from_str_radix(trimmed, 8).map_err(|e| {
                format!("invalid --listen-uds-mode {s:?} (expected octal, e.g. 0o660): {e}")
            })?)
        }
        None => None,
    };
    // See `run`'s identical block below for why `DEFAULT_GATEWAY` must not count as "configured".
    let configured_gateway = gateway_url
        .or_else(|| stored_settings.default_gateway_url.clone())
        .or_else(|| key.is_some().then(|| DEFAULT_GATEWAY.to_string()));
    Ok(ServeResolved {
        system,
        append_system_prompt,
        stored_settings,
        bash_shell_path,
        bash_command_prefix,
        compaction_reserve_tokens,
        compaction_keep_recent_tokens,
        branch_summary_reserve_tokens,
        retry_max_retries,
        retry_base_delay_ms,
        retry_max_backoff_ms,
        idle_timeout_ms,
        block_images,
        image_auto_resize,
        models,
        extra_skill_paths,
        extra_prompt_template_paths,
        model_explicit,
        reasoning_effort_explicit,
        resolved_model,
        reasoning_effort,
        steering_mode,
        follow_up_mode,
        resolved_session_dir,
        mcp_tools,
        mcp_catalog,
        mcp_apps,
        mcp_http,
        listen_uds_mode,
        configured_gateway,
    })
}

/// Connects the operator's configured MCP servers once, process-wide; none in service mode.
async fn connect_operator_mcp(
    mode: &Mode,
    stored_settings: &beyond_ai_agent::settings::Settings,
) -> (
    tools::mcp::McpTools,
    tools::mcp::McpCatalog,
    Option<tools::mcp_apps::McpAppsPool>,
) {
    // Connected exactly once, here, before `serve`'s long-lived session loop starts — not
    // re-done on every `set_model`/`set_thinking` registry rebuild (`build_tools` reads
    // `cfg.mcp_tools` as already-resolved). Fail-soft: a server that fails to connect is
    // skipped with a warning, matching `has_gated_resources`'s own "warn, don't block the run"
    // convention in the `run` path above. `stored_settings.mcp_servers` is already trust-gated —
    // see that field's own doc comment.
    //
    // Skipped entirely in service mode: these are the operator's own configured servers,
    // connected once process-wide and shared by every session — the exact opposite of what
    // a tenant needs. A tenant's connectors come from its grant, per session (a later PR);
    // until then it has none rather than the replica's.
    let (mcp_tools, mcp_catalog) = if mode.connects_operator_mcp() {
        let (mcp_tools, mcp_catalog, mcp_warnings) = tools::mcp::connect_all(
            stored_settings.mcp_servers.as_deref().unwrap_or(&[]),
            tools::mcp::idle_reap_after_from_env(),
            mcp_manifest_dir().as_ref(),
        )
        .await;
        for warning in &mcp_warnings {
            eprintln!("warning: {warning}");
        }
        (mcp_tools, mcp_catalog)
    } else {
        (
            tools::mcp::McpTools::default(),
            tools::mcp::McpCatalog::default(),
        )
    };
    // The same servers, dialed again with MCP Apps advertised — in the background, the first
    // time a session's client declares it renders apps (`set_mcp_apps`); never otherwise.
    // Service mode has none here: a tenant's apps pool is built per session, from its grant.
    let mcp_apps = mode.connects_operator_mcp().then(|| {
        tools::mcp::apps_pool(
            stored_settings.mcp_servers.as_deref().unwrap_or(&[]),
            tools::mcp::idle_reap_after_from_env(),
            mcp_manifest_dir(),
        )
    });
    (mcp_tools, mcp_catalog, mcp_apps)
}

/// The `ServeConfig` literal: every flag plus every `resolve`d setting.
fn build_config(
    cmd: Command,
    trust: ServeTrust,
    transport: &ServeTransport,
    resolved: ServeResolved,
) -> Result<serve::ServeConfig, Box<dyn std::error::Error>> {
    let Command::Serve {
        key,
        session_file,
        listen,
        listen_uds,
        session_idle_timeout,
        upstream_http2,
        max_live_sessions,
        notify_origin,
        drain_grace,
        mcp_events_callback_url,
        mcp_events_reapable,
        mcp_events_session,
        session_id,
        r#continue: continue_session,
        no_session_persistence,
        memory,
        no_memory,
        no_session_memory,
        max_steps,
        max_tokens,
        no_context_files,
        context_window,
        compact_at,
        no_tool_masking,
        tool_masking_keep_turns,
        tool_masking_keep_tokens,
        tool_masking_min_tokens,
        no_cached_summary,
        compaction_candidates,
        cache_ttl,
        cache_long,
        thinking,
        temperature,
        trust_project,
        force_untrusted,
        no_compaction,
        retry_rate_limit_secs,
        web_allow_private,
        web_allow_host,
        web_timeout_ms,
        exec_url,
        exec_header,
        exec_max_response_bytes,
        exec_cmd,
        lifecycle_url,
        lifecycle_header,
        lifecycle_heartbeat_secs,
        tools,
        exclude_tools,
        no_tools,
        sequential_tools,
        code_mode,
        mcp_tool_loading,
        deny_tool,
        deny_bash_pattern,
        deny_path,
        approve,
        approval_timeout,
        no_skills,
        approve_mcp_skills,
        no_prompt_templates,
        name,
        ..
    } = cmd
    else {
        unreachable!("dispatched on Command::Serve")
    };
    let ServeTrust {
        mode,
        grant_verifier,
        shards,
    } = trust;
    let ServeResolved {
        system,
        append_system_prompt,
        stored_settings,
        bash_shell_path,
        bash_command_prefix,
        compaction_reserve_tokens,
        compaction_keep_recent_tokens,
        branch_summary_reserve_tokens,
        retry_max_retries,
        retry_base_delay_ms,
        retry_max_backoff_ms,
        idle_timeout_ms,
        block_images,
        image_auto_resize,
        models,
        extra_skill_paths,
        extra_prompt_template_paths,
        model_explicit,
        reasoning_effort_explicit,
        resolved_model,
        reasoning_effort,
        steering_mode,
        follow_up_mode,
        resolved_session_dir,
        mcp_tools,
        mcp_catalog,
        mcp_apps,
        mcp_http,
        listen_uds_mode,
        configured_gateway,
    } = resolved;
    let serve_cfg = serve::ServeConfig {
        // Service mode reads no ambient provider env at all: `AI_DIRECT`/`AI_PROVIDER`/
        // `ANTHROPIC_API_KEY` and friends are this replica's environment, and a session's
        // only credential is the one sealed in its grant (see `serve::build_gateway_client`).
        // An empty `ProviderEnv` keeps every later `resolve_gateway_credential` call inert.
        provider_env: mode.provider_env(configured_gateway.is_some()),
        gateway: configured_gateway.unwrap_or_else(|| DEFAULT_GATEWAY.to_string()),
        key,
        model: resolved_model,
        model_explicit,
        reasoning_effort_explicit,
        max_steps,
        max_tokens,
        system,
        append_system: append_system_prompt,
        context_files: !no_context_files,
        session_file,
        session_dir: resolved_session_dir,
        // Fold the stored default in when neither flag/env was given, mirroring the other
        // `default_*` resolutions above. The backend itself is resolved by `serve_session` (which
        // knows the resolved `cwd`), like `agents`.
        memory: memory.or_else(|| stored_settings.default_memory_backend.clone()),
        no_memory,
        no_session_memory,
        listen,
        listen_uds: listen_uds.clone(),
        listen_uds_mode,
        grant_verifier,
        job_notify: None,
        notify_origin: notify_origin.clone(),
        session_idle_timeout: session_idle_timeout.map(std::time::Duration::from_secs),
        upstream_http2,
        // The daemon path (`serve_ws`) fills this from `upstream_http2` before spawning any
        // session; the stdio/`run` path leaves it `None` and never pools.
        shared_http: None,
        mode,
        shards,
        // Per connection, never per process: filled in by `serve_ws::session_cfg` from the
        // grant that spawned the session.
        max_live_sessions,
        mcp_http,
        mcp_events_callback_url,
        mcp_events_reapable,
        mcp_events_session: mcp_events_session
            .clone()
            .unwrap_or_else(|| tools::mcp_events::DEFAULT_EVENTS_SESSION.to_owned()),
        // Over stdio the one session owns the configured subscriptions; the daemon picks
        // its events session per session (`serve_ws::session_cfg`).
        mcp_events_owner: true,
        metrics: transport.metrics.clone(),
        // The process `main` built: it owns the signal handler. `serve_ws::session_cfg`
        // flips this for each session it spawns.
        supervised: false,
        drain_grace: std::time::Duration::from_secs(drain_grace),
        session_id,
        continue_session,
        no_session_persistence,
        compaction: serve::CompactionFlags {
            context_window,
            compact_at,
            reserve_tokens: compaction_reserve_tokens,
            keep_recent_tokens: compaction_keep_recent_tokens,
            no_tool_masking,
            tool_masking_keep_turns,
            tool_masking_keep_tokens,
            tool_masking_min_tokens,
            no_cached_summary,
            candidates: agent_core::CompactionCandidates::from_names(
                &compaction_candidates.unwrap_or_default(),
            )?,
        },
        cache_long: CacheTtl::resolve(cache_ttl, cache_long, CacheTtl::OneHour),
        thinking,
        reasoning_effort,
        temperature,
        trust_project,
        force_untrusted,
        branch_summary_reserve_tokens,
        no_compaction,
        retry_max_retries,
        retry_base_delay_ms: retry_base_delay_ms.map(std::time::Duration::from_millis),
        retry_max_backoff_ms: retry_max_backoff_ms.map(std::time::Duration::from_millis),
        retry_rate_limit_secs,
        idle_timeout_ms,
        block_images,
        image_auto_resize,
        bash_shell_path,
        bash_command_prefix,
        web_allow_private,
        web_allow_hosts: web_allow_host,
        web_timeout_ms,
        exec_url,
        exec_header,
        exec_max_response_bytes: exec_max_response_bytes.map_or(
            beyond_ai_agent::exec_endpoint::DEFAULT_MAX_RESPONSE_BYTES,
            |n| usize::try_from(n).unwrap_or(usize::MAX),
        ),
        exec_cmd,
        lifecycle: beyond_ai_agent::lifecycle::open(lifecycle_url.as_deref(), &lifecycle_header)
            .unwrap_or_else(|e| {
                eprintln!("{e}");
                exit_process(2);
            }),
        lifecycle_heartbeat: std::time::Duration::from_secs(lifecycle_heartbeat_secs),
        tools,
        exclude_tools,
        no_tools,
        mcp_tools,
        mcp_catalog,
        mcp_apps,
        // Discovered by `serve_session` itself, after it resolves project trust (a project-local
        // definition is trust-gated, like a skill) — not here, where the interactive trust grant
        // hasn't happened yet.
        agents: Vec::new(),
        sequential_tools,
        code_mode,
        mcp_loading: mcp_tool_loading.unwrap_or_default(),
        deny_tool,
        deny_bash_pattern,
        deny_path,
        // Fail fast on an unrecognized value, before any session is opened — the same discipline
        // `ToolPolicy::validate_deny_path_patterns` uses for a malformed `--deny-path` glob.
        approve: beyond_ai_agent::approval::GatedSet::parse(&approve).unwrap_or_else(|e| {
            eprintln!("{e}");
            exit_process(2);
        }),
        approval_timeout: (approval_timeout > 0)
            .then(|| std::time::Duration::from_secs(approval_timeout)),
        models: models.unwrap_or_default(),
        no_skills,
        approve_mcp_skills,
        no_prompt_templates,
        extra_skill_paths,
        extra_prompt_template_paths,
        name,
        // Fix 1 (pi-parity gap): previously never threaded into `serve` at all, so a persisted
        // `agent settings --default-project-trust` policy had zero effect on serve sessions even
        // though `run` above already partially honored it — see `serve::resolve_project_trust`,
        // the shared precedence both now consult.
        default_project_trust: stored_settings.default_project_trust,
        steering_mode,
        follow_up_mode,
    };
    Ok(serve_cfg)
}

/// Stdio, or the WebSocket listeners (TCP, TLS, UDS, systemd-activated), until shutdown.
async fn run_transport(
    serve_cfg: serve::ServeConfig,
    transport: ServeTransport,
) -> Result<Option<serve::Signal>, Box<dyn std::error::Error>> {
    let ServeTransport {
        tls,
        systemd_activated,
        metrics_addr,
        metrics,
    } = transport;
    // WebSocket transport (TCP and/or UDS): one session per `?session_id=`, each outliving its
    // connection so a dropped client re-attaches to a still-running run (see `serve_ws`). Both
    // listeners front one shared supervisor. Absent both ⇒ the default stdio transport.
    //
    // systemd socket activation: if systemd started us with a passed socket (`LISTEN_FDS`) and
    // no explicit listen flag was given, adopt that socket instead of binding — the socket then
    // outlives a `systemctl restart` (connections queue in the kernel). Unix/systemd only, and
    // deferred to `--listen`/`--listen-uds` when either is set. Computed once, up with the
    // `--service` validation, since that refuses stdio and so has to know.
    let use_ws = serve_cfg.listen.is_some()
        || tls.is_some()
        || serve_cfg.listen_uds.is_some()
        || systemd_activated;
    let shutdown_cause = if use_ws {
        #[cfg(not(unix))]
        if serve_cfg.listen_uds.is_some() {
            return Err("--listen-uds is only supported on unix targets".into());
        }
        // Bound here, before the agent's own listeners, so a port clash fails startup
        // rather than leaving a replica serving tenants with no way to observe it. Its task
        // is detached: a scrape endpoint must never be able to end the process it reports on.
        if let (Some(addr), Some(metrics)) = (metrics_addr, metrics.clone()) {
            let listener = tokio::net::TcpListener::bind(addr)
                .await
                .map_err(|e| format!("--metrics-listen {addr}: {e}"))?;
            tokio::spawn(beyond_ai_agent::metrics::serve_metrics(listener, metrics));
        }
        let listeners = serve_ws::ServeListeners {
            tcp: serve_cfg.listen,
            tls,
            #[cfg(unix)]
            uds: serve_cfg.listen_uds.clone(),
            #[cfg(unix)]
            uds_mode: serve_cfg.listen_uds_mode,
            systemd: systemd_activated,
        };
        Box::pin(serve_ws::serve_ws(serve_cfg, listeners)).await?
    } else {
        Box::pin(serve::serve(serve_cfg)).await?
    };
    Ok(shutdown_cause)
}
