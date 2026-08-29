//! Beyond agent harness — CLI.
//!
//! `run` drives a one-shot coding task to completion through the gateway. `serve` exposes the
//! headless control protocol (newline-delimited JSON over stdio). `tools` lists the advertised tool
//! set. Model traffic always flows through the gateway (`AI_GATEWAY_URL`) authenticated with a
//! `bai_v1` key (`AI_AGENT_KEY`).

// Unit tests assert preconditions with `.unwrap()`; allow that under `test` (matches the gateway and
// agent-core crate roots). Production paths stay panic-free per the workspace lints.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

// mimalloc, matching `edge`/`logfwd`/`orchestrator`/`tunnel` (the fleet default); it also fixes
// musl's slow multithreaded malloc, which matters for the static musl build of this CLI.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::ffi::{OsStr, OsString};
use std::fs;
use std::io::{IsTerminal, Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use agent_core::{Agent, GatewayClient, Session, StreamEvent, Tool};
use beyond_ai_agent::gateway_credential::{
    GatewayCredential, resolve_gateway_credential_and_headers,
};
use beyond_ai_agent::policy::ToolPolicy;
use beyond_ai_agent::session_store::{
    SessionMeta, SessionRepo, SessionStore, canonical_cwd, default_session_dir, fork_by_arg,
    is_path_like, open_session_by_id, sessions_root,
};
use beyond_ai_agent::{serve, serve_ws, tools};
use usage::{Cli, Subcommands};

/// Default model when neither `--model` nor `AI_AGENT_MODEL` is set.
const DEFAULT_MODEL: &str = "claude-opus-4-8";
/// Default gateway base URL.
const DEFAULT_GATEWAY: &str = "http://ai.internal";

/// Parse `--reasoning-effort`'s value into the wire-neutral [`agent_core::ThinkingLevel`] — the same
/// off-inclusive portable depth `--model <pattern>:<level>`'s suffix and the RPC-facing
/// `set_reasoning_effort`/`set_thinking` commands already parse via `ThinkingLevel::parse`. Task 2
/// (pi-parity fix, pass 19): previously returned `agent_core::ReasoningEffort` (no `Off` variant at
/// all), so `"off"` was a hard clap error here even though pi's own `--thinking off` and this crate's
/// own RPC toggle both already treat it as a first-class, explicit "disable reasoning" request — the
/// only workaround was the unrelated `--model <pattern>:off` colon-suffix shorthand. A caller that needs
/// the wire-level `ReasoningEffort` (`None` for `Off`) converts via `ThinkingLevel::reasoning_effort`
/// once every other candidate source (CLI flag, model suffix, stored setting) has had its turn — see
/// `run_task`'s/`Command::Serve`'s own resolution chains.
fn parse_reasoning_effort(s: &str) -> Result<agent_core::ThinkingLevel, String> {
    agent_core::ThinkingLevel::parse(s).ok_or_else(|| {
        format!("invalid reasoning effort {s:?}; expected one of off/minimal/low/medium/high/xhigh")
    })
}

/// Parse `settings::Settings::thinking_budget_overrides`'s plain-string keys into the
/// `agent_core::ReasoningEffort`-keyed table `agent_core::models::budget_for_effort_with_override`
/// needs (Task #36, pi-parity feature) — an unrecognized key (a hand-edited typo in `settings.json`) is
/// skipped rather than failing the whole lookup, matching this crate's usual "corrupt/unknown persisted
/// value degrades to not-set" convention (see `default_reasoning_effort`'s identical precedent above).
/// `None` when no table is configured at all, or every entry in it was unrecognized. `parse_reasoning_effort`
/// now also accepts `"off"` (Task 2, pi-parity fix) — an `off=<tokens>` entry parses fine but is dropped
/// here (`ThinkingLevel::Off.reasoning_effort()` is `None`, filtered out by the trailing `and_then`) since
/// a token-budget override keyed on "no reasoning at all" has nothing to ever apply to; harmless, and
/// consistent with this same "unrecognized/inapplicable degrades to not-set" convention rather than a
/// second, diverging validator.
fn resolve_thinking_budget_overrides(
    settings: &beyond_ai_agent::settings::Settings,
) -> Option<std::collections::HashMap<agent_core::ReasoningEffort, u32>> {
    let table = settings.thinking_budget_overrides.as_ref()?;
    let map: std::collections::HashMap<agent_core::ReasoningEffort, u32> = table
        .iter()
        .filter_map(|(k, v)| {
            parse_reasoning_effort(k)
                .ok()
                .and_then(|level| level.reasoning_effort())
                .map(|effort| (effort, *v))
        })
        .collect();
    (!map.is_empty()).then_some(map)
}

fn unknown_provider_error(provider: &str) -> String {
    format!(
        "unknown provider {provider:?}; expected one of: anthropic, github-copilot, openai-codex"
    )
}

/// The `run` path's per-model client construction, factored out so a **subagent** can build a
/// transport for a child that names its own `model:`. Credentials and routing are model-keyed
/// (`resolve_gateway_credential`, never cached across a model switch), so a child on a different model
/// cannot reuse the parent's transport — it must re-resolve from the raw key and rebuild. Mirrors the
/// parent build in `run_task` exactly (retry, header override, backoff, idle timeout); keep the two in
/// step. Returns `String` errors because that's what [`tools::subagent::TransportFactory`] wants.
#[allow(clippy::too_many_arguments)]
fn build_run_gateway_client(
    raw_key: Option<String>,
    gateway: &str,
    model: &str,
    provider_env: &beyond_ai_agent::gateway_credential::ProviderEnv,
    retry_max_retries: Option<u32>,
    retry_base_delay_ms: Option<u64>,
    retry_max_backoff_ms: Option<u64>,
    idle_timeout_ms: Option<u64>,
) -> Result<GatewayClient, String> {
    // One `models.json` parse feeds both the credential and the extra headers (T9-F3) — see
    // `resolve_gateway_credential_and_headers`.
    let (credential, extra_headers) =
        resolve_gateway_credential_and_headers(raw_key, model, provider_env)?;
    let mut client = match credential {
        GatewayCredential::Static(key) => {
            GatewayClient::new(gateway.to_string(), key).map_err(|e| e.to_string())?
        }
        GatewayCredential::Oauth(source) => {
            GatewayClient::with_credential_source(gateway.to_string(), source)
                .map_err(|e| e.to_string())?
        }
    }
    .with_retry(
        retry_max_retries.unwrap_or(agent_core::client::MAX_RETRIES),
        retry_base_delay_ms
            .map(std::time::Duration::from_millis)
            .unwrap_or(agent_core::client::BASE_BACKOFF),
    )
    .with_extra_headers(extra_headers);
    if let Some(ms) = retry_max_backoff_ms {
        client = client.with_max_backoff(std::time::Duration::from_millis(ms));
    }
    if let Some(ms) = idle_timeout_ms {
        client = client
            .with_idle_timeout(std::time::Duration::from_millis(ms))
            .map_err(|e| e.to_string())?;
    }
    Ok(client)
}

/// Drives `agent login`'s interactive prompts over stderr/stdin — the CLI's one implementation of
/// [`beyond_ai_agent::oauth::LoginCallbacks`]. `agent login` is a dedicated, single-purpose blocking
/// invocation with no concurrent-command concerns (unlike `serve`), so blocking stdin reads (moved to
/// a `spawn_blocking` task, out of hygiene rather than necessity here) are the whole interaction —
/// there's no need for `serve`'s RPC surface's separate ack-now/respond-later, push-frame shape.
struct CliLoginCallbacks;

#[async_trait::async_trait]
impl beyond_ai_agent::oauth::LoginCallbacks for CliLoginCallbacks {
    async fn show_auth_url(&self, url: &str, instructions: Option<&str>) {
        eprintln!("Open this URL in a browser to continue:\n\n  {url}\n");
        if let Some(instructions) = instructions {
            eprintln!("{instructions}");
        }
    }

    async fn show_device_code(&self, info: &beyond_ai_agent::oauth::DeviceCodeInfo) {
        eprintln!(
            "Go to {} and enter this code: {}",
            info.verification_uri, info.user_code
        );
        eprintln!("Waiting for authorization...");
    }

    async fn progress(&self, message: &str) {
        eprintln!("{message}");
    }

    async fn prompt_text(
        &self,
        prompt: &beyond_ai_agent::oauth::TextPrompt<'_>,
    ) -> Result<String, beyond_ai_agent::oauth::OAuthError> {
        eprint!("{}", prompt.message);
        if let Some(placeholder) = prompt.placeholder {
            eprint!(" [{placeholder}]");
        }
        eprint!(": ");
        let _ = std::io::stderr().flush();
        tokio::task::spawn_blocking(|| {
            let mut line = String::new();
            std::io::stdin()
                .read_line(&mut line)
                .map_err(|e| beyond_ai_agent::oauth::OAuthError::InvalidInput(e.to_string()))?;
            Ok(line.trim().to_string())
        })
        .await
        .map_err(|e| beyond_ai_agent::oauth::OAuthError::InvalidInput(e.to_string()))?
    }

    async fn select(
        &self,
        prompt: &beyond_ai_agent::oauth::SelectPrompt<'_>,
    ) -> Result<Option<String>, beyond_ai_agent::oauth::OAuthError> {
        eprintln!("{}", prompt.message);
        for (i, opt) in prompt.options.iter().enumerate() {
            eprintln!("  {}. {} ({})", i + 1, opt.label, opt.id);
        }
        eprint!("Enter a number [1]: ");
        let _ = std::io::stderr().flush();
        let choice = tokio::task::spawn_blocking(|| {
            let mut line = String::new();
            let _ = std::io::stdin().read_line(&mut line);
            line.trim().to_string()
        })
        .await
        .unwrap_or_default();

        let options: Vec<&str> = prompt.options.iter().map(|o| o.id.as_str()).collect();
        if choice.is_empty() {
            return Ok(options.first().map(|s| s.to_string()));
        }
        if let Ok(n) = choice.parse::<usize>()
            && n >= 1
            && n <= options.len()
        {
            return Ok(Some(options[n - 1].to_string()));
        }
        // Also accept typing the id directly.
        Ok(options
            .into_iter()
            .find(|id| *id == choice)
            .map(str::to_string))
    }
}

#[derive(Cli)]
#[usage(
    bin = "beyond-ai-agent",
    version,
    about = "Beyond agent harness",
    unknown_flags = "error"
)]
struct Cli {
    #[usage(subcommand)]
    command: Command,
}

#[derive(Subcommands)]
enum Command {
    /// Run a one-shot agent task to completion, streaming output to stdout.
    Run {
        /// The task prompt for the agent. Multiple messages run as separate, sequential turns (the
        /// second is sent only after the first fully completes). An argument starting with `@` is a
        /// file reference instead of a message: its contents are read and wrapped in a
        /// `<file name="...">` block prepended to the *first* message (stdin, if piped, comes before
        /// that). At least one of a message, `@file`, or piped stdin is required.
        tasks: Vec<String>,
        /// Model id (default `claude-opus-4-8`, or `AI_AGENT_MODEL`).
        #[usage(long, env = "AI_AGENT_MODEL")]
        model: Option<String>,
        /// Gateway base URL (default `http://ai.internal`, or `AI_GATEWAY_URL`).
        #[usage(long, env = "AI_GATEWAY_URL")]
        gateway_url: Option<String>,
        /// Virtual key (`bai_v1…`) or BYO provider key. Required; or set `AI_AGENT_KEY`.
        #[usage(long, env = "AI_AGENT_KEY")]
        key: Option<String>,
        /// Opt-in cap on model turns before bailing with an error (default: unbounded).
        #[usage(long)]
        max_steps: Option<u32>,
        /// Per-turn output token ceiling. `serve`'s identical flag; defaults to the model's own
        /// capability-table `max_output` (see `agent_core::models::capabilities`) when omitted.
        #[usage(long, env = "AI_AGENT_MAX_TOKENS")]
        max_tokens: Option<u32>,
        /// Use the 1-hour prompt-cache TTL (vs 5 minutes); helps when turns are spaced out. `serve`'s
        /// identical flag; `run`'s one-shot single-turn case rarely benefits, but a multi-message
        /// invocation (several `tasks` sent as sequential turns) can.
        #[usage(long)]
        cache_long: bool,
        /// Enable extended thinking with this token budget (must be below the per-turn max tokens). A
        /// raw token count, not pi's own `--thinking <level>` (off/minimal/low/medium/high/xhigh) — see
        /// `--reasoning-effort` for that portable level instead. `serve`'s identical flag; unlike
        /// `serve`, `run` has no thinking-level cycling, so this is applied as-is with no per-model
        /// default derivation when omitted.
        #[usage(long)]
        thinking: Option<u32>,
        /// Reasoning effort for models driven by an effort level rather than a token budget (OpenAI
        /// reasoning models via `reasoning_effort`; Anthropic adaptive-thinking models via
        /// `output_config.effort`). One of off/minimal/low/medium/high/xhigh — see `--thinking` for a
        /// raw token-budget override instead. Ignored by models that take neither shape. Falls back to
        /// `AI_AGENT_REASONING_EFFORT`, then the stored `agent settings --default-reasoning-effort`
        /// default (Fix 2 — pi-parity gap: previously the only numeric/string CLI tunable with no
        /// persisted-default fallback at all), before finally leaving it unset. Task 2 (pi-parity fix,
        /// pass 19): `off` is now accepted too, explicitly disabling reasoning — previously the only way
        /// to do that from a startup flag was the unrelated `--model <pattern>:off` colon-suffix
        /// shorthand. `serve`'s identical flag.
        #[usage(long, env = "AI_AGENT_REASONING_EFFORT")]
        reasoning_effort: Option<agent_core::ThinkingLevel>,
        /// How much of the mid-run *steer* lane a single drain point consumes per turn boundary
        /// (`agent_core::QueueMode`) — `one_at_a_time` (the default, matching pi) injects only the oldest
        /// queued message per drain, leaving the rest queued for the next one; `all` folds everything
        /// queued into a single injection (this crate's original behavior). Task 1 (pi-parity fix, pass
        /// 19): `run` has no way to actually queue a steer message mid-invocation today (its `tasks` list
        /// runs as separate, sequential turns — see `tasks`'s own doc comment — not steer injections), so
        /// this has no observable effect yet; wired through anyway for parity with `serve`'s identical
        /// flag and pi's own persisted `steeringMode`, which applies at agent/session construction time
        /// in every mode, not just its TUI. Falls back to the persisted setting `serve`'s own
        /// `set_steering_mode` RPC command maintains (`settings::Settings::steering_mode`), before
        /// finally defaulting to `one_at_a_time`.
        #[usage(long, env = "AI_AGENT_STEERING_MODE")]
        steering_mode: Option<agent_core::QueueMode>,
        /// Same idea as `--steering-mode`, for the follow-up lane drained at a stop boundary (plus any
        /// stranded steer messages swept in there) — matches pi's own separate `followUpMode`.
        #[usage(long, env = "AI_AGENT_FOLLOW_UP_MODE")]
        follow_up_mode: Option<agent_core::QueueMode>,
        /// Sampling temperature. Omitted (leaving the provider default) unless set. Silently ignored by
        /// Anthropic while `--thinking` is set (Anthropic forbids the two together). `serve`'s identical
        /// flag.
        #[usage(long)]
        temperature: Option<f64>,
        /// Replace the built-in base system prompt entirely. `serve`'s identical flag — e.g. a
        /// specialized reviewer/persona invocation for automation that needs a wholly different agent
        /// identity, not just extra instructions layered on top (see `--append-system-prompt`).
        #[usage(long, env = "AI_AGENT_SYSTEM_PROMPT")]
        system_prompt: Option<String>,
        /// Append extra instructions after the base system prompt (built-in, or `--system-prompt` if
        /// also given). Repeatable — pi-parity fix: previously a second occurrence silently clobbered
        /// the first instead of accumulating (matches pi, whose `appendSystemPrompt` is itself an
        /// array). Each occurrence is joined with the others by a blank line, in the order given.
        /// `serve`'s identical flag.
        #[usage(long, env = "AI_AGENT_APPEND_SYSTEM_PROMPT")]
        append_system_prompt: Vec<String>,
        /// Trust `cwd` for this run only, so a project-local `.claude/SYSTEM.md` is honored even if
        /// `cwd` isn't in the persisted allowlist (`agent trust <path>`). A session-scoped override,
        /// not a permanent grant — see `agent trust` to record one. `-a` matches pi's own
        /// `--approve`/`-a` (same "trust this project" meaning, different flag name here).
        #[usage(short = 'a', long)]
        trust_project: bool,
        /// Force `cwd` *untrusted* for this run only, overriding both `--trust-project` and the
        /// persisted allowlist (`agent trust <path>`) — e.g. to test untrusted behavior against a
        /// directory that's otherwise permanently trusted. Wins over `--trust-project` if both are
        /// somehow given. `-na` matches pi's own `--no-approve`/`-na`.
        #[usage(long)]
        force_untrusted: bool,
        /// Model context window (tokens); the loop summarizes older turns to stay below it. Defaults
        /// to the model's own capability-table window (see `agent_core::models::capabilities`) — only
        /// pass this to pin a fixed budget regardless of which model ends up used. `serve`'s identical
        /// flag.
        #[usage(long, env = "AI_AGENT_CONTEXT_WINDOW")]
        context_window: Option<u32>,
        /// Compaction headroom (tokens) reserved below the context window before it fires. Defaults to
        /// `CompactionConfig::default()`'s 16,384. `serve`'s identical flag.
        #[usage(long, env = "AI_AGENT_COMPACTION_RESERVE_TOKENS")]
        compaction_reserve_tokens: Option<u32>,
        /// Roughly how many tokens of recent conversation compaction keeps verbatim. Defaults to
        /// `CompactionConfig::default()`'s 20,000. `serve`'s identical flag.
        #[usage(long, env = "AI_AGENT_COMPACTION_KEEP_RECENT_TOKENS")]
        compaction_keep_recent_tokens: Option<u32>,
        /// Token budget reserved below the context window when summarizing an abandoned tree branch —
        /// independent of `--compaction-reserve-tokens` (ordinary compaction's own reserve). Defaults to
        /// hard-tying to whatever `--compaction-reserve-tokens` resolves to (this crate's prior
        /// behavior); pi's own `branchSummary.reserveTokens` default is 16384. `serve`'s identical flag
        /// (Task #31, pi-parity feature: `agent_core::Agent::with_branch_summary_reserve_tokens`
        /// previously had no caller in either binary).
        #[usage(long, env = "AI_AGENT_BRANCH_SUMMARY_RESERVE_TOKENS")]
        branch_summary_reserve_tokens: Option<u32>,
        /// Disable automatic (threshold-triggered) compaction entirely — the loop only ever compacts on
        /// a genuine overflow (`agent_core::CompactionConfig::enabled`'s own doc comment: manual/overflow
        /// compaction ignores this setting), never proactively. For a caller that would rather fail/see
        /// the raw context-window error than have older turns silently summarized away.
        #[usage(long, env = "AI_AGENT_NO_COMPACTION")]
        no_compaction: bool,
        /// How many times to retry a gateway request that fails before the first response byte
        /// arrives. Defaults to 3. `serve`'s identical flag.
        #[usage(long, env = "AI_AGENT_RETRY_MAX_RETRIES")]
        retry_max_retries: Option<u32>,
        /// Base of the exponential backoff between those retries, in milliseconds. Defaults to 250.
        /// `serve`'s identical flag.
        #[usage(long, env = "AI_AGENT_RETRY_BASE_DELAY_MS")]
        retry_base_delay_ms: Option<u64>,
        /// Ceiling on that exponential backoff, in milliseconds — overrides `agent_core::client::
        /// GatewayClient::with_max_backoff`'s built-in 60s default (`agent_core::client::MAX_BACKOFF`).
        /// `serve`'s identical flag (Task #30, pi-parity feature: the retry cluster's third knob,
        /// previously with no CLI flag or persisted override at all, unlike its two siblings above).
        #[usage(long, env = "AI_AGENT_RETRY_MAX_BACKOFF_MS")]
        retry_max_backoff_ms: Option<u64>,
        /// Disable the whole-run auto-retry-after-error loop outright — pi's `RetrySettings.enabled:
        /// false` (`settings-manager.ts:28`). Matches this codebase's own `--no-x` convention for
        /// negating an on-by-default behavior (`--no-compaction`, `--no-skills`, ...). Only gates the
        /// whole-run retry-after-error loop (a run that already exhausted every within-turn retry and
        /// still ended in a transient-looking error, re-invoked from scratch — see
        /// `beyond_ai_agent::retry`'s own module doc comment); the separate pre-connect/mid-stream layer
        /// just above (`--retry-max-retries`/`--retry-base-delay-ms`, `agent_core::client`) is
        /// unaffected either way — matches pi's own `RetrySettings.enabled`, which gates only its
        /// equivalent whole-run loop. Functionally equivalent to `--retry-max-retries 0` (Task #52,
        /// pi-parity fix: previously the only discoverable spelling of "never retry a whole run"), just
        /// under a name that says what it means without requiring the operator to already know `0` has
        /// this effect.
        #[usage(long, env = "AI_AGENT_NO_RETRY")]
        no_retry: bool,
        /// Idle-read timeout between response chunks on the gateway HTTP client, in milliseconds —
        /// overrides `agent_core::client::GatewayClient`'s built-in default, tuned for the gateway's own
        /// upstream assumption. Consulted for every routing path (proxied through the gateway, or a
        /// direct-routed/custom `models.json` `base_url` override, which bypasses the gateway entirely)
        /// since a self-hosted or alternate-provider endpoint's own slow/fast tail has no reason to
        /// match the gateway's (Task #19, pi-parity feature).
        #[usage(long, env = "AI_AGENT_IDLE_TIMEOUT_MS")]
        idle_timeout_ms: Option<u64>,
        /// Force every image down the same downgrade-to-text-placeholder path a vision-incapable model
        /// already gets, regardless of the active model's real `supports_vision` capability — for an
        /// operator who wants image bytes kept out of the prompt entirely (bandwidth, compliance, a
        /// proxy that strips/rejects multipart image content) even on a vision-capable model. Falls back
        /// to the persisted `agent settings --block-images` default when not explicitly given (Task #26,
        /// pi-parity feature).
        #[usage(long, env = "AI_AGENT_BLOCK_IMAGES")]
        block_images: bool,
        /// Force `--block-images` off for this invocation, even when a persisted `agent settings
        /// --block-images` default is `true` — `--block-images` above only ever ORs an explicit `true`
        /// in, so previously there was no way to override a persisted `true` default back to `false`
        /// for a single `run` (pass 20, pi-parity fix). Wins outright over both the persisted default
        /// and an explicit `--block-images`, mirroring `--no-image-auto-resize`'s identical
        /// escape-hatch shape just below.
        #[usage(long, env = "AI_AGENT_NO_BLOCK_IMAGES")]
        no_block_images: bool,
        /// Skip `read`'s resize/downscale path for an oversized image entirely, shipping its
        /// normalized (format-converted, if needed) bytes as-is regardless of size or pixel
        /// dimensions — pi's `ImageSettings.autoResize` (default enabled) inverted, matching this
        /// codebase's `--no-x` convention for negating an on-by-default behavior (`--no-compaction`,
        /// `--no-skills`, ...). Falls back to the persisted `agent settings --image-auto-resize`
        /// default when not explicitly given (Task #4, pi-parity feature).
        #[usage(long, env = "AI_AGENT_NO_IMAGE_AUTO_RESIZE")]
        no_image_auto_resize: bool,
        /// Default `bash` command timeout (ms) when the model omits `timeout_ms`. Defaults to 1,800,000
        /// (30 minutes). `serve`'s identical flag.
        #[usage(long, env = "AI_AGENT_BASH_TIMEOUT_MS")]
        bash_timeout_ms: Option<u64>,
        /// Run `bash` commands through this shell instead of the auto-resolved one (`/bin/bash`, else
        /// `bash` on `$PATH`, else `sh`). Matches pi's own `shellPath` setting. `serve`'s identical flag.
        #[usage(long, env = "AI_AGENT_BASH_SHELL_PATH")]
        bash_shell_path: Option<String>,
        /// Prepend this line to every `bash` command, in the same shell invocation (e.g. sourcing a
        /// project's env setup, activating a venv). Matches pi's own `shellCommandPrefix` setting.
        /// `serve`'s identical flag.
        #[usage(long, env = "AI_AGENT_BASH_COMMAND_PREFIX")]
        bash_command_prefix: Option<String>,
        /// Let the `web` tool reach loopback/private/link-local addresses. Off by default: the tool
        /// refuses them to prevent SSRF (it fetches URLs the model chose). `serve`'s identical flag.
        #[usage(long, env = "AI_AGENT_WEB_ALLOW_PRIVATE")]
        web_allow_private: bool,
        /// A hostname the `web` tool may reach even with private egress off (repeatable) — an internal
        /// service, or `127.0.0.1` for local testing. `serve`'s identical flag.
        #[usage(long, env = "AI_AGENT_WEB_ALLOW_HOST")]
        web_allow_host: Vec<String>,
        /// The `web` tool's per-request timeout (ms). Default 30,000. `serve`'s identical flag.
        #[usage(long, env = "AI_AGENT_WEB_TIMEOUT_MS")]
        web_timeout_ms: Option<u64>,
        /// Run the filesystem tools (`read`/`write`/`edit`/`ls`/`grep`/`find`) against a remote
        /// **exec endpoint** instead of this host. Any URL that accepts
        /// `POST {command, args, cwd, timeout_ms}` and answers `{exit_code, stdout, stderr}`.
        ///
        /// Vendor-agnostic by construction: put a few dozen lines in front of Daytona, E2B, Modal, a
        /// container service or your own runner, and the agent neither knows nor cares which. The
        /// endpoint must already exist — provisioning and teardown are the caller's, not the agent's.
        #[usage(long, env = "AI_AGENT_EXEC_URL", conflicts = "exec_cmd")]
        exec_url: Option<String>,
        /// A header to send with every exec request, `Name: value`. Repeatable. This is where auth
        /// goes; which scheme the endpoint wants is the endpoint's business.
        #[usage(long, env = "AI_AGENT_EXEC_HEADER")]
        exec_header: Vec<String>,
        /// For targets with no HTTP surface: an argv template whose `{}` is replaced by the command,
        /// e.g. `--exec-cmd 'ssh build-host -- {}'` or `--exec-cmd 'docker exec ctr {}'`. The command
        /// expands to separate argv entries, never into a shell string.
        #[usage(long, env = "AI_AGENT_EXEC_CMD", conflicts = "exec_url")]
        exec_cmd: Option<String>,
        /// Restrict the tool set to exactly these names (comma-separated), dropping everything else.
        /// Combine with `--exclude-tools` to carve one back out of the allow-list. `serve`'s identical
        /// flag/env var — a deployment convention setting this env var to sandbox an agent must apply
        /// here too, not just to `serve`. `-t` matches pi's own `--tools`/`-t`.
        #[usage(short = 't', long, env = "AI_AGENT_TOOLS", delimiter = ',')]
        tools: Option<Vec<String>>,
        /// Drop these tools (comma-separated) from the default set — e.g. `--exclude-tools bash,write`
        /// for a read-only reviewer that can't run shell commands or mutate files. `serve`'s identical
        /// flag/env var. `-xt` matches pi's own `--exclude-tools`/`-xt`.
        #[usage(long, env = "AI_AGENT_EXCLUDE_TOOLS", delimiter = ',')]
        exclude_tools: Option<Vec<String>>,
        /// Register no tools at all — a pure-conversation run. Wins over `--tools`/`--exclude-tools`.
        /// `-nt` matches pi's own `--no-tools`/`-nt`.
        #[usage(long)]
        no_tools: bool,
        /// Force every batch of tool calls in a turn to run one at a time instead of the default
        /// bounded-concurrent dispatch (`agent_core::Agent::with_sequential_tools`) — e.g. for a
        /// deterministic repro, or a host policy that never wants two tool calls actually overlapping.
        /// `serve`'s identical flag.
        #[usage(long)]
        sequential_tools: bool,
        /// Block every call to this tool (comma-separated, repeatable), even though it stays visible
        /// and registered — unlike `--exclude-tools` (the model never sees an excluded tool exists at
        /// all), a denied call still surfaces to the model as a normal error `tool_result` explaining
        /// it was blocked by policy. Installs an `agent_core::AgentHooks` permission gate
        /// (`policy::ToolPolicy`) on the agent; a no-op (no hook installed at all) when combined with
        /// `--deny-bash-pattern` leaves the list empty.
        #[usage(long, env = "AI_AGENT_DENY_TOOL", delimiter = ',')]
        deny_tool: Vec<String>,
        /// Block a `bash` call whenever its command contains this substring, case-insensitively
        /// (comma-separated, repeatable) — e.g. `--deny-bash-pattern "rm -rf"`. Combines with
        /// `--deny-tool` under the same policy hook.
        #[usage(long, env = "AI_AGENT_DENY_BASH_PATTERN", delimiter = ',')]
        deny_bash_pattern: Vec<String>,
        /// Block a `write`/`edit` call whenever its `path` argument matches this glob (comma-separated,
        /// repeatable) — e.g. `--deny-path '.env,**/secrets/**'`. Same glob engine as `grep`'s
        /// `--glob`/`find`'s pattern matching (`globset::Glob`). Combines with `--deny-tool`/
        /// `--deny-bash-pattern` under the same policy hook.
        #[usage(long, env = "AI_AGENT_DENY_PATH", delimiter = ',')]
        deny_path: Vec<String>,
        /// Disable *standard-root* skills discovery/loading (`~/.claude/skills`, `<cwd>/.claude/skills`)
        /// — no `<available_skills>` listing in the system prompt from either, and a `/skill:name`
        /// invocation in the task message is sent through unexpanded unless it resolves against a
        /// `--skill` path instead. An explicit `--skill <path>` is still honored even so — pi's own
        /// `--no-skills` does the same (a documented, tested combination: it's a way to say "nothing
        /// auto-discovered, only what I explicitly listed", not "no skills at all"). A one-shot `run`
        /// has no `reload` to re-enable it mid-process, unlike `serve`. `-ns` matches pi's own
        /// `--no-skills`/`-ns`.
        #[usage(long)]
        no_skills: bool,
        /// Disable *standard-root* prompt-template discovery/loading (`~/.claude/prompts`,
        /// `<cwd>/.claude/prompts`) — a `/name` invocation in the task message is sent through
        /// unexpanded unless it resolves against a `--prompt-template` path instead. An explicit
        /// `--prompt-template <path>` is still honored even so, matching `--no-skills`'s identical
        /// carve-out and pi's own `--no-prompt-templates`. `-np` matches pi's own
        /// `--no-prompt-templates`/`-np`.
        #[usage(long)]
        no_prompt_templates: bool,
        /// Do not discover/inject AGENTS.md / CLAUDE.md project-instruction files. Matches `serve`'s
        /// identical flag — `run` previously hardcoded this on with no way to opt out. `-nc` matches
        /// pi's own `--no-context-files`/`-nc`.
        #[usage(long)]
        no_context_files: bool,
        /// Discover skills from this directory too, in addition to the two standard roots (repeatable,
        /// or comma-separated via `AI_AGENT_SKILL_PATH` — matching `--tools`/`AI_AGENT_TOOLS`'s own
        /// comma-separated env-var convention). Matches pi's own `--skill <path>`. A path that doesn't
        /// exist is warned about, not silently ignored. Wins over the standard roots on a name collision.
        #[usage(
            long = "skill",
            env = "AI_AGENT_SKILL_PATH",
            delimiter = ',',
            value_name = "PATH"
        )]
        extra_skill_paths: Vec<String>,
        /// Discover prompt templates from this directory too, in addition to the two standard roots
        /// (repeatable, or comma-separated via `AI_AGENT_PROMPT_TEMPLATE_PATH`). Matches pi's own
        /// `--prompt-template <path>`; see `--skill`'s doc comment for the missing-path/shadow-order
        /// behavior, which applies identically here.
        #[usage(
            long = "prompt-template",
            env = "AI_AGENT_PROMPT_TEMPLATE_PATH",
            delimiter = ',',
            value_name = "PATH"
        )]
        extra_prompt_template_paths: Vec<String>,
        /// Set this run's session name up front, before the first turn even starts — a whitespace-only
        /// value is rejected rather than silently producing a blank/meaningless name, matching pi's own
        /// `--name` validation. Unlike pi (renames unconditionally on every invocation), only takes
        /// effect on a genuinely fresh session — see the fresh-only check in `serve`, a deliberate
        /// deviation. The RPC `set_session_name` command covers renaming an already-running session.
        #[usage(short = 'n', long)]
        name: Option<String>,
        /// An extra guideline bullet appended to the default system prompt's `Guidelines:` section
        /// (repeatable) — pi's own `promptGuidelines`. Deduplicated and trimmed against the built-in
        /// guidelines.
        #[usage(long = "prompt-guideline", value_name = "TEXT")]
        prompt_guidelines: Vec<String>,
        /// Fork an existing session into a brand-new one and continue from there, rather than reopening
        /// it in place — by id (searched in this project first, then every other project's own session
        /// directory under `~/.claude/sessions/`) or by a direct path to its `.jsonl` file (any
        /// project). Matches pi's own cross-project `--fork <path|id>`; the forked copy's `cwd` is the
        /// *current* directory, not wherever the source session was originally recorded against. Wins
        /// over `--session`/`--continue` if more than one is given — a fork always starts a fresh child,
        /// never reopens one in place. Forks the whole active transcript; `serve`'s `fork`/`fork_at_entry`
        /// RPC commands cover forking at an earlier point once a session is running.
        #[usage(long, value_name = "PATH_OR_ID")]
        fork: Option<String>,
        /// Persist this run to a specific session, creating it if missing or continuing it if it already
        /// exists — so a later `run --session <path|id>` picks up where this one left off. Accepts
        /// either a direct path to a `.jsonl` file (created fresh if it doesn't exist yet) or a bare
        /// session id/unique prefix, resolved against the current project's own repo first, then every
        /// other project's under `--session-dir`'s root — matching pi's own `--session <path|id>`. Wins
        /// over `--continue` if both are given.
        #[usage(long)]
        session: Option<String>,
        /// Address this exact session: continue it if it already exists, or create it under exactly this
        /// id if it doesn't. Gives a caller (a script, an orchestrator, a test harness) a known,
        /// predictable name to route on rather than parsing an id back out of the run's own output, and
        /// re-running with the same id is idempotent — same conversation, every time.
        ///
        /// Outranks `--continue`, which only describes a session ("whatever ran here last") where this
        /// names one. Distinct ids in one directory are distinct sessions: this used to be discarded
        /// whenever *any* session already existed for the current cwd, which silently collapsed every id
        /// onto one shared conversation. Ignored with `--no-session-persistence` (nothing is written, so
        /// it only names the in-memory session) and with `--session <path>` (that already names a file).
        #[usage(long)]
        session_id: Option<String>,
        /// Continue the most recent session for the current directory (the same
        /// `~/.claude/sessions/<encoded-cwd>/` repo `serve` defaults to), creating one if this is the
        /// first run here. This is the *only* flag that reattaches implicitly — a plain no-flag `run`
        /// starts a new session (still persisted; see `--no-session-persistence`). Ignored if
        /// `--session`/`--session-id` is also given, both of which name a session outright.
        #[usage(long, short = 'c')]
        r#continue: bool,
        /// Use this directory as the session repo instead of the default `~/.claude/sessions/
        /// <encoded-cwd>/` — matches `serve`'s own `--session-dir`/`AI_AGENT_SESSION_DIR` (same flag,
        /// same meaning: the directory itself becomes the repo root, not a further per-cwd subdirectory
        /// under it). Affects `--continue`, `--fork <id>`'s target project and cross-project search root
        /// (that search then spans this directory's own siblings, matching how `serve`'s
        /// `list_all_sessions` scopes its cross-project scan off `--session-dir`'s parent), and a plain
        /// no-flag run's own default repo. Has no effect on `--session <path>` (already names an exact
        /// file directly) or `--no-session-persistence` (opts out of persistence entirely, so there is no
        /// repo to redirect).
        #[usage(long, env = "AI_AGENT_SESSION_DIR")]
        session_dir: Option<String>,
        /// Skip persistence entirely, even without `--session`/`--continue`/`--fork`. Without this, a
        /// plain no-flag `run` writes a new session to the same per-cwd repo `serve` uses
        /// (`~/.claude/sessions/<encoded-cwd>/`, or `--session-dir`) rather than running in-memory-only —
        /// pass this for the rare case that's genuinely what you want (e.g. a short-lived script that
        /// mustn't leave a session file behind). Matches `serve`'s identical flag, so the CLI vocabulary
        /// for opting out is the same either way. `--continue` overrides it; `--session-id` does not.
        #[usage(long)]
        no_session_persistence: bool,
        /// Persistent-memory backend DSN. Absent ⇒ the stored `default_memory_backend` setting, else a
        /// per-project local-file store under `~/.claude/projects/<cwd>/memory/`. A bare path or
        /// `file://` names a directory; `redis://`/`postgres://` select a networked backend (recognized,
        /// not yet implemented). See [`beyond_ai_agent::memory`].
        #[usage(long, env = "AI_AGENT_MEMORY_URL")]
        memory: Option<String>,
        /// Disable persistent memory entirely: don't register the `memory` tool or inject the MEMORY.md
        /// index. Mirrors `--no-tools`'s opt-out style.
        #[usage(long)]
        no_memory: bool,
        /// Disable only the per-session `/session` working-memory mount, keeping durable `/memories`. The
        /// working store is the compaction-surviving scratchpad; on by default when memory is enabled.
        #[usage(long)]
        no_session_memory: bool,
        /// After the run completes, export the transcript as a self-contained HTML file at this path
        /// (parent directories are created as needed) — the same rendering `serve`'s `export_html` RPC
        /// command produces, for a one-shot run with no server involved.
        #[usage(long)]
        export: Option<String>,
        /// Emit newline-delimited JSON to stdout instead of human-readable text: one leading session
        /// header line, then one `AgentEvent` object per line (tool calls/results and turn boundaries
        /// included, not just raw text deltas) — the same event shape `serve`'s NDJSON protocol streams,
        /// for a scripting caller that wants structured output without spawning `serve`.
        #[usage(long)]
        json: bool,
        /// Make this run a callable function: a JSON Schema (inline, or a path to a `.json` file) the
        /// agent must fill in and return via the `structured_output` tool instead of ending in prose.
        /// The validated payload is printed as the last stdout line; the run exits non-zero if the model
        /// never produced one. Registered after `--tools`/`--exclude-tools` filtering, so an unrelated
        /// allow-list can't strip the one tool this flag exists to add.
        #[usage(long, env = "AI_AGENT_OUTPUT_SCHEMA")]
        output_schema: Option<String>,
        /// Override the `structured_output` tool's description — what the model is told the payload is
        /// for. Ignored without `--output-schema`.
        #[usage(long)]
        output_description: Option<String>,
    },
    /// Run the headless agent server: a newline-delimited JSON control protocol over stdio.
    Serve {
        /// Model id (default `claude-opus-4-8`, or `AI_AGENT_MODEL`).
        #[usage(long, env = "AI_AGENT_MODEL")]
        model: Option<String>,
        /// Gateway base URL (default `http://ai.internal`, or `AI_GATEWAY_URL`).
        #[usage(long, env = "AI_GATEWAY_URL")]
        gateway_url: Option<String>,
        /// Virtual key (`bai_v1…`) or BYO provider key. Required; or set `AI_AGENT_KEY`.
        #[usage(long, env = "AI_AGENT_KEY")]
        key: Option<String>,
        /// Persist one session to this JSONL file so a later `serve` reattaches with the transcript.
        #[usage(long, env = "AI_AGENT_SESSION_FILE")]
        session_file: Option<String>,
        /// Persist many sessions under this directory (enables list/switch/fork/name commands).
        #[usage(long, env = "AI_AGENT_SESSION_DIR")]
        session_dir: Option<String>,
        /// Offer the control protocol over a WebSocket on this address instead of stdio (e.g.
        /// `127.0.0.1:8787`). Each connection drives a session at `/_beyond/agent?session_id=<id>`, and
        /// a session outlives a dropped connection so a reconnecting client re-attaches to a still-
        /// running run. Bind loopback/internal only: the agent authenticates no caller — it trusts the
        /// front door. Pair with `--session-dir` so sessions survive a process restart. Absent ⇒ stdio.
        #[usage(long, env = "AI_AGENT_LISTEN")]
        listen: Option<std::net::SocketAddr>,
        /// Also (or instead) offer the control protocol over a Unix-domain socket at this path — a
        /// same-VM client gets kernel-enforced local authz via the socket's filesystem permissions,
        /// which loopback TCP does not provide. Bound on the *same* supervisor as `--listen`, so a
        /// session created over either transport is reachable over the other by its `?session_id=`.
        /// Only on unix targets. Pair with `--session-dir` for restart durability.
        #[usage(long, env = "AI_AGENT_LISTEN_UDS")]
        listen_uds: Option<std::path::PathBuf>,
        /// Octal permission mode to `chmod` the `--listen-uds` socket to after binding (e.g. `0o660`
        /// or `660` for a shared group). Default `0o600` (owner-only). Ignored without `--listen-uds`.
        #[usage(long, env = "AI_AGENT_LISTEN_UDS_MODE")]
        listen_uds_mode: Option<String>,
        /// Daemon mode only: reap a session that has had no attached connection for this many seconds
        /// and isn't mid-run — dropping it so it persists and exits, exactly like a graceful shutdown
        /// does per-session. Nothing is lost: reconnecting to a reaped id respawns it and replays from
        /// disk. Absent ⇒ 3600 (one hour) — long enough that a client can drop its socket and re-attach
        /// to a still-running session, finite so an unattended daemon's threads and gateway pools don't
        /// accumulate forever. Pass `0` to disable reaping entirely (every session then lives until the
        /// daemon stops). Ignored without `--listen`/`--listen-uds`.
        #[usage(long, env = "AI_AGENT_SESSION_IDLE_TIMEOUT")]
        session_idle_timeout: Option<u64>,
        /// How the daemon pools its upstream (agent→gateway) connections across sessions: `off` (the
        /// default — each session opens its own pool, as before), `auto` (one shared client, HTTP/1.1
        /// pooling now, h2 if the hop later gains ALPN), or `h2c` (one shared HTTP/2-cleartext client
        /// multiplexing all sessions over ~one connection). `h2c` **requires** an h2c-capable gateway —
        /// against an h1-only gateway every request fails — so it stays opt-in. Only meaningful with
        /// `--listen`/`--listen-uds`; ignored on the stdio path.
        #[usage(long, env = "AI_AGENT_UPSTREAM_HTTP2", default = "off")]
        upstream_http2: serve::UpstreamHttp2,
        /// Address this exact session: reattach to it if it already exists, or create it under exactly
        /// this id if it doesn't. Gives a caller a known, predictable name to route on rather than
        /// parsing an id back out of `get_state`/the startup `{"kind":"session", id, …}` banner.
        ///
        /// This is the right flag for a supervised (systemd, container) `serve`: it's deterministic and
        /// idempotent, so a restart lands back on the same conversation, where `--continue`'s "most
        /// recent for this cwd" silently depends on whatever else touched the directory meanwhile. It is
        /// also what makes `serve` multi-tenant — distinct ids are distinct sessions even in a shared
        /// `--session-dir`, which is exactly what the daemon's own `?session_id=` routing relies on.
        /// Outranks `--continue`. Matches `run`'s identical flag/contract (`main.rs::Run::session_id`).
        #[usage(long)]
        session_id: Option<String>,
        /// Reattach to the most recent session for the current directory instead of starting a fresh
        /// one, creating one if this is the first `serve` here. The only flag that reattaches
        /// implicitly: without it (and without `--session-id`/`--session-file`, both of which name a
        /// session outright) each launch starts its own session, so two servers sharing a directory
        /// don't silently drive the same on-disk transcript. Matches `run`'s identical flag.
        #[usage(long, short = 'c')]
        r#continue: bool,
        /// Skip persistence entirely, even without `--session-file`/`--session-dir`. Without this,
        /// `serve` defaults to `~/.claude/sessions/<encoded-cwd>/` rather than silently running
        /// in-memory-only — pass this for the rare case that's genuinely what you want (e.g. a
        /// short-lived test harness).
        #[usage(long)]
        no_session_persistence: bool,
        /// Persistent-memory backend DSN. Absent ⇒ the stored `default_memory_backend` setting, else a
        /// per-project local-file store. A bare path or `file://` names a directory; `redis://`/
        /// `postgres://` select a networked backend (recognized, not yet implemented).
        #[usage(long, env = "AI_AGENT_MEMORY_URL")]
        memory: Option<String>,
        /// Disable persistent memory entirely: don't register the `memory` tool or inject the index.
        #[usage(long)]
        no_memory: bool,
        /// Disable only the per-session `/session` working-memory mount, keeping durable `/memories`. On
        /// by default when memory is enabled.
        #[usage(long)]
        no_session_memory: bool,
        /// Opt-in cap on model turns per prompt before bailing with an error (default: unbounded).
        #[usage(long)]
        max_steps: Option<u32>,
        /// Replace the built-in base system prompt entirely.
        #[usage(long, env = "AI_AGENT_SYSTEM_PROMPT")]
        system_prompt: Option<String>,
        /// Append extra instructions after the base system prompt. Repeatable — `run`'s identical flag;
        /// each occurrence is joined with the others by a blank line, in the order given.
        #[usage(long, env = "AI_AGENT_APPEND_SYSTEM_PROMPT")]
        append_system_prompt: Vec<String>,
        /// Do not discover/inject AGENTS.md / CLAUDE.md project-instruction files. `-nc` matches pi's
        /// own `--no-context-files`/`-nc`.
        #[usage(long)]
        no_context_files: bool,
        /// Model context window (tokens); the loop summarizes older turns to stay below it. Defaults
        /// to the model's own capability-table window (see `agent_core::models::capabilities`) — only
        /// pass this to pin a fixed budget that survives a `set_model` switch to a different model.
        #[usage(long, env = "AI_AGENT_CONTEXT_WINDOW")]
        context_window: Option<u32>,
        /// Per-turn output token ceiling. Defaults to the model's own capability-table `max_output`
        /// (see `agent_core::models::capabilities`), floored at a sane minimum — only pass this to
        /// override that, e.g. capping generation length or lifting it past the model-derived default.
        #[usage(long, env = "AI_AGENT_MAX_TOKENS")]
        max_tokens: Option<u32>,
        /// Use the 1-hour prompt-cache TTL (vs 5 minutes); helps when turns are spaced out.
        #[usage(long)]
        cache_long: bool,
        /// Enable extended thinking with this token budget (must be below the per-turn max tokens). A
        /// raw token count, not pi's own `--thinking <level>` (off/minimal/low/medium/high/xhigh) — see
        /// `--reasoning-effort` for that portable level instead.
        #[usage(long)]
        thinking: Option<u32>,
        /// Reasoning effort for models driven by an effort level rather than a token budget (OpenAI
        /// reasoning models via `reasoning_effort`; Anthropic adaptive-thinking models via
        /// `output_config.effort`). One of off/minimal/low/medium/high/xhigh — see `--thinking` for a
        /// raw token-budget override instead. Ignored by models that take neither shape. Falls back to
        /// `AI_AGENT_REASONING_EFFORT`, then the stored `agent settings --default-reasoning-effort`
        /// default, before finally leaving it unset. Task 2 (pi-parity fix, pass 19): `off` is now
        /// accepted too, explicitly disabling reasoning. `run`'s identical flag.
        #[usage(long, env = "AI_AGENT_REASONING_EFFORT")]
        reasoning_effort: Option<agent_core::ThinkingLevel>,
        /// How much of the mid-run *steer* lane a single drain point consumes per turn boundary
        /// (`agent_core::QueueMode`) — `one_at_a_time` (the default, matching pi) injects only the oldest
        /// queued message per drain, leaving the rest queued for the next one; `all` folds everything
        /// queued into a single injection (this crate's original behavior). Falls back to the persisted
        /// setting `serve`'s own `set_steering_mode` RPC command maintains
        /// (`settings::Settings::steering_mode`), before finally defaulting to `one_at_a_time`. `run`'s
        /// identical flag.
        #[usage(long, env = "AI_AGENT_STEERING_MODE")]
        steering_mode: Option<agent_core::QueueMode>,
        /// Same idea as `--steering-mode`, for the follow-up lane drained at a stop boundary (plus any
        /// stranded steer messages swept in there) — matches pi's own separate `followUpMode`. `run`'s
        /// identical flag.
        #[usage(long, env = "AI_AGENT_FOLLOW_UP_MODE")]
        follow_up_mode: Option<agent_core::QueueMode>,
        /// Sampling temperature. Omitted (leaving the provider default) unless set. Silently ignored by
        /// Anthropic while `--thinking` is set (Anthropic forbids the two together). `run`'s identical
        /// flag.
        #[usage(long)]
        temperature: Option<f64>,
        /// Trust `cwd` for this run only, so a project-local `.claude/SYSTEM.md` is honored even if
        /// `cwd` isn't in the persisted allowlist (`agent trust <path>`). A session-scoped override,
        /// not a permanent grant — see `agent trust` to record one. `-a` matches pi's own
        /// `--approve`/`-a` (same "trust this project" meaning, different flag name here).
        #[usage(short = 'a', long)]
        trust_project: bool,
        /// Force `cwd` *untrusted* for this session only, overriding both `--trust-project` and the
        /// persisted allowlist (`agent trust <path>`) — e.g. to test untrusted behavior against a
        /// directory that's otherwise permanently trusted. Wins over `--trust-project` if both are
        /// somehow given. `-na` matches pi's own `--no-approve`/`-na`.
        #[usage(long)]
        force_untrusted: bool,
        /// Compaction headroom (tokens) reserved below the context window before it fires. Defaults to
        /// `CompactionConfig::default()`'s 16,384.
        #[usage(long, env = "AI_AGENT_COMPACTION_RESERVE_TOKENS")]
        compaction_reserve_tokens: Option<u32>,
        /// Roughly how many tokens of recent conversation compaction keeps verbatim. Defaults to
        /// `CompactionConfig::default()`'s 20,000.
        #[usage(long, env = "AI_AGENT_COMPACTION_KEEP_RECENT_TOKENS")]
        compaction_keep_recent_tokens: Option<u32>,
        /// Token budget reserved below the context window when summarizing an abandoned tree branch —
        /// independent of `--compaction-reserve-tokens`. `run`'s identical flag (Task #31, pi-parity
        /// feature).
        #[usage(long, env = "AI_AGENT_BRANCH_SUMMARY_RESERVE_TOKENS")]
        branch_summary_reserve_tokens: Option<u32>,
        /// Disable automatic (threshold-triggered) compaction entirely — `run`'s identical flag. When
        /// absent (and `AI_AGENT_NO_COMPACTION` unset), falls back to the persisted `agent settings`
        /// `compaction_enabled` override before finally defaulting to enabled — see
        /// `serve::ServeConfig::no_compaction`'s doc comment.
        #[usage(long, env = "AI_AGENT_NO_COMPACTION")]
        no_compaction: bool,
        /// How many times to retry a gateway request that fails before the first response byte
        /// arrives. Defaults to 3.
        #[usage(long, env = "AI_AGENT_RETRY_MAX_RETRIES")]
        retry_max_retries: Option<u32>,
        /// Base of the exponential backoff between those retries, in milliseconds. Defaults to 250.
        #[usage(long, env = "AI_AGENT_RETRY_BASE_DELAY_MS")]
        retry_base_delay_ms: Option<u64>,
        /// Ceiling on that exponential backoff, in milliseconds. `run`'s identical flag (Task #30,
        /// pi-parity feature).
        #[usage(long, env = "AI_AGENT_RETRY_MAX_BACKOFF_MS")]
        retry_max_backoff_ms: Option<u64>,
        /// Idle-read timeout between response chunks on the gateway HTTP client, in milliseconds —
        /// overrides `agent_core::client::GatewayClient`'s built-in default. `run`'s identical flag
        /// (Task #38, pi-parity fix: `serve` previously had no equivalent at all, so
        /// `--idle-timeout-ms`/`AI_AGENT_IDLE_TIMEOUT_MS`/the persisted `default_provider_timeout_ms`
        /// setting had no effect on a `serve` process).
        #[usage(long, env = "AI_AGENT_IDLE_TIMEOUT_MS")]
        idle_timeout_ms: Option<u64>,
        /// Force every image down the same downgrade-to-text-placeholder path a vision-incapable model
        /// already gets, regardless of the active model's real `supports_vision` capability. Falls back
        /// to the persisted `agent settings --block-images` default when not explicitly given. `run`'s
        /// identical flag (Task #34, pi-parity fix: `serve` previously had no equivalent at all).
        #[usage(long, env = "AI_AGENT_BLOCK_IMAGES")]
        block_images: bool,
        /// Force `--block-images` off for this invocation, even when a persisted `agent settings
        /// --block-images` default is `true` (pass 20, pi-parity fix). `run`'s identical flag — see its
        /// own doc comment.
        #[usage(long, env = "AI_AGENT_NO_BLOCK_IMAGES")]
        no_block_images: bool,
        /// Skip `read`'s resize/downscale path for an oversized image entirely, shipping its
        /// normalized (format-converted, if needed) bytes as-is regardless of size or pixel dimensions.
        /// Falls back to the persisted `agent settings --image-auto-resize` default when not explicitly
        /// given. `run`'s identical flag (Task #34, pi-parity fix: `serve` previously always hardcoded
        /// image auto-resize on, with no way to turn it off).
        #[usage(long, env = "AI_AGENT_NO_IMAGE_AUTO_RESIZE")]
        no_image_auto_resize: bool,
        /// Default `bash` command timeout (ms) when the model omits `timeout_ms`. Defaults to 1,800,000
        /// (30 minutes) — see `tools::bash`'s doc comment for why this deliberately deviates from the
        /// reference agent's no-default.
        #[usage(long, env = "AI_AGENT_BASH_TIMEOUT_MS")]
        bash_timeout_ms: Option<u64>,
        /// Run `bash` commands through this shell instead of the auto-resolved one (`/bin/bash`, else
        /// `bash` on `$PATH`, else `sh`) — for a non-standard environment (Cygwin, a container without
        /// `/bin/bash` at the expected path, a hardened/audited shell wrapper) where auto-detection
        /// would pick the wrong binary. Matches pi's own `shellPath` setting. Checked to exist once
        /// here, at startup — a bad path fails the process immediately instead of surfacing as a
        /// confusing spawn error on the first `bash` call.
        #[usage(long, env = "AI_AGENT_BASH_SHELL_PATH")]
        bash_shell_path: Option<String>,
        /// Prepend this line to every `bash` command, in the same shell invocation (e.g. sourcing a
        /// project's env setup, activating a venv). Matches pi's own `shellCommandPrefix` setting.
        /// Fixed for the process, like `--bash-shell-path`; survives `set_model`/`set_thinking` rebuilds.
        #[usage(long, env = "AI_AGENT_BASH_COMMAND_PREFIX")]
        bash_command_prefix: Option<String>,
        /// Let the `web` tool reach loopback/private/link-local addresses. Off by default: the tool
        /// refuses them to prevent SSRF (it fetches URLs the model chose). `serve`'s identical flag.
        #[usage(long, env = "AI_AGENT_WEB_ALLOW_PRIVATE")]
        web_allow_private: bool,
        /// A hostname the `web` tool may reach even with private egress off (repeatable) — an internal
        /// service, or `127.0.0.1` for local testing. `serve`'s identical flag.
        #[usage(long, env = "AI_AGENT_WEB_ALLOW_HOST")]
        web_allow_host: Vec<String>,
        /// The `web` tool's per-request timeout (ms). Default 30,000. `serve`'s identical flag.
        #[usage(long, env = "AI_AGENT_WEB_TIMEOUT_MS")]
        web_timeout_ms: Option<u64>,
        /// Point every session's tools at a remote exec endpoint by default. A multi-tenant server
        /// leaves this unset and uses the `set_exec_endpoint` command per session instead.
        #[usage(long, env = "AI_AGENT_EXEC_URL", conflicts = "exec_cmd")]
        exec_url: Option<String>,
        /// A header sent with every exec request, `Name: value`. Repeatable.
        #[usage(long, env = "AI_AGENT_EXEC_HEADER")]
        exec_header: Vec<String>,
        /// An argv template for targets with no HTTP surface, e.g. `ssh host -- {}`.
        #[usage(long, env = "AI_AGENT_EXEC_CMD", conflicts = "exec_url")]
        exec_cmd: Option<String>,
        /// Restrict the tool set to exactly these names (comma-separated), dropping everything else.
        /// Fixed for the process, like `--system-prompt`; survives `set_model`/`set_thinking` rebuilds.
        /// `-t` matches pi's own `--tools`/`-t`.
        #[usage(short = 't', long, env = "AI_AGENT_TOOLS", delimiter = ',')]
        tools: Option<Vec<String>>,
        /// Drop these tools (comma-separated) from the default set — e.g. `--exclude-tools bash,write`
        /// for a read-only reviewer that can't run shell commands or mutate files. `-xt` matches pi's
        /// own `--exclude-tools`/`-xt`.
        #[usage(long, env = "AI_AGENT_EXCLUDE_TOOLS", delimiter = ',')]
        exclude_tools: Option<Vec<String>>,
        /// Register no tools at all — a pure-conversation session. Wins over `--tools`/`--exclude-tools`.
        /// `-nt` matches pi's own `--no-tools`/`-nt`.
        #[usage(long)]
        no_tools: bool,
        /// Force every batch of tool calls in a turn to run one at a time instead of the default
        /// bounded-concurrent dispatch (`agent_core::Agent::with_sequential_tools`). `run`'s identical
        /// flag.
        #[usage(long)]
        sequential_tools: bool,
        /// Block every call to this tool (comma-separated, repeatable), even though it stays visible
        /// and registered — unlike `--exclude-tools`, a denied call still surfaces to the model as a
        /// normal error `tool_result` explaining it was blocked by policy, rather than the tool being
        /// invisible outright. `run`'s identical flag.
        #[usage(long, env = "AI_AGENT_DENY_TOOL", delimiter = ',')]
        deny_tool: Vec<String>,
        /// Block a `bash` call whenever its command contains this substring, case-insensitively
        /// (comma-separated, repeatable). `run`'s identical flag.
        #[usage(long, env = "AI_AGENT_DENY_BASH_PATTERN", delimiter = ',')]
        deny_bash_pattern: Vec<String>,
        /// Block a `write`/`edit` call whenever its `path` argument matches this glob (comma-separated,
        /// repeatable). `run`'s identical flag.
        #[usage(long, env = "AI_AGENT_DENY_PATH", delimiter = ',')]
        deny_path: Vec<String>,
        /// Require a human to approve a tool call before it runs: `off` (default), `writes`
        /// (`write`/`edit`), `all` (everything except the read-only tools), or `tools:<name>,<name>`.
        /// `serve` broadcasts an `approval_request` frame to every attached client and blocks the call
        /// until one of them answers with an `approve` command. An unrecognized value is an error, not a
        /// silently empty gate.
        ///
        /// Fails closed: a timeout, an abort, or no attached client all deny the call (the model gets an
        /// error `tool_result` and the run continues). The static `--deny-*` lists still win first, with
        /// no round trip. `run` has no equivalent — it has no client to ask.
        #[usage(long, env = "AI_AGENT_APPROVE", default = "off")]
        approve: String,
        /// Seconds an unanswered approval request waits before it is denied. `0` waits forever, which is
        /// only safe with a reliably-attached client: a question nobody answers pins the session until an
        /// `abort`.
        #[usage(long, env = "AI_AGENT_APPROVAL_TIMEOUT", default = "300")]
        approval_timeout: u64,
        /// Restrict `cycle_model`'s candidate list to exactly these ids, in this order
        /// (comma-separated) — e.g. `--models claude-opus-4-8,claude-sonnet-4-5,gpt-5`.
        /// `set_model`/`get_available_models` are unaffected; empty/absent cycles the full known-model
        /// list instead.
        #[usage(long, env = "AI_AGENT_MODELS", delimiter = ',')]
        models: Option<Vec<String>>,
        /// Disable *standard-root* skills discovery/loading (`~/.claude/skills`, `<cwd>/.claude/skills`)
        /// — no `<available_skills>` listing in the system prompt from either, and a `/skill:name`
        /// invocation (however it reaches the session — `prompt`, `steer`, `follow_up`) is sent through
        /// unexpanded unless it resolves against a `--skill` path instead. An explicit `--skill <path>`
        /// is still honored even so, matching `run`'s identical flag and pi's own `--no-skills`. Applies
        /// on every `reload` too. `-ns` matches pi's own `--no-skills`/`-ns`.
        #[usage(long)]
        no_skills: bool,
        /// Disable *standard-root* prompt-template discovery/loading (`~/.claude/prompts`,
        /// `<cwd>/.claude/prompts`) — a `/name` invocation is sent through unexpanded unless it resolves
        /// against a `--prompt-template` path instead. An explicit `--prompt-template <path>` is still
        /// honored even so, matching `run`'s identical flag and pi's own `--no-prompt-templates`. Applies
        /// on every `reload` too. `-np` matches pi's own `--no-prompt-templates`/`-np`.
        #[usage(long)]
        no_prompt_templates: bool,
        /// Discover skills from this directory too, in addition to the two standard roots (repeatable,
        /// or comma-separated via `AI_AGENT_SKILL_PATH`). Matches pi's own `--skill <path>` and `run`'s
        /// identical flag; applies on every `reload` too.
        #[usage(
            long = "skill",
            env = "AI_AGENT_SKILL_PATH",
            delimiter = ',',
            value_name = "PATH"
        )]
        extra_skill_paths: Vec<String>,
        /// Discover prompt templates from this directory too, in addition to the two standard roots
        /// (repeatable, or comma-separated via `AI_AGENT_PROMPT_TEMPLATE_PATH`). Matches pi's own
        /// `--prompt-template <path>` and `run`'s identical flag; applies on every `reload` too.
        #[usage(
            long = "prompt-template",
            env = "AI_AGENT_PROMPT_TEMPLATE_PATH",
            delimiter = ',',
            value_name = "PATH"
        )]
        extra_prompt_template_paths: Vec<String>,
        /// Set the initial session's name up front, before the first turn even starts — a whitespace-only
        /// value is rejected, matching pi's own `--name`. Unlike pi (which renames unconditionally on
        /// every invocation, last-write-wins), this only ever takes effect on a genuinely fresh session
        /// — see the fresh-only check in `run_task`/`serve` for why: a deliberate deviation, not an
        /// oversight. The RPC `set_session_name` command covers renaming an existing session afterward.
        #[usage(short = 'n', long)]
        name: Option<String>,
        /// An extra guideline bullet appended to the default system prompt's `Guidelines:` section
        /// (repeatable) — pi's own `promptGuidelines`; `run`'s identical flag. Has no effect when
        /// `--system-prompt` supplies a full custom prompt (matches pi: a custom prompt replaces the
        /// whole guidelines mechanism, not just extends it).
        #[usage(long = "prompt-guideline", value_name = "TEXT")]
        prompt_guidelines: Vec<String>,
    },
    /// List the tools the agent advertises to the model.
    Tools,
    /// List a small, non-exhaustive set of model ids the capabilities table recognizes (a convenience
    /// hint for a model picker — the gateway forwards any id verbatim, so `--model`/`set_model` accept
    /// ids outside this list too).
    ListModels {
        /// Only print rows whose model id contains this substring, case-insensitively — a convenience
        /// filter for a long list, matching pi's own `--list-models <search>`. Absent: print every row.
        search: Option<String>,
    },
    /// Record `path` (default: the current directory) in the persisted project-trust allowlist
    /// (`~/.claude/trusted-projects.json`), so its `.claude/SYSTEM.md` is honored on future runs
    /// without needing `--trust-project` every time. Idempotent — trusting an already-trusted path is
    /// a no-op.
    Trust {
        /// The project directory to trust. Defaults to the current directory.
        path: Option<String>,
    },
    /// Record `path` (default: the current directory) as explicitly *untrusted*, overriding any
    /// trust it would otherwise inherit from a trusted ancestor directory. Idempotent.
    Untrust {
        /// The project directory to untrust. Defaults to the current directory.
        path: Option<String>,
    },
    /// Remove `path`'s (default: the current directory) own trust/untrust entry, without recording a
    /// new one — unlike `trust`/`untrust`, which always leave `path` pinned to its own explicit
    /// grant or denial. `path` reverts to inheriting whatever its nearest trusted/untrusted ancestor
    /// decides (or unknown, if none does). Idempotent.
    ClearTrust {
        /// The project directory to clear. Defaults to the current directory.
        path: Option<String>,
    },
    /// Report `path`'s (default: the current directory) tri-state trust decision — `trusted`,
    /// `untrusted`, or `unknown` — walking up through its ancestors for the first explicit entry
    /// (`TrustStore::lookup`), the same resolution `trust`/`untrust`/`clear-trust` use internally but
    /// previously had no read-only way to actually query.
    TrustStatus {
        /// The project directory to query. Defaults to the current directory.
        path: Option<String>,
    },
    /// Log into a subscription provider (`anthropic`, `github-copilot`, or `openai-codex`) instead of
    /// a metered API key — an OAuth PKCE or device-code flow, printing progress to stderr and
    /// blocking until it completes, is cancelled (Ctrl-C), or times out. Overwrites any existing
    /// stored credential for `provider` on success only. See `beyond_ai_agent::oauth`/`auth_store`.
    Login {
        /// `anthropic`, `github-copilot`, or `openai-codex`.
        provider: String,
    },
    /// Remove `provider`'s stored subscription credential, if any. Idempotent.
    Logout {
        /// `anthropic`, `github-copilot`, or `openai-codex`.
        provider: String,
    },
    /// Report stored subscription-login status — `logged_in`/`logged_out`/`needs_reauth` — for
    /// `provider`, or every known provider when omitted. A pure read of the local store; never makes
    /// a network call (so a `needs_reauth` credential still shows as configured until an actual
    /// request or `agent login` re-establishes it).
    AuthStatus {
        /// `anthropic`, `github-copilot`, or `openai-codex`. Omit to report every known provider.
        provider: Option<String>,
    },
    /// Interactively log into an MCP server's own OAuth 2.1 authorization flow (protected-resource
    /// metadata discovery, dynamic client registration, PKCE) — for an `mcp_servers` entry using the
    /// `http` transport that requires it. `name` must already be configured (global or, if trusted,
    /// project `settings.json` — see `settings::Settings::mcp_servers`); a `stdio` server has no login
    /// of its own (use its `env` for a static credential instead). Unlike `agent login`'s fixed
    /// per-provider callback ports, this registers its own client against a freshly chosen local port
    /// each run — there's no pre-registered redirect URI to reuse. Overwrites any existing stored
    /// login for `name` on success only, mirroring `agent login`. See `tools::mcp`'s module doc
    /// comment and `mcp_auth_store.rs` for how the resulting credential is used/persisted.
    McpLogin {
        /// The `mcp_servers` entry's own `name`, as configured in `settings.json`.
        name: String,
    },
    /// Remove `name`'s stored MCP OAuth login, if any. Idempotent.
    McpLogout {
        /// The `mcp_servers` entry's own `name`.
        name: String,
    },
    /// View or update persisted defaults for `run`/`serve` flags — model, gateway URL, session
    /// directory — stored at `~/.claude/settings.json` (see `settings::SettingsStore`) and consulted as
    /// the last fallback after an explicit `--flag`/environment variable, before this crate's own
    /// built-in default. With no flags, prints the currently stored values. Mirrors `agent trust`/
    /// `agent untrust` managing the trust store the same out-of-band way.
    Settings {
        /// Set the stored default model (used when neither `--model` nor `AI_AGENT_MODEL` is given).
        #[usage(long)]
        model: Option<String>,
        /// Clear the stored default model.
        #[usage(long)]
        clear_model: bool,
        /// Set the stored default gateway URL (used when neither `--gateway-url` nor `AI_GATEWAY_URL`
        /// is given).
        #[usage(long)]
        gateway_url: Option<String>,
        /// Clear the stored default gateway URL.
        #[usage(long)]
        clear_gateway_url: bool,
        /// Set the stored default session directory (used when neither `--session-dir` nor
        /// `AI_AGENT_SESSION_DIR` is given).
        #[usage(long)]
        session_dir: Option<String>,
        /// Clear the stored default session directory.
        #[usage(long)]
        clear_session_dir: bool,
        /// Set the stored default project-trust policy — `always`/`never`/`ask` (used when neither
        /// `--trust-project` nor `--force-untrusted` is given; see `settings::TrustPolicy`).
        #[usage(long, value_enum)]
        default_project_trust: Option<beyond_ai_agent::settings::TrustPolicy>,
        /// Clear the stored default project-trust policy.
        #[usage(long)]
        clear_default_project_trust: bool,
        /// Set the stored default reasoning effort — one of off/minimal/low/medium/high/xhigh (used when
        /// neither `--reasoning-effort` nor `AI_AGENT_REASONING_EFFORT` is given; Task 2, pi-parity fix,
        /// pass 19: `off` is now accepted here too, explicitly persisting "no reasoning" as the default
        /// rather than a parse error). Fix 2 (pi-parity gap): previously the only numeric/string CLI
        /// tunable with no persisted-default surface at all, unlike `--model`/`--gateway-url`/
        /// `--session-dir` above.
        #[usage(long)]
        default_reasoning_effort: Option<agent_core::ThinkingLevel>,
        /// Clear the stored default reasoning effort.
        #[usage(long)]
        clear_default_reasoning_effort: bool,
        /// Set the stored default for `--block-images` (used when the flag isn't explicitly passed on a
        /// given `run` invocation) — Task #26 (pi-parity feature). `true` behaves as if `--block-images`
        /// were always given.
        #[usage(long)]
        block_images: Option<bool>,
        /// Clear the stored default for `--block-images`, reverting to `run`'s own built-in default
        /// (images allowed).
        #[usage(long)]
        clear_block_images: bool,
        /// Set the stored default for image auto-resize (used when `--no-image-auto-resize` isn't
        /// explicitly passed on a given `run` invocation) — Task #4 (pi-parity feature). `false`
        /// behaves as if `--no-image-auto-resize` were always given.
        #[usage(long)]
        image_auto_resize: Option<bool>,
        /// Clear the stored default for image auto-resize, reverting to `run`'s own built-in default
        /// (resize enabled).
        #[usage(long)]
        clear_image_auto_resize: bool,
        /// Set a persisted thinking-token-budget override for one reasoning-effort level —
        /// `<effort>=<tokens>` (e.g. `high=40000`), one of minimal/low/medium/high/xhigh — Task #36
        /// (pi-parity feature). Repeatable; consulted by `run` in place of the built-in
        /// effort-to-budget ladder wherever a turn's thinking budget is derived from
        /// `--reasoning-effort` (see `agent_core::models::budget_for_effort_with_override`).
        #[usage(long = "thinking-budget", value_name = "EFFORT=TOKENS")]
        thinking_budget: Vec<String>,
        /// Clear a persisted thinking-token-budget override for this reasoning-effort level.
        /// Repeatable.
        #[usage(long = "clear-thinking-budget", value_name = "EFFORT")]
        clear_thinking_budget: Vec<String>,
        /// Set the stored default `--bash-shell-path` (used when neither `--bash-shell-path` nor
        /// `AI_AGENT_BASH_SHELL_PATH` is given, for both `run` and `serve`) — Round 3 (pi-parity fix).
        #[usage(long)]
        default_bash_shell_path: Option<String>,
        /// Clear the stored default `--bash-shell-path`.
        #[usage(long)]
        clear_default_bash_shell_path: bool,
        /// Set the stored default `--bash-command-prefix` (used when neither the flag nor
        /// `AI_AGENT_BASH_COMMAND_PREFIX` is given) — Round 3 (pi-parity fix).
        #[usage(long)]
        default_bash_command_prefix: Option<String>,
        /// Clear the stored default `--bash-command-prefix`.
        #[usage(long)]
        clear_default_bash_command_prefix: bool,
        /// Set the stored default compaction reserve-token override (used when neither
        /// `--compaction-reserve-tokens` nor `AI_AGENT_COMPACTION_RESERVE_TOKENS` is given) — Round 3
        /// (pi-parity fix).
        #[usage(long)]
        default_compaction_reserve_tokens: Option<u32>,
        /// Clear the stored default compaction reserve-token override.
        #[usage(long)]
        clear_default_compaction_reserve_tokens: bool,
        /// Set the stored default compaction keep-recent-token override (used when neither
        /// `--compaction-keep-recent-tokens` nor `AI_AGENT_COMPACTION_KEEP_RECENT_TOKENS` is given) —
        /// Round 3 (pi-parity fix).
        #[usage(long)]
        default_compaction_keep_recent_tokens: Option<u32>,
        /// Clear the stored default compaction keep-recent-token override.
        #[usage(long)]
        clear_default_compaction_keep_recent_tokens: bool,
        /// Set the stored default retry max-retries override (used when neither `--retry-max-retries`
        /// nor `AI_AGENT_RETRY_MAX_RETRIES` is given) — Round 3 (pi-parity fix).
        #[usage(long)]
        default_retry_max_retries: Option<u32>,
        /// Clear the stored default retry max-retries override.
        #[usage(long)]
        clear_default_retry_max_retries: bool,
        /// Set the stored default retry base-delay override, in milliseconds (used when neither
        /// `--retry-base-delay-ms` nor `AI_AGENT_RETRY_BASE_DELAY_MS` is given) — Round 3 (pi-parity
        /// fix).
        #[usage(long)]
        default_retry_base_delay_ms: Option<u64>,
        /// Clear the stored default retry base-delay override.
        #[usage(long)]
        clear_default_retry_base_delay_ms: bool,
        /// Set the stored default provider (idle-read) timeout override, in milliseconds (used when
        /// neither `--idle-timeout-ms` nor `AI_AGENT_IDLE_TIMEOUT_MS` is given, for both `run` and
        /// `serve`) — Round 3 (pi-parity fix); Task #38 extended it to cover `serve` too.
        #[usage(long)]
        default_provider_timeout_ms: Option<u64>,
        /// Clear the stored default provider-timeout override.
        #[usage(long)]
        clear_default_provider_timeout_ms: bool,
        /// Set the stored default retry backoff-ceiling override, in milliseconds (used when neither
        /// `--retry-max-backoff-ms` nor `AI_AGENT_RETRY_MAX_BACKOFF_MS` is given) — Task #30 (pi-parity
        /// feature): the retry cluster's third knob, `agent_core::client::GatewayClient::
        /// with_max_backoff`, previously had no CLI flag or persisted override at all.
        #[usage(long)]
        default_retry_max_backoff_ms: Option<u64>,
        /// Clear the stored default retry backoff-ceiling override.
        #[usage(long)]
        clear_default_retry_max_backoff_ms: bool,
        /// Set the stored default `--models` scoping/cycling candidate list, comma-separated (used when
        /// neither `--models` nor `AI_AGENT_MODELS` is given; `serve`-only) — Round 3 (pi-parity fix).
        #[usage(long = "default-models", delimiter = ',')]
        default_models: Option<Vec<String>>,
        /// Clear the stored default `--models` list.
        #[usage(long)]
        clear_default_models: bool,
        /// Set the stored default extra skill-discovery paths, comma-separated (used when no `--skill
        /// <path>`/`AI_AGENT_SKILL_PATH` is given) — Round 3 (pi-parity fix).
        #[usage(long = "default-skill-paths", delimiter = ',')]
        default_skill_paths: Option<Vec<String>>,
        /// Clear the stored default extra skill-discovery paths.
        #[usage(long)]
        clear_default_skill_paths: bool,
        /// Set the stored default extra prompt-template-discovery paths, comma-separated (used when no
        /// `--prompt-template <path>`/`AI_AGENT_PROMPT_TEMPLATE_PATH` is given) — Round 3 (pi-parity
        /// fix).
        #[usage(long = "default-prompt-template-paths", delimiter = ',')]
        default_prompt_template_paths: Option<Vec<String>>,
        /// Clear the stored default extra prompt-template-discovery paths.
        #[usage(long)]
        clear_default_prompt_template_paths: bool,
        /// Set the stored default branch-summary reserve-token budget (used when neither
        /// `--branch-summary-reserve-tokens` nor `AI_AGENT_BRANCH_SUMMARY_RESERVE_TOKENS` is given) —
        /// Task #31 (pi-parity feature): `agent_core::Agent::with_branch_summary_reserve_tokens`
        /// previously had no caller at all, so a branch summary's reserve was always hard-tied to
        /// ordinary compaction's own `--compaction-reserve-tokens`, matching pi's independently
        /// configurable `branchSummary.reserveTokens` (default 16384).
        #[usage(long)]
        default_branch_summary_reserve_tokens: Option<u32>,
        /// Clear the stored default branch-summary reserve-token budget.
        #[usage(long)]
        clear_default_branch_summary_reserve_tokens: bool,
    },
    /// Render an existing session's `.jsonl` file as a self-contained HTML transcript and exit — pure
    /// offline rendering of what's already on disk, no gateway/key/model involved at all (unlike `run
    /// --export`, which exports only after a live run completes). The same rendering `serve`'s
    /// `export_html` RPC command and `run --export` use.
    Export {
        /// Path to the session's `.jsonl` file (as passed to `--session-file`, or one file inside a
        /// `--session-dir` tree).
        session: String,
        /// Output HTML path. Defaults to `session-<timestamp>.html` in the current directory.
        output: Option<String>,
    },
}

/// Whether `candidate` fuzzy-matches `query`, and if so a score for ranking (lower is a better match) —
/// `Command::ListModels`'s `--list-models <search>` fuzzy filter, porting pi's own `fuzzyMatch`
/// (`packages/tui/src/fuzzy.ts`): every character of `query` must appear in `candidate`, in order and
/// case-insensitively, but not necessarily adjacent — so "sn5" matches "claude-sonnet-4-5", which a
/// plain substring check never would. A consecutive run of matched characters, and a match starting
/// right at a word boundary (candidate index 0, or right after `-`/`_`/`.`/`/`/`:`/whitespace), both
/// score better (more negative); a gap between two matches and a later match position both score
/// slightly worse. `None` when `query` doesn't match at all (including as the alpha/digit-swapped
/// fallback below).
fn fuzzy_match(query: &str, candidate: &str) -> Option<f64> {
    fn match_subsequence(query: &str, candidate: &str) -> Option<f64> {
        if query.is_empty() {
            return Some(0.0);
        }
        let candidate_chars: Vec<char> = candidate.chars().collect();
        let query_chars: Vec<char> = query.chars().collect();
        if query_chars.len() > candidate_chars.len() {
            return None;
        }
        let mut query_index = 0usize;
        let mut score = 0.0f64;
        let mut last_match_index: i64 = -1;
        let mut consecutive: i64 = 0;
        for (i, &c) in candidate_chars.iter().enumerate() {
            if query_index >= query_chars.len() {
                break;
            }
            if c != query_chars[query_index] {
                continue;
            }
            let i64_i = i as i64;
            let is_word_boundary =
                i == 0 || matches!(candidate_chars[i - 1], ' ' | '-' | '_' | '.' | '/' | ':');
            if last_match_index == i64_i - 1 {
                consecutive += 1;
                score -= (consecutive * 5) as f64;
            } else {
                consecutive = 0;
                if last_match_index >= 0 {
                    score += ((i64_i - last_match_index - 1) * 2) as f64;
                }
            }
            if is_word_boundary {
                score -= 10.0;
            }
            score += i as f64 * 0.1;
            last_match_index = i64_i;
            query_index += 1;
        }
        if query_index < query_chars.len() {
            return None;
        }
        if query == candidate {
            score -= 100.0;
        }
        Some(score)
    }

    let query_lower = query.to_ascii_lowercase();
    let candidate_lower = candidate.to_ascii_lowercase();
    if let Some(score) = match_subsequence(&query_lower, &candidate_lower) {
        return Some(score);
    }
    // A query typed in the opposite letter/digit order (e.g. "5sonnet" for "sonnet5") — pi's own
    // regex-based fallback, tried only once the direct match fails outright, with a flat penalty for
    // having needed it.
    swap_alpha_digit(&query_lower)
        .and_then(|swapped| match_subsequence(&swapped, &candidate_lower))
        .map(|score| score + 5.0)
}

/// Swap a query that's entirely `letters` followed by `digits` (or vice versa) to the other order — the
/// two shapes [`fuzzy_match`]'s fallback tries, matching pi's own `^[a-z]+[0-9]+$`/`^[0-9]+[a-z]+$`
/// regex pair. `None` for anything else (mixed/interleaved characters, or already all one class).
fn swap_alpha_digit(query: &str) -> Option<String> {
    let chars: Vec<char> = query.chars().collect();
    if let Some(split) = chars.iter().position(|c| !c.is_ascii_lowercase())
        && split > 0
        && chars[split..].iter().all(char::is_ascii_digit)
    {
        let (letters, digits) = (&chars[..split], &chars[split..]);
        return Some(digits.iter().chain(letters).collect());
    }
    if let Some(split) = chars.iter().position(|c| !c.is_ascii_digit())
        && split > 0
        && chars[split..].iter().all(char::is_ascii_lowercase)
    {
        let (digits, letters) = (&chars[..split], &chars[split..]);
        return Some(letters.iter().chain(digits).collect());
    }
    None
}

/// Render a token count the way pi's own `--list-models` does (`formatTokenCount`,
/// `packages/coding-agent/src/cli/list-models.ts`) — Task #39 (pi-parity fix, cosmetic): `200000`/
/// `1000000` as `"200K"`/`"1M"` rather than a raw integer, with one decimal place only when the
/// abbreviated value isn't a whole number (`"1.5M"`, but plain `"2M"`). Below 1,000, the plain integer
/// is already as short as any abbreviation would be.
fn format_token_count(n: u32) -> String {
    let n = n as f64;
    if n >= 1_000_000.0 {
        let millions = n / 1_000_000.0;
        return if millions.fract() == 0.0 {
            format!("{millions:.0}M")
        } else {
            format!("{millions:.1}M")
        };
    }
    if n >= 1_000.0 {
        let thousands = n / 1_000.0;
        return if thousands.fract() == 0.0 {
            format!("{thousands:.0}K")
        } else {
            format!("{thousands:.1}K")
        };
    }
    format!("{n:.0}")
}

/// Rewrites the multi-character short-flag aliases pi's own hand-rolled CLI parser accepts
/// (`cli/args.ts`) to their long-flag equivalent before clap ever sees them. clap's own `short`
/// mechanism (used below for the single-character aliases, e.g. `-t`/`-a`) is exactly one ASCII
/// character, so a two-character form like `-nt` can't be expressed that way directly — pi's parser has
/// no such restriction (it's hand-rolled, not clap-based). An exact whole-token match only (mirrors
/// pi's own `arg === "-nt"` checks): never a prefix, so this can't misfire against an unrelated value
/// that merely starts with the same two characters, and never touches anything after `--` (clap's own
/// end-of-options marker) since that's the operator explicitly opting every remaining argument out of
/// flag parsing.
fn expand_short_aliases(args: Vec<String>) -> Vec<String> {
    let expand = |a: &str| -> Option<&'static str> {
        match a {
            "-nt" => Some("--no-tools"),
            "-xt" => Some("--exclude-tools"),
            "-ns" => Some("--no-skills"),
            "-np" => Some("--no-prompt-templates"),
            "-nc" => Some("--no-context-files"),
            "-na" => Some("--force-untrusted"),
            // Task #43: clap's auto-generated version flag only binds the capital `-V`; pi documents a
            // lowercase `-v` alias too (`cli/args.ts`). A one-character alias could in principle use
            // clap's own `short` mechanism directly (unlike the two-character aliases above, which
            // can't), but doing it here keeps every alias in this one table rather than splitting the
            // convention across two different mechanisms for no real benefit.
            "-v" => Some("--version"),
            _ => None,
        }
    };
    let mut past_end_of_options = false;
    args.into_iter()
        .map(|a| {
            if past_end_of_options {
                return a;
            }
            if a == "--" {
                past_end_of_options = true;
                return a;
            }
            expand(&a).map(str::to_string).unwrap_or(a)
        })
        .collect()
}

/// [`Cli::parse`], except a `--help`/`-h` triggered while `run --json` is also present renders to
/// stderr and exits 0 instead of clap's own default of stdout — matching pi's own `--mode json
/// --help`/`-p --help` behavior (`stdout-cleanliness.test.ts`). Plain `run --help` (no `--json`) and
/// top-level `--help`/`--version` are untouched: clap's stdout default is correct there (nothing is
/// consuming stdout as a data stream), and `run_binary_help_flag_prints_usage_to_stdout_with_empty_stderr`/
/// `run_binary_version_flag_prints_only_the_version_to_stdout` already pin that down. `--json` marks
/// `run`'s stdout as the NDJSON `AgentEvent` stream (see `run_turn_once`) — the same invariant
/// `serve`'s `#![deny(clippy::print_stdout)]` protects for its own protocol — but clap's `--help`
/// short-circuit fires from inside `Cli::parse()`, before any application code (and thus before that
/// lint's module boundary) ever runs, so it can't be caught statically; this is the runtime backstop.
///
/// `Cli::parse()` can't tell us this itself: on `--help`, clap returns an error *before* the `run`
/// subcommand's fields (including `json`) are ever populated, so there's no parsed `Cli::Run { json,
/// .. }` to inspect. Scanning the raw argv instead — subcommand `run` at position 1, `--json` and a
/// help flag present anywhere else — sidesteps that: it doesn't need parsing to have succeeded, and a
/// `--json`/`--help`/`-h` substring can only appear as those literal flags here, never as a task
/// message (an argument starting with `-` is consumed as a flag by clap, not a positional, unless
/// explicitly escaped with `--`). `args` here is already run through [`expand_short_aliases`], so the
/// argv-position/substring checks below see the expanded (long-flag) form too.
fn cli() -> Cli {
    let args = expand_short_aliases(std::env::args().collect());
    let run_json_help = args.get(1).map(String::as_str) == Some("run")
        && args.iter().any(|a| a == "--json")
        && args.iter().any(|a| a == "--help" || a == "-h");
    let argv: Vec<OsString> = args.into_iter().map(OsString::from).collect();
    let argv_refs: Vec<&OsStr> = argv.iter().map(|s| s.as_os_str()).collect();
    match Cli::try_parse_from(&argv_refs) {
        Ok(cli) => cli,
        Err(usage::Error::Help { cmd, long }) => {
            let text = Cli::render_help(cmd, long).unwrap_or_default();
            if run_json_help {
                eprint!("{text}");
            } else {
                print!("{text}");
            }
            std::process::exit(0);
        }
        Err(usage::Error::Version { long }) => {
            let spec = Cli::spec();
            let bin = spec.bin.unwrap_or(spec.name);
            let version = if long {
                spec.long_version.or(spec.version).unwrap_or_default()
            } else {
                spec.version.unwrap_or_default()
            };
            println!("{bin} {version}");
            std::process::exit(0);
        }
        Err(err) => {
            eprint!("{}", Cli::render_failure(&argv_refs, &err));
            std::process::exit(2);
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Always stderr, never stdout: `serve`'s NDJSON control protocol and `run`'s streamed output both
    // live on stdout, and a line-based client reading it can't tell a stray log line from a protocol
    // frame. `RUST_LOG=debug` (or any filter admitting a `warn!`/`info!` already present on a live
    // path — e.g. `session_store.rs`'s corrupt-line warning, `skills.rs`'s discovery warning) must
    // never corrupt that stream.
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    match cli().command {
        Command::Run {
            tasks,
            model,
            gateway_url,
            key,
            max_steps,
            max_tokens,
            cache_long,
            thinking,
            reasoning_effort,
            steering_mode,
            follow_up_mode,
            temperature,
            system_prompt,
            append_system_prompt,
            trust_project,
            force_untrusted,
            context_window,
            compaction_reserve_tokens,
            compaction_keep_recent_tokens,
            branch_summary_reserve_tokens,
            no_compaction,
            retry_max_retries,
            retry_base_delay_ms,
            retry_max_backoff_ms,
            no_retry,
            idle_timeout_ms,
            block_images,
            no_block_images,
            no_image_auto_resize,
            bash_timeout_ms,
            bash_shell_path,
            bash_command_prefix,
            web_allow_private,
            web_allow_host,
            web_timeout_ms,
            exec_url,
            exec_header,
            exec_cmd,
            tools,
            exclude_tools,
            no_tools,
            sequential_tools,
            deny_tool,
            deny_bash_pattern,
            deny_path,
            no_skills,
            no_prompt_templates,
            no_context_files,
            extra_skill_paths,
            extra_prompt_template_paths,
            name,
            prompt_guidelines,
            fork,
            session,
            session_id,
            r#continue,
            session_dir,
            no_session_persistence,
            memory,
            no_memory,
            no_session_memory,
            export,
            json,
            output_schema,
            output_description,
        } => {
            run_task(
                tasks,
                model,
                gateway_url,
                key,
                max_steps,
                max_tokens,
                cache_long,
                thinking,
                reasoning_effort,
                steering_mode,
                follow_up_mode,
                temperature,
                system_prompt,
                append_system_prompt,
                trust_project,
                force_untrusted,
                context_window,
                compaction_reserve_tokens,
                compaction_keep_recent_tokens,
                branch_summary_reserve_tokens,
                no_compaction,
                retry_max_retries,
                retry_base_delay_ms,
                retry_max_backoff_ms,
                no_retry,
                idle_timeout_ms,
                block_images,
                no_block_images,
                no_image_auto_resize,
                bash_timeout_ms,
                bash_shell_path,
                bash_command_prefix,
                web_allow_private,
                web_allow_host,
                web_timeout_ms,
                exec_url,
                exec_header,
                exec_cmd,
                tools,
                exclude_tools,
                no_tools,
                sequential_tools,
                deny_tool,
                deny_bash_pattern,
                deny_path,
                no_skills,
                no_prompt_templates,
                no_context_files,
                extra_skill_paths,
                extra_prompt_template_paths,
                name,
                prompt_guidelines,
                fork,
                session,
                session_id,
                r#continue,
                session_dir,
                no_session_persistence,
                memory,
                no_memory,
                no_session_memory,
                export,
                json,
                output_schema,
                output_description,
            )
            .await?;
        }
        Command::Serve {
            model,
            gateway_url,
            key,
            session_file,
            session_dir,
            listen,
            listen_uds,
            listen_uds_mode,
            session_idle_timeout,
            upstream_http2,
            session_id,
            r#continue: continue_session,
            no_session_persistence,
            memory,
            no_memory,
            no_session_memory,
            max_steps,
            max_tokens,
            system_prompt,
            append_system_prompt,
            no_context_files,
            context_window,
            cache_long,
            thinking,
            reasoning_effort,
            steering_mode,
            follow_up_mode,
            temperature,
            trust_project,
            force_untrusted,
            compaction_reserve_tokens,
            compaction_keep_recent_tokens,
            branch_summary_reserve_tokens,
            no_compaction,
            retry_max_retries,
            retry_base_delay_ms,
            retry_max_backoff_ms,
            idle_timeout_ms,
            block_images,
            no_block_images,
            no_image_auto_resize,
            bash_timeout_ms,
            bash_shell_path,
            bash_command_prefix,
            web_allow_private,
            web_allow_host,
            web_timeout_ms,
            exec_url,
            exec_header,
            exec_cmd,
            tools,
            exclude_tools,
            no_tools,
            sequential_tools,
            deny_tool,
            deny_bash_pattern,
            deny_path,
            approve,
            approval_timeout,
            models,
            no_skills,
            no_prompt_templates,
            extra_skill_paths,
            extra_prompt_template_paths,
            name,
            prompt_guidelines,
        } => {
            // Fail fast, before starting the server — see `run_task`'s identical check for why this
            // rejects rather than silently clearing (pi's own `--name` behavior).
            if let Some(n) = &name
                && n.trim().is_empty()
            {
                return Err("--name requires a non-empty value".into());
            }
            // A malformed `--deny-path` glob must never silently produce a no-op policy — see
            // `ToolPolicy::deny_path`'s doc comment for the fail-open this closes.
            if let Err(e) = ToolPolicy::validate_deny_path_patterns(&deny_path) {
                return Err(e.into());
            }
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
            // `--system-prompt`/`--append-system-prompt` may each name an existing, readable file
            // instead of literal text (pi-parity fix — matches pi's own `resolvePromptInput`). `run`'s
            // identical resolution (`main.rs::resolve_prompt_input`).
            let system = system_prompt
                .as_deref()
                .map(resolve_prompt_input)
                .unwrap_or_else(|| {
                    // Shell-path override doesn't affect this registry's use (listing tool
                    // names/descriptions for the default system prompt) — `describe()` doesn't mention it.
                    let mut reg = tools::default_registry_with(bash_timeout_ms, None);
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
            // `<cwd>/.claude/settings.json` tier on top of the global one first — see
            // `settings::effective_settings_for_cwd`'s own doc comment for the trust-gating rationale.
            let cwd = canonical_cwd(&std::env::current_dir().unwrap_or_default());
            let stored_settings = beyond_ai_agent::settings::effective_settings_for_cwd(&cwd);
            // Round 3 (pi-parity fix): same "explicit flag/env, then stored setting, then built-in
            // default" precedence as every other `stored_settings`-backed fallback here — see
            // `run_task`'s identical block, just below in this file, for the full set.
            let bash_shell_path =
                bash_shell_path.or_else(|| stored_settings.default_bash_shell_path.clone());
            let bash_command_prefix =
                bash_command_prefix.or_else(|| stored_settings.default_bash_command_prefix.clone());
            let compaction_reserve_tokens =
                compaction_reserve_tokens.or(stored_settings.default_compaction_reserve_tokens);
            let compaction_keep_recent_tokens = compaction_keep_recent_tokens
                .or(stored_settings.default_compaction_keep_recent_tokens);
            // Task #31 (pi-parity fix): `run_task`'s identical resolution, just below in this file.
            let branch_summary_reserve_tokens = branch_summary_reserve_tokens
                .or(stored_settings.default_branch_summary_reserve_tokens);
            let retry_max_retries = retry_max_retries.or(stored_settings.default_retry_max_retries);
            let retry_base_delay_ms =
                retry_base_delay_ms.or(stored_settings.default_retry_base_delay_ms);
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
                if session_file.is_none() {
                    stored_settings.default_session_dir.clone()
                } else {
                    None
                }
            });
            // Connected exactly once, here, before `serve`'s long-lived session loop starts — not
            // re-done on every `set_model`/`set_thinking` registry rebuild (`build_tools` reads
            // `cfg.mcp_tools` as already-resolved). Fail-soft: a server that fails to connect is
            // skipped with a warning, matching `has_gated_resources`'s own "warn, don't block the run"
            // convention in the `run` path above. `stored_settings.mcp_servers` is already trust-gated —
            // see that field's own doc comment.
            let (mcp_tools, mcp_warnings) =
                tools::mcp::connect_all(stored_settings.mcp_servers.as_deref().unwrap_or(&[]))
                    .await;
            for warning in &mcp_warnings {
                eprintln!("warning: {warning}");
            }
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
            let serve_cfg = serve::ServeConfig {
                provider_env: beyond_ai_agent::gateway_credential::ProviderEnv::from_process_env(
                    configured_gateway.is_some(),
                ),
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
                session_idle_timeout: session_idle_timeout.map(std::time::Duration::from_secs),
                upstream_http2,
                // The daemon path (`serve_ws`) fills this from `upstream_http2` before spawning any
                // session; the stdio/`run` path leaves it `None` and never pools.
                shared_http: None,
                session_id,
                continue_session,
                no_session_persistence,
                context_window,
                cache_long,
                thinking,
                reasoning_effort,
                temperature,
                trust_project,
                force_untrusted,
                compaction_reserve_tokens,
                compaction_keep_recent_tokens,
                branch_summary_reserve_tokens,
                no_compaction,
                retry_max_retries,
                retry_base_delay_ms: retry_base_delay_ms.map(std::time::Duration::from_millis),
                retry_max_backoff_ms: retry_max_backoff_ms.map(std::time::Duration::from_millis),
                idle_timeout_ms,
                block_images,
                image_auto_resize,
                bash_timeout_ms,
                bash_shell_path,
                bash_command_prefix,
                web_allow_private,
                web_allow_hosts: web_allow_host,
                web_timeout_ms,
                exec_url,
                exec_header,
                exec_cmd,
                tools,
                exclude_tools,
                no_tools,
                mcp_tools,
                // Discovered by `serve_session` itself, after it resolves project trust (a project-local
                // definition is trust-gated, like a skill) — not here, where the interactive trust grant
                // hasn't happened yet.
                agents: Vec::new(),
                sequential_tools,
                deny_tool,
                deny_bash_pattern,
                deny_path,
                // Fail fast on an unrecognized value, before any session is opened — the same discipline
                // `ToolPolicy::validate_deny_path_patterns` uses for a malformed `--deny-path` glob.
                approve: beyond_ai_agent::approval::GatedSet::parse(&approve).unwrap_or_else(|e| {
                    eprintln!("{e}");
                    std::process::exit(2);
                }),
                approval_timeout: (approval_timeout > 0)
                    .then(|| std::time::Duration::from_secs(approval_timeout)),
                models: models.unwrap_or_default(),
                no_skills,
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
            // WebSocket transport (TCP and/or UDS): one session per `?session_id=`, each outliving its
            // connection so a dropped client re-attaches to a still-running run (see `serve_ws`). Both
            // listeners front one shared supervisor. Absent both ⇒ the default stdio transport.
            //
            // systemd socket activation: if systemd started us with a passed socket (`LISTEN_FDS`) and
            // no explicit listen flag was given, adopt that socket instead of binding — the socket then
            // outlives a `systemctl restart` (connections queue in the kernel). Unix/systemd only, and
            // deferred to `--listen`/`--listen-uds` when either is set.
            let systemd_activated = cfg!(unix)
                && serve_cfg.listen.is_none()
                && serve_cfg.listen_uds.is_none()
                && std::env::var_os("LISTEN_FDS").is_some();
            let use_ws =
                serve_cfg.listen.is_some() || serve_cfg.listen_uds.is_some() || systemd_activated;
            let shutdown_cause = if use_ws {
                #[cfg(not(unix))]
                if listen_uds.is_some() {
                    return Err("--listen-uds is only supported on unix targets".into());
                }
                let listeners = serve_ws::ServeListeners {
                    tcp: serve_cfg.listen,
                    #[cfg(unix)]
                    uds: serve_cfg.listen_uds.clone(),
                    #[cfg(unix)]
                    uds_mode: serve_cfg.listen_uds_mode,
                    systemd: systemd_activated,
                };
                serve_ws::serve_ws(serve_cfg, listeners).await?
            } else {
                serve::serve(serve_cfg).await?
            };
            // `serve` reads stdin via `tokio::io::stdin()`, which parks a dedicated blocking OS
            // thread doing a blocking read for the life of the process. If stdin is never closed
            // (a client that doesn't hang up, or — the case this matters for — a SIGTERM/SIGINT
            // whose handler cancels the run and returns without stdin ever reaching EOF), that
            // thread is still parked here even though all async work is done. Falling through to
            // `#[tokio::main]`'s implicit runtime shutdown would then hang indefinitely: dropping
            // a `Runtime` waits for every outstanding blocking task, and a parked stdin read never
            // completes on its own. Exit explicitly instead — `serve` has already drained,
            // persisted, and flushed everything before returning, so there's nothing left to lose.
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
            #[cfg(unix)]
            tools::exec::wait_for_pending_group_kills(std::time::Duration::from_secs(2));
            std::process::exit(shutdown_cause.map(serve::Signal::exit_code).unwrap_or(0));
        }
        Command::Tools => {
            let reg = tools::default_registry();
            println!("{} tools:\n", reg.len());
            println!("{}", serde_json::to_string_pretty(&reg.definitions())?);
        }
        Command::ListModels { search } => {
            // Pi-parity fix: previously a bare list of ids — pi's own `--list-models` prints a table
            // (provider/model/context/max-out/thinking/images) built from data its model catalogue
            // already carries. Beyond has no separate provider field (a model id is forwarded verbatim;
            // see `agent_core::models`'s own module doc comment), so this mirrors the rest of pi's
            // columns from `agent_core::capabilities`, which already computes every one of them for
            // wire-shaping — nothing new is invented here, just surfaced.
            //
            // Task #51 (pi-parity fix): an optional positional `search` fuzzy-filters model ids —
            // matches pi's own `--list-models <search>` (`fuzzyFilter`/`fuzzyMatch`,
            // `packages/tui/src/fuzzy.ts`), a non-contiguous, order-preserving, word-boundary-scored
            // subsequence match rather than a plain substring check, so e.g. "sn5" finds
            // "claude-sonnet-4-5" the way pi's own table search does. Previously a plain
            // case-insensitive `contains`, which that query would never match at all.
            let models: Vec<&str> = match &search {
                Some(query) => {
                    let mut scored: Vec<(&str, f64)> = serve::available_models()
                        .iter()
                        .filter_map(|m| fuzzy_match(query, m).map(|score| (*m, score)))
                        .collect();
                    // Lower score is a better match (mirrors pi's own ascending sort) — a stable sort
                    // keeps `available_models()`'s own relative order as the tie-break, same as pi's
                    // `Array.prototype.sort` (stable per spec).
                    scored
                        .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                    scored.into_iter().map(|(m, _)| m).collect()
                }
                None => serve::available_models().to_vec(),
            };
            println!(
                "{:<22} {:>10} {:>9} {:<8} {:<6}",
                "model", "context", "max-out", "thinking", "vision"
            );
            for model in models {
                let caps = agent_core::capabilities(model);
                let thinking = caps.reasoning_effort
                    || caps.thinking != agent_core::models::ThinkingShape::None;
                println!(
                    "{:<22} {:>10} {:>9} {:<8} {:<6}",
                    model,
                    format_token_count(caps.context_window),
                    format_token_count(caps.max_output),
                    if thinking { "yes" } else { "no" },
                    if caps.supports_vision { "yes" } else { "no" },
                );
            }
        }
        Command::Trust { path } => {
            let dir = match path {
                Some(p) => PathBuf::from(p),
                None => std::env::current_dir()?,
            };
            let mut store = beyond_ai_agent::trust_store::TrustStore::open_default();
            store.trust(&dir)?;
            println!("trusted: {}", dir.display());
        }
        Command::Untrust { path } => {
            let dir = match path {
                Some(p) => PathBuf::from(p),
                None => std::env::current_dir()?,
            };
            let mut store = beyond_ai_agent::trust_store::TrustStore::open_default();
            store.distrust(&dir)?;
            println!("untrusted: {}", dir.display());
        }
        Command::ClearTrust { path } => {
            let dir = match path {
                Some(p) => PathBuf::from(p),
                None => std::env::current_dir()?,
            };
            let mut store = beyond_ai_agent::trust_store::TrustStore::open_default();
            store.clear(&dir)?;
            println!("cleared: {}", dir.display());
        }
        Command::TrustStatus { path } => {
            let dir = match path {
                Some(p) => PathBuf::from(p),
                None => std::env::current_dir()?,
            };
            let store = beyond_ai_agent::trust_store::TrustStore::open_default();
            let status = match store.lookup(&dir) {
                beyond_ai_agent::trust_store::Trust::Trusted => "trusted",
                beyond_ai_agent::trust_store::Trust::Untrusted => "untrusted",
                beyond_ai_agent::trust_store::Trust::Unknown => "unknown",
            };
            println!("{status}: {}", dir.display());
        }
        Command::Login { provider } => {
            let provider_id = beyond_ai_agent::oauth::OAuthProviderId::parse(&provider)
                .ok_or_else(|| unknown_provider_error(&provider))?;
            let cancel = agent_core::CancellationToken::new();
            let credential =
                beyond_ai_agent::oauth::login(provider_id, &CliLoginCallbacks, &cancel).await?;
            let mut store = beyond_ai_agent::auth_store::AuthStore::open_default();
            store.set(provider_id.store_key(), credential)?;
            println!("logged in: {provider_id}");
        }
        Command::Logout { provider } => {
            let provider_id = beyond_ai_agent::oauth::OAuthProviderId::parse(&provider)
                .ok_or_else(|| unknown_provider_error(&provider))?;
            let mut store = beyond_ai_agent::auth_store::AuthStore::open_default();
            if store.remove(provider_id.store_key())? {
                println!("logged out: {provider_id}");
            } else {
                println!("not logged in: {provider_id}");
            }
        }
        Command::AuthStatus { provider } => {
            let store = beyond_ai_agent::auth_store::AuthStore::open_default();
            let providers = match &provider {
                Some(p) => vec![
                    beyond_ai_agent::oauth::OAuthProviderId::parse(p)
                        .ok_or_else(|| unknown_provider_error(p))?,
                ],
                None => beyond_ai_agent::oauth::OAuthProviderId::all().to_vec(),
            };
            for id in providers {
                let status = match store.get(id.store_key()) {
                    None => "logged_out",
                    Some(stored) if stored.last_refresh_error.is_some() => "needs_reauth",
                    Some(_) => "logged_in",
                };
                println!("{id}: {status}");
            }
        }
        Command::McpLogin { name } => {
            let cwd = canonical_cwd(&std::env::current_dir().unwrap_or_default());
            let settings = beyond_ai_agent::settings::effective_settings_for_cwd(&cwd);
            let servers = settings.mcp_servers.unwrap_or_default();
            let server = servers.iter().find(|s| s.name == name).ok_or_else(|| {
                format!(
                    "no MCP server named `{name}` is configured (checked global and, if this \
                     directory is trusted, project settings.json)"
                )
            })?;
            let url = match &server.transport {
                beyond_ai_agent::settings::McpTransport::Http { url, .. } => url.clone(),
                beyond_ai_agent::settings::McpTransport::Stdio { .. } => {
                    return Err(format!(
                        "`{name}` uses the stdio transport, which has no OAuth login of its own — \
                         configure a credential via its `env` instead"
                    )
                    .into());
                }
            };

            let mut manager = rmcp::transport::auth::AuthorizationManager::new(&url)
                .await
                .map_err(|e| format!("failed to start MCP OAuth for `{name}`: {e}"))?;
            let auth_store = beyond_ai_agent::mcp_auth_store::McpAuthStore::open_default();
            manager.set_credential_store(auth_store.scoped(&name));
            let metadata = manager.discover_metadata().await.map_err(|e| {
                format!("`{name}` does not support MCP OAuth (or discovery failed): {e}")
            })?;
            manager.set_metadata(metadata);
            let scopes = manager.select_scopes(None, &[]);
            let scope_refs: Vec<&str> = scopes.iter().map(String::as_str).collect();

            // Dynamic client registration means we choose our own redirect URI (unlike `agent
            // login`'s providers, each registered in advance against a fixed port) — pick a free
            // local one, same small "check then reuse" window `agent trust`-adjacent tooling already
            // accepts elsewhere in this codebase for a one-shot interactive command.
            let port = std::net::TcpListener::bind("127.0.0.1:0")
                .and_then(|l| l.local_addr())
                .map_err(|e| format!("failed to allocate a local OAuth callback port: {e}"))?
                .port();
            let redirect_uri = format!("http://127.0.0.1:{port}/callback");

            let session = rmcp::transport::auth::AuthorizationSession::new(
                manager,
                &scope_refs,
                &redirect_uri,
                Some("beyond-ai-agent"),
                None,
            )
            .await
            .map_err(|e| format!("failed to start MCP OAuth authorization for `{name}`: {e}"))?;
            let auth_url = session.get_authorization_url().to_string();
            let csrf_token = url::Url::parse(&auth_url)
                .ok()
                .and_then(|u| {
                    u.query_pairs()
                        .find(|(k, _)| k == "state")
                        .map(|(_, v)| v.into_owned())
                })
                .ok_or("the generated authorization URL is missing its `state` parameter")?;

            let listener =
                beyond_ai_agent::oauth::callback_server::CallbackServer::bind("127.0.0.1", port)
                    .map_err(|e| {
                        format!("failed to bind the local OAuth callback listener: {e}")
                    })?;
            eprintln!("Open this URL in a browser to continue:\n\n  {auth_url}\n");
            let cancel = agent_core::CancellationToken::new();
            let code = listener
                .wait_for_callback("/callback", csrf_token.clone(), cancel)
                .await
                .ok_or("login cancelled (no callback received)")?;
            session
                .handle_callback(&code, &csrf_token)
                .await
                .map_err(|e| format!("failed to complete MCP OAuth login for `{name}`: {e}"))?;
            println!("logged in: {name}");
        }
        Command::McpLogout { name } => {
            let store = beyond_ai_agent::mcp_auth_store::McpAuthStore::open_default();
            let had_credential = store.has_credential(&name);
            store.clear(&name)?;
            if had_credential {
                println!("logged out: {name}");
            } else {
                println!("not logged in: {name}");
            }
        }
        Command::Settings {
            model,
            clear_model,
            gateway_url,
            clear_gateway_url,
            session_dir,
            clear_session_dir,
            default_project_trust,
            clear_default_project_trust,
            default_reasoning_effort,
            clear_default_reasoning_effort,
            block_images,
            clear_block_images,
            image_auto_resize,
            clear_image_auto_resize,
            thinking_budget,
            clear_thinking_budget,
            default_bash_shell_path,
            clear_default_bash_shell_path,
            default_bash_command_prefix,
            clear_default_bash_command_prefix,
            default_compaction_reserve_tokens,
            clear_default_compaction_reserve_tokens,
            default_compaction_keep_recent_tokens,
            clear_default_compaction_keep_recent_tokens,
            default_retry_max_retries,
            clear_default_retry_max_retries,
            default_retry_base_delay_ms,
            clear_default_retry_base_delay_ms,
            default_provider_timeout_ms,
            clear_default_provider_timeout_ms,
            default_retry_max_backoff_ms,
            clear_default_retry_max_backoff_ms,
            default_models,
            clear_default_models,
            default_skill_paths,
            clear_default_skill_paths,
            default_prompt_template_paths,
            clear_default_prompt_template_paths,
            default_branch_summary_reserve_tokens,
            clear_default_branch_summary_reserve_tokens,
        } => {
            let mut store = beyond_ai_agent::settings::SettingsStore::open_default();
            let any_write = model.is_some()
                || clear_model
                || gateway_url.is_some()
                || clear_gateway_url
                || session_dir.is_some()
                || clear_session_dir
                || default_project_trust.is_some()
                || clear_default_project_trust
                || default_reasoning_effort.is_some()
                || clear_default_reasoning_effort
                || block_images.is_some()
                || clear_block_images
                || image_auto_resize.is_some()
                || clear_image_auto_resize
                || !thinking_budget.is_empty()
                || !clear_thinking_budget.is_empty()
                || default_bash_shell_path.is_some()
                || clear_default_bash_shell_path
                || default_bash_command_prefix.is_some()
                || clear_default_bash_command_prefix
                || default_compaction_reserve_tokens.is_some()
                || clear_default_compaction_reserve_tokens
                || default_compaction_keep_recent_tokens.is_some()
                || clear_default_compaction_keep_recent_tokens
                || default_retry_max_retries.is_some()
                || clear_default_retry_max_retries
                || default_retry_base_delay_ms.is_some()
                || clear_default_retry_base_delay_ms
                || default_provider_timeout_ms.is_some()
                || clear_default_provider_timeout_ms
                || default_retry_max_backoff_ms.is_some()
                || clear_default_retry_max_backoff_ms
                || default_models.is_some()
                || clear_default_models
                || default_skill_paths.is_some()
                || clear_default_skill_paths
                || default_prompt_template_paths.is_some()
                || clear_default_prompt_template_paths
                || default_branch_summary_reserve_tokens.is_some()
                || clear_default_branch_summary_reserve_tokens;
            if model.is_some() || clear_model {
                store.set_default_model(model)?;
            }
            if gateway_url.is_some() || clear_gateway_url {
                store.set_default_gateway_url(gateway_url)?;
            }
            if session_dir.is_some() || clear_session_dir {
                store.set_default_session_dir(session_dir)?;
            }
            if default_project_trust.is_some() || clear_default_project_trust {
                store.set_default_project_trust(default_project_trust)?;
            }
            if default_reasoning_effort.is_some() || clear_default_reasoning_effort {
                store.set_default_reasoning_effort(
                    default_reasoning_effort.map(|e| e.as_str().to_string()),
                )?;
            }
            if block_images.is_some() || clear_block_images {
                store.set_block_images(if clear_block_images {
                    None
                } else {
                    block_images
                })?;
            }
            if image_auto_resize.is_some() || clear_image_auto_resize {
                store.set_image_auto_resize(if clear_image_auto_resize {
                    None
                } else {
                    image_auto_resize
                })?;
            }
            for kv in &thinking_budget {
                let (effort, tokens) = kv
                    .split_once('=')
                    .ok_or_else(|| format!("--thinking-budget {kv:?} must be EFFORT=TOKENS"))?;
                parse_reasoning_effort(effort)
                    .map_err(|e| format!("--thinking-budget {kv:?}: {e}"))?;
                let tokens: u32 = tokens.parse().map_err(|_| {
                    format!("--thinking-budget {kv:?}: TOKENS must be a non-negative integer")
                })?;
                store.set_thinking_budget_override(effort.to_string(), Some(tokens))?;
            }
            for effort in &clear_thinking_budget {
                parse_reasoning_effort(effort)
                    .map_err(|e| format!("--clear-thinking-budget {effort:?}: {e}"))?;
                store.set_thinking_budget_override(effort.to_string(), None)?;
            }
            if default_bash_shell_path.is_some() || clear_default_bash_shell_path {
                store.set_default_bash_shell_path(default_bash_shell_path)?;
            }
            if default_bash_command_prefix.is_some() || clear_default_bash_command_prefix {
                store.set_default_bash_command_prefix(default_bash_command_prefix)?;
            }
            if default_compaction_reserve_tokens.is_some()
                || clear_default_compaction_reserve_tokens
            {
                store.set_default_compaction_reserve_tokens(default_compaction_reserve_tokens)?;
            }
            if default_compaction_keep_recent_tokens.is_some()
                || clear_default_compaction_keep_recent_tokens
            {
                store.set_default_compaction_keep_recent_tokens(
                    default_compaction_keep_recent_tokens,
                )?;
            }
            if default_retry_max_retries.is_some() || clear_default_retry_max_retries {
                store.set_default_retry_max_retries(default_retry_max_retries)?;
            }
            if default_retry_base_delay_ms.is_some() || clear_default_retry_base_delay_ms {
                store.set_default_retry_base_delay_ms(default_retry_base_delay_ms)?;
            }
            if default_provider_timeout_ms.is_some() || clear_default_provider_timeout_ms {
                store.set_default_provider_timeout_ms(default_provider_timeout_ms)?;
            }
            if default_retry_max_backoff_ms.is_some() || clear_default_retry_max_backoff_ms {
                store.set_default_retry_max_backoff_ms(default_retry_max_backoff_ms)?;
            }
            if default_models.is_some() || clear_default_models {
                store.set_default_models_list(default_models)?;
            }
            if default_skill_paths.is_some() || clear_default_skill_paths {
                store.set_default_skill_paths(default_skill_paths)?;
            }
            if default_prompt_template_paths.is_some() || clear_default_prompt_template_paths {
                store.set_default_prompt_template_paths(default_prompt_template_paths)?;
            }
            if default_branch_summary_reserve_tokens.is_some()
                || clear_default_branch_summary_reserve_tokens
            {
                store.set_default_branch_summary_reserve_tokens(
                    default_branch_summary_reserve_tokens,
                )?;
            }
            if any_write {
                println!("updated settings:");
            }
            let s = store.get();
            println!(
                "default_model: {}",
                s.default_model.as_deref().unwrap_or("(not set)")
            );
            println!(
                "default_gateway_url: {}",
                s.default_gateway_url.as_deref().unwrap_or("(not set)")
            );
            println!(
                "default_session_dir: {}",
                s.default_session_dir.as_deref().unwrap_or("(not set)")
            );
            println!(
                "default_project_trust: {}",
                match s.default_project_trust {
                    Some(beyond_ai_agent::settings::TrustPolicy::Always) => "always",
                    Some(beyond_ai_agent::settings::TrustPolicy::Never) => "never",
                    Some(beyond_ai_agent::settings::TrustPolicy::Ask) => "ask",
                    None => "(not set)",
                }
            );
            println!(
                "default_reasoning_effort: {}",
                s.default_reasoning_effort.as_deref().unwrap_or("(not set)")
            );
            println!(
                "block_images: {}",
                match s.block_images {
                    Some(true) => "true",
                    Some(false) => "false",
                    None => "(not set)",
                }
            );
            println!(
                "image_auto_resize: {}",
                match s.image_auto_resize {
                    Some(true) => "true",
                    Some(false) => "false",
                    None => "(not set)",
                }
            );
            match &s.thinking_budget_overrides {
                Some(overrides) if !overrides.is_empty() => {
                    let rendered = overrides
                        .iter()
                        .map(|(effort, tokens)| format!("{effort}={tokens}"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    println!("thinking_budget_overrides: {rendered}");
                }
                _ => println!("thinking_budget_overrides: (not set)"),
            }
            println!(
                "default_bash_shell_path: {}",
                s.default_bash_shell_path.as_deref().unwrap_or("(not set)")
            );
            println!(
                "default_bash_command_prefix: {}",
                s.default_bash_command_prefix
                    .as_deref()
                    .unwrap_or("(not set)")
            );
            println!(
                "default_compaction_reserve_tokens: {}",
                s.default_compaction_reserve_tokens
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "(not set)".to_string())
            );
            println!(
                "default_compaction_keep_recent_tokens: {}",
                s.default_compaction_keep_recent_tokens
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "(not set)".to_string())
            );
            println!(
                "default_retry_max_retries: {}",
                s.default_retry_max_retries
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "(not set)".to_string())
            );
            println!(
                "default_retry_base_delay_ms: {}",
                s.default_retry_base_delay_ms
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "(not set)".to_string())
            );
            println!(
                "default_provider_timeout_ms: {}",
                s.default_provider_timeout_ms
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "(not set)".to_string())
            );
            println!(
                "default_retry_max_backoff_ms: {}",
                s.default_retry_max_backoff_ms
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "(not set)".to_string())
            );
            println!(
                "default_models_list: {}",
                s.default_models_list
                    .as_ref()
                    .map(|v| v.join(","))
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| "(not set)".to_string())
            );
            println!(
                "default_skill_paths: {}",
                s.default_skill_paths
                    .as_ref()
                    .map(|v| v.join(","))
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| "(not set)".to_string())
            );
            println!(
                "default_prompt_template_paths: {}",
                s.default_prompt_template_paths
                    .as_ref()
                    .map(|v| v.join(","))
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| "(not set)".to_string())
            );
            println!(
                "default_branch_summary_reserve_tokens: {}",
                s.default_branch_summary_reserve_tokens
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "(not set)".to_string())
            );
            // Fix 9's "CLI-visible" requirement: this file is entirely hand-edited (like pi's own
            // `models.json`), with no `--set`/`--clear` flags of its own here — just enough surface so
            // an operator debugging "why is this model still hitting the gateway" can confirm the file
            // is actually being read, and how many overrides it currently holds, without a dedicated
            // dump-the-whole-file command.
            let overrides = beyond_ai_agent::settings::ModelOverrides::open_default();
            let overrides_path = beyond_ai_agent::settings::model_overrides_path();
            if overrides.is_empty() {
                println!(
                    "model_overrides: {} (not present or empty)",
                    overrides_path.display()
                );
            } else {
                println!(
                    "model_overrides: {} ({} model id(s) overridden)",
                    overrides_path.display(),
                    overrides.len()
                );
            }
        }
        Command::Export { session, output } => {
            let (store, sess) =
                beyond_ai_agent::session_store::SessionStore::open(PathBuf::from(&session))
                    .map_err(|e| format!("failed to open session {session}: {e}"))?;
            let branches = store.abandoned_branches();
            // `export_html_full` (Task #44 integration), but with `system_prompt`/`tools` genuinely
            // `None`: this standalone subcommand renders an already-persisted session file straight off
            // disk with no gateway/key/model involved at all (see this crate's own ARCHITECTURE.md), so
            // there's no live `Agent`/`ToolRegistry` here to pull either from — and the session file
            // itself records neither the exact system prompt text nor which `--tools`/`--exclude-tools`
            // filter (if any) a past run used, so reconstructing either would mean fabricating data
            // that may not match what actually ran. Usage totals are different: Fix 6 (pi-parity gap):
            // `sess`'s own running token counters (`input_tokens`/etc.) are never persisted/restored
            // across a process restart, but they're fully reconstructable by summing each message's own
            // `usage` (`serve::message_export_usage_totals` — the same computation `serve`'s
            // `export_html` RPC and `run --export` derive their own totals from, one layer up), which
            // *is* right here in `sess.messages` — previously this passed `usage: None` unconditionally,
            // the one of the three export entry points that silently omitted the line instead.
            let path = beyond_ai_agent::export::export_html_full(
                store.meta(),
                &sess.messages,
                &branches,
                Some(beyond_ai_agent::serve::message_export_usage_totals(&sess)),
                store.export_events(),
                None,
                None,
                output.as_deref(),
            )?;
            println!("Exported to: {}", path.display());
        }
    }
    Ok(())
}

/// Split `run`'s positional `tasks` into file references (an `@`-prefixed argument, path with the
/// prefix stripped) and plain message strings, each preserving its own relative order.
fn partition_tasks(tasks: Vec<String>) -> (Vec<String>, Vec<String>) {
    let mut file_refs = Vec::new();
    let mut messages = Vec::new();
    for t in tasks {
        match t.strip_prefix('@') {
            Some(path) => file_refs.push(path.to_string()),
            None => messages.push(t),
        }
    }
    (file_refs, messages)
}

/// Text plus image attachments gathered from `@file` references, kept separate so the caller can
/// build a `Message::user_with_images` turn when any image was found, rather than folding raw binary
/// bytes into the same string every plain-text `@file` reference produces.
#[derive(Debug)]
struct FileRefs {
    text: String,
    images: Vec<agent_core::ImageSource>,
}

/// How many leading bytes are read to identify a supported image format by magic bytes (PNG's 8-byte
/// signature, WebP's 12-byte `RIFF....WEBP` header, ...). Deliberately far short of `tools::read`'s own
/// 4100-byte sniff window — that budget exists solely to reach a PNG's `acTL` chunk for the
/// animated-PNG check, which this probe doesn't need to make itself (see [`looks_like_image`]'s doc
/// comment): it only decides whether a file is worth routing through the `read` tool's full image
/// pipeline at all, so every ordinary (non-image) `@file` reference — the overwhelming majority — pays
/// for just this one short read, not a second full-file pass.
const IMAGE_SNIFF_LEN: usize = 32;

/// Whether `path`'s leading bytes match one of the image formats the `read` tool can inline as an
/// attachment. Mirrors `tools::read`'s own magic-byte probe (matching only the five formats it can
/// actually encode/re-encode) rather than reinventing format detection — `tools::read`'s sniffing
/// helpers are private to that module, so the same `image::guess_format` call it wraps is made
/// directly here instead. A `false` here doesn't rule out `path` truly being an image under a
/// corrupted/truncated header, nor does it fall back to guessing by extension the way the `read`
/// *tool* does for a model-issued call — matching pi's own CLI `@file` processor
/// (`detectSupportedImageMimeTypeFromFile`), which likewise never falls back to extension guessing at
/// this layer (only `read.ts`'s tool-call path does). A file that only *looks* like an image by name
/// still reads as plain text below, same as before this fix.
fn looks_like_image(path: &Path) -> bool {
    let Ok(mut f) = std::fs::File::open(path) else {
        return false;
    };
    let mut buf = [0u8; IMAGE_SNIFF_LEN];
    let Ok(n) = f.read(&mut buf) else {
        return false;
    };
    matches!(
        image::guess_format(&buf[..n]),
        Ok(image::ImageFormat::Png
            | image::ImageFormat::Jpeg
            | image::ImageFormat::Gif
            | image::ImageFormat::WebP
            | image::ImageFormat::Bmp)
    )
}

/// Read each of `file_refs` (tilde-expanded, then resolved against `cwd`; an already-absolute ref is
/// used as-is). A plain (non-image) file's contents are wrapped in a `<file name="...">` block,
/// concatenated in argument order — unchanged from before this fix. A zero-byte file is skipped
/// entirely (Task #38). A file whose leading bytes identify it as an image (see
/// [`looks_like_image`]) is instead run through the `read` tool's own image pipeline (sniffing,
/// decode/validate, downscale-to-budget, format conversion) so it can be attached as a real
/// [`agent_core::ImageSource`] rather than handed to `std::fs::read_to_string`, which errors outright
/// on binary image bytes — the crash this fix closes (`run @screenshot.png "..."` previously failed
/// instead of attaching the screenshot). Errors naming the first unreadable (or undecodable) file, so
/// a typo'd `@path` — or a genuinely corrupt image — fails loudly instead of silently vanishing from
/// the prompt.
///
/// Tasks #35/#36 (pi-parity fix): `block_images`/`image_auto_resize` (the fully-resolved
/// `--block-images`/`--no-image-auto-resize` flags — explicit flag, then stored `agent settings`
/// default, then built-in default, exactly like every other such flag) are threaded down to the
/// `read` tool's own image pipeline for a CLI `@file` attachment, the same way `Agent::block_images`'s
/// tool-dispatch gate and `default_registry_with_prefix_and_image_auto_resize` already do for a
/// model-issued `read` tool call. Previously neither flag had any effect at all here: this call site
/// built a bare `tools::read::Read::default()` with no awareness of either, so a CLI attachment's
/// images spliced straight into the first `Message` regardless of `--block-images`, at their original
/// size regardless of `--no-image-auto-resize`.
///
/// `model_supports_vision` (Task 3, pi-parity fix, pass 19) is the active model's real
/// `agent_core::models::capabilities(&model).supports_vision`, resolved by the caller (`run_task`
/// resolves it before this is ever called — see that call site's own doc comment for why the model must
/// be resolved first). ANDed with `!block_images` below, exactly like the model-issued `read` tool-call
/// dispatch path in `agent_core::agent` already does — previously this omitted the real capability
/// entirely (`_model_supports_vision: !block_images`), so a non-vision model with `block_images` left at
/// its default (`false`) got a CLI attachment dispatched claiming vision support it didn't have, and
/// `read.rs` never appended its non-vision-image explanatory note the way an equivalent model-issued call
/// would have (cosmetic only: the wire-level dialect filter still stripped the image correctly before it
/// ever reached the provider).
async fn read_file_refs(
    file_refs: &[String],
    cwd: &Path,
    block_images: bool,
    image_auto_resize: bool,
    model_supports_vision: bool,
) -> Result<FileRefs, Box<dyn std::error::Error>> {
    read_file_refs_with_home(
        file_refs,
        cwd,
        std::env::var("HOME").ok().as_deref(),
        block_images,
        image_auto_resize,
        model_supports_vision,
    )
    .await
}

/// [`read_file_refs`], with `home` passed explicitly instead of read fresh from `$HOME` — split out
/// purely so the Task #20 tilde-expansion behavior is unit-testable without mutating the real,
/// process-wide (and test-parallelism-unsafe) environment, the same reasoning `tools::expand_tilde`
/// itself was already split out for.
async fn read_file_refs_with_home(
    file_refs: &[String],
    cwd: &Path,
    home: Option<&str>,
    block_images: bool,
    image_auto_resize: bool,
    model_supports_vision: bool,
) -> Result<FileRefs, Box<dyn std::error::Error>> {
    let mut text = String::new();
    let mut images = Vec::new();
    for r in file_refs {
        // Task #20 (pi-parity fix): expand a leading `~` before resolving against `cwd` — the same
        // `tools::expand_tilde` helper every tool already uses for a model-supplied path. The `@`
        // prefix on a CLI file reference defeats the shell's own tilde-expansion (it's inside a single
        // argument, so nothing ever substitutes it), so `agent run @~/notes.md "..."` previously failed
        // outright with "failed to read ~/notes.md: No such file or directory" instead of reading the
        // intended file.
        let expanded = tools::expand_tilde(r, home);
        let path = cwd.join(&expanded);
        // Task #38 (pi-parity fix): skip a zero-byte file outright, before any image-format sniffing —
        // matches pi's own `file-processor.ts` (`stats.size === 0` → `continue`). Without this, an
        // empty `@file` produced an empty-but-present `<file name="...">\n\n</file>` block instead of
        // contributing nothing at all. `unwrap_or(1)` (not `0`) on a failed `metadata()` call — a
        // missing/unreadable file must still fall through to the normal "failed to read" error below,
        // not be silently skipped as if it were merely empty.
        if std::fs::metadata(&path).map(|m| m.len()).unwrap_or(1) == 0 {
            continue;
        }
        if looks_like_image(&path) {
            let path_str = path.to_string_lossy().into_owned();
            // Tasks #35/#36: `with_image_auto_resize` matches the registry's own construction
            // (`tools::default_registry_with_prefix_and_image_auto_resize`); `_model_supports_vision` is
            // the same schema-undocumented input field `agent_core::agent`'s tool-dispatch loop injects
            // before a model-issued `read` call (see `Read::run`'s own doc comment) — reused here so
            // `--block-images` downgrades a CLI attachment through the exact same path (image dropped,
            // `NON_VISION_IMAGE_NOTE` appended) rather than a second, divergent implementation. Task 3
            // (pi-parity fix, pass 19): ANDs in the real `model_supports_vision` capability too, matching
            // the model-issued dispatch path's own `capabilities(&self.model).supports_vision &&
            // !this.block_images` — previously this was `!block_images` alone, so a non-vision model
            // (with `block_images` at its default `false`) got a CLI attachment dispatched as if the
            // model supported vision.
            let out = tools::read::Read::default()
                .with_image_auto_resize(image_auto_resize)
                .run(serde_json::json!({
                    "path": path_str,
                    "_model_supports_vision": model_supports_vision && !block_images,
                }))
                .await
                .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
            // `out.images` is empty here only when the `read` tool sniffed a real image but couldn't
            // inline it (too large to downscale under budget, or a BMP that failed to convert) —
            // `out.text` already carries a `"[Image omitted: ...]"` explanation in that case, so use it
            // as the note rather than falling through to a UTF-8 read of binary image bytes.
            images.extend(out.images);
            if !out.text.is_empty() {
                text.push_str(&format!(
                    "<file name=\"{}\">{}</file>\n",
                    path.display(),
                    out.text
                ));
            }
            continue;
        }
        let content = std::fs::read_to_string(&path)
            .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
        text.push_str(&format!(
            "<file name=\"{}\">\n{content}\n</file>\n",
            path.display()
        ));
    }
    Ok(FileRefs { text, images })
}

/// Resolve a `--system-prompt`/`--append-system-prompt` value: if it names an existing, readable file,
/// its contents are used instead of the literal string — matches pi's own `resolvePromptInput`
/// (`existsSync(input)` check, then reads the file if so). Falls back to the literal value on a read
/// error (permission denied, a race where the file vanished between the exists check and the read)
/// rather than failing the whole invocation over what might still be a perfectly good literal string
/// that merely happens to look like a path.
fn resolve_prompt_input(raw: &str) -> String {
    let path = Path::new(raw);
    if path.is_file()
        && let Ok(contents) = fs::read_to_string(path)
    {
        return contents;
    }
    raw.to_string()
}

/// The full contents of stdin, if it's piped (not an interactive terminal) and non-empty once trimmed.
/// `None` otherwise — including on a read error (a broken pipe just means there was nothing to add) or
/// whitespace-only input. Task #37 (pi-parity fix): matches pi's own `readPipedStdin`'s `data.trim() ||
/// undefined` — previously a whitespace-only pipe (e.g. a trailing newline from an empty upstream
/// command in `some-cmd | agent run "..."`) was included verbatim as the composed message's leading
/// content instead of being treated as "nothing was actually piped".
fn read_stdin_if_piped() -> Option<String> {
    let stdin = std::io::stdin();
    if stdin.is_terminal() {
        return None;
    }
    let mut buf = String::new();
    match stdin.lock().read_to_string(&mut buf) {
        Ok(_) => trim_piped_stdin(&buf),
        Err(_) => None,
    }
}

/// [`read_stdin_if_piped`]'s trim/blank-check logic, split out so it's unit-testable without a real
/// piped stdin — Task #37 (pi-parity fix): matches pi's own `readPipedStdin`'s `data.trim() ||
/// undefined`. `None` for an empty or whitespace-only buffer (e.g. just a trailing newline from an
/// empty upstream command in `some-cmd | agent run "..."`), previously included verbatim as the
/// composed message's leading content instead of being treated as "nothing was actually piped".
fn trim_piped_stdin(buf: &str) -> Option<String> {
    let trimmed = buf.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

/// [`run_turn_once`], wrapped with the same whole-run auto-retry `serve.rs`'s `"prompt"` command gets
/// (see `beyond_ai_agent::retry`) — a run that ends in a transient-looking error (one already
/// exhausted `agent_core`'s own within-turn retries) is re-invoked from scratch against the same
/// session, up to `retry::MAX_RUN_RETRIES` times with backoff, rather than failing a whole `agent run`
/// invocation (plausibly unattended — a cron job, a CI step) outright on a hiccup that `serve` would
/// have quietly recovered from. A retried attempt's own streamed output (text/JSON events) follows
/// directly after a `[retrying...]` stderr notice — nothing is erased, matching how `serve` demarcates
/// attempts with an `auto_retry_start` frame rather than hiding the failed one.
/// A cancelled turn (SIGTERM/SIGHUP/Ctrl-C — see the `ShutdownSignal` wiring in `run_task`, or a future
/// `--timeout` equivalent) is an expected, clean stop, not a crash: printing it through `main`'s
/// default `Result` `Termination` would dump `Error: Cancelled` (the bare enum variant, via `Debug`)
/// with no context a script/CI caller could act on. Matches the `[refused]`/exit(1) precedent just
/// below in `run_task` — a clear bracketed status line on stderr, then a distinct process exit
/// instead of unwinding further. Any other error still propagates normally via `?`.
///
/// `shutdown_cause` (Task #41 pi-parity fix) picks the exit code: the matching POSIX `128 + signal`
/// code when a real shutdown signal caused this cancellation, or the prior bare `exit(1)` for a genuine
/// non-signal cancellation (there is currently no other way `run_task`'s own `cancel` token gets
/// tripped — see its construction in `run_task` — but this doesn't assume that stays true forever).
fn unwrap_turn_result(
    result: agent_core::Result<agent_core::StopReason>,
    shutdown_cause: &std::sync::Mutex<Option<serve::Signal>>,
) -> Result<agent_core::StopReason, Box<dyn std::error::Error>> {
    match result {
        Ok(reason) => Ok(reason),
        Err(agent_core::Error::Cancelled) => {
            eprintln!("[cancelled]");
            let code = shutdown_cause
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .map(serve::Signal::exit_code)
                .unwrap_or(1);
            // A cancellation this recent may have just dropped a `GroupKillGuard` for an in-flight
            // bash tool call — that guard's cleanup runs on a detached thread `process::exit` below
            // won't wait for on its own, so an in-flight timed-out/backgrounded grandchild would be
            // silently orphaned without this. Bounded, not indefinite: a hung `kill`/`ps` shell-out
            // must not hang the whole process's own shutdown.
            #[cfg(unix)]
            tools::exec::wait_for_pending_group_kills(std::time::Duration::from_secs(2));
            std::process::exit(code);
        }
        Err(e) => Err(e.into()),
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_turn(
    agent: &Agent,
    session: &mut Session,
    json: bool,
    cancel: &agent_core::CancellationToken,
    retry_policy: &beyond_ai_agent::retry::RunRetryPolicy,
    broken_pipe: &AtomicBool,
    steering: &agent_core::Steering,
    session_memory_active: bool,
    pressure_point: u32,
) -> agent_core::Result<agent_core::StopReason> {
    let mut attempt = 0u32;
    loop {
        let result = run_turn_once(
            agent,
            session,
            json,
            cancel,
            broken_pipe,
            steering,
            session_memory_active,
            pressure_point,
        )
        .await;
        match &result {
            Err(e)
                if attempt < retry_policy.max_retries
                    && beyond_ai_agent::retry::is_retryable_whole_run(e) =>
            {
                attempt += 1;
                let delay = retry_policy.backoff(attempt);
                eprintln!(
                    "\n[transient error, retrying {attempt}/{}: {e}]",
                    retry_policy.max_retries
                );
                // The failed attempt's closing error record must not survive into the retry — see
                // `Session::pop_error_record`'s doc comment (this is the same run resuming from
                // scratch, not a fresh prompt).
                session.pop_error_record();
                tokio::time::sleep(delay).await;
            }
            _ => return result,
        }
    }
}

/// Stream one turn's assistant reply to stdout. In text mode (`json: false`): live text, a
/// `[tool: name]` marker when the model calls one, then a trailing blank line once the turn ends. In
/// JSON mode (`--json`): one `AgentEvent` object per line — the full observation surface (tool
/// calls/results, turn boundaries, compaction), the same shape `serve`'s NDJSON protocol streams,
/// rather than only the raw model-stream deltas `StreamEvent` carries.
///
/// Returns the turn's final [`agent_core::StopReason`] — the *last* one observed, for a multi-step
/// turn that made several model round-trips before actually finishing — so the caller can tell a
/// refusal apart from a normal completion after streaming ends (`run_task`'s exit-code check).
/// Write `text` to stdout through a locked handle, flushing immediately (matching the streamed,
/// byte-at-a-time output `run_turn_once`'s callbacks need). Task #10 (pi-parity fix): `println!`/
/// `print!` panic internally on any stdout write error — including `EPIPE` — so a downstream consumer
/// that closes its end early (`agent run --json "task" | head -1`) previously crashed this process with
/// "Broken pipe" (exit 101) instead of exiting cleanly, the moment `head` hung up. A closed read end is
/// the normal, expected way a *nix pipeline ends a producer it's done reading from, not a bug in this
/// process to report loudly — matches `serve.rs`'s own writer task, which already exits gracefully on a
/// write failure instead of panicking.
///
/// On `BrokenPipe`, sets `broken_pipe` and trips `cancel` rather than calling `process::exit`
/// directly — this function is called from deep inside `run_turn_once`'s streaming callbacks, which
/// run *before* `run_task`'s `persist_run_tail`; exiting right here used to skip that persist entirely,
/// silently dropping the turn's final (possibly-partial) assistant message from the session store —
/// exactly the loss the Ctrl-C/SIGTERM path already takes care to avoid. Tripping `cancel` instead
/// reuses that identical, already-correct "stop the turn, then still persist" machinery
/// (`agent_core::Agent::run_events_cancellable`'s cooperative cancellation); `run_task` checks
/// `broken_pipe` right after its own unconditional `persist_run_tail` call and exits 0 there instead —
/// same eventual exit code as before, just after the write that used to be skipped. Any other write
/// error (a full disk, a redirected-to-file target that vanished) is swallowed instead, matching this
/// callback's own prior `let _ = ...flush()` convention — exceptionally rare for a terminal/pipe fd, and
/// not worth complicating this hot streaming path over.
fn write_stdout_or_exit(
    text: &str,
    cancel: &agent_core::CancellationToken,
    broken_pipe: &AtomicBool,
) {
    let mut stdout = std::io::stdout().lock();
    let result = stdout
        .write_all(text.as_bytes())
        .and_then(|()| stdout.flush());
    if let Err(e) = result {
        handle_stdout_write_error(&e, cancel, broken_pipe);
    }
}

/// The reaction to a failed stdout write, split out from [`write_stdout_or_exit`] so it's testable
/// without a real broken pipe (stdout itself isn't injectable — the split point is the error, not the
/// write). See `write_stdout_or_exit`'s own doc comment for why a `BrokenPipe` traps into `cancel`
/// rather than exiting on the spot.
fn handle_stdout_write_error(
    e: &std::io::Error,
    cancel: &agent_core::CancellationToken,
    broken_pipe: &AtomicBool,
) {
    if e.kind() == std::io::ErrorKind::BrokenPipe {
        broken_pipe.store(true, Ordering::Relaxed);
        cancel.cancel();
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_turn_once(
    agent: &Agent,
    session: &mut Session,
    json: bool,
    cancel: &agent_core::CancellationToken,
    broken_pipe: &AtomicBool,
    steering: &agent_core::Steering,
    // Whether a `/session` working-memory mount is active — gates both the pre-compaction pressure nudge
    // and the post-compaction recall reminder, mirroring `serve`'s own observer. See
    // `beyond_ai_agent::memory`.
    session_memory_active: bool,
    // The live-prompt size at which to warn of an approaching compaction
    // (`memory::compaction_pressure_point`), fixed for the run (`run`'s model can't change mid-run).
    pressure_point: u32,
) -> agent_core::Result<agent_core::StopReason> {
    // Two `/session` steers, mirroring `serve`'s observer: a *pre*-compaction pressure nudge (checkpoint
    // now, while detail is intact) fired at most once per fill cycle, and a *post*-compaction recall
    // reminder (read it back) fired on the cut. `pressure_armed` disarms on the pressure fire and re-arms
    // on `Compacted`. Both are no-ops without a session mount.
    let pressure_armed = std::sync::atomic::AtomicBool::new(true);
    let session_steers = |ev: &agent_core::AgentEvent| {
        if !session_memory_active {
            return;
        }
        match ev {
            agent_core::AgentEvent::Compacted { .. } => {
                pressure_armed.store(true, std::sync::atomic::Ordering::Relaxed);
                steering.push_steer(agent_core::SteeringMessage::new(
                    beyond_ai_agent::memory::COMPACTION_REMINDER.to_string(),
                    Vec::new(),
                ));
            }
            agent_core::AgentEvent::Stream(agent_core::StreamEvent::Usage(usage)) => {
                if beyond_ai_agent::memory::live_prompt_tokens(usage) > pressure_point
                    && pressure_armed.swap(false, std::sync::atomic::Ordering::Relaxed)
                {
                    steering.push_steer(agent_core::SteeringMessage::new(
                        beyond_ai_agent::memory::PRESSURE_NUDGE.to_string(),
                        Vec::new(),
                    ));
                }
            }
            _ => {}
        }
    };
    let mut stop_reason = agent_core::StopReason::default();
    if json {
        agent
            .run_events_steered(
                session,
                |ev| {
                    if let agent_core::AgentEvent::TurnEnd { stop_reason: r, .. } = &ev {
                        stop_reason = *r;
                    }
                    session_steers(&ev);
                    if let Ok(line) = serde_json::to_string(&ev) {
                        write_stdout_or_exit(&line, cancel, broken_pipe);
                        write_stdout_or_exit("\n", cancel, broken_pipe);
                    }
                },
                cancel.clone(),
                steering.clone(),
            )
            .await?;
        return Ok(stop_reason);
    }
    // Task 1 (pi-parity fix, pass 19): `run_events_steered` directly (rather than the plain
    // `run_cancellable`, which always builds its own default `Steering::new()` internally), so `run`'s
    // own resolved `steering_mode`/`follow_up_mode` (see `run_task`'s construction of `steering`) is
    // actually in effect at agent/session construction time, matching pi's own agent construction in
    // every mode — same `AgentEvent::Stream` filter `Agent::run_cancellable` itself applies internally.
    agent
        .run_events_steered(
            session,
            |ev| {
                session_steers(&ev);
                let agent_core::AgentEvent::Stream(ev) = &ev else {
                    return;
                };
                match ev {
                    StreamEvent::TextDelta { text, .. } => {
                        write_stdout_or_exit(text, cancel, broken_pipe);
                    }
                    StreamEvent::ToolUseStart { name, .. } => {
                        // No trailing newline: `InputJsonDelta` fragments print immediately after, live,
                        // on this same line — a growing preview of the call's arguments as they stream
                        // in, rather than the model appearing to hang until the whole call (and its
                        // result) land.
                        write_stdout_or_exit(&format!("\n[tool: {name}] "), cancel, broken_pipe);
                    }
                    StreamEvent::InputJsonDelta { partial_json, .. } => {
                        write_stdout_or_exit(partial_json, cancel, broken_pipe);
                    }
                    StreamEvent::MessageStop { stop_reason: r } => {
                        stop_reason = *r;
                    }
                    _ => {}
                }
            },
            cancel.clone(),
            steering.clone(),
        )
        .await?;
    write_stdout_or_exit("\n", cancel, broken_pipe);
    Ok(stop_reason)
}

/// A [`agent_core::CheckpointHook`] for one-shot `run`. Unlike `serve`'s channel-based
/// `ChannelCheckpoint` (which forwards through an `mpsc` channel to avoid stalling a `select!` loop
/// reading stdin concurrently), `run` has no concurrent event source to interleave with — a direct
/// blocking append inside the async callback is the simplest correct thing here, not a missing
/// optimization. Persists every mid-run checkpoint incrementally, the same guarantee `serve` gives
/// every session: without this, only the *end* of each whole turn was ever persisted (via
/// `persist_run_tail`, after `run_turn` returns), so a crash mid-turn — after several tool
/// round-trips already ran real commands or edited real files — lost all record of them, with the
/// session file (if any) unable to distinguish that from "nothing happened yet".
struct DirectCheckpoint(Arc<std::sync::Mutex<Option<SessionStore>>>);

#[async_trait::async_trait]
impl agent_core::CheckpointHook for DirectCheckpoint {
    async fn checkpoint(&self, session: &Session) {
        // Best-effort, matching `serve`'s own checkpoint hook: the run itself must not fail just
        // because incremental persistence couldn't (a real I/O failure here is still surfaced —
        // eprintln, not silently swallowed — and the next successful persist, or `persist_run_tail`
        // after the turn ends, will catch up whatever this attempt missed).
        let mut guard = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(store) = guard.as_mut()
            && let Err(e) = store.append_new(&session.messages)
        {
            eprintln!("run: failed to persist checkpoint: {e}");
        }
    }
}

/// Persist whatever's new in `session` since the last append — the tail-covering persist after a
/// whole turn ends (a checkpoint never fires for the turn's own final assistant message; see
/// `agent_core::Agent::run_turn`'s doc comment on where checkpoints land). A no-op when `run` isn't
/// persisting at all (`store`'s inner `Option` is `None`) or when `DirectCheckpoint` already covered
/// everything (`SessionStore::append_new`'s own `messages.len() <= self.persisted` dedup guard).
fn persist_run_tail(
    store: &Arc<std::sync::Mutex<Option<SessionStore>>>,
    session: &Session,
) -> std::io::Result<()> {
    if let Some(store) = store.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        store.append_new(&session.messages)?;
    }
    Ok(())
}

/// Expand an explicit `/skill:name` invocation first (its own prefix, so it can't collide with a
/// `/name` prompt template), then fall through to prompt-template expansion — a no-op on whichever
/// message reaches it unmatched. Mirrors `serve`'s own `"prompt"` handler exactly (see `serve.rs`).
fn expand_message(
    message: &str,
    skills: &[beyond_ai_agent::skills::Skill],
    prompt_templates: &[beyond_ai_agent::prompts::PromptTemplate],
) -> String {
    let message = beyond_ai_agent::skills::expand_if_skill_invocation(message, skills);
    beyond_ai_agent::prompts::expand_if_slash(&message, prompt_templates)
}

/// Whether `id` is safe to embed directly in a filename component. Delegates to the canonical
/// implementation in [`beyond_ai_agent::session_store::is_valid_session_id`] (shared with the
/// WebSocket transport, which validates a client-supplied `?session_id=` the same way).
fn is_valid_session_id(id: &str) -> bool {
    beyond_ai_agent::session_store::is_valid_session_id(id)
}

#[allow(clippy::too_many_arguments)]
async fn run_task(
    tasks: Vec<String>,
    model: Option<String>,
    gateway_url: Option<String>,
    key: Option<String>,
    max_steps: Option<u32>,
    max_tokens: Option<u32>,
    cache_long: bool,
    thinking: Option<u32>,
    reasoning_effort: Option<agent_core::ThinkingLevel>,
    steering_mode: Option<agent_core::QueueMode>,
    follow_up_mode: Option<agent_core::QueueMode>,
    temperature: Option<f64>,
    system_prompt: Option<String>,
    append_system_prompt: Vec<String>,
    trust_project: bool,
    force_untrusted: bool,
    context_window: Option<u32>,
    compaction_reserve_tokens: Option<u32>,
    compaction_keep_recent_tokens: Option<u32>,
    branch_summary_reserve_tokens: Option<u32>,
    no_compaction: bool,
    retry_max_retries: Option<u32>,
    retry_base_delay_ms: Option<u64>,
    retry_max_backoff_ms: Option<u64>,
    no_retry: bool,
    idle_timeout_ms: Option<u64>,
    block_images: bool,
    no_block_images: bool,
    no_image_auto_resize: bool,
    bash_timeout_ms: Option<u64>,
    bash_shell_path: Option<String>,
    bash_command_prefix: Option<String>,
    web_allow_private: bool,
    web_allow_host: Vec<String>,
    web_timeout_ms: Option<u64>,
    exec_url: Option<String>,
    exec_header: Vec<String>,
    exec_cmd: Option<String>,
    tools_allow: Option<Vec<String>>,
    tools_exclude: Option<Vec<String>>,
    no_tools: bool,
    sequential_tools: bool,
    deny_tool: Vec<String>,
    deny_bash_pattern: Vec<String>,
    deny_path: Vec<String>,
    no_skills: bool,
    no_prompt_templates: bool,
    no_context_files: bool,
    extra_skill_paths: Vec<String>,
    extra_prompt_template_paths: Vec<String>,
    name: Option<String>,
    prompt_guidelines: Vec<String>,
    fork: Option<String>,
    session_path: Option<String>,
    session_id: Option<String>,
    continue_session: bool,
    session_dir: Option<String>,
    no_session_persistence: bool,
    memory: Option<String>,
    no_memory: bool,
    no_session_memory: bool,
    export: Option<String>,
    json: bool,
    output_schema: Option<String>,
    output_description: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Fail fast, before touching any files — matches pi's own `--name` validation. Whitespace-only is
    // rejected outright here (a startup argument the operator clearly meant to be meaningful), unlike
    // the RPC `set_session_name` command's "empty clears the title" convention (renaming an
    // already-running session to nothing is a deliberate, different action).
    if let Some(n) = &name
        && n.trim().is_empty()
    {
        return Err("--name requires a non-empty value".into());
    }
    // A malformed `--deny-path` glob must never silently produce a no-op policy — see
    // `ToolPolicy::deny_path`'s doc comment for the fail-open this closes.
    if let Err(e) = ToolPolicy::validate_deny_path_patterns(&deny_path) {
        return Err(e.into());
    }
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
    // own `<cwd>/.claude/settings.json` tier on top of the global one first — see
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
    // `Persistence::model_and_level_at_active` in `serve.rs`). A merely-stored default counts as *not*
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
    let stdin_content = read_stdin_if_piped();
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
    // handler above, for the full set (including `--models`, `serve`-only).
    let bash_shell_path =
        bash_shell_path.or_else(|| stored_settings.default_bash_shell_path.clone());
    // Task #49 (pi-parity fix): `Command::Serve`'s handler (above in this file) checks this upfront and
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
    // tracking, above in this file.
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
             --trust-project or run `agent trust {}` to enable them (a project's own settings.json \
             additionally requires a *persisted* `agent trust`, not just a one-off --trust-project — see \
             `settings::effective_settings_for_cwd`'s doc comment)",
            cwd.display(),
            cwd.display()
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
    // Fail-soft (Task: MCP client support): a server that fails to connect is skipped with a warning,
    // not a fatal error — matches `has_gated_resources`'s own "warn, don't block the run" convention
    // just above. `stored_settings.mcp_servers` is already trust-gated (a project's own
    // `.claude/settings.json` — where a project-tier `mcp_servers` entry would live — only merges in at
    // all when `effective_settings_for_cwd` found `cwd` trusted; the global tier always applies).
    let (mcp_tools, mcp_warnings) =
        tools::mcp::connect_all(stored_settings.mcp_servers.as_deref().unwrap_or(&[])).await;
    for warning in &mcp_warnings {
        eprintln!("warning: {warning}");
    }
    timing.mark("connect mcp servers");
    // Pointing at a remote endpoint swaps *one* thing: where the filesystem tools' I/O lands. The
    // tools, their names, descriptions and schemas are untouched, so the model cannot tell the
    // difference — see `tools::fs`.
    let exec_runner: Option<std::sync::Arc<dyn tools::exec::CommandRunner>> =
        match (&exec_url, &exec_cmd) {
            (Some(url), _) => {
                let mut runner = beyond_ai_agent::exec_endpoint::HttpExecRunner::new(url.clone())
                    .map_err(std::io::Error::other)?;
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
    let mut registry = tools::default_registry_with_config(&tools::ToolConfig {
        bash_timeout_ms,
        bash_shell_path: bash_shell_path.as_deref(),
        bash_command_prefix: bash_command_prefix.as_deref(),
        web_allow_private,
        web_allow_hosts: &web_allow_host,
        web_timeout_ms,
        image_auto_resize,
        mcp_tools: &mcp_tools,
        fs_backend: fs_backend.clone(),
        command_runner: exec_runner.clone(),
        ..tools::ToolConfig::new()
    });
    tools::apply_filter(
        &mut registry,
        tools_allow.as_deref(),
        tools_exclude.as_deref(),
        no_tools,
    );

    // `--session`/`--continue` persist this run (and load prior history to continue it) exactly like
    // `serve`'s own repo/file modes. pi-parity fix: neither given previously kept `run` in-memory-only —
    // pi's own default (no flags at all, including one-shot print-mode) is a persisted, disk-backed
    // session, matching `serve`'s own default repo-mode persistence; only an explicit
    // `--no-session-persistence` now opts back out to the old ephemeral behavior (see the final `None`
    // arm below).
    let cwd_str = cwd.to_string_lossy().into_owned();
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
    // `--session-dir` (matching `serve`'s own flag/env var exactly) redirects the repo root that
    // `--continue` and `--fork` both use, in place of the default `~/.claude/sessions/<encoded-cwd>/`.
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
    let (mut store, mut session) = if let Some(arg) = &fork {
        // `--fork` wins over `--session`/`--continue`: a fork always starts a fresh child session,
        // never reopens one in place, so there is no meaningful way to combine it with either.
        let target = SessionRepo::open(&repo_dir)?;
        let (store, session) = fork_by_arg(arg, &target, &cwd_str, &fork_search_root, usize::MAX)
            .map_err(|e| format!("--fork {arg:?}: {e}"))?;
        (Some(store), session)
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
                        (Some(store), session)
                    } else {
                        let store = SessionStore::create(literal_path, fresh_meta())?;
                        (Some(store), Session::new())
                    }
                } else {
                    let repo = SessionRepo::open(&repo_dir)?;
                    let (store, session) = open_session_by_id(&arg, &repo, &fork_search_root)
                        .map_err(|e| format!("--session {arg:?}: {e}"))?;
                    (Some(store), session)
                }
            }
            // Pure in-memory. `--continue` still overrides `--no-session-persistence` (it always has:
            // asking to continue a persisted session is a direct contradiction of not persisting, and
            // the explicit verb wins), but `--session-id` does not — there it just names the ephemeral
            // session for correlation, exactly as it does in `serve`'s own no-persistence branch.
            None if no_session_persistence && !continue_session => (None, Session::new()),
            // The selection ladder, most specific first — the same one `serve` applies via
            // `serve::SessionSelect`, so the flags mean identically the same thing in both binaries.
            None => {
                let repo = SessionRepo::open(&repo_dir)?;
                let (store, session) = match (session_id.as_deref(), continue_session) {
                    // `--session-id` addresses one session outright: open it, or create it under
                    // exactly that id. It outranks `--continue`, which only *describes* a session
                    // ("whatever ran here last"). It used to be discarded outright whenever any session
                    // already existed for this cwd, which collapsed every distinct id in a shared
                    // directory onto one shared conversation.
                    (Some(id), _) => repo.open_or_create_id(id, &cwd_str, &model)?,
                    (None, true) => repo.resume_latest_or_create(&cwd_str, &model)?,
                    // A bare `run` starts a genuinely new session — persisted, so nothing is lost, but
                    // its own. It briefly reattached to this cwd's most recent session instead, which
                    // meant two shells in one repo drove the same store: `append_new` is count-keyed,
                    // so neither could observe the other's writes and the transcript interleaved into
                    // nonsense. `--continue` is how you ask for the old behavior, and it is now the
                    // only thing that reattaches implicitly.
                    (None, false) => (repo.create(fresh_meta())?, Session::new()),
                };
                (Some(store), session)
            }
        }
    };

    // Persistent, cross-session memory. Resolved here (before the subagent context is built) so the same
    // backend `Arc` is shared with every child — the whole subagent tree reads/writes one durable store.
    // Registered *after* `apply_filter` (like `structured_output`/`subagent`) so a `--tools` allow-list
    // can't strip the agent's memory; `--no-memory`/`--no-tools` opt out. A bad backend fails fast before
    // a model call is billed. The `MEMORY.md` index is injected into the system prompt below.
    let memory_backend: Option<Arc<dyn beyond_ai_agent::memory::MemoryBackend>> =
        if no_memory || no_tools {
            None
        } else {
            let dsn = memory.or_else(|| stored_settings.default_memory_backend.clone());
            Some(
                beyond_ai_agent::memory::open(dsn.as_deref(), &cwd).unwrap_or_else(|e| {
                    eprintln!("{e}");
                    std::process::exit(2);
                }),
            )
        };
    // Build the mount list: durable `/memories`, plus (unless `--no-session-memory`) a per-session
    // `/session` working-memory store — the compaction-surviving scratchpad. `run` is single-session, so
    // the session dir is fixed for the whole process: the `<...>.memory/` sibling of a persisted session
    // file (so `session_store` trashes/restores it together), or an id-keyed tempdir when ephemeral. The
    // session-open block was hoisted above this so `store` is already resolved here. Shared with every
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
    let output_slot = tools::structured_output::OutputSlot::new();
    let wants_structured_output = output_schema.is_some();
    if let Some(arg) = &output_schema {
        let schema = tools::structured_output::load_schema(arg).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(2);
        });
        let tool = tools::structured_output::StructuredOutput::new(
            schema,
            output_description,
            output_slot.clone(),
        )
        .unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(2);
        });
        registry.register(Arc::new(tool));
    }

    // Subagents. A shared write-lock registry (rather than `Agent::new`'s private default) so the parent
    // and every child serialize same-path writes against each other. Registered only when at least one
    // agent definition exists — a `subagent` tool with no agents to call is dead weight in the prompt.
    let write_locks = Arc::new(agent_core::WriteLockRegistry::new());
    beyond_ai_agent::worktree::sweep(&cwd); // reap any worktree orphaned by a previous crash
    let agent_defs = beyond_ai_agent::agents::discover(&cwd, project_trusted);
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
            mcp_tools: mcp_tools.clone(),
            memory_mounts: mounts.clone(),
            tool_cfg: subagent::ChildToolConfig {
                bash_timeout_ms,
                bash_shell_path: bash_shell_path.clone(),
                bash_command_prefix: bash_command_prefix.clone(),
                web_allow_private,
                web_allow_hosts: web_allow_host.clone(),
                web_timeout_ms,
                image_auto_resize,
                // A child acts on the same machine as its parent. Handing down `None` here was a
                // sandbox escape: the model could reach the host simply by delegating to a subagent.
                fs_backend: fs_backend.clone(),
                command_runner: exec_runner.clone(),
            },
            cwd: cwd.clone(),
            project_trusted,
            prompt_guidelines: prompt_guidelines.clone(),
            parent_model: model.clone(),
            parent_cache_key: model.clone(),
            parent_tools,
            deny_tool: deny_tool.clone(),
            deny_bash_pattern: deny_bash_pattern.clone(),
            deny_path: deny_path.clone(),
            child_max_steps: None,
            max_depth: subagent::DEFAULT_MAX_DEPTH,
            // `run` has no client to ask, so it has no interactive gate — only the static `--deny-*`
            // lists, which `ChildHooks` still installs. See `crate::approval`'s module doc.
            approval: None,
        });
        registry.register(Arc::new(subagent::Subagent::new(ctx)));
    }

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
        beyond_ai_agent::resources::default_system_prompt(&registry, &prompt_guidelines);
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
            cwd: &cwd,
            include_context_files: !no_context_files,
            skills: &skills,
            has_read,
            has_todo,
            has_structured_output,
            has_memory,
            memory_sections: &memory_sections,
            project_trusted,
            agents: &agent_defs,
        },
    );
    timing.mark("build system prompt");

    // `--name`, applied uniformly across every path above (mirrors `serve`'s own startup check) —
    // only for a genuinely fresh session (no messages, no title yet). `--continue`'s `resume_or_create`
    // branch above mints its own fresh `SessionMeta` internally when no cwd match exists, bypassing the
    // `fresh_meta` closure other branches use, so this was previously the one path `--name` silently
    // never reached even when it *did* open a brand-new session.
    if let Some(name) = &name
        && session.messages.is_empty()
        && let Some(store) = &mut store
        && store.meta().title.is_none()
    {
        store.set_title(name)?;
    }
    let meta = store
        .as_ref()
        .map(|s| s.meta().clone())
        .unwrap_or_else(fresh_meta);
    // Prefer the session's own persisted model over the CLI-resolved default when reopening an
    // existing `--session`/`--continue` session and the operator didn't explicitly pass `--model` —
    // the same bug class `switch_session` had in `serve.rs` (see `Persistence::model_and_level_at_active`
    // there): without this, reattaching to a session last driven on `gpt-5` without re-passing `--model`
    // silently continued it on whatever `DEFAULT_MODEL` resolves to instead, no warning. For a
    // genuinely fresh session `meta.model` already equals `model` (`fresh_meta` seeds it from the same
    // value), so this is a no-op there.
    let model = if model_explicit {
        model
    } else {
        meta.model.clone()
    };
    // A genuinely fresh session's `cwd` always equals the current one (just stamped by `fresh_meta`),
    // so this only fires for a reopened `--session`/`--continue` session — the recorded directory was
    // moved/deleted, or this process simply isn't running where the session was created (e.g. a
    // `--session-dir` shared across projects). `serve` already surfaces the identical check as
    // `cwd_stale` on its RPC responses; `run` had no equivalent at all, matching pi's
    // `MissingSessionCwdError` guard. Informational, not fatal — the tools underneath will surface
    // their own, more specific errors if this actually matters for the task at hand.
    if serve::cwd_is_stale(&meta.cwd, &cwd) {
        eprintln!(
            "warning: this session's recorded working directory ({}) does not match the current one \
             ({}); tools will operate against the current directory",
            meta.cwd,
            cwd.display()
        );
    }
    timing.mark("open session");
    timing.print();

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
    // Shared with `DirectCheckpoint` below (built before `agent`, so the hook can be installed at
    // construction) so a long multi-step turn (many tool round-trips) is persisted incrementally —
    // the same guarantee `serve` gives every session. Without this, only the *end* of each whole
    // turn was ever persisted (the `persist_run_tail` calls below, after `run_turn` returns), so a
    // crash mid-turn — after several tool round-trips already ran real commands/edited real files —
    // lost all record of them with no session trace at all.
    let store = Arc::new(std::sync::Mutex::new(store));
    // Matches `serve`'s own `build_agent`: defaults to the model's own capability-table context
    // window when `--context-window` isn't given, then applies the reserve/keep-recent overrides.
    let mut compaction = agent_core::CompactionConfig {
        context_window: context_window
            .unwrap_or_else(|| agent_core::capabilities(&model).context_window),
        ..agent_core::CompactionConfig::default()
    };
    if let Some(reserve) = compaction_reserve_tokens {
        compaction.reserve_tokens = reserve;
    }
    if let Some(keep_recent) = compaction_keep_recent_tokens {
        compaction.keep_recent_tokens = keep_recent;
    }
    if no_compaction {
        compaction.enabled = false;
    }
    // The live-prompt size at which to warn the model that a compaction is approaching (see
    // `run_turn_once`), derived from the same resolved window/reserve the agent will compact against.
    let pressure_point = beyond_ai_agent::memory::compaction_pressure_point(
        compaction.context_window,
        compaction.reserve_tokens,
    );
    // Captured before each is moved into the builder chain below — `Agent` exposes no getter for
    // either back, and `run --export`'s own call to `export_html_full` further down needs the exact
    // system prompt/tool set this run actually used, not a recomputed (and possibly out-of-sync) guess.
    let tool_defs = registry.definitions();
    let system_for_export = system.clone();
    let mut agent = Agent::new(Arc::new(client), model.clone())
        .with_tools(registry)
        .with_system(system)
        .with_max_steps(max_steps)
        .with_compaction(compaction)
        .with_cache_long(cache_long)
        .with_sequential_tools(sequential_tools)
        .with_cache_key(meta.id.clone())
        // Shared with every subagent (see the `write_locks` construction above) so the parent's own
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
    let policy = ToolPolicy::from_lists(&deny_tool, &deny_bash_pattern, &deny_path);
    if !policy.is_empty() {
        agent = agent.with_hooks(Arc::new(policy));
    }

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
    // unlike `serve.rs`'s own `shutdown_cause`) since this is genuinely shared across a task boundary:
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
        }
    });
    // Set by `write_stdout_or_exit` on a broken stdout pipe (a downstream consumer that closed its
    // read end early, e.g. `agent run --json "task" | head -1`) — trips `cancel` the same as a real
    // shutdown signal so the turn stops cooperatively instead of `process::exit`ing from deep inside
    // a streaming callback, which used to skip the `persist_run_tail` calls below entirely and
    // silently drop the turn's final (possibly-partial) assistant message. Checked right after each
    // `persist_run_tail`, below, so the exit still happens (with the same code 0 as before) but only
    // once that write has actually landed.
    let broken_pipe = AtomicBool::new(false);

    let initial_message = expand_message(&initial_message, &skills, &prompt_templates);
    if initial_images.is_empty() {
        session.user(initial_message);
    } else {
        session.push(agent_core::Message::user_with_images(
            initial_message,
            initial_images,
        ));
    }
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
    )
    .await;
    // Persist whatever's in `session` regardless of outcome: `run_events_cancellable` mutates
    // `session` in place as it streams, so a cancelled turn still leaves behind whatever
    // assistant/tool content had already landed — the same partial-content guarantee `serve` gives
    // every session, not just the happy path. `DirectCheckpoint` already covers most of this
    // incrementally, but the turn's own tail (its final, possibly-partial assistant message) is
    // only ever captured here.
    persist_run_tail(&store, &session)?;
    if broken_pipe.load(Ordering::Relaxed) {
        // Reached *because* `write_stdout_or_exit` tripped `cancel`, so the same in-flight bash tool
        // future a signal would have dropped has just been dropped here too — with the same detached
        // `GroupKillGuard` cleanup thread still running, and the same `process::exit` about to kill it
        // mid-`kill`. See `unwrap_turn_result`, which pays this toll for the signal path.
        #[cfg(unix)]
        tools::exec::wait_for_pending_group_kills(std::time::Duration::from_secs(2));
        std::process::exit(0);
    }
    let mut stop_reason = unwrap_turn_result(turn_result, &shutdown_cause)?;
    for message in messages {
        session.user(expand_message(&message, &skills, &prompt_templates));
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
        )
        .await;
        persist_run_tail(&store, &session)?;
        if broken_pipe.load(Ordering::Relaxed) {
            // Same broken-pipe cancellation as the first turn's exit above — drain the pending
            // process-group kills before tearing their threads down.
            #[cfg(unix)]
            tools::exec::wait_for_pending_group_kills(std::time::Duration::from_secs(2));
            std::process::exit(0);
        }
        stop_reason = unwrap_turn_result(turn_result, &shutdown_cause)?;
    }

    if let Some(export) = export {
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
            Some(&export),
        ) {
            Ok(path) => eprintln!("[exported transcript to {}]", path.display()),
            Err(e) => eprintln!("[failed to export transcript: {e}]"),
        }
    }

    eprintln!(
        "[done in {} step(s); {} in / {} out tokens]",
        session.steps, session.input_tokens, session.output_tokens
    );

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
            }
            None => {
                // The contract the run was started with was never met. Exiting 0 here would be
                // indistinguishable from success to a script that pipes stdout into `jq`.
                eprintln!(
                    "[no structured output: the model ended the run without calling `{}`]",
                    tools::structured_output::NAME
                );
                std::process::exit(1);
            }
        }
    }

    // Text mode has no other failure signal a script/CI caller could key off of — a refusal would
    // otherwise still exit 0, indistinguishable from a normal completion, unless the last turn's
    // stop reason is checked explicitly here. JSON mode already carries `stop_reason` on every
    // `TurnEnd` event in its own output stream, so it's unaffected either way — see
    // `text_mode_failure_message`'s own doc comment for the exact contract (including why `Aborted` is
    // checked too, defensively, even though it's currently unreachable here).
    if let Some(message) = text_mode_failure_message(json, stop_reason) {
        eprintln!("{message}");
        std::process::exit(1);
    }
    Ok(())
}

/// The diagnostic to print and exit(1) on for a finished run's final stop reason, or `None` to exit
/// 0 normally. `None` unconditionally in JSON mode: `stop_reason` is already on every `TurnEnd` event
/// in that mode's own output stream, so its exit code stays reserved for a genuine process failure.
///
/// In text mode, a refusal would otherwise still exit 0, indistinguishable from a normal completion —
/// matches pi's own print-mode, which treats a refusal (folded into its generic "error" stop reason
/// there, unlike this crate's distinct `Refusal` variant) the same way. `Aborted` is checked
/// defensively alongside it: `unwrap_turn_result` already exits with the matching signal code (or 1)
/// on every currently-reachable cancellation path (`Err(Error::Cancelled)`, e.g. `ShutdownSignal`-triggered SIGTERM/SIGHUP/Ctrl-C —
/// see `agent_core::Agent::run_events_cancellable`'s doc comment, which guarantees cancellation always
/// surfaces that way, never as an `Ok(..)` carrying `Aborted`), so this arm is currently unreachable
/// from `run_task` — but a mid-stream cancellation genuinely can produce `Ok(Turn { stop_reason:
/// Aborted, .. })` at lower layers (see `Agent::run_turn_once`'s own doc comment), just not through
/// any path this binary's own `run_turn_once` currently reaches. Handling it here too costs nothing
/// and closes the gap outright if that internal contract ever changes, rather than silently exiting 0
/// on what would still be an interrupted, incomplete run.
fn text_mode_failure_message(
    json: bool,
    stop_reason: agent_core::StopReason,
) -> Option<&'static str> {
    if json {
        return None;
    }
    match stop_reason {
        agent_core::StopReason::Refusal => Some("[refused]"),
        agent_core::StopReason::Aborted => Some("[cancelled]"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_core::Error;
    use agent_core::mock::{MockTransport, turn};

    #[test]
    fn a_broken_pipe_trips_cancel_and_the_flag_instead_of_exiting_on_the_spot() {
        // Regression: `write_stdout_or_exit` used to call `process::exit(0)` directly on a broken
        // pipe, which — because it's invoked from deep inside `run_turn_once`'s streaming callbacks —
        // ran *before* `run_task`'s `persist_run_tail`, silently dropping the turn's final assistant
        // message. The fix routes a broken pipe through the same cooperative-cancellation machinery
        // Ctrl-C/SIGTERM already use, so `run_task` can persist first and exit afterward. This exercises
        // `handle_stdout_write_error` directly (the reaction, split out from the unavoidably real
        // `std::io::stdout()` write in `write_stdout_or_exit`) rather than an actual broken pipe.
        let cancel = agent_core::CancellationToken::new();
        let broken_pipe = AtomicBool::new(false);
        assert!(!cancel.is_cancelled());

        handle_stdout_write_error(
            &std::io::Error::from(std::io::ErrorKind::BrokenPipe),
            &cancel,
            &broken_pipe,
        );

        assert!(
            broken_pipe.load(Ordering::Relaxed),
            "a broken pipe must set the flag `run_task` checks after persisting"
        );
        assert!(
            cancel.is_cancelled(),
            "a broken pipe must trip `cancel` so the turn stops cooperatively, not via a bare exit"
        );
    }

    #[test]
    fn a_non_broken_pipe_write_error_does_not_trip_cancel_or_the_flag() {
        // Any other stdout write error (full disk, redirected-to-file target vanished) is swallowed —
        // matches the callback's own prior `let _ = ...flush()` convention — so neither `cancel` nor
        // `broken_pipe` should be touched.
        let cancel = agent_core::CancellationToken::new();
        let broken_pipe = AtomicBool::new(false);

        handle_stdout_write_error(
            &std::io::Error::from(std::io::ErrorKind::Other),
            &cancel,
            &broken_pipe,
        );

        assert!(!broken_pipe.load(Ordering::Relaxed));
        assert!(!cancel.is_cancelled());
    }

    #[test]
    fn resolve_prompt_input_reads_an_existing_files_contents() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prompt.txt");
        std::fs::write(&path, "FILE CONTENTS").unwrap();
        assert_eq!(
            resolve_prompt_input(path.to_str().unwrap()),
            "FILE CONTENTS"
        );
    }

    #[test]
    fn resolve_prompt_input_treats_a_non_existent_path_as_a_literal_string() {
        assert_eq!(
            resolve_prompt_input("this is not a real file on disk"),
            "this is not a real file on disk"
        );
    }

    #[test]
    fn resolve_prompt_input_treats_a_directory_as_a_literal_string_not_a_file() {
        let dir = tempfile::tempdir().unwrap();
        // `is_file()` is false for a directory — must fall through to the literal-string path rather
        // than erroring on a directory that happens to share a name with the input.
        assert_eq!(
            resolve_prompt_input(dir.path().to_str().unwrap()),
            dir.path().to_str().unwrap()
        );
    }

    #[test]
    fn thinking_and_reasoning_effort_help_text_cross_reference_each_other() {
        // Fix 3 (pi-parity, low priority): beyond's `--thinking` (a raw token-budget override) and
        // pi's own `--thinking <level>` (off/minimal/low/medium/high/xhigh) share a name but mean
        // different things — beyond's portable-level equivalent is `--reasoning-effort`. No behavior
        // change here, just making the naming collision self-explanatory in `--help`.
        let help = Cli::render_help(Cli::command(), true).unwrap_or_default();
        for sub in ["run", "serve"] {
            let sub_cmd = Cli::command()
                .subcommands
                .iter()
                .copied()
                .find(|c| c.name == sub)
                .unwrap_or_else(|| panic!("no {sub} subcommand in: {help}"));
            let sub_help = Cli::render_help(sub_cmd, true).unwrap_or_default();
            let normalized: String = sub_help.split_whitespace().collect::<Vec<_>>().join(" ");
            assert!(
                normalized.contains("see `--reasoning-effort`"),
                "{sub} --thinking help must reference --reasoning-effort: {sub_help}"
            );
            assert!(
                normalized.contains("see `--thinking`"),
                "{sub} --reasoning-effort help must reference --thinking: {sub_help}"
            );
        }
    }

    /// In-process argv parse cost only (no process start, no help rendering). Printed under
    /// `--nocapture` so a release-mode run can be compared against process-level hyperfine numbers:
    /// process startup for this binary is ~2 ms either way; the parse itself is microseconds.
    #[test]
    fn cli_parse_in_process_microbench() {
        use std::hint::black_box;
        use std::time::Instant;

        let cases: &[&[&str]] = &[
            &["beyond-ai-agent", "tools"],
            &["beyond-ai-agent", "run", "--model", "claude-opus-4-8", "hi"],
            &[
                "beyond-ai-agent",
                "serve",
                "--upstream-http2",
                "off",
                "--tools",
                "bash,read,write",
            ],
            &[
                "beyond-ai-agent",
                "settings",
                "--model",
                "claude-sonnet-4-5",
            ],
        ];
        const WARMUP: u32 = 200;
        const ITERS: u32 = 5_000;

        for argv in cases {
            let os: Vec<&OsStr> = argv.iter().map(OsStr::new).collect();
            for _ in 0..WARMUP {
                let _ = black_box(Cli::try_parse_from(&os));
            }
            let start = Instant::now();
            for _ in 0..ITERS {
                let _ = black_box(Cli::try_parse_from(&os));
            }
            let elapsed = start.elapsed();
            let per = elapsed / ITERS;
            eprintln!(
                "cli_parse_in_process: {argv:?} → {per:?}/parse ({} over {ITERS})",
                argv.join(" "),
            );
            // Sanity: a successful parse path must stay well under a millisecond in-process.
            assert!(
                per.as_micros() < 1_000,
                "unexpectedly slow parse for {argv:?}: {per:?}"
            );
        }
    }

    #[test]
    fn is_valid_session_id_accepts_ordinary_ids() {
        assert!(is_valid_session_id("abc123"));
        assert!(is_valid_session_id("a"));
        assert!(is_valid_session_id("my-session_id.v2"));
        assert!(is_valid_session_id("18be91b27c544ffa-19b6811ee53adb5c-0"));
    }

    #[test]
    fn is_valid_session_id_rejects_path_traversal_and_separators() {
        // pi-parity fix: this id is embedded directly into a filename component with no other
        // sanitization — must reject anything that could resolve outside the sessions directory.
        assert!(!is_valid_session_id("../../../tmp/pwned/evil"));
        assert!(!is_valid_session_id("/etc/passwd"));
        assert!(!is_valid_session_id("foo/bar"));
        assert!(!is_valid_session_id("foo\\bar"));
    }

    #[test]
    fn is_valid_session_id_rejects_empty_and_edge_punctuation() {
        assert!(!is_valid_session_id(""));
        assert!(!is_valid_session_id(".hidden"));
        assert!(!is_valid_session_id("trailing-"));
        assert!(!is_valid_session_id("-leading"));
    }

    #[test]
    fn fuzzy_match_finds_a_non_contiguous_subsequence_a_substring_check_would_miss() {
        // Task #51: "sn5" is a valid in-order subsequence of "claude-sonnet-4-5" (s..n..5) even though
        // it's never a literal substring of it.
        assert!(fuzzy_match("sn5", "claude-sonnet-4-5").is_some());
        assert!(
            !"claude-sonnet-4-5".contains("sn5"),
            "sanity: not a real substring"
        );
        assert!(
            fuzzy_match("sn5", "gpt-5-mini").is_none(),
            "no 's' at all in this candidate"
        );
    }

    #[test]
    fn fuzzy_match_is_case_insensitive_and_rejects_out_of_order_characters() {
        assert!(fuzzy_match("SONNET", "claude-sonnet-4-5").is_some());
        assert!(
            fuzzy_match("5-sonnet", "claude-sonnet-4-5").is_none(),
            "the query's own character order must still be respected"
        );
    }

    #[test]
    fn fuzzy_match_scores_a_consecutive_word_boundary_match_better_than_a_scattered_one() {
        // "sonnet" matches "claude-sonnet-4-5" as one consecutive run starting right at a word
        // boundary; the same characters also appear scattered (worse) in a longer candidate — the
        // consecutive, word-boundary-aligned match must score lower (better).
        let tight = fuzzy_match("sonnet", "claude-sonnet-4-5").unwrap();
        let scattered = fuzzy_match("sonnet", "s-o-n-n-e-t-mixed-up-id").unwrap();
        assert!(
            tight < scattered,
            "tight={tight} scattered={scattered}: a consecutive run should score better"
        );
    }

    #[test]
    fn text_mode_failure_message_flags_a_refusal_and_an_aborted_stop_reason_as_failures() {
        // pi-parity fix: `Aborted` previously wasn't checked at all alongside the existing `Refusal`
        // check — defensive, since `unwrap_turn_result` already exits with the matching signal code
        // (or 1) on every currently-reachable cancellation path, but this closes the gap outright if
        // that internal contract ever changes.
        assert_eq!(
            text_mode_failure_message(false, agent_core::StopReason::Refusal),
            Some("[refused]")
        );
        assert_eq!(
            text_mode_failure_message(false, agent_core::StopReason::Aborted),
            Some("[cancelled]")
        );
    }

    #[test]
    fn text_mode_failure_message_is_none_for_a_normal_end_of_turn() {
        assert_eq!(
            text_mode_failure_message(false, agent_core::StopReason::EndTurn),
            None
        );
        assert_eq!(
            text_mode_failure_message(false, agent_core::StopReason::ToolUse),
            None
        );
    }

    #[test]
    fn text_mode_failure_message_is_always_none_in_json_mode() {
        assert_eq!(
            text_mode_failure_message(true, agent_core::StopReason::Refusal),
            None
        );
        assert_eq!(
            text_mode_failure_message(true, agent_core::StopReason::Aborted),
            None
        );
    }

    #[tokio::test]
    async fn direct_checkpoint_persists_incrementally_during_a_multi_tool_round_trip_run() {
        // Two tool round-trips, then a final text turn. `DirectCheckpoint` must have already written
        // both round-trips' worth of messages to disk by the time they happen — not just once, at the
        // very end, via `persist_run_tail` (which only ever runs after `run_turn` returns `Ok`, and so
        // never covers a crash or hard failure partway through a long multi-step turn). Proven here by
        // reading the session file back with a *fresh* `SessionStore::open` before `run_turn` even
        // returns — a completely independent read path from anything `run_task`'s own bookkeeping
        // could accidentally make look right.
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("target.txt");
        std::fs::write(&target, "hello").unwrap();
        let session_path = dir.path().join("s.jsonl");
        let store =
            SessionStore::create(session_path.clone(), SessionMeta::new("/w", "claude-test"))
                .unwrap();
        let store = Arc::new(std::sync::Mutex::new(Some(store)));

        let read_args = serde_json::json!({ "path": target.to_str().unwrap() }).to_string();
        let transport = Arc::new(MockTransport::new(vec![
            turn::tool_call("1", "read", &read_args),
            turn::tool_call("2", "read", &read_args),
            turn::text("done"),
        ]));
        let agent = Agent::new(transport, "claude-test")
            .with_tools(tools::default_registry())
            .with_checkpoint_hook(Arc::new(DirectCheckpoint(store.clone())));

        let mut session = Session::new();
        session.user("read the file twice");
        run_turn(
            &agent,
            &mut session,
            false,
            &agent_core::CancellationToken::new(),
            &beyond_ai_agent::retry::RunRetryPolicy::default(),
            &AtomicBool::new(false),
            &agent_core::Steering::new(),
            false,
            u32::MAX,
        )
        .await
        .unwrap();

        // Read independently of `store` (which the test itself still holds a live handle to) —
        // exactly what a process restarting after a crash would do.
        let (_, disk_session) = SessionStore::open(session_path).unwrap();
        assert!(
            disk_session.messages.len() >= 4,
            "checkpoints during the run must have persisted the tool round-trips that already \
             happened, not just whatever `persist_run_tail` would add after the fact: {:?}",
            disk_session.messages
        );
    }

    #[tokio::test]
    async fn cancelling_a_real_write_call_mid_flight_still_serializes_a_second_real_write_behind_it()
     {
        // pi: file-mutation-queue.test.ts, "keeps write queue locked while an aborted write is still
        // in flight" — the exact end-to-end scenario: while a write is cancelled but still conceptually
        // "in flight", a second write to the *same path*, dispatched concurrently on a completely
        // separate `Agent`/session, must not even *start* until the first's lock is genuinely released
        // — and must end up as the file's final content (no interleaving/corruption). `write_lock.rs`'s
        // own unit tests already prove the registry's `Drop`-tied release in isolation, with a synthetic
        // critical section (`aborting_a_lock_holder_mid_critical_section_releases_the_lock_only_at_that_
        // point`); this drives the same guarantee through the *real* `write` tool
        // (`beyond_ai_agent::tools::write::Write`) and the real `agent_core::Agent`
        // dispatch/`write_target`-grouping path (`agent.rs`'s `group_runs`), across two independent,
        // genuinely concurrent `Agent::run` calls sharing one `WriteLockRegistry` — the same
        // two-runs-sharing-a-registry shape `same_write_target_serializes_across_two_agent_runs_sharing_
        // a_registry` (agent-core's own test module) uses for the non-cancellation version of this,
        // with cancellation of the first layered in.
        //
        // Both runs (plus a third "controller" branch) are driven concurrently via one `tokio::join!`
        // — `run_events_cancellable`'s sink is a boxed `dyn FnMut`, so the resulting future isn't
        // `Send` and can't be `tokio::spawn`ed directly; `join!` polls all three cooperatively on this
        // task instead, which is all genuine interleaving needs here. The controller branch is what
        // makes this a real concurrency proof rather than two runs that merely happen to execute in a
        // safe order: it waits until A has demonstrably started (and so acquired the lock), asserts B's
        // own `run` has *not* started within a generous window while A is still holding the lock and
        // uncancelled, and only then triggers cancellation. (An earlier, sequential-only version of
        // this test — run A to completion, then run B — passed even with the cross-run lock acquisition
        // deleted outright, since nothing was left to race by the time B started; this shape doesn't:
        // deleting the lock acquisition makes the "B must still be blocked" assertion below fail.)
        //
        // Real `Write::run` has no internal `.await` at all (`write_atomic` is synchronous fs I/O — see
        // its doc comment): so, unlike pi's Node `fs.writeFile`, a task cancellation can never land
        // *mid*-write for this tool; it only ever lands strictly before the mutation starts or strictly
        // after it's already committed (a *stronger* guarantee than pi's own — a cancelled write here
        // can never leave a half-written file, full stop). `GatedWrite` below simulates the "genuinely
        // still in flight" window pi's async write creates by delegating `write_target` to the real tool
        // (so grouping/locking uses the real path-normalization logic) but gating entry to the real
        // mutation behind a signal the tool itself never releases — cancellation is therefore the *only*
        // way that call ever ends.
        use agent_core::{CancellationToken, Error, ToolOutput, ToolRegistry, WriteLockRegistry};
        use async_trait::async_trait;
        use std::time::Duration;

        /// Delegates schema/`write_target` to the real `write` tool, but signals `started` (via a
        /// `watch` — not `Notify`, since two independent branches below each need to observe this same
        /// transition) and then blocks forever instead of ever performing the real mutation —
        /// cancellation is the only way this call ends, so its lock is held for as long as the run is
        /// willing to wait.
        struct GatedWrite {
            started: tokio::sync::watch::Sender<bool>,
            inner: tools::write::Write,
        }
        #[async_trait]
        impl agent_core::tool::Tool for GatedWrite {
            fn name(&self) -> &str {
                "write"
            }
            fn description(&self) -> &str {
                "gated write (test double delegating to the real `write` tool's schema/write_target)"
            }
            fn input_schema(&self) -> serde_json::Value {
                self.inner.input_schema()
            }
            fn write_target(&self, input: &serde_json::Value) -> Option<String> {
                self.inner.write_target(input)
            }
            async fn run(
                &self,
                _input: serde_json::Value,
            ) -> std::result::Result<ToolOutput, agent_core::ToolError> {
                let _ = self.started.send(true);
                futures::future::pending::<()>().await;
                unreachable!(
                    "cancellation must have ended this call before the pending future ever resolves"
                )
            }
        }

        /// The real `write` tool, plus a `started` signal fired the instant its `run` actually begins
        /// — i.e. the instant it has already acquired the write lock — so the test can tell "blocked,
        /// still waiting on the lock" apart from "running".
        ///
        /// Holds the real tool rather than constructing one per call: `Tool::description` returns a
        /// `&str` borrowed from `&self`, so it has nothing to borrow from a temporary.
        struct ObservedWrite {
            started: Arc<tokio::sync::Notify>,
            inner: tools::write::Write,
        }
        #[async_trait]
        impl agent_core::tool::Tool for ObservedWrite {
            fn name(&self) -> &str {
                "write"
            }
            fn description(&self) -> &str {
                self.inner.description()
            }
            fn input_schema(&self) -> serde_json::Value {
                self.inner.input_schema()
            }
            fn write_target(&self, input: &serde_json::Value) -> Option<String> {
                self.inner.write_target(input)
            }
            async fn run(
                &self,
                input: serde_json::Value,
            ) -> std::result::Result<ToolOutput, agent_core::ToolError> {
                self.started.notify_one();
                self.inner.run(input).await
            }
        }

        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("shared.txt");
        let registry = Arc::new(WriteLockRegistry::new());

        // Run A: the gated write. Its guard is held until cancellation, and only ever fires
        // `a_started` from inside `Tool::run` — i.e. strictly *after* the group has already acquired
        // `target`'s write lock (see `agent.rs`'s `group_runs`).
        let (a_started_tx, a_started_rx) = tokio::sync::watch::channel(false);
        let mut tools_a = ToolRegistry::new();
        tools_a.register(Arc::new(GatedWrite {
            started: a_started_tx,
            inner: tools::write::Write::default(),
        }));
        let write_args_a =
            serde_json::json!({ "path": target.to_str().unwrap(), "content": "first\n" })
                .to_string();
        let mock_a = Arc::new(MockTransport::new(vec![
            turn::tool_call("1", "write", &write_args_a),
            turn::text("done"),
        ]));
        let agent_a = Agent::new(mock_a, "claude-test")
            .with_tools(tools_a)
            .with_write_locks(registry.clone());
        let cancel = CancellationToken::new();
        let cancel_for_a = cancel.clone();
        let a_run = async move {
            let mut session_a = Session::new();
            session_a.user("write the first file");
            agent_a
                .run_events_cancellable(&mut session_a, |_| {}, cancel_for_a)
                .await
        };

        // Run B: the real `write` tool, targeting the *same* path, on a completely separate `Agent`,
        // sharing the same registry — but its dispatch doesn't even begin until `a_started` fires, so
        // it can only ever race the lock *after* A has demonstrably already acquired it (never before).
        let b_started = Arc::new(tokio::sync::Notify::new());
        let mut tools_b = ToolRegistry::new();
        tools_b.register(Arc::new(ObservedWrite {
            started: b_started.clone(),
            inner: tools::write::Write::default(),
        }));
        let write_args_b =
            serde_json::json!({ "path": target.to_str().unwrap(), "content": "second\n" })
                .to_string();
        let mock_b = Arc::new(MockTransport::new(vec![
            turn::tool_call("2", "write", &write_args_b),
            turn::text("done"),
        ]));
        let agent_b = Agent::new(mock_b, "claude-test")
            .with_tools(tools_b)
            .with_write_locks(registry.clone());
        let mut a_started_rx_for_b = a_started_rx.clone();
        let b_run = async move {
            let _ = a_started_rx_for_b.changed().await;
            let mut session_b = Session::new();
            session_b.user("write the second file");
            agent_b.run(&mut session_b, |_| {}).await
        };

        // The controller: the crux of the test. Once A has genuinely started (and so holds the lock),
        // confirm B has *not* — a generous window, well past anything scheduling jitter could explain
        // — then only trigger cancellation once that's confirmed.
        let mut a_started_rx_for_controller = a_started_rx;
        let target_for_controller = target.clone();
        let controller = async move {
            let _ = a_started_rx_for_controller.changed().await;
            assert!(
                tokio::time::timeout(Duration::from_millis(200), b_started.notified())
                    .await
                    .is_err(),
                "the second write must not start while the first call's lock is still held and \
                 uncancelled"
            );
            assert!(
                !target_for_controller.exists(),
                "neither write has actually run yet — the file must not exist"
            );
            cancel.cancel();
        };

        let (result_a, result_b, ()) = tokio::time::timeout(
            Duration::from_secs(5),
            futures::future::join3(a_run, b_run, controller),
        )
        .await
        .expect("the whole scenario must not deadlock");

        assert!(
            matches!(result_a, Err(Error::Cancelled)),
            "got: {result_a:?}"
        );
        // B must have completed cleanly once A's (now-released) lock let it proceed.
        result_b.expect("run B's own result must be Ok");

        assert_eq!(
            std::fs::read_to_string(&target).unwrap(),
            "second\n",
            "the second, real write must be the file's final content — no interleaving/corruption \
             from the cancelled first call"
        );
    }

    /// `agent_core::Agent::run_turn`'s own within-turn retry exhausts after this many *failed*
    /// attempts (`agent.rs::MAX_MID_STREAM_RETRIES`) before propagating the error to the caller — the
    /// point our own whole-run retry (`run_turn`, this file) is meant to catch. Scripting exactly this
    /// many failing turns, then a real one, exercises our layer specifically without depending on
    /// exactly *why* the inner layer gave up.
    const INNER_RETRY_ATTEMPTS: usize = 4;

    #[tokio::test]
    async fn run_turn_recovers_from_a_whole_run_transient_failure() {
        // Every attempt agent_core's own mid-stream retry makes fails with a retryable error (matches
        // `is_retryable_mid_stream`'s "overloaded" check), exhausting it — the resulting `Err` is
        // exactly what propagates out to `agent.run(...)` inside `run_turn_once`. Our new whole-run
        // wrapper (`run_turn`) must catch that and retry the whole call again, which finally succeeds.
        let mut turns: Vec<Vec<Result<StreamEvent, Error>>> = (0..INNER_RETRY_ATTEMPTS)
            .map(|_| vec![Err(Error::Transport("overloaded_error: overloaded".into()))])
            .collect();
        turns.push(turn::text("recovered").into_iter().map(Ok).collect());
        let transport = std::sync::Arc::new(MockTransport::scripted(turns));
        let agent = Agent::new(transport.clone(), "claude-test");
        let mut session = Session::new();
        session.user("hi");

        run_turn(
            &agent,
            &mut session,
            false,
            &agent_core::CancellationToken::new(),
            &beyond_ai_agent::retry::RunRetryPolicy::default(),
            &AtomicBool::new(false),
            &agent_core::Steering::new(),
            false,
            u32::MAX,
        )
        .await
        .expect("the whole-run retry must recover once a real turn is finally scripted");

        // agent_core's own internal retry consumed the 4 failing turns; ours consumed the 5th
        // (successful) one on its first — and only necessary — retry.
        assert_eq!(transport.calls(), INNER_RETRY_ATTEMPTS + 1);
        let dump = format!("{:?}", session.messages);
        assert!(
            dump.contains("recovered"),
            "session must contain the recovered reply: {dump}"
        );
    }

    #[tokio::test]
    async fn run_turn_gives_up_after_max_run_retries_of_whole_run_failures() {
        // Every single attempt (both agent_core's own retries AND every one of our whole-run retries)
        // fails — after `retry::MAX_RUN_RETRIES` whole-run retries, `run_turn` must give up and
        // propagate the error rather than retrying forever.
        let total_attempts =
            (beyond_ai_agent::retry::MAX_RUN_RETRIES as usize + 1) * INNER_RETRY_ATTEMPTS;
        let turns: Vec<Vec<Result<StreamEvent, Error>>> = (0..total_attempts)
            .map(|_| vec![Err(Error::Transport("overloaded_error: overloaded".into()))])
            .collect();
        let transport = std::sync::Arc::new(MockTransport::scripted(turns));
        let agent = Agent::new(transport.clone(), "claude-test");
        let mut session = Session::new();
        session.user("hi");

        let err = run_turn(
            &agent,
            &mut session,
            false,
            &agent_core::CancellationToken::new(),
            &beyond_ai_agent::retry::RunRetryPolicy::default(),
            &AtomicBool::new(false),
            &agent_core::Steering::new(),
            false,
            u32::MAX,
        )
        .await
        .expect_err("must eventually give up, not retry forever");
        assert!(matches!(err, Error::Transport(_)));
        assert_eq!(transport.calls(), total_attempts);
    }

    #[tokio::test]
    async fn a_no_retry_disabled_policy_never_retries_a_whole_run_transient_failure() {
        // Task #52 (pi-parity fix): `--no-retry` must reach `run_turn`'s own whole-run retry gate and
        // actually stop it from retrying — not just theoretically compute `max_retries: 0` somewhere
        // unused. Same failure shape as `run_turn_recovers_from_a_whole_run_transient_failure` above
        // (which proves the *enabled* case recovers), but built via
        // `RunRetryPolicy::from_overrides_with_enabled(false, ..)` (what `--no-retry` actually
        // constructs in `main`) with a real would-have-recovered turn scripted right after the failing
        // ones — proving the recovery opportunity was skipped specifically because retry is disabled,
        // not because none was scripted.
        let mut turns: Vec<Vec<Result<StreamEvent, Error>>> = (0..INNER_RETRY_ATTEMPTS)
            .map(|_| vec![Err(Error::Transport("overloaded_error: overloaded".into()))])
            .collect();
        turns.push(
            turn::text("would have recovered")
                .into_iter()
                .map(Ok)
                .collect(),
        );
        let transport = std::sync::Arc::new(MockTransport::scripted(turns));
        let agent = Agent::new(transport.clone(), "claude-test");
        let mut session = Session::new();
        session.user("hi");

        // Mirrors `--no-retry` alongside a nonzero `--retry-max-retries`: disabled must win outright.
        let policy = beyond_ai_agent::retry::RunRetryPolicy::from_overrides_with_enabled(
            false,
            Some(99),
            None,
        );
        let err = run_turn(
            &agent,
            &mut session,
            false,
            &agent_core::CancellationToken::new(),
            &policy,
            &AtomicBool::new(false),
            &agent_core::Steering::new(),
            false,
            u32::MAX,
        )
        .await
        .expect_err(
            "a disabled whole-run retry policy must never retry, even a recoverable failure",
        );
        assert!(matches!(err, Error::Transport(_)));
        // Only agent_core's own internal (mid-stream) retries ran; our whole-run wrapper made no
        // second attempt at all, so the final scripted (would-have-succeeded) turn was never reached.
        assert_eq!(transport.calls(), INNER_RETRY_ATTEMPTS);
    }

    #[test]
    fn partition_tasks_separates_at_file_refs_from_plain_messages() {
        let (files, messages) = partition_tasks(vec![
            "@notes.txt".to_string(),
            "first message".to_string(),
            "@img.png".to_string(),
            "second message".to_string(),
        ]);
        assert_eq!(files, vec!["notes.txt", "img.png"]);
        assert_eq!(messages, vec!["first message", "second message"]);
    }

    #[test]
    fn partition_tasks_with_no_at_refs_returns_all_as_messages() {
        let (files, messages) = partition_tasks(vec!["just a message".to_string()]);
        assert!(files.is_empty());
        assert_eq!(messages, vec!["just a message"]);
    }

    #[tokio::test]
    async fn read_file_refs_wraps_contents_in_a_file_tag_with_the_resolved_path() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "hello world").unwrap();
        let out = read_file_refs(&["a.txt".to_string()], dir.path(), false, true, true)
            .await
            .unwrap();
        assert!(out.text.contains("hello world"));
        assert!(
            out.text
                .contains(&format!("name=\"{}\"", dir.path().join("a.txt").display()))
        );
        assert!(out.images.is_empty());
    }

    #[tokio::test]
    async fn read_file_refs_errors_naming_the_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let err = read_file_refs(
            &["does-not-exist.txt".to_string()],
            dir.path(),
            false,
            true,
            true,
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(err.contains("does-not-exist.txt"), "got: {err}");
    }

    #[tokio::test]
    async fn read_file_refs_concatenates_multiple_files_in_order() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "AAA").unwrap();
        std::fs::write(dir.path().join("b.txt"), "BBB").unwrap();
        let out = read_file_refs(
            &["a.txt".to_string(), "b.txt".to_string()],
            dir.path(),
            false,
            true,
            true,
        )
        .await
        .unwrap();
        assert!(out.text.find("AAA").unwrap() < out.text.find("BBB").unwrap());
    }

    #[tokio::test]
    async fn read_file_refs_attaches_an_at_referenced_image_instead_of_erroring() {
        // Track L20 (pi-parity fix): `run @screenshot.png "..."` used to crash — `read_file_refs`
        // called plain `std::fs::read_to_string` on every `@file` ref, which errors outright on binary
        // image bytes. An image ref must now come back as an `ImageSource` attachment instead.
        let img = image::RgbImage::from_pixel(4, 4, image::Rgb([200, 10, 10]));
        let mut png_bytes = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut png_bytes),
                image::ImageFormat::Png,
            )
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("shot.png"), &png_bytes).unwrap();

        let out = read_file_refs(&["shot.png".to_string()], dir.path(), false, true, true)
            .await
            .unwrap();
        assert_eq!(
            out.images.len(),
            1,
            "the image must be attached, not read as text"
        );
        assert_eq!(out.images[0].media_type, "image/png");
        assert!(!out.images[0].data.is_empty());
    }

    #[tokio::test]
    async fn read_file_refs_block_images_true_drops_a_cli_attached_image() {
        // Task #35 (pi-parity fix): `--block-images`/`agent settings --block-images` previously had no
        // effect on a CLI `@file.png` attachment at all — only on a model-issued `read` tool call
        // (`Agent::block_images`'s tool-dispatch gate). `read_file_refs`'s own `block_images` parameter
        // must thread `_model_supports_vision: false` down to the `read` tool the same way, so the
        // image is dropped and the same non-vision placeholder note appended instead of splicing
        // straight into the first `Message`.
        let img = image::RgbImage::from_pixel(4, 4, image::Rgb([200, 10, 10]));
        let mut png_bytes = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut png_bytes),
                image::ImageFormat::Png,
            )
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("shot.png"), &png_bytes).unwrap();

        let out = read_file_refs(&["shot.png".to_string()], dir.path(), true, true, true)
            .await
            .unwrap();
        assert!(
            out.images.is_empty(),
            "--block-images must drop the image entirely: {out:?}"
        );
        assert!(
            out.text.contains("does not support images"),
            "--block-images must leave the same non-vision placeholder note `read` uses: {}",
            out.text
        );
    }

    #[tokio::test]
    async fn read_file_refs_appends_the_non_vision_note_for_a_real_non_vision_model_even_with_block_images_false()
     {
        // Task 3 (pi-parity fix, pass 19): `read_file_refs_with_home` used to pass
        // `"_model_supports_vision": !block_images` — omitting `&& caps.supports_vision` — unlike the
        // model-issued `read` tool-call dispatch path in `agent_core::agent`, which correctly ANDs in the
        // real model capability. So a genuinely non-vision model, with `--block-images` left at its
        // default (`false`), got a CLI `@screenshot.png` attachment dispatched with
        // `_model_supports_vision: true` — `read.rs` never appended its non-vision-image explanatory note
        // the way an equivalent model-issued call would have. This test passes `model_supports_vision:
        // false` (simulating a real non-vision model) with `block_images: false` (the default, NOT
        // forcing the downgrade) — the exact combination the bug missed — and asserts the note still
        // appears, matching `read_file_refs_block_images_true_drops_a_cli_attached_image`'s identical
        // assertion for the other (operator-forced) path to the same note.
        let img = image::RgbImage::from_pixel(4, 4, image::Rgb([200, 10, 10]));
        let mut png_bytes = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut png_bytes),
                image::ImageFormat::Png,
            )
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("shot.png"), &png_bytes).unwrap();

        let out = read_file_refs(&["shot.png".to_string()], dir.path(), false, true, false)
            .await
            .unwrap();
        assert!(
            out.images.is_empty(),
            "a non-vision model must not get the image spliced in: {out:?}"
        );
        assert!(
            out.text.contains("does not support images"),
            "a non-vision model must get the same non-vision placeholder note `read` uses even when \
             --block-images is false: {}",
            out.text
        );
    }

    #[tokio::test]
    async fn read_file_refs_image_auto_resize_false_skips_downscaling_an_oversized_cli_image() {
        // Task #36 (pi-parity fix): `--no-image-auto-resize`/`agent settings --image-auto-resize false`
        // previously had no effect on a CLI `@file.png` attachment — this call site always built a bare
        // `tools::read::Read::default()`, which defaults `image_auto_resize` to `true` regardless of
        // the flag. An oversized image (bigger than `read`'s `MAX_IMAGE_DIMENSION`) must ship at its
        // original pixel dimensions when the flag disables resizing, matching `tools::read`'s own
        // `image_auto_resize_off_ships_an_oversized_image_without_downscaling_it` unit test.
        let img = image::RgbImage::from_pixel(2200, 2200, image::Rgb([10, 200, 10]));
        let mut png_bytes = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut png_bytes),
                image::ImageFormat::Png,
            )
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("big.png"), &png_bytes).unwrap();

        let out = read_file_refs(&["big.png".to_string()], dir.path(), false, false, true)
            .await
            .unwrap();
        assert_eq!(out.images.len(), 1, "got: {out:?}");
        let decoded = image::load_from_memory(
            &base64::Engine::decode(
                &base64::engine::general_purpose::STANDARD,
                &*out.images[0].data,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            decoded.width(),
            2200,
            "--no-image-auto-resize must ship the image at its original width, not downscaled"
        );
    }

    #[test]
    fn looks_like_image_is_false_for_an_ordinary_text_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("notes.txt");
        std::fs::write(&path, "just some text").unwrap();
        assert!(!looks_like_image(&path));
    }

    #[tokio::test]
    async fn read_file_refs_expands_a_leading_tilde_before_joining_with_cwd() {
        // Task #20 (pi-parity fix): `@~/notes.md` previously failed outright — the `@` prefix defeats
        // the shell's own tilde-expansion (it's inside a single argument), and `cwd.join("~/notes.md")`
        // literally means `<cwd>/~/notes.md`, not the home directory.
        let home_dir = tempfile::tempdir().unwrap();
        std::fs::write(home_dir.path().join("notes.md"), "home directory contents").unwrap();
        let cwd_dir = tempfile::tempdir().unwrap();

        let out = read_file_refs_with_home(
            &["~/notes.md".to_string()],
            cwd_dir.path(),
            Some(home_dir.path().to_str().unwrap()),
            false,
            true,
            true,
        )
        .await
        .unwrap();
        assert!(
            out.text.contains("home directory contents"),
            "must read the file under the expanded home directory: {}",
            out.text
        );
    }

    #[tokio::test]
    async fn read_file_refs_a_bare_tilde_with_no_slash_resolves_to_home_itself() {
        let home_dir = tempfile::tempdir().unwrap();
        std::fs::write(home_dir.path().join("direct.txt"), "direct child of home").unwrap();
        let cwd_dir = tempfile::tempdir().unwrap();

        // `~` alone (no trailing slash) followed by a plain relative join still lands inside the home
        // directory via `cwd.join(expanded)`'s own absolute-path-replaces-base semantics.
        let out = read_file_refs_with_home(
            &["~/direct.txt".to_string()],
            cwd_dir.path(),
            Some(home_dir.path().to_str().unwrap()),
            false,
            true,
            true,
        )
        .await
        .unwrap();
        assert!(out.text.contains("direct child of home"));
    }

    #[tokio::test]
    async fn read_file_refs_a_non_tilde_ref_is_unaffected_by_the_home_directory() {
        let home_dir = tempfile::tempdir().unwrap();
        let cwd_dir = tempfile::tempdir().unwrap();
        std::fs::write(cwd_dir.path().join("plain.txt"), "plain cwd file").unwrap();

        let out = read_file_refs_with_home(
            &["plain.txt".to_string()],
            cwd_dir.path(),
            Some(home_dir.path().to_str().unwrap()),
            false,
            true,
            true,
        )
        .await
        .unwrap();
        assert!(out.text.contains("plain cwd file"));
    }

    #[tokio::test]
    async fn read_file_refs_skips_a_zero_byte_file_entirely() {
        // Task #38 (pi-parity fix): matches pi's own `file-processor.ts` (`stats.size === 0` →
        // `continue`) — an empty `@file` must contribute nothing, not an empty `<file
        // name="...">\n\n</file>` block.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("empty.txt"), b"").unwrap();
        let out = read_file_refs(&["empty.txt".to_string()], dir.path(), false, true, true)
            .await
            .unwrap();
        assert_eq!(
            out.text, "",
            "a zero-byte file must contribute nothing at all"
        );
        assert!(out.images.is_empty());
    }

    #[tokio::test]
    async fn read_file_refs_skips_only_the_empty_file_among_several() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "AAA").unwrap();
        std::fs::write(dir.path().join("empty.txt"), b"").unwrap();
        std::fs::write(dir.path().join("b.txt"), "BBB").unwrap();
        let out = read_file_refs(
            &[
                "a.txt".to_string(),
                "empty.txt".to_string(),
                "b.txt".to_string(),
            ],
            dir.path(),
            false,
            true,
            true,
        )
        .await
        .unwrap();
        assert!(out.text.contains("AAA"));
        assert!(out.text.contains("BBB"));
        assert!(!out.text.contains("empty.txt"));
    }

    #[tokio::test]
    async fn read_file_refs_still_errors_on_a_missing_file_rather_than_skipping_it() {
        // A zero-byte skip must not swallow the genuinely-missing-file error — `metadata()` failing
        // falls through to the normal read attempt, which reports the real problem.
        let dir = tempfile::tempdir().unwrap();
        let err = read_file_refs(
            &["does-not-exist.txt".to_string()],
            dir.path(),
            false,
            true,
            true,
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(err.contains("does-not-exist.txt"), "got: {err}");
    }

    #[test]
    fn trim_piped_stdin_trims_surrounding_whitespace() {
        assert_eq!(
            trim_piped_stdin("  hello world  \n"),
            Some("hello world".to_string())
        );
    }

    #[test]
    fn trim_piped_stdin_a_whitespace_only_buffer_is_none() {
        assert_eq!(trim_piped_stdin("   \n\t  "), None);
    }

    #[test]
    fn trim_piped_stdin_an_empty_buffer_is_none() {
        assert_eq!(trim_piped_stdin(""), None);
    }

    #[test]
    fn trim_piped_stdin_leaves_interior_whitespace_untouched() {
        assert_eq!(
            trim_piped_stdin("  line one\nline two  "),
            Some("line one\nline two".to_string())
        );
    }

    #[test]
    fn format_token_count_below_a_thousand_is_a_plain_integer() {
        assert_eq!(format_token_count(512), "512");
        assert_eq!(format_token_count(0), "0");
    }

    #[test]
    fn format_token_count_thousands_round_to_a_whole_k_when_exact() {
        assert_eq!(format_token_count(200_000), "200K");
        assert_eq!(format_token_count(1_000), "1K");
    }

    #[test]
    fn format_token_count_thousands_keep_one_decimal_when_not_exact() {
        assert_eq!(format_token_count(1_500), "1.5K");
    }

    #[test]
    fn format_token_count_millions_round_to_a_whole_m_when_exact() {
        assert_eq!(format_token_count(1_000_000), "1M");
        assert_eq!(format_token_count(2_000_000), "2M");
    }

    #[test]
    fn format_token_count_millions_keep_one_decimal_when_not_exact() {
        assert_eq!(format_token_count(1_500_000), "1.5M");
    }

    #[test]
    fn resolve_thinking_budget_overrides_returns_none_when_nothing_configured() {
        let settings = beyond_ai_agent::settings::Settings::default();
        assert_eq!(resolve_thinking_budget_overrides(&settings), None);
    }

    #[test]
    fn resolve_thinking_budget_overrides_parses_recognized_wire_strings() {
        let mut table = std::collections::BTreeMap::new();
        table.insert("high".to_string(), 40_000u32);
        table.insert("low".to_string(), 1_000u32);
        let settings = beyond_ai_agent::settings::Settings {
            thinking_budget_overrides: Some(table),
            ..Default::default()
        };
        let overrides = resolve_thinking_budget_overrides(&settings).unwrap();
        assert_eq!(
            overrides.get(&agent_core::ReasoningEffort::High),
            Some(&40_000)
        );
        assert_eq!(
            overrides.get(&agent_core::ReasoningEffort::Low),
            Some(&1_000)
        );
    }

    #[test]
    fn resolve_thinking_budget_overrides_skips_an_unrecognized_key() {
        let mut table = std::collections::BTreeMap::new();
        table.insert("not-a-real-effort".to_string(), 1u32);
        let settings = beyond_ai_agent::settings::Settings {
            thinking_budget_overrides: Some(table),
            ..Default::default()
        };
        assert_eq!(
            resolve_thinking_budget_overrides(&settings),
            None,
            "an override table with only unrecognized keys must resolve to no overrides at all"
        );
    }
}
