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
use beyond_agent::gateway_credential::{GatewayCredential, resolve_gateway_credential_and_headers};
use beyond_agent::mode::Mode;
use beyond_agent::session_store::{SessionRepo, SessionStore};
use beyond_agent::{serve, tools};

mod commands;

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
/// `--cache-ttl`: how long the provider keeps a prompt-cache entry alive between requests. The wire
/// still carries one bool (`ModelRequest::cache_long`); this only decides it, per mode — `serve`
/// defaults to [`CacheTtl::OneHour`], `run` to [`CacheTtl::FiveMinutes`]. See "Prompt-cache TTL by
/// mode" in `crates/agent/ARCHITECTURE.md` for the break-even behind the defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CacheTtl {
    FiveMinutes,
    OneHour,
}

impl std::str::FromStr for CacheTtl {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "5m" => Ok(Self::FiveMinutes),
            "1h" => Ok(Self::OneHour),
            other => Err(format!("invalid cache TTL {other:?}; expected 5m or 1h")),
        }
    }
}

impl CacheTtl {
    /// Whether to ask for the 1-hour TTL: `--cache-ttl` if given, else `--cache-long`, else the
    /// mode's default.
    fn resolve(flag: Option<Self>, cache_long: bool, mode_default: Self) -> bool {
        let ttl = flag.unwrap_or(if cache_long {
            Self::OneHour
        } else {
            mode_default
        });
        ttl == Self::OneHour
    }
}

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
    settings: &beyond_agent::settings::Settings,
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
    provider_env: &beyond_agent::gateway_credential::ProviderEnv,
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
/// [`beyond_agent::oauth::LoginCallbacks`]. `agent login` is a dedicated, single-purpose blocking
/// invocation with no concurrent-command concerns (unlike `serve`), so blocking stdin reads (moved to
/// a `spawn_blocking` task, out of hygiene rather than necessity here) are the whole interaction —
/// there's no need for `serve`'s RPC surface's separate ack-now/respond-later, push-frame shape.
struct CliLoginCallbacks;

#[async_trait::async_trait]
impl beyond_agent::oauth::LoginCallbacks for CliLoginCallbacks {
    async fn show_auth_url(&self, url: &str, instructions: Option<&str>) {
        eprintln!("Open this URL in a browser to continue:\n\n  {url}\n");
        if let Some(instructions) = instructions {
            eprintln!("{instructions}");
        }
    }

    async fn show_device_code(&self, info: &beyond_agent::oauth::DeviceCodeInfo) {
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
        prompt: &beyond_agent::oauth::TextPrompt<'_>,
    ) -> Result<String, beyond_agent::oauth::OAuthError> {
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
                .map_err(|e| beyond_agent::oauth::OAuthError::InvalidInput(e.to_string()))?;
            Ok(line.trim().to_string())
        })
        .await
        .map_err(|e| beyond_agent::oauth::OAuthError::InvalidInput(e.to_string()))?
    }

    async fn select(
        &self,
        prompt: &beyond_agent::oauth::SelectPrompt<'_>,
    ) -> Result<Option<String>, beyond_agent::oauth::OAuthError> {
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
    bin = "agent",
    version,
    about = "Beyond agent harness",
    unknown_flags = "error"
)]
struct Cli {
    #[usage(subcommand)]
    command: Command,
}

// One `Command` is parsed per process and matched once; the flag-heavy `Serve` variant outgrowing the
// small ones costs a few hundred bytes of stack at startup, not worth boxing every field for.
#[allow(clippy::large_enum_variant)]
#[derive(Subcommands, Clone)]
enum Command {
    /// Run one agent task to completion, streaming output to stdout.
    Run {
        /// The task. Several run as separate turns, in order. `@path` attaches a file; `-` (or no
        /// task) reads stdin.
        tasks: Vec<String>,
        /// Model id. Defaults to the stored `settings --model`, else {DEFAULT_MODEL}.
        #[usage(long, env = "AI_AGENT_MODEL")]
        model: Option<String>,
        /// Gateway base URL. Defaults to the stored setting, else `http://ai.internal`.
        #[usage(long, env = "AI_GATEWAY_URL")]
        gateway_url: Option<String>,
        /// API key: a gateway virtual key (`bai_v1…`) or a provider key. Not needed after `login`.
        #[usage(long, env = "AI_AGENT_KEY")]
        key: Option<String>,
        /// Stop after this many model turns (default: no limit). The model gets 5 extra turns to
        /// finish cleanly.
        #[usage(long)]
        max_steps: Option<u32>,
        /// Output token limit per turn. Defaults to the model's maximum.
        #[usage(long, env = "AI_AGENT_MAX_TOKENS")]
        max_tokens: Option<u32>,
        /// Prompt-cache lifetime: `5m` (default) or `1h`. Use `1h` if a tool can run longer than
        /// five minutes.
        #[usage(long, env = "AI_AGENT_CACHE_TTL")]
        cache_ttl: Option<CacheTtl>,
        /// Same as `--cache-ttl 1h`.
        #[usage(long)]
        cache_long: bool,
        /// Extended-thinking budget in tokens (must be below --max-tokens). For a level instead,
        /// see `--reasoning-effort`.
        #[usage(long)]
        thinking: Option<u32>,
        /// Reasoning effort: off, minimal, low, medium, high or xhigh. For a token budget instead,
        /// see `--thinking`.
        #[usage(long, env = "AI_AGENT_REASONING_EFFORT")]
        reasoning_effort: Option<agent_core::ThinkingLevel>,
        /// How many queued steer messages are delivered at each turn boundary: `one_at_a_time`
        /// (default) or `all`.
        #[usage(long, env = "AI_AGENT_STEERING_MODE")]
        steering_mode: Option<agent_core::QueueMode>,
        /// How many queued follow-up messages are delivered when the agent stops: `one_at_a_time`
        /// (default) or `all`.
        #[usage(long, env = "AI_AGENT_FOLLOW_UP_MODE")]
        follow_up_mode: Option<agent_core::QueueMode>,
        /// Sampling temperature. Unset uses the provider's default; ignored by Anthropic when
        /// thinking is on.
        #[usage(long)]
        temperature: Option<f64>,
        /// Replace the built-in system prompt. Takes text or a path to a file.
        #[usage(long, env = "AI_AGENT_SYSTEM_PROMPT")]
        system_prompt: Option<String>,
        /// Add instructions after the system prompt. Takes text or a file path. Repeatable.
        #[usage(long, env = "AI_AGENT_APPEND_SYSTEM_PROMPT")]
        append_system_prompt: Vec<String>,
        /// Trust the current directory for this run only.
        #[usage(short = 'a', long)]
        trust_project: bool,
        /// Treat the current directory as untrusted, even if it is trusted. Wins over
        /// --trust-project.
        #[usage(long)]
        force_untrusted: bool,
        /// Context window in tokens. Defaults to the model's; older turns are summarized to stay
        /// under it.
        #[usage(long, env = "AI_AGENT_CONTEXT_WINDOW")]
        context_window: Option<u32>,
        /// Compact once the prompt passes this many tokens. Defaults to the model's cap; `0` means
        /// the window.
        #[usage(long, env = "AI_AGENT_COMPACT_AT")]
        compact_at: Option<u32>,
        /// Summarize at once when compacting, without first replacing old tool output with
        /// placeholders.
        #[usage(long, env = "AI_AGENT_NO_TOOL_MASKING")]
        no_tool_masking: bool,
        /// Recent turns whose tool output is never masked. Default 10.
        #[usage(long, env = "AI_AGENT_TOOL_MASKING_KEEP_TURNS")]
        tool_masking_keep_turns: Option<usize>,
        /// Recent tokens whose tool output is never masked. Default: a quarter of the compaction
        /// trigger.
        #[usage(long, env = "AI_AGENT_TOOL_MASKING_KEEP_TOKENS")]
        tool_masking_keep_tokens: Option<u32>,
        /// Only mask if it frees at least this many tokens. Default 25,000.
        #[usage(long, env = "AI_AGENT_TOOL_MASKING_MIN_TOKENS")]
        tool_masking_min_tokens: Option<u32>,
        /// Build the compaction summary from a re-rendered transcript instead of the cached
        /// conversation.
        #[usage(long, env = "AI_AGENT_NO_CACHED_SUMMARY")]
        no_cached_summary: bool,
        /// Experimental compaction changes to turn on, comma-separated: `state`, `tail`.
        #[usage(long, env = "AI_AGENT_COMPACTION_CANDIDATES", delimiter = ',')]
        compaction_candidates: Option<Vec<String>>,
        /// Tokens kept free below the context window before compaction runs. Default 16,384.
        #[usage(long, env = "AI_AGENT_COMPACTION_RESERVE_TOKENS")]
        compaction_reserve_tokens: Option<u32>,
        /// Tokens of recent conversation compaction keeps verbatim. Defaults by model family.
        #[usage(long, env = "AI_AGENT_COMPACTION_KEEP_RECENT_TOKENS")]
        compaction_keep_recent_tokens: Option<u32>,
        /// Tokens kept free when summarizing an abandoned branch. Defaults to
        /// --compaction-reserve-tokens.
        #[usage(long, env = "AI_AGENT_BRANCH_SUMMARY_RESERVE_TOKENS")]
        branch_summary_reserve_tokens: Option<u32>,
        /// Never compact proactively; only when the context actually overflows.
        #[usage(long, env = "AI_AGENT_NO_COMPACTION")]
        no_compaction: bool,
        /// Retries for a request that fails before any response arrives. Default 3.
        #[usage(long, env = "AI_AGENT_RETRY_MAX_RETRIES")]
        retry_max_retries: Option<u32>,
        /// First retry delay in milliseconds, doubling each time. Default 250.
        #[usage(long, env = "AI_AGENT_RETRY_BASE_DELAY_MS")]
        retry_base_delay_ms: Option<u64>,
        /// Longest single retry delay in milliseconds. Default 60,000.
        #[usage(long, env = "AI_AGENT_RETRY_MAX_BACKOFF_MS")]
        retry_max_backoff_ms: Option<u64>,
        /// How long a turn keeps retrying a rate limit (429/529), in seconds. Default 300; `0`
        /// fails at once.
        #[usage(long, env = "AI_AGENT_RETRY_RATE_LIMIT_SECS")]
        retry_rate_limit_secs: Option<u64>,
        /// Don't re-run the whole task after it fails with a transient error.
        #[usage(long, env = "AI_AGENT_NO_RETRY")]
        no_retry: bool,
        /// Fail a response that sends nothing for this many milliseconds.
        #[usage(long, env = "AI_AGENT_IDLE_TIMEOUT_MS")]
        idle_timeout_ms: Option<u64>,
        /// Never send images to the model; replace them with a text note.
        #[usage(long, env = "AI_AGENT_BLOCK_IMAGES")]
        block_images: bool,
        /// Send images even if the stored setting blocks them.
        #[usage(long, env = "AI_AGENT_NO_BLOCK_IMAGES")]
        no_block_images: bool,
        /// Send images at full size instead of downscaling large ones.
        #[usage(long, env = "AI_AGENT_NO_IMAGE_AUTO_RESIZE")]
        no_image_auto_resize: bool,
        /// Shell for the `bash` tool. Defaults to `/bin/bash`, else `bash` on PATH, else `sh`.
        #[usage(long, env = "AI_AGENT_BASH_SHELL_PATH")]
        bash_shell_path: Option<String>,
        /// A line run before every `bash` command, e.g. `source .venv/bin/activate`.
        #[usage(long, env = "AI_AGENT_BASH_COMMAND_PREFIX")]
        bash_command_prefix: Option<String>,
        /// Let the `web` tool reach loopback and private addresses (blocked by default).
        #[usage(long, env = "AI_AGENT_WEB_ALLOW_PRIVATE")]
        web_allow_private: bool,
        /// A host the `web` tool may reach even if it is private. Repeatable.
        #[usage(long, env = "AI_AGENT_WEB_ALLOW_HOST")]
        web_allow_host: Vec<String>,
        /// Per-request timeout for the `web` tool, in milliseconds. Default 30,000.
        #[usage(long, env = "AI_AGENT_WEB_TIMEOUT_MS")]
        web_timeout_ms: Option<u64>,
        /// Run the file tools on a remote exec endpoint (`POST {command, args, cwd}`) instead of
        /// this machine.
        #[usage(long, env = "AI_AGENT_EXEC_URL", conflicts = "exec_cmd")]
        exec_url: Option<String>,
        /// A header for every exec request, `Name: value`. Repeatable.
        #[usage(long, env = "AI_AGENT_EXEC_HEADER")]
        exec_header: Vec<String>,
        /// Largest exec response accepted, in bytes. Default 16 MiB.
        #[usage(long, env = "AI_AGENT_EXEC_MAX_RESPONSE_BYTES")]
        exec_max_response_bytes: Option<u64>,
        /// Run the file tools through a command template instead, e.g. `ssh host -- {}`.
        #[usage(long, env = "AI_AGENT_EXEC_CMD", conflicts = "exec_url")]
        exec_cmd: Option<String>,
        /// POST the run's lifecycle events (started, progress, finished) to this URL (`http(s)://`
        /// or `unix://`).
        #[usage(long, env = "AI_AGENT_LIFECYCLE_URL")]
        lifecycle_url: Option<String>,
        /// A header for every lifecycle POST, `Name: value`. Repeatable.
        #[usage(long, env = "AI_AGENT_LIFECYCLE_HEADER")]
        lifecycle_header: Vec<String>,
        /// Seconds between progress POSTs while running. Default 10; `0` turns the timer off.
        #[usage(long, env = "AI_AGENT_LIFECYCLE_HEARTBEAT_SECS", default = "10")]
        lifecycle_heartbeat_secs: u64,
        /// Only offer these tools (comma-separated).
        #[usage(short = 't', long, env = "AI_AGENT_TOOLS", delimiter = ',')]
        tools: Option<Vec<String>>,
        /// Don't offer these tools (comma-separated).
        #[usage(long, env = "AI_AGENT_EXCLUDE_TOOLS", delimiter = ',')]
        exclude_tools: Option<Vec<String>>,
        /// Offer no tools at all.
        #[usage(long)]
        no_tools: bool,
        /// Run a turn's tool calls one at a time instead of concurrently.
        #[usage(long)]
        sequential_tools: bool,
        /// Expose MCP tools through one JavaScript `execute` tool. Needs a build with `--features
        /// code-mode`.
        #[usage(long, env = "AI_AGENT_CODE_MODE")]
        code_mode: bool,
        /// How MCP tools reach the model: `auto` (default), `search` (loaded on demand) or `eager`
        /// (all, always).
        #[usage(long, env = "AI_AGENT_MCP_TOOL_LOADING")]
        mcp_tool_loading: Option<beyond_agent::tools::tool_search::McpLoading>,
        /// Refuse every call to this tool; the model sees an error. Repeatable.
        #[usage(long, env = "AI_AGENT_DENY_TOOL", delimiter = ',')]
        deny_tool: Vec<String>,
        /// Refuse a `bash` command containing this text (case-insensitive). Repeatable.
        #[usage(long, env = "AI_AGENT_DENY_BASH_PATTERN", delimiter = ',')]
        deny_bash_pattern: Vec<String>,
        /// Refuse a `write`/`edit` to a path matching this glob. Repeatable.
        #[usage(long, env = "AI_AGENT_DENY_PATH", delimiter = ',')]
        deny_path: Vec<String>,
        /// Don't load skills from `~/.claude/skills` or `.claude/skills` (a --skill path still
        /// loads).
        #[usage(long)]
        no_skills: bool,
        /// Allow skills served by MCP servers, and running code while one is active, without
        /// asking.
        #[usage(long, env = "AI_AGENT_APPROVE_MCP_SKILLS")]
        approve_mcp_skills: bool,
        /// Don't load prompt templates from `~/.claude/prompts` or `.claude/prompts`.
        #[usage(long)]
        no_prompt_templates: bool,
        /// Don't read AGENTS.md or CLAUDE.md files.
        #[usage(long)]
        no_context_files: bool,
        /// Also load skills from this directory or `.md` file. Repeatable.
        #[usage(
            long = "skill",
            env = "AI_AGENT_SKILL_PATH",
            delimiter = ',',
            value_name = "PATH"
        )]
        extra_skill_paths: Vec<String>,
        /// Also load prompt templates from this directory or `.md` file. Repeatable.
        #[usage(
            long = "prompt-template",
            env = "AI_AGENT_PROMPT_TEMPLATE_PATH",
            delimiter = ',',
            value_name = "PATH"
        )]
        extra_prompt_template_paths: Vec<String>,
        /// Name a new session.
        #[usage(short = 'n', long)]
        name: Option<String>,
        /// Add a line to the system prompt's guidelines. Repeatable.
        #[usage(long = "prompt-guideline", value_name = "TEXT")]
        prompt_guidelines: Vec<String>,
        /// Start a new session copied from this one (a session id or `.jsonl` path).
        #[usage(long, value_name = "PATH_OR_ID")]
        fork: Option<String>,
        /// Continue this session (an id or `.jsonl` path), creating it if missing.
        #[usage(long)]
        session: Option<String>,
        /// Continue the session with this id, or create it under this id.
        #[usage(long)]
        session_id: Option<String>,
        /// Continue the most recent session in this directory.
        #[usage(long, short = 'c')]
        r#continue: bool,
        /// Directory to keep sessions in. Default `~/.beyond/sessions/<current directory>`.
        #[usage(long, env = "AI_AGENT_SESSION_DIR")]
        session_dir: Option<String>,
        /// Don't save the session.
        #[usage(long)]
        no_session_persistence: bool,
        /// Where persistent memory lives: a directory path. Defaults to one per project under
        /// `~/.beyond/projects`.
        #[usage(long, env = "AI_AGENT_MEMORY_URL")]
        memory: Option<String>,
        /// Turn persistent memory off.
        #[usage(long)]
        no_memory: bool,
        /// Turn off the per-session `/session` scratch memory; keep `/memories`.
        #[usage(long)]
        no_session_memory: bool,
        /// When the run ends, write the transcript as HTML to this path.
        #[usage(long)]
        export: Option<String>,
        /// Print JSON lines (one event per line) instead of text.
        #[usage(long)]
        json: bool,
        /// Make the agent return JSON matching this schema (inline or a file path); printed as the
        /// last line.
        #[usage(long, env = "AI_AGENT_OUTPUT_SCHEMA")]
        output_schema: Option<String>,
        /// What the --output-schema result is for, as told to the model.
        #[usage(long)]
        output_description: Option<String>,
    },
    /// Run the agent as a server: a JSON-lines control protocol over stdio, a WebSocket or a Unix
    /// socket.
    Serve {
        /// Model id. Defaults to the stored `settings --model`, else {DEFAULT_MODEL}.
        #[usage(long, env = "AI_AGENT_MODEL")]
        model: Option<String>,
        /// Gateway base URL. Defaults to the stored setting, else `http://ai.internal`.
        #[usage(long, env = "AI_GATEWAY_URL")]
        gateway_url: Option<String>,
        /// API key: a gateway virtual key (`bai_v1…`) or a provider key. Not needed after `login`.
        #[usage(long, env = "AI_AGENT_KEY")]
        key: Option<String>,
        /// Keep one session in this `.jsonl` file.
        #[usage(long, env = "AI_AGENT_SESSION_FILE")]
        session_file: Option<String>,
        /// Directory to keep sessions in (enables listing, switching and forking sessions).
        #[usage(long, env = "AI_AGENT_SESSION_DIR")]
        session_dir: Option<String>,
        /// Serve over WebSocket and HTTP on this address (e.g. `127.0.0.1:8787`) instead of stdio.
        /// Bind it privately.
        #[usage(long, env = "AI_AGENT_LISTEN")]
        listen: Option<std::net::SocketAddr>,
        /// Also serve the same protocol over TLS on this address (needs --tls-cert and
        /// --tls-key). The certificate is read once; rotating it takes a restart.
        #[usage(long, env = "AI_AGENT_TLS_LISTEN")]
        tls_listen: Option<std::net::SocketAddr>,
        /// PEM certificate chain (leaf first) for --tls-listen.
        #[usage(long, env = "AI_AGENT_TLS_CERT")]
        tls_cert: Option<std::path::PathBuf>,
        /// PEM private key for --tls-listen.
        #[usage(long, env = "AI_AGENT_TLS_KEY")]
        tls_key: Option<std::path::PathBuf>,
        /// Serve over a Unix socket at this path.
        #[usage(long, env = "AI_AGENT_LISTEN_UDS")]
        listen_uds: Option<std::path::PathBuf>,
        /// Permissions for the --listen-uds socket, in octal. Default `600`.
        #[usage(long, env = "AI_AGENT_LISTEN_UDS_MODE")]
        listen_uds_mode: Option<String>,
        /// Stop a session with no client attached after this many seconds. Default 3600 (60 with
        /// --service); `0` never.
        #[usage(long, env = "AI_AGENT_SESSION_IDLE_TIMEOUT")]
        session_idle_timeout: Option<u64>,
        /// How sessions share connections to the gateway: `auto` (default), `off`, or `h2c`.
        #[usage(long, env = "AI_AGENT_UPSTREAM_HTTP2", default = "auto")]
        upstream_http2: serve::UpstreamHttp2,
        /// Trust session grants signed by this key, as `<kid>=<base64 Ed25519 public key>`.
        /// Repeatable.
        #[usage(long, env = "AI_AGENT_GRANT_KEY", delimiter = ',')]
        grant_key: Vec<String>,
        /// File with this fleet's X25519 secret key (base64), used to open grant secrets. The key
        /// itself may instead be given as base64 in `AI_AGENT_SEAL_KEY_B64`; not both.
        #[usage(long, env = "AI_AGENT_SEAL_KEY")]
        seal_key: Option<std::path::PathBuf>,
        /// During a seal-key rotation: file with the previous X25519 secret (base64), under which
        /// notify tokens are still accepted (never minted). Tokens carry no time, so drop it after 24 h
        /// (routine) or 5 min after the minter switched (urgent); see the README. May instead be
        /// base64 in `AI_AGENT_SEAL_KEY_PREVIOUS_B64`; not both.
        #[usage(long, env = "AI_AGENT_SEAL_KEY_PREVIOUS")]
        seal_key_previous: Option<std::path::PathBuf>,
        /// Run as a multi-tenant service: every connection must carry a signed session grant.
        #[usage(long, env = "AI_AGENT_SERVICE")]
        service: bool,
        /// A tenant-data mount, `<name>=</absolute/path>`. Repeatable; --service needs at least
        /// one.
        #[usage(long, env = "AI_AGENT_SHARD", delimiter = ',')]
        shard: Vec<String>,
        /// With --service, refuse new sessions past this many. `0` means no limit.
        #[usage(long, env = "AI_AGENT_MAX_LIVE_SESSIONS", default = "20000")]
        max_live_sessions: usize,
        /// Serve Prometheus metrics on this loopback address.
        #[usage(long, env = "AI_AGENT_METRICS_LISTEN")]
        metrics_listen: Option<String>,
        /// With --service, the origin the fleet's edge serves sessions at: sandboxes that offer
        /// job notify (exec v1.4) report a kept job's end to `<origin>/_beyond/agent`.
        #[usage(long, env = "AI_AGENT_NOTIFY_ORIGIN")]
        notify_origin: Option<String>,
        /// On shutdown, wait up to this many seconds for running prompts to finish. Default 0.
        #[usage(long, env = "AI_AGENT_DRAIN_GRACE", default = "0")]
        drain_grace: u64,
        /// With --service, let MCP servers be reached at private addresses (for development only).
        #[usage(long, env = "AI_AGENT_MCP_ALLOW_PRIVATE")]
        mcp_allow_private: bool,
        /// With --service, how many MiB of unpacked agent bundles to keep on disk; the least
        /// recently used that no session holds are removed beyond it. Default 1024.
        #[usage(long, env = "AI_AGENT_BUNDLE_CACHE_MB")]
        bundle_cache_mb: Option<u64>,
        /// Public base URL of this server, so MCP servers can deliver events by webhook.
        #[usage(long, env = "AI_AGENT_MCP_EVENTS_CALLBACK_URL")]
        mcp_events_callback_url: Option<String>,
        /// Let the idle timeout stop a session even while it has MCP event subscriptions.
        #[usage(long, env = "AI_AGENT_MCP_EVENTS_REAPABLE")]
        mcp_events_reapable: bool,
        /// The session that receives the MCP events configured in settings.
        #[usage(long, env = "AI_AGENT_MCP_EVENTS_SESSION")]
        mcp_events_session: Option<String>,
        /// Continue the session with this id, or create it under this id.
        #[usage(long)]
        session_id: Option<String>,
        /// Reattach to the most recent session in this directory.
        #[usage(long, short = 'c')]
        r#continue: bool,
        /// Don't save sessions.
        #[usage(long)]
        no_session_persistence: bool,
        /// Where persistent memory lives: a directory path. Defaults to one per project under
        /// `~/.beyond/projects`.
        #[usage(long, env = "AI_AGENT_MEMORY_URL")]
        memory: Option<String>,
        /// Turn persistent memory off.
        #[usage(long)]
        no_memory: bool,
        /// Turn off the per-session `/session` scratch memory; keep `/memories`.
        #[usage(long)]
        no_session_memory: bool,
        /// Stop a prompt after this many model turns (default: no limit).
        #[usage(long)]
        max_steps: Option<u32>,
        /// Replace the built-in system prompt. Takes text or a path to a file.
        #[usage(long, env = "AI_AGENT_SYSTEM_PROMPT")]
        system_prompt: Option<String>,
        /// Add instructions after the system prompt. Takes text or a file path. Repeatable.
        #[usage(long, env = "AI_AGENT_APPEND_SYSTEM_PROMPT")]
        append_system_prompt: Vec<String>,
        /// Don't read AGENTS.md or CLAUDE.md files.
        #[usage(long)]
        no_context_files: bool,
        /// Context window in tokens. Defaults to the model's; older turns are summarized to stay
        /// under it.
        #[usage(long, env = "AI_AGENT_CONTEXT_WINDOW")]
        context_window: Option<u32>,
        /// Compact once the prompt passes this many tokens. Defaults to the model's cap; `0` means
        /// the window.
        #[usage(long, env = "AI_AGENT_COMPACT_AT")]
        compact_at: Option<u32>,
        /// Summarize at once when compacting, without first replacing old tool output with
        /// placeholders.
        #[usage(long, env = "AI_AGENT_NO_TOOL_MASKING")]
        no_tool_masking: bool,
        /// Recent turns whose tool output is never masked. Default 10.
        #[usage(long, env = "AI_AGENT_TOOL_MASKING_KEEP_TURNS")]
        tool_masking_keep_turns: Option<usize>,
        /// Recent tokens whose tool output is never masked. Default: a quarter of the compaction
        /// trigger.
        #[usage(long, env = "AI_AGENT_TOOL_MASKING_KEEP_TOKENS")]
        tool_masking_keep_tokens: Option<u32>,
        /// Only mask if it frees at least this many tokens. Default 25,000.
        #[usage(long, env = "AI_AGENT_TOOL_MASKING_MIN_TOKENS")]
        tool_masking_min_tokens: Option<u32>,
        /// Build the compaction summary from a re-rendered transcript instead of the cached
        /// conversation.
        #[usage(long, env = "AI_AGENT_NO_CACHED_SUMMARY")]
        no_cached_summary: bool,
        /// Experimental compaction changes to turn on, comma-separated: `state`, `tail`.
        #[usage(long, env = "AI_AGENT_COMPACTION_CANDIDATES", delimiter = ',')]
        compaction_candidates: Option<Vec<String>>,
        /// Output token limit per turn. Defaults to the model's maximum.
        #[usage(long, env = "AI_AGENT_MAX_TOKENS")]
        max_tokens: Option<u32>,
        /// Prompt-cache lifetime: `1h` (default, for a person typing between prompts) or `5m` (for
        /// automation).
        #[usage(long, env = "AI_AGENT_CACHE_TTL")]
        cache_ttl: Option<CacheTtl>,
        /// Same as `--cache-ttl 1h`.
        #[usage(long)]
        cache_long: bool,
        /// Extended-thinking budget in tokens (must be below --max-tokens). For a level instead,
        /// see `--reasoning-effort`.
        #[usage(long)]
        thinking: Option<u32>,
        /// Reasoning effort: off, minimal, low, medium, high or xhigh. For a token budget instead,
        /// see `--thinking`.
        #[usage(long, env = "AI_AGENT_REASONING_EFFORT")]
        reasoning_effort: Option<agent_core::ThinkingLevel>,
        /// How many queued steer messages are delivered at each turn boundary: `one_at_a_time`
        /// (default) or `all`.
        #[usage(long, env = "AI_AGENT_STEERING_MODE")]
        steering_mode: Option<agent_core::QueueMode>,
        /// How many queued follow-up messages are delivered when the agent stops: `one_at_a_time`
        /// (default) or `all`.
        #[usage(long, env = "AI_AGENT_FOLLOW_UP_MODE")]
        follow_up_mode: Option<agent_core::QueueMode>,
        /// Sampling temperature. Unset uses the provider's default; ignored by Anthropic when
        /// thinking is on.
        #[usage(long)]
        temperature: Option<f64>,
        /// Trust the current directory for this server only.
        #[usage(short = 'a', long)]
        trust_project: bool,
        /// Treat the current directory as untrusted, even if it is trusted. Wins over
        /// --trust-project.
        #[usage(long)]
        force_untrusted: bool,
        /// Tokens kept free below the context window before compaction runs. Default 16,384.
        #[usage(long, env = "AI_AGENT_COMPACTION_RESERVE_TOKENS")]
        compaction_reserve_tokens: Option<u32>,
        /// Tokens of recent conversation compaction keeps verbatim. Defaults by model family.
        #[usage(long, env = "AI_AGENT_COMPACTION_KEEP_RECENT_TOKENS")]
        compaction_keep_recent_tokens: Option<u32>,
        /// Tokens kept free when summarizing an abandoned branch. Defaults to
        /// --compaction-reserve-tokens.
        #[usage(long, env = "AI_AGENT_BRANCH_SUMMARY_RESERVE_TOKENS")]
        branch_summary_reserve_tokens: Option<u32>,
        /// Never compact proactively; only when the context actually overflows.
        #[usage(long, env = "AI_AGENT_NO_COMPACTION")]
        no_compaction: bool,
        /// Retries for a request that fails before any response arrives. Default 3.
        #[usage(long, env = "AI_AGENT_RETRY_MAX_RETRIES")]
        retry_max_retries: Option<u32>,
        /// First retry delay in milliseconds, doubling each time. Default 250.
        #[usage(long, env = "AI_AGENT_RETRY_BASE_DELAY_MS")]
        retry_base_delay_ms: Option<u64>,
        /// Longest single retry delay in milliseconds. Default 60,000.
        #[usage(long, env = "AI_AGENT_RETRY_MAX_BACKOFF_MS")]
        retry_max_backoff_ms: Option<u64>,
        /// How long a turn keeps retrying a rate limit (429/529), in seconds. Default 300; `0`
        /// fails at once.
        #[usage(long, env = "AI_AGENT_RETRY_RATE_LIMIT_SECS")]
        retry_rate_limit_secs: Option<u64>,
        /// Fail a response that sends nothing for this many milliseconds.
        #[usage(long, env = "AI_AGENT_IDLE_TIMEOUT_MS")]
        idle_timeout_ms: Option<u64>,
        /// Never send images to the model; replace them with a text note.
        #[usage(long, env = "AI_AGENT_BLOCK_IMAGES")]
        block_images: bool,
        /// Send images even if the stored setting blocks them.
        #[usage(long, env = "AI_AGENT_NO_BLOCK_IMAGES")]
        no_block_images: bool,
        /// Send images at full size instead of downscaling large ones.
        #[usage(long, env = "AI_AGENT_NO_IMAGE_AUTO_RESIZE")]
        no_image_auto_resize: bool,
        /// Shell for the `bash` tool. Defaults to `/bin/bash`, else `bash` on PATH, else `sh`.
        #[usage(long, env = "AI_AGENT_BASH_SHELL_PATH")]
        bash_shell_path: Option<String>,
        /// A line run before every `bash` command, e.g. `source .venv/bin/activate`.
        #[usage(long, env = "AI_AGENT_BASH_COMMAND_PREFIX")]
        bash_command_prefix: Option<String>,
        /// Let the `web` tool reach loopback and private addresses (blocked by default).
        #[usage(long, env = "AI_AGENT_WEB_ALLOW_PRIVATE")]
        web_allow_private: bool,
        /// A host the `web` tool may reach even if it is private. Repeatable.
        #[usage(long, env = "AI_AGENT_WEB_ALLOW_HOST")]
        web_allow_host: Vec<String>,
        /// Per-request timeout for the `web` tool, in milliseconds. Default 30,000.
        #[usage(long, env = "AI_AGENT_WEB_TIMEOUT_MS")]
        web_timeout_ms: Option<u64>,
        /// Default remote exec endpoint for every session's file tools (see `run --help`).
        #[usage(long, env = "AI_AGENT_EXEC_URL", conflicts = "exec_cmd")]
        exec_url: Option<String>,
        /// A header for every exec request, `Name: value`. Repeatable.
        #[usage(long, env = "AI_AGENT_EXEC_HEADER")]
        exec_header: Vec<String>,
        /// Largest exec response accepted, in bytes. Default 16 MiB.
        #[usage(long, env = "AI_AGENT_EXEC_MAX_RESPONSE_BYTES")]
        exec_max_response_bytes: Option<u64>,
        /// Run the file tools through a command template instead, e.g. `ssh host -- {}`.
        #[usage(long, env = "AI_AGENT_EXEC_CMD", conflicts = "exec_url")]
        exec_cmd: Option<String>,
        /// POST each prompt's lifecycle events (started, progress, finished) to this URL
        /// (`http(s)://` or `unix://`).
        #[usage(long, env = "AI_AGENT_LIFECYCLE_URL")]
        lifecycle_url: Option<String>,
        /// A header for every lifecycle POST, `Name: value`. Repeatable.
        #[usage(long, env = "AI_AGENT_LIFECYCLE_HEADER")]
        lifecycle_header: Vec<String>,
        /// Seconds between progress POSTs while running. Default 10; `0` turns the timer off.
        #[usage(long, env = "AI_AGENT_LIFECYCLE_HEARTBEAT_SECS", default = "10")]
        lifecycle_heartbeat_secs: u64,
        /// Only offer these tools (comma-separated).
        #[usage(short = 't', long, env = "AI_AGENT_TOOLS", delimiter = ',')]
        tools: Option<Vec<String>>,
        /// Don't offer these tools (comma-separated).
        #[usage(long, env = "AI_AGENT_EXCLUDE_TOOLS", delimiter = ',')]
        exclude_tools: Option<Vec<String>>,
        /// Offer no tools at all.
        #[usage(long)]
        no_tools: bool,
        /// Run a turn's tool calls one at a time instead of concurrently.
        #[usage(long)]
        sequential_tools: bool,
        /// Expose MCP tools through one JavaScript `execute` tool. Needs a build with `--features
        /// code-mode`.
        #[usage(long, env = "AI_AGENT_CODE_MODE")]
        code_mode: bool,
        /// How MCP tools reach the model: `auto` (default), `search` (loaded on demand) or `eager`
        /// (all, always).
        #[usage(long, env = "AI_AGENT_MCP_TOOL_LOADING")]
        mcp_tool_loading: Option<beyond_agent::tools::tool_search::McpLoading>,
        /// Refuse every call to this tool; the model sees an error. Repeatable.
        #[usage(long, env = "AI_AGENT_DENY_TOOL", delimiter = ',')]
        deny_tool: Vec<String>,
        /// Refuse a `bash` command containing this text (case-insensitive). Repeatable.
        #[usage(long, env = "AI_AGENT_DENY_BASH_PATTERN", delimiter = ',')]
        deny_bash_pattern: Vec<String>,
        /// Refuse a `write`/`edit` to a path matching this glob. Repeatable.
        #[usage(long, env = "AI_AGENT_DENY_PATH", delimiter = ',')]
        deny_path: Vec<String>,
        /// Ask attached clients before running tools: `off` (default), `writes`, `all`, or
        /// `tools:<name>,…`.
        #[usage(long, env = "AI_AGENT_APPROVE", default = "off")]
        approve: String,
        /// Seconds to wait for an approval before denying. `0` waits forever.
        #[usage(long, env = "AI_AGENT_APPROVAL_TIMEOUT", default = "300")]
        approval_timeout: u64,
        /// The models `cycle_model` steps through, comma-separated.
        #[usage(long, env = "AI_AGENT_MODELS", delimiter = ',')]
        models: Option<Vec<String>>,
        /// Don't load skills from `~/.claude/skills` or `.claude/skills` (a --skill path still
        /// loads).
        #[usage(long)]
        no_skills: bool,
        /// Allow skills served by MCP servers without asking attached clients.
        #[usage(long, env = "AI_AGENT_APPROVE_MCP_SKILLS")]
        approve_mcp_skills: bool,
        /// Don't load prompt templates from `~/.claude/prompts` or `.claude/prompts`.
        #[usage(long)]
        no_prompt_templates: bool,
        /// Also load skills from this directory or `.md` file. Repeatable.
        #[usage(
            long = "skill",
            env = "AI_AGENT_SKILL_PATH",
            delimiter = ',',
            value_name = "PATH"
        )]
        extra_skill_paths: Vec<String>,
        /// Also load prompt templates from this directory or `.md` file. Repeatable.
        #[usage(
            long = "prompt-template",
            env = "AI_AGENT_PROMPT_TEMPLATE_PATH",
            delimiter = ',',
            value_name = "PATH"
        )]
        extra_prompt_template_paths: Vec<String>,
        /// Name the session if it is new.
        #[usage(short = 'n', long)]
        name: Option<String>,
        /// Add a line to the system prompt's guidelines. Repeatable.
        #[usage(long = "prompt-guideline", value_name = "TEXT")]
        prompt_guidelines: Vec<String>,
    },
    /// List the tools the agent offers the model.
    Tools,
    /// List known model ids and their capabilities. Any other id is passed through as is.
    ListModels {
        /// Only show models whose id fuzzy-matches this text.
        search: Option<String>,
    },
    /// Trust a project directory, so its `.claude/SYSTEM.md` and project settings are used.
    Trust {
        /// The project directory. Defaults to the current directory.
        path: Option<String>,
    },
    /// Mark a project directory untrusted, overriding trust inherited from a parent directory.
    Untrust {
        /// The project directory. Defaults to the current directory.
        path: Option<String>,
    },
    /// Remove a directory's own trust entry, so it inherits from its parents again.
    ClearTrust {
        /// The project directory. Defaults to the current directory.
        path: Option<String>,
    },
    /// Show whether a directory is trusted, untrusted or unknown.
    TrustStatus {
        /// The project directory. Defaults to the current directory.
        path: Option<String>,
    },
    /// Log in to a subscription provider instead of using an API key.
    Login {
        /// `anthropic`, `github-copilot` or `openai-codex`.
        provider: String,
    },
    /// Remove a provider's stored login.
    Logout {
        /// `anthropic`, `github-copilot` or `openai-codex`.
        provider: String,
    },
    /// Show stored subscription logins. Makes no network call.
    AuthStatus {
        /// `anthropic`, `github-copilot` or `openai-codex`. Omit to show all.
        provider: Option<String>,
    },
    /// Log in to an MCP server that requires OAuth (an `http` server in `mcp_servers`).
    McpLogin {
        /// The server's `name` in `mcp_servers`.
        name: String,
    },
    /// Remove an MCP server's stored login.
    McpLogout {
        /// The server's `name` in `mcp_servers`.
        name: String,
    },
    /// Show or change stored defaults for `run` and `serve` (in `~/.beyond/settings.json`). With no
    /// flags, prints them.
    Settings {
        /// Default model.
        #[usage(long)]
        model: Option<String>,
        /// Clear the default model.
        #[usage(long)]
        clear_model: bool,
        /// Default gateway URL.
        #[usage(long)]
        gateway_url: Option<String>,
        /// Clear the default gateway URL.
        #[usage(long)]
        clear_gateway_url: bool,
        /// Default session directory.
        #[usage(long)]
        session_dir: Option<String>,
        /// Clear the default session directory.
        #[usage(long)]
        clear_session_dir: bool,
        /// Default project trust when neither --trust-project nor --force-untrusted is given:
        /// `always`, `never` or `ask`.
        #[usage(long, value_enum)]
        default_project_trust: Option<beyond_agent::settings::TrustPolicy>,
        /// Clear the default project trust.
        #[usage(long)]
        clear_default_project_trust: bool,
        /// Default reasoning effort: off, minimal, low, medium, high or xhigh.
        #[usage(long)]
        default_reasoning_effort: Option<agent_core::ThinkingLevel>,
        /// Clear the default reasoning effort.
        #[usage(long)]
        clear_default_reasoning_effort: bool,
        /// Default for --block-images.
        #[usage(long)]
        block_images: Option<bool>,
        /// Clear the default for --block-images.
        #[usage(long)]
        clear_block_images: bool,
        /// Default for image auto-resize (`false` acts like --no-image-auto-resize).
        #[usage(long)]
        image_auto_resize: Option<bool>,
        /// Clear the default for image auto-resize.
        #[usage(long)]
        clear_image_auto_resize: bool,
        /// Thinking-token budget for one reasoning-effort level, as `<effort>=<tokens>` (e.g.
        /// `high=40000`). Repeatable.
        #[usage(long = "thinking-budget", value_name = "EFFORT=TOKENS")]
        thinking_budget: Vec<String>,
        /// Clear the thinking-token budget for this effort level. Repeatable.
        #[usage(long = "clear-thinking-budget", value_name = "EFFORT")]
        clear_thinking_budget: Vec<String>,
        /// Default --bash-shell-path.
        #[usage(long)]
        default_bash_shell_path: Option<String>,
        /// Clear the default --bash-shell-path.
        #[usage(long)]
        clear_default_bash_shell_path: bool,
        /// Default --bash-command-prefix.
        #[usage(long)]
        default_bash_command_prefix: Option<String>,
        /// Clear the default --bash-command-prefix.
        #[usage(long)]
        clear_default_bash_command_prefix: bool,
        /// Default --compaction-reserve-tokens.
        #[usage(long)]
        default_compaction_reserve_tokens: Option<u32>,
        /// Clear the default --compaction-reserve-tokens.
        #[usage(long)]
        clear_default_compaction_reserve_tokens: bool,
        /// Default --compaction-keep-recent-tokens.
        #[usage(long)]
        default_compaction_keep_recent_tokens: Option<u32>,
        /// Clear the default --compaction-keep-recent-tokens.
        #[usage(long)]
        clear_default_compaction_keep_recent_tokens: bool,
        /// Default --retry-max-retries.
        #[usage(long)]
        default_retry_max_retries: Option<u32>,
        /// Clear the default --retry-max-retries.
        #[usage(long)]
        clear_default_retry_max_retries: bool,
        /// Default --retry-base-delay-ms.
        #[usage(long)]
        default_retry_base_delay_ms: Option<u64>,
        /// Clear the default --retry-base-delay-ms.
        #[usage(long)]
        clear_default_retry_base_delay_ms: bool,
        /// Default --idle-timeout-ms.
        #[usage(long)]
        default_provider_timeout_ms: Option<u64>,
        /// Clear the default --idle-timeout-ms.
        #[usage(long)]
        clear_default_provider_timeout_ms: bool,
        /// Default --retry-max-backoff-ms.
        #[usage(long)]
        default_retry_max_backoff_ms: Option<u64>,
        /// Clear the default --retry-max-backoff-ms.
        #[usage(long)]
        clear_default_retry_max_backoff_ms: bool,
        /// Default `serve --models` list, comma-separated.
        #[usage(long = "default-models", delimiter = ',')]
        default_models: Option<Vec<String>>,
        /// Clear the default --models list.
        #[usage(long)]
        clear_default_models: bool,
        /// Default --skill paths, comma-separated.
        #[usage(long = "default-skill-paths", delimiter = ',')]
        default_skill_paths: Option<Vec<String>>,
        /// Clear the default --skill paths.
        #[usage(long)]
        clear_default_skill_paths: bool,
        /// Default --prompt-template paths, comma-separated.
        #[usage(long = "default-prompt-template-paths", delimiter = ',')]
        default_prompt_template_paths: Option<Vec<String>>,
        /// Clear the default --prompt-template paths.
        #[usage(long)]
        clear_default_prompt_template_paths: bool,
        /// Default --branch-summary-reserve-tokens.
        #[usage(long)]
        default_branch_summary_reserve_tokens: Option<u32>,
        /// Clear the default --branch-summary-reserve-tokens.
        #[usage(long)]
        clear_default_branch_summary_reserve_tokens: bool,
    },
    /// Render a saved session as a self-contained HTML file.
    Export {
        /// The session's `.jsonl` file.
        session: String,
        /// Output path. Defaults to `session-<timestamp>.html` in the current directory.
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
/// Written in help text where the built-in default model belongs, and replaced with
/// [`DEFAULT_MODEL`] when the help is rendered, so the two can't drift apart.
const DEFAULT_MODEL_PLACEHOLDER: &str = "{DEFAULT_MODEL}";

/// Written in help text where the program's own name belongs (`{PROGRAM} trust <path>`), and
/// replaced with [`beyond_agent::PROGRAM`] when the help is rendered.
const PROGRAM_PLACEHOLDER: &str = "{PROGRAM}";

/// A command's rendered `--help`/`-h` page, with [`DEFAULT_MODEL_PLACEHOLDER`] filled in.
fn render_help(cmd: &'static usage::Command, long: bool) -> String {
    Cli::render_help(cmd, long)
        .unwrap_or_default()
        .replace(DEFAULT_MODEL_PLACEHOLDER, DEFAULT_MODEL)
        .replace(PROGRAM_PLACEHOLDER, beyond_agent::PROGRAM)
}

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
            let text = render_help(cmd, long);
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

/// Where the MCP tool manifest is cached, so a boot whose servers are already known starts none of
/// them. `None` disables the cache — every boot then discovers by connecting, the behavior before it
/// existed — which is what happens when there is no `HOME` to put it under.
fn mcp_manifest_dir() -> Option<tools::mcp_manifest::ManifestDir> {
    tools::mcp_manifest::ManifestDir::from_home()
}

/// The `web` tool's isolated HTML parser, dispatched *before* anything else exists.
///
/// Not a `Command` variant, and deliberately not routed through the CLI parser: the point of this
/// process is to be small and single-threaded when its seccomp filter goes on. Building a tokio
/// runtime before the function body runs parks a worker in `epoll_wait` — a syscall the parse
/// allowlist has no reason to permit, which would either widen the filter or kill the process on
/// teardown. Intercepting argv here means the child is one thread that has opened nothing. See
/// `tools::web::isolate`.
fn main() -> std::process::ExitCode {
    let subcommand = std::env::args_os().nth(1);
    if subcommand
        .as_ref()
        .is_some_and(|a| a == tools::web::isolate::SUBCOMMAND)
    {
        std::process::exit(tools::web::isolate::child_main());
    }
    // A stdio MCP server's start: arm its parent-death signal and exec it in place. Nothing of the
    // agent's is wanted here either.
    if subcommand
        .as_ref()
        .is_some_and(|a| a == beyond_agent::subreaper::TRAMPOLINE)
    {
        std::process::exit(beyond_agent::subreaper::trampoline_main());
    }
    // Before anything is spawned: the agent becomes a child subreaper, the main thread turns to
    // reaping the orphans that brings, and the rest runs on a thread of its own.
    beyond_agent::subreaper::run_main(agent_main)
}

/// `main`, past the subcommands that must run before anything exists — on its own thread under
/// [`beyond_agent::subreaper::run_main`]. Returns the exit status.
fn agent_main() -> u8 {
    #[cfg(unix)]
    raise_open_file_limit();

    // Everything else wants the async runtime. Constructed explicitly so the isolate branch above
    // can run first, before any worker exists. The flavour depends on whether this process is a
    // fleet replica — see [`build_runtime`] — which has to be decided from the raw arguments,
    // because the runtime must exist before anything can parse them properly.
    // The ordinary way out (`run` finishing, every subcommand that returns) and a panic unwinding
    // out of `run` both pass through this guard's `Drop`: the same cleanup as `exit_process`.
    let _sweep = tools::mcp_stdio::ExitSweep;
    let runtime = match build_runtime(Mode::process(is_service_mode()).scales_workers_with_cores())
    {
        Ok(runtime) => runtime,
        Err(e) => return report_error(e.as_ref()),
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        runtime.block_on(run_main())
    })) {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => report_error(e.as_ref()),
        // A panic out of `run` (the panic hook has already reported it). Dropping the runtime
        // here would wait for every blocking task — and `serve`'s stdin reader is one, parked on a
        // read that may never return — so the process would hang instead of exiting. Abandon the
        // runtime's threads, clean up exactly as every other exit does, and go, with Rust's own
        // panic status. The whole cleanup, not just the MCP sweep: the unwind drops only what the
        // main future owned, and a daemon's sessions (their own threads) still hold `bash` jobs
        // that only `kill_live_jobs` reaches.
        Err(_) => {
            runtime.shutdown_background();
            finish_process_cleanup();
            std::process::exit(101);
        }
    }
}

/// Print a failed run's error to stderr as one plain line, `error: <message>`, and exit 1. Rust's
/// default for an `Err` out of `main` is the error's `Debug` form (`Error: Os { code: 13, kind:
/// PermissionDenied, .. }`, `Transport("…")` with every quote escaped), which reads as a crash dump.
/// A source chain is appended unless the message already says it.
fn report_error(e: &(dyn std::error::Error + 'static)) -> u8 {
    eprintln!("error: {}", render_error(e));
    1
}

fn render_error(e: &(dyn std::error::Error + 'static)) -> String {
    let mut msg = e.to_string();
    let mut source = e.source();
    while let Some(cause) = source {
        let cause_msg = cause.to_string();
        if !msg.contains(&cause_msg) {
            msg.push_str(": ");
            msg.push_str(&cause_msg);
        }
        source = cause.source();
    }
    msg
}

/// Raise the soft `RLIMIT_NOFILE` to the hard limit. A `serve` daemon holds one descriptor per
/// connection, session file and MCP pipe, and the soft limit a shell hands it is often far below
/// what the hard limit allows (256 on a stock macOS shell, 1024 on most Linux ones) — a limit that
/// costs nothing to lift and that, once hit, turns every `accept` into `EMFILE`. Best effort: a
/// failure leaves the inherited limit in place, which is what the process had anyway.
#[cfg(unix)]
fn raise_open_file_limit() {
    use rustix::process::{Resource, Rlimit, getrlimit, setrlimit};
    let Rlimit { current, maximum } = getrlimit(Resource::Nofile);
    // `None` is unlimited. macOS refuses an unlimited soft limit (and anything above `OPEN_MAX`) for
    // this resource even when the hard limit says unlimited, so the target is capped there; the
    // hard limit itself is left as it was.
    #[cfg(target_os = "macos")]
    let target = Some(maximum.map_or(10_240, |m| m.min(10_240)));
    #[cfg(not(target_os = "macos"))]
    let target = maximum;
    let raise = match (current, target) {
        (Some(soft), Some(target)) => soft < target,
        (Some(_), None) => true,
        (None, _) => false,
    };
    if raise {
        let _ = setrlimit(
            Resource::Nofile,
            Rlimit {
                current: target,
                maximum,
            },
        );
    }
}

/// Stop `run`'s jobs with `how` off the runtime thread. Each kill blocks until its processes are
/// gone, and a remote job's kill waits for its stream's future to let the connection go — a future
/// on this (current-thread) runtime, which a kill run inline would starve: every v1.2 job would cost
/// its kill's full bound, and its stream would close only at the process exit.
async fn stop_jobs(
    jobs: &std::sync::Arc<tools::bash_jobs::BashJobs>,
    how: fn(&tools::bash_jobs::BashJobs) -> Vec<tools::bash_jobs::StoppedJob>,
) -> Vec<tools::bash_jobs::StoppedJob> {
    let jobs = jobs.clone();
    tokio::task::spawn_blocking(move || how(&jobs))
        .await
        .unwrap_or_default()
}

/// [`run`] — raced, in debug builds and only when `BEYOND_AI_AGENT_TEST_PANIC_MAIN_ON` names a
/// path, against that path appearing, which panics **in the main future**: how a test reaches the
/// panic exit path while work (a daemon's sessions, on their own threads) is still running.
async fn run_main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(debug_assertions)]
    if let Some(path) = std::env::var_os("BEYOND_AI_AGENT_TEST_PANIC_MAIN_ON") {
        let path = std::path::PathBuf::from(path);
        let appeared = async {
            while !path.exists() {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        };
        return tokio::select! {
            r = run() => r,
            () = appeared => {
                #[allow(clippy::panic)]
                {
                    panic!("test panic requested in the main future");
                }
            }
        };
    }
    run().await
}

/// What every way out of the process does first: retire every stdio MCP server still running
/// (close its stdin, give it its grace, sweep its process group) and wait for those sweeps and for
/// any `bash` group kill still in flight — see `mcp_stdio::sweep_before_exit`.
fn finish_process_cleanup() {
    // First, so the sweep's wait for pending group kills covers these too.
    tools::bash::kill_live_jobs();
    tools::mcp_stdio::sweep_before_exit();
}

/// `std::process::exit` after [`finish_process_cleanup`]. Every exit of `run` and `serve` that
/// happens after their tools exist goes through here.
fn exit_process(code: i32) -> ! {
    finish_process_cleanup();
    std::process::exit(code)
}

/// Is this process a `serve --service` replica?
///
/// Answered by scanning the raw arguments rather than by parsing them, because the runtime is built
/// before the parser runs. It only has to be right about one flag, and it deliberately mirrors the
/// two ways service mode is actually turned on. A false negative costs the old single-threaded
/// default; a false positive costs a few worker threads on a process that will not use them.
fn is_service_mode() -> bool {
    let mut args = std::env::args_os().skip(1);
    if args.next().is_none_or(|a| a != "serve") {
        return false;
    }
    std::env::args_os().any(|a| a == "--service")
        || std::env::var("AI_AGENT_SERVICE").is_ok_and(|v| !v.is_empty() && v != "0")
}

/// How many tokio *async* worker threads the process runtime should run.
///
/// `None` (unset, empty, or `1`) is a `current_thread` runtime. `Some(0)` is tokio's multi-thread
/// default (one worker per core). `Some(n)` for `n >= 2` pins that many workers.
///
/// Set explicitly, it wins over the mode-based default in [`build_runtime`] in both directions — so
/// `=1` forces a replica back onto the single-threaded scheduler without a rebuild, and `=0` gives a
/// local daemon work-stealing.
fn tokio_worker_threads_from_env() -> Result<Option<usize>, Box<dyn std::error::Error>> {
    match std::env::var("BEYOND_AI_AGENT_TOKIO_WORKER_THREADS") {
        Err(_) => Ok(None),
        Ok(s) if s.is_empty() => Ok(None),
        Ok(s) => {
            let n: usize = s.parse().map_err(|_| {
                format!(
                    "BEYOND_AI_AGENT_TOKIO_WORKER_THREADS={s:?} is not a thread count \
                     (unset/1 = current_thread, 0 = one worker per core, N>=2 = N workers)"
                )
            })?;
            Ok(if n == 1 { None } else { Some(n) })
        }
    }
}

/// How many workers to run, given the environment and the mode.
///
/// Split out from [`build_runtime`] so the decision can be tested without standing up a runtime —
/// and because the one case that is easy to get wrong is invisible otherwise: `configured` is `None`
/// both when the variable is unset *and* when it is explicitly `1`, so an operator pinning a replica
/// back to the single-threaded scheduler looks identical to one who said nothing. `env_set`
/// distinguishes them.
fn runtime_workers(
    configured: Option<usize>,
    env_set: bool,
    scales_workers_with_cores: bool,
) -> Option<usize> {
    match configured {
        Some(n) => Some(n),
        // `Some(0)` is tokio's own default: one worker per core, and `available_parallelism`
        // respects a cgroup CPU quota — so a replica gets workers in proportion to the CPU its task
        // was actually given, which is the thing worth scaling with.
        None if scales_workers_with_cores && !env_set => Some(0),
        None => None,
    }
}

/// What a replica's worker count buys, measured through `crates/fleet-sim` at 120 sessions on 3
/// replicas over real NFS with chaos off — kept here because the tradeoff is not obvious and the
/// last decision was made on a benchmark that could not see it:
///
/// ```text
/// workers   turns/s   vs 1     peak RSS   per replica   MB per turn/s
///    1       274.2    1.00x      649 MB       216 MB        2.37
///    2       385.1    1.40x      854 MB       285 MB        2.22
///    4       514.1    1.88x      774 MB       258 MB        1.51
///    8       619.6    2.26x     1168 MB       389 MB        1.88
///   16       674.6    2.46x     1362 MB       454 MB        2.02
/// ```
///
/// Memory grows **sub-linearly** in the worker count and stays small in absolute terms: even at one
/// worker per core on a 16-core host a replica sits at ~454 MB while serving 40 concurrently active
/// sessions. The per-thread stacks and mimalloc heaps that argued for a single thread are a density
/// budget from the **in-VM** agent, which shares a 768 MB guest with the tenant's workload. A
/// replica outside the sandbox has its own task allocation and nothing to share it with, so the same
/// arithmetic does not apply.
///
/// Four workers is the most *efficient* point (1.51 MB per turn/s) if a deployment is genuinely
/// memory-tight; pin it with `BEYOND_AI_AGENT_TOKIO_WORKER_THREADS=4`.
const _WORKER_TRADEOFF: () = ();

/// The process-wide tokio runtime: `current_thread` for a local daemon, work-stealing for a fleet
/// replica.
///
/// Session tasks (`serve_ws`), the accept loop, the idle reaper, stdio `serve` and one-shot `run`
/// all share it — `serve_session` is `Send`, so there is no per-session OS thread. CPU-bound tool
/// work (`grep`/`find`/image resize) is `spawn_blocking`, which a current-thread runtime still has a
/// blocking pool for.
///
/// **Why the default splits by mode.** A single-threaded runtime was chosen for both, on a benchmark
/// that showed the two schedulers tying on throughput while work-stealing cost roughly nine times
/// the RSS. That benchmark ran **one session** — where nothing contends for the thread, so of course
/// they tied, and per-worker stacks and mimalloc heaps were essentially the whole memory delta. It
/// was the right answer to the question it asked, and the wrong default for a replica.
///
/// Asked again at 120 sessions across 3 replicas, with the fleet simulator driving real sessions
/// over a real NFS mount: the single thread was **87% of all CPU a replica spent**, a turn costing
/// ~7 ms of CPU took ~311 ms of wall time because it crosses that thread a dozen times and queues
/// behind every other session on each crossing, and four workers gave **1.90× the throughput at
/// identical RSS** (456 MB vs 460 MB). At that scale session state dominates memory and the arenas
/// disappear into it.
///
/// So a replica gets work-stealing and a laptop daemon keeps its one thread, because there the
/// single-session measurement is the accurate one.
/// `BEYOND_AI_AGENT_TOKIO_WORKER_THREADS` overrides either direction without a rebuild.
fn build_runtime(
    scales_workers_with_cores: bool,
) -> Result<tokio::runtime::Runtime, Box<dyn std::error::Error>> {
    let choice = runtime_workers(
        tokio_worker_threads_from_env()?,
        std::env::var_os("BEYOND_AI_AGENT_TOKIO_WORKER_THREADS").is_some(),
        scales_workers_with_cores,
    );
    let mut builder = match choice {
        None => tokio::runtime::Builder::new_current_thread(),
        Some(0) => tokio::runtime::Builder::new_multi_thread(),
        Some(n) => {
            let mut b = tokio::runtime::Builder::new_multi_thread();
            b.worker_threads(n);
            b
        }
    };
    builder.enable_all();
    Ok(builder.build()?)
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    // Name the rustls crypto provider for the process up front. Every client-construction site calls
    // this too (it is a `Once`), so this is belt-and-braces rather than the only guard — but doing it
    // first means a provider is in place before any code path, including one added later, can reach a
    // `reqwest::Client`. See `agent_core::tls` for why the provider must be installed at all.
    agent_core::ensure_provider();

    // Always stderr, never stdout: `serve`'s NDJSON control protocol and `run`'s streamed output both
    // live on stdout, and a line-based client reading it can't tell a stray log line from a protocol
    // frame. `RUST_LOG=debug` (or any filter admitting a `warn!`/`info!` already present on a live
    // path — e.g. `session_store/`'s corrupt-line warning, `skills.rs`'s discovery warning) must
    // never corrupt that stream.
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    match cli().command {
        cmd @ Command::Run { .. } => Box::pin(commands::run::run_task(cmd)).await?,
        cmd @ Command::Serve { .. } => Box::pin(commands::serve::serve(cmd)).await?,
        Command::Tools => commands::inspect::tools()?,
        Command::ListModels { search } => commands::inspect::list_models(search),
        Command::Trust { path } => commands::trust::trust(path)?,
        Command::Untrust { path } => commands::trust::untrust(path)?,
        Command::ClearTrust { path } => commands::trust::clear_trust(path)?,
        Command::TrustStatus { path } => commands::trust::trust_status(path)?,
        Command::Login { provider } => Box::pin(commands::auth::login(provider)).await?,
        Command::Logout { provider } => commands::auth::logout(provider)?,
        Command::AuthStatus { provider } => commands::auth::auth_status(provider)?,
        Command::McpLogin { name } => Box::pin(commands::auth::mcp_login(name)).await?,
        Command::McpLogout { name } => commands::auth::mcp_logout(name)?,
        cmd @ Command::Settings { .. } => commands::settings::settings(cmd)?,
        Command::Export { session, output } => commands::inspect::export(session, output)?,
    }
    Ok(())
}

/// What `run`'s session flags select — opened *before* anything model-keyed is resolved, so a resumed
/// session's own model drives the credential, route and capabilities (see `run_task`).
enum RunSession {
    /// An existing session (`--session`, `--session-id`, `--continue`), or a `--fork` of one.
    Existing(Box<(SessionStore, Session)>),
    /// A brand-new session, created only once the invocation has passed every check — a session
    /// created and then abandoned by a failed invocation is what the next `--continue` would resume.
    Fresh(FreshSession),
    /// `--no-session-persistence`.
    Ephemeral,
}

/// Where a [`RunSession::Fresh`] session is created.
enum FreshSession {
    /// A literal `--session <path>` that doesn't exist yet (or is empty).
    AtPath(PathBuf),
    /// This project's repo.
    InRepo(SessionRepo),
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
/// tool-dispatch gate and `ToolConfig::image_auto_resize` already do for a model-issued `read` tool
/// call. Previously neither flag had any effect at all here: this call site
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
            // (`ToolConfig::image_auto_resize`); `_model_supports_vision` is
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

/// Stdin's full contents for an explicit `-` message — read even from a terminal (type, then EOF), since
/// the operator asked for it; trimmed and blank-checked like [`read_stdin_if_piped`].
fn read_stdin_explicit() -> Option<String> {
    let mut buf = String::new();
    match std::io::stdin().lock().read_to_string(&mut buf) {
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

/// How a CLI `run` invocation ended, matching the process's own exit contract.
enum CliTerminal {
    Succeeded { refused: bool },
    Failed { error: String, refused: bool },
    Aborted,
}

async fn emit_cli_lifecycle(
    life: Option<&std::sync::Arc<beyond_agent::lifecycle::Run>>,
    session: &Session,
    kind: CliTerminal,
    structured: Option<serde_json::Value>,
) {
    let Some(life) = life else {
        return;
    };
    let summary = beyond_agent::lifecycle::closing_summary(session);
    match kind {
        CliTerminal::Succeeded { refused } => {
            life.succeeded(session.steps, refused, summary, structured);
        }
        CliTerminal::Failed { error, refused } => {
            life.failed(session.steps, error, refused, summary, structured);
        }
        CliTerminal::Aborted => life.aborted(session.steps, summary),
    }
    life.drain().await;
}

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
        Err(agent_core::Error::Cancelled) => exit_cancelled(shutdown_cause),
        Err(e) => Err(e.into()),
    }
}

/// End a cancelled `run`: `[cancelled]` on stderr, and the triggering signal's POSIX code (else 1).
fn exit_cancelled(shutdown_cause: &std::sync::Mutex<Option<serve::Signal>>) -> ! {
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
    exit_process(code);
}

/// [`run_turn_once`], wrapped with the same whole-run auto-retry `serve/prompt.rs`'s `"prompt"` command gets
/// (see `beyond_agent::retry`) — a run that ends in a transient-looking error (one already
/// exhausted `agent_core`'s own within-turn retries) is re-invoked from scratch against the same
/// session, up to `retry::MAX_RUN_RETRIES` times with backoff, rather than failing a whole `agent run`
/// invocation (plausibly unattended — a cron job, a CI step) outright on a hiccup that `serve` would
/// have quietly recovered from. In `--json` mode a retried attempt's events follow directly after a
/// `[retrying...]` stderr notice — nothing is erased, matching how `serve` demarcates attempts with an
/// `auto_retry_start` frame rather than hiding the failed one. In text mode, stdout must read as the
/// answer: when it isn't a terminal, a failed attempt's buffered text is discarded (see
/// [`TextOut`]); on a terminal, a visible separator follows the already-shown partial text.
#[allow(clippy::too_many_arguments)]
async fn run_turn(
    agent: &Agent,
    session: &mut Session,
    json: bool,
    cancel: &agent_core::CancellationToken,
    retry_policy: &beyond_agent::retry::RunRetryPolicy,
    broken_pipe: &AtomicBool,
    steering: &agent_core::Steering,
    session_memory_active: bool,
    pressure_point: u32,
    life: Option<&std::sync::Arc<beyond_agent::lifecycle::Run>>,
    heartbeat: std::time::Duration,
) -> agent_core::Result<agent_core::StopReason> {
    let mut attempt = 0u32;
    let mut text = TextOut::new(std::io::stdout().is_terminal());
    // One observer for the whole prompt, retries included: a whole-run retry is the same fill cycle,
    // so the pressure nudge must not re-arm across it (`serve` holds its observer the same way).
    let session_steers =
        beyond_agent::memory::SessionSteers::new(session_memory_active, pressure_point);
    loop {
        if let Some(life) = life {
            life.set_attempt(attempt);
        }
        let steps_before = session.steps;
        let result = run_turn_once(
            agent,
            session,
            json,
            &mut text,
            cancel,
            broken_pipe,
            steering,
            &session_steers,
            life,
            heartbeat,
        )
        .await;
        // The retry budget is per stall, not per run: an attempt that committed a step before failing
        // made progress, so a long run that recovers from one blip doesn't carry it against the next.
        if session.steps > steps_before {
            attempt = 0;
        }
        match &result {
            Err(e)
                if attempt < retry_policy.max_retries
                    && !cancel.is_cancelled()
                    && beyond_agent::retry::is_retryable_whole_run(e) =>
            {
                attempt += 1;
                let delay = retry_policy.backoff(attempt);
                eprintln!(
                    "\n[transient error, retrying {attempt}/{}: {e}]",
                    retry_policy.max_retries
                );
                text.discard_attempt(cancel, broken_pipe);
                // Ctrl-C during the wait ends the run now, not after it — with the closing error
                // record still in place, so the persisted transcript stays well-formed.
                tokio::select! {
                    () = tokio::time::sleep(delay) => {}
                    () = cancel.cancelled() => return Err(agent_core::Error::Cancelled),
                }
                // The failed attempt's closing error record must not survive into the retry — see
                // `Session::pop_error_record`'s doc comment (this is the same run resuming from
                // scratch, not a fresh prompt).
                session.pop_error_record();
            }
            _ => {
                // A final failure (or a cancel) keeps the partial text the session keeps too.
                text.flush(cancel, broken_pipe);
                return result;
            }
        }
    }
}

/// Text-mode stdout for `run`: what a script capturing stdout reads must be the answer, not the answer
/// glued onto the partial text of attempts that failed and were retried. When stdout is not a
/// terminal, each step's output is held until the step completes (`AgentEvent::TurnEnd`) and a retried
/// attempt's held output is dropped. On a terminal, output streams live (a person is watching) and a
/// retry writes a visible separator instead, since what's on screen can't be taken back.
struct TextOut {
    live: bool,
    pending: String,
    /// Whether anything of the current attempt reached the terminal (gates the separator).
    shown: bool,
    /// The last thing written was a tool marker or its streamed arguments, which carry no trailing
    /// newline; the next model text must start on a fresh line.
    after_tool: bool,
}

impl TextOut {
    fn new(live: bool) -> Self {
        Self {
            live,
            pending: String::new(),
            shown: false,
            after_tool: false,
        }
    }

    /// Model text. Starts on its own line when it follows a tool marker.
    fn write_text(
        &mut self,
        text: &str,
        cancel: &agent_core::CancellationToken,
        broken_pipe: &AtomicBool,
    ) {
        if text.is_empty() {
            return;
        }
        if std::mem::take(&mut self.after_tool) {
            self.write("\n", cancel, broken_pipe);
        }
        self.write(text, cancel, broken_pipe);
    }

    /// A tool marker or its argument fragments: leaves the line open.
    fn write_tool(
        &mut self,
        text: &str,
        cancel: &agent_core::CancellationToken,
        broken_pipe: &AtomicBool,
    ) {
        self.after_tool = true;
        self.write(text, cancel, broken_pipe);
    }

    fn write(
        &mut self,
        text: &str,
        cancel: &agent_core::CancellationToken,
        broken_pipe: &AtomicBool,
    ) {
        if self.live {
            self.shown |= !text.is_empty();
            write_stdout_or_exit(text, cancel, broken_pipe);
        } else {
            self.pending.push_str(text);
        }
    }

    /// The step completed: its output is final.
    fn flush(&mut self, cancel: &agent_core::CancellationToken, broken_pipe: &AtomicBool) {
        self.shown = false;
        if !self.pending.is_empty() {
            write_stdout_or_exit(&self.pending, cancel, broken_pipe);
            self.pending.clear();
        }
    }

    /// The in-flight attempt failed and is about to be re-sent: nothing it streamed is kept.
    fn discard_attempt(
        &mut self,
        cancel: &agent_core::CancellationToken,
        broken_pipe: &AtomicBool,
    ) {
        self.pending.clear();
        self.after_tool = false;
        if std::mem::take(&mut self.shown) {
            write_stdout_or_exit(RETRY_SEPARATOR, cancel, broken_pipe);
        }
    }
}

/// Written to a terminal stdout after a failed attempt's partial text, before the retry streams.
const RETRY_SEPARATOR: &str = "\n[attempt failed; the text above is discarded, retrying]\n";

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
/// process to report loudly — matches `serve/`'s own writer task, which already exits gracefully on a
/// write failure instead of panicking.
///
/// On `BrokenPipe`, sets `broken_pipe` and trips `cancel` rather than calling `process::exit`
/// directly — this function is called from deep inside `run_turn_once`'s streaming callbacks, which
/// run *before* `run_task`'s `persist_run`; exiting right here used to skip that persist entirely,
/// silently dropping the turn's final (possibly-partial) assistant message from the session store —
/// exactly the loss the Ctrl-C/SIGTERM path already takes care to avoid. Tripping `cancel` instead
/// reuses that identical, already-correct "stop the turn, then still persist" machinery
/// (`agent_core::Agent::run_events_cancellable`'s cooperative cancellation); `run_task` checks
/// `broken_pipe` right after its own unconditional `persist_run` call and exits 0 there instead —
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
    text_out: &mut TextOut,
    cancel: &agent_core::CancellationToken,
    broken_pipe: &AtomicBool,
    steering: &agent_core::Steering,
    // The `/session` steers (pressure nudge, compaction reminder) — the observer `serve` shares. See
    // `beyond_agent::memory::SessionSteers`.
    session_steers: &beyond_agent::memory::SessionSteers,
    life: Option<&std::sync::Arc<beyond_agent::lifecycle::Run>>,
    heartbeat_every: std::time::Duration,
) -> agent_core::Result<agent_core::StopReason> {
    let session_steers = |ev: &agent_core::AgentEvent| {
        if let Some(journal) = RUN_JOURNAL.get() {
            journal.observe(ev);
        }
        if let agent_core::AgentEvent::WrapUp { .. } = ev {
            RUN_WRAPPED_UP.store(true, Ordering::Relaxed);
        }
        session_steers.observe(ev, steering);
    };
    let mut stop_reason = agent_core::StopReason::default();
    let life_obs = life.cloned();
    let heartbeat = match (life.cloned(), heartbeat_every) {
        (Some(life), every) if !every.is_zero() => Some(tokio::spawn(async move {
            let Some(mut interval) = beyond_agent::lifecycle::heartbeat_interval(every) else {
                return;
            };
            interval.tick().await;
            loop {
                interval.tick().await;
                life.heartbeat();
            }
        })),
        _ => None,
    };
    let result = if json {
        agent
            .run_events_steered(
                session,
                |ev| {
                    if let Some(life) = &life_obs {
                        life.observe(&ev);
                    }
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
            .await
    } else {
        // Task 1 (pi-parity fix, pass 19): `run_events_steered` directly (rather than the plain
        // `run_cancellable`, which always builds its own default `Steering::new()` internally), so `run`'s
        // own resolved `steering_mode`/`follow_up_mode` (see `run_task`'s construction of `steering`) is
        // actually in effect at agent/session construction time, matching pi's own agent construction in
        // every mode — same `AgentEvent::Stream` filter `Agent::run_cancellable` itself applies internally.
        let r = agent
            .run_events_steered(
                session,
                |ev| {
                    if let Some(life) = &life_obs {
                        life.observe(&ev);
                    }
                    session_steers(&ev);
                    if let agent_core::AgentEvent::TurnEnd { .. } = &ev {
                        text_out.flush(cancel, broken_pipe);
                    }
                    if let agent_core::AgentEvent::Retry {
                        attempt,
                        delay_ms,
                        rate_limited,
                        overloaded,
                        message,
                    } = &ev
                    {
                        // A rate-limit wait can run for minutes; say so rather than look hung.
                        let what = if *overloaded {
                            "provider overloaded"
                        } else if *rate_limited {
                            "rate limited"
                        } else {
                            "stream error"
                        };
                        eprintln!(
                            "\n[{what}, retrying in {:.1}s (attempt {attempt}): {message}]",
                            *delay_ms as f64 / 1000.0
                        );
                        text_out.discard_attempt(cancel, broken_pipe);
                    }
                    let agent_core::AgentEvent::Stream(ev) = &ev else {
                        return;
                    };
                    match ev {
                        StreamEvent::TextDelta { text, .. } => {
                            text_out.write_text(text, cancel, broken_pipe);
                        }
                        StreamEvent::ToolUseStart { name, .. } => {
                            // No trailing newline: `InputJsonDelta` fragments print immediately after, live,
                            // on this same line — a growing preview of the call's arguments as they stream
                            // in, rather than the model appearing to hang until the whole call (and its
                            // result) land.
                            text_out.write_tool(&format!("\n[tool: {name}] "), cancel, broken_pipe);
                        }
                        StreamEvent::InputJsonDelta { partial_json, .. } => {
                            text_out.write_tool(partial_json, cancel, broken_pipe);
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
            .await;
        if r.is_ok() {
            text_out.write("\n", cancel, broken_pipe);
            text_out.flush(cancel, broken_pipe);
        }
        r
    };
    if let Some(hb) = heartbeat {
        hb.abort();
    }
    result?;
    Ok(stop_reason)
}

/// A [`agent_core::CheckpointHook`] for one-shot `run`. Unlike `serve`'s channel-based
/// `ChannelCheckpoint` (which forwards through an `mpsc` channel to avoid stalling a `select!` loop
/// reading stdin concurrently), `run` has no concurrent event source to interleave with — a direct
/// blocking append inside the async callback is the simplest correct thing here, not a missing
/// optimization. Persists every mid-run checkpoint incrementally, the same guarantee `serve` gives
/// every session: without this, only the *end* of each whole turn was ever persisted (via
/// `persist_run`, after `run_turn` returns), so a crash mid-turn — after several tool
/// round-trips already ran real commands or edited real files — lost all record of them, with the
/// session file (if any) unable to distinguish that from "nothing happened yet".
/// `run`'s MCP task journal: the same `mcp_task` custom entries `serve` writes (see
/// `beyond_agent::mcp_resume`), so a `run` that dies mid-task can be resumed by
/// `run --continue` (or by `serve`). Process-wide because `run` drives exactly one session per
/// process; set once, when that session is persisted.
static RUN_JOURNAL: std::sync::OnceLock<RunJournal> = std::sync::OnceLock::new();

/// Set when this process's run reached its `--max-steps` ceiling and the model was granted its
/// wrap-up steps (`AgentEvent::WrapUp`): the run then ends normally on the model's own wrap-up answer,
/// and the end of `run_task` says so on stderr. Process-wide for the same reason as [`RUN_JOURNAL`].
static RUN_WRAPPED_UP: AtomicBool = AtomicBool::new(false);

struct RunJournal {
    store: Arc<std::sync::Mutex<Option<SessionStore>>>,
    session_id: String,
}

impl RunJournal {
    /// Journal a task record a tool emitted (`tool_progress` `details.mcpTask`).
    fn observe(&self, ev: &agent_core::AgentEvent) {
        let agent_core::AgentEvent::ToolProgress {
            id,
            details: Some(details),
            ..
        } = ev
        else {
            return;
        };
        let Some(record) = details.get(tools::mcp::MCP_TASK_DETAILS_KEY) else {
            return;
        };
        self.append(
            beyond_agent::mcp_resume::TASK_ENTRY_KIND,
            beyond_agent::mcp_resume::task_entry(record, id, &self.session_id),
        );
    }

    fn append(&self, kind: &str, data: serde_json::Value) {
        let mut guard = self.store.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(store) = guard.as_mut()
            && let Err(e) = store.append_journal(kind, data)
        {
            eprintln!("run: failed to journal a session entry ({kind}): {e}");
        }
    }
}

/// Resolve the MCP tasks a previous process left in flight on this session (journaled by `run` or
/// `serve`), before the next turn, so the model sees what actually happened. `run` has no client to
/// keep company while it waits, so each is polled to its result in turn, with a line on stderr;
/// Ctrl-C ends the wait as it ends the run. Every result is journaled as it lands, and returned for
/// the caller to splice into the turn it is about to send, with the journal's placeholder marks
/// (the calls whose `tool_result` on the path is only a "result pending" placeholder).
async fn resume_mcp_tasks_for_run(
    journal: Option<&RunJournal>,
    session: &Session,
    catalog: &tools::mcp::McpCatalog,
    cancel: &agent_core::CancellationToken,
) -> agent_core::Result<(
    Vec<beyond_agent::mcp_resume::Resolved>,
    std::collections::HashSet<String>,
)> {
    use beyond_agent::mcp_resume::{
        PLACEHOLDER_ENTRY_KIND, RESULT_ENTRY_KIND, Resolved, TASK_ENTRY_KIND,
    };
    let Some(journal) = journal else {
        return Ok(Default::default());
    };
    let (tasks, results, mut placeholders) = {
        let guard = journal.store.lock().unwrap_or_else(|e| e.into_inner());
        match guard.as_ref() {
            Some(store) => (
                store.active_journal(TASK_ENTRY_KIND),
                beyond_agent::mcp_resume::results(&store.active_journal(RESULT_ENTRY_KIND)),
                beyond_agent::mcp_resume::placeholder_ids(
                    &store.active_journal(PLACEHOLDER_ENTRY_KIND),
                ),
            ),
            None => return Ok(Default::default()),
        }
    };
    let mut pending = beyond_agent::mcp_resume::pending(
        &session.messages,
        &tasks,
        &results,
        &placeholders,
        &journal.session_id,
    );
    // Only a configured server is ever polled.
    pending.retain(|p| catalog.has_server(&p.record.server));
    let mut resolved = Vec::with_capacity(pending.len());
    for task in pending {
        eprintln!(
            "[resuming MCP task {} ({}) left in flight by a previous run]",
            task.record.task_id, task.name
        );
        // A task still working can be polled for as long as it runs: Ctrl-C ends the wait (the
        // task stays journaled for the next `--continue`) rather than being ignored until it lands.
        let host = tools::mcp::host();
        let resumed = tokio::select! {
            r = catalog.resume_task(task.record.clone(), &host, None) => r,
            () = cancel.cancelled() => return Err(agent_core::Error::Cancelled),
        };
        let result = match resumed {
            Ok(output) => Resolved {
                tool_use_id: task.tool_use_id,
                name: task.name,
                content: output.text,
                is_error: false,
                images: output.images,
                pending: false,
            },
            // The server is unreachable: the turn is told the result is pending, and the task
            // stays journaled for the next `run --continue`.
            Err(tools::mcp::ResumeError::Unreachable(why)) => {
                eprintln!("[MCP task {} is still pending: {why}]", task.record.task_id);
                journal.append(
                    PLACEHOLDER_ENTRY_KIND,
                    beyond_agent::mcp_resume::placeholder_entry(
                        &task.tool_use_id,
                        &journal.session_id,
                    ),
                );
                placeholders.insert(task.tool_use_id.clone());
                resolved.push(beyond_agent::mcp_resume::pending_placeholder(&task, &why));
                continue;
            }
            Err(tools::mcp::ResumeError::Final(e)) => Resolved {
                tool_use_id: task.tool_use_id,
                name: task.name,
                content: e.to_string(),
                is_error: true,
                images: Vec::new(),
                pending: false,
            },
        };
        let mut data = serde_json::to_value(&result).unwrap_or_default();
        data["sessionId"] = serde_json::json!(journal.session_id);
        journal.append(RESULT_ENTRY_KIND, data);
        resolved.push(result);
    }
    Ok((resolved, placeholders))
}

/// `run`'s checkpoint hook: [`persist_run`] against the run's shared store.
struct DirectCheckpoint(RunStore);

#[async_trait::async_trait]
impl agent_core::CheckpointHook for DirectCheckpoint {
    /// Append what is new, and say whether it landed — matching `serve`'s own checkpoint hook. The
    /// agent acts on a failure only before a tool dispatch, where it fails closed (the tool does not
    /// run: `agent_core::CheckpointHook`); anywhere else the next successful persist, or
    /// `persist_run` after the turn ends, catches up whatever this attempt missed.
    async fn checkpoint(&self, session: &Session) -> Result<(), String> {
        persist_run(&self.0, session).map_err(|e| {
            eprintln!("run: failed to persist checkpoint: {e}");
            e.to_string()
        })
    }
}

/// `run`'s session store (`None` when not persisting), shared by [`DirectCheckpoint`], the
/// end-of-turn [`persist_run`] and the run journal.
type RunStore = Arc<std::sync::Mutex<Option<SessionStore>>>;

/// Persist `session` ([`SessionStore::persist`]: a compaction since the last write, then any masked
/// results, then whatever is new). Called at every checkpoint and after each whole turn (a checkpoint
/// never fires for the turn's own final assistant message; see `agent_core::Agent::run_turn`'s doc
/// comment on where checkpoints land). A no-op when `run` isn't persisting at all, or when nothing is
/// new. `Agent::compact` checkpoints right after its rewrite, so it lands before anything else.
fn persist_run(run: &RunStore, session: &Session) -> std::io::Result<()> {
    let mut guard = run.lock().unwrap_or_else(|e| e.into_inner());
    let Some(store) = guard.as_mut() else {
        return Ok(());
    };
    store.persist(
        &session.messages,
        &session.compaction,
        &session.edited,
        &session.cuts,
    )
}

/// Expand an explicit `/skill:name` invocation first (its own prefix, so it can't collide with a
/// `/name` prompt template), then fall through to prompt-template expansion — a no-op on whichever
/// message reaches it unmatched. Mirrors `serve`'s own `"prompt"` handler exactly (see `serve/`).
///
/// An MCP-served skill (`/skill:<server>:<name>`) is tried first; its `SKILL.md` is fetched now, at the
/// invocation, never prefetched (see `tools::mcp_skills`).
async fn expand_message(
    message: &str,
    skills: &[beyond_agent::skills::Skill],
    prompt_templates: &[beyond_agent::prompts::PromptTemplate],
    mcp_skills: &tools::mcp_skills::McpSkills,
) -> String {
    if let Some(expanded) = mcp_skills.expand_invocation(message, skills).await {
        return expanded;
    }
    let message = beyond_agent::skills::expand_if_skill_invocation(message, skills);
    beyond_agent::prompts::expand_if_slash(&message, prompt_templates)
}

/// Whether `id` is safe to embed directly in a filename component. Delegates to the canonical
/// implementation in [`beyond_agent::session_store::is_valid_session_id`] (shared with the
/// WebSocket transport, which validates a client-supplied `?session_id=` the same way).
fn is_valid_session_id(id: &str) -> bool {
    beyond_agent::session_store::is_valid_session_id(id)
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
    use beyond_agent::session_store::SessionMeta;

    /// A shell's low soft descriptor limit (256 on stock macOS) is lifted to the hard limit.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_soft_open_file_limit_is_raised_to_the_hard_limit() {
        use rustix::process::{Resource, Rlimit, getrlimit, setrlimit};
        let before = getrlimit(Resource::Nofile);
        if before.maximum.is_some_and(|m| m <= 256) {
            return; // Nothing to lower from on this host.
        }
        setrlimit(
            Resource::Nofile,
            Rlimit {
                current: Some(256),
                maximum: before.maximum,
            },
        )
        .unwrap();
        raise_open_file_limit();
        let after = getrlimit(Resource::Nofile);
        assert_eq!(after.current, before.maximum);
        assert_eq!(after.maximum, before.maximum);
    }

    /// The TTL a real command line resolves to, through the same `CacheTtl::resolve` call and mode
    /// default `main` uses for that subcommand.
    fn resolved_cache_long(args: &[&str]) -> bool {
        match parse_cli(args).expect("parses").command {
            Command::Run {
                cache_ttl,
                cache_long,
                ..
            } => CacheTtl::resolve(cache_ttl, cache_long, CacheTtl::FiveMinutes),
            Command::Serve {
                cache_ttl,
                cache_long,
                ..
            } => CacheTtl::resolve(cache_ttl, cache_long, CacheTtl::OneHour),
            _ => unreachable!("only run and serve carry a cache TTL"),
        }
    }

    #[test]
    fn cache_ttl_defaults_to_5m_for_run_and_1h_for_serve() {
        assert!(!resolved_cache_long(&["a", "run", "hi"]));
        assert!(resolved_cache_long(&["a", "serve"]));
    }

    #[test]
    fn an_explicit_cache_ttl_overrides_the_mode_default() {
        assert!(resolved_cache_long(&[
            "a",
            "run",
            "--cache-ttl",
            "1h",
            "hi"
        ]));
        assert!(!resolved_cache_long(&["a", "serve", "--cache-ttl", "5m"]));
        // `--cache-long` still means 1h, and an explicit `--cache-ttl` beats it.
        assert!(resolved_cache_long(&["a", "run", "--cache-long", "hi"]));
        assert!(!resolved_cache_long(&[
            "a",
            "run",
            "--cache-long",
            "--cache-ttl",
            "5m",
            "hi"
        ]));
    }

    #[test]
    fn cache_ttl_rejects_anything_but_5m_or_1h() {
        assert!(parse_cli(&["a", "serve", "--cache-ttl", "24h"]).is_none());
    }

    fn parse_cli(args: &[&str]) -> Option<Cli> {
        let os: Vec<&OsStr> = args.iter().map(OsStr::new).collect();
        Cli::try_parse_from(&os).ok()
    }

    #[test]
    fn a_broken_pipe_trips_cancel_and_the_flag_instead_of_exiting_on_the_spot() {
        // Regression: `write_stdout_or_exit` used to call `process::exit(0)` directly on a broken
        // pipe, which — because it's invoked from deep inside `run_turn_once`'s streaming callbacks —
        // ran *before* `run_task`'s `persist_run`, silently dropping the turn's final assistant
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

    #[test]
    fn help_names_the_real_default_model() {
        for sub in ["run", "serve"] {
            let cmd = Cli::command()
                .subcommands
                .iter()
                .copied()
                .find(|c| c.name == sub)
                .unwrap_or_else(|| panic!("no {sub} subcommand"));
            for long in [false, true] {
                let help = render_help(cmd, long);
                let normalized: String = help.split_whitespace().collect::<Vec<_>>().join(" ");
                assert!(
                    normalized.contains(&format!("else {DEFAULT_MODEL}")),
                    "{sub} --model help must name {DEFAULT_MODEL}: {help}"
                );
                assert!(!help.contains(DEFAULT_MODEL_PLACEHOLDER), "{help}");
            }
        }
    }

    #[test]
    fn help_names_the_program_by_its_real_name() {
        let mut pages = vec![render_help(Cli::command(), true)];
        for sub in Cli::command().subcommands.iter().copied() {
            pages.push(render_help(sub, true));
            pages.push(render_help(sub, false));
        }
        for page in pages {
            let words: String = page.split_whitespace().collect::<Vec<_>>().join(" ");
            assert!(!words.contains(PROGRAM_PLACEHOLDER), "{page}");
            for wrong in ["`agent ", "| agent "] {
                assert!(
                    !words.contains(wrong),
                    "help names a command that doesn't exist: {page}"
                );
            }
        }
    }

    /// `--help` is for someone running the command: what each flag does, in a line or two. Design
    /// rationale belongs in `crates/agent/ARCHITECTURE.md` ("CLI flags: rationale"), not here.
    #[test]
    fn help_is_short_user_facing_text() {
        let top = render_help(Cli::command(), true);
        assert!(
            top.lines().count() <= 60,
            "top-level help is {} lines",
            top.lines().count()
        );
        for sub in Cli::command().subcommands.iter().copied() {
            for long in [false, true] {
                let page = render_help(sub, long);
                let lines = page.lines().count();
                let budget = match sub.name {
                    "run" | "serve" => 350,
                    _ => 120,
                };
                assert!(
                    lines <= budget,
                    "`{}` help is {lines} lines (budget {budget})",
                    sub.name
                );
                let text = page.to_ascii_lowercase();
                for banned in [
                    "pi-parity",
                    "pi's ",
                    "(pi",
                    "not yet implemented",
                    "::",
                    ".ts`",
                    ".rs`",
                    "task #",
                ] {
                    assert!(
                        !text.contains(banned),
                        "`{}` help mentions {banned:?}:\n{page}",
                        sub.name
                    );
                }
                // Development history: "Task 12", "Fix 3", "Round 2", "pass 19".
                let words: Vec<&str> = text.split_whitespace().collect();
                for pair in words.windows(2) {
                    let numbered = pair[1]
                        .trim_matches(|c: char| !c.is_ascii_alphanumeric())
                        .parse::<u32>()
                        .is_ok();
                    assert!(
                        !(numbered && ["task", "fix", "round", "pass"].contains(&pair[0])),
                        "`{}` help carries development history ({} {}):\n{page}",
                        sub.name,
                        pair[0],
                        pair[1]
                    );
                }
            }
        }
    }

    #[test]
    fn an_error_renders_as_its_message_and_its_causes() {
        #[derive(Debug)]
        struct Outer(std::io::Error);
        impl std::fmt::Display for Outer {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("cannot open session")
            }
        }
        impl std::error::Error for Outer {
            fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                Some(&self.0)
            }
        }
        let e = Outer(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "denied",
        ));
        assert_eq!(render_error(&e), "cannot open session: denied");
        let plain: Box<dyn std::error::Error> = "--fork x: no such session".into();
        assert_eq!(render_error(plain.as_ref()), "--fork x: no such session");
    }

    /// In-process argv parse cost only (no process start, no help rendering). Printed under
    /// `--nocapture` so a release-mode run can be compared against process-level hyperfine numbers:
    /// process startup for this binary is ~2 ms either way; the parse itself is microseconds.
    #[test]
    fn cli_parse_in_process_microbench() {
        use std::hint::black_box;
        use std::time::Instant;

        let cases: &[&[&str]] = &[
            &["agent", "tools"],
            &["agent", "run", "--model", "claude-opus-4-8", "hi"],
            &[
                "agent",
                "serve",
                "--upstream-http2",
                "off",
                "--tools",
                "bash,read,write",
            ],
            &["agent", "settings", "--model", "claude-sonnet-4-5"],
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
        // very end, via `persist_run` (which only ever runs after `run_turn` returns `Ok`, and so
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
        let store: RunStore = Arc::new(std::sync::Mutex::new(Some(store)));

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
            &beyond_agent::retry::RunRetryPolicy::default(),
            &AtomicBool::new(false),
            &agent_core::Steering::new(),
            false,
            u32::MAX,
            None,
            std::time::Duration::ZERO,
        )
        .await
        .unwrap();

        // Read independently of `store` (which the test itself still holds a live handle to) —
        // exactly what a process restarting after a crash would do.
        let (_, disk_session) = SessionStore::open(session_path).unwrap();
        assert!(
            disk_session.messages.len() >= 4,
            "checkpoints during the run must have persisted the tool round-trips that already \
             happened, not just whatever `persist_run` would add after the fact: {:?}",
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
        // (`beyond_agent::tools::write::Write`) and the real `agent_core::Agent`
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
        // `is_retryable_mid_stream`'s "stream ended before" check), exhausting it — the resulting `Err` is
        // exactly what propagates out to `agent.run(...)` inside `run_turn_once`. Our new whole-run
        // wrapper (`run_turn`) must catch that and retry the whole call again, which finally succeeds.
        let mut turns: Vec<Vec<Result<StreamEvent, Error>>> = (0..INNER_RETRY_ATTEMPTS)
            .map(|_| {
                vec![Err(Error::Transport(
                    "Anthropic stream ended before message_stop".into(),
                ))]
            })
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
            &beyond_agent::retry::RunRetryPolicy::default(),
            &AtomicBool::new(false),
            &agent_core::Steering::new(),
            false,
            u32::MAX,
            None,
            std::time::Duration::ZERO,
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
            (beyond_agent::retry::MAX_RUN_RETRIES as usize + 1) * INNER_RETRY_ATTEMPTS;
        let turns: Vec<Vec<Result<StreamEvent, Error>>> = (0..total_attempts)
            .map(|_| {
                vec![Err(Error::Transport(
                    "Anthropic stream ended before message_stop".into(),
                ))]
            })
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
            &beyond_agent::retry::RunRetryPolicy::default(),
            &AtomicBool::new(false),
            &agent_core::Steering::new(),
            false,
            u32::MAX,
            None,
            std::time::Duration::ZERO,
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
            .map(|_| {
                vec![Err(Error::Transport(
                    "Anthropic stream ended before message_stop".into(),
                ))]
            })
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
        let policy =
            beyond_agent::retry::RunRetryPolicy::from_overrides_with_enabled(false, Some(99), None);
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
            None,
            std::time::Duration::ZERO,
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

    #[tokio::test]
    async fn the_whole_run_retry_ceiling_counts_stalls_not_blips_over_a_long_run() {
        // The eval failure's other half: a long run that recovered from earlier blips died on a later
        // one because the ceiling counted every retry of the run. With a ceiling of 1, a run that fails,
        // recovers and commits a step, then fails again must still get its retry — the second failure
        // is a new stall, not the run's second strike.
        let fail = || {
            (0..INNER_RETRY_ATTEMPTS)
                .map(|_| {
                    vec![Err(Error::Transport(
                        "Anthropic stream ended before message_stop".into(),
                    ))]
                })
                .collect::<Vec<Vec<Result<StreamEvent, Error>>>>()
        };
        let mut turns = fail();
        turns.push(
            turn::tool_call("t1", "missing", "{}")
                .into_iter()
                .map(Ok)
                .collect(),
        );
        turns.extend(fail());
        turns.push(turn::text("finished").into_iter().map(Ok).collect());
        let transport = std::sync::Arc::new(MockTransport::scripted(turns));
        let agent = Agent::new(transport.clone(), "claude-test");
        let mut session = Session::new();
        session.user("hi");
        let policy = beyond_agent::retry::RunRetryPolicy::from_overrides(
            Some(1),
            Some(std::time::Duration::from_millis(1)),
        );

        run_turn(
            &agent,
            &mut session,
            false,
            &agent_core::CancellationToken::new(),
            &policy,
            &AtomicBool::new(false),
            &agent_core::Steering::new(),
            false,
            u32::MAX,
            None,
            std::time::Duration::ZERO,
        )
        .await
        .expect("a failure after progress is a fresh stall with its own retry");
        assert_eq!(transport.calls(), 2 * INNER_RETRY_ATTEMPTS + 2);
        assert!(format!("{:?}", session.messages).contains("finished"));
    }

    #[tokio::test]
    async fn cancel_during_a_whole_run_retry_wait_returns_at_once_with_the_record_kept() {
        let turns: Vec<Vec<Result<StreamEvent, Error>>> = (0..INNER_RETRY_ATTEMPTS)
            .map(|_| {
                vec![Err(Error::Transport(
                    "Anthropic stream ended before message_stop".into(),
                ))]
            })
            .collect();
        let transport = std::sync::Arc::new(MockTransport::scripted(turns));
        let agent = Agent::new(transport.clone(), "claude-test");
        let mut session = Session::new();
        session.user("hi");
        // A 60s whole-run backoff the cancel has to cut short.
        let policy = beyond_agent::retry::RunRetryPolicy::from_overrides(
            Some(3),
            Some(std::time::Duration::from_secs(60)),
        );
        let cancel = agent_core::CancellationToken::new();
        let trigger = cancel.clone();
        tokio::spawn(async move {
            // After agent_core's own ~1.75s of mid-stream retries, inside the whole-run wait.
            tokio::time::sleep(std::time::Duration::from_millis(3000)).await;
            trigger.cancel();
        });
        let started = std::time::Instant::now();
        let err = run_turn(
            &agent,
            &mut session,
            false,
            &cancel,
            &policy,
            &AtomicBool::new(false),
            &agent_core::Steering::new(),
            false,
            u32::MAX,
            None,
            std::time::Duration::ZERO,
        )
        .await
        .expect_err("cancelled");
        assert!(matches!(err, Error::Cancelled), "{err:?}");
        assert!(started.elapsed() < std::time::Duration::from_secs(10));
        assert!(
            session.messages.last().unwrap().error_message.is_some(),
            "the closing error record stays, so the transcript keeps its role alternation"
        );
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
        let settings = beyond_agent::settings::Settings::default();
        assert_eq!(resolve_thinking_budget_overrides(&settings), None);
    }

    #[test]
    fn resolve_thinking_budget_overrides_parses_recognized_wire_strings() {
        let mut table = std::collections::BTreeMap::new();
        table.insert("high".to_string(), 40_000u32);
        table.insert("low".to_string(), 1_000u32);
        let settings = beyond_agent::settings::Settings {
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
        let settings = beyond_agent::settings::Settings {
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

#[cfg(test)]
mod runtime_tests {
    use super::runtime_workers;

    /// The decision table, including the case that reads the same as "unset" unless you look.
    #[test]
    fn a_replica_gets_workers_and_a_daemon_does_not_unless_told() {
        // Nothing set: the laptop daemon keeps its one thread; the replica gets a capped worker
        // count, never more than the cores it actually has.
        assert_eq!(runtime_workers(None, false, false), None);
        // `Some(0)` is tokio's own "one worker per core", not a literal zero workers.
        assert_eq!(runtime_workers(None, false, true), Some(0));

        // `=1` parses to `None`, exactly like unset — but it is an operator pinning a replica back
        // onto the single-threaded scheduler, and it has to win.
        assert_eq!(runtime_workers(None, true, true), None);
        assert_eq!(runtime_workers(None, true, false), None);

        // An explicit count wins in both directions, including work-stealing for a local daemon.
        assert_eq!(runtime_workers(Some(4), true, false), Some(4));
        assert_eq!(runtime_workers(Some(0), true, false), Some(0));
        assert_eq!(runtime_workers(Some(2), true, true), Some(2));
    }
}
