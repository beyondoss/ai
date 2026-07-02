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

use std::io::{IsTerminal, Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use agent_core::{Agent, GatewayClient, Session, StreamEvent};
use beyond_ai_agent::session_store::{
    SessionMeta, SessionRepo, SessionStore, canonical_cwd, default_session_dir,
};
use beyond_ai_agent::{serve, tools};
use clap::{Parser, Subcommand};

/// Default model when neither `--model` nor `AI_AGENT_MODEL` is set.
const DEFAULT_MODEL: &str = "claude-opus-4-8";
/// Default gateway base URL.
const DEFAULT_GATEWAY: &str = "http://ai.internal";

/// The agent's base identity/instructions. The tool list is generated from `registry` — the tools this
/// process actually registered, after any `--tools`/`--exclude-tools`/`--no-tools` filtering — rather
/// than hand-listed as a static string or assumed to be the full default set. A prior hardcoded version
/// silently omitted the Beyond platform tools (fork/sync/logs) entirely, and a version that always
/// listed `default_registry()` regardless of filtering would claim tools a restricted agent doesn't
/// actually have, inviting the model to call one that gets rejected.
fn default_system_prompt(registry: &agent_core::ToolRegistry) -> String {
    let names: Vec<String> = registry.definitions().into_iter().map(|d| d.name).collect();
    format!(
        "You are the Beyond coding agent. You operate inside a real working directory with tools: {}. \
         Use them to accomplish the user's task directly — inspect before you change, make minimal \
         edits, and verify your work. Be concise.",
        names.join(", ")
    )
}

/// Parse `--reasoning-effort`'s value into the wire-neutral [`agent_core::ReasoningEffort`] enum.
fn parse_reasoning_effort(s: &str) -> Result<agent_core::ReasoningEffort, String> {
    use agent_core::ReasoningEffort::*;
    match s {
        "minimal" => Ok(Minimal),
        "low" => Ok(Low),
        "medium" => Ok(Medium),
        "high" => Ok(High),
        "xhigh" => Ok(XHigh),
        other => Err(format!(
            "invalid reasoning effort {other:?}; expected one of minimal/low/medium/high/xhigh"
        )),
    }
}

#[derive(Parser)]
#[command(name = "beyond-ai-agent", version, about = "Beyond agent harness")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
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
        #[arg(long, env = "AI_AGENT_MODEL")]
        model: Option<String>,
        /// Gateway base URL (default `http://ai.internal`, or `AI_GATEWAY_URL`).
        #[arg(long, env = "AI_GATEWAY_URL")]
        gateway_url: Option<String>,
        /// Virtual key (`bai_v1…`) or BYO provider key. Required; or set `AI_AGENT_KEY`.
        #[arg(long, env = "AI_AGENT_KEY")]
        key: Option<String>,
        /// Max loop iterations before bailing.
        #[arg(long, default_value_t = agent_core::agent::DEFAULT_MAX_STEPS)]
        max_steps: u32,
        /// Trust `cwd` for this run only, so a project-local `.claude/SYSTEM.md` is honored even if
        /// `cwd` isn't in the persisted allowlist (`agent trust <path>`). A session-scoped override,
        /// not a permanent grant — see `agent trust` to record one.
        #[arg(long, default_value_t = false)]
        trust_project: bool,
        /// Force `cwd` *untrusted* for this run only, overriding both `--trust-project` and the
        /// persisted allowlist (`agent trust <path>`) — e.g. to test untrusted behavior against a
        /// directory that's otherwise permanently trusted. Wins over `--trust-project` if both are
        /// somehow given.
        #[arg(long, default_value_t = false)]
        force_untrusted: bool,
        /// Restrict the tool set to exactly these names (comma-separated), dropping everything else.
        /// Combine with `--exclude-tools` to carve one back out of the allow-list.
        #[arg(long, value_delimiter = ',')]
        tools: Option<Vec<String>>,
        /// Drop these tools (comma-separated) from the default set — e.g. `--exclude-tools bash,write`
        /// for a read-only reviewer that can't run shell commands or mutate files.
        #[arg(long, value_delimiter = ',')]
        exclude_tools: Option<Vec<String>>,
        /// Register no tools at all — a pure-conversation run. Wins over `--tools`/`--exclude-tools`.
        #[arg(long, default_value_t = false)]
        no_tools: bool,
        /// Disable skills discovery/loading — no `<available_skills>` listing in the system prompt, and
        /// a `/skill:name` invocation in the task message is sent through unexpanded. Matches pi's own
        /// `--no-skills`. A one-shot `run` has no `reload` to re-enable it mid-process, unlike `serve`.
        #[arg(long, default_value_t = false)]
        no_skills: bool,
        /// Disable prompt-template discovery/loading — a `/name` invocation in the task message is sent
        /// through unexpanded instead of being resolved against `.claude/prompts/*.md`. Matches pi's own
        /// `--no-prompt-templates`.
        #[arg(long, default_value_t = false)]
        no_prompt_templates: bool,
        /// Persist this run to a specific session file, creating it if missing or continuing it if it
        /// already exists — so a later `run --session <path>` picks up where this one left off. Wins
        /// over `--continue` if both are given.
        #[arg(long)]
        session: Option<String>,
        /// Use this exact session id instead of a freshly generated one — a caller (a script, a test
        /// harness) that wants a known, predictable id to correlate against rather than parsing it back
        /// out of the run's own output. Applies whenever a *new* `SessionMeta` is minted: a fresh
        /// `--session <path>` (one that doesn't already exist) or a plain ephemeral run with neither
        /// `--session` nor `--continue`; ignored when reopening an existing `--session <path>` or
        /// resuming via `--continue` (the id is already fixed by whatever's on disk). Matches pi's own
        /// `--session-id` flag.
        #[arg(long)]
        session_id: Option<String>,
        /// Continue the most recent session for the current directory (the same
        /// `~/.claude/sessions/<encoded-cwd>/` repo `serve` defaults to), creating one if this is the
        /// first run here. Ignored if `--session` is also given.
        #[arg(long, short = 'c', default_value_t = false)]
        r#continue: bool,
        /// After the run completes, export the transcript as a self-contained HTML file at this path
        /// (parent directories are created as needed) — the same rendering `serve`'s `export_html` RPC
        /// command produces, for a one-shot run with no server involved.
        #[arg(long)]
        export: Option<String>,
        /// Emit newline-delimited JSON to stdout instead of human-readable text: one leading session
        /// header line, then one `AgentEvent` object per line (tool calls/results and turn boundaries
        /// included, not just raw text deltas) — the same event shape `serve`'s NDJSON protocol streams,
        /// for a scripting caller that wants structured output without spawning `serve`.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Run the headless agent server: a newline-delimited JSON control protocol over stdio.
    Serve {
        /// Model id (default `claude-opus-4-8`, or `AI_AGENT_MODEL`).
        #[arg(long, env = "AI_AGENT_MODEL")]
        model: Option<String>,
        /// Gateway base URL (default `http://ai.internal`, or `AI_GATEWAY_URL`).
        #[arg(long, env = "AI_GATEWAY_URL")]
        gateway_url: Option<String>,
        /// Virtual key (`bai_v1…`) or BYO provider key. Required; or set `AI_AGENT_KEY`.
        #[arg(long, env = "AI_AGENT_KEY")]
        key: Option<String>,
        /// Persist one session to this JSONL file so a later `serve` reattaches with the transcript.
        #[arg(long, env = "AI_AGENT_SESSION_FILE")]
        session_file: Option<String>,
        /// Persist many sessions under this directory (enables list/switch/fork/name commands).
        #[arg(long, env = "AI_AGENT_SESSION_DIR")]
        session_dir: Option<String>,
        /// Skip persistence entirely, even without `--session-file`/`--session-dir`. Without this,
        /// `serve` defaults to `~/.claude/sessions/<encoded-cwd>/` rather than silently running
        /// in-memory-only — pass this for the rare case that's genuinely what you want (e.g. a
        /// short-lived test harness).
        #[arg(long, default_value_t = false)]
        no_session_persistence: bool,
        /// Max loop iterations per prompt before bailing.
        #[arg(long, default_value_t = agent_core::agent::DEFAULT_MAX_STEPS)]
        max_steps: u32,
        /// Replace the built-in base system prompt entirely.
        #[arg(long, env = "AI_AGENT_SYSTEM_PROMPT")]
        system_prompt: Option<String>,
        /// Append extra instructions after the base system prompt.
        #[arg(long, env = "AI_AGENT_APPEND_SYSTEM_PROMPT")]
        append_system_prompt: Option<String>,
        /// Do not discover/inject AGENTS.md / CLAUDE.md project-instruction files.
        #[arg(long, default_value_t = false)]
        no_context_files: bool,
        /// Model context window (tokens); the loop summarizes older turns to stay below it. Defaults
        /// to the model's own capability-table window (see `agent_core::models::capabilities`) — only
        /// pass this to pin a fixed budget that survives a `set_model` switch to a different model.
        #[arg(long, env = "AI_AGENT_CONTEXT_WINDOW")]
        context_window: Option<u32>,
        /// Use the 1-hour prompt-cache TTL (vs 5 minutes); helps when turns are spaced out.
        #[arg(long, default_value_t = false)]
        cache_long: bool,
        /// Enable extended thinking with this token budget (must be below the per-turn max tokens).
        #[arg(long)]
        thinking: Option<u32>,
        /// Reasoning effort for models driven by an effort level rather than a token budget (OpenAI
        /// reasoning models via `reasoning_effort`; Anthropic adaptive-thinking models via
        /// `output_config.effort`). One of minimal/low/medium/high/xhigh. Ignored by models that take
        /// neither shape.
        #[arg(long, value_parser = parse_reasoning_effort)]
        reasoning_effort: Option<agent_core::ReasoningEffort>,
        /// Trust `cwd` for this run only, so a project-local `.claude/SYSTEM.md` is honored even if
        /// `cwd` isn't in the persisted allowlist (`agent trust <path>`). A session-scoped override,
        /// not a permanent grant — see `agent trust` to record one.
        #[arg(long, default_value_t = false)]
        trust_project: bool,
        /// Force `cwd` *untrusted* for this session only, overriding both `--trust-project` and the
        /// persisted allowlist (`agent trust <path>`) — e.g. to test untrusted behavior against a
        /// directory that's otherwise permanently trusted. Wins over `--trust-project` if both are
        /// somehow given.
        #[arg(long, default_value_t = false)]
        force_untrusted: bool,
        /// Compaction headroom (tokens) reserved below the context window before it fires. Defaults to
        /// `CompactionConfig::default()`'s 24,000.
        #[arg(long, env = "AI_AGENT_COMPACTION_RESERVE_TOKENS")]
        compaction_reserve_tokens: Option<u32>,
        /// Roughly how many tokens of recent conversation compaction keeps verbatim. Defaults to
        /// `CompactionConfig::default()`'s 40,000.
        #[arg(long, env = "AI_AGENT_COMPACTION_KEEP_RECENT_TOKENS")]
        compaction_keep_recent_tokens: Option<u32>,
        /// How many times to retry a gateway request that fails before the first response byte
        /// arrives. Defaults to 3.
        #[arg(long, env = "AI_AGENT_RETRY_MAX_RETRIES")]
        retry_max_retries: Option<u32>,
        /// Base of the exponential backoff between those retries, in milliseconds. Defaults to 250.
        #[arg(long, env = "AI_AGENT_RETRY_BASE_DELAY_MS")]
        retry_base_delay_ms: Option<u64>,
        /// Default `bash` command timeout (ms) when the model omits `timeout_ms`. Defaults to 1,800,000
        /// (30 minutes) — see `tools::bash`'s doc comment for why this deliberately deviates from the
        /// reference agent's no-default.
        #[arg(long, env = "AI_AGENT_BASH_TIMEOUT_MS")]
        bash_timeout_ms: Option<u64>,
        /// Run `bash` commands through this shell instead of the auto-resolved one (`/bin/bash`, else
        /// `bash` on `$PATH`, else `sh`) — for a non-standard environment (Cygwin, a container without
        /// `/bin/bash` at the expected path, a hardened/audited shell wrapper) where auto-detection
        /// would pick the wrong binary. Matches pi's own `shellPath` setting. Checked to exist once
        /// here, at startup — a bad path fails the process immediately instead of surfacing as a
        /// confusing spawn error on the first `bash` call.
        #[arg(long, env = "AI_AGENT_BASH_SHELL_PATH")]
        bash_shell_path: Option<String>,
        /// Restrict the tool set to exactly these names (comma-separated), dropping everything else.
        /// Fixed for the process, like `--system-prompt`; survives `set_model`/`set_thinking` rebuilds.
        #[arg(long, env = "AI_AGENT_TOOLS", value_delimiter = ',')]
        tools: Option<Vec<String>>,
        /// Drop these tools (comma-separated) from the default set — e.g. `--exclude-tools bash,write`
        /// for a read-only reviewer that can't run shell commands or mutate files.
        #[arg(long, env = "AI_AGENT_EXCLUDE_TOOLS", value_delimiter = ',')]
        exclude_tools: Option<Vec<String>>,
        /// Register no tools at all — a pure-conversation session. Wins over `--tools`/`--exclude-tools`.
        #[arg(long, default_value_t = false)]
        no_tools: bool,
        /// Restrict `cycle_model`'s candidate list to exactly these ids, in this order
        /// (comma-separated) — e.g. `--models claude-opus-4-8,claude-sonnet-4-5,gpt-5`.
        /// `set_model`/`get_available_models` are unaffected; empty/absent cycles the full known-model
        /// list instead.
        #[arg(long, env = "AI_AGENT_MODELS", value_delimiter = ',')]
        models: Option<Vec<String>>,
    },
    /// List the tools the agent advertises to the model.
    Tools,
    /// List a small, non-exhaustive set of model ids the capabilities table recognizes (a convenience
    /// hint for a model picker — the gateway forwards any id verbatim, so `--model`/`set_model` accept
    /// ids outside this list too).
    ListModels,
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

    match Cli::parse().command {
        Command::Run {
            tasks,
            model,
            gateway_url,
            key,
            max_steps,
            trust_project,
            force_untrusted,
            tools,
            exclude_tools,
            no_tools,
            no_skills,
            no_prompt_templates,
            session,
            session_id,
            r#continue,
            export,
            json,
        } => {
            run_task(
                tasks,
                model,
                gateway_url,
                key,
                max_steps,
                trust_project,
                force_untrusted,
                tools,
                exclude_tools,
                no_tools,
                no_skills,
                no_prompt_templates,
                session,
                session_id,
                r#continue,
                export,
                json,
            )
            .await?;
        }
        Command::Serve {
            model,
            gateway_url,
            key,
            session_file,
            session_dir,
            no_session_persistence,
            max_steps,
            system_prompt,
            append_system_prompt,
            no_context_files,
            context_window,
            cache_long,
            thinking,
            reasoning_effort,
            trust_project,
            force_untrusted,
            compaction_reserve_tokens,
            compaction_keep_recent_tokens,
            retry_max_retries,
            retry_base_delay_ms,
            bash_timeout_ms,
            bash_shell_path,
            tools,
            exclude_tools,
            no_tools,
            models,
        } => {
            let key = key
                .ok_or("no gateway key: pass --key or set AI_AGENT_KEY (a bai_v1… virtual key)")?;
            if let Some(path) = &bash_shell_path {
                if !std::path::Path::new(path).exists() {
                    return Err(format!("--bash-shell-path not found: {path}").into());
                }
            }
            let system = system_prompt.unwrap_or_else(|| {
                // Shell-path override doesn't affect this registry's use (listing tool
                // names/descriptions for the default system prompt) — `describe()` doesn't mention it.
                let mut reg = tools::default_registry_with(bash_timeout_ms, None);
                tools::apply_filter(
                    &mut reg,
                    tools.as_deref(),
                    exclude_tools.as_deref(),
                    no_tools,
                );
                default_system_prompt(&reg)
            });
            serve::serve(serve::ServeConfig {
                gateway: gateway_url.unwrap_or_else(|| DEFAULT_GATEWAY.to_string()),
                key,
                model: model.unwrap_or_else(|| DEFAULT_MODEL.to_string()),
                max_steps,
                system,
                append_system: append_system_prompt,
                context_files: !no_context_files,
                session_file,
                session_dir,
                no_session_persistence,
                context_window,
                cache_long,
                thinking,
                reasoning_effort,
                trust_project,
                force_untrusted,
                compaction_reserve_tokens,
                compaction_keep_recent_tokens,
                retry_max_retries,
                retry_base_delay_ms: retry_base_delay_ms.map(std::time::Duration::from_millis),
                bash_timeout_ms,
                bash_shell_path,
                tools,
                exclude_tools,
                no_tools,
                models: models.unwrap_or_default(),
            })
            .await?;
            // `serve` reads stdin via `tokio::io::stdin()`, which parks a dedicated blocking OS
            // thread doing a blocking read for the life of the process. If stdin is never closed
            // (a client that doesn't hang up, or — the case this matters for — a SIGTERM/SIGINT
            // whose handler cancels the run and returns without stdin ever reaching EOF), that
            // thread is still parked here even though all async work is done. Falling through to
            // `#[tokio::main]`'s implicit runtime shutdown would then hang indefinitely: dropping
            // a `Runtime` waits for every outstanding blocking task, and a parked stdin read never
            // completes on its own. Exit explicitly instead — `serve` has already drained,
            // persisted, and flushed everything before returning, so there's nothing left to lose.
            std::process::exit(0);
        }
        Command::Tools => {
            let reg = tools::default_registry();
            println!("{} tools:\n", reg.len());
            println!("{}", serde_json::to_string_pretty(&reg.definitions())?);
        }
        Command::ListModels => {
            for model in serve::available_models() {
                println!("{model}");
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
        Command::Export { session, output } => {
            let (store, sess) =
                beyond_ai_agent::session_store::SessionStore::open(PathBuf::from(&session))
                    .map_err(|e| format!("failed to open session {session}: {e}"))?;
            let branches = store.abandoned_branches();
            let path = beyond_ai_agent::export::export_html(
                store.meta(),
                &sess.messages,
                &branches,
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

/// Read each of `file_refs` (resolved against `cwd`; an already-absolute ref is used as-is) and wrap
/// its contents in a `<file name="...">` block, concatenated in argument order. Errors naming the
/// first unreadable file, so a typo'd `@path` fails loudly instead of silently vanishing from the
/// prompt.
fn read_file_refs(file_refs: &[String], cwd: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let mut out = String::new();
    for r in file_refs {
        let path = cwd.join(r);
        let content = std::fs::read_to_string(&path)
            .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
        out.push_str(&format!(
            "<file name=\"{}\">\n{content}\n</file>\n",
            path.display()
        ));
    }
    Ok(out)
}

/// The full contents of stdin, if it's piped (not an interactive terminal) and non-empty. `None`
/// otherwise — including on a read error, since a broken pipe just means there was nothing to add.
fn read_stdin_if_piped() -> Option<String> {
    let stdin = std::io::stdin();
    if stdin.is_terminal() {
        return None;
    }
    let mut buf = String::new();
    match stdin.lock().read_to_string(&mut buf) {
        Ok(_) if !buf.is_empty() => Some(buf),
        _ => None,
    }
}

/// [`run_turn_once`], wrapped with the same whole-run auto-retry `serve.rs`'s `"prompt"` command gets
/// (see `beyond_ai_agent::retry`) — a run that ends in a transient-looking error (one already
/// exhausted `agent_core`'s own within-turn retries) is re-invoked from scratch against the same
/// session, up to `retry::MAX_RUN_RETRIES` times with backoff, rather than failing a whole `agent run`
/// invocation (plausibly unattended — a cron job, a CI step) outright on a hiccup that `serve` would
/// have quietly recovered from. A retried attempt's own streamed output (text/JSON events) follows
/// directly after a `[retrying...]` stderr notice — nothing is erased, matching how `serve` demarcates
/// attempts with an `auto_retry` frame rather than hiding the failed one.
async fn run_turn(
    agent: &Agent,
    session: &mut Session,
    json: bool,
) -> agent_core::Result<agent_core::StopReason> {
    let mut attempt = 0u32;
    loop {
        let result = run_turn_once(agent, session, json).await;
        match &result {
            Err(e)
                if attempt < beyond_ai_agent::retry::MAX_RUN_RETRIES
                    && beyond_ai_agent::retry::is_retryable_whole_run(e) =>
            {
                attempt += 1;
                let delay = beyond_ai_agent::retry::backoff(attempt);
                eprintln!(
                    "\n[transient error, retrying {attempt}/{}: {e}]",
                    beyond_ai_agent::retry::MAX_RUN_RETRIES
                );
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
async fn run_turn_once(
    agent: &Agent,
    session: &mut Session,
    json: bool,
) -> agent_core::Result<agent_core::StopReason> {
    let mut stop_reason = agent_core::StopReason::default();
    if json {
        agent
            .run_events(session, |ev| {
                if let agent_core::AgentEvent::TurnEnd { stop_reason: r, .. } = &ev {
                    stop_reason = *r;
                }
                if let Ok(line) = serde_json::to_string(&ev) {
                    println!("{line}");
                    let _ = std::io::stdout().flush();
                }
            })
            .await?;
        return Ok(stop_reason);
    }
    agent
        .run(session, |ev| match ev {
            StreamEvent::TextDelta { text } => {
                print!("{text}");
                let _ = std::io::stdout().flush();
            }
            StreamEvent::ToolUseStart { name, .. } => {
                // No trailing newline: `InputJsonDelta` fragments print immediately after, live, on
                // this same line — a growing preview of the call's arguments as they stream in,
                // rather than the model appearing to hang until the whole call (and its result) land.
                print!("\n[tool: {name}] ");
                let _ = std::io::stdout().flush();
            }
            StreamEvent::InputJsonDelta { partial_json } => {
                print!("{partial_json}");
                let _ = std::io::stdout().flush();
            }
            StreamEvent::MessageStop { stop_reason: r } => {
                stop_reason = *r;
            }
            _ => {}
        })
        .await?;
    println!();
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
        if let Some(store) = guard.as_mut() {
            if let Err(e) = store.append_new(&session.messages) {
                eprintln!("run: failed to persist checkpoint: {e}");
            }
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

#[allow(clippy::too_many_arguments)]
async fn run_task(
    tasks: Vec<String>,
    model: Option<String>,
    gateway_url: Option<String>,
    key: Option<String>,
    max_steps: u32,
    trust_project: bool,
    force_untrusted: bool,
    tools_allow: Option<Vec<String>>,
    tools_exclude: Option<Vec<String>>,
    no_tools: bool,
    no_skills: bool,
    no_prompt_templates: bool,
    session_path: Option<String>,
    session_id: Option<String>,
    continue_session: bool,
    export: Option<String>,
    json: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut timing = beyond_ai_agent::timing::StartupTiming::new();
    let cwd = canonical_cwd(&std::env::current_dir().unwrap_or_default());

    // Compose the first message from (in order) piped stdin, `@file` contents, then the first
    // plain-text message argument — mirroring the reference agent's own composition order. At least
    // one source must contribute something; a typo'd invocation with none of the three fails loudly
    // here rather than sending the model an empty prompt.
    let (file_refs, mut messages) = partition_tasks(tasks);
    let stdin_content = read_stdin_if_piped();
    let file_content = read_file_refs(&file_refs, &cwd)?;
    let mut parts = Vec::new();
    if let Some(s) = stdin_content {
        parts.push(s);
    }
    if !file_content.is_empty() {
        parts.push(file_content);
    }
    if !messages.is_empty() {
        parts.push(messages.remove(0));
    }
    if parts.is_empty() {
        return Err("no task given: pass a message, an @file, or pipe input via stdin".into());
    }
    let initial_message = parts.join("");
    timing.mark("compose initial message");

    let gateway = gateway_url.unwrap_or_else(|| DEFAULT_GATEWAY.to_string());
    let model = model.unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let key =
        key.ok_or("no gateway key: pass --key or set AI_AGENT_KEY (a bai_v1… virtual key)")?;

    let project_trusted = !force_untrusted
        && (trust_project
            || beyond_ai_agent::trust_store::TrustStore::open_default().is_trusted(&cwd));
    // Discovered once, up front: a one-shot `run` has no `reload` to re-discover mid-process, unlike
    // `serve`. `/skill:name` and `/name` prompt-template invocations are expanded here exactly like
    // `serve`'s own "prompt" handler does — this was previously silently skipped in `run`, so a message
    // starting with either was sent to the model as a literal, unexpanded string instead.
    // `--no-skills`/`--no-prompt-templates` skip discovery outright rather than discovering and then
    // discarding — matching pi's own flags, and avoiding a needless filesystem walk when the operator
    // has already said neither is wanted.
    let skills = if no_skills {
        Vec::new()
    } else {
        beyond_ai_agent::skills::discover(&cwd, project_trusted)
    };
    let prompt_templates = if no_prompt_templates {
        Vec::new()
    } else {
        beyond_ai_agent::prompts::discover(&cwd, project_trusted)
    };
    timing.mark("discover skills/prompt templates");
    let mut registry = tools::default_registry();
    tools::apply_filter(
        &mut registry,
        tools_allow.as_deref(),
        tools_exclude.as_deref(),
        no_tools,
    );
    let base = default_system_prompt(&registry);
    let system = beyond_ai_agent::resources::build_system_prompt(
        &beyond_ai_agent::resources::PromptOptions {
            base: &base,
            append: None,
            cwd: &cwd,
            include_context_files: true,
            include_skills: !no_skills,
            project_trusted,
        },
    );
    timing.mark("build system prompt");

    // `--session`/`--continue` persist this run (and load prior history to continue it) exactly like
    // `serve`'s own repo/file modes; neither given keeps `run` in-memory-only, as before.
    let cwd_str = cwd.to_string_lossy().into_owned();
    // `--session-id`, when given, applies only where a *new* `SessionMeta` is actually minted below —
    // reopening an existing `--session <path>` or resuming via `--continue` already has a fixed id from
    // disk. Matches pi's own `--session-id`: a known, predictable id for a script/test harness to
    // correlate against, instead of parsing it back out of the run's own output.
    let fresh_meta = || match &session_id {
        Some(id) => SessionMeta::with_id(id.clone(), &cwd_str, &model),
        None => SessionMeta::new(&cwd_str, &model),
    };
    let (store, mut session) = match session_path {
        Some(path) => {
            let path = PathBuf::from(path);
            // A zero-byte file at `path` (e.g. `touch`'d ahead of time, or left over from a crash
            // before the header write landed) has nothing to open — route it through `create`, which
            // now initializes an empty file in place rather than failing (see its own doc comment).
            let has_content = path.metadata().is_ok_and(|m| m.len() > 0);
            if has_content {
                let (store, session) = SessionStore::open(path)?;
                (Some(store), session)
            } else {
                let store = SessionStore::create(path, fresh_meta())?;
                (Some(store), Session::new())
            }
        }
        None if continue_session => {
            let repo = SessionRepo::open(default_session_dir(&cwd_str))?;
            let (store, session) = repo.resume_or_create(&cwd_str, &model)?;
            (Some(store), session)
        }
        None => (None, Session::new()),
    };
    let meta = store
        .as_ref()
        .map(|s| s.meta().clone())
        .unwrap_or_else(fresh_meta);
    timing.mark("open session");
    timing.print();

    let client = GatewayClient::new(gateway, key)?;
    // Shared with `DirectCheckpoint` below (built before `agent`, so the hook can be installed at
    // construction) so a long multi-step turn (many tool round-trips) is persisted incrementally —
    // the same guarantee `serve` gives every session. Without this, only the *end* of each whole
    // turn was ever persisted (the `persist_run_tail` calls below, after `run_turn` returns), so a
    // crash mid-turn — after several tool round-trips already ran real commands/edited real files —
    // lost all record of them with no session trace at all.
    let store = Arc::new(std::sync::Mutex::new(store));
    let agent = Agent::new(Arc::new(client), model.clone())
        .with_tools(registry)
        .with_system(system)
        .with_max_steps(max_steps)
        .with_checkpoint_hook(Arc::new(DirectCheckpoint(store.clone())));

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

    session.user(expand_message(&initial_message, &skills, &prompt_templates));
    let mut stop_reason = run_turn(&agent, &mut session, json).await?;
    persist_run_tail(&store, &session)?;
    for message in messages {
        session.user(expand_message(&message, &skills, &prompt_templates));
        stop_reason = run_turn(&agent, &mut session, json).await?;
        persist_run_tail(&store, &session)?;
    }

    if let Some(export) = export {
        let branches = store
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map(|s| s.abandoned_branches())
            .unwrap_or_default();
        match beyond_ai_agent::export::export_html(
            &meta,
            &session.messages,
            &branches,
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
    // Text mode has no other failure signal a script/CI caller could key off of — a refusal would
    // otherwise still exit 0, indistinguishable from a normal completion, unless the last turn's
    // stop reason is checked explicitly here. JSON mode already carries `stop_reason` on every
    // `TurnEnd` event in its own output stream, so it's unaffected either way — that exit code stays
    // reserved for a genuine process failure. Matches pi's own print-mode, which treats a refusal
    // (folded into its generic "error" stop reason there, unlike this crate's distinct `Refusal`
    // variant) the same way, in text mode only.
    if !json && stop_reason == agent_core::StopReason::Refusal {
        eprintln!("[refused]");
        std::process::exit(1);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_core::Error;
    use agent_core::mock::{MockTransport, turn};

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
        run_turn(&agent, &mut session, false).await.unwrap();

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

        run_turn(&agent, &mut session, false)
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

        let err = run_turn(&agent, &mut session, false)
            .await
            .expect_err("must eventually give up, not retry forever");
        assert!(matches!(err, Error::Transport(_)));
        assert_eq!(transport.calls(), total_attempts);
    }

    #[test]
    fn default_system_prompt_lists_every_registered_tool() {
        // The whole point of generating this dynamically: it can't silently omit a tool the way the
        // prior hardcoded string did (it never mentioned the Beyond platform tools at all).
        let registry = tools::default_registry();
        let prompt = default_system_prompt(&registry);
        for def in tools::default_registry().definitions() {
            assert!(
                prompt.contains(&def.name),
                "system prompt is missing registered tool {:?}: {prompt}",
                def.name
            );
        }
    }

    #[test]
    fn default_system_prompt_reflects_a_restricted_registry() {
        // A tool-restricted agent's own system prompt must not claim tools it doesn't actually have —
        // otherwise the model is invited to call one that's guaranteed to be rejected.
        let mut registry = tools::default_registry();
        tools::apply_filter(&mut registry, None, Some(&["bash".to_string()]), false);
        let prompt = default_system_prompt(&registry);
        assert!(!prompt.contains("bash"));
        assert!(prompt.contains("read"));
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

    #[test]
    fn read_file_refs_wraps_contents_in_a_file_tag_with_the_resolved_path() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "hello world").unwrap();
        let out = read_file_refs(&["a.txt".to_string()], dir.path()).unwrap();
        assert!(out.contains("hello world"));
        assert!(out.contains(&format!("name=\"{}\"", dir.path().join("a.txt").display())));
    }

    #[test]
    fn read_file_refs_errors_naming_the_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let err = read_file_refs(&["does-not-exist.txt".to_string()], dir.path())
            .unwrap_err()
            .to_string();
        assert!(err.contains("does-not-exist.txt"), "got: {err}");
    }

    #[test]
    fn read_file_refs_concatenates_multiple_files_in_order() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "AAA").unwrap();
        std::fs::write(dir.path().join("b.txt"), "BBB").unwrap();
        let out = read_file_refs(&["a.txt".to_string(), "b.txt".to_string()], dir.path()).unwrap();
        assert!(out.find("AAA").unwrap() < out.find("BBB").unwrap());
    }
}
