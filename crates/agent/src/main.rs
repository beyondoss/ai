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

use std::fs;
use std::io::{IsTerminal, Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use agent_core::{Agent, GatewayClient, Session, StreamEvent, Tool};
use beyond_ai_agent::policy::ToolPolicy;
use beyond_ai_agent::session_store::{
    SessionMeta, SessionRepo, SessionStore, canonical_cwd, default_session_dir, fork_by_arg,
    is_path_like, open_session_by_id, sessions_root,
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
/// `extra_guidelines` are operator-supplied bullets (`--prompt-guideline`, repeatable) appended after
/// the built-in ones — pi's own `promptGuidelines` (deduplicated and trimmed, matching pi's
/// `buildSystemPrompt`). Deliberately *not* a full port of pi's system prompt: pi also renders a
/// redundant per-tool text snippet list ("Available tools:\n- bash: Execute bash commands...")
/// alongside the native tool-call JSON schema already describing each tool to the model — the same
/// information twice, in two different places the model reads. This function's own dynamic tool-name
/// listing (`Use them to accomplish...with tools: {names}`) already avoids that duplication, so only
/// the genuinely useful, non-redundant half of pi's feature is ported here: the guideline-bullet
/// mechanism itself, including its one built-in conditional (`bash` registered but none of its usual
/// companions).
fn default_system_prompt(
    registry: &agent_core::ToolRegistry,
    extra_guidelines: &[String],
) -> String {
    let names: Vec<String> = registry.definitions().into_iter().map(|d| d.name).collect();
    let has = |n: &str| names.iter().any(|x| x == n);

    let mut guidelines: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let add =
        |g: String, guidelines: &mut Vec<String>, seen: &mut std::collections::HashSet<String>| {
            if seen.insert(g.clone()) {
                guidelines.push(g);
            }
        };
    // Matches pi's own conditional exactly: when `bash` is the only exploration tool registered (none
    // of `grep`/`find`/`ls`), the model needs to be told it's the fallback for those operations too.
    if has("bash") && !has("grep") && !has("find") && !has("ls") {
        add(
            "Use bash for file operations like ls, rg, find".to_string(),
            &mut guidelines,
            &mut seen,
        );
    }
    // pi's own per-tool `promptGuidelines` (`read.ts`/`edit.ts`/`write.ts`) — declared on the tool
    // definition itself and collected from whatever's actually registered. Adapted, not ported
    // verbatim: pi's edit tool takes an `edits[].oldText`/`newText` array, ours takes `edits[].old_string`/
    // `new_string` (see `tools/edit.rs`'s own schema) — porting pi's exact field names would tell the
    // model to look for parameters that don't exist on our tool. `bash`/`grep`/`find`/`ls` carry no
    // `promptGuidelines` on pi's side, so there's nothing to port for those.
    if has("read") {
        add(
            "Use read to examine files instead of cat or sed.".to_string(),
            &mut guidelines,
            &mut seen,
        );
    }
    if has("edit") {
        for g in [
            "Use edit for precise changes (edits[].old_string must match exactly)",
            "When changing multiple separate locations in one file, use one edit call with multiple \
             entries in edits[] instead of multiple edit calls",
            "Each edits[].old_string is matched against the original file, not after earlier edits are \
             applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.",
            "Keep edits[].old_string as small as possible while still being unique in the file. Do not \
             pad with large unchanged regions.",
        ] {
            add(g.to_string(), &mut guidelines, &mut seen);
        }
    }
    if has("write") {
        add(
            "Use write only for new files or complete rewrites.".to_string(),
            &mut guidelines,
            &mut seen,
        );
    }
    for g in extra_guidelines {
        let g = g.trim();
        if !g.is_empty() {
            add(g.to_string(), &mut guidelines, &mut seen);
        }
    }
    add(
        "Show file paths clearly when working with files".to_string(),
        &mut guidelines,
        &mut seen,
    );
    let guidelines = guidelines
        .into_iter()
        .map(|g| format!("- {g}"))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "You are the Beyond coding agent. You operate inside a real working directory with tools: {}. \
         Use them to accomplish the user's task directly — inspect before you change, make minimal \
         edits, and verify your work. Be concise.\n\nGuidelines:\n{guidelines}",
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

/// The one further fallback tier below `--key`/`AI_AGENT_KEY`: an inferred, stored OAuth
/// subscription login for whichever provider `model` implies. Consulted only when no explicit
/// key/env var was given — an explicit `--key`/`AI_AGENT_KEY` always wins outright, unchanged.
///
/// Inference is deliberately non-uniform across providers — a bare model id is unambiguous for some,
/// not others:
/// - Anthropic: any Claude-dialect model id ([`agent_core::dialect::Dialect::for_model`]'s own rule)
///   — there's only one possible account for a Claude request, safe to infer unconditionally.
/// - OpenAI Codex: a model id containing `"codex"` — this crate has no ChatGPT-Codex model catalog of
///   its own to consult, so a name heuristic is as precise as it gets today.
/// - GitHub Copilot: *not* inferred by model-id prefix at all — Copilot's model set is dynamic,
///   discovered at login time and recorded in the stored credential's own `available_model_ids`, so
///   it's matched only if `model` actually appears there.
///
/// If both a direct Anthropic credential and a Copilot credential (whose `available_model_ids`
/// includes `model`) are stored, the direct credential wins — checked first below, preferring the
/// more specific, directly-authenticated relationship over a proxy.
/// Adapts a plain OAuth [`agent_core::client::CredentialSource`] (bearer token + `is_oauth`, from
/// `OAuthCredentialSource`) with routing info the source itself has no way to compute — GitHub
/// Copilot's account-specific proxy host, or OpenAI Codex's distinct backend path/headers — neither
/// of which the gateway's static `KNOWN_PROVIDERS` table can express as a row the way it does for
/// Anthropic/OpenAI direct (see `agent_core::client::RouteOverride`'s doc comment). `routing` is
/// computed once here, from whichever credential is on disk *now* — the same granularity
/// `GatewayClient::base_url` itself already resolves at construction, not fresh per turn — rather
/// than re-derived from the live, possibly-just-refreshed token on every request; the two pieces of
/// data this stands in for (Copilot's `proxy-ep` host, Codex's account id) are properties of the
/// *account*, not of any one token, so they don't change across an in-process refresh of the same
/// login.
struct DirectRoutedCredentialSource {
    inner: Arc<dyn agent_core::client::CredentialSource>,
    routing: agent_core::client::DirectRouting,
}

#[async_trait::async_trait]
impl agent_core::client::CredentialSource for DirectRoutedCredentialSource {
    async fn credential(&self) -> agent_core::Result<agent_core::client::Credential> {
        let credential = self.inner.credential().await?;
        Ok(credential.with_direct_routing(self.routing.clone()))
    }
}

/// A [`agent_core::client::CredentialSource`] for a `models.json` `base_url` override (Fix 9 —
/// pi-parity feature: pi's own `model-registry.ts` custom-model/override support). Unlike
/// [`DirectRoutedCredentialSource`] above (which wraps an existing OAuth source and only adds routing),
/// this *is* the credential: a fixed bearer token — the override's own `api_key`, else whatever
/// `--key`/`AI_AGENT_KEY` resolved to, else empty (many self-hosted OpenAI-compatible servers, like
/// Ollama/LM Studio, ignore the `Authorization` header entirely) — plus the same [`agent_core::client::
/// DirectRouting`] mechanism reused, not duplicated, to send the request straight to the override's
/// `base_url`, bypassing the gateway outright.
struct StaticDirectCredentialSource {
    bearer: String,
    routing: agent_core::client::DirectRouting,
}

#[async_trait::async_trait]
impl agent_core::client::CredentialSource for StaticDirectCredentialSource {
    async fn credential(&self) -> agent_core::Result<agent_core::client::Credential> {
        Ok(
            agent_core::client::Credential::new(self.bearer.clone(), false)
                .with_direct_routing(self.routing.clone()),
        )
    }
}

/// OpenAI's own approved identity string for this tool's OAuth grant (`build_authorize_url` in
/// `oauth/openai_codex.rs` already sends this as the `originator` query param at login time) — reused
/// verbatim here so a live Codex inference request presents the same identity the account authorized,
/// rather than a second, inconsistent one.
const CODEX_ORIGINATOR: &str = "beyond-ai-agent";

/// GitHub Copilot's fixed editor-identity headers, required on every live inference request (not just
/// the OAuth/model-management calls) — see `GITHUB_COPILOT_MODELS`' own per-model `headers` field in
/// pi-mono (`packages/ai/src/providers/github-copilot.models.ts`). Duplicated from the private
/// constants of the same values in `oauth::github_copilot` (`COPILOT_USER_AGENT` et al.) rather than
/// exported from there: those are login/refresh-flow internals, and this crate's own convention
/// (`auth_store.rs`'s `FileLock`, `Secret`) is to duplicate a small, self-contained constant rather
/// than widen another module's private surface for one call site.
const COPILOT_STATIC_HEADERS: [(&str, &str); 4] = [
    ("User-Agent", "GitHubCopilotChat/0.35.0"),
    ("Editor-Version", "vscode/1.107.0"),
    ("Editor-Plugin-Version", "copilot-chat/0.35.0"),
    ("Copilot-Integration-Id", "vscode-chat"),
];

/// GitHub Copilot's real endpoint path per dialect — **not** the dialect's own default
/// `endpoint_path()`: Copilot's OpenAI-wire endpoints omit the `/v1` prefix pi's official SDKs bake
/// into their default `baseURL` (so the SDK's own `/chat/completions`/`/responses` relative paths land
/// bare on Copilot's host), while its Anthropic-wire endpoint matches the dialect default verbatim
/// (the Anthropic SDK's default `baseURL` carries no version segment, so it *always* appends
/// `/v1/messages` itself). See `packages/ai/src/api/{anthropic-messages,openai-completions,
/// openai-responses}.ts` in pi-mono (each sets `baseURL: model.baseUrl` on the vendor SDK, which then
/// appends its own fixed relative path).
fn copilot_endpoint_path(dialect: agent_core::dialect::Dialect) -> &'static str {
    match dialect {
        agent_core::dialect::Dialect::Anthropic => "/v1/messages",
        agent_core::dialect::Dialect::OpenAi => "/chat/completions",
        agent_core::dialect::Dialect::OpenAiResponses => "/responses",
    }
}

fn resolve_gateway_credential(
    key: Option<String>,
    model: &str,
) -> Result<serve::GatewayCredential, String> {
    // Fix 9 (pi-parity feature): a `models.json` override naming a `base_url` for this exact model id
    // wins outright, regardless of whether `--key`/`AI_AGENT_KEY` was also given — the override
    // redirects *where* the request goes (a locally-hosted or alternate-provider endpoint, entirely
    // bypassing the gateway), which is orthogonal to *how* it authenticates. Reuses the same
    // `DirectRouting`/`RouteOverride::Direct` mechanism the GitHub-Copilot OAuth routing below already
    // relies on, rather than duplicating it — see `settings::ModelOverride`'s own doc comment for the
    // on-disk schema.
    if let Some(over) = beyond_ai_agent::settings::ModelOverrides::open_default().get(model) {
        if let Some(base_url) = over.base_url.clone() {
            let dialect = agent_core::dialect::Dialect::for_model(model);
            let bearer = over
                .api_key
                .clone()
                .or_else(|| key.clone())
                .unwrap_or_default();
            let routing = agent_core::client::DirectRouting {
                route: agent_core::client::RouteOverride::Direct {
                    base_url,
                    path: dialect.endpoint_path(),
                },
                static_headers: Vec::new(),
                copilot_dynamic_headers: false,
            };
            return Ok(serve::GatewayCredential::Oauth(Arc::new(
                StaticDirectCredentialSource { bearer, routing },
            )));
        }
    }

    if let Some(key) = key {
        return Ok(serve::GatewayCredential::Static(key));
    }

    let store = beyond_ai_agent::auth_store::AuthStore::open_default();
    let oauth_source = |provider: beyond_ai_agent::oauth::OAuthProviderId| {
        Arc::new(beyond_ai_agent::auth_credential_source::OAuthCredentialSource::new(
            provider,
            beyond_ai_agent::auth_store::default_path(),
        )) as Arc<dyn agent_core::client::CredentialSource>
    };

    if agent_core::dialect::Dialect::for_model(model) == agent_core::dialect::Dialect::Anthropic
        && store.get("anthropic").is_some()
    {
        return Ok(serve::GatewayCredential::Oauth(oauth_source(
            beyond_ai_agent::oauth::OAuthProviderId::Anthropic,
        )));
    }
    if model.contains("codex") {
        if let Some(stored) = store.get("openai-codex") {
            if let beyond_ai_agent::oauth::OAuthCredential::OpenaiCodex(c) = &stored.credential {
                // Still relayed through the gateway (a new `KNOWN_PROVIDERS` row: `chatgpt.com` is a
                // genuinely static host) under the `/openai-codex` prefix, with the account id this
                // backend requires attached as a static header — see `RouteOverride::Prefixed`.
                let routing = agent_core::client::DirectRouting {
                    route: agent_core::client::RouteOverride::Prefixed {
                        prefix: "/openai-codex",
                        path: "/backend-api/codex/responses",
                    },
                    static_headers: vec![
                        ("chatgpt-account-id", c.account_id.clone()),
                        ("originator", CODEX_ORIGINATOR.to_string()),
                        ("OpenAI-Beta", "responses=experimental".to_string()),
                    ],
                    copilot_dynamic_headers: false,
                };
                return Ok(serve::GatewayCredential::Oauth(Arc::new(
                    DirectRoutedCredentialSource {
                        inner: oauth_source(beyond_ai_agent::oauth::OAuthProviderId::OpenaiCodex),
                        routing,
                    },
                )));
            }
        }
    }
    if let Some(stored) = store.get("github-copilot") {
        if let beyond_ai_agent::oauth::OAuthCredential::GithubCopilot(c) = &stored.credential {
            if c.available_model_ids.iter().any(|m| m == model) {
                // Bypasses the gateway entirely: GitHub hands back a *different* proxy host per
                // account/enterprise, embedded in the access token itself (`proxy-ep=…`) — not a
                // static host the gateway's `KNOWN_PROVIDERS` table could ever hold as a row. See
                // `RouteOverride::Direct`.
                let dialect = agent_core::dialect::Dialect::for_model(model);
                let base_url = beyond_ai_agent::oauth::github_copilot::base_url_from_token(
                    Some(c.access.as_str()),
                    c.enterprise_url.as_deref(),
                );
                let routing = agent_core::client::DirectRouting {
                    route: agent_core::client::RouteOverride::Direct {
                        base_url,
                        path: copilot_endpoint_path(dialect),
                    },
                    static_headers: COPILOT_STATIC_HEADERS
                        .iter()
                        .map(|(name, value)| (*name, value.to_string()))
                        .collect(),
                    copilot_dynamic_headers: true,
                };
                return Ok(serve::GatewayCredential::Oauth(Arc::new(
                    DirectRoutedCredentialSource {
                        inner: oauth_source(beyond_ai_agent::oauth::OAuthProviderId::GithubCopilot),
                        routing,
                    },
                )));
            }
        }
    }

    Err(format!(
        "no gateway key: pass --key or set AI_AGENT_KEY (a bai_v1… virtual key), or run `agent login \
         <provider>` to use a subscription for model {model:?}"
    ))
}

fn unknown_provider_error(provider: &str) -> String {
    format!("unknown provider {provider:?}; expected one of: anthropic, github-copilot, openai-codex")
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
        if let Ok(n) = choice.parse::<usize>() {
            if n >= 1 && n <= options.len() {
                return Ok(Some(options[n - 1].to_string()));
            }
        }
        // Also accept typing the id directly.
        Ok(options
            .into_iter()
            .find(|id| *id == choice)
            .map(str::to_string))
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
        /// Per-turn output token ceiling. `serve`'s identical flag; defaults to the model's own
        /// capability-table `max_output` (see `agent_core::models::capabilities`) when omitted.
        #[arg(long, env = "AI_AGENT_MAX_TOKENS")]
        max_tokens: Option<u32>,
        /// Use the 1-hour prompt-cache TTL (vs 5 minutes); helps when turns are spaced out. `serve`'s
        /// identical flag; `run`'s one-shot single-turn case rarely benefits, but a multi-message
        /// invocation (several `tasks` sent as sequential turns) can.
        #[arg(long, default_value_t = false)]
        cache_long: bool,
        /// Enable extended thinking with this token budget (must be below the per-turn max tokens).
        /// `serve`'s identical flag; unlike `serve`, `run` has no thinking-level cycling, so this is
        /// applied as-is with no per-model default derivation when omitted.
        #[arg(long)]
        thinking: Option<u32>,
        /// Reasoning effort for models driven by an effort level rather than a token budget (OpenAI
        /// reasoning models via `reasoning_effort`; Anthropic adaptive-thinking models via
        /// `output_config.effort`). One of minimal/low/medium/high/xhigh. Ignored by models that take
        /// neither shape. Falls back to `AI_AGENT_REASONING_EFFORT`, then the stored
        /// `agent settings --default-reasoning-effort` default (Fix 2 — pi-parity gap: previously the
        /// only numeric/string CLI tunable with no persisted-default fallback at all), before finally
        /// leaving it unset. `serve`'s identical flag.
        #[arg(long, env = "AI_AGENT_REASONING_EFFORT", value_parser = parse_reasoning_effort)]
        reasoning_effort: Option<agent_core::ReasoningEffort>,
        /// Sampling temperature. Omitted (leaving the provider default) unless set. Silently ignored by
        /// Anthropic while `--thinking` is set (Anthropic forbids the two together). `serve`'s identical
        /// flag.
        #[arg(long)]
        temperature: Option<f64>,
        /// Replace the built-in base system prompt entirely. `serve`'s identical flag — e.g. a
        /// specialized reviewer/persona invocation for automation that needs a wholly different agent
        /// identity, not just extra instructions layered on top (see `--append-system-prompt`).
        #[arg(long, env = "AI_AGENT_SYSTEM_PROMPT")]
        system_prompt: Option<String>,
        /// Append extra instructions after the base system prompt (built-in, or `--system-prompt` if
        /// also given). Repeatable — pi-parity fix: previously a second occurrence silently clobbered
        /// the first instead of accumulating (matches pi, whose `appendSystemPrompt` is itself an
        /// array). Each occurrence is joined with the others by a blank line, in the order given.
        /// `serve`'s identical flag.
        #[arg(long, env = "AI_AGENT_APPEND_SYSTEM_PROMPT")]
        append_system_prompt: Vec<String>,
        /// Trust `cwd` for this run only, so a project-local `.claude/SYSTEM.md` is honored even if
        /// `cwd` isn't in the persisted allowlist (`agent trust <path>`). A session-scoped override,
        /// not a permanent grant — see `agent trust` to record one. `-a` matches pi's own
        /// `--approve`/`-a` (same "trust this project" meaning, different flag name here).
        #[arg(short = 'a', long, default_value_t = false)]
        trust_project: bool,
        /// Force `cwd` *untrusted* for this run only, overriding both `--trust-project` and the
        /// persisted allowlist (`agent trust <path>`) — e.g. to test untrusted behavior against a
        /// directory that's otherwise permanently trusted. Wins over `--trust-project` if both are
        /// somehow given. `-na` matches pi's own `--no-approve`/`-na`.
        #[arg(long, default_value_t = false)]
        force_untrusted: bool,
        /// Model context window (tokens); the loop summarizes older turns to stay below it. Defaults
        /// to the model's own capability-table window (see `agent_core::models::capabilities`) — only
        /// pass this to pin a fixed budget regardless of which model ends up used. `serve`'s identical
        /// flag.
        #[arg(long, env = "AI_AGENT_CONTEXT_WINDOW")]
        context_window: Option<u32>,
        /// Compaction headroom (tokens) reserved below the context window before it fires. Defaults to
        /// `CompactionConfig::default()`'s 24,000. `serve`'s identical flag.
        #[arg(long, env = "AI_AGENT_COMPACTION_RESERVE_TOKENS")]
        compaction_reserve_tokens: Option<u32>,
        /// Roughly how many tokens of recent conversation compaction keeps verbatim. Defaults to
        /// `CompactionConfig::default()`'s 40,000. `serve`'s identical flag.
        #[arg(long, env = "AI_AGENT_COMPACTION_KEEP_RECENT_TOKENS")]
        compaction_keep_recent_tokens: Option<u32>,
        /// Disable automatic (threshold-triggered) compaction entirely — the loop only ever compacts on
        /// a genuine overflow (`agent_core::CompactionConfig::enabled`'s own doc comment: manual/overflow
        /// compaction ignores this setting), never proactively. For a caller that would rather fail/see
        /// the raw context-window error than have older turns silently summarized away.
        #[arg(long, env = "AI_AGENT_NO_COMPACTION", default_value_t = false)]
        no_compaction: bool,
        /// How many times to retry a gateway request that fails before the first response byte
        /// arrives. Defaults to 3. `serve`'s identical flag.
        #[arg(long, env = "AI_AGENT_RETRY_MAX_RETRIES")]
        retry_max_retries: Option<u32>,
        /// Base of the exponential backoff between those retries, in milliseconds. Defaults to 250.
        /// `serve`'s identical flag.
        #[arg(long, env = "AI_AGENT_RETRY_BASE_DELAY_MS")]
        retry_base_delay_ms: Option<u64>,
        /// Default `bash` command timeout (ms) when the model omits `timeout_ms`. Defaults to 1,800,000
        /// (30 minutes). `serve`'s identical flag.
        #[arg(long, env = "AI_AGENT_BASH_TIMEOUT_MS")]
        bash_timeout_ms: Option<u64>,
        /// Run `bash` commands through this shell instead of the auto-resolved one (`/bin/bash`, else
        /// `bash` on `$PATH`, else `sh`). Matches pi's own `shellPath` setting. `serve`'s identical flag.
        #[arg(long, env = "AI_AGENT_BASH_SHELL_PATH")]
        bash_shell_path: Option<String>,
        /// Prepend this line to every `bash` command, in the same shell invocation (e.g. sourcing a
        /// project's env setup, activating a venv). Matches pi's own `shellCommandPrefix` setting.
        /// `serve`'s identical flag.
        #[arg(long, env = "AI_AGENT_BASH_COMMAND_PREFIX")]
        bash_command_prefix: Option<String>,
        /// Restrict the tool set to exactly these names (comma-separated), dropping everything else.
        /// Combine with `--exclude-tools` to carve one back out of the allow-list. `serve`'s identical
        /// flag/env var — a deployment convention setting this env var to sandbox an agent must apply
        /// here too, not just to `serve`. `-t` matches pi's own `--tools`/`-t`.
        #[arg(short = 't', long, env = "AI_AGENT_TOOLS", value_delimiter = ',')]
        tools: Option<Vec<String>>,
        /// Drop these tools (comma-separated) from the default set — e.g. `--exclude-tools bash,write`
        /// for a read-only reviewer that can't run shell commands or mutate files. `serve`'s identical
        /// flag/env var. `-xt` matches pi's own `--exclude-tools`/`-xt`.
        #[arg(long, env = "AI_AGENT_EXCLUDE_TOOLS", value_delimiter = ',')]
        exclude_tools: Option<Vec<String>>,
        /// Register no tools at all — a pure-conversation run. Wins over `--tools`/`--exclude-tools`.
        /// `-nt` matches pi's own `--no-tools`/`-nt`.
        #[arg(long, default_value_t = false)]
        no_tools: bool,
        /// Force every batch of tool calls in a turn to run one at a time instead of the default
        /// bounded-concurrent dispatch (`agent_core::Agent::with_sequential_tools`) — e.g. for a
        /// deterministic repro, or a host policy that never wants two tool calls actually overlapping.
        /// `serve`'s identical flag.
        #[arg(long, default_value_t = false)]
        sequential_tools: bool,
        /// Block every call to this tool (comma-separated, repeatable), even though it stays visible
        /// and registered — unlike `--exclude-tools` (the model never sees an excluded tool exists at
        /// all), a denied call still surfaces to the model as a normal error `tool_result` explaining
        /// it was blocked by policy. Installs an `agent_core::AgentHooks` permission gate
        /// (`policy::ToolPolicy`) on the agent; a no-op (no hook installed at all) when combined with
        /// `--deny-bash-pattern` leaves the list empty.
        #[arg(long, env = "AI_AGENT_DENY_TOOL", value_delimiter = ',')]
        deny_tool: Vec<String>,
        /// Block a `bash` call whenever its command contains this substring, case-insensitively
        /// (comma-separated, repeatable) — e.g. `--deny-bash-pattern "rm -rf"`. Combines with
        /// `--deny-tool` under the same policy hook.
        #[arg(long, env = "AI_AGENT_DENY_BASH_PATTERN", value_delimiter = ',')]
        deny_bash_pattern: Vec<String>,
        /// Block a `write`/`edit` call whenever its `path` argument matches this glob (comma-separated,
        /// repeatable) — e.g. `--deny-path '.env,**/secrets/**'`. Same glob engine as `grep`'s
        /// `--glob`/`find`'s pattern matching (`globset::Glob`). Combines with `--deny-tool`/
        /// `--deny-bash-pattern` under the same policy hook.
        #[arg(long, env = "AI_AGENT_DENY_PATH", value_delimiter = ',')]
        deny_path: Vec<String>,
        /// Disable *standard-root* skills discovery/loading (`~/.claude/skills`, `<cwd>/.claude/skills`)
        /// — no `<available_skills>` listing in the system prompt from either, and a `/skill:name`
        /// invocation in the task message is sent through unexpanded unless it resolves against a
        /// `--skill` path instead. An explicit `--skill <path>` is still honored even so — pi's own
        /// `--no-skills` does the same (a documented, tested combination: it's a way to say "nothing
        /// auto-discovered, only what I explicitly listed", not "no skills at all"). A one-shot `run`
        /// has no `reload` to re-enable it mid-process, unlike `serve`. `-ns` matches pi's own
        /// `--no-skills`/`-ns`.
        #[arg(long, default_value_t = false)]
        no_skills: bool,
        /// Disable *standard-root* prompt-template discovery/loading (`~/.claude/prompts`,
        /// `<cwd>/.claude/prompts`) — a `/name` invocation in the task message is sent through
        /// unexpanded unless it resolves against a `--prompt-template` path instead. An explicit
        /// `--prompt-template <path>` is still honored even so, matching `--no-skills`'s identical
        /// carve-out and pi's own `--no-prompt-templates`. `-np` matches pi's own
        /// `--no-prompt-templates`/`-np`.
        #[arg(long, default_value_t = false)]
        no_prompt_templates: bool,
        /// Do not discover/inject AGENTS.md / CLAUDE.md project-instruction files. Matches `serve`'s
        /// identical flag — `run` previously hardcoded this on with no way to opt out. `-nc` matches
        /// pi's own `--no-context-files`/`-nc`.
        #[arg(long, default_value_t = false)]
        no_context_files: bool,
        /// Discover skills from this directory too, in addition to the two standard roots (repeatable,
        /// or comma-separated via `AI_AGENT_SKILL_PATH` — matching `--tools`/`AI_AGENT_TOOLS`'s own
        /// comma-separated env-var convention). Matches pi's own `--skill <path>`. A path that doesn't
        /// exist is warned about, not silently ignored. Wins over the standard roots on a name collision.
        #[arg(
            long = "skill",
            env = "AI_AGENT_SKILL_PATH",
            value_delimiter = ',',
            value_name = "PATH"
        )]
        extra_skill_paths: Vec<String>,
        /// Discover prompt templates from this directory too, in addition to the two standard roots
        /// (repeatable, or comma-separated via `AI_AGENT_PROMPT_TEMPLATE_PATH`). Matches pi's own
        /// `--prompt-template <path>`; see `--skill`'s doc comment for the missing-path/shadow-order
        /// behavior, which applies identically here.
        #[arg(
            long = "prompt-template",
            env = "AI_AGENT_PROMPT_TEMPLATE_PATH",
            value_delimiter = ',',
            value_name = "PATH"
        )]
        extra_prompt_template_paths: Vec<String>,
        /// Set this run's session name up front, before the first turn even starts — a whitespace-only
        /// value is rejected rather than silently producing a blank/meaningless name, matching pi's own
        /// `--name` validation. Unlike pi (renames unconditionally on every invocation), only takes
        /// effect on a genuinely fresh session — see the fresh-only check in `serve`, a deliberate
        /// deviation. The RPC `set_session_name` command covers renaming an already-running session.
        #[arg(short = 'n', long)]
        name: Option<String>,
        /// An extra guideline bullet appended to the default system prompt's `Guidelines:` section
        /// (repeatable) — pi's own `promptGuidelines`. Deduplicated and trimmed against the built-in
        /// guidelines.
        #[arg(long = "prompt-guideline", value_name = "TEXT")]
        prompt_guidelines: Vec<String>,
        /// Fork an existing session into a brand-new one and continue from there, rather than reopening
        /// it in place — by id (searched in this project first, then every other project's own session
        /// directory under `~/.claude/sessions/`) or by a direct path to its `.jsonl` file (any
        /// project). Matches pi's own cross-project `--fork <path|id>`; the forked copy's `cwd` is the
        /// *current* directory, not wherever the source session was originally recorded against. Wins
        /// over `--session`/`--continue` if more than one is given — a fork always starts a fresh child,
        /// never reopens one in place. Forks the whole active transcript; `serve`'s `fork`/`fork_at_entry`
        /// RPC commands cover forking at an earlier point once a session is running.
        #[arg(long, value_name = "PATH_OR_ID")]
        fork: Option<String>,
        /// Persist this run to a specific session, creating it if missing or continuing it if it already
        /// exists — so a later `run --session <path|id>` picks up where this one left off. Accepts
        /// either a direct path to a `.jsonl` file (created fresh if it doesn't exist yet) or a bare
        /// session id/unique prefix, resolved against the current project's own repo first, then every
        /// other project's under `--session-dir`'s root — matching pi's own `--session <path|id>`. Wins
        /// over `--continue` if both are given.
        #[arg(long)]
        session: Option<String>,
        /// Use this exact session id instead of a freshly generated one — a caller (a script, a test
        /// harness) that wants a known, predictable id to correlate against rather than parsing it back
        /// out of the run's own output. Applies whenever a *new* `SessionMeta` is minted: a fresh
        /// `--session <path>` (one that doesn't already exist) or a plain run with neither `--session`
        /// nor `--continue` given (still persisted by default — see `--no-session-persistence`); ignored
        /// when reopening an existing `--session <path>` or resuming via `--continue` (the id is already
        /// fixed by whatever's on disk). Matches pi's own `--session-id` flag.
        #[arg(long)]
        session_id: Option<String>,
        /// Continue the most recent session for the current directory (the same
        /// `~/.claude/sessions/<encoded-cwd>/` repo `serve` defaults to), creating one if this is the
        /// first run here. Ignored if `--session` is also given. Kept as an explicit, self-documenting
        /// spelling of what a plain no-flag `run` now does by default too (pi-parity fix — see
        /// `--no-session-persistence`); harmless to pass either way.
        #[arg(long, short = 'c', default_value_t = false)]
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
        #[arg(long, env = "AI_AGENT_SESSION_DIR")]
        session_dir: Option<String>,
        /// Skip persistence entirely, even without `--session`/`--continue`/`--fork`. Without this, a
        /// plain no-flag `run` now defaults to the same per-cwd repo `serve` does
        /// (`~/.claude/sessions/<encoded-cwd>/`, or `--session-dir`) rather than running in-memory-only —
        /// pass this for the rare case that's genuinely what you want (e.g. a short-lived script that
        /// mustn't leave a session file behind). Matches `serve`'s identical flag, so the CLI vocabulary
        /// for opting out is the same either way.
        #[arg(long, default_value_t = false)]
        no_session_persistence: bool,
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
        /// Use this exact session id instead of a freshly generated one — a caller (a script, a test
        /// harness) that wants a known, predictable id to correlate against rather than parsing it back
        /// out of `get_state`/the startup `{"kind":"session", id, …}` banner. Applies only when a *new*
        /// `SessionMeta` is actually minted: a brand-new `--session-file` (one that doesn't already
        /// exist), `--no-session-persistence`, or the default/`--session-dir` repo mode when no existing
        /// session matches this `cwd` yet; ignored when reattaching to an existing one (already has a
        /// fixed id from disk) — matches `run`'s identical flag/contract exactly (`main.rs::Run::
        /// session_id`).
        #[arg(long)]
        session_id: Option<String>,
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
        /// Append extra instructions after the base system prompt. Repeatable — `run`'s identical flag;
        /// each occurrence is joined with the others by a blank line, in the order given.
        #[arg(long, env = "AI_AGENT_APPEND_SYSTEM_PROMPT")]
        append_system_prompt: Vec<String>,
        /// Do not discover/inject AGENTS.md / CLAUDE.md project-instruction files. `-nc` matches pi's
        /// own `--no-context-files`/`-nc`.
        #[arg(long, default_value_t = false)]
        no_context_files: bool,
        /// Model context window (tokens); the loop summarizes older turns to stay below it. Defaults
        /// to the model's own capability-table window (see `agent_core::models::capabilities`) — only
        /// pass this to pin a fixed budget that survives a `set_model` switch to a different model.
        #[arg(long, env = "AI_AGENT_CONTEXT_WINDOW")]
        context_window: Option<u32>,
        /// Per-turn output token ceiling. Defaults to the model's own capability-table `max_output`
        /// (see `agent_core::models::capabilities`), floored at a sane minimum — only pass this to
        /// override that, e.g. capping generation length or lifting it past the model-derived default.
        #[arg(long, env = "AI_AGENT_MAX_TOKENS")]
        max_tokens: Option<u32>,
        /// Use the 1-hour prompt-cache TTL (vs 5 minutes); helps when turns are spaced out.
        #[arg(long, default_value_t = false)]
        cache_long: bool,
        /// Enable extended thinking with this token budget (must be below the per-turn max tokens).
        #[arg(long)]
        thinking: Option<u32>,
        /// Reasoning effort for models driven by an effort level rather than a token budget (OpenAI
        /// reasoning models via `reasoning_effort`; Anthropic adaptive-thinking models via
        /// `output_config.effort`). One of minimal/low/medium/high/xhigh. Ignored by models that take
        /// neither shape. Falls back to `AI_AGENT_REASONING_EFFORT`, then the stored
        /// `agent settings --default-reasoning-effort` default, before finally leaving it unset. `run`'s
        /// identical flag.
        #[arg(long, env = "AI_AGENT_REASONING_EFFORT", value_parser = parse_reasoning_effort)]
        reasoning_effort: Option<agent_core::ReasoningEffort>,
        /// Sampling temperature. Omitted (leaving the provider default) unless set. Silently ignored by
        /// Anthropic while `--thinking` is set (Anthropic forbids the two together). `run`'s identical
        /// flag.
        #[arg(long)]
        temperature: Option<f64>,
        /// Trust `cwd` for this run only, so a project-local `.claude/SYSTEM.md` is honored even if
        /// `cwd` isn't in the persisted allowlist (`agent trust <path>`). A session-scoped override,
        /// not a permanent grant — see `agent trust` to record one. `-a` matches pi's own
        /// `--approve`/`-a` (same "trust this project" meaning, different flag name here).
        #[arg(short = 'a', long, default_value_t = false)]
        trust_project: bool,
        /// Force `cwd` *untrusted* for this session only, overriding both `--trust-project` and the
        /// persisted allowlist (`agent trust <path>`) — e.g. to test untrusted behavior against a
        /// directory that's otherwise permanently trusted. Wins over `--trust-project` if both are
        /// somehow given. `-na` matches pi's own `--no-approve`/`-na`.
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
        /// Disable automatic (threshold-triggered) compaction entirely — `run`'s identical flag. When
        /// absent (and `AI_AGENT_NO_COMPACTION` unset), falls back to the persisted `agent settings`
        /// `compaction_enabled` override before finally defaulting to enabled — see
        /// `serve::ServeConfig::no_compaction`'s doc comment.
        #[arg(long, env = "AI_AGENT_NO_COMPACTION", default_value_t = false)]
        no_compaction: bool,
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
        /// Prepend this line to every `bash` command, in the same shell invocation (e.g. sourcing a
        /// project's env setup, activating a venv). Matches pi's own `shellCommandPrefix` setting.
        /// Fixed for the process, like `--bash-shell-path`; survives `set_model`/`set_thinking` rebuilds.
        #[arg(long, env = "AI_AGENT_BASH_COMMAND_PREFIX")]
        bash_command_prefix: Option<String>,
        /// Restrict the tool set to exactly these names (comma-separated), dropping everything else.
        /// Fixed for the process, like `--system-prompt`; survives `set_model`/`set_thinking` rebuilds.
        /// `-t` matches pi's own `--tools`/`-t`.
        #[arg(short = 't', long, env = "AI_AGENT_TOOLS", value_delimiter = ',')]
        tools: Option<Vec<String>>,
        /// Drop these tools (comma-separated) from the default set — e.g. `--exclude-tools bash,write`
        /// for a read-only reviewer that can't run shell commands or mutate files. `-xt` matches pi's
        /// own `--exclude-tools`/`-xt`.
        #[arg(long, env = "AI_AGENT_EXCLUDE_TOOLS", value_delimiter = ',')]
        exclude_tools: Option<Vec<String>>,
        /// Register no tools at all — a pure-conversation session. Wins over `--tools`/`--exclude-tools`.
        /// `-nt` matches pi's own `--no-tools`/`-nt`.
        #[arg(long, default_value_t = false)]
        no_tools: bool,
        /// Force every batch of tool calls in a turn to run one at a time instead of the default
        /// bounded-concurrent dispatch (`agent_core::Agent::with_sequential_tools`). `run`'s identical
        /// flag.
        #[arg(long, default_value_t = false)]
        sequential_tools: bool,
        /// Block every call to this tool (comma-separated, repeatable), even though it stays visible
        /// and registered — unlike `--exclude-tools`, a denied call still surfaces to the model as a
        /// normal error `tool_result` explaining it was blocked by policy, rather than the tool being
        /// invisible outright. `run`'s identical flag.
        #[arg(long, env = "AI_AGENT_DENY_TOOL", value_delimiter = ',')]
        deny_tool: Vec<String>,
        /// Block a `bash` call whenever its command contains this substring, case-insensitively
        /// (comma-separated, repeatable). `run`'s identical flag.
        #[arg(long, env = "AI_AGENT_DENY_BASH_PATTERN", value_delimiter = ',')]
        deny_bash_pattern: Vec<String>,
        /// Block a `write`/`edit` call whenever its `path` argument matches this glob (comma-separated,
        /// repeatable). `run`'s identical flag.
        #[arg(long, env = "AI_AGENT_DENY_PATH", value_delimiter = ',')]
        deny_path: Vec<String>,
        /// Restrict `cycle_model`'s candidate list to exactly these ids, in this order
        /// (comma-separated) — e.g. `--models claude-opus-4-8,claude-sonnet-4-5,gpt-5`.
        /// `set_model`/`get_available_models` are unaffected; empty/absent cycles the full known-model
        /// list instead.
        #[arg(long, env = "AI_AGENT_MODELS", value_delimiter = ',')]
        models: Option<Vec<String>>,
        /// Disable *standard-root* skills discovery/loading (`~/.claude/skills`, `<cwd>/.claude/skills`)
        /// — no `<available_skills>` listing in the system prompt from either, and a `/skill:name`
        /// invocation (however it reaches the session — `prompt`, `steer`, `follow_up`) is sent through
        /// unexpanded unless it resolves against a `--skill` path instead. An explicit `--skill <path>`
        /// is still honored even so, matching `run`'s identical flag and pi's own `--no-skills`. Applies
        /// on every `reload` too. `-ns` matches pi's own `--no-skills`/`-ns`.
        #[arg(long, default_value_t = false)]
        no_skills: bool,
        /// Disable *standard-root* prompt-template discovery/loading (`~/.claude/prompts`,
        /// `<cwd>/.claude/prompts`) — a `/name` invocation is sent through unexpanded unless it resolves
        /// against a `--prompt-template` path instead. An explicit `--prompt-template <path>` is still
        /// honored even so, matching `run`'s identical flag and pi's own `--no-prompt-templates`. Applies
        /// on every `reload` too. `-np` matches pi's own `--no-prompt-templates`/`-np`.
        #[arg(long, default_value_t = false)]
        no_prompt_templates: bool,
        /// Discover skills from this directory too, in addition to the two standard roots (repeatable,
        /// or comma-separated via `AI_AGENT_SKILL_PATH`). Matches pi's own `--skill <path>` and `run`'s
        /// identical flag; applies on every `reload` too.
        #[arg(
            long = "skill",
            env = "AI_AGENT_SKILL_PATH",
            value_delimiter = ',',
            value_name = "PATH"
        )]
        extra_skill_paths: Vec<String>,
        /// Discover prompt templates from this directory too, in addition to the two standard roots
        /// (repeatable, or comma-separated via `AI_AGENT_PROMPT_TEMPLATE_PATH`). Matches pi's own
        /// `--prompt-template <path>` and `run`'s identical flag; applies on every `reload` too.
        #[arg(
            long = "prompt-template",
            env = "AI_AGENT_PROMPT_TEMPLATE_PATH",
            value_delimiter = ',',
            value_name = "PATH"
        )]
        extra_prompt_template_paths: Vec<String>,
        /// Set the initial session's name up front, before the first turn even starts — a whitespace-only
        /// value is rejected, matching pi's own `--name`. Unlike pi (which renames unconditionally on
        /// every invocation, last-write-wins), this only ever takes effect on a genuinely fresh session
        /// — see the fresh-only check in `run_task`/`serve` for why: a deliberate deviation, not an
        /// oversight. The RPC `set_session_name` command covers renaming an existing session afterward.
        #[arg(short = 'n', long)]
        name: Option<String>,
        /// An extra guideline bullet appended to the default system prompt's `Guidelines:` section
        /// (repeatable) — pi's own `promptGuidelines`; `run`'s identical flag. Has no effect when
        /// `--system-prompt` supplies a full custom prompt (matches pi: a custom prompt replaces the
        /// whole guidelines mechanism, not just extends it).
        #[arg(long = "prompt-guideline", value_name = "TEXT")]
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
    /// View or update persisted defaults for `run`/`serve` flags — model, gateway URL, session
    /// directory — stored at `~/.claude/settings.json` (see `settings::SettingsStore`) and consulted as
    /// the last fallback after an explicit `--flag`/environment variable, before this crate's own
    /// built-in default. With no flags, prints the currently stored values. Mirrors `agent trust`/
    /// `agent untrust` managing the trust store the same out-of-band way.
    Settings {
        /// Set the stored default model (used when neither `--model` nor `AI_AGENT_MODEL` is given).
        #[arg(long)]
        model: Option<String>,
        /// Clear the stored default model.
        #[arg(long, default_value_t = false)]
        clear_model: bool,
        /// Set the stored default gateway URL (used when neither `--gateway-url` nor `AI_GATEWAY_URL`
        /// is given).
        #[arg(long)]
        gateway_url: Option<String>,
        /// Clear the stored default gateway URL.
        #[arg(long, default_value_t = false)]
        clear_gateway_url: bool,
        /// Set the stored default session directory (used when neither `--session-dir` nor
        /// `AI_AGENT_SESSION_DIR` is given).
        #[arg(long)]
        session_dir: Option<String>,
        /// Clear the stored default session directory.
        #[arg(long, default_value_t = false)]
        clear_session_dir: bool,
        /// Set the stored default project-trust policy — `always`/`never`/`ask` (used when neither
        /// `--trust-project` nor `--force-untrusted` is given; see `settings::TrustPolicy`).
        #[arg(long)]
        default_project_trust: Option<beyond_ai_agent::settings::TrustPolicy>,
        /// Clear the stored default project-trust policy.
        #[arg(long, default_value_t = false)]
        clear_default_project_trust: bool,
        /// Set the stored default reasoning effort — one of minimal/low/medium/high/xhigh (used when
        /// neither `--reasoning-effort` nor `AI_AGENT_REASONING_EFFORT` is given). Fix 2 (pi-parity
        /// gap): previously the only numeric/string CLI tunable with no persisted-default surface at
        /// all, unlike `--model`/`--gateway-url`/`--session-dir` above.
        #[arg(long, value_parser = parse_reasoning_effort)]
        default_reasoning_effort: Option<agent_core::ReasoningEffort>,
        /// Clear the stored default reasoning effort.
        #[arg(long, default_value_t = false)]
        clear_default_reasoning_effort: bool,
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
    if let Some(split) = chars.iter().position(|c| !c.is_ascii_lowercase()) {
        if split > 0 && chars[split..].iter().all(char::is_ascii_digit) {
            let (letters, digits) = (&chars[..split], &chars[split..]);
            return Some(digits.iter().chain(letters).collect());
        }
    }
    if let Some(split) = chars.iter().position(|c| !c.is_ascii_digit()) {
        if split > 0 && chars[split..].iter().all(char::is_ascii_lowercase) {
            let (digits, letters) = (&chars[..split], &chars[split..]);
            return Some(letters.iter().chain(digits).collect());
        }
    }
    None
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
    match Cli::try_parse_from(&args) {
        Ok(cli) => cli,
        Err(e) => {
            let run_json_help = args.get(1).map(String::as_str) == Some("run")
                && args.iter().any(|a| a == "--json")
                && args.iter().any(|a| a == "--help" || a == "-h");
            if run_json_help
                && matches!(
                    e.kind(),
                    clap::error::ErrorKind::DisplayHelp
                        | clap::error::ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand
                )
            {
                eprint!("{}", e.render());
                std::process::exit(0);
            }
            // Every other case (a real usage error, `--version`, bare `--help`) keeps clap's own
            // default stream/exit-code behavior — see the doc comment above for why only the
            // `run --json --help` combination needs overriding.
            e.exit();
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
            temperature,
            system_prompt,
            append_system_prompt,
            trust_project,
            force_untrusted,
            context_window,
            compaction_reserve_tokens,
            compaction_keep_recent_tokens,
            no_compaction,
            retry_max_retries,
            retry_base_delay_ms,
            bash_timeout_ms,
            bash_shell_path,
            bash_command_prefix,
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
            export,
            json,
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
                temperature,
                system_prompt,
                append_system_prompt,
                trust_project,
                force_untrusted,
                context_window,
                compaction_reserve_tokens,
                compaction_keep_recent_tokens,
                no_compaction,
                retry_max_retries,
                retry_base_delay_ms,
                bash_timeout_ms,
                bash_shell_path,
                bash_command_prefix,
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
            session_id,
            no_session_persistence,
            max_steps,
            max_tokens,
            system_prompt,
            append_system_prompt,
            no_context_files,
            context_window,
            cache_long,
            thinking,
            reasoning_effort,
            temperature,
            trust_project,
            force_untrusted,
            compaction_reserve_tokens,
            compaction_keep_recent_tokens,
            no_compaction,
            retry_max_retries,
            retry_base_delay_ms,
            bash_timeout_ms,
            bash_shell_path,
            bash_command_prefix,
            tools,
            exclude_tools,
            no_tools,
            sequential_tools,
            deny_tool,
            deny_bash_pattern,
            deny_path,
            models,
            no_skills,
            no_prompt_templates,
            extra_skill_paths,
            extra_prompt_template_paths,
            name,
            prompt_guidelines,
        } => {
            if let Some(path) = &bash_shell_path {
                if !std::path::Path::new(path).exists() {
                    return Err(format!("--bash-shell-path not found: {path}").into());
                }
            }
            // Fail fast, before starting the server — see `run_task`'s identical check for why this
            // rejects rather than silently clearing (pi's own `--name` behavior).
            if let Some(n) = &name {
                if n.trim().is_empty() {
                    return Err("--name requires a non-empty value".into());
                }
            }
            // Same filesystem-path-injection concern as `run`'s identical check (`--session-id` becomes
            // part of a persisted session's filename) — see `is_valid_session_id`'s doc comment.
            if let Some(id) = &session_id {
                if !is_valid_session_id(id) {
                    return Err(format!(
                        "--session-id {id:?} is invalid: must contain only letters, digits, '.', '_', \
                         '-', and start/end with a letter or digit — it becomes part of a filesystem path"
                    )
                    .into());
                }
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
                    default_system_prompt(&reg, &prompt_guidelines)
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
            let stored_settings = beyond_ai_agent::settings::SettingsStore::open_default();
            let resolved_model = model
                .or_else(|| stored_settings.get().default_model.clone())
                .unwrap_or_else(|| DEFAULT_MODEL.to_string());
            // Fix 10 (pi-parity feature): `run`'s identical resolution — see that call site's doc
            // comment for why this must happen before `resolve_gateway_credential` below.
            let resolved_model = serve::resolve_model_id(&resolved_model, serve::available_models())
                .map_err(|e| format!("--model {resolved_model:?}: {e}"))?;
            // The one further fallback tier below `--key`/`AI_AGENT_KEY`: an inferred, stored OAuth
            // subscription login for whichever provider `resolved_model` implies. See
            // `resolve_gateway_credential`'s own doc comment.
            let key = resolve_gateway_credential(key, &resolved_model)?;
            // Fix 2 (pi-parity gap): `run`'s identical stored-default fallback for `--reasoning-effort`
            // — see that call site's doc comment.
            let reasoning_effort = reasoning_effort.or_else(|| {
                stored_settings
                    .get()
                    .default_reasoning_effort
                    .as_deref()
                    .and_then(|s| parse_reasoning_effort(s).ok())
            });
            let resolved_session_dir = session_dir.or_else(|| {
                // Only synthesize a stored default when *neither* explicit flag was given —
                // `Persistence::open` checks `session_dir` before `session_file`, so filling in a
                // stored session-dir default even when the operator explicitly chose `--session-file`
                // would silently switch them into repo mode instead of the file mode they asked for.
                if session_file.is_none() {
                    stored_settings.get().default_session_dir.clone()
                } else {
                    None
                }
            });
            let shutdown_cause = serve::serve(serve::ServeConfig {
                gateway: gateway_url
                    .or_else(|| stored_settings.get().default_gateway_url.clone())
                    .unwrap_or_else(|| DEFAULT_GATEWAY.to_string()),
                key,
                model: resolved_model,
                max_steps,
                max_tokens,
                system,
                append_system: append_system_prompt,
                context_files: !no_context_files,
                session_file,
                session_dir: resolved_session_dir,
                session_id,
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
                no_compaction,
                retry_max_retries,
                retry_base_delay_ms: retry_base_delay_ms.map(std::time::Duration::from_millis),
                bash_timeout_ms,
                bash_shell_path,
                bash_command_prefix,
                tools,
                exclude_tools,
                no_tools,
                sequential_tools,
                deny_tool,
                deny_bash_pattern,
                deny_path,
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
                default_project_trust: stored_settings.get().default_project_trust,
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
            // Task #41 (pi-parity fix): `shutdown_cause` distinguishes a real signal-triggered
            // shutdown from a clean stdin-EOF one — previously every graceful path exited 0
            // unconditionally, matching neither pi's own `rpc-mode.ts` (143/129 for SIGTERM/SIGHUP)
            // nor a shell's own convention for reporting which signal actually stopped a process.
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
                    caps.context_window,
                    caps.max_output,
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
                || clear_default_reasoning_effort;
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
            // that may not match what actually ran. `usage: None` for the same reason `sess`'s own
            // token counters are never persisted/restored across a process restart (only
            // `last_input_tokens` is, for compaction — see `SessionStore::open`) — a bare zero would
            // misrepresent unknown as "no usage at all".
            let path = beyond_ai_agent::export::export_html_full(
                store.meta(),
                &sess.messages,
                &branches,
                None,
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

/// Read each of `file_refs` (resolved against `cwd`; an already-absolute ref is used as-is). A plain
/// (non-image) file's contents are wrapped in a `<file name="...">` block, concatenated in argument
/// order — unchanged from before this fix. A file whose leading bytes identify it as an image (see
/// [`looks_like_image`]) is instead run through the `read` tool's own image pipeline (sniffing,
/// decode/validate, downscale-to-budget, format conversion) so it can be attached as a real
/// [`agent_core::ImageSource`] rather than handed to `std::fs::read_to_string`, which errors outright
/// on binary image bytes — the crash this fix closes (`run @screenshot.png "..."` previously failed
/// instead of attaching the screenshot). Errors naming the first unreadable (or undecodable) file, so
/// a typo'd `@path` — or a genuinely corrupt image — fails loudly instead of silently vanishing from
/// the prompt.
async fn read_file_refs(
    file_refs: &[String],
    cwd: &Path,
) -> Result<FileRefs, Box<dyn std::error::Error>> {
    let mut text = String::new();
    let mut images = Vec::new();
    for r in file_refs {
        let path = cwd.join(r);
        if looks_like_image(&path) {
            let path_str = path.to_string_lossy().into_owned();
            let out = tools::read::Read
                .run(serde_json::json!({ "path": path_str }))
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
    if path.is_file() {
        if let Ok(contents) = fs::read_to_string(path) {
            return contents;
        }
    }
    raw.to_string()
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
            std::process::exit(code);
        }
        Err(e) => Err(e.into()),
    }
}

async fn run_turn(
    agent: &Agent,
    session: &mut Session,
    json: bool,
    cancel: &agent_core::CancellationToken,
    retry_policy: &beyond_ai_agent::retry::RunRetryPolicy,
) -> agent_core::Result<agent_core::StopReason> {
    let mut attempt = 0u32;
    loop {
        let result = run_turn_once(agent, session, json, cancel).await;
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
async fn run_turn_once(
    agent: &Agent,
    session: &mut Session,
    json: bool,
    cancel: &agent_core::CancellationToken,
) -> agent_core::Result<agent_core::StopReason> {
    let mut stop_reason = agent_core::StopReason::default();
    if json {
        agent
            .run_events_cancellable(
                session,
                |ev| {
                    if let agent_core::AgentEvent::TurnEnd { stop_reason: r, .. } = &ev {
                        stop_reason = *r;
                    }
                    if let Ok(line) = serde_json::to_string(&ev) {
                        println!("{line}");
                        let _ = std::io::stdout().flush();
                    }
                },
                cancel.clone(),
            )
            .await?;
        return Ok(stop_reason);
    }
    agent
        .run_cancellable(
            session,
            |ev| match ev {
                StreamEvent::TextDelta { text, .. } => {
                    print!("{text}");
                    let _ = std::io::stdout().flush();
                }
                StreamEvent::ToolUseStart { name, .. } => {
                    // No trailing newline: `InputJsonDelta` fragments print immediately after, live,
                    // on this same line — a growing preview of the call's arguments as they stream in,
                    // rather than the model appearing to hang until the whole call (and its result)
                    // land.
                    print!("\n[tool: {name}] ");
                    let _ = std::io::stdout().flush();
                }
                StreamEvent::InputJsonDelta { partial_json, .. } => {
                    print!("{partial_json}");
                    let _ = std::io::stdout().flush();
                }
                StreamEvent::MessageStop { stop_reason: r } => {
                    stop_reason = *r;
                }
                _ => {}
            },
            cancel.clone(),
        )
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

/// Whether `id` is safe to embed directly in a filename component — alphanumeric, optionally with
/// `.`/`_`/`-` in the middle, starting and ending with a letter or digit. Matches pi's
/// `assertValidSessionId` (`^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$`); rejects anything that could
/// resolve to a path outside the sessions directory (a leading `/` or `..`, an embedded `/`, etc.) or
/// be empty.
fn is_valid_session_id(id: &str) -> bool {
    let bytes = id.as_bytes();
    let is_alnum = |b: u8| b.is_ascii_alphanumeric();
    match bytes {
        [] => false,
        [only] => is_alnum(*only),
        [first, .., last] => {
            is_alnum(*first)
                && is_alnum(*last)
                && bytes
                    .iter()
                    .all(|&b| is_alnum(b) || b == b'.' || b == b'_' || b == b'-')
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_task(
    tasks: Vec<String>,
    model: Option<String>,
    gateway_url: Option<String>,
    key: Option<String>,
    max_steps: u32,
    max_tokens: Option<u32>,
    cache_long: bool,
    thinking: Option<u32>,
    reasoning_effort: Option<agent_core::ReasoningEffort>,
    temperature: Option<f64>,
    system_prompt: Option<String>,
    append_system_prompt: Vec<String>,
    trust_project: bool,
    force_untrusted: bool,
    context_window: Option<u32>,
    compaction_reserve_tokens: Option<u32>,
    compaction_keep_recent_tokens: Option<u32>,
    no_compaction: bool,
    retry_max_retries: Option<u32>,
    retry_base_delay_ms: Option<u64>,
    bash_timeout_ms: Option<u64>,
    bash_shell_path: Option<String>,
    bash_command_prefix: Option<String>,
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
    export: Option<String>,
    json: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Fail fast, before touching any files — matches pi's own `--name` validation. Whitespace-only is
    // rejected outright here (a startup argument the operator clearly meant to be meaningful), unlike
    // the RPC `set_session_name` command's "empty clears the title" convention (renaming an
    // already-running session to nothing is a deliberate, different action).
    if let Some(n) = &name {
        if n.trim().is_empty() {
            return Err("--name requires a non-empty value".into());
        }
    }
    // `--session-id` is embedded directly into a filename (`SessionMeta::with_id` →
    // `SessionRepo::path_for`'s `{created_at}_{id}.jsonl`) with no other sanitization — an id like
    // `../../../tmp/pwned/evil` would write (and `mkdir -p`, since `SessionStore::create` does that
    // too) outside the intended sessions directory. Matches pi's own `assertValidSessionId`.
    if let Some(id) = &session_id {
        if !is_valid_session_id(id) {
            return Err(format!(
                "--session-id {id:?} is invalid: must contain only letters, digits, '.', '_', '-', \
                 and start/end with a letter or digit — it becomes part of a filesystem path"
            )
            .into());
        }
    }
    let mut timing = beyond_ai_agent::timing::StartupTiming::new();
    let cwd = canonical_cwd(&std::env::current_dir().unwrap_or_default());

    // Compose the first message from (in order) piped stdin, `@file` contents, then the first
    // plain-text message argument — mirroring the reference agent's own composition order. At least
    // one source must contribute something; a typo'd invocation with none of the three fails loudly
    // here rather than sending the model an empty prompt. An `@file` reference that's actually an image
    // (see `read_file_refs`) contributes no text of its own but still counts as "something", so an
    // invocation like `run @screenshot.png` with no other text still proceeds.
    let (file_refs, mut messages) = partition_tasks(tasks);
    let stdin_content = read_stdin_if_piped();
    let file_refs = read_file_refs(&file_refs, &cwd).await?;
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

    // A stored `agent settings` default sits between an explicit flag/env var and this crate's own
    // built-in default — checked here, once, rather than threading `SettingsStore` through every
    // individual flag's own resolution site.
    let stored_settings = beyond_ai_agent::settings::SettingsStore::open_default();
    let gateway = gateway_url
        .or_else(|| stored_settings.get().default_gateway_url.clone())
        .unwrap_or_else(|| DEFAULT_GATEWAY.to_string());
    // Fix 2 (pi-parity gap): `--reasoning-effort` previously had no persisted stored-default fallback at
    // all, unlike `default_model`/`default_gateway_url`/`default_session_dir` — same precedence tier
    // (explicit flag, already resolved against `AI_AGENT_REASONING_EFFORT` by clap, then this stored
    // setting, then finally left unset).
    let reasoning_effort = reasoning_effort.or_else(|| {
        stored_settings
            .get()
            .default_reasoning_effort
            .as_deref()
            .and_then(|s| parse_reasoning_effort(s).ok())
    });
    // Whether the operator explicitly passed `--model`, as opposed to `run` falling back to a stored
    // default or `DEFAULT_MODEL` — the distinction a reopened `--session`/`--continue` needs below to
    // know whether to keep going on the model the session was actually last driven on instead of
    // quietly switching it, the same bug class `switch_session` had (see
    // `Persistence::model_and_level_at_active` in `serve.rs`). A merely-stored default counts as *not*
    // explicit here — same as an unset flag — since the operator didn't ask for this specific
    // invocation to use it.
    let model_explicit = model.is_some();
    let model = model
        .or_else(|| stored_settings.get().default_model.clone())
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());
    // Fix 10 (pi-parity feature): resolve a partial/fuzzy `--model` id against the known-model hint list
    // before it's used for anything else — dialect inference, OAuth-provider inference, and the model
    // itself all key off the *resolved* id, so this must happen before `resolve_gateway_credential`
    // below. An ambiguous partial match fails the whole invocation clearly (naming every candidate)
    // rather than guessing; a genuinely unrecognized id (no partial match at all) is forwarded
    // unchanged — see `serve::resolve_model_id`'s own doc comment.
    let model = serve::resolve_model_id(&model, serve::available_models())
        .map_err(|e| format!("--model {model:?}: {e}"))?;
    let key = resolve_gateway_credential(key, &model)?;

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
        stored_settings.get().default_project_trust,
        beyond_ai_agent::trust_store::TrustStore::open_default().lookup(&cwd),
        has_gated_resources,
    );
    // pi-parity fix: an untrusted project with a `SYSTEM.md`/skills/prompts on disk silently skipped all
    // of them with no signal at all that anything was there — an operator debugging "why isn't my
    // SYSTEM.md taking effect" had nothing to go on. One line, matching this function's existing
    // `warning: ...` convention (see the `cwd_is_stale` check further down).
    if !project_trusted && has_gated_resources {
        eprintln!(
            "warning: {} has a project-local SYSTEM.md/APPEND_SYSTEM.md, skills, or prompt templates \
             on disk, but the project isn't trusted, so they were skipped — pass --trust-project or run \
             `agent trust {}` to enable them",
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
    let mut registry = tools::default_registry_with_prefix(
        bash_timeout_ms,
        bash_shell_path.as_deref(),
        bash_command_prefix.as_deref(),
    );
    tools::apply_filter(
        &mut registry,
        tools_allow.as_deref(),
        tools_exclude.as_deref(),
        no_tools,
    );
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
    let default_base = default_system_prompt(&registry, &prompt_guidelines);
    // Skills are discovered by path, not inlined into the prompt — invoking one relies on the model
    // being able to open its `SKILL.md` itself, so advertising them at all when `read` isn't registered
    // (a restricted `--tools`/`--exclude-tools` invocation) just adds dead weight (pi-parity fix).
    let has_read = registry.get("read").is_some();
    let system = beyond_ai_agent::resources::build_system_prompt(
        &beyond_ai_agent::resources::PromptOptions {
            base: system_prompt.as_deref(),
            default_base: &default_base,
            append: append_system_prompt.as_deref(),
            cwd: &cwd,
            include_context_files: !no_context_files,
            skills: &skills,
            has_read,
            project_trusted,
        },
    );
    timing.mark("build system prompt");

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
        match session_dir.or_else(|| stored_settings.get().default_session_dir.clone()) {
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
            None if continue_session => {
                let repo = SessionRepo::open(&repo_dir)?;
                // `None` here, not `session_id` — see `resume_or_create`'s doc comment: `--session-id`
                // is documented (and tested, above) to apply only to a genuinely fresh `--session <path>`
                // or a plain ephemeral run, never `--continue`.
                let (store, session) = repo.resume_or_create(&cwd_str, &model, None)?;
                (Some(store), session)
            }
            // pi-parity fix: previously always `(None, Session::new())` — in-memory only. Matches
            // `serve`'s own default (no `--session-file`/`--session-dir` given) exactly: the same
            // per-cwd repo, reattaching to this directory's most recent session if one already exists
            // rather than starting fresh every single invocation (`SessionRepo::resume_or_create`'s own
            // doc comment). `--session-id` *does* apply here (unlike the `--continue` arm above) — this
            // is exactly the "plain ephemeral run" case that flag's own doc comment already documents it
            // for.
            None if !no_session_persistence => {
                let repo = SessionRepo::open(&repo_dir)?;
                let (store, session) =
                    repo.resume_or_create(&cwd_str, &model, session_id.as_deref())?;
                (Some(store), session)
            }
            None => (None, Session::new()),
        }
    };
    // `--name`, applied uniformly across every path above (mirrors `serve`'s own startup check) —
    // only for a genuinely fresh session (no messages, no title yet). `--continue`'s `resume_or_create`
    // branch above mints its own fresh `SessionMeta` internally when no cwd match exists, bypassing the
    // `fresh_meta` closure other branches use, so this was previously the one path `--name` silently
    // never reached even when it *did* open a brand-new session.
    if let Some(name) = &name {
        if session.messages.is_empty() {
            if let Some(store) = &mut store {
                if store.meta().title.is_none() {
                    store.set_title(name)?;
                }
            }
        }
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

    let client = match key {
        serve::GatewayCredential::Static(key) => GatewayClient::new(gateway, key)?,
        serve::GatewayCredential::Oauth(source) => {
            GatewayClient::with_credential_source(gateway, source)?
        }
    }
    .with_retry(
        retry_max_retries.unwrap_or(agent_core::client::MAX_RETRIES),
        retry_base_delay_ms
            .map(std::time::Duration::from_millis)
            .unwrap_or(agent_core::client::BASE_BACKOFF),
    );
    // Task #50: the same two operator-supplied overrides also drive the *whole-run* retry layer
    // (`retry::RunRetryPolicy`), not just the pre-connect/mid-stream layer just above — previously
    // `--retry-max-retries`/`--retry-base-delay-ms` silently had no effect on `run_turn`'s own retry loop.
    let retry_policy = beyond_ai_agent::retry::RunRetryPolicy::from_overrides(
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
        .with_checkpoint_hook(Arc::new(DirectCheckpoint(store.clone())));
    // Unlike `serve`, `run` has no thinking-level cycling — these are applied as-is, with no per-model
    // default derivation when omitted (matching `run`'s prior behavior of not setting either at all).
    if let Some(budget) = thinking {
        agent = agent.with_thinking(budget);
    }
    if let Some(effort) = reasoning_effort {
        agent = agent.with_reasoning_effort(effort);
    }
    if let Some(temperature) = temperature {
        agent = agent.with_temperature(temperature);
    }
    if let Some(max_tokens) = max_tokens {
        agent = agent.with_max_tokens(max_tokens);
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

    let initial_message = expand_message(&initial_message, &skills, &prompt_templates);
    if initial_images.is_empty() {
        session.user(initial_message);
    } else {
        session.push(agent_core::Message::user_with_images(
            initial_message,
            initial_images,
        ));
    }
    let turn_result = run_turn(&agent, &mut session, json, &cancel, &retry_policy).await;
    // Persist whatever's in `session` regardless of outcome: `run_events_cancellable` mutates
    // `session` in place as it streams, so a cancelled turn still leaves behind whatever
    // assistant/tool content had already landed — the same partial-content guarantee `serve` gives
    // every session, not just the happy path. `DirectCheckpoint` already covers most of this
    // incrementally, but the turn's own tail (its final, possibly-partial assistant message) is
    // only ever captured here.
    persist_run_tail(&store, &session)?;
    let mut stop_reason = unwrap_turn_result(turn_result, &shutdown_cause)?;
    for message in messages {
        session.user(expand_message(&message, &skills, &prompt_templates));
        let turn_result = run_turn(&agent, &mut session, json, &cancel, &retry_policy).await;
        persist_run_tail(&store, &session)?;
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
    fn copilot_endpoint_path_strips_v1_for_openai_wire_dialects_but_not_anthropic() {
        // pi-parity fix: GitHub Copilot's OpenAI-wire endpoints omit the `/v1` prefix the dialect's
        // own default `endpoint_path()` carries (pi's own SDKs set `baseURL: model.baseUrl` with no
        // `/v1` in it, then the vendor SDK appends its fixed relative path) — only the Anthropic-wire
        // endpoint matches the dialect default verbatim (the Anthropic SDK's default `baseURL` has no
        // version segment at all, so it always appends `/v1/messages` itself).
        assert_eq!(
            copilot_endpoint_path(agent_core::dialect::Dialect::Anthropic),
            "/v1/messages"
        );
        assert_eq!(
            copilot_endpoint_path(agent_core::dialect::Dialect::OpenAi),
            "/chat/completions"
        );
        assert_eq!(
            copilot_endpoint_path(agent_core::dialect::Dialect::OpenAiResponses),
            "/responses"
        );
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
                tools::write::Write.input_schema()
            }
            fn write_target(&self, input: &serde_json::Value) -> Option<String> {
                tools::write::Write.write_target(input)
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
        struct ObservedWrite {
            started: Arc<tokio::sync::Notify>,
        }
        #[async_trait]
        impl agent_core::tool::Tool for ObservedWrite {
            fn name(&self) -> &str {
                "write"
            }
            fn description(&self) -> &str {
                tools::write::Write.description()
            }
            fn input_schema(&self) -> serde_json::Value {
                tools::write::Write.input_schema()
            }
            fn write_target(&self, input: &serde_json::Value) -> Option<String> {
                tools::write::Write.write_target(input)
            }
            async fn run(
                &self,
                input: serde_json::Value,
            ) -> std::result::Result<ToolOutput, agent_core::ToolError> {
                self.started.notify_one();
                tools::write::Write.run(input).await
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
        )
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
        let prompt = default_system_prompt(&registry, &[]);
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
        let prompt = default_system_prompt(&registry, &[]);
        assert!(!prompt.contains("bash"));
        assert!(prompt.contains("read"));
    }

    #[test]
    fn default_system_prompt_always_shows_the_file_paths_guideline() {
        // pi: system-prompt.test.ts, "shows file paths guideline even with no tools" — a built-in
        // guideline, always present regardless of the tool set (unlike the conditional bash one below).
        let registry = tools::default_registry();
        let prompt = default_system_prompt(&registry, &[]);
        assert!(prompt.contains("Show file paths clearly when working with files"));
    }

    #[test]
    fn default_system_prompt_tells_the_model_to_use_bash_for_exploration_without_grep_find_ls() {
        // pi: the one built-in conditional guideline — only fires when `bash` is registered but none
        // of its usual companions are, since the model then has no other way to explore the filesystem.
        let mut only_bash = agent_core::tool::ToolRegistry::new();
        only_bash.register(std::sync::Arc::new(tools::bash::Bash::real()));
        let prompt = default_system_prompt(&only_bash, &[]);
        assert!(prompt.contains("Use bash for file operations like ls, rg, find"));

        // The guideline must not fire when grep/find/ls are also registered — bash isn't the only
        // exploration tool anymore.
        let full = tools::default_registry();
        let prompt = default_system_prompt(&full, &[]);
        assert!(!prompt.contains("Use bash for file operations like ls, rg, find"));
    }

    #[test]
    fn default_system_prompt_includes_pis_per_tool_guidelines_for_read_edit_write() {
        // pi-parity fix: pi declares real default guidance on its read/edit/write tool definitions
        // (`promptGuidelines`), collected automatically whenever the tool is registered — we ported
        // only the operator-typed `--prompt-guideline` mechanism, not this content, so a model never
        // got told (for example) edit's exact-match/non-overlapping-edit semantics unless an operator
        // happened to type the same guidance in by hand.
        let registry = tools::default_registry();
        let prompt = default_system_prompt(&registry, &[]);
        assert!(
            prompt.contains("Use read to examine files instead of cat or sed."),
            "got: {prompt}"
        );
        assert!(
            prompt.contains("Use edit for precise changes (edits[].old_string must match exactly)"),
            "got: {prompt}"
        );
        assert!(
            prompt.contains("Keep edits[].old_string as small as possible"),
            "got: {prompt}"
        );
        assert!(
            prompt.contains("Use write only for new files or complete rewrites."),
            "got: {prompt}"
        );
    }

    #[test]
    fn default_system_prompt_omits_a_tools_guidelines_when_the_tool_is_not_registered() {
        let mut only_bash = agent_core::tool::ToolRegistry::new();
        only_bash.register(std::sync::Arc::new(tools::bash::Bash::real()));
        let prompt = default_system_prompt(&only_bash, &[]);
        assert!(!prompt.contains("Use read to examine files"));
        assert!(!prompt.contains("Use edit for precise changes"));
        assert!(!prompt.contains("Use write only for new files"));
    }

    #[test]
    fn default_system_prompt_appends_and_dedupes_extra_guidelines() {
        // pi: system-prompt.test.ts, "appends promptGuidelines to default guidelines" /
        // "deduplicates and trims promptGuidelines".
        let registry = tools::default_registry();
        let prompt = default_system_prompt(
            &registry,
            &[
                "Use dynamic_tool for project summaries.".to_string(),
                "  Use dynamic_tool for project summaries.  ".to_string(),
                "   ".to_string(),
            ],
        );
        assert_eq!(
            prompt
                .matches("- Use dynamic_tool for project summaries.")
                .count(),
            1,
            "got: {prompt}"
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
        let out = read_file_refs(&["a.txt".to_string()], dir.path())
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
        let err = read_file_refs(&["does-not-exist.txt".to_string()], dir.path())
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
        let out = read_file_refs(&["a.txt".to_string(), "b.txt".to_string()], dir.path())
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

        let out = read_file_refs(&["shot.png".to_string()], dir.path())
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

    #[test]
    fn looks_like_image_is_false_for_an_ordinary_text_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("notes.txt");
        std::fs::write(&path, "just some text").unwrap();
        assert!(!looks_like_image(&path));
    }
}
