//! Resolves which credential/routing a given model id should use to reach the gateway (or, for a
//! `models.json` override/GitHub Copilot/OpenAI Codex, bypass it) — the one seam both `main.rs`'s CLI
//! startup (`run`/`serve`) and `serve.rs`'s runtime model-switching commands (`set_model`,
//! `cycle_model`, and any other command that changes the active model — `switch_session`, `fork`,
//! `clone`, `switch_branch`) go through.
//!
//! [`resolve_gateway_credential`] is keyed on `model`, not on the process: a model switch can cross
//! providers (Anthropic OAuth → GitHub Copilot OAuth, say), so this is re-run every time the active
//! model changes rather than resolved once at process start and frozen for the rest of the process's
//! life — see `crate::serve::build_gateway_client`, the one place both the initial `serve` startup and
//! every later model-switch RPC command call back into this function.
//!
//! Fix #36 (pi-parity audit — perf, not correctness): `build_gateway_client` currently rebuilds a
//! brand-new [`agent_core::client::GatewayClient`] (and the `reqwest::Client`/connection pool that
//! construction always creates) on *every* call, even when the newly-resolved credential is for the
//! exact same upstream as the one already in use — a same-provider model switch (Claude Opus →
//! Claude Sonnet, say) discards a warm TCP/TLS connection for no reason. [`GatewayCredentialIdentity`]
//! is the primitive a fix would build on: a cheap, `PartialEq`-comparable summary of which upstream a
//! resolved [`GatewayCredential`] talks to, returned alongside it by
//! [`resolve_gateway_credential_with_identity`]. A future `build_gateway_client` could keep its
//! existing `Arc<GatewayClient>` across a model switch whenever the newly-resolved identity equals the
//! previous one, instead of rebuilding unconditionally — see that type's own doc comment for exactly
//! what such a caller would need to store and compare. Not yet wired into `serve.rs` (a sibling
//! agent's file this round); this module only lands the identity-computation primitive itself, plus a
//! unit test proving it distinguishes different upstreams and treats equivalent ones as equal.

use std::collections::HashMap;
use std::sync::Arc;

use agent_core::client::{Credential, CredentialSource, DirectRouting, RouteOverride};
use agent_core::models::AggregatorHost;
use providers::ProviderSpec;

use crate::oauth::{OAuthCredential, OAuthProviderId};

/// Everything the resolver needs to know about the process environment, read **once** at the CLI
/// boundary and passed in.
///
/// Not a convenience: this module reads no `std::env` at all, and must not start. `std::env::set_var`
/// is `unsafe` under a multi-threaded test binary, so any code that reads process env directly is
/// untestable except serially — a convention this repo states in several places (`resources.rs`,
/// `skills.rs`, `agents.rs`) and has zero call sites against. `AI_AGENT_KEY` already arrives here as a
/// parameter for exactly this reason (clap binds it in `main.rs`); the direct-routing env follows it
/// through the same door, which is what lets every precedence rule below be an ordinary unit test.
/// Pseudo-key under which `OPENAI_BASE_URL` is stashed in [`ProviderEnv::keys`] — a name no provider's
/// `env_var` can collide with, since it isn't one.
const OPENAI_BASE_URL_KEY: &str = "OPENAI_BASE_URL";

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProviderEnv {
    /// Provider key env vars that are actually set, keyed by the var name from
    /// [`providers::ProviderSpec::env_var`] — `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, ….
    keys: HashMap<&'static str, String>,
    /// `AI_PROVIDER` — names a provider row explicitly. The only way to reach an *aggregator*
    /// (OpenRouter, Groq, …), which by design no model id resolves to on its own.
    pub provider: Option<String>,
    /// `AI_BASE_URL` — the OpenAI-compatible escape hatch for the long tail (vLLM, Ollama, LM Studio,
    /// a corporate proxy). Also fed by `OPENAI_BASE_URL`, but **only** to override the OpenAI row's own
    /// base URL — see [`Self::from_process_env`].
    pub base_url: Option<String>,
    /// `AI_API_KEY` — the key that goes with [`Self::base_url`], and the fallback for an
    /// `AI_PROVIDER` row whose own env var isn't set.
    pub api_key: Option<String>,
    /// `AI_DIRECT=1` — route direct even though a gateway *is* configured. Without it, a deployment
    /// that has a gateway keeps using it: a stray `ANTHROPIC_API_KEY` in some engineer's shell must
    /// never silently reroute production traffic off the gateway (and off its metering).
    pub force_direct: bool,
    /// Whether a gateway is genuinely *configured* — `--gateway-url`/`AI_GATEWAY_URL`, or
    /// `settings.json`'s `default_gateway_url`, or a `--key`/`AI_AGENT_KEY`. Note the `DEFAULT_GATEWAY`
    /// constant (`http://ai.internal`) is a **fallback, not configuration**: falling back to it must not
    /// count, or the gateway could never be optional. Computed in `main.rs`, where all three inputs are.
    pub gateway_configured: bool,
}

impl ProviderEnv {
    /// Read the direct-routing environment. **The only place this crate touches `std::env`** for
    /// provider resolution; everything below takes the result as a parameter.
    ///
    /// `gateway_configured` is passed in rather than read here because it also depends on the parsed
    /// CLI flags and on `settings.json`, neither of which is env.
    pub fn from_process_env(gateway_configured: bool) -> Self {
        Self::from_lookup(&|name| std::env::var(name).ok(), gateway_configured)
    }

    /// [`Self::from_process_env`] over an explicit `(name, value)` list instead of the process — the same
    /// code path, so a test can never diverge from what the binary actually does. Public because the
    /// wire-level integration tests (`tests/direct_routing_wire.rs`) live outside this crate.
    pub fn from_vars(vars: &[(&str, &str)], gateway_configured: bool) -> Self {
        Self::from_lookup(
            &|name| {
                vars.iter()
                    .find(|(k, _)| *k == name)
                    .map(|(_, v)| (*v).to_string())
            },
            gateway_configured,
        )
    }

    fn from_lookup(lookup: &dyn Fn(&str) -> Option<String>, gateway_configured: bool) -> Self {
        let var = |name: &str| {
            lookup(name)
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
        };
        let mut keys = HashMap::new();
        for spec in providers::PROVIDERS {
            if let Some(env_var) = spec.env_var
                && let Some(value) = var(env_var)
            {
                keys.insert(env_var, value);
            }
        }
        // `OPENAI_BASE_URL` is honored, but ONLY as the OpenAI row's own base URL — never as a second
        // global escape hatch. It is a widely-exported variable (the OpenAI SDK reads it), and someone
        // who points it at a proxy for one tool must not thereby have their *Anthropic* traffic silently
        // redirected there. `AI_BASE_URL` is the generic hatch, and it wins.
        if let Some(openai_base) = var("OPENAI_BASE_URL") {
            keys.insert(OPENAI_BASE_URL_KEY, openai_base);
        }
        ProviderEnv {
            keys,
            provider: var("AI_PROVIDER"),
            base_url: var("AI_BASE_URL"),
            api_key: var("AI_API_KEY"),
            force_direct: matches!(var("AI_DIRECT").as_deref(), Some("1" | "true" | "yes")),
            gateway_configured,
        }
    }

    /// The base URL to dial for `spec` — its own, unless `OPENAI_BASE_URL` overrides the OpenAI row's.
    fn base_url_for(&self, spec: &ProviderSpec) -> Option<&str> {
        if spec.id == AggregatorHost::OpenAi
            && let Some(base) = self.keys.get(OPENAI_BASE_URL_KEY)
        {
            return Some(base);
        }
        spec.base_url
    }

    /// Whether this invocation routes straight to a provider rather than through the gateway. Direct is
    /// the default *absence* case — no gateway configured means no gateway to use — and `AI_DIRECT=1`
    /// is the opt-in override for when one is configured but unwanted.
    pub fn direct(&self) -> bool {
        self.force_direct || !self.gateway_configured
    }

    /// The user's own key for a provider row, if we have one: the row's conventional env var first
    /// (`ANTHROPIC_API_KEY`), then `AI_API_KEY`, then whatever `--key`/`AI_AGENT_KEY` held.
    ///
    /// **A row's env var only ever pays for that row.** `ANTHROPIC_API_KEY` is never sent to
    /// `AI_BASE_URL`, or to any other provider's host — the lookup is keyed on the row, not on "some
    /// key we happen to have". A key leaked to the wrong host is a key the user must rotate.
    fn key_for(&self, spec: &ProviderSpec, cli_key: Option<&str>) -> Option<String> {
        spec.env_var
            .and_then(|v| self.keys.get(v).cloned())
            .or_else(|| self.api_key.clone())
            .or_else(|| cli_key.map(str::to_string))
    }
}

/// How the gateway client attaches a credential to every request — either a fixed `bai_v1…`/BYO
/// string (the ordinary case), or a live, transparently-refreshed OAuth subscription login (see
/// `crate::auth_credential_source::OAuthCredentialSource`). Resolved by [`resolve_gateway_credential`]
/// fresh for whichever model is currently active, not cached for the process's lifetime.
pub enum GatewayCredential {
    Static(String),
    Oauth(Arc<dyn CredentialSource>),
}

/// OpenAI's own approved identity string for this tool's OAuth grant (`build_authorize_url` in
/// `oauth/openai_codex.rs` already sends this as the `originator` query param at login time) — reused
/// verbatim here so a live Codex inference request presents the same identity the account authorized,
/// rather than a second, inconsistent one.
const CODEX_ORIGINATOR: &str = "beyond-ai-agent";

/// Adapts a plain OAuth [`CredentialSource`] (bearer token + `is_oauth`, from `OAuthCredentialSource`)
/// with routing info the source itself has no way to compute — currently only OpenAI Codex's distinct
/// backend path/headers, which genuinely are fixed for the account (the `chatgpt-account-id` claim
/// doesn't change across an in-process token refresh, unlike GitHub Copilot's proxy host — see
/// `crate::oauth::github_copilot::CopilotRoutedCredentialSource` for that one, which needs to re-derive
/// its routing on every call instead of reusing a value computed once here).
struct DirectRoutedCredentialSource {
    inner: Arc<dyn CredentialSource>,
    routing: DirectRouting,
}

#[async_trait::async_trait]
impl CredentialSource for DirectRoutedCredentialSource {
    async fn credential(&self) -> agent_core::Result<Credential> {
        let credential = self.inner.credential().await?;
        Ok(credential.with_direct_routing(self.routing.clone()))
    }
}

/// A [`CredentialSource`] for a `models.json` `base_url` override (Fix 9 — pi-parity feature: pi's own
/// `model-registry.ts` custom-model/override support). Unlike [`DirectRoutedCredentialSource`] above
/// (which wraps an existing OAuth source and only adds routing), this *is* the credential: a fixed
/// bearer token — the override's own `api_key`, else whatever `--key`/`AI_AGENT_KEY` resolved to, else
/// empty (many self-hosted OpenAI-compatible servers, like Ollama/LM Studio, ignore the
/// `Authorization` header entirely) — plus the same [`DirectRouting`] mechanism reused, not
/// duplicated, to send the request straight to the override's `base_url`, bypassing the gateway
/// outright.
struct StaticDirectCredentialSource {
    bearer: String,
    routing: DirectRouting,
}

#[async_trait::async_trait]
impl CredentialSource for StaticDirectCredentialSource {
    async fn credential(&self) -> agent_core::Result<Credential> {
        Ok(Credential::new(self.bearer.clone(), false).with_direct_routing(self.routing.clone()))
    }
}

/// The one further fallback tier below `--key`/`AI_AGENT_KEY`: an inferred, stored OAuth subscription
/// login for whichever provider `model` implies, or a `models.json` `base_url` override naming `model`
/// explicitly. Consulted fresh every time the active model changes — at process start, and again on
/// every runtime `set_model`/`cycle_model`/`switch_session`/`fork`/`clone`/`switch_branch` (see
/// `crate::serve::build_gateway_client`) — never cached across a model switch, since the resolved
/// provider/routing can differ entirely from one model to the next.
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
pub fn resolve_gateway_credential(
    key: Option<String>,
    model: &str,
    env: &ProviderEnv,
) -> Result<GatewayCredential, String> {
    resolve_gateway_credential_with_identity(key, model, env)
        .map(|(credential, _identity)| credential)
}

/// [`resolve_gateway_credential`], additionally returning a [`GatewayCredentialIdentity`] alongside
/// the resolved credential — see that type's own doc comment (Fix #36, pi-parity audit). The sole
/// source of truth for both: [`resolve_gateway_credential`] is a thin wrapper around this function that
/// discards the identity half, so the two can never drift out of sync with each other.
pub fn resolve_gateway_credential_with_identity(
    key: Option<String>,
    model: &str,
    env: &ProviderEnv,
) -> Result<(GatewayCredential, GatewayCredentialIdentity), String> {
    // Opened unconditionally, up front: both the `base_url`-override branch below (its own OAuth
    // fallback, pi-parity remediation pass 19 Task 2) and the plain non-override branches further down
    // need a stored-credential lookup, and a missing file is the cheap, ordinary case either way (see
    // `AuthStore::open_default`'s own doc comment).
    resolve_with(
        key,
        model,
        env,
        &crate::auth_store::AuthStore::open_default(),
        crate::auth_store::default_path(),
        &crate::settings::ModelOverrides::open_default(),
    )
}

/// Resolve `model`'s gateway credential AND its extra per-request headers from a SINGLE parse of
/// `models.json` (T9-F2/F3). A client build needs both — the credential (which routing/auth) and any
/// `ModelOverride::headers` merged onto every request via `GatewayClient::with_extra_headers` — and
/// they come from the same on-disk override row. Resolving them separately — `resolve_gateway_credential`
/// alongside a standalone `ModelOverrides::open_default().get(model).resolved_headers()` — re-opened and
/// re-parsed `models.json` a second time microseconds later, once per client build (per subagent spawn,
/// per `run` start). This parses it once and feeds the one `ModelOverrides` to both.
///
/// `auth.json` is still read fresh here (via `AuthStore::open_default`) rather than from a longer-lived
/// snapshot: a token refresh rewrites it mid-run, and a stale snapshot would misroute — so the
/// deduplication is deliberately scoped to the per-build `models.json` double-parse, not a
/// process-lifetime cache of either file.
pub fn resolve_gateway_credential_and_headers(
    key: Option<String>,
    model: &str,
    env: &ProviderEnv,
) -> Result<(GatewayCredential, std::collections::HashMap<String, String>), String> {
    let overrides = crate::settings::ModelOverrides::open_default();
    let (credential, _identity) = resolve_with(
        key,
        model,
        env,
        &crate::auth_store::AuthStore::open_default(),
        crate::auth_store::default_path(),
        &overrides,
    )?;
    let headers = overrides
        .get(model)
        .map(|over| over.resolved_headers())
        .unwrap_or_default();
    Ok((credential, headers))
}

/// [`resolve_gateway_credential_with_identity`] with its two `$HOME`-dependent inputs — the stored-OAuth
/// credentials and `models.json` — passed in rather than read from disk.
///
/// The whole precedence ladder lives here, and this is the seam every test drives it through. Reading
/// `~/.claude/*` inside the resolver would make the ladder testable only on a machine with the right
/// files (and untestable in parallel, since the alternative is mutating `$HOME`) — so the two file reads
/// stay in the thin wrapper above, and the logic below is a pure function of its arguments. Same
/// convention `ProviderEnv` follows for process env.
#[allow(clippy::too_many_arguments)]
fn resolve_with(
    key: Option<String>,
    model: &str,
    env: &ProviderEnv,
    store: &crate::auth_store::AuthStore,
    store_path: std::path::PathBuf,
    overrides: &crate::settings::ModelOverrides,
) -> Result<(GatewayCredential, GatewayCredentialIdentity), String> {
    let oauth_source = |provider: OAuthProviderId| {
        Arc::new(crate::auth_credential_source::OAuthCredentialSource::new(
            provider,
            store_path.clone(),
        )) as Arc<dyn CredentialSource>
    };

    // Fix 9 (pi-parity feature): a `models.json` override naming a `base_url` for this exact model id
    // redirects *where* the request goes (a locally-hosted or alternate-provider endpoint, entirely
    // bypassing the gateway) — reusing the same `DirectRouting`/`RouteOverride::Direct` mechanism the
    // GitHub-Copilot OAuth routing below already relies on, rather than duplicating it (see
    // `settings::ModelOverride`'s own doc comment for the on-disk schema). *How* it authenticates is a
    // genuinely separate, unconditional question, resolved the same way whether or not an override is in
    // play (pi-parity remediation pass 19, Task 2): an override with no `api_key`/`auth_header` of its
    // own still falls through to `--key`/`AI_AGENT_KEY`, then a stored OAuth login (Anthropic/OpenAI
    // Codex — see [`oauth_fallback_provider`]'s own doc comment for why not GitHub Copilot too), exactly
    // as if there were no override at all — only when *none* of those resolve anything does this override
    // finally send an empty bearer (many self-hosted OpenAI-compatible servers ignore `Authorization`
    // entirely, so that's a usable default, not a hard failure). This doc comment used to claim exactly
    // this ("orthogonal to how it authenticates") without the implementation actually delivering it —
    // OAuth was never consulted at all for an override, silently going bearer-less for a model an
    // operator had a perfectly good subscription login for; this fix (and the paragraph above) make the
    // two agree.
    if let Some(over) = overrides.get(model)
        && let Some(base_url) = over.base_url.clone()
    {
        // Fix #39 (pi-parity audit, judgment call): a handful of third-party-aggregator model ids
        // (OpenCode Zen's `gemini-3-flash`/`gemini-3.1-pro`/`gemini-3.5-flash`) speak Google's own
        // Generative AI wire format, which `agent_core::dialect::Dialect` has no variant for (the
        // standing "Gemini-direct native dialect" deferral — out of scope to build here). Left
        // unchecked, a BYO override naming one of these ids with no explicit `dialect` would
        // silently fall back to `Dialect::for_model`'s OpenAI-family default and send an
        // OpenAI-shaped body to an endpoint that doesn't understand it — a confusing *provider*-side
        // failure instead of a clear one from us. An explicit `dialect` on the override is the
        // escape hatch (e.g. an operator fronting Gemini with an OpenAI-compatible proxy of their
        // own) — this only fires when that's genuinely unset.
        if over.dialect.is_none()
            && let Some(reason) = unsupported_wire_format_reason(model)
        {
            return Err(reason);
        }
        // pi-parity pass 20, Task 3/5: which known third-party aggregator (if any) this override's
        // `base_url` names — see [`aggregator_host_for_base_url`]'s own doc comment. Computed here,
        // before `dialect` below, so both the dialect selection AND `DirectRouting::aggregator_host`
        // (threaded across the crate boundary to `agent_core::client::GatewayClient::stream`, which
        // sets `ModelRequest::host` from it) reuse the identical value — no second, independent
        // detection path.
        let aggregator_host = aggregator_host_for_base_url(&base_url);
        // Fix 1 (pi-parity, Round 2): an explicit `dialect` override wins over
        // `Dialect::for_model_with_host`'s name/host heuristic — consulted here (to pick the right
        // endpoint path for a provider whose model ids don't match the heuristic, e.g. Kimi-Coding's
        // `kimi-k2-thinking`) AND threaded into `DirectRouting::dialect_override` below (for the
        // actual body-building/decoding dialect `GatewayClient::stream` picks), so the two never
        // disagree. Failing that, `for_model_with_host` is already host-aware (pi-parity pass 20,
        // Task 6): a handful of OpenCode Zen/OpenCode-Go id/host combinations need a *different*
        // default than `NATIVE_ANTHROPIC_WIRE_BARE_IDS`'s own host-agnostic default provides, and
        // `aggregator_host` (just computed above) is exactly the signal that resolves them —
        // formerly a separate `opencode_dialect_override` helper in this file (pi-parity remediation
        // pass 19, Task 1), now folded into the one shared mechanism instead of a second, parallel
        // "which host is this" check.
        let dialect = over.dialect.unwrap_or_else(|| {
            agent_core::dialect::Dialect::for_model_with_host(model, aggregator_host)
        });
        // Task #11 (pi-parity feature): resolved through `!command`/`$VAR`/literal syntax (see
        // `ModelOverride::resolved_api_key`'s own doc comment) rather than used as a raw literal —
        // lets an operator avoid storing a plaintext secret in `models.json`.
        let bearer = over.resolved_api_key(key.as_deref());
        // Fix 3 (pi-parity, Round 2): computed together so a `deployment_name` override's URL
        // path segment (Task 46) and the `/v1`-doubling fix (Task 45) never fight each other — see
        // `direct_route_base_and_path`'s own doc comment. Fix #33/#34 (pi-parity audit): also skips
        // the deployment-segment insertion for the Responses dialect, and auto-normalizes a bare/
        // partial Azure `base_url` to the canonical `/openai/v1` — see that function's own doc
        // comment for both.
        let (base_url, path) =
            direct_route_base_and_path(&base_url, dialect, over.deployment_name.as_deref());
        // Fix #35 (pi-parity audit): whether this override is genuinely an Azure endpoint at all —
        // gates `azure_api_version_query`'s default so a plain (non-Azure) BYO override never picks
        // up an unsolicited `?api-version=v1` it never asked for.
        let is_azure = over.deployment_name.is_some() || is_azure_host(&base_url);
        let query = azure_api_version_query(over.api_version.as_deref(), is_azure);
        let routing = DirectRouting {
            route: RouteOverride::Direct {
                base_url: base_url.clone(),
                path,
            },
            static_headers: Vec::new(),
            copilot_dynamic_headers: false,
            // Task #8 (pi-parity: Azure OpenAI routing support) — an operator-configured
            // `auth_header` (e.g. `"api-key"` for Azure) sends `bearer` through that named header
            // and omits `Authorization` entirely, instead of leaking a Bearer-shaped credential
            // (or, worse, a silent fallback to the gateway's own virtual key) to an endpoint that
            // doesn't want it.
            auth_header: over.auth_header.clone(),
            // Fix 4 (pi-parity, Round 2): Cloudflare AI Gateway's Bearer-prefixed named header
            // (`cf-aig-authorization: Bearer <key>`) — `None` preserves `auth_header`'s existing
            // bare-value behavior (Azure's `api-key`).
            auth_header_prefix: over.auth_header_prefix.clone(),
            dialect_override: over.dialect,
            // Fix 2 (pi-parity, Round 2): Azure OpenAI's deployment-name mapping — sends this
            // instead of `model` as the wire-level `"model"` field.
            deployment_name: over.deployment_name.clone(),
            // Fix 2 (pi-parity, Round 2): Azure's dated `api-version` query param.
            query: query.clone(),
            // pi-parity pass 20, Task 5: threads the same `aggregator_host` computed above across
            // the crate boundary so `GatewayClient::stream` can set `ModelRequest::host` from it —
            // see `DirectRouting::aggregator_host`'s own doc comment.
            aggregator_host,
        };
        // pi-parity remediation pass 19, Task 2: the fallback tier below both this override's own
        // credential (`bearer`, just above) and `--key`/`AI_AGENT_KEY` — a stored OAuth login, still
        // routed to this override's own `base_url`/`routing` rather than the provider's usual
        // endpoint. Reuses `DirectRoutedCredentialSource` (already used for Codex's own fixed
        // routing further below) rather than inventing a second wrapper — the only thing that
        // differs per call site is which `DirectRouting` it carries.
        if let Some(provider) = oauth_fallback_provider(model, over, key.as_deref())
            && let Some(stored) = store.get(provider.store_key())
            && stored.credential.provider() == provider
        {
            let identity = GatewayCredentialIdentity::DirectOverrideOauth {
                base_url: base_url.clone(),
                path,
                auth_header: over.auth_header.clone(),
                auth_header_prefix: over.auth_header_prefix.clone(),
                deployment_name: over.deployment_name.clone(),
                query: query.clone(),
                aggregator_host,
                provider,
            };
            return Ok((
                GatewayCredential::Oauth(Arc::new(DirectRoutedCredentialSource {
                    inner: oauth_source(provider),
                    routing: routing.clone(),
                })),
                identity,
            ));
        }
        let identity = GatewayCredentialIdentity::DirectOverride {
            base_url,
            path,
            bearer: bearer.clone(),
            auth_header: over.auth_header.clone(),
            auth_header_prefix: over.auth_header_prefix.clone(),
            deployment_name: over.deployment_name.clone(),
            query,
            aggregator_host,
        };
        return Ok((
            GatewayCredential::Oauth(Arc::new(StaticDirectCredentialSource { bearer, routing })),
            identity,
        ));
    }

    // --- Direct tiers, part 1: the two *explicit* ones. -------------------------------------------
    //
    // Both sit above stored OAuth because both are per-invocation acts: someone who typed
    // `AI_BASE_URL=…` or `AI_PROVIDER=…` for this run means it. The *ambient* tier (a provider key that
    // merely happens to be exported in the shell) sits BELOW OAuth instead — see part 2, further down.
    if env.direct() {
        // The long tail: any OpenAI-compatible endpoint — vLLM, Ollama, LM Studio, a corporate proxy —
        // with no registry row required. If the URL does happen to name a known provider we use that
        // row (right auth header, right dialect) rather than assuming OpenAI-wire-and-Bearer; otherwise
        // the id-only heuristic picks the wire and the key goes out as a plain Bearer. An empty key is
        // legitimate here, not a failure: local servers routinely ignore `Authorization` entirely.
        if let Some(base_url) = env.base_url.as_deref() {
            if let Some(spec) = provider_for_base_url(base_url) {
                let bearer = env.key_for(spec, key.as_deref()).unwrap_or_default();
                let (routing, base_url, path) = registry_direct_routing(spec, base_url, model);
                return Ok(direct_static(
                    bearer,
                    routing,
                    base_url,
                    path,
                    Some(spec.id),
                ));
            }
            let dialect = agent_core::dialect::Dialect::for_model_via_provider(model, None);
            let (base_url, path) = direct_route_base_and_path(base_url, dialect, None);
            // NOT `key_for`: with no row, there is no row-scoped env var to draw on, and reaching for
            // one would be exactly the leak the invariant forbids — `ANTHROPIC_API_KEY` must never be
            // handed to an arbitrary `AI_BASE_URL`. Only the key the user paired with this URL.
            let bearer = env
                .api_key
                .clone()
                .or_else(|| key.clone())
                .unwrap_or_default();
            let routing = DirectRouting {
                route: RouteOverride::Direct {
                    base_url: base_url.clone(),
                    path,
                },
                static_headers: Vec::new(),
                copilot_dynamic_headers: false,
                auth_header: None,
                auth_header_prefix: None,
                dialect_override: Some(dialect),
                deployment_name: None,
                query: None,
                aggregator_host: None,
            };
            return Ok(direct_static(bearer, routing, base_url, path, None));
        }
        // `AI_PROVIDER=openrouter` — the only way to reach an aggregator, and deliberately so: an
        // aggregator serves other vendors' model ids, so no id can imply one without stealing that
        // vendor's own native route (`providers::ProviderSpec::model_id_match`).
        if let Some(name) = env.provider.as_deref() {
            let spec = providers::by_name(name).ok_or_else(|| {
                let known = providers::PROVIDERS
                    .iter()
                    .filter(|p| p.base_url.is_some())
                    .map(|p| p.name)
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("unknown AI_PROVIDER {name:?}; known providers: {known}")
            })?;
            let base_url = env.base_url_for(spec).ok_or_else(|| {
                format!(
                    "provider {:?} has no direct route: it is reachable only through a stored OAuth \
                     login or an explicit models.json base_url",
                    spec.name
                )
            })?;
            let bearer = env.key_for(spec, key.as_deref()).ok_or_else(|| {
                format!(
                    "no key for provider {:?}: set {} (or AI_API_KEY)",
                    spec.name,
                    spec.env_var.unwrap_or("a provider key")
                )
            })?;
            let (routing, base_url, path) = registry_direct_routing(spec, base_url, model);
            return Ok(direct_static(
                bearer,
                routing,
                base_url,
                path,
                Some(spec.id),
            ));
        }
    }

    // The gateway's own credential. Skipped in direct mode: `--key`/`AI_AGENT_KEY` is a *gateway*
    // credential (a `bai_v1…` virtual key), and handing it to `GatewayClient` with no route override
    // would send it to a gateway base URL we've just established isn't there. It stays available to the
    // direct tiers as a last-resort BYO key (`ProviderEnv::key_for`), which is what makes
    // `AI_DIRECT=1 --key sk-ant-…` do the obvious thing.
    if !env.direct()
        && let Some(key) = key
    {
        let identity = GatewayCredentialIdentity::StaticKey(key.clone());
        return Ok((GatewayCredential::Static(key), identity));
    }

    if agent_core::dialect::Dialect::for_model(model) == agent_core::dialect::Dialect::Anthropic
        && store.get("anthropic").is_some()
    {
        // Gateway mode: no route override at all — relayed through the gateway, which knows Anthropic as
        // a `KNOWN_PROVIDERS` row. Direct mode: the same OAuth token, sent straight to
        // `api.anthropic.com`. The identity headers Anthropic's OAuth endpoint requires
        // (`claude-code`/`oauth` betas, the CLI user-agent) survive the route override because
        // `GatewayClient::stream` asks the routing *which provider it points at* rather than assuming
        // any override means a third party — see its `routed_to_anthropic_natively`.
        if env.direct() {
            let spec = providers::by_id(AggregatorHost::Anthropic);
            if let Some(base_url) = spec.base_url {
                let (mut routing, base_url, path) = registry_direct_routing(spec, base_url, model);
                // The row's `x-api-key` scheme describes how Anthropic takes an **API key**. An OAuth
                // access token is not an API key: it goes in `Authorization: Bearer`, and sending it in
                // `x-api-key` gets it rejected outright. The row is right about the host, the wire, and
                // the path; the credential decides the header.
                routing.auth_header = None;
                routing.auth_header_prefix = None;
                return Ok(direct_oauth(
                    oauth_source(OAuthProviderId::Anthropic),
                    routing,
                    base_url,
                    path,
                    Some(spec.id),
                    OAuthProviderId::Anthropic,
                ));
            }
        }
        return Ok((
            GatewayCredential::Oauth(oauth_source(OAuthProviderId::Anthropic)),
            GatewayCredentialIdentity::Anthropic,
        ));
    }
    if model.contains("codex")
        && let Some(stored) = store.get("openai-codex")
        && let OAuthCredential::OpenaiCodex(c) = &stored.credential
    {
        // Gateway mode relays through the `/openai-codex` prefix (`chatgpt.com` is a genuinely
        // static host, so it gets a real provider row). Direct mode dials that host itself — the
        // prefix route is *defined* in terms of the gateway's base URL, so with no gateway it
        // would resolve against a host that isn't there. Same account header, same path, same
        // bearer either way; only where the URL is rooted changes.
        let route = if env.direct() {
            RouteOverride::Direct {
                base_url: "https://chatgpt.com".to_string(),
                path: "/backend-api/codex/responses",
            }
        } else {
            RouteOverride::Prefixed {
                prefix: "/openai-codex",
                path: "/backend-api/codex/responses",
            }
        };
        let routing = DirectRouting {
            route,
            static_headers: vec![
                ("chatgpt-account-id", c.account_id.clone()),
                ("originator", CODEX_ORIGINATOR.to_string()),
                ("OpenAI-Beta", "responses=experimental".to_string()),
            ],
            copilot_dynamic_headers: false,
            auth_header: None,
            auth_header_prefix: None,
            dialect_override: None,
            deployment_name: None,
            query: None,
            // What tells `GatewayClient::stream` this is the Codex backend (`req.is_codex`, and
            // with it the Responses-body/zstd-SSE branch). It used to read that off the
            // `Prefixed` route shape, which stopped identifying Codex the moment Codex also
            // became reachable as a plain `Direct` route above.
            aggregator_host: Some(AggregatorHost::OpenAiCodex),
        };
        let identity = GatewayCredentialIdentity::OpenaiCodex {
            account_id: c.account_id.clone(),
        };
        return Ok((
            GatewayCredential::Oauth(Arc::new(DirectRoutedCredentialSource {
                inner: oauth_source(OAuthProviderId::OpenaiCodex),
                routing,
            })),
            identity,
        ));
    }
    if let Some(stored) = store.get("github-copilot")
        && let OAuthCredential::GithubCopilot(c) = &stored.credential
        && c.available_model_ids.iter().any(|m| m == model)
    {
        // Bypasses the gateway entirely: GitHub hands back a *different* proxy host per
        // account/enterprise, embedded in the access token itself (`proxy-ep=…`) — not a
        // static host the gateway's `KNOWN_PROVIDERS` table could ever hold as a row. See
        // `RouteOverride::Direct`. Re-derived from the CURRENT token on every request by
        // `CopilotRoutedCredentialSource` itself (not computed here and frozen) — see that
        // type's own doc comment for why.
        //
        // `for_model_via_copilot(.., true, ..)`, not plain `for_model`: at least one id
        // (`gpt-4.1`) is a different dialect under Copilot than it is natively (pi-parity — see
        // that function's doc comment), and this dialect also picks `copilot_endpoint_path`'s
        // baked-in `path` below, so getting it wrong here would send that id's Chat-Completions
        // body to a `/responses` path. `host: None` — Copilot is matched via a stored
        // credential's `available_model_ids`, never a BYO `base_url` override, so it has no
        // `AggregatorHost` of its own to report.
        let dialect = agent_core::dialect::Dialect::for_model_via_copilot(model, true, None);
        let path = crate::oauth::github_copilot::copilot_endpoint_path(dialect);
        let identity = GatewayCredentialIdentity::GithubCopilot {
            enterprise_url: c.enterprise_url.clone(),
            path,
        };
        return Ok((
            GatewayCredential::Oauth(Arc::new(
                crate::oauth::github_copilot::CopilotRoutedCredentialSource {
                    inner: oauth_source(OAuthProviderId::GithubCopilot),
                    store_path: store_path.clone(),
                    enterprise_url: c.enterprise_url.clone(),
                    path,
                    cached_routing: std::sync::Mutex::new(None),
                },
            )),
            identity,
        ));
    }

    // --- Direct tiers, part 2: the *ambient* one. -------------------------------------------------
    //
    // A provider key sitting in the environment. This is the zero-config path — export
    // `ANTHROPIC_API_KEY`, run the agent, done — and it is deliberately the LAST thing consulted.
    //
    // It sits below stored OAuth on purpose. `agent login anthropic` is an explicit, durable act;
    // `ANTHROPIC_API_KEY` is very often exported in a shell for some unrelated tool. Ranking the key
    // above the login would silently move a user who *has* a subscription onto pay-per-token API
    // billing, with nothing on screen to say so. If someone genuinely wants the key despite a stored
    // login, `AI_PROVIDER=anthropic` says so explicitly and is honored above (part 1).
    if env.direct()
        && let Some(spec) = providers::for_model_id(model)
        && let Some(base_url) = env.base_url_for(spec)
        && let Some(bearer) = spec
            .env_var
            .and_then(|v| env.keys.get(v).cloned())
            .or_else(|| env.api_key.clone())
    {
        let (routing, base_url, path) = registry_direct_routing(spec, base_url, model);
        return Ok(direct_static(
            bearer,
            routing,
            base_url,
            path,
            Some(spec.id),
        ));
    }

    // Name the one thing that would fix it. In direct mode we know which provider serves this model, so
    // we can name its env var exactly; if we don't recognize the id at all, say *that* rather than
    // suggest a variable that would be ignored.
    if env.direct() {
        return Err(match providers::for_model_id(model) {
            Some(spec) => format!(
                "no credential for model {model:?}: set {} (or run `agent login`, or set AI_PROVIDER \
                 + AI_API_KEY, or point AI_BASE_URL at an OpenAI-compatible endpoint)",
                spec.env_var.unwrap_or("a provider key")
            ),
            None => format!(
                "no provider recognizes model {model:?}: set AI_PROVIDER (e.g. AI_PROVIDER=openrouter) \
                 with that provider's key, or point AI_BASE_URL + AI_API_KEY at an OpenAI-compatible \
                 endpoint"
            ),
        });
    }
    Err(format!(
        "no gateway key: pass --key or set AI_AGENT_KEY (a bai_v1… virtual key), or run `agent login \
         <provider>` to use a subscription for model {model:?}"
    ))
}

/// A direct route authenticated by a fixed key — the shape every registry/`AI_BASE_URL` tier returns.
/// Reuses [`StaticDirectCredentialSource`] and the existing [`GatewayCredentialIdentity::DirectOverride`]
/// rather than minting parallel types: from `GatewayClient`'s point of view a registry-resolved route
/// and a hand-written `models.json` route are the same thing, and they should stay the same thing.
fn direct_static(
    bearer: String,
    routing: DirectRouting,
    base_url: String,
    path: &'static str,
    aggregator_host: Option<AggregatorHost>,
) -> (GatewayCredential, GatewayCredentialIdentity) {
    let identity = GatewayCredentialIdentity::DirectOverride {
        base_url,
        path,
        bearer: bearer.clone(),
        auth_header: routing.auth_header.clone(),
        auth_header_prefix: routing.auth_header_prefix.clone(),
        deployment_name: None,
        query: None,
        aggregator_host,
    };
    (
        GatewayCredential::Oauth(Arc::new(StaticDirectCredentialSource { bearer, routing })),
        identity,
    )
}

/// A direct route authenticated by a live, refreshing OAuth login — direct-mode Anthropic. Same
/// reuse argument as [`direct_static`]: [`DirectRoutedCredentialSource`] already exists to bolt routing
/// onto an OAuth source, and this is that, with a different destination.
fn direct_oauth(
    inner: Arc<dyn CredentialSource>,
    routing: DirectRouting,
    base_url: String,
    path: &'static str,
    aggregator_host: Option<AggregatorHost>,
    provider: OAuthProviderId,
) -> (GatewayCredential, GatewayCredentialIdentity) {
    let identity = GatewayCredentialIdentity::DirectOverrideOauth {
        base_url,
        path,
        auth_header: routing.auth_header.clone(),
        auth_header_prefix: routing.auth_header_prefix.clone(),
        deployment_name: None,
        query: None,
        aggregator_host,
        provider,
    };
    (
        GatewayCredential::Oauth(Arc::new(DirectRoutedCredentialSource { inner, routing })),
        identity,
    )
}

/// Build the [`DirectRouting`] for a provider row: base URL, endpoint path, wire dialect, auth header,
/// and any headers the provider requires — all of it derived from the row, none of it configured.
///
/// This is the whole point of the shared table. **The user never spells out an auth scheme.** Anthropic
/// gets `x-api-key` with a bare value and OpenAI gets `Authorization: Bearer` because
/// [`providers::AuthScheme`] says so, and it says so in the same rows the gateway swaps its pool keys
/// with — so the two can't disagree about a provider. `AuthScheme::Bearer` maps to `auth_header: None`,
/// which is `GatewayClient`'s own `Authorization: Bearer` default (it would otherwise send the header
/// twice).
///
/// `base_url` is passed rather than read from `spec.base_url` so `OPENAI_BASE_URL` can override the
/// OpenAI row's own endpoint without a second copy of this function.
fn registry_direct_routing(
    spec: &'static ProviderSpec,
    base_url: &str,
    model: &str,
) -> (DirectRouting, String, &'static str) {
    // Provider first, dialect second — see `Dialect::for_model_via_provider`'s doc comment. Passing the
    // row here is what stops an OpenRouter-served `anthropic/claude-…` from being built as an
    // Anthropic-wire body.
    let dialect = agent_core::dialect::Dialect::for_model_via_provider(model, Some(spec.id));
    // Reused, not reimplemented: the same helper the `models.json` branch uses, so a registry route and
    // a hand-written override compose their URL the same way (`/v1`-doubling detection included — which
    // is exactly why Anthropic's `base_url` carries no `/v1` and the OpenAI-wire rows' do).
    let (base_url, path) = direct_route_base_and_path(base_url, dialect, None);
    let (auth_header, auth_header_prefix) = match spec.auth {
        // `None` ⇒ `GatewayClient` sends its default `Authorization: Bearer <key>`.
        providers::AuthScheme::Bearer => (None, None),
        other => (
            Some(other.header().to_string()),
            other.value_prefix().map(str::to_string),
        ),
    };
    let routing = DirectRouting {
        route: RouteOverride::Direct {
            base_url: base_url.clone(),
            path,
        },
        // Hard upstream requirements, not conveniences: NVIDIA NIM's poll timeout, Kimi's `User-Agent`
        // allowlist check. Empty for every other row.
        static_headers: spec
            .default_headers
            .iter()
            .map(|(name, value)| (*name, (*value).to_string()))
            .collect(),
        copilot_dynamic_headers: false,
        auth_header,
        auth_header_prefix,
        // Pinned rather than left to be re-derived downstream: `GatewayClient::stream` would otherwise
        // fall back to the id-only heuristic and disagree with the `path` just chosen above.
        dialect_override: Some(dialect),
        deployment_name: None,
        query: None,
        aggregator_host: Some(spec.id),
    };
    (routing, base_url, path)
}

/// Model ids that speak a wire format `agent_core::dialect::Dialect` has no variant for — currently
/// only OpenCode Zen's 3 Gemini-hosted ids (Fix #39, pi-parity audit): pi's own `opencode.models.ts`
/// declares these with `api: "google-generative-ai"`. Neither `opencode` nor `opencode-go`'s *own*
/// default catalog routes here — this only ever fires for an operator-authored `models.json` override
/// that names one of these exact ids with no explicit `dialect`, a narrow, opt-out-respecting guard
/// rather than a broad heuristic.
const UNSUPPORTED_GOOGLE_GENERATIVE_AI_MODEL_IDS: &[&str] =
    &["gemini-3-flash", "gemini-3.1-pro", "gemini-3.5-flash"];

/// `Some(reason)` if `model` is a known-unroutable id (see
/// [`UNSUPPORTED_GOOGLE_GENERATIVE_AI_MODEL_IDS`]), else `None` — factored out so
/// [`resolve_gateway_credential_with_identity`]'s call site reads as a plain early-return and this is
/// independently unit-testable.
fn unsupported_wire_format_reason(model: &str) -> Option<String> {
    UNSUPPORTED_GOOGLE_GENERATIVE_AI_MODEL_IDS
        .contains(&model)
        .then(|| {
            format!(
                "model {model:?} speaks Google's own Generative AI wire format (e.g. via OpenCode \
                 Zen), which beyond has no native dialect for — set an explicit `dialect` on this \
                 model's models.json override to pick a supported wire shape instead of silently \
                 sending a mis-shapen request body"
            )
        })
}

/// Which OAuth provider (if any) a `models.json` `base_url` override for `model` should fall through to
/// when it supplies no bearer credential of its own (pi-parity remediation pass 19, Task 2). `over`'s own
/// explicit `api_key`/`auth_header`, and `key` (`--key`/`AI_AGENT_KEY`, already resolved by the caller),
/// all take priority over OAuth when set — mirrors pi's real precedence (`model-registry.ts`/
/// `auth-storage.ts`: CLI flag → stored `api_key` → OAuth → env var; `key` here already folds pi's
/// separate CLI-flag/env-var tiers into the one value this crate resolves them to). Uses the exact same
/// id heuristics [`resolve_gateway_credential_with_identity`]'s own non-override OAuth branches use
/// further below (a Claude-dialect id ⇒ Anthropic; an id containing `"codex"` ⇒ OpenAI Codex) so a
/// `base_url` override never gets a second, diverging notion of "which provider does this model belong
/// to".
///
/// GitHub Copilot is deliberately excluded, matching pi's own documented exception (its OAuth provider's
/// `modifyModels` hook unconditionally rewrites `baseUrl` for Copilot-hosted ids, so OAuth already wins
/// there by construction — untouched by this fix): Copilot's model set is dynamic, matched only via a
/// stored credential's own `available_model_ids`, which is meaningless once the request's destination has
/// *already* been redirected by an arbitrary operator-authored `base_url` — there's no way to tell whether
/// that endpoint is even reachable through Copilot's dynamically-issued proxy host at all.
///
/// Factored out from [`resolve_gateway_credential_with_identity`] so this eligibility decision is unit
/// testable directly, without touching the real stored OAuth credential file `AuthStore::open_default`
/// reads from — mirrors [`direct_route_base_and_path`]'s identical reasoning for the same
/// file-avoidance goal. Whether a credential is actually *stored* for the returned provider is a separate
/// question the caller still has to check (`AuthStore::get`) — this only decides eligibility/precedence.
fn oauth_fallback_provider(
    model: &str,
    over: &crate::settings::ModelOverride,
    key: Option<&str>,
) -> Option<OAuthProviderId> {
    if over.api_key.is_some() || over.auth_header.is_some() || key.is_some() {
        return None;
    }
    if agent_core::dialect::Dialect::for_model(model) == agent_core::dialect::Dialect::Anthropic {
        return Some(OAuthProviderId::Anthropic);
    }
    if model.contains("codex") {
        return Some(OAuthProviderId::OpenaiCodex);
    }
    None
}

/// Which provider a user-supplied `base_url` names — the `models.json`-override path, where the
/// upstream is known only by where it points.
///
/// The host table itself lives in the shared `providers` crate (one row per upstream, the same rows the
/// gateway proxies to); this is the thin parsing shim above it, kept here because `providers` is
/// deliberately dependency-free and does not take `url`. `opencode.ai` needs the path as well as the
/// host — it serves two providers that disagree on wire format — so both are passed through.
///
/// **This only ever covers a `base_url`-carrying route.** A *gateway*-routed request (a `bai_v1…`
/// virtual key, no override) genuinely has no client-visible signal of which provider will serve it: the
/// model id goes to the gateway's bare-path default route, there is no `/{provider}/…` segment, and the
/// opaque virtual key (`crates/gateway/src/key.rs`) reveals nothing. So the gateway-routed case of
/// `Together`/`Groq`/`OpenRouter` still can't be resolved from this crate — it would need the gateway to
/// echo the resolved provider back. Unchanged by the direct-routing work, which sidesteps it: a *direct*
/// route always knows its provider, because it picked it.
/// Which of the 9 known third-party aggregator platforms (`agent_core::models::AggregatorHost`) a
/// `models.json` override's (already-normalized) `base_url` names, if any (pi-parity remediation pass 19,
/// Task 3; OpenCode Zen/OpenCode-Go added in pass 20, Task 5) — the "BYO/direct-routed" half of the host
/// signal `agent_core::models::capabilities_for_route_with_host` exists to consume; see
/// `agent_core::transport::ModelRequest::host`'s own doc comment for the full mechanism this is meant to
/// close the loop on. Matched by parsed hostname (see [`is_azure_host`]'s identical reasoning for why not
/// a raw substring check), against each aggregator's real, documented base authority — pi's own
/// `huggingface.models.ts`/`nvidia.models.ts`/`kimi-coding.models.ts`/`together.models.ts`/
/// `groq.models.ts`/`openrouter.models.ts` `baseUrl`s. Fireworks is deliberately excluded:
/// `agent_core::client::GatewayClient::stream` already resolves it unconditionally from the model id's
/// own shape (`is_fireworks_model`) before this signal would ever be consulted, so there's no
/// base_url-matching case to add for it here.
///
/// OpenCode Zen and OpenCode-Go are the one pair that needs more than a hostname match: pi's
/// `opencode.models.ts` (Zen) and `opencode-go.models.ts` (Go) are two distinct aggregators nested under
/// the *same* registered domain — `opencode.ai/zen`(`/v1`) vs. `opencode.ai/zen/go`(`/v1`) — each with
/// its own catalogue and, for a handful of bare ids (`"minimax-m2.7"`, `"minimax-m3"`, `"qwen3.6-plus"`,
/// `"glm-5.1"`), a genuinely different real wire dialect/capability numbers than the identical id gets on
/// the other (see `agent_core::dialect::anthropic_wire_bare_id_for_host` and
/// `agent_core::models::capabilities_for_route_with_host`'s own `OpenCodeZen`/`OpenCodeGo` arms) — so this
/// also inspects the path's `/go` segment, the same way [`is_azure_host`] cannot for those two.
///
/// **This only ever covers a `base_url`-carrying override.** A plain gateway-routed request (a
/// `bai_v1…` virtual key, no override) genuinely has no client-visible signal of which
/// `crates/gateway::route::KNOWN_PROVIDERS` row will actually serve it: the request forwards the model id
/// verbatim to the gateway's bare-path default route (dialect-keyed, not provider-keyed —
/// `agent_core::client::GatewayClient::stream`'s `None => format!("{}{}", self.base_url,
/// dialect.endpoint_path())` branch), with no explicit `/{provider}/…` path segment, no `--provider`
/// flag anywhere in this crate, and no provider name recoverable from the opaque
/// `bai_v1.{kid}.{payload}.{sig}` virtual key (`crates/gateway/src/key.rs`) for the client to read back
/// out. So `AggregatorHost::{Together,Groq,OpenRouter}`'s *gateway*-routed case genuinely can't be
/// resolved from `crates/agent` as it exists today — it would need either a gateway-side change (echoing
/// the resolved provider back to the client) or a new client-side provider-pinning surface, neither of
/// which exists yet. Left unaddressed here rather than guessed at; this function only ever returns one of
/// those 3 variants for the (also legitimate) case where an operator's `base_url` override points
/// directly at that provider's own official endpoint, bypassing the gateway entirely — the identical
/// shape this function already handles for the 5 BYO-only hosts (HuggingFace/NVIDIA/Kimi-Coding/OpenCode
/// Zen/OpenCode-Go, none of which have a gateway route at all).
///
/// Threaded to `agent_core::client::DirectRouting::aggregator_host` (pi-parity pass 20, Task 5) by
/// [`resolve_gateway_credential_with_identity`]'s own BYO-override branch, which reads
/// `ModelRequest::host` from it via `GatewayClient::stream` — as well as this file's own
/// [`GatewayCredentialIdentity`] `DirectOverride`/`DirectOverrideOauth` variants, unchanged.
pub(crate) fn provider_for_base_url(base_url: &str) -> Option<&'static ProviderSpec> {
    let url = url::Url::parse(base_url).ok()?;
    let host = url.host_str()?.to_ascii_lowercase();
    providers::for_host(&host, url.path())
}

/// Which upstream a `base_url` names, as an [`AggregatorHost`] — [`provider_for_base_url`]'s id half,
/// which is what `DirectRouting::aggregator_host` and [`GatewayCredentialIdentity`] carry.
pub(crate) fn aggregator_host_for_base_url(base_url: &str) -> Option<AggregatorHost> {
    provider_for_base_url(base_url).map(|spec| spec.id)
}

/// A cheap, `PartialEq`-comparable summary of *which upstream* a resolved [`GatewayCredential`] talks
/// to — deliberately not the credential/token value's own identity in every case (an OAuth token's
/// bearer content can rotate independently of "is this still the same provider relationship", and
/// GitHub Copilot's own proxy host is *known* to rotate mid-session on the same stored login — see
/// `crate::oauth::github_copilot::CopilotRoutedCredentialSource`'s own doc comment), just the shape of
/// the connection: which provider, and (for a direct-routed credential) which
/// base_url/dialect/deployment/auth-header combination it resolved to.
///
/// Fix #36 (pi-parity audit, perf primitive): two calls to [`resolve_gateway_credential_with_identity`]
/// for different models that produce `==` identities are safe to keep serving from the same
/// already-built `agent_core::client::GatewayClient` (and its `reqwest::Client` connection pool)
/// instead of tearing one down and building a fresh one on every model switch — see this module's own
/// doc comment for `crate::serve::build_gateway_client`'s current unconditional-rebuild behavior. A
/// future `build_gateway_client` would need to: (1) store the `GatewayCredentialIdentity` alongside
/// the `Arc<GatewayClient>` it currently caches per-process, (2) call
/// `resolve_gateway_credential_with_identity` instead of `resolve_gateway_credential`, and (3) only
/// rebuild when the new identity differs from the stored one — reusing the existing `Arc<GatewayClient>`
/// (still re-applying `with_retry`/`with_max_backoff`/`with_idle_timeout` from the *current* `cfg`,
/// since those can change independently of the credential) otherwise. Not wired in this round — see
/// this module's own doc comment.
///
/// No `Debug` derive: [`Self::DirectOverride`]'s `bearer` field is a live, resolved API key — this type
/// implements [`std::fmt::Debug`] by hand instead, redacting only that field, matching
/// `agent_core::client::ApiKey`'s own redaction convention.
#[derive(Clone, PartialEq, Eq)]
pub enum GatewayCredentialIdentity {
    /// `GatewayCredential::Static` — a fixed `--key`/`AI_AGENT_KEY` string.
    StaticKey(String),
    /// A `models.json` `base_url` override's resolved `StaticDirectCredentialSource` — every field of
    /// the resolved route that changes what's actually sent on the wire (see
    /// `resolve_gateway_credential_with_identity`'s BYO-override branch for where each comes from).
    DirectOverride {
        base_url: String,
        path: &'static str,
        bearer: String,
        auth_header: Option<String>,
        auth_header_prefix: Option<String>,
        deployment_name: Option<String>,
        query: Option<String>,
        /// See [`aggregator_host_for_base_url`]'s own doc comment (pi-parity remediation pass 19,
        /// Task 3) — derived purely from `base_url`, so it never changes the equality this type already
        /// gets from that field; carried here just so a caller inspecting this identity doesn't have to
        /// re-derive it.
        aggregator_host: Option<AggregatorHost>,
    },
    /// A `models.json` `base_url` override for a model with an active OAuth login and no
    /// override-supplied bearer of its own (pi-parity remediation pass 19, Task 2) — the override still
    /// redirects *where* the request goes (exactly like [`Self::DirectOverride`]), but *how* it
    /// authenticates is a live, auto-refreshed OAuth token rather than a fixed string, so (unlike
    /// `DirectOverride`) there is no `bearer` field to compare or redact: two resolutions for the same
    /// override configuration are the same upstream relationship regardless of the live token's current
    /// value — the same reasoning [`Self::Anthropic`]/[`Self::OpenaiCodex`] already rely on for their own
    /// OAuth-backed variants.
    DirectOverrideOauth {
        base_url: String,
        path: &'static str,
        auth_header: Option<String>,
        auth_header_prefix: Option<String>,
        deployment_name: Option<String>,
        query: Option<String>,
        aggregator_host: Option<AggregatorHost>,
        provider: OAuthProviderId,
    },
    /// A stored Anthropic OAuth subscription login — a fixed relationship to a fixed backend, so no
    /// further distinguishing fields are needed (unlike GitHub Copilot's own dynamically-hosted proxy;
    /// see [`Self::GithubCopilot`]).
    Anthropic,
    /// A stored OpenAI Codex OAuth subscription login, distinguished by the account id the backend
    /// requires (not a secret — see `crate::oauth::openai_codex::OpenaiCodexCredential`'s own doc
    /// comment; safe to hold and compare directly here).
    OpenaiCodex { account_id: String },
    /// A stored GitHub Copilot OAuth subscription login. Deliberately excludes the resolved proxy host:
    /// it's re-derived fresh from the current access token on every request and can rotate mid-session
    /// on the exact same stored login (see `CopilotRoutedCredentialSource`'s own doc comment) — treating
    /// a host change alone as "a different upstream" would defeat the whole point of this type, forcing
    /// a client rebuild on a rotation that isn't actually a provider switch. `enterprise_url` and `path`
    /// are the two inputs that genuinely are fixed for a given stored login/model pairing.
    GithubCopilot {
        enterprise_url: Option<String>,
        path: &'static str,
    },
}

impl std::fmt::Debug for GatewayCredentialIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StaticKey(_) => f.write_str("StaticKey(***)"),
            Self::DirectOverride {
                base_url,
                path,
                auth_header,
                auth_header_prefix,
                deployment_name,
                query,
                aggregator_host,
                ..
            } => f
                .debug_struct("DirectOverride")
                .field("base_url", base_url)
                .field("path", path)
                .field("bearer", &"***")
                .field("auth_header", auth_header)
                .field("auth_header_prefix", auth_header_prefix)
                .field("deployment_name", deployment_name)
                .field("query", query)
                .field("aggregator_host", aggregator_host)
                .finish(),
            Self::DirectOverrideOauth {
                base_url,
                path,
                auth_header,
                auth_header_prefix,
                deployment_name,
                query,
                aggregator_host,
                provider,
            } => f
                .debug_struct("DirectOverrideOauth")
                .field("base_url", base_url)
                .field("path", path)
                .field("auth_header", auth_header)
                .field("auth_header_prefix", auth_header_prefix)
                .field("deployment_name", deployment_name)
                .field("query", query)
                .field("aggregator_host", aggregator_host)
                .field("provider", provider)
                .finish(),
            Self::Anthropic => f.write_str("Anthropic"),
            Self::OpenaiCodex { account_id } => f
                .debug_struct("OpenaiCodex")
                .field("account_id", account_id)
                .finish(),
            Self::GithubCopilot {
                enterprise_url,
                path,
            } => f
                .debug_struct("GithubCopilot")
                .field("enterprise_url", enterprise_url)
                .field("path", path)
                .finish(),
        }
    }
}

/// Whether `base_url` already carries a `/v1` version segment (ignoring a trailing slash) — the signal
/// [`direct_route_path`] uses to decide whether the dialect's usual `/v1`-prefixed endpoint path would
/// double up an already-present one.
fn base_url_has_v1_segment(base_url: &str) -> bool {
    base_url.trim_end_matches('/').ends_with("/v1")
}

/// Fix 3 (pi-parity, Round 2 — Task 45): the endpoint path to append to a `models.json` `base_url`
/// override's own `base_url` for `dialect`'s `RouteOverride::Direct` route — the exact bug class GitHub
/// Copilot's own routing already worked around (see
/// [`crate::oauth::github_copilot::copilot_endpoint_path`]'s doc comment): pi's official SDKs set a
/// vendor client's `baseURL` with no `/v1` of their own, then let the SDK append its own fixed *bare*
/// relative path (`/chat/completions`, `/responses`) — except the Anthropic SDK, whose default
/// `baseURL` never carries a version segment either, so it always appends its full `/v1/messages`
/// verbatim regardless of `base_url`'s own shape. [`agent_core::dialect::Dialect::endpoint_path`]'s own
/// constant is always `/v1`-prefixed (correct for beyond's *own* default gateway routing convention,
/// where `base_url` is bare), so appending it verbatim to an already-`/v1`-suffixed BYO `base_url` (a
/// natural way to configure an Azure-style or OpenAI-API-compatible endpoint, e.g.
/// `"http://host/openai/v1"`) doubled the segment into `"/v1/v1/…"` — this was the bug.
///
/// Detected from `base_url`'s own shape rather than a second `models.json` field: an operator who
/// already wrote `/v1` into `base_url` almost certainly means it as the version segment the dialect's
/// own default path would otherwise contribute a second time, so stripping it there is the one reading
/// that can't produce a broken URL either way — a `base_url` with no such suffix still gets the
/// dialect's full default path, unchanged, preserving every existing override's behavior exactly as
/// before this fix. Reuses [`crate::oauth::github_copilot::copilot_endpoint_path`]'s existing per-dialect
/// bare-path table rather than duplicating it — that function already encodes precisely which dialects
/// omit `/v1` and which don't, for the identical underlying SDK-convention reason.
fn direct_route_path(dialect: agent_core::dialect::Dialect, base_url: &str) -> &'static str {
    if base_url_has_v1_segment(base_url) {
        crate::oauth::github_copilot::copilot_endpoint_path(dialect)
    } else {
        dialect.endpoint_path()
    }
}

/// Fix 3 continued (Task 46): the `(base_url, path)` pair for a `models.json` override's
/// `RouteOverride::Direct` route, folding [`crate::settings::ModelOverride::deployment_name`] in as a
/// URL path segment when set — Azure's classic dated-`api-version` REST convention
/// (`/openai/deployments/{name}/chat/completions?api-version=…`) addresses the deployment purely by URL
/// path segment, never a `/v1` marker (that's the *other*, newer unified-API convention's own signal,
/// which addresses a deployment purely through the wire body's `"model"` field instead — already
/// handled independently, and left untouched by this fn, by
/// `agent_core::client::GatewayClient::stream`'s own `deployment_name` body substitution, Fix 2). So
/// when `deployment_name` is set, this bypasses [`direct_route_path`]'s `/v1`-detection heuristic
/// entirely rather than composing with it — there's no `/v1` for it to detect in the classic
/// convention's own path shape, and the two conventions are never meant to be mixed in one override.
///
/// The deployment name is folded into `base_url` (the one field `RouteOverride::Direct` allows to vary
/// per-override at all — its `path` is a fixed `&'static str` constant, not an interpolated owned
/// string) rather than `path`, so this composes within `agent_core::client`'s existing types without
/// needing a change there. Factored out from `resolve_gateway_credential` so both this and
/// [`direct_route_path`] are unit-testable directly, without touching the real `~/.claude/models.json`
/// file `ModelOverrides::open_default` reads from.
///
/// Fix #33 (pi-parity audit): the deployment-segment insertion above is skipped for
/// [`agent_core::dialect::Dialect::OpenAiResponses`] — verified against both the `openai` npm package's
/// `AzureOpenAI` client and Microsoft's Azure OpenAI OpenAPI spec, the Responses API never takes a
/// deployment-prefixed URL segment at all; a deployment is addressed purely through the request body's
/// `"model"` field there (already handled independently by `GatewayClient::stream`'s own substitution,
/// untouched by this fn). Folding `deployment_name` into the URL unconditionally for every dialect
/// previously produced `.../openai/deployments/{name}/responses`, a path the Responses API doesn't
/// recognize. So a Responses-dialect request with `deployment_name` set now falls through to the same
/// branch a `deployment_name`-unset request takes below — including Fix #34's Azure host
/// auto-normalization, which is exactly the URL shape Azure's Responses API actually expects.
///
/// Fix #34 (pi-parity audit): when there's no deployment segment to insert (either because
/// `deployment_name` is unset, or because it's a Responses-dialect request per Fix #33 above),
/// [`normalize_azure_base_url`] gets first look at `base_url` — an operator pasting a bare/partial Azure
/// Portal endpoint (`https://my-resource.openai.azure.com`, or that plus a stray `/openai` or
/// `/openai/v1/responses`) is auto-rewritten to Azure's canonical `/openai/v1`, mirroring pi's own
/// `normalizeAzureBaseUrl` (`azure-openai-responses.ts`). A non-Azure `base_url` (or an Azure one whose
/// path doesn't match a recognized shape — see that function's own doc comment) is returned unchanged,
/// falling through to [`direct_route_path`]'s existing `/v1`-detection heuristic exactly as before this
/// fix.
fn direct_route_base_and_path(
    base_url: &str,
    dialect: agent_core::dialect::Dialect,
    deployment_name: Option<&str>,
) -> (String, &'static str) {
    let trimmed = base_url.trim_end_matches('/');
    match deployment_name {
        Some(name) if dialect != agent_core::dialect::Dialect::OpenAiResponses => (
            format!("{trimmed}/openai/deployments/{name}"),
            crate::oauth::github_copilot::copilot_endpoint_path(dialect),
        ),
        _ => match normalize_azure_base_url(trimmed) {
            Some(normalized) => {
                let path = direct_route_path(dialect, &normalized);
                (normalized, path)
            }
            None => (trimmed.to_string(), direct_route_path(dialect, trimmed)),
        },
    }
}

/// Azure OpenAI hostnames [`normalize_azure_base_url`]/[`is_azure_host`] recognize — mirrors pi's own
/// `normalizeAzureBaseUrl` host detection (`azure-openai-responses.ts`): the classic Azure OpenAI
/// resource domain, Azure AI's newer Cognitive Services umbrella domain, and Azure AI Foundry's own
/// domain — all three are real, current hostnames Azure-hosted OpenAI deployments are reachable under.
const AZURE_HOST_SUFFIXES: &[&str] = &[
    ".openai.azure.com",
    ".cognitiveservices.azure.com",
    ".ai.azure.com",
];

/// Fix #34 (pi-parity audit): whether `base_url`'s host is a recognized Azure OpenAI hostname (see
/// [`AZURE_HOST_SUFFIXES`]) — used both by [`normalize_azure_base_url`] and, independently, by
/// [`resolve_gateway_credential_with_identity`] to gate [`azure_api_version_query`]'s Fix #35 default
/// (an unparsable `base_url`, or one with no host at all, is never treated as Azure).
fn is_azure_host(base_url: &str) -> bool {
    url::Url::parse(base_url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_string))
        .is_some_and(|host| {
            AZURE_HOST_SUFFIXES
                .iter()
                .any(|suffix| host.ends_with(suffix))
        })
}

/// Fix #34 (pi-parity audit): if `base_url`'s host is Azure-shaped (see [`is_azure_host`]) *and* its
/// path is one of the handful of shapes an operator pasting a Azure Portal endpoint actually produces —
/// bare root, `/openai`, or the full `/openai/v1/responses` (each with or without a trailing slash;
/// query strings are always dropped/rebuilt, matching the "with/without trailing query" cases pi's own
/// `normalizeAzureBaseUrl` also collapses) — rewrite it to Azure's canonical unified `/openai/v1`.
/// Returns `None` for a non-Azure host, an unparsable `base_url`, or an Azure host whose path doesn't
/// match one of those recognized shapes (e.g. the classic per-deployment path `direct_route_base_and_path`
/// already builds itself, or some genuinely custom proxy path an operator set up on purpose) — in every
/// `None` case the caller falls through to its own existing behavior, unchanged.
fn normalize_azure_base_url(base_url: &str) -> Option<String> {
    let url = url::Url::parse(base_url).ok()?;
    let host = url.host_str()?;
    if !AZURE_HOST_SUFFIXES
        .iter()
        .any(|suffix| host.ends_with(suffix))
    {
        return None;
    }
    let path = url.path().trim_end_matches('/');
    if !matches!(path, "" | "/openai" | "/openai/v1" | "/openai/v1/responses") {
        return None;
    }
    let authority = match url.port() {
        Some(port) => format!("{host}:{port}"),
        None => host.to_string(),
    };
    Some(format!("{}://{authority}/openai/v1", url.scheme()))
}

/// Default `api-version` (Fix #35, pi-parity audit) sent whenever a request is genuinely Azure-routed
/// (see [`is_azure_host`]/`ModelOverride::deployment_name`) and the operator left
/// [`crate::settings::ModelOverride::api_version`] unset — pi's own `DEFAULT_AZURE_API_VERSION`
/// (`azure-openai-responses.ts`). Harmless for the recommended unified `/openai/v1` endpoint (Fix #34
/// routes most bare Azure configs there by default, and that surface doesn't require `api-version` at
/// all) — better than silently sending no query string at all for an operator on the classic
/// dated-`api-version` endpoint (via `deployment_name`) who forgot to set this field explicitly.
const DEFAULT_AZURE_API_VERSION: &str = "v1";

/// Build the `api-version=…` query string from a `models.json` override's [`ModelOverride::api_version`]
/// field (Fix 2, pi-parity Round 2 — Azure OpenAI's dated REST `api-version`), or `None` if unset/empty
/// and `is_azure` is `false`. `is_azure` (Fix #35, pi-parity audit) gates [`DEFAULT_AZURE_API_VERSION`]'s
/// default: `true` for a `deployment_name`-carrying override or one whose `base_url` is a recognized
/// Azure host (see [`is_azure_host`]) — a plain non-Azure BYO override (Ollama, Kimi-Coding, Cloudflare
/// AI Gateway, …) must never pick up an unsolicited `?api-version=v1` query string it never asked for.
/// Percent-encoded via [`url::form_urlencoded`] — the same general-purpose query-param encoder any other
/// query value would go through, rather than a hand-rolled `format!("api-version={v}")` that would
/// silently misbuild the URL if an operator's value ever contained a character needing escaping.
fn azure_api_version_query(api_version: Option<&str>, is_azure: bool) -> Option<String> {
    let version = match api_version {
        Some(v) if !v.is_empty() => v,
        _ if is_azure => DEFAULT_AZURE_API_VERSION,
        _ => return None,
    };
    Some(
        url::form_urlencoded::Serializer::new(String::new())
            .append_pair("api-version", version)
            .finish(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn azure_api_version_query_builds_the_expected_query_string() {
        assert_eq!(
            azure_api_version_query(Some("2024-08-01-preview"), true),
            Some("api-version=2024-08-01-preview".to_string())
        );
        // An explicit value wins even for a non-Azure route — `is_azure` only gates the *default*.
        assert_eq!(
            azure_api_version_query(Some("2024-08-01-preview"), false),
            Some("api-version=2024-08-01-preview".to_string())
        );
    }

    #[test]
    fn azure_api_version_query_percent_encodes_special_characters() {
        // A defensive check, not a realistic input: Azure's own api-version strings never carry a
        // space, but the encoder must still do the right thing rather than build a broken URL.
        assert_eq!(
            azure_api_version_query(Some("2024 08 01"), true),
            Some("api-version=2024+08+01".to_string())
        );
    }

    #[test]
    fn azure_api_version_query_is_none_when_unset_or_empty_and_not_azure() {
        assert_eq!(azure_api_version_query(None, false), None);
        assert_eq!(azure_api_version_query(Some(""), false), None);
    }

    // Fix #35 (pi-parity audit): an Azure-routed request whose `api_version` was left unset defaults
    // to pi's own `DEFAULT_AZURE_API_VERSION` ("v1") rather than sending no query string at all.

    #[test]
    fn azure_api_version_query_defaults_to_v1_when_azure_and_unset() {
        assert_eq!(
            azure_api_version_query(None, true),
            Some("api-version=v1".to_string())
        );
    }

    #[test]
    fn azure_api_version_query_defaults_to_v1_when_azure_and_empty() {
        assert_eq!(
            azure_api_version_query(Some(""), true),
            Some("api-version=v1".to_string())
        );
    }

    #[test]
    fn azure_api_version_query_never_defaults_for_a_non_azure_route() {
        // A plain BYO override (Ollama, Kimi-Coding, Cloudflare AI Gateway, ...) must never pick up an
        // unsolicited `?api-version=v1` query string it never asked for.
        assert_eq!(azure_api_version_query(None, false), None);
    }

    // Fix #34 (pi-parity audit): `normalize_azure_base_url`/`is_azure_host` auto-detect an Azure OpenAI
    // host and rewrite any of the shapes an operator pasting the bare Azure Portal endpoint would
    // actually produce to Azure's canonical unified `/openai/v1`.

    #[test]
    fn is_azure_host_recognizes_all_three_documented_azure_domains() {
        assert!(is_azure_host("https://my-resource.openai.azure.com"));
        assert!(is_azure_host(
            "https://my-resource.cognitiveservices.azure.com"
        ));
        assert!(is_azure_host("https://my-resource.ai.azure.com"));
    }

    #[test]
    fn is_azure_host_is_false_for_a_non_azure_host_or_an_unparsable_url() {
        assert!(!is_azure_host("https://api.openai.com"));
        assert!(!is_azure_host("not a url at all"));
    }

    #[test]
    fn normalize_azure_base_url_rewrites_a_bare_azure_root_to_the_canonical_v1_path() {
        assert_eq!(
            normalize_azure_base_url("https://my-resource.openai.azure.com"),
            Some("https://my-resource.openai.azure.com/openai/v1".to_string())
        );
    }

    #[test]
    fn normalize_azure_base_url_rewrites_a_bare_openai_segment_to_the_canonical_v1_path() {
        assert_eq!(
            normalize_azure_base_url("https://my-resource.openai.azure.com/openai"),
            Some("https://my-resource.openai.azure.com/openai/v1".to_string())
        );
    }

    #[test]
    fn normalize_azure_base_url_rewrites_the_full_v1_responses_shape_dropping_the_bare_path_suffix()
    {
        assert_eq!(
            normalize_azure_base_url("https://my-resource.openai.azure.com/openai/v1/responses"),
            Some("https://my-resource.openai.azure.com/openai/v1".to_string())
        );
    }

    #[test]
    fn normalize_azure_base_url_ignores_a_trailing_slash_and_a_query_string() {
        assert_eq!(
            normalize_azure_base_url(
                "https://my-resource.openai.azure.com/openai/v1/responses/?api-version=v1"
            ),
            Some("https://my-resource.openai.azure.com/openai/v1".to_string())
        );
    }

    #[test]
    fn normalize_azure_base_url_recognizes_the_cognitiveservices_and_ai_azure_domains_too() {
        assert_eq!(
            normalize_azure_base_url("https://my-resource.cognitiveservices.azure.com"),
            Some("https://my-resource.cognitiveservices.azure.com/openai/v1".to_string())
        );
        assert_eq!(
            normalize_azure_base_url("https://my-resource.ai.azure.com/openai"),
            Some("https://my-resource.ai.azure.com/openai/v1".to_string())
        );
    }

    #[test]
    fn normalize_azure_base_url_returns_none_for_a_non_azure_host() {
        assert_eq!(
            normalize_azure_base_url("https://my-ollama-box.example.com"),
            None
        );
    }

    #[test]
    fn normalize_azure_base_url_returns_none_for_an_azure_host_with_an_unrecognized_path() {
        // The classic per-deployment path shape (`direct_route_base_and_path` builds this one itself)
        // and any other genuinely custom proxy path an operator set up on purpose must not be rewritten.
        assert_eq!(
            normalize_azure_base_url(
                "https://my-resource.openai.azure.com/openai/deployments/my-deployment"
            ),
            None
        );
    }

    // Task 45: `direct_route_path`/`direct_route_base_and_path` fix the `/v1`-doubling bug for a
    // `models.json` BYO `base_url` pointed at an OpenAI-wire dialect.

    #[test]
    fn direct_route_path_strips_v1_when_base_url_already_ends_with_it_for_openai_wire_dialects() {
        use agent_core::dialect::Dialect;
        assert_eq!(
            direct_route_path(Dialect::OpenAiResponses, "http://host/openai/v1"),
            "/responses"
        );
        assert_eq!(
            direct_route_path(Dialect::OpenAi, "http://host/v1/"),
            "/chat/completions",
            "a trailing slash on base_url must not defeat the /v1 detection"
        );
    }

    #[test]
    fn direct_route_path_keeps_the_v1_prefixed_default_when_base_url_has_no_v1_segment() {
        use agent_core::dialect::Dialect;
        assert_eq!(
            direct_route_path(Dialect::OpenAiResponses, "http://host"),
            "/v1/responses"
        );
        assert_eq!(
            direct_route_path(Dialect::OpenAi, "http://host/openai"),
            "/v1/chat/completions"
        );
    }

    #[test]
    fn direct_route_path_leaves_the_anthropic_wire_case_unaffected_either_way() {
        // The Anthropic SDK's own baseURL convention carries no version segment of its own, so it
        // always appends its full `/v1/messages` verbatim — unlike the OpenAI-wire dialects above,
        // whether or not base_url itself already ends in `/v1` must never change this.
        use agent_core::dialect::Dialect;
        assert_eq!(
            direct_route_path(Dialect::Anthropic, "http://host/v1"),
            "/v1/messages"
        );
        assert_eq!(
            direct_route_path(Dialect::Anthropic, "http://host"),
            "/v1/messages"
        );
    }

    #[test]
    fn base_url_ending_in_v1_no_longer_doubles_the_segment_for_an_openai_wire_dialect() {
        // The exact bug empirically confirmed for Task 45: an Azure-style BYO `base_url` that already
        // carries `/v1` (a natural way to configure such an endpoint) previously produced a doubled
        // "/v1/v1/responses" when routed.
        use agent_core::dialect::Dialect;
        let (base_url, path) =
            direct_route_base_and_path("http://host/openai/v1", Dialect::OpenAiResponses, None);
        assert_eq!(
            format!("{base_url}{path}"),
            "http://host/openai/v1/responses"
        );
    }

    #[test]
    fn base_url_without_v1_still_gets_the_full_default_path_unchanged() {
        use agent_core::dialect::Dialect;
        let (base_url, path) =
            direct_route_base_and_path("http://host", Dialect::OpenAiResponses, None);
        assert_eq!(format!("{base_url}{path}"), "http://host/v1/responses");
    }

    // Task 46: `deployment_name` becomes a URL path segment (Azure's classic dated-`api-version` REST
    // convention), composing cleanly with Task 45's fix rather than fighting it.

    #[test]
    fn deployment_name_inserts_a_url_path_segment_for_chat_completions_and_composes_with_api_version_query()
     {
        use agent_core::dialect::Dialect;
        let (base_url, path) = direct_route_base_and_path(
            "https://my-resource.openai.azure.com",
            Dialect::OpenAi,
            Some("my-deployment"),
        );
        let query = azure_api_version_query(Some("2024-08-01-preview"), true).unwrap();
        let url = format!("{base_url}{path}?{query}");
        assert_eq!(
            url,
            "https://my-resource.openai.azure.com/openai/deployments/my-deployment/chat/completions\
             ?api-version=2024-08-01-preview"
        );
    }

    // Fix #33 (pi-parity audit): the Responses API never takes a deployment-prefixed URL segment at
    // all — a deployment is addressed purely through the request body's own `"model"` field there
    // (`agent_core::client::GatewayClient::stream`'s independent `deployment_name` substitution). This
    // used to be the exact regression: `deployment_name` set + `Dialect::OpenAiResponses` previously
    // produced `.../openai/deployments/{name}/responses`, a path the Responses API doesn't recognize.

    #[test]
    fn deployment_name_is_not_folded_into_the_url_for_the_responses_dialect() {
        use agent_core::dialect::Dialect;
        let (base_url, path) = direct_route_base_and_path(
            "https://my-resource.openai.azure.com",
            Dialect::OpenAiResponses,
            Some("my-deployment"),
        );
        assert!(
            !base_url.contains("deployments"),
            "the Responses dialect must never get a /openai/deployments/{{name}} URL segment, got: \
             {base_url}"
        );
        // Falls through to Fix #34's Azure host auto-normalization instead, since this base_url is a
        // bare Azure root with no deployment-specific path to preserve.
        assert_eq!(
            format!("{base_url}{path}"),
            "https://my-resource.openai.azure.com/openai/v1/responses"
        );
    }

    #[test]
    fn deployment_name_and_responses_dialect_together_compose_with_the_api_version_default() {
        use agent_core::dialect::Dialect;
        let (base_url, path) = direct_route_base_and_path(
            "https://my-resource.openai.azure.com",
            Dialect::OpenAiResponses,
            Some("my-deployment"),
        );
        // The deployment is still addressed — just via the body's "model" field (Fix 2, handled
        // independently by `GatewayClient::stream`), not this URL — so `is_azure` must still be `true`
        // for the api-version default (Fix #35) to apply, exactly as if a bare Azure host were used
        // with no `deployment_name` at all.
        let query = azure_api_version_query(None, true).unwrap();
        let url = format!("{base_url}{path}?{query}");
        assert_eq!(
            url,
            "https://my-resource.openai.azure.com/openai/v1/responses?api-version=v1"
        );
    }

    #[test]
    fn deployment_name_bypasses_the_v1_stripping_heuristic_entirely() {
        // Even when `base_url` happens to end in `/v1`, the classic deployment convention's own path
        // shape never carries a `/v1` segment at all — this must not try to also strip/detect one.
        use agent_core::dialect::Dialect;
        let (base_url, path) = direct_route_base_and_path(
            "https://my-resource.openai.azure.com/v1",
            Dialect::OpenAi,
            Some("gpt4"),
        );
        assert_eq!(
            format!("{base_url}{path}"),
            "https://my-resource.openai.azure.com/v1/openai/deployments/gpt4/chat/completions"
        );
    }

    #[test]
    fn deployment_name_trims_a_trailing_slash_on_base_url_before_inserting_the_segment() {
        use agent_core::dialect::Dialect;
        let (base_url, path) = direct_route_base_and_path(
            "https://my-resource.openai.azure.com/",
            Dialect::OpenAi,
            Some("gpt4"),
        );
        assert_eq!(
            format!("{base_url}{path}"),
            "https://my-resource.openai.azure.com/openai/deployments/gpt4/chat/completions"
        );
    }

    // Fix #39 (pi-parity audit, judgment call): OpenCode Zen's 3 Gemini-hosted ids speak a wire format
    // this crate has no dialect for — `unsupported_wire_format_reason` is the narrow, opt-out-respecting
    // guard `resolve_gateway_credential_with_identity` consults before silently mis-routing one.

    #[test]
    fn unsupported_wire_format_reason_flags_all_three_known_opencode_zen_gemini_ids() {
        for model in ["gemini-3-flash", "gemini-3.1-pro", "gemini-3.5-flash"] {
            assert!(
                unsupported_wire_format_reason(model).is_some(),
                "expected {model:?} to be flagged as an unsupported wire format"
            );
        }
    }

    #[test]
    fn unsupported_wire_format_reason_is_none_for_an_ordinary_model_id() {
        assert_eq!(unsupported_wire_format_reason("gpt-5"), None);
        assert_eq!(unsupported_wire_format_reason("claude-opus-4-8"), None);
        // Not an exact match for any of the 3 known ids — must not false-positive on a substring.
        assert_eq!(
            unsupported_wire_format_reason("gemini-3-flash-preview"),
            None
        );
    }

    #[test]
    fn unsupported_wire_format_reason_message_names_the_model_and_suggests_the_escape_hatch() {
        let reason = unsupported_wire_format_reason("gemini-3-flash").unwrap();
        assert!(reason.contains("gemini-3-flash"), "got: {reason}");
        assert!(reason.contains("dialect"), "got: {reason}");
    }

    // pi-parity pass 20, Task 5/6: `aggregator_host_for_base_url` (host detection) +
    // `agent_core::dialect::Dialect::for_model_with_host` (host-aware dialect resolution) together
    // replace remediation pass 19 Task 1's own bespoke `opencode_host`/`opencode_dialect_override`
    // helpers — the exact same 3 real id/host collisions, now resolved through the one shared
    // `AggregatorHost` mechanism instead of a second, parallel "which host is this" check.

    #[test]
    fn aggregator_host_for_base_url_recognizes_opencode_zen_with_or_without_the_v1_suffix() {
        for url in [
            "https://opencode.ai/zen",
            "https://opencode.ai/zen/v1",
            "https://opencode.ai/zen/",
        ] {
            assert_eq!(
                aggregator_host_for_base_url(url),
                Some(AggregatorHost::OpenCodeZen),
                "{url}"
            );
        }
    }

    #[test]
    fn aggregator_host_for_base_url_recognizes_opencode_go_with_or_without_the_v1_suffix() {
        for url in [
            "https://opencode.ai/zen/go",
            "https://opencode.ai/zen/go/v1",
            "https://opencode.ai/zen/go/",
        ] {
            assert_eq!(
                aggregator_host_for_base_url(url),
                Some(AggregatorHost::OpenCodeGo),
                "{url}"
            );
        }
    }

    #[test]
    fn aggregator_host_for_base_url_is_none_for_an_opencode_ai_unrelated_path() {
        // Same registered domain as Zen/Go, but neither the `/zen` nor `/zen/go` prefix — must not
        // false-positive into either variant.
        assert_eq!(
            aggregator_host_for_base_url("https://opencode.ai/other"),
            None
        );
    }

    #[test]
    fn host_aware_dialect_resolves_the_three_real_opencode_collisions_from_a_byo_base_url() {
        use agent_core::dialect::Dialect;
        // The exact 3 real, host-dependent bare-id wire collisions (pi-parity pass 20, Task 6) — each
        // resolved here via the same `aggregator_host_for_base_url` → `Dialect::for_model_with_host`
        // path a real BYO `models.json` override with no explicit `dialect` would take.
        // "minimax-m2.7": openai-completions on both OpenCode Zen and OpenCode-Go (only native MiniMax,
        // no aggregator host at all, is genuinely anthropic-wire).
        assert_eq!(
            Dialect::for_model_with_host(
                "minimax-m2.7",
                aggregator_host_for_base_url("https://opencode.ai/zen/v1")
            ),
            Dialect::OpenAi
        );
        assert_eq!(
            Dialect::for_model_with_host(
                "minimax-m2.7",
                aggregator_host_for_base_url("https://opencode.ai/zen/go/v1")
            ),
            Dialect::OpenAi
        );
        // "minimax-m3": openai-completions on OpenCode Zen specifically; still anthropic-wire on
        // OpenCode-Go (matching the host-agnostic default).
        assert_eq!(
            Dialect::for_model_with_host(
                "minimax-m3",
                aggregator_host_for_base_url("https://opencode.ai/zen/v1")
            ),
            Dialect::OpenAi
        );
        assert_eq!(
            Dialect::for_model_with_host(
                "minimax-m3",
                aggregator_host_for_base_url("https://opencode.ai/zen/go/v1")
            ),
            Dialect::Anthropic
        );
        // "qwen3.6-plus": openai-completions on OpenCode-Go specifically; still anthropic-wire on
        // OpenCode Zen (matching the host-agnostic default).
        assert_eq!(
            Dialect::for_model_with_host(
                "qwen3.6-plus",
                aggregator_host_for_base_url("https://opencode.ai/zen/go/v1")
            ),
            Dialect::OpenAi
        );
        assert_eq!(
            Dialect::for_model_with_host(
                "qwen3.6-plus",
                aggregator_host_for_base_url("https://opencode.ai/zen/v1")
            ),
            Dialect::Anthropic
        );
        // Ids that are already correctly Anthropic-wire on their own host stay that way too.
        assert_eq!(
            Dialect::for_model_with_host(
                "qwen3.7-max",
                aggregator_host_for_base_url("https://opencode.ai/zen/go")
            ),
            Dialect::Anthropic
        );
        // An unrelated host (or unresolvable `base_url`) is a no-op: falls back to the same
        // host-agnostic default `Dialect::for_model` already gave.
        assert_eq!(
            Dialect::for_model_with_host(
                "minimax-m3",
                aggregator_host_for_base_url("https://api.together.ai/v1")
            ),
            Dialect::Anthropic
        );
    }

    // pi-parity remediation pass 19, Task 2: `oauth_fallback_provider` decides whether a `models.json`
    // `base_url` override with no bearer of its own should still authenticate via a stored OAuth login.

    #[test]
    fn oauth_fallback_provider_picks_anthropic_for_a_claude_dialect_model_with_no_credential_of_its_own()
     {
        let over = crate::settings::ModelOverride::default();
        assert_eq!(
            oauth_fallback_provider("claude-opus-4-8", &over, None),
            Some(OAuthProviderId::Anthropic)
        );
    }

    #[test]
    fn oauth_fallback_provider_picks_codex_for_a_codex_named_model() {
        let over = crate::settings::ModelOverride::default();
        assert_eq!(
            oauth_fallback_provider("gpt-5.1-codex", &over, None),
            Some(OAuthProviderId::OpenaiCodex)
        );
    }

    #[test]
    fn oauth_fallback_provider_is_none_when_the_override_has_its_own_api_key() {
        let over = crate::settings::ModelOverride {
            api_key: Some("sk-my-own-key".to_string()),
            ..Default::default()
        };
        assert_eq!(
            oauth_fallback_provider("claude-opus-4-8", &over, None),
            None
        );
    }

    #[test]
    fn oauth_fallback_provider_is_none_when_the_override_has_its_own_auth_header() {
        let over = crate::settings::ModelOverride {
            auth_header: Some("api-key".to_string()),
            ..Default::default()
        };
        assert_eq!(oauth_fallback_provider("gpt-5.1-codex", &over, None), None);
    }

    #[test]
    fn oauth_fallback_provider_is_none_when_a_key_flag_or_env_var_was_given() {
        let over = crate::settings::ModelOverride::default();
        assert_eq!(
            oauth_fallback_provider("claude-opus-4-8", &over, Some("bai_v1_abc")),
            None
        );
    }

    #[test]
    fn oauth_fallback_provider_is_none_for_a_third_party_model_that_is_neither_anthropic_nor_codex()
    {
        let over = crate::settings::ModelOverride::default();
        assert_eq!(oauth_fallback_provider("llama-3.1-70b", &over, None), None);
    }

    // pi-parity remediation pass 19, Task 3: `aggregator_host_for_base_url` recognizes a `models.json`
    // override's `base_url` naming a known third-party aggregator's real host.

    #[test]
    fn aggregator_host_for_base_url_recognizes_the_byo_only_hosts() {
        assert_eq!(
            aggregator_host_for_base_url("https://router.huggingface.co/v1"),
            Some(AggregatorHost::HuggingFace)
        );
        assert_eq!(
            aggregator_host_for_base_url("https://integrate.api.nvidia.com/v1"),
            Some(AggregatorHost::Nvidia)
        );
        assert_eq!(
            aggregator_host_for_base_url("https://api.kimi.com/coding"),
            Some(AggregatorHost::KimiCoding)
        );
    }

    #[test]
    fn aggregator_host_for_base_url_recognizes_a_direct_routed_gateway_native_provider() {
        // A `models.json` override can also point directly at a gateway-native provider's own official
        // endpoint (bypassing the gateway entirely) — this is the one shape this file has any
        // visibility into for those hosts at all (see this function's own doc comment for why the
        // plain gateway-routed case can't be resolved here).
        assert_eq!(
            aggregator_host_for_base_url("https://api.together.ai/v1"),
            Some(AggregatorHost::Together)
        );
        assert_eq!(
            aggregator_host_for_base_url("https://api.together.xyz/v1"),
            Some(AggregatorHost::Together)
        );
        assert_eq!(
            aggregator_host_for_base_url("https://api.groq.com/openai/v1"),
            Some(AggregatorHost::Groq)
        );
        assert_eq!(
            aggregator_host_for_base_url("https://openrouter.ai/api/v1"),
            Some(AggregatorHost::OpenRouter)
        );
    }

    #[test]
    fn aggregator_host_for_base_url_is_none_for_an_unrelated_host_or_an_unparsable_url() {
        assert_eq!(
            aggregator_host_for_base_url("https://my-ollama-box.example.com"),
            None
        );
        assert_eq!(aggregator_host_for_base_url("not a url at all"), None);
    }

    // Fix #36 (pi-parity audit): `GatewayCredentialIdentity` compares equal for two credentials that
    // resolve to the same upstream/config, and unequal for ones that don't — the property a future
    // `build_gateway_client` skip-rebuild check would rely on.

    #[test]
    fn identity_static_key_is_equal_for_the_same_key_and_differs_for_a_different_one() {
        let a = GatewayCredentialIdentity::StaticKey("bai_v1_abc".to_string());
        let b = GatewayCredentialIdentity::StaticKey("bai_v1_abc".to_string());
        let c = GatewayCredentialIdentity::StaticKey("bai_v1_xyz".to_string());
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn identity_direct_override_is_equal_only_when_every_distinguishing_field_matches() {
        let make =
            |base_url: &str, deployment: Option<&str>| GatewayCredentialIdentity::DirectOverride {
                base_url: base_url.to_string(),
                path: "/chat/completions",
                bearer: "key-1".to_string(),
                auth_header: None,
                auth_header_prefix: None,
                deployment_name: deployment.map(str::to_string),
                query: None,
                aggregator_host: None,
            };
        assert_eq!(
            make("https://host/openai/v1", None),
            make("https://host/openai/v1", None)
        );
        assert_ne!(
            make("https://host-a/openai/v1", None),
            make("https://host-b/openai/v1", None),
            "a different base_url must be a different identity"
        );
        assert_ne!(
            make("https://host/openai/v1", Some("dep-a")),
            make("https://host/openai/v1", Some("dep-b")),
            "a different deployment_name must be a different identity"
        );
    }

    #[test]
    fn identity_direct_override_differs_from_a_static_key_even_with_the_same_bearer_value() {
        let direct = GatewayCredentialIdentity::DirectOverride {
            base_url: "https://host".to_string(),
            path: "/chat/completions",
            bearer: "same-value".to_string(),
            auth_header: None,
            auth_header_prefix: None,
            deployment_name: None,
            query: None,
            aggregator_host: None,
        };
        let static_key = GatewayCredentialIdentity::StaticKey("same-value".to_string());
        assert_ne!(direct, static_key);
    }

    #[test]
    fn identity_anthropic_is_a_singleton_equal_to_itself_and_unequal_to_every_other_variant() {
        assert_eq!(
            GatewayCredentialIdentity::Anthropic,
            GatewayCredentialIdentity::Anthropic
        );
        assert_ne!(
            GatewayCredentialIdentity::Anthropic,
            GatewayCredentialIdentity::StaticKey("anthropic".to_string())
        );
    }

    #[test]
    fn identity_openai_codex_differs_by_account_id() {
        let a = GatewayCredentialIdentity::OpenaiCodex {
            account_id: "acct_1".to_string(),
        };
        let b = GatewayCredentialIdentity::OpenaiCodex {
            account_id: "acct_1".to_string(),
        };
        let c = GatewayCredentialIdentity::OpenaiCodex {
            account_id: "acct_2".to_string(),
        };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn identity_github_copilot_differs_by_enterprise_url_and_path_but_not_by_anything_else() {
        let a = GatewayCredentialIdentity::GithubCopilot {
            enterprise_url: Some("company.ghe.com".to_string()),
            path: "/chat/completions",
        };
        let same_as_a = GatewayCredentialIdentity::GithubCopilot {
            enterprise_url: Some("company.ghe.com".to_string()),
            path: "/chat/completions",
        };
        let different_enterprise = GatewayCredentialIdentity::GithubCopilot {
            enterprise_url: Some("other.ghe.com".to_string()),
            path: "/chat/completions",
        };
        let different_path = GatewayCredentialIdentity::GithubCopilot {
            enterprise_url: Some("company.ghe.com".to_string()),
            path: "/responses",
        };
        assert_eq!(a, same_as_a);
        assert_ne!(a, different_enterprise);
        assert_ne!(a, different_path);
    }

    #[test]
    fn identity_debug_never_prints_the_raw_bearer_value() {
        let direct = GatewayCredentialIdentity::DirectOverride {
            base_url: "https://host".to_string(),
            path: "/chat/completions",
            bearer: "super-secret-value".to_string(),
            auth_header: None,
            auth_header_prefix: None,
            deployment_name: None,
            query: None,
            aggregator_host: None,
        };
        let debug = format!("{direct:?}");
        assert!(
            !debug.contains("super-secret-value"),
            "Debug must redact the bearer value, got: {debug}"
        );

        let static_key = GatewayCredentialIdentity::StaticKey("another-secret".to_string());
        let debug = format!("{static_key:?}");
        assert!(
            !debug.contains("another-secret"),
            "Debug must redact the static key value, got: {debug}"
        );
    }

    // pi-parity remediation pass 19, Task 2: `GatewayCredentialIdentity::DirectOverrideOauth` — an
    // override with no bearer of its own, falling through to a stored OAuth login.

    #[test]
    fn identity_direct_override_oauth_is_equal_only_when_every_distinguishing_field_matches() {
        let make = |base_url: &str, provider: OAuthProviderId| {
            GatewayCredentialIdentity::DirectOverrideOauth {
                base_url: base_url.to_string(),
                path: "/v1/messages",
                auth_header: None,
                auth_header_prefix: None,
                deployment_name: None,
                query: None,
                aggregator_host: None,
                provider,
            }
        };
        assert_eq!(
            make("https://host", OAuthProviderId::Anthropic),
            make("https://host", OAuthProviderId::Anthropic)
        );
        assert_ne!(
            make("https://host-a", OAuthProviderId::Anthropic),
            make("https://host-b", OAuthProviderId::Anthropic),
            "a different base_url must be a different identity"
        );
        assert_ne!(
            make("https://host", OAuthProviderId::Anthropic),
            make("https://host", OAuthProviderId::OpenaiCodex),
            "a different provider must be a different identity even with the same base_url"
        );
    }

    #[test]
    fn identity_direct_override_oauth_differs_from_a_static_bearer_direct_override() {
        // The two variants exist precisely because a live OAuth token isn't comparable the same way a
        // fixed bearer string is — they must never compare equal to each other even when every other
        // field lines up.
        let oauth = GatewayCredentialIdentity::DirectOverrideOauth {
            base_url: "https://host".to_string(),
            path: "/v1/messages",
            auth_header: None,
            auth_header_prefix: None,
            deployment_name: None,
            query: None,
            aggregator_host: None,
            provider: OAuthProviderId::Anthropic,
        };
        let static_bearer = GatewayCredentialIdentity::DirectOverride {
            base_url: "https://host".to_string(),
            path: "/v1/messages",
            bearer: String::new(),
            auth_header: None,
            auth_header_prefix: None,
            deployment_name: None,
            query: None,
            aggregator_host: None,
        };
        assert_ne!(oauth, static_bearer);
    }

    #[test]
    fn identity_direct_override_oauth_debug_shows_the_provider_and_never_panics() {
        let oauth = GatewayCredentialIdentity::DirectOverrideOauth {
            base_url: "https://host".to_string(),
            path: "/v1/messages",
            auth_header: None,
            auth_header_prefix: None,
            deployment_name: None,
            query: None,
            aggregator_host: None,
            provider: OAuthProviderId::OpenaiCodex,
        };
        let debug = format!("{oauth:?}");
        assert!(debug.contains("OpenaiCodex"), "got: {debug}");
    }

    #[test]
    fn identity_direct_override_carries_the_aggregator_host_it_was_built_with() {
        let direct = GatewayCredentialIdentity::DirectOverride {
            base_url: "https://api.together.ai/v1".to_string(),
            path: "/chat/completions",
            bearer: "key".to_string(),
            auth_header: None,
            auth_header_prefix: None,
            deployment_name: None,
            query: None,
            aggregator_host: Some(AggregatorHost::Together),
        };
        let debug = format!("{direct:?}");
        assert!(debug.contains("Together"), "got: {debug}");
    }

    // ---- Direct provider routing -----------------------------------------------------------------
    //
    // The whole precedence ladder, driven through `resolve_with` so every input — process env, stored
    // OAuth, `models.json` — is injected rather than read from the machine the test happens to run on.
    // No `std::env::set_var` anywhere: these are ordinary parallel-safe unit tests.

    /// An empty stored-credential file and an empty `models.json` — the ordinary "developer with a key
    /// in their shell and nothing else" starting state.
    fn no_stored_creds() -> crate::auth_store::AuthStore {
        crate::auth_store::AuthStore::open(std::path::PathBuf::from(
            "/nonexistent/beyond-ai-test/auth.json",
        ))
    }
    fn no_overrides() -> crate::settings::ModelOverrides {
        crate::settings::ModelOverrides::open(std::path::PathBuf::from(
            "/nonexistent/beyond-ai-test/models.json",
        ))
    }

    fn resolve(
        key: Option<&str>,
        model: &str,
        env: &ProviderEnv,
    ) -> Result<GatewayCredentialIdentity, String> {
        resolve_with(
            key.map(str::to_string),
            model,
            env,
            &no_stored_creds(),
            std::path::PathBuf::from("/nonexistent/beyond-ai-test/auth.json"),
            &no_overrides(),
        )
        .map(|(_credential, identity)| identity)
    }

    /// Pull the wire-level facts out of a resolved identity: where the request goes, and how it
    /// authenticates. These four are the acceptance criteria.
    fn wire(identity: &GatewayCredentialIdentity) -> (&str, &str, Option<&str>, Option<&str>) {
        match identity {
            GatewayCredentialIdentity::DirectOverride {
                base_url,
                path,
                auth_header,
                auth_header_prefix,
                ..
            }
            | GatewayCredentialIdentity::DirectOverrideOauth {
                base_url,
                path,
                auth_header,
                auth_header_prefix,
                ..
            } => (
                base_url,
                path,
                auth_header.as_deref(),
                auth_header_prefix.as_deref(),
            ),
            other => panic!("expected a direct route, got {other:?}"),
        }
    }

    /// THE acceptance criterion: no gateway, `ANTHROPIC_API_KEY` exported, a Claude model — the request
    /// goes to Anthropic's own endpoint with an `x-api-key` header carrying the bare key. Nobody
    /// configured the header; it fell out of the provider row.
    #[test]
    fn anthropic_key_alone_routes_direct_to_anthropic_with_x_api_key() {
        let env = ProviderEnv::from_vars(&[("ANTHROPIC_API_KEY", "sk-ant-test")], false);
        let identity = resolve(None, "claude-opus-4-8", &env).expect("must resolve");
        assert_eq!(
            wire(&identity),
            (
                "https://api.anthropic.com",
                "/v1/messages",
                Some("x-api-key"),
                None
            ),
        );
        let GatewayCredentialIdentity::DirectOverride {
            bearer,
            aggregator_host,
            ..
        } = &identity
        else {
            panic!("expected a static direct route")
        };
        assert_eq!(bearer, "sk-ant-test");
        assert_eq!(*aggregator_host, Some(AggregatorHost::Anthropic));
    }

    /// Same for OpenAI — and note it lands on `/responses`, not `/chat/completions`: the provider row
    /// says OpenAI-wire, the *model* says Responses. Both facts are needed and neither is guessed.
    #[test]
    fn openai_key_alone_routes_direct_to_openai_with_bearer() {
        let env = ProviderEnv::from_vars(&[("OPENAI_API_KEY", "sk-test")], false);
        let identity = resolve(None, "gpt-5", &env).expect("must resolve");
        assert_eq!(
            wire(&identity),
            // `auth_header: None` is `GatewayClient`'s own `Authorization: Bearer` default.
            ("https://api.openai.com/v1", "/responses", None, None),
        );
    }

    #[test]
    fn a_chat_completions_model_on_the_same_openai_wire_row_takes_the_other_path() {
        let env = ProviderEnv::from_vars(&[("DEEPSEEK_API_KEY", "sk-ds")], false);
        let identity = resolve(None, "deepseek-v3", &env).expect("must resolve");
        assert_eq!(
            wire(&identity),
            (
                "https://api.deepseek.com/v1",
                "/chat/completions",
                None,
                None
            ),
        );
    }

    /// OpenRouter is an aggregator: no model id resolves to it, so it must be named. And once named, an
    /// `anthropic/claude-…` id builds an **OpenAI**-wire body — the mis-route this design exists to
    /// prevent. Provider first, dialect second.
    #[test]
    fn openrouter_is_named_explicitly_and_serves_claude_over_the_openai_wire() {
        let env = ProviderEnv::from_vars(
            &[
                ("OPENROUTER_API_KEY", "sk-or"),
                ("AI_PROVIDER", "openrouter"),
            ],
            false,
        );
        let identity = resolve(None, "anthropic/claude-sonnet-4.5", &env).expect("must resolve");
        assert_eq!(
            wire(&identity),
            (
                "https://openrouter.ai/api/v1",
                "/chat/completions",
                None,
                None
            ),
            "OpenRouter speaks the OpenAI wire even for Claude ids"
        );
    }

    /// Without `AI_PROVIDER`, an OpenRouter key is not enough — the id belongs to no native provider, so
    /// there is nothing to infer, and we say so instead of guessing.
    #[test]
    fn an_aggregator_key_alone_does_not_capture_a_vendor_slug_id() {
        let env = ProviderEnv::from_vars(&[("OPENROUTER_API_KEY", "sk-or")], false);
        let err = resolve(None, "moonshotai/kimi-k2.6", &env).expect_err("must not guess");
        assert!(err.contains("AI_PROVIDER"), "{err}");
    }

    /// The long tail: any OpenAI-compatible endpoint, no registry row.
    #[test]
    fn ai_base_url_routes_to_an_arbitrary_openai_compatible_endpoint() {
        let env = ProviderEnv::from_vars(
            &[
                ("AI_BASE_URL", "http://localhost:8000/v1"),
                ("AI_API_KEY", "local"),
            ],
            false,
        );
        let identity = resolve(None, "qwen3-coder", &env).expect("must resolve");
        assert_eq!(
            wire(&identity),
            ("http://localhost:8000/v1", "/chat/completions", None, None),
        );
    }

    /// A local server that ignores auth entirely is a legitimate setup, not a failure.
    #[test]
    fn ai_base_url_without_a_key_still_resolves() {
        let env = ProviderEnv::from_vars(&[("AI_BASE_URL", "http://localhost:11434/v1")], false);
        let identity = resolve(None, "qwen3-coder", &env).expect("must resolve");
        let GatewayCredentialIdentity::DirectOverride { bearer, .. } = &identity else {
            panic!("expected a static direct route")
        };
        assert_eq!(bearer, "");
    }

    /// **The key-leak invariant.** A row's env var pays for that row and nothing else. If someone points
    /// `AI_BASE_URL` at their own endpoint while `ANTHROPIC_API_KEY` happens to be exported, that
    /// Anthropic key must not be handed to their endpoint — a key sent to the wrong host is a key the
    /// user has to rotate.
    #[test]
    fn a_row_scoped_key_is_never_sent_to_an_unrelated_base_url() {
        let env = ProviderEnv::from_vars(
            &[
                ("ANTHROPIC_API_KEY", "sk-ant-secret"),
                ("AI_BASE_URL", "https://someone-elses-proxy.example.com/v1"),
            ],
            false,
        );
        let identity = resolve(None, "claude-opus-4-8", &env).expect("must resolve");
        let GatewayCredentialIdentity::DirectOverride {
            bearer, base_url, ..
        } = &identity
        else {
            panic!("expected a static direct route")
        };
        assert_eq!(base_url, "https://someone-elses-proxy.example.com/v1");
        assert_eq!(
            bearer, "",
            "ANTHROPIC_API_KEY must not travel to an arbitrary AI_BASE_URL"
        );
    }

    /// …but pointing `AI_BASE_URL` at Anthropic's *own* host is a different matter: that IS the row, so
    /// the row's key and its `x-api-key` scheme apply.
    #[test]
    fn ai_base_url_naming_a_known_host_adopts_that_rows_auth_scheme() {
        let env = ProviderEnv::from_vars(
            &[
                ("ANTHROPIC_API_KEY", "sk-ant-secret"),
                ("AI_BASE_URL", "https://api.anthropic.com"),
            ],
            false,
        );
        let identity = resolve(None, "claude-opus-4-8", &env).expect("must resolve");
        assert_eq!(wire(&identity).2, Some("x-api-key"));
        let GatewayCredentialIdentity::DirectOverride { bearer, .. } = &identity else {
            panic!("expected a static direct route")
        };
        assert_eq!(bearer, "sk-ant-secret");
    }

    /// **Regression: a configured gateway is never silently rerouted.** A key in the environment must not
    /// move a gateway deployment's traffic off the gateway (and off its metering). Both ways of
    /// configuring one are covered.
    #[test]
    fn a_configured_gateway_still_wins_over_an_ambient_provider_key() {
        // Configured by `AI_GATEWAY_URL` (gateway_configured = true) with a virtual key.
        let env = ProviderEnv::from_vars(&[("ANTHROPIC_API_KEY", "sk-ant")], true);
        let identity = resolve(Some("bai_v1.abc"), "claude-opus-4-8", &env).expect("must resolve");
        assert_eq!(
            identity,
            GatewayCredentialIdentity::StaticKey("bai_v1.abc".to_string()),
            "a configured gateway must keep using the gateway"
        );
    }

    /// `AI_DIRECT=1` is the explicit opt-out for a deployment that has a gateway but doesn't want it.
    #[test]
    fn ai_direct_forces_direct_even_with_a_gateway_configured() {
        let env =
            ProviderEnv::from_vars(&[("ANTHROPIC_API_KEY", "sk-ant"), ("AI_DIRECT", "1")], true);
        let identity = resolve(Some("bai_v1.abc"), "claude-opus-4-8", &env).expect("must resolve");
        assert_eq!(wire(&identity).0, "https://api.anthropic.com");
    }

    /// In direct mode `--key` is a BYO provider key, not a virtual key — so `AI_DIRECT=1 --key sk-ant-…`
    /// does the obvious thing rather than trying to present it to a gateway that isn't there.
    #[test]
    fn direct_mode_falls_back_to_the_cli_key_for_the_resolved_provider() {
        let env = ProviderEnv::from_vars(&[("AI_PROVIDER", "anthropic")], false);
        let identity = resolve(Some("sk-ant-cli"), "claude-opus-4-8", &env).expect("must resolve");
        let GatewayCredentialIdentity::DirectOverride { bearer, .. } = &identity else {
            panic!("expected a static direct route")
        };
        assert_eq!(bearer, "sk-ant-cli");
    }

    /// With no gateway and no key at all, name the one variable that would fix it.
    #[test]
    fn the_no_credential_error_names_the_expected_env_var() {
        let env = ProviderEnv::from_vars(&[], false);
        let err = resolve(None, "claude-opus-4-8", &env).expect_err("nothing to resolve");
        assert!(err.contains("ANTHROPIC_API_KEY"), "{err}");

        let err = resolve(None, "gpt-5", &env).expect_err("nothing to resolve");
        assert!(err.contains("OPENAI_API_KEY"), "{err}");

        // An id no provider claims: say *that*, rather than name a variable that would be ignored.
        let err = resolve(None, "some-local-model", &env).expect_err("nothing to resolve");
        assert!(
            err.contains("AI_PROVIDER") && err.contains("AI_BASE_URL"),
            "{err}"
        );
    }

    #[test]
    fn an_unknown_ai_provider_lists_the_known_ones() {
        let env = ProviderEnv::from_vars(&[("AI_PROVIDER", "nope"), ("AI_API_KEY", "k")], false);
        let err = resolve(None, "gpt-5", &env).expect_err("unknown provider");
        assert!(err.contains("unknown AI_PROVIDER"), "{err}");
        assert!(err.contains("openrouter"), "{err}");
    }

    /// Naming a provider whose key isn't set is a hard error that says which variable to set — not a
    /// silent fallthrough to some other provider that happens to have one.
    #[test]
    fn a_named_provider_with_no_key_errors_instead_of_falling_through() {
        let env = ProviderEnv::from_vars(
            &[("AI_PROVIDER", "groq"), ("ANTHROPIC_API_KEY", "sk-ant")],
            false,
        );
        let err = resolve(None, "claude-opus-4-8", &env).expect_err("groq has no key");
        assert!(err.contains("GROQ_API_KEY"), "{err}");
    }

    /// A provider row's required headers ride along automatically — NVIDIA's NIM poll timeout is an
    /// upstream requirement, not a tuning knob.
    #[test]
    fn a_rows_required_headers_are_attached_to_its_direct_route() {
        let spec = providers::by_id(AggregatorHost::Nvidia);
        let (routing, _, _) = registry_direct_routing(
            spec,
            spec.base_url.expect("nvidia has a base_url"),
            "deepseek-ai/deepseek-v3",
        );
        assert_eq!(
            routing.static_headers,
            vec![("NVCF-POLL-SECONDS", "3600".to_string())]
        );
    }

    /// A store with a stored Anthropic subscription login, written to a real temp file (the store is
    /// file-backed; nothing here touches `$HOME` or process env).
    fn store_with_anthropic_login(
        dir: &std::path::Path,
    ) -> (crate::auth_store::AuthStore, std::path::PathBuf) {
        let path = dir.join("auth.json");
        let mut store = crate::auth_store::AuthStore::open(path.clone());
        store
            .set(
                "anthropic",
                OAuthCredential::Anthropic(crate::oauth::anthropic::AnthropicCredential {
                    access: "oauth-access".to_string(),
                    refresh: "oauth-refresh".to_string(),
                    expires_at_ms: i64::MAX,
                }),
            )
            .expect("write the temp store");
        (crate::auth_store::AuthStore::open(path.clone()), path)
    }

    /// **The precedence decision.** `agent login anthropic` is an explicit, durable act;
    /// `ANTHROPIC_API_KEY` is very often exported in a shell for some unrelated tool. The login wins, so
    /// a subscribed user is never silently moved onto pay-per-token API billing by a stray shell export.
    #[test]
    fn a_stored_oauth_login_beats_an_ambient_provider_key() {
        let dir = tempdir();
        let (store, path) = store_with_anthropic_login(&dir);
        let env = ProviderEnv::from_vars(&[("ANTHROPIC_API_KEY", "sk-ant-ambient")], false);
        let (_credential, identity) =
            resolve_with(None, "claude-opus-4-8", &env, &store, path, &no_overrides())
                .expect("must resolve");
        // Direct-routed (there's no gateway), but authenticated by the OAuth source — not the API key.
        let GatewayCredentialIdentity::DirectOverrideOauth {
            provider, base_url, ..
        } = &identity
        else {
            panic!("the stored login must win over the ambient key, got {identity:?}")
        };
        assert_eq!(*provider, OAuthProviderId::Anthropic);
        assert_eq!(base_url, "https://api.anthropic.com");
    }

    /// …but an *explicit* per-invocation choice still beats the login: someone who types
    /// `AI_PROVIDER=anthropic` with a key in hand means it.
    #[test]
    fn an_explicit_ai_provider_beats_a_stored_login() {
        let dir = tempdir();
        let (store, path) = store_with_anthropic_login(&dir);
        let env = ProviderEnv::from_vars(
            &[
                ("ANTHROPIC_API_KEY", "sk-ant-ambient"),
                ("AI_PROVIDER", "anthropic"),
            ],
            false,
        );
        let (_credential, identity) =
            resolve_with(None, "claude-opus-4-8", &env, &store, path, &no_overrides())
                .expect("must resolve");
        let GatewayCredentialIdentity::DirectOverride { bearer, .. } = &identity else {
            panic!("an explicit AI_PROVIDER must win, got {identity:?}")
        };
        assert_eq!(bearer, "sk-ant-ambient");
    }

    /// Direct-mode Anthropic OAuth must still reach Anthropic's *own* endpoint, not the gateway's.
    /// Gateway mode is unchanged: no route override at all, relayed as before.
    #[test]
    fn anthropic_oauth_relays_through_a_configured_gateway_but_dials_anthropic_directly_without_one()
     {
        let dir = tempdir();
        let (store, path) = store_with_anthropic_login(&dir);

        let gateway_env = ProviderEnv::from_vars(&[], true);
        let (_c, identity) = resolve_with(
            None,
            "claude-opus-4-8",
            &gateway_env,
            &store,
            path.clone(),
            &no_overrides(),
        )
        .expect("must resolve");
        assert_eq!(
            identity,
            GatewayCredentialIdentity::Anthropic,
            "with a gateway configured, OAuth is relayed through it exactly as before"
        );

        let direct_env = ProviderEnv::from_vars(&[], false);
        let (_c, identity) = resolve_with(
            None,
            "claude-opus-4-8",
            &direct_env,
            &store,
            path,
            &no_overrides(),
        )
        .expect("must resolve");
        assert_eq!(wire(&identity).0, "https://api.anthropic.com");
    }

    /// A unique temp dir without pulling in a dev-dependency — the store is file-backed and these tests
    /// run in parallel, so each needs its own.
    fn tempdir() -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        let dir = std::env::temp_dir().join(format!(
            "beyond-ai-cred-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    /// An OAuth access token is **not** an API key. The Anthropic row's scheme (`x-api-key`, bare)
    /// describes how Anthropic takes an API key; a subscription token goes in `Authorization: Bearer` and
    /// is rejected outright in `x-api-key`. The row is right about the host, the wire, and the path — the
    /// credential decides the header.
    #[test]
    fn a_direct_anthropic_oauth_route_authenticates_with_bearer_not_x_api_key() {
        let dir = tempdir();
        let (store, path) = store_with_anthropic_login(&dir);
        let env = ProviderEnv::from_vars(&[], false);
        let (_c, identity) =
            resolve_with(None, "claude-opus-4-8", &env, &store, path, &no_overrides())
                .expect("must resolve");
        assert_eq!(
            wire(&identity),
            ("https://api.anthropic.com", "/v1/messages", None, None),
            "auth_header must be None (i.e. Authorization: Bearer) for an OAuth token"
        );
    }

    /// …while the same host reached with an actual API key does take `x-api-key`. Both rules, one row.
    #[test]
    fn a_direct_anthropic_api_key_route_still_uses_x_api_key() {
        let env = ProviderEnv::from_vars(&[("ANTHROPIC_API_KEY", "sk-ant")], false);
        let identity = resolve(None, "claude-opus-4-8", &env).expect("must resolve");
        assert_eq!(wire(&identity).2, Some("x-api-key"));
    }

    /// `OPENAI_BASE_URL` moves the OpenAI row's endpoint and *nothing else* — a variable exported for
    /// some other tool must not silently redirect Anthropic traffic.
    #[test]
    fn openai_base_url_is_scoped_to_the_openai_row() {
        let env = ProviderEnv::from_vars(
            &[
                ("OPENAI_BASE_URL", "https://my-proxy.example.com/v1"),
                ("OPENAI_API_KEY", "sk-oai"),
                ("ANTHROPIC_API_KEY", "sk-ant"),
            ],
            false,
        );
        let openai = resolve(None, "gpt-5", &env).expect("must resolve");
        assert_eq!(wire(&openai).0, "https://my-proxy.example.com/v1");

        let anthropic = resolve(None, "claude-opus-4-8", &env).expect("must resolve");
        assert_eq!(
            wire(&anthropic).0,
            "https://api.anthropic.com",
            "OPENAI_BASE_URL must not redirect a non-OpenAI provider"
        );
    }
}
