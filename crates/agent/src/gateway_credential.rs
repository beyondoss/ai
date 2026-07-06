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

use std::sync::Arc;

use agent_core::client::{Credential, CredentialSource, DirectRouting, RouteOverride};
use agent_core::models::AggregatorHost;

use crate::oauth::{OAuthCredential, OAuthProviderId};

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
pub fn resolve_gateway_credential(key: Option<String>, model: &str) -> Result<GatewayCredential, String> {
    resolve_gateway_credential_with_identity(key, model).map(|(credential, _identity)| credential)
}

/// [`resolve_gateway_credential`], additionally returning a [`GatewayCredentialIdentity`] alongside
/// the resolved credential — see that type's own doc comment (Fix #36, pi-parity audit). The sole
/// source of truth for both: [`resolve_gateway_credential`] is a thin wrapper around this function that
/// discards the identity half, so the two can never drift out of sync with each other.
pub fn resolve_gateway_credential_with_identity(
    key: Option<String>,
    model: &str,
) -> Result<(GatewayCredential, GatewayCredentialIdentity), String> {
    // Opened unconditionally, up front: both the `base_url`-override branch below (its own OAuth
    // fallback, pi-parity remediation pass 19 Task 2) and the plain non-override branches further down
    // need a stored-credential lookup, and a missing file is the cheap, ordinary case either way (see
    // `AuthStore::open_default`'s own doc comment).
    let store = crate::auth_store::AuthStore::open_default();
    let oauth_source = |provider: OAuthProviderId| {
        Arc::new(crate::auth_credential_source::OAuthCredentialSource::new(
            provider,
            crate::auth_store::default_path(),
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
    if let Some(over) = crate::settings::ModelOverrides::open_default().get(model) {
        if let Some(base_url) = over.base_url.clone() {
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
            if over.dialect.is_none() {
                if let Some(reason) = unsupported_wire_format_reason(model) {
                    return Err(reason);
                }
            }
            // Fix 1 (pi-parity, Round 2): an explicit `dialect` override wins over `Dialect::for_model`'s
            // name heuristic — consulted here (to pick the right endpoint path for a provider whose
            // model ids don't match the heuristic, e.g. Kimi-Coding's `kimi-k2-thinking`) AND threaded
            // into `DirectRouting::dialect_override` below (for the actual body-building/decoding
            // dialect `GatewayClient::stream` picks), so the two never disagree. Failing that, a handful
            // of OpenCode Zen/OpenCode-Go id/host combinations need a *different* default than
            // `Dialect::for_model`'s own generic bare-id table (`NATIVE_ANTHROPIC_WIRE_BARE_IDS`)
            // provides (pi-parity remediation pass 19, Task 1) — see [`opencode_dialect_override`]'s own
            // doc comment.
            let dialect = over.dialect.unwrap_or_else(|| {
                opencode_dialect_override(model, &base_url)
                    .unwrap_or_else(|| agent_core::dialect::Dialect::for_model(model))
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
            // pi-parity remediation pass 19, Task 3: which known third-party aggregator (if any) this
            // override's `base_url` names — see [`aggregator_host_for_base_url`]'s own doc comment for
            // what this does (and doesn't yet) close the loop on.
            let aggregator_host = aggregator_host_for_base_url(&base_url);
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
            };
            // pi-parity remediation pass 19, Task 2: the fallback tier below both this override's own
            // credential (`bearer`, just above) and `--key`/`AI_AGENT_KEY` — a stored OAuth login, still
            // routed to this override's own `base_url`/`routing` rather than the provider's usual
            // endpoint. Reuses `DirectRoutedCredentialSource` (already used for Codex's own fixed
            // routing further below) rather than inventing a second wrapper — the only thing that
            // differs per call site is which `DirectRouting` it carries.
            if let Some(provider) = oauth_fallback_provider(model, over, key.as_deref()) {
                if let Some(stored) = store.get(provider.store_key()) {
                    if stored.credential.provider() == provider {
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
                }
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
    }

    if let Some(key) = key {
        let identity = GatewayCredentialIdentity::StaticKey(key.clone());
        return Ok((GatewayCredential::Static(key), identity));
    }

    if agent_core::dialect::Dialect::for_model(model) == agent_core::dialect::Dialect::Anthropic
        && store.get("anthropic").is_some()
    {
        return Ok((
            GatewayCredential::Oauth(oauth_source(OAuthProviderId::Anthropic)),
            GatewayCredentialIdentity::Anthropic,
        ));
    }
    if model.contains("codex") {
        if let Some(stored) = store.get("openai-codex") {
            if let OAuthCredential::OpenaiCodex(c) = &stored.credential {
                // Still relayed through the gateway (a new `KNOWN_PROVIDERS` row: `chatgpt.com` is a
                // genuinely static host) under the `/openai-codex` prefix, with the account id this
                // backend requires attached as a static header — see `RouteOverride::Prefixed`.
                let routing = DirectRouting {
                    route: RouteOverride::Prefixed {
                        prefix: "/openai-codex",
                        path: "/backend-api/codex/responses",
                    },
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
        }
    }
    if let Some(stored) = store.get("github-copilot") {
        if let OAuthCredential::GithubCopilot(c) = &stored.credential {
            if c.available_model_ids.iter().any(|m| m == model) {
                // Bypasses the gateway entirely: GitHub hands back a *different* proxy host per
                // account/enterprise, embedded in the access token itself (`proxy-ep=…`) — not a
                // static host the gateway's `KNOWN_PROVIDERS` table could ever hold as a row. See
                // `RouteOverride::Direct`. Re-derived from the CURRENT token on every request by
                // `CopilotRoutedCredentialSource` itself (not computed here and frozen) — see that
                // type's own doc comment for why.
                //
                // `for_model_via_copilot(.., true)`, not plain `for_model`: at least one id (`gpt-4.1`)
                // is a different dialect under Copilot than it is natively (pi-parity — see that
                // function's doc comment), and this dialect also picks `copilot_endpoint_path`'s
                // baked-in `path` below, so getting it wrong here would send that id's Chat-Completions
                // body to a `/responses` path.
                let dialect = agent_core::dialect::Dialect::for_model_via_copilot(model, true);
                let path = crate::oauth::github_copilot::copilot_endpoint_path(dialect);
                let identity = GatewayCredentialIdentity::GithubCopilot {
                    enterprise_url: c.enterprise_url.clone(),
                    path,
                };
                return Ok((
                    GatewayCredential::Oauth(Arc::new(
                        crate::oauth::github_copilot::CopilotRoutedCredentialSource {
                            inner: oauth_source(OAuthProviderId::GithubCopilot),
                            store_path: crate::auth_store::default_path(),
                            enterprise_url: c.enterprise_url.clone(),
                            path,
                        },
                    )),
                    identity,
                ));
            }
        }
    }

    Err(format!(
        "no gateway key: pass --key or set AI_AGENT_KEY (a bai_v1… virtual key), or run `agent login \
         <provider>` to use a subscription for model {model:?}"
    ))
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

/// Which OpenCode aggregator a `models.json` override's `base_url` names, if either (pi-parity
/// remediation pass 19, Task 1). pi's own `packages/ai/src/providers/opencode.models.ts` (OpenCode Zen)
/// and `opencode-go.models.ts` (OpenCode-Go) are two distinct aggregators nested under the same
/// registered domain — `opencode.ai/zen`(`/v1`) vs. `opencode.ai/zen/go`(`/v1`) — each with its own
/// catalogue and, for a handful of ids, a genuinely different real wire dialect than the identical bare
/// id gets on the other. A host-only check (à la [`is_azure_host`]) can't distinguish the two — they
/// share the same registered domain — so this also inspects the path's `/go` segment. See
/// [`opencode_dialect_override`]'s own doc comment for the concrete id/host combinations this exists to
/// resolve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpenCodeHost {
    /// `opencode.ai/zen`(`/v1`) — `opencode.models.ts`.
    Zen,
    /// `opencode.ai/zen/go`(`/v1`) — `opencode-go.models.ts`.
    Go,
}

fn opencode_host(base_url: &str) -> Option<OpenCodeHost> {
    let url = url::Url::parse(base_url).ok()?;
    if url.host_str() != Some("opencode.ai") {
        return None;
    }
    let path = url.path().trim_end_matches('/');
    if path == "/zen/go" || path.starts_with("/zen/go/") {
        Some(OpenCodeHost::Go)
    } else if path == "/zen" || path.starts_with("/zen/") {
        Some(OpenCodeHost::Zen)
    } else {
        None
    }
}

/// The 4 known id/host combinations (pi-parity remediation pass 19, Task 1) where
/// `agent_core::dialect::NATIVE_ANTHROPIC_WIRE_BARE_IDS`'s otherwise-correct generic bare-id default
/// (Anthropic wire) is wrong for one *specific* OpenCode Zen/OpenCode-Go host, because that id is
/// genuinely served over OpenAI Chat Completions there instead — a `models.json` override naming one of
/// these exact ids on the matching host, with no explicit `dialect` of its own, would otherwise silently
/// build an Anthropic-shaped body and send it to an endpoint that speaks Chat Completions.
///
/// Verified against pi's real catalogues:
/// - `packages/ai/src/providers/opencode.models.ts` (OpenCode Zen): `"minimax-m2.7"`/`"minimax-m3"` are
///   `api: "openai-completions"` there (`NATIVE_ANTHROPIC_WIRE_BARE_IDS` lists both only for *native*
///   MiniMax's/MiniMax-CN's sake — genuinely Anthropic-wire on those hosts, not on OpenCode Zen).
/// - `packages/ai/src/providers/opencode-go.models.ts` (OpenCode-Go): `"minimax-m2.7"`/`"qwen3.6-plus"`
///   are `api: "openai-completions"` there too — a *different* split again from OpenCode Zen, where
///   `"qwen3.6-plus"` is genuinely Anthropic-wire. `dialect/mod.rs`'s own `NATIVE_ANTHROPIC_WIRE_BARE_IDS`
///   doc comment already flagged both of OpenCode-Go's ids as "genuinely `openai-completions`" there —
///   this is that flag finally acted on.
///
/// OpenCode Zen's own `qwen3.5-plus`/`qwen3.6-plus` and OpenCode-Go's own `minimax-m3`/`qwen3.7-max`/
/// `qwen3.7-plus` are deliberately *not* in this table: those are genuinely Anthropic-wire on their
/// respective host, so `NATIVE_ANTHROPIC_WIRE_BARE_IDS`'s existing default is already correct for them
/// and this function returns `None`, falling through unchanged.
///
/// Unlike [`unsupported_wire_format_reason`]'s sibling fix for OpenCode Zen's 3 Gemini-hosted ids, this
/// resolves to a *supported* dialect (OpenAI Chat Completions) rather than a hard failure: beyond has a
/// perfectly good dialect for these 4 combinations, so silently defaulting to the wrong one had a real,
/// avoidable fix available — unlike the Gemini case, where no dialect exists for that wire format at all
/// and failing loudly is the only honest option.
fn opencode_dialect_override(model: &str, base_url: &str) -> Option<agent_core::dialect::Dialect> {
    let m = model.to_ascii_lowercase();
    match (opencode_host(base_url)?, m.as_str()) {
        (OpenCodeHost::Zen, "minimax-m2.7" | "minimax-m3") => Some(agent_core::dialect::Dialect::OpenAi),
        (OpenCodeHost::Go, "minimax-m2.7" | "qwen3.6-plus") => Some(agent_core::dialect::Dialect::OpenAi),
        _ => None,
    }
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

/// Which of the 7 known third-party aggregator platforms (`agent_core::models::AggregatorHost`) a
/// `models.json` override's (already-normalized) `base_url` names, if any (pi-parity remediation pass 19,
/// Task 3) — the "BYO/direct-routed" half of the host signal
/// `agent_core::models::capabilities_for_route_with_host` exists to consume; see
/// `agent_core::transport::ModelRequest::host`'s own doc comment for the full mechanism this is meant to
/// close the loop on. Matched by parsed hostname (see [`is_azure_host`]'s identical reasoning for why not
/// a raw substring check), against each aggregator's real, documented base authority — pi's own
/// `huggingface.models.ts`/`nvidia.models.ts`/`kimi-coding.models.ts`/`together.models.ts`/
/// `groq.models.ts`/`openrouter.models.ts` `baseUrl`s. Fireworks is deliberately excluded:
/// `agent_core::client::GatewayClient::stream` already resolves it unconditionally from the model id's
/// own shape (`is_fireworks_model`) before this signal would ever be consulted, so there's no
/// base_url-matching case to add for it here.
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
/// shape this function already handles for the 3 BYO-only hosts (HuggingFace/NVIDIA/Kimi-Coding, which
/// have no gateway route at all).
///
/// Not yet threaded any further than this file's own [`GatewayCredentialIdentity`] — see that type's
/// `DirectOverride`/`DirectOverrideOauth` variants. Actually reaching `ModelRequest::host` would need
/// either a new field on `agent_core::client::DirectRouting` (agent-core, unowned this round) for
/// `GatewayClient::stream` to read back out, or `crates/agent/src/serve.rs` (a sibling agent's file this
/// round) calling this function directly when it builds a `ModelRequest` — this module lands the
/// detection primitive itself, matching the precedent [`GatewayCredentialIdentity`] (Fix #36) already set
/// for landing an unwired primitive when the consuming call site belongs to someone else this round.
pub(crate) fn aggregator_host_for_base_url(base_url: &str) -> Option<AggregatorHost> {
    let host = url::Url::parse(base_url).ok()?.host_str()?.to_ascii_lowercase();
    match host.as_str() {
        "router.huggingface.co" => Some(AggregatorHost::HuggingFace),
        "integrate.api.nvidia.com" => Some(AggregatorHost::Nvidia),
        "api.kimi.com" => Some(AggregatorHost::KimiCoding),
        "api.together.ai" | "api.together.xyz" => Some(AggregatorHost::Together),
        "api.groq.com" => Some(AggregatorHost::Groq),
        "openrouter.ai" => Some(AggregatorHost::OpenRouter),
        _ => None,
    }
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
            Self::OpenaiCodex { account_id } => {
                f.debug_struct("OpenaiCodex").field("account_id", account_id).finish()
            }
            Self::GithubCopilot { enterprise_url, path } => f
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
const AZURE_HOST_SUFFIXES: &[&str] = &[".openai.azure.com", ".cognitiveservices.azure.com", ".ai.azure.com"];

/// Fix #34 (pi-parity audit): whether `base_url`'s host is a recognized Azure OpenAI hostname (see
/// [`AZURE_HOST_SUFFIXES`]) — used both by [`normalize_azure_base_url`] and, independently, by
/// [`resolve_gateway_credential_with_identity`] to gate [`azure_api_version_query`]'s Fix #35 default
/// (an unparsable `base_url`, or one with no host at all, is never treated as Azure).
fn is_azure_host(base_url: &str) -> bool {
    url::Url::parse(base_url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_string))
        .is_some_and(|host| AZURE_HOST_SUFFIXES.iter().any(|suffix| host.ends_with(suffix)))
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
    if !AZURE_HOST_SUFFIXES.iter().any(|suffix| host.ends_with(suffix)) {
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
        assert!(is_azure_host("https://my-resource.cognitiveservices.azure.com"));
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
    fn normalize_azure_base_url_rewrites_the_full_v1_responses_shape_dropping_the_bare_path_suffix() {
        assert_eq!(
            normalize_azure_base_url("https://my-resource.openai.azure.com/openai/v1/responses"),
            Some("https://my-resource.openai.azure.com/openai/v1".to_string())
        );
    }

    #[test]
    fn normalize_azure_base_url_ignores_a_trailing_slash_and_a_query_string() {
        assert_eq!(
            normalize_azure_base_url("https://my-resource.openai.azure.com/openai/v1/responses/?api-version=v1"),
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
            normalize_azure_base_url("https://my-resource.openai.azure.com/openai/deployments/my-deployment"),
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
        assert_eq!(direct_route_path(Dialect::Anthropic, "http://host"), "/v1/messages");
    }

    #[test]
    fn base_url_ending_in_v1_no_longer_doubles_the_segment_for_an_openai_wire_dialect() {
        // The exact bug empirically confirmed for Task 45: an Azure-style BYO `base_url` that already
        // carries `/v1` (a natural way to configure such an endpoint) previously produced a doubled
        // "/v1/v1/responses" when routed.
        use agent_core::dialect::Dialect;
        let (base_url, path) =
            direct_route_base_and_path("http://host/openai/v1", Dialect::OpenAiResponses, None);
        assert_eq!(format!("{base_url}{path}"), "http://host/openai/v1/responses");
    }

    #[test]
    fn base_url_without_v1_still_gets_the_full_default_path_unchanged() {
        use agent_core::dialect::Dialect;
        let (base_url, path) = direct_route_base_and_path("http://host", Dialect::OpenAiResponses, None);
        assert_eq!(format!("{base_url}{path}"), "http://host/v1/responses");
    }

    // Task 46: `deployment_name` becomes a URL path segment (Azure's classic dated-`api-version` REST
    // convention), composing cleanly with Task 45's fix rather than fighting it.

    #[test]
    fn deployment_name_inserts_a_url_path_segment_for_chat_completions_and_composes_with_api_version_query() {
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
        let (base_url, path) =
            direct_route_base_and_path("https://my-resource.openai.azure.com/v1", Dialect::OpenAi, Some("gpt4"));
        assert_eq!(
            format!("{base_url}{path}"),
            "https://my-resource.openai.azure.com/v1/openai/deployments/gpt4/chat/completions"
        );
    }

    #[test]
    fn deployment_name_trims_a_trailing_slash_on_base_url_before_inserting_the_segment() {
        use agent_core::dialect::Dialect;
        let (base_url, path) =
            direct_route_base_and_path("https://my-resource.openai.azure.com/", Dialect::OpenAi, Some("gpt4"));
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
        assert_eq!(unsupported_wire_format_reason("gemini-3-flash-preview"), None);
    }

    #[test]
    fn unsupported_wire_format_reason_message_names_the_model_and_suggests_the_escape_hatch() {
        let reason = unsupported_wire_format_reason("gemini-3-flash").unwrap();
        assert!(reason.contains("gemini-3-flash"), "got: {reason}");
        assert!(reason.contains("dialect"), "got: {reason}");
    }

    // pi-parity remediation pass 19, Task 1: `opencode_host`/`opencode_dialect_override` fix 4 known
    // id/host combinations where OpenCode Zen/OpenCode-Go serve a bare id (already in
    // `NATIVE_ANTHROPIC_WIRE_BARE_IDS` for another host's sake) over OpenAI Chat Completions instead of
    // the generic Anthropic-wire default.

    #[test]
    fn opencode_host_recognizes_zen_with_or_without_the_v1_suffix() {
        for url in [
            "https://opencode.ai/zen",
            "https://opencode.ai/zen/v1",
            "https://opencode.ai/zen/",
        ] {
            assert_eq!(opencode_host(url), Some(OpenCodeHost::Zen), "{url}");
        }
    }

    #[test]
    fn opencode_host_recognizes_go_with_or_without_the_v1_suffix() {
        for url in [
            "https://opencode.ai/zen/go",
            "https://opencode.ai/zen/go/v1",
            "https://opencode.ai/zen/go/",
        ] {
            assert_eq!(opencode_host(url), Some(OpenCodeHost::Go), "{url}");
        }
    }

    #[test]
    fn opencode_host_is_none_for_an_unrelated_host_an_unrelated_path_or_an_unparsable_url() {
        assert_eq!(opencode_host("https://api.together.ai/v1"), None);
        assert_eq!(opencode_host("https://opencode.ai/other"), None);
        assert_eq!(opencode_host("not a url at all"), None);
    }

    #[test]
    fn opencode_dialect_override_forces_openai_for_all_4_known_mis_dialected_combinations() {
        use agent_core::dialect::Dialect;
        // OpenCode Zen: minimax-m2.7/minimax-m3 are `api: "openai-completions"` there
        // (opencode.models.ts), despite both being in `NATIVE_ANTHROPIC_WIRE_BARE_IDS` for native
        // MiniMax's/MiniMax-CN's sake.
        assert_eq!(
            opencode_dialect_override("minimax-m2.7", "https://opencode.ai/zen/v1"),
            Some(Dialect::OpenAi)
        );
        assert_eq!(
            opencode_dialect_override("minimax-m3", "https://opencode.ai/zen/v1"),
            Some(Dialect::OpenAi)
        );
        // OpenCode-Go: minimax-m2.7/qwen3.6-plus are `api: "openai-completions"` there too
        // (opencode-go.models.ts) — a different split again from OpenCode Zen, where qwen3.6-plus is
        // genuinely Anthropic-wire.
        assert_eq!(
            opencode_dialect_override("minimax-m2.7", "https://opencode.ai/zen/go/v1"),
            Some(Dialect::OpenAi)
        );
        assert_eq!(
            opencode_dialect_override("qwen3.6-plus", "https://opencode.ai/zen/go/v1"),
            Some(Dialect::OpenAi)
        );
    }

    #[test]
    fn opencode_dialect_override_is_case_insensitive_on_the_model_id() {
        use agent_core::dialect::Dialect;
        assert_eq!(
            opencode_dialect_override("MiniMax-M3", "https://opencode.ai/zen/v1"),
            Some(Dialect::OpenAi)
        );
    }

    #[test]
    fn opencode_dialect_override_is_none_for_ids_that_are_already_correctly_anthropic_on_their_host() {
        // OpenCode Zen's own qwen3.6-plus is genuinely Anthropic-wire — the existing
        // `NATIVE_ANTHROPIC_WIRE_BARE_IDS` default is already correct, so this must not override it.
        assert_eq!(opencode_dialect_override("qwen3.6-plus", "https://opencode.ai/zen"), None);
        // OpenCode-Go's own minimax-m3/qwen3.7-max are also genuinely Anthropic-wire there.
        assert_eq!(opencode_dialect_override("minimax-m3", "https://opencode.ai/zen/go"), None);
        assert_eq!(opencode_dialect_override("qwen3.7-max", "https://opencode.ai/zen/go"), None);
    }

    #[test]
    fn opencode_dialect_override_is_none_for_an_unrelated_host() {
        assert_eq!(
            opencode_dialect_override("minimax-m3", "https://api.together.ai/v1"),
            None
        );
    }

    // pi-parity remediation pass 19, Task 2: `oauth_fallback_provider` decides whether a `models.json`
    // `base_url` override with no bearer of its own should still authenticate via a stored OAuth login.

    #[test]
    fn oauth_fallback_provider_picks_anthropic_for_a_claude_dialect_model_with_no_credential_of_its_own() {
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
        assert_eq!(oauth_fallback_provider("claude-opus-4-8", &over, None), None);
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
    fn oauth_fallback_provider_is_none_for_a_third_party_model_that_is_neither_anthropic_nor_codex() {
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
        assert_eq!(aggregator_host_for_base_url("https://my-ollama-box.example.com"), None);
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
        let make = |base_url: &str, deployment: Option<&str>| GatewayCredentialIdentity::DirectOverride {
            base_url: base_url.to_string(),
            path: "/chat/completions",
            bearer: "key-1".to_string(),
            auth_header: None,
            auth_header_prefix: None,
            deployment_name: deployment.map(str::to_string),
            query: None,
            aggregator_host: None,
        };
        assert_eq!(make("https://host/openai/v1", None), make("https://host/openai/v1", None));
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
        assert_eq!(GatewayCredentialIdentity::Anthropic, GatewayCredentialIdentity::Anthropic);
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
        let make = |base_url: &str, provider: OAuthProviderId| GatewayCredentialIdentity::DirectOverrideOauth {
            base_url: base_url.to_string(),
            path: "/v1/messages",
            auth_header: None,
            auth_header_prefix: None,
            deployment_name: None,
            query: None,
            aggregator_host: None,
            provider,
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
}
