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

use std::sync::Arc;

use agent_core::client::{Credential, CredentialSource, DirectRouting, RouteOverride};

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
    // Fix 9 (pi-parity feature): a `models.json` override naming a `base_url` for this exact model id
    // wins outright, regardless of whether `--key`/`AI_AGENT_KEY` was also given — the override
    // redirects *where* the request goes (a locally-hosted or alternate-provider endpoint, entirely
    // bypassing the gateway), which is orthogonal to *how* it authenticates. Reuses the same
    // `DirectRouting`/`RouteOverride::Direct` mechanism the GitHub-Copilot OAuth routing below already
    // relies on, rather than duplicating it — see `settings::ModelOverride`'s own doc comment for the
    // on-disk schema.
    if let Some(over) = crate::settings::ModelOverrides::open_default().get(model) {
        if let Some(base_url) = over.base_url.clone() {
            let dialect = agent_core::dialect::Dialect::for_model(model);
            // Task #11 (pi-parity feature): resolved through `!command`/`$VAR`/literal syntax (see
            // `ModelOverride::resolved_api_key`'s own doc comment) rather than used as a raw literal —
            // lets an operator avoid storing a plaintext secret in `models.json`.
            let bearer = over.resolved_api_key(key.as_deref());
            let routing = DirectRouting {
                route: RouteOverride::Direct {
                    base_url,
                    path: dialect.endpoint_path(),
                },
                static_headers: Vec::new(),
                copilot_dynamic_headers: false,
                // Task #8 (pi-parity: Azure OpenAI routing support) — an operator-configured
                // `auth_header` (e.g. `"api-key"` for Azure) sends `bearer` through that named header
                // and omits `Authorization` entirely, instead of leaking a Bearer-shaped credential
                // (or, worse, a silent fallback to the gateway's own virtual key) to an endpoint that
                // doesn't want it.
                auth_header: over.auth_header.clone(),
            };
            return Ok(GatewayCredential::Oauth(Arc::new(StaticDirectCredentialSource {
                bearer,
                routing,
            })));
        }
    }

    if let Some(key) = key {
        return Ok(GatewayCredential::Static(key));
    }

    let store = crate::auth_store::AuthStore::open_default();
    let oauth_source = |provider: OAuthProviderId| {
        Arc::new(crate::auth_credential_source::OAuthCredentialSource::new(
            provider,
            crate::auth_store::default_path(),
        )) as Arc<dyn CredentialSource>
    };

    if agent_core::dialect::Dialect::for_model(model) == agent_core::dialect::Dialect::Anthropic
        && store.get("anthropic").is_some()
    {
        return Ok(GatewayCredential::Oauth(oauth_source(OAuthProviderId::Anthropic)));
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
                };
                return Ok(GatewayCredential::Oauth(Arc::new(DirectRoutedCredentialSource {
                    inner: oauth_source(OAuthProviderId::OpenaiCodex),
                    routing,
                })));
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
                return Ok(GatewayCredential::Oauth(Arc::new(
                    crate::oauth::github_copilot::CopilotRoutedCredentialSource {
                        inner: oauth_source(OAuthProviderId::GithubCopilot),
                        store_path: crate::auth_store::default_path(),
                        enterprise_url: c.enterprise_url.clone(),
                        path: crate::oauth::github_copilot::copilot_endpoint_path(dialect),
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
