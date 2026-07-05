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
            // Fix 1 (pi-parity, Round 2): an explicit `dialect` override wins over `Dialect::for_model`'s
            // name heuristic — consulted here (to pick the right endpoint path for a provider whose
            // model ids don't match the heuristic, e.g. Kimi-Coding's `kimi-k2-thinking`) AND threaded
            // into `DirectRouting::dialect_override` below (for the actual body-building/decoding
            // dialect `GatewayClient::stream` picks), so the two never disagree.
            let dialect = over
                .dialect
                .unwrap_or_else(|| agent_core::dialect::Dialect::for_model(model));
            // Task #11 (pi-parity feature): resolved through `!command`/`$VAR`/literal syntax (see
            // `ModelOverride::resolved_api_key`'s own doc comment) rather than used as a raw literal —
            // lets an operator avoid storing a plaintext secret in `models.json`.
            let bearer = over.resolved_api_key(key.as_deref());
            // Fix 3 (pi-parity, Round 2): computed together so a `deployment_name` override's URL
            // path segment (Task 46) and the `/v1`-doubling fix (Task 45) never fight each other — see
            // `direct_route_base_and_path`'s own doc comment.
            let (base_url, path) =
                direct_route_base_and_path(&base_url, dialect, over.deployment_name.as_deref());
            let routing = DirectRouting {
                route: RouteOverride::Direct { base_url, path },
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
                query: azure_api_version_query(over.api_version.as_deref()),
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
                    auth_header_prefix: None,
                    dialect_override: None,
                    deployment_name: None,
                    query: None,
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
fn direct_route_base_and_path(
    base_url: &str,
    dialect: agent_core::dialect::Dialect,
    deployment_name: Option<&str>,
) -> (String, &'static str) {
    let trimmed = base_url.trim_end_matches('/');
    match deployment_name {
        Some(name) => (
            format!("{trimmed}/openai/deployments/{name}"),
            crate::oauth::github_copilot::copilot_endpoint_path(dialect),
        ),
        None => (trimmed.to_string(), direct_route_path(dialect, trimmed)),
    }
}

/// Build the `api-version=…` query string from a `models.json` override's [`ModelOverride::api_version`]
/// field (Fix 2, pi-parity Round 2 — Azure OpenAI's dated REST `api-version`), or `None` if unset/empty.
/// Percent-encoded via [`url::form_urlencoded`] — the same general-purpose query-param encoder any other
/// query value would go through, rather than a hand-rolled `format!("api-version={v}")` that would
/// silently misbuild the URL if an operator's value ever contained a character needing escaping.
fn azure_api_version_query(api_version: Option<&str>) -> Option<String> {
    let version = api_version?;
    if version.is_empty() {
        return None;
    }
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
            azure_api_version_query(Some("2024-08-01-preview")),
            Some("api-version=2024-08-01-preview".to_string())
        );
    }

    #[test]
    fn azure_api_version_query_percent_encodes_special_characters() {
        // A defensive check, not a realistic input: Azure's own api-version strings never carry a
        // space, but the encoder must still do the right thing rather than build a broken URL.
        assert_eq!(
            azure_api_version_query(Some("2024 08 01")),
            Some("api-version=2024+08+01".to_string())
        );
    }

    #[test]
    fn azure_api_version_query_is_none_when_unset_or_empty() {
        assert_eq!(azure_api_version_query(None), None);
        assert_eq!(azure_api_version_query(Some("")), None);
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
    fn deployment_name_inserts_a_url_path_segment_and_composes_with_api_version_query() {
        use agent_core::dialect::Dialect;
        let (base_url, path) = direct_route_base_and_path(
            "https://my-resource.openai.azure.com",
            Dialect::OpenAiResponses,
            Some("my-deployment"),
        );
        let query = azure_api_version_query(Some("2024-08-01-preview")).unwrap();
        let url = format!("{base_url}{path}?{query}");
        assert_eq!(
            url,
            "https://my-resource.openai.azure.com/openai/deployments/my-deployment/responses\
             ?api-version=2024-08-01-preview"
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
}
