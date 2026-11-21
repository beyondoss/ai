//! OAuth client credentials for a configured `http` MCP server, and step-up on `403
//! insufficient_scope`.
//!
//! The pieces:
//!
//! - [`acquire`] gets a token from the server's authorization server with the client-credentials
//!   grant (`io.modelcontextprotocol/oauth-client-credentials`, ext-auth
//!   `specification/stable/oauth-client-credentials/2026-07-28`): rmcp discovers the AS from the
//!   server's protected-resource metadata, checks the AS supports the method, and posts the grant
//!   with `resource` set to the server's own URL. The secret is resolved for this one call and
//!   dropped. The token is held in memory only ([`ServerAuth`](super::mcp_oauth::ServerAuth)),
//!   keyed by `(server name, URL)`: never written to the `mcp-login` store, so it cannot be offered
//!   to another URL under the same name.
//! - The extension is declared on every request by
//!   (`mcp::client_info`), which rmcp turns into `_meta["io.modelcontextprotocol/clientCapabilities"]`.
//! - [`step_up`] is the pure decision for a `403 insufficient_scope`: what to do given the scopes
//!   already requested and the challenge. At most **one** step-up per request (the caller retries
//!   once and returns whatever that answers), and a scope the AS has refused is never asked for
//!   again.
//!
//! Service mode never reaches any of this: a grant connector's dial has no settings
//! (`mcp::connect_http`), so nothing here is called for it.

use std::time::Duration;

use rmcp::transport::auth::{
    ClientCredentialsConfig, EXTENSION_OAUTH_CLIENT_CREDENTIALS, JwtSigningAlgorithm, OAuthState,
};

use crate::settings::{McpClientCredentials, ResolvedClientCredential};
use crate::tools::mcp_oauth::RefreshError;

/// The extension's identifier, as declared in a request's client capabilities.
pub(crate) const EXTENSION_ID: &str = EXTENSION_OAUTH_CLIENT_CREDENTIALS;

/// What the authorization server issued.
pub(crate) struct Acquired {
    pub(crate) token: String,
    /// How long it lives, when the AS said.
    pub(crate) lifetime: Option<Duration>,
}

/// `scopes` in order, each once.
pub(crate) fn union_scopes(current: &[String], extra: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(current.len() + extra.len());
    for s in current.iter().chain(extra) {
        if !out.contains(s) {
            out.push(s.clone());
        }
    }
    out
}

/// The scopes in a space-separated `scope` value.
pub(crate) fn split_scopes(scope: &str) -> Vec<String> {
    scope.split_whitespace().map(str::to_owned).collect()
}

/// What a `403 insufficient_scope` calls for from a client-credentials server.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum StepUp {
    /// Request a token for exactly these scopes (the union of what was requested and what the
    /// challenge names), then retry the request once.
    Request(Vec<String>),
    /// The challenge named no scope: nothing to step up to.
    NoScope,
    /// Everything the challenge names was already requested (the AS granted it, or refused it): a
    /// second token for the same scopes would change nothing.
    AlreadyRequested,
    /// The AS refused the scopes this step-up would need; it was not asked again.
    Refused,
}

/// The step-up decision. `requested`: the scopes the current token was requested with. `challenged`:
/// the `scope` of the 403's challenge. `refused`: scope sets the AS has already refused.
pub(crate) fn step_up(
    requested: &[String],
    challenged: &[String],
    refused: &[Vec<String>],
) -> StepUp {
    if challenged.is_empty() {
        return StepUp::NoScope;
    }
    let wanted = union_scopes(requested, challenged);
    if wanted.len() == requested.len() {
        return StepUp::AlreadyRequested;
    }
    if refused.contains(&wanted) {
        return StepUp::Refused;
    }
    StepUp::Request(wanted)
}

fn signing_algorithm(name: &str) -> Option<JwtSigningAlgorithm> {
    Some(match name {
        "RS256" => JwtSigningAlgorithm::RS256,
        "RS384" => JwtSigningAlgorithm::RS384,
        "RS512" => JwtSigningAlgorithm::RS512,
        "ES256" => JwtSigningAlgorithm::ES256,
        "ES384" => JwtSigningAlgorithm::ES384,
        _ => return None,
    })
}

/// Get a token for `scopes` on `server` at `url` using `cc`. A failure is classified from the
/// authorization server's own answer where there is one: an RFC 6749 error code (`invalid_client`,
/// `invalid_scope`, ...) is definitive (asking again changes nothing); an unreachable AS or a 5xx is
/// transient.
pub(crate) async fn acquire(
    server: &str,
    url: &str,
    cc: &McpClientCredentials,
    scopes: &[String],
) -> Result<Acquired, RefreshError> {
    let credential = cc.resolve(server).map_err(RefreshError::definitive)?;
    let resource = Some(url.to_owned());
    let config = match credential {
        ResolvedClientCredential::Secret(client_secret) => ClientCredentialsConfig::ClientSecret {
            client_id: cc.client_id.clone(),
            client_secret,
            scopes: scopes.to_vec(),
            resource,
        },
        ResolvedClientCredential::PrivateKey { pem, algorithm } => {
            let signing_algorithm = signing_algorithm(&algorithm).ok_or_else(|| {
                RefreshError::definitive(format!("unsupported signing algorithm `{algorithm}`"))
            })?;
            ClientCredentialsConfig::PrivateKeyJwt {
                client_id: cc.client_id.clone(),
                signing_key: pem.into_bytes(),
                signing_algorithm,
                token_endpoint_audience: None,
                scopes: scopes.to_vec(),
                resource,
            }
        }
    };
    let http = std::sync::Arc::new(crate::tools::mcp_oauth::RecordingOAuthHttp::new()?);
    // rmcp 3.11.0's client-credentials future is not `Send` (it holds a `form_urlencoded`
    // serializer across an await), and this one is awaited from spawned tasks. So it runs to
    // completion on a blocking thread, built there.
    let handle = tokio::runtime::Handle::current();
    let (url_owned, http_owned) = (url.to_owned(), http.clone());
    let exchanged = tokio::task::spawn_blocking(move || {
        handle.block_on(exchange(url_owned, http_owned, config))
    })
    .await
    .map_err(|e| RefreshError::transient(format!("the token request did not finish: {e}")))?;
    let (token, lifetime) = match exchanged {
        Ok(got) => got,
        Err(e) => {
            return Err(http
                .token_response()
                .and_then(|(status, body)| {
                    crate::tools::mcp_oauth::classify_token_response(status, &body)
                })
                .unwrap_or(e));
        }
    };
    Ok(Acquired { token, lifetime })
}

/// rmcp's discovery, method check and grant for `config` against `url`: the token and its lifetime.
async fn exchange(
    url: String,
    http: std::sync::Arc<crate::tools::mcp_oauth::RecordingOAuthHttp>,
    config: ClientCredentialsConfig,
) -> Result<(String, Option<Duration>), RefreshError> {
    let mut state = OAuthState::new_with_oauth_http_client(url, http)
        .await
        .map_err(RefreshError::from_auth)?;
    state
        .authenticate_client_credentials(config)
        .await
        .map_err(RefreshError::from_auth)?;
    // `OAuthTokenResponse`'s accessors live in `oauth2`, which this crate does not name; its
    // serialized form is the RFC 6749 token response.
    let (_, response) = state
        .get_credentials()
        .await
        .map_err(RefreshError::from_auth)?;
    let value = response
        .and_then(|r| serde_json::to_value(&r).ok())
        .ok_or_else(|| RefreshError::transient("the token response was not kept"))?;
    let token = value
        .get("access_token")
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .ok_or_else(|| RefreshError::transient("the token response carried no access_token"))?;
    let lifetime = value
        .get("expires_in")
        .and_then(|v| v.as_u64())
        .map(Duration::from_secs);
    Ok((token, lifetime))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(s: &[&str]) -> Vec<String> {
        s.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn the_union_keeps_order_and_drops_repeats() {
        assert_eq!(
            union_scopes(&v(&["a", "b"]), &v(&["b", "c", "a"])),
            v(&["a", "b", "c"])
        );
    }

    #[test]
    fn a_challenge_with_a_new_scope_requests_the_union() {
        assert_eq!(
            step_up(&v(&["mcp:read"]), &v(&["mcp:write"]), &[]),
            StepUp::Request(v(&["mcp:read", "mcp:write"]))
        );
    }

    #[test]
    fn a_challenge_for_what_was_already_requested_is_not_retried() {
        assert_eq!(
            step_up(&v(&["mcp:read", "mcp:write"]), &v(&["mcp:write"]), &[]),
            StepUp::AlreadyRequested
        );
    }

    #[test]
    fn a_challenge_without_a_scope_has_nothing_to_step_up_to() {
        assert_eq!(step_up(&v(&["a"]), &[], &[]), StepUp::NoScope);
    }

    #[test]
    fn a_refused_union_is_not_asked_for_again() {
        let refused = vec![v(&["a", "admin"])];
        assert_eq!(
            step_up(&v(&["a"]), &v(&["admin"]), &refused),
            StepUp::Refused
        );
    }
}
