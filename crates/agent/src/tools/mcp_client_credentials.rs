//! OAuth client credentials for a configured `http` MCP server, and step-up on `403
//! insufficient_scope`.
//!
//! The pieces:
//!
//! - [`acquire`] gets a token from the server's authorization server with the client-credentials
//!   grant (`io.modelcontextprotocol/oauth-client-credentials`, ext-auth
//!   `specification/stable/oauth-client-credentials/2026-07-28`): rmcp discovers the AS from the
//!   server's protected-resource metadata; **the AS must be the pinned `issuer`** (exact match) or
//!   no credential is resolved or sent; the grant goes to that metadata's token endpoint with
//!   `resource` set to the server's own URL. The secret is resolved for this one call and
//!   dropped. The `private_key_jwt` assertion is signed here with `ring` (rmcp's own JWT feature
//!   would link aws-lc-rs into the agent). The token is held in memory only ([`ServerAuth`](super::mcp_oauth::ServerAuth)),
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

use rmcp::transport::auth::EXTENSION_OAUTH_CLIENT_CREDENTIALS;

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

/// The authorization server must be the pinned one: exact string equality with the metadata's
/// `issuer` (RFC 8414 §3.3). Nothing the resource server's own metadata names is trusted with a
/// credential until this holds.
pub(crate) fn check_issuer(pinned: &str, discovered: Option<&str>) -> Result<(), String> {
    match discovered {
        Some(d) if d == pinned => Ok(()),
        Some(d) => Err(format!(
            "the server's authorization server is `{d}`, not the pinned issuer `{pinned}`; \
             credentials not sent"
        )),
        None => Err(format!(
            "the server's authorization server metadata names no issuer, not the pinned `{pinned}`; \
             credentials not sent"
        )),
    }
}

/// A token endpoint credentials may be sent to: https, or http on a loopback host (local testing).
pub(crate) fn endpoint_allowed(endpoint: &str) -> bool {
    let Ok(u) = url::Url::parse(endpoint) else {
        return false;
    };
    match u.scheme() {
        "https" => true,
        "http" => match u.host() {
            Some(url::Host::Domain(d)) => d == "localhost",
            Some(url::Host::Ipv4(ip)) => ip.is_loopback(),
            Some(url::Host::Ipv6(ip)) => ip.is_loopback(),
            None => false,
        },
        _ => false,
    }
}

fn b64(bytes: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

/// The RFC 7523 client assertion for `client_id`: `iss` = `sub` = the client, `aud` = the issuer as
/// the only value (RFC 10070), short-lived, a fresh `jti`. `pem` is a PKCS#8 key (any algorithm) or a
/// PKCS#1 `RSA PRIVATE KEY`. Never includes the key in an error.
pub(crate) fn client_assertion(
    client_id: &str,
    issuer: &str,
    algorithm: &str,
    pem: &str,
    now_secs: u64,
) -> Result<String, String> {
    use ring::rand::SystemRandom;
    use ring::signature::{
        ECDSA_P256_SHA256_FIXED_SIGNING, ECDSA_P384_SHA384_FIXED_SIGNING, EcdsaKeyPair,
        RSA_PKCS1_SHA256, RSA_PKCS1_SHA384, RSA_PKCS1_SHA512, RsaKeyPair,
    };
    let bad = |what: &str| format!("the private key could not be used: {what}");
    let (label, der) = {
        use base64::Engine as _;
        let mut lines = pem.lines().map(str::trim).filter(|l| !l.is_empty());
        let begin = lines.next().ok_or_else(|| bad("empty"))?;
        let label = begin
            .strip_prefix("-----BEGIN ")
            .and_then(|l| l.strip_suffix("-----"))
            .ok_or_else(|| bad("not PEM"))?
            .to_owned();
        let body: String = lines.take_while(|l| !l.starts_with("-----END")).collect();
        let der = base64::engine::general_purpose::STANDARD
            .decode(body)
            .map_err(|_| bad("not base64"))?;
        (label, der)
    };
    let rng = SystemRandom::new();
    let header = serde_json::json!({"alg": algorithm, "typ": "client-authentication+jwt"});
    let mut jti = [0u8; 16];
    ring::rand::SecureRandom::fill(&rng, &mut jti).map_err(|_| bad("no randomness"))?;
    let claims = serde_json::json!({
        "iss": client_id, "sub": client_id, "aud": issuer,
        "iat": now_secs, "exp": now_secs + 60, "jti": hex::encode(jti),
    });
    let signing_input = format!(
        "{}.{}",
        b64(header.to_string().as_bytes()),
        b64(claims.to_string().as_bytes())
    );
    let sig = match algorithm {
        "RS256" | "RS384" | "RS512" => {
            let pair = if label == "RSA PRIVATE KEY" {
                RsaKeyPair::from_der(&der)
            } else {
                RsaKeyPair::from_pkcs8(&der)
            }
            .map_err(|_| bad("not an RSA key"))?;
            let padding = match algorithm {
                "RS256" => &RSA_PKCS1_SHA256,
                "RS384" => &RSA_PKCS1_SHA384,
                _ => &RSA_PKCS1_SHA512,
            };
            let mut sig = vec![0u8; pair.public().modulus_len()];
            pair.sign(padding, &rng, signing_input.as_bytes(), &mut sig)
                .map_err(|_| bad("signing failed"))?;
            sig
        }
        "ES256" | "ES384" => {
            let alg = if algorithm == "ES256" {
                &ECDSA_P256_SHA256_FIXED_SIGNING
            } else {
                &ECDSA_P384_SHA384_FIXED_SIGNING
            };
            let pair = EcdsaKeyPair::from_pkcs8(alg, &der, &rng)
                .map_err(|_| bad("not a PKCS#8 EC key for this algorithm"))?;
            pair.sign(&rng, signing_input.as_bytes())
                .map_err(|_| bad("signing failed"))?
                .as_ref()
                .to_vec()
        }
        other => return Err(format!("unsupported signing algorithm `{other}`")),
    };
    Ok(format!("{signing_input}.{}", b64(&sig)))
}

/// Get a token for `scopes` on `server` at `url` using `cc`.
///
/// 1. Discover the authorization server from the server's protected-resource metadata (rmcp). No
///    credential is involved yet.
/// 2. **Only if its `issuer` equals `cc.issuer` exactly** (and it was really published, and its token
///    endpoint is https or loopback) is the secret resolved and the grant sent, to the token
///    endpoint of *that* metadata. A server cannot steer credentials to an authorization server of
///    its own choosing.
/// 3. The grant: `client_credentials` with `resource` = the server's URL, authenticating with the
///    secret in the body (or HTTP Basic if that is all the AS offers) or an RFC 7523 assertion.
///
/// A failure is classified from the authorization server's own answer where there is one: an RFC
/// 6749 error code (`invalid_client`, `invalid_scope`, ...) is definitive (asking again changes
/// nothing); an unreachable AS or a 5xx is transient.
pub(crate) async fn acquire(
    server: &str,
    url: &str,
    cc: &McpClientCredentials,
    scopes: &[String],
) -> Result<Acquired, RefreshError> {
    agent_core::ensure_provider();
    let http = std::sync::Arc::new(crate::tools::mcp_oauth::RecordingOAuthHttp::new()?);
    let manager =
        rmcp::transport::auth::AuthorizationManager::new_with_oauth_http_client(url, http)
            .await
            .map_err(RefreshError::from_auth)?;
    let resolution = manager
        .resolve_metadata()
        .await
        .map_err(RefreshError::from_auth)?;
    if !resolution.source.is_discovered() {
        return Err(RefreshError::definitive(
            "the server published no authorization server metadata; credentials not sent",
        ));
    }
    let metadata = resolution.metadata;
    check_issuer(&cc.issuer, metadata.issuer.as_deref()).map_err(RefreshError::definitive)?;
    if !endpoint_allowed(&metadata.token_endpoint) {
        return Err(RefreshError::definitive(
            "the token endpoint is not https; credentials not sent",
        ));
    }
    let supported = |key: &str, want: &[&str]| {
        metadata
            .additional_fields
            .get(key)
            .and_then(|v| v.as_array())
            .is_none_or(|list| {
                list.iter()
                    .any(|m| m.as_str().is_some_and(|m| want.contains(&m)))
            })
    };
    // Built in a block: the form serializer is not `Send` and must not live across the await below.
    let (body, basic) = {
        let credential = cc.resolve(server).map_err(RefreshError::definitive)?;
        let mut form = url::form_urlencoded::Serializer::new(String::new());
        form.append_pair("grant_type", "client_credentials");
        if !scopes.is_empty() {
            form.append_pair("scope", &scopes.join(" "));
        }
        form.append_pair("resource", url);
        let mut basic: Option<String> = None;
        match credential {
            ResolvedClientCredential::Secret(secret) => {
                if supported(
                    "token_endpoint_auth_methods_supported",
                    &["client_secret_post"],
                ) {
                    form.append_pair("client_id", &cc.client_id);
                    form.append_pair("client_secret", &secret);
                } else if supported(
                    "token_endpoint_auth_methods_supported",
                    &["client_secret_basic"],
                ) {
                    let enc = |s: &str| {
                        url::form_urlencoded::byte_serialize(s.as_bytes()).collect::<String>()
                    };
                    use base64::Engine as _;
                    basic = Some(format!(
                        "Basic {}",
                        base64::engine::general_purpose::STANDARD.encode(format!(
                            "{}:{}",
                            enc(&cc.client_id),
                            enc(&secret)
                        ))
                    ));
                } else {
                    return Err(RefreshError::definitive(
                        "the authorization server supports neither client_secret_post nor client_secret_basic",
                    ));
                }
            }
            ResolvedClientCredential::PrivateKey { pem, algorithm } => {
                if !supported(
                    "token_endpoint_auth_methods_supported",
                    &["private_key_jwt"],
                ) {
                    return Err(RefreshError::definitive(
                        "the authorization server does not support private_key_jwt",
                    ));
                }
                if !supported(
                    "token_endpoint_auth_signing_alg_values_supported",
                    &[algorithm.as_str()],
                ) {
                    return Err(RefreshError::definitive(format!(
                        "the authorization server does not support the signing algorithm {algorithm}"
                    )));
                }
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_secs());
                let assertion = client_assertion(&cc.client_id, &cc.issuer, &algorithm, &pem, now)
                    .map_err(RefreshError::definitive)?;
                form.append_pair(
                    "client_assertion_type",
                    "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
                );
                form.append_pair("client_assertion", &assertion);
            }
        }
        (form.finish(), basic)
    };
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|e| RefreshError::transient(e.to_string()))?;
    let mut request = client
        .post(&metadata.token_endpoint)
        .header("content-type", "application/x-www-form-urlencoded")
        .header("accept", "application/json")
        .body(body);
    if let Some(basic) = basic {
        request = request.header("authorization", basic);
    }
    let response = request.send().await.map_err(|e| {
        RefreshError::transient(format!("the token endpoint was not reachable: {e}"))
    })?;
    let status = response.status().as_u16();
    let bytes = response
        .bytes()
        .await
        .map_err(|e| RefreshError::transient(format!("the token response was cut off: {e}")))?;
    if bytes.len() > 1024 * 1024 {
        return Err(RefreshError::transient("the token response was too large"));
    }
    if status >= 400 {
        return Err(
            crate::tools::mcp_oauth::classify_token_response(status, &bytes).unwrap_or_else(|| {
                RefreshError::transient(format!("the token endpoint answered HTTP {status}"))
            }),
        );
    }
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|_| RefreshError::transient("the token response was not JSON"))?;
    let token = value
        .get("access_token")
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .ok_or_else(|| RefreshError::transient("the token response carried no access_token"))?;
    let lifetime = value
        .get("expires_in")
        .and_then(|v| v.as_u64())
        .map(Duration::from_secs);
    Ok(Acquired { token, lifetime })
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

    #[test]
    fn only_the_pinned_issuer_matches_exactly() {
        assert!(check_issuer("https://as.example/r", Some("https://as.example/r")).is_ok());
        for other in [
            "https://as.example/r/",
            "https://evil.example/r",
            "HTTPS://as.example/r",
        ] {
            let e = check_issuer("https://as.example/r", Some(other)).unwrap_err();
            assert!(e.contains("credentials not sent"), "{e}");
        }
        assert!(check_issuer("https://as.example/r", None).is_err());
    }

    #[test]
    fn a_token_endpoint_must_be_https_or_loopback() {
        assert!(endpoint_allowed("https://as.example/token"));
        assert!(endpoint_allowed("http://127.0.0.1:8080/token"));
        assert!(endpoint_allowed("http://localhost/token"));
        assert!(!endpoint_allowed("http://as.example/token"));
        assert!(!endpoint_allowed("ftp://as.example/token"));
        assert!(!endpoint_allowed("not a url"));
    }

    #[test]
    fn the_client_assertion_is_a_verifiable_es256_jwt_for_the_issuer() {
        use base64::Engine as _;
        use ring::signature::{
            ECDSA_P256_SHA256_FIXED, ECDSA_P256_SHA256_FIXED_SIGNING, EcdsaKeyPair, KeyPair as _,
            UnparsedPublicKey,
        };
        let rng = ring::rand::SystemRandom::new();
        let pkcs8 = EcdsaKeyPair::generate_pkcs8(&ECDSA_P256_SHA256_FIXED_SIGNING, &rng).unwrap();
        let pem = format!(
            "-----BEGIN PRIVATE KEY-----\n{}\n-----END PRIVATE KEY-----\n",
            base64::engine::general_purpose::STANDARD.encode(pkcs8.as_ref())
        );
        let jwt = client_assertion("agent", "https://as.example/r", "ES256", &pem, 1000).unwrap();
        let parts: Vec<&str> = jwt.split('.').collect();
        let dec = |p: &str| {
            base64::engine::general_purpose::URL_SAFE_NO_PAD
                .decode(p)
                .unwrap()
        };
        let claims: serde_json::Value = serde_json::from_slice(&dec(parts[1])).unwrap();
        assert_eq!(claims["iss"], "agent");
        assert_eq!(claims["sub"], "agent");
        assert_eq!(claims["aud"], "https://as.example/r");
        assert_eq!(claims["exp"], 1060);
        let pair = EcdsaKeyPair::from_pkcs8(&ECDSA_P256_SHA256_FIXED_SIGNING, pkcs8.as_ref(), &rng)
            .unwrap();
        UnparsedPublicKey::new(&ECDSA_P256_SHA256_FIXED, pair.public_key().as_ref())
            .verify(
                format!("{}.{}", parts[0], parts[1]).as_bytes(),
                &dec(parts[2]),
            )
            .expect("the signature verifies under the key's public half");
        let e = client_assertion(
            "agent",
            "i",
            "ES256",
            "-----BEGIN PRIVATE KEY-----\nAAAA\n-----END PRIVATE KEY-----",
            0,
        )
        .unwrap_err();
        assert!(!e.contains("AAAA"), "{e}");
    }

    type Log = std::sync::Arc<std::sync::Mutex<Vec<String>>>;

    /// A loopback server answering by path prefix (`{base}` in a body is its own origin), 404
    /// otherwise, logging each request (head and body, lower-cased).
    async fn server(routes: Vec<(&'static str, String)>) -> (String, Log) {
        let listener = beyond_test_support::ports::tokio_listener().await;
        let base = format!("http://{}", listener.local_addr().unwrap());
        let seen = Log::default();
        let (log, origin) = (seen.clone(), base.clone());
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                let (log, routes, origin) = (log.clone(), routes.clone(), origin.clone());
                tokio::spawn(async move {
                    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
                    let mut buf = vec![0u8; 65536];
                    let n = stream.read(&mut buf).await.unwrap_or(0);
                    let req = String::from_utf8_lossy(&buf[..n]).to_ascii_lowercase();
                    log.lock().unwrap().push(req.clone());
                    let path = req.split_whitespace().nth(1).unwrap_or_default().to_owned();
                    let (status, body) = routes
                        .iter()
                        .find(|(p, _)| path.starts_with(p))
                        .map_or(("404 Not Found", String::new()), |(_, b)| {
                            ("200 OK", b.replace("{base}", &origin))
                        });
                    let reply = format!(
                        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = stream.write_all(reply.as_bytes()).await;
                });
            }
        });
        (base, seen)
    }

    /// A resource server that names `attacker` as its authorization server, and that authorization
    /// server (metadata naming itself as issuer, and a token endpoint that would issue anything).
    async fn hostile_pair() -> (String, String, Log) {
        let metadata = r#"{"issuer":"{base}","authorization_endpoint":"{base}/authorize","token_endpoint":"{base}/token","token_endpoint_auth_methods_supported":["client_secret_post"]}"#;
        let (attacker, log) = server(vec![
            ("/.well-known/", metadata.into()),
            (
                "/token",
                r#"{"access_token":"stolen","expires_in":60}"#.into(),
            ),
        ])
        .await;
        let prm = r#"{"resource":"{base}/mcp","authorization_servers":["ATTACKER"]}"#
            .replace("ATTACKER", &attacker);
        let (rs, _) = server(vec![("/.well-known/oauth-protected-resource", prm)]).await;
        (format!("{rs}/mcp"), attacker, log)
    }

    fn secret_cc(issuer: &str) -> McpClientCredentials {
        McpClientCredentials {
            issuer: issuer.into(),
            client_id: "agent".into(),
            client_secret: Some("!echo s3cr3t-value".into()),
            private_key: None,
            signing_algorithm: None,
            scopes: vec![],
        }
    }

    #[tokio::test]
    async fn a_server_naming_another_authorization_server_never_receives_the_credential() {
        let (url, attacker, log) = hostile_pair().await;
        let err = acquire("c", &url, &secret_cc("http://127.0.0.1:9"), &["s".into()])
            .await
            .err()
            .expect("refused");
        assert!(
            err.definitive && err.reason.contains("not the pinned issuer"),
            "{}",
            err.reason
        );
        let seen = log.lock().unwrap().join("\n");
        assert!(
            !seen.contains("s3cr3t-value") && !seen.contains("client_secret"),
            "{seen}"
        );
        assert!(
            !seen.contains("post /token"),
            "no token request at all: {seen}"
        );
        // The control that makes the test able to fail: pinned to that authorization server, the
        // very same flow does send it the credential.
        acquire("c", &url, &secret_cc(&attacker), &["s".into()])
            .await
            .expect("the pinned issuer is trusted");
        assert!(
            log.lock()
                .unwrap()
                .join("\n")
                .contains("client_secret=s3cr3t-value")
        );
    }
}
