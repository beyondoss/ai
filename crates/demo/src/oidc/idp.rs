//! The broker's side of an OpenID Connect provider: discovery, the authorization-code exchange with
//! PKCE, refresh, and revocation (RFC 7009). The provider's own address (what it calls itself in
//! tokens and in the browser's redirects) and the address the broker reaches it at are two things:
//! discovery gives the paths, and requests go to `internal`.

use base64::Engine as _;
use base64::engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::http::{Res, form_body, now_ms, request};

pub struct Idp {
    /// `host:port` the broker reaches the provider at.
    pub internal: String,
    pub issuer: String,
    /// Where the user's browser is sent (the provider's own name for itself).
    pub authorization_endpoint: String,
    token_path: String,
    revocation_path: String,
    pub jwks_path: String,
    client_id: String,
    client_secret: String,
}

pub struct Tokens {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub id_token: Option<String>,
    pub expires_in: u64,
    pub scope: String,
}

#[derive(Debug)]
pub enum IdpError {
    /// The provider says this grant is dead: revoked, expired, or never valid.
    InvalidGrant(String),
    /// Anything else: the provider is down, or answered nonsense.
    Unavailable(String),
}

impl std::fmt::Display for IdpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidGrant(m) => write!(f, "invalid_grant: {m}"),
            Self::Unavailable(m) => write!(f, "identity provider unavailable: {m}"),
        }
    }
}

/// `https://host:port/path?q` -> (`host:port`, `/path?q`).
fn split_url(url: &str) -> Option<(&str, &str)> {
    let rest = url.split_once("://")?.1;
    Some(match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    })
}

impl Idp {
    /// Read the provider's discovery document (`issuer` + `/.well-known/openid-configuration`), and
    /// keep the paths of its token, revocation and key endpoints.
    pub async fn discover(
        internal: &str,
        issuer: &str,
        client_id: &str,
        client_secret: &str,
    ) -> Res<Self> {
        let path = format!(
            "{}/.well-known/openid-configuration",
            split_url(issuer).ok_or("bad issuer")?.1
        );
        let (status, _, body) = request("GET", internal, &path, &[], &[]).await?;
        if status != 200 {
            return Err(format!("discovery answered {status}").into());
        }
        let d: Value = serde_json::from_slice(&body)?;
        if d["issuer"].as_str() != Some(issuer) {
            return Err(
                "the provider calls itself something other than the configured issuer".into(),
            );
        }
        let path_of = |k: &str| -> Res<String> {
            Ok(split_url(
                d[k].as_str()
                    .ok_or_else(|| format!("discovery has no {k}"))?,
            )
            .ok_or("bad endpoint")?
            .1
            .to_string())
        };
        Ok(Self {
            internal: internal.into(),
            issuer: issuer.into(),
            authorization_endpoint: d["authorization_endpoint"]
                .as_str()
                .ok_or("discovery has no authorization_endpoint")?
                .into(),
            token_path: path_of("token_endpoint")?,
            revocation_path: path_of("revocation_endpoint")?,
            jwks_path: path_of("jwks_uri")?,
            client_id: client_id.into(),
            client_secret: client_secret.into(),
        })
    }

    pub fn client_id(&self) -> &str {
        &self.client_id
    }

    fn basic(&self) -> (String, String) {
        (
            "authorization".into(),
            format!(
                "Basic {}",
                STANDARD.encode(format!(
                    "{}:{}",
                    form_enc(&self.client_id),
                    form_enc(&self.client_secret)
                ))
            ),
        )
    }

    async fn form(&self, path: &str, pairs: &[(&str, &str)]) -> Result<(u16, Value), IdpError> {
        let headers = [
            self.basic(),
            (
                "content-type".to_string(),
                "application/x-www-form-urlencoded".to_string(),
            ),
        ];
        let (status, _, body) = request("POST", &self.internal, path, &headers, &form_body(pairs))
            .await
            .map_err(|e| IdpError::Unavailable(e.to_string()))?;
        let v = serde_json::from_slice(&body).unwrap_or(Value::Null);
        Ok((status, v))
    }

    fn tokens(status: u16, v: &Value) -> Result<Tokens, IdpError> {
        if status == 400 || status == 401 {
            let why = v["error_description"]
                .as_str()
                .or(v["error"].as_str())
                .unwrap_or("rejected");
            return Err(IdpError::InvalidGrant(why.to_string()));
        }
        if status != 200 {
            return Err(IdpError::Unavailable(format!(
                "token endpoint answered {status}"
            )));
        }
        let take = |k: &str| v[k].as_str().map(str::to_owned);
        Ok(Tokens {
            access_token: take("access_token")
                .ok_or_else(|| IdpError::Unavailable("no access_token".into()))?,
            refresh_token: take("refresh_token"),
            id_token: take("id_token"),
            expires_in: v["expires_in"].as_u64().unwrap_or(60),
            scope: take("scope").unwrap_or_default(),
        })
    }

    /// The authorization-code grant, with the PKCE verifier the code was challenged with.
    pub async fn exchange(
        &self,
        code: &str,
        verifier: &str,
        redirect_uri: &str,
    ) -> Result<Tokens, IdpError> {
        let (s, v) = self
            .form(
                &self.token_path,
                &[
                    ("grant_type", "authorization_code"),
                    ("code", code),
                    ("code_verifier", verifier),
                    ("redirect_uri", redirect_uri),
                ],
            )
            .await?;
        Self::tokens(s, &v)
    }

    pub async fn refresh(&self, refresh_token: &str) -> Result<Tokens, IdpError> {
        let (s, v) = self
            .form(
                &self.token_path,
                &[
                    ("grant_type", "refresh_token"),
                    ("refresh_token", refresh_token),
                ],
            )
            .await?;
        Self::tokens(s, &v)
    }

    /// RFC 7009: ask the provider to forget a refresh token. A provider answers 200 whether or not it
    /// knew the token, so success here is "asked", and [`Self::refresh`] is what proves it dead.
    pub async fn revoke(&self, refresh_token: &str) -> Result<(), IdpError> {
        let (s, _) = self
            .form(
                &self.revocation_path,
                &[
                    ("token", refresh_token),
                    ("token_type_hint", "refresh_token"),
                ],
            )
            .await?;
        if s == 200 {
            Ok(())
        } else {
            Err(IdpError::Unavailable(format!(
                "revocation endpoint answered {s}"
            )))
        }
    }
}

fn form_enc(s: &str) -> String {
    crate::http::pct_encode(s)
}

/// A fresh PKCE verifier and its S256 challenge.
pub fn pkce() -> Option<(String, String)> {
    let mut raw = [0u8; 48];
    getrandom::fill(&mut raw).ok()?;
    let verifier = URL_SAFE_NO_PAD.encode(raw);
    let challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
    Some((verifier, challenge))
}

/// `n` random bytes as URL-safe text.
pub fn random_token(n: usize) -> Option<String> {
    let mut raw = vec![0u8; n];
    getrandom::fill(&mut raw).ok()?;
    Some(URL_SAFE_NO_PAD.encode(raw))
}

/// When an access token that lives `expires_in` seconds from now stops being usable, in ms.
pub fn expiry_ms(expires_in: u64) -> u64 {
    now_ms() + expires_in * 1000
}
