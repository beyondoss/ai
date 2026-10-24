//! Verifying a JWT the way a resource server must: the signature against the provider's JWKS (RS256
//! or ES256, and nothing else: no `none`, no HMAC), then issuer, audience and expiry. Scope is the
//! caller's business (it differs per tool), so the claims come back for it to check.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use ring::signature::{
    ECDSA_P256_SHA256_FIXED, RSA_PKCS1_2048_8192_SHA256, RsaPublicKeyComponents, UnparsedPublicKey,
};
use serde_json::Value;

use crate::http::{Res, request};

#[derive(Debug, PartialEq, Eq)]
pub enum JwtError {
    Malformed,
    /// An algorithm this resource does not accept.
    Alg,
    UnknownKid,
    BadSignature,
    Issuer,
    Audience,
    Expired,
    NotYet,
}

impl std::fmt::Display for JwtError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Malformed => "malformed token",
            Self::Alg => "unsupported algorithm",
            Self::UnknownKid => "unknown signing key",
            Self::BadSignature => "bad signature",
            Self::Issuer => "wrong issuer",
            Self::Audience => "wrong audience",
            Self::Expired => "token expired",
            Self::NotYet => "token not yet valid",
        })
    }
}

enum Key {
    Rsa {
        n: Vec<u8>,
        e: Vec<u8>,
    },
    /// An uncompressed P-256 point.
    Ec(Vec<u8>),
}

struct Jwk {
    kid: Option<String>,
    key: Key,
}

pub struct Jwks(Vec<Jwk>);

fn b64(v: &Value) -> Option<Vec<u8>> {
    URL_SAFE_NO_PAD.decode(v.as_str()?).ok()
}

impl Jwks {
    pub fn parse(json: &[u8]) -> Option<Self> {
        let v: Value = serde_json::from_slice(json).ok()?;
        let keys = v["keys"]
            .as_array()?
            .iter()
            .filter(|k| k["use"].as_str().is_none_or(|u| u == "sig"))
            .filter_map(|k| {
                let key = match k["kty"].as_str()? {
                    "RSA" => Key::Rsa {
                        n: b64(&k["n"])?,
                        e: b64(&k["e"])?,
                    },
                    "EC" if k["crv"] == "P-256" => {
                        let mut p = vec![4u8];
                        p.extend(b64(&k["x"])?);
                        p.extend(b64(&k["y"])?);
                        Key::Ec(p)
                    }
                    _ => return None,
                };
                Some(Jwk {
                    kid: k["kid"].as_str().map(str::to_owned),
                    key,
                })
            })
            .collect();
        Some(Self(keys))
    }

    fn has(&self, kid: Option<&str>) -> bool {
        self.0
            .iter()
            .any(|k| kid.is_none() || k.kid.as_deref() == kid)
    }
}

/// What a token must be: from this issuer, for this audience.
pub struct Expect<'a> {
    pub issuer: &'a str,
    pub audience: &'a str,
}

/// The header's `kid`, without trusting anything else in the token.
fn header_kid(token: &str) -> Option<Option<String>> {
    let h = token.split('.').next()?;
    let h: Value = serde_json::from_slice(&URL_SAFE_NO_PAD.decode(h).ok()?).ok()?;
    Some(h["kid"].as_str().map(str::to_owned))
}

/// Verify `token` against `jwks` at `now` (unix seconds); its claims if it passes.
pub fn verify(token: &str, jwks: &Jwks, expect: &Expect<'_>, now: u64) -> Result<Value, JwtError> {
    let mut parts = token.split('.');
    let (Some(h), Some(p), Some(s), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return Err(JwtError::Malformed);
    };
    let header: Value =
        serde_json::from_slice(&URL_SAFE_NO_PAD.decode(h).map_err(|_| JwtError::Malformed)?)
            .map_err(|_| JwtError::Malformed)?;
    let sig = URL_SAFE_NO_PAD.decode(s).map_err(|_| JwtError::Malformed)?;
    let alg = header["alg"].as_str().unwrap_or("");
    if alg != "RS256" && alg != "ES256" {
        return Err(JwtError::Alg);
    }
    let kid = header["kid"].as_str();
    let signed = &token.as_bytes()[..h.len() + 1 + p.len()];
    let mut known = false;
    let ok = jwks
        .0
        .iter()
        .filter(|k| kid.is_none() || k.kid.as_deref() == kid)
        .any(|k| {
            known = true;
            match (&k.key, alg) {
                (Key::Rsa { n, e }, "RS256") => RsaPublicKeyComponents { n, e }
                    .verify(&RSA_PKCS1_2048_8192_SHA256, signed, &sig)
                    .is_ok(),
                (Key::Ec(point), "ES256") => {
                    UnparsedPublicKey::new(&ECDSA_P256_SHA256_FIXED, point)
                        .verify(signed, &sig)
                        .is_ok()
                }
                _ => false,
            }
        });
    if !known {
        return Err(JwtError::UnknownKid);
    }
    if !ok {
        return Err(JwtError::BadSignature);
    }
    let claims: Value =
        serde_json::from_slice(&URL_SAFE_NO_PAD.decode(p).map_err(|_| JwtError::Malformed)?)
            .map_err(|_| JwtError::Malformed)?;
    if claims["iss"].as_str() != Some(expect.issuer) {
        return Err(JwtError::Issuer);
    }
    let aud_ok = match &claims["aud"] {
        Value::String(a) => a == expect.audience,
        Value::Array(a) => a.iter().any(|a| a.as_str() == Some(expect.audience)),
        _ => false,
    };
    if !aud_ok {
        return Err(JwtError::Audience);
    }
    // No `exp` is not "never expires": a token that cannot say when it ends is refused.
    match claims["exp"].as_u64() {
        Some(exp) if exp > now => {}
        _ => return Err(JwtError::Expired),
    }
    if claims["nbf"].as_u64().is_some_and(|nbf| nbf > now + 5) {
        return Err(JwtError::NotYet);
    }
    Ok(claims)
}

/// The scopes in a token's `scope` claim.
pub fn scopes(claims: &Value) -> Vec<&str> {
    claims["scope"]
        .as_str()
        .unwrap_or("")
        .split_whitespace()
        .collect()
}

/// A provider's JWKS, fetched when first needed and again when a token names a key it does not
/// hold (a rotation) or after a while. Never more often than every ten seconds.
pub struct JwksCache {
    addr: String,
    path: String,
    state: Mutex<(Option<std::sync::Arc<Jwks>>, Option<Instant>)>,
}

const JWKS_TTL: Duration = Duration::from_secs(300);
const JWKS_MIN_GAP: Duration = Duration::from_secs(10);

impl JwksCache {
    pub fn new(addr: &str, path: &str) -> Self {
        Self {
            addr: addr.into(),
            path: path.into(),
            state: Mutex::new((None, None)),
        }
    }

    /// The keys to check `token` against.
    pub async fn for_token(&self, token: &str) -> Res<std::sync::Arc<Jwks>> {
        let kid = header_kid(token).ok_or("malformed token")?;
        let (cached, fetched) = {
            let st = self.state.lock().map_err(|_| "jwks lock poisoned")?;
            (st.0.clone(), st.1)
        };
        if let (Some(c), Some(at)) = (&cached, fetched) {
            let fresh = at.elapsed() < JWKS_TTL;
            if (fresh && c.has(kid.as_deref())) || at.elapsed() < JWKS_MIN_GAP {
                return Ok(c.clone());
            }
        }
        let (status, _, body) = request("GET", &self.addr, &self.path, &[], &[]).await?;
        if status != 200 {
            return Err(format!("jwks fetch answered {status}").into());
        }
        let jwks = std::sync::Arc::new(Jwks::parse(&body).ok_or("jwks is not a key set")?);
        let mut st = self.state.lock().map_err(|_| "jwks lock poisoned")?;
        *st = (Some(jwks.clone()), Some(Instant::now()));
        Ok(jwks)
    }
}

#[cfg(test)]
pub mod testing {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use super::*;
    use ring::rand::SystemRandom;
    use ring::signature::{ECDSA_P256_SHA256_FIXED_SIGNING, EcdsaKeyPair, KeyPair};
    use serde_json::json;

    /// An ES256 signing key and the JWKS that verifies it.
    pub struct TestIdp {
        pair: EcdsaKeyPair,
        pub kid: String,
    }

    impl TestIdp {
        pub fn new(kid: &str) -> Self {
            let rng = SystemRandom::new();
            let pkcs8 =
                EcdsaKeyPair::generate_pkcs8(&ECDSA_P256_SHA256_FIXED_SIGNING, &rng).unwrap();
            let pair =
                EcdsaKeyPair::from_pkcs8(&ECDSA_P256_SHA256_FIXED_SIGNING, pkcs8.as_ref(), &rng)
                    .unwrap();
            Self {
                pair,
                kid: kid.into(),
            }
        }

        pub fn jwks_json(&self) -> String {
            let p = self.pair.public_key().as_ref();
            json!({"keys": [{"kty": "EC", "crv": "P-256", "use": "sig", "alg": "ES256", "kid": self.kid,
                "x": URL_SAFE_NO_PAD.encode(&p[1..33]), "y": URL_SAFE_NO_PAD.encode(&p[33..65])}]})
            .to_string()
        }

        pub fn sign(&self, claims: &Value) -> String {
            self.sign_with(json!({"alg": "ES256", "kid": self.kid}), claims)
        }

        pub fn sign_with(&self, header: Value, claims: &Value) -> String {
            let signed = format!(
                "{}.{}",
                URL_SAFE_NO_PAD.encode(header.to_string()),
                URL_SAFE_NO_PAD.encode(claims.to_string())
            );
            let sig = self
                .pair
                .sign(&SystemRandom::new(), signed.as_bytes())
                .unwrap();
            format!("{signed}.{}", URL_SAFE_NO_PAD.encode(sig.as_ref()))
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use super::testing::TestIdp;
    use super::*;
    use serde_json::json;

    const ISS: &str = "http://idp.test/realms/demo";
    const NOW: u64 = 1_000_000;

    fn claims() -> Value {
        json!({"iss": ISS, "aud": "notes-api", "sub": "u1", "exp": NOW + 60, "scope": "openid notes:read"})
    }

    fn check(idp: &TestIdp, token: &str) -> Result<Value, JwtError> {
        let jwks = Jwks::parse(idp.jwks_json().as_bytes()).unwrap();
        verify(
            token,
            &jwks,
            &Expect {
                issuer: ISS,
                audience: "notes-api",
            },
            NOW,
        )
    }

    #[test]
    fn a_good_token_yields_its_claims() {
        let idp = TestIdp::new("k1");
        let c = check(&idp, &idp.sign(&claims())).unwrap();
        assert_eq!(c["sub"], "u1");
        assert_eq!(scopes(&c), ["openid", "notes:read"]);
    }

    #[test]
    fn every_way_to_be_wrong_is_refused_for_its_own_reason() {
        let idp = TestIdp::new("k1");
        let other = TestIdp::new("k1");
        let with = |f: fn(&mut Value)| {
            let mut c = claims();
            f(&mut c);
            idp.sign(&c)
        };
        assert_eq!(
            check(&idp, &other.sign(&claims())),
            Err(JwtError::BadSignature)
        );
        assert_eq!(
            check(
                &idp,
                &idp.sign_with(json!({"alg": "ES256", "kid": "zzz"}), &claims())
            ),
            Err(JwtError::UnknownKid)
        );
        assert_eq!(
            check(&idp, &idp.sign_with(json!({"alg": "none"}), &claims())),
            Err(JwtError::Alg)
        );
        assert_eq!(
            check(
                &idp,
                &idp.sign_with(json!({"alg": "HS256", "kid": "k1"}), &claims())
            ),
            Err(JwtError::Alg)
        );
        assert_eq!(
            check(&idp, &with(|c| c["iss"] = json!("http://evil"))),
            Err(JwtError::Issuer)
        );
        assert_eq!(
            check(&idp, &with(|c| c["aud"] = json!("other-api"))),
            Err(JwtError::Audience)
        );
        assert_eq!(
            check(&idp, &with(|c| c["aud"] = json!(["x", "notes-api"]))).map(|_| ()),
            Ok(())
        );
        assert_eq!(
            check(&idp, &with(|c| c["exp"] = json!(NOW))),
            Err(JwtError::Expired)
        );
        assert_eq!(
            check(
                &idp,
                &with(|c| {
                    c.as_object_mut().unwrap().remove("exp");
                })
            ),
            Err(JwtError::Expired)
        );
        assert_eq!(
            check(&idp, &with(|c| c["nbf"] = json!(NOW + 600))),
            Err(JwtError::NotYet)
        );
        assert_eq!(check(&idp, "not.a.jwt.at.all"), Err(JwtError::Malformed));
        // a tampered payload under the original signature
        let good = idp.sign(&claims());
        let mut p = good.split('.').map(str::to_owned).collect::<Vec<_>>();
        p[1] = URL_SAFE_NO_PAD.encode(
            json!({"iss": ISS, "aud": "notes-api", "sub": "u2", "exp": NOW + 60}).to_string(),
        );
        assert_eq!(check(&idp, &p.join(".")), Err(JwtError::BadSignature));
    }
}
