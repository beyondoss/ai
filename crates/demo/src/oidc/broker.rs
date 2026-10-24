//! The broker: a managed-MCP endpoint that acts as a signed-in user.
//!
//! What it holds, per **connection** (one user's grant of one connector to one tenant): the user's
//! refresh token, sealed at rest under the tenant's key ([`super::vault`]), and an access token in
//! memory only. What a session holds: a `bmc_v1` token, and a **binding** the control plane made for
//! it, `(tenant, session, connector) -> connection`. A request is therefore authenticated twice: the
//! token proves *which session* is calling, the binding says *whose connection* that session may use.
//! Nothing in the request can name another connection; one that tries is refused.
//!
//! Flows:
//! - **consent** (`POST /connect/start`, the user's browser, `GET /oauth/callback`): authorization
//!   code + PKCE (S256) against the provider; the callback exchanges the code, validates the ID token
//!   (signature, issuer, audience, nonce), and seals the refresh token.
//! - **bind** (`POST /sessions/bind`): the session may use the connection, if both are the same tenant's.
//! - **call** (`POST /managed/<connector>/mcp`): verify the token, find the binding, get a live access
//!   token (refreshing it here when it is about to lapse), forward with `Authorization: Bearer <access>`.
//! - **revoke** (`POST /connections/<id>/revoke`): RFC 7009 at the provider, then the stored token is
//!   erased; the next call is refused with a plain message and the session carries on.
//!
//! The control endpoints answer only the admin bearer (the control plane's credential), except the
//! OAuth callback, which a browser reaches with a one-time `state`.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use beyond_ai_agent::managed_mcp_token::{ManagedKey, Reject};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::Mutex as AsyncMutex;

use super::idp::{Idp, IdpError, Tokens, expiry_ms, pkce, random_token};
use super::jwt::{self, Expect, JwksCache};
use super::vault::Vault;
use crate::http::{
    Headers, Req, constant_time_eq, error_body, form_body, json_headers, now_ms, post,
};

/// A token is refreshed when it has less than this left, so a call never meets one mid-expiry.
const SKEW_MS: u64 = 10_000;
const PENDING_TTL_MS: u64 = 10 * 60 * 1000;
const PENDING_MAX: usize = 1000;

pub struct Config {
    /// Where the resource server listens (loopback); the broker forwards to it.
    pub resource_addr: String,
    /// The callback the provider redirects the browser to.
    pub redirect_uri: String,
    /// The control plane's credential.
    pub admin: String,
    /// Where connection records are kept.
    pub data_dir: PathBuf,
    /// The scopes each connector asks for (besides `openid offline_access`).
    pub connectors: BTreeMap<String, Vec<String>>,
}

#[derive(Clone, Serialize, Deserialize)]
struct Record {
    id: String,
    tenant: String,
    connector: String,
    sub: String,
    username: String,
    /// What the provider actually granted.
    scope: String,
    created_ms: u64,
    /// `active`, `revoked` (the user withdrew it here) or `reauth_required` (the provider stopped
    /// honouring it).
    status: String,
    /// The refresh token, sealed. Gone once the connection is not active.
    refresh_sealed: Option<String>,
    refreshes: u64,
}

struct Access {
    token: String,
    exp_ms: u64,
}

struct Conn {
    rec: Record,
    access: Option<Access>,
}

struct Pending {
    tenant: String,
    connector: String,
    verifier: String,
    nonce: String,
    created_ms: u64,
}

/// The tail of a token: enough for a scan to recognise it, not enough to use it.
#[derive(Clone, Serialize)]
struct Fingerprint {
    connection: String,
    kind: &'static str,
    tail: String,
}

#[derive(Default)]
struct State {
    pending: HashMap<String, Pending>,
    conns: HashMap<String, Arc<AsyncMutex<Conn>>>,
    /// `tenant \0 session \0 connector` -> connection id.
    bindings: BTreeMap<String, String>,
    /// What each connector's `tools/list` last said: the catalog is not the user's data, and it lets
    /// a session whose connection is gone still see what the tools are.
    catalog: HashMap<String, Value>,
    fingerprints: Vec<Fingerprint>,
}

pub struct Broker {
    cfg: Config,
    idp: Idp,
    vault: Vault,
    key: ManagedKey,
    jwks: JwksCache,
    state: Mutex<State>,
}

type Resp = (u16, Headers, Vec<u8>);

fn line(event: &str, detail: Value) {
    println!(
        "{}",
        json!({"ts_ms": now_ms(), "src": "broker", "event": event, "detail": detail})
    );
}

fn reply(status: u16, v: Value) -> Resp {
    (status, json_headers(), v.to_string().into_bytes())
}

fn tail(token: &str) -> String {
    token
        .chars()
        .rev()
        .take(24)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
}

fn bind_key(tenant: &str, session: &str, connector: &str) -> String {
    format!("{tenant}\0{session}\0{connector}")
}

/// Why a connection cannot supply an access token.
enum Refusal {
    /// The user withdrew it, or the provider no longer honours it.
    Gone(String),
    Unavailable(String),
}

impl Broker {
    pub fn new(cfg: Config, idp: Idp, vault: Vault, key: ManagedKey) -> Arc<Self> {
        let jwks = JwksCache::new(&idp.internal, &idp.jwks_path);
        let b = Arc::new(Self {
            cfg,
            idp,
            vault,
            key,
            jwks,
            state: Mutex::default(),
        });
        b.load();
        b
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    // ---- storage -------------------------------------------------------------------------------

    fn path(&self, name: &str) -> PathBuf {
        self.cfg.data_dir.join(name)
    }

    /// Write a file whole or not at all: a temporary beside it, then a rename.
    fn write_atomic(&self, name: &str, bytes: &[u8]) {
        let tmp = self.path(&format!("{name}.tmp"));
        let ok = std::fs::create_dir_all(&self.cfg.data_dir).is_ok()
            && std::fs::write(&tmp, bytes).is_ok()
            && std::fs::rename(&tmp, self.path(name)).is_ok();
        if !ok {
            line("store_error", json!({"file": name}));
        }
    }

    fn persist(&self, rec: &Record) {
        if let Ok(b) = serde_json::to_vec(rec) {
            self.write_atomic(&format!("{}.json", rec.id), &b);
        }
    }

    fn persist_bindings(&self, bindings: &BTreeMap<String, String>) {
        if let Ok(b) = serde_json::to_vec(bindings) {
            self.write_atomic("bindings.state", &b);
        }
    }

    /// Pick up what an earlier run of this broker stored.
    fn load(&self) {
        let Ok(dir) = std::fs::read_dir(&self.cfg.data_dir) else {
            return;
        };
        let mut st = self.lock();
        for e in dir.flatten() {
            let p = e.path();
            if p.extension().is_some_and(|x| x == "json")
                && let Ok(rec) = std::fs::read(&p)
                    .map_err(|_| ())
                    .and_then(|b| serde_json::from_slice::<Record>(&b).map_err(|_| ()))
            {
                st.conns.insert(
                    rec.id.clone(),
                    Arc::new(AsyncMutex::new(Conn { rec, access: None })),
                );
            }
        }
        if let Ok(b) = std::fs::read(self.path("bindings.state"))
            && let Ok(m) = serde_json::from_slice(&b)
        {
            st.bindings = m;
        }
    }

    // ---- the control plane ---------------------------------------------------------------------

    pub async fn control(&self, req: &Req) -> Resp {
        let route = req.route().to_string();
        if route == "/healthz" {
            return reply(200, json!({"ok": true}));
        }
        if route == "/oauth/callback" && req.method == "GET" {
            return self.callback(req).await;
        }
        let authed = req
            .header("authorization")
            .and_then(|v| v.strip_prefix("Bearer "))
            .is_some_and(|t| constant_time_eq(t.as_bytes(), self.cfg.admin.as_bytes()));
        if !authed {
            return reply(
                401,
                json!({"error": "the control plane's credential is required"}),
            );
        }
        let body: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
        let seg: Vec<&str> = route.trim_matches('/').split('/').collect();
        match (req.method.as_str(), seg.as_slice()) {
            ("POST", ["connect", "start"]) => self.connect_start(&body),
            ("POST", ["sessions", "bind"]) => self.bind(&body),
            ("GET", ["connections", id]) => {
                self.connection_view(id, req.query("audit").is_some()).await
            }
            ("GET", ["connections", id, "fingerprints"]) => self.fingerprints(id),
            ("POST", ["connections", id, "revoke"]) => self.revoke(id).await,
            _ => reply(404, json!({"error": "not found"})),
        }
    }

    fn connect_start(&self, body: &Value) -> Resp {
        let (Some(tenant), Some(connector)) = (body["tenant"].as_str(), body["connector"].as_str())
        else {
            return reply(400, json!({"error": "tenant and connector are required"}));
        };
        let Some(default_scopes) = self.cfg.connectors.get(connector) else {
            return reply(404, json!({"error": format!("no connector {connector}")}));
        };
        let scopes: Vec<String> = match body["scopes"].as_array() {
            Some(a) => a
                .iter()
                .filter_map(|s| s.as_str().map(str::to_owned))
                .collect(),
            None => default_scopes.clone(),
        };
        let (Some((verifier, challenge)), Some(state), Some(nonce)) =
            (pkce(), random_token(24), random_token(16))
        else {
            return reply(500, json!({"error": "no randomness"}));
        };
        let scope = format!("openid offline_access {}", scopes.join(" "));
        let url = format!(
            "{}?{}",
            self.idp.authorization_endpoint,
            String::from_utf8_lossy(&form_body(&[
                ("response_type", "code"),
                ("client_id", self.idp.client_id()),
                ("redirect_uri", &self.cfg.redirect_uri),
                ("scope", &scope),
                ("state", &state),
                ("nonce", &nonce),
                ("code_challenge", &challenge),
                ("code_challenge_method", "S256"),
            ]))
        );
        let now = now_ms();
        let mut st = self.lock();
        st.pending
            .retain(|_, p| now - p.created_ms < PENDING_TTL_MS);
        if st.pending.len() >= PENDING_MAX {
            return reply(429, json!({"error": "too many consents in flight"}));
        }
        st.pending.insert(
            state.clone(),
            Pending {
                tenant: tenant.into(),
                connector: connector.into(),
                verifier,
                nonce,
                created_ms: now,
            },
        );
        drop(st);
        line(
            "consent_started",
            json!({"tenant": tenant, "connector": connector, "scope": scope, "pkce": "S256"}),
        );
        reply(
            200,
            json!({"authorize_url": url, "state": state, "redirect_uri": self.cfg.redirect_uri}),
        )
    }

    async fn callback(&self, req: &Req) -> Resp {
        let Some(state) = req.query("state") else {
            return reply(400, json!({"error": "no state"}));
        };
        // One time: a replayed or forged state finds nothing.
        let pending = self.lock().pending.remove(&state);
        let Some(p) = pending.filter(|p| now_ms() - p.created_ms < PENDING_TTL_MS) else {
            line("consent_refused", json!({"why": "unknown or used state"}));
            return reply(
                400,
                json!({"error": "unknown, expired or already used state"}),
            );
        };
        if let Some(err) = req.query("error") {
            let why = req.query("error_description").unwrap_or_default();
            line(
                "consent_declined",
                json!({"tenant": p.tenant, "error": err}),
            );
            return reply(400, json!({"error": err, "error_description": why}));
        }
        let Some(code) = req.query("code") else {
            return reply(400, json!({"error": "no code"}));
        };
        let tokens = match self
            .idp
            .exchange(&code, &p.verifier, &self.cfg.redirect_uri)
            .await
        {
            Ok(t) => t,
            Err(e) => {
                line("consent_failed", json!({"error": e.to_string()}));
                let status = if matches!(e, IdpError::InvalidGrant(_)) {
                    400
                } else {
                    502
                };
                return reply(status, json!({"error": e.to_string()}));
            }
        };
        match self.complete(&p, tokens).await {
            Ok(v) => reply(200, v),
            Err(why) => {
                line("consent_failed", json!({"error": why}));
                reply(400, json!({"error": why}))
            }
        }
    }

    /// The code is exchanged: check who signed in, and store what the connection needs.
    async fn complete(&self, p: &Pending, t: Tokens) -> Result<Value, String> {
        let id_token = t
            .id_token
            .as_deref()
            .ok_or("the provider issued no ID token")?;
        let jwks = self
            .jwks
            .for_token(id_token)
            .await
            .map_err(|e| e.to_string())?;
        let claims = jwt::verify(
            id_token,
            &jwks,
            &Expect {
                issuer: &self.idp.issuer,
                audience: self.idp.client_id(),
            },
            now_ms() / 1000,
        )
        .map_err(|e| format!("ID token: {e}"))?;
        if claims["nonce"].as_str() != Some(p.nonce.as_str()) {
            return Err("ID token: wrong nonce".into());
        }
        let sub = claims["sub"]
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or("ID token has no subject")?;
        let username = claims["preferred_username"].as_str().unwrap_or(sub);
        let refresh = t
            .refresh_token
            .as_deref()
            .ok_or("the provider issued no refresh token (offline access was not granted)")?;

        // One connection per user per connector per tenant: signing in again renews it.
        let all: Vec<_> = self.lock().conns.values().cloned().collect();
        let mut existing = None;
        for c in all {
            let c = c.lock().await;
            if c.rec.tenant == p.tenant && c.rec.connector == p.connector && c.rec.sub == sub {
                existing = Some(c.rec.id.clone());
            }
        }
        let id = match existing {
            Some(id) => id,
            None => format!("conn_{}", random_token(9).ok_or("no randomness")?),
        };
        let sealed = self
            .vault
            .seal(&p.tenant, &id, refresh.as_bytes())
            .ok_or("could not seal the refresh token")?;
        let rec = Record {
            id: id.clone(),
            tenant: p.tenant.clone(),
            connector: p.connector.clone(),
            sub: sub.into(),
            username: username.into(),
            scope: t.scope.clone(),
            created_ms: now_ms(),
            status: "active".into(),
            refresh_sealed: Some(sealed.clone()),
            refreshes: 0,
        };
        self.persist(&rec);
        let conn = Arc::new(AsyncMutex::new(Conn {
            rec,
            access: Some(Access {
                token: t.access_token.clone(),
                exp_ms: expiry_ms(t.expires_in),
            }),
        }));
        let mut st = self.lock();
        st.conns.insert(id.clone(), conn);
        st.fingerprints.push(Fingerprint {
            connection: id.clone(),
            kind: "access",
            tail: tail(&t.access_token),
        });
        st.fingerprints.push(Fingerprint {
            connection: id.clone(),
            kind: "refresh",
            tail: tail(refresh),
        });
        drop(st);
        line(
            "connection_created",
            json!({"connection": id, "tenant": p.tenant, "connector": p.connector, "sub": sub, "user": username,
                   "scope": t.scope, "refresh_token_stored": "sealed", "refresh_sealed_bytes": sealed.len(),
                   "access_ttl_s": t.expires_in}),
        );
        Ok(json!({"connection": id, "sub": sub, "user": username, "scope": t.scope}))
    }

    fn bind(&self, body: &Value) -> Resp {
        let (Some(tenant), Some(session), Some(connector), Some(connection)) = (
            body["tenant"].as_str(),
            body["session"].as_str(),
            body["connector"].as_str(),
            body["connection"].as_str(),
        ) else {
            return reply(
                400,
                json!({"error": "tenant, session, connector and connection are required"}),
            );
        };
        let mut st = self.lock();
        let Some(conn) = st.conns.get(connection).cloned() else {
            return reply(404, json!({"error": "no such connection"}));
        };
        // The record is read without awaiting: a connection's tenant and connector never change.
        let Ok(c) = conn.try_lock() else {
            return reply(409, json!({"error": "connection busy; retry"}));
        };
        if c.rec.tenant != tenant || c.rec.connector != connector {
            line(
                "bind_refused",
                json!({"tenant": tenant, "session": session, "connection": connection, "why": "another tenant's or connector's connection"}),
            );
            return reply(
                403,
                json!({"error": "that connection belongs to another tenant or connector"}),
            );
        }
        if c.rec.status != "active" {
            return reply(
                409,
                json!({"error": format!("connection is {}", c.rec.status)}),
            );
        }
        let key = bind_key(tenant, session, connector);
        match st.bindings.get(&key) {
            Some(b) if b != connection => {
                return reply(
                    409,
                    json!({"error": "that session is already bound to another connection"}),
                );
            }
            Some(_) => {}
            None => {
                st.bindings.insert(key, connection.into());
                self.persist_bindings(&st.bindings);
            }
        }
        let (sub, user) = (c.rec.sub.clone(), c.rec.username.clone());
        drop(c);
        drop(st);
        line(
            "session_bound",
            json!({"tenant": tenant, "session": session, "connector": connector, "connection": connection, "sub": sub, "user": user}),
        );
        reply(
            200,
            json!({"bound": true, "connection": connection, "sub": sub}),
        )
    }

    async fn connection_view(&self, id: &str, audit: bool) -> Resp {
        let conn = self.lock().conns.get(id).cloned();
        let Some(conn) = conn else {
            return reply(404, json!({"error": "no such connection"}));
        };
        let c = conn.lock().await;
        let mut v = json!({
            "id": c.rec.id, "tenant": c.rec.tenant, "connector": c.rec.connector, "sub": c.rec.sub, "user": c.rec.username,
            "scope": c.rec.scope, "status": c.rec.status, "created_ms": c.rec.created_ms, "refreshes": c.rec.refreshes,
            "refresh_token_stored": c.rec.refresh_sealed.is_some(),
            "access_token_expires_in_s": c.access.as_ref().map(|a| a.exp_ms.saturating_sub(now_ms()) / 1000),
        });
        if audit {
            v["at_rest"] = self.at_rest(&c.rec);
        }
        reply(200, v)
    }

    /// What is on disk for this connection, and whether it is the token or only its sealed form.
    fn at_rest(&self, rec: &Record) -> Value {
        let file = std::fs::read(self.path(&format!("{}.json", rec.id))).unwrap_or_default();
        let text = String::from_utf8_lossy(&file).into_owned();
        let Some(sealed) = &rec.refresh_sealed else {
            return json!({"file_bytes": file.len(), "sealed": false});
        };
        let plain = self.vault.open(&rec.tenant, &rec.id, sealed);
        let leaked = plain
            .as_ref()
            .is_some_and(|p| text.contains(String::from_utf8_lossy(p).as_ref()));
        let fp_tail = plain.as_ref().map(|p| tail(&String::from_utf8_lossy(p)));
        json!({
            "file_bytes": file.len(),
            "sealed_bytes": sealed.len(),
            "sealed_head": sealed.chars().take(16).collect::<String>(),
            "cipher": "XChaCha20-Poly1305, key = HKDF-SHA256(master, tenant), aad = connection id",
            "opens_under_the_tenants_key": plain.is_some(),
            "opens_under_another_tenants_key": self.vault.open("another-tenant", &rec.id, sealed).is_some(),
            "opens_as_another_connection": self.vault.open(&rec.tenant, "conn_other", sealed).is_some(),
            "plaintext_token_in_file": leaked,
            "token_tail_in_file": fp_tail.is_some_and(|t| text.contains(&t)),
        })
    }

    fn fingerprints(&self, id: &str) -> Resp {
        let st = self.lock();
        let list: Vec<_> = st
            .fingerprints
            .iter()
            .filter(|f| f.connection == id)
            .cloned()
            .collect();
        reply(200, json!({"fingerprints": list}))
    }

    async fn revoke(&self, id: &str) -> Resp {
        let conn = self.lock().conns.get(id).cloned();
        let Some(conn) = conn else {
            return reply(404, json!({"error": "no such connection"}));
        };
        let mut c = conn.lock().await;
        if c.rec.status == "revoked" {
            return reply(200, json!({"revoked": true, "already": true}));
        }
        let mut out = json!({"revoked": true});
        if let Some(sealed) = c.rec.refresh_sealed.clone()
            && let Some(rt) = self.vault.open(&c.rec.tenant, &c.rec.id, &sealed)
        {
            let rt = String::from_utf8_lossy(&rt).into_owned();
            match self.idp.revoke(&rt).await {
                Ok(()) => out["upstream_revoked"] = json!(true),
                Err(e) => {
                    line(
                        "revoke_failed",
                        json!({"connection": id, "error": e.to_string()}),
                    );
                    return reply(502, json!({"error": e.to_string()}));
                }
            }
            // The provider answers 200 whether it knew the token or not: only using it proves it dead.
            out["refresh_after_revoke"] = match self.idp.refresh(&rt).await {
                Err(IdpError::InvalidGrant(_)) => json!("invalid_grant"),
                Ok(_) => json!("STILL VALID"),
                Err(e) => json!(e.to_string()),
            };
        }
        c.rec.status = "revoked".into();
        c.rec.refresh_sealed = None;
        c.access = None;
        self.persist(&c.rec);
        line(
            "connection_revoked",
            json!({"connection": id, "tenant": c.rec.tenant, "sub": c.rec.sub, "result": out}),
        );
        reply(200, out)
    }

    // ---- the managed endpoint ------------------------------------------------------------------

    /// A live access token for the connection, refreshing it first if it is lapsing (or `force`).
    async fn access_token(
        &self,
        conn: &Arc<AsyncMutex<Conn>>,
        force: bool,
    ) -> Result<String, Refusal> {
        let mut c = conn.lock().await;
        if c.rec.status != "active" {
            return Err(Refusal::Gone(match c.rec.status.as_str() {
                "revoked" => "the user revoked this connection".into(),
                _ => "the identity provider no longer honours this connection".into(),
            }));
        }
        if !force
            && let Some(a) = &c.access
            && a.exp_ms > now_ms() + SKEW_MS
        {
            return Ok(a.token.clone());
        }
        let sealed = c
            .rec
            .refresh_sealed
            .clone()
            .ok_or_else(|| Refusal::Gone("no refresh token is stored".into()))?;
        let old = self
            .vault
            .open(&c.rec.tenant, &c.rec.id, &sealed)
            .map(|b| String::from_utf8_lossy(&b).into_owned())
            .ok_or_else(|| Refusal::Unavailable("the stored token cannot be opened".into()))?;
        match self.idp.refresh(&old).await {
            Ok(t) => {
                let rotated = t.refresh_token.as_ref().is_some_and(|n| *n != old);
                if let Some(n) = t.refresh_token.as_ref().filter(|n| **n != old)
                    && let Some(s) = self.vault.seal(&c.rec.tenant, &c.rec.id, n.as_bytes())
                {
                    c.rec.refresh_sealed = Some(s);
                    self.lock().fingerprints.push(Fingerprint {
                        connection: c.rec.id.clone(),
                        kind: "refresh",
                        tail: tail(n),
                    });
                }
                c.rec.refreshes += 1;
                self.persist(&c.rec);
                self.lock().fingerprints.push(Fingerprint {
                    connection: c.rec.id.clone(),
                    kind: "access",
                    tail: tail(&t.access_token),
                });
                line(
                    "token_refreshed",
                    json!({"connection": c.rec.id, "sub": c.rec.sub, "access_ttl_s": t.expires_in, "refresh_rotated": rotated,
                           "refreshes": c.rec.refreshes, "forced": force}),
                );
                let token = t.access_token.clone();
                c.access = Some(Access {
                    token: t.access_token,
                    exp_ms: expiry_ms(t.expires_in),
                });
                Ok(token)
            }
            Err(IdpError::InvalidGrant(why)) => {
                c.rec.status = "reauth_required".into();
                c.rec.refresh_sealed = None;
                c.access = None;
                self.persist(&c.rec);
                line(
                    "refresh_rejected",
                    json!({"connection": c.rec.id, "sub": c.rec.sub, "why": why}),
                );
                Err(Refusal::Gone(
                    "the identity provider no longer honours this connection".into(),
                ))
            }
            Err(e) => Err(Refusal::Unavailable(e.to_string())),
        }
    }

    pub async fn managed(&self, req: &Req) -> Resp {
        let Some(connector) = req
            .route()
            .strip_prefix("/managed/")
            .and_then(|p| p.strip_suffix("/mcp"))
            .filter(|c| !c.is_empty() && !c.contains('/'))
        else {
            return (404, json_headers(), error_body("not found"));
        };
        let refuse = |why: &str| {
            line(
                "managed_refused",
                json!({"connector": connector, "reason": why}),
            );
            (401, json_headers(), error_body("unauthorized"))
        };
        if req.method == "GET" {
            return (405, json_headers(), error_body("no standalone stream"));
        }
        if req.method == "DELETE" {
            return (200, json_headers(), Vec::new());
        }
        if req.method != "POST" {
            return (405, json_headers(), error_body("method not allowed"));
        }
        let Some(token) = req
            .header("authorization")
            .and_then(|v| v.strip_prefix("Bearer "))
        else {
            return refuse("no bearer token");
        };
        let claims = match self.key.verify(token, now_ms() / 1000) {
            Ok(c) => c,
            Err(e) => {
                return refuse(match e {
                    Reject::Malformed => "malformed token",
                    Reject::UnknownKid => "unknown kid",
                    Reject::BadMac => "bad mac",
                    Reject::Expired => "expired",
                });
            }
        };
        if claims.connector != connector {
            return refuse("token names another connector");
        }
        let rpc: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
        let method = rpc["method"].as_str().unwrap_or("").to_string();

        let bound = self
            .lock()
            .bindings
            .get(&bind_key(&claims.tenant, &claims.session_id, connector))
            .cloned();
        let Some(bound) = bound else {
            line(
                "managed_refused",
                json!({"connector": connector, "tenant": claims.tenant, "session": claims.session_id, "reason": "no connection bound to this session"}),
            );
            return self.refusal(connector, &rpc, "No user has connected their account to this session, so there is nothing to act as. The user has to sign in first.");
        };
        // A request names no connection: the binding is the only way to one. One that tries is refused.
        if let Some(asked) = req
            .header("x-beyond-connection")
            .map(str::to_owned)
            .or_else(|| req.query("connection"))
            && asked != bound
        {
            line(
                "managed_refused",
                json!({"connector": connector, "tenant": claims.tenant, "session": claims.session_id,
                "reason": "request named a connection this session is not bound to", "bound": bound, "asked_for": asked}),
            );
            return (
                403,
                json_headers(),
                error_body("this session's token is not bound to that connection"),
            );
        }
        let conn = self.lock().conns.get(&bound).cloned();
        let Some(conn) = conn else {
            return self.refusal(connector, &rpc, "The connection no longer exists.");
        };

        let mut force = false;
        loop {
            let access = match self.access_token(&conn, force).await {
                Ok(a) => a,
                Err(Refusal::Gone(why)) => {
                    line(
                        "managed_refused",
                        json!({"connector": connector, "tenant": claims.tenant, "session": claims.session_id, "connection": bound, "reason": why}),
                    );
                    return self.refusal(connector, &rpc, &format!("The connection to this account is no longer active: {why}. The user has to sign in again to reconnect it; nothing else is wrong and the session is fine."));
                }
                Err(Refusal::Unavailable(why)) => {
                    line(
                        "managed_upstream_error",
                        json!({"connector": connector, "connection": bound, "error": why}),
                    );
                    return (502, json_headers(), error_body("upstream unavailable"));
                }
            };
            let mut fwd = vec![("authorization".to_string(), format!("Bearer {access}"))];
            for name in [
                "content-type",
                "accept",
                "mcp-session-id",
                "mcp-protocol-version",
            ] {
                if let Some(v) = req.header(name) {
                    fwd.push((name.to_string(), v.to_string()));
                }
            }
            match post(&self.cfg.resource_addr, "/mcp", &fwd, &req.body).await {
                // The resource did not take the token (the provider ended it early): one forced refresh.
                Ok((401, ..)) if !force => {
                    force = true;
                    continue;
                }
                Ok((status, headers, body)) => {
                    let sub = conn.lock().await.rec.sub.clone();
                    line(
                        "managed_call",
                        json!({"connector": connector, "tenant": claims.tenant, "session": claims.session_id,
                        "connection": bound, "sub": sub, "method": method, "upstream_status": status}),
                    );
                    if method == "tools/list"
                        && status == 200
                        && let Ok(v) = serde_json::from_slice::<Value>(&body)
                    {
                        self.lock()
                            .catalog
                            .insert(connector.to_string(), v["result"].clone());
                    }
                    let keep: Vec<_> = headers
                        .into_iter()
                        .filter(|(n, _)| {
                            n == "content-type" || n == "mcp-session-id" || n == "www-authenticate"
                        })
                        .collect();
                    return (status, keep, body);
                }
                Err(e) => {
                    line(
                        "managed_upstream_error",
                        json!({"connector": connector, "error": e.to_string()}),
                    );
                    return (502, json_headers(), error_body("upstream unavailable"));
                }
            }
        }
    }

    /// A refusal the model can read. A tool call gets a tool error (so the model sees why, and the
    /// session carries on); the handshake and the tool list are answered from what the broker already
    /// knows, so a connection that is gone does not make the connector vanish from the session.
    fn refusal(&self, connector: &str, rpc: &Value, text: &str) -> Resp {
        let method = rpc["method"].as_str().unwrap_or("");
        let Some(id) = rpc.get("id").cloned() else {
            return (202, json_headers(), Vec::new());
        };
        let result = match method {
            "tools/call" => json!({"content": [{"type": "text", "text": text}], "isError": true}),
            "initialize" => json!({
                "protocolVersion": rpc["params"]["protocolVersion"].as_str().unwrap_or("2025-06-18"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "managed-broker", "version": "0.1.0"},
            }),
            "ping" => json!({}),
            "tools/list" => match self.lock().catalog.get(connector).cloned() {
                Some(c) => c,
                None => {
                    return reply(
                        200,
                        json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32001, "message": text}}),
                    );
                }
            },
            _ => {
                return reply(
                    200,
                    json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32001, "message": text}}),
                );
            }
        };
        reply(200, json!({"jsonrpc": "2.0", "id": id, "result": result}))
    }
}
