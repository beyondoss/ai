//! The broker and the notes service, end to end against a fake identity provider, dialed by the
//! agent's real MCP client the way a hosted session dials its grant's connectors.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use base64::Engine as _;
use base64::engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD};
use beyond_ai_agent::grant::{GrantVerifier, SealKeySource};
use beyond_ai_agent::managed_mcp_token::{Claims, ManagedKey};
use beyond_ai_agent::tools::mcp::{self, McpEgress, McpTools};
use beyond_ai_agent::tools::web::ssrf::EgressPolicy;
use beyond_ai_test_support::grant::Minter;
use beyond_ai_test_support::ports;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::jwt::testing::TestIdp;
use super::{Args, Oidc, idp::Idp};
use crate::http::{form, now_ms, request};
use crate::keys::{Connector, GrantKeys, MintArgs, mint_grant};

const ISSUER: &str = "http://idp.test/realms/demo";
const CLIENT_SECRET: &str = "client-secret-1";
const ADMIN: &str = "admin-bearer";
const MANAGED: [u8; 32] = [0x35; 32];

// ---- a fake identity provider ----------------------------------------------------------------

struct Code {
    sub: String,
    name: String,
    scope: String,
    nonce: String,
    challenge: String,
}

struct Refresh {
    sub: String,
    name: String,
    scope: String,
    active: bool,
}

#[derive(Default)]
struct IdpState {
    codes: HashMap<String, Code>,
    refresh: HashMap<String, Refresh>,
    refresh_grants: u32,
    n: u32,
}

struct FakeIdp {
    key: TestIdp,
    ttl: u64,
    st: Mutex<IdpState>,
}

impl FakeIdp {
    /// The browser's part: sign `sub` in and consent to the URL's scopes. The code, and the state to
    /// hand back with it.
    fn authorize(&self, url: &str, sub: &str, name: &str) -> (String, String) {
        let q: HashMap<_, _> = form(url.split_once('?').unwrap().1).into_iter().collect();
        assert_eq!(q["code_challenge_method"], "S256");
        assert_eq!(q["response_type"], "code");
        let mut st = self.st.lock().unwrap();
        st.n += 1;
        let code = format!("code-{}", st.n);
        st.codes.insert(
            code.clone(),
            Code {
                sub: sub.into(),
                name: name.into(),
                scope: q["scope"].clone(),
                nonce: q["nonce"].clone(),
                challenge: q["code_challenge"].clone(),
            },
        );
        (code, q["state"].clone())
    }

    fn access(&self, sub: &str, name: &str, scope: &str, n: u32) -> String {
        let now = now_ms() / 1000;
        self.key.sign(&json!({"iss": ISSUER, "aud": "notes-api", "sub": sub, "preferred_username": name,
            "scope": scope, "azp": "notes-broker", "iat": now, "exp": now + self.ttl, "jti": format!("jti-{n}")}))
    }

    fn handle(&self, req: &crate::http::Req) -> (u16, Vec<(String, String)>, Vec<u8>) {
        let json_ = |s: u16, v: Value| (s, crate::http::json_headers(), v.to_string().into_bytes());
        let base = "http://idp.test/realms/demo/protocol/openid-connect";
        match req.route() {
            "/realms/demo/.well-known/openid-configuration" => json_(
                200,
                json!({"issuer": ISSUER, "authorization_endpoint": format!("{base}/auth"), "token_endpoint": format!("{base}/token"),
                       "revocation_endpoint": format!("{base}/revoke"), "jwks_uri": format!("{base}/certs")}),
            ),
            "/realms/demo/protocol/openid-connect/certs" => (
                200,
                crate::http::json_headers(),
                self.key.jwks_json().into_bytes(),
            ),
            "/realms/demo/protocol/openid-connect/token"
            | "/realms/demo/protocol/openid-connect/revoke" => {
                let want = format!(
                    "Basic {}",
                    STANDARD.encode(format!("notes-broker:{CLIENT_SECRET}"))
                );
                if req.header("authorization") != Some(want.as_str()) {
                    return json_(401, json!({"error": "invalid_client"}));
                }
                let f: HashMap<_, _> = form(&String::from_utf8_lossy(&req.body))
                    .into_iter()
                    .collect();
                let mut st = self.st.lock().unwrap();
                if req.route().ends_with("/revoke") {
                    if let Some(r) = st.refresh.get_mut(&f["token"]) {
                        r.active = false;
                    }
                    return json_(200, json!({}));
                }
                st.n += 1;
                let n = st.n;
                match f["grant_type"].as_str() {
                    "authorization_code" => {
                        let Some(c) = st.codes.remove(&f["code"]) else {
                            return json_(
                                400,
                                json!({"error": "invalid_grant", "error_description": "code not valid"}),
                            );
                        };
                        let pkce =
                            URL_SAFE_NO_PAD.encode(Sha256::digest(f["code_verifier"].as_bytes()));
                        if pkce != c.challenge {
                            return json_(
                                400,
                                json!({"error": "invalid_grant", "error_description": "PKCE verification failed"}),
                            );
                        }
                        let rt = format!("refresh-token-secret-{n}");
                        st.refresh.insert(
                            rt.clone(),
                            Refresh {
                                sub: c.sub.clone(),
                                name: c.name.clone(),
                                scope: c.scope.clone(),
                                active: true,
                            },
                        );
                        let granted = c
                            .scope
                            .split(' ')
                            .filter(|s| *s != "notes:write")
                            .collect::<Vec<_>>()
                            .join(" ");
                        let now = now_ms() / 1000;
                        let id = self.key.sign(&json!({"iss": ISSUER, "aud": "notes-broker", "sub": c.sub, "preferred_username": c.name,
                            "nonce": c.nonce, "iat": now, "exp": now + self.ttl}));
                        json_(
                            200,
                            json!({"access_token": self.access(&c.sub, &c.name, &granted, n), "refresh_token": rt, "id_token": id,
                            "expires_in": self.ttl, "scope": granted}),
                        )
                    }
                    "refresh_token" => {
                        let Some(r) = st.refresh.get(&f["refresh_token"]).filter(|r| r.active)
                        else {
                            return json_(
                                400,
                                json!({"error": "invalid_grant", "error_description": "Token is not active"}),
                            );
                        };
                        let (sub, name, scope) = (r.sub.clone(), r.name.clone(), r.scope.clone());
                        st.refresh_grants += 1;
                        let granted = scope
                            .split(' ')
                            .filter(|s| *s != "notes:write")
                            .collect::<Vec<_>>()
                            .join(" ");
                        json_(
                            200,
                            json!({"access_token": self.access(&sub, &name, &granted, n), "refresh_token": f["refresh_token"],
                            "expires_in": self.ttl, "scope": granted}),
                        )
                    }
                    _ => json_(400, json!({"error": "unsupported_grant_type"})),
                }
            }
            _ => json_(404, json!({})),
        }
    }
}

// ---- the harness -----------------------------------------------------------------------------

struct Harness {
    idp: Arc<FakeIdp>,
    oidc: Oidc,
    dir: tempfile::TempDir,
}

async fn harness(ttl: u64) -> Harness {
    let fake = Arc::new(FakeIdp {
        key: TestIdp::new("k1"),
        ttl,
        st: Mutex::default(),
    });
    let idp_l = ports::tokio_listener().await;
    let idp_addr = idp_l.local_addr().unwrap().to_string();
    let f = fake.clone();
    tokio::spawn(super::accept(idp_l, move |r| {
        let f = f.clone();
        async move { f.handle(&r) }
    }));
    let idp = Idp::discover(&idp_addr, ISSUER, "notes-broker", CLIENT_SECRET)
        .await
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let args = Args {
        managed_listen: String::new(),
        control_listen: String::new(),
        resource_listen: String::new(),
        idp_internal: idp_addr,
        issuer: ISSUER.into(),
        client_id: "notes-broker".into(),
        redirect_uri: "http://broker.test/oauth/callback".into(),
        data_dir: dir.path().to_path_buf(),
    };
    let oidc = Oidc::start(
        &args,
        ports::tokio_listener().await,
        ports::tokio_listener().await,
        ports::tokio_listener().await,
        idp,
        &MANAGED,
        [9; 32],
        ADMIN.into(),
    )
    .unwrap();
    Harness {
        idp: fake,
        oidc,
        dir,
    }
}

impl Harness {
    async fn control(&self, method: &str, path: &str, body: Value) -> (u16, Value) {
        let h = vec![("authorization".to_string(), format!("Bearer {ADMIN}"))];
        let (s, _, b) = request(
            method,
            &self.oidc.control_addr,
            path,
            &h,
            body.to_string().as_bytes(),
        )
        .await
        .unwrap();
        (s, serde_json::from_slice(&b).unwrap_or(Value::Null))
    }

    /// The whole consent: start, the user signs in at the provider, the callback.
    async fn consent(&self, tenant: &str, sub: &str, name: &str, scopes: Option<Value>) -> String {
        let mut body = json!({"tenant": tenant, "connector": "notes"});
        if let Some(s) = scopes {
            body["scopes"] = s;
        }
        let (s, start) = self.control("POST", "/connect/start", body).await;
        assert_eq!(s, 200, "{start}");
        let (code, state) = self
            .idp
            .authorize(start["authorize_url"].as_str().unwrap(), sub, name);
        let (s, _, b) = request(
            "GET",
            &self.oidc.control_addr,
            &format!("/oauth/callback?code={code}&state={state}"),
            &[],
            &[],
        )
        .await
        .unwrap();
        let v: Value = serde_json::from_slice(&b).unwrap();
        assert_eq!(s, 200, "{v}");
        v["connection"].as_str().unwrap().to_string()
    }

    async fn bind(&self, tenant: &str, session: &str, connection: &str) -> (u16, Value) {
        self.control(
            "POST",
            "/sessions/bind",
            json!({"tenant": tenant, "session": session, "connector": "notes", "connection": connection}),
        )
        .await
    }

    fn bmc(&self, tenant: &str, session: &str) -> String {
        ManagedKey::from_secret_bytes(&MANAGED).mint(&Claims {
            tenant: tenant.into(),
            session_id: session.into(),
            connector: "notes".into(),
            exp: now_ms() / 1000 + 3600,
        })
    }

    /// The tools a session has, connected the way a hosted session connects its grant.
    async fn session(&self, tenant: &str, session: &str) -> McpTools {
        let keys = GrantKeys {
            grant_seed: [0x31; 32],
            fleet_secret: [0x32; 32],
            gateway_key: "bai_v1.demo-gateway-key".into(),
            dek: [0x33; 32],
            managed_secret: Some(MANAGED),
        };
        let token = mint_grant(
            &keys,
            &MintArgs {
                tenant: tenant.into(),
                session: session.into(),
                shard: "demo".into(),
                exec_url: "http://sandbox/exec".into(),
                workspace: "/workspace".into(),
                ttl_secs: 600,
                connectors: vec![Connector {
                    name: "notes".into(),
                    url: format!("http://{}/managed/notes/mcp", self.oidc.managed_addr),
                    managed: true,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .unwrap();
        let flag = Minter::with_keys(self.dir.path(), 1, keys.grant_seed, keys.fleet_secret)
            .grant_key_flag();
        let verifier = GrantVerifier::from_flags(
            &[flag],
            Some(
                SealKeySource::pick(None, Some(&STANDARD.encode(keys.fleet_secret)))
                    .unwrap()
                    .unwrap(),
            ),
        )
        .unwrap()
        .unwrap();
        let grant = verifier.verify(&token, now_ms() / 1000).unwrap();
        let egress = McpEgress::new(EgressPolicy::new(true, &[])).unwrap();
        let (tools, _, warnings) = mcp::connect_granted(
            &grant.mcp,
            &grant.secrets.mcp_headers,
            &egress,
            &grant.exec_url,
            mcp::host(),
            Duration::ZERO,
            &Default::default(),
        )
        .await;
        assert!(warnings.is_empty(), "{warnings:?}");
        tools
    }
}

async fn call(tools: &McpTools, name: &str, args: Value) -> Result<String, String> {
    let tool = tools
        .snapshot()
        .iter()
        .find(|t| t.name() == format!("mcp__notes__{name}"))
        .unwrap_or_else(|| panic!("no tool {name}"))
        .clone();
    tool.run(args)
        .await
        .map(|o| o.text)
        .map_err(|e| e.to_string())
}

mod kit;

// ---- the tests -------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn a_session_acts_as_the_user_who_connected_and_as_no_one_else() {
    let h = harness(60).await;
    let alice = h.consent("acme", "sub-alice", "alice", None).await;
    let bob = h.consent("acme", "sub-bob", "bob", None).await;
    assert_ne!(alice, bob);
    assert_eq!(h.bind("acme", "s.alice", &alice).await.0, 200);
    assert_eq!(h.bind("acme", "s.bob", &bob).await.0, 200);

    let a = h.session("acme", "s.alice").await;
    let b = h.session("acme", "s.bob").await;
    let ta = call(&a, "my_notes", json!({})).await.unwrap();
    let tb = call(&b, "my_notes", json!({})).await.unwrap();
    assert!(ta.contains("notes for sub-alice (alice)"), "{ta}");
    assert!(tb.contains("notes for sub-bob (bob)"), "{tb}");
    assert!(!ta.contains("bob") && !tb.contains("alice"), "{ta}\n{tb}");

    // asking for the other user's notes by name is refused by the service, with a reason
    let err = call(&a, "my_notes", json!({"owner": "bob"}))
        .await
        .unwrap_err();
    assert!(err.contains("only read your own notes"), "{err}");

    // nothing in a request names another connection: a token naming one gets nowhere
    let h2 = vec![
        (
            "authorization".to_string(),
            format!("Bearer {}", h.bmc("acme", "s.alice")),
        ),
        ("content-type".to_string(), "application/json".to_string()),
        ("x-beyond-connection".to_string(), bob.clone()),
    ];
    let rpc = json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "my_notes", "arguments": {}}}).to_string();
    let (status, _, body) = crate::http::post(
        &h.oidc.managed_addr,
        "/managed/notes/mcp",
        &h2,
        rpc.as_bytes(),
    )
    .await
    .unwrap();
    assert_eq!(status, 403, "{}", String::from_utf8_lossy(&body));
    assert!(!String::from_utf8_lossy(&body).contains("sub-bob"));

    // a valid token for a session nobody connected reaches nothing
    let c = h.session("acme", "s.nobody").await;
    let err = call(&c, "my_notes", json!({})).await.unwrap_err();
    assert!(err.contains("nothing to act as"), "{err}");
}

#[tokio::test(flavor = "multi_thread")]
async fn binding_rules_hold() {
    let h = harness(60).await;
    let alice = h.consent("acme", "sub-alice", "alice", None).await;
    let bob = h.consent("acme", "sub-bob", "bob", None).await;
    let other = h.consent("globex", "sub-carol", "carol", None).await;
    assert_eq!(h.bind("acme", "s1", &alice).await.0, 200);
    assert_eq!(h.bind("acme", "s1", &alice).await.0, 200, "idempotent");
    assert_eq!(
        h.bind("acme", "s1", &bob).await.0,
        409,
        "a session has one connection"
    );
    assert_eq!(
        h.bind("acme", "s2", &other).await.0,
        403,
        "another tenant's connection"
    );
    assert_eq!(h.bind("acme", "s2", "conn_nope").await.0, 404);
    // signing in again renews the same connection
    assert_eq!(h.consent("acme", "sub-alice", "alice", None).await, alice);
    // the control plane's credential is required
    let (s, ..) = request("POST", &h.oidc.control_addr, "/sessions/bind", &[], b"{}")
        .await
        .unwrap();
    assert_eq!(s, 401);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_consent_flow_refuses_a_replayed_state_a_forged_state_and_a_wrong_verifier() {
    let h = harness(60).await;
    let (_, start) = h
        .control(
            "POST",
            "/connect/start",
            json!({"tenant": "acme", "connector": "notes"}),
        )
        .await;
    let (code, state) = h.idp.authorize(
        start["authorize_url"].as_str().unwrap(),
        "sub-alice",
        "alice",
    );
    let cb = |code: &str, state: &str| format!("/oauth/callback?code={code}&state={state}");
    let get = |p: String| {
        let addr = h.oidc.control_addr.clone();
        async move { request("GET", &addr, &p, &[], &[]).await.unwrap().0 }
    };
    assert_eq!(get(cb(&code, "forged")).await, 400);
    assert_eq!(get(cb(&code, &state)).await, 200);
    assert_eq!(get(cb(&code, &state)).await, 400, "a state is good once");
    // a code obtained with a different challenge fails PKCE at the provider
    let (_, s2) = h
        .control(
            "POST",
            "/connect/start",
            json!({"tenant": "acme", "connector": "notes"}),
        )
        .await;
    let (_, s3) = h
        .control(
            "POST",
            "/connect/start",
            json!({"tenant": "acme", "connector": "notes"}),
        )
        .await;
    let (code3, _) = h
        .idp
        .authorize(s3["authorize_url"].as_str().unwrap(), "sub-eve", "eve");
    assert_eq!(
        get(cb(&code3, s2["state"].as_str().unwrap())).await,
        400,
        "a code for another verifier"
    );
    // the provider's consent being declined
    let (_, s4) = h
        .control(
            "POST",
            "/connect/start",
            json!({"tenant": "acme", "connector": "notes"}),
        )
        .await;
    let declined = format!(
        "/oauth/callback?error=access_denied&state={}",
        s4["state"].as_str().unwrap()
    );
    assert_eq!(get(declined).await, 400);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_refresh_token_is_sealed_at_rest_and_never_reaches_the_session() {
    let h = harness(60).await;
    let alice = h.consent("acme", "sub-alice", "alice", None).await;
    h.bind("acme", "s.alice", &alice).await;
    let (s, v) = h
        .control("GET", &format!("/connections/{alice}?audit=1"), Value::Null)
        .await;
    assert_eq!(s, 200);
    let ar = &v["at_rest"];
    assert_eq!(ar["opens_under_the_tenants_key"], true);
    assert_eq!(ar["opens_under_another_tenants_key"], false);
    assert_eq!(ar["opens_as_another_connection"], false);
    assert_eq!(ar["plaintext_token_in_file"], false);
    let on_disk = std::fs::read_to_string(h.dir.path().join(format!("{alice}.json"))).unwrap();
    assert!(!on_disk.contains("refresh-token-secret"), "{on_disk}");
    assert!(on_disk.contains("refresh_sealed"));

    // what a session sees, from the handshake to a tool call, holds no token
    let a = h.session("acme", "s.alice").await;
    let out = call(&a, "my_notes", json!({})).await.unwrap();
    let (_, fp) = h
        .control(
            "GET",
            &format!("/connections/{alice}/fingerprints"),
            Value::Null,
        )
        .await;
    let tails: Vec<_> = fp["fingerprints"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["tail"].as_str().unwrap().to_string())
        .collect();
    assert!(tails.len() >= 2);
    for t in &tails {
        assert!(!out.contains(t.as_str()));
    }
    assert!(!out.contains("refresh-token-secret") && !out.contains("eyJ"));
}

#[tokio::test(flavor = "multi_thread")]
async fn an_expired_access_token_is_refreshed_by_the_broker_and_the_session_never_notices() {
    // a 2 s lifetime is inside the broker's 10 s skew: every call refreshes
    let h = harness(2).await;
    let alice = h.consent("acme", "sub-alice", "alice", None).await;
    h.bind("acme", "s.alice", &alice).await;
    let a = h.session("acme", "s.alice").await;
    for _ in 0..3 {
        assert!(
            call(&a, "my_notes", json!({}))
                .await
                .unwrap()
                .contains("sub-alice")
        );
    }
    assert!(h.idp.st.lock().unwrap().refresh_grants >= 3);
    let (_, v) = h
        .control("GET", &format!("/connections/{alice}"), Value::Null)
        .await;
    assert!(v["refreshes"].as_u64().unwrap() >= 3);

    // with a 60 s lifetime the token is reused
    let h = harness(60).await;
    let alice = h.consent("acme", "sub-alice", "alice", None).await;
    h.bind("acme", "s.alice", &alice).await;
    let a = h.session("acme", "s.alice").await;
    for _ in 0..3 {
        call(&a, "my_notes", json!({})).await.unwrap();
    }
    assert_eq!(h.idp.st.lock().unwrap().refresh_grants, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_tool_that_needs_a_scope_the_connection_lacks_is_refused_upstream_with_a_clear_error() {
    let h = harness(60).await;
    // the user is asked for notes:write too and the provider does not grant it
    let alice = h
        .consent(
            "acme",
            "sub-alice",
            "alice",
            Some(json!(["notes:read", "notes:write"])),
        )
        .await;
    h.bind("acme", "s.alice", &alice).await;
    let a = h.session("acme", "s.alice").await;
    let err = call(&a, "add_note", json!({"text": "buy milk"}))
        .await
        .unwrap_err();
    assert!(
        err.contains("insufficient_scope") && err.contains("notes:write"),
        "{err}"
    );
    // the read side still works
    assert!(
        call(&a, "my_notes", json!({}))
            .await
            .unwrap()
            .contains("sub-alice")
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn after_revocation_the_next_call_says_why_and_the_session_carries_on() {
    let h = harness(60).await;
    let alice = h.consent("acme", "sub-alice", "alice", None).await;
    let bob = h.consent("acme", "sub-bob", "bob", None).await;
    h.bind("acme", "s.alice", &alice).await;
    h.bind("acme", "s.bob", &bob).await;
    let a = h.session("acme", "s.alice").await;
    let b = h.session("acme", "s.bob").await;
    assert!(call(&a, "my_notes", json!({})).await.is_ok());

    let (s, v) = h
        .control("POST", &format!("/connections/{alice}/revoke"), Value::Null)
        .await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["upstream_revoked"], true);
    assert_eq!(v["refresh_after_revoke"], "invalid_grant");

    let err = call(&a, "my_notes", json!({})).await.unwrap_err();
    assert!(
        err.contains("no longer active") && err.contains("revoked"),
        "{err}"
    );
    // the session is fine: asked again it says the same, its tools are still listed
    assert!(call(&a, "my_notes", json!({})).await.is_err());
    assert!(
        a.snapshot()
            .iter()
            .any(|t| t.name() == "mcp__notes__my_notes")
    );
    // another user's connection is untouched
    assert!(
        call(&b, "my_notes", json!({}))
            .await
            .unwrap()
            .contains("sub-bob")
    );
    // the stored token is gone, and revoking twice is fine
    let (_, v) = h
        .control("GET", &format!("/connections/{alice}"), Value::Null)
        .await;
    assert_eq!(v["status"], "revoked");
    assert_eq!(v["refresh_token_stored"], false);
    assert_eq!(
        h.control("POST", &format!("/connections/{alice}/revoke"), Value::Null)
            .await
            .0,
        200
    );
}

/// A connector's tool list is cached so a session whose connection is gone still sees its tools.
/// The cache is per tenant: one tenant's list must never answer another tenant's session.
#[tokio::test(flavor = "multi_thread")]
async fn a_cached_tool_list_is_never_served_to_another_tenant() {
    let h = harness(60).await;
    let alice = h.consent("acme", "sub-alice", "alice", None).await;
    h.bind("acme", "s.alice", &alice).await;
    // Connecting lists the tools, which caches acme's list.
    let a = h.session("acme", "s.alice").await;
    assert!(call(&a, "my_notes", json!({})).await.is_ok());

    let list = |tenant: &str| {
        let headers = vec![
            (
                "authorization".to_string(),
                format!("Bearer {}", h.bmc(tenant, "s.x")),
            ),
            ("content-type".to_string(), "application/json".to_string()),
            (
                "accept".to_string(),
                "application/json, text/event-stream".to_string(),
            ),
        ];
        let body = json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).to_string();
        let addr = h.oidc.managed_addr.clone();
        async move {
            let (_, _, b) = request(
                "POST",
                &addr,
                "/managed/notes/mcp",
                &headers,
                body.as_bytes(),
            )
            .await
            .unwrap();
            serde_json::from_slice::<Value>(&b).unwrap()
        }
    };
    // Another tenant's session, bound to nothing: it gets the refusal, not acme's tools.
    let other = list("globex").await;
    assert!(
        other.get("result").is_none() && other["error"]["code"] == -32001,
        "{other}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_provider_that_stops_honouring_the_token_ends_the_connection_cleanly() {
    let h = harness(2).await;
    let alice = h.consent("acme", "sub-alice", "alice", None).await;
    h.bind("acme", "s.alice", &alice).await;
    let a = h.session("acme", "s.alice").await;
    assert!(call(&a, "my_notes", json!({})).await.is_ok());
    // the user withdrew consent at the provider, behind the broker's back
    for r in h.idp.st.lock().unwrap().refresh.values_mut() {
        r.active = false;
    }
    let err = call(&a, "my_notes", json!({})).await.unwrap_err();
    assert!(err.contains("no longer honours"), "{err}");
    let (_, v) = h
        .control("GET", &format!("/connections/{alice}"), Value::Null)
        .await;
    assert_eq!(v["status"], "reauth_required");
}

#[tokio::test(flavor = "multi_thread")]
async fn connections_and_bindings_survive_a_restart() {
    let h = harness(60).await;
    let alice = h.consent("acme", "sub-alice", "alice", None).await;
    h.bind("acme", "s.alice", &alice).await;
    // a second broker over the same directory and provider
    let idp_l = ports::tokio_listener().await;
    let idp_addr = idp_l.local_addr().unwrap().to_string();
    let f = h.idp.clone();
    tokio::spawn(super::accept(idp_l, move |r| {
        let f = f.clone();
        async move { f.handle(&r) }
    }));
    let again = Oidc::start(
        &Args {
            managed_listen: String::new(),
            control_listen: String::new(),
            resource_listen: String::new(),
            idp_internal: idp_addr.clone(),
            issuer: ISSUER.into(),
            client_id: "notes-broker".into(),
            redirect_uri: "http://broker.test/oauth/callback".into(),
            data_dir: h.dir.path().to_path_buf(),
        },
        ports::tokio_listener().await,
        ports::tokio_listener().await,
        ports::tokio_listener().await,
        Idp::discover(&idp_addr, ISSUER, "notes-broker", CLIENT_SECRET)
            .await
            .unwrap(),
        &MANAGED,
        [9; 32],
        ADMIN.into(),
    )
    .unwrap();
    let hh = vec![(
        "authorization".to_string(),
        format!("Bearer {}", h.bmc("acme", "s.alice")),
    )];
    let rpc = json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "my_notes", "arguments": {}}}).to_string();
    // the resource of the new broker does not know the old one's key: it fetches the same JWKS
    let (s, _, b) = crate::http::post(
        &again.managed_addr,
        "/managed/notes/mcp",
        &hh,
        rpc.as_bytes(),
    )
    .await
    .unwrap();
    assert_eq!(s, 200);
    assert!(
        String::from_utf8_lossy(&b).contains("sub-alice"),
        "{}",
        String::from_utf8_lossy(&b)
    );
}
