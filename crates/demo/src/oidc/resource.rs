//! The protected resource: a small notes service that speaks MCP and trusts only a validated access
//! token. Every request must carry `Authorization: Bearer <JWT>`; the service checks the signature
//! against the identity provider's JWKS, the issuer, the audience (`notes-api`) and the expiry, and
//! the scope each tool needs. It answers with the data of the token's own `sub` and no one else's.
//!
//! - `my_notes`  needs `notes:read`
//! - `add_note`  needs `notes:write`, which the demo's connection is never granted
//!
//! It never sees the agent, the `bmc_v1` token, or the broker's storage: a bearer token is all it is
//! given, as with any third-party API.

use std::sync::Arc;

use hkdf::Hkdf;
use serde_json::{Value, json};
use sha2::Sha256;

use super::jwt::{self, Expect, JwksCache};
use crate::http::{Headers, Req, error_body, json_headers};

pub const AUDIENCE: &str = "notes-api";

pub struct Resource {
    issuer: String,
    jwks: JwksCache,
    /// New at every start: what a user's notes carry that the model could not have guessed.
    nonce: [u8; 16],
    added: std::sync::Mutex<std::collections::HashMap<String, Vec<String>>>,
}

impl Resource {
    pub fn new(issuer: &str, idp_addr: &str, jwks_path: &str) -> Arc<Self> {
        let mut nonce = [0u8; 16];
        let _ = getrandom::fill(&mut nonce);
        Arc::new(Self {
            issuer: issuer.into(),
            jwks: JwksCache::new(idp_addr, jwks_path),
            nonce,
            added: Default::default(),
        })
    }

    /// The reference a user's passport note carries: stable for this user while the service runs.
    pub fn reference(&self, sub: &str) -> String {
        let mut out = [0u8; 4];
        let _ =
            Hkdf::<Sha256>::new(Some(&self.nonce), sub.as_bytes()).expand(b"notes ref", &mut out);
        format!(
            "PR-{}",
            out.iter().map(|b| format!("{b:02x}")).collect::<String>()
        )
    }

    fn notes_for(&self, sub: &str, name: &str) -> Vec<String> {
        let mut notes = vec![
            format!(
                "{name}: passport renewal, reference {}",
                self.reference(sub)
            ),
            format!("{name}: dentist on Friday at 09:00"),
        ];
        if let Ok(added) = self.added.lock() {
            notes.extend(added.get(sub).cloned().unwrap_or_default());
        }
        notes
    }

    pub async fn handle(&self, req: &Req) -> (u16, Headers, Vec<u8>) {
        if req.route() != "/mcp" || req.method != "POST" {
            return (404, json_headers(), error_body("not found"));
        }
        let unauthorized = |why: &str| {
            line("resource_refused", json!({"why": why}));
            let mut h = json_headers();
            h.push((
                "www-authenticate".into(),
                format!("Bearer error=\"invalid_token\", error_description=\"{why}\""),
            ));
            (401, h, error_body("invalid_token"))
        };
        let Some(token) = req
            .header("authorization")
            .and_then(|v| v.strip_prefix("Bearer "))
        else {
            return unauthorized("no bearer token");
        };
        let jwks = match self.jwks.for_token(token).await {
            Ok(j) => j,
            Err(e) => {
                line(
                    "resource_refused",
                    json!({"why": format!("keys unavailable: {e}")}),
                );
                return (
                    503,
                    json_headers(),
                    error_body("cannot reach the identity provider's keys"),
                );
            }
        };
        let now = crate::http::now_ms() / 1000;
        let claims = match jwt::verify(
            token,
            &jwks,
            &Expect {
                issuer: &self.issuer,
                audience: AUDIENCE,
            },
            now,
        ) {
            Ok(c) => c,
            Err(e) => return unauthorized(&e.to_string()),
        };
        let scopes = jwt::scopes(&claims);
        let Some(sub) = claims["sub"].as_str().filter(|s| !s.is_empty()) else {
            return unauthorized("no subject");
        };
        if !scopes.contains(&"notes:read") {
            line(
                "resource_refused",
                json!({"sub": sub, "why": "scope", "needs": "notes:read", "scope": scopes}),
            );
            let mut h = json_headers();
            h.push((
                "www-authenticate".into(),
                "Bearer error=\"insufficient_scope\", scope=\"notes:read\"".into(),
            ));
            return (403, h, error_body("insufficient_scope"));
        }
        let name = claims["preferred_username"].as_str().unwrap_or(sub);
        let Ok(rpc) = serde_json::from_slice::<Value>(&req.body) else {
            return (400, json_headers(), error_body("not json"));
        };
        let method = rpc["method"].as_str().unwrap_or("");
        let Some(id) = rpc.get("id").cloned() else {
            return (202, json_headers(), Vec::new());
        };
        let result = match method {
            "initialize" => Ok(json!({
                "protocolVersion": rpc["params"]["protocolVersion"].as_str().unwrap_or("2025-06-18"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "notes", "version": "0.1.0"},
            })),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(json!({"tools": [
                {"name": "my_notes",
                 "description": "Read the signed-in user's own notes. Only the signed-in user's notes are ever returned.",
                 "inputSchema": {"type": "object", "properties": {
                    "owner": {"type": "string", "description": "optional; must be the signed-in user's own name"}}}},
                {"name": "add_note",
                 "description": "Add a note to the signed-in user's own notes.",
                 "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
            ]})),
            "tools/call" => Ok(self.call(sub, name, &scopes, &claims, &rpc["params"])),
            _ => Err(json!({"code": -32601, "message": format!("method not found: {method}")})),
        };
        let body = match result {
            Ok(r) => json!({"jsonrpc": "2.0", "id": id, "result": r}),
            Err(e) => json!({"jsonrpc": "2.0", "id": id, "error": e}),
        };
        (200, json_headers(), body.to_string().into_bytes())
    }

    fn call(
        &self,
        sub: &str,
        name: &str,
        scopes: &[&str],
        claims: &Value,
        params: &Value,
    ) -> Value {
        let tool = params["name"].as_str().unwrap_or("");
        let args = &params["arguments"];
        let seen = json!({"sub": sub, "user": name, "tool": tool, "iss": claims["iss"], "aud": claims["aud"],
            "scope": scopes, "azp": claims["azp"], "exp": claims["exp"], "iat": claims["iat"],
            "jti": claims["jti"]});
        let refuse =
            |text: String| json!({"content": [{"type": "text", "text": text}], "isError": true});
        match tool {
            "my_notes" => {
                if let Some(owner) = args["owner"].as_str().filter(|o| !o.is_empty())
                    && owner != name
                    && owner != sub
                {
                    line(
                        "resource_refused",
                        json!({"sub": sub, "why": "another user's notes", "asked_for": owner}),
                    );
                    return refuse(format!(
                        "forbidden: you are signed in as {name}; you can only read your own notes, not {owner}'s"
                    ));
                }
                let notes = self.notes_for(sub, name);
                line(
                    "resource_call",
                    json!({"seen": seen, "ref": self.reference(sub), "notes": notes.len()}),
                );
                json!({"content": [{"type": "text", "text": format!("notes for {sub} ({name}):\n- {}", notes.join("\n- "))}], "isError": false})
            }
            "add_note" => {
                if !scopes.contains(&"notes:write") {
                    line(
                        "resource_refused",
                        json!({"sub": sub, "why": "scope", "needs": "notes:write", "scope": scopes, "tool": tool}),
                    );
                    return refuse(format!(
                        "insufficient_scope: add_note needs the notes:write scope, which this connection was not granted (it has: {}). The user would have to reconnect and approve it.",
                        scopes.join(" ")
                    ));
                }
                let text = args["text"].as_str().unwrap_or("").to_string();
                if let Ok(mut added) = self.added.lock() {
                    added
                        .entry(sub.to_string())
                        .or_default()
                        .push(format!("{name}: {text}"));
                }
                line("resource_call", json!({"seen": seen, "wrote": true}));
                json!({"content": [{"type": "text", "text": "note added"}], "isError": false})
            }
            other => refuse(format!("unknown tool {other}")),
        }
    }
}

fn line(event: &str, detail: Value) {
    println!(
        "{}",
        json!({"ts_ms": crate::http::now_ms(), "src": "resource", "event": event, "detail": detail})
    );
}
