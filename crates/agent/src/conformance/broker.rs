//! The conformance kit for the connector broker: `BEYOND.md`, "What Beyond builds, per part", row
//! "Connector broker", and decision 3 (`bmc_v1`). Beyond runs it against its own broker to prove the
//! broker meets this repo's contract. Run it with
//!
//! ```sh
//! AGENT_CONFORMANCE_BROKER_URL='https://broker.example/managed/notes/mcp' \
//! AGENT_CONFORMANCE_BROKER_HOOK=/path/to/operator-hook \
//! AGENT_CONFORMANCE_BROKER_TOOL=my_notes \
//!   cargo nextest run -p beyond-agent -E 'test(/^broker_conformance::/)' --no-capture
//! ```
//!
//! - `AGENT_CONFORMANCE_BROKER_URL`: the broker's managed endpoint for one connector, as a grant
//!   carries it (`…/managed/<connector>/mcp`). The connector is the last-but-one path segment.
//! - `AGENT_CONFORMANCE_BROKER_TOOL` (and `_TOOL_ARGS`, JSON, default `{}`): a tool of the connector
//!   that answers as the connected user, so two users get different answers. The checks that need a
//!   call are skipped without it.
//! - `AGENT_CONFORMANCE_BROKER_HOOK`: the operator hook, below.
//!
//! `bmc_v1` is this repo's contract (`managed_mcp_token`), so the kit knows the token's vectors, but
//! the broker's live key is its own: the kit asks the hook for tokens.
//!
//! # The operator hook
//!
//! Called as `<hook> <op> <args…>`. Each op is justified by the check that cannot run without it.
//!
//! | op | prints | needed by |
//! |---|---|---|
//! | `mint <secret_hex or -> <tenant> <session> <connector> <exp>` | a `bmc_v1` token: under the given 32-byte secret, or (`-`) under the broker's live key. `exp` is unix seconds | the vectors, every call, expiry |
//! | `connect <tenant> <user>` | the id of a new active connection for that user (the whole consent) | isolation, revocation |
//! | `bind <tenant> <session> <connection>` | nothing | every call |
//! | `revoke <connection>` | nothing | revocation |
//! | `delete-session <tenant> <session>` | nothing: the session ended, so its bindings go | deleted session |
//! | `secrets <connection>` | strings (a whole upstream token, or its tail) that must never reach the agent, one per line | no upstream token |

use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

use super::{CommandHook, Report, fail, pass, skip};
use crate::exec_endpoint::conformance::Check;

pub const FIXTURE: &str = "the minter reproduces tests/fixtures/managed_mcp/v1.json";
pub const ISOLATION: &str = "a token for one connection cannot reach another's";
pub const REVOKED: &str = "a revoked connection is refused at once";
pub const SESSION_DELETED: &str = "a deleted session's bindings are refused at once";
pub const EXPIRED: &str = "an expired token is refused";
pub const CACHE: &str = "a tool list cached for one tenant is not served to another";
pub const NO_UPSTREAM_TOKEN: &str = "no upstream user token reaches the agent";

/// Every check, in run order.
pub const CHECKS: [&str; 7] = [
    FIXTURE,
    ISOLATION,
    REVOKED,
    SESSION_DELETED,
    EXPIRED,
    CACHE,
    NO_UPSTREAM_TOKEN,
];

/// The golden vectors the broker's minter must reproduce.
const VECTORS: &str = include_str!("../../tests/fixtures/managed_mcp/v1.json");

/// The operator hook's ops (the module doc has the contract). [`CommandHook`] runs them as a
/// command; a test of the kit implements them in-process.
#[allow(async_fn_in_trait)]
pub trait Hooks {
    async fn mint(
        &self,
        secret_hex: Option<&str>,
        tenant: &str,
        session: &str,
        connector: &str,
        exp: u64,
    ) -> Result<String, String>;
    async fn connect(&self, tenant: &str, user: &str) -> Result<String, String>;
    async fn bind(&self, tenant: &str, session: &str, connection: &str) -> Result<(), String>;
    async fn revoke(&self, connection: &str) -> Result<(), String>;
    async fn delete_session(&self, tenant: &str, session: &str) -> Result<(), String>;
    async fn secrets(&self, connection: &str) -> Result<Vec<String>, String>;
}

impl Hooks for CommandHook {
    async fn mint(
        &self,
        secret_hex: Option<&str>,
        tenant: &str,
        session: &str,
        connector: &str,
        exp: u64,
    ) -> Result<String, String> {
        let exp = exp.to_string();
        let out = self
            .run(
                "mint",
                &[secret_hex.unwrap_or("-"), tenant, session, connector, &exp],
            )
            .await?;
        Ok(out.trim().to_string())
    }

    async fn connect(&self, tenant: &str, user: &str) -> Result<String, String> {
        Ok(self
            .run("connect", &[tenant, user])
            .await?
            .trim()
            .to_string())
    }

    async fn bind(&self, tenant: &str, session: &str, connection: &str) -> Result<(), String> {
        self.run("bind", &[tenant, session, connection])
            .await
            .map(drop)
    }

    async fn revoke(&self, connection: &str) -> Result<(), String> {
        self.run("revoke", &[connection]).await.map(drop)
    }

    async fn delete_session(&self, tenant: &str, session: &str) -> Result<(), String> {
        self.run("delete-session", &[tenant, session])
            .await
            .map(drop)
    }

    async fn secrets(&self, connection: &str) -> Result<Vec<String>, String> {
        let out = self.run("secrets", &[connection]).await?;
        Ok(out
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(str::to_string)
            .collect())
    }
}

/// Where the broker is and which tool answers as the user.
#[derive(Debug, Clone)]
pub struct BrokerTarget {
    /// The managed endpoint of one connector (`…/managed/<connector>/mcp`).
    pub url: String,
    /// A tool that answers as the connected user, and its arguments.
    pub tool: Option<(String, Value)>,
}

impl BrokerTarget {
    fn connector(&self) -> String {
        let mut segs = self.url.trim_end_matches('/').rsplit('/');
        segs.next();
        segs.next().unwrap_or_default().to_string()
    }
}

/// What one request to the broker came to.
#[derive(Debug)]
struct Reply {
    status: u16,
    body: Value,
    /// Headers and body as text, for the secret scan.
    raw: String,
}

impl Reply {
    /// The text of a tool result that succeeded.
    fn answer(&self) -> Option<String> {
        if self.status != 200
            || self.body["result"]["isError"] == true
            || self.body["error"].is_object()
        {
            return None;
        }
        let text: Vec<_> = self.body["result"]["content"]
            .as_array()?
            .iter()
            .filter_map(|c| c["text"].as_str())
            .collect();
        Some(text.join("\n"))
    }

    fn describe(&self) -> String {
        let body = self.raw.split("\n\n").nth(1).unwrap_or(&self.raw);
        format!("{} {}", self.status, &body[..body.len().min(160)])
    }
}

struct Kit<'a, H> {
    target: &'a BrokerTarget,
    hooks: &'a H,
    client: reqwest::Client,
    nonce: String,
    ids: std::sync::atomic::AtomicUsize,
    /// Everything the agent side saw, for the secret scan.
    seen: std::sync::Mutex<String>,
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default()
}

/// Run every check against the broker.
pub async fn verify<H: Hooks>(target: &BrokerTarget, hooks: &H) -> Report {
    agent_core::ensure_provider();
    let kit = Kit {
        target,
        hooks,
        client: reqwest::Client::new(),
        nonce: format!("{}{}", std::process::id(), now() % 100_000),
        ids: Default::default(),
        seen: Default::default(),
    };
    let mut connections = Vec::new();
    let checks = vec![
        kit.fixture().await,
        kit.isolation(&mut connections).await,
        kit.revoked(&mut connections).await,
        kit.session_deleted(&mut connections).await,
        kit.expired().await,
        kit.cache().await,
        kit.no_upstream_token(&connections).await,
    ];
    Report {
        kit: "connector broker",
        checks,
    }
}

impl<H: Hooks> Kit<'_, H> {
    /// A tenant, session and user name no earlier check has used.
    fn fresh(&self, what: &str) -> String {
        let n = self.ids.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        format!("{what}-{}-{n}", self.nonce)
    }

    async fn token(&self, tenant: &str, session: &str, exp: u64) -> Result<String, String> {
        self.hooks
            .mint(None, tenant, session, &self.target.connector(), exp)
            .await
    }

    /// One JSON-RPC request, as the agent makes it. `connection` is the (illegitimate) attempt to
    /// name one.
    async fn rpc(
        &self,
        token: &str,
        body: Value,
        connection: Option<(&str, &str)>,
    ) -> Result<Reply, String> {
        let url = match connection {
            Some(("query", id)) => format!("{}?connection={id}", self.target.url),
            _ => self.target.url.clone(),
        };
        let mut req = self
            .client
            .post(url)
            .header("authorization", format!("Bearer {token}"))
            .header("accept", "application/json, text/event-stream")
            .json(&body);
        if let Some(("header", id)) = connection {
            req = req.header("x-beyond-connection", id);
        }
        let resp = req.send().await.map_err(|e| format!("no answer: {e}"))?;
        let status = resp.status().as_u16();
        let mut raw = format!("{:?}\n\n", resp.headers());
        let text = resp.text().await.map_err(|e| e.to_string())?;
        raw.push_str(&text);
        self.seen
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push_str(&raw);
        // A streamable-HTTP answer may be one SSE event.
        let json_text = text
            .lines()
            .find_map(|l| l.strip_prefix("data:"))
            .map(str::trim)
            .unwrap_or(text.trim());
        Ok(Reply {
            status,
            body: serde_json::from_str(json_text).unwrap_or(Value::Null),
            raw,
        })
    }

    async fn call_tool(&self, token: &str) -> Result<Reply, String> {
        let (name, args) = self.target.tool.clone().ok_or("no tool configured")?;
        self.rpc(
            token,
            json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": args}}),
            None,
        )
        .await
    }

    async fn list_tools(&self, token: &str) -> Result<Reply, String> {
        self.rpc(
            token,
            json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
            None,
        )
        .await
    }

    /// A tenant with one bound session for a user who has connected; returns the session's token.
    async fn user(
        &self,
        tenant: &str,
        who: &str,
        connections: &mut Vec<String>,
    ) -> Result<(String, String, String), String> {
        let session = self.fresh(&format!("s.{who}"));
        let connection = self.hooks.connect(tenant, &self.fresh(who)).await?;
        connections.push(connection.clone());
        self.hooks.bind(tenant, &session, &connection).await?;
        let token = self.token(tenant, &session, now() + 600).await?;
        Ok((session, connection, token))
    }

    /// Decision 3's wire: `BEYOND.md` "The broker reproduces `tests/fixtures/managed_mcp/v1.json`
    /// byte for byte." The broker's minter, given the fixture's secret and claims, must print the
    /// fixture's token.
    async fn fixture(&self) -> Check {
        let v: Value = serde_json::from_str(VECTORS).unwrap_or(Value::Null);
        let (Some(secret), Some(want)) = (v["secret_hex"].as_str(), v["token"].as_str()) else {
            return fail(FIXTURE, "the fixture has no secret_hex and token");
        };
        let c = &v["claims"];
        let (Some(tenant), Some(session), Some(connector), Some(exp)) = (
            c["tenant"].as_str(),
            c["session_id"].as_str(),
            c["connector"].as_str(),
            c["exp"].as_u64(),
        ) else {
            return fail(FIXTURE, "the fixture's claims are incomplete");
        };
        match self
            .hooks
            .mint(Some(secret), tenant, session, connector, exp)
            .await
        {
            Err(e) => fail(FIXTURE, e),
            Ok(got) if got == want => pass(FIXTURE, "byte for byte"),
            Ok(got) => fail(FIXTURE, format!("minted {got}, want {want}")),
        }
    }

    /// `BEYOND.md`: "A token for one connection cannot reach another's." Two users of one tenant, a
    /// session each. The broker finds the connection from the token's claims alone: naming another
    /// connection is refused, another tenant's token for the same session id reaches nothing, and
    /// the two sessions are not the same user.
    async fn isolation(&self, connections: &mut Vec<String>) -> Check {
        if self.target.tool.is_none() {
            return skip(ISOLATION, "AGENT_CONFORMANCE_BROKER_TOOL is not set");
        }
        let tenant = self.fresh("acme");
        let run = async {
            let (session_a, _, token_a) = self.user(&tenant, "alice", connections).await?;
            let (_, conn_b, token_b) = self.user(&tenant, "bob", connections).await?;
            let a = self.call_tool(&token_a).await?;
            let b = self.call_tool(&token_b).await?;
            let by_header = self
                .rpc(&token_a, json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": self.target.tool.as_ref().map(|t| t.0.clone()), "arguments": {}}}), Some(("header", &conn_b)))
                .await?;
            let by_query = self
                .rpc(&token_a, json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": self.target.tool.as_ref().map(|t| t.0.clone()), "arguments": {}}}), Some(("query", &conn_b)))
                .await?;
            let other_tenant = self
                .token(&self.fresh("globex"), &session_a, now() + 600)
                .await?;
            let c = self.call_tool(&other_tenant).await?;
            Ok::<_, String>((a, b, by_header, by_query, c))
        };
        match run.await {
            Err(e) => fail(ISOLATION, e),
            Ok((a, b, ..)) if a.answer().is_none() || b.answer().is_none() => fail(
                ISOLATION,
                format!(
                    "setup: a connected session's call got {} / {}",
                    a.describe(),
                    b.describe()
                ),
            ),
            Ok((a, b, ..)) if a.answer() == b.answer() => fail(
                ISOLATION,
                "two users' sessions got the same answer from the user tool",
            ),
            Ok((_, _, by_header, by_query, c)) => {
                if by_header.status != 403 {
                    fail(
                        ISOLATION,
                        format!(
                            "a token naming another connection in a header got: {}",
                            by_header.describe()
                        ),
                    )
                } else if by_query.status != 403 {
                    fail(
                        ISOLATION,
                        format!(
                            "a token naming another connection in the query got: {}",
                            by_query.describe()
                        ),
                    )
                } else if c.answer().is_some() {
                    fail(
                        ISOLATION,
                        format!(
                            "another tenant's token for the same session id was answered: {}",
                            c.describe()
                        ),
                    )
                } else {
                    pass(ISOLATION, "the binding is the only way to a connection")
                }
            }
        }
    }

    /// Decision 3: "a revoked connection is refused at once." The very next call after the revoke,
    /// with no waiting, is refused; another user's connection is untouched.
    async fn revoked(&self, connections: &mut Vec<String>) -> Check {
        if self.target.tool.is_none() {
            return skip(REVOKED, "AGENT_CONFORMANCE_BROKER_TOOL is not set");
        }
        let tenant = self.fresh("acme");
        let run = async {
            let (_, conn_a, token_a) = self.user(&tenant, "alice", connections).await?;
            let (_, _, token_b) = self.user(&tenant, "bob", connections).await?;
            let before = self.call_tool(&token_a).await?;
            self.hooks.revoke(&conn_a).await?;
            let after = self.call_tool(&token_a).await?;
            let other = self.call_tool(&token_b).await?;
            Ok::<_, String>((before, after, other))
        };
        match run.await {
            Err(e) => fail(REVOKED, e),
            Ok((before, ..)) if before.answer().is_none() => fail(
                REVOKED,
                format!("setup: the call before revoking got: {}", before.describe()),
            ),
            Ok((_, after, _)) if after.answer().is_some() => fail(
                REVOKED,
                "the call right after the revoke was still answered",
            ),
            Ok((.., other)) if other.answer().is_none() => fail(
                REVOKED,
                format!(
                    "revoking one user's connection broke another's: {}",
                    other.describe()
                ),
            ),
            Ok(_) => pass(
                REVOKED,
                "refused on the next call; the other user's session unaffected",
            ),
        }
    }

    /// Decision 3: "deleting a session removes its bindings."
    async fn session_deleted(&self, connections: &mut Vec<String>) -> Check {
        if self.target.tool.is_none() {
            return skip(SESSION_DELETED, "AGENT_CONFORMANCE_BROKER_TOOL is not set");
        }
        let tenant = self.fresh("acme");
        let run = async {
            let (session, _, token) = self.user(&tenant, "alice", connections).await?;
            let before = self.call_tool(&token).await?;
            self.hooks.delete_session(&tenant, &session).await?;
            let after = self.call_tool(&token).await?;
            Ok::<_, String>((before, after))
        };
        match run.await {
            Err(e) => fail(SESSION_DELETED, e),
            Ok((before, _)) if before.answer().is_none() => fail(
                SESSION_DELETED,
                format!("setup: the call before deleting got: {}", before.describe()),
            ),
            Ok((_, after)) if after.answer().is_some() => fail(
                SESSION_DELETED,
                "a deleted session's token was still answered",
            ),
            Ok(_) => pass(SESSION_DELETED, "refused on the next call"),
        }
    }

    /// `BEYOND.md`: "An expired token is refused." An authentic token past its `exp` gets 401; so
    /// does one whose MAC is wrong.
    async fn expired(&self) -> Check {
        let (tenant, session) = (self.fresh("acme"), self.fresh("s"));
        let run = async {
            let live = self.token(&tenant, &session, now() + 600).await?;
            let past = self
                .token(&tenant, &session, now().saturating_sub(60))
                .await?;
            let mut forged = live.clone();
            forged.pop();
            forged.push(if live.ends_with('A') { 'B' } else { 'A' });
            let ping = |t: String| async move {
                self.rpc(
                    &t,
                    json!({"jsonrpc": "2.0", "id": 1, "method": "ping"}),
                    None,
                )
                .await
            };
            Ok::<_, String>((ping(live).await?, ping(past).await?, ping(forged).await?))
        };
        match run.await {
            Err(e) => fail(EXPIRED, e),
            Ok((live, ..)) if live.status == 401 => {
                fail(EXPIRED, "setup: a live token was refused")
            }
            Ok((_, past, _)) if past.status != 401 => fail(
                EXPIRED,
                format!("an expired token got: {}", past.describe()),
            ),
            Ok((.., forged)) if forged.status != 401 => fail(
                EXPIRED,
                format!("a token with a wrong MAC got: {}", forged.describe()),
            ),
            Ok(_) => pass(EXPIRED, "an expired token and a forged one got 401"),
        }
    }

    /// Decision 3's cache: a tenant's `tools/list` may be kept so a session whose connection is gone
    /// still sees its tools, but never answers another tenant. After one tenant lists, a session of
    /// another tenant with no connection gets no tool list.
    async fn cache(&self) -> Check {
        let mut connections = Vec::new();
        let (a, b) = (self.fresh("acme"), self.fresh("globex"));
        let run = async {
            let (_, _, token_a) = self.user(&a, "alice", &mut connections).await?;
            let listed = self.list_tools(&token_a).await?;
            let stranger = self.token(&b, &self.fresh("s"), now() + 600).await?;
            let leaked = self.list_tools(&stranger).await?;
            Ok::<_, String>((listed, leaked))
        };
        match run.await {
            Err(e) => fail(CACHE, e),
            Ok((listed, _))
                if listed.body["result"]["tools"]
                    .as_array()
                    .is_none_or(Vec::is_empty) =>
            {
                fail(
                    CACHE,
                    format!(
                        "setup: the connected tenant got no tool list: {}",
                        listed.describe()
                    ),
                )
            }
            Ok((_, leaked)) if leaked.body["result"]["tools"].is_array() => fail(
                CACHE,
                "another tenant's session with no connection was served a tool list",
            ),
            Ok(_) => pass(CACHE, "a tenant with no connection gets no tool list"),
        }
    }

    /// Decision 3: "the agent never sees a user token." Everything every earlier check saw is
    /// searched for each connection's upstream tokens (or their tails).
    async fn no_upstream_token(&self, connections: &[String]) -> Check {
        let mut secrets = Vec::new();
        for c in connections {
            match self.hooks.secrets(c).await {
                Ok(s) => secrets.extend(s),
                Err(e) => return fail(NO_UPSTREAM_TOKEN, e),
            }
        }
        if secrets.is_empty() {
            return fail(
                NO_UPSTREAM_TOKEN,
                "the hook gave no upstream token to look for, so nothing was proved",
            );
        }
        let seen = self.seen.lock().unwrap_or_else(|e| e.into_inner());
        match secrets.iter().find(|s| seen.contains(s.as_str())) {
            Some(s) => fail(
                NO_UPSTREAM_TOKEN,
                format!(
                    "an upstream token (…{}) reached the agent",
                    &s[s.len().saturating_sub(6)..]
                ),
            ),
            None => pass(
                NO_UPSTREAM_TOKEN,
                format!(
                    "{} upstream strings, {} bytes of answers searched",
                    secrets.len(),
                    seen.len()
                ),
            ),
        }
    }
}
