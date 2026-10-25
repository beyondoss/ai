//! The conformance kit for Beyond's exec front / edge: the part of `BEYOND.md` ("What Beyond builds,
//! per part", row "Exec front / edge", and decisions 1, 2, 4 and 9) that sits between the agent and
//! its sandboxes. Beyond runs it against its own front to prove the front meets this repo's contract.
//!
//! It checks the front's behavior, not the exec protocol: that is [`super::verify`], which this kit
//! can run through the front too (`AGENT_CONFORMANCE_FRONT_EXEC=1`). Run it with
//!
//! ```sh
//! AGENT_CONFORMANCE_FRONT_URL='https://front.example/exec/{session}' \
//! AGENT_CONFORMANCE_FRONT_HOOK=/path/to/operator-hook \
//! AGENT_CONFORMANCE_FRONT_TENANT_CAP=3 \
//!   cargo nextest run -p beyond-ai-agent -E 'test(/^exec_front_conformance::/)' --no-capture
//! ```
//!
//! - `AGENT_CONFORMANCE_FRONT_URL`: the exec URL of one session, `{session}` standing for its id
//!   (any query string the front needs goes in it).
//! - `AGENT_CONFORMANCE_FRONT_HEALTH_URL` (optional): a sandbox health path of one session, also with
//!   `{session}`. Without it the health check is skipped.
//! - `AGENT_CONFORMANCE_FRONT_TENANT_CAP`: the VMs one tenant may hold. Without it the cap check is
//!   skipped.
//! - `AGENT_CONFORMANCE_FRONT_HOOK`: the operator hook, below.
//! - `AGENT_CONFORMANCE_FRONT_PROBE_SECS` / `_SILENCE_SECS` / `_SETTLE_SECS`: the agent's 10 s
//!   probe and 30 s silence limits, and how long a gone VM may take to leave the routes (10 s).
//!
//! # The operator hook
//!
//! Some checks need to see or provoke state only the implementer can reach. The kit stays ignorant of
//! Beyond's internals (and `bagt_v1` is Beyond's contract, so the kit never mints or parses one): the
//! implementer supplies one command, called as `<hook> <op> <tenant> [<session>]`. The kit picks
//! fresh tenant and session ids for every check. Each op is justified by the check that cannot run
//! without it:
//!
//! | op | prints | needed by |
//! |---|---|---|
//! | `token <tenant> <session>` | the exec headers of a grant for that session's sandbox, one `Name: value` per line, at the sandbox's current epoch | every request, and the token checks |
//! | `state <tenant>` | `{"vms": N, "routes": M}`: the tenant's live VMs and route entries | single-flight, failed create, cap, gone VM |
//! | `fail-next-create <tenant> <session>` | nothing | failed create |
//! | `kill-vm <tenant> <session>` | nothing | gone VM, waking |
//! | `end <tenant> <session>` | nothing | stale epoch (ending a session bumps its epoch) |
//!
//! A hook that exits non-zero fails the check that called it.

use std::time::{Duration, Instant};

use serde_json::{Value, json};

use super::{Check, Verdict, fail, pass, skip};
use crate::exec_endpoint::stream::{Frame, FrameReader, NDJSON};

pub const SINGLE_FLIGHT: &str = "concurrent first requests create one VM";
pub const FAILED_CREATE: &str = "a failed create leaves no VM";
pub const TENANT_CAP: &str = "the tenant's N+1th VM is 503 with Retry-After";
pub const GONE_VM: &str = "a VM that is gone leaves no route entry";
pub const EXTENSIONS: &str = "a refused extension is 400 or 422";
pub const NO_PLACEHOLDER: &str = "a waking sandbox never gets a placeholder 200";
pub const LATENCY: &str = "first frame within the 10 s probe and 30 s silence limits";
pub const TOKEN_MISSING: &str = "a missing token is refused";
pub const TOKEN_GARBAGE: &str = "a garbage token is refused";
pub const TOKEN_OTHER_SANDBOX: &str = "a token for sandbox A is refused at sandbox B";
pub const TOKEN_STALE_EPOCH: &str = "a token below the sandbox's epoch is refused";
pub const TOKEN_HEALTH: &str = "sandbox health paths need the token";
pub const EXEC_PROTOCOL: &str = "the exec protocol holds through the front";

/// Every check, in run order.
pub const CHECKS: [&str; 13] = [
    SINGLE_FLIGHT,
    FAILED_CREATE,
    TENANT_CAP,
    GONE_VM,
    EXTENSIONS,
    NO_PLACEHOLDER,
    LATENCY,
    TOKEN_MISSING,
    TOKEN_GARBAGE,
    TOKEN_OTHER_SANDBOX,
    TOKEN_STALE_EPOCH,
    TOKEN_HEALTH,
    EXEC_PROTOCOL,
];

/// How many requests race for one session's first VM.
const RACERS: usize = 8;

/// Where the front is and what it is held to.
#[derive(Debug, Clone)]
pub struct FrontTarget {
    /// One session's exec URL; `{session}` stands for the id.
    pub exec_url: String,
    /// One session's sandbox health URL, if the front has one.
    pub health_url: Option<String>,
    /// The VMs one tenant may hold, if known.
    pub tenant_cap: Option<usize>,
    /// Also run the exec protocol checks through the front (about fifteen seconds).
    pub exec_protocol: bool,
    /// The agent's probe limit: a create or wake must produce its first frame within it.
    pub probe: Duration,
    /// The agent's silence limit: no gap between frames may exceed it.
    pub silence: Duration,
    /// How long a gone VM may take to leave the routes.
    pub settle: Duration,
}

impl FrontTarget {
    pub fn new(exec_url: impl Into<String>) -> Self {
        Self {
            exec_url: exec_url.into(),
            health_url: None,
            tenant_cap: None,
            exec_protocol: false,
            probe: Duration::from_secs(10),
            silence: Duration::from_secs(30),
            settle: Duration::from_secs(10),
        }
    }
}

/// What the operator hook reports about a tenant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct State {
    pub vms: usize,
    pub routes: usize,
}

/// The operator hook's ops (the module doc has the contract). [`CommandHooks`] runs them as a
/// command; a test of the kit implements them in-process.
#[allow(async_fn_in_trait)]
pub trait Hooks {
    async fn token(&self, tenant: &str, session: &str) -> Result<Vec<(String, String)>, String>;
    async fn state(&self, tenant: &str) -> Result<State, String>;
    async fn fail_next_create(&self, tenant: &str, session: &str) -> Result<(), String>;
    async fn kill_vm(&self, tenant: &str, session: &str) -> Result<(), String>;
    async fn end(&self, tenant: &str, session: &str) -> Result<(), String>;
}

/// The hooks as one command, `<command> <op> <args…>`, run through `sh -c`.
pub struct CommandHooks(pub String);

impl CommandHooks {
    async fn run(&self, op: &str, args: &[&str]) -> Result<String, String> {
        let script = format!("{} \"$@\"", self.0);
        let out = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(script)
            .arg("hook")
            .arg(op)
            .args(args)
            .output()
            .await
            .map_err(|e| format!("hook `{op}` did not start: {e}"))?;
        if !out.status.success() {
            return Err(format!(
                "hook `{op}` exited {}: {}",
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    }
}

impl Hooks for CommandHooks {
    async fn token(&self, tenant: &str, session: &str) -> Result<Vec<(String, String)>, String> {
        let out = self.run("token", &[tenant, session]).await?;
        let headers: Vec<_> = out
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(crate::exec_endpoint::HttpExecRunner::parse_header)
            .collect::<Result<_, _>>()?;
        if headers.is_empty() {
            return Err("hook `token` printed no `Name: value` header".into());
        }
        Ok(headers)
    }

    async fn state(&self, tenant: &str) -> Result<State, String> {
        let out = self.run("state", &[tenant]).await?;
        let v: Value = serde_json::from_str(out.trim())
            .map_err(|e| format!("hook `state` printed no JSON ({e}): {out}"))?;
        let n = |k: &str| {
            v[k].as_u64()
                .map(|n| n as usize)
                .ok_or_else(|| format!("hook `state` lacks a number `{k}`: {out}"))
        };
        Ok(State {
            vms: n("vms")?,
            routes: n("routes")?,
        })
    }

    async fn fail_next_create(&self, tenant: &str, session: &str) -> Result<(), String> {
        self.run("fail-next-create", &[tenant, session])
            .await
            .map(drop)
    }

    async fn kill_vm(&self, tenant: &str, session: &str) -> Result<(), String> {
        self.run("kill-vm", &[tenant, session]).await.map(drop)
    }

    async fn end(&self, tenant: &str, session: &str) -> Result<(), String> {
        self.run("end", &[tenant, session]).await.map(drop)
    }
}

/// The kit's verdict: one line per check.
#[derive(Debug, Clone)]
pub struct FrontReport {
    pub checks: Vec<Check>,
}

impl FrontReport {
    /// No check failed (a skipped one is not a pass, but it is not a failure either).
    pub fn passed(&self) -> bool {
        self.checks.iter().all(|c| c.verdict != Verdict::Fail)
    }

    pub fn check(&self, name: &str) -> Option<&Check> {
        self.checks.iter().find(|c| c.name == name)
    }

    pub fn render(&self) -> String {
        let mut out = String::from("exec front:\n");
        for c in &self.checks {
            let mark = match c.verdict {
                Verdict::Pass => "PASS",
                Verdict::Fail => "FAIL",
                Verdict::Skip => "skip",
            };
            out.push_str(&format!("  {mark}  {}: {}\n", c.name, c.detail));
        }
        out
    }
}

/// The grant's headers with the token itself replaced by junk. A scheme word before it (`Bearer`)
/// stays, so the front sees a token that fails verification and not a missing one.
fn garbage(headers: &[(String, String)]) -> Vec<(String, String)> {
    headers
        .iter()
        .map(|(n, v)| {
            let scheme = v.split_once(' ').map_or("", |(s, _)| s);
            (n.clone(), format!("{scheme} garbage").trim().to_string())
        })
        .collect()
}

/// What one exec request to the front came to.
#[derive(Debug)]
enum Outcome {
    /// The command ran: `200`, its marker in the output, a clean exit.
    Ran {
        /// Request sent to the first frame (the whole answer, for a v1.1 envelope).
        first: Duration,
        /// The longest gap between frames, the start counting as one.
        max_gap: Duration,
    },
    /// A non-2xx answer.
    Refused {
        status: u16,
        retry_after: Option<String>,
    },
    /// Anything else: a `200` that is not the command's result, a stream that failed, a transport
    /// error or a stall.
    Broke(String),
}

impl Outcome {
    fn describe(&self) -> String {
        match self {
            Outcome::Ran { first, .. } => format!("ran (first frame after {first:?})"),
            Outcome::Refused {
                status,
                retry_after: Some(r),
            } => format!("{status} (Retry-After: {r})"),
            Outcome::Refused { status, .. } => format!("{status}"),
            Outcome::Broke(why) => why.clone(),
        }
    }

    fn ran(&self) -> bool {
        matches!(self, Outcome::Ran { .. })
    }

    fn refused_with(&self, statuses: &[u16]) -> bool {
        matches!(self, Outcome::Refused { status, .. } if statuses.contains(status))
    }
}

struct Kit<'a, H> {
    target: &'a FrontTarget,
    hooks: &'a H,
    client: reqwest::Client,
    nonce: String,
    ids: std::sync::atomic::AtomicUsize,
}

/// Run every check against the front.
pub async fn verify<H: Hooks>(target: &FrontTarget, hooks: &H) -> FrontReport {
    agent_core::ensure_provider();
    let kit = Kit {
        target,
        hooks,
        client: reqwest::Client::new(),
        nonce: format!(
            "{}{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() % 1_000_000)
                .unwrap_or_default()
        ),
        ids: Default::default(),
    };
    let mut checks = vec![
        kit.single_flight().await,
        kit.failed_create().await,
        kit.tenant_cap().await,
        kit.gone_vm().await,
        kit.extensions().await,
        kit.no_placeholder().await,
        kit.latency().await,
        kit.token_missing().await,
        kit.token_garbage().await,
        kit.token_other_sandbox().await,
        kit.token_stale_epoch().await,
        kit.token_health().await,
    ];
    checks.push(kit.exec_protocol().await);
    FrontReport { checks }
}

impl<H: Hooks> Kit<'_, H> {
    /// A tenant and session no earlier check has used.
    fn fresh(&self, what: &str) -> (String, String) {
        let n = self.ids.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        (
            format!("t-{}-{n}-{what}", self.nonce),
            format!("s-{}-{n}-{what}", self.nonce),
        )
    }

    fn url(&self, template: &str, session: &str) -> String {
        template.replace("{session}", session)
    }

    fn exec_url(&self, session: &str) -> String {
        self.url(&self.target.exec_url, session)
    }

    /// The headers of a valid grant for `session`.
    async fn auth(&self, tenant: &str, session: &str) -> Result<Vec<(String, String)>, String> {
        self.hooks.token(tenant, session).await
    }

    /// Run a short command as `session`, v1.2-streamed. `extra` is merged into the request body.
    async fn call(
        &self,
        session: &str,
        headers: &[(String, String)],
        script: &str,
        extra: Value,
    ) -> Outcome {
        let marker = format!("ok-{}", self.nonce);
        let script = script.replace("MARKER", &marker);
        let mut body = json!({
            "command": "sh", "args": ["-c", script], "timeout_ms": 20_000, "stream": true,
        });
        if let (Some(b), Some(e)) = (body.as_object_mut(), extra.as_object()) {
            b.extend(e.clone());
        }
        let started = Instant::now();
        let mut req = self
            .client
            .post(self.exec_url(session))
            .timeout(self.target.probe + self.target.silence)
            .json(&body);
        for (n, v) in headers {
            req = req.header(n.as_str(), v.as_str());
        }
        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => return Outcome::Broke(format!("no answer: {e}")),
        };
        let status = resp.status();
        if !status.is_success() {
            let retry_after = resp
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok())
                .map(str::to_string);
            return Outcome::Refused {
                status: status.as_u16(),
                retry_after,
            };
        }
        let ndjson = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| {
                v.split(';')
                    .next()
                    .is_some_and(|m| m.trim().eq_ignore_ascii_case(NDJSON))
            });
        if !ndjson {
            let Ok(bytes) = resp.bytes().await else {
                return Outcome::Broke("the answer's body failed".into());
            };
            let first = started.elapsed();
            return match serde_json::from_slice::<Value>(&bytes) {
                Ok(v)
                    if v["exit_code"] == 0
                        && v["stdout"].as_str().is_some_and(|s| s.contains(&marker)) =>
                {
                    Outcome::Ran {
                        first,
                        max_gap: first,
                    }
                }
                _ => Outcome::Broke(format!(
                    "200 that is not the command's result: {}",
                    String::from_utf8_lossy(&bytes[..bytes.len().min(120)])
                )),
            };
        }
        let mut reader = FrameReader::new(resp);
        let (mut first, mut last, mut max_gap) = (None, Duration::ZERO, Duration::ZERO);
        let mut stdout = Vec::new();
        loop {
            let frame = match reader.next(self.target.silence).await {
                Ok(Some(f)) => f,
                Ok(None) => return Outcome::Broke("the stream ended without a final frame".into()),
                Err(e) => return Outcome::Broke(format!("the stream stalled or broke: {e}")),
            };
            let at = started.elapsed();
            first.get_or_insert(at);
            max_gap = max_gap.max(at - last);
            last = at;
            match frame {
                Frame::Stdout(b) => stdout.extend(b),
                Frame::Exit(0) => break,
                f if f.is_final() => return Outcome::Broke(format!("the stream ended with {f:?}")),
                _ => {}
            }
        }
        if String::from_utf8_lossy(&stdout).contains(&marker) {
            Outcome::Ran {
                first: first.unwrap_or(last),
                max_gap,
            }
        } else {
            Outcome::Broke("200 stream that never carried the command's output".into())
        }
    }

    async fn state(&self, tenant: &str) -> Result<State, String> {
        self.hooks.state(tenant).await
    }

    /// Poll the tenant's state until `done` or the settle window passes; returns the last state.
    async fn settle(&self, tenant: &str, done: impl Fn(&State) -> bool) -> Result<State, String> {
        let until = Instant::now() + self.target.settle;
        loop {
            let s = self.state(tenant).await?;
            if done(&s) || Instant::now() >= until {
                return Ok(s);
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// Decision 9: "single-flight on the session id: the first sandbox request creates the VM".
    /// BEYOND.md: "N concurrent first requests for one session create one VM."
    async fn single_flight(&self) -> Check {
        let (tenant, session) = self.fresh("race");
        let run = async {
            let auth = self.auth(&tenant, &session).await?;
            let calls = (0..RACERS).map(|_| self.call(&session, &auth, "printf MARKER", json!({})));
            let outcomes = futures::future::join_all(calls).await;
            let state = self.state(&tenant).await?;
            Ok::<_, String>((outcomes, state))
        };
        match run.await {
            Err(e) => fail(SINGLE_FLIGHT, e),
            Ok((outcomes, s)) => {
                if let Some(bad) = outcomes.iter().find(|o| !o.ran()) {
                    fail(
                        SINGLE_FLIGHT,
                        format!("a racing first request got: {}", bad.describe()),
                    )
                } else if s.vms != 1 {
                    fail(
                        SINGLE_FLIGHT,
                        format!("{RACERS} concurrent first requests made {} VMs", s.vms),
                    )
                } else {
                    pass(
                        SINGLE_FLIGHT,
                        format!("{RACERS} concurrent first requests, one VM"),
                    )
                }
            }
        }
    }

    /// BEYOND.md: "A failed create leaves no VM." Decision 9: "kill a VM whose create failed".
    /// The hook makes the next create fail; whatever the front answers, a request that did not run
    /// must leave neither a VM nor a route, and the retry must work.
    async fn failed_create(&self) -> Check {
        let (tenant, session) = self.fresh("fail");
        let run = async {
            let auth = self.auth(&tenant, &session).await?;
            self.hooks.fail_next_create(&tenant, &session).await?;
            let first = self.call(&session, &auth, "printf MARKER", json!({})).await;
            let after = self.state(&tenant).await?;
            let retry = self.call(&session, &auth, "printf MARKER", json!({})).await;
            let then = self.state(&tenant).await?;
            Ok::<_, String>((first, after, retry, then))
        };
        match run.await {
            Err(e) => fail(FAILED_CREATE, e),
            Ok((first, after, retry, then)) => {
                if !first.ran() && (after.vms != 0 || after.routes != 0) {
                    fail(
                        FAILED_CREATE,
                        format!(
                            "the create failed ({}) and left {} VM(s) and {} route(s)",
                            first.describe(),
                            after.vms,
                            after.routes
                        ),
                    )
                } else if !retry.ran() {
                    fail(
                        FAILED_CREATE,
                        format!("the retry after a failed create got: {}", retry.describe()),
                    )
                } else if then.vms != 1 {
                    fail(
                        FAILED_CREATE,
                        format!("after the retry the session has {} VMs", then.vms),
                    )
                } else {
                    pass(
                        FAILED_CREATE,
                        format!(
                            "failed create ({}) left nothing; the retry ran",
                            first.describe()
                        ),
                    )
                }
            }
        }
    }

    /// Decision 9: "A tenant at its VM cap gets `503` with `Retry-After`."
    /// BEYOND.md: "The tenant's N+1th VM gets `503`."
    async fn tenant_cap(&self) -> Check {
        let Some(cap) = self.target.tenant_cap else {
            return skip(TENANT_CAP, "AGENT_CONFORMANCE_FRONT_TENANT_CAP is not set");
        };
        let tenant = self.fresh("cap").0;
        let run = async {
            for i in 0..cap {
                let session = format!("{}-{i}", self.fresh("cap").1);
                let auth = self.auth(&tenant, &session).await?;
                let o = self.call(&session, &auth, "printf MARKER", json!({})).await;
                if !o.ran() {
                    return Ok(Err(format!("VM {} of {cap} got: {}", i + 1, o.describe())));
                }
            }
            let over = self.fresh("cap").1;
            let auth = self.auth(&tenant, &over).await?;
            let o = self.call(&over, &auth, "printf MARKER", json!({})).await;
            let s = self.state(&tenant).await?;
            Ok::<_, String>(Ok((o, s)))
        };
        match run.await {
            Err(e) | Ok(Err(e)) => fail(TENANT_CAP, e),
            Ok(Ok((o, s))) => match o {
                Outcome::Refused {
                    status: 503,
                    retry_after: Some(r),
                } if r.trim().parse::<u64>().is_ok() => {
                    if s.vms > cap {
                        fail(
                            TENANT_CAP,
                            format!(
                                "refused, but the tenant holds {} VMs past its cap of {cap}",
                                s.vms
                            ),
                        )
                    } else {
                        pass(
                            TENANT_CAP,
                            format!("VM {} got 503, Retry-After: {r}", cap + 1),
                        )
                    }
                }
                Outcome::Refused {
                    status: 503,
                    retry_after,
                } => fail(
                    TENANT_CAP,
                    format!("503 without a numeric Retry-After ({retry_after:?})"),
                ),
                other => fail(
                    TENANT_CAP,
                    format!("VM {} of a cap of {cap} got: {}", cap + 1, other.describe()),
                ),
            },
        }
    }

    /// BEYOND.md: "A VM that is gone leaves no route entry." Decision 9: "No entry outlives its VM."
    async fn gone_vm(&self) -> Check {
        let (tenant, session) = self.fresh("gone");
        let run = async {
            let auth = self.auth(&tenant, &session).await?;
            let up = self.call(&session, &auth, "printf MARKER", json!({})).await;
            self.hooks.kill_vm(&tenant, &session).await?;
            let s = self.settle(&tenant, |s| s.routes == 0).await?;
            Ok::<_, String>((up, s))
        };
        match run.await {
            Err(e) => fail(GONE_VM, e),
            Ok((up, _)) if !up.ran() => fail(
                GONE_VM,
                format!("setup: the first request got: {}", up.describe()),
            ),
            Ok((_, s)) if s.routes != 0 => fail(
                GONE_VM,
                format!(
                    "{} route entr{} still point at a VM that is gone after {:?}",
                    s.routes,
                    if s.routes == 1 { "y" } else { "ies" },
                    self.target.settle
                ),
            ),
            Ok(_) => pass(GONE_VM, "no route entry outlives its VM"),
        }
    }

    /// Decision 9: "A refused extension (stream, job or notify) is `400` or `422`." An extension the
    /// front cannot honor must be refused with one of those, not a 5xx, a 404 or a `200` that is not
    /// the command's result. An endpoint that ignores a field it does not know (the protocol allows
    /// it) answers with the command's result, which is also fine.
    async fn extensions(&self) -> Check {
        let (tenant, session) = self.fresh("ext");
        let auth = match self.auth(&tenant, &session).await {
            Ok(a) => a,
            Err(e) => return fail(EXTENSIONS, e),
        };
        let cases = [
            ("stream", json!({"stream": "not-a-bool"})),
            ("job", json!({"job": "not-an-object"})),
            ("job op", json!({"job": {"op": "no-such-op"}})),
            (
                "notify",
                json!({"job": {"op": "keep", "id": "0", "num": 1, "kind": "x", "seen": 0, "notify": "not-an-object"}}),
            ),
        ];
        let mut ok = Vec::new();
        for (name, extra) in cases {
            let o = self.call(&session, &auth, "printf MARKER", extra).await;
            if !(o.ran() || o.refused_with(&[400, 422])) {
                return fail(
                    EXTENSIONS,
                    format!("a malformed `{name}` extension got: {}", o.describe()),
                );
            }
            ok.push(format!(
                "{name}: {}",
                if o.ran() { "ignored" } else { "refused" }
            ));
        }
        pass(EXTENSIONS, ok.join(", "))
    }

    /// Decision 9: "A waking sandbox is never answered with a placeholder `200`." The first request
    /// of a session and the first after its VM is gone both wake a sandbox; each must carry the
    /// command's real result or be a `503` with `Retry-After`.
    async fn no_placeholder(&self) -> Check {
        let (tenant, session) = self.fresh("wake");
        let run = async {
            let auth = self.auth(&tenant, &session).await?;
            let create = self.call(&session, &auth, "printf MARKER", json!({})).await;
            self.hooks.kill_vm(&tenant, &session).await?;
            let wake = self.call(&session, &auth, "printf MARKER", json!({})).await;
            Ok::<_, String>([("create", create), ("wake", wake)])
        };
        match run.await {
            Err(e) => fail(NO_PLACEHOLDER, e),
            Ok(both) => {
                for (what, o) in &both {
                    let honest = o.ran()
                        || matches!(
                            o,
                            Outcome::Refused {
                                status: 503,
                                retry_after: Some(_)
                            }
                        );
                    if !honest {
                        return fail(
                            NO_PLACEHOLDER,
                            format!("the {what} request got: {}", o.describe()),
                        );
                    }
                }
                pass(
                    NO_PLACEHOLDER,
                    "create and wake answered with the result or 503 + Retry-After",
                )
            }
        }
    }

    /// Decision 9: "a create or wake finishes, or streams its first frame, inside the agent's 10 s
    /// probe and 30 s silence limits."
    async fn latency(&self) -> Check {
        let (tenant, session) = self.fresh("late");
        let run = async {
            let auth = self.auth(&tenant, &session).await?;
            let create = self
                .call(
                    &session,
                    &auth,
                    "printf MARKER; sleep 1; printf x",
                    json!({}),
                )
                .await;
            self.hooks.kill_vm(&tenant, &session).await?;
            let wake = self.call(&session, &auth, "printf MARKER", json!({})).await;
            Ok::<_, String>([("create", create), ("wake", wake)])
        };
        match run.await {
            Err(e) => fail(LATENCY, e),
            Ok(both) => {
                let mut worst = Duration::ZERO;
                for (what, o) in &both {
                    match o {
                        Outcome::Ran { first, max_gap } => {
                            if *first > self.target.probe {
                                return fail(
                                    LATENCY,
                                    format!(
                                        "the {what}'s first frame took {first:?}, past the {:?} probe",
                                        self.target.probe
                                    ),
                                );
                            }
                            if *max_gap > self.target.silence {
                                return fail(
                                    LATENCY,
                                    format!(
                                        "the {what}'s stream was silent for {max_gap:?}, past {:?}",
                                        self.target.silence
                                    ),
                                );
                            }
                            worst = worst.max(*first);
                        }
                        other => {
                            return fail(
                                LATENCY,
                                format!("the {what} request got: {}", other.describe()),
                            );
                        }
                    }
                }
                pass(
                    LATENCY,
                    format!("create and wake streamed within {worst:?}"),
                )
            }
        }
    }

    /// Decision 1: the edge checks the token on every sandbox path. A refused request must also
    /// create nothing.
    async fn refused_without_effect(
        &self,
        name: &'static str,
        what: &str,
        tenant: &str,
        session: &str,
        headers: &[(String, String)],
    ) -> Check {
        let before = match self.state(tenant).await {
            Ok(s) => s,
            Err(e) => return fail(name, e),
        };
        let o = self
            .call(session, headers, "printf MARKER", json!({}))
            .await;
        let after = match self.state(tenant).await {
            Ok(s) => s,
            Err(e) => return fail(name, e),
        };
        if !o.refused_with(&[401, 403]) {
            fail(name, format!("{what} got: {}", o.describe()))
        } else if after.vms != before.vms || after.routes != before.routes {
            fail(
                name,
                format!("{what} was refused but still changed the tenant's VMs/routes"),
            )
        } else {
            pass(name, format!("{what} got {}", o.describe()))
        }
    }

    /// Decision 1: "The edge checks it on every sandbox path." A request without a token is refused.
    async fn token_missing(&self) -> Check {
        let (tenant, session) = self.fresh("none");
        self.refused_without_effect(
            TOKEN_MISSING,
            "a request with no token",
            &tenant,
            &session,
            &[],
        )
        .await
    }

    async fn token_garbage(&self) -> Check {
        let (tenant, session) = self.fresh("junk");
        let garbage = match self.auth(&tenant, &session).await {
            Ok(h) => garbage(&h),
            Err(e) => return fail(TOKEN_GARBAGE, e),
        };
        self.refused_without_effect(
            TOKEN_GARBAGE,
            "a request with a garbage token",
            &tenant,
            &session,
            &garbage,
        )
        .await
    }

    /// Decision 1: the token is "scoped to one sandbox". A token for session A, presented for B.
    async fn token_other_sandbox(&self) -> Check {
        let (tenant, a) = self.fresh("a");
        let b = self.fresh("b").1;
        let run = async {
            let auth_a = self.auth(&tenant, &a).await?;
            let up = self.call(&a, &auth_a, "printf MARKER", json!({})).await;
            Ok::<_, String>((auth_a, up))
        };
        match run.await {
            Err(e) => fail(TOKEN_OTHER_SANDBOX, e),
            Ok((_, up)) if !up.ran() => fail(
                TOKEN_OTHER_SANDBOX,
                format!("setup: A's own token got: {}", up.describe()),
            ),
            Ok((auth_a, _)) => {
                self.refused_without_effect(
                    TOKEN_OTHER_SANDBOX,
                    "A's token at B",
                    &tenant,
                    &b,
                    &auth_a,
                )
                .await
            }
        }
    }

    /// Decision 2: "refuse any token below the sandbox's current epoch"; "ending a session bumps the
    /// epoch". A token that worked is refused after the session ends, and refusing creates nothing.
    async fn token_stale_epoch(&self) -> Check {
        let (tenant, session) = self.fresh("epoch");
        let run = async {
            let old = self.auth(&tenant, &session).await?;
            let up = self.call(&session, &old, "printf MARKER", json!({})).await;
            self.hooks.end(&tenant, &session).await?;
            Ok::<_, String>((old, up))
        };
        match run.await {
            Err(e) => fail(TOKEN_STALE_EPOCH, e),
            Ok((_, up)) if !up.ran() => fail(
                TOKEN_STALE_EPOCH,
                format!("setup: the token's first use got: {}", up.describe()),
            ),
            Ok((old, _)) => {
                self.refused_without_effect(
                    TOKEN_STALE_EPOCH,
                    "the token of an ended session",
                    &tenant,
                    &session,
                    &old,
                )
                .await
            }
        }
    }

    /// Decision 1: the edge checks the token "on every sandbox path, sandbox health checks included".
    async fn token_health(&self) -> Check {
        let Some(template) = &self.target.health_url else {
            return skip(
                TOKEN_HEALTH,
                "AGENT_CONFORMANCE_FRONT_HEALTH_URL is not set",
            );
        };
        let (tenant, session) = self.fresh("health");
        let url = self.url(template, &session);
        let run = async {
            let auth = self.auth(&tenant, &session).await?;
            let up = self.call(&session, &auth, "printf MARKER", json!({})).await;
            let get = |headers: Vec<(String, String)>| {
                let mut req = self.client.get(&url).timeout(self.target.probe);
                for (n, v) in headers {
                    req = req.header(n, v);
                }
                async move {
                    req.send()
                        .await
                        .map(|r| r.status().as_u16())
                        .map_err(|e| e.to_string())
                }
            };
            let garbage = garbage(&auth);
            Ok::<_, String>((
                up,
                get(vec![]).await?,
                get(garbage).await?,
                get(auth).await?,
            ))
        };
        match run.await {
            Err(e) => fail(TOKEN_HEALTH, e),
            Ok((up, ..)) if !up.ran() => fail(
                TOKEN_HEALTH,
                format!("setup: the sandbox did not come up: {}", up.describe()),
            ),
            Ok((_, none, junk, good)) => {
                if ![401, 403].contains(&none) {
                    fail(
                        TOKEN_HEALTH,
                        format!("a health check with no token got {none}"),
                    )
                } else if ![401, 403].contains(&junk) {
                    fail(
                        TOKEN_HEALTH,
                        format!("a health check with a garbage token got {junk}"),
                    )
                } else if [401, 403].contains(&good) {
                    fail(
                        TOKEN_HEALTH,
                        format!("a health check with a valid token got {good}"),
                    )
                } else {
                    pass(
                        TOKEN_HEALTH,
                        format!(
                            "{none} without a token, {junk} with garbage, {good} with the grant's"
                        ),
                    )
                }
            }
        }
    }

    /// The exec protocol (`super::verify`) through the front, for a session with a grant. Opt in:
    /// it takes about fifteen seconds.
    async fn exec_protocol(&self) -> Check {
        if !self.target.exec_protocol {
            return skip(EXEC_PROTOCOL, "AGENT_CONFORMANCE_FRONT_EXEC is not 1");
        }
        let (tenant, session) = self.fresh("proto");
        let auth = match self.auth(&tenant, &session).await {
            Ok(a) => a,
            Err(e) => return fail(EXEC_PROTOCOL, e),
        };
        let report = super::verify(
            &self.exec_url(&session),
            &auth,
            &super::Expectations::default(),
        )
        .await;
        let failed: Vec<_> = report
            .checks
            .iter()
            .filter(|c| c.verdict == Verdict::Fail)
            .map(|c| c.name)
            .collect();
        if failed.is_empty() {
            pass(
                EXEC_PROTOCOL,
                format!("{} protocol checks", report.checks.len()),
            )
        } else {
            fail(EXEC_PROTOCOL, format!("failed: {}", failed.join("; ")))
        }
    }
}
