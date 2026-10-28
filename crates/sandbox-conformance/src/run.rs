//! Running scenarios: the steps, `requires`, `poll`, strict `xfail`, and the per-check outcome.
use std::path::Path;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde_json::{Map, Value, json};
use sha2::Digest as _;

use crate::eval::{Vars, derive, dotted, matches, subst};

/// A driver, as a thing that answers one request with one response, in order.
#[async_trait]
pub trait Link: Send + Sync {
    async fn call(&self, req: &Value) -> Value;
}

/// How a backend starts a driver. A scenario with `connect` asks for one with its own credentials:
/// `bearer: Some(None)` means no bearer, `None` keeps the backend's own.
#[async_trait]
pub trait Connector: Send + Sync {
    async fn connect(&self, bearer: Option<Option<String>>, url: Option<String>) -> Box<dyn Link>;
}

/// A started backend with its driver connected and probed.
pub struct Backend {
    pub name: String,
    pub link: Box<dyn Link>,
    pub connector: Box<dyn Connector>,
    /// What `capabilities` answered (after `CONFORMANCE_CAPS`).
    pub caps: Value,
    /// What `hello` listed.
    pub ops: Vec<String>,
    /// `hello.features`: what the driver can do that a scenario's `requires.features` names.
    pub features: Vec<String>,
    /// `hello.missing`: why the driver lacks a feature, by feature name (for the report).
    pub missing: Value,
    /// `hello.driver`: the name an `xfail` mark is matched against.
    pub driver: String,
    /// The backend demands a bearer (`requires.auth`).
    pub auth: bool,
    /// A directory in the sandbox the scenarios may use (`${WORK}`).
    pub work: String,
    /// Directories this environment provides beyond `${WORK}`, by variable name (`RO`, `TINY`):
    /// a scenario that `requires.vars` them is skipped where they are absent.
    pub vars: std::collections::BTreeMap<String, String>,
}

impl Backend {
    pub async fn open(
        name: &str,
        connector: Box<dyn Connector>,
        auth: bool,
        work: &str,
    ) -> Backend {
        let link = connector.connect(None, None).await;
        let hello = link.call(&json!({"id": 0, "op": "hello"})).await;
        let ops = hello["ok"]["ops"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|o| o.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let features = hello["ok"]["features"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|o| o.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let missing = hello["ok"]["missing"].clone();
        let driver = hello["ok"]["driver"].as_str().unwrap_or("").to_string();
        let caps = link.call(&json!({"id": 0, "op": "capabilities"})).await["ok"].clone();
        Backend {
            name: name.to_string(),
            link,
            connector,
            caps,
            ops,
            features,
            missing,
            driver,
            auth,
            work: work.to_string(),
            vars: Default::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Pass,
    Fail,
    /// An `xfail` check that failed as marked.
    KnownDefect,
    /// The driver lacks an op the scenario uses. Never a pass.
    Unsupported,
    /// Planned, or `requires` unmet. Never a pass.
    Skipped,
}

#[derive(Clone, Debug)]
pub struct Check {
    /// `<scenario id>/<step id>`
    pub id: String,
    pub status: Status,
    /// Why it failed or was skipped.
    pub detail: String,
}

#[derive(Default, Debug)]
pub struct Tally {
    pub passed: usize,
    pub failed: usize,
    pub known_defect: usize,
    pub unsupported: usize,
    pub skipped: usize,
    pub failing: Vec<String>,
}

impl Tally {
    pub fn add(&mut self, checks: &[Check]) {
        for c in checks {
            match c.status {
                Status::Pass => self.passed += 1,
                Status::Fail => {
                    self.failed += 1;
                    self.failing.push(c.id.clone());
                }
                Status::KnownDefect => self.known_defect += 1,
                Status::Unsupported => self.unsupported += 1,
                Status::Skipped => self.skipped += 1,
            }
        }
    }

    pub fn total(&self) -> usize {
        self.passed + self.failed + self.known_defect + self.unsupported + self.skipped
    }
}

fn request(id: u64, step: &Value, vars: &Vars) -> Value {
    let mut req = Map::new();
    for (k, v) in step.as_object().unwrap() {
        if !matches!(
            k.as_str(),
            "id" | "expect" | "save" | "poll" | "note" | "xfail" | "via"
        ) {
            req.insert(k.clone(), subst(v, vars));
        }
    }
    req.insert("id".into(), json!(id));
    Value::Object(req)
}

async fn exec(be: &Backend, command: &str, args: &[&str]) -> Value {
    be.link
        .call(&json!({"id": 0, "op": "exec", "command": command, "args": args}))
        .await
}

/// `requires` against what this backend can do. `None` means run it; `Some(why)` is the skip line.
async fn unmet(req: &Value, be: &Backend) -> Option<String> {
    let req = req.as_object()?;
    for (k, want) in req {
        match k.as_str() {
            "auth" => {
                if want.as_bool() != Some(be.auth) {
                    return Some(format!("requires auth={want}"));
                }
            }
            "search_engine" => {
                let have = be.caps["search_engine"].clone();
                let ok = match want {
                    Value::Array(a) => a.contains(&have),
                    w => *w == have,
                };
                if !ok {
                    return Some(format!("requires search_engine={want}, backend has {have}"));
                }
            }
            // Judged for the driver, not the backend (see `lacks_feature`).
            "features" => {}
            "vars" => {
                for v in want.as_array().unwrap() {
                    let name = v.as_str().unwrap();
                    if !be.vars.contains_key(name) {
                        return Some(format!("requires the environment variable ${{{name}}}"));
                    }
                }
            }
            "not_root" => {
                let r = exec(be, "id", &["-u"]).await;
                let root = decode_utf8(&r["ok"]["stdout_b64"]).trim() == "0";
                if root == want.as_bool().unwrap_or(true) {
                    return Some("requires a non-root sandbox user".to_string());
                }
            }
            "commands" => {
                for c in want.as_array().unwrap() {
                    let r = exec(
                        be,
                        "sh",
                        &["-c", "command -v \"$1\"", "sh", c.as_str().unwrap()],
                    )
                    .await;
                    if r["ok"]["exit_code"] != 0 {
                        return Some(format!("requires command {c}"));
                    }
                }
            }
            cap => {
                if be.caps[cap] != *want {
                    return Some(format!(
                        "requires {cap}={want}, backend has {}",
                        be.caps[cap]
                    ));
                }
            }
        }
    }
    None
}

/// `requires.features` against what the driver's `hello` listed: `Some(why)` for the first one it
/// lacks, with the reason the driver gave in `hello.missing`.
fn lacks_feature(req: &Value, be: &Backend) -> Option<String> {
    for f in req["features"].as_array().into_iter().flatten() {
        let f = f.as_str()?;
        if !be.features.iter().any(|x| x == f) {
            let why = be.missing[f].as_str().unwrap_or("no reason given");
            return Some(format!("driver {} lacks feature {f}: {why}", be.driver));
        }
    }
    None
}

fn decode_utf8(v: &Value) -> String {
    use base64::Engine as _;
    let bytes = crate::eval::B64
        .decode(v.as_str().unwrap_or(""))
        .unwrap_or_default();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// One step: send, optionally poll until it matches, save, return the failure if any.
async fn run_step(
    link: &dyn Link,
    step: &Value,
    vars: &mut Vars,
    seq: &mut u64,
    must_succeed_only: bool,
) -> Result<(), String> {
    let poll = step.get("poll");
    let deadline = poll
        .map(|p| Instant::now() + Duration::from_millis(p["timeout_ms"].as_u64().unwrap_or(5000)));
    let every = Duration::from_millis(poll.and_then(|p| p["interval_ms"].as_u64()).unwrap_or(50));
    loop {
        *seq += 1;
        let req = request(*seq, step, vars);
        let mut resp = link.call(&req).await;
        let clip = |v: &Value, n: usize| v.to_string().chars().take(n).collect::<String>();
        if resp["id"] != req["id"] {
            return Err(format!(
                "the response must echo the request id\n      request: {}\n      response: {}",
                clip(&req, 400),
                clip(&resp, 600)
            ));
        }
        derive(&mut resp);
        if std::env::var_os("CONFORMANCE_TRACE").is_some() {
            println!("  > {}\n  < {}", clip(&req, 500), clip(&resp, 500));
        }
        let exp = if must_succeed_only {
            json!({"ok": {}})
        } else {
            subst(step.get("expect").unwrap_or(&json!({"ok": {}})), vars)
        };
        let verdict = (|| {
            for key in ["ok", "error"] {
                if let Some(e) = exp.get(key) {
                    let Some(a) = resp.get(key) else {
                        return Err(format!("expected `{key}`, got {resp}"));
                    };
                    matches(e, a, key)?;
                }
            }
            Ok(())
        })();
        let verdict = verdict.and_then(|()| {
            if let Some(Value::Object(save)) = step.get("save") {
                for (name, path) in save {
                    let root = resp.get("ok").unwrap_or(&resp);
                    let v = dotted(root, path.as_str().unwrap())
                        .ok_or_else(|| format!("save {name}: no {path} in {resp}"))?;
                    vars.insert(name.clone(), v.clone());
                }
            }
            Ok(())
        });
        match verdict {
            Ok(()) => return Ok(()),
            Err(why) => {
                if deadline.is_some_and(|d| Instant::now() < d) {
                    tokio::time::sleep(every).await;
                    continue;
                }
                return Err(format!(
                    "{why}\n      request: {}\n      response: {}",
                    clip(&req, 400),
                    clip(&resp, 600)
                ));
            }
        }
    }
}

fn all(sid: &str, steps: &[Value], status: Status, detail: &str) -> Vec<Check> {
    steps
        .iter()
        .map(|s| Check {
            id: format!("{sid}/{}", s["id"].as_str().unwrap_or("?")),
            status,
            detail: detail.to_string(),
        })
        .collect()
}

/// Run one scenario against a backend: the checks, one per step, in order. A relay scenario (steps
/// with `via`) needs two drivers; here it is skipped (see [`crate::relay`]).
pub async fn run_scenario(sc: &Value, be: &Backend) -> Vec<Check> {
    run_scenario_via(sc, be, None).await
}

/// [`run_scenario`] with a second driver on the same sandbox: steps with `"via": "B"` go to `peer`,
/// all others to `be`.
pub async fn run_scenario_via(sc: &Value, be: &Backend, peer: Option<&Backend>) -> Vec<Check> {
    let sid = sc["id"].as_str().unwrap();
    let steps = sc["steps"].as_array().unwrap();
    if peer.is_none() && crate::relay::is_relay(sc) {
        return all(
            sid,
            steps,
            Status::Skipped,
            "relay scenario: needs two drivers",
        );
    }
    if let Some(p) = sc.get("planned") {
        return all(sid, steps, Status::Skipped, &format!("planned ({p})"));
    }
    // An op the driver lacks makes the checks that need it unsupported. In `setup` that is the whole
    // scenario (nothing can be built); among the steps it is those steps alone, so a scenario that
    // only touches the op once (one error kind among a dozen) still proves the rest. A missing step
    // that `save`s a value the later steps use cannot be skipped over, so it takes the scenario.
    fn lacks_op<'a>(s: &'a Value, be: &Backend, peer: Option<&Backend>) -> Option<&'a str> {
        s["op"]
            .as_str()
            .filter(|op| !crate::relay::side(s, be, peer).ops.iter().any(|o| o == op))
    }
    let setup_lacks: Vec<&str> = sc["setup"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|s| lacks_op(s, be, peer))
        .collect();
    let step_lacks: Vec<&str> = steps.iter().filter_map(|s| lacks_op(s, be, peer)).collect();
    let per_step = setup_lacks.is_empty()
        && steps
            .iter()
            .all(|s| lacks_op(s, be, peer).is_none() || s.get("save").is_none());
    if !per_step && sc["steps"][0]["expect"]["error"]["kind"] != "unsupported" {
        let used: Vec<&str> = setup_lacks.into_iter().chain(step_lacks).collect();
        return all(sid, steps, Status::Unsupported, &format!("no op {used:?}"));
    }
    // A feature the driver lacks makes the whole scenario unsupported, with the driver's reason.
    for b in std::iter::once(be).chain(peer) {
        if let Some(why) = lacks_feature(&sc["requires"], b) {
            return all(sid, steps, Status::Unsupported, &why);
        }
    }
    for b in std::iter::once(be).chain(peer) {
        if let Some(why) = unmet(&sc["requires"], b).await {
            return all(sid, steps, Status::Skipped, &why);
        }
    }

    // A scenario may connect with its own credentials (a wrong bearer); the rest share the
    // backend's driver.
    let own;
    let link: &dyn Link = match sc.get("connect") {
        Some(c) => {
            let bearer = c.get("bearer").map(|b| b.as_str().map(str::to_string));
            own = be
                .connector
                .connect(bearer, c["url"].as_str().map(str::to_string))
                .await;
            own.as_ref()
        }
        None => be.link.as_ref(),
    };

    let root = format!("{}/{}", be.work, sid.replace('.', "_"));
    let mut vars = Vars::new();
    vars.insert("ROOT".into(), json!(root));
    vars.insert("WORK".into(), json!(be.work));
    for (k, v) in &be.vars {
        vars.insert(k.clone(), json!(v));
    }
    vars.insert(
        "SCOPE".into(),
        json!(format!("sc{:x}", {
            let d = sha2::Sha256::digest(format!("{}{sid}", be.name));
            u64::from_be_bytes(d[..8].try_into().unwrap())
        })),
    );
    let mut seq = 0u64;
    let boot = json!({"op": "exec", "command": "mkdir", "args": ["-p", "${ROOT}"]});
    if let Err(why) = run_step(be.link.as_ref(), &boot, &mut vars, &mut seq, true).await {
        let mut out = vec![Check {
            id: format!("{sid}/boot"),
            status: Status::Fail,
            detail: format!("create the scenario root: {why}"),
        }];
        out.extend(all(sid, steps, Status::Skipped, "no scenario root"));
        return out;
    }
    for (i, s) in sc["setup"].as_array().into_iter().flatten().enumerate() {
        let l = crate::relay::side(s, be, peer).link.as_ref();
        let l = if peer.is_some() { l } else { link };
        if let Err(why) = run_step(l, s, &mut vars, &mut seq, true).await {
            let mut out = vec![Check {
                id: format!("{sid}/setup[{i}]"),
                status: Status::Fail,
                detail: why,
            }];
            out.extend(all(sid, steps, Status::Skipped, "setup failed"));
            return out;
        }
    }
    let mut out = Vec::with_capacity(steps.len());
    for step in steps {
        let check = format!(
            "{sid}/{}",
            step["id"].as_str().expect("every step has an id")
        );
        let side = crate::relay::side(step, be, peer);
        if let Some(op) = lacks_op(step, be, peer).filter(|_| per_step)
            && step["expect"]["error"]["kind"] != "unsupported"
        {
            out.push(Check {
                id: check,
                status: Status::Unsupported,
                detail: format!("no op [{op:?}]"),
            });
            continue;
        }
        let known = match step.get("xfail") {
            Some(x) => {
                x["drivers"]
                    .as_array()
                    .is_some_and(|d| d.iter().any(|n| n == side.driver.as_str()))
                    && unmet(&x["when"], side).await.is_none()
            }
            None => false,
        };
        let (status, detail) = match (
            run_step(
                if peer.is_some() {
                    side.link.as_ref()
                } else {
                    link
                },
                step,
                &mut vars,
                &mut seq,
                false,
            )
            .await,
            known,
        ) {
            (Ok(()), false) => (Status::Pass, String::new()),
            (Err(why), false) => (Status::Fail, why),
            (Err(_), true) => (
                Status::KnownDefect,
                step["xfail"]["reason"].as_str().unwrap_or("").to_string(),
            ),
            (Ok(()), true) => (
                Status::Fail,
                "marked xfail for this driver but it passed; remove the mark".to_string(),
            ),
        };
        out.push(Check {
            id: check,
            status,
            detail,
        });
    }
    // Best-effort cleanup: the scope's jobs, then the directory.
    let _ = be
        .link
        .call(&json!({"id": 0, "op": "job_wipe", "scope": vars["SCOPE"]}))
        .await;
    let _ = exec(be, "rm", &["-rf", &root]).await;
    out
}

/// Every scenario file in `dir`, in name order, as `(file name, parsed)`.
pub fn load_scenarios(dir: &Path) -> Result<Vec<(String, Value)>, String> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .map(|e| e.map(|e| e.path()))
        .collect::<Result<_, _>>()
        .map_err(|e| e.to_string())?;
    files.retain(|p| p.extension().is_some_and(|e| e == "json"));
    files.sort();
    files
        .into_iter()
        .map(|p| {
            let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            let v: Value = serde_json::from_str(&text)
                .map_err(|e| format!("{}: not JSON: {e}", p.display()))?;
            Ok((p.file_name().unwrap().to_string_lossy().into_owned(), v))
        })
        .collect()
}

/// `*` matches any run of characters, `?` one; nothing else is special.
pub fn glob_match(pat: &str, s: &str) -> bool {
    fn go(p: &[u8], s: &[u8]) -> bool {
        match p.first() {
            None => s.is_empty(),
            Some(b'*') => (0..=s.len()).any(|i| go(&p[1..], &s[i..])),
            Some(b'?') => !s.is_empty() && go(&p[1..], &s[1..]),
            Some(c) => s.first() == Some(c) && go(&p[1..], &s[1..]),
        }
    }
    go(pat.as_bytes(), s.as_bytes())
}
