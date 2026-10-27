//! The reference check for `sdk/conformance`: every scenario file is replayed through the agent's own
//! `ShellFs` + `HttpExecRunner` (the code the spec was derived from) against a loopback `execd`, in
//! several backend profiles. If this passes, the scenarios say what the agent does today; a client in
//! another language that passes the same files does what the agent does.
//!
//! This file is also the reference *harness*: its scenario format, variable substitution, matchers
//! and `requires` evaluation are the ones `sdk/conformance/DRIVER.md` specifies, so it is the thing to
//! read when a rule there is unclear. The driver it feeds is in `sdk_conformance_driver.rs`.
//!
//! A scenario marked `planned` is skipped, visibly, and never counted as a pass.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use base64::Engine as _;
use beyond_ai_test_support::exec_mock::{ExecMock, MockOptions, Protocol};
use serde_json::{Map, Value, json};
use sha2::Digest as _;

use crate::sdk_conformance_driver::{B64, CapsOverride, Config, Driver};

const BEARER: &str = "conformance-secret";

fn scenario_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../sdk/conformance/scenarios")
}

/// A backend configuration: what the endpoint is, and which probe results the driver is told to
/// believe instead (`CONFORMANCE_CAPS`), so one machine can walk every rung of the fallback ladder.
struct Profile {
    name: &'static str,
    options: MockOptions,
    bearer: bool,
    caps: CapsOverride,
    /// The suites this profile runs. The two profiles served by the mock's own (deliberately
    /// imperfect) endpoint exist to prove the client's probes and fallbacks, so they run the
    /// suites that go through those and leave the endpoint-behaviour suites to execd.
    suites: &'static [&'static str],
}

fn profiles() -> Vec<Profile> {
    let base = || Profile {
        name: "",
        options: MockOptions::default(),
        bearer: false,
        caps: CapsOverride::default(),
        suites: &[],
    };
    vec![
        Profile {
            name: "execd",
            ..base()
        },
        Profile {
            name: "execd-bearer",
            bearer: true,
            ..base()
        },
        Profile {
            // A v1 endpoint: ignores `stdin_base64`, so the probe finds no stdin by itself.
            name: "v1-no-stdin",
            options: MockOptions {
                honors_stdin: false,
                ..MockOptions::default()
            },
            suites: &["files", "errors", "helpers"],
            ..base()
        },
        Profile {
            // An endpoint that never learned streaming: `stream` is an unknown field to it.
            name: "v11-endpoint",
            options: MockOptions {
                protocol: Protocol::V11,
                ..MockOptions::default()
            },
            suites: &["files", "errors", "helpers"],
            ..base()
        },
        Profile {
            name: "execd-no-stdin",
            caps: CapsOverride {
                stdin: Some(false),
                ..CapsOverride::default()
            },
            ..base()
        },
        Profile {
            name: "execd-grep",
            caps: CapsOverride {
                search_engine: Some("grep"),
                ..CapsOverride::default()
            },
            ..base()
        },
        Profile {
            name: "execd-busybox-grep",
            caps: CapsOverride {
                search_engine: Some("busybox-grep"),
                find_printf: Some(false),
                ..CapsOverride::default()
            },
            ..base()
        },
        Profile {
            // Native jobs off: the endpoint offers plain exec only, so jobs are the baseline.
            name: "execd-baseline-jobs",
            options: MockOptions {
                jobs: false,
                ..MockOptions::default()
            },
            ..base()
        },
    ]
}

// ---------------------------------------------------------------------------------------------
// Substitution and derived views
// ---------------------------------------------------------------------------------------------

type Vars = BTreeMap<String, Value>;

/// Replace `${name}` in every string. A string that is exactly one variable becomes the saved
/// value, type and all; otherwise the value is interpolated as text. `{"$repeat": ...}` expands to
/// base64.
fn subst(v: &Value, vars: &Vars) -> Value {
    match v {
        Value::String(s) => {
            if let Some(name) = s.strip_prefix("${").and_then(|r| r.strip_suffix('}'))
                && !name.contains("${")
                && let Some(val) = vars.get(name)
            {
                return val.clone();
            }
            let mut out = s.clone();
            for (k, val) in vars {
                let text = match val {
                    Value::String(t) => t.clone(),
                    other => other.to_string(),
                };
                out = out.replace(&format!("${{{k}}}"), &text);
            }
            Value::String(out)
        }
        Value::Array(a) => Value::Array(a.iter().map(|x| subst(x, vars)).collect()),
        Value::Object(o) => {
            if o.len() == 1
                && let Some(r) = o.get("$repeat")
            {
                let chunk = B64
                    .decode(r["b64"].as_str().expect("$repeat.b64"))
                    .expect("$repeat.b64 is base64");
                let times = r["times"].as_u64().expect("$repeat.times") as usize;
                let mut bytes = chunk.repeat(times);
                if let Some(len) = r["len"].as_u64() {
                    bytes.truncate(len as usize);
                }
                return Value::String(B64.encode(bytes));
            }
            Value::Object(o.iter().map(|(k, x)| (k.clone(), subst(x, vars))).collect())
        }
        other => other.clone(),
    }
}

/// Add `<name>_utf8`, `<name>_len` and `<name>_sha256` beside every `<name>_b64` string, so a
/// scenario can state an expectation about bytes in text or by digest.
fn derive(v: &mut Value) {
    match v {
        Value::Object(o) => {
            let mut extra = Vec::new();
            for (k, x) in o.iter() {
                if let (Some(base), Value::String(s)) = (k.strip_suffix("_b64"), x) {
                    let bytes = B64.decode(s).unwrap_or_default();
                    extra.push((
                        format!("{base}_utf8"),
                        json!(String::from_utf8_lossy(&bytes)),
                    ));
                    extra.push((format!("{base}_len"), json!(bytes.len())));
                    extra.push((
                        format!("{base}_sha256"),
                        json!(hex::encode(sha2::Sha256::digest(&bytes))),
                    ));
                }
            }
            o.extend(extra);
            for x in o.values_mut() {
                derive(x);
            }
        }
        Value::Array(a) => a.iter_mut().for_each(derive),
        _ => {}
    }
}

fn dotted<'a>(v: &'a Value, path: &str) -> Option<&'a Value> {
    path.split('.').try_fold(v, |cur, key| match cur {
        Value::Array(a) => key.parse::<usize>().ok().and_then(|i| a.get(i)),
        _ => cur.get(key),
    })
}

// ---------------------------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------------------------

fn matches(exp: &Value, act: &Value, at: &str) -> Result<(), String> {
    if let Value::Object(o) = exp
        && o.len() == 1
        && let Some((k, arg)) = o.iter().next()
        && k.starts_with('$')
    {
        return matcher(k, arg, act, at);
    }
    match (exp, act) {
        (Value::Object(e), Value::Object(a)) => {
            for (k, ev) in e {
                matches(ev, a.get(k).unwrap_or(&Value::Null), &format!("{at}.{k}"))?;
            }
            Ok(())
        }
        (Value::Object(e), other) => Err(format!("{at}: expected an object {e:?}, got {other}")),
        (Value::Array(e), Value::Array(a)) => {
            if e.len() != a.len() {
                return Err(format!(
                    "{at}: expected {} items, got {}: {act}",
                    e.len(),
                    a.len()
                ));
            }
            for (i, (ev, av)) in e.iter().zip(a).enumerate() {
                matches(ev, av, &format!("{at}[{i}]"))?;
            }
            Ok(())
        }
        (Value::Number(e), Value::Number(a)) if e.as_f64() == a.as_f64() => Ok(()),
        (e, a) if e == a => Ok(()),
        (e, a) => Err(format!("{at}: expected {e}, got {a}")),
    }
}

fn matcher(name: &str, arg: &Value, act: &Value, at: &str) -> Result<(), String> {
    let fail = |why: String| Err(format!("{at}: {name} {arg}: {why} (got {act})"));
    match name {
        "$any" => {
            if act.is_null() {
                return fail("absent".into());
            }
            Ok(())
        }
        "$absent" => {
            if act.is_null() {
                Ok(())
            } else {
                fail("present".into())
            }
        }
        "$type" => {
            let t = match act {
                Value::Null => "null",
                Value::Bool(_) => "boolean",
                Value::Number(_) => "number",
                Value::String(_) => "string",
                Value::Array(_) => "array",
                Value::Object(_) => "object",
            };
            if Some(t) == arg.as_str() {
                Ok(())
            } else {
                fail(format!("is {t}"))
            }
        }
        "$re" => {
            let re = regex::Regex::new(arg.as_str().unwrap()).expect("$re is a regex");
            match act.as_str() {
                Some(s) if re.is_match(s) => Ok(()),
                _ => fail("no match".into()),
            }
        }
        "$contains" => match act.as_str() {
            Some(s) if s.contains(arg.as_str().unwrap()) => Ok(()),
            _ => fail("not contained".into()),
        },
        "$gte" | "$lte" | "$gt" | "$lt" => {
            let (a, b) = (act.as_f64(), arg.as_f64());
            let ok = match (name, a, b) {
                ("$gte", Some(a), Some(b)) => a >= b,
                ("$lte", Some(a), Some(b)) => a <= b,
                ("$gt", Some(a), Some(b)) => a > b,
                ("$lt", Some(a), Some(b)) => a < b,
                _ => false,
            };
            if ok {
                Ok(())
            } else {
                fail("comparison failed".into())
            }
        }
        "$oneof" => {
            let any = arg
                .as_array()
                .unwrap()
                .iter()
                .any(|e| matches(e, act, at).is_ok());
            if any {
                Ok(())
            } else {
                fail("none equal".into())
            }
        }
        "$not" => {
            if matches(arg, act, at).is_err() {
                Ok(())
            } else {
                fail("matched".into())
            }
        }
        "$len" => {
            let n = act
                .as_array()
                .map(Vec::len)
                .or_else(|| act.as_str().map(str::len));
            if n == arg.as_u64().map(|n| n as usize) {
                Ok(())
            } else {
                fail(format!("length {n:?}"))
            }
        }
        // The same items in any order.
        "$unordered" | "$includes" => {
            let Some(items) = act.as_array() else {
                return fail("not an array".into());
            };
            let want = arg.as_array().unwrap();
            if name == "$unordered" && want.len() != items.len() {
                return fail(format!("{} items", items.len()));
            }
            let mut used = vec![false; items.len()];
            for w in want {
                let slot = items
                    .iter()
                    .enumerate()
                    .position(|(i, a)| !used[i] && matches(w, a, at).is_ok());
                match slot {
                    Some(i) => used[i] = true,
                    None => return fail(format!("no item matches {w}")),
                }
            }
            Ok(())
        }
        other => Err(format!("{at}: unknown matcher {other}")),
    }
}

// ---------------------------------------------------------------------------------------------
// Running
// ---------------------------------------------------------------------------------------------

#[derive(Default)]
struct Tally {
    passed: usize,
    failed: usize,
    /// Checks a scenario marks as a known defect of this driver: they must fail, and are reported.
    xfailed: usize,
    /// Checks of scenarios that use an op the driver does not list in `hello`. Never a pass.
    unsupported: usize,
    skipped: usize,
    scenarios_run: usize,
    scenarios_skipped: usize,
    failures: Vec<String>,
}

struct Backend {
    driver: Driver,
    caps: Value,
    cfg: Config,
    auth: bool,
    work: PathBuf,
}

fn request(id: u64, step: &Value, vars: &Vars) -> Value {
    let mut req = Map::new();
    for (k, v) in step.as_object().unwrap() {
        if !matches!(k.as_str(), "id" | "expect" | "save" | "poll" | "note") {
            req.insert(k.clone(), subst(v, vars));
        }
    }
    req.insert("id".into(), json!(id));
    Value::Object(req)
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
            "jobs" => {
                let have = be.caps["jobs"].as_str().unwrap_or("none");
                let ok = match want.as_str() {
                    Some("any") => have != "none",
                    Some(w) => w == have,
                    None => false,
                };
                if !ok {
                    return Some(format!("requires jobs={want}, backend has {have}"));
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
            "not_root" => {
                let r = be
                    .driver
                    .handle(&json!({"id": 0, "op": "exec", "command": "id", "args": ["-u"]}))
                    .await;
                let root = r["ok"]["stdout_utf8"].as_str().unwrap_or("0").trim() == "0";
                if root == want.as_bool().unwrap_or(true) {
                    return Some("requires a non-root sandbox user".to_string());
                }
            }
            "commands" => {
                for c in want.as_array().unwrap() {
                    let r = be
                        .driver
                        .handle(&json!({"id": 0, "op": "exec", "command": "sh",
                            "args": ["-c", "command -v \"$1\"", "sh", c]}))
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

/// One step: send, optionally poll until it matches, save, return the failure if any.
async fn run_step(
    be: &Backend,
    drv: &Driver,
    step: &Value,
    vars: &mut Vars,
    seq: &mut u64,
    must_succeed_only: bool,
) -> Result<(), String> {
    let poll = step.get("poll");
    let deadline = poll
        .map(|p| Instant::now() + Duration::from_millis(p["timeout_ms"].as_u64().unwrap_or(5000)));
    let every = Duration::from_millis(poll.and_then(|p| p["interval_ms"].as_u64()).unwrap_or(50));
    let _ = be;
    loop {
        *seq += 1;
        let req = request(*seq, step, vars);
        let mut resp = drv.handle(&req).await;
        assert_eq!(
            resp["id"], req["id"],
            "the response must echo the request id"
        );
        derive(&mut resp);
        if std::env::var_os("CONFORMANCE_TRACE").is_some() {
            let clip = |v: &Value| v.to_string().chars().take(500).collect::<String>();
            println!("  > {}\n  < {}", clip(&req), clip(&resp));
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
        match verdict {
            Ok(()) => {
                if let Some(Value::Object(save)) = step.get("save") {
                    for (name, path) in save {
                        let root = resp.get("ok").unwrap_or(&resp);
                        let v = dotted(root, path.as_str().unwrap())
                            .unwrap_or_else(|| panic!("save {name}: no {path} in {resp}"));
                        vars.insert(name.clone(), v.clone());
                    }
                }
                return Ok(());
            }
            Err(why) => {
                if deadline.is_some_and(|d| Instant::now() < d) {
                    tokio::time::sleep(every).await;
                    continue;
                }
                let shown = resp.to_string();
                let shown: String = shown.chars().take(600).collect();
                return Err(format!(
                    "{why}\n      request: {}\n      response: {shown}",
                    {
                        let r = req.to_string();
                        r.chars().take(400).collect::<String>()
                    }
                ));
            }
        }
    }
}

async fn run_scenario(sc: &Value, profile: &Profile, be: &Backend, suite: &str, tally: &mut Tally) {
    let sid = sc["id"].as_str().unwrap();
    let steps = sc["steps"].as_array().unwrap();
    if let Some(p) = sc.get("planned") {
        println!("[{}] skipped: planned ({p}) {sid}", profile.name);
        tally.scenarios_skipped += 1;
        tally.skipped += steps.len();
        return;
    }
    let ops = Driver::ops();
    let used: Vec<&str> = sc["setup"]
        .as_array()
        .into_iter()
        .flatten()
        .chain(steps)
        .filter_map(|s| s["op"].as_str())
        .filter(|op| !ops.contains(op))
        .collect();
    if !used.is_empty() && sc["steps"][0]["expect"]["error"]["kind"] != "unsupported" {
        println!("[{}] unsupported: {used:?}: {sid}", profile.name);
        tally.unsupported += steps.len();
        return;
    }
    if let Some(why) = unmet(&sc["requires"], be).await {
        println!("[{}] skipped: {why}: {sid}", profile.name);
        tally.scenarios_skipped += 1;
        tally.skipped += steps.len();
        return;
    }
    tally.scenarios_run += 1;

    // A scenario may connect with its own credentials (a wrong bearer); the rest share the
    // backend's driver.
    let own;
    let drv = match sc.get("connect") {
        Some(c) => {
            let mut cfg = be.cfg.clone();
            if let Some(bearer) = c.get("bearer") {
                cfg.bearer = bearer.as_str().map(str::to_string);
            }
            if let Some(url) = c["url"].as_str() {
                cfg.url = url.to_string();
            }
            own = Driver::connect(&cfg).await;
            &own
        }
        None => &be.driver,
    };

    let root = be.work.join(sid.replace('.', "_"));
    let mut vars = Vars::new();
    vars.insert("ROOT".into(), json!(root.to_string_lossy()));
    vars.insert("WORK".into(), json!(be.work.to_string_lossy()));
    vars.insert(
        "SCOPE".into(),
        json!(format!("sc{:x}", {
            let d = sha2::Sha256::digest(format!("{}{sid}", profile.name));
            u64::from_be_bytes(d[..8].try_into().unwrap())
        })),
    );
    let mut seq = 0u64;
    let boot = json!({"op": "exec", "command": "mkdir", "args": ["-p", "${ROOT}"]});
    run_step(be, &be.driver, &boot, &mut vars, &mut seq, true)
        .await
        .expect("create the scenario root");
    for (i, s) in sc["setup"].as_array().into_iter().flatten().enumerate() {
        if let Err(why) = run_step(be, drv, s, &mut vars, &mut seq, true).await {
            tally.failed += 1;
            tally.skipped += steps.len();
            tally
                .failures
                .push(format!("[{}] {sid} setup[{i}]: {why}", profile.name));
            return;
        }
    }
    for step in steps {
        let check = format!(
            "{sid}/{}",
            step["id"].as_str().expect("every step has an id")
        );
        let known = match step.get("xfail") {
            Some(x) => {
                x["drivers"]
                    .as_array()
                    .is_some_and(|d| d.iter().any(|n| n == "agent-reference"))
                    && unmet(&x["when"], be).await.is_none()
            }
            None => false,
        };
        match (
            run_step(be, drv, step, &mut vars, &mut seq, false).await,
            known,
        ) {
            (Ok(()), false) => tally.passed += 1,
            (Err(why), false) => {
                tally.failed += 1;
                tally
                    .failures
                    .push(format!("[{}] {check}: {why}", profile.name));
            }
            (Err(_), true) => {
                tally.xfailed += 1;
                println!(
                    "[{}] known defect: {check}: {}",
                    profile.name, step["xfail"]["reason"]
                );
            }
            (Ok(()), true) => {
                tally.failed += 1;
                tally.failures.push(format!(
                    "[{}] {check}: marked xfail for this driver but it passed; remove the mark",
                    profile.name
                ));
            }
        }
    }
    let _ = suite;
    // Best-effort cleanup: the scope's jobs, then the directory.
    let _ = be
        .driver
        .handle(&json!({"id": 0, "op": "job_wipe", "scope": vars["SCOPE"]}))
        .await;
    let _ = std::fs::remove_dir_all(&root);
}

/// Every scenario file, in name order.
fn load() -> Vec<(String, Value)> {
    let mut files: Vec<_> = std::fs::read_dir(scenario_dir())
        .expect("sdk/conformance/scenarios exists")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();
    files
        .into_iter()
        .map(|p| {
            let text = std::fs::read_to_string(&p).unwrap();
            let v: Value = serde_json::from_str(&text)
                .unwrap_or_else(|e| panic!("{}: not JSON: {e}", p.display()));
            (p.file_name().unwrap().to_string_lossy().into_owned(), v)
        })
        .collect()
}

#[test]
fn scenario_files_are_well_formed() {
    let files = load();
    assert!(!files.is_empty());
    let mut ids = std::collections::BTreeSet::new();
    for (name, f) in &files {
        let suite = f["suite"]
            .as_str()
            .unwrap_or_else(|| panic!("{name}: no suite"));
        assert_eq!(
            format!("{suite}.json"),
            *name,
            "a file is named for its suite"
        );
        for sc in f["scenarios"].as_array().unwrap() {
            let id = sc["id"].as_str().unwrap();
            assert!(
                id.starts_with(&format!("{suite}.")),
                "{id} is not in {suite}"
            );
            assert!(ids.insert(id.to_string()), "duplicate scenario id {id}");
            assert!(sc["description"].is_string(), "{id}: no description");
            let mut step_ids = std::collections::BTreeSet::new();
            for s in sc["steps"].as_array().unwrap() {
                let sid = s["id"]
                    .as_str()
                    .unwrap_or_else(|| panic!("{id}: step without id"));
                assert!(step_ids.insert(sid), "{id}: duplicate step id {sid}");
                assert!(s["op"].is_string(), "{id}/{sid}: no op");
            }
            assert!(
                !sc["steps"].as_array().unwrap().is_empty(),
                "{id}: no steps"
            );
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reference_agent_passes_every_scenario() {
    let files = load();
    let mut grand = Tally::default();
    for profile in profiles() {
        let dir = tempfile::tempdir().unwrap();
        let work = dir.path().canonicalize().unwrap();
        let mut options = profile.options.clone();
        if profile.bearer {
            options.require_bearer = Some(BEARER.to_string());
        }
        let mock = ExecMock::start_with(&work, options).await;
        let cfg = Config {
            url: mock.url.clone(),
            bearer: profile.bearer.then(|| BEARER.to_string()),
            caps: profile.caps,
            jobs_root: work.join(".jobs"),
        };
        let driver = Driver::connect(&cfg).await;
        let caps = driver.handle(&json!({"id": 0, "op": "capabilities"})).await["ok"].clone();
        let be = Backend {
            driver,
            caps,
            cfg,
            auth: profile.bearer,
            work,
        };
        let mut tally = Tally::default();
        for (_, file) in &files {
            let suite = file["suite"].as_str().unwrap();
            if !profile.suites.is_empty() && !profile.suites.contains(&suite) {
                continue;
            }
            for sc in file["scenarios"].as_array().unwrap() {
                run_scenario(sc, &profile, &be, suite, &mut tally).await;
            }
        }
        println!(
            "== {}: {} checks passed, {} failed, {} known defects, {} unsupported, {} skipped ({} scenarios run, {} skipped)",
            profile.name,
            tally.passed,
            tally.failed,
            tally.xfailed,
            tally.unsupported,
            tally.skipped,
            tally.scenarios_run,
            tally.scenarios_skipped
        );
        grand.xfailed += tally.xfailed;
        grand.unsupported += tally.unsupported;
        grand.passed += tally.passed;
        grand.failed += tally.failed;
        grand.skipped += tally.skipped;
        grand.scenarios_run += tally.scenarios_run;
        grand.scenarios_skipped += tally.scenarios_skipped;
        grand.failures.extend(tally.failures);
    }
    println!(
        "== total: {} checks passed, {} failed, {} known defects, {} unsupported, {} skipped",
        grand.passed, grand.failed, grand.xfailed, grand.unsupported, grand.skipped
    );
    assert!(
        grand.failures.is_empty(),
        "{} check(s) failed:\n{}",
        grand.failures.len(),
        grand.failures.join("\n")
    );
    assert!(grand.passed > 0);
    assert_eq!(grand.unsupported, 0, "the reference driver has every op");
}

/// The wire shapes FILES-OVER-EXEC.md states as exact: what a write sends with stdin, and without.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn write_wire_shapes_match_the_spec() {
    for stdin in [true, false] {
        let dir = tempfile::tempdir().unwrap();
        let work = dir.path().canonicalize().unwrap();
        let mock = ExecMock::start(&work, stdin).await;
        let cfg = Config {
            url: mock.url.clone(),
            jobs_root: work.join(".jobs"),
            ..Config::default()
        };
        let d = Driver::connect(&cfg).await;
        let n = 100_000usize;
        let before = mock.bodies().len();
        let r = d
            .handle(
                &json!({"id": 1, "op": "write", "path": work.join("f").to_string_lossy(),
                "data_b64": B64.encode(vec![b'x'; n])}),
            )
            .await;
        assert!(r.get("ok").is_some(), "{r}");
        let sent = &mock.bodies()[before..];
        for b in sent {
            assert_eq!(b["command"], "sh");
            assert_eq!(b["args"][0], "-c");
            assert_eq!(b["args"][2], "sh");
        }
        if stdin {
            assert_eq!(sent.len(), 1, "one command with the content on stdin");
            assert_eq!(sent[0]["args"].as_array().unwrap().len(), 6);
            assert_eq!(sent[0]["args"][5], n.to_string());
            assert_eq!(
                sent[0]["stdin_base64"].as_str().unwrap().len(),
                n.div_ceil(3) * 4
            );
        } else {
            assert_eq!(sent.len(), n.div_ceil(48 * 1024), "one command per 48 KiB");
            for (i, b) in sent.iter().enumerate() {
                let a = b["args"].as_array().unwrap();
                assert_eq!(
                    a.len(),
                    9,
                    "-c, script, sh, target, suffix, chunk, mode, commit, size"
                );
                assert!(a[5].as_str().unwrap().len() <= 65536);
                assert_eq!(a[6], if i == 0 { "new" } else { "append" });
                assert_eq!(a[7], if i + 1 == sent.len() { "commit" } else { "" });
                assert_eq!(a[8], n.to_string());
                assert!(b.get("stdin_base64").is_none());
            }
        }
    }
}
