//! One backend with every registered driver connected to it, and the lockstep runner: each op goes to
//! every driver (each in its own fresh root), the answers are compared with each other and, on a
//! local backend, with the filesystem itself.
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use sandbox_conformance::backends::{self, Binaries, Started};
use sandbox_conformance::eval::{Vars, subst};
use sandbox_conformance::link::ProcessConnector;
use sandbox_conformance::matrix;
use sandbox_conformance::run::Backend;
use serde_json::{Value, json};

use super::norm::canon;
use super::ops::{Guard, Op};
use super::oracle::{self, Ctx, search_truth::Engine};

pub struct Drv {
    pub name: String,
    pub be: Backend,
}

pub struct World {
    pub backend: String,
    pub started: Started,
    pub drivers: Vec<Drv>,
    /// The sandbox's filesystem is this host's, so `std::fs` is ground truth.
    pub local: bool,
    cases: AtomicU64,
    /// Open bugs seen and tolerated, by id.
    pub known: Mutex<BTreeMap<&'static str, u64>>,
    pub ops_run: AtomicU64,
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

pub fn bins() -> Binaries {
    let guess = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent()?.parent().map(|p| p.join("execd")))
        .filter(|p| p.is_file());
    Binaries {
        execd: std::env::var_os("EXECD_BIN").map(PathBuf::from).or(guess),
        execd_static: None,
    }
}

pub fn docker_available() -> bool {
    std::process::Command::new("docker")
        .arg("info")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

impl World {
    /// Start the backend and connect every driver of the registry (`DIFFERENTIAL_DRIVERS` narrows
    /// it to a comma-separated list of names).
    pub async fn start(backend: &str) -> Result<World, String> {
        std::env::set_current_dir(repo_root()).map_err(|e| e.to_string())?;
        let build = std::env::var_os("DIFFERENTIAL_BUILD").is_some();
        let only: Option<Vec<String>> = std::env::var("DIFFERENTIAL_DRIVERS")
            .ok()
            .map(|v| v.split(',').map(str::to_string).collect());
        let mut registry = matrix::registry(Path::new("sdk/conformance/drivers.toml"), build)?;
        registry.retain(|d| only.as_ref().is_none_or(|o| o.contains(&d.name)));
        let started = backends::start(backend, &bins()).await?;
        let mut drivers = Vec::new();
        for d in registry {
            let connector = ProcessConnector {
                cmd: d.cmd.clone(),
                url: started.url.clone(),
                bearer: started.bearer.clone(),
                caps: started.caps.clone(),
            };
            let be = Backend::open(
                &format!("{}__{backend}", d.name),
                Box::new(connector),
                started.bearer.is_some(),
                &started.work,
            )
            .await;
            if be.ops.is_empty() {
                return Err(format!(
                    "driver {} did not answer hello (is it built? DIFFERENTIAL_BUILD=1 builds the registry)",
                    d.name
                ));
            }
            drivers.push(Drv { name: d.name, be });
        }
        if drivers.len() < 2 {
            return Err("a differential run needs at least two drivers".into());
        }
        Ok(World {
            backend: backend.to_string(),
            local: backend.starts_with("local") || backend == "embedded",
            started,
            drivers,
            cases: AtomicU64::new(0),
            known: Mutex::default(),
            ops_run: AtomicU64::new(0),
        })
    }

    /// Every driver's `capabilities`, which must agree (they probe the same sandbox).
    pub async fn capabilities_agree(&self) -> Result<(), String> {
        let mut seen: Vec<(String, Value)> = vec![];
        for d in &self.drivers {
            let mut c = d.be.caps.clone();
            // `read_window` and friends are probe results; a driver may report extra keys.
            if let Some(o) = c.as_object_mut() {
                o.retain(|k, _| {
                    [
                        "stdin",
                        "stream",
                        "jobs",
                        "search_engine",
                        "find_printf",
                        "read_window",
                    ]
                    .contains(&k.as_str())
                });
            }
            seen.push((d.name.clone(), c));
        }
        if seen.iter().any(|(_, c)| *c != seen[0].1) {
            return Err(format!(
                "drivers report different capabilities of the same sandbox: {}",
                seen.iter()
                    .map(|(n, c)| format!("{n}: {c}"))
                    .collect::<Vec<_>>()
                    .join("; ")
            ));
        }
        Ok(())
    }
}

struct Run<'a> {
    d: &'a Drv,
    root: String,
    seq: u64,
    vars: Vars,
    ctx: Option<Ctx>,
}

impl Run<'_> {
    async fn call(&mut self, step: &Value) -> (Value, Value) {
        self.seq += 1;
        let mut req = subst(step, &self.vars);
        req["id"] = json!(self.seq);
        let mut resp = self.d.be.link.call(&req).await;
        if resp["id"] != req["id"] {
            resp = json!({"error": {"kind": "bad_id", "detail": format!("answered id {}", resp["id"])}});
        }
        (req, resp)
    }
}

pub struct Failure {
    pub text: String,
}

pub async fn run_case(world: &World, ops: &[Op]) -> Result<(), Failure> {
    let case = world.cases.fetch_add(1, Ordering::Relaxed);
    let tag = std::process::id();
    let mut runs: Vec<Run> = world
        .drivers
        .iter()
        .enumerate()
        .map(|(idx, d)| {
            // No `-DIGITS-` in the name (busybox grep cannot tell it from a context-line
            // separator), and the same length for every driver (a symlink's size is the length of
            // its target, a path under the root).
            let root = format!("{}/df{tag}c{case}d{idx}", world.started.work);
            let mut vars = Vars::new();
            vars.insert("ROOT".into(), json!(root));
            let ctx = world.local.then(|| Ctx {
                root: PathBuf::from(&root),
                engine: match d.be.caps["search_engine"].as_str() {
                    Some("rg") => Engine::Rg,
                    Some("busybox-grep") => Engine::Busybox,
                    _ => Engine::Grep,
                },
                find_printf: d.be.caps["find_printf"] != false,
                busybox: d.be.caps["search_engine"] == "busybox-grep",
            });
            Run {
                d,
                root,
                seq: 0,
                vars,
                ctx,
            }
        })
        .collect();
    let result = lockstep(world, &mut runs, ops).await;
    // Whatever happened, give the tree back (a directory the case made unreadable first).
    let clean = json!({"op": "exec", "command": "sh", "args": ["-c", "chmod -R u+rwx \"$1\" 2>/dev/null; rm -rf -- \"$1\"", "sh", "${ROOT}"]});
    futures::future::join_all(runs.iter_mut().map(|r| async {
        let _ = r.call(&clean).await;
    }))
    .await;
    result
}

async fn lockstep(world: &World, runs: &mut [Run<'_>], ops: &[Op]) -> Result<(), Failure> {
    let mkroot = json!({"op": "exec", "command": "mkdir", "args": ["-p", "${ROOT}"]});
    let made = futures::future::join_all(runs.iter_mut().map(|r| r.call(&mkroot))).await;
    for (r, (_, resp)) in runs.iter().zip(made) {
        if resp.get("ok").is_none() {
            return Err(Failure {
                text: format!(
                    "{}: cannot create the case root {}: {resp}",
                    r.d.name, r.root
                ),
            });
        }
    }
    // The reference history, for printing the sequence as a scenario.
    let mut history: Vec<Value> = vec![];
    let mut mtimes: Vec<HashMap<usize, Value>> = vec![HashMap::new(); runs.len()];
    for (i, op) in ops.iter().enumerate() {
        world.ops_run.fetch_add(1, Ordering::Relaxed);
        let step = op.step();
        for (r, m) in runs.iter_mut().zip(&mtimes) {
            for (j, v) in m {
                r.vars.insert(format!("m{j}"), v.clone());
            }
        }
        // An op is compared across the drivers whose `hello` lists it (`ks`). A driver without it
        // is not asked, takes no part in the comparison and is never a divergence; the others are
        // compared with each other and with the filesystem exactly as if it were absent.
        let ks: Vec<usize> = (0..runs.len())
            .filter(|&k| runs[k].d.be.ops.iter().any(|o| o == op.name()))
            .collect();
        if ks.is_empty() {
            return Err(Failure {
                text: format!("no driver implements {}", op.name()),
            });
        }
        // Look at each tree and prepare the request before anything is sent.
        let pres: Vec<Option<(oracle::Pre, Value)>> = ks
            .iter()
            .map(|&k| {
                let r = &runs[k];
                r.ctx.as_ref().map(|c| {
                    let req = subst(&step, &r.vars);
                    (oracle::before(op, c), req)
                })
            })
            .collect();
        let answers = futures::future::join_all(
            runs.iter_mut()
                .enumerate()
                .filter(|(k, _)| ks.contains(k))
                .map(|(_, r)| r.call(&step)),
        )
        .await;
        let mut canons = vec![];
        let mut known: Option<&'static str> = None;
        let mut oracle_msgs: Vec<String> = vec![];
        for (j, (&k, (req, resp))) in ks.iter().zip(&answers).enumerate() {
            let r = &runs[k];
            canons.push(canon(op.name(), resp, &r.root));
            if let Op::Stat { .. } = op {
                mtimes[k].insert(i, resp["ok"]["mtime"].clone());
            }
            if let (Some(ctx), Some((pre, _))) = (&r.ctx, &pres[j]) {
                match oracle::after(op, req, resp, pre, ctx) {
                    Ok(Some(id)) => known = Some(id),
                    Ok(None) => {}
                    Err(why) => oracle_msgs.push(format!("{}: {why}", r.d.name)),
                }
            }
        }
        let agree = canons.iter().all(|c| *c == canons[0]);
        if oracle_msgs.is_empty() && (agree || known.is_some()) {
            if let Some(id) = known {
                *world.known.lock().unwrap().entry(id).or_default() += 1;
                if std::env::var_os("DIFFERENTIAL_VERBOSE").is_some() {
                    eprintln!("KNOWN OPEN BUG {id}: {} {}", op.name(), step);
                }
            }
            mirror(runs, &ks, op, &canons[0]).await;
            history.push(canons.swap_remove(0));
            continue;
        }
        // The drivers that are not with the majority, and those the oracle faulted.
        let mut groups: Vec<(&Value, Vec<&str>)> = vec![];
        for (&k, c) in ks.iter().zip(&canons) {
            let r = &runs[k];
            match groups.iter_mut().find(|(g, _)| *g == c) {
                Some((_, m)) => m.push(&r.d.name),
                None => groups.push((c, vec![&r.d.name])),
            }
        }
        let majority = groups
            .iter()
            .max_by_key(|(_, m)| {
                (
                    m.len(),
                    std::cmp::Reverse(runs.iter().position(|r| r.d.name == m[0])),
                )
            })
            .map(|(_, m)| m.clone())
            .unwrap_or_default();
        let faulted: Vec<&str> = oracle_msgs
            .iter()
            .map(|m| m.split(':').next().unwrap_or(""))
            .collect();
        let differing: Vec<&str> = ks
            .iter()
            .map(|&k| runs[k].d.name.as_str())
            .filter(|n| !majority.contains(n) || faulted.contains(n))
            .collect();
        if let Some(id) = super::known::divergence(&step, &differing, &canons) {
            *world.known.lock().unwrap().entry(id).or_default() += 1;
            if std::env::var_os("DIFFERENTIAL_VERBOSE").is_some() {
                eprintln!("KNOWN DIVERGENCE {id}: {differing:?} {step}");
            }
            mirror(runs, &ks, op, &canons[0]).await;
            history.push(canons.swap_remove(0));
            continue;
        }
        // A divergence. The expectation printed for this step is the answer of a driver the
        // oracle had no complaint about, else the first driver's.
        let bad: Vec<&str> = oracle_msgs
            .iter()
            .map(|m| m.split(':').next().unwrap_or(""))
            .collect();
        let pick = ks
            .iter()
            .position(|&k| !bad.contains(&runs[k].d.name.as_str()))
            .unwrap_or(0);
        let mut text = format!(
            "backend {}: step {i} ({}) {}\n",
            world.backend,
            op.name(),
            serde_json::to_string(&step).unwrap()
        );
        if !agree {
            text += "the drivers answered differently:\n";
            for (&k, c) in ks.iter().zip(&canons) {
                text += &format!("  {:<12} {c}\n", runs[k].d.name);
            }
        }
        for m in &oracle_msgs {
            text += &format!("ground truth disagrees with {m}\n");
        }
        history.push(canons[pick].clone());
        let json = serde_json::to_string_pretty(&scenario(world, ops, &history)).unwrap();
        let file = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/differential")
            .join(format!("{}.json", world.backend));
        let _ = std::fs::create_dir_all(file.parent().unwrap());
        let saved = std::fs::write(&file, &json).is_ok();
        text += &format!(
            "\nas a scenario (expect = {}'s answer; edit it to the oracle's before adding it){}:\n{json}\n",
            runs[ks[pick]].d.name,
            if saved {
                format!(", also in {}", file.display())
            } else {
                String::new()
            }
        );
        return Err(Failure { text });
    }
    Ok(())
}

/// A driver that cannot run `op` must still end the step with the tree the others have. Only
/// `write_if_unchanged` changes a tree; when the others applied it, the same bytes are written
/// with `write`, which is what a guard that passed does. A driver with neither op simply cannot
/// follow, and the read-back shows it.
async fn mirror(runs: &mut [Run<'_>], ks: &[usize], op: &Op, consensus: &Value) {
    let Op::WriteIf { rel, data, .. } = op else {
        return;
    };
    if consensus["ok"].is_null() {
        return; // a conflict or an error: nothing was written
    }
    let step = Op::Write {
        rel: rel.clone(),
        data: data.clone(),
    }
    .step();
    for (k, r) in runs.iter_mut().enumerate() {
        if !ks.contains(&k) && r.d.be.ops.iter().any(|o| o == "write") {
            let _ = r.call(&step).await;
        }
    }
}

/// The sequence up to the failing step, as a file of `sdk/conformance/scenarios` format.
fn scenario(world: &World, ops: &[Op], history: &[Value]) -> Value {
    let referenced: Vec<usize> = ops
        .iter()
        .filter_map(|o| match o {
            Op::WriteIf {
                guard: Guard::Current(j),
                ..
            } => Some(*j),
            _ => None,
        })
        .collect();
    let steps: Vec<Value> = history
        .iter()
        .enumerate()
        .map(|(i, h)| {
            let mut s = ops[i].step();
            s["id"] = json!(format!("s{i}"));
            if referenced.contains(&i) && h["ok"]["exists"] == true {
                s["save"] = json!({format!("m{i}"): "mtime"});
            }
            if let Op::WriteIf {
                guard: Guard::Current(j),
                ..
            } = &ops[i]
                && history[*j]["ok"]["exists"] != true
            {
                s["expected_mtime"] = Value::Null;
            }
            s["expect"] = h.clone();
            s
        })
        .collect();
    json!({
        "suite": "differential-found",
        "version": 1,
        "description": format!("Found by the differential run on {}.", world.backend),
        "scenarios": [{
            "id": "differential-found.new",
            "description": "A sequence two clients, or a client and the filesystem, answered differently.",
            "steps": steps,
        }]
    })
}
