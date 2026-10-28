//! Drivers x backends x scenarios: the matrix itself.
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use serde_json::Value;
use tokio::sync::Semaphore;

use crate::backends::{self, Binaries};
use crate::link::ProcessConnector;
use crate::relay;
use crate::report::Cell;
use crate::run::{Backend, Check, Status, glob_match, run_scenario};

#[derive(Clone, Debug)]
pub struct Driver {
    pub name: String,
    pub cmd: String,
}

pub struct Plan {
    pub drivers: Vec<Driver>,
    /// Expanded backend names.
    pub backends: Vec<String>,
    pub scenarios: Arc<Vec<(String, Value)>>,
    pub filter: Option<String>,
    pub bins: Binaries,
    /// Backends started at once.
    pub parallel: usize,
}

/// Run every driver against every backend. Backends run in parallel (each started once, its drivers
/// one after another); the cells come back in plan order.
pub async fn run(plan: Plan) -> Vec<Cell> {
    let plan = Arc::new(plan);
    let gate = Arc::new(Semaphore::new(plan.parallel.max(1)));
    let mut tasks = Vec::new();
    for name in plan.backends.clone() {
        let (plan, gate) = (plan.clone(), gate.clone());
        tasks.push(tokio::spawn(async move {
            let _permit = gate.acquire().await.expect("semaphore open");
            run_backend(&plan, &name).await
        }));
    }
    let mut cells = Vec::new();
    for t in tasks {
        cells.extend(t.await.expect("backend task"));
    }
    // Plan order: driver-major.
    let order = |c: &Cell| {
        (
            plan.drivers
                .iter()
                .position(|d| d.name == c.driver)
                .unwrap_or(usize::MAX),
            plan.backends.iter().position(|b| *b == c.backend),
        )
    };
    cells.sort_by_key(order);
    cells
}

async fn run_backend(plan: &Plan, name: &str) -> Vec<Cell> {
    let started = match backends::start(name, &plan.bins).await {
        Ok(s) => s,
        Err(why) => {
            return plan
                .drivers
                .iter()
                .map(|d| Cell::failed_to_run(&d.name, name, "backend/start", why.clone()))
                .collect();
        }
    };
    let mut cells = Vec::new();
    let open = |d: &Driver| {
        let connector = ProcessConnector {
            cmd: d.cmd.clone(),
            url: started.url.clone(),
            bearer: started.bearer.clone(),
            caps: started.caps.clone(),
        };
        let (cell_name, auth, work, vars) = (
            format!("{}__{}", d.name, name),
            started.bearer.is_some(),
            started.work.clone(),
            started.vars.clone(),
        );
        async move {
            let mut be = Backend::open(&cell_name, Box::new(connector), auth, &work).await;
            be.vars = vars;
            be
        }
    };
    let mut opened = Vec::new();
    for d in &plan.drivers {
        let be = open(d).await;
        if be.ops.is_empty() {
            let why = be
                .link
                .call(&serde_json::json!({"id": 0, "op": "hello"}))
                .await;
            cells.push(Cell::failed_to_run(
                &d.name,
                name,
                "driver/hello",
                format!("the driver did not answer `hello` with its ops: {why}"),
            ));
            continue;
        }
        cells.push(run_cell(plan, &d.name, name, &be, started.suites).await);
        opened.push((d, be));
    }
    // Relay: every ordered pair on this one sandbox. Pairs of a driver with itself use a second
    // process of it, and run only the scenarios that ask.
    if started.suites.is_none() {
        for (da, a) in &opened {
            for (db, b) in &opened {
                let same = da.name == db.name;
                let second;
                let b = if same {
                    second = open(db).await;
                    &second
                } else {
                    b
                };
                let filter = plan.filter.clone();
                let keep = move |id: &str| filter.as_deref().is_none_or(|f| glob_match(f, id));
                let checks =
                    relay::run_pair(&plan.scenarios, a, b, same, started.suites, &keep).await;
                if relay::ran(&checks) {
                    cells.push(Cell::from_checks(
                        &format!("relay:{}>{}", da.name, db.name),
                        name,
                        &checks,
                    ));
                }
            }
        }
    }
    cells
}

async fn run_cell(
    plan: &Plan,
    driver: &str,
    backend: &str,
    be: &Backend,
    suites: Option<&[&str]>,
) -> Cell {
    let mut checks: Vec<Check> = Vec::new();
    for (_, file) in plan.scenarios.iter() {
        let suite = file["suite"].as_str().unwrap_or("");
        if suites.is_some_and(|s| !s.contains(&suite)) {
            continue;
        }
        for sc in file["scenarios"].as_array().into_iter().flatten() {
            let id = sc["id"].as_str().unwrap_or("");
            if relay::is_relay(sc) {
                continue;
            }
            if plan.filter.as_deref().is_some_and(|f| !glob_match(f, id)) {
                continue;
            }
            checks.extend(run_scenario(sc, be).await);
        }
    }
    // A backend is only worth its cell if something ran.
    if !checks.iter().any(|c| c.status == Status::Pass)
        && checks.iter().all(|c| c.status != Status::Fail)
    {
        checks.push(Check {
            id: "matrix/empty".into(),
            status: Status::Fail,
            detail: "no check passed (is the filter right?)".into(),
        });
    }
    Cell::from_checks(driver, backend, &checks)
}

/// `NAME=CMD`.
pub fn parse_driver(spec: &str) -> Result<Driver, String> {
    match spec.split_once('=') {
        Some((n, c)) if !n.is_empty() && !c.trim().is_empty() => Ok(Driver {
            name: n.to_string(),
            cmd: c.to_string(),
        }),
        _ => Err(format!("--driver wants NAME=CMD, got {spec:?}")),
    }
}

/// The host's cargo build lock: `$BEYOND_BUILD_LOCK`, else `$HOME/.cache/beyond-locks/rust-build.lock`.
/// `Some` only when its directory exists and `flock` is on PATH (CI has neither, so builds run bare).
fn build_lock() -> Option<std::path::PathBuf> {
    let lock = match std::env::var_os("BEYOND_BUILD_LOCK") {
        Some(p) => std::path::PathBuf::from(p),
        None => std::path::PathBuf::from(std::env::var_os("HOME")?)
            .join(".cache/beyond-locks/rust-build.lock"),
    };
    let dir_ok = lock.parent().is_some_and(|d| d.is_dir());
    let flock_ok = std::env::var_os("PATH")
        .is_some_and(|p| std::env::split_paths(&p).any(|d| d.join("flock").is_file()));
    (dir_ok && flock_ok && !held_by_ancestor(&lock)).then_some(lock)
}

/// Whether an ancestor already runs `flock ... LOCK` (`flock -o LOCK mise run test:clients`, say).
/// Taking the lock again would wait on the process that holds it, forever.
fn held_by_ancestor(lock: &Path) -> bool {
    use std::os::unix::ffi::OsStrExt;
    let mut pid = std::process::id();
    while pid > 1 {
        let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) else {
            return false;
        };
        // Field 4 of /proc/PID/stat is the parent pid; it follows the parenthesised command name.
        let ppid = stat
            .rsplit(')')
            .next()
            .and_then(|r| r.split_whitespace().nth(1))
            .and_then(|p| p.parse::<u32>().ok());
        let Some(ppid) = ppid else { return false };
        if let Ok(cmd) = std::fs::read(format!("/proc/{ppid}/cmdline")) {
            let mut argv = cmd.split(|b| *b == 0);
            let is_flock = argv.next().is_some_and(|a| a.ends_with(b"flock"));
            if is_flock && argv.any(|a| a == lock.as_os_str().as_bytes()) {
                return true;
            }
        }
        pid = ppid;
    }
    false
}

/// The drivers of a registry file. Each `[[driver]]` has a `name`, a `build` command (optional) and
/// a `run` command; commands run from the current directory (the repository root). A build runs
/// under the host's build lock when there is one (see [`build_lock`]).
pub fn registry(path: &Path, build: bool) -> Result<Vec<Driver>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let doc: toml::Table = text
        .parse()
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = Vec::new();
    for d in doc
        .get("driver")
        .and_then(|d| d.as_array())
        .into_iter()
        .flatten()
    {
        let field = |k: &str| d.get(k).and_then(|v| v.as_str());
        let (Some(name), Some(run)) = (field("name"), field("run")) else {
            return Err(format!(
                "{}: a [[driver]] needs `name` and `run`",
                path.display()
            ));
        };
        if let (true, Some(cmd)) = (build, field("build")) {
            eprintln!("sandbox-matrix: building {name}: {cmd}");
            let mut c = match build_lock() {
                Some(lock) => {
                    let mut c = std::process::Command::new("flock");
                    c.arg("-o").arg(lock).arg("sh");
                    c
                }
                None => std::process::Command::new("sh"),
            };
            let ok = c
                .arg("-c")
                .arg(cmd)
                .status()
                .map_err(|e| format!("{name}: {e}"))?
                .success();
            if !ok {
                return Err(format!("{name}: the build failed: {cmd}"));
            }
        }
        out.push(Driver {
            name: name.into(),
            cmd: run.into(),
        });
    }
    Ok(out)
}

pub fn load(dir: &Path) -> Result<Vec<(String, Value)>, String> {
    crate::run::load_scenarios(dir)
}

/// The commit the report is for.
pub fn commit() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

pub fn report_json(cells: &[Cell]) -> Value {
    let (commit, updated) = (commit(), crate::report::now_rfc3339());
    let map: BTreeMap<String, Value> = cells
        .iter()
        .map(|c| (c.name(), c.json(&commit, &updated)))
        .collect();
    serde_json::to_value(map).expect("map of values")
}
