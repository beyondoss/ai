//! Drivers x backends x scenarios: the matrix itself.
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use serde_json::Value;
use tokio::sync::Semaphore;

use crate::backends::{self, Binaries};
use crate::link::ProcessConnector;
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
            plan.drivers.iter().position(|d| d.name == c.driver),
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
    for d in &plan.drivers {
        let cell_name = format!("{}__{}", d.name, name);
        let connector = ProcessConnector {
            cmd: d.cmd.clone(),
            url: started.url.clone(),
            bearer: started.bearer.clone(),
            jobs_root: started.jobs_root.clone(),
            caps: started.caps.clone(),
        };
        let mut be = Backend::open(
            &cell_name,
            Box::new(connector),
            started.bearer.is_some(),
            &started.work,
        )
        .await;
        be.vars = started.vars.clone();
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
