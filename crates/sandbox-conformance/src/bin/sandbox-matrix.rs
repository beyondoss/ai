//! `sandbox-matrix`: every client's driver against every backend, over the shared scenarios.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use clap::Parser;
use sandbox_conformance::backends::{self, Binaries};
use sandbox_conformance::matrix::{self, Plan};
use sandbox_conformance::report;

#[derive(Parser)]
#[command(about = "Run every client's conformance driver against every backend")]
struct Args {
    /// NAME=CMD: a shell command line that starts a driver. Repeatable.
    #[arg(long = "driver")]
    drivers: Vec<String>,
    /// A registry of drivers (`sdk/conformance/drivers.toml`): each entry's `build` runs, then its
    /// `run` command is a driver. Combines with `--driver`.
    #[arg(long)]
    registry: Option<PathBuf>,
    /// With `--registry`: do not run the builds.
    #[arg(long)]
    no_build: bool,
    /// A backend, a group (`local`, `fault`) or a `*` glob. Repeatable.
    #[arg(long = "backend", required = true)]
    backends: Vec<String>,
    #[arg(long, default_value = "sdk/conformance/scenarios")]
    scenarios: PathBuf,
    /// Write the JSON report here.
    #[arg(long)]
    report: Option<PathBuf>,
    /// Only scenarios whose id matches this glob (`*`, `?`).
    #[arg(long)]
    filter: Option<String>,
    /// execd for this host (default: $EXECD_BIN, or `execd` next to this binary).
    #[arg(long)]
    execd: Option<PathBuf>,
    /// A static execd for the containers (default: $EXECD_STATIC_BIN, or the musl build).
    #[arg(long)]
    execd_static: Option<PathBuf>,
    /// Also list every known-defect, unsupported and skipped check with its reason.
    #[arg(long)]
    verbose: bool,
    /// Backends running at once.
    #[arg(long, default_value_t = 4)]
    parallel: usize,
}

#[tokio::main]
async fn main() -> ExitCode {
    let args = Args::parse();
    let plan = (|| -> Result<Plan, String> {
        let mut drivers: Vec<_> = args
            .drivers
            .iter()
            .map(|d| matrix::parse_driver(d))
            .collect::<Result<_, _>>()?;
        if let Some(reg) = &args.registry {
            drivers.extend(matrix::registry(reg, !args.no_build)?);
        }
        if drivers.is_empty() {
            return Err("no drivers: give --driver NAME=CMD or --registry FILE".into());
        }
        Ok(Plan {
            drivers,
            backends: backends::expand(&args.backends)?,
            scenarios: Arc::new(matrix::load(&args.scenarios)?),
            filter: args.filter.clone(),
            bins: Binaries {
                execd: args.execd.clone(),
                execd_static: args.execd_static.clone(),
            },
            parallel: args.parallel,
        })
    })();
    let plan = match plan {
        Ok(p) => p,
        Err(e) => {
            eprintln!("sandbox-matrix: {e}");
            return ExitCode::from(2);
        }
    };
    let cells = matrix::run(plan).await;
    print!("{}", report::table(&cells, args.verbose));
    if let Some(path) = &args.report {
        let json = serde_json::to_string_pretty(&matrix::report_json(&cells)).expect("json");
        if let Err(e) = std::fs::write(path, json + "\n") {
            eprintln!("sandbox-matrix: {}: {e}", path.display());
            return ExitCode::from(2);
        }
    }
    if cells.iter().all(|c| c.passes()) {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
