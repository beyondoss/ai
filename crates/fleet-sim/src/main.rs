//! The fleet simulator.
//!
//! Runs real `serve --service` replicas over a shared filesystem, drives them with a real edge
//! double and a real workload, breaks things on purpose, and checks the claims the design makes.
//!
//! ```text
//! fleet-sim matrix [--substrate local-dir|nfs]   every scenario, once, deterministic — the gate
//! fleet-sim soak [--duration S] [--seed N]      randomized chaos, then the checker — the burn-in
//! fleet-sim list                                 the scenarios and which substrate each needs
//! ```
//!
//! It is not a CI shard. The substrate that makes the interesting claims checkable needs root, an
//! NFS export and mount namespaces; the local-directory substrate runs anywhere but is blind to
//! exactly the cross-client behaviour the storage design is about. Both facts are load-bearing, and
//! the matrix output says which scenarios actually ran.

// Scaffolding the remaining scenarios need — chaos injectors, the history reader, the rest of the
// checker. Kept compiled and reviewed rather than added later in a rush, but not yet called.
#![allow(dead_code)]

mod check;
mod edge;
mod history;
mod replica;
mod scenarios;
mod soak;
mod substrate;
mod workload;

use substrate::Kind;

#[tokio::main]
async fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let command = args.first().map(String::as_str).unwrap_or("matrix");

    let kind = match flag(&args, "--substrate") {
        Some(s) => match Kind::parse(&s) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("fleet-sim: {e}");
                return std::process::ExitCode::from(2);
            }
        },
        None => Kind::LocalDir,
    };

    match command {
        "list" => {
            println!("scenario                    claims     substrate");
            for s in scenarios::ALL {
                println!(
                    "{:<27} {:<10} {}",
                    s.name,
                    s.claims,
                    if s.needs_shared_fs { "nfs only" } else { "any" }
                );
            }
            std::process::ExitCode::SUCCESS
        }
        "matrix" => run_matrix(kind).await,
        "soak" => {
            let secs = flag(&args, "--duration")
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(120);
            let seed = flag(&args, "--seed")
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(1);
            let sessions = flag(&args, "--sessions")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(4);
            if soak::run(kind, std::time::Duration::from_secs(secs), seed, sessions).await {
                std::process::ExitCode::SUCCESS
            } else {
                std::process::ExitCode::FAILURE
            }
        }
        other => {
            eprintln!("fleet-sim: unknown command {other:?} (expected `matrix` or `list`)");
            std::process::ExitCode::from(2)
        }
    }
}

fn flag(args: &[String], name: &str) -> Option<String> {
    let i = args.iter().position(|a| a == name)?;
    args.get(i + 1).cloned()
}

async fn run_matrix(kind: Kind) -> std::process::ExitCode {
    let dir = match tempfile::tempdir() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("fleet-sim: history directory: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    println!("fleet-sim matrix · substrate: {kind:?}");
    if !kind.is_shared_filesystem() {
        println!(
            "note: a local directory has one client, so its `O_APPEND` is atomic and a lock dies \
             with its process.\n      Scenarios that depend on neither being true will skip rather \
             than pass vacuously."
        );
    }

    let mut passed = 0;
    let mut failed = 0;
    let mut skipped = 0;

    for s in scenarios::ALL {
        let history = dir.path().join(format!("{}.jsonl", s.name));
        let outcome = match s.name {
            "owner-refuses-non-owner" => scenarios::owner_refuses_non_owner(kind, &history).await,
            "takeover-after-hard-kill" => scenarios::takeover_after_hard_kill(kind, &history).await,
            "drain-keeps-serving-what-it-owns" => {
                scenarios::drain_keeps_serving_what_it_owns(kind, &history).await
            }
            "live-session-cap-refuses" => scenarios::live_session_cap_refuses(kind, &history).await,
            "metrics-name-no-tenant" => scenarios::metrics_name_no_tenant(kind, &history).await,
            "unmounted-shard-is-misdirected" => {
                scenarios::unmounted_shard_is_misdirected(kind, &history).await
            }
            other => scenarios::Outcome::Skipped(format!("no runner for {other:?}")),
        };
        match outcome {
            scenarios::Outcome::Passed => passed += 1,
            scenarios::Outcome::Failed => failed += 1,
            scenarios::Outcome::Skipped(why) => {
                skipped += 1;
                println!("\nSKIP {}\n  {why}", s.name);
            }
        }
    }

    println!("\n{passed} passed · {failed} failed · {skipped} skipped");
    if failed > 0 {
        std::process::ExitCode::FAILURE
    } else {
        std::process::ExitCode::SUCCESS
    }
}
