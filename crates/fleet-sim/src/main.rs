//! The fleet simulator.
//!
//! Runs real `serve --service` replicas over a shared filesystem, drives them with a real edge
//! double and a real workload, breaks things on purpose, and checks the claims the design makes.
//!
//! ```text
//! fleet-sim matrix [--substrate local-dir|nfs]   every scenario, once, deterministic — the gate
//! fleet-sim soak [--duration S] [--seed N]      randomized chaos, then the checker — the burn-in
//!              [--sessions N] [--tenants N] [--shards N]
//! fleet-sim list                                 the scenarios and which substrate each needs
//! fleet-sim keys --dir D                        write the seal key, print the grant-key flag
//!
//! `--substrate attached` runs the same scenarios against a fleet **somebody else is running**, on
//! storage somebody else provisioned: `--shard name=/path` (repeatable), `--replica host:port`
//! (repeatable) and `--fault-cmd CMD`, which the simulator invokes as `CMD <action> <replica>` for
//! `kill|term|partition|heal|restart`. Nothing in here knows what provides any of it, which is what
//! lets the identical binary grade a local NFS fleet and a real ECS+EFS one.
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
mod faults;
mod history;
mod replica;
mod scenarios;
mod soak;
mod substrate;
mod workload;

use substrate::Kind;

#[tokio::main]
async fn main() -> std::process::ExitCode {
    install_teardown_on_signal();
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
        "keys" => {
            // So the infrastructure can start replicas with the matching keys *before* the simulator
            // runs. Deterministic by design — the seeds are fixed so a failure reproduces — which
            // also means they are not secret and an attached fleet must be private.
            let dir = flag(&args, "--dir").unwrap_or_else(|| ".".to_owned());
            let minter = beyond_ai_test_support::grant::Minter::new(std::path::Path::new(&dir));
            println!("AI_AGENT_GRANT_KEY={}", minter.grant_key_flag());
            println!("AI_AGENT_SEAL_KEY={}", minter.seal_key().display());
            std::process::ExitCode::SUCCESS
        }
        "matrix" => run_matrix(kind, &args).await,
        "soak" => {
            let secs = flag(&args, "--duration")
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(120);
            let seed = flag(&args, "--seed")
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(1);
            let sessions = flag(&args, "--sessions")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(6);
            // More than one of each by default. A single-tenant, single-shard soak exercises the
            // storage layout and proves nothing about the two properties the layout is *for*.
            let tenants = flag(&args, "--tenants")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(3);
            let shards = flag(&args, "--shards")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(2);
            // How often a client drops its socket and comes back, in turns. `0` holds it for the
            // whole run; `1` reconnects every turn, which is what this did before it was a knob —
            // and which charges a TCP handshake, a WebSocket upgrade, a grant verification and a
            // session attach to every single turn.
            let reconnect_every = flag(&args, "--reconnect-every")
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(20);
            if soak::run(soak::Soak {
                kind,
                duration: std::time::Duration::from_secs(secs),
                seed,
                sessions,
                tenants,
                shards,
                reconnect_every,
                chaos: !args.iter().any(|a| a == "--no-chaos"),
            })
            .await
            {
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

/// Bring the substrate down on `SIGINT`/`SIGTERM`.
///
/// `Drop` does not run when a process is signalled, so an interrupted or `timeout`-ed run left its
/// NFS export and mounts on the host — and because an export is a machine-wide resource, the next
/// run then collided with the last one's corpse. A harness that damages the machine it borrows when
/// you press Ctrl-C is a harness people stop reaching for.
fn install_teardown_on_signal() {
    for kind in [
        tokio::signal::unix::SignalKind::interrupt(),
        tokio::signal::unix::SignalKind::terminate(),
    ] {
        let Ok(mut sig) = tokio::signal::unix::signal(kind) else {
            continue;
        };
        tokio::spawn(async move {
            sig.recv().await;
            eprintln!("\nfleet-sim: interrupted — tearing the substrate down");
            // On the blocking pool: teardown shells out, and the runtime is about to stop.
            let _ = tokio::task::spawn_blocking(|| {
                // Replicas first: one still holding a session lock on a mount we are about to
                // withdraw is the orphan that confuses the next run.
                replica::kill_all();
                substrate::teardown_all();
            })
            .await;
            std::process::exit(130);
        });
    }
}

/// Every occurrence of a repeatable flag, in order.
fn flags(args: &[String], name: &str) -> Vec<String> {
    args.iter()
        .enumerate()
        .filter(|(_, a)| a.as_str() == name)
        .filter_map(|(i, _)| args.get(i + 1).cloned())
        .collect()
}

/// `--shard name=/path`, parsed the same way the agent parses its own.
fn shard_args(args: &[String]) -> Result<Vec<(String, std::path::PathBuf)>, String> {
    flags(args, "--shard")
        .iter()
        .map(|s| {
            s.split_once('=')
                .map(|(n, p)| (n.to_owned(), std::path::PathBuf::from(p)))
                .ok_or_else(|| format!("--shard {s:?} is not name=/path"))
        })
        .collect()
}

fn flag(args: &[String], name: &str) -> Option<String> {
    let i = args.iter().position(|a| a == name)?;
    args.get(i + 1).cloned()
}

fn attached_from(args: &[String]) -> Result<Option<scenarios::Attachment>, String> {
    if flag(args, "--substrate").as_deref() != Some("attached") {
        return Ok(None);
    }
    let replicas = flags(args, "--replica")
        .iter()
        .map(|s| crate::edge::Addr::parse(s))
        .collect::<Result<Vec<_>, _>>()?;
    let fault_cmd = flag(args, "--fault-cmd").ok_or_else(|| {
        "`--substrate attached` needs --fault-cmd CMD: the simulator does not own these replicas, \
         so it cannot kill or partition them itself"
            .to_owned()
    })?;
    Ok(Some(scenarios::Attachment {
        shards: shard_args(args)?,
        replicas,
        fault_cmd,
    }))
}

async fn run_matrix(kind: Kind, args: &[String]) -> std::process::ExitCode {
    match attached_from(args) {
        Ok(Some(spec)) => {
            println!(
                "attached: {} shard(s), {} replica(s), faults via {:?}",
                spec.shards.len(),
                spec.replicas.len(),
                spec.fault_cmd
            );
            scenarios::attach_to(spec);
        }
        Ok(None) => {}
        Err(e) => {
            eprintln!("fleet-sim: {e}");
            return std::process::ExitCode::from(2);
        }
    }
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

    // `--only <name>` runs one scenario. A matrix that can only be run whole is a matrix nobody
    // iterates on: the interesting scenarios are the slow ones, and re-running six to debug one is
    // how a harness stops being used.
    let only = flag(&std::env::args().collect::<Vec<_>>(), "--only");
    for s in scenarios::ALL {
        if let Some(want) = &only
            && s.name != want
        {
            continue;
        }
        let history = dir.path().join(format!("{}.jsonl", s.name));
        let outcome = match s.name {
            "owner-refuses-non-owner" => scenarios::owner_refuses_non_owner(kind, &history).await,
            "takeover-after-hard-kill" => scenarios::takeover_after_hard_kill(kind, &history).await,
            "drain-keeps-serving-what-it-owns" => {
                scenarios::drain_keeps_serving_what_it_owns(kind, &history).await
            }
            "live-session-cap-refuses" => scenarios::live_session_cap_refuses(kind, &history).await,
            "metrics-name-no-tenant" => scenarios::metrics_name_no_tenant(kind, &history).await,
            "one-tenant-cannot-read-another" => {
                scenarios::one_tenant_cannot_read_another(kind, &history).await
            }
            "fenced-owner-stops-and-says-so" => {
                scenarios::fenced_owner_stops_and_says_so(kind, &history).await
            }
            "hung-mount-is-reported-not-leaked" => {
                scenarios::hung_mount_is_reported_not_leaked(kind, &history).await
            }
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
