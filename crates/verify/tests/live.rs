//! Live cells: real clients, through a gateway built from this commit, to real providers.
//!
//! Each cell is one test, named `CLAIMS::client::route::probe` (claims joined by `+`), so nextest
//! filters, isolates and reports them, and `verify status` reads their outcomes from the JUnit
//! report. A cell boots its own gateway and nats-server with the real pool keys from `.env`, runs
//! one probe in a pinned client (`verify/clients/`), and then checks two witnesses:
//!
//! 1. The client's own verdict: the probe parsed everything and the task's structure held.
//! 2. The gateway's ledger: for every HTTP call the client made, exactly one `ai.usage` row with
//!    that `x-beyond-request-id`, whose tokens equal what the client was shown (normalized for the
//!    wire), served by the provider the route expects.
//!
//! Cells are listed only with `VERIFY_LIVE=1`, so an ordinary test run never spends money. A cell
//! whose key or client is missing is not listed at all; `verify status` reports that client as
//! missing for the claim (PARTIAL), never as a pass.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read as _, Write as _};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use libtest_mimic::{Arguments, Failed, Trial};
use serde_json::Value;

/// The dev signing key (seed `[7; 32]`, kid 1) and the tenant-1 token minted from it; the same
/// constants `mise run ai:mint-dev-key` prints.
const DEV_PUBKEY_B64: &str = "6kpsY+KcUgq+9VB7Ey7F+ZVHdq6+vnuSQh7qaRRG0iw=";
const DEV_TOKEN: &str = "bai_v1.1.AQAAAAAAAAABAAAAAAAAAA.WrWcPbklu91PS-4WuR6GnBNF3h4nROpH0EQQlfJf06f7_lEnlQOCSBimhH2JMwXFJgw40BniTB7-yIdFnpldDw";

/// Where a cell sends its traffic: a catalog row and the pool keys the gateway holds.
#[derive(Clone, Copy)]
struct Route {
    name: &'static str,
    model: &'static str,
    /// `(provider, env var)` pool keys; every one must be set for the cell to be listed.
    pools: &'static [(&'static str, &'static str)],
    /// Providers pointed at a dead port, to force failover.
    dead: &'static [&'static str],
    /// The provider every billing row must name.
    serves: &'static str,
}

const CLAUDE: Route = Route {
    name: "claude",
    model: "claude-haiku-4-5",
    pools: &[("anthropic", "ANTHROPIC_API_KEY")],
    dead: &[],
    serves: "anthropic",
};
const GPT: Route = Route {
    name: "gpt",
    model: "gpt-5-mini",
    pools: &[("openai", "OPENAI_API_KEY")],
    dead: &[],
    serves: "openai",
};
const OPENROUTER: Route = Route {
    name: "openrouter",
    model: "claude-sonnet-4",
    pools: &[("openrouter", "OPENROUTER_API_KEY")],
    dead: &[],
    serves: "openrouter",
};
/// Anthropic unreachable: the Claude row must fail over to OpenRouter before the client notices.
const FAILOVER: Route = Route {
    name: "failover",
    model: "claude-haiku-4-5",
    pools: &[
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
    ],
    dead: &["anthropic"],
    serves: "openrouter",
};
const EMBED: Route = Route {
    name: "embed",
    model: "text-embedding-3-small",
    pools: &[("openai", "OPENAI_API_KEY")],
    dead: &[],
    serves: "openai",
};

#[derive(Clone, Copy)]
enum Runtime {
    Python,
}

struct Cell {
    claims: &'static str,
    client: &'static str,
    runtime: Runtime,
    probe: &'static str,
    routes: &'static [Route],
    /// Claims a failover-route cell additionally proves.
    failover_claims: &'static str,
}

const GEN: &[Route] = &[CLAUDE, GPT, OPENROUTER, FAILOVER];

fn cells() -> Vec<Cell> {
    use Runtime::Python;
    vec![
        Cell {
            claims: "E1+B1+S1",
            client: "openai-py",
            runtime: Python,
            probe: "chat_basic",
            routes: GEN,
            failover_claims: "R1",
        },
        Cell {
            claims: "E2+B1+S1",
            client: "anthropic-py",
            runtime: Python,
            probe: "messages_basic",
            routes: GEN,
            failover_claims: "R1",
        },
        Cell {
            claims: "E3+TRN-1+CAT-9+B1",
            client: "openai-py",
            runtime: Python,
            probe: "responses_basic",
            routes: &[CLAUDE, GPT, OPENROUTER],
            failover_claims: "",
        },
        Cell {
            claims: "E4",
            client: "openai-py",
            runtime: Python,
            probe: "models_list",
            routes: &[GPT],
            failover_claims: "",
        },
        Cell {
            claims: "T1+B1",
            client: "openai-py",
            runtime: Python,
            probe: "tools_chat",
            routes: GEN,
            failover_claims: "R1",
        },
        Cell {
            claims: "T1+B1",
            client: "anthropic-py",
            runtime: Python,
            probe: "tools_messages",
            routes: GEN,
            failover_claims: "R1",
        },
        Cell {
            claims: "M1+B1",
            client: "openai-py",
            runtime: Python,
            probe: "embeddings",
            routes: &[EMBED],
            failover_claims: "",
        },
    ]
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// `.env` at the repo root, then the process environment (which wins).
fn env_keys() -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    if let Ok(s) = std::fs::read_to_string(repo_root().join(".env")) {
        for line in s.lines() {
            let line = line.trim();
            if line.starts_with('#') {
                continue;
            }
            if let Some((k, v)) = line.split_once('=') {
                out.insert(k.trim().to_owned(), v.trim().trim_matches('"').to_owned());
            }
        }
    }
    for (k, v) in std::env::vars() {
        out.insert(k, v);
    }
    out
}

fn interpreter(rt: Runtime) -> PathBuf {
    match rt {
        Runtime::Python => repo_root().join("verify/clients/py/.venv/bin/python"),
    }
}

fn probe_script(rt: Runtime) -> PathBuf {
    match rt {
        Runtime::Python => repo_root().join("verify/clients/py/probe.py"),
    }
}

fn gateway_bin() -> PathBuf {
    std::env::var_os("VERIFY_GATEWAY_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("target/debug/beyond-ai"))
}

fn main() {
    let args = Arguments::from_args();
    let mut trials = Vec::new();
    if std::env::var("VERIFY_LIVE").as_deref() == Ok("1") {
        let keys = env_keys();
        for cell in cells() {
            for route in cell.routes {
                let have_keys = route
                    .pools
                    .iter()
                    .all(|(_, var)| keys.get(*var).is_some_and(|v| !v.is_empty()));
                if !have_keys || !interpreter(cell.runtime).exists() || !gateway_bin().exists() {
                    continue;
                }
                let claims = if route.name == "failover" && !cell.failover_claims.is_empty() {
                    format!("{}+{}", cell.claims, cell.failover_claims)
                } else {
                    cell.claims.to_owned()
                };
                let name = format!("{claims}::{}::{}::{}", cell.client, route.name, cell.probe);
                let (rt, probe, route, keys) = (cell.runtime, cell.probe, *route, keys.clone());
                trials.push(Trial::test(name, move || run_cell(rt, probe, route, &keys)));
            }
        }
    }
    libtest_mimic::run(&args, trials).exit();
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// Kills its process on drop, so a failing cell never leaks a gateway or nats-server.
struct Guard(Child);

impl Drop for Guard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn wait_ready(metrics_port: u16, gw: &mut Child, log: &Path) -> Result<(), Failed> {
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        if let Ok(Some(status)) = gw.try_wait() {
            return Err(format!("gateway exited {status}: {}", tail(log)).into());
        }
        if let Ok(mut s) = std::net::TcpStream::connect(("127.0.0.1", metrics_port)) {
            // `Connection: close` and a read timeout: the admin server keeps an idle connection
            // open for a minute, and reading to EOF without them waits that long.
            let _ = s.set_read_timeout(Some(Duration::from_secs(2)));
            let _ = s.write_all(b"GET /metrics HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n");
            let mut body = String::new();
            let _ = s.read_to_string(&mut body);
            if body.lines().any(|l| l.starts_with("ai_allowance_ready 1")) {
                return Ok(());
            }
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    Err(format!("gateway never became ready: {}", tail(log)).into())
}

fn tail(path: &Path) -> String {
    let s = std::fs::read_to_string(path).unwrap_or_default();
    let start = s.len().saturating_sub(2000);
    s[start..].to_owned()
}

fn run_cell(
    rt: Runtime,
    probe: &str,
    route: Route,
    keys: &BTreeMap<String, String>,
) -> Result<(), Failed> {
    let dir = std::env::temp_dir().join(format!(
        "verify-live-{}-{}",
        std::process::id(),
        free_port()
    ));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    let nats_port = free_port();
    let _nats = Guard(
        Command::new("nats-server")
            .args([
                "-js",
                "-a",
                "127.0.0.1",
                "-p",
                &nats_port.to_string(),
                "-sd",
            ])
            .arg(dir.join("nats"))
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("nats-server: {e}"))?,
    );

    let (port, metrics_port) = (free_port(), free_port());
    let mut cfg = format!(
        "listen = \"127.0.0.1:{port}\"\nmetrics_listen = \"127.0.0.1:{metrics_port}\"\n\
         nats_url = \"nats://127.0.0.1:{nats_port}\"\nconfig_bucket = \"ai-gateway\"\nupstream_tls = true\n\n[pool_keys]\n"
    );
    for (provider, var) in route.pools {
        cfg.push_str(&format!("{provider} = [{:?}]\n", keys[*var]));
    }
    if !route.dead.is_empty() {
        cfg.push_str("\n[provider_authorities]\n");
        for p in route.dead {
            cfg.push_str(&format!("{p} = \"127.0.0.1:9\"\n"));
        }
    }
    cfg.push_str(&format!("\n[signing_keys]\n1 = \"{DEV_PUBKEY_B64}\"\n"));
    let cfg_path = dir.join("gateway.toml");
    std::fs::write(&cfg_path, cfg).map_err(|e| e.to_string())?;

    let log_path = dir.join("gateway.log");
    let log = std::fs::File::create(&log_path).map_err(|e| e.to_string())?;
    let mut gw = Guard(
        Command::new(gateway_bin())
            .args(["run", "-c"])
            .arg(&cfg_path)
            .env("AI_LOG", "warn,ai.usage=info")
            .stdout(log.try_clone().map_err(|e| e.to_string())?)
            .stderr(log)
            .spawn()
            .map_err(|e| format!("gateway: {e}"))?,
    );
    // nats needs a moment before the gateway's first connect; readiness waits for the scan.
    wait_ready(metrics_port, &mut gw.0, &log_path)?;

    let out = Command::new(interpreter(rt))
        .arg(probe_script(rt))
        .arg(probe)
        .env("VERIFY_BASE", format!("http://127.0.0.1:{port}"))
        .env("VERIFY_KEY", DEV_TOKEN)
        .env("VERIFY_MODEL", route.model)
        .stderr(Stdio::inherit())
        .output()
        .map_err(|e| format!("probe: {e}"))?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let verdict: Value = stdout
        .lines()
        .rev()
        .find_map(|l| l.strip_prefix("VERIFY "))
        .and_then(|j| serde_json::from_str(j).ok())
        .ok_or_else(|| format!("probe printed no VERIFY line:\n{stdout}"))?;

    // Witness 1: the client's own verdict.
    if verdict["ok"] != true {
        return Err(format!(
            "client verdict failed: {}\n--- gateway log ---\n{}",
            verdict["detail"],
            tail(&log_path)
        )
        .into());
    }

    // Witness 2: the ledger. Rows can land a beat after the response; give them a moment.
    let calls = verdict["calls"].as_array().cloned().unwrap_or_default();
    let deadline = Instant::now() + Duration::from_secs(5);
    let rows = loop {
        let rows = usage_rows(&log_path);
        if rows.len() >= calls.len() || Instant::now() >= deadline {
            break rows;
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    let mut problems = Vec::new();
    if rows.len() != calls.len() {
        problems.push(format!(
            "{} client calls but {} ai.usage rows",
            calls.len(),
            rows.len()
        ));
    }
    for call in &calls {
        let Some(id) = call["request_id"].as_str() else {
            problems.push(format!("a call carried no x-beyond-request-id: {call}"));
            continue;
        };
        let matching: Vec<&Value> = rows.iter().filter(|r| r["request_id"] == id).collect();
        let [row] = matching.as_slice() else {
            problems.push(format!(
                "{id}: {} ai.usage rows, want exactly 1",
                matching.len()
            ));
            continue;
        };
        if row["provider"] != route.serves {
            problems.push(format!(
                "{id}: served by {}, route expects {}",
                row["provider"], route.serves
            ));
        }
        if let Some(u) = call["usage"].as_object() {
            let n = |v: &Value| v.as_u64().unwrap_or(0);
            let anthropic_wire = matches!(row["provider"].as_str(), Some("anthropic" | "bedrock"));
            let row_input = n(&row["input_tokens"])
                + if anthropic_wire {
                    n(&row["cache_read_tokens"]) + n(&row["cache_write_tokens"])
                } else {
                    0
                };
            if row_input != n(&u["input_total"]) {
                problems.push(format!(
                    "{id}: client saw {} input tokens, row bills {row_input} ({row})",
                    u["input_total"]
                ));
            }
            if n(&row["output_tokens"]) != n(&u["output"]) {
                problems.push(format!(
                    "{id}: client saw {} output tokens, row bills {} ({row})",
                    u["output"], row["output_tokens"]
                ));
            }
            if row["usage_estimated"] == true {
                problems.push(format!(
                    "{id}: row is an estimate on a completed call ({row})"
                ));
            }
        } else {
            problems.push(format!("{id}: the client was shown no usage"));
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    if problems.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "ledger disagrees with the client:\n  {}\n--- detail --- {}",
            problems.join("\n  "),
            verdict["detail"]
        )
        .into())
    }
}

fn usage_rows(log: &Path) -> Vec<Value> {
    let Ok(f) = std::fs::File::open(log) else {
        return Vec::new();
    };
    BufReader::new(f)
        .lines()
        .map_while(Result::ok)
        .filter(|l| l.contains("\"ai.usage\""))
        .filter_map(|l| serde_json::from_str::<Value>(&l).ok())
        .map(|v| v.get("fields").cloned().unwrap_or(v))
        .collect()
}
