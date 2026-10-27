//! The load driver: many tenants' sessions against one service-mode replica, in waves.
//!
//! It builds the bundles (a tenant's agent folder, a few with 5 MB of skills), uploads them to the
//! stand-in bundle host, then opens sessions the way a fleet's edge would: a grant per session, naming
//! a bundle by URL and hash, no sandbox. Each session does a few turns against the scripted model,
//! every turn reading the bundle's skill. It records what a client can see (open latency, turn latency,
//! errors, how many sessions were open); the replica's own memory, fds and threads come from the
//! sampler in its task (`load.sh` joins the two).

// A stand-in for a test run: a poisoned counter lock is a bug to stop on, not to handle.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::time::{Duration, Instant};

use beyond_test_support::bundle::{dir, file, sha256_hex, tar};
use futures::{SinkExt as _, StreamExt as _};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest as _;
use tokio_tungstenite::tungstenite::http::HeaderValue;

use crate::chat::now_ms;
use crate::keys::{Bundle as GrantBundle, GrantKeys, Limits, MintArgs, mint_grant};

type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;
type Ws =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

pub struct Args {
    /// The replica: host and port of its plain WebSocket listener.
    pub host: String,
    pub port: u16,
    /// The bundle host as the replica reaches it (`ip:port`), and the token for its uploads and stats.
    pub loadhost: String,
    pub token: String,
    pub out: std::path::PathBuf,
    pub sessions: usize,
    pub burst: usize,
    pub tenants: usize,
    pub ramp_secs: u64,
    pub hold_secs: u64,
    pub idle_secs: u64,
    /// Waves of `sessions` concurrent sessions, each followed by `idle_secs` of nothing.
    pub waves: usize,
}

struct Bundle {
    sha: String,
    tenant: String,
    bytes: Vec<u8>,
}

/// A deterministic, incompressible filler of `n` bytes.
fn filler(seed: &str, n: usize) -> Vec<u8> {
    let mut x = u64::from_str_radix(&sha256_hex(seed.as_bytes())[..16], 16).unwrap_or(1) | 1;
    let mut out = Vec::with_capacity(n + 8);
    while out.len() < n {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        out.extend_from_slice(&x.to_le_bytes());
    }
    out.truncate(n);
    out
}

fn build(label: &str, tenant: &str, big: bool, mcp_servers: usize, loadhost: &str) -> Bundle {
    let agents = format!(
        "---\ntools: read\nmax-steps: 4\n---\nYou are the {label} agent of {tenant}. Read your skill, then answer in one line.\n"
    );
    let skill = format!(
        "---\nname: house-style\ndescription: How {tenant} writes its answers. Read it when asked about style.\n---\n\n# House style\n\nHOUSE-STYLE-MARKER: {label}\n"
    );
    let data = if big {
        filler(label, 5 * 1024 * 1024)
    } else {
        Vec::new()
    };
    let mcp = (mcp_servers > 0).then(|| {
        let servers: serde_json::Map<String, Value> = (0..mcp_servers)
            .map(|i| {
                (
                    format!("s{i:03}"),
                    json!({"type":"http","url":format!("http://{loadhost}/mcp/{i}")}),
                )
            })
            .collect();
        json!({"mcpServers": servers}).to_string()
    });
    let mut entries = vec![
        file("AGENTS.md", &agents),
        dir(".agents/"),
        dir(".agents/skills/"),
        dir(".agents/skills/house-style/"),
        file(".agents/skills/house-style/SKILL.md", &skill),
    ];
    let mut data_entry = file(".agents/skills/house-style/data.bin", "");
    data_entry.body = &data;
    if big {
        entries.push(data_entry);
    }
    if let Some(m) = &mcp {
        entries.push(file(".mcp.json", m));
    }
    let bytes = tar(&entries);
    Bundle {
        sha: sha256_hex(&bytes),
        tenant: tenant.to_string(),
        bytes,
    }
}

/// One request to the bundle host over a raw socket.
async fn host(
    addr: &str,
    token: &str,
    method: &str,
    path: &str,
    body: &[u8],
) -> Res<(u16, Vec<u8>)> {
    let mut s = tokio::net::TcpStream::connect(addr).await?;
    let head = format!(
        "{method} {path} HTTP/1.1\r\nhost: {addr}\r\nauthorization: Bearer {token}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
        body.len()
    );
    s.write_all(head.as_bytes()).await?;
    s.write_all(body).await?;
    let mut resp = Vec::new();
    s.read_to_end(&mut resp).await?;
    let split = resp
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or("no response")?;
    let status = std::str::from_utf8(&resp[..split])?
        .split_whitespace()
        .nth(1)
        .unwrap_or("0")
        .parse()?;
    Ok((status, resp[split + 4..].to_vec()))
}

async fn stats(a: &Args, reset: bool) -> Value {
    let path = if reset { "/stats?reset=1" } else { "/stats" };
    match host(&a.loadhost, &a.token, "GET", path, b"").await {
        Ok((200, body)) => serde_json::from_slice(&body).unwrap_or(Value::Null),
        _ => Value::Null,
    }
}

#[derive(Default)]
struct Rec {
    opens: std::sync::Mutex<Vec<Value>>,
    live: AtomicUsize,
    peak: AtomicUsize,
}

struct Ctx {
    a: Args,
    keys: GrantKeys,
    rec: Rec,
    seq: AtomicUsize,
}

struct Conn {
    ws: Ws,
    ctx: Arc<Ctx>,
}

impl Drop for Conn {
    fn drop(&mut self) {
        self.ctx.rec.live.fetch_sub(1, SeqCst);
    }
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[(((v.len() - 1) as f64) * p).round() as usize]
}

/// Open a session on `b` and wait for its first answer. The open is recorded whether it worked or not.
async fn open(ctx: &Arc<Ctx>, phase: &str, b: &Bundle, limit: Option<u32>) -> Option<Conn> {
    let n = ctx.seq.fetch_add(1, SeqCst);
    let session = format!("l{n:06}");
    let grant = mint_grant(
        &ctx.keys,
        &MintArgs {
            tenant: b.tenant.clone(),
            session: session.clone(),
            shard: "demo".into(),
            exec_url: String::new(),
            workspace: "/workspace/load".into(),
            ttl_secs: 7200,
            tamper: false,
            exec_bearer: None,
            exec_headers: Vec::new(),
            connectors: Vec::new(),
            limits: limit.map(|n| Limits {
                max_mcp_json_servers: Some(n),
                ..Default::default()
            }),
            bundle: Some(GrantBundle {
                url: format!("http://{}/b/{}", ctx.a.loadhost, b.sha),
                sha256: b.sha.clone(),
            }),
        },
    )
    .ok()?;
    let url = format!(
        "ws://{}:{}/_beyond/agent?session_id={session}",
        ctx.a.host, ctx.a.port
    );
    let mut req = url.into_client_request().ok()?;
    req.headers_mut()
        .insert("x-beyond-grant", HeaderValue::from_str(&grant).ok()?);
    let t0 = Instant::now();
    let result: Res<(Ws, f64)> = async {
        let (mut ws, _) = tokio::time::timeout(
            Duration::from_secs(60),
            tokio_tungstenite::connect_async(req),
        )
        .await??;
        let connect_ms = t0.elapsed().as_secs_f64() * 1000.0;
        ws.send(Message::text(
            json!({"type":"get_state","id":"g"}).to_string(),
        ))
        .await?;
        tokio::time::timeout(Duration::from_secs(60), async {
            while let Some(m) = ws.next().await {
                if let Message::Text(t) = m? {
                    let f: Value = serde_json::from_str(&t).unwrap_or(Value::Null);
                    if f["type"] == "error" {
                        return Err::<(), Box<dyn std::error::Error + Send + Sync>>(
                            format!("{f}").into(),
                        );
                    }
                    if f["type"] == "response" && f["command"] == "get_state" {
                        return Ok(());
                    }
                }
            }
            Err("closed before get_state".into())
        })
        .await??;
        Ok((ws, connect_ms))
    }
    .await;
    let at_ms = now_ms();
    match result {
        Ok((ws, connect_ms)) => {
            let live = ctx.rec.live.fetch_add(1, SeqCst) + 1;
            ctx.rec.peak.fetch_max(live, SeqCst);
            ctx.rec.opens.lock().unwrap().push(json!({
                "phase": phase, "tenant": b.tenant, "session": session, "at_ms": at_ms, "ok": true,
                "connect_ms": connect_ms, "ready_ms": t0.elapsed().as_secs_f64() * 1000.0,
            }));
            Some(Conn {
                ws,
                ctx: ctx.clone(),
            })
        }
        Err(e) => {
            ctx.rec.opens.lock().unwrap().push(json!({
                "phase": phase, "tenant": b.tenant, "session": session, "at_ms": at_ms, "ok": false,
                "ready_ms": t0.elapsed().as_secs_f64() * 1000.0,
                "error": e.to_string().chars().take(200).collect::<String>(),
            }));
            None
        }
    }
}

/// One turn: a prompt whose answer needs the bundle's skill.
async fn turn(c: &mut Conn, phase: &str) {
    let t0 = Instant::now();
    let id = format!("p{}", c.ctx.seq.fetch_add(1, SeqCst));
    let sent =
        c.ws.send(Message::text(
            json!({"type":"prompt","id":id,"message":"What is the house style?"}).to_string(),
        ))
        .await;
    let mut read_skill = false;
    if sent.is_ok() {
        let _ = tokio::time::timeout(Duration::from_secs(60), async {
            while let Some(Ok(m)) = c.ws.next().await {
                let Message::Text(t) = m else { continue };
                let f: Value = serde_json::from_str(&t).unwrap_or(Value::Null);
                if f["type"] == "event" && f["event"]["kind"] == "tool_end" {
                    read_skill |= f["event"]["result"]
                        .to_string()
                        .contains("HOUSE-STYLE-MARKER");
                }
                if f["type"] == "response" && f["command"] == "prompt" && f["id"] == id.as_str() {
                    return;
                }
            }
        })
        .await;
    }
    c.ctx.rec.opens.lock().unwrap().push(json!({
        "phase": phase, "turn_ms": t0.elapsed().as_secs_f64() * 1000.0, "read_skill": read_skill,
    }));
}

fn jitter(ctx: &Ctx, lo: u64, hi: u64) -> Duration {
    Duration::from_millis(
        lo + (ctx.seq.fetch_add(7, SeqCst) as u64 * 2654435761) % (hi - lo).max(1),
    )
}

/// The slots ramp up over `ramp`, then all are held until `ramp + hold`. 60% are long sessions (three
/// turns, then one more every 30 s). The rest churn on a second set of bundles (open, two turns, close,
/// repeat), so bundles come in while others are released and the cache has something to evict.
async fn wave(ctx: &Arc<Ctx>, name: &str, long: &[Arc<Bundle>], churn: &[Arc<Bundle>]) {
    let (ramp, hold) = (
        Duration::from_secs(ctx.a.ramp_secs),
        Duration::from_secs(ctx.a.hold_secs),
    );
    let end = Instant::now() + ramp + hold;
    let n = ctx.a.sessions;
    let mut tasks = Vec::new();
    for i in 0..n {
        let (ctx, name) = (ctx.clone(), name.to_string());
        let (long, churn) = (long.to_vec(), churn.to_vec());
        tasks.push(tokio::spawn(async move {
            tokio::time::sleep(ramp.mul_f64(i as f64 / n as f64)).await;
            let pick = |set: &[Arc<Bundle>], ctx: &Ctx| {
                set[(ctx.seq.fetch_add(1, SeqCst) * 7919) % set.len()].clone()
            };
            if i % 5 < 3 {
                let b = pick(&long, &ctx);
                let Some(mut c) = open(&ctx, &name, &b, None).await else {
                    return;
                };
                for _ in 0..3 {
                    turn(&mut c, &name).await;
                    tokio::time::sleep(jitter(&ctx, 500, 2000)).await;
                }
                while Instant::now() + Duration::from_secs(30) < end {
                    tokio::time::sleep(Duration::from_secs(30)).await;
                    turn(&mut c, &name).await;
                }
                tokio::time::sleep_until(end.into()).await;
                let _ = c.ws.close(None).await;
            } else {
                while Instant::now() < end {
                    let b = pick(&churn, &ctx);
                    let Some(mut c) = open(&ctx, &name, &b, None).await else {
                        tokio::time::sleep(Duration::from_secs(2)).await;
                        continue;
                    };
                    for _ in 0..2 {
                        turn(&mut c, &name).await;
                    }
                    tokio::time::sleep(jitter(&ctx, 3000, 8000)).await;
                    let _ = c.ws.close(None).await;
                }
            }
        }));
    }
    for t in tasks {
        let _ = t.await;
    }
}

/// `count` sessions at once over the given bundles, one turn each, held `hold`.
async fn burst(ctx: &Arc<Ctx>, name: &str, bundles: &[Arc<Bundle>], count: usize, hold: Duration) {
    let mut tasks = Vec::new();
    for i in 0..count {
        let (ctx, name, b) = (
            ctx.clone(),
            name.to_string(),
            bundles[i % bundles.len()].clone(),
        );
        tasks.push(tokio::spawn(async move {
            let Some(mut c) = open(&ctx, &name, &b, None).await else {
                return;
            };
            turn(&mut c, &name).await;
            tokio::time::sleep(hold).await;
            let _ = c.ws.close(None).await;
        }));
    }
    for t in tasks {
        let _ = t.await;
    }
}

fn summary(ctx: &Ctx, phase: &str) -> Value {
    let rows = ctx.rec.opens.lock().unwrap();
    let (mut ready, mut turns) = (Vec::new(), Vec::new());
    let (mut opens, mut failed, mut skill_ok, mut skill_bad) = (0, 0, 0, 0);
    for r in rows.iter().filter(|r| r["phase"] == phase) {
        if let Some(ms) = r["turn_ms"].as_f64() {
            turns.push(ms);
            if r["read_skill"] == true {
                skill_ok += 1;
            } else {
                skill_bad += 1;
            }
        } else if r["ok"] == true {
            opens += 1;
            ready.push(r["ready_ms"].as_f64().unwrap_or(0.0));
        } else {
            failed += 1;
        }
    }
    json!({
        "opens": opens, "open_failures": failed, "turns": turns.len(),
        "turns_that_read_the_skill": skill_ok, "turns_that_did_not": skill_bad,
        "open_p50_ms": pct(&mut ready, 0.5), "open_p99_ms": pct(&mut ready, 0.99),
        "open_max_ms": pct(&mut ready, 1.0),
        "turn_p50_ms": pct(&mut turns, 0.5), "turn_p99_ms": pct(&mut turns, 0.99),
    })
}

pub async fn run(a: Args) -> Res<Value> {
    let keys = GrantKeys::from_env().map_err(|e| e.to_string())?;
    let ctx = Arc::new(Ctx {
        a,
        keys,
        rec: Rec::default(),
        seq: AtomicUsize::new(0),
    });
    let a = &ctx.a;

    // Round trip to the replica, so the open latencies can be read net of the network.
    let mut rtts = Vec::new();
    for _ in 0..7 {
        let t = Instant::now();
        let _ = tokio::net::TcpStream::connect((a.host.as_str(), a.port)).await?;
        rtts.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let rtt_ms = pct(&mut rtts, 0.5);

    // 150 hashes: the first 100 are the waves' long sessions' (3 of them 5 MB), the next 50 the
    // churn's (7 of them 5 MB). The burst brings 100 more, never fetched before it.
    let mut all = Vec::new();
    for i in 0..150 {
        let big = matches!(i, 5 | 40 | 77 | 105 | 110 | 115 | 120 | 125 | 135 | 145);
        all.push(Arc::new(build(
            &format!("v{}-{i:03}", i % 3 + 1),
            &format!("tenant{:02}", i % a.tenants),
            big,
            0,
            &a.loadhost,
        )));
    }
    let burst_bundles: Vec<_> = (0..a.burst / 2)
        .map(|i| {
            Arc::new(build(
                &format!("burst-{i:03}"),
                &format!("tenant{:02}", i % a.tenants),
                false,
                0,
                &a.loadhost,
            ))
        })
        .collect();
    let mcp_bundle = Arc::new(build("mcp200", "tenant-mcp", false, 200, &a.loadhost));
    let mut uploaded = 0usize;
    for b in all.iter().chain(&burst_bundles).chain([&mcp_bundle]) {
        let (status, _) = host(
            &a.loadhost,
            &a.token,
            "PUT",
            &format!("/b/{}", b.sha),
            &b.bytes,
        )
        .await?;
        if status != 200 {
            return Err(format!("uploading a bundle: HTTP {status}").into());
        }
        uploaded += b.bytes.len();
    }
    eprintln!(
        "uploaded {} bundles, {:.1} MB",
        all.len() + burst_bundles.len() + 1,
        uploaded as f64 / 1e6
    );
    let (long, churn) = (all[..100].to_vec(), all[100..].to_vec());
    stats(a, true).await;

    // The open-session count a client sees, once a second, for the whole run.
    let live_log = Arc::new(std::sync::Mutex::new(Vec::<Value>::new()));
    let live_task = {
        let (ctx, log) = (ctx.clone(), live_log.clone());
        tokio::spawn(async move {
            loop {
                log.lock()
                    .unwrap()
                    .push(json!([now_ms(), ctx.rec.live.load(SeqCst)]));
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        })
    };

    let mut phases = Vec::new();
    let mut host_stats = serde_json::Map::new();
    macro_rules! phase {
        ($name:expr, $body:expr) => {{
            let name: String = $name;
            let start = now_ms();
            ctx.rec.peak.store(ctx.rec.live.load(SeqCst), SeqCst);
            eprintln!("[{start}] {name} starts");
            $body;
            let mut s = summary(&ctx, &name);
            s["name"] = json!(name);
            s["start_ms"] = json!(start);
            s["end_ms"] = json!(now_ms());
            s["peak_open_sessions"] = json!(ctx.rec.peak.load(SeqCst));
            host_stats.insert(name.clone(), stats(a, true).await);
            eprintln!("{s}");
            phases.push(s);
        }};
    }
    let idle = Duration::from_secs(a.idle_secs);
    for w in 1..=a.waves {
        phase!(
            format!("wave{w}"),
            wave(&ctx, &format!("wave{w}"), &long, &churn).await
        );
        phase!(format!("idle{w}"), tokio::time::sleep(idle).await);
    }
    phase!(
        "burst".to_string(),
        burst(
            &ctx,
            "burst",
            &burst_bundles,
            a.burst,
            Duration::from_secs(10)
        )
        .await
    );
    phase!("idle-burst".to_string(), tokio::time::sleep(idle / 2).await);
    for (label, limit) in [("mcp-unset", None), ("mcp-limit5", Some(5u32))] {
        phase!(label.to_string(), {
            let mut tasks = Vec::new();
            for _ in 0..5 {
                let (ctx, b) = (ctx.clone(), mcp_bundle.clone());
                tasks.push(tokio::spawn(async move {
                    if let Some(mut c) = open(&ctx, label, &b, limit).await {
                        turn(&mut c, label).await;
                        tokio::time::sleep(Duration::from_secs(8)).await;
                        let _ = c.ws.close(None).await;
                    }
                }));
            }
            for t in tasks {
                let _ = t.await;
            }
        });
        phase!(
            format!("idle-{label}"),
            tokio::time::sleep(Duration::from_secs(20)).await
        );
    }
    live_task.abort();
    let opens: Vec<Value> = ctx.rec.opens.lock().unwrap().clone();
    let report = json!({
        "rtt_ms": rtt_ms,
        "config": {"sessions": a.sessions, "burst_sessions": a.burst, "tenants": a.tenants,
                   "bundles_in_waves": all.len(), "bundles_in_burst": burst_bundles.len(),
                   "ramp_secs": a.ramp_secs, "hold_secs": a.hold_secs, "idle_secs": a.idle_secs,
                   "uploaded_mb": uploaded as f64 / 1e6, "waves": a.waves},
        "phases": phases,
        "host": host_stats,
        "client_open_sessions": live_log.lock().unwrap().clone(),
        "opens": opens,
    });
    std::fs::write(&a.out, serde_json::to_vec(&report)?)?;
    Ok(report["phases"].clone())
}
