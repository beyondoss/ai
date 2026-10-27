//! execd against a baseline exec endpoint, A/B, over HTTP on loopback.
//!
//! ```sh
//! EXEC_BENCH_A=/path/to/beyond-ai-demo cargo bench -p beyond-execd --bench exec_compare
//! ```
//!
//! **A** is the baseline, a `beyond-ai-demo exec-stub` binary built from the commit before execd
//! (`EXEC_BENCH_A`, required; the stub on a branch that has execd delegates to execd and is not a
//! baseline). **B** is the `execd` binary of this build. Both are configured with the same `EXECD_*` settings and
//! fresh root and home directories. Work is split into rounds; within a round A and B each run the
//! whole battery, in alternating order (AB, BA, AB...), so drift in this machine's load lands on
//! both.
//!
//! Metrics: one-shot round trip (sequential and at concurrency 16), streaming throughput and server CPU per
//! GB, time to first frame, server RSS (idle, 100 idle streams, 16 slow readers) and the latency of
//! kill-on-disconnect. A summary (per metric, side and round: n, p50, p90, p99, max, mean) goes to
//! the JSON file `EXEC_BENCH_OUT` (default `crates/execd/benches/results/<date>.json`) and a table
//! is printed. `EXEC_BENCH_RAW=1` also writes every sample to `EXEC_BENCH_RAW_OUT` (default under
//! `/tmp`); `EXEC_BENCH_SUMMARIZE=<raw file>` turns such a file into the summary.
//!
//! Environment: `EXEC_BENCH_A` (baseline binary), `EXEC_BENCH_B` (default: this build's execd),
//! `EXEC_BENCH_ROUNDS` (default 5), `EXEC_BENCH_SCALE` (multiplies request counts, default 1),
//! `EXEC_BENCH_STREAM_MIB` (stream size per run, default 1024), `EXEC_BENCH_LABEL` (suffix of the
//! results file name), `EXEC_BENCH_A_REV` (recorded only).
//!
//! The client sizes a frame's payload from its base64 length instead of decoding it, so it is never
//! the bottleneck of the throughput runs. Server CPU is the server process's own utime + stime; the
//! commands it spawns are not included, and neither is their memory in the RSS figures.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use nix::sys::signal::{Signal, killpg};
use nix::unistd::Pid;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;

type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

// ---------------------------------------------------------------- servers

#[derive(Clone)]
struct Side {
    name: &'static str,
    bin: PathBuf,
    stub: bool,
    /// Extra environment for the server, from `EXEC_BENCH_<side>_ENV=K=V,K=V`.
    env: Vec<(String, String)>,
}

struct Server {
    child: Child,
    addr: std::net::SocketAddr,
    root: PathBuf,
    _dir: tempfile::TempDir,
}

impl Server {
    fn start(side: &Side) -> Res<Server> {
        let dir = tempfile::tempdir()?;
        let root = dir.path().join("root");
        let home = dir.path().join("home");
        std::fs::create_dir_all(&root)?;
        std::fs::create_dir_all(&home)?;
        std::fs::write(root.join("f4k"), vec![b'x'; 4096])?;
        // `EXEC_BENCH_CPUS=0,1` pins the servers (not the client) as a small VM would be.
        let mut cmd = match std::env::var("EXEC_BENCH_CPUS") {
            Ok(cpus) => {
                let mut c = Command::new("taskset");
                c.args(["-c", &cpus]).arg(&side.bin);
                c
            }
            Err(_) => Command::new(&side.bin),
        };
        cmd.envs(side.env.iter().map(|(k, v)| (k, v)));
        // port-0: the server binds it and keeps it; the port is read back from its own sockets.
        let listen = "127.0.0.1:0";
        if side.stub {
            // The baseline stub predates execd's environment-only configuration: it takes flags.
            cmd.args(["exec-stub", "--listen", listen, "--root"])
                .arg(&root)
                .arg("--home")
                .arg(&home);
        } else {
            cmd.env("EXECD_LISTEN", listen)
                .env("EXECD_ROOT", &root)
                .env("EXECD_HOME", &home);
        }
        cmd.stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        let mut child = cmd.spawn()?;
        let t = Instant::now();
        let addr = loop {
            if let Some(a) = beyond_test_support::ports::listening_on(child.id()).first() {
                break std::net::SocketAddr::V4(*a);
            }
            if t.elapsed() > Duration::from_secs(20) {
                let _ = child.kill();
                let _ = child.wait();
                return Err("server did not listen".into());
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        Ok(Server {
            child,
            addr,
            root,
            _dir: dir,
        })
    }

    fn pid(&self) -> u32 {
        self.child.id()
    }

    /// (VmRSS, VmHWM) in KiB.
    fn rss(&self) -> (u64, u64) {
        let s = std::fs::read_to_string(format!("/proc/{}/status", self.pid())).unwrap_or_default();
        let get = |k: &str| {
            s.lines()
                .find(|l| l.starts_with(k))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse().ok())
                .unwrap_or(0)
        };
        (get("VmRSS:"), get("VmHWM:"))
    }

    /// utime + stime of the server process itself, in seconds (USER_HZ is 100 on Linux).
    fn cpu(&self) -> f64 {
        let s = std::fs::read_to_string(format!("/proc/{}/stat", self.pid())).unwrap_or_default();
        let f = stat_fields(&s);
        let t = |i: usize| f.get(i).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
        (t(11) + t(12)) / 100.0
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        // Whatever the server still runs dies with it; left alone, 100 idle `sleep`s would crowd
        // `/proc` for the next measurement.
        let pid = self.pid();
        let mut kids = Vec::new();
        for e in std::fs::read_dir("/proc").into_iter().flatten().flatten() {
            let Some(k) = e.file_name().to_str().and_then(|n| n.parse::<u32>().ok()) else {
                continue;
            };
            let st = std::fs::read_to_string(format!("/proc/{k}/stat")).unwrap_or_default();
            if stat_fields(&st).get(1).and_then(|x| x.parse::<u32>().ok()) == Some(pid) {
                kids.push(k);
            }
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
        for k in kids {
            let _ = killpg(Pid::from_raw(k as i32), Signal::SIGKILL);
            let _ = nix::sys::signal::kill(Pid::from_raw(k as i32), Signal::SIGKILL);
        }
    }
}

/// The fields of `/proc/<pid>/stat` after the command name: [0] state, [1] ppid, [2] pgrp, ...
fn stat_fields(stat: &str) -> Vec<&str> {
    stat.rsplit_once(')')
        .map_or("", |x| x.1)
        .split_whitespace()
        .collect()
}

// ---------------------------------------------------------------- client

enum Frame {
    Out(usize),
    Other,
    Final(Value),
}

struct Conn {
    addr: std::net::SocketAddr,
    /// The peer said `Connection: close` (the baseline does, on every response): the next request
    /// opens a new connection, and that connect is part of the round trip.
    close: bool,
    r: BufReader<TcpStream>,
    pending: Vec<u8>,
    start: usize,
    chunked: bool,
    clen: usize,
    ended: bool,
}

fn classify(line: &[u8]) -> Frame {
    for p in [&b"{\"stdout\":\""[..], &b"{\"stderr\":\""[..]] {
        if line.starts_with(p) && line.ends_with(b"\"}") {
            let b64 = line.len() - p.len() - 2;
            let pad = line[..line.len() - 2]
                .iter()
                .rev()
                .take(2)
                .filter(|&&c| c == b'=')
                .count();
            return Frame::Out(b64 / 4 * 3 - pad);
        }
    }
    let v: Value = serde_json::from_slice(line).unwrap_or(Value::Null);
    let Some(o) = v.as_object() else {
        return Frame::Other;
    };
    if ["exit_code", "signal", "timed_out", "error"]
        .iter()
        .any(|k| o.contains_key(*k))
    {
        Frame::Final(v)
    } else if let Some(s) = o.get("stdout").or(o.get("stderr")).and_then(Value::as_str) {
        Frame::Out(s.len() / 4 * 3)
    } else {
        Frame::Other
    }
}

impl Conn {
    async fn open(addr: std::net::SocketAddr) -> Res<Conn> {
        let s = TcpStream::connect(addr).await?;
        s.set_nodelay(true)?;
        Ok(Conn {
            addr,
            close: false,
            r: BufReader::with_capacity(128 * 1024, s),
            pending: Vec::new(),
            start: 0,
            chunked: false,
            clen: 0,
            ended: false,
        })
    }

    /// Send one request and read the response head. Returns the status.
    async fn request(&mut self, body: &Value) -> Res<u16> {
        if self.close {
            let s = TcpStream::connect(self.addr).await?;
            s.set_nodelay(true)?;
            self.r = BufReader::with_capacity(128 * 1024, s);
            self.close = false;
        }
        let b = body.to_string();
        let mut msg = format!(
            "POST /exec HTTP/1.1\r\nHost: bench\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            b.len()
        )
        .into_bytes();
        msg.extend_from_slice(b.as_bytes());
        self.r.get_mut().write_all(&msg).await?;
        self.pending.clear();
        self.start = 0;
        self.ended = false;
        self.chunked = false;
        self.clen = 0;
        let mut line = String::new();
        self.r.read_line(&mut line).await?;
        let status: u16 = line
            .split_whitespace()
            .nth(1)
            .ok_or_else(|| format!("bad status line {line:?}"))?
            .parse()?;
        loop {
            line.clear();
            if self.r.read_line(&mut line).await? == 0 {
                return Err("eof in head".into());
            }
            let l = line.trim_end().to_ascii_lowercase();
            if l.is_empty() {
                break;
            }
            if let Some(v) = l.strip_prefix("content-length:") {
                self.clen = v.trim().parse()?;
            } else if l.starts_with("transfer-encoding:") && l.contains("chunked") {
                self.chunked = true;
            } else if l.starts_with("connection:") && l.contains("close") {
                self.close = true;
            }
        }
        Ok(status)
    }

    /// Read more of the body into `pending`; false at its end.
    async fn fill(&mut self) -> Res<bool> {
        if self.ended {
            return Ok(false);
        }
        if self.start > 0 {
            self.pending.drain(..self.start);
            self.start = 0;
        }
        let size = if self.chunked {
            let mut l = String::new();
            self.r.read_line(&mut l).await?;
            usize::from_str_radix(l.trim().split(';').next().unwrap_or("0"), 16)?
        } else {
            self.ended = true;
            self.clen
        };
        if size == 0 {
            self.ended = true;
            if self.chunked {
                let mut l = String::new();
                self.r.read_line(&mut l).await?;
            }
            return Ok(false);
        }
        let old = self.pending.len();
        self.pending.resize(old + size, 0);
        self.r.read_exact(&mut self.pending[old..]).await?;
        if self.chunked {
            let mut crlf = [0u8; 2];
            self.r.read_exact(&mut crlf).await?;
        }
        Ok(true)
    }

    async fn next_frame(&mut self) -> Res<Option<Frame>> {
        loop {
            if let Some(p) = memchr::memchr(b'\n', &self.pending[self.start..]) {
                let f = classify(&self.pending[self.start..self.start + p]);
                self.start += p + 1;
                return Ok(Some(f));
            }
            if !self.fill().await? {
                return Ok(None);
            }
        }
    }

    /// A one-shot exchange, whole: the exit code must be 0.
    async fn exec(&mut self, body: &Value) -> Res<()> {
        let st = self.request(body).await?;
        while self.fill().await? {}
        if st != 200 {
            return Err(format!("status {st}").into());
        }
        let v: Value = serde_json::from_slice(&self.pending)?;
        if v["exit_code"] != 0 {
            return Err(format!("exit {v}").into());
        }
        Ok(())
    }

    /// Open a stream.
    async fn stream(&mut self, mut body: Value) -> Res<()> {
        body["stream"] = json!(true);
        let st = self.request(&body).await?;
        if st != 200 {
            return Err(format!("stream status {st}").into());
        }
        Ok(())
    }

    /// Drain the stream, returning decoded output bytes. The final frame must be exit 0.
    async fn drain(&mut self) -> Res<u64> {
        let mut n = 0u64;
        while let Some(f) = self.next_frame().await? {
            match f {
                Frame::Out(k) => n += k as u64,
                Frame::Final(v) => {
                    if v["exit_code"] != 0 {
                        return Err(format!("final {v}").into());
                    }
                    return Ok(n);
                }
                Frame::Other => {}
            }
        }
        Err("stream ended without a final frame".into())
    }

    /// Wait for the first output frame.
    async fn first_output(&mut self) -> Res<()> {
        loop {
            match self.next_frame().await? {
                Some(Frame::Out(_)) => return Ok(()),
                Some(_) => {}
                None => return Err("stream ended before any output".into()),
            }
        }
    }
}

fn req(cmd: &str, args: &[&str], cwd: &Path) -> Value {
    json!({"command": cmd, "args": args, "cwd": cwd, "timeout_ms": 600_000})
}

// ---------------------------------------------------------------- stats

fn pct(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[((p * s.len() as f64).ceil() as usize).clamp(1, s.len()) - 1]
}

fn min_max(v: &[f64]) -> (f64, f64) {
    (
        v.iter().copied().fold(f64::INFINITY, f64::min),
        v.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    )
}

// ---------------------------------------------------------------- batteries

/// `fresh` opens a new connection for every request, whatever the server allows.
async fn v11_seq(s: &Server, body: Value, n: usize, fresh: bool) -> Res<Vec<f64>> {
    let mut c = Conn::open(s.addr).await?;
    for i in 0..20 {
        c.exec(&body)
            .await
            .map_err(|e| format!("warmup {i}: {e}"))?;
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        c.close |= fresh;
        let t = Instant::now();
        c.exec(&body)
            .await
            .map_err(|e| format!("request {i}: {e}"))?;
        out.push(t.elapsed().as_secs_f64() * 1e3);
    }
    Ok(out)
}

async fn v11_conc(s: &Server, body: Value, per: usize) -> Res<Vec<f64>> {
    let mut hs = Vec::new();
    for _ in 0..16 {
        let (addr, body) = (s.addr, body.clone());
        hs.push(tokio::spawn(async move {
            let mut c = Conn::open(addr).await?;
            c.exec(&body).await?;
            let mut out = Vec::with_capacity(per);
            for _ in 0..per {
                let t = Instant::now();
                c.exec(&body).await?;
                out.push(t.elapsed().as_secs_f64() * 1e3);
            }
            Res::Ok(out)
        }));
    }
    let mut all = Vec::new();
    for h in hs {
        all.extend(h.await??);
    }
    Ok(all)
}

/// (MB/s of decoded payload, server CPU-seconds per GB)
async fn throughput(s: &Server, body: Value, mib: u64) -> Res<(f64, f64)> {
    let mut c = Conn::open(s.addr).await?;
    let cpu0 = s.cpu();
    let t = Instant::now();
    c.stream(body).await?;
    let n = c.drain().await?;
    let secs = t.elapsed().as_secs_f64();
    let cpu = s.cpu() - cpu0;
    if n != mib << 20 {
        return Err(format!("expected {} bytes, got {n}", mib << 20).into());
    }
    let gb = n as f64 / 1e9;
    Ok((n as f64 / 1e6 / secs, cpu / gb))
}

/// 16 concurrent streams of `mib` each, read as fast as the client can: (aggregate MB/s of decoded
/// payload, server CPU-seconds per GB).
async fn throughput16(s: &Server, mib: u64) -> Res<(f64, f64)> {
    let body = req(
        "head",
        &["-c", &(mib << 20).to_string(), "/dev/zero"],
        &s.root,
    );
    let cpu0 = s.cpu();
    let t = Instant::now();
    let mut hs = Vec::new();
    for _ in 0..16 {
        let (addr, body) = (s.addr, body.clone());
        hs.push(tokio::spawn(async move {
            let mut c = Conn::open(addr).await?;
            c.stream(body).await?;
            c.drain().await
        }));
    }
    let mut total = 0u64;
    for h in hs {
        total += h.await??;
    }
    let secs = t.elapsed().as_secs_f64();
    let cpu = s.cpu() - cpu0;
    let gb = total as f64 / 1e9;
    Ok((gb * 1e3 / secs, cpu / gb))
}

/// Request to first output frame; the connection is dropped there, which kills the `sleep`.
async fn ttff(s: &Server, n: usize) -> Res<Vec<f64>> {
    let body = req("sh", &["-c", "printf a; sleep 1"], &s.root);
    let mut out = Vec::new();
    for i in 0..n + 5 {
        let mut c = Conn::open(s.addr).await?;
        let t = Instant::now();
        c.stream(body.clone()).await?;
        c.first_output().await?;
        if i >= 5 {
            out.push(t.elapsed().as_secs_f64() * 1e3);
        }
    }
    Ok(out)
}

/// Processes whose parent is `ppid`, which lead their own group, and whose command line holds
/// `sleep 600`.
fn group_leaders(ppid: u32) -> Vec<u32> {
    let mut v = Vec::new();
    for e in std::fs::read_dir("/proc").into_iter().flatten().flatten() {
        let Some(pid) = e.file_name().to_str().and_then(|n| n.parse::<u32>().ok()) else {
            continue;
        };
        let Ok(st) = std::fs::read_to_string(format!("/proc/{pid}/stat")) else {
            continue;
        };
        let f = stat_fields(&st);
        let num = |i: usize| f.get(i).and_then(|x| x.parse::<u32>().ok());
        if num(1) == Some(ppid) && num(2) == Some(pid) {
            let cl = std::fs::read(format!("/proc/{pid}/cmdline")).unwrap_or_default();
            if String::from_utf8_lossy(&cl).contains("sleep 600") {
                v.push(pid);
            }
        }
    }
    v
}

/// Live (non-zombie) members of a process group: one scan of `/proc`.
fn group_members(pgid: u32) -> Vec<u32> {
    let mut v = Vec::new();
    for e in std::fs::read_dir("/proc").into_iter().flatten().flatten() {
        let Some(pid) = e.file_name().to_str().and_then(|n| n.parse::<u32>().ok()) else {
            continue;
        };
        if let Ok(st) = std::fs::read_to_string(format!("/proc/{pid}/stat")) {
            let f = stat_fields(&st);
            if f.get(2).and_then(|x| x.parse::<u32>().ok()) == Some(pgid) && f.first() != Some(&"Z")
            {
                v.push(pid);
            }
        }
    }
    v
}

/// Whether `pid` runs nothing any more: gone, or a zombie waiting for its reaper.
fn is_dead(pid: u32) -> bool {
    std::fs::read_to_string(format!("/proc/{pid}/stat"))
        .map_or(true, |st| stat_fields(&st).first() == Some(&"Z"))
}

/// Milliseconds from dropping the connection until the command's whole group is gone.
async fn kill_latency(s: &Server) -> Res<f64> {
    let before: HashSet<u32> = group_leaders(s.pid()).into_iter().collect();
    let mut c = Conn::open(s.addr).await?;
    c.stream(req(
        "sh",
        &["-c", "printf a; sleep 600 & sleep 600"],
        &s.root,
    ))
    .await?;
    c.first_output().await?;
    let t0 = Instant::now();
    let pgid = loop {
        let found = group_leaders(s.pid())
            .into_iter()
            .find(|p| !before.contains(p));
        if let Some(p) = found.filter(|&p| group_members(p).len() >= 2) {
            break p;
        }
        if t0.elapsed() > Duration::from_secs(5) {
            return Err("command group not found".into());
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    };
    let pg = Pid::from_raw(pgid as i32);
    // The members are listed before the drop, so the poll reads only their own stat files and its
    // cost does not grow with the number of processes on the host.
    let members = group_members(pgid);
    drop(c);
    let t = Instant::now();
    loop {
        // Gone means no live member: a zombie waiting for its reaper runs nothing.
        if members.iter().all(|&p| is_dead(p)) {
            return Ok(t.elapsed().as_secs_f64() * 1e3);
        }
        if t.elapsed() > Duration::from_secs(15) {
            let _ = killpg(pg, Signal::SIGKILL);
            return Ok(15_000.0);
        }
        tokio::time::sleep(Duration::from_micros(500)).await;
    }
}

/// Server CPU seconds spent over `secs` while one stream holds a command whose shell has exited
/// and left a process running (a kept dev server): what the leftover bookkeeping costs.
async fn leftover_cpu(s: &Server, secs: u64) -> Res<f64> {
    let mut c = Conn::open(s.addr).await?;
    c.stream(req("sh", &["-c", "sleep 3600 &"], &s.root))
        .await?;
    // The shell is gone and the stream stays open: wait for `shell_exit`, and a moment more.
    tokio::time::sleep(Duration::from_secs(2)).await;
    let cpu0 = s.cpu();
    tokio::time::sleep(Duration::from_secs(secs)).await;
    let used = s.cpu() - cpu0;
    drop(c);
    Ok(used)
}

async fn mem_idle(side: &Side) -> Res<u64> {
    let s = Server::start(side)?;
    tokio::time::sleep(Duration::from_millis(500)).await;
    Ok(s.rss().0)
}

async fn mem_streams(side: &Side) -> Res<u64> {
    let s = Server::start(side)?;
    let mut cs = Vec::new();
    for _ in 0..100 {
        let mut c = Conn::open(s.addr).await?;
        c.stream(req("sleep", &["30"], &s.root)).await?;
        cs.push(c);
    }
    tokio::time::sleep(Duration::from_secs(1)).await;
    Ok(s.rss().0)
}

/// (max sampled RSS, VmHWM) in KiB with 16 slow readers of 256 MiB streams.
async fn mem_slow(side: &Side, secs: u64) -> Res<(u64, u64)> {
    let s = Server::start(side)?;
    let mut hs = Vec::new();
    for _ in 0..16 {
        let (addr, root) = (s.addr, s.root.clone());
        hs.push(tokio::spawn(async move {
            let mut c = Conn::open(addr).await?;
            c.stream(req("head", &["-c", "268435456", "/dev/zero"], &root))
                .await?;
            // One frame per 50 ms: about 1.3 MB/s, so the stream outlasts the sampling window.
            while c.next_frame().await?.is_some() {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            Res::Ok(())
        }));
    }
    let mut peak = 0;
    let t = Instant::now();
    while t.elapsed() < Duration::from_secs(secs) {
        peak = peak.max(s.rss().0);
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let hwm = s.rss().1;
    for h in hs {
        h.abort();
    }
    Ok((peak, hwm))
}

// ---------------------------------------------------------------- driver

#[derive(Default)]
struct Samples {
    seq_true: Vec<Vec<f64>>,
    seq_cat: Vec<Vec<f64>>,
    fresh_true: Vec<Vec<f64>>,
    /// `true` round trips with `SLEEPERS[k]` idle processes added to the host, by k.
    crowded: [Vec<Vec<f64>>; 3],
    conc_true: Vec<Vec<f64>>,
    conc_cat: Vec<Vec<f64>>,
    zeros_mbs: Vec<f64>,
    zeros_cpu: Vec<f64>,
    rand_mbs: Vec<f64>,
    rand_cpu: Vec<f64>,
    ttff: Vec<Vec<f64>>,
    kill: Vec<Vec<f64>>,
    rss_idle: Vec<u64>,
    rss_100: Vec<u64>,
    rss_slow_peak: Vec<u64>,
    rss_slow_hwm: Vec<u64>,
    leftover_cpu: Vec<f64>,
    stream16_mbs: Vec<f64>,
    stream16_cpu: Vec<f64>,
}

/// Idle processes added to the host for the crowded round trips.
const SLEEPERS: [usize; 3] = [0, 100, 1000];

fn side_env(var: &str) -> Vec<(String, String)> {
    std::env::var(var)
        .unwrap_or_default()
        .split(',')
        .filter_map(|kv| kv.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn count_procs() -> usize {
    std::fs::read_dir("/proc")
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| {
            e.file_name()
                .to_str()
                .is_some_and(|n| n.parse::<u32>().is_ok())
        })
        .count()
}

fn env_num(k: &str, d: u64) -> u64 {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .unwrap_or_default()
        .lines()
        .find(|l| l.starts_with("model name"))
        .and_then(|l| l.split(':').nth(1))
        .map_or("unknown".into(), |s| s.trim().to_string())
}

fn cmd_out(c: &str, a: &[&str]) -> String {
    Command::new(c)
        .args(a)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

fn lat_json(rounds: &[Vec<f64>]) -> Value {
    let all: Vec<f64> = rounds.iter().flatten().copied().collect();
    let p50s: Vec<f64> = rounds.iter().map(|r| pct(r, 0.5)).collect();
    let p99s: Vec<f64> = rounds.iter().map(|r| pct(r, 0.99)).collect();
    json!({
        "n": all.len(), "unit": "ms",
        "p50": pct(&all, 0.5), "p99": pct(&all, 0.99), "max": pct(&all, 1.0),
        "round_p50_min_max": min_max(&p50s), "round_p99_min_max": min_max(&p99s),
        "samples_by_round": rounds,
    })
}

/// n, p50, p90, p99, max and mean of one batch of samples.
fn stats(v: &[f64]) -> Value {
    let mean = v.iter().sum::<f64>() / v.len().max(1) as f64;
    json!({
        "n": v.len(), "p50": pct(v, 0.5), "p90": pct(v, 0.9), "p99": pct(v, 0.99),
        "max": pct(v, 1.0), "mean": mean,
    })
}

/// The committed form of a result: every latency metric keeps its pooled statistics and one row of
/// statistics per round (the per-round medians behind the spreads quoted in ARCHITECTURE.md), and
/// drops the samples. Throughput, CPU and memory metrics are a handful of numbers per side and stay
/// whole. Set `EXEC_BENCH_RAW` to keep the samples as well, in a file outside the repository.
fn summarize(v: &Value) -> Value {
    match v {
        Value::Object(o) if o.contains_key("samples_by_round") => {
            let rounds: Vec<Vec<f64>> =
                serde_json::from_value(o["samples_by_round"].clone()).unwrap_or_default();
            let all: Vec<f64> = rounds.iter().flatten().copied().collect();
            let mut out = o.clone();
            out.remove("samples_by_round");
            let pooled = stats(&all);
            for k in ["p90", "mean"] {
                out.insert(k.into(), pooled[k].clone());
            }
            let rows: Vec<Value> = rounds.iter().map(|r| stats(r)).collect();
            out.insert("rounds".into(), Value::Array(rows));
            Value::Object(out)
        }
        Value::Object(o) => {
            Value::Object(o.iter().map(|(k, x)| (k.clone(), summarize(x))).collect())
        }
        other => other.clone(),
    }
}

fn scalar_json(v: &[f64], unit: &str) -> Value {
    let (min, max) = min_max(v);
    json!({"n": v.len(), "unit": unit, "median": pct(v, 0.5), "min": min, "max": max, "samples": v})
}

fn mib(v: &[u64]) -> Value {
    let f: Vec<f64> = v.iter().map(|x| *x as f64 / 1024.0).collect();
    scalar_json(&f, "MiB")
}

fn side_json(s: &Samples) -> Value {
    json!({
        "v11_true_seq": lat_json(&s.seq_true), "v11_cat4k_seq": lat_json(&s.seq_cat),
        "v11_true_new_conn_each": lat_json(&s.fresh_true),
        "v11_true_plus_0_idle_procs": lat_json(&s.crowded[0]),
        "v11_true_plus_100_idle_procs": lat_json(&s.crowded[1]),
        "v11_true_plus_1000_idle_procs": lat_json(&s.crowded[2]),
        "v11_true_conc16": lat_json(&s.conc_true), "v11_cat4k_conc16": lat_json(&s.conc_cat),
        "stream_zeros_mb_s": scalar_json(&s.zeros_mbs, "MB/s"),
        "stream_zeros_cpu_s_per_gb": scalar_json(&s.zeros_cpu, "cpu-s/GB"),
        "stream_random_mb_s": scalar_json(&s.rand_mbs, "MB/s"),
        "stream_random_cpu_s_per_gb": scalar_json(&s.rand_cpu, "cpu-s/GB"),
        "ttff": lat_json(&s.ttff), "kill_on_disconnect": lat_json(&s.kill),
        "rss_idle": mib(&s.rss_idle), "rss_100_streams": mib(&s.rss_100),
        "rss_slow_16x256MiB_peak_sampled": mib(&s.rss_slow_peak),
        "rss_slow_16x256MiB_hwm": mib(&s.rss_slow_hwm),
        "leftover_stream_server_cpu_s": scalar_json(&s.leftover_cpu, "cpu-s"),
        "stream16_256MiB_mb_s": scalar_json(&s.stream16_mbs, "MB/s aggregate"),
        "stream16_256MiB_cpu_s_per_gb": scalar_json(&s.stream16_cpu, "cpu-s/GB"),
    })
}

fn print_rows(name: &str, a: &Value, b: &Value, keys: &[(&str, &str)]) {
    let f = |v: &Value, k: &str| v.get(k).and_then(Value::as_f64).unwrap_or(f64::NAN);
    for (k, label) in keys {
        println!(
            "{name:<34} {label:<7} A {:>10.3}   B {:>10.3}",
            f(a, k),
            f(b, k)
        );
    }
}

#[tokio::main]
async fn main() -> Res<()> {
    // Turn a raw results file into the committed summary and stop.
    if let Ok(raw) = std::env::var("EXEC_BENCH_SUMMARIZE") {
        let out = std::env::var("EXEC_BENCH_OUT").map_err(|_| "set EXEC_BENCH_OUT")?;
        let v: Value = serde_json::from_str(&std::fs::read_to_string(raw)?)?;
        std::fs::write(&out, serde_json::to_string_pretty(&summarize(&v))? + "\n")?;
        return Ok(());
    }
    let started = Instant::now();
    let procs_visible = std::fs::read_dir("/proc")?
        .flatten()
        .filter(|e| {
            e.file_name()
                .to_str()
                .is_some_and(|n| n.parse::<u32>().is_ok())
        })
        .count();
    let a_bin = std::env::var("EXEC_BENCH_A").map_err(|_| {
        "set EXEC_BENCH_A to the baseline beyond-ai-demo binary (built from the commit before execd)"
    })?;
    let b_bin =
        std::env::var("EXEC_BENCH_B").unwrap_or_else(|_| env!("CARGO_BIN_EXE_execd").to_string());
    let sides = [
        Side {
            name: "A",
            bin: a_bin.into(),
            // `EXEC_BENCH_A_PLAIN=1`: A is an execd binary too (comparing two builds of it).
            stub: std::env::var_os("EXEC_BENCH_A_PLAIN").is_none(),
            env: side_env("EXEC_BENCH_A_ENV"),
        },
        Side {
            name: "B",
            bin: b_bin.into(),
            stub: false,
            env: side_env("EXEC_BENCH_B_ENV"),
        },
    ];
    let rounds = env_num("EXEC_BENCH_ROUNDS", 5) as usize;
    let scale = env_num("EXEC_BENCH_SCALE", 1) as usize;
    let stream_mib = env_num("EXEC_BENCH_STREAM_MIB", 1024);
    let (n_seq, n_conc, n_ttff, n_kill) = (400 * scale, 40 * scale, 40 * scale, 10 * scale);
    let mem_rounds = 2.min(rounds);
    // Which parts to run (default all): v11, sleepers, stream, stream16, ttff, kill, mem.
    let parts = std::env::var("EXEC_BENCH_PARTS")
        .unwrap_or_else(|_| "v11,sleepers,stream,ttff,kill,mem".to_string());
    let on = |p: &str| parts.split(',').any(|x| x == p);

    // Incompressible payload, generated once.
    let random = tempfile::NamedTempFile::new()?;
    let st = Command::new("head")
        .args(["-c", &(stream_mib << 20).to_string(), "/dev/urandom"])
        .stdout(random.reopen()?)
        .status()?;
    assert!(st.success());
    let random_path = random.path().to_str().unwrap().to_string();
    let size = (stream_mib << 20).to_string();

    let srv = [Server::start(&sides[0])?, Server::start(&sides[1])?];
    let mut res: [Samples; 2] = [Samples::default(), Samples::default()];
    for round in 0..rounds {
        let order = if round % 2 == 0 { [0, 1] } else { [1, 0] };
        for i in order {
            let (s, r) = (&srv[i], &mut res[i]);
            eprintln!(
                "round {round} side {} start (processes visible: {})",
                sides[i].name,
                count_procs()
            );
            let t = Instant::now();
            let tr = req("true", &[], &s.root);
            let cat = req("cat", &["f4k"], &s.root);
            if on("v11") {
                r.seq_true.push(v11_seq(s, tr.clone(), n_seq, false).await?);
                r.seq_cat.push(v11_seq(s, cat.clone(), n_seq, false).await?);
                r.fresh_true
                    .push(v11_seq(s, tr.clone(), n_seq, true).await?);
                r.conc_true.push(v11_conc(s, tr.clone(), n_conc).await?);
                r.conc_cat.push(v11_conc(s, cat.clone(), n_conc).await?);
            }
            // The same round trip with the host holding more processes: what a scan of `/proc`
            // costs, if the endpoint does one per command.
            for (k, extra) in SLEEPERS.iter().enumerate().filter(|_| on("sleepers")) {
                let mut idle: Vec<Child> = (0..*extra)
                    .map(|_| {
                        Command::new("sleep")
                            .arg("3600")
                            .stdin(Stdio::null())
                            .spawn()
                    })
                    .collect::<Result<_, _>>()?;
                let lat = v11_seq(s, tr.clone(), n_seq / 2, false).await;
                for c in &mut idle {
                    let _ = c.kill();
                    let _ = c.wait();
                }
                r.crowded[k].push(lat?);
            }
            if on("stream") {
                let zeros = req("head", &["-c", &size, "/dev/zero"], &s.root);
                let (m, c) = throughput(s, zeros, stream_mib).await?;
                r.zeros_mbs.push(m);
                r.zeros_cpu.push(c);
                let rand = req("cat", &[&random_path], &s.root);
                let (m, c) = throughput(s, rand, stream_mib).await?;
                r.rand_mbs.push(m);
                r.rand_cpu.push(c);
            }
            if on("stream16") {
                let (m, c) = throughput16(s, 256).await?;
                r.stream16_mbs.push(m);
                r.stream16_cpu.push(c);
            }
            if on("ttff") {
                r.ttff.push(ttff(s, n_ttff).await?);
            }
            if on("kill") {
                let mut k = Vec::new();
                for _ in 0..n_kill {
                    k.push(kill_latency(s).await?);
                }
                r.kill.push(k);
            }
            if on("mem") && round < mem_rounds {
                r.rss_idle.push(mem_idle(&sides[i]).await?);
                r.rss_100.push(mem_streams(&sides[i]).await?);
                // Once: a baseline that buffers a slow reader's output holds gigabytes here.
                if round == 0 {
                    let (p, h) = mem_slow(&sides[i], 8).await?;
                    r.rss_slow_peak.push(p);
                    r.rss_slow_hwm.push(h);
                    let secs = env_num("EXEC_BENCH_LEFTOVER_SECS", 30);
                    r.leftover_cpu.push(leftover_cpu(s, secs).await?);
                }
            }
            eprintln!(
                "round {round} side {} done in {:.1}s",
                sides[i].name,
                t.elapsed().as_secs_f64()
            );
        }
    }
    drop(srv);

    let wall = started.elapsed().as_secs_f64();
    let date = cmd_out("date", &["+%F"]);
    let out = json!({
        "date": date, "wall_seconds": wall, "rounds": rounds, "scale": scale, "stream_mib": stream_mib,
        "host": {
            "cpu": cpu_model(),
            "processes_visible": procs_visible, "label": std::env::var("EXEC_BENCH_LABEL").unwrap_or_default(),
            "cores": std::thread::available_parallelism().map_or(0, |n| n.get()),
            "kernel": cmd_out("uname", &["-r"]),
            "loadavg_at_end": std::fs::read_to_string("/proc/loadavg").unwrap_or_default().trim(),
        },
        "a": {"bin": sides[0].bin, "rev": std::env::var("EXEC_BENCH_A_REV").unwrap_or_default()},
        "b": {"bin": sides[1].bin},
        "results": {"A": side_json(&res[0]), "B": side_json(&res[1])},
    });
    let tag = std::env::var("EXEC_BENCH_LABEL").map_or(String::new(), |l| format!("-{l}"));
    let path = std::env::var("EXEC_BENCH_OUT").unwrap_or_else(|_| {
        format!(
            "{}/benches/results/{date}{tag}.json",
            env!("CARGO_MANIFEST_DIR")
        )
    });
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&summarize(&out))? + "\n",
    )?;
    if std::env::var_os("EXEC_BENCH_RAW").is_some() {
        let raw = std::env::var("EXEC_BENCH_RAW_OUT")
            .unwrap_or_else(|_| format!("/tmp/exec_bench_raw-{date}{tag}.json"));
        std::fs::write(&raw, serde_json::to_string_pretty(&out)? + "\n")?;
        println!("raw samples: {raw}");
    }
    println!("\nwrote {path}  (wall {wall:.0}s)\n");
    let (ra, rb) = (&out["results"]["A"], &out["results"]["B"]);
    for k in [
        "v11_true_seq",
        "v11_cat4k_seq",
        "v11_true_new_conn_each",
        "v11_true_plus_0_idle_procs",
        "v11_true_plus_100_idle_procs",
        "v11_true_plus_1000_idle_procs",
        "v11_true_conc16",
        "v11_cat4k_conc16",
        "ttff",
        "kill_on_disconnect",
    ] {
        print_rows(
            k,
            &ra[k],
            &rb[k],
            &[("p50", "p50 ms"), ("p99", "p99 ms"), ("max", "max ms")],
        );
    }
    for k in [
        "stream_zeros_mb_s",
        "stream_zeros_cpu_s_per_gb",
        "stream_random_mb_s",
        "stream_random_cpu_s_per_gb",
        "stream16_256MiB_mb_s",
        "stream16_256MiB_cpu_s_per_gb",
        "rss_idle",
        "rss_100_streams",
        "rss_slow_16x256MiB_peak_sampled",
        "rss_slow_16x256MiB_hwm",
        "leftover_stream_server_cpu_s",
    ] {
        print_rows(
            k,
            &ra[k],
            &rb[k],
            &[("median", "median"), ("min", "min"), ("max", "max")],
        );
    }
    Ok(())
}
