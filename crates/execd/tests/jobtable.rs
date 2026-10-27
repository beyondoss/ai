//! A stateful randomized test of the job table, against the real binary.
//!
//! Random sequences of job operations (start held or unheld, keep with and without `notify`, attach
//! from offsets, read, kill with and without remove, list, remove, wipe, seen, and a few malformed
//! requests) are drawn from a seed and driven across several scopes with colliding and adversarial
//! ids, with a small model alongside. After every operation the whole state is checked:
//!
//! - every scope lists exactly the model's jobs for it, and nothing of another scope: no id, no
//!   output, no state, no error text;
//! - the processes of every job are in the state the model says: running ones alive (a kill or wipe
//!   in another scope never touched them), killed, wiped and removed ones gone, and gone *before the
//!   answer* of the operation that killed them (the spec's bound is the request's own `grace_ms`
//!   plus a second);
//! - no operation panicked or left a lock poisoned: the log never says `panicked`, and every later
//!   request still works;
//! - the answer of every operation matches the spec's table for the state the model is in;
//! - job-end callbacks only ever name a job kept with a `notify`, with that job's own token;
//! - at the end, `SIGTERM` leaves no process of any job alive.
//!
//! The operation list does not depend on the state it runs in (an operation on a job that does not
//! exist is part of the test), so a failing list can be shrunk by deleting operations. The run
//! prints the seed and the shrunk list.
//!
//! **Knobs.** Fixed seeds run by default (well under a minute). `EXECD_JOBS_SEED=<n>` replays one
//! seed; `EXECD_JOBS_OPS` sets the length; `EXECD_JOBS_CASES=<n>` runs `n` seeds; `EXECD_JOBS_SECS=<s>`
//! keeps drawing fresh seeds for that long (the deep workflow runs it for ten minutes).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::collections::{BTreeSet, HashMap, HashSet};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use base64::Engine as _;
use bytes::Bytes;
use http_body_util::{BodyExt as _, Full};
use hyper_util::rt::TokioIo;
use serde_json::{Value, json};
use support::baseline::Baseline;
use support::{Execd, http::post};
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

// ---------------------------------------------------------------------------------------------
// the pools every operation draws from
// ---------------------------------------------------------------------------------------------

/// Scopes: three that most operations use, and lookalikes (a trailing space, another case, a NUL).
const SCOPES: [&str; 6] = ["s1", "s2", "s3", "s1 ", "S1", "s\u{0}1"];

/// Ids: three that every scope reuses, and adversarial ones (empty, a path, another case, a
/// newline, non-ASCII, and a long one built in [`id`]).
const IDS: [&str; 9] = [
    "a1",
    "b2",
    "c3",
    "",
    "../a1",
    "A1",
    "a1\n",
    "\u{e9}\u{4e2d}",
    "LONG",
];

fn id(i: usize) -> String {
    if IDS[i] == "LONG" {
        "x".repeat(4096)
    } else {
        IDS[i].to_string()
    }
}

// ---------------------------------------------------------------------------------------------
// the operations
// ---------------------------------------------------------------------------------------------

/// What a job's command does. Every one writes its pids to a file first, so the test can see its
/// processes whatever the endpoint says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// Runs until killed.
    Long,
    /// Runs until `SIGKILL`: ignores `SIGTERM`, and so does its child.
    Stubborn,
    /// Runs until the test creates its gate file, then exits 5.
    Gate,
    /// As [`Kind::Gate`] but exits 6 leaving a background process: state `exited`.
    GateLeft,
    /// Exits 7 at once.
    Quick,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Hold {
    /// `hold: false`: kept from the start.
    No,
    /// A held start, then `keep`, then the stream is dropped: the job survives.
    Keep,
    /// A held start whose stream is dropped unkept: the endpoint kills the job.
    Drop,
    /// A held start read to its final frame (the job is released first if it has a gate).
    ToEnd,
}

#[derive(Clone, Debug)]
enum Op {
    Start {
        s: usize,
        id: usize,
        kind: Kind,
        hold: Hold,
    },
    Keep {
        s: usize,
        id: usize,
        notify: bool,
    },
    Seen {
        s: usize,
        id: usize,
        off: u64,
    },
    Read {
        s: usize,
        id: usize,
        off: u64,
        max: u64,
    },
    Attach {
        s: usize,
        id: usize,
        off: u64,
    },
    Kill {
        s: usize,
        ids: Vec<usize>,
        remove: bool,
        grace_ms: u64,
    },
    Remove {
        s: usize,
        ids: Vec<usize>,
    },
    Wipe {
        s: usize,
        grace_ms: u64,
    },
    List {
        s: usize,
    },
    Release {
        s: usize,
        id: usize,
    },
    Garbage {
        s: usize,
        variant: usize,
    },
    /// `n` unheld starts of one id at the same moment.
    Race {
        s: usize,
        id: usize,
        n: usize,
    },
}

/// xorshift64*, so a seed reproduces a run exactly on any machine.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
    fn chance(&mut self, one_in: usize) -> bool {
        self.below(one_in) == 0
    }
}

fn gen_ops(seed: u64, n: usize) -> Vec<Op> {
    let mut r = Rng::new(seed);
    let mut ops = Vec::with_capacity(n);
    for _ in 0..n {
        // Most operations use the three main scopes and ids; one in five reaches for a lookalike.
        let s = if r.chance(8) {
            r.below(SCOPES.len())
        } else {
            r.below(3)
        };
        let pick_id = |r: &mut Rng| {
            if r.chance(5) {
                r.below(IDS.len())
            } else {
                r.below(3)
            }
        };
        let i = pick_id(&mut r);
        let op = match r.below(100) {
            0..=29 => Op::Start {
                s,
                id: i,
                kind: [
                    Kind::Long,
                    Kind::Stubborn,
                    Kind::Gate,
                    Kind::GateLeft,
                    Kind::Quick,
                ][r.below(5)],
                hold: [Hold::No, Hold::No, Hold::Keep, Hold::Drop, Hold::ToEnd][r.below(5)],
            },
            30..=35 => Op::Keep {
                s,
                id: i,
                notify: r.chance(2),
            },
            36..=38 => Op::Seen {
                s,
                id: i,
                off: [0, 3, 9, 1 << 40][r.below(4)],
            },
            39..=50 => Op::Read {
                s,
                id: i,
                off: [0, 0, 1, 5, 9, 4096, u64::MAX][r.below(7)],
                max: [0, 1, 4, 1 << 20, u64::MAX][r.below(5)],
            },
            51..=56 => Op::Attach {
                s,
                id: i,
                off: [0, 0, 3, 9, 4096][r.below(5)],
            },
            57..=66 => Op::Kill {
                s,
                ids: (0..1 + r.below(3)).map(|_| pick_id(&mut r)).collect(),
                remove: r.chance(2),
                grace_ms: [0, 20, 100, 300][r.below(4)],
            },
            67..=71 => Op::Remove {
                s,
                ids: (0..1 + r.below(3)).map(|_| pick_id(&mut r)).collect(),
            },
            72..=73 => Op::Wipe {
                s,
                grace_ms: [0, 50, 300][r.below(3)],
            },
            74..=77 => Op::List { s },
            78..=92 => Op::Release { s, id: i },
            93..=95 => Op::Race {
                s,
                id: i,
                n: 2 + r.below(3),
            },
            _ => Op::Garbage {
                s,
                variant: r.below(10),
            },
        };
        ops.push(op);
    }
    ops
}

// ---------------------------------------------------------------------------------------------
// the model
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Running,
    /// The shell exited with a status and left a process behind.
    ExitedLeft(i32),
    /// Ended by itself with this status.
    Ended(i32),
    /// Ended by a signal the endpoint sent.
    Killed,
}

impl Phase {
    fn state(self) -> &'static str {
        match self {
            Phase::Running => "running",
            Phase::ExitedLeft(_) => "exited",
            Phase::Ended(_) | Phase::Killed => "ended",
        }
    }
    fn is_ended(self) -> bool {
        matches!(self, Phase::Ended(_) | Phase::Killed)
    }
}

struct Job {
    kind: Kind,
    label: String,
    pids: PathBuf,
    gate: PathBuf,
    kept: bool,
    num: Option<u64>,
    kind_str: Option<String>,
    seen: u64,
    phase: Phase,
}

impl Job {
    /// Every byte this job's command will ever have written.
    fn full_output(&self) -> String {
        let l = &self.label;
        match self.kind {
            Kind::Long | Kind::Stubborn => format!("start-{l}\n"),
            Kind::Gate | Kind::GateLeft => match self.phase {
                Phase::Running => format!("start-{l}\n"),
                _ => format!("start-{l}\ndone-{l}\n"),
            },
            Kind::Quick => format!("q-{l}\n{}", "p".repeat(QUICK_PAD)),
        }
    }

    /// Whether output is complete, which is when no byte can still arrive.
    fn output_final(&self) -> bool {
        match self.kind {
            Kind::Quick => true,
            _ => self.phase.is_ended() || matches!(self.phase, Phase::ExitedLeft(_)),
        }
    }

    /// Pids the command has recorded.
    fn pids(&self) -> Vec<u32> {
        std::fs::read_to_string(&self.pids)
            .unwrap_or_default()
            .lines()
            .filter_map(|l| l.trim().parse().ok())
            .collect()
    }

    /// How many pids the command records before it does anything else.
    fn pid_lines(&self) -> usize {
        match self.kind {
            Kind::Gate | Kind::Quick => 1,
            Kind::Long | Kind::Stubborn | Kind::GateLeft => 2,
        }
    }
}

fn alive(pid: u32) -> bool {
    match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
        Ok(s) => s
            .rsplit_once(')')
            .and_then(|(_, rest)| rest.trim_start().chars().next())
            .is_some_and(|st| st != 'Z' && st != 'X'),
        Err(_) => false,
    }
}

macro_rules! ensure {
    ($c:expr, $($a:tt)*) => {
        if !$c {
            return Err(format!($($a)*));
        }
    };
}

// ---------------------------------------------------------------------------------------------
// a streaming client
// ---------------------------------------------------------------------------------------------

struct Stream {
    status: u16,
    body: hyper::body::Incoming,
    buf: Vec<u8>,
    conn: tokio::task::JoinHandle<()>,
    _send: hyper::client::conn::http1::SendRequest<Full<Bytes>>,
}

impl Drop for Stream {
    fn drop(&mut self) {
        self.conn.abort();
    }
}

impl Stream {
    async fn open(addr: SocketAddr, bearer: &str, body: &Value) -> Stream {
        let sock = tokio::net::TcpStream::connect(addr).await.unwrap();
        let (mut send, conn) = hyper::client::conn::http1::handshake(TokioIo::new(sock))
            .await
            .unwrap();
        let conn = tokio::spawn(async move {
            let _ = conn.await;
        });
        let req = hyper::Request::builder()
            .method("POST")
            .uri("/exec")
            .header("Host", "execd")
            .header("Authorization", format!("Bearer {bearer}"))
            .header("Content-Type", "application/json")
            .body(Full::new(Bytes::from(body.to_string())))
            .unwrap();
        let resp = send.send_request(req).await.unwrap();
        Stream {
            status: resp.status().as_u16(),
            body: resp.into_body(),
            buf: Vec::new(),
            conn,
            _send: send,
        }
    }

    /// The next frame (heartbeats skipped), or `None` at the end of the stream or after `within`.
    async fn next(&mut self, within: Duration) -> Option<Value> {
        let deadline = Instant::now() + within;
        loop {
            if let Some(nl) = self.buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = self.buf.drain(..=nl).collect();
                let v: Value = serde_json::from_slice(&line).unwrap_or(Value::Null);
                if v.as_object().is_some_and(serde_json::Map::is_empty) {
                    continue;
                }
                return Some(v);
            }
            let left = deadline.saturating_duration_since(Instant::now());
            match tokio::time::timeout(left, self.body.frame()).await {
                Ok(Some(Ok(frame))) => {
                    if let Ok(data) = frame.into_data() {
                        self.buf.extend_from_slice(&data);
                    }
                }
                _ => return None,
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// the callback receiver
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Callback {
    path: String,
    auth: String,
    body: String,
}

fn receiver() -> (SocketAddr, Arc<Mutex<Vec<Callback>>>) {
    let listener = beyond_test_support::ports::listener();
    listener.set_nonblocking(true).unwrap();
    let addr = listener.local_addr().unwrap();
    let got = Arc::new(Mutex::new(Vec::new()));
    let got2 = got.clone();
    tokio::spawn(async move {
        let listener = tokio::net::TcpListener::from_std(listener).unwrap();
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            let got = got2.clone();
            tokio::spawn(async move {
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                loop {
                    let Ok(n) = sock.read(&mut chunk).await else {
                        return;
                    };
                    if n == 0 {
                        return;
                    }
                    buf.extend_from_slice(&chunk[..n]);
                    let Some(split) = buf.windows(4).position(|w| w == b"\r\n\r\n") else {
                        continue;
                    };
                    let head = String::from_utf8_lossy(&buf[..split]).into_owned();
                    let len = head
                        .lines()
                        .find_map(|l| {
                            l.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                        })
                        .unwrap_or(0);
                    if buf.len() < split + 4 + len {
                        continue;
                    }
                    let path = head.split_whitespace().nth(1).unwrap_or("").to_string();
                    let auth = head
                        .lines()
                        .find_map(|l| {
                            l.strip_prefix("authorization: ")
                                .or_else(|| l.strip_prefix("Authorization: "))
                        })
                        .unwrap_or("")
                        .to_string();
                    let body =
                        String::from_utf8_lossy(&buf[split + 4..split + 4 + len]).into_owned();
                    got.lock().unwrap().push(Callback { path, auth, body });
                    let _ = sock
                        .write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                        )
                        .await;
                    return;
                }
            });
        }
    });
    (addr, got)
}

// ---------------------------------------------------------------------------------------------
// the world: a real execd, and the model beside it
// ---------------------------------------------------------------------------------------------

struct World {
    execd: Execd,
    addr: SocketAddr,
    bearer: String,
    jobs: HashMap<(usize, usize), Job>,
    scopes: BTreeSet<usize>,
    /// Every pids file of every job ever started: the shutdown check reads them all.
    all_pids: Vec<PathBuf>,
    started: u64,
    callbacks: Arc<Mutex<Vec<Callback>>>,
    recv_addr: SocketAddr,
    notified: HashSet<(usize, usize)>,
    /// How many operations had something to act on, by kind: a run whose kills all hit nothing
    /// proves nothing.
    stats: std::collections::BTreeMap<&'static str, usize>,
    /// Where execd spools job output, and what `/proc` showed of it when it was idle and new.
    jobs_base: tempfile::TempDir,
    baseline: Baseline,
}

impl World {
    /// Count what `op` is about to do, from the model's state.
    fn tally(&mut self, op: &Op) {
        let here = |w: &World, s: usize, i: usize| w.jobs.contains_key(&(s, i));
        let elsewhere =
            |w: &World, s: usize, i: usize| w.jobs.keys().any(|(sc, ii)| *ii == i && *sc != s);
        let name = match op {
            Op::Start { s, id: i, .. } => {
                if here(self, *s, *i) {
                    "start collides"
                } else {
                    "start"
                }
            }
            Op::Keep { s, id: i, .. } => {
                if here(self, *s, *i) {
                    "keep"
                } else if elsewhere(self, *s, *i) {
                    "keep of another scope's id"
                } else {
                    "keep of nothing"
                }
            }
            Op::Seen { s, id: i, .. } => {
                if here(self, *s, *i) {
                    "seen"
                } else {
                    "seen of nothing"
                }
            }
            Op::Read { s, id: i, .. } => {
                if here(self, *s, *i) {
                    "read"
                } else if elsewhere(self, *s, *i) {
                    "read of another scope's id"
                } else {
                    "read of nothing"
                }
            }
            Op::Attach { s, id: i, .. } => {
                if here(self, *s, *i) {
                    "attach"
                } else if elsewhere(self, *s, *i) {
                    "attach of another scope's id"
                } else {
                    "attach of nothing"
                }
            }
            Op::Kill { s, ids, .. } => {
                if ids.iter().any(|i| here(self, *s, *i)) {
                    "kill"
                } else if ids.iter().any(|i| elsewhere(self, *s, *i)) {
                    "kill of another scope's ids only"
                } else {
                    "kill of nothing"
                }
            }
            Op::Remove { s, ids } => {
                if ids.iter().any(|i| here(self, *s, *i)) {
                    "remove"
                } else if ids.iter().any(|i| elsewhere(self, *s, *i)) {
                    "remove of another scope's ids only"
                } else {
                    "remove of nothing"
                }
            }
            Op::Wipe { s, .. } => {
                if self.jobs.keys().any(|(sc, _)| sc == s) {
                    "wipe"
                } else if self.jobs.is_empty() {
                    "wipe of nothing"
                } else {
                    "wipe of an empty scope beside others"
                }
            }
            Op::List { .. } => "list",
            Op::Release { s, id: i } => {
                if self.jobs.get(&(*s, *i)).is_some_and(|j| {
                    matches!(j.kind, Kind::Gate | Kind::GateLeft) && j.phase == Phase::Running
                }) {
                    "release"
                } else {
                    "release of nothing"
                }
            }
            Op::Garbage { .. } => "garbage",
            Op::Race { .. } => "race",
        };
        *self.stats.entry(name).or_default() += 1;
    }

    fn new(seed: u64) -> Self {
        let bearer = format!("jobs-{seed}-{}", std::process::id());
        let jobs_base = tempfile::tempdir().unwrap();
        let execd = Execd::start(
            &[],
            &[
                ("EXECD_BEARER", &bearer),
                ("EXECD_JOBS_DIR", jobs_base.path().to_str().unwrap()),
            ],
        );
        let baseline = Baseline::take(&execd);
        let addr = SocketAddr::V4(execd.addr);
        let (recv_addr, callbacks) = receiver_blocking();
        Self {
            execd,
            addr,
            bearer,
            jobs: HashMap::new(),
            scopes: BTreeSet::new(),
            all_pids: Vec::new(),
            started: 0,
            callbacks,
            recv_addr,
            notified: HashSet::new(),
            stats: std::collections::BTreeMap::new(),
            jobs_base,
            baseline,
        }
    }

    fn dir(&self) -> &Path {
        self.execd.dir.path()
    }

    /// POST a job op for scope `s`; the request's own command is the harmless `true` the spec says.
    async fn op(&self, s: usize, mut job: Value) -> support::http::Answer {
        job["scope"] = json!(SCOPES[s]);
        self.post_raw(&json!({ "command": "true", "args": [], "job": job }))
            .await
    }

    async fn post_raw(&self, body: &Value) -> support::http::Answer {
        post_as(self.addr, &self.bearer, body).await
    }

    fn model_ids(&self, s: usize) -> BTreeSet<usize> {
        self.jobs
            .keys()
            .filter(|(sc, _)| *sc == s)
            .map(|(_, i)| *i)
            .collect()
    }

    // --- one operation ----------------------------------------------------------------------

    async fn apply(&mut self, op: &Op) -> Result<(), String> {
        self.tally(op);
        match op {
            Op::Start {
                s,
                id: i,
                kind,
                hold,
            } => self.start(*s, *i, *kind, *hold).await,
            Op::Keep { s, id: i, notify } => self.keep(*s, *i, *notify).await,
            Op::Seen { s, id: i, off } => {
                self.scopes.insert(*s);
                let a = self
                    .op(*s, json!({ "op": "seen", "id": id(*i), "offset": off }))
                    .await;
                match self.jobs.get_mut(&(*s, *i)) {
                    Some(j) => {
                        ensure!(
                            a.status == 200 && a.json() == json!({}),
                            "seen of a job: {} {}",
                            a.status,
                            a.json()
                        );
                        j.seen = *off;
                    }
                    None => ensure!(not_found(&a), "seen of no job: {} {}", a.status, a.json()),
                }
                Ok(())
            }
            Op::Read { s, id: i, off, max } => self.read(*s, *i, *off, *max).await,
            Op::Attach { s, id: i, off } => self.attach(*s, *i, *off).await,
            Op::Kill {
                s,
                ids,
                remove,
                grace_ms,
            } => self.kill(*s, ids, *remove, *grace_ms).await,
            Op::Remove { s, ids } => {
                self.scopes.insert(*s);
                let a = self.op(*s, json!({ "op": "remove", "ids": ids.iter().map(|i| id(*i)).collect::<Vec<_>>() })).await;
                ensure!(
                    a.status == 200 && a.json() == json!({}),
                    "remove: {} {}",
                    a.status,
                    a.json()
                );
                for i in ids {
                    if self.jobs.get(&(*s, *i)).is_some_and(|j| j.phase.is_ended()) {
                        self.jobs.remove(&(*s, *i));
                    }
                }
                Ok(())
            }
            Op::Wipe { s, grace_ms } => {
                self.scopes.insert(*s);
                let a = self
                    .op(*s, json!({ "op": "wipe", "grace_ms": grace_ms }))
                    .await;
                ensure!(
                    a.status == 200 && a.json() == json!({ "survivors": [] }),
                    "wipe: {} {}",
                    a.status,
                    a.json()
                );
                let mine: Vec<usize> = self.model_ids(*s).into_iter().collect();
                for i in &mine {
                    self.killed_gone(*s, *i)?;
                }
                for i in mine {
                    self.jobs.remove(&(*s, i));
                }
                Ok(())
            }
            Op::List { s } => {
                self.scopes.insert(*s);
                Ok(())
            }
            Op::Release { s, id: i } => {
                if let Some(j) = self.jobs.get_mut(&(*s, *i))
                    && matches!(j.kind, Kind::Gate | Kind::GateLeft)
                    && j.phase == Phase::Running
                {
                    std::fs::write(&j.gate, b"go").unwrap();
                    j.phase = if j.kind == Kind::Gate {
                        Phase::Ended(5)
                    } else {
                        Phase::ExitedLeft(6)
                    };
                }
                Ok(())
            }
            Op::Garbage { s, variant } => self.garbage(*s, *variant).await,
            Op::Race { s, id: i, n } => self.race(*s, *i, *n).await,
        }
    }

    async fn start(&mut self, s: usize, i: usize, kind: Kind, hold: Hold) -> Result<(), String> {
        self.scopes.insert(s);
        // What a held start can mean depends on what the command does.
        let hold = match (kind, hold) {
            (Kind::Long | Kind::Stubborn | Kind::GateLeft, Hold::ToEnd) => Hold::Keep,
            (Kind::Quick, Hold::Drop) => Hold::ToEnd,
            (_, h) => h,
        };
        self.started += 1;
        let n = self.started;
        let label = format!("L{n}s{s}i{i}");
        let pids = self.dir().join(format!("p-{n}"));
        let gate = self.dir().join(format!("g-{n}"));
        self.all_pids.push(pids.clone());
        let script = script(kind, &pids, &gate, &label);
        let exists = self.jobs.contains_key(&(s, i));
        let held = hold != Hold::No;
        let mut body = json!({ "command": "sh", "args": ["-c", script], "cwd": self.dir(),
            "job": { "op": "start", "scope": SCOPES[s], "id": id(i), "hold": held, "label": label } });
        let mut stream = None;
        if held {
            body["stream"] = json!(true);
            let mut st = Stream::open(self.addr, &self.bearer, &body).await;
            if exists {
                ensure!(
                    st.status == 409,
                    "a held start of an existing job answered {}",
                    st.status
                );
                tokio::time::sleep(Duration::from_millis(150)).await;
                ensure!(!pids.exists(), "the second start of one id ran its command");
                return Ok(());
            }
            ensure!(st.status == 200, "held start: status {}", st.status);
            let first = st.next(Duration::from_secs(5)).await;
            ensure!(
                first
                    .as_ref()
                    .is_some_and(|f| f["started"].as_i64().is_some_and(|p| p > 1)),
                "a held start's first frame is not {{started}}: {first:?}"
            );
            stream = Some(st);
        } else {
            let a = self.post_raw(&body).await;
            if exists {
                ensure!(
                    a.status == 409,
                    "a second start of one id answered {} {}",
                    a.status,
                    a.json()
                );
                tokio::time::sleep(Duration::from_millis(150)).await;
                ensure!(!pids.exists(), "the second start of one id ran its command");
                return Ok(());
            }
            ensure!(
                a.status == 200 && a.json()["started"].as_i64().is_some_and(|p| p > 1),
                "start: {} {}",
                a.status,
                a.json()
            );
        }
        let mut job = Job {
            kind,
            label,
            pids,
            gate,
            kept: !held,
            num: None,
            kind_str: None,
            seen: 0,
            phase: Phase::Running,
        };
        // The command has begun once it has said which processes it is.
        wait_for(
            || job.pids().len() >= job.pid_lines(),
            "the command never started",
        )
        .await?;
        match hold {
            Hold::No => {}
            Hold::Keep => {
                let a = self
                    .op(s, json!({ "op": "keep", "id": id(i), "num": 4, "kind": "yielded", "seen": 0 }))
                    .await;
                ensure!(
                    a.status == 200,
                    "keep after a held start: {} {}",
                    a.status,
                    a.json()
                );
                job.kept = true;
                job.num = Some(4);
                job.kind_str = Some("yielded".into());
                drop(stream);
            }
            Hold::Drop => {
                drop(stream);
                job.phase = Phase::Killed;
            }
            Hold::ToEnd => {
                let mut st = stream.take().unwrap();
                if job.kind == Kind::Gate {
                    std::fs::write(&job.gate, b"go").unwrap();
                }
                let mut out = Vec::new();
                let mut last = None;
                while let Some(f) = st.next(Duration::from_secs(10)).await {
                    if let Some(b) = f["stdout"].as_str() {
                        out.extend(base64::engine::general_purpose::STANDARD.decode(b).unwrap());
                    }
                    if f.get("exit_code").is_some()
                        || f.get("signal").is_some()
                        || f.get("error").is_some()
                    {
                        last = Some(f);
                        break;
                    }
                }
                let code = if job.kind == Kind::Gate { 5 } else { 7 };
                ensure!(
                    last == Some(json!({ "exit_code": code })),
                    "a held job's final frame: {last:?}"
                );
                let want = match job.kind {
                    Kind::Gate => format!("start-{0}\ndone-{0}\n", job.label),
                    _ => job.full_output(),
                };
                ensure!(
                    String::from_utf8_lossy(&out) == want,
                    "a held job streamed {:?}, wanted {want:?}",
                    String::from_utf8_lossy(&out)
                );
                job.phase = Phase::Ended(code);
            }
        }
        if kind == Kind::Quick && job.phase == Phase::Running {
            job.phase = Phase::Ended(7);
        }
        self.jobs.insert((s, i), job);
        // A quick job is ended before anything else may race it.
        if kind == Kind::Quick {
            self.check_all().await?;
        }
        Ok(())
    }

    async fn keep(&mut self, s: usize, i: usize, notify: bool) -> Result<(), String> {
        self.scopes.insert(s);
        let mut job = json!({ "op": "keep", "id": id(i), "num": 9, "kind": "intended", "seen": 2 });
        let token = format!("tk-{s}-{i}-{}", self.started);
        if notify {
            job["notify"] =
                json!({ "url": format!("http://{}/{s}/{i}/", self.recv_addr), "token": token });
        }
        let a = self.op(s, job).await;
        match self.jobs.get_mut(&(s, i)) {
            Some(j) => {
                ensure!(
                    a.status == 200 && a.json() == json!({}),
                    "keep of a job: {} {}",
                    a.status,
                    a.json()
                );
                j.kept = true;
                j.num = Some(9);
                j.kind_str = Some("intended".into());
                j.seen = 2;
                if notify {
                    self.notified.insert((s, i));
                    // The proof of reach goes out at keep.
                    let want = format!("/{s}/{i}/");
                    let recv = self.callbacks.clone();
                    wait_for(
                        || recv.lock().unwrap().iter().any(|c| c.path == want),
                        "no proof reached the receiver",
                    )
                    .await?;
                }
            }
            None => ensure!(not_found(&a), "keep of no job: {} {}", a.status, a.json()),
        }
        Ok(())
    }

    async fn read(&mut self, s: usize, i: usize, off: u64, max: u64) -> Result<(), String> {
        self.scopes.insert(s);
        let a = self
            .op(
                s,
                json!({ "op": "read", "id": id(i), "offset": off, "max": max }),
            )
            .await;
        ensure!(a.status == 200, "read: status {} {}", a.status, a.json());
        let r = a.json();
        let Some(j) = self.jobs.get(&(s, i)) else {
            ensure!(
                r == json!({ "state": "gone", "offset": 0, "end": 0, "data": "" }),
                "read of no job (another scope's id included): {r}"
            );
            return Ok(());
        };
        let full = j.full_output();
        let data = base64::engine::general_purpose::STANDARD
            .decode(r["data"].as_str().unwrap_or("!"))
            .map_err(|e| format!("read data is not base64: {e}"))?;
        let end = r["end"].as_u64().unwrap_or(u64::MAX);
        ensure!(
            end as usize <= full.len(),
            "read end {end} is past the output {:?} of this job: {r}",
            full
        );
        if j.output_final() && j.kind != Kind::Quick {
            // Settled, but a job that has only just ended may lag the model by a poll: checked below
            // by `check_all`; here only what could never be true.
        }
        if off <= end {
            ensure!(
                r["offset"].as_u64() == Some(off),
                "read offset {} for a request at {off}: {r}",
                r["offset"]
            );
            let want_len = (end - off).min(max) as usize;
            ensure!(
                data.len() == want_len,
                "read returned {} bytes, wanted {want_len}: {r}",
                data.len()
            );
            ensure!(
                data == full.as_bytes()[off as usize..off as usize + want_len],
                "read returned bytes that are not this job's output: {:?}",
                String::from_utf8_lossy(&data)
            );
        } else {
            ensure!(data.is_empty(), "read past the end returned data: {r}");
        }
        let st = r["state"].as_str().unwrap_or("");
        ensure!(
            ["running", "exited", "ended"].contains(&st),
            "read state {st:?}"
        );
        Ok(())
    }

    async fn attach(&mut self, s: usize, i: usize, off: u64) -> Result<(), String> {
        self.scopes.insert(s);
        let body = json!({ "command": "true", "stream": true,
            "job": { "op": "attach", "scope": SCOPES[s], "id": id(i), "offset": off } });
        let mut st = Stream::open(self.addr, &self.bearer, &body).await;
        let Some(j) = self.jobs.get(&(s, i)) else {
            ensure!(
                st.status == 404,
                "attach of no job (another scope's id included) answered {}",
                st.status
            );
            return Ok(());
        };
        ensure!(st.status == 200, "attach of a job answered {}", st.status);
        let full = j.full_output();
        let mut out = Vec::new();
        let mut finals = Vec::new();
        let settled = j.phase.is_ended();
        let within = if settled {
            Duration::from_secs(3)
        } else {
            Duration::from_millis(400)
        };
        let deadline = Instant::now() + within;
        while let Some(f) = st
            .next(deadline.saturating_duration_since(Instant::now()))
            .await
        {
            ensure!(
                f.get("gap").is_none(),
                "an attach reported a gap in a tiny output: {f}"
            );
            ensure!(
                f.get("error").is_none(),
                "an attach of a live job reported an error: {f}"
            );
            if let Some(b) = f["stdout"].as_str() {
                out.extend(base64::engine::general_purpose::STANDARD.decode(b).unwrap());
            } else if f.get("exit_code").is_some() || f.get("signal").is_some() {
                finals.push(f);
                break;
            } else if let Some(c) = f["shell_exit"].as_i64() {
                ensure!(
                    matches!(j.phase, Phase::ExitedLeft(x) if i64::from(x) == c)
                        || j.phase.is_ended(),
                    "shell_exit {c} for phase {:?}",
                    j.phase
                );
            }
        }
        let want = full.as_bytes().get(off as usize..).unwrap_or(&[]);
        ensure!(
            want.starts_with(&out),
            "an attach streamed {:?}, which is not this job's output from {off} ({:?})",
            String::from_utf8_lossy(&out),
            full
        );
        if settled && j.phase != Phase::Killed {
            ensure!(
                out == want,
                "an attach of an ended job streamed {:?}, wanted {:?}",
                String::from_utf8_lossy(&out),
                String::from_utf8_lossy(want)
            );
        }
        if settled {
            ensure!(
                finals.len() == 1,
                "an attach of an ended job sent no final frame: {finals:?}"
            );
        } else {
            ensure!(
                finals.is_empty(),
                "an attach of a live job ended: {finals:?}"
            );
        }
        Ok(())
    }

    async fn kill(
        &mut self,
        s: usize,
        ids: &[usize],
        remove: bool,
        grace_ms: u64,
    ) -> Result<(), String> {
        self.scopes.insert(s);
        let named: BTreeSet<usize> = ids.iter().copied().collect();
        // A grace this long is only safe where `SIGTERM` is enough.
        let stubborn = named.iter().any(|i| {
            self.jobs
                .get(&(s, *i))
                .is_some_and(|j| j.kind == Kind::Stubborn && !j.phase.is_ended())
        });
        let grace = if stubborn {
            grace_ms
        } else {
            grace_ms.max(1) * 1000
        };
        let started = Instant::now();
        let a = self
            .op(s, json!({ "op": "kill", "ids": ids.iter().map(|i| id(*i)).collect::<Vec<_>>(), "remove": remove, "grace_ms": grace }))
            .await;
        ensure!(a.status == 200, "kill: {} {}", a.status, a.json());
        let r = a.json();
        ensure!(r["survivors"] == json!([]), "kill left survivors: {r}");
        ensure!(
            started.elapsed() < Duration::from_millis(grace_ms.max(1) + 1000 + 3000),
            "kill took {:?} with grace {grace_ms}",
            started.elapsed()
        );
        for i in &named {
            if self.jobs.contains_key(&(s, *i)) {
                self.killed_gone(s, *i)?;
            }
        }
        if remove {
            for i in &named {
                self.jobs.remove(&(s, *i));
            }
        }
        Ok(())
    }

    /// The model has the job dead (a kill or wipe reached it): its processes must be gone now, and
    /// its phase becomes the ended one.
    fn killed_gone(&mut self, s: usize, i: usize) -> Result<(), String> {
        let j = self.jobs.get_mut(&(s, i)).unwrap();
        match j.phase {
            Phase::Running => j.phase = Phase::Killed,
            Phase::ExitedLeft(c) => j.phase = Phase::Ended(c),
            Phase::Ended(_) | Phase::Killed => {}
        }
        for p in j.pids() {
            ensure!(
                !alive(p),
                "process {p} of the killed job {} (scope {s}) is still alive after the kill answered",
                j.label
            );
        }
        Ok(())
    }

    /// `n` starts of one id at once: exactly one wins (or none, if the id is taken), and only the
    /// winner's command ever runs.
    async fn race(&mut self, s: usize, i: usize, n: usize) -> Result<(), String> {
        self.scopes.insert(s);
        let exists = self.jobs.contains_key(&(s, i));
        let mut tries = Vec::new();
        for _ in 0..n {
            self.started += 1;
            let label = format!("L{}s{s}i{i}", self.started);
            let pids = self.dir().join(format!("p-{}", self.started));
            let gate = self.dir().join(format!("g-{}", self.started));
            self.all_pids.push(pids.clone());
            let body = json!({ "command": "sh", "args": ["-c", script(Kind::Long, &pids, &gate, &label)],
                "cwd": self.dir(),
                "job": { "op": "start", "scope": SCOPES[s], "id": id(i), "hold": false, "label": label } });
            tries.push((label, pids, gate, body));
        }
        let mut set = tokio::task::JoinSet::new();
        for (k, (_, _, _, body)) in tries.iter().enumerate() {
            let (addr, bearer, body) = (self.addr, self.bearer.clone(), body.clone());
            set.spawn(async move { (k, post_as(addr, &bearer, &body).await.status) });
        }
        let mut won = Vec::new();
        while let Some(r) = set.join_next().await {
            let (k, status) = r.map_err(|e| e.to_string())?;
            match status {
                200 => won.push(k),
                409 => {}
                other => return Err(format!("a raced start answered {other}")),
            }
        }
        if exists {
            ensure!(won.is_empty(), "a start of a taken id won");
        } else {
            ensure!(
                won.len() == 1,
                "{} of {n} racing starts of one id won, wanted exactly one",
                won.len()
            );
        }
        tokio::time::sleep(Duration::from_millis(150)).await;
        for (k, (_, pids, _, _)) in tries.iter().enumerate() {
            if !won.contains(&k) {
                ensure!(
                    !pids.exists(),
                    "the command of a start that lost the race ran"
                );
            }
        }
        if let Some(&k) = won.first() {
            let (label, pids, gate, _) = tries.swap_remove(k);
            let job = Job {
                kind: Kind::Long,
                label,
                pids,
                gate,
                kept: true,
                num: None,
                kind_str: None,
                seen: 0,
                phase: Phase::Running,
            };
            wait_for(
                || job.pids().len() >= job.pid_lines(),
                "the winner never started",
            )
            .await?;
            self.jobs.insert((s, i), job);
        }
        Ok(())
    }

    async fn garbage(&mut self, s: usize, variant: usize) -> Result<(), String> {
        self.scopes.insert(s);
        let scope = SCOPES[s];
        let (body, want): (Value, Option<u16>) = match variant {
            0 => (
                json!({ "command": "true", "job": { "op": "frobnicate", "scope": scope } }),
                Some(400),
            ),
            1 => (
                json!({ "command": "true", "job": { "op": "read", "scope": scope, "id": "a1", "offset": -1 } }),
                Some(400),
            ),
            2 => (
                json!({ "command": "true", "job": { "op": "read", "scope": scope, "id": "a1", "max": 1.5 } }),
                Some(400),
            ),
            3 => (
                json!({ "command": "true", "job": { "op": "kill", "scope": scope, "ids": "a1" } }),
                Some(400),
            ),
            4 => (
                json!({ "command": "true", "job": { "op": "keep", "scope": scope, "id": "a1", "notify": { "url": "ftp://x/", "token": "t" } } }),
                Some(400),
            ),
            5 => (
                json!({ "command": "true", "job": { "op": "keep", "scope": scope, "id": "b2", "notify": { "url": "https://example.com/", "token": "not a token!" } } }),
                Some(400),
            ),
            6 => (
                json!({ "command": "true", "job": { "op": 5, "scope": scope } }),
                Some(400),
            ),
            7 => (json!({ "command": "true", "job": [] }), None),
            8 => (
                json!({ "command": "true", "job": { "op": "kill", "scope": scope, "ids": (0..5000).map(|n| format!("n{n}")).collect::<Vec<_>>(), "grace_ms": 0 } }),
                Some(200),
            ),
            _ => (
                json!({ "command": "true", "job": { "scope": scope } }),
                Some(400),
            ),
        };
        let a = self.post_raw(&body).await;
        if let Some(w) = want {
            ensure!(
                a.status == w,
                "garbage variant {variant}: status {} {}, wanted {w}",
                a.status,
                a.json()
            );
        }
        // An error says nothing of any job: no label, no id of another scope.
        if a.status >= 400 {
            let text = String::from_utf8_lossy(&a.body).into_owned();
            ensure!(
                !text.contains("tk-") && !text.contains(&self.bearer),
                "an error carried a secret: {text}"
            );
            for j in self.jobs.values() {
                ensure!(!text.contains(&j.label), "an error named a job: {text}");
            }
        }
        Ok(())
    }

    // --- the invariants ---------------------------------------------------------------------

    /// All of them, retried for a few seconds: the endpoint notices a job's end by polling, so the
    /// model's state is reached a moment after the operation that caused it.
    async fn check_all(&mut self) -> Result<(), String> {
        let deadline = Instant::now()
            + Duration::from_millis(SETTLE_MS.load(std::sync::atomic::Ordering::Relaxed));
        loop {
            match self.check_once().await {
                Ok(()) => return Ok(()),
                Err(e) if Instant::now() > deadline => return Err(e),
                Err(_) => tokio::time::sleep(Duration::from_millis(40)).await,
            }
        }
    }

    async fn check_once(&self) -> Result<(), String> {
        let log = self.execd.log();
        ensure!(
            !log.contains("panicked"),
            "execd panicked:\n{}",
            log.lines()
                .filter(|l| l.contains("panicked"))
                .collect::<Vec<_>>()
                .join("\n")
        );
        // Every scope ever used, and one nobody used.
        let mut scopes: Vec<usize> = self.scopes.iter().copied().collect();
        scopes.push(usize::MAX);
        for s in scopes {
            let a = if s == usize::MAX {
                self.post_raw(
                    &json!({ "command": "true", "job": { "op": "list", "scope": "never-used" } }),
                )
                .await
            } else {
                self.op(s, json!({ "op": "list" })).await
            };
            ensure!(a.status == 200, "list answered {} {}", a.status, a.json());
            let list = a.json()["jobs"]
                .as_array()
                .cloned()
                .ok_or("list has no jobs array")?;
            let model = if s == usize::MAX {
                BTreeSet::new()
            } else {
                self.model_ids(s)
            };
            let got: Vec<String> = list
                .iter()
                .map(|e| e["id"].as_str().unwrap_or("?").to_string())
                .collect();
            let want: Vec<String> = model.iter().map(|i| id(*i)).collect();
            let (mut g, mut w) = (got.clone(), want.clone());
            g.sort();
            w.sort();
            ensure!(
                g == w,
                "scope {s} lists {:?}, the model has {:?}",
                short(&g),
                short(&w)
            );
            for e in &list {
                let i = IDS
                    .iter()
                    .position(|x| {
                        *x == e["id"].as_str().unwrap_or("?")
                            || (*x == "LONG" && e["id"].as_str().is_some_and(|v| v.len() == 4096))
                    })
                    .unwrap();
                let j = &self.jobs[&(s, i)];
                ensure!(
                    e["label"] == json!(j.label),
                    "scope {s} id {i}: label {} (another job's?), wanted {}",
                    e["label"],
                    j.label
                );
                ensure!(
                    e["kept"] == json!(j.kept),
                    "scope {s} id {i} ({}): kept {}, wanted {}",
                    j.label,
                    e["kept"],
                    j.kept
                );
                ensure!(
                    e["state"] == json!(j.phase.state()),
                    "scope {s} id {i} ({}): state {}, wanted {:?}",
                    j.label,
                    e["state"],
                    j.phase
                );
                if j.num.is_some() {
                    ensure!(
                        e["num"] == json!(j.num) && e["kind"] == json!(j.kind_str),
                        "scope {s} id {i}: num/kind {} {}",
                        e["num"],
                        e["kind"]
                    );
                }
                ensure!(
                    e["seen"] == json!(j.seen),
                    "scope {s} id {i} ({}): seen {}, wanted {}",
                    j.label,
                    e["seen"],
                    j.seen
                );
                match j.phase {
                    Phase::Running => {
                        ensure!(e["code"].is_null(), "running job has code {}", e["code"])
                    }
                    Phase::ExitedLeft(c) | Phase::Ended(c) => ensure!(
                        e["code"] == json!(c),
                        "scope {s} id {i} ({}): code {}, wanted {c}",
                        j.label,
                        e["code"]
                    ),
                    Phase::Killed => ensure!(
                        [137, 143].contains(&e["code"].as_i64().unwrap_or(0)),
                        "killed job code {}",
                        e["code"]
                    ),
                }
                ensure!(e["started"].as_u64().unwrap_or(0) > 0, "started missing");
                ensure!(
                    (e["ended"].as_u64().unwrap_or(0) > 0) == j.phase.is_ended(),
                    "ended {} for {:?}",
                    e["ended"],
                    j.phase
                );
            }
        }
        // The processes.
        for ((s, i), j) in &self.jobs {
            let pids = j.pids();
            let (shell, rest) = pids
                .split_first()
                .map_or((None, &[][..]), |(a, b)| (Some(*a), b));
            match j.phase {
                Phase::Running => {
                    for p in &pids {
                        ensure!(
                            alive(*p),
                            "process {p} of the running job {} (scope {s}, id {i}) is gone: something killed it",
                            j.label
                        );
                    }
                }
                Phase::ExitedLeft(_) => {
                    ensure!(
                        shell.is_none_or(|p| !alive(p)),
                        "the shell of an exited job lives"
                    );
                    for p in rest {
                        ensure!(
                            alive(*p),
                            "the leftover {p} of {} (scope {s}) is gone",
                            j.label
                        );
                    }
                }
                Phase::Ended(_) | Phase::Killed => {
                    for p in &pids {
                        ensure!(
                            !alive(*p),
                            "process {p} of the ended job {} (scope {s}) is alive",
                            j.label
                        );
                    }
                }
            }
        }
        // Callbacks: only for a job kept with a notify, with that job's own token and names.
        for c in self.callbacks.lock().unwrap().iter() {
            let parts: Vec<&str> = c.path.trim_matches('/').split('/').collect();
            let (Some(s), Some(i)) = (
                parts.first().and_then(|p| p.parse::<usize>().ok()),
                parts.get(1).and_then(|p| p.parse::<usize>().ok()),
            ) else {
                return Err(format!("a callback to an unknown path {:?}", c.path));
            };
            ensure!(
                self.notified.contains(&(s, i)),
                "a callback for ({s},{i}), which was never kept with a notify"
            );
            ensure!(
                c.auth.starts_with(&format!("Bearer tk-{s}-{i}-")),
                "a callback to /{s}/{i}/ carried another job's token: {:?}",
                c.auth
            );
            ensure!(
                c.body == exec_notify::body(SCOPES[s], &id(i)),
                "a callback to /{s}/{i}/ named another job: {}",
                c.body
            );
        }
        Ok(())
    }

    /// Every scope wiped and every connection closed: execd is back at what it showed when new
    /// (same descriptors and threads, no zombie child, an empty jobs directory).
    async fn baseline_check(&self) -> Result<(), String> {
        for s in 0..SCOPES.len() {
            let a = self.op(s, json!({ "op": "wipe", "grace_ms": 200 })).await;
            ensure!(a.status == 200, "wipe: {} {}", a.status, a.json());
            ensure!(a.json()["survivors"] == json!([]), "wipe left survivors");
        }
        let base = self.jobs_base.path().to_owned();
        let before = &self.baseline;
        let settle = Duration::from_secs(15);
        // The check sleeps; keep it off the runtime's threads.
        tokio::task::block_in_place(|| before.restored(&base, settle))
            .map_err(|e| format!("execd is not back at its starting state: {e}"))
    }

    /// `SIGTERM`: every process of every job ever started must be gone, however it ended.
    async fn shutdown_check(&mut self) -> Result<(), String> {
        let pid = nix::unistd::Pid::from_raw(i32::try_from(self.execd.pid()).unwrap());
        nix::sys::signal::kill(pid, nix::sys::signal::Signal::SIGTERM)
            .map_err(|e| e.to_string())?;
        let deadline = Instant::now() + Duration::from_secs(10);
        while self.execd.child.try_wait().unwrap().is_none() {
            ensure!(Instant::now() < deadline, "execd did not exit on SIGTERM");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let mut left = Vec::new();
            for f in &self.all_pids {
                for l in std::fs::read_to_string(f).unwrap_or_default().lines() {
                    if let Ok(p) = l.trim().parse::<u32>()
                        && alive(p)
                    {
                        left.push((f.file_name().unwrap().to_string_lossy().into_owned(), p));
                    }
                }
            }
            if left.is_empty() {
                break;
            }
            ensure!(
                Instant::now() < deadline,
                "processes of jobs survived the shutdown: {left:?}"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let log = self.execd.log();
        ensure!(
            !log.contains("panicked"),
            "execd panicked during shutdown:\n{log}"
        );
        Ok(())
    }
}

/// POST `body` with the bearer. The client helper panics when the connection closes with no answer
/// (the server died mid-request): that is an answer of its own here, status 0.
async fn post_as(addr: SocketAddr, bearer: &str, body: &Value) -> support::http::Answer {
    let auth = format!("Bearer {bearer}");
    let body = body.clone();
    let sent =
        tokio::spawn(async move { post(addr, &[("Authorization", auth.as_str())], &body).await })
            .await;
    sent.unwrap_or_else(|_| support::http::Answer {
        status: 0,
        headers: Vec::new(),
        body: b"no answer: the connection closed".to_vec(),
    })
}

fn short(v: &[String]) -> Vec<String> {
    v.iter().map(|s| s.chars().take(12).collect()).collect()
}

fn not_found(a: &support::http::Answer) -> bool {
    a.status == 404 && a.json() == json!({ "error": "no such job" })
}

/// What a quick job writes after its label: more than one 64 KiB chunk, so its output reaches the
/// jobs directory and the end-of-run check of that directory has something to find.
const QUICK_PAD: usize = 70_000;

fn script(kind: Kind, pids: &Path, gate: &Path, l: &str) -> String {
    let (p, g) = (pids.display(), gate.display());
    match kind {
        Kind::Long => {
            format!("echo $$ >> '{p}'; sleep 600 & echo $! >> '{p}'; echo start-{l}; wait")
        }
        Kind::Stubborn => format!(
            "trap '' TERM; echo $$ >> '{p}'; sleep 600 & echo $! >> '{p}'; echo start-{l}; wait"
        ),
        Kind::Gate => format!(
            "echo $$ >> '{p}'; echo start-{l}; while [ ! -e '{g}' ]; do sleep 0.05; done; echo done-{l}; exit 5"
        ),
        Kind::GateLeft => format!(
            "echo $$ >> '{p}'; sleep 600 & echo $! >> '{p}'; echo start-{l}; while [ ! -e '{g}' ]; do sleep 0.05; done; echo done-{l}; exit 6"
        ),
        Kind::Quick => format!(
            "echo $$ >> '{p}'; echo q-{l}; head -c {QUICK_PAD} /dev/zero | tr '\\0' p; exit 7"
        ),
    }
}

async fn wait_for(mut cond: impl FnMut() -> bool, what: &str) -> Result<(), String> {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !cond() {
        ensure!(Instant::now() < deadline, "{what}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    Ok(())
}

/// The receiver needs a runtime to spawn on; this is called from inside one.
fn receiver_blocking() -> (SocketAddr, Arc<Mutex<Vec<Callback>>>) {
    receiver()
}

// ---------------------------------------------------------------------------------------------
// running, shrinking, reporting
// ---------------------------------------------------------------------------------------------

/// Run `ops` against a fresh execd. `Err` is the first invariant that broke, and where.
async fn run_case(seed: u64, ops: &[Op]) -> Result<(), String> {
    let mut w = World::new(seed);
    let mut broke = None;
    for (n, op) in ops.iter().enumerate() {
        if let Err(e) = w.apply(op).await {
            broke = Some(format!("op {n} {op:?}: {e}"));
            break;
        }
        if let Err(e) = w.check_all().await {
            broke = Some(format!("after op {n} {op:?}: {e}"));
            break;
        }
    }
    if std::env::var_os("EXECD_JOBS_STATS").is_some() {
        println!("seed {seed}: {:?}", w.stats);
    }
    let returned = if broke.is_none() {
        w.baseline_check().await
    } else {
        Ok(())
    };
    // However the run ended, execd is stopped the polite way, so a failed case leaves no job behind.
    let down = w
        .shutdown_check()
        .await
        .map_err(|e| format!("at shutdown: {e}"));
    match broke {
        Some(e) => Err(e),
        None => returned.and(down),
    }
}

/// How long a check keeps retrying before it reports a violation: long enough for the endpoint's
/// polling to catch up, shorter while shrinking (every candidate that fails pays it once).
static SETTLE_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(6000);

/// Delete operations while the run still fails: chunks first, then singles.
async fn shrink(seed: u64, mut ops: Vec<Op>, budget: Duration) -> Vec<Op> {
    SETTLE_MS.store(1500, std::sync::atomic::Ordering::Relaxed);
    let deadline = Instant::now() + budget;
    let mut chunk = ops.len() / 2;
    while chunk >= 1 && Instant::now() < deadline {
        let mut at = 0;
        while at < ops.len() && Instant::now() < deadline {
            let mut cand = ops.clone();
            cand.drain(at..(at + chunk).min(ops.len()));
            if !cand.is_empty() && run_case(seed, &cand).await.is_err() {
                ops = cand;
            } else {
                at += chunk;
            }
        }
        chunk /= 2;
    }
    ops
}

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_job_table_keeps_its_scopes_and_its_kills() {
    let ops_n = env_u64("EXECD_JOBS_OPS").unwrap_or(80) as usize;
    let secs = env_u64("EXECD_JOBS_SECS");
    let seeds: Vec<u64> = match (env_u64("EXECD_JOBS_SEED"), env_u64("EXECD_JOBS_CASES")) {
        (Some(s), _) => vec![s],
        (None, Some(n)) => (1..=n).collect(),
        (None, None) if secs.is_some() => Vec::new(),
        (None, None) => (1..=6).collect(),
    };
    let started = Instant::now();
    let mut failures = Vec::new();
    let mut ran = 0;
    let mut next_seed = env_u64("EXECD_JOBS_FIRST_SEED").unwrap_or(1000);
    let mut queue: std::collections::VecDeque<u64> = seeds.into();
    // Cases run three at a time, each with an execd of its own.
    loop {
        let mut batch = Vec::new();
        for _ in 0..3 {
            let seed = match queue.pop_front() {
                Some(s) => s,
                None if secs.is_some_and(|s| started.elapsed() < Duration::from_secs(s)) => {
                    next_seed += 1;
                    next_seed
                }
                None => break,
            };
            batch.push(tokio::spawn(async move {
                let ops = gen_ops(seed, ops_n);
                (seed, ops.clone(), run_case(seed, &ops).await)
            }));
        }
        if batch.is_empty() {
            break;
        }
        for h in batch {
            let (seed, ops, res) = h.await.unwrap();
            ran += 1;
            if let Err(e) = res {
                eprintln!("seed {seed} failed: {e}\nshrinking...");
                let small = shrink(seed, ops, Duration::from_secs(90)).await;
                let again = run_case(seed, &small)
                    .await
                    .err()
                    .unwrap_or_else(|| "(did not fail again)".into());
                failures.push(format!(
                    "seed {seed} (replay: EXECD_JOBS_SEED={seed} EXECD_JOBS_OPS={ops_n})\nfirst failure: {e}\nshrunk to {} ops: {small:#?}\nfailure of the shrunk list: {again}",
                    small.len()
                ));
            }
        }
    }
    println!(
        "{ran} cases of {ops_n} operations in {:?}",
        started.elapsed()
    );
    assert!(failures.is_empty(), "\n{}\n", failures.join("\n\n"));
}
