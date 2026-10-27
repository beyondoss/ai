//! The end-to-end fleet for the job notify proof suites: real `serve --service` replicas, a fleet
//! front that routes by session id over a hash ring (retrying and walking on `503`, attaching
//! nothing), the reference endpoint behind a proxy that counts every byte crossing a
//! replica-to-endpoint connection, and a model that answers by each request's last message.
//!
//! Every suite runs against the endpoint in [`Mode::Native`]: it keeps jobs itself and calls back.

#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use beyond_agent::exec_endpoint::jobs::{JobId, JobScope};
use beyond_agent::notify_token::NotifyKey;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use super::exec_mock::{MockOptions, StreamEnd};
use super::service::{Options, Peer, Service};
use super::{TestWs, turn_text, turn_tool_use, ws_connect_with_headers, ws_send};

pub const TENANT: &str = "00tenant1";

/// How the sandbox's endpoint is set up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Native,
}

impl Mode {
    pub const ALL: [Mode; 1] = [Mode::Native];

    /// The endpoint for this mode.
    pub fn mock(self, _notify: bool) -> MockOptions {
        MockOptions {
            // A heartbeat often enough that a held stream is plainly not silent.
            heartbeat: Duration::from_secs(1),
            ..MockOptions::default()
        }
    }

    /// Whether a kept job is let go of (`false`: held to its end, as today).
    pub fn releases(self) -> bool {
        true
    }

    pub fn name(self) -> &'static str {
        match self {
            Mode::Native => "native",
        }
    }
}

// ---- the model

/// What the model is asked to start: a prompt `RUN<<command>>` gets a `bash` call starting it in
/// the background; anything else, text.
pub fn run_prompt(command: &str) -> String {
    format!("RUN<<{command}>>")
}

static TOOL_IDS: AtomicUsize = AtomicUsize::new(1);

/// The model: a reply by the request's last message alone, so any number of sessions can share it.
fn answer(raw: &str) -> String {
    let last = last_message_text(raw);
    if let Some(cmd) = last
        .split_once("RUN<<")
        .and_then(|(_, rest)| rest.split_once(">>"))
        .map(|(cmd, _)| cmd.to_string())
    {
        let id = format!("tu{}", TOOL_IDS.fetch_add(1, Ordering::Relaxed));
        return turn_tool_use(
            &id,
            "bash",
            &json!({"command": cmd, "run_in_background": true}).to_string(),
        );
    }
    turn_text("ok")
}

/// The texts of a model request's last message, joined.
pub fn last_message_text(raw: &str) -> String {
    let body = super::body_json(raw);
    let Some(last) = body["messages"].as_array().and_then(|m| m.last()) else {
        return String::new();
    };
    texts(&last["content"]).join("\n")
}

fn texts(content: &Value) -> Vec<String> {
    match content {
        Value::String(s) => vec![s.clone()],
        Value::Array(blocks) => blocks
            .iter()
            .flat_map(|b| {
                let mut out = Vec::new();
                if let Some(t) = b["text"].as_str() {
                    out.push(t.to_string());
                }
                if b.get("content").is_some() {
                    out.extend(texts(&b["content"]));
                }
                out
            })
            .collect(),
        _ => Vec::new(),
    }
}

// ---- the byte meter in front of the endpoint

/// A TCP proxy in front of the endpoint that counts every byte crossing it, either way, with the
/// moment it crossed. Replicas reach the endpoint only through it (the grant's `exec_url`).
#[derive(Clone)]
pub struct Meter {
    pub url: String,
    chunks: Arc<Mutex<Vec<(Instant, usize)>>>,
    down: Arc<AtomicBool>,
    pumps: Arc<Mutex<Vec<tokio::task::AbortHandle>>>,
    opened: Arc<AtomicUsize>,
}

impl Meter {
    pub async fn start(target_url: &str) -> Self {
        let listener = beyond_test_support::ports::tokio_listener().await;
        let addr = listener.local_addr().unwrap();
        let rest = target_url.trim_start_matches("http://");
        let (target, path) = rest.split_once('/').unwrap();
        let target = target.to_string();
        let meter = Self {
            url: format!("http://{addr}/{path}"),
            chunks: Arc::default(),
            down: Arc::default(),
            pumps: Arc::default(),
            opened: Arc::default(),
        };
        let m = meter.clone();
        tokio::spawn(async move {
            while let Ok((inbound, _)) = listener.accept().await {
                if m.down.load(Ordering::SeqCst) {
                    drop(inbound);
                    continue;
                }
                m.opened.fetch_add(1, Ordering::SeqCst);
                let (m2, target) = (m.clone(), target.clone());
                let task = tokio::spawn(async move {
                    let Ok(out) = TcpStream::connect(&target).await else {
                        return;
                    };
                    let (ri, wi) = inbound.into_split();
                    let (ro, wo) = out.into_split();
                    // Both directions in this one task, so cutting it cuts the connection.
                    tokio::join!(
                        pump(ri, wo, m2.chunks.clone()),
                        pump(ro, wi, m2.chunks.clone())
                    );
                });
                m.pumps.lock().unwrap().push(task.abort_handle());
            }
        });
        meter
    }

    /// Bytes that crossed at or after `since`.
    pub fn bytes_since(&self, since: Instant) -> usize {
        self.chunks
            .lock()
            .unwrap()
            .iter()
            .filter(|(t, _)| *t >= since)
            .map(|(_, n)| n)
            .sum()
    }

    /// Every crossing, with its moment.
    pub fn chunks(&self) -> Vec<(Instant, usize)> {
        self.chunks.lock().unwrap().clone()
    }

    /// Connections opened through the meter so far.
    pub fn opened(&self) -> usize {
        self.opened.load(Ordering::SeqCst)
    }

    /// The sandbox dies: every open connection is cut and every new one refused.
    pub fn go_down(&self) {
        self.down.store(true, Ordering::SeqCst);
        for p in self.pumps.lock().unwrap().drain(..) {
            p.abort();
        }
    }
}

async fn pump(
    mut from: tokio::net::tcp::OwnedReadHalf,
    mut to: tokio::net::tcp::OwnedWriteHalf,
    chunks: Arc<Mutex<Vec<(Instant, usize)>>>,
) {
    let mut buf = vec![0u8; 16 * 1024];
    loop {
        match from.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                chunks.lock().unwrap().push((Instant::now(), n));
                if to.write_all(&buf[..n]).await.is_err() {
                    break;
                }
            }
        }
    }
    let _ = to.shutdown().await;
}

// ---- the fleet's front

/// What the front does with a callback before routing it: route it, answer it a status itself
/// (the path refusing it), or drop the connection (the path losing it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Route,
    /// Route it after holding it this long (a slow path).
    Delay(Duration),
    Answer(u16),
    Drop,
}

/// A callback's fate, by what the front knows of it: its session, its job id, and how many
/// callbacks for that job the front has seen (`nth`, 1 for the first, which is the job's proof).
pub type Policy = Arc<dyn Fn(&SeenCallback) -> Verdict + Send + Sync>;

/// One callback the front saw.
#[derive(Clone, Debug)]
pub struct SeenCallback {
    pub at: Instant,
    pub session: String,
    pub job: String,
    /// Its count among the job's callbacks at the front, from 1.
    pub nth: usize,
    /// What the sandbox was answered (`0`: the connection was dropped).
    pub status: u16,
    /// Whether it reached a replica (`false`: the front answered or dropped it itself).
    pub routed: bool,
    /// When the front had its final answer.
    pub answered: Instant,
    /// Each replica slot tried, with its answer (`0`: refused the connection).
    pub tried: Vec<(usize, u16)>,
}

/// The fleet's front: classifies nothing beyond what routing needs, and routes every request for
/// a session id (a client's WebSocket and a sandbox's callback alike) over a hash ring of replica
/// slots: the hashed slot first, once more on `503`, then the rest of the ring in its order, all
/// within 25 s. It attaches, mints and checks nothing.
#[derive(Clone)]
pub struct Front {
    pub origin: String,
    pub port: u16,
    slots: Arc<Mutex<Vec<Option<u16>>>>,
    policy: Arc<Mutex<Option<Policy>>>,
    seen: Arc<Mutex<Vec<SeenCallback>>>,
    per_job: Arc<Mutex<HashMap<String, usize>>>,
}

impl Front {
    pub async fn bind(slots: usize) -> (Self, tokio::net::TcpListener) {
        let listener = beyond_test_support::ports::tokio_listener().await;
        let addr = listener.local_addr().unwrap();
        (
            Self {
                origin: format!("http://{addr}"),
                port: addr.port(),
                slots: Arc::new(Mutex::new(vec![None; slots])),
                policy: Arc::default(),
                seen: Arc::default(),
                per_job: Arc::default(),
            },
            listener,
        )
    }

    pub fn set_slot(&self, slot: usize, port: Option<u16>) {
        self.slots.lock().unwrap()[slot] = port;
    }

    pub fn set_policy(&self, policy: Option<Policy>) {
        *self.policy.lock().unwrap() = policy;
    }

    pub fn seen(&self) -> Vec<SeenCallback> {
        self.seen.lock().unwrap().clone()
    }

    /// The ring's order for a session id: rendezvous hashing over the slots.
    pub fn order(&self, session: &str) -> Vec<usize> {
        let n = self.slots.lock().unwrap().len();
        ring_order(session, n)
    }

    pub fn serve(&self, listener: tokio::net::TcpListener) {
        let front = self.clone();
        tokio::spawn(async move {
            while let Ok((inbound, _)) = listener.accept().await {
                let front = front.clone();
                tokio::spawn(async move {
                    // E4's 25 s bounds the routing; a WebSocket it opened then lives on.
                    let routed =
                        tokio::time::timeout(Duration::from_secs(25), front.route(inbound)).await;
                    if let Ok(Some((mut inbound, mut out))) = routed {
                        let _ = tokio::io::copy_bidirectional(&mut inbound, &mut out).await;
                    }
                });
            }
        });
    }

    /// Route one request; the two ends to join once a replica has answered.
    async fn route(&self, mut inbound: TcpStream) -> Option<(TcpStream, TcpStream)> {
        let (head, body) = read_request(&mut inbound).await?;
        let head_text = String::from_utf8_lossy(&head).into_owned();
        let target = head_text.split(' ').nth(1).unwrap_or("").to_string();
        let session = query_value(&target, "session_id").unwrap_or_default();
        let callback = head_text.lines().any(|l| {
            l.to_ascii_lowercase()
                .starts_with("authorization: bearer bjn_v1.")
        });
        let mut record = callback.then(|| {
            let job = serde_json::from_slice::<Value>(&body)
                .ok()
                .and_then(|v| v["id"].as_str().map(str::to_string))
                .unwrap_or_default();
            let nth = {
                let mut per = self.per_job.lock().unwrap();
                let n = per.entry(job.clone()).or_default();
                *n += 1;
                *n
            };
            SeenCallback {
                at: Instant::now(),
                session: session.clone(),
                job,
                nth,
                status: 0,
                routed: false,
                answered: Instant::now(),
                tried: Vec::new(),
            }
        });
        if let Some(rec) = &mut record {
            let policy = self.policy.lock().unwrap().clone();
            match policy.map_or(Verdict::Route, |p| p(rec)) {
                Verdict::Route => {}
                Verdict::Delay(d) => {
                    tokio::time::sleep(d).await;
                    // `at` is when it goes on to a replica.
                    rec.at = Instant::now();
                }
                Verdict::Answer(status) => {
                    let _ = inbound.write_all(fixed_answer(status).as_bytes()).await;
                    rec.status = status;
                    rec.answered = Instant::now();
                    self.seen.lock().unwrap().push(rec.clone());
                    return None;
                }
                Verdict::Drop => {
                    self.seen.lock().unwrap().push(rec.clone());
                    return None;
                }
            }
        }
        // The hashed slot, once more on 503, then the rest of the ring.
        let order = self.order(&session);
        let mut tries: Vec<usize> = Vec::new();
        if let Some(first) = order.first() {
            tries.push(*first);
            tries.push(*first);
        }
        tries.extend(order.iter().skip(1));
        let mut tried = Vec::new();
        for slot in tries {
            let Some(port) = self.slots.lock().unwrap()[slot] else {
                tried.push((slot, 0));
                continue;
            };
            let Ok(mut out) = TcpStream::connect(("127.0.0.1", port)).await else {
                tried.push((slot, 0));
                continue;
            };
            if out.write_all(&head).await.is_err() || out.write_all(&body).await.is_err() {
                tried.push((slot, 0));
                continue;
            }
            let Some(answer_head) = read_head(&mut out).await else {
                tried.push((slot, 0));
                continue;
            };
            let status = status_of(&answer_head);
            tried.push((slot, status));
            if status == 503 {
                tokio::time::sleep(Duration::from_millis(50)).await;
                continue;
            }
            if let Some(rec) = &mut record {
                rec.status = status;
                rec.routed = true;
                rec.answered = Instant::now();
                rec.tried = tried.clone();
                self.seen.lock().unwrap().push(rec.clone());
            }
            if inbound.write_all(&answer_head).await.is_err() {
                return None;
            }
            return Some((inbound, out));
        }
        if let Some(rec) = &mut record {
            rec.status = 503;
            rec.answered = Instant::now();
            rec.routed = tried.iter().any(|(_, s)| *s != 0);
            rec.tried = tried;
            self.seen.lock().unwrap().push(rec.clone());
        }
        let _ = inbound.write_all(fixed_answer(503).as_bytes()).await;
        None
    }
}

/// Rendezvous order of `n` slots for `session`.
pub fn ring_order(session: &str, n: usize) -> Vec<usize> {
    let mut order: Vec<(u64, usize)> = (0..n)
        .map(|i| {
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for b in session.bytes().chain([b'#', i as u8]) {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
            (h, i)
        })
        .collect();
    order.sort_by(|a, b| b.cmp(a));
    order.into_iter().map(|(_, i)| i).collect()
}

/// A session id no other run on this host uses. The baseline keeps every session's jobs under
/// one directory of the host (`/tmp/beyond-jobs`, by a digest of tenant and session), so two runs
/// sharing a session id would read back each other's jobs.
pub fn unique(stem: &str) -> String {
    static N: AtomicUsize = AtomicUsize::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!(
        "s1.{stem}-{:x}-{nanos:x}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    )
}

/// A [`unique`] session id whose ring order starts at `slot`.
pub fn session_on(stem: &str, slot: usize, slots: usize) -> String {
    std::iter::repeat_with(|| unique(stem))
        .find(|s| ring_order(s, slots)[0] == slot)
        .unwrap()
}

fn fixed_answer(status: u16) -> String {
    format!("HTTP/1.1 {status} Front\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
}

fn status_of(head: &[u8]) -> u16 {
    String::from_utf8_lossy(head)
        .split(' ')
        .nth(1)
        .and_then(|c| c.parse().ok())
        .unwrap_or(0)
}

/// A query parameter's value, undecoded (session ids here need no decoding).
fn query_value(target: &str, key: &str) -> Option<String> {
    let (_, q) = target.split_once('?')?;
    q.split('&')
        .find_map(|kv| kv.strip_prefix(&format!("{key}=")).map(str::to_string))
}

/// A head, up to and including its blank line, and whatever followed it in the same reads.
async fn read_head(s: &mut TcpStream) -> Option<Vec<u8>> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        if buf.windows(4).any(|w| w == b"\r\n\r\n") {
            return Some(buf);
        }
        let n = s.read(&mut chunk).await.ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
}

/// A request's head and its body (by `Content-Length`; none for a WebSocket upgrade).
async fn read_request(s: &mut TcpStream) -> Option<(Vec<u8>, Vec<u8>)> {
    let mut buf = read_head(s).await?;
    let end = buf.windows(4).position(|w| w == b"\r\n\r\n")? + 4;
    let len = String::from_utf8_lossy(&buf[..end])
        .lines()
        .find_map(|l| {
            let (k, v) = l.split_once(':')?;
            k.trim()
                .eq_ignore_ascii_case("content-length")
                .then(|| v.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0);
    let mut chunk = [0u8; 4096];
    while buf.len() < end + len {
        let n = s.read(&mut chunk).await.ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    let body = buf.split_off(end);
    Some((buf, body))
}

// ---- the fleet

/// A replica slot's process.
pub enum Proc {
    /// The first replica, which the [`Service`] owns.
    First,
    Peer(Peer),
    Down,
}

/// How a fleet is started.
pub struct Setup {
    pub mode: Mode,
    /// Replica slots on the ring.
    pub replicas: usize,
    /// Whether the endpoint's deployment offers notify (off: today's behaviour).
    pub notify: bool,
    /// A detached session's idle timeout, in seconds.
    pub idle_timeout: Option<u64>,
    /// A change to the mode's endpoint.
    pub tweak: fn(&mut MockOptions),
    /// Extra environment for the replicas.
    pub env: Vec<(String, String)>,
}

impl Setup {
    pub fn new(mode: Mode) -> Self {
        Self {
            mode,
            replicas: 1,
            notify: true,
            idle_timeout: None,
            tweak: |_| {},
            env: Vec::new(),
        }
    }
}

/// Real replicas on a ring behind the front, one endpoint behind the meter, one model.
pub struct E2e {
    pub mode: Mode,
    pub svc: Service,
    pub front: Front,
    pub meter: Meter,
    pub requests: Arc<Mutex<Vec<String>>>,
    pub procs: Vec<Proc>,
    replica_args: Vec<String>,
    /// Sessions whose sandbox is reached through a meter of their own ([`Self::isolate`]).
    meters: Mutex<HashMap<String, Meter>>,
}

impl E2e {
    pub async fn start(setup: Setup) -> Self {
        let (base, requests) = beyond_test_support::spawn_model_server_fn(answer);
        let (front, listener) = Front::bind(setup.replicas).await;
        let mut replica_args = vec!["--notify-origin".to_string(), front.origin.clone()];
        if let Some(secs) = setup.idle_timeout {
            replica_args.extend(["--session-idle-timeout".to_string(), secs.to_string()]);
        }
        let mut mock = setup.mode.mock(setup.notify);
        (setup.tweak)(&mut mock);
        let svc = Service::start_with(
            &base,
            &["s1"],
            Options {
                extra_args: replica_args.clone(),
                exec: Some(mock),
                env: setup.env,
                ..Default::default()
            },
        )
        .await;
        let meter = Meter::start(&svc.exec.url).await;
        front.set_slot(0, Some(svc.port));
        let mut procs = vec![Proc::First];
        for slot in 1..setup.replicas {
            let args: Vec<&str> = replica_args.iter().map(String::as_str).collect();
            let peer = svc.start_peer(&args);
            front.set_slot(slot, Some(peer.port));
            procs.push(Proc::Peer(peer));
        }
        front.serve(listener);
        Self {
            mode: setup.mode,
            svc,
            front,
            meter,
            requests,
            procs,
            replica_args,
            meters: Mutex::default(),
        }
    }

    /// A grant for `session` whose sandbox is reached through the meter.
    pub fn grant(&self, session: &str) -> String {
        let url = self
            .meters
            .lock()
            .unwrap()
            .get(session)
            .map_or_else(|| self.meter.url.clone(), |m| m.url.clone());
        self.svc
            .token_with(TENANT, session, move |c| c.exec_url = url)
    }

    /// Open `session` through the front, retrying while the ring says "not here yet" (a
    /// takeover).
    pub async fn open(&self, session: &str) -> TestWs {
        let token = self.grant(session);
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            match ws_connect_with_headers(self.front.port, Some(session), &self.svc.header(&token))
                .await
            {
                Ok(ws) => return ws,
                Err(status) => {
                    assert!(
                        Instant::now() < deadline,
                        "{session} still refused with {status}"
                    );
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
            }
        }
    }

    /// Open `session` and have the model start `command` in the background; the client stays
    /// attached.
    pub async fn start_job(&self, session: &str, command: &str) -> TestWs {
        let mut ws = self.open(session).await;
        prompt(&mut ws, &run_prompt(command)).await;
        ws
    }

    /// The endpoint's record of the job's stream (native: the job's own; baseline: the
    /// supervisor's), by a needle of its command.
    pub async fn stream(&self, needle: &str) -> u32 {
        self.svc
            .exec
            .wait_for_stream(needle, Duration::from_secs(30))
            .await
            .unwrap_or_else(|| {
                panic!(
                    "no stream for {needle:?}: {:?}",
                    self.svc
                        .exec
                        .streams()
                        .iter()
                        .map(|s| s.command.clone())
                        .collect::<Vec<_>>()
                )
            })
            .pgid
    }

    /// Wait for the job's stream to be let go of with the job still running (released).
    pub async fn released(&self, pgid: u32) {
        // The baseline's stream is its `start` command's, which the endpoint ends by its group
        // once the replica lets go; the job's own group runs on under its `setsid` supervisor.
        let end = self
            .svc
            .exec
            .wait_for_end(pgid, Duration::from_secs(15))
            .await;
        assert!(
            matches!(end, Some(StreamEnd::Disconnected { .. })),
            "{}: released: {end:?}",
            self.mode.name()
        );
    }

    /// The first callback the front saw for `session` (the job's proof), waiting for it.
    pub async fn proof(&self, session: &str) -> SeenCallback {
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            if let Some(c) = self.front.seen().into_iter().find(|c| c.session == session) {
                return c;
            }
            assert!(Instant::now() < deadline, "no proof for {session}");
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// The job's real token, as the fleet mints it for the session's sandbox.
    pub fn token(&self, session: &str, job: &str) -> String {
        key(&self.svc).mint(TENANT, session, &JobId::parse(job).unwrap())
    }

    /// A callback for `job` of `session`, as the sandbox sends it, through the front.
    pub async fn callback(&self, session: &str, job: &str) -> u16 {
        let token = self.token(session, job);
        self.post(session, &token, &scope(session), job).await
    }

    /// POST a callback with whatever token, scope and job to `session`'s address on the front.
    pub async fn post(&self, session: &str, token: &str, scope: &str, job: &str) -> u16 {
        post_to(self.front.port, session, token, scope, job).await
    }

    /// Give `session` a meter of its own, so the bytes its replica exchanges with the sandbox are
    /// told apart from every other session's on the same endpoint.
    pub async fn isolate(&self, session: &str) -> Meter {
        let meter = Meter::start(&self.svc.exec.url).await;
        self.meters
            .lock()
            .unwrap()
            .insert(session.to_string(), meter.clone());
        meter
    }

    /// The port of `slot`'s replica (panics when it is down).
    pub fn svc_port(&self, slot: usize) -> u16 {
        match &self.procs[slot] {
            Proc::First => self.svc.port,
            Proc::Peer(p) => p.port,
            Proc::Down => panic!("slot {slot} is down"),
        }
    }

    /// Kill `slot`'s replica with nothing unwound.
    pub fn kill(&mut self, slot: usize) {
        match std::mem::replace(&mut self.procs[slot], Proc::Down) {
            Proc::First => {
                let _ = self.svc.child.kill();
                let _ = self.svc.child.wait();
            }
            Proc::Peer(mut p) => {
                let _ = p.child.kill();
                let _ = p.child.wait();
            }
            Proc::Down => {}
        }
        self.front.set_slot(slot, None);
    }

    /// A fresh replica in `slot` (a deploy's replacement), on the same mounts and keys.
    pub fn restart(&mut self, slot: usize) -> u16 {
        let args: Vec<&str> = self.replica_args.iter().map(String::as_str).collect();
        let peer = self.svc.start_peer(&args);
        let port = peer.port;
        self.procs[slot] = Proc::Peer(peer);
        self.front.set_slot(slot, Some(port));
        port
    }

    /// Model requests whose last message carries `needle`.
    pub fn turns_with(&self, needle: &str) -> Vec<String> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .filter(|r| last_message_text(r).contains(needle))
            .cloned()
            .collect()
    }

    pub fn model_requests(&self) -> usize {
        self.requests.lock().unwrap().len()
    }

    /// Wait (bounded) until a model request's last message carries `needle`: a turn the notice
    /// started or joined.
    pub async fn wake_with(&self, needle: &str, within: Duration) -> Option<String> {
        let deadline = Instant::now() + within;
        loop {
            if let Some(r) = self.turns_with(needle).into_iter().next() {
                return Some(r);
            }
            if Instant::now() >= deadline {
                return None;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// How many times `needle` was reported in `session`'s transcript: the count in the request
    /// of one more prompt, which carries the whole conversation.
    pub async fn reports(&self, ws: &mut TestWs, needle: &str) -> usize {
        static ASKS: AtomicUsize = AtomicUsize::new(1);
        let ask = format!("how are things {}", ASKS.fetch_add(1, Ordering::Relaxed));
        prompt_when_idle(ws, &ask).await;
        // A prompt sent during a turn is queued behind it: wait for the request that carries it.
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let found = self
                .requests
                .lock()
                .unwrap()
                .iter()
                .map(|r| super::body_json(r)["messages"].to_string())
                .rfind(|m| m.contains(&ask));
            if let Some(messages) = found {
                return messages.matches(needle).count();
            }
            assert!(
                Instant::now() < deadline,
                "the prompt never reached the model"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// Each message of the latest model request that carries `needle`, as role and text: what a
    /// failed report count prints.
    pub fn report_contexts(&self, needle: &str) -> Vec<String> {
        let reqs = self.requests.lock().unwrap();
        let Some(last) = reqs.iter().rfind(|r| r.contains(needle)) else {
            return Vec::new();
        };
        super::body_json(last)["messages"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|m| m.to_string().contains(needle))
            .map(|m| format!("{}: {}", m["role"], texts(&m["content"]).join(" | ")))
            .collect()
    }

    /// Exec requests the endpoint received so far.
    pub fn exec_requests(&self) -> usize {
        self.svc.exec.requests().len()
    }
}

/// Send `message` as a prompt and wait for its response.
pub async fn prompt(ws: &mut TestWs, message: &str) {
    ws_send(ws, json!({"type":"prompt","id":"p","message":message})).await;
    let frames = super::ws_read_until_response(ws, "prompt").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:?}");
}

/// Send `message` as a prompt once the session is not running one (a wake turn may be), and wait
/// for its response.
pub async fn prompt_when_idle(ws: &mut TestWs, message: &str) {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        ws_send(ws, json!({"type":"prompt","id":"p","message":message})).await;
        let frames = super::ws_read_until_response(ws, "prompt").await;
        let last = frames.last().unwrap();
        if last["success"] == true {
            return;
        }
        assert!(
            last["error"]
                .as_str()
                .is_some_and(|e| e.starts_with("busy"))
                && Instant::now() < deadline,
            "{frames:?}"
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

/// A client command that uses the sandbox (a real use).
pub async fn use_sandbox(ws: &mut TestWs) {
    ws_send(ws, json!({"type":"bash","id":"b","command":"true"})).await;
    super::ws_read_until_response(ws, "bash").await;
}

/// The session's job scope, as the sandbox's callbacks name it.
pub fn scope(session: &str) -> String {
    JobScope::for_session(Some(TENANT), session)
        .name()
        .to_string()
}

pub fn key(svc: &Service) -> NotifyKey {
    super::notify_fleet::seal_key(svc)
}

/// POST a callback to a port (the front, or one replica directly); the status.
pub async fn post_to(port: u16, session: &str, token: &str, scope: &str, job: &str) -> u16 {
    let body = format!(r#"{{"type":"job_ended","scope":"{scope}","id":"{job}"}}"#);
    super::notify_fleet::post(
        &format!("http://127.0.0.1:{port}"),
        &format!("/_beyond/agent?session_id={session}&tenant={TENANT}"),
        token,
        &body,
    )
    .await
}

/// The notice text in a model request, with the run's own timing taken out (`after N s`).
pub fn notice(raw: &str, needle: &str) -> String {
    let text = last_message_text(raw);
    let start = text.find("[job ").unwrap_or(0);
    let tail = &text[start..];
    let end = tail.find(needle).map_or(tail.len(), |i| i + needle.len());
    normalize(&tail[..end])
}

fn normalize(s: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some(i) = rest.find("after ") {
        out.push_str(&rest[..i + 6]);
        rest = &rest[i + 6..];
        let digits = rest
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.')
            .count();
        if digits > 0 {
            out.push('N');
            rest = &rest[digits..];
        }
    }
    out.push_str(rest);
    out
}

/// Kills a job's group a failed assertion would leave running.
pub use super::notify_fleet::Reap;

/// Whether a replica holds `session` open, asked the one way that has no effect: a callback with
/// a real token for a job the session never had, sent to the replica directly. A replica that
/// holds the session answers without contacting the sandbox (N32: `410`, or `200` until its open
/// reconcile has succeeded); one that does not, `503` (N24).
pub async fn holds(svc: &Service, port: u16, session: &str) -> bool {
    let job = JobId::fresh();
    let token = key(svc).mint(TENANT, session, &job);
    match post_to(port, session, &token, &scope(session), job.as_str()).await {
        // 200 before the session's open reconcile has succeeded (N32), 410 after.
        200 | 410 => true,
        503 => false,
        other => panic!("unexpected answer {other} to a probe of {session}"),
    }
}

/// Wait (bounded) until no replica in `ports` holds `session` (it was reaped).
pub async fn wait_reaped(svc: &Service, ports: &[u16], session: &str) {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let mut any = false;
        for p in ports {
            any |= holds(svc, *p, session).await;
        }
        if !any {
            return;
        }
        assert!(Instant::now() < deadline, "{session} was never reaped");
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

/// One test per endpoint mode, `case::native`, each running `case(Mode)`.
#[allow(unused_macros)]
macro_rules! in_every_mode {
    ($($case:ident),* $(,)?) => {$(
        mod $case {
            use super::*;
            #[tokio::test(flavor = "multi_thread")]
            async fn native() {
                super::$case(Mode::Native).await;
            }

        }
    )*};
}
#[allow(unused_imports)]
pub(crate) use in_every_mode;

/// As [`in_every_mode`], for a case that needs a callback path.
#[allow(unused_macros)]
macro_rules! in_notify_modes {
    ($($case:ident),* $(,)?) => {$(
        mod $case {
            use super::*;
            #[tokio::test(flavor = "multi_thread")]
            async fn native() {
                super::$case(Mode::Native).await;
            }

        }
    )*};
}
#[allow(unused_imports)]
pub(crate) use in_notify_modes;
