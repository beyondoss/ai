//! exec notify under randomized orderings, against the real code. Each seed is a plan: how long the
//! job runs, what the path does with its proof and with its end's callback (route it, hold it,
//! lose one attempt, or refuse it so the sender stops), and a few actions at random moments
//! (a real use of the sandbox, a duplicate or forged callback, killing the replica that has the
//! session, closing the session, reopening it). The plan runs on the end-to-end fleet of
//! `common::notify_e2e` (two real `serve --service` replicas on a ring behind the front), in the
//! seed's endpoint mode, and every run must hold:
//!
//! - **at most one report per end**: the transcript carries the job's end once;
//! - **at least one while open**: with nothing lost and the session never closed or moved, the end
//!   is reported by the stream or a callback without help; otherwise the next use or open reports
//!   it;
//! - **no session is ever opened by a callback**: while the session is closed, callbacks are
//!   answered `503`, no replica holds it, nothing reaches its sandbox and no model turn runs;
//! - **nothing on a timer after release**: while the job is let go of, every byte between its
//!   replica and its sandbox (counted at the socket, on a meter of the session's own) follows
//!   something that asked for it: the release, a use, a callback that reached the replica.
//!
//! 200 seeds run by default (`NOTIFY_E2E_SEEDS` sets the count, `NOTIFY_E2E_FIRST_SEED` the first,
//! `NOTIFY_E2E_WORKERS` the fleets per mode, `NOTIFY_E2E_BUDGET_SECS` a time box after which no
//! new seed starts). A failing seed prints the command that replays it alone.
#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use beyond_agent::exec_endpoint::jobs::JobId;
use common::TestWs;
use common::exec_mock::StreamEnd;
use common::notify_e2e::{
    E2e, Mode, SeenCallback, Setup, TENANT, Verdict, holds, key, post_to, run_prompt, scope,
    unique, wait_reaped,
};

// ---- the plan

/// SplitMix64: a seed's whole plan follows from it.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    /// One of `weights`' indices, in proportion.
    fn pick(&mut self, weights: &[u64]) -> usize {
        let mut r = self.below(weights.iter().sum());
        for (i, w) in weights.iter().enumerate() {
            if r < *w {
                return i;
            }
            r -= w;
        }
        unreachable!()
    }
}

/// What the path does with one callback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fate {
    Route,
    /// Held this long, then routed.
    Delay(u64),
    /// The attempt is lost (the connection drops); the sender tries again.
    Drop,
    /// Refused `404`: the sender stops, and the hint is lost.
    Refuse,
}

impl Fate {
    fn verdict(self) -> Verdict {
        match self {
            Fate::Route => Verdict::Route,
            Fate::Delay(ms) => Verdict::Delay(Duration::from_millis(ms)),
            Fate::Drop => Verdict::Drop,
            Fate::Refuse => Verdict::Answer(404),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Act {
    /// A real use of the sandbox (a client `bash` command).
    Use,
    /// The job's real callback, sent again, straight to the replicas.
    Duplicate,
    /// A callback whose token is not the fleet's.
    Forged,
    /// Kill the replica that has the session; its slot restarts empty.
    Kill,
    /// Close the session (the client leaves; it is reaped).
    Close,
    /// Open it again, through the front.
    Reopen,
}

#[derive(Debug)]
struct Plan {
    seed: u64,
    mode: Mode,
    job_ms: u64,
    proof: Fate,
    end: Fate,
    acts: Vec<(u64, Act)>,
}

impl Plan {
    fn new(seed: u64) -> Self {
        let mode = mode_of(seed);
        let mut rng = Rng(seed ^ 0x6a09_e667_f3bc_c908);
        let job_ms = 300 + rng.below(5700);
        let fate = |rng: &mut Rng, weights: &[u64]| match rng.pick(weights) {
            0 => Fate::Route,
            1 => Fate::Delay(rng.below(3000)),
            2 => Fate::Drop,
            _ => Fate::Refuse,
        };
        let proof = fate(&mut rng, &[55, 25, 10, 10]);
        let end = fate(&mut rng, &[45, 25, 15, 15]);
        let horizon = job_ms + 2500;
        let mut acts: Vec<(u64, Act)> = (0..rng.below(5))
            .map(|_| {
                let at = rng.below(horizon);
                let act = match rng.pick(&[30, 25, 15, 10, 10, 10]) {
                    0 => Act::Use,
                    1 => Act::Duplicate,
                    2 => Act::Forged,
                    3 => Act::Kill,
                    4 => Act::Close,
                    _ => Act::Reopen,
                };
                (at, act)
            })
            .collect();
        acts.sort_by_key(|(at, _)| *at);
        Self {
            seed,
            mode,
            job_ms,
            proof,
            end,
            acts,
        }
    }

    /// Nothing can lose the end: the session never closes or moves, and the end's callback is at
    /// worst late (or the stream holds the job, which no callback can lose).
    fn clean(&self) -> bool {
        let moved = self
            .acts
            .iter()
            .any(|(_, a)| matches!(a, Act::Kill | Act::Close));
        // A refused proof leaves the job held, unless a duplicate (a real callback) arrives
        // inside the proof window and proves the path itself.
        let duplicated = self.acts.iter().any(|(_, a)| *a == Act::Duplicate);
        let end_safe =
            (self.proof == Fate::Refuse && !duplicated) || !matches!(self.end, Fate::Refuse);
        !moved && end_safe
    }
}

fn mode_of(seed: u64) -> Mode {
    Mode::ALL[(seed as usize) % Mode::ALL.len()]
}

// ---- one run

/// A moment the run did something that may contact the sandbox.
#[derive(Clone, Copy, Debug)]
enum Cause {
    Use,
    Duplicate,
    Reopen,
}

struct Run<'a> {
    f: &'a mut E2e,
    plan: &'a Plan,
    session: String,
    marker: String,
    command: String,
    ws: Option<TestWs>,
    /// When the session was last closed (reaped or killed), while it is.
    closed_at: Option<Instant>,
    closed_windows: Vec<(Instant, Instant)>,
    causes: Vec<(Instant, Cause)>,
    /// When the session last stopped being where its job was let go of (a kill or a close).
    moved_at: Option<Instant>,
    reported_before_kill: bool,
    /// The client was sent the end's notice (a `background` event carrying the marker).
    end_event_seen: bool,
    meter: common::notify_e2e::Meter,
}

/// How long after something that asked for it a byte may still cross: a reconcile's `list`, a
/// follow's reads to the end, a use's command.
const GRACE: Duration = Duration::from_secs(4);

impl Run<'_> {
    fn ports(&self) -> Vec<u16> {
        (0..2).map(|s| self.f.svc_port(s)).collect()
    }

    fn job_id(&self) -> Option<String> {
        if self.plan.mode.releases() {
            return self
                .f
                .front
                .seen()
                .into_iter()
                .find(|c| c.session == self.session)
                .map(|c| c.job);
        }
        let scope = scope(&self.session);
        self.f
            .svc
            .exec
            .bodies()
            .iter()
            .find(|b| b["job"]["op"] == "keep" && b["job"]["scope"] == scope.as_str())
            .and_then(|b| b["job"]["id"].as_str().map(str::to_string))
    }

    async fn act(&mut self, act: Act) -> Result<(), String> {
        match act {
            Act::Use => {
                if self.ws.is_some() {
                    self.causes.push((Instant::now(), Cause::Use));
                    self.use_sandbox().await;
                }
            }
            Act::Duplicate => {
                let Some(id) = self.job_id() else {
                    return Ok(());
                };
                let token = self.f.token(&self.session, &id);
                self.causes.push((Instant::now(), Cause::Duplicate));
                for port in self.ports() {
                    let status =
                        post_to(port, &self.session, &token, &scope(&self.session), &id).await;
                    if self.closed_at.is_some() && status != 503 {
                        return Err(format!(
                            "a callback to a closed session got {status} from :{port}"
                        ));
                    }
                    if ![200, 410, 503].contains(&status) {
                        return Err(format!("a duplicate callback got {status}"));
                    }
                }
            }
            Act::Forged => {
                let id = self
                    .job_id()
                    .unwrap_or_else(|| JobId::fresh().as_str().into());
                let mut token =
                    key(&self.f.svc).mint(TENANT, &self.session, &JobId::parse(&id).unwrap());
                let last = token.pop().unwrap();
                token.push(if last == 'A' { 'B' } else { 'A' });
                for port in self.ports() {
                    let status =
                        post_to(port, &self.session, &token, &scope(&self.session), &id).await;
                    if status != 401 {
                        return Err(format!("a forged callback got {status}"));
                    }
                }
            }
            Act::Kill => {
                if self.closed_at.is_some() {
                    return Ok(());
                }
                // A session just opened is held once its task is up; ask until it is.
                let mut holder = None;
                let deadline = Instant::now() + Duration::from_secs(10);
                while holder.is_none() && Instant::now() < deadline {
                    for slot in 0..2 {
                        if holds(&self.f.svc, self.f.svc_port(slot), &self.session).await {
                            holder = Some(slot);
                        }
                    }
                    if holder.is_none() {
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                }
                let Some(slot) = holder else {
                    return Err("an open session that no replica holds".into());
                };
                // A job that already ended may have been reported by the replica about to die,
                // in a turn it persisted before removing the job's record: the known crash gap.
                self.reported_before_kill |= self.job_ended();
                // What the client was sent before the kill, its end's notice included.
                if let Some(mut ws) = self.ws.take() {
                    use futures::StreamExt as _;
                    while let Ok(Some(Ok(msg))) =
                        tokio::time::timeout(Duration::from_millis(100), ws.next()).await
                    {
                        if msg.to_text().is_ok_and(|t| t.contains(&self.marker)) {
                            self.end_event_seen = true;
                        }
                    }
                }
                self.reported_before_kill |= self.end_event_seen
                    || !self.f.turns_with(&self.marker).is_empty()
                    || self.reported_in_transcript();
                self.f.kill(slot);
                self.f.restart(slot);
                let now = Instant::now();
                self.closed_at = Some(now);
                self.moved_at.get_or_insert(now);
            }
            Act::Close => {
                let Some(ws) = self.ws.take() else {
                    return Ok(());
                };
                drop(ws);
                let ports = self.ports();
                wait_reaped(&self.f.svc, &ports, &self.session).await;
                let now = Instant::now();
                self.closed_at = Some(now);
                self.moved_at.get_or_insert(now);
            }
            Act::Reopen => self.reopen().await?,
        }
        Ok(())
    }

    /// Whether any model request so far carries the marker anywhere.
    fn reported_in_transcript(&self) -> bool {
        self.f
            .requests
            .lock()
            .unwrap()
            .iter()
            .any(|r| r.contains(&self.marker))
    }

    /// Reopen a closed session, first checking that nothing opened it while it was closed.
    async fn reopen(&mut self) -> Result<(), String> {
        let Some(closed) = self.closed_at.take() else {
            return Ok(());
        };
        for port in self.ports() {
            if holds(&self.f.svc, port, &self.session).await {
                return Err(format!("a closed session was open on :{port}"));
            }
        }
        self.closed_windows.push((closed, Instant::now()));
        self.causes.push((Instant::now(), Cause::Reopen));
        self.ws = Some(self.f.open(&self.session).await);
        Ok(())
    }

    /// Close (if open) and reopen, then use: the sanctioned way an end no callback delivered is
    /// found.
    async fn recover(&mut self) -> Result<(), String> {
        if let Some(ws) = self.ws.take() {
            drop(ws);
            let ports = self.ports();
            wait_reaped(&self.f.svc, &ports, &self.session).await;
            self.closed_at = Some(Instant::now());
            self.moved_at.get_or_insert(Instant::now());
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
        self.reopen().await?;
        self.causes.push((Instant::now(), Cause::Use));
        self.use_sandbox().await;
        Ok(())
    }

    /// Whether the job has ended in the sandbox: its held stream finished or its process group is
    /// gone (native), or its supervisor wrote its `end` (baseline).
    fn job_ended(&self) -> bool {
        let tag = &self.marker["fin-".len()..];
        self.f
            .svc
            .exec
            .streams()
            .into_iter()
            .find(|s| s.args.iter().any(|a| a.contains(tag)))
            .is_some_and(|s| {
                matches!(s.end, Some((StreamEnd::Finished, _)))
                    || !common::exec_mock::group_alive(s.pgid)
            })
    }

    /// Note whether any frame the client was sent carries the end's notice.
    fn scan(&mut self, frames: &[serde_json::Value]) {
        self.end_event_seen |= frames.iter().any(|f| f.to_string().contains(&self.marker));
    }

    /// A real use of the sandbox: a client `bash` command.
    async fn use_sandbox(&mut self) {
        let ws = self.ws.as_mut().unwrap();
        common::ws_send(
            ws,
            serde_json::json!({"type":"bash","id":"b","command":"true"}),
        )
        .await;
        let frames = common::ws_read_until_response(ws, "bash").await;
        self.scan(&frames);
    }

    /// Every check on what crossed while the session was closed, and while the job was let go of.
    /// Whether the job was let go of, and so its silence checked.
    fn check_contact(&self, model_counts: &[(Instant, usize)]) -> Result<bool, String> {
        let chunks = self.meter.chunks();
        let seen: Vec<SeenCallback> = self
            .f
            .front
            .seen()
            .into_iter()
            .filter(|c| c.session == self.session)
            .collect();
        for (from, to) in &self.closed_windows {
            let from = *from + Duration::from_millis(500);
            if let Some((t, n)) = chunks.iter().find(|(t, _)| *t >= from && t < to) {
                return Err(format!(
                    "{n} bytes reached the sandbox {:?} into a closed window",
                    t.duration_since(from)
                ));
            }
            if let Some(c) = seen
                .iter()
                .find(|c| c.routed && c.at >= from && c.answered < *to && c.status != 503)
            {
                return Err(format!(
                    "a callback to a closed session got {}: {c:?}",
                    c.status
                ));
            }
            // A turn the model server is still taking in when the session closes.
            let from = from + Duration::from_millis(1500);
            let at = |t: Instant| {
                model_counts
                    .iter()
                    .rfind(|(m, _)| *m <= t)
                    .map_or(0, |(_, n)| *n)
            };
            if at(*to) > at(from) {
                return Err("a model turn ran while the session was closed".into());
            }
        }
        // The job's release: its stream let go of while it ran, after a proof was taken.
        if !self.plan.mode.releases() {
            return Ok(false);
        }
        let tag = &self.marker["fin-".len()..];
        let released_at = self
            .f
            .svc
            .exec
            .streams()
            .into_iter()
            .filter(|s| s.command.contains(tag) || s.args.iter().any(|a| a.contains(tag)))
            .find_map(|s| match s.end {
                Some((StreamEnd::Disconnected { .. }, t)) => Some(t),
                _ => None,
            });
        let proved = seen
            .iter()
            .any(|c| c.nth == 1 && c.routed && c.status == 200);
        let (Some(released_at), true) = (released_at, proved) else {
            return Ok(false);
        };
        let from = released_at + Duration::from_secs(3);
        let mut to = Instant::now();
        if let Some(m) = self.moved_at {
            to = to.min(m);
        }
        if let Some(c) = seen
            .iter()
            .find(|c| c.nth >= 2 && c.routed && c.at >= released_at)
        {
            to = to.min(c.at);
        }
        for (t, n) in chunks.iter().filter(|(t, _)| *t >= from && *t < to) {
            let asked = self
                .causes
                .iter()
                .any(|(c, _)| *c <= *t && *t <= *c + GRACE);
            if !asked {
                return Err(format!(
                    "{n} bytes crossed {:?} after release with nothing asking (causes: {:?})",
                    t.duration_since(released_at),
                    self.causes
                        .iter()
                        .map(|(c, k)| (c.saturating_duration_since(released_at), *k))
                        .collect::<Vec<_>>()
                ));
            }
        }
        Ok(true)
    }
}

/// What one seed came to, every check having held.
#[derive(Default)]
struct Outcome {
    /// The job was let go of, so its silence after release was checked.
    released: bool,
    /// The session was closed or its replica killed, so the closed-window checks ran.
    closed: bool,
    /// The end needed the next open and use to be found.
    recovered: bool,
    /// The end was reported before a kill and again by the next owner: ARCHITECTURE.md's "a crash
    /// can report a sandbox job's end twice", today's behaviour and not notify's.
    known_gap: bool,
}

async fn run_seed(f: &mut E2e, seed: u64) -> Result<Outcome, String> {
    let plan = Plan::new(seed);
    assert_eq!(plan.mode, f.mode);
    let session = unique(&format!("o{seed}"));
    let meter = f.isolate(&session).await;
    let marker = format!("fin-q{seed}z");
    let command = format!(
        "sleep {}.{:03}; printf 'fin-%s\\n' q{seed}z",
        plan.job_ms / 1000,
        plan.job_ms % 1000
    );
    let (proof, end, s2) = (plan.proof, plan.end, session.clone());
    f.front.set_policy(Some(Arc::new(move |c: &SeenCallback| {
        if c.session != s2 {
            return Verdict::Route;
        }
        match c.nth {
            1 => proof.verdict(),
            2 => end.verdict(),
            _ => Verdict::Route,
        }
    })));
    // This session's model requests: every one carries its first prompt.
    let own = format!("q{seed}z");
    let model_counts = Arc::new(Mutex::new(Vec::new()));
    let mut run = Run {
        f,
        plan: &plan,
        session,
        marker,
        command,
        ws: None,
        closed_at: None,
        closed_windows: Vec::new(),
        causes: Vec::new(),
        moved_at: None,
        reported_before_kill: false,
        end_event_seen: false,
        meter,
    };
    let mut ws = run.f.open(&run.session).await;
    common::ws_send(
        &mut ws,
        serde_json::json!({"type":"prompt","id":"p","message":run_prompt(&run.command)}),
    )
    .await;
    let frames = common::ws_read_until_response(&mut ws, "prompt").await;
    if frames.last().unwrap()["success"] != true {
        return Err(format!("the prompt that starts the job failed: {frames:?}"));
    }
    run.ws = Some(ws);
    run.scan(&frames);
    let started = Instant::now();
    // The model-request count over time, for the closed windows.
    let watcher = {
        let (counts, requests) = (model_counts.clone(), run.f.requests.clone());
        tokio::spawn(async move {
            loop {
                let n = requests
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|r| r.contains(&own))
                    .count();
                counts.lock().unwrap().push((Instant::now(), n));
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        })
    };

    for (at, act) in &plan.acts {
        let when = started + Duration::from_millis(*at);
        tokio::time::sleep(when.saturating_duration_since(Instant::now())).await;
        run.act(*act).await?;
    }
    if run.closed_at.is_some() {
        run.reopen().await?;
    }

    // The end must be reported: without help when nothing could lose it, else at a use or open.
    let end_at = started + Duration::from_millis(plan.job_ms);
    let wait = Duration::from_secs(if plan.clean() { 15 } else { 8 });
    let deadline = end_at.max(Instant::now()) + wait;
    while !run.reported_in_transcript() && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let mut outcome = Outcome::default();
    if !run.reported_in_transcript() {
        outcome.recovered = true;
        if plan.clean() {
            return Err("the end was never reported while the session was open".into());
        }
        run.recover().await?;
        let deadline = Instant::now() + Duration::from_secs(15);
        while !run.reported_in_transcript() && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        if !run.reported_in_transcript() {
            return Err("the end was not reported at the next open and use".into());
        }
    }
    let (marker, ws) = (run.marker.clone(), run.ws.as_mut().unwrap());
    let reports = {
        let f: &E2e = run.f;
        f.reports(ws, &marker).await
    };
    watcher.abort();
    let counts = model_counts.lock().unwrap().clone();
    outcome.released = run.check_contact(&counts)?;
    outcome.closed = !run.closed_windows.is_empty();
    match reports {
        1 => {}
        2 if run.reported_before_kill => outcome.known_gap = true,
        n => {
            return Err(format!(
                "the end was reported {n} times (reported before a kill: {}): {:?}",
                run.reported_before_kill,
                run.f.report_contexts(&run.marker)
            ));
        }
    }
    // The session is left to the reaper: every check is per session, by its own meter and its
    // own model requests.
    Ok(outcome)
}

// ---- the driver

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

async fn fleet(mode: Mode) -> E2e {
    E2e::start(Setup {
        replicas: 2,
        idle_timeout: Some(1),
        ..Setup::new(mode)
    })
    .await
}

/// One seed on `f`, panics and stalls included; the fleet comes back unless the seed broke it.
async fn guarded(mut f: E2e, seed: u64) -> (Option<E2e>, Result<Outcome, String>) {
    let task = tokio::spawn(async move {
        let r = tokio::time::timeout(Duration::from_secs(150), run_seed(&mut f, seed)).await;
        (f, r)
    });
    match task.await {
        Ok((f, Ok(Ok(outcome)))) => (Some(f), Ok(outcome)),
        Ok((_, Ok(Err(e)))) => (None, Err(e)),
        Ok((_, Err(_))) => (None, Err("the seed stalled past 150 s".into())),
        Err(e) => {
            let msg = e
                .try_into_panic()
                .ok()
                .and_then(|p| {
                    p.downcast_ref::<String>()
                        .cloned()
                        .or_else(|| p.downcast_ref::<&str>().map(|s| (*s).to_string()))
                })
                .unwrap_or_else(|| "the seed's task failed".into());
            (None, Err(format!("panicked: {msg}")))
        }
    }
}

fn replay_command(seed: u64) -> String {
    format!(
        "NOTIFY_E2E_SEED={seed} cargo nextest run -p beyond-agent -E 'test(/^notify_e2e_orderings::/ and test(replay))' --no-capture"
    )
}

/// The test functions each mode's seeds are split across, so they run side by side.
const SHARDS: u64 = 3;

/// Every seed of `mode` in the configured range that falls to `shard`, on `NOTIFY_E2E_WORKERS`
/// fleets of their own.
async fn orderings(mode: Mode, shard: u64) {
    let first = env_u64("NOTIFY_E2E_FIRST_SEED", 0);
    let total = env_u64("NOTIFY_E2E_SEEDS", 200);
    let workers = env_u64("NOTIFY_E2E_WORKERS", 4);
    let budget = std::env::var("NOTIFY_E2E_BUDGET_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .map(Duration::from_secs);
    let seeds: Arc<Vec<u64>> = Arc::new(
        (first..first + total)
            .filter(|s| mode_of(*s) == mode && (s / 3) % SHARDS == shard)
            .collect(),
    );
    let next = Arc::new(AtomicUsize::new(0));
    // Seeds run; and of those that held, how many released, closed, recovered, hit the gap.
    let tally: Arc<[AtomicU64; 5]> = Arc::new(Default::default());
    let failures = Arc::new(Mutex::new(Vec::<(u64, String)>::new()));
    let begun = Instant::now();
    let mut tasks = Vec::new();
    for _ in 0..workers {
        let (seeds, next, tally, failures) =
            (seeds.clone(), next.clone(), tally.clone(), failures.clone());
        tasks.push(tokio::spawn(async move {
            let mut f = Some(fleet(mode).await);
            loop {
                if budget.is_some_and(|b| begun.elapsed() >= b) {
                    return;
                }
                let i = next.fetch_add(1, Ordering::SeqCst);
                let Some(&seed) = seeds.get(i) else { return };
                let fl = match f.take() {
                    Some(fl) => fl,
                    None => fleet(mode).await,
                };
                let (back, result) = guarded(fl, seed).await;
                f = back;
                tally[0].fetch_add(1, Ordering::SeqCst);
                match result {
                    Ok(o) => {
                        for (i, hit) in [o.released, o.closed, o.recovered, o.known_gap]
                            .into_iter()
                            .enumerate()
                        {
                            tally[i + 1].fetch_add(u64::from(hit), Ordering::SeqCst);
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "seed {seed} ({mode:?}) FAILED: {e}\n  plan: {:?}\n  replay: {}",
                            Plan::new(seed),
                            replay_command(seed)
                        );
                        failures.lock().unwrap().push((seed, e));
                    }
                }
            }
        }));
    }
    for t in tasks {
        t.await.unwrap();
    }
    let failures = failures.lock().unwrap().clone();
    let t: Vec<u64> = tally.iter().map(|n| n.load(Ordering::SeqCst)).collect();
    eprintln!(
        "orderings {mode:?} shard {shard}: {} of {} seeds run in {:?}; {} failed; of the rest, {} \
         released, {} closed or moved, {} found at the next open, {} hit the known crash gap",
        t[0],
        seeds.len(),
        begun.elapsed(),
        failures.len(),
        t[1],
        t[2],
        t[3],
        t[4]
    );
    assert!(
        failures.is_empty(),
        "{} seeds failed:\n{}",
        failures.len(),
        failures
            .iter()
            .map(|(s, e)| format!("  seed {s}: {e}\n    replay: {}", replay_command(*s)))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

macro_rules! shards {
    ($($name:ident: $mode:expr, $shard:expr;)*) => {$(
        #[tokio::test(flavor = "multi_thread")]
        async fn $name() {
            orderings($mode, $shard).await;
        }
    )*};
}

shards! {
    orderings_native_0: Mode::Native, 0;
    orderings_native_1: Mode::Native, 1;
    orderings_native_2: Mode::Native, 2;
}

/// Replay one seed alone (`NOTIFY_E2E_SEED`), printing its plan; without the variable, nothing.
#[tokio::test(flavor = "multi_thread")]
async fn replay() {
    let Some(seed) = std::env::var("NOTIFY_E2E_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
    else {
        return;
    };
    let plan = Plan::new(seed);
    eprintln!("seed {}: {plan:?} (clean: {})", plan.seed, plan.clean());
    let (_, result) = guarded(fleet(plan.mode).await, seed).await;
    match result {
        Ok(o) => eprintln!(
            "seed {seed}: every check held (released {}, closed {}, recovered {}, known gap {})",
            o.released, o.closed, o.recovered, o.known_gap
        ),
        Err(e) => panic!("seed {seed}: {e}"),
    }
}
