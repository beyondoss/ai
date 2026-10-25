//! Instrumented `spawn_blocking`: how many blocking-pool jobs are waiting or running, and for how long.
//!
//! The process runtime is `current_thread`; everything that touches a filesystem (a session's lock,
//! its persist and fsync, a bundle unpack, a tool's read) goes to tokio's blocking pool, which holds up
//! to 512 threads. Once those are all busy a new job queues behind them, and nothing about the queue is
//! visible from outside: the thread count tops out at the cap whether the pool is saturated or merely
//! busy. [`spawn`] is `tokio::task::spawn_blocking` plus three numbers per named call site:
//!
//! - **queued** and **running** gauges (their sum is the site's in-flight count);
//! - **queue wait** — submit to closure start — and **run time**, each as fixed cumulative buckets plus
//!   a sum and a count, and the largest seen since the previous scrape.
//!
//! The cost is a handful of relaxed atomics and two clock reads per job, on a static array indexed by
//! [`Site`]: no allocation, no lock, no map, and the label set is closed, so nothing here can grow
//! with load. [`render`] writes the Prometheus text that [`crate::metrics::Metrics::encode`] appends,
//! and [`inflight`] gives the all-sites total for a log line.
//!
//! The same module carries the runtime's scheduling lag ([`spawn_lag_probe`]): how late a 50 ms timer
//! fires on the `current_thread` runtime. A saturated blocking pool and a busy runtime thread both
//! show up as slow opens; this tells them apart.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};

use tokio::task::JoinHandle;

/// The closed set of call sites. Adding one is a variant, a name in [`Site::NAME`] and nothing else.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Site {
    /// Taking a session's directory and advisory lock (`create_dir_all` + lock).
    SessionLock,
    /// Releasing a session's lock and removing an empty directory.
    SessionUnlock,
    /// `Persistence::open`: resolve the directory and replay the transcript.
    SessionOpen,
    /// A turn's persist (`sync_all`, directory fsync).
    SessionPersist,
    /// A mid-run checkpoint's append.
    SessionCheckpoint,
    /// Listing sessions (the picker).
    SessionList,
    /// Unpacking a fetched bundle into the cache.
    BundleInstall,
    /// Deleting evicted bundles.
    BundleReap,
    /// The `/readyz` shard probe.
    ReadyProbe,
    /// The other `serve` work: job teardown and the MCP Events session scans.
    Serve,
    /// The file tools (`read`, `edit`, `ls`, `find`, `grep`).
    FileTool,
    /// MCP state stores (manifest, events state, auth) and memory files.
    Store,
}

impl Site {
    const ALL: [Self; 12] = [
        Self::SessionLock,
        Self::SessionUnlock,
        Self::SessionOpen,
        Self::SessionPersist,
        Self::SessionCheckpoint,
        Self::SessionList,
        Self::BundleInstall,
        Self::BundleReap,
        Self::ReadyProbe,
        Self::Serve,
        Self::FileTool,
        Self::Store,
    ];
    const NAME: [&'static str; 12] = [
        "session_lock",
        "session_unlock",
        "session_open",
        "session_persist",
        "session_checkpoint",
        "session_list",
        "bundle_install",
        "bundle_reap",
        "ready_probe",
        "serve",
        "file_tool",
        "store",
    ];
}

/// Upper bounds, in microseconds, of the cumulative buckets: 1 ms, 10 ms, 100 ms, 500 ms, 1 s, 5 s,
/// then everything above.
const BOUNDS_US: [u64; 6] = [1_000, 10_000, 100_000, 500_000, 1_000_000, 5_000_000];
const BOUNDS_LE: [&str; 6] = ["0.001", "0.01", "0.1", "0.5", "1", "5"];

/// One duration distribution: per-bucket counts (not cumulative), the sum, and the largest since the
/// last [`render`].
struct Dist {
    buckets: [AtomicU64; 7],
    sum_us: AtomicU64,
    max_us: AtomicU64,
}

impl Dist {
    const fn new() -> Self {
        Self {
            buckets: [const { AtomicU64::new(0) }; 7],
            sum_us: AtomicU64::new(0),
            max_us: AtomicU64::new(0),
        }
    }

    fn observe(&self, d: Duration) {
        let us = u64::try_from(d.as_micros()).unwrap_or(u64::MAX);
        let i = BOUNDS_US.iter().position(|b| us <= *b).unwrap_or(6);
        self.buckets[i].fetch_add(1, Relaxed);
        self.sum_us.fetch_add(us, Relaxed);
        self.max_us.fetch_max(us, Relaxed);
    }

    #[cfg(test)]
    fn count(&self) -> u64 {
        self.buckets.iter().map(|b| b.load(Relaxed)).sum()
    }

    fn render(&self, out: &mut String, what: &str, site: &str) {
        use std::fmt::Write as _;
        let mut cum = 0;
        for (i, le) in BOUNDS_LE.iter().enumerate() {
            cum += self.buckets[i].load(Relaxed);
            let _ = writeln!(
                out,
                "agent_blocking_{what}_seconds_bucket{{site=\"{site}\",le=\"{le}\"}} {cum}"
            );
        }
        cum += self.buckets[6].load(Relaxed);
        let _ = writeln!(
            out,
            "agent_blocking_{what}_seconds_bucket{{site=\"{site}\",le=\"+Inf\"}} {cum}"
        );
        let _ = writeln!(
            out,
            "agent_blocking_{what}_seconds_sum{{site=\"{site}\"}} {:.6}",
            self.sum_us.load(Relaxed) as f64 / 1e6
        );
        let _ = writeln!(
            out,
            "agent_blocking_{what}_seconds_count{{site=\"{site}\"}} {cum}"
        );
        let _ = writeln!(
            out,
            "agent_blocking_{what}_max_seconds{{site=\"{site}\"}} {:.6}",
            self.max_us.swap(0, Relaxed) as f64 / 1e6
        );
    }
}

/// One call site's instruments.
pub struct Stat {
    queued: AtomicI64,
    running: AtomicI64,
    wait: Dist,
    run: Dist,
}

impl Stat {
    pub const fn new() -> Self {
        Self {
            queued: AtomicI64::new(0),
            running: AtomicI64::new(0),
            wait: Dist::new(),
            run: Dist::new(),
        }
    }

    /// Jobs submitted and not yet finished.
    pub fn inflight(&self) -> i64 {
        self.queued.load(Relaxed) + self.running.load(Relaxed)
    }

    /// Jobs submitted and not yet started.
    pub fn queued(&self) -> i64 {
        self.queued.load(Relaxed)
    }
}

impl Default for Stat {
    fn default() -> Self {
        Self::new()
    }
}

static STATS: [Stat; 12] = [const { Stat::new() }; 12];

/// A submitted job, until its closure starts. Dropped unstarted (the runtime is going away) it takes
/// itself back out of the queued gauge.
struct Queued {
    stat: &'static Stat,
    submitted: Instant,
    started: bool,
}

/// A running job. Dropped — returned or unwound — it leaves the running gauge and records its run time.
struct Running {
    stat: &'static Stat,
    began: Instant,
}

impl Queued {
    fn start(mut self) -> Running {
        let began = Instant::now();
        self.stat.wait.observe(began - self.submitted);
        self.stat.queued.fetch_sub(1, Relaxed);
        self.stat.running.fetch_add(1, Relaxed);
        self.started = true;
        Running {
            stat: self.stat,
            began,
        }
    }
}

impl Drop for Queued {
    fn drop(&mut self) {
        if !self.started {
            self.stat.queued.fetch_sub(1, Relaxed);
        }
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        self.stat.run.observe(self.began.elapsed());
        self.stat.running.fetch_sub(1, Relaxed);
    }
}

/// `tokio::task::spawn_blocking`, counted against `site`.
pub fn spawn<F, T>(site: Site, f: F) -> JoinHandle<T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    spawn_on(&STATS[site as usize], f)
}

/// [`spawn`] against an explicit [`Stat`], so a test can use one of its own.
pub fn spawn_on<F, T>(stat: &'static Stat, f: F) -> JoinHandle<T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    stat.queued.fetch_add(1, Relaxed);
    let queued = Queued {
        stat,
        submitted: Instant::now(),
        started: false,
    };
    tokio::task::spawn_blocking(move || {
        let _running = queued.start();
        f()
    })
}

/// Jobs in flight (queued plus running) across every site, and how many of those are still queued.
pub fn inflight() -> (i64, i64) {
    STATS
        .iter()
        .fold((0, 0), |(i, q), s| (i + s.inflight(), q + s.queued()))
}

/// The Prometheus text for every site, plus the runtime lag. Each `*_max_seconds` is the largest value
/// since the previous call, so one scraper sees an interval's worst case and a second would split it.
pub fn render(out: &mut String) {
    use std::fmt::Write as _;
    for (site, name) in Site::ALL.iter().zip(Site::NAME) {
        let s = &STATS[*site as usize];
        let _ = writeln!(
            out,
            "agent_blocking_queued{{site=\"{name}\"}} {}",
            s.queued.load(Relaxed)
        );
        let _ = writeln!(
            out,
            "agent_blocking_running{{site=\"{name}\"}} {}",
            s.running.load(Relaxed)
        );
        s.wait.render(out, "wait", name);
        s.run.render(out, "run", name);
    }
    let _ = writeln!(
        out,
        "agent_runtime_lag_max_seconds {:.6}",
        LAG_MAX_US.swap(0, Relaxed) as f64 / 1e6
    );
}

static LAG_MAX_US: AtomicU64 = AtomicU64::new(0);

/// How often the lag probe's timer is due.
const LAG_TICK: Duration = Duration::from_millis(50);

/// Start the runtime-lag probe: a task that sleeps [`LAG_TICK`] in a loop and records how late it
/// woke. On the `current_thread` runtime that lateness is the time every other task waited for the
/// thread.
pub fn spawn_lag_probe() {
    tokio::spawn(async move {
        loop {
            let due = tokio::time::Instant::now() + LAG_TICK;
            tokio::time::sleep_until(due).await;
            let late = tokio::time::Instant::now().saturating_duration_since(due);
            LAG_MAX_US.fetch_max(u64::try_from(late.as_micros()).unwrap_or(u64::MAX), Relaxed);
        }
    });
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    fn leaked() -> &'static Stat {
        Box::leak(Box::new(Stat::new()))
    }

    /// Hold `n` closures on a barrier: while they are held the gauge reads exactly `n` running, and
    /// once released it reads zero with `n` run times recorded.
    #[tokio::test]
    async fn in_flight_counts_the_closures_held_on_a_barrier() {
        const N: usize = 8;
        let stat = leaked();
        let held = Arc::new(Barrier::new(N + 1));
        let release = Arc::new(Barrier::new(N + 1));
        let jobs: Vec<_> = (0..N)
            .map(|_| {
                let (held, release) = (held.clone(), release.clone());
                spawn_on(stat, move || {
                    held.wait();
                    release.wait();
                })
            })
            .collect();
        tokio::task::spawn_blocking({
            let held = held.clone();
            move || {
                held.wait();
            }
        })
        .await
        .unwrap();
        assert_eq!((stat.queued(), stat.inflight()), (0, N as i64));
        assert_eq!(stat.running.load(Relaxed), N as i64);

        tokio::task::spawn_blocking(move || {
            release.wait();
        })
        .await
        .unwrap();
        for j in jobs {
            j.await.unwrap();
        }
        assert_eq!(stat.inflight(), 0);
        assert_eq!(stat.run.count(), N as u64);
        assert_eq!(stat.wait.count(), N as u64);
    }

    /// With one blocking thread, a second job waits behind the first: it shows as queued, and its
    /// queue wait is at least as long as the first job held the thread.
    #[test]
    fn queue_wait_is_the_time_spent_behind_a_busy_pool() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .max_blocking_threads(1)
            .build()
            .unwrap();
        rt.block_on(async {
            let stat = leaked();
            let (started_tx, started_rx) = std::sync::mpsc::channel();
            let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
            let first = spawn_on(stat, move || {
                started_tx.send(()).unwrap();
                stop_rx.recv().unwrap();
            });
            started_rx.recv().unwrap();
            let second = spawn_on(stat, || ());
            assert_eq!((stat.queued(), stat.inflight()), (1, 2));

            tokio::time::sleep(Duration::from_millis(40)).await;
            stop_tx.send(()).unwrap();
            first.await.unwrap();
            second.await.unwrap();

            assert_eq!(stat.inflight(), 0);
            assert!(
                stat.wait.max_us.load(Relaxed) >= 40_000,
                "the queued job waited at least 40 ms, saw {} us",
                stat.wait.max_us.load(Relaxed)
            );
            assert_eq!(stat.wait.count(), 2);
        });
    }

    #[tokio::test]
    async fn a_panicking_closure_still_leaves_the_gauge() {
        let stat = leaked();
        let r = spawn_on(stat, || -> () { panic!("boom") }).await;
        assert!(r.is_err());
        assert_eq!(stat.inflight(), 0);
        assert_eq!(stat.run.count(), 1);
    }

    #[test]
    fn render_names_every_site_and_resets_the_interval_max() {
        let mut out = String::new();
        render(&mut out);
        for name in Site::NAME {
            assert!(
                out.contains(&format!("agent_blocking_running{{site=\"{name}\"}} ")),
                "{name} missing:\n{out}"
            );
        }
        assert!(out.contains("agent_runtime_lag_max_seconds "), "{out}");
        assert_eq!(Site::ALL.len(), Site::NAME.len());
    }
}
