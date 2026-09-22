//! A replica under test: a real `beyond-ai-agent serve --service` process.
//!
//! Not a mock and not a library call. The claims this simulator checks are about what happens when a
//! replica **dies** — its locks outliving it, its segments being sealed by whoever takes over — and
//! none of that is observable from inside the process that is supposed to be dead. So every replica
//! here is a child process, killed with a real signal.

use std::path::Path;

use crate::edge::Addr;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// Everything a replica is launched with. One struct rather than eight positional arguments, which
/// is both what clippy asks for and what stops a caller silently swapping two `&str`s that happen to
/// typecheck — `grant_key_flag` and `gateway_url` are both strings, and getting them the wrong way
/// round would fail as a mysterious verification error at the first connection.
pub struct Launch<'a> {
    pub name: &'a str,
    pub bin: &'a str,
    pub gateway_url: &'a str,
    pub port: u16,
    pub grant_key_flag: &'a str,
    pub seal_key: &'a Path,
    pub shards: &'a [(&'a str, &'a Path)],
    /// `--drain-grace`, when the scenario is about a drain.
    pub drain_grace: Option<u64>,
    /// `--max-live-sessions`, when the scenario is about the cap.
    pub max_live_sessions: Option<usize>,
    /// `--metrics-listen`, when the scenario reads the scrape.
    pub metrics_port: Option<u16>,
}

/// Every replica this process has spawned and not yet reaped.
///
/// `Drop` does not run when the simulator is signalled, and the comment on [`Replica::drop`] says
/// exactly why that matters: a leaked replica holds a session lock the next run is about to reason
/// over, and it goes on listening on a port the next run may be handed. One interrupted run left an
/// orphan that the *next* investigation then mistook for the replica under test. Killing them from
/// the signal handler is what keeps an interrupted run from lying to the run after it.
static LIVE: std::sync::Mutex<Vec<u32>> = std::sync::Mutex::new(Vec::new());

fn forget(pid: u32) {
    if let Ok(mut live) = LIVE.lock() {
        live.retain(|p| *p != pid);
    }
}

/// `SIGKILL` every replica still running. Best-effort, idempotent, signal-path only.
pub fn kill_all() {
    let pids = std::mem::take(&mut *LIVE.lock().unwrap_or_else(|e| e.into_inner()));
    for pid in pids {
        let _ = Command::new("kill").args(["-9", &pid.to_string()]).status();
    }
}

/// A running replica.
pub struct Replica {
    pub name: String,
    /// Where this replica answers. Always loopback for one this process spawned.
    pub addr: Addr,
    child: Option<Child>,
    /// True when this replica was **attached to**, not spawned: an ECS task, or anything else the
    /// simulator did not start and cannot signal. Its process-shaped methods refuse, and the fleet
    /// routes those faults through the fault command instead.
    attached: bool,
    /// Everything the replica has said, drained continuously by a thread that owns the pipe.
    ///
    /// Two things make the draining load-bearing rather than tidy. The read end must stay **open**:
    /// closing it kills the child with `SIGPIPE` on its next write, and since a drain's first act is
    /// to log that it has begun, a harness that drops this pipe kills the replica at exactly the
    /// moment a drain scenario needs it alive — which once looked precisely like "drain does not work
    /// in service mode". And it must stay **drained**: a pipe nobody reads holds 64 KiB, after which
    /// the replica blocks in `write` and the simulator reports a hang it caused itself.
    ///
    /// So the thread reads until EOF and parks the text here, where a scenario can ask for it at any
    /// point rather than only once, and only on the way out.
    said: std::sync::Arc<std::sync::Mutex<String>>,
}

impl Replica {
    /// Launch a replica serving `shards`, verified against `grant_key_flag`/`seal_key`.
    pub fn start(launch: &Launch<'_>) -> Result<Self, String> {
        let Launch {
            name,
            bin,
            gateway_url,
            port,
            grant_key_flag,
            seal_key,
            shards,
            drain_grace,
            max_live_sessions,
            metrics_port,
        } = *launch;
        let mut c = Command::new(bin);
        c.args([
            "serve",
            "--service",
            "--gateway-url",
            gateway_url,
            "--model",
            "claude-test",
            "--listen",
            &format!("127.0.0.1:{port}"),
            "--grant-key",
            grant_key_flag,
            "--seal-key",
            &seal_key.to_string_lossy(),
        ]);
        for (shard, path) in shards {
            c.arg("--shard").arg(format!("{shard}={}", path.display()));
        }
        if let Some(secs) = drain_grace {
            c.arg("--drain-grace").arg(secs.to_string());
        }
        if let Some(max) = max_live_sessions {
            c.arg("--max-live-sessions").arg(max.to_string());
        }
        if let Some(port) = metrics_port {
            // Loopback only, which the flag enforces — the scrape describes every tenant here.
            c.arg("--metrics-listen").arg(format!("127.0.0.1:{port}"));
        }
        // The replica's own `$HOME` must not be reachable: in service mode a host default is a
        // tenancy bug, and pointing it at a path that does not exist is how the integration tests
        // prove the replica never falls back to one.
        // Passed through so a run can A/B the replica's runtime flavour. The single-threaded default
        // was chosen on a benchmark with one session, where nothing contends for the thread; this is
        // what lets the same question be asked at a hundred sessions per replica, where it is 87% of
        // all the CPU a replica spends.
        if let Ok(n) = std::env::var("FLEET_SIM_AGENT_WORKER_THREADS") {
            c.env("BEYOND_AI_AGENT_TOKIO_WORKER_THREADS", n);
        }
        c.env("HOME", "/nonexistent-fleet-sim-home")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            // Piped, not discarded: a replica that refuses to start says why on stderr, and a
            // simulator that throws that away reports "never bound" for a one-line flag mistake.
            .stderr(Stdio::piped());
        let mut child = c.spawn().map_err(|e| format!("spawn {name}: {e}"))?;
        if let Ok(mut live) = LIVE.lock() {
            live.push(child.id());
        }
        let said = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        if let Some(mut pipe) = child.stderr.take() {
            let sink = std::sync::Arc::clone(&said);
            std::thread::spawn(move || {
                use std::io::Read as _;
                let mut buf = [0u8; 4096];
                while let Ok(n) = pipe.read(&mut buf) {
                    if n == 0 {
                        return;
                    }
                    if let Ok(mut s) = sink.lock() {
                        s.push_str(&String::from_utf8_lossy(&buf[..n]));
                    }
                }
            });
        }
        let mut replica = Self {
            name: name.to_string(),
            addr: Addr::local(port),
            child: Some(child),
            attached: false,
            said,
        };
        if let Err(e) = replica.wait_until_listening() {
            let said = replica.said();
            let _ = replica.kill_hard();
            return Err(if said.trim().is_empty() {
                e
            } else {
                format!("{e}; it said: {}", said.trim())
            });
        }
        Ok(replica)
    }

    /// A replica somebody else is running, at `addr`.
    ///
    /// Nothing is spawned and nothing is owned: no child to signal, no stderr to drain, and `Drop`
    /// must not kill it. Everything a scenario would do to a local child — kill, term, restart —
    /// goes through the fault command; its liveness is the `/livez` probe the placement path already
    /// performs, not a pid.
    pub fn attach(name: &str, addr: Addr) -> Self {
        Self {
            name: name.to_string(),
            addr,
            child: None,
            attached: true,
            said: std::sync::Arc::new(std::sync::Mutex::new(String::new())),
        }
    }

    /// Whether this replica was attached to rather than spawned.
    pub fn is_attached(&self) -> bool {
        self.attached
    }

    fn wait_until_listening(&self) -> Result<(), String> {
        let deadline = Instant::now() + Duration::from_secs(20);
        while Instant::now() < deadline {
            if std::net::TcpStream::connect((self.addr.host.as_str(), self.addr.port)).is_ok() {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        Err(format!("replica {} never bound {}", self.name, self.addr))
    }

    /// Whatever the replica has written to stderr so far. Non-consuming and callable at any time —
    /// a scenario that is *about* to give up can read it without first having to end.
    pub fn said(&self) -> String {
        self.said
            .lock()
            .map(|s| s.clone())
            .unwrap_or_else(|e| e.into_inner().clone())
    }

    pub fn pid(&self) -> Option<u32> {
        self.child.as_ref().map(Child::id)
    }

    /// Kill it the way a machine failure does: no signal handler, no destructors, no chance to
    /// release a lock or seal a segment. This is the case the epoch fence exists for.
    pub fn kill_hard(&mut self) -> Result<(), String> {
        let Some(child) = self.child.as_mut() else {
            return Ok(());
        };
        let pid = child.id();
        child.kill().map_err(|e| format!("kill: {e}"))?;
        let _ = child.wait();
        forget(pid);
        self.child = None;
        Ok(())
    }

    /// SIGTERM — the deploy case, which a replica is supposed to survive gracefully.
    pub fn signal_term(&self) -> Result<(), String> {
        let Some(pid) = self.pid() else {
            return Ok(());
        };
        let ok = Command::new("kill")
            .args(["-TERM", &pid.to_string()])
            .status()
            .map_err(|e| format!("kill -TERM: {e}"))?
            .success();
        ok.then_some(())
            .ok_or_else(|| format!("kill -TERM {pid} failed"))
    }

    /// Wait for the process to exit on its own, as a drain should let it.
    pub fn wait_for_exit(&mut self, within: Duration) -> Result<(), String> {
        let Some(child) = self.child.as_mut() else {
            return Ok(());
        };
        let deadline = Instant::now() + within;
        while Instant::now() < deadline {
            match child.try_wait() {
                Ok(Some(_)) => {
                    self.child = None;
                    return Ok(());
                }
                Ok(None) => std::thread::sleep(Duration::from_millis(50)),
                Err(e) => return Err(format!("wait: {e}")),
            }
        }
        Err(format!(
            "replica {} did not exit within {within:?}",
            self.name
        ))
    }

    /// Whether the fleet should still route to this replica.
    ///
    /// For a spawned replica that is exactly "we have not killed it". An attached one is always
    /// routable as far as the fleet is concerned — whoever owns it decides whether it is up, and the
    /// placement path finds out for real by probing. Answering `false` here would drop it from the
    /// ring permanently the first time a scenario stopped it.
    pub fn is_running(&self) -> bool {
        self.attached || self.child.is_some()
    }
}

impl Drop for Replica {
    fn drop(&mut self) {
        // A simulator that leaks replicas poisons the next run's ports and, worse, leaves a process
        // holding a lock the next scenario is about to reason over.
        if let Some(child) = self.child.as_mut() {
            let pid = child.id();
            let _ = child.kill();
            let _ = child.wait();
            forget(pid);
        }
    }
}
