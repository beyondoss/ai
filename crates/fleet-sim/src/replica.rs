//! A replica under test: a real `beyond-ai-agent serve --service` process.
//!
//! Not a mock and not a library call. The claims this simulator checks are about what happens when a
//! replica **dies** — its locks outliving it, its segments being sealed by whoever takes over — and
//! none of that is observable from inside the process that is supposed to be dead. So every replica
//! here is a child process, killed with a real signal.

use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// A running replica.
pub struct Replica {
    pub name: String,
    pub port: u16,
    child: Option<Child>,
}

impl Replica {
    /// Launch a replica serving `shards`, verified against `grant_key_flag`/`seal_key`.
    pub fn start(
        name: &str,
        bin: &str,
        gateway_url: &str,
        port: u16,
        grant_key_flag: &str,
        seal_key: &Path,
        shards: &[(&str, &Path)],
        drain_grace: Option<u64>,
    ) -> Result<Self, String> {
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
        // The replica's own `$HOME` must not be reachable: in service mode a host default is a
        // tenancy bug, and pointing it at a path that does not exist is how the integration tests
        // prove the replica never falls back to one.
        c.env("HOME", "/nonexistent-fleet-sim-home")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            // Piped, not discarded: a replica that refuses to start says why on stderr, and a
            // simulator that throws that away reports "never bound" for a one-line flag mistake.
            .stderr(Stdio::piped());
        let mut child = c.spawn().map_err(|e| format!("spawn {name}: {e}"))?;
        let stderr = child.stderr.take();
        let mut replica = Self {
            name: name.to_string(),
            port,
            child: Some(child),
        };
        if let Err(e) = replica.wait_until_listening() {
            let said = stderr
                .map(|mut s| {
                    let mut buf = String::new();
                    use std::io::Read as _;
                    let _ = s.read_to_string(&mut buf);
                    buf
                })
                .unwrap_or_default();
            let _ = replica.kill_hard();
            return Err(if said.trim().is_empty() {
                e
            } else {
                format!("{e}; it said: {}", said.trim())
            });
        }
        Ok(replica)
    }

    fn wait_until_listening(&self) -> Result<(), String> {
        let deadline = Instant::now() + Duration::from_secs(20);
        while Instant::now() < deadline {
            if std::net::TcpStream::connect(("127.0.0.1", self.port)).is_ok() {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        Err(format!("replica {} never bound {}", self.name, self.port))
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
        child.kill().map_err(|e| format!("kill: {e}"))?;
        let _ = child.wait();
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

    pub fn is_running(&self) -> bool {
        self.child.is_some()
    }
}

impl Drop for Replica {
    fn drop(&mut self) {
        // A simulator that leaks replicas poisons the next run's ports and, worse, leaves a process
        // holding a lock the next scenario is about to reason over.
        if let Some(child) = self.child.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}
