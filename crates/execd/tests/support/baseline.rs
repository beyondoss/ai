//! What an idle execd looks like from outside, read from `/proc`: how many descriptors and threads
//! it holds. Taken once the server is up and idle; after a workload has finished, every connection
//! is closed and every job removed, [`Baseline::restored`] says whether execd is back to it. It
//! reads only what the kernel shows, so it does not depend on execd's internals.
#![allow(dead_code, clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::Path;
use std::time::{Duration, Instant};

use super::Execd;

pub struct Baseline {
    pid: u32,
    fds: usize,
    threads: usize,
}

impl Baseline {
    pub fn take(execd: &Execd) -> Self {
        let pid = execd.pid();
        Self {
            pid,
            fds: fd_targets(pid).len(),
            threads: thread_count(pid),
        }
    }

    /// Wait up to `settle` for execd to be back at the baseline: the same descriptors and threads,
    /// no zombie child, and nothing under `jobs_base` but execd's own (empty) directory. Threads are
    /// the slow one: the runtime's blocking pool lets an idle thread go after 10 s.
    pub fn restored(&self, jobs_base: &Path, settle: Duration) -> Result<(), String> {
        let deadline = Instant::now() + settle;
        loop {
            match self.diff(jobs_base) {
                Ok(()) => return Ok(()),
                Err(e) if Instant::now() >= deadline => return Err(e),
                Err(_) => std::thread::sleep(Duration::from_millis(50)),
            }
        }
    }

    fn diff(&self, jobs_base: &Path) -> Result<(), String> {
        let mut bad = Vec::new();
        let fds = fd_targets(self.pid);
        if fds.len() != self.fds {
            bad.push(format!("{} open fds, was {}: {fds:?}", fds.len(), self.fds));
        }
        let threads = thread_count(self.pid);
        if threads != self.threads {
            bad.push(format!("{threads} threads, was {}", self.threads));
        }
        let zombies = zombie_children(self.pid);
        if !zombies.is_empty() {
            bad.push(format!("zombie children: {zombies:?}"));
        }
        let left = leftovers(jobs_base);
        if !left.is_empty() {
            bad.push(format!("left in the jobs dir: {left:?}"));
        }
        if bad.is_empty() {
            Ok(())
        } else {
            Err(bad.join("; "))
        }
    }
}

/// Where each of the process's descriptors points.
fn fd_targets(pid: u32) -> Vec<String> {
    let mut out: Vec<String> = std::fs::read_dir(format!("/proc/{pid}/fd"))
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| {
            std::fs::read_link(e.path())
                .map(|p| p.display().to_string())
                .unwrap_or_default()
        })
        .collect();
    out.sort();
    out
}

fn thread_count(pid: u32) -> usize {
    std::fs::read_dir(format!("/proc/{pid}/task")).map_or(0, Iterator::count)
}

/// Children of `pid` that have exited and not been waited for.
fn zombie_children(pid: u32) -> Vec<u32> {
    let mut out = Vec::new();
    for e in std::fs::read_dir("/proc").into_iter().flatten().flatten() {
        let Some(child) = e.file_name().to_str().and_then(|n| n.parse::<u32>().ok()) else {
            continue;
        };
        let Ok(stat) = std::fs::read_to_string(e.path().join("stat")) else {
            continue;
        };
        // "pid (comm) S ppid ...": comm may hold anything, so split after the last ')'.
        let Some((_, rest)) = stat.rsplit_once(')') else {
            continue;
        };
        let mut f = rest.split_whitespace();
        if f.next() == Some("Z") && f.next().and_then(|p| p.parse().ok()) == Some(pid) {
            out.push(child);
        }
    }
    out
}

/// Everything inside the jobs directories execd made in `base` (its own `execd-jobs-*` directory
/// is not a leftover; what it holds is).
fn leftovers(base: &Path) -> Vec<String> {
    let mut out = Vec::new();
    for d in std::fs::read_dir(base).into_iter().flatten().flatten() {
        for e in std::fs::read_dir(d.path()).into_iter().flatten().flatten() {
            out.push(e.path().display().to_string());
        }
    }
    out
}
