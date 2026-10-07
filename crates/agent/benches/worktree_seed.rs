// Bench target: `.unwrap()`/`.expect()` set up fixtures; not production code.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Subagent worktree creation, scaled by how many untracked files the parent has.
//!
//! `Worktree::create` seeds the new checkout with the parent's untracked-but-not-ignored files. The
//! number that matters is how its cost grows with that count, and on the remote path it is dominated by
//! *commands*, not bytes: every exec is a round trip to the sandbox. So each row reports the exec count
//! beside wall time, and the remote rows run with an injected per-exec delay standing in for the
//! replica → sandbox hop.
//!
//! The remote backend is `ShellFs` over the same runner, which is what a replica actually uses — not
//! `LocalFs`, which would turn every remote filesystem call into an in-process syscall and hide the
//! very cost under test.
//!
//! Custom harness: this is a scaling table (median of a few runs per row), not a microbench.
//! Run: `cargo bench -p beyond-ai-agent --bench worktree_seed`.

use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use beyond_ai_agent::tools::exec::{CommandRunner, ExecResult, RealRunner};
use beyond_ai_agent::tools::fs::shell::ShellFs;
use beyond_ai_agent::worktree::{Git, Worktree, preflight};

const RUNS: usize = 3;
const FILES_PER_DIR: usize = 10;

/// `RealRunner` plus a fixed delay per exec, counting every call.
struct HopRunner {
    hop: Duration,
    calls: AtomicUsize,
}

#[async_trait]
impl CommandRunner for HopRunner {
    async fn run(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
    ) -> std::io::Result<ExecResult> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        tokio::time::sleep(self.hop).await;
        RealRunner.run(program, args, cwd, timeout).await
    }

    async fn run_with_stdin(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        stdin: &[u8],
    ) -> std::io::Result<ExecResult> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        tokio::time::sleep(self.hop).await;
        RealRunner
            .run_with_stdin(program, args, cwd, timeout, stdin)
            .await
    }
}

fn git(dir: &Path, args: &[&str]) {
    let st = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .status()
        .unwrap();
    assert!(st.success(), "git {args:?}");
}

/// A committed repo plus `untracked` new files spread over subdirectories, ~1 KiB each.
fn fixture(untracked: usize) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path();
    git(p, &["init", "--quiet", "-b", "main"]);
    git(p, &["config", "user.name", "b"]);
    git(p, &["config", "user.email", "b@b"]);
    std::fs::write(p.join("tracked.txt"), "x\n").unwrap();
    git(p, &["add", "-A"]);
    git(p, &["commit", "--quiet", "-m", "init"]);
    let body = "fn f() {}\n".repeat(100);
    for i in 0..untracked {
        let d = p.join(format!("src/m{}", i / FILES_PER_DIR));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join(format!("f{i}.rs")), &body).unwrap();
    }
    dir
}

async fn measure(label: &str, untracked: usize, hop: Option<Duration>) {
    let repo = fixture(untracked);
    let mut times = Vec::with_capacity(RUNS);
    let mut execs = 0;
    for run in 0..RUNS {
        let (g, counter) = match hop {
            None => (Git::Local, None),
            Some(hop) => {
                let runner = Arc::new(HopRunner {
                    hop,
                    calls: AtomicUsize::new(0),
                });
                let backend = Arc::new(ShellFs::connect(runner.clone()).await);
                (Git::remote(runner.clone(), backend), Some(runner))
            }
        };
        let root = preflight(&g, repo.path()).await.unwrap();
        let before = counter
            .as_ref()
            .map_or(0, |c| c.calls.load(Ordering::Relaxed));
        let t = Instant::now();
        let wt = Worktree::create(&g, &root, &format!("bench-{run}"))
            .await
            .unwrap();
        times.push(t.elapsed());
        execs = counter
            .as_ref()
            .map_or(0, |c| c.calls.load(Ordering::Relaxed))
            - before;
        let last = wt.path().join(format!(
            "src/m{}/f{}.rs",
            (untracked - 1) / FILES_PER_DIR,
            untracked - 1
        ));
        assert!(last.is_file(), "seeding must carry every untracked file");
        wt.remove().await.unwrap();
    }
    times.sort();
    let median = times[RUNS / 2];
    let execs = if hop.is_some() {
        execs.to_string()
    } else {
        "-".into()
    };
    println!(
        "{label:<14} {untracked:>6} {execs:>7} {:>11.1}",
        median.as_secs_f64() * 1e3
    );
}

#[tokio::main]
async fn main() {
    println!(
        "{:<14} {:>6} {:>7} {:>11}",
        "mode", "files", "execs", "median ms"
    );
    for n in [10, 100, 1000] {
        measure("local", n, None).await;
        measure("remote 0ms", n, Some(Duration::ZERO)).await;
        measure("remote 2ms", n, Some(Duration::from_millis(2))).await;
    }
}
