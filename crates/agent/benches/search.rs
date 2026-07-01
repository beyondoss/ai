// Bench target: `.unwrap()`/`.expect()` set up fixtures; not production code.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Search-tool bench with **allocations** beside timing. `divan`'s `AllocProfiler` (the global
//! allocator below) reports alloc count + bytes per sample next to ns/iter, so both the speed and the
//! allocation cost are visible in one table — no more "faster, trust me."
//!
//! The headline comparison: our in-process `grep` (ripgrep's `grep-searcher`/`grep-regex` engine
//! linked in) vs shelling out to the `rg` **binary** (what pi does) over the same ~5k-file corpus.
//! `grep` is measured single-threaded (`1`) and auto/parallel (`0`); `find`'s sequential walk gets one
//! number. Run: `cargo bench -p beyond-ai-agent --bench search`.
//!
//! Reading the two grep rows: `grep_engine` allocations are the *real* work — one line + path per hit
//! held as structured results. `grep_rg_subprocess` allocations are what *our* process spends to
//! marshal a subprocess (spawn + slurp its stdout); its time includes the process spawn the in-process
//! path skips entirely. The numbers are the argument.

use std::hint::black_box;
use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use beyond_ai_agent::tools::{find, grep};
use divan::Bencher;
use globset::Glob;
use tempfile::TempDir;

#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

const DIRS: usize = 50;
const FILES_PER_DIR: usize = 100; // 5000 files total
const LINES_PER_FILE: usize = 50;
const NEEDLE: &str = "TARGET_NEEDLE";

/// One reproducible tree, built once and kept alive for the whole run (the `TempDir` lives in the
/// `OnceLock`). Every 10th file carries the needle once; the rest are plain source lines.
fn tree() -> &'static PathBuf {
    static TREE: OnceLock<(TempDir, PathBuf)> = OnceLock::new();
    &TREE
        .get_or_init(|| {
            let dir = tempfile::tempdir().unwrap();
            for d in 0..DIRS {
                let sub = dir.path().join(format!("dir_{d:03}"));
                std::fs::create_dir_all(&sub).unwrap();
                for f in 0..FILES_PER_DIR {
                    let mut content = String::with_capacity(LINES_PER_FILE * 40);
                    for line in 0..LINES_PER_FILE {
                        if line == LINES_PER_FILE / 2 && f % 10 == 0 {
                            content.push_str(NEEDLE);
                            content.push('\n');
                        } else {
                            content.push_str("some ordinary line of source code here\n");
                        }
                    }
                    std::fs::write(sub.join(format!("file_{f:03}.rs")), content).unwrap();
                }
            }
            let root = dir.path().to_path_buf();
            (dir, root)
        })
        .1
}

/// Our in-process grep — ripgrep's engine, no subprocess. `1` = single-threaded baseline, `0` = auto
/// (parallel walk over ≈CPU count).
#[divan::bench(args = [1, 0])]
fn grep_engine(bencher: Bencher, threads: usize) {
    let root = tree();
    let job = grep::GrepJob::new(NEEDLE, false, false, None, root.clone(), 100).unwrap();
    bencher.bench_local(|| {
        let (m, _) = grep::search(&job, threads);
        black_box(m.len());
    });
}

/// The pi approach: spawn the `rg` binary and read its stdout. Alloc columns are *our* process's cost
/// to marshal the subprocess; the time includes the spawn the in-process path avoids.
#[divan::bench]
fn grep_rg_subprocess(bencher: Bencher) {
    let root = tree().to_str().unwrap().to_string();
    bencher.bench_local(|| {
        let out = Command::new("rg")
            .args(["--no-heading", "--line-number", "--color=never", NEEDLE])
            .arg(&root)
            .output()
            .unwrap();
        black_box(out.stdout.len());
    });
}

/// Our in-process find (fd's `ignore` walk). Sequential — parallelizing regressed on this tree (its
/// per-file work is too cheap to amortize thread overhead).
#[divan::bench]
fn find_walk(bencher: Bencher) {
    let root = tree();
    let job = find::FindJob::new(
        Glob::new("*.rs").unwrap().compile_matcher(),
        true,
        root.clone(),
        1000,
    );
    bencher.bench_local(|| {
        let (p, _) = find::search(&job);
        black_box(p.len());
    });
}
