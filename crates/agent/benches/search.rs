// Bench target: `.unwrap()`/`.expect()` set up the fixture; not production code.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Macro-bench for the search tools. Builds one temp tree (~5k files across nested dirs, a known
//! fraction carrying the needle) and times the *real* search code path — without duplicating any
//! logic. `grep` is measured at `threads = 1` (single-threaded baseline) vs `threads = 0` (auto, ≈
//! CPU count); the ratio is the parallel-walk win. `find` is sequential (parallelizing it regressed
//! on this tree — its per-file work is too cheap to amortize thread overhead), so it gets one number.
//!
//! Run: `cargo bench -p beyond-ai-agent`. Throughput is files-scanned/sec.

use std::fs;

use beyond_ai_agent::tools::{find, grep};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use globset::Glob;
use regex::Regex;
use tempfile::TempDir;

const DIRS: usize = 50;
const FILES_PER_DIR: usize = 100; // 5000 files total
const LINES_PER_FILE: usize = 50;
const NEEDLE: &str = "TARGET_NEEDLE";

/// A reproducible tree: every 10th file carries the needle once, the rest are plain source lines.
fn build_tree() -> TempDir {
    let dir = tempfile::tempdir().unwrap();
    for d in 0..DIRS {
        let sub = dir.path().join(format!("dir_{d:03}"));
        fs::create_dir_all(&sub).unwrap();
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
            fs::write(sub.join(format!("file_{f:03}.rs")), content).unwrap();
        }
    }
    dir
}

fn bench_search(c: &mut Criterion) {
    let tree = build_tree();
    let root = tree.path().to_path_buf();
    let files = (DIRS * FILES_PER_DIR) as u64;
    let thread_counts = [("single", 1usize), ("auto", 0usize)];

    // Compile the matchers once, outside the timed loop, so each iteration measures only the walk.
    let re = Regex::new(NEEDLE).unwrap();
    let mut g = c.benchmark_group("grep");
    g.throughput(Throughput::Elements(files));
    for (label, threads) in thread_counts {
        let job = grep::GrepJob::new(re.clone(), None, root.clone(), 100);
        g.bench_function(label, |b| {
            b.iter(|| {
                let (matches, _) = grep::search(&job, threads);
                std::hint::black_box(matches.len());
            });
        });
    }
    g.finish();

    // find's walk is sequential (parallel regressed it — see the module docs); one measurement.
    let job = find::FindJob::new(
        Glob::new("*.rs").unwrap().compile_matcher(),
        true,
        root.clone(),
        1000,
    );
    let mut f = c.benchmark_group("find");
    f.throughput(Throughput::Elements(files));
    f.bench_function("sequential", |b| {
        b.iter(|| {
            let (paths, _) = find::search(&job);
            std::hint::black_box(paths.len());
        });
    });
    f.finish();
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
