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

use agent_core::Tool;
use beyond_ai_agent::session_store::{SessionMeta, search_sessions};
use beyond_ai_agent::tools::exec::{CommandRunner, RealRunner};
use beyond_ai_agent::tools::{edit, find, grep, ls, output, read};
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
        let (m, _, _) = grep::search(&job, threads);
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

/// High match-density: a pattern on (nearly) every line, so many hits share one file — the case where
/// per-hit path allocation dominates. Collects up to the hard cap (~10k hits over ~200 files), so the
/// alloc columns expose the `Hit` path representation. **Single-threaded on purpose**: divan's alloc
/// profiler counts the measured thread, so the parallel walk would leave worker-thread allocs
/// unattributed; `threads=1` keeps every allocation on-thread for a clean number to compare across the
/// `PathBuf` → `Arc<Path>` change.
#[divan::bench]
fn grep_dense(bencher: Bencher) {
    let root = tree();
    let job = grep::GrepJob::new("ordinary", false, false, None, root.clone(), 100).unwrap();
    bencher.bench_local(|| {
        let (m, _, _) = grep::search(&job, 1);
        black_box(m.len());
    });
}

const READ_LINES: usize = 3000;

/// One large text file (`READ_LINES` lines), built once, for the `read` bench.
fn big_file() -> &'static PathBuf {
    static F: OnceLock<(TempDir, PathBuf)> = OnceLock::new();
    &F.get_or_init(|| {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("big.txt");
        let mut content = String::with_capacity(READ_LINES * 56);
        for i in 0..READ_LINES {
            content.push_str(&format!(
                "line {i}: some source code content goes here for realism\n"
            ));
        }
        std::fs::write(&path, content).unwrap();
        (dir, path)
    })
    .1
}

/// `read` on a many-line file: the per-line output formatting is the hot path, and it's fully
/// on-thread (synchronous file I/O), so divan's alloc counting is exact. Watch the alloc count across
/// the `format!`-per-line → `write!`-into-buffer change.
#[divan::bench]
fn read_file(bencher: Bencher) {
    let path = big_file().to_str().unwrap().to_string();
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    bencher.bench_local(|| {
        let out = rt
            .block_on(
                read::Read::default().run(serde_json::json!({ "path": path, "limit": READ_LINES })),
            )
            .unwrap();
        black_box(out.text.len());
    });
}

/// `edit`'s fuzzy-fallback normalization over a large file — the allocation-heavy path (NFKC + fold +
/// trailing-whitespace with an offset map back to the original). Fully on-thread, so alloc counting is
/// exact. Watch the alloc bytes across the two-pass `Vec<usize>` → fused `Vec<u32>` change.
#[divan::bench]
fn edit_normalize(bencher: Bencher) {
    // A realistic mostly-ASCII file with occasional trailing whitespace and a smart quote or two.
    let mut src = String::with_capacity(READ_LINES * 56);
    for i in 0..READ_LINES {
        if i % 20 == 0 {
            src.push_str(&format!("let s = \u{201c}line {i}\u{201d};   \n")); // smart quotes + trailing ws
        } else {
            src.push_str(&format!("    let x_{i} = compute(i, {i}) + adjust();\n"));
        }
    }
    bencher.bench_local(|| {
        let (norm, map) = edit::normalize_with_map(&src);
        black_box((norm.len(), map.len()));
    });
}

/// The same normalization *without* the offset map — the exact-match path's version. Sits next to
/// `edit_normalize` on purpose: the pair is an A/B in a single run, so both arms see the same machine
/// load. The gap between them is the entire prize for building the map lazily, and this box runs other
/// work, so a cross-invocation comparison would be measuring the neighbours.
#[divan::bench]
fn edit_normalize_only(bencher: Bencher) {
    let src = edit_src();
    bencher.bench_local(|| {
        let norm = edit::normalize_only(&src);
        black_box(norm.len());
    });
}

/// The same normalization over a **pure-ASCII** file — which is what source code overwhelmingly is.
/// NFKC is the identity on ASCII and `fold_char` has nothing to fold, so everything the normalizer
/// does here except trailing-whitespace stripping is provably wasted. Compare against
/// `edit_normalize_only` (identical size, a handful of non-ASCII chars) to see what the Unicode
/// machinery costs when there is no Unicode.
#[divan::bench]
fn edit_normalize_ascii(bencher: Bencher) {
    let mut src = String::with_capacity(READ_LINES * 56);
    for i in 0..READ_LINES {
        src.push_str(&format!("    let x_{i} = compute(i, {i}) + adjust();\n"));
    }
    bencher.bench_local(|| {
        let norm = edit::normalize_only(&src);
        black_box(norm.len());
    });
}

/// The realistic source file `edit_run_exact` edits — same shape `edit_normalize` normalizes.
fn edit_src() -> String {
    let mut src = String::with_capacity(READ_LINES * 56);
    for i in 0..READ_LINES {
        if i % 20 == 0 {
            src.push_str(&format!("let s = \u{201c}line {i}\u{201d};   \n"));
        } else {
            src.push_str(&format!("    let x_{i} = compute(i, {i}) + adjust();\n"));
        }
    }
    src
}

/// `edit`'s **whole** `run` over a pure-ASCII source file — the realistic common case twice over: the
/// file is ASCII (source code overwhelmingly is) and the `old_string` matches exactly (the model
/// reproduces text it just read). This is the number that represents what an `edit` tool call actually
/// costs in practice.
#[divan::bench(sample_count = 50)]
fn edit_run_exact_ascii(bencher: Bencher) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("subject.rs");
    let mut src = String::with_capacity(READ_LINES * 56);
    for i in 0..READ_LINES {
        src.push_str(&format!("    let x_{i} = compute(i, {i}) + adjust();\n"));
    }
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let tool = edit::Edit::new(dir.path());
    let p = path.to_str().unwrap().to_string();
    let mut n = 0usize;
    bencher.bench_local(|| {
        std::fs::write(&path, &src).unwrap();
        n += 1;
        let out = rt
            .block_on(tool.run(serde_json::json!({
                "path": p,
                "old_string": "    let x_1501 = compute(i, 1501) + adjust();",
                "new_string": format!("    let x_1501 = compute(i, 1501) + adjust(); // {n}"),
            })))
            .unwrap();
        black_box(out.text.len());
    });
}

/// `edit`'s whole `run` over a **large (~4 MB) pure-LF ASCII** file — the case M11 targets. A file
/// this size is where copying the entire body, building a 4×-file-size `Vec<u32>` offset map, and
/// re-validating UTF-8 on every edit actually hurt (the doc cites ~73 ms on 4 MB). With no `\r` in the
/// file the post-change path borrows the body and splices with raw offsets, skipping the copy + map +
/// validation entirely — watch the alloc bytes and time collapse here. Includes the `fs::write` of the
/// subject on both sides, so the delta is the CPU/alloc the edit itself no longer spends.
#[divan::bench(sample_count = 20)]
fn edit_run_large_lf(bencher: Bencher) {
    const LARGE_LF_LINES: usize = 75_000; // ~4 MB at ~56 bytes/line
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("large.rs");
    let mut src = String::with_capacity(LARGE_LF_LINES * 56);
    for i in 0..LARGE_LF_LINES {
        src.push_str(&format!("    let x_{i} = compute(i, {i}) + adjust();\n"));
    }
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let tool = edit::Edit::new(dir.path());
    let p = path.to_str().unwrap().to_string();
    let mut n = 0usize;
    bencher.bench_local(|| {
        std::fs::write(&path, &src).unwrap();
        n += 1;
        let out = rt
            .block_on(tool.run(serde_json::json!({
                "path": p,
                "old_string": "    let x_37500 = compute(i, 37500) + adjust();",
                "new_string": format!("    let x_37500 = compute(i, 37500) + adjust(); // {n}"),
            })))
            .unwrap();
        black_box(out.text.len());
    });
}

/// The same `run`, but over a file carrying a few non-ASCII characters, so it takes the general
/// Unicode normalizer. Kept beside `edit_run_exact_ascii` as the guard rail: the ASCII fast path must
/// not have *regressed* the path that still needs full NFKC.
#[divan::bench(sample_count = 50)]
fn edit_run_exact_unicode(bencher: Bencher) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("subject.rs");
    let src = edit_src();
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let tool = edit::Edit::new(dir.path());
    let p = path.to_str().unwrap().to_string();
    let mut n = 0usize;
    bencher.bench_local(|| {
        // Rewrite the subject each iteration so every sample edits the same pristine input.
        std::fs::write(&path, &src).unwrap();
        n += 1;
        let out = rt
            .block_on(tool.run(serde_json::json!({
                "path": p,
                "old_string": "    let x_1501 = compute(i, 1501) + adjust();",
                "new_string": format!("    let x_1501 = compute(i, 1501) + adjust(); // {n}"),
            })))
            .unwrap();
        black_box(out.text.len());
    });
}

/// `OutputAccumulator::snapshot` on a still-running (unfinished) command whose rolling tail is
/// saturated at `2 × max_bytes` — the ~100 ms live progress tick `bash` emits. `snapshot_text` used to
/// decode the *whole* tail before `truncate_tail` discarded the front half; now a live tick decodes
/// only the trailing window. On-thread with `persist_if_truncated = false`, so there's no temp-file
/// I/O — the alloc columns are purely the decode + truncate cost. Watch alloc bytes drop by ~half.
#[divan::bench]
fn snapshot_live_tick(bencher: Bencher) {
    let mut acc = output::OutputAccumulator::new();
    // Well past the rolling cap so the tail is saturated and `truncate_tail` really does drop the
    // front. Appended in a few big chunks (not per-line) to keep setup's spill writes cheap.
    let mut block: Vec<u8> = Vec::with_capacity(40 * 1024);
    while block.len() < 40 * 1024 {
        block.extend_from_slice(b"some ordinary line of streamed command output goes here\n");
    }
    for _ in 0..8 {
        acc.append(&block); // ~320 KiB total, tail saturates at 2 × 50 KiB
    }
    bencher.bench_local(|| {
        let snap = acc.snapshot(false); // unfinished → live-tick windowed decode
        black_box(snap.content.len());
    });
}

/// A streaming `bash`-style command emitting well past the ≤256 KiB head/tail capture window, with a
/// no-op live sink. The streaming path now skips the discarded `Capture` + `from_utf8_lossy` entirely,
/// so the alloc columns should shed the head/tail buffers and the two output strings. Time is
/// subprocess-spawn dominated (noisy); the alloc delta is the signal, not ns/iter.
#[divan::bench(sample_count = 20)]
fn exec_stream_capture(bencher: Bencher) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let runner = RealRunner;
    let sink: &(dyn Fn(&[u8]) + Send + Sync) = &|_chunk: &[u8]| {};
    let args = [
        "-c".to_string(),
        "head -c 400000 /dev/zero | tr '\\0' a".to_string(),
    ];
    bencher.bench_local(|| {
        let res = rt
            .block_on(runner.run_streaming(
                "sh",
                &args,
                None,
                std::time::Duration::from_secs(30),
                sink,
            ))
            .unwrap();
        black_box(res.stdout.len());
    });
}

/// A directory of 500 subdirectories, for the `ls` bench — directory entries are exactly where the old
/// `format!("{name}/")` allocated a second String per entry (and dropped the first).
fn dir_of_subdirs() -> &'static PathBuf {
    static D: OnceLock<(TempDir, PathBuf)> = OnceLock::new();
    &D.get_or_init(|| {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..500 {
            std::fs::create_dir(dir.path().join(format!("subdir_{i:04}"))).unwrap();
        }
        let root = dir.path().to_path_buf();
        (dir, root)
    })
    .1
}

/// `ls` of a directory of subdirectories: builds + sorts the entry list. On-thread, so alloc counting
/// is exact. Watch the alloc count across the `format!` → in-place suffix change.
#[divan::bench]
fn ls_dir(bencher: Bencher) {
    let path = dir_of_subdirs().to_str().unwrap().to_string();
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    bencher.bench_local(|| {
        let out = rt
            .block_on(ls::Ls::default().run(serde_json::json!({ "path": path, "limit": 1000 })))
            .unwrap();
        black_box(out.text.len());
    });
}

// ---------------------------------------------------------------------------------------------
// `search_rank` (session_store): case-insensitive substring match of one query against N sessions,
// each carrying a ~50 KB `search_text` corpus. The worst case is a query that matches *nothing*,
// because ranking then has to scan every field of every session — including the full 50 KB
// `search_text` — before giving up. The old code did `field.to_lowercase()` per field (a fresh
// 50 KB allocation per session, discarded), so this bench's alloc column is the whole point: it is
// exactly the allocate-and-throw-away that the allocation-free scan removes.
// ---------------------------------------------------------------------------------------------

const SEARCH_TEXT_BYTES: usize = 50_000;
/// A query no field contains — forces the full scan of every session's 50 KB corpus.
const NO_MATCH_QUERY: &str = "zqxj_never_present_needle";

/// `n` sessions, each with a ~50 KB mixed-case `search_text` that does **not** contain
/// [`NO_MATCH_QUERY`] — the realistic large-history worst case for `search_rank`.
fn session_corpus(n: usize) -> Vec<SessionMeta> {
    // Mixed case on purpose: the case-insensitive scan must lowercase-compare every byte, so upper
    // case in the haystack is part of the work under test.
    let unit = "The function threads its Result alias through the call site and re-exports IT. ";
    let mut text = String::with_capacity(SEARCH_TEXT_BYTES + unit.len());
    while text.len() < SEARCH_TEXT_BYTES {
        text.push_str(unit);
    }
    (0..n)
        .map(|i| {
            let mut m = SessionMeta::new(format!("/home/user/project_{i}"), "claude-sonnet-5");
            m.preview = Some("looking at the parser module".to_string());
            m.search_text = text.clone();
            m.updated_at = 1_700_000_000 + i as u64;
            m
        })
        .collect()
}

/// One search query over `n` sessions. The corpus is cloned per sample via `with_inputs` (untimed),
/// so the measured region is exactly `search_sessions` → `search_rank` × n, and the alloc columns
/// report allocs/query + bytes/query: `n` discarded 50 KB lowercased copies before the fix, ~none
/// after. Non-matching query = every field of every session is scanned to the end (worst case).
#[divan::bench(args = [20, 200], sample_count = 20)]
fn search_rank(bencher: Bencher, n: usize) {
    let corpus = session_corpus(n);
    bencher
        .with_inputs(|| corpus.clone())
        .bench_local_values(|sessions| {
            let out = search_sessions(sessions, Some(NO_MATCH_QUERY));
            black_box(out.len())
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
        let (p, _, _) = find::search(&job);
        black_box(p.len());
    });
}
