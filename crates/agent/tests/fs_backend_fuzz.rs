//! Randomized differential testing: `LocalFs` vs `ShellFs`, over generated trees and queries.
//!
//! The hand-written parity suite (`fs_backend_parity.rs`) checks the cases *I thought of*. That is
//! exactly its weakness — it cannot find a divergence nobody anticipated, and two of the divergences
//! that do exist were found by accident rather than by design. This file attacks the same claim from
//! the other side: generate trees and queries pseudo-randomly, run both backends, and diff.
//!
//! Deterministic by construction. The PRNG is a fixed-seed xorshift, so a failure reproduces exactly
//! and CI does not flake; `FS_FUZZ_SEED` and `FS_FUZZ_CASES` widen the search locally without making
//! the committed run non-reproducible.
//!
//! ## What is compared, and what is deliberately excluded
//!
//! Only the **`rg` rung** is diffed strictly. The POSIX fallback has two known, asserted divergences
//! (gitignore blindness, and `grep -I` dropping non-UTF-8 files), so including it here would just
//! rediscover those on every run and drown a real finding.
//!
//! **Truncated results are excluded from strict comparison.** `LocalFs`'s parallel walk quits the
//! instant it has `limit` matches, so which matches survive is scheduling-dependent — the module
//! documents this. Comparing membership there asserts a coin flip. Those cases instead assert the
//! weaker invariant that actually holds: the same *count*, and the same truncation notice.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::Path;
use std::sync::Arc;

use beyond_ai_agent::tools::exec::RealRunner;
use beyond_ai_agent::tools::fs::FsBackend;
use beyond_ai_agent::tools::fs::local::LocalFs;
use beyond_ai_agent::tools::fs::shell::{SearchEngine, ShellFs};
use beyond_ai_agent::tools::{ToolConfig, default_registry_with_config};
use serde_json::{Value, json};

/// xorshift64*, so a seed reproduces a run exactly on any machine.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len())]
    }
    fn chance(&mut self, one_in: usize) -> bool {
        self.below(one_in) == 0
    }
}

/// Filename fragments, weighted toward the shapes that break naive implementations.
const NAME_PARTS: &[&str] = &[
    "alpha",
    "beta",
    "gamma",
    "src",
    "lib",
    "mod",
    "a b",
    "odd:colon",
    "we\nird",
    "dot.ted",
    "UPPER",
    "ünï",
    "漢字",
    "'quote",
    "$dollar",
    "star*",
    "dash-",
    "under_",
    "#hash",
    "back\\slash",
];
const EXTS: &[&str] = &["rs", "txt", "md", "toml", "", "RS", "tar.gz"];

/// Content shapes, including the ones that decide encoding behavior.
fn body(rng: &mut Rng, needle: &str) -> Vec<u8> {
    match rng.below(8) {
        0 => Vec::new(),                                          // empty
        1 => format!("{needle}\r\nsecond line\r\n").into_bytes(), // CRLF
        2 => {
            let mut v = b"caf\xe9 ".to_vec(); // invalid UTF-8
            v.extend_from_slice(format!("{needle} tail\n").as_bytes());
            v
        }
        3 => {
            let mut v = vec![0u8, 0u8]; // binary (NUL) — must be skipped by both
            v.extend_from_slice(format!(" {needle} ").as_bytes());
            v.push(0);
            v
        }
        4 => format!("{}{needle}\n", "x".repeat(1200)).into_bytes(), // long line, clipped
        5 => format!("{}{needle}\n", "漢".repeat(700)).into_bytes(), // multi-byte clipping
        6 => {
            let n = rng.below(40) + 1;
            format!("{needle}\n").repeat(n).into_bytes() // match-dense
        }
        _ => format!("fn f() {{ {needle} }}\nfn g() {{}}\n// trailing\n").into_bytes(),
    }
}

/// Build a random tree. Returns its root.
fn tree(rng: &mut Rng) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let mut dirs = vec![root.to_path_buf()];

    for _ in 0..rng.below(4) {
        let mut d = root.to_path_buf();
        for _ in 0..(1 + rng.below(2)) {
            d = d.join(rng.pick(NAME_PARTS));
        }
        if std::fs::create_dir_all(&d).is_ok() {
            dirs.push(d);
        }
    }
    for _ in 0..(3 + rng.below(12)) {
        let parent = rng.pick(&dirs).clone();
        let ext = rng.pick(EXTS);
        let stem = rng.pick(NAME_PARTS);
        let name = if ext.is_empty() {
            stem.to_string()
        } else {
            format!("{stem}.{ext}")
        };
        let _ = std::fs::write(parent.join(name), body(rng, "NEEDLE"));
    }
    // Sometimes a .gitignore, sometimes a dotfile, sometimes symlinks — each changes walk behavior.
    if rng.chance(2) {
        let _ = std::fs::write(root.join(".gitignore"), "ignored/\n*.log\n");
        let _ = std::fs::create_dir_all(root.join("ignored"));
        let _ = std::fs::write(root.join("ignored/inner.rs"), b"NEEDLE ignored\n");
        let _ = std::fs::write(root.join("noisy.log"), b"NEEDLE logged\n");
    }
    if rng.chance(3) {
        let _ = std::fs::write(root.join(".hidden"), b"NEEDLE hidden\n");
    }
    if rng.chance(3) {
        let target = root.join("alpha.rs");
        let _ = std::fs::write(&target, b"NEEDLE linked\n");
        let _ = std::os::unix::fs::symlink(&target, root.join("link.rs"));
        let _ = std::os::unix::fs::symlink(root.join("no-such"), root.join("broken.rs"));
    }
    dir
}

/// A random tool call against `root`.
fn query(rng: &mut Rng, root: &Path) -> (String, Value) {
    let p = root.to_str().unwrap().to_string();
    match rng.below(4) {
        0 => {
            let mut q = json!({
                "pattern": rng.pick(&["NEEDLE", "needle", "NEED.E", "fn .", "zzz-none", "^f"]),
                "path": p,
            });
            if rng.chance(3) {
                q["glob"] = json!(rng.pick(&["*.rs", "!*.rs", "*.txt", "**/*.md"]));
            }
            if rng.chance(3) {
                q["ignore_case"] = json!(true);
            }
            if rng.chance(4) {
                q["literal"] = json!(true);
            }
            if rng.chance(3) {
                q["context"] = json!(rng.below(3));
            }
            if rng.chance(3) {
                q["limit"] = json!(1 + rng.below(200));
            }
            ("grep".into(), q)
        }
        1 => {
            let mut q = json!({
                "pattern": rng.pick(&["*.rs", "*", "*.txt", "src", "**/*.md", "alpha*", "*.RS"]),
                "path": p,
            });
            if rng.chance(3) {
                q["limit"] = json!(1 + rng.below(50));
            }
            ("find".into(), q)
        }
        2 => {
            let mut q = json!({ "path": p });
            if rng.chance(2) {
                q["all"] = json!(true);
            }
            if rng.chance(3) {
                q["limit"] = json!(1 + rng.below(20));
            }
            ("ls".into(), q)
        }
        _ => {
            // Read a real file from the tree when there is one, else a miss.
            let mut files: Vec<_> = walk_files(root);
            files.sort();
            let target = if files.is_empty() || rng.chance(6) {
                root.join("definitely-absent.txt")
            } else {
                files[rng.below(files.len())].clone()
            };
            let mut q = json!({ "path": target.to_str().unwrap() });
            if rng.chance(3) {
                q["offset"] = json!(1 + rng.below(5));
            }
            if rng.chance(3) {
                q["limit"] = json!(1 + rng.below(10));
            }
            ("read".into(), q)
        }
    }
}

fn walk_files(root: &Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.flatten() {
            let Ok(ft) = e.file_type() else { continue };
            if ft.is_dir() {
                stack.push(e.path());
            } else if ft.is_file() {
                out.push(e.path());
            }
        }
    }
    out
}

/// Run one call through a registry built on `backend`, flattening errors into comparable text so a
/// divergence in *failure* is caught just as strictly as one in success.
async fn run(backend: Arc<dyn FsBackend>, tool: &str, input: Value) -> String {
    let reg = default_registry_with_config(&ToolConfig {
        fs_backend: Some(backend),
        ..ToolConfig::new()
    });
    match reg.get(tool).expect("tool").run(input).await {
        Ok(o) => format!("OK|{}|{}", o.images.len(), o.text),
        Err(e) => format!("ERR|{e}"),
    }
}

fn truncated(s: &str) -> bool {
    s.contains("limit ") && s.contains("reached")
}

fn match_count(s: &str) -> usize {
    s.lines().filter(|l| l.contains(": ")).count()
}

#[tokio::test]
async fn randomized_trees_and_queries_agree_across_backends() {
    /// Accepts `0x…` as well as decimal — the failure message below prints the seed in hex, so a
    /// decimal-only parser would hand out a reproduce command that silently re-runs the default seed
    /// instead of the failing one. (It did exactly that before this was fixed.)
    fn parse_seed(s: &str) -> Option<u64> {
        let s = s.trim();
        match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
            Some(hex) => u64::from_str_radix(hex, 16).ok(),
            None => s.parse().ok(),
        }
    }
    let seed: u64 = std::env::var("FS_FUZZ_SEED")
        .ok()
        .and_then(|s| parse_seed(&s))
        .unwrap_or(0xB3_1D_5A_7C_00_11_22_33);
    let cases: usize = std::env::var("FS_FUZZ_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);

    let shell = ShellFs::connect(Arc::new(RealRunner)).await;
    assert_eq!(
        shell.capabilities().search_engine(),
        SearchEngine::Ripgrep,
        "this host has no `rg`; the strict rung cannot be fuzzed here"
    );
    let shell: Arc<dyn FsBackend> = Arc::new(shell);
    let local: Arc<dyn FsBackend> = Arc::new(LocalFs::new());

    let mut rng = Rng::new(seed);
    let mut compared = 0usize;
    let mut skipped_truncated = 0usize;
    let mut failures = Vec::new();

    for case in 0..cases {
        let dir = tree(&mut rng);
        let (tool, input) = query(&mut rng, dir.path());

        let l = run(local.clone(), &tool, input.clone()).await;
        let s = run(shell.clone(), &tool, input.clone()).await;

        if truncated(&l) || truncated(&s) {
            // Documented non-determinism: assert the invariant that does hold.
            skipped_truncated += 1;
            if match_count(&l) != match_count(&s) || truncated(&l) != truncated(&s) {
                failures.push(format!(
                    "case {case} (truncated): counts/notice differ\n  tool={tool} input={input}\n  \
                     local_matches={} shell_matches={} local_trunc={} shell_trunc={}",
                    match_count(&l),
                    match_count(&s),
                    truncated(&l),
                    truncated(&s)
                ));
            }
            continue;
        }

        compared += 1;
        if l != s {
            failures.push(format!(
                "case {case}: MISMATCH\n  tool={tool}\n  input={input}\n  --- local ---\n{}\n  \
                 --- shell ---\n{}",
                &l[..l.len().min(1200)],
                &s[..s.len().min(1200)]
            ));
        }
    }

    eprintln!(
        "fuzz seed={seed:#x} cases={cases}: {compared} compared strictly, \
         {skipped_truncated} truncated (count-only)"
    );
    assert!(
        failures.is_empty(),
        "{} divergence(s) found. Reproduce with FS_FUZZ_SEED={seed:#x}\n\n{}",
        failures.len(),
        failures.join("\n\n")
    );
    assert!(
        compared > cases / 4,
        "too few strictly-compared cases ({compared}/{cases}) — the generator is producing mostly \
         truncated results and is not testing what it claims"
    );
}
