//! The golden vectors stay independently reproducible: `verify/pricing_reference.py`, an
//! exact-rational implementation of the pricing contract that shares only the rates with
//! `providers::pricing`, recomputes every vector's `expect` from the compiled table and must agree
//! with `verify/pricing_vectors.json` exactly. (`crates/providers/tests/pricing_vectors.rs` holds
//! the Rust pricer to the same file, so the two implementations are held to each other.)
//! `mise run pricing:vectors` rewrites the file from the reference.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use rates_sync::{canon, repo_root};
use std::process::Command;

#[test]
fn golden_vectors_reproduce_from_the_reference_pricer() {
    let root = repo_root();
    let table = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("rate-table.json");
    std::fs::write(&table, canon::spec_json()).unwrap();
    let out = Command::new("python3")
        .arg("-I")
        .arg(root.join("verify/pricing_reference.py"))
        .arg(&table)
        .arg(root.join("verify/pricing_vectors.json"))
        .arg("--check")
        .output()
        .expect("python3 runs the reference pricer");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out.status.success(),
        "the reference pricer disagrees with verify/pricing_vectors.json (if a vector's row or a \
         rate changed on purpose, `mise run pricing:vectors` rewrites it):\n{text}"
    );
}
