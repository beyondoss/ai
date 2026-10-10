//! `rates-sync`: the providers crate's rate table, generated from its primary sources.
//!
//! - `fetch` downloads every source ([`sources::all`]) and writes its normalized snapshot and the
//!   manifest under `verify/rates_sources/`.
//! - `generate` is offline and deterministic: it reads the snapshots and the hand-entered rules
//!   in `verify/catalog_truth.toml` ([`spec`]), parses each source strictly ([`vendors`]),
//!   cross-checks every pair of sources for the same fact, and renders
//!   `crates/providers/src/rates/generated.rs` ([`emit`]).
//!
//! See `crates/providers/ARCHITECTURE.md`, "Rate data and versions".

pub mod audit;
pub mod build;
pub mod canon;
pub mod classify;
pub mod dec;
pub mod diff;
pub mod emit;
pub mod fetch;
pub mod invoice;
pub mod json;
pub mod md;
pub mod snapshot;
pub mod sources;
pub mod spec;
pub mod vendors;

use std::path::{Path, PathBuf};

pub type Result<T> = std::result::Result<T, String>;

/// The generated module, relative to the repo root.
pub const GENERATED: &str = "crates/providers/src/rates/generated.rs";
/// The snapshot store, relative to the repo root.
pub const SOURCES: &str = "verify/rates_sources";
/// The hand-entered rules, relative to the repo root.
pub const TRUTH: &str = "verify/catalog_truth.toml";

pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

pub fn read_spec(root: &Path) -> Result<spec::Spec> {
    let path = root.join(TRUTH);
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    spec::parse(&text)
}

/// The generated module's text, from the snapshots in `sources_dir` and the truth file.
pub fn generate(root: &Path, sources_dir: &Path) -> Result<String> {
    let spec = read_spec(root)?;
    let defs = sources::all(&spec.openrouter_slugs());
    let store = snapshot::load(sources_dir, &defs)?;
    let table = build::build(&store, &spec)?;
    emit::render(&table)
}
