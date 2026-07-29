//! Claim 3: **how much would remoting the toolset actually cost?**
//!
//! The plan asserts that per-call transport overhead is not the constraint, on the reasoning that an
//! LLM turn takes 2–20 s while an exec takes 20–60 ms. That was an argument, not a measurement, and it
//! is the claim that decides whether a remote backend is worth building at all — so this file measures
//! the input it depends on: **how many filesystem operations does one turn's worth of tool calls
//! actually issue?**
//!
//! It counts operations rather than timing them. A wall-clock number here would measure this host's
//! page cache, not a sandbox round trip; the honest quantity is the op count, which the reader
//! multiplies by whatever a round trip costs on the transport they are considering.
//!
//! The failure branch is real. If a turn issues ~20 ops, remoting costs well under a second against a
//! multi-second turn and a remote backend is worth building. If it issues hundreds, the right answer
//! is to run the agent *inside* the sandbox instead and never build one.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use beyond_ai_agent::tools::fs::local::LocalFs;
use beyond_ai_agent::tools::fs::{
    DirEntry, FsBackend, FsError, GlobOutcome, GlobQuery, Meta, PathWorld, SearchOutcome,
    SearchQuery,
};
use beyond_ai_agent::tools::{ToolConfig, default_registry_with_config};
use serde_json::json;

/// Wraps a real backend and counts every call that would become a round trip.
///
/// Delegates to [`LocalFs`] rather than faking results, so the tools take exactly the branches they
/// take in production — a counter over stubbed responses would measure a code path nobody runs.
struct Counting {
    inner: LocalFs,
    ops: Arc<AtomicUsize>,
}

/// A backend paired with the counter recording its round trips.
type Counted = (Arc<dyn FsBackend>, Arc<AtomicUsize>);

/// One tool call: its name and its JSON input.
type Call<'a> = (&'a str, serde_json::Value);

impl Counting {
    /// Not `-> Self`: the caller needs the counter as well as the backend, and handing back both
    /// keeps the `Arc<dyn FsBackend>` erasure in one place.
    fn paired() -> Counted {
        let ops = Arc::new(AtomicUsize::new(0));
        (
            Arc::new(Self {
                inner: LocalFs::new(),
                ops: ops.clone(),
            }),
            ops,
        )
    }

    fn tick(&self) {
        self.ops.fetch_add(1, Ordering::Relaxed);
    }
}

#[async_trait]
impl FsBackend for Counting {
    fn world(&self) -> PathWorld {
        PathWorld::Local
    }
    async fn search(&self, q: &SearchQuery) -> Result<SearchOutcome, FsError> {
        self.tick();
        self.inner.search(q).await
    }
    async fn stat(&self, path: &Path) -> Result<Option<Meta>, FsError> {
        self.tick();
        self.inner.stat(path).await
    }
    async fn read_bytes(&self, path: &Path, offset: u64, max: usize) -> Result<Vec<u8>, FsError> {
        self.tick();
        self.inner.read_bytes(path, offset, max).await
    }
    async fn write_bytes(&self, path: &Path, bytes: &[u8]) -> Result<(), FsError> {
        self.tick();
        self.inner.write_bytes(path, bytes).await
    }
    async fn write_if_unchanged(
        &self,
        path: &Path,
        bytes: &[u8],
        expected: Option<std::time::SystemTime>,
    ) -> Result<bool, FsError> {
        self.tick();
        self.inner.write_if_unchanged(path, bytes, expected).await
    }
    async fn create_dir_all(&self, path: &Path) -> Result<(), FsError> {
        self.tick();
        self.inner.create_dir_all(path).await
    }
    async fn list_dir(
        &self,
        path: &Path,
        cap: usize,
        include_hidden: bool,
    ) -> Result<Vec<DirEntry>, FsError> {
        self.tick();
        self.inner.list_dir(path, cap, include_hidden).await
    }
    async fn glob(&self, q: &GlobQuery) -> Result<GlobOutcome, FsError> {
        self.tick();
        self.inner.glob(q).await
    }
}

/// A small but realistic source tree.
fn project() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    for (name, body) in [
        ("main.rs", "fn main() {\n    let value = compute();\n}\n"),
        ("lib.rs", "pub fn compute() -> u32 {\n    41\n}\n"),
        ("util.rs", "pub fn helper() {}\n"),
    ] {
        std::fs::write(src.join(name), body).unwrap();
    }
    std::fs::write(dir.path().join("README.md"), "# demo\n").unwrap();
    dir
}

/// Run a sequence of tool calls and report how many backend operations they cost.
async fn ops_for(calls: &[Call<'_>]) -> usize {
    let (backend, ops) = Counting::paired();
    let reg = default_registry_with_config(&ToolConfig {
        fs_backend: Some(backend),
        ..ToolConfig::new()
    });
    for (name, input) in calls {
        let tool = reg.get(name).expect("tool registered");
        // A call that legitimately errors still cost its operations; that is what we're counting.
        let _ = tool.run(input.clone()).await;
    }
    ops.load(Ordering::Relaxed)
}

#[tokio::test]
async fn a_representative_turn_costs_a_bounded_number_of_operations() {
    // A turn shaped like real work: orient, search, read two files, then edit one.
    let dir = project();
    let root = dir.path().to_str().unwrap().to_string();
    let main_rs = dir.path().join("src/main.rs").to_str().unwrap().to_string();
    let lib_rs = dir.path().join("src/lib.rs").to_str().unwrap().to_string();

    let ops = ops_for(&[
        ("ls", json!({ "path": root })),
        ("find", json!({ "pattern": "*.rs", "path": root })),
        ("grep", json!({ "pattern": "compute", "path": root })),
        ("read", json!({ "path": main_rs })),
        ("read", json!({ "path": lib_rs.clone() })),
        (
            "edit",
            json!({ "path": lib_rs, "old_string": "41", "new_string": "42" }),
        ),
    ])
    .await;

    // The measured figure, pinned so a future change that multiplies round trips per call fails here
    // rather than quietly making a remote backend a worse idea than it looks.
    eprintln!("representative 6-call turn: {ops} backend operations");
    assert!(
        ops <= 20,
        "a 6-call turn should cost well under 20 backend operations; measured {ops}. If this grew, \
         re-run the Phase 2 cost argument before building a remote backend."
    );
    assert!(
        ops >= 6,
        "sanity: each call must cost at least one op, got {ops}"
    );
}

#[tokio::test]
async fn each_tool_costs_the_expected_number_of_operations() {
    // Per-tool costs, so a regression names the tool that got more expensive rather than a total that
    // drifted for unclear reasons. These are the multipliers the Phase 2 latency estimate rests on.
    let dir = project();
    let root = dir.path().to_str().unwrap().to_string();
    let lib_rs = dir.path().join("src/lib.rs").to_str().unwrap().to_string();
    let fresh = dir.path().join("new.txt").to_str().unwrap().to_string();

    let cases: Vec<(&str, Vec<Call<'_>>, usize)> = vec![
        ("ls", vec![("ls", json!({ "path": root.clone() }))], 2),
        (
            "find",
            vec![("find", json!({ "pattern": "*.rs", "path": root.clone() }))],
            2,
        ),
        (
            "grep",
            vec![(
                "grep",
                json!({ "pattern": "compute", "path": root.clone() }),
            )],
            1,
        ),
        ("read", vec![("read", json!({ "path": lib_rs.clone() }))], 3),
        (
            "write",
            vec![("write", json!({ "path": fresh, "content": "hello\n" }))],
            3,
        ),
        (
            "edit",
            vec![(
                "edit",
                json!({ "path": lib_rs, "old_string": "41", "new_string": "43" }),
            )],
            3,
        ),
    ];

    for (name, calls, expected) in cases {
        let ops = ops_for(&calls).await;
        eprintln!("{name}: {ops} backend operations");
        assert_eq!(
            ops, expected,
            "{name} changed its backend-operation count (was {expected}, now {ops}) — this is the \
             per-call multiplier the remote-backend cost estimate depends on, so a change here \
             should be deliberate"
        );
    }
}

#[tokio::test]
async fn the_cost_estimate_for_a_remote_turn_is_stated_in_the_output() {
    // Not an assertion so much as a recorded conclusion: turn the measured op count into the latency
    // figure the Phase 2 decision actually needs, at a round-trip cost the reader can re-scale.
    let dir = project();
    let root = dir.path().to_str().unwrap().to_string();
    let lib_rs = dir.path().join("src/lib.rs").to_str().unwrap().to_string();
    let ops = ops_for(&[
        ("ls", json!({ "path": root.clone() })),
        ("grep", json!({ "pattern": "compute", "path": root })),
        ("read", json!({ "path": lib_rs.clone() })),
        (
            "edit",
            json!({ "path": lib_rs, "old_string": "41", "new_string": "44" }),
        ),
    ])
    .await;
    for rt_ms in [20u64, 40, 60] {
        eprintln!(
            "4-call turn: {ops} ops x {rt_ms}ms round trip = {}ms of transport",
            ops as u64 * rt_ms
        );
    }
    assert!(ops > 0);
}
