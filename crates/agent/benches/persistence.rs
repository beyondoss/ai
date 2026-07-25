// Bench target: `.unwrap()`/`.expect()` set up fixtures; not production code.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Session-persistence bench — the half of `crates/agent` that had no coverage at all, and the only
//! cost in the crate that grows without bound in a user's history.
//!
//! Two paths, measured against a *realistic* transcript (a coding session's lines are dominated by
//! tool results — a `read` output, a `grep` dump — not by prose):
//!
//! - `listing`: `SessionRepo::list_with_progress` → `scan_listings` → `read_listing`, which today
//!   fully deserializes **every** message of **every** session file to recover a preview, a count,
//!   and a max timestamp. Cost is O(total transcript bytes). Args are the per-session message count,
//!   so the slope across them *is* the scaling claim.
//! - `append_new`: one turn's durable append, on a **real ext4/NVMe temp dir** (`TempDir::new_in`
//!   under the repo, not `/tmp` — which is tmpfs here, where `sync_all` is nearly free and would
//!   flatter any CPU-side fix). This is the number that says whether shaving the serializer is
//!   visible at all next to the fsync, or whether it drowns.
//! - `serialize_entry`: the serializer in isolation, both strategies, so the CPU delta is legible
//!   even when `append_new`'s fsync hides it. `Entry`/`write_line` are private to `session_store`
//!   (deliberately), so — following `benches/serve_events.rs`'s own precedent — this replicates both
//!   strategies against the same public `agent_core::Message` payload the real `Entry` wraps.
//!
//! Run: `cargo bench -p beyond-ai-agent --bench persistence`.

use std::hint::black_box;
use std::path::{Path, PathBuf};

use agent_core::{ContentBlock, Message};
use beyond_ai_agent::session_store::{SessionMeta, SessionRepo, SessionStore};
use divan::Bencher;
use serde::Serialize;
use serde_json::Value;
use tempfile::TempDir;

#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

/// A ~64 KB `read`-tool output — the single most common shape of bytes in a coding transcript, and
/// the thing `read_listing` currently materializes in full only to throw away.
fn big_tool_result() -> String {
    "    1\tuse std::io;\n".repeat(3200)
}

/// One turn as it actually lands on disk: a user prompt, an assistant turn that *both* narrates and
/// calls a tool, and the tool result carrying the bulk of the bytes.
///
/// The assistant prose is not decoration. `read_listing` accumulates a 50,000-char search corpus from
/// every user/assistant **text** block, and a transcript's text is overwhelmingly the assistant's —
/// user prompts are one line, tool results contribute no text at all. A workload with no assistant
/// prose never fills that corpus, which silently changes which branch of the scan is under test. Ask
/// a realistic question or you measure a fictional one.
fn turn(i: usize) -> Vec<Message> {
    let prose = format!(
        "Looking at file {i} now. The function it defines threads its error type through a \
         `Result` alias, which is why the call site two modules up compiles even though the \
         signatures look mismatched at a glance. I'll read it and confirm before changing anything, \
         since the alias is re-exported and a change here would ripple. "
    )
    .repeat(3);
    vec![
        Message::user(format!("please look at file number {i} and explain it")),
        Message::assistant(vec![
            ContentBlock::Text {
                text: prose.into(),
                id: None,
                phase: None,
            },
            ContentBlock::ToolUse {
                id: format!("toolu_{i:06}"),
                name: "read".to_string(),
                input: serde_json::json!({ "path": format!("src/file_{i}.rs") }),
                thought_signature: None,
            },
        ]),
        Message::tool_result(format!("toolu_{i:06}"), big_tool_result(), false),
    ]
}

/// A temp dir on the **real** filesystem (ext4/NVMe here), not `/tmp` (tmpfs). `sync_all` on tmpfs is
/// essentially a no-op, which would make every durability cost vanish and every CPU-side fix look
/// like a win. The append path's whole question is "is the serializer visible next to the fsync?" —
/// asking it on tmpfs answers a question nobody has.
fn real_disk_tempdir() -> TempDir {
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    let scratch = here.join("target").join("bench-scratch");
    std::fs::create_dir_all(&scratch).unwrap();
    TempDir::new_in(&scratch).unwrap()
}

/// Build a session directory holding one session file of `turns` turns, and hand back the repo dir.
/// Kept alive by the returned `TempDir`.
fn session_dir(turns: usize) -> (TempDir, PathBuf) {
    let dir = real_disk_tempdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo
        .create(SessionMeta::new("/repo", "claude-sonnet-5"))
        .unwrap();
    let mut messages = Vec::new();
    for i in 0..turns {
        messages.extend(turn(i));
        store.append_new(&messages).unwrap();
    }
    let path = dir.path().to_path_buf();
    (dir, path)
}

// ---------------------------------------------------------------------------------------------
// The constraint: listing cost is O(total transcript bytes), because every line is fully parsed.
// ---------------------------------------------------------------------------------------------

/// The **cold** listing: the sidecar index is thrown away before every sample, so each one pays the
/// full O(total transcript bytes) parse. This is what every listing cost before the index existed, and
/// what the first listing after a session is written still costs.
///
/// `turns` × 3 messages, of which every third carries ~64 KB. 20 turns ≈ 1.4 MB, 200 turns ≈ 14 MB —
/// an ordinary long coding session. The slope from one to the other is the scaling claim.
#[divan::bench(args = [20, 200], sample_count = 10)]
fn listing_cold(bencher: Bencher, turns: usize) {
    let (_keep, dir) = session_dir(turns);
    let repo = SessionRepo::open(&dir).unwrap();
    bencher.bench_local(|| {
        let _ = std::fs::remove_file(dir.join(".listings.json"));
        let metas = repo.list_with_progress(|_, _| {}).unwrap();
        black_box(metas.len());
    });
}

/// The **warm** listing: the sidecar index is valid, so every file is answered by one `stat` instead of
/// a full parse. This is the steady state — a session's `(size, mtime)` only moves when it's actually
/// appended to, so at most the one session you're currently talking to is ever a miss.
///
/// The gap between this and `listing_cold` is the whole point of the index, and it should widen with
/// transcript size, because this side doesn't depend on it at all.
#[divan::bench(args = [20, 200], sample_count = 10)]
fn listing_warm(bencher: Bencher, turns: usize) {
    let (_keep, dir) = session_dir(turns);
    let repo = SessionRepo::open(&dir).unwrap();
    // Prime the index once, outside the measured region.
    black_box(repo.list_with_progress(|_, _| {}).unwrap().len());
    bencher.bench_local(|| {
        let metas = repo.list_with_progress(|_, _| {}).unwrap();
        black_box(metas.len());
    });
}

/// The **floor** for any listing scan that still reads the file: open it, pull every line, touch
/// nothing else. Whatever `listing` costs above this number is parsing; whatever it costs *at* this
/// number is unavoidable as long as the bytes are read at all. This is the control that decides
/// whether a faster parser is worth anything, or whether the only real fix is to stop reading.
#[divan::bench(args = [200], sample_count = 10)]
fn listing_read_floor(bencher: Bencher, turns: usize) {
    use std::io::BufRead;
    let (_keep, dir) = session_dir(turns);
    let file = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|x| x == "jsonl"))
        .unwrap();
    bencher.bench_local(|| {
        let mut reader = std::io::BufReader::new(std::fs::File::open(&file).unwrap());
        let mut buf = Vec::new();
        let mut lines = 0usize;
        while reader.read_until(b'\n', &mut buf).unwrap() > 0 {
            lines += 1;
            buf.clear();
        }
        black_box(lines);
    });
}

// ---------------------------------------------------------------------------------------------
// The append path: does the serializer matter next to `sync_all`?
// ---------------------------------------------------------------------------------------------

/// One turn appended to an already-long session — `append_new`'s own `messages.len() <= persisted`
/// guard means only the new messages are serialized, so this measures exactly one turn's marginal
/// durable cost: serialize ~64 KB + `write_all` + `flush` + `sync_all`, on a real disk.
#[divan::bench(sample_count = 20)]
fn append_new(bencher: Bencher) {
    let dir = real_disk_tempdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo
        .create(SessionMeta::new("/repo", "claude-sonnet-5"))
        .unwrap();
    let mut messages = Vec::new();
    for i in 0..20 {
        messages.extend(turn(i));
    }
    store.append_new(&messages).unwrap();
    let mut next = 20usize;
    bencher.bench_local(|| {
        messages.extend(turn(next));
        next += 1;
        store.append_new(&messages).unwrap();
    });
}

// ---------------------------------------------------------------------------------------------
// The serializer in isolation, both strategies. `Entry` is private; this replicates its shape.
// ---------------------------------------------------------------------------------------------

/// Structurally what `Entry::Message` is: the envelope fields plus a **cloned** `Message`. The clone
/// is the point — `session_store::append_new` clones the message into the `Entry` purely to serialize
/// it, so it's part of the cost being measured.
#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OwnedEntry {
    Message {
        id: Option<String>,
        parent_id: Option<String>,
        timestamp: u64,
        message: Message,
    },
}

/// The same envelope, borrowing the `Message` instead of cloning it — the fix under test.
#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum BorrowedEntry<'a> {
    Message {
        id: Option<&'a str>,
        parent_id: Option<&'a str>,
        timestamp: u64,
        message: &'a Message,
    },
}

fn payload() -> Message {
    Message::tool_result("toolu_000001", big_tool_result(), false)
}

/// Today's `write_line`: clone the message into an owned `Entry`, serialize that to an owned `Value`
/// tree, then serialize the tree to a `String`. Three full passes over ~64 KB.
#[divan::bench]
fn serialize_entry_clone_to_value_to_string(bencher: Bencher) {
    let msg = payload();
    bencher.bench_local(|| {
        let entry = OwnedEntry::Message {
            id: Some("abc".to_string()),
            parent_id: Some("def".to_string()),
            timestamp: 1_700_000_000,
            message: msg.clone(),
        };
        let v = serde_json::to_value(&entry).unwrap();
        let mut buf: Vec<u8> = Vec::new();
        use std::io::Write;
        writeln!(&mut buf, "{}", Value::to_string(&v)).unwrap();
        black_box(buf.len());
    });
}

/// The fix: borrow the message, serialize straight to the writer. One pass, no `Value` tree, no clone.
#[divan::bench]
fn serialize_entry_borrow_to_writer(bencher: Bencher) {
    let msg = payload();
    bencher.bench_local(|| {
        let entry = BorrowedEntry::Message {
            id: Some("abc"),
            parent_id: Some("def"),
            timestamp: 1_700_000_000,
            message: &msg,
        };
        let mut buf: Vec<u8> = Vec::new();
        serde_json::to_writer(&mut buf, &entry).unwrap();
        buf.push(b'\n');
        black_box(buf.len());
    });
}

/// Keep `SessionStore` in the import graph even if a future edit drops its only other use.
#[allow(dead_code)]
fn _assert_store_type(_: &SessionStore) {}
