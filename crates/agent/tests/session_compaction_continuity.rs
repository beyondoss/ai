//! A compacted session keeps its on-disk history through whatever comes next: a session rebuilt
//! without the store's compaction provenance, and a compaction persisted while the store lagged the
//! session. Neither may fall back to rewriting the file — a rewrite drops every side record (model
//! changes, labels, titles, earlier compaction records) and re-mints every id.

// Test target: `.unwrap()`/`panic!` are how a test reports a failed setup.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::fs;
use std::path::Path;

use agent_core::{ContentBlock, Message, Session};
use beyond_ai_agent::session_store::{SessionMeta, SessionStore};
use tempfile::TempDir;

fn turn(s: &mut Session, i: usize) {
    s.push(Message::user(format!("u{i}")));
    s.push(Message::assistant(vec![ContentBlock::text(format!(
        "a{i}"
    ))]));
}

fn texts(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .map(|m| match &m.content[0] {
            ContentBlock::Text { text, .. } => text.to_string(),
            other => format!("{other:?}"),
        })
        .collect()
}

fn compaction_records(path: &Path) -> usize {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| l.contains(r#""type":"compaction""#))
        .count()
}

/// Compact `s` the way `Agent::compact` does: a summary in place of `messages[..first_kept]`, the
/// generation bumped, provenance recorded. Each summary is distinct, as a real one is.
fn compact(s: &mut Session, first_kept: usize) {
    static ROUND: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let round = ROUND.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let summary = format!("earlier work, round {round}");
    agent_core::compaction::apply_summary(s, first_kept, &summary, 1000);
    s.compaction.compactions += 1;
    s.compaction.read_files = vec!["src/important.rs".into()];
}

/// Hunt t8: a session rebuilt from a compacted store's messages but with default provenance (what
/// `serve`'s branch switch used to hand back) persists as the continuation it is. The store used to
/// read the reset generation as a compaction and write a spurious round.
#[test]
fn a_session_rebuilt_without_the_stores_generation_appends_instead_of_compacting_again() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.jsonl");
    let mut store = SessionStore::create(path.clone(), SessionMeta::new("/w", "model-a")).unwrap();
    let mut s = Session::new();
    for i in 0..4 {
        turn(&mut s, i);
    }
    store
        .persist(&s.messages, &s.compaction, &s.edited)
        .unwrap();
    compact(&mut s, 4);
    store
        .persist(&s.messages, &s.compaction, &s.edited)
        .unwrap();
    store.record_model_change("model-b").unwrap();
    let tip = store.active_ids().last().unwrap().clone();
    store.set_label(&tip, Some("good state")).unwrap();

    let summary = store.active_ids()[0].clone();
    let mut rebuilt = Session::new();
    rebuilt.messages = std::sync::Arc::new(store.switch_active(&summary).unwrap());
    rebuilt.push(Message::assistant(vec![ContentBlock::text("new reply")]));
    rebuilt.push(Message::user("new question"));
    let before = fs::read(&path).unwrap();
    store
        .persist(&rebuilt.messages, &rebuilt.compaction, &rebuilt.edited)
        .unwrap();
    assert!(fs::read(&path).unwrap().starts_with(&before));
    assert_eq!(compaction_records(&path), 1);
    drop(store);

    let (store, reopened) = SessionStore::open(path).unwrap();
    assert_eq!(reopened.messages.len(), 3);
    assert_eq!(reopened.compaction.read_files, ["src/important.rs"]);
    assert_eq!(reopened.compaction.compactions, 1);
    assert_eq!(store.get_label(&tip), Some("good state"));
}

/// The adopted generation still lets the next real compaction through: the store follows the
/// session's count from then on.
#[test]
fn after_adopting_a_sessions_generation_the_next_real_compaction_is_written() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.jsonl");
    let mut store = SessionStore::create(path.clone(), SessionMeta::new("/w", "m")).unwrap();
    let mut s = Session::new();
    for i in 0..4 {
        turn(&mut s, i);
    }
    store
        .persist(&s.messages, &s.compaction, &s.edited)
        .unwrap();
    compact(&mut s, 4);
    store
        .persist(&s.messages, &s.compaction, &s.edited)
        .unwrap();

    let mut rebuilt = Session::new();
    rebuilt.messages = s.messages.clone();
    turn(&mut rebuilt, 9);
    store
        .persist(&rebuilt.messages, &rebuilt.compaction, &rebuilt.edited)
        .unwrap();
    compact(&mut rebuilt, 3);
    store
        .persist(&rebuilt.messages, &rebuilt.compaction, &rebuilt.edited)
        .unwrap();
    assert_eq!(compaction_records(&path), 2);
    drop(store);
    let (_store, reopened) = SessionStore::open(path).unwrap();
    assert_eq!(texts(&reopened.messages), texts(&rebuilt.messages));
}

/// Hunt t3: a compaction persisted while the store lags the session (checkpoints failed) is an
/// append-only round. It used to see "nothing folded" by count and fall back to a full rewrite,
/// dropping the model change, the title and the label from disk.
#[test]
fn a_compaction_persisted_while_the_store_lagged_keeps_every_side_record() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.jsonl");
    let mut store = SessionStore::create(path.clone(), SessionMeta::new("/w", "model-a")).unwrap();
    let mut s = Session::new();
    turn(&mut s, 0);
    store
        .persist(&s.messages, &s.compaction, &s.edited)
        .unwrap();
    store.record_model_change("model-b").unwrap();
    store.set_title("my title").unwrap();
    let a0 = store.active_ids()[1].clone();
    store.set_label(&a0, Some("bookmark")).unwrap();
    // Two more turns never reach disk, then a compaction folds the first.
    turn(&mut s, 1);
    turn(&mut s, 2);
    compact(&mut s, 2);
    let before = fs::read(&path).unwrap();
    store
        .persist(&s.messages, &s.compaction, &s.edited)
        .unwrap();
    assert!(fs::read(&path).unwrap().starts_with(&before));
    drop(store);

    let (store, reopened) = SessionStore::open(path).unwrap();
    assert_eq!(texts(&reopened.messages), texts(&s.messages));
    assert_eq!(store.meta().title.as_deref(), Some("my title"));
    assert_eq!(store.model_at_root(), Some("model-b"));
    assert_eq!(store.get_label(&a0), Some("bookmark"));
    assert_eq!(reopened.compaction.read_files, ["src/important.rs"]);
}

/// The lag can end mid-suffix: the stored tail that the kept suffix begins with is carried, and the
/// kept messages after it are appended.
#[test]
fn a_compaction_whose_kept_suffix_is_partly_on_disk_writes_exactly_the_new_transcript() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.jsonl");
    let mut store = SessionStore::create(path.clone(), SessionMeta::new("/w", "m")).unwrap();
    let mut s = Session::new();
    turn(&mut s, 0);
    s.push(Message::user("u1"));
    store
        .persist(&s.messages, &s.compaction, &s.edited)
        .unwrap();
    s.push(Message::assistant(vec![ContentBlock::text("a1")]));
    turn(&mut s, 2);
    compact(&mut s, 2);
    store
        .persist(&s.messages, &s.compaction, &s.edited)
        .unwrap();
    drop(store);
    let (_store, reopened) = SessionStore::open(path.clone()).unwrap();
    assert_eq!(texts(&reopened.messages)[1..], ["u1", "a1", "u2", "a2"]);
    let raw = fs::read_to_string(&path).unwrap();
    let record = raw
        .lines()
        .find(|l| l.contains(r#""type":"compaction""#))
        .unwrap();
    let record: serde_json::Value = serde_json::from_str(record).unwrap();
    assert_eq!(
        record["folded_ids"].as_array().unwrap().len(),
        2,
        "{record}"
    );
}
