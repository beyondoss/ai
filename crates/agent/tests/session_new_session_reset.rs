//! `/new` on a single-file session (`SessionStore::reset_for_new_session`, what `serve`'s
//! `new_session` does when it reuses the file) starts a new conversation: nothing of the old one —
//! not its abandoned branches, not its title or compaction counters — survives on disk.

// Test target: `.unwrap()`/`panic!` are how a test reports a failed setup.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;

use agent_core::{ContentBlock, Message, Session};
use beyond_ai_agent::session_store::{SessionMeta, SessionStore};
use tempfile::TempDir;

/// Message texts are words no id, hash or timestamp can contain: the reset test searches the raw
/// file for them, and a short marker like `a0` turned up inside a random hex id.
fn turn(s: &mut Session, i: usize) {
    s.push(Message::user(format!("old-user-turn-{i}")));
    s.push(Message::assistant(vec![ContentBlock::text(format!(
        "old-assistant-turn-{i}"
    ))]));
}

/// Hunt t2: a branch abandoned before `/new` stayed in the file, in the new session's tree and branch
/// list, and the reset itself was counted as a compaction.
#[test]
fn new_session_keeps_nothing_of_the_old_conversation() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.jsonl");
    let mut store = SessionStore::create(path.clone(), SessionMeta::new("/w", "m")).unwrap();
    let mut s = Session::new();
    turn(&mut s, 0);
    turn(&mut s, 1);
    store.append_new(&s.messages).unwrap();
    // A branch: back to after u0/a0, then somewhere else.
    let a0 = store.active_ids()[1].clone();
    let mut s = Session::new();
    s.messages = Arc::new(store.switch_active(&a0).unwrap());
    s.push(Message::user("secret-branch"));
    store.append_new(&s.messages).unwrap();
    store.set_title("old conversation").unwrap();
    assert_eq!(store.list_branches().len(), 2);

    store.reset_for_new_session().unwrap();
    assert!(store.tree().is_empty(), "the old tree survived the reset");
    assert_eq!(store.meta().compactions, 0);
    assert_eq!(store.meta().dropped_messages, 0);
    assert_eq!(store.meta().title, None);

    let mut fresh = Session::new();
    fresh.push(Message::user("new session first"));
    store.append_new(&fresh.messages).unwrap();
    drop(store);

    let raw = std::fs::read_to_string(&path).unwrap();
    for old in [
        "secret-branch",
        "old-user-turn-1",
        "old-assistant-turn-0",
        "old conversation",
    ] {
        assert!(!raw.contains(old), "{old:?} survived /new on disk: {raw}");
    }
    let (store, reopened) = SessionStore::open(path).unwrap();
    assert_eq!(reopened.messages.len(), 1);
    assert_eq!(store.list_branches().len(), 1);
    assert_eq!(store.tree().len(), 1);
    assert_eq!(store.meta().compactions, 0);
    assert_eq!(reopened.compaction.compactions, 0);
}

/// A narrow rewrite (a fork's) still carries the other branches along — only the full reset drops
/// them.
#[test]
fn a_narrow_rewrite_still_keeps_other_branches() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.jsonl");
    let mut store = SessionStore::create(path.clone(), SessionMeta::new("/w", "m")).unwrap();
    let mut s = Session::new();
    turn(&mut s, 0);
    turn(&mut s, 1);
    store.append_new(&s.messages).unwrap();
    let a0 = store.active_ids()[1].clone();
    let mut s = Session::new();
    s.messages = Arc::new(store.switch_active(&a0).unwrap());
    s.push(Message::user("other-branch"));
    store.append_new(&s.messages).unwrap();

    store.rewrite(&[Message::user("rewritten")]).unwrap();
    drop(store);
    // The branch switched away from (u1/a1) is the one off the active path.
    assert!(
        std::fs::read_to_string(&path)
            .unwrap()
            .contains(r#""text":"old-user-turn-1""#)
    );
}
