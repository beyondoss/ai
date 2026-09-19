//! Addressing a segmented session: where its working memory lives, what deleting it does, and the
//! advisory lock that keeps a live session from being deleted out from under its owner.

// Test target: `.unwrap()`/`panic!` are how a test reports a failed setup.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::fs;
use std::path::Path;

use agent_core::Message;
use beyond_ai_agent::memory;
use beyond_ai_agent::session_store::{
    Layout, RepoOptions, SessionMeta, SessionRepo, acquire_session_lock,
};
use tempfile::TempDir;

fn seg_repo(dir: &Path, prefix: Option<&str>) -> SessionRepo {
    SessionRepo::open_with(
        dir,
        RepoOptions {
            layout: Layout::Segmented { codec: None },
            id_prefix: prefix.map(str::to_string),
        },
    )
    .unwrap()
}

#[test]
fn each_session_gets_its_own_memory_directory_even_with_dotted_ids() {
    let dir = TempDir::new().unwrap();
    let repo = seg_repo(dir.path(), Some("a07"));
    let one = repo
        .create(SessionMeta::with_id("a07.one", "/w", "m"))
        .unwrap();
    let two = repo
        .create(SessionMeta::with_id("a07.two", "/w", "m"))
        .unwrap();

    assert_eq!(one.memory_dir(), dir.path().join("a07.one/memory"));
    assert_eq!(two.memory_dir(), dir.path().join("a07.two/memory"));
    assert_ne!(one.memory_dir(), two.memory_dir());

    // The bug this replaces: `with_extension` treats everything after the last `.` as the extension,
    // so two sessions sharing a shard prefix would share one working-memory directory.
    assert_eq!(
        memory::session_dir(Some(one.path()), "a07.one"),
        memory::session_dir(Some(two.path()), "a07.two"),
        "the path-derived helper really does collide for dotted ids"
    );

    // The single-file layout keeps its historical sibling naming.
    let plain_dir = TempDir::new().unwrap();
    let plain = SessionRepo::open(plain_dir.path()).unwrap();
    let store = plain.create(SessionMeta::with_id("s1", "/w", "m")).unwrap();
    assert_eq!(
        store.memory_dir(),
        store.path().with_extension("memory"),
        "unchanged for the single-file layout"
    );
}

#[test]
fn deleting_trashes_the_whole_session_directory_and_survives_repeats() {
    let dir = TempDir::new().unwrap();
    let repo = seg_repo(dir.path(), None);
    for round in 0..2 {
        let mut store = repo.create(SessionMeta::with_id("s1", "/w", "m")).unwrap();
        store
            .append_new(&[Message::user(format!("round {round}"))])
            .unwrap();
        // Working memory lives inside the session directory, so it trashes with it.
        fs::create_dir_all(store.memory_dir()).unwrap();
        fs::write(store.memory_dir().join("notes.md"), b"x").unwrap();
        drop(store);
        repo.delete("s1").unwrap();
        assert!(!dir.path().join("s1").exists());
    }

    let trash = repo.list_trash().unwrap();
    assert_eq!(trash.len(), 2, "each delete lands on its own name");
    assert!(trash.iter().all(|e| e.id == "s1"));
    assert!(repo.list().unwrap().is_empty());
    for entry in fs::read_dir(dir.path().join(".trash")).unwrap().flatten() {
        assert!(entry.path().join("memory/notes.md").is_file());
    }

    // Deleting something that isn't there is a successful no-op.
    repo.delete("s1").unwrap();
    repo.delete("never-existed").unwrap();

    // Restoring brings back the most recent copy.
    assert!(repo.restore_session("s1").unwrap());
    let (_store, session) = repo.open_or_create_id("s1", "/w", "m").unwrap();
    assert_eq!(session.messages.len(), 1);
    assert!(dir.path().join("s1/memory/notes.md").is_file());
    assert_eq!(repo.list_trash().unwrap().len(), 1);
    assert!(!repo.restore_session("absent").unwrap());
}

#[test]
fn deleting_a_session_someone_is_holding_is_refused() {
    let dir = TempDir::new().unwrap();
    let repo = seg_repo(dir.path(), None);
    let store = repo.create(SessionMeta::with_id("s1", "/w", "m")).unwrap();
    let path = store.path().to_path_buf();

    let guard = acquire_session_lock(&path).unwrap();
    assert!(guard.is_some(), "the session was free");
    // A second acquire — in this process or any other — reports it held rather than taking it.
    assert!(acquire_session_lock(&path).unwrap().is_none());

    let err = repo.delete("s1").unwrap_err();
    assert!(
        err.to_string().contains("in use"),
        "expected a refusal, got {err}"
    );
    assert!(dir.path().join("s1/000001.jsonl").is_file());

    drop(guard);
    repo.delete("s1").unwrap();
    assert!(!dir.path().join("s1").exists());
}

#[test]
fn the_lock_follows_the_inode_not_the_path() {
    let dir = TempDir::new().unwrap();
    let repo = seg_repo(dir.path(), None);
    let store = repo.create(SessionMeta::with_id("s1", "/w", "m")).unwrap();
    let path = store.path().to_path_buf();

    let guard = acquire_session_lock(&path).unwrap().unwrap();
    // The session directory is renamed away underneath the holder — exactly what a delete into
    // `.trash/` does — and a brand-new directory takes its place at the same path.
    let trashed = dir.path().join("moved-aside");
    fs::rename(&path, &trashed).unwrap();
    fs::create_dir_all(&path).unwrap();
    drop(guard);

    let fresh = acquire_session_lock(&path).unwrap();
    assert!(
        fresh.is_some(),
        "the new directory's lock is a different inode and is free"
    );
    assert!(trashed.join("lock").is_file(), "the old lock moved with it");
}

#[test]
fn locking_works_for_the_single_file_layout_too() {
    let dir = TempDir::new().unwrap();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let store = repo.create(SessionMeta::with_id("s1", "/w", "m")).unwrap();
    let guard = acquire_session_lock(store.path()).unwrap();
    assert!(guard.is_some());
    assert!(acquire_session_lock(store.path()).unwrap().is_none());
    drop(guard);
    assert!(acquire_session_lock(store.path()).unwrap().is_some());
}

#[test]
fn derived_ids_carry_the_repos_prefix() {
    let dir = TempDir::new().unwrap();
    let repo = seg_repo(dir.path(), Some("a07-s1"));
    assert!(repo.mint_id().starts_with("a07-s1."));

    let mut store = repo
        .create(SessionMeta::with_id("a07-s1.parent", "/w", "m"))
        .unwrap();
    store
        .append_new(&[Message::user("one"), Message::user("two")])
        .unwrap();
    drop(store);

    let (forked, session) = repo.fork("a07-s1.parent", 1).unwrap();
    assert!(
        forked.meta().id.starts_with("a07-s1."),
        "a fork stays on its parent's shard: {}",
        forked.meta().id
    );
    assert_eq!(forked.meta().parent.as_deref(), Some("a07-s1.parent"));
    assert_eq!(session.messages.len(), 1);
    assert!(dir.path().join(&forked.meta().id).is_dir());

    // A caller-supplied id is never rewritten.
    let explicit = repo
        .create(SessionMeta::with_id("other.explicit", "/w", "m"))
        .unwrap();
    assert_eq!(explicit.meta().id, "other.explicit");

    // Without a prefix, ids are exactly what they always were.
    let plain_dir = TempDir::new().unwrap();
    let plain = seg_repo(plain_dir.path(), None);
    assert!(!plain.mint_id().contains('.'));
}

#[test]
fn forking_leaves_the_source_untouched() {
    let dir = TempDir::new().unwrap();
    let repo = seg_repo(dir.path(), Some("a07"));
    let mut store = repo
        .create(SessionMeta::with_id("a07.parent", "/w", "m"))
        .unwrap();
    store
        .append_new(&[Message::user("one"), Message::user("two")])
        .unwrap();
    drop(store);
    let before: Vec<_> = fs::read_dir(dir.path().join("a07.parent"))
        .unwrap()
        .flatten()
        .map(|e| e.file_name())
        .collect();

    let (_forked, _) = repo.fork("a07.parent", 1).unwrap();

    let after: Vec<_> = fs::read_dir(dir.path().join("a07.parent"))
        .unwrap()
        .flatten()
        .map(|e| e.file_name())
        .collect();
    assert_eq!(
        before.len(),
        after.len(),
        "a fork source is opened read-only and rolls no epoch"
    );

    // The source is still writable by its real owner afterwards.
    let (mut store, session) = repo.open_or_create_id("a07.parent", "/w", "m").unwrap();
    assert_eq!(session.messages.len(), 2);
    store
        .append_new(&[
            Message::user("one"),
            Message::user("two"),
            Message::user("three"),
        ])
        .unwrap();
}

#[test]
fn an_exact_id_resolves_without_a_directory_scan_and_a_prefix_still_works() {
    let dir = TempDir::new().unwrap();
    let repo = seg_repo(dir.path(), None);
    repo.create(SessionMeta::with_id("alpha", "/w", "m"))
        .unwrap();
    repo.create(SessionMeta::with_id("alphabet", "/w", "m"))
        .unwrap();

    // Exact wins over any prefix neighbour — a session id is an address.
    let (store, _) = repo.open_or_create_id("alpha", "/w", "m").unwrap();
    assert_eq!(store.meta().id, "alpha");

    // An unambiguous prefix still resolves for the human-convenience lookup.
    let (store, _) = repo.open_id("alphab").unwrap();
    assert_eq!(store.meta().id, "alphabet");

    // An ambiguous one is an error naming the candidates, not a guess.
    let Err(err) = repo.open_id("alph") else {
        panic!("an ambiguous prefix must not resolve");
    };
    assert!(err.to_string().contains("matches more than one session"));
}
