//! The epoch fence: who owns a segmented session, what happens to whoever doesn't, and how the log
//! survives a write that fails halfway.

// Test target: `.unwrap()`/`panic!` are how a test reports a failed setup.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use agent_core::{ContentBlock, Message};
use beyond_ai_agent::session_store::{
    Layout, RepoOptions, SessionMeta, SessionRepo, SessionStore, is_superseded,
};
use tempfile::TempDir;

fn seg_repo(dir: &Path) -> SessionRepo {
    SessionRepo::open_with(
        dir,
        RepoOptions {
            layout: Layout::Segmented { codec: None },
            id_prefix: None,
        },
    )
    .unwrap()
}

fn text_of(m: &Message) -> String {
    m.content
        .iter()
        .find_map(|b| match b {
            ContentBlock::Text { text, .. } => Some(text.to_string()),
            _ => None,
        })
        .unwrap_or_default()
}

fn replay(repo: &SessionRepo, id: &str) -> Vec<String> {
    let (_store, session) = repo.open_or_create_id(id, "/w", "m").unwrap();
    session.messages.iter().map(text_of).collect()
}

fn segment_count(session_dir: &Path) -> usize {
    fs::read_dir(session_dir)
        .unwrap()
        .flatten()
        .filter(|e| {
            e.file_name()
                .to_str()
                .is_some_and(|n| n.ends_with(".jsonl"))
        })
        .count()
}

/// Every segment header in the session, oldest first.
fn headers(session_dir: &Path) -> Vec<serde_json::Value> {
    let mut paths: Vec<_> = fs::read_dir(session_dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".jsonl"))
        })
        .collect();
    paths.sort();
    paths
        .iter()
        .filter_map(|p| {
            let raw = fs::read_to_string(p).ok()?;
            serde_json::from_str(raw.lines().next()?).ok()
        })
        .collect()
}

#[test]
fn a_takeover_seals_the_old_epoch_and_a_stale_writers_later_bytes_never_replay() {
    let dir = TempDir::new().unwrap();
    let repo = seg_repo(dir.path());
    let mut old = repo.create(SessionMeta::with_id("s1", "/w", "m")).unwrap();
    old.append_new(&[Message::user("one")]).unwrap();

    // A second process replays what is on disk and takes over by creating the next epoch.
    let (mut new_owner, session) = repo.open_or_create_id("s1", "/w", "m").unwrap();
    assert_eq!(session.messages.len(), 1);
    new_owner
        .append_new(&[Message::user("one"), Message::user("two")])
        .unwrap();

    // The old owner is still alive and still holds an open append target. Its writes land on disk —
    // nothing can stop them — but they are past the seal the new epoch's header recorded, so no
    // reader ever sees them again.
    old.append_new(&[Message::user("one"), Message::user("stale")])
        .unwrap();

    let sealed_at = headers(&dir.path().join("s1"))[1]["sealed"]["1"]
        .as_u64()
        .unwrap();
    let seg1_len = fs::metadata(dir.path().join("s1/000001.jsonl"))
        .unwrap()
        .len();
    assert!(
        sealed_at < seg1_len,
        "the stale writer appended past the seal ({sealed_at} of {seg1_len} bytes)"
    );
    assert_eq!(
        replay(&repo, "s1"),
        vec!["one", "two"],
        "the replay stops at the seal"
    );
}

#[test]
fn a_predecessor_that_grew_since_the_replay_is_a_takeover() {
    let dir = TempDir::new().unwrap();
    let repo = seg_repo(dir.path());
    let mut owner = repo.create(SessionMeta::with_id("s1", "/w", "m")).unwrap();
    owner.append_new(&[Message::user("one")]).unwrap();

    // This store replayed the session and is about to write — but the incumbent appends first, so
    // the segment it measured is no longer the length it read.
    let (mut latecomer, _) = repo.open_or_create_id("s1", "/w", "m").unwrap();
    owner
        .append_new(&[Message::user("one"), Message::user("two")])
        .unwrap();

    let err = latecomer
        .append_new(&[Message::user("one"), Message::user("mine")])
        .unwrap_err();
    assert!(is_superseded(&err), "expected a takeover, got {err}");
    assert!(latecomer.superseded());
    assert_eq!(replay(&repo, "s1"), vec!["one", "two"]);
}

#[test]
fn losing_the_race_to_create_the_next_epoch_is_a_takeover() {
    let dir = TempDir::new().unwrap();
    let repo = seg_repo(dir.path());
    let mut first = repo.create(SessionMeta::with_id("s1", "/w", "m")).unwrap();
    first.append_new(&[Message::user("one")]).unwrap();
    drop(first);

    // Two stores replay the same state, so both intend to create the same next epoch.
    let (mut a, _) = repo.open_or_create_id("s1", "/w", "m").unwrap();
    let (mut b, _) = repo.open_or_create_id("s1", "/w", "m").unwrap();
    a.append_new(&[Message::user("one"), Message::user("a")])
        .unwrap();
    let err = b
        .append_new(&[Message::user("one"), Message::user("b")])
        .unwrap_err();
    assert!(is_superseded(&err), "expected a takeover, got {err}");
    assert_eq!(replay(&repo, "s1"), vec!["one", "a"]);
}

#[test]
fn a_superseded_store_never_writes_again() {
    let dir = TempDir::new().unwrap();
    let repo = seg_repo(dir.path());
    let mut owner = repo.create(SessionMeta::with_id("s1", "/w", "m")).unwrap();
    owner.append_new(&[Message::user("one")]).unwrap();
    let (mut loser, _) = repo.open_or_create_id("s1", "/w", "m").unwrap();
    owner
        .append_new(&[Message::user("one"), Message::user("two")])
        .unwrap();
    assert!(
        loser
            .append_new(&[Message::user("one"), Message::user("x")])
            .is_err()
    );
    assert!(loser.superseded());

    // Every other kind of write is poisoned too — a title, a label, a model change, a rewrite.
    for err in [
        loser.set_title("t").unwrap_err(),
        loser.record_model_change("other").unwrap_err(),
        loser.append_custom("k", serde_json::json!({})).unwrap_err(),
        loser.rewrite(&[Message::user("z")]).unwrap_err(),
    ] {
        assert!(is_superseded(&err), "expected a takeover, got {err}");
    }
    assert_eq!(replay(&repo, "s1"), vec!["one", "two"]);
}

#[test]
fn racing_creates_leave_exactly_one_owner() {
    let dir = TempDir::new().unwrap();
    let winners = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(|| {
                let repo = seg_repo(dir.path());
                if repo
                    .create(SessionMeta::with_id("contested", "/w", "m"))
                    .is_ok()
                {
                    winners.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
    });
    assert_eq!(
        winners.load(Ordering::Relaxed),
        1,
        "the O_EXCL create of epoch 1 is the fence"
    );
    assert_eq!(segment_count(&dir.path().join("contested")), 1);
}

#[test]
fn a_read_only_open_creates_nothing_and_refuses_every_write() {
    let dir = TempDir::new().unwrap();
    let repo = seg_repo(dir.path());
    let mut store = repo.create(SessionMeta::with_id("s1", "/w", "m")).unwrap();
    store.append_new(&[Message::user("one")]).unwrap();
    let path = store.path().to_path_buf();
    drop(store);

    let before = fs::read_dir(&path).unwrap().flatten().count();
    let (mut ro, session) = SessionStore::open_read_only(path.clone()).unwrap();
    assert_eq!(session.messages.len(), 1);
    assert!(
        ro.append_new(&[Message::user("one"), Message::user("x")])
            .is_err()
    );
    assert!(ro.set_title("t").is_err());
    assert!(ro.rewrite(&[]).is_err());
    assert_eq!(
        fs::read_dir(&path).unwrap().flatten().count(),
        before,
        "a read-only open rolls no epoch and writes no file"
    );
    assert!(ro.read_only());
    assert!(!ro.superseded(), "refusing a write is not a takeover");
    assert_eq!(replay(&repo, "s1"), vec!["one"]);

    // The guard is the store's, not the layout's: a single-file session refuses the same writes.
    let plain_dir = TempDir::new().unwrap();
    let plain = SessionRepo::open(plain_dir.path()).unwrap();
    let mut store = plain.create(SessionMeta::with_id("f1", "/w", "m")).unwrap();
    store.append_new(&[Message::user("one")]).unwrap();
    let file = store.path().to_path_buf();
    let bytes = fs::metadata(&file).unwrap().len();
    drop(store);

    let (mut ro, _) = SessionStore::open_read_only(file.clone()).unwrap();
    assert!(
        ro.append_new(&[Message::user("one"), Message::user("x")])
            .is_err()
    );
    assert!(ro.set_title("t").is_err());
    assert_eq!(fs::metadata(&file).unwrap().len(), bytes);
}

#[test]
fn a_reattach_streak_consolidates_into_a_base() {
    let dir = TempDir::new().unwrap();
    let repo = seg_repo(dir.path());
    repo.create(SessionMeta::with_id("s1", "/w", "m")).unwrap();

    // Each attach-and-write cycle rolls one epoch — the reap-and-reattach pattern this bounds.
    let mut history: Vec<Message> = Vec::new();
    for i in 0..24 {
        history.push(Message::user(format!("turn {i}")));
        let (mut store, _) = repo.open_or_create_id("s1", "/w", "m").unwrap();
        store.append_new(&history).unwrap();
    }

    let session_dir = dir.path().join("s1");
    let hs = headers(&session_dir);
    assert!(
        hs.iter()
            .any(|h| h["base"] == true && h["epoch"].as_u64().unwrap_or(0) > 1),
        "a base was written once the segment count passed the budget: {hs:?}"
    );
    assert!(
        segment_count(&session_dir) < 24,
        "consolidation bounds growth, got {} segments after 24 attaches",
        segment_count(&session_dir)
    );
    assert!(
        !session_dir.join("000002.jsonl").exists(),
        "a second base retires the segments behind the base it superseded"
    );
    assert!(
        session_dir.join("000001.jsonl").exists(),
        "epoch 1 is never pruned — it is the fence that says this session exists"
    );
    assert_eq!(
        replay(&repo, "s1"),
        (0..24).map(|i| format!("turn {i}")).collect::<Vec<_>>(),
        "consolidating loses nothing"
    );
}

#[cfg(unix)]
#[test]
fn an_append_that_fails_seals_the_segment_and_rolls_to_a_new_one() {
    use std::os::unix::fs::PermissionsExt;

    let dir = TempDir::new().unwrap();
    let repo = seg_repo(dir.path());
    let mut store = repo.create(SessionMeta::with_id("s1", "/w", "m")).unwrap();
    store.append_new(&[Message::user("one")]).unwrap();

    // The append target becomes unwritable underneath the owner — one ENOSPC-shaped failure.
    let seg1 = dir.path().join("s1/000001.jsonl");
    let good_len = fs::metadata(&seg1).unwrap().len();
    fs::set_permissions(&seg1, fs::Permissions::from_mode(0o400)).unwrap();
    assert!(
        store
            .append_new(&[Message::user("one"), Message::user("lost")])
            .is_err()
    );
    assert!(!store.superseded(), "an I/O error is not a takeover");

    // The next write rolls instead of retrying into a segment nobody can bound.
    fs::set_permissions(&seg1, fs::Permissions::from_mode(0o600)).unwrap();
    store
        .append_new(&[Message::user("one"), Message::user("after")])
        .unwrap();
    assert_eq!(segment_count(&dir.path().join("s1")), 2);
    assert_eq!(
        headers(&dir.path().join("s1"))[1]["sealed"]["1"]
            .as_u64()
            .unwrap(),
        good_len,
        "sealed at the last offset known good"
    );
    drop(store);
    assert_eq!(replay(&repo, "s1"), vec!["one", "after"]);
}

#[test]
fn the_listing_reflects_a_roll_a_base_and_a_prune() {
    let dir = TempDir::new().unwrap();
    let repo = seg_repo(dir.path());
    let mut store = repo.create(SessionMeta::with_id("s1", "/w", "m")).unwrap();
    store.append_new(&[Message::user("hello")]).unwrap();
    drop(store);
    let first = repo.list().unwrap();
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].message_count, 1);
    assert_eq!(first[0].preview.as_deref(), Some("hello"));

    // A warm listing must not serve the pre-roll answer.
    let (mut store, _) = repo.open_or_create_id("s1", "/w", "m").unwrap();
    store
        .append_new(&[Message::user("hello"), Message::user("again")])
        .unwrap();
    drop(store);
    let second = repo.list().unwrap();
    assert_eq!(second[0].message_count, 2);

    let (mut store, _) = repo.open_or_create_id("s1", "/w", "m").unwrap();
    store.rewrite(&[Message::user("compacted")]).unwrap();
    drop(store);
    let third = repo.list().unwrap();
    assert_eq!(third[0].message_count, 1);
    assert_eq!(third[0].preview.as_deref(), Some("compacted"));
}

#[test]
fn a_fresh_store_appends_into_the_segment_it_just_created() {
    let dir = TempDir::new().unwrap();
    let repo = seg_repo(dir.path());
    let mut store = repo.create(SessionMeta::with_id("s1", "/w", "m")).unwrap();
    for i in 0..5 {
        let history: Vec<Message> = (0..=i).map(|j| Message::user(format!("t{j}"))).collect();
        store.append_new(&history).unwrap();
    }
    assert_eq!(
        segment_count(&dir.path().join("s1")),
        1,
        "an owner that already holds a segment never rolls another"
    );
}
