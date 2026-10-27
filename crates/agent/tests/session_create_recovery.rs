//! A session's first file is created whole, and one a crash left headerless no longer bricks its id.
//!
//! A create that crashed between making the file and writing its header used to leave an empty (or
//! torn) first file: the id was then neither openable ("no header") nor creatable (it exists), in
//! both layouts, forever — a pinned daemon session id lost for good.

// Test target: `.unwrap()`/`panic!` are how a test reports a failed setup.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::fs;
use std::path::Path;

use agent_core::Message;
use beyond_agent::session_store::{Layout, RepoOptions, SessionRepo, SessionStore};
use tempfile::TempDir;

fn segmented(dir: &Path) -> SessionRepo {
    SessionRepo::open_with(
        dir,
        RepoOptions {
            layout: Layout::Segmented { codec: None },
            id_prefix: None,
        },
    )
    .unwrap()
}

/// Open-or-create `id`, append one message, and reopen it: the id works end to end.
fn usable(repo: &SessionRepo, id: &str) {
    let (mut store, mut s) = repo.open_or_create_id(id, "/w", "m").unwrap();
    assert_eq!(store.meta().id, id);
    s.push(Message::user("hello"));
    store.append_new(&s.messages).unwrap();
    drop(store);
    let (_store, reopened) = repo.open_or_create_id(id, "/w", "m").unwrap();
    assert_eq!(reopened.messages.len(), 1);
}

/// Hunt t7, segmented: an empty `000001.jsonl` (the crash between the exclusive create and the
/// header write).
#[test]
fn an_empty_first_segment_no_longer_bricks_the_id() {
    let dir = TempDir::new().unwrap();
    let repo = segmented(dir.path());
    fs::create_dir_all(dir.path().join("pinned1")).unwrap();
    fs::write(dir.path().join("pinned1/000001.jsonl"), b"").unwrap();
    usable(&repo, "pinned1");
}

/// Hunt t7, single file: an empty `<created>_<id>.jsonl`.
#[test]
fn an_empty_session_file_no_longer_bricks_the_id() {
    let dir = TempDir::new().unwrap();
    let repo = SessionRepo::open(dir.path()).unwrap();
    fs::write(dir.path().join("1700000000_pinned2.jsonl"), b"").unwrap();
    usable(&repo, "pinned2");
}

/// The same for a header the crash cut part-way: no newline, and only the start of a header line.
#[test]
fn a_torn_header_no_longer_bricks_the_id_in_either_layout() {
    let dir = TempDir::new().unwrap();
    let repo = SessionRepo::open(dir.path()).unwrap();
    fs::write(
        dir.path().join("1700000000_torn.jsonl"),
        br#"{"type":"session","ver"#,
    )
    .unwrap();
    usable(&repo, "torn");

    let dir = TempDir::new().unwrap();
    let repo = segmented(dir.path());
    fs::create_dir_all(dir.path().join("torn")).unwrap();
    fs::write(dir.path().join("torn/000001.jsonl"), br#"{"type":"segm"#).unwrap();
    usable(&repo, "torn");
}

/// Only a headerless file is taken for one: a file holding anything else is never cleared.
#[test]
fn a_file_with_real_content_but_no_header_is_left_alone() {
    let dir = TempDir::new().unwrap();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let path = dir.path().join("1700000000_mine.jsonl");
    fs::write(&path, b"not a session file, but definitely somebody's").unwrap();
    assert!(repo.open_or_create_id("mine", "/w", "m").is_err());
    assert_eq!(
        fs::read(&path).unwrap(),
        b"not a session file, but definitely somebody's"
    );

    let line = format!("{}\n", r#"{"type":"message","role":"user","content":[]}"#);
    fs::write(&path, &line).unwrap();
    assert!(repo.open_or_create_id("mine", "/w", "m").is_err());
    assert_eq!(fs::read_to_string(&path).unwrap(), line);
}

/// A create leaves nothing behind but the session itself, and still refuses an id that exists.
#[test]
fn a_create_is_whole_and_still_exclusive() {
    let dir = TempDir::new().unwrap();
    let repo = segmented(dir.path());
    let (store, _) = repo.open_or_create_id("one", "/w", "m").unwrap();
    let session_dir = store.path().to_path_buf();
    drop(store);
    let names: Vec<String> = fs::read_dir(&session_dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| !n.contains("lock"))
        .collect();
    assert_eq!(names, ["000001.jsonl"], "temp files left behind");
    assert!(
        repo.create(beyond_agent::session_store::SessionMeta::with_id(
            "one", "/w", "m"
        ))
        .is_err()
    );

    // Single file: a zero-byte file `touch`ed ahead of time is still initialized in place.
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.jsonl");
    fs::write(&path, b"").unwrap();
    let store = SessionStore::create(
        path.clone(),
        beyond_agent::session_store::SessionMeta::new("/w", "m"),
    )
    .unwrap();
    drop(store);
    assert!(fs::read_to_string(&path).unwrap().ends_with('\n'));
    let leftovers = fs::read_dir(dir.path()).unwrap().count();
    assert_eq!(leftovers, 1, "temp files left behind");
}
