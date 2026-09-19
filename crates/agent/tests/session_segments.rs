//! Segmented session storage: the layout itself, and what a replay makes of a damaged one.
//!
//! Fencing lives in `session_segments_fencing.rs`, sealing in `session_segments_sealing.rs`, and
//! paths/locks/trash in `session_segments_paths.rs`.

// Test target: `.unwrap()`/`panic!` are how a test reports a failed setup.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::fs;
use std::path::{Path, PathBuf};

use agent_core::{ContentBlock, Message};
use beyond_ai_agent::session_store::{Layout, RepoOptions, SessionMeta, SessionRepo, SessionStore};
use serde_json::Value;
use tempfile::TempDir;

/// A segmented repo with no sealing — the layout on its own.
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

fn segments(session_dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = fs::read_dir(session_dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".jsonl"))
        })
        .collect();
    out.sort();
    out
}

fn line_types(path: &Path) -> Vec<String> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            serde_json::from_str::<Value>(l).unwrap()["type"]
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect()
}

/// The first text block of a message — what these tests identify a turn by.
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

#[test]
fn the_single_file_layout_writes_exactly_what_it_always_did() {
    // The default repo is unchanged by everything the segmented layout adds: one `.jsonl` per
    // session, one JSON object per line, no framing of any kind, mode 0600.
    let dir = TempDir::new().unwrap();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo
        .create(SessionMeta::with_id("plain", "/w", "m"))
        .unwrap();
    store
        .append_new(&[Message::user("one"), Message::user("two")])
        .unwrap();
    store.set_title("t").unwrap();

    let path = store.path().to_path_buf();
    assert_eq!(
        path.file_name().unwrap().to_str().unwrap().split_once('_'),
        Some((store.meta().created_at.to_string().as_str(), "plain.jsonl")),
        "the file is still <created_at>_<id>.jsonl"
    );
    assert_eq!(
        line_types(&path),
        vec!["session", "message", "message", "title_change"],
        "no segment header, no base trailer, no reordering"
    );
    let raw = fs::read_to_string(&path).unwrap();
    assert!(
        !raw.contains("\"type\":\"segment\"") && !raw.contains("base_complete"),
        "the single-file layout carries no log framing"
    );
    assert!(
        raw.ends_with('\n'),
        "every entry is a whole newline-terminated line"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
    assert_eq!(
        fs::read_dir(dir.path()).unwrap().count(),
        1,
        "nothing else is created beside it"
    );
}

#[test]
fn create_append_and_reopen_round_trip() {
    let dir = TempDir::new().unwrap();
    let repo = seg_repo(dir.path());
    let mut store = repo.create(SessionMeta::with_id("s1", "/w", "m")).unwrap();
    assert_eq!(store.path(), dir.path().join("s1"));
    store.append_new(&[Message::user("one")]).unwrap();
    drop(store);

    // A fresh owner replays, then appends: its first write rolls a new epoch sealing the old one.
    let (mut store, session) = repo.open_or_create_id("s1", "/w", "m").unwrap();
    assert_eq!(session.messages.len(), 1);
    store
        .append_new(&[Message::user("one"), Message::user("two")])
        .unwrap();
    drop(store);

    assert_eq!(replay(&repo, "s1"), vec!["one", "two"]);
    let segs = segments(&dir.path().join("s1"));
    assert_eq!(segs.len(), 2);
    assert_eq!(
        segs[0].file_name().unwrap().to_str().unwrap(),
        "000001.jsonl"
    );
    assert_eq!(
        segs[1].file_name().unwrap().to_str().unwrap(),
        "000002.jsonl"
    );
    assert_eq!(
        line_types(&segs[0]),
        vec!["segment", "session", "base_complete", "message"],
        "epoch 1 is a complete base; later appends land after its trailer"
    );
    assert_eq!(
        line_types(&segs[1]),
        vec!["segment", "message"],
        "`append_new` is count-keyed, so only the turn that is actually new is written"
    );
}

#[test]
fn a_session_directory_with_no_segments_is_not_a_session() {
    let dir = TempDir::new().unwrap();
    let repo = seg_repo(dir.path());
    fs::create_dir_all(dir.path().join("empty")).unwrap();
    fs::write(dir.path().join("empty/lock"), b"").unwrap();
    assert!(repo.list().unwrap().is_empty());
    // Addressing it creates it for real, rather than resolving to the bare directory.
    let (_s, session) = repo.open_or_create_id("empty", "/w", "m").unwrap();
    assert!(session.messages.is_empty());
    assert!(dir.path().join("empty/000001.jsonl").is_file());
}

#[test]
fn a_torn_final_line_is_dropped_and_everything_before_it_survives() {
    let dir = TempDir::new().unwrap();
    let repo = seg_repo(dir.path());
    let mut store = repo.create(SessionMeta::with_id("s1", "/w", "m")).unwrap();
    store
        .append_new(&[Message::user("one"), Message::user("two")])
        .unwrap();
    drop(store);

    // A crash mid-append: the last line never got its newline.
    let seg = dir.path().join("s1/000001.jsonl");
    let mut raw = fs::read(&seg).unwrap();
    raw.extend_from_slice(br#"{"type":"message","id":"x","timestamp":0,"role":"user","con"#);
    fs::write(&seg, &raw).unwrap();

    assert_eq!(replay(&repo, "s1"), vec!["one", "two"]);
}

#[test]
fn a_segment_with_a_torn_header_counts_as_empty() {
    let dir = TempDir::new().unwrap();
    let repo = seg_repo(dir.path());
    let mut store = repo.create(SessionMeta::with_id("s1", "/w", "m")).unwrap();
    store.append_new(&[Message::user("one")]).unwrap();
    drop(store);
    let (mut store, _) = repo.open_or_create_id("s1", "/w", "m").unwrap();
    store
        .append_new(&[Message::user("one"), Message::user("two")])
        .unwrap();
    drop(store);

    // Epoch 2 died before its own header line was complete.
    let seg2 = dir.path().join("s1/000002.jsonl");
    fs::write(&seg2, br#"{"type":"segm"#).unwrap();

    assert_eq!(
        replay(&repo, "s1"),
        vec!["one"],
        "a headerless segment is sealed at 0, not replayed as content"
    );
    // And the session keeps working: the next owner rolls past it.
    let (mut store, _) = repo.open_or_create_id("s1", "/w", "m").unwrap();
    store
        .append_new(&[Message::user("one"), Message::user("three")])
        .unwrap();
    drop(store);
    assert_eq!(replay(&repo, "s1"), vec!["one", "three"]);
}

#[test]
fn a_crash_midway_through_a_base_replays_the_previous_one() {
    let dir = TempDir::new().unwrap();
    let repo = seg_repo(dir.path());
    let mut store = repo.create(SessionMeta::with_id("s1", "/w", "m")).unwrap();
    store
        .append_new(&[Message::user("one"), Message::user("two")])
        .unwrap();
    drop(store);

    // A base that never got its trailer — a compaction killed halfway through its rewrite. Its
    // content is a *partial* replacement of the whole transcript, so replaying it would silently
    // truncate the session.
    let sealed_at = fs::metadata(dir.path().join("s1/000001.jsonl"))
        .unwrap()
        .len();
    let header =
        format!(r#"{{"type":"segment","epoch":2,"sealed":{{"1":{sealed_at}}},"base":true}}"#);
    fs::write(
        dir.path().join("s1/000002.jsonl"),
        format!(
            "{header}\n{}\n",
            r#"{"type":"session","version":1,"id":"s1","created_at":1,"cwd":"/w","model":"m"}"#
        ),
    )
    .unwrap();

    assert_eq!(
        replay(&repo, "s1"),
        vec!["one", "two"],
        "the incomplete base contributes nothing; the last complete base still does"
    );

    // Recovery: the next owner rolls past the debris and the session carries on.
    let (mut store, _) = repo.open_or_create_id("s1", "/w", "m").unwrap();
    store
        .append_new(&[
            Message::user("one"),
            Message::user("two"),
            Message::user("three"),
        ])
        .unwrap();
    drop(store);
    assert_eq!(replay(&repo, "s1"), vec!["one", "two", "three"]);
}

#[test]
fn a_rewrite_becomes_a_new_base_and_older_segments_go_away_only_behind_the_previous_one() {
    let dir = TempDir::new().unwrap();
    let repo = seg_repo(dir.path());
    let mut store = repo.create(SessionMeta::with_id("s1", "/w", "m")).unwrap();
    store
        .append_new(&[Message::user("one"), Message::user("two")])
        .unwrap();
    // Compaction's shape: replace the active path wholesale.
    store.rewrite(&[Message::user("summary")]).unwrap();
    drop(store);

    assert_eq!(replay(&repo, "s1"), vec!["summary"]);
    let segs = segments(&dir.path().join("s1"));
    assert_eq!(segs.len(), 2, "the new base does not delete the old one");
    assert_eq!(
        line_types(&segs[1]),
        vec!["segment", "session", "message", "base_complete"],
        "a base is header, content, trailer"
    );

    // A second base retires everything behind the base it supersedes — except epoch 1, which is the
    // O_EXCL fence that says this session exists at all.
    let (mut store, _) = repo.open_or_create_id("s1", "/w", "m").unwrap();
    store.rewrite(&[Message::user("shorter")]).unwrap();
    drop(store);
    assert_eq!(replay(&repo, "s1"), vec!["shorter"]);
    assert!(dir.path().join("s1/000001.jsonl").is_file());
}

#[test]
fn the_layout_is_detected_from_the_path_so_a_bare_open_still_works() {
    let dir = TempDir::new().unwrap();
    let repo = seg_repo(dir.path());
    let mut store = repo.create(SessionMeta::with_id("s1", "/w", "m")).unwrap();
    store.append_new(&[Message::user("one")]).unwrap();
    let path = store.path().to_path_buf();
    drop(store);

    // What `main.rs`/export/`fork_from_path` do: hand a path over with no layout in hand.
    let (store, session) = SessionStore::open(path).unwrap();
    assert_eq!(session.messages.len(), 1);
    assert_eq!(store.meta().id, "s1");
}
