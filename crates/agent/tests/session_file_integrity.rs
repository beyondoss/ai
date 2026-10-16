//! A single-file session survives damage to its own lines: a torn tail, an unreadable line mid-chain,
//! content written by a newer build, an append that fails half-way. The bar is always the same — the
//! damage costs at most the damaged entries, never the history before them.

// Test target: `.unwrap()`/`panic!` are how a test reports a failed setup.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use agent_core::{ContentBlock, Message, Session, StopReason};
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

/// A session of `turns` user/assistant pairs, on disk at the returned path.
fn session_with(dir: &TempDir, turns: usize) -> (PathBuf, Session) {
    let path = dir.path().join("s.jsonl");
    let mut store = SessionStore::create(path.clone(), SessionMeta::new("/w", "m")).unwrap();
    let mut s = Session::new();
    for i in 0..turns {
        turn(&mut s, i);
    }
    store.append_new(&s.messages).unwrap();
    (path, s)
}

fn append_raw(path: &Path, bytes: &str) {
    let mut f = OpenOptions::new().append(true).open(path).unwrap();
    f.write_all(bytes.as_bytes()).unwrap();
}

/// Replace the one line containing `needle` with `with`.
fn replace_line(path: &Path, needle: &str, with: &str) {
    let raw = fs::read_to_string(path).unwrap();
    let mut hit = false;
    let out: Vec<&str> = raw
        .lines()
        .map(|l| {
            if !hit && l.contains(needle) {
                hit = true;
                with
            } else {
                l
            }
        })
        .collect();
    assert!(hit, "no line contains {needle}");
    fs::write(path, out.join("\n") + "\n").unwrap();
}

/// A crash mid-append leaves half a line. The next turn must not be glued onto it: before the fix it
/// was, the glued line failed to parse, and every message chained after it lost its parent — 8
/// messages reopened as 1.
#[test]
fn a_turn_after_a_torn_tail_reopens_with_the_whole_history() {
    let dir = TempDir::new().unwrap();
    let (path, _) = session_with(&dir, 3);
    append_raw(&path, r#"{"type":"message","id":"x","role":"user","cont"#);

    let (mut store, mut s) = SessionStore::open(path.clone()).unwrap();
    assert_eq!(s.messages.len(), 6);
    turn(&mut s, 3);
    store.append_new(&s.messages).unwrap();
    drop(store);

    let (_store, reopened) = SessionStore::open(path.clone()).unwrap();
    assert_eq!(texts(&reopened.messages), texts(&s.messages));
    // And the torn bytes are gone from disk, not just skipped.
    assert!(!fs::read_to_string(&path).unwrap().contains(r#""id":"x""#));
}

/// The cut belongs to a writer: a read-only open (a preview, an export, a fork source) leaves the file
/// exactly as it found it.
#[test]
fn only_a_writable_open_cuts_a_torn_tail() {
    let dir = TempDir::new().unwrap();
    let (path, _) = session_with(&dir, 1);
    append_raw(&path, r#"{"type":"mess"#);
    let before = fs::read(&path).unwrap();

    let (_ro, s) = SessionStore::open_read_only(path.clone()).unwrap();
    assert_eq!(s.messages.len(), 2);
    assert_eq!(fs::read(&path).unwrap(), before);

    let (_rw, s) = SessionStore::open(path.clone()).unwrap();
    assert_eq!(s.messages.len(), 2);
    let after = fs::read(&path).unwrap();
    assert!(after.ends_with(b"\n"));
    assert_eq!(after.len(), before.len() - r#"{"type":"mess"#.len());
}

/// A complete final entry that only lost its newline is a readable entry, not a torn one: it stays,
/// and the next append goes on a line of its own.
#[test]
fn a_readable_final_entry_missing_its_newline_is_kept_and_the_next_append_starts_a_new_line() {
    let dir = TempDir::new().unwrap();
    let (path, _) = session_with(&dir, 1);
    let raw = fs::read_to_string(&path).unwrap();
    fs::write(&path, raw.trim_end_matches('\n')).unwrap();

    let (mut store, mut s) = SessionStore::open(path.clone()).unwrap();
    assert_eq!(texts(&s.messages), ["u0", "a0"]);
    turn(&mut s, 1);
    store.append_new(&s.messages).unwrap();
    drop(store);

    let (_store, reopened) = SessionStore::open(path).unwrap();
    assert_eq!(texts(&reopened.messages), ["u0", "a0", "u1", "a1"]);
}

/// One unreadable line mid-chain (bit rot, a bad hand edit) costs that one message. Its children
/// re-chain to the entry before it rather than taking everything before the gap off the active path.
#[test]
fn an_unreadable_line_mid_chain_costs_only_that_message() {
    let dir = TempDir::new().unwrap();
    let (path, _) = session_with(&dir, 3);
    replace_line(&path, r#""text":"a1""#, "{this is not json");

    let (_store, s) = SessionStore::open(path).unwrap();
    assert_eq!(texts(&s.messages), ["u0", "a0", "u1", "u2", "a2"]);
}

/// A message a newer build wrote with a content block this build has no variant for is read as a
/// degraded copy — the block named, in place — rather than dropped with everything before it.
#[test]
fn a_content_block_from_a_newer_build_degrades_instead_of_dropping_the_message() {
    let dir = TempDir::new().unwrap();
    let (path, _) = session_with(&dir, 3);
    let raw = fs::read_to_string(&path).unwrap();
    let patched = raw.replacen(
        r#"{"type":"text","text":"a1"}"#,
        r#"{"type":"text","text":"a1"},{"type":"future_block","payload":"x"}"#,
        1,
    );
    assert_ne!(raw, patched);
    fs::write(&path, &patched).unwrap();

    let (_store, s) = SessionStore::open(path.clone()).unwrap();
    assert_eq!(texts(&s.messages), ["u0", "a0", "u1", "a1", "u2", "a2"]);
    let degraded = &s.messages[3].content;
    assert_eq!(degraded.len(), 2);
    assert!(
        matches!(&degraded[1], ContentBlock::Text { text, .. } if text.contains("future_block")),
        "{degraded:?}"
    );
    // The original line is untouched for the build that can read it.
    assert_eq!(fs::read_to_string(&path).unwrap(), patched);
}

/// Same for a stop reason this build does not know: the message keeps its place, read as `Other`.
#[test]
fn a_stop_reason_from_a_newer_build_reads_as_other() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.jsonl");
    let mut store = SessionStore::create(path.clone(), SessionMeta::new("/w", "m")).unwrap();
    let mut s = Session::new();
    s.push(Message::user("u0"));
    s.push(
        Message::assistant(vec![ContentBlock::text("a0")]).with_stop_reason(StopReason::EndTurn),
    );
    s.push(Message::user("u1"));
    store.append_new(&s.messages).unwrap();
    drop(store);
    let raw = fs::read_to_string(&path).unwrap();
    fs::write(
        &path,
        raw.replacen(
            r#""stop_reason":"end_turn""#,
            r#""stop_reason":"paused""#,
            1,
        ),
    )
    .unwrap();

    let (_store, s) = SessionStore::open(path).unwrap();
    assert_eq!(texts(&s.messages), ["u0", "a0", "u1"]);
    assert_eq!(s.messages[1].stop_reason, Some(StopReason::Other));
}

/// A branch switch whose target line is unreadable is ignored, not followed into an empty session.
#[test]
fn a_leaf_naming_an_unreadable_entry_does_not_empty_the_session() {
    let dir = TempDir::new().unwrap();
    let (path, _) = session_with(&dir, 2);
    let (mut store, _) = SessionStore::open(path.clone()).unwrap();
    let a0 = store.active_ids()[1].clone();
    store.switch_active(&a0).unwrap();
    drop(store);
    replace_line(&path, r#""text":"a0""#, "{garbage");

    let (_store, s) = SessionStore::open(path).unwrap();
    assert!(!s.messages.is_empty());
    assert_eq!(texts(&s.messages)[0], "u0");
}

const ROLLBACK_CHILD: &str = "SESSION_INTEGRITY_ROLLBACK_CHILD";

/// [`a_failed_append_leaves_the_file_as_it_was`]'s child, run under a file-size limit just past the
/// session's current length: an append writes part of its bytes and then fails (`EFBIG`). Reports the
/// file length before and after through stdout.
#[test]
#[ignore = "a child process of a_failed_append_leaves_the_file_as_it_was"]
fn failed_append_child() {
    let Ok(path) = std::env::var(ROLLBACK_CHILD) else {
        return;
    };
    let path = PathBuf::from(path);
    let (mut store, mut s) = SessionStore::open(path.clone()).unwrap();
    let before = fs::metadata(&path).unwrap().len();
    s.push(Message::user("x".repeat(10_000)));
    let err = store.append_new(&s.messages).unwrap_err();
    let after = fs::metadata(&path).unwrap().len();
    println!("ROLLBACK before={before} after={after} err={err}");
}

/// An append that fails half-way (a full disk, a size limit) puts the file back exactly as long as it
/// was, so the retry is not written after half a line.
#[test]
fn a_failed_append_leaves_the_file_as_it_was() {
    let dir = TempDir::new().unwrap();
    let (path, _) = session_with(&dir, 1);
    // The file-size limit (in 512-byte blocks) sits just past the session's length, and SIGXFSZ is
    // ignored, so the over-limit write fails with EFBIG after writing what fits instead of killing
    // the child. Both survive the exec.
    let blocks = fs::metadata(&path).unwrap().len() / 512 + 1;
    let out = std::process::Command::new("sh")
        .arg("-c")
        .arg(format!(
            "ulimit -f {blocks} && trap '' XFSZ && exec \"$0\" --exact failed_append_child --ignored --nocapture"
        ))
        .arg(std::env::current_exe().unwrap())
        .env(ROLLBACK_CHILD, &path)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let line = stdout
        .lines()
        .find(|l| l.starts_with("ROLLBACK "))
        .unwrap_or_else(|| panic!("no report: {stdout}"));
    let field = |k: &str| -> u64 {
        line.split_whitespace()
            .find_map(|w| w.strip_prefix(k))
            .unwrap()
            .parse()
            .unwrap()
    };
    assert_eq!(field("before="), field("after="), "{line}");

    // And the session keeps working: a retry lands whole.
    let (mut store, mut s) = SessionStore::open(path.clone()).unwrap();
    s.push(Message::user("retry"));
    store.append_new(&s.messages).unwrap();
    drop(store);
    let (_store, s) = SessionStore::open(path).unwrap();
    assert_eq!(texts(&s.messages), ["u0", "a0", "retry"]);
}
