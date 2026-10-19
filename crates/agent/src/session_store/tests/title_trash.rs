use super::*;

#[test]
fn set_title_persists() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let id = store.meta().id.clone();
    let mut session = Session::new();
    session.user("hi");
    store.append_new(&session.messages).unwrap();
    store.set_title("My Session").unwrap();
    let metas = repo.list().unwrap();
    let found = metas.iter().find(|m| m.id == id).unwrap();
    assert_eq!(found.title.as_deref(), Some("My Session"));
    // A pure title rewrite drops no messages, so it leaves no compaction provenance.
    assert_eq!(found.compactions, 0);
}

#[test]
fn set_title_appends_rather_than_rewriting_the_file() {
    // The whole point of Track M17: renaming must cost an O(1) append, not a rewrite of every
    // message already on disk. Prove it directly — the bytes written before the rename are still
    // there afterward, byte for byte, with the rename's line appended after them.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("hi");
    store.append_new(&session.messages).unwrap();

    let before = std::fs::read(store.path()).unwrap();
    store.set_title("Renamed").unwrap();
    let after = std::fs::read(store.path()).unwrap();

    assert!(
        after.starts_with(&before),
        "a rename must only append — every byte written before it must survive unchanged"
    );
    assert!(
        after.len() > before.len(),
        "the rename must add at least the new TitleChange line"
    );
}

#[test]
fn set_title_last_write_wins_and_survives_reopen() {
    let dir = tmpdir();
    let path = dir.path().join("s.jsonl");
    let mut store = SessionStore::create(path.clone(), SessionMeta::new("/w", "m")).unwrap();
    store.set_title("First").unwrap();
    store.set_title("Second").unwrap();
    assert_eq!(store.meta().title.as_deref(), Some("Second"));

    // Both `SessionStore::open` (the tree-building read) and `read_listing` (the cheap listing
    // scan) must independently resolve the *last* rename, not the first.
    let (reopened, _session) = SessionStore::open(path.clone()).unwrap();
    assert_eq!(reopened.meta().title.as_deref(), Some("Second"));

    let listed = read_listing(&path, None, &Layout::File).unwrap();
    assert_eq!(listed.title.as_deref(), Some("Second"));
}

#[test]
fn set_title_strips_newlines() {
    // Matches pi's `appendSessionInfo` sanitization: a raw newline in a caller-supplied title
    // (RPC client, extension) would otherwise split a session-list line or corrupt a terminal
    // display — collapse any run of them into a single space instead.
    let dir = tmpdir();
    let path = dir.path().join("s.jsonl");
    let mut store = SessionStore::create(path.clone(), SessionMeta::new("/w", "m")).unwrap();
    store.set_title("hello\nworld\r\nagain").unwrap();
    assert_eq!(store.meta().title.as_deref(), Some("hello world again"));

    let (reopened, _session) = SessionStore::open(path.clone()).unwrap();
    assert_eq!(reopened.meta().title.as_deref(), Some("hello world again"));
    let listed = read_listing(&path, None, &Layout::File).unwrap();
    assert_eq!(listed.title.as_deref(), Some("hello world again"));
}

#[test]
fn set_title_to_empty_or_whitespace_clears_the_title() {
    // Matches pi's "empty names explicitly clear the session title" — a blank title isn't a
    // present-but-empty display value, it's the caller asking to unset the title entirely.
    let dir = tmpdir();
    let path = dir.path().join("s.jsonl");
    let mut store = SessionStore::create(path.clone(), SessionMeta::new("/w", "m")).unwrap();
    store.set_title("My Session").unwrap();
    assert_eq!(store.meta().title.as_deref(), Some("My Session"));

    store.set_title("   \n\r\n  ").unwrap();
    assert_eq!(store.meta().title, None);

    let (reopened, _session) = SessionStore::open(path.clone()).unwrap();
    assert_eq!(reopened.meta().title, None);
    let listed = read_listing(&path, None, &Layout::File).unwrap();
    assert_eq!(listed.title, None);
}

#[test]
fn delete_removes_session() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let id = store.meta().id.clone();
    assert_eq!(repo.list().unwrap().len(), 1);
    repo.delete(&id).unwrap();
    assert_eq!(repo.list().unwrap().len(), 0);
}

#[test]
fn delete_is_idempotent() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let id = store.meta().id.clone();

    // First delete removes it; a second delete (and a delete of a never-seen id) are no-op successes.
    repo.delete(&id).unwrap();
    repo.delete(&id).unwrap();
    repo.delete("does-not-exist").unwrap();
    assert!(repo.list().unwrap().is_empty());
}

#[test]
fn delete_moves_to_trash_and_list_no_longer_shows_it() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let id = store.meta().id.clone();
    let original_path = store.path().to_path_buf();

    repo.delete(&id).unwrap();

    assert!(
        repo.list().unwrap().is_empty(),
        "a trashed session must not appear in list()"
    );
    assert!(
        !original_path.exists(),
        "the file must no longer be at its original path"
    );
    let trash_dir = dir.path().join(".trash");
    let trashed: Vec<_> = fs::read_dir(&trash_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .collect();
    assert_eq!(
        trashed.len(),
        1,
        "the deleted session's file should have moved into .trash: {trashed:?}"
    );
    assert_eq!(trashed[0].file_name(), original_path.file_name().unwrap());
}

#[test]
fn list_trash_is_empty_before_anything_is_deleted() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    repo.create(SessionMeta::new("/w", "m")).unwrap();
    assert!(
        repo.list_trash().unwrap().is_empty(),
        "nothing has been deleted yet, so .trash/ doesn't even exist"
    );
}

#[test]
fn list_trash_reports_a_deleted_session_by_id() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let id = store.meta().id.clone();

    repo.delete(&id).unwrap();

    let trash = repo.list_trash().unwrap();
    assert_eq!(trash.len(), 1);
    assert_eq!(trash[0].id, id);
    assert!(trash[0].deleted_at.is_some());
    assert!(trash[0].original_path.ends_with(&format!("_{id}.jsonl")));
}

#[test]
fn restore_session_moves_a_trashed_session_back_and_makes_it_listable_again() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let id = store.meta().id.clone();

    repo.delete(&id).unwrap();
    assert!(repo.list().unwrap().is_empty());

    let restored = repo.restore_session(&id).unwrap();
    assert!(
        restored,
        "restore_session must report it actually moved something"
    );
    assert!(
        repo.list_trash().unwrap().is_empty(),
        "the entry must no longer be in .trash/ once restored"
    );
    let listed = repo.list().unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].id, id);
}

#[test]
fn delete_and_restore_move_the_sibling_session_memory_dir_too() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let id = store.meta().id.clone();
    let jsonl = store.path().to_path_buf();
    // Simulate `/session` working memory written during the session — the `<...>.memory/` sibling.
    let mem_dir = jsonl.with_extension("memory");
    fs::create_dir_all(&mem_dir).unwrap();
    fs::write(mem_dir.join("facts.md"), "the prod DB port is 5433\n").unwrap();

    repo.delete(&id).unwrap();
    assert!(
        !mem_dir.exists(),
        "the memory dir must leave its original location when the session is trashed"
    );
    let trashed_mem = dir.path().join(".trash").join(mem_dir.file_name().unwrap());
    assert!(
        trashed_mem.join("facts.md").exists(),
        "the memory dir and its contents must move into .trash/ alongside the session"
    );

    assert!(repo.restore_session(&id).unwrap());
    assert!(
        !trashed_mem.exists(),
        "the memory dir must leave .trash/ when the session is restored"
    );
    assert_eq!(
        fs::read_to_string(mem_dir.join("facts.md")).unwrap(),
        "the prod DB port is 5433\n",
        "the restored working memory must have its contents intact"
    );
}

#[test]
fn delete_restore_and_hard_delete_carry_the_mcp_app_sidecars_with_the_session() {
    use crate::tools::mcp_apps::Sidecar;
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let id = store.meta().id.clone();
    let sidecars: Vec<PathBuf> = [Sidecar::CONTEXT, Sidecar::VIEWS]
        .into_iter()
        .map(|kind| {
            let s = Sidecar::new(store.path(), kind, None);
            s.write(Some(b"<html>secret view</html>")).unwrap();
            s.path().to_path_buf()
        })
        .collect();
    let trashed = |f: &PathBuf| dir.path().join(".trash").join(f.file_name().unwrap());

    repo.delete(&id).unwrap();
    for f in &sidecars {
        assert!(!f.exists(), "{} must leave with its session", f.display());
        assert!(trashed(f).exists(), "and land in .trash/ beside it");
    }

    assert!(repo.restore_session(&id).unwrap());
    for f in &sidecars {
        assert_eq!(fs::read(f).unwrap(), b"<html>secret view</html>");
        assert!(!trashed(f).exists(), "a restore brings it back out");
    }

    // No `.trash/` to move into (a file is in the way): the hard delete removes them too.
    fs::remove_dir_all(dir.path().join(".trash")).unwrap();
    fs::write(dir.path().join(".trash"), "").unwrap();
    repo.delete(&id).unwrap();
    assert!(!store.path().exists());
    for f in &sidecars {
        assert!(
            !f.exists(),
            "{} must not outlive a hard delete",
            f.display()
        );
    }
}

#[test]
fn delete_and_restore_move_the_mcp_events_state_file_too() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let id = store.meta().id.clone();
    let sidecar = store.path().with_extension("mcp-events.json");
    let log = store.path().with_extension("mcp-events.log");
    fs::write(&sidecar, "{}").unwrap();
    fs::write(&log, "{\"add\":{}}\n").unwrap();

    repo.delete(&id).unwrap();
    for f in [&sidecar, &log] {
        assert!(!f.exists(), "the events state leaves with its session");
        let trashed = dir.path().join(".trash").join(f.file_name().unwrap());
        assert!(trashed.exists(), "and lands in .trash/ beside it");
    }

    assert!(repo.restore_session(&id).unwrap());
    for f in [&sidecar, &log] {
        assert!(f.exists(), "a restore brings it back");
        assert!(
            !dir.path()
                .join(".trash")
                .join(f.file_name().unwrap())
                .exists()
        );
    }
}

#[test]
fn restore_session_of_an_unknown_id_returns_false_not_an_error() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    assert!(!repo.restore_session("never-existed").unwrap());
}

#[test]
fn restore_session_fails_clearly_when_the_destination_is_already_occupied() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let id = store.meta().id.clone();
    let original_path = store.path().to_path_buf();

    repo.delete(&id).unwrap();
    // Something else now occupies the original path (e.g. a new session using a colliding id is
    // vanishingly unlikely in practice, but a hand-placed file is enough to prove the guard works).
    fs::write(&original_path, "occupied").unwrap();

    let err = repo.restore_session(&id).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::AlreadyExists);
    // The trashed copy must still be sitting in `.trash/` untouched — a failed restore is not a
    // silent delete of the only remaining copy.
    assert_eq!(repo.list_trash().unwrap().len(), 1);
}
