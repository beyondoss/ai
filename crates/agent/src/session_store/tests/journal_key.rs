use super::*;

/// A session's first journal write makes its MCP task journal key, beside it (`0600`); a
/// session that never journals gets no file. Entries it seals read back after a reopen; a line
/// planted into the file without the key does not.
#[test]
fn a_session_keeps_its_journal_key_beside_it_and_refuses_planted_entries() {
    use crate::mcp_resume::TASK_ENTRY_KIND;
    let dir = tmpdir();
    let path = dir.path().join("s.jsonl");
    let mut store = SessionStore::create(path.clone(), SessionMeta::new("/w", "m")).unwrap();
    let key = journal_key_beside(&path);
    assert!(!key.exists(), "no key until the session journals something");
    store
        .append_journal(TASK_ENTRY_KIND, serde_json::json!({ "taskId": "real" }))
        .unwrap();
    assert!(key.is_file());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            fs::metadata(&key).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
    store
        .append_custom(TASK_ENTRY_KIND, serde_json::json!({ "taskId": "planted" }))
        .unwrap();
    drop(store);
    let (store, _) = SessionStore::open(path).unwrap();
    let tasks = store.active_journal(TASK_ENTRY_KIND);
    assert_eq!(tasks.len(), 1, "{tasks:?}");
    assert_eq!(tasks[0]["taskId"], "real");
    assert_eq!(store.active_custom(TASK_ENTRY_KIND).len(), 2);
}

/// A session from before per-session keys (journal entries, no key file) keeps the entries it
/// already has: opening it writes nothing, and its next journal write makes the key with those
/// entries accepted. A line planted after that is refused.
#[test]
fn a_session_without_a_journal_key_keeps_its_existing_entries() {
    use crate::mcp_resume::{RESULT_ENTRY_KIND, TASK_ENTRY_KIND};
    let dir = tmpdir();
    let path = dir.path().join("s.jsonl");
    let mut store = SessionStore::create(path.clone(), SessionMeta::new("/w", "m")).unwrap();
    store
        .append_custom(TASK_ENTRY_KIND, serde_json::json!({ "taskId": "old" }))
        .unwrap();
    store
        .append_custom(RESULT_ENTRY_KIND, serde_json::json!({ "toolUseId": "tu" }))
        .unwrap();
    drop(store);
    let key = journal_key_beside(&path);

    let (mut store, _) = SessionStore::open(path.clone()).unwrap();
    assert!(!key.exists(), "opening writes no key");
    assert_eq!(store.active_journal(TASK_ENTRY_KIND).len(), 1);
    assert_eq!(store.active_journal(RESULT_ENTRY_KIND).len(), 1);
    store
        .append_journal(TASK_ENTRY_KIND, serde_json::json!({ "taskId": "new" }))
        .unwrap();
    assert!(key.is_file());
    store
        .append_custom(TASK_ENTRY_KIND, serde_json::json!({ "taskId": "planted" }))
        .unwrap();
    drop(store);
    let (store, _) = SessionStore::open(path).unwrap();
    let tasks: Vec<_> = store
        .active_journal(TASK_ENTRY_KIND)
        .into_iter()
        .map(|t| t["taskId"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(tasks, ["old", "new"]);
    assert_eq!(store.active_journal(RESULT_ENTRY_KIND).len(), 1);
}

/// Deleting a session moves its journal key out of the session directory into `.trash` with
/// it; restoring brings it back, and the journal still verifies.
#[test]
fn a_trashed_session_takes_its_journal_key_along_and_brings_it_back() {
    use crate::mcp_resume::TASK_ENTRY_KIND;
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    store
        .append_journal(TASK_ENTRY_KIND, serde_json::json!({ "taskId": "t" }))
        .unwrap();
    let id = store.meta().id.clone();
    let path = store.path().to_path_buf();
    drop(store);
    let key = journal_key_beside(&path);
    assert!(key.is_file());

    repo.delete(&id).unwrap();
    assert!(!key.exists(), "the key left the session directory");
    assert!(
        dir.path()
            .join(".trash")
            .join(key.file_name().unwrap())
            .is_file(),
        "the key is in the trash beside its session"
    );

    assert!(repo.restore_session(&id).unwrap());
    assert!(key.is_file(), "the key came back with its session");
    let (store, _) = SessionStore::open(path).unwrap();
    assert_eq!(store.active_journal(TASK_ENTRY_KIND).len(), 1);
}

/// A key still under the old name (not yet carried over) is trashed and restored with its
/// session too: left behind, the restored session would read as keyless.
#[test]
fn a_trashed_session_takes_an_old_name_key_along() {
    use crate::mcp_resume::{JournalAuth, TASK_ENTRY_KIND};
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    store
        .append_journal(TASK_ENTRY_KIND, serde_json::json!({ "taskId": "t" }))
        .unwrap();
    let id = store.meta().id.clone();
    let path = store.path().to_path_buf();
    drop(store);
    let legacy = path.with_extension(JournalAuth::SIDECAR);
    fs::rename(journal_key_beside(&path), &legacy).unwrap();

    repo.delete(&id).unwrap();
    assert!(
        !legacy.exists(),
        "the old-name key left the session directory"
    );
    assert!(
        dir.path()
            .join(".trash")
            .join(legacy.file_name().unwrap())
            .is_file()
    );
    assert!(repo.restore_session(&id).unwrap());
    assert!(legacy.is_file(), "the old-name key came back");
    let (store, _) = SessionStore::open(path).unwrap();
    assert_eq!(store.active_journal(TASK_ENTRY_KIND).len(), 1);
    assert!(
        matches!(store.journal, JournalAuth::Keyed { .. }),
        "the restored session is keyed, not keyless"
    );
}

/// Sessions whose names differ only after their last dot each get their own key: the key's
/// name is a suffix on the whole file name, not a replaced extension.
#[test]
fn sessions_named_alike_keep_separate_journal_keys() {
    use crate::mcp_resume::TASK_ENTRY_KIND;
    let dir = tmpdir();
    let mut keys = Vec::new();
    for name in ["work.1", "work.2"] {
        let path = dir.path().join(name);
        let mut store = SessionStore::create(path.clone(), SessionMeta::new("/w", "m")).unwrap();
        store
            .append_journal(TASK_ENTRY_KIND, serde_json::json!({ "taskId": name }))
            .unwrap();
        keys.push(fs::read(journal_key_beside(&path)).unwrap());
    }
    assert_ne!(keys[0], keys[1], "one key file shared by two sessions");
    for name in ["work.1", "work.2"] {
        let (store, _) = SessionStore::open(dir.path().join(name)).unwrap();
        let tasks = store.active_journal(TASK_ENTRY_KIND);
        assert_eq!(tasks.len(), 1, "{name}: {tasks:?}");
        assert_eq!(tasks[0]["taskId"], name);
    }
}

/// A key under the old `with_extension` name still verifies the entries it sealed, and the next
/// journal write carries it over to the suffixed name rather than making a new one.
#[test]
fn a_journal_key_under_the_old_name_is_carried_over() {
    use crate::mcp_resume::{JournalAuth, TASK_ENTRY_KIND};
    let dir = tmpdir();
    let path = dir.path().join("s.jsonl");
    let mut store = SessionStore::create(path.clone(), SessionMeta::new("/w", "m")).unwrap();
    store
        .append_journal(TASK_ENTRY_KIND, serde_json::json!({ "taskId": "old" }))
        .unwrap();
    drop(store);
    let legacy = path.with_extension(JournalAuth::SIDECAR);
    fs::rename(journal_key_beside(&path), &legacy).unwrap();

    let (mut store, _) = SessionStore::open(path.clone()).unwrap();
    assert_eq!(store.active_journal(TASK_ENTRY_KIND).len(), 1);
    store
        .append_custom(TASK_ENTRY_KIND, serde_json::json!({ "taskId": "planted" }))
        .unwrap();
    store
        .append_journal(TASK_ENTRY_KIND, serde_json::json!({ "taskId": "new" }))
        .unwrap();
    assert_eq!(
        fs::read(journal_key_beside(&path)).unwrap(),
        fs::read(&legacy).unwrap(),
        "the old key is carried over, not replaced"
    );
    drop(store);
    let (store, _) = SessionStore::open(path).unwrap();
    let tasks: Vec<_> = store
        .active_journal(TASK_ENTRY_KIND)
        .into_iter()
        .map(|t| t["taskId"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(tasks, ["old", "new"]);
}

/// Set by [`concurrent_first_journal_writers_all_use_one_key`] for the child processes it
/// runs this test binary as.
const WRITER_SESSION: &str = "JOURNAL_KEY_WRITER_SESSION";

/// One of [`concurrent_first_journal_writers_all_use_one_key`]'s writer processes: waits for
/// the go file, opens the session and journals one entry. A no-op when run on its own.
#[test]
#[ignore = "a child process of concurrent_first_journal_writers_all_use_one_key"]
fn journal_key_writer_child() {
    let Ok(session) = std::env::var(WRITER_SESSION) else {
        return;
    };
    let session = PathBuf::from(session);
    let id = std::env::var("JOURNAL_KEY_WRITER_ID").unwrap();
    journal_one_after_go(&session, &id);
}

fn journal_one_after_go(session: &Path, id: &str) {
    let go = session.with_extension("go");
    fs::write(session.with_extension(format!("ready.{id}")), b"").unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while !go.exists() {
        assert!(std::time::Instant::now() < deadline, "no go");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let (mut store, _) = SessionStore::open(session.to_path_buf()).unwrap();
    store
        .append_journal(
            crate::mcp_resume::TASK_ENTRY_KIND,
            serde_json::json!({ "taskId": id }),
        )
        .unwrap();
}

/// Every first journal writer, threads of this process and other processes alike, ends up
/// using the one key on disk, so every entry any of them journals verifies against it. A writer
/// that believed in a key that lost the race would have its entries refused on replay.
#[test]
fn concurrent_first_journal_writers_all_use_one_key() {
    use crate::mcp_resume::{JournalAuth, TASK_ENTRY_KIND};
    const PROCESSES: usize = 4;
    const THREADS: usize = 8;
    let dir = tmpdir();
    let path = dir.path().join("s.jsonl");
    drop(SessionStore::create(path.clone(), SessionMeta::new("/w", "m")).unwrap());

    let exe = std::env::current_exe().unwrap();
    let children: Vec<_> = (0..PROCESSES)
        .map(|i| {
            std::process::Command::new(&exe)
                .args([
                    "session_store::tests::journal_key::journal_key_writer_child",
                    "--exact",
                    "--ignored",
                    "--test-threads=1",
                ])
                .env(WRITER_SESSION, &path)
                .env("JOURNAL_KEY_WRITER_ID", format!("process-{i}"))
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .unwrap()
        })
        .collect();
    let threads: Vec<_> = (0..THREADS)
        .map(|i| {
            let path = path.clone();
            std::thread::spawn(move || journal_one_after_go(&path, &format!("thread-{i}")))
        })
        .collect();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while fs::read_dir(dir.path())
        .unwrap()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().contains(".ready."))
        .count()
        < PROCESSES + THREADS
    {
        assert!(
            std::time::Instant::now() < deadline,
            "writers never got ready"
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    fs::write(path.with_extension("go"), b"").unwrap();
    for thread in threads {
        thread.join().unwrap();
    }
    for mut child in children {
        assert!(child.wait().unwrap().success(), "a writer process failed");
    }

    let auth = JournalAuth::from_sidecar(&fs::read(journal_key_beside(&path)).unwrap())
        .expect("one key on disk");
    let entries: Vec<serde_json::Value> = fs::read_to_string(&path)
        .unwrap()
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter(|v| v["type"] == "custom" && v["kind"] == TASK_ENTRY_KIND)
        .map(|v| v["data"].clone())
        .collect();
    assert_eq!(entries.len(), PROCESSES + THREADS);
    for entry in &entries {
        assert!(
            entry.get("mac").is_some() && auth.accepts(TASK_ENTRY_KIND, entry),
            "sealed under a key that is not the one on disk: {entry}"
        );
    }
}
