use super::*;

/// The root cause of `serve_ws::tests::a_session_directory_with_a_segment_is_left_alone`'s
/// flakes under load: a child forked by *another* thread (a tool, an MCP server) shares the
/// lock's open file description until it execs, so a lock released by closing its descriptor was
/// still held about 6% of the time while processes were being spawned (180/3000 measured). The
/// OFD lock is inherited the same way, but it is released with an explicit unlock, which drops it
/// whatever copies a child holds: zero.
#[test]
fn a_released_session_lock_is_free_at_once_while_other_threads_spawn_processes() {
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let spawners: Vec<_> = (0..8)
        .map(|_| {
            let stop = stop.clone();
            std::thread::spawn(move || {
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    let _ = std::process::Command::new("true").status();
                }
            })
        })
        .collect();
    let root = tempfile::tempdir().unwrap();
    let mut held = 0;
    for i in 0..3000 {
        let p = root.path().join(format!("s{i}"));
        std::fs::create_dir_all(&p).unwrap();
        let lock = acquire_session_lock(&p).unwrap().unwrap();
        drop(lock);
        if acquire_session_lock(&p).unwrap().is_none() {
            held += 1;
        }
    }
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    for s in spawners {
        s.join().unwrap();
    }
    assert_eq!(
        held, 0,
        "a released lock was still held by a forked child {held}/3000 times"
    );
}

/// The lock reached through another name — a symlink (`innocent.txt -> <session>/<lock file>`,
/// which the model can make with `bash`) or a hard link — by every in-process reader that does
/// not know it is a lock file: the file tools' read, write, edit and search, the memory tool's
/// view (under `<session>/memory`), and the `@file` read (`std::fs::read_to_string`). Each opens and
/// closes the lock file on its own descriptor, and none of them releases the lock, which belongs
/// to the open file description that took it.
#[cfg(unix)]
#[tokio::test]
async fn every_in_process_reader_reaching_the_lock_file_leaves_it_held() {
    use agent_core::Tool as _;
    let dir = tmpdir();
    let session_dir = dir.path().join("s1");
    std::fs::create_dir_all(session_dir.join("memory")).unwrap();
    let lock = acquire_session_lock(&session_dir).unwrap().unwrap();
    let target = crate::file_lock::Target::Dir(&session_dir);
    let Some(true) = crate::file_lock::tests::held_for_another_process(target) else {
        eprintln!("no python3: skipping");
        return;
    };
    let record = crate::file_lock::tests::record_path(target);
    let work = dir.path().join("work");
    std::fs::create_dir_all(&work).unwrap();
    let symlink = work.join("innocent.txt");
    std::os::unix::fs::symlink(&record, &symlink).unwrap();
    let hardlink = work.join("also-innocent.txt");
    std::fs::hard_link(&record, &hardlink).unwrap();
    std::os::unix::fs::symlink(&record, session_dir.join("memory").join("notes.md")).unwrap();

    for link in [&symlink, &hardlink] {
        let p = link.to_str().unwrap();
        let _ = crate::tools::read::Read::new(&work)
            .run(serde_json::json!({ "path": p }))
            .await;
        let _ = crate::tools::write::Write::new(&work)
            .run(serde_json::json!({ "path": p, "content": "scribble" }))
            .await;
        let _ = crate::tools::edit::Edit::new(&work)
            .run(serde_json::json!({ "path": p, "old_string": "", "new_string": "x" }))
            .await;
        // What `@file` does with the path it is given.
        let _ = std::fs::read_to_string(link);
    }
    let _ = crate::tools::grep::Grep::new(&work)
        .run(serde_json::json!({ "pattern": "x", "path": work.to_str().unwrap() }))
        .await;
    let memory = crate::tools::memory::Memory::new(std::sync::Arc::new(
        crate::memory::file::FileBackend::session_at(session_dir.join("memory")),
    ));
    let _ = memory
        .run(serde_json::json!({ "command": "view", "path": "/session/notes.md" }))
        .await;

    assert_eq!(
        crate::file_lock::tests::held_for_another_process(target),
        Some(true),
        "no reader of the lock file released it"
    );
    assert!(
        acquire_session_lock(&session_dir).unwrap().is_none(),
        "nor can this process take it a second time"
    );
    assert_eq!(
        std::fs::metadata(&record).unwrap().len(),
        0,
        "and nothing replaced or wrote into the lock file"
    );
    drop(lock);
    assert_eq!(
        crate::file_lock::tests::held_for_another_process(target),
        Some(false)
    );
}

/// Every public path that reads a locked session directory — listing, opening, forking, the
/// file tools' read and search of every file in it, the lock file included — leaves the lock
/// held for another process: none of their descriptors is the one that holds it.
#[tokio::test]
async fn reading_a_locked_session_through_any_public_api_leaves_it_locked() {
    use crate::tools::fs::FsBackend as _;
    let dir = tmpdir();
    let repo = SessionRepo::open_with(
        dir.path(),
        RepoOptions {
            layout: Layout::Segmented { codec: None },
            ..RepoOptions::default()
        },
    )
    .unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("hello lock");
    store.append_new(&session.messages).unwrap();
    let id = store.meta().id.clone();
    let session_dir = store.path().to_path_buf();
    drop(store);
    let lock = acquire_session_lock(&session_dir).unwrap().unwrap();
    let target = crate::file_lock::Target::Dir(&session_dir);
    let Some(true) = crate::file_lock::tests::held_for_another_process(target) else {
        eprintln!("no python3: skipping");
        return;
    };

    let _ = repo.list().unwrap();
    let _ = repo.open_id_read_only(&id).unwrap();
    let _ = repo.fork(&id, usize::MAX).unwrap();
    let fs = crate::tools::fs::local::LocalFs::new();
    for entry in std::fs::read_dir(&session_dir).unwrap().flatten() {
        // Read every file there by the tool path; a lock file is refused, never opened.
        let _ = fs.read_bytes(&entry.path(), 0, 1024).await;
    }
    let _ = fs
        .search(&crate::tools::fs::local::query(
            "lock",
            dir.path().to_path_buf(),
            100,
        ))
        .await;
    assert_eq!(
        crate::file_lock::tests::held_for_another_process(target),
        Some(true),
        "no public read of the session released its lock"
    );
    drop(lock);
    assert_eq!(
        crate::file_lock::tests::held_for_another_process(target),
        Some(false)
    );
}

/// A single-file session whose writer lock is held is not deleted out from under its writer: the
/// delete is refused, and once the writer lets go it goes through.
#[test]
fn deleting_a_single_file_session_its_writer_holds_is_refused() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let id = store.meta().id.clone();
    let path = repo.find_path(&id).unwrap().unwrap();
    let lock = lock_session_writer(&path).unwrap().unwrap();

    let err = repo.delete(&id).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::WouldBlock);
    assert!(path.is_file(), "the held session must stay where it is");
    let mut session = Session::new();
    session.user("still saves");
    store.append_new(&session.messages).unwrap();

    drop(lock);
    repo.delete(&id).unwrap();
    assert!(!path.exists());
}
