use super::*;

#[test]
fn list_is_newest_first_and_fork_links_parent() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut a = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut sa = Session::new();
    sa.user("one");
    sa.user("two");
    a.append_new(&sa.messages).unwrap();
    let id_a = a.meta().id.clone();

    // Fork the first message only.
    let (forked, fsession) = repo.fork(&id_a, 1).unwrap();
    assert_eq!(fsession.messages.len(), 1);
    assert_eq!(forked.meta().parent.as_deref(), Some(id_a.as_str()));

    let metas = repo.list().unwrap();
    assert_eq!(metas.len(), 2);
    // Newest first: the fork was created after `a`.
    assert!(metas[0].created_at >= metas[1].created_at);
}

#[test]
fn a_fork_does_not_inherit_the_sources_exec_endpoint() {
    // A fork is a new session, and a sandbox belongs to the session it was provisioned for.
    // Inheriting it would be the same cross-session reuse that clearing on switch exists to
    // prevent — and it is the kind of thing that stays correct only while somebody remembers, so
    // it is pinned here rather than left to `fork`'s field-by-field copying staying honest.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut src = repo
        .create(SessionMeta::new(String::from("/w"), "m"))
        .unwrap();
    let mut s = Session::new();
    s.user("hi");
    src.append_new(&s.messages).unwrap();
    src.set_exec_endpoint(Some(serde_json::json!({ "command": "docker exec a {}" })))
        .unwrap();
    assert!(
        src.meta().exec_endpoint.is_some(),
        "sanity: the source has one"
    );
    let src_id = src.meta().id.clone();

    let (forked, _) = repo.fork(&src_id, usize::MAX).unwrap();
    assert_eq!(
        forked.meta().exec_endpoint,
        None,
        "the fork inherited the source's sandbox"
    );
}

#[test]
fn fork_from_path_copies_a_source_from_an_unrelated_repo_and_stamps_the_target_cwd() {
    // The cross-project primitive `fork_by_arg` builds on: the source need not live in `self.dir`
    // at all, and the new session's `cwd` is the *target* project, not wherever the source was
    // originally recorded against — matching pi's `forkFrom(path, targetCwd, …)`.
    let source_dir = tmpdir();
    let source_repo = SessionRepo::open(source_dir.path()).unwrap();
    let mut source = source_repo
        .create(SessionMeta::new("/some/other/project", "m"))
        .unwrap();
    let mut s = Session::new();
    s.user("hello from project A");
    source.append_new(&s.messages).unwrap();
    let source_id = source.meta().id.clone();
    let source_path = source_dir
        .path()
        .read_dir()
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();

    let target_dir = tmpdir();
    let target_repo = SessionRepo::open(target_dir.path()).unwrap();
    let (forked, fsession) = target_repo
        .fork_from_path(&source_path, "/current/project", usize::MAX)
        .unwrap();

    assert_eq!(fsession.messages.len(), 1);
    assert_eq!(forked.meta().parent.as_deref(), Some(source_id.as_str()));
    assert_eq!(forked.meta().cwd, "/current/project");
    // The source file itself is untouched — this is a copy, not a move.
    let (_, resumed_source) = SessionStore::open(source_path).unwrap();
    assert_eq!(resumed_source.messages.len(), 1);
}

#[test]
fn fork_by_arg_with_a_path_like_argument_opens_that_file_directly_no_search() {
    let source_dir = tmpdir();
    let source_repo = SessionRepo::open(source_dir.path()).unwrap();
    source_repo
        .create(SessionMeta::new("/wherever", "m"))
        .unwrap();
    let source_path = source_dir
        .path()
        .read_dir()
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();

    let target_dir = tmpdir();
    let target_repo = SessionRepo::open(target_dir.path()).unwrap();
    // An empty `sessions_root` proves this path never falls through to the cross-project search —
    // a path-like argument is resolved directly.
    let empty_root = tmpdir();
    let (forked, _) = fork_by_arg(
        source_path.to_str().unwrap(),
        &target_repo,
        "/current",
        empty_root.path(),
        usize::MAX,
    )
    .unwrap();
    assert_eq!(forked.meta().cwd, "/current");
}

#[test]
fn fork_by_arg_with_an_id_present_in_the_current_project_forks_in_place() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let source = repo.create(SessionMeta::new("/current", "m")).unwrap();
    let source_id = source.meta().id.clone();

    // No other project directory exists at all — proves the same-project match is tried (and
    // wins) before any cross-project search would even be attempted.
    let empty_root = tmpdir();
    let (forked, _) =
        fork_by_arg(&source_id, &repo, "/current", empty_root.path(), usize::MAX).unwrap();
    assert_eq!(forked.meta().parent.as_deref(), Some(source_id.as_str()));
    assert_eq!(forked.meta().cwd, "/current");
}

#[test]
fn fork_by_arg_with_an_id_found_in_no_project_is_a_clear_not_found_error() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let empty_root = tmpdir();
    let result = fork_by_arg(
        "no-such-id",
        &repo,
        "/current",
        empty_root.path(),
        usize::MAX,
    );
    let err = match result {
        Ok(_) => panic!("expected a not-found error"),
        Err(e) => e,
    };
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    assert!(err.to_string().contains("no-such-id"), "{err}");
}

#[test]
fn fork_by_arg_falls_back_to_a_session_found_in_a_different_projects_own_directory() {
    // The cross-project search itself, exercised directly against a synthetic `sessions_root`
    // rather than the real `$HOME` (which `default_session_dir` reads) — the end-to-end version of
    // this same fallback, wired through the real default directory, lives in the `run` binary's own
    // `--fork` e2e tests.
    let root = tmpdir();
    let other_project_repo =
        SessionRepo::open(root.path().join(encode_cwd("/some/other/project"))).unwrap();
    let source = other_project_repo
        .create(SessionMeta::new("/some/other/project", "m"))
        .unwrap();
    let source_id = source.meta().id.clone();

    let current_repo = SessionRepo::open(root.path().join(encode_cwd("/current/project"))).unwrap();
    let (forked, _) = fork_by_arg(
        &source_id,
        &current_repo,
        "/current/project",
        root.path(),
        usize::MAX,
    )
    .unwrap();
    assert_eq!(forked.meta().parent.as_deref(), Some(source_id.as_str()));
    assert_eq!(
        forked.meta().cwd,
        "/current/project",
        "the fork's cwd must be the current project, not the source's original one"
    );
}

#[test]
fn fork_by_arg_finds_a_cross_project_session_even_when_its_directory_name_does_not_encode_its_cwd()
{
    // Pi-parity fix (`run --session-dir`/`AI_AGENT_SESSION_DIR`): the cross-project fallback used to
    // recompute the source's directory from its own recorded `cwd` via
    // `default_session_dir`/`encode_cwd`, which only holds for the *default* `~/.beyond/sessions/
    // <encoded-cwd>/` layout. An arbitrarily-named directory (exactly what `--session-dir` lets a
    // caller point at) broke that assumption and the fork would fail with a bogus "not found" even
    // though the session was sitting right there under `sessions_root`.
    let root = tmpdir();
    // Deliberately an arbitrary name, unrelated to `encode_cwd` of the session's own `cwd` below.
    let other_project_repo = SessionRepo::open(root.path().join("arbitrarily-named-repo")).unwrap();
    let source = other_project_repo
        .create(SessionMeta::new("/some/other/project", "m"))
        .unwrap();
    let source_id = source.meta().id.clone();

    let current_repo = SessionRepo::open(root.path().join("another-arbitrary-name")).unwrap();
    let (forked, _) = fork_by_arg(
        &source_id,
        &current_repo,
        "/current/project",
        root.path(),
        usize::MAX,
    )
    .unwrap();
    assert_eq!(forked.meta().parent.as_deref(), Some(source_id.as_str()));
    assert_eq!(forked.meta().cwd, "/current/project");
}

#[test]
fn find_path_resolves_a_unique_prefix_when_no_exact_match_exists() {
    // Pi-parity fix: matching used to be exact-only everywhere — a caller had to type the full id,
    // unlike pi's own `resolveSessionPath`, which accepts a shortened prefix.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let id = store.meta().id.clone();
    let prefix = &id[..id.len() / 2];

    let (_, restored) = repo.open_id(prefix).unwrap();
    assert_eq!(restored.messages.len(), 0);
}

#[test]
fn find_path_prefers_an_exact_match_over_a_prefix_match() {
    // An id that happens to *also* be a prefix of some other session's id must still resolve to
    // itself exactly, not get shadowed by the ambiguity check below.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let short = repo.create(SessionMeta::with_id("abc", "/w", "m")).unwrap();
    repo.create(SessionMeta::with_id("abcdef", "/w", "m"))
        .unwrap();

    let (_, _) = repo.open_id("abc").unwrap();
    let path = repo.find_path("abc").unwrap().unwrap();
    assert!(path.to_string_lossy().contains(&short.meta().id));
}

#[test]
fn find_path_reports_an_ambiguous_prefix_naming_every_candidate_instead_of_guessing() {
    // Pi's own `resolveSessionPath` silently picks whichever candidate sorts first on an ambiguous
    // prefix — a real footgun (acting on the wrong session with no warning). This crate errors
    // instead, naming every id that matched.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    repo.create(SessionMeta::with_id("abc111", "/w", "m"))
        .unwrap();
    repo.create(SessionMeta::with_id("abc222", "/w", "m"))
        .unwrap();

    let err = repo.open_id("abc").map(|_| ()).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    let msg = err.to_string();
    assert!(msg.contains("abc111") && msg.contains("abc222"), "{msg}");
}

#[test]
fn find_path_returns_not_found_for_a_prefix_matching_nothing_at_all() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    repo.create(SessionMeta::with_id("abc111", "/w", "m"))
        .unwrap();

    let err = repo.open_id("zzz").map(|_| ()).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
}

#[test]
fn delete_by_prefix_removes_the_uniquely_matched_session() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let id = store.meta().id.clone();
    let prefix = &id[..id.len() / 2];

    repo.delete(prefix).unwrap();
    assert!(repo.list().unwrap().is_empty());
}

#[test]
fn delete_by_an_ambiguous_prefix_errors_instead_of_silently_no_opping() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    repo.create(SessionMeta::with_id("abc111", "/w", "m"))
        .unwrap();
    repo.create(SessionMeta::with_id("abc222", "/w", "m"))
        .unwrap();

    let err = repo.delete("abc").unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    // Neither candidate was touched.
    assert_eq!(repo.list().unwrap().len(), 2);
}

#[test]
fn fork_by_arg_resolves_a_prefix_within_the_current_project_without_a_cross_project_search() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let store = repo.create(SessionMeta::new("/current", "m")).unwrap();
    let id = store.meta().id.clone();
    let prefix = &id[..id.len() / 2];

    // An empty `sessions_root` proves the cross-project fallback was never needed — the prefix
    // resolved within `repo` itself.
    let empty_root = tmpdir();
    let (forked, _) =
        fork_by_arg(prefix, &repo, "/current", empty_root.path(), usize::MAX).unwrap();
    assert_eq!(forked.meta().parent.as_deref(), Some(id.as_str()));
}

#[test]
fn fork_by_arg_finds_a_unique_prefix_across_projects() {
    let root = tmpdir();
    let other_repo = SessionRepo::open(root.path().join("other-project")).unwrap();
    let source = other_repo
        .create(SessionMeta::new("/other/project", "m"))
        .unwrap();
    let source_id = source.meta().id.clone();
    let prefix = &source_id[..source_id.len() / 2];

    let current_repo = SessionRepo::open(root.path().join("current-project")).unwrap();
    let (forked, _) = fork_by_arg(
        prefix,
        &current_repo,
        "/current/project",
        root.path(),
        usize::MAX,
    )
    .unwrap();
    assert_eq!(forked.meta().parent.as_deref(), Some(source_id.as_str()));
}

#[test]
fn list_with_progress_reports_every_file_exactly_once_up_to_the_total() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    for i in 0..12 {
        repo.create(SessionMeta::new(format!("/w{i}"), "m"))
            .unwrap();
    }

    let calls: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
    let metas = repo
        .list_with_progress(|scanned, total| calls.lock().unwrap().push((scanned, total)))
        .unwrap();
    assert_eq!(metas.len(), 12, "parallel scan must not drop any file");

    let calls = calls.into_inner().unwrap();
    assert_eq!(calls.len(), 12, "exactly one progress call per file");
    assert!(
        calls.iter().all(|&(_, total)| total == 12),
        "total must be stable across every call: {calls:?}"
    );
    let mut scanned: Vec<usize> = calls.iter().map(|&(s, _)| s).collect();
    scanned.sort_unstable();
    assert_eq!(
        scanned,
        (1..=12).collect::<Vec<_>>(),
        "scanned counts must cover 1..=total exactly once each, even with concurrent workers: {scanned:?}"
    );
}

#[test]
fn list_with_progress_returns_the_same_sessions_as_list() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    for i in 0..8 {
        repo.create(SessionMeta::new(format!("/w{i}"), "m"))
            .unwrap();
    }

    let mut plain: Vec<String> = repo.list().unwrap().into_iter().map(|m| m.id).collect();
    let mut via_progress: Vec<String> = repo
        .list_with_progress(|_, _| {})
        .unwrap()
        .into_iter()
        .map(|m| m.id)
        .collect();
    plain.sort();
    via_progress.sort();
    assert_eq!(plain, via_progress);
}

#[test]
fn list_with_progress_on_an_empty_repo_calls_nothing_and_returns_empty() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let calls = Mutex::new(0usize);
    let metas = repo
        .list_with_progress(|_, _| *calls.lock().unwrap() += 1)
        .unwrap();
    assert!(metas.is_empty());
    assert_eq!(*calls.lock().unwrap(), 0);
}

#[test]
fn list_all_with_progress_spans_every_project_in_one_pool() {
    let root = tmpdir();
    let repo_a = SessionRepo::open(root.path().join("proj-a")).unwrap();
    for i in 0..5 {
        repo_a
            .create(SessionMeta::new(format!("/a{i}"), "m"))
            .unwrap();
    }
    let repo_b = SessionRepo::open(root.path().join("proj-b")).unwrap();
    for i in 0..5 {
        repo_b
            .create(SessionMeta::new(format!("/b{i}"), "m"))
            .unwrap();
    }

    let calls: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
    let metas = SessionRepo::list_all_with_progress(root.path(), |scanned, total| {
        calls.lock().unwrap().push((scanned, total))
    })
    .unwrap();
    assert_eq!(metas.len(), 10, "must merge both projects' sessions");

    let calls = calls.into_inner().unwrap();
    assert_eq!(
        calls.len(),
        10,
        "progress spans both projects, not per-project"
    );
    assert!(calls.iter().all(|&(_, total)| total == 10));
}

#[test]
fn resume_latest_or_create_reopens_the_session_matching_cwd() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut other = repo.create(SessionMeta::new("/other", "m")).unwrap();
    let mut sa = Session::new();
    sa.user("from another project");
    other.append_new(&sa.messages).unwrap();

    let mut mine = repo.create(SessionMeta::new("/mine", "m")).unwrap();
    let mut sb = Session::new();
    sb.user("from my project");
    mine.append_new(&sb.messages).unwrap();

    let (store, session) = repo.resume_latest_or_create("/mine", "m").unwrap();
    assert_eq!(store.meta().id, mine.meta().id);
    assert_eq!(session.messages.len(), 1);
}

#[test]
fn resume_latest_or_create_passes_over_a_newer_header_only_session() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut real = repo.create(SessionMeta::new("/mine", "m")).unwrap();
    let mut s = Session::new();
    s.user("the real conversation");
    real.append_new(&s.messages).unwrap();
    let old = std::time::SystemTime::now() - std::time::Duration::from_secs(60);
    fs::File::options()
        .write(true)
        .open(real.path())
        .unwrap()
        .set_modified(old)
        .unwrap();
    // A later invocation created its session, then failed before writing anything.
    let empty = repo.create(SessionMeta::new("/mine", "m")).unwrap();
    assert_eq!(
        repo.list().unwrap()[0].id,
        empty.meta().id,
        "the empty one is newer"
    );

    let (store, session) = repo.resume_latest_or_create("/mine", "m").unwrap();
    assert_eq!(store.meta().id, real.meta().id);
    assert_eq!(session.messages.len(), 1);

    // With only empty sessions, the newest one is reused rather than minting another.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let empty = repo.create(SessionMeta::new("/mine", "m")).unwrap();
    let (store, _) = repo.resume_latest_or_create("/mine", "m").unwrap();
    assert_eq!(store.meta().id, empty.meta().id);
    assert_eq!(repo.list().unwrap().len(), 1);
}

#[test]
fn the_create_free_halves_open_what_exists_and_create_nothing() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    assert!(repo.resume_latest("/mine").unwrap().is_none());
    assert!(repo.open_id_if_exists("named-one").unwrap().is_none());
    assert!(repo.list().unwrap().is_empty(), "neither created a session");

    let named = repo
        .create(SessionMeta::with_id("named-one", "/mine", "m"))
        .unwrap();
    let (store, _) = repo.open_id_if_exists("named-one").unwrap().unwrap();
    assert_eq!(store.meta().id, named.meta().id);
    assert!(
        repo.open_id_if_exists("named").unwrap().is_none(),
        "an id is matched exactly, never by prefix"
    );
    let (store, _) = repo.resume_latest("/mine").unwrap().unwrap();
    assert_eq!(store.meta().id, "named-one");
}

#[test]
fn resume_latest_or_create_makes_a_fresh_session_when_no_cwd_matches() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let (store, session) = repo.resume_latest_or_create("/brand/new", "m").unwrap();
    assert_eq!(store.meta().cwd, "/brand/new");
    assert!(session.messages.is_empty());
}

#[test]
fn open_or_create_id_mints_a_session_under_exactly_the_given_id() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let (store, session) = repo
        .open_or_create_id("my-chosen-id", "/brand/new", "m")
        .unwrap();
    assert_eq!(store.meta().id, "my-chosen-id");
    assert!(session.messages.is_empty());
}

#[test]
fn open_or_create_id_wins_over_a_session_matching_the_cwd() {
    // The regression that started this: an explicit id used to be silently discarded whenever *any*
    // session already existed for the same cwd, so the caller got a stranger's conversation.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut mine = repo.create(SessionMeta::new("/mine", "m")).unwrap();
    let mut s = Session::new();
    s.user("already here");
    mine.append_new(&s.messages).unwrap();

    let (store, session) = repo.open_or_create_id("my-own-id", "/mine", "m").unwrap();
    assert_eq!(
        store.meta().id,
        "my-own-id",
        "a caller-supplied id must never resolve to a different session"
    );
    assert_ne!(store.meta().id, mine.meta().id);
    assert!(
        session.messages.is_empty(),
        "the named session is brand new, so it must not inherit the cwd match's transcript"
    );
}

#[test]
fn distinct_ids_in_one_cwd_stay_distinct_sessions() {
    // Multi-tenancy in one line: N ids in a shared directory must be N sessions, not one. This is
    // the property `serve --session-id` and the daemon's `?session_id=` both route on.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    for id in ["tenant-a", "tenant-b", "tenant-c"] {
        let (mut store, mut session) = repo.open_or_create_id(id, "/shared", "m").unwrap();
        session.user(id);
        store.append_new(&session.messages).unwrap();
    }
    let ids: HashSet<String> = repo.list().unwrap().into_iter().map(|m| m.id).collect();
    assert_eq!(ids.len(), 3, "each id must own its own session file");

    // ...and each reopens to its own transcript, not a neighbour's.
    for id in ["tenant-a", "tenant-b", "tenant-c"] {
        let (_, session) = repo.open_or_create_id(id, "/shared", "m").unwrap();
        assert_eq!(session.messages.len(), 1);
        assert!(format!("{:?}", session.messages[0]).contains(id));
    }
}

#[test]
fn open_or_create_id_is_idempotent() {
    // Re-running with the same id lands back on the same conversation rather than stacking up a new
    // session per invocation — the property a supervised `serve --session-id` restart depends on.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let (mut store, mut session) = repo.open_or_create_id("pinned", "/w", "m").unwrap();
    session.user("first");
    store.append_new(&session.messages).unwrap();

    let (reopened, session) = repo.open_or_create_id("pinned", "/w", "m").unwrap();
    assert_eq!(reopened.meta().id, "pinned");
    assert_eq!(session.messages.len(), 1);
    assert_eq!(repo.list().unwrap().len(), 1, "no duplicate was created");
}

#[test]
fn open_or_create_id_does_not_confuse_an_id_with_an_underscored_suffix_of_another() {
    // Ids may legally contain `_`, and sessions are stored as `<created_at>_<id>.jsonl`. Matching on
    // an `_<id>.jsonl` *suffix* rather than the parsed component would resolve `b` onto `a_b`.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let (mut store, mut session) = repo.open_or_create_id("a_b", "/w", "m").unwrap();
    session.user("belongs to a_b");
    store.append_new(&session.messages).unwrap();

    let (short, session) = repo.open_or_create_id("b", "/w", "m").unwrap();
    assert_eq!(short.meta().id, "b");
    assert!(
        session.messages.is_empty(),
        "`b` must not resolve onto the session named `a_b`"
    );
    assert_eq!(repo.list().unwrap().len(), 2);

    // …and `a_b` still resolves to itself, by both lookups.
    let (reopened, session) = repo.open_or_create_id("a_b", "/w", "m").unwrap();
    assert_eq!(reopened.meta().id, "a_b");
    assert_eq!(session.messages.len(), 1);
    assert_eq!(repo.open_id("a_b").unwrap().0.meta().id, "a_b");
}

#[test]
fn open_or_create_id_matches_exactly_never_by_prefix() {
    // `find_path`'s unique-prefix convenience is for a human typing a shortened id. Applying it to
    // addressing would resolve `abc` onto `abcdef` *and* suppress creating `abc` — two wrongs that
    // would silently collapse a set of prefix-sharing ids onto whichever landed first.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let (mut store, mut session) = repo.open_or_create_id("abcdef", "/w", "m").unwrap();
    session.user("the long one");
    store.append_new(&session.messages).unwrap();

    let (short, session) = repo.open_or_create_id("abc", "/w", "m").unwrap();
    assert_eq!(short.meta().id, "abc");
    assert!(session.messages.is_empty());
    assert_eq!(repo.list().unwrap().len(), 2);
}

#[test]
fn canonical_cwd_resolves_a_symlink_to_its_real_target() {
    let dir = tmpdir();
    let real = dir.path().join("real");
    fs::create_dir(&real).unwrap();
    let link = dir.path().join("link");
    std::os::unix::fs::symlink(&real, &link).unwrap();

    assert_eq!(canonical_cwd(&link), canonical_cwd(&real));
}

#[test]
fn canonical_cwd_falls_back_to_the_given_path_when_it_does_not_exist() {
    let missing = Path::new("/definitely/does/not/exist/xyz-canonical-cwd-test");
    assert_eq!(canonical_cwd(missing), missing);
}

#[test]
fn resume_latest_or_create_matches_a_session_recorded_under_a_symlinked_cwd_once_canonicalized() {
    // The regression this guards: a project reached through a symlink one time and its real path
    // another must resolve to the same session, not silently fork into two.
    let dir = tmpdir();
    let real = dir.path().join("project");
    fs::create_dir(&real).unwrap();
    let link = dir.path().join("project-link");
    std::os::unix::fs::symlink(&real, &link).unwrap();

    let repo_dir = tmpdir();
    let repo = SessionRepo::open(repo_dir.path()).unwrap();
    let real_cwd = canonical_cwd(&real).to_string_lossy().into_owned();
    let mut store = repo.create(SessionMeta::new(&real_cwd, "m")).unwrap();
    let mut s = Session::new();
    s.user("hello from the real path");
    store.append_new(&s.messages).unwrap();

    let link_cwd = canonical_cwd(&link).to_string_lossy().into_owned();
    let (reopened, session) = repo.resume_latest_or_create(&link_cwd, "m").unwrap();
    assert_eq!(
        reopened.meta().id,
        store.meta().id,
        "a symlinked cwd must canonicalize to the same session as its real path"
    );
    assert_eq!(session.messages.len(), 1);
}

#[test]
fn list_all_merges_every_project_subdirectory() {
    // The layout `serve`'s default session directory follows: one subdirectory per project under
    // a shared root, each an independent `SessionRepo`.
    let root = tmpdir();
    let repo_a = SessionRepo::open(root.path().join("-home-jared-project-a")).unwrap();
    repo_a
        .create(SessionMeta::new("/home/jared/project-a", "m"))
        .unwrap();
    let repo_b = SessionRepo::open(root.path().join("-home-jared-project-b")).unwrap();
    repo_b
        .create(SessionMeta::new("/home/jared/project-b", "m"))
        .unwrap();

    let all = SessionRepo::list_all(root.path()).unwrap();
    assert_eq!(all.len(), 2);
    let cwds: std::collections::HashSet<&str> = all.iter().map(|m| m.cwd.as_str()).collect();
    assert!(cwds.contains("/home/jared/project-a"));
    assert!(cwds.contains("/home/jared/project-b"));
}

#[test]
fn list_all_is_newest_first_across_projects() {
    let root = tmpdir();
    let repo_a = SessionRepo::open(root.path().join("proj-a")).unwrap();
    let older = repo_a.create(SessionMeta::new("/a", "m")).unwrap();
    let repo_b = SessionRepo::open(root.path().join("proj-b")).unwrap();
    let newer = repo_b.create(SessionMeta::new("/b", "m")).unwrap();

    // Stamp mtimes explicitly (the sort key, `updated_at`, falls back to mtime for a session with
    // no messages) rather than relying on real wall-clock ordering between two back-to-back
    // creates, which can tie at second granularity.
    let earlier = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000_000);
    let later = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000_100);
    fs::File::options()
        .write(true)
        .open(repo_a.path_for(older.meta()))
        .unwrap()
        .set_modified(earlier)
        .unwrap();
    fs::File::options()
        .write(true)
        .open(repo_b.path_for(newer.meta()))
        .unwrap()
        .set_modified(later)
        .unwrap();

    let all = SessionRepo::list_all(root.path()).unwrap();
    assert_eq!(all.len(), 2);
    assert_eq!(
        all[0].cwd, "/b",
        "the more recently active session must sort first"
    );
    assert_eq!(all[1].cwd, "/a");
}

#[test]
fn list_sorts_by_last_activity_not_creation_time() {
    // `older` is created first but is the one touched most recently — a listing ordered by last
    // activity (matching pi's own session list, sorted by `modified`) must surface it first
    // despite `newer_by_creation` having a later `created_at`.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let older = repo.create(SessionMeta::new("/older", "m")).unwrap();
    let older_id = older.meta().id.clone();
    let newer_by_creation = repo.create(SessionMeta::new("/newer", "m")).unwrap();
    let newer_id = newer_by_creation.meta().id.clone();

    let earlier = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000_000);
    let later = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000_100);
    fs::File::options()
        .write(true)
        .open(repo.path_for(newer_by_creation.meta()))
        .unwrap()
        .set_modified(earlier)
        .unwrap();
    fs::File::options()
        .write(true)
        .open(repo.path_for(older.meta()))
        .unwrap()
        .set_modified(later)
        .unwrap();

    let metas = repo.list().unwrap();
    assert_eq!(
        metas[0].id, older_id,
        "last-activity ordering must beat creation order"
    );
    assert_eq!(metas[1].id, newer_id);
}

#[test]
fn list_all_ignores_non_repo_entries_and_a_missing_root() {
    let root = tmpdir();
    // A stray file at the root (not a project subdirectory) must not break the scan.
    std::fs::write(root.path().join("not-a-project.txt"), "x").unwrap();
    // An empty subdirectory (a project with no sessions yet) contributes nothing, not an error.
    std::fs::create_dir(root.path().join("empty-project")).unwrap();
    let repo = SessionRepo::open(root.path().join("real-project")).unwrap();
    repo.create(SessionMeta::new("/real", "m")).unwrap();

    let all = SessionRepo::list_all(root.path()).unwrap();
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].cwd, "/real");

    // A root that doesn't exist at all (nothing has ever been persisted) is an empty list, not an
    // I/O error.
    let missing = root.path().join("does-not-exist");
    assert!(SessionRepo::list_all(&missing).unwrap().is_empty());
}
