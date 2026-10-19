use super::*;

#[test]
fn switch_active_then_append_forks_a_branch() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("a");
    session.user("b");
    session.user("c");
    store.append_new(&session.messages).unwrap();
    let ids = store.active_ids().to_vec();
    assert_eq!(ids.len(), 3);

    // Navigate back to the first message and fork off it.
    let branch_root = store.switch_active(&ids[0]).unwrap();
    assert_eq!(branch_root.len(), 1);
    let mut branch_session = Session::new();
    branch_session.messages = Arc::new(branch_root);
    branch_session.user("d");
    store.append_new(&branch_session.messages).unwrap();

    // The active path is now [a, d] — b/c are off-branch, not part of it.
    assert_eq!(store.active_ids().len(), 2);

    // Reopening resolves the tip via the persisted `Leaf` marker to the same branch, not the
    // physically-earlier b/c continuation.
    let (reopened, restored) = repo.open_id(&store.meta().id.clone()).unwrap();
    assert_eq!(restored.messages.len(), 2);
    let dump = serde_json::to_string(restored.messages.as_ref()).unwrap();
    assert!(dump.contains("\"a\"") && dump.contains("\"d\""));
    assert!(!dump.contains("\"b\"") && !dump.contains("\"c\""));
    assert_eq!(reopened.active_ids().len(), 2);
}

#[test]
fn fork_at_entry_works_on_an_off_active_path_branch() {
    // The whole point of `fork_at_entry` over `fork`'s `upto` count: it can target a message that
    // isn't on the *current* active path at all, without first `switch_active`-ing to it.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("a");
    session.user("b");
    session.user("c");
    store.append_new(&session.messages).unwrap();
    let ids = store.active_ids().to_vec();

    // Navigate back to the first message and fork off it with "d" — b/c fall off the active path.
    let branch_root = store.switch_active(&ids[0]).unwrap();
    let mut branch_session = Session::new();
    branch_session.messages = Arc::new(branch_root);
    branch_session.user("d");
    store.append_new(&branch_session.messages).unwrap();
    assert_eq!(store.active_ids().len(), 2); // [a, d]

    let session_id = store.meta().id.clone();

    // Fork "at" b (off-branch) — includes b itself.
    let (forked, fsession) = repo.fork_at_entry(&session_id, &ids[1], false).unwrap();
    assert_eq!(fsession.messages.len(), 2);
    let dump = serde_json::to_string(fsession.messages.as_ref()).unwrap();
    assert!(dump.contains("\"a\"") && dump.contains("\"b\""));
    assert!(!dump.contains("\"c\"") && !dump.contains("\"d\""));
    assert_eq!(forked.meta().parent.as_deref(), Some(session_id.as_str()));

    // Fork "before" c (off-branch) — excludes c itself, same result as forking "at" b.
    let (_, fsession_before) = repo.fork_at_entry(&session_id, &ids[2], true).unwrap();
    assert_eq!(fsession_before.messages.len(), 2);
    let dump = serde_json::to_string(fsession_before.messages.as_ref()).unwrap();
    assert!(dump.contains("\"a\"") && dump.contains("\"b\""));
    assert!(!dump.contains("\"c\""));

    // Fork "at" c (off-branch) — includes c itself, the whole original a->b->c line.
    let (_, fsession_at_c) = repo.fork_at_entry(&session_id, &ids[2], false).unwrap();
    assert_eq!(fsession_at_c.messages.len(), 3);

    // An unknown entry id is rejected, matching `switch_active`'s own NotFound convention.
    match repo.fork_at_entry(&session_id, "does-not-exist", false) {
        Ok(_) => panic!("expected NotFound for an unknown entry id"),
        Err(e) => assert_eq!(e.kind(), std::io::ErrorKind::NotFound),
    }
}

#[test]
fn fork_at_entry_before_a_non_user_message_is_rejected() {
    // Track L27: `before` (now the default — see `serve/commands/`'s `fork`/`preview_fork`) means "fork
    // right before this entry", which pi's own `getEntriesToFork` (`repo-utils.ts`) only allows
    // anchored to a user turn — forking "before" an assistant reply is ambiguous (which point,
    // exactly, is "before" a reply that has no message of its own between it and the prior turn?).
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("a");
    session.push(Message::assistant(vec![ContentBlock::text("a-reply")]));
    store.append_new(&session.messages).unwrap();
    let ids = store.active_ids().to_vec(); // [user "a", assistant "a-reply"]

    let session_id = store.meta().id.clone();
    match repo.fork_at_entry(&session_id, &ids[1], true) {
        Ok(_) => {
            panic!("expected invalid_fork_target for a `before` fork at an assistant reply")
        }
        Err(e) => {
            assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput);
            assert!(
                e.to_string().contains("invalid_fork_target"),
                "error must be identifiable as pi's own `invalid_fork_target` case: {e}"
            );
        }
    }

    // "at" (before:false) the very same entry is perfectly valid — the restriction is specific to
    // `before`, not to targeting a non-user entry at all.
    let (_, fsession) = repo.fork_at_entry(&session_id, &ids[1], false).unwrap();
    assert_eq!(fsession.messages.len(), 2);
}

#[test]
fn fork_at_entry_rejects_an_anchored_side_channel_entry_as_a_target_and_it_is_unreachable_anyway() {
    // Investigated (Round 2 pi-parity remediation, Low severity, left unfixed — see
    // `fork_at_entry_prefix`'s own doc comment for the full assessment). Pi's uniform tree model
    // lets a caller fork/switch at *any* entry id, including a `Label`/`ModelChange`/
    // `ThinkingLevelChange`/`TitleChange` — beyond's narrower one requires `entry_id` to be a real
    // `nodes` entry (`Message`/`BranchSummary`/`Custom`), so one of these 404s. Confirmed here two
    // ways: (a) `fork_at_entry` really does reject such an id, and (b) that id was never something
    // a client could have discovered in the first place — `tree()`, the only surface that lists
    // forkable ids, never lists one of these at all — so no real caller can actually hit case (a).
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("a");
    store.append_new(&session.messages).unwrap();
    let msg_id = store.active_ids()[0].clone();
    store.set_label(&msg_id, Some("checkpoint")).unwrap();

    // The label entry's own id is never returned by any public method — recovered here only by
    // parsing the raw file, which no real client does either (see `tree()`'s own doc comment: it
    // reports the label as the *target* node's `label` field, never as its own addressable entry).
    let raw = fs::read_to_string(store.path()).unwrap();
    let label_entry_id = raw
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<Value>(l).unwrap())
        .find(|v| v["type"] == json!("label"))
        .and_then(|v| v["id"].as_str().map(str::to_string))
        .expect("set_label must have appended an Entry::Label line with its own id");

    // (b) unreachable via the client-facing surface: no `TreeNode` ever carries this id.
    assert!(
        store.tree().iter().all(|n| n.id != label_entry_id),
        "a label's own entry id must never be independently listed by `tree()`"
    );

    // (a) and, even if a caller somehow obtained it anyway (as this test does, by reading the raw
    // file), it's still rejected as a fork target.
    let session_id = store.meta().id.clone();
    match repo.fork_at_entry(&session_id, &label_entry_id, false) {
        Ok(_) => panic!("expected NotFound when forking at a label entry's own id"),
        Err(e) => assert_eq!(e.kind(), std::io::ErrorKind::NotFound),
    }
}
