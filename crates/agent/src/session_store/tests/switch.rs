use super::*;

#[test]
fn switch_active_rejects_unknown_id() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let err = store.switch_active("does-not-exist").unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
}

#[test]
fn switch_active_to_the_already_active_leaf_is_a_no_op() {
    // pi-parity fix (B-L2): `agent-session-tree-navigation.test.ts`'s "should handle navigation to
    // same position (no-op)" — navigating to wherever the session already is must not append a
    // redundant `Leaf` entry (or any entry at all): the entry count on disk must not grow, and the
    // in-memory active path must be unchanged.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("a");
    session.user("b");
    store.append_new(&session.messages).unwrap();
    let ids = store.active_ids().to_vec();
    let tip = ids.last().cloned().unwrap();

    let before = std::fs::read_to_string(store.path()).unwrap();
    let messages = store.switch_active(&tip).unwrap();

    assert_eq!(
        messages.len(),
        2,
        "the no-op switch must still return the current branch's materialized messages"
    );
    assert_eq!(
        store.active_ids(),
        ids.as_slice(),
        "the active path must be unchanged by a no-op switch"
    );
    let after = std::fs::read_to_string(store.path()).unwrap();
    assert_eq!(
        before, after,
        "navigating to the already-active leaf must not write anything to disk"
    );

    // Reopening must agree: no phantom `Leaf` entry was ever persisted.
    let (reopened, restored) = repo.open_id(&store.meta().id.clone()).unwrap();
    assert_eq!(restored.messages.len(), 2);
    assert_eq!(reopened.active_ids(), ids.as_slice());
}

#[test]
fn rewrite_preserves_off_branch_messages() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("alpha");
    session.user("beta");
    store.append_new(&session.messages).unwrap();
    let ids = store.active_ids().to_vec();

    // Navigate back to `alpha` and fork a second branch off it — `beta` becomes off-branch.
    let root = store.switch_active(&ids[0]).unwrap();
    let mut forked = Session::new();
    forked.messages = Arc::new(root);
    forked.user("gamma");
    store.append_new(&forked.messages).unwrap();

    // Compact the *active* path (alpha, gamma) down to a single summary — this must not touch the
    // off-branch `beta`.
    store.rewrite(&[Message::user("summary")]).unwrap();
    assert_eq!(store.active_ids().len(), 1);

    // `beta`'s content is still physically on disk, even though it's unreachable from the new tip.
    let raw = fs::read_to_string(store.path()).unwrap();
    assert!(
        raw.contains("\"beta\""),
        "off-branch message was deleted:\n{raw}"
    );
    assert!(
        !raw.contains("\"alpha\""),
        "active-path message should have been compacted away"
    );

    // The active session itself only sees the compacted summary.
    let (_reopened, restored) = repo.open_id(&store.meta().id.clone()).unwrap();
    assert_eq!(restored.messages.len(), 1);
    let dump = serde_json::to_string(restored.messages.as_ref()).unwrap();
    assert!(dump.contains("summary"));
}

#[test]
fn open_resolves_tip_via_last_leaf_entry() {
    // Hand-construct a branched file: root -> [second (m2), branched (m3)], with a `Leaf` pointing
    // the active tip at `m3` — proving `open` prefers the leaf's target over "last line in file".
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let meta = SessionMeta::new("/w", "m");
    let id = meta.id.clone();
    let path = repo.path_for(&meta);

    let mut lines = vec![serde_json::to_string(&Entry::Session(meta)).unwrap()];
    lines.push(
        serde_json::to_string(&Entry::Message {
            id: Some("m1".into()),
            parent_id: None,
            timestamp: 0,
            message: Message::user("first"),
        })
        .unwrap(),
    );
    lines.push(
        serde_json::to_string(&Entry::Message {
            id: Some("m2".into()),
            parent_id: Some("m1".into()),
            timestamp: 0,
            message: Message::user("second"),
        })
        .unwrap(),
    );
    lines.push(
        serde_json::to_string(&Entry::Message {
            id: Some("m3".into()),
            parent_id: Some("m1".into()),
            timestamp: 0,
            message: Message::user("branched"),
        })
        .unwrap(),
    );
    lines.push(
        serde_json::to_string(&Entry::Leaf {
            id: "l1".into(),
            parent_id: None,
            target_id: Some("m3".into()),
        })
        .unwrap(),
    );
    fs::write(&path, lines.join("\n") + "\n").unwrap();

    let (store, session) = repo.open_id(&id).unwrap();
    assert_eq!(session.messages.len(), 2);
    let dump = serde_json::to_string(session.messages.as_ref()).unwrap();
    assert!(dump.contains("first") && dump.contains("branched"));
    assert!(!dump.contains("second"));
    assert_eq!(store.active_ids(), &["m1".to_string(), "m3".to_string()]);
}

#[test]
fn tree_and_list_branches_carry_each_node_s_timestamp_and_sort_chronologically() {
    // Pi-parity fix: `TreeNode`/`BranchInfo` now carry a `timestamp` field and order by it (not by
    // id) — hand-construct a file where the alphabetically-earlier leaf ("m2") was actually created
    // *after* the alphabetically-later one ("m3"), so an id-based sort and a timestamp-based sort
    // disagree — proving the fix actually changed the sort key, not just added an unused field.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let meta = SessionMeta::new("/w", "m");
    let id = meta.id.clone();
    let path = repo.path_for(&meta);

    let lines = [
        serde_json::to_string(&Entry::Session(meta)).unwrap(),
        serde_json::to_string(&Entry::Message {
            id: Some("m1".into()),
            parent_id: None,
            timestamp: 0,
            message: Message::user("root"),
        })
        .unwrap(),
        serde_json::to_string(&Entry::Message {
            id: Some("m3".into()),
            parent_id: Some("m1".into()),
            timestamp: 100,
            message: Message::user("earlier-branch"),
        })
        .unwrap(),
        serde_json::to_string(&Entry::Message {
            id: Some("m2".into()),
            parent_id: Some("m1".into()),
            timestamp: 200,
            message: Message::user("later-branch"),
        })
        .unwrap(),
        serde_json::to_string(&Entry::Leaf {
            id: "l1".into(),
            parent_id: None,
            target_id: Some("m2".into()),
        })
        .unwrap(),
    ];
    fs::write(&path, lines.join("\n") + "\n").unwrap();

    let (store, _session) = repo.open_id(&id).unwrap();

    let tree = store.tree();
    let leaves: Vec<&TreeNode> = tree.iter().filter(|n| n.id != "m1").collect();
    assert_eq!(
        leaves.iter().map(|n| n.id.as_str()).collect::<Vec<_>>(),
        vec!["m3", "m2"],
        "must sort by timestamp (m3=100 before m2=200), not by id: {leaves:?}"
    );
    assert_eq!(tree.iter().find(|n| n.id == "m1").unwrap().timestamp, 0);
    assert_eq!(leaves[0].timestamp, 100);
    assert_eq!(leaves[1].timestamp, 200);

    let branches = store.list_branches();
    assert_eq!(
        branches
            .iter()
            .map(|b| b.leaf_id.as_str())
            .collect::<Vec<_>>(),
        vec!["m3", "m2"],
        "must sort by timestamp (m3=100 before m2=200), not by leaf id: {branches:?}"
    );
    assert_eq!(branches[0].timestamp, 100);
    assert_eq!(branches[1].timestamp, 200);
    assert!(!branches[0].is_active);
    assert!(branches[1].is_active, "the leaf entry points the tip at m2");
}

#[test]
fn switch_active_with_summary_persists_counters_and_survives_reopen() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("alpha");
    session.user("beta");
    store.append_new(&session.messages).unwrap();
    let ids = store.active_ids().to_vec();

    // Navigate away from `beta` (the abandoned tip) back to `alpha`, recording a summary of what
    // was abandoned in the same step.
    let abandoned_tip = ids[1].clone();
    let messages = store
        .switch_active_with_summary(
            &ids[0],
            "recap of the abandoned branch",
            &abandoned_tip,
            BranchSummaryDetails {
                read_files: vec!["a.rs".into()],
                modified_files: vec![],
                summarized_messages: 1,
            },
        )
        .unwrap();

    assert_eq!(store.meta().branch_summaries, 1);
    assert_eq!(store.meta().summarized_branch_messages, 1);
    // The new active path is alpha *plus* the summary message — not just alpha, and not the
    // untouched two-message branch either.
    assert_eq!(messages.len(), 2);
    assert_eq!(store.active_ids().len(), 2);

    // Everything survives reopen: the header counters, `beta`'s content (still physically on disk
    // even though it's off the active path), and the summary as the new tip.
    let (reopened, restored) = repo.open_id(&store.meta().id.clone()).unwrap();
    assert_eq!(reopened.meta().branch_summaries, 1);
    assert_eq!(reopened.meta().summarized_branch_messages, 1);
    assert_eq!(restored.messages.len(), 2);
    let raw = fs::read_to_string(reopened.path()).unwrap();
    assert!(
        raw.contains("\"beta\""),
        "off-branch message was lost:\n{raw}"
    );
    assert!(raw.contains("recap of the abandoned branch"));
    assert!(raw.contains(&abandoned_tip));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            fs::metadata(reopened.path()).unwrap().permissions().mode() & 0o777,
            0o600,
            "switch_active_with_summary's temp-file-then-rename must not loosen permissions"
        );
    }
}

#[test]
fn branch_summary_carries_a_real_timestamp_not_the_zero_placeholder() {
    // Track L45 (pi-parity fix): `Entry::BranchSummary` used to have no `timestamp` field at all,
    // unlike `Entry::Message`/`Entry::Custom` — a materialized branch-summary node always reported
    // `timestamp: 0` in `tree()`'s output regardless of when it was actually created, breaking a
    // client's ability to sort/reconstruct branch order chronologically for that one entry kind.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("alpha");
    session.user("beta");
    store.append_new(&session.messages).unwrap();
    let ids = store.active_ids().to_vec();

    let before = now_secs();
    store
        .switch_active_with_summary(
            &ids[0],
            "recap of the abandoned branch",
            &ids[1],
            BranchSummaryDetails::default(),
        )
        .unwrap();
    let summary_id = store.active_ids().last().unwrap().clone();

    // The live, in-memory instance already reports a real timestamp, not the old `0` placeholder.
    let node = store
        .tree()
        .into_iter()
        .find(|n| n.id == summary_id)
        .unwrap();
    assert!(
        node.timestamp >= before,
        "expected a real timestamp (>= {before}), got {}",
        node.timestamp
    );

    // Survives reopen from disk, read back from the persisted `Entry::BranchSummary` line itself.
    let (reopened, _) = repo.open_id(&store.meta().id.clone()).unwrap();
    let reopened_node = reopened
        .tree()
        .into_iter()
        .find(|n| n.id == summary_id)
        .unwrap();
    assert_eq!(
        reopened_node.timestamp, node.timestamp,
        "a reopened session must recover the same real timestamp, not fall back to 0"
    );
}

#[test]
fn switch_active_with_summary_surfaces_summary_as_first_message_of_new_branch() {
    // Unlike a plain `switch_active`, this must actually redirect the tip: the summary becomes a
    // real message, part of the live transcript the model sees on the next turn — not an inert
    // provenance record sitting off to the side.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("only message");
    store.append_new(&session.messages).unwrap();
    let root_id = store.active_ids()[0].clone();

    let messages = store
        .switch_active_with_summary(
            &root_id,
            "some recap",
            "abandoned-branch-tip",
            BranchSummaryDetails::default(),
        )
        .unwrap();
    assert_eq!(messages.len(), 2);
    let summary_text = match &messages[1].content[0] {
        ContentBlock::Text { text, .. } => text,
        other => panic!("expected a text block, got {other:?}"),
    };
    assert!(summary_text.contains("some recap"));

    let (reopened, restored) = repo.open_id(&store.meta().id.clone()).unwrap();
    assert_eq!(restored.messages.len(), 2);
    assert_ne!(
        reopened.active_ids(),
        &[root_id],
        "the tip must have moved to include the summary message, not stayed at the root"
    );
}

#[test]
fn switch_active_with_summary_attaches_to_a_root_target_and_becomes_the_new_leaf() {
    // B-M6 pi-parity test gap (fixed): the exact tree shape a branch summary produces was never
    // asserted, only that *some* summary got applied. This is pi's "summary attached to root
    // node" scenario (`agent-session-tree-navigation.test.ts:78-106`) — this codebase's uniform
    // contract (no editor-rewind distinction between user/assistant targets, unlike pi) is: the
    // summary always attaches as a *child of the target itself*, regardless of shape, and always
    // becomes the new leaf.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("first message"); // u1 — the root, no parent
    session.push(Message::assistant(vec![ContentBlock::text("a1")]));
    session.user("second message");
    session.push(Message::assistant(vec![ContentBlock::text("a2")]));
    store.append_new(&session.messages).unwrap();
    let ids = store.active_ids().to_vec();
    let u1 = ids[0].clone();
    let a1 = ids[1].clone();

    store
        .switch_active_with_summary(&u1, "root recap", &ids[3], BranchSummaryDetails::default())
        .unwrap();

    // u1 itself is confirmed root (no parent) — the scenario this test is actually exercising.
    let tree = store.tree();
    let u1_node = tree.iter().find(|n| n.id == u1).unwrap();
    assert!(u1_node.parent_id.is_none(), "u1 must be the tree's root");

    let summary_id = store.active_ids().last().unwrap().clone();
    let summary_node = tree.iter().find(|n| n.id == summary_id).unwrap();
    assert_eq!(
        summary_node.parent_id.as_deref(),
        Some(u1.as_str()),
        "the summary must attach as a child of the target itself"
    );

    // u1 now has two children: its original next message (a1) and the new summary.
    let children: Vec<&str> = tree
        .iter()
        .filter(|n| n.parent_id.as_deref() == Some(u1.as_str()))
        .map(|n| n.id.as_str())
        .collect();
    assert_eq!(
        children.len(),
        2,
        "expected two children of u1: {children:?}"
    );
    assert!(children.contains(&a1.as_str()));
    assert!(children.contains(&summary_id.as_str()));

    assert_eq!(
        store.active_ids().last(),
        Some(&summary_id),
        "the summary must become the new leaf"
    );
}

#[test]
fn parent_of_reports_root_a_real_parent_and_unknown_ids_distinctly() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("first");
    session.user("second");
    store.append_new(&session.messages).unwrap();
    let ids = store.active_ids().to_vec();

    assert_eq!(
        store.parent_of(&ids[0]),
        Some(None),
        "the first message's parent is the root — no parent, but a known node"
    );
    assert_eq!(
        store.parent_of(&ids[1]),
        Some(Some(ids[0].clone())),
        "the second message's parent is the first"
    );
    assert_eq!(
        store.parent_of("does-not-exist"),
        None,
        "an unknown id is distinguishable from a real root"
    );
}

#[test]
fn switch_active_to_root_clears_the_active_path_and_persists_across_reopen() {
    // Pi-parity fix: no way existed to navigate back to before the very first message (redo it in
    // place) — pi's own `SessionManager::resetLeaf`.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("first");
    session.user("second");
    store.append_new(&session.messages).unwrap();

    let messages = store.switch_active_to_root().unwrap();
    assert!(messages.is_empty());
    assert!(
        store.active_ids().is_empty(),
        "the active path must be empty at the root"
    );

    // Appending fresh messages now chains off the root (`parent_id: None`), redoing the first
    // message in place rather than continuing the old branch.
    let mut redo = Session::new();
    redo.user("redone first message");
    store.append_new(&redo.messages).unwrap();
    let new_ids = store.active_ids().to_vec();
    assert_eq!(new_ids.len(), 1);
    assert_eq!(store.parent_of(&new_ids[0]), Some(None));

    // Survives a reopen: the `Leaf{target_id: None}` marker resolves back to root, not to
    // whatever message happens to be last in the file.
    let (reopened, session) = repo.open_id(&store.meta().id.clone()).unwrap();
    assert_eq!(session.messages.len(), 1);
    assert_eq!(reopened.active_ids(), &new_ids[..]);
}

#[test]
fn switch_active_to_root_is_a_no_op_when_already_there() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    assert!(store.active_ids().is_empty());

    store.switch_active_to_root().unwrap();
    assert!(store.active_ids().is_empty());

    // No `Leaf` entry appended — the file only ever gained the header line.
    let raw = fs::read_to_string(store.path()).unwrap();
    assert!(!raw.contains("\"leaf\""), "expected no Leaf entry: {raw}");
}

#[test]
fn abandoned_to_root_reports_the_whole_active_path() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    assert!(
        store.abandoned_to_root().is_empty(),
        "nothing to abandon at the root"
    );

    let mut session = Session::new();
    session.user("first");
    session.user("second");
    store.append_new(&session.messages).unwrap();

    let abandoned = store.abandoned_to_root();
    assert_eq!(abandoned.len(), 2, "the whole active path is abandoned");
    assert_eq!(abandoned[0].0, store.active_ids()[0]);
    assert_eq!(abandoned[1].0, store.active_ids()[1]);
}

#[test]
fn switch_active_to_root_with_summary_makes_the_summary_a_new_root_and_the_new_leaf() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("first");
    session.user("second");
    store.append_new(&session.messages).unwrap();
    let old_tip = store.active_ids().last().cloned().unwrap();

    let messages = store
        .switch_active_to_root_with_summary(
            "recap of the whole abandoned conversation",
            &old_tip,
            BranchSummaryDetails::default(),
        )
        .unwrap();
    assert_eq!(messages.len(), 1, "the summary is the sole active message");

    let summary_id = store.active_ids().last().unwrap().clone();
    assert_eq!(
        store.parent_of(&summary_id),
        Some(None),
        "the summary must become a genuine new root, not a child of the old tree"
    );
    assert_eq!(store.active_ids(), std::slice::from_ref(&summary_id));

    // Survives a reopen.
    let (reopened, session) = repo.open_id(&store.meta().id.clone()).unwrap();
    assert_eq!(session.messages.len(), 1);
    assert_eq!(reopened.active_ids(), &[summary_id]);
}

#[test]
fn switch_active_with_summary_attaches_to_a_nested_user_target_and_becomes_the_new_leaf() {
    // B-M6 pi-parity test gap (fixed): pi's "attach summary to correct parent when navigating to
    // nested user message" scenario (`agent-session-tree-navigation.test.ts:108-145`) — there, pi
    // attaches to the target's *parent* (an editor-rewind semantic this codebase has no
    // equivalent of). Here the contract is uniform: the summary attaches as a child of the
    // nested user target itself, alongside its existing child, and becomes the new leaf.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("message one");
    session.push(Message::assistant(vec![ContentBlock::text("a1")]));
    session.user("message two"); // u2 — nested (not root), the target
    session.push(Message::assistant(vec![ContentBlock::text("a2")]));
    session.user("message three");
    session.push(Message::assistant(vec![ContentBlock::text("a3")]));
    store.append_new(&session.messages).unwrap();
    let ids = store.active_ids().to_vec();
    let u2 = ids[2].clone();
    let a1 = ids[1].clone();
    let a2 = ids[3].clone();

    // u2's parent is confirmed a1 — the "nested, not root" shape this test exercises.
    let tree_before = store.tree();
    assert_eq!(
        tree_before
            .iter()
            .find(|n| n.id == u2)
            .unwrap()
            .parent_id
            .as_deref(),
        Some(a1.as_str())
    );

    store
        .switch_active_with_summary(
            &u2,
            "nested recap",
            &ids[5],
            BranchSummaryDetails::default(),
        )
        .unwrap();

    let tree = store.tree();
    let summary_id = store.active_ids().last().unwrap().clone();
    let summary_node = tree.iter().find(|n| n.id == summary_id).unwrap();
    assert_eq!(
        summary_node.parent_id.as_deref(),
        Some(u2.as_str()),
        "the summary must attach as a child of u2 itself, not u2's parent"
    );

    // u2 now has two children: its original next message (a2) and the new summary.
    let children: Vec<&str> = tree
        .iter()
        .filter(|n| n.parent_id.as_deref() == Some(u2.as_str()))
        .map(|n| n.id.as_str())
        .collect();
    assert_eq!(
        children.len(),
        2,
        "expected two children of u2: {children:?}"
    );
    assert!(children.contains(&a2.as_str()));
    assert!(children.contains(&summary_id.as_str()));

    assert_eq!(store.active_ids().last(), Some(&summary_id));
}

#[test]
fn switch_active_with_summary_attaches_to_an_assistant_target_and_becomes_the_new_leaf() {
    // B-M6 pi-parity test gap (fixed): pi's "attach summary to selected node when navigating to
    // assistant message" scenario (`agent-session-tree-navigation.test.ts:147-173`) — pi attaches
    // to the assistant target itself (no rewind semantic for a non-user target), matching this
    // codebase's uniform contract exactly.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("hello");
    session.push(Message::assistant(vec![ContentBlock::text("a1")])); // the target
    session.user("goodbye");
    session.push(Message::assistant(vec![ContentBlock::text("a2")]));
    store.append_new(&session.messages).unwrap();
    let ids = store.active_ids().to_vec();
    let a1 = ids[1].clone();
    let u2 = ids[2].clone();

    store
        .switch_active_with_summary(
            &a1,
            "assistant recap",
            &ids[3],
            BranchSummaryDetails::default(),
        )
        .unwrap();

    let tree = store.tree();
    let summary_id = store.active_ids().last().unwrap().clone();
    let summary_node = tree.iter().find(|n| n.id == summary_id).unwrap();
    assert_eq!(
        summary_node.parent_id.as_deref(),
        Some(a1.as_str()),
        "the summary must attach as a child of the selected assistant node"
    );

    // a1 now has two children: its original next message (u2) and the new summary.
    let children: Vec<&str> = tree
        .iter()
        .filter(|n| n.parent_id.as_deref() == Some(a1.as_str()))
        .map(|n| n.id.as_str())
        .collect();
    assert_eq!(
        children.len(),
        2,
        "expected two children of a1: {children:?}"
    );
    assert!(children.contains(&u2.as_str()));
    assert!(children.contains(&summary_id.as_str()));

    assert_eq!(
        store.active_ids().last(),
        Some(&summary_id),
        "the summary must become the new leaf"
    );
}

#[test]
fn branch_summary_details_within_folds_forward_a_nested_summary() {
    // A detour off a detour: root -> (branch summary S1, carrying file-tracking) -> one more
    // message -> abandon back to root. The abandoned range's *messages* only expose S1's prose
    // recap (extract_file_ops would find nothing new in it, since it's plain text, not a tool
    // call) — `branch_summary_details_within` must still recover S1's original read/modified files
    // from the index, not just whatever the plain message scan finds.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("root");
    store.append_new(&session.messages).unwrap();
    let root_id = store.active_ids()[0].clone();

    // Record S1 at the root, carrying file-tracking from whatever it summarized.
    store
        .switch_active_with_summary(
            &root_id,
            "recap of the first detour",
            "some-abandoned-tip",
            BranchSummaryDetails {
                read_files: vec!["src/a.rs".to_string()],
                modified_files: vec!["src/b.rs".to_string()],
                summarized_messages: 4,
            },
        )
        .unwrap();
    let s1_id = store.active_ids().last().unwrap().clone();

    // Continue past S1 with one more ordinary message, then abandon everything back to root.
    let mut continued = Session::new();
    continued.messages = Arc::new(
        store
            .switch_active(&s1_id)
            .unwrap_or_else(|_| panic!("s1 must be a valid target")),
    );
    continued.user("one more message after the detour");
    store.append_new(&continued.messages).unwrap();

    let abandoned = store.abandoned_by_switch(&root_id);
    assert_eq!(abandoned.len(), 2, "S1 plus the one message after it");
    let ids: Vec<String> = abandoned.iter().map(|(id, _)| id.clone()).collect();
    assert!(ids.contains(&s1_id));

    let folded = store.branch_summary_details_within(&ids);
    assert_eq!(folded.read_files, vec!["src/a.rs".to_string()]);
    assert_eq!(folded.modified_files, vec!["src/b.rs".to_string()]);

    // Survives a reopen — the index is rebuilt from the persisted `Entry::BranchSummary`, not just
    // held in this process's live memory.
    let (reopened, _) = repo.open_id(&store.meta().id.clone()).unwrap();
    let refolded = reopened.branch_summary_details_within(&ids);
    assert_eq!(refolded.read_files, vec!["src/a.rs".to_string()]);
    assert_eq!(refolded.modified_files, vec!["src/b.rs".to_string()]);
}

#[test]
fn switching_back_twice_does_not_duplicate_summary() {
    // A second, independent detour-and-return through the same point produces a second, sibling
    // summary entry (different abandoned content) — never overwriting or deduping against the
    // first one, and a *plain* switch_active to the same target afterward bypasses any summary
    // entirely (it goes straight back to the raw message, not through either recap).
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("root");
    store.append_new(&session.messages).unwrap();
    let root_id = store.active_ids()[0].clone();

    store
        .switch_active_with_summary(&root_id, "first recap", "branch-a", Default::default())
        .unwrap();
    store
        .switch_active_with_summary(&root_id, "second recap", "branch-b", Default::default())
        .unwrap();
    assert_eq!(store.meta().branch_summaries, 2);

    let raw = fs::read_to_string(store.path()).unwrap();
    assert!(raw.contains("first recap"));
    assert!(raw.contains("second recap"));

    // A later plain switch_active to the same root bypasses both summaries.
    let messages = store.switch_active(&root_id).unwrap();
    assert_eq!(messages.len(), 1);
}

#[test]
fn abandoned_by_switch_covers_ancestor_tip_and_sibling_cases() {
    // Build [a, b, c, d] then fork [a, b, e] off `b` — the tree looks like:
    //   a -> b -> c -> d   (the original continuation, still active before any switch below)
    //          \-> e       (a sibling branch, created by switching to b then appending)
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("a");
    session.user("b");
    session.user("c");
    session.user("d");
    store.append_new(&session.messages).unwrap();
    let ids = store.active_ids().to_vec(); // [a, b, c, d]

    // Case 1: target == the current tip (d) — a no-op switch, nothing abandoned.
    assert!(store.abandoned_by_switch(&ids[3]).is_empty());

    // Case 2: target is an ancestor still on the active path (b) — c and d are abandoned.
    let abandoned = store.abandoned_by_switch(&ids[1]);
    assert_eq!(abandoned.len(), 2);
    assert!(
        matches!(&abandoned[0].1.content[0], ContentBlock::Text{text, ..} if text.as_ref() == "c")
    );
    assert!(
        matches!(&abandoned[1].1.content[0], ContentBlock::Text{text, ..} if text.as_ref() == "d")
    );

    // Case 3: unknown target — nothing abandoned (the switch itself will fail).
    assert!(store.abandoned_by_switch("does-not-exist").is_empty());

    // Now actually switch to b and fork a sibling branch e off it.
    let root = store.switch_active(&ids[1]).unwrap();
    let mut forked = Session::new();
    forked.messages = Arc::new(root);
    forked.user("e");
    store.append_new(&forked.messages).unwrap();

    // Case 4: target is the *other* sibling (d, now off the active path) — switching there
    // abandons only `e` (the active path's own suffix after the common ancestor `b`), not `c`
    // (which lives on the *other* branch, not the one currently active).
    let abandoned = store.abandoned_by_switch(&ids[3]);
    assert_eq!(abandoned.len(), 1);
    assert!(
        matches!(&abandoned[0].1.content[0], ContentBlock::Text{text, ..} if text.as_ref() == "e")
    );
}

#[test]
fn orphaned_off_branch_survives_compaction_of_its_ancestor() {
    // The documented-but-previously-unverified edge case (module doc comment's
    // "compaction vs. branching" section): build [a, b, c, d], switch back to `b` and fork `e`
    // off it (leaving [c, d] off-branch, rooted at `b`), then compact the *active* path [a, b, e]
    // down to a single summary. `b` (c's parent) is deleted by the compaction along with `a` and
    // `e` — this must not panic, corrupt the tree, or silently delete `c`/`d`'s *content*; it
    // should leave them reachable as their own orphaned subtree, rooted at the now-missing `b`.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("a");
    session.user("b");
    session.user("c");
    session.user("d");
    store.append_new(&session.messages).unwrap();
    let ids = store.active_ids().to_vec(); // [a, b, c, d]
    let d_id = ids[3].clone();

    let root = store.switch_active(&ids[1]).unwrap(); // back to b
    let mut forked = Session::new();
    forked.messages = Arc::new(root);
    forked.user("e");
    store.append_new(&forked.messages).unwrap(); // active path is now [a, b, e]; c, d are off-branch

    // Compact the active path away entirely — a, b, and e are all deleted; c and d (off-branch,
    // rooted at the now-gone `b`) must survive on disk.
    store.rewrite(&[Message::user("summary")]).unwrap();
    assert_eq!(store.active_ids().len(), 1);

    let raw = fs::read_to_string(store.path()).unwrap();
    assert!(
        raw.contains("\"c\""),
        "orphaned branch content was deleted:\n{raw}"
    );
    assert!(
        raw.contains("\"d\""),
        "orphaned branch content was deleted:\n{raw}"
    );
    assert!(!raw.contains("\"a\"") && !raw.contains("\"e\""));

    // It's correctly reported by `list_branches` — not silently dropped from the listing, and
    // (checked *before* navigating to it) correctly NOT the active branch.
    let branches = store.list_branches();
    let orphan_branch = branches
        .iter()
        .find(|b| b.leaf_id == d_id)
        .expect("the orphaned branch should still be listed");
    assert_eq!(orphan_branch.message_count, 2);
    assert!(!orphan_branch.is_active);

    // The orphaned subtree is still fully navigable from its own tip: no panic, and its content
    // reconstructs correctly even though the path can't walk back past the deleted `b`.
    let orphaned = store.switch_active(&d_id).unwrap();
    assert_eq!(
        orphaned.len(),
        2,
        "expected just [c, d] — the walk stops at the missing `b`"
    );
    assert!(
        matches!(&orphaned[0].content[0], ContentBlock::Text{text, ..} if text.as_ref() == "c")
    );
    assert!(
        matches!(&orphaned[1].content[0], ContentBlock::Text{text, ..} if text.as_ref() == "d")
    );

    // And a fresh reopen of the file reconstructs the same picture — the orphaning survives a
    // round-trip through disk, not just the in-memory state from the same process. Having
    // switched to it above, it's now the active branch on reopen too.
    let (reopened, _) = repo.open_id(&store.meta().id.clone()).unwrap();
    let reopened_orphan = reopened
        .list_branches()
        .into_iter()
        .find(|b| b.leaf_id == d_id)
        .expect("the orphaned branch survives a reopen");
    assert_eq!(reopened_orphan.message_count, 2);
    assert!(reopened_orphan.is_active);
}

#[test]
fn abandoned_branches_reports_the_full_off_path_chain_and_its_shared_prefix() {
    // Build [a, b, c, d], switch back to `b`, and fork [a, b, e] off it — leaving [c, d] abandoned,
    // rooted at `b`. `abandoned_branches` must report the *whole* [a, b, c, d] chain (not just
    // [c, d]) alongside `shared_prefix_len: 2` (it shares [a, b] with the new active path
    // [a, b, e]), so a caller can choose to skip re-rendering the shared prefix.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("a");
    session.user("b");
    session.user("c");
    session.user("d");
    store.append_new(&session.messages).unwrap();
    let ids = store.active_ids().to_vec(); // [a, b, c, d]

    let root = store.switch_active(&ids[1]).unwrap(); // back to b
    let mut forked = Session::new();
    forked.messages = Arc::new(root);
    forked.user("e");
    store.append_new(&forked.messages).unwrap(); // active path is now [a, b, e]

    // A session with no branches at all reports nothing abandoned.
    let fresh = repo.create(SessionMeta::new("/w2", "m")).unwrap();
    assert!(fresh.abandoned_branches().is_empty());

    let abandoned = store.abandoned_branches();
    assert_eq!(
        abandoned.len(),
        1,
        "exactly one abandoned leaf (d): {abandoned:?}"
    );
    let (shared, messages) = &abandoned[0];
    assert_eq!(
        *shared, 2,
        "shares [a, b] with the new active path [a, b, e]"
    );
    assert_eq!(
        messages.len(),
        4,
        "the whole [a, b, c, d] chain, not just the divergent suffix"
    );
    let texts: Vec<&str> = messages
        .iter()
        .map(|m| match &m.content[0] {
            ContentBlock::Text { text, .. } => text.as_ref(),
            other => panic!("expected a text block, got {other:?}"),
        })
        .collect();
    assert_eq!(texts, vec!["a", "b", "c", "d"]);

    // The active leaf itself must never appear as an "abandoned" branch.
    let active_tip = store.active_ids().last().cloned().unwrap();
    assert_eq!(store.list_branches().len(), 2, "d (abandoned) + e (active)");
    assert!(
        store
            .list_branches()
            .iter()
            .any(|b| b.leaf_id == active_tip && b.is_active)
    );
}

#[test]
fn abandoned_branches_of_an_abandoned_branch_collide_on_the_same_shared_prefix() {
    // Task #34 (pi-parity audit, low-priority/unconfirmed edge case) — investigating whether
    // branching off an *already-abandoned* branch (not off the active path) is handled correctly.
    //
    // Build active [m1, m2]. Branch A off m1: [m1, a1, a2]. Branch B off A's own a1 (not off
    // active at all): [m1, a1, b1]. Restore active to [m1, m2] so both A and B are abandoned
    // leaves. `shared` is always computed against the *active* path only (see
    // `abandoned_branches`'s own loop, `path.zip(self.active.iter())`) — since neither A nor B
    // shares anything with active beyond `m1`, both come out with the *same* `shared == 1`, even
    // though B actually shares two messages ([m1, a1]) with A, not just one ([m1]) with active.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("m1");
    session.user("m2");
    store.append_new(&session.messages).unwrap();
    let ids = store.active_ids().to_vec(); // [m1, m2]

    // Branch A off m1: [m1, a1, a2].
    let root = store.switch_active(&ids[0]).unwrap();
    let mut branch_a = Session::new();
    branch_a.messages = Arc::new(root);
    branch_a.user("a1");
    branch_a.user("a2");
    store.append_new(&branch_a.messages).unwrap();
    let a_ids = store.active_ids().to_vec(); // [m1, a1, a2]
    let a1_id = a_ids[1].clone();

    // Branch B off A's own a1 (an already-abandoned branch, not the active path at all): [m1, a1, b1].
    let root = store.switch_active(&a1_id).unwrap();
    let mut branch_b = Session::new();
    branch_b.messages = Arc::new(root);
    branch_b.user("b1");
    store.append_new(&branch_b.messages).unwrap();

    // Restore active to the original [m1, m2] — both A and B are now abandoned leaves.
    store.switch_active(&ids[1]).unwrap();

    let mut abandoned = store.abandoned_branches();
    assert_eq!(
        abandoned.len(),
        2,
        "both A (leaf a2) and B (leaf b1): {abandoned:?}"
    );
    abandoned.sort_by_key(|(_, messages)| messages.len());
    let (shared_b, messages_b) = &abandoned[0]; // [m1, a1, b1] — the shorter chain
    let (shared_a, messages_a) = &abandoned[1]; // [m1, a1, a2]
    assert_eq!(messages_b.len(), 3);
    assert_eq!(messages_a.len(), 3);

    // The bug this test confirms: both report the same `shared` (their common prefix with
    // *active*), even though B's true divergence point is from A's own a1, two messages deep —
    // not from active's m1, one message deep.
    assert_eq!(*shared_a, 1);
    assert_eq!(
        *shared_b, 1,
        "confirms Task #34: B collides with A's own `shared` value despite actually forking off \
         A (sharing [m1, a1], not just [m1])"
    );

    // Concretely: `export.rs`'s `render_branches_diverging_at` renders `branch_messages[shared..]`
    // for *every* branch at a given `shared` value as sibling `<details>` blocks — so with both at
    // `shared == 1`, B's own box would render `[a1, b1]` (duplicating `a1`, already shown inside
    // A's own separate box as part of `[a1, a2]`), rather than being nested inside A's box showing
    // only its own net-new `[b1]`. Confirmed real; fixing it properly means reshaping
    // `abandoned_branches`'s flat `(shared, messages)` list into an actual tree (and
    // `render_branches_diverging_at` into a recursive renderer) — a disproportionate restructuring
    // for a narrow edge case this crate's RPC surface allows but no real workflow is known to
    // exercise, per this task's own low-priority/unconfirmed framing. Left as documented,
    // proven-real, deliberately unfixed.
    assert_eq!(&messages_b[..*shared_b], &messages_a[..*shared_a]);
}
