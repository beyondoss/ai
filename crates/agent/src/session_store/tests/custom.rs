use super::*;

#[test]
fn torn_leaf_write_recovers_to_pre_switch_state() {
    // A crash mid-write during `switch_active`'s append (the `Leaf` marker) must not corrupt the
    // session — the same "only the last entry can be lost" crash-recovery contract
    // `torn_last_line_is_recovered` proves for `append_new` must hold for this append path too.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let id = store.meta().id.clone();
    let mut session = Session::new();
    session.user("a");
    session.user("b");
    session.user("c");
    store.append_new(&session.messages).unwrap();
    let ids = store.active_ids().to_vec();

    // Simulate a crash partway through appending the `Leaf` marker for a `switch_active(&ids[0])`
    // that never completes: a half-written, unterminated JSON line.
    let path = repo.find_path(&id).unwrap().unwrap();
    let mut f = OpenOptions::new().append(true).open(&path).unwrap();
    write!(f, "{{\"type\":\"leaf\",\"id\":\"x").unwrap();
    drop(f);

    let (reopened, restored) = repo.open_id(&id).unwrap();
    // The torn leaf marker is dropped; the session reopens at its last *complete* state — the
    // full linear chain, tip still `c` — not corrupted, and not silently switched to whatever
    // target the half-recorded marker would have pointed at.
    assert_eq!(restored.messages.len(), 3);
    assert_eq!(reopened.active_ids(), ids.as_slice());
}

#[test]
fn crash_before_rename_leaves_the_original_file_untouched() {
    // `rewrite` (and `rewrite_compacted`'s degenerate fallback to it) writes a complete new file to
    // `.jsonl.tmp`, then atomically renames it over the original — genuinely replacing the active
    // path, so it needs a full atomic swap. (`rewrite_compacted`'s and
    // `switch_active_with_summary`'s *normal* paths are append-only instead — see their own doc
    // comments — since neither one ever needs to replace anything already on disk.) A crash
    // *before* rename's atomic swap — simulated here by leaving a half-written `.tmp` file behind,
    // never renamed — must leave the original completely untouched: the entire point of temp+rename.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let id = store.meta().id.clone();
    let mut session = Session::new();
    session.user("original content");
    store.append_new(&session.messages).unwrap();

    let path = repo.find_path(&id).unwrap().unwrap();
    let original_contents = fs::read_to_string(&path).unwrap();

    // Simulate the crash: a half-written temp file, never renamed over the original.
    let tmp = path.with_extension("jsonl.tmp");
    fs::write(&tmp, "{\"type\":\"session\",\"id\":\"corrupt").unwrap();

    // The original file is untouched — the stray tmp file is simply ignored (`list`/`open_id`
    // only ever look at `.jsonl`, not `.jsonl.tmp`).
    let (_reopened, restored) = repo.open_id(&id).unwrap();
    assert_eq!(restored.messages.len(), 1);
    let dump = serde_json::to_string(restored.messages.as_ref()).unwrap();
    assert!(dump.contains("original content"));
    assert_eq!(fs::read_to_string(&path).unwrap(), original_contents);
    assert_eq!(
        repo.list().unwrap().len(),
        1,
        "the stray .tmp file must not be picked up as a second session"
    );
}

#[test]
fn cyclic_parent_chain_terminates_instead_of_hanging() {
    // A hand-edited or corrupted file could carry a cycle in `parent_id` (no in-process mutation
    // can produce one — every real parent chains to a strictly earlier id). Without a visited-set
    // guard, `path_from_root` would loop forever growing its output unboundedly instead of
    // degrading to "nothing" the way every other malformed-input case in this module does. Two
    // messages pointing at each other as their own parent is the minimal cycle.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let meta = SessionMeta::new("/w", "m");
    let id = meta.id.clone();
    let path = repo.path_for(&meta);
    let mut lines = vec![serde_json::to_string(&Entry::Session(meta)).unwrap()];
    lines.push(
        serde_json::to_string(&Entry::Message {
            id: Some("m1".into()),
            parent_id: Some("m2".into()),
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
    fs::write(&path, lines.join("\n") + "\n").unwrap();

    // Must return promptly (not hang) and produce *some* bounded, non-panicking result rather
    // than an unbounded/cyclic one.
    let (store, session) = repo.open_id(&id).unwrap();
    assert!(
        session.messages.len() <= 2,
        "a 2-node cycle must not produce a longer path: {}",
        session.messages.len()
    );
    assert!(store.active_ids().len() <= 2);

    // list_branches and abandoned_by_switch must also terminate rather than hang, walking the
    // same cyclic structure from different entry points.
    let _ = store.list_branches();
    let _ = store.abandoned_by_switch("m1");
}

#[test]
fn an_oversized_custom_entry_is_refused_rather_than_written() {
    // `data` is opaque and, on the `serve` path, remote-client-supplied — and it is retained three
    // times over (the appended entry, `nodes`, `events`). It is the one input a client can make
    // arbitrarily large for free, so it is the one that needs a ceiling.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();

    let ok = store.append_custom("note", json!({ "v": "x".repeat(1024) }));
    assert!(ok.is_ok(), "an ordinary payload must still be accepted");

    let before = store.active_ids().len();
    let bytes_before = std::fs::metadata(store.path()).unwrap().len();

    let huge = json!({ "v": "x".repeat(MAX_CUSTOM_ENTRY_BYTES + 1) });
    let err = store
        .append_custom("note", huge)
        .expect_err("a payload past the ceiling must be refused");
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);

    // Refused means *nothing happened*: the rejection must not half-append, advance the tip, or
    // leave the blob behind in any of the three places it would otherwise be retained.
    assert_eq!(
        store.active_ids().len(),
        before,
        "a refused entry must not occupy a slot on the active chain"
    );
    assert_eq!(
        std::fs::metadata(store.path()).unwrap().len(),
        bytes_before,
        "a refused entry must not have reached the file either"
    );
}

#[test]
fn append_custom_participates_in_tree_traversal_but_is_skipped_from_active_messages() {
    // pi: session-manager/save-entry.test.ts, "saves custom entries and includes them in tree
    // traversal" — a custom entry chains into the tree like any other entry (a later message's
    // parent is the custom entry's id, not skipped over it), is reported by full-tree traversal,
    // but contributes nothing to the materialized `Session.messages`/LLM context.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("hello");
    store.append_new(&session.messages).unwrap();
    let msg1_id = store.active_ids()[0].clone();

    let custom_id = store
        .append_custom("my_data", json!({"foo": "bar"}))
        .unwrap();

    session.push(Message::assistant(vec![ContentBlock::text("hi")]));
    store.append_new(&session.messages).unwrap();
    let msg2_id = store.active_ids().last().unwrap().clone();

    // The custom entry sits between msg1 and msg2 in the chain.
    assert_eq!(
        msg2_id,
        store.active_ids()[2],
        "path must be [msg1, custom, msg2]"
    );
    assert_eq!(store.active_ids()[1], custom_id);

    let tree = store.tree();
    let custom_node = tree.iter().find(|n| n.id == custom_id).unwrap();
    assert_eq!(custom_node.parent_id.as_deref(), Some(msg1_id.as_str()));
    assert_eq!(custom_node.role, None, "a custom entry has no message role");
    let msg2_node = tree.iter().find(|n| n.id == msg2_id).unwrap();
    assert_eq!(
        msg2_node.parent_id.as_deref(),
        Some(custom_id.as_str()),
        "the message appended after a custom entry must chain directly off it"
    );

    // buildSessionContext-equivalent: only the two real messages, not the custom entry.
    assert_eq!(store.active_ids().len(), 3);
    let (_, restored) = repo.open_id(&store.meta().id.clone()).unwrap();
    assert_eq!(
        restored.messages.len(),
        2,
        "a custom entry must be skipped when materializing Session.messages"
    );
}

#[test]
fn custom_entries_survive_reopen_and_a_compaction_of_an_unrelated_later_range() {
    // A custom entry appended mid-session must round-trip through a reopen with its content intact,
    // and must not confuse `rewrite_compacted`'s folded-message counting when a *later* compaction
    // folds messages that came after it — the whole point of the message-counting walk (as opposed
    // to a raw positional slice) in `rewrite_compacted`.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("a");
    store.append_new(&session.messages).unwrap();
    let custom_id = store
        .append_custom("preset-state", json!({"name": "plan"}))
        .unwrap();
    session.push(Message::assistant(vec![ContentBlock::text("b")]));
    session.user("c");
    store.append_new(&session.messages).unwrap();

    let (reopened, restored) = repo.open_id(&store.meta().id.clone()).unwrap();
    assert_eq!(restored.messages.len(), 3);
    let custom_node = reopened
        .tree()
        .into_iter()
        .find(|n| n.id == custom_id)
        .unwrap();
    assert_eq!(
        custom_node.preview.as_deref(),
        Some("[custom: preset-state]")
    );

    // Compact away everything: a summary replacing all 3 messages down to 1.
    let mut summary_session = Session::new();
    summary_session.push(Message::user(format!(
        "{}\n\nsummary text",
        agent_core::compaction::SUMMARY_MARKER
    )));
    store
        .rewrite_compacted(
            &summary_session.messages,
            CompactionMeta {
                tokens_before: 100,
                provenance: Default::default(),
            },
        )
        .unwrap();

    // The custom entry (off the new active path, part of the folded-away region) must still be
    // readable on disk — compaction preserves folded content, it doesn't delete it (see
    // `rewrite_compacted`'s own doc comment) — and the new active path must be exactly the summary.
    let (_, after) = repo.open_id(&store.meta().id.clone()).unwrap();
    assert_eq!(after.messages.len(), 1);
    let tree_after = store.tree();
    assert!(
        tree_after.iter().any(|n| n.id == custom_id),
        "the custom entry must still be present in the tree after compaction folded past it"
    );
}

#[test]
fn fork_at_entry_drops_custom_entries_but_keeps_the_surrounding_messages() {
    // Track C-M2: a custom entry on the forked path has no message representation to carry into the
    // new session (see `fork_at_entry`'s doc comment) — it must be silently dropped, while the real
    // messages before and after it fork normally with an unbroken parent chain.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("a");
    store.append_new(&session.messages).unwrap();
    store.append_custom("marker", json!({"k": "v"})).unwrap();
    session.push(Message::assistant(vec![ContentBlock::text("b")]));
    store.append_new(&session.messages).unwrap();
    let msg2_id = store.active_ids().last().unwrap().clone();

    let session_id = store.meta().id.clone();
    let (forked, fsession) = repo.fork_at_entry(&session_id, &msg2_id, false).unwrap();
    assert_eq!(
        fsession.messages.len(),
        2,
        "both real messages must survive the fork"
    );
    let dump = serde_json::to_string(fsession.messages.as_ref()).unwrap();
    assert!(dump.contains("\"a\"") && dump.contains("\"b\""));

    let new_ids = forked.active_ids().to_vec();
    assert_eq!(
        new_ids.len(),
        2,
        "the custom entry must not occupy a slot in the forked chain"
    );
    assert_eq!(
        forked
            .tree()
            .iter()
            .find(|n| n.id == new_ids[1])
            .unwrap()
            .parent_id
            .as_deref(),
        Some(new_ids[0].as_str()),
        "the second message's parent must be the first message directly, not the dropped custom entry"
    );
}

#[test]
fn rewrite_compacted_folded_ids_are_correct_when_a_custom_entry_sits_inside_the_folded_range() {
    // Regression guard for the `rewrite_compacted` fix that came with Track C-M2: a plain
    // `self.active[..folded_count]` positional slice (the original implementation) would
    // misalign the moment a custom entry occupies a slot in `self.active` without contributing to
    // `self.persisted` — a custom entry inside the folded region would silently eat one of the
    // slots meant for a real folded message, leaving the true last folded message unlisted in the
    // `Entry::Compaction` provenance record. This drives that exact shape: [msg1, custom, msg2,
    // msg3] compacted down to a 1-message summary (folding all 3 real messages away) must list
    // all 3 real ids (and the custom id, since it structurally sits within the folded region) in
    // `folded_ids` — not silently drop msg3.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("one");
    store.append_new(&session.messages).unwrap();
    let msg1_id = store.active_ids()[0].clone();
    let custom_id = store.append_custom("marker", json!({"k": "v"})).unwrap();
    session.user("two");
    session.user("three");
    store.append_new(&session.messages).unwrap();
    let all_ids = store.active_ids().to_vec();
    assert_eq!(
        all_ids,
        vec![
            msg1_id.clone(),
            custom_id.clone(),
            all_ids[2].clone(),
            all_ids[3].clone()
        ]
    );
    let msg2_id = all_ids[2].clone();
    let msg3_id = all_ids[3].clone();

    let compacted_messages = vec![Message::user(format!(
        "{}\n\nrecap",
        agent_core::compaction::SUMMARY_MARKER
    ))];
    store
        .rewrite_compacted(
            &compacted_messages,
            CompactionMeta {
                tokens_before: 999,
                provenance: Default::default(),
            },
        )
        .unwrap();
    assert_eq!(store.active_ids().len(), 1);

    let raw = fs::read_to_string(store.path()).unwrap();
    let lines: Vec<Value> = raw
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let compaction_entry = lines
        .iter()
        .find(|v| v["type"] == json!("compaction"))
        .expect("exactly one compaction entry");
    let folded_ids: Vec<String> = compaction_entry["folded_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();

    assert!(
        folded_ids.contains(&msg1_id),
        "folded_ids must list the first folded message: {folded_ids:?}"
    );
    assert!(
        folded_ids.contains(&msg2_id),
        "folded_ids must list the second folded message: {folded_ids:?}"
    );
    assert!(
        folded_ids.contains(&msg3_id),
        "folded_ids must list the third (last) folded message — this is exactly what a naive \
         positional slice would drop once a custom entry occupies a slot: {folded_ids:?}"
    );
}

#[test]
fn rewrite_compacted_preserves_the_model_and_thinking_level_active_at_the_old_tip() {
    // pi-parity gap (fixed): `rewrite_compacted`'s new active chain starts fully detached
    // (`parent: None`), so `change_at`'s ancestor walk from any new-chain id could never reach a
    // `model_changes`/`level_changes` entry anchored on the now-folded chain — a model/thinking-
    // level switch made before a compaction became permanently unrecoverable. Fixed by resolving
    // the effective value at the old tip and re-anchoring it onto the new chain's `None` baseline.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let id = store.meta().id.clone();
    let mut session = Session::new();
    session.user("one");
    store.append_new(&session.messages).unwrap();
    store.record_model_change("gpt-5").unwrap();
    store.record_thinking_level_change("high").unwrap();
    session.user("two");
    session.user("three");
    store.append_new(&session.messages).unwrap();

    let old_tip = store.active_ids().last().unwrap().clone();
    assert_eq!(store.model_at(&old_tip), Some("gpt-5"));
    assert_eq!(store.thinking_level_at(&old_tip), Some("high"));

    let compacted_messages = vec![Message::user(format!(
        "{}\n\nrecap",
        agent_core::compaction::SUMMARY_MARKER
    ))];
    store
        .rewrite_compacted(
            &compacted_messages,
            CompactionMeta {
                tokens_before: 999,
                provenance: Default::default(),
            },
        )
        .unwrap();

    let new_tip = store.active_ids().last().unwrap().clone();
    assert_eq!(
        store.model_at(&new_tip),
        Some("gpt-5"),
        "the model switch must survive the compaction, not vanish"
    );
    assert_eq!(
        store.thinking_level_at(&new_tip),
        Some("high"),
        "the thinking-level switch must survive the compaction, not vanish"
    );

    // Must survive a reopen too — persisted to disk, not just patched into the live in-memory maps.
    drop(store);
    let (reopened, _restored) = repo.open_id(&id).unwrap();
    let reopened_tip = reopened.active_ids().last().unwrap().clone();
    assert_eq!(reopened.model_at(&reopened_tip), Some("gpt-5"));
    assert_eq!(reopened.thinking_level_at(&reopened_tip), Some("high"));
}

#[test]
fn rewrite_compacted_preserves_a_title_resolvable_by_a_later_fork() {
    // Same "detached new chain" gap as
    // `rewrite_compacted_preserves_the_model_and_thinking_level_active_at_the_old_tip`, now also
    // closed for `title_changes` (pass 15 pi-parity fix): a rename set before a compaction must
    // still be resolvable by a fork landing after it — `title_at_or_root` has no public accessor of
    // its own, so this drives it through its only real consumer, `fork`.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let id = store.meta().id.clone();
    let mut session = Session::new();
    session.user("one");
    store.append_new(&session.messages).unwrap();
    store.set_title("Pre-Compaction Name").unwrap();
    session.user("two");
    session.user("three");
    store.append_new(&session.messages).unwrap();

    let compacted_messages = vec![Message::user(format!(
        "{}\n\nrecap",
        agent_core::compaction::SUMMARY_MARKER
    ))];
    store
        .rewrite_compacted(
            &compacted_messages,
            CompactionMeta {
                tokens_before: 999,
                provenance: Default::default(),
            },
        )
        .unwrap();

    let (forked, _) = repo.fork(&id, usize::MAX).unwrap();
    assert_eq!(
        forked.meta().title.as_deref(),
        Some("Pre-Compaction Name"),
        "a title set before a compaction must remain resolvable by a fork landing after it"
    );
}

#[test]
fn rewrite_compacted_carries_a_custom_entry_forward_when_it_sits_in_the_kept_suffix() {
    // pi-parity gap (fixed): a *partial* compaction only folds the front of `self.active`; a
    // custom entry (Track C-M2) sitting in the surviving *kept* suffix was previously neither
    // recorded as folded provenance nor carried into the new chain (`new_nodes` was built solely
    // from `messages: &[Message]`, which has no representation for a custom entry) — it just
    // silently disappeared. This drives that exact shape: [msg1, msg2, custom, msg3, msg4]
    // partially compacted (fold msg1+msg2 only) must keep the custom entry, in position, on the
    // new active path.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("one");
    session.user("two");
    store.append_new(&session.messages).unwrap();
    store.append_custom("marker", json!({"k": "v"})).unwrap();
    session.user("three");
    session.user("four");
    store.append_new(&session.messages).unwrap();
    assert_eq!(
        store.active_ids().len(),
        5,
        "msg1, msg2, custom, msg3, msg4"
    );

    // Fold only msg1+msg2 away — msg3/msg4 (and the custom entry between them and the fold
    // boundary) are the kept suffix.
    let compacted_messages = vec![
        Message::user(format!(
            "{}\n\nrecap",
            agent_core::compaction::SUMMARY_MARKER
        )),
        Message::user("three"),
        Message::user("four"),
    ];
    store
        .rewrite_compacted(
            &compacted_messages,
            CompactionMeta {
                tokens_before: 999,
                provenance: Default::default(),
            },
        )
        .unwrap();

    let active_ids = store.active_ids().to_vec();
    assert_eq!(
        active_ids.len(),
        4,
        "summary, custom, msg3, msg4 — the custom entry must not vanish: {active_ids:?}"
    );
    let active: std::collections::HashSet<&str> = active_ids.iter().map(String::as_str).collect();
    let custom_survivors: Vec<TreeNode> = store
        .tree()
        .into_iter()
        .filter(|n| active.contains(n.id.as_str()) && n.role.is_none())
        .collect();
    assert_eq!(
        custom_survivors.len(),
        1,
        "exactly one custom node must be on the new active path: {active_ids:?}"
    );
    assert_eq!(
        custom_survivors[0].preview.as_deref(),
        Some("[custom: marker]")
    );

    // The materialized message content is unaffected (custom entries contribute nothing to
    // `Session.messages`) — still exactly [summary, "three", "four"].
    let (_repo2, session) = SessionStore::open(store.path().to_path_buf()).unwrap();
    assert_eq!(session.messages.len(), 3);
    assert!(
        matches!(&session.messages[1].content[0], ContentBlock::Text { text, .. } if text.as_ref() == "three")
    );
    assert!(
        matches!(&session.messages[2].content[0], ContentBlock::Text { text, .. } if text.as_ref() == "four")
    );
}
