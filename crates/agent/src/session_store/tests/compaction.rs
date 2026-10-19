use super::*;

#[test]
fn rewrite_compacted_preserves_folded_messages_and_records_provenance() {
    // Compact 6 messages down to [summary, kept5, kept6] — the first 4 are folded away. Unlike
    // `rewrite`, this must (a) keep every folded message physically readable on disk by its
    // original id, (b) write exactly one `Entry::Compaction` provenance record naming them, and
    // (c) still start the new active path as its own fresh, detached chain — *not* linked back
    // into the folded prefix, which would resurrect it into the live transcript.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    for text in ["one", "two", "three", "four", "five", "six"] {
        session.user(text);
    }
    store.append_new(&session.messages).unwrap();
    let old_ids = store.active_ids().to_vec();
    assert_eq!(old_ids.len(), 6);

    let compacted_messages = vec![
        Message::user(format!(
            "{}\n\nrecap of the folded work",
            agent_core::compaction::SUMMARY_MARKER
        )),
        session.messages[4].clone(),
        session.messages[5].clone(),
    ];
    store
        .rewrite_compacted(
            &compacted_messages,
            CompactionMeta {
                tokens_before: 12345,
                provenance: Default::default(),
            },
        )
        .unwrap();
    assert_eq!(store.active_ids().len(), 3);
    assert_eq!(store.meta().compactions, 1);
    assert_eq!(store.meta().dropped_messages, 3); // net shrinkage: 4 folded -> 1 summary

    let (reopened, restored) = repo.open_id(&store.meta().id.clone()).unwrap();
    assert_eq!(restored.messages.len(), 3);
    assert_eq!(reopened.active_ids().len(), 3);

    // Parse every line and find the two entries of interest, rather than substring-sniffing raw
    // JSON (id ordering across the file isn't guaranteed).
    let raw = fs::read_to_string(reopened.path()).unwrap();
    let lines: Vec<Value> = raw
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();

    // Every folded message is still physically present, by its original id and content.
    for (id, text) in old_ids[..4].iter().zip(["one", "two", "three", "four"]) {
        let found = lines.iter().find(|v| v["id"] == json!(id));
        assert!(
            found.is_some(),
            "folded message {id} ({text}) missing from disk: {raw}"
        );
    }

    // Exactly one compaction provenance record, naming the folded ids and carrying tokens_before.
    let compaction_entries: Vec<&Value> = lines
        .iter()
        .filter(|v| v["type"] == json!("compaction"))
        .collect();
    assert_eq!(
        compaction_entries.len(),
        1,
        "expected exactly one compaction entry: {raw}"
    );
    let entry = compaction_entries[0];
    assert_eq!(entry["tokens_before"], json!(12345));
    assert_eq!(entry["summary"], json!("recap of the folded work"));
    assert_eq!(
        entry["folded_ids"].as_array().unwrap().len(),
        4,
        "should name all 4 folded messages: {entry:#?}"
    );
    for id in &old_ids[..4] {
        assert!(
            entry["folded_ids"].as_array().unwrap().contains(&json!(id)),
            "folded_ids missing {id}: {entry:#?}"
        );
    }

    // The new active path's first node (the summary) starts a fresh, detached chain — *not*
    // linked back into the folded prefix. `path_from_root` walks a tip's whole parent chain to
    // build the live session, so a real link there would resurrect every folded message back into
    // the "active" transcript, undoing the compaction (already caught once: `restored.messages`
    // above would be 7, not 3, if this regressed).
    let summary_entry = lines
        .iter()
        .find(|v| {
            v["type"] == json!("message") && v.to_string().contains("recap of the folded work")
        })
        .expect("summary message entry not found");
    assert_eq!(
        summary_entry["parent_id"],
        Value::Null,
        "the summary message must start a fresh chain, not link back into the folded prefix: {summary_entry:#?}"
    );

    // The folded prefix survives as its own self-contained sub-chain (still linked to each other
    // exactly as before compaction), unreachable from the new tip but not orphaned from *root* —
    // the first folded message's parent_id is whatever it always was (`None`, the session root).
    let first_folded_entry = lines
        .iter()
        .find(|v| v["id"] == json!(old_ids[0]))
        .expect("first folded message entry not found");
    assert_eq!(first_folded_entry["parent_id"], Value::Null);
    let second_folded_entry = lines
        .iter()
        .find(|v| v["id"] == json!(old_ids[1]))
        .expect("second folded message entry not found");
    assert_eq!(second_folded_entry["parent_id"], json!(old_ids[0]));
}

#[test]
fn rewrite_compacted_persists_file_provenance_and_it_survives_a_reopen() {
    // Fix 2 (pi-parity remediation, Round 2): `agent_core::Session.compaction`
    // (`CompactionProvenance`'s `read_files`/`modified_files`/`last_reason`, folded forward
    // turn-to-turn by `agent_core::compaction::merge_provenance`) was purely in-memory — the
    // on-disk `Entry::Compaction` record had no field to carry it at all, and
    // `SessionStore::open` never restored it. So a `serve` restart or session reattach after
    // even one compaction round silently forgot every file that round already knew about: the
    // *next* compaction's `<read-files>`/`<modified-files>` tags would quietly omit everything
    // from before the restart. Reproduced here exactly as that would happen: seed
    // `Session.compaction` as if a live compaction round just fired, persist it via
    // `rewrite_compacted`, then reopen the session fresh (a brand-new
    // `SessionStore::open`/`repo.open_id` call, not the same in-memory `store`) and assert the
    // provenance is back.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("a");
    session.user("b");
    store.append_new(&session.messages).unwrap();

    let provenance = CompactionProvenance {
        read_files: vec!["src/main.rs".to_string(), "README.md".to_string()],
        modified_files: vec!["src/lib.rs".to_string()],
        compactions: 1,
        last_reason: Some(agent_core::compaction::CompactionReason::Threshold),
        todos: None,
        memory_notes: vec![],
        user_requests: vec!["Fix the parser; don't touch the tests.".to_string()],
    };
    store
        .rewrite_compacted(
            &[Message::user("summary")],
            CompactionMeta {
                tokens_before: 4242,
                provenance: provenance.clone(),
            },
        )
        .unwrap();

    let (_reopened, restored) = repo.open_id(&store.meta().id.clone()).unwrap();
    assert_eq!(
        restored.compaction.read_files, provenance.read_files,
        "read_files must survive a fresh reopen, not reset to empty"
    );
    assert_eq!(
        restored.compaction.modified_files, provenance.modified_files,
        "modified_files must survive a fresh reopen, not reset to empty"
    );
    assert_eq!(
        restored.compaction.last_reason, provenance.last_reason,
        "last_reason must survive a fresh reopen"
    );
    assert_eq!(
        restored.compaction.user_requests, provenance.user_requests,
        "the carried user requests must survive a fresh reopen"
    );
    // The counter is restored from `SessionMeta::compactions` (already correctly persisted
    // separately), not trusted verbatim from the record itself — see `SessionStore::open`'s own
    // doc comment on why.
    assert_eq!(restored.compaction.compactions, store.meta().compactions);
}

#[test]
fn a_compacted_away_todo_list_survives_a_fresh_reopen() {
    // Once `apply_summary` has dropped the `todo` tool_use block, `Session.compaction.todos` is the
    // only copy of the model's plan that exists. If it didn't round-trip through the store, every
    // `serve` restart or reattach past a compaction would silently lose it — and the very next
    // compaction would write a summary with no `<todo_list>` block at all.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("a");
    session.user("b");
    store.append_new(&session.messages).unwrap();

    let todos = serde_json::json!([
        { "content": "Wire the retry loop", "activeForm": "Wiring the retry loop", "status": "in_progress" },
    ]);
    store
        .rewrite_compacted(
            &[Message::user("summary")],
            CompactionMeta {
                tokens_before: 4242,
                provenance: CompactionProvenance {
                    todos: Some(todos.clone()),
                    compactions: 1,
                    ..Default::default()
                },
            },
        )
        .unwrap();

    let (_reopened, restored) = repo.open_id(&store.meta().id.clone()).unwrap();
    assert_eq!(
        restored.compaction.todos,
        Some(todos),
        "the todo list must survive a fresh reopen, not reset to None"
    );
}

#[test]
fn a_compaction_record_written_before_the_todos_field_existed_still_loads() {
    // `#[serde(default)]` on the field, exercised against the literal on-disk shape an older build
    // wrote: a `compaction` entry with no `todos` key at all must parse (and simply restore `None`),
    // not fail the line and silently drop the whole provenance record with it.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("a");
    session.user("b");
    store.append_new(&session.messages).unwrap();
    store
        .rewrite_compacted(
            &[Message::user("summary")],
            CompactionMeta {
                tokens_before: 1,
                provenance: CompactionProvenance {
                    read_files: vec!["a.rs".into()],
                    todos: Some(serde_json::json!([{ "content": "x", "status": "pending" }])),
                    compactions: 1,
                    ..Default::default()
                },
            },
        )
        .unwrap();

    // Strip the field back out of the on-disk record, exactly as a pre-`todos` build left it.
    let path = store.path().to_path_buf();
    let stripped = fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(|line| {
            let mut v: serde_json::Value = serde_json::from_str(line).unwrap();
            if v.get("type").and_then(|t| t.as_str()) == Some("compaction") {
                assert!(
                    v.as_object_mut().unwrap().remove("todos").is_some(),
                    "the field must have been written in the first place"
                );
            }
            serde_json::to_string(&v).unwrap()
        })
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(&path, format!("{stripped}\n")).unwrap();

    let (_reopened, restored) = repo.open_id(&store.meta().id.clone()).unwrap();
    assert_eq!(restored.compaction.todos, None);
    assert_eq!(
        restored.compaction.read_files,
        vec!["a.rs".to_string()],
        "the rest of the record must still restore — the line parsed, it wasn't skipped"
    );
}

#[test]
fn rewrite_compacted_folds_file_provenance_forward_across_two_rounds_and_both_survive_a_reopen() {
    // The forward-folding half of Fix 2: a *second* compaction round's own `CompactionMeta
    // .provenance` (what `agent_core::compaction::merge_provenance` already folds forward in
    // memory) must still be a complete snapshot on its own — `SessionStore::open` only ever reads
    // the *last* `Entry::Compaction` record, trusting that it already carries every earlier
    // round's files, exactly like `Session.compaction` already does turn-to-turn in memory.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("a");
    session.user("b");
    session.user("c");
    session.user("d");
    store.append_new(&session.messages).unwrap();

    let round1 = CompactionProvenance {
        read_files: vec!["a.rs".to_string()],
        modified_files: vec![],
        compactions: 1,
        last_reason: Some(agent_core::compaction::CompactionReason::Threshold),
        todos: None,
        memory_notes: vec![],
        user_requests: vec![],
    };
    store
        .rewrite_compacted(
            &[
                Message::user("summary1"),
                Message::user("c"),
                Message::user("d"),
            ],
            CompactionMeta {
                tokens_before: 100,
                provenance: round1,
            },
        )
        .unwrap();

    // Round 2 folds round 1's `a.rs` forward with its own new activity, matching what
    // `merge_provenance` would actually do — the persisted record must reflect that
    // already-folded union, not just round 2's own new files.
    let round2 = CompactionProvenance {
        read_files: vec!["a.rs".to_string(), "b.rs".to_string()],
        modified_files: vec!["c.rs".to_string()],
        compactions: 2,
        last_reason: Some(agent_core::compaction::CompactionReason::Manual),
        todos: None,
        memory_notes: vec![],
        user_requests: vec![],
    };
    store
        .rewrite_compacted(
            &[Message::user("summary2")],
            CompactionMeta {
                tokens_before: 200,
                provenance: round2,
            },
        )
        .unwrap();

    let (_reopened, restored) = repo.open_id(&store.meta().id.clone()).unwrap();
    assert_eq!(restored.compaction.read_files, vec!["a.rs", "b.rs"]);
    assert_eq!(restored.compaction.modified_files, vec!["c.rs"]);
    assert_eq!(
        restored.compaction.last_reason,
        Some(agent_core::compaction::CompactionReason::Manual)
    );
    assert_eq!(restored.compaction.compactions, 2);
}

#[test]
fn tree_reports_entry_kind_for_a_compaction_and_a_branch_summary_not_just_a_plain_message() {
    // Track L25 (pi-parity fix): `Entry::BranchSummary`/`Entry::Compaction` used to be
    // indistinguishable from an ordinary message once materialized into `tree()`'s output — a
    // client had no way to tell "this is a recap"/"a compaction happened here" apart from an
    // everyday turn without guessing off `preview` text. Proves `entry_kind` recovers both, and
    // that an ordinary message still reports `"message"`.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    for text in ["one", "two", "three", "four"] {
        session.user(text);
    }
    store.append_new(&session.messages).unwrap();
    let ids = store.active_ids().to_vec();

    // An ordinary message reports the baseline kind.
    let plain = store.tree().into_iter().find(|n| n.id == ids[0]).unwrap();
    assert_eq!(plain.entry_kind, "message");

    // A branch-summary recap, still a real `NodeContent::Message` under the hood (so it reaches
    // the model), reports its own distinct kind instead.
    store
        .switch_active_with_summary(
            &ids[0],
            "recap of the abandoned branch",
            &ids[3],
            BranchSummaryDetails::default(),
        )
        .unwrap();
    let summary_id = store.active_ids().last().unwrap().clone();
    let tree = store.tree();
    let summary_node = tree.iter().find(|n| n.id == summary_id).unwrap();
    assert_eq!(summary_node.entry_kind, "branch_summary");
    assert!(
        summary_node.role.is_some(),
        "a branch-summary node is still a real message with a role, just distinguishable by kind"
    );

    // A compaction round's own provenance record, never itself a chain node, is still surfaced as
    // its own synthetic entry.
    store
        .rewrite_compacted(
            &[Message::user("kept after compaction")],
            CompactionMeta {
                tokens_before: 999,
                provenance: Default::default(),
            },
        )
        .unwrap();
    let tree = store.tree();
    let compaction_node = tree
        .iter()
        .find(|n| n.entry_kind == "compaction")
        .expect("a compaction node must be recoverable from tree()'s output");
    assert!(
        compaction_node
            .preview
            .as_deref()
            .is_some_and(|p| p.contains("999")),
        "the compaction node's preview should still mention tokens_before: {compaction_node:#?}"
    );

    // Reopening from disk must agree — this isn't just an in-memory-only artifact of the live
    // instance that ran the compaction.
    let (reopened, _) = repo.open_id(&store.meta().id.clone()).unwrap();
    let reopened_tree = reopened.tree();
    assert!(
        reopened_tree.iter().any(|n| n.entry_kind == "compaction"),
        "a reopened session must still report the compaction node: {reopened_tree:#?}"
    );
    assert!(
        reopened_tree
            .iter()
            .any(|n| n.entry_kind == "branch_summary"),
        "a reopened session must still report the branch-summary node: {reopened_tree:#?}"
    );
}

#[test]
fn rewrite_compacted_appends_without_rewriting_existing_bytes() {
    // The whole point of H-7: every byte already on disk before the call must still be there,
    // byte-for-byte, as a strict *prefix* of the file afterward — proof this is a pure append, not
    // a rewrite (which would reserialize and potentially reorder everything).
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    for text in ["one", "two", "three", "four", "five"] {
        session.user(text);
    }
    store.append_new(&session.messages).unwrap();
    let path = repo.find_path(&store.meta().id.clone()).unwrap().unwrap();
    let before = fs::read(&path).unwrap();

    let compacted = vec![
        Message::user(format!(
            "{}\n\nrecap",
            agent_core::compaction::SUMMARY_MARKER
        )),
        session.messages[4].clone(),
    ];
    store
        .rewrite_compacted(
            &compacted,
            CompactionMeta {
                tokens_before: 1,
                provenance: Default::default(),
            },
        )
        .unwrap();

    let after = fs::read(&path).unwrap();
    assert!(
        after.starts_with(&before),
        "every pre-existing byte must survive untouched as a prefix of the file"
    );
    assert!(
        after.len() > before.len(),
        "new content must have been appended"
    );
}

#[test]
fn rewrite_compacted_handles_two_sequential_compactions_back_to_back() {
    // B-L6 pi-parity test gap (fixed): every existing test exercised a single `rewrite_compacted`
    // call in isolation — matches pi's "should handle multiple compactions (only latest matters)"
    // (`compaction.test.ts:377-398`). A second compaction, appended right after the first with no
    // reopen in between, must independently record its own provenance (not overwrite or merge
    // with the first), and the materialized session must reflect only the *latest* cut.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    for text in ["one", "two", "three", "four", "five"] {
        session.user(text);
    }
    store.append_new(&session.messages).unwrap();
    let round1_ids = store.active_ids().to_vec();
    assert_eq!(round1_ids.len(), 5);

    // Round 1: fold "one","two","three" down to [summary1, four, five] — dropped = 2.
    let round1_messages = vec![
        Message::user(format!(
            "{}\n\nfirst summary",
            agent_core::compaction::SUMMARY_MARKER
        )),
        session.messages[3].clone(), // "four"
        session.messages[4].clone(), // "five"
    ];
    store
        .rewrite_compacted(
            &round1_messages,
            CompactionMeta {
                tokens_before: 100,
                provenance: Default::default(),
            },
        )
        .unwrap();
    assert_eq!(store.meta().compactions, 1);
    assert_eq!(store.meta().dropped_messages, 2);
    assert_eq!(store.active_ids().len(), 3);

    // Continue past round 1 with two more ordinary messages — no reopen in between, exercising
    // the in-process (not just persisted-and-reopened) state right after a compaction.
    let mut continued = Session::new();
    continued.messages = Arc::new(round1_messages.clone());
    continued.user("six");
    continued.user("seven");
    store.append_new(&continued.messages).unwrap();
    assert_eq!(store.active_ids().len(), 5);

    // Round 2: fold everything except the very last message ("seven") — dropped = 3.
    let round2_messages = vec![
        Message::user(format!(
            "{}\n\nsecond summary",
            agent_core::compaction::SUMMARY_MARKER
        )),
        continued.messages[4].clone(), // "seven"
    ];
    store
        .rewrite_compacted(
            &round2_messages,
            CompactionMeta {
                tokens_before: 200,
                provenance: Default::default(),
            },
        )
        .unwrap();

    // Both rounds' provenance accumulate independently — not overwritten.
    assert_eq!(store.meta().compactions, 2);
    assert_eq!(store.meta().dropped_messages, 5); // 2 + 3
    assert_eq!(store.active_ids().len(), 2);

    // The materialized session (after a reopen) reflects only the *latest* cut.
    let (reopened, restored) = repo.open_id(&store.meta().id.clone()).unwrap();
    assert_eq!(restored.messages.len(), 2);
    assert!(
        matches!(&restored.messages[0].content[0], ContentBlock::Text { text, .. }
            if text.contains("second summary")),
        "the active path must show the second (latest) summary, not the first: {:?}",
        restored.messages[0].content
    );

    // Exactly two `Entry::Compaction` provenance records — one per round, independently readable.
    let raw = fs::read_to_string(reopened.path()).unwrap();
    let lines: Vec<Value> = raw
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let compaction_entries: Vec<&Value> = lines
        .iter()
        .filter(|v| v["type"] == json!("compaction"))
        .collect();
    assert_eq!(
        compaction_entries.len(),
        2,
        "expected one entry per round: {raw}"
    );
    assert_eq!(compaction_entries[0]["summary"], json!("first summary"));
    assert_eq!(compaction_entries[1]["summary"], json!("second summary"));

    // Round 1's folded originals ("one","two","three") are still physically present, untouched
    // by round 2.
    for (id, text) in round1_ids[..3].iter().zip(["one", "two", "three"]) {
        let found = lines.iter().find(|v| v["id"] == json!(id));
        assert!(
            found.is_some(),
            "round 1's folded message {id} ({text}) missing: {raw}"
        );
    }
}

#[test]
fn rewrite_compacted_never_falls_back_to_a_full_rewrite() {
    // A round whose message count did not shrink (the store lagged the session, or a degenerate
    // round) used to fall back to a full `rewrite`, which drops every side record and re-mints
    // every id. It is an append-only round like any other: the stored messages it replaced are
    // folded, and nothing already on disk is touched.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("one");
    session.user("two");
    store.append_new(&session.messages).unwrap();
    store.record_model_change("model-b").unwrap();
    let before = fs::read(store.path()).unwrap();
    let originals = store.active_ids().to_vec();

    // Same length in as out — neither stored message survives verbatim.
    let same_length = vec![
        Message::user("one (rewritten)"),
        Message::user("two (rewritten)"),
    ];
    store
        .rewrite_compacted(
            &same_length,
            CompactionMeta {
                tokens_before: 1,
                provenance: Default::default(),
            },
        )
        .unwrap();

    let after = fs::read(store.path()).unwrap();
    assert!(after.starts_with(&before), "existing bytes were rewritten");
    assert_eq!(store.meta().compactions, 1);
    let (reopened, restored) = repo.open_id(&store.meta().id.clone()).unwrap();
    assert_eq!(restored.messages.len(), 2);
    let ContentBlock::Text { text, .. } = &restored.messages[0].content[0] else {
        panic!("expected text");
    };
    assert_eq!(text.as_ref(), "one (rewritten)");
    assert_eq!(reopened.model_at_root(), Some("model-b"));
    let raw = fs::read_to_string(reopened.path()).unwrap();
    let folded: Vec<Value> = raw
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter(|v| v["type"] == json!("compaction"))
        .map(|v| v["folded_ids"].clone())
        .collect();
    assert_eq!(folded, vec![json!(originals)], "{raw}");
}

#[test]
fn switch_active_with_summary_appends_without_rewriting_existing_bytes() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("only message");
    store.append_new(&session.messages).unwrap();
    let root_id = store.active_ids()[0].clone();
    let path = repo.find_path(&store.meta().id.clone()).unwrap().unwrap();
    let before = fs::read(&path).unwrap();

    store
        .switch_active_with_summary(
            &root_id,
            "a recap",
            "abandoned-tip",
            BranchSummaryDetails::default(),
        )
        .unwrap();

    let after = fs::read(&path).unwrap();
    assert!(
        after.starts_with(&before),
        "every pre-existing byte must survive untouched as a prefix of the file"
    );
    assert!(
        after.len() > before.len(),
        "new content must have been appended"
    );
}

#[test]
fn torn_compaction_append_leaves_the_pre_compaction_state_fully_valid() {
    // A crash between the `Entry::Compaction` record and the new active-path messages (or mid-way
    // through one of those messages) must not corrupt anything: the *old* tip and everything it
    // points to were never touched by an append-only compaction, so they must still resolve exactly
    // as they did before the interrupted compaction ever started.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    for text in ["one", "two", "three", "four"] {
        session.user(text);
    }
    store.append_new(&session.messages).unwrap();
    let old_ids = store.active_ids().to_vec();
    let id = store.meta().id.clone();
    let path = repo.find_path(&id).unwrap().unwrap();

    // Simulate a crash mid-append: a torn, unterminated compaction-record line with nothing after
    // it — as if the process died right after starting to write the batch.
    let mut f = OpenOptions::new().append(true).open(&path).unwrap();
    write!(f, "{{\"type\":\"compaction\",\"id\":\"x").unwrap();
    drop(f);

    let (reopened, restored) = repo.open_id(&id).unwrap();
    assert_eq!(
        restored.messages.len(),
        4,
        "the pre-compaction transcript must be fully intact"
    );
    assert_eq!(
        reopened.active_ids(),
        old_ids.as_slice(),
        "the tip must still be the pre-compaction tip — an interrupted compaction simply never took effect"
    );
}

#[test]
fn a_compaction_batch_torn_inside_the_new_chain_reopens_at_the_pre_compaction_tip() {
    // The batch is one append: header, compaction record, then the new detached chain. A crash that
    // keeps the first chain messages but not the last must not leave the tip on that partial chain
    // (the summary plus some of the kept messages): the round either took effect whole or not at all.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    for text in ["one", "two", "three", "four", "five", "six"] {
        session.user(text);
    }
    store.append_new(&session.messages).unwrap();
    let old_ids = store.active_ids().to_vec();
    let id = store.meta().id.clone();
    let path = repo.find_path(&id).unwrap().unwrap();

    let compacted = vec![
        Message::user(format!(
            "{}\n\nrecap",
            agent_core::compaction::SUMMARY_MARKER
        )),
        session.messages[4].clone(),
        session.messages[5].clone(),
    ];
    store
        .rewrite_compacted(
            &compacted,
            CompactionMeta {
                tokens_before: 1,
                provenance: Default::default(),
            },
        )
        .unwrap();
    drop(store);

    // Tear the file inside the last chain message: everything before it is intact.
    let raw = fs::read(&path).unwrap();
    let last_start = raw[..raw.len() - 1]
        .iter()
        .rposition(|&b| b == b'\n')
        .unwrap()
        + 1;
    fs::write(&path, &raw[..last_start + 10]).unwrap();

    let (reopened, restored) = repo.open_id(&id).unwrap();
    assert_eq!(
        reopened.active_ids(),
        old_ids.as_slice(),
        "a partially written compaction chain must not become the active path"
    );
    assert_eq!(restored.messages.len(), 6);
}
