use super::*;

#[test]
fn set_label_sets_and_gets() {
    // pi: session-manager/labels.test.ts, "sets and gets labels".
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("hello");
    store.append_new(&session.messages).unwrap();
    let msg_id = store.active_ids()[0].clone();

    assert_eq!(store.get_label(&msg_id), None, "no label initially");
    store.set_label(&msg_id, Some("checkpoint")).unwrap();
    assert_eq!(store.get_label(&msg_id), Some("checkpoint"));

    // The label is visible via `tree()` too, attached to the labeled node.
    let node = store.tree().into_iter().find(|n| n.id == msg_id).unwrap();
    assert_eq!(node.label.as_deref(), Some("checkpoint"));
}

#[test]
fn export_events_records_model_thinking_label_and_custom_changes_in_order() {
    // Track L36 (pi-parity fix): `Entry::ModelChange`/`Entry::ThinkingLevelChange`/`Entry::Label`/
    // `Entry::Custom` were all durably tracked but never reached an HTML export at all —
    // `export_events` is the ordered event log that now lets `crate::export` render them.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("hello");
    store.append_new(&session.messages).unwrap();
    let msg_id = store.active_ids()[0].clone();

    assert!(
        store.export_events().is_empty(),
        "no events recorded yet: {:?}",
        store.export_events()
    );

    store.record_model_change("model-b").unwrap();
    store.record_thinking_level_change("high").unwrap();
    store.set_label(&msg_id, Some("checkpoint")).unwrap();
    let custom_id = store
        .append_custom("beyond:sync", json!({"marker": "m1"}))
        .unwrap();

    assert_eq!(
        store.export_events(),
        &[
            ExportEvent::ModelChange("model-b".to_string()),
            ExportEvent::ThinkingLevelChange("high".to_string()),
            ExportEvent::Label {
                target_id: msg_id.clone(),
                label: Some("checkpoint".to_string()),
            },
            ExportEvent::Custom {
                kind: "beyond:sync".to_string(),
                data: json!({"marker": "m1"}),
            },
        ],
        "events must be recorded in file order"
    );

    // Reopening from disk must recover the exact same ordered log — not just the live instance's
    // own in-memory bookkeeping.
    let (reopened, _) = repo.open_id(&store.meta().id.clone()).unwrap();
    assert_eq!(reopened.export_events(), store.export_events());
    // `custom_id` really is the id `append_custom` returned, sanity-checking the fixture itself.
    assert!(reopened.tree().iter().any(|n| n.id == custom_id));
}

#[test]
fn set_label_clears_with_none() {
    // pi: session-manager/labels.test.ts, "clears labels with undefined".
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("hello");
    store.append_new(&session.messages).unwrap();
    let msg_id = store.active_ids()[0].clone();

    store.set_label(&msg_id, Some("checkpoint")).unwrap();
    assert_eq!(store.get_label(&msg_id), Some("checkpoint"));
    store.set_label(&msg_id, None).unwrap();
    assert_eq!(store.get_label(&msg_id), None);
}

#[test]
fn set_label_last_write_wins() {
    // pi: session-manager/labels.test.ts, "last label wins".
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("hello");
    store.append_new(&session.messages).unwrap();
    let msg_id = store.active_ids()[0].clone();

    store.set_label(&msg_id, Some("first")).unwrap();
    store.set_label(&msg_id, Some("second")).unwrap();
    store.set_label(&msg_id, Some("third")).unwrap();
    assert_eq!(store.get_label(&msg_id), Some("third"));

    // Survives reopen from disk too, not just in-memory state.
    let (reopened, _) = repo.open_id(&store.meta().id.clone()).unwrap();
    assert_eq!(reopened.get_label(&msg_id), Some("third"));
}

#[test]
fn set_label_rejects_unknown_target_id() {
    // pi: session-manager/labels.test.ts, "throws when labeling non-existent entry".
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let err = store.set_label("non-existent", Some("label")).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
}

#[test]
fn labels_are_excluded_from_the_active_session_messages() {
    // pi: session-manager/labels.test.ts, "labels are not included in buildSessionContext". In this
    // module labels never occupy a slot in the message chain at all (see `Entry::Label`'s doc
    // comment), so this holds trivially — proven here so a future refactor can't silently regress
    // it by folding labels into `nodes`/`active`.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("hello");
    store.append_new(&session.messages).unwrap();
    let msg_id = store.active_ids()[0].clone();
    store.set_label(&msg_id, Some("checkpoint")).unwrap();

    assert_eq!(store.active_ids().len(), 1);
    let (_, restored) = repo.open_id(&store.meta().id.clone()).unwrap();
    assert_eq!(restored.messages.len(), 1);
}

#[test]
fn labels_are_preserved_across_fork_at_entry_and_dropped_when_off_path() {
    // pi: session-manager/labels.test.ts, "labels are preserved in createBranchedSession" and
    // "labels not on path are not preserved" — combined into one scenario: msg1/msg2 are labeled
    // and both end up on the forked path; msg3 is labeled but forked *before* (excluded), so its
    // label has nothing to carry forward to.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("first");
    session.push(Message::assistant(vec![ContentBlock::text("hi")]));
    session.user("followup");
    store.append_new(&session.messages).unwrap();
    let ids = store.active_ids().to_vec(); // [msg1, msg2, msg3]

    store.set_label(&ids[0], Some("first")).unwrap();
    store.set_label(&ids[1], Some("second")).unwrap();
    store.set_label(&ids[2], Some("third")).unwrap();

    let session_id = store.meta().id.clone();
    // Fork "before" msg3 — keeps msg1/msg2, excludes msg3.
    let (forked, fsession) = repo.fork_at_entry(&session_id, &ids[2], true).unwrap();
    assert_eq!(fsession.messages.len(), 2);
    let new_ids = forked.active_ids().to_vec();
    assert_eq!(new_ids.len(), 2);

    assert_eq!(
        forked.get_label(&new_ids[0]),
        Some("first"),
        "msg1's label must carry forward under msg1's new id in the forked session"
    );
    assert_eq!(
        forked.get_label(&new_ids[1]),
        Some("second"),
        "msg2's label must carry forward under msg2's new id in the forked session"
    );

    // msg3's label had nothing on the forked path to attach to — it must not silently reappear
    // under some other id, and the forked session must not carry any *extra* stray label.
    assert!(
        !forked
            .tree()
            .iter()
            .any(|n| n.label.as_deref() == Some("third")),
        "a label whose target fell off the forked path must not be preserved"
    );

    // The original session is completely unaffected by the fork.
    assert_eq!(store.get_label(&ids[0]), Some("first"));
    assert_eq!(store.get_label(&ids[1]), Some("second"));
    assert_eq!(store.get_label(&ids[2]), Some("third"));

    // Reopening the forked session from disk must agree (labels actually persisted, not just held
    // in memory on the freshly-created store).
    let (reforked, _) = repo.open_id(&forked.meta().id.clone()).unwrap();
    assert_eq!(reforked.get_label(&new_ids[0]), Some("first"));
    assert_eq!(reforked.get_label(&new_ids[1]), Some("second"));
}

#[test]
fn forking_a_session_with_labels_produces_an_unbroken_parent_chain() {
    // pi: session-manager/labels.test.ts, "rewires children of removed labels when forking" — pi
    // needs an explicit rewiring step there because a label is a real tree node other entries chain
    // off of, so dropping one during a fork would otherwise orphan its children. This module's
    // labels never occupy a chain slot to begin with (see `Entry::Label`'s doc comment), so there is
    // no rewiring problem to solve; this proves the structural guarantee that replaces it: after
    // forking a session with labels interspersed, the new session's parent chain is a single
    // unbroken line from root to tip with no gaps.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("first");
    store.append_new(&session.messages).unwrap();
    let msg1_id = store.active_ids()[0].clone();
    store.set_label(&msg1_id, Some("checkpoint")).unwrap();
    store.record_model_change("model-b").unwrap();
    session.user("followup");
    store.append_new(&session.messages).unwrap();
    let msg2_id = store.active_ids()[1].clone();

    let session_id = store.meta().id.clone();
    let (forked, fsession) = repo.fork_at_entry(&session_id, &msg2_id, false).unwrap();
    assert_eq!(fsession.messages.len(), 2);

    let new_ids = forked.active_ids().to_vec();
    assert_eq!(new_ids.len(), 2);
    assert_eq!(
        forked
            .tree()
            .iter()
            .find(|n| n.id == new_ids[1])
            .unwrap()
            .parent_id
            .as_deref(),
        Some(new_ids[0].as_str()),
        "the second message's parent must be the first message directly, with no gap"
    );
}

#[test]
fn model_and_thinking_level_changes_are_branch_scoped() {
    // A model/thinking-level change recorded on one branch must not leak onto a sibling branch that
    // never passed through the message it was anchored to — the whole point of H6. A change
    // anchored *at* a message describes what applies *after* it (children), not the message itself
    // — switching back to the exact anchor point must NOT see it (that's the point: recovering
    // whatever was true *before* the change, e.g. before a `set_model` call).
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("a");
    session.user("b");
    store.append_new(&session.messages).unwrap();
    let ids = store.active_ids().to_vec(); // [a, b]

    // Recorded at the tip (b) — a "b"-anchored change, applying to whatever comes after b.
    store.record_model_change("model-B").unwrap();
    store.record_thinking_level_change("high").unwrap();
    assert_eq!(
        store.model_at(&ids[1]),
        None,
        "a change anchored AT b must not apply when querying b itself"
    );
    assert_eq!(store.thinking_level_at(&ids[1]), None);

    // Continue past b with "c" — c is b's child, so it must see the change.
    session.user("c");
    store.append_new(&session.messages).unwrap();
    let ids = store.active_ids().to_vec(); // [a, b, c]
    assert_eq!(
        store.model_at(&ids[2]),
        Some("model-B"),
        "b's own change must propagate forward to its child c"
    );
    assert_eq!(store.thinking_level_at(&ids[2]), Some("high"));

    // Branch off *a* (not b) with a new message "d" — this branch never passes through b at all.
    let branch_root = store.switch_active(&ids[0]).unwrap();
    let mut branch_session = Session::new();
    branch_session.messages = Arc::new(branch_root);
    branch_session.user("d");
    store.append_new(&branch_session.messages).unwrap();
    let d_id = store.active_ids()[1].clone();

    assert_eq!(
        store.model_at(&d_id),
        None,
        "the model-B change lives on the a->b->c branch, not the a->d one"
    );
    assert_eq!(store.thinking_level_at(&d_id), None);
    assert_eq!(
        store.model_at(&ids[0]),
        None,
        "a itself has no change recorded strictly before it"
    );

    // Reopening the file must recover the same lookups from disk (not just in-memory state).
    let (reopened, _) = repo.open_id(&store.meta().id.clone()).unwrap();
    assert_eq!(reopened.model_at(&ids[2]), Some("model-B"));
    assert_eq!(reopened.model_at(&d_id), None);
}

#[test]
fn record_model_change_does_not_mutate_metas_own_creation_time_model() {
    // Task #18 (pi-parity investigation): an earlier version of this fix had
    // `record_model_change` also assign `self.meta.model = model`, on the theory that leaving
    // `meta.model` frozen at the session's creation-time model was simply a staleness bug. It
    // regressed `Persistence::model_and_level_at`'s existing fallback contract instead — that
    // lookup relies on `meta.model` staying the session's true *original* model forever, as the
    // "nothing was ever recorded reaching this point" baseline for `switch_branch`/`switch_session`
    // (see `SessionStore::record_model_change`'s own doc comment for the full story, and
    // `tests/serve_session_tree.rs::serve_switch_branch_restores_the_model_active_on_that_branch`,
    // the real end-to-end regression that caught it). This pins the corrected, deliberate
    // non-mutation down so it can't quietly regress back.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "model-a")).unwrap();
    assert_eq!(store.meta().model, "model-a");

    store.record_model_change("model-b").unwrap();
    assert_eq!(
        store.meta().model,
        "model-a",
        "meta.model must stay the session's true creation-time value, even after a set_model"
    );

    // Must hold across a reopen too, not just in the same in-memory `self.meta`.
    let (reopened, _) = repo.open_id(&store.meta().id.clone()).unwrap();
    assert_eq!(reopened.meta().model, "model-a");
    // The per-branch lookup map — the mechanism that actually IS supposed to track this — sees the
    // change instead (anchored at the tree's own root here, since no message was ever pushed).
    assert_eq!(store.model_at_root(), Some("model-b"));
}

#[test]
fn record_thinking_level_change_does_not_mutate_metas_thinking_level_field() {
    // Same reasoning as `record_model_change_does_not_mutate_metas_own_creation_time_model` — the
    // new `SessionMeta::thinking_level` field (Task #18) exists (available for a future consumer,
    // e.g. `run --continue`'s reopen path) but is deliberately never auto-populated by this method,
    // for the identical "would break the tree-fallback contract" reason `model` has.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "model-a")).unwrap();
    assert_eq!(store.meta().thinking_level, None);

    store.record_thinking_level_change("high").unwrap();
    assert_eq!(
        store.meta().thinking_level,
        None,
        "meta.thinking_level is not auto-populated by record_thinking_level_change"
    );
}

#[test]
fn fork_resolves_the_updated_model_via_the_tree_not_the_sessions_stale_meta() {
    // Task #18's actual, real-world fix location: `SessionRepo::fork`/`fork_at_entry`/
    // `fork_from_path` used to copy `src.meta.model` verbatim into the new session's own header —
    // always the *original* creation-time model, regardless of any `set_model` since. Fixed by
    // resolving the model active at the fork point via the same per-branch `model_at` lookup
    // `Persistence::model_and_level_at` (`serve/persistence.rs`) already uses for `switch_branch`/
    // `switch_session`, rather than by mutating `meta.model` itself (which would have broken that
    // other lookup's own fallback — see `record_model_change`'s doc comment).
    //
    // "b" is a *child* of the model change's anchor ("a"), so it's the point that actually observes
    // the new model — forking at the anchor itself ("a") would still correctly see the *original*
    // model, matching the same "anchored-at, not before" contract this whole file already tests
    // elsewhere (`model_and_thinking_level_changes_are_branch_scoped`).
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "model-a")).unwrap();
    let mut session = Session::new();
    session.user("a");
    store.append_new(&session.messages).unwrap();
    store.record_model_change("model-b").unwrap(); // anchored at "a"
    session.user("b"); // "a"'s child — observes the model-b change
    store.append_new(&session.messages).unwrap();

    let session_id = store.meta().id.clone();
    let (forked, _) = repo.fork(&session_id, usize::MAX).unwrap();
    assert_eq!(
        forked.meta().model,
        "model-b",
        "a fork's own header must carry the model active at the fork point, not the session's \
         original one"
    );
}

#[test]
fn fork_does_not_inherit_a_rename_that_happened_after_the_fork_point() {
    // Pass 15 pi-parity fix, the title analogue of the Task #18 fix just above:
    // `SessionRepo::fork`/`fork_at_entry`/`fork_from_path` used to copy `src.meta.title` verbatim
    // into the new session's own header — always the whole-file *latest* rename, regardless of how
    // much later it happened relative to the fork point (`Entry::TitleChange` is whole-session-
    // scoped by design for a *live* session's own displayed title — see that variant's doc comment —
    // but a fork must resolve path-scoped instead, the same way it already does for
    // `model`/`thinking_level`).
    //
    // Exact repro from the audit: two messages, then a rename (doesn't move the tip), then fork
    // from message 1 — the new session must NOT carry "Renamed" forward, since that name was chosen
    // for a completely different, later conversation this branch never actually had.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("first");
    store.append_new(&session.messages).unwrap();
    session.user("second");
    store.append_new(&session.messages).unwrap();
    store.set_title("Renamed").unwrap();
    assert_eq!(store.meta().title.as_deref(), Some("Renamed"));

    let session_id = store.meta().id.clone();
    let (forked, _) = repo.fork(&session_id, 1).unwrap();
    assert_eq!(
        forked.meta().title,
        None,
        "a fork from before the rename must not inherit a title chosen for a later conversation"
    );
}

#[test]
fn fork_inherits_the_title_in_effect_at_the_fork_point_not_a_later_rename() {
    // The path-scoped resolution isn't just "always omit the title" — a rename that *was* actually
    // in effect at the fork point must still carry forward, exactly like `model`/`thinking_level`.
    //
    // "second" is a *child* of the first rename's anchor ("first"), so it's the point that
    // actually observes "Early Name" — same "anchored-at, not before" contract
    // `fork_resolves_the_updated_model_via_the_tree_not_the_sessions_stale_meta` already exercises
    // for model: forking exactly at "second" (the second rename's own anchor) must NOT yet observe
    // "Renamed Later" either, only whatever was already in effect reaching that point.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("first");
    store.append_new(&session.messages).unwrap();
    store.set_title("Early Name").unwrap(); // anchored at "first"
    session.user("second"); // "first"'s child — observes "Early Name"
    store.append_new(&session.messages).unwrap();
    store.set_title("Renamed Later").unwrap(); // anchored at "second", not yet observed by "second" itself
    session.user("third");
    store.append_new(&session.messages).unwrap();

    let session_id = store.meta().id.clone();
    let (forked, _) = repo.fork(&session_id, 2).unwrap();
    assert_eq!(
        forked.meta().title.as_deref(),
        Some("Early Name"),
        "a fork must inherit whatever rename was actually in effect at the fork point, not a \
         later one"
    );
}

#[test]
fn whole_transcript_fork_keeps_a_rename_made_after_the_last_message() {
    // A rename lands anchored at the newest message (the usual auto-title shape: generated after
    // the turn persisted). Cloning the whole transcript copies that moment, so the title comes along;
    // forking one message earlier must still not see it, and a cleared title stays cleared.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("a");
    session.user("b");
    store.append_new(&session.messages).unwrap();
    store.set_title("Named After B").unwrap();
    let id = store.meta().id.clone();

    let (whole, _) = repo.fork(&id, usize::MAX).unwrap();
    assert_eq!(whole.meta().title.as_deref(), Some("Named After B"));
    let (earlier, _) = repo.fork(&id, 1).unwrap();
    assert_eq!(earlier.meta().title, None);

    store.set_title("").unwrap();
    let (cleared, _) = repo.fork(&id, usize::MAX).unwrap();
    assert_eq!(cleared.meta().title, None);
}

#[test]
fn fork_at_entry_resolves_title_path_scoped_not_whole_file_latest() {
    // `fork_at_entry_prefix` computes its own path (possibly off the active branch entirely, and
    // via a different resolution than `fork`'s `upto` count) — a rename recorded after the target
    // entry must not leak into a fork targeting an earlier point on the same chain.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("a");
    store.append_new(&session.messages).unwrap();
    let ids = store.active_ids().to_vec();

    session.user("b");
    store.append_new(&session.messages).unwrap();
    store.set_title("Renamed After B").unwrap(); // whole-file-latest, but only ever in effect after "b"

    let session_id = store.meta().id.clone();
    let (forked, _) = repo.fork_at_entry(&session_id, &ids[0], false).unwrap();
    assert_eq!(
        forked.meta().title,
        None,
        "forking at a point that predates every rename on its own path must not inherit one made \
         later on the same chain"
    );
}

#[test]
fn fork_carries_the_active_thinking_level_forward_and_it_survives_a_reopen() {
    // Fix 1 (pi-parity remediation, Round 2): unlike `model`/`title` just above (each already
    // fixed by `model_at_or_created`/`title_at_or_root`), none of `fork`/`fork_from_path`/
    // `fork_at_entry_prefix` ever resolved or persisted the source's active thinking level at all
    // — a fork silently dropped whatever reasoning-effort depth the source session had actually
    // settled on. The bug only shows up on a genuine reopen (a still-live in-memory store keeps
    // whatever the caller already had in scope) — exactly what a fresh process would do on `run
    // --continue`, a `serve` restart, or `switch_session` back to this session — so this reproduces
    // that: fork, then reopen the forked *file* from scratch via a brand-new `SessionStore::open`,
    // not the same `forked` handle `fork` already returned.
    // "b" is a *child* of the level change's anchor ("a"), so it's the point that actually
    // observes "high" — same "anchored-at, not before" contract
    // `fork_resolves_the_updated_model_via_the_tree_not_the_sessions_stale_meta` already exercises
    // for model: forking exactly at "a" (the change's own anchor) would still correctly see
    // whatever level was active *before* it, not "high" yet.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("a");
    store.append_new(&session.messages).unwrap();
    store.record_thinking_level_change("high").unwrap(); // anchored at "a"
    session.user("b"); // "a"'s child — observes the "high" change
    store.append_new(&session.messages).unwrap();

    let session_id = store.meta().id.clone();
    let (forked, _) = repo.fork(&session_id, usize::MAX).unwrap();
    assert_eq!(
        forked.meta().thinking_level.as_deref(),
        Some("high"),
        "a fork's own header must carry the thinking level active at the fork point"
    );
    let forked_path = forked.path().to_path_buf();
    drop(forked);

    let (reopened, _) = SessionStore::open(forked_path).unwrap();
    let tip = reopened
        .active_ids()
        .last()
        .cloned()
        .expect("forked session has at least one message");
    assert_eq!(
        reopened.thinking_level_at(&tip),
        Some("high"),
        "the active thinking level must survive a fresh reopen of the forked file, not just live \
         in the in-memory store `fork` already returned"
    );
}
