use super::*;

#[test]
fn newer_version_is_rejected() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let id = store.meta().id.clone();
    let path = repo.find_path(&id).unwrap().unwrap();

    // Rewrite the header with a version newer than this build understands.
    let content = fs::read_to_string(&path).unwrap();
    let mut lines = content.lines();
    let mut header: Value = serde_json::from_str(lines.next().unwrap()).unwrap();
    header["version"] = serde_json::json!(999);
    let rest: Vec<&str> = lines.collect();
    fs::write(&path, format!("{}\n{}", header, rest.join("\n"))).unwrap();

    // Opening errors clearly rather than mis-parsing... (`map` drops the non-`Debug` success value
    // so `unwrap_err` can format it).
    let err = repo.open_id(&id).map(|_| ()).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    // ...and the listing skips the unreadable file instead of including it.
    assert!(repo.list().unwrap().is_empty());
}

#[test]
fn pre_versioning_header_with_no_version_field_opens_as_v1() {
    // A header written before `version` existed omits the field entirely (not `version: 0` — the
    // key is just absent, and `#[serde(default)]` fills it with `0` on read). `migrate`'s `0 |
    // VERSION => Ok(meta)` arm treats that as wire-compatible with v1; this is the one arm of the
    // match `newer_version_is_rejected` doesn't exercise.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let id = store.meta().id.clone();
    let path = repo.find_path(&id).unwrap().unwrap();

    // Rewrite the header with `version` removed entirely, simulating a file from before the field
    // existed.
    let content = fs::read_to_string(&path).unwrap();
    let mut lines = content.lines();
    let mut header: Value = serde_json::from_str(lines.next().unwrap()).unwrap();
    header.as_object_mut().unwrap().remove("version");
    let rest: Vec<&str> = lines.collect();
    fs::write(&path, format!("{}\n{}", header, rest.join("\n"))).unwrap();

    // Opens cleanly — no migration error — and the listing includes it like any other session.
    let (_store, _session) = repo.open_id(&id).unwrap();
    assert_eq!(repo.list().unwrap().len(), 1);
}

#[test]
fn writes_round_trip_after_fsync() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let id = store.meta().id.clone();
    let mut session = Session::new();
    session.user("durable");
    // Exercises the synced append path...
    store.append_new(&session.messages).unwrap();
    // ...and the synced atomic-rewrite path.
    store.rewrite(&session.messages).unwrap();

    let (_store, restored) = repo.open_id(&id).unwrap();
    assert_eq!(restored.messages.len(), 1);
    let dump = serde_json::to_string(restored.messages.as_ref()).unwrap();
    assert!(dump.contains("durable"));
}

#[test]
fn open_restores_a_proactive_compaction_signal_from_the_persisted_transcript() {
    // B-M14 pi-parity gap (fixed): `SessionStore::open` never restored `last_input_tokens` from
    // the persisted history — a freshly-built `Session` defaults it to 0, and `should_compact`/
    // `is_hard_overflow` both require it to be positive to fire at all. A resumed large session
    // wouldn't proactively compact until a *new* turn produced fresh real usage, a whole turn
    // later than it should (pi's own `pre-prompt-compaction-no-continue` regression test covers
    // the same gap). Usage isn't persisted per-message anywhere in this format, so the estimate
    // must come from the same char/4 heuristic `compaction::trailing_tokens` already uses
    // elsewhere for spans with no exact provider-reported figure.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let id = store.meta().id.clone();
    let mut session = Session::new();
    // Real, non-trivial text so the char/4 estimate is comfortably nonzero and easy to bound.
    session.user("a".repeat(400)); // ~100 estimated tokens
    session.push(Message::assistant(vec![ContentBlock::text(
        "b".repeat(400),
    )])); // ~100 more
    store.append_new(&session.messages).unwrap();

    let (_reopened, restored) = repo.open_id(&id).unwrap();
    assert!(
        restored.last_input_tokens > 0,
        "a resumed session must have a nonzero live-context estimate, not the zero default"
    );
    assert!(
        restored.last_input_tokens >= 190,
        "expected roughly 200 estimated tokens across both messages, got {}",
        restored.last_input_tokens
    );
    // Every persisted message must be marked as already accounted for, so a subsequent
    // `trailing_tokens` call (after a fresh turn) doesn't double-count this estimate.
    assert_eq!(restored.last_usage_message_count, restored.messages.len());
}

#[test]
fn compaction_records_provenance() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let id = store.meta().id.clone();
    let mut session = Session::new();
    session.user("a");
    session.user("b");
    session.user("c");
    store.append_new(&session.messages).unwrap();

    // Compact 3 messages down to 1 — two dropped.
    store.rewrite(&[Message::user("summary")]).unwrap();
    assert_eq!(store.meta().compactions, 1);
    assert_eq!(store.meta().dropped_messages, 2);

    // Provenance is persisted in the header and survives reopen.
    let (reopened, _) = repo.open_id(&id).unwrap();
    assert_eq!(reopened.meta().compactions, 1);
    assert_eq!(reopened.meta().dropped_messages, 2);
}

#[test]
fn legacy_flat_file_migrates_ids_in_memory_only() {
    // Hand-write a pre-tree file: `Message` lines with no `id`/`parent_id` fields at all — the
    // exact shape a file predating this module's tree support would have.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let meta = SessionMeta::new("/w", "m");
    let id = meta.id.clone();
    let path = repo.path_for(&meta);
    let header = serde_json::to_string(&Entry::Session(meta)).unwrap();
    let raw = format!(
        "{header}\n\
         {{\"type\":\"message\",\"role\":\"user\",\"content\":[{{\"type\":\"text\",\"text\":\"first\"}}]}}\n\
         {{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{{\"type\":\"text\",\"text\":\"second\"}}]}}\n"
    );
    fs::write(&path, raw).unwrap();

    let (store, session) = repo.open_id(&id).unwrap();
    // Reconstructs the same linear order a flat file always had.
    assert_eq!(session.messages.len(), 2);
    assert_eq!(store.active_ids().len(), 2);
    // Synthesized ids are distinct and chained (not both defaulting to the same value).
    assert_ne!(store.active_ids()[0], store.active_ids()[1]);

    // Never persisted back: re-reading the raw file, the two *message* lines still carry no `id`
    // field (the header legitimately has one — `SessionMeta.id` — so check those lines only).
    let raw_after = fs::read_to_string(&path).unwrap();
    for line in raw_after.lines().skip(1) {
        assert!(
            !line.contains("\"id\""),
            "message line gained a persisted id: {line}"
        );
    }
}
