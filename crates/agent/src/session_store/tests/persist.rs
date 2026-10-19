use super::*;

#[test]
fn append_then_reopen_restores_transcript() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo
        .create(SessionMeta::new("/work", "claude-test"))
        .unwrap();
    let id = store.meta().id.clone();

    let mut session = Session::new();
    session.user("hello");
    store.append_new(&session.messages).unwrap();
    session.user("world");
    store.append_new(&session.messages).unwrap();

    let (_store, restored) = repo.open_id(&id).unwrap();
    assert_eq!(restored.messages.len(), 2);
    let dump = serde_json::to_string(restored.messages.as_ref()).unwrap();
    assert!(dump.contains("hello") && dump.contains("world"));
}

#[test]
#[cfg(unix)]
fn session_files_are_created_private_and_stay_private_through_rewrite() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo
        .create(SessionMeta::new("/work", "claude-test"))
        .unwrap();

    let mode_of = |p: &std::path::Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
    // Transcripts carry whatever `read` pulled off disk — never group/world-readable, on a
    // shared host in particular, regardless of the process umask.
    assert_eq!(
        mode_of(store.path()),
        0o600,
        "create() must set 0600 atomically"
    );

    let mut session = Session::new();
    session.user("a");
    session.user("b");
    store.append_new(&session.messages).unwrap();
    assert_eq!(
        mode_of(store.path()),
        0o600,
        "append must not loosen permissions"
    );

    store.rewrite(&[Message::user("summary")]).unwrap();
    assert_eq!(
        mode_of(store.path()),
        0o600,
        "rewrite's temp-file-then-rename must not loosen permissions"
    );
}

/// A session file that exists with a looser mode (an older version's, a restore's, a
/// `chmod`) is tightened to `0600` by the next append: the transcript, and the MCP task ids it
/// journals (possible bearer tokens), are never written into a group/world-readable file.
#[test]
#[cfg(unix)]
fn an_append_tightens_a_loosened_session_file_to_private() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tmpdir();
    let path = dir.path().join("s.jsonl");
    let mut store = SessionStore::create(path.clone(), SessionMeta::new("/w", "m")).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
    store
        .append_custom(
            "mcp_task",
            serde_json::json!({ "taskId": "SECRET-TASK-ID", "sessionId": "s" }),
        )
        .unwrap();
    assert_eq!(
        fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o600,
        "the append must tighten the file before writing a task id into it"
    );
}

#[test]
fn create_initializes_an_existing_empty_file_in_place_instead_of_failing() {
    // Track L8: a zero-byte file at the target path (e.g. `touch`'d ahead of time, or left over
    // from a crash before the header write landed) must not hard-fail `create` — it's
    // indistinguishable in intent from "not created yet."
    let dir = tmpdir();
    let path = dir.path().join("s.jsonl");
    std::fs::write(&path, b"").unwrap();
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        0,
        "must start empty"
    );

    let store = SessionStore::create(path.clone(), SessionMeta::new("/w", "m")).unwrap();
    assert_eq!(store.meta().cwd, "/w");

    // The header actually landed on disk, not just in memory.
    let (reopened, session) = SessionStore::open(path).unwrap();
    assert_eq!(reopened.meta().id, store.meta().id);
    assert_eq!(session.messages.len(), 0);
}

#[test]
fn create_still_refuses_a_genuinely_non_empty_existing_file() {
    // The empty-file allowance must not become a general "overwrite anything" escape hatch — a
    // file that already holds real (even non-session) content is never silently clobbered.
    let dir = tmpdir();
    let path = dir.path().join("s.jsonl");
    std::fs::write(&path, b"not a session file, but definitely not empty").unwrap();

    match SessionStore::create(path, SessionMeta::new("/w", "m")) {
        Err(e) => assert_eq!(e.kind(), std::io::ErrorKind::AlreadyExists),
        Ok(_) => panic!("must not silently clobber a non-empty existing file"),
    }
}

#[test]
fn create_fork_and_fork_at_entry_all_write_their_file_immediately_even_for_a_single_message() {
    // pi-parity (C-M4), pinning current behavior: pi's `createBranchedSession` defers the actual
    // file write until an assistant message lands (`session-manager/tree-traversal.test.ts:464-532`),
    // specifically to avoid littering the session directory with a fork abandoned after one user
    // message. This module deliberately does NOT port that deferral — see the "Tree-shaped history"
    // section of ARCHITECTURE.md for the full reasoning (short version: `rewrite` already writes the
    // whole prefix atomically in one temp-file-then-rename call, so there's no truncated-file risk
    // to guard against, and threading a "created but not yet flushed" state through `SessionStore`
    // isn't worth it for a cosmetic directory-listing concern). This test pins that decision: a
    // fresh `create`, and a `fork`/`fork_at_entry` of a single-user-message prefix, must all have a
    // real file on disk immediately, before any assistant message ever exists.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();

    let store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    assert!(
        store.path().exists() && std::fs::metadata(store.path()).unwrap().len() > 0,
        "create must write a real header immediately"
    );

    let mut src = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let src_id = src.meta().id.clone();
    let mut session = Session::new();
    session.user("only a user message, no assistant reply yet");
    src.append_new(&session.messages).unwrap();
    let user_id = src.active_ids()[0].clone();

    let (forked, fsession) = repo.fork(&src_id, usize::MAX).unwrap();
    assert_eq!(fsession.messages.len(), 1);
    assert!(
        forked.path().exists() && std::fs::metadata(forked.path()).unwrap().len() > 0,
        "fork of a single-user-message (no assistant reply) prefix must still write immediately"
    );

    let (forked_at_entry, fsession2) = repo.fork_at_entry(&src_id, &user_id, false).unwrap();
    assert_eq!(fsession2.messages.len(), 1);
    assert!(
        forked_at_entry.path().exists()
            && std::fs::metadata(forked_at_entry.path()).unwrap().len() > 0,
        "fork_at_entry of a single-user-message prefix must still write immediately"
    );
}

#[test]
fn append_new_batches_many_new_messages_in_one_call() {
    // Regression guard for the buffered-write fix: a multi-message batch (not the usual
    // one-or-two-at-a-time turn) must still land every line, in order, with the cursor advanced
    // exactly to the batch length — proving the single `write_all` path handles N > 1 correctly,
    // not just the common case.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo
        .create(SessionMeta::new("/work", "claude-test"))
        .unwrap();
    let id = store.meta().id.clone();

    let mut session = Session::new();
    for i in 0..50 {
        session.user(format!("message-{i}"));
    }
    store.append_new(&session.messages).unwrap();

    let (_store, restored) = repo.open_id(&id).unwrap();
    assert_eq!(restored.messages.len(), 50);
    for i in 0..50 {
        let dump = serde_json::to_string(&restored.messages[i]).unwrap();
        assert!(dump.contains(&format!("message-{i}")), "line {i}: {dump}");
    }

    // A follow-up call with a strict extension of the same slice only appends the new tail, not a
    // second copy of the first 50.
    session.user("message-50");
    store.append_new(&session.messages).unwrap();
    let (_store, restored) = repo.open_id(&id).unwrap();
    assert_eq!(restored.messages.len(), 51);
}

#[test]
fn read_capped_line_bounds_allocation_and_flags_oversized_lines() {
    use std::io::Cursor;

    // A line beyond the cap must not grow `buf` past it, must still land the reader on the next
    // line (not lose sync), and must be flagged so `SessionStore::open` can treat it like a torn
    // write — a corrupted length delimiter or concatenated lines shouldn't be able to make `open`
    // allocate without bound.
    let mut input = vec![b'x'; MAX_LINE_BYTES + 10];
    input.push(b'\n');
    input.extend_from_slice(b"next\n");
    let mut reader = BufReader::new(Cursor::new(input));

    let mut buf = Vec::new();
    let oversized = read_capped_line(&mut reader, &mut buf).unwrap();
    assert_eq!(oversized, Some(true));
    assert_eq!(buf.len(), MAX_LINE_BYTES, "buf must not grow past the cap");

    let oversized = read_capped_line(&mut reader, &mut buf).unwrap();
    assert_eq!(oversized, Some(false));
    assert_eq!(buf, b"next");

    assert_eq!(read_capped_line(&mut reader, &mut buf).unwrap(), None);
}

#[test]
fn oversized_line_is_recovered_like_a_torn_write() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo
        .create(SessionMeta::new("/work", "claude-test"))
        .unwrap();
    let id = store.meta().id.clone();
    let mut session = Session::new();
    session.user("first");
    store.append_new(&session.messages).unwrap();

    // A pathologically long line (well past MAX_LINE_BYTES) — the on-disk analogue of a
    // corrupted length delimiter or concatenated lines — must not be allocated in full; `open`
    // treats it as unreadable, the same as a torn write.
    let path = repo.find_path(&id).unwrap().unwrap();
    let mut f = OpenOptions::new().append(true).open(&path).unwrap();
    write!(f, "\"").unwrap();
    for _ in 0..(MAX_LINE_BYTES / 1024 + 10) {
        write!(f, "{}", "x".repeat(1024)).unwrap();
    }
    writeln!(f, "\"").unwrap();
    drop(f);

    let (_store, restored) = repo.open_id(&id).unwrap();
    // The intact first message survives; the oversized record is dropped.
    assert_eq!(restored.messages.len(), 1);
}

#[test]
fn torn_last_line_is_recovered() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo
        .create(SessionMeta::new("/work", "claude-test"))
        .unwrap();
    let id = store.meta().id.clone();
    let mut session = Session::new();
    session.user("first");
    store.append_new(&session.messages).unwrap();

    // Simulate a crash mid-append: a half-written, unterminated JSON line.
    let path = repo.find_path(&id).unwrap().unwrap();
    let mut f = OpenOptions::new().append(true).open(&path).unwrap();
    write!(f, "{{\"type\":\"message\",\"role\":\"user\",\"cont").unwrap();
    drop(f);

    let (_store, restored) = repo.open_id(&id).unwrap();
    // The intact first message survives; the torn record is dropped.
    assert_eq!(restored.messages.len(), 1);
}

#[test]
fn message_usage_round_trips_through_persist_and_reopen() {
    // Task #6 (pi-parity fix): Round 1 added `Message.usage` (`agent_core::TokenUsage`) — confirm
    // this persistence format actually carries it through a real append + reopen, not just relying
    // on `Message`'s `Serialize`/`Deserialize` derive being correct in isolation. No format/version
    // change was needed: `Entry::Message.message` embeds the whole `Message`, so the new
    // `#[serde(default, skip_serializing_if = "Option::is_none")]` field round-trips automatically.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo
        .create(SessionMeta::new("/work", "claude-test"))
        .unwrap();
    let usage = agent_core::TokenUsage {
        input_tokens: 120,
        output_tokens: 45,
        cache_read_tokens: 10,
        cache_write_tokens: 5,
        cache_write_1h_tokens: 0,
        reasoning_tokens: 7,
    };
    let mut session = Session::new();
    session.user("hi");
    session.push(
        Message::assistant(vec![ContentBlock::text("hello")])
            .with_model_id("claude-test")
            .with_usage(usage),
    );
    store.append_new(&session.messages).unwrap();

    let id = store.meta().id.clone();
    let (_reopened, restored) = repo.open_id(&id).unwrap();
    assert_eq!(restored.messages.len(), 2);
    assert_eq!(
        restored.messages[0].usage, None,
        "a user turn never carries usage"
    );
    assert_eq!(
        restored.messages[1].usage,
        Some(usage),
        "the assistant turn's own usage must survive a real append + reopen round trip"
    );
}

#[test]
fn message_usage_round_trips_when_absent_matching_a_pre_round_1_file() {
    // Backward compatibility: a session with no `usage` on any message (every message predates
    // Round 1's field, or was never populated) must still load cleanly — `#[serde(default)]` reads
    // it as `None` rather than failing to deserialize the line at all.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo
        .create(SessionMeta::new("/work", "claude-test"))
        .unwrap();
    let mut session = Session::new();
    session.user("hi");
    session.push(Message::assistant(vec![ContentBlock::text("hello")]));
    store.append_new(&session.messages).unwrap();

    let id = store.meta().id.clone();
    let (_reopened, restored) = repo.open_id(&id).unwrap();
    assert_eq!(restored.messages.len(), 2);
    assert!(restored.messages.iter().all(|m| m.usage.is_none()));
}

#[test]
fn open_restores_last_input_tokens_exactly_from_the_most_recent_messages_usage() {
    // Task #6 (pi-parity fix): when the active path's most recent usage-carrying message is
    // available, restoration must use its exact provider-reported figures — the same
    // `input + cache_read + cache_write` combination `Session::record_usage`'s own `live_input`
    // computes — rather than the coarser whole-transcript char/4 estimate.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo
        .create(SessionMeta::new("/work", "claude-test"))
        .unwrap();
    let usage = agent_core::TokenUsage {
        input_tokens: 1_000,
        output_tokens: 250,
        cache_read_tokens: 200,
        cache_write_tokens: 50,
        cache_write_1h_tokens: 0,
        reasoning_tokens: 0,
    };
    let mut session = Session::new();
    session.user("go");
    session.push(
        Message::assistant(vec![ContentBlock::text("ok")])
            .with_model_id("claude-test")
            .with_usage(usage),
    );
    store.append_new(&session.messages).unwrap();

    let id = store.meta().id.clone();
    let (_reopened, restored) = repo.open_id(&id).unwrap();
    assert_eq!(restored.last_input_tokens, 1_000 + 200 + 50);
    assert_eq!(restored.last_output_tokens, 250);
    // Positioned at the assistant message's own index (1), not `messages.len()` (2) — so
    // `compaction::trailing_tokens` treats that message itself as the "just completed turn" slot
    // (substituting `last_output_tokens` for it) rather than double-counting it as trailing.
    assert_eq!(restored.last_usage_message_count, 1);
    assert_eq!(
        agent_core::compaction::trailing_tokens(&restored, agent_core::TokenFamily::Claude),
        250,
        "the assistant message at the snapshot position must use last_output_tokens, not a fresh \
         char/4 estimate of its short reply text"
    );
}

#[test]
fn open_falls_back_to_the_char4_estimate_when_no_message_has_usage() {
    // Pre-Round-1 (or otherwise usage-less) sessions must keep the previous whole-transcript
    // estimate behavior — this is the fallback branch `open_restores_last_input_tokens_exactly_...`
    // above bypasses whenever real usage is available.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo
        .create(SessionMeta::new("/work", "claude-test"))
        .unwrap();
    let mut session = Session::new();
    session.user("x".repeat(400)); // ~100 estimated tokens (char/4)
    store.append_new(&session.messages).unwrap();

    let id = store.meta().id.clone();
    let (_reopened, restored) = repo.open_id(&id).unwrap();
    assert!(
        restored.last_input_tokens > 0,
        "must still estimate something so should_compact/is_hard_overflow can fire on a resumed \
         session that's already over threshold"
    );
    assert_eq!(
        restored.last_usage_message_count,
        restored.messages.len(),
        "with no real usage snapshot, every persisted message is treated as already accounted for"
    );
}

#[test]
fn oversized_line_mid_file_does_not_discard_subsequent_good_entries() {
    // The bug: an oversized line was treated like a torn *final* write (stop reading, keep what's
    // valid so far) even when it wasn't the last line — silently discarding every good entry after
    // it. It must instead be skipped like `corrupt_line_mid_file_does_not_discard_subsequent_good_entries`'s
    // unparseable-JSON case.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo
        .create(SessionMeta::new("/work", "claude-test"))
        .unwrap();
    let id = store.meta().id.clone();
    let mut session = Session::new();
    session.user("first");
    store.append_new(&session.messages).unwrap();

    let path = repo.find_path(&id).unwrap().unwrap();
    let mut f = OpenOptions::new().append(true).open(&path).unwrap();
    write!(f, "\"").unwrap();
    for _ in 0..(MAX_LINE_BYTES / 1024 + 10) {
        write!(f, "{}", "x".repeat(1024)).unwrap();
    }
    writeln!(f, "\"").unwrap();
    drop(f);

    session.user("third");
    store.append_new(&session.messages).unwrap();

    let (_store, restored) = repo.open_id(&id).unwrap();
    assert_eq!(
        restored.messages.len(),
        2,
        "both good entries must survive an oversized line between them, not just the ones before it"
    );
}

#[test]
fn invalid_utf8_line_mid_file_does_not_discard_subsequent_good_entries() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo
        .create(SessionMeta::new("/work", "claude-test"))
        .unwrap();
    let id = store.meta().id.clone();
    let mut session = Session::new();
    session.user("first");
    store.append_new(&session.messages).unwrap();

    // A line with invalid UTF-8 bytes (simulating bit rot) landing mid-file, not at the end.
    let path = repo.find_path(&id).unwrap().unwrap();
    let mut f = OpenOptions::new().append(true).open(&path).unwrap();
    f.write_all(b"\xff\xfe not valid utf-8\n").unwrap();
    drop(f);

    session.user("third");
    store.append_new(&session.messages).unwrap();

    let (_store, restored) = repo.open_id(&id).unwrap();
    assert_eq!(
        restored.messages.len(),
        2,
        "both good entries must survive an invalid-UTF-8 line between them, not just the ones before it"
    );
}

#[test]
fn corrupt_line_mid_file_does_not_discard_subsequent_good_entries() {
    // A bad line anywhere in the file — not only a torn final write — must be skipped, not treated
    // as "nothing valid follows a half-written record": the good entries appended *after* it must
    // still be recovered on open.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo
        .create(SessionMeta::new("/work", "claude-test"))
        .unwrap();
    let id = store.meta().id.clone();
    let mut session = Session::new();
    session.user("first");
    store.append_new(&session.messages).unwrap();

    // A complete (newline-terminated), well-formed JSON line that isn't a valid `Entry` — not a
    // torn write, just corrupted/foreign content landing mid-file (disk bit rot, a manual edit).
    let path = repo.find_path(&id).unwrap().unwrap();
    let mut f = OpenOptions::new().append(true).open(&path).unwrap();
    writeln!(f, r#"{{"not":"a valid entry"}}"#).unwrap();
    drop(f);

    // Appended using the same in-memory `store` (its `active`/`persisted` state is untouched by
    // the raw write above), so this chains correctly off "first" regardless of the bad line
    // sitting between them on disk.
    session.user("third");
    store.append_new(&session.messages).unwrap();

    let (_store, restored) = repo.open_id(&id).unwrap();
    assert_eq!(
        restored.messages.len(),
        2,
        "both good entries must survive a bad line between them, not just the ones before it"
    );
}

#[test]
fn open_recovers_a_message_line_hand_edited_to_have_null_content_instead_of_dropping_it() {
    // Pass 20 (pi-parity fix): `Message::content` (`Vec<ContentBlock>`) has no `#[serde(default)]`
    // — deliberately, a live in-process write should never omit it — so a line hand-edited (or
    // externally corrupted) down to `"content":null` used to fail `Entry` deserialization entirely
    // and fall into the same generic "skip this whole line" recovery as genuine garbage, silently
    // losing the message's role/id/tree position along with its content. `parse_entry_lenient`
    // recovers this one specific shape by treating the content as empty instead, matching pi's own
    // equivalent fix, rather than dropping the entry outright.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo
        .create(SessionMeta::new("/work", "claude-test"))
        .unwrap();
    let id = store.meta().id.clone();
    let mut session = Session::new();
    session.user("first");
    store.append_new(&session.messages).unwrap();
    let first_id = store.active_ids().last().unwrap().clone();

    // A hand-edited message line: well-formed JSON, a real `"type":"message"` entry, but its
    // `content` was blanked out to `null` (as if someone had redacted it in a text editor) — chained
    // off "first" so replay's tip lands on it as the new last message.
    let path = repo.find_path(&id).unwrap().unwrap();
    let mut f = OpenOptions::new().append(true).open(&path).unwrap();
    writeln!(
        f,
        r#"{{"type":"message","id":"hand-edited","parent_id":"{first_id}","timestamp":0,"role":"user","content":null}}"#
    )
    .unwrap();
    drop(f);

    let (_store, restored) = repo.open_id(&id).unwrap();
    assert_eq!(
        restored.messages.len(),
        2,
        "the hand-edited line must be recovered as a message, not dropped entirely: {:?}",
        restored.messages
    );
    let recovered = &restored.messages[1];
    assert_eq!(recovered.role, Role::User, "role must survive the recovery");
    assert!(
        recovered.content.is_empty(),
        "content must recover as empty, not cause the whole entry to be lost: {:?}",
        recovered.content
    );
}

#[test]
fn open_still_drops_a_message_line_whose_content_is_some_other_invalid_shape() {
    // The recovery `open_recovers_a_message_line_hand_edited_to_have_null_content_instead_of_dropping_it`
    // proves is narrowly scoped to `content` being absent or `null` specifically — a `content` field
    // that's present but some *other* invalid shape (here, a bare string instead of an array of
    // content blocks) is a different, genuine corruption, not this one known-recoverable case, and
    // must still fall through to the ordinary skip-this-line recovery.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo
        .create(SessionMeta::new("/work", "claude-test"))
        .unwrap();
    let id = store.meta().id.clone();
    let mut session = Session::new();
    session.user("first");
    store.append_new(&session.messages).unwrap();
    let first_id = store.active_ids().last().unwrap().clone();

    let path = repo.find_path(&id).unwrap().unwrap();
    let mut f = OpenOptions::new().append(true).open(&path).unwrap();
    writeln!(
        f,
        r#"{{"type":"message","id":"bad-content-shape","parent_id":"{first_id}","timestamp":0,"role":"user","content":"not an array"}}"#
    )
    .unwrap();
    drop(f);

    let (_store, restored) = repo.open_id(&id).unwrap();
    assert_eq!(
        restored.messages.len(),
        1,
        "a content field that's present but the wrong shape (not null/missing) must still be \
         dropped as genuinely unparseable, not recovered: {:?}",
        restored.messages
    );
}

#[test]
fn read_listing_survives_an_oversized_invalid_utf8_and_corrupt_line_all_mid_file() {
    // pi-parity gap (fixed, L4): `read_listing` shares `SessionStore::open`'s exact skip-and-
    // continue recovery logic (same `read_capped_line` primitive, same three corruption cases),
    // but only `open`'s recovery had a dedicated test — this one drives all three corruption
    // shapes through the *listing* scan specifically, proving `message_count`/`preview`/
    // `preview` all still reflect every good entry, not just the ones before the first bad
    // line.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo
        .create(SessionMeta::new("/work", "claude-test"))
        .unwrap();
    let id = store.meta().id.clone();
    let mut session = Session::new();
    session.user("first good message");
    store.append_new(&session.messages).unwrap();

    let path = repo.find_path(&id).unwrap().unwrap();
    let mut f = OpenOptions::new().append(true).open(&path).unwrap();
    // Oversized line.
    write!(f, "\"").unwrap();
    for _ in 0..(MAX_LINE_BYTES / 1024 + 10) {
        write!(f, "{}", "x".repeat(1024)).unwrap();
    }
    writeln!(f, "\"").unwrap();
    // Invalid UTF-8 line.
    f.write_all(b"\xff\xfe not valid utf-8\n").unwrap();
    // Well-formed JSON that isn't a valid `Entry`.
    writeln!(f, r#"{{"not":"a valid entry"}}"#).unwrap();
    drop(f);

    session.user("second good message");
    store.append_new(&session.messages).unwrap();

    let listed = read_listing(&path, None, &Layout::File).unwrap();
    assert_eq!(
        listed.message_count, 2,
        "both good messages must be counted, not just the one before the corruption"
    );
}

#[test]
fn rewrite_replaces_whole_transcript() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo
        .create(SessionMeta::new("/work", "claude-test"))
        .unwrap();
    let id = store.meta().id.clone();
    let mut session = Session::new();
    session.user("a");
    session.user("b");
    session.user("c");
    store.append_new(&session.messages).unwrap();

    // Compaction-style rewrite to a shorter list.
    let compacted = vec![Message::user("summary")];
    store.rewrite(&compacted).unwrap();

    let (_store, restored) = repo.open_id(&id).unwrap();
    assert_eq!(restored.messages.len(), 1);
    let dump = serde_json::to_string(restored.messages.as_ref()).unwrap();
    assert!(dump.contains("summary") && !dump.contains("\"a\""));
}

#[test]
fn rewrite_cleans_up_the_temp_file_when_the_final_rename_fails() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("hello");
    store.append_new(&session.messages).unwrap();

    // Replace the session file with an (empty) directory of the same name: `create_private(&tmp)`
    // still succeeds (it's a different filename), but the final `fs::rename(&tmp, &self.path)`
    // must fail — a file can never be renamed onto an existing directory. A genuine in-process
    // error, not a crash, so `rewrite` gets the chance to clean up after itself.
    let path = store.path().to_path_buf();
    fs::remove_file(&path).unwrap();
    fs::create_dir(&path).unwrap();

    let tmp = path.with_extension("jsonl.tmp");
    let err = store.rewrite(&[]).unwrap_err();
    assert!(
        !tmp.exists(),
        "the orphaned temp file must be removed after a failed rewrite: {err}"
    );
}

#[test]
fn reset_for_new_session_clears_model_thinking_label_and_event_state_a_plain_rewrite_leaves_behind()
{
    // Pass 20 (pi-parity fix): single-file mode's `/new`-equivalent (`Persistence::new_session`,
    // `serve/`) resets an *already-populated* `SessionStore` in place (there's no fresh store to
    // swap in the way repo mode's `SessionRepo::create` gives it) — it used to call plain
    // `rewrite(&[])`, which only ever clears the message tree itself, never the id-keyed side
    // indexes this test drives into every one of. `reset_for_new_session` must clear all of them.
    let dir = tmpdir();
    let path = dir.path().join("s.jsonl");
    let mut store = SessionStore::create(path, SessionMeta::new("/work", "model-a")).unwrap();

    // A model/thinking-level change recorded before any message exists anchors at the tree root
    // (`self.active.last()` is `None`) — exactly the `switch_branch{before: true}`-reachable case
    // the audit named.
    store.record_model_change("model-b").unwrap();
    store.record_thinking_level_change("high").unwrap();

    let mut session = Session::new();
    session.user("hello");
    store.append_new(&session.messages).unwrap();
    let msg_id = store.active_ids().last().unwrap().clone();
    store.set_label(&msg_id, Some("bookmark")).unwrap();
    store
        .append_custom("note", serde_json::json!({"k": "v"}))
        .unwrap();

    // Sanity check: every one of these is actually populated before the reset, or the assertions
    // below would pass vacuously.
    assert_eq!(store.model_at_root(), Some("model-b"));
    assert_eq!(store.thinking_level_at_root(), Some("high"));
    assert_eq!(store.get_label(&msg_id), Some("bookmark"));
    assert!(!store.export_events().is_empty());

    store.reset_for_new_session().unwrap();

    assert_eq!(
        store.model_at_root(),
        None,
        "a stale model_changes entry at root must not leak into the reset session"
    );
    assert_eq!(
        store.thinking_level_at_root(),
        None,
        "a stale level_changes entry at root must not leak into the reset session"
    );
    assert_eq!(
        store.get_label(&msg_id),
        None,
        "a label on a message from the discarded session must not survive the reset"
    );
    assert!(
        store.export_events().is_empty(),
        "export_events must not replay the discarded session's ModelChange/ThinkingLevelChange/\
         Label/Custom history: {:?}",
        store.export_events()
    );
}

#[test]
fn reset_for_new_session_does_not_affect_a_narrow_rewrites_side_indexes() {
    // The flip side of the test just above: `rewrite` itself (used by `fork`/`fork_at_entry`/
    // `fork_from_path`'s already-fresh stores, and `rewrite_compacted`'s degenerate same-length
    // fallback on *this* store) must keep its narrower, unchanged behavior — this pass's fix must
    // not leak into every other `rewrite` caller.
    let dir = tmpdir();
    let path = dir.path().join("s.jsonl");
    let mut store = SessionStore::create(path, SessionMeta::new("/work", "model-a")).unwrap();
    store.record_model_change("model-b").unwrap();

    let mut session = Session::new();
    session.user("hello");
    store.append_new(&session.messages).unwrap();
    let msg_id = store.active_ids().last().unwrap().clone();
    store.set_label(&msg_id, Some("bookmark")).unwrap();

    // A same-length rewrite (mirrors `rewrite_compacted`'s degenerate fallback) on the same store.
    store.rewrite(&session.messages).unwrap();

    assert_eq!(
        store.model_at_root(),
        Some("model-b"),
        "a plain `rewrite` must not clear model_changes — only `reset_for_new_session` does"
    );
    assert_eq!(
        store.get_label(&msg_id),
        Some("bookmark"),
        "a plain `rewrite` must not clear labels — only `reset_for_new_session` does"
    );
}

#[test]
fn record_edits_persists_in_place_rewrites_idempotently_and_survives_a_compaction() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut messages = vec![
        Message::user("task"),
        Message::assistant(vec![agent_core::ContentBlock::tool_use(
            "c0",
            "read",
            json!({"path": "a"}),
        )]),
        Message::tool_result("c0", "x".repeat(2_000), false),
        Message::assistant(vec![agent_core::ContentBlock::text("ok")]),
    ];
    store.append_new(&messages).unwrap();
    // A custom entry on the active path holds no message slot: indices still count messages.
    store.append_custom("note", json!({})).unwrap();
    messages[2] = Message::tool_result("c0", "[masked]", false);
    // Index 9 is past the list, 4 past what is stored: both skipped.
    store.record_edits(&messages, &[2, 4, 9]).unwrap();
    let len = fs::metadata(store.path()).unwrap().len();
    store.record_edits(&messages, &[2]).unwrap();
    assert_eq!(
        fs::metadata(store.path()).unwrap().len(),
        len,
        "an edit already stored writes nothing"
    );

    let (mut reopened, restored) = repo.open_id(&store.meta().id.clone()).unwrap();
    assert_eq!(*restored.messages, messages, "the edit reloads");
    // The original output is still on disk, under the edit.
    assert!(
        fs::read_to_string(reopened.path())
            .unwrap()
            .contains(&"x".repeat(2_000))
    );

    // A compaction after the edit carries the edited content into the new active path.
    let compacted = vec![
        Message::user(format!(
            "{}\n\nrecap",
            agent_core::compaction::SUMMARY_MARKER
        )),
        messages[1].clone(),
        messages[2].clone(),
        messages[3].clone(),
    ];
    reopened.append_new(&messages).unwrap();
    reopened
        .rewrite_compacted(
            &compacted,
            CompactionMeta {
                tokens_before: 1,
                provenance: Default::default(),
            },
        )
        .unwrap();
    let (_, restored) = repo.open_id(&store.meta().id.clone()).unwrap();
    assert_eq!(*restored.messages, compacted);
}

#[test]
fn record_edits_never_persists_a_cross_model_scrub() {
    // A GPT session resumed on Claude: the scrub truncates `call|item` ids in memory only. A
    // masking pass then rewrites a result; the edit must carry the masked content under the
    // *stored* id, or the reloaded transcript pairs `call_1` against `call_1|fc_1` and the original
    // model rejects it.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "gpt-5.5")).unwrap();
    let mut session = Session::new();
    session.user("task");
    session.push(
        Message::assistant(vec![agent_core::ContentBlock::tool_use(
            "call_1|fc_1",
            "read",
            json!({"path": "a"}),
        )])
        .with_model_id("gpt-5.5"),
    );
    session.push(Message::tool_result(
        "call_1|fc_1",
        "x".repeat(2_000),
        false,
    ));
    session.push(
        Message::assistant(vec![agent_core::ContentBlock::text("ok")]).with_model_id("gpt-5.5"),
    );
    store.append_new(&session.messages).unwrap();

    session.scrub_cross_model_state("claude-opus-5-5");
    let plan = agent_core::compaction::mask_tool_results(
        &session.messages,
        &agent_core::token_size::message_sizes(
            &session.messages,
            agent_core::TokenFamily::Other,
            None,
        ),
        agent_core::TokenFamily::Other,
        1,
        u32::MAX,
        &[],
    )
    .unwrap();
    assert!(
        format!("{:?}", plan.messages[2]).contains("\"call_1\""),
        "scrubbed in memory"
    );
    store.record_edits(&plan.messages, &plan.edited).unwrap();

    let (_, restored) = repo.open_id(&store.meta().id.clone()).unwrap();
    let agent_core::ContentBlock::ToolResult {
        tool_use_id,
        content,
        ..
    } = &restored.messages[2].content[0]
    else {
        panic!("{:?}", restored.messages[2]);
    };
    assert_eq!(
        tool_use_id, "call_1|fc_1",
        "the stored id, not the scrubbed one"
    );
    assert!(content.starts_with(agent_core::compaction::MASKED_PREFIX));
    // And it is idempotent against the scrubbed list too.
    let len = fs::metadata(store.path()).unwrap().len();
    store.record_edits(&plan.messages, &plan.edited).unwrap();
    assert_eq!(fs::metadata(store.path()).unwrap().len(), len);
}
