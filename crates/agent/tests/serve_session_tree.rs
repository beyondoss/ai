//! `serve` e2e: The session tree: branches, forks, clone, delete, `get_tree`/`get_messages` by id.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

use common::{
    message_ids, read_until_response, serve_cmd, serve_dir_cmd, spawn_model_server,
    spawn_model_server_with_stalled_response, turn_text,
};
use serde_json::{Value, json};

#[test]
fn serve_get_messages_since_returns_only_what_was_appended_after_a_known_id() {
    // Track M21: a client that already has messages through some tree id shouldn't have to
    // re-transfer the whole transcript just to see what's new — pi's own `get_entries({since})`.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) =
        spawn_model_server(vec![turn_text("first answer"), turn_text("second answer")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "first" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "second" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let all = frames.last().unwrap()["data"]["messages"]
        .as_array()
        .unwrap()
        .clone();
    assert_eq!(
        all.len(),
        4,
        "expected [first, first answer, second, second answer]: {all:#?}"
    );
    let first_answer_id = all[1]["id"].as_str().unwrap().to_string();

    // Only what's new since the first turn's assistant reply: the second turn's user + assistant.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "get_messages", "since": first_answer_id })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], true, "{resp:#?}");
    let since_messages = resp["data"]["messages"].as_array().unwrap();
    assert_eq!(since_messages.len(), 2, "{since_messages:#?}");
    assert!(since_messages[0]["content"].to_string().contains("second"));
    assert!(
        since_messages[1]["content"]
            .to_string()
            .contains("second answer")
    );

    // An unknown `since` id is an error, not a silent full re-fetch.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "get_messages", "since": "does-not-exist" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], false, "{resp:#?}");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_set_label_and_get_label_round_trip_over_the_wire() {
    // Pi-parity audit H3: `SessionStore::set_label`/`get_label` were fully built, persisted, and
    // carried across forks, but had no RPC command at all — this proves the actual wire commands,
    // not just the underlying `SessionStore` method.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![turn_text("hi there")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let messages = frames.last().unwrap()["data"]["messages"]
        .as_array()
        .unwrap()
        .clone();
    let target_id = messages[0]["id"].as_str().unwrap().to_string();

    // No label yet.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "get_label", "target_id": target_id })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_label");
    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], true, "{resp:#?}");
    assert_eq!(resp["data"]["label"], Value::Null, "{resp:#?}");

    // Set it.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_label", "target_id": target_id, "label": "checkpoint" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_label");
    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], true, "{resp:#?}");
    assert_eq!(resp["data"]["label"], "checkpoint", "{resp:#?}");

    // Read it back.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "get_label", "target_id": target_id })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_label");
    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], true, "{resp:#?}");
    assert_eq!(resp["data"]["label"], "checkpoint", "{resp:#?}");

    // Clear it (explicit `null`, not a missing key).
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_label", "target_id": target_id, "label": Value::Null })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_label");
    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], true, "{resp:#?}");
    assert_eq!(resp["data"]["label"], Value::Null, "{resp:#?}");

    writeln!(
        stdin,
        "{}",
        json!({ "type": "get_label", "target_id": target_id })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_label");
    let resp = frames.last().unwrap();
    assert_eq!(resp["data"]["label"], Value::Null, "{resp:#?}");

    // An unknown target_id is a clear error, not a silent no-op.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_label", "target_id": "does-not-exist", "label": "x" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_label");
    assert_eq!(frames.last().unwrap()["success"], false);

    // A missing `label` key (as opposed to an explicit `null`) is an error too — matches
    // `set_thinking`'s own "present-but-null clears, missing is a mistake" contract.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_label", "target_id": target_id })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_label");
    assert_eq!(frames.last().unwrap()["success"], false);

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_append_custom_reaches_the_tree_but_not_the_active_messages() {
    // Pi-parity audit: `SessionStore::append_custom` was fully built and tested (custom tree entries
    // that occupy a real slot in the tree without contributing to `Session.messages`/LLM context) but
    // had no RPC command at all — this proves the actual wire command, not just the underlying
    // `SessionStore` method.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![turn_text("hi there")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    writeln!(
        stdin,
        "{}",
        json!({ "type": "append_custom", "kind": "checkpoint-marker", "data": {"foo": "bar"} })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "append_custom");
    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], true, "{resp:#?}");
    let custom_id = resp["data"]["id"]
        .as_str()
        .expect("append_custom must return the new entry's id")
        .to_string();

    // It occupies a real slot in the tree...
    writeln!(stdin, "{}", json!({ "type": "get_tree" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_tree");
    let nodes = frames.last().unwrap()["data"]["nodes"].as_array().unwrap();
    let custom_node = nodes
        .iter()
        .find(|n| n["id"] == custom_id)
        .unwrap_or_else(|| panic!("custom entry must appear in get_tree: {nodes:#?}"));
    assert_eq!(custom_node["preview"], "[custom: checkpoint-marker]");
    assert_eq!(
        custom_node["role"],
        Value::Null,
        "a custom entry has no role of its own"
    );

    // ...but contributes nothing to the materialized message list.
    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let messages = frames.last().unwrap()["data"]["messages"]
        .as_array()
        .unwrap();
    assert!(
        messages.iter().all(|m| m["id"] != custom_id),
        "a custom entry must not appear in get_messages: {messages:#?}"
    );

    // `data` defaults to `{}` when omitted.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "append_custom", "kind": "bare" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "append_custom");
    assert_eq!(frames.last().unwrap()["success"], true);

    // A missing `kind` is a clear error, not a silent no-op.
    writeln!(stdin, "{}", json!({ "type": "append_custom" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "append_custom");
    assert_eq!(frames.last().unwrap()["success"], false);

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_repo_lists_switches_and_forks_sessions() {
    let dir = tempfile::tempdir().unwrap();
    let session_dir = dir.path().join("sessions").to_string_lossy().into_owned();

    // Two text turns: one per prompt.
    let (base, _bodies) = spawn_model_server(vec![turn_text("first answer"), turn_text("second")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_dir_cmd(bin, &base, &session_dir).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // Read the `ready` banner to learn the first session's id.
    let mut ready = String::new();
    stdout.read_line(&mut ready).unwrap();
    let ready: Value = serde_json::from_str(ready.trim()).unwrap();
    let first_id = ready["session_id"].as_str().unwrap().to_string();

    // Prompt in session 1.
    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    // Start a second session.
    writeln!(stdin, "{}", json!({ "type": "new_session", "id": "n" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "new_session");
    let new_session_data = &frames.last().unwrap()["data"];
    let second_id = new_session_data["session_id"].as_str().unwrap().to_string();
    assert_ne!(first_id, second_id, "new_session must mint a new id");
    assert_eq!(
        new_session_data["parent"], first_id,
        "the fresh session's lineage marker must point back at whatever was active before it: \
         {new_session_data:#?}"
    );

    // Prompt in session 2.
    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "yo" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    // List shows both, newest first.
    writeln!(stdin, "{}", json!({ "type": "list_sessions" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "list_sessions");
    let sessions = frames.last().unwrap()["data"]["sessions"]
        .as_array()
        .unwrap();
    assert!(
        sessions.len() >= 2,
        "both sessions should be listed: {sessions:#?}"
    );
    // Derived listing fields (`preview`/`message_count`/`updated_at`/`search_text`) live behind
    // `#[serde(skip)]` on `SessionMeta` so they never leak into the on-disk header — `list_sessions`
    // must still surface them to the client via `SessionMeta::to_listing_json`.
    let first_session = &sessions[0];
    assert!(
        first_session["message_count"].as_u64().unwrap() > 0,
        "message_count must be populated: {first_session:#?}"
    );
    assert!(
        first_session["updated_at"].as_u64().unwrap() > 0,
        "updated_at must be populated: {first_session:#?}"
    );
    assert!(
        first_session["preview"].is_string(),
        "preview must be populated: {first_session:#?}"
    );
    assert!(
        first_session["search_text"].is_string(),
        "search_text must be populated: {first_session:#?}"
    );
    // The lineage marker also persists to disk and survives into `list_sessions`, not just the
    // `new_session` response.
    let second_session = sessions
        .iter()
        .find(|s| s["id"] == second_id)
        .expect("session 2 must be listed");
    assert_eq!(
        second_session["parent"], first_id,
        "list_sessions must surface the persisted lineage marker too: {second_session:#?}"
    );

    // Switch back to session 1 and confirm its transcript is restored.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "switch_session", "session_id": first_id })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "switch_session");
    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let dump = frames.last().unwrap()["data"]["messages"].to_string();
    assert!(
        dump.contains("first answer"),
        "switched-to session must restore its transcript: {dump}"
    );

    // `preview_fork` previews the same prefix `fork` would copy, without creating anything: the
    // session count in `list_sessions` must be unchanged afterward.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "preview_fork", "session_id": first_id, "upto": 1 })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "preview_fork");
    let preview = frames.last().unwrap();
    assert_eq!(preview["success"], true);
    let preview_messages = preview["data"]["messages"].as_array().unwrap();
    assert_eq!(
        preview_messages.len(),
        1,
        "upto:1 previews just the first message: {preview_messages:#?}"
    );

    writeln!(stdin, "{}", json!({ "type": "list_sessions" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "list_sessions");
    let count_before_fork = frames.last().unwrap()["data"]["sessions"]
        .as_array()
        .unwrap()
        .len();

    // Fork the current session; the fork gets a new id.
    writeln!(stdin, "{}", json!({ "type": "fork" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "fork");
    let fork_id = frames.last().unwrap()["data"]["session_id"]
        .as_str()
        .unwrap();
    assert_ne!(fork_id, first_id, "a fork is a distinct session");

    // The preview above must not have created a session of its own — only the real `fork` above added
    // exactly one to the count taken right before it.
    writeln!(stdin, "{}", json!({ "type": "list_sessions" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "list_sessions");
    let count_after_fork = frames.last().unwrap()["data"]["sessions"]
        .as_array()
        .unwrap()
        .len();
    assert_eq!(
        count_after_fork,
        count_before_fork + 1,
        "preview_fork must not itself have created a session"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_clone_forks_the_current_session_at_its_current_tip() {
    // Track L17: pi's own `clone` command — a thin, argument-free alias over `fork` at the session's
    // current tip. This crate's `fork` already defaults to exactly that when called with no
    // `upto`/`target_id`, so `clone` must behave identically to a bare `fork`, just under pi's name.
    let dir = tempfile::tempdir().unwrap();
    let session_dir = dir.path().join("sessions").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![turn_text("first answer")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_dir_cmd(bin, &base, &session_dir).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    let mut ready = String::new();
    stdout.read_line(&mut ready).unwrap();
    let ready: Value = serde_json::from_str(ready.trim()).unwrap();
    let source_id = ready["session_id"].as_str().unwrap().to_string();

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    writeln!(stdin, "{}", json!({ "type": "clone" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "clone");
    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], true, "frames: {frames:#?}");
    let clone_id = resp["data"]["session_id"].as_str().unwrap().to_string();
    assert_ne!(clone_id, source_id, "a clone is a distinct session");

    // The clone must carry the source's full transcript (the same active-path-at-current-tip a bare
    // `fork` would copy) — not an empty session.
    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let dump = frames.last().unwrap()["data"]["messages"].to_string();
    assert!(
        dump.contains("first answer") && dump.contains("\"hi\""),
        "clone must carry the source session's full transcript: {dump}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_new_session_parent_session_overrides_the_default_lineage() {
    // Track L17: `new_session`'s `parent` previously always pointed at whatever session was active
    // immediately before the call, with no way to link a fresh session to a *different* one instead.
    let dir = tempfile::tempdir().unwrap();
    let session_dir = dir.path().join("sessions").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_dir_cmd(bin, &base, &session_dir).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    let mut ready = String::new();
    stdout.read_line(&mut ready).unwrap();
    let ready: Value = serde_json::from_str(ready.trim()).unwrap();
    let active_id = ready["session_id"].as_str().unwrap().to_string();

    // A `new_session` with no override still links to whatever was active — the pre-existing default,
    // unchanged by adding the override.
    writeln!(stdin, "{}", json!({ "type": "new_session" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "new_session");
    let default_data = &frames.last().unwrap()["data"];
    assert_eq!(
        default_data["parent"], active_id,
        "omitting parent_session must keep linking to whatever was active: {default_data:#?}"
    );

    // An explicit `parent_session` naming an unrelated id wins outright, even though it names neither
    // the session that was active before this call nor the one just created above.
    let explicit_parent = "some-other-session-id-entirely";
    writeln!(
        stdin,
        "{}",
        json!({ "type": "new_session", "parent_session": explicit_parent })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "new_session");
    let override_data = &frames.last().unwrap()["data"];
    assert_eq!(
        override_data["parent"], explicit_parent,
        "an explicit parent_session must override the default lineage: {override_data:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_delete_session_soft_deletes_another_session_but_refuses_the_active_one() {
    // Track L5: `SessionRepo::delete` had no RPC command wired to it at all — genuinely unreachable in
    // production. Deleting a *different* session must remove it from `list_sessions` (moved to
    // `.trash`, not gone outright); deleting the *currently active* one must be refused, since that
    // would move the file out from under the in-memory `Session` a client is still using.
    let dir = tempfile::tempdir().unwrap();
    let session_dir = dir.path().join("sessions").to_string_lossy().into_owned();

    let (base, _bodies) = spawn_model_server(vec![turn_text("first answer")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_dir_cmd(bin, &base, &session_dir).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    let mut ready = String::new();
    stdout.read_line(&mut ready).unwrap();
    let ready: Value = serde_json::from_str(ready.trim()).unwrap();
    let first_id = ready["session_id"].as_str().unwrap().to_string();

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    // A second session becomes active; the first is now just another entry in the repo.
    writeln!(stdin, "{}", json!({ "type": "new_session" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "new_session");
    let second_id = frames.last().unwrap()["data"]["session_id"]
        .as_str()
        .unwrap()
        .to_string();
    assert_ne!(first_id, second_id);

    // Refuses to delete the currently active session (session 2).
    writeln!(
        stdin,
        "{}",
        json!({ "type": "delete_session", "session_id": second_id })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "delete_session");
    let resp = frames.last().unwrap();
    assert_eq!(
        resp["success"], false,
        "must refuse to delete the currently active session: {resp:#?}"
    );

    // Deletes the *other* session (session 1) successfully.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "delete_session", "session_id": first_id })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "delete_session");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");

    // No longer listed.
    writeln!(stdin, "{}", json!({ "type": "list_sessions" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "list_sessions");
    let sessions = frames.last().unwrap()["data"]["sessions"]
        .as_array()
        .unwrap();
    assert!(
        !sessions.iter().any(|s| s["id"] == first_id),
        "the deleted session must no longer be listed: {sessions:#?}"
    );

    // Idempotent: deleting it again (already gone) is still a success, not an error.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "delete_session", "session_id": first_id })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "delete_session");
    assert_eq!(
        frames.last().unwrap()["success"],
        true,
        "deleting an already-deleted session must be a no-op success: {frames:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_fork_by_target_id_reaches_an_off_active_path_branch() {
    // `fork`'s `upto` count only ever copies a prefix of whatever is *currently* the active path — it
    // has no way to reach a branch the client has since navigated away from. `target_id` does: fork
    // directly from any tree entry, on or off the active path, without first `switch_branch`-ing to it
    // (which would itself mutate the live session just to preview/fork from it).
    let dir = tempfile::tempdir().unwrap();
    let session_dir = dir.path().join("sessions").to_string_lossy().into_owned();

    let (base, _bodies) = spawn_model_server(vec![
        turn_text("a-reply"), // prompt "a"
        turn_text("b-reply"), // prompt "b"
        turn_text("c-reply"), // prompt "c"
        turn_text("d-reply"), // prompt "d", after switching back to a's leaf
    ]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_dir_cmd(bin, &base, &session_dir).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    for msg in ["a", "b", "c"] {
        writeln!(stdin, "{}", json!({ "type": "prompt", "message": msg })).unwrap();
        stdin.flush().unwrap();
        read_until_response(&mut stdout, "prompt");
    }

    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let messages = frames.last().unwrap()["data"]["messages"]
        .as_array()
        .unwrap()
        .clone();
    assert_eq!(
        messages.len(),
        6,
        "3 user + 3 assistant turns: {messages:#?}"
    );
    let ids: Vec<String> = messages
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_string())
        .collect();
    // ids[0] = user "a", ids[1] = assistant "a-reply", ids[2] = user "b", ids[3] = assistant
    // "b-reply", ids[4] = user "c", ids[5] = assistant "c-reply".

    // Switch back to a's own leaf and append "d" — b/c's turns fall off the active path entirely.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "switch_branch", "target_id": ids[1], "summarize": false })
    )
    .unwrap();
    stdin.flush().unwrap();
    assert_eq!(
        read_until_response(&mut stdout, "switch_branch")
            .last()
            .unwrap()["success"],
        true
    );
    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "d" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    // `preview_fork` targeting b's own user turn (off-branch) with `before:true` previews just
    // [a, a-reply] — b's user message excluded (its own parent is a-reply, so excluding just b's
    // message, not "the whole b pair", is exactly what `before` means: fork right before this entry).
    writeln!(stdin, "{}", json!({ "type": "list_sessions" })).unwrap();
    stdin.flush().unwrap();
    let count_before_preview = read_until_response(&mut stdout, "list_sessions")
        .last()
        .unwrap()["data"]["sessions"]
        .as_array()
        .unwrap()
        .len();

    writeln!(
        stdin,
        "{}",
        json!({ "type": "preview_fork", "target_id": ids[2], "before": true })
    )
    .unwrap();
    stdin.flush().unwrap();
    let preview = read_until_response(&mut stdout, "preview_fork");
    let preview_messages = preview.last().unwrap()["data"]["messages"]
        .as_array()
        .unwrap()
        .clone();
    assert_eq!(
        preview_messages.len(),
        2,
        "before:true at b's own user turn excludes it, leaving just a's turn: {preview_messages:#?}"
    );

    // The `target_id`-based preview must not have created a session file either — it previously did,
    // permanently polluting `list_sessions` on every branch-point preview (the documented common case).
    writeln!(stdin, "{}", json!({ "type": "list_sessions" })).unwrap();
    stdin.flush().unwrap();
    let count_after_preview = read_until_response(&mut stdout, "list_sessions")
        .last()
        .unwrap()["data"]["sessions"]
        .as_array()
        .unwrap()
        .len();
    assert_eq!(
        count_after_preview, count_before_preview,
        "preview_fork with target_id must not create a session file"
    );

    // A real `fork` at c's assistant reply (off-branch), explicit `before:false` — includes it,
    // reaching the full original a->b->c line, none of which is the *current* active path (a->d).
    // Explicit here because `before` now defaults to `true` (matching pi's real client), and `true`
    // at a non-user-message entry (c's assistant reply) is an invalid fork target (Track L27).
    writeln!(
        stdin,
        "{}",
        json!({ "type": "fork", "target_id": ids[5], "before": false })
    )
    .unwrap();
    stdin.flush().unwrap();
    let fork_response = read_until_response(&mut stdout, "fork");
    assert_eq!(fork_response.last().unwrap()["success"], true);

    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let dump = frames.last().unwrap()["data"]["messages"].to_string();
    assert!(dump.contains("a-reply") && dump.contains("b-reply") && dump.contains("c-reply"));
    assert!(
        !dump.contains("d-reply"),
        "the fork reached the off-branch a->b->c line, not the active a->d one: {dump}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_fork_includes_the_forked_from_messages_text() {
    // Fix 3 (pi-parity remediation, Round 2): `fork`'s response previously had no `text` field at all
    // — pi's own `{success, data:{text, cancelled}}` shape echoes the forked-from message's own content
    // (`selectedText`, `extractUserMessageText`) so a client doesn't need a second
    // `get_fork_messages`-style round trip just to redisplay what it already selected. Covers both of
    // this crate's own fork resolutions: a `target_id`-named entry (here, an assistant reply — proving
    // `text` isn't silently role-gated the way `get_fork_messages`'s own user-turn-only candidate list
    // is) and a bare `upto` count with no `target_id` at all.
    let dir = tempfile::tempdir().unwrap();
    let session_dir = dir.path().join("sessions").to_string_lossy().into_owned();

    let (base, _bodies) = spawn_model_server(vec![turn_text("a-reply"), turn_text("b-reply")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_dir_cmd(bin, &base, &session_dir).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    for msg in ["a", "b"] {
        writeln!(stdin, "{}", json!({ "type": "prompt", "message": msg })).unwrap();
        stdin.flush().unwrap();
        read_until_response(&mut stdout, "prompt");
    }

    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let messages = frames.last().unwrap()["data"]["messages"]
        .as_array()
        .unwrap()
        .clone();
    let ids: Vec<String> = messages
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_string())
        .collect();
    // ids[0] = user "a", ids[1] = assistant "a-reply", ids[2] = user "b", ids[3] = assistant "b-reply".

    // `target_id` at an assistant reply (non-user role); `before:false` since `true` at a non-user
    // entry is an invalid fork target (Track L27) — `text` must echo that exact reply's own content
    // regardless, not anything derived from the copied prefix.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "fork", "target_id": ids[1], "before": false })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "fork");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    assert_eq!(
        frames.last().unwrap()["data"]["text"],
        "a-reply",
        "got: {frames:#?}"
    );

    // Bare `upto` (no `target_id`) now operates on the just-forked session (`[user "a", assistant
    // "a-reply"]`): `text` is the last message the copied `upto:1` prefix actually ends on.
    writeln!(stdin, "{}", json!({ "type": "fork", "upto": 1 })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "fork");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    assert_eq!(
        frames.last().unwrap()["data"]["text"],
        "a",
        "got: {frames:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_fork_and_preview_fork_default_before_to_true() {
    // Track L17: pi's real production client always forks with `position:"before"` — this crate's
    // `before` used to default to `false` (include the target entry), the opposite of pi's actual
    // convention. Omitting `before` entirely from both `fork` and `preview_fork` must now exclude the
    // targeted entry, matching pi without the client having to say so explicitly every time.
    let dir = tempfile::tempdir().unwrap();
    let session_dir = dir.path().join("sessions").to_string_lossy().into_owned();

    let (base, _bodies) = spawn_model_server(vec![turn_text("a-reply"), turn_text("b-reply")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_dir_cmd(bin, &base, &session_dir).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    for msg in ["a", "b"] {
        writeln!(stdin, "{}", json!({ "type": "prompt", "message": msg })).unwrap();
        stdin.flush().unwrap();
        read_until_response(&mut stdout, "prompt");
    }

    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let ids: Vec<String> = frames.last().unwrap()["data"]["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_string())
        .collect();
    // ids[0] = user "a", ids[1] = assistant "a-reply", ids[2] = user "b", ids[3] = assistant "b-reply".

    // `preview_fork` at b's own user turn, no `before` field at all — must exclude it, leaving just a's
    // pair, exactly as an explicit `before:true` would.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "preview_fork", "target_id": ids[2] })
    )
    .unwrap();
    stdin.flush().unwrap();
    let preview = read_until_response(&mut stdout, "preview_fork");
    let preview_messages = preview.last().unwrap()["data"]["messages"]
        .as_array()
        .unwrap()
        .clone();
    assert_eq!(
        preview_messages.len(),
        2,
        "no `before` field defaults to true, excluding b's own turn: {preview_messages:#?}"
    );

    // A real `fork` at the same target, again with no `before` field — same exclusion.
    writeln!(stdin, "{}", json!({ "type": "fork", "target_id": ids[2] })).unwrap();
    stdin.flush().unwrap();
    assert_eq!(
        read_until_response(&mut stdout, "fork").last().unwrap()["success"],
        true
    );
    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let dump = read_until_response(&mut stdout, "get_messages")
        .last()
        .unwrap()["data"]["messages"]
        .to_string();
    assert!(dump.contains("a-reply"));
    assert!(
        !dump.contains("b-reply"),
        "no `before` field must exclude b's own turn from the forked session: {dump}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_list_all_sessions_spans_every_project_under_the_shared_root() {
    // Two independent `serve` processes, each rooted at its own subdirectory of one shared parent —
    // the layout `default_session_dir` produces per-project. `list_sessions` from either must see only
    // its own project's session; `list_all_sessions` must see both.
    let root = tempfile::tempdir().unwrap();
    let dir_a = root.path().join("proj-a").to_string_lossy().into_owned();
    let dir_b = root.path().join("proj-b").to_string_lossy().into_owned();

    let (base_a, _bodies_a) = spawn_model_server(vec![turn_text("answer from a")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child_a = serve_dir_cmd(bin, &base_a, &dir_a).spawn().unwrap();
    let mut stdin_a = child_a.stdin.take().unwrap();
    let mut stdout_a = BufReader::new(child_a.stdout.take().unwrap());
    writeln!(stdin_a, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin_a.flush().unwrap();
    read_until_response(&mut stdout_a, "prompt");

    let (base_b, _bodies_b) = spawn_model_server(vec![turn_text("answer from b")]);
    let mut child_b = serve_dir_cmd(bin, &base_b, &dir_b).spawn().unwrap();
    let mut stdin_b = child_b.stdin.take().unwrap();
    let mut stdout_b = BufReader::new(child_b.stdout.take().unwrap());
    writeln!(stdin_b, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin_b.flush().unwrap();
    read_until_response(&mut stdout_b, "prompt");

    // `list_sessions` from process A sees only its own project's session.
    writeln!(stdin_a, "{}", json!({ "type": "list_sessions" })).unwrap();
    stdin_a.flush().unwrap();
    let frames = read_until_response(&mut stdout_a, "list_sessions");
    let sessions = frames.last().unwrap()["data"]["sessions"]
        .as_array()
        .unwrap();
    assert_eq!(
        sessions.len(),
        1,
        "list_sessions must stay scoped to this project: {sessions:#?}"
    );

    // `list_all_sessions` from process A sees both projects' sessions.
    writeln!(stdin_a, "{}", json!({ "type": "list_all_sessions" })).unwrap();
    stdin_a.flush().unwrap();
    let frames = read_until_response(&mut stdout_a, "list_all_sessions");
    let response = frames.last().unwrap();
    assert_eq!(response["success"], true, "got: {response:#?}");
    let sessions = response["data"]["sessions"].as_array().unwrap();
    assert_eq!(
        sessions.len(),
        2,
        "list_all_sessions must span both projects: {sessions:#?}"
    );

    drop(stdin_a);
    child_a.wait().unwrap();
    drop(stdin_b);
    child_b.wait().unwrap();
}

#[test]
fn serve_list_sessions_filters_by_cwd_under_one_shared_session_dir() {
    // Track L28 (pi-parity fix): unlike `serve_list_all_sessions_spans_every_project_under_the_shared_root`
    // (above), which uses one subdirectory *per* project — so `list_sessions` there was only ever
    // "correct" by construction (each repo directory physically holds just one project's sessions) —
    // this drives two `serve` processes at *one* literal `--session-dir`, from two different cwds. Only
    // `resume_or_create` (at startup) used to filter by cwd; `list_sessions` returned every session in
    // the shared directory unfiltered, leaking another project's sessions into this one's listing.
    let shared_dir = tempfile::tempdir().unwrap();
    let cwd_a = tempfile::tempdir().unwrap();
    let cwd_b = tempfile::tempdir().unwrap();

    let (base_a, _bodies_a) = spawn_model_server(vec![turn_text("answer from a")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child_a = serve_dir_cmd(bin, &base_a, &shared_dir.path().to_string_lossy())
        .current_dir(cwd_a.path())
        .spawn()
        .unwrap();
    let mut stdin_a = child_a.stdin.take().unwrap();
    let mut stdout_a = BufReader::new(child_a.stdout.take().unwrap());
    writeln!(stdin_a, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin_a.flush().unwrap();
    read_until_response(&mut stdout_a, "prompt");

    let (base_b, _bodies_b) = spawn_model_server(vec![turn_text("answer from b")]);
    let mut child_b = serve_dir_cmd(bin, &base_b, &shared_dir.path().to_string_lossy())
        .current_dir(cwd_b.path())
        .spawn()
        .unwrap();
    let mut stdin_b = child_b.stdin.take().unwrap();
    let mut stdout_b = BufReader::new(child_b.stdout.take().unwrap());
    writeln!(stdin_b, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin_b.flush().unwrap();
    read_until_response(&mut stdout_b, "prompt");

    // Both sessions really do live in the one shared directory.
    let jsonl_count = std::fs::read_dir(shared_dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("jsonl"))
        .count();
    assert_eq!(
        jsonl_count, 2,
        "expected both sessions physically in the one shared --session-dir"
    );

    // `list_sessions` from process A must only ever return *its own* cwd's session.
    writeln!(stdin_a, "{}", json!({ "type": "list_sessions" })).unwrap();
    stdin_a.flush().unwrap();
    let frames = read_until_response(&mut stdout_a, "list_sessions");
    let sessions = frames.last().unwrap()["data"]["sessions"]
        .as_array()
        .unwrap();
    assert_eq!(
        sessions.len(),
        1,
        "list_sessions must filter to this process's own cwd under a shared --session-dir: {sessions:#?}"
    );

    drop(stdin_a);
    child_a.wait().unwrap();
    drop(stdin_b);
    child_b.wait().unwrap();
}

#[test]
fn serve_list_all_sessions_errors_outside_repo_mode() {
    // Single-file persistence (`--session-file`) has no per-project sibling directories to scan.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "list_all_sessions" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "list_all_sessions");
    let response = frames.last().unwrap();
    assert_eq!(response["success"], false, "got: {response:#?}");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_get_tree_reports_every_node_not_just_leaves() {
    // `list_branches` reports only leaves; `get_tree` must report every node on every branch — proven
    // by branching (via `fork`) and confirming `get_tree`'s node count exceeds what a leaves-only view
    // would show, and that every node carries a role and (for text turns) a preview.
    let dir = tempfile::tempdir().unwrap();
    let session_dir = dir.path().join("sessions").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![turn_text("first answer")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_dir_cmd(bin, &base, &session_dir).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    writeln!(stdin, "{}", json!({ "type": "get_tree" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_tree");
    let nodes = frames.last().unwrap()["data"]["nodes"].as_array().unwrap();
    // user + assistant.
    assert_eq!(nodes.len(), 2, "got: {nodes:#?}");
    assert!(nodes.iter().any(|n| n["role"] == "user"));
    assert!(nodes.iter().any(|n| n["role"] == "assistant"));
    assert!(
        nodes
            .iter()
            .any(|n| n["preview"].as_str().is_some_and(|p| p.contains("hi"))),
        "the user node should preview its own text: {nodes:#?}"
    );
    // The root node has no parent.
    assert!(nodes.iter().any(|n| n["parent_id"].is_null()));

    // Pi-parity fix: `get_tree` must also report the active path's own tip — pi's own `leafId` — so a
    // client can tell which node is "where the session currently is" without a separate round trip.
    let leaf_id = frames.last().unwrap()["data"]["leaf_id"].as_str().unwrap();
    let assistant_id = nodes.iter().find(|n| n["role"] == "assistant").unwrap()["id"]
        .as_str()
        .unwrap();
    assert_eq!(
        leaf_id, assistant_id,
        "leaf_id must be the active path's own tip: {frames:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_get_tree_since_returns_only_what_was_appended_after_a_known_id() {
    // Task #48 (pi-parity gap): `get_tree`'s own `since` mirrors `get_messages`'s (see
    // `serve_get_messages_since_returns_only_what_was_appended_after_a_known_id` above), but scoped to
    // *every* entry type, not just plain LLM messages — pi's own `SessionManager.getEntries({since})`
    // backs both commands the same way. Proven here with a `custom` entry (a different `entry_kind`
    // than an ordinary message) appended between the two turns: a `since` that only ever walked
    // `Session.messages` would miss it entirely.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) =
        spawn_model_server(vec![turn_text("first answer"), turn_text("second answer")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "first" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    writeln!(stdin, "{}", json!({ "type": "get_tree" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_tree");
    let nodes = frames.last().unwrap()["data"]["nodes"]
        .as_array()
        .unwrap()
        .clone();
    assert_eq!(nodes.len(), 2, "expected [user, assistant]: {nodes:#?}");
    let baseline_id = nodes.iter().find(|n| n["role"] == "assistant").unwrap()["id"]
        .as_str()
        .unwrap()
        .to_string();

    // A custom entry — a different entry type than an ordinary message — appended after the baseline.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "append_custom", "kind": "checkpoint-marker" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "append_custom");

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "second" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    // Only what's new since the first turn's assistant reply: the custom entry, plus the second turn's
    // user + assistant.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "get_tree", "since": baseline_id })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_tree");
    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], true, "{resp:#?}");
    let since_nodes = resp["data"]["nodes"].as_array().unwrap();
    assert_eq!(since_nodes.len(), 3, "{since_nodes:#?}");
    assert!(
        since_nodes.iter().any(|n| n["entry_kind"] == "custom"),
        "a custom entry must be included, not just plain messages: {since_nodes:#?}"
    );
    assert!(
        since_nodes.iter().any(|n| n["role"] == "user"
            && n["preview"]
                .as_str()
                .is_some_and(|p| p.contains("second"))),
        "{since_nodes:#?}"
    );
    assert!(
        since_nodes.iter().any(|n| n["role"] == "assistant"
            && n["preview"]
                .as_str()
                .is_some_and(|p| p.contains("second answer"))),
        "{since_nodes:#?}"
    );
    // Nothing from before (or at) the baseline leaks through.
    assert!(!since_nodes.iter().any(|n| n["id"] == baseline_id));
    // `get_tree`'s own `leaf_id` still reports the *current* active tip, not the `since`-filtered one.
    assert!(resp["data"]["leaf_id"].as_str().is_some());

    // An unknown `since` id is an error, not a silent full re-fetch — same contract as `get_messages`.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "get_tree", "since": "does-not-exist" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_tree");
    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], false, "{resp:#?}");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_get_tree_leaf_id_is_null_without_persistence_configured() {
    // Mirrors `serve_get_fork_messages_is_empty_without_persistence_configured`'s reasoning: pure
    // in-memory mode has no `SessionStore` to track an active tip at all.
    let (base, _bodies) = spawn_model_server(vec![]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = Command::new(bin)
        .args([
            "serve",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--no-session-persistence",
        ])
        .env("HOME", "/nonexistent-beyond-ai-agent-test-home")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "get_tree" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_tree");
    let resp = frames.last().unwrap();
    assert_eq!(resp["data"]["nodes"], json!([]), "got: {resp:#?}");
    assert!(resp["data"]["leaf_id"].is_null(), "got: {resp:#?}");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_switch_branch_summarizes_abandoned_activity_and_navigates() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    // Two turns build a linear history, a third is the branch-summarization call `switch_branch`
    // triggers, a fourth answers the prompt issued after navigating back.
    let (base, _bodies) = spawn_model_server(vec![
        turn_text("first answer"),
        turn_text("second answer"),
        turn_text("recap: explored a dead end"),
        turn_text("continued from the original branch"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "first" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "second" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    // A single, unbranched history reports exactly one branch, 4 messages deep.
    writeln!(stdin, "{}", json!({ "type": "list_branches" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "list_branches");
    let branches = frames.last().unwrap()["data"]["branches"]
        .as_array()
        .unwrap()
        .clone();
    assert_eq!(branches.len(), 1, "no branching yet: {branches:#?}");
    assert_eq!(branches[0]["is_active"], true);
    assert_eq!(branches[0]["message_count"], 4);

    // Navigate back to the first turn's assistant reply (message index 1), abandoning the second
    // turn's user+assistant messages.
    let ids = message_ids(&session_file);
    assert_eq!(ids.len(), 4, "expected 4 persisted messages: {ids:?}");
    let rewind_to = ids[1].clone();
    let abandoned_tip = ids[3].clone();

    writeln!(
        stdin,
        "{}",
        json!({ "type": "switch_branch", "target_id": rewind_to })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "switch_branch");
    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], true, "switch_branch failed: {resp:#?}");
    assert_eq!(resp["data"]["target_id"], rewind_to);

    // The active transcript is now the first turn *plus* the abandoned branch's summary — the recap
    // must actually reach the model-facing transcript, not just sit persisted off to the side.
    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let dump = frames.last().unwrap()["data"]["messages"].to_string();
    assert!(dump.contains("first answer"));
    assert!(
        dump.contains("recap: explored a dead end"),
        "the branch summary must be part of the live, model-facing transcript: {dump}"
    );
    assert!(
        !dump.contains("second answer"),
        "the abandoned turn must not appear on the restored branch: {dump}"
    );

    // Two branches now exist: the abandoned one (inactive, still 4 deep) and the active one — the
    // first turn (2) plus the summary message now folded into it (1) = 3 deep.
    writeln!(stdin, "{}", json!({ "type": "list_branches" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "list_branches");
    let branches = frames.last().unwrap()["data"]["branches"]
        .as_array()
        .unwrap()
        .clone();
    assert_eq!(branches.len(), 2, "expected two branches: {branches:#?}");
    let abandoned = branches
        .iter()
        .find(|b| b["leaf_id"] == json!(abandoned_tip))
        .expect("the old tip should still be listed as a branch");
    assert_eq!(abandoned["is_active"], false);
    assert_eq!(abandoned["message_count"], 4);
    let active = branches.iter().find(|b| b["is_active"] == true).unwrap();
    assert_eq!(active["message_count"], 3);

    // The abandoned branch's summary was generated (consuming the 3rd mock response) and persisted.
    let raw = std::fs::read_to_string(&session_file).unwrap();
    assert!(
        raw.contains("recap: explored a dead end"),
        "the branch summary should be persisted in the session file:\n{raw}"
    );
    assert!(raw.contains("\"branch_summary\""));

    // Continuing from the restored branch forks a *new* line of history off it, not a resumption of
    // the abandoned one.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "continue" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let dump = frames.last().unwrap()["data"]["messages"].to_string();
    assert!(dump.contains("continued from the original branch"));
    assert!(!dump.contains("second answer"));

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_branch_summary_reserve_tokens_flag_independently_bounds_the_summarization_window() {
    // Task #31 (pi-parity feature): `agent_core::Agent::with_branch_summary_reserve_tokens` existed
    // with no caller in either binary — a branch summary's own input-token budget was always hard-tied
    // to whatever `--compaction-reserve-tokens` resolved to, with no independent override (matching
    // pi's own separately-configurable `branchSummary.reserveTokens`). `--compaction-reserve-tokens` is
    // set small (100) here so the *buggy* fallback path would leave a generous ~19900-token
    // summarization budget (no windowing needed for this tiny two-message abandoned branch), while
    // `--branch-summary-reserve-tokens` is set almost equal to `--context-window` so the *fixed* path
    // leaves only a 1-token budget — forcing `windowed_by_budget` to drop the abandoned branch's older
    // message and stamp the summarization request with its own "omitted to fit the summarization
    // budget" note (see `agent_core::branch_summary::branch_summary_request`'s doc comment). Only the
    // fixed path produces that note, so its presence proves `--branch-summary-reserve-tokens` — not
    // `--compaction-reserve-tokens`'s own much larger reserve — actually bounded this call.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, bodies) = spawn_model_server(vec![
        turn_text("first answer"),
        turn_text("second answer"),
        turn_text("recap"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file)
        .args([
            "--context-window",
            "20000",
            "--compaction-reserve-tokens",
            "100",
            "--branch-summary-reserve-tokens",
            "19999",
        ])
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "first" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "second" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    let ids = message_ids(&session_file);
    assert_eq!(ids.len(), 4, "expected 4 persisted messages: {ids:?}");
    let rewind_to = ids[1].clone();

    writeln!(
        stdin,
        "{}",
        json!({ "type": "switch_branch", "target_id": rewind_to })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "switch_branch");
    assert_eq!(frames.last().unwrap()["success"], true, "got: {frames:#?}");
    drop(stdin);
    child.wait().unwrap();

    let bodies = bodies.lock().unwrap();
    assert_eq!(
        bodies.len(),
        3,
        "expected 2 turns + 1 branch-summary call: {bodies:#?}"
    );
    assert!(
        bodies[2].contains("omitted to fit the summarization budget"),
        "--branch-summary-reserve-tokens=19999 (a 1-token budget out of a 20000 context window) must \
         force the summarization call to window out the abandoned branch's older message: {}",
        bodies[2]
    );
}

#[test]
fn serve_switch_branch_abort_cancels_summarization_and_leaves_session_unchanged() {
    use std::time::{Duration, Instant};

    // pi-parity fix (`packages/coding-agent/test/agent-session-tree-navigation.test.ts:175-212`,
    // "should handle abort during summarization"): `switch_branch{summarize:true}`'s LLM call used to
    // run on a fresh, unreachable `CancellationToken` with no way for a client `abort` to ever reach
    // it — the whole RPC loop just blocked until the call finished, however long that took. This
    // proves `abort` actually interrupts a provably in-flight branch summarization promptly, the
    // response reports `cancelled`/`aborted` rather than an error, and the session (branches/leaf) is
    // left completely unchanged, matching pi's `{cancelled:true, aborted:true, summaryEntry:undefined}`
    // contract.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let base = spawn_model_server_with_stalled_response(
        vec![turn_text("first answer"), turn_text("second answer")],
        Duration::from_secs(5),
        Vec::new(),
    );

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "first" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "second" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    writeln!(stdin, "{}", json!({ "type": "list_branches" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "list_branches");
    let branches_before = frames.last().unwrap()["data"]["branches"].clone();
    assert_eq!(branches_before.as_array().unwrap().len(), 1);

    let ids = message_ids(&session_file);
    assert_eq!(ids.len(), 4, "expected 4 persisted messages: {ids:?}");
    let rewind_to = ids[1].clone();

    writeln!(
        stdin,
        "{}",
        json!({ "type": "switch_branch", "target_id": rewind_to, "summarize": true })
    )
    .unwrap();
    stdin.flush().unwrap();

    // Give the summarization request time to actually reach the stalled server (it writes a partial
    // body immediately on accept, well before its 5s stall completes) before aborting.
    std::thread::sleep(Duration::from_millis(300));
    let start = Instant::now();
    writeln!(stdin, "{}", json!({ "type": "abort", "id": "a1" })).unwrap();
    stdin.flush().unwrap();

    let abort_frames = read_until_response(&mut stdout, "abort");
    assert_eq!(abort_frames.last().unwrap()["success"], true);

    let frames = read_until_response(&mut stdout, "switch_branch");
    let elapsed = start.elapsed();
    assert!(
        elapsed < Duration::from_secs(3),
        "abort must interrupt the stalled summarization promptly, not wait out its 5s stall: {elapsed:?}"
    );
    let resp = frames.last().unwrap();
    assert_eq!(
        resp["success"], true,
        "a cancelled switch is not an RPC-level failure: {resp:#?}"
    );
    assert_eq!(resp["data"]["cancelled"], true);
    assert_eq!(resp["data"]["aborted"], true);

    // The session must be completely unchanged: still one branch, still 4 messages, no partial
    // summary entry anywhere in the persisted file.
    writeln!(stdin, "{}", json!({ "type": "list_branches" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "list_branches");
    let branches_after = frames.last().unwrap()["data"]["branches"].clone();
    assert_eq!(
        branches_after, branches_before,
        "branch structure must be untouched by a cancelled switch"
    );
    assert_eq!(message_ids(&session_file).len(), 4);
    let raw = std::fs::read_to_string(&session_file).unwrap();
    assert!(
        !raw.contains("\"branch_summary\""),
        "a cancelled summarization must not persist a partial summary entry:\n{raw}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_switch_branch_restores_the_model_active_on_that_branch() {
    // `set_model` between two turns records a branch-local change (H6); navigating back to the point
    // right before that change must restore the model that was actually active there, not silently
    // continue with whatever model the process's global setting has since moved on to.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, _bodies) = spawn_model_server(vec![
        turn_text("first answer"),  // prompt "first" on the original model
        turn_text("second answer"), // prompt "second" on the switched-to model
        turn_text("third answer"),  // prompt "third" after switching back
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "first" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    // The tip right now (after "first answer") is where the upcoming model change is anchored.
    let ids = message_ids(&session_file);
    assert_eq!(ids.len(), 2, "expected 2 persisted messages: {ids:?}");
    let anchor = ids[1].clone();

    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_model", "model": "claude-test-2" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_model");
    assert_eq!(frames.last().unwrap()["success"], true, "got: {frames:#?}");

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "second" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    // Navigate back to right after "first answer" — before the model change ever happened.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "switch_branch", "target_id": anchor, "summarize": false })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "switch_branch");
    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], true, "got: {resp:#?}");
    assert_eq!(
        resp["data"]["model"], "claude-test",
        "switching back before the set_model must restore the original model: {resp:#?}"
    );

    // `get_state` (not just the switch_branch response) must reflect the restored model too.
    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    assert_eq!(frames.last().unwrap()["data"]["model"], "claude-test");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_switch_branch_before_restores_the_model_at_the_resolved_parent_not_the_raw_target() {
    // Regression guard for the `before: true` model/thinking-level restoration bug this feature
    // could easily reintroduce: the restored model must be queried against the *resolved* target
    // (`target_id`'s own parent), not the raw, pre-resolution `target_id` argument. Those two only
    // disagree when a model change is anchored exactly at the parent itself — anchored-at-a-node
    // changes take effect for what comes *next*, so querying the unresolved child would wrongly see
    // its own parent's anchor as already in effect for the parent, not just its descendants.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, _bodies) = spawn_model_server(vec![
        turn_text("first answer"),  // m1 (user "first"), m2 (assistant)
        turn_text("second answer"), // m3 (user "second"), m4 (assistant) — under model-B
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "first" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    // Anchored at m2 (the current tip) — takes effect for m3 onward, not for m2 itself.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_model", "model": "claude-test-2" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_model");
    assert_eq!(frames.last().unwrap()["success"], true, "got: {frames:#?}");

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "second" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    let ids = message_ids(&session_file);
    assert_eq!(ids.len(), 4, "expected 4 persisted messages: {ids:?}");
    let m3 = ids[2].clone(); // the user message right after the model-B change took effect

    // `before: true` on m3 resolves to m2 — must restore the model active AT m2 (before the change
    // anchored there took effect), i.e. the original "claude-test", not "claude-test-2".
    writeln!(
        stdin,
        "{}",
        json!({ "type": "switch_branch", "target_id": m3, "before": true, "summarize": false })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "switch_branch");
    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], true, "got: {resp:#?}");
    assert_eq!(
        resp["data"]["model"], "claude-test",
        "before:true must restore the model at the resolved parent (m2), not wrongly pick up a \
         change anchored there that only applies to its descendants: {resp:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_switch_branch_resets_thinking_level_instead_of_bleeding_a_sibling_branchs_setting() {
    // Track L4: a `set_reasoning_effort` recorded on one branch must not silently keep applying after
    // switching to a point that never had it — that point genuinely never had a level recorded, so it
    // must resolve to the *process's own starting level* (here, the default "medium" — Fix 1,
    // pi-parity gap: `claude-test` supports reasoning, so a fresh process now starts there rather than
    // "off" — see `serve_starts_clamped_not_off_for_a_model_that_cannot_disable_reasoning`), not
    // whatever the global runtime setting happens to still be from the branch just abandoned.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, _bodies) = spawn_model_server(vec![turn_text("first answer")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "first" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    // The tip right now (after "first answer") never has — and never will have — its own recorded
    // thinking-level change; the upcoming `set_reasoning_effort` is anchored *after* it.
    let ids = message_ids(&session_file);
    assert_eq!(ids.len(), 2, "expected 2 persisted messages: {ids:?}");
    let anchor = ids[1].clone();

    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_reasoning_effort", "effort": "high" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_reasoning_effort");
    assert_eq!(frames.last().unwrap()["success"], true, "got: {frames:#?}");
    assert_eq!(frames.last().unwrap()["data"]["level"], "high");

    // Switch back to the anchor — a point that predates the level change entirely, and has no
    // ThinkingLevelChange of its own anchored at-or-before it.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "switch_branch", "target_id": anchor, "summarize": false })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "switch_branch");
    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], true, "got: {resp:#?}");
    assert_eq!(
        resp["data"]["reasoning_effort"], "medium",
        "switching to a point with no recorded level change must reset to the process's own \
         starting level, not bleed the abandoned branch's \"high\": {resp:#?}"
    );

    // `get_state` (not just the switch_branch response) must reflect the reset level too.
    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    assert_eq!(frames.last().unwrap()["data"]["thinking_level"], "medium");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_get_messages_ids_enable_forking_from_any_point() {
    // Closes the gap `list_branches` alone leaves: it only ever reports a branch's *leaf*, so a
    // client that wants to fork from an arbitrary point in the middle of the visible transcript needs
    // ids from somewhere else. This proves `get_messages`'s tagged ids are real, usable
    // `switch_branch` targets — not just present, but round-trip through the actual RPC surface.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, _bodies) = spawn_model_server(vec![
        turn_text("first answer"),
        turn_text("second answer"),
        turn_text("forked from message index 1"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "first" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "second" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let messages = frames.last().unwrap()["data"]["messages"]
        .as_array()
        .unwrap()
        .clone();
    assert_eq!(messages.len(), 4, "expected 4 messages: {messages:#?}");
    let ids: Vec<String> = messages
        .iter()
        .map(|m| {
            m["id"]
                .as_str()
                .expect("every message should be tagged with an id")
                .to_string()
        })
        .collect();
    // All four ids are distinct.
    let unique: std::collections::HashSet<&String> = ids.iter().collect();
    assert_eq!(
        unique.len(),
        4,
        "message ids should all be distinct: {ids:?}"
    );

    // Fork from message index 1 (the first turn's assistant reply) — a point `list_branches` alone
    // could never have named, since it only reports the (single, so far) branch's leaf.
    // `summarize:false`: the summarization path itself is covered by
    // `serve_switch_branch_summarizes_abandoned_activity_and_navigates`; this test is about ids, not
    // that, and skipping it keeps the mock response count matched to what's actually queued.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "switch_branch", "target_id": ids[1], "summarize": false })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "switch_branch");
    assert_eq!(frames.last().unwrap()["success"], true);

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "continue from here" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let dump = frames.last().unwrap()["data"]["messages"].to_string();
    assert!(dump.contains("forked from message index 1"));
    assert!(
        !dump.contains("second answer"),
        "forking from index 1 must not carry over the second turn: {dump}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_switch_branch_rejects_unknown_target() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![turn_text("hi")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    writeln!(
        stdin,
        "{}",
        json!({ "type": "switch_branch", "target_id": "does-not-exist" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "switch_branch");
    assert_eq!(frames.last().unwrap()["success"], false);

    writeln!(stdin, "{}", json!({ "type": "switch_branch" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "switch_branch");
    assert_eq!(frames.last().unwrap()["success"], false);

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_switch_branch_before_the_first_message_resets_to_root() {
    // Pi-parity fix: no way existed to navigate back to before the very first message — pi's own
    // `SessionManager::resetLeaf`, exposed by pointing `switch_branch` at the first message's own
    // `target_id` with `before: true` (mirroring `fork`'s identical `before` semantics), rather than a
    // separate root sentinel on the wire.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![
        turn_text("first answer"),
        turn_text("second answer"),
        turn_text("redone first answer"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "first" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "second" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    let ids = message_ids(&session_file);
    assert_eq!(ids.len(), 4, "expected 4 persisted messages: {ids:?}");
    let first_message_id = ids[0].clone();

    writeln!(
        stdin,
        "{}",
        json!({ "type": "switch_branch", "target_id": first_message_id, "before": true, "summarize": false })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "switch_branch");
    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], true, "switch_branch failed: {resp:#?}");

    // The active transcript is now genuinely empty — reset all the way to root, not just to the
    // first message itself.
    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let messages = frames.last().unwrap()["data"]["messages"]
        .as_array()
        .unwrap()
        .clone();
    assert!(
        messages.is_empty(),
        "expected an empty transcript: {messages:#?}"
    );

    // A fresh prompt now redoes the first message in place — the model sees no prior turns at all.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "redo the first message" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    assert_eq!(
        frames.last().unwrap()["success"],
        true,
        "the redone turn must actually run: {frames:#?}"
    );

    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let messages = frames.last().unwrap()["data"]["messages"]
        .as_array()
        .unwrap()
        .clone();
    assert_eq!(
        messages.len(),
        2,
        "expected exactly the redone user+assistant turn, no trace of the old branch: {messages:#?}"
    );
    let dump = format!("{messages:?}");
    assert!(dump.contains("redone first answer"), "{dump}");
    assert!(!dump.contains("second answer"), "{dump}");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_list_sessions_query_filters_to_matching_sessions_only() {
    // Pi-parity fix: `search_text`/`preview` were computed and serialized into every listing entry, but
    // nothing ever filtered or ranked by them — `list_sessions`/`list_all_sessions` always returned
    // every session regardless of any query. A `query` field must now narrow the result to sessions
    // whose recorded text actually contains it.
    let session_dir = tempfile::tempdir().unwrap();
    let session_dir_str = session_dir.path().to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![turn_text("ok"), turn_text("ok")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_dir_cmd(bin, &base, &session_dir_str).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "remember the marker: zephyr-unique-42" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    writeln!(stdin, "{}", json!({ "type": "new_session" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "new_session");
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "a totally unrelated topic" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    // No query: both sessions listed, matching today's existing behavior exactly.
    writeln!(stdin, "{}", json!({ "type": "list_sessions" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "list_sessions");
    let sessions = frames.last().unwrap()["data"]["sessions"]
        .as_array()
        .unwrap();
    assert_eq!(
        sessions.len(),
        2,
        "an absent query must list every session, unfiltered: {sessions:#?}"
    );

    // A query matching only the first session's content.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "list_sessions", "query": "zephyr-unique-42" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "list_sessions");
    let sessions = frames.last().unwrap()["data"]["sessions"]
        .as_array()
        .unwrap();
    assert_eq!(
        sessions.len(),
        1,
        "the query must filter out the unrelated session: {sessions:#?}"
    );
    assert!(
        sessions[0]["search_text"]
            .as_str()
            .unwrap()
            .contains("zephyr-unique-42"),
        "the surviving session must be the one that actually matched: {sessions:#?}"
    );

    // A query matching nothing at all is an empty (still successful) result, not an error.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "list_sessions", "query": "no-such-term-anywhere-xyz" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "list_sessions");
    let response = frames.last().unwrap();
    assert_eq!(response["success"], true, "got: {response:#?}");
    assert!(
        response["data"]["sessions"].as_array().unwrap().is_empty(),
        "an unmatched query returns no sessions, not a fault: {response:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_list_all_sessions_query_filters_across_every_project() {
    // The cross-project half of the same fix: `list_all_sessions`'s `query` must narrow results spanning
    // every project's own directory, not just the current one.
    let root = tempfile::tempdir().unwrap();
    let dir_a = root.path().join("proj-a").to_string_lossy().into_owned();
    let dir_b = root.path().join("proj-b").to_string_lossy().into_owned();

    let (base_a, _bodies_a) = spawn_model_server(vec![turn_text("ok")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child_a = serve_dir_cmd(bin, &base_a, &dir_a).spawn().unwrap();
    let mut stdin_a = child_a.stdin.take().unwrap();
    let mut stdout_a = BufReader::new(child_a.stdout.take().unwrap());
    writeln!(
        stdin_a,
        "{}",
        json!({ "type": "prompt", "message": "project A discusses widget-frobnication" })
    )
    .unwrap();
    stdin_a.flush().unwrap();
    read_until_response(&mut stdout_a, "prompt");

    let (base_b, _bodies_b) = spawn_model_server(vec![turn_text("ok")]);
    let mut child_b = serve_dir_cmd(bin, &base_b, &dir_b).spawn().unwrap();
    let mut stdin_b = child_b.stdin.take().unwrap();
    let mut stdout_b = BufReader::new(child_b.stdout.take().unwrap());
    writeln!(
        stdin_b,
        "{}",
        json!({ "type": "prompt", "message": "project B is about something else entirely" })
    )
    .unwrap();
    stdin_b.flush().unwrap();
    read_until_response(&mut stdout_b, "prompt");

    writeln!(
        stdin_a,
        "{}",
        json!({ "type": "list_all_sessions", "query": "widget-frobnication" })
    )
    .unwrap();
    stdin_a.flush().unwrap();
    let frames = read_until_response(&mut stdout_a, "list_all_sessions");
    let response = frames.last().unwrap();
    assert_eq!(response["success"], true, "got: {response:#?}");
    let sessions = response["data"]["sessions"].as_array().unwrap();
    assert_eq!(
        sessions.len(),
        1,
        "the query must narrow the cross-project result to the one matching session: {sessions:#?}"
    );
    assert!(
        sessions[0]["search_text"]
            .as_str()
            .unwrap()
            .contains("widget-frobnication"),
        "the surviving session must be project A's, the one that actually matched: {sessions:#?}"
    );

    drop(stdin_a);
    child_a.wait().unwrap();
    drop(stdin_b);
    child_b.wait().unwrap();
}

#[test]
fn serve_get_fork_messages_lists_user_turn_candidates_for_the_active_path() {
    // Pi-parity fix: `get_fork_messages` used to be this crate's own arbitrary-session/arbitrary-point
    // fork *preview* (now `preview_fork`) — a different contract than pi's own same-named command, which
    // takes no parameters and lists every user-turn entry on the *current* session's active path as
    // `{entry_id, text}` fork-point candidates. A pi-compatible client's zero-arg call must now get that
    // shape back, not the old preview's full multi-role `Message` array.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) =
        spawn_model_server(vec![turn_text("first answer"), turn_text("second answer")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "first user message" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "second user message" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    // No parameters — pi's own contract.
    writeln!(stdin, "{}", json!({ "type": "get_fork_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_fork_messages");
    let response = frames.last().unwrap();
    assert_eq!(response["success"], true, "got: {response:#?}");
    let candidates = response["data"]["messages"].as_array().unwrap();
    assert_eq!(
        candidates.len(),
        2,
        "only the two user turns are candidates, not the assistant replies: {candidates:#?}"
    );
    assert_eq!(candidates[0]["text"], "first user message");
    assert_eq!(candidates[1]["text"], "second user message");
    for c in candidates {
        assert!(
            c["entry_id"].as_str().is_some_and(|s| !s.is_empty()),
            "each candidate must carry a usable entry_id: {c:#?}"
        );
    }

    // Each `entry_id` must be a real, usable `target_id` — the same id `get_messages` tags that exact
    // user turn with, so a client can feed it straight into `fork`'s own `target_id`.
    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let messages = frames.last().unwrap()["data"]["messages"]
        .as_array()
        .unwrap()
        .clone();
    let user_ids: Vec<&str> = messages
        .iter()
        .filter(|m| m["role"] == "user")
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert_eq!(
        candidates[0]["entry_id"].as_str().unwrap(),
        user_ids[0],
        "entry_id must match get_messages' own id for the same user turn"
    );
    assert_eq!(candidates[1]["entry_id"].as_str().unwrap(), user_ids[1]);

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_get_fork_messages_spans_every_branch_not_just_the_active_path() {
    // Task #25 (pi-parity fix): `get_fork_messages` used to build its candidate list from
    // `persistence.active_ids()` only — once a branch was navigated away from, its own user messages
    // silently stopped being fork candidates at all. pi's own `getUserMessagesForForking` walks
    // `SessionManager::getEntries()` — every entry ever appended, spanning the whole tree — so a message
    // on an abandoned branch must still show up here.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![
        turn_text("first answer"),
        turn_text("second answer"),
        turn_text("forked reply"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "first" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "second" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let messages = frames.last().unwrap()["data"]["messages"]
        .as_array()
        .unwrap()
        .clone();
    let first_reply_id = messages[1]["id"].as_str().unwrap().to_string();

    // Rewind to right after the first turn, then send a new prompt — this abandons the "second" turn's
    // own branch (still on disk, just off the new active path) and starts a sibling branch in its place.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "switch_branch", "target_id": first_reply_id, "summarize": false })
    )
    .unwrap();
    stdin.flush().unwrap();
    assert_eq!(
        read_until_response(&mut stdout, "switch_branch")
            .last()
            .unwrap()["success"],
        true
    );

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "continue from here" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    writeln!(stdin, "{}", json!({ "type": "get_fork_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_fork_messages");
    let response = frames.last().unwrap();
    assert_eq!(response["success"], true, "got: {response:#?}");
    let candidates = response["data"]["messages"].as_array().unwrap();
    let texts: Vec<&str> = candidates
        .iter()
        .map(|c| c["text"].as_str().unwrap())
        .collect();
    assert_eq!(
        texts.len(),
        3,
        "expected one candidate per user turn across both branches: {texts:?}"
    );
    assert!(
        texts.contains(&"first"),
        "shared prefix message missing: {texts:?}"
    );
    assert!(
        texts.contains(&"second"),
        "the abandoned branch's own user message must still be a fork candidate: {texts:?}"
    );
    assert!(
        texts.contains(&"continue from here"),
        "the new active branch's own message must also be a candidate: {texts:?}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_get_fork_messages_is_empty_without_persistence_configured() {
    // In pure in-memory mode there are no stable entry ids to hand back for a later `fork` `target_id`
    // — `get_fork_messages` must degrade to an empty (still successful) list, matching `get_messages`'s
    // own precedent for the same mismatch, rather than erroring.
    let (base, _bodies) = spawn_model_server(vec![turn_text("ok")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = Command::new(bin)
        .args([
            "serve",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--no-session-persistence",
        ])
        .env("HOME", "/nonexistent-beyond-ai-agent-test-home")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    writeln!(stdin, "{}", json!({ "type": "get_fork_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_fork_messages");
    let response = frames.last().unwrap();
    assert_eq!(response["success"], true, "got: {response:#?}");
    assert!(
        response["data"]["messages"].as_array().unwrap().is_empty(),
        "no persistence means no stable ids to offer: {response:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_switch_session_resolves_a_unique_id_prefix() {
    // Pi-parity fix: matching was exact-only everywhere — pi's own `resolveSessionPath` accepts a
    // shortened, unambiguous prefix as a typing convenience.
    let dir = tempfile::tempdir().unwrap();
    let session_dir = dir.path().join("sessions").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![turn_text("first answer")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_dir_cmd(bin, &base, &session_dir).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    let mut ready = String::new();
    stdout.read_line(&mut ready).unwrap();
    let ready: Value = serde_json::from_str(ready.trim()).unwrap();
    let first_id = ready["session_id"].as_str().unwrap().to_string();

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    writeln!(stdin, "{}", json!({ "type": "new_session" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "new_session");

    let prefix = &first_id[..first_id.len() / 2];
    writeln!(
        stdin,
        "{}",
        json!({ "type": "switch_session", "session_id": prefix })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "switch_session");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");

    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let dump = frames.last().unwrap()["data"]["messages"].to_string();
    assert!(
        dump.contains("first answer"),
        "switching via a prefix must reach the session it uniquely resolves to: {dump}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_switch_session_rejects_an_ambiguous_prefix_naming_every_candidate() {
    let dir = tempfile::tempdir().unwrap();
    let session_dir = dir.path().join("sessions").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![turn_text("first"), turn_text("second")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_dir_cmd(bin, &base, &session_dir).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    writeln!(stdin, "{}", json!({ "type": "new_session" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "new_session");
    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "yo" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    // A single character is virtually guaranteed to match both generated ids' shared leading hex nanos
    // digit(s) is not reliable, so instead assert against the empty-string prefix, which unambiguously
    // matches every session that exists.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "switch_session", "session_id": "" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "switch_session");
    let response = frames.last().unwrap();
    assert_eq!(response["success"], false, "{response:#?}");
    assert!(
        response["error"]
            .as_str()
            .unwrap()
            .contains("more than one"),
        "{response:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_switch_session_response_carries_reasoning_effort_like_its_three_siblings() {
    // Task 3 (pi-parity fix, serve pass 19): `switch_branch`/`fork`/`clone` all restore whichever
    // model/thinking-level the target point actually last recorded and echo the result as
    // `data.reasoning_effort` on their own response — `switch_session` runs the identical restoration
    // logic (see `Persistence::model_and_level_at_active`) but previously omitted the field from its
    // response entirely, forcing a client to make a separate `get_state` round trip after
    // `switch_session` alone. This proves the restored (non-default) level round-trips through the
    // response directly.
    let dir = tempfile::tempdir().unwrap();
    let session_dir = dir.path().join("sessions").to_string_lossy().into_owned();
    let (base, _bodies) =
        spawn_model_server(vec![turn_text("first answer"), turn_text("second answer")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_dir_cmd(bin, &base, &session_dir).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    let first_id = frames.last().unwrap()["data"]["session_id"]
        .as_str()
        .unwrap()
        .to_string();

    // `claude-test` supports reasoning and starts at "medium" (see
    // `serve_switch_branch_resets_thinking_level_instead_of_bleeding_a_sibling_branchs_setting`) — bump
    // it to a non-default rung, anchored (per `SessionStore::record_thinking_level_change`) at the tip
    // as of right now.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_reasoning_effort", "effort": "high" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_reasoning_effort");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    assert_eq!(frames.last().unwrap()["data"]["level"], "high");

    // A change is anchored *at* the tip it was recorded at, not retroactively covering it
    // (`SessionStore::change_at` deliberately excludes `target_id` itself — a change takes effect for
    // whatever comes *after* it, not the point it was recorded at) — so a second turn is needed here to
    // give the "high" change a real descendant to be visible *at*, the same way
    // `serve_switch_branch_restores_the_model_active_on_that_branch` does for `switch_branch`.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "confirm" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    // Switch away to a fresh session (starts at the process's own default level, not "high") …
    writeln!(stdin, "{}", json!({ "type": "new_session" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "new_session");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");

    // … then back to the first session: its own recorded "high" must be restored and echoed directly
    // on the `switch_session` response, matching `fork`/`clone`/`switch_branch`'s exact response shape.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "switch_session", "session_id": first_id })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "switch_session");
    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], true, "{resp:#?}");
    assert_eq!(
        resp["data"]["reasoning_effort"], "high",
        "switch_session must restore and echo the session's own recorded thinking level, matching \
         fork/clone/switch_branch, without a separate get_state round trip: {resp:#?}"
    );
    // The pre-existing fields must still be there too — this is additive, not a replacement.
    assert_eq!(resp["data"]["session_id"], first_id, "{resp:#?}");
    assert_eq!(resp["data"]["model"], "claude-test", "{resp:#?}");
    assert!(resp["data"]["cwd_stale"].is_boolean(), "{resp:#?}");

    drop(stdin);
    child.wait().unwrap();
}
