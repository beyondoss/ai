//! `serve` e2e: a listing is a **page**, and it is metadata.
//!
//! Both properties are about what a listing costs rather than what it says. A listing used to be
//! unbounded in cardinality *and* to carry up to 50 KB of each session's conversation text in every
//! entry — so one `list_sessions` asked the process to build a tenant's whole transcript set in
//! memory and write it to a single frame. In service mode the caller is a tenant holding one
//! session's grant, which makes that a request anybody can make.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufReader, Write};

use common::{SpawnGuarded, read_until_response, serve_dir_cmd, spawn_model_server, turn_text};
use serde_json::{Value, json};

/// Three sessions, each whose first user message is its own marker, listed newest-first.
fn three_sessions() -> (tempfile::TempDir, common::ChildGuard) {
    let dir = tempfile::tempdir().unwrap();
    let dir_str = dir.path().to_string_lossy().into_owned();
    let (base, _bodies) =
        spawn_model_server(vec![turn_text("ok"), turn_text("ok"), turn_text("ok")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_dir_cmd(bin, &base, &dir_str).spawn_guarded();
    {
        let mut stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        for (i, marker) in ["alpha-one", "beta-two", "gamma-three"].iter().enumerate() {
            if i > 0 {
                writeln!(stdin, "{}", json!({ "type": "new_session" })).unwrap();
                stdin.flush().unwrap();
                read_until_response(&mut stdout, "new_session");
            }
            writeln!(stdin, "{}", json!({ "type": "prompt", "message": marker })).unwrap();
            stdin.flush().unwrap();
            read_until_response(&mut stdout, "prompt");
        }
        child.stdin = Some(stdin);
        child.stdout = Some(stdout.into_inner());
    }
    (dir, child)
}

#[test]
fn a_listing_is_one_page_and_reports_the_total() {
    let (_dir, mut child) = three_sessions();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    fn page<W: Write, R: std::io::BufRead>(stdin: &mut W, stdout: &mut R, cmd: Value) -> Value {
        writeln!(stdin, "{cmd}").unwrap();
        stdin.flush().unwrap();
        read_until_response(stdout, "list_sessions").last().unwrap()["data"].clone()
    }

    // A page of one, out of three. `total` is the match set, not the page, so a client knows to ask
    // again without guessing.
    let data = page(
        &mut stdin,
        &mut stdout,
        json!({ "type": "list_sessions", "limit": 1 }),
    );
    assert_eq!(
        data["total"], 3,
        "total must count the whole match set: {data:#?}"
    );
    let first = data["sessions"].as_array().unwrap();
    assert_eq!(first.len(), 1, "limit must bound the page: {data:#?}");

    // `offset` walks it, and does not repeat what the first page already returned.
    let data = page(
        &mut stdin,
        &mut stdout,
        json!({ "type": "list_sessions", "limit": 1, "offset": 1 }),
    );
    let second = data["sessions"].as_array().unwrap();
    assert_eq!(second.len(), 1);
    assert_ne!(
        second[0]["id"], first[0]["id"],
        "offset must move the window, not re-serve page one: {data:#?}"
    );

    // An offset past the end is an empty page, not an error and not a wrap-around.
    let data = page(
        &mut stdin,
        &mut stdout,
        json!({ "type": "list_sessions", "limit": 10, "offset": 99 }),
    );
    assert_eq!(data["total"], 3);
    assert!(data["sessions"].as_array().unwrap().is_empty());

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn a_listing_entry_carries_metadata_and_not_the_transcript() {
    let (_dir, mut child) = three_sessions();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "list_sessions" })).unwrap();
    stdin.flush().unwrap();
    let data = read_until_response(&mut stdout, "list_sessions")
        .last()
        .unwrap()["data"]
        .clone();
    let sessions = data["sessions"].as_array().unwrap();
    assert_eq!(
        sessions.len(),
        3,
        "the default page covers three: {data:#?}"
    );

    for s in sessions {
        // The derived fields a picker needs are still here...
        assert!(s["preview"].is_string(), "preview must survive: {s:#?}");
        assert!(s["message_count"].as_u64().unwrap() > 0);
        assert!(s["updated_at"].as_u64().unwrap() > 0);
        // ...and the session's text is not. Matching happens server-side, so a client has never
        // needed the corpus to filter locally — and returning it made a listing's size the whole
        // transcript set rather than a page of metadata.
        assert!(
            s.get("search_text").is_none(),
            "a listing entry must not carry the session's text: {s:#?}"
        );
    }

    // And a query still narrows, over the metadata that remains.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "list_sessions", "query": "beta-two" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let data = read_until_response(&mut stdout, "list_sessions")
        .last()
        .unwrap()["data"]
        .clone();
    assert_eq!(data["total"], 1, "the query must still filter: {data:#?}");
    assert!(
        data["sessions"][0]["preview"]
            .as_str()
            .unwrap()
            .contains("beta-two")
    );

    drop(stdin);
    child.wait().unwrap();
}
