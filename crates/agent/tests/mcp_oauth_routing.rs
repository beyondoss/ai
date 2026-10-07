//! An MCP server with an `agent mcp-login` is spoken to over this crate's own POST path
//! (`mcp_wire::HttpClient::post_bounded`, which answers every streamable-HTTP POST) instead of
//! rmcp's client — so that path has to carry everything a streamable-HTTP server does: sessions
//! (`Mcp-Session-Id`, a 404 for an expired one), SSE responses, the standalone `GET` stream, the
//! legacy handshake's cues. Each test here drives a real dial (`agent run`) against the OAuth fixture
//! speaking that feature, and pins one piece of the routing.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufRead, Write};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use common::mcp_oauth_fixture::{echo, logged_in, run, tool_results};
use common::{SpawnGuarded, read_until_response, turn_text};
use serde_json::{Value, json};

/// `toolu_<n>`'s result in the model request that followed it: `(text, is_error)`.
fn result_of(bodies: &[String], request: usize, id: &str) -> (String, bool) {
    let results = tool_results(&bodies[request]);
    let (_, text, is_error) = results
        .into_iter()
        .find(|(i, ..)| i == id)
        .unwrap_or_else(|| panic!("no result for {id} in request {request}"));
    (text, is_error)
}

#[test]
fn a_401_carrying_a_json_rpc_error_body_refreshes_through_a_real_dial() {
    // rmcp's own client reads this 401 as an ordinary error *response*; only `post_bounded`
    // (every POST's path) sees the status, so the OAuth layer refreshes.
    let (home, fixture) = logged_in();
    fixture.reject_with_json_body.store(true, Ordering::SeqCst);
    fixture.revoke_after_calls.store(1, Ordering::SeqCst);
    let bodies = run(
        home.path(),
        vec![
            echo("toolu_1", "first"),
            echo("toolu_2", "after-revocation"),
            turn_text("done"),
        ],
    );
    let (text, is_error) = result_of(&bodies, 2, "toolu_2");
    assert!(
        !is_error && text.contains("after-revocation"),
        "the call after the revocation must succeed on a refreshed token: {text}"
    );
    assert_eq!(fixture.rejected_calls.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.refresh_grants.load(Ordering::SeqCst), 1);
}

#[test]
fn a_legacy_servers_4xx_to_server_discover_falls_back_to_initialize() {
    // A pre-2026-07-28 server may answer the discovery probe with a plain 400; rmcp reads that as
    // "no discover, use initialize" — so must the OAuth path, or the handshake fails.
    let (home, fixture) = logged_in();
    *fixture.discover_reply.lock().unwrap() =
        Some(("400 Bad Request".into(), "unknown method".into()));
    let bodies = run(
        home.path(),
        vec![echo("toolu_1", "legacy-handshake"), turn_text("done")],
    );
    let (text, is_error) = result_of(&bodies, 1, "toolu_1");
    assert!(
        !is_error && text.contains("legacy-handshake"),
        "the server must be connected through the legacy handshake: {text}"
    );
    assert_eq!(fixture.initializes.load(Ordering::SeqCst), 1);
}

#[test]
fn an_empty_200_to_a_notification_is_an_acceptance() {
    // `notifications/initialized` answered by an empty `200 OK` (no content type) rather than 202:
    // rmcp accepts it, and so must the OAuth path, or the handshake fails.
    let (home, fixture) = logged_in();
    fixture
        .notifications_empty_200
        .store(true, Ordering::SeqCst);
    let bodies = run(
        home.path(),
        vec![echo("toolu_1", "empty-200"), turn_text("done")],
    );
    let (text, is_error) = result_of(&bodies, 1, "toolu_1");
    assert!(
        !is_error && text.contains("empty-200"),
        "the server must be connected: {text}"
    );
}

#[test]
fn sessions_sse_responses_and_the_standalone_stream_work_for_an_oauth_server() {
    let (home, fixture) = logged_in();
    fixture.sessions.store(true, Ordering::SeqCst);
    fixture.sse_responses.store(true, Ordering::SeqCst);
    let bodies = run(
        home.path(),
        vec![
            echo("toolu_1", "over-sse-1"),
            echo("toolu_2", "over-sse-2"),
            turn_text("done"),
        ],
    );
    for (request, id, text) in [(1, "toolu_1", "over-sse-1"), (2, "toolu_2", "over-sse-2")] {
        let (got, is_error) = result_of(&bodies, request, id);
        assert!(!is_error && got.contains(text), "{id}: {got}");
    }
    assert_eq!(
        fixture.session_refusals.load(Ordering::SeqCst),
        0,
        "every POST after initialize carried the live session"
    );
    let sessions = fixture.call_sessions.lock().unwrap().clone();
    assert_eq!(
        sessions,
        [Some("session-0".to_owned()), Some("session-0".to_owned())]
    );
    assert!(
        fixture.get_streams.load(Ordering::SeqCst) >= 1,
        "the standalone SSE stream was opened, with the bearer and the session"
    );
}

#[test]
fn closing_an_oauth_servers_connection_ends_its_session_with_an_authenticated_delete() {
    // The idle reaper closes the connection a second after the call: rmcp ends the session with a
    // DELETE, which must carry the bearer (through the OAuth layer) and the session id.
    let (home, fixture) = logged_in();
    fixture.sessions.store(true, Ordering::SeqCst);
    let (base, _bodies) = common::spawn_model_server(vec![echo("toolu_1", "x"), turn_text("done")]);
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let mut child = common::serve_cmd(common::BIN, &base, &session_file)
        .env("HOME", home.path())
        .env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "1")
        .spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = common::child_frames(&mut child);
    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "go" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    let deadline = Instant::now() + Duration::from_secs(15);
    while fixture.deletes.load(Ordering::SeqCst) == 0 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(100));
    }
    drop(stdin);
    child.wait().unwrap();
    assert_eq!(
        fixture.deletes.load(Ordering::SeqCst),
        1,
        "the reaped connection ends its session with a DELETE"
    );
    assert_eq!(
        *fixture.delete_requests.lock().unwrap(),
        [(
            Some("Bearer access-token-0".to_owned()),
            Some("session-0".to_owned())
        )],
        "the DELETE carries the bearer and the session id"
    );
}

#[test]
fn an_agent_run_ends_its_oauth_session_with_one_authenticated_delete_before_it_exits() {
    // A process that simply exits closes nothing on its own; the exit sweep ends every HTTP
    // session still open (`mcp_http_exit`), so by the time `run` has exited the DELETE is in.
    let (home, fixture) = logged_in();
    fixture.sessions.store(true, Ordering::SeqCst);
    let bodies = run(home.path(), vec![echo("toolu_1", "x"), turn_text("done")]);
    let (text, is_error) = result_of(&bodies, 1, "toolu_1");
    assert!(!is_error && text.contains('x'), "{text}");
    assert_eq!(
        *fixture.delete_requests.lock().unwrap(),
        [(
            Some("Bearer access-token-0".to_owned()),
            Some("session-0".to_owned())
        )],
        "exactly one DELETE, carrying the bearer and the session id, before the process exited"
    );
    assert_eq!(
        fixture.deletes.load(Ordering::SeqCst),
        1,
        "and it was accepted"
    );
}

#[test]
fn an_exit_whose_delete_is_never_answered_stays_bounded() {
    let (home, fixture) = logged_in();
    fixture.sessions.store(true, Ordering::SeqCst);
    let started = Instant::now();
    run(home.path(), vec![echo("toolu_1", "x"), turn_text("done")]);
    let baseline = started.elapsed();

    let (home, fixture2) = logged_in();
    fixture2.sessions.store(true, Ordering::SeqCst);
    fixture2.hang_deletes.store(true, Ordering::SeqCst);
    let started = Instant::now();
    run(home.path(), vec![echo("toolu_1", "x"), turn_text("done")]);
    let hung = started.elapsed();
    assert_eq!(
        fixture2.delete_requests.lock().unwrap().len(),
        1,
        "the DELETE was sent"
    );
    // The server holds the DELETE for 30 s; the exit waits at most its deadline (1.5 s).
    assert!(
        hung < baseline + Duration::from_secs(4),
        "an unanswered DELETE must not hold the exit: {hung:?} (baseline {baseline:?})"
    );
    drop(fixture);
}

#[test]
fn an_oauth_servers_in_post_question_is_routed_to_the_client_and_answered_mid_stream() {
    // The fixture answers `ask_user` on the POST's own SSE stream, flushed event by event: a
    // progress notification, then an `elicitation/create` to the client — and holds the response
    // until the client's answer has come back on another POST. So the call only completes with
    // the answer if each event was handled as it arrived, before the stream ended.
    let (home, fixture) = logged_in();
    fixture.sessions.store(true, Ordering::SeqCst);
    fixture.sse_responses.store(true, Ordering::SeqCst);
    let (base, _bodies) = common::spawn_model_server(vec![
        common::turn_tool_use("toolu_e", "mcp__protected__ask_user", "{}"),
        turn_text("done"),
    ]);
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let mut child = common::serve_cmd(common::BIN, &base, &session_file)
        .env("HOME", home.path())
        .env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0")
        .spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = common::child_frames(&mut child);
    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "ask" })).unwrap();
    stdin.flush().unwrap();

    // Frames up to the question: the progress event must already be among them.
    let mut before_question = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut line = String::new();
    let request_id = loop {
        assert!(
            Instant::now() < deadline,
            "no elicitation_request: {before_question:?}"
        );
        line.clear();
        assert_ne!(stdout.read_line(&mut line).unwrap(), 0, "serve exited");
        let Ok(frame) = serde_json::from_str::<Value>(line.trim()) else {
            continue;
        };
        if frame["type"] == "elicitation_request" {
            assert_eq!(frame["server"], "protected");
            break frame["request_id"].as_str().unwrap().to_owned();
        }
        before_question.push(frame);
    };
    assert!(
        before_question.iter().any(|f| f["type"] == "event"
            && f["event"]["kind"] == "tool_progress"
            && f.to_string().contains("asking the user")),
        "the stream's first event reached the client before its last was sent: {before_question:?}"
    );

    writeln!(
        stdin,
        "{}",
        json!({ "type": "elicit", "request_id": request_id, "action": "accept",
                "content": { "name": "Ferris" } })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    let tool_text: String = frames
        .iter()
        .filter(|f| f["type"] == "event" && f["event"]["kind"] == "tool_end")
        .filter_map(|f| f["event"]["result"].as_str())
        .collect();
    assert!(
        tool_text.contains("hello-Ferris"),
        "the answer reached the server mid-stream and the call completed with it: {tool_text:?}"
    );
    let answers = fixture.client_answers.lock().unwrap().clone();
    assert_eq!(answers.len(), 1, "{answers:?}");
    assert_eq!(answers[0]["result"]["action"], "accept");
}

#[test]
fn an_expired_session_is_reinitialized_and_the_call_retried() {
    // The 404 for an expired session is rmcp's `SessionExpired`: it re-initializes and retries.
    let (home, fixture) = logged_in();
    fixture.sessions.store(true, Ordering::SeqCst);
    fixture
        .expire_sessions_after_calls
        .store(1, Ordering::SeqCst);
    let bodies = run(
        home.path(),
        vec![
            echo("toolu_1", "first"),
            echo("toolu_2", "after-expiry"),
            turn_text("done"),
        ],
    );
    let (text, is_error) = result_of(&bodies, 2, "toolu_2");
    assert!(
        !is_error && text.contains("after-expiry"),
        "the call after the session expired must succeed on a new session: {text}"
    );
    assert_eq!(fixture.initializes.load(Ordering::SeqCst), 2);
    assert!(fixture.session_refusals.load(Ordering::SeqCst) >= 1);
    let sessions = fixture.call_sessions.lock().unwrap().clone();
    assert_eq!(
        sessions.last().cloned().flatten().as_deref(),
        Some("session-1")
    );
}
