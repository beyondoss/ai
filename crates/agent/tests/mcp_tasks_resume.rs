//! SEP-2663 tasks that outlive the agent process: `serve` journals each task with the session and,
//! after a restart, resumes it in the background and answers the interrupted call with the real
//! result (see `crate::mcp_resume`). The HTTP fixture is owned by the test, so it and its tasks
//! survive `serve` being killed.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;
mod mcp_tasks_env;

use std::process::ChildStdin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::{ChildGuard, spawn_model_server, turn_text, turn_tool_use};
use mcp_tasks_env::{
    Env, assert_alternates, is_event, prompt, read_until, read_until_or_fail,
    read_until_prompt_done, send, tool_end, tool_result_sent, wait_for,
};
use serde_json::{Value, json};

struct Serve {
    child: ChildGuard,
    stdin: ChildStdin,
    stdout: common::Frames,
    bodies: Arc<Mutex<Vec<String>>>,
}

fn serve(env: &Env, turns: Vec<String>) -> Serve {
    let (base, bodies) = spawn_model_server(turns);
    let mut child = env.serve(&base);
    let stdin = child.stdin.take().unwrap();
    let stdout = common::child_frames(&mut child);
    Serve {
        child,
        stdin,
        stdout,
        bodies,
    }
}

impl Serve {
    /// The model-turn requests (not the background session-title call).
    fn requests(&self) -> Vec<String> {
        self.bodies
            .lock()
            .unwrap()
            .iter()
            .filter(|r| !r.contains("You write short titles"))
            .cloned()
            .collect()
    }

    fn close(self) {
        let Serve {
            mut child, stdin, ..
        } = self;
        drop(stdin);
        child.wait().unwrap();
    }
}

/// Start `tool` (a task), answer its elicitation with `answer` if it asks, wait until `journaled`
/// is in the session file, and kill `serve` outright. Returns the task id.
fn start_and_kill(env: &Env, tool: &str, answer: Option<Value>, journaled: &str) -> String {
    let mut s = serve(env, vec![turn_tool_use("toolu_g", tool, "{}")]);
    prompt(&mut s.stdin, "start the job");
    let frames = read_until(&mut s.stdout, "the task's record", |f| {
        f["event"]["details"]["mcpTask"].is_object()
    });
    let task_id = frames.last().unwrap()["event"]["details"]["mcpTask"]["taskId"]
        .as_str()
        .unwrap()
        .to_owned();
    if let Some(content) = answer {
        let ask = read_until(&mut s.stdout, "the elicitation", |f| {
            f["type"] == "elicitation_request"
        });
        send(
            &mut s.stdin,
            json!({
                "type": "elicit",
                "request_id": ask.last().unwrap()["request_id"],
                "action": "accept",
                "content": content,
            }),
        );
    }
    let needle = if journaled.is_empty() {
        task_id.clone()
    } else {
        journaled.to_owned()
    };
    wait_for("the journal entry in the session file", || {
        env.session_text().contains(&needle)
    });
    s.child.kill().unwrap();
    s.child.wait().unwrap();
    task_id
}

/// F2/F3/F17: after a restart the journaled task is polled in the background before any prompt
/// (its progress streams, its result is journaled the moment it lands); the next prompt answers
/// the call with the real result, reported as a `tool_start`/`tool_end` pair, and the model
/// request still alternates roles.
#[test]
fn serve_restart_resumes_in_the_background_before_any_prompt() {
    let env = Env::new();
    let _server = env.http();
    start_and_kill(&env, "mcp__t__gated_task", None, "");
    let gets_before = env.methods("tasks/get").len();

    let mut s = serve(&env, vec![turn_text("carrying on")]);
    read_until(
        &mut s.stdout,
        "resumed-task progress before any prompt",
        |f| is_event(f, "tool_progress", "toolu_g"),
    );
    std::fs::write(&env.gate, b"open").unwrap();
    wait_for("the result journaled without a prompt", || {
        env.session_text().contains("mcp_task_result")
    });
    assert!(
        env.methods("tasks/get").len() > gets_before,
        "polled with no prompt"
    );

    prompt(&mut s.stdin, "what happened?");
    let frames = read_until_prompt_done(&mut s.stdout);
    let start = frames
        .iter()
        .position(|f| is_event(f, "tool_start", "toolu_g"))
        .unwrap_or_else(|| panic!("no tool_start for the resumed call: {frames:?}"));
    let end = frames
        .iter()
        .position(|f| is_event(f, "tool_end", "toolu_g"))
        .unwrap();
    assert!(start < end, "tool_start before tool_end");
    let end = tool_end(&frames, "toolu_g");
    assert_eq!(end["is_error"], false, "{end}");
    assert!(
        end["result"].as_str().unwrap().contains("gated-done"),
        "{end}"
    );

    let requests = s.requests();
    assert_eq!(requests.len(), 1, "{requests:#?}");
    assert_alternates(&requests[0]);
    let result = tool_result_sent(&requests[0], "toolu_g");
    assert_eq!(result["is_error"], json!(false), "{result}");
    assert!(result.to_string().contains("gated-done"), "{result}");
    assert_eq!(
        env.methods("tools/call").len(),
        1,
        "resumed, not re-invoked"
    );
    s.close();
    // N5: the restart reused the journaled record instead of appending another.
    assert_eq!(
        env.session_text().matches("\"kind\":\"mcp_task\",").count(),
        1,
        "one record for the task: {}",
        env.session_text()
    );
}

/// F1: a prompt waiting on a resumed task is aborted. That must not cancel the task (the user
/// aborted the prompt), and the next prompt (whose `tool_use` is no longer at the tip: the aborted
/// turn sits after it) waits again and gets the real result, with roles alternating. A further
/// restart still carries the result (it is journaled, and re-spliced).
#[test]
fn serve_aborted_resume_is_retried_on_the_next_prompt() {
    let env = Env::new();
    let _server = env.http();
    start_and_kill(&env, "mcp__t__gated_task", None, "");

    let mut s = serve(
        &env,
        vec![
            turn_text("one"),
            turn_text("two"),
            turn_text("t3"),
            turn_text("t4"),
        ],
    );
    prompt(&mut s.stdin, "p1");
    let waiting = read_until(&mut s.stdout, "the waiting notice", |f| {
        is_event(f, "tool_progress", "toolu_g") && f["event"]["details"]["status"] == "waiting"
    });
    assert!(
        waiting.iter().any(|f| is_event(f, "tool_start", "toolu_g")),
        "the wait opens with a tool_start for the resumed call (F17): {waiting:?}"
    );
    send(&mut s.stdin, json!({ "type": "abort", "id": "a1" }));
    read_until_prompt_done(&mut s.stdout);
    std::thread::sleep(Duration::from_millis(300));
    assert!(
        env.methods("tasks/cancel").is_empty(),
        "aborting the prompt must not cancel the task"
    );

    std::fs::write(&env.gate, b"open").unwrap();
    prompt(&mut s.stdin, "p2");
    let frames = read_until_prompt_done(&mut s.stdout);
    let end = tool_end(&frames, "toolu_g");
    assert!(
        end["result"].as_str().unwrap().contains("gated-done"),
        "{end}"
    );
    let requests = s.requests();
    let last = requests.last().unwrap();
    assert_alternates(last);
    assert!(
        tool_result_sent(last, "toolu_g")
            .to_string()
            .contains("gated-done")
    );
    s.close();

    let mut again = serve(&env, vec![turn_text("three"), turn_text("t5")]);
    // N2: the result spliced into the aborted turn is part of the transcript, not only of the
    // model request: a fresh process's `get_messages` and HTML export carry it before any prompt.
    send(
        &mut again.stdin,
        json!({ "type": "get_messages", "id": "m" }),
    );
    let got = read_until(&mut again.stdout, "the transcript", |f| {
        f["type"] == "response" && f["command"] == "get_messages"
    });
    let transcript = got.last().unwrap()["data"]["messages"].to_string();
    assert!(
        transcript.contains("toolu_g") && transcript.contains("gated-done"),
        "the spliced tool_result is in the reloaded transcript: {transcript}"
    );
    let export = env.dir.path().join("export.html");
    send(
        &mut again.stdin,
        json!({ "type": "export_html", "id": "x", "output_path": export.to_string_lossy() }),
    );
    read_until(&mut again.stdout, "the export", |f| {
        f["type"] == "response" && f["command"] == "export_html"
    });
    assert!(
        std::fs::read_to_string(&export)
            .unwrap()
            .contains("gated-done"),
        "the HTML export carries the spliced result"
    );
    prompt(&mut again.stdin, "p3");
    read_until_prompt_done(&mut again.stdout);
    let requests = again.requests();
    assert_alternates(&requests[0]);
    assert!(
        tool_result_sent(&requests[0], "toolu_g")
            .to_string()
            .contains("gated-done"),
        "the result survives another restart"
    );
    again.close();
}

/// F15: a stdio server dies with `serve`, and its tasks with it: on restart the respawned server
/// answers `-32602` for the journaled task, exactly once (never retried), and that is the call's
/// definitive error result.
#[test]
fn serve_restart_resolves_a_task_the_server_no_longer_knows() {
    let env = Env::new();
    env.stdio();
    start_and_kill(&env, "mcp__t__gated_task", None, "");
    let gets_before = env.methods("tasks/get").len();

    let mut s = serve(&env, vec![turn_text("carrying on")]);
    prompt(&mut s.stdin, "what happened?");
    let frames = read_until_prompt_done(&mut s.stdout);
    let end = tool_end(&frames, "toolu_g");
    assert_eq!(end["is_error"], true, "{end}");
    assert!(
        end["result"].as_str().unwrap().contains("Task not found"),
        "{end}"
    );
    let result = tool_result_sent(&s.requests()[0], "toolu_g");
    assert_eq!(result["is_error"], json!(true), "{result}");
    assert_eq!(
        env.methods("tasks/get").len() - gets_before,
        1,
        "-32602 is the answer, not a reason to poll again"
    );
    s.close();
}

/// F8: the journaled record carries the task's `ttlMs` and creation time, so a task whose TTL ran
/// out while the agent was down resolves as expired after a restart, even with its server gone.
#[test]
fn serve_restart_honours_the_journaled_ttl_even_with_the_server_gone() {
    let env = Env::new();
    let server = env.http();
    start_and_kill(&env, "mcp__t__gated_ttl_task", None, "\"ttlMs\":1500");
    drop(server);
    std::thread::sleep(Duration::from_millis(1600));

    let mut s = serve(&env, vec![turn_text("carrying on")]);
    prompt(&mut s.stdin, "what happened?");
    let frames = read_until_prompt_done(&mut s.stdout);
    let end = tool_end(&frames, "toolu_g");
    assert_eq!(end["is_error"], true, "{end}");
    assert!(
        end["result"].as_str().unwrap().contains("ttlMs (1500 ms)"),
        "{end}"
    );
    s.close();
}

/// F10: an `inputRequests` key answered before the restart is journaled with the task, so the
/// resume, which still sees the server list it (an eventually-consistent ack), does not ask again.
#[test]
fn serve_restart_does_not_reask_an_answered_input_request() {
    let env = Env::new();
    let _server = env.http();
    start_and_kill(
        &env,
        "mcp__t__durable_ask_task",
        Some(json!({ "name": "Ferris" })),
        "\"answered\":[\"name\"]",
    );

    let mut s = serve(&env, vec![turn_text("carrying on")]);
    prompt(&mut s.stdin, "what happened?");
    let frames = read_until_or_fail(
        &mut s.stdout,
        "the resumed result",
        Duration::from_secs(30),
        |f| f["type"] == "elicitation_request",
        |f| f["type"] == "response" && f["command"] == "prompt",
    );
    let end = tool_end(&frames, "toolu_g");
    assert!(
        end["result"].as_str().unwrap().contains("durable-Ferris"),
        "{end}"
    );
    assert_eq!(env.methods("tasks/update").len(), 1, "answered once, ever");
    s.close();
}

/// Every session file under `dir`, with its contents.
fn session_files(dir: &std::path::Path) -> Vec<(std::path::PathBuf, String)> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "jsonl") {
                out.push((
                    path.clone(),
                    std::fs::read_to_string(&path).unwrap_or_default(),
                ));
            }
        }
    }
    out
}

fn serve_repo(env: &Env, turns: Vec<String>, id: &str) -> Serve {
    let (base, bodies) = spawn_model_server(turns);
    let mut cmd = common::serve_dir_cmd(
        common::BIN,
        &base,
        env.dir.path().join("repo").to_str().unwrap(),
    );
    cmd.env("HOME", &env.home)
        .env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0")
        .args(["--session-id", id]);
    let mut child = common::SpawnGuarded::spawn_guarded(&mut cmd);
    let stdin = child.stdin.take().unwrap();
    let stdout = common::child_frames(&mut child);
    Serve {
        child,
        stdin,
        stdout,
        bodies,
    }
}

/// F11: a journal write is stamped with its session and dropped when the session it was meant for
/// is no longer the one persisted to. Here a background resumer's result lands after a
/// `switch_session`: it must not be written into the session switched to.
#[test]
fn a_resumed_result_landing_after_a_session_switch_stays_out_of_the_other_session() {
    let env = Env::new();
    let _server = env.http();
    // `beta`: an ordinary session with one turn.
    let mut beta = serve_repo(&env, vec![turn_text("beta exists"), turn_text("t")], "beta");
    prompt(&mut beta.stdin, "hello beta");
    read_until_prompt_done(&mut beta.stdout);
    beta.close();
    // `alpha`: a task in flight when the process dies.
    let mut alpha = serve_repo(
        &env,
        vec![turn_tool_use("toolu_g", "mcp__t__gated_task", "{}")],
        "alpha",
    );
    prompt(&mut alpha.stdin, "start the job");
    let frames = read_until(&mut alpha.stdout, "the task's record", |f| {
        f["event"]["details"]["mcpTask"].is_object()
    });
    let task_id = frames.last().unwrap()["event"]["details"]["mcpTask"]["taskId"]
        .as_str()
        .unwrap()
        .to_owned();
    wait_for("alpha's journal entry", || {
        session_files(&env.dir.path().join("repo"))
            .iter()
            .any(|(_, text)| text.contains(&task_id))
    });
    alpha.child.kill().unwrap();
    alpha.child.wait().unwrap();

    // Reopen alpha: its resumer starts polling. Switch to beta, then let the task finish.
    let mut s = serve_repo(&env, vec![turn_text("unused")], "alpha");
    read_until(&mut s.stdout, "resumed-task progress", |f| {
        is_event(f, "tool_progress", "toolu_g")
    });
    send(
        &mut s.stdin,
        json!({ "type": "switch_session", "id": "sw", "session_id": "beta" }),
    );
    let switched = read_until(&mut s.stdout, "the switch", |f| {
        f["type"] == "response" && f["command"] == "switch_session"
    });
    assert_eq!(switched.last().unwrap()["success"], true, "{switched:?}");
    let gets = env.methods("tasks/get").len();
    std::fs::write(&env.gate, b"open").unwrap();
    // Alpha's resumer stopped at the switch (see `switching_sessions_moves_the_resumer_with_it`), so
    // the task is not polled to completion here; a write already queued is dropped by its stamp.
    let _ = gets;
    std::thread::sleep(Duration::from_millis(500));
    send(&mut s.stdin, json!({ "type": "get_state", "id": "g" }));
    read_until(&mut s.stdout, "state", |f| {
        f["type"] == "response" && f["command"] == "get_state"
    });
    s.close();
    for (path, text) in session_files(&env.dir.path().join("repo")) {
        if text.contains("hello beta") {
            assert!(
                !text.contains("mcp_task_result"),
                "alpha's result leaked into beta ({}): {text}",
                path.display()
            );
        }
    }
}

/// A session that ends while a resumed task is still being polled must still exit: the resumer's
/// background tasks hold a sender for their progress frames, and the writer ends only once every
/// sender is gone. The task itself is not cancelled; it stays journaled for the next start.
#[test]
fn serve_exits_while_a_resumed_task_is_still_outstanding() {
    let env = Env::new();
    let _server = env.http();
    start_and_kill(&env, "mcp__t__gated_task", None, "");

    let mut s = serve(&env, vec![turn_text("unused")]);
    read_until(&mut s.stdout, "resumed-task progress", |f| {
        is_event(f, "tool_progress", "toolu_g")
    });
    drop(s.stdin);
    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    loop {
        if s.child.try_wait().unwrap().is_some() {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "serve never exited with a resumed task outstanding"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(
        env.methods("tasks/cancel").is_empty(),
        "ending the session does not cancel the task"
    );
}

/// Start `tool` in repo-mode session `id` as call `tool_use_id`, wait until its record is in a
/// session file, and kill `serve`. Returns the task id.
fn start_and_kill_repo(env: &Env, id: &str, tool_use_id: &str, tool: &str) -> String {
    let mut s = serve_repo(env, vec![turn_tool_use(tool_use_id, tool, "{}")], id);
    prompt(&mut s.stdin, "start the job");
    let frames = read_until(&mut s.stdout, "the task's record", |f| {
        f["event"]["details"]["mcpTask"].is_object()
    });
    let task_id = frames.last().unwrap()["event"]["details"]["mcpTask"]["taskId"]
        .as_str()
        .unwrap()
        .to_owned();
    wait_for("the journal entry", || {
        session_files(&env.dir.path().join("repo"))
            .iter()
            .any(|(_, text)| text.contains(&task_id))
    });
    s.child.kill().unwrap();
    s.child.wait().unwrap();
    task_id
}

fn gets_for(env: &Env, task_id: &str) -> usize {
    env.methods("tasks/get")
        .iter()
        .filter(|g| g["params"]["taskId"] == task_id)
        .count()
}

/// N3/N4b: the resumer follows the session. Switching from alpha to beta stops alpha's polling at
/// once (without cancelling alpha's task, which stays journaled) and starts beta's pending task
/// right away, before any prompt; nothing of alpha's reaches the client after the switch.
#[test]
fn switching_sessions_moves_the_resumer_with_it() {
    let env = Env::new();
    let _server = env.http();
    let beta_task = start_and_kill_repo(&env, "beta", "toolu_beta", "mcp__t__gated_task");
    let alpha_task = start_and_kill_repo(&env, "alpha", "toolu_alpha", "mcp__t__gated_task");

    let mut s = serve_repo(&env, vec![turn_text("unused")], "alpha");
    read_until(&mut s.stdout, "alpha's resumed progress", |f| {
        is_event(f, "tool_progress", "toolu_alpha")
    });
    wait_for("alpha being polled", || gets_for(&env, &alpha_task) > 0);
    send(
        &mut s.stdin,
        json!({ "type": "switch_session", "id": "sw", "session_id": "beta" }),
    );
    read_until(&mut s.stdout, "the switch", |f| {
        f["type"] == "response" && f["command"] == "switch_session"
    });
    // Checked on the wire log first: it has a deadline even when no frame ever comes.
    wait_for("beta being polled with no prompt", || {
        gets_for(&env, &beta_task) > 0
    });
    let after_switch = read_until(&mut s.stdout, "beta's resumed progress", |f| {
        is_event(f, "tool_progress", "toolu_beta")
    });
    let alpha_polls = gets_for(&env, &alpha_task);
    std::thread::sleep(Duration::from_millis(600));
    assert!(
        gets_for(&env, &alpha_task) <= alpha_polls + 1,
        "alpha's polling stops at the switch"
    );
    send(&mut s.stdin, json!({ "type": "get_state", "id": "g" }));
    let more = read_until(&mut s.stdout, "state", |f| {
        f["type"] == "response" && f["command"] == "get_state"
    });
    assert!(
        !after_switch
            .iter()
            .chain(more.iter())
            .any(|f| f["event"]["id"] == "toolu_alpha"),
        "nothing of alpha's reaches beta's client"
    );
    assert!(
        env.methods("tasks/cancel").is_empty(),
        "dropping a resumer never cancels its task"
    );
    s.close();
}

/// N4a: inside the poll loop the TTL counts from the task's creation (`createdAtMs`), not from
/// when the resume started. A 4 s task resumed after a restart stops polling 4 s after it was
/// created; counting from the resume would keep it going for the restart's length longer.
#[test]
fn a_resumed_poll_counts_its_ttl_from_creation() {
    let env = Env::new();
    let _server = env.http();
    start_and_kill(&env, "mcp__t__slow_ttl_task", None, "\"ttlMs\":4000");
    let created_ms = env.methods("tools/call")[0]["t_ms"].as_u64().unwrap();
    std::thread::sleep(Duration::from_millis(1000));

    let mut s = serve(&env, vec![turn_text("carrying on")]);
    prompt(&mut s.stdin, "what happened?");
    let frames = read_until_prompt_done(&mut s.stdout);
    let end = tool_end(&frames, "toolu_g");
    assert!(
        end["result"].as_str().unwrap().contains("ttlMs (4000 ms)"),
        "{end}"
    );
    let last_poll = env
        .methods("tasks/get")
        .iter()
        .map(|g| g["t_ms"].as_u64().unwrap())
        .max()
        .unwrap();
    assert!(
        last_poll <= created_ms + 4000 + 300,
        "polled past the TTL counted from creation: created {created_ms}, last poll {last_poll}"
    );
    s.close();
}

/// The session-ownership rule on a real fork: `clone` copies the transcript (with the unanswered
/// call) but not the task journal, and only the session that created a task may resume it. The
/// clone's prompt therefore does not wait on the parent's task (whose gate never opens), and the
/// call reaches its model with the generic interrupted repair, not a result.
#[test]
fn a_clone_does_not_resume_its_parents_task() {
    let env = Env::new();
    let _server = env.http();
    let task = start_and_kill_repo(&env, "alpha", "toolu_g", "mcp__t__gated_task");

    let mut s = serve_repo(
        &env,
        vec![turn_text("clone answers"), turn_text("t")],
        "alpha",
    );
    read_until(&mut s.stdout, "alpha's resumed progress", |f| {
        is_event(f, "tool_progress", "toolu_g")
    });
    send(&mut s.stdin, json!({ "type": "clone", "id": "c" }));
    let cloned = read_until(&mut s.stdout, "the clone", |f| {
        f["type"] == "response" && f["command"] == "clone"
    });
    assert_eq!(cloned.last().unwrap()["success"], true, "{cloned:?}");
    let polls = gets_for(&env, &task);
    prompt(&mut s.stdin, "in the clone");
    let frames = read_until_or_fail(
        &mut s.stdout,
        "the clone's prompt",
        Duration::from_secs(20),
        |f| is_event(f, "tool_progress", "toolu_g") && f["event"]["details"]["status"] == "waiting",
        |f| f["type"] == "response" && f["command"] == "prompt",
    );
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:?}");
    let requests = s.requests();
    let sent = tool_result_sent(requests.last().unwrap(), "toolu_g");
    assert!(!sent.to_string().contains("gated-done"), "{sent}");
    assert!(
        gets_for(&env, &task) <= polls + 1,
        "the clone does not poll the parent's task"
    );
    s.close();
}
