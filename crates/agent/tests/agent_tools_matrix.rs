//! `@beyond.dev/tools` through the production path, one E2B microVM per session:
//! **beyond-agent (`serve --service`) -> sandbox MCP connector -> exec front -> compiled `tools-mcp`
//! in the VM -> tools.**
//!
//! Driven by `deploy/demo/tools_matrix_e2e.sh`, which builds the binary, stages the bundles, starts
//! the exec front (`deploy/demo/front.py`) and exports the `MATRIX_*` variables below. The model is
//! scripted (so model requests can be counted) except in `r14_*`, which uses a real cheap model.
//! Skips when `MATRIX_FRONT` is unset. Run with `--test-threads=1`: rows share VMs on purpose, and
//! the first test to touch a session is the one that creates its VM.
#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::process::Command;
use std::time::{Duration, Instant};

use beyond_test_support::bundle::BundleServer;
use common::grant::{Bundle, Secrets};
use common::service::{Options, Service};
use common::{
    SESSION_TITLE_MARKER, TestWs, advertised_tools, spawn_model_server, turn_text, turn_tool_use,
    ws_connect_with_headers, ws_next_frame_within, ws_read_until_response, ws_send,
};
use serde_json::{Value, json};

struct Env {
    front: String,
    tools_url: String,
    with_path: String,
    with_url: String,
    without_path: String,
    without_url: String,
    front_log: String,
    vmcount: String,
    run: String,
}

fn env() -> Option<Env> {
    let v = |k: &str| std::env::var(k).ok().filter(|s| !s.is_empty());
    Some(Env {
        front: v("MATRIX_FRONT")?,
        tools_url: v("MATRIX_TOOLS_URL")?,
        with_path: v("MATRIX_BUNDLE_WITH_PATH")?,
        with_url: v("MATRIX_BUNDLE_WITH_URL")?,
        without_path: v("MATRIX_BUNDLE_WITHOUT_PATH")?,
        without_url: v("MATRIX_BUNDLE_WITHOUT_URL")?,
        front_log: v("MATRIX_FRONT_LOG")?,
        vmcount: v("MATRIX_VMCOUNT")?,
        run: v("MATRIX_RUN")?,
    })
}

macro_rules! env_or_skip {
    () => {
        match env() {
            Some(e) => e,
            None => {
                eprintln!("skip: set MATRIX_* (see deploy/demo/tools_matrix_e2e.sh)");
                return;
            }
        }
    };
}

#[derive(Clone, Copy, PartialEq)]
enum Menu {
    Saved,
    Live,
}

impl Env {
    /// The front session names the VM: one per name. `kind` keeps the two bundles apart.
    fn session(&self, name: &str) -> String {
        format!("tm{}{}", self.run, name)
    }
    fn bundle(&self, menu: Menu) -> (Vec<u8>, &str) {
        match menu {
            Menu::Saved => (std::fs::read(&self.with_path).unwrap(), &self.with_url),
            Menu::Live => (
                std::fs::read(&self.without_path).unwrap(),
                &self.without_url,
            ),
        }
    }
    fn mcp_url(&self, session: &str, menu: Menu) -> String {
        let (_, url) = self.bundle(menu);
        format!(
            "{}/mcp/{}?bundle={}&tools={}",
            self.front,
            session,
            enc(url),
            enc(&self.tools_url)
        )
    }
    /// `vm_create_start` events the front logged for `session`.
    fn front_creates(&self, session: &str) -> usize {
        std::fs::read_to_string(&self.front_log)
            .unwrap_or_default()
            .lines()
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .filter(|v| v["event"] == "vm_create_start" && v["detail"]["session"] == session)
            .count()
    }
    /// E2B's own listing (running or paused) for the session's metadata.
    fn e2b_vms(&self, session: &str) -> usize {
        let out = Command::new("bash")
            .arg(&self.vmcount)
            .arg(session)
            .output()
            .expect("vmcount");
        assert!(
            out.status.success(),
            "vmcount: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).trim().parse().unwrap()
    }
}

fn enc(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect()
}

/// Model requests that are not the session-title side call.
fn model_calls(requests: &[String]) -> Vec<String> {
    requests
        .iter()
        .filter(|r| !r.contains(SESSION_TITLE_MARKER))
        .cloned()
        .collect()
}

async fn service(base: &str) -> (Service, tempfile::TempDir) {
    let cache = tempfile::tempdir().unwrap();
    let svc = Service::start_with(
        base,
        &["s1"],
        Options {
            // The front is a loopback address here; in production the grant's own origin is allowed.
            extra_args: vec!["--mcp-allow-private".into()],
            env: vec![("TMPDIR".into(), cache.path().to_string_lossy().into_owned())],
            ..Default::default()
        },
    )
    .await;
    (svc, cache)
}

struct Opened {
    ws: TestWs,
    _server: BundleServer,
}

async fn open(
    e: &Env,
    svc: &Service,
    front_session: &str,
    ws_session: &str,
    menu: Menu,
    sandbox: bool,
) -> Opened {
    let (bytes, _) = e.bundle(menu);
    let server = BundleServer::start(bytes.clone());
    let mcp = e.mcp_url(front_session, menu);
    let token = svc.token_with("t1", ws_session, |c| {
        c.mcp = vec![("tools".into(), mcp.clone())];
        if sandbox {
            c.mcp_sandbox = vec!["tools".into()];
        }
        c.bundle = Some(Bundle {
            url: server.url.clone(),
            sha256: beyond_test_support::bundle::sha256_hex(&bytes),
        });
    });
    let ws = ws_connect_with_headers(svc.port, Some(ws_session), &svc.header(&token))
        .await
        .unwrap();
    Opened {
        ws,
        _server: server,
    }
}

async fn prompt(svc: &Service, ws: &mut TestWs, id: &str, message: &str) -> Vec<Value> {
    ws_send(ws, json!({"type": "prompt", "id": id, "message": message})).await;
    let frames = ws_read_until_response(ws, "prompt").await;
    assert_eq!(
        frames.last().unwrap()["success"],
        true,
        "{frames:#?}\n--- replica stderr ---\n{}",
        std::fs::read_to_string(&svc.stderr_log).unwrap_or_default()
    );
    frames
}

async fn mcp_tools(ws: &mut TestWs) -> Vec<String> {
    ws_send(ws, json!({"type": "get_mcp", "id": "m1"})).await;
    let frames = ws_read_until_response(ws, "get_mcp").await;
    let mut v: Vec<String> = frames.last().unwrap()["data"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_str)
        .map(str::to_string)
        .collect();
    v.sort();
    v
}

fn blob(frames: &[Value]) -> String {
    serde_json::to_string(frames).unwrap()
}

fn tool(name: &str) -> String {
    format!("mcp__tools__{name}")
}

fn log(row: &str, msg: impl AsRef<str>) {
    eprintln!("[matrix {row}] {}", msg.as_ref());
}

fn all_tool_names() -> Vec<String> {
    let mut v: Vec<String> = [
        "boom",
        "decide",
        "finish",
        "get_weather",
        "quiet",
        "read_seed",
        "slow",
    ]
    .iter()
    .map(|n| tool(n))
    .collect();
    v.sort();
    v
}

/// Row 1. Saved menu: open advertises the tools with no VM; the first call creates it.
#[tokio::test(flavor = "multi_thread")]
async fn r01_cold_open_with_saved_menu_creates_no_vm_until_the_first_call() {
    let e = env_or_skip!();
    let s = e.session("a");
    assert_eq!(e.front_creates(&s), 0);
    assert_eq!(e.e2b_vms(&s), 0, "no VM may exist before the open");
    let (base, requests) = spawn_model_server(vec![
        turn_tool_use("t1", &tool("get_weather"), r#"{"city":"SF"}"#),
        turn_text("It is sunny."),
    ]);
    let (svc, _c) = service(&base).await;
    let t0 = Instant::now();
    let mut o = open(&e, &svc, &s, "s1.r01", Menu::Saved, true).await;
    let tools = mcp_tools(&mut o.ws).await;
    let open_ms = t0.elapsed().as_millis();
    assert_eq!(tools, all_tool_names(), "saved menu not advertised");
    let (f_open, e_open) = (e.front_creates(&s), e.e2b_vms(&s));
    log(
        "r01",
        format!(
            "after open+get_mcp ({open_ms} ms): front vm_create_start={f_open} e2b_vms={e_open} tools={}",
            tools.len()
        ),
    );
    assert_eq!((f_open, e_open), (0, 0), "open created a VM");

    let t1 = Instant::now();
    let frames = prompt(&svc, &mut o.ws, "p1", "weather in SF").await;
    let call_ms = t1.elapsed().as_millis();
    let (f_call, e_call) = (e.front_creates(&s), e.e2b_vms(&s));
    log(
        "r01",
        format!("after first call ({call_ms} ms): front vm_create_start={f_call} e2b_vms={e_call}"),
    );
    assert_eq!(
        (f_call, e_call),
        (1, 1),
        "first call must create exactly one VM"
    );
    assert!(blob(&frames).contains("72"), "{}", blob(&frames));
    assert_eq!(model_calls(&requests.lock().unwrap()).len(), 2);
}

/// Row 2. No `.mcp-tools.json`: the connector dials at open and lists live.
#[tokio::test(flavor = "multi_thread")]
async fn r02_open_without_saved_menu_lists_tools_live() {
    let e = env_or_skip!();
    let s = e.session("b");
    assert_eq!(e.e2b_vms(&s), 0);
    let (base, _requests) = spawn_model_server(vec![]);
    let (svc, _c) = service(&base).await;
    let t0 = Instant::now();
    let mut o = open(&e, &svc, &s, "s1.r02", Menu::Live, true).await;
    let tools = mcp_tools(&mut o.ws).await;
    log(
        "r02",
        format!(
            "live list ({} ms): {} tools; front vm_create_start={} e2b_vms={}",
            t0.elapsed().as_millis(),
            tools.len(),
            e.front_creates(&s),
            e.e2b_vms(&s)
        ),
    );
    assert_eq!(tools, all_tool_names());
    assert_eq!(
        e.e2b_vms(&s),
        1,
        "a live dial at open is what creates the VM"
    );
}

/// Row 3. endsTurn: true ends the run after the tool; the result reaches the websocket; no more model calls.
#[tokio::test(flavor = "multi_thread")]
async fn r03_ends_turn_true_ends_the_run() {
    let e = env_or_skip!();
    let s = e.session("a");
    // Only one turn is scripted: a second model request is refused by the mock.
    let (base, requests) = spawn_model_server(vec![turn_tool_use("t1", &tool("finish"), "{}")]);
    let (svc, _c) = service(&base).await;
    let mut o = open(&e, &svc, &s, "s1.r03", Menu::Saved, true).await;
    let frames = prompt(&svc, &mut o.ws, "p1", "finish").await;
    let b = blob(&frames);
    let n = model_calls(&requests.lock().unwrap()).len();
    log(
        "r03",
        format!(
            "model requests={n}; result on ws: {}",
            b.contains("\\\"done\\\":true") || b.contains("\"done\":true") || b.contains("done")
        ),
    );
    assert!(
        b.contains("done"),
        "tool result never reached the websocket: {b}"
    );
    assert_eq!(n, 1, "the model was called again after endsTurn");
}

/// Row 4. endsTurn as a function: true ends, false goes back to the model.
#[tokio::test(flavor = "multi_thread")]
async fn r04_ends_turn_function_decides_after_execute() {
    let e = env_or_skip!();
    let s = e.session("a");
    let (base, requests) = spawn_model_server(vec![turn_tool_use(
        "t1",
        &tool("decide"),
        r#"{"stop":true}"#,
    )]);
    let (svc, _c) = service(&base).await;
    let mut o = open(&e, &svc, &s, "s1.r04a", Menu::Saved, true).await;
    prompt(&svc, &mut o.ws, "p1", "decide").await;
    let n = model_calls(&requests.lock().unwrap()).len();
    log("r04", format!("stop:true -> model requests={n}"));
    assert_eq!(n, 1);

    let (base, requests) = spawn_model_server(vec![
        turn_tool_use("t1", &tool("decide"), r#"{"stop":false}"#),
        turn_text("kept going"),
    ]);
    let (svc, _c) = service(&base).await;
    let mut o = open(&e, &svc, &s, "s1.r04b", Menu::Saved, true).await;
    prompt(&svc, &mut o.ws, "p1", "decide").await;
    let seen = model_calls(&requests.lock().unwrap());
    log(
        "r04",
        format!("stop:false -> model requests={}", seen.len()),
    );
    assert_eq!(seen.len(), 2, "stop:false must reach the model again");
    assert!(
        seen[1].contains("stop"),
        "tool result not in the next request"
    );
}

/// Row 5. A throwing endsTurn tool is an error result and the run continues.
#[tokio::test(flavor = "multi_thread")]
async fn r05_throwing_ends_turn_tool_continues_to_the_model() {
    let e = env_or_skip!();
    let s = e.session("a");
    let (base, requests) = spawn_model_server(vec![
        turn_tool_use("t1", &tool("boom"), "{}"),
        turn_text("recovered"),
    ]);
    let (svc, _c) = service(&base).await;
    let mut o = open(&e, &svc, &s, "s1.r05", Menu::Saved, true).await;
    let frames = prompt(&svc, &mut o.ws, "p1", "boom").await;
    let seen = model_calls(&requests.lock().unwrap());
    log(
        "r05",
        format!(
            "model requests={}; error text in 2nd request: {}",
            seen.len(),
            seen.get(1).is_some_and(|r| r.contains("boom-failure"))
        ),
    );
    assert_eq!(seen.len(), 2);
    assert!(
        seen[1].contains("boom-failure"),
        "error not shown to the model"
    );
    assert!(blob(&frames).contains("recovered"));
}

/// Row 6. A plain tool: the run continues and the model's answer follows the result.
#[tokio::test(flavor = "multi_thread")]
async fn r06_plain_tool_continues_and_the_answer_uses_it() {
    let e = env_or_skip!();
    let s = e.session("a");
    let (base, requests) = spawn_model_server(vec![
        turn_tool_use("t1", &tool("get_weather"), r#"{"city":"Oslo"}"#),
        turn_text("final"),
    ]);
    let (svc, _c) = service(&base).await;
    let mut o = open(&e, &svc, &s, "s1.r06", Menu::Saved, true).await;
    let frames = prompt(&svc, &mut o.ws, "p1", "weather").await;
    let seen = model_calls(&requests.lock().unwrap());
    log(
        "r06",
        format!(
            "model requests={}; 2nd request carries Oslo/72: {}",
            seen.len(),
            seen[1].contains("Oslo") && seen[1].contains("72")
        ),
    );
    assert_eq!(seen.len(), 2);
    assert!(seen[1].contains("Oslo") && seen[1].contains("72"));
    assert!(blob(&frames).contains("final"));
}

/// Row 7. endsTurn in an MCP-Apps-capable session, without a saved menu (and with one).
#[tokio::test(flavor = "multi_thread")]
async fn r07_ends_turn_in_an_apps_session() {
    let e = env_or_skip!();
    for (menu, s, label) in [
        (Menu::Live, e.session("b"), "no saved menu"),
        (Menu::Saved, e.session("a"), "saved menu"),
    ] {
        let (base, requests) = spawn_model_server(vec![turn_tool_use("t1", &tool("finish"), "{}")]);
        let (svc, _c) = service(&base).await;
        let mut o = open(&e, &svc, &s, "s1.r07", menu, true).await;
        ws_send(&mut o.ws, json!({"type": "set_mcp_apps", "enabled": true})).await;
        let declared = ws_read_until_response(&mut o.ws, "set_mcp_apps").await;
        assert_eq!(declared.last().unwrap()["success"], true, "{declared:?}");
        let frames = prompt(&svc, &mut o.ws, "p1", "finish").await;
        let n = model_calls(&requests.lock().unwrap()).len();
        log("r07", format!("{label}: model requests={n}"));
        assert!(blob(&frames).contains("done"));
        assert_eq!(n, 1, "{label}: an apps session must still end the run");
    }
}

/// Row 8. The same server as an ordinary (non-sandbox) connector: `_meta` is ignored.
#[tokio::test(flavor = "multi_thread")]
async fn r08_non_sandbox_connector_does_not_end_the_run() {
    let e = env_or_skip!();
    let s = e.session("c");
    let (base, requests) = spawn_model_server(vec![
        turn_tool_use("t1", &tool("finish"), "{}"),
        turn_text("after"),
    ]);
    let (svc, _c) = service(&base).await;
    let mut o = open(&e, &svc, &s, "s1.r08", Menu::Live, false).await;
    let frames = prompt(&svc, &mut o.ws, "p1", "finish").await;
    let seen = model_calls(&requests.lock().unwrap());
    log(
        "r08",
        format!(
            "model requests={}; saw tool result in 2nd request: {}",
            seen.len(),
            seen.get(1).is_some_and(|r| r.contains("done"))
        ),
    );
    assert_eq!(
        seen.len(),
        2,
        "a non-sandbox connector must not end the run"
    );
    assert!(seen[1].contains("done"));
    assert!(blob(&frames).contains("after"));
}

/// Row 9. availableInSubagents:false: the parent sees `quiet`, a subagent does not.
#[tokio::test(flavor = "multi_thread")]
async fn r09_available_in_subagents_false_hides_the_tool_from_a_subagent() {
    let e = env_or_skip!();
    let s = e.session("a");
    let (base, requests) = spawn_model_server(vec![
        turn_tool_use(
            "t1",
            "subagent",
            &json!({"agent": "scout", "task": "look around"}).to_string(),
        ),
        turn_text("CHILD-REPORT"),
        turn_text("parent done"),
    ]);
    let (svc, _c) = service(&base).await;
    let mut o = open(&e, &svc, &s, "s1.r09", Menu::Saved, true).await;
    prompt(&svc, &mut o.ws, "p1", "delegate").await;
    let seen = model_calls(&requests.lock().unwrap());
    assert!(
        seen.len() >= 3,
        "expected parent, child, parent: {}",
        seen.len()
    );
    let parent = advertised_tools(&seen[0]);
    let child = advertised_tools(&seen[1]);
    log("r09", format!("parent tools: {parent:?}"));
    log("r09", format!("child tools: {child:?}"));
    assert!(parent.contains(&tool("quiet")), "parent must see quiet");
    assert!(
        child.contains(&tool("get_weather")),
        "control: the child must see an ordinary sandbox tool: {child:?}"
    );
    assert!(!child.contains(&tool("quiet")), "child must not see quiet");
}

/// Row 10. Workspace root: the seeded workspace, not the tools dir.
#[tokio::test(flavor = "multi_thread")]
async fn r10_workspace_root_is_the_seeded_workspace() {
    let e = env_or_skip!();
    let s = e.session("a");
    let (base, requests) = spawn_model_server(vec![
        turn_tool_use("t1", &tool("read_seed"), r#"{"path":"seed.txt"}"#),
        turn_text("read"),
    ]);
    let (svc, _c) = service(&base).await;
    let mut o = open(&e, &svc, &s, "s1.r10", Menu::Saved, true).await;
    prompt(&svc, &mut o.ws, "p1", "read seed").await;
    let seen = model_calls(&requests.lock().unwrap());
    let result = &seen[1];
    let right = result.matches("RIGHT-workspace").count();
    let root = result
        .find("workspaceRoot")
        .map(|i| result[i..result.len().min(i + 80)].to_string());
    log(
        "r10",
        format!(
            "RIGHT-workspace occurrences={right} (ctx + getSandbox); WRONG-tools-dir present={}; {root:?}",
            result.contains("WRONG-tools-dir")
        ),
    );
    assert!(
        right >= 2,
        "both ctx.workspaceRoot and getSandbox() must read the workspace file"
    );
    assert!(!result.contains("WRONG-tools-dir"));
}

/// Row 11. Abort a turn while a slow tool runs: the tool observes it, the run ends, the next turn works.
#[tokio::test(flavor = "multi_thread")]
async fn r11_abort_cancels_the_running_tool() {
    let e = env_or_skip!();
    let s = e.session("a");
    let (base, requests) = spawn_model_server(vec![
        // Warm-up: the first call to a session creates its VM; the abort must land on a call that
        // is already running in it, not on one still waiting for the VM.
        turn_tool_use("w1", &tool("get_weather"), r#"{"city":"Oslo"}"#),
        turn_text("warm"),
        turn_tool_use("t1", &tool("slow"), r#"{"seconds":60}"#),
        // After the abort: a fresh turn reads the marker the tool should have written.
        turn_tool_use("t2", &tool("read_seed"), r#"{"path":"abort-marker.txt"}"#),
        turn_text("next turn fine"),
    ]);
    let (svc, _c) = service(&base).await;
    let mut o = open(&e, &svc, &s, "s1.r11", Menu::Saved, true).await;
    prompt(&svc, &mut o.ws, "p0", "warm up").await;
    ws_send(
        &mut o.ws,
        json!({"type": "prompt", "id": "p1", "message": "be slow"}),
    )
    .await;
    // Wait until the slow tool has started.
    let t0 = Instant::now();
    loop {
        let f = ws_next_frame_within(&mut o.ws, Duration::from_secs(120))
            .await
            .expect("frame before tool_start");
        if f["type"] == "event" && f["event"]["kind"] == "tool_start" {
            break;
        }
    }
    // Let the call reach the VM and the tool begin polling.
    tokio::time::sleep(Duration::from_secs(4)).await;
    let t_abort = Instant::now();
    ws_send(&mut o.ws, json!({"type": "abort", "id": "a1"})).await;
    let frames = ws_read_until_response(&mut o.ws, "prompt").await;
    log(
        "r11",
        format!(
            "abort->prompt response in {} ms (tool had been running {} ms); success={}",
            t_abort.elapsed().as_millis(),
            t0.elapsed().as_millis(),
            frames.last().unwrap()["success"]
        ),
    );
    assert!(
        t_abort.elapsed() < Duration::from_secs(30),
        "the run did not end promptly after the abort"
    );
    // Give the cancel a moment to cross the front and the E2B edge.
    tokio::time::sleep(Duration::from_secs(3)).await;
    let frames = prompt(&svc, &mut o.ws, "p2", "what did the tool see").await;
    let seen = model_calls(&requests.lock().unwrap());
    let msgs = common::body_json(seen.last().unwrap())["messages"].to_string();
    let marker = &msgs;
    let observed = marker.contains("ABORT-OBSERVED");
    log(
        "r11",
        format!(
            "next turn ok (model requests={}); marker file seen in VM: {observed}; last messages tail: {}",
            seen.len(),
            &msgs[msgs.len().saturating_sub(500)..]
        ),
    );
    assert!(blob(&frames).contains("next turn fine"));
    // The agent's MCP client sends `notifications/cancelled` when the running `tools/call` is
    // dropped by the abort; it crosses the front and the guest cancels the tool.
    assert!(
        observed,
        "the tool never observed the abort: {}",
        &msgs[msgs.len().saturating_sub(600)..]
    );
}

/// Row 11b. The guest's own cancellation, over the same front: a `notifications/cancelled` for a
/// running `slow` call aborts it (the tool observes `ctx.abortSignal`) and the call answers.
#[tokio::test(flavor = "multi_thread")]
async fn r11b_notifications_cancelled_aborts_the_tool_in_the_vm() {
    let e = env_or_skip!();
    let s = e.session("a");
    let url = e.mcp_url(&s, Menu::Saved);
    let post = |sid: Option<&str>, body: &str, max: &str| -> (String, String) {
        let mut c = Command::new("curl");
        c.args(["-sS", "-m", max, "-D", "-", "-X", "POST"])
            .args(["-H", "content-type: application/json"])
            .args(["-H", "accept: application/json, text/event-stream"]);
        if let Some(sid) = sid {
            c.args(["-H", &format!("mcp-session-id: {sid}")]);
        }
        let out = c.args(["-d", body, &url]).output().unwrap();
        let text = String::from_utf8_lossy(&out.stdout).to_string();
        let (h, b) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
        (h.to_string(), b.to_string())
    };
    let (h, _) = post(
        None,
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"c","version":"0"}}}"#,
        "300",
    );
    let sid = h
        .lines()
        .find_map(|l| {
            l.to_ascii_lowercase()
                .starts_with("mcp-session-id:")
                .then(|| l.split_once(':').unwrap().1.trim().to_string())
        })
        .expect("no session id");
    let _ = post(
        Some(&sid),
        r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
        "60",
    );
    let sid2 = sid.clone();
    let url2 = url.clone();
    let slow = std::thread::spawn(move || {
        let t = Instant::now();
        let out = Command::new("curl")
            .args(["-sS", "-m", "120", "-X", "POST", "-H", "content-type: application/json"])
            .args(["-H", "accept: application/json, text/event-stream"])
            .args(["-H", &format!("mcp-session-id: {sid2}")])
            .args(["-d", r#"{"jsonrpc":"2.0","id":11,"method":"tools/call","params":{"name":"slow","arguments":{"seconds":60}}}"#, &url2])
            .output()
            .unwrap();
        (
            String::from_utf8_lossy(&out.stdout).to_string(),
            t.elapsed(),
        )
    });
    std::thread::sleep(Duration::from_secs(4));
    let _ = post(
        Some(&sid),
        r#"{"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":11,"reason":"matrix"}}"#,
        "60",
    );
    let (answer, took) = slow.join().unwrap();
    let (_, read) = post(
        Some(&sid),
        r#"{"jsonrpc":"2.0","id":12,"method":"tools/call","params":{"name":"read_seed","arguments":{"path":"abort-marker.txt"}}}"#,
        "60",
    );
    log(
        "r11b",
        format!(
            "slow call answered after {} ms (tool asked for 60 s): {}; marker read: {}",
            took.as_millis(),
            answer.chars().take(160).collect::<String>(),
            read.chars().take(200).collect::<String>()
        ),
    );
    assert!(
        took < Duration::from_secs(30),
        "the cancel did not end the call"
    );
    assert!(
        read.contains("ABORT-OBSERVED"),
        "tool never saw the abort: {read}"
    );
}

/// Row 12. The in-VM server's session gate, reached through the front like the agent does.
#[tokio::test(flavor = "multi_thread")]
async fn r12_http_session_gate() {
    let e = env_or_skip!();
    let s = e.session("d");
    let url = e.mcp_url(&s, Menu::Saved);
    let curl = |sid: Option<&str>, body: &str| -> (String, String) {
        let mut c = Command::new("curl");
        c.args(["-sS", "-m", "300", "-D", "-", "-X", "POST"])
            .args(["-H", "content-type: application/json"])
            .args(["-H", "accept: application/json, text/event-stream"]);
        if let Some(sid) = sid {
            c.args(["-H", &format!("mcp-session-id: {sid}")]);
        }
        let out = c.args(["-d", body, &url]).output().unwrap();
        let text = String::from_utf8_lossy(&out.stdout).to_string();
        let (head, body) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
        (head.to_string(), body.to_string())
    };
    let status = |head: &str| head.lines().next().unwrap_or("").to_string();
    let call = r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"get_weather","arguments":{"city":"SF"}}}"#;

    let (h, b) = curl(None, call);
    log(
        "r12",
        format!(
            "tools/call, no session -> {} {}",
            status(&h),
            b.chars().take(120).collect::<String>()
        ),
    );
    assert!(status(&h).contains(" 400"), "{h}\n{b}");

    let (h, b) = curl(Some("not-an-issued-id"), call);
    log(
        "r12",
        format!(
            "tools/call, unknown session id -> {} {}",
            status(&h),
            b.chars().take(120).collect::<String>()
        ),
    );
    assert!(status(&h).contains(" 404"), "{h}\n{b}");

    let init = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"gate","version":"0"}}}"#;
    let (h, _) = curl(None, init);
    log("r12", format!("initialize -> {}", status(&h)));
    assert!(status(&h).contains(" 200"), "{h}");
    let sid = h
        .lines()
        .find_map(|l| {
            l.to_ascii_lowercase()
                .starts_with("mcp-session-id:")
                .then(|| l.split_once(':').unwrap().1.trim().to_string())
        })
        .expect("no Mcp-Session-Id issued");
    let _ = curl(
        Some(&sid),
        r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
    );
    let (h, b) = curl(Some(&sid), call);
    log(
        "r12",
        format!(
            "tools/call, issued session -> {} {}",
            status(&h),
            b.chars().take(160).collect::<String>()
        ),
    );
    assert!(status(&h).contains(" 200") && b.contains("72"), "{h}\n{b}");
}

/// Row 13. Two sessions at once, each on its own VM.
#[tokio::test(flavor = "multi_thread")]
async fn r13_two_sessions_concurrently_on_separate_vms() {
    let e = env_or_skip!();
    let (sa, sb) = (e.session("e"), e.session("f"));
    async fn one(e: &Env, s: &str, ws: &str, city: &str) -> String {
        let (base, requests) = spawn_model_server(vec![
            turn_tool_use(
                "t1",
                &tool("get_weather"),
                &format!(r#"{{"city":"{city}"}}"#),
            ),
            turn_text(&format!("done {city}")),
        ]);
        let (svc, _c) = service(&base).await;
        let mut o = open(e, &svc, s, ws, Menu::Saved, true).await;
        let t = Instant::now();
        prompt(&svc, &mut o.ws, "p1", "weather").await;
        let seen = model_calls(&requests.lock().unwrap());
        assert_eq!(seen.len(), 2);
        assert!(seen[1].contains(city) && seen[1].contains("72"));
        format!("{city}: ok in {} ms", t.elapsed().as_millis())
    }
    let (a, b) = tokio::join!(
        one(&e, &sa, "s1.r13a", "Lima"),
        one(&e, &sb, "s1.r13b", "Kyiv")
    );
    let (va, vb) = (e.e2b_vms(&sa), e.e2b_vms(&sb));
    log("r13", format!("{a}; {b}; e2b vms: {sa}={va} {sb}={vb}"));
    assert_eq!((va, vb), (1, 1));
}

/// Row 14. A real cheap model picks and calls the tools (get_weather, then the endsTurn tool).
#[tokio::test(flavor = "multi_thread")]
async fn r14_real_model_chooses_and_calls_the_tools() {
    let e = env_or_skip!();
    let Some(key) = std::env::var("OPENAI_API_KEY")
        .ok()
        .filter(|k| !k.is_empty())
    else {
        eprintln!("skip: OPENAI_API_KEY");
        return;
    };
    let model = std::env::var("MATRIX_MODEL").unwrap_or_else(|_| "gpt-5-mini".into());
    let s = e.session("a");
    let (svc, _c) = service("https://api.openai.com").await;
    let (bytes, _) = e.bundle(Menu::Saved);
    let server = BundleServer::start(bytes.clone());
    let mcp = e.mcp_url(&s, Menu::Saved);
    let secrets = Secrets {
        gateway_key: key,
        ..svc.secrets()
    };
    let home = svc.shards[0].0.clone();
    let mut claims = svc.claims("t1", "s1.r14", &home);
    claims.mcp = vec![("tools".into(), mcp)];
    claims.mcp_sandbox = vec!["tools".into()];
    claims.bundle = Some(Bundle {
        url: server.url.clone(),
        sha256: beyond_test_support::bundle::sha256_hex(&bytes),
    });
    let token = svc.minter.mint(&claims, &secrets);
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.r14"), &svc.header(&token))
        .await
        .unwrap();
    ws_send(
        &mut ws,
        json!({"type": "set_model", "id": "sm", "model": model}),
    )
    .await;
    let r = ws_read_until_response(&mut ws, "set_model").await;
    assert_eq!(
        r.last().unwrap()["success"],
        true,
        "set_model: {:?}",
        r.last().unwrap()
    );
    let frames = prompt(
        &svc,
        &mut ws,
        "p1",
        "What is the weather in Paris right now? Use your get_weather tool, tell me the temperature in one sentence, \
         and then call the finish tool to wrap up the conversation.",
    )
    .await;
    let calls: Vec<String> = frames
        .iter()
        .filter(|f| f["type"] == "event" && f["event"]["kind"] == "tool_start")
        .map(|f| f["event"]["name"].as_str().unwrap_or("?").to_string())
        .collect();
    let usage = frames
        .iter()
        .filter(|f| f["type"] == "event" && f["event"]["kind"] == "usage")
        .count();
    log(
        "r14",
        format!("model={model} tool calls in order: {calls:?}; usage events={usage}"),
    );
    let b = blob(&frames);
    assert!(
        calls.iter().any(|c| c.ends_with("get_weather")),
        "{calls:?}"
    );
    assert!(b.contains("72"), "weather result missing");
    assert!(calls.iter().any(|c| c.ends_with("finish")), "{calls:?}");
    // The run ended on finish: no tool call after it.
    assert!(calls.last().unwrap().ends_with("finish"), "{calls:?}");
}
