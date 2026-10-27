//! sandbox-mcp driven the way an outside agent drives it: a generic MCP client launches the real
//! binary, lists its tools, and uses a sandbox through them. The sandbox is the exec double
//! (`beyond_test_support::exec_mock`), which runs real commands on this host under a temp root.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::time::Duration;

use beyond_test_support::exec_mock::ExecMock;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, ReadResourceRequestParams, ResourceContents,
};
use rmcp::service::{Peer, RunningService};
use rmcp::transport::TokioChildProcess;
use rmcp::{RoleClient, ServiceExt as _};
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_sandbox-mcp");
/// Nothing in these tests may take longer than this; a hang is a failure.
const DEADLINE: Duration = Duration::from_secs(60);

fn server_command(url: &str) -> tokio::process::Command {
    let mut cmd = tokio::process::Command::new(BIN);
    for k in [
        "SANDBOX_TOKEN",
        "SANDBOX_TOKEN_CMD",
        "SANDBOX_CAPS",
        "SANDBOX_JOBS_ROOT",
        "SANDBOX_MCP_LISTEN",
    ] {
        cmd.env_remove(k);
    }
    cmd.env("SANDBOX_URL", url).kill_on_drop(true);
    cmd
}

async fn connect(url: &str) -> RunningService<RoleClient, ()> {
    let transport = TokioChildProcess::new(server_command(url)).unwrap();
    ().serve(transport).await.unwrap()
}

async fn call(peer: &Peer<RoleClient>, tool: &str, args: Value) -> CallToolResult {
    let params = CallToolRequestParams::new(tool.to_string())
        .with_arguments(args.as_object().unwrap().clone());
    peer.call_tool(params).await.unwrap()
}

fn ok(r: &CallToolResult) -> &Value {
    assert_ne!(
        r.is_error,
        Some(true),
        "tool failed: {:?}",
        r.structured_content
    );
    r.structured_content.as_ref().unwrap()
}

fn error_kind(r: &CallToolResult) -> String {
    assert_eq!(
        r.is_error,
        Some(true),
        "expected a tool error: {:?}",
        r.structured_content
    );
    r.structured_content
        .as_ref()
        .unwrap_or_else(|| panic!("no structured error: {r:?}"))["error"]["kind"]
        .as_str()
        .unwrap()
        .to_string()
}

async fn sandbox() -> (tempfile::TempDir, ExecMock) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    let mock = ExecMock::start(&root, true).await;
    (dir, mock)
}

#[tokio::test]
async fn an_agent_lists_the_tools_writes_a_file_runs_cat_and_reads_it_back() {
    tokio::time::timeout(DEADLINE, async {
        let (dir, mock) = sandbox().await;
        let root = dir.path().canonicalize().unwrap();
        let file = format!("{}/hello.txt", root.display());
        let client = connect(&mock.url).await;
        let peer = client.peer();

        let tools = peer.list_all_tools().await.unwrap();
        assert!(
            tools.iter().any(|t| t.name == "exec"),
            "exec is always present"
        );
        assert_eq!(tools.len(), 12);

        let w = call(
            peer,
            "write_file",
            json!({"path": file, "content": "héllo\n"}),
        )
        .await;
        assert_eq!(ok(&w)["bytes"], 7);

        let cat = call(peer, "exec", json!({"command": "cat", "args": [file]})).await;
        let cat = ok(&cat);
        assert_eq!(cat["exit_code"], 0);
        assert_eq!(cat["stdout"], "héllo\n");

        let r = call(peer, "read_file", json!({"path": file})).await;
        let r = ok(&r);
        assert_eq!(r["content"], "héllo\n");
        assert_eq!(r["encoding"], "utf8");
        assert_eq!(r["eof"], true);

        client.cancel().await.unwrap();
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn files_edit_list_search_and_stat_work_through_the_tools() {
    tokio::time::timeout(DEADLINE, async {
        let (dir, mock) = sandbox().await;
        let root = dir.path().canonicalize().unwrap();
        let p = |name: &str| format!("{}/{name}", root.display());
        let client = connect(&mock.url).await;
        let peer = client.peer();

        ok(&call(peer, "mkdir", json!({"path": p("src/deep")})).await);
        ok(&call(
            peer,
            "write_file",
            json!({"path": p("src/a.txt"), "content": "one\ntwo\nthree\n"}),
        )
        .await);
        ok(&call(
            peer,
            "write_file",
            json!({"path": p("src/.hidden"), "content": "x"}),
        )
        .await);

        let e = call(
            peer,
            "edit_file",
            json!({"path": p("src/a.txt"), "old_str": "two", "new_str": "2"}),
        )
        .await;
        assert_eq!(ok(&e)["replacements"], 1);
        let r = call(peer, "read_file", json!({"path": p("src/a.txt")})).await;
        assert_eq!(ok(&r)["content"], "one\n2\nthree\n");

        // An edit that does not match exactly once is an error the agent can read, and writes nothing.
        let none = call(
            peer,
            "edit_file",
            json!({"path": p("src/a.txt"), "old_str": "zzz", "new_str": "y"}),
        )
        .await;
        assert_eq!(error_kind(&none), "invalid_request");
        let twice = call(
            peer,
            "edit_file",
            json!({"path": p("src/a.txt"), "old_str": "e", "new_str": "E"}),
        )
        .await;
        assert_eq!(error_kind(&twice), "invalid_request");

        let l = call(peer, "list_dir", json!({"path": p("src")})).await;
        let names: Vec<&str> = ok(&l)["entries"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["a.txt", "deep"]);
        let l = call(
            peer,
            "list_dir",
            json!({"path": p("src"), "include_hidden": true, "limit": 1}),
        )
        .await;
        assert_eq!(ok(&l)["truncated"], true);

        let s = call(
            peer,
            "search",
            json!({"pattern": "thr", "path": p("src"), "context": 1}),
        )
        .await;
        let hits = ok(&s)["hits"].as_array().unwrap();
        assert!(
            hits.iter()
                .any(|h| h["is_match"] == true && h["text"] == "three")
        );
        assert!(
            hits.iter()
                .any(|h| h["is_match"] == false && h["text"] == "2")
        );

        let st = call(peer, "stat", json!({"path": p("src/a.txt")})).await;
        let st = ok(&st);
        assert_eq!(
            (st["exists"].clone(), st["kind"].clone(), st["size"].clone()),
            (json!(true), json!("file"), json!(12))
        );
        let gone = call(peer, "stat", json!({"path": p("nope")})).await;
        assert_eq!(ok(&gone)["exists"], false);

        client.cancel().await.unwrap();
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn binary_files_travel_as_base64_and_a_cut_character_is_left_for_the_next_read() {
    tokio::time::timeout(DEADLINE, async {
        let (dir, mock) = sandbox().await;
        let file = format!("{}/bin", dir.path().canonicalize().unwrap().display());
        let client = connect(&mock.url).await;
        let peer = client.peer();

        // "é" is two bytes; a 2-byte window starting at "a" ends inside it.
        ok(&call(peer, "write_file", json!({"path": file, "content": "aé"})).await);
        let r = call(peer, "read_file", json!({"path": file, "max": 2})).await;
        let r = ok(&r);
        assert_eq!(
            (
                r["content"].clone(),
                r["bytes"].clone(),
                r["next_offset"].clone(),
                r["eof"].clone()
            ),
            (json!("a"), json!(1), json!(1), json!(false))
        );
        let r = call(peer, "read_file", json!({"path": file, "offset": 1})).await;
        assert_eq!(ok(&r)["content"], "é");

        ok(&call(
            peer,
            "write_file",
            json!({"path": file, "content_base64": "/wCA"}),
        )
        .await);
        let r = call(peer, "read_file", json!({"path": file})).await;
        let r = ok(&r);
        assert_eq!(
            (r["encoding"].clone(), r["content"].clone()),
            (json!("base64"), json!("/wCA"))
        );

        client.cancel().await.unwrap();
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn failures_are_tool_results_with_the_spec_kind_never_protocol_errors() {
    tokio::time::timeout(DEADLINE, async {
        let (dir, mock) = sandbox().await;
        let root = dir.path().canonicalize().unwrap();
        let client = connect(&mock.url).await;
        let peer = client.peer();

        let r = call(
            peer,
            "read_file",
            json!({"path": format!("{}/missing", root.display())}),
        )
        .await;
        assert_eq!(error_kind(&r), "not_found");
        let r = call(
            peer,
            "read_file",
            json!({"path": root.display().to_string()}),
        )
        .await;
        assert_eq!(error_kind(&r), "is_directory");
        let r = call(
            peer,
            "search",
            json!({"pattern": "(", "path": root.display().to_string()}),
        )
        .await;
        assert_eq!(error_kind(&r), "invalid_query");
        let r = call(peer, "job_output", json!({"id": "Not Valid"})).await;
        assert_eq!(error_kind(&r), "invalid_request");
        // Both content forms, and neither, are the agent's mistake to read, not a protocol error.
        let r = call(peer, "write_file", json!({"path": "/x"})).await;
        assert_eq!(error_kind(&r), "invalid_request");
        // A wrongly typed argument (the schema says a string) is also a result.
        let r = call(peer, "stat", json!({"path": 7})).await;
        assert_eq!(error_kind(&r), "invalid_request");

        client.cancel().await.unwrap();
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn jobs_start_report_output_and_stop() {
    tokio::time::timeout(DEADLINE, async {
        let (_dir, mock) = sandbox().await;
        let client = connect(&mock.url).await;
        let peer = client.peer();

        let s = call(
            peer,
            "job_start",
            json!({"command": "sh", "args": ["-c", "echo started; sleep 300"], "label": "waiter"}),
        )
        .await;
        let id = ok(&s)["id"].as_str().unwrap().to_string();

        let mut out = String::new();
        let mut offset = 0;
        while !out.contains("started") {
            let o = call(peer, "job_output", json!({"id": id, "offset": offset})).await;
            let o = ok(&o);
            out.push_str(o["content"].as_str().unwrap());
            offset = o["next_offset"].as_u64().unwrap();
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let l = call(peer, "job_list", json!({})).await;
        let job = ok(&l)["jobs"]
            .as_array()
            .unwrap()
            .iter()
            .find(|j| j["id"] == id.as_str())
            .unwrap()
            .clone();
        assert_eq!(
            (job["state"].clone(), job["label"].clone()),
            (json!("running"), json!("waiter"))
        );

        let k = call(peer, "job_kill", json!({"ids": [id], "remove": true})).await;
        assert_eq!(ok(&k)["survivors"], json!([]));
        let l = call(peer, "job_list", json!({})).await;
        assert!(
            ok(&l)["jobs"]
                .as_array()
                .unwrap()
                .iter()
                .all(|j| j["id"] != id.as_str())
        );

        client.cancel().await.unwrap();
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn an_unreachable_sandbox_starts_the_server_and_fails_each_call_with_a_kind() {
    tokio::time::timeout(DEADLINE, async {
        // Nothing listens on port 1.
        let client = connect("http://127.0.0.1:1/").await;
        let peer = client.peer();
        assert_eq!(peer.list_all_tools().await.unwrap().len(), 12);
        let r = call(peer, "exec", json!({"command": "true"})).await;
        assert_eq!(error_kind(&r), "transport");

        // The capabilities resource reports the same, as content rather than a protocol error.
        let caps = peer
            .read_resource(ReadResourceRequestParams::new(
                "sandbox://capabilities".to_string(),
            ))
            .await
            .unwrap();
        let ResourceContents::TextResourceContents { text, .. } = &caps.contents[0] else {
            panic!("not text")
        };
        assert_eq!(
            serde_json::from_str::<Value>(text).unwrap()["error"]["kind"],
            "transport"
        );

        client.cancel().await.unwrap();
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn the_capabilities_resource_says_what_the_sandbox_supports() {
    tokio::time::timeout(DEADLINE, async {
        let (_dir, mock) = sandbox().await;
        let client = connect(&mock.url).await;
        let caps = client
            .peer()
            .read_resource(ReadResourceRequestParams::new(
                "sandbox://capabilities".to_string(),
            ))
            .await
            .unwrap();
        let ResourceContents::TextResourceContents { text, .. } = &caps.contents[0] else {
            panic!("not text")
        };
        let v: Value = serde_json::from_str(text).unwrap();
        assert_eq!(v["stdin"], true);
        assert!(["rg", "grep", "busybox-grep"].contains(&v["search_engine"].as_str().unwrap()));
        assert!(["native", "baseline", "none"].contains(&v["jobs"].as_str().unwrap()));
        client.cancel().await.unwrap();
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn a_token_command_supplies_the_bearer_and_both_token_variables_together_are_refused() {
    tokio::time::timeout(DEADLINE, async {
        let (_dir, mock) = sandbox().await;
        let mut cmd = server_command(&mock.url);
        cmd.env("SANDBOX_TOKEN_CMD", "echo secret-from-command");
        let client = ().serve(TokioChildProcess::new(cmd).unwrap()).await.unwrap();
        ok(&call(client.peer(), "exec", json!({"command": "true"})).await);
        client.cancel().await.unwrap();
        let sent: Vec<Option<String>> = mock
            .requests()
            .iter()
            .map(|r| r.header("authorization").map(str::to_string))
            .collect();
        assert!(!sent.is_empty());
        assert!(
            sent.iter()
                .all(|h| h.as_deref() == Some("Bearer secret-from-command")),
            "{sent:?}"
        );

        let out = server_command(&mock.url)
            .env("SANDBOX_TOKEN", "a")
            .env("SANDBOX_TOKEN_CMD", "echo b")
            .output()
            .await
            .unwrap();
        assert!(!out.status.success());
        assert!(String::from_utf8_lossy(&out.stderr).contains("not both"));
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn streamable_http_serves_the_same_tools_on_loopback_and_refuses_a_public_address() {
    tokio::time::timeout(DEADLINE, async {
        use rmcp::transport::StreamableHttpClientTransport;
        use tokio::io::{AsyncBufReadExt as _, BufReader};

        let _ = rustls::crypto::ring::default_provider().install_default();
        let (_dir, mock) = sandbox().await;

        let mut cmd = server_command(&mock.url);
        cmd.env("SANDBOX_MCP_LISTEN", "127.0.0.1:0")
            .stderr(std::process::Stdio::piped());
        let mut child = cmd.spawn().unwrap();
        let mut lines = BufReader::new(child.stderr.take().unwrap()).lines();
        let first = lines.next_line().await.unwrap().unwrap();
        let url = first
            .split("listening on ")
            .nth(1)
            .unwrap()
            .trim()
            .to_string();

        let client = ().serve(StreamableHttpClientTransport::from_uri(url)).await.unwrap();
        assert_eq!(client.peer().list_all_tools().await.unwrap().len(), 12);
        let r = call(
            client.peer(),
            "exec",
            json!({"command": "echo", "args": ["over http"]}),
        )
        .await;
        assert_eq!(ok(&r)["stdout"], "over http\n");
        client.cancel().await.unwrap();
        child.kill().await.unwrap();

        let out = server_command(&mock.url)
            .env("SANDBOX_MCP_LISTEN", "0.0.0.0:0")
            .output()
            .await
            .unwrap();
        assert!(!out.status.success());
        assert!(String::from_utf8_lossy(&out.stderr).contains("not a loopback address"));
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn a_wrong_token_is_an_auth_error_and_a_token_command_is_asked_again_after_a_401() {
    tokio::time::timeout(DEADLINE, async {
        use beyond_test_support::exec_mock::MockOptions;
        let dir = tempfile::tempdir().unwrap();
        let mock = ExecMock::start_with(
            &dir.path().canonicalize().unwrap(),
            MockOptions {
                require_bearer: Some("good".to_string()),
                ..MockOptions::default()
            },
        )
        .await;

        // A fixed wrong token: the server starts, and every call says `auth`.
        let mut cmd = server_command(&mock.url);
        cmd.env("SANDBOX_TOKEN", "bad");
        let client = ().serve(TokioChildProcess::new(cmd).unwrap()).await.unwrap();
        assert_eq!(
            error_kind(&call(client.peer(), "exec", json!({"command": "true"})).await),
            "auth"
        );
        client.cancel().await.unwrap();

        // A token command that hands out a stale token first and a good one after: the client
        // library asks again once on the 401, and the call succeeds.
        let state = dir.path().join("asked");
        let script = format!(
            "if [ -e {0} ]; then echo good; else touch {0}; echo stale; fi",
            state.display()
        );
        let mut cmd = server_command(&mock.url);
        cmd.env("SANDBOX_TOKEN_CMD", script);
        let client = ().serve(TokioChildProcess::new(cmd).unwrap()).await.unwrap();
        ok(&call(client.peer(), "exec", json!({"command": "true"})).await);
        client.cancel().await.unwrap();
    })
    .await
    .unwrap();
}
