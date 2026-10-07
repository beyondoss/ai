//! Shared harness for the MCP Apps suites (`tests/mcp_apps*.rs`): a `$HOME` configuring the
//! `mcp_apps_fixture_server`, the daemon, and the renderer-side commands a test client sends.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, dead_code)]

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use serde_json::{Value, json};

use super::{ChildGuard, SpawnGuarded, ws_read_until_response, ws_send};

pub const FIXTURE: &str = env!("CARGO_BIN_EXE_mcp_apps_fixture_server");

/// A `$HOME` whose settings configure the fixture as server `wx`, logging every request it sees.
pub struct Home {
    _dir: tempfile::TempDir,
    pub home: PathBuf,
    pub log: PathBuf,
}

pub fn home_with_fixture(env: Value) -> Home {
    home_with_servers(&["wx"], env)
}

/// A `$HOME` configuring the fixture once per name in `servers`, all logging to the same file.
pub fn home_with_servers(servers: &[&str], env: Value) -> Home {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let log = dir.path().join("fixture.log");
    std::fs::create_dir_all(home.join(".claude")).unwrap();
    let mut env = env;
    env["MCP_APPS_FIXTURE_LOG"] = json!(log.to_string_lossy());
    let configs: Vec<Value> = servers
        .iter()
        .map(|name| {
            json!({ "name": name, "transport": "stdio", "command": FIXTURE, "args": [], "env": env })
        })
        .collect();
    std::fs::write(
        home.join(".claude/settings.json"),
        serde_json::to_string_pretty(&json!({ "mcp_servers": configs })).unwrap(),
    )
    .unwrap();
    Home {
        _dir: dir,
        home,
        log,
    }
}

impl Home {
    /// Every request the fixture logged: `(method, ui_advertised)`.
    pub fn requests(&self) -> Vec<(String, bool)> {
        std::fs::read_to_string(&self.log)
            .unwrap_or_default()
            .lines()
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .map(|v| {
                (
                    v["method"].as_str().unwrap_or_default().to_owned(),
                    v["ui"].as_bool().unwrap_or(false),
                )
            })
            .collect()
    }

    /// How many times the fixture served `resources/read` for `uri`.
    pub fn reads_of(&self, uri: &str) -> usize {
        std::fs::read_to_string(&self.log)
            .unwrap_or_default()
            .lines()
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .filter(|v| v["method"] == "resources/read" && v["uri"] == uri)
            .count()
    }

    pub fn saw_ui(&self) -> bool {
        self.requests().iter().any(|(_, ui)| *ui)
    }

    pub fn saw_plain_tools_list(&self) -> bool {
        self.requests()
            .iter()
            .any(|(m, ui)| m == "tools/list" && !ui)
    }
}

/// A running daemon on a Unix-domain socket of its own — no TCP port to race another test for.
pub struct Daemon {
    pub sock: PathBuf,
    _child: ChildGuard,
    _dir: tempfile::TempDir,
}

impl Daemon {
    /// A client attached to session `id`.
    pub async fn connect(&self, id: &str) -> Ws {
        super::ws_connect_uds(&self.sock, Some(id)).await
    }
}

/// The renderer's connection to the daemon.
pub type Ws = super::TestWsUds;

pub async fn daemon(home: &Path, base: &str, session_dir: &Path, idle_secs: &str) -> Daemon {
    daemon_with(home, base, session_dir, idle_secs, &[]).await
}

/// [`daemon`] with extra `serve` flags (`--approve all`, say).
pub async fn daemon_with(
    home: &Path,
    base: &str,
    session_dir: &Path,
    idle_secs: &str,
    extra: &[&str],
) -> Daemon {
    daemon_opts(home, base, session_dir, idle_secs, extra, &[], None).await
}

/// [`daemon_with`], plus extra environment and (optionally) its stderr captured to a file.
pub async fn daemon_opts(
    home: &Path,
    base: &str,
    session_dir: &Path,
    idle_secs: &str,
    extra: &[&str],
    envs: &[(&str, &str)],
    stderr: Option<&Path>,
) -> Daemon {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("serve.sock");
    let child = Command::new(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .args([
            "serve",
            "--listen-uds",
            sock.to_str().unwrap(),
            "--gateway-url",
            base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session-dir",
            session_dir.to_str().unwrap(),
        ])
        .args(extra)
        .env("HOME", home)
        .env("BEYOND_AI_AGENT_MCP_IDLE_SECS", idle_secs)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .envs(envs.iter().copied())
        .stderr(match stderr {
            Some(path) => Stdio::from(std::fs::File::create(path).unwrap()),
            None if std::env::var_os("MCP_APPS_TEST_STDERR").is_some() => Stdio::inherit(),
            None => Stdio::null(),
        })
        .spawn_guarded();
    for _ in 0..2000 {
        if tokio::net::UnixStream::connect(&sock).await.is_ok() {
            return Daemon {
                sock,
                _child: child,
                _dir: dir,
            };
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("daemon socket {} never came up", sock.display());
}

pub async fn command(ws: &mut Ws, cmd: Value) -> Value {
    let name = cmd["type"].as_str().unwrap().to_owned();
    ws_send(ws, cmd).await;
    let frames = tokio::time::timeout(Duration::from_secs(30), ws_read_until_response(ws, &name))
        .await
        .expect("response in time");
    frames.last().cloned().unwrap()
}

pub async fn declare_apps(ws: &mut Ws) -> Value {
    let r = command(ws, json!({ "type": "set_mcp_apps", "enabled": true })).await;
    assert_eq!(r["success"], true, "{r}");
    r
}

pub async fn app_request(
    ws: &mut Ws,
    server: &str,
    app_id: &str,
    method: &str,
    params: Value,
) -> Value {
    command(
        ws,
        json!({
            "type": "mcp_app_request",
            "id": "bridge",
            "server": server,
            "app_id": app_id,
            "request": { "method": method, "params": params },
        }),
    )
    .await
}

/// The app id of view `n` opened by [`views_model`]'s `zqopen{n}z` prompt.
pub fn view_id(n: usize) -> String {
    format!("toolu_v{n}_x")
}

/// A model server that opens a `show_weather` view (app id [`view_id`]`(n)`) for each prompt
/// `zqopen{n}z`, answers its tool result with `done`, and names the session `title`. Routed by
/// request body — first match wins, latest view first, because a later prompt's body carries every
/// earlier view's ids in its history. `extra` routes are checked before these.
pub fn views_model(
    views: usize,
    extra: Vec<(String, String)>,
) -> (String, std::sync::Arc<std::sync::Mutex<Vec<String>>>) {
    let mut routes = extra;
    routes.push(("Title:".into(), super::turn_text("title")));
    for n in (1..=views).rev() {
        routes.push((view_id(n), super::turn_text("done")));
        routes.push((
            format!("zqopen{n}z"),
            super::turn_tool_use(
                &view_id(n),
                "mcp__wx__show_weather",
                &json!({ "city": "Oslo" }).to_string(),
            ),
        ));
    }
    super::spawn_model_server_routed(routes, super::turn_text("ok"))
}

/// Open view `n` (see [`views_model`]) and return its app id, once its `mcp_app_open` arrived.
pub async fn open_view(ws: &mut Ws, n: usize) -> String {
    let mut frames = prompt(ws, &format!("zqopen{n}z")).await;
    let id = view_id(n);
    let opened = |frames: &[Value]| {
        frames
            .iter()
            .any(|f| f["type"] == "mcp_app_open" && f["app_id"] == id.as_str())
    };
    // The model's result never waits for its view, so when the view's read loses the race to the
    // tool call, its frames follow the prompt's response. Wait for them.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    while !opened(&frames) {
        match tokio::time::timeout_at(deadline, super::ws_next_frame(ws)).await {
            Ok(Some(f)) => frames.push(f),
            _ => panic!("view {id} did not open: {frames:#?}"),
        }
    }
    // And its result, so the next command's frames start clean.
    while !frames
        .iter()
        .any(|f| f["type"] == "mcp_app_result" && f["app_id"] == id.as_str())
    {
        match tokio::time::timeout_at(deadline, super::ws_next_frame(ws)).await {
            Ok(Some(f)) => frames.push(f),
            _ => panic!("view {id} never got its result: {frames:#?}"),
        }
    }
    id
}

pub async fn prompt(ws: &mut Ws, message: &str) -> Vec<Value> {
    ws_send(ws, json!({ "type": "prompt", "message": message })).await;
    tokio::time::timeout(
        Duration::from_secs(60),
        ws_read_until_response(ws, "prompt"),
    )
    .await
    .expect("prompt finishes in time")
}

pub fn of_type<'a>(frames: &'a [Value], ty: &str) -> Vec<&'a Value> {
    frames.iter().filter(|f| f["type"] == ty).collect()
}

pub fn tool_end_text(frames: &[Value]) -> String {
    frames
        .iter()
        .filter(|f| f["type"] == "event" && f["event"]["kind"] == "tool_end")
        .filter_map(|f| f["event"]["result"].as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

/// The request body (from a recorded model request) whose last message contains `needle`.
pub fn request_with(bodies: &[String], needle: &str) -> Value {
    bodies
        .iter()
        .map(|b| super::body_json(b))
        .find(|b| {
            b["messages"]
                .as_array()
                .and_then(|m| m.last())
                .is_some_and(|m| m.to_string().contains(needle))
        })
        .unwrap_or_else(|| panic!("no request's last message carried `{needle}`"))
}

/// `value` without any `cache_control` markers — those move to the newest turn each request, by
/// design; every other byte of a cached prefix must not.
pub fn strip_cache_control(mut value: Value) -> Value {
    match &mut value {
        Value::Object(map) => {
            map.remove("cache_control");
            for v in map.values_mut() {
                *v = strip_cache_control(v.take());
            }
        }
        Value::Array(items) => {
            for v in items.iter_mut() {
                *v = strip_cache_control(v.take());
            }
        }
        _ => {}
    }
    value
}

/// The text of every block of a request's newest message.
pub fn newest_turn_blocks(body: &Value) -> Vec<String> {
    body["messages"]
        .as_array()
        .and_then(|m| m.last())
        .and_then(|m| m["content"].as_array())
        .map(|blocks| {
            blocks
                .iter()
                .map(|b| {
                    b["text"]
                        .as_str()
                        .map_or_else(|| b.to_string(), str::to_owned)
                })
                .collect()
        })
        .unwrap_or_default()
}
