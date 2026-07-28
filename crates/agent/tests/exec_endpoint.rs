//! The remote exec endpoint, end to end, against a mock server that stands in for **any** provider.
//!
//! The agent is handed a URL and POSTs commands to it. Nothing here — and nothing in the code under
//! test — knows or cares what is behind that URL. The mock is deliberately ~30 lines: if standing up
//! a fake provider took more than that, the protocol would be too big.
//!
//! The assertion that matters most is [`bash_runs_on_the_endpoint_not_the_host`]: a toolset whose
//! `edit` lands remotely while its `bash` runs locally is not partially sandboxed, it is broken —
//! the model writes a file and then runs a command that cannot see it, and the one tool that most
//! needs containment isn't contained.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use beyond_ai_agent::exec_endpoint::{HttpExecRunner, TemplateRunner};
use beyond_ai_agent::tools::exec::CommandRunner;
use beyond_ai_agent::tools::fs::FsBackend;
use beyond_ai_agent::tools::fs::shell::ShellFs;
use beyond_ai_agent::tools::{ToolConfig, default_registry_with_config};
use serde_json::{Value, json};

/// A stand-in exec provider: accepts the protocol, runs the command in `root`, answers the protocol.
/// This is exactly the shim someone writes in front of Daytona, E2B, a container, or a CI runner.
async fn mock_provider(
    root: std::path::PathBuf,
    seen: Arc<AtomicUsize>,
) -> (String, Arc<AtomicUsize>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let hits = seen.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            let root = root.clone();
            let hits = hits.clone();
            tokio::spawn(async move {
                use tokio::io::{AsyncReadExt, AsyncWriteExt};
                let mut buf = Vec::new();
                let mut chunk = [0u8; 8192];
                // Read until the body is complete (headers, blank line, then Content-Length bytes).
                let (mut need, mut head_end) = (0usize, None);
                loop {
                    let Ok(n) = sock.read(&mut chunk).await else {
                        return;
                    };
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&chunk[..n]);
                    if head_end.is_none()
                        && let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n")
                    {
                        head_end = Some(p + 4);
                        let head = String::from_utf8_lossy(&buf[..p]).to_lowercase();
                        need = head
                            .lines()
                            .find_map(|l| l.strip_prefix("content-length:"))
                            .and_then(|v| v.trim().parse().ok())
                            .unwrap_or(0);
                    }
                    if let Some(h) = head_end
                        && buf.len() >= h + need
                    {
                        break;
                    }
                }
                let body = &buf[head_end.unwrap_or(buf.len()).min(buf.len())..];
                let req: Value = serde_json::from_slice(body).unwrap_or(json!({}));
                hits.fetch_add(1, Ordering::Relaxed);

                let program = req["command"].as_str().unwrap_or("true").to_string();
                let args: Vec<String> = req["args"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(str::to_string))
                            .collect()
                    })
                    .unwrap_or_default();
                let cwd = req["cwd"].as_str().map(str::to_string);

                let out = tokio::process::Command::new(&program)
                    .args(&args)
                    .current_dir(cwd.unwrap_or_else(|| root.to_string_lossy().into_owned()))
                    .output()
                    .await;
                let payload = match out {
                    Ok(o) => json!({
                        "exit_code": o.status.code().unwrap_or(-1),
                        "stdout": String::from_utf8_lossy(&o.stdout),
                        "stderr": String::from_utf8_lossy(&o.stderr),
                    }),
                    Err(e) => json!({ "exit_code": 127, "stdout": "", "stderr": e.to_string() }),
                };
                let body = payload.to_string();
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = sock.write_all(resp.as_bytes()).await;
                let _ = sock.shutdown().await;
            });
        }
    });
    (format!("http://{addr}/exec"), seen)
}

async fn registry_over(runner: Arc<dyn CommandRunner>) -> agent_core::ToolRegistry {
    let backend = ShellFs::connect(runner.clone()).await;
    default_registry_with_config(&ToolConfig {
        fs_backend: Some(Arc::new(backend) as Arc<dyn FsBackend>),
        command_runner: Some(runner),
        ..ToolConfig::new()
    })
}

async fn call(reg: &agent_core::ToolRegistry, tool: &str, input: Value) -> String {
    reg.get(tool)
        .expect("tool registered")
        .run(input)
        .await
        .unwrap_or_else(|e| panic!("{tool}: {e}"))
        .text
}

#[tokio::test]
async fn the_filesystem_tools_run_on_the_endpoint() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.rs"), "fn one() { NEEDLE }\n").unwrap();
    let (url, hits) = mock_provider(dir.path().to_path_buf(), Arc::new(AtomicUsize::new(0))).await;

    let reg = registry_over(Arc::new(HttpExecRunner::new(&url).unwrap())).await;
    let out = call(
        &reg,
        "grep",
        json!({ "pattern": "NEEDLE", "path": dir.path().to_str().unwrap() }),
    )
    .await;
    assert!(out.contains("a.rs"), "{out}");
    assert!(
        hits.load(Ordering::Relaxed) > 0,
        "the endpoint was never called"
    );
}

#[tokio::test]
async fn bash_runs_on_the_endpoint_not_the_host() {
    // The hole this closes. `bash` must reach the same machine as `edit`, or the model writes a file
    // and then runs a command that cannot see it — and the tool most in need of containment isn't
    // contained. Proven by a marker only the endpoint's side can observe.
    let dir = tempfile::tempdir().unwrap();
    let marker = "ENDPOINT-SIDE-ONLY-4c1f";
    std::fs::write(dir.path().join("marker.txt"), format!("{marker}\n")).unwrap();
    let (url, hits) = mock_provider(dir.path().to_path_buf(), Arc::new(AtomicUsize::new(0))).await;

    let reg = registry_over(Arc::new(HttpExecRunner::new(&url).unwrap())).await;
    let before = hits.load(Ordering::Relaxed);
    let out = call(&reg, "bash", json!({ "command": "cat marker.txt" })).await;

    assert!(
        out.contains(marker),
        "`bash` did not run on the endpoint: {out}"
    );
    assert!(
        hits.load(Ordering::Relaxed) > before,
        "`bash` never reached the endpoint — it ran locally"
    );
}

#[tokio::test]
async fn write_then_bash_see_the_same_filesystem() {
    // The coherence property: one machine, one filesystem. This is what a split toolset breaks.
    let dir = tempfile::tempdir().unwrap();
    let (url, _) = mock_provider(dir.path().to_path_buf(), Arc::new(AtomicUsize::new(0))).await;
    let reg = registry_over(Arc::new(HttpExecRunner::new(&url).unwrap())).await;

    let path = dir.path().join("written-by-the-tool.txt");
    call(
        &reg,
        "write",
        json!({ "path": path.to_str().unwrap(), "content": "hello from write\n" }),
    )
    .await;
    let seen = call(
        &reg,
        "bash",
        json!({ "command": "cat written-by-the-tool.txt" }),
    )
    .await;
    assert!(
        seen.contains("hello from write"),
        "`bash` cannot see what `write` just wrote: {seen}"
    );
}

#[tokio::test]
async fn an_unreachable_endpoint_fails_loudly_rather_than_looking_empty() {
    // Nothing listening. Every tool must report it, not return a plausible empty result.
    let runner = Arc::new(HttpExecRunner::new("http://127.0.0.1:1/exec").unwrap());
    let reg = registry_over(runner).await;
    let err = reg
        .get("ls")
        .unwrap()
        .run(json!({ "path": "/tmp" }))
        .await
        .expect_err("an unreachable endpoint must be an error");
    assert!(
        err.to_string().contains("127.0.0.1:1"),
        "the error must name the endpoint: {err}"
    );
}

#[tokio::test]
async fn auth_headers_reach_the_endpoint() {
    // Every real provider needs some form of auth; which form is theirs to decide.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    tokio::spawn(async move {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        if let Ok((mut sock, _)) = listener.accept().await {
            let mut buf = vec![0u8; 4096];
            let n = sock.read(&mut buf).await.unwrap_or(0);
            let _ = tx.send(String::from_utf8_lossy(&buf[..n]).into_owned());
            let body = json!({"exit_code":0,"stdout":"","stderr":""}).to_string();
            let _ = sock
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    )
                    .as_bytes(),
                )
                .await;
        }
    });

    let runner = HttpExecRunner::new(format!("http://{addr}/exec"))
        .unwrap()
        .with_header("Authorization", "Bearer secret-token")
        .with_header("X-Tenant", "acme");
    let _ = runner
        .run("true", &[], None, std::time::Duration::from_secs(10))
        .await;
    let req = rx.recv_timeout(std::time::Duration::from_secs(10)).unwrap();
    assert!(req.contains("authorization: Bearer secret-token"), "{req}");
    assert!(req.contains("x-tenant: acme"), "{req}");
}

#[tokio::test]
async fn a_command_template_reaches_a_target_with_no_http_surface() {
    // `env` stands in for `ssh host --` / `docker exec ctr` / `kubectl exec … --`: it runs whatever
    // follows it, which is the shape every CLI-based provider has.
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("t.txt"), "TEMPLATE-NEEDLE\n").unwrap();
    let reg = registry_over(Arc::new(TemplateRunner::parse("env {}").unwrap())).await;
    let out = call(
        &reg,
        "grep",
        json!({ "pattern": "TEMPLATE-NEEDLE", "path": dir.path().to_str().unwrap() }),
    )
    .await;
    assert!(out.contains("t.txt"), "{out}");
}

#[tokio::test]
async fn the_model_sees_the_same_toolset_over_an_endpoint_as_it_does_locally() {
    let dir = tempfile::tempdir().unwrap();
    let (url, _) = mock_provider(dir.path().to_path_buf(), Arc::new(AtomicUsize::new(0))).await;
    let remote = registry_over(Arc::new(HttpExecRunner::new(&url).unwrap()))
        .await
        .definitions();
    let local = default_registry_with_config(&ToolConfig::new()).definitions();
    assert_eq!(
        serde_json::to_string(&remote).unwrap(),
        serde_json::to_string(&local).unwrap(),
        "pointing at an endpoint must not change one byte of what the model is offered"
    );
}
