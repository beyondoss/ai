//! MCP Events (draft): a stdio events server that crashes is dialed again by its poll subscription.
//! The subscription's next poll asks for the server's connection; handing back the dead client
//! would leave a configured trigger silently dead until `serve` restarts.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::Stdio;
use std::time::{Duration, Instant};

use common::mcp_events_fixture::{
    Frames, fast_knobs, stdio_server, wait_active, wait_control_file, write_settings,
};
use common::{BIN, SpawnGuarded, serve_cmd, spawn_model_server_routed, turn_text};
use serde_json::json;

#[test]
fn a_crashed_stdio_events_server_is_redialed_by_its_poll_subscription() {
    let home = tempfile::tempdir().unwrap();
    let control_file = home.path().join("control");
    let server = stdio_server(
        "tickets",
        &control_file,
        json!({}),
        json!([{ "name": "ticket.updated", "arguments": { "project": "alpha" }, "delivery": "poll" }]),
    );
    write_settings(home.path(), json!([server]));
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let session = home.path().join("s.jsonl");
    let stderr = home.path().join("serve.stderr");
    let mut cmd = serve_cmd(BIN, &base, &session.to_string_lossy());
    // `fast_knobs` turns the idle reaper off: the redial must come from noticing the death.
    fast_knobs(&mut cmd)
        .env("HOME", home.path())
        .stderr(Stdio::from(std::fs::File::create(&stderr).unwrap()));
    let mut child = cmd.spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut frames = Frames::new(&mut child, Some(stderr.clone()));
    let first = wait_control_file(&control_file);
    wait_active(&mut stdin, &mut frames, 1);

    // `serve`'s only children are its MCP servers.
    let out = std::process::Command::new("pgrep")
        .args(["-P", &child.id().to_string()])
        .output()
        .unwrap();
    let kids: Vec<String> = String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .map(str::to_owned)
        .collect();
    assert!(!kids.is_empty(), "the events server was never started");
    // A fresh process writes its control address again.
    std::fs::remove_file(&control_file).unwrap();
    for kid in &kids {
        std::process::Command::new("kill")
            .args(["-9", kid])
            .status()
            .unwrap();
    }

    let deadline = Instant::now() + Duration::from_secs(30);
    let mut redialed = None;
    while Instant::now() < deadline {
        if let Ok(s) = std::fs::read_to_string(&control_file)
            && !s.is_empty()
        {
            redialed = Some(s);
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(
        redialed.is_some(),
        "the crashed events server (was {first}) was never redialed; serve stderr:\n{}",
        std::fs::read_to_string(&stderr).unwrap_or_default()
    );
    // …and the subscription is live on it again.
    wait_active(&mut stdin, &mut frames, 1);
}
