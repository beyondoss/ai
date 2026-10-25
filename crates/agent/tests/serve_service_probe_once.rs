//! A service session probes its sandbox once, however often its tools are used or rebuilt.
//!
//! Every exec request the endpoint saw is counted, so "probed again" is a number. The endpoint here
//! is a strict front: it refuses the protocol's optional extensions (`stream`, `job`) with a 400
//! instead of ignoring them. A refusal of an extension is an answer, not a sign the sandbox was
//! replaced; treating it as one made every call after it rebuild the target, probes and all.
#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::collections::BTreeMap;

use common::exec_mock::MockOptions;
use common::service::{Options, Service};
use common::{
    spawn_model_server, turn_text, turn_tool_use, ws_connect_with_headers, ws_read_until_response,
    ws_send,
};
use serde_json::json;

fn key(r: &common::exec_mock::ExecRequest) -> String {
    format!("{} {}", r.command(), r.args().join(" "))
}

#[tokio::test(flavor = "multi_thread")]
async fn a_front_that_refuses_extensions_is_probed_once_not_per_call() {
    let (base, _requests) = spawn_model_server(vec![
        turn_tool_use("t1", "bash", r#"{"command":"echo one"}"#),
        turn_text("a"),
        turn_tool_use("t2", "bash", r#"{"command":"echo two"}"#),
        turn_text("b"),
        turn_tool_use("t3", "bash", r#"{"command":"echo three"}"#),
        turn_text("c"),
    ]);
    let svc = Service::start_with(
        &base,
        &["s1"],
        Options {
            exec: Some(MockOptions {
                refuses_extensions: true,
                ..MockOptions::default()
            }),
            ..Options::default()
        },
    )
    .await;
    let token = svc.token("t1", "s1.alpha");
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .unwrap();
    for (i, cmd) in [
        json!({"type": "prompt", "message": "one"}),
        json!({"type": "set_model", "model": "claude-haiku-4-5"}),
        json!({"type": "prompt", "message": "two"}),
        json!({"type": "set_thinking", "budget": 2048}),
        json!({"type": "prompt", "message": "three"}),
    ]
    .into_iter()
    .enumerate()
    {
        let ty = cmd["type"].as_str().unwrap().to_string();
        let mut cmd = cmd;
        cmd["id"] = json!(format!("c{i}"));
        ws_send(&mut ws, cmd).await;
        ws_read_until_response(&mut ws, &ty).await;
    }
    // Only the short commands: the probes, not the scripts the filesystem tools send.
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for r in svc.exec.requests() {
        let k = key(&r);
        if k.len() < 60 {
            *counts.entry(k).or_default() += 1;
        }
    }
    for probe in [
        r#"-c printf %s "$HOME""#,
        "-c command -v bash",
        "rg --version",
        "cat",
        "true",
        "sh -c printf exec-v1.2",
    ] {
        let n = counts.get(probe).copied().unwrap_or(0);
        assert!(n <= 1, "`{probe}` was sent {n} times: {counts:#?}");
    }
    let homes = counts
        .iter()
        .filter(|(k, _)| k.contains(r#"printf %s "$HOME""#))
        .map(|(_, n)| n)
        .sum::<usize>();
    assert_eq!(homes, 1, "{counts:#?}");
}
