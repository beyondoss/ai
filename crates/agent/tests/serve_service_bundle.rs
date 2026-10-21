//! A grant's optional `bundle` makes the agent a deployment: the folder (`AGENTS.md`, `.mcp.json`,
//! skills, subagents) is fetched once as a content-addressed tar, verified, unpacked into a cache
//! shared by every session that names it, and read from there, not through the sandbox. A grant with
//! a bundle and no exec endpoint is a session with no sandbox at all.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use beyond_ai_test_support::bundle::{BundleServer, Entry, dir, file, sha256_hex, tar};
use common::grant::Bundle;
use common::service::{Options, Service};
use common::{
    TestWs, advertised_tools, body_json, spawn_model_server, turn_text, turn_tool_use,
    ws_connect_with_headers, ws_next_frame, ws_read_until_response, ws_send,
};
use serde_json::json;

const AGENTS_MD: &str = "---\nmodel: claude-bundled\n---\nBUNDLE-PERSONA: answer like a pirate.\n";

/// The replica, with loopback egress open for the bundle server and a bundle cache of its own (the
/// cache is keyed by hash, so a shared one would make a second test's fetch a hit).
async fn service(base: &str) -> (Service, tempfile::TempDir) {
    let cache = tempfile::tempdir().unwrap();
    let svc = Service::start_with(
        base,
        &["s1"],
        Options {
            extra_args: vec!["--mcp-allow-private".into()],
            env: vec![("TMPDIR".into(), cache.path().to_string_lossy().into_owned())],
            ..Default::default()
        },
    )
    .await;
    (svc, cache)
}

fn well_formed() -> Vec<u8> {
    tar(&[
        file("AGENTS.md", AGENTS_MD),
        dir(".agents/"),
        dir(".agents/skills/"),
        dir(".agents/skills/greet/"),
        file(
            ".agents/skills/greet/SKILL.md",
            "---\nname: greet\ndescription: say hello in the bundle's way\n---\nSay hello.\n",
        ),
    ])
}

/// Open session `s1.alpha` on a grant carrying `bundle`; `sandbox` keeps the mock's exec endpoint.
async fn open(svc: &Service, bundle: Bundle, sandbox: bool) -> TestWs {
    let token = svc.token_with("t1", "s1.alpha", |c| {
        if !sandbox {
            c.exec_url = String::new();
        }
        c.bundle = Some(bundle);
    });
    ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .expect("the grant is valid, so the socket is accepted")
}

async fn first_error(ws: &mut TestWs) -> String {
    while let Some(frame) = ws_next_frame(ws).await {
        if frame["type"] == "error" {
            return frame["error"].as_str().unwrap_or_default().to_string();
        }
    }
    panic!("a session that fails to open must say so");
}

#[tokio::test]
async fn a_bundled_agent_with_no_sandbox_answers_from_its_bundle_and_calls_nothing() {
    let bytes = well_formed();
    let server = BundleServer::start(bytes.clone());
    let (base, requests) = spawn_model_server(vec![turn_text("arr")]);
    let (svc, _cache) = service(&base).await;
    // The sandbox's own copy of the definition must not be what the session reads.
    svc.write_in_workspace("AGENTS.md", "SANDBOX-PERSONA: answer like a poet.\n");

    let mut ws = open(
        &svc,
        Bundle {
            url: server.url.clone(),
            sha256: sha256_hex(&bytes),
        },
        false,
    )
    .await;
    ws_send(
        &mut ws,
        json!({"type":"prompt","id":"p1","message":"hello"}),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "prompt").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");

    let seen = requests.lock().unwrap().clone();
    let body = body_json(&seen[0]);
    assert_eq!(body["model"], "claude-bundled");
    let system = body["system"].to_string();
    assert!(system.contains("BUNDLE-PERSONA"), "{system}");
    assert!(!system.contains("SANDBOX-PERSONA"), "{system}");
    let mut tools = advertised_tools(&seen[0]);
    tools.retain(|t| t != "memory" && t != "todo");
    assert_eq!(
        tools,
        vec!["read".to_string()],
        "no sandbox: only the read-only `read` over the bundle"
    );
    assert!(svc.exec.requests().is_empty(), "no exec request was made");
    assert!(
        seen[0].contains("- greet: "),
        "the bundle's skill is listed: {}",
        seen[0]
    );
    assert_eq!(server.hits(), 1);
}

#[tokio::test]
async fn a_bundle_replaces_the_sandbox_as_the_source_of_the_definition() {
    let bytes = well_formed();
    let server = BundleServer::start(bytes.clone());
    let (base, requests) = spawn_model_server(vec![turn_text("arr")]);
    let (svc, _cache) = service(&base).await;
    svc.write_in_workspace("AGENTS.md", "SANDBOX-PERSONA: answer like a poet.\n");
    svc.write_in_workspace(
        ".mcp.json",
        r#"{"mcpServers":{"x":{"url":"http://x.test/"}}}"#,
    );

    let mut ws = open(
        &svc,
        Bundle {
            url: server.url.clone(),
            sha256: sha256_hex(&bytes),
        },
        true,
    )
    .await;
    ws_send(
        &mut ws,
        json!({"type":"prompt","id":"p1","message":"hello"}),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "prompt").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");

    let seen = requests.lock().unwrap().clone();
    let system = body_json(&seen[0])["system"].to_string();
    assert!(system.contains("BUNDLE-PERSONA"), "{system}");
    assert!(!system.contains("SANDBOX-PERSONA"), "{system}");
    assert!(
        seen[0].contains("- greet: "),
        "the bundle's skill is listed: {}",
        seen[0]
    );
    // The sandbox is only asked what it is (its home and shell), never for the definition.
    for request in svc.exec.requests() {
        let call = format!("{} {:?}", request.command(), request.args());
        for definition in ["AGENTS.md", ".mcp.json", "SKILL.md", ".agents", ".claude"] {
            assert!(!call.contains(definition), "exec was asked: {call}");
        }
    }
}

#[tokio::test]
async fn bytes_that_do_not_match_the_granted_hash_refuse_the_session() {
    let server = BundleServer::start(well_formed());
    let (base, requests) = spawn_model_server(vec![turn_text("never")]);
    let (svc, _cache) = service(&base).await;

    let mut ws = open(
        &svc,
        Bundle {
            url: server.url.clone(),
            sha256: sha256_hex(b"some other bytes"),
        },
        false,
    )
    .await;
    let error = first_error(&mut ws).await;
    assert!(error.contains("sha256"), "{error}");
    assert!(requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn an_unsafe_entry_refuses_the_session() {
    let cases: Vec<(&str, Vec<u8>)> = vec![
        (
            "..",
            tar(&[file("../escape.md", "x"), file("AGENTS.md", AGENTS_MD)]),
        ),
        ("absolute", tar(&[file("/etc/escape.md", "x")])),
        (
            "symlink",
            tar(&[Entry {
                path: "AGENTS.md",
                kind: b'2',
                body: b"",
                link: "/etc/passwd",
            }]),
        ),
        (
            "hardlink",
            tar(&[Entry {
                path: "AGENTS.md",
                kind: b'1',
                body: b"",
                link: "other",
            }]),
        ),
        (
            "device",
            tar(&[Entry {
                path: "dev",
                kind: b'3',
                body: b"",
                link: "",
            }]),
        ),
    ];
    for (name, bytes) in cases {
        let server = BundleServer::start(bytes.clone());
        let (base, requests) = spawn_model_server(vec![turn_text("never")]);
        let (svc, _cache) = service(&base).await;
        let mut ws = open(
            &svc,
            Bundle {
                url: server.url.clone(),
                sha256: sha256_hex(&bytes),
            },
            false,
        )
        .await;
        let error = first_error(&mut ws).await;
        assert!(error.contains("unsafe"), "{name}: {error}");
        assert!(requests.lock().unwrap().is_empty(), "{name}");
        assert!(
            !svc.dir.path().join("escape.md").exists(),
            "{name}: nothing was written outside the cache"
        );
    }
}

#[tokio::test]
async fn sessions_naming_the_same_hash_share_one_fetch() {
    let bytes = well_formed();
    let server = BundleServer::start(bytes.clone());
    let (base, requests) = spawn_model_server(vec![turn_text("one"), turn_text("two")]);
    let (svc, _cache) = service(&base).await;
    let bundle = Bundle {
        url: server.url.clone(),
        sha256: sha256_hex(&bytes),
    };
    for (session, id) in [("s1.alpha", "p1"), ("s1.beta", "p2")] {
        let token = svc.token_with("t1", session, |c| {
            c.exec_url = String::new();
            c.bundle = Some(bundle.clone());
        });
        let mut ws = ws_connect_with_headers(svc.port, Some(session), &svc.header(&token))
            .await
            .unwrap();
        ws_send(&mut ws, json!({"type":"prompt","id":id,"message":"hello"})).await;
        let frames = ws_read_until_response(&mut ws, "prompt").await;
        assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    }
    assert_eq!(requests.lock().unwrap().len(), 2);
    assert_eq!(
        server.hits(),
        1,
        "the second session reuses the unpacked copy"
    );
}

#[tokio::test]
async fn a_skill_is_read_from_the_bundle_and_nothing_outside_it_is() {
    let bytes = tar(&[
        file("AGENTS.md", AGENTS_MD),
        file(
            ".agents/skills/greet/SKILL.md",
            "---\nname: greet\ndescription: say hello\n---\nSKILL-BODY-MARKER\n",
        ),
        file(".agents/skills/greet/extra.md", "SKILL-EXTRA-MARKER"),
    ]);
    let server = BundleServer::start(bytes.clone());
    let (base, requests) = spawn_model_server(vec![
        turn_tool_use("t1", "read", r#"{"path":"skill:greet"}"#),
        turn_tool_use("t2", "read", r#"{"path":"skill:greet/extra.md"}"#),
        turn_tool_use("t3", "read", r#"{"path":"/etc/passwd"}"#),
        turn_text("done"),
    ]);
    let (svc, _cache) = service(&base).await;
    let mut ws = open(
        &svc,
        Bundle {
            url: server.url.clone(),
            sha256: sha256_hex(&bytes),
        },
        false,
    )
    .await;
    ws_send(&mut ws, json!({"type":"prompt","id":"p1","message":"hi"})).await;
    let frames = ws_read_until_response(&mut ws, "prompt").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");

    let seen = requests.lock().unwrap().clone();
    assert!(seen[1].contains("SKILL-BODY-MARKER"), "{}", seen[1]);
    assert!(seen[2].contains("SKILL-EXTRA-MARKER"), "{}", seen[2]);
    assert!(
        !seen[3].contains("root:"),
        "a path outside the bundle is refused"
    );
    assert!(svc.exec.requests().is_empty(), "no exec request was made");
}

#[tokio::test]
async fn an_allow_list_without_read_leaves_a_sandbox_free_session_with_no_skills() {
    let bytes = tar(&[
        file("AGENTS.md", "---\ntools: grep\n---\nPERSONA\n"),
        file(
            ".agents/skills/greet/SKILL.md",
            "---\nname: greet\ndescription: say hello\n---\nbody\n",
        ),
    ]);
    let server = BundleServer::start(bytes.clone());
    let (base, requests) = spawn_model_server(vec![turn_text("ok")]);
    let (svc, _cache) = service(&base).await;
    let mut ws = open(
        &svc,
        Bundle {
            url: server.url.clone(),
            sha256: sha256_hex(&bytes),
        },
        false,
    )
    .await;
    ws_send(&mut ws, json!({"type":"prompt","id":"p1","message":"hi"})).await;
    let frames = ws_read_until_response(&mut ws, "prompt").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    let seen = requests.lock().unwrap().clone();
    let mut tools = advertised_tools(&seen[0]);
    tools.retain(|t| t != "memory");
    assert!(
        tools.is_empty(),
        "the folder's allow-list omits read: {tools:?}"
    );
    assert!(!seen[0].contains("- greet: "));
}
