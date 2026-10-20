//! A managed MCP connector: the hosted agent calls an authenticated third-party tool without ever
//! holding the third-party credential.
//!
//! The grant carries only a session-scoped `bmc_v1` token. A fake managed endpoint verifies it and
//! injects the provider's real key upstream; the provider accepts nothing else. The proof is that the
//! call succeeds and the real key is then found nowhere the agent can reach: the decrypted grant, the
//! session store, the transcript, the replica's logs, or the replica's memory.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::sync::atomic::Ordering;

use beyond_ai_agent::grant::{GrantVerifier, SealKeySource};
use beyond_ai_agent::managed_mcp_token::{Claims, ManagedKey};
use common::managed_mcp::{process_memory_contains, spawn_endpoint, spawn_provider};
use common::service::{Options, Service, far_future, files_under};
use common::{
    spawn_model_server, turn_text, turn_tool_use, ws_connect_with_headers, ws_read_until_response,
    ws_send,
};
use serde_json::json;

const TENANT: &str = "tenant-a";
const SESSION: &str = "s1.managed";

fn real_token() -> String {
    let mut bytes = [0u8; 12];
    getrandom::fill(&mut bytes).unwrap();
    format!("REAL-PROVIDER-TOKEN-{}", hex::encode(bytes))
}

struct Run {
    bmc: String,
    real: String,
    tool_result: String,
    svc: Service,
    token: String,
    provider_refused: usize,
    endpoint_forwarded: usize,
    endpoint_refused: usize,
    // Held so the session, and the servers, outlive the assertions.
    _ws: common::TestWs,
}

/// Run one hosted session with a managed connector `vault`, make one `echo` call through it, and
/// leave the session open.
async fn managed_call(reflect: bool) -> Run {
    let real = real_token();
    let provider = spawn_provider(real.clone(), reflect).await;
    let key = ManagedKey::from_secret_bytes(&[0x61; 32]);
    let bmc = key.mint(&Claims {
        tenant: TENANT.into(),
        session_id: SESSION.into(),
        connector: "vault".into(),
        exp: far_future(),
    });
    let endpoint = spawn_endpoint(key, provider.url.clone(), real.clone()).await;

    let (base, _requests) = spawn_model_server(vec![
        turn_tool_use("toolu_m", "mcp__vault__echo", r#"{"text":"hello vault"}"#),
        turn_text("done"),
    ]);
    let svc = Service::start_with(
        &base,
        &["s1"],
        Options {
            extra_args: vec!["--mcp-allow-private".into()],
            capture_stdout: true,
            ..Default::default()
        },
    )
    .await;

    let mut claims = svc.claims(TENANT, SESSION, &svc.shards[0].0);
    claims.mcp = vec![("vault".into(), endpoint.url.clone())];
    claims.mcp_managed = vec!["vault".into()];
    let mut secrets = svc.secrets();
    secrets.mcp_headers.insert(
        "vault".into(),
        vec![("Authorization".into(), format!("Bearer {bmc}"))],
    );
    let token = svc.minter.mint(&claims, &secrets);
    let mut ws = ws_connect_with_headers(svc.port, Some(SESSION), &svc.header(&token))
        .await
        .unwrap_or_else(|status| panic!("managed grant refused with HTTP {status}"));
    ws_send(
        &mut ws,
        json!({ "type": "prompt", "id": "p1", "message": "call the vault" }),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "prompt").await;
    let end = frames
        .iter()
        .find(|f| f["type"] == "event" && f["event"]["kind"] == "tool_end")
        .unwrap_or_else(|| panic!("no tool_end: {frames:#?}"));
    assert_eq!(end["event"]["is_error"], false, "{end}");
    Run {
        tool_result: end["event"]["result"].as_str().unwrap().to_string(),
        bmc,
        real,
        svc,
        token,
        provider_refused: provider.refused.load(Ordering::SeqCst),
        endpoint_forwarded: endpoint.forwarded.load(Ordering::SeqCst),
        endpoint_refused: endpoint.refused.load(Ordering::SeqCst),
        _ws: ws,
    }
}

/// Where the real key must not be, and a positive control for each place that it is not.
fn assert_real_key_nowhere(run: &Run) {
    let real = run.real.as_bytes();

    // The decrypted grant: exactly the bmc token, nothing of the provider's.
    let verifier = GrantVerifier::from_flags(
        &[run.svc.minter.grant_key_flag()],
        Some(SealKeySource::File(run.svc.minter.seal_key())),
    )
    .unwrap()
    .unwrap();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let grant = verifier.verify(&run.token, now).unwrap();
    let headers = &grant.secrets.mcp_headers["vault"];
    assert_eq!(headers.len(), 1);
    assert_eq!(headers[0].value.expose(), format!("Bearer {}", run.bmc));
    assert!(!headers[0].value.expose().contains(&run.real));

    // The session store and every file on the mount, raw and decoded.
    let (_store, session) = run
        .svc
        .tenant_repo("s1", TENANT)
        .open_id_read_only(SESSION)
        .unwrap();
    let transcript = serde_json::to_string(&session).unwrap();
    assert!(
        transcript.contains("hello vault"),
        "the call must be in the transcript: {transcript}"
    );
    assert!(!transcript.contains(&run.real), "transcript holds the key");
    for file in files_under(run.svc.dir.path()) {
        let bytes = std::fs::read(&file).unwrap_or_default();
        assert!(
            !bytes.windows(real.len()).any(|w| w == real),
            "{} holds the real key",
            file.display()
        );
    }
    // The logs exist and were captured (not an empty file that proves nothing).
    assert!(run.svc.stderr_log.exists() && run.svc.stdout_log.exists());

    // The replica's memory. Positive control first: it does hold the bmc token, so the scan reads.
    if cfg!(target_os = "linux") {
        let pid = run.svc.child.id();
        assert!(
            process_memory_contains(pid, run.bmc.as_bytes()),
            "the scan must be able to see the bmc token the replica holds"
        );
        assert!(
            !process_memory_contains(pid, real),
            "the replica's memory holds the real key"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_managed_connector_calls_the_provider_without_the_agent_ever_holding_its_key() {
    let run = managed_call(false).await;
    assert!(
        run.tool_result.contains("hello vault"),
        "{}",
        run.tool_result
    );
    assert!(run.endpoint_forwarded > 0);
    assert_eq!(run.endpoint_refused, 0);
    assert_eq!(
        run.provider_refused, 0,
        "the provider only ever saw its key"
    );
    assert_real_key_nowhere(&run);
}

/// A provider that reflects every request header into its tool result sees the injected key. The
/// managed endpoint must not pass it back: the agent sees its own `bmc_v1` token and nothing else.
#[tokio::test(flavor = "multi_thread")]
async fn a_provider_that_echoes_its_headers_shows_the_agent_only_the_bmc_token() {
    let run = managed_call(true).await;
    assert!(
        run.tool_result.contains(&run.bmc),
        "the reflected Authorization header should be the bmc token: {}",
        run.tool_result
    );
    assert!(
        !run.tool_result.contains(&run.real),
        "the real key reached the tool result: {}",
        run.tool_result
    );
    assert_real_key_nowhere(&run);
}

/// The endpoint is the only holder of the real key: a request carrying anything but a valid token
/// is refused before it reaches the provider.
#[tokio::test(flavor = "multi_thread")]
async fn the_endpoint_refuses_a_call_without_a_valid_bmc_token() {
    let real = real_token();
    let provider = spawn_provider(real.clone(), false).await;
    agent_core::ensure_provider();
    let endpoint = spawn_endpoint(
        ManagedKey::from_secret_bytes(&[0x61; 32]),
        provider.url,
        real,
    )
    .await;
    let status = reqwest::Client::new()
        .post(&endpoint.url)
        .header("authorization", "Bearer some-third-party-token")
        .body("{}")
        .send()
        .await
        .unwrap()
        .status();
    assert_eq!(status, 401);
    assert_eq!(
        provider.refused.load(Ordering::SeqCst),
        0,
        "never reached it"
    );
    assert_eq!(endpoint.refused.load(Ordering::SeqCst), 1);
}
