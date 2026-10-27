//! The stand-in managed endpoint against the agent's real MCP client, dialed the way a hosted
//! session dials its grant's connectors: the connector holds only its `bmc_v1` token, the call
//! reaches the provider (which wants the real secret), and the secret never comes back.
#![cfg(feature = "agent-lib")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::time::Duration;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use beyond_agent::grant::{GrantVerifier, SealKeySource};
use beyond_agent::tools::mcp::{self, McpEgress};
use beyond_agent::tools::web::ssrf::EgressPolicy;
use beyond_demo::keys::{Connector, GrantKeys, MintArgs, mint_grant};
use beyond_demo::managed::Standin;
use beyond_test_support::grant::Minter;
use beyond_test_support::ports;
use serde_json::json;

const REAL_SECRET: &str = "psk_live_must_never_reach_the_agent";

#[tokio::test(flavor = "multi_thread")]
async fn a_managed_connector_calls_through_the_endpoint_holding_no_third_party_key() {
    let managed_secret = [0x35u8; 32];
    let standin = Standin::serve(
        ports::tokio_listener().await,
        ports::tokio_listener().await,
        &STANDARD.encode(managed_secret),
        REAL_SECRET.into(),
    )
    .unwrap();

    let keys = GrantKeys {
        grant_seed: [0x31; 32],
        fleet_secret: [0x32; 32],
        gateway_key: "bai_v1.demo-gateway-key".into(),
        dek: [0x33; 32],
        managed_secret: Some(managed_secret),
    };
    let token = mint_grant(
        &keys,
        &MintArgs {
            tenant: "t".into(),
            session: "demo.s".into(),
            shard: "demo".into(),
            exec_url: "http://sandbox/exec".into(),
            workspace: "/workspace".into(),
            ttl_secs: 600,
            connectors: vec![Connector {
                name: "vault".into(),
                url: format!("http://{}/managed/vault/mcp", standin.managed_addr),
                managed: true,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let flag =
        Minter::with_keys(dir.path(), 1, keys.grant_seed, keys.fleet_secret).grant_key_flag();
    let seal = STANDARD.encode(keys.fleet_secret);
    let verifier = GrantVerifier::from_flags(
        &[flag],
        Some(SealKeySource::pick(None, Some(&seal)).unwrap().unwrap()),
    )
    .unwrap()
    .unwrap();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let grant = verifier.verify(&token, now).unwrap();
    assert!(
        !format!("{grant:?}{token}").contains(REAL_SECRET),
        "the grant holds the real secret"
    );

    let egress = McpEgress::new(EgressPolicy::new(true, &[])).unwrap();
    let (tools, _catalog, warnings) = mcp::connect_granted(
        &grant.mcp,
        &grant.secrets.mcp_headers,
        &egress,
        &grant.exec_url,
        mcp::host(),
        Duration::ZERO,
        &std::collections::BTreeMap::new(),
    )
    .await;
    assert!(warnings.is_empty(), "{warnings:?}");
    let tool = tools
        .snapshot()
        .iter()
        .find(|t| t.name() == "mcp__vault__account_status")
        .expect("the provider's tool is advertised through the endpoint")
        .clone();
    let out = tool.run(json!({"customer": "acme"})).await.unwrap();
    assert!(out.text.contains("vault_ref=VAULT-"), "{out:?}");
    assert!(out.text.contains("upstream_auth=verified"), "{out:?}");
    assert!(!out.text.contains(REAL_SECRET), "{out:?}");
}
