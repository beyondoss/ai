//! The demo broker against the connector broker conformance kit (`BEYOND.md`, "Conformance"): the
//! kit's operator hook is this harness's control plane and the fake identity provider, in process.
use beyond_agent::conformance::broker::{self, BrokerTarget, Hooks, verify};
use beyond_agent::exec_endpoint::conformance::Verdict;
use beyond_agent::managed_mcp_token::{Claims, ManagedKey};
use serde_json::{Value, json};

use super::{Harness, MANAGED, harness};

impl Hooks for Harness {
    async fn mint(
        &self,
        secret_hex: Option<&str>,
        tenant: &str,
        session: &str,
        connector: &str,
        exp: u64,
    ) -> Result<String, String> {
        let secret: [u8; 32] = match secret_hex {
            Some(h) => (0..h.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&h[i..i + 2], 16).unwrap())
                .collect::<Vec<_>>()
                .try_into()
                .map_err(|_| "secret is not 32 bytes")?,
            None => MANAGED,
        };
        Ok(ManagedKey::from_secret_bytes(&secret).mint(&Claims {
            tenant: tenant.into(),
            session_id: session.into(),
            connector: connector.into(),
            exp,
        }))
    }

    async fn connect(&self, tenant: &str, user: &str) -> Result<String, String> {
        Ok(self
            .consent(tenant, &format!("sub-{user}"), user, None)
            .await)
    }

    async fn bind(&self, tenant: &str, session: &str, connection: &str) -> Result<(), String> {
        match self.bind(tenant, session, connection).await {
            (200, _) => Ok(()),
            (s, v) => Err(format!("bind: {s} {v}")),
        }
    }

    async fn revoke(&self, connection: &str) -> Result<(), String> {
        match self
            .control(
                "POST",
                &format!("/connections/{connection}/revoke"),
                Value::Null,
            )
            .await
        {
            (200, _) => Ok(()),
            (s, v) => Err(format!("revoke: {s} {v}")),
        }
    }

    async fn delete_session(&self, tenant: &str, session: &str) -> Result<(), String> {
        match self
            .control(
                "POST",
                "/sessions/unbind",
                json!({"tenant": tenant, "session": session}),
            )
            .await
        {
            (200, _) => Ok(()),
            (s, v) => Err(format!("unbind: {s} {v}")),
        }
    }

    /// The broker keeps the tail of every upstream token it handles, for exactly this scan.
    async fn secrets(&self, connection: &str) -> Result<Vec<String>, String> {
        let (_, v) = self
            .control(
                "GET",
                &format!("/connections/{connection}/fingerprints"),
                Value::Null,
            )
            .await;
        Ok(v["fingerprints"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|f| f["tail"].as_str().map(str::to_string))
            .collect())
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn the_demo_broker_passes_the_broker_kit() {
    let h = harness(60).await;
    let target = BrokerTarget {
        url: format!("http://{}/managed/notes/mcp", h.oidc.managed_addr),
        tool: Some(("my_notes".into(), json!({}))),
    };
    let report = verify(&target, &h).await;
    println!("{}", report.render());
    assert_eq!(report.checks.len(), broker::CHECKS.len());
    assert!(
        report.checks.iter().all(|c| c.verdict == Verdict::Pass),
        "{}",
        report.render()
    );
}
