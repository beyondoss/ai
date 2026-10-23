//! The demo's grant minter against the agent's own verifier: a grant it mints is accepted, with
//! every claim and secret it was given, and a tampered one is refused.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use beyond_ai_agent::grant::{GrantError, GrantVerifier, SealKeySource};
use beyond_ai_demo::keys::{GrantKeys, MintArgs, mint_grant};
use beyond_ai_test_support::grant::Minter;

fn keys() -> GrantKeys {
    GrantKeys {
        grant_seed: [0x31; 32],
        fleet_secret: [0x32; 32],
        gateway_key: "bai_v1.demo-gateway-key".into(),
        dek: [0x33; 32],
        managed_secret: Some([0x34; 32]),
    }
}

fn args(tamper: bool) -> MintArgs {
    MintArgs {
        tenant: "demo-tenant".into(),
        session: "demo.abc".into(),
        shard: "demo".into(),
        exec_url: "http://sandbox.demo.local:8081/exec".into(),
        workspace: "/workspace".into(),
        ttl_secs: 600,
        tamper,
        exec_bearer: None,
        exec_headers: vec![],
        connectors: vec![],
        limits: None,
        bundle: None,
    }
}

/// The verifier exactly as `serve --service` builds it from the flags `up.sh` derives from the same
/// keys: `--grant-key <kid>=<public key>` and the seal key as base64 (`AI_AGENT_SEAL_KEY_B64`).
fn verifier(k: &GrantKeys) -> GrantVerifier {
    let dir = tempfile::tempdir().unwrap();
    let flag = Minter::with_keys(dir.path(), 1, k.grant_seed, k.fleet_secret).grant_key_flag();
    let seal_b64 = STANDARD.encode(k.fleet_secret);
    GrantVerifier::from_flags(
        &[flag],
        Some(SealKeySource::pick(None, Some(&seal_b64)).unwrap().unwrap()),
    )
    .unwrap()
    .unwrap()
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[test]
fn the_agent_accepts_a_minted_grant_with_everything_in_it() {
    let k = keys();
    let grant = verifier(&k)
        .verify(&mint_grant(&k, &args(false)).unwrap(), now())
        .unwrap();
    assert_eq!(grant.tenant, "demo-tenant");
    assert_eq!(grant.session_id, "demo.abc");
    assert_eq!(grant.home_shard, "demo");
    assert_eq!(grant.workspace_root, "/workspace");
    assert_eq!(grant.exec_url, "http://sandbox.demo.local:8081/exec");
    assert_eq!(grant.secrets.dek.expose(), &[0x33; 32]);
}

/// The bearer the public sandbox requires travels in the grant's sealed exec headers.
#[test]
fn the_exec_bearer_travels_in_the_grants_sealed_exec_headers() {
    let k = keys();
    let a = MintArgs {
        exec_bearer: Some("s3cret-token".into()),
        ..args(false)
    };
    let grant = verifier(&k)
        .verify(&mint_grant(&k, &a).unwrap(), now())
        .unwrap();
    let h = &grant.secrets.exec_headers;
    assert_eq!(h.len(), 1);
    assert_eq!(h[0].name, "Authorization");
    assert_eq!(h[0].value.expose(), "Bearer s3cret-token");
}

#[test]
fn a_tampered_grant_is_refused() {
    let k = keys();
    let err = verifier(&k)
        .verify(&mint_grant(&k, &args(true)).unwrap(), now())
        .unwrap_err();
    assert_eq!(err, GrantError::BadSignature);
}

#[test]
fn a_grant_from_other_keys_is_refused() {
    let theirs = GrantKeys {
        grant_seed: [0x41; 32],
        ..keys()
    };
    let err = verifier(&keys())
        .verify(&mint_grant(&theirs, &args(false)).unwrap(), now())
        .unwrap_err();
    assert_eq!(err, GrantError::BadSignature);
}

/// `keys` produces a `bai_v1` key the gateway's own keyring accepts under the public key it prints.
#[cfg(feature = "gateway-key")]
#[test]
fn generated_gateway_key_verifies_under_its_public_key() {
    use beyond_ai::__test::key::Keyring;
    use ed25519_dalek::VerifyingKey;

    let v = beyond_ai_demo::keys::generate().unwrap();
    let pubkey: [u8; 32] = STANDARD
        .decode(v["gateway_signing_pub"].as_str().unwrap())
        .unwrap()
        .try_into()
        .unwrap();
    let ring: Keyring = [(1u32, VerifyingKey::from_bytes(&pubkey).unwrap())]
        .into_iter()
        .collect();
    let id = ring.verify(v["gateway_key"].as_str().unwrap()).unwrap();
    assert_eq!((id.tenant_id, id.vpc_id), (1, 1));
    // The grant key flag is `<kid>=<base64 public key>` and the seal key decodes to 32 bytes.
    assert!(v["grant_key_flag"].as_str().unwrap().starts_with("1="));
    assert_eq!(
        STANDARD
            .decode(v["fleet_secret"].as_str().unwrap())
            .unwrap()
            .len(),
        32
    );
}

/// A forged callback token has the real kid and a wrong tag, so the agent's key refuses it for the
/// tag (a 401), not for an unknown kid (a 503); the unforged one verifies.
#[cfg(feature = "agent-lib")]
#[test]
fn a_forged_notify_token_fails_on_its_tag_and_a_real_one_verifies() {
    use beyond_ai_agent::notify_token::{NotifyKey, Verdict};

    let k = keys();
    // SAFETY-free: the minter reads the secret from the environment, so give it one through the
    // key type it wraps instead.
    let key = NotifyKey::from_secret_bytes(&k.fleet_secret);
    let (tenant, session, job) = ("demo-tenant", "demo.abc", "0123456789abcdef");
    let scope = beyond_ai_agent::exec_endpoint::jobs::JobScope::for_session(Some(tenant), session);
    let id = beyond_ai_agent::exec_endpoint::jobs::JobId::parse(job).unwrap();
    let real = key.mint(tenant, session, &id);
    assert!(matches!(
        key.verify(&real, tenant, session, scope.name(), job),
        Verdict::Valid
    ));
    let mut forged = real.clone();
    let at = forged.rfind('.').unwrap() + 1;
    forged.replace_range(at..=at, if &real[at..=at] == "A" { "B" } else { "A" });
    assert!(matches!(
        key.verify(&forged, tenant, session, scope.name(), job),
        Verdict::BadMac
    ));
}

fn connector(name: &str, url: &str) -> beyond_ai_demo::keys::Connector {
    beyond_ai_demo::keys::Connector {
        name: name.into(),
        url: url.into(),
        ..Default::default()
    }
}

/// Connectors, a sandbox flag, a sealed header and a `limits` claim all reach the verifier, and a
/// managed connector holds exactly its own `bmc_v1` token: the agent's verifier would refuse a
/// grant that gave it anything else.
#[test]
fn connectors_limits_and_a_managed_token_are_accepted_by_the_agents_verifier() {
    let k = keys();
    let a = MintArgs {
        connectors: vec![
            connector("deepwiki", "https://mcp.deepwiki.com/mcp"),
            beyond_ai_demo::keys::Connector {
                sandbox: true,
                headers: vec![("e2b-traffic-access-token".into(), "tt".into())],
                ..connector("browser", "https://8931-x.e2b.app/mcp")
            },
            beyond_ai_demo::keys::Connector {
                managed: true,
                ..connector("vault", "http://1.2.3.4:8082/managed/vault/mcp")
            },
        ],
        limits: Some(beyond_ai_demo::keys::Limits {
            models: Some(vec!["gpt-5-mini".into()]),
            tools: Some(vec!["read".into()]),
            max_steps: Some(9),
            max_mcp_json_servers: Some(4),
        }),
        ..args(false)
    };
    let grant = verifier(&k)
        .verify(&mint_grant(&k, &a).unwrap(), now())
        .unwrap();
    let flags: Vec<_> = grant
        .mcp
        .iter()
        .map(|c| (c.name.as_str(), c.sandbox, c.managed))
        .collect();
    assert_eq!(
        flags,
        [
            ("deepwiki", false, false),
            ("browser", true, false),
            ("vault", false, true)
        ]
    );
    assert_eq!(
        grant.limits.models.as_deref(),
        Some(&["gpt-5-mini".to_string()][..])
    );
    assert_eq!(grant.limits.max_steps, Some(9));
    assert_eq!(grant.limits.max_mcp_json_servers, Some(4));
    let vault = &grant.secrets.mcp_headers["vault"];
    assert_eq!(vault.len(), 1);
    assert!(vault[0].value.expose().starts_with("Bearer bmc_v1."));
    assert_eq!(
        grant.secrets.mcp_headers["browser"][0].name,
        "e2b-traffic-access-token"
    );
}

#[test]
fn a_managed_connector_without_the_shared_secret_is_not_minted() {
    let k = GrantKeys {
        managed_secret: None,
        ..keys()
    };
    let a = MintArgs {
        connectors: vec![beyond_ai_demo::keys::Connector {
            managed: true,
            ..connector("vault", "http://x/managed/vault/mcp")
        }],
        ..args(false)
    };
    assert!(mint_grant(&k, &a).is_err());
}

/// `inspect-grant` unseals with the agent's own verifier and can find a needle where it is, and say
/// it is not where it is not: the managed token is inside, the third party's key is not, and the
/// sealed bytes never show up in the token's own text.
#[cfg(feature = "agent-lib")]
#[test]
fn inspecting_a_managed_grant_finds_the_token_and_not_a_third_party_key() {
    use beyond_ai_demo::keys::inspect_grant;

    let k = keys();
    let a = MintArgs {
        connectors: vec![beyond_ai_demo::keys::Connector {
            managed: true,
            ..connector("vault", "http://1.2.3.4:8082/managed/vault/mcp")
        }],
        ..args(false)
    };
    let token = mint_grant(&k, &a).unwrap();
    let real = inspect_grant(&k, &token, "bsk_live_demo_").unwrap();
    let c = &real["connectors"][0];
    assert_eq!(c["managed"], true);
    assert_eq!(c["holds_a_bmc_token"], true);
    assert_eq!(c["header_values_contain_needle"], false);
    assert_eq!(real["token_text_contains_needle"], false);
    // The control: the needle that is there is found.
    let ctl = inspect_grant(&k, &token, "bmc_v1.").unwrap();
    assert_eq!(ctl["connectors"][0]["header_values_contain_needle"], true);
    assert_eq!(
        ctl["token_text_contains_needle"], false,
        "sealed, not in clear"
    );
    assert!(inspect_grant(&k, &token[..token.len() - 3], "x").is_err());
}

/// A bundle grant with no sandbox is recognisable from the inside: no exec endpoint, the bundle's hash.
#[cfg(feature = "agent-lib")]
#[test]
fn inspecting_a_bundle_grant_shows_no_exec_endpoint_and_the_bundles_hash() {
    use beyond_ai_demo::keys::{Bundle, inspect_grant};

    let k = keys();
    let sha = "ab".repeat(32);
    let bundled = MintArgs {
        exec_url: String::new(),
        bundle: Some(Bundle {
            url: "https://example.invalid/b.tar".into(),
            sha256: sha.clone(),
        }),
        ..args(false)
    };
    let got = inspect_grant(&k, &mint_grant(&k, &bundled).unwrap(), "").unwrap();
    assert_eq!(got["exec_url_empty"], true);
    assert_eq!(got["bundle_sha256"], sha);
    let plain = inspect_grant(&k, &mint_grant(&k, &args(false)).unwrap(), "").unwrap();
    assert_eq!(plain["exec_url_empty"], false);
    assert!(plain["bundle_sha256"].is_null());
}
