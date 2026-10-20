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
    assert!(matches!(key.verify(&real, tenant, session, scope.name(), job), Verdict::Valid));
    let mut forged = real.clone();
    let at = forged.rfind('.').unwrap() + 1;
    forged.replace_range(at..=at, if &real[at..=at] == "A" { "B" } else { "A" });
    assert!(matches!(key.verify(&forged, tenant, session, scope.name(), job), Verdict::BadMac));
}
