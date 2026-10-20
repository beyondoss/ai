//! Key generation and grant minting for the demo.
//!
//! The grant minter is the repo's own (`beyond_ai_test_support::grant::Minter`, pinned to the
//! golden vectors); `tests/grant.rs` proves the agent's verifier accepts what it mints. Secrets are
//! read from the environment, never argv.

use base64::Engine as _;
use base64::engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD};
use beyond_ai_test_support::grant::{Claims, Minter, Secrets};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

#[cfg(feature = "gateway-key")]
fn random32() -> Res<[u8; 32]> {
    let mut b = [0u8; 32];
    getrandom::fill(&mut b)?;
    Ok(b)
}

fn env(name: &str) -> Res<String> {
    std::env::var(name).map_err(|_| format!("{name} is not set").into())
}

fn b32(name: &str) -> Res<[u8; 32]> {
    STANDARD
        .decode(env(name)?.trim())?
        .try_into()
        .map_err(|_| format!("{name} is not base64 of 32 bytes").into())
}

/// A scratch directory for `Minter`, which writes its seal-key file somewhere; removed at once.
fn scratch(tag: &str) -> Res<std::path::PathBuf> {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("beyond-demo-{tag}-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Everything `up.sh` stores in SSM, as one JSON object. Generated once; `up.sh` keeps what it
/// stores and never calls this again while those parameters exist.
#[cfg(feature = "gateway-key")]
pub fn generate() -> Res<serde_json::Value> {
    use beyond_ai::__test::key::{VirtualKey, mint};
    use ed25519_dalek::SigningKey;
    use serde_json::json;

    let (grant_seed, fleet_secret, gw_seed, id_key, dek) = (
        random32()?,
        random32()?,
        random32()?,
        random32()?,
        random32()?,
    );
    let dir = scratch("keys")?;
    let minter = Minter::with_keys(&dir, 1, grant_seed, fleet_secret);
    std::fs::remove_dir_all(&dir)?;

    let gw = SigningKey::from_bytes(&gw_seed);
    let vk = VirtualKey {
        tenant_id: 1,
        vpc_id: 1,
        key_id: None,
    };
    Ok(json!({
        "grant_seed": STANDARD.encode(grant_seed),
        "fleet_secret": STANDARD.encode(fleet_secret),
        "grant_key_flag": minter.grant_key_flag(),
        "gateway_signing_pub": STANDARD.encode(gw.verifying_key().to_bytes()),
        "gateway_id_signing_key": STANDARD.encode(id_key),
        "dek": STANDARD.encode(dek),
        "gateway_key": mint(&vk, 1, &gw),
    }))
}

/// One MCP connector the grant names.
#[derive(Clone, Debug, Default)]
pub struct Connector {
    pub name: String,
    pub url: String,
    /// The server sleeps when idle: the agent holds no standing connection to it.
    pub sandbox: bool,
    /// The URL is a managed-MCP endpoint: the grant carries a `bmc_v1` token for it and no other key.
    pub managed: bool,
    /// Headers the connector needs, sealed in the grant (never for a managed connector).
    pub headers: Vec<(String, String)>,
}

/// The grant's `limits` claim: ceilings on the model, tools and steps of a session.
#[derive(Clone, Debug, Default)]
pub struct Limits {
    pub models: Option<Vec<String>>,
    pub tools: Option<Vec<String>>,
    pub max_steps: Option<u32>,
}

#[derive(Default)]
pub struct MintArgs {
    pub tenant: String,
    pub session: String,
    pub shard: String,
    pub exec_url: String,
    pub workspace: String,
    pub ttl_secs: u64,
    pub tamper: bool,
    /// A bearer the exec endpoint requires: carried in the grant's sealed exec headers.
    pub exec_bearer: Option<String>,
    /// More headers for the exec endpoint, sealed in the grant (E2B's traffic token, for one).
    pub exec_headers: Vec<(String, String)>,
    pub connectors: Vec<Connector>,
    pub limits: Option<Limits>,
}

/// The secrets a grant is minted from: what `up.sh` put in SSM and `demo.sh` hands over.
pub struct GrantKeys {
    pub grant_seed: [u8; 32],
    pub fleet_secret: [u8; 32],
    pub gateway_key: String,
    pub dek: [u8; 32],
    /// The secret a managed-MCP endpoint shares with the minter; only managed connectors need it.
    pub managed_secret: Option<[u8; 32]>,
}

impl GrantKeys {
    /// From `DEMO_GRANT_SEED`, `DEMO_FLEET_SECRET`, `DEMO_GATEWAY_KEY` and `DEMO_DEK`.
    pub fn from_env() -> Res<Self> {
        Ok(Self {
            grant_seed: b32("DEMO_GRANT_SEED")?,
            fleet_secret: b32("DEMO_FLEET_SECRET")?,
            gateway_key: env("DEMO_GATEWAY_KEY")?,
            dek: b32("DEMO_DEK")?,
            managed_secret: match std::env::var("DEMO_MANAGED_SECRET") {
                Ok(_) => Some(b32("DEMO_MANAGED_SECRET")?),
                Err(_) => None,
            },
        })
    }
}

/// One `bsg_v1` grant. With `tamper`, the signature's first byte is flipped: a grant a replica
/// must refuse.
pub fn mint_grant(keys: &GrantKeys, a: &MintArgs) -> Res<String> {
    let dir = scratch("grant")?;
    let minter = Minter::with_keys(&dir, 1, keys.grant_seed, keys.fleet_secret);
    std::fs::remove_dir_all(&dir)?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    let claims = Claims {
        tenant: a.tenant.clone(),
        session_id: a.session.clone(),
        home_shard: a.shard.clone(),
        workspace_root: a.workspace.clone(),
        exec_url: a.exec_url.clone(),
        mcp: a
            .connectors
            .iter()
            .map(|c| (c.name.clone(), c.url.clone()))
            .collect(),
        mcp_sandbox: names(&a.connectors, |c| c.sandbox),
        mcp_managed: names(&a.connectors, |c| c.managed),
        exp: now + a.ttl_secs,
        limits: a
            .limits
            .as_ref()
            .map(|l| beyond_ai_test_support::grant::Limits {
                models: l.models.clone(),
                tools: l.tools.clone(),
                max_steps: l.max_steps,
            }),
    };
    let mut mcp_headers = std::collections::BTreeMap::new();
    for c in &a.connectors {
        let mut headers = c.headers.clone();
        if c.managed {
            let exp = now + a.ttl_secs;
            let token = managed_token(keys, &a.tenant, &a.session, &c.name, exp)?;
            headers.push(("Authorization".to_string(), format!("Bearer {token}")));
        }
        if !headers.is_empty() {
            mcp_headers.insert(c.name.clone(), headers);
        }
    }
    let secrets = Secrets {
        mcp_headers,
        gateway_key: keys.gateway_key.clone(),
        dek: keys.dek,
        exec_headers: a
            .exec_bearer
            .iter()
            .map(|b| ("Authorization".to_string(), format!("Bearer {b}")))
            .chain(a.exec_headers.iter().cloned())
            .collect(),
    };
    let token = minter.mint(&claims, &secrets);
    if !a.tamper {
        return Ok(token);
    }
    let (signed, sig) = token.rsplit_once('.').ok_or("malformed token")?;
    let mut sig = URL_SAFE_NO_PAD.decode(sig)?;
    sig[0] ^= 1;
    Ok(format!("{signed}.{}", URL_SAFE_NO_PAD.encode(sig)))
}

fn names(cs: &[Connector], pick: impl Fn(&Connector) -> bool) -> Vec<String> {
    cs.iter()
        .filter(|c| pick(c))
        .map(|c| c.name.clone())
        .collect()
}

/// The `bmc_v1` token a managed connector holds in place of the third party's key: bound to this
/// tenant, session and connector, minted with the agent crate's own minter.
#[cfg(feature = "agent-lib")]
fn managed_token(
    keys: &GrantKeys,
    tenant: &str,
    session: &str,
    connector: &str,
    exp: u64,
) -> Res<String> {
    use beyond_ai_agent::managed_mcp_token::{Claims, ManagedKey};
    let secret = keys
        .managed_secret
        .ok_or("DEMO_MANAGED_SECRET is not set (a managed connector needs it)")?;
    Ok(ManagedKey::from_secret_bytes(&secret).mint(&Claims {
        tenant: tenant.into(),
        session_id: session.into(),
        connector: connector.into(),
        exp,
    }))
}

#[cfg(not(feature = "agent-lib"))]
fn managed_token(_: &GrantKeys, _: &str, _: &str, _: &str, _: u64) -> Res<String> {
    Err("this build has no managed-connector minter (feature agent-lib)".into())
}

/// A `bjn_v1` job-end callback token for `job` of `session`, minted with the fleet key from
/// `DEMO_FLEET_SECRET`. With `forge`, its tag's first character is changed: the kid is the real
/// one, the tag is not, which is what a forger who knows the format but not the key can make.
/// Returns `(token, scope)`; the scope is the session's own, as the body of a real callback names it.
#[cfg(feature = "agent-lib")]
pub fn notify_token(tenant: &str, session: &str, job: &str, forge: bool) -> Res<(String, String)> {
    use beyond_ai_agent::exec_endpoint::jobs::{JobId, JobScope};
    use beyond_ai_agent::notify_token::NotifyKey;

    let key = NotifyKey::from_secret_bytes(&b32("DEMO_FLEET_SECRET")?);
    let id = JobId::parse(job).ok_or("job id must be lowercase hex")?;
    let mut token = key.mint(tenant, session, &id);
    if forge {
        let at = token.rfind('.').ok_or("malformed token")? + 1;
        let swapped = if token.as_bytes()[at] == b'A' {
            "B"
        } else {
            "A"
        };
        token.replace_range(at..=at, swapped);
    }
    let scope = JobScope::for_session(Some(tenant), session)
        .name()
        .to_string();
    Ok((token, scope))
}

/// What a grant holds, without its secrets: the demo unseals a grant it minted (with the same keys
/// the agent unseals with) and looks for `needle` in everything sealed inside, in the claims and in
/// the token's own text. Header and key values are never printed; only their names and whether the
/// needle is in any of them.
#[cfg(feature = "agent-lib")]
pub fn inspect_grant(keys: &GrantKeys, token: &str, needle: &str) -> Res<serde_json::Value> {
    use beyond_ai_agent::grant::{GrantVerifier, SealKeySource};
    use serde_json::json;

    let dir = scratch("inspect")?;
    let flag = Minter::with_keys(&dir, 1, keys.grant_seed, keys.fleet_secret).grant_key_flag();
    std::fs::remove_dir_all(&dir)?;
    let seal = STANDARD.encode(keys.fleet_secret);
    let verifier = GrantVerifier::from_flags(
        &[flag],
        Some(SealKeySource::pick(None, Some(&seal))?.ok_or("no seal key")?),
    )?
    .ok_or("no verifier")?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    let grant = verifier
        .verify(token, now)
        .map_err(|e| format!("the grant does not verify: {e:?}"))?;
    let has = |s: &str| !needle.is_empty() && s.contains(needle);
    let connectors: Vec<_> = grant
        .mcp
        .iter()
        .map(|c| {
            let headers = grant.secrets.mcp_headers.get(&c.name);
            json!({
                "name": c.name,
                "url": c.url,
                "sandbox": c.sandbox,
                "managed": c.managed,
                "header_names": headers.map(|h| h.iter().map(|x| x.name.clone()).collect::<Vec<_>>()).unwrap_or_default(),
                "holds_a_bmc_token": headers.is_some_and(|h| h.iter().any(|x| x.value.expose().starts_with("Bearer bmc_v1."))),
                "header_values_contain_needle": headers.is_some_and(|h| h.iter().any(|x| has(x.value.expose()))),
            })
        })
        .collect();
    Ok(json!({
        "tenant": grant.tenant,
        "session_id": grant.session_id,
        "connectors": connectors,
        "exec_header_names": grant.secrets.exec_headers.iter().map(|h| h.name.clone()).collect::<Vec<_>>(),
        "exec_header_values_contain_needle": grant.secrets.exec_headers.iter().any(|h| has(h.value.expose())),
        "gateway_key_contains_needle": has(grant.secrets.gateway_key.expose()),
        "token_text_contains_needle": has(token),
        "limits": {
            "models": grant.limits.models,
            "tools": grant.limits.tools,
            "max_steps": grant.limits.max_steps,
        },
    }))
}
