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

pub struct MintArgs {
    pub tenant: String,
    pub session: String,
    pub shard: String,
    pub exec_url: String,
    pub workspace: String,
    pub ttl_secs: u64,
    pub tamper: bool,
}

/// The secrets a grant is minted from: what `up.sh` put in SSM and `demo.sh` hands over.
pub struct GrantKeys {
    pub grant_seed: [u8; 32],
    pub fleet_secret: [u8; 32],
    pub gateway_key: String,
    pub dek: [u8; 32],
}

impl GrantKeys {
    /// From `DEMO_GRANT_SEED`, `DEMO_FLEET_SECRET`, `DEMO_GATEWAY_KEY` and `DEMO_DEK`.
    pub fn from_env() -> Res<Self> {
        Ok(Self {
            grant_seed: b32("DEMO_GRANT_SEED")?,
            fleet_secret: b32("DEMO_FLEET_SECRET")?,
            gateway_key: env("DEMO_GATEWAY_KEY")?,
            dek: b32("DEMO_DEK")?,
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
        mcp: Vec::new(),
        exp: now + a.ttl_secs,
    };
    let secrets = Secrets {
        gateway_key: keys.gateway_key.clone(),
        dek: keys.dek,
        ..Secrets::default()
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
