//! The demo's "agent acts as a signed-in user" stand-ins, in one process with four listeners:
//!
//! - **managed** (public): `POST /managed/<connector>/mcp`, the broker's call path ([`broker`]).
//! - **control** (the control plane's): consent, binding, revocation and the OAuth callback.
//! - **resource** (loopback): a notes service that trusts only a validated JWT ([`resource`]).
//! - the identity provider is a separate process (Keycloak); the broker and the resource reach it by
//!   its internal address and know it by its issuer.
//!
//! Secrets come from the environment, never argv: `DEMO_MANAGED_SECRET` (the `bmc_v1` key, base64 of
//! 32 bytes), `DEMO_BROKER_MASTER` (the key tenants' keys derive from, same shape), `DEMO_BROKER_ADMIN`
//! (the control plane's credential) and `DEMO_OIDC_CLIENT_SECRET` (the broker's client secret at the
//! provider). Every request is one timeline line on stdout; no line holds a token.

pub mod broker;
pub mod idp;
pub mod jwt;
pub mod resource;
pub mod vault;

use std::collections::BTreeMap;
use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use base64::Engine as _;
use beyond_agent::managed_mcp_token::ManagedKey;
use tokio::net::{TcpListener, TcpStream};

use crate::http::{Headers, Req, Res, read_request, respond};

pub struct Args {
    pub managed_listen: String,
    pub control_listen: String,
    pub resource_listen: String,
    /// `host:port` the provider is reached at.
    pub idp_internal: String,
    /// The provider's issuer URL: what its tokens say, and what discovery is read from.
    pub issuer: String,
    /// The broker's client id at the provider.
    pub client_id: String,
    /// The callback URL registered at the provider.
    pub redirect_uri: String,
    pub data_dir: PathBuf,
}

fn secret32(var: &str) -> Res<[u8; 32]> {
    let b64 = std::env::var(var).map_err(|_| format!("{var} is not set"))?;
    base64::engine::general_purpose::STANDARD
        .decode(b64.trim())?
        .try_into()
        .map_err(|_| format!("{var} is not base64 of 32 bytes").into())
}

pub async fn run(args: Args) -> Res<()> {
    let managed_secret = secret32("DEMO_MANAGED_SECRET")?;
    let master = secret32("DEMO_BROKER_MASTER")?;
    let admin = std::env::var("DEMO_BROKER_ADMIN").map_err(|_| "DEMO_BROKER_ADMIN is not set")?;
    let client_secret = std::env::var("DEMO_OIDC_CLIENT_SECRET")
        .map_err(|_| "DEMO_OIDC_CLIENT_SECRET is not set")?;
    // The provider starts alongside: wait for it to answer discovery.
    let mut idp = None;
    for _ in 0..120 {
        match idp::Idp::discover(
            &args.idp_internal,
            &args.issuer,
            &args.client_id,
            &client_secret,
        )
        .await
        {
            Ok(i) => {
                idp = Some(i);
                break;
            }
            Err(_) => tokio::time::sleep(Duration::from_secs(2)).await,
        }
    }
    let idp = idp.ok_or("the identity provider never answered discovery")?;
    let started = Oidc::start(
        &args,
        TcpListener::bind(&args.managed_listen).await?,
        TcpListener::bind(&args.control_listen).await?,
        TcpListener::bind(&args.resource_listen).await?,
        idp,
        &managed_secret,
        master,
        admin,
    )?;
    println!(
        "{}",
        serde_json::json!({"ts_ms": crate::http::now_ms(), "src": "broker", "event": "listening",
            "detail": {"managed": started.managed_addr, "control": started.control_addr, "resource": started.resource_addr,
                       "issuer": args.issuer}})
    );
    std::future::pending::<()>().await;
    Ok(())
}

/// A running broker and resource: the addresses they really bound.
pub struct Oidc {
    pub managed_addr: String,
    pub control_addr: String,
    pub resource_addr: String,
}

impl Oidc {
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        args: &Args,
        managed_l: TcpListener,
        control_l: TcpListener,
        resource_l: TcpListener,
        idp: idp::Idp,
        managed_secret: &[u8; 32],
        master: [u8; 32],
        admin: String,
    ) -> Res<Self> {
        let out = Self {
            managed_addr: managed_l.local_addr()?.to_string(),
            control_addr: control_l.local_addr()?.to_string(),
            resource_addr: resource_l.local_addr()?.to_string(),
        };
        let res = resource::Resource::new(&idp.issuer, &idp.internal, &idp.jwks_path);
        let broker = broker::Broker::new(
            broker::Config {
                resource_addr: out.resource_addr.clone(),
                redirect_uri: args.redirect_uri.clone(),
                admin,
                data_dir: args.data_dir.clone(),
                connectors: BTreeMap::from([("notes".to_string(), vec!["notes:read".to_string()])]),
            },
            idp,
            vault::Vault::new(master),
            ManagedKey::from_secret_bytes(managed_secret),
        );
        let b = broker.clone();
        tokio::spawn(accept(managed_l, move |r| {
            let b = b.clone();
            async move { b.managed(&r).await }
        }));
        let b = broker;
        tokio::spawn(accept(control_l, move |r| {
            let b = b.clone();
            async move { b.control(&r).await }
        }));
        tokio::spawn(accept(resource_l, move |r| {
            let res = res.clone();
            async move { res.handle(&r).await }
        }));
        Ok(out)
    }
}

async fn accept<F, Fut>(l: TcpListener, handler: F)
where
    F: Fn(Req) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = (u16, Headers, Vec<u8>)> + Send + 'static,
{
    let handler = Arc::new(handler);
    loop {
        let Ok((s, _)) = l.accept().await else {
            continue;
        };
        let handler = handler.clone();
        tokio::spawn(async move {
            let _ = serve(s, &*handler).await;
        });
    }
}

async fn serve<F, Fut>(mut s: TcpStream, handler: &F) -> Res<()>
where
    F: Fn(Req) -> Fut,
    Fut: Future<Output = (u16, Headers, Vec<u8>)>,
{
    let Some(req) = read_request(&mut s).await? else {
        return Ok(());
    };
    let (status, headers, body) = handler(req).await;
    respond(&mut s, status, &headers, &body).await
}

#[cfg(test)]
mod tests;
