//! A reference control plane and bundle store for the bundle kit's own tests
//! (`tests/bundle_conformance.rs`): what `BEYOND.md` asks of Beyond's deploy and store, small, plus
//! one broken mode per check. The deploy validates with the agent's own unpacker, the contract's
//! reading of a safe bundle. URLs are `http://127.0.0.1/<name>?exp=<ms>&sig=<hex>`, signed with a
//! secret of the mock's.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use beyond_agent::conformance::bundle::Hooks;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// How a deploy or store gets one requirement wrong (or [`Fault::None`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    None,
    /// The deploy stores whatever it is given.
    NoValidation,
    /// A bundle is named by something other than its bytes' hash.
    NamedByLength,
    /// A URL needs no signature.
    UnsignedUrl,
    /// A URL never expires.
    ImmortalUrl,
    /// A URL expires with the grant, leaving no fetch window.
    NoWindow,
    /// The deploy drops `.mcp-tools.json`.
    DropsMcpTools,
}

const WINDOW: Duration = Duration::from_secs(2);
const SECRET: &str = "mock-signing-secret";

pub struct BundleMock {
    store: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    addr: std::net::SocketAddr,
    fault: Fault,
}

impl BundleMock {
    pub const WINDOW: Duration = WINDOW;

    pub async fn start(fault: Fault) -> Self {
        let listener = beyond_test_support::ports::tokio_listener().await;
        let addr = listener.local_addr().unwrap();
        let store: Arc<Mutex<HashMap<String, Vec<u8>>>> = Arc::default();
        let shared = store.clone();
        tokio::spawn(async move {
            loop {
                let Ok((sock, _)) = listener.accept().await else {
                    return;
                };
                let store = shared.clone();
                tokio::spawn(async move { serve(sock, store, fault).await });
            }
        });
        Self { store, addr, fault }
    }
}

fn now_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis()
}

fn sign(name: &str, exp: u128) -> String {
    Sha256::digest(format!("{SECRET}|{name}|{exp}"))
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

impl Hooks for BundleMock {
    async fn deploy(
        &self,
        _tenant: &str,
        source: &std::path::Path,
    ) -> Result<Option<String>, String> {
        // Header checks need only the first blocks; a sparse oversize source is never read whole.
        let mut f = std::fs::File::open(source).map_err(|e| e.to_string())?;
        let len = f.metadata().map_err(|e| e.to_string())?.len();
        let mut bytes = Vec::new();
        if len > beyond_agent::bundle::MAX_BYTES as u64 {
            if self.fault != Fault::NoValidation {
                return Ok(None);
            }
            return Ok(Some("0".repeat(64)));
        }
        std::io::Read::read_to_end(&mut f, &mut bytes).map_err(|e| e.to_string())?;
        let scratch = tempfile::tempdir().map_err(|e| e.to_string())?;
        if self.fault != Fault::NoValidation
            && beyond_agent::bundle::unpack(&bytes, scratch.path()).is_err()
        {
            return Ok(None);
        }
        if self.fault == Fault::DropsMcpTools {
            // Rebuild without the file: blank its name so the entry is not at the root.
            if let Some(at) = bytes.windows(15).position(|w| w == b".mcp-tools.json") {
                bytes[at] = b'_';
                // Fix the checksum of that header.
                let start = at - at % 512;
                let sum: u32 = (0..512)
                    .map(|i| {
                        if (148..156).contains(&i) {
                            32
                        } else {
                            u32::from(bytes[start + i])
                        }
                    })
                    .sum();
                bytes[start + 148..start + 155].copy_from_slice(format!("{sum:06o}\0").as_bytes());
            }
        }
        let name = if self.fault == Fault::NamedByLength {
            format!("{:064}", bytes.len())
        } else {
            hex(&bytes)
        };
        self.store.lock().unwrap().insert(name.clone(), bytes);
        Ok(Some(name))
    }

    async fn url(&self, _tenant: &str, name: &str, ttl_secs: u64) -> Result<String, String> {
        let window = if self.fault == Fault::NoWindow {
            Duration::ZERO
        } else {
            WINDOW
        };
        let exp = now_ms() + u128::from(ttl_secs) * 1000 + window.as_millis();
        Ok(format!(
            "http://{}/{name}?exp={exp}&sig={}",
            self.addr,
            sign(name, exp)
        ))
    }
}

async fn serve(
    mut sock: tokio::net::TcpStream,
    store: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    fault: Fault,
) {
    let mut buf = Vec::new();
    while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
        let mut chunk = [0u8; 2048];
        match sock.read(&mut chunk).await {
            Ok(0) | Err(_) => return,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
        }
    }
    let head = String::from_utf8_lossy(&buf).into_owned();
    let target = head
        .lines()
        .next()
        .and_then(|l| l.split(' ').nth(1))
        .unwrap_or("/");
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let name = path.trim_start_matches('/');
    let param = |k: &str| {
        query
            .split('&')
            .find_map(|p| p.strip_prefix(&format!("{k}=")))
    };
    let exp: u128 = param("exp").and_then(|v| v.parse().ok()).unwrap_or(0);
    let signed = param("sig") == Some(sign(name, exp).as_str());
    let live = fault == Fault::ImmortalUrl || exp > now_ms();
    let body = store.lock().unwrap().get(name).cloned();
    let (status, body) = match body {
        Some(b) if fault == Fault::UnsignedUrl || (signed && live) => (200, b),
        _ => (403, Vec::new()),
    };
    let head = format!(
        "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = sock.write_all(head.as_bytes()).await;
    let _ = sock.write_all(&body).await;
}
