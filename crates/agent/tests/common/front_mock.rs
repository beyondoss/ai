//! A reference exec front for the front kit's own tests (`tests/exec_front_conformance.rs`): the
//! behavior `BEYOND.md` asks of Beyond's front, in about two hundred lines, plus one broken mode per
//! check. The kit must pass the conforming front and fail each broken one by name.
//!
//! Tokens are `Authorization: Bearer tok.<tenant>.<session>.<epoch>`, the mock's own format: the kit
//! never sees inside one, it asks the hook.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use base64::Engine as _;
use beyond_agent::exec_endpoint::conformance::front::{Hooks, State};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// How a front gets one requirement wrong (or [`Fault::None`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    None,
    /// Every racing first request makes its own VM.
    NoSingleFlight,
    /// A create that fails after the VM exists leaves the VM and its route.
    LeaksOnFailedCreate,
    /// No tenant cap.
    NoCap,
    /// The cap answers 503 without `Retry-After`.
    CapWithoutRetryAfter,
    /// A VM that dies keeps its route entry.
    StaleRoute,
    /// An extension it cannot honor is a 500.
    ExtensionIs500,
    /// A create answers `200 {"status":"waking"}`.
    Placeholder,
    /// A create takes 1.2 s to answer (the test shrinks the probe to 500 ms).
    SlowCreate,
    /// No token needed.
    AcceptsMissingToken,
    /// A token that does not parse is accepted.
    AcceptsGarbageToken,
    /// A token for another sandbox is accepted.
    IgnoresSandboxScope,
    /// A token below the sandbox's epoch is accepted.
    IgnoresEpoch,
    /// The health path needs no token.
    HealthIsOpen,
    /// A refused token still creates the VM.
    CreatesBeforeAuth,
}

#[derive(Default)]
struct Inner {
    /// (id, tenant, session, alive)
    vms: Vec<(usize, String, String, bool)>,
    routes: HashMap<String, String>, // session -> tenant
    epoch: HashMap<String, u32>,
    fail_next: HashSet<String>,
    locks: HashMap<String, Arc<tokio::sync::Mutex<()>>>,
    stale_routes: bool,
}

pub struct FrontMock {
    pub addr: std::net::SocketAddr,
    inner: Arc<Mutex<Inner>>,
}

impl FrontMock {
    pub async fn start(fault: Fault, cap: usize) -> Self {
        let listener = beyond_test_support::ports::tokio_listener().await;
        let addr = listener.local_addr().unwrap();
        let inner = Arc::new(Mutex::new(Inner {
            stale_routes: fault == Fault::StaleRoute,
            ..Inner::default()
        }));
        let shared = inner.clone();
        tokio::spawn(async move {
            loop {
                let Ok((sock, _)) = listener.accept().await else {
                    return;
                };
                let inner = shared.clone();
                tokio::spawn(async move { serve(sock, inner, fault, cap).await });
            }
        });
        Self { addr, inner }
    }

    pub fn exec_url(&self) -> String {
        format!("http://{}/exec/{{session}}", self.addr)
    }

    pub fn health_url(&self) -> String {
        format!("http://{}/health/{{session}}", self.addr)
    }
}

impl Hooks for FrontMock {
    async fn token(&self, tenant: &str, session: &str) -> Result<Vec<(String, String)>, String> {
        let e = self
            .inner
            .lock()
            .unwrap()
            .epoch
            .get(session)
            .copied()
            .unwrap_or(0);
        Ok(vec![(
            "Authorization".into(),
            format!("Bearer tok.{tenant}.{session}.{e}"),
        )])
    }

    async fn state(&self, tenant: &str) -> Result<State, String> {
        let i = self.inner.lock().unwrap();
        Ok(State {
            vms: i.vms.iter().filter(|v| v.1 == tenant && v.3).count(),
            routes: i.routes.values().filter(|t| *t == tenant).count(),
        })
    }

    async fn fail_next_create(&self, _: &str, session: &str) -> Result<(), String> {
        self.inner.lock().unwrap().fail_next.insert(session.into());
        Ok(())
    }

    async fn kill_vm(&self, _: &str, session: &str) -> Result<(), String> {
        let mut i = self.inner.lock().unwrap();
        i.vms
            .iter_mut()
            .filter(|v| v.2 == session)
            .for_each(|v| v.3 = false);
        if !i.stale_routes {
            i.routes.remove(session);
        }
        Ok(())
    }

    async fn end(&self, _: &str, session: &str) -> Result<(), String> {
        let mut i = self.inner.lock().unwrap();
        i.vms
            .iter_mut()
            .filter(|v| v.2 == session)
            .for_each(|v| v.3 = false);
        i.routes.remove(session);
        *i.epoch.entry(session.into()).or_insert(0) += 1;
        Ok(())
    }
}

async fn read_request(
    sock: &mut tokio::net::TcpStream,
) -> Option<(String, String, HashMap<String, String>, Vec<u8>)> {
    let mut buf = Vec::new();
    let head_end = loop {
        let mut chunk = [0u8; 4096];
        let n = sock.read(&mut chunk).await.ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break p;
        }
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).into_owned();
    let mut lines = head.lines();
    let mut first = lines.next()?.split(' ');
    let (method, path) = (first.next()?.to_string(), first.next()?.to_string());
    let headers: HashMap<_, _> = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
        .collect();
    let len: usize = headers
        .get("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut body = buf[head_end + 4..].to_vec();
    while body.len() < len {
        let mut chunk = [0u8; 4096];
        let n = sock.read(&mut chunk).await.ok()?;
        if n == 0 {
            return None;
        }
        body.extend_from_slice(&chunk[..n]);
    }
    Some((method, path, headers, body))
}

async fn reply(
    sock: &mut tokio::net::TcpStream,
    status: u16,
    extra: &str,
    ctype: &str,
    body: &str,
) {
    let head =
        format!("HTTP/1.1 {status} X\r\n{extra}Content-Type: {ctype}\r\nConnection: close\r\n\r\n");
    let _ = sock.write_all(head.as_bytes()).await;
    let _ = sock.write_all(body.as_bytes()).await;
}

/// The (tenant, session, epoch) a token names, or None for anything else.
fn parse(token: &str) -> Option<(String, String, u32)> {
    let mut p = token.strip_prefix("tok.")?.rsplitn(2, '.');
    let epoch = p.next()?.parse().ok()?;
    let (tenant, session) = p.next()?.split_once('.')?;
    Some((tenant.into(), session.into(), epoch))
}

/// Check the token for `session`; `Err` is the status to answer. `Ok` is the tenant.
fn authorize(
    i: &Inner,
    headers: &HashMap<String, String>,
    session: &str,
    fault: Fault,
) -> Result<String, u16> {
    let Some(raw) = headers
        .get("authorization")
        .and_then(|v| v.strip_prefix("Bearer "))
    else {
        return if fault == Fault::AcceptsMissingToken {
            Ok("anon".into())
        } else {
            Err(401)
        };
    };
    let Some((tenant, tok_session, epoch)) = parse(raw) else {
        return if fault == Fault::AcceptsGarbageToken {
            Ok("anon".into())
        } else {
            Err(401)
        };
    };
    if tok_session != session && fault != Fault::IgnoresSandboxScope {
        return Err(403);
    }
    if epoch < i.epoch.get(&tok_session).copied().unwrap_or(0) && fault != Fault::IgnoresEpoch {
        return Err(401);
    }
    Ok(tenant)
}

async fn serve(
    mut sock: tokio::net::TcpStream,
    inner: Arc<Mutex<Inner>>,
    fault: Fault,
    cap: usize,
) {
    let Some((method, path, headers, body)) = read_request(&mut sock).await else {
        return;
    };
    if let Some(session) = path.strip_prefix("/health/") {
        let ok = fault == Fault::HealthIsOpen
            || authorize(&inner.lock().unwrap(), &headers, session, fault).is_ok();
        let (status, b) = if ok {
            (200, r#"{"ok":true}"#)
        } else {
            (401, "{}")
        };
        return reply(&mut sock, status, "", "application/json", b).await;
    }
    let Some(session) = path
        .strip_prefix("/exec/")
        .map(str::to_string)
        .filter(|_| method == "POST")
    else {
        return reply(&mut sock, 404, "", "application/json", "{}").await;
    };
    let auth = authorize(&inner.lock().unwrap(), &headers, &session, fault);
    let tenant = match auth {
        Ok(t) => t,
        Err(status) => {
            let named = headers
                .get("authorization")
                .and_then(|v| v.strip_prefix("Bearer "))
                .and_then(parse);
            if let (Fault::CreatesBeforeAuth, Some((tenant, ..))) = (fault, named) {
                let mut i = inner.lock().unwrap();
                let id = i.vms.len();
                i.vms.push((id, tenant.clone(), session.clone(), true));
                i.routes.insert(session.clone(), tenant);
            }
            return reply(&mut sock, status, "", "application/json", "{}").await;
        }
    };
    let req: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    let streams = match req.get("stream") {
        None => false,
        Some(Value::Bool(b)) => *b,
        Some(_) => return refuse_extension(&mut sock, fault).await,
    };
    if let Some(job) = req.get("job") {
        let valid = job.as_object().is_some_and(|j| {
            j.get("op")
                .and_then(Value::as_str)
                .is_some_and(|op| ["probe", "start", "keep"].contains(&op))
                && j.get("notify").is_none_or(Value::is_object)
        });
        if !valid {
            return refuse_extension(&mut sock, fault).await;
        }
    }
    // Get or create the session's VM, single-flight on the session.
    let lock = inner
        .lock()
        .unwrap()
        .locks
        .entry(session.clone())
        .or_default()
        .clone();
    let _guard = if fault == Fault::NoSingleFlight {
        None
    } else {
        Some(lock.lock_owned().await)
    };
    let alive = {
        let i = inner.lock().unwrap();
        i.routes.contains_key(&session) && i.vms.iter().any(|v| v.2 == session && v.3)
    };
    let created = !alive;
    if created {
        let over_cap = {
            let mut i = inner.lock().unwrap();
            i.routes.remove(&session); // any route left points at a VM that is gone
            let live = i.vms.iter().filter(|v| v.1 == tenant && v.3).count();
            fault != Fault::NoCap && live >= cap
        };
        if over_cap {
            let extra = if fault == Fault::CapWithoutRetryAfter {
                ""
            } else {
                "Retry-After: 2\r\n"
            };
            return reply(&mut sock, 503, extra, "application/json", "{}").await;
        }
        tokio::time::sleep(Duration::from_millis(150)).await; // the create takes a moment
        if fault == Fault::SlowCreate {
            tokio::time::sleep(Duration::from_millis(1200)).await;
        }
        let failed = {
            let mut i = inner.lock().unwrap();
            let id = i.vms.len();
            i.vms.push((id, tenant.clone(), session.clone(), true));
            i.routes.insert(session.clone(), tenant.clone());
            let failed = i.fail_next.remove(&session);
            if failed && fault != Fault::LeaksOnFailedCreate {
                i.vms[id].3 = false;
                i.routes.remove(&session);
            }
            failed
        };
        if failed {
            return reply(
                &mut sock,
                502,
                "",
                "application/json",
                r#"{"error":"create failed"}"#,
            )
            .await;
        }
        if fault == Fault::Placeholder {
            return reply(
                &mut sock,
                200,
                "",
                "application/json",
                r#"{"status":"waking"}"#,
            )
            .await;
        }
    }
    drop(_guard);
    let script = req["args"][1].as_str().unwrap_or("true").to_string();
    let out = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(&script)
        .output()
        .await;
    let Ok(out) = out else {
        return reply(&mut sock, 500, "", "application/json", "{}").await;
    };
    if streams {
        let b64 = base64::engine::general_purpose::STANDARD.encode(&out.stdout);
        let body = format!("{}\n{}\n", json!({"stdout": b64}), json!({"exit_code": 0}));
        reply(
            &mut sock,
            200,
            "Cache-Control: no-store\r\n",
            "application/x-ndjson",
            &body,
        )
        .await;
    } else {
        let body =
            json!({"exit_code": 0, "stdout": String::from_utf8_lossy(&out.stdout), "stderr": ""});
        reply(&mut sock, 200, "", "application/json", &body.to_string()).await;
    }
}

async fn refuse_extension(sock: &mut tokio::net::TcpStream, fault: Fault) {
    let status = if fault == Fault::ExtensionIs500 {
        500
    } else {
        422
    };
    reply(
        sock,
        status,
        "",
        "application/json",
        r#"{"error":"unsupported extension"}"#,
    )
    .await;
}
