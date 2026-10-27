//! The v1.4 (job notify) checks of [`super::verify`]: the endpoint's half of "Sandbox job notify
//! (v1.4)", measured from the outside. The verifier runs its own callback receiver (plain HTTP on
//! loopback by default, and a TLS listener whose certificate nobody trusts), keeps jobs with
//! `notify` pointing at it, and watches what arrives and when.
//!
//! Each check works in a scope of its own and on its own receiver key (`k=` in the query), so they
//! run side by side; [`IDLE`] alone runs last, by itself, since any request would wake the sandbox
//! it is measuring.

use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::{Check, Verifier, fail, pass, skip};
use crate::exec_endpoint::jobs::{JobId, JobNotify, JobScope, JobStatus, SandboxJobs as _};
use crate::exec_endpoint::jobs_native::NativeJobs;

pub const IDLE: &str = "a kept job survives idling";
pub const KEEP_INVALID: &str = "keep refuses an invalid notify";
pub const LATEST: &str = "notify follows the latest keep";
pub const SHAPE: &str = "callback request shape";
pub const SAME_URL: &str = "the proof and the end reach the same url";
pub const NO_WAIT: &str = "keep does not wait for its callback";
pub const FIRST: &str = "first callback at keep";
pub const AFTER_END: &str = "callback after the job ends";
pub const DURING: &str = "a change during an attempt is called back";
pub const NO_CALLBACK: &str = "no callback without notify";
pub const REDIRECTS: &str = "callback does not follow redirects";
pub const TLS: &str = "callback verifies TLS";
pub const RETRY: &str = "callback retry and stop";
pub const TOKEN: &str = "the token never reaches argv, environment or answers";

/// Every notify check, in the order the report lists them.
pub const CHECKS: [&str; 14] = [
    KEEP_INVALID,
    LATEST,
    SHAPE,
    SAME_URL,
    NO_WAIT,
    FIRST,
    AFTER_END,
    DURING,
    NO_CALLBACK,
    REDIRECTS,
    TLS,
    RETRY,
    TOKEN,
    IDLE,
];

/// How long after it should have begun an attempt may arrive and still count: the spec's 2 s plus
/// the path from the sandbox to the receiver.
const SLACK: Duration = Duration::from_millis(500);

/// The window in which an attempt that must not happen is watched for.
const QUIET: Duration = Duration::from_secs(3);

/// How long a callback that must arrive is waited for before its check fails. A conforming proof
/// comes within 2 s (and the check that times it says so); the rest is room for a loaded host, and
/// costs nothing when it comes sooner.
const PROOF_WAIT: Duration = Duration::from_secs(6);

/// The `keep` vectors: the refused cases are sent as written.
const KEEP_VECTORS: &str = include_str!("../../../../sandbox/tests/fixtures/exec/notify-keep.json");

/// A throwaway self-signed certificate (and its key) for `localhost` and `127.0.0.1`: no trust
/// store has it, so an endpoint that verifies the server must refuse the handshake.
const UNTRUSTED_CERT: &str = "-----BEGIN CERTIFICATE-----
MIIBvjCCAWOgAwIBAgIUM4GMLnALV1so/gDIKbXZfAykQ+EwCgYIKoZIzj0EAwIw
JTEjMCEGA1UEAwwaZXhlYy1jb25mb3JtYW5jZS11bnRydXN0ZWQwIBcNMjYxMDE4
MjM1ODI1WhgPMjEyNjA5MjQyMzU4MjVaMCUxIzAhBgNVBAMMGmV4ZWMtY29uZm9y
bWFuY2UtdW50cnVzdGVkMFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEgdFI6fgY
NK3Td9de8eYrsLWZh+tsbplP+MeoMEwprstqUamW6L3EiSa8E09owfOJVbhgHiVL
woQqU4J4d8SBiaNvMG0wHQYDVR0OBBYEFB5NIelo9QAeiZyAdVcMMqC7NW26MB8G
A1UdIwQYMBaAFB5NIelo9QAeiZyAdVcMMqC7NW26MA8GA1UdEwEB/wQFMAMBAf8w
GgYDVR0RBBMwEYIJbG9jYWxob3N0hwR/AAABMAoGCCqGSM49BAMCA0kAMEYCIQCM
LLyyP5SxI66dJYVVaKA1LeKN7z4LscBVm/h/G1oazAIhAJaVJYEGYYO0wSuH4qR6
bbML6FCur9CI5Sid7Gam6QG9
-----END CERTIFICATE-----
";
const UNTRUSTED_KEY: &str = "-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQg3RuAjux4ROyfQaTS
sqNgn1TNp/EKIGdLddE2JLfXau2hRANCAASB0Ujp+Bg0rdN3117x5iuwtZmH62xu
mU/4x6gwTCmuy2pRqZbovcSJJrwTT2jB84lVuGAeJUvChCpTgnh3xIGJ
-----END PRIVATE KEY-----
";

// ---------------------------------------------------------------------------------------------
// The receiver.
// ---------------------------------------------------------------------------------------------

/// How the receiver answers one callback.
#[derive(Clone, Debug)]
enum Answer {
    Status(u16),
    /// `302` to this URL.
    Redirect(String),
    /// Never answer.
    Hang,
    /// Answer with this status after this long.
    Slow(Duration, u16),
}

/// One request the receiver read.
#[derive(Clone, Debug)]
struct Received {
    method: String,
    target: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    peer: std::net::IpAddr,
    at: Instant,
}

impl Received {
    fn values(&self, name: &str) -> Vec<&str> {
        self.headers
            .iter()
            .filter(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
            .collect()
    }
}

#[derive(Default)]
struct Book {
    got: HashMap<String, Vec<Received>>,
    scripts: HashMap<String, VecDeque<Answer>>,
}

/// The verifier's callback receiver: every request, by the `k` its URL carries.
#[derive(Clone)]
struct Receiver {
    /// Where a URL's origin points: `http://<listen address>`, or the operator's origin.
    origin: String,
    book: Arc<Mutex<Book>>,
}

impl Receiver {
    async fn start(listen: &str, origin: Option<&str>) -> Result<Self, String> {
        let listener = tokio::net::TcpListener::bind(listen)
            .await
            .map_err(|e| format!("the callback receiver cannot listen on {listen}: {e}"))?;
        let addr = listener.local_addr().map_err(|e| e.to_string())?;
        let me = Self {
            origin: origin.map_or_else(
                || format!("http://{addr}"),
                |o| o.trim_end_matches('/').to_string(),
            ),
            book: Arc::default(),
        };
        let book = me.book.clone();
        tokio::spawn(async move {
            while let Ok((sock, peer)) = listener.accept().await {
                tokio::spawn(receive(sock, peer, book.clone()));
            }
        });
        Ok(me)
    }

    /// The callback URL for `key`: the session's address, as the replica sends it.
    fn url(&self, key: &str) -> String {
        format!(
            "{}/_beyond/agent?session_id=conformance-{key}&tenant=conformance&k={key}",
            self.origin
        )
    }

    fn script(&self, key: &str, answers: impl IntoIterator<Item = Answer>) {
        let mut b = self.book.lock().unwrap_or_else(|e| e.into_inner());
        b.scripts
            .entry(key.to_string())
            .or_default()
            .extend(answers);
    }

    fn got(&self, key: &str) -> Vec<Received> {
        let b = self.book.lock().unwrap_or_else(|e| e.into_inner());
        b.got.get(key).cloned().unwrap_or_default()
    }

    /// Wait (bounded) for `n` requests on `key`; all of them so far.
    async fn wait(&self, key: &str, n: usize, within: Duration) -> Vec<Received> {
        let deadline = Instant::now() + within;
        loop {
            let got = self.got(key);
            if got.len() >= n || Instant::now() >= deadline {
                return got;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}

/// The `k` of a request target.
fn key_of(target: &str) -> String {
    target
        .split_once('?')
        .map(|(_, q)| q)
        .unwrap_or("")
        .split('&')
        .find_map(|kv| kv.strip_prefix("k="))
        .unwrap_or("")
        .to_string()
}

async fn receive(mut sock: tokio::net::TcpStream, peer: SocketAddr, book: Arc<Mutex<Book>>) {
    let (mut buf, mut chunk) = (Vec::new(), [0u8; 8192]);
    let head_end = loop {
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break p;
        }
        if buf.len() > 64 * 1024 {
            return;
        }
        match sock.read(&mut chunk).await {
            Ok(0) | Err(_) => return,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
        }
    };
    let at = Instant::now();
    let head = String::from_utf8_lossy(&buf[..head_end]).into_owned();
    let mut lines = head.split("\r\n");
    let mut first = lines.next().unwrap_or("").split(' ');
    let (method, target) = (
        first.next().unwrap_or("").to_string(),
        first.next().unwrap_or("").to_string(),
    );
    let headers: Vec<(String, String)> = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(n, v)| (n.to_string(), v.trim().to_string()))
        .collect();
    let len: usize = headers
        .iter()
        .find(|(n, _)| n.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, v)| v.parse().ok())
        .unwrap_or(0)
        .min(64 * 1024);
    let mut body = buf[head_end + 4..].to_vec();
    while body.len() < len {
        match sock.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(n) => body.extend_from_slice(&chunk[..n]),
        }
    }
    let key = key_of(&target);
    let answer = {
        let mut b = book.lock().unwrap_or_else(|e| e.into_inner());
        b.got.entry(key.clone()).or_default().push(Received {
            method,
            target,
            headers,
            body,
            peer: peer.ip(),
            at,
        });
        b.scripts
            .get_mut(&key)
            .and_then(VecDeque::pop_front)
            .unwrap_or(Answer::Status(200))
    };
    let status_line =
        |s: u16| format!("HTTP/1.1 {s} X\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
    let resp = match answer {
        Answer::Status(s) => status_line(s),
        Answer::Slow(after, s) => {
            tokio::time::sleep(after).await;
            status_line(s)
        }
        Answer::Redirect(to) => format!(
            "HTTP/1.1 302 Found\r\nLocation: {to}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        ),
        Answer::Hang => {
            tokio::time::sleep(Duration::from_secs(120)).await;
            return;
        }
    };
    let _ = sock.write_all(resp.as_bytes()).await;
    let _ = sock.shutdown().await;
}

/// A TLS listener with [`UNTRUSTED_CERT`]: how many connections came, and how many got as far as a
/// request (a handshake the client completed, then bytes).
struct TlsTrap {
    port: u16,
    seen: Arc<Mutex<(usize, usize)>>,
}

impl TlsTrap {
    async fn start(listen_ip: std::net::IpAddr) -> Result<Self, String> {
        use rustls::pki_types::pem::PemObject as _;
        use rustls::pki_types::{CertificateDer, PrivateKeyDer};
        agent_core::ensure_provider();
        let cert =
            CertificateDer::from_pem_slice(UNTRUSTED_CERT.as_bytes()).map_err(|e| e.to_string())?;
        let key =
            PrivateKeyDer::from_pem_slice(UNTRUSTED_KEY.as_bytes()).map_err(|e| e.to_string())?;
        let config = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(vec![cert], key)
            .map_err(|e| e.to_string())?;
        let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(config));
        // port-0: held for the whole run; the URL `keep` sends names the port it got.
        let listener = tokio::net::TcpListener::bind((listen_ip, 0))
            .await
            .map_err(|e| e.to_string())?;
        let port = listener.local_addr().map_err(|e| e.to_string())?.port();
        let seen = Arc::new(Mutex::new((0, 0)));
        let s = seen.clone();
        tokio::spawn(async move {
            while let Ok((sock, _)) = listener.accept().await {
                s.lock().unwrap_or_else(|e| e.into_inner()).0 += 1;
                let (acceptor, s) = (acceptor.clone(), s.clone());
                tokio::spawn(async move {
                    if let Ok(mut tls) = acceptor.accept(sock).await {
                        let mut b = [0u8; 1];
                        if tls.read(&mut b).await.is_ok_and(|n| n > 0) {
                            s.lock().unwrap_or_else(|e| e.into_inner()).1 += 1;
                        }
                    }
                });
            }
        });
        Ok(Self { port, seen })
    }

    fn seen(&self) -> (usize, usize) {
        *self.seen.lock().unwrap_or_else(|e| e.into_inner())
    }
}

// ---------------------------------------------------------------------------------------------
// The checks.
// ---------------------------------------------------------------------------------------------

/// What the v1.4 run needs beyond the verifier itself.
struct Run<'a> {
    v: &'a Verifier,
    rx: Receiver,
    token: String,
}

/// Run the notify checks side by side, and the idle check last, alone.
pub(super) async fn checks(v: &Verifier, out: &mut Vec<Check>) {
    let rx = match Receiver::start(
        &v.expect.callback_listen,
        v.expect.callback_origin.as_deref(),
    )
    .await
    {
        Ok(rx) => rx,
        Err(e) => {
            out.extend(CHECKS.iter().copied().map(|n| fail(n, e.clone())));
            return;
        }
    };
    let run = Run {
        v,
        rx,
        token: token(&v.nonce),
    };
    let (keep_invalid, latest, (shape, first, after_end, same_url), no_wait, during) = tokio::join!(
        run.keep_invalid(),
        run.latest(),
        run.lifecycle(),
        run.no_wait(),
        run.during(),
    );
    let (no_callback, redirects, tls, retry, token) = tokio::join!(
        run.no_callback(),
        run.redirects(),
        run.tls(),
        run.retry(),
        run.token_secret(),
    );
    out.extend([
        keep_invalid,
        latest,
        shape,
        same_url,
        no_wait,
        first,
        after_end,
        during,
        no_callback,
        redirects,
        tls,
        retry,
        token,
    ]);
    out.push(run.idle().await);
}

/// A token for this run: `bjn_v1`-shaped, unique, and using every `token68` character class so a
/// rewrite of any of them shows.
fn token(nonce: &str) -> String {
    let mut bytes = [0u8; 24];
    let _ = getrandom::fill(&mut bytes);
    use base64::Engine as _;
    let mac = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes);
    let tag: String = nonce
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .take(8)
        .collect();
    format!("bjn_v1.{tag}.{mac}~+/==")
}

impl Run<'_> {
    fn jobs(&self) -> NativeJobs<'_> {
        NativeJobs::over(&self.v.runner)
    }

    fn notify(&self, key: &str) -> JobNotify {
        JobNotify {
            url: self.rx.url(key),
            token: self.token.clone(),
        }
    }

    /// A kept job running `script` in a scope of its own.
    async fn kept(&self, which: &str, script: &str) -> Result<(JobScope, JobId), String> {
        let scope = self.v.scope(which);
        let id = JobId::fresh();
        self.jobs()
            .start_kept(&scope, &id, &Verifier::spec(script))
            .await
            .map_err(|e| format!("start: {e}"))?;
        Ok((scope, id))
    }

    async fn keep(&self, scope: &JobScope, id: &JobId, num: u64, key: &str) -> Result<(), String> {
        self.jobs()
            .keep_notify(scope, id, num, "background", 0, &self.notify(key))
            .await
            .map_err(|e| format!("keep with notify: {e}"))
    }

    async fn status(&self, scope: &JobScope, id: &JobId) -> Result<JobStatus, String> {
        let l = self
            .jobs()
            .list(scope)
            .await
            .map_err(|e| format!("list: {e}"))?;
        Ok(l.iter()
            .find(|j| &j.id == id)
            .map_or(JobStatus::Gone, |j| j.status))
    }

    async fn done(&self, scope: &JobScope) {
        let _ = self.jobs().wipe(scope).await;
    }

    /// N4, N5: every refused vector is a `400` that records nothing; a good one is recorded.
    async fn keep_invalid(&self) -> Check {
        let r = async {
            let (scope, id) = self.kept("notify-invalid", "sleep 60").await?;
            let r = self.keep_invalid_on(&scope, &id).await;
            self.done(&scope).await;
            r
        }
        .await;
        match r {
            Ok(d) => pass(KEEP_INVALID, d),
            Err(e) => fail(KEEP_INVALID, e),
        }
    }

    async fn keep_invalid_on(&self, scope: &JobScope, id: &JobId) -> Result<String, String> {
        let file: Value = serde_json::from_str(KEEP_VECTORS).map_err(|e| e.to_string())?;
        let cases = file["cases"].as_array().cloned().unwrap_or_default();
        let mut refused = 0;
        for (i, case) in cases.iter().enumerate() {
            if case["expect"]["accepted"].as_bool() != Some(false) {
                continue;
            }
            let name = case["name"].as_str().unwrap_or("?");
            let num = 100 + i as u64;
            let job = json!({
                "op": "keep", "scope": scope.name(), "id": id.as_str(), "num": num,
                "kind": "background", "seen": 0, "notify": case["notify"],
            });
            match self.jobs().call(job).await {
                Err(e) if e.kind() == std::io::ErrorKind::InvalidInput => {}
                Err(e) => return Err(format!("{name}: want 400, got {e}")),
                Ok(_) => return Err(format!("{name}: accepted, want 400")),
            }
            let l = self
                .jobs()
                .list(scope)
                .await
                .map_err(|e| format!("list: {e}"))?;
            match l.iter().find(|j| &j.id == id) {
                Some(j) if j.num.is_none() => {}
                other => {
                    return Err(format!(
                        "{name}: refused, but the keep was recorded: {other:?}"
                    ));
                }
            }
            refused += 1;
        }
        self.keep(scope, id, 7, "notify-invalid").await?;
        let l = self
            .jobs()
            .list(scope)
            .await
            .map_err(|e| format!("list: {e}"))?;
        if !l.iter().any(|j| &j.id == id && j.num == Some(7)) {
            return Err("a valid notify's keep was not recorded".into());
        }
        Ok(format!(
            "{refused} refused notifies were each a 400 that recorded nothing; a valid one was kept"
        ))
    }

    /// N6: after a second keep with another URL, every attempt goes there.
    async fn latest(&self) -> Check {
        let r = async {
            let (scope, id) = self.kept("notify-latest", "sleep 60").await?;
            let r = async {
                self.keep(&scope, &id, 1, "latest-a").await?;
                if self.rx.wait("latest-a", 1, PROOF_WAIT).await.is_empty() {
                    return Err("no proof at the first keep's url".to_string());
                }
                self.keep(&scope, &id, 1, "latest-b").await?;
                let switched = Instant::now();
                if self.rx.wait("latest-b", 1, PROOF_WAIT).await.is_empty() {
                    return Err("no proof at the second keep's url".to_string());
                }
                self.jobs()
                    .kill(&scope, std::slice::from_ref(&id), false)
                    .await
                    .map_err(|e| format!("kill: {e}"))?;
                if self.rx.wait("latest-b", 2, PROOF_WAIT).await.len() < 2 {
                    return Err("the end did not reach the second keep's url".to_string());
                }
                let stale = self
                    .rx
                    .got("latest-a")
                    .iter()
                    .filter(|r| r.at > switched)
                    .count();
                if stale > 0 {
                    return Err(format!(
                        "{stale} attempt(s) went to the first keep's url after the second keep"
                    ));
                }
                Ok("the proof and the end went to the latest keep's url, none to the first's")
            }
            .await;
            self.done(&scope).await;
            r
        }
        .await;
        match r {
            Ok(d) => pass(LATEST, d),
            Err(e) => fail(LATEST, e),
        }
    }

    /// N11, N12, N14, N7, N8, from one job's life: the proof, its shape, a callback at the shell's
    /// exit (leftovers alive) and at the end, all to one url from one sender.
    async fn lifecycle(&self) -> (Check, Check, Check, Check) {
        let key = "life";
        let started = self.kept("notify-life", "sleep 1; sleep 4 & exit 0").await;
        let (scope, id) = match started {
            Ok(s) => s,
            Err(e) => {
                return (
                    fail(SHAPE, &e),
                    fail(FIRST, &e),
                    fail(AFTER_END, &e),
                    fail(SAME_URL, e),
                );
            }
        };
        if let Err(e) = self.keep(&scope, &id, 7, key).await {
            self.done(&scope).await;
            return (
                fail(SHAPE, &e),
                fail(FIRST, &e),
                fail(AFTER_END, &e),
                fail(SAME_URL, e),
            );
        }
        let answered = Instant::now();
        // When the shell's exit and the end became observable, as near as a list can tell.
        let (mut exited, mut ended) = (None, None);
        let deadline = answered + Duration::from_secs(9);
        while ended.is_none() && Instant::now() < deadline {
            let before = Instant::now();
            match self.status(&scope, &id).await {
                Ok(JobStatus::ShellExited(_)) => {
                    exited.get_or_insert(before);
                }
                Ok(JobStatus::Ended(_)) => {
                    ended = Some(before);
                }
                _ => {}
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        tokio::time::sleep(Duration::from_secs(2) + SLACK * 2).await;
        let got = self.rx.got(key);
        self.done(&scope).await;

        let first = match got.first() {
            None => fail(FIRST, "no callback arrived after keep"),
            Some(r) if r.at > answered + Duration::from_secs(2) + SLACK => fail(
                FIRST,
                format!(
                    "the proof arrived {} ms after keep was answered: want within 2 s",
                    r.at.duration_since(answered).as_millis()
                ),
            ),
            Some(_) => pass(FIRST, "the proof arrived within 2 s of keep's answer"),
        };
        let shape = match got.first() {
            None => fail(SHAPE, "no callback arrived to inspect"),
            Some(r) => match self.shape_of(r, &scope, &id, key) {
                Ok(()) => pass(
                    SHAPE,
                    "one POST to the url as given, the headers and the exact body, the token byte \
                     for byte",
                ),
                Err(e) => fail(SHAPE, e),
            },
        };
        let near = |moment: Option<Instant>| {
            moment.is_some_and(|m| {
                got.iter()
                    .skip(1)
                    .any(|r| r.at + SLACK >= m && r.at <= m + Duration::from_secs(2) + SLACK * 2)
            })
        };
        let after_end = match (exited, ended) {
            (_, None) => fail(AFTER_END, "the job never ended"),
            (None, Some(_)) => fail(AFTER_END, "the job was never listed `exited`"),
            (Some(_), Some(_)) if !near(exited) => fail(
                AFTER_END,
                "no callback within 2 s of the shell exiting with a live group (a held stream's \
                 `shell_exit`)",
            ),
            (Some(_), Some(_)) if !near(ended) => fail(
                AFTER_END,
                "no callback within 2 s of the job's end becoming observable",
            ),
            _ => pass(
                AFTER_END,
                "a callback within 2 s of the shell's exit with leftovers, and of the end",
            ),
        };
        let same_url = match got.as_slice() {
            [] | [_] => fail(
                SAME_URL,
                "the job's end never arrived to compare with its proof",
            ),
            [proof, rest @ ..] => {
                if let Some(r) = rest.iter().find(|r| r.target != proof.target) {
                    fail(
                        SAME_URL,
                        format!(
                            "the proof went to {} and a later one to {}",
                            proof.target, r.target
                        ),
                    )
                } else if let Some(r) = rest.iter().find(|r| r.peer != proof.peer) {
                    fail(
                        SAME_URL,
                        format!(
                            "the proof came from {} and a later one from {}",
                            proof.peer, r.peer
                        ),
                    )
                } else {
                    pass(
                        SAME_URL,
                        "the proof and the end reached one url from one address",
                    )
                }
            }
        };
        (shape, first, after_end, same_url)
    }

    fn shape_of(
        &self,
        r: &Received,
        scope: &JobScope,
        id: &JobId,
        key: &str,
    ) -> Result<(), String> {
        if r.method != "POST" {
            return Err(format!("method {}, want POST", r.method));
        }
        let url = self.rx.url(key);
        let want_target = url.find("/_beyond").map_or(url.as_str(), |i| &url[i..]);
        if r.target != want_target {
            return Err(format!("target {}, want {want_target} exactly", r.target));
        }
        if r.values("content-type") != ["application/json"] {
            return Err(format!("Content-Type {:?}", r.values("content-type")));
        }
        let auth = format!("Bearer {}", self.token);
        if r.values("authorization") != [auth.as_str()] {
            return Err(
                "Authorization is not exactly one `Bearer <token>`, the token as given".into(),
            );
        }
        if r.values("content-length").len() != 1 {
            return Err("no single Content-Length".into());
        }
        for h in ["transfer-encoding", "cookie"] {
            if !r.values(h).is_empty() {
                return Err(format!("sent {h}"));
            }
        }
        let want = exec_notify::body(scope.name(), id.as_str());
        if r.body != want.as_bytes() {
            return Err(format!(
                "body {:?}, want {want}",
                String::from_utf8_lossy(&r.body)
            ));
        }
        if r.headers
            .iter()
            .filter(|(n, _)| !n.eq_ignore_ascii_case("authorization"))
            .any(|(_, v)| v.contains(&self.token))
            || r.target.contains(&self.token)
        {
            return Err("the token appears outside Authorization".into());
        }
        Ok(())
    }

    /// N10: keep is answered while its proof's attempt hangs.
    async fn no_wait(&self) -> Check {
        let key = "nowait";
        self.rx.script(key, [Answer::Hang]);
        let r = async {
            let (scope, id) = self.kept("notify-nowait", "sleep 60").await?;
            let t = Instant::now();
            let kept =
                tokio::time::timeout(Duration::from_secs(5), self.keep(&scope, &id, 1, key)).await;
            let r = match kept {
                Err(_) => {
                    Err("keep was not answered within 5 s while its callback hung".to_string())
                }
                Ok(Err(e)) => Err(e),
                Ok(Ok(())) if self.rx.wait(key, 1, PROOF_WAIT).await.is_empty() => {
                    Err("no proof attempt arrived".to_string())
                }
                Ok(Ok(())) => Ok(format!(
                    "keep was answered in {} ms with its proof's attempt still hanging",
                    t.elapsed().as_millis()
                )),
            };
            self.done(&scope).await;
            r
        }
        .await;
        match r {
            Ok(d) => pass(NO_WAIT, d),
            Err(e) => fail(NO_WAIT, e),
        }
    }

    /// N12, N16: the job ends while the proof's attempt is in flight; a new attempt follows once
    /// that one is answered (its 2xx covers only the proof), never beside it.
    async fn during(&self) -> Check {
        let key = "during";
        let hold = Duration::from_secs(2);
        self.rx.script(key, [Answer::Slow(hold, 200)]);
        let r = async {
            let (scope, id) = self.kept("notify-during", "sleep 1").await?;
            self.keep(&scope, &id, 1, key).await?;
            let got = self.rx.wait(key, 2, hold + PROOF_WAIT).await;
            self.done(&scope).await;
            match got.as_slice() {
                [] => Err("no proof arrived".to_string()),
                [_] => Err(
                    "the end, during the proof's attempt, was never called back: an answer covers \
                     only what was owed when its attempt began"
                        .to_string(),
                ),
                [a, b, ..] => {
                    let gap = b.at.duration_since(a.at);
                    if gap + SLACK < hold {
                        Err(format!(
                            "a second attempt began {} ms into the first: one at a time",
                            gap.as_millis()
                        ))
                    } else if gap > hold + Duration::from_secs(2) + SLACK {
                        Err(format!(
                            "the next attempt came {} ms after the first began, not as soon as it \
                             was answered",
                            gap.as_millis()
                        ))
                    } else {
                        Ok("the end got an attempt of its own as soon as the proof's was answered")
                    }
                }
            }
        }
        .await;
        match r {
            Ok(d) => pass(DURING, d),
            Err(e) => fail(DURING, e),
        }
    }

    /// N13: nothing after a keep without notify, nor after removal.
    async fn no_callback(&self) -> Check {
        let r = async {
            let refuse = || std::iter::repeat_n(Answer::Status(503), 30);
            // A keep without notify removes it: the retries stop, and the end goes unannounced.
            let key = "quiet-keep";
            self.rx.script(key, refuse());
            let (scope, id) = self.kept("notify-quiet", "sleep 2").await?;
            self.keep(&scope, &id, 1, key).await?;
            if self.rx.wait(key, 1, PROOF_WAIT).await.is_empty() {
                self.done(&scope).await;
                return Err("no proof arrived".to_string());
            }
            self.jobs()
                .keep(&scope, &id, 1, "background", 0)
                .await
                .map_err(|e| format!("keep: {e}"))?;
            let after = Instant::now();
            tokio::time::sleep(QUIET).await;
            let late = self
                .rx
                .got(key)
                .iter()
                .filter(|r| r.at > after + Duration::from_millis(100))
                .count();
            self.done(&scope).await;
            if late > 0 {
                return Err(format!(
                    "{late} attempt(s) after a keep without notify (the job's end, or a retry)"
                ));
            }
            // Removed: nothing after it.
            let key = "quiet-removed";
            self.rx.script(key, refuse());
            let (scope, id) = self.kept("notify-removed", "sleep 60").await?;
            self.keep(&scope, &id, 1, key).await?;
            if self.rx.wait(key, 1, PROOF_WAIT).await.is_empty() {
                self.done(&scope).await;
                return Err("no proof arrived".to_string());
            }
            self.jobs()
                .kill(&scope, std::slice::from_ref(&id), true)
                .await
                .map_err(|e| format!("kill: {e}"))?;
            let after = Instant::now();
            tokio::time::sleep(QUIET).await;
            let late = self
                .rx
                .got(key)
                .iter()
                .filter(|r| r.at > after + Duration::from_millis(100))
                .count();
            self.done(&scope).await;
            if late > 0 {
                return Err(format!("{late} attempt(s) after the job was removed"));
            }
            Ok("nothing after a keep without notify, nor after kill with remove")
        }
        .await;
        match r {
            Ok(d) => pass(NO_CALLBACK, d),
            Err(e) => fail(NO_CALLBACK, e),
        }
    }

    /// N15: a 302 is a stop, and its location is never contacted.
    async fn redirects(&self) -> Check {
        let (key, elsewhere) = ("redirect", "redirect-target");
        self.rx
            .script(key, [Answer::Redirect(self.rx.url(elsewhere))]);
        let r = async {
            let (scope, id) = self.kept("notify-redirect", "sleep 60").await?;
            self.keep(&scope, &id, 1, key).await?;
            let first = self.rx.wait(key, 1, PROOF_WAIT).await;
            tokio::time::sleep(QUIET).await;
            let (here, there) = (self.rx.got(key).len(), self.rx.got(elsewhere).len());
            self.done(&scope).await;
            if first.is_empty() {
                Err("no proof arrived".to_string())
            } else if there > 0 {
                Err("the redirect was followed: the token went where it pointed".to_string())
            } else if here > 1 {
                Err(format!(
                    "a 302 was sent again ({here} attempts): it is a stop"
                ))
            } else {
                Ok("a 302 ended the notification; its location got nothing")
            }
        }
        .await;
        match r {
            Ok(d) => pass(REDIRECTS, d),
            Err(e) => fail(REDIRECTS, e),
        }
    }

    /// N15: a server nobody trusts gets no request, and the failed handshake is retried.
    async fn tls(&self) -> Check {
        if self.v.expect.callback_origin.is_some() {
            return skip(
                TLS,
                "the callback origin is the operator's: the verifier cannot present a certificate \
                 there",
            );
        }
        let ip = self
            .v
            .expect
            .callback_listen
            .parse::<SocketAddr>()
            .map_or(std::net::IpAddr::from([127, 0, 0, 1]), |a| a.ip());
        let trap = match TlsTrap::start(ip).await {
            Ok(t) => t,
            Err(e) => return fail(TLS, format!("the TLS listener: {e}")),
        };
        let r = async {
            let (scope, id) = self.kept("notify-tls", "sleep 60").await?;
            let notify = JobNotify {
                url: format!(
                    "https://localhost:{}/_beyond/agent?session_id=conformance-tls&tenant=conformance",
                    trap.port
                ),
                token: self.token.clone(),
            };
            self.jobs()
                .keep_notify(&scope, &id, 1, "background", 0, &notify)
                .await
                .map_err(|e| format!("keep with notify: {e}"))?;
            let deadline = Instant::now() + PROOF_WAIT;
            while trap.seen().0 == 0 && Instant::now() < deadline {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            let first = Instant::now();
            while trap.seen().0 < 2 && first.elapsed() < Duration::from_secs(1) + SLACK * 2 {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            let (connections, requests) = trap.seen();
            self.done(&scope).await;
            if connections == 0 {
                Err("no attempt reached the https url".to_string())
            } else if requests > 0 {
                Err("a request was sent to a server whose certificate nobody trusts".to_string())
            } else if connections < 2 {
                Err("the failed handshake was not retried within its first wait (1 s)".to_string())
            } else {
                Ok("the handshake was refused, no request was sent, and it was retried")
            }
        }
        .await;
        match r {
            Ok(d) => pass(TLS, d),
            Err(e) => fail(TLS, e),
        }
    }

    /// N16–N18: `503` twice then `204`, on the schedule's first two waits; a `404` stops.
    async fn retry(&self) -> Check {
        let r = async {
            let key = "retry";
            self.rx.script(
                key,
                [
                    Answer::Status(503),
                    Answer::Status(503),
                    Answer::Status(204),
                ],
            );
            let (scope, id) = self.kept("notify-retry", "sleep 60").await?;
            self.keep(&scope, &id, 1, key).await?;
            let got = self
                .rx
                .wait(key, 3, PROOF_WAIT + Duration::from_secs(4))
                .await;
            if got.len() < 3 {
                self.done(&scope).await;
                return Err(format!(
                    "{} attempt(s) for 503, 503, 204: want 3",
                    got.len()
                ));
            }
            for (i, (lo, hi)) in [(0.5, 1.0), (1.0, 2.0)].into_iter().enumerate() {
                let gap = got[i + 1].at.duration_since(got[i].at);
                if gap + SLACK < Duration::from_secs_f64(lo)
                    || gap > Duration::from_secs_f64(hi) + SLACK
                {
                    self.done(&scope).await;
                    return Err(format!(
                        "the wait after attempt {} was {} ms: want {lo}–{hi} s",
                        i + 1,
                        gap.as_millis()
                    ));
                }
            }
            tokio::time::sleep(Duration::from_secs(2)).await;
            let more = self.rx.got(key).len();
            self.done(&scope).await;
            if more > 3 {
                return Err("attempts went on after a 204".to_string());
            }
            let key = "retry-stop";
            self.rx.script(key, [Answer::Status(404)]);
            let (scope, id) = self.kept("notify-stop", "sleep 60").await?;
            self.keep(&scope, &id, 1, key).await?;
            let first = self.rx.wait(key, 1, PROOF_WAIT).await;
            tokio::time::sleep(QUIET).await;
            let n = self.rx.got(key).len();
            self.done(&scope).await;
            if first.is_empty() {
                return Err("no proof arrived".to_string());
            }
            if n > 1 {
                return Err(format!("a 404 was retried ({n} attempts): it is a stop"));
            }
            Ok("503 retried after 0.5–1 s and 1–2 s, a 204 ended it, a 404 stopped at once")
        }
        .await;
        match r {
            Ok(d) => pass(RETRY, d),
            Err(e) => fail(RETRY, e),
        }
    }

    /// N19: the token is in no process's arguments or environment in the sandbox, and in no answer.
    async fn token_secret(&self) -> Check {
        let key = "token";
        let r = async {
            let (scope, id) = self.kept("notify-token", "printf out; sleep 60").await?;
            let r = async {
                self.keep(&scope, &id, 1, key).await?;
                if self.rx.wait(key, 1, PROOF_WAIT).await.is_empty() {
                    return Err("no proof arrived".to_string());
                }
                // The scan carries the token in two halves and joins them in an unexported
                // variable, so its own arguments never hold it whole.
                let (a, b) = self.token.split_at(self.token.len() / 2);
                let script = "t=\"$1$2\"; for f in /proc/[0-9]*/cmdline /proc/[0-9]*/environ; do \
                              c=$(tr '\\000' '\\n' <\"$f\" 2>/dev/null) || continue; \
                              case $c in *\"$t\"*) echo \"$f\";; esac; done";
                let found = self
                    .v
                    .runner
                    .post(
                        "sh",
                        &[
                            "-c".to_string(),
                            script.to_string(),
                            "sh".to_string(),
                            a.to_string(),
                            b.to_string(),
                        ],
                        None,
                        Duration::from_secs(30),
                        None,
                    )
                    .await
                    .map_err(|e| format!("scan: {e}"))?;
                if !found.stdout.trim().is_empty() {
                    return Err(format!("the token is in {}", found.stdout.trim().replace('\n', ", ")));
                }
                let jobs = self.jobs();
                let mut answers = Vec::new();
                for job in [
                    json!({ "op": "list", "scope": scope.name() }),
                    json!({ "op": "read", "scope": scope.name(), "id": id.as_str(), "offset": 0, "max": 4096 }),
                ] {
                    answers.push(jobs.call(job).await.map_err(|e| e.to_string())?.to_string());
                }
                // A refusal is an answer too.
                let bad = json!({
                    "op": "keep", "scope": scope.name(), "id": id.as_str(), "num": 1,
                    "kind": "background", "seen": 0,
                    "notify": { "url": "ftp://x/", "token": self.token },
                });
                if let Err(e) = jobs.call(bad).await {
                    answers.push(e.to_string());
                }
                if answers.iter().any(|a| a.contains(&self.token)) {
                    return Err("the token is in an answer (list, read, or an error)".to_string());
                }
                Ok("in no process's argv or environment, and in no list, read or error")
            }
            .await;
            self.done(&scope).await;
            r
        }
        .await;
        match r {
            Ok(d) => pass(TOKEN, d),
            Err(e) => fail(TOKEN, e),
        }
    }

    /// N1: a kept job survives the sandbox going idle — nothing is sent for the whole window.
    async fn idle(&self) -> Check {
        let idle = self.v.expect.notify_idle;
        let r = async {
            let (scope, id) = self.kept("notify-idle", "sleep 600").await?;
            tokio::time::sleep(idle).await;
            let st = self.status(&scope, &id).await;
            self.done(&scope).await;
            match st? {
                JobStatus::Running => Ok(format!(
                    "a kept job still ran after {} s with nothing sent to the sandbox",
                    idle.as_secs_f64()
                )),
                other => Err(format!(
                    "a kept job was {other:?} after {} s of idling: an endpoint whose sandbox ends \
                     idle jobs does not conform",
                    idle.as_secs_f64()
                )),
            }
        }
        .await;
        match r {
            Ok(d) => pass(IDLE, d),
            Err(e) => fail(IDLE, e),
        }
    }
}
