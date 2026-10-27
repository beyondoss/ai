//! The adversarial authentication matrix, against the real binary with a bearer set, over HTTP/1.1
//! and prior-knowledge HTTP/2 (h2c).
//!
//! Every case carries an exec request whose command would create a marker file. A refused case must
//! be answered with a non-2xx status (or a closed connection) AND leave no marker: refusing is
//! worthless if the command ran anyway. The cases that are *meant* to run (the controls, and the
//! few where HTTP itself says the credential is the same) must create it, which is what proves the
//! vehicle of every other case would have run its command had the request been accepted.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::net::SocketAddr;
use std::path::Path;
use std::time::Duration;

use bytes::Bytes;
use http_body_util::Full;
use hyper_util::rt::{TokioExecutor, TokioIo};
use serde_json::json;
use support::Execd;
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

const TOKEN: &str = "tok-Zq81xW5vN3pLr0aK";

/// One request, protocol-neutral: what a case sends.
#[derive(Clone)]
struct Req {
    method: &'static str,
    path: String,
    headers: Vec<(String, Vec<u8>)>,
    body: Vec<u8>,
}

impl Req {
    fn post(marker: &Path) -> Self {
        Self {
            method: "POST",
            path: "/exec".into(),
            headers: Vec::new(),
            body: exec_body(marker),
        }
    }

    fn with(mut self, name: &str, value: impl AsRef<[u8]>) -> Self {
        self.headers.push((name.into(), value.as_ref().to_vec()));
        self
    }

    fn auth(self) -> Self {
        self.with("Authorization", format!("Bearer {TOKEN}"))
    }

    fn at(mut self, path: &str) -> Self {
        self.path = path.into();
        self
    }

    fn method(mut self, m: &'static str) -> Self {
        self.method = m;
        self
    }
}

/// An exec request that creates `marker`.
fn exec_body(marker: &Path) -> Vec<u8> {
    json!({ "command": "sh", "args": ["-c", format!("touch '{}'", marker.display())],
            "timeout_ms": 20_000 })
    .to_string()
    .into_bytes()
}

/// What came back: a status, or `None` when the connection was closed without an answer.
type Status = Option<u16>;

async fn over_h1(addr: SocketAddr, raw: Vec<u8>) -> Status {
    let mut sock = tokio::net::TcpStream::connect(addr).await.unwrap();
    // A refusal may close the socket while the rest is still being written: that is an answer.
    let _ = sock.write_all(&raw).await;
    let mut out = Vec::new();
    let _ = tokio::time::timeout(Duration::from_secs(3), sock.read_to_end(&mut out)).await;
    let line = String::from_utf8_lossy(&out);
    line.split(' ').nth(1).and_then(|s| s.parse().ok())
}

/// The wire form of `req` over HTTP/1.1.
fn h1_bytes(req: &Req, content_length: Option<usize>) -> Vec<u8> {
    let mut out = format!(
        "{} {} HTTP/1.1\r\nHost: execd\r\nConnection: close\r\n",
        req.method, req.path
    )
    .into_bytes();
    for (n, v) in &req.headers {
        out.extend_from_slice(n.as_bytes());
        out.extend_from_slice(b": ");
        out.extend_from_slice(v);
        out.extend_from_slice(b"\r\n");
    }
    if !req.body.is_empty() || req.method == "POST" {
        out.extend_from_slice(
            format!(
                "Content-Length: {}\r\n",
                content_length.unwrap_or(req.body.len())
            )
            .as_bytes(),
        );
    }
    out.extend_from_slice(b"\r\n");
    out.extend_from_slice(&req.body);
    out
}

/// `None`: this request cannot be expressed over HTTP/2 (the client library refuses to build it).
async fn over_h2(addr: SocketAddr, req: &Req) -> Option<Status> {
    let uri: hyper::Uri = format!("http://{addr}{}", req.path).parse().ok()?;
    let mut b = hyper::Request::builder().method(req.method).uri(uri);
    for (n, v) in &req.headers {
        let name = hyper::header::HeaderName::from_bytes(n.as_bytes()).ok()?;
        let value = hyper::header::HeaderValue::from_bytes(v).ok()?;
        b = b.header(name, value);
    }
    let request = b.body(Full::new(Bytes::from(req.body.clone()))).ok()?;
    let sock = tokio::net::TcpStream::connect(addr).await.unwrap();
    let (mut send, conn) =
        hyper::client::conn::http2::handshake(TokioExecutor::new(), TokioIo::new(sock))
            .await
            .ok()?;
    tokio::spawn(conn);
    let sent = tokio::time::timeout(Duration::from_secs(10), send.send_request(request)).await;
    Some(match sent {
        Ok(Ok(resp)) => Some(resp.status().as_u16()),
        _ => None,
    })
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Proto {
    H1,
    H2,
}

/// Whether the command of a case ran, which settles by the time its answer is in (a v1.1 answer
/// follows the command's exit); a short wait covers a connection closed first.
async fn ran(marker: &Path) -> bool {
    ran_within(marker, 100).await
}

async fn ran_within(marker: &Path, ms: u64) -> bool {
    for _ in 0..=(ms / 10) {
        if marker.exists() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    false
}

/// Which way the server under test authenticates.
#[derive(Clone, Copy)]
enum Auth {
    /// The real binary with `EXECD_BEARER`.
    Bearer,
    /// The library, embedded, with an `Options::authorizer` that accepts exactly the one
    /// `Authorization: Bearer <token>` header and nothing else.
    Authorizer,
}

struct Matrix {
    dir: std::path::PathBuf,
    /// Keeps the server (and its directory) alive until the matrix ends.
    _server: Box<dyn std::any::Any>,
    addr: SocketAddr,
    failures: Vec<String>,
    cases: usize,
    ran_by_design: usize,
}

impl Matrix {
    async fn new(auth: Auth) -> Self {
        let (addr, dir, server): (SocketAddr, std::path::PathBuf, Box<dyn std::any::Any>) =
            match auth {
                Auth::Bearer => {
                    let execd = Execd::start(&[], &[("EXECD_BEARER", TOKEN)]);
                    let dir = execd.dir.path().to_path_buf();
                    (SocketAddr::V4(execd.addr), dir, Box::new(execd))
                }
                Auth::Authorizer => {
                    let dir = tempfile::tempdir().unwrap();
                    let mut o = execd::Options::new(dir.path());
                    let want = format!("Bearer {TOKEN}");
                    o.authorizer = Some(std::sync::Arc::new(move |h: &execd::HeaderMap| {
                        let mut all = h.get_all("authorization").iter();
                        matches!((all.next(), all.next()), (Some(v), None) if v.as_bytes() == want.as_bytes())
                    }));
                    let listener = beyond_test_support::ports::tokio_listener().await;
                    let server = execd::serve(listener, o).unwrap();
                    (
                        server.local_addr(),
                        dir.path().to_path_buf(),
                        Box::new((server, dir)),
                    )
                }
            };
        Self {
            dir,
            _server: server,
            addr,
            failures: Vec::new(),
            cases: 0,
            ran_by_design: 0,
        }
    }

    fn marker(&self, n: usize) -> std::path::PathBuf {
        self.dir.join(format!("marker-{n}"))
    }

    /// Send `build(marker)` over `proto`. `runs`: the case is one HTTP itself says is authorized, and
    /// must create the marker; otherwise it must be refused and must not.
    async fn case(&mut self, label: &str, proto: Proto, runs: bool, build: impl Fn(&Path) -> Req) {
        self.cases += 1;
        let marker = self.marker(self.cases);
        let req = build(&marker);
        let status = match proto {
            Proto::H1 => over_h1(self.addr, h1_bytes(&req, None)).await,
            Proto::H2 => match over_h2(self.addr, &req).await {
                Some(s) => s,
                None => {
                    // Not expressible over h2: nothing to send, so nothing can have run.
                    self.cases -= 1;
                    return;
                }
            },
        };
        let did_run = ran_within(&marker, if status.is_some() { 10 } else { 150 }).await;
        let ok_status = status.is_some_and(|s| (200..300).contains(&s));
        let verdict = match (runs, did_run, ok_status) {
            (true, true, true) => {
                self.ran_by_design += 1;
                None
            }
            (false, false, false) => None,
            _ => Some(format!(
                "{proto:?} {label}: status {status:?}, command ran: {did_run} (wanted it to {})",
                if runs {
                    "run"
                } else {
                    "be refused and not run"
                }
            )),
        };
        if let Some(v) = verdict {
            self.failures.push(v);
        }
    }

    /// Both protocols.
    async fn both(&mut self, label: &str, runs: bool, build: impl Fn(&Path) -> Req + Clone) {
        self.case(label, Proto::H1, runs, build.clone()).await;
        self.case(label, Proto::H2, runs, build).await;
    }

    /// A raw HTTP/1.1 exchange: `bytes` as written, then `later` unauthenticated markers must stay
    /// absent whatever hyper made of the framing.
    async fn raw(
        &mut self,
        label: &str,
        bytes: Vec<u8>,
        smuggled: &[std::path::PathBuf],
        first_runs: Option<bool>,
        first: &Path,
    ) {
        self.cases += 1;
        let status = over_h1(self.addr, bytes).await;
        let mut bad = Vec::new();
        for m in smuggled {
            if ran(m).await {
                bad.push(format!("the smuggled command ran ({})", m.display()));
            }
        }
        if let Some(want) = first_runs {
            let did = ran(first).await;
            if did != want {
                bad.push(format!(
                    "the first request {} (wanted {want})",
                    if did { "ran" } else { "did not run" }
                ));
            }
        }
        if !bad.is_empty() {
            self.failures
                .push(format!("H1 {label}: status {status:?}: {}", bad.join("; ")));
        }
    }

    fn finish(self) {
        println!("{} cases, {} ran by design", self.cases, self.ran_by_design);
        assert!(self.failures.is_empty(), "\n{}\n", self.failures.join("\n"));
        drop(self._server);
    }
}

#[tokio::test]
async fn nothing_runs_without_the_bearer_over_http1_and_h2c() {
    paths_methods_and_headers(Auth::Bearer).await;
}

#[tokio::test]
async fn nothing_runs_without_the_authorizer_saying_so_over_http1_and_h2c() {
    paths_methods_and_headers(Auth::Authorizer).await;
}

async fn paths_methods_and_headers(auth: Auth) {
    let mut m = Matrix::new(auth).await;

    // Controls: the vehicle works. A valid bearer on the exec path runs the command.
    m.both("control: valid bearer", true, |p| Req::post(p).auth())
        .await;

    // --- paths: the bearer decides, not the path -------------------------------------------------
    for path in [
        "/livez",
        "/livez/",
        "//livez",
        "/livez/..",
        "/./livez",
        "/%6civez",
        "/livez?x",
        "/LIVEZ",
        "/livez%00",
        "/livez;x",
        "/exec/../livez",
        "/healthz",
        "/healthz/",
        "//healthz",
        "/healthz/..",
        "/./healthz",
        "/%68ealthz",
        "/healthz?x",
        "/HEALTHZ",
        "/healthz%00",
        "/healthz;x",
        "/exec/../healthz",
        "/",
        "",
    ] {
        let path = path.to_string();
        m.both(&format!("path {path:?}"), false, move |p| {
            Req::post(p).at(&path)
        })
        .await;
    }
    // The fragment is not sent by any client; a raw one is part of the path and is just another path.
    m.case("path /livez#", Proto::H1, false, |p| {
        Req::post(p).at("/livez#")
    })
    .await;
    m.case("path /healthz#", Proto::H1, false, |p| {
        Req::post(p).at("/healthz#")
    })
    .await;
    // Absolute-form request target.
    m.case("absolute-form http://host/livez", Proto::H1, false, |p| {
        Req::post(p).at("http://execd/livez")
    })
    .await;
    m.case("absolute-form http://host/exec", Proto::H1, false, |p| {
        Req::post(p).at("http://execd/exec")
    })
    .await;
    m.case("absolute-form over h2", Proto::H2, false, |p| {
        Req::post(p).at("/livez")
    })
    .await;
    m.case("absolute-form http://host/healthz", Proto::H1, false, |p| {
        Req::post(p).at("http://execd/healthz")
    })
    .await;
    m.case("absolute-form over h2 /healthz", Proto::H2, false, |p| {
        Req::post(p).at("/healthz")
    })
    .await;
    // A valid bearer on `/livez` is an exec request like any other (the path never mattered).
    m.both("POST /livez with the bearer", true, |p| {
        Req::post(p).auth().at("/livez")
    })
    .await;
    m.both("POST /healthz with the bearer", true, |p| {
        Req::post(p).auth().at("/healthz")
    })
    .await;

    // --- methods: only POST execs ----------------------------------------------------------------
    for method in ["GET", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD", "TRACE"] {
        m.both(&format!("{method} with a body"), false, move |p| {
            Req::post(p).method(method)
        })
        .await;
        m.both(
            &format!("{method} with a body and the bearer"),
            false,
            move |p| Req::post(p).method(method).auth(),
        )
        .await;
        if method != "GET" && method != "HEAD" {
            m.both(&format!("{method} /livez with a body"), false, move |p| {
                Req::post(p).method(method).at("/livez")
            })
            .await;
            m.both(&format!("{method} /healthz with a body"), false, move |p| {
                Req::post(p).method(method).at("/healthz")
            })
            .await;
        }
    }
    m.both("POST /livez carrying an exec body", false, |p| {
        Req::post(p).at("/livez")
    })
    .await;
    m.both("POST /healthz carrying an exec body", false, |p| {
        Req::post(p).at("/healthz")
    })
    .await;

    // --- Authorization ---------------------------------------------------------------------------
    m.both("missing", false, Req::post).await;
    m.both("empty", false, |p| Req::post(p).with("Authorization", ""))
        .await;
    m.both("Bearer with no token", false, |p| {
        Req::post(p).with("Authorization", "Bearer")
    })
    .await;
    m.both("Bearer and a space", false, |p| {
        Req::post(p).with("Authorization", "Bearer ")
    })
    .await;
    m.both("Bearer and the empty token twice", false, |p| {
        Req::post(p).with("Authorization", "Bearer  ")
    })
    .await;
    m.both("lowercase scheme", false, |p| {
        Req::post(p).with("Authorization", format!("bearer {TOKEN}"))
    })
    .await;
    m.both("uppercase scheme", false, |p| {
        Req::post(p).with("Authorization", format!("BEARER {TOKEN}"))
    })
    .await;
    m.both("two spaces", false, |p| {
        Req::post(p).with("Authorization", format!("Bearer  {TOKEN}"))
    })
    .await;
    m.both("a tab", false, |p| {
        Req::post(p).with("Authorization", format!("Bearer\t{TOKEN}"))
    })
    .await;
    m.both("no scheme", false, |p| {
        Req::post(p).with("Authorization", TOKEN)
    })
    .await;
    m.both("Basic", false, |p| {
        Req::post(p).with("Authorization", format!("Basic {TOKEN}"))
    })
    .await;
    m.both("Token scheme", false, |p| {
        Req::post(p).with("Authorization", format!("Token {TOKEN}"))
    })
    .await;
    m.both("a prefix of the token", false, |p| {
        Req::post(p).with(
            "Authorization",
            format!("Bearer {}", &TOKEN[..TOKEN.len() - 1]),
        )
    })
    .await;
    m.both("a suffix of the token", false, |p| {
        Req::post(p).with("Authorization", format!("Bearer {}", &TOKEN[1..]))
    })
    .await;
    m.both("the token and one more byte", false, |p| {
        Req::post(p).with("Authorization", format!("Bearer {TOKEN}x"))
    })
    .await;
    m.both("one byte and the token", false, |p| {
        Req::post(p).with("Authorization", format!("Bearer x{TOKEN}"))
    })
    .await;
    m.both("the token twice", false, |p| {
        Req::post(p).with("Authorization", format!("Bearer {TOKEN}{TOKEN}"))
    })
    .await;
    m.both("the token and a comma", false, |p| {
        Req::post(p).with("Authorization", format!("Bearer {TOKEN},"))
    })
    .await;
    m.both("the token in quotes", false, |p| {
        Req::post(p).with("Authorization", format!("Bearer \"{TOKEN}\""))
    })
    .await;
    m.both("the token and a CR", false, |p| {
        Req::post(p).with("Authorization", format!("Bearer {TOKEN}\r"))
    })
    .await;
    m.both("the token and a NUL", false, |p| {
        Req::post(p).with(
            "Authorization",
            [format!("Bearer {TOKEN}").as_bytes(), b"\0"].concat(),
        )
    })
    .await;
    m.both("a NUL before the token", false, |p| {
        Req::post(p).with(
            "Authorization",
            [b"Bearer \0".as_slice(), TOKEN.as_bytes()].concat(),
        )
    })
    .await;
    m.both("the token and a high byte", false, |p| {
        Req::post(p).with(
            "Authorization",
            [format!("Bearer {TOKEN}").as_bytes(), b"\xff"].concat(),
        )
    })
    .await;
    m.both("a different token of the same length", false, |p| {
        Req::post(p).with(
            "Authorization",
            format!("Bearer {}", TOKEN.replace('t', "T")),
        )
    })
    .await;
    // Trailing whitespace. HTTP/1.1 strips optional whitespace around a field value before anything
    // sees it, so on the wire `Bearer <token> ` *is* `Bearer <token>`: the same credential, which
    // runs. HTTP/2 strips nothing, so there the extra byte makes it a different one.
    m.case("the token and a trailing space", Proto::H1, true, |p| {
        Req::post(p).with("Authorization", format!("Bearer {TOKEN} "))
    })
    .await;
    m.case("the token and a trailing tab", Proto::H1, true, |p| {
        Req::post(p).with("Authorization", format!("Bearer {TOKEN}\t"))
    })
    .await;
    m.case("the token and a trailing space", Proto::H2, false, |p| {
        Req::post(p).with("Authorization", format!("Bearer {TOKEN} "))
    })
    .await;
    m.case("the token and a trailing tab", Proto::H2, false, |p| {
        Req::post(p).with("Authorization", format!("Bearer {TOKEN}\t"))
    })
    .await;
    // Two headers: ambiguous, so refused whichever is first.
    m.both("valid then invalid", false, |p| {
        Req::post(p).auth().with("Authorization", "Bearer wrong")
    })
    .await;
    m.both("invalid then valid", false, |p| {
        Req::post(p).with("Authorization", "Bearer wrong").auth()
    })
    .await;
    m.both("valid twice", false, |p| Req::post(p).auth().auth())
        .await;
    m.both("valid then empty", false, |p| {
        Req::post(p).auth().with("Authorization", "")
    })
    .await;
    // Other headers that look like credentials.
    m.both("Proxy-Authorization carrying the token", false, |p| {
        Req::post(p).with("Proxy-Authorization", format!("Bearer {TOKEN}"))
    })
    .await;
    m.both(
        "Proxy-Authorization and a wrong Authorization",
        false,
        |p| {
            Req::post(p)
                .with("Proxy-Authorization", format!("Bearer {TOKEN}"))
                .with("Authorization", "Bearer wrong")
        },
    )
    .await;
    m.both("X-Authorization", false, |p| {
        Req::post(p).with("X-Authorization", format!("Bearer {TOKEN}"))
    })
    .await;
    m.both("Cookie", false, |p| {
        Req::post(p).with("Cookie", format!("Authorization=Bearer {TOKEN}"))
    })
    .await;
    m.both("the token as a query", false, |p| {
        Req::post(p).at(&format!("/exec?token={TOKEN}&access_token={TOKEN}"))
    })
    .await;
    m.both("X-Forwarded-For loopback", false, |p| {
        Req::post(p)
            .with("X-Forwarded-For", "127.0.0.1")
            .with("X-Real-IP", "127.0.0.1")
    })
    .await;
    // Header names are case-insensitive: a valid bearer in another case runs.
    m.case("lowercase header name", Proto::H1, true, |p| {
        Req::post(p).with("authorization", format!("Bearer {TOKEN}"))
    })
    .await;
    m.case("uppercase header name", Proto::H1, true, |p| {
        Req::post(p).with("AUTHORIZATION", format!("Bearer {TOKEN}"))
    })
    .await;

    m.finish();
}

/// Bodies and framing, over HTTP/1.1 where the framing is the client's to get wrong.
#[tokio::test]
async fn framing_cannot_get_a_command_past_the_bearer_or_smuggle_one() {
    framing(Auth::Bearer).await;
}

#[tokio::test]
async fn framing_cannot_get_a_command_past_the_authorizer_or_smuggle_one() {
    framing(Auth::Authorizer).await;
}

async fn framing(mode: Auth) {
    let mut m = Matrix::new(mode).await;
    let good = |n: usize, m: &Matrix| (m.marker(n), m.marker(n + 1));
    let auth = format!("Bearer {TOKEN}");

    // A chunked body without the bearer, and with it (the control: chunked works).
    let chunked = |marker: &Path, with_auth: bool| {
        let body = exec_body(marker);
        let mut v = b"POST /exec HTTP/1.1\r\nHost: execd\r\nConnection: close\r\nTransfer-Encoding: chunked\r\n".to_vec();
        if with_auth {
            v.extend_from_slice(format!("Authorization: {auth}\r\n").as_bytes());
        }
        v.extend_from_slice(format!("\r\n{:x}\r\n", body.len()).as_bytes());
        v.extend_from_slice(&body);
        v.extend_from_slice(b"\r\n0\r\n\r\n");
        v
    };
    let (a, _) = good(100, &m);
    m.raw(
        "chunked, no bearer",
        chunked(&a, false),
        &[],
        Some(false),
        &a,
    )
    .await;
    let (a, _) = good(102, &m);
    m.raw(
        "chunked, valid bearer (control)",
        chunked(&a, true),
        &[],
        Some(true),
        &a,
    )
    .await;

    // A Content-Length that disagrees with the body, no bearer: the front request is refused whatever
    // its body is taken to be, and what follows it is a request of its own, also refused.
    for (label, cl_delta) in [
        ("longer", 10isize),
        ("shorter", -10),
        ("zero", i64::MIN as isize),
    ] {
        let (a, b) = good(110 + cl_delta.unsigned_abs() % 7, &m);
        let mut body = exec_body(&a);
        body.extend_from_slice(b"\r\n");
        let tail = h1_bytes(&Req::post(&b), None);
        body.extend_from_slice(&tail);
        let cl = if cl_delta == i64::MIN as isize {
            0
        } else {
            usize::try_from(isize::try_from(exec_body(&a).len()).unwrap() + cl_delta).unwrap()
        };
        let req = Req {
            method: "POST",
            path: "/exec".into(),
            headers: vec![],
            body,
        };
        m.raw(
            &format!("Content-Length {label}, no bearer, a request behind it"),
            h1_bytes(&req, Some(cl)),
            &[a.clone(), b.clone()],
            Some(false),
            &a,
        )
        .await;
    }
    // With the bearer on the front request only: the request behind it has none.
    for (label, delta) in [("exact", 0isize), ("shorter", -5)] {
        let a = m.marker(130 + usize::from(delta != 0));
        let b = m.marker(140 + usize::from(delta != 0));
        let front = exec_body(&a);
        let mut body = front.clone();
        body.extend_from_slice(&h1_bytes(&Req::post(&b), None));
        let cl = usize::try_from(isize::try_from(front.len()).unwrap() + delta).unwrap();
        let req = Req {
            method: "POST",
            path: "/exec".into(),
            headers: vec![("Authorization".into(), auth.clone().into_bytes())],
            body,
        };
        m.raw(
            &format!("Content-Length {label}, bearer, an unauthenticated request behind it"),
            h1_bytes(&req, Some(cl)),
            &[b],
            None,
            &a,
        )
        .await;
    }
    // A Content-Length longer than the body that follows: the body never completes, nothing runs.
    let a = m.marker(150);
    let req = Req {
        method: "POST",
        path: "/exec".into(),
        headers: vec![("Authorization".into(), auth.clone().into_bytes())],
        body: exec_body(&a),
    };
    m.raw(
        "Content-Length longer than the body, bearer",
        h1_bytes(&req, Some(req.body.len() + 100)),
        &[],
        Some(false),
        &a,
    )
    .await;
    // Two different Content-Lengths.
    let a = m.marker(151);
    let mut raw = h1_bytes(
        &Req {
            method: "POST",
            path: "/exec".into(),
            headers: vec![
                ("Authorization".into(), auth.clone().into_bytes()),
                ("Content-Length".into(), b"3".to_vec()),
            ],
            body: exec_body(&a),
        },
        None,
    );
    m.raw(
        "two Content-Lengths that differ, bearer",
        std::mem::take(&mut raw),
        &[],
        Some(false),
        &a,
    )
    .await;

    // Request smuggling: Content-Length and Transfer-Encoding together, the second request behind the
    // first being an unauthenticated exec. Whatever hyper takes the framing to be, the second request
    // is a request of its own and is refused.
    for (label, te) in [
        ("CL and TE chunked", "chunked"),
        ("TE chunked, identity", "chunked, identity"),
        ("TE identity, chunked", "identity, chunked"),
        ("TE Chunked", "Chunked"),
        ("TE xchunked", "xchunked"),
        ("TE chunked chunked", "chunked, chunked"),
        ("TE with a trailing comma", "chunked,"),
    ] {
        let a = m.marker(160 + label.len());
        let b = m.marker(260 + label.len());
        let front = exec_body(&a);
        let chunk = format!("{:x}\r\n", front.len()).into_bytes();
        let mut body = chunk;
        body.extend_from_slice(&front);
        body.extend_from_slice(b"\r\n0\r\n\r\n");
        let cl = body.len();
        body.extend_from_slice(&h1_bytes(&Req::post(&b), None));
        let req = Req {
            method: "POST",
            path: "/exec".into(),
            headers: vec![
                ("Authorization".into(), auth.clone().into_bytes()),
                ("Transfer-Encoding".into(), te.as_bytes().to_vec()),
            ],
            body,
        };
        m.raw(
            &format!("{label}, bearer"),
            h1_bytes(&req, Some(cl)),
            std::slice::from_ref(&b),
            None,
            &a,
        )
        .await;
        // The same without the bearer: neither runs.
        let a2 = m.marker(360 + label.len());
        let b2 = m.marker(460 + label.len());
        let mut req2 = req.clone();
        req2.headers.remove(0);
        let front = exec_body(&a2);
        let mut body = format!("{:x}\r\n", front.len()).into_bytes();
        body.extend_from_slice(&front);
        body.extend_from_slice(b"\r\n0\r\n\r\n");
        let cl = body.len();
        body.extend_from_slice(&h1_bytes(&Req::post(&b2), None));
        req2.body = body;
        m.raw(
            &format!("{label}, no bearer"),
            h1_bytes(&req2, Some(cl)),
            &[a2.clone(), b2],
            Some(false),
            &a2,
        )
        .await;
    }
    // A header with a space before its colon, an obsolete line fold, a bare LF: framing tricks that
    // hyper must refuse or read the way it reads any request.
    let a = m.marker(600);
    let b = m.marker(601);
    let body = exec_body(&a);
    let mut raw = format!("POST /exec HTTP/1.1\r\nHost: execd\r\nAuthorization: {auth}\r\nTransfer-Encoding : chunked\r\nContent-Length: {}\r\n\r\n", body.len()).into_bytes();
    raw.extend_from_slice(&body);
    raw.extend_from_slice(&h1_bytes(&Req::post(&b), None));
    m.raw(
        "space before the colon of Transfer-Encoding",
        raw,
        &[b],
        None,
        &a,
    )
    .await;
    let a = m.marker(602);
    let b = m.marker(603);
    let body = exec_body(&a);
    let mut raw = format!("POST /exec HTTP/1.1\r\nHost: execd\r\nAuthorization: {auth}\r\nX-Fold: a\r\n  b\r\nContent-Length: {}\r\n\r\n", body.len()).into_bytes();
    raw.extend_from_slice(&body);
    raw.extend_from_slice(&h1_bytes(&Req::post(&b), None));
    m.raw("an obsolete line fold", raw, &[b], None, &a).await;
    let a = m.marker(604);
    let body = exec_body(&a);
    let mut raw = format!(
        "POST /exec HTTP/1.1\nHost: execd\nContent-Length: {}\n\n",
        body.len()
    )
    .into_bytes();
    raw.extend_from_slice(&body);
    m.raw(
        "bare LF line endings, no bearer",
        raw,
        std::slice::from_ref(&a),
        Some(false),
        &a,
    )
    .await;
    // HTTP/1.0 and a pipelined unauthenticated request behind an authorized one.
    let a = m.marker(605);
    let b = m.marker(606);
    let mut raw = h1_bytes(&Req::post(&a).auth(), None);
    raw.extend_from_slice(&h1_bytes(&Req::post(&b), None));
    m.raw(
        "pipelined: authorized then unauthenticated",
        raw,
        &[b],
        Some(true),
        &a,
    )
    .await;

    // A header block over any limit, with and without the bearer: refused, nothing runs.
    for (label, size, with_auth) in [
        ("1 MiB", 1 << 20, true),
        ("1 MiB", 1 << 20, false),
        ("64 MiB", 64 << 20, true),
    ] {
        let a = m.marker(700 + size / (1 << 20));
        let mut req = Req::post(&a);
        if with_auth {
            req = req.auth();
        }
        req = req.with("X-Padding", vec![b'a'; size]);
        m.raw(
            &format!("a {label} header block, bearer: {with_auth}"),
            h1_bytes(&req, None),
            &[],
            Some(false),
            &a,
        )
        .await;
    }
    // Thousands of small headers.
    let a = m.marker(720);
    let mut req = Req::post(&a).auth();
    for i in 0..20_000 {
        req = req.with(&format!("X-H{i}"), "v");
    }
    m.raw(
        "20,000 headers, bearer",
        h1_bytes(&req, None),
        &[],
        Some(false),
        &a,
    )
    .await;
    // An absurdly long request target.
    let a = m.marker(721);
    let req = Req::post(&a)
        .auth()
        .at(&format!("/{}", "a".repeat(1 << 20)));
    m.raw(
        "a 1 MiB request target, bearer",
        h1_bytes(&req, None),
        &[],
        Some(false),
        &a,
    )
    .await;

    m.finish();
}

/// `/livez` is the one unauthenticated answer: it reads, says the process is up, and nothing else.
#[tokio::test]
async fn livez_answers_a_read_and_reveals_nothing() {
    let execd = Execd::start(&[], &[("EXECD_BEARER", TOKEN)]);
    let addr = SocketAddr::V4(execd.addr);
    for health in ["/livez", "/healthz"] {
        let get = support::http::send(addr, "GET", health, &[], b"").await;
        assert_eq!(get.status, 200);
        assert_eq!(get.body, br#"{"ok":true}"#);
        let names: Vec<String> = get.headers.iter().map(|(n, _)| n.to_lowercase()).collect();
        for n in &names {
            assert!(
                ["content-type", "content-length", "date", "connection"].contains(&n.as_str()),
                "unexpected header {n}: {names:?}"
            );
        }
        let head = support::http::send(addr, "HEAD", health, &[], b"").await;
        assert_eq!(head.status, 200);
        // A read of `/livez` that carries an exec body is still only a read: the body is never looked at.
        let marker = execd.dir.path().join("health-marker");
        let with_body = support::http::send(addr, "GET", health, &[], &exec_body(&marker)).await;
        assert_eq!(
            (with_body.status, &with_body.body[..]),
            (200, &br#"{"ok":true}"#[..])
        );
        assert!(!ran(&marker).await);
    }
    // Nothing else is open: a read of any other path is refused, with or without the bearer.
    for path in [
        "/", "/exec", "/metrics", "/debug", "/status", "/version", "/health", "/jobs",
    ] {
        let a = support::http::send(addr, "GET", path, &[], b"").await;
        assert_eq!(a.status, 405, "{path}");
        assert!(!String::from_utf8_lossy(&a.body).contains(TOKEN));
    }
    // A refusal says nothing about why, nor about the token.
    let bad = support::http::send(
        addr,
        "POST",
        "/exec",
        &[("Authorization", "Bearer nope")],
        b"{}",
    )
    .await;
    assert_eq!(bad.status, 401);
    assert_eq!(bad.body, br#"{"error":"unauthorized"}"#);
    let none = support::http::send(addr, "POST", "/exec", &[], b"{}").await;
    assert_eq!((none.status, none.body.clone()), (401, bad.body.clone()));
}
