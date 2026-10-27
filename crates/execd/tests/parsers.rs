//! Seeded random request bodies against the real binary: byte-level mutations of valid requests,
//! hostile JSON shapes and plain random bytes. The parsers must never panic, execd must keep
//! answering, and a body the parsers refuse must get a `4xx`, never a `5xx` and never a closed
//! connection with no answer.
//!
//! A body the parsers *accept* would run its command, so the test only sends one whose command is
//! the harmless `true` (a mutated request that became some other command is counted and skipped).
//! The accept/refuse oracle is the library's own parser entry point, the one the fuzz targets in
//! `fuzz/` drive without a server.
//!
//! Deterministic by seed. `EXECD_PARSE_CASES` sets the number of bodies, `EXECD_PARSE_SECS` runs
//! fresh seeds for that long (the deep workflow), `EXECD_PARSE_SEED` replays one.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::net::SocketAddr;
use std::time::{Duration, Instant};

use bytes::Bytes;
use http_body_util::{BodyExt as _, Full};
use hyper_util::rt::TokioIo;
use support::Execd;

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

/// Valid requests, each harmless if sent as written.
const TEMPLATES: &[&str] = &[
    r#"{"command":"true","args":[],"timeout_ms":5000}"#,
    r#"{"command":"true","args":["a","b"],"cwd":".","stdin":"aGVsbG8=","stream":false}"#,
    r#"{"command":"true","job":{"op":"probe"}}"#,
    r#"{"command":"true","job":{"op":"list","scope":"s"}}"#,
    r#"{"command":"true","job":{"op":"read","scope":"s","id":"a1","offset":0,"max":4096}}"#,
    r#"{"command":"true","job":{"op":"kill","scope":"s","ids":["a1","b2"],"remove":true,"grace_ms":0}}"#,
    r#"{"command":"true","job":{"op":"remove","scope":"s","ids":["a1"]}}"#,
    r#"{"command":"true","job":{"op":"wipe","scope":"s","grace_ms":0}}"#,
    r#"{"command":"true","job":{"op":"seen","scope":"s","id":"a1","offset":9}}"#,
    r#"{"command":"true","job":{"op":"keep","scope":"s","id":"a1","num":3,"kind":"background","seen":1,"notify":{"url":"http://127.0.0.1:1/x","token":"tk-1"}}}"#,
    r#"{"command":"true","stream":true,"job":{"op":"attach","scope":"s","id":"zz","offset":0}}"#,
];

/// Strings and numbers a parser may trip on.
const NASTY: &[&str] = &[
    "18446744073709551615",
    "18446744073709551616",
    "-1",
    "1.5",
    "1e999",
    "-0",
    "null",
    "true",
    "[]",
    "{}",
    "\"\"",
    "\"\\u0000\"",
    "\"\\ud800\"",
    "\"\\udc00\\ud800\"",
    "\"\u{1F600}\"",
    "\"%s%n%x\"",
    "\"../../etc/passwd\"",
    "[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[",
];

fn mutate(r: &mut Rng) -> Vec<u8> {
    let mut b = TEMPLATES[r.below(TEMPLATES.len())].as_bytes().to_vec();
    for _ in 0..1 + r.below(4) {
        if b.is_empty() {
            break;
        }
        match r.below(9) {
            0 => {
                let i = r.below(b.len());
                b[i] ^= 1 << r.below(8);
            }
            1 => {
                let i = r.below(b.len());
                b[i] = r.next() as u8;
            }
            2 => b.truncate(r.below(b.len())),
            3 => {
                let (i, n) = (r.below(b.len()), r.below(8));
                b.drain(i..(i + n).min(b.len()));
            }
            4 => {
                let (i, n) = (r.below(b.len()), 1 + r.below(16));
                let piece: Vec<u8> = b[i..(i + n).min(b.len())].to_vec();
                let at = r.below(b.len());
                b.splice(at..at, piece);
            }
            5 => {
                // Replace one value with something nasty: from a `:` to the next `,` or `}`.
                if let Some(colon) = b.iter().skip(r.below(b.len())).position(|&c| c == b':') {
                    let _ = colon;
                }
                let s = String::from_utf8_lossy(&b).into_owned();
                let colons: Vec<usize> = s.match_indices(':').map(|(i, _)| i).collect();
                if !colons.is_empty() {
                    let at = colons[r.below(colons.len())] + 1;
                    let end = s[at..].find([',', '}']).map_or(s.len(), |e| at + e);
                    let mut out = s.as_bytes()[..at].to_vec();
                    out.extend_from_slice(NASTY[r.below(NASTY.len())].as_bytes());
                    out.extend_from_slice(&s.as_bytes()[end..]);
                    b = out;
                }
            }
            6 => {
                let at = r.below(b.len());
                let junk: Vec<u8> = (0..1 + r.below(6)).map(|_| r.next() as u8).collect();
                b.splice(at..at, junk);
            }
            7 => {
                // Bytes that are not UTF-8.
                let at = r.below(b.len());
                b.splice(at..at, [0xff, 0xfe, 0xc0, 0x80]);
            }
            _ => b.extend_from_slice(b" "),
        }
    }
    b
}

fn random_bytes(r: &mut Rng) -> Vec<u8> {
    match r.below(6) {
        0 => Vec::new(),
        1 => b"null".to_vec(),
        2 => b"[]".to_vec(),
        3 => vec![b'['; 1 + r.below(100_000)],
        4 => (0..r.below(64)).map(|_| r.next() as u8).collect(),
        _ => (0..r.below(4096)).map(|_| r.next() as u8).collect(),
    }
}

/// One persistent connection, reopened when the server closes it.
struct Client {
    addr: SocketAddr,
    bearer: String,
    send: Option<hyper::client::conn::http1::SendRequest<Full<Bytes>>>,
}

impl Client {
    async fn post(&mut self, body: Vec<u8>) -> Option<u16> {
        for _ in 0..2 {
            if self.send.as_mut().is_none_or(|s| s.is_closed()) {
                let sock = tokio::net::TcpStream::connect(self.addr).await.ok()?;
                let (send, conn) = hyper::client::conn::http1::handshake(TokioIo::new(sock))
                    .await
                    .ok()?;
                tokio::spawn(async move {
                    let _ = conn.await;
                });
                self.send = Some(send);
            }
            let req = hyper::Request::builder()
                .method("POST")
                .uri("/exec")
                .header("Host", "execd")
                .header("Authorization", format!("Bearer {}", self.bearer))
                .body(Full::new(Bytes::from(body.clone())))
                .unwrap();
            let mut send = self.send.take().unwrap();
            let answered = tokio::time::timeout(Duration::from_secs(20), async {
                let resp = send.send_request(req).await.ok()?;
                let status = resp.status().as_u16();
                // A stream answer would never end; the parsers' refusals are all short.
                if resp
                    .headers()
                    .get("content-type")
                    .is_some_and(|v| v == "application/x-ndjson")
                {
                    return Some((status, false));
                }
                let _ = resp.into_body().collect().await;
                Some((status, true))
            })
            .await;
            if let Ok(Some((status, reusable))) = answered {
                if reusable {
                    self.send = Some(send);
                }
                return Some(status);
            }
        }
        None
    }
}

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

#[tokio::test]
async fn no_body_panics_the_parsers_and_a_refused_body_is_a_4xx() {
    let bearer = format!("parse-{}", std::process::id());
    let execd = Execd::start(&[], &[("EXECD_BEARER", &bearer)]);
    let mut client = Client {
        addr: SocketAddr::V4(execd.addr),
        bearer,
        send: None,
    };
    let secs = env_u64("EXECD_PARSE_SECS");
    let cases = env_u64("EXECD_PARSE_CASES").unwrap_or(3000);
    let first = env_u64("EXECD_PARSE_SEED").unwrap_or(0xC0FFEE);
    let started = Instant::now();
    let mut r = Rng::new(first);
    let (mut refused, mut accepted, mut skipped, mut n) = (0u64, 0u64, 0u64, 0u64);
    let mut failures: Vec<String> = Vec::new();
    loop {
        let more = match secs {
            Some(s) => started.elapsed() < Duration::from_secs(s),
            None => n < cases,
        };
        if !more || failures.len() >= 5 {
            break;
        }
        n += 1;
        let case_seed = r.next();
        let mut cr = Rng::new(case_seed);
        let body = if cr.below(4) == 0 {
            random_bytes(&mut cr)
        } else {
            mutate(&mut cr)
        };
        let ok = execd::__fuzz::request(&body);
        if ok {
            // Accepted means it would run: only if it is the harmless command.
            let harmless = serde_json::from_slice::<serde_json::Value>(&body)
                .ok()
                .is_some_and(|v| {
                    v["command"] == "true"
                        && v["args"]
                            .as_array()
                            .is_none_or(|a| a.is_empty() || a.iter().all(|x| x.is_string()))
                });
            if !harmless {
                skipped += 1;
                continue;
            }
            accepted += 1;
        } else {
            refused += 1;
        }
        let status = client.post(body.clone()).await;
        let bad = match (ok, status) {
            (_, None) => Some("no answer: the connection closed".to_string()),
            (false, Some(s)) if !(400..500).contains(&s) => {
                Some(format!("refused body answered {s}"))
            }
            (true, Some(s)) if s >= 500 && s != 500 => Some(format!("accepted body answered {s}")),
            _ => None,
        };
        if let Some(why) = bad {
            failures.push(format!(
                "case seed {case_seed:#x} (EXECD_PARSE_SEED={first:#x}, case {n}): {why}\n  body ({} bytes): {:?}",
                body.len(),
                String::from_utf8_lossy(&body[..body.len().min(300)])
            ));
        }
    }
    println!(
        "{n} bodies in {:?}: {refused} refused, {accepted} accepted, {skipped} skipped",
        started.elapsed()
    );
    // The server is alive and unharmed.
    let still = client
        .post(br#"{"command":"true","args":[]}"#.to_vec())
        .await;
    assert_eq!(still, Some(200), "execd stopped answering");
    let log = execd.log();
    assert!(!log.contains("panicked"), "execd panicked:\n{log}");
    assert!(failures.is_empty(), "\n{}\n", failures.join("\n"));
    assert!(
        refused > n / 10,
        "the mutations refused almost nothing: {refused} of {n}"
    );
}
