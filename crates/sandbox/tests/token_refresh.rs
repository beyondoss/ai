//! A bearer token that expires: on a `401` the client asks its source once for a new token and
//! repeats that request once. Nothing else is retried.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use beyond_sandbox::{Exec, HttpExec, TokenSource};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// An endpoint that accepts exactly one bearer token (or none), and records every `Authorization`
/// it was shown.
struct Endpoint {
    url: String,
    seen: Arc<Mutex<Vec<String>>>,
}

async fn endpoint(accepts: Option<&'static str>) -> Endpoint {
    let listener = beyond_test_support::ports::tokio_listener().await;
    let url = format!("http://{}/exec", listener.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = seen.clone();
    tokio::spawn(async move {
        loop {
            let (mut sock, _) = listener.accept().await.unwrap();
            let log = log.clone();
            tokio::spawn(async move {
                loop {
                    let mut head = Vec::new();
                    let mut b = [0u8; 1];
                    while !head.ends_with(b"\r\n\r\n") {
                        if sock.read(&mut b).await.unwrap_or(0) == 0 {
                            return;
                        }
                        head.push(b[0]);
                    }
                    let head = String::from_utf8_lossy(&head).to_string();
                    let header = |name: &str| {
                        head.lines().find_map(|l| {
                            let (k, v) = l.split_once(':')?;
                            k.eq_ignore_ascii_case(name).then(|| v.trim().to_string())
                        })
                    };
                    let len: usize = header("content-length").unwrap().parse().unwrap();
                    let mut body = vec![0u8; len];
                    sock.read_exact(&mut body).await.unwrap();
                    let auth = header("authorization").unwrap_or_default();
                    log.lock().unwrap().push(auth.clone());
                    let ok = accepts.is_some_and(|t| auth == format!("Bearer {t}"));
                    let (status, payload) = if ok {
                        ("200 OK", r#"{"exit_code":0,"stdout":"aGk=","stderr":""}"#)
                    } else {
                        ("401 Unauthorized", r#"{"error":"token expired"}"#)
                    };
                    let resp = format!(
                        "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{payload}",
                        payload.len()
                    );
                    if sock.write_all(resp.as_bytes()).await.is_err() {
                        return;
                    }
                }
            });
        }
    });
    Endpoint { url, seen }
}

/// Mints `t1`, `t2`, ... and counts how often it was asked.
fn counting() -> (TokenSource, Arc<AtomicUsize>) {
    let mints = Arc::new(AtomicUsize::new(0));
    let m = mints.clone();
    let source = TokenSource::from_fn(move || {
        let n = m.fetch_add(1, Ordering::SeqCst) + 1;
        async move { Ok(format!("t{n}")) }
    });
    (source, mints)
}

async fn run(exec: &HttpExec) -> std::io::Result<String> {
    exec.run("true", &[], None, Duration::from_secs(10))
        .await
        .map(|r| r.stdout_text().into_owned())
}

#[tokio::test]
async fn a_401_asks_the_source_once_and_repeats_the_request_once() {
    let ep = endpoint(Some("t2")).await;
    let (source, mints) = counting();
    let exec = HttpExec::new(&ep.url).unwrap().bearer(source);
    assert_eq!(run(&exec).await.unwrap(), "hi");
    assert_eq!(mints.load(Ordering::SeqCst), 2);
    assert_eq!(*ep.seen.lock().unwrap(), ["Bearer t1", "Bearer t2"]);
}

#[tokio::test]
async fn a_token_that_works_is_never_asked_for_again() {
    let ep = endpoint(Some("t1")).await;
    let (source, mints) = counting();
    let exec = HttpExec::new(&ep.url).unwrap().bearer(source);
    for _ in 0..3 {
        run(&exec).await.unwrap();
    }
    assert_eq!(mints.load(Ordering::SeqCst), 1);
    assert_eq!(ep.seen.lock().unwrap().len(), 3);
}

#[tokio::test]
async fn a_second_401_is_final() {
    let ep = endpoint(None).await;
    let (source, mints) = counting();
    let exec = HttpExec::new(&ep.url).unwrap().bearer(source);
    let err = run(&exec).await.unwrap_err().to_string();
    assert!(err.contains("returned 401"), "{err}");
    assert_eq!(mints.load(Ordering::SeqCst), 2, "one mint, one renewal");
    assert_eq!(ep.seen.lock().unwrap().len(), 2, "one request, one repeat");
}

#[tokio::test]
async fn a_source_that_gives_the_same_token_again_is_still_repeated_once() {
    let ep = endpoint(Some("other")).await;
    let mints = Arc::new(AtomicUsize::new(0));
    let counted = mints.clone();
    let source = TokenSource::from_fn(move || {
        counted.fetch_add(1, Ordering::SeqCst);
        async { Ok("same".to_string()) }
    });
    let exec = HttpExec::new(&ep.url).unwrap().bearer(source);
    let err = run(&exec).await.unwrap_err().to_string();
    assert!(err.contains("returned 401"), "{err}");
    assert_eq!(
        mints.load(Ordering::SeqCst),
        2,
        "the first token, one renewal"
    );
    assert_eq!(*ep.seen.lock().unwrap(), ["Bearer same", "Bearer same"]);
}

#[tokio::test]
async fn a_fixed_token_has_nothing_to_ask_again() {
    let ep = endpoint(Some("other")).await;
    let exec = HttpExec::new(&ep.url).unwrap().bearer("stale");
    let err = run(&exec).await.unwrap_err().to_string();
    assert!(err.contains("returned 401"), "{err}");
    assert_eq!(*ep.seen.lock().unwrap(), ["Bearer stale"]);
}

#[tokio::test]
async fn concurrent_requests_that_all_fail_share_one_renewal() {
    let ep = endpoint(Some("t2")).await;
    let (source, mints) = counting();
    let exec = Arc::new(HttpExec::new(&ep.url).unwrap().bearer(source));
    let calls: Vec<_> = (0..8)
        .map(|_| {
            let exec = exec.clone();
            tokio::spawn(async move { run(&exec).await })
        })
        .collect();
    for c in calls {
        assert_eq!(c.await.unwrap().unwrap(), "hi");
    }
    assert_eq!(
        mints.load(Ordering::SeqCst),
        2,
        "the first token and one renewal"
    );
}

#[tokio::test]
async fn a_failing_source_is_the_requests_error() {
    let ep = endpoint(Some("t1")).await;
    let exec = HttpExec::new(&ep.url)
        .unwrap()
        .bearer(TokenSource::from_fn(|| async {
            Err("vault is sealed".to_string())
        }));
    let err = run(&exec).await.unwrap_err().to_string();
    assert!(err.contains("vault is sealed"), "{err}");
    assert!(
        ep.seen.lock().unwrap().is_empty(),
        "no request without a token"
    );
}

#[tokio::test]
async fn a_token_command_prints_the_token() {
    let ep = endpoint(Some("from-cmd")).await;
    let exec = HttpExec::new(&ep.url)
        .unwrap()
        .bearer(TokenSource::command(["sh", "-c", "echo from-cmd"]));
    assert_eq!(run(&exec).await.unwrap(), "hi");
    let failing = TokenSource::command(["sh", "-c", "exit 3"]);
    let exec = HttpExec::new(&ep.url).unwrap().bearer(failing);
    assert!(
        run(&exec)
            .await
            .unwrap_err()
            .to_string()
            .contains("exited with")
    );
}
