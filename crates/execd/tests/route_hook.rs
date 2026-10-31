//! `Options::route_hook`: an embedder claims a request before execd reads it. Claimed requests skip
//! the activity count, still pass the authorizer, stream both ways, and leave every other path as it
//! was. All in-process, through `execd::serve`, over HTTP/2 so one connection carries a duplex stream.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll};
use std::time::Duration;

use beyond_agent::exec_endpoint::conformance::{Expectations, Verdict, verify};
use beyond_test_support::ports;
use bytes::Bytes;
use http_body_util::BodyExt as _;
use hyper::body::{Body, Frame, SizeHint};
use hyper_util::rt::{TokioExecutor, TokioIo};
use serde_json::json;
use support::http::{post, send};
use tokio::sync::mpsc;

const CLAIMED: &str = "/_embed/echo";

/// A body fed from a channel: ends when the sender is dropped.
struct Chan(mpsc::Receiver<Bytes>);

impl Body for Chan {
    type Data = Bytes;
    type Error = std::io::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, std::io::Error>>> {
        self.0.poll_recv(cx).map(|o| o.map(|b| Ok(Frame::data(b))))
    }

    fn size_hint(&self) -> SizeHint {
        SizeHint::default()
    }
}

/// Claims `POST CLAIMED`: answers at once, then echoes each request frame back as it arrives.
fn echo_hook(seen: Arc<AtomicUsize>) -> execd::RouteHook {
    Arc::new(move |req: execd::HttpRequest| {
        if req.method() != hyper::Method::POST || req.uri().path() != CLAIMED {
            return execd::Route::Declined(req);
        }
        seen.fetch_add(1, Ordering::SeqCst);
        execd::Route::Claimed(Box::pin(async move {
            let (tx, rx) = mpsc::channel(8);
            let mut body = req.into_body();
            tokio::spawn(async move {
                while let Some(Ok(frame)) = body.frame().await {
                    if let Ok(data) = frame.into_data()
                        && tx.send(data).await.is_err()
                    {
                        return;
                    }
                }
            });
            hyper::Response::new(Chan(rx).boxed())
        }))
    })
}

async fn start(opts: impl FnOnce(&mut execd::Options)) -> (execd::Handle, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let mut o = execd::Options::new(dir.path());
    opts(&mut o);
    let server = execd::serve(ports::tokio_listener().await, o).unwrap();
    (server, dir)
}

type Client = hyper::client::conn::http2::SendRequest<
    http_body_util::combinators::BoxBody<Bytes, std::io::Error>,
>;

async fn h2(addr: SocketAddr) -> Client {
    let sock = tokio::net::TcpStream::connect(addr).await.unwrap();
    let (send, conn) =
        hyper::client::conn::http2::handshake(TokioExecutor::new(), TokioIo::new(sock))
            .await
            .unwrap();
    tokio::spawn(conn);
    send
}

/// Opens the claimed stream with `headers`; returns the answer and the sender of the request body.
async fn open_echo(
    addr: SocketAddr,
    headers: &[(&str, &str)],
) -> (hyper::Response<hyper::body::Incoming>, mpsc::Sender<Bytes>) {
    let (tx, rx) = mpsc::channel(8);
    let mut req = hyper::Request::post(format!("http://{addr}{CLAIMED}"));
    for (n, v) in headers {
        req = req.header(*n, *v);
    }
    let resp = h2(addr)
        .await
        .send_request(req.body(Chan(rx).boxed()).unwrap())
        .await
        .unwrap();
    (resp, tx)
}

fn accept_key() -> execd::Authorizer {
    Arc::new(|h: &execd::HeaderMap| h.get("x-key").is_some_and(|v| v == "k"))
}

/// A claimed handler's head arrives before any request frame is sent, and then bytes flow both ways.
/// While the stream is open, execd counts nothing in flight.
#[tokio::test]
async fn a_claimed_stream_is_duplex_and_not_counted_as_activity() {
    let seen = Arc::new(AtomicUsize::new(0));
    let (server, _dir) = start(|o| o.route_hook = Some(echo_hook(seen.clone()))).await;
    let (resp, tx) = open_echo(server.local_addr(), &[]).await;
    assert_eq!(resp.status(), 200);
    let mut out = resp.into_body();
    for word in ["ping", "pong"] {
        tx.send(Bytes::from(word)).await.unwrap();
        let frame = tokio::time::timeout(Duration::from_secs(5), out.frame())
            .await
            .expect("echoed")
            .unwrap()
            .unwrap();
        assert_eq!(frame.into_data().unwrap(), word);
        assert_eq!(
            server.activity().in_flight,
            0,
            "a claimed stream must not count"
        );
    }
    // An exec request beside it still counts, so the zero above was the exclusion, not a dead counter.
    let busy = {
        let addr = server.local_addr();
        tokio::spawn(async move {
            post(
                addr,
                &[],
                &json!({"command":"sleep","args":["1"],"timeout_ms":10_000}),
            )
            .await
        })
    };
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(server.activity().in_flight, 1);
    assert_eq!(busy.await.unwrap().status, 200);
    assert_eq!(server.activity().in_flight, 0);
    drop(tx);
    assert_eq!(seen.load(Ordering::SeqCst), 1);
    server.shutdown().await;
}

/// The authorizer decides first: a refused claimed request never reaches the hook.
#[tokio::test]
async fn the_authorizer_still_applies_to_a_claimed_path() {
    let seen = Arc::new(AtomicUsize::new(0));
    let (server, _dir) = start(|o| {
        o.authorizer = Some(accept_key());
        o.route_hook = Some(echo_hook(seen.clone()));
    })
    .await;
    let (resp, _tx) = open_echo(server.local_addr(), &[]).await;
    assert_eq!(resp.status(), 401);
    assert_eq!(seen.load(Ordering::SeqCst), 0);
    let (resp, _tx) = open_echo(server.local_addr(), &[("x-key", "k")]).await;
    assert_eq!(resp.status(), 200);
    assert_eq!(seen.load(Ordering::SeqCst), 1);
    server.shutdown().await;
}

/// A hook that declines leaves execd as it was: the full verifier passes, the authorizer runs once per
/// request, a refusal is a `401`, and a path nobody claims is still an exec request or a `405`.
#[tokio::test]
async fn unclaimed_paths_behave_as_without_a_hook() {
    let seen = Arc::new(AtomicUsize::new(0));
    let calls = Arc::new(AtomicUsize::new(0));
    let c = calls.clone();
    let (server, _dir) = start(|o| {
        o.authorizer = Some(Arc::new(move |h: &execd::HeaderMap| {
            c.fetch_add(1, Ordering::SeqCst);
            h.get("x-key").is_some_and(|v| v == "k")
        }));
        o.route_hook = Some(echo_hook(seen.clone()));
    })
    .await;
    let addr = server.local_addr();
    let url = format!("http://{addr}/exec");
    let headers = [("x-key".to_string(), "k".to_string())];
    let report = verify(&url, &headers, &Expectations::default()).await;
    let not_passed: Vec<_> = report
        .checks
        .iter()
        .filter(|c| c.verdict != Verdict::Pass)
        .map(|c| (c.name, c.verdict))
        .collect();
    assert!(not_passed.is_empty(), "{not_passed:?}\n{}", report.render());

    calls.store(0, Ordering::SeqCst);
    let ok = post(
        addr,
        &[("x-key", "k")],
        &json!({"command":"true","timeout_ms":5000}),
    )
    .await;
    assert_eq!(ok.status, 200);
    assert_eq!(calls.load(Ordering::SeqCst), 1, "once per declined request");
    assert_eq!(
        post(addr, &[], &json!({"command":"true"})).await.status,
        401
    );
    assert_eq!(
        send(addr, "GET", "/exec", &[("x-key", "k")], b"")
            .await
            .status,
        405
    );
    assert_eq!(send(addr, "GET", "/livez", &[], b"").await.status, 200);
    assert_eq!(seen.load(Ordering::SeqCst), 0);
    // The same path on another method is not the hook's, and is an exec-less 405.
    assert_eq!(
        send(addr, "GET", CLAIMED, &[("x-key", "k")], b"")
            .await
            .status,
        405
    );
    server.shutdown().await;
}
