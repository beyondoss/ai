//! The Rust floor for execd's memory: a tokio `current_thread` runtime and a hyper HTTP/1.1 server
//! that answers every request with `ok`, and nothing else. Build it the way execd ships, to see what
//! the runtime and the HTTP stack cost before execd adds anything:
//!
//! ```sh
//! RUSTFLAGS="-C relocation-model=static" cargo build --release --target x86_64-unknown-linux-musl \
//!     -p beyond-execd --example floor
//! target/x86_64-unknown-linux-musl/release/examples/floor 127.0.0.1:8080
//! ```
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::convert::Infallible;

use bytes::Bytes;
use http_body_util::Full;
use hyper::service::service_fn;
use hyper::{Request, Response};
use hyper_util::rt::TokioIo;

fn main() {
    let addr = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "127.0.0.1:8080".into());
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
        loop {
            let (sock, _) = listener.accept().await.unwrap();
            tokio::spawn(async move {
                let svc = service_fn(|_: Request<hyper::body::Incoming>| async {
                    Ok::<_, Infallible>(Response::new(Full::new(Bytes::from_static(b"ok"))))
                });
                let _ = hyper::server::conn::http1::Builder::new()
                    .serve_connection(TokioIo::new(sock), svc)
                    .await;
            });
        }
    });
}
