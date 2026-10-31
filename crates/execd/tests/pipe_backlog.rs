//! A command that exits with more in its pipe than the pipe's size said it could hold still has
//! all of it delivered. `F_GETPIPE_SZ` is read when the command starts; a command can raise its
//! own pipe afterwards (and Linux 7.0 packs large writes into folios, so a pipe holds more bytes
//! than its nominal size), so the bytes owed at the exit are what the pipe holds then.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use base64::Engine as _;
use beyond_test_support::ports;
use serde_json::{Value, json};
use support::http::post;

const BYTES: usize = 600_000;

/// Waits for execd to have sized the pipe, raises it to 1 MiB (`F_SETPIPE_SZ`), writes `BYTES`
/// into it in one go and exits at once, long before a reader could have taken them.
const GROWS_ITS_PIPE: &str = "perl -e 'use Fcntl; select(undef, undef, undef, 0.3); \
     fcntl(STDOUT, 1031, 1048576) or die; syswrite(STDOUT, \"x\" x 600000)'";

async fn start() -> (execd::Handle, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let o = execd::Options::new(dir.path());
    (execd::serve(ports::tokio_listener().await, o).unwrap(), dir)
}

fn decode(v: &str) -> Vec<u8> {
    base64::engine::general_purpose::STANDARD.decode(v).unwrap()
}

#[tokio::test]
async fn a_stream_delivers_what_a_grown_pipe_held_at_the_exit() {
    let (server, _dir) = start().await;
    let a = post(
        server.local_addr(),
        &[],
        &json!({ "command": "sh", "args": ["-c", GROWS_ITS_PIPE], "stream": true }),
    )
    .await;
    assert_eq!(a.status, 200);
    let mut got = 0;
    for f in a.frames() {
        if let Some(b) = f.get("stdout").and_then(Value::as_str) {
            got += decode(b).len();
        }
    }
    assert_eq!(got, BYTES);
    server.shutdown().await;
}

#[tokio::test]
async fn an_answer_carries_what_a_grown_pipe_held_at_the_exit() {
    let (server, _dir) = start().await;
    let a = post(
        server.local_addr(),
        &[],
        &json!({ "command": "sh", "args": ["-c", GROWS_ITS_PIPE] }),
    )
    .await
    .json();
    assert_eq!(a["exit_code"], 0, "{a}");
    assert_eq!(decode(a["stdout"].as_str().unwrap()).len(), BYTES);
    server.shutdown().await;
}
