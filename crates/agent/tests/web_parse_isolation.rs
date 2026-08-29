//! The `web` tool's HTML parsing runs in a separate, seccomp-confined process
//! (`tools::web::isolate`). These live here rather than in the unit suite because they need a real
//! agent binary to re-exec: `current_exe()` under `cargo test` is the test harness, which does not
//! answer to `__web-parse`. `CARGO_BIN_EXE_*` gives integration tests the built binary, and
//! `BEYOND_AI_AGENT_WEB_PARSER` points the tool at it.
//!
//! What is actually being defended here: the `web` tool is the only one whose input is bytes an
//! arbitrary web server chose, and `scraper`/`htmd` reach ~190 `unsafe` blocks with them. The child
//! holds no credentials and can barely syscall, so a memory-safety bug in `tendril` buys an attacker
//! a process that cannot open a file, dial a socket, or exec.
//!
//! Not covered here: the parent's own `env_clear()`/`current_dir("/")` on the spawn. Asserting those
//! from a test would mean setting `AI_AGENT_KEY` in this process, and `std::env::set_var` is `unsafe`
//! in edition 2024 while this crate `forbid`s `unsafe_code`. They are single lines in
//! `isolate::parse` and are reviewed there rather than tested here.
//!
//! The load-bearing assertion in most of these is **exit status**, not just output. A child can write
//! a correct answer and *then* trip its own filter on the way out — that really happened, via a
//! missing `sigaltstack` — so a test that only checked the text would have called a misconfigured
//! sandbox a pass.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::io::Write as _;
use std::process::{Command, Stdio};

use serde_json::{Value, json};

const AGENT: &str = env!("CARGO_BIN_EXE_beyond-ai-agent");

const PAGE: &str = r#"<html><head><title>Doc</title></head><body>
<h1>Hi</h1><p>Some <b>bold</b> prose.</p>
<table><tr><th>a</th><th>b</th></tr><tr><td>1</td><td>2</td></tr></table>
<ul><li>one</li><li>two</li></ul></body></html>"#;

/// Drive the child directly: write one request, read one response, return it with the exit status.
fn parse(mode: &str, input: Value, html: &str) -> (Value, std::process::ExitStatus) {
    let mut child = Command::new(AGENT)
        .arg("__web-parse")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn parser");
    let req = serde_json::to_vec(&json!({ "mode": mode, "input": input, "html": html })).unwrap();
    child.stdin.take().unwrap().write_all(&req).unwrap();
    let out = child.wait_with_output().expect("wait parser");
    let parsed = serde_json::from_slice(&out.stdout).unwrap_or_else(|e| {
        panic!(
            "response was not JSON ({e}); stdout={:?} stderr={:?} status={:?}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
            out.status
        )
    });
    (parsed, out.status)
}

/// Every parsing mode must survive its own sandbox. A syscall missing from the allowlist shows up
/// here as a non-zero exit even when the output is perfectly correct.
#[test]
fn every_parsing_mode_completes_inside_the_sandbox() {
    let cases = [
        (
            "markdown",
            json!({ "url": "http://x/", "mode": "markdown" }),
            "# Hi",
        ),
        (
            "outline",
            json!({ "url": "http://x/", "mode": "outline" }),
            "selector",
        ),
        (
            "table",
            json!({ "url": "http://x/", "mode": "table" }),
            "a\tb",
        ),
        (
            "locate",
            json!({ "url": "http://x/", "mode": "locate", "text": "one" }),
            "li",
        ),
        (
            "extract",
            json!({ "url": "http://x/", "mode": "extract", "selector": "li", "fields": { "t": "self" } }),
            "row(s)",
        ),
    ];
    for (mode, input, needle) in cases {
        let (resp, status) = parse(mode, input, PAGE);
        assert!(
            status.success(),
            "`{mode}` did not exit cleanly ({status:?}) — a non-zero status here usually means the \
             seccomp allowlist is missing a syscall, even when the output below looks right: {resp}"
        );
        let text = resp
            .get("Ok")
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("`{mode}` returned an error: {resp}"));
        assert!(
            text.contains(needle),
            "`{mode}` output missing {needle:?}: {text}"
        );
    }
}

/// A parse *error* is a result, not a crash: it must come back over the wire as `Err` with the
/// parser's own wording, and the child must still exit cleanly.
#[test]
fn a_parser_error_returns_cleanly_rather_than_killing_the_child() {
    let (resp, status) = parse(
        "extract",
        json!({ "url": "http://x/", "mode": "extract", "selector": ".", "fields": { "t": "self" } }),
        PAGE,
    );
    assert!(status.success(), "child died on a bad selector: {status:?}");
    let err = resp
        .get("Err")
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("expected an Err response, got {resp}"));
    assert!(
        err.contains("selector"),
        "error should name the bad selector: {err}"
    );
}

/// A page far larger than a pipe buffer must not deadlock: the parent writes the request on one
/// thread while reading the reply on another, and this is the case that breaks if that is ever
/// collapsed back into a single sequential write-then-read.
#[test]
fn a_page_larger_than_a_pipe_buffer_round_trips() {
    let big = format!(
        "<html><body>{}</body></html>",
        "<p>lorem ipsum dolor sit amet</p>".repeat(40_000)
    );
    assert!(big.len() > 1 << 20, "fixture should exceed a pipe buffer");
    let (resp, status) = parse(
        "outline",
        json!({ "url": "http://x/", "mode": "outline" }),
        &big,
    );
    assert!(
        status.success(),
        "large page did not exit cleanly: {status:?}"
    );
    let text = resp.get("Ok").and_then(Value::as_str).expect("Ok response");
    assert!(
        text.contains("40000\tp"),
        "outline should count every <p>: {text}"
    );
}
