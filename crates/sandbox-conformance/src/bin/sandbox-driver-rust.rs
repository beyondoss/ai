//! The Rust client's conformance driver (`sdk/conformance/DRIVER.md`): reads requests as JSON lines on
//! stdin, answers on stdout. The driver itself is the reference driver of the `beyond-sandbox` tests, included
//! rather than copied.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, dead_code)]

#[path = "../../../sandbox/tests/sdk_conformance/driver.rs"]
mod driver;

use std::io::Write as _;

use driver::{CapsOverride, Config, Driver};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt as _, BufReader};

/// What a scenario's `requires.features` may name (sdk/conformance/DRIVER.md): this client reads the
/// raw stream, so it has all of them.
const FEATURES: [&str; 3] = ["frames", "unbounded_stream", "exit_signal"];

fn config() -> Config {
    let url = std::env::var("CONFORMANCE_URL").unwrap_or_default();
    let bearer = std::env::var("CONFORMANCE_BEARER")
        .ok()
        .filter(|b| !b.is_empty());
    let token_cmd = std::env::var("CONFORMANCE_TOKEN_CMD")
        .ok()
        .filter(|c| !c.is_empty());
    let caps: Value = std::env::var("CONFORMANCE_CAPS")
        .ok()
        .filter(|c| !c.is_empty())
        .map(|c| serde_json::from_str(&c).expect("CONFORMANCE_CAPS is JSON"))
        .unwrap_or(Value::Null);
    Config {
        url,
        bearer,
        token_cmd,
        caps: CapsOverride {
            find_printf: caps["find_printf"].as_bool(),
            // Leaked once per process: the override is `'static` in the shared driver.
            search_engine: caps["search_engine"]
                .as_str()
                .map(|s| &*Box::leak(s.to_string().into_boxed_str())),
        },
        jobs_root: None,
    }
}

#[tokio::main]
async fn main() {
    let cfg = config();
    // Connect (probe) on the first request that needs it; `hello` never reaches the endpoint.
    let mut driver: Option<Driver> = None;
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let mut out = std::io::stdout().lock();
    while let Some(line) = lines.next_line().await.expect("stdin") {
        if line.trim().is_empty() {
            continue;
        }
        let resp = match serde_json::from_str::<Value>(&line) {
            Err(e) => {
                json!({"id": null, "error": {"kind": "invalid_request", "detail": e.to_string()}})
            }
            Ok(req) if req["op"] == "hello" => {
                json!({"id": req["id"], "ok": {"driver": "agent-reference", "protocol": 1,
                    "ops": Driver::ops_with(true), "features": FEATURES}})
            }
            Ok(req) => {
                if driver.is_none() {
                    driver = Some(Driver::connect(&cfg).await);
                }
                driver.as_ref().unwrap().handle(&req).await
            }
        };
        writeln!(out, "{resp}").unwrap();
        out.flush().unwrap();
    }
}
