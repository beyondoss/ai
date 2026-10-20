//! The sandbox: the repo's reference exec endpoint (`ExecMock`), run as a service.
//!
//! It is the same code the conformance suite grades, so it speaks exec v1.1, v1.2 streaming, v1.3
//! sandbox jobs and v1.4 job-end notify, and runs every command as a real process in this
//! container. One timeline line per exec request and per callback goes to stdout (CloudWatch).

use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use beyond_ai_test_support::exec_mock::{ExecMock, MockOptions};
use serde_json::json;

pub struct Args {
    pub listen: String,
    pub root: PathBuf,
    pub home: PathBuf,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// The options a deployment runs: a conforming v1.4 endpoint (kept jobs survive idling, which is
/// what lets it advertise notify) with the spec's heartbeat.
///
/// With `bearer` the endpoint answers 401 to any request without it: an endpoint on the public
/// internet, reached with the bearer its session's grant carries.
pub fn options(home: PathBuf, bearer: Option<String>) -> MockOptions {
    MockOptions {
        home: Some(home),
        require_bearer: bearer,
        ..MockOptions::notifying()
    }
}

fn line(event: &str, detail: serde_json::Value) {
    println!(
        "{}",
        json!({"ts_ms": now_ms(), "src": "sandbox", "event": event, "detail": detail})
    );
}

pub async fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(&args.root)?;
    std::fs::create_dir_all(&args.home)?;
    // From the environment, not an argument: a command line is visible to every process on the host.
    let bearer = std::env::var("DEMO_EXEC_BEARER")
        .ok()
        .filter(|b| !b.is_empty());
    let mock = ExecMock::start_on_with(
        &args.listen,
        &args.root,
        options(args.home.clone(), bearer.clone()),
    )
    .await?;
    line(
        "listening",
        json!({"listen": args.listen, "url": mock.url, "bearer_required": bearer.is_some()}),
    );

    let (mut streams_seen, mut ended_seen, mut callbacks_seen) = (0usize, 0usize, 0usize);
    let mut requests_seen = 0usize;
    loop {
        // Every request the endpoint answered, streamed or not: how a run can say which calls the
        // sandbox saw, and when.
        let requests = mock.requests();
        for r in requests.iter().skip(requests_seen) {
            let cmd: String = format!("{} {}", r.command(), r.body["args"])
                .chars()
                .take(160)
                .collect();
            line("exec_request", json!({"command": cmd}));
        }
        requests_seen = requests.len();
        let streams = mock.streams();
        for s in streams.iter().skip(streams_seen) {
            let cmd: String = format!("{} {}", s.command, s.args.join(" "))
                .chars()
                .take(160)
                .collect();
            line("exec_start", json!({"command": cmd}));
        }
        streams_seen = streams.len();
        let ended = streams.iter().filter(|s| s.end.is_some()).count();
        if ended != ended_seen {
            line("exec_end", json!({"ended": ended}));
            ended_seen = ended;
        }
        let callbacks = mock.callbacks();
        for c in callbacks.iter().skip(callbacks_seen) {
            line(
                "job_ended_callback",
                json!({"job": c.id, "attempt": c.n, "status": c.status, "error": c.error,
                       "took_ms": c.ended.duration_since(c.started).as_millis() as u64}),
            );
        }
        callbacks_seen = callbacks.len();
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}
