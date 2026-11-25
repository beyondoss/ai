//! The mark on the wire against the real binary: which incarnation of an id a client is talking to.
//! `mark` on start, list and the streams' first frame; an attach and a remove that name a mark touch
//! only that incarnation; a status-only attach streams no output; `remove` reports what it skipped; an
//! id is 1 to 64 characters of `[0-9a-z_-]`. Each test is red on a binary without the feature (the field
//! is absent or ignored).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::net::SocketAddr;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use support::Execd;
use support::http::post;

async fn job(addr: SocketAddr, op: Value) -> (u16, Value) {
    let a = post(addr, &[], &json!({ "command": "true", "job": op })).await;
    (a.status, a.json())
}

async fn start(addr: SocketAddr, id: &str, cmd: &[&str]) -> (u16, Value) {
    let body = json!({ "command": cmd[0], "args": &cmd[1..],
        "job": { "op": "start", "scope": "sm", "id": id, "hold": false } });
    let a = post(addr, &[], &body).await;
    (a.status, a.json())
}

async fn wait_ended(addr: SocketAddr, id: &str) {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let (_, v) = job(
            addr,
            json!({"op":"read","scope":"sm","id":id,"offset":0,"max":1}),
        )
        .await;
        if v["state"] == "ended" {
            return;
        }
        assert!(Instant::now() < deadline, "job never ended: {v}");
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
}

async fn listed(addr: SocketAddr, id: &str) -> Option<Value> {
    let (_, v) = job(addr, json!({"op":"list","scope":"sm"})).await;
    v["jobs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|j| j["id"] == id)
        .cloned()
}

async fn attach(addr: SocketAddr, id: &str, extra: Value) -> support::http::Answer {
    let mut j = json!({"op":"attach","scope":"sm","id":id,"offset":0});
    j.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    post(addr, &[], &json!({"command":"true","stream":true,"job":j})).await
}

#[tokio::test]
async fn start_and_list_carry_the_mark_the_command_and_the_output_window() {
    let execd = Execd::start(&[], &[]);
    let addr = SocketAddr::V4(execd.addr);
    let (s, a) = start(addr, "a1", &["sh", "-c", "printf hello"]).await;
    assert_eq!(s, 200);
    let mark = a["mark"].as_str().unwrap().to_string();
    assert!(!mark.is_empty());
    wait_ended(addr, "a1").await;
    let e = listed(addr, "a1").await.unwrap();
    assert_eq!(e["mark"], mark);
    assert_eq!(e["command"], "sh");
    assert_eq!(e["args"], json!(["-c", "printf hello"]));
    assert_eq!(e["outputStart"], 0);
    assert_eq!(e["outputEnd"], 5);
    // The id reused after a remove is a new incarnation with a new mark.
    let (_, r) = job(addr, json!({"op":"remove","scope":"sm","ids":["a1"]})).await;
    assert_eq!(r, json!({"running": [], "mismatch": []}));
    let (_, b) = start(addr, "a1", &["true"]).await;
    assert_ne!(b["mark"].as_str().unwrap(), mark);
}

#[tokio::test]
async fn a_held_start_says_its_mark_in_its_first_frame() {
    let execd = Execd::start(&[], &[]);
    let addr = SocketAddr::V4(execd.addr);
    let body = json!({"command":"true","stream":true,
        "job":{"op":"start","scope":"sm","id":"h1","hold":true}});
    let a = post(addr, &[], &body).await;
    let first = &a.frames()[0];
    assert!(first["started"].as_i64().is_some_and(|p| p > 1), "{first}");
    let mark = first["mark"].as_str().unwrap();
    assert_eq!(listed(addr, "h1").await.unwrap()["mark"], mark);
}

#[tokio::test]
async fn an_attach_answers_first_with_the_mark_and_where_it_starts() {
    let execd = Execd::start(&[], &[]);
    let addr = SocketAddr::V4(execd.addr);
    let (_, a) = start(addr, "a1", &["sh", "-c", "printf hello"]).await;
    let mark = a["mark"].as_str().unwrap().to_string();
    wait_ended(addr, "a1").await;
    for extra in [json!({}), json!({"mark": mark})] {
        let f = attach(addr, "a1", extra).await.frames();
        assert_eq!(
            f[0],
            json!({"attached": {"mark": mark, "from": 0}}),
            "{f:?}"
        );
        assert!(f.iter().any(|f| f["stdout"].is_string()), "{f:?}");
        assert!(f.last().unwrap()["exit_code"].is_i64(), "{f:?}");
    }
    // From an offset: `from` is that offset.
    let f = attach(addr, "a1", json!({"offset": 3})).await.frames();
    assert_eq!(f[0]["attached"]["from"], 3);
}

#[tokio::test]
async fn an_attach_naming_another_incarnation_is_refused_and_streams_nothing() {
    let execd = Execd::start(&[], &[]);
    let addr = SocketAddr::V4(execd.addr);
    let (_, a) = start(addr, "a1", &["sh", "-c", "printf first"]).await;
    let old = a["mark"].as_str().unwrap().to_string();
    wait_ended(addr, "a1").await;
    job(addr, json!({"op":"remove","scope":"sm","ids":["a1"]})).await;
    let (_, b) = start(addr, "a1", &["sh", "-c", "printf second"]).await;
    let new = b["mark"].as_str().unwrap().to_string();
    wait_ended(addr, "a1").await;
    let a = attach(addr, "a1", json!({"mark": old})).await;
    assert_eq!(a.status, 404, "{:?}", String::from_utf8_lossy(&a.body));
    // The retry with the right mark streams (the refusal is not a state).
    let f = attach(addr, "a1", json!({"mark": new})).await.frames();
    assert_eq!(f[0]["attached"]["mark"], new);
    assert!(f.iter().any(|f| f["stdout"].is_string()));
}

#[tokio::test]
async fn a_status_only_attach_starts_at_the_end_and_sends_no_output() {
    let execd = Execd::start(&[], &[]);
    let addr = SocketAddr::V4(execd.addr);
    start(
        addr,
        "a1",
        &["sh", "-c", "printf hello; sleep 1; printf later"],
    )
    .await;
    // Wait until the first bytes are in.
    let deadline = Instant::now() + Duration::from_secs(10);
    while listed(addr, "a1").await.unwrap()["outputEnd"] == 0 {
        assert!(Instant::now() < deadline, "no output");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let end = listed(addr, "a1").await.unwrap()["outputEnd"]
        .as_u64()
        .unwrap();
    assert!(end >= 5);
    let f = attach(addr, "a1", json!({"output": false})).await.frames();
    // From the end as it was at the attach: never before the end seen earlier, so no gap.
    assert!(f[0]["attached"]["from"].as_u64().unwrap() >= end);
    for frame in &f {
        assert!(
            !frame["stdout"].is_string() && !frame["stderr"].is_string() && frame["gap"].is_null(),
            "a status-only stream sent output: {frame}"
        );
    }
    assert_eq!(f.last().unwrap()["exit_code"], 0, "{f:?}");
}

#[tokio::test]
async fn remove_reports_what_it_skipped() {
    let execd = Execd::start(&[], &[]);
    let addr = SocketAddr::V4(execd.addr);
    let (_, run) = start(addr, "run", &["sleep", "30"]).await;
    let (_, done) = start(addr, "done", &["true"]).await;
    wait_ended(addr, "done").await;
    let run_mark = run["mark"].as_str().unwrap().to_string();
    let done_mark = done["mark"].as_str().unwrap().to_string();
    // A running job is reported running, and stays.
    let (_, r) = job(
        addr,
        json!({"op":"remove","scope":"sm","ids":["run","done","absent"],
               "marks":{"done": run_mark}}),
    )
    .await;
    assert_eq!(r, json!({"running": ["run"], "mismatch": ["done"]}), "{r}");
    assert!(listed(addr, "run").await.is_some());
    assert!(
        listed(addr, "done").await.is_some(),
        "a mismatch removed it"
    );
    // The retry with the right mark removes it.
    let (_, r) = job(
        addr,
        json!({"op":"remove","scope":"sm","ids":["done"],"marks":{"done": done_mark}}),
    )
    .await;
    assert_eq!(r, json!({"running": [], "mismatch": []}));
    assert!(listed(addr, "done").await.is_none());
    job(
        addr,
        json!({"op":"kill","scope":"sm","ids":["run"],"remove":true,"grace_ms":100}),
    )
    .await;
}

#[tokio::test]
async fn an_id_is_one_to_sixty_four_of_lowercase_digits_dash_and_underscore() {
    let execd = Execd::start(&[], &[]);
    let addr = SocketAddr::V4(execd.addr);
    for ok in ["a", "a-b_c", "0", &"z".repeat(64), "-", "_"] {
        let (s, _) = start(addr, ok, &["true"]).await;
        assert_eq!(s, 200, "{ok:?} refused");
    }
    for bad in ["", "A", "a.b", "a/b", "a b", &"z".repeat(65), "é"] {
        let (s, _) = start(addr, bad, &["true"]).await;
        assert_eq!(s, 400, "{bad:?} accepted");
    }
}
