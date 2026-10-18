//! `AI_USAGE_DIR`: `ai.usage` rows go to hourly `.jsonl` files instead of stdout, in the same format.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::*;

const BODY: &str = r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#;

async fn one_completion(gw: &Gateway, sk: &ed25519_dalek::SigningKey, tenant: u64) {
    let resp = test_client()
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header(
            "authorization",
            format!("Bearer {}", billing_vkey(sk, tenant)),
        )
        .header("content-type", "application/json")
        .body(BODY)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let _ = resp.bytes().await;
}

fn fresh_dir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("usage-files-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn rows_in(dir: &std::path::Path) -> Vec<(String, String)> {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.map(|e| {
                let p = e.unwrap().path();
                (
                    p.file_name().unwrap().to_str().unwrap().to_owned(),
                    std::fs::read_to_string(&p).unwrap(),
                )
            })
            .collect()
        })
        .unwrap_or_default()
}

/// With `AI_USAGE_DIR` set, one completion writes exactly one row, to the current hour's file, and
/// none to stdout. The row is the line stdout would have carried: same fields, one JSON object per
/// line, `.jsonl` so logfwd's `*.jsonl` glob picks it up.
#[tokio::test]
async fn a_completion_writes_one_row_to_the_hourly_file_and_none_to_stdout() {
    let (pubkey, sk) = test_keypair(91);
    let dir = fresh_dir("e2e");

    // Reference: the same request on a stdout gateway.
    let mock = MockUpstream::start(Mode::Json).await;
    let reference = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    one_completion(&reference, &sk, 91).await;
    let want = wait_usage_rows(&reference, 1, 5).await;
    assert_eq!(want.len(), 1);

    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .env("AI_USAGE_DIR", dir.to_str().unwrap())
        .start()
        .await;
    one_completion(&gw, &sk, 91).await;

    // The row is written when the request's accounting finishes, just after the response.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let mut files = rows_in(&dir);
    while files.iter().all(|(_, t)| !t.ends_with('\n')) && std::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        files = rows_in(&dir);
    }
    assert_eq!(files.len(), 1, "{files:?}");
    let (name, text) = &files[0];
    assert!(
        name.starts_with("ai-usage-")
            && name.ends_with(".jsonl")
            && name.len() == "ai-usage-YYYYMMDDTHH.jsonl".len(),
        "{name}"
    );
    assert!(text.ends_with('\n'));
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 1, "{text}");
    let row: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(row["target"], "ai.usage");
    let fields = row["fields"].as_object().unwrap();
    let want_fields = want[0].as_object().unwrap();
    let keys =
        |m: &serde_json::Map<String, serde_json::Value>| m.keys().cloned().collect::<Vec<_>>();
    assert_eq!(keys(fields), keys(want_fields));
    assert_eq!(fields["input_tokens"], want_fields["input_tokens"]);
    // Byte-level shape: the fmt layer's JSON line, timestamp first.
    assert!(lines[0].starts_with(r#"{"timestamp":""#), "{}", lines[0]);

    assert!(
        usage_rows_of(&gw).is_empty(),
        "rows must go only to the file; log:\n{}",
        gw.log()
    );
    let _ = std::fs::remove_dir_all(&dir);
}
