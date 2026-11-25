//! M7: what a keeper costs, in a release build (a debug build's keeper is larger, and is not what runs).
//! 30 idle terminals: the proportional set size (`Pss`: a keeper is execd's own binary, so most of its
//! resident pages are text it shares with execd and its siblings) of every keeper stays under 1 MiB.
//! Run it with `mise run test:execd-keeper-release`; the file is compiled only without debug
//! assertions, so a debug `cargo test` does not see it.
#![cfg(not(debug_assertions))]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use serde_json::json;
use support::Execd;
use support::http::post;

/// The keeper processes of terminals whose spool is under `jobs`.
fn keepers(jobs: &std::path::Path) -> Vec<i32> {
    let needle = jobs.to_string_lossy().to_string();
    std::fs::read_dir("/proc")
        .unwrap()
        .flatten()
        .filter_map(|e| {
            let raw = std::fs::read(e.path().join("cmdline")).ok()?;
            let text = String::from_utf8_lossy(&raw).replace('\0', " ");
            (text.contains("--execd-run-as-helper keep ") && text.contains(&needle))
                .then(|| e.file_name().to_string_lossy().parse::<i32>().ok())?
        })
        .collect()
}

fn smaps_kib(pid: i32, field: &str) -> u64 {
    let text = std::fs::read_to_string(format!("/proc/{pid}/smaps_rollup")).unwrap();
    text.lines()
        .find(|l| l.starts_with(field))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|n| n.parse().ok())
        .unwrap()
}

#[tokio::test]
async fn thirty_keepers_are_each_under_one_mebibyte_of_pss() {
    let jobs = tempfile::tempdir().unwrap();
    let e = Execd::start(&[("EXECD_JOBS_DIR", jobs.path().to_str().unwrap())], &[]);
    let a = std::net::SocketAddr::V4(e.addr);
    for n in 0..30 {
        let body = json!({
            "command": "sh", "args": ["-c", "exec sleep 300"],
            "job": { "op": "start", "scope": "s", "id": format!("k{n}"), "hold": false,
                     "pty": { "cols": 80, "rows": 24 } },
        });
        let r = post(a, &[], &body).await;
        assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    }
    let ks = keepers(jobs.path());
    assert_eq!(ks.len(), 30, "{ks:?}");
    let mut rss: Vec<u64> = ks.iter().map(|p| smaps_kib(*p, "Rss:")).collect();
    let mut pss: Vec<u64> = ks.iter().map(|p| smaps_kib(*p, "Pss:")).collect();
    rss.sort_unstable();
    pss.sort_unstable();
    let at = |v: &[u64], q: usize| v[(v.len() * q / 100).min(v.len() - 1)];
    println!(
        "keeper n=30 Rss KiB p50 {} p95 {} max {}; Pss KiB p50 {} p95 {} max {}",
        at(&rss, 50),
        at(&rss, 95),
        rss[29],
        at(&pss, 50),
        at(&pss, 95),
        pss[29]
    );
    assert!(
        pss[29] < 1024,
        "a keeper is {} KiB of Pss, over the 1 MiB budget",
        pss[29]
    );
    let _ = post(
        a,
        &[],
        &json!({"command":"true","job":{"op":"wipe","scope":"s","grace_ms":200}}),
    )
    .await;
}
