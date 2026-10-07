//! Booting from the on-disk snapshots with NATS unreachable: which sets a snapshot may seed, and
//! that a fail-closed set stays fail-closed when the snapshot is not proof of a complete read.
//!
//! Hermetic: NATS is a closed port, and the snapshot files are written here before boot.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::path::{Path, PathBuf};

use beyond_ai::key::{VirtualKey, mint};
use common::*;
use store::snapshot::SnapshotWriter;
use store::{KvEntry, KvUpdate, VersionToken, WatchCursor};

const CHAT_PATH: &str = "/openai/v1/chat/completions";
const CHAT: &str = r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#;

fn vkey(sk: &ed25519_dalek::SigningKey, tenant_id: u64) -> String {
    mint(
        &VirtualKey {
            tenant_id,
            vpc_id: 7,
            key_id: None,
        },
        1,
        sk,
    )
}

/// A per-test deny snapshot path; the allowance snapshot is `{path}.allowance`.
struct SnapPaths(PathBuf);

impl SnapPaths {
    fn new(tag: &str) -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "beyond-ai-snap-{tag}-{}-{n}.log",
            std::process::id()
        ));
        let s = Self(p);
        s.cleanup();
        s
    }
    fn deny(&self) -> &Path {
        &self.0
    }
    fn allowance(&self) -> PathBuf {
        PathBuf::from(format!("{}.allowance", self.0.display()))
    }
    fn as_str(&self) -> String {
        self.0.to_str().unwrap().to_string()
    }
    fn cleanup(&self) {
        let _ = std::fs::remove_file(self.deny());
        let _ = std::fs::remove_file(self.allowance());
    }
}

impl Drop for SnapPaths {
    fn drop(&mut self) {
        self.cleanup();
    }
}

/// Write a snapshot holding one `Put` per key and, when `cursor` is `Some`, a resume cursor.
/// `None` is what a rebuild that crashed before its cursor record leaves behind.
fn write_snapshot(path: &Path, keys: &[(&str, &[u8])], cursor: Option<u64>) {
    let mut w = SnapshotWriter::open(path, u64::MAX).unwrap();
    for (i, (key, value)) in keys.iter().enumerate() {
        w.write_update(&KvUpdate::Put(KvEntry {
            key: (*key).to_string(),
            value: value.to_vec(),
            version: VersionToken::from_u64(i as u64 + 1),
        }))
        .unwrap();
    }
    match cursor {
        Some(rev) => {
            let _ = w.checkpoint(&WatchCursor::from_u64(rev)).unwrap();
        }
        None => w.flush().unwrap(),
    }
}

async fn status(gw: &Gateway, key: &str) -> (u16, String) {
    let resp = test_client()
        .post(format!("{}{CHAT_PATH}", gw.url()))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .body(CHAT)
        .send()
        .await
        .unwrap();
    let s = resp.status().as_u16();
    (s, resp.text().await.unwrap_or_default())
}

/// An allowance snapshot without a cursor record is not proof of a complete read (a rebuild that
/// crashed midway leaves some exhausted tenants and no cursor). Booted with NATS down, the
/// allowance-set must stay unseeded: managed requests get 503 and `/readyz` is 503, rather than
/// serving every exhausted tenant the partial file is missing.
/// claim: REL-5, REL-13
/// defect: D257
#[tokio::test]
async fn an_allowance_snapshot_without_a_cursor_stays_fail_closed() {
    let snap = SnapPaths::new("allow-nocursor");
    write_snapshot(&snap.allowance(), &[("allowance.2571", b"1")], None);
    let (pubkey, sk) = test_keypair(57);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(closed_port(), &mock.authority(), &b64(&pubkey))
        .snapshot_path(snap.as_str())
        .skip_allowance_ready()
        .start()
        .await;
    // Give the watcher time to load the snapshot (it would have flipped ready by now).
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    for tenant in [2571, 2572] {
        let (s, body) = status(&gw, &vkey(&sk, tenant)).await;
        assert_eq!(
            s, 503,
            "tenant {tenant}: a partial allowance snapshot seeded the set ({s} {body})"
        );
        assert!(body.contains("allowance unavailable"), "{body}");
    }
    let (ready, body) = gw.admin_get("/readyz").await;
    assert_eq!(
        ready, 503,
        "readyz {ready} {body} from a cursorless snapshot"
    );
    assert_eq!(
        mock.hits(),
        0,
        "an unseeded allowance-set must not reach upstream"
    );
}

/// The same allowance snapshot with a resumable cursor is a complete read: it seeds the set and the
/// pod is ready before NATS is reachable. The exhausted tenant gets 402, the others are served.
/// claim: REL-5, REL-13
/// defect: D257
#[tokio::test]
async fn an_allowance_snapshot_with_a_cursor_seeds_ready() {
    let snap = SnapPaths::new("allow-cursor");
    write_snapshot(&snap.allowance(), &[("allowance.2573", b"1")], Some(5));
    let (pubkey, sk) = test_keypair(58);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(closed_port(), &mock.authority(), &b64(&pubkey))
        .snapshot_path(snap.as_str())
        .start()
        .await;
    let (s, body) = status(&gw, &vkey(&sk, 2573)).await;
    assert_eq!(s, 402, "the exhausted tenant from the snapshot: {body}");
    let (s, body) = status(&gw, &vkey(&sk, 2574)).await;
    assert_eq!(s, 200, "a tenant absent from the snapshot: {body}");
    let (ready, body) = gw.admin_get("/readyz").await;
    assert_eq!(ready, 200, "readyz {ready} {body}");
}

/// The deny-set is fail-open, so any entry surviving in a snapshot is strictly better than none: a
/// deny snapshot without a cursor still blocks its denied tenant while NATS is down.
/// claim: REL-13
/// defect: D257
#[tokio::test]
async fn a_deny_snapshot_without_a_cursor_still_blocks_with_nats_down() {
    let snap = SnapPaths::new("deny-nocursor");
    write_snapshot(snap.deny(), &[("blackhole.2575", b"fraud")], None);
    // A complete allowance snapshot, so the pod is ready and the deny check is what decides. It is
    // the empty scan's revision 0: complete (a cursor record) but not resumable, which still seeds.
    write_snapshot(&snap.allowance(), &[], Some(0));
    let (pubkey, sk) = test_keypair(59);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(closed_port(), &mock.authority(), &b64(&pubkey))
        .snapshot_path(snap.as_str())
        .start()
        .await;
    // The deny watcher loads its snapshot independently of the allowance one the builder waited
    // for, so poll briefly.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let (s, body) = loop {
        let got = status(&gw, &vkey(&sk, 2575)).await;
        if got.0 == 403 || std::time::Instant::now() > deadline {
            break got;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    };
    assert_eq!(
        s, 403,
        "the denied tenant from a cursorless snapshot: {body}"
    );
    let (s, body) = status(&gw, &vkey(&sk, 2576)).await;
    assert_eq!(s, 200, "an undenied tenant: {body}");
}
