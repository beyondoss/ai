//! Per-tenant sealing: what a shared filesystem is allowed to hold, and what happens to a reader
//! holding the wrong key or looking at the wrong session.

// Test target: `.unwrap()`/`panic!` are how a test reports a failed setup.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use agent_core::Message;
use beyond_ai_agent::memory::file::FileBackend;
use beyond_ai_agent::memory::{MemPath, MemoryBackend, View};
use beyond_ai_agent::session_store::{
    Layout, RepoOptions, SessionMeta, SessionRepo, SessionStore, TenantCodec,
};
use tempfile::TempDir;

const DEK: [u8; 32] = [0x5a; 32];

fn sealed_repo(dir: &Path, tenant: &str, dek: &[u8; 32]) -> SessionRepo {
    SessionRepo::open_with(
        dir,
        RepoOptions {
            layout: Layout::Segmented {
                codec: Some(Arc::new(TenantCodec::new(tenant, dek))),
            },
            id_prefix: None,
        },
    )
    .unwrap()
}

/// Every regular file under `root`, recursively.
fn files_under(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir(root) else {
        return out;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            out.extend(files_under(&p));
        } else {
            out.push(p);
        }
    }
    out
}

fn assert_no_plaintext(root: &Path, needles: &[&str]) {
    let files = files_under(root);
    assert!(
        !files.is_empty(),
        "expected something on disk under {root:?}"
    );
    for path in files {
        let raw = fs::read(&path).unwrap();
        for needle in needles {
            assert!(
                !raw.windows(needle.len()).any(|w| w == needle.as_bytes()),
                "{} leaked {needle:?} in plaintext",
                path.display()
            );
        }
    }
}

#[test]
fn transcripts_listings_and_memory_hold_no_plaintext() {
    let dir = TempDir::new().unwrap();
    let repo = sealed_repo(dir.path(), "tenant-a", &DEK);
    let mut store = repo.create(SessionMeta::with_id("s1", "/w", "m")).unwrap();
    store
        .append_new(&[Message::user("SECRET-TRANSCRIPT")])
        .unwrap();
    store.set_title("SECRET-TITLE").unwrap();
    drop(store);
    // The listing cache holds the title, the preview and up to 50 KiB of transcript text.
    let listed = repo.list().unwrap();
    assert_eq!(listed[0].title.as_deref(), Some("SECRET-TITLE"));
    assert_eq!(listed[0].preview.as_deref(), Some("SECRET-TRANSCRIPT"));
    assert!(dir.path().join(".listings.json").is_file());

    assert_no_plaintext(dir.path(), &["SECRET-TRANSCRIPT", "SECRET-TITLE"]);

    // And it still reads back correctly with the right key.
    let (_s, session) = repo.open_or_create_id("s1", "/w", "m").unwrap();
    assert_eq!(session.messages.len(), 1);
}

#[test]
fn a_base_seals_its_content_too() {
    // Compaction replaces the whole transcript at once. That content is still transcript.
    let dir = TempDir::new().unwrap();
    let repo = sealed_repo(dir.path(), "tenant-a", &DEK);
    let mut store = repo.create(SessionMeta::with_id("s1", "/w", "m")).unwrap();
    store
        .append_new(&[
            Message::user("SECRET-BEFORE"),
            Message::user("SECRET-ALSO-BEFORE"),
        ])
        .unwrap();
    store.rewrite(&[Message::user("SECRET-SUMMARY")]).unwrap();
    drop(store);

    assert_no_plaintext(
        dir.path(),
        &["SECRET-BEFORE", "SECRET-ALSO-BEFORE", "SECRET-SUMMARY"],
    );

    // Consolidation copies the view into a base; that copy must stay sealed as well.
    let mut history = vec![Message::user("SECRET-SUMMARY")];
    for i in 0..12 {
        history.push(Message::user(format!("SECRET-TURN-{i}")));
        let (mut store, _) = repo.open_or_create_id("s1", "/w", "m").unwrap();
        store.append_new(&history).unwrap();
    }
    let needles: Vec<String> = (0..12).map(|i| format!("SECRET-TURN-{i}")).collect();
    let needles: Vec<&str> = needles.iter().map(String::as_str).collect();
    assert_no_plaintext(dir.path(), &needles);

    let (_s, session) = repo.open_or_create_id("s1", "/w", "m").unwrap();
    assert_eq!(session.messages.len(), 13, "and it all still reads back");
}

#[test]
fn segment_framing_stays_plaintext_so_a_reader_can_bound_a_file_it_cannot_open() {
    let dir = TempDir::new().unwrap();
    let repo = sealed_repo(dir.path(), "tenant-a", &DEK);
    let mut store = repo.create(SessionMeta::with_id("s1", "/w", "m")).unwrap();
    store.append_new(&[Message::user("hello")]).unwrap();
    drop(store);
    let (mut store, _) = repo.open_or_create_id("s1", "/w", "m").unwrap();
    store
        .append_new(&[Message::user("hello"), Message::user("again")])
        .unwrap();
    drop(store);

    for (name, expect_base) in [("000001.jsonl", true), ("000002.jsonl", false)] {
        let raw = fs::read_to_string(dir.path().join("s1").join(name)).unwrap();
        let header: serde_json::Value = serde_json::from_str(raw.lines().next().unwrap()).unwrap();
        assert_eq!(header["type"], "segment");
        assert_eq!(header["base"], expect_base);
        // Every content line is framed, and none of them is JSON.
        for line in raw.lines().skip(1).filter(|l| !l.is_empty()) {
            if line == r#"{"type":"base_complete"}"# {
                continue;
            }
            assert!(
                line.starts_with("e1.") && line.contains(':'),
                "content line is not sealed: {line}"
            );
            assert!(!line.starts_with('{'), "a sealed line never starts with {{");
        }
    }
}

#[test]
fn the_wrong_key_fails_hard_rather_than_reading_an_empty_session() {
    let dir = TempDir::new().unwrap();
    let repo = sealed_repo(dir.path(), "tenant-a", &DEK);
    let mut store = repo.create(SessionMeta::with_id("s1", "/w", "m")).unwrap();
    store.append_new(&[Message::user("hello")]).unwrap();
    drop(store);

    let wrong_key = sealed_repo(dir.path(), "tenant-a", &[0x11; 32]);
    let Err(err) = wrong_key.open_id("s1") else {
        panic!("a reader with the wrong key must not open the session");
    };
    assert!(
        err.to_string().contains("failed to open"),
        "expected a decryption failure, got {err}"
    );

    // A silent empty read would be worse than an error: the next write would look like a brand-new
    // session and overwrite real history.
    let no_key = SessionStore::open(dir.path().join("s1"));
    assert!(no_key.is_err(), "an unsealed reader must not see through");
}

#[test]
fn the_wrong_tenant_or_the_wrong_session_id_fails_hard() {
    let dir = TempDir::new().unwrap();
    let repo = sealed_repo(dir.path(), "tenant-a", &DEK);
    let mut store = repo.create(SessionMeta::with_id("s1", "/w", "m")).unwrap();
    store.append_new(&[Message::user("hello")]).unwrap();
    drop(store);

    // Same DEK, different tenant: the AAD no longer matches.
    let other_tenant = sealed_repo(dir.path(), "tenant-b", &DEK);
    assert!(other_tenant.open_id("s1").is_err());

    // Same tenant and key, but the bytes were moved into another session's directory. The AAD is
    // built per opened id, so a transplanted transcript does not open.
    let moved = dir.path().join("s2");
    fs::create_dir_all(&moved).unwrap();
    fs::copy(
        dir.path().join("s1/000001.jsonl"),
        moved.join("000001.jsonl"),
    )
    .unwrap();
    let Err(err) = repo.open_id("s2") else {
        panic!("a transplanted transcript must not open under another id");
    };
    assert!(
        err.to_string().contains("failed to open"),
        "expected an AAD mismatch, got {err}"
    );
}

#[test]
fn a_listing_cache_sealed_for_another_tenant_is_ignored_not_trusted() {
    let dir = TempDir::new().unwrap();
    let repo = sealed_repo(dir.path(), "tenant-a", &DEK);
    let mut store = repo.create(SessionMeta::with_id("s1", "/w", "m")).unwrap();
    store.append_new(&[Message::user("hello")]).unwrap();
    drop(store);
    assert_eq!(repo.list().unwrap().len(), 1);

    // Someone drops a cache this tenant cannot open. The listing rebuilds from the transcripts
    // instead of serving — or failing on — a blob it can't authenticate.
    fs::write(dir.path().join(".listings.json"), b"e1.deadbeef:not-real").unwrap();
    let listed = repo.list().unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].preview.as_deref(), Some("hello"));
}

#[tokio::test]
async fn memory_documents_are_sealed_whole_file_and_survive_a_rename() {
    let dir = TempDir::new().unwrap();
    let codec = Arc::new(TenantCodec::new("tenant-a", &DEK));
    let mem_dir = dir.path().join("memory");
    let backend = FileBackend::at(mem_dir.clone()).sealed(codec.clone());

    let p = |s: &str| MemPath::parse(s).unwrap();
    backend
        .create(&p("/memories/notes.md"), "SECRET-MEMORY\n")
        .await
        .unwrap();
    backend
        .create(
            &p("/memories/MEMORY.md"),
            "- [notes](notes.md) SECRET-MEMORY\n",
        )
        .await
        .unwrap();

    assert_no_plaintext(&mem_dir, &["SECRET-MEMORY"]);

    // Reads, the index and search all go back through the codec.
    let View::Document(text) = backend.view(&p("/memories/notes.md"), None).await.unwrap() else {
        panic!("expected a document");
    };
    assert_eq!(text, "SECRET-MEMORY\n");
    assert!(backend.index().await.unwrap().contains("SECRET-MEMORY"));
    let hits = backend.search("secret-memory").await.unwrap();
    assert_eq!(hits.len(), 2);

    // A rename is a pure `rename(2)` — the AAD is the tenant, not the path — so the moved document
    // still opens.
    backend
        .rename(&p("/memories/notes.md"), &p("/memories/sub/moved.md"))
        .await
        .unwrap();
    let View::Document(text) = backend
        .view(&p("/memories/sub/moved.md"), None)
        .await
        .unwrap()
    else {
        panic!("expected a document");
    };
    assert_eq!(text, "SECRET-MEMORY\n");

    // A backend holding another tenant's key cannot read it.
    let stranger = FileBackend::at(mem_dir).sealed(Arc::new(TenantCodec::new("tenant-b", &DEK)));
    assert!(
        stranger
            .view(&p("/memories/sub/moved.md"), None)
            .await
            .is_err()
    );
    assert_eq!(
        stranger.index().await.unwrap(),
        "",
        "an index it cannot open reads as empty rather than as garbage"
    );
}

#[tokio::test]
async fn an_unsealed_memory_store_is_byte_for_byte_what_it_always_was() {
    let dir = TempDir::new().unwrap();
    let backend = FileBackend::at(dir.path().join("memory"));
    let p = MemPath::parse("/memories/notes.md").unwrap();
    backend.create(&p, "plain text\n").await.unwrap();
    assert_eq!(
        fs::read_to_string(dir.path().join("memory/notes.md")).unwrap(),
        "plain text\n"
    );
}
