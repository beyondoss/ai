// Test target: `.unwrap()` asserts preconditions; that's the point.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! The sidecar listing index (`.listings.json`) is a pure cache in front of `read_listing`, which is
//! O(total transcript bytes) and so the one cost in this crate that grows without bound with a user's
//! history. Warm, a listing is one `stat` per file instead of a full parse.
//!
//! A cache that can serve a stale answer is worse than no cache, so these tests are about exactly one
//! question: **can the index ever disagree with a from-scratch scan?** Every case below therefore
//! asserts the cached result equals the uncached one — never merely that it "looks right".

use std::path::Path;

use agent_core::Message;
use beyond_ai_agent::session_store::{Layout, RepoOptions, SessionMeta, SessionRepo};
use tempfile::TempDir;

const INDEX: &str = ".listings.json";

fn repo() -> (TempDir, SessionRepo) {
    let dir = tempfile::tempdir().unwrap();
    let repo = SessionRepo::open(dir.path()).unwrap();
    (dir, repo)
}

/// A listing computed with the index removed — the ground truth every cached answer is compared to.
fn uncached(dir: &Path, repo: &SessionRepo) -> Vec<SessionMeta> {
    let _ = std::fs::remove_file(dir.join(INDEX));
    repo.list().unwrap()
}

/// Compares the two listings as *sets*, keyed by id. Deliberately not order-sensitive: `list` sorts by
/// `updated_at`, and sessions written within the same second tie — `sort_by` is stable, so a tie is
/// broken by scan order, which is already arbitrary (the uncached scan is a parallel fan-in). Asserting
/// on that order would be asserting on nondeterminism that predates the index.
fn assert_same(a: &[SessionMeta], b: &[SessionMeta]) {
    assert_eq!(a.len(), b.len(), "different session counts");
    let mut a = a.to_vec();
    let mut b = b.to_vec();
    a.sort_by(|x, y| x.id.cmp(&y.id));
    b.sort_by(|x, y| x.id.cmp(&y.id));
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.id, y.id);
        assert_eq!(
            x.updated_at, y.updated_at,
            "updated_at diverged for {}",
            x.id
        );
        assert_eq!(
            x.message_count, y.message_count,
            "message_count diverged for {}",
            x.id
        );
        assert_eq!(x.preview, y.preview, "preview diverged for {}", x.id);
        assert_eq!(x.title, y.title, "title diverged for {}", x.id);
        assert_eq!(x.model, y.model, "model diverged for {}", x.id);
    }
}

#[test]
fn a_warm_listing_matches_a_cold_one() {
    let (dir, repo) = repo();
    for i in 0..3 {
        let mut store = repo
            .create(SessionMeta::new("/repo", "claude-sonnet-5"))
            .unwrap();
        store
            .append_new(&[
                Message::user(format!("question number {i}")),
                Message::assistant(vec![agent_core::ContentBlock::Text {
                    text: format!("answer number {i}").into(),
                    id: None,
                    phase: None,
                }]),
            ])
            .unwrap();
    }

    let cold = uncached(dir.path(), &repo);
    let built = repo.list().unwrap(); // writes the index
    assert!(dir.path().join(INDEX).exists(), "index was never written");
    let warm = repo.list().unwrap(); // served from it

    assert_same(&cold, &built);
    assert_same(&cold, &warm);
}

/// The invalidation that actually matters: a session that is appended to must not keep serving the
/// listing it had *before* the append. Size and mtime both move on an append, and either alone is
/// enough to miss.
#[test]
fn appending_to_a_session_invalidates_its_cached_listing() {
    let (dir, repo) = repo();
    let mut store = repo
        .create(SessionMeta::new("/repo", "claude-sonnet-5"))
        .unwrap();
    let mut messages = vec![Message::user("first")];
    store.append_new(&messages).unwrap();

    let before = repo.list().unwrap(); // warms the index
    assert_eq!(before[0].message_count, 1);

    messages.push(Message::assistant(vec![agent_core::ContentBlock::Text {
        text: "second".into(),
        id: None,
        phase: None,
    }]));
    messages.push(Message::user("third"));
    store.append_new(&messages).unwrap();

    let after = repo.list().unwrap();
    assert_eq!(
        after[0].message_count, 3,
        "the index served a listing from before the append — stale cache"
    );
    assert_same(&uncached(dir.path(), &repo), &repo.list().unwrap());
}

/// A truncated/garbage index must be ignored, not trusted and not fatal.
#[test]
fn a_corrupt_index_falls_back_to_a_full_scan() {
    let (dir, repo) = repo();
    let mut store = repo
        .create(SessionMeta::new("/repo", "claude-sonnet-5"))
        .unwrap();
    store.append_new(&[Message::user("hello")]).unwrap();
    let expected = repo.list().unwrap();

    for garbage in [
        "",
        "{",
        "null",
        r#"{"version":999,"entries":{}}"#,
        "not json at all",
    ] {
        std::fs::write(dir.path().join(INDEX), garbage).unwrap();
        let got = repo.list().unwrap();
        assert_same(&expected, &got);
    }
}

/// An index written by an older build with different derived-field semantics must be discarded
/// wholesale rather than half-trusted — that's what the version field is for.
#[test]
fn an_index_at_an_unknown_version_is_discarded() {
    let (dir, repo) = repo();
    let mut store = repo
        .create(SessionMeta::new("/repo", "claude-sonnet-5"))
        .unwrap();
    store.append_new(&[Message::user("hello")]).unwrap();
    let expected = repo.list().unwrap();

    // A well-formed index, correct in every way except that it claims a version we don't speak — and
    // whose payload is deliberately wrong, so trusting it would be visible. Derived from whatever
    // version this build writes rather than a literal, so the assertion survives a version bump.
    let raw = std::fs::read_to_string(dir.path().join(INDEX)).unwrap();
    let mut index: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let version = index["version"]
        .as_u64()
        .expect("the index carries a version");
    index["version"] = serde_json::json!(version + 1);
    std::fs::write(dir.path().join(INDEX), index.to_string()).unwrap();

    assert_same(&expected, &repo.list().unwrap());
}

/// The segmented layout's stamp is the **newest segment's** `(len, mtime, epoch)` plus the segment
/// count — cheap enough to run on every candidate path of every listing, warm or cold, which is what
/// it has to be: it is the validator, so a warm 100%-hit listing pays it in full. (It used to open
/// and header-parse every segment of every session just to find the base epoch.)
///
/// So this walks a segmented session through every shape of change the stamp has to notice, and after
/// each one asserts the cached listing still equals a from-scratch scan.
#[test]
fn a_segmented_listing_is_invalidated_by_an_append_a_roll_and_a_base() {
    let dir = tempfile::tempdir().unwrap();
    let repo = SessionRepo::open_with(
        dir.path(),
        RepoOptions {
            layout: Layout::Segmented { codec: None },
            id_prefix: None,
        },
    )
    .unwrap();

    let mut store = repo
        .create(SessionMeta::with_id("s1", "/repo", "claude-sonnet-5"))
        .unwrap();
    let session_dir = store.path().to_path_buf();
    // Warms the index. The sidecar sits beside the session *directories* it describes, one per
    // directory that holds sessions — asserted here so the rest of this test cannot pass by
    // comparing two uncached scans.
    assert_eq!(repo.list().unwrap().len(), 1);
    assert!(
        dir.path().join(INDEX).is_file(),
        "the segmented layout must write a sidecar index too"
    );

    // An append grows the newest segment.
    store.append_new(&[Message::user("hello")]).unwrap();
    assert_same(&uncached_in(dir.path(), &repo), &repo.list().unwrap());

    // A roll starts a new segment: a different newest epoch, and one more of them.
    drop(store);
    let (mut store, _) = repo
        .open_or_create_id("s1", "/repo", "claude-sonnet-5")
        .unwrap();
    store
        .append_new(&[Message::user("hello"), Message::user("again")])
        .unwrap();
    assert_same(&uncached_in(dir.path(), &repo), &repo.list().unwrap());

    // A base rewrite — the change the dropped `base_epoch` field used to be carried for. A base is
    // always a *new* segment (`pred + 1`, `create_new`), so the newest epoch moves and the stamp with
    // it; a listing served from the pre-base cache here would be the regression.
    store.rewrite(&[Message::user("compacted")]).unwrap();
    let warm = repo.list().unwrap();
    assert_same(&uncached_in(dir.path(), &repo), &warm);
    assert_eq!(warm.len(), 1);
    assert_eq!(warm[0].message_count, 1, "the base is what is listed now");

    // And a prune, the one change that shrinks rather than grows — caught by the count alone.
    drop(store);
    std::fs::remove_file(session_dir.join("000001.jsonl")).unwrap();
    assert_same(&uncached_in(dir.path(), &repo), &repo.list().unwrap());
}

/// [`uncached`] for a repo whose sidecar lives inside each session's own directory rather than beside
/// the `.jsonl` files — the segmented layout keeps one index per session-holding directory.
fn uncached_in(root: &Path, repo: &SessionRepo) -> Vec<SessionMeta> {
    for entry in std::fs::read_dir(root).unwrap().flatten() {
        let _ = std::fs::remove_file(entry.path().join(INDEX));
    }
    let _ = std::fs::remove_file(root.join(INDEX));
    repo.list().unwrap()
}

/// A deleted session must fall out of the index rather than lingering in it forever.
#[test]
fn a_deleted_session_is_dropped_from_the_index() {
    let (dir, repo) = repo();
    let keep = repo
        .create(SessionMeta::new("/repo", "claude-sonnet-5"))
        .unwrap();
    let mut keep_store = keep;
    keep_store.append_new(&[Message::user("keep me")]).unwrap();

    let mut gone = repo
        .create(SessionMeta::new("/repo", "claude-sonnet-5"))
        .unwrap();
    gone.append_new(&[Message::user("delete me")]).unwrap();
    let gone_id = gone.meta().id.clone();
    let gone_path = gone.path().to_path_buf();

    assert_eq!(repo.list().unwrap().len(), 2); // warms the index with both
    drop(gone);
    std::fs::remove_file(&gone_path).unwrap();

    let after = repo.list().unwrap();
    assert_eq!(after.len(), 1, "a deleted session survived in the listing");
    assert!(!after.iter().any(|m| m.id == gone_id));

    let raw = std::fs::read_to_string(dir.path().join(INDEX)).unwrap();
    assert!(
        !raw.contains(&gone_id),
        "the deleted session's entry is still in the index, which would grow without bound"
    );
}

/// The index is a cache, so a directory it cannot be written into must still list correctly — just
/// without the speedup. A listing that fails because it couldn't save a cache would be a regression
/// invented by the optimization itself.
#[cfg(unix)]
#[test]
fn a_read_only_directory_still_lists() {
    use std::os::unix::fs::PermissionsExt;

    let (dir, repo) = repo();
    let mut store = repo
        .create(SessionMeta::new("/repo", "claude-sonnet-5"))
        .unwrap();
    store.append_new(&[Message::user("hello")]).unwrap();
    let expected = uncached(dir.path(), &repo);

    let perms = std::fs::metadata(dir.path()).unwrap().permissions();
    let mut ro = perms.clone();
    ro.set_mode(0o500); // r-x: readable, not writable
    std::fs::set_permissions(dir.path(), ro).unwrap();

    let got = repo.list();

    // Restore before asserting, so a failure can't leave an undeletable temp dir behind.
    std::fs::set_permissions(dir.path(), perms).unwrap();

    assert_same(
        &expected,
        &got.expect("listing must not fail just because the cache can't be saved"),
    );
}
