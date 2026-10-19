use super::*;

#[test]
fn ids_are_unique_and_shaped() {
    // All ids in a batch are distinct (the per-process salt + monotonic seq guarantee it within a
    // process; the salt guards across processes), and the shape is `<hex>-<hex>-<hex>`.
    let ids: std::collections::HashSet<String> = (0..10_000).map(|_| new_id()).collect();
    assert_eq!(ids.len(), 10_000);

    let id = new_id();
    assert_eq!(id.matches('-').count(), 2);
    assert!(id.chars().all(|c| c.is_ascii_hexdigit() || c == '-'));
}

#[test]
fn session_meta_with_id_uses_the_given_id_not_new_id() {
    let meta = SessionMeta::with_id("my-custom-id", "/w", "m");
    assert_eq!(meta.id, "my-custom-id");

    // It round-trips through the repo like any other session.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let store = repo.create(meta).unwrap();
    assert_eq!(store.meta().id, "my-custom-id");
    let (reopened, _session) = repo.open_id("my-custom-id").unwrap();
    assert_eq!(reopened.meta().id, "my-custom-id");
}

#[test]
fn list_reports_updated_count_and_preview() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let id = store.meta().id.clone();
    let mut session = Session::new();
    session.user("hello world, this is the first message");
    session.user("second");
    store.append_new(&session.messages).unwrap();

    let listings = repo.list().unwrap();
    let l = listings.iter().find(|l| l.id == id).unwrap();
    assert_eq!(l.message_count, 2);
    assert_eq!(
        l.preview.as_deref(),
        Some("hello world, this is the first message")
    );
    assert!(l.updated_at > 0);
}

#[test]
fn updated_at_prefers_message_timestamp_over_a_stale_file_mtime() {
    // A copy/restore/sync that doesn't preserve mtime exactly (or one that's simply wrong) must
    // not make a session look stale, or falsely fresh, in a listing — the content itself carries
    // the real signal now.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let id = store.meta().id.clone();
    let path = repo.path_for(store.meta());
    let mut session = Session::new();
    session.user("hello");
    store.append_new(&session.messages).unwrap();

    // Stomp the file's mtime to something obviously wrong (year-2000-ish) — if `read_listing`
    // were still trusting mtime, this would leak straight through into `updated_at`.
    let bogus = std::time::UNIX_EPOCH + std::time::Duration::from_secs(946_684_800);
    let file = fs::File::options().write(true).open(&path).unwrap();
    file.set_modified(bogus).unwrap();
    assert_eq!(mtime_secs(&path), 946_684_800, "mtime stomp didn't take");

    let listings = repo.list().unwrap();
    let l = listings.iter().find(|l| l.id == id).unwrap();
    assert_ne!(
        l.updated_at, 946_684_800,
        "updated_at must not come from the (deliberately wrong) file mtime"
    );
    assert!(
        l.updated_at >= now_secs().saturating_sub(60),
        "updated_at should reflect the message's own recent timestamp, got {}",
        l.updated_at
    );
}

#[test]
fn updated_at_falls_back_to_mtime_for_a_legacy_file_with_no_stamped_message() {
    // A file written before this field existed carries no `timestamp` on its `Entry::Message`
    // lines at all (not even a `0` — the key is simply absent, same as `id`/`parent_id` on a
    // pre-tree file). `#[serde(default)]` reads that as `0` ("unknown"), and `read_listing` must
    // fall back to mtime exactly as if the feature didn't exist for this file.
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let meta = SessionMeta::new("/w", "m");
    let id = meta.id.clone();
    let path = repo.path_for(&meta);
    let lines = [
        serde_json::to_string(&Entry::Session(meta)).unwrap(),
        "{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"hi\"}]}"
            .to_string(),
    ];
    fs::write(&path, lines.join("\n") + "\n").unwrap();

    let known_mtime = std::time::UNIX_EPOCH + std::time::Duration::from_secs(946_684_800);
    let file = fs::File::options().write(true).open(&path).unwrap();
    file.set_modified(known_mtime).unwrap();

    let listings = repo.list().unwrap();
    let l = listings.iter().find(|l| l.id == id).unwrap();
    assert_eq!(
        l.updated_at, 946_684_800,
        "with no stamped message, updated_at must fall back to file mtime"
    );
}

#[test]
fn to_listing_json_surfaces_derived_fields_that_serde_skip_hides() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("hello world");
    store.append_new(&session.messages).unwrap();

    let listing = repo.list().unwrap().into_iter().next().unwrap();
    // The plain derive hides these — that's the bug `to_listing_json` exists to work around.
    let bare = serde_json::to_value(&listing).unwrap();
    assert!(bare.get("updated_at").is_none());
    assert!(bare.get("message_count").is_none());
    assert!(bare.get("preview").is_none());

    let full = listing.to_listing_json();
    assert_eq!(full["message_count"], 1);
    assert_eq!(full["preview"], "hello world");
    assert!(full["updated_at"].as_u64().unwrap() > 0);
    // Persisted fields still round-trip through the same call.
    assert_eq!(full["id"], listing.id);
    assert_eq!(full["cwd"], "/w");
}

/// `body` lands in `preview` — a real field `search_rank` checks, and one ranked *below*
/// `title`, which is the ordering every test below turns on.
fn meta_for_search(id: &str, title: Option<&str>, body: &str, updated_at: u64) -> SessionMeta {
    let mut m = SessionMeta::with_id(id, "/w", "m");
    m.title = title.map(str::to_string);
    m.preview = Some(body.to_string());
    m.updated_at = updated_at;
    m
}

/// Sessions active in the same second (one-second `updated_at`) still have one order, and it is
/// newest-first: a generated id leads with its creation time, so the later one sorts first. With
/// no tie-break (or an ascending one) `offset` paging over them could repeat or skip a session.
#[test]
fn same_second_sessions_list_newest_first_in_one_stable_order() {
    let ids: Vec<String> = (0..4).map(|_| new_id()).collect();
    let mut metas: Vec<SessionMeta> = ids
        .iter()
        .map(|id| meta_for_search(id, None, "x", 100))
        .collect();
    metas.push(meta_for_search("older", None, "x", 99));
    // Oldest first in, whatever the scan order was.
    metas.sort_by(by_recency);
    let got: Vec<&str> = metas.iter().map(|m| m.id.as_str()).collect();
    let mut want: Vec<&str> = ids.iter().rev().map(String::as_str).collect();
    want.push("older");
    assert_eq!(got, want);
    // The same order from any starting order.
    metas.reverse();
    metas.sort_by(by_recency);
    assert_eq!(
        metas.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
        want
    );
}

#[test]
fn search_sessions_with_no_query_returns_the_input_unchanged_and_in_order() {
    let sessions = vec![
        meta_for_search("a", None, "alpha", 2),
        meta_for_search("b", None, "beta", 1),
    ];
    let out = search_sessions(sessions.clone(), None);
    assert_eq!(
        out.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
        vec!["a", "b"],
        "no query must be a true no-op, not even a re-sort"
    );

    // Empty/whitespace-only queries are treated the same as no query at all.
    let out = search_sessions(sessions.clone(), Some(""));
    assert_eq!(out.len(), 2);
    let out = search_sessions(sessions, Some("   "));
    assert_eq!(out.len(), 2);
}

#[test]
fn search_sessions_filters_out_non_matching_sessions() {
    let sessions = vec![
        meta_for_search("a", None, "discusses widget-frobnication", 1),
        meta_for_search("b", None, "a totally unrelated topic", 2),
    ];
    let out = search_sessions(sessions, Some("widget-frobnication"));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].id, "a");
}

#[test]
fn search_sessions_matching_nothing_returns_empty_not_everything() {
    let sessions = vec![meta_for_search("a", None, "alpha", 1)];
    let out = search_sessions(sessions, Some("no-such-term"));
    assert!(out.is_empty());
}

#[test]
fn search_sessions_is_case_insensitive() {
    let sessions = vec![meta_for_search("a", None, "Zephyr Marker", 1)];
    let out = search_sessions(sessions, Some("zephyr marker"));
    assert_eq!(out.len(), 1);
}

#[test]
fn search_sessions_ranks_a_title_match_above_a_body_only_match() {
    let sessions = vec![
        meta_for_search(
            "text-only",
            None,
            "mentions rustacean deep in the transcript",
            5,
        ),
        meta_for_search(
            "title-match",
            Some("rustacean project"),
            "unrelated body",
            1,
        ),
    ];
    let out = search_sessions(sessions, Some("rustacean"));
    assert_eq!(
        out.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
        vec!["title-match", "text-only"],
        "a title hit must outrank a lower-field hit even though it's less recent"
    );
}

#[test]
fn search_sessions_breaks_ties_by_most_recently_active() {
    let sessions = vec![
        meta_for_search("older", None, "shared-term here", 1),
        meta_for_search("newer", None, "shared-term here too", 2),
    ];
    let out = search_sessions(sessions, Some("shared-term"));
    assert_eq!(
        out.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
        vec!["newer", "older"],
        "equally-ranked matches must tiebreak by recency, newest first"
    );
}

#[test]
fn find_case_insensitive_matches_the_lowercase_find_reference() {
    // The allocation-free matcher must be byte-for-byte indistinguishable from the expression it
    // replaced — `haystack.to_lowercase().find(needle_lower)` — in both the match/no-match verdict
    // and the returned byte offset, across ASCII case-mixing, leftmost-match selection, needle-
    // longer-than-haystack, end-of-string hits, and the non-ASCII fallback.
    let cases = [
        ("Hello World", "world"),
        ("Hello World", "world"), // (query pre-lowered below)
        ("aAaAbB", "ab"),         // leftmost case-insensitive match, many false starts
        ("abcabc", "cab"),
        ("no match here", "xyz"),
        ("", "x"),
        ("x", "xx"), // needle longer than haystack
        ("MiXeDcAsE", "mixedcase"),
        ("ends with Z", "z"),
        ("ZZZfooZZZ", "foo"),
        ("café society", "society"), // non-ASCII haystack → fallback
        ("Straße", "strasse"),       // eszett does not fold to ss → must NOT match, via fallback
        ("naïve", "naïve"),          // non-ASCII query → fallback (query lowered below)
    ];
    for (haystack, query) in cases {
        let needle_lower = query.to_lowercase();
        let expected = haystack.to_lowercase().find(&needle_lower);
        let got = find_case_insensitive(haystack, &needle_lower);
        assert_eq!(
            got, expected,
            "find_case_insensitive({haystack:?}, {needle_lower:?})"
        );
    }
}

#[test]
fn preview_truncates_long_first_message() {
    let dir = tmpdir();
    let repo = SessionRepo::open(dir.path()).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
    let mut session = Session::new();
    session.user("x".repeat(PREVIEW_MAX + 50));
    store.append_new(&session.messages).unwrap();

    let listings = repo.list().unwrap();
    let preview = listings[0].preview.as_deref().unwrap();
    // PREVIEW_MAX chars plus the ellipsis marker.
    assert_eq!(preview.chars().count(), PREVIEW_MAX + 1);
    assert!(preview.ends_with('…'));
}
