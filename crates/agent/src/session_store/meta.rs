use serde::{Deserialize, Serialize};

use super::*;

/// On-disk format version, written into every header. The load path refuses any header whose version
/// is *newer* than this — a forward-compat guard so an older binary never silently mis-parses a shape
/// it doesn't understand — and treats `0` (a pre-versioning header, where the field was absent and
/// `serde` defaulted it) as v1-equivalent. The contract: additive, optional fields (each `#[serde(default)]`)
/// do **not** need a bump, since old and new files round-trip through each other; only an incompatible
/// reshaping of existing fields does. When that happens, bump this and add a [`migrate`] arm that
/// upgrades the older shape in place.
pub(super) const VERSION: u32 = 1;

/// How many characters of the first user message a listing preview keeps.
pub(super) const PREVIEW_MAX: usize = 80;

/// Stable identity + metadata for one session, persisted as the file's header line.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMeta {
    /// On-disk format version (see [`VERSION`]) — present so a future migration can branch on it.
    #[serde(default)]
    pub version: u32,
    pub id: String,
    /// Unix seconds at creation. Orders the repo listing.
    pub created_at: u64,
    /// Working directory the session was started in. Callers are expected to have passed this through
    /// [`canonical_cwd`] first, so it's a canonical (symlink-resolved, no trailing separator) path.
    pub cwd: String,
    /// Model the session runs against.
    pub model: String,
    /// Human-readable title, if set.
    #[serde(default)]
    pub title: Option<String>,
    /// The session this was forked from, if any.
    #[serde(default)]
    pub parent: Option<String>,
    /// Compaction provenance: how many times this session was rewritten with *fewer* messages than
    /// were on disk (i.e. a compaction replaced a prefix with a summary). Defaults to 0 for sessions
    /// that were never compacted, so older headers round-trip unchanged.
    #[serde(default)]
    pub compactions: u32,
    /// Total messages dropped across those compactions — a coarse measure of how much transcript the
    /// summaries stand in for. Defaults to 0.
    #[serde(default)]
    pub dropped_messages: u64,
    /// Branch-summarization provenance (Track L2), the same counter shape as `compactions`/
    /// `dropped_messages` above but for abandoned-branch summaries instead of compaction rounds: how
    /// many times [`SessionStore::switch_active_with_summary`] has run on this session. Defaults to 0.
    #[serde(default)]
    pub branch_summaries: u32,
    /// Total messages folded into those branch summaries — the branch-summary analog of
    /// `dropped_messages`. Defaults to 0.
    #[serde(default)]
    pub summarized_branch_messages: u64,
    /// The portable thinking level ([`agent_core::ThinkingLevel::as_str`]'s wire string, e.g. `"high"`)
    /// most recently recorded via [`SessionStore::record_thinking_level_change`] — `None` when no
    /// change was ever recorded (a session that never called `set_reasoning_effort`/
    /// `cycle_thinking_level`, or one written before this field existed). Task #18 (pi-parity fix):
    /// added alongside `model`'s own "keep meta current" fix for the same reason — a flat "last known"
    /// figure a simple consumer (a future `run --continue` reopen path) can read directly, without
    /// needing the full per-branch `SessionStore::thinking_level_at` tree lookup `switch_session`/
    /// `switch_branch` already use. `#[serde(default)]` so older headers (written before this field
    /// existed) round-trip unchanged.
    #[serde(default)]
    pub thinking_level: Option<String>,

    /// The exec endpoint this session's tools act on, if any — `{"command": …}` or
    /// `{"url": …, "headers": [...]}`, the same shape `set_exec_endpoint` accepts.
    ///
    /// Persisted so resuming a session reattaches it rather than silently falling back to the host,
    /// which for a sandboxed session would move a tenant's work onto the server. Additive and
    /// `#[serde(default)]`, so no format-version bump (see this module's format contract) and an old
    /// session file simply has no endpoint.
    ///
    /// **Not** copied by `fork`: a fork is a new session, and a sandbox belongs to the session it was
    /// provisioned for. Inheriting it would be the same cross-session reuse that clearing on switch
    /// exists to prevent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exec_endpoint: Option<serde_json::Value>,

    // --- Derived listing fields ---
    // Populated only by [`SessionRepo::list`] (from the file's mtime and a light scan), never persisted
    // — `#[serde(skip)]` keeps them out of the on-disk header so they can't go stale, and defaults them
    // to zero on a freshly created or opened session. They let a client sort by last-active and show a
    // preview without opening every transcript.
    /// Unix seconds of the file's last modification — when the session was last written to. `0` outside
    /// of a listing. Sort by this for "most recently active" ordering; `created_at` only orders by birth.
    #[serde(skip)]
    pub updated_at: u64,
    /// Number of message lines on disk (a torn final line is excluded, matching load semantics). `0`
    /// outside of a listing.
    #[serde(skip)]
    pub message_count: usize,
    /// The first user message's text, truncated to [`PREVIEW_MAX`] chars — `None` outside of a listing,
    /// or when the session has no user text yet (e.g. empty, or only tool-result turns so far).
    #[serde(skip)]
    pub preview: Option<String>,
}

impl SessionMeta {
    /// Fresh metadata with a generated id and the current timestamp.
    pub fn new(cwd: impl Into<String>, model: impl Into<String>) -> Self {
        Self::with_id(new_id(), cwd, model)
    }

    /// Fresh metadata with a caller-supplied id, for a client that wants a deterministic session id
    /// rather than the generated one. No collision check here: [`SessionStore::create`]'s
    /// `create_new(true)` already fails loudly (`AlreadyExists`) if the id's derived path already
    /// exists, rather than silently clobbering another session.
    pub fn with_id(
        id: impl Into<String>,
        cwd: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            version: VERSION,
            id: id.into(),
            created_at: now_secs(),
            cwd: cwd.into(),
            model: model.into(),
            title: None,
            parent: None,
            compactions: 0,
            dropped_messages: 0,
            branch_summaries: 0,
            summarized_branch_messages: 0,
            thinking_level: None,
            exec_endpoint: None,
            updated_at: 0,
            message_count: 0,
            preview: None,
        }
    }

    /// The listing view as JSON: every persisted field, plus the derived-only fields
    /// (`updated_at`/`message_count`/`preview`) that `#[serde(skip)]` deliberately keeps out of the
    /// on-disk header — so a stale scan can never leak into it — but that a client browsing a
    /// listing needs to see. Use this instead of `serde_json::to_value` when serializing a listing
    /// entry for an RPC response.
    ///
    /// **Not** the session's text. Matching is done here (see [`search_sessions`]), so a client has
    /// never needed the corpus to filter locally — and returning it made a listing's size the
    /// tenant's whole transcript set rather than a page of metadata.
    pub fn to_listing_json(&self) -> serde_json::Value {
        let mut v = serde_json::to_value(self).unwrap_or(serde_json::Value::Null);
        if let serde_json::Value::Object(map) = &mut v {
            map.insert("updated_at".into(), serde_json::json!(self.updated_at));
            map.insert(
                "message_count".into(),
                serde_json::json!(self.message_count),
            );
            map.insert("preview".into(), serde_json::json!(self.preview));
        }
        v
    }
}

/// One session's best match against a lowercased, already-trimmed `query`, for ranking search results —
/// `(field_priority, byte_offset)`, ascending order is "more relevant" (a match in a higher-priority
/// field beats any match in a lower one; within the same field, an earlier match beats a later one).
/// Checked in priority order — `title`, `id`, `preview`, `cwd` — and stops at the first field that
/// matches, so a title hit is never outranked by a coincidental hit in a lower one. `None` when
/// `query` doesn't appear in any field at all (the session doesn't match).
///
/// **Metadata only.** This used to scan a per-session corpus of the conversation itself; that corpus
/// was up to 50 KB per session, was cached on the shard, was returned to the client on every listing
/// entry, and was silently truncated after roughly the first fifty turns — so it answered "is this
/// word in the first part of this session" while reading as "search". Content search, if it is ever
/// wanted, belongs somewhere that can do it properly; see `crates/agent/FLEET.md`.
///
/// Deliberately simpler than pi's own TUI session-picker scorer (`fuzzyMatch` in
/// `packages/tui/src/fuzzy.ts`): that one is a fuzzy subsequence matcher with a hand-tuned heuristic
/// (consecutive-run bonus, word-boundary bonus, gap penalty) built for a human eyeballing highlighted
/// matches as they type. A scripting RPC/CLI caller — the only consumer here, since Beyond has no
/// interactive TUI of its own — wants predictable "does this text contain the query" filtering instead
/// of typo-tolerant fuzzy scoring, so this is a plain case-insensitive substring search.
pub(super) fn search_rank(meta: &SessionMeta, query_lower: &str) -> Option<(usize, usize)> {
    let fields: [&str; 4] = [
        meta.title.as_deref().unwrap_or(""),
        &meta.id,
        meta.preview.as_deref().unwrap_or(""),
        &meta.cwd,
    ];
    fields.iter().enumerate().find_map(|(priority, field)| {
        find_case_insensitive(field, query_lower).map(|offset| (priority, offset))
    })
}

/// Byte offset of the first case-insensitive occurrence of `needle_lower` (already lowercased by the
/// caller) in `haystack`, or `None` — the allocation-free replacement for
/// `haystack.to_lowercase().find(needle_lower)`.
///
/// The old expression allocated a fully-lowercased copy of *every* field for every session on every
/// query, ran `find` on it, and threw it away. This scans in place. The common case (an all-ASCII haystack and needle, which session text and a typed query
/// overwhelmingly are) takes a `memchr`-accelerated scan with no allocation at all. Any non-ASCII in
/// either falls back to the exact original expression, so the match set *and* the returned byte offset
/// stay
/// byte-for-byte identical to before — the locale-specific corners of `str::to_lowercase` (final
/// sigma, length-changing folds) are preserved verbatim rather than reimplemented.
pub(super) fn find_case_insensitive(haystack: &str, needle_lower: &str) -> Option<usize> {
    if needle_lower.is_empty() {
        // `"".find("")` is `Some(0)`; match it (the caller never passes an empty query, but keep the
        // helper total).
        return Some(0);
    }
    if haystack.is_ascii() && needle_lower.is_ascii() {
        return find_ascii_case_insensitive(haystack, needle_lower);
    }
    haystack.to_lowercase().find(needle_lower)
}

/// `haystack.find(needle)` under ASCII case folding, allocation-free. `needle` is already ASCII
/// lowercase (from the caller's one-time `to_lowercase`); both strings are known ASCII, so every byte
/// is its own char and ASCII lowercasing never changes byte length — the returned offset therefore
/// equals the one `haystack.to_lowercase().find(needle)` would give.
///
/// The shape mirrors how `str::find(&str)` itself works, so it keeps that path's speed while dropping
/// its 50 KB allocation: skip to the next position whose byte could start a match with `str::find` on
/// a single char — which is SIMD/`memchr`-accelerated in the standard library — trying both cases of
/// the needle's first byte, then verify the (short) remainder there. A non-matching query, the worst
/// case, is thus one or two vectorized passes over the field and no allocation at all, rather than
/// the scalar per-byte scan a hand-rolled loop would be.
pub(super) fn find_ascii_case_insensitive(haystack: &str, needle: &str) -> Option<usize> {
    let hb = haystack.as_bytes();
    let nb = needle.as_bytes();
    if nb.len() > hb.len() {
        return None;
    }
    let first_lower = nb[0] as char;
    let first_upper = nb[0].to_ascii_uppercase() as char;
    let mut base = 0;
    while base + nb.len() <= hb.len() {
        let rest = &haystack[base..];
        // Earliest position (from `base`) whose byte is the needle's first byte in either case —
        // `str::find(char)` reaches for `memchr` for an ASCII char, so this is a vectorized skip.
        let cand = if first_upper == first_lower {
            rest.find(first_lower)?
        } else {
            match (rest.find(first_lower), rest.find(first_upper)) {
                (Some(a), Some(b)) => a.min(b),
                (Some(a), None) | (None, Some(a)) => a,
                (None, None) => return None,
            }
        };
        let start = base + cand;
        if start + nb.len() > hb.len() {
            return None;
        }
        if hb[start..start + nb.len()]
            .iter()
            .zip(nb)
            .all(|(h, n)| h.eq_ignore_ascii_case(n))
        {
            return Some(start);
        }
        base = start + 1;
    }
    None
}

/// Filter `sessions` to those matching `query` (case-insensitive substring against `title`/`id`/
/// `preview`/`cwd` — see [`search_rank`]), sorted best-match-first with a most-recently-
/// active tiebreak. `query: None` (or empty/whitespace-only) returns `sessions` unchanged, in whatever
/// order the caller already sorted them (recency, from [`SessionRepo::list`]/[`SessionRepo::list_all`]) —
/// so an absent query is a true no-op, not just "matches everything."
pub fn search_sessions(sessions: Vec<SessionMeta>, query: Option<&str>) -> Vec<SessionMeta> {
    let Some(query) = query.map(str::trim).filter(|q| !q.is_empty()) else {
        return sessions;
    };
    let query_lower = query.to_lowercase();
    let mut ranked: Vec<(SessionMeta, (usize, usize))> = sessions
        .into_iter()
        .filter_map(|m| {
            let rank = search_rank(&m, &query_lower)?;
            Some((m, rank))
        })
        .collect();
    ranked.sort_by(|(a_meta, a_rank), (b_meta, b_rank)| {
        a_rank.cmp(b_rank).then_with(|| by_recency(a_meta, b_meta))
    });
    ranked.into_iter().map(|(m, _)| m).collect()
}

/// Custom-entry kinds the agent writes itself, and acts on when a session is replayed: an MCP task to
/// resume, a resumed task's result and a "result pending" placeholder's mark (`crate::mcp_resume`),
/// a local `bash` job's processes to stop if a killed process left them running
/// (`crate::tools::bash_jobs::PROCS_ENTRY_KIND`),
/// and `mcp_skill_approval`, which is no
/// longer written (an approval read back from disk is one a model with file-write tools could have
/// planted, so approvals are never persisted) but stays reserved so no client can make an older
/// reader trust one. A client's `append_custom` of any of these is refused. A new host-written kind
/// belongs here.
pub const HOST_CUSTOM_KINDS: &[&str] = &[
    crate::mcp_resume::TASK_ENTRY_KIND,
    crate::mcp_resume::RESULT_ENTRY_KIND,
    crate::mcp_resume::PLACEHOLDER_ENTRY_KIND,
    "mcp_skill_approval",
    crate::tools::bash_jobs::PROCS_ENTRY_KIND,
];

/// Whether `kind` is one of [`HOST_CUSTOM_KINDS`].
pub fn is_host_custom_kind(kind: &str) -> bool {
    HOST_CUSTOM_KINDS.contains(&kind)
}

/// Most recently active first; ties (same second — `updated_at` has one-second resolution, so a
/// burst of sessions shares one) broken by id, descending, so a listing has one order however the
/// directory happened to be scanned. Without the tie-break, `offset` paging over sessions written in
/// the same second could serve one session twice and skip another. Descending, because a generated
/// id ([`new_id`]) leads with its creation time in fixed-width hex nanoseconds: within the second the
/// newest still comes first. (A client-chosen id has no such meaning, but still one stable order.)
pub(crate) fn by_recency(a: &SessionMeta, b: &SessionMeta) -> std::cmp::Ordering {
    b.updated_at
        .cmp(&a.updated_at)
        .then_with(|| b.id.cmp(&a.id))
}

/// Collapse any run of `\r`/`\n` into a single space and trim, matching pi's `appendSessionInfo`
/// (`name.replace(/[\r\n]+/g, " ").trim()`) — a title is a single display line (session lists, HTML
/// export headers), so embedded newlines would otherwise corrupt that rendering.
pub(super) fn sanitize_title(title: &str) -> String {
    let mut out = String::with_capacity(title.len());
    let mut last_was_newline_run = false;
    for c in title.chars() {
        if c == '\r' || c == '\n' {
            if !last_was_newline_run {
                out.push(' ');
            }
            last_was_newline_run = true;
        } else {
            out.push(c);
            last_was_newline_run = false;
        }
    }
    out.trim().to_string()
}

/// A title that sanitizes to empty explicitly clears the session's title, matching pi's
/// `getSessionName` ("empty names explicitly clear the session title") rather than persisting a blank
/// string that would display as a present-but-empty title.
pub(super) fn title_or_clear(title: String) -> Option<String> {
    if title.is_empty() { None } else { Some(title) }
}
