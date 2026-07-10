//! JSONL session persistence: a per-session append log and a multi-session repository.
//!
//! Each session is one newline-delimited JSON file: a header line carrying [`SessionMeta`] (stable id,
//! cwd, model, timestamps, optional title and `parent` lineage), then one line per conversation
//! message. A turn **appends** its new messages — O(new), not O(transcript) — and a torn final line
//! (a crash mid-append) is dropped on load, so at most the last entry is lost. More generally, any
//! single line that fails to deserialize — not only the last one — is skipped rather than truncating
//! the scan: reading resumes with whatever comes after it, since a fully-read line's boundaries are
//! already known regardless of whether its *contents* parse.
//!
//! **Tree-shaped history.** Every message entry carries an `id` and a `parent_id`, so a session's
//! history is a tree, not just a line — the "active path" (what `Session.messages` holds) is the
//! chain from the root to the active tip. In the common case (no branching) that's just every message
//! in file order, identical to a flat file. A `Leaf` entry marks a navigation event — the active tip
//! moved to some other message's id — so a later `open()` can tell the active branch apart from "the
//! last line in the file" once one exists (see [`SessionStore::switch_active`]). Branching is
//! append-only: navigating never deletes anything.
//!
//! `agent_core::Message`/`Session` stay flat and untouched — the tree lives entirely at this storage
//! layer, in the `id`/`parent_id` this module stamps onto each [`Entry::Message`] line.
//!
//! **Compaction vs. branching.** [`SessionStore::rewrite`] (used by compaction, which replaces the
//! active path's messages with a shorter summarized prefix) is still destructive **to the active
//! path's own entries** — the prior flat-history simplification stands there. But it now preserves
//! every entry that belongs to some *other* branch (created by navigating away and appending
//! elsewhere): those are re-written verbatim alongside the new, freshly-compacted active path, rather
//! than being discarded along with it. Net effect: "everything not kept is deleted" becomes
//! "everything not on the compacted active path is deleted; other branches persist" — a real, but
//! bounded and honest, weakening of the original simplification, not a full reversal of it. One known
//! edge case: an *off-branch* entry whose `parent_id` points at an active-path message that compaction
//! just summarized away is not deleted (its content survives on disk), but its link to the root is
//! severed — orphaned, not lost. No feature reconnects it today; a future branch-listing feature
//! should treat an unreachable `parent_id` as a second kind of root.
//!
//! **Legacy migration.** A file written before this module tracked ids has `id`/`parent_id` absent on
//! every message line (`#[serde(default)]` reads them as `None`). `SessionStore::open` synthesizes ids
//! for those in memory — position-based, each chained to the message immediately before it — so the
//! reconstructed active path is identical to the old flat-file behavior. Synthesized ids are never
//! persisted back; they're a derived field, following the same pattern already used for `updated_at`/
//! `message_count`/`preview`. No `VERSION` bump: `id`/`parent_id` are additive `#[serde(default)]`
//! fields, so old and new binaries still round-trip through each other's files.
//!
//! A [`SessionRepo`] is a directory of those files: it lists sessions (newest first), creates and
//! opens them, deletes, and **forks** one (copy its prefix into a new session that links back via
//! `parent`) — the headless-relevant slice of pi's session tree. (A "fork" is a *new file*, a
//! session-level split; the tree above is *within* one file — the two are independent mechanisms.)

use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use agent_core::compaction::{CompactionProvenance, CompactionReason};
use agent_core::{ContentBlock, Message, Role, Session};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// On-disk format version, written into every header. The load path refuses any header whose version
/// is *newer* than this — a forward-compat guard so an older binary never silently mis-parses a shape
/// it doesn't understand — and treats `0` (a pre-versioning header, where the field was absent and
/// `serde` defaulted it) as v1-equivalent. The contract: additive, optional fields (each `#[serde(default)]`)
/// do **not** need a bump, since old and new files round-trip through each other; only an incompatible
/// reshaping of existing fields does. When that happens, bump this and add a [`migrate`] arm that
/// upgrades the older shape in place.
const VERSION: u32 = 1;

/// How many characters of the first user message a listing preview keeps.
const PREVIEW_MAX: usize = 80;
/// How many characters of accumulated user-*and*-assistant-message text a listing's `search_text`
/// keeps — enough to substring-match a session by topic without opening every transcript, capped so a
/// huge session doesn't bloat every listing response. pi's own `allMessagesText` is fully uncapped; this
/// stays capped (rather than matching that exactly) because `list_all_sessions` holds one of these per
/// session across a fan-out scan of potentially hundreds of files at once — an uncapped string per
/// session risks real memory pressure at that scale. 50,000 (25x the original 2,000, matching the
/// `50 * 1024`-byte "generous but bounded" budget `tools::output::DEFAULT_MAX_BYTES` already uses
/// elsewhere in this codebase) comfortably covers a realistic session's worth of conversation text —
/// the old 2,000-char cap could truncate after as little as a couple of turns.
const SEARCH_TEXT_MAX_CHARS: usize = 50_000;

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
    /// User *and* assistant message text accumulated across the whole session (not just the first
    /// message, and not just the user's side of it — matching pi's own `allMessagesText`), space-joined
    /// and truncated to [`SEARCH_TEXT_MAX_CHARS`] — a broader surface than `preview` for a client to
    /// substring/fuzzy-match a session by topic (including something only the assistant said) without
    /// opening every transcript. Empty outside of a listing, or when the session has no text yet.
    #[serde(skip)]
    pub search_text: String,
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
            updated_at: 0,
            message_count: 0,
            preview: None,
            search_text: String::new(),
        }
    }

    /// The listing view as JSON: every persisted field, plus the derived-only fields
    /// (`updated_at`/`message_count`/`preview`/`search_text`) that `#[serde(skip)]` deliberately
    /// keeps out of the on-disk header — so a stale scan can never leak into it — but that a
    /// client browsing a listing needs to see. Use this instead of `serde_json::to_value` when
    /// serializing a listing entry for an RPC response.
    pub fn to_listing_json(&self) -> serde_json::Value {
        let mut v = serde_json::to_value(self).unwrap_or(serde_json::Value::Null);
        if let serde_json::Value::Object(map) = &mut v {
            map.insert("updated_at".into(), serde_json::json!(self.updated_at));
            map.insert(
                "message_count".into(),
                serde_json::json!(self.message_count),
            );
            map.insert("preview".into(), serde_json::json!(self.preview));
            map.insert("search_text".into(), serde_json::json!(self.search_text));
        }
        v
    }
}

/// One session's best match against a lowercased, already-trimmed `query`, for ranking search results —
/// `(field_priority, byte_offset)`, ascending order is "more relevant" (a match in a higher-priority
/// field beats any match in a lower one; within the same field, an earlier match beats a later one).
/// Checked in priority order — `title`, `id`, `preview`, `cwd`, then the full `search_text` — and stops
/// at the first field that matches, so a title hit is never outranked by a coincidental `search_text`
/// hit elsewhere. `None` when `query` doesn't appear in any field at all (the session doesn't match).
///
/// Deliberately simpler than pi's own TUI session-picker scorer (`fuzzyMatch` in
/// `packages/tui/src/fuzzy.ts`): that one is a fuzzy subsequence matcher with a hand-tuned heuristic
/// (consecutive-run bonus, word-boundary bonus, gap penalty) built for a human eyeballing highlighted
/// matches as they type. A scripting RPC/CLI caller — the only consumer here, since Beyond has no
/// interactive TUI of its own — wants predictable "does this text contain the query" filtering instead
/// of typo-tolerant fuzzy scoring, so this is a plain case-insensitive substring search.
fn search_rank(meta: &SessionMeta, query_lower: &str) -> Option<(usize, usize)> {
    let fields: [&str; 5] = [
        meta.title.as_deref().unwrap_or(""),
        &meta.id,
        meta.preview.as_deref().unwrap_or(""),
        &meta.cwd,
        &meta.search_text,
    ];
    fields.iter().enumerate().find_map(|(priority, field)| {
        field
            .to_lowercase()
            .find(query_lower)
            .map(|offset| (priority, offset))
    })
}

/// Filter `sessions` to those matching `query` (case-insensitive substring against `title`/`id`/
/// `preview`/`cwd`/`search_text` — see [`search_rank`]), sorted best-match-first with a most-recently-
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
        a_rank
            .cmp(b_rank)
            .then_with(|| b_meta.updated_at.cmp(&a_meta.updated_at))
    });
    ranked.into_iter().map(|(m, _)| m).collect()
}

/// One persisted line. Internally tagged on `type`. A `Message` entry flattens the wrapped
/// `agent_core::Message` (`{role, content}`) alongside its tree fields, so the wire shape is
/// `{"type":"message", id, parent_id, role, content}` — `agent_core::Message` itself carries no tree
/// knowledge (see the module doc comment).
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Entry {
    Session(SessionMeta),
    Message {
        /// This entry's id in the session's tree. Absent (not `null`) on a legacy pre-tree file —
        /// [`SessionStore::open`] synthesizes one in memory; never persisted back.
        #[serde(default)]
        id: Option<String>,
        /// The message this one continues from; `None` at the tree's root. Absent on a legacy file
        /// for the same reason as `id`.
        #[serde(default)]
        parent_id: Option<String>,
        /// Unix seconds this message was appended — the content-derived source for a listing's
        /// `updated_at` (preferred over the file's OS mtime, which a copy/restore/sync elsewhere can
        /// leave stale or wrong without the content itself having changed). `0` (via `#[serde(default)]`)
        /// on a legacy file written before this field existed; a reader treats that as "unknown" and
        /// falls back to mtime, exactly as if the field were absent.
        #[serde(default)]
        timestamp: u64,
        #[serde(flatten)]
        message: Message,
    },
    /// A branch-navigation marker: the active tip moved to `target_id`. `id`/`parent_id` chain leaf
    /// markers the same way message entries chain messages (so the most recent one is unambiguous even
    /// if several land in one file); `target_id` is the payload — the message id now at the tip, or
    /// `None` for the tree's own root (before any message) — see [`SessionStore::switch_active_to_root`],
    /// which lets a client redo the very first message in place. A legacy file's own `target_id` was
    /// always a bare, non-null string, which deserializes as `Some(..)` here with no migration needed.
    Leaf {
        id: String,
        parent_id: Option<String>,
        target_id: Option<String>,
    },
    /// An LLM-generated recap of an abandoned branch (Track L2), persisted when navigating away from
    /// it — see [`SessionStore::switch_active_with_summary`]. `from_id` is the abandoned branch's old
    /// tip (what was summarized); `parent_id` is the branch point being *returned to* — `id`/`parent_id`
    /// chain like every other entry, and `SessionStore::open` treats this one as a real navigation
    /// event too: it becomes the new active tip, materialized into a message so the recap actually
    /// reaches the model, not just a provenance record sitting inert on disk.
    BranchSummary {
        id: String,
        parent_id: Option<String>,
        summary: String,
        from_id: String,
        details: BranchSummaryDetails,
        /// Unix seconds this entry was appended — absent (`#[serde(default)]`, reads as `0`) on a file
        /// written before this field existed. Track L45 (pi-parity fix): previously missing entirely,
        /// so a materialized branch-summary node always reported `timestamp: 0` in `Node`/`TreeNode`
        /// regardless of when it was actually created, unlike `Entry::Message`/`Entry::Custom`.
        #[serde(default)]
        timestamp: u64,
    },
    /// Provenance for a compaction round — see [`SessionStore::rewrite_compacted`]. Unlike
    /// [`Entry::BranchSummary`], this is purely a provenance record: the folded messages named in
    /// `folded_ids` are preserved verbatim elsewhere in the file (never deleted), so `summary` here
    /// would duplicate content that's already live as an ordinary `Message` entry right after this
    /// one — this entry exists so a reader can tell *that* a compaction happened here, and exactly
    /// what it folded, without needing to diff two versions of the file. `id`/`parent_id` chain like
    /// every other entry (`parent_id` is the last folded message's id) but this is inert for tip
    /// resolution, same as a `BranchSummary` used to be before it started redirecting the tip — this
    /// one never should, since the very next entry (the new active-path message) already does that.
    Compaction {
        id: String,
        parent_id: Option<String>,
        /// Estimated input tokens at the moment this compaction fired (before the reset) — the same
        /// value carried on `agent_core::AgentEvent::Compacted`.
        tokens_before: u32,
        /// Ids of every message this round folded away (the old active path's dropped prefix),
        /// oldest-first. Still readable elsewhere in this file by these ids — not deleted.
        folded_ids: Vec<String>,
        /// The generated summary text (without its `SUMMARY_MARKER` wrapper) — duplicated from the
        /// neighboring `Message` entry purely so this record is self-describing without needing to
        /// cross-reference it.
        summary: String,
        /// `agent_core::Session.compaction`'s provenance (`read_files`/`modified_files`/`todos`, folded
        /// forward via `merge_provenance`) as of *this* round — i.e. this round's own new activity
        /// already merged with every prior round's. Since each record is already a complete,
        /// self-contained snapshot, `SessionStore::open` only needs the *last* one in file order to
        /// restore `Session.compaction` in full, not fold every record together itself.
        /// `#[serde(default)]` so a file written before any of these fields existed round-trips
        /// unchanged — such a session simply doesn't get that part of its provenance restored across a
        /// reopen (no worse than before), rather than failing to parse. Fixes a bug where this
        /// provenance was purely in-memory: every `serve` restart or session reattach past a compaction
        /// silently forgot it, so the *next* compaction's `<read-files>`/`<modified-files>` tags quietly
        /// omitted everything from before the restart.
        #[serde(default)]
        read_files: Vec<String>,
        #[serde(default)]
        modified_files: Vec<String>,
        #[serde(default)]
        last_reason: Option<CompactionReason>,
        /// The `todo` list carried across this round (see `CompactionProvenance::todos`). Restored the
        /// same way, so a `serve` restart past a compaction doesn't lose the model's plan — the one
        /// piece of the run's working state that lives nowhere else once `apply_summary` has dropped the
        /// `tool_use` block that carried it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        todos: Option<serde_json::Value>,
    },
    /// A record that the active model changed, anchored to whatever message was the tip at the moment
    /// it did — see [`SessionStore::record_model_change`]/[`SessionStore::model_at`]. `parent_id` here
    /// means "this change applies immediately *after* this message", not "chain the tip here": unlike
    /// `Leaf`/`BranchSummary`, this entry never redirects `self.active` — a plain O(1) append, purely a
    /// lookup record consulted when navigating to a specific point in the tree (`switch_branch`) to
    /// recover whatever model was actually active on that branch, rather than silently continuing with
    /// whatever the process's current global setting happens to be.
    ModelChange {
        id: String,
        parent_id: Option<String>,
        model: String,
    },
    /// Same idea as [`Entry::ModelChange`], for the portable `agent_core::ThinkingLevel` (not a raw
    /// budget override — see [`SessionStore::record_thinking_level_change`]/
    /// [`SessionStore::thinking_level_at`]).
    ThinkingLevelChange {
        id: String,
        parent_id: Option<String>,
        level: String,
    },
    /// A session rename — see [`SessionStore::set_title`]. For the *live* session's own displayed
    /// title (`meta.title`), this isn't anchored/branch-scoped the way `ModelChange`/`ThinkingLevelChange`
    /// are: a rename applies to the whole session regardless of which branch is active, so the most
    /// recent one *anywhere in the file* wins (matching pi's own `session_info` entries) rather than
    /// being looked up per tree point. `parent_id` still chains like every other entry, purely for
    /// on-disk provenance/ordering, for that whole-file resolution — but (pass 15 pi-parity fix) it *is*
    /// separately consulted, anchor-keyed exactly like `ModelChange`/`ThinkingLevelChange`, by
    /// [`SessionStore::title_at_or_root`] — what a fork's own header resolves its title from, so a
    /// rename recorded past the fork point (on any branch) doesn't leak into a branch that never
    /// actually had that name. Replaces the old behavior of rewriting the whole file (with every message
    /// in it) just to update the header's `title` field.
    TitleChange {
        id: String,
        parent_id: Option<String>,
        title: String,
    },
    /// A user-defined bookmark/marker set on another entry — pi's own `LabelEntry`
    /// (`session-manager/labels.test.ts`). `target_id` names the labeled entry; `label: None` clears
    /// whatever label `target_id` currently carries; the most recent `Entry::Label` for a given
    /// `target_id` wins (last-write-wins), matching pi's own `appendLabelChange`.
    ///
    /// Unlike pi — where a label is a full tree node other entries can become a child of, requiring a
    /// "rewire the children of a dropped label" step when forking past one — this is a pure anchored
    /// side-channel record, the same shape as `ModelChange`/`ThinkingLevelChange`/`TitleChange`: an
    /// O(1) append, `id`/`parent_id` chained purely for on-disk provenance/ordering, never redirecting
    /// `self.active` and never itself a parent any other entry chains onto. A label therefore never
    /// occupies a slot in the message chain to begin with, so there is no rewiring problem to solve —
    /// see [`SessionStore::set_label`]/[`SessionStore::get_label`] and
    /// [`SessionRepo::fork_at_entry`]'s label carry-over.
    Label {
        id: String,
        parent_id: Option<String>,
        target_id: String,
        #[serde(default)]
        label: Option<String>,
    },
    /// An opaque, caller-defined entry — pi's own `CustomEntry`
    /// (`session-manager/save-entry.test.ts`), for an extension/tool to attach app-defined data at a
    /// specific point in the tree (`kind` identifies the shape of `data` to whatever produced it; this
    /// module never interprets either). Unlike `Label`/`ModelChange`/`ThinkingLevelChange`/`TitleChange`
    /// (anchored side-channel records that never occupy a chain slot), this IS a real tree node: it
    /// becomes the new active tip when appended (see [`SessionStore::append_custom`]), and a later
    /// message's `parent_id` can point at it — but it contributes nothing when the active path is
    /// materialized into `Session.messages`/LLM context (see [`Node::as_message`]), matching pi's own
    /// `buildSessionContext` skipping `"custom"`-typed entries. Still reported by [`SessionStore::tree`]
    /// for full-tree traversal, so a client can see it happened even though the model never does.
    Custom {
        id: String,
        parent_id: Option<String>,
        #[serde(default)]
        timestamp: u64,
        kind: String,
        #[serde(default)]
        data: Value,
    },
}

/// File-tracking details persisted alongside an [`Entry::BranchSummary`] — the branch-summary analog
/// of [`agent_core::CompactionProvenance`]'s read/modified lists, scoped to just the summarized branch
/// rather than folded forward across rounds (a branch is summarized once, not repeatedly).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BranchSummaryDetails {
    #[serde(default)]
    pub read_files: Vec<String>,
    #[serde(default)]
    pub modified_files: Vec<String>,
    /// How many messages were folded into this summary — what
    /// `SessionMeta::summarized_branch_messages` accumulates.
    #[serde(default)]
    pub summarized_messages: u64,
}

/// Provenance passed to [`SessionStore::rewrite_compacted`], recorded on the new [`Entry::Compaction`]
/// line alongside what that method derives on its own (`folded_ids`, `summary`).
pub struct CompactionMeta {
    /// Estimated input tokens at the moment this compaction fired (before the reset) — the same value
    /// carried on `agent_core::AgentEvent::Compacted`.
    pub tokens_before: u32,
    /// `Session.compaction` right after this round's own `merge_file_ops` fold — persisted onto the new
    /// `Entry::Compaction` record's `read_files`/`modified_files`/`last_reason` (see that variant's own
    /// doc comment) so a later `SessionStore::open` can restore it instead of silently starting over
    /// from empty.
    pub provenance: CompactionProvenance,
}

/// An [`Entry::Compaction`] record, indexed by its own id — read back out for [`SessionStore::tree`]
/// (Track L25). `Entry::Compaction` is deliberately never a real chain node (see its own doc comment:
/// "this one never should [redirect the tip]"), so it's tracked in this side index instead of
/// `nodes`/`NodeContent`, the same way `branch_summary_details` tracks structured data for an
/// `Entry::BranchSummary` alongside (rather than instead of) that one's real `NodeContent::Message`
/// node.
#[derive(Debug, Clone)]
struct CompactionRecord {
    parent_id: Option<String>,
    tokens_before: u32,
    folded_ids: Vec<String>,
}

/// A non-message session event, in file order — Track L36 (pi-parity fix): `Entry::ModelChange`/
/// `Entry::ThinkingLevelChange`/`Entry::Label`/`Entry::Custom` are all durably tracked (as last-write-
/// wins lookup maps, or — for `Custom` — a real tree node), but none of that is an ordered event log a
/// human reader would want surfaced in an HTML export; before this existed, an export showed none of
/// them at all. Deliberately not the internal `Entry` wire format itself (that stays private to this
/// module) — just enough of each variant's payload for [`crate::export`] to render a simple block.
/// `Entry::Message`/`Entry::BranchSummary` already reach an export via `messages`/`branches`;
/// `Entry::Leaf`/`Entry::TitleChange`/`Entry::Session`/`Entry::Compaction` carry nothing a reader would
/// want as its own block here, so this deliberately covers only the four that do.
#[derive(Debug, Clone, PartialEq)]
pub enum ExportEvent {
    ModelChange(String),
    ThinkingLevelChange(String),
    Label {
        target_id: String,
        label: Option<String>,
    },
    Custom {
        kind: String,
        data: Value,
    },
}

/// What [`SessionRepo::fork_at_entry_prefix`] computes before anything is written — the would-be
/// child's metadata, its message prefix, and enough of the source's own bookkeeping
/// ([`original_ids`](Self::original_ids), [`labels`](Self::labels)) for [`SessionRepo::fork_at_entry`]
/// to carry labels forward once [`SessionStore::rewrite`] mints the copy's real ids.
struct ForkPrefix {
    meta: SessionMeta,
    messages: Vec<Message>,
    /// The source session's own ids for `messages`, in the same order — i.e. before `rewrite` replaces
    /// them with a fresh chain. `rewrite` cannot preserve ids in general (a compaction's summary
    /// message has no 1:1 original to key off of), but a *fork*'s prefix genuinely is an unmodified
    /// copy, assigned fresh ids in the same order it was read — so zipping this against the new store's
    /// `active_ids()` after `rewrite` recovers the old-id → new-id mapping without `rewrite` itself
    /// needing to know or preserve anything.
    original_ids: Vec<String>,
    /// Labels recorded in the source session against any id in `original_ids`, as `(target_id, label)`
    /// pairs — see [`SessionStore::labels_within`].
    labels: Vec<(String, String)>,
    /// The thinking level actually in effect at the resolved fork point — see
    /// [`SessionStore::thinking_level_at_or_created`]'s own doc comment. `None` when nothing was ever
    /// recorded reaching that point (a fork of a session, or a branch of one, that never touched its
    /// thinking level), in which case [`SessionRepo::fork_at_entry`] writes no entry for it at all.
    thinking_level: Option<String>,
}

/// One branch in the session's tree, as reported by [`SessionStore::list_branches`] — a leaf (a node
/// with no children) plus enough to render a picker: whether it's the currently active one, how deep
/// its path from the root runs, and a preview of its first user text.
#[derive(Debug, Clone, Serialize)]
pub struct BranchInfo {
    /// The id of the message at this branch's tip.
    pub leaf_id: String,
    /// Whether this is the branch the session is currently on.
    pub is_active: bool,
    /// Number of messages from the root through this branch's tip.
    pub message_count: usize,
    /// The first user message's text on this branch's path, truncated like a session listing's
    /// preview. `None` if the branch has no user text at all.
    pub preview: Option<String>,
    /// Unix seconds the tip message was appended — `0` if not recorded (see [`Node::timestamp`]). Lets
    /// a client reconstruct which of several branches forking from the same point came first, matching
    /// pi's own `SessionTreeNode.entry.timestamp`.
    pub timestamp: u64,
}

/// A [`Node`]'s content — either a real conversation message, or an opaque caller-defined entry (see
/// [`Entry::Custom`]) that occupies a slot in the tree but contributes nothing when the active path is
/// materialized into `Session.messages`/LLM context.
#[derive(Clone)]
enum NodeContent {
    Message(Message),
    Custom { kind: String, data: Value },
}

/// One node in the in-memory tree index: an entry's parent link and its content. Spans the *whole*
/// file (every branch), not just the active path — built once on [`SessionStore::open`]/`create` and
/// kept in sync by every mutating method, so a branch can be materialized without re-reading the file.
#[derive(Clone)]
struct Node {
    parent_id: Option<String>,
    content: NodeContent,
    /// Unix seconds this entry was actually appended, from [`Entry::Message`]/[`Entry::Custom`]'s own
    /// `timestamp` field — `0` for a node with no recorded timestamp (a legacy file written before this
    /// field existed, or a branch-summary-materialized node, which doesn't carry one). See
    /// [`read_listing`]'s `updated_at` computation for why this exists: a content-derived signal,
    /// preferred over the file's OS mtime, which a copy/restore/sync that doesn't preserve it exactly
    /// can leave stale or wrong.
    timestamp: u64,
}

impl Node {
    /// This node's message content, or `None` for a non-message node (currently only
    /// [`NodeContent::Custom`]) — what every "materialize the active path/a branch into messages" call
    /// site filters through, so a custom entry contributes nothing to `Session.messages`/LLM context
    /// while still being a real, positioned node for tree traversal (`SessionStore::tree`) — matching
    /// pi's own `buildSessionContext` skipping `"custom"`-typed entries. See [`Entry::Custom`]'s doc
    /// comment.
    fn as_message(&self) -> Option<&Message> {
        match &self.content {
            NodeContent::Message(m) => Some(m),
            NodeContent::Custom { .. } => None,
        }
    }
}

/// One node in the session's tree, as reported by [`SessionStore::tree`] — every message (not just the
/// active path, and not just a branch's leaf like [`BranchInfo`]), with its own parent link, role, and
/// a short preview of its own text content.
#[derive(Debug, Clone, Serialize)]
pub struct TreeNode {
    pub id: String,
    /// `None` at the tree's root.
    pub parent_id: Option<String>,
    /// `None` for a non-message node ([`Entry::Custom`] — see [`Node::as_message`]), which has no role
    /// of its own.
    pub role: Option<Role>,
    /// A preview of this message's own text content, or `None` for a pure tool-use/tool-result/
    /// thinking/image turn with no plain-text block. For a custom entry, `Some("[custom: {kind}]")`.
    pub preview: Option<String>,
    /// The label currently set on this node, if any — see [`SessionStore::set_label`]. Pi's own
    /// `SessionTreeNode.label`.
    pub label: Option<String>,
    /// Unix seconds this entry was appended — `0` if not recorded (see [`Node::timestamp`]). Lets a
    /// client reconstruct sibling/branch order chronologically, matching pi's own
    /// `SessionTreeNode.entry.timestamp` (which pi sorts each node's children by when rendering a tree).
    pub timestamp: u64,
    /// The real [`Entry`] variant this node came from: `"message"` for an ordinary conversation turn,
    /// `"custom"` for [`Entry::Custom`], `"branch_summary"` for an [`Entry::BranchSummary`] recap
    /// (materialized into a real `Message` so the model actually sees it — see
    /// `branch_summary_message` — but still distinguishable here from an ordinary message), or
    /// `"compaction"` for an [`Entry::Compaction`] provenance record (never part of the active-path
    /// chain itself — see that variant's own doc comment — but still reported here, with its
    /// `tokens_before`/folded count folded into `preview`, so a client can tell *that* a compaction
    /// happened without re-parsing the raw session file). Track L25 (pi-parity fix): previously a
    /// branch summary collapsed indistinguishably into a plain `"message"`-shaped node and a
    /// compaction never appeared in [`SessionStore::tree`] at all.
    pub entry_kind: &'static str,
}

/// Turn a persisted branch summary's text into the message that materializes at the tip of the branch
/// being returned to — how the recap actually reaches the model on the next turn, mirroring
/// `agent_core::compaction`'s `SUMMARY_MARKER`-prefixed user message for the same purpose.
fn branch_summary_message(summary: &str) -> Message {
    Message::user(format!(
        "{}\n\n{}",
        agent_core::BRANCH_SUMMARY_MARKER,
        summary
    ))
}

/// Walk `nodes`' parent chain from `tip` back to the root, returning ids root-first. `tip = None` (an
/// empty session) or a `tip` naming no known node both yield an empty path — a dangling/unknown
/// reference degrades to "nothing here" rather than panicking, matching this module's crash-recovery
/// posture elsewhere (an unreadable tail is dropped, not fatal).
///
/// Guards against a cyclic `parent_id` chain (only reachable via a hand-edited or corrupted session
/// file — no in-process mutation can produce one, since every parent always chains to a strictly
/// earlier id): without the visited-set check, a cycle would walk forever, growing `rev` unboundedly
/// rather than degrading to "nothing" the way every other malformed-input case here does. A repeated
/// id ends the walk at that point (treating whatever was reached as the root) instead of looping.
fn path_from_root(nodes: &HashMap<String, Node>, tip: Option<&str>) -> Vec<String> {
    let mut rev = Vec::new();
    let mut visited = HashSet::new();
    let mut cur = tip.map(str::to_string);
    while let Some(id) = cur {
        if !visited.insert(id.clone()) {
            break;
        }
        let Some(node) = nodes.get(&id) else { break };
        cur = node.parent_id.clone();
        rev.push(id);
    }
    rev.reverse();
    rev
}

/// A handle to one session's append log.
///
/// Not `Sync`-guarded internally, and deliberately so: every mutating method takes `&mut self`, and
/// every call site (`serve`'s single-threaded control loop) holds its `SessionStore` as a plain owned
/// value, never behind an `Arc`. Rust's borrow checker already makes a concurrent `append_new`/`rewrite`
/// race on one `SessionStore` unrepresentable *within a process* — there is no code path that could
/// call two mutating methods on the same store at once, so an in-process lock would guard against a
/// scenario that can't happen. What this does *not* cover is two separate **processes** opening the
/// same session file (e.g. an operator accidentally pointing two `serve --session-file` invocations at
/// one path) — nothing here takes an OS-level advisory lock, so that remains a real, if currently
/// unreachable, hazard: no feature today opens one session file from more than one process. If that
/// ever changes (a future multi-process feature), add a `flock`-based lock in `open`/`create` then —
/// not before, per this project's minimum-effective-abstraction standard.
pub struct SessionStore {
    path: PathBuf,
    meta: SessionMeta,
    /// How many messages are already on disk (on the active path) — the append cursor.
    persisted: usize,
    /// Every known node (message), by id — spans the whole tree, not just the active path.
    nodes: HashMap<String, Node>,
    /// Ids of the messages on the currently active path, root-first, parallel to `Session.messages`.
    /// `append_new` chains new messages' `parent_id` off `active.last()`; `rewrite` replaces this
    /// wholesale (a fresh linear chain) while preserving every id NOT in it (see the module doc's
    /// compaction-vs-branching section).
    active: Vec<String>,
    /// [`BranchSummaryDetails`] for every [`Entry::BranchSummary`] seen, by that entry's own id — a
    /// `Node`'s materialized message keeps only the prose recap, not this structured file-tracking data,
    /// so a later branch summary that itself abandons a range containing an earlier one needs this index
    /// to fold that earlier summary's `read_files`/`modified_files` forward (see
    /// [`Self::branch_summary_details_within`]) rather than losing them once the prose-only message is
    /// all that's left to scan.
    branch_summary_details: HashMap<String, BranchSummaryDetails>,
    /// The model in effect immediately after each message id (`None` = before the session's first
    /// message) — the last [`Entry::ModelChange`] anchored there, if any. See
    /// [`Self::record_model_change`]/[`Self::model_at`].
    model_changes: HashMap<Option<String>, String>,
    /// Same idea as `model_changes`, for the portable thinking level (see [`Entry::ThinkingLevelChange`]/
    /// [`Self::record_thinking_level_change`]/[`Self::thinking_level_at`]).
    level_changes: HashMap<Option<String>, String>,
    /// The title in effect immediately after each message id (`None` = before the session's first
    /// message) — the last [`Entry::TitleChange`] anchored there, if any, keyed the same way as
    /// `model_changes`/`level_changes`. `Some(None)` records an explicit clear (a rename that sanitized
    /// to empty) actually in effect at that anchor, distinct from no entry at all ("nothing ever
    /// recorded reaching this point"). Unlike `model_changes`/`level_changes`, this is consulted only
    /// for [`SessionRepo::fork`]/`fork_from_path`/`fork_at_entry_prefix`'s path-scoped resolution (pass
    /// 15 pi-parity fix, [`Self::title_at_or_root`]) — the *live* session's own displayed title
    /// (`meta.title`, restored by [`Self::open`]) still resolves whole-file-latest, matching pi's own
    /// `getSessionName` scanning every entry regardless of branch.
    title_changes: HashMap<Option<String>, Option<String>>,
    /// The label currently set on each target id, by that id — the last (by file order)
    /// [`Entry::Label`] seen for it, with a `label: None` entry removing it from this map entirely
    /// (last-write-wins). See [`Self::set_label`]/[`Self::get_label`].
    labels: HashMap<String, String>,
    /// Every [`Entry::Compaction`] record seen, by its own id — see [`CompactionRecord`]'s doc comment
    /// for why this lives in its own side index rather than `nodes`. Read back out by
    /// [`Self::tree`] (Track L25).
    compactions: HashMap<String, CompactionRecord>,
    /// Every [`ExportEvent`] seen, in file order — see that type's own doc comment. Read back out by
    /// [`Self::export_events`] (Track L36).
    events: Vec<ExportEvent>,
}

impl SessionStore {
    /// Create a new session file at `path`, writing its header. Errors if the file already exists —
    /// *unless* it's empty: a zero-byte file at `path` (e.g. `touch`'d ahead of time by a caller that
    /// wants the path to already exist, or left over from a crash before the header write landed) is
    /// indistinguishable in intent from "not created yet," so it's initialized in place instead of
    /// failing (idempotent per this crate's "check before create; don't error if it's [effectively]
    /// gone" convention). A genuinely non-empty file at `path` is real, possibly conflicting data, and
    /// is never silently clobbered.
    pub fn create(path: PathBuf, meta: SessionMeta) -> std::io::Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let open = |truncate_existing: bool| -> std::io::Result<File> {
            let mut opts = OpenOptions::new();
            opts.write(true);
            if truncate_existing {
                opts.truncate(true);
            } else {
                opts.create_new(true);
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                opts.mode(0o600);
            }
            opts.open(&path)
        };
        let mut f = match open(false) {
            Ok(f) => f,
            // The atomic fast path failed because *something* is already there — only initialize in
            // place if it's genuinely empty; otherwise propagate the original error rather than risk
            // clobbering real data on a race.
            Err(e)
                if e.kind() == std::io::ErrorKind::AlreadyExists
                    && fs::metadata(&path).is_ok_and(|m| m.len() == 0) =>
            {
                open(true)?
            }
            Err(e) => return Err(e),
        };
        write_line(&mut f, &Entry::Session(meta.clone()))?;
        // Durability: get the header bytes (and the new file's directory entry) onto stable storage
        // before returning, so a crash right after `create` can't lose the session entirely.
        f.flush()?;
        f.sync_all()?;
        fsync_dir(&path)?;
        Ok(Self {
            path,
            meta,
            persisted: 0,
            nodes: HashMap::new(),
            active: Vec::new(),
            branch_summary_details: HashMap::new(),
            model_changes: HashMap::new(),
            level_changes: HashMap::new(),
            title_changes: HashMap::new(),
            labels: HashMap::new(),
            compactions: HashMap::new(),
            events: Vec::new(),
        })
    }

    /// Open an existing session file, returning the store and the restored [`Session`] (the active
    /// path's messages). A torn final line (crash mid-append) is skipped; a header is required.
    pub fn open(path: PathBuf) -> std::io::Result<(Self, Session)> {
        let file = File::open(&path)?;
        let mut meta: Option<SessionMeta> = None;
        let mut nodes: HashMap<String, Node> = HashMap::new();
        // The active tip, updated in *file order* by whichever kind of entry sets it: a `Message`
        // entry always becomes the new tip (an append always continues the active chain), and a `Leaf`
        // entry redirects it to `target_id`. Tracking one variable this way — rather than remembering
        // "the last message" and "the last leaf" separately and picking between them afterward — is
        // what makes this correct when messages are appended *after* a `Leaf` (the tip must move back
        // to that latest message, not stay pinned at the leaf's target).
        let mut tip: Option<String> = None;
        let mut next_synth: u64 = 0;
        let mut branch_summary_details: HashMap<String, BranchSummaryDetails> = HashMap::new();
        let mut model_changes: HashMap<Option<String>, String> = HashMap::new();
        let mut level_changes: HashMap<Option<String>, String> = HashMap::new();
        let mut title_changes: HashMap<Option<String>, Option<String>> = HashMap::new();
        let mut labels: HashMap<String, String> = HashMap::new();
        let mut compactions: HashMap<String, CompactionRecord> = HashMap::new();
        // The most recent `Entry::Compaction` record's file-provenance seen so far, in file order — see
        // that variant's own doc comment for why "most recent" is already "complete" and needs no
        // folding here. `None` when this session was never compacted, matching `Session.compaction`'s
        // own all-empty default.
        let mut compaction_provenance: Option<CompactionProvenance> = None;
        let mut events: Vec<ExportEvent> = Vec::new();

        let mut reader = BufReader::new(file);
        let mut raw = Vec::new();
        loop {
            // A genuine I/O read failure stops the load, keeping whatever was valid so far (only the
            // in-flight entry can be lost). An oversized or invalid-UTF-8 *line*, though, is a fully
            // read, boundary-known line, exactly like the deserialize-failure case below — skip just
            // that one and keep scanning, rather than discarding every good entry after it (which used
            // to happen here: a single bit-rotted or hand-edited line anywhere in the file silently
            // truncated the whole session).
            let oversized = match read_capped_line(&mut reader, &mut raw) {
                Ok(None) => break,
                Ok(Some(oversized)) => oversized,
                Err(_) => break,
            };
            if oversized {
                tracing::warn!(path = %path.display(), "skipping oversized session entry line");
                continue;
            }
            let Ok(line) = std::str::from_utf8(&raw) else {
                tracing::warn!(path = %path.display(), "skipping non-UTF-8 session entry line");
                continue;
            };
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            match parse_entry_lenient(&path, line) {
                Ok(Entry::Session(m)) => meta = Some(m),
                Ok(Entry::Message {
                    id,
                    parent_id,
                    timestamp,
                    message,
                }) => {
                    // Legacy (pre-tree) migration: a *missing* id is the actual legacy signal — only
                    // then synthesize one and chain its parent off whatever message came immediately
                    // before, reproducing the old flat file's implicit linear order exactly. Neither is
                    // ever written back (see the module doc comment). A tree-aware entry's `parent_id`
                    // is trusted exactly as persisted even when it's `None` — that's a genuine root, not
                    // a missing field, and `#[serde(default)]` can't tell the two apart once
                    // deserialized (both read as `None`), so the branch must be on `id`, not `parent_id`.
                    let is_legacy = id.is_none();
                    let id = id.unwrap_or_else(|| {
                        let synth = format!("legacy-{next_synth}");
                        next_synth += 1;
                        synth
                    });
                    let parent_id = if is_legacy {
                        parent_id.or_else(|| tip.clone())
                    } else {
                        parent_id
                    };
                    nodes.insert(
                        id.clone(),
                        Node {
                            parent_id,
                            content: NodeContent::Message(message),
                            timestamp,
                        },
                    );
                    tip = Some(id);
                }
                Ok(Entry::Leaf { target_id, .. }) => tip = target_id,
                // A branch summary *does* become the new tip — it's a child of the branch point being
                // returned to (see `switch_active_with_summary`), materialized into a real message so
                // the recap actually reaches the model on the next turn, not just sitting on disk.
                Ok(Entry::BranchSummary {
                    id,
                    parent_id,
                    summary,
                    details,
                    timestamp,
                    ..
                }) => {
                    nodes.insert(
                        id.clone(),
                        Node {
                            parent_id,
                            content: NodeContent::Message(branch_summary_message(&summary)),
                            // Track L45 (pi-parity fix): `Entry::BranchSummary` now carries its own
                            // `timestamp`, same as `Entry::Message`/`Entry::Custom` — `0` here only for
                            // a legacy file written before this field existed (`#[serde(default)]`),
                            // in which case `read_listing`'s `updated_at` computation already treats
                            // that as "no signal" and falls through to whatever else it finds (a later
                            // real message, or mtime).
                            timestamp,
                        },
                    );
                    branch_summary_details.insert(id.clone(), details);
                    tip = Some(id);
                }
                // Becomes the new tip just like a `Message` (a real, positioned tree node — see
                // `Entry::Custom`'s doc comment) — a later message's `parent_id` can point at it.
                Ok(Entry::Custom {
                    id,
                    parent_id,
                    timestamp,
                    kind,
                    data,
                }) => {
                    // Track L36: also recorded as an `ExportEvent` — a custom entry contributes
                    // nothing to `Session.messages` (see this variant's own doc comment), so an HTML
                    // export has no other way to know it happened at all.
                    events.push(ExportEvent::Custom {
                        kind: kind.clone(),
                        data: data.clone(),
                    });
                    nodes.insert(
                        id.clone(),
                        Node {
                            parent_id,
                            content: NodeContent::Custom { kind, data },
                            timestamp,
                        },
                    );
                    tip = Some(id);
                }
                // Purely a provenance record (see `Entry::Compaction`'s doc comment) — the very next
                // entry in the file is the real, live message this compaction produced, so this one
                // must never itself move the tip. Still indexed by id (Track L25) so `SessionStore::tree`
                // can report that a compaction happened here, distinctly from an ordinary message.
                Ok(Entry::Compaction {
                    id,
                    parent_id,
                    tokens_before,
                    folded_ids,
                    read_files,
                    modified_files,
                    last_reason,
                    todos,
                    ..
                }) => {
                    // Each record already carries every earlier round's file-provenance folded in (via
                    // `merge_file_ops`, before it was even written — see `CompactionMeta::provenance`'s
                    // doc comment), so it's a complete, self-contained snapshot: the *last* one in file
                    // order (this loop runs in file order, so a later assignment here simply overwrites
                    // an earlier one) is all `Session.compaction` needs, restored below once `meta` (for
                    // its `compactions` counter) is available. Fixes a bug where this provenance was
                    // purely in-memory and silently reset to empty on every reopen past a compaction.
                    compaction_provenance = Some(CompactionProvenance {
                        read_files,
                        modified_files,
                        compactions: 0, // replaced by `meta.compactions` (the authoritative counter) below
                        last_reason,
                        todos,
                    });
                    compactions.insert(
                        id,
                        CompactionRecord {
                            parent_id,
                            tokens_before,
                            folded_ids,
                        },
                    );
                }
                // Neither moves the tip — a pure lookup record, last-write-wins per anchor (see
                // `Entry::ModelChange`'s doc comment). Deliberately does NOT touch `meta.model`/
                // `meta.thinking_level` (Task #18, pi-parity investigation) — see
                // `SessionStore::record_model_change`'s own doc comment for why: `meta.model` must stay
                // the session's true creation-time value (never overwritten by a later change) for
                // `Persistence::model_and_level_at`'s fallback to resolve correctly on reopen, exactly
                // as it already does for a still-running process.
                Ok(Entry::ModelChange {
                    parent_id, model, ..
                }) => {
                    events.push(ExportEvent::ModelChange(model.clone()));
                    model_changes.insert(parent_id, model);
                }
                // Same idea as `ModelChange` just above.
                Ok(Entry::ThinkingLevelChange {
                    parent_id, level, ..
                }) => {
                    events.push(ExportEvent::ThinkingLevelChange(level.clone()));
                    level_changes.insert(parent_id, level);
                }
                // `meta.title` (the live session's own displayed title) is whole-session-scoped, not
                // anchored to `parent_id` — the most recent one in file order wins, regardless of tree
                // position, matching pi's own `getSessionName` scanning every entry regardless of
                // branch. `meta` is always `Some` by the time a `TitleChange` is read, since the header
                // entry is always first.
                //
                // `title_changes` (pass 15 pi-parity fix) *is* keyed by `parent_id`, same as
                // `model_changes`/`level_changes` — consulted separately by `title_at_or_root` for a
                // fork's own path-scoped resolution (see that method's doc comment).
                Ok(Entry::TitleChange {
                    parent_id, title, ..
                }) => {
                    let resolved = title_or_clear(title);
                    if let Some(m) = &mut meta {
                        m.title = resolved.clone();
                    }
                    title_changes.insert(parent_id, resolved);
                }
                // Never redirects `tip` — a pure lookup record keyed directly by `target_id` (unlike
                // `model_changes`/`level_changes`, which are keyed by *anchor* and apply to descendants;
                // a label applies only to the exact entry named). Last-write-wins per `target_id`,
                // matching pi's own `appendLabelChange`.
                Ok(Entry::Label {
                    target_id, label, ..
                }) => {
                    events.push(ExportEvent::Label {
                        target_id: target_id.clone(),
                        label: label.clone(),
                    });
                    match label {
                        Some(l) => {
                            labels.insert(target_id, l);
                        }
                        None => {
                            labels.remove(&target_id);
                        }
                    }
                }
                // A line that read fully (valid UTF-8, under the size cap) but failed to deserialize as
                // an `Entry` — a bad line *other* than a torn final write (disk bit rot, a manual edit,
                // a future `Entry` variant an older binary doesn't know about yet). Unlike the
                // read-level failures above (where we can't be sure where the next line even begins),
                // we already have this line's exact boundaries, so skip just this one and keep reading:
                // a single bad line no longer discards every good entry that follows it.
                Err(e) => {
                    tracing::warn!(path = %path.display(), error = %e, "skipping unparseable session entry line");
                }
            }
        }
        let meta = meta.ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "session file has no header",
            )
        })?;
        // Validate (and, in a future format, upgrade) the header before trusting the rest of the file.
        let meta = migrate(meta, &path)?;

        let active = path_from_root(&nodes, tip.as_deref());
        // A custom entry (`NodeContent::Custom`) contributes nothing here — it's a real, positioned
        // node in `active`'s chain (see `Entry::Custom`'s doc comment), but not a message, so
        // `as_message` filters it out of the materialized `Session.messages`/LLM context.
        let messages: Vec<Message> = active
            .iter()
            .filter_map(|id| nodes[id].as_message().cloned())
            .collect();
        let persisted = messages.len();
        let mut session = Session::new();
        session.messages = Arc::new(messages);
        // Restore `Session.compaction` (Fix 2, pi-parity fix) from the last `Entry::Compaction` record's
        // file-provenance, if this session was ever compacted — see that variant's own doc comment.
        // `compactions` is replaced with `meta.compactions` (the counter already correctly restored by
        // `migrate` above) rather than trusting the record's own implicit count, so this can't drift
        // from the one other place this session already tracks it.
        if let Some(provenance) = compaction_provenance {
            session.compaction = CompactionProvenance {
                compactions: meta.compactions,
                ..provenance
            };
        }
        // Restore a proactive-compaction trigger signal for a resumed session — a freshly-opened
        // session's `last_input_tokens` otherwise defaults to 0, and `should_compact`/`is_hard_overflow`
        // both require it to be positive to fire at all. Left unset, a resumed session already well
        // over the compaction threshold wouldn't proactively compact until a *new* turn produced fresh
        // real usage — one whole turn later than it should (pi's own regression test:
        // `pre-prompt-compaction-no-continue`).
        //
        // Task #6 (pi-parity fix): Round 1 added per-message `usage` (`Message::usage`/
        // `Message::with_usage`), persisted automatically as part of `Message`'s own `Serialize`/
        // `Deserialize` derive — no format change needed here, and `#[serde(default,
        // skip_serializing_if = "Option::is_none")]` means a session file written before that field
        // existed just deserializes every message's `usage` as `None` (full backward compatibility: no
        // migration, no version bump). When the most recent message carrying one exists, its own
        // provider-reported figures reconstruct `last_input_tokens`/`last_output_tokens`/
        // `last_usage_message_count` *exactly* as a still-running process would have them immediately
        // after that turn (the same `input + cache_read + cache_write` combination
        // `Session::record_usage`'s own `live_input` computes, and the same "snapshot taken right
        // before this message was pushed" positioning `trailing_tokens`'s own substitution relies on —
        // see its doc comment) — a real figure, not the char/4 estimate below. Everything *after* that
        // message (usually nothing) is still covered by `trailing_tokens`' own per-message estimate,
        // exactly like a live session's own turns since its last real usage snapshot.
        //
        // A session with no `usage`-carrying message at all (every message predates Round 1, or the
        // active path is entirely a compaction/branch-summary recap with none recorded) falls back to
        // the previous whole-transcript char/4 estimate, unchanged: treat every persisted message as
        // "trailing" (by calling `trailing_tokens` while `last_usage_message_count` is still its
        // default 0) and then mark it all as already accounted for, so the very next real
        // `trailing_tokens` call doesn't double-count it.
        match session
            .messages
            .iter()
            .enumerate()
            .rev()
            .find_map(|(i, m)| m.usage.map(|u| (i, u)))
        {
            Some((i, usage)) => {
                session.last_usage_message_count = i;
                session.last_output_tokens = usage.output_tokens;
                session.last_input_tokens = usage
                    .input_tokens
                    .saturating_add(usage.cache_read_tokens)
                    .saturating_add(usage.cache_write_tokens);
            }
            None => {
                session.last_input_tokens = agent_core::compaction::trailing_tokens(&session);
                session.last_usage_message_count = session.messages.len();
            }
        }
        Ok((
            Self {
                path,
                meta,
                persisted,
                nodes,
                active,
                branch_summary_details,
                model_changes,
                level_changes,
                title_changes,
                labels,
                compactions,
                events,
            },
            session,
        ))
    }

    /// The session's metadata.
    pub fn meta(&self) -> &SessionMeta {
        &self.meta
    }

    /// The on-disk path this session's JSONL file lives at — pi's `sessionFile`, surfaced via
    /// `get_state` for a client that wants to know exactly what's being written to.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Ids of the active path's messages, root-first — parallel to the `Session.messages` this store
    /// last produced (via `open`, `append_new`, `rewrite`, or `switch_active`). What a caller quotes
    /// back to [`Self::switch_active`] to navigate to a specific point in the history.
    pub fn active_ids(&self) -> &[String] {
        &self.active
    }

    /// The message at tree entry `id`, anywhere in the whole tree (on or off the active path) —
    /// `None` for an unknown id or one naming a non-message node ([`Entry::Custom`]). Used by `fork`'s
    /// own response (`serve.rs`'s `Persistence::fork_source_text`) to echo the forked-from message's
    /// text back to the caller without a second `get_fork_messages`-style round trip.
    pub fn message_at(&self, id: &str) -> Option<&Message> {
        self.nodes.get(id).and_then(Node::as_message)
    }

    /// Append the messages added since the last persist. O(new messages).
    ///
    /// Every new line is serialized into an in-memory buffer first, then written with one `write_all`
    /// — not one `write_line` call per message straight to the file. A per-message write is *not* safe
    /// to retry: the OS page cache reflects each successful write immediately (well before `sync_all`),
    /// so if line k of an N-line batch failed (I/O error, or a `serde_json` failure on that specific
    /// message), lines `0..k` would already be durably visible on disk while `self.persisted` never
    /// advanced — a caller that retries the same batch after the error would then re-append `0..k`,
    /// duplicating history. Serializing to a buffer first means a mid-batch failure never touches the
    /// file at all; the one `write_all` against the open file is then all-or-nothing at this layer
    /// (and even a partial short write only leaves a torn trailing line, which `open`'s load path
    /// already drops per the module's crash-recovery contract).
    pub fn append_new(&mut self, messages: &[Message]) -> std::io::Result<()> {
        if messages.len() <= self.persisted {
            return Ok(());
        }
        let mut buf = Vec::new();
        // Staged locally and only merged into `self.nodes`/`self.active` after the write succeeds —
        // mirrors the buffer-then-write pattern above: a mid-batch failure must never leave in-memory
        // tree state ahead of what's actually durable on disk.
        let mut staged: Vec<(String, Node)> = Vec::with_capacity(messages.len() - self.persisted);
        let mut parent = self.active.last().cloned();
        for msg in &messages[self.persisted..] {
            let id = new_id();
            let timestamp = now_secs();
            write_line(
                &mut buf,
                &Entry::Message {
                    id: Some(id.clone()),
                    parent_id: parent.clone(),
                    timestamp,
                    message: msg.clone(),
                },
            )?;
            staged.push((
                id.clone(),
                Node {
                    parent_id: parent.clone(),
                    content: NodeContent::Message(msg.clone()),
                    timestamp,
                },
            ));
            parent = Some(id);
        }
        let mut f = OpenOptions::new().append(true).open(&self.path)?;
        f.write_all(&buf)?;
        // `flush` only pushes past our buffer into the OS; `sync_all` forces the bytes to disk, which
        // is what the module's crash-safety claim actually requires. The parent dir is unchanged on an
        // append (same inode, same dentry), so no directory fsync is needed here.
        f.flush()?;
        f.sync_all()?;
        for (id, node) in staged {
            self.active.push(id.clone());
            self.nodes.insert(id, node);
        }
        self.persisted = messages.len();
        Ok(())
    }

    /// Append an opaque, caller-defined entry as a child of the current active tip, then advance the
    /// tip to it — pi's own `appendCustomEntry` (`session-manager/save-entry.test.ts`). Returns the new
    /// entry's id. `kind` identifies the shape of `data` to whatever produced it; this module never
    /// interprets either.
    ///
    /// Unlike [`append_new`](Self::append_new) (which only ever adds real conversation messages), this
    /// grows `self.active` by one — the entry genuinely occupies a slot in the chain, so a later
    /// `append_new`'s next message correctly parents onto it — but contributes nothing when the active
    /// path is materialized into `Session.messages`: `self.persisted` (a *message* count) is
    /// deliberately left untouched here, so it stays exactly what `append_new`'s own `messages[self
    /// .persisted..]` diffing needs regardless of how many custom entries have been interspersed. See
    /// [`Entry::Custom`]'s doc comment for why this can safely be a real tree node without disturbing
    /// compaction's own message-counting (`rewrite_compacted`'s `folded_ids` walk already accounts for
    /// this).
    pub fn append_custom(
        &mut self,
        kind: impl Into<String>,
        data: serde_json::Value,
    ) -> std::io::Result<String> {
        let kind = kind.into();
        let id = new_id();
        let timestamp = now_secs();
        let parent_id = self.active.last().cloned();
        let entry = Entry::Custom {
            id: id.clone(),
            parent_id: parent_id.clone(),
            timestamp,
            kind: kind.clone(),
            data: data.clone(),
        };
        append_line(&self.path, &entry)?;
        self.events.push(ExportEvent::Custom {
            kind: kind.clone(),
            data: data.clone(),
        });
        self.nodes.insert(
            id.clone(),
            Node {
                parent_id,
                content: NodeContent::Custom { kind, data },
                timestamp,
            },
        );
        self.active.push(id.clone());
        Ok(id)
    }

    /// Rewrite the whole file atomically (header + every message). Used when the message list was
    /// rewritten rather than extended — e.g. after compaction replaced the prefix with a summary.
    ///
    /// Only the *active path's own* entries are ever replaced: every node belonging to some other
    /// branch (created by navigating away and appending elsewhere — see [`Self::switch_active`])
    /// is preserved verbatim. The new active path gets a fresh linear chain of ids — a rewrite has no
    /// reliable way to know which, if any, of the new messages correspond to originals (compaction may
    /// summarize, split, or otherwise transform content) — written *after* the preserved nodes, so "the
    /// last message in the file" still resolves to the active tip with no `Leaf` marker needed, exactly
    /// like a plain flat file. See the module doc comment's compaction-vs-branching section.
    ///
    /// Every existing caller — `fork`/`fork_from_path`/`fork_at_entry` (all rewriting a store they just
    /// got back from [`SessionRepo::create`], so every id-keyed side index below starts empty anyway) and
    /// `rewrite_compacted`'s degenerate same-length fallback (rewriting *this* store in place, where
    /// `labels`/`model_changes`/`level_changes`/`title_changes`/`branch_summary_details`/`compactions`/
    /// `events` must legitimately survive since it's still the same session) — wants exactly this
    /// narrower reset. See [`Self::reset_for_new_session`] for the one caller that genuinely wants a
    /// blank slate.
    pub fn rewrite(&mut self, messages: &[Message]) -> std::io::Result<()> {
        self.rewrite_impl(messages, false)
    }

    /// Like [`rewrite`](Self::rewrite), but for single-file mode's `/new`-equivalent reset (pi-parity
    /// fix, pass 20): starting a genuinely brand-new session while reusing the same on-disk file/id,
    /// rather than repo mode's `SessionRepo::create` (which mints a fresh, already-empty
    /// [`SessionStore`]). A plain `rewrite(&[])` on an already-populated store clears the message tree
    /// but leaves every id-keyed side index untouched — `labels`, `model_changes`, `level_changes`,
    /// `title_changes`, `branch_summary_details`, the `Entry::Compaction` record map, and the flat
    /// `events` log all keep whatever the *discarded* session left in them. Two concretely observable
    /// leaks that fixed: (1) a stale `model_changes`/`level_changes` entry anchored at the tree root
    /// (`None`) makes `model_at_root()`/`thinking_level_at_root()` report the old session's model/level
    /// as if it were the new session's own change — reachable from a `switch_branch{before: true}` that
    /// resolves to root — silently misrouting every subsequent turn to the wrong provider; (2) `events`
    /// feeds `export_events`/`export_html` directly, so exporting the "new" session kept showing the old
    /// session's `ModelChange`/`ThinkingLevelChange`/`Label`/`Custom` log. This clears all of that, so the
    /// store afterward is indistinguishable — other than its file path/id — from one just returned by
    /// [`SessionRepo::create`].
    pub fn reset_for_new_session(&mut self) -> std::io::Result<()> {
        self.rewrite_impl(&[], true)
    }

    fn rewrite_impl(&mut self, messages: &[Message], full_reset: bool) -> std::io::Result<()> {
        // Compaction provenance: a rewrite that ends with *fewer* messages than were on disk dropped a
        // prefix (the compaction case), so record it in the header before writing. A same-length rewrite
        // (e.g. `set_title`) or a fresh fork (`persisted == 0`) drops nothing and records nothing.
        let dropped = self.persisted.saturating_sub(messages.len());
        if dropped > 0 {
            self.meta.compactions = self.meta.compactions.saturating_add(1);
            self.meta.dropped_messages = self.meta.dropped_messages.saturating_add(dropped as u64);
        }

        let old_active: HashSet<&str> = self.active.iter().map(String::as_str).collect();
        let preserved: Vec<(String, Node)> = self
            .nodes
            .iter()
            .filter(|(id, _)| !old_active.contains(id.as_str()))
            .map(|(id, node)| (id.clone(), node.clone()))
            .collect();

        let mut new_nodes: Vec<(String, Node)> = Vec::with_capacity(messages.len());
        let mut new_active = Vec::with_capacity(messages.len());
        let mut parent: Option<String> = None;
        for m in messages {
            let id = new_id();
            new_nodes.push((
                id.clone(),
                Node {
                    parent_id: parent.clone(),
                    content: NodeContent::Message(m.clone()),
                    timestamp: now_secs(),
                },
            ));
            new_active.push(id.clone());
            parent = Some(id);
        }

        let tmp = self.path.with_extension("jsonl.tmp");
        let mut f = create_private(&tmp)?;

        // Cleans up `tmp` if any step below returns early via `?` — a genuine in-process error (disk
        // full, a permission error mid-write), not a hard crash: the process is still alive here and can
        // just remove it, unlike the crash case (already safe on its own — a stray `.tmp` is never read
        // back as a session, see `read_listing`'s extension filter — and the next `rewrite` call reuses
        // this same deterministic path anyway, so leaving it behind was never a correctness hazard, just
        // litter). Disarmed once the rename actually succeeds, since `tmp` no longer exists under that
        // name by then.
        struct RemoveTmpOnError<'a> {
            path: &'a Path,
            armed: bool,
        }
        impl Drop for RemoveTmpOnError<'_> {
            fn drop(&mut self) {
                if self.armed {
                    let _ = fs::remove_file(self.path);
                }
            }
        }
        let mut cleanup = RemoveTmpOnError {
            path: &tmp,
            armed: true,
        };

        write_line(&mut f, &Entry::Session(self.meta.clone()))?;
        for (id, node) in preserved.iter().chain(new_nodes.iter()) {
            // `preserved` (off-active nodes carried through unchanged) can be either shape — a real
            // message or a custom entry (Track C-M2) that happened to live on some other branch;
            // `new_nodes` (the freshly compacted/forked active path) is always real messages, since
            // `messages: &[Message]` admits nothing else. Round-tripping each back through its own
            // original `Entry` variant, not unconditionally `Entry::Message`, is what keeps a preserved
            // custom entry from being silently corrupted into an empty/default message by a rewrite.
            match &node.content {
                NodeContent::Message(m) => write_line(
                    &mut f,
                    &Entry::Message {
                        id: Some(id.clone()),
                        parent_id: node.parent_id.clone(),
                        // Preserved nodes write back their own original timestamp — they're being
                        // physically relocated during the rewrite, not newly created, so re-stamping
                        // `now_secs()` here would make old content look freshly updated. Freshly
                        // constructed `new_nodes` above already carry their own real `now_secs()`.
                        timestamp: node.timestamp,
                        message: m.clone(),
                    },
                )?,
                NodeContent::Custom { kind, data } => write_line(
                    &mut f,
                    &Entry::Custom {
                        id: id.clone(),
                        parent_id: node.parent_id.clone(),
                        timestamp: node.timestamp,
                        kind: kind.clone(),
                        data: data.clone(),
                    },
                )?,
            }
        }
        // Sync the temp file's contents, then rename (atomic), then fsync the parent directory so the
        // rename itself is durable: without the dir fsync a crash could surface the old file — or, in the
        // window between, neither — even though the new bytes had reached disk.
        f.flush()?;
        f.sync_all()?;
        fs::rename(&tmp, &self.path)?;
        cleanup.armed = false;
        fsync_dir(&self.path)?;

        self.nodes = preserved.into_iter().collect();
        self.nodes.extend(new_nodes);
        self.active = new_active;
        self.persisted = messages.len();
        if full_reset {
            self.labels.clear();
            self.model_changes.clear();
            self.level_changes.clear();
            self.title_changes.clear();
            self.branch_summary_details.clear();
            self.compactions.clear();
            self.events.clear();
        }
        Ok(())
    }

    /// Like [`rewrite`](Self::rewrite), but for the compaction case specifically: the folded-away
    /// prefix is **preserved** on disk (by its original ids, still readable, listed in the new
    /// [`Entry::Compaction`] record's `folded_ids`) instead of being deleted — a compaction round
    /// becomes fully non-destructive and auditable, not just an aggregate-counter summary of what was
    /// lost. It becomes an inert, self-contained sub-chain off to the side (structurally the same as
    /// an abandoned branch), *not* reconnected to the new active path — `path_from_root` walks a
    /// whole parent chain to build the live session, so linking the two would resurrect every folded
    /// message back into the "active" transcript, undoing the compaction. `meta.tokens_before` is
    /// recorded alongside the new entry (the same value carried on `agent_core::AgentEvent::Compacted`).
    ///
    /// **O(1) append, not O(total preserved size)**: unlike a plain [`rewrite`](Self::rewrite) (which
    /// genuinely replaces the active path and so must be a full atomic swap), a compaction never touches
    /// anything already on disk — the folded prefix and every other branch are already durable exactly
    /// where they are, untouched. So this only ever *appends* the new entries (an updated header, the
    /// provenance record, the new active-path messages), the same append-only write `append_new` already
    /// does. A prior version of this method re-wrote the *entire* file (every preserved node) on every
    /// single call — since preserved content only ever grows, each subsequent compaction rewrote more
    /// bytes than the last, a compounding cost over a long session's lifetime that this avoids entirely.
    /// Crash-safety: if the process dies mid-append, the reader's existing torn/partial-line recovery
    /// (see [`Self::open`]) applies exactly as it already does to an interrupted [`append_new`] — and
    /// since `tip` only moves once the new active-path messages are actually read back, a compaction cut
    /// short mid-write simply doesn't take effect (the old tip, and everything it points to, is
    /// completely unaffected by an `Entry::Compaction` record with no messages after it yet).
    ///
    /// `messages` is the new active-path list — by construction (this is only ever called from the
    /// compaction path, right after `agent_core::compaction::apply_summary`) always exactly one
    /// summary message followed by the kept suffix verbatim. That invariant is what lets this method
    /// recover *which* original messages were folded purely from lengths, with no extra parameter:
    /// `old_persisted - (messages.len() - 1)` is the net shrinkage (one summary message took every
    /// folded message's place), so `+ 1` recovers the true folded count.
    pub fn rewrite_compacted(
        &mut self,
        messages: &[Message],
        meta: CompactionMeta,
    ) -> std::io::Result<()> {
        let dropped = self.persisted.saturating_sub(messages.len());
        if dropped == 0 {
            // Nothing was actually folded (a degenerate one-message-in-one-message-out round) — no
            // `Entry::Compaction` provenance is meaningful, so fall back to a plain rewrite.
            return self.rewrite(messages);
        }
        self.meta.compactions = self.meta.compactions.saturating_add(1);
        self.meta.dropped_messages = self.meta.dropped_messages.saturating_add(dropped as u64);

        // Walk from the front of `self.active` counting only *message*-bearing ids (a custom entry —
        // Track C-M2 — can sit interspersed on the active path without contributing to `self.persisted`,
        // so a plain positional slice by count would misalign once one exists); any custom id
        // encountered along the way is folded together with the messages around it, since it sat
        // structurally within the region compaction just summarized away. `target` message ids is
        // `dropped + 1` (one summary message replaces every folded message — see this method's doc
        // comment on why `+ 1`), capped so a pathological `dropped` can't run past the end of `active`.
        // Walk from the front of `self.active` counting only *message*-bearing ids (a custom entry —
        // Track C-M2 — can sit interspersed on the active path without contributing to `self.persisted`,
        // so a plain positional slice by count would misalign once one exists); any custom id
        // encountered along the way is folded together with the messages around it, since it sat
        // structurally within the region compaction just summarized away. `target` message ids is
        // `dropped + 1` (one summary message replaces every folded message — see this method's doc
        // comment on why `+ 1`), capped so a pathological `dropped` can't run past the end of `active`.
        let target = dropped.saturating_add(1);
        let mut folded_ids: Vec<String> = Vec::new();
        let mut folded_messages = 0usize;
        for id in &self.active {
            if folded_messages >= target {
                break;
            }
            folded_ids.push(id.clone());
            if self.nodes[id].as_message().is_some() {
                folded_messages += 1;
            }
        }
        // Everything on the active path past the fold point: the real kept messages (verbatim in
        // `messages[1..]`) and any custom entry (Track C-M2) interleaved among them. The fold loop
        // above only ever walks up to the fold boundary, so a custom entry positioned here was
        // previously neither recorded as folded provenance nor carried into the new chain below — it
        // just silently vanished. Carried forward instead, in its original relative position.
        let kept_suffix_ids: Vec<String> = self.active[folded_ids.len()..].to_vec();

        // Resolve the model/thinking-level/title effective at the old tip being folded away, before
        // it's gone. A change anchored exactly at the tip takes effect for whatever comes *next* (see
        // `record_model_change`'s anchor semantics), so check that anchor first, then fall back to
        // `change_at`'s ancestor walk (which deliberately excludes the query id's own anchor — the
        // opposite of what's wanted here). The new active path built below starts a fresh, detached
        // chain, so `path_from_root` from any of its ids can never reach back into `model_changes`/
        // `level_changes`/`title_changes` entries anchored on the now-folded chain — without
        // re-anchoring the already-resolved value onto the new chain's `None` baseline (the same
        // mechanism a change recorded before the session's very first message already uses), a
        // model/thinking-level switch (or, pass 15 pi-parity fix, a title resolved for a later fork) made
        // before this compaction becomes permanently unrecoverable.
        let old_tip = self.active.last().cloned();
        let effective_model = old_tip.as_deref().and_then(|tip| {
            self.model_changes
                .get(&Some(tip.to_string()))
                .cloned()
                .or_else(|| change_at(&self.nodes, &self.model_changes, tip).cloned())
        });
        let effective_level = old_tip.as_deref().and_then(|tip| {
            self.level_changes
                .get(&Some(tip.to_string()))
                .cloned()
                .or_else(|| change_at(&self.nodes, &self.level_changes, tip).cloned())
        });
        // Outer `Option` here means "something was recorded reaching this point" (mirroring
        // `effective_model`/`effective_level`'s own terseness discipline just below — nothing written
        // when nothing changed); the inner `Option<String>` is the title itself, `None` recording an
        // explicit clear that must survive the re-anchor exactly like a real title would, rather than
        // silently falling back to whatever rename preceded it on the folded chain.
        let effective_title: Option<Option<String>> = old_tip.as_deref().and_then(|tip| {
            self.title_changes
                .get(&Some(tip.to_string()))
                .cloned()
                .or_else(|| change_at(&self.nodes, &self.title_changes, tip).cloned())
        });

        let mut new_nodes: Vec<(String, Node)> =
            Vec::with_capacity(messages.len() + kept_suffix_ids.len());
        // The new active path starts a fresh, detached chain (`parent: None`), exactly like a plain
        // `rewrite` — *not* chained onto the last folded message. `path_from_root` walks a tip's whole
        // parent chain to build the live session, so linking back into the folded prefix would just
        // resurrect every folded message into the "active" transcript, defeating the point of
        // compacting them away. The folded prefix stays exactly where it already was on disk — its own
        // self-contained sub-chain, reachable by id and named in `folded_ids` below, structurally off
        // to the side, the same way an abandoned branch already is.
        let mut parent: Option<String> = None;
        let mut rest = messages.iter();
        // First node: the summary itself — synthesized fresh by compaction, with no counterpart on
        // `self.active` (it replaces the whole folded prefix, not any single node within it).
        if let Some(summary) = rest.next() {
            let id = new_id();
            new_nodes.push((
                id.clone(),
                Node {
                    parent_id: parent.clone(),
                    content: NodeContent::Message(summary.clone()),
                    timestamp: now_secs(),
                },
            ));
            parent = Some(id);
        }
        // The kept suffix: walk `self.active`'s own surviving ids in order, rather than `messages[1..]`
        // alone, so a custom entry among them rides along in its original relative position instead of
        // being silently dropped. A message-bearing id pulls the next verbatim `Message` (the two are
        // in lockstep by construction — see this method's doc comment); a custom id clones its content
        // unchanged.
        for id in &kept_suffix_ids {
            let content = match &self.nodes[id].content {
                NodeContent::Message(_) => match rest.next() {
                    Some(m) => NodeContent::Message(m.clone()),
                    None => unreachable!(
                        "kept_suffix_ids and messages[1..] must have the same message count"
                    ),
                },
                custom @ NodeContent::Custom { .. } => custom.clone(),
            };
            let new_node_id = new_id();
            new_nodes.push((
                new_node_id.clone(),
                Node {
                    parent_id: parent.clone(),
                    content,
                    timestamp: now_secs(),
                },
            ));
            parent = Some(new_node_id);
        }
        let new_active: Vec<String> = new_nodes.iter().map(|(id, _)| id.clone()).collect();

        // The summary text without its `SUMMARY_MARKER` wrapper, for the provenance record — reuses
        // `agent_core`'s own parser rather than re-deriving the same prefix-stripping logic here.
        let summary = messages
            .first()
            .and_then(|m| {
                agent_core::compaction::previous_summary(std::slice::from_ref(m))
                    .map(str::to_string)
            })
            .unwrap_or_default();
        let compaction_id = new_id();
        let compaction_parent_id = folded_ids.last().cloned();
        let compaction_entry = Entry::Compaction {
            id: compaction_id.clone(),
            parent_id: compaction_parent_id.clone(),
            tokens_before: meta.tokens_before,
            folded_ids: folded_ids.clone(),
            summary,
            read_files: meta.provenance.read_files,
            modified_files: meta.provenance.modified_files,
            last_reason: meta.provenance.last_reason,
            todos: meta.provenance.todos,
        };

        // An updated header snapshot (the new `compactions`/`dropped_messages` counters) first, then the
        // provenance record, then the new active path itself — `open`'s replay takes the *last*
        // `Entry::Session` line as the header (so appending a fresh one updates it in place) and treats
        // `Entry::Compaction` as inert for tip purposes, so "the last message in the file" still
        // resolves to the true tip with no `Leaf` marker needed.
        let mut buf = Vec::new();
        write_line(&mut buf, &Entry::Session(self.meta.clone()))?;
        write_line(&mut buf, &compaction_entry)?;
        // Re-anchor the resolved pre-compaction model/thinking-level onto the new chain's `None`
        // baseline — see the resolution comment above. Only written when something was actually
        // recorded on the folded chain (`record_model_change`'s own "call only when it actually
        // changed" discipline), so a session that never changed either stays exactly as terse as
        // before.
        if let Some(model) = &effective_model {
            write_line(
                &mut buf,
                &Entry::ModelChange {
                    id: new_id(),
                    parent_id: None,
                    model: model.clone(),
                },
            )?;
        }
        if let Some(level) = &effective_level {
            write_line(
                &mut buf,
                &Entry::ThinkingLevelChange {
                    id: new_id(),
                    parent_id: None,
                    level: level.clone(),
                },
            )?;
        }
        // Same re-anchoring, for a resolved title (pass 15 pi-parity fix) — including an explicit
        // clear (`None`, written as an empty string per `title_or_clear`'s own round-trip), which must
        // survive the re-anchor exactly like a real title would rather than reverting to whatever
        // earlier rename preceded it on the folded chain.
        if let Some(title) = &effective_title {
            write_line(
                &mut buf,
                &Entry::TitleChange {
                    id: new_id(),
                    parent_id: None,
                    title: title.clone().unwrap_or_default(),
                },
            )?;
        }
        for (id, node) in &new_nodes {
            // `new_nodes` can now hold either shape: a real message, or a custom entry (Track C-M2)
            // carried forward from the kept suffix — round-tripped through its own original `Entry`
            // variant, not unconditionally `Entry::Message`, mirroring `rewrite`'s own preserved-node
            // handling.
            match &node.content {
                NodeContent::Message(m) => write_line(
                    &mut buf,
                    &Entry::Message {
                        id: Some(id.clone()),
                        parent_id: node.parent_id.clone(),
                        timestamp: node.timestamp,
                        message: m.clone(),
                    },
                )?,
                NodeContent::Custom { kind, data } => write_line(
                    &mut buf,
                    &Entry::Custom {
                        id: id.clone(),
                        parent_id: node.parent_id.clone(),
                        timestamp: node.timestamp,
                        kind: kind.clone(),
                        data: data.clone(),
                    },
                )?,
            }
        }
        let mut f = OpenOptions::new().append(true).open(&self.path)?;
        f.write_all(&buf)?;
        // The parent dir is unchanged on an append (same inode, same dentry) — no directory fsync
        // needed, exactly like `append_new`.
        f.flush()?;
        f.sync_all()?;

        self.nodes.extend(new_nodes);
        self.active = new_active;
        self.persisted = messages.len();
        // So `tree()` reflects this round immediately, without requiring a reopen from disk first —
        // `open()`'s replay populates the same map from the just-written `compaction_entry` line, but
        // this is the same live `SessionStore` instance the caller is still holding (Track L25).
        self.compactions.insert(
            compaction_id,
            CompactionRecord {
                parent_id: compaction_parent_id,
                tokens_before: meta.tokens_before,
                folded_ids,
            },
        );
        if let Some(model) = effective_model {
            self.events.push(ExportEvent::ModelChange(model.clone()));
            self.model_changes.insert(None, model);
        }
        if let Some(level) = effective_level {
            self.events
                .push(ExportEvent::ThinkingLevelChange(level.clone()));
            self.level_changes.insert(None, level);
        }
        // No `ExportEvent` push here — `TitleChange` never participates in that stream even on the
        // ordinary `set_title` path (see its doc comment), so re-anchoring one during compaction stays
        // consistent with that.
        if let Some(title) = effective_title {
            self.title_changes.insert(None, title);
        }
        Ok(())
    }

    /// The messages that would become unreachable from the active tip if it switched to `target_id`
    /// right now — the suffix of the *current* active path after its deepest ancestor shared with
    /// `target_id`. Covers every case uniformly: `target_id` off the active path entirely (a sibling
    /// branch) abandons everything after their common ancestor; `target_id` an *ancestor* of the
    /// current tip (still on the active path, just not at the end) abandons everything after it —
    /// it's its own common ancestor with itself; `target_id` *equal to* the current tip is a no-op
    /// switch and abandons nothing (the "after" slice is empty). Empty (nothing abandoned) for an
    /// unknown `target_id` too — [`Self::switch_active`] will reject it, so there's nothing to
    /// summarize for a switch that's about to fail anyway.
    ///
    /// What a caller (Track L3's `switch_branch` RPC handler) feeds to
    /// `agent_core::branch_summary_request` *before* actually calling [`Self::switch_active`] — the
    /// messages would otherwise already be gone from `self.active` by the time anything could
    /// summarize them. Paired with each message's own id so a caller can also fold forward any nested
    /// branch summary's file-tracking via [`Self::branch_summary_details_within`] — a plain `Message`
    /// alone doesn't carry that structured data (see [`Entry::BranchSummary`]'s doc comment).
    pub fn abandoned_by_switch(&self, target_id: &str) -> Vec<(String, Message)> {
        let target_path = path_from_root(&self.nodes, Some(target_id));
        // A valid id's own path always includes at least itself; empty means `target_id` is unknown.
        if target_path.is_empty() {
            return Vec::new();
        }
        let target_ancestors: HashSet<&str> = target_path.iter().map(String::as_str).collect();
        // The deepest active-path id that's also an ancestor of (or equal to) `target_id` — their
        // common ancestor.
        let common_idx = self
            .active
            .iter()
            .rposition(|id| target_ancestors.contains(id.as_str()));
        let from = common_idx.map_or(0, |i| i + 1);
        // A custom entry (Track C-M2) among the abandoned range has no message content to summarize —
        // skip it, same as everywhere else a `Vec<Message>` is materialized from the tree.
        self.active[from..]
            .iter()
            .filter_map(|id| self.nodes[id].as_message().map(|m| (id.clone(), m.clone())))
            .collect()
    }

    /// Fold forward the `read_files`/`modified_files` of any [`Entry::BranchSummary`] whose id appears
    /// in `ids` (e.g. an [`Self::abandoned_by_switch`] range) — pi's own branch-summarization pass over
    /// nested detours: navigating *into* an abandoned branch via an earlier branch summary, then
    /// abandoning that branch too, would otherwise lose the earlier summary's file awareness the moment
    /// its prose-only materialized message is all that's left to scan (`extract_file_ops` only
    /// recognizes `read`/`write`/`edit` tool calls, not a summary's own metadata). Deduplicated and
    /// order-preserving, same as [`crate::compaction`]'s own file-op merging.
    pub fn branch_summary_details_within(&self, ids: &[String]) -> BranchSummaryDetails {
        let mut merged = BranchSummaryDetails::default();
        for id in ids {
            let Some(details) = self.branch_summary_details.get(id) else {
                continue;
            };
            for f in &details.read_files {
                if !merged.read_files.contains(f) {
                    merged.read_files.push(f.clone());
                }
            }
            for f in &details.modified_files {
                if !merged.modified_files.contains(f) {
                    merged.modified_files.push(f.clone());
                }
            }
        }
        merged
    }

    /// Every branch in the tree: one entry per leaf (a node with no children) *plus* the active tip
    /// itself, even when it isn't a leaf. The active tip can be an interior node — navigating to an
    /// ancestor of the original tip (rather than forking a genuinely new line from it) leaves it with
    /// a child (the abandoned continuation) — and it must still be reported: it's where the session
    /// currently *is*, which is exactly the thing a branch listing needs to show regardless of whether
    /// anything has forked from it yet. A session that's never branched has exactly one entry: the
    /// active tip. Order is by each tip's own `timestamp` (oldest first), falling back to leaf id when
    /// timestamps tie or are both the legacy `0` — matching pi's own chronological branch ordering
    /// while staying stable and deterministic for a client rendering a list.
    pub fn list_branches(&self) -> Vec<BranchInfo> {
        let parents: HashSet<&str> = self
            .nodes
            .values()
            .filter_map(|n| n.parent_id.as_deref())
            .collect();
        let active_tip = self.active.last().map(String::as_str);
        let mut leaf_ids: HashSet<&str> = self
            .nodes
            .keys()
            .map(String::as_str)
            .filter(|id| !parents.contains(id))
            .collect();
        if let Some(tip) = active_tip {
            leaf_ids.insert(tip);
        }
        let mut branches: Vec<BranchInfo> = leaf_ids
            .into_iter()
            .map(|id| {
                let path = path_from_root(&self.nodes, Some(id));
                let preview = path
                    .iter()
                    .rev()
                    .find_map(|mid| self.nodes[mid].as_message().and_then(first_user_text))
                    .map(preview_of);
                BranchInfo {
                    leaf_id: id.to_string(),
                    is_active: active_tip == Some(id),
                    message_count: path.len(),
                    preview,
                    timestamp: self.nodes[id].timestamp,
                }
            })
            .collect();
        branches.sort_by(|a, b| {
            a.timestamp
                .cmp(&b.timestamp)
                .then_with(|| a.leaf_id.cmp(&b.leaf_id))
        });
        branches
    }

    /// Every [`ExportEvent`] recorded in this session, in file order — Track L36 (pi-parity fix), for
    /// a caller (HTML export) that wants to surface a model/thinking-level switch, a label, or a custom
    /// entry as its own visible block, the same way [`Self::abandoned_branches`] exists so an export can
    /// render every branch's actual conversation instead of just the active path.
    pub fn export_events(&self) -> &[ExportEvent] {
        &self.events
    }

    /// The full root-to-leaf message chain of every abandoned branch (every leaf *except* the active
    /// tip) — the full-content counterpart of [`Self::list_branches`] (which carries only a preview),
    /// for a caller (HTML export) that wants to render every branch's actual conversation, not just
    /// the active path. Each entry is `(shared_prefix_len, messages)`: `messages` is the *whole*
    /// root-to-leaf chain, and `shared_prefix_len` is how many of its leading messages are identical
    /// (by id, positionally) to the active path's own leading messages — so a caller can render only
    /// the part that actually diverges, prefixed with a note of where it forked, rather than
    /// duplicating content already shown as the main transcript. Order is chronological by leaf
    /// timestamp, matching `list_branches`.
    pub fn abandoned_branches(&self) -> Vec<(usize, Vec<Message>)> {
        let parents: HashSet<&str> = self
            .nodes
            .values()
            .filter_map(|n| n.parent_id.as_deref())
            .collect();
        let active_tip = self.active.last().map(String::as_str);
        let mut leaf_ids: Vec<&str> = self
            .nodes
            .keys()
            .map(String::as_str)
            .filter(|id| !parents.contains(id) && Some(*id) != active_tip)
            .collect();
        leaf_ids.sort_by(|a, b| {
            self.nodes[*a]
                .timestamp
                .cmp(&self.nodes[*b].timestamp)
                .then_with(|| a.cmp(b))
        });
        leaf_ids
            .into_iter()
            .map(|id| {
                let path = path_from_root(&self.nodes, Some(id));
                // `shared` must be a valid message-count index into the *filtered* `messages` below
                // (and into the caller's own filtered active-path messages, which it's compared
                // against) — so a custom entry (Track C-M2) anywhere in the common prefix must not
                // inflate this count, exactly as it contributes nothing to either message list.
                let shared = path
                    .iter()
                    .zip(self.active.iter())
                    .take_while(|(a, b)| a == b)
                    .filter(|(mid, _)| self.nodes[mid.as_str()].as_message().is_some())
                    .count();
                let messages: Vec<Message> = path
                    .iter()
                    .filter_map(|mid| self.nodes[mid].as_message().cloned())
                    .collect();
                (shared, messages)
            })
            .collect()
    }

    /// Every node in the session's tree — every message on every branch, not just the active path
    /// [`BranchInfo`]/[`Self::list_branches`] surfaces only the leaves of. The `nodes` map already spans
    /// the whole file, so this is a single pass over it with no new indexing. Order is chronological by
    /// each node's own `timestamp`, falling back to id when timestamps tie or are both the legacy `0` —
    /// matching pi's own `SessionTreeNode` ordering (pi sorts each node's children by timestamp when
    /// rendering a tree) and letting a client reconstruct sibling/branch order without re-deriving it.
    pub fn tree(&self) -> Vec<TreeNode> {
        let mut nodes: Vec<TreeNode> = self
            .nodes
            .iter()
            .map(|(id, node)| {
                let (role, preview, entry_kind) = match &node.content {
                    // A materialized branch-summary recap is a real `NodeContent::Message` (so it
                    // actually reaches the model — see `branch_summary_message`), indistinguishable
                    // from an ordinary message by content alone; `branch_summary_details` is exactly
                    // the structured, id-keyed record of which nodes are actually recaps (Track L25).
                    NodeContent::Message(m) => {
                        let kind = if self.branch_summary_details.contains_key(id) {
                            "branch_summary"
                        } else {
                            "message"
                        };
                        (Some(m.role), message_text_preview(m), kind)
                    }
                    NodeContent::Custom { kind, .. } => {
                        (None, Some(format!("[custom: {kind}]")), "custom")
                    }
                };
                TreeNode {
                    id: id.clone(),
                    parent_id: node.parent_id.clone(),
                    role,
                    preview,
                    label: self.labels.get(id).cloned(),
                    timestamp: node.timestamp,
                    entry_kind,
                }
            })
            .collect();
        // `Entry::Compaction` records are never `nodes` (see `CompactionRecord`'s doc comment), so they
        // aren't covered by the map above at all — reported here as their own synthetic nodes instead,
        // purely for this listing; nothing about active-path/fork traversal changes.
        nodes.extend(self.compactions.iter().map(|(id, rec)| TreeNode {
            id: id.clone(),
            parent_id: rec.parent_id.clone(),
            role: None,
            preview: Some(format!(
                "[compaction: folded {} message(s), {} tokens before]",
                rec.folded_ids.len(),
                rec.tokens_before
            )),
            label: self.labels.get(id).cloned(),
            timestamp: 0,
            entry_kind: "compaction",
        }));
        nodes.sort_by(|a, b| a.timestamp.cmp(&b.timestamp).then_with(|| a.id.cmp(&b.id)));
        nodes
    }

    /// Every user-turn message anywhere in the tree — every branch, not just the active path
    /// [`Self::active_ids`]/live `Session.messages` covers — as `(id, Message)` pairs, chronologically
    /// ordered like [`Self::tree`] (by each node's own `timestamp`, falling back to id on a tie). Feeds
    /// `serve`'s `get_fork_messages`: pi's real `getUserMessagesForForking` returns candidates from the
    /// *whole* tree, not only whichever branch happens to be active right now — a client building a
    /// fork-point picker needs every message that could still be forked from, including ones on a
    /// branch the session already navigated away from. `Node::as_message` already excludes non-message
    /// nodes ([`Entry::Custom`]) the same way [`Self::tree`] does.
    pub fn all_user_messages(&self) -> Vec<(String, Message)> {
        let mut items: Vec<(String, Message)> = self
            .nodes
            .iter()
            .filter_map(|(id, node)| {
                let m = node.as_message()?;
                (m.role == Role::User).then(|| (id.clone(), m.clone()))
            })
            .collect();
        items.sort_by(|(a_id, _), (b_id, _)| {
            self.nodes[a_id]
                .timestamp
                .cmp(&self.nodes[b_id].timestamp)
                .then_with(|| a_id.cmp(b_id))
        });
        items
    }

    /// Switch the active branch to the message `target_id` — anywhere in the tree, on or off the
    /// current active path — persisting a `Leaf` marker so a later `open()` resolves the new tip.
    /// Returns the branch's materialized messages (root through `target_id`, root-first); the caller
    /// installs them as the live `Session.messages`. A later `append_new` against that returned slice
    /// naturally forks off `target_id` — it chains new messages off `self.active.last()`, which this
    /// sets to `target_id`. Errors (`NotFound`) if `target_id` names no known message.
    ///
    /// A no-op — no `Leaf` entry appended, nothing re-read — when `target_id` already *is* the active
    /// tip (pi-parity fix B-L2, `agent-session-tree-navigation.test.ts`'s "should handle navigation to
    /// same position (no-op)"): navigating to where the session already is shouldn't grow the file with
    /// a redundant marker every time a client re-confirms the current position.
    pub fn switch_active(&mut self, target_id: &str) -> std::io::Result<Vec<Message>> {
        if self.active.last().is_some_and(|id| id == target_id) {
            return Ok(self
                .active
                .iter()
                .filter_map(|id| self.nodes[id].as_message().cloned())
                .collect());
        }
        if !self.nodes.contains_key(target_id) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("no message with id {target_id} in this session"),
            ));
        }
        let leaf = Entry::Leaf {
            id: new_id(),
            parent_id: self.active.last().cloned(),
            target_id: Some(target_id.to_string()),
        };
        let mut buf = Vec::new();
        write_line(&mut buf, &leaf)?;
        let mut f = OpenOptions::new().append(true).open(&self.path)?;
        f.write_all(&buf)?;
        f.flush()?;
        f.sync_all()?;

        let active = path_from_root(&self.nodes, Some(target_id));
        let messages: Vec<Message> = active
            .iter()
            .filter_map(|id| self.nodes[id].as_message().cloned())
            .collect();
        self.persisted = messages.len();
        self.active = active;
        Ok(messages)
    }

    /// Switch to `target_id` *and* record a branch summary — an LLM-generated recap of the branch
    /// abandoned by navigating to `target_id` (Track L3's branch-navigation RPC handler generates
    /// `summary` via `agent_core::branch_summary_request` and calls this to record and apply it; this
    /// method only persists/applies the result, it doesn't call the model itself — same
    /// network-free/storage split as compaction).
    ///
    /// Unlike a plain [`switch_active`](Self::switch_active), the summary becomes part of the *new*
    /// active path: it's attached as a child of `target_id` and installed as the new tip, so it
    /// actually reaches the model on the next turn (the recap the summary was meant to preserve),
    /// rather than sitting on disk unreferenced by anything live. A later, unrelated
    /// `switch_active(target_id)` — no new abandonment — bypasses it (writes a plain `Leaf` straight
    /// to `target_id`); a second detour-and-return produces a sibling summary entry, not a duplicate.
    ///
    /// **O(1) append, not O(total tree size)**: the new entry's `parent_id` points at `target_id`,
    /// already durable on disk wherever it was originally written — tree structure is id-based, not
    /// file-position-based (see the module doc comment), so nothing that already exists needs to be
    /// touched, let alone rewritten. This only ever *appends* an updated header snapshot (so the new
    /// `branch_summaries`/`summarized_branch_messages` counters land in the same batch as the entry that
    /// caused them, never drifting out of sync even across a crash between them) and the
    /// [`Entry::BranchSummary`] record itself — the same append-only write [`Self::append_new`] already
    /// does. A prior version of this method rewrote the *entire* file (every node, every branch) on
    /// every call; since preserved content only ever grows, that cost compounded across a session's
    /// life the same way [`Self::rewrite_compacted`]'s equivalent rewrite used to (see its doc comment).
    pub fn switch_active_with_summary(
        &mut self,
        target_id: &str,
        summary: impl Into<String>,
        from_id: impl Into<String>,
        details: BranchSummaryDetails,
    ) -> std::io::Result<Vec<Message>> {
        if !self.nodes.contains_key(target_id) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("no message with id {target_id} in this session"),
            ));
        }
        self.meta.branch_summaries = self.meta.branch_summaries.saturating_add(1);
        self.meta.summarized_branch_messages = self
            .meta
            .summarized_branch_messages
            .saturating_add(details.summarized_messages);

        let summary = summary.into();
        let entry_id = new_id();
        let details_for_index = details.clone();
        let timestamp = now_secs();
        let entry = Entry::BranchSummary {
            id: entry_id.clone(),
            parent_id: Some(target_id.to_string()),
            summary: summary.clone(),
            from_id: from_id.into(),
            details,
            timestamp,
        };

        // The updated header first (so a fresh `open()`'s last-`Entry::Session`-wins replay picks it
        // up), then the summary entry — which becomes the new tip both on that fresh `open()` (the last
        // tip-setting entry in the file wins) and in this process's own in-memory state, updated below.
        let mut buf = Vec::new();
        write_line(&mut buf, &Entry::Session(self.meta.clone()))?;
        write_line(&mut buf, &entry)?;
        let mut f = OpenOptions::new().append(true).open(&self.path)?;
        f.write_all(&buf)?;
        // The parent dir is unchanged on an append (same inode, same dentry) — no directory fsync
        // needed, exactly like `append_new`.
        f.flush()?;
        f.sync_all()?;

        self.nodes.insert(
            entry_id.clone(),
            Node {
                parent_id: Some(target_id.to_string()),
                content: NodeContent::Message(branch_summary_message(&summary)),
                timestamp,
            },
        );
        self.branch_summary_details
            .insert(entry_id.clone(), details_for_index);
        self.active = path_from_root(&self.nodes, Some(&entry_id));
        let messages: Vec<Message> = self
            .active
            .iter()
            .filter_map(|id| self.nodes[id].as_message().cloned())
            .collect();
        self.persisted = messages.len();
        Ok(messages)
    }

    /// The immediate parent of `id` in the tree — `Some(None)` at the root (no parent), `Some(Some(p))`
    /// for any other node, or `None` if `id` names no known message. Lets a caller resolve a "switch to
    /// the position right before this entry" request (mirroring `SessionRepo::fork_at_entry`'s own
    /// `before: bool`) into either a normal target (a real parent) or the tree's own root, without
    /// touching `self.active` first.
    pub fn parent_of(&self, id: &str) -> Option<Option<String>> {
        self.nodes.get(id).map(|n| n.parent_id.clone())
    }

    /// Every message on the active path, paired with its own id — what a caller resetting to the
    /// tree's root (see [`Self::switch_active_to_root`]) is abandoning: the whole active path, since
    /// there's no target node to compute a common ancestor against (unlike
    /// [`Self::abandoned_by_switch`], which abandons only the suffix past wherever `target_id` and the
    /// active path last coincide). Empty when the session is already at its root.
    pub fn abandoned_to_root(&self) -> Vec<(String, Message)> {
        self.active
            .iter()
            .filter_map(|id| self.nodes[id].as_message().map(|m| (id.clone(), m.clone())))
            .collect()
    }

    /// Switch the active branch back to the tree's own root — before any message — so the very first
    /// message can be redone in place. A no-op (no `Leaf` entry appended) when already there, mirroring
    /// [`Self::switch_active`]'s identical same-position no-op. Pi's own `SessionManager::resetLeaf`.
    pub fn switch_active_to_root(&mut self) -> std::io::Result<Vec<Message>> {
        if self.active.is_empty() {
            return Ok(Vec::new());
        }
        let leaf = Entry::Leaf {
            id: new_id(),
            parent_id: self.active.last().cloned(),
            target_id: None,
        };
        let mut buf = Vec::new();
        write_line(&mut buf, &leaf)?;
        let mut f = OpenOptions::new().append(true).open(&self.path)?;
        f.write_all(&buf)?;
        f.flush()?;
        f.sync_all()?;

        self.persisted = 0;
        self.active = Vec::new();
        Ok(Vec::new())
    }

    /// Switch to the tree's root *and* record a branch summary of everything abandoned by doing so —
    /// the root-reset counterpart of [`Self::switch_active_with_summary`]. The summary becomes a new
    /// root message (its own `parent_id: None`) and the new active path, so the recap actually reaches
    /// the model on the next turn.
    pub fn switch_active_to_root_with_summary(
        &mut self,
        summary: impl Into<String>,
        from_id: impl Into<String>,
        details: BranchSummaryDetails,
    ) -> std::io::Result<Vec<Message>> {
        self.meta.branch_summaries = self.meta.branch_summaries.saturating_add(1);
        self.meta.summarized_branch_messages = self
            .meta
            .summarized_branch_messages
            .saturating_add(details.summarized_messages);

        let summary = summary.into();
        let entry_id = new_id();
        let details_for_index = details.clone();
        let timestamp = now_secs();
        let entry = Entry::BranchSummary {
            id: entry_id.clone(),
            parent_id: None,
            summary: summary.clone(),
            from_id: from_id.into(),
            details,
            timestamp,
        };

        let mut buf = Vec::new();
        write_line(&mut buf, &Entry::Session(self.meta.clone()))?;
        write_line(&mut buf, &entry)?;
        let mut f = OpenOptions::new().append(true).open(&self.path)?;
        f.write_all(&buf)?;
        f.flush()?;
        f.sync_all()?;

        self.nodes.insert(
            entry_id.clone(),
            Node {
                parent_id: None,
                content: NodeContent::Message(branch_summary_message(&summary)),
                timestamp,
            },
        );
        self.branch_summary_details
            .insert(entry_id.clone(), details_for_index);
        self.active = vec![entry_id];
        let messages: Vec<Message> = self
            .active
            .iter()
            .filter_map(|id| self.nodes[id].as_message().cloned())
            .collect();
        self.persisted = messages.len();
        Ok(messages)
    }

    /// Record that the active model changed to `model`, anchored to the current active tip — an O(1)
    /// append (see [`Entry::ModelChange`]'s doc comment), not a rewrite. Call only when the model
    /// actually changed (the caller compares against its own current value first) — recording a no-op
    /// change would just bloat the file with a redundant entry.
    ///
    /// Deliberately does **not** touch `self.meta.model` (Task #18, pi-parity investigation): an earlier
    /// version of this fix did exactly that, on the theory that `meta.model` staying frozen at the
    /// session's creation-time model was simply a staleness bug reaching `SessionRepo::fork`'s copied
    /// header and a future `run --continue` consumer. It regressed `Persistence::model_and_level_at`'s
    /// existing, load-bearing fallback contract instead — that lookup already relies on `meta.model`
    /// meaning specifically "the model in effect *before any change was ever recorded*", to resolve
    /// what a `switch_branch`/`switch_session`/`fork` target should restore when no `ModelChange` is
    /// found on the path to it (see that method's own doc comment: "every session has a real starting
    /// model"). Mutating `meta.model` here overwrites that baseline the moment the *first*
    /// `set_model` ever fires, so navigating back to a point that predates every recorded change then
    /// incorrectly resolves to whatever model is *currently* active instead of the session's real
    /// original one — caught by `serve_switch_branch_restores_the_model_active_on_that_branch`
    /// (`tests/serve_session_tree.rs`), which this exact regression broke.
    ///
    /// The actual bug this task named — `SessionRepo::fork`/`fork_at_entry`/`fork_from_path` copying
    /// `src.meta.model` verbatim into a forked session's own header, even when the fork point is well
    /// past the source's last recorded `set_model` — is instead fixed at the fork call sites themselves:
    /// each one resolves the correct "model active at the fork point" via the same per-branch
    /// `model_at`/`change_at` lookup this method's own `model_changes` map already supports, rather than
    /// reading `meta.model` at all. See [`SessionRepo::fork`]'s own doc comment.
    pub fn record_model_change(&mut self, model: &str) -> std::io::Result<()> {
        let anchor = self.active.last().cloned();
        let entry = Entry::ModelChange {
            id: new_id(),
            parent_id: anchor.clone(),
            model: model.to_string(),
        };
        append_line(&self.path, &entry)?;
        self.events
            .push(ExportEvent::ModelChange(model.to_string()));
        self.model_changes.insert(anchor, model.to_string());
        Ok(())
    }

    /// Same idea as [`Self::record_model_change`], for the portable thinking level (`level` is
    /// [`agent_core::ThinkingLevel::as_str`]'s wire string, e.g. `"high"`) — same deliberate
    /// non-mutation of `self.meta`, for the identical reason (see that method's doc comment).
    pub fn record_thinking_level_change(&mut self, level: &str) -> std::io::Result<()> {
        let anchor = self.active.last().cloned();
        let entry = Entry::ThinkingLevelChange {
            id: new_id(),
            parent_id: anchor.clone(),
            level: level.to_string(),
        };
        append_line(&self.path, &entry)?;
        self.events
            .push(ExportEvent::ThinkingLevelChange(level.to_string()));
        self.level_changes.insert(anchor, level.to_string());
        Ok(())
    }

    /// Seed a resolved thinking level onto this session's tree root — always anchored at `None`
    /// regardless of `self.active`'s current tip, unlike [`Self::record_thinking_level_change`]'s
    /// "wherever the tip currently is" anchor. What `SessionRepo::fork`/`fork_from_path`/
    /// `fork_at_entry` call, *after* [`Self::rewrite`] has already replaced the file's own bytes
    /// wholesale (a full temp-file-then-rename swap, not an append — see that method's own doc
    /// comment): calling `record_thinking_level_change` *before* `rewrite` would get silently
    /// discarded by that swap (`rewrite` only knows about `self.nodes`/the new message prefix, not
    /// this module's other side-channel append-only records), and calling it *after* would anchor at
    /// the new chain's own tip — which `change_at`'s deliberate "anchored-at, not before" exclusion
    /// (see that function's own doc comment) would then never see when a caller queries "the level at
    /// this same tip", exactly the query `Persistence::model_and_level_at_active` (`serve.rs`) makes
    /// on every reopen. Anchoring at the root instead means it's picked up by `change_at`'s own base
    /// case (`changes.get(&None)`, checked before any ancestor) for every id in the new tree, the same
    /// way `rewrite_compacted` already re-anchors a resolved model/level onto a fresh chain's `None`
    /// baseline.
    fn seed_thinking_level_at_root(&mut self, level: &str) -> std::io::Result<()> {
        let entry = Entry::ThinkingLevelChange {
            id: new_id(),
            parent_id: None,
            level: level.to_string(),
        };
        append_line(&self.path, &entry)?;
        self.events
            .push(ExportEvent::ThinkingLevelChange(level.to_string()));
        self.level_changes.insert(None, level.to_string());
        Ok(())
    }

    /// The model recorded as active at `target_id` — the most recent [`Entry::ModelChange`] anchored at
    /// `target_id` itself or any of its ancestors back to the root, if any were ever recorded on this
    /// branch. `None` means no change was ever recorded reaching this point — the caller should keep
    /// whatever model is already active (there's nothing branch-specific to restore).
    pub fn model_at(&self, target_id: &str) -> Option<&str> {
        change_at(&self.nodes, &self.model_changes, target_id).map(String::as_str)
    }

    /// Same idea as [`Self::model_at`], for the portable thinking level.
    pub fn thinking_level_at(&self, target_id: &str) -> Option<&str> {
        change_at(&self.nodes, &self.level_changes, target_id).map(String::as_str)
    }

    /// The model recorded as active at the tree's own root (before any message) — the base case every
    /// [`Self::model_at`] ancestor walk itself starts from (`changes.get(&None)`). What a caller needs
    /// when switching to root directly (`before: true` reaching the very first message) rather than to
    /// any real node — `model_at`/`thinking_level_at` require an existing id and can't express "root".
    pub fn model_at_root(&self) -> Option<&str> {
        self.model_changes.get(&None).map(String::as_str)
    }

    /// Same idea as [`Self::model_at_root`], for the portable thinking level.
    pub fn thinking_level_at_root(&self) -> Option<&str> {
        self.level_changes.get(&None).map(String::as_str)
    }

    /// The model that was actually active at `target_id` (or, when `None`, at the tree's own root) —
    /// what a fresh fork's own header should carry (Task #18, pi-parity fix), since forking doesn't
    /// carry `ModelChange` bookkeeping into the new file at all (see this struct's module doc and
    /// [`Self::record_model_change`]'s own doc comment on why blindly copying `meta.model` is wrong
    /// once even one `set_model` has happened past the fork point — or, for a fork landing *earlier*
    /// than the source's *first* `set_model`, wrong in the other direction). Falls back to
    /// `self.meta.model` (the session's true creation-time value) only when nothing was ever recorded
    /// reaching that point — the exact same "nothing changed on this branch yet" resolution
    /// `Persistence::model_and_level_at` (`serve.rs`) already applies for `switch_branch`/
    /// `switch_session`, just scoped to the one value a fork's own new header needs up front.
    fn model_at_or_created(&self, target_id: Option<&str>) -> &str {
        target_id
            .and_then(|id| self.model_at(id))
            .or_else(|| self.model_at_root())
            .unwrap_or(&self.meta.model)
    }

    /// The title in effect at `target_id` (or, when `None`, at the tree's own root) — the title
    /// analogue of [`Self::model_at_or_created`], for the same fork-header use (pass 15 pi-parity fix).
    /// Unlike model/thinking-level, there's no "session's own creation-time value" to fall back to: a
    /// title only ever exists because some [`Entry::TitleChange`] recorded one, so a fork landing at a
    /// point that never saw a rename on its own path — or whose most recent rename on that path
    /// explicitly cleared the title — simply gets no title at all (`None`), rather than inheriting
    /// `src.meta.title`'s whole-file-latest value the way this used to work. Matches pi's own
    /// `createBranchedSession`/`getEntriesToFork`, which only ever copy `session_info` entries
    /// physically present on the path being forked — a rename recorded on a different branch, or after
    /// the fork point, was never on that path to begin with.
    fn title_at_or_root(&self, target_id: Option<&str>) -> Option<&str> {
        target_id
            .and_then(|id| change_at(&self.nodes, &self.title_changes, id))
            .or_else(|| self.title_changes.get(&None))
            .and_then(|title| title.as_deref())
    }

    /// The thinking level that was actually active at `target_id` (or, when `None`, at the tree's own
    /// root) — the thinking-level analogue of [`Self::model_at_or_created`], for the same fork-header
    /// use. Mirrors its three-tier fallback exactly (an anchored change reaching `target_id`, then
    /// whatever was recorded at the tree's own root, then the session's own creation-time value) but
    /// returns `Option<&str>` rather than `&str`: unlike `meta.model` (always populated), `meta
    /// .thinking_level` is itself optional (`None` for a session that never called
    /// `set_reasoning_effort`/`cycle_thinking_level` at all — see that field's own doc comment), so a
    /// fork of one has no creation-time value to fall back to either and should carry none of its own,
    /// letting the forking process's own default apply exactly as it would for a brand-new session.
    ///
    /// Before this existed, none of `fork`/`fork_from_path`/`fork_at_entry_prefix` resolved the
    /// thinking level at all (unlike `model`/`title`, which already had their own pi-parity fixes for
    /// this) — a fork silently dropped whatever reasoning-effort level the source session had actually
    /// settled on, reverting to the forking process's bare default the moment the new session was
    /// reopened in a fresh process (`run --continue`, a `serve` restart, `switch_session` back to it).
    fn thinking_level_at_or_created(&self, target_id: Option<&str>) -> Option<&str> {
        target_id
            .and_then(|id| self.thinking_level_at(id))
            .or_else(|| self.thinking_level_at_root())
            .or(self.meta.thinking_level.as_deref())
    }

    /// Set (and persist) the session title — an O(1) append (see [`Entry::TitleChange`]'s doc
    /// comment), not a rewrite of the whole file. A rename used to cost a full rewrite (every message
    /// in the session) just to update the header's `title` field; renaming a long session is now as
    /// cheap as renaming a brand-new one.
    ///
    /// `title` is sanitized first (see [`sanitize_title`]): a caller-supplied title can come straight
    /// from an RPC client or extension, and a raw newline would otherwise split a session-list line or
    /// corrupt a terminal display. A title that sanitizes to empty explicitly clears the session's
    /// title (mirrors pi's `session_info` handling — "empty names explicitly clear the session title")
    /// rather than persisting a blank string that would render as an empty-but-present title.
    ///
    /// Also records the anchor-keyed `title_changes` entry (pass 15 pi-parity fix) that
    /// [`Self::title_at_or_root`] later resolves a fork's own title from — without this, a fork or
    /// compaction run against this same still-open store (rather than a freshly reopened one) would see
    /// an empty `title_changes` and never find this rename, exactly the gap `record_model_change`/
    /// `record_thinking_level_change` already close for `model_changes`/`level_changes`.
    pub fn set_title(&mut self, title: impl Into<String>) -> std::io::Result<()> {
        let title = sanitize_title(&title.into());
        let anchor = self.active.last().cloned();
        let entry = Entry::TitleChange {
            id: new_id(),
            parent_id: anchor.clone(),
            title: title.clone(),
        };
        append_line(&self.path, &entry)?;
        let resolved = title_or_clear(title);
        self.meta.title = resolved.clone();
        self.title_changes.insert(anchor, resolved);
        Ok(())
    }

    /// The label currently set on `target_id`, if any — the most recent [`Entry::Label`] recorded
    /// against it (last-write-wins), or `None` if it was never labeled or was last cleared. Pi's own
    /// `SessionManager.getLabel`.
    pub fn get_label(&self, target_id: &str) -> Option<&str> {
        self.labels.get(target_id).map(String::as_str)
    }

    /// Set (`label: Some`) or clear (`label: None`) a user-defined bookmark/marker on `target_id` — an
    /// O(1) [`Entry::Label`] append, pi's own `appendLabelChange` (`session-manager/labels.test.ts`).
    /// Errors (`NotFound`) if `target_id` names no known entry in this session, matching pi's own
    /// `Entry {id} not found`. See [`Entry::Label`]'s doc comment for why this never needs to rewire
    /// any other entry's `parent_id`, unlike pi's own version.
    pub fn set_label(&mut self, target_id: &str, label: Option<&str>) -> std::io::Result<()> {
        if !self.nodes.contains_key(target_id) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("no entry with id {target_id} in this session"),
            ));
        }
        let entry = Entry::Label {
            id: new_id(),
            parent_id: self.active.last().cloned(),
            target_id: target_id.to_string(),
            label: label.map(str::to_string),
        };
        append_line(&self.path, &entry)?;
        self.events.push(ExportEvent::Label {
            target_id: target_id.to_string(),
            label: label.map(str::to_string),
        });
        match label {
            Some(l) => {
                self.labels.insert(target_id.to_string(), l.to_string());
            }
            None => {
                self.labels.remove(target_id);
            }
        }
        Ok(())
    }

    /// Every recorded label whose target id appears in `ids`, as `(target_id, label)` pairs — the
    /// label analogue of [`Self::branch_summary_details_within`], used by
    /// [`SessionRepo::fork_at_entry`] to carry labels forward when their labeled entry survives into
    /// the forked prefix.
    fn labels_within(&self, ids: &[String]) -> Vec<(String, String)> {
        ids.iter()
            .filter_map(|id| self.labels.get(id).map(|l| (id.clone(), l.clone())))
            .collect()
    }
}

/// Collapse any run of `\r`/`\n` into a single space and trim, matching pi's `appendSessionInfo`
/// (`name.replace(/[\r\n]+/g, " ").trim()`) — a title is a single display line (session lists, HTML
/// export headers), so embedded newlines would otherwise corrupt that rendering.
fn sanitize_title(title: &str) -> String {
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
fn title_or_clear(title: String) -> Option<String> {
    if title.is_empty() { None } else { Some(title) }
}

/// Basic metadata for one entry sitting in a [`SessionRepo`]'s `.trash/` subdirectory — see
/// [`SessionRepo::list_trash`]. Deliberately minimal: enough to identify and restore an entry, not a
/// full trash-management surface (title/preview/message-count are all still reachable by restoring the
/// session and reading it normally).
#[derive(Debug, Clone, Serialize)]
pub struct TrashEntry {
    /// The session id (the same `<id>` component [`SessionRepo::find_path`]/`path_for` key off).
    pub id: String,
    /// When the entry was moved into `.trash/` (its file's own mtime, Unix seconds) — `None` if the
    /// filesystem couldn't report one.
    pub deleted_at: Option<u64>,
    /// Where [`SessionRepo::restore_session`] would move this entry back to — this repo's own
    /// directory, under the file's original name (not `.trash/`'s copy).
    pub original_path: String,
}

/// A directory of session files. `Clone` is just a `PathBuf` copy — cheap, and lets a caller move an
/// owned handle into a `spawn_blocking` closure (see `serve.rs`'s `Persistence::list_with_progress`)
/// without holding a borrow of the original across the `.await`.
#[derive(Clone)]
pub struct SessionRepo {
    dir: PathBuf,
}

impl SessionRepo {
    /// Open (creating if needed) a repository rooted at `dir`.
    pub fn open(dir: impl Into<PathBuf>) -> std::io::Result<Self> {
        let dir = dir.into();
        fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    /// The directory this repo is rooted at.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Reopen the most recent session whose recorded `cwd` matches, or create a fresh one — not just
    /// the globally newest session, so a shared repo directory spanning multiple projects doesn't
    /// resume a stranger's unrelated session. Matching is an exact string comparison — callers are
    /// expected to have already passed `cwd` through [`canonical_cwd`], so a symlinked or
    /// trailing-slashed spelling of the same real directory still matches (`serve`'s own startup
    /// reattach and `run --continue` both do). Shared by both, so they pick up "my last session in
    /// this directory" the same way. `id`, when given, names the fresh session in the no-match branch
    /// instead of a freshly generated one — a caller-chosen id (`serve`'s own `--session-id`); ignored
    /// when an existing session is reattached instead (already has a fixed id from disk). `run
    /// --continue` always passes `None` here, matching its own documented contract that `--session-id`
    /// applies only to a genuinely fresh `--session <path>` or a plain ephemeral run, never `--continue`.
    pub fn resume_or_create(
        &self,
        cwd: &str,
        model: &str,
        id: Option<&str>,
    ) -> std::io::Result<(SessionStore, Session)> {
        match self.list()?.into_iter().find(|m| m.cwd == cwd) {
            Some(meta) => self.open_id(&meta.id),
            None => {
                let meta = match id {
                    Some(id) => SessionMeta::with_id(id.to_string(), cwd, model),
                    None => SessionMeta::new(cwd, model),
                };
                let store = self.create(meta)?;
                Ok((store, Session::new()))
            }
        }
    }

    /// List every session across every project's own repo directory — each immediate subdirectory of
    /// `sessions_root` is treated as one project's [`SessionRepo`] (the convention `serve`'s default
    /// session directory follows: `<sessions_root>/<encoded-cwd>/`). Unlike [`Self::list`], which is
    /// scoped to one project, this is pi's cross-project `listAll`: each session's own `cwd` field
    /// (recorded at creation) already identifies which project it belongs to, so callers don't need the
    /// subdirectory name itself. A missing root is an empty list, not an error (nothing has ever been
    /// persisted there yet); a subdirectory that isn't a valid repo, or can't be read, contributes
    /// nothing rather than failing the whole scan — the same per-entry skip semantics [`Self::list`]
    /// already applies one level down, one level up.
    pub fn list_all(sessions_root: &Path) -> std::io::Result<Vec<SessionMeta>> {
        Self::list_all_with_progress(sessions_root, |_, _| {})
    }

    /// Same as [`Self::list_all`], but reports scan progress — see [`Self::list_with_progress`], whose
    /// parallel-scan strategy this shares: every project directory's `.jsonl` files are gathered up
    /// front into one flat list, then scanned together across one worker pool instead of one project
    /// (and its own pool) at a time, so `on_progress`'s `total` — and the parallelism — spans the whole
    /// root, not just whichever project happens to be scanning at a given moment.
    pub fn list_all_with_progress(
        sessions_root: &Path,
        on_progress: impl Fn(usize, usize) + Send + Sync,
    ) -> std::io::Result<Vec<SessionMeta>> {
        let entries = match fs::read_dir(sessions_root) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };
        let mut paths = Vec::new();
        for entry in entries {
            let Ok(entry) = entry else { continue };
            let project_dir = entry.path();
            if !project_dir.is_dir() {
                continue;
            }
            let Ok(project_entries) = fs::read_dir(&project_dir) else {
                continue;
            };
            paths.extend(
                project_entries
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("jsonl")),
            );
        }
        let mut metas = scan_listings(paths, &on_progress);
        metas.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(metas)
    }

    /// The on-disk path a session with this metadata would live at within this repo. `pub(crate)`: also
    /// used by [`fork_by_arg`] to locate a source session discovered via [`Self::list_all`] in some
    /// *other* project's own repo (a `SessionRepo` opened just long enough to compute the path, not to
    /// hold onto).
    pub(crate) fn path_for(&self, meta: &SessionMeta) -> PathBuf {
        self.dir
            .join(format!("{}_{}.jsonl", meta.created_at, meta.id))
    }

    /// Create a new, empty session and return its store.
    pub fn create(&self, meta: SessionMeta) -> std::io::Result<SessionStore> {
        SessionStore::create(self.path_for(&meta), meta)
    }

    /// All sessions' metadata, most-recently-active first (by `updated_at` — matches pi's own session
    /// list, sorted by `modified`). Each entry carries the derived listing fields (`updated_at`,
    /// `message_count`, `preview`, `search_text`). Files that fail to read, lack a header, or carry an
    /// unreadable version are skipped.
    pub fn list(&self) -> std::io::Result<Vec<SessionMeta>> {
        self.list_with_progress(|_, _| {})
    }

    /// Same as [`Self::list`], but invokes `on_progress(scanned, total)` once per file as the scan
    /// completes it (`total` known up front from the directory listing; `scanned` counts up
    /// monotonically to it, including files that turn out unreadable — those still count as scanned,
    /// just contribute nothing to the result). Each file's `read_listing` is pure disk I/O plus parsing
    /// with no cross-file dependency, so the scan fans out across a small worker pool instead of running
    /// one file at a time — a repo directory with hundreds of sessions (or, via
    /// [`Self::list_all_with_progress`], hundreds spread across many projects) scans in wall-clock time
    /// bounded by how many can run concurrently, not by their sum. A client can surface `on_progress` as
    /// a live "scanning…" indicator for a listing large enough to take a moment.
    pub fn list_with_progress(
        &self,
        on_progress: impl Fn(usize, usize) + Send + Sync,
    ) -> std::io::Result<Vec<SessionMeta>> {
        let mut paths = Vec::new();
        for entry in fs::read_dir(&self.dir)? {
            let path = entry?.path();
            if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
                paths.push(path);
            }
        }
        let mut metas = scan_listings(paths, &on_progress);
        metas.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(metas)
    }

    /// Open a session by id.
    pub fn open_id(&self, id: &str) -> std::io::Result<(SessionStore, Session)> {
        let path = self.find_path(id)?.ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, format!("no session {id}"))
        })?;
        SessionStore::open(path)
    }

    /// Delete a session by id. Idempotent per the repo invariant "check before destroy; don't error if
    /// it's gone": deleting an absent (or already-deleted) session is a successful no-op.
    ///
    /// Prefers a soft delete: moves the file into a `.trash` subdirectory rather than removing it
    /// outright, so a session deleted by mistake is still recoverable. `.trash` is itself a directory
    /// (no `.jsonl` extension), so [`Self::list`]'s flat, extension-filtered scan already excludes it
    /// with no further filtering needed. Falls back to a hard delete if the trash directory can't be
    /// created or the move fails for any reason (a read-only filesystem, a cross-device rename) —
    /// losing the undo, not the delete itself.
    pub fn delete(&self, id: &str) -> std::io::Result<()> {
        // An ambiguous prefix is a real error (surfaced to the caller so they can be more specific), not
        // treated as "nothing to delete" — only a genuine zero-match lookup is a no-op.
        let Some(path) = self.find_path(id)? else {
            return Ok(());
        };
        if let Some(file_name) = path.file_name() {
            let trash_dir = self.dir.join(".trash");
            if fs::create_dir_all(&trash_dir).is_ok()
                && fs::rename(&path, trash_dir.join(file_name)).is_ok()
            {
                return Ok(());
            }
        }
        match fs::remove_file(&path) {
            Ok(()) => Ok(()),
            // Raced with another deleter: the file vanished between `find_path` and `remove`. Still a
            // no-op success — the post-condition ("no session with this id") holds either way.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e),
        }
    }

    /// List every session currently sitting in this repo's `.trash/` subdirectory (see [`Self::delete`]),
    /// most-recently-deleted first — pi-parity gap (Fix 7): `delete` has always soft-deleted into
    /// `.trash/`, but nothing anywhere read, listed, restored from, or pruned it, so a mistaken delete
    /// was recoverable only by reaching for a shell. Deliberately minimal (id, when it was trashed, and
    /// where it would be restored to) rather than a full trash-management surface — a client wanting more
    /// (title/preview/message count) can always resolve `id` back through `--session`/`switch_session`'s
    /// own machinery once restored.
    ///
    /// Returns an empty list (not an error) when `.trash/` doesn't exist yet — nothing has ever been
    /// deleted here.
    pub fn list_trash(&self) -> std::io::Result<Vec<TrashEntry>> {
        let trash_dir = self.dir.join(".trash");
        let read_dir = match fs::read_dir(&trash_dir) {
            Ok(rd) => rd,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };
        let mut entries = Vec::new();
        for entry in read_dir {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let Some(rest) = name.strip_suffix(".jsonl") else {
                continue;
            };
            let Some((_, id)) = rest.split_once('_') else {
                continue;
            };
            // Best-effort: a filesystem that can't report an mtime (rare) just leaves this `None` rather
            // than failing the whole listing over one unreadable timestamp.
            let deleted_at = entry
                .metadata()
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs());
            entries.push(TrashEntry {
                id: id.to_string(),
                deleted_at,
                original_path: self.dir.join(name).to_string_lossy().into_owned(),
            });
        }
        entries.sort_by(|a, b| b.deleted_at.cmp(&a.deleted_at));
        Ok(entries)
    }

    /// Move `id` back out of this repo's `.trash/` subdirectory to its original location (see
    /// [`Self::delete`]/[`Self::list_trash`]). Exact-id match only (unlike `delete`'s own
    /// prefix-fallback via `find_path`) — a client restoring an id almost always has the exact one
    /// straight from `list_trash`'s own output, so the extra ambiguity surface isn't worth it here.
    ///
    /// `Ok(false)` when nothing in `.trash/` matches `id` — not found is not an error, matching
    /// `delete`'s own idempotent convention; unlike `delete`, though, there's no meaningful "already
    /// restored" no-op to collapse into `Ok(())`, so the caller must check this to know whether
    /// anything actually happened. Fails clearly (rather than silently overwriting) if a session already
    /// occupies the destination path — e.g. a new session was created with the same id after the delete.
    pub fn restore_session(&self, id: &str) -> std::io::Result<bool> {
        let trash_dir = self.dir.join(".trash");
        let read_dir = match fs::read_dir(&trash_dir) {
            Ok(rd) => rd,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(e) => return Err(e),
        };
        let exact_suffix = format!("_{id}.jsonl");
        let mut matched: Option<PathBuf> = None;
        for entry in read_dir {
            let path = entry?.path();
            if path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(&exact_suffix))
            {
                matched = Some(path);
                break;
            }
        }
        let Some(path) = matched else {
            return Ok(false);
        };
        // `path` was just built from a real directory entry's own file name, so this is always `Some`
        // in practice — but production code here stays panic-free regardless (workspace lint), so a
        // `None` (which should never happen) is a clear error instead of a panic.
        let Some(file_name) = path.file_name() else {
            return Err(std::io::Error::other(format!(
                "trash entry for {id} has no file name: {}",
                path.display()
            )));
        };
        let dest = self.dir.join(file_name);
        if dest.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!(
                    "cannot restore session {id}: {} already exists",
                    dest.display()
                ),
            ));
        }
        fs::rename(&path, &dest)?;
        Ok(true)
    }

    /// Fork session `id` at `upto` messages: a new session whose transcript is the first `upto`
    /// messages of the original, linked back via `parent`. `upto` is clamped to the source length, so
    /// `usize::MAX` clones the whole session. Returns the new store and its restored session.
    pub fn fork(&self, id: &str, upto: usize) -> std::io::Result<(SessionStore, Session)> {
        let (src, src_session) = self.open_id(id)?;
        let upto = upto.min(src_session.messages.len());
        // Task #18 (pi-parity fix): the model actually active at the copied prefix's own last message
        // — not `src.meta.model` (the source's creation-time value, blindly copied here previously) —
        // see `model_at_or_created`'s own doc comment for why that matters once a `set_model` has
        // happened anywhere relative to the fork point.
        let target_id = (upto > 0)
            .then(|| src.active_ids().get(upto - 1))
            .flatten()
            .map(String::as_str);
        let model = src.model_at_or_created(target_id).to_string();
        let mut meta = SessionMeta::new(src.meta.cwd.clone(), model);
        // The source's own resolved id, not the caller's raw `id` argument — since `open_id` now accepts
        // a unique prefix, blindly echoing `id` back would persist the *prefix* as `parent` instead of
        // the real full id it resolved to.
        meta.parent = Some(src.meta.id.clone());
        // Pass 15 (pi-parity fix): the title actually in effect at that same copied prefix's own last
        // message — not `src.meta.title` (the source's whole-file-latest rename, blindly copied here
        // previously) — see `title_at_or_root`'s own doc comment for why that matters once a rename has
        // happened anywhere relative to the fork point.
        meta.title = src.title_at_or_root(target_id).map(str::to_string);
        // Thinking-level analogue of the model/title fixes just above — see
        // `thinking_level_at_or_created`'s own doc comment.
        let thinking_level = src
            .thinking_level_at_or_created(target_id)
            .map(str::to_string);
        meta.thinking_level = thinking_level.clone();

        let mut store = self.create(meta)?;
        let prefix: Vec<Message> = src_session.messages[..upto].to_vec();
        store.rewrite(&prefix)?;
        // Re-anchor the resolved level onto the new session's own tree root — *after* `rewrite` above,
        // never before: `rewrite` is a full temp-file-then-rename swap of the whole file (see its own
        // doc comment), not an append, so anything appended earlier would just be silently discarded
        // by it. `seed_thinking_level_at_root` anchors at `None` (the tree's own root) rather than
        // `record_thinking_level_change`'s usual "wherever the tip currently is" — see that method's
        // own doc comment for why: anchoring at the new chain's own tip would put it exactly where
        // `change_at`'s "anchored-at, not before" exclusion never sees it once queried at that same
        // tip, the precise query `Persistence::model_and_level_at_active` (`serve.rs`) makes on every
        // reopen. Without this, a fresh reopen of the forked file (`run --continue`, a `serve`
        // restart, `switch_session` back to it) finds nothing recorded on the fork's brand-new chain
        // at all — the source's own `ThinkingLevelChange` entries are anchored to the *source*'s old
        // message ids, which don't exist in this file — and silently falls back to the process's own
        // starting level instead of the level actually in effect at the fork point.
        if let Some(level) = &thinking_level {
            store.seed_thinking_level_at_root(level)?;
        }
        let mut session = Session::new();
        session.messages = Arc::new(prefix);
        Ok((store, session))
    }

    /// Fork an arbitrary source session — one that need not live in `self.dir` at all — into a
    /// brand-new session under `self`. This is [`fork`](Self::fork) generalized for pi's cross-project
    /// `--fork <path|id>`: `source_path` is a fully resolved on-disk path (see [`fork_by_arg`], which
    /// finds it by searching the current project and then every other project's own session directory),
    /// and `target_cwd` becomes the new session's own `cwd` — the project being forked *into*, not
    /// wherever the source was originally recorded against — matching pi's own
    /// `SessionManager.forkFrom(path, targetCwd, …)`.
    pub fn fork_from_path(
        &self,
        source_path: &Path,
        target_cwd: &str,
        upto: usize,
    ) -> std::io::Result<(SessionStore, Session)> {
        let (src, src_session) = SessionStore::open(source_path.to_path_buf())?;
        let upto = upto.min(src_session.messages.len());
        // Task #18 (pi-parity fix): same reasoning as `fork`'s identical resolution just above.
        let target_id = (upto > 0)
            .then(|| src.active_ids().get(upto - 1))
            .flatten()
            .map(String::as_str);
        let model = src.model_at_or_created(target_id).to_string();
        let mut meta = SessionMeta::new(target_cwd.to_string(), model);
        meta.parent = Some(src.meta.id.clone());
        // Pass 15 (pi-parity fix): same reasoning as `fork`'s identical resolution just above.
        meta.title = src.title_at_or_root(target_id).map(str::to_string);
        // Same reasoning as `fork`'s identical resolution just above — see
        // `thinking_level_at_or_created`'s own doc comment.
        let thinking_level = src
            .thinking_level_at_or_created(target_id)
            .map(str::to_string);
        meta.thinking_level = thinking_level.clone();

        let mut store = self.create(meta)?;
        let prefix: Vec<Message> = src_session.messages[..upto].to_vec();
        store.rewrite(&prefix)?;
        // See `fork`'s identical re-anchoring for why this must happen after `rewrite` above, not
        // before.
        if let Some(level) = &thinking_level {
            store.seed_thinking_level_at_root(level)?;
        }
        let mut session = Session::new();
        session.messages = Arc::new(prefix);
        Ok((store, session))
    }

    /// Fork session `id` at an arbitrary tree entry `entry_id` — anywhere in the *whole* tree, on or
    /// off whatever the active path currently is, unlike [`fork`](Self::fork)'s active-path-only
    /// `upto` count. Mirrors pi's `createBranchedSession(leafId)`: `before` (pi's `position:"before"`)
    /// excludes `entry_id` itself from the forked prefix — forking right before a message the caller
    /// wants to redo, rather than keep; `false` (pi's `"at"`, the default) includes it. Errors
    /// (`NotFound`) if `entry_id` names no known message in `id`'s tree.
    ///
    /// Labels carry forward (pi-parity C-M1, `session-manager/labels.test.ts`'s "labels are preserved
    /// in createBranchedSession"): any label recorded in the source session against a message that
    /// survives into the forked prefix is re-applied in the new store, against that message's *new* id
    /// — [`rewrite`](Self::rewrite) mints fresh ids for a forked copy (it has to: see its own doc
    /// comment on why a rewrite can't in general reuse originals), so `fork_at_entry_prefix`'s
    /// `original_ids` (positionally parallel to the prefix `rewrite` just wrote) is what maps each old
    /// id to its replacement. A label whose target fell *outside* the forked prefix (off `entry_id`'s
    /// path, or past it when `before` excludes it) has no surviving id to re-attach to and is simply
    /// dropped, matching pi's "labels not on path are not preserved."
    ///
    /// Custom entries (Track C-M2, [`Entry::Custom`]) on the forked path are **not** carried into the
    /// new session — the copied prefix is a plain `Vec<Message>` fed through the ordinary
    /// [`rewrite`](Self::rewrite), which only ever produces message nodes; a custom entry's opaque
    /// `data` has no message representation to carry forward. This matches this method's existing
    /// precedent for every *other* side-channel entry type (`ModelChange`/`ThinkingLevelChange`/
    /// `BranchSummary` provenance): forking already only ever preserves the message content itself,
    /// never the surrounding bookkeeping — labels are carried forward as a deliberate, narrower
    /// exception (see above), not a general rule this extends.
    pub fn fork_at_entry(
        &self,
        id: &str,
        entry_id: &str,
        before: bool,
    ) -> std::io::Result<(SessionStore, Session)> {
        let prefix = self.fork_at_entry_prefix(id, entry_id, before)?;
        let mut store = self.create(prefix.meta)?;
        store.rewrite(&prefix.messages)?;
        // See `fork`'s identical re-anchoring for why this must happen after `rewrite` above, not
        // before — anchors at the tree's own root rather than the source's now-nonexistent message
        // ids.
        if let Some(level) = &prefix.thinking_level {
            store.seed_thinking_level_at_root(level)?;
        }
        if !prefix.labels.is_empty() {
            let new_ids: Vec<String> = store.active_ids().to_vec();
            debug_assert_eq!(new_ids.len(), prefix.original_ids.len());
            let old_to_new: HashMap<&str, &str> = prefix
                .original_ids
                .iter()
                .map(String::as_str)
                .zip(new_ids.iter().map(String::as_str))
                .collect();
            for (target_id, label) in &prefix.labels {
                if let Some(&new_id) = old_to_new.get(target_id.as_str()) {
                    store.set_label(new_id, Some(label))?;
                }
            }
        }
        let mut session = Session::new();
        session.messages = Arc::new(prefix.messages);
        Ok((store, session))
    }

    /// Compute what [`fork_at_entry`](Self::fork_at_entry) would write, without writing it: the
    /// would-be child's metadata, message prefix, the source's own ids for that prefix (positionally
    /// parallel — see [`ForkPrefix::original_ids`]), and any labels found within it. Pure/in-memory
    /// beyond the initial `open_id` read — no file is created. Used both by `fork_at_entry` itself (to
    /// avoid duplicating the prefix logic) and by [`fork_at_entry_messages`](Self::fork_at_entry_messages)
    /// for side-effect-free previews.
    ///
    /// Investigated (Round 2 pi-parity remediation, Low severity, left unfixed): `entry_id` must name a
    /// real tree node (a key in `nodes` — a `Message`/`BranchSummary`/`Custom` entry), so it 404s for
    /// one of the anchored side-channel entries (`Entry::Label`/`ModelChange`/`ThinkingLevelChange`/
    /// `TitleChange`) below `nodes.contains_key` — those deliberately never occupy a tree slot at all
    /// (see [`Entry::Label`]'s own doc comment on why). Pi's own uniform tree model allows forking at
    /// any entry id, including these; beyond's narrower one doesn't.
    ///
    /// Assessed as confirmed-real but disproportionate to fix, the same judgment call already made for
    /// the abandoned-branch export-nesting edge case (see `orphaned_off_branch_survives_compaction_of_its_ancestor`'s
    /// sibling test, "confirms Task #34"): no known workflow needs this, and — more than
    /// "unconfirmed" — it's provably unreachable through any surface this crate currently exposes.
    /// [`Self::tree`] (a client's only way to discover an entry id worth forking at) synthesizes
    /// `TreeNode`s solely from `nodes` and `self.compactions`; a label is surfaced only as the `label`
    /// *field* of the node it targets, never as its own addressable entry, and `ModelChange`/
    /// `ThinkingLevelChange`/`TitleChange` aren't surfaced as tree nodes at all. So no client can ever
    /// have actually *seen* one of these entries' own ids to pass back in here in the first place —
    /// there is no live caller this gap could affect today. Properly supporting it would mean first
    /// giving each of these four variants its own discoverable identity in `tree()` (a real new surface,
    /// not a bug fix) and then reworking this method's resolution to translate such an id to the
    /// nearest real tree position — a meaningful reshaping of the entry model for a target no code path
    /// can currently reach. Left as documented, deliberately unfixed.
    fn fork_at_entry_prefix(
        &self,
        id: &str,
        entry_id: &str,
        before: bool,
    ) -> std::io::Result<ForkPrefix> {
        let (src, _src_session) = self.open_id(id)?;
        if !src.nodes.contains_key(entry_id) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("no message with id {entry_id} in session {id}"),
            ));
        }
        // `before` (now the default — see `serve.rs`'s `fork`/`preview_fork` handlers) means "fork
        // right before this entry", which only makes sense anchored to a user turn: popping a
        // non-user entry (an assistant reply, a materialized branch-summary recap, or a custom entry)
        // would silently land the fork one entry earlier than the caller actually asked for, with no
        // way for them to tell. pi's own `getEntriesToFork` (`repo-utils.ts`) rejects the same case as
        // `SessionError("invalid_fork_target")` — mirrored here rather than guessing.
        if before {
            let is_user_message = src
                .nodes
                .get(entry_id)
                .and_then(Node::as_message)
                .is_some_and(|m| m.role == Role::User);
            if !is_user_message {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("invalid_fork_target: entry {entry_id} is not a user message"),
                ));
            }
        }

        let mut path = path_from_root(&src.nodes, Some(entry_id));
        if before {
            path.pop();
        }
        // Task #18 (pi-parity fix): the model actually active at the resolved fork point (`path`'s own
        // last id post-`before`-adjustment, or the tree's root if `before` popped the very first entry)
        // — not `src.meta.model` (the source's creation-time value) — see `model_at_or_created`'s own
        // doc comment. Computed *after* the `before` adjustment above, since that's what determines the
        // actual point being forked at.
        let model = src
            .model_at_or_created(path.last().map(String::as_str))
            .to_string();
        let mut meta = SessionMeta::new(src.meta.cwd.clone(), model);
        // The source's own resolved id, not the caller's raw `id` argument — see `fork`'s identical fix
        // for why: `open_id` now accepts a unique prefix, so `id` itself may not be the real full id.
        meta.parent = Some(src.meta.id.clone());
        // Pass 15 (pi-parity fix): same reasoning as `fork`'s identical resolution — the title actually
        // in effect at the resolved fork point, not `src.meta.title`'s whole-file-latest rename.
        meta.title = src
            .title_at_or_root(path.last().map(String::as_str))
            .map(str::to_string);
        // Same reasoning as `fork`'s identical resolution — the thinking level actually in effect at
        // the resolved fork point. See `thinking_level_at_or_created`'s own doc comment.
        let thinking_level = src
            .thinking_level_at_or_created(path.last().map(String::as_str))
            .map(str::to_string);
        meta.thinking_level = thinking_level.clone();

        // Labels are looked up against the *full* path (including any custom entry's id) — a label
        // whose target is a custom entry that then gets filtered out below simply won't be found in
        // `fork_at_entry`'s old-id → new-id map, and is correctly dropped the same way an
        // off-path label already is. `original_ids`/`messages` below, by contrast, must stay message-
        // only and positionally parallel to each other: `rewrite` (called next, on `messages`) only
        // ever produces message nodes, so a custom entry has no counterpart id in the destination
        // store to carry a label forward to anyway (Track C-M2 forks don't carry custom entries
        // themselves — see `SessionRepo::fork_at_entry`'s doc comment).
        let labels = src.labels_within(&path);
        let mut original_ids = Vec::with_capacity(path.len());
        let mut messages = Vec::with_capacity(path.len());
        for id in &path {
            if let Some(m) = src.nodes[id].as_message() {
                original_ids.push(id.clone());
                messages.push(m.clone());
            }
        }
        Ok(ForkPrefix {
            meta,
            messages,
            original_ids,
            labels,
            thinking_level,
        })
    }

    /// Preview what [`fork_at_entry`](Self::fork_at_entry) would produce — the exact message prefix —
    /// without creating a new session file. The read-only counterpart to `fork_at_entry`, for a client
    /// browsing a fork point before committing to it (see `fork_messages` in `serve.rs`).
    pub fn fork_at_entry_messages(
        &self,
        id: &str,
        entry_id: &str,
        before: bool,
    ) -> std::io::Result<Vec<Message>> {
        Ok(self.fork_at_entry_prefix(id, entry_id, before)?.messages)
    }

    /// Resolve `id` to its on-disk path in this repo: an exact match first (cheap, unambiguous), then a
    /// unique-prefix match — pi's own convenience for typing a shortened id (`main.ts`'s
    /// `resolveSessionPath`), but *not* pi's own silent "pick whichever sorts first" when a prefix
    /// matches more than one session: that's a real footgun (operating on session B when the caller
    /// typed a prefix meaning to reach session A), so an ambiguous prefix is an error naming every
    /// candidate instead of a guess. `Ok(None)` when nothing matches at all — not found is not an error
    /// here, matching every existing caller's own "session may not exist" handling.
    fn find_path(&self, id: &str) -> std::io::Result<Option<PathBuf>> {
        let entries: Vec<PathBuf> = match fs::read_dir(&self.dir) {
            Ok(entries) => entries.flatten().map(|e| e.path()).collect(),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e),
        };
        let exact_suffix = format!("_{id}.jsonl");
        if let Some(path) = entries.iter().find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(&exact_suffix))
        }) {
            return Ok(Some(path.clone()));
        }
        // No exact match: fall back to a unique-prefix match over each file's own `<id>` component
        // (`split_once` on the *first* underscore only, since `<created_at>` is always plain digits —
        // never itself containing one — while a caller-supplied `--session-id` legally can).
        let matches: Vec<(&str, &PathBuf)> = entries
            .iter()
            .filter_map(|path| {
                let name = path.file_name()?.to_str()?;
                let rest = name.strip_suffix(".jsonl")?;
                let (_, file_id) = rest.split_once('_')?;
                file_id.starts_with(id).then_some((file_id, path))
            })
            .collect();
        match matches.as_slice() {
            [] => Ok(None),
            [(_, path)] => Ok(Some((*path).clone())),
            _ => {
                let mut ids: Vec<&str> = matches.iter().map(|(id, _)| *id).collect();
                ids.sort_unstable();
                Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("\"{id}\" matches more than one session: {}", ids.join(", ")),
                ))
            }
        }
    }
}

/// Canonicalize `cwd` for session-matching purposes: resolves symlinks and `.`/`..` components (and, as
/// a side effect, any trailing separator) so two different-but-equivalent spellings of the same real
/// directory — a project reached through a symlink one time and its real path another, or a caller that
/// happens to include a trailing `/` — match the same session instead of silently fragmenting into two.
/// Falls back to `cwd` unchanged if it can't be resolved (removed out from under the process, a
/// permission error): matching degrades to today's exact-string comparison, never a new failure mode.
/// Every path this module ever records into or matches against a [`SessionMeta::cwd`] should be passed
/// through this first — `serve`'s own startup cwd and `run --continue`'s both do.
pub fn canonical_cwd(cwd: &Path) -> PathBuf {
    cwd.canonicalize().unwrap_or_else(|_| cwd.to_path_buf())
}

/// Encode `cwd` into a filesystem-safe directory-name component: every path separator becomes `-`, so
/// `/home/jared/ai` becomes `-home-jared-ai` — the same convention this repo's other per-project state
/// already uses (`trust_store.rs`'s `~/.claude/trusted-projects.json`, `prompts.rs`'s
/// `~/.claude/prompts`), extended here to give each project its own session subdirectory.
pub fn encode_cwd(cwd: &str) -> String {
    cwd.chars()
        .map(|c| if c == '/' || c == '\\' { '-' } else { c })
        .collect()
}

/// The root every project's own session directory lives under: `~/.claude/sessions/`. Each immediate
/// subdirectory is one project's `<encoded-cwd>/` (see [`default_session_dir`]) — this is the root
/// [`SessionRepo::list_all`] scans for pi's cross-project session search (`--fork <id>` when the id
/// isn't found in the current project — see [`fork_by_arg`]).
pub fn sessions_root() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_default();
    home.join(".claude/sessions")
}

/// The default session directory for `cwd` when nothing more specific was given: `~/.claude/sessions/
/// <encoded-cwd>/`, one subdirectory per project so unrelated projects' sessions never mix in the same
/// listing. Shared by `serve`'s own default (`--session-file`/`--session-dir` absent) and `run
/// --continue`.
pub fn default_session_dir(cwd: &str) -> PathBuf {
    sessions_root().join(encode_cwd(cwd))
}

/// Whether a `--fork`/`--session` argument looks like a path rather than a bare session id — pi's own
/// `resolveSessionPath` treats any of these as "don't search, just resolve this as a file": a path
/// separator, a leading `.`/`~`, or a `.jsonl` extension. A session id (`new_id()`'s shape, or a
/// caller-supplied `--session-id`) never looks like this, so there's no realistic ambiguity. `pub`
/// (crosses the `main.rs` binary/lib boundary): shared with `run --session <arg>`'s own resolution
/// (Task #24), which needs the identical literal-path-or-bare-id classification [`fork_by_arg`] already
/// applies for `--fork <arg>`.
pub fn is_path_like(arg: &str) -> bool {
    arg.contains('/') || arg.starts_with('.') || arg.starts_with('~') || arg.ends_with(".jsonl")
}

/// Resolve a path-like `--fork`/`--session` argument against `cwd`: `~`/`~/...` expands to the home
/// directory (reusing the same convention `--skill`/`--prompt-template` extra paths already get via
/// `tools::expand_tilde`), and a relative path resolves against `cwd` — matching pi's own
/// `resolvePath(arg, cwd)`. An already-absolute path (post tilde-expansion) is returned unchanged.
/// Existence is deliberately not checked here — the caller's eventual `SessionStore::open` surfaces a
/// clear not-found error if it doesn't exist, consistent with every other path this crate resolves
/// lazily rather than pre-validating.
fn resolve_path_like(arg: &str, cwd: &str) -> PathBuf {
    let expanded = crate::tools::expand_tilde(arg, std::env::var("HOME").ok().as_deref());
    let path = PathBuf::from(expanded);
    if path.is_absolute() {
        path
    } else {
        Path::new(cwd).join(path)
    }
}

/// Resolve a `--fork <arg>` argument and fork it into a brand-new session under `target`'s own repo —
/// pi's own cross-project `--fork <path|id>` (`SessionManager.forkFrom`/`resolveSessionPath`). `arg` may
/// be:
/// - a direct path to a `.jsonl` file (any project — see [`is_path_like`]/[`resolve_path_like`]),
/// - a session id that exists in `target`'s own project (forked in place via [`SessionRepo::fork`],
///   identical to today's same-project fork),
/// - a session id that exists in some *other* project's own directory under `sessions_root` (found via
///   [`SessionRepo::list_all`], then forked cross-project via [`SessionRepo::fork_from_path`], with
///   `target_cwd` — the project being forked *into* — becoming the new session's own `cwd`, not
///   wherever the source was originally recorded against).
///
/// Prefix matching is intentionally out of scope here (a bare id must match exactly) — see the
/// partial-session-id-resolution finding this crate tracks separately; this only ever resolves an exact
/// id or an explicit path.
pub fn fork_by_arg(
    arg: &str,
    target: &SessionRepo,
    target_cwd: &str,
    sessions_root: &Path,
    upto: usize,
) -> std::io::Result<(SessionStore, Session)> {
    if is_path_like(arg) {
        let path = resolve_path_like(arg, target_cwd);
        return target.fork_from_path(&path, target_cwd, upto);
    }
    // Try the current project first — `fork`'s own `open_id` is exact-then-unique-prefix aware (see
    // `SessionRepo::find_path`), so this alone already covers a prefix that resolves within this
    // project. An ambiguous prefix here is a real error to surface immediately, not a reason to widen
    // the search to other projects (that would risk silently reinterpreting the same prefix against a
    // *different* set of candidates instead of just reporting the one ambiguity the caller already hit).
    match target.fork(arg, upto) {
        Ok(result) => return Ok(result),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }
    match find_session_path_under(sessions_root, arg)? {
        Some(source_path) => target.fork_from_path(&source_path, target_cwd, upto),
        None => Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("no session matching \"{arg}\" in this project or any other"),
        )),
    }
}

/// Resolve a bare session id (or unique prefix) for `run --session <arg>` (Task #24) and reopen it *in
/// place* — pi's own `resolveSessionPath`'s "local"/"global" branches (`main.ts`), except a cross-project
/// match is reopened directly rather than routed through pi's own interactive fork-confirmation prompt:
/// this crate's `run` is headless, with no TTY-prompt path anywhere else to hang that on. Only ever
/// consulted once the caller has already ruled out a literal path (`is_path_like`, or one that already
/// exists on disk as-is) — unlike [`fork_by_arg`], which this otherwise mirrors tier-for-tier, `arg`
/// here is never treated as a path at all. Tries `target`'s own project first (`SessionRepo::open_id`,
/// exact-then-unique-prefix — the same first tier `--fork <id>` uses), then, on a genuine not-found,
/// every other project's own directory under `sessions_root` (`find_session_path_under`, `--fork`'s
/// identical second tier). `Err(NotFound)` naming `arg` when nothing matches anywhere — unlike a literal
/// path (which `--session` creates fresh when absent), a bare identifier that resolves to nothing is a
/// likely typo, not a request to start a new session named after it.
pub fn open_session_by_id(
    arg: &str,
    target: &SessionRepo,
    sessions_root: &Path,
) -> std::io::Result<(SessionStore, Session)> {
    match target.open_id(arg) {
        Ok(result) => Ok(result),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            match find_session_path_under(sessions_root, arg)? {
                Some(path) => SessionStore::open(path),
                None => Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("no session matching \"{arg}\" found as a path or session id"),
                )),
            }
        }
        Err(e) => Err(e),
    }
}

/// Locate a session matching `id` somewhere under `sessions_root`, by walking each immediate
/// subdirectory the same way [`SessionRepo::list_all_with_progress`] does and checking filenames
/// directly (`<created_at>_<id>.jsonl`) — deliberately *not* by recomputing a path from the session's
/// own recorded `cwd` via `default_session_dir`/`encode_cwd`. A caller-supplied `--session-dir`/
/// `AI_AGENT_SESSION_DIR` override need not follow that naming convention at all (it may be an
/// arbitrarily-named shared directory), so reconstructing a path from `cwd` would silently miss a real
/// session sitting right there under `sessions_root`. Cheaper than a full [`SessionRepo::list_all`] scan
/// too: this only ever inspects filenames, never a file's contents. `pub` (crosses the `main.rs`
/// binary/lib boundary): shared with [`open_session_by_id`]'s own cross-project fallback (Task #24).
pub fn find_session_path_under(sessions_root: &Path, id: &str) -> std::io::Result<Option<PathBuf>> {
    let project_dirs: Vec<PathBuf> = match fs::read_dir(sessions_root) {
        Ok(entries) => entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    let exact_suffix = format!("_{id}.jsonl");
    let all_files: Vec<PathBuf> = project_dirs
        .iter()
        .filter_map(|d| fs::read_dir(d).ok())
        .flat_map(|entries| entries.flatten().map(|e| e.path()))
        .collect();
    if let Some(path) = all_files.iter().find(|p| {
        p.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with(&exact_suffix))
    }) {
        return Ok(Some(path.clone()));
    }
    // No exact match anywhere: fall back to a unique-prefix match spanning every project — see
    // `SessionRepo::find_path`'s identical reasoning (ambiguous is an error naming every candidate, not
    // pi's own silent most-recent-wins).
    let matches: Vec<(&str, &PathBuf)> = all_files
        .iter()
        .filter_map(|path| {
            let name = path.file_name()?.to_str()?;
            let rest = name.strip_suffix(".jsonl")?;
            let (_, file_id) = rest.split_once('_')?;
            file_id.starts_with(id).then_some((file_id, path))
        })
        .collect();
    match matches.as_slice() {
        [] => Ok(None),
        [(_, path)] => Ok(Some((*path).clone())),
        _ => {
            let mut ids: Vec<&str> = matches.iter().map(|(id, _)| *id).collect();
            ids.sort_unstable();
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("\"{id}\" matches more than one session: {}", ids.join(", ")),
            ))
        }
    }
}

/// Validate — and, in a future format, migrate — a header to the current [`VERSION`]. Today there is
/// exactly one readable shape; a header whose version is *newer* than this build understands means we'd
/// be guessing at fields we don't know, so we refuse it loudly rather than mis-parse. The `match` is the
/// extension point: a later bump keeps the newer-version guard and adds an arm per older version that
/// upgrades it in place.
fn migrate(meta: SessionMeta, path: &Path) -> std::io::Result<SessionMeta> {
    match meta.version {
        // `0` is a pre-versioning header (the field was absent and defaulted); it is wire-compatible
        // with v1. `VERSION` is the current shape.
        0 | VERSION => Ok(meta),
        v => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "session {} is format version {v}, which this build (version {VERSION}) cannot read",
                path.display()
            ),
        )),
    }
}

/// Gather the `*.jsonl` files directly under `dir` — the flat, extension-filtered candidate set the
/// daemon's `list_daemon_sessions` feeds to [`scan_listings`]. A single non-recursive `read_dir`;
/// an unreadable directory (or a missing one) yields an empty list rather than erroring, matching the
/// skip-and-continue semantics of the listing scans that consume it.
pub(crate) fn scan_session_dir(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("jsonl"))
        .collect()
}

/// Scan every path in `paths` for its listing metadata, calling `on_progress(scanned, total)` once per
/// path — including ones that turn out unreadable, which still count as scanned and just contribute
/// nothing. `read_listing` is pure disk I/O plus parsing with no dependency between files, so the work
/// fans out across a small worker pool (`std::thread::available_parallelism`, capped at one thread per
/// path) rather than running strictly one file at a time; below two candidate workers it just runs
/// inline; no thread pool to justify the setup cost for a one- or two-file listing. Returned in
/// arbitrary order — every caller sorts the result itself.
pub(crate) fn scan_listings(
    paths: Vec<PathBuf>,
    on_progress: &(impl Fn(usize, usize) + Send + Sync),
) -> Vec<SessionMeta> {
    let total = paths.len();
    let workers = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1)
        .min(total);
    if workers < 2 {
        return paths
            .iter()
            .enumerate()
            .filter_map(|(i, path)| {
                let meta = read_listing(path);
                on_progress(i + 1, total);
                meta
            })
            .collect();
    }

    let scanned = AtomicUsize::new(0);
    let metas = Mutex::new(Vec::with_capacity(total));
    let scanned_ref = &scanned;
    let metas_ref = &metas;
    let chunk_size = total.div_ceil(workers);
    std::thread::scope(|scope| {
        for chunk in paths.chunks(chunk_size) {
            scope.spawn(move || {
                for path in chunk {
                    if let Some(meta) = read_listing(path) {
                        metas_ref
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .push(meta);
                    }
                    on_progress(scanned_ref.fetch_add(1, Ordering::Relaxed) + 1, total);
                }
            });
        }
    });
    metas
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Read a session file's listing metadata: its (version-checked) header with the derived `updated_at` /
/// `message_count` / `preview` / `search_text` fields filled in. One streaming pass — lines are read and
/// parsed individually, never collected — so this stays light even for long transcripts (the header
/// alone gives id/title/etc.; only the count and preview/search text need the scan). Returns `None` for
/// a file that isn't a readable session (no/invalid header, or an unreadable version), matching `list`'s
/// skip semantics.
pub(crate) fn read_listing(path: &Path) -> Option<SessionMeta> {
    let mtime = mtime_secs(path);
    let file = File::open(path).ok()?;
    let mut reader = BufReader::new(file);
    let mut header = String::new();

    // The header is the first line.
    reader.read_line(&mut header).ok()?;
    let mut meta = match serde_json::from_str::<Entry>(header.trim()).ok()? {
        Entry::Session(m) => migrate(m, path).ok()?,
        Entry::Message { .. } | Entry::Leaf { .. } | Entry::BranchSummary { .. } => return None,
        Entry::Compaction { .. } => return None,
        Entry::ModelChange { .. } | Entry::ThinkingLevelChange { .. } => return None,
        Entry::TitleChange { .. } => return None,
        Entry::Label { .. } => return None,
        Entry::Custom { .. } => return None,
    };

    // A streaming line count, not a tree walk: it counts every `Message` line in the file, which for a
    // branched session (Track L3, once wired) can exceed the *active* path's length by however many
    // off-branch entries exist. A display convenience, not a correctness input — accurate for every
    // session today, since nothing yet writes an off-branch entry.
    let mut message_count = 0usize;
    let mut preview = None;
    let mut search_text = String::new();
    let mut search_chars = 0usize;
    // The most recent stamped `Entry::Message` timestamp seen — preferred over `mtime` below when
    // any message actually carries one (see `Entry::Message::timestamp`'s doc comment). `0` means
    // "no stamped message seen" (an all-legacy file, or one with no message lines at all), in which
    // case `mtime` is the only signal available.
    let mut max_message_timestamp = 0u64;
    let mut raw = Vec::new();
    loop {
        // Same lenient, skip-just-this-line recovery as `SessionStore::open` (see its comment):
        // `read_capped_line` lets an oversized or invalid-UTF-8 line be skipped without losing the
        // count/preview derived from every good line after it, and only a genuine I/O failure stops
        // the scan early.
        let oversized = match read_capped_line(&mut reader, &mut raw) {
            Ok(None) => break,
            Ok(Some(oversized)) => oversized,
            Err(_) => break,
        };
        if oversized {
            tracing::warn!(path = %path.display(), "skipping oversized session entry line while listing");
            continue;
        }
        let Ok(line) = std::str::from_utf8(&raw) else {
            tracing::warn!(path = %path.display(), "skipping non-UTF-8 session entry line while listing");
            continue;
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match parse_entry_lenient(path, line) {
            Ok(Entry::Message {
                message, timestamp, ..
            }) => {
                message_count += 1;
                max_message_timestamp = max_message_timestamp.max(timestamp);
                if let Some(text) = first_user_text(&message) {
                    if preview.is_none() {
                        preview = Some(preview_of(text));
                    }
                }
                // The search corpus is broader than the preview: every user *and* assistant message's
                // text (not just the first user turn) — matching pi's own `allMessagesText`, so a
                // session is findable by something only the assistant said (a file path it named, an
                // error it printed), not only by what the user typed.
                if let Some(text) = message_search_text(&message) {
                    if search_chars < SEARCH_TEXT_MAX_CHARS {
                        if !search_text.is_empty() {
                            search_text.push(' ');
                            search_chars += 1;
                        }
                        let remaining = SEARCH_TEXT_MAX_CHARS - search_chars;
                        let taken: String = text.chars().take(remaining).collect();
                        search_chars += taken.chars().count();
                        search_text.push_str(&taken);
                    }
                }
            }
            // A stray header mid-file (or a branch-navigation/summary/compaction-provenance/label/custom
            // marker) is ignored — a custom entry contributes no message text of its own, matching
            // `message_count`'s "real conversation messages only" semantics.
            Ok(Entry::Session(_))
            | Ok(Entry::Leaf { .. })
            | Ok(Entry::BranchSummary { .. })
            | Ok(Entry::Compaction { .. })
            | Ok(Entry::Label { .. })
            | Ok(Entry::Custom { .. }) => {}
            // Whole-session-scoped: the most recent one anywhere in the file wins (Task #18, pi-parity
            // fix, for `model`/`thinking_level`) — so a `list_sessions`/`list_all_sessions` listing shows
            // the session's actual last-used model rather than its frozen creation-time one. Safe here
            // specifically because `read_listing`'s own `SessionMeta` is a display-only, throwaway
            // value (never fed into `Persistence::model_and_level_at`'s tree-fallback resolution the way
            // a real `SessionStore::open` would be) — see `SessionStore::record_model_change`'s doc
            // comment for why the *real*, operative `meta.model` (built by `open`, not this listing scan)
            // must NOT be mutated the same way.
            Ok(Entry::ModelChange { model, .. }) => {
                meta.model = model;
            }
            Ok(Entry::ThinkingLevelChange { level, .. }) => {
                meta.thinking_level = Some(level);
            }
            Ok(Entry::TitleChange { title, .. }) => {
                meta.title = title_or_clear(title);
            }
            // A fully-read line that failed to deserialize — skip just this one and keep scanning,
            // same relaxed recovery as `SessionStore::open` (see its comment): we know this line's
            // exact boundaries, so one bad line (anywhere in the file, not only a torn tail) no longer
            // truncates the count/preview derived from every good line after it.
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "skipping unparseable session entry line");
            }
        }
    }

    // Content-derived timestamp preferred over the file's OS mtime; mtime is only a fallback for a
    // legacy file with no stamped message (see `Entry::Message::timestamp`'s doc comment).
    meta.updated_at = if max_message_timestamp > 0 {
        max_message_timestamp
    } else {
        mtime
    };
    meta.message_count = message_count;
    meta.preview = preview;
    meta.search_text = search_text;
    Some(meta)
}

/// The first plain-text block of a user message — what a preview shows. Tool-result user turns carry no
/// `Text` block, so they yield `None` and are skipped; assistant turns aren't user input.
fn first_user_text(msg: &Message) -> Option<&str> {
    if msg.role != Role::User {
        return None;
    }
    msg.content.iter().find_map(|b| match b {
        ContentBlock::Text { text, .. } => Some(text.as_str()),
        _ => None,
    })
}

/// Every plain-text block of a user *or* assistant message, space-joined — the broader search-corpus
/// counterpart of [`first_user_text`] (which looks only at a `User` message's first block, for the
/// one-line preview). `None` for a tool-result-only turn or a message with no text content at all
/// (a pure tool-call turn), same as `first_user_text`.
fn message_search_text(msg: &Message) -> Option<String> {
    if msg.role != Role::User && msg.role != Role::Assistant {
        return None;
    }
    let joined = msg
        .content
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ");
    if joined.is_empty() {
        None
    } else {
        Some(joined)
    }
}

/// A one-line, length-bounded preview: the first non-blank line of `text`, trimmed, truncated on a char
/// boundary to [`PREVIEW_MAX`] with an ellipsis when it overran.
fn preview_of(text: &str) -> String {
    let line = text
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .trim();
    let mut out: String = line.chars().take(PREVIEW_MAX).collect();
    if line.chars().count() > PREVIEW_MAX {
        out.push('…');
    }
    out
}

/// A preview of any message's own text content (unlike [`first_user_text`], not restricted to the
/// `User` role) — [`SessionStore::tree`]'s per-node preview, one message at a time rather than a
/// branch's first user turn. `None` for a message with no plain-text block (a pure tool-use/tool-result/
/// thinking/image turn).
fn message_text_preview(msg: &Message) -> Option<String> {
    msg.content.iter().find_map(|b| match b {
        ContentBlock::Text { text, .. } if !text.trim().is_empty() => Some(preview_of(text)),
        _ => None,
    })
}

/// The file's last-modified time as Unix seconds, read from `fs` metadata (no file content). Falls back
/// to 0 if the platform/clock can't supply it — a listing field, not a correctness input.
fn mtime_secs(path: &Path) -> u64 {
    fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// fsync a path's parent directory so a just-created or just-renamed entry in it is itself durable
/// (a file `sync_all` persists contents, not the dentry that names them). A best-effort no-op if the
/// path has no parent.
/// Parse one session-file line into an [`Entry`], recovering from one specific, narrow corruption
/// instead of dropping the whole line the way the caller's generic `Err` handling otherwise would
/// (pi-parity fix, pass 20): a hand-edited or externally corrupted `message` line with its `content`
/// field set to `null` (or removed outright). `agent_core::Message::content` is a plain
/// `Vec<ContentBlock>` with no `#[serde(default)]` — deliberately, since a *live*, in-process write
/// should never omit it — so neither shape deserializes as a valid `Message`, and ordinary recovery
/// would discard the entry's `role`/`id`/`parent_id`/tree position along with it, potentially orphaning
/// whatever later message chained its own `parent_id` off this one's `id`. pi's own recovery for the
/// same corruption keeps the entry and simply treats it as having empty content instead of dropping it.
///
/// This only ever retries that one specific shape — `"type":"message"` with `content` absent or
/// `null` — by re-parsing the line as a bare [`Value`](serde_json::Value), patching in an empty
/// `content` array, and re-attempting the real typed deserialize. A `content` field that's present but
/// some *other* invalid shape (a string, a number), a missing `role`, an unknown future `type`, or
/// genuinely unparseable JSON all fall straight through to the original error — this is a targeted
/// repair for one known corruption, not a generally permissive parse.
fn parse_entry_lenient(path: &Path, line: &str) -> Result<Entry, serde_json::Error> {
    let original_err = match serde_json::from_str::<Entry>(line) {
        Ok(entry) => return Ok(entry),
        Err(e) => e,
    };
    (|| -> Option<Entry> {
        let mut value = serde_json::from_str::<serde_json::Value>(line).ok()?;
        let obj = value.as_object_mut()?;
        if obj.get("type").and_then(serde_json::Value::as_str) != Some("message") {
            return None;
        }
        match obj.get("content") {
            None | Some(serde_json::Value::Null) => {}
            Some(_) => return None,
        }
        obj.insert("content".to_string(), serde_json::Value::Array(Vec::new()));
        serde_json::from_value::<Entry>(value).ok()
    })()
    .inspect(|_| {
        tracing::warn!(
            path = %path.display(),
            "recovered session entry line with null/missing content by treating it as empty"
        );
    })
    .ok_or(original_err)
}

/// Ceiling on one session-file line (one JSON entry) while loading. A legitimate entry — even one
/// carrying a large embedded payload (a big base64 image in a message) — stays orders of magnitude
/// under this; it exists to bound how much a corrupted or hand-edited file (a stray length delimiter,
/// concatenated lines) can make [`SessionStore::open`] allocate before the line is treated as
/// unreadable, the same recovery path already used for a torn write.
const MAX_LINE_BYTES: usize = 64 * 1024 * 1024;

/// Read one `\n`-terminated line from `reader` into `buf` (cleared first, bytes only — the caller
/// validates UTF-8), via `fill_buf`/`consume` so a pathologically long or unterminated line is
/// processed in bounded chunks rather than by growing an unbounded `String` one byte at a time, the
/// way [`BufReader::lines`] would. `buf` never grows past [`MAX_LINE_BYTES`]; bytes beyond the cap are
/// still consumed from `reader` (so it lands correctly on the next line) but discarded, and the
/// returned flag reports whether that happened. `Ok(None)` at EOF with no more lines.
fn read_capped_line(reader: &mut impl BufRead, buf: &mut Vec<u8>) -> std::io::Result<Option<bool>> {
    buf.clear();
    let mut total = 0usize;
    let mut saw_any_byte = false;
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            break;
        }
        saw_any_byte = true;
        let (chunk, hit_newline) = match available.iter().position(|&b| b == b'\n') {
            Some(pos) => (&available[..pos], true),
            None => (available, false),
        };
        total += chunk.len();
        let room = MAX_LINE_BYTES.saturating_sub(buf.len());
        let take = chunk.len().min(room);
        buf.extend_from_slice(&chunk[..take]);
        let consumed = if hit_newline {
            chunk.len() + 1
        } else {
            chunk.len()
        };
        reader.consume(consumed);
        if hit_newline {
            break;
        }
    }
    if !saw_any_byte {
        return Ok(None);
    }
    Ok(Some(total > MAX_LINE_BYTES))
}

/// Create (or truncate) `path` for exclusive access: `0600` on Unix, set atomically at creation
/// rather than via a `set_permissions` call afterward (which would leave a window where the file is
/// world/group-readable). Session files carry the full transcript — including whatever `read` has
/// pulled off disk — so they should never be readable by anyone but the owner, independent of the
/// process umask.
fn create_private(path: &Path) -> std::io::Result<File> {
    let mut opts = OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    opts.open(path)
}

fn fsync_dir(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        // Opening the directory read-only and `sync_all`ing it flushes its updated entries. Linux/macOS
        // allow fsync on a directory handle; this is the standard atomic-rename durability step.
        File::open(parent)?.sync_all()?;
    }
    Ok(())
}

/// Serialize one entry as a single JSON line (no embedded newlines — `serde_json` escapes them).
fn write_line(w: &mut impl Write, entry: &Entry) -> std::io::Result<()> {
    let v = serde_json::to_value(entry).map_err(std::io::Error::other)?;
    // Defensive: a serialized line must be one physical line.
    debug_assert!(!serde_json::to_string(&v).unwrap_or_default().contains('\n'));
    writeln!(w, "{}", Value::to_string(&v))
}

/// Append one entry to the session file — an O(1) write, not a rewrite. Same durability posture as
/// every other append in this module (flush + `sync_all`; the parent directory's own dentry is
/// unchanged by an append, so no directory fsync is needed here either).
fn append_line(path: &Path, entry: &Entry) -> std::io::Result<()> {
    let mut buf = Vec::new();
    write_line(&mut buf, entry)?;
    let mut f = OpenOptions::new().append(true).open(path)?;
    f.write_all(&buf)?;
    f.flush()?;
    f.sync_all()
}

/// Look up the most recent entry in `changes` (keyed by anchor message id, `None` = before the first
/// message) that was already in effect **at** `target_id` — walking its *strict* ancestor path
/// root-first (deliberately excluding `target_id` itself) and keeping the last match, so a change
/// recorded further from the root always wins over an earlier one.
///
/// Excluding `target_id` itself is the crux: a change anchored *at* some message `X` means "this
/// applies to whatever gets appended *after* `X`" (see `record_model_change`'s doc comment) — it
/// describes `X`'s children, not `X` itself. Querying "what was active when switching to `X`" (the
/// point of this function — restoring branch-local settings on `switch_branch`) must recover whatever
/// was true *while `X` was being generated*, i.e. before that change, not after it. Concretely: switch
/// to a message right before a `set_model` call and this must report the *old* model, not the new one
/// the (abandoned, or not-yet-taken) next turn would have used.
///
/// Known limitation: if two different branches both grow from `X`, a change anchored at `X` is shared
/// by both (the map has no notion of "which branch"), so a query against a *descendant* on the second
/// branch can see a stale change actually made on the first. Restoring on `switch_branch` itself is
/// unaffected (it only ever queries the target being switched *to*, never a descendant of it), so this
/// only matters for a hypothetical future caller querying arbitrarily deep into a branch that grew
/// after a restore — accepted as a documented edge case rather than the fuller (and here unwarranted)
/// fix of threading these changes through the same per-branch chain messages use.
fn change_at<'a, V>(
    nodes: &HashMap<String, Node>,
    changes: &'a HashMap<Option<String>, V>,
    target_id: &str,
) -> Option<&'a V> {
    let mut last = changes.get(&None);
    let path = path_from_root(nodes, Some(target_id));
    let ancestors = path.len().saturating_sub(1); // exclude target_id itself
    for id in &path[..ancestors] {
        if let Some(v) = changes.get(&Some(id.clone())) {
            last = Some(v);
        }
    }
    last
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// A collision-resistant, roughly time-ordered session id: `<nanos>-<salt>-<seq>`.
///
/// The old `{nanos}{seq}` scheme was only process-local — two processes that started in the same
/// nanosecond produced the same id. We splice in a per-process **salt** seeded once from `RandomState`
/// (the OS-seeded hasher `HashMap` uses), so independent processes diverge even at the same instant;
/// within a process the monotonic `seq` breaks same-nanosecond ties. The nanosecond prefix keeps ids
/// sortable by creation. A UUID-shaped string isn't required — global uniqueness and ordering are.
pub(crate) fn new_id() -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::BuildHasher;
    use std::sync::atomic::{AtomicU64, Ordering};

    // Seeded once per process. `RandomState` draws OS entropy at construction, so hashing a fixed value
    // with it yields a value that's stable for this process but random across processes.
    static SALT: OnceLock<u64> = OnceLock::new();
    let salt = *SALT.get_or_init(|| RandomState::new().hash_one(0xC0FFEEu64));

    static SEQ: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    format!("{nanos:x}-{salt:016x}-{seq:x}")
}

/// Whether `id` is safe to embed directly in a filename component — alphanumeric, optionally with
/// `.`/`_`/`-` in the middle, starting and ending with a letter or digit. Matches pi's
/// `assertValidSessionId` (`^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$`); rejects anything that could
/// resolve to a path outside the sessions directory (a leading `/` or `..`, an embedded `/`, etc.) or
/// be empty. Lives here (not just in `main.rs`) so the WebSocket transport ([`crate::serve_ws`]) can
/// validate a client-supplied `?session_id=` before it ever becomes a filename component.
pub fn is_valid_session_id(id: &str) -> bool {
    let bytes = id.as_bytes();
    let is_alnum = |b: u8| b.is_ascii_alphanumeric();
    match bytes {
        [] => false,
        [only] => is_alnum(*only),
        [first, .., last] => {
            is_alnum(*first)
                && is_alnum(*last)
                && bytes
                    .iter()
                    .all(|&b| is_alnum(b) || b == b'.' || b == b'_' || b == b'-')
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tmpdir() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn append_then_reopen_restores_transcript() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo
            .create(SessionMeta::new("/work", "claude-test"))
            .unwrap();
        let id = store.meta().id.clone();

        let mut session = Session::new();
        session.user("hello");
        store.append_new(&session.messages).unwrap();
        session.user("world");
        store.append_new(&session.messages).unwrap();

        let (_store, restored) = repo.open_id(&id).unwrap();
        assert_eq!(restored.messages.len(), 2);
        let dump = serde_json::to_string(restored.messages.as_ref()).unwrap();
        assert!(dump.contains("hello") && dump.contains("world"));
    }

    #[test]
    #[cfg(unix)]
    fn session_files_are_created_private_and_stay_private_through_rewrite() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo
            .create(SessionMeta::new("/work", "claude-test"))
            .unwrap();

        let mode_of =
            |p: &std::path::Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        // Transcripts carry whatever `read` pulled off disk — never group/world-readable, on a
        // shared host in particular, regardless of the process umask.
        assert_eq!(
            mode_of(&store.path),
            0o600,
            "create() must set 0600 atomically"
        );

        let mut session = Session::new();
        session.user("a");
        session.user("b");
        store.append_new(&session.messages).unwrap();
        assert_eq!(
            mode_of(&store.path),
            0o600,
            "append must not loosen permissions"
        );

        store.rewrite(&[Message::user("summary")]).unwrap();
        assert_eq!(
            mode_of(&store.path),
            0o600,
            "rewrite's temp-file-then-rename must not loosen permissions"
        );
    }

    #[test]
    fn create_initializes_an_existing_empty_file_in_place_instead_of_failing() {
        // Track L8: a zero-byte file at the target path (e.g. `touch`'d ahead of time, or left over
        // from a crash before the header write landed) must not hard-fail `create` — it's
        // indistinguishable in intent from "not created yet."
        let dir = tmpdir();
        let path = dir.path().join("s.jsonl");
        std::fs::write(&path, b"").unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            0,
            "must start empty"
        );

        let store = SessionStore::create(path.clone(), SessionMeta::new("/w", "m")).unwrap();
        assert_eq!(store.meta().cwd, "/w");

        // The header actually landed on disk, not just in memory.
        let (reopened, session) = SessionStore::open(path).unwrap();
        assert_eq!(reopened.meta().id, store.meta().id);
        assert_eq!(session.messages.len(), 0);
    }

    #[test]
    fn create_still_refuses_a_genuinely_non_empty_existing_file() {
        // The empty-file allowance must not become a general "overwrite anything" escape hatch — a
        // file that already holds real (even non-session) content is never silently clobbered.
        let dir = tmpdir();
        let path = dir.path().join("s.jsonl");
        std::fs::write(&path, b"not a session file, but definitely not empty").unwrap();

        match SessionStore::create(path, SessionMeta::new("/w", "m")) {
            Err(e) => assert_eq!(e.kind(), std::io::ErrorKind::AlreadyExists),
            Ok(_) => panic!("must not silently clobber a non-empty existing file"),
        }
    }

    #[test]
    fn create_fork_and_fork_at_entry_all_write_their_file_immediately_even_for_a_single_message() {
        // pi-parity (C-M4), pinning current behavior: pi's `createBranchedSession` defers the actual
        // file write until an assistant message lands (`session-manager/tree-traversal.test.ts:464-532`),
        // specifically to avoid littering the session directory with a fork abandoned after one user
        // message. This module deliberately does NOT port that deferral — see the "Tree-shaped history"
        // section of ARCHITECTURE.md for the full reasoning (short version: `rewrite` already writes the
        // whole prefix atomically in one temp-file-then-rename call, so there's no truncated-file risk
        // to guard against, and threading a "created but not yet flushed" state through `SessionStore`
        // isn't worth it for a cosmetic directory-listing concern). This test pins that decision: a
        // fresh `create`, and a `fork`/`fork_at_entry` of a single-user-message prefix, must all have a
        // real file on disk immediately, before any assistant message ever exists.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();

        let store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        assert!(
            store.path().exists() && std::fs::metadata(store.path()).unwrap().len() > 0,
            "create must write a real header immediately"
        );

        let mut src = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let src_id = src.meta().id.clone();
        let mut session = Session::new();
        session.user("only a user message, no assistant reply yet");
        src.append_new(&session.messages).unwrap();
        let user_id = src.active_ids()[0].clone();

        let (forked, fsession) = repo.fork(&src_id, usize::MAX).unwrap();
        assert_eq!(fsession.messages.len(), 1);
        assert!(
            forked.path().exists() && std::fs::metadata(forked.path()).unwrap().len() > 0,
            "fork of a single-user-message (no assistant reply) prefix must still write immediately"
        );

        let (forked_at_entry, fsession2) = repo.fork_at_entry(&src_id, &user_id, false).unwrap();
        assert_eq!(fsession2.messages.len(), 1);
        assert!(
            forked_at_entry.path().exists()
                && std::fs::metadata(forked_at_entry.path()).unwrap().len() > 0,
            "fork_at_entry of a single-user-message prefix must still write immediately"
        );
    }

    #[test]
    fn append_new_batches_many_new_messages_in_one_call() {
        // Regression guard for the buffered-write fix: a multi-message batch (not the usual
        // one-or-two-at-a-time turn) must still land every line, in order, with the cursor advanced
        // exactly to the batch length — proving the single `write_all` path handles N > 1 correctly,
        // not just the common case.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo
            .create(SessionMeta::new("/work", "claude-test"))
            .unwrap();
        let id = store.meta().id.clone();

        let mut session = Session::new();
        for i in 0..50 {
            session.user(format!("message-{i}"));
        }
        store.append_new(&session.messages).unwrap();

        let (_store, restored) = repo.open_id(&id).unwrap();
        assert_eq!(restored.messages.len(), 50);
        for i in 0..50 {
            let dump = serde_json::to_string(&restored.messages[i]).unwrap();
            assert!(dump.contains(&format!("message-{i}")), "line {i}: {dump}");
        }

        // A follow-up call with a strict extension of the same slice only appends the new tail, not a
        // second copy of the first 50.
        session.user("message-50");
        store.append_new(&session.messages).unwrap();
        let (_store, restored) = repo.open_id(&id).unwrap();
        assert_eq!(restored.messages.len(), 51);
    }

    #[test]
    fn read_capped_line_bounds_allocation_and_flags_oversized_lines() {
        use std::io::Cursor;

        // A line beyond the cap must not grow `buf` past it, must still land the reader on the next
        // line (not lose sync), and must be flagged so `SessionStore::open` can treat it like a torn
        // write — a corrupted length delimiter or concatenated lines shouldn't be able to make `open`
        // allocate without bound.
        let mut input = vec![b'x'; MAX_LINE_BYTES + 10];
        input.push(b'\n');
        input.extend_from_slice(b"next\n");
        let mut reader = BufReader::new(Cursor::new(input));

        let mut buf = Vec::new();
        let oversized = read_capped_line(&mut reader, &mut buf).unwrap();
        assert_eq!(oversized, Some(true));
        assert_eq!(buf.len(), MAX_LINE_BYTES, "buf must not grow past the cap");

        let oversized = read_capped_line(&mut reader, &mut buf).unwrap();
        assert_eq!(oversized, Some(false));
        assert_eq!(buf, b"next");

        assert_eq!(read_capped_line(&mut reader, &mut buf).unwrap(), None);
    }

    #[test]
    fn oversized_line_is_recovered_like_a_torn_write() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo
            .create(SessionMeta::new("/work", "claude-test"))
            .unwrap();
        let id = store.meta().id.clone();
        let mut session = Session::new();
        session.user("first");
        store.append_new(&session.messages).unwrap();

        // A pathologically long line (well past MAX_LINE_BYTES) — the on-disk analogue of a
        // corrupted length delimiter or concatenated lines — must not be allocated in full; `open`
        // treats it as unreadable, the same as a torn write.
        let path = repo.find_path(&id).unwrap().unwrap();
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        write!(f, "\"").unwrap();
        for _ in 0..(MAX_LINE_BYTES / 1024 + 10) {
            write!(f, "{}", "x".repeat(1024)).unwrap();
        }
        writeln!(f, "\"").unwrap();
        drop(f);

        let (_store, restored) = repo.open_id(&id).unwrap();
        // The intact first message survives; the oversized record is dropped.
        assert_eq!(restored.messages.len(), 1);
    }

    #[test]
    fn torn_last_line_is_recovered() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo
            .create(SessionMeta::new("/work", "claude-test"))
            .unwrap();
        let id = store.meta().id.clone();
        let mut session = Session::new();
        session.user("first");
        store.append_new(&session.messages).unwrap();

        // Simulate a crash mid-append: a half-written, unterminated JSON line.
        let path = repo.find_path(&id).unwrap().unwrap();
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        write!(f, "{{\"type\":\"message\",\"role\":\"user\",\"cont").unwrap();
        drop(f);

        let (_store, restored) = repo.open_id(&id).unwrap();
        // The intact first message survives; the torn record is dropped.
        assert_eq!(restored.messages.len(), 1);
    }

    #[test]
    fn message_usage_round_trips_through_persist_and_reopen() {
        // Task #6 (pi-parity fix): Round 1 added `Message.usage` (`agent_core::TokenUsage`) — confirm
        // this persistence format actually carries it through a real append + reopen, not just relying
        // on `Message`'s `Serialize`/`Deserialize` derive being correct in isolation. No format/version
        // change was needed: `Entry::Message.message` embeds the whole `Message`, so the new
        // `#[serde(default, skip_serializing_if = "Option::is_none")]` field round-trips automatically.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo
            .create(SessionMeta::new("/work", "claude-test"))
            .unwrap();
        let usage = agent_core::TokenUsage {
            input_tokens: 120,
            output_tokens: 45,
            cache_read_tokens: 10,
            cache_write_tokens: 5,
            cache_write_1h_tokens: 0,
            reasoning_tokens: 7,
        };
        let mut session = Session::new();
        session.user("hi");
        session.push(
            Message::assistant(vec![ContentBlock::text("hello")])
                .with_model_id("claude-test")
                .with_usage(usage),
        );
        store.append_new(&session.messages).unwrap();

        let id = store.meta().id.clone();
        let (_reopened, restored) = repo.open_id(&id).unwrap();
        assert_eq!(restored.messages.len(), 2);
        assert_eq!(
            restored.messages[0].usage, None,
            "a user turn never carries usage"
        );
        assert_eq!(
            restored.messages[1].usage,
            Some(usage),
            "the assistant turn's own usage must survive a real append + reopen round trip"
        );
    }

    #[test]
    fn message_usage_round_trips_when_absent_matching_a_pre_round_1_file() {
        // Backward compatibility: a session with no `usage` on any message (every message predates
        // Round 1's field, or was never populated) must still load cleanly — `#[serde(default)]` reads
        // it as `None` rather than failing to deserialize the line at all.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo
            .create(SessionMeta::new("/work", "claude-test"))
            .unwrap();
        let mut session = Session::new();
        session.user("hi");
        session.push(Message::assistant(vec![ContentBlock::text("hello")]));
        store.append_new(&session.messages).unwrap();

        let id = store.meta().id.clone();
        let (_reopened, restored) = repo.open_id(&id).unwrap();
        assert_eq!(restored.messages.len(), 2);
        assert!(restored.messages.iter().all(|m| m.usage.is_none()));
    }

    #[test]
    fn open_restores_last_input_tokens_exactly_from_the_most_recent_messages_usage() {
        // Task #6 (pi-parity fix): when the active path's most recent usage-carrying message is
        // available, restoration must use its exact provider-reported figures — the same
        // `input + cache_read + cache_write` combination `Session::record_usage`'s own `live_input`
        // computes — rather than the coarser whole-transcript char/4 estimate.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo
            .create(SessionMeta::new("/work", "claude-test"))
            .unwrap();
        let usage = agent_core::TokenUsage {
            input_tokens: 1_000,
            output_tokens: 250,
            cache_read_tokens: 200,
            cache_write_tokens: 50,
            cache_write_1h_tokens: 0,
            reasoning_tokens: 0,
        };
        let mut session = Session::new();
        session.user("go");
        session.push(
            Message::assistant(vec![ContentBlock::text("ok")])
                .with_model_id("claude-test")
                .with_usage(usage),
        );
        store.append_new(&session.messages).unwrap();

        let id = store.meta().id.clone();
        let (_reopened, restored) = repo.open_id(&id).unwrap();
        assert_eq!(restored.last_input_tokens, 1_000 + 200 + 50);
        assert_eq!(restored.last_output_tokens, 250);
        // Positioned at the assistant message's own index (1), not `messages.len()` (2) — so
        // `compaction::trailing_tokens` treats that message itself as the "just completed turn" slot
        // (substituting `last_output_tokens` for it) rather than double-counting it as trailing.
        assert_eq!(restored.last_usage_message_count, 1);
        assert_eq!(
            agent_core::compaction::trailing_tokens(&restored),
            250,
            "the assistant message at the snapshot position must use last_output_tokens, not a fresh \
             char/4 estimate of its short reply text"
        );
    }

    #[test]
    fn open_falls_back_to_the_char4_estimate_when_no_message_has_usage() {
        // Pre-Round-1 (or otherwise usage-less) sessions must keep the previous whole-transcript
        // estimate behavior — this is the fallback branch `open_restores_last_input_tokens_exactly_...`
        // above bypasses whenever real usage is available.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo
            .create(SessionMeta::new("/work", "claude-test"))
            .unwrap();
        let mut session = Session::new();
        session.user("x".repeat(400)); // ~100 estimated tokens (char/4)
        store.append_new(&session.messages).unwrap();

        let id = store.meta().id.clone();
        let (_reopened, restored) = repo.open_id(&id).unwrap();
        assert!(
            restored.last_input_tokens > 0,
            "must still estimate something so should_compact/is_hard_overflow can fire on a resumed \
             session that's already over threshold"
        );
        assert_eq!(
            restored.last_usage_message_count,
            restored.messages.len(),
            "with no real usage snapshot, every persisted message is treated as already accounted for"
        );
    }

    #[test]
    fn oversized_line_mid_file_does_not_discard_subsequent_good_entries() {
        // The bug: an oversized line was treated like a torn *final* write (stop reading, keep what's
        // valid so far) even when it wasn't the last line — silently discarding every good entry after
        // it. It must instead be skipped like `corrupt_line_mid_file_does_not_discard_subsequent_good_entries`'s
        // unparseable-JSON case.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo
            .create(SessionMeta::new("/work", "claude-test"))
            .unwrap();
        let id = store.meta().id.clone();
        let mut session = Session::new();
        session.user("first");
        store.append_new(&session.messages).unwrap();

        let path = repo.find_path(&id).unwrap().unwrap();
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        write!(f, "\"").unwrap();
        for _ in 0..(MAX_LINE_BYTES / 1024 + 10) {
            write!(f, "{}", "x".repeat(1024)).unwrap();
        }
        writeln!(f, "\"").unwrap();
        drop(f);

        session.user("third");
        store.append_new(&session.messages).unwrap();

        let (_store, restored) = repo.open_id(&id).unwrap();
        assert_eq!(
            restored.messages.len(),
            2,
            "both good entries must survive an oversized line between them, not just the ones before it"
        );
    }

    #[test]
    fn invalid_utf8_line_mid_file_does_not_discard_subsequent_good_entries() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo
            .create(SessionMeta::new("/work", "claude-test"))
            .unwrap();
        let id = store.meta().id.clone();
        let mut session = Session::new();
        session.user("first");
        store.append_new(&session.messages).unwrap();

        // A line with invalid UTF-8 bytes (simulating bit rot) landing mid-file, not at the end.
        let path = repo.find_path(&id).unwrap().unwrap();
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(b"\xff\xfe not valid utf-8\n").unwrap();
        drop(f);

        session.user("third");
        store.append_new(&session.messages).unwrap();

        let (_store, restored) = repo.open_id(&id).unwrap();
        assert_eq!(
            restored.messages.len(),
            2,
            "both good entries must survive an invalid-UTF-8 line between them, not just the ones before it"
        );
    }

    #[test]
    fn corrupt_line_mid_file_does_not_discard_subsequent_good_entries() {
        // A bad line anywhere in the file — not only a torn final write — must be skipped, not treated
        // as "nothing valid follows a half-written record": the good entries appended *after* it must
        // still be recovered on open.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo
            .create(SessionMeta::new("/work", "claude-test"))
            .unwrap();
        let id = store.meta().id.clone();
        let mut session = Session::new();
        session.user("first");
        store.append_new(&session.messages).unwrap();

        // A complete (newline-terminated), well-formed JSON line that isn't a valid `Entry` — not a
        // torn write, just corrupted/foreign content landing mid-file (disk bit rot, a manual edit).
        let path = repo.find_path(&id).unwrap().unwrap();
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        writeln!(f, r#"{{"not":"a valid entry"}}"#).unwrap();
        drop(f);

        // Appended using the same in-memory `store` (its `active`/`persisted` state is untouched by
        // the raw write above), so this chains correctly off "first" regardless of the bad line
        // sitting between them on disk.
        session.user("third");
        store.append_new(&session.messages).unwrap();

        let (_store, restored) = repo.open_id(&id).unwrap();
        assert_eq!(
            restored.messages.len(),
            2,
            "both good entries must survive a bad line between them, not just the ones before it"
        );
    }

    #[test]
    fn open_recovers_a_message_line_hand_edited_to_have_null_content_instead_of_dropping_it() {
        // Pass 20 (pi-parity fix): `Message::content` (`Vec<ContentBlock>`) has no `#[serde(default)]`
        // — deliberately, a live in-process write should never omit it — so a line hand-edited (or
        // externally corrupted) down to `"content":null` used to fail `Entry` deserialization entirely
        // and fall into the same generic "skip this whole line" recovery as genuine garbage, silently
        // losing the message's role/id/tree position along with its content. `parse_entry_lenient`
        // recovers this one specific shape by treating the content as empty instead, matching pi's own
        // equivalent fix, rather than dropping the entry outright.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo
            .create(SessionMeta::new("/work", "claude-test"))
            .unwrap();
        let id = store.meta().id.clone();
        let mut session = Session::new();
        session.user("first");
        store.append_new(&session.messages).unwrap();
        let first_id = store.active_ids().last().unwrap().clone();

        // A hand-edited message line: well-formed JSON, a real `"type":"message"` entry, but its
        // `content` was blanked out to `null` (as if someone had redacted it in a text editor) — chained
        // off "first" so replay's tip lands on it as the new last message.
        let path = repo.find_path(&id).unwrap().unwrap();
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        writeln!(
            f,
            r#"{{"type":"message","id":"hand-edited","parent_id":"{first_id}","timestamp":0,"role":"user","content":null}}"#
        )
        .unwrap();
        drop(f);

        let (_store, restored) = repo.open_id(&id).unwrap();
        assert_eq!(
            restored.messages.len(),
            2,
            "the hand-edited line must be recovered as a message, not dropped entirely: {:?}",
            restored.messages
        );
        let recovered = &restored.messages[1];
        assert_eq!(recovered.role, Role::User, "role must survive the recovery");
        assert!(
            recovered.content.is_empty(),
            "content must recover as empty, not cause the whole entry to be lost: {:?}",
            recovered.content
        );
    }

    #[test]
    fn open_still_drops_a_message_line_whose_content_is_some_other_invalid_shape() {
        // The recovery `open_recovers_a_message_line_hand_edited_to_have_null_content_instead_of_dropping_it`
        // proves is narrowly scoped to `content` being absent or `null` specifically — a `content` field
        // that's present but some *other* invalid shape (here, a bare string instead of an array of
        // content blocks) is a different, genuine corruption, not this one known-recoverable case, and
        // must still fall through to the ordinary skip-this-line recovery.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo
            .create(SessionMeta::new("/work", "claude-test"))
            .unwrap();
        let id = store.meta().id.clone();
        let mut session = Session::new();
        session.user("first");
        store.append_new(&session.messages).unwrap();
        let first_id = store.active_ids().last().unwrap().clone();

        let path = repo.find_path(&id).unwrap().unwrap();
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        writeln!(
            f,
            r#"{{"type":"message","id":"bad-content-shape","parent_id":"{first_id}","timestamp":0,"role":"user","content":"not an array"}}"#
        )
        .unwrap();
        drop(f);

        let (_store, restored) = repo.open_id(&id).unwrap();
        assert_eq!(
            restored.messages.len(),
            1,
            "a content field that's present but the wrong shape (not null/missing) must still be \
             dropped as genuinely unparseable, not recovered: {:?}",
            restored.messages
        );
    }

    #[test]
    fn read_listing_survives_an_oversized_invalid_utf8_and_corrupt_line_all_mid_file() {
        // pi-parity gap (fixed, L4): `read_listing` shares `SessionStore::open`'s exact skip-and-
        // continue recovery logic (same `read_capped_line` primitive, same three corruption cases),
        // but only `open`'s recovery had a dedicated test — this one drives all three corruption
        // shapes through the *listing* scan specifically, proving `message_count`/`preview`/
        // `search_text` all still reflect every good entry, not just the ones before the first bad
        // line.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo
            .create(SessionMeta::new("/work", "claude-test"))
            .unwrap();
        let id = store.meta().id.clone();
        let mut session = Session::new();
        session.user("first good message");
        store.append_new(&session.messages).unwrap();

        let path = repo.find_path(&id).unwrap().unwrap();
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        // Oversized line.
        write!(f, "\"").unwrap();
        for _ in 0..(MAX_LINE_BYTES / 1024 + 10) {
            write!(f, "{}", "x".repeat(1024)).unwrap();
        }
        writeln!(f, "\"").unwrap();
        // Invalid UTF-8 line.
        f.write_all(b"\xff\xfe not valid utf-8\n").unwrap();
        // Well-formed JSON that isn't a valid `Entry`.
        writeln!(f, r#"{{"not":"a valid entry"}}"#).unwrap();
        drop(f);

        session.user("second good message");
        store.append_new(&session.messages).unwrap();

        let listed = read_listing(&path).unwrap();
        assert_eq!(
            listed.message_count, 2,
            "both good messages must be counted, not just the one before the corruption"
        );
        assert!(
            listed.search_text.contains("second good message"),
            "the message appended after the corrupted lines must still reach the search corpus: {}",
            listed.search_text
        );
    }

    #[test]
    fn rewrite_replaces_whole_transcript() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo
            .create(SessionMeta::new("/work", "claude-test"))
            .unwrap();
        let id = store.meta().id.clone();
        let mut session = Session::new();
        session.user("a");
        session.user("b");
        session.user("c");
        store.append_new(&session.messages).unwrap();

        // Compaction-style rewrite to a shorter list.
        let compacted = vec![Message::user("summary")];
        store.rewrite(&compacted).unwrap();

        let (_store, restored) = repo.open_id(&id).unwrap();
        assert_eq!(restored.messages.len(), 1);
        let dump = serde_json::to_string(restored.messages.as_ref()).unwrap();
        assert!(dump.contains("summary") && !dump.contains("\"a\""));
    }

    #[test]
    fn rewrite_cleans_up_the_temp_file_when_the_final_rename_fails() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("hello");
        store.append_new(&session.messages).unwrap();

        // Replace the session file with an (empty) directory of the same name: `create_private(&tmp)`
        // still succeeds (it's a different filename), but the final `fs::rename(&tmp, &self.path)`
        // must fail — a file can never be renamed onto an existing directory. A genuine in-process
        // error, not a crash, so `rewrite` gets the chance to clean up after itself.
        let path = store.path.clone();
        fs::remove_file(&path).unwrap();
        fs::create_dir(&path).unwrap();

        let tmp = path.with_extension("jsonl.tmp");
        let err = store.rewrite(&[]).unwrap_err();
        assert!(
            !tmp.exists(),
            "the orphaned temp file must be removed after a failed rewrite: {err}"
        );
    }

    #[test]
    fn reset_for_new_session_clears_model_thinking_label_and_event_state_a_plain_rewrite_leaves_behind()
     {
        // Pass 20 (pi-parity fix): single-file mode's `/new`-equivalent (`Persistence::new_session`,
        // `serve.rs`) resets an *already-populated* `SessionStore` in place (there's no fresh store to
        // swap in the way repo mode's `SessionRepo::create` gives it) — it used to call plain
        // `rewrite(&[])`, which only ever clears the message tree itself, never the id-keyed side
        // indexes this test drives into every one of. `reset_for_new_session` must clear all of them.
        let dir = tmpdir();
        let path = dir.path().join("s.jsonl");
        let mut store = SessionStore::create(path, SessionMeta::new("/work", "model-a")).unwrap();

        // A model/thinking-level change recorded before any message exists anchors at the tree root
        // (`self.active.last()` is `None`) — exactly the `switch_branch{before: true}`-reachable case
        // the audit named.
        store.record_model_change("model-b").unwrap();
        store.record_thinking_level_change("high").unwrap();

        let mut session = Session::new();
        session.user("hello");
        store.append_new(&session.messages).unwrap();
        let msg_id = store.active_ids().last().unwrap().clone();
        store.set_label(&msg_id, Some("bookmark")).unwrap();
        store
            .append_custom("note", serde_json::json!({"k": "v"}))
            .unwrap();

        // Sanity check: every one of these is actually populated before the reset, or the assertions
        // below would pass vacuously.
        assert_eq!(store.model_at_root(), Some("model-b"));
        assert_eq!(store.thinking_level_at_root(), Some("high"));
        assert_eq!(store.get_label(&msg_id), Some("bookmark"));
        assert!(!store.export_events().is_empty());

        store.reset_for_new_session().unwrap();

        assert_eq!(
            store.model_at_root(),
            None,
            "a stale model_changes entry at root must not leak into the reset session"
        );
        assert_eq!(
            store.thinking_level_at_root(),
            None,
            "a stale level_changes entry at root must not leak into the reset session"
        );
        assert_eq!(
            store.get_label(&msg_id),
            None,
            "a label on a message from the discarded session must not survive the reset"
        );
        assert!(
            store.export_events().is_empty(),
            "export_events must not replay the discarded session's ModelChange/ThinkingLevelChange/\
             Label/Custom history: {:?}",
            store.export_events()
        );
    }

    #[test]
    fn reset_for_new_session_does_not_affect_a_narrow_rewrites_side_indexes() {
        // The flip side of the test just above: `rewrite` itself (used by `fork`/`fork_at_entry`/
        // `fork_from_path`'s already-fresh stores, and `rewrite_compacted`'s degenerate same-length
        // fallback on *this* store) must keep its narrower, unchanged behavior — this pass's fix must
        // not leak into every other `rewrite` caller.
        let dir = tmpdir();
        let path = dir.path().join("s.jsonl");
        let mut store = SessionStore::create(path, SessionMeta::new("/work", "model-a")).unwrap();
        store.record_model_change("model-b").unwrap();

        let mut session = Session::new();
        session.user("hello");
        store.append_new(&session.messages).unwrap();
        let msg_id = store.active_ids().last().unwrap().clone();
        store.set_label(&msg_id, Some("bookmark")).unwrap();

        // A same-length rewrite (mirrors `rewrite_compacted`'s degenerate fallback) on the same store.
        store.rewrite(&session.messages).unwrap();

        assert_eq!(
            store.model_at_root(),
            Some("model-b"),
            "a plain `rewrite` must not clear model_changes — only `reset_for_new_session` does"
        );
        assert_eq!(
            store.get_label(&msg_id),
            Some("bookmark"),
            "a plain `rewrite` must not clear labels — only `reset_for_new_session` does"
        );
    }

    #[test]
    fn rewrite_compacted_preserves_folded_messages_and_records_provenance() {
        // Compact 6 messages down to [summary, kept5, kept6] — the first 4 are folded away. Unlike
        // `rewrite`, this must (a) keep every folded message physically readable on disk by its
        // original id, (b) write exactly one `Entry::Compaction` provenance record naming them, and
        // (c) still start the new active path as its own fresh, detached chain — *not* linked back
        // into the folded prefix, which would resurrect it into the live transcript.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        for text in ["one", "two", "three", "four", "five", "six"] {
            session.user(text);
        }
        store.append_new(&session.messages).unwrap();
        let old_ids = store.active_ids().to_vec();
        assert_eq!(old_ids.len(), 6);

        let compacted_messages = vec![
            Message::user(format!(
                "{}\n\nrecap of the folded work",
                agent_core::compaction::SUMMARY_MARKER
            )),
            session.messages[4].clone(),
            session.messages[5].clone(),
        ];
        store
            .rewrite_compacted(
                &compacted_messages,
                CompactionMeta {
                    tokens_before: 12345,
                    provenance: Default::default(),
                },
            )
            .unwrap();
        assert_eq!(store.active_ids().len(), 3);
        assert_eq!(store.meta().compactions, 1);
        assert_eq!(store.meta().dropped_messages, 3); // net shrinkage: 4 folded -> 1 summary

        let (reopened, restored) = repo.open_id(&store.meta().id.clone()).unwrap();
        assert_eq!(restored.messages.len(), 3);
        assert_eq!(reopened.active_ids().len(), 3);

        // Parse every line and find the two entries of interest, rather than substring-sniffing raw
        // JSON (id ordering across the file isn't guaranteed).
        let raw = fs::read_to_string(&reopened.path).unwrap();
        let lines: Vec<Value> = raw
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();

        // Every folded message is still physically present, by its original id and content.
        for (id, text) in old_ids[..4].iter().zip(["one", "two", "three", "four"]) {
            let found = lines.iter().find(|v| v["id"] == json!(id));
            assert!(
                found.is_some(),
                "folded message {id} ({text}) missing from disk: {raw}"
            );
        }

        // Exactly one compaction provenance record, naming the folded ids and carrying tokens_before.
        let compaction_entries: Vec<&Value> = lines
            .iter()
            .filter(|v| v["type"] == json!("compaction"))
            .collect();
        assert_eq!(
            compaction_entries.len(),
            1,
            "expected exactly one compaction entry: {raw}"
        );
        let entry = compaction_entries[0];
        assert_eq!(entry["tokens_before"], json!(12345));
        assert_eq!(entry["summary"], json!("recap of the folded work"));
        assert_eq!(
            entry["folded_ids"].as_array().unwrap().len(),
            4,
            "should name all 4 folded messages: {entry:#?}"
        );
        for id in &old_ids[..4] {
            assert!(
                entry["folded_ids"].as_array().unwrap().contains(&json!(id)),
                "folded_ids missing {id}: {entry:#?}"
            );
        }

        // The new active path's first node (the summary) starts a fresh, detached chain — *not*
        // linked back into the folded prefix. `path_from_root` walks a tip's whole parent chain to
        // build the live session, so a real link there would resurrect every folded message back into
        // the "active" transcript, undoing the compaction (already caught once: `restored.messages`
        // above would be 7, not 3, if this regressed).
        let summary_entry = lines
            .iter()
            .find(|v| {
                v["type"] == json!("message") && v.to_string().contains("recap of the folded work")
            })
            .expect("summary message entry not found");
        assert_eq!(
            summary_entry["parent_id"],
            Value::Null,
            "the summary message must start a fresh chain, not link back into the folded prefix: {summary_entry:#?}"
        );

        // The folded prefix survives as its own self-contained sub-chain (still linked to each other
        // exactly as before compaction), unreachable from the new tip but not orphaned from *root* —
        // the first folded message's parent_id is whatever it always was (`None`, the session root).
        let first_folded_entry = lines
            .iter()
            .find(|v| v["id"] == json!(old_ids[0]))
            .expect("first folded message entry not found");
        assert_eq!(first_folded_entry["parent_id"], Value::Null);
        let second_folded_entry = lines
            .iter()
            .find(|v| v["id"] == json!(old_ids[1]))
            .expect("second folded message entry not found");
        assert_eq!(second_folded_entry["parent_id"], json!(old_ids[0]));
    }

    #[test]
    fn rewrite_compacted_persists_file_provenance_and_it_survives_a_reopen() {
        // Fix 2 (pi-parity remediation, Round 2): `agent_core::Session.compaction`
        // (`CompactionProvenance`'s `read_files`/`modified_files`/`last_reason`, folded forward
        // turn-to-turn by `agent_core::compaction::merge_file_ops`) was purely in-memory — the on-disk
        // `Entry::Compaction` record had no field to carry it at all, and `SessionStore::open` never
        // restored it. So a `serve` restart or session reattach after even one compaction round
        // silently forgot every file that round already knew about: the *next* compaction's
        // `<read-files>`/`<modified-files>` tags would quietly omit everything from before the
        // restart. Reproduced here exactly as that would happen: seed `Session.compaction` as if a
        // live compaction round just fired, persist it via `rewrite_compacted`, then reopen the
        // session fresh (a brand-new `SessionStore::open`/`repo.open_id` call, not the same in-memory
        // `store`) and assert the provenance is back.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("a");
        session.user("b");
        store.append_new(&session.messages).unwrap();

        let provenance = CompactionProvenance {
            read_files: vec!["src/main.rs".to_string(), "README.md".to_string()],
            modified_files: vec!["src/lib.rs".to_string()],
            compactions: 1,
            last_reason: Some(agent_core::compaction::CompactionReason::Threshold),
            todos: None,
        };
        store
            .rewrite_compacted(
                &[Message::user("summary")],
                CompactionMeta {
                    tokens_before: 4242,
                    provenance: provenance.clone(),
                },
            )
            .unwrap();

        let (_reopened, restored) = repo.open_id(&store.meta().id.clone()).unwrap();
        assert_eq!(
            restored.compaction.read_files, provenance.read_files,
            "read_files must survive a fresh reopen, not reset to empty"
        );
        assert_eq!(
            restored.compaction.modified_files, provenance.modified_files,
            "modified_files must survive a fresh reopen, not reset to empty"
        );
        assert_eq!(
            restored.compaction.last_reason, provenance.last_reason,
            "last_reason must survive a fresh reopen"
        );
        // The counter is restored from `SessionMeta::compactions` (already correctly persisted
        // separately), not trusted verbatim from the record itself — see `SessionStore::open`'s own
        // doc comment on why.
        assert_eq!(restored.compaction.compactions, store.meta().compactions);
    }

    #[test]
    fn a_compacted_away_todo_list_survives_a_fresh_reopen() {
        // Once `apply_summary` has dropped the `todo` tool_use block, `Session.compaction.todos` is the
        // only copy of the model's plan that exists. If it didn't round-trip through the store, every
        // `serve` restart or reattach past a compaction would silently lose it — and the very next
        // compaction would write a summary with no `<todo_list>` block at all.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("a");
        session.user("b");
        store.append_new(&session.messages).unwrap();

        let todos = serde_json::json!([
            { "content": "Wire the retry loop", "activeForm": "Wiring the retry loop", "status": "in_progress" },
        ]);
        store
            .rewrite_compacted(
                &[Message::user("summary")],
                CompactionMeta {
                    tokens_before: 4242,
                    provenance: CompactionProvenance {
                        todos: Some(todos.clone()),
                        compactions: 1,
                        ..Default::default()
                    },
                },
            )
            .unwrap();

        let (_reopened, restored) = repo.open_id(&store.meta().id.clone()).unwrap();
        assert_eq!(
            restored.compaction.todos,
            Some(todos),
            "the todo list must survive a fresh reopen, not reset to None"
        );
    }

    #[test]
    fn a_compaction_record_written_before_the_todos_field_existed_still_loads() {
        // `#[serde(default)]` on the field, exercised against the literal on-disk shape an older build
        // wrote: a `compaction` entry with no `todos` key at all must parse (and simply restore `None`),
        // not fail the line and silently drop the whole provenance record with it.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("a");
        session.user("b");
        store.append_new(&session.messages).unwrap();
        store
            .rewrite_compacted(
                &[Message::user("summary")],
                CompactionMeta {
                    tokens_before: 1,
                    provenance: CompactionProvenance {
                        read_files: vec!["a.rs".into()],
                        todos: Some(serde_json::json!([{ "content": "x", "status": "pending" }])),
                        compactions: 1,
                        ..Default::default()
                    },
                },
            )
            .unwrap();

        // Strip the field back out of the on-disk record, exactly as a pre-`todos` build left it.
        let path = store.path.clone();
        let stripped = fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|line| {
                let mut v: serde_json::Value = serde_json::from_str(line).unwrap();
                if v.get("type").and_then(|t| t.as_str()) == Some("compaction") {
                    assert!(
                        v.as_object_mut().unwrap().remove("todos").is_some(),
                        "the field must have been written in the first place"
                    );
                }
                serde_json::to_string(&v).unwrap()
            })
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&path, format!("{stripped}\n")).unwrap();

        let (_reopened, restored) = repo.open_id(&store.meta().id.clone()).unwrap();
        assert_eq!(restored.compaction.todos, None);
        assert_eq!(
            restored.compaction.read_files,
            vec!["a.rs".to_string()],
            "the rest of the record must still restore — the line parsed, it wasn't skipped"
        );
    }

    #[test]
    fn rewrite_compacted_folds_file_provenance_forward_across_two_rounds_and_both_survive_a_reopen()
    {
        // The forward-folding half of Fix 2: a *second* compaction round's own `CompactionMeta
        // .provenance` (what `agent_core::compaction::merge_file_ops` already folds forward in
        // memory) must still be a complete snapshot on its own — `SessionStore::open` only ever reads
        // the *last* `Entry::Compaction` record, trusting that it already carries every earlier
        // round's files, exactly like `Session.compaction` already does turn-to-turn in memory.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("a");
        session.user("b");
        session.user("c");
        session.user("d");
        store.append_new(&session.messages).unwrap();

        let round1 = CompactionProvenance {
            read_files: vec!["a.rs".to_string()],
            modified_files: vec![],
            compactions: 1,
            last_reason: Some(agent_core::compaction::CompactionReason::Threshold),
            todos: None,
        };
        store
            .rewrite_compacted(
                &[
                    Message::user("summary1"),
                    Message::user("c"),
                    Message::user("d"),
                ],
                CompactionMeta {
                    tokens_before: 100,
                    provenance: round1,
                },
            )
            .unwrap();

        // Round 2 folds round 1's `a.rs` forward with its own new activity, matching what
        // `merge_file_ops` would actually do — the persisted record must reflect that already-folded
        // union, not just round 2's own new files.
        let round2 = CompactionProvenance {
            read_files: vec!["a.rs".to_string(), "b.rs".to_string()],
            modified_files: vec!["c.rs".to_string()],
            compactions: 2,
            last_reason: Some(agent_core::compaction::CompactionReason::Manual),
            todos: None,
        };
        store
            .rewrite_compacted(
                &[Message::user("summary2")],
                CompactionMeta {
                    tokens_before: 200,
                    provenance: round2,
                },
            )
            .unwrap();

        let (_reopened, restored) = repo.open_id(&store.meta().id.clone()).unwrap();
        assert_eq!(restored.compaction.read_files, vec!["a.rs", "b.rs"]);
        assert_eq!(restored.compaction.modified_files, vec!["c.rs"]);
        assert_eq!(
            restored.compaction.last_reason,
            Some(agent_core::compaction::CompactionReason::Manual)
        );
        assert_eq!(restored.compaction.compactions, 2);
    }

    #[test]
    fn tree_reports_entry_kind_for_a_compaction_and_a_branch_summary_not_just_a_plain_message() {
        // Track L25 (pi-parity fix): `Entry::BranchSummary`/`Entry::Compaction` used to be
        // indistinguishable from an ordinary message once materialized into `tree()`'s output — a
        // client had no way to tell "this is a recap"/"a compaction happened here" apart from an
        // everyday turn without guessing off `preview` text. Proves `entry_kind` recovers both, and
        // that an ordinary message still reports `"message"`.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        for text in ["one", "two", "three", "four"] {
            session.user(text);
        }
        store.append_new(&session.messages).unwrap();
        let ids = store.active_ids().to_vec();

        // An ordinary message reports the baseline kind.
        let plain = store.tree().into_iter().find(|n| n.id == ids[0]).unwrap();
        assert_eq!(plain.entry_kind, "message");

        // A branch-summary recap, still a real `NodeContent::Message` under the hood (so it reaches
        // the model), reports its own distinct kind instead.
        store
            .switch_active_with_summary(
                &ids[0],
                "recap of the abandoned branch",
                &ids[3],
                BranchSummaryDetails::default(),
            )
            .unwrap();
        let summary_id = store.active_ids().last().unwrap().clone();
        let tree = store.tree();
        let summary_node = tree.iter().find(|n| n.id == summary_id).unwrap();
        assert_eq!(summary_node.entry_kind, "branch_summary");
        assert!(
            summary_node.role.is_some(),
            "a branch-summary node is still a real message with a role, just distinguishable by kind"
        );

        // A compaction round's own provenance record, never itself a chain node, is still surfaced as
        // its own synthetic entry.
        store
            .rewrite_compacted(
                &[Message::user("kept after compaction")],
                CompactionMeta {
                    tokens_before: 999,
                    provenance: Default::default(),
                },
            )
            .unwrap();
        let tree = store.tree();
        let compaction_node = tree
            .iter()
            .find(|n| n.entry_kind == "compaction")
            .expect("a compaction node must be recoverable from tree()'s output");
        assert!(
            compaction_node
                .preview
                .as_deref()
                .is_some_and(|p| p.contains("999")),
            "the compaction node's preview should still mention tokens_before: {compaction_node:#?}"
        );

        // Reopening from disk must agree — this isn't just an in-memory-only artifact of the live
        // instance that ran the compaction.
        let (reopened, _) = repo.open_id(&store.meta().id.clone()).unwrap();
        let reopened_tree = reopened.tree();
        assert!(
            reopened_tree.iter().any(|n| n.entry_kind == "compaction"),
            "a reopened session must still report the compaction node: {reopened_tree:#?}"
        );
        assert!(
            reopened_tree
                .iter()
                .any(|n| n.entry_kind == "branch_summary"),
            "a reopened session must still report the branch-summary node: {reopened_tree:#?}"
        );
    }

    #[test]
    fn rewrite_compacted_appends_without_rewriting_existing_bytes() {
        // The whole point of H-7: every byte already on disk before the call must still be there,
        // byte-for-byte, as a strict *prefix* of the file afterward — proof this is a pure append, not
        // a rewrite (which would reserialize and potentially reorder everything).
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        for text in ["one", "two", "three", "four", "five"] {
            session.user(text);
        }
        store.append_new(&session.messages).unwrap();
        let path = repo.find_path(&store.meta().id.clone()).unwrap().unwrap();
        let before = fs::read(&path).unwrap();

        let compacted = vec![
            Message::user(format!(
                "{}\n\nrecap",
                agent_core::compaction::SUMMARY_MARKER
            )),
            session.messages[4].clone(),
        ];
        store
            .rewrite_compacted(
                &compacted,
                CompactionMeta {
                    tokens_before: 1,
                    provenance: Default::default(),
                },
            )
            .unwrap();

        let after = fs::read(&path).unwrap();
        assert!(
            after.starts_with(&before),
            "every pre-existing byte must survive untouched as a prefix of the file"
        );
        assert!(
            after.len() > before.len(),
            "new content must have been appended"
        );
    }

    #[test]
    fn rewrite_compacted_handles_two_sequential_compactions_back_to_back() {
        // B-L6 pi-parity test gap (fixed): every existing test exercised a single `rewrite_compacted`
        // call in isolation — matches pi's "should handle multiple compactions (only latest matters)"
        // (`compaction.test.ts:377-398`). A second compaction, appended right after the first with no
        // reopen in between, must independently record its own provenance (not overwrite or merge
        // with the first), and the materialized session must reflect only the *latest* cut.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        for text in ["one", "two", "three", "four", "five"] {
            session.user(text);
        }
        store.append_new(&session.messages).unwrap();
        let round1_ids = store.active_ids().to_vec();
        assert_eq!(round1_ids.len(), 5);

        // Round 1: fold "one","two","three" down to [summary1, four, five] — dropped = 2.
        let round1_messages = vec![
            Message::user(format!(
                "{}\n\nfirst summary",
                agent_core::compaction::SUMMARY_MARKER
            )),
            session.messages[3].clone(), // "four"
            session.messages[4].clone(), // "five"
        ];
        store
            .rewrite_compacted(
                &round1_messages,
                CompactionMeta {
                    tokens_before: 100,
                    provenance: Default::default(),
                },
            )
            .unwrap();
        assert_eq!(store.meta().compactions, 1);
        assert_eq!(store.meta().dropped_messages, 2);
        assert_eq!(store.active_ids().len(), 3);

        // Continue past round 1 with two more ordinary messages — no reopen in between, exercising
        // the in-process (not just persisted-and-reopened) state right after a compaction.
        let mut continued = Session::new();
        continued.messages = Arc::new(round1_messages.clone());
        continued.user("six");
        continued.user("seven");
        store.append_new(&continued.messages).unwrap();
        assert_eq!(store.active_ids().len(), 5);

        // Round 2: fold everything except the very last message ("seven") — dropped = 3.
        let round2_messages = vec![
            Message::user(format!(
                "{}\n\nsecond summary",
                agent_core::compaction::SUMMARY_MARKER
            )),
            continued.messages[4].clone(), // "seven"
        ];
        store
            .rewrite_compacted(
                &round2_messages,
                CompactionMeta {
                    tokens_before: 200,
                    provenance: Default::default(),
                },
            )
            .unwrap();

        // Both rounds' provenance accumulate independently — not overwritten.
        assert_eq!(store.meta().compactions, 2);
        assert_eq!(store.meta().dropped_messages, 5); // 2 + 3
        assert_eq!(store.active_ids().len(), 2);

        // The materialized session (after a reopen) reflects only the *latest* cut.
        let (reopened, restored) = repo.open_id(&store.meta().id.clone()).unwrap();
        assert_eq!(restored.messages.len(), 2);
        assert!(
            matches!(&restored.messages[0].content[0], ContentBlock::Text { text, .. }
                if text.contains("second summary")),
            "the active path must show the second (latest) summary, not the first: {:?}",
            restored.messages[0].content
        );

        // Exactly two `Entry::Compaction` provenance records — one per round, independently readable.
        let raw = fs::read_to_string(&reopened.path).unwrap();
        let lines: Vec<Value> = raw
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        let compaction_entries: Vec<&Value> = lines
            .iter()
            .filter(|v| v["type"] == json!("compaction"))
            .collect();
        assert_eq!(
            compaction_entries.len(),
            2,
            "expected one entry per round: {raw}"
        );
        assert_eq!(compaction_entries[0]["summary"], json!("first summary"));
        assert_eq!(compaction_entries[1]["summary"], json!("second summary"));

        // Round 1's folded originals ("one","two","three") are still physically present, untouched
        // by round 2.
        for (id, text) in round1_ids[..3].iter().zip(["one", "two", "three"]) {
            let found = lines.iter().find(|v| v["id"] == json!(id));
            assert!(
                found.is_some(),
                "round 1's folded message {id} ({text}) missing: {raw}"
            );
        }
    }

    #[test]
    fn rewrite_compacted_falls_back_to_a_plain_rewrite_when_nothing_was_folded() {
        // B-L6 pi-parity test gap (fixed): a degenerate "compaction" that doesn't actually shrink the
        // active path (e.g. new message count matching the old one) has no folded prefix worth
        // recording — `rewrite_compacted` must fall back to a plain `rewrite`, not write a
        // meaningless `Entry::Compaction` record naming zero folded messages.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("one");
        session.user("two");
        store.append_new(&session.messages).unwrap();

        // Same length in as out — nothing folded.
        let same_length = vec![
            Message::user("one (rewritten)"),
            Message::user("two (rewritten)"),
        ];
        store
            .rewrite_compacted(
                &same_length,
                CompactionMeta {
                    tokens_before: 1,
                    provenance: Default::default(),
                },
            )
            .unwrap();

        assert_eq!(
            store.meta().compactions,
            0,
            "nothing was folded; must not count as a compaction"
        );
        assert_eq!(store.meta().dropped_messages, 0);
        assert_eq!(store.active_ids().len(), 2);

        let (reopened, restored) = repo.open_id(&store.meta().id.clone()).unwrap();
        assert_eq!(restored.messages.len(), 2);
        let ContentBlock::Text { text, .. } = &restored.messages[0].content[0] else {
            panic!("expected text");
        };
        assert_eq!(text, "one (rewritten)");

        // No `Entry::Compaction` record at all.
        let raw = fs::read_to_string(&reopened.path).unwrap();
        let compaction_entries = raw
            .lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .filter(|v| v["type"] == json!("compaction"))
            .count();
        assert_eq!(
            compaction_entries, 0,
            "a no-op fold must not write a meaningless compaction provenance record: {raw}"
        );
    }

    #[test]
    fn switch_active_with_summary_appends_without_rewriting_existing_bytes() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("only message");
        store.append_new(&session.messages).unwrap();
        let root_id = store.active_ids()[0].clone();
        let path = repo.find_path(&store.meta().id.clone()).unwrap().unwrap();
        let before = fs::read(&path).unwrap();

        store
            .switch_active_with_summary(
                &root_id,
                "a recap",
                "abandoned-tip",
                BranchSummaryDetails::default(),
            )
            .unwrap();

        let after = fs::read(&path).unwrap();
        assert!(
            after.starts_with(&before),
            "every pre-existing byte must survive untouched as a prefix of the file"
        );
        assert!(
            after.len() > before.len(),
            "new content must have been appended"
        );
    }

    #[test]
    fn torn_compaction_append_leaves_the_pre_compaction_state_fully_valid() {
        // A crash between the `Entry::Compaction` record and the new active-path messages (or mid-way
        // through one of those messages) must not corrupt anything: the *old* tip and everything it
        // points to were never touched by an append-only compaction, so they must still resolve exactly
        // as they did before the interrupted compaction ever started.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        for text in ["one", "two", "three", "four"] {
            session.user(text);
        }
        store.append_new(&session.messages).unwrap();
        let old_ids = store.active_ids().to_vec();
        let id = store.meta().id.clone();
        let path = repo.find_path(&id).unwrap().unwrap();

        // Simulate a crash mid-append: a torn, unterminated compaction-record line with nothing after
        // it — as if the process died right after starting to write the batch.
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        write!(f, "{{\"type\":\"compaction\",\"id\":\"x").unwrap();
        drop(f);

        let (reopened, restored) = repo.open_id(&id).unwrap();
        assert_eq!(
            restored.messages.len(),
            4,
            "the pre-compaction transcript must be fully intact"
        );
        assert_eq!(
            reopened.active_ids(),
            old_ids.as_slice(),
            "the tip must still be the pre-compaction tip — an interrupted compaction simply never took effect"
        );
    }

    #[test]
    fn list_is_newest_first_and_fork_links_parent() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut a = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut sa = Session::new();
        sa.user("one");
        sa.user("two");
        a.append_new(&sa.messages).unwrap();
        let id_a = a.meta().id.clone();

        // Fork the first message only.
        let (forked, fsession) = repo.fork(&id_a, 1).unwrap();
        assert_eq!(fsession.messages.len(), 1);
        assert_eq!(forked.meta().parent.as_deref(), Some(id_a.as_str()));

        let metas = repo.list().unwrap();
        assert_eq!(metas.len(), 2);
        // Newest first: the fork was created after `a`.
        assert!(metas[0].created_at >= metas[1].created_at);
    }

    #[test]
    fn fork_from_path_copies_a_source_from_an_unrelated_repo_and_stamps_the_target_cwd() {
        // The cross-project primitive `fork_by_arg` builds on: the source need not live in `self.dir`
        // at all, and the new session's `cwd` is the *target* project, not wherever the source was
        // originally recorded against — matching pi's `forkFrom(path, targetCwd, …)`.
        let source_dir = tmpdir();
        let source_repo = SessionRepo::open(source_dir.path()).unwrap();
        let mut source = source_repo
            .create(SessionMeta::new("/some/other/project", "m"))
            .unwrap();
        let mut s = Session::new();
        s.user("hello from project A");
        source.append_new(&s.messages).unwrap();
        let source_id = source.meta().id.clone();
        let source_path = source_dir
            .path()
            .read_dir()
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();

        let target_dir = tmpdir();
        let target_repo = SessionRepo::open(target_dir.path()).unwrap();
        let (forked, fsession) = target_repo
            .fork_from_path(&source_path, "/current/project", usize::MAX)
            .unwrap();

        assert_eq!(fsession.messages.len(), 1);
        assert_eq!(forked.meta().parent.as_deref(), Some(source_id.as_str()));
        assert_eq!(forked.meta().cwd, "/current/project");
        // The source file itself is untouched — this is a copy, not a move.
        let (_, resumed_source) = SessionStore::open(source_path).unwrap();
        assert_eq!(resumed_source.messages.len(), 1);
    }

    #[test]
    fn fork_by_arg_with_a_path_like_argument_opens_that_file_directly_no_search() {
        let source_dir = tmpdir();
        let source_repo = SessionRepo::open(source_dir.path()).unwrap();
        source_repo
            .create(SessionMeta::new("/wherever", "m"))
            .unwrap();
        let source_path = source_dir
            .path()
            .read_dir()
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();

        let target_dir = tmpdir();
        let target_repo = SessionRepo::open(target_dir.path()).unwrap();
        // An empty `sessions_root` proves this path never falls through to the cross-project search —
        // a path-like argument is resolved directly.
        let empty_root = tmpdir();
        let (forked, _) = fork_by_arg(
            source_path.to_str().unwrap(),
            &target_repo,
            "/current",
            empty_root.path(),
            usize::MAX,
        )
        .unwrap();
        assert_eq!(forked.meta().cwd, "/current");
    }

    #[test]
    fn fork_by_arg_with_an_id_present_in_the_current_project_forks_in_place() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let source = repo.create(SessionMeta::new("/current", "m")).unwrap();
        let source_id = source.meta().id.clone();

        // No other project directory exists at all — proves the same-project match is tried (and
        // wins) before any cross-project search would even be attempted.
        let empty_root = tmpdir();
        let (forked, _) =
            fork_by_arg(&source_id, &repo, "/current", empty_root.path(), usize::MAX).unwrap();
        assert_eq!(forked.meta().parent.as_deref(), Some(source_id.as_str()));
        assert_eq!(forked.meta().cwd, "/current");
    }

    #[test]
    fn fork_by_arg_with_an_id_found_in_no_project_is_a_clear_not_found_error() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let empty_root = tmpdir();
        let result = fork_by_arg(
            "no-such-id",
            &repo,
            "/current",
            empty_root.path(),
            usize::MAX,
        );
        let err = match result {
            Ok(_) => panic!("expected a not-found error"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        assert!(err.to_string().contains("no-such-id"), "{err}");
    }

    #[test]
    fn fork_by_arg_falls_back_to_a_session_found_in_a_different_projects_own_directory() {
        // The cross-project search itself, exercised directly against a synthetic `sessions_root`
        // rather than the real `$HOME` (which `default_session_dir` reads) — the end-to-end version of
        // this same fallback, wired through the real default directory, lives in the `run` binary's own
        // `--fork` e2e tests.
        let root = tmpdir();
        let other_project_repo =
            SessionRepo::open(root.path().join(encode_cwd("/some/other/project"))).unwrap();
        let source = other_project_repo
            .create(SessionMeta::new("/some/other/project", "m"))
            .unwrap();
        let source_id = source.meta().id.clone();

        let current_repo =
            SessionRepo::open(root.path().join(encode_cwd("/current/project"))).unwrap();
        let (forked, _) = fork_by_arg(
            &source_id,
            &current_repo,
            "/current/project",
            root.path(),
            usize::MAX,
        )
        .unwrap();
        assert_eq!(forked.meta().parent.as_deref(), Some(source_id.as_str()));
        assert_eq!(
            forked.meta().cwd,
            "/current/project",
            "the fork's cwd must be the current project, not the source's original one"
        );
    }

    #[test]
    fn fork_by_arg_finds_a_cross_project_session_even_when_its_directory_name_does_not_encode_its_cwd()
     {
        // Pi-parity fix (`run --session-dir`/`AI_AGENT_SESSION_DIR`): the cross-project fallback used to
        // recompute the source's directory from its own recorded `cwd` via
        // `default_session_dir`/`encode_cwd`, which only holds for the *default* `~/.claude/sessions/
        // <encoded-cwd>/` layout. An arbitrarily-named directory (exactly what `--session-dir` lets a
        // caller point at) broke that assumption and the fork would fail with a bogus "not found" even
        // though the session was sitting right there under `sessions_root`.
        let root = tmpdir();
        // Deliberately an arbitrary name, unrelated to `encode_cwd` of the session's own `cwd` below.
        let other_project_repo =
            SessionRepo::open(root.path().join("arbitrarily-named-repo")).unwrap();
        let source = other_project_repo
            .create(SessionMeta::new("/some/other/project", "m"))
            .unwrap();
        let source_id = source.meta().id.clone();

        let current_repo = SessionRepo::open(root.path().join("another-arbitrary-name")).unwrap();
        let (forked, _) = fork_by_arg(
            &source_id,
            &current_repo,
            "/current/project",
            root.path(),
            usize::MAX,
        )
        .unwrap();
        assert_eq!(forked.meta().parent.as_deref(), Some(source_id.as_str()));
        assert_eq!(forked.meta().cwd, "/current/project");
    }

    #[test]
    fn find_path_resolves_a_unique_prefix_when_no_exact_match_exists() {
        // Pi-parity fix: matching used to be exact-only everywhere — a caller had to type the full id,
        // unlike pi's own `resolveSessionPath`, which accepts a shortened prefix.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let id = store.meta().id.clone();
        let prefix = &id[..id.len() / 2];

        let (_, restored) = repo.open_id(prefix).unwrap();
        assert_eq!(restored.messages.len(), 0);
    }

    #[test]
    fn find_path_prefers_an_exact_match_over_a_prefix_match() {
        // An id that happens to *also* be a prefix of some other session's id must still resolve to
        // itself exactly, not get shadowed by the ambiguity check below.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let short = repo.create(SessionMeta::with_id("abc", "/w", "m")).unwrap();
        repo.create(SessionMeta::with_id("abcdef", "/w", "m"))
            .unwrap();

        let (_, _) = repo.open_id("abc").unwrap();
        let path = repo.find_path("abc").unwrap().unwrap();
        assert!(path.to_string_lossy().contains(&short.meta().id));
    }

    #[test]
    fn find_path_reports_an_ambiguous_prefix_naming_every_candidate_instead_of_guessing() {
        // Pi's own `resolveSessionPath` silently picks whichever candidate sorts first on an ambiguous
        // prefix — a real footgun (acting on the wrong session with no warning). This crate errors
        // instead, naming every id that matched.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        repo.create(SessionMeta::with_id("abc111", "/w", "m"))
            .unwrap();
        repo.create(SessionMeta::with_id("abc222", "/w", "m"))
            .unwrap();

        let err = repo.open_id("abc").map(|_| ()).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        let msg = err.to_string();
        assert!(msg.contains("abc111") && msg.contains("abc222"), "{msg}");
    }

    #[test]
    fn find_path_returns_not_found_for_a_prefix_matching_nothing_at_all() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        repo.create(SessionMeta::with_id("abc111", "/w", "m"))
            .unwrap();

        let err = repo.open_id("zzz").map(|_| ()).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }

    #[test]
    fn delete_by_prefix_removes_the_uniquely_matched_session() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let id = store.meta().id.clone();
        let prefix = &id[..id.len() / 2];

        repo.delete(prefix).unwrap();
        assert!(repo.list().unwrap().is_empty());
    }

    #[test]
    fn delete_by_an_ambiguous_prefix_errors_instead_of_silently_no_opping() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        repo.create(SessionMeta::with_id("abc111", "/w", "m"))
            .unwrap();
        repo.create(SessionMeta::with_id("abc222", "/w", "m"))
            .unwrap();

        let err = repo.delete("abc").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        // Neither candidate was touched.
        assert_eq!(repo.list().unwrap().len(), 2);
    }

    #[test]
    fn fork_by_arg_resolves_a_prefix_within_the_current_project_without_a_cross_project_search() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let store = repo.create(SessionMeta::new("/current", "m")).unwrap();
        let id = store.meta().id.clone();
        let prefix = &id[..id.len() / 2];

        // An empty `sessions_root` proves the cross-project fallback was never needed — the prefix
        // resolved within `repo` itself.
        let empty_root = tmpdir();
        let (forked, _) =
            fork_by_arg(prefix, &repo, "/current", empty_root.path(), usize::MAX).unwrap();
        assert_eq!(forked.meta().parent.as_deref(), Some(id.as_str()));
    }

    #[test]
    fn fork_by_arg_finds_a_unique_prefix_across_projects() {
        let root = tmpdir();
        let other_repo = SessionRepo::open(root.path().join("other-project")).unwrap();
        let source = other_repo
            .create(SessionMeta::new("/other/project", "m"))
            .unwrap();
        let source_id = source.meta().id.clone();
        let prefix = &source_id[..source_id.len() / 2];

        let current_repo = SessionRepo::open(root.path().join("current-project")).unwrap();
        let (forked, _) = fork_by_arg(
            prefix,
            &current_repo,
            "/current/project",
            root.path(),
            usize::MAX,
        )
        .unwrap();
        assert_eq!(forked.meta().parent.as_deref(), Some(source_id.as_str()));
    }

    #[test]
    fn list_with_progress_reports_every_file_exactly_once_up_to_the_total() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        for i in 0..12 {
            repo.create(SessionMeta::new(format!("/w{i}"), "m"))
                .unwrap();
        }

        let calls: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
        let metas = repo
            .list_with_progress(|scanned, total| calls.lock().unwrap().push((scanned, total)))
            .unwrap();
        assert_eq!(metas.len(), 12, "parallel scan must not drop any file");

        let calls = calls.into_inner().unwrap();
        assert_eq!(calls.len(), 12, "exactly one progress call per file");
        assert!(
            calls.iter().all(|&(_, total)| total == 12),
            "total must be stable across every call: {calls:?}"
        );
        let mut scanned: Vec<usize> = calls.iter().map(|&(s, _)| s).collect();
        scanned.sort_unstable();
        assert_eq!(
            scanned,
            (1..=12).collect::<Vec<_>>(),
            "scanned counts must cover 1..=total exactly once each, even with concurrent workers: {scanned:?}"
        );
    }

    #[test]
    fn list_with_progress_returns_the_same_sessions_as_list() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        for i in 0..8 {
            repo.create(SessionMeta::new(format!("/w{i}"), "m"))
                .unwrap();
        }

        let mut plain: Vec<String> = repo.list().unwrap().into_iter().map(|m| m.id).collect();
        let mut via_progress: Vec<String> = repo
            .list_with_progress(|_, _| {})
            .unwrap()
            .into_iter()
            .map(|m| m.id)
            .collect();
        plain.sort();
        via_progress.sort();
        assert_eq!(plain, via_progress);
    }

    #[test]
    fn list_with_progress_on_an_empty_repo_calls_nothing_and_returns_empty() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let calls = Mutex::new(0usize);
        let metas = repo
            .list_with_progress(|_, _| *calls.lock().unwrap() += 1)
            .unwrap();
        assert!(metas.is_empty());
        assert_eq!(*calls.lock().unwrap(), 0);
    }

    #[test]
    fn list_all_with_progress_spans_every_project_in_one_pool() {
        let root = tmpdir();
        let repo_a = SessionRepo::open(root.path().join("proj-a")).unwrap();
        for i in 0..5 {
            repo_a
                .create(SessionMeta::new(format!("/a{i}"), "m"))
                .unwrap();
        }
        let repo_b = SessionRepo::open(root.path().join("proj-b")).unwrap();
        for i in 0..5 {
            repo_b
                .create(SessionMeta::new(format!("/b{i}"), "m"))
                .unwrap();
        }

        let calls: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
        let metas = SessionRepo::list_all_with_progress(root.path(), |scanned, total| {
            calls.lock().unwrap().push((scanned, total))
        })
        .unwrap();
        assert_eq!(metas.len(), 10, "must merge both projects' sessions");

        let calls = calls.into_inner().unwrap();
        assert_eq!(
            calls.len(),
            10,
            "progress spans both projects, not per-project"
        );
        assert!(calls.iter().all(|&(_, total)| total == 10));
    }

    #[test]
    fn resume_or_create_reopens_the_session_matching_cwd() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut other = repo.create(SessionMeta::new("/other", "m")).unwrap();
        let mut sa = Session::new();
        sa.user("from another project");
        other.append_new(&sa.messages).unwrap();

        let mut mine = repo.create(SessionMeta::new("/mine", "m")).unwrap();
        let mut sb = Session::new();
        sb.user("from my project");
        mine.append_new(&sb.messages).unwrap();

        let (store, session) = repo.resume_or_create("/mine", "m", None).unwrap();
        assert_eq!(store.meta().id, mine.meta().id);
        assert_eq!(session.messages.len(), 1);
    }

    #[test]
    fn resume_or_create_makes_a_fresh_session_when_no_cwd_matches() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let (store, session) = repo.resume_or_create("/brand/new", "m", None).unwrap();
        assert_eq!(store.meta().cwd, "/brand/new");
        assert!(session.messages.is_empty());
    }

    #[test]
    fn resume_or_create_uses_the_given_id_for_a_genuinely_fresh_session() {
        // Backs `serve`'s own `--session-id` flag (pi-parity: `run` already had this, `serve` didn't).
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let (store, session) = repo
            .resume_or_create("/brand/new", "m", Some("my-chosen-id"))
            .unwrap();
        assert_eq!(store.meta().id, "my-chosen-id");
        assert!(session.messages.is_empty());
    }

    #[test]
    fn resume_or_create_ignores_the_given_id_when_an_existing_session_matches_the_cwd() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut mine = repo.create(SessionMeta::new("/mine", "m")).unwrap();
        let mut s = Session::new();
        s.user("already here");
        mine.append_new(&s.messages).unwrap();

        let (store, session) = repo
            .resume_or_create("/mine", "m", Some("should-be-ignored"))
            .unwrap();
        assert_eq!(
            store.meta().id,
            mine.meta().id,
            "an existing session's own id must win over a caller-supplied one"
        );
        assert_eq!(session.messages.len(), 1);
    }

    #[test]
    fn canonical_cwd_resolves_a_symlink_to_its_real_target() {
        let dir = tmpdir();
        let real = dir.path().join("real");
        fs::create_dir(&real).unwrap();
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();

        assert_eq!(canonical_cwd(&link), canonical_cwd(&real));
    }

    #[test]
    fn canonical_cwd_falls_back_to_the_given_path_when_it_does_not_exist() {
        let missing = Path::new("/definitely/does/not/exist/xyz-canonical-cwd-test");
        assert_eq!(canonical_cwd(missing), missing);
    }

    #[test]
    fn resume_or_create_matches_a_session_recorded_under_a_symlinked_cwd_once_canonicalized() {
        // The regression this guards: a project reached through a symlink one time and its real path
        // another must resolve to the same session, not silently fork into two.
        let dir = tmpdir();
        let real = dir.path().join("project");
        fs::create_dir(&real).unwrap();
        let link = dir.path().join("project-link");
        std::os::unix::fs::symlink(&real, &link).unwrap();

        let repo_dir = tmpdir();
        let repo = SessionRepo::open(repo_dir.path()).unwrap();
        let real_cwd = canonical_cwd(&real).to_string_lossy().into_owned();
        let mut store = repo.create(SessionMeta::new(&real_cwd, "m")).unwrap();
        let mut s = Session::new();
        s.user("hello from the real path");
        store.append_new(&s.messages).unwrap();

        let link_cwd = canonical_cwd(&link).to_string_lossy().into_owned();
        let (reopened, session) = repo.resume_or_create(&link_cwd, "m", None).unwrap();
        assert_eq!(
            reopened.meta().id,
            store.meta().id,
            "a symlinked cwd must canonicalize to the same session as its real path"
        );
        assert_eq!(session.messages.len(), 1);
    }

    #[test]
    fn list_all_merges_every_project_subdirectory() {
        // The layout `serve`'s default session directory follows: one subdirectory per project under
        // a shared root, each an independent `SessionRepo`.
        let root = tmpdir();
        let repo_a = SessionRepo::open(root.path().join("-home-jared-project-a")).unwrap();
        repo_a
            .create(SessionMeta::new("/home/jared/project-a", "m"))
            .unwrap();
        let repo_b = SessionRepo::open(root.path().join("-home-jared-project-b")).unwrap();
        repo_b
            .create(SessionMeta::new("/home/jared/project-b", "m"))
            .unwrap();

        let all = SessionRepo::list_all(root.path()).unwrap();
        assert_eq!(all.len(), 2);
        let cwds: std::collections::HashSet<&str> = all.iter().map(|m| m.cwd.as_str()).collect();
        assert!(cwds.contains("/home/jared/project-a"));
        assert!(cwds.contains("/home/jared/project-b"));
    }

    #[test]
    fn list_all_is_newest_first_across_projects() {
        let root = tmpdir();
        let repo_a = SessionRepo::open(root.path().join("proj-a")).unwrap();
        let older = repo_a.create(SessionMeta::new("/a", "m")).unwrap();
        let repo_b = SessionRepo::open(root.path().join("proj-b")).unwrap();
        let newer = repo_b.create(SessionMeta::new("/b", "m")).unwrap();

        // Stamp mtimes explicitly (the sort key, `updated_at`, falls back to mtime for a session with
        // no messages) rather than relying on real wall-clock ordering between two back-to-back
        // creates, which can tie at second granularity.
        let earlier = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000_000);
        let later = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000_100);
        fs::File::options()
            .write(true)
            .open(repo_a.path_for(older.meta()))
            .unwrap()
            .set_modified(earlier)
            .unwrap();
        fs::File::options()
            .write(true)
            .open(repo_b.path_for(newer.meta()))
            .unwrap()
            .set_modified(later)
            .unwrap();

        let all = SessionRepo::list_all(root.path()).unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(
            all[0].cwd, "/b",
            "the more recently active session must sort first"
        );
        assert_eq!(all[1].cwd, "/a");
    }

    #[test]
    fn list_sorts_by_last_activity_not_creation_time() {
        // `older` is created first but is the one touched most recently — a listing ordered by last
        // activity (matching pi's own session list, sorted by `modified`) must surface it first
        // despite `newer_by_creation` having a later `created_at`.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let older = repo.create(SessionMeta::new("/older", "m")).unwrap();
        let older_id = older.meta().id.clone();
        let newer_by_creation = repo.create(SessionMeta::new("/newer", "m")).unwrap();
        let newer_id = newer_by_creation.meta().id.clone();

        let earlier = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000_000);
        let later = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000_100);
        fs::File::options()
            .write(true)
            .open(repo.path_for(newer_by_creation.meta()))
            .unwrap()
            .set_modified(earlier)
            .unwrap();
        fs::File::options()
            .write(true)
            .open(repo.path_for(older.meta()))
            .unwrap()
            .set_modified(later)
            .unwrap();

        let metas = repo.list().unwrap();
        assert_eq!(
            metas[0].id, older_id,
            "last-activity ordering must beat creation order"
        );
        assert_eq!(metas[1].id, newer_id);
    }

    #[test]
    fn list_all_ignores_non_repo_entries_and_a_missing_root() {
        let root = tmpdir();
        // A stray file at the root (not a project subdirectory) must not break the scan.
        std::fs::write(root.path().join("not-a-project.txt"), "x").unwrap();
        // An empty subdirectory (a project with no sessions yet) contributes nothing, not an error.
        std::fs::create_dir(root.path().join("empty-project")).unwrap();
        let repo = SessionRepo::open(root.path().join("real-project")).unwrap();
        repo.create(SessionMeta::new("/real", "m")).unwrap();

        let all = SessionRepo::list_all(root.path()).unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].cwd, "/real");

        // A root that doesn't exist at all (nothing has ever been persisted) is an empty list, not an
        // I/O error.
        let missing = root.path().join("does-not-exist");
        assert!(SessionRepo::list_all(&missing).unwrap().is_empty());
    }

    #[test]
    fn set_title_persists() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let id = store.meta().id.clone();
        let mut session = Session::new();
        session.user("hi");
        store.append_new(&session.messages).unwrap();
        store.set_title("My Session").unwrap();
        let metas = repo.list().unwrap();
        let found = metas.iter().find(|m| m.id == id).unwrap();
        assert_eq!(found.title.as_deref(), Some("My Session"));
        // A pure title rewrite drops no messages, so it leaves no compaction provenance.
        assert_eq!(found.compactions, 0);
    }

    #[test]
    fn set_title_appends_rather_than_rewriting_the_file() {
        // The whole point of Track M17: renaming must cost an O(1) append, not a rewrite of every
        // message already on disk. Prove it directly — the bytes written before the rename are still
        // there afterward, byte for byte, with the rename's line appended after them.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("hi");
        store.append_new(&session.messages).unwrap();

        let before = std::fs::read(&store.path).unwrap();
        store.set_title("Renamed").unwrap();
        let after = std::fs::read(&store.path).unwrap();

        assert!(
            after.starts_with(&before),
            "a rename must only append — every byte written before it must survive unchanged"
        );
        assert!(
            after.len() > before.len(),
            "the rename must add at least the new TitleChange line"
        );
    }

    #[test]
    fn set_title_last_write_wins_and_survives_reopen() {
        let dir = tmpdir();
        let path = dir.path().join("s.jsonl");
        let mut store = SessionStore::create(path.clone(), SessionMeta::new("/w", "m")).unwrap();
        store.set_title("First").unwrap();
        store.set_title("Second").unwrap();
        assert_eq!(store.meta().title.as_deref(), Some("Second"));

        // Both `SessionStore::open` (the tree-building read) and `read_listing` (the cheap listing
        // scan) must independently resolve the *last* rename, not the first.
        let (reopened, _session) = SessionStore::open(path.clone()).unwrap();
        assert_eq!(reopened.meta().title.as_deref(), Some("Second"));

        let listed = read_listing(&path).unwrap();
        assert_eq!(listed.title.as_deref(), Some("Second"));
    }

    #[test]
    fn set_title_strips_newlines() {
        // Matches pi's `appendSessionInfo` sanitization: a raw newline in a caller-supplied title
        // (RPC client, extension) would otherwise split a session-list line or corrupt a terminal
        // display — collapse any run of them into a single space instead.
        let dir = tmpdir();
        let path = dir.path().join("s.jsonl");
        let mut store = SessionStore::create(path.clone(), SessionMeta::new("/w", "m")).unwrap();
        store.set_title("hello\nworld\r\nagain").unwrap();
        assert_eq!(store.meta().title.as_deref(), Some("hello world again"));

        let (reopened, _session) = SessionStore::open(path.clone()).unwrap();
        assert_eq!(reopened.meta().title.as_deref(), Some("hello world again"));
        let listed = read_listing(&path).unwrap();
        assert_eq!(listed.title.as_deref(), Some("hello world again"));
    }

    #[test]
    fn set_title_to_empty_or_whitespace_clears_the_title() {
        // Matches pi's "empty names explicitly clear the session title" — a blank title isn't a
        // present-but-empty display value, it's the caller asking to unset the title entirely.
        let dir = tmpdir();
        let path = dir.path().join("s.jsonl");
        let mut store = SessionStore::create(path.clone(), SessionMeta::new("/w", "m")).unwrap();
        store.set_title("My Session").unwrap();
        assert_eq!(store.meta().title.as_deref(), Some("My Session"));

        store.set_title("   \n\r\n  ").unwrap();
        assert_eq!(store.meta().title, None);

        let (reopened, _session) = SessionStore::open(path.clone()).unwrap();
        assert_eq!(reopened.meta().title, None);
        let listed = read_listing(&path).unwrap();
        assert_eq!(listed.title, None);
    }

    #[test]
    fn delete_removes_session() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let id = store.meta().id.clone();
        assert_eq!(repo.list().unwrap().len(), 1);
        repo.delete(&id).unwrap();
        assert_eq!(repo.list().unwrap().len(), 0);
    }

    #[test]
    fn delete_is_idempotent() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let id = store.meta().id.clone();

        // First delete removes it; a second delete (and a delete of a never-seen id) are no-op successes.
        repo.delete(&id).unwrap();
        repo.delete(&id).unwrap();
        repo.delete("does-not-exist").unwrap();
        assert!(repo.list().unwrap().is_empty());
    }

    #[test]
    fn delete_moves_to_trash_and_list_no_longer_shows_it() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let id = store.meta().id.clone();
        let original_path = store.path.clone();

        repo.delete(&id).unwrap();

        assert!(
            repo.list().unwrap().is_empty(),
            "a trashed session must not appear in list()"
        );
        assert!(
            !original_path.exists(),
            "the file must no longer be at its original path"
        );
        let trash_dir = dir.path().join(".trash");
        let trashed: Vec<_> = fs::read_dir(&trash_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(
            trashed.len(),
            1,
            "the deleted session's file should have moved into .trash: {trashed:?}"
        );
        assert_eq!(trashed[0].file_name(), original_path.file_name().unwrap());
    }

    #[test]
    fn list_trash_is_empty_before_anything_is_deleted() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        repo.create(SessionMeta::new("/w", "m")).unwrap();
        assert!(
            repo.list_trash().unwrap().is_empty(),
            "nothing has been deleted yet, so .trash/ doesn't even exist"
        );
    }

    #[test]
    fn list_trash_reports_a_deleted_session_by_id() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let id = store.meta().id.clone();

        repo.delete(&id).unwrap();

        let trash = repo.list_trash().unwrap();
        assert_eq!(trash.len(), 1);
        assert_eq!(trash[0].id, id);
        assert!(trash[0].deleted_at.is_some());
        assert!(trash[0].original_path.ends_with(&format!("_{id}.jsonl")));
    }

    #[test]
    fn restore_session_moves_a_trashed_session_back_and_makes_it_listable_again() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let id = store.meta().id.clone();

        repo.delete(&id).unwrap();
        assert!(repo.list().unwrap().is_empty());

        let restored = repo.restore_session(&id).unwrap();
        assert!(
            restored,
            "restore_session must report it actually moved something"
        );
        assert!(
            repo.list_trash().unwrap().is_empty(),
            "the entry must no longer be in .trash/ once restored"
        );
        let listed = repo.list().unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, id);
    }

    #[test]
    fn restore_session_of_an_unknown_id_returns_false_not_an_error() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        assert!(!repo.restore_session("never-existed").unwrap());
    }

    #[test]
    fn restore_session_fails_clearly_when_the_destination_is_already_occupied() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let id = store.meta().id.clone();
        let original_path = store.path.clone();

        repo.delete(&id).unwrap();
        // Something else now occupies the original path (e.g. a new session using a colliding id is
        // vanishingly unlikely in practice, but a hand-placed file is enough to prove the guard works).
        fs::write(&original_path, "occupied").unwrap();

        let err = repo.restore_session(&id).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::AlreadyExists);
        // The trashed copy must still be sitting in `.trash/` untouched — a failed restore is not a
        // silent delete of the only remaining copy.
        assert_eq!(repo.list_trash().unwrap().len(), 1);
    }

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
        // Unlike `preview` (first message only), `search_text` accumulates every user message.
        assert!(
            l.search_text
                .contains("hello world, this is the first message")
        );
        assert!(l.search_text.contains("second"));
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
    fn search_text_is_capped() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("x".repeat(SEARCH_TEXT_MAX_CHARS + 500));
        session.user("this must not appear: budget already spent");
        store.append_new(&session.messages).unwrap();

        let listings = repo.list().unwrap();
        let search_text = &listings[0].search_text;
        assert!(
            search_text.chars().count() <= SEARCH_TEXT_MAX_CHARS,
            "got {} chars",
            search_text.chars().count()
        );
        assert!(!search_text.contains("this must not appear"));
    }

    #[test]
    fn search_text_captures_a_marker_well_beyond_the_original_2000_char_cap() {
        // Track L7: the cap was raised from 2,000 to 50,000 chars — a session whose distinguishing
        // text lands past where the *old* cap would have cut it off must still be findable. This
        // padding is comfortably past 2,000 chars but still well under the new cap.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("x".repeat(10_000));
        session.user("findable-only-past-the-old-cap-marker");
        store.append_new(&session.messages).unwrap();

        let listings = repo.list().unwrap();
        assert!(
            listings[0]
                .search_text
                .contains("findable-only-past-the-old-cap-marker"),
            "a marker past the old 2,000-char cap must still be captured under the raised cap"
        );
    }

    #[test]
    fn search_text_includes_assistant_replies_not_just_user_turns() {
        // The whole point of Track M18: a session must be findable by something only the *assistant*
        // said — matching pi's own `allMessagesText` (which joins every user AND assistant message's
        // text, not just the user's).
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("what's broken?");
        session.push(Message::assistant(vec![ContentBlock::text(
            "the bug is in unique-marker-xyz.rs",
        )]));
        store.append_new(&session.messages).unwrap();

        let listings = repo.list().unwrap();
        assert!(
            listings[0].search_text.contains("unique-marker-xyz.rs"),
            "search_text must include assistant text: {:?}",
            listings[0].search_text
        );
    }

    #[test]
    fn search_text_is_empty_when_the_session_has_no_user_text() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let id = store.meta().id.clone();
        let listings = repo.list().unwrap();
        let l = listings.iter().find(|l| l.id == id).unwrap();
        assert_eq!(l.search_text, "");
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
        assert!(bare.get("search_text").is_none());

        let full = listing.to_listing_json();
        assert_eq!(full["message_count"], 1);
        assert_eq!(full["preview"], "hello world");
        assert_eq!(full["search_text"], "hello world");
        assert!(full["updated_at"].as_u64().unwrap() > 0);
        // Persisted fields still round-trip through the same call.
        assert_eq!(full["id"], listing.id);
        assert_eq!(full["cwd"], "/w");
    }

    fn meta_for_search(
        id: &str,
        title: Option<&str>,
        search_text: &str,
        updated_at: u64,
    ) -> SessionMeta {
        let mut m = SessionMeta::with_id(id, "/w", "m");
        m.title = title.map(str::to_string);
        m.search_text = search_text.to_string();
        m.updated_at = updated_at;
        m
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
    fn search_sessions_ranks_a_title_match_above_a_search_text_only_match() {
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
            "a title hit must outrank a search_text-only hit even though it's less recent"
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

    #[test]
    fn newer_version_is_rejected() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let id = store.meta().id.clone();
        let path = repo.find_path(&id).unwrap().unwrap();

        // Rewrite the header with a version newer than this build understands.
        let content = fs::read_to_string(&path).unwrap();
        let mut lines = content.lines();
        let mut header: Value = serde_json::from_str(lines.next().unwrap()).unwrap();
        header["version"] = serde_json::json!(999);
        let rest: Vec<&str> = lines.collect();
        fs::write(&path, format!("{}\n{}", header, rest.join("\n"))).unwrap();

        // Opening errors clearly rather than mis-parsing... (`map` drops the non-`Debug` success value
        // so `unwrap_err` can format it).
        let err = repo.open_id(&id).map(|_| ()).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        // ...and the listing skips the unreadable file instead of including it.
        assert!(repo.list().unwrap().is_empty());
    }

    #[test]
    fn pre_versioning_header_with_no_version_field_opens_as_v1() {
        // A header written before `version` existed omits the field entirely (not `version: 0` — the
        // key is just absent, and `#[serde(default)]` fills it with `0` on read). `migrate`'s `0 |
        // VERSION => Ok(meta)` arm treats that as wire-compatible with v1; this is the one arm of the
        // match `newer_version_is_rejected` doesn't exercise.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let id = store.meta().id.clone();
        let path = repo.find_path(&id).unwrap().unwrap();

        // Rewrite the header with `version` removed entirely, simulating a file from before the field
        // existed.
        let content = fs::read_to_string(&path).unwrap();
        let mut lines = content.lines();
        let mut header: Value = serde_json::from_str(lines.next().unwrap()).unwrap();
        header.as_object_mut().unwrap().remove("version");
        let rest: Vec<&str> = lines.collect();
        fs::write(&path, format!("{}\n{}", header, rest.join("\n"))).unwrap();

        // Opens cleanly — no migration error — and the listing includes it like any other session.
        let (_store, _session) = repo.open_id(&id).unwrap();
        assert_eq!(repo.list().unwrap().len(), 1);
    }

    #[test]
    fn writes_round_trip_after_fsync() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let id = store.meta().id.clone();
        let mut session = Session::new();
        session.user("durable");
        // Exercises the synced append path...
        store.append_new(&session.messages).unwrap();
        // ...and the synced atomic-rewrite path.
        store.rewrite(&session.messages).unwrap();

        let (_store, restored) = repo.open_id(&id).unwrap();
        assert_eq!(restored.messages.len(), 1);
        let dump = serde_json::to_string(restored.messages.as_ref()).unwrap();
        assert!(dump.contains("durable"));
    }

    #[test]
    fn open_restores_a_proactive_compaction_signal_from_the_persisted_transcript() {
        // B-M14 pi-parity gap (fixed): `SessionStore::open` never restored `last_input_tokens` from
        // the persisted history — a freshly-built `Session` defaults it to 0, and `should_compact`/
        // `is_hard_overflow` both require it to be positive to fire at all. A resumed large session
        // wouldn't proactively compact until a *new* turn produced fresh real usage, a whole turn
        // later than it should (pi's own `pre-prompt-compaction-no-continue` regression test covers
        // the same gap). Usage isn't persisted per-message anywhere in this format, so the estimate
        // must come from the same char/4 heuristic `compaction::trailing_tokens` already uses
        // elsewhere for spans with no exact provider-reported figure.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let id = store.meta().id.clone();
        let mut session = Session::new();
        // Real, non-trivial text so the char/4 estimate is comfortably nonzero and easy to bound.
        session.user("a".repeat(400)); // ~100 estimated tokens
        session.push(Message::assistant(vec![ContentBlock::text(
            "b".repeat(400),
        )])); // ~100 more
        store.append_new(&session.messages).unwrap();

        let (_reopened, restored) = repo.open_id(&id).unwrap();
        assert!(
            restored.last_input_tokens > 0,
            "a resumed session must have a nonzero live-context estimate, not the zero default"
        );
        assert!(
            restored.last_input_tokens >= 190,
            "expected roughly 200 estimated tokens across both messages, got {}",
            restored.last_input_tokens
        );
        // Every persisted message must be marked as already accounted for, so a subsequent
        // `trailing_tokens` call (after a fresh turn) doesn't double-count this estimate.
        assert_eq!(restored.last_usage_message_count, restored.messages.len());
    }

    #[test]
    fn compaction_records_provenance() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let id = store.meta().id.clone();
        let mut session = Session::new();
        session.user("a");
        session.user("b");
        session.user("c");
        store.append_new(&session.messages).unwrap();

        // Compact 3 messages down to 1 — two dropped.
        store.rewrite(&[Message::user("summary")]).unwrap();
        assert_eq!(store.meta().compactions, 1);
        assert_eq!(store.meta().dropped_messages, 2);

        // Provenance is persisted in the header and survives reopen.
        let (reopened, _) = repo.open_id(&id).unwrap();
        assert_eq!(reopened.meta().compactions, 1);
        assert_eq!(reopened.meta().dropped_messages, 2);
    }

    #[test]
    fn legacy_flat_file_migrates_ids_in_memory_only() {
        // Hand-write a pre-tree file: `Message` lines with no `id`/`parent_id` fields at all — the
        // exact shape a file predating this module's tree support would have.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let meta = SessionMeta::new("/w", "m");
        let id = meta.id.clone();
        let path = repo.path_for(&meta);
        let header = serde_json::to_string(&Entry::Session(meta)).unwrap();
        let raw = format!(
            "{header}\n\
             {{\"type\":\"message\",\"role\":\"user\",\"content\":[{{\"type\":\"text\",\"text\":\"first\"}}]}}\n\
             {{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{{\"type\":\"text\",\"text\":\"second\"}}]}}\n"
        );
        fs::write(&path, raw).unwrap();

        let (store, session) = repo.open_id(&id).unwrap();
        // Reconstructs the same linear order a flat file always had.
        assert_eq!(session.messages.len(), 2);
        assert_eq!(store.active_ids().len(), 2);
        // Synthesized ids are distinct and chained (not both defaulting to the same value).
        assert_ne!(store.active_ids()[0], store.active_ids()[1]);

        // Never persisted back: re-reading the raw file, the two *message* lines still carry no `id`
        // field (the header legitimately has one — `SessionMeta.id` — so check those lines only).
        let raw_after = fs::read_to_string(&path).unwrap();
        for line in raw_after.lines().skip(1) {
            assert!(
                !line.contains("\"id\""),
                "message line gained a persisted id: {line}"
            );
        }
    }

    #[test]
    fn switch_active_then_append_forks_a_branch() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("a");
        session.user("b");
        session.user("c");
        store.append_new(&session.messages).unwrap();
        let ids = store.active_ids().to_vec();
        assert_eq!(ids.len(), 3);

        // Navigate back to the first message and fork off it.
        let branch_root = store.switch_active(&ids[0]).unwrap();
        assert_eq!(branch_root.len(), 1);
        let mut branch_session = Session::new();
        branch_session.messages = Arc::new(branch_root);
        branch_session.user("d");
        store.append_new(&branch_session.messages).unwrap();

        // The active path is now [a, d] — b/c are off-branch, not part of it.
        assert_eq!(store.active_ids().len(), 2);

        // Reopening resolves the tip via the persisted `Leaf` marker to the same branch, not the
        // physically-earlier b/c continuation.
        let (reopened, restored) = repo.open_id(&store.meta().id.clone()).unwrap();
        assert_eq!(restored.messages.len(), 2);
        let dump = serde_json::to_string(restored.messages.as_ref()).unwrap();
        assert!(dump.contains("\"a\"") && dump.contains("\"d\""));
        assert!(!dump.contains("\"b\"") && !dump.contains("\"c\""));
        assert_eq!(reopened.active_ids().len(), 2);
    }

    #[test]
    fn fork_at_entry_works_on_an_off_active_path_branch() {
        // The whole point of `fork_at_entry` over `fork`'s `upto` count: it can target a message that
        // isn't on the *current* active path at all, without first `switch_active`-ing to it.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("a");
        session.user("b");
        session.user("c");
        store.append_new(&session.messages).unwrap();
        let ids = store.active_ids().to_vec();

        // Navigate back to the first message and fork off it with "d" — b/c fall off the active path.
        let branch_root = store.switch_active(&ids[0]).unwrap();
        let mut branch_session = Session::new();
        branch_session.messages = Arc::new(branch_root);
        branch_session.user("d");
        store.append_new(&branch_session.messages).unwrap();
        assert_eq!(store.active_ids().len(), 2); // [a, d]

        let session_id = store.meta().id.clone();

        // Fork "at" b (off-branch) — includes b itself.
        let (forked, fsession) = repo.fork_at_entry(&session_id, &ids[1], false).unwrap();
        assert_eq!(fsession.messages.len(), 2);
        let dump = serde_json::to_string(fsession.messages.as_ref()).unwrap();
        assert!(dump.contains("\"a\"") && dump.contains("\"b\""));
        assert!(!dump.contains("\"c\"") && !dump.contains("\"d\""));
        assert_eq!(forked.meta().parent.as_deref(), Some(session_id.as_str()));

        // Fork "before" c (off-branch) — excludes c itself, same result as forking "at" b.
        let (_, fsession_before) = repo.fork_at_entry(&session_id, &ids[2], true).unwrap();
        assert_eq!(fsession_before.messages.len(), 2);
        let dump = serde_json::to_string(fsession_before.messages.as_ref()).unwrap();
        assert!(dump.contains("\"a\"") && dump.contains("\"b\""));
        assert!(!dump.contains("\"c\""));

        // Fork "at" c (off-branch) — includes c itself, the whole original a->b->c line.
        let (_, fsession_at_c) = repo.fork_at_entry(&session_id, &ids[2], false).unwrap();
        assert_eq!(fsession_at_c.messages.len(), 3);

        // An unknown entry id is rejected, matching `switch_active`'s own NotFound convention.
        match repo.fork_at_entry(&session_id, "does-not-exist", false) {
            Ok(_) => panic!("expected NotFound for an unknown entry id"),
            Err(e) => assert_eq!(e.kind(), std::io::ErrorKind::NotFound),
        }
    }

    #[test]
    fn fork_at_entry_before_a_non_user_message_is_rejected() {
        // Track L27: `before` (now the default — see `serve.rs`'s `fork`/`preview_fork`) means "fork
        // right before this entry", which pi's own `getEntriesToFork` (`repo-utils.ts`) only allows
        // anchored to a user turn — forking "before" an assistant reply is ambiguous (which point,
        // exactly, is "before" a reply that has no message of its own between it and the prior turn?).
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("a");
        session.push(Message::assistant(vec![ContentBlock::text("a-reply")]));
        store.append_new(&session.messages).unwrap();
        let ids = store.active_ids().to_vec(); // [user "a", assistant "a-reply"]

        let session_id = store.meta().id.clone();
        match repo.fork_at_entry(&session_id, &ids[1], true) {
            Ok(_) => {
                panic!("expected invalid_fork_target for a `before` fork at an assistant reply")
            }
            Err(e) => {
                assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput);
                assert!(
                    e.to_string().contains("invalid_fork_target"),
                    "error must be identifiable as pi's own `invalid_fork_target` case: {e}"
                );
            }
        }

        // "at" (before:false) the very same entry is perfectly valid — the restriction is specific to
        // `before`, not to targeting a non-user entry at all.
        let (_, fsession) = repo.fork_at_entry(&session_id, &ids[1], false).unwrap();
        assert_eq!(fsession.messages.len(), 2);
    }

    #[test]
    fn fork_at_entry_rejects_an_anchored_side_channel_entry_as_a_target_and_it_is_unreachable_anyway()
     {
        // Investigated (Round 2 pi-parity remediation, Low severity, left unfixed — see
        // `fork_at_entry_prefix`'s own doc comment for the full assessment). Pi's uniform tree model
        // lets a caller fork/switch at *any* entry id, including a `Label`/`ModelChange`/
        // `ThinkingLevelChange`/`TitleChange` — beyond's narrower one requires `entry_id` to be a real
        // `nodes` entry (`Message`/`BranchSummary`/`Custom`), so one of these 404s. Confirmed here two
        // ways: (a) `fork_at_entry` really does reject such an id, and (b) that id was never something
        // a client could have discovered in the first place — `tree()`, the only surface that lists
        // forkable ids, never lists one of these at all — so no real caller can actually hit case (a).
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("a");
        store.append_new(&session.messages).unwrap();
        let msg_id = store.active_ids()[0].clone();
        store.set_label(&msg_id, Some("checkpoint")).unwrap();

        // The label entry's own id is never returned by any public method — recovered here only by
        // parsing the raw file, which no real client does either (see `tree()`'s own doc comment: it
        // reports the label as the *target* node's `label` field, never as its own addressable entry).
        let raw = fs::read_to_string(&store.path).unwrap();
        let label_entry_id = raw
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str::<Value>(l).unwrap())
            .find(|v| v["type"] == json!("label"))
            .and_then(|v| v["id"].as_str().map(str::to_string))
            .expect("set_label must have appended an Entry::Label line with its own id");

        // (b) unreachable via the client-facing surface: no `TreeNode` ever carries this id.
        assert!(
            store.tree().iter().all(|n| n.id != label_entry_id),
            "a label's own entry id must never be independently listed by `tree()`"
        );

        // (a) and, even if a caller somehow obtained it anyway (as this test does, by reading the raw
        // file), it's still rejected as a fork target.
        let session_id = store.meta().id.clone();
        match repo.fork_at_entry(&session_id, &label_entry_id, false) {
            Ok(_) => panic!("expected NotFound when forking at a label entry's own id"),
            Err(e) => assert_eq!(e.kind(), std::io::ErrorKind::NotFound),
        }
    }

    #[test]
    fn set_label_sets_and_gets() {
        // pi: session-manager/labels.test.ts, "sets and gets labels".
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("hello");
        store.append_new(&session.messages).unwrap();
        let msg_id = store.active_ids()[0].clone();

        assert_eq!(store.get_label(&msg_id), None, "no label initially");
        store.set_label(&msg_id, Some("checkpoint")).unwrap();
        assert_eq!(store.get_label(&msg_id), Some("checkpoint"));

        // The label is visible via `tree()` too, attached to the labeled node.
        let node = store.tree().into_iter().find(|n| n.id == msg_id).unwrap();
        assert_eq!(node.label.as_deref(), Some("checkpoint"));
    }

    #[test]
    fn export_events_records_model_thinking_label_and_custom_changes_in_order() {
        // Track L36 (pi-parity fix): `Entry::ModelChange`/`Entry::ThinkingLevelChange`/`Entry::Label`/
        // `Entry::Custom` were all durably tracked but never reached an HTML export at all —
        // `export_events` is the ordered event log that now lets `crate::export` render them.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("hello");
        store.append_new(&session.messages).unwrap();
        let msg_id = store.active_ids()[0].clone();

        assert!(
            store.export_events().is_empty(),
            "no events recorded yet: {:?}",
            store.export_events()
        );

        store.record_model_change("model-b").unwrap();
        store.record_thinking_level_change("high").unwrap();
        store.set_label(&msg_id, Some("checkpoint")).unwrap();
        let custom_id = store
            .append_custom("beyond:sync", json!({"marker": "m1"}))
            .unwrap();

        assert_eq!(
            store.export_events(),
            &[
                ExportEvent::ModelChange("model-b".to_string()),
                ExportEvent::ThinkingLevelChange("high".to_string()),
                ExportEvent::Label {
                    target_id: msg_id.clone(),
                    label: Some("checkpoint".to_string()),
                },
                ExportEvent::Custom {
                    kind: "beyond:sync".to_string(),
                    data: json!({"marker": "m1"}),
                },
            ],
            "events must be recorded in file order"
        );

        // Reopening from disk must recover the exact same ordered log — not just the live instance's
        // own in-memory bookkeeping.
        let (reopened, _) = repo.open_id(&store.meta().id.clone()).unwrap();
        assert_eq!(reopened.export_events(), store.export_events());
        // `custom_id` really is the id `append_custom` returned, sanity-checking the fixture itself.
        assert!(reopened.tree().iter().any(|n| n.id == custom_id));
    }

    #[test]
    fn set_label_clears_with_none() {
        // pi: session-manager/labels.test.ts, "clears labels with undefined".
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("hello");
        store.append_new(&session.messages).unwrap();
        let msg_id = store.active_ids()[0].clone();

        store.set_label(&msg_id, Some("checkpoint")).unwrap();
        assert_eq!(store.get_label(&msg_id), Some("checkpoint"));
        store.set_label(&msg_id, None).unwrap();
        assert_eq!(store.get_label(&msg_id), None);
    }

    #[test]
    fn set_label_last_write_wins() {
        // pi: session-manager/labels.test.ts, "last label wins".
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("hello");
        store.append_new(&session.messages).unwrap();
        let msg_id = store.active_ids()[0].clone();

        store.set_label(&msg_id, Some("first")).unwrap();
        store.set_label(&msg_id, Some("second")).unwrap();
        store.set_label(&msg_id, Some("third")).unwrap();
        assert_eq!(store.get_label(&msg_id), Some("third"));

        // Survives reopen from disk too, not just in-memory state.
        let (reopened, _) = repo.open_id(&store.meta().id.clone()).unwrap();
        assert_eq!(reopened.get_label(&msg_id), Some("third"));
    }

    #[test]
    fn set_label_rejects_unknown_target_id() {
        // pi: session-manager/labels.test.ts, "throws when labeling non-existent entry".
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let err = store.set_label("non-existent", Some("label")).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }

    #[test]
    fn labels_are_excluded_from_the_active_session_messages() {
        // pi: session-manager/labels.test.ts, "labels are not included in buildSessionContext". In this
        // module labels never occupy a slot in the message chain at all (see `Entry::Label`'s doc
        // comment), so this holds trivially — proven here so a future refactor can't silently regress
        // it by folding labels into `nodes`/`active`.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("hello");
        store.append_new(&session.messages).unwrap();
        let msg_id = store.active_ids()[0].clone();
        store.set_label(&msg_id, Some("checkpoint")).unwrap();

        assert_eq!(store.active_ids().len(), 1);
        let (_, restored) = repo.open_id(&store.meta().id.clone()).unwrap();
        assert_eq!(restored.messages.len(), 1);
    }

    #[test]
    fn labels_are_preserved_across_fork_at_entry_and_dropped_when_off_path() {
        // pi: session-manager/labels.test.ts, "labels are preserved in createBranchedSession" and
        // "labels not on path are not preserved" — combined into one scenario: msg1/msg2 are labeled
        // and both end up on the forked path; msg3 is labeled but forked *before* (excluded), so its
        // label has nothing to carry forward to.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("first");
        session.push(Message::assistant(vec![ContentBlock::text("hi")]));
        session.user("followup");
        store.append_new(&session.messages).unwrap();
        let ids = store.active_ids().to_vec(); // [msg1, msg2, msg3]

        store.set_label(&ids[0], Some("first")).unwrap();
        store.set_label(&ids[1], Some("second")).unwrap();
        store.set_label(&ids[2], Some("third")).unwrap();

        let session_id = store.meta().id.clone();
        // Fork "before" msg3 — keeps msg1/msg2, excludes msg3.
        let (forked, fsession) = repo.fork_at_entry(&session_id, &ids[2], true).unwrap();
        assert_eq!(fsession.messages.len(), 2);
        let new_ids = forked.active_ids().to_vec();
        assert_eq!(new_ids.len(), 2);

        assert_eq!(
            forked.get_label(&new_ids[0]),
            Some("first"),
            "msg1's label must carry forward under msg1's new id in the forked session"
        );
        assert_eq!(
            forked.get_label(&new_ids[1]),
            Some("second"),
            "msg2's label must carry forward under msg2's new id in the forked session"
        );

        // msg3's label had nothing on the forked path to attach to — it must not silently reappear
        // under some other id, and the forked session must not carry any *extra* stray label.
        assert!(
            !forked
                .tree()
                .iter()
                .any(|n| n.label.as_deref() == Some("third")),
            "a label whose target fell off the forked path must not be preserved"
        );

        // The original session is completely unaffected by the fork.
        assert_eq!(store.get_label(&ids[0]), Some("first"));
        assert_eq!(store.get_label(&ids[1]), Some("second"));
        assert_eq!(store.get_label(&ids[2]), Some("third"));

        // Reopening the forked session from disk must agree (labels actually persisted, not just held
        // in memory on the freshly-created store).
        let (reforked, _) = repo.open_id(&forked.meta().id.clone()).unwrap();
        assert_eq!(reforked.get_label(&new_ids[0]), Some("first"));
        assert_eq!(reforked.get_label(&new_ids[1]), Some("second"));
    }

    #[test]
    fn forking_a_session_with_labels_produces_an_unbroken_parent_chain() {
        // pi: session-manager/labels.test.ts, "rewires children of removed labels when forking" — pi
        // needs an explicit rewiring step there because a label is a real tree node other entries chain
        // off of, so dropping one during a fork would otherwise orphan its children. This module's
        // labels never occupy a chain slot to begin with (see `Entry::Label`'s doc comment), so there is
        // no rewiring problem to solve; this proves the structural guarantee that replaces it: after
        // forking a session with labels interspersed, the new session's parent chain is a single
        // unbroken line from root to tip with no gaps.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("first");
        store.append_new(&session.messages).unwrap();
        let msg1_id = store.active_ids()[0].clone();
        store.set_label(&msg1_id, Some("checkpoint")).unwrap();
        store.record_model_change("model-b").unwrap();
        session.user("followup");
        store.append_new(&session.messages).unwrap();
        let msg2_id = store.active_ids()[1].clone();

        let session_id = store.meta().id.clone();
        let (forked, fsession) = repo.fork_at_entry(&session_id, &msg2_id, false).unwrap();
        assert_eq!(fsession.messages.len(), 2);

        let new_ids = forked.active_ids().to_vec();
        assert_eq!(new_ids.len(), 2);
        assert_eq!(
            forked
                .tree()
                .iter()
                .find(|n| n.id == new_ids[1])
                .unwrap()
                .parent_id
                .as_deref(),
            Some(new_ids[0].as_str()),
            "the second message's parent must be the first message directly, with no gap"
        );
    }

    #[test]
    fn model_and_thinking_level_changes_are_branch_scoped() {
        // A model/thinking-level change recorded on one branch must not leak onto a sibling branch that
        // never passed through the message it was anchored to — the whole point of H6. A change
        // anchored *at* a message describes what applies *after* it (children), not the message itself
        // — switching back to the exact anchor point must NOT see it (that's the point: recovering
        // whatever was true *before* the change, e.g. before a `set_model` call).
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("a");
        session.user("b");
        store.append_new(&session.messages).unwrap();
        let ids = store.active_ids().to_vec(); // [a, b]

        // Recorded at the tip (b) — a "b"-anchored change, applying to whatever comes after b.
        store.record_model_change("model-B").unwrap();
        store.record_thinking_level_change("high").unwrap();
        assert_eq!(
            store.model_at(&ids[1]),
            None,
            "a change anchored AT b must not apply when querying b itself"
        );
        assert_eq!(store.thinking_level_at(&ids[1]), None);

        // Continue past b with "c" — c is b's child, so it must see the change.
        session.user("c");
        store.append_new(&session.messages).unwrap();
        let ids = store.active_ids().to_vec(); // [a, b, c]
        assert_eq!(
            store.model_at(&ids[2]),
            Some("model-B"),
            "b's own change must propagate forward to its child c"
        );
        assert_eq!(store.thinking_level_at(&ids[2]), Some("high"));

        // Branch off *a* (not b) with a new message "d" — this branch never passes through b at all.
        let branch_root = store.switch_active(&ids[0]).unwrap();
        let mut branch_session = Session::new();
        branch_session.messages = Arc::new(branch_root);
        branch_session.user("d");
        store.append_new(&branch_session.messages).unwrap();
        let d_id = store.active_ids()[1].clone();

        assert_eq!(
            store.model_at(&d_id),
            None,
            "the model-B change lives on the a->b->c branch, not the a->d one"
        );
        assert_eq!(store.thinking_level_at(&d_id), None);
        assert_eq!(
            store.model_at(&ids[0]),
            None,
            "a itself has no change recorded strictly before it"
        );

        // Reopening the file must recover the same lookups from disk (not just in-memory state).
        let (reopened, _) = repo.open_id(&store.meta().id.clone()).unwrap();
        assert_eq!(reopened.model_at(&ids[2]), Some("model-B"));
        assert_eq!(reopened.model_at(&d_id), None);
    }

    #[test]
    fn record_model_change_does_not_mutate_metas_own_creation_time_model() {
        // Task #18 (pi-parity investigation): an earlier version of this fix had
        // `record_model_change` also assign `self.meta.model = model`, on the theory that leaving
        // `meta.model` frozen at the session's creation-time model was simply a staleness bug. It
        // regressed `Persistence::model_and_level_at`'s existing fallback contract instead — that
        // lookup relies on `meta.model` staying the session's true *original* model forever, as the
        // "nothing was ever recorded reaching this point" baseline for `switch_branch`/`switch_session`
        // (see `SessionStore::record_model_change`'s own doc comment for the full story, and
        // `tests/serve_session_tree.rs::serve_switch_branch_restores_the_model_active_on_that_branch`,
        // the real end-to-end regression that caught it). This pins the corrected, deliberate
        // non-mutation down so it can't quietly regress back.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "model-a")).unwrap();
        assert_eq!(store.meta().model, "model-a");

        store.record_model_change("model-b").unwrap();
        assert_eq!(
            store.meta().model,
            "model-a",
            "meta.model must stay the session's true creation-time value, even after a set_model"
        );

        // Must hold across a reopen too, not just in the same in-memory `self.meta`.
        let (reopened, _) = repo.open_id(&store.meta().id.clone()).unwrap();
        assert_eq!(reopened.meta().model, "model-a");
        // The per-branch lookup map — the mechanism that actually IS supposed to track this — sees the
        // change instead (anchored at the tree's own root here, since no message was ever pushed).
        assert_eq!(store.model_at_root(), Some("model-b"));
    }

    #[test]
    fn record_thinking_level_change_does_not_mutate_metas_thinking_level_field() {
        // Same reasoning as `record_model_change_does_not_mutate_metas_own_creation_time_model` — the
        // new `SessionMeta::thinking_level` field (Task #18) exists (available for a future consumer,
        // e.g. `run --continue`'s reopen path) but is deliberately never auto-populated by this method,
        // for the identical "would break the tree-fallback contract" reason `model` has.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "model-a")).unwrap();
        assert_eq!(store.meta().thinking_level, None);

        store.record_thinking_level_change("high").unwrap();
        assert_eq!(
            store.meta().thinking_level,
            None,
            "meta.thinking_level is not auto-populated by record_thinking_level_change"
        );
    }

    #[test]
    fn fork_resolves_the_updated_model_via_the_tree_not_the_sessions_stale_meta() {
        // Task #18's actual, real-world fix location: `SessionRepo::fork`/`fork_at_entry`/
        // `fork_from_path` used to copy `src.meta.model` verbatim into the new session's own header —
        // always the *original* creation-time model, regardless of any `set_model` since. Fixed by
        // resolving the model active at the fork point via the same per-branch `model_at` lookup
        // `Persistence::model_and_level_at` (`serve.rs`) already uses for `switch_branch`/
        // `switch_session`, rather than by mutating `meta.model` itself (which would have broken that
        // other lookup's own fallback — see `record_model_change`'s doc comment).
        //
        // "b" is a *child* of the model change's anchor ("a"), so it's the point that actually observes
        // the new model — forking at the anchor itself ("a") would still correctly see the *original*
        // model, matching the same "anchored-at, not before" contract this whole file already tests
        // elsewhere (`model_and_thinking_level_changes_are_branch_scoped`).
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "model-a")).unwrap();
        let mut session = Session::new();
        session.user("a");
        store.append_new(&session.messages).unwrap();
        store.record_model_change("model-b").unwrap(); // anchored at "a"
        session.user("b"); // "a"'s child — observes the model-b change
        store.append_new(&session.messages).unwrap();

        let session_id = store.meta().id.clone();
        let (forked, _) = repo.fork(&session_id, usize::MAX).unwrap();
        assert_eq!(
            forked.meta().model,
            "model-b",
            "a fork's own header must carry the model active at the fork point, not the session's \
             original one"
        );
    }

    #[test]
    fn fork_does_not_inherit_a_rename_that_happened_after_the_fork_point() {
        // Pass 15 pi-parity fix, the title analogue of the Task #18 fix just above:
        // `SessionRepo::fork`/`fork_at_entry`/`fork_from_path` used to copy `src.meta.title` verbatim
        // into the new session's own header — always the whole-file *latest* rename, regardless of how
        // much later it happened relative to the fork point (`Entry::TitleChange` is whole-session-
        // scoped by design for a *live* session's own displayed title — see that variant's doc comment —
        // but a fork must resolve path-scoped instead, the same way it already does for
        // `model`/`thinking_level`).
        //
        // Exact repro from the audit: two messages, then a rename (doesn't move the tip), then fork
        // from message 1 — the new session must NOT carry "Renamed" forward, since that name was chosen
        // for a completely different, later conversation this branch never actually had.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("first");
        store.append_new(&session.messages).unwrap();
        session.user("second");
        store.append_new(&session.messages).unwrap();
        store.set_title("Renamed").unwrap();
        assert_eq!(store.meta().title.as_deref(), Some("Renamed"));

        let session_id = store.meta().id.clone();
        let (forked, _) = repo.fork(&session_id, 1).unwrap();
        assert_eq!(
            forked.meta().title,
            None,
            "a fork from before the rename must not inherit a title chosen for a later conversation"
        );
    }

    #[test]
    fn fork_inherits_the_title_in_effect_at_the_fork_point_not_a_later_rename() {
        // The path-scoped resolution isn't just "always omit the title" — a rename that *was* actually
        // in effect at the fork point must still carry forward, exactly like `model`/`thinking_level`.
        //
        // "second" is a *child* of the first rename's anchor ("first"), so it's the point that
        // actually observes "Early Name" — same "anchored-at, not before" contract
        // `fork_resolves_the_updated_model_via_the_tree_not_the_sessions_stale_meta` already exercises
        // for model: forking exactly at "second" (the second rename's own anchor) must NOT yet observe
        // "Renamed Later" either, only whatever was already in effect reaching that point.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("first");
        store.append_new(&session.messages).unwrap();
        store.set_title("Early Name").unwrap(); // anchored at "first"
        session.user("second"); // "first"'s child — observes "Early Name"
        store.append_new(&session.messages).unwrap();
        store.set_title("Renamed Later").unwrap(); // anchored at "second", not yet observed by "second" itself
        session.user("third");
        store.append_new(&session.messages).unwrap();

        let session_id = store.meta().id.clone();
        let (forked, _) = repo.fork(&session_id, 2).unwrap();
        assert_eq!(
            forked.meta().title.as_deref(),
            Some("Early Name"),
            "a fork must inherit whatever rename was actually in effect at the fork point, not a \
             later one"
        );
    }

    #[test]
    fn fork_at_entry_resolves_title_path_scoped_not_whole_file_latest() {
        // `fork_at_entry_prefix` computes its own path (possibly off the active branch entirely, and
        // via a different resolution than `fork`'s `upto` count) — a rename recorded after the target
        // entry must not leak into a fork targeting an earlier point on the same chain.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("a");
        store.append_new(&session.messages).unwrap();
        let ids = store.active_ids().to_vec();

        session.user("b");
        store.append_new(&session.messages).unwrap();
        store.set_title("Renamed After B").unwrap(); // whole-file-latest, but only ever in effect after "b"

        let session_id = store.meta().id.clone();
        let (forked, _) = repo.fork_at_entry(&session_id, &ids[0], false).unwrap();
        assert_eq!(
            forked.meta().title,
            None,
            "forking at a point that predates every rename on its own path must not inherit one made \
             later on the same chain"
        );
    }

    #[test]
    fn fork_carries_the_active_thinking_level_forward_and_it_survives_a_reopen() {
        // Fix 1 (pi-parity remediation, Round 2): unlike `model`/`title` just above (each already
        // fixed by `model_at_or_created`/`title_at_or_root`), none of `fork`/`fork_from_path`/
        // `fork_at_entry_prefix` ever resolved or persisted the source's active thinking level at all
        // — a fork silently dropped whatever reasoning-effort depth the source session had actually
        // settled on. The bug only shows up on a genuine reopen (a still-live in-memory store keeps
        // whatever the caller already had in scope) — exactly what a fresh process would do on `run
        // --continue`, a `serve` restart, or `switch_session` back to this session — so this reproduces
        // that: fork, then reopen the forked *file* from scratch via a brand-new `SessionStore::open`,
        // not the same `forked` handle `fork` already returned.
        // "b" is a *child* of the level change's anchor ("a"), so it's the point that actually
        // observes "high" — same "anchored-at, not before" contract
        // `fork_resolves_the_updated_model_via_the_tree_not_the_sessions_stale_meta` already exercises
        // for model: forking exactly at "a" (the change's own anchor) would still correctly see
        // whatever level was active *before* it, not "high" yet.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("a");
        store.append_new(&session.messages).unwrap();
        store.record_thinking_level_change("high").unwrap(); // anchored at "a"
        session.user("b"); // "a"'s child — observes the "high" change
        store.append_new(&session.messages).unwrap();

        let session_id = store.meta().id.clone();
        let (forked, _) = repo.fork(&session_id, usize::MAX).unwrap();
        assert_eq!(
            forked.meta().thinking_level.as_deref(),
            Some("high"),
            "a fork's own header must carry the thinking level active at the fork point"
        );
        let forked_path = forked.path().to_path_buf();
        drop(forked);

        let (reopened, _) = SessionStore::open(forked_path).unwrap();
        let tip = reopened
            .active_ids()
            .last()
            .cloned()
            .expect("forked session has at least one message");
        assert_eq!(
            reopened.thinking_level_at(&tip),
            Some("high"),
            "the active thinking level must survive a fresh reopen of the forked file, not just live \
             in the in-memory store `fork` already returned"
        );
    }

    #[test]
    fn switch_active_rejects_unknown_id() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let err = store.switch_active("does-not-exist").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }

    #[test]
    fn switch_active_to_the_already_active_leaf_is_a_no_op() {
        // pi-parity fix (B-L2): `agent-session-tree-navigation.test.ts`'s "should handle navigation to
        // same position (no-op)" — navigating to wherever the session already is must not append a
        // redundant `Leaf` entry (or any entry at all): the entry count on disk must not grow, and the
        // in-memory active path must be unchanged.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("a");
        session.user("b");
        store.append_new(&session.messages).unwrap();
        let ids = store.active_ids().to_vec();
        let tip = ids.last().cloned().unwrap();

        let before = std::fs::read_to_string(store.path()).unwrap();
        let messages = store.switch_active(&tip).unwrap();

        assert_eq!(
            messages.len(),
            2,
            "the no-op switch must still return the current branch's materialized messages"
        );
        assert_eq!(
            store.active_ids(),
            ids.as_slice(),
            "the active path must be unchanged by a no-op switch"
        );
        let after = std::fs::read_to_string(store.path()).unwrap();
        assert_eq!(
            before, after,
            "navigating to the already-active leaf must not write anything to disk"
        );

        // Reopening must agree: no phantom `Leaf` entry was ever persisted.
        let (reopened, restored) = repo.open_id(&store.meta().id.clone()).unwrap();
        assert_eq!(restored.messages.len(), 2);
        assert_eq!(reopened.active_ids(), ids.as_slice());
    }

    #[test]
    fn rewrite_preserves_off_branch_messages() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("alpha");
        session.user("beta");
        store.append_new(&session.messages).unwrap();
        let ids = store.active_ids().to_vec();

        // Navigate back to `alpha` and fork a second branch off it — `beta` becomes off-branch.
        let root = store.switch_active(&ids[0]).unwrap();
        let mut forked = Session::new();
        forked.messages = Arc::new(root);
        forked.user("gamma");
        store.append_new(&forked.messages).unwrap();

        // Compact the *active* path (alpha, gamma) down to a single summary — this must not touch the
        // off-branch `beta`.
        store.rewrite(&[Message::user("summary")]).unwrap();
        assert_eq!(store.active_ids().len(), 1);

        // `beta`'s content is still physically on disk, even though it's unreachable from the new tip.
        let raw = fs::read_to_string(&store.path).unwrap();
        assert!(
            raw.contains("\"beta\""),
            "off-branch message was deleted:\n{raw}"
        );
        assert!(
            !raw.contains("\"alpha\""),
            "active-path message should have been compacted away"
        );

        // The active session itself only sees the compacted summary.
        let (_reopened, restored) = repo.open_id(&store.meta().id.clone()).unwrap();
        assert_eq!(restored.messages.len(), 1);
        let dump = serde_json::to_string(restored.messages.as_ref()).unwrap();
        assert!(dump.contains("summary"));
    }

    #[test]
    fn open_resolves_tip_via_last_leaf_entry() {
        // Hand-construct a branched file: root -> [second (m2), branched (m3)], with a `Leaf` pointing
        // the active tip at `m3` — proving `open` prefers the leaf's target over "last line in file".
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let meta = SessionMeta::new("/w", "m");
        let id = meta.id.clone();
        let path = repo.path_for(&meta);

        let mut lines = vec![serde_json::to_string(&Entry::Session(meta)).unwrap()];
        lines.push(
            serde_json::to_string(&Entry::Message {
                id: Some("m1".into()),
                parent_id: None,
                timestamp: 0,
                message: Message::user("first"),
            })
            .unwrap(),
        );
        lines.push(
            serde_json::to_string(&Entry::Message {
                id: Some("m2".into()),
                parent_id: Some("m1".into()),
                timestamp: 0,
                message: Message::user("second"),
            })
            .unwrap(),
        );
        lines.push(
            serde_json::to_string(&Entry::Message {
                id: Some("m3".into()),
                parent_id: Some("m1".into()),
                timestamp: 0,
                message: Message::user("branched"),
            })
            .unwrap(),
        );
        lines.push(
            serde_json::to_string(&Entry::Leaf {
                id: "l1".into(),
                parent_id: None,
                target_id: Some("m3".into()),
            })
            .unwrap(),
        );
        fs::write(&path, lines.join("\n") + "\n").unwrap();

        let (store, session) = repo.open_id(&id).unwrap();
        assert_eq!(session.messages.len(), 2);
        let dump = serde_json::to_string(session.messages.as_ref()).unwrap();
        assert!(dump.contains("first") && dump.contains("branched"));
        assert!(!dump.contains("second"));
        assert_eq!(store.active_ids(), &["m1".to_string(), "m3".to_string()]);
    }

    #[test]
    fn tree_and_list_branches_carry_each_node_s_timestamp_and_sort_chronologically() {
        // Pi-parity fix: `TreeNode`/`BranchInfo` now carry a `timestamp` field and order by it (not by
        // id) — hand-construct a file where the alphabetically-earlier leaf ("m2") was actually created
        // *after* the alphabetically-later one ("m3"), so an id-based sort and a timestamp-based sort
        // disagree — proving the fix actually changed the sort key, not just added an unused field.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let meta = SessionMeta::new("/w", "m");
        let id = meta.id.clone();
        let path = repo.path_for(&meta);

        let lines = [
            serde_json::to_string(&Entry::Session(meta)).unwrap(),
            serde_json::to_string(&Entry::Message {
                id: Some("m1".into()),
                parent_id: None,
                timestamp: 0,
                message: Message::user("root"),
            })
            .unwrap(),
            serde_json::to_string(&Entry::Message {
                id: Some("m3".into()),
                parent_id: Some("m1".into()),
                timestamp: 100,
                message: Message::user("earlier-branch"),
            })
            .unwrap(),
            serde_json::to_string(&Entry::Message {
                id: Some("m2".into()),
                parent_id: Some("m1".into()),
                timestamp: 200,
                message: Message::user("later-branch"),
            })
            .unwrap(),
            serde_json::to_string(&Entry::Leaf {
                id: "l1".into(),
                parent_id: None,
                target_id: Some("m2".into()),
            })
            .unwrap(),
        ];
        fs::write(&path, lines.join("\n") + "\n").unwrap();

        let (store, _session) = repo.open_id(&id).unwrap();

        let tree = store.tree();
        let leaves: Vec<&TreeNode> = tree.iter().filter(|n| n.id != "m1").collect();
        assert_eq!(
            leaves.iter().map(|n| n.id.as_str()).collect::<Vec<_>>(),
            vec!["m3", "m2"],
            "must sort by timestamp (m3=100 before m2=200), not by id: {leaves:?}"
        );
        assert_eq!(tree.iter().find(|n| n.id == "m1").unwrap().timestamp, 0);
        assert_eq!(leaves[0].timestamp, 100);
        assert_eq!(leaves[1].timestamp, 200);

        let branches = store.list_branches();
        assert_eq!(
            branches
                .iter()
                .map(|b| b.leaf_id.as_str())
                .collect::<Vec<_>>(),
            vec!["m3", "m2"],
            "must sort by timestamp (m3=100 before m2=200), not by leaf id: {branches:?}"
        );
        assert_eq!(branches[0].timestamp, 100);
        assert_eq!(branches[1].timestamp, 200);
        assert!(!branches[0].is_active);
        assert!(branches[1].is_active, "the leaf entry points the tip at m2");
    }

    #[test]
    fn switch_active_with_summary_persists_counters_and_survives_reopen() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("alpha");
        session.user("beta");
        store.append_new(&session.messages).unwrap();
        let ids = store.active_ids().to_vec();

        // Navigate away from `beta` (the abandoned tip) back to `alpha`, recording a summary of what
        // was abandoned in the same step.
        let abandoned_tip = ids[1].clone();
        let messages = store
            .switch_active_with_summary(
                &ids[0],
                "recap of the abandoned branch",
                &abandoned_tip,
                BranchSummaryDetails {
                    read_files: vec!["a.rs".into()],
                    modified_files: vec![],
                    summarized_messages: 1,
                },
            )
            .unwrap();

        assert_eq!(store.meta().branch_summaries, 1);
        assert_eq!(store.meta().summarized_branch_messages, 1);
        // The new active path is alpha *plus* the summary message — not just alpha, and not the
        // untouched two-message branch either.
        assert_eq!(messages.len(), 2);
        assert_eq!(store.active_ids().len(), 2);

        // Everything survives reopen: the header counters, `beta`'s content (still physically on disk
        // even though it's off the active path), and the summary as the new tip.
        let (reopened, restored) = repo.open_id(&store.meta().id.clone()).unwrap();
        assert_eq!(reopened.meta().branch_summaries, 1);
        assert_eq!(reopened.meta().summarized_branch_messages, 1);
        assert_eq!(restored.messages.len(), 2);
        let raw = fs::read_to_string(&reopened.path).unwrap();
        assert!(
            raw.contains("\"beta\""),
            "off-branch message was lost:\n{raw}"
        );
        assert!(raw.contains("recap of the abandoned branch"));
        assert!(raw.contains(&abandoned_tip));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&reopened.path).unwrap().permissions().mode() & 0o777,
                0o600,
                "switch_active_with_summary's temp-file-then-rename must not loosen permissions"
            );
        }
    }

    #[test]
    fn branch_summary_carries_a_real_timestamp_not_the_zero_placeholder() {
        // Track L45 (pi-parity fix): `Entry::BranchSummary` used to have no `timestamp` field at all,
        // unlike `Entry::Message`/`Entry::Custom` — a materialized branch-summary node always reported
        // `timestamp: 0` in `tree()`'s output regardless of when it was actually created, breaking a
        // client's ability to sort/reconstruct branch order chronologically for that one entry kind.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("alpha");
        session.user("beta");
        store.append_new(&session.messages).unwrap();
        let ids = store.active_ids().to_vec();

        let before = now_secs();
        store
            .switch_active_with_summary(
                &ids[0],
                "recap of the abandoned branch",
                &ids[1],
                BranchSummaryDetails::default(),
            )
            .unwrap();
        let summary_id = store.active_ids().last().unwrap().clone();

        // The live, in-memory instance already reports a real timestamp, not the old `0` placeholder.
        let node = store
            .tree()
            .into_iter()
            .find(|n| n.id == summary_id)
            .unwrap();
        assert!(
            node.timestamp >= before,
            "expected a real timestamp (>= {before}), got {}",
            node.timestamp
        );

        // Survives reopen from disk, read back from the persisted `Entry::BranchSummary` line itself.
        let (reopened, _) = repo.open_id(&store.meta().id.clone()).unwrap();
        let reopened_node = reopened
            .tree()
            .into_iter()
            .find(|n| n.id == summary_id)
            .unwrap();
        assert_eq!(
            reopened_node.timestamp, node.timestamp,
            "a reopened session must recover the same real timestamp, not fall back to 0"
        );
    }

    #[test]
    fn switch_active_with_summary_surfaces_summary_as_first_message_of_new_branch() {
        // Unlike a plain `switch_active`, this must actually redirect the tip: the summary becomes a
        // real message, part of the live transcript the model sees on the next turn — not an inert
        // provenance record sitting off to the side.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("only message");
        store.append_new(&session.messages).unwrap();
        let root_id = store.active_ids()[0].clone();

        let messages = store
            .switch_active_with_summary(
                &root_id,
                "some recap",
                "abandoned-branch-tip",
                BranchSummaryDetails::default(),
            )
            .unwrap();
        assert_eq!(messages.len(), 2);
        let summary_text = match &messages[1].content[0] {
            ContentBlock::Text { text, .. } => text,
            other => panic!("expected a text block, got {other:?}"),
        };
        assert!(summary_text.contains("some recap"));

        let (reopened, restored) = repo.open_id(&store.meta().id.clone()).unwrap();
        assert_eq!(restored.messages.len(), 2);
        assert_ne!(
            reopened.active_ids(),
            &[root_id],
            "the tip must have moved to include the summary message, not stayed at the root"
        );
    }

    #[test]
    fn switch_active_with_summary_attaches_to_a_root_target_and_becomes_the_new_leaf() {
        // B-M6 pi-parity test gap (fixed): the exact tree shape a branch summary produces was never
        // asserted, only that *some* summary got applied. This is pi's "summary attached to root
        // node" scenario (`agent-session-tree-navigation.test.ts:78-106`) — this codebase's uniform
        // contract (no editor-rewind distinction between user/assistant targets, unlike pi) is: the
        // summary always attaches as a *child of the target itself*, regardless of shape, and always
        // becomes the new leaf.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("first message"); // u1 — the root, no parent
        session.push(Message::assistant(vec![ContentBlock::text("a1")]));
        session.user("second message");
        session.push(Message::assistant(vec![ContentBlock::text("a2")]));
        store.append_new(&session.messages).unwrap();
        let ids = store.active_ids().to_vec();
        let u1 = ids[0].clone();
        let a1 = ids[1].clone();

        store
            .switch_active_with_summary(&u1, "root recap", &ids[3], BranchSummaryDetails::default())
            .unwrap();

        // u1 itself is confirmed root (no parent) — the scenario this test is actually exercising.
        let tree = store.tree();
        let u1_node = tree.iter().find(|n| n.id == u1).unwrap();
        assert!(u1_node.parent_id.is_none(), "u1 must be the tree's root");

        let summary_id = store.active_ids().last().unwrap().clone();
        let summary_node = tree.iter().find(|n| n.id == summary_id).unwrap();
        assert_eq!(
            summary_node.parent_id.as_deref(),
            Some(u1.as_str()),
            "the summary must attach as a child of the target itself"
        );

        // u1 now has two children: its original next message (a1) and the new summary.
        let children: Vec<&str> = tree
            .iter()
            .filter(|n| n.parent_id.as_deref() == Some(u1.as_str()))
            .map(|n| n.id.as_str())
            .collect();
        assert_eq!(
            children.len(),
            2,
            "expected two children of u1: {children:?}"
        );
        assert!(children.contains(&a1.as_str()));
        assert!(children.contains(&summary_id.as_str()));

        assert_eq!(
            store.active_ids().last(),
            Some(&summary_id),
            "the summary must become the new leaf"
        );
    }

    #[test]
    fn parent_of_reports_root_a_real_parent_and_unknown_ids_distinctly() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("first");
        session.user("second");
        store.append_new(&session.messages).unwrap();
        let ids = store.active_ids().to_vec();

        assert_eq!(
            store.parent_of(&ids[0]),
            Some(None),
            "the first message's parent is the root — no parent, but a known node"
        );
        assert_eq!(
            store.parent_of(&ids[1]),
            Some(Some(ids[0].clone())),
            "the second message's parent is the first"
        );
        assert_eq!(
            store.parent_of("does-not-exist"),
            None,
            "an unknown id is distinguishable from a real root"
        );
    }

    #[test]
    fn switch_active_to_root_clears_the_active_path_and_persists_across_reopen() {
        // Pi-parity fix: no way existed to navigate back to before the very first message (redo it in
        // place) — pi's own `SessionManager::resetLeaf`.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("first");
        session.user("second");
        store.append_new(&session.messages).unwrap();

        let messages = store.switch_active_to_root().unwrap();
        assert!(messages.is_empty());
        assert!(
            store.active_ids().is_empty(),
            "the active path must be empty at the root"
        );

        // Appending fresh messages now chains off the root (`parent_id: None`), redoing the first
        // message in place rather than continuing the old branch.
        let mut redo = Session::new();
        redo.user("redone first message");
        store.append_new(&redo.messages).unwrap();
        let new_ids = store.active_ids().to_vec();
        assert_eq!(new_ids.len(), 1);
        assert_eq!(store.parent_of(&new_ids[0]), Some(None));

        // Survives a reopen: the `Leaf{target_id: None}` marker resolves back to root, not to
        // whatever message happens to be last in the file.
        let (reopened, session) = repo.open_id(&store.meta().id.clone()).unwrap();
        assert_eq!(session.messages.len(), 1);
        assert_eq!(reopened.active_ids(), &new_ids[..]);
    }

    #[test]
    fn switch_active_to_root_is_a_no_op_when_already_there() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        assert!(store.active_ids().is_empty());

        store.switch_active_to_root().unwrap();
        assert!(store.active_ids().is_empty());

        // No `Leaf` entry appended — the file only ever gained the header line.
        let raw = fs::read_to_string(&store.path).unwrap();
        assert!(!raw.contains("\"leaf\""), "expected no Leaf entry: {raw}");
    }

    #[test]
    fn abandoned_to_root_reports_the_whole_active_path() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        assert!(
            store.abandoned_to_root().is_empty(),
            "nothing to abandon at the root"
        );

        let mut session = Session::new();
        session.user("first");
        session.user("second");
        store.append_new(&session.messages).unwrap();

        let abandoned = store.abandoned_to_root();
        assert_eq!(abandoned.len(), 2, "the whole active path is abandoned");
        assert_eq!(abandoned[0].0, store.active_ids()[0]);
        assert_eq!(abandoned[1].0, store.active_ids()[1]);
    }

    #[test]
    fn switch_active_to_root_with_summary_makes_the_summary_a_new_root_and_the_new_leaf() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("first");
        session.user("second");
        store.append_new(&session.messages).unwrap();
        let old_tip = store.active_ids().last().cloned().unwrap();

        let messages = store
            .switch_active_to_root_with_summary(
                "recap of the whole abandoned conversation",
                &old_tip,
                BranchSummaryDetails::default(),
            )
            .unwrap();
        assert_eq!(messages.len(), 1, "the summary is the sole active message");

        let summary_id = store.active_ids().last().unwrap().clone();
        assert_eq!(
            store.parent_of(&summary_id),
            Some(None),
            "the summary must become a genuine new root, not a child of the old tree"
        );
        assert_eq!(store.active_ids(), std::slice::from_ref(&summary_id));

        // Survives a reopen.
        let (reopened, session) = repo.open_id(&store.meta().id.clone()).unwrap();
        assert_eq!(session.messages.len(), 1);
        assert_eq!(reopened.active_ids(), &[summary_id]);
    }

    #[test]
    fn switch_active_with_summary_attaches_to_a_nested_user_target_and_becomes_the_new_leaf() {
        // B-M6 pi-parity test gap (fixed): pi's "attach summary to correct parent when navigating to
        // nested user message" scenario (`agent-session-tree-navigation.test.ts:108-145`) — there, pi
        // attaches to the target's *parent* (an editor-rewind semantic this codebase has no
        // equivalent of). Here the contract is uniform: the summary attaches as a child of the
        // nested user target itself, alongside its existing child, and becomes the new leaf.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("message one");
        session.push(Message::assistant(vec![ContentBlock::text("a1")]));
        session.user("message two"); // u2 — nested (not root), the target
        session.push(Message::assistant(vec![ContentBlock::text("a2")]));
        session.user("message three");
        session.push(Message::assistant(vec![ContentBlock::text("a3")]));
        store.append_new(&session.messages).unwrap();
        let ids = store.active_ids().to_vec();
        let u2 = ids[2].clone();
        let a1 = ids[1].clone();
        let a2 = ids[3].clone();

        // u2's parent is confirmed a1 — the "nested, not root" shape this test exercises.
        let tree_before = store.tree();
        assert_eq!(
            tree_before
                .iter()
                .find(|n| n.id == u2)
                .unwrap()
                .parent_id
                .as_deref(),
            Some(a1.as_str())
        );

        store
            .switch_active_with_summary(
                &u2,
                "nested recap",
                &ids[5],
                BranchSummaryDetails::default(),
            )
            .unwrap();

        let tree = store.tree();
        let summary_id = store.active_ids().last().unwrap().clone();
        let summary_node = tree.iter().find(|n| n.id == summary_id).unwrap();
        assert_eq!(
            summary_node.parent_id.as_deref(),
            Some(u2.as_str()),
            "the summary must attach as a child of u2 itself, not u2's parent"
        );

        // u2 now has two children: its original next message (a2) and the new summary.
        let children: Vec<&str> = tree
            .iter()
            .filter(|n| n.parent_id.as_deref() == Some(u2.as_str()))
            .map(|n| n.id.as_str())
            .collect();
        assert_eq!(
            children.len(),
            2,
            "expected two children of u2: {children:?}"
        );
        assert!(children.contains(&a2.as_str()));
        assert!(children.contains(&summary_id.as_str()));

        assert_eq!(store.active_ids().last(), Some(&summary_id));
    }

    #[test]
    fn switch_active_with_summary_attaches_to_an_assistant_target_and_becomes_the_new_leaf() {
        // B-M6 pi-parity test gap (fixed): pi's "attach summary to selected node when navigating to
        // assistant message" scenario (`agent-session-tree-navigation.test.ts:147-173`) — pi attaches
        // to the assistant target itself (no rewind semantic for a non-user target), matching this
        // codebase's uniform contract exactly.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("hello");
        session.push(Message::assistant(vec![ContentBlock::text("a1")])); // the target
        session.user("goodbye");
        session.push(Message::assistant(vec![ContentBlock::text("a2")]));
        store.append_new(&session.messages).unwrap();
        let ids = store.active_ids().to_vec();
        let a1 = ids[1].clone();
        let u2 = ids[2].clone();

        store
            .switch_active_with_summary(
                &a1,
                "assistant recap",
                &ids[3],
                BranchSummaryDetails::default(),
            )
            .unwrap();

        let tree = store.tree();
        let summary_id = store.active_ids().last().unwrap().clone();
        let summary_node = tree.iter().find(|n| n.id == summary_id).unwrap();
        assert_eq!(
            summary_node.parent_id.as_deref(),
            Some(a1.as_str()),
            "the summary must attach as a child of the selected assistant node"
        );

        // a1 now has two children: its original next message (u2) and the new summary.
        let children: Vec<&str> = tree
            .iter()
            .filter(|n| n.parent_id.as_deref() == Some(a1.as_str()))
            .map(|n| n.id.as_str())
            .collect();
        assert_eq!(
            children.len(),
            2,
            "expected two children of a1: {children:?}"
        );
        assert!(children.contains(&u2.as_str()));
        assert!(children.contains(&summary_id.as_str()));

        assert_eq!(
            store.active_ids().last(),
            Some(&summary_id),
            "the summary must become the new leaf"
        );
    }

    #[test]
    fn branch_summary_details_within_folds_forward_a_nested_summary() {
        // A detour off a detour: root -> (branch summary S1, carrying file-tracking) -> one more
        // message -> abandon back to root. The abandoned range's *messages* only expose S1's prose
        // recap (extract_file_ops would find nothing new in it, since it's plain text, not a tool
        // call) — `branch_summary_details_within` must still recover S1's original read/modified files
        // from the index, not just whatever the plain message scan finds.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("root");
        store.append_new(&session.messages).unwrap();
        let root_id = store.active_ids()[0].clone();

        // Record S1 at the root, carrying file-tracking from whatever it summarized.
        store
            .switch_active_with_summary(
                &root_id,
                "recap of the first detour",
                "some-abandoned-tip",
                BranchSummaryDetails {
                    read_files: vec!["src/a.rs".to_string()],
                    modified_files: vec!["src/b.rs".to_string()],
                    summarized_messages: 4,
                },
            )
            .unwrap();
        let s1_id = store.active_ids().last().unwrap().clone();

        // Continue past S1 with one more ordinary message, then abandon everything back to root.
        let mut continued = Session::new();
        continued.messages = Arc::new(
            store
                .switch_active(&s1_id)
                .unwrap_or_else(|_| panic!("s1 must be a valid target")),
        );
        continued.user("one more message after the detour");
        store.append_new(&continued.messages).unwrap();

        let abandoned = store.abandoned_by_switch(&root_id);
        assert_eq!(abandoned.len(), 2, "S1 plus the one message after it");
        let ids: Vec<String> = abandoned.iter().map(|(id, _)| id.clone()).collect();
        assert!(ids.contains(&s1_id));

        let folded = store.branch_summary_details_within(&ids);
        assert_eq!(folded.read_files, vec!["src/a.rs".to_string()]);
        assert_eq!(folded.modified_files, vec!["src/b.rs".to_string()]);

        // Survives a reopen — the index is rebuilt from the persisted `Entry::BranchSummary`, not just
        // held in this process's live memory.
        let (reopened, _) = repo.open_id(&store.meta().id.clone()).unwrap();
        let refolded = reopened.branch_summary_details_within(&ids);
        assert_eq!(refolded.read_files, vec!["src/a.rs".to_string()]);
        assert_eq!(refolded.modified_files, vec!["src/b.rs".to_string()]);
    }

    #[test]
    fn switching_back_twice_does_not_duplicate_summary() {
        // A second, independent detour-and-return through the same point produces a second, sibling
        // summary entry (different abandoned content) — never overwriting or deduping against the
        // first one, and a *plain* switch_active to the same target afterward bypasses any summary
        // entirely (it goes straight back to the raw message, not through either recap).
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("root");
        store.append_new(&session.messages).unwrap();
        let root_id = store.active_ids()[0].clone();

        store
            .switch_active_with_summary(&root_id, "first recap", "branch-a", Default::default())
            .unwrap();
        store
            .switch_active_with_summary(&root_id, "second recap", "branch-b", Default::default())
            .unwrap();
        assert_eq!(store.meta().branch_summaries, 2);

        let raw = fs::read_to_string(&store.path).unwrap();
        assert!(raw.contains("first recap"));
        assert!(raw.contains("second recap"));

        // A later plain switch_active to the same root bypasses both summaries.
        let messages = store.switch_active(&root_id).unwrap();
        assert_eq!(messages.len(), 1);
    }

    #[test]
    fn abandoned_by_switch_covers_ancestor_tip_and_sibling_cases() {
        // Build [a, b, c, d] then fork [a, b, e] off `b` — the tree looks like:
        //   a -> b -> c -> d   (the original continuation, still active before any switch below)
        //          \-> e       (a sibling branch, created by switching to b then appending)
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("a");
        session.user("b");
        session.user("c");
        session.user("d");
        store.append_new(&session.messages).unwrap();
        let ids = store.active_ids().to_vec(); // [a, b, c, d]

        // Case 1: target == the current tip (d) — a no-op switch, nothing abandoned.
        assert!(store.abandoned_by_switch(&ids[3]).is_empty());

        // Case 2: target is an ancestor still on the active path (b) — c and d are abandoned.
        let abandoned = store.abandoned_by_switch(&ids[1]);
        assert_eq!(abandoned.len(), 2);
        assert!(matches!(&abandoned[0].1.content[0], ContentBlock::Text{text, ..} if text == "c"));
        assert!(matches!(&abandoned[1].1.content[0], ContentBlock::Text{text, ..} if text == "d"));

        // Case 3: unknown target — nothing abandoned (the switch itself will fail).
        assert!(store.abandoned_by_switch("does-not-exist").is_empty());

        // Now actually switch to b and fork a sibling branch e off it.
        let root = store.switch_active(&ids[1]).unwrap();
        let mut forked = Session::new();
        forked.messages = Arc::new(root);
        forked.user("e");
        store.append_new(&forked.messages).unwrap();

        // Case 4: target is the *other* sibling (d, now off the active path) — switching there
        // abandons only `e` (the active path's own suffix after the common ancestor `b`), not `c`
        // (which lives on the *other* branch, not the one currently active).
        let abandoned = store.abandoned_by_switch(&ids[3]);
        assert_eq!(abandoned.len(), 1);
        assert!(matches!(&abandoned[0].1.content[0], ContentBlock::Text{text, ..} if text == "e"));
    }

    #[test]
    fn orphaned_off_branch_survives_compaction_of_its_ancestor() {
        // The documented-but-previously-unverified edge case (module doc comment's
        // "compaction vs. branching" section): build [a, b, c, d], switch back to `b` and fork `e`
        // off it (leaving [c, d] off-branch, rooted at `b`), then compact the *active* path [a, b, e]
        // down to a single summary. `b` (c's parent) is deleted by the compaction along with `a` and
        // `e` — this must not panic, corrupt the tree, or silently delete `c`/`d`'s *content*; it
        // should leave them reachable as their own orphaned subtree, rooted at the now-missing `b`.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("a");
        session.user("b");
        session.user("c");
        session.user("d");
        store.append_new(&session.messages).unwrap();
        let ids = store.active_ids().to_vec(); // [a, b, c, d]
        let d_id = ids[3].clone();

        let root = store.switch_active(&ids[1]).unwrap(); // back to b
        let mut forked = Session::new();
        forked.messages = Arc::new(root);
        forked.user("e");
        store.append_new(&forked.messages).unwrap(); // active path is now [a, b, e]; c, d are off-branch

        // Compact the active path away entirely — a, b, and e are all deleted; c and d (off-branch,
        // rooted at the now-gone `b`) must survive on disk.
        store.rewrite(&[Message::user("summary")]).unwrap();
        assert_eq!(store.active_ids().len(), 1);

        let raw = fs::read_to_string(&store.path).unwrap();
        assert!(
            raw.contains("\"c\""),
            "orphaned branch content was deleted:\n{raw}"
        );
        assert!(
            raw.contains("\"d\""),
            "orphaned branch content was deleted:\n{raw}"
        );
        assert!(!raw.contains("\"a\"") && !raw.contains("\"e\""));

        // It's correctly reported by `list_branches` — not silently dropped from the listing, and
        // (checked *before* navigating to it) correctly NOT the active branch.
        let branches = store.list_branches();
        let orphan_branch = branches
            .iter()
            .find(|b| b.leaf_id == d_id)
            .expect("the orphaned branch should still be listed");
        assert_eq!(orphan_branch.message_count, 2);
        assert!(!orphan_branch.is_active);

        // The orphaned subtree is still fully navigable from its own tip: no panic, and its content
        // reconstructs correctly even though the path can't walk back past the deleted `b`.
        let orphaned = store.switch_active(&d_id).unwrap();
        assert_eq!(
            orphaned.len(),
            2,
            "expected just [c, d] — the walk stops at the missing `b`"
        );
        assert!(matches!(&orphaned[0].content[0], ContentBlock::Text{text, ..} if text == "c"));
        assert!(matches!(&orphaned[1].content[0], ContentBlock::Text{text, ..} if text == "d"));

        // And a fresh reopen of the file reconstructs the same picture — the orphaning survives a
        // round-trip through disk, not just the in-memory state from the same process. Having
        // switched to it above, it's now the active branch on reopen too.
        let (reopened, _) = repo.open_id(&store.meta().id.clone()).unwrap();
        let reopened_orphan = reopened
            .list_branches()
            .into_iter()
            .find(|b| b.leaf_id == d_id)
            .expect("the orphaned branch survives a reopen");
        assert_eq!(reopened_orphan.message_count, 2);
        assert!(reopened_orphan.is_active);
    }

    #[test]
    fn abandoned_branches_reports_the_full_off_path_chain_and_its_shared_prefix() {
        // Build [a, b, c, d], switch back to `b`, and fork [a, b, e] off it — leaving [c, d] abandoned,
        // rooted at `b`. `abandoned_branches` must report the *whole* [a, b, c, d] chain (not just
        // [c, d]) alongside `shared_prefix_len: 2` (it shares [a, b] with the new active path
        // [a, b, e]), so a caller can choose to skip re-rendering the shared prefix.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("a");
        session.user("b");
        session.user("c");
        session.user("d");
        store.append_new(&session.messages).unwrap();
        let ids = store.active_ids().to_vec(); // [a, b, c, d]

        let root = store.switch_active(&ids[1]).unwrap(); // back to b
        let mut forked = Session::new();
        forked.messages = Arc::new(root);
        forked.user("e");
        store.append_new(&forked.messages).unwrap(); // active path is now [a, b, e]

        // A session with no branches at all reports nothing abandoned.
        let fresh = repo.create(SessionMeta::new("/w2", "m")).unwrap();
        assert!(fresh.abandoned_branches().is_empty());

        let abandoned = store.abandoned_branches();
        assert_eq!(
            abandoned.len(),
            1,
            "exactly one abandoned leaf (d): {abandoned:?}"
        );
        let (shared, messages) = &abandoned[0];
        assert_eq!(
            *shared, 2,
            "shares [a, b] with the new active path [a, b, e]"
        );
        assert_eq!(
            messages.len(),
            4,
            "the whole [a, b, c, d] chain, not just the divergent suffix"
        );
        let texts: Vec<&str> = messages
            .iter()
            .map(|m| match &m.content[0] {
                ContentBlock::Text { text, .. } => text.as_str(),
                other => panic!("expected a text block, got {other:?}"),
            })
            .collect();
        assert_eq!(texts, vec!["a", "b", "c", "d"]);

        // The active leaf itself must never appear as an "abandoned" branch.
        let active_tip = store.active_ids().last().cloned().unwrap();
        assert_eq!(store.list_branches().len(), 2, "d (abandoned) + e (active)");
        assert!(
            store
                .list_branches()
                .iter()
                .any(|b| b.leaf_id == active_tip && b.is_active)
        );
    }

    #[test]
    fn abandoned_branches_of_an_abandoned_branch_collide_on_the_same_shared_prefix() {
        // Task #34 (pi-parity audit, low-priority/unconfirmed edge case) — investigating whether
        // branching off an *already-abandoned* branch (not off the active path) is handled correctly.
        //
        // Build active [m1, m2]. Branch A off m1: [m1, a1, a2]. Branch B off A's own a1 (not off
        // active at all): [m1, a1, b1]. Restore active to [m1, m2] so both A and B are abandoned
        // leaves. `shared` is always computed against the *active* path only (see
        // `abandoned_branches`'s own loop, `path.zip(self.active.iter())`) — since neither A nor B
        // shares anything with active beyond `m1`, both come out with the *same* `shared == 1`, even
        // though B actually shares two messages ([m1, a1]) with A, not just one ([m1]) with active.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("m1");
        session.user("m2");
        store.append_new(&session.messages).unwrap();
        let ids = store.active_ids().to_vec(); // [m1, m2]

        // Branch A off m1: [m1, a1, a2].
        let root = store.switch_active(&ids[0]).unwrap();
        let mut branch_a = Session::new();
        branch_a.messages = Arc::new(root);
        branch_a.user("a1");
        branch_a.user("a2");
        store.append_new(&branch_a.messages).unwrap();
        let a_ids = store.active_ids().to_vec(); // [m1, a1, a2]
        let a1_id = a_ids[1].clone();

        // Branch B off A's own a1 (an already-abandoned branch, not the active path at all): [m1, a1, b1].
        let root = store.switch_active(&a1_id).unwrap();
        let mut branch_b = Session::new();
        branch_b.messages = Arc::new(root);
        branch_b.user("b1");
        store.append_new(&branch_b.messages).unwrap();

        // Restore active to the original [m1, m2] — both A and B are now abandoned leaves.
        store.switch_active(&ids[1]).unwrap();

        let mut abandoned = store.abandoned_branches();
        assert_eq!(
            abandoned.len(),
            2,
            "both A (leaf a2) and B (leaf b1): {abandoned:?}"
        );
        abandoned.sort_by_key(|(_, messages)| messages.len());
        let (shared_b, messages_b) = &abandoned[0]; // [m1, a1, b1] — the shorter chain
        let (shared_a, messages_a) = &abandoned[1]; // [m1, a1, a2]
        assert_eq!(messages_b.len(), 3);
        assert_eq!(messages_a.len(), 3);

        // The bug this test confirms: both report the same `shared` (their common prefix with
        // *active*), even though B's true divergence point is from A's own a1, two messages deep —
        // not from active's m1, one message deep.
        assert_eq!(*shared_a, 1);
        assert_eq!(
            *shared_b, 1,
            "confirms Task #34: B collides with A's own `shared` value despite actually forking off \
             A (sharing [m1, a1], not just [m1])"
        );

        // Concretely: `export.rs`'s `render_branches_diverging_at` renders `branch_messages[shared..]`
        // for *every* branch at a given `shared` value as sibling `<details>` blocks — so with both at
        // `shared == 1`, B's own box would render `[a1, b1]` (duplicating `a1`, already shown inside
        // A's own separate box as part of `[a1, a2]`), rather than being nested inside A's box showing
        // only its own net-new `[b1]`. Confirmed real; fixing it properly means reshaping
        // `abandoned_branches`'s flat `(shared, messages)` list into an actual tree (and
        // `render_branches_diverging_at` into a recursive renderer) — a disproportionate restructuring
        // for a narrow edge case this crate's RPC surface allows but no real workflow is known to
        // exercise, per this task's own low-priority/unconfirmed framing. Left as documented,
        // proven-real, deliberately unfixed.
        assert_eq!(&messages_b[..*shared_b], &messages_a[..*shared_a]);
    }

    #[test]
    fn torn_leaf_write_recovers_to_pre_switch_state() {
        // A crash mid-write during `switch_active`'s append (the `Leaf` marker) must not corrupt the
        // session — the same "only the last entry can be lost" crash-recovery contract
        // `torn_last_line_is_recovered` proves for `append_new` must hold for this append path too.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let id = store.meta().id.clone();
        let mut session = Session::new();
        session.user("a");
        session.user("b");
        session.user("c");
        store.append_new(&session.messages).unwrap();
        let ids = store.active_ids().to_vec();

        // Simulate a crash partway through appending the `Leaf` marker for a `switch_active(&ids[0])`
        // that never completes: a half-written, unterminated JSON line.
        let path = repo.find_path(&id).unwrap().unwrap();
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        write!(f, "{{\"type\":\"leaf\",\"id\":\"x").unwrap();
        drop(f);

        let (reopened, restored) = repo.open_id(&id).unwrap();
        // The torn leaf marker is dropped; the session reopens at its last *complete* state — the
        // full linear chain, tip still `c` — not corrupted, and not silently switched to whatever
        // target the half-recorded marker would have pointed at.
        assert_eq!(restored.messages.len(), 3);
        assert_eq!(reopened.active_ids(), ids.as_slice());
    }

    #[test]
    fn crash_before_rename_leaves_the_original_file_untouched() {
        // `rewrite` (and `rewrite_compacted`'s degenerate fallback to it) writes a complete new file to
        // `.jsonl.tmp`, then atomically renames it over the original — genuinely replacing the active
        // path, so it needs a full atomic swap. (`rewrite_compacted`'s and
        // `switch_active_with_summary`'s *normal* paths are append-only instead — see their own doc
        // comments — since neither one ever needs to replace anything already on disk.) A crash
        // *before* rename's atomic swap — simulated here by leaving a half-written `.tmp` file behind,
        // never renamed — must leave the original completely untouched: the entire point of temp+rename.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let id = store.meta().id.clone();
        let mut session = Session::new();
        session.user("original content");
        store.append_new(&session.messages).unwrap();

        let path = repo.find_path(&id).unwrap().unwrap();
        let original_contents = fs::read_to_string(&path).unwrap();

        // Simulate the crash: a half-written temp file, never renamed over the original.
        let tmp = path.with_extension("jsonl.tmp");
        fs::write(&tmp, "{\"type\":\"session\",\"id\":\"corrupt").unwrap();

        // The original file is untouched — the stray tmp file is simply ignored (`list`/`open_id`
        // only ever look at `.jsonl`, not `.jsonl.tmp`).
        let (_reopened, restored) = repo.open_id(&id).unwrap();
        assert_eq!(restored.messages.len(), 1);
        let dump = serde_json::to_string(restored.messages.as_ref()).unwrap();
        assert!(dump.contains("original content"));
        assert_eq!(fs::read_to_string(&path).unwrap(), original_contents);
        assert_eq!(
            repo.list().unwrap().len(),
            1,
            "the stray .tmp file must not be picked up as a second session"
        );
    }

    #[test]
    fn cyclic_parent_chain_terminates_instead_of_hanging() {
        // A hand-edited or corrupted file could carry a cycle in `parent_id` (no in-process mutation
        // can produce one — every real parent chains to a strictly earlier id). Without a visited-set
        // guard, `path_from_root` would loop forever growing its output unboundedly instead of
        // degrading to "nothing" the way every other malformed-input case in this module does. Two
        // messages pointing at each other as their own parent is the minimal cycle.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let meta = SessionMeta::new("/w", "m");
        let id = meta.id.clone();
        let path = repo.path_for(&meta);
        let mut lines = vec![serde_json::to_string(&Entry::Session(meta)).unwrap()];
        lines.push(
            serde_json::to_string(&Entry::Message {
                id: Some("m1".into()),
                parent_id: Some("m2".into()),
                timestamp: 0,
                message: Message::user("first"),
            })
            .unwrap(),
        );
        lines.push(
            serde_json::to_string(&Entry::Message {
                id: Some("m2".into()),
                parent_id: Some("m1".into()),
                timestamp: 0,
                message: Message::user("second"),
            })
            .unwrap(),
        );
        fs::write(&path, lines.join("\n") + "\n").unwrap();

        // Must return promptly (not hang) and produce *some* bounded, non-panicking result rather
        // than an unbounded/cyclic one.
        let (store, session) = repo.open_id(&id).unwrap();
        assert!(
            session.messages.len() <= 2,
            "a 2-node cycle must not produce a longer path: {}",
            session.messages.len()
        );
        assert!(store.active_ids().len() <= 2);

        // list_branches and abandoned_by_switch must also terminate rather than hang, walking the
        // same cyclic structure from different entry points.
        let _ = store.list_branches();
        let _ = store.abandoned_by_switch("m1");
    }

    #[test]
    fn append_custom_participates_in_tree_traversal_but_is_skipped_from_active_messages() {
        // pi: session-manager/save-entry.test.ts, "saves custom entries and includes them in tree
        // traversal" — a custom entry chains into the tree like any other entry (a later message's
        // parent is the custom entry's id, not skipped over it), is reported by full-tree traversal,
        // but contributes nothing to the materialized `Session.messages`/LLM context.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("hello");
        store.append_new(&session.messages).unwrap();
        let msg1_id = store.active_ids()[0].clone();

        let custom_id = store
            .append_custom("my_data", json!({"foo": "bar"}))
            .unwrap();

        session.push(Message::assistant(vec![ContentBlock::text("hi")]));
        store.append_new(&session.messages).unwrap();
        let msg2_id = store.active_ids().last().unwrap().clone();

        // The custom entry sits between msg1 and msg2 in the chain.
        assert_eq!(
            msg2_id,
            store.active_ids()[2],
            "path must be [msg1, custom, msg2]"
        );
        assert_eq!(store.active_ids()[1], custom_id);

        let tree = store.tree();
        let custom_node = tree.iter().find(|n| n.id == custom_id).unwrap();
        assert_eq!(custom_node.parent_id.as_deref(), Some(msg1_id.as_str()));
        assert_eq!(custom_node.role, None, "a custom entry has no message role");
        let msg2_node = tree.iter().find(|n| n.id == msg2_id).unwrap();
        assert_eq!(
            msg2_node.parent_id.as_deref(),
            Some(custom_id.as_str()),
            "the message appended after a custom entry must chain directly off it"
        );

        // buildSessionContext-equivalent: only the two real messages, not the custom entry.
        assert_eq!(store.active_ids().len(), 3);
        let (_, restored) = repo.open_id(&store.meta().id.clone()).unwrap();
        assert_eq!(
            restored.messages.len(),
            2,
            "a custom entry must be skipped when materializing Session.messages"
        );
    }

    #[test]
    fn custom_entries_survive_reopen_and_a_compaction_of_an_unrelated_later_range() {
        // A custom entry appended mid-session must round-trip through a reopen with its content intact,
        // and must not confuse `rewrite_compacted`'s folded-message counting when a *later* compaction
        // folds messages that came after it — the whole point of the message-counting walk (as opposed
        // to a raw positional slice) in `rewrite_compacted`.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("a");
        store.append_new(&session.messages).unwrap();
        let custom_id = store
            .append_custom("preset-state", json!({"name": "plan"}))
            .unwrap();
        session.push(Message::assistant(vec![ContentBlock::text("b")]));
        session.user("c");
        store.append_new(&session.messages).unwrap();

        let (reopened, restored) = repo.open_id(&store.meta().id.clone()).unwrap();
        assert_eq!(restored.messages.len(), 3);
        let custom_node = reopened
            .tree()
            .into_iter()
            .find(|n| n.id == custom_id)
            .unwrap();
        assert_eq!(
            custom_node.preview.as_deref(),
            Some("[custom: preset-state]")
        );

        // Compact away everything: a summary replacing all 3 messages down to 1.
        let mut summary_session = Session::new();
        summary_session.push(Message::user(format!(
            "{}\n\nsummary text",
            agent_core::compaction::SUMMARY_MARKER
        )));
        store
            .rewrite_compacted(
                &summary_session.messages,
                CompactionMeta {
                    tokens_before: 100,
                    provenance: Default::default(),
                },
            )
            .unwrap();

        // The custom entry (off the new active path, part of the folded-away region) must still be
        // readable on disk — compaction preserves folded content, it doesn't delete it (see
        // `rewrite_compacted`'s own doc comment) — and the new active path must be exactly the summary.
        let (_, after) = repo.open_id(&store.meta().id.clone()).unwrap();
        assert_eq!(after.messages.len(), 1);
        let tree_after = store.tree();
        assert!(
            tree_after.iter().any(|n| n.id == custom_id),
            "the custom entry must still be present in the tree after compaction folded past it"
        );
    }

    #[test]
    fn fork_at_entry_drops_custom_entries_but_keeps_the_surrounding_messages() {
        // Track C-M2: a custom entry on the forked path has no message representation to carry into the
        // new session (see `fork_at_entry`'s doc comment) — it must be silently dropped, while the real
        // messages before and after it fork normally with an unbroken parent chain.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("a");
        store.append_new(&session.messages).unwrap();
        store.append_custom("marker", json!({"k": "v"})).unwrap();
        session.push(Message::assistant(vec![ContentBlock::text("b")]));
        store.append_new(&session.messages).unwrap();
        let msg2_id = store.active_ids().last().unwrap().clone();

        let session_id = store.meta().id.clone();
        let (forked, fsession) = repo.fork_at_entry(&session_id, &msg2_id, false).unwrap();
        assert_eq!(
            fsession.messages.len(),
            2,
            "both real messages must survive the fork"
        );
        let dump = serde_json::to_string(fsession.messages.as_ref()).unwrap();
        assert!(dump.contains("\"a\"") && dump.contains("\"b\""));

        let new_ids = forked.active_ids().to_vec();
        assert_eq!(
            new_ids.len(),
            2,
            "the custom entry must not occupy a slot in the forked chain"
        );
        assert_eq!(
            forked
                .tree()
                .iter()
                .find(|n| n.id == new_ids[1])
                .unwrap()
                .parent_id
                .as_deref(),
            Some(new_ids[0].as_str()),
            "the second message's parent must be the first message directly, not the dropped custom entry"
        );
    }

    #[test]
    fn rewrite_compacted_folded_ids_are_correct_when_a_custom_entry_sits_inside_the_folded_range() {
        // Regression guard for the `rewrite_compacted` fix that came with Track C-M2: a plain
        // `self.active[..folded_count]` positional slice (the original implementation) would
        // misalign the moment a custom entry occupies a slot in `self.active` without contributing to
        // `self.persisted` — a custom entry inside the folded region would silently eat one of the
        // slots meant for a real folded message, leaving the true last folded message unlisted in the
        // `Entry::Compaction` provenance record. This drives that exact shape: [msg1, custom, msg2,
        // msg3] compacted down to a 1-message summary (folding all 3 real messages away) must list
        // all 3 real ids (and the custom id, since it structurally sits within the folded region) in
        // `folded_ids` — not silently drop msg3.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("one");
        store.append_new(&session.messages).unwrap();
        let msg1_id = store.active_ids()[0].clone();
        let custom_id = store.append_custom("marker", json!({"k": "v"})).unwrap();
        session.user("two");
        session.user("three");
        store.append_new(&session.messages).unwrap();
        let all_ids = store.active_ids().to_vec();
        assert_eq!(
            all_ids,
            vec![
                msg1_id.clone(),
                custom_id.clone(),
                all_ids[2].clone(),
                all_ids[3].clone()
            ]
        );
        let msg2_id = all_ids[2].clone();
        let msg3_id = all_ids[3].clone();

        let compacted_messages = vec![Message::user(format!(
            "{}\n\nrecap",
            agent_core::compaction::SUMMARY_MARKER
        ))];
        store
            .rewrite_compacted(
                &compacted_messages,
                CompactionMeta {
                    tokens_before: 999,
                    provenance: Default::default(),
                },
            )
            .unwrap();
        assert_eq!(store.active_ids().len(), 1);

        let raw = fs::read_to_string(&store.path).unwrap();
        let lines: Vec<Value> = raw
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        let compaction_entry = lines
            .iter()
            .find(|v| v["type"] == json!("compaction"))
            .expect("exactly one compaction entry");
        let folded_ids: Vec<String> = compaction_entry["folded_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();

        assert!(
            folded_ids.contains(&msg1_id),
            "folded_ids must list the first folded message: {folded_ids:?}"
        );
        assert!(
            folded_ids.contains(&msg2_id),
            "folded_ids must list the second folded message: {folded_ids:?}"
        );
        assert!(
            folded_ids.contains(&msg3_id),
            "folded_ids must list the third (last) folded message — this is exactly what a naive \
             positional slice would drop once a custom entry occupies a slot: {folded_ids:?}"
        );
    }

    #[test]
    fn rewrite_compacted_preserves_the_model_and_thinking_level_active_at_the_old_tip() {
        // pi-parity gap (fixed): `rewrite_compacted`'s new active chain starts fully detached
        // (`parent: None`), so `change_at`'s ancestor walk from any new-chain id could never reach a
        // `model_changes`/`level_changes` entry anchored on the now-folded chain — a model/thinking-
        // level switch made before a compaction became permanently unrecoverable. Fixed by resolving
        // the effective value at the old tip and re-anchoring it onto the new chain's `None` baseline.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let id = store.meta().id.clone();
        let mut session = Session::new();
        session.user("one");
        store.append_new(&session.messages).unwrap();
        store.record_model_change("gpt-5").unwrap();
        store.record_thinking_level_change("high").unwrap();
        session.user("two");
        session.user("three");
        store.append_new(&session.messages).unwrap();

        let old_tip = store.active_ids().last().unwrap().clone();
        assert_eq!(store.model_at(&old_tip), Some("gpt-5"));
        assert_eq!(store.thinking_level_at(&old_tip), Some("high"));

        let compacted_messages = vec![Message::user(format!(
            "{}\n\nrecap",
            agent_core::compaction::SUMMARY_MARKER
        ))];
        store
            .rewrite_compacted(
                &compacted_messages,
                CompactionMeta {
                    tokens_before: 999,
                    provenance: Default::default(),
                },
            )
            .unwrap();

        let new_tip = store.active_ids().last().unwrap().clone();
        assert_eq!(
            store.model_at(&new_tip),
            Some("gpt-5"),
            "the model switch must survive the compaction, not vanish"
        );
        assert_eq!(
            store.thinking_level_at(&new_tip),
            Some("high"),
            "the thinking-level switch must survive the compaction, not vanish"
        );

        // Must survive a reopen too — persisted to disk, not just patched into the live in-memory maps.
        drop(store);
        let (reopened, _restored) = repo.open_id(&id).unwrap();
        let reopened_tip = reopened.active_ids().last().unwrap().clone();
        assert_eq!(reopened.model_at(&reopened_tip), Some("gpt-5"));
        assert_eq!(reopened.thinking_level_at(&reopened_tip), Some("high"));
    }

    #[test]
    fn rewrite_compacted_preserves_a_title_resolvable_by_a_later_fork() {
        // Same "detached new chain" gap as
        // `rewrite_compacted_preserves_the_model_and_thinking_level_active_at_the_old_tip`, now also
        // closed for `title_changes` (pass 15 pi-parity fix): a rename set before a compaction must
        // still be resolvable by a fork landing after it — `title_at_or_root` has no public accessor of
        // its own, so this drives it through its only real consumer, `fork`.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let id = store.meta().id.clone();
        let mut session = Session::new();
        session.user("one");
        store.append_new(&session.messages).unwrap();
        store.set_title("Pre-Compaction Name").unwrap();
        session.user("two");
        session.user("three");
        store.append_new(&session.messages).unwrap();

        let compacted_messages = vec![Message::user(format!(
            "{}\n\nrecap",
            agent_core::compaction::SUMMARY_MARKER
        ))];
        store
            .rewrite_compacted(
                &compacted_messages,
                CompactionMeta {
                    tokens_before: 999,
                    provenance: Default::default(),
                },
            )
            .unwrap();

        let (forked, _) = repo.fork(&id, usize::MAX).unwrap();
        assert_eq!(
            forked.meta().title.as_deref(),
            Some("Pre-Compaction Name"),
            "a title set before a compaction must remain resolvable by a fork landing after it"
        );
    }

    #[test]
    fn rewrite_compacted_carries_a_custom_entry_forward_when_it_sits_in_the_kept_suffix() {
        // pi-parity gap (fixed): a *partial* compaction only folds the front of `self.active`; a
        // custom entry (Track C-M2) sitting in the surviving *kept* suffix was previously neither
        // recorded as folded provenance nor carried into the new chain (`new_nodes` was built solely
        // from `messages: &[Message]`, which has no representation for a custom entry) — it just
        // silently disappeared. This drives that exact shape: [msg1, msg2, custom, msg3, msg4]
        // partially compacted (fold msg1+msg2 only) must keep the custom entry, in position, on the
        // new active path.
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("one");
        session.user("two");
        store.append_new(&session.messages).unwrap();
        store.append_custom("marker", json!({"k": "v"})).unwrap();
        session.user("three");
        session.user("four");
        store.append_new(&session.messages).unwrap();
        assert_eq!(
            store.active_ids().len(),
            5,
            "msg1, msg2, custom, msg3, msg4"
        );

        // Fold only msg1+msg2 away — msg3/msg4 (and the custom entry between them and the fold
        // boundary) are the kept suffix.
        let compacted_messages = vec![
            Message::user(format!(
                "{}\n\nrecap",
                agent_core::compaction::SUMMARY_MARKER
            )),
            Message::user("three"),
            Message::user("four"),
        ];
        store
            .rewrite_compacted(
                &compacted_messages,
                CompactionMeta {
                    tokens_before: 999,
                    provenance: Default::default(),
                },
            )
            .unwrap();

        let active_ids = store.active_ids().to_vec();
        assert_eq!(
            active_ids.len(),
            4,
            "summary, custom, msg3, msg4 — the custom entry must not vanish: {active_ids:?}"
        );
        let active: std::collections::HashSet<&str> =
            active_ids.iter().map(String::as_str).collect();
        let custom_survivors: Vec<TreeNode> = store
            .tree()
            .into_iter()
            .filter(|n| active.contains(n.id.as_str()) && n.role.is_none())
            .collect();
        assert_eq!(
            custom_survivors.len(),
            1,
            "exactly one custom node must be on the new active path: {active_ids:?}"
        );
        assert_eq!(
            custom_survivors[0].preview.as_deref(),
            Some("[custom: marker]")
        );

        // The materialized message content is unaffected (custom entries contribute nothing to
        // `Session.messages`) — still exactly [summary, "three", "four"].
        let (_repo2, session) = SessionStore::open(store.path.clone()).unwrap();
        assert_eq!(session.messages.len(), 3);
        assert!(
            matches!(&session.messages[1].content[0], ContentBlock::Text { text, .. } if text == "three")
        );
        assert!(
            matches!(&session.messages[2].content[0], ContentBlock::Text { text, .. } if text == "four")
        );
    }
}
