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
use std::sync::{Arc, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

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

/// Stable identity + metadata for one session, persisted as the file's header line.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMeta {
    /// On-disk format version (see [`VERSION`]) — present so a future migration can branch on it.
    #[serde(default)]
    pub version: u32,
    pub id: String,
    /// Unix seconds at creation. Orders the repo listing.
    pub created_at: u64,
    /// Working directory the session was started in.
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
            updated_at: 0,
            message_count: 0,
            preview: None,
        }
    }
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
        #[serde(flatten)]
        message: Message,
    },
    /// A branch-navigation marker: the active tip moved to `target_id`. `id`/`parent_id` chain leaf
    /// markers the same way message entries chain messages (so the most recent one is unambiguous even
    /// if several land in one file); `target_id` is the payload — the message id now at the tip.
    Leaf {
        id: String,
        parent_id: Option<String>,
        target_id: String,
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
}

/// One node in the in-memory tree index: a message's parent link and its content. Spans the *whole*
/// file (every branch), not just the active path — built once on [`SessionStore::open`]/`create` and
/// kept in sync by every mutating method, so a branch can be materialized without re-reading the file.
#[derive(Clone)]
struct Node {
    parent_id: Option<String>,
    message: Message,
}

/// One node in the session's tree, as reported by [`SessionStore::tree`] — every message (not just the
/// active path, and not just a branch's leaf like [`BranchInfo`]), with its own parent link, role, and
/// a short preview of its own text content.
#[derive(Debug, Clone, Serialize)]
pub struct TreeNode {
    pub id: String,
    /// `None` at the tree's root.
    pub parent_id: Option<String>,
    pub role: Role,
    /// A preview of this message's own text content, or `None` for a pure tool-use/tool-result/
    /// thinking/image turn with no plain-text block.
    pub preview: Option<String>,
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
}

impl SessionStore {
    /// Create a new session file at `path`, writing its header. Errors if the file already exists.
    pub fn create(path: PathBuf, meta: SessionMeta) -> std::io::Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut opts = OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let mut f = opts.open(&path)?;
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
            match serde_json::from_str::<Entry>(line) {
                Ok(Entry::Session(m)) => meta = Some(m),
                Ok(Entry::Message {
                    id,
                    parent_id,
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
                    nodes.insert(id.clone(), Node { parent_id, message });
                    tip = Some(id);
                }
                Ok(Entry::Leaf { target_id, .. }) => tip = Some(target_id),
                // A branch summary *does* become the new tip — it's a child of the branch point being
                // returned to (see `switch_active_with_summary`), materialized into a real message so
                // the recap actually reaches the model on the next turn, not just sitting on disk.
                Ok(Entry::BranchSummary {
                    id,
                    parent_id,
                    summary,
                    details,
                    ..
                }) => {
                    nodes.insert(
                        id.clone(),
                        Node {
                            parent_id,
                            message: branch_summary_message(&summary),
                        },
                    );
                    branch_summary_details.insert(id.clone(), details);
                    tip = Some(id);
                }
                // Purely a provenance record (see `Entry::Compaction`'s doc comment) — the very next
                // entry in the file is the real, live message this compaction produced, so this one
                // must never itself move the tip.
                Ok(Entry::Compaction { .. }) => {}
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
        let messages: Vec<Message> = active.iter().map(|id| nodes[id].message.clone()).collect();
        let persisted = messages.len();
        let mut session = Session::new();
        session.messages = Arc::new(messages);
        Ok((
            Self {
                path,
                meta,
                persisted,
                nodes,
                active,
                branch_summary_details,
            },
            session,
        ))
    }

    /// The session's metadata.
    pub fn meta(&self) -> &SessionMeta {
        &self.meta
    }

    /// Ids of the active path's messages, root-first — parallel to the `Session.messages` this store
    /// last produced (via `open`, `append_new`, `rewrite`, or `switch_active`). What a caller quotes
    /// back to [`Self::switch_active`] to navigate to a specific point in the history.
    pub fn active_ids(&self) -> &[String] {
        &self.active
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
            write_line(
                &mut buf,
                &Entry::Message {
                    id: Some(id.clone()),
                    parent_id: parent.clone(),
                    message: msg.clone(),
                },
            )?;
            staged.push((
                id.clone(),
                Node {
                    parent_id: parent.clone(),
                    message: msg.clone(),
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
    pub fn rewrite(&mut self, messages: &[Message]) -> std::io::Result<()> {
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
                    message: m.clone(),
                },
            ));
            new_active.push(id.clone());
            parent = Some(id);
        }

        let tmp = self.path.with_extension("jsonl.tmp");
        let mut f = create_private(&tmp)?;
        write_line(&mut f, &Entry::Session(self.meta.clone()))?;
        for (id, node) in preserved.iter().chain(new_nodes.iter()) {
            write_line(
                &mut f,
                &Entry::Message {
                    id: Some(id.clone()),
                    parent_id: node.parent_id.clone(),
                    message: node.message.clone(),
                },
            )?;
        }
        // Sync the temp file's contents, then rename (atomic), then fsync the parent directory so the
        // rename itself is durable: without the dir fsync a crash could surface the old file — or, in the
        // window between, neither — even though the new bytes had reached disk.
        f.flush()?;
        f.sync_all()?;
        fs::rename(&tmp, &self.path)?;
        fsync_dir(&self.path)?;

        self.nodes = preserved.into_iter().collect();
        self.nodes.extend(new_nodes);
        self.active = new_active;
        self.persisted = messages.len();
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

        let folded_count = dropped.saturating_add(1).min(self.active.len());
        let folded_ids: Vec<String> = self.active[..folded_count].to_vec();

        let mut new_nodes: Vec<(String, Node)> = Vec::with_capacity(messages.len());
        // The new active path starts a fresh, detached chain (`parent: None`), exactly like a plain
        // `rewrite` — *not* chained onto the last folded message. `path_from_root` walks a tip's whole
        // parent chain to build the live session, so linking back into the folded prefix would just
        // resurrect every folded message into the "active" transcript, defeating the point of
        // compacting them away. The folded prefix stays exactly where it already was on disk — its own
        // self-contained sub-chain, reachable by id and named in `folded_ids` below, structurally off
        // to the side, the same way an abandoned branch already is.
        let mut parent: Option<String> = None;
        for m in messages {
            let id = new_id();
            new_nodes.push((
                id.clone(),
                Node {
                    parent_id: parent.clone(),
                    message: m.clone(),
                },
            ));
            parent = Some(id);
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
        let compaction_entry = Entry::Compaction {
            id: new_id(),
            parent_id: folded_ids.last().cloned(),
            tokens_before: meta.tokens_before,
            folded_ids,
            summary,
        };

        // An updated header snapshot (the new `compactions`/`dropped_messages` counters) first, then the
        // provenance record, then the new active path itself — `open`'s replay takes the *last*
        // `Entry::Session` line as the header (so appending a fresh one updates it in place) and treats
        // `Entry::Compaction` as inert for tip purposes, so "the last message in the file" still
        // resolves to the true tip with no `Leaf` marker needed.
        let mut buf = Vec::new();
        write_line(&mut buf, &Entry::Session(self.meta.clone()))?;
        write_line(&mut buf, &compaction_entry)?;
        for (id, node) in &new_nodes {
            write_line(
                &mut buf,
                &Entry::Message {
                    id: Some(id.clone()),
                    parent_id: node.parent_id.clone(),
                    message: node.message.clone(),
                },
            )?;
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
        self.active[from..]
            .iter()
            .map(|id| (id.clone(), self.nodes[id].message.clone()))
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
    /// active tip. Order is by leaf id, not creation time (ids aren't strictly time-ordered against
    /// each other across branches) — stable and deterministic for a client rendering a list, not
    /// meaningful beyond that.
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
                    .find_map(|mid| first_user_text(&self.nodes[mid].message))
                    .map(preview_of);
                BranchInfo {
                    leaf_id: id.to_string(),
                    is_active: active_tip == Some(id),
                    message_count: path.len(),
                    preview,
                }
            })
            .collect();
        branches.sort_by(|a, b| a.leaf_id.cmp(&b.leaf_id));
        branches
    }

    /// Every node in the session's tree — every message on every branch, not just the active path
    /// [`BranchInfo`]/[`Self::list_branches`] surfaces only the leaves of. The `nodes` map already spans
    /// the whole file, so this is a single pass over it with no new indexing. Order is by id, for
    /// stable/deterministic client rendering.
    pub fn tree(&self) -> Vec<TreeNode> {
        let mut nodes: Vec<TreeNode> = self
            .nodes
            .iter()
            .map(|(id, node)| TreeNode {
                id: id.clone(),
                parent_id: node.parent_id.clone(),
                role: node.message.role,
                preview: message_text_preview(&node.message),
            })
            .collect();
        nodes.sort_by(|a, b| a.id.cmp(&b.id));
        nodes
    }

    /// Switch the active branch to the message `target_id` — anywhere in the tree, on or off the
    /// current active path — persisting a `Leaf` marker so a later `open()` resolves the new tip.
    /// Returns the branch's materialized messages (root through `target_id`, root-first); the caller
    /// installs them as the live `Session.messages`. A later `append_new` against that returned slice
    /// naturally forks off `target_id` — it chains new messages off `self.active.last()`, which this
    /// sets to `target_id`. Errors (`NotFound`) if `target_id` names no known message.
    pub fn switch_active(&mut self, target_id: &str) -> std::io::Result<Vec<Message>> {
        if !self.nodes.contains_key(target_id) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("no message with id {target_id} in this session"),
            ));
        }
        let leaf = Entry::Leaf {
            id: new_id(),
            parent_id: self.active.last().cloned(),
            target_id: target_id.to_string(),
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
            .map(|id| self.nodes[id].message.clone())
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
        let entry = Entry::BranchSummary {
            id: entry_id.clone(),
            parent_id: Some(target_id.to_string()),
            summary: summary.clone(),
            from_id: from_id.into(),
            details,
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
                message: branch_summary_message(&summary),
            },
        );
        self.branch_summary_details
            .insert(entry_id.clone(), details_for_index);
        self.active = path_from_root(&self.nodes, Some(&entry_id));
        let messages: Vec<Message> = self
            .active
            .iter()
            .map(|id| self.nodes[id].message.clone())
            .collect();
        self.persisted = messages.len();
        Ok(messages)
    }

    /// Set (and persist) the session title. The title lives in the header so the repo listing can read
    /// it cheaply, so this rewrites the file (titles change rarely); pass the current messages.
    pub fn set_title(
        &mut self,
        title: impl Into<String>,
        messages: &[Message],
    ) -> std::io::Result<()> {
        self.meta.title = Some(title.into());
        self.rewrite(messages)
    }
}

/// A directory of session files.
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
        let entries = match fs::read_dir(sessions_root) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };
        let mut metas = Vec::new();
        for entry in entries {
            let Ok(entry) = entry else { continue };
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            if let Ok(repo) = SessionRepo::open(&path) {
                if let Ok(mut project_metas) = repo.list() {
                    metas.append(&mut project_metas);
                }
            }
        }
        metas.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(metas)
    }

    fn path_for(&self, meta: &SessionMeta) -> PathBuf {
        self.dir
            .join(format!("{}_{}.jsonl", meta.created_at, meta.id))
    }

    /// Create a new, empty session and return its store.
    pub fn create(&self, meta: SessionMeta) -> std::io::Result<SessionStore> {
        SessionStore::create(self.path_for(&meta), meta)
    }

    /// All sessions' metadata, newest first (by `created_at`; clients can re-sort by `updated_at`). Each
    /// entry carries the derived listing fields (`updated_at`, `message_count`, `preview`). Files that
    /// fail to read, lack a header, or carry an unreadable version are skipped.
    pub fn list(&self) -> std::io::Result<Vec<SessionMeta>> {
        let mut metas = Vec::new();
        for entry in fs::read_dir(&self.dir)? {
            let path = entry?.path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            if let Some(meta) = read_listing(&path) {
                metas.push(meta);
            }
        }
        metas.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(metas)
    }

    /// Open a session by id.
    pub fn open_id(&self, id: &str) -> std::io::Result<(SessionStore, Session)> {
        let path = self.find_path(id).ok_or_else(|| {
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
        let Some(path) = self.find_path(id) else {
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

    /// Fork session `id` at `upto` messages: a new session whose transcript is the first `upto`
    /// messages of the original, linked back via `parent`. `upto` is clamped to the source length, so
    /// `usize::MAX` clones the whole session. Returns the new store and its restored session.
    pub fn fork(&self, id: &str, upto: usize) -> std::io::Result<(SessionStore, Session)> {
        let (src, src_session) = self.open_id(id)?;
        let upto = upto.min(src_session.messages.len());
        let mut meta = SessionMeta::new(src.meta.cwd.clone(), src.meta.model.clone());
        meta.parent = Some(id.to_string());
        meta.title = src.meta.title.clone();

        let mut store = self.create(meta)?;
        let prefix: Vec<Message> = src_session.messages[..upto].to_vec();
        store.rewrite(&prefix)?;
        let mut session = Session::new();
        session.messages = Arc::new(prefix);
        Ok((store, session))
    }

    fn find_path(&self, id: &str) -> Option<PathBuf> {
        fs::read_dir(&self.dir).ok()?.flatten().find_map(|e| {
            let path = e.path();
            let name = path.file_name()?.to_str()?;
            // `<created_at>_<id>.jsonl`
            (name.ends_with(&format!("_{id}.jsonl"))).then_some(path)
        })
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

/// Read a session file's listing metadata: its (version-checked) header with the derived `updated_at` /
/// `message_count` / `preview` fields filled in. One streaming pass — lines are read and parsed
/// individually, never collected — so this stays light even for long transcripts (the header alone gives
/// id/title/etc.; only the count and preview need the scan). Returns `None` for a file that isn't a
/// readable session (no/invalid header, or an unreadable version), matching `list`'s skip semantics.
fn read_listing(path: &Path) -> Option<SessionMeta> {
    let updated_at = mtime_secs(path);
    let file = File::open(path).ok()?;
    let mut reader = BufReader::new(file);
    let mut header = String::new();

    // The header is the first line.
    reader.read_line(&mut header).ok()?;
    let mut meta = match serde_json::from_str::<Entry>(header.trim()).ok()? {
        Entry::Session(m) => migrate(m, path).ok()?,
        Entry::Message { .. } | Entry::Leaf { .. } | Entry::BranchSummary { .. } => return None,
        Entry::Compaction { .. } => return None,
    };

    // A streaming line count, not a tree walk: it counts every `Message` line in the file, which for a
    // branched session (Track L3, once wired) can exceed the *active* path's length by however many
    // off-branch entries exist. A display convenience, not a correctness input — accurate for every
    // session today, since nothing yet writes an off-branch entry.
    let mut message_count = 0usize;
    let mut preview = None;
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
        match serde_json::from_str::<Entry>(line) {
            Ok(Entry::Message { message, .. }) => {
                message_count += 1;
                if preview.is_none() {
                    if let Some(text) = first_user_text(&message) {
                        preview = Some(preview_of(text));
                    }
                }
            }
            // A stray header mid-file (or a branch-navigation/summary/compaction-provenance marker) is
            // ignored.
            Ok(Entry::Session(_))
            | Ok(Entry::Leaf { .. })
            | Ok(Entry::BranchSummary { .. })
            | Ok(Entry::Compaction { .. }) => {}
            // A fully-read line that failed to deserialize — skip just this one and keep scanning,
            // same relaxed recovery as `SessionStore::open` (see its comment): we know this line's
            // exact boundaries, so one bad line (anywhere in the file, not only a torn tail) no longer
            // truncates the count/preview derived from every good line after it.
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "skipping unparseable session entry line");
            }
        }
    }

    meta.updated_at = updated_at;
    meta.message_count = message_count;
    meta.preview = preview;
    Some(meta)
}

/// The first plain-text block of a user message — what a preview shows. Tool-result user turns carry no
/// `Text` block, so they yield `None` and are skipped; assistant turns aren't user input.
fn first_user_text(msg: &Message) -> Option<&str> {
    if msg.role != Role::User {
        return None;
    }
    msg.content.iter().find_map(|b| match b {
        ContentBlock::Text { text } => Some(text.as_str()),
        _ => None,
    })
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
        ContentBlock::Text { text } if !text.trim().is_empty() => Some(preview_of(text)),
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
fn new_id() -> String {
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
        let path = repo.find_path(&id).unwrap();
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
        let path = repo.find_path(&id).unwrap();
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        write!(f, "{{\"type\":\"message\",\"role\":\"user\",\"cont").unwrap();
        drop(f);

        let (_store, restored) = repo.open_id(&id).unwrap();
        // The intact first message survives; the torn record is dropped.
        assert_eq!(restored.messages.len(), 1);
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

        let path = repo.find_path(&id).unwrap();
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
        let path = repo.find_path(&id).unwrap();
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
        let path = repo.find_path(&id).unwrap();
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
        let path = repo.find_path(&store.meta().id.clone()).unwrap();
        let before = fs::read(&path).unwrap();

        let compacted = vec![
            Message::user(format!(
                "{}\n\nrecap",
                agent_core::compaction::SUMMARY_MARKER
            )),
            session.messages[4].clone(),
        ];
        store
            .rewrite_compacted(&compacted, CompactionMeta { tokens_before: 1 })
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
    fn switch_active_with_summary_appends_without_rewriting_existing_bytes() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let mut session = Session::new();
        session.user("only message");
        store.append_new(&session.messages).unwrap();
        let root_id = store.active_ids()[0].clone();
        let path = repo.find_path(&store.meta().id.clone()).unwrap();
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
        let path = repo.find_path(&id).unwrap();

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
        let mut newer_meta = SessionMeta::new("/b", "m");
        newer_meta.created_at = older.meta().created_at + 1;
        repo_b.create(newer_meta).unwrap();

        let all = SessionRepo::list_all(root.path()).unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].cwd, "/b", "the newer session must sort first");
        assert_eq!(all[1].cwd, "/a");
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
        store.set_title("My Session", &session.messages).unwrap();
        let metas = repo.list().unwrap();
        let found = metas.iter().find(|m| m.id == id).unwrap();
        assert_eq!(found.title.as_deref(), Some("My Session"));
        // A pure title rewrite drops no messages, so it leaves no compaction provenance.
        assert_eq!(found.compactions, 0);
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
        let path = repo.find_path(&id).unwrap();

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
        let path = repo.find_path(&id).unwrap();

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
    fn switch_active_rejects_unknown_id() {
        let dir = tmpdir();
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "m")).unwrap();
        let err = store.switch_active("does-not-exist").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
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
                message: Message::user("first"),
            })
            .unwrap(),
        );
        lines.push(
            serde_json::to_string(&Entry::Message {
                id: Some("m2".into()),
                parent_id: Some("m1".into()),
                message: Message::user("second"),
            })
            .unwrap(),
        );
        lines.push(
            serde_json::to_string(&Entry::Message {
                id: Some("m3".into()),
                parent_id: Some("m1".into()),
                message: Message::user("branched"),
            })
            .unwrap(),
        );
        lines.push(
            serde_json::to_string(&Entry::Leaf {
                id: "l1".into(),
                parent_id: None,
                target_id: "m3".into(),
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
            ContentBlock::Text { text } => text,
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
        assert!(matches!(&abandoned[0].1.content[0], ContentBlock::Text{text} if text == "c"));
        assert!(matches!(&abandoned[1].1.content[0], ContentBlock::Text{text} if text == "d"));

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
        assert!(matches!(&abandoned[0].1.content[0], ContentBlock::Text{text} if text == "e"));
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
        assert!(matches!(&orphaned[0].content[0], ContentBlock::Text{text} if text == "c"));
        assert!(matches!(&orphaned[1].content[0], ContentBlock::Text{text} if text == "d"));

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
        let path = repo.find_path(&id).unwrap();
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

        let path = repo.find_path(&id).unwrap();
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
                message: Message::user("first"),
            })
            .unwrap(),
        );
        lines.push(
            serde_json::to_string(&Entry::Message {
                id: Some("m2".into()),
                parent_id: Some("m1".into()),
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
}
