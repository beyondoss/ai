use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

use agent_core::compaction::{CompactionProvenance, CompactionReason};
use agent_core::{Message, Role};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::*;

/// One persisted line. Internally tagged on `type`. A `Message` entry flattens the wrapped
/// `agent_core::Message` (`{role, content}`) alongside its tree fields, so the wire shape is
/// `{"type":"message", id, parent_id, role, content}` — `agent_core::Message` itself carries no tree
/// knowledge (see the module doc comment).
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(super) enum Entry {
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
        /// The `memory`-note paths carried across this round (see `CompactionProvenance::memory_notes`),
        /// restored the same way so a restart past a compaction doesn't lose the model's awareness of what
        /// it wrote to its working/durable memory.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        memory_notes: Vec<String>,
        /// The user's own requests carried across this round (see
        /// `CompactionProvenance::user_requests`), restored the same way. Empty on a record written
        /// before requests were carried.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        user_requests: Vec<String>,
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
    /// A message rewritten in place: from here on, the message entry `target_id` reads as `message`
    /// (the last edit in file order wins). Written by [`SessionStore::record_edits`] for tool-output
    /// masking (`agent_core::compaction::mask_tool_results`), which changes old tool results without
    /// changing the message count, so an append can't carry it. Non-destructive and O(edited): the
    /// original stays on disk, and an older binary skips this line as an unknown entry and simply
    /// reloads the unmasked transcript — a valid, just larger, state.
    MessageEdit {
        target_id: String,
        #[serde(flatten)]
        message: Message,
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
    /// `Session.compaction` right after this round's own `merge_provenance` fold — persisted onto
    /// the new `Entry::Compaction` record's `read_files`/`modified_files`/`last_reason` (see that
    /// variant's own doc comment) so a later `SessionStore::open` can restore it instead of
    /// silently starting over from empty.
    pub provenance: CompactionProvenance,
}

/// An [`Entry::Compaction`] record, indexed by its own id — read back out for [`SessionStore::tree`]
/// (Track L25). `Entry::Compaction` is deliberately never a real chain node (see its own doc comment:
/// "this one never should [redirect the tip]"), so it's tracked in this side index instead of
/// `nodes`/`NodeContent`, the same way `branch_summary_details` tracks structured data for an
/// `Entry::BranchSummary` alongside (rather than instead of) that one's real `NodeContent::Message`
/// node.
#[derive(Debug, Clone)]
pub(super) struct CompactionRecord {
    pub(super) parent_id: Option<String>,
    pub(super) tokens_before: u32,
    pub(super) folded_ids: Vec<String>,
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
pub(super) struct ForkPrefix {
    pub(super) meta: SessionMeta,
    pub(super) messages: Vec<Message>,
    /// The source session's own ids for `messages`, in the same order — i.e. before `rewrite` replaces
    /// them with a fresh chain. `rewrite` cannot preserve ids in general (a compaction's summary
    /// message has no 1:1 original to key off of), but a *fork*'s prefix genuinely is an unmodified
    /// copy, assigned fresh ids in the same order it was read — so zipping this against the new store's
    /// `active_ids()` after `rewrite` recovers the old-id → new-id mapping without `rewrite` itself
    /// needing to know or preserve anything.
    pub(super) original_ids: Vec<String>,
    /// Labels recorded in the source session against any id in `original_ids`, as `(target_id, label)`
    /// pairs — see [`SessionStore::labels_within`].
    pub(super) labels: Vec<(String, String)>,
    /// The thinking level actually in effect at the resolved fork point — see
    /// [`SessionStore::thinking_level_at_or_created`]'s own doc comment. `None` when nothing was ever
    /// recorded reaching that point (a fork of a session, or a branch of one, that never touched its
    /// thinking level), in which case [`SessionRepo::fork_at_entry`] writes no entry for it at all.
    pub(super) thinking_level: Option<String>,
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
pub(super) enum NodeContent {
    Message(Message),
    Custom { kind: String, data: Value },
}

/// One node in the in-memory tree index: an entry's parent link and its content. Spans the *whole*
/// file (every branch), not just the active path — built once on [`SessionStore::open`]/`create` and
/// kept in sync by every mutating method, so a branch can be materialized without re-reading the file.
#[derive(Clone)]
pub(super) struct Node {
    pub(super) parent_id: Option<String>,
    /// Behind an `Arc` because a node's content is immutable once inserted (nothing here ever mutates
    /// an existing entry — the file is append-only, and so is this index), and one message's content
    /// can legitimately belong to two nodes at once: [`SessionStore::rewrite_compacted`] deliberately
    /// keeps the pre-compaction chain intact (its provenance guarantee) *and* re-emits the surviving
    /// suffix under fresh ids on the new active chain, so every kept message would otherwise be held
    /// twice in memory — once per compaction round, forever. Sharing makes that duplication cost a
    /// refcount instead of a full [`Message`] clone.
    pub(super) content: Arc<NodeContent>,
    /// Unix seconds this entry was actually appended, from [`Entry::Message`]/[`Entry::Custom`]'s own
    /// `timestamp` field — `0` for a node with no recorded timestamp (a legacy file written before this
    /// field existed, or a branch-summary-materialized node, which doesn't carry one). See
    /// [`read_listing`]'s `updated_at` computation for why this exists: a content-derived signal,
    /// preferred over the file's OS mtime, which a copy/restore/sync that doesn't preserve it exactly
    /// can leave stale or wrong.
    pub(super) timestamp: u64,
}

impl Node {
    /// This node's message content, or `None` for a non-message node (currently only
    /// [`NodeContent::Custom`]) — what every "materialize the active path/a branch into messages" call
    /// site filters through, so a custom entry contributes nothing to `Session.messages`/LLM context
    /// while still being a real, positioned node for tree traversal (`SessionStore::tree`) — matching
    /// pi's own `buildSessionContext` skipping `"custom"`-typed entries. See [`Entry::Custom`]'s doc
    /// comment.
    pub(super) fn as_message(&self) -> Option<&Message> {
        match &*self.content {
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
pub(super) fn branch_summary_message(summary: &str) -> Message {
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
/// The messages on `path`, as the model sees them: every message node, with each resolved MCP task
/// result journaled on the path (a `mcp_task_result` custom entry, see `crate::mcp_resume`) spliced
/// in front of the first user turn after the call it answers. The splice is part of the transcript,
/// not just of the next request, so `get_messages`, the HTML export and `run --continue` agree with
/// what `serve` sent the model. Only journal entries `journal` accepts count; custom entries are
/// otherwise invisible here.
pub(super) fn materialize(
    nodes: &HashMap<String, Node>,
    path: &[String],
    journal: &crate::mcp_resume::JournalAuth,
) -> Vec<Message> {
    use crate::mcp_resume::{PLACEHOLDER_ENTRY_KIND, RESULT_ENTRY_KIND};
    let mut messages = Vec::new();
    let mut results = Vec::new();
    let mut placeholders = Vec::new();
    for id in path {
        match nodes.get(id).map(|n| n.content.as_ref()) {
            Some(NodeContent::Message(m)) => messages.push(m.clone()),
            Some(NodeContent::Custom { kind, data })
                if kind == RESULT_ENTRY_KIND && journal.accepts(kind, data) =>
            {
                if let Ok(result) = serde_json::from_value(data.clone()) {
                    results.push(result);
                }
            }
            Some(NodeContent::Custom { kind, data })
                if kind == PLACEHOLDER_ENTRY_KIND && journal.accepts(kind, data) =>
            {
                placeholders.push(data.clone());
            }
            _ => {}
        }
    }
    if !results.is_empty() {
        let placeholders = crate::mcp_resume::placeholder_ids(&placeholders);
        crate::mcp_resume::splice(&mut messages, &results, &placeholders);
    }
    messages
}

pub(super) fn path_from_root(nodes: &HashMap<String, Node>, tip: Option<&str>) -> Vec<String> {
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

/// Re-chain every node whose `parent_id` names a node the tree never got — its line was torn,
/// unreadable, or of a kind this build does not know — onto the tip in effect when the orphan was read
/// (`tip_at_read`). Appends always chain off the active tip, so that is the orphan's nearest readable
/// ancestor: the gap costs exactly the unreadable entries, never the history before them. Without this
/// [`path_from_root`] stops at the gap, and everything before it falls off the active path.
///
/// Runs after the whole scan, not per line: a full rewrite writes off-path nodes in no particular
/// order, so a parent can legitimately come after its child in the file. Only the in-memory index is
/// re-chained; the file is left as written, and every open derives the same tree from it.
pub(super) fn rechain_orphans(
    nodes: &mut HashMap<String, Node>,
    tip_at_read: &HashMap<String, Option<String>>,
) {
    let orphans: Vec<String> = nodes
        .iter()
        .filter(|(_, node)| {
            node.parent_id
                .as_ref()
                .is_some_and(|p| !nodes.contains_key(p))
        })
        .map(|(id, _)| id.clone())
        .collect();
    for id in orphans {
        let adopted = tip_at_read
            .get(&id)
            .cloned()
            .flatten()
            .filter(|t| *t != id && nodes.contains_key(t));
        tracing::warn!(id = %id, adopted = ?adopted, "re-chaining a session entry whose parent is unreadable");
        if let Some(node) = nodes.get_mut(&id) {
            node.parent_id = adopted;
        }
    }
}

/// Cut a single-file session back to its last complete line — a torn tail left by a crash, an OOM
/// kill or a full disk mid-append. A file with no newline at all (a torn header) is left alone; the
/// open that called this has already refused it.
pub(super) fn trim_torn_tail(path: &Path) -> std::io::Result<()> {
    use std::io::{Seek, SeekFrom};
    let mut f = OpenOptions::new().read(true).write(true).open(path)?;
    let len = f.metadata()?.len();
    let mut end = len;
    let mut chunk = vec![0u8; 64 * 1024];
    while end > 0 {
        let start = end.saturating_sub(chunk.len() as u64);
        let n = usize::try_from(end - start).unwrap_or(chunk.len());
        f.seek(SeekFrom::Start(start))?;
        f.read_exact(&mut chunk[..n])?;
        if let Some(pos) = chunk[..n].iter().rposition(|&b| b == b'\n') {
            let keep = start + pos as u64 + 1;
            if keep < len {
                tracing::warn!(path = %path.display(), cut = len - keep, "cutting a torn session tail");
                f.set_len(keep)?;
                f.sync_all()?;
            }
            return Ok(());
        }
        end = start;
    }
    Ok(())
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
pub(super) fn change_at<'a, V>(
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
