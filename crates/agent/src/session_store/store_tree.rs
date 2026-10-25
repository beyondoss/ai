use std::collections::HashSet;
use std::sync::Arc;

use agent_core::{Message, Role};

use super::*;

impl SessionStore {
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
                let (role, preview, entry_kind) = match &*node.content {
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
            return Ok(materialize(&self.nodes, &self.active, &self.journal));
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
        self.log.append(&buf)?;

        let active = path_from_root(&self.nodes, Some(target_id));
        let messages: Vec<Message> = materialize(&self.nodes, &active, &self.journal);
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
        self.log.append(&buf)?;

        self.nodes.insert(
            entry_id.clone(),
            Node {
                parent_id: Some(target_id.to_string()),
                content: Arc::new(NodeContent::Message(branch_summary_message(&summary))),
                timestamp,
            },
        );
        self.branch_summary_details
            .insert(entry_id.clone(), details_for_index);
        self.active = path_from_root(&self.nodes, Some(&entry_id));
        let messages: Vec<Message> = materialize(&self.nodes, &self.active, &self.journal);
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
        self.log.append(&buf)?;

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
        self.log.append(&buf)?;

        self.nodes.insert(
            entry_id.clone(),
            Node {
                parent_id: None,
                content: Arc::new(NodeContent::Message(branch_summary_message(&summary))),
                timestamp,
            },
        );
        self.branch_summary_details
            .insert(entry_id.clone(), details_for_index);
        self.active = vec![entry_id];
        let messages: Vec<Message> = materialize(&self.nodes, &self.active, &self.journal);
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
        self.append_entry(&entry)?;
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
        self.append_entry(&entry)?;
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
    /// this same tip", exactly the query `Persistence::model_and_level_at_active` (`serve/persistence.rs`) makes
    /// on every reopen. Anchoring at the root instead means it's picked up by `change_at`'s own base
    /// case (`changes.get(&None)`, checked before any ancestor) for every id in the new tree, the same
    /// way `rewrite_compacted` already re-anchors a resolved model/level onto a fresh chain's `None`
    /// baseline.
    pub(super) fn seed_thinking_level_at_root(&mut self, level: &str) -> std::io::Result<()> {
        let entry = Entry::ThinkingLevelChange {
            id: new_id(),
            parent_id: None,
            level: level.to_string(),
        };
        self.append_entry(&entry)?;
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
    /// `Persistence::model_and_level_at` (`serve/persistence.rs`) already applies for `switch_branch`/
    /// `switch_session`, just scoped to the one value a fork's own new header needs up front.
    pub(super) fn model_at_or_created(&self, target_id: Option<&str>) -> &str {
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
    pub(super) fn title_at_or_root(&self, target_id: Option<&str>) -> Option<&str> {
        target_id
            .and_then(|id| change_at(&self.nodes, &self.title_changes, id))
            .or_else(|| self.title_changes.get(&None))
            .and_then(|title| title.as_deref())
    }

    /// The title a rename recorded directly at `anchor` set: `None` when no rename is anchored there,
    /// `Some(None)` when it cleared the title. For a fork copying through the source's tip, where that rename is
    /// part of what was copied — see [`SessionRepo::fork`].
    pub(super) fn title_anchored_at(&self, anchor: Option<&str>) -> Option<Option<&str>> {
        self.title_changes
            .get(&anchor.map(str::to_string))
            .map(Option::as_deref)
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
    pub(super) fn thinking_level_at_or_created(&self, target_id: Option<&str>) -> Option<&str> {
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
    /// Record this session's exec endpoint (or clear it) by appending a fresh header line.
    ///
    /// Header updates are appends, not rewrites — `open` takes the *last* header it sees (see this
    /// module's doc), so this is O(1) and crash-safe: a torn append leaves the previous header intact
    /// rather than corrupting the session.
    pub fn set_exec_endpoint(
        &mut self,
        endpoint: Option<serde_json::Value>,
    ) -> std::io::Result<()> {
        self.meta.exec_endpoint = endpoint;
        self.append_entry(&Entry::Session(self.meta.clone()))
    }

    pub fn set_title(&mut self, title: impl Into<String>) -> std::io::Result<()> {
        let title = sanitize_title(&title.into());
        let anchor = self.active.last().cloned();
        let entry = Entry::TitleChange {
            id: new_id(),
            parent_id: anchor.clone(),
            title: title.clone(),
        };
        self.append_entry(&entry)?;
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
        self.append_entry(&entry)?;
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
    pub(super) fn labels_within(&self, ids: &[String]) -> Vec<(String, String)> {
        ids.iter()
            .filter_map(|id| self.labels.get(id).map(|l| (id.clone(), l.clone())))
            .collect()
    }
}
