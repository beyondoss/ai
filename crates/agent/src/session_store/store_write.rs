use std::collections::HashSet;
use std::sync::Arc;

use agent_core::compaction::CompactionProvenance;
use agent_core::{ContentBlock, Cuts, Message};

use super::*;

impl SessionStore {
    /// Ids of the active path's messages, root-first — parallel to the `Session.messages` this store
    /// last produced (via `open`, `append_new`, `rewrite`, or `switch_active`). What a caller quotes
    /// back to [`Self::switch_active`] to navigate to a specific point in the history.
    pub fn active_ids(&self) -> &[String] {
        &self.active
    }

    /// The `data` of every custom entry of `kind` on the active path, oldest first — see
    /// [`Self::append_custom`].
    pub fn active_custom(&self, kind: &str) -> Vec<serde_json::Value> {
        self.active_custom_where(kind, |_| true)
    }

    /// The MCP task journal entries of `kind` on the active path that this session's journal auth
    /// accepts (see [`crate::mcp_resume::JournalAuth`]): planted or tampered lines are left out.
    pub fn active_journal(&self, kind: &str) -> Vec<serde_json::Value> {
        self.active_custom_where(kind, |data| self.journal.accepts(kind, data))
    }

    /// Append an MCP task journal entry, sealed for this session (see
    /// [`crate::mcp_resume::JournalAuth`]); otherwise [`append_custom`](Self::append_custom).
    pub fn append_journal(
        &mut self,
        kind: &str,
        mut data: serde_json::Value,
    ) -> std::io::Result<String> {
        // The key on disk is the session's key: another writer (a thread, another process) may
        // have made it since this store opened. With none there (none yet, one under the old name
        // to carry over, or one removed or torn since), make it now. One small read per journal
        // write.
        if !self.log.read_only()
            && let Some(key) = self.log.journal_key_path()
            && let Some(auth) =
                read_journal_key(&key).or_else(|| make_journal_key(&self.log, &self.nodes))
        {
            self.journal = auth;
        }
        self.journal.seal(kind, &mut data);
        self.append_custom(kind, data)
    }

    pub(super) fn active_custom_where(
        &self,
        kind: &str,
        keep: impl Fn(&serde_json::Value) -> bool,
    ) -> Vec<serde_json::Value> {
        self.active
            .iter()
            .filter_map(|id| match self.nodes.get(id).map(|n| n.content.as_ref()) {
                Some(NodeContent::Custom { kind: k, data }) if k == kind && keep(data) => {
                    Some(data.clone())
                }
                _ => None,
            })
            .collect()
    }

    /// The message at tree entry `id`, anywhere in the whole tree (on or off the active path) —
    /// `None` for an unknown id or one naming a non-message node ([`Entry::Custom`]). Used by `fork`'s
    /// own response (`serve.rs`'s `fork_text`) to echo the forked-from message's
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
    /// (a partial short write is rolled back by `Log::append`, and one a crash leaves behind is a torn
    /// trailing line `open` drops and cuts, per the module's crash-recovery contract).
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
            // Serialized through `EntryRef`, which borrows `msg` — the `Node` below owns the one copy
            // this loop legitimately needs (the in-memory tree keeps it), and the serializer no longer
            // takes a second one it would drop on the next line.
            write_line_ref(
                &mut buf,
                &EntryRef::Message {
                    id: Some(&id),
                    parent_id: parent.as_deref(),
                    timestamp,
                    message: msg,
                },
            )?;
            staged.push((
                id.clone(),
                Node {
                    parent_id: parent.clone(),
                    content: Arc::new(NodeContent::Message(msg.clone())),
                    timestamp,
                },
            ));
            parent = Some(id);
        }
        self.log.append(&buf)?;
        for (id, node) in staged {
            self.active.push(id.clone());
            self.nodes.insert(id, node);
        }
        self.persisted = messages.len();
        Ok(())
    }

    /// Persist a session's transcript, whatever happened to it since the last write: a compaction
    /// (`provenance.compactions` ahead of what is on disk) is written as one with
    /// [`Self::rewrite_compacted`] — its `tokens_before` read back off the summary — then any tool
    /// results a masking pass rewrote in place ([`Self::record_edits`]), then whatever is new
    /// ([`Self::append_new`]). Every persist path (`run`'s, `serve`'s checkpoints, end of turn and
    /// manual `compact`) goes through this, so none can append a post-compaction tail onto the
    /// uncompacted history: `append_new` alone is count-keyed and reads a compaction's shrink as
    /// "nothing new", then, once the list outgrows its old length, chains the wrong messages on.
    ///
    /// The generation is a claim, and it is checked: a compaction always puts a fresh summary at
    /// `messages[0]`, so a generation that moved while the first message is still the one on disk is
    /// a session built without this store's provenance (a branch switch that started from defaults),
    /// not a compaction. That count is adopted and the turn appended as usual. Taking it for a
    /// compaction instead wrote a spurious compaction round — new ids for the whole transcript, the
    /// provenance replaced by the session's empty one.
    ///
    /// A transcript cut back in place since the last write (`cuts`, the session's
    /// [`agent_core::Session::cuts`]) is rewound first: the active tip moves back (a `Leaf`) to the
    /// last message the cut left standing, and the append goes on from there. The agent's
    /// silent-truncation recovery pops a cut-off reply the compaction's own checkpoint already wrote,
    /// then pushes the retry — the count is back where it was, so a count-keyed append wrote nothing,
    /// and a reopen got the cut-off reply (or a tool result without its call) instead of the retry.
    /// The cut, not a comparison of contents, is the signal: what is in memory legitimately differs
    /// from disk in ways that must never be written back (a cross-model scrub). A list shorter than
    /// what is on disk with no compaction behind it is a cut too, recorded or not.
    pub fn persist(
        &mut self,
        messages: &[Message],
        provenance: &CompactionProvenance,
        edited: &[usize],
        cuts: &Cuts,
    ) -> std::io::Result<()> {
        let cut = cuts.since(self.cuts_seen);
        self.persist_inner(messages, provenance, edited, cut.map(|(_, floor)| floor))?;
        if let Some((newest, _)) = cut {
            self.cuts_seen = newest;
        }
        Ok(())
    }

    pub(super) fn persist_inner(
        &mut self,
        messages: &[Message],
        provenance: &CompactionProvenance,
        edited: &[usize],
        cut: Option<usize>,
    ) -> std::io::Result<()> {
        if provenance.compactions != self.written.compactions {
            if self.first_differs(messages) {
                let tokens_before = messages
                    .first()
                    .and_then(agent_core::compaction::summary_tokens_before)
                    .unwrap_or(0);
                self.rewrite_compacted(
                    messages,
                    CompactionMeta {
                        tokens_before,
                        provenance: provenance.clone(),
                    },
                )?;
                self.written = provenance.clone();
                return Ok(());
            }
            tracing::warn!(
                session = %self.meta.id,
                session_generation = provenance.compactions,
                stored_generation = self.written.compactions,
                "a session's compaction count disagrees with the store but its history does not; adopting it"
            );
            self.written.compactions = provenance.compactions;
        }
        let shrunk = (messages.len() < self.persisted).then_some(messages.len());
        if let Some(floor) = cut.into_iter().chain(shrunk).min()
            && floor < self.persisted
        {
            self.rewind_to(floor)?;
        }
        self.record_edits(messages, edited)?;
        self.append_new(messages)
    }

    /// Move the active tip back so only the first `keep` messages stay on it — a `Leaf` to the last
    /// active-path entry before message `keep` (custom entries after the last kept message stay with
    /// it), or to the root for `keep == 0`. The messages past it stay on disk as an abandoned branch.
    pub(super) fn rewind_to(&mut self, keep: usize) -> std::io::Result<()> {
        let Some(first_dropped) = self
            .active
            .iter()
            .enumerate()
            .filter(|(_, id)| {
                self.nodes
                    .get(*id)
                    .is_some_and(|n| n.as_message().is_some())
            })
            .nth(keep)
            .map(|(i, _)| i)
        else {
            return Ok(());
        };
        tracing::debug!(session = %self.meta.id, keep, "rewinding past a cut before appending");
        match first_dropped.checked_sub(1) {
            Some(tip) => {
                let tip = self.active[tip].clone();
                self.switch_active(&tip)?;
            }
            None => {
                self.switch_active_to_root()?;
            }
        }
        Ok(())
    }

    /// Whether `messages` starts somewhere other than the stored active path: its first message is
    /// not the first message on disk, or nothing is on disk yet. `messages[0]` is the one message no
    /// in-memory-only rewrite touches (a cross-model scrub only changes assistant turns and
    /// `prior_turn_tokens` from the third message on; a masking pass and an MCP splice only change
    /// tool results, which never open a transcript), so a difference there is a compaction's.
    pub(super) fn first_differs(&self, messages: &[Message]) -> bool {
        let stored = self
            .active
            .iter()
            .find_map(|id| self.nodes.get(id).and_then(Node::as_message));
        match (messages.first(), stored) {
            (Some(m), Some(s)) => m != s,
            (_, None) => true,
            (None, Some(_)) => false,
        }
    }

    /// The compaction provenance this store last wrote or restored, generation included — what a
    /// [`Session`] rebuilt from this store's messages (a branch switch) must carry, so its next
    /// [`persist`](Self::persist) reads as the continuation it is and the provenance survives the
    /// next compaction.
    pub fn compaction_provenance(&self) -> CompactionProvenance {
        self.written.clone()
    }

    /// Write the tool results a masking pass rewrote in place since they were stored — at the message
    /// `indices` in `agent_core::Session::edited` — as [`Entry::MessageEdit`]s.
    /// [`Self::append_new`] is count-keyed and never sees a change that keeps the count.
    ///
    /// Only a tool result's `content` and `images` are taken from `messages`; everything else stays as
    /// stored. The in-memory transcript can differ from disk in ways that must not be persisted: a
    /// cross-model resume scrubs tool-call ids and thinking in memory only
    /// (`Session::scrub_cross_model_state`), so writing the whole in-memory message would pair a
    /// scrubbed result id against the stored, unscrubbed `tool_use` id, and the original model would
    /// reject the reloaded transcript. A message whose block layout no longer matches what is stored
    /// is left alone.
    ///
    /// Idempotent: an index whose stored results already match writes nothing, so a caller passes the
    /// session's list on every persist without tracking what it wrote. An index past what is stored
    /// is skipped too: [`Self::append_new`] writes those as they are. Only message entries count
    /// toward an index (a custom entry on the active path holds no message slot).
    pub fn record_edits(&mut self, messages: &[Message], indices: &[usize]) -> std::io::Result<()> {
        if indices.is_empty() {
            return Ok(());
        }
        let ids: Vec<&String> = self
            .active
            .iter()
            .filter(|id| {
                self.nodes
                    .get(*id)
                    .is_some_and(|n| n.as_message().is_some())
            })
            .collect();
        let mut buf = Vec::new();
        let mut staged = Vec::new();
        for &i in indices {
            let (Some(id), Some(m)) = (ids.get(i), messages.get(i)) else {
                continue;
            };
            let Some(stored) = self.nodes[*id].as_message() else {
                continue;
            };
            if i >= self.persisted || stored.content.len() != m.content.len() {
                continue;
            }
            let mut edited = stored.clone();
            for (old, new) in edited.content.iter_mut().zip(&m.content) {
                if let (
                    ContentBlock::ToolResult {
                        content, images, ..
                    },
                    ContentBlock::ToolResult {
                        content: c,
                        images: im,
                        ..
                    },
                ) = (old, new)
                {
                    *content = c.clone();
                    *images = im.clone();
                }
            }
            if edited == *stored {
                continue;
            }
            write_line(
                &mut buf,
                &Entry::MessageEdit {
                    target_id: (*id).clone(),
                    message: edited.clone(),
                },
            )?;
            staged.push(((*id).clone(), edited));
        }
        if buf.is_empty() {
            return Ok(());
        }
        self.log.append(&buf)?;
        for (id, m) in staged {
            if let Some(node) = self.nodes.get_mut(&id) {
                node.content = Arc::new(NodeContent::Message(m));
            }
        }
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
        // `data` is opaque and caller-supplied, and on the `serve` path the caller is a remote client
        // (`{type:"append_custom", kind, data?}`). It is retained three times over — in the appended
        // `Entry`, in `self.nodes`, and in `self.events` — and nothing downstream ever interprets or
        // shrinks it, so an unbounded blob here is an unbounded write amplified threefold, on a single
        // request. Everything else this store retains is at least proportional to conversation that
        // actually happened; this is the one input a client can make arbitrarily large for free.
        //
        // Rejected outright rather than truncated: the entry is opaque by contract, so a *partial*
        // `data` is not a degraded version of itself — it's corrupt, and this module has no way to know
        // what that would break for whoever wrote it. The caller gets a real error and can decide.
        let size = serde_json::to_vec(&data)
            .map(|v| v.len())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        if size > MAX_CUSTOM_ENTRY_BYTES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "custom entry `data` is {size} bytes, over the {MAX_CUSTOM_ENTRY_BYTES}-byte limit"
                ),
            ));
        }
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
        self.append_entry(&entry)?;
        self.events.push(ExportEvent::Custom {
            kind: kind.clone(),
            data: data.clone(),
        });
        self.nodes.insert(
            id.clone(),
            Node {
                parent_id,
                content: Arc::new(NodeContent::Custom { kind, data }),
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
    /// Every existing caller — `fork`/`fork_from_path`/`fork_at_entry`, all rewriting a store they just
    /// got back from [`SessionRepo::create`], so every id-keyed side index below starts empty anyway —
    /// wants exactly this narrower reset. See [`Self::reset_for_new_session`] for the one caller that
    /// genuinely wants a blank slate.
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
    /// session's `ModelChange`/`ThinkingLevelChange`/`Label`/`Custom` log. This clears all of that —
    /// and keeps no entry of the old conversation's tree (its abandoned branches used to survive into
    /// the new one) and none of its header provenance (title, compaction and branch-summary counters;
    /// the reset itself is not a compaction) — so the store afterward is indistinguishable, other than
    /// its file path/id, from one just returned by [`SessionRepo::create`].
    pub fn reset_for_new_session(&mut self) -> std::io::Result<()> {
        self.rewrite_impl(&[], true)
    }

    pub(super) fn rewrite_impl(
        &mut self,
        messages: &[Message],
        full_reset: bool,
    ) -> std::io::Result<()> {
        if full_reset {
            // A new conversation, not a compacted one: nothing was folded into a summary, and the
            // header's provenance and title described the conversation that is going away.
            self.meta.title = None;
            self.meta.compactions = 0;
            self.meta.dropped_messages = 0;
            self.meta.branch_summaries = 0;
            self.meta.summarized_branch_messages = 0;
        } else {
            // Compaction provenance: a rewrite that ends with *fewer* messages than were on disk
            // dropped a prefix (the compaction case), so record it in the header before writing. A
            // same-length rewrite or a fresh fork (`persisted == 0`) drops nothing and records nothing.
            let dropped = self.persisted.saturating_sub(messages.len());
            if dropped > 0 {
                self.meta.compactions = self.meta.compactions.saturating_add(1);
                self.meta.dropped_messages =
                    self.meta.dropped_messages.saturating_add(dropped as u64);
            }
        }

        // Every entry off the active path — another branch — rides along a narrow rewrite. A full
        // reset keeps none of them: `/new` reusing this file used to carry every abandoned branch of
        // the old conversation into the new one, still on disk and listed in its tree.
        let old_active: HashSet<&str> = self.active.iter().map(String::as_str).collect();
        let preserved: Vec<(String, Node)> = if full_reset {
            Vec::new()
        } else {
            self.nodes
                .iter()
                .filter(|(id, _)| !old_active.contains(id.as_str()))
                .map(|(id, node)| (id.clone(), node.clone()))
                .collect()
        };

        let mut new_nodes: Vec<(String, Node)> = Vec::with_capacity(messages.len());
        let mut new_active = Vec::with_capacity(messages.len());
        let mut parent: Option<String> = None;
        for m in messages {
            let id = new_id();
            new_nodes.push((
                id.clone(),
                Node {
                    parent_id: parent.clone(),
                    content: Arc::new(NodeContent::Message(m.clone())),
                    timestamp: now_secs(),
                },
            ));
            new_active.push(id.clone());
            parent = Some(id);
        }

        // Serialized whole, then handed to the log as one replacement. Buffering first is what lets
        // either layout implement "replace everything" the way it can do so atomically — a temp file
        // and a rename for the single file, a fresh base segment with a trailer for the segmented one
        // — without this function knowing which. A rewrite is already O(transcript) in time; it is now
        // O(transcript) in memory for the duration of the swap too.
        let mut buf = Vec::new();
        let mut f = &mut buf;

        write_line(&mut f, &Entry::Session(self.meta.clone()))?;
        for (id, node) in preserved.iter().chain(new_nodes.iter()) {
            // `preserved` (off-active nodes carried through unchanged) can be either shape — a real
            // message or a custom entry (Track C-M2) that happened to live on some other branch;
            // `new_nodes` (the freshly compacted/forked active path) is always real messages, since
            // `messages: &[Message]` admits nothing else. Round-tripping each back through its own
            // original `Entry` variant, not unconditionally `Entry::Message`, is what keeps a preserved
            // custom entry from being silently corrupted into an empty/default message by a rewrite.
            match &*node.content {
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
        self.log.replace_all(&buf)?;

        self.nodes = preserved.into_iter().collect();
        self.nodes.extend(new_nodes);
        self.active = new_active;
        self.persisted = messages.len();
        if full_reset {
            self.written = CompactionProvenance::default();
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
}
