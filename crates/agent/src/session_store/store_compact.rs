use std::sync::Arc;

use agent_core::Message;

use super::*;

impl SessionStore {
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
    /// recover *which* stored messages were folded with no extra parameter: the stored active path up
    /// to where the kept suffix begins. It is found by content, so a store that lagged the session
    /// (kept messages not on disk yet) still writes an append-only round — never a full rewrite,
    /// which would lose every side record.
    pub fn rewrite_compacted(
        &mut self,
        messages: &[Message],
        meta: CompactionMeta,
    ) -> std::io::Result<()> {
        // Which stored messages the round folded: everything on the active path before the stored
        // messages `messages[1..]` (the kept suffix) still begins with. Found by content, not by
        // count, because the store can lag the session — checkpoints that failed (or never ran)
        // before the compaction leave the kept suffix partly, or wholly, unwritten, and the old
        // count-based `persisted - (messages.len() - 1)` then read as "nothing folded" and fell back
        // to a full rewrite that dropped every side record (model changes, labels, titles, earlier
        // compaction records) and re-minted every id. A kept message not on disk yet is simply
        // appended after the ones that are.
        let stored: Vec<(usize, &Message)> = self
            .active
            .iter()
            .enumerate()
            .filter_map(|(i, id)| self.nodes.get(id)?.as_message().map(|m| (i, m)))
            .collect();
        let kept = messages.get(1..).unwrap_or_default();
        let overlap_from = (0..stored.len())
            .find(|&i| {
                let tail = &stored[i..];
                tail.len() <= kept.len() && tail.iter().zip(kept).all(|((_, s), k)| *s == k)
            })
            .unwrap_or(stored.len());
        let folded_messages = overlap_from;
        // The fold boundary on `self.active`: just past the last folded message, so a custom entry
        // (Track C-M2) among the folded messages is folded with them, and one after them rides along
        // with the kept suffix.
        let boundary = folded_messages
            .checked_sub(1)
            .map_or(0, |last| stored[last].0 + 1);
        let folded_ids: Vec<String> = self.active[..boundary].to_vec();
        self.meta.compactions = self.meta.compactions.saturating_add(1);
        // Net shrink, as before: the summary takes the folded messages' place.
        self.meta.dropped_messages = self
            .meta
            .dropped_messages
            .saturating_add(folded_messages.saturating_sub(1) as u64);
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
                    content: Arc::new(NodeContent::Message(summary.clone())),
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
            // The original node stays exactly where it is (that's the provenance guarantee above), so
            // this new node's content is a second, identical copy of it — shared rather than cloned,
            // since a node's content is immutable once inserted. Otherwise every compaction round would
            // permanently double the memory held for every message that survived it. The equality check
            // keeps the sharing an optimization and not a *trust*: `messages[1..]` is the kept suffix
            // verbatim by construction, but if a caller ever broke that, its message still wins — the
            // behavior is identical to cloning unconditionally, only cheaper.
            let original = &self.nodes[id].content;
            let content = match &**original {
                NodeContent::Message(existing) => match rest.next() {
                    Some(m) if m == existing => Arc::clone(original),
                    Some(m) => Arc::new(NodeContent::Message(m.clone())),
                    None => unreachable!(
                        "kept_suffix_ids holds the stored messages messages[1..] begins with"
                    ),
                },
                NodeContent::Custom { .. } => Arc::clone(original),
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
        // Kept messages the store never got (it lagged the session): new nodes, like an append.
        for m in rest {
            let id = new_id();
            new_nodes.push((
                id.clone(),
                Node {
                    parent_id: parent.clone(),
                    content: Arc::new(NodeContent::Message(m.clone())),
                    timestamp: now_secs(),
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
            memory_notes: meta.provenance.memory_notes,
            user_requests: meta.provenance.user_requests,
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
            match &*node.content {
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
        self.log.append(&buf)?;

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
}
