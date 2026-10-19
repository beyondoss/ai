use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use agent_core::compaction::CompactionProvenance;
use agent_core::{Message, Session};

use super::*;

/// A handle to one session's append log.
///
/// Not `Sync`-guarded internally, and deliberately so: every mutating method takes `&mut self`, and
/// every call site (`serve`'s single-threaded control loop) holds its `SessionStore` as a plain owned
/// value, never behind an `Arc`. Rust's borrow checker already makes a concurrent `append_new`/`rewrite`
/// race on one `SessionStore` unrepresentable *within a process* — there is no code path that could
/// call two mutating methods on the same store at once, so an in-process lock would guard against a
/// scenario that can't happen. Two separate **processes** writing one single-file session (`run -c`
/// in two terminals, `run` against a session a `serve` has open) are kept apart by the writer lock
/// [`lock_session_writer`], which every writing binary takes right after opening and holds with the
/// store; the store itself takes no lock, so a read-only consumer (listing, export, a fork source) is
/// never blocked. The segmented layout is fenced by epoch instead (see "Ownership: the epoch fence").
pub struct SessionStore {
    /// Where this session's entries actually live — one file, or a directory of epoch segments. The
    /// tree logic below never looks inside it: every read goes through `log.lines()` and every write
    /// through `log.append`/`log.replace_all`.
    pub(super) log: Log,
    /// How this session's MCP task journal entries are sealed and checked (see
    /// `crate::mcp_resume::JournalAuth`).
    pub(super) journal: crate::mcp_resume::JournalAuth,
    pub(super) meta: SessionMeta,
    /// How many messages are already on disk (on the active path) — the append cursor.
    pub(super) persisted: usize,
    /// `Session::compaction` as of the last write: its `compactions` is the compaction generation on
    /// disk. A session whose generation differs was compacted since (see [`Self::persist`] for how
    /// that is confirmed), and `persist` writes that compaction before anything else — the signal
    /// `persisted` (a message count) can't give, since a compaction's shrink looks to it like nothing
    /// new. Set from the session `open` returns, so a legacy file whose provenance never restored
    /// starts in step with what it reloads; handed back out by [`Self::compaction_provenance`] so a
    /// session rebuilt from this store (a branch switch) starts in step too.
    pub(super) written: CompactionProvenance,
    /// The newest `agent_core::Cuts` id this store has applied (or, at open/create, the process's
    /// watermark then: every cut that exists already happened to a transcript it never wrote). See
    /// [`Self::persist`].
    pub(super) cuts_seen: u64,
    /// Every known node (message), by id — spans the whole tree, not just the active path.
    pub(super) nodes: HashMap<String, Node>,
    /// Ids of the messages on the currently active path, root-first, parallel to `Session.messages`.
    /// `append_new` chains new messages' `parent_id` off `active.last()`; `rewrite` replaces this
    /// wholesale (a fresh linear chain) while preserving every id NOT in it (see the module doc's
    /// compaction-vs-branching section).
    pub(super) active: Vec<String>,
    /// [`BranchSummaryDetails`] for every [`Entry::BranchSummary`] seen, by that entry's own id — a
    /// `Node`'s materialized message keeps only the prose recap, not this structured file-tracking data,
    /// so a later branch summary that itself abandons a range containing an earlier one needs this index
    /// to fold that earlier summary's `read_files`/`modified_files` forward (see
    /// [`Self::branch_summary_details_within`]) rather than losing them once the prose-only message is
    /// all that's left to scan.
    pub(super) branch_summary_details: HashMap<String, BranchSummaryDetails>,
    /// The model in effect immediately after each message id (`None` = before the session's first
    /// message) — the last [`Entry::ModelChange`] anchored there, if any. See
    /// [`Self::record_model_change`]/[`Self::model_at`].
    pub(super) model_changes: HashMap<Option<String>, String>,
    /// Same idea as `model_changes`, for the portable thinking level (see [`Entry::ThinkingLevelChange`]/
    /// [`Self::record_thinking_level_change`]/[`Self::thinking_level_at`]).
    pub(super) level_changes: HashMap<Option<String>, String>,
    /// The title in effect immediately after each message id (`None` = before the session's first
    /// message) — the last [`Entry::TitleChange`] anchored there, if any, keyed the same way as
    /// `model_changes`/`level_changes`. `Some(None)` records an explicit clear (a rename that sanitized
    /// to empty) actually in effect at that anchor, distinct from no entry at all ("nothing ever
    /// recorded reaching this point"). Unlike `model_changes`/`level_changes`, this is consulted only
    /// for [`SessionRepo::fork`]/`fork_from_path`/`fork_at_entry_prefix`'s path-scoped resolution (pass
    /// 15 pi-parity fix, [`Self::title_at_or_root`]) — the *live* session's own displayed title
    /// (`meta.title`, restored by [`Self::open`]) still resolves whole-file-latest, matching pi's own
    /// `getSessionName` scanning every entry regardless of branch.
    pub(super) title_changes: HashMap<Option<String>, Option<String>>,
    /// The label currently set on each target id, by that id — the last (by file order)
    /// [`Entry::Label`] seen for it, with a `label: None` entry removing it from this map entirely
    /// (last-write-wins). See [`Self::set_label`]/[`Self::get_label`].
    pub(super) labels: HashMap<String, String>,
    /// Every [`Entry::Compaction`] record seen, by its own id — see [`CompactionRecord`]'s doc comment
    /// for why this lives in its own side index rather than `nodes`. Read back out by
    /// [`Self::tree`] (Track L25).
    pub(super) compactions: HashMap<String, CompactionRecord>,
    /// Every [`ExportEvent`] seen, in file order — see that type's own doc comment. Read back out by
    /// [`Self::export_events`] (Track L36).
    pub(super) events: Vec<ExportEvent>,
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
        Self::create_with(path, meta, &Layout::File)
    }

    /// [`create`](Self::create) in a specific [`Layout`] — what [`SessionRepo::create`] calls once its
    /// repo is configured for the segmented, per-tenant-sealed shape.
    pub(crate) fn create_with(
        path: PathBuf,
        meta: SessionMeta,
        layout: &Layout,
    ) -> std::io::Result<Self> {
        let mut header = Vec::new();
        write_line(&mut header, &Entry::Session(meta.clone()))?;
        let log = Log::create(path, &meta.id, layout, &header)?;
        let journal = journal_auth(&log);
        Ok(Self {
            journal,
            log,
            meta,
            persisted: 0,
            written: CompactionProvenance::default(),
            cuts_seen: agent_core::cut_watermark(),
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
    /// path's messages). A torn final line (crash mid-append) is skipped, and cut off the file since
    /// this open may write; a header is required.
    pub fn open(path: PathBuf) -> std::io::Result<(Self, Session)> {
        Self::open_with(path, None, &Layout::File, false)
    }

    /// [`open`](Self::open), but read-only: every write on the returned store errors instead of
    /// touching the log. What a fork source, a preview, an export and the archive source get — none of
    /// them owns the session, and in service mode a stray write from one would steal an epoch from
    /// whoever does.
    pub fn open_read_only(path: PathBuf) -> std::io::Result<(Self, Session)> {
        Self::open_with(path, None, &Layout::File, true)
    }

    /// Whether this store refuses writes (see [`open_read_only`](Self::open_read_only)).
    pub fn read_only(&self) -> bool {
        self.log.read_only()
    }

    /// The real open. `layout` forces the segmented shape (and supplies the codec); otherwise the
    /// layout is probed with `is_dir`, so a caller holding only a path — `main.rs`, `fork_from_path`,
    /// export — needs to know nothing about either shape.
    pub(crate) fn open_with(
        path: PathBuf,
        id_hint: Option<&str>,
        layout: &Layout,
        read_only: bool,
    ) -> std::io::Result<(Self, Session)> {
        let mut log = Log::open(path, id_hint, layout, read_only)?;
        let path = log.path().to_path_buf();
        let mut meta: Option<SessionMeta> = None;
        let mut nodes: HashMap<String, Node> = HashMap::new();
        // The active tip, updated in *file order* by whichever kind of entry sets it: a `Message`
        // entry always becomes the new tip (an append always continues the active chain), and a `Leaf`
        // entry redirects it to `target_id`. Tracking one variable this way — rather than remembering
        // "the last message" and "the last leaf" separately and picking between them afterward — is
        // what makes this correct when messages are appended *after* a `Leaf` (the tip must move back
        // to that latest message, not stay pinned at the leaf's target).
        let mut tip: Option<String> = None;
        // Chain nodes of a compaction round before its last one (`Entry::Compaction::chain_ids`):
        // read into the tree but not yet allowed to move `tip`, so a batch cut short mid-write leaves
        // the tip where it was.
        let mut held_tips: HashSet<String> = HashSet::new();
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
        // The active tip at the moment each tree node was read, by node id: where a node whose parent
        // never made it into the tree (an unreadable line) is re-chained once the scan is done — see
        // `rechain_orphans`.
        let mut tip_at_read: HashMap<String, Option<String>> = HashMap::new();
        // Whether the last line read was a whole, readable entry, and whether the scan stopped on a
        // read error rather than at the end — what decides `trim_torn_tail` below.
        let mut last_line_ok = true;
        let mut read_failed = false;

        let mut reader = log.lines()?;
        let mut raw = Vec::new();
        loop {
            // A genuine I/O read failure stops the load, keeping whatever was valid so far (only the
            // in-flight entry can be lost). An oversized or invalid-UTF-8 *line*, though, is a fully
            // read, boundary-known line, exactly like the deserialize-failure case below — skip just
            // that one and keep scanning, rather than discarding every good entry after it (which used
            // to happen here: a single bit-rotted or hand-edited line anywhere in the file silently
            // truncated the whole session).
            let oversized = match reader.next_line(&mut raw) {
                Ok(None) => break,
                Ok(Some(oversized)) => oversized,
                // A sealed line that will not open is never recoverable by reading on: the bytes are
                // not this session's, and a partial view would be silently overwritten by the next
                // write. Every other read failure keeps whatever was valid so far.
                Err(e) if is_seal_failure(&e) => return Err(e),
                Err(_) => {
                    read_failed = true;
                    break;
                }
            };
            last_line_ok = false;
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
            let parsed = parse_entry_lenient(&path, line);
            last_line_ok = parsed.is_ok();
            // Every tree node remembers the tip it was read under (see `tip_at_read`).
            if let Ok(
                Entry::Message { id: Some(id), .. }
                | Entry::BranchSummary { id, .. }
                | Entry::Custom { id, .. },
            ) = &parsed
            {
                tip_at_read.insert(id.clone(), tip.clone());
            }
            match parsed {
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
                            content: Arc::new(NodeContent::Message(message)),
                            timestamp,
                        },
                    );
                    if !held_tips.contains(&id) {
                        tip = Some(id);
                    }
                }
                // A target the tree never got (its line was unreadable) is ignored rather than
                // followed: a tip naming no node would materialize as an empty session.
                Ok(Entry::Leaf { target_id, .. }) => {
                    if target_id.as_ref().is_none_or(|t| nodes.contains_key(t)) {
                        tip = target_id;
                    }
                }
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
                            content: Arc::new(NodeContent::Message(branch_summary_message(
                                &summary,
                            ))),
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
                            content: Arc::new(NodeContent::Custom { kind, data }),
                            timestamp,
                        },
                    );
                    if !held_tips.contains(&id) {
                        tip = Some(id);
                    }
                }
                // Rewrites the target's content in place; never moves the tip (see `Entry::MessageEdit`).
                Ok(Entry::MessageEdit { target_id, message }) => {
                    if let Some(node) = nodes.get_mut(&target_id)
                        && node.as_message().is_some()
                    {
                        node.content = Arc::new(NodeContent::Message(message));
                    }
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
                    memory_notes,
                    user_requests,
                    chain_ids,
                    ..
                }) => {
                    if let Some((_, held)) = chain_ids.split_last() {
                        held_tips.extend(held.iter().cloned());
                    }
                    // Each record already carries every earlier round's file-provenance folded in
                    // (via `merge_provenance`, before it was even written — see
                    // `CompactionMeta::provenance`'s doc comment), so it's a complete,
                    // self-contained snapshot: the *last* one in file order (this loop runs in file
                    // order, so a later assignment here simply overwrites an earlier one) is all
                    // `Session.compaction` needs, restored below once `meta` (for its `compactions`
                    // counter) is available. Fixes a bug where this provenance was purely in-memory
                    // and silently reset to empty on every reopen past a compaction.
                    compaction_provenance = Some(CompactionProvenance {
                        read_files,
                        modified_files,
                        compactions: 0, // replaced by `meta.compactions` (the authoritative counter) below
                        last_reason,
                        todos,
                        memory_notes,
                        user_requests,
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
        // How far the replay actually got is the other half of the epoch fence: the first write
        // refuses unless the newest segment is still exactly this long.
        log.observe(&reader);
        drop(reader);
        let meta = meta.ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "session file has no header",
            )
        })?;
        // Validate (and, in a future format, upgrade) the header before trusting the rest of the file.
        let meta = migrate(meta, &path)?;
        // A writable open owns the tail: a torn final line (a crash, OOM or full disk mid-append) is
        // cut off before this store's first write, so that write starts on a line of its own instead
        // of being glued onto the fragment and lost with it. Only when the scan reached the end — a
        // read error leaves lines unread that the cut would take with it. Deferred to the first
        // write rather than done here, because an open is not yet ownership: a caller takes the
        // session lock after opening (`lock_session_writer`) and refuses if another process holds
        // it, and that process's in-flight append looks exactly like a torn tail. The segmented
        // layout never needs this: its reads are bounded and a writer only ever appends to a segment
        // it created.
        if !read_failed
            && !last_line_ok
            && let Log::File {
                path,
                read_only: false,
                torn_at,
            } = &mut log
        {
            *torn_at = Some(fs::metadata(&*path)?.len());
        }

        rechain_orphans(&mut nodes, &tip_at_read);
        let active = path_from_root(&nodes, tip.as_deref());
        // A custom entry (`NodeContent::Custom`) contributes nothing here — it's a real, positioned
        // node in `active`'s chain (see `Entry::Custom`'s doc comment), but not a message, so
        // `as_message` filters it out of the materialized `Session.messages`/LLM context.
        let journal = journal_auth(&log);
        let messages: Vec<Message> = materialize(&nodes, &active, &journal);
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
        // see its doc comment) — a real figure, not the estimate below. Everything *after* that
        // message (usually nothing) is still covered by `trailing_tokens`' own per-message estimate,
        // exactly like a live session's own turns since its last real usage snapshot.
        //
        // A session with no `usage`-carrying message at all (every message predates Round 1, or the
        // active path is entirely a compaction/branch-summary recap with none recorded) falls back to
        // a whole-transcript estimate, and marks it all as already accounted for, so the very next
        // real `trailing_tokens` call doesn't double-count it.
        //
        // `Session::restore_usage_snapshot` does both, and also recognizes a usage that predates the
        // last compaction (a reopen between a compaction and the next reply), which would otherwise
        // report the pre-compaction size. Its estimate sizes every turn a persisted
        // `Message::prior_turn_tokens` measured exactly.
        session.restore_usage_snapshot(agent_core::TokenFamily::for_model(&meta.model));
        Ok((
            Self {
                journal,
                log,
                meta,
                persisted,
                written: session.compaction.clone(),
                cuts_seen: agent_core::cut_watermark(),
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

    /// The on-disk path this session lives at — pi's `sessionFile`, surfaced via `get_state` for a
    /// client that wants to know exactly what's being written to. The `.jsonl` file in the single-file
    /// layout; the session's own directory in the segmented one.
    pub fn path(&self) -> &Path {
        self.log.path()
    }

    /// Where this session's `/session` working memory belongs — a `<...>.memory/` sibling of the
    /// `.jsonl`, or `<id>/memory` inside the session directory. Always ask the store rather than
    /// deriving it from [`path`](Self::path): `with_extension` on a dotted id (`<shard>.<opaque>`)
    /// would hand every session on a shard the same `<shard>.memory` directory.
    pub fn memory_dir(&self) -> PathBuf {
        self.log.memory_dir()
    }

    /// Whether a write has already found another owner in possession (see [`Superseded`]). Once true
    /// it stays true, and every later write fails — the owner checks this after each write and ends
    /// the session rather than pretending it still holds it.
    pub fn superseded(&self) -> bool {
        self.log.superseded()
    }

    /// Append one already-framed entry to the log.
    pub(super) fn append_entry(&mut self, entry: &Entry) -> std::io::Result<()> {
        let mut buf = Vec::new();
        write_line(&mut buf, entry)?;
        self.log.append(&buf)
    }
}
