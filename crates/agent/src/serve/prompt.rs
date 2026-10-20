//! The `prompt` command: the one command that runs the model.
//!
//! `cmd_prompt` is the whole path, in phases. `install_output_spec` and `refresh_prompt_inputs` make the
//! agent and system prompt match this turn; `begin_prompt` takes the message, acks it and mints the
//! per-prompt state ({PromptRun}); then the attempt loop: `run_attempt` drives one model run while the
//! session keeps answering stdin, `retry_backoff` waits out a retry delay the same way, and `end_prompt`
//! sends the terminal response and runs what the prompt left deferred.
//!
//! The run future holds `&mut session` for the whole attempt, so the busy loop inside `run_attempt`
//! reads only the fields the run does not borrow, as `cmd_compact`'s does.

use crate::serve::*;

/// What one `prompt` carries from its ack to its terminal response, across every retry attempt.
pub(super) struct PromptRun {
    /// The lifecycle record of this prompt; absent when no lifecycle sink is configured.
    life: Option<Arc<crate::lifecycle::Run>>,
    cancel: CancellationToken,
    /// Whether the run's last turn ended in a refusal.
    refused: Arc<std::sync::atomic::AtomicBool>,
    /// Whether an auto-triggered compaction is in flight this instant.
    is_compacting: Arc<std::sync::atomic::AtomicBool>,
    session_steers: Arc<crate::memory::SessionSteers>,
    /// False once stdin closed or a shutdown signal arrived during the run.
    stdin_open: bool,
    /// Commands that arrived mid-run and self-abort-and-proceed once it is idle.
    pending_deferred: Vec<(Option<String>, Value)>,
    /// MCP Events injection batch this prompt carries, if it is one.
    own_injection: Option<u64>,
    /// Batches steered in while the run was going.
    steered_injections: Vec<u64>,
    /// MCP tasks a previous process left in flight are resumed by the first attempt only.
    mcp_resume_first: bool,
    mcp_journaled: Vec<crate::mcp_resume::Resolved>,
    mcp_placeholders: std::collections::HashSet<String>,
}

impl SessionLoop {
    /// Runs one `prompt` to its terminal response. Returns whether the session goes on (`false` ends the
    /// loop: the writer went away, or stdin closed / a shutdown signal arrived mid-run).
    pub(super) async fn cmd_prompt(&mut self, cmd: &Value, id: Option<String>) -> bool {
        // Whatever moved the model since the last check — a branch or session restore included — a
        // prompt never runs on one the grant's `limits` forbid.
        if let Some(why) = self.mode.model_refusal(&self.current_model) {
            return self.emit(response(id, "prompt", false, None, Some(&why)));
        }
        if let Err(alive) = self.install_output_spec(cmd, &id).await {
            return alive;
        }
        self.refresh_prompt_inputs().await;
        let mut prompt = match self.begin_prompt(cmd, &id).await {
            Ok(prompt) => prompt,
            Err(alive) => return alive,
        };
        // A run that ends in a transient-looking `Err` (`crate::retry::is_retryable_whole_run`
        // — a superset of `run_turn`'s own mid-stream retry's `is_retryable_mid_stream`, plus
        // raw HTTP status digits appropriate only at this outer layer) is automatically
        // re-invoked against the *same* session — resuming from wherever it left off, not
        // restarting the turn — up to `retry::MAX_RUN_RETRIES` times with backoff. pi's
        // `agent-session.ts` equivalent (`_prepareRetry`/`isRetryableAssistantError`); ours is
        // gated by the same `current_auto_retry` toggle as the mid-stream layer below it (one
        // user-facing "retries on/off" concept, even though internally it's two layers), and
        // never retries past a client disconnect or an explicit `abort`
        // (`is_retryable_whole_run` already excludes `Error::Cancelled` transitively through
        // `is_retryable_mid_stream`, and `stdin_open`/`cancel.is_cancelled()` below are
        // redundant, cheap belt-and-suspenders on top of that). The backoff sleep itself *is*
        // raced against `abort`/`abort_retry`/shutdown (see the `select!` around the `sleep`
        // below) — a client abort during backoff takes effect immediately, not just once the
        // next attempt starts.
        let mut retry_attempt: u32 = 0;
        // Retries since the run last made progress — what `max_retries` and the backoff count.
        // `retry_attempt` above keeps counting every retry of this prompt (lifecycle, the
        // closing `auto_retry_end`); this one resets when a failed attempt still committed a
        // step, so a long run that recovers from one blip doesn't carry it against the next.
        let mut stalled_retries: u32 = 0;
        // Whether the final post-attempt persist (below) failed to write to disk — surfaced in
        // the terminal `prompt` response instead of being silently swallowed, since an agent run
        // that appears to succeed while its transcript failed to persist is exactly the failure
        // mode this exists to catch. Always set from *that* persist's own outcome alone, once
        // per attempt: it re-persists the whole current session state fresh, so it's a strict
        // superset of any mid-run checkpoint along the way (see the `checkpoint_rx.recv()` arm's
        // comment below) — a checkpoint that failed but was then followed by a successful final
        // persist must report success, not a stale failure the superseding write already fixed.
        let mut persist_error: Option<String>;
        let result = 'retry: loop {
            let steps_at_start = self.session.steps;
            prompt.refused.store(false, Ordering::Relaxed);
            prompt.is_compacting.store(false, Ordering::Relaxed);
            prompt.pending_deferred.clear();
            let attempt_result = self.run_attempt(&mut prompt, retry_attempt).await;
            // Do **not** clear `running` here. The attempt's future has resolved, but the
            // prompt may still retry (backoff below) or emit its terminal response; an HTTP
            // POST that already unpinned on the ack would otherwise be reapable in that gap.
            // Cleared after `emit!(frame)` once the command is actually done.

            // Steers still waiting on a skill belong to this run, which is over: cancel them
            // (before any `new_session`/`switch_session`/`fork`/`clone` waiting behind it runs)
            // and say so; anything ready behind them is queued in order.
            self.pending_steers
                .end_run(
                    matches!(&attempt_result, Err(agent_core::Error::Cancelled)),
                    &self.mcp_skills,
                    &self.steering,
                    &self.out_tx,
                )
                .await;
            while self.steer_done_rx.try_recv().is_ok() {}

            // A checkpoint sent right as the run above ended (see `agent_core::Agent`'s final,
            // unconditional post-run `checkpoint.checkpoint(session)`) races the `r = &mut run`
            // arm above under `biased` select: whichever is ready first wins, so a checkpoint
            // that arrived in the same instant the run itself completed can be left sitting in
            // `checkpoint_rx`'s buffer, undrained. Track (pi-parity fix, found via this crate's
            // own test suite going red): left alone, that stale message would only ever get
            // drained during some *later*, unrelated prompt's own inner loop — by which point an
            // intervening `new_session`/`switch_session`/`fork`/`switch_branch` may have already
            // swapped `persistence` onto a *different* session's store, so draining it there
            // would silently write *this* turn's content into that other session's file (and,
            // via `append_new`'s `messages.len() <= self.persisted` dedup guard, could mask that
            // later turn's own real content entirely). Discarded here, not persisted: the
            // unconditional `persist_blocking` right below is a strict superset of it for this
            // turn's own final state, and this must run *before* any later command gets a chance
            // to swap `persistence` out from under it.
            while self.checkpoint_rx.try_recv().is_ok() {}
            // MCP journal writes still queued belong to *this* session: persist them now,
            // before a deferred command can swap `persistence` (they would be dropped by the
            // session-id check, but this session would lose them).
            while let Ok((kind, data)) = self.mcp_task_rx.try_recv() {
                journal_mcp(&mut self.persistence, &kind, data);
            }

            // Whether this attempt is about to be retried. Decided *here*, before the persist
            // below, and reused verbatim as the retry arm's own guard — the two must not be
            // able to disagree, because the closing error record has to be dropped on the same
            // side of the persist in memory as it is on disk.
            if self.session.steps > steps_at_start {
                stalled_retries = 0;
            }
            let will_retry = matches!(&attempt_result, Err(e)
            if self.current_auto_retry
                && prompt.stdin_open
                && !prompt.cancel.is_cancelled()
                && stalled_retries < self.run_retry_policy.max_retries
                && crate::retry::is_retryable_whole_run(e));

            // The failed attempt's closing error record (`run_events_steered` always pushes one
            // on an `Err`-ending run) must not survive into the retry — this is the *same* run
            // resuming from scratch, not a client's own fresh prompt, so it has to be genuinely
            // invisible in the transcript rather than stacked under the retry's real response.
            //
            // Popped *before* the persist below, not after it. `SessionStore::append_new` is
            // keyed on a count (`messages.len() <= self.persisted` skips, and each append sets
            // `persisted = messages.len()`), so it has no way to notice the message list got
            // *shorter*: persisting N+1 messages and then shrinking to N desynced the two for
            // the rest of the session. The retry re-produced the assistant message, `len` was
            // back to N+1, the append early-returned, and that message was silently **never
            // written** — with the following `tool_result` then chaining onto the error record's
            // node instead. Nothing reconciles a shrink (`rewrite` is only reachable via
            // compaction/fork/branch), and the existing e2e coverage missed it entirely by
            // asserting against the live in-memory transcript rather than a reload.
            //
            // Kept rather than dropped: if the retry is cancelled during its backoff below, this
            // run ends here after all, and the record has to go back — it is the only thing
            // keeping the session's role alternation valid for whatever prompt comes next (see
            // `a_prompt_after_a_failed_run_does_not_double_push_a_user_turn` in `agent_core`;
            // there is no guard against a double-pushed `user` turn anywhere else).
            let mut withheld_error_record = None;
            if will_retry {
                withheld_error_record = self
                    .session
                    .messages
                    .last()
                    .filter(|m| m.error_message.is_some())
                    .cloned();
                self.session.pop_error_record();
            }

            // Persist whatever this attempt produced (even a failed one may have committed a tool
            // round-trip or two before the error) before deciding whether to retry.
            {
                let (p, r) =
                    persist_blocking(self.persistence.take_for_blocking(), self.session.clone())
                        .await;
                self.persistence = p;
                persist_error = r.err().map(|e| e.to_string());
            }

            match &attempt_result {
                Err(e) if will_retry => {
                    retry_attempt += 1;
                    stalled_retries += 1;
                    let delay = self.run_retry_policy.backoff(stalled_retries);
                    let _ = self.out_tx.send(auto_retry_frame(
                        id.clone(),
                        stalled_retries,
                        self.run_retry_policy.max_retries,
                        delay.as_millis() as u64,
                        &e.to_string(),
                    ));
                    let retry_cancelled = self.retry_backoff(&mut prompt, delay).await;
                    if retry_cancelled {
                        // No retry after all — this run ends on the attempt that just failed,
                        // so the closing error record withheld above has to go back, in memory
                        // *and* on disk. Without it the session's last message is the user's own
                        // unanswered prompt, and the next `prompt` would stack a second
                        // consecutive `user` turn, a shape no dialect accepts. A plain append
                        // (never a compacted rewrite): any compaction this attempt performed was
                        // already written by the persist above, and this only adds one message
                        // on top of it.
                        if let Some(record) = withheld_error_record.take() {
                            self.session.push(record);
                            let (p, r) = persist_blocking(
                                self.persistence.take_for_blocking(),
                                self.session.clone(),
                            )
                            .await;
                            self.persistence = p;
                            if let Err(e) = r {
                                persist_error = Some(e.to_string());
                            }
                        }
                        let _ = self.out_tx.send(auto_retry_end_frame(
                            id.clone(),
                            false,
                            retry_attempt,
                            Some("retry cancelled"),
                        ));
                        break 'retry attempt_result;
                    }
                    continue 'retry;
                }
                _ => {
                    // A terminal notice only for a sequence that actually retried at least
                    // once — a first-attempt success or an immediately-non-retryable failure
                    // never emitted `auto_retry`, so it needs no "the retries ended" notice
                    // either. Mirrors pi's own `this._retryAttempt > 0` guard.
                    if retry_attempt > 0 {
                        let (success, final_error) = match &attempt_result {
                            Ok(_) => (true, None),
                            Err(e) => (false, Some(e.to_string())),
                        };
                        let _ = self.out_tx.send(auto_retry_end_frame(
                            id.clone(),
                            success,
                            retry_attempt,
                            final_error.as_deref(),
                        ));
                    }
                    break 'retry attempt_result;
                }
            }
        };

        self.end_prompt(prompt, id, result, persist_error).await
    }

    /// `output_schema`: install, change or remove the structured-output tool for this prompt.
    /// `Err` carries whether the session goes on, when the prompt was rejected.
    async fn install_output_spec(&mut self, cmd: &Value, id: &Option<String>) -> Result<(), bool> {
        // `output_schema` makes this prompt a callable function: the model must fill the schema
        // in via `structured_output` rather than answering in prose, and the validated payload
        // rides the terminal response's `data.structured_output`. Handled before the ack, so a
        // malformed schema is rejected outright rather than acknowledged and then failed.
        //
        // Installing (or removing, or changing) a schema changes both the *tool set* and the
        // system-prompt section that tells the model to use it, so both are rebuilt — through
        // the very same `build_agent` path `set_model` already uses. A prompt repeating the
        // schema it already had rebuilds nothing.
        match parse_output_spec(cmd) {
            Err(e) => {
                if !self.emit(response(id.clone(), "prompt", false, None, Some(&e))) {
                    return Err(false);
                }
                return Err(true);
            }
            Ok(spec) if spec != self.current_output_spec => {
                self.structured_output = match &spec {
                    Some((schema, description)) => {
                        match crate::tools::structured_output::StructuredOutput::new(
                            schema.clone(),
                            description.clone(),
                            self.output_slot.clone(),
                        ) {
                            Ok(tool) => Some(Arc::new(tool)),
                            Err(e) => {
                                if !self.emit(response(id.clone(), "prompt", false, None, Some(&e)))
                                {
                                    return Err(false);
                                }
                                return Err(true);
                            }
                        }
                    }
                    None => None,
                };
                self.current_output_spec = spec;
                self.memory_sections = crate::memory::mount_sections(&self.mounts).await;
                self.static_system = crate::resources::build_static_system_prompt_with_context(
                    &crate::resources::PromptOptions {
                        base: self.sandbox_system.as_deref(),
                        default_base: &self.cfg.system,
                        append: self
                            .cfg
                            .append_system
                            .as_deref()
                            .or(self.sandbox_append.as_deref()),
                        cwd: &self.cwd,
                        include_context_files: self.cfg.context_files,
                        skills: &self.skills,
                        context_window: agent_core::capabilities(&self.cfg.model).context_window,
                        has_read: self.has_read,
                        has_todo: self.has_todo,
                        has_structured_output: self.structured_output.is_some(),
                        has_memory: self.has_memory,
                        memory_sections: &self.memory_sections,
                        project_trusted: self.project_trusted,
                        agents: &self.cfg.agents,
                        disk_overrides: self.disk_overrides,
                    },
                    &self.context_files,
                );
                self.agent = build_agent(
                    &self.exec_cell,
                    self.client.clone(),
                    &full_system(&self.static_system, &self.cwd, &self.mcp_skills),
                    &self.cfg,
                    &self.current_model,
                    self.current_thinking,
                    self.current_level,
                    self.current_auto_compaction,
                    self.current_auto_retry,
                    self.current_block_images,
                    self.current_image_auto_resize,
                    self.persistence.session_id(),
                    &self.write_locks,
                    &self.checkpoint,
                    self.subagent_ctx.as_ref(),
                    self.structured_output.as_ref(),
                    self.memory_tool.as_ref(),
                    self.approval.as_ref(),
                    &self.mcp_enabled,
                    &self.skills,
                );
            }
            Ok(_) => {}
        }
        Ok(())
    }

    /// What every turn refreshes before it runs: the output slot, the MCP skills listing, the agent when
    /// MCP Apps or a server's tools moved, and the system prompt's date.
    async fn refresh_prompt_inputs(&mut self) {
        // A previous prompt's answer must never be mistaken for this one's.
        self.output_slot.clear();
        // Refresh the cheap, time-varying part of the system prompt (the date) every turn —
        // the expensive discovery-based static half only changes on `set_model`/`set_thinking`/
        // `reload`, so it's cached rather than recomputed here.
        // A skills listing whose `ttlMs` ran out, or that a server said changed, is re-listed
        // now — when it is needed — so a skill published after connect is offered this turn.
        self.mcp_skills.refresh().await;
        // The session may be one this process never ran (a switch, a fork, a restart) whose
        // transcript still holds an MCP skill's SKILL.md: the model is acting on it, so the
        // code-execution gate must be too.
        self.mcp_enabled
            .skill_session()
            .restore_from_transcript(&self.session.messages);
        // The session's MCP Apps connections moved since the agent was built (an apps dial
        // landed, or went into backoff): rebuild, so this turn advertises what is up now.
        let apps_generation_now = self.mcp_enabled.apps().map(|view| view.generation());
        // Likewise a server re-listed its tools since then.
        let mcp_tools_now = self.cfg.mcp_tools.generation();
        if apps_generation_now != self.apps_generation || mcp_tools_now != self.mcp_tools_generation
        {
            self.apps_generation = apps_generation_now;
            self.mcp_tools_generation = mcp_tools_now;
            self.agent = build_agent(
                &self.exec_cell,
                self.client.clone(),
                &full_system(&self.static_system, &self.cwd, &self.mcp_skills),
                &self.cfg,
                &self.current_model,
                self.current_thinking,
                self.current_level,
                self.current_auto_compaction,
                self.current_auto_retry,
                self.current_block_images,
                self.current_image_auto_resize,
                self.persistence.session_id(),
                &self.write_locks,
                &self.checkpoint,
                self.subagent_ctx.as_ref(),
                self.structured_output.as_ref(),
                self.memory_tool.as_ref(),
                self.approval.as_ref(),
                &self.mcp_enabled,
                &self.skills,
            );
        }
        // And one that changes them during this run reaches it at the next turn boundary.
        self.steering
            .set_tool_source(Some(crate::tools::mcp::tool_source(
                self.cfg.mcp_tools.clone(),
                self.mcp_enabled.clone(),
                self.mcp_tools_generation,
                self.mcp_filter.clone(),
            )));
        self.agent.set_system(full_system(
            &self.static_system,
            &self.cwd,
            &self.mcp_skills,
        ));
    }

    /// Takes the message, queues it on the session, acks it and mints the prompt's state. `Err` carries
    /// whether the session goes on, when the prompt was rejected before it started.
    async fn begin_prompt(&mut self, cmd: &Value, id: &Option<String>) -> Result<PromptRun, bool> {
        // A `prompt` with no (or non-string) `message` is a malformed command: running an
        // empty user turn would spend a model call on nothing and still report success.
        let Some(message) = cmd.get("message").and_then(Value::as_str) else {
            if !self.emit(response(
                id.clone(),
                "prompt",
                false,
                None,
                Some("missing `message`"),
            )) {
                return Err(false);
            }
            return Err(true);
        };
        let message = expand_message(
            message,
            &self.skills,
            &self.prompt_templates,
            &self.mcp_skills,
        )
        .await;
        // Optional image attachments: `images: [{media_type, data}]` (base64). Builds a
        // multimodal user turn; absent or empty → a plain text turn.
        let images = parse_images(cmd.get("images"));
        // MCP tasks a previous process left in flight are waited on and answered before this
        // turn's model call (see `crate::mcp_resume`), by the first attempt only.
        // A resumer that finished with a task's server unreachable left that task pending:
        // a fresh one tries it again now (its finished results carry over).
        //
        // Journal writes still queued in `mcp_task_rx` are not drained here, and nothing
        // below depends on them. A finished task's result: a restarted resumer is handed the
        // old one's results (`mcp_carried`), so it never re-polls. A placeholder mark queued
        // as this prompt arrives: its placeholder is not on the path yet (this prompt's
        // `resume_into_turn` splices it, adding the resumer's own placeholder ids to the
        // journal's), and once a run has put one there, the run-end drain journals its mark
        // before the next prompt reads the journal.
        let mut mcp_carried = Vec::new();
        if self.mcp_resumer.session_id() != self.persistence.session_id() {
            self.mcp_resumer = self.start_mcp_resumer(&[]);
        } else if self.mcp_resumer.needs_retry() {
            mcp_carried = self.mcp_resumer.finals();
            self.mcp_resumer = self.start_mcp_resumer(&mcp_carried);
        }
        let mut mcp_journaled = crate::mcp_resume::results(
            &self
                .persistence
                .active_journal(crate::mcp_resume::RESULT_ENTRY_KIND),
        );
        mcp_journaled.extend(mcp_carried);
        let mcp_placeholders = crate::mcp_resume::placeholder_ids(
            &self
                .persistence
                .active_journal(crate::mcp_resume::PLACEHOLDER_ENTRY_KIND),
        );
        if images.is_empty() {
            self.session.user(message);
        } else {
            self.session
                .push(agent_core::Message::user_with_images(message, images));
        }
        // MCP App views' new model context rides this turn as a request-only block of its
        // own — untrusted data, labelled as such, never in the user's text, never persisted,
        // and never in the system prompt, whose bytes (and so the prompt cache) it would
        // otherwise churn on every update. See `Session::attach_request_block`.
        if let Some(context) = crate::tools::mcp_apps::take_undelivered_context(&self.mcp_enabled) {
            self.session.attach_request_block(context);
        }
        // Acknowledge immediately — the turn is queued and about to start — rather than
        // leaving a client with no signal until the (possibly much later) terminal response.
        if !self.emit(ack(id.clone(), "prompt")) {
            return Err(false);
        }
        // The reaper must see this session as mid-run from the ack, not from when the model
        // call actually starts. An HTTP POST returns 202 on this ack and unpins; the gap
        // between ack and `run_events_steered` (and the retry-backoff window between
        // attempts) is still "a prompt is in flight" — reaping here would drop a run the
        // client was just told had started. Cleared after the terminal `prompt` response.
        // `swap`, not `store`: the gauge must follow *transitions*. The retry path below
        // re-asserts this flag while a run is already in flight, and an unconditional `inc`
        // there would count one prompt twice and never come back down.
        if !self.running.swap(true, Ordering::Relaxed)
            && let Some(m) = &self.cfg.metrics
        {
            m.runs_in_flight.inc();
        }
        // One run_id for the whole prompt, including whole-run retries. Minted here, next to
        // the ack the client just got, so a crash between accept and the loop still has a
        // `started` in flight. Unconfigured (`NoLifecycle`) skips construction entirely.
        let life = self.cfg.lifecycle.enabled().then(|| {
            crate::lifecycle::Run::begin(
                Arc::clone(&self.cfg.lifecycle),
                self.persistence.session_id().to_string(),
                // One replica emits for many tenants in service mode, so every event says
                // whose run it is. Absent (and omitted on the wire) otherwise.
                self.mode.tenant().map(str::to_owned),
                id.clone(),
                self.current_model.clone(),
            )
        });
        let cancel = CancellationToken::new();
        // Whether the run's *last* turn ended in a refusal — set from every `TurnEnd`, so by
        // the time the run returns it reflects only the final one (any refusal earlier in a
        // multi-turn run is superseded once the model goes on to call tools or answer plainly).
        // Reset per retry attempt, so the value read after the loop reflects only the last.
        let refused = Arc::new(std::sync::atomic::AtomicBool::new(false));
        // Whether an auto-triggered compaction is *currently* in flight this instant — pi's
        // `isCompacting`, surfaced from `get_state`. Set on `AgentEvent::CompactionStart`,
        // cleared on `Compacted`/`CompactionFailed` *or* the very next event of any other kind
        // (`TurnStart` for the run's next turn, ordinarily) — `Agent::compact` can end without
        // ever emitting `Compacted` (an empty-summary no-op, or a failure reported via
        // `CompactionFailed` rather than propagated — see that event's doc comment), and its own
        // model call uses a discarding inner sink, so nothing else can arrive between
        // `CompactionStart` and whatever naturally follows it; clearing on "anything else" is
        // exact, not a guess.
        // Manual `compact`/branch-summary-on-`switch_branch` don't touch this: both only ever
        // run from the idle main loop, which processes one command to completion before reading
        // the next — no concurrent `get_state` could ever observe them in flight anyway.
        let is_compacting = Arc::new(std::sync::atomic::AtomicBool::new(false));
        // The `/session` steers (pressure nudge, compaction reminder) — the observer `run`
        // shares, see `crate::memory::SessionSteers`. Declared here (per prompt-run) so its
        // once-per-fill-cycle nudge state survives the per-retry-attempt observer rebuild below;
        // the pressure point is re-pointed per attempt (a `set_model` changes the trigger).
        let session_steers = Arc::new(crate::memory::SessionSteers::new(
            self.session_memory_active,
            u32::MAX,
        ));

        // Start recording this run's frames for mid-turn replay (see `OutFanout::turn`). Here,
        // outside the auto-retry loop below, so one recording spans the whole `prompt`: a client
        // that never dropped saw every attempt's frames and the `auto_retry_start` notices
        // between them, and a client that re-attaches is owed the same. The idle loop's
        // `sync_history!` ends the recording once the run is over and its messages have committed.
        lock_ignoring_poison(&self.out_conn).begin_turn();

        // Drive the run while staying responsive to stdin: `abort` cancels it, `steer` queues a
        // mid-run injection and `follow_up` a stop-boundary one; any other command is
        // rejected as busy (the session is borrowed by the in-flight run). If stdin closes
        // mid-run, cancel and drain. The block scopes the run's `&mut session` borrow so we can
        // persist after.
        // pi-parity (Task 4, serve pass 19): `compact`/`switch_session`/`fork`/`clone`/
        // `new_session` received while this run (or its auto-retry backoff wait) is in flight
        // self-abort-and-proceed instead of being rejected as busy — cancelled via the exact
        // same `cancel.cancel()`/deferred-until-idle discipline `abort` itself already uses
        // (see `pending_abort_acks` just below), then queued here (raw command + its own `id`)
        // to run their ordinary idle-path logic once this attempt has actually gone idle, in
        // the order received. Declared outside the loop (not inside, like `pending_abort_acks`)
        // and `clear()`-reset every attempt, not re-declared: a deferred command always forces
        // this attempt to end via cancellation (a cancelled/aborted result is never itself
        // retryable — see the match on `attempt_result` below), so it's always populated and
        // drained within the very same iteration, but the drain happens *after* this loop
        // returns, once `session`/`persistence` are no longer borrowed by `run` — a variable
        // declared inside the loop body would be dropped at `break` and unreachable there.
        let pending_deferred: Vec<(Option<String>, Value)> = Vec::new();
        // MCP Events injection batches this run carries: its own prompt's, and each one
        // steered in while it runs (with the text the model would receive). Settled once
        // the run is over — see "A batch is delivered only once" below.
        let own_injection = crate::tools::mcp_events::injection_batch(cmd);
        let steered_injections: Vec<u64> = Vec::new();
        Ok(PromptRun {
            life,
            cancel,
            refused,
            is_compacting,
            session_steers,
            stdin_open: true,
            pending_deferred,
            own_injection,
            steered_injections,
            mcp_resume_first: true,
            mcp_journaled,
            mcp_placeholders,
        })
    }

    /// One attempt at the run: the model call, driven while the session keeps answering stdin.
    async fn run_attempt(
        &mut self,
        prompt: &mut PromptRun,
        retry_attempt: u32,
    ) -> Result<(), agent_core::Error> {
        let cancel = prompt.cancel.clone();
        let refused = prompt.refused.clone();
        let is_compacting = prompt.is_compacting.clone();
        let session_steers = prompt.session_steers.clone();
        let life = prompt.life.clone();
        // `abort` command ids received while this run is still unwinding — acked only once
        // the run has actually gone idle (see the flush right after `attempt_result` below,
        // and that ack's own doc comment for why it's deferred rather than sent the instant
        // `cancel.cancel()` is called).
        let mut pending_abort_acks: Vec<Option<String>> = Vec::new();
        let tx = self.out_tx.clone();
        let mcp_task_journal = self.mcp_task_tx.clone();
        let mcp_journal_session = self.persistence.session_id().to_owned();
        let refused_sink = refused.clone();
        let is_compacting_sink = is_compacting.clone();
        // Live token/step counters a `get_state`/`get_session_stats` sent while this run is in
        // flight can answer from (see the busy-loop's own arms for those types below) — seeded
        // fresh from the session's *current* totals every attempt (a retry may follow a partial,
        // already-persisted success from an earlier attempt in this same loop), then kept current
        // from the same events a streaming client already observes.
        let live_stats = Arc::new(LiveStats::from_session(&self.session));
        let live_stats_sink = live_stats.clone();
        // A clone for the observer to push the compaction reminder + pressure nudge through —
        // consumed by this attempt's `move` closure, so re-cloned per attempt.
        let steer_on_compact = self.steering.clone();
        let session_steers_sink = session_steers.clone();
        // The live-prompt size at which to warn of an approaching compaction, against the
        // *current* model (a `set_model` mid-session changes it): the same trigger
        // `build_agent` hands this session's `Agent`.
        session_steers.set_pressure_point(crate::memory::compaction_pressure_point(
            compaction_config(&self.cfg, &self.current_model).trigger_tokens(),
        ));

        // Mark this session busy for the reaper's benefit (daemon mode). Already set at
        // the ack above; re-asserted here so a retry attempt is still protected if anything
        // cleared it. The matching clear is *after* the whole prompt, not after this
        // attempt — retry backoff is still mid-run.
        if !self.running.swap(true, Ordering::Relaxed)
            && let Some(m) = &self.cfg.metrics
        {
            m.runs_in_flight.inc();
        }
        if let Some(life) = &life {
            life.set_attempt(retry_attempt);
        }
        let life_obs = life.clone();
        let attempt_result = {
            // A steered MCP Events batch is delivered the moment the model receives it: the
            // run's `Steered` event reports its tag after the transcript holding it is
            // checkpointed, and it is recorded — durably — right then, never re-derived
            // from the transcript afterwards (a compaction may have summarized it away).
            let receipts_sink = self.mcp_events.as_ref().map(|hub| hub.receipts());
            let mut sink = move |ev: AgentEvent| {
                if let AgentEvent::Steered { tags, .. } = &ev
                    && let Some(receipts) = &receipts_sink
                {
                    for tag in tags {
                        receipts.received(*tag);
                    }
                }
                // Set on `CompactionStart`, cleared on literally anything else — see
                // `is_compacting`'s own declaration above for why that's exact, not a
                // conservative approximation.
                is_compacting_sink.store(
                    matches!(ev, AgentEvent::CompactionStart { .. }),
                    Ordering::Relaxed,
                );
                // The `/session` steers: the nudge before a compaction, the recall
                // reminder after one.
                session_steers_sink.observe(&ev, &steer_on_compact);
                if let AgentEvent::TurnEnd { stop_reason, step } = &ev {
                    refused_sink.store(*stop_reason == StopReason::Refusal, Ordering::Relaxed);
                    live_stats_sink.set_steps(*step);
                }
                // A subagent's requests count toward this session's totals (the loop
                // already folds them into the `Session`); mirrored here so mid-run
                // stats agree with what the session will say once the run ends.
                if let AgentEvent::ToolUsage { usage, .. } = &ev {
                    live_stats_sink.record_delegated_usage(usage);
                }
                if let AgentEvent::Stream(StreamEvent::Usage(usage)) = &ev {
                    live_stats_sink.record_usage(usage);
                }
                // B-L1: mirrors the same events into `pending_tool_ids`, so `get_state`
                // can answer "which calls are still running" mid-turn — see that
                // field's own doc comment.
                if let AgentEvent::ToolStart { id, .. } = &ev {
                    live_stats_sink.tool_started(id.clone());
                }
                if let AgentEvent::ToolEnd { id, .. } = &ev {
                    live_stats_sink.tool_ended(id);
                }
                // Same mirroring, for the `todo` list — see `LiveStats::todos`. The tool
                // validates before it emits, so a rejected call never lands here and the
                // mirror only ever holds a list the model actually committed.
                if let AgentEvent::ToolProgress {
                    name,
                    details: Some(d),
                    ..
                } = &ev
                    && name == crate::tools::todo::NAME
                    && let Some(todos) = d.get("todos")
                {
                    live_stats_sink.todos_updated(todos.clone());
                }
                // Best-effort: a sync sink can't break the control loop. If the writer is
                // gone the send fails here and the terminal response send below detects it
                // via `emit!` and stops the loop. An unserializable event is skipped rather
                // than emitted as a malformed frame (see `event_frame`).
                if let AgentEvent::ToolProgress {
                    id,
                    details: Some(d),
                    ..
                } = &ev
                    && let Some(record) = d.get(crate::tools::mcp::MCP_TASK_DETAILS_KEY)
                {
                    let _ = mcp_task_journal.send((
                        crate::mcp_resume::TASK_ENTRY_KIND.to_owned(),
                        crate::mcp_resume::task_entry(record, id, &mcp_journal_session),
                    ));
                }
                if let Some(life) = &life_obs {
                    life.observe(&ev);
                }
                if let Some(frame) = event_frame(ev) {
                    let _ = tx.send(frame);
                }
            };
            let resume_first = std::mem::take(&mut prompt.mcp_resume_first);
            let resumer = &self.mcp_resumer;
            let journaled = &prompt.mcp_journaled;
            let placeholders = &prompt.mcp_placeholders;
            let run_cancel = cancel.clone();
            let run = crate::tools::mcp::with_session_host(self.mcp_host.clone(), async {
                if resume_first {
                    crate::mcp_resume::resume_into_turn(
                        &mut self.session,
                        resumer,
                        journaled,
                        placeholders,
                        &mut sink,
                        &run_cancel,
                    )
                    .await;
                }
                self.agent
                    .run_events_steered(
                        &mut self.session,
                        &mut sink,
                        run_cancel.clone(),
                        self.steering.clone(),
                    )
                    .await
            });
            tokio::pin!(run);
            // No timer when unconfigured or `--lifecycle-heartbeat-secs 0`. Tool-edge
            // samples still fire from `observe` above.
            let mut heartbeat = life
                .as_ref()
                .and_then(|_| crate::lifecycle::heartbeat_interval(self.cfg.lifecycle_heartbeat));
            // Consume the immediate first tick so the first sample is one interval in,
            // not stacked on `started`.
            if let Some(h) = heartbeat.as_mut() {
                h.tick().await;
            }
            let life_beat = life.clone();
            loop {
                tokio::select! {
                    biased;
                    r = &mut run => break r,
                    // A `/skill:` steer's expansion finished: queue it, and anything that
                    // arrived after it, in order.
                    Some(done) = self.steer_done_rx.recv() => {
                        self.pending_steers.expanded(done, &self.mcp_skills, &self.steering, &self.out_tx);
                    }
                    // A shutdown request mid-run gets the same treatment as stdin closing:
                    // cancel and let the run unwind so the code below can persist before we
                    // fall out of the outer loop (`!stdin_open` breaks it, further down).
                    sig = self.shutdown.wait() => {
                        self.shutdown_cause = Some(sig);
                        prompt.stdin_open = false;
                        cancel.cancel();
                    }
                    // Drain and persist each mid-run checkpoint as it arrives (see
                    // `ChannelCheckpoint`). `persistence` isn't touched by `run` itself (only
                    // `session` is borrowed there), so reassigning it here is safe exactly like
                    // `stdin_open`/`cancel` being mutated from a sibling branch above. Any
                    // checkpoint still left undrained once the run ends is explicitly swept up
                    // right after this loop, below, before it can leak into some later, unrelated
                    // turn — see that drain's own comment for why "harmless, the final persist
                    // is a superset" isn't quite true on its own.
                    Some((messages, provenance, edited, cuts, ack)) = self.checkpoint_rx.recv() => {
                        // The agent is waiting on the acknowledgement before it dispatches
                        // the next tool, and acts on a failure — the tool does not run (see
                        // `ChannelCheckpoint`). Sent only once the write is done.
                        let (p, r) = write_then_ack(
                            persist_messages_blocking(self.persistence.take_for_blocking(), messages, provenance, edited, cuts),
                            ack,
                        )
                        .await;
                        self.persistence = p;
                        // Note this deliberately does *not* republish the fanout's catch-up
                        // history, even though a perfectly good snapshot is right here. The
                        // catch-up's contract is base-plus-frames: `history` is the transcript
                        // as of this run's start and `OutFanout::turn` is every frame since, so
                        // a reconnecting client reconstructs exactly what a never-dropped one
                        // saw. Advancing `history` here would put a mid-turn-committed message
                        // in the catch-up *and* again in the replayed frames — rendered twice.
                        // See `sync_history!`, which republishes once the run is over.
                        //
                        // Logged, not tracked in `persist_error`: the unconditional final
                        // persist below re-persists the whole session fresh regardless, so
                        // this checkpoint's own outcome never affects what's reported to the
                        // client — only worth knowing about here for operational visibility.
                        if let Err(e) = r {
                            tracing::warn!(error = %e, "mid-run checkpoint failed to persist");
                        }
                    }
                    // After the checkpoint arm (`biased`), so the assistant's `tool_use` is on
                    // disk before the task record that answers it.
                    Some((kind, data)) = self.mcp_task_rx.recv() => {
                        journal_mcp(&mut self.persistence, &kind, data);
                    }
                    maybe_line = self.input_rx.recv(), if prompt.stdin_open => match maybe_line {
                        Some(l) => {
                            let l = l.trim();
                            if l.is_empty() {
                                continue;
                            }
                            let c: Value = match serde_json::from_str(l) {
                                Ok(v) => v,
                                Err(e) => {
                                    let _ = self.out_tx.send(response(None, "parse", false, None, Some(&format!("invalid JSON: {e}"))));
                                    continue;
                                }
                            };
                            let cid = c.get("id").and_then(Value::as_str).map(str::to_string);
                            let ctype = c.get("type").and_then(Value::as_str).unwrap_or("");
                            // Service mode gates here too, *ahead* of the self-abort arm
                            // below: `fork`/`clone`/`new_session`/`switch_session` cancel
                            // the running prompt before they defer, so refusing them only
                            // at the idle dispatch would let a refused command kill a run.
                            if let Some(why) = self.mode.refused_command(ctype)
                            {
                                let _ = self.out_tx.send(response(cid, ctype, false, None, Some(why)));
                                continue;
                            }
                            // In service mode `fork`/`clone` derive a *new* session and
                            // leave this one running (see `fork_detached`), so there is
                            // nothing here worth a run for: "busy" costs the client a
                            // retry, cancelling would cost it the turn.
                            if let Some(why) = self.mode.busy_refusal_for_fork_clone(ctype, "running") {
                                let _ = self.out_tx.send(response(cid, ctype, false, None, Some(&why)));
                                continue;
                            }
                            match ctype {
                                // Fix 4 (pi-parity gap): the ack is *not* sent here — see
                                // the flush right after this busy-loop exits, below. Sending
                                // it immediately (the instant `cancel.cancel()` is called)
                                // let a client that — correctly, per this RPC's own contract
                                // — treats the ack as "safe to send the next command" get
                                // rejected as busy: this same busy-loop keeps rejecting every
                                // non-abort command as busy until `run` itself actually
                                // resolves, which can lag well behind the cancellation
                                // request while in-flight tool calls/HTTP streams unwind.
                                // Matches pi's `agent-session.ts`, which awaits
                                // `waitForIdle()` before acking `abort`.
                                "abort" => {
                                    cancel.cancel();
                                    // A view's bridge requests are the user's work too.
                                    crate::tools::mcp_apps::cancel_requests(&self.mcp_enabled);
                                    pending_abort_acks.push(cid);
                                }
                                "stop_after_turn" => {
                                    // Graceful, not a cancel: the current turn's tool calls (if
                                    // any) still finish and their results are committed; the run
                                    // just doesn't start another model call afterward. See
                                    // `agent_core::Steering::request_stop`.
                                    self.steering.request_stop();
                                    let _ = self.out_tx.send(response(cid, "stop_after_turn", true, None, None));
                                }
                                "switch_model" => {
                                    // Pi-parity gap (pi's `prepareNextTurn`): unlike `set_model`
                                    // (which only takes effect on the *next* `prompt`), this
                                    // retargets the run already in flight — the current turn's
                                    // own request is unaffected, but every subsequent turn of
                                    // this same run targets the new model. See
                                    // `agent_core::Steering::request_model_switch`.
                                    match c.get("model").and_then(Value::as_str) {
                                        Some(model) => {
                                            let thinking = c
                                                .get("thinking")
                                                .and_then(Value::as_u64)
                                                .map(|t| t as u32);
                                            self.steering.request_model_switch(model, thinking);
                                            let _ = self.out_tx.send(response(cid, "switch_model", true, None, None));
                                        }
                                        None => {
                                            let _ = self.out_tx.send(response(cid, "switch_model", false, None, Some("missing `model`")));
                                        }
                                    }
                                }
                                cmd @ ("steer" | "follow_up") => {
                                    match c.get("message").and_then(Value::as_str) {
                                        Some(m) => {
                                            // Same expansion (and the same optional
                                            // `images` attachments) a fresh `prompt` gets —
                                            // a `/skill:name`/`/name` invocation, or an
                                            // image, steered or queued this way must not
                                            // reach the model unexpanded/dropped just
                                            // because it arrived on a different command
                                            // type.
                                            // `steer` redirects mid-run (injected at the next
                                            // tool turn); `follow_up` waits for the stop
                                            // boundary. Two separate lanes. The pushing
                                            // client learns what's actually queued from its
                                            // own ack (Fix 5), including a full lane's
                                            // refusal. See `PendingSteers` for why an
                                            // MCP `/skill:` is expanded off this loop.
                                            let lane = if cmd == "steer" { SteerLane::Steer } else { SteerLane::FollowUp };
                                            self.pending_steers.receive(cid, m, parse_images(c.get("images")), lane, SteerAck::Command(if cmd == "steer" { "steer" } else { "follow_up" }), &self.skills, &self.prompt_templates, &self.mcp_skills, &self.steering, &self.out_tx);
                                        }
                                        None => {
                                            let _ = self.out_tx.send(response(cid, cmd, false, None, Some("missing `message`")));
                                        }
                                    }
                                }
                                // A `prompt` sent while busy is rejected by default (the
                                // session is borrowed by the in-flight run) — *unless* it
                                // carries `streaming_behavior: "steer"|"follow_up"`, in which
                                // case its `message` is routed through the same `Steering`
                                // queue as an explicit `steer`/`follow_up` command, rather than
                                // forcing the client to re-encode it as a different command type.
                                // An injected event follow-up runs as its own prompt once this run is
                                // over (it is what the coalescer would have sent then anyway).
                                "prompt" if crate::tools::mcp_events::is_injection(&c)
                                    && c.get("streaming_behavior").and_then(Value::as_str) != Some("steer") =>
                                {
                                    self.deferred_events.push_back(l.to_string());
                                }
                                "prompt" => {
                                    match (
                                        c.get("streaming_behavior").and_then(Value::as_str),
                                        c.get("message").and_then(Value::as_str),
                                    ) {
                                        (Some("steer"), Some(m)) => {
                                            // An MCP Events injection is host-authored text, never a
                                            // `/skill:`: queued at once, so its delivery bookkeeping
                                            // (and deferral on a full lane) stays exact.
                                            if let Some(batch) = crate::tools::mcp_events::injection_batch(&c) {
                                                let text = crate::skills::expand_if_skill_invocation(m, &self.skills);
                                                let text = crate::prompts::expand_if_slash(&text, &self.prompt_templates);
                                                let m = agent_core::SteeringMessage::new(text, parse_images(c.get("images")))
                                                    .with_tag(batch);
                                                let queued = self.steering.push_steer(m);
                                                if !queued {
                                                    self.deferred_events.push_back(l.to_string());
                                                    continue;
                                                }
                                                prompt.steered_injections.push(batch);
                                                let mut data = queue_content(&self.steering);
                                                data["queued_as"] = json!("steer");
                                                let _ = self.out_tx.send(response(cid, "prompt", true, Some(data), None));
                                                continue;
                                            }
                                            // Fix 5 (pi-parity gap): same queue-content
                                            // visibility the dedicated `steer`/`follow_up`
                                            // commands' own acks carry, plus `queued_as`.
                                            self.pending_steers.receive(cid, m, parse_images(c.get("images")), SteerLane::Steer, SteerAck::Prompt, &self.skills, &self.prompt_templates, &self.mcp_skills, &self.steering, &self.out_tx);
                                        }
                                        (Some("follow_up"), Some(m)) => {
                                            self.pending_steers.receive(cid, m, parse_images(c.get("images")), SteerLane::FollowUp, SteerAck::Prompt, &self.skills, &self.prompt_templates, &self.mcp_skills, &self.steering, &self.out_tx);
                                        }
                                        _ => {
                                            let _ = self.out_tx.send(response(cid, "prompt", false, None, Some("busy: a prompt is running; only `abort`/`steer`/`follow_up`, or a `prompt` with `streaming_behavior: \"steer\"|\"follow_up\"`, are accepted")));
                                        }
                                    }
                                }
                                // Read-only commands that don't need the `&mut Session` this
                                // run holds — answered live instead of rejected as busy, so a
                                // client polling for progress (tokens/steps so far) during a
                                // long tool-heavy turn isn't left with nothing until it ends.
                                // `get_state`/`get_session_stats` read `live_stats` (mirrored
                                // from this run's own events, see above) instead of `session`
                                // directly; `message_count` is `null` here specifically — it's
                                // the one `get_state` field that genuinely needs `&session` to
                                // report exactly, and a stale/guessed number would be worse
                                // than an honest "not available mid-run".
                                "get_state" => {
                                    // Resolved before the borrow below: in service mode it
                                    // is a round trip into the sandbox, which cannot happen
                                    // while `data` is mutably borrowed.
                                    let branch = self.mode.git_branch(&self.exec_cell, &self.cwd).await;
                                    let frame = get_state_frame(
                                        cid,
                                        live_stats.snapshot(),
                                        Value::Null,
                                        Activity::Running { is_compacting: is_compacting.load(Ordering::Relaxed) },
                                        StateView {
                                            persistence: &self.persistence,
                                            mode: &self.mode,
                                            model: &self.current_model,
                                            cwd: &self.cwd,
                                            branch,
                                            runtime: runtime_settings(self.current_level, self.current_auto_compaction, self.current_auto_retry, self.current_block_images, self.current_image_auto_resize, &self.steering),
                                        },
                                    );
                                    let _ = self.out_tx.send(frame);
                                }
                                // Answered from the fanout's committed-history snapshot — the
                                // same source an attach-time catch-up is seeded from — rather
                                // than from `session`, which this run holds `&mut` for its whole
                                // duration. Rejecting this as busy (as it was) left a client that
                                // re-attached mid-run unable to fetch the history it had missed
                                // for as long as the run lasted, having already been shown that
                                // run's live output: it rendered the tail of a turn whose earlier
                                // messages it had never seen, and could only reconcile once the
                                // run finally ended.
                                //
                                // The snapshot is the transcript as of this run's start (see
                                // `sync_history!` for why it deliberately doesn't advance
                                // mid-run). The turn in flight is carried by the event frames the
                                // client is concurrently receiving — replayed from `OutFanout::
                                // turn` if it attached partway in — so base-plus-frames still
                                // adds up to the whole picture. Its messages land here the moment
                                // the run ends.
                                "get_messages" => {
                                    let (msgs, ids) = {
                                        let f = lock_ignoring_poison(&self.out_conn);
                                        f.history_snapshot()
                                    };
                                    let _ = self.out_tx.send(get_messages_frame(cid, c.get("since").and_then(Value::as_str), &msgs, &ids));
                                }
                                "get_session_stats" => {
                                    // Fix 8 (pi-parity gap): previously the raw
                                    // `LiveStats::snapshot()` verbatim — dropping
                                    // session_id/session_file (this process's own live
                                    // sources, same as `get_state`'s busy arm just above)
                                    // and user_messages/assistant_messages/tool_calls/
                                    // tool_results/total_messages/context_usage (genuinely
                                    // unavailable mid-run: `&session` is exclusively
                                    // borrowed by the in-flight turn, the same reason
                                    // `get_state`'s own `message_count` is `null` here) —
                                    // present as `null` rather than silently absent, so a
                                    // client sees the same field *set* idle or busy, values
                                    // aside.
                                    let _ = self.out_tx.send(get_session_stats_frame(cid, live_stats.snapshot(), Activity::Running { is_compacting: false }, &self.persistence, &self.mode));
                                }
                                "get_commands" => {
                                    let _ = self.out_tx.send(get_commands_frame(cid, &self.skills, &self.mcp_skills, &self.prompt_templates, &self.skill_collisions, &self.prompt_collisions));
                                }
                                "list_branches" => {
                                    let _ = self.out_tx.send(list_branches_frame(cid, &self.persistence));
                                }
                                "get_todos" => {
                                    // From the live mirror: `&session` is exclusively borrowed by
                                    // the in-flight turn. See `LiveStats::todos`.
                                    let _ = self.out_tx.send(response(cid, "get_todos", true, Some(json!({ "todos": live_stats.todos() })), None));
                                }
                                // The arm that matters: an approval question is only ever
                                // outstanding *while a run is blocked inside the gate*, so this
                                // must be answerable mid-run. `accepted:false` tells the losing
                                // client in a multi-attach race that its answer arrived too late.
                                "approve" => {
                                    let _ = self.out_tx.send(handle_approve(cid, &c, &self.pending_approvals, self.approval.is_some()));
                                }
                                "elicit" => {
                                    let _ = self.out_tx.send(handle_elicit(cid, &c, Some(&self.pending_elicitations)));
                                }
                                "sample" => {
                                    let _ = self.out_tx.send(handle_sample(cid, &c, &self.pending_samplings));
                                }
                                // A view is interactive while the model runs, too.
                                "mcp_app_request" => spawn_app_request(cid, c.clone(), &self.out_tx, &self.mcp_enabled, &self.app_filter, &self.approval, &self.app_detached, &self.mcp_host),
                                "get_tree" => {
                                    let _ = self.out_tx.send(get_tree_frame(cid, &c, &self.persistence));
                                }
                                "list_sessions" => {
                                    let frame = list_sessions_frame(cid, &c, &self.persistence, &self.mode, &self.out_tx).await;
                                    let _ = self.out_tx.send(frame);
                                }
                                "list_all_sessions" => {
                                    let frame = list_all_sessions_frame(cid, &c, &self.persistence, &self.mode, &self.out_tx).await;
                                    let _ = self.out_tx.send(frame);
                                }
                                "get_available_models" => {
                                    let _ = self.out_tx.send(get_available_models_frame(cid));
                                }
                                // A run is actively in flight — no retry backoff is pending yet
                                // (that only happens after an attempt fails, outside this loop),
                                // so there's nothing to cancel. Acknowledge idempotently, same
                                // as `abort`'s own idle-mode no-op below.
                                "abort_retry" => {
                                    let _ = self.out_tx.send(response(cid, "abort_retry", true, None, None));
                                }
                                // pi-parity (Task 4): pi's real product has no busy/idle
                                // gating at all for these — `compact()`/`switchSession`/
                                // `newSession`/`fork` all call `abort()` unconditionally
                                // before replacing the session, with no check for whether a
                                // prompt is currently streaming. Mirrored here: cancel this
                                // run exactly like an explicit `abort` (same token), then
                                // queue the raw command to run its ordinary idle-path logic
                                // once the run has actually gone idle — see
                                // `pending_deferred`'s own doc comment above. No ack is sent
                                // here; the command's own idle-path response (sent once
                                // idle, below) doubles as it, same as `abort`'s own deferred
                                // ack just above.
                                "compact" | "switch_session" | "fork" | "clone"
                                | "new_session" => {
                                    cancel.cancel();
                                    prompt.pending_deferred.push((cid, c.clone()));
                                }
                                events_cmd if events_cmd.starts_with("mcp_events_") => dispatch_events(&self.events_cmds, &self.out_tx, cid, events_cmd, &c),
                                other => {
                                    let _ = self.out_tx.send(response(cid, other, false, None, Some("busy: a prompt is running; only `abort`/`abort_retry`/`steer`/`follow_up`, `compact`/`switch_session`/`fork`/`clone`/`new_session` (which self-abort-and-proceed), or a handful of read-only commands (get_state/get_session_stats/get_messages/get_commands/list_branches/get_tree/list_sessions/list_all_sessions/get_available_models), are accepted")));
                                }
                            }
                        }
                        // stdin closed (or errored) mid-run: cancel and let the run unwind, then
                        // we'll fall out of the outer loop below.
                        None => {
                            prompt.stdin_open = false;
                            cancel.cancel();
                        }
                    },
                    _ = async {
                        match heartbeat.as_mut() {
                            Some(h) => {
                                h.tick().await;
                            }
                            None => std::future::pending::<()>().await,
                        }
                    } => {
                        if let Some(life) = &life_beat {
                            life.heartbeat();
                        }
                    }
                }
            }
        };
        // Fix 4: the run has now actually gone idle — `run_events_steered`'s future only
        // resolves once the whole run (including tool cleanup) has stopped touching
        // `session` — so it's safe to tell a client-issued `abort` it can now send its next
        // command without racing this busy-loop's own "still draining" rejection above.
        for cid in pending_abort_acks.drain(..) {
            let _ = self.out_tx.send(response(cid, "abort", true, None, None));
        }
        attempt_result
    }

    /// Waits out a retry delay while the session keeps answering stdin. Returns whether the retry was
    /// cancelled (an `abort`, a command that self-aborts, stdin closing or a shutdown signal).
    async fn retry_backoff(&mut self, prompt: &mut PromptRun, delay: std::time::Duration) -> bool {
        // Race the backoff itself against `abort`/`abort_retry`/shutdown, instead
        // of a bare `sleep` no command can interrupt — no run is in flight during
        // this wait, so only those two commands (plus stdin closing) are accepted;
        // anything else is rejected as busy, the same shape the live-run busy-loop
        // uses. A cancelled retry surfaces the *real* underlying error that
        // triggered it (`attempt_result`, set below), not a synthetic
        // `Error::Cancelled` — nothing was actually cancelled mid-flight, the
        // automatic retry was just declined.
        let mut retry_cancelled = false;
        let sleep = tokio::time::sleep(delay);
        tokio::pin!(sleep);
        loop {
            tokio::select! {
                biased;
                () = &mut sleep => break,
                sig = self.shutdown.wait() => {
                    self.shutdown_cause = Some(sig);
                    prompt.stdin_open = false;
                    retry_cancelled = true;
                    break;
                }
                maybe_line = self.input_rx.recv(), if prompt.stdin_open => match maybe_line {
                    Some(l) => {
                        let l = l.trim();
                        if l.is_empty() {
                            continue;
                        }
                        let c: Value = match serde_json::from_str(l) {
                            Ok(v) => v,
                            Err(e) => {
                                let _ = self.out_tx.send(response(None, "parse", false, None, Some(&format!("invalid JSON: {e}"))));
                                continue;
                            }
                        };
                        let cid = c.get("id").and_then(Value::as_str).map(str::to_string);
                        let cmd_type = c.get("type").and_then(Value::as_str).unwrap_or("");
                        // Same gate as the live-run loop above: a refused command
                        // must not end the retry sequence a run is waiting on.
                        if let Some(why) = self.mode.refused_command(cmd_type)
                        {
                            let _ = self.out_tx.send(response(cid, cmd_type, false, None, Some(why)));
                            continue;
                        }
                        // Same as the live-run loop: in service mode a
                        // `fork`/`clone` has no reason to end the retry
                        // sequence this run is waiting on.
                        if let Some(why) = self.mode.busy_refusal_for_fork_clone(cmd_type, "retrying") {
                            let _ = self.out_tx.send(response(cid, cmd_type, false, None, Some(&why)));
                            continue;
                        }
                        match cmd_type {
                            "abort" | "abort_retry" => {
                                retry_cancelled = true;
                                let _ = self.out_tx.send(response(cid, cmd_type, true, None, None));
                                break;
                            }
                            // pi-parity (Task 4): same self-abort-and-proceed
                            // treatment as the live-run busy-loop's own arm above —
                            // this backoff wait is still "a prompt is running" from
                            // the client's point of view. Ends the retry sequence
                            // (like `abort_retry`) and queues the command to run
                            // once idle, via the same `pending_deferred` drained
                            // right after this whole retry loop.
                            "compact" | "switch_session" | "fork" | "clone"
                            | "new_session" => {
                                retry_cancelled = true;
                                prompt.pending_deferred.push((cid, c.clone()));
                                break;
                            }
                            "prompt" if crate::tools::mcp_events::is_injection(&c) => {
                                self.deferred_events.push_back(l.to_string());
                            }
                            events_cmd if events_cmd.starts_with("mcp_events_") => dispatch_events(&self.events_cmds, &self.out_tx, cid, events_cmd, &c),
                            other => {
                                let _ = self.out_tx.send(response(cid, other, false, None, Some("busy: retrying after a transient error; only `abort`/`abort_retry`/`compact`/`switch_session`/`fork`/`clone`/`new_session` are accepted")));
                            }
                        }
                    }
                    None => {
                        prompt.stdin_open = false;
                        retry_cancelled = true;
                        break;
                    }
                }
            }
        }
        retry_cancelled
    }

    /// The prompt is over, however it ended: stops its yielded jobs, sends the terminal response, names
    /// the session, settles MCP Events batches, and runs what arrived mid-run. Returns whether the session
    /// goes on.
    async fn end_prompt(
        &mut self,
        mut prompt: PromptRun,
        id: Option<String>,
        result: Result<(), agent_core::Error>,
        persist_error: Option<String>,
    ) -> bool {
        let PromptRun {
            life,
            cancel,
            refused,
            ..
        } = &prompt;
        let own_injection = prompt.own_injection;

        // The turn is over, however it ended: its yielded `bash` jobs (a foreground command
        // that went quiet or kept printing) stop with it, as the stop reminder told the model
        // — an idle session must not accumulate hung loops. Intended jobs keep running. Said
        // to the client before the terminal response, like `run`'s `--json` line.
        let stopped_jobs = {
            let jobs = self.bash_jobs.clone();
            tokio::task::spawn_blocking(move || jobs.end_turn())
                .await
                .unwrap_or_default()
        };
        if !stopped_jobs.is_empty()
            && !self.emit(OutFrame::Value(json!({
                "type": "bash_jobs_stopped",
                "jobs": stopped_jobs,
            })))
        {
            return false;
        }

        let frame = match result {
            Ok(()) => {
                let mut data = session_stats(
                    &self.session,
                    &compaction_config(&self.cfg, &self.current_model),
                );
                if let Value::Object(m) = &mut data {
                    m.insert("refused".into(), json!(refused.load(Ordering::Relaxed)));
                    // Present exactly when this prompt asked for typed output, so a client can
                    // tell "the model never produced a payload" (`null`) apart from "this run
                    // wasn't asked for one" (field absent) — the same reason `refused` is always
                    // present rather than only when true. Read after the run has fully drained,
                    // so a mixed batch (`structured_output` + `edit` in one turn, which the
                    // loop's unanimous-terminate rule lets continue) reports the final value.
                    if self.current_output_spec.is_some() {
                        m.insert(
                            "structured_output".into(),
                            self.output_slot.get().unwrap_or(Value::Null),
                        );
                    }
                }
                match &persist_error {
                    // The run itself succeeded, but its transcript failed to durably persist —
                    // report failure rather than a false success (still with `data`: the run
                    // did produce a real result, just not one safely on disk yet).
                    Some(e) => response(
                        id.clone(),
                        "prompt",
                        false,
                        Some(data),
                        Some(&format!("run completed but failed to persist: {e}")),
                    ),
                    None => response(id.clone(), "prompt", true, Some(data), None),
                }
            }
            Err(ref e) => response(id.clone(), "prompt", false, None, Some(&e.to_string())),
        };
        if let Some(life) = &life {
            let summary = crate::lifecycle::closing_summary(&self.session);
            let structured = self
                .current_output_spec
                .is_some()
                .then(|| self.output_slot.get().unwrap_or(Value::Null));
            let refused_now = refused.load(Ordering::Relaxed);
            match &result {
                Ok(()) => match &persist_error {
                    Some(e) => life.failed(
                        self.session.steps,
                        format!("run completed but failed to persist: {e}"),
                        refused_now,
                        summary,
                        structured,
                    ),
                    None => life.succeeded(self.session.steps, refused_now, summary, structured),
                },
                Err(agent_core::Error::Cancelled) => {
                    life.aborted(self.session.steps, summary);
                }
                Err(e) => {
                    life.failed(
                        self.session.steps,
                        e.to_string(),
                        false,
                        summary,
                        structured,
                    );
                }
            }
        }
        if !self.emit(frame) {
            return false;
        }

        // Name the session, once, from the exchange that just happened.
        //
        // **After `emit!(frame)`**, deliberately: this is a second model call, and running
        // it before the response would charge its latency to the client's first turn for a
        // decoration. The `SessionNamed` lifecycle event is its own event rather than a
        // field on the terminal one precisely so it can arrive a moment later without
        // anything depending on the order.
        //
        // Only after a run that succeeded *and* persisted: a title describes what a session
        // turned out to be about, and there is nothing to describe about a first turn that
        // failed. Generated once and never regenerated — a session is about what it was
        // opened for, and re-titling it later would make it change names underneath whoever
        // was looking for it.
        //
        // Best-effort at every step. A session that works without a name is worth more than
        // one that fails because a decoration could not be generated, so a model error, a
        // refusal, or an unusable reply all leave the title exactly as it was: absent.
        // No store means `set_title` is a no-op, so the call could only ever be waste.
        if self.persistence.store.is_some()
            && matches!(result, Ok(()))
            && persist_error.is_none()
            && self.persistence.meta.title.is_none()
            && self.title_attempted_for.as_deref() != Some(self.persistence.meta.id.as_str())
            && !self.session.messages.is_empty()
        {
            self.title_attempted_for = Some(self.persistence.meta.id.clone());
            match self.agent.title_for(&self.session.messages, cancel).await {
                Ok(Some(title)) => {
                    if let Err(e) = self.persistence.set_title(&title) {
                        eprintln!("serve: could not persist the generated session title: {e}");
                    }
                    // Announced whether or not it persisted: the consumer's catalog is a
                    // separate copy, and a title that reached one of the two is better than
                    // a title that reached neither.
                    if let Some(life) = &life {
                        life.named(title);
                    }
                }
                Ok(None) => {}
                Err(e) => eprintln!("serve: could not generate a session title: {e}"),
            }
        }

        // A batch is delivered only once the model has it in a transcript that persisted:
        // the run's own prompt when the run did not fail (an abort still leaves the prompt
        // in the transcript the next run reads); a steered one exactly when the run's
        // `Steered` event reported it (already recorded then — see the `Steered` arm above)
        // — an abort clears the steer lane (`clear_run_scoped`) before the model sees what
        // is queued there. Everything else goes back to pending, to be injected again.
        if let Some(hub) = &self.mcp_events
            && (own_injection.is_some() || !prompt.steered_injections.is_empty())
        {
            let run_ok = persist_error.is_none()
                && matches!(result, Ok(()) | Err(agent_core::Error::Cancelled));
            let (mut delivered, mut returned) = (Vec::new(), Vec::new());
            if let Some(b) = own_injection {
                if run_ok {
                    delivered.push(b)
                } else {
                    returned.push(b)
                }
            }
            // A steered batch the model received has already left the pending queue (see
            // `receipts_sink`); returning it is a no-op. One it never received — an abort
            // cleared the steer lane — goes back to pending.
            returned.append(&mut prompt.steered_injections);
            hub.finish_run(&delivered, &returned).await;
        }
        if self.running.swap(false, Ordering::Relaxed)
            && let Some(m) = &self.cfg.metrics
        {
            m.runs_in_flight.dec();
        }
        // pi-parity (Task 4): this run has now actually gone idle (same guarantee
        // `pending_abort_acks` relies on above) and its own terminal response has just been
        // sent — run each command that arrived mid-run and self-aborted-and-proceeded through
        // its ordinary idle-path logic now, in the order received. Exactly what a client
        // manually sending `abort`, waiting for the response, then retrying the command would
        // have produced, minus the extra round trip.
        for (dcid, dcmd) in prompt.pending_deferred.drain(..) {
            match dcmd.get("type").and_then(Value::as_str).unwrap_or("") {
                "new_session" => self.cmd_new_session(&dcmd, dcid).await,
                "switch_session" => self.cmd_switch_session(&dcmd, dcid).await,
                "fork" => self.cmd_fork(&dcmd, dcid).await,
                "clone" => self.cmd_clone(dcid).await,
                "compact" => {
                    if !self.cmd_compact(&dcmd, dcid).await {
                        prompt.stdin_open = false;
                    }
                }
                _ => {}
            }
        }
        if !prompt.stdin_open {
            return false;
        }
        true
    }
}
