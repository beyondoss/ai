//! The control loop: one command per pass until stdin closes, a shutdown signal arrives or the writer goes
//! away. `run` reads a line (`next_idle_line`), parses it, refuses what the mode refuses and hands the
//! rest to `dispatch`, the idle command table; `idle_housekeeping` is the work every pass does before it
//! reads. A running `prompt` or `compact` has its own loop and does not come back here until it is over.

use crate::serve::*;

/// What `next_idle_line` found.
enum IdleLine {
    /// A command line, from the client or synthesised (a deferred MCP event, a woken `bash` job).
    Line(String),
    /// Something was handled that is not a command (a journal write); read again.
    Skip,
    /// The session ends: shutdown, stdin closed, or the writer went away.
    End,
}

impl SessionLoop {
    /// The control loop: one command per pass, until stdin closes, a shutdown signal arrives or the
    /// writer goes away.
    pub(super) async fn run(&mut self) {
        loop {
            if !self.idle_housekeeping().await {
                break;
            }
            let line = match self.next_idle_line().await {
                IdleLine::Line(line) => line,
                IdleLine::Skip => continue,
                IdleLine::End => break,
            };
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let cmd: Value = match serde_json::from_str(line) {
                Ok(v) => v,
                Err(e) => {
                    if !self.emit(response(
                        None,
                        "parse",
                        false,
                        None,
                        Some(&format!("invalid JSON: {e}")),
                    )) {
                        break;
                    }
                    continue;
                }
            };
            let id = cmd.get("id").and_then(Value::as_str).map(str::to_string);
            let ctype = cmd
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();

            // Service mode refuses a handful of commands outright (see `service::refused_command`).
            // Checked here, ahead of the dispatch — and again at each busy-loop intercept below, so a
            // command that is going to be refused never gets there by cancelling a running prompt first.
            if let Some(why) = self.mode.refused_command(&ctype) {
                if !self.emit(response(id, &ctype, false, None, Some(why))) {
                    break;
                }
                continue;
            }
            if !self.dispatch(ctype.as_str(), &cmd, id).await {
                break;
            }
        }
    }

    /// What every pass does before it reads a command. `false` ends the session: the writer went away,
    /// or another owner fenced this one.
    async fn idle_housekeeping(&mut self) -> bool {
        // Idle: the transcript is whatever the last command left behind (including a run that just
        // finished). Publish it for the next connection to attach — see `sync_history!`. Also seeds the
        // very first snapshot, from the history `Persistence::open` restored off disk, so a client
        // attaching to a resumed session is caught up before it has issued a single command.
        self.sync_history();
        // Every report a run made is persisted by here: the sandbox directories of the jobs whose
        // end it reported can go. A replica that died before this point leaves them for the next
        // one to report again — once, since that one's report is persisted before it deletes them.
        self.bash_jobs.collect_garbage().await;
        // A command just switched sessions (`switch_session`, `new_session`, `fork`, `clone`): the
        // old session's resumer stops (it must not emit into this session's clients, and never
        // cancels its tasks, which stay journaled) and the new session's pending tasks resume now,
        // not on its first prompt.
        if self.mcp_resumer.session_id() != self.persistence.session_id() {
            self.mcp_resumer = self.start_mcp_resumer(&[]);
        }
        // The transcript changed (`new_session`, `switch_session`, `fork`, …): the jobs the old one
        // started belong to its history, not the new one's, so they stop here rather than wake the
        // new transcript with notices about commands it never ran. Keyed on the transcript, not the
        // session id: a pinned session (every daemon session) and a `--session-file` session keep
        // their id across `new_session`.
        //
        // Off the runtime thread, as at the end of a turn and of the session: each kill blocks until
        // its processes are gone, and a remote job's kill waits for its stream's future — which runs
        // on this runtime — to let the connection go, so stopping it here would stall that future
        // (and every other session on a daemon's current-thread runtime) for the kill's full bound.
        // Said to the client, like every other stop.
        if self.bash_jobs_transcript != self.persistence.transcript() {
            self.bash_jobs_transcript = self.persistence.transcript();
            let stopped_jobs = {
                let jobs = self.bash_jobs.clone();
                crate::blocking::spawn(crate::blocking::Site::Serve, move || jobs.stop_all())
                    .await
                    .unwrap_or_default()
            };
            // v1.4's state of the old transcript's jobs goes with them; a reconcile that began on
            // it hands nothing over.
            if let Some(n) = &self.job_notify {
                n.feed(crate::tools::job_notify::kernel::Event::TranscriptSwitched);
            }
            // The old transcript is ended, so nothing of it may live on in the sandbox — not even
            // a job this table no longer knew. The new one may have jobs a replica left running.
            if let Some(old) = self.bash_jobs.scope() {
                crate::tools::bash_sandbox::wipe(self.exec_cell.runner(), &old).await;
            }
            self.bash_jobs
                .set_scope(Some(crate::exec_endpoint::jobs::JobScope::for_session(
                    self.mode.tenant(),
                    self.persistence.session_id(),
                )));
            if self.exec_cell.pending() {
                crate::tools::bash_sandbox::reattach_later(
                    &self.bash_jobs,
                    self.exec_cell.runner(),
                );
            } else {
                crate::tools::bash_sandbox::reattach(&self.bash_jobs, self.exec_cell.runner())
                    .await;
            }
            stop_survivors(&self.bash_jobs, &self.persistence).await;
            // The snapshot belongs beside the transcript the session is on now.
            self.sandbox_snapshot_dirty = self.mode.keeps_sandbox_snapshot();
            self.bash_jobs.set_recorder(Some(procs_recorder(
                &self.mcp_task_tx,
                self.persistence.session_id(),
            )));
            if !stopped_jobs.is_empty()
                && !self.emit(OutFrame::Value(json!({
                    "type": "bash_jobs_stopped",
                    "jobs": stopped_jobs,
                })))
            {
                return false;
            }
        }
        // Once the sandbox is awake — something used it — what a reopen deferred is due: the jobs
        // it kept are read back, and discovery runs again. A changed set rebuilds the prompt (the
        // one cache miss a change costs); an unchanged one leaves it byte for byte as it was. Never
        // before: a sandbox nobody has used is not woken for either.
        if !self.exec_cell.pending() {
            if self.bash_jobs.reattach_pending()
                && let Err(e) = crate::tools::bash_sandbox::reconcile(&self.bash_jobs).await
            {
                tracing::warn!(error = %e, "could not read back the session's sandbox jobs");
            }
            if self.sandbox_refresh
                && let Some(found) = self
                    .mode
                    .discover_sandbox_resources(&self.exec_cell, self.cfg.context_files)
                    .await
            {
                self.sandbox_refresh = false;
                if self.sandbox_resources.as_ref() != Some(&found) {
                    self.sandbox_resources = Some(found.clone());
                    self.apply_discovered(Discovered::from(found)).await;
                }
                // Re-saved either way: the probe that woke the sandbox may have answered differently.
                self.sandbox_snapshot_dirty = true;
            }
        }
        if self.sandbox_snapshot_dirty
            && let (Some(resources), Some(file)) =
                (&self.sandbox_resources, self.persistence.session_file())
            && let Some(saved) = self.mode.save_sandbox_snapshot(
                file,
                self.persistence.session_id(),
                self.cfg.context_files,
                resources,
            )
        {
            match saved {
                Ok(()) => self.sandbox_snapshot_dirty = false,
                // Its directory does not exist yet (a session with nothing on disk): next time.
                Err(e) => tracing::debug!(error = %e, "sandbox snapshot not saved yet"),
            }
        }
        // Checked here because every write path — a run's persist, a mid-run checkpoint, a title, a
        // label, a compaction rewrite — comes back through this loop before the next command is read.
        // Another owner has created a newer epoch on this session's storage, so nothing this process
        // writes can land again: end the session with an event that tells the client to reconnect
        // (which respawns against the new epoch) rather than silently accepting turns that are lost.
        if self.persistence.superseded() {
            // Rare by design: an owner discovering it was fenced. A rising rate means sessions are
            // changing hands when nothing asked them to.
            if let Some(m) = &self.cfg.metrics {
                m.sessions_superseded.inc();
            }
            self.emit(OutFrame::Value(json!({
                "type": "session_superseded",
                "session_id": self.persistence.session_id(),
                "tenant": self.mode.tenant(),
            })));
            return false;
        }
        // The events state follows the active transcript (`new_session`, `switch_session`, …) —
        // moved and merged before the next command is read, so nothing that command (or an
        // injection behind it) does can land in the old one.
        if let Some(hub) = &self.mcp_events {
            let file = self
                .persistence
                .session_file()
                .map(std::path::Path::to_path_buf);
            if file != self.events_state_file {
                hub.relocate(file.as_deref()).await;
                self.events_state_file = file;
            }
        }
        true
    }

    /// The next command line while idle: a deferred MCP event first, else whichever of shutdown, stdin, a
    /// resumed MCP task's journal write or a finished `bash` job comes first.
    async fn next_idle_line(&mut self) -> IdleLine {
        if let Some(deferred) = self.deferred_events.pop_front() {
            return IdleLine::Line(deferred);
        }
        tokio::select! {
            biased;
            // Idle between commands: nothing is in flight, so a shutdown request needs no drain —
            // just stop reading and fall out to the writer join below.
            sig = self.shutdown.wait() => {
                self.shutdown_cause = Some(sig);
                IdleLine::End
            }
            // A closed `input_rx` (the stdio reader hit EOF and dropped its sender, or a WebSocket
            // supervisor deliberately dropped this session's retained sender to tear it down) ends the
            // session gracefully — the same clean-shutdown path stdin EOF always drove. A merely
            // *detached* WebSocket connection does NOT close `input_rx` (the supervisor keeps the
            // sender), so this arm simply pends across a reconnect rather than firing.
            maybe_line = self.input_rx.recv() => match maybe_line {
                Some(l) => IdleLine::Line(l),
                None => IdleLine::End,
            },
            // A resumed MCP task's progress record or result, landing while idle.
            Some((kind, data)) = self.mcp_task_rx.recv() => {
                journal_mcp(&mut self.persistence, &kind, data);
                IdleLine::Skip
            }
            // A `bash` job ended while nothing was running: the session wakes, and the notice
            // is the input of an ordinary `prompt` turn (persisted, streamed, retried like any
            // other). Last in this biased select, so a client's own command that is already
            // waiting goes first — its run then delivers the notice itself, at its first tool
            // results or its stop, and this arm finds nothing left to take. Several jobs ending
            // together fold into one turn (`bash_job_notice_due`'s settle).
            () = bash_job_notice_due(&self.bash_jobs, &mut self.bash_jobs_wake) => {
                let Some(text) = agent_core::Background::take_notices(self.bash_jobs.as_ref()) else {
                    return IdleLine::Skip;
                };
                if let Some(frame) = event_frame(AgentEvent::Background {
                    reason: agent_core::BackgroundReason::Finished,
                    text: text.clone(),
                    wake: true,
                })
                    && !self.emit(frame) { return IdleLine::End; }
                IdleLine::Line(json!({ "type": "prompt", "message": text, "bash_jobs_wake": true }).to_string())
            }
        }
    }

    /// The idle command table. `false` ends the session (the writer went away).
    async fn dispatch(&mut self, ctype: &str, cmd: &Value, id: Option<String>) -> bool {
        match ctype {
            "prompt" => {
                if !self.cmd_prompt(cmd, id).await {
                    return false;
                }
            }
            // Reachable while idle (a mid-run `steer`/`follow_up` is handled inside the `prompt` arm's
            // own busy-loop above, which shares this same persistent `steering` handle) — queues for
            // whichever `prompt` runs next, rather than being rejected as an unknown command just
            // because nothing is in flight *yet*.
            cmd_type @ ("steer" | "follow_up") => {
                match cmd.get("message").and_then(Value::as_str) {
                    Some(m) => {
                        let m = expand_message(
                            m,
                            &self.skills,
                            &self.prompt_templates,
                            &self.mcp_skills,
                        )
                        .await;
                        let m =
                            agent_core::SteeringMessage::new(m, parse_images(cmd.get("images")));
                        let queued = if cmd_type == "steer" {
                            self.steering.push_steer(m)
                        } else {
                            self.steering.push(m)
                        };
                        // Fix 5 (pi-parity gap): see `queue_content`'s own doc comment. A lane at its
                        // cap refuses the message; the ack reports that rather than claiming it queued.
                        if !self.emit(response(
                            id,
                            cmd_type,
                            queued,
                            Some(queue_content(&self.steering)),
                            (!queued).then_some(STEERING_QUEUE_FULL),
                        )) {
                            return false;
                        }
                    }
                    None => {
                        if !self.emit(response(
                            id,
                            cmd_type,
                            false,
                            None,
                            Some("missing `message`"),
                        )) {
                            return false;
                        }
                    }
                }
            }
            "abort" => {
                // No run is in flight (a mid-run abort is handled inside the `prompt` arm above), so
                // the only work to cancel is a view's bridge requests — acknowledge idempotently.
                crate::tools::mcp_apps::cancel_requests(&self.mcp_enabled);
                if !self.emit(response(id, "abort", true, None, None)) {
                    return false;
                }
            }
            "abort_retry" => {
                // Same idea as `abort` above: interrupting a pending whole-run-retry backoff only
                // means something while a `prompt` is between attempts (handled inside that arm's own
                // retry loop) — idle, there's no backoff in progress to cancel.
                if !self.emit(response(id, "abort_retry", true, None, None)) {
                    return false;
                }
            }
            "stop_after_turn" => {
                // No run is in flight, so there is no turn boundary to stop at. Unlike `steer`/
                // `follow_up`, deliberately *not* forwarded to `steering.request_stop()` here: that
                // would silently cut the *next* `prompt` off after its first turn, surprising a client
                // that sent this expecting to affect a run that no longer exists. Acknowledge as a
                // no-op instead, matching `abort`'s idle behavior.
                if !self.emit(response(id, "stop_after_turn", true, None, None)) {
                    return false;
                }
            }
            "switch_model" => {
                // No run is in flight, so there is no next turn to retarget — changing the model while
                // idle is `set_model`'s job (it takes effect immediately, on the very next `prompt`).
                // Acknowledge as a no-op rather than silently queuing a switch that would surprise a
                // later, unrelated run, matching `stop_after_turn`'s idle behavior.
                if !self.emit(response(
                    id,
                    "switch_model",
                    true,
                    Some(json!({ "note": "no run in flight; use set_model instead" })),
                    None,
                )) {
                    return false;
                }
            }
            "set_exec_endpoint" => {
                if !self.cmd_set_exec_endpoint(cmd, id).await {
                    return false;
                }
            }
            "set_mcp_enabled" => {
                if !self.cmd_set_mcp_enabled(cmd, id) {
                    return false;
                }
            }
            "get_mcp" => {
                if !self.cmd_get_mcp(id) {
                    return false;
                }
            }
            "mcp_complete" => {
                if !self.cmd_mcp_complete(cmd, id).await {
                    return false;
                }
            }
            "elicit" => {
                if !self.emit(handle_elicit(id, cmd, Some(&self.pending_elicitations))) {
                    return false;
                }
            }
            "sample" => {
                if !self.emit(handle_sample(id, cmd, &self.pending_samplings)) {
                    return false;
                }
            }
            "mcp_app_request" => spawn_app_request(
                id,
                cmd.clone(),
                &self.out_tx,
                &self.mcp_enabled,
                &self.app_filter,
                &self.approval,
                &self.app_detached,
                &self.mcp_host,
            ),
            "set_mcp_apps" => {
                if !self.cmd_set_mcp_apps(cmd, id).await {
                    return false;
                }
            }
            // Test-only, in debug builds and only with `BEYOND_AI_AGENT_TEST_PANICS` set: panic right
            // here, inside the session loop, so the way a panic leaves `serve` can be tested.
            #[cfg(debug_assertions)]
            "__test_panic" if std::env::var_os("BEYOND_AI_AGENT_TEST_PANICS").is_some() => {
                #[allow(clippy::panic)]
                {
                    panic!("test panic requested inside serve");
                }
            }
            "get_state" => {
                if !self.cmd_get_state(id).await {
                    return false;
                }
            }
            "get_messages" => {
                if !self.cmd_get_messages(cmd, id) {
                    return false;
                }
            }
            "new_session" => self.cmd_new_session(cmd, id).await,
            "list_sessions" => {
                if !self.cmd_list_sessions(cmd, id).await {
                    return false;
                }
            }
            "list_all_sessions" => {
                if !self.cmd_list_all_sessions(cmd, id).await {
                    return false;
                }
            }
            "switch_session" => self.cmd_switch_session(cmd, id).await,
            "delete_session" => {
                if !self.cmd_delete_session(cmd, id).await {
                    return false;
                }
            }
            "list_trash" => {
                if !self.cmd_list_trash(id) {
                    return false;
                }
            }
            "restore_session" => {
                if !self.cmd_restore_session(cmd, id) {
                    return false;
                }
            }
            "fork" => self.cmd_fork(cmd, id).await,
            // pi's own `clone` — fork the current session at its current tip, with no arguments —
            // exists there because pi's `fork` *requires* an explicit `entryId`; this crate's `fork`
            // already defaults to exactly that (no `upto`/`target_id` given), so `clone` is a thin,
            // deliberately-argument-free alias over the same call for a client speaking pi's protocol
            // shape, not a second code path.
            "clone" => self.cmd_clone(id).await,
            "get_fork_messages" => {
                if !self.cmd_get_fork_messages(id) {
                    return false;
                }
            }
            "preview_fork" => {
                if !self.cmd_preview_fork(cmd, id) {
                    return false;
                }
            }
            "export_html" => {
                if !self.cmd_export_html(cmd, id) {
                    return false;
                }
            }
            "set_session_name" => {
                if !self.cmd_set_session_name(cmd, id) {
                    return false;
                }
            }
            "set_label" => {
                if !self.cmd_set_label(cmd, id) {
                    return false;
                }
            }
            "get_label" => {
                if !self.cmd_get_label(cmd, id) {
                    return false;
                }
            }
            "append_custom" => {
                if !self.cmd_append_custom(cmd, id) {
                    return false;
                }
            }
            "compact" => {
                if !self.cmd_compact(cmd, id).await {
                    return false;
                }
            }
            "get_last_assistant_text" => {
                if !self.cmd_get_last_assistant_text(id) {
                    return false;
                }
            }
            "approve" => {
                if !self.cmd_approve(cmd, id) {
                    return false;
                }
            }
            "get_todos" => {
                if !self.cmd_get_todos(id) {
                    return false;
                }
            }
            "get_session_stats" => {
                if !self.cmd_get_session_stats(id) {
                    return false;
                }
            }
            "get_commands" => {
                if !self.cmd_get_commands(id) {
                    return false;
                }
            }
            "reload" => {
                if !self.cmd_reload(id).await {
                    return false;
                }
            }
            "set_model" => {
                if !self.cmd_set_model(cmd, id) {
                    return false;
                }
            }
            "set_thinking" => {
                if !self.cmd_set_thinking(cmd, id) {
                    return false;
                }
            }
            "set_reasoning_effort" => {
                if !self.cmd_set_reasoning_effort(cmd, id) {
                    return false;
                }
            }
            "cycle_model" => {
                if !self.cmd_cycle_model(id) {
                    return false;
                }
            }
            "cycle_thinking_level" => {
                if !self.cmd_cycle_thinking_level(id) {
                    return false;
                }
            }
            "set_auto_compaction" => {
                if !self.cmd_set_auto_compaction(cmd, id) {
                    return false;
                }
            }
            "set_auto_retry" => {
                if !self.cmd_set_auto_retry(cmd, id) {
                    return false;
                }
            }
            "set_block_images" => {
                if !self.cmd_set_block_images(cmd, id) {
                    return false;
                }
            }
            "set_image_auto_resize" => {
                if !self.cmd_set_image_auto_resize(cmd, id) {
                    return false;
                }
            }
            "set_steering_mode" => {
                if !self.cmd_set_steering_mode(cmd, id) {
                    return false;
                }
            }
            "set_follow_up_mode" => {
                if !self.cmd_set_follow_up_mode(cmd, id) {
                    return false;
                }
            }
            "get_available_models" => {
                if !self.cmd_get_available_models(id) {
                    return false;
                }
            }
            "list_branches" => {
                if !self.cmd_list_branches(id) {
                    return false;
                }
            }
            "get_tree" => {
                if !self.cmd_get_tree(cmd, id) {
                    return false;
                }
            }
            "switch_branch" => {
                if !self.cmd_switch_branch(cmd, id).await {
                    return false;
                }
            }
            "bash" => {
                if !self.cmd_bash(cmd, id).await {
                    return false;
                }
            }
            "abort_bash" => {
                if !self.cmd_abort_bash(id) {
                    return false;
                }
            }
            "login" => {
                if !self.cmd_login(cmd, id).await {
                    return false;
                }
            }
            "submit_code" => {
                if !self.cmd_submit_code(cmd, id) {
                    return false;
                }
            }
            "abort_login" => {
                if !self.cmd_abort_login(id) {
                    return false;
                }
            }
            "logout" => {
                if !self.cmd_logout(cmd, id) {
                    return false;
                }
            }
            "auth_status" => {
                if !self.cmd_auth_status(cmd, id) {
                    return false;
                }
            }
            events_cmd if events_cmd.starts_with("mcp_events_") => {
                dispatch_events(&self.events_cmds, &self.out_tx, id, events_cmd, cmd)
            }
            other => {
                if !self.emit(response(id, other, false, None, Some("unknown command"))) {
                    return false;
                }
            }
        }
        true
    }
}
