use crate::serve::*;

/// The `list_sessions` response. The scan's progress frames go straight to `out_tx` (they are
/// events, not the response), ahead of the returned frame.
pub(in crate::serve) async fn list_sessions_frame(
    id: Option<String>,
    cmd: &Value,
    persistence: &Persistence,
    mode: &Mode,
    out_tx: &OutTx,
) -> OutFrame {
    let progress_id = id.clone();
    let progress_tx = out_tx.clone();
    let sessions = persistence
        .list_with_progress(mode, move |scanned, total| {
            if should_report_scan_progress(scanned, total) {
                let _ = progress_tx.send(list_progress_frame(
                    progress_id.clone(),
                    "list_sessions",
                    scanned,
                    total,
                ));
            }
        })
        .await;
    response(
        id,
        "list_sessions",
        true,
        Some(listing_page(sessions, cmd)),
        None,
    )
}

/// The `list_all_sessions` response; see [`list_sessions_frame`].
pub(in crate::serve) async fn list_all_sessions_frame(
    id: Option<String>,
    cmd: &Value,
    persistence: &Persistence,
    mode: &Mode,
    out_tx: &OutTx,
) -> OutFrame {
    let progress_id = id.clone();
    let progress_tx = out_tx.clone();
    match persistence
        .list_all_with_progress(mode, move |scanned, total| {
            if should_report_scan_progress(scanned, total) {
                let _ = progress_tx.send(list_progress_frame(
                    progress_id.clone(),
                    "list_all_sessions",
                    scanned,
                    total,
                ));
            }
        })
        .await
    {
        Ok(sessions) => response(
            id,
            "list_all_sessions",
            true,
            Some(listing_page(sessions, cmd)),
            None,
        ),
        Err(e) => response(id, "list_all_sessions", false, None, Some(&e.to_string())),
    }
}

impl SessionLoop {
    pub(in crate::serve) async fn cmd_list_sessions(
        &mut self,
        cmd: &Value,
        id: Option<String>,
    ) -> bool {
        let frame = list_sessions_frame(id, cmd, &self.persistence, &self.mode, &self.out_tx).await;
        self.emit(frame)
    }

    pub(in crate::serve) async fn cmd_list_all_sessions(
        &mut self,
        cmd: &Value,
        id: Option<String>,
    ) -> bool {
        let frame =
            list_all_sessions_frame(id, cmd, &self.persistence, &self.mode, &self.out_tx).await;
        self.emit(frame)
    }

    // Soft-deletes (moves to `.trash`) another session — never the one currently active, see
    // `Persistence::delete`'s doc comment. Idempotent: deleting an absent (or already-deleted)
    // session id is a successful no-op.
    pub(in crate::serve) async fn cmd_delete_session(
        &mut self,
        cmd: &Value,
        id: Option<String>,
    ) -> bool {
        match cmd.get("session_id").and_then(Value::as_str) {
            Some(target) => match self.persistence.delete(&self.mode, target) {
                Ok(()) => {
                    // A deleted session is ended: its jobs in this sandbox stop. (In another
                    // sandbox they end with it — tearing that down is its owner's business.)
                    crate::tools::bash_sandbox::wipe(
                        self.exec_cell.runner(),
                        &crate::exec_endpoint::jobs::JobScope::for_session(
                            self.mode.tenant(),
                            target,
                        ),
                    )
                    .await;
                    if !self.emit(response(id, "delete_session", true, None, None)) {
                        return false;
                    }
                }
                Err(e) => {
                    if !self.emit(response(
                        id,
                        "delete_session",
                        false,
                        None,
                        Some(&e.to_string()),
                    )) {
                        return false;
                    }
                }
            },
            None => {
                if !self.emit(response(
                    id,
                    "delete_session",
                    false,
                    None,
                    Some("missing `session_id`"),
                )) {
                    return false;
                }
            }
        }
        true
    }

    // Minimal surface (Fix 7, pi-parity gap) over `delete_session`'s own `.trash/` soft-delete —
    // nothing previously read, listed, restored from, or pruned it, so a mistaken delete was
    // recoverable only by reaching for a shell. `list_trash` reports basic metadata
    // (id/deleted_at/original_path — see `session_store::TrashEntry`); `restore_session` moves an
    // entry back out. Deliberately not a full trash-management UI (no bulk purge, no age-based
    // pruning) — a low-priority nice-to-have, not a core session-lifecycle feature.
    pub(in crate::serve) fn cmd_list_trash(&mut self, id: Option<String>) -> bool {
        match self.persistence.list_trash(&self.mode) {
            Ok(mut trash) => {
                // `original_path` is a replica mount path. The id is what `restore_session`
                // takes, so the path was only ever informational — redacted rather than dropped
                // so the shape of the response doesn't change per mode.
                if self.mode.redacts_host_paths() {
                    for entry in &mut trash {
                        entry.original_path = self.mode.redact(&entry.original_path);
                    }
                }
                if !self.emit(response(
                    id,
                    "list_trash",
                    true,
                    Some(json!({ "trash": trash })),
                    None,
                )) {
                    return false;
                }
            }
            Err(e) => {
                if !self.emit(response(
                    id,
                    "list_trash",
                    false,
                    None,
                    Some(&e.to_string()),
                )) {
                    return false;
                }
            }
        }
        true
    }

    pub(in crate::serve) fn cmd_restore_session(
        &mut self,
        cmd: &Value,
        id: Option<String>,
    ) -> bool {
        match cmd.get("session_id").and_then(Value::as_str) {
            Some(target) => match self.persistence.restore_session(&self.mode, target) {
                Ok(true) => {
                    if !self.emit(response(id, "restore_session", true, None, None)) {
                        return false;
                    }
                }
                Ok(false) => {
                    if !self.emit(response(
                        id,
                        "restore_session",
                        false,
                        None,
                        Some(&format!("no trashed session {target}")),
                    )) {
                        return false;
                    }
                }
                Err(e) => {
                    if !self.emit(response(
                        id,
                        "restore_session",
                        false,
                        None,
                        Some(&e.to_string()),
                    )) {
                        return false;
                    }
                }
            },
            None => {
                if !self.emit(response(
                    id,
                    "restore_session",
                    false,
                    None,
                    Some("missing `session_id`"),
                )) {
                    return false;
                }
            }
        }
        true
    }

    pub(in crate::serve) fn cmd_get_fork_messages(&mut self, id: Option<String>) -> bool {
        // pi-compatible contract: no parameters, every user-turn entry across the WHOLE session
        // tree — every branch, not just the active path — as a flat `{entry_id, text}` candidate
        // list (pi's own `getUserMessagesForForking`), for a client to build a fork-point picker
        // from and then feed one `entry_id` to `fork`'s own `target_id`/`fork_at_entry`. This is
        // a listing, not a preview of any one fork's output — see `preview_fork` for that.
        // Track (pi-parity fix): previously scoped to `persistence.active_ids()` only, so a
        // message on a branch the session had already navigated away from (via `switch_branch`)
        // never appeared as a fork candidate at all — empty (not an error) in pure in-memory mode,
        // same as every other persistence-dependent listing here.
        let candidates: Vec<Value> = self
            .persistence
            .all_user_messages()
            .iter()
            .filter_map(|(entry_id, m)| {
                user_message_text(m).map(|text| json!({ "entry_id": entry_id, "text": text }))
            })
            .collect();
        if !self.emit(response(
            id,
            "get_fork_messages",
            true,
            Some(json!({ "messages": candidates })),
            None,
        )) {
            return false;
        }
        true
    }

    pub(in crate::serve) fn cmd_preview_fork(&mut self, cmd: &Value, id: Option<String>) -> bool {
        // A read-only preview of what `fork` would produce for `session_id` (default: the
        // current session) at `upto` messages — no new session file, no switch. Lets a client
        // browsing `list_sessions` show a fork point before committing to it. This crate's own
        // extension beyond pi's protocol (previously misnamed `get_fork_messages`, colliding
        // with pi's own same-named but differently-shaped command — see the module doc comment).
        let target_id = cmd
            .get("session_id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| self.persistence.session_id().to_string());
        let upto = cmd
            .get("upto")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(usize::MAX);
        let entry_id = cmd.get("target_id").and_then(Value::as_str);
        // Same default as `fork` itself, above — a preview must match what `fork` would
        // actually produce.
        let before = cmd.get("before").and_then(Value::as_bool).unwrap_or(true);
        match self
            .persistence
            .fork_messages(&self.mode, &target_id, upto, entry_id, before)
        {
            Ok(messages) => {
                if !self.emit(response(
                    id,
                    "preview_fork",
                    true,
                    Some(json!({ "messages": messages })),
                    None,
                )) {
                    return false;
                }
            }
            Err(e) => {
                if !self.emit(response(
                    id,
                    "preview_fork",
                    false,
                    None,
                    Some(&e.to_string()),
                )) {
                    return false;
                }
            }
        }
        true
    }

    pub(in crate::serve) fn cmd_set_session_name(
        &mut self,
        cmd: &Value,
        id: Option<String>,
    ) -> bool {
        match cmd.get("title").and_then(Value::as_str) {
            Some(title) => match self.persistence.set_title(title) {
                Ok(()) => {
                    // pi: `session_info_changed` (`rpc-mode.ts:632-639`) — a push notification so a
                    // client learns the final *sanitized* name (newlines collapsed, or cleared
                    // entirely if the sanitized result is empty — see `sanitize_title`) without a
                    // second `get_state` round trip. Sent as its own unsolicited frame (like
                    // `list_progress`/`auto_retry` above) *and* echoed on the response's own `data`,
                    // since unlike pi's multi-subscriber extension model this protocol has exactly
                    // one client and the response is already the more direct notification path.
                    let title = self.persistence.meta.title.clone();
                    let _ = self
                        .out_tx
                        .send(session_info_changed_frame(id.clone(), title.clone()));
                    if !self.emit(response(
                        id,
                        "set_session_name",
                        true,
                        Some(json!({ "title": title })),
                        None,
                    )) {
                        return false;
                    }
                }
                Err(e) => {
                    if !self.emit(response(
                        id,
                        "set_session_name",
                        false,
                        None,
                        Some(&e.to_string()),
                    )) {
                        return false;
                    }
                }
            },
            None => {
                if !self.emit(response(
                    id,
                    "set_session_name",
                    false,
                    None,
                    Some("missing `title`"),
                )) {
                    return false;
                }
            }
        }
        true
    }

    /// pi-parity (Task 4, serve pass 19): the ordinary idle-path logic for `new_session`, shared with
    /// `switch_session`/`fork`/`clone`/`compact` below. Each is called from two places: its ordinary
    /// command-match arm, and the `"prompt"` arm's deferred-until-idle path (`pending_deferred`)
    /// once a run one of these commands interrupted has actually gone idle — see the module's own
    /// `serve` doc comment / Task 4 for why these five self-abort-and-proceed instead of being
    /// rejected as busy. `cmd`/`id` are whichever command `Value`/response-id are in play: the
    /// arm's own when idle, or a deferred command's stored `Value`/id once its abort has resolved.
    ///
    /// Responses are sent via a bare `out_tx.send` rather than `emit`, and the method returns
    /// nothing: a dead writer is not worth ending the short-lived deferred drain for, and the idle
    /// loop finds out on its next read.
    pub(in crate::serve) async fn cmd_new_session(&mut self, cmd: &Value, id: Option<String>) {
        match self.persistence.new_session(
            &self.current_model,
            cmd.get("parent_session").and_then(Value::as_str),
        ) {
            Ok(s) => {
                self.session = s;
                self.repoint_session_memory();
                self.reset_exec_endpoint().await;
                self.reset_mcp_enabled();
                self.reset_approval_memory();
                self.steering.clear();
                // Fresh session defaults to all MCP servers enabled — rebuild so a previous
                // session's allowlist cannot linger on this process's Agent.
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
                let _ = self.out_tx.send(response(
                    id,
                    "new_session",
                    true,
                    Some(json!({
                        "session_id": self.persistence.session_id(),
                        "cwd_stale": self.mode.cwd_is_stale(&self.persistence.meta.cwd, &self.cwd),
                        "parent": self.persistence.meta.parent,
                    })),
                    None,
                ));
            }
            Err(e) => {
                let _ = self.out_tx.send(response(
                    id,
                    "new_session",
                    false,
                    None,
                    Some(&e.to_string()),
                ));
            }
        }
    }

    pub(in crate::serve) async fn cmd_switch_session(&mut self, cmd: &Value, id: Option<String>) {
        match cmd.get("session_id").and_then(Value::as_str) {
            Some(target) => match self.persistence.switch(target) {
                Ok(s) => {
                    self.session = s;
                    close_interrupted_prompt(&mut self.session);
                    self.repoint_session_memory();
                    self.reset_exec_endpoint().await;
                    self.reset_mcp_enabled();
                    self.reset_approval_memory();
                    self.steering.clear();
                    // Restore whichever model/thinking-level this session was actually last
                    // running on, the same way `switch_branch` already does — without this, the
                    // process's current global model/level (possibly set on a *different* session
                    // entirely) silently bled into the reattached session. See
                    // `model_and_level_at_active`'s doc comment.
                    let (restored_model, restored_level) = self
                        .persistence
                        .model_and_level_at_active(self.starting_level);
                    let restored_level = agent_core::clamp_thinking_level(
                        &agent_core::capabilities(&restored_model),
                        restored_level,
                    );
                    if restored_model != self.current_model {
                        self.session.scrub_cross_model_state(&restored_model);
                        match build_gateway_client(&self.cfg, &restored_model) {
                            Ok(new_client) => self.client = Arc::new(new_client),
                            Err(e) => eprintln!(
                                "serve: could not resolve a gateway credential for restored model {restored_model:?}: \
                                     {e} — keeping the previous client"
                            ),
                        }
                        self.current_model = restored_model;
                    }
                    if restored_level != self.current_level {
                        self.current_level = restored_level;
                        self.current_thinking = None;
                    }
                    // MCP kit reset to all-enabled above — rebuild so the advertised tool set
                    // cannot keep the previous session's allowlist.
                    // always rebuild (MCP kit may have reset; model/level may have changed)
                    {
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
                    let _ = self.out_tx.send(response(
                        id,
                        "switch_session",
                        true,
                        Some(json!({
                            "session_id": self.persistence.session_id(),
                            "cwd_stale": self.mode.cwd_is_stale(&self.persistence.meta.cwd, &self.cwd),
                            "model": self.current_model,
                            // Task 3 (pi-parity fix, serve pass 19): matches `switch_branch`/`fork`/
                            // `clone`'s own response shape — previously omitted here, forcing a
                            // client to make a separate `get_state` round trip after every
                            // `switch_session` its 3 sibling commands don't need.
                            "reasoning_effort": self.current_level.as_str(),
                        })),
                        None,
                    ));
                }
                Err(e) => {
                    let _ = self.out_tx.send(response(
                        id,
                        "switch_session",
                        false,
                        None,
                        Some(&e.to_string()),
                    ));
                }
            },
            None => {
                let _ = self.out_tx.send(response(
                    id,
                    "switch_session",
                    false,
                    None,
                    Some("missing `session_id`"),
                ));
            }
        }
    }

    pub(in crate::serve) async fn cmd_fork(&mut self, cmd: &Value, id: Option<String>) {
        // `upto` messages to copy into the new session; absent = clone the whole session.
        // `target_id`, when given, forks at that specific tree entry instead — anywhere in the
        // whole tree, not just a message-count prefix of the active path (`before` excludes the
        // entry itself); wins over `upto` if both are present.
        let upto = cmd
            .get("upto")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(usize::MAX);
        let target_id = cmd.get("target_id").and_then(Value::as_str);
        // Defaults to `true` (excluding the target entry itself), matching pi's real production
        // client convention.
        let before = cmd.get("before").and_then(Value::as_bool).unwrap_or(true);
        // Resolved *before* `persistence.fork` below swaps `self.store` onto the freshly created
        // session — `target_id` can name an entry that isn't even on the copied prefix (an
        // off-active-path fork point, or one `before` excludes), so it wouldn't survive into the
        // new tree to look up afterward.
        let fork_text = match target_id {
            Some(tid) => self
                .persistence
                .store
                .as_ref()
                .and_then(|s| s.message_at(tid))
                .and_then(message_text),
            None => upto
                .min(self.session.messages.len())
                .checked_sub(1)
                .and_then(|i| self.session.messages.get(i))
                .and_then(message_text),
        };
        // Service mode derives without switching: this task keeps its own id, its lock and its
        // grant, and the client gets the new id to connect to. See `fork_detached`.
        if self.mode.forks_detached() {
            let frame = match self.persistence.fork_detached(upto, target_id, before) {
                Ok(new_id) => response(
                    id,
                    "fork",
                    true,
                    Some(json!({
                        "session_id": new_id,
                        "switched": false,
                        "text": fork_text,
                    })),
                    None,
                ),
                Err(e) => response(id, "fork", false, None, Some(&e.to_string())),
            };
            let _ = self.out_tx.send(frame);
            return;
        }
        match self
            .persistence
            .fork(upto, target_id, before, self.starting_level)
        {
            Ok((mut s, restored_model, restored_level)) => {
                // The fork keeps the view context of the turns it copies (see `mcp_apps::carry_context`).
                crate::tools::mcp_apps::carry_context(&self.session, &mut s);
                self.session = s;
                self.repoint_session_memory();
                self.reset_exec_endpoint().await;
                self.reset_mcp_enabled();
                self.steering.clear();
                // Restore whichever model/thinking-level was actually active at the forked-from
                // point, the same way `switch_session`/`switch_branch` already do.
                let restored_level = agent_core::clamp_thinking_level(
                    &agent_core::capabilities(&restored_model),
                    restored_level,
                );
                if restored_model != self.current_model {
                    self.session.scrub_cross_model_state(&restored_model);
                    match build_gateway_client(&self.cfg, &restored_model) {
                        Ok(new_client) => self.client = Arc::new(new_client),
                        Err(e) => eprintln!(
                            "serve: could not resolve a gateway credential for restored model {restored_model:?}: \
                                 {e} — keeping the previous client"
                        ),
                    }
                    self.current_model = restored_model;
                }
                if restored_level != self.current_level {
                    self.current_level = restored_level;
                    self.current_thinking = None;
                }
                // MCP kit was reset to all-enabled with the session switch — rebuild so the
                // advertised tool set cannot keep the previous session's allowlist.
                // always rebuild (MCP kit may have reset; model/level may have changed)
                {
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
                let _ = self.out_tx.send(response(
                    id,
                    "fork",
                    true,
                    Some(json!({
                        "session_id": self.persistence.session_id(),
                        "cwd_stale": self.mode.cwd_is_stale(&self.persistence.meta.cwd, &self.cwd),
                        "model": self.current_model,
                        "reasoning_effort": self.current_level.as_str(),
                        "text": fork_text,
                    })),
                    None,
                ));
            }
            Err(e) => {
                let _ = self
                    .out_tx
                    .send(response(id, "fork", false, None, Some(&e.to_string())));
            }
        }
    }

    pub(in crate::serve) async fn cmd_clone(&mut self, id: Option<String>) {
        // pi's own `clone` — fork the current session at its current tip, with no arguments.
        // Service mode derives without switching, exactly like `fork` above.
        if self.mode.forks_detached() {
            let frame = match self.persistence.fork_detached(usize::MAX, None, false) {
                Ok(new_id) => response(
                    id,
                    "clone",
                    true,
                    Some(json!({ "session_id": new_id, "switched": false })),
                    None,
                ),
                Err(e) => response(id, "clone", false, None, Some(&e.to_string())),
            };
            let _ = self.out_tx.send(frame);
            return;
        }
        match self
            .persistence
            .fork(usize::MAX, None, false, self.starting_level)
        {
            Ok((mut s, restored_model, restored_level)) => {
                // The clone keeps the view context its turns carried (see `mcp_apps::carry_context`).
                crate::tools::mcp_apps::carry_context(&self.session, &mut s);
                self.session = s;
                self.repoint_session_memory();
                self.reset_exec_endpoint().await;
                self.reset_mcp_enabled();
                self.steering.clear();
                let restored_level = agent_core::clamp_thinking_level(
                    &agent_core::capabilities(&restored_model),
                    restored_level,
                );
                if restored_model != self.current_model {
                    self.session.scrub_cross_model_state(&restored_model);
                    match build_gateway_client(&self.cfg, &restored_model) {
                        Ok(new_client) => self.client = Arc::new(new_client),
                        Err(e) => eprintln!(
                            "serve: could not resolve a gateway credential for restored model {restored_model:?}: \
                                 {e} — keeping the previous client"
                        ),
                    }
                    self.current_model = restored_model;
                }
                if restored_level != self.current_level {
                    self.current_level = restored_level;
                    self.current_thinking = None;
                }
                // MCP kit was reset to all-enabled with the session switch — rebuild so the
                // advertised tool set cannot keep the previous session's allowlist.
                // always rebuild (MCP kit may have reset; model/level may have changed)
                {
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
                let _ = self.out_tx.send(response(
                    id,
                    "clone",
                    true,
                    Some(json!({
                        "session_id": self.persistence.session_id(),
                        "cwd_stale": self.mode.cwd_is_stale(&self.persistence.meta.cwd, &self.cwd),
                        "model": self.current_model,
                        "reasoning_effort": self.current_level.as_str(),
                    })),
                    None,
                ));
            }
            Err(e) => {
                let _ = self
                    .out_tx
                    .send(response(id, "clone", false, None, Some(&e.to_string())));
            }
        }
    }

    pub(in crate::serve) async fn cmd_compact(&mut self, cmd: &Value, id: Option<String>) -> bool {
        // Manual compaction (no run in flight here). Streams a `compacted` event if it cuts.
        // `custom_instructions`, when given, steers what the summary emphasizes.
        //
        // Evaluates to whether the session goes on: `false` once stdin closed or a shutdown
        // signal arrived mid-compaction, for the caller to end the session loop.
        //
        // The summary is a model call (up to the stream idle timeout per call), so it runs inside
        // a select that keeps the session answering, the same shape as the host-`bash` loop:
        // `abort` cancels it, read-only commands are answered from a pre-compaction snapshot,
        // anything else is refused as busy, and shutdown or EOF cancel it.
        let custom_instructions = cmd.get("custom_instructions").and_then(Value::as_str);
        // Busy for the reaper and drain until the response is sent (see `BusyGuard`).
        let _busy = BusyGuard::mark(&self.running);
        let tx = self.out_tx.clone();
        let mut compacted_tokens_before: Option<u32> = None;
        let mut compacted_summary: Option<String> = None;
        let mut compacted_tokens_after: Option<u32> = None;
        let mut compacted_first_kept: Option<usize> = None;
        // A compaction ends in a checkpoint, which waits until it is on disk (see
        // `ChannelCheckpoint`): persist checkpoints while it runs, as the prompt loop does. Scoped
        // so the compaction's borrows (`session`, the `compacted_*` slots) end with it.
        // The pre-compaction active path, for `first_kept_entry_id` below: the compaction's own
        // checkpoint rewrites the store before the call returns.
        let ids_before: Vec<String> = self
            .persistence
            .store
            .as_ref()
            .map(|s| s.active_ids().to_vec())
            .unwrap_or_default();
        // What the read-only commands answer while the compaction holds `&mut session`: the
        // session as it stands until the compaction lands.
        let stats_before = session_stats(
            &self.session,
            &compaction_config(&self.cfg, &self.current_model),
        );
        let message_count_before = self.session.messages.len();
        let mut compact_open = true;
        // An `abort` is acknowledged once the compaction has actually stopped, after its own
        // response, as a mid-run abort is.
        let mut compact_abort_acks: Vec<Option<String>> = Vec::new();
        let result = {
            let compaction_cancel = CancellationToken::new();
            // No usage reaches a compaction's sink, so no pressure point applies.
            let manual_steers =
                crate::memory::SessionSteers::new(self.session_memory_active, u32::MAX);
            let mut on_event = |ev: AgentEvent| {
                if let AgentEvent::Compacted {
                    tokens_before,
                    summary,
                    tokens_after,
                    first_kept,
                    ..
                } = &ev
                {
                    compacted_tokens_before = Some(*tokens_before);
                    compacted_summary = Some(summary.clone());
                    compacted_tokens_after = Some(*tokens_after);
                    compacted_first_kept = Some(*first_kept);
                }
                // The shared steer observer, as in the `prompt` run: a manual `compact` also
                // folds specifics into a lossy summary, so the model is steered to read back its
                // `/session` working memory on its next turn.
                manual_steers.observe(&ev, &self.steering);
                if let Some(frame) = event_frame(ev) {
                    let _ = tx.send(frame);
                }
            };
            let compaction = self.agent.compact(
                &mut self.session,
                agent_core::CompactionReason::Manual,
                &compaction_cancel,
                &mut on_event,
                custom_instructions,
            );
            tokio::pin!(compaction);
            loop {
                tokio::select! {
                    biased;
                    Some((messages, provenance, edited, cuts, ack)) = self.checkpoint_rx.recv() => {
                        let (p, r) = write_then_ack(
                            persist_messages_blocking(self.persistence.take_for_blocking(), messages, provenance, edited, cuts),
                            ack,
                        )
                        .await;
                        self.persistence = p;
                        if let Err(e) = r {
                            tracing::warn!(error = %e, "compaction checkpoint failed to persist");
                        }
                    }
                    r = &mut compaction => break r,
                    sig = self.shutdown.wait(), if compact_open => {
                        self.shutdown_cause = Some(sig);
                        compact_open = false;
                        compaction_cancel.cancel();
                    }
                    maybe_line = self.input_rx.recv(), if compact_open => match maybe_line {
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
                            match c.get("type").and_then(Value::as_str).unwrap_or("") {
                                "abort" => {
                                    compaction_cancel.cancel();
                                    compact_abort_acks.push(cid);
                                }
                                "get_state" => {
                                    let branch = self.mode.git_branch(&self.exec_cell, &self.cwd).await;
                                    let frame = get_state_frame(
                                        cid,
                                        stats_before.clone(),
                                        json!(message_count_before),
                                        Activity::Compacting,
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
                                "get_session_stats" => {
                                    let _ = self.out_tx.send(get_session_stats_frame(cid, stats_before.clone(), Activity::Compacting, &self.persistence, &self.mode));
                                }
                                // The fanout's snapshot is the transcript as it stood when the
                                // compaction began (published by the idle loop) — what the busy
                                // prompt loop answers from too.
                                "get_messages" => {
                                    let (msgs, ids) = {
                                        let f = lock_ignoring_poison(&self.out_conn);
                                        f.history_snapshot()
                                    };
                                    let _ = self.out_tx.send(get_messages_frame(cid, c.get("since").and_then(Value::as_str), &msgs, &ids));
                                }
                                // The reads that need no `&mut session`, answered through the same
                                // `<type>_frame` functions as the idle table and the prompt busy loop.
                                "get_commands" => {
                                    let _ = self.out_tx.send(get_commands_frame(cid, &self.skills, &self.mcp_skills, &self.prompt_templates, &self.skill_collisions, &self.prompt_collisions));
                                }
                                "list_branches" => {
                                    let _ = self.out_tx.send(list_branches_frame(cid, &self.persistence));
                                }
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
                                "prompt" if crate::tools::mcp_events::is_injection(&c) => {
                                    self.deferred_events.push_back(l.to_string());
                                }
                                events_cmd if events_cmd.starts_with("mcp_events_") => dispatch_events(&self.events_cmds, &self.out_tx, cid, events_cmd, &c),
                                other => {
                                    let _ = self.out_tx.send(response(cid, other, false, None, Some("busy: a compaction is running; only `abort` (which cancels it) and the read-only commands (get_state/get_session_stats/get_messages/get_commands/list_branches/get_tree/list_sessions/list_all_sessions/get_available_models) are accepted")));
                                }
                            }
                        }
                        None => {
                            compact_open = false;
                            compaction_cancel.cancel();
                        }
                    },
                }
            }
        };
        match result {
            Ok(outcome) => {
                // The tree entry that begins the retained (post-compaction) portion of history —
                // resolved against the active path as it was before the compaction rewrote it.
                let first_kept_entry_id =
                    compacted_first_kept.and_then(|first_kept| ids_before.get(first_kept).cloned());
                while self.checkpoint_rx.try_recv().is_ok() {}
                let (p, persist_result) =
                    persist_blocking(self.persistence.take_for_blocking(), self.session.clone())
                        .await;
                self.persistence = p;
                match persist_result {
                    Ok(()) => {
                        let _ = self.out_tx.send(response(
                            id,
                            "compact",
                            true,
                            Some(json!({
                                "compacted": outcome.compacted(),
                                "reason": outcome.reason(),
                                "summary": compacted_summary,
                                "tokens_before": compacted_tokens_before,
                                "tokens_after": compacted_tokens_after,
                                "first_kept_entry_id": first_kept_entry_id,
                            })),
                            None,
                        ));
                    }
                    Err(e) => {
                        let _ = self.out_tx.send(response(
                            id,
                            "compact",
                            false,
                            None,
                            Some(&format!("compacted but failed to persist: {e}")),
                        ));
                    }
                }
            }
            Err(e) => {
                let _ =
                    self.out_tx
                        .send(response(id, "compact", false, None, Some(&e.to_string())));
            }
        }
        for ack in compact_abort_acks {
            let _ = self.out_tx.send(response(ack, "abort", true, None, None));
        }
        compact_open
    }
}
