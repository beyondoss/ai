use crate::serve::*;

impl SessionLoop {
    pub(in crate::serve) fn cmd_set_label(&mut self, cmd: &Value, id: Option<String>) -> bool {
        match cmd.get("target_id").and_then(Value::as_str) {
            Some(target_id) => match cmd.get("label") {
                // Present-but-null is the explicit "clear it" signal; a missing key is an error
                // (so a typo can't silently no-op) — matches `set_thinking`'s own `budget` contract.
                Some(Value::Null) => match self.persistence.set_label(target_id, None) {
                    Ok(()) => {
                        if !self.emit(response(
                            id,
                            "set_label",
                            true,
                            Some(json!({ "target_id": target_id, "label": Value::Null })),
                            None,
                        )) {
                            return false;
                        }
                    }
                    Err(e) => {
                        if !self.emit(response(id, "set_label", false, None, Some(&e.to_string())))
                        {
                            return false;
                        }
                    }
                },
                Some(Value::String(label)) => {
                    match self.persistence.set_label(target_id, Some(label)) {
                        Ok(()) => {
                            if !self.emit(response(
                                id,
                                "set_label",
                                true,
                                Some(json!({ "target_id": target_id, "label": label })),
                                None,
                            )) {
                                return false;
                            }
                        }
                        Err(e) => {
                            if !self.emit(response(
                                id,
                                "set_label",
                                false,
                                None,
                                Some(&e.to_string()),
                            )) {
                                return false;
                            }
                        }
                    }
                }
                Some(_) => {
                    if !self.emit(response(
                        id,
                        "set_label",
                        false,
                        None,
                        Some("`label` must be a string or null"),
                    )) {
                        return false;
                    }
                }
                None => {
                    if !self.emit(response(
                        id,
                        "set_label",
                        false,
                        None,
                        Some("missing `label` — pass a string to set it or null to clear it"),
                    )) {
                        return false;
                    }
                }
            },
            None => {
                if !self.emit(response(
                    id,
                    "set_label",
                    false,
                    None,
                    Some("missing `target_id`"),
                )) {
                    return false;
                }
            }
        }
        true
    }

    pub(in crate::serve) fn cmd_get_label(&mut self, cmd: &Value, id: Option<String>) -> bool {
        match cmd.get("target_id").and_then(Value::as_str) {
            Some(target_id) => match self.persistence.get_label(target_id) {
                Ok(label) => {
                    if !self.emit(response(
                        id,
                        "get_label",
                        true,
                        Some(json!({ "target_id": target_id, "label": label })),
                        None,
                    )) {
                        return false;
                    }
                }
                Err(e) => {
                    if !self.emit(response(id, "get_label", false, None, Some(&e.to_string()))) {
                        return false;
                    }
                }
            },
            None => {
                if !self.emit(response(
                    id,
                    "get_label",
                    false,
                    None,
                    Some("missing `target_id`"),
                )) {
                    return false;
                }
            }
        }
        true
    }

    pub(in crate::serve) fn cmd_append_custom(&mut self, cmd: &Value, id: Option<String>) -> bool {
        match cmd.get("kind").and_then(Value::as_str) {
            // The host's own journal kinds: an entry of one is trusted when the session is
            // replayed (a task to resume, a result to deliver), so a client must not be able to
            // write one.
            Some(kind) if crate::session_store::is_host_custom_kind(kind) => {
                if !self.emit(response(
                    id,
                    "append_custom",
                    false,
                    None,
                    Some(&format!(
                        "custom entry kind `{kind}` is reserved for entries the agent writes itself"
                    )),
                )) {
                    return false;
                }
            }
            Some(kind) => {
                let data = cmd.get("data").cloned().unwrap_or_else(|| json!({}));
                match self.persistence.append_custom(kind, data) {
                    Ok(new_id) => {
                        if !self.emit(response(
                            id,
                            "append_custom",
                            true,
                            Some(json!({ "id": new_id })),
                            None,
                        )) {
                            return false;
                        }
                    }
                    Err(e) => {
                        if !self.emit(response(
                            id,
                            "append_custom",
                            false,
                            None,
                            Some(&e.to_string()),
                        )) {
                            return false;
                        }
                    }
                }
            }
            None => {
                if !self.emit(response(
                    id,
                    "append_custom",
                    false,
                    None,
                    Some("missing `kind`"),
                )) {
                    return false;
                }
            }
        }
        true
    }

    pub(in crate::serve) fn cmd_list_branches(&mut self, id: Option<String>) -> bool {
        if !self.emit(response(
            id,
            "list_branches",
            true,
            Some(json!({ "branches": self.persistence.list_branches() })),
            None,
        )) {
            return false;
        }
        true
    }

    pub(in crate::serve) fn cmd_get_tree(&mut self, cmd: &Value, id: Option<String>) -> bool {
        // `since` (Task #48, pi-parity gap): return only entries appended after a tree id the
        // client already has, across every entry type — see [`nodes_since`]. Unmatched is an
        // error, not a silent full re-fetch, mirroring `get_messages`'s own `since` above.
        let nodes = match cmd.get("since").and_then(Value::as_str) {
            Some(since) => match nodes_since(self.persistence.tree(), since) {
                Ok(nodes) => nodes,
                Err(e) => {
                    if !self.emit(response(id, "get_tree", false, None, Some(&e))) {
                        return false;
                    }
                    return true;
                }
            },
            None => self.persistence.tree(),
        };
        if !self.emit(response(
            id,
            "get_tree",
            true,
            Some(json!({ "nodes": nodes, "leaf_id": self.persistence.active_ids().last() })),
            None,
        )) {
            return false;
        }
        true
    }

    pub(in crate::serve) async fn cmd_switch_branch(
        &mut self,
        cmd: &Value,
        id: Option<String>,
    ) -> bool {
        match cmd.get("target_id").and_then(Value::as_str) {
            Some(target_id) => {
                // When set, switches to `target_id`'s own parent instead — the tree's root (before
                // any message) when `target_id` is the very first one, letting a client redo it in
                // place. Mirrors `fork`'s identical `before` flag.
                let before = cmd.get("before").and_then(Value::as_bool).unwrap_or(false);
                // Defaults to summarizing the abandoned branch's activity (mirroring pi's
                // `navigateTree`); a client can pass `summarize:false` for a quick, cheap switch.
                let summarize = cmd
                    .get("summarize")
                    .and_then(Value::as_bool)
                    .unwrap_or(true);
                // Steers what the abandoned branch's recap emphasizes — the same "Additional
                // focus" framing the `compact` command already supports (see its own
                // `custom_instructions` handling above). Ignored when `summarize` is false, since
                // no summarization call happens at all in that case.
                let custom_instructions = cmd.get("custom_instructions").and_then(Value::as_str);
                // Task #17 (pi-parity fix): `true` uses `custom_instructions` as the *entire*
                // instruction section instead of appending it after the default structured
                // template — see `Agent::summarize_branch`'s own doc comment for the exact
                // semantics. Read the same way `custom_instructions` itself is, just above; a no-op
                // when `custom_instructions` is absent, or when `summarize` is false (no
                // summarization call happens at all in that case).
                let replace_instructions = cmd
                    .get("replace_instructions")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let target_id = target_id.to_string();
                // Branch summarization is one LLM call, but it's still a real network round trip
                // — potentially the slowest single operation this loop ever awaits outside a
                // `prompt` run. Racing it against stdin (mirroring `prompt`'s own busy-select
                // below, scaled down to this operation's much narrower surface — pi exposes only
                // `abortBranchSummary` here, nothing like steer/follow_up) keeps an `abort` able to
                // interrupt it instead of the whole RPC loop going unresponsive until the call
                // either lands or times out. A stdin close/shutdown mid-call also cancels, so the
                // loop can still unwind and exit cleanly instead of hanging on this one `.await`.
                let branch_cancel = CancellationToken::new();
                let mut branch_stdin_open = true;
                let switch_result = {
                    let fut = self.persistence.switch_branch(
                        &self.agent,
                        &target_id,
                        before,
                        summarize,
                        custom_instructions,
                        replace_instructions,
                        &branch_cancel,
                    );
                    tokio::pin!(fut);
                    loop {
                        tokio::select! {
                            biased;
                            r = &mut fut => break r,
                            sig = self.shutdown.wait() => {
                                self.shutdown_cause = Some(sig);
                                branch_stdin_open = false;
                                branch_cancel.cancel();
                            }
                            maybe_line = self.input_rx.recv(), if branch_stdin_open => match maybe_line {
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
                                            branch_cancel.cancel();
                                            let _ = self.out_tx.send(response(cid, "abort", true, None, None));
                                        }
                                        "prompt" if crate::tools::mcp_events::is_injection(&c) => {
                                            self.deferred_events.push_back(l.to_string());
                                        }
                                        events_cmd if events_cmd.starts_with("mcp_events_") => dispatch_events(&self.events_cmds, &self.out_tx, cid, events_cmd, &c),
                                        other => {
                                            let _ = self.out_tx.send(response(cid, other, false, None, Some("busy: a branch switch is summarizing; only `abort` is accepted until it settles")));
                                        }
                                    }
                                }
                                None => {
                                    branch_stdin_open = false;
                                    branch_cancel.cancel();
                                }
                            }
                        }
                    }
                };
                match switch_result {
                    Ok((s, resolved_target)) => {
                        self.session = s;
                        self.steering.clear();
                        // Restore whichever model/thinking-level was actually active on this
                        // branch, rather than silently continuing with the process's current
                        // global setting (which may have since moved on via a `set_model`/
                        // `cycle_thinking_level` made on a *different* branch — that's the bleed
                        // this guards against). Both always resolve to something real: the model
                        // falls back to the session's own creation-time model, the level to the
                        // process's own starting level (`starting_level`) — see
                        // `model_and_level_at`'s doc comment. Queried against `resolved_target`
                        // (`None` when `before` landed on the tree's own root), not the raw
                        // `target_id` argument, which names the entry navigated *relative to*, not
                        // necessarily where the session actually ended up.
                        let (restored_model, restored_level) =
                            self.persistence.model_and_level_at_opt(
                                resolved_target.as_deref(),
                                self.starting_level,
                            );
                        // Clamp against the *restored* model's capabilities, not the model that
                        // was active before the switch — a branch recorded at `Off` on a model
                        // that's since been superseded by a non-disable-capable one must not
                        // resurrect an illegal `Off` state.
                        let restored_level = agent_core::clamp_thinking_level(
                            &agent_core::capabilities(&restored_model),
                            restored_level,
                        );
                        if restored_model != self.current_model {
                            self.session.scrub_cross_model_state(&restored_model);
                            // The restored model can be on a different OAuth provider than whatever was
                            // active before this switch — best-effort, matching this restore path's existing
                            // "recorded state wins" philosophy: a failure here leaves `client` on the previous
                            // (now-wrong-for-`current_model`) credential rather than aborting the whole switch,
                            // so the next `prompt` surfaces its own clear transport-level error instead.
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
                        if !self.emit(response(
                            id,
                            "switch_branch",
                            true,
                            Some(json!({
                                "target_id": target_id,
                                "model": self.current_model,
                                "reasoning_effort": self.current_level.as_str(),
                            })),
                            None,
                        )) {
                            return false;
                        }
                    }
                    // Not a failure — the client asked to abort, and the session is left exactly
                    // as it was (no partial summary, no switch): mirrors pi's `navigateTree`
                    // result shape (`{cancelled:true, aborted:true}`) rather than surfacing this as
                    // an RPC-level error.
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {
                        if !self.emit(response(
                            id,
                            "switch_branch",
                            true,
                            Some(json!({
                                "target_id": target_id,
                                "cancelled": true,
                                "aborted": true,
                            })),
                            None,
                        )) {
                            return false;
                        }
                    }
                    Err(e) => {
                        if !self.emit(response(
                            id,
                            "switch_branch",
                            false,
                            None,
                            Some(&e.to_string()),
                        )) {
                            return false;
                        }
                    }
                }
            }
            None => {
                if !self.emit(response(
                    id,
                    "switch_branch",
                    false,
                    None,
                    Some("missing `target_id`"),
                )) {
                    return false;
                }
            }
        }
        true
    }
}
