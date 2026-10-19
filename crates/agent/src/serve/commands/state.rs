use crate::serve::*;

impl SessionLoop {
    pub(in crate::serve) async fn cmd_get_state(&mut self, id: Option<String>) -> bool {
        let mut data = session_stats(
            &self.session,
            &compaction_config(&self.cfg, &self.current_model),
        );
        if let Value::Object(m) = &mut data {
            insert_session_identity(m, &self.persistence, &self.mode);
            m.insert("model".into(), json!(self.current_model));
            m.insert("message_count".into(), json!(self.session.messages.len()));
            m.insert("title".into(), json!(self.persistence.meta.title));
            // Never stale in service mode: the recorded cwd *is* `workspace_root`, and the
            // staleness check stats the path — on the replica, where it doesn't exist.
            m.insert(
                "cwd_stale".into(),
                json!(
                    self.mode
                        .cwd_is_stale(&self.persistence.meta.cwd, &self.cwd)
                ),
            );
            // Task #25 (pi-parity fix): the directory (and, best-effort, branch) the agent's
            // tools are actually operating against — the live process `cwd`, not
            // `persistence.meta.cwd` (already separately surfaced via `cwd_stale` when the two
            // disagree), since that's what a remote client with no shared filesystem needs to
            // know. See `git_branch`'s own doc comment for why a lookup failure is `null`, not
            // an error.
            m.insert("cwd".into(), json!(self.cwd.display().to_string()));
            // In service mode the lookup runs **in the sandbox**, through this session's own
            // runner: `cwd` is a sandbox path, so asking this host about it would either fail
            // or — worse — answer about the replica's own checkout.
            m.insert(
                "git_branch".into(),
                json!(self.mode.git_branch(&self.exec_cell, &self.cwd).await),
            );
            // Both hardcoded, not stale placeholders: no `prompt`/compaction can possibly be
            // in flight here at all — this arm only ever runs from the idle main loop, which
            // processes one command to completion before reading the next, so there is no
            // concurrently-running turn or compaction this response could be racing.
            m.insert("is_streaming".into(), json!(false));
            m.insert("is_compacting".into(), json!(false));
            // Same reasoning as `is_streaming`/`is_compacting` above: idle means nothing is
            // dispatching a tool call either.
            m.insert("pending_tool_ids".into(), json!(Vec::<String>::new()));
            if let Value::Object(rt) = runtime_settings(
                self.current_level,
                self.current_auto_compaction,
                self.current_auto_retry,
                self.current_block_images,
                self.current_image_auto_resize,
                &self.steering,
            ) {
                m.extend(rt);
            }
        }
        if !self.emit(response(id, "get_state", true, Some(data), None)) {
            return false;
        }
        true
    }

    pub(in crate::serve) fn cmd_get_messages(&mut self, cmd: &Value, id: Option<String>) -> bool {
        // Idle: `session` is ours to read. The busy loop answers the same command from the
        // fanout's committed-history snapshot instead (the run holds `&mut session`), through
        // this same [`messages_payload`] — one payload shape either way.
        let msg_ids = self.persistence.active_ids();
        match messages_payload(
            &self.session.messages,
            msg_ids,
            cmd.get("since").and_then(Value::as_str),
        ) {
            Ok(data) => {
                if !self.emit(response(id, "get_messages", true, Some(data), None)) {
                    return false;
                }
            }
            Err(e) => {
                if !self.emit(response(id, "get_messages", false, None, Some(&e))) {
                    return false;
                }
            }
        }
        true
    }

    pub(in crate::serve) fn cmd_get_last_assistant_text(&mut self, id: Option<String>) -> bool {
        let text = last_assistant_text(&self.session);
        if !self.emit(response(
            id,
            "get_last_assistant_text",
            true,
            Some(json!({ "text": text })),
            None,
        )) {
            return false;
        }
        true
    }

    pub(in crate::serve) fn cmd_get_todos(&mut self, id: Option<String>) -> bool {
        // Straight from the session while idle — no mirror needed, and no chance of one going
        // stale against a `switch_branch`/`fork`/`compact` that happened since the last turn.
        if !self.emit(response(
            id,
            "get_todos",
            true,
            Some(json!({ "todos": current_todos(&self.session) })),
            None,
        )) {
            return false;
        }
        true
    }

    pub(in crate::serve) fn cmd_get_session_stats(&mut self, id: Option<String>) -> bool {
        // Fix 8 (pi-parity gap): backfills session_id/session_file (mirroring get_state's own
        // idle arm, above) and pending_tool_ids (mirroring get_state's own "nothing pending
        // while idle" convention) — previously omitted, unlike the busy-mode arm's
        // `LiveStats::snapshot()`-derived `pending_tool_ids`, so idle and busy reported
        // different field sets for the same command.
        let mut data = session_stats(
            &self.session,
            &compaction_config(&self.cfg, &self.current_model),
        );
        if let Value::Object(m) = &mut data {
            insert_session_identity(m, &self.persistence, &self.mode);
            m.insert("pending_tool_ids".into(), json!(Vec::<String>::new()));
        }
        if !self.emit(response(id, "get_session_stats", true, Some(data), None)) {
            return false;
        }
        true
    }

    pub(in crate::serve) fn cmd_get_commands(&mut self, id: Option<String>) -> bool {
        // Skills (read-on-demand) and prompt templates (`/name`), for client autocomplete.
        // `scope`/`path` (Task #39 pi-parity fix — previously omitted entirely) mirror pi's own
        // `get_commands` `sourceInfo.scope`/`sourceInfo.path`: which discovery root a command
        // actually came from (`"user"`/`"project"`/`"temporary"`) and its on-disk location.
        let mut commands: Vec<Value> = self
            .skills
            .iter()
            .map(|s| {
                json!({
                    "name": format!("skill:{}", s.name),
                    "source": "skill",
                    "description": s.description,
                    "scope": s.scope,
                    "path": s.path,
                })
            })
            .collect();
        // MCP-served skills (`scope: "mcp"`, `path` = the SKILL.md URI) of enabled servers.
        commands.extend(self.mcp_skills.commands());
        commands.extend(self.prompt_templates.iter().map(|t| {
            json!({
                "name": t.name,
                "source": "prompt",
                "description": t.description,
                "scope": t.scope,
                "path": t.path,
            })
        }));
        // Every shadowed name (a skill or template defined at more than one path) is otherwise
        // silently resolved with no way for a client to notice — surfaced here instead.
        // MCP skills' diagnostics too: entries declined (invalid, dynamic, over the
        // per-skill limits), same-name entries, and on-disk name clashes.
        let mcp_diagnostics = self.mcp_skills.diagnostics(&self.skills);
        let collisions: Vec<&crate::skills::Collision> = self
            .skill_collisions
            .iter()
            .chain(self.prompt_collisions.iter())
            .chain(mcp_diagnostics.iter())
            .collect();
        if !self.emit(response(
            id,
            "get_commands",
            true,
            Some(json!({ "commands": commands, "collisions": collisions })),
            None,
        )) {
            return false;
        }
        true
    }

    pub(in crate::serve) fn cmd_get_available_models(&mut self, id: Option<String>) -> bool {
        // Deliberately the full known-model hint list, not `cycle_models` — `--models` scopes
        // what `cycle_model` steps through, not what a client's model *picker* can see. pi's
        // own `/model` selector defaults its view to the scope but still lets the operator Tab
        // to the full catalog; since this RPC has no such secondary toggle, it always answers
        // with the unscoped list and leaves any scoped-vs-all UI distinction to the client.
        let models: Vec<Value> = listed_models().iter().map(|m| model_info(m)).collect();
        if !self.emit(response(
            id,
            "get_available_models",
            true,
            Some(json!({ "models": models })),
            None,
        )) {
            return false;
        }
        true
    }
}
