use crate::serve::*;

/// What the session is doing when a read-only command is answered. The idle command table, the
/// busy loop of a running `prompt` and the busy loop of a running `compact` all answer the same
/// reads; this is the one place their differences are spelled out.
#[derive(Clone, Copy)]
pub(in crate::serve) enum Activity {
    Idle,
    /// A manual `compact` holds the session.
    Compacting,
    /// A `prompt` holds the session (`&mut session`), so the figures come from the live mirror.
    /// `is_compacting` is whether an auto-compaction is in flight inside that run.
    Running {
        is_compacting: bool,
    },
}

/// The `SessionLoop` fields `get_state` reports, borrowed one by one: a running `prompt` holds
/// `&mut self.session` for its whole duration, so no `&self` method may be called from its busy loop.
pub(in crate::serve) struct StateView<'a> {
    pub persistence: &'a Persistence,
    pub mode: &'a Mode,
    pub model: &'a str,
    pub cwd: &'a Path,
    /// Resolved by the caller before it borrows anything: in service mode it is a round trip into
    /// the sandbox.
    pub branch: Option<String>,
    /// [`runtime_settings`]'s output.
    pub runtime: Value,
}

/// The `get_state` response. `data` is the stats base (`session_stats`, or the live mirror's
/// snapshot while running) and `message_count` is `null` while running: the one field that needs
/// `&session` to report exactly, and a stale number would be worse than an honest "not available".
pub(in crate::serve) fn get_state_frame(
    id: Option<String>,
    mut data: Value,
    message_count: Value,
    activity: Activity,
    v: StateView<'_>,
) -> OutFrame {
    if let Value::Object(m) = &mut data {
        insert_session_identity(m, v.persistence, v.mode);
        m.insert("model".into(), json!(v.model));
        m.insert("message_count".into(), message_count);
        m.insert("title".into(), json!(v.persistence.meta.title));
        // Never stale in service mode: the recorded cwd *is* `workspace_root`, and the
        // staleness check stats the path — on the replica, where it doesn't exist.
        m.insert(
            "cwd_stale".into(),
            json!(v.mode.cwd_is_stale(&v.persistence.meta.cwd, v.cwd)),
        );
        // Task #25 (pi-parity fix): the directory (and, best-effort, branch) the agent's
        // tools are actually operating against — the live process `cwd`, not
        // `persistence.meta.cwd` (already separately surfaced via `cwd_stale` when the two
        // disagree), since that's what a remote client with no shared filesystem needs to
        // know. See `git_branch`'s own doc comment for why a lookup failure is `null`, not
        // an error.
        m.insert("cwd".into(), json!(v.cwd.display().to_string()));
        // In service mode the lookup runs **in the sandbox**, through this session's own
        // runner: `cwd` is a sandbox path, so asking this host about it would either fail
        // or — worse — answer about the replica's own checkout.
        m.insert("git_branch".into(), json!(v.branch));
        // Hardcoded when no `prompt` is running: nothing is streaming or dispatching a tool call
        // (the idle loop processes one command to completion before reading the next, and a
        // compaction is not a stream). A running prompt's `pending_tool_ids` is already in `data`,
        // from the live mirror.
        m.insert(
            "is_streaming".into(),
            json!(matches!(activity, Activity::Running { .. })),
        );
        m.insert(
            "is_compacting".into(),
            json!(match activity {
                Activity::Idle => false,
                Activity::Compacting => true,
                Activity::Running { is_compacting } => is_compacting,
            }),
        );
        if !matches!(activity, Activity::Running { .. }) {
            m.insert("pending_tool_ids".into(), json!(Vec::<String>::new()));
        }
        if let Value::Object(rt) = v.runtime {
            m.extend(rt);
        }
    }
    response(id, "get_state", true, Some(data), None)
}

/// The `get_session_stats` response. While a `prompt` runs, `&session` is exclusively borrowed by
/// the in-flight turn, so the fields only it can report are present as `null` rather than silently
/// absent (Fix 8, pi-parity gap): a client sees the same field *set* idle or busy, values aside.
/// Otherwise nothing is pending.
pub(in crate::serve) fn get_session_stats_frame(
    id: Option<String>,
    mut data: Value,
    activity: Activity,
    persistence: &Persistence,
    mode: &Mode,
) -> OutFrame {
    if let Value::Object(m) = &mut data {
        insert_session_identity(m, persistence, mode);
        match activity {
            Activity::Idle | Activity::Compacting => {
                m.insert("pending_tool_ids".into(), json!(Vec::<String>::new()));
            }
            Activity::Running { .. } => {
                for field in [
                    "context_usage",
                    "user_messages",
                    "assistant_messages",
                    "tool_calls",
                    "tool_results",
                    "total_messages",
                ] {
                    m.insert(field.to_string(), Value::Null);
                }
            }
        }
    }
    response(id, "get_session_stats", true, Some(data), None)
}

/// The `get_messages` response over a given transcript: the live session while idle, the fanout's
/// committed-history snapshot while a run or compaction holds `&mut session` (the same
/// [`messages_payload`] either way — one payload shape).
pub(in crate::serve) fn get_messages_frame(
    id: Option<String>,
    since: Option<&str>,
    messages: &[agent_core::Message],
    msg_ids: &[String],
) -> OutFrame {
    match messages_payload(messages, msg_ids, since) {
        Ok(data) => response(id, "get_messages", true, Some(data), None),
        Err(e) => response(id, "get_messages", false, None, Some(&e)),
    }
}

/// The `get_commands` response: skills (read-on-demand) and prompt templates (`/name`), for client
/// autocomplete. `scope`/`path` (Task #39 pi-parity fix — previously omitted entirely) mirror pi's
/// own `get_commands` `sourceInfo.scope`/`sourceInfo.path`: which discovery root a command
/// actually came from (`"user"`/`"project"`/`"temporary"`) and its on-disk location.
pub(in crate::serve) fn get_commands_frame(
    id: Option<String>,
    skills: &[crate::skills::Skill],
    mcp_skills: &crate::tools::mcp_skills::McpSkills,
    prompt_templates: &[crate::prompts::PromptTemplate],
    skill_collisions: &[crate::skills::Collision],
    prompt_collisions: &[crate::skills::Collision],
) -> OutFrame {
    let mut commands: Vec<Value> = skills
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
    commands.extend(mcp_skills.commands());
    commands.extend(prompt_templates.iter().map(|t| {
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
    let mcp_diagnostics = mcp_skills.diagnostics(skills);
    let collisions: Vec<&crate::skills::Collision> = skill_collisions
        .iter()
        .chain(prompt_collisions.iter())
        .chain(mcp_diagnostics.iter())
        .collect();
    response(
        id,
        "get_commands",
        true,
        Some(json!({ "commands": commands, "collisions": collisions })),
        None,
    )
}

/// The `get_available_models` response. Deliberately the full known-model hint list, not
/// `cycle_models` — `--models` scopes what `cycle_model` steps through, not what a client's model
/// *picker* can see. pi's own `/model` selector defaults its view to the scope but still lets the
/// operator Tab to the full catalog; since this RPC has no such secondary toggle, it always
/// answers with the unscoped list and leaves any scoped-vs-all UI distinction to the client.
pub(in crate::serve) fn get_available_models_frame(id: Option<String>) -> OutFrame {
    let models: Vec<Value> = listed_models().iter().map(|m| model_info(m)).collect();
    response(
        id,
        "get_available_models",
        true,
        Some(json!({ "models": models })),
        None,
    )
}

impl SessionLoop {
    pub(in crate::serve) async fn cmd_get_state(&mut self, id: Option<String>) -> bool {
        let data = session_stats(
            &self.session,
            &compaction_config(&self.cfg, &self.current_model),
        );
        let branch = self.mode.git_branch(&self.exec_cell, &self.cwd).await;
        let frame = get_state_frame(
            id,
            data,
            json!(self.session.messages.len()),
            Activity::Idle,
            StateView {
                persistence: &self.persistence,
                mode: &self.mode,
                model: &self.current_model,
                cwd: &self.cwd,
                branch,
                runtime: runtime_settings(
                    self.current_level,
                    self.current_auto_compaction,
                    self.current_auto_retry,
                    self.current_block_images,
                    self.current_image_auto_resize,
                    &self.steering,
                ),
            },
        );
        self.emit(frame)
    }

    pub(in crate::serve) fn cmd_get_messages(&mut self, cmd: &Value, id: Option<String>) -> bool {
        // Idle: `session` is ours to read. The busy loops answer the same command from the
        // fanout's committed-history snapshot instead (the run holds `&mut session`).
        let frame = get_messages_frame(
            id,
            cmd.get("since").and_then(Value::as_str),
            &self.session.messages,
            self.persistence.active_ids(),
        );
        self.emit(frame)
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
        let data = session_stats(
            &self.session,
            &compaction_config(&self.cfg, &self.current_model),
        );
        let frame =
            get_session_stats_frame(id, data, Activity::Idle, &self.persistence, &self.mode);
        self.emit(frame)
    }

    pub(in crate::serve) fn cmd_get_commands(&mut self, id: Option<String>) -> bool {
        let frame = get_commands_frame(
            id,
            &self.skills,
            &self.mcp_skills,
            &self.prompt_templates,
            &self.skill_collisions,
            &self.prompt_collisions,
        );
        self.emit(frame)
    }

    pub(in crate::serve) fn cmd_get_available_models(&mut self, id: Option<String>) -> bool {
        self.emit(get_available_models_frame(id))
    }
}
