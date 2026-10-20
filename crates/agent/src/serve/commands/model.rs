use crate::serve::*;

impl SessionLoop {
    // Rejects an empty/whitespace-only id, and — Fix 10 (pi-parity feature) — resolves a
    // partial/fuzzy id against the known-model hint list first (`resolve_model_id`, mirroring
    // `--model`'s identical resolution in `main.rs`): a genuinely unrecognized id (no partial
    // match at all) is still forwarded verbatim, unlike pi (which talks directly to each
    // provider and can validate against a live, authoritative registry of what it's actually
    // configured to reach) — this process has no local source of truth to validate a real id
    // against, and `available_models()` is explicitly documented as a non-exhaustive picker
    // hint, not an allowlist (see its own doc comment). An empty string sneaking through
    // `Value::as_str` and getting durably recorded via `record_model_change` is caught here too,
    // the same class of fix `set_session_name` already got (reject empty, don't pretend to
    // validate against a list this process can't actually authoritatively check). `model` may
    // also carry a `:<level>` suffix (Fix 2, pi-parity gap — e.g. `"sonnet:high"`); when present,
    // `resolve_model_id` returns it alongside the resolved id and it wins outright over whatever
    // level was already active.
    pub(in crate::serve) fn cmd_set_model(&mut self, cmd: &Value, id: Option<String>) -> bool {
        match cmd
            .get("model")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|raw| resolve_model_id(raw, available_models()))
        {
            Some(Ok((model, thinking_level))) => {
                let model = model.as_str();
                if let Some(why) = self.mode.model_refusal(model) {
                    return self.emit(response(id, "set_model", false, None, Some(&why)));
                }
                // Re-derive the gateway credential/routing for `model` before touching any other
                // state — a switch can cross OAuth providers (or `models.json` overrides), and
                // this must fail closed (leaving `current_model`/the persisted lineage untouched)
                // rather than silently keep using whichever client happened to be active before.
                // Skipped when `model` is already the active one, for the same reason
                // `record_result` below skips a redundant persist: a no-op `set_model <current>`
                // shouldn't discard `OAuthCredentialSource`'s in-memory token cache for nothing.
                let new_client = if model == self.current_model {
                    Ok(None)
                } else {
                    build_gateway_client(&self.cfg, model).map(|c| Some(Arc::new(c)))
                };
                match new_client {
                    Ok(new_client) => {
                        // Persist the lineage marker *before* applying the switch in memory: if it
                        // fails to write, leave the live model unchanged too, rather than forking
                        // live state away from what's durably recorded (an `Err` here aborts the
                        // whole switch).
                        let record_result = if model != self.current_model {
                            self.persistence.record_model_change(model)
                        } else {
                            Ok(())
                        };
                        match record_result {
                            Ok(()) => {
                                // A signed thinking block is only valid for replay to the model
                                // that produced it, and a combined OpenAI-Responses tool-call id
                                // only means anything back to that same model — scrub both from
                                // any message not already stamped with the model we're switching
                                // to.
                                self.session.scrub_cross_model_state(model);
                                self.current_model = model.to_string();
                                // Fix 2 (pi-parity gap): a `:<level>` suffix on `model` (e.g.
                                // `sonnet:high`) wins outright over whatever level was already
                                // active — the operator explicitly asked for this depth on the new
                                // model. No suffix falls back to re-clamping the *existing* level
                                // against the *new* model instead, same as before this fix: e.g. a
                                // session sitting at `Off` on a disable-capable model must not
                                // silently carry that `Off` over to a model that can't actually
                                // disable reasoning.
                                self.current_level = agent_core::clamp_thinking_level(
                                    &agent_core::capabilities(&self.current_model),
                                    thinking_level.unwrap_or(self.current_level),
                                );
                                if let Some(new_client) = new_client {
                                    self.client = new_client;
                                }
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
                                if !self.emit(response(
                                    id,
                                    "set_model",
                                    true,
                                    Some(model_switch_response(
                                        &self.current_model,
                                        self.current_level,
                                    )),
                                    None,
                                )) {
                                    return false;
                                }
                            }
                            Err(e) => {
                                if !self.emit(response(
                                    id,
                                    "set_model",
                                    false,
                                    None,
                                    Some(&e.to_string()),
                                )) {
                                    return false;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        if !self.emit(response(id, "set_model", false, None, Some(&e))) {
                            return false;
                        }
                    }
                }
            }
            Some(Err(e)) => {
                if !self.emit(response(id, "set_model", false, None, Some(&e))) {
                    return false;
                }
            }
            None => {
                if !self.emit(response(
                    id,
                    "set_model",
                    false,
                    None,
                    Some("missing or empty `model`"),
                )) {
                    return false;
                }
            }
        }
        true
    }

    pub(in crate::serve) fn cmd_set_thinking(&mut self, cmd: &Value, id: Option<String>) -> bool {
        // `budget` is a positive integer to enable extended thinking, or `null` to disable it.
        // A present-but-null value is the explicit "turn it off" signal; a missing key is an
        // error (so a typo can't silently no-op).
        match cmd.get("budget") {
            Some(Value::Null) => {
                self.current_thinking = None;
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
                if !self.emit(response(
                    id,
                    "set_thinking",
                    true,
                    Some(json!({ "thinking": Value::Null })),
                    None,
                )) {
                    return false;
                }
            }
            Some(v) if v.as_u64().is_some() => {
                self.current_thinking = v.as_u64().map(|n| n as u32);
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
                if !self.emit(response(
                    id,
                    "set_thinking",
                    true,
                    Some(json!({ "thinking": self.current_thinking })),
                    None,
                )) {
                    return false;
                }
            }
            _ => {
                if !self.emit(response(
                    id,
                    "set_thinking",
                    false,
                    None,
                    Some("`budget` must be a non-negative integer or null"),
                )) {
                    return false;
                }
            }
        }
        true
    }

    pub(in crate::serve) fn cmd_set_reasoning_effort(
        &mut self,
        cmd: &Value,
        id: Option<String>,
    ) -> bool {
        // `effort` is one of the portable level names ("off"/"minimal"/"low"/"medium"/"high"/
        // "xhigh"), or `null` as an alias for `"off"`. Unlike `set_thinking`, this sets
        // `current_level` directly (no raw-override concept — a named effort already *is* a
        // ladder rung) and clears any pending `set_thinking` override, for the same reason
        // `cycle_thinking_level` does: the newly-requested level should take visible effect
        // immediately, not be silently masked by a stale raw budget.
        let parsed = match cmd.get("effort") {
            Some(Value::Null) => Some(agent_core::ThinkingLevel::Off),
            Some(Value::String(s)) => agent_core::ThinkingLevel::parse(s),
            _ => None,
        };
        match parsed {
            Some(level) => {
                // Clamp against `current_model`'s capabilities before storing or recording:
                // an explicit `"off"` request is only honored if the model can actually
                // disable reasoning; otherwise it's bumped to the model's own floor rather
                // than silently persisted as a level the wire can't represent.
                let level = agent_core::clamp_thinking_level(
                    &agent_core::capabilities(&self.current_model),
                    level,
                );
                let record_result = if level != self.current_level {
                    self.persistence
                        .record_thinking_level_change(level.as_str())
                } else {
                    Ok(())
                };
                match record_result {
                    Ok(()) => {
                        self.current_level = level;
                        self.current_thinking = None;
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
                        let (thinking, reasoning_effort) = agent_core::thinking_for_level(
                            &agent_core::capabilities(&self.current_model),
                            self.current_level,
                        );
                        if !self.emit(response(
                            id,
                            "set_reasoning_effort",
                            true,
                            Some(json!({
                                "level": self.current_level.as_str(),
                                "thinking": thinking,
                                "reasoning_effort": reasoning_effort.map(|e| e.as_str()),
                            })),
                            None,
                        )) {
                            return false;
                        }
                    }
                    Err(e) => {
                        if !self.emit(response(
                            id,
                            "set_reasoning_effort",
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
                    "set_reasoning_effort",
                    false,
                    None,
                    Some("`effort` must be one of off/minimal/low/medium/high/xhigh, or null"),
                )) {
                    return false;
                }
            }
        }
        true
    }

    pub(in crate::serve) fn cmd_cycle_model(&mut self, id: Option<String>) -> bool {
        // Advance through `cycle_models` (the `--models`-scoped list, or the full known-model
        // hint list when unscoped), wrapping — a quick way for a client to step through the
        // candidate list without needing its own copy of it. An id outside the list (a custom
        // `set_model` the client issued directly) wraps to the first entry, same as "not found".
        if self.cycle_models.is_empty() {
            return self.emit(response(
                id,
                "cycle_model",
                false,
                None,
                Some("no model to cycle to: this session's grant limits allow none of them"),
            ));
        }
        let next_idx = self
            .cycle_models
            .iter()
            .position(|m| m == &self.current_model)
            .map_or(0, |i| (i + 1) % self.cycle_models.len());
        let next_model = self.cycle_models[next_idx].clone();
        // A `pattern:level` pin from `--models` (e.g. `gpt-4o:high`) rides along whenever
        // cycling lands on that model — pi's own `ScopedModel.thinkingLevel`/
        // `_cycleScopedModel`. Landing on an unpinned model just keeps whatever level was
        // already active, same as pi.
        let pinned_level = self
            .scoped_models
            .iter()
            .find(|m| m.id == next_model)
            .and_then(|m| m.thinking_level);
        // See `set_model`'s identical re-derivation for why this must happen before anything
        // else, and why it's skipped when `next_model` (wrapping back around) is already the
        // active one.
        let new_client = if next_model == self.current_model {
            Ok(None)
        } else {
            build_gateway_client(&self.cfg, &next_model).map(|c| Some(Arc::new(c)))
        };
        match new_client {
            Ok(new_client) => {
                let record_result = if next_model != self.current_model {
                    self.persistence.record_model_change(&next_model)
                } else {
                    Ok(())
                };
                match record_result {
                    Ok(()) => {
                        self.session.scrub_cross_model_state(&next_model);
                        self.current_model = next_model;
                        if let Some(level) = pinned_level {
                            // Same staleness hazard `cycle_thinking_level` already guards
                            // against: a stale raw-budget override must not silently outlive
                            // the level it was pinned over.
                            self.current_thinking = None;
                            self.current_level = level;
                        }
                        // See `set_model`'s identical re-clamp for why this can't be skipped.
                        self.current_level = agent_core::clamp_thinking_level(
                            &agent_core::capabilities(&self.current_model),
                            self.current_level,
                        );
                        if let Some(new_client) = new_client {
                            self.client = new_client;
                        }
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
                        let mut resp_data =
                            model_switch_response(&self.current_model, self.current_level);
                        if let Value::Object(map) = &mut resp_data {
                            map.insert("scoped".to_string(), json!(!self.cfg.models.is_empty()));
                        }
                        if !self.emit(response(id, "cycle_model", true, Some(resp_data), None)) {
                            return false;
                        }
                    }
                    Err(e) => {
                        if !self.emit(response(
                            id,
                            "cycle_model",
                            false,
                            None,
                            Some(&e.to_string()),
                        )) {
                            return false;
                        }
                    }
                }
            }
            Err(e) => {
                if !self.emit(response(id, "cycle_model", false, None, Some(&e))) {
                    return false;
                }
            }
        }
        true
    }

    pub(in crate::serve) fn cmd_cycle_thinking_level(&mut self, id: Option<String>) -> bool {
        // Advance the portable Off/Minimal/Low/Medium/High/XHigh ladder, wrapping, and clear
        // any explicit raw-budget override (`set_thinking`) — otherwise a stale override could
        // silently mask the level having just changed. `thinking_for_level` reports what the
        // new level actually resolves to for `current_model` specifically, so the response
        // reflects reality rather than a number that may be meaningless for this model's shape.
        // Advances through only the levels `current_model` actually supports (see
        // `next_available_thinking_level`'s doc comment) rather than the raw 6-rung ladder —
        // a blind `.next()` would let the cycle land on (and durably record) an `Off` or
        // `xhigh` state the active model can't represent on the wire.
        let next_level = agent_core::next_available_thinking_level(
            &agent_core::capabilities(&self.current_model),
            self.current_level,
        );
        match self
            .persistence
            .record_thinking_level_change(next_level.as_str())
        {
            Ok(()) => {
                self.current_level = next_level;
                self.current_thinking = None;
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
                let (thinking, reasoning_effort) = agent_core::thinking_for_level(
                    &agent_core::capabilities(&self.current_model),
                    self.current_level,
                );
                if !self.emit(response(
                    id,
                    "cycle_thinking_level",
                    true,
                    Some(json!({
                        "level": self.current_level.as_str(),
                        "thinking": thinking,
                        "reasoning_effort": reasoning_effort.map(|e| e.as_str()),
                    })),
                    None,
                )) {
                    return false;
                }
            }
            Err(e) => {
                if !self.emit(response(
                    id,
                    "cycle_thinking_level",
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
}
