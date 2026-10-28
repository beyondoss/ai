use crate::serve::*;

impl SessionLoop {
    pub(in crate::serve) async fn cmd_set_exec_endpoint(
        &mut self,
        cmd: &Value,
        id: Option<String>,
    ) -> bool {
        // Point *this session's* tools at an exec endpoint, or clear it back to the host.
        //
        // Per-session rather than per-process because that is what a multi-tenant server
        // needs: one machine per tenant, established when the session is set up and torn down
        // with it. It takes effect on the very next tool call — the tools read the cell rather
        // than capturing a target — so no registry rebuild is involved and none can be missed.
        let url = cmd.get("url").and_then(Value::as_str);
        let headers: Vec<String> = cmd
            .get("headers")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        let outcome = match url {
            Some(u) => crate::exec_endpoint::ExecTarget::http(
                u,
                &headers,
                self.cfg.exec_max_response_bytes,
            )
            .await
            .map(Some),
            // No url: detach, back to this host. Explicit rather than an error so a client
            // can hand a session back to local execution without tearing it down.
            None => Ok(None),
        };
        match outcome {
            Ok(target) => {
                let engine = target
                    .as_ref()
                    .map(|t| format!("{:?}", t.search_engine()))
                    .unwrap_or_else(|| "local".to_string());
                // Recorded on the session header so a resume reattaches rather than silently
                // falling back to the host — which for a sandboxed session would move the
                // tenant's work onto the server.
                self.persistence.set_exec_endpoint(if target.is_some() {
                    Some(json!({
                        "url": url,
                        "headers": headers,
                    }))
                } else {
                    None
                });
                self.exec_cell.set(target);
                if !self.emit(response(
                    id,
                    "set_exec_endpoint",
                    true,
                    Some(json!({ "attached": url.is_some(), "engine": engine })),
                    None,
                )) {
                    return false;
                }
            }
            Err(e) => {
                if !self.emit(response(id, "set_exec_endpoint", false, None, Some(&e))) {
                    return false;
                }
            }
        }
        true
    }

    // Reachable while idle only for a stale/duplicate answer (`accepted:false`): a real question
    // can only be outstanding while a run is in flight, which is the busy arm above.
    pub(in crate::serve) fn cmd_approve(&mut self, cmd: &Value, id: Option<String>) -> bool {
        if !self.emit(handle_approve(
            id,
            cmd,
            &self.pending_approvals,
            self.approval.is_some(),
        )) {
            return false;
        }
        true
    }

    pub(in crate::serve) async fn cmd_reload(&mut self, id: Option<String>) -> bool {
        // Re-run the full (expensive) discovery pi's `/reload` triggers: project
        // instructions, `SYSTEM.md`, skills, prompt templates and agent definitions may all
        // have changed since startup — and, on this host, so may trust (`agent trust`/
        // `--trust-project`). The per-turn `full_system` refresh alone only ever picks up the
        // cheap date/cwd footer.
        //
        // Allowed in service mode precisely because this re-walks the **tenant's** sandbox
        // through the same seam startup used, not the replica's filesystem. Trust plays no
        // part there: a sandbox's contents are the one tenant's own (see `resolve_project_trust`'s
        // startup call site).
        if self.mode.trust_gate_applies() {
            let has_gated_resources = crate::trust_store::has_trust_gated_resources(&self.cwd);
            self.project_trusted = resolve_project_trust(
                self.cfg.trust_project,
                self.cfg.force_untrusted,
                self.cfg.default_project_trust,
                crate::trust_store::TrustStore::open_default().lookup(&self.cwd),
                has_gated_resources,
            );
            // Track L32 (pi-parity fix): same warning as startup, above — trust may have just
            // changed to untrusted (or gated resources may have just appeared on disk) as of
            // this very `reload`, and an operator watching stderr deserves the same signal
            // they'd have gotten from a fresh `serve` invocation instead of silence.
            if !self.project_trusted && has_gated_resources {
                eprintln!(
                    "warning: {} has a project-local SYSTEM.md/APPEND_SYSTEM.md, skills, \
                     prompt templates, or a settings.json on disk, but the project isn't \
                     trusted, so they were skipped — pass --trust-project or run `{program} \
                     trust {}` to enable them (a project's own settings.json additionally \
                     requires a *persisted* `{program} trust`, not just a one-off \
                     --trust-project)",
                    self.cwd.display(),
                    self.cwd.display(),
                    program = crate::PROGRAM
                );
            }
        }
        let discovered = match self
            .mode
            .discover_sandbox_resources(&self.exec_cell, self.cfg.context_files)
            .await
        {
            // The tenant's sandbox, through the same seam startup used; what it found is
            // what the snapshot saves from now on.
            Some(found) => {
                self.sandbox_resources = Some(found.clone());
                self.sandbox_snapshot_dirty = true;
                self.sandbox_refresh = false;
                Discovered::from(found)
            }
            None => discover_resources(&self.cfg, &self.cwd, self.project_trusted).await,
        };
        self.apply_discovered(discovered).await;
        if !self.emit(response(id, "reload", true, None, None)) {
            return false;
        }
        true
    }

    pub(in crate::serve) fn cmd_set_auto_compaction(
        &mut self,
        cmd: &Value,
        id: Option<String>,
    ) -> bool {
        match cmd.get("enabled").and_then(Value::as_bool) {
            Some(enabled) => {
                self.current_auto_compaction = enabled;
                // Track L26 (pi-parity fix): previously mutated only this in-process local, so a
                // restarted `serve` (with neither `--no-compaction` nor its env var given) silently
                // reverted to the built-in enabled-by-default behavior — surviving a restart is the
                // whole point of `agent settings`' persisted-default tier (same as
                // `default_project_trust`). Best-effort: a failed write still applies for the rest
                // of *this* process (the in-memory toggle above already took effect), it just won't
                // survive a restart.
                persist_setting(&mut self.settings_store, "auto-compaction", |store| {
                    store.set_compaction_enabled(Some(enabled))
                });
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
                    "set_auto_compaction",
                    true,
                    Some(json!({ "auto_compaction": self.current_auto_compaction })),
                    None,
                )) {
                    return false;
                }
            }
            None => {
                if !self.emit(response(
                    id,
                    "set_auto_compaction",
                    false,
                    None,
                    Some("missing boolean `enabled`"),
                )) {
                    return false;
                }
            }
        }
        true
    }

    pub(in crate::serve) fn cmd_set_auto_retry(&mut self, cmd: &Value, id: Option<String>) -> bool {
        match cmd.get("enabled").and_then(Value::as_bool) {
            Some(enabled) => {
                self.current_auto_retry = enabled;
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
                    "set_auto_retry",
                    true,
                    Some(json!({ "auto_retry": self.current_auto_retry })),
                    None,
                )) {
                    return false;
                }
            }
            None => {
                if !self.emit(response(
                    id,
                    "set_auto_retry",
                    false,
                    None,
                    Some("missing boolean `enabled`"),
                )) {
                    return false;
                }
            }
        }
        true
    }

    // Pass 20 (pi-parity fix): `build_agent` already threaded `cfg.block_images` into
    // `Agent::with_block_images` (Task #34) and `cfg.image_auto_resize` into `build_tools`'s
    // registry (Task #34), but neither had a live RPC toggle the way `auto_compaction`/
    // `auto_retry` just above do — an operator could only change either by restarting `serve`
    // with a different `--block-images`/`--no-image-auto-resize` flag or persisted `agent
    // settings` default. Mirrors `set_auto_compaction`'s exact shape: mutate the runtime-mutable
    // local, persist best-effort (survives a restart, same as `set_auto_compaction`), rebuild
    // `agent` so the very next turn actually sees it, then ack over RPC.
    pub(in crate::serve) fn cmd_set_block_images(
        &mut self,
        cmd: &Value,
        id: Option<String>,
    ) -> bool {
        match cmd.get("enabled").and_then(Value::as_bool) {
            Some(enabled) => {
                self.current_block_images = enabled;
                persist_setting(&mut self.settings_store, "block-images", |store| {
                    store.set_block_images(Some(enabled))
                });
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
                    "set_block_images",
                    true,
                    Some(json!({ "block_images": self.current_block_images })),
                    None,
                )) {
                    return false;
                }
            }
            None => {
                if !self.emit(response(
                    id,
                    "set_block_images",
                    false,
                    None,
                    Some("missing boolean `enabled`"),
                )) {
                    return false;
                }
            }
        }
        true
    }

    // Same shape as `set_block_images` just above, for `image_auto_resize`.
    pub(in crate::serve) fn cmd_set_image_auto_resize(
        &mut self,
        cmd: &Value,
        id: Option<String>,
    ) -> bool {
        match cmd.get("enabled").and_then(Value::as_bool) {
            Some(enabled) => {
                self.current_image_auto_resize = enabled;
                persist_setting(&mut self.settings_store, "image-auto-resize", |store| {
                    store.set_image_auto_resize(Some(enabled))
                });
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
                    "set_image_auto_resize",
                    true,
                    Some(json!({ "image_auto_resize": self.current_image_auto_resize })),
                    None,
                )) {
                    return false;
                }
            }
            None => {
                if !self.emit(response(
                    id,
                    "set_image_auto_resize",
                    false,
                    None,
                    Some("missing boolean `enabled`"),
                )) {
                    return false;
                }
            }
        }
        true
    }

    // Toggle how much of the steer lane a single mid-run drain point consumes (see
    // `agent_core::QueueMode`) — pi's default is `"one_at_a_time"`, one queued message per
    // drain, leaving the rest queued for the next one; `"all"` folds everything queued into a
    // single injection (this crate's original behavior). Independent of `set_follow_up_mode`
    // below — pi carries the same split (`steeringMode`/`followUpMode`). Owned entirely by the
    // `Steering` handle itself (shared with the loop, no `Agent` rebuild needed), so this takes
    // effect immediately, including mid-run.
    pub(in crate::serve) fn cmd_set_steering_mode(
        &mut self,
        cmd: &Value,
        id: Option<String>,
    ) -> bool {
        match cmd.get("mode").and_then(Value::as_str) {
            Some(mode @ ("one_at_a_time" | "all")) => {
                self.steering.set_steering_mode(if mode == "all" {
                    agent_core::QueueMode::All
                } else {
                    agent_core::QueueMode::OneAtATime
                });
                // Same "best-effort, in-memory toggle already took effect either way" persistence
                // `set_auto_compaction` above uses — a failed write just won't survive a restart.
                // See `settings::Settings::steering_mode`'s own doc comment: this RPC handler is
                // exactly where that persistence is expected to happen.
                persist_setting(&mut self.settings_store, "steering_mode", |store| {
                    store.set_steering_mode(Some(mode.to_string()))
                });
                if !self.emit(response(
                    id,
                    "set_steering_mode",
                    true,
                    Some(json!({ "mode": mode })),
                    None,
                )) {
                    return false;
                }
            }
            _ => {
                if !self.emit(response(
                    id,
                    "set_steering_mode",
                    false,
                    None,
                    Some("missing `mode`; expected \"one_at_a_time\" or \"all\""),
                )) {
                    return false;
                }
            }
        }
        true
    }

    // Same idea, for the follow-up lane drained at a stop boundary (plus any stranded steer
    // messages swept in there — see `Steering::drain_at_stop`).
    pub(in crate::serve) fn cmd_set_follow_up_mode(
        &mut self,
        cmd: &Value,
        id: Option<String>,
    ) -> bool {
        match cmd.get("mode").and_then(Value::as_str) {
            Some(mode @ ("one_at_a_time" | "all")) => {
                self.steering.set_follow_up_mode(if mode == "all" {
                    agent_core::QueueMode::All
                } else {
                    agent_core::QueueMode::OneAtATime
                });
                persist_setting(&mut self.settings_store, "follow_up_mode", |store| {
                    store.set_follow_up_mode(Some(mode.to_string()))
                });
                if !self.emit(response(
                    id,
                    "set_follow_up_mode",
                    true,
                    Some(json!({ "mode": mode })),
                    None,
                )) {
                    return false;
                }
            }
            _ => {
                if !self.emit(response(
                    id,
                    "set_follow_up_mode",
                    false,
                    None,
                    Some("missing `mode`; expected \"one_at_a_time\" or \"all\""),
                )) {
                    return false;
                }
            }
        }
        true
    }
}
