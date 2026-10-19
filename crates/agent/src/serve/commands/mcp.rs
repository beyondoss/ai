use crate::serve::*;

impl SessionLoop {
    pub(in crate::serve) fn cmd_set_mcp_enabled(
        &mut self,
        cmd: &Value,
        id: Option<String>,
    ) -> bool {
        // Session-scoped MCP kit: which configured servers' tools are advertised.
        //
        // - omit `servers` / `servers: null` → all configured servers enabled (default)
        // - `servers: []` → none (MCP tools gone from the registry)
        // - `servers: ["a","b"]` → only those names
        //
        // Unknown names are rejected so a typo cannot silently empty the kit. Takes effect on
        // the next model turn via an `Agent` rebuild (idle) — same shape as `set_model`.
        let configured = crate::tools::mcp::server_names_from_tools(&self.cfg.mcp_tools.snapshot());
        let outcome = match cmd.get("servers") {
            None | Some(Value::Null) => {
                self.mcp_enabled.set(None);
                Ok(json!({
                    "mode": "all",
                    "enabled": configured.clone(),
                    "configured": configured,
                }))
            }
            Some(Value::Array(arr)) => {
                let mut names = std::collections::HashSet::new();
                let mut unknown = Vec::new();
                let mut bad_entry = false;
                for v in arr {
                    let Some(name) = v.as_str() else {
                        bad_entry = true;
                        break;
                    };
                    if configured.iter().any(|c| c == name) {
                        names.insert(name.to_string());
                    } else {
                        unknown.push(name.to_string());
                    }
                }
                if bad_entry {
                    Err("`servers` entries must be strings".into())
                } else if !unknown.is_empty() {
                    Err(format!(
                        "unknown MCP server(s): {}; configured: {}",
                        unknown.join(", "),
                        if configured.is_empty() {
                            "(none)".into()
                        } else {
                            configured.join(", ")
                        }
                    ))
                } else {
                    let enabled: Vec<String> = {
                        let mut v: Vec<_> = names.iter().cloned().collect();
                        v.sort();
                        v
                    };
                    self.mcp_enabled.set(Some(names));
                    Ok(json!({
                        "mode": "allowlist",
                        "enabled": enabled,
                        "configured": configured,
                    }))
                }
            }
            Some(_) => Err("`servers` must be an array of server names, or null".into()),
        };
        match outcome {
            Ok(data) => {
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
                if !self.emit(response(id, "set_mcp_enabled", true, Some(data), None)) {
                    return false;
                }
            }
            Err(e) => {
                if !self.emit(response(id, "set_mcp_enabled", false, None, Some(&e))) {
                    return false;
                }
            }
        }
        true
    }

    pub(in crate::serve) fn cmd_get_mcp(&mut self, id: Option<String>) -> bool {
        let configured = crate::tools::mcp::server_names_from_tools(&self.cfg.mcp_tools.snapshot());
        let snapshot = self.mcp_enabled.snapshot();
        let (mode, enabled) = match &snapshot {
            None => ("all".to_string(), configured.clone()),
            Some(set) => {
                let mut v: Vec<_> = set.iter().cloned().collect();
                v.sort();
                ("allowlist".to_string(), v)
            }
        };
        // `names`, not `definitions`: a tool deferred behind `tool_search` is still loaded.
        let tools: Vec<String> = build_tools(
            &self.cfg,
            self.cfg.image_auto_resize,
            &self.exec_cell,
            &self.mcp_enabled,
            &self.skills,
            None,
        )
        .names()
        .into_iter()
        .filter(|n| n.starts_with("mcp__"))
        .collect();
        let catalogs = self.cfg.mcp_catalog.filter_enabled(&self.mcp_enabled);
        let resources: Vec<_> = catalogs
            .iter()
            .flat_map(|s| s.resources.iter().cloned())
            .collect();
        let prompts: Vec<_> = catalogs
            .iter()
            .flat_map(|s| s.prompts.iter().cloned())
            .collect();
        let servers: Vec<_> = catalogs
            .iter()
            .map(|s| {
                json!({
                    "name": s.name,
                    "protocol_version": s.protocol_version,
                    "resources": s.resources.len(),
                    "prompts": s.prompts.len(),
                })
            })
            .collect();
        if !self.emit(response(
            id,
            "get_mcp",
            true,
            Some(json!({
                "mode": mode,
                "enabled": enabled,
                "configured": configured,
                "tools": tools,
                "resources": resources,
                "prompts": prompts,
                "servers": servers,
                "apps": self.mcp_enabled.apps().map(|view| json!({
                    "spec": crate::tools::mcp_apps::SPEC_REVISION,
                    "servers": view.servers(),
                })),
            })),
            None,
        )) {
            return false;
        }
        true
    }

    pub(in crate::serve) async fn cmd_mcp_complete(
        &mut self,
        cmd: &Value,
        id: Option<String>,
    ) -> bool {
        let Some(server) = cmd.get("server").and_then(Value::as_str) else {
            if !self.emit(response(
                id,
                "mcp_complete",
                false,
                None,
                Some("missing `server`"),
            )) {
                return false;
            }
            return true;
        };
        let Some(params_val) = cmd.get("params") else {
            if !self.emit(response(
                id,
                "mcp_complete",
                false,
                None,
                Some("missing `params` (CompleteRequestParams JSON)"),
            )) {
                return false;
            }
            return true;
        };
        let params: rmcp::model::CompleteRequestParams =
            match serde_json::from_value(params_val.clone()) {
                Ok(p) => p,
                Err(e) => {
                    if !self.emit(response(
                        id,
                        "mcp_complete",
                        false,
                        None,
                        Some(&format!("invalid `params`: {e}")),
                    )) {
                        return false;
                    }
                    return true;
                }
            };
        match self.cfg.mcp_catalog.complete(server, params).await {
            Ok(result) => {
                if !self.emit(response(
                    id,
                    "mcp_complete",
                    true,
                    Some(serde_json::to_value(result).unwrap_or(Value::Null)),
                    None,
                )) {
                    return false;
                }
            }
            Err(e) => {
                if !self.emit(response(id, "mcp_complete", false, None, Some(&e))) {
                    return false;
                }
            }
        }
        true
    }

    pub(in crate::serve) async fn cmd_set_mcp_apps(
        &mut self,
        cmd: &Value,
        id: Option<String>,
    ) -> bool {
        // A client declares (or withdraws) that it renders MCP Apps for this session. The
        // tool set changes, so — like `set_mcp_enabled` — the agent is rebuilt. Declaring
        // answers once the apps connections have dialed, from a task of its own: the
        // command loop never waits on a dial, and until it lands the session keeps its
        // plain tools (a prompt rebuilds onto the apps flavor when it does).
        match cmd.get("enabled").and_then(Value::as_bool) {
            Some(false) => {
                let mut epoch = lock_ignoring_poison(&self.apps_epoch);
                *epoch += 1;
                crate::tools::mcp_apps::close(&self.mcp_enabled, "mcp_apps_disabled");
                lock_ignoring_poison(&self.out_conn).clear_app_views();
                self.apps_generation = None;
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
                    "set_mcp_apps",
                    true,
                    Some(json!({ "enabled": false })),
                    None,
                )) {
                    return false;
                }
                drop(epoch);
            }
            Some(true) => match crate::tools::mcp_apps::configure(
                self.cfg.mcp_apps.as_ref(),
                &self.mcp_enabled,
                cmd,
                self.cfg.no_tools,
                self.app_sink.clone(),
                self.app_redact.clone(),
                lock_ignoring_poison(&self.out_conn).app_view_servers(),
            ) {
                Ok(pool) => {
                    self.apps_generation = self.mcp_enabled.apps().map(|view| view.generation());
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
                    let out_tx = self.out_tx.clone();
                    let mode = self.mode.clone();
                    let epoch = self.apps_epoch.clone();
                    let mine = {
                        let mut e = lock_ignoring_poison(&self.apps_epoch);
                        *e += 1;
                        *e
                    };
                    tokio::spawn(async move {
                        let status = pool.ensure().await;
                        let current = lock_ignoring_poison(&epoch);
                        let frame = if *current == mine {
                            response(
                                id,
                                "set_mcp_apps",
                                true,
                                Some(crate::tools::mcp_apps::configured(status)),
                                None,
                            )
                        } else {
                            response(
                                id,
                                "set_mcp_apps",
                                false,
                                None,
                                Some(
                                    "superseded: a later set_mcp_apps (or session switch) \
                                     replaced this declaration",
                                ),
                            )
                        };
                        let _ = out_tx.send(redact_frame(&mode, frame));
                        drop(current);
                    });
                }
                Err(e) => {
                    if !self.emit(response(id, "set_mcp_apps", false, None, Some(&e))) {
                        return false;
                    }
                }
            },
            None => {
                if !self.emit(response(
                    id,
                    "set_mcp_apps",
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
}
