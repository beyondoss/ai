use crate::serve::*;

impl SessionLoop {
    pub(in crate::serve) fn cmd_export_html(&mut self, cmd: &Value, id: Option<String>) -> bool {
        let output_path = cmd.get("output_path").and_then(Value::as_str);
        let branches = self.persistence.abandoned_branches();
        let events = self.persistence.export_events();
        // `export_html_full` (Task #44 integration): this session's actual system prompt (the
        // same static-plus-dynamic-footer join every `prompt` sends — see `full_system`) and
        // tool set (`build_tools(&cfg)`, the same on-demand rebuild `has_read`/`bash_tool`
        // already call elsewhere in this function rather than keeping a long-lived registry
        // variable around), plus `session`'s own running token totals — previously omitted
        // entirely via the plainer `export_html_with_entries`.
        let system_prompt = full_system(&self.static_system, &self.cwd, &self.mcp_skills);
        let registry = build_tools(
            &self.cfg,
            self.cfg.image_auto_resize,
            &self.exec_cell,
            &self.mcp_enabled,
            &self.skills,
            None,
        );
        let mut tool_defs = registry.definitions();
        tool_defs.extend(registry.deferred_definitions());
        let usage = crate::export::UsageTotals {
            input_tokens: self.session.input_tokens,
            output_tokens: self.session.output_tokens,
            cache_read_tokens: self.session.cache_read_tokens,
            cache_write_tokens: self.session.cache_write_tokens,
        };
        // Service mode returns the document **inline**: writing a file would put a whole
        // transcript on the replica's disk (and hand the caller a host path to read it
        // back), and there is no per-tenant filesystem here to write it to instead. An
        // explicit `output_path` is refused rather than silently ignored.
        if self.mode.exports_inline() {
            if output_path.is_some() {
                if !self.emit(response(
                    id,
                    "export_html",
                    false,
                    None,
                    Some(
                        "refused in service mode: `output_path` writes to the replica's \
                          disk — omit it and read `data.html` from this response",
                    ),
                )) {
                    return false;
                }
                return true;
            }
            let html = crate::export::render_html_full(
                &self.persistence.meta,
                &self.session.messages,
                &branches,
                Some(usage),
                events,
                Some(&system_prompt),
                Some(&tool_defs),
            );
            if !self.emit(response(
                id,
                "export_html",
                true,
                Some(json!({ "html": html })),
                None,
            )) {
                return false;
            }
            return true;
        }
        match crate::export::export_html_full(
            &self.persistence.meta,
            &self.session.messages,
            &branches,
            Some(usage),
            events,
            Some(&system_prompt),
            Some(&tool_defs),
            output_path,
        ) {
            Ok(path) => {
                if !self.emit(response(
                    id,
                    "export_html",
                    true,
                    Some(json!({ "path": path.to_string_lossy() })),
                    None,
                )) {
                    return false;
                }
            }
            Err(e) => {
                if !self.emit(response(
                    id,
                    "export_html",
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
