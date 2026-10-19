use crate::serve::*;

impl SessionLoop {
    pub(in crate::serve) async fn cmd_bash(&mut self, cmd: &Value, id: Option<String>) -> bool {
        let Some(command) = cmd.get("command").and_then(Value::as_str) else {
            if !self.emit(response(id, "bash", false, None, Some("missing `command`"))) {
                return false;
            }
            return true;
        };
        let Some(tool) = self.bash_tool.clone() else {
            if !self.emit(response(
                id,
                "bash",
                false,
                None,
                Some(
                    "the `bash` tool is not registered for this process \
                     (excluded via --exclude-tools/--no-tools)",
                ),
            )) {
                return false;
            }
            return true;
        };
        // Nobody reads the client's table for notices, so its ended jobs are dropped here
        // rather than kept for a `bash_job` that cannot reach them.
        self.host_bash_jobs.forget_ended();
        let mut input = json!({ "command": command });
        if let Some(cwd) = cmd.get("cwd").and_then(Value::as_str) {
            input["cwd"] = json!(cwd);
        }
        // The client's own bound on this command, enforced here as a cancel. Not handed to
        // the tool: its `timeout_ms` only decides when a model's call returns (the command
        // keeps running as a job), and this instance waits for exit — the client is the one
        // waiting, and `abort_bash` or this deadline is how it gives up.
        let deadline = cmd
            .get("timeout_ms")
            .and_then(Value::as_u64)
            .map(std::time::Duration::from_millis);
        // Track L23 (pi-parity fix): this host command used to call `tool.run_streaming`
        // directly, bypassing `--deny-tool`/`--deny-bash-pattern`/`--deny-path` entirely (only
        // `--exclude-tools bash` — the `bash_tool` check above — actually gated it). Checked
        // before the `ToolStart` event fires, matching the "not registered" early-return just
        // above: a blocked call never starts, so it never needs a `tool_end` to close it out.
        if let Some(reason) = self
            .bash_policy
            .before_tool_call("bash", &input, &self.session, &CancellationToken::new())
            .await
        {
            if !self.emit(response(id, "bash", false, None, Some(&reason))) {
                return false;
            }
            return true;
        }
        // Always recorded into `session`/persisted storage/`export.rs`'s rendered output — pi's
        // `recordBashResult` (`agent-session.ts`) unconditionally records too. `exclude_from_context`
        // (Fix 9, pi-parity gap: previously misdescribed here as gating *recording* itself, and
        // implemented that way too — see the `session.push`/`persist` call below, now unconditional)
        // instead gates only the separate transform that builds what's actually sent to the model
        // on its *next* turn (`ServeHooks::before_provider_request`) — pi's identical two-layer split
        // (`recordBashResult` vs. `convertToLlm`, `messages.ts`). So a diagnostic command run outside
        // the model's own turn is always visible in history; only whether the model itself sees it
        // next turn is what this flag controls.
        let exclude_from_context = cmd
            .get("exclude_from_context")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let command = command.to_string();
        // Busy for the reaper and drain until the result is persisted and answered (see
        // `BusyGuard`): a detached client's command finishes rather than being cancelled.
        let _busy = BusyGuard::mark(&self.running);
        self.host_bash_seq += 1;
        let call_id = format!("host-bash-{}", self.host_bash_seq);
        if let Some(frame) = event_frame(AgentEvent::ToolStart {
            not_run: false,
            id: call_id.clone(),
            name: "bash".to_string(),
            input: input.clone(),
        }) && !self.emit(frame)
        {
            return false;
        }

        // Independent of the model loop and the session — runs directly against the
        // registered `bash` tool, reusing its own streaming/truncation/cancellation
        // machinery. Races the command against continued stdin reads so `abort_bash`/`abort`
        // (or shutdown) can cancel it; any other command is rejected as busy, the same shape
        // as the `prompt` arm's own busy-loop, simplified with no session/steering involved.
        let cancel = CancellationToken::new();
        let (prog_tx, mut prog_rx) = futures::channel::mpsc::unbounded::<ToolUpdate>();
        let progress = agent_core::ToolProgress::new(
            prog_tx,
            call_id.clone(),
            "bash".to_string(),
            cancel.clone(),
        );
        let mut stdin_open = true;
        // Fix 7 (pi-parity gap): `tools::bash`'s own `truncation_details` (the `{truncation,
        // full_output_path}` payload — see that module's doc comment) only ever streams on an
        // interim `ToolUpdate::Progress`, never on the terminal outcome — captured here so the
        // response built below can carry it as its own structured field instead of only ever
        // reaching a client via a progress event that a client polling just for the final
        // result never sees at all.
        let mut last_bash_details: Option<Value> = None;
        let mut deadline_hit = false;
        let outcome = {
            let run = tool.run_streaming(input, &progress);
            tokio::pin!(run);
            let deadline_sleep = sleep_opt(deadline);
            tokio::pin!(deadline_sleep);
            loop {
                tokio::select! {
                    biased;
                    r = &mut run => break r,
                    () = &mut deadline_sleep, if !deadline_hit => {
                        deadline_hit = true;
                        cancel.cancel();
                    }
                    // `cancel.cancel()` alone (set by `abort_bash`/`abort` below, or by
                    // shutdown) doesn't, by itself, interrupt an in-flight `run` — but `bash`'s
                    // own `exec` races the same token internally (see `tools::bash`) and returns
                    // a graceful "Command cancelled" error carrying whatever output had already
                    // accumulated, so in practice `r = &mut run` above resolves first and wins
                    // this biased select on a real cancellation. This arm stays as a fallback for
                    // a future tool reachable from this same host-command path that *isn't*
                    // internally cancellation-aware — breaking here drops the pinned `run`
                    // future, which still kills a `bash` subprocess via its
                    // `kill_on_drop`/process-group guard even if that tool never noticed.
                    () = cancel.cancelled() => {
                        break Err(agent_core::ToolError::Execution("cancelled".to_string()));
                    }
                    sig = self.shutdown.wait() => {
                        self.shutdown_cause = Some(sig);
                        stdin_open = false;
                        cancel.cancel();
                    }
                    update = prog_rx.next() => {
                        if let Some(ToolUpdate::Progress { id, name, snapshot, details }) = update {
                            if details.is_some() {
                                last_bash_details = details.clone();
                            }
                            if let Some(frame) = event_frame(AgentEvent::ToolProgress { id, name, snapshot, details }) {
                                let _ = self.out_tx.send(frame);
                            }
                        }
                    }
                    maybe_line = self.input_rx.recv(), if stdin_open => match maybe_line {
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
                                "abort_bash" | "abort" => {
                                    cancel.cancel();
                                    let _ = self.out_tx.send(response(cid, "abort_bash", true, None, None));
                                }
                                "prompt" if crate::tools::mcp_events::is_injection(&c) => {
                                    self.deferred_events.push_back(l.to_string());
                                }
                                events_cmd if events_cmd.starts_with("mcp_events_") => dispatch_events(&self.events_cmds, &self.out_tx, cid, events_cmd, &c),
                                other => {
                                    let _ = self.out_tx.send(response(cid, other, false, None, Some("busy: a host bash command is running; only `abort_bash`/`abort` are accepted")));
                                }
                            }
                        }
                        None => {
                            stdin_open = false;
                            cancel.cancel();
                        }
                    }
                }
            }
        };
        // Flush anything buffered between the final poll above and the tool's own return.
        while let Ok(update) = prog_rx.try_recv() {
            if let ToolUpdate::Progress {
                id,
                name,
                snapshot,
                details,
            } = update
            {
                if details.is_some() {
                    last_bash_details = details.clone();
                }
                if let Some(frame) = event_frame(AgentEvent::ToolProgress {
                    id,
                    name,
                    snapshot,
                    details,
                }) {
                    let _ = self.out_tx.send(frame);
                }
            }
        }

        let (result_text, is_error) = match outcome {
            Ok(output) => (output.text, false),
            Err(e) => (e.to_string(), true),
        };
        // Cancelled by the client's deadline rather than an abort: say which.
        let result_text = match deadline {
            Some(d) if deadline_hit && is_error => {
                match result_text.strip_suffix("Command cancelled") {
                    Some(head) => {
                        format!(
                            "{head}Command timed out after {}",
                            crate::tools::format_duration(d)
                        )
                    }
                    None => result_text,
                }
            }
            _ => result_text,
        };
        // Fix 7 (pi-parity gap): pi's own `BashResult{output, exitCode, cancelled, truncated,
        // fullOutputPath}` (`bash-executor.ts`) alongside the text, not flattened into it — see
        // `bash_exit_code_from_status_line`/`bash_result_was_cancelled`'s own doc comments for
        // how each is recovered from what `tools::bash` already reports (its own status-line
        // text, and the last progress `details` captured above).
        let exit_code = bash_exit_code_from_status_line(&result_text, is_error);
        let cancelled = bash_result_was_cancelled(&result_text, is_error);
        let truncated = last_bash_details
            .as_ref()
            .and_then(|d| d.get("truncation"))
            .and_then(|t| t.get("truncated"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let full_output_path = last_bash_details
            .as_ref()
            .and_then(|d| d.get("full_output_path"))
            .and_then(Value::as_str)
            .map(str::to_string);
        // Fix 3 (pi-parity gap): thread these same four fields into the *persisted* message
        // too, as a leading status line `export.rs`'s host-bash parser now understands — see
        // `HOST_BASH_STATUS_LINE_PREFIX`'s own doc comment for why this rides on the marker
        // text rather than a new `agent_core::Message` field. Previously only a bare `is_error`
        // bool (the `"(error)\n"` marker below) ever reached the persisted message, even though
        // this RPC response has always reported the real exit code/cancelled/truncated/
        // full-output-path live, a few lines down.
        let status_line = format!(
            "{HOST_BASH_STATUS_LINE_PREFIX}{}\n",
            json!({
                "exit_code": exit_code,
                "cancelled": cancelled,
                "truncated": truncated,
                "full_output_path": full_output_path,
            })
        );
        if let Some(frame) = event_frame(AgentEvent::ToolEnd {
            id: call_id,
            name: "bash".to_string(),
            result: result_text.clone(),
            is_error,
        }) && !self.emit(frame)
        {
            return false;
        }
        // This command has no `tool_use` to pair a `tool_result` against (it never went
        // through the model), so it's recorded as a plainly-labeled informational user message
        // instead — structurally impossible to run this arm while a turn is streaming (the
        // busy-loop above rejects everything but `abort_bash`/`abort` while a host bash command
        // is in flight, and this command is only reachable from the idle loop to begin with), so
        // there's no tool_use/tool_result ordering to protect against, unlike pi's own
        // stream-in-flight deferral.
        //
        // Fix 9 (pi-parity gap): always pushed and persisted now, regardless of
        // `exclude_from_context` — pi's own `recordBashResult` (`agent-session.ts`) unconditionally
        // records into `agent.state.messages` too; `excludeFromContext` is consulted only later, in
        // the separate `convertToLlm` transform that builds what's actually sent to the model
        // (`messages.ts`). This used to skip recording entirely when the flag was set, losing the
        // command/output outright instead of merely hiding it from the model — it never showed up
        // in `session.messages`, persisted storage, or `export.rs`'s rendered output either.
        // `HOST_BASH_EXCLUDED_LABEL` is this crate's stand-in for pi's dedicated
        // `BashExecutionMessage.excludeFromContext` field (`agent_core::Message` has no such field —
        // see the label's own doc comment); `ServeHooks::before_provider_request`, installed on
        // every `build_agent` agent, is the "what's actually sent to the model" transform that
        // consults it.
        let label = if exclude_from_context {
            HOST_BASH_EXCLUDED_LABEL
        } else {
            HOST_BASH_LABEL
        };
        self.session.push(agent_core::Message::user(format!(
            "{label}\n$ {command}\n\n{status_line}{}{result_text}",
            if is_error { "(error)\n" } else { "" }
        )));
        let (p, r) =
            persist_blocking(self.persistence.take_for_blocking(), self.session.clone()).await;
        self.persistence = p;
        if let Err(e) = r {
            eprintln!("serve: failed to persist host bash result: {e}");
        }
        if !self.emit(response(
            id,
            "bash",
            true,
            Some(json!({
                "result": result_text,
                "is_error": is_error,
                "exit_code": exit_code,
                "cancelled": cancelled,
                "truncated": truncated,
                "full_output_path": full_output_path,
            })),
            None,
        )) {
            return false;
        }
        if !stdin_open {
            return false;
        }
        true
    }

    pub(in crate::serve) fn cmd_abort_bash(&mut self, id: Option<String>) -> bool {
        // No host bash is in flight (a running one is handled inside the `bash` arm above), so
        // there is nothing to cancel — acknowledge idempotently, matching `abort`'s idle shape.
        if !self.emit(response(id, "abort_bash", true, None, None)) {
            return false;
        }
        true
    }
}
