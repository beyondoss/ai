use crate::serve::*;

impl SessionLoop {
    pub(in crate::serve) async fn cmd_login(&mut self, cmd: &Value, id: Option<String>) -> bool {
        let provider_id = cmd
            .get("provider")
            .and_then(Value::as_str)
            .and_then(crate::oauth::OAuthProviderId::parse);
        match provider_id {
            None => {
                if !self.emit(response(
                    id,
                    "login",
                    false,
                    None,
                    Some("missing or unknown `provider`"),
                )) {
                    return false;
                }
            }
            Some(_) if lock_ignoring_poison(&self.pending_login).is_some() => {
                if !self.emit(response(
                    id,
                    "login",
                    false,
                    None,
                    Some(
                        "busy: a login is already in flight; use `abort_login` to cancel it \
                         first",
                    ),
                )) {
                    return false;
                }
            }
            Some(provider_id) => {
                if !self.emit(ack(id.clone(), "login")) {
                    return false;
                }
                // Runs as a detached task, not inline like `prompt`'s own busy loop: unlike a
                // model turn, a login has nothing to do with `session`/`agent` state, so there's
                // no reason every *other* idle command (`list_sessions`, `logout` of a different
                // provider, …) should have to wait out however long a user takes to finish a
                // browser flow. Only a second concurrent `login` is rejected (above).
                let login_cancel = agent_core::CancellationToken::new();
                let pending_code: PendingCodeSlot = Arc::new(std::sync::Mutex::new(None));
                self.login_generation += 1;
                let generation = self.login_generation;
                *lock_ignoring_poison(&self.pending_login) = Some(PendingLogin {
                    generation,
                    cancel: login_cancel.clone(),
                    pending_code: pending_code.clone(),
                });
                let out_tx_bg = self.out_tx.clone();
                let pending_login_bg = self.pending_login.clone();
                let login_id = id.clone();
                tokio::spawn(async move {
                    // Always clears `pending_login` when this task ends, even on panic — but only
                    // if the slot still holds *this* login (see `PendingLoginGuard`'s doc comment).
                    let _reset_pending_login_on_exit = PendingLoginGuard {
                        slot: pending_login_bg.clone(),
                        generation,
                    };
                    let callbacks = ServeLoginCallbacks {
                        out_tx: out_tx_bg.clone(),
                        id: login_id.clone(),
                        provider: provider_id,
                        pending_code,
                        cancel: login_cancel.clone(),
                    };
                    let result = crate::oauth::login(provider_id, &callbacks, &login_cancel).await;
                    let frame = match result {
                        Ok(credential) => {
                            // `AuthStore::set`'s cross-process `FileLock` can block briefly under
                            // contention — kept off this task's own async context the same way
                            // `persist_blocking` keeps session persistence off the caller's.
                            let saved = tokio::task::spawn_blocking(move || {
                                crate::auth_store::AuthStore::open_default()
                                    .set(provider_id.store_key(), credential)
                            })
                            .await;
                            match saved {
                                Ok(Ok(())) => response(
                                    login_id,
                                    "login",
                                    true,
                                    Some(json!({
                                        "provider": provider_id.to_string(),
                                        "status": "logged_in",
                                    })),
                                    None,
                                ),
                                Ok(Err(e)) => response(
                                    login_id,
                                    "login",
                                    false,
                                    None,
                                    Some(&format!("logged in but failed to save credential: {e}")),
                                ),
                                Err(e) => response(
                                    login_id,
                                    "login",
                                    false,
                                    None,
                                    Some(&format!("logged in but failed to save credential: {e}")),
                                ),
                            }
                        }
                        Err(e) => response(login_id, "login", false, None, Some(&e.to_string())),
                    };
                    let _ = out_tx_bg.send(frame);
                    // `_reset_pending_login_on_exit`'s `Drop` clears `pending_login` here.
                });
            }
        }
        true
    }

    pub(in crate::serve) fn cmd_submit_code(&mut self, cmd: &Value, id: Option<String>) -> bool {
        let code = cmd.get("code").and_then(Value::as_str).map(str::to_string);
        let accepted = match (code, lock_ignoring_poison(&self.pending_login).as_ref()) {
            (Some(code), Some(p)) => match lock_ignoring_poison(&p.pending_code).take() {
                Some(tx) => tx.send(code).is_ok(),
                None => false,
            },
            _ => false,
        };
        if !self.emit(response(
            id,
            "submit_code",
            true,
            Some(json!({ "accepted": accepted })),
            None,
        )) {
            return false;
        }
        true
    }

    pub(in crate::serve) fn cmd_abort_login(&mut self, id: Option<String>) -> bool {
        // Idempotent no-op if none is in flight — matches `abort`'s own idle-mode convention.
        // `take`, not `as_ref`: clear the slot *synchronously* here rather than leaving it for the
        // detached task's `Drop` guard, so the very next `login` is accepted the instant this abort
        // is acknowledged. Waiting for the guard raced the next command — a `login` arriving before
        // the cancelled task wound down was spuriously rejected as "already in flight", hanging a
        // client that (reasonably) expected `abort_login`'s success to mean the slot was free. The
        // cancelled task still sends its own terminal `login` response; clearing the slot only frees
        // the next login, and the guard's generation check keeps that task from wiping the successor.
        if let Some(p) = lock_ignoring_poison(&self.pending_login).take() {
            p.cancel.cancel();
        }
        if !self.emit(response(id, "abort_login", true, None, None)) {
            return false;
        }
        true
    }

    pub(in crate::serve) fn cmd_logout(&mut self, cmd: &Value, id: Option<String>) -> bool {
        let provider_id = cmd
            .get("provider")
            .and_then(Value::as_str)
            .and_then(crate::oauth::OAuthProviderId::parse);
        match provider_id {
            None => {
                if !self.emit(response(
                    id,
                    "logout",
                    false,
                    None,
                    Some("missing or unknown `provider`"),
                )) {
                    return false;
                }
            }
            Some(provider_id) => {
                let mut store = crate::auth_store::AuthStore::open_default();
                match store.remove(provider_id.store_key()) {
                    Ok(was_logged_in) => {
                        if !self.emit(response(
                            id,
                            "logout",
                            true,
                            Some(json!({
                                "provider": provider_id.to_string(),
                                "was_logged_in": was_logged_in,
                            })),
                            None,
                        )) {
                            return false;
                        }
                    }
                    Err(e) => {
                        if !self.emit(response(id, "logout", false, None, Some(&e.to_string()))) {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    pub(in crate::serve) fn cmd_auth_status(&mut self, cmd: &Value, id: Option<String>) -> bool {
        let store = crate::auth_store::AuthStore::open_default();
        let status_of = |pid: crate::oauth::OAuthProviderId| {
            let status = match store.get(pid.store_key()) {
                None => "logged_out",
                Some(stored) if stored.last_refresh_error.is_some() => "needs_reauth",
                Some(_) => "logged_in",
            };
            json!({ "provider": pid.to_string(), "status": status })
        };
        match cmd.get("provider").and_then(Value::as_str) {
            Some(p) => match crate::oauth::OAuthProviderId::parse(p) {
                Some(pid) => {
                    if !self.emit(response(
                        id,
                        "auth_status",
                        true,
                        Some(status_of(pid)),
                        None,
                    )) {
                        return false;
                    }
                }
                None => {
                    if !self.emit(response(
                        id,
                        "auth_status",
                        false,
                        None,
                        Some("unknown `provider`"),
                    )) {
                        return false;
                    }
                }
            },
            None => {
                let providers: Vec<Value> = crate::oauth::OAuthProviderId::all()
                    .iter()
                    .map(|&pid| status_of(pid))
                    .collect();
                if !self.emit(response(
                    id,
                    "auth_status",
                    true,
                    Some(json!({ "providers": providers })),
                    None,
                )) {
                    return false;
                }
            }
        }
        true
    }
}
