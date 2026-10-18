//! The `ProxyHttp` hooks, in the order pingora calls them.

use super::*;

#[async_trait]
impl ProxyHttp for AiProxy {
    type CTX = Ctx;

    fn new_ctx(&self) -> Self::CTX {
        self.state.request_began();
        Ctx {
            rc: None,
            held: Held {
                state: self.state,
                request_id: None,
                in_flight: false,
                active_stream: false,
                tenant: None,
                body_bytes: 0,
                body_exempt: false,
                client_left: AtomicBool::new(false),
                drain: Drain::Off,
            },
        }
    }

    /// Pingora calls this with `DownstreamCache` exactly when it ignores a downstream error to keep
    /// reading the upstream: on this gateway, only for a stream [`drain::arm`]ed it. That is the
    /// client leaving a drainable stream; `response_body_filter` takes it from there. Its warn
    /// line is replaced by `ai_usage_drains_in_flight` / `ai_usage_drains_total`.
    fn suppress_proxy_warn_log(
        &self,
        _session: &Session,
        ctx: &Self::CTX,
        _error: &pingora_core::Error,
        context: ProxyWarnLogContext,
    ) -> bool {
        if matches!(context, ProxyWarnLogContext::DownstreamCache) {
            ctx.held.client_left();
            return true;
        }
        false
    }

    /// The request line pingora prints on its own error lines. Its default prints the path
    /// **with** the query, which is where a Gemini-style `?key=` credential (a virtual key or a
    /// BYO Google key) lives. Logged without the query.
    fn request_summary(&self, session: &Session, _ctx: &Self::CTX) -> String {
        summary_line(session.req_header())
    }

    fn allow_spawning_subrequest(&self, _session: &Session, _ctx: &Self::CTX) -> bool {
        // Needed for `relay_full_body`, which is the only place the gateway spawns one.
        true
    }

    /// Admission: every check and refusal before the first upstream byte, as the phases in
    /// `phases` (see its table). A refusal any of them returns is written here, so each is
    /// counted, logged and answered the same way.
    async fn request_filter(&self, session: &mut Session, ctx: &mut Self::CTX) -> Result<bool> {
        let arrival = self.arrive(session, &mut ctx.held);
        // The tenant slot a full body read takes: owned here, so a refusal written after the
        // read still holds it, and moved into the request context on admission.
        let mut early_slot = None;
        match self
            .admit_request(session, ctx, &arrival, &mut early_slot)
            .await
        {
            Ok(answered) => Ok(answered),
            Err(Stop::Refused(refusal)) => self.refuse(session, &arrival.request_id, refusal).await,
            Err(Stop::Failed(e)) => Err(e),
        }
    }

    async fn upstream_peer(
        &self,
        session: &mut Session,
        ctx: &mut Self::CTX,
    ) -> Result<Box<HttpPeer>> {
        // `ctx` is set by `request_filter` for every admitted request; a missing ctx here means an
        // unadmitted request reached `upstream_peer` (a Pingora ordering change or future refactor).
        // Surface it as an error rather than panicking the worker.
        let Some(rc) = ctx.as_mut() else {
            return Err(pingora_core::Error::new_str(
                "upstream_peer reached without request context",
            ));
        };
        self.state.fault_point("upstream_peer");

        // Pingora calls this once per attempt, always before a body byte moves, so it is where each
        // attempt starts on a fresh body record.
        rc.begin_attempt();
        let retry = rc.leg.retry.take();

        // Model-routed: this hook owns the candidate walk *and* the breaker ledger.
        //
        // It is the only place `rc.leg` changes, which is what makes double-recording structurally
        // impossible — and it is the only hook that runs on every attempt, including the ones
        // pingora starts without consulting `fail_to_connect` (its default `error_while_proxy`
        // marks a reused-connection failure retryable on its own). A design that recorded in
        // `fail_to_connect` would miss exactly those.
        //
        // Not for the same leg again (see `Retry::SameLeg`): that stays on this provider, keeps
        // its outstanding breaker permit, and must not enter the walk.
        if retry != Some(Retry::SameLeg) && rc.auto.is_some() {
            // The hook that moved the walk on declared the outgoing candidate failed (see
            // `Retry::NextCandidate`): resolve its permit before touching anything else, so
            // `logging` only ever sees the final candidate's. A first attempt owes nothing yet.
            // A permit still pending on any other entry is a retry that did not declare itself:
            // its leg would be released here rather than failed.
            debug_assert!(
                rc.leg.permit.is_none() || retry == Some(Retry::NextCandidate),
                "an undeclared retry entered the walk owing a permit ({retry:?})",
            );
            if retry == Some(Retry::NextCandidate) {
                rc.leg.fail();
            }

            loop {
                // Read the cursor out of the boxed state; `rc` stays mutably borrowable below.
                let (usable, at) = match rc.auto.as_ref() {
                    Some(a) => (a.usable, a.candidate),
                    None => {
                        return Err(pingora_core::Error::new_str("model routing state missing"));
                    }
                };
                let Some(i) = first_usable(usable, at) else {
                    // Out of candidates. `Error::new` defaults `retry` to false, so the proxy loop
                    // stops here rather than spinning; `logging` finds nothing pending to record.
                    // What the client is told depends on why: every candidate was tried and failed
                    // (502), or every one was skipped because its breaker is open (503, retry once
                    // the soonest one half-opens).
                    let attempted = rc.auto.as_ref().is_some_and(|a| a.attempted);
                    return Err(if attempted {
                        gateway_error(502, "no provider could be reached")
                    } else {
                        gateway_error(503, "provider temporarily unavailable")
                    });
                };
                if let Some(a) = rc.auto.as_mut() {
                    a.candidate = i;
                }

                let Some(candidate) = rc.auto.as_ref().and_then(|a| a.candidate_at(i)) else {
                    rc.advance_candidate(i);
                    continue;
                };
                if rc.auto.as_ref().is_some_and(|a| {
                    a.session_field.is_some() && !route::candidate_path_is_responses(candidate.path)
                }) {
                    // Session state must not walk onto Chat Completions / Messages. The walk is
                    // already filtered; this is the belt if a 5xx retry cursor drifted.
                    rc.advance_candidate(i);
                    continue;
                }
                let Some(p) = self.state.provider_by_id(candidate.provider).cloned() else {
                    // Unreachable: `usable` bits are only set for candidates that resolved.
                    rc.advance_candidate(i);
                    continue;
                };

                // Gate on *this* candidate's breaker. An open one is skipped without claiming a
                // permit and without an attempt — the entire point of holding a candidate list.
                //
                // Deliberately `allow()` rather than reading `state()`: the OPEN → HALF_OPEN
                // transition happens *inside* `allow()`, so a `state()`-based pre-check would report
                // `Open` past the reset timeout, skip a candidate that `allow()` would have admitted
                // as a probe, and leave the breaker with no way to ever close.
                let permit = match p.breaker.as_ref() {
                    None => None,
                    Some(b) => match b.allow() {
                        Ok(permit) => Some(permit),
                        Err(_) => {
                            self.state.metrics.rejection(Rejection::CircuitOpen).inc();
                            if let Some(a) = rc.auto.as_mut() {
                                let secs = u16::try_from(b.retry_after_secs()).unwrap_or(u16::MAX);
                                a.open_retry_after =
                                    Some(a.open_retry_after.map_or(secs, |s| s.min(secs)));
                            }
                            rc.advance_candidate(i);
                            continue;
                        }
                    },
                };
                // New vendor ⇒ that vendor's first key not cooling off (D71). A re-run of a large
                // body resumes the key walk an earlier attempt started on this candidate (see
                // `FullBody`); one with no walk starts past the cooling keys like any other request
                // (D83). A new leg starts at the candidate's first address, with its own
                // refused-stream resend, owing the permit (if this breaker has one to give).
                let pool_key = full_body_ctx(session)
                    .zip(rc.auto.as_ref().and_then(|a| a.walk.catalog_index(i)))
                    .and_then(|(fb, orig)| fb.keys.get(usize::from(orig)).copied())
                    .filter(|&k| k != NO_KEY_WALK)
                    .unwrap_or_else(|| p.first_key());
                rc.leg = Leg::new(p.clone(), permit, pool_key);
                apply_serving_candidate(rc);
                if let Some(a) = rc.auto.as_mut() {
                    a.attempted = true;
                }

                match self.state.resolve(&p.authority, 0).await {
                    Ok((addr, addrs)) => {
                        // Time this attempt from here, so a candidate that burned its connect
                        // timeout does not charge that to whichever provider ends up serving.
                        if let Some(a) = rc.auto.as_mut() {
                            a.attempt_start = Instant::now();
                            a.addrs = u8::try_from(addrs).unwrap_or(u8::MAX);
                        }
                        let sub = rc.auto.as_ref().and_then(|a| a.sub);
                        rc.set_forward_path(candidate.path, sub);
                        rc.upstream_phase = UpstreamPhase::Attempted;
                        let peer = self.build_peer(addr, &p, rc.auto.as_ref().map(|a| a.route));
                        return self.cap_read_at_deadline(rc, peer);
                    }
                    Err(e) => {
                        // DNS failure is handled *here*, inside the walk, rather than by returning
                        // an error: pingora does not call `fail_to_connect` when `upstream_peer`
                        // itself fails (lib.rs returns early), so a returned error would end the
                        // request instead of trying the next candidate — and a provider that has
                        // vanished from DNS is precisely a case failover exists for.
                        warn!(
                            request_id = %rc.request_id,
                            provider = p.name.as_str(),
                            authority = p.authority.as_str(),
                            candidate = i,
                            error = %e,
                            "upstream dns resolution failed; trying the next candidate",
                        );
                        rc.leg.fail();
                        rc.advance_candidate(i);
                        continue;
                    }
                }
            }
        }

        // A provider-routed attempt, or a walk's same leg again, which is timed from here.
        if let Some(a) = rc.auto.as_mut() {
            a.attempt_start = Instant::now();
        }
        // Resolve via the TTL cache (async, non-blocking) rather than `HttpPeer::new`'s eager
        // blocking `getaddrinfo`. SNI/Host = the configured host; TLS on for real providers (the
        // e2e harness flips `upstream_tls=false` for a plaintext mock).
        // Each connect retry (`fail_to_connect`) takes the next resolved address, so a name whose
        // first address is dead (`localhost` → `::1` first) still connects.
        let addr = match self
            .state
            .resolve(&rc.leg.provider.authority, rc.leg.addr)
            .await
        {
            Ok((a, _)) => a,
            Err(e) => {
                // DNS failures are rare and usually mean a misconfigured `provider_authorities`
                // override — so keep the diagnostic (provider name + authority + the resolver error,
                // already formatted into `e`) instead of discarding it behind an opaque static string.
                // `error_because` chains `e` as the cause so it shows in the Pingora error log.
                warn!(
                    request_id = %rc.request_id,
                    provider = rc.leg.provider.name.as_str(),
                    authority = rc.leg.provider.authority.as_str(),
                    error = %e,
                    "upstream dns resolution failed",
                );
                return Err(pingora_core::Error::because(
                    pingora_core::ErrorType::ConnectError,
                    "upstream dns resolution failed",
                    e,
                ));
            }
        };
        rc.upstream_phase = UpstreamPhase::Attempted;
        let peer = self.build_peer(addr, &rc.leg.provider, rc.auto.as_ref().map(|a| a.route));
        self.cap_read_at_deadline(rc, peer)
    }

    /// Fail over — or walk a pool key — before a byte of the error reaches the client.
    ///
    /// This hook runs strictly before anything is written downstream (`h1_response_filter` /
    /// `h2_response_filter` both call it ahead of `write_response_tasks`), so returning a retryable
    /// error here re-enters pingora's retry loop. That is the only place in the response path where
    /// abandoning an answer is still possible.
    ///
    /// Erroring *here* rather than in `response_filter` also keeps the per-attempt state clean for
    /// free: `response_filter` never runs for the abandoned attempt, so nothing increments
    /// `active_streams`, observes TTFT, or sets `upstream_status`, and `response_body_filter` never
    /// feeds the tail, head, or response model scanner. The next attempt starts from the same slate
    /// the first one did.
    ///
    /// Two distinct retries, never mixed:
    ///
    /// - **Managed 429 or 401 → next unused key, same provider.** A 429 is a healthy provider
    ///   throttling *that credential*, not a vendor outage; a 401 is that credential revoked
    ///   (D71), which also cools the key off for later requests
    ///   (`Provider::mark_key_bad`, counted on `ai_key_auth_failures_total`). Walks `/{provider}`
    ///   and `/auto`. BYO does not walk. The last 429 is relayed, `Retry-After` included; the last
    ///   key's 401 is relayed on a provider route and is a candidate failure on a catalog walk.
    ///   Counted on `ai_key_walks_total`. A 403 never walks or cools a key (D84): it is usually
    ///   about the request; one whose body names the key cools it from `logging`.
    /// - **Model-routed 5xx → next candidate.** A provider-routed request named its provider; there
    ///   is nowhere else to go. A 429 is *not* a vendor failover — re-asking a different vendor
    ///   would convert a self-healing throttle into spend somewhere else. Counted on
    ///   `ai_candidate_failovers_total`.
    ///
    /// Both require a **replayable** body. Pingora's replay buffer is a private 64 KiB constant;
    /// past it a retry would send headers describing a body it then never writes. That case is
    /// relayed, not attempted. [`retry::decide`] makes the call; this hook reports the status and
    /// acts on the answer.
    async fn upstream_response_filter(
        &self,
        session: &mut Session,
        upstream_response: &mut ResponseHeader,
        ctx: &mut Self::CTX,
    ) -> Result<()> {
        let Some(rc) = ctx.as_mut() else {
            return Ok(());
        };
        let status = upstream_response.status.as_u16();
        // A 2xx says this key's account pays: forget its unread OpenRouter 402s (D261).
        if rc.managed() && (200..300).contains(&status) {
            rc.leg.provider.key_served(rc.leg.pool_key);
        }

        // Managed 401: this pool key is revoked (D71). Never the caller's fault. Cool it off so
        // later requests start on a good key, then walk like a 429. Not a 403 (D84, see
        // `is_pool_key_failure`).
        let key_auth = rc.managed() && is_pool_key_failure(status);
        if key_auth {
            self.state.metrics.key_cooled(KeyCooled::Revoked);
            rc.leg.provider.mark_key_bad(rc.leg.pool_key);
        }

        // Below 400 nothing walks: the decision is always to relay, so the hot path reads no
        // further fact.
        if status < 400 {
            return Ok(());
        }
        let walk = rc.walk_facts();
        let decision = retry::decide(&retry::Failed {
            failure: retry::Failure::Status {
                status,
                managed: rc.managed(),
                next_key: rc.leg.next_key(),
            },
            walk,
            body: body_facts(session, walk),
        });
        let at = rc.auto.as_ref().map_or(0, |a| a.candidate);
        // A walk's 401–403 resolves its breaker permit as the success it is (the provider
        // answered); a 5xx's failure is recorded by `upstream_peer`'s prologue, which still sees
        // the permit outstanding, so recording it here as well would double-count.
        let key_failure = rc.managed() && is_candidate_refusal(status);
        // Failing over abandons this response at its head, so `Redact` never reads the body that
        // tells `logging` an account is out of credit (D180), and every later request would pay a
        // round trip to it first. A 402 says so by its status alone except on OpenRouter, whose
        // 402 may be one request too large for the balance (`remedy::unfunded_402`): cool that key
        // here, as `logging` would have, on either abandoning path below (D258). OpenRouter's
        // unread 402 is a strike against its key instead, and enough in a row cool it (D261).
        let unfunded_402 = || {
            (key_failure && status == 402)
                .then(|| rc.auto.as_ref().and_then(|a| a.candidate_at(at)))
                .flatten()
                .map(|c| remedy::unfunded_402(c.provider))
        };
        match decision {
            // The next pool key on this provider.
            Decision::Resend(Retry::SameLeg) => {
                self.state.metrics.key_walks_total.inc();
                warn!(
                    request_id = %rc.request_id,
                    provider = rc.leg.provider.name.as_str(),
                    key = rc.leg.pool_key,
                    status,
                    "upstream returned {status}; trying the next pool key",
                );
                if let Some(next) = rc.leg.next_key() {
                    rc.leg.pool_key = next;
                }
                rc.leg.retry = Some(Retry::SameLeg);
                let mut e = pingora_core::Error::new(pingora_core::ErrorType::HTTPStatus(status));
                e.set_retry(true);
                Err(e)
            }
            Decision::Resend(Retry::NextCandidate) => {
                self.state.metrics.candidate_failovers_total.inc();
                warn!(
                    request_id = %rc.request_id,
                    provider = rc.leg.provider.name.as_str(),
                    candidate = at,
                    status,
                    "upstream returned {status}; trying the next candidate",
                );
                let unfunded_402 = unfunded_402();
                if key_failure {
                    rc.leg.succeed();
                }
                self.abandon_402(rc, unfunded_402, status);
                rc.fail_over(at);
                let mut e = pingora_core::Error::new(pingora_core::ErrorType::HTTPStatus(status));
                e.set_retry(true);
                Err(e)
            }
            // The parent holds the body: it drops this response and re-runs.
            Decision::Rerun(retry @ RelayRetry::Key { .. }) => {
                self.state.metrics.key_walks_total.inc();
                warn!(
                    request_id = %rc.request_id,
                    provider = rc.leg.provider.name.as_str(),
                    key = rc.leg.pool_key,
                    status,
                    "upstream returned {status}; re-running the full body on the next pool key",
                );
                rc.attempt.relay_abandoned =
                    full_body_ctx(session).is_some_and(|fb| fb.record(retry));
                Ok(())
            }
            Decision::Rerun(retry) => {
                // `response_filter` still runs for it, so the breaker sees the failure.
                self.state.metrics.candidate_failovers_total.inc();
                warn!(
                    request_id = %rc.request_id,
                    provider = rc.leg.provider.name.as_str(),
                    candidate = at,
                    status,
                    "upstream returned {status}; re-running the full body on the next candidate",
                );
                let unfunded_402 = unfunded_402();
                rc.attempt.relay_abandoned =
                    full_body_ctx(session).is_some_and(|fb| fb.record(retry));
                // Not recorded (the final attempt): this answer is relayed and `logging` reads it.
                if rc.attempt.relay_abandoned {
                    self.abandon_402(rc, unfunded_402, status);
                }
                Ok(())
            }
            // Relayed, `Retry-After` included.
            Decision::Stand(stand) => {
                if stand.key_walk_unreplayable {
                    warn!(
                        request_id = %rc.request_id,
                        provider = rc.leg.provider.name.as_str(),
                        status,
                        body_done = session.as_mut().is_body_done(),
                        "upstream returned {status} but the request body is not provably replayable; not walking keys",
                    );
                }
                if stand.failover_unreplayable {
                    self.state.metrics.failover_unreplayable_total.inc();
                    warn!(
                        request_id = %rc.request_id,
                        provider = rc.leg.provider.name.as_str(),
                        status,
                        body_done = session.as_mut().is_body_done(),
                        "upstream failed but the request body is not provably replayable; relaying the error",
                    );
                }
                Ok(())
            }
        }
    }

    async fn upstream_request_filter(
        &self,
        _session: &mut Session,
        upstream_request: &mut pingora::http::RequestHeader,
        ctx: &mut Self::CTX,
    ) -> Result<()> {
        let Some(rc) = ctx.as_mut() else {
            return Ok(());
        };
        // Pingora runs this once the connection is up, just before the request head goes out.
        rc.upstream_phase = UpstreamPhase::Connected;
        let rc = &*rc;

        // Managed: swap the virtual key for the real pool key (precomputed at boot) in the scheme
        // the upstream wants — removing every inbound static-key header first (see
        // `STATIC_KEY_HEADERS`) so the virtual key never leaks upstream, and so a provider whose own
        // auth header (e.g. Azure's `api-key` — Task #8) happens to be the same header the client
        // presented its virtual key in doesn't end up with two stacked values. BYO (`!managed`):
        // leave the user's own auth header exactly as presented.
        if rc.managed() {
            // Strip **unconditionally**, before deciding whether there is a pool key to insert.
            //
            // These two used to sit inside the `if let Some(av)` below, which was safe only because
            // a managed request without a pool key is rejected earlier. That is a load-bearing
            // invariant expressed nowhere near here, and if it ever failed the consequence was not a
            // degraded request but a *credential disclosure*: nothing removed, nothing inserted, and
            // the caller's Ed25519 virtual key forwarded verbatim to a third-party provider. Now the
            // worst case is an unauthenticated request the provider rejects with a 401.
            //
            // The allowlist would drop these too; they are removed by name so this invariant does
            // not hang on the list's contents.
            let eager_tool_streaming = rc
                .auto
                .as_ref()
                .and_then(|a| a.candidate_at(a.candidate))
                .is_none_or(providers::catalog::serves_eager_input_streaming);
            retain_managed_client_headers(
                upstream_request,
                rc.leg.provider.id == Some(ProviderId::Anthropic),
                eager_tool_streaming,
            )?;
            upstream_request.remove_header("authorization");
            for header in STATIC_KEY_HEADERS {
                upstream_request.remove_header(header);
            }
            // Ask for an uncompressed response. Stock Python and Node SDKs send
            // `Accept-Encoding: gzip`, OpenAI and OpenRouter honor it, and every byte the gateway
            // reads afterwards (the usage tail, the cache fill, translation) assumes plain JSON/SSE:
            // a gzipped body billed zero tokens and a cache hit replayed gzip bytes without their
            // `Content-Encoding`. `identity` rather than removing the header: with no
            // `Accept-Encoding` at all, HTTP allows any coding. Managed only; a BYO response is never
            // parsed for billing.
            upstream_request.insert_header("accept-encoding", "identity")?;
            if let Some(auth) = rc.leg.provider.pool_auth.get(usize::from(rc.leg.pool_key)) {
                // Clone the boot-built `HeaderValue` (a refcount bump) rather than re-validating and
                // re-copying the key out of a `&str` on every managed request. The `&str` path is
                // kept as a fallback for a key that isn't a legal header value, which could never
                // have worked anyway — see `PoolAuth`.
                match &auth.header {
                    Some(hv) => {
                        upstream_request.insert_header(rc.leg.provider.auth.header(), hv.clone())?
                    }
                    None => upstream_request
                        .insert_header(rc.leg.provider.auth.header(), auth.value.expose())?,
                }
            }
        }

        // `x-beyond-*` is Beyond's namespace: the routing header, the control headers, and any name
        // the gateway does not (yet) define. None of it reaches a provider, on any route, managed or
        // BYO (D67) — a provider that rejects unknown headers would turn an opt-in into their 400,
        // and a header the gateway adds later must not have leaked from old clients first. A sweep
        // by prefix rather than a list of names, so a new header cannot be forgotten here. Runs on
        // every attempt (pingora rebuilds the head from the downstream request each time), and
        // before anything below adds a header of its own. Allocates nothing.
        strip_beyond_headers(upstream_request);

        // Point Host at the upstream. Same precomputed-value trick as the pool key above.
        match &rc.leg.provider.host_header {
            Some(hv) => upstream_request.insert_header("host", hv.clone())?,
            None => upstream_request.insert_header("host", rc.leg.provider.host.as_str())?,
        }

        // Dashboard-attribution headers (OpenRouter, managed traffic only — Task #22, see
        // `apply_provider_attribution`).
        apply_provider_attribution(upstream_request, rc.leg.provider.id, rc.managed())?;

        // A stock OpenAI SDK does not send `anthropic-version`. Anthropic (and Bedrock Messages)
        // require it; inject the current version when we translated Chat Completions or Responses
        // → Messages.
        if rc
            .auto
            .as_ref()
            .and_then(|a| a.translate.as_ref())
            .is_some_and(|t| {
                t.client != route::Endpoint::Messages && rc.dialect == Dialect::Anthropic
            })
            && upstream_request.headers.get("anthropic-version").is_none()
        {
            upstream_request.insert_header("anthropic-version", "2023-06-01")?;
        }
        if rc
            .auto
            .as_ref()
            .and_then(|a| a.translate.as_ref())
            .is_some_and(|_| rc.dialect == Dialect::OpenAi)
        {
            upstream_request.remove_header("anthropic-version");
        }
        // Preserved thinking on a translated walk: `translate::request` sets
        // `thinking.block_binding` for Anthropic's own conversation-binding models, a 400 without
        // this beta. Decided from the candidate alone, since headers leave before the body is
        // translated; merged with any beta value the client sent.
        if let Some(a) = rc.auto.as_ref()
            && catalog_translating(a)
            && let Some(c) = a.candidate_at(a.candidate)
            && c.provider == providers::ProviderId::Anthropic
            && route::Endpoint::of_upstream_path(c.path) == route::Endpoint::Messages
        {
            if let Some(beta) = translate::messages_beta(c.upstream_model) {
                merge_anthropic_beta(upstream_request, beta)?;
            }
            // A Chat Completions or Responses client's `service_tier: "priority"`, translated to
            // fast mode on a candidate that serves it (D266). The same decision makes
            // `translate::request_with_tools` add `speed: "fast"` to this attempt's body.
            if translated_fast(a, c) {
                merge_anthropic_beta(upstream_request, translate::FAST_MODE_BETA)?;
            }
        }

        // Forward the provider-native path (computed in `request_filter`): the client path with the
        // `/{provider}` segment stripped. Sent verbatim — no per-provider rewriting. The body's
        // framing (Content-Length / chunked) is preserved.
        //
        // `None` means the bare-path default, where the path is already what the upstream should
        // see, so there is nothing to build and nothing to parse. That used to be expressed by
        // reconstructing the path anyway and comparing it against the inbound `path_and_query`;
        // encoding it in the type instead skips the allocation rather than detecting it after
        // the fact.
        if let Some(forward_path) = rc.forward_path.as_str()
            && let Ok(uri) = forward_path.parse()
        {
            upstream_request.set_uri(uri);
        } else if rc.managed()
            && let Some(stripped) = upstream_request
                .uri
                .path_and_query()
                .and_then(|pq| strip_key_param(pq.as_str()))
            && let Ok(uri) = stripped.parse()
        {
            // The inbound path goes out as-is, and on a managed request its `key` params are the
            // virtual key: `request_filter` strips them from a built `forward_path`, this from the
            // path nobody rebuilt.
            upstream_request.set_uri(uri);
        }

        // Injection-eligible (OpenAI managed stream): the body is rewritten in `request_body_filter`,
        // changing its length, and we can't know the new length here (headers go out before the body
        // filter runs). So drop the client's `Content-Length`; how the now-unknown length is framed
        // depends on the **negotiated upstream protocol**, which is reliably readable here as
        // `upstream_request.version`: pingora-proxy sets it to HTTP/2 before this filter on the H2 path
        // (`proxy_h2.rs`) and to HTTP/1.1 on the H1 path (`proxy_h1.rs`).
        //
        //   - **H1**: a body with neither `content-length` nor `transfer-encoding` is framed as
        //     *zero-length* by pingora's H1 client (RFC 9112 §6.3) — the injected body would be
        //     silently dropped. So we must set `transfer-encoding: chunked`.
        //   - **H2**: bodies are delimited by `END_STREAM`, and `transfer-encoding` is a forbidden
        //     connection-specific header — the `h2` crate *rejects the whole request*
        //     (`UserError::MalformedHeaders`) if it's present. So we must NOT set it; removing
        //     `content-length` is sufficient and correct.
        //
        // Keyed on `rewrites_body`, not `stream_options`, so the model rewrite gets the same
        // treatment: `openai/gpt-4o-mini` is longer than `gpt-4o-mini`, and forwarding the client's
        // original `Content-Length` alongside a longer body truncates it at the upstream.
        if rc.plan.rewrites_body() {
            upstream_request.remove_header("content-length");
            if upstream_request.version != http::Version::HTTP_2 {
                upstream_request.insert_header("transfer-encoding", "chunked")?;
            }
        }
        Ok(())
    }

    async fn request_body_filter(
        &self,
        session: &mut Session,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
        ctx: &mut Self::CTX,
    ) -> Result<()> {
        let Ctx { rc, held } = ctx;
        let Some(rc) = rc.as_mut() else {
            return Ok(());
        };
        // An upload still trickling in at the request's deadline ends there (see
        // `response_body_filter`).
        if crate::deadline::expired(rc.deadline) {
            return Err(self.deadline_cut(&rc.request_id));
        }
        // Feed the body through the structural scanner as it passes (never withheld, never
        // buffered) to extract the exact root-level `model` — but only for **managed** traffic,
        // which is the only path that reads it. `rc.model` is used at exactly two places, both
        // inside the managed (`Billing::Managed`) block in `logging` (the billing-log fallback and
        // `requested_model`). Scanning it for BYO meant walking the whole request body — a
        // structural, depth- and escape-aware pass — to produce a value guaranteed to be discarded.
        if let Some(chunk) = body.as_ref() {
            // Enforce the body cap on the *streamed* size too: the up-front `Content-Length` check in
            // `request_filter` can't see a chunked-encoded body (no declared length). We don't buffer
            // — we just count — and abort the proxied request once the running total crosses the cap.
            // Aborting (vs. a clean 413) is acceptable here: headers are already away to the upstream,
            // and this is an abuse guard, not a normal client path.
            //
            // Tagged `Downstream` because it is the client's fault: an untagged error counted
            // against the provider's breaker, so one caller sending oversized bodies could open it
            // and 503 every tenant. `fail_to_proxy` answers it with the refusal, if it still can.
            if !rc.take_request_chunk(chunk) {
                return Err(self.refusal_error(Refusal::BodyTooLarge).into_down());
            }
            // Eligible requests are buffered so we can splice the root object before any byte reaches
            // the upstream (injection inserts near the front, so we can't have forwarded it already).
            // When we're buffering anyway, the incremental scan is skipped entirely and both answers
            // come from one walk of the finished buffer below — the body was previously traversed
            // twice, once here for `model` and once by the injection planner, over the same bytes
            // with the same depth/string/escape bookkeeping.
            // Only where no copy of the body will be left for `logging` to read (`tally_eager`):
            // everywhere else the estimate is made there, and only when a row needs one.
            if rc.tally_eager {
                rc.attempt.input_tally.feed(chunk);
                // The price knobs, for the same reason: no copy of this body is left for
                // `logging` to read (see `requested_knobs`).
                if let Some(taps) = rc.billing.tap() {
                    let (scan, kept) = taps.knobs.get_or_insert_with(|| {
                        (peek::ModelScanner::new(), peek::Kept::request_knobs())
                    });
                    scan.feed_keeping(chunk, kept);
                }
            }

            if rc.plan.rewrites_body() {
                // A large body buffered for a rewrite (the `/{provider}` usage splice; a catalog
                // walk's large body is a `FullBody` re-run, already reserved) holds budget as it
                // grows. Tagged downstream: the client's size, not the provider's health.
                let held_len = rc.attempt.req_buf.len();
                if past_replay_buffer(held_len) && !held.reserve_body(held_len) {
                    return Err(self.refusal_error(Refusal::BodyMemory).into_down());
                }
            } else if rc.managed() {
                rc.attempt.model_scanner.feed(chunk);
            }
        }

        if rc.plan.rewrites_body() {
            if end_of_stream {
                // One structural walk for every answer (see `peek::scan_buffered`).
                let mut buf = std::mem::take(&mut rc.attempt.req_buf);
                // --- signed ids (`signed_id`): the provider gets its own ids back ---
                // The client's body, per attempt. A catalog walk was checked before connecting; a
                // `/{provider}` body is refused here, before a byte of it goes upstream.
                if rc.signed.is_some()
                    && let Some(signer) = self.state.id_signer.as_ref()
                    && let Some(id) = rc.billing.identity()
                {
                    match signer.unsign_request(id.tenant_id, &buf, rc.auto.is_some()) {
                        Ok(Some(raw)) => buf = raw,
                        Ok(None) => {}
                        Err(_) => {
                            return Err(self.refusal_error(Refusal::ForeignIdLate).into_down());
                        }
                    }
                }
                // --- end signed ids ---
                let mut scanned = ScannedBody::new(buf);
                // The body rules (`body_rules`), on the client's body: ahead of translation, which
                // re-serializes it and would hide a second `model`, and of every rewrite, so an
                // `:online` model is named for what it is. Each refusal ends the request before a
                // byte of the body goes upstream. Every body buffered here is managed.
                let inspection = if rc.managed() {
                    let model = rc
                        .auto
                        .as_ref()
                        .map(|a| a.route.model)
                        .or(scanned.model())
                        .unwrap_or(rc.model.as_str());
                    let messages = session.req_header().uri.path().ends_with("/messages");
                    unpriced::inspect(scanned.bytes(), model, messages)
                } else {
                    unpriced::Inspection::default()
                };
                let refused = {
                    // A managed `/{provider}/…` billable call naming no catalog row (D267): its
                    // billing row would carry no `price_model`. Told by the client's id.
                    let miss = (rc.plan.catalog_check
                        && scanned.model().and_then(price_row).is_none())
                    .then(|| match rc.model.is_empty() {
                        true => scanned
                            .take_model()
                            .map_or(Cow::Borrowed(""), sanitize_model),
                        false => Cow::Borrowed(rc.model.as_str()),
                    });
                    let facts = body_rules::Facts {
                        // A catalog walk routed on one `model` and re-spells one, and a managed
                        // `/{provider}` body is billed by the first: the provider's parser picks
                        // its own (usually the last).
                        duplicate_model: (rc.auto.is_some() || rc.plan.catalog_check)
                            && scanned.duplicate_model(),
                        // A limit with no span (a string, a fraction, a negative): the clamp below
                        // cannot hold it to the row's maximum.
                        inexact_limit: scanned.inexact_limit()
                            && rc
                                .auto
                                .as_ref()
                                .is_some_and(|a| a.route.card.output_cap().is_some()),
                        refused_input: None,
                        unpriced: inspection.refused,
                        catalog_miss: miss.as_deref(),
                        // Managed `/{provider}/…/responses` asking for `background: true` (D202).
                        background: rc.plan.catalog_check
                            && rc.forward_path.as_str().is_some_and(|p| {
                                route::forward_is_responses(p.split_once('?').map_or(p, |(p, _)| p))
                            })
                            && requests_background(scanned.bytes()),
                    };
                    body_rules::refusal(body_rules::Point::Buffered, facts)
                };
                if let Some(refusal) = refused {
                    return Err(self.refusal_error(refusal).into_down());
                }
                if inspection.web_search_preview
                    && let Some(taps) = rc.billing.tap()
                {
                    taps.web_search_preview = true;
                }
                // Wire mismatch on this *candidate*: map the inbound JSON onto this path's
                // endpoint *before* the model splice. Keep the original client body in `req_buf`
                // (cleared and replayed per attempt) so a mixed-row failover re-translates rather
                // than forwarding the previous candidate's wire. OpenAI→Anthropic drops
                // `stream_options` here; Anthropic→OpenAI leaves `include_usage` to the inject
                // below, on the translated Chat Completions body. Same-endpoint Responses
                // (session arm) is `from == to` and is a byte relay, less the reasoning items the
                // gateway minted from Claude's thinking (see `translate::strip_gateway_reasoning`).
                if let Some(a) = rc.auto.as_mut() {
                    // An output limit past the row's maximum is a 400 by name; capped before the
                    // translation, so what it derives (a thinking budget) fits under the cap too.
                    // Only a vendor-published maximum: an unpublished one is a placeholder (D85).
                    // The edits below are opaque to the scan, so they open the body and close it
                    // re-scanned when one changed it. A clamp alone no longer re-scans, so a
                    // `model` the miss path took stays taken: unreachable, as every miss refuses at
                    // `Point::Buffered` (`Rule::CatalogMiss`) before this.
                    let mut buf = scanned
                        .clamp_output_limits(a.route.card.output_cap().unwrap_or(0))
                        .edit();
                    let mut changed = false;
                    let serving = catalog_serving_endpoint(a.as_ref());
                    let candidate = a.candidate_at(a.candidate);
                    let upstream_model = candidate.map_or("", |c| c.upstream_model);
                    let openai_host =
                        candidate.is_some_and(|c| c.provider == providers::ProviderId::OpenAi);
                    // Read once: this attempt's body edits ask the candidate several questions.
                    let caps = candidate.map(route::Candidate::caps);
                    let has = |flag| caps.is_some_and(|c: Caps| c.contains(flag));
                    let stream_only = has(Caps::STREAM_ONLY);
                    let fast = candidate.is_some_and(|c| translated_fast(a, c));
                    let reads_developer = caps.is_none() || has(Caps::DEVELOPER_ROLE);
                    let tool_thinking = candidate.map_or(
                        providers::catalog::ToolThinking::Free,
                        providers::catalog::tool_thinking,
                    );
                    let eager_tool_streaming = caps.is_none() || has(Caps::EAGER_INPUT_STREAMING);
                    let claude = has(Caps::CLAUDE);
                    if let Some(t) = a.translate.as_mut()
                        && let Some(to) = serving
                    {
                        t.tools = translate::ToolNames::default();
                        t.gateway_cache = false;
                        t.cache_1h = false;
                        if t.client != to {
                            // Translation builds `Value`s of the body: its heap is held in the
                            // body budget for the step, or the body is refused (D216).
                            let _heap = self.hold_translation_heap(&buf)?;
                            // The tool names this attempt's response maps calls back through, and
                            // whether its cache breakpoints are the gateway's (per attempt: a
                            // failover candidate on another wire re-decides both).
                            (*buf, t.tools, t.gateway_cache, t.cache_1h) =
                                translate::request_with_tools(
                                    t.client,
                                    to,
                                    &buf,
                                    upstream_model,
                                    fast,
                                );
                            changed = true;
                        } else if to == route::Endpoint::Responses {
                            let len = buf.len();
                            buf.map(translate::strip_gateway_reasoning);
                            // A row with no Responses arm walks its candidates only for a one-shot
                            // (session state there is a 400): onto a Responses candidate (xAI's)
                            // it must not be stored, and xAI stores by default. Nothing there holds
                            // a tool step's `item_reference` either: dropped, as translation drops
                            // it (one for an earlier turn was the 400, D175).
                            changed |= buf.len() != len;
                            if a.route.responses.is_empty() {
                                changed |= translate::store_false(&mut buf);
                                changed |= translate::strip_item_references(&mut buf);
                            }
                        } else if to == route::Endpoint::ChatCompletions && claude {
                            // A Claude model behind Chat Completions (OpenRouter): a tool turn
                            // with no thinking to replay goes without reasoning (D14's rule).
                            let len = buf.len();
                            buf.map(|b| translate::claude_chat_relay_reasoning(b, upstream_model));
                            changed |= buf.len() != len;
                        }
                        // Same-wire Chat to another host: an explicit null is "not set", as
                        // translation treats it; OpenRouter 400s `user: null` (D101).
                        if t.client == to && to == route::Endpoint::ChatCompletions && !openai_host
                        {
                            changed |= peek::remove_root_nulls(&mut buf);
                        }
                        if to == route::Endpoint::ChatCompletions {
                            // `developer` is `system` to a host that is not OpenAI's (D173); a
                            // translated Responses body already says so.
                            if t.client == to && !reads_developer {
                                changed |= translate::developer_as_system(&mut buf);
                            }
                            // A candidate whose thinking breaks tools gets them with thinking off
                            // (D171, D172): every candidate of its row refuses or garbles the
                            // combination, so steering around it would not help.
                            changed |= translate::fit_tools_to_thinking(&mut buf, tool_thinking);
                        }
                        // A candidate that answers only streams, for a client that did not ask
                        // for one: ask it for the stream and assemble the answer into the
                        // client's own body (D147). Per attempt, like the tools: a failover
                        // candidate gets the client's body as it came. Its usage is asked for by
                        // the splice below, as on every Chat Completions stream
                        // (`ScannedBody::into_upstream`); a Messages body takes no
                        // `stream_options`.
                        t.assemble = stream_only
                            && to != route::Endpoint::Responses
                            && translate::force_stream(&mut buf, false);
                        changed |= t.assemble;
                    }
                    // Fine-grained tool streaming only where the candidate takes it (D272): an
                    // older Bedrock stack 400s the field, and a 400 is not failed over. Per
                    // attempt, so direct Anthropic still gets the client's bytes (and its prompt
                    // cache key) unchanged; `upstream_request_filter` drops the matching beta.
                    if serving == Some(route::Endpoint::Messages) && !eager_tool_streaming {
                        changed |= translate::strip_eager_input_streaming(&mut buf);
                    }
                    scanned = buf.finish(changed);
                    // Native OpenAI Chat takes `max_completion_tokens` on every model and 400s
                    // `max_tokens` on its reasoning ones; a translated body already says the former.
                    if candidate.is_some_and(native_openai_chat) {
                        scanned = scanned.rename_max_tokens();
                    }
                }
                if rc.model.is_empty()
                    && let Some(m) = scanned.take_model()
                {
                    // The *client's* id, captured before any rewrite below — this is what
                    // `requested_model` means, and it stays the canonical catalog name on a
                    // model-routed request even though the upstream is about to be told
                    // something else.
                    rc.model = sanitize_model(m).into_owned();
                }
                // Model-routed: re-spell `model` as the candidate serving *this attempt* spells it.
                // Providers essentially never agree on an id — Anthropic's `claude-opus-4-8` is
                // OpenRouter's `anthropic/claude-opus-4-8` — so without this a failover would ask
                // the fallback for a model it has never heard of.
                //
                // `plan.stream_options` is the OpenAI-upstream gate: a translated Anthropic body also
                // carries `"stream":true`, and splicing `stream_options` into it would be a field
                // that API does not recognize. The edits' order is `into_upstream`'s.
                let candidate = rc.auto.as_ref().and_then(|a| a.candidate_at(a.candidate));
                let buf = scanned.into_upstream(
                    rc.plan.stream_options,
                    candidate.map(|c| c.upstream_model.as_bytes()),
                );
                // Emit the whole (possibly rewritten) body in one shot; `transfer-encoding: chunked`
                // (set in `upstream_request_filter`) makes the changed length fine.
                let buf = match candidate {
                    Some(c) if openrouter_chat(c) => disable_openrouter_compression(buf),
                    _ => buf,
                };
                *body = Some(Bytes::from(buf));
            } else {
                // Withhold — the bytes are buffered above; nothing goes upstream until end-of-stream.
                // Use an *empty* chunk, not `None`: pingora derives end-of-body as
                // `end_of_body || data.is_none()` (proxy_h1.rs / proxy_h2.rs), so withholding with
                // `None` would signal end-of-body on the *first* withheld chunk and forward a truncated
                // (empty) body — silently dropping every request body that spans more than one chunk.
                // An empty `Some` is recognized as "nothing to write yet" without ending the body.
                *body = Some(Bytes::new());
            }
        }

        // The streamed (non-buffered) path; the buffered one resolved `model` above from its single
        // fused walk, and its scanner was never fed.
        if end_of_stream
            && rc.managed()
            && !rc.plan.stream_options
            && rc.model.is_empty()
            && let Some(m) = rc.attempt.model_scanner.take_model()
        {
            rc.model = sanitize_model(m).into_owned();
        }
        Ok(())
    }

    async fn response_filter(
        &self,
        session: &mut Session,
        upstream_response: &mut ResponseHeader,
        ctx: &mut Self::CTX,
    ) -> Result<()> {
        let Ctx { rc, held } = ctx;
        if let Some(rc) = rc.as_mut() {
            // Headers arrived ≈ time-to-first-byte. Per-provider handle resolved once at boot (see
            // `ProviderMetrics`) — first-token latency is per-provider, so an unlabeled histogram
            // can't tell you which one regressed.
            //
            // Timed from `attempt_start`, not `start`. They are the same instant for a request that
            // connects first try; they differ once a candidate has been abandoned, and charging the
            // dead candidate's `connect_timeout_secs` to the provider that actually answered would
            // render an outage at A as a latency regression at B — inverting the one thing the
            // per-provider label is for.
            // Per-provider response counter, bucketed by status class — the signal that a provider
            // is degrading (429/5xx) before it shows up only as latency or a missing usage event.
            let status = upstream_response.status.as_u16();
            rc.leg
                .provider
                .metrics
                .ttft_seconds
                .observe(rc.attempt_start().elapsed().as_secs_f64());
            rc.leg.provider.metrics.record_response(status);
            rc.upstream_status = held_status(status);
            // Resolve the breaker permit here, at the head, not at the end of the body: the head
            // is the provider's answer (a 5xx is broken, anything else is reachable), and a
            // half-open probe resolved only at end of stream let one long or stalled stream 503
            // the provider for everyone until it ended. `logging` resolves only the attempts that
            // never got a head.
            let answered = if status >= 500 {
                rc.leg.fail();
                None
            } else {
                rc.leg.succeed()
            };
            // A 2xx is not yet known to be healthy: a provider can answer 200 with an error body
            // (OpenRouter's error-in-200, an SSE stream whose first event is an error), so its
            // breaker verdict waits for the body's first bytes (`response_body_filter`).
            if (200..300).contains(&status)
                && let Some(a) = rc.auto.as_mut()
            {
                a.health = Some(PendingHealth {
                    prefix: Vec::new(),
                    permit: answered,
                });
            }

            // Derive streaming from the response, not the request: SSE ⇒ use the streaming usage
            // parser; otherwise the body is a single JSON object.
            rc.streaming = upstream_response
                .headers
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .is_some_and(|ct| ct.contains("event-stream"));
            // Track concurrent SSE streams. Incremented here (response head is in), decremented in
            // `logging` once the stream completes — so the gauge reflects in-flight streams, not a
            // counter that only ever climbs. Non-streaming responses don't touch it.
            if rc.streaming {
                held.open_stream();
            }
            if rc.managed() {
                tap_response_head(rc, upstream_response);
            }
            // Drain-on-cancel, decided per head (a walk's next attempt may be another provider).
            // A drain already under way is left as it is.
            if matches!(held.drain, Drain::Off | Drain::Armed) {
                held.drain = if drains_on_cancel(rc, status) && full_body_ctx(session).is_none() {
                    if drain::arm(&mut session.cache) {
                        Drain::Armed
                    } else {
                        Drain::Off
                    }
                } else {
                    Drain::Off
                };
            }
            self.state.fault_point("response_filter");

            // `x-beyond-*` is the gateway's namespace. A provider (or anything between us and it)
            // that sent its own `x-beyond-provider` or `x-beyond-cache-status` would otherwise reach
            // the client beside, or instead of, ours. Allocates only when one is present.
            let spoofed: Vec<http::HeaderName> = upstream_response
                .headers
                .keys()
                .filter(|k| k.as_str().starts_with("x-beyond-"))
                .cloned()
                .collect();
            for name in &spoofed {
                upstream_response.remove_header(name);
            }

            // The pool key never reaches the client (D66). A provider, or anything between us and
            // it, that echoes the credential it was sent would otherwise hand Beyond's key to a
            // tenant: any header carrying it is dropped, and an error body (>= 400, where an echo
            // lives: "Incorrect API key provided: …") is scrubbed as it streams (`Redact`). A 2xx
            // body is an answer and is not scanned; a streamed answer would pay for it per chunk.
            rc.redact = None;
            if rc.managed()
                && let Some(finder) = rc
                    .leg
                    .provider
                    .pool_auth
                    .get(usize::from(rc.leg.pool_key))
                    .map(route::PoolAuth::finder)
                    .filter(|f| !f.needle().is_empty())
            {
                // Collects nothing (and so allocates nothing) unless a header echoes the key.
                let echoed: Vec<http::HeaderName> = upstream_response
                    .headers
                    .iter()
                    .filter(|(_, v)| finder.find(v.as_bytes()).is_some())
                    .map(|(k, _)| k.clone())
                    .collect();
                for name in &echoed {
                    upstream_response.remove_header(name);
                }
            }
            // A managed JSON error is held whole and checked for a provider-account remedy (D174):
            // the rewrite changes its length, so it loses its `Content-Length` here, on the error
            // path only. A BYO error is the caller's own account talking and is relayed as sent.
            if rc.managed() && status >= 400 {
                let whole = !rc.streaming
                    && upstream_response
                        .headers
                        .get("content-type")
                        .and_then(|v| v.to_str().ok())
                        .is_some_and(|ct| ct.contains("json"));
                if whole {
                    upstream_response.remove_header("content-length");
                    if upstream_response.version != http::Version::HTTP_2 {
                        upstream_response.insert_header("transfer-encoding", "chunked")?;
                    }
                }
                let keyed = rc
                    .leg
                    .provider
                    .pool_auth
                    .get(usize::from(rc.leg.pool_key))
                    .is_some_and(|a| !a.finder().needle().is_empty());
                if whole || keyed {
                    rc.redact = Some(Box::new(Redact {
                        whole,
                        status,
                        ..Redact::default()
                    }));
                }
            }

            // Echo the request id so a client (or an oncall reading a captured response) can quote it
            // and land on this request's log line. `insert_header` only fails on an invalid value;
            // our id is `[0-9a-f-]`, always valid — but surface a failure rather than silently drop.
            upstream_response.insert_header(REQUEST_ID_HEADER, rc.request_id.as_str())?;
            if let Some(hv) = rc.leg.provider.name_header.as_ref() {
                upstream_response.insert_header(PROVIDER_HEADER, hv.clone())?;
            }
            if let Some(c) = rc.auto.as_ref().and_then(|a| a.candidate_at(a.candidate)) {
                upstream_response.insert_header(UPSTREAM_MODEL_HEADER, c.upstream_model)?;
            }

            // An assembled answer (D147): the upstream streams, the client gets one JSON body.
            if rc.streaming && rc.auto.as_ref().is_some_and(|a| catalog_assembling(a)) {
                upstream_response.insert_header("content-type", "application/json")?;
            }
            if let Some(cache::Pending::Fill { content_type, .. }) =
                rc.auto.as_mut().and_then(|a| a.cache.as_mut())
            {
                *content_type = upstream_response
                    .headers
                    .get("content-type")
                    .and_then(|v| v.to_str().ok())
                    .map(|s| s.to_owned().into_boxed_str());
            }

            // Translate changes the body length (and SSE event count). Drop the upstream
            // Content-Length so the client is not truncated; H1 needs chunked framing.
            // Same-endpoint candidates stay a byte relay, except a Chat Completions stream from a
            // vendor other than OpenAI: it relays through `SseBridge`, which drops the identity
            // fields OpenRouter repeats on every chunk (see `translate::ChatIdentity`).
            if rc.auto.as_ref().is_some_and(|a| {
                catalog_translating(a)
                    || (rc.streaming && (catalog_chat_relay(a) || catalog_assembling(a)))
                    || catalog_error_relay(a, rc.upstream_status(), rc.streaming)
            }) {
                let streaming = rc.streaming;
                let upstream = rc
                    .auto
                    .as_ref()
                    .and_then(|a| catalog_serving_endpoint(a))
                    .unwrap_or_else(|| route::Endpoint::of_wire(rc.dialect));
                if let Some(t) = rc.auto.as_mut().and_then(|a| a.translate.as_mut())
                    && streaming
                {
                    let bridge = if t.assemble {
                        translate::SseBridge::assembling(upstream)
                    } else {
                        translate::SseBridge::new(t.client, upstream)
                    };
                    t.sse = Some(
                        bridge
                            .with_tools(t.tools.clone())
                            .with_gateway_cache(t.gateway_cache),
                    );
                }
                upstream_response.remove_header("content-length");
                if upstream_response.version != http::Version::HTTP_2 {
                    upstream_response.insert_header("transfer-encoding", "chunked")?;
                }
            }
            // --- signed ids (`signed_id`): a 2xx's ids are rewritten, so its length changes ---
            if let Some(s) = rc.signed.as_mut()
                && s.begin(status, rc.streaming)
            {
                upstream_response.remove_header("content-length");
                if upstream_response.version != http::Version::HTTP_2 {
                    upstream_response.insert_header("transfer-encoding", "chunked")?;
                }
            }
            // --- end signed ids ---
        }
        Ok(())
    }

    fn response_body_filter(
        &self,
        _session: &mut Session,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
        ctx: &mut Self::CTX,
    ) -> Result<Option<Duration>>
    where
        Self::CTX: Send + Sync,
    {
        // Usage taps read the *upstream* dialect. Cache fill (#68) and capture read the bytes
        // the client sees — post-translate on a wire-mismatched catalog walk, the relayed
        // chunk otherwise. SSE is converted event-by-event; the full stream is never buffered.
        let Ctx { rc, held } = ctx;
        let Some(rc) = rc.as_mut() else {
            return Ok(None);
        };
        self.state.fault_point("response_body_filter");
        // A response still moving at the request's deadline is cut (a silent one is ended by the
        // capped read timeout instead). One atomic load: see `crate::deadline`.
        if crate::deadline::expired(rc.deadline) {
            return Err(self.deadline_cut(&rc.request_id));
        }
        // First, so nothing downstream — translation, capture, the cache, the usage tail — ever
        // holds the pool key (D66).
        if let Some(r) = rc.redact.as_mut() {
            let keys = rc
                .leg
                .provider
                .pool_auth
                .get(usize::from(rc.leg.pool_key))
                .map_or(&[][..], route::PoolAuth::finders);
            *body = r.feed(keys, body.take(), end_of_stream);
        }
        let chunk = body.as_deref().unwrap_or(&[]);
        // A catalog walk's 2xx waits here for its health verdict (see `response_filter`): the
        // first event or JSON key says whether the 200 carries an answer or an error. Bounded, and
        // done after a few hundred bytes.
        if rc.auto.as_ref().is_some_and(|a| a.health.is_some()) {
            self.settle_health(rc, chunk, end_of_stream);
        }
        if !chunk.is_empty() {
            // Tap the provider-reported (resolved/billed) model from the response *head* — the
            // scanner stops at the first root `model`, so this is O(1) and cheap (it finds the model
            // in the first chunk and ignores the rest). Kept separate from the tail because the model
            // is at the start of the response while the usage event is at the end.
            //
            // Managed only, for the same reason as the request-side scanner: the extracted value is
            // read at exactly one place, inside the managed (`Billing::Managed`) block in `logging`. A BYO
            // request has no Beyond identity and emits no billing row, so scanning its response was
            // pure waste — and *unbounded* waste on any response with no root-level `model`, since
            // the scanner never reaches its `done` short-circuit and walks every byte.
            if rc.managed() {
                match rc.billing.taps_mut() {
                    Some(t) => {
                        let kept = t.resp_kept.get_or_insert_with(peek::Kept::response);
                        rc.resp_model_scanner.feed_keeping(chunk, kept);
                        if let Some(tools) = t.tools.as_mut() {
                            tools.feed(chunk);
                        }
                    }
                    None => rc.resp_model_scanner.feed(chunk),
                }
            }

            // Anthropic SSE only: the head that keeps `message_start` (see `keep_usage_head`).
            rc.keep_usage_head(chunk);

            rc.resp_tail.push(chunk);
            // Managed only. Counts *upstream* bytes (pre-translate), like the tail.
            if rc.managed() {
                rc.resp_bytes = rc
                    .resp_bytes
                    .saturating_add(u32::try_from(chunk.len()).unwrap_or(u32::MAX));
            }
        }

        // Drain-on-cancel: once the client has left an armed stream, a chunk feeds the usage taps
        // above and nothing else, and the client is sent nothing more (see `crate::drain`).
        if held.drain != Drain::Off && drain_chunk(rc, held, end_of_stream) {
            *body = None;
            return Ok(None);
        }

        // --- signed ids (`signed_id`): after the usage taps, before anything the client sees ---
        if let (Some(s), Some(signer)) = (rc.signed.as_mut(), self.state.id_signer.as_ref()) {
            match s.feed(signer, body.as_deref().unwrap_or(&[]), end_of_stream) {
                Ok(Some(out)) => *body = Some(Bytes::from(out)),
                Ok(None) => {}
                Err(signed_id::Overflow) => {
                    return Err(self.translate_overflow(&rc.request_id, "signed_id"));
                }
            }
        }
        let chunk = body.as_deref().unwrap_or(&[]);
        // --- end signed ids ---

        // A translation, a Chat Completions relay that `response_filter` gave a bridge, or a
        // same-endpoint error put in the client's envelope.
        let translating = rc.auto.as_ref().is_some_and(|a| {
            catalog_translating(a)
                || a.translate.as_ref().is_some_and(|t| t.sse.is_some())
                || catalog_error_relay(a, rc.upstream_status(), rc.streaming)
        });
        if translating {
            let streaming = rc.streaming;
            let dialect = rc.dialect;
            let status = rc.upstream_status().unwrap_or(200);
            let upstream = rc
                .auto
                .as_ref()
                .and_then(|a| catalog_serving_endpoint(a))
                .unwrap_or_else(|| route::Endpoint::of_wire(dialect));
            let out = if let Some(t) = rc.auto.as_mut().and_then(|a| a.translate.as_mut()) {
                if streaming {
                    let (client, tools, gateway_cache) = (t.client, &t.tools, t.gateway_cache);
                    let sse = t.sse.get_or_insert_with(|| {
                        translate::SseBridge::new(client, upstream)
                            .with_tools(tools.clone())
                            .with_gateway_cache(gateway_cache)
                    });
                    let mut out = sse.feed(chunk, end_of_stream);
                    if let Some(buffer) = sse.overflow() {
                        return Err(self.translate_overflow(&rc.request_id, buffer));
                    }
                    // An assembling bridge writes nothing until the upstream has ended.
                    if t.assemble && end_of_stream {
                        out = sse.assembled(client);
                    }
                    out
                } else {
                    if !t.push_json(chunk) {
                        return Err(self.translate_overflow(&rc.request_id, "json_body"));
                    }
                    if end_of_stream {
                        // The response's `Value`s are held in the body budget like a request's
                        // (D216). Headers are already downstream, so a refusal aborts the response,
                        // as `translate_overflow` does. A same-wire 2xx is relayed as bytes.
                        let _heap = if upstream == t.client && (200..300).contains(&status) {
                            None
                        } else {
                            self.hold_translation_heap(&t.json_buf).inspect_err(|_| {
                                warn!(
                                    request_id = %rc.request_id,
                                    bytes = t.json_buf.len(),
                                    "translated response exceeds the body budget; aborting",
                                );
                            })?
                        };
                        translate::response_json_tools(
                            upstream,
                            t.client,
                            status,
                            &t.json_buf,
                            &t.tools,
                            t.gateway_cache,
                        )
                    } else {
                        Vec::new()
                    }
                }
            } else {
                Vec::new()
            };
            if !out.is_empty() {
                if let Some(c) = rc.control.as_mut().and_then(|c| c.capture.as_mut()) {
                    c.push_resp(&out);
                }
                if let Some(cache::Pending::Fill { tap, .. }) =
                    rc.auto.as_mut().and_then(|a| a.cache.as_mut())
                {
                    tap.push(&out);
                }
            }
            if rc.tracks_terminal() {
                rc.terminal.feed(&out);
            }
            *body = Some(Bytes::from(out));
        } else if !chunk.is_empty() {
            if rc.tracks_terminal() {
                rc.terminal.feed(chunk);
            }
            // Capture tap — same passive-tap contract as the usage tail above (copy, never withhold),
            // differing only in which end it keeps. `resp_tail` keeps the *last* 64 KB because usage
            // rides the final event; capture keeps the *first* `max_bytes` because that's where the
            // answer starts. On a captured Anthropic SSE response all three buffers coexist, which is
            // fine — each is independently bounded.
            if let Some(c) = rc.control.as_mut().and_then(|c| c.capture.as_mut()) {
                c.push_resp(chunk);
            }

            if let Some(cache::Pending::Fill { tap, .. }) =
                rc.auto.as_mut().and_then(|a| a.cache.as_mut())
            {
                tap.push(chunk);
            }
        }
        Ok(None)
    }

    /// Keep pingora 0.8's retry policy for an error after the connection is up — except that a
    /// body the provider already has is never sent again.
    ///
    /// Pingora 0.9's default refuses to retry any non-idempotent method, and every LLM call is a
    /// `POST` — so the default silently turned off both retries this gateway decides for itself:
    /// the managed 429 key walk and the model-routed 5xx vendor walk. `upstream_response_filter`
    /// only marks those retryable after `body_replayable` has proven the body can be resent, which
    /// is the safety condition the new default approximates with the method.
    ///
    /// A failure with no response (a reset, an early close, a read timeout) is resent only when it
    /// happened before the upstream had the whole body ([`body_delivered`]): a reset mid-upload, a
    /// write error. After that the provider may well be generating, and billing, the answer, so
    /// sending the body again — to the same candidate or the next — duplicates the spend; the
    /// request ends instead, with a JSON 504/502 (`fail_to_proxy`). That holds for pingora's own
    /// reused-connection retry and for the `FullBody` reset retry alike. A 5xx or 429 is an
    /// answer, not a failure, and keeps walking as before.
    ///
    /// The one exception is a stream the provider refused ([`upstream_refused_stream`], D72): an
    /// HTTP/2 GOAWAY that left it above `last_stream_id`, or `RST_STREAM(REFUSED_STREAM)`. That is
    /// the provider's guarantee it processed none of the request, so it is resent — on a fresh
    /// connection to the same candidate — however much of the body went out. Routine H2
    /// connection recycling at a provider would otherwise surface as client 502s.
    ///
    /// [`retry::decide`] makes every one of these calls; this hook reports what the error says
    /// about the body and acts on the answer.
    fn error_while_proxy(
        &self,
        peer: &HttpPeer,
        session: &mut Session,
        e: Box<pingora_core::Error>,
        ctx: &mut Self::CTX,
        client_reused: bool,
    ) -> Box<pingora_core::Error> {
        use pingora_core::ErrorType as T;
        let mut e = e.more_context(format!("Peer: {peer}"));
        // A read timeout capped at the request's deadline is the deadline passing, not the
        // provider going quiet: the gateway's own decision, like those below.
        if e.etype() == &T::ReadTimedout
            && let Some(rc) = ctx.rc.as_ref()
            && rc.attempt.read_capped
        {
            return self.deadline_cut(&rc.request_id);
        }
        // Our own decisions (a 5xx / 401 vendor walk, a 429 key walk, a body cap) are made.
        if matches!(e.etype(), T::HTTPStatus(_) | T::CustomCode(..)) {
            return e;
        }
        // A client that stalled mid-upload timed its own request out (D260): retag the timeout as
        // the client's, so the walk below ends here rather than failing over with a body nobody
        // will finish, `logging` charges no breaker and bills no estimate, and `fail_to_proxy`
        // answers 408 rather than blaming the provider with a 504.
        if ctx
            .rc
            .as_ref()
            .is_some_and(|rc| client_stalled_upload(session, rc, &e))
        {
            e.as_down();
        }
        let Some(rc) = ctx.rc.as_mut() else {
            e.set_retry(false);
            return e;
        };
        // What this failure says about the body (see `retry::decide`): whether the provider may
        // have all of it, or proved it took none (an HTTP/2 refusal, a pooled HTTP/1.1 connection
        // it closed with the request unread).
        //
        // A fact `decide` cannot read past an earlier one is not computed: the delivery check
        // only matters while the client is there, nothing reached it and nothing proved the body
        // unprocessed; a GOAWAY only for a refusal; an unsent HTTP/2 body only for one that is not.
        let downstream = *e.esource() == pingora_core::ErrorSource::Downstream;
        let responded = session.as_downstream().response_written().is_some();
        let refused = upstream_refused_stream(&e);
        let reset_unread = reset_before_reading(&e, client_reused);
        let open = !downstream && !responded;
        let lost = retry::Lost {
            downstream,
            responded,
            delivered: open && !refused && !reset_unread && body_delivered(session, rc, Some(&*e)),
            refused,
            goaway: refused && upstream_goaway(&e),
            reset_unread,
            reused: client_reused,
            refused_resent: rc.leg.refused_resent,
            h2_body_unsent: open && !refused && h2_body_unsent(&e),
            pingora_retries: {
                let mut verdict = e.retry;
                verdict.decide_reuse(client_reused);
                verdict.retry()
            },
        };
        let walk = rc.walk_facts();
        let decision = retry::decide(&retry::Failed {
            failure: retry::Failure::Lost(lost),
            walk,
            body: body_facts(session, walk),
        });
        if lost.may_resend() && refused {
            if lost.resends_refusal() {
                warn!(
                    request_id = %rc.request_id,
                    provider = rc.leg.provider.name.as_str(),
                    error = %e,
                    "upstream refused the stream unprocessed (GOAWAY / REFUSED_STREAM); resending once",
                );
            } else {
                warn!(
                    request_id = %rc.request_id,
                    provider = rc.leg.provider.name.as_str(),
                    error = %e,
                    "upstream refused the stream again; treating it as a provider failure",
                );
            }
        }
        match decision {
            // The same candidate on a fresh connection, keeping its breaker permit: it was the
            // connection, not the provider.
            Decision::Resend(Retry::SameLeg) => {
                rc.leg.refused_resent |= lost.spends_resend();
                rc.leg.retry = Some(Retry::SameLeg);
                e.set_retry(true);
            }
            Decision::Resend(Retry::NextCandidate) => {
                self.state.metrics.candidate_failovers_total.inc();
                rc.fail_over(rc.auto.as_ref().map_or(0, |a| a.candidate));
                e.set_retry(true);
            }
            // A large body: pingora cannot resend it, but the parent holds it.
            Decision::Rerun(retry) => {
                e.set_retry(false);
                // A same-candidate retry is the connection's fault (or a stream refused before
                // any processing), not the provider's: give the permit back without an outcome,
                // as a small body keeps it for its own same-candidate retry.
                if matches!(retry, RelayRetry::Reset(_)) {
                    rc.leg.release();
                }
                rc.attempt.relay_abandoned =
                    full_body_ctx(session).is_some_and(|fb| fb.record(retry));
            }
            Decision::Stand(_) => e.set_retry(false),
        }
        e
    }

    /// Answer every request the gateway itself ends, after admission or during the proxy loop,
    /// with the same JSON error envelope a refusal uses: `content-type: application/json`,
    /// `{"error":{"message","type"}}`, and `x-beyond-request-id`. Pingora's default wrote a bare
    /// status with an empty body (a 500 for "every breaker open", which is a 503), which an SDK
    /// cannot parse and an oncall cannot correlate.
    ///
    /// The status follows the cause (see [`failure_response`]): a connect failure on every
    /// candidate is a 502, every candidate's breaker open a 503 with `Retry-After`, an upstream
    /// timeout a 504, a chunked body over the cap a 413. An upstream that failed after it had the
    /// whole request (the walk ends there rather than resending, see `error_while_proxy`) says so:
    /// `upstream timed out after receiving the request` (504) or `upstream failed after receiving
    /// the request` (502). A stream the provider refused twice says it was not processed (502). A
    /// client that is already gone gets nothing, and a response that already started cannot be
    /// replaced.
    async fn fail_to_proxy(
        &self,
        session: &mut Session,
        e: &pingora_core::Error,
        ctx: &mut Self::CTX,
    ) -> FailToProxy
    where
        Self::CTX: Send + Sync,
    {
        let Some((status, typ, mut msg)) = failure_response(e) else {
            return FailToProxy {
                error_code: 0,
                can_reuse_downstream: false,
            };
        };
        if session.response_written().is_some() {
            return FailToProxy {
                error_code: status,
                can_reuse_downstream: false,
            };
        }
        // The provider had the whole request: name that, so a client does not read a resend-safe
        // failure into it (the gateway did not resend; a client retry may run it twice).
        let after_delivery = *e.esource() == pingora_core::ErrorSource::Upstream
            && !matches!(
                e.etype(),
                pingora_core::ErrorType::HTTPStatus(_) | pingora_core::ErrorType::CustomCode(..)
            )
            && ctx
                .rc
                .as_ref()
                .is_some_and(|rc| body_delivered(session, rc, Some(e)));
        // A stream the provider refused (D72) was not processed, however much of the body went
        // out: say exactly that, so a client knows its own retry is safe (D91).
        let refused = upstream_refused_stream(e);
        let status = match (after_delivery && !refused, status) {
            _ if refused => {
                msg = "the provider refused the stream; it was not processed";
                502
            }
            (true, 504) => {
                msg = "upstream timed out after receiving the request";
                504
            }
            (true, _) => {
                msg = "upstream failed after receiving the request";
                502
            }
            (false, status) => status,
        };
        // A refusal raised past `request_filter` (a `CustomCode`, so neither adjustment above
        // applies to it) is answered with its own body; any other error with the static one for
        // its class.
        let body = match Refusal::of(e) {
            Some(refusal) => refusal.body(),
            None => error_body(typ, msg),
        };
        // Pingora closes the client connection after a proxy error; say so (`connection: close`),
        // as its own error responses do, so a pooled client does not send its next request into a
        // socket about to close.
        session.as_downstream_mut().set_keepalive(None);
        // One of the two sources of a gateway `Retry-After`: this one answers every 503 raised past
        // `request_filter` (a refusal there included, such as `BodyMemory`), with the soonest open
        // breaker of the walk, else 1. A refusal written in `request_filter` carries
        // `Refusal::retry_after` instead.
        let retry_after = (status == 503).then(|| {
            ctx.rc
                .as_ref()
                .and_then(|rc| rc.auto.as_ref())
                .and_then(|a| a.open_retry_after)
                .map_or(1, u64::from)
        });
        let request_id = ctx
            .held
            .request_id
            .or_else(|| ctx.rc.as_ref().map(|rc| rc.request_id))
            .unwrap_or_default();
        if let Err(write_err) = Box::pin(write_json_body(
            session,
            &request_id,
            status,
            body,
            retry_after,
        ))
        .await
        {
            warn!(
                request_id = %request_id,
                status,
                error = %write_err,
                "failed to send the error response downstream",
            );
        }
        FailToProxy {
            error_code: status,
            can_reuse_downstream: false,
        }
    }

    fn fail_to_connect(
        &self,
        _session: &mut Session,
        _peer: &HttpPeer,
        ctx: &mut Self::CTX,
        mut e: Box<pingora_core::Error>,
    ) -> Box<pingora_core::Error> {
        let Some(rc) = ctx.as_mut() else { return e };
        let walk = rc.walk_facts();
        // A connect failure moves no body byte, so the body it would resend is never a question.
        let decision = retry::decide(&retry::Failed {
            failure: retry::Failure::Connect {
                retries_left: rc.leg.addr < MAX_CONNECT_RETRIES,
            },
            walk,
            body: retry::Body::default(),
        });
        let at = rc.auto.as_ref().map_or(0, |a| a.candidate);
        match (walk, decision) {
            // The candidate's next address, or a provider-routed request's peer again: same
            // provider, same breaker permit (one address refusing is not the provider failing).
            (_, Decision::Resend(Retry::SameLeg)) => {
                rc.leg.provider.metrics.connect_retries_total.inc();
                rc.leg.addr += 1;
                rc.leg.retry = Some(Retry::SameLeg);
                if walk.is_some() {
                    warn!(
                        request_id = %rc.request_id,
                        provider = rc.leg.provider.name.as_str(),
                        candidate = at,
                        address = rc.leg.addr,
                        error = %e,
                        "upstream connect failed; trying the candidate's next address",
                    );
                } else {
                    // Without this, a partially-down provider TCP layer (or an egress-IP ban —
                    // connect is where that first bites) shows up only as extra latency on
                    // `upstream_latency_seconds`, indistinguishable from a slow model.
                    warn!(
                        request_id = %rc.request_id,
                        provider = rc.leg.provider.name.as_str(),
                        attempt = rc.leg.addr,
                        error = %e,
                        "upstream connect failed; retrying",
                    );
                }
                e.set_retry(true);
            }
            // The failure itself is recorded by `upstream_peer`'s prologue, when it moves off this
            // candidate. Recording here as well would double-count whenever there is no next
            // candidate, since `logging` would then also resolve the still-pending permit — which
            // would trip the breaker at half its configured threshold on the last candidate,
            // exactly where everything lands once the primaries are sick.
            (Some(_), decision) => {
                rc.leg.provider.metrics.connect_retries_total.inc();
                warn!(
                    request_id = %rc.request_id,
                    provider = rc.leg.provider.name.as_str(),
                    candidate = at,
                    error = %e,
                    "upstream connect failed; trying the next candidate",
                );
                // With nowhere to go `retry` stays false, so the proxy loop stops and `logging`
                // resolves the outstanding permit against this, the last candidate.
                if decision == Decision::Resend(Retry::NextCandidate) {
                    self.state.metrics.candidate_failovers_total.inc();
                    rc.fail_over(at);
                    e.set_retry(true);
                }
            }
            // A provider-routed peer out of retries: the error stands, uncounted as a retry.
            (None, _) => {}
        }
        e
    }

    async fn logging(
        &self,
        session: &mut Session,
        e: Option<&pingora_core::Error>,
        ctx: &mut Self::CTX,
    ) {
        let Ctx { rc, held } = ctx;
        // Balance the in-flight gauge and the tenant slot taken at admission. A panic that skips
        // this is covered by `Ctx`'s drop.
        held.release_in_flight();
        held.release_tenant();
        let Some(rc) = rc.as_mut() else { return };
        let managed = rc.managed();
        // Set once the response head arrived (or from the cache hit) and never after, so one read
        // serves the whole hook.
        let upstream_status = rc.upstream_status();
        // A client that closed after its stream's terminal event reached it has the whole answer:
        // the request completed, whatever the provider's end of stream was still doing.
        let e = e.filter(|e| !closed_after_terminal(rc, e));
        // Drain-on-cancel (`crate::drain`): the client left an armed stream before its terminal
        // event, and whether the gateway then read the upstream to its end (`drained`: the usage
        // tail holds the vendor's final usage).
        let left = held.client_left.load(Ordering::Relaxed);
        let client_left = left && !rc.terminal.ended();
        let drain_end = drain_end(held.drain, left, e);
        if let Some(end) = drain_end {
            held.close_drain(end);
        }
        let drained = client_left && drain_end == Some(DrainEnd::Settled);

        // An upstream error (DNS/connect timeout, read timeout, abort) lands here with `Some(e)` but
        // no `ai.usage` row (no parseable body) — and the earlier `warn!` in `upstream_peer` only
        // fires for DNS, not connect/read failures. Log it with the full identity so "why did tenant
        // 42 get 502s for 5 minutes" is one grep on the request_id, not a reconstruction.
        if let Some(e) = e {
            warn!(
                request_id = %rc.request_id,
                tenant_id = rc.billing.identity().map_or(0, |id| id.tenant_id),
                vpc_id = rc.billing.identity().map_or(0, |id| id.vpc_id),
                provider = rc.leg.provider.name.as_str(),
                error = %e,
                "upstream request errored",
            );
        }

        // Resolve the outstanding circuit-breaker permit, if this request still owes one.
        //
        // The leg is the ledger (see `Leg`): it holds the permit from when it is claimed until it
        // is taken to resolve it, so exactly one outcome lands per `allow()`. On the model-routed
        // path a candidate switch resolves the outgoing candidate's leg in `upstream_peer`, which
        // is what stops this from double-recording against whichever candidate happened to be
        // current at the end.
        //
        // Failure = the provider is *broken*: a 5xx response, or no response at all paired with an
        // upstream error (connect/read failure). Success = the provider *answered* — 2xx/3xx, and
        // deliberately **4xx/429 too**: a 429 is a healthy provider throttling our pool key, which the
        // rate limiter and the client's `Retry-After` own, NOT a reason to cut all traffic to it.
        if rc.leg.permit.is_some() {
            match upstream_status {
                // Defensive only, so its mutants are equivalent (excluded in .cargo/mutants.toml):
                // `response_filter` sets `upstream_status` and takes the permit in the same step,
                // and nothing else sets it on a request that holds a permit (a cache hit never
                // claims one), so a pending permit here always comes with `None`.
                Some(s) if s >= 500 => rc.leg.fail(),
                Some(_) => {
                    rc.leg.succeed();
                }
                // No response head arrived. Blame the provider only when the failure actually came
                // *from* upstream. Pingora tags a client-side abort `ErrorSource::Downstream` (the
                // `into_down()` at proxy_h1.rs's downstream read/write sites), and a user hitting
                // ESC on a slow turn says nothing about the provider's health. Counting those was a
                // live bug: cancellation is routine for a coding agent, and
                // `circuit_breaker_threshold` cancellations inside `circuit_breaker_window_secs`
                // would open the breaker and 503 *everyone*. Worse, `half_open_permits` is 1, so a
                // cancel-prone request drawn as the probe reopened it every time — the breaker could
                // not recover while users were cancelling.
                //
                // Nor is a client still uploading when the request died: the upstream was waiting
                // on *our* bytes, so its read timeout or reset says the client stalled, not that
                // the provider is sick. A connect failure moved no body byte, so it still counts.
                None if is_upstream_failure(e) && !client_still_uploading(session, rc) => {
                    rc.leg.fail();
                }
                // Client went away, its upload stalled, or the request ended with no error at all:
                // no provider outcome. Give the permit back without one (D86). A success here closed
                // a half-open breaker on a probe that never heard from the provider, letting every
                // caller flood one that may still be broken; `release` returns the probe permit so
                // the next request probes instead.
                None => rc.leg.release(),
            }
        }

        let cache_hit = if matches!(
            rc.auto.as_ref().and_then(|a| a.cache.as_ref()),
            Some(cache::Pending::Hit(_))
        ) {
            match rc.auto.as_mut().and_then(|a| a.cache.take()) {
                Some(cache::Pending::Hit(h)) => Some(h),
                _ => None,
            }
        } else {
            None
        };

        // The last `USAGE_TAIL_CAP` bytes of the response, oldest first (see `UsageTail`). Short
        // responses are the whole body; long ones are rotated into order here, once. Skipped on a
        // cache hit — there is no tail; tokens come from the stored entry.
        // A free sub-resource (a token count) is not a billable call: it carries no usage block,
        // writes no billing row, and is not a usage-shape regression. On every route: a catalog
        // walk records it, a `/{provider}` route names it in the forwarded path.
        let free = rc
            .auto
            .as_ref()
            .and_then(|a| a.sub)
            .or_else(|| {
                rc.forward_path
                    .as_str()
                    .and_then(route::SubResource::of_forward_path)
            })
            .is_some_and(|sub| !sub.billed());
        // Someone gave up waiting for the response head after the provider had the whole request
        // (a long reasoning turn, a huge prompt): the client, or the gateway's own read timeout
        // with the connection still up (D130). The provider cannot tell who hung up and bills that
        // prompt either way. A request that never reached it (every breaker open, a connect
        // failure) costs nothing and stays 0, and so does a peer that closed or reset the
        // connection before answering: that is a refusal, not a wait (see `gave_up_waiting`).
        let no_head = managed
            && cache_hit.is_none()
            && upstream_status.is_none()
            && e.is_some_and(gave_up_waiting)
            && body_delivered(session, rc, e);
        let (parsed, count) = if let Some(h) = cache_hit.as_ref() {
            (Some(h.usage), Count::Reported)
        } else {
            let tail = rc.resp_tail.contiguous();
            // A relayed managed 403 whose body names the key (not the request) is that key being
            // refused: cool it so later requests start past it, as a 401 does at the head (D84).
            // This request already answered; it does not walk.
            if managed && upstream_status == Some(403) && body_names_the_key(tail) {
                self.state.metrics.key_cooled(KeyCooled::KeyNamed403);
                rc.leg.provider.mark_key_bad(rc.leg.pool_key);
            }
            // A relayed managed error that says the account is out of credit or quota (Anthropic's
            // credit-balance 400, OpenAI's insufficient_quota 429: read from the body by `Redact`)
            // is that key refused until someone pays, which waiting does not fix. Cool it as a 401
            // is cooled: later requests start past it, and a catalog walk leaves the provider out
            // once all its keys cool (D180). This request already answered. Not an abandoned
            // `FullBody` attempt: its body never reaches the client and is read only if it raced
            // the parent's drop, and a walk's 402 was already judged at the head (D258).
            if managed
                && !rc.attempt.relay_abandoned
                && rc.redact.as_ref().is_some_and(|r| r.unfunded)
            {
                let status = upstream_status.unwrap_or(0);
                self.cool_unfunded_key(&rc.request_id, &rc.leg.provider, rc.leg.pool_key, status);
            }
            // Extract usage facts in the dialect this attempt was answered in. Every case reads the
            // tail; an Anthropic stream *additionally* reads the head, because that's where
            // `message_start` put the input and cache token counts. The two buffers may overlap on
            // a short response — harmless, since every field is assigned rather than accumulated.
            let parsed = if rc.streaming {
                usage::stream(rc.dialect, &rc.resp_head, tail)
            } else {
                usage::body(rc.dialect, tail)
            };
            // A managed 2xx stream that ended before its usage block: the client hung up (a
            // cancelled agent turn) or the upstream died mid-stream. The provider bills us for what
            // it generated before it noticed, so bill an estimate rather than the zero this used to
            // emit — which made "stream, then disconnect before the last event" free. Anthropic's
            // `message_start` already carries exact input and cache counts, so only the missing
            // side is estimated. See `usage`'s estimate section: input is a pre-token lower bound,
            // output a measured divisor; both err low.
            let ok_2xx = upstream_status.is_some_and(|s| (200..300).contains(&s));
            let cut_short = managed
                && rc.streaming
                && ok_2xx
                && match rc.dialect {
                    Dialect::OpenAi => parsed.is_none(),
                    // A `message_delta` that arrived but did not parse is as missing as one that
                    // never came. A usage with no token count (only a server tool's count, from a
                    // stream whose message_start was lost) is missing its tokens too: estimated.
                    Dialect::Anthropic => {
                        parsed
                            .as_ref()
                            .is_none_or(|u| u.input_tokens == 0 && u.output_tokens == 0)
                            || !usage::anthropic_stream_finished(tail)
                    }
                };
            // A non-stream 2xx that died before its `usage` (the body is last): the provider
            // generated, and bills, the whole answer; we relayed part of it. Always estimated —
            // the 2xx is the proof the provider took the request.
            let body_cut = managed && !rc.streaming && ok_2xx && parsed.is_none() && e.is_some();
            // A non-stream 2xx that ended cleanly with no usage (OpenRouter's `"usage": null` on
            // an answer that failed mid-generation, a shape change) is a turn the provider took
            // and may bill, as a finished stream without usage is: estimated, never 0/0 (D195).
            // Except a body that is only an error object (OpenRouter's error-in-200, `{"error":
            // {...}}` with no answer), the non-stream form of an error-only stream: not work we
            // were billed for. Read from the root's members, in any order (D205). The tail is the
            // whole body when the body fits in it.
            let error_only_body = !rc.streaming
                && ok_2xx
                && parsed.is_none()
                && u64::from(rc.resp_bytes) <= tail.len() as u64
                && json_is_error_only(tail);
            let body_unmetered = managed
                && !free
                && !rc.streaming
                && ok_2xx
                && parsed.is_none()
                && e.is_none()
                && !error_only_body;
            let output = if cut_short {
                usage::estimate_stream_output(tail, u64::from(rc.resp_bytes))
            } else if body_cut || body_unmetered {
                usage::estimate_body_output(tail, u64::from(rc.resp_bytes))
            } else {
                0
            };
            // The 2xx head is the provider's word that it took the request, and it bills the
            // prompt from there: a stream that went silent or died before its first event is
            // estimated too (D123), like one cut after a few. Only a 200 stream carrying nothing
            // but an error event (`overloaded_error` before any output) is not work we were billed
            // for. A finished stream whose usage we could not read (a shape change, a final event
            // we could not recover) is a turn the provider billed, never a silent 0/0.
            let started = parsed.is_some() || output > 0 || !usage::stream_carried_error(tail);
            if (cut_short && started) || body_cut || body_unmetered || no_head {
                let mut u = parsed.unwrap_or_default();
                let mut parts = EstimatedParts::default();
                if u.input_tokens == 0 {
                    u.input_tokens = input_estimate(session, rc);
                    parts.input = true;
                }
                if output > u.output_tokens {
                    u.output_tokens = output;
                    parts.output = true;
                }
                (Some(u), Count::Estimated(parts))
            } else {
                (parsed, Count::Reported)
            }
        };
        if count.estimated() {
            self.state.metrics.usage_estimated_total.inc();
        }
        // A managed 2xx response is *expected* to carry usage; `None` there means the provider's
        // usage block changed shape (a new API version, a wire change) and we're about to emit a
        // zero-token billing row that looks exactly like a (non-existent) legitimate zero-token
        // generation — silently zeroing that tenant's bill. Surface it on a counter + a warn so it
        // can be alerted on. A `None` on a 4xx/5xx (error body has no usage) is normal, not logged.
        // Cache hits never trip this: they carry the tokens stored from the fill.
        //
        // A stream that ended *cleanly* without usage is that same shape-change signal even though
        // it is now billed an estimate, so it still counts here; one cut short by an error does not.
        if (parsed.is_none() || (count.estimated() && e.is_none()))
            && let Some(id) = rc.billing.identity()
            && !free
            && cache_hit.is_none()
            && let Some(s) = upstream_status
            && (200..300).contains(&s)
        {
            self.state.metrics.usage_parse_errors_total.inc();
            warn!(
                request_id = %rc.request_id,
                tenant_id = id.tenant_id,
                provider = rc.leg.provider.name.as_str(),
                dialect = ?rc.dialect,
                stream = rc.streaming,
                status = s,
                "managed 2xx response ended cleanly without parseable usage; billing an estimate or zero",
            );
        }
        let mut usage = parsed.unwrap_or_default();
        if cache_hit.is_none() {
            merge_taps(rc, &mut usage);
        }
        // Writes caused by breakpoints the gateway added bill as input (see
        // `Usage::bill_gateway_cache_writes`). A cache hit replays the fill's already-billed usage.
        if cache_hit.is_none()
            && rc
                .auto
                .as_ref()
                .and_then(|a| a.translate.as_ref())
                .is_some_and(|t| t.gateway_cache)
        {
            usage.bill_gateway_cache_writes(usage.wire.unwrap_or(rc.dialect));
        }
        // A client's 1-hour breakpoints, carried onto Chat Completions (the OpenRouter failover):
        // billed at the 1-hour rate upstream, reported with no TTL split (see
        // `translate::chat_cache_is_1h`).
        if cache_hit.is_none()
            && rc
                .auto
                .as_ref()
                .and_then(|a| a.translate.as_ref())
                .is_some_and(|t| t.cache_1h)
        {
            usage.attribute_cache_writes_1h();
        }

        let m = &self.state.metrics;
        if cache_hit.is_some() {
            m.cache_hits_total.inc();
        }
        // Pre-resolved fixed-label children, and zeros skipped (see `Metrics::record_tokens`). Cache
        // tokens are counted here as well as in the `ai.usage` billing log below, because that log
        // ships with lag — the counter is the alerting surface for a cache-hit-rate cliff after a
        // deploy.
        m.record_tokens(&usage);
        // Read the clock once for both consumers below. Beyond saving a vDSO call, this is a
        // correctness fix: the latency histogram and the `ai.usage` billing line used to call
        // `elapsed()` about forty lines apart, so they reported *different* durations for the same
        // request and could never be reconciled against each other.
        let elapsed = rc.start.elapsed();
        if cache_hit.is_none() {
            rc.leg
                .provider
                .metrics
                .upstream_latency_seconds
                .observe(elapsed.as_secs_f64());
            // Balance the `active_streams` increment from `response_filter` (or `Ctx`'s drop does).
            held.release_stream();
        }

        // Emit the priced usage row on a dedicated target — **managed only**. The event is an
        // identity-keyed billing record (logfwd/OTLP ships `ai.usage` → ClickHouse → beyond, which
        // batches it for invoicing); BYO carries no Beyond identity, so a BYO event would be a billing row
        // with `tenant_id=0` — unbillable, unattributable, and a footgun for any consumer that sums
        // without filtering it out. Aggregate gateway throughput (incl. BYO) is already covered by
        // the Prometheus metrics above, which is the right tool for non-billing observability.
        // Whether the request writes a row at all, and what the row says of how it ended, is its
        // `Ending`'s (see `Ending::row`).
        if let Some(&id) = rc.billing.identity()
            && let Some(end) = Ending::of(&EndFacts {
                relay_abandoned: rc.attempt.relay_abandoned,
                free,
                cache_hit: cache_hit.is_some(),
                status: upstream_status,
                fault: e.map(Fault::of),
                client_left,
                called: rc.upstream_phase != UpstreamPhase::None,
                drained,
                count,
            })
            .row()
        {
            // What the client asked for that changes the price (read before the borrows below).
            let requested = requested_knobs(session, rc);
            // Emit BOTH models. `model` is the one the *provider* resolved + billed (echoed in its
            // response) — the key for pricing AND for reconciling against the provider's invoice,
            // which itemizes by the pinned snapshot. `requested_model` is the alias the client sent —
            // product analytics ("what they asked for") and a fallback rate when a snapshot is newer
            // than the downstream price table. They're equal when the response carried no model (e.g.
            // an error body), where `model` falls back to the request alias. Both sanitized.
            let billed = rc.resp_model_scanner.take_model().map(sanitize_model);
            let models = Models::of(
                cache_hit.as_deref(),
                rc.auto.as_ref().map(|a| a.route.model),
                &rc.model,
                billed.as_deref(),
            );
            if models.header_disagrees {
                self.state.metrics.model_header_body_mismatch_total.inc();
            }
            let Models {
                billed: billed_model,
                requested: requested_model,
                routed: routed_model,
                ..
            } = models;
            billing::emit(
                self.state,
                session,
                rc,
                &Row {
                    id,
                    end: &end,
                    usage: &usage,
                    models: &models,
                    requested: &requested,
                    cache_hit: cache_hit.as_deref(),
                    upstream_status,
                    elapsed,
                },
            );

            // Payload capture, on its own target so a stalled payload sink can never delay or drop a
            // billing row (see `main::init_tracing`, which gives the two targets different writers).
            // Correlated by `request_id`, which is also on the line above and in the response's
            // `x-beyond-request-id` header — so a user quoting that id resolves straight to their
            // conversation with no join table.
            if cache_hit.is_none()
                && let Some(cap) = rc.control.as_ref().and_then(|c| c.capture.as_ref())
            {
                m.captures_total.inc();
                m.capture_bytes_total.inc_by(cap.bytes() as u64);
                info!(
                    target: "ai.payload",
                    request_id = %rc.request_id,
                    tenant_id = id.tenant_id,
                    provider = rc.leg.provider.name.as_str(),
                    model = billed_model,
                    stream = rc.streaming,
                    status = upstream_status,
                    metadata = rc.control.as_ref().and_then(|c| c.metadata.as_deref()),
                    // What the client sent, before either body rewrite (see `request_body_filter`).
                    request_body = cap.req_str(),
                    response_body = cap.resp_str(),
                    // Truncation and completeness are separate facts and both are load-bearing.
                    // `truncated` means we hit the byte cap; `complete` means the upstream actually
                    // finished. A capture that reads as whole when it isn't produces confident wrong
                    // conclusions during the incident this feature exists to serve — and "the stream
                    // died at token 400" is frequently the answer, so a partial is kept, not dropped.
                    request_truncated = cap.req_truncated(),
                    response_truncated = cap.resp_truncated(),
                    complete = e.is_none() && !client_left,
                    "payload"
                );
            }

            // Fill: complete 2xx only. Client abort, 4xx/5xx, and truncation are all skips — a
            // partial or error body must never be replayed as a success.
            if e.is_none()
                && !end.count.estimated()
                && upstream_status.is_some_and(|s| (200..300).contains(&s))
                && let Some(cache::Pending::Fill {
                    key,
                    tap,
                    content_type,
                }) = rc.auto.as_mut().and_then(|a| a.cache.take())
                && !tap.truncated()
                && let Some(body) = tap.complete_body()
                && let Some(store) = &self.state.cache
            {
                store.insert(
                    key,
                    cache::CachedResponse {
                        status: upstream_status.unwrap_or(200),
                        content_type: content_type.unwrap_or_else(|| "application/json".into()),
                        body: Bytes::copy_from_slice(body),
                        usage: usage.for_cache(),
                        billed_model: billed_model.to_owned().into_boxed_str(),
                        requested_model: requested_model.to_owned().into_boxed_str(),
                        routed_model,
                        provider: rc.leg.provider.name.clone().into_boxed_str(),
                        streaming: rc.streaming,
                    },
                );
            }
        }
    }
}
