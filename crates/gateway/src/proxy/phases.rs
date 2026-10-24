//! `request_filter`'s phases, in order, and the types each hands the next.
//!
//! [`AiProxy::arrive`] stamps the request; then [`AiProxy::admit_request`] runs the rest, each
//! phase returning its decision or the [`Refusal`] that ends the request:
//!
//! | phase                         | decides                                     | hands on                   |
//! | ----------------------------- | ------------------------------------------- | -------------------------- |
//! | [`AiProxy::route`]            | what the first path segment names           | [`Route`]                  |
//! | [`AiProxy::authenticate`]     | the presented key, the guards, its identity | `Option<Identity>`         |
//! | [`admit`]                     | what the route is, for this key             | [`Admission`]              |
//! | [`managed_endpoint`]          | whether a managed key may call this         |                            |
//! | [`AiProxy::read_body`]        | the catalog row, and the body if read       | [`Read`]                   |
//! | [`AiProxy::dispatch`]         | the walk, the wire, the cache, the slot     | [`Admitted`]               |
//!
//! The order is the refusal precedence (`tests/refusal_precedence.rs`): a phase runs only once
//! every earlier one passed. Each phase returns its refusal rather than writing it, and
//! `request_filter` writes it in one place, so no answer is written mid-phase.

use super::*;

/// What `request_filter` knows of a request on arrival, before checking any of it.
pub(super) struct Arrival {
    /// Set only on a subrequest the gateway spawned to re-run a large body (see `FullBody`).
    pub(super) full_body: Option<FullBody>,
    pub(super) start: Instant,
    pub(super) deadline: u64,
    /// One id per request, made before any refusal so even a 400/401 carries it (in the log line
    /// and the `x-beyond-request-id` header). A `FullBody` attempt is the parent's request, so it
    /// keeps the parent's id and seq.
    pub(super) request_id: RequestId,
    pub(super) request_seq: u64,
    /// The `Content-Length` the client declared, if it parses.
    pub(super) declared_len: Option<usize>,
}

/// Why `request_filter` stopped short of admitting a request.
pub(super) enum Stop {
    /// The gateway refuses it: `request_filter` counts, logs and writes the answer.
    Refused(Refusal),
    /// Pingora's own error: a body read failed, or a reply could not be written.
    Failed(Box<pingora_core::Error>),
}

impl From<Refusal> for Stop {
    fn from(refusal: Refusal) -> Self {
        Stop::Refused(refusal)
    }
}

impl From<Box<pingora_core::Error>> for Stop {
    fn from(e: Box<pingora_core::Error>) -> Self {
        Stop::Failed(e)
    }
}

/// What the first path segment resolved to, before the key says whether the request may walk the
/// catalog.
///
/// Holds only routes that go somewhere: a path that resolves to nothing is a [`RouteMiss`], so no
/// later step has to wonder whether it is looking at one.
pub(super) enum Route {
    /// `/{provider}/…`. The provider is named by the request. Not a catalog walk — the escape hatch.
    Provider {
        provider: &'static Arc<Provider>,
        /// The inbound path with the `/{provider}` segment stripped, query kept.
        forward_path: String,
    },
    /// Bare `/v1` default. BYO keeps the dialect-picked provider; managed becomes a catalog walk.
    BareDefault { provider: &'static Arc<Provider> },
    /// `/auto/…`. `header` is the catalog row if `x-beyond-model` resolved; `None` means peek the
    /// body's root `model` after identity is known. An *unknown* header is [`RouteMiss::UnknownModel`],
    /// not this — the header wins, and a catalog miss is the allowlist.
    Auto {
        header: Option<&'static route::ModelRoute>,
    },
}

/// A path that resolves to nothing the gateway serves. Each shape gets its own rejection (404
/// unknown provider, 404 unknown model, 400 ambiguous credential), which is why it is a sum type and
/// not an `Option`.
pub(super) enum RouteMiss {
    /// A routing header (on `/auto`) naming a model we do not serve. Collapsed with a missing body
    /// model into one 404: a value we cannot match is a value we do not serve.
    UnknownModel,
    /// The first segment matches no provider, is not the bare default, and is not `/auto`.
    UnknownProvider,
    /// Bare `/v1` with BYO keys for two different providers: a 400, never a guess.
    AmbiguousCredential,
}

/// What `request_filter` decided a request is, once its key is known: carried whole through
/// admission, so the route is never re-derived from which of several locals happen to be set.
pub(super) enum Plan {
    /// `/{provider}/…`, or a BYO bare `/v1` on its dialect-default provider.
    Provider {
        provider: &'static Arc<Provider>,
        /// [`ForwardPath::Stripped`] on `/{provider}/…`, [`ForwardPath::Inbound`] on the bare
        /// default.
        forward: ForwardPath,
    },
    /// A managed catalog walk (`/auto`, or a managed bare `/v1`) over a resolved row. Only a
    /// verified key gets one, so it carries the identity it was admitted on.
    Catalog(Walk),
}

/// A request on its way to a [`Plan`] once its key is known: ready, or a managed catalog walk that
/// still has to read its row from `x-beyond-model` or the body's root `model`.
pub(super) enum Admission {
    Ready(Plan),
    NeedsRow(Identity),
}

/// A catalog walk as `request_filter` admits it.
pub(super) struct Walk {
    pub(super) id: Identity,
    pub(super) row: &'static route::ModelRoute,
    /// The body, read in full, when it outgrew pingora's retry buffer: the request is re-run as a
    /// [`FullBody`] subrequest carrying it rather than proxied here.
    pub(super) relay: Option<Vec<u8>>,
}

/// The [`Plan`] once the body is read as far as it needs to be.
pub(super) struct Read {
    pub(super) plan: Plan,
    /// The whole body, when it was read up front and fits pingora's replay buffer (a larger one
    /// is the walk's `relay`).
    pub(super) body: Option<Vec<u8>>,
}

/// The candidates a request walks, as `dispatch` settled them. A provider route walks no catalog,
/// so its lineup is its one provider.
pub(super) struct Lineup {
    /// The first attempt's provider.
    pub(super) provider: Arc<Provider>,
    /// Bit i ⇒ walk slot i is registered here *and* has a pool key. Computed once; every later
    /// attempt reads this instead of re-deriving it.
    pub(super) usable: u8,
    pub(super) walk: control::Walk,
    pub(super) arms: &'static [route::Candidate],
    /// The client asked for priority processing (`route::SpeedAsk::Priority`).
    pub(super) priority: bool,
}

/// An admitted request's context, and how it is answered.
pub(super) enum Admitted {
    /// Proxied upstream.
    Proxy(RequestCtx),
    /// Answered from the response cache: the stored response is written, and its context then
    /// carries the hit for `logging`.
    Hit(Box<cache::CachedResponse>, RequestCtx),
}

impl Admission {
    /// A `FullBody` re-run walks the row its parent chose.
    pub(super) fn rerun(self, row: &'static route::ModelRoute) -> Self {
        match self {
            Admission::NeedsRow(id) => Admission::Ready(Plan::Catalog(Walk {
                id,
                row,
                relay: None,
            })),
            Admission::Ready(Plan::Catalog(w)) => {
                Admission::Ready(Plan::Catalog(Walk { row, ..w }))
            }
            ready => ready,
        }
    }
}

impl Plan {
    /// The catalog row this request walks; `None` on a provider route.
    pub(super) fn row(&self) -> Option<&'static route::ModelRoute> {
        match self {
            Plan::Catalog(w) => Some(w.row),
            Plan::Provider { .. } => None,
        }
    }

    /// The forwarded `/{provider}` path (query included), when the client named the provider.
    pub(super) fn stripped(&self) -> Option<&str> {
        match self {
            Plan::Provider {
                forward: ForwardPath::Stripped(p),
                ..
            } => Some(p),
            _ => None,
        }
    }
}

/// What path an attempt sends upstream.
#[derive(Debug)]
pub(super) enum ForwardPath {
    /// The inbound path and query, as the client sent them (the bare `/v1` default). The common
    /// route builds nothing, so it allocates nothing. On a managed request its `key` params are the
    /// virtual key, stripped as the head goes out.
    Inbound,
    /// The client's path with the `/{provider}` segment stripped, its query kept (minus a managed
    /// request's `key` params). Forwarded **verbatim** — the gateway does no per-provider path
    /// rewriting. Built once, so held as a `Box<str>`: beside `Catalog`'s `String` that keeps the
    /// enum at a `String`'s size.
    Stripped(Box<str>),
    /// The serving candidate's own catalog path (plus a sub-resource's suffix), set as each
    /// attempt dials it; rewritten in place, so a failover allocates nothing after the first.
    /// Taken wholesale from the catalog rather than composed from a mount and the client's suffix:
    /// providers disagree on where an endpoint lives and the disagreement is not a prefix.
    Catalog(String),
}

impl ForwardPath {
    /// The path (and query) to set on the upstream request; `None` keeps the inbound one.
    pub(super) fn as_str(&self) -> Option<&str> {
        match self {
            ForwardPath::Inbound => None,
            ForwardPath::Stripped(p) => Some(p),
            ForwardPath::Catalog(p) => Some(p),
        }
    }
}

impl AiProxy {
    /// Count the request, bound the client's reads and writes, and give it its id and deadline.
    pub(super) fn arrive(&self, session: &mut Session, held: &mut Held) -> Arrival {
        let full_body = full_body_ctx(session);
        if full_body.is_none() {
            self.state.metrics.requests_total.inc();
            // A client that stops reading is cut off after this long rather than holding its slot,
            // its upstream connection and any breaker permit forever (see the config field).
            let secs = self.state.config.client_write_timeout_secs();
            if secs > 0 {
                session
                    .downstream_session
                    .set_write_timeout(Some(Duration::from_secs(secs)));
            }
            // A client that stops sending its body is bounded the same way: per read, pingora's
            // own timeout on HTTP/1.1 (a no-op on HTTP/2, which `peek_body_model` bounds itself).
            let secs = self.state.config.client_read_timeout_secs();
            session
                .downstream_session
                .set_read_timeout((secs > 0).then(|| Duration::from_secs(secs)));
        }
        let start = Instant::now();
        let deadline = match &full_body {
            Some(fb) => fb.deadline,
            None => crate::deadline::at(start, self.state.config.request_max_secs()),
        };
        // Cheap: a counter bump + a short `format!` (see `next_request_id`).
        let (request_id, request_seq) = match &full_body {
            Some(fb) => (fb.request_id, fb.request_seq),
            None => self.state.next_request_id_seq(),
        };
        held.request_id = Some(request_id);
        let declared_len = session
            .req_header()
            .headers
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<usize>().ok());
        Arrival {
            full_body,
            start,
            deadline,
            request_id,
            request_seq,
            declared_len,
        }
    }

    /// Every phase after arrival, in refusal-precedence order. `Ok(true)` when the gateway answered
    /// the request itself (the models list, a cache hit, a `FullBody` relay).
    ///
    /// `early_slot` is the tenant slot a full body read takes, owned by the caller: a refusal
    /// written after the read still holds it, as it holds the body budget.
    pub(super) async fn admit_request(
        &self,
        session: &mut Session,
        ctx: &mut Ctx,
        arrival: &Arrival,
        early_slot: &mut Option<TenantSlot<'static>>,
    ) -> std::result::Result<bool, Stop> {
        let route = self.route(session)?;
        let identity = self.authenticate(session.req_header(), &route, arrival)?;
        let managed = identity.is_some();
        let admission = admit(route, identity)?;

        // Stock SDKs list models at GET /v1/models. A managed caller is answered here, from the
        // catalog rows this deployment's pool keys can serve (built once at boot), so an empty GET
        // is not a missing-model 404. A BYO caller never uses the catalog: like any BYO `/v1`
        // request it relays to the provider its key belongs to, which answers with its own list.
        if managed && is_v1_models_list(session) {
            let body = self.state.models_list.clone();
            return Ok(Self::reply_models_list_boxed(session, &arrival.request_id, body).await?);
        }
        if managed && arrival.full_body.is_none() {
            managed_endpoint(session.req_header(), &admission)?;
        }
        let admission = match &arrival.full_body {
            Some(fb) => admission.rerun(fb.route),
            None => admission,
        };

        let Read { mut plan, body } = self
            .read_body(session, &mut ctx.held, admission, arrival, early_slot)
            .await?;
        // A body read in full past pingora's retry buffer: re-run the request as a `FullBody`
        // subrequest that carries it, rather than proxy it here.
        if let Plan::Catalog(w) = &mut plan
            && let Some(whole) = w.relay.take()
        {
            // --- signed ids (`signed_id`) ---
            let path = session.req_header().uri.path();
            if let Some(refusal) = self.foreign_ids(path, w.row, &whole, w.id.tenant_id) {
                return Err(refusal.into());
            }
            let slot_held = early_slot.is_some();
            let parent = Parent {
                request_id: arrival.request_id,
                request_seq: arrival.request_seq,
                deadline: arrival.deadline,
            };
            let relayed = self
                .relay_full_body(session, parent, w.row, whole, slot_held)
                .await;
            drop(early_slot.take());
            return Ok(relayed?);
        }

        // Settled in a `let`, so the dispatch result is not kept across the cache hit's write
        // (its temporaries would be, in every request's future).
        let (hit, mut rc) = match self.dispatch(
            session,
            &mut ctx.held,
            arrival,
            plan,
            identity,
            body,
            early_slot,
        )? {
            Admitted::Proxy(rc) => {
                ctx.rc = Some(rc);
                // Admitted: count it in-flight. Released in `logging`, or by `Ctx`'s drop if a
                // panic skipped `logging`, so the gauge cannot leak. `active_streams` only covers
                // SSE; this covers every request.
                ctx.held.admit();
                self.state.fault_point("request_filter");
                return Ok(false);
            }
            Admitted::Hit(hit, rc) => (hit, rc),
        };
        // The context is kept only once the stored answer is written: a failed write leaves the
        // request unadmitted, with no row.
        Self::reply_cache_hit_boxed(session, &arrival.request_id, &hit).await?;
        if let Some(auto) = rc.auto.as_deref_mut() {
            auto.cache = Some(cache::Pending::Hit(hit));
        }
        ctx.rc = Some(rc);
        ctx.held.admit();
        Ok(true)
    }

    /// Route by the **first path segment** = provider; forward the rest of the path verbatim
    /// (native passthrough — the gateway holds no per-provider mount knowledge). A path with no
    /// provider segment that is exactly `/v1` or starts with `/v1/` (boundary-checked — see
    /// `bare_default_provider_name`/`route::is_default_prefix`, not a raw prefix test) is the
    /// drop-in default: dialect picks openai/anthropic and the path is forwarded as-is. Anything
    /// else → unknown provider (404). Resolved before auth: an unknown route is cheap.
    ///
    /// Catalog resolution needs identity: `/auto` is managed-only, a candidate is only usable if
    /// we hold a pool key for it, and managed `/v1` peeks the body only after the key is known
    /// (BYO `/v1` stays dialect-default passthrough and must not drain the body). The route is
    /// kept whole until then, and becomes a [`Plan`] once the key says which it is.
    pub(super) fn route(&self, session: &Session) -> std::result::Result<Route, Refusal> {
        self.resolve(session).map_err(|miss| match miss {
            RouteMiss::UnknownModel => header_catalog_miss(session),
            RouteMiss::UnknownProvider => Refusal::UnknownProvider,
            RouteMiss::AmbiguousCredential => Refusal::AmbiguousCredential,
        })
    }

    fn resolve(&self, session: &Session) -> std::result::Result<Route, RouteMiss> {
        let req = session.req_header();
        let uri = &req.uri;
        let path = uri.path();
        let query = uri.query();
        // `nth(1)`: `/openai/v1/…` → "openai"; `/v1/…` → "v1"; "/" or "" → "".
        let first = path.split('/').nth(1).unwrap_or("");
        let with_query = |p: &str| match query {
            // Sized exactly, so the `Box<str>` it becomes keeps the allocation.
            Some(q) => [p, "?", q].concat(),
            None => p.to_string(),
        };
        // A config-added provider written with capitals (`MiniMax`) is served at its name, and at
        // `/minimax/…`: the table is lowercase, since config names are lowercased at load. A
        // built-in name is exact (`/OpenAI/…` is no route). The retry runs on a miss only.
        let provider = self.state.provider(first).or_else(|| {
            let lower = first.to_ascii_lowercase();
            (lower != first && providers::by_name(&lower).is_none())
                .then(|| self.state.provider(&lower))
                .flatten()
        });
        if let Some(p) = provider {
            // Provider-prefixed: strip the leading `/{first}` segment, forward the remainder.
            // `first` is non-empty here (an empty first segment matches no provider), so this
            // always differs from the inbound path and always needs the URI rewritten.
            let rest = &path[1 + first.len()..];
            let rest = if rest.is_empty() { "/" } else { rest };
            Ok(Route::Provider {
                provider: p,
                forward_path: with_query(rest),
            })
        } else if route::is_default_prefix(path) {
            // Bare default: the BYO credential, else the path's dialect, picks the provider;
            // managed traffic becomes a catalog walk after identity. Path is forwarded
            // unchanged for BYO (`None`).
            match byo_credential_dialect(req) {
                // BYO keys for two providers. A managed key anywhere makes the request managed
                // (every credential location is stripped), so only BYO is ambiguous.
                Err(()) if !extract_virtual_key(req).is_some_and(|p| p.managed) => {
                    Err(RouteMiss::AmbiguousCredential)
                }
                credential => match bare_default_provider_name(path, credential.ok().flatten())
                    .and_then(|name| self.state.provider(name))
                {
                    Some(p) => Ok(Route::BareDefault { provider: p }),
                    None => Err(RouteMiss::UnknownProvider),
                },
            }
        } else if first == route::AUTO_SEGMENT {
            // Model-routed. Reached only after a provider-table miss, so the established routes
            // pay nothing for this arm — and `state::build_providers` refuses to boot with a
            // provider named `auto`, so the miss is guaranteed rather than merely likely.
            // An unknown header 404s here (header wins, catalog miss is the allowlist). An
            // *absent* header waits until after identity so a managed caller can put `model` in
            // the body the way a stock SDK does.
            match catalog_from_header(session) {
                CatalogHeader::Known(route) => Ok(Route::Auto {
                    header: Some(route),
                }),
                CatalogHeader::Absent => Ok(Route::Auto { header: None }),
                CatalogHeader::Unknown => Err(RouteMiss::UnknownModel),
            }
        } else {
            Err(RouteMiss::UnknownProvider)
        }
    }

    /// The presented key — a managed virtual key (`bai_v1…`) or a raw BYO provider token — and
    /// who it proves: `Some` identity for a managed key that passed every check, `None` for BYO.
    pub(super) fn authenticate(
        &self,
        req: &pingora::http::RequestHeader,
        route: &Route,
        arrival: &Arrival,
    ) -> std::result::Result<Option<Identity>, Refusal> {
        let Some(Presented {
            key: raw_key,
            managed: managed_key,
        }) = extract_virtual_key(req)
        else {
            return Err(Refusal::MissingKey);
        };

        // Rate guardrails (see `ratelimit`), charged on the *raw presented key* **before** any
        // verification or upstream connect. Keying on the credential we already hold (rather than
        // the verified tenant id) is what lets this sit ahead of the Ed25519 verify: a single
        // leaked, runaway, or forged key can't drive unbounded crypto work (per-credential tier),
        // and a flood of distinct random BYO tokens can't drive junk-auth connects to providers
        // from our egress IPs (global BYO tier — managed traffic is exempt, see `ratelimit`).
        if arrival.full_body.is_none()
            && let Some(rl) = &self.state.rate_limit
            && let Some(throttled) = rl.check_at(raw_key, managed_key, arrival.start)
        {
            return Err(Refusal::RateLimited(throttled));
        }

        // Reject oversized bodies up front (Content-Length) so we never buffer a huge upload.
        if let Some(len) = arrival.declared_len
            && len > MAX_REQUEST_BODY
        {
            return Err(Refusal::BodyTooLarge);
        }

        // Identity + key handling. One branch: `bai_v1`/`bai_v2`/`bai_v3` is fail-closed (prefix match →
        // verify; any verify failure is 401, never BYO). Anything else → BYO: the user's own
        // provider token, passed through unchanged (no Beyond identity, so no deny-set, no
        // allowance, and no per-tenant attribution). A public listener without this split would
        // forward a forged virtual key as BYO — junk-auth egress, and the rate guard already
        // exempted it from the BYO aggregate.
        if !managed_key {
            return Ok(None);
        }
        // One deny-set load serves both the distrusted-kid check inside verify and the tenant/key
        // check below. Expiry is judged here, at admission: a stream already admitted is not cut
        // when its key passes `exp`, the same rule as a deny.
        let deny = self.state.deny.load();
        let identity = match self.state.keyring.verify_with(
            raw_key,
            &self.state.live_keys.load(),
            &deny,
            crate::key::unix_now,
        ) {
            Ok(identity) => identity,
            Err(crate::key::KeyError::Expired) => return Err(Refusal::KeyExpired),
            Err(_) => return Err(Refusal::InvalidKey),
        };
        // Deny-set: O(1), default-allow. Tenant deny OR (v2/v3) key deny. Tenant wins if both.
        if let Some(reason) = deny.reason_for(identity.tenant_id, identity.key_id) {
            // Counted per reason (`Refusal::metric`): an `Unknown` is a control plane writing a
            // reason this build does not know (a deploy ahead of the gateway's), and folded into
            // `deny_fraud` it would spike the fraud signal and mask real fraud.
            return Err(Refusal::Denied(reason));
        }
        // Allowance: remaining-ok vs exhausted, same WatchedSet shape as deny. Fail-closed
        // while the set has not been read. 402 before cache and before `upstream_peer`.
        if let Some(reason) = self
            .state
            .allowance
            .load()
            .reason_for(identity.tenant_id, identity.key_id)
        {
            return Err(match reason {
                crate::allowance::AllowanceReject::Exhausted => Refusal::QuotaExhausted,
                // Not seeded yet: transient, so a retryable 503 with `Retry-After`, not the
                // 402 an SDK treats as final ("quota exhausted" above is the final one).
                crate::allowance::AllowanceReject::Unavailable => Refusal::AllowanceUnavailable,
            });
        }
        // The actual `Bearer …`/`x-api-key` value is precomputed in the provider registry and
        // applied in `upstream_request_filter`; here we only confirm a pool key exists.
        //
        // Skipped for a catalog walk: there a pool key is a property of each *candidate*,
        // and the first one lacking a key is a reason to try the next, not to fail the request.
        // The equivalent gate is the usable-candidate check in `dispatch`. Also skipped for
        // managed `/v1` before the body peek — the dialect-default provider is not the allowlist.
        if let Route::Provider { provider, .. } = route
            && !provider.has_pool_key()
        {
            return Err(Refusal::NoPoolKey);
        }
        Ok(Some(identity))
    }

    /// Settle the catalog row and read the body as far as the walk needs it.
    ///
    /// A headerless catalog walk reads the whole body before choosing a row, rather than
    /// stopping at `model`: a large body can only fail over once fully read (see `FullBody`), a
    /// small one fits pingora's replay buffer either way and costs nothing more, and with the
    /// whole body in hand the gateway can hash it for the cache and read Responses session
    /// fields.
    pub(super) async fn read_body(
        &self,
        session: &mut Session,
        held: &mut Held,
        admission: Admission,
        arrival: &Arrival,
        early_slot: &mut Option<TenantSlot<'static>>,
    ) -> std::result::Result<Read, Stop> {
        let responses = route::is_responses_path(session.req_header().uri.path());
        let large = arrival.declared_len.is_none_or(past_replay_buffer);
        let mut body = None;
        let mut peeked = false;
        let mut plan = match admission {
            Admission::Ready(plan) => plan,
            Admission::NeedsRow(id) => {
                // Header wins if present (unknown → 404, no fall-through to the body). Absent → peek.
                let (row, relay) = match catalog_from_header(session) {
                    CatalogHeader::Known(route) => (route, None),
                    CatalogHeader::Unknown => return Err(header_catalog_miss(session).into()),
                    CatalogHeader::Absent => {
                        let peek = self
                            .read_whole(session, held, &id, arrival, early_slot)
                            .await?;
                        peeked = true;
                        let Some(name) = peek.model.filter(|n| !n.is_empty()) else {
                            return Err(Refusal::CatalogMiss {
                                name: None,
                                header: true,
                            }
                            .into());
                        };
                        let Some(route) = route::model_route(&name) else {
                            return Err(Refusal::CatalogMiss {
                                name: Some(name.into()),
                                header: true,
                            }
                            .into());
                        };
                        if peek.relay {
                            (route, Some(peek.complete.unwrap_or_default()))
                        } else {
                            body = peek.complete;
                            (route, None)
                        }
                    }
                };
                Plan::Catalog(Walk { id, row, relay })
            }
        };

        // Header-won catalog walks skip the body peek above. The same reasons to read the whole
        // body apply (Responses session state, a body pingora could not replay for failover).
        if let Plan::Catalog(w) = &mut plan
            && arrival.full_body.is_none()
            && !peeked
            && (responses
                || large
                || route::walk_reads_body(w.row)
                || route::walk_reads_tools(w.row, session.req_header().uri.path()))
        {
            let peek = self
                .read_whole(session, held, &w.id, arrival, early_slot)
                .await?;
            if peek.relay {
                w.relay = Some(peek.complete.unwrap_or_default());
            } else {
                body = peek.complete;
            }
        }
        Ok(Read { plan, body })
    }

    /// Read the whole body of a catalog walk. The tenant slot is taken first, so
    /// `tenant_max_in_flight` bounds the bodies held in memory and not only the requests in
    /// flight; a `FullBody` re-run's parent already holds it.
    async fn read_whole(
        &self,
        session: &mut Session,
        held: &mut Held,
        id: &Identity,
        arrival: &Arrival,
        early_slot: &mut Option<TenantSlot<'static>>,
    ) -> std::result::Result<BodyPeek, Stop> {
        if arrival.full_body.is_none() {
            match self.take_slot(id) {
                Ok(guard) => *early_slot = guard,
                Err(()) => return Err(Refusal::TenantBusy.into()),
            }
        }
        // A large body is held twice (here and in its `FullBody` re-run): reserve both before
        // reading a byte of it.
        if let Some(n) = arrival.declared_len.filter(|&n| past_replay_buffer(n))
            && !held.reserve_body(n.saturating_mul(2))
        {
            return Err(Refusal::BodyMemory.into());
        }
        let reserve = arrival.declared_len.unwrap_or(0).min(MAX_REQUEST_BODY);
        let peek = Box::pin(peek_body_model(
            session,
            reserve,
            held,
            self.up_front_read_timeout(session),
            arrival.deadline,
        ))
        .await?;
        if peek.over_cap {
            return Err(Refusal::BodyTooLarge.into());
        }
        if peek.over_budget {
            return Err(Refusal::BodyMemory.into());
        }
        Ok(peek)
    }

    /// Everything decided once the body is in hand, none of it waiting on I/O: the body rules,
    /// the control surface, the catalog walk, signed ids, the wire, the cache, the tenant slot
    /// and the breaker, in that order, and the request context they make.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch(
        &self,
        session: &Session,
        held: &mut Held,
        arrival: &Arrival,
        plan: Plan,
        identity: Option<Identity>,
        body: Option<Vec<u8>>,
        early_slot: &mut Option<TenantSlot<'static>>,
    ) -> std::result::Result<Admitted, Refusal> {
        let Arrival {
            ref full_body,
            start,
            deadline,
            request_id,
            request_seq,
            declared_len,
        } = *arrival;
        let managed = identity.is_some();
        let body_complete = body;
        let model_route = plan.row();
        let path = session.req_header().uri.path();

        // Two root `model` keys, refused before connecting wherever the body is already in
        // hand, with a JSON 400 and the request id. A body streamed through is still refused
        // in `request_body_filter`, after the request headers went upstream.
        let shape = shape(model_route, body_complete.as_deref())?;

        // A `FullBody` re-run's body is its parent's, already reserved for both copies.
        held.body_exempt = full_body.is_some();

        // Per-request control surface (`x-beyond-*`). Managed only: a BYO request carries no verified
        // identity, so a tag on it would be an unattributable row and a capture would be storing
        // prompts we can't attribute to an account that asked us to — the same reason `ai.usage`
        // itself is managed-only.
        //
        // Parsed here, before the candidate walk, so `order` / `only` / `split` can permute the
        // list `first_usable` sees. Nothing here can reject the request: `Control::parse` drops
        // what it can't use and counts it. See `control`'s module docs.
        let parsed_control = if managed {
            let parsed = control::Control::parse(session.req_header());
            if parsed.malformed {
                self.state.metrics.control_header_errors_total.inc();
            }
            if parsed.unknown_provider {
                self.state.metrics.control_unknown_provider_total.inc();
            }
            Some(parsed)
        } else {
            None
        };

        let inbound_responses = model_route.is_some() && route::is_responses_path(path);
        let session_field = match full_body {
            Some(fb) => fb.session_field,
            None if inbound_responses => translate::responses_session_field(
                body_complete.as_deref().unwrap_or(&[]),
                model_route.is_some_and(|r| !r.responses.is_empty()),
            ),
            None => None,
        };
        // `/v1/messages/count_tokens`, `/v1/responses/compact`, … (see `route::SubResource`).
        let sub = model_route.and(route::SubResource::of_path(path));
        let lineup = match &plan {
            // Provider-routed, or BYO `/v1` dialect default.
            Plan::Provider { provider, .. } => Lineup {
                provider: Arc::clone(provider),
                usable: 0,
                walk: control::Walk::identity(0),
                arms: &[],
                priority: false,
            },
            // Only a managed key walks the catalog (a BYO token belongs to exactly one provider, so
            // selecting among candidates would be a guess, and failing over would send its key to
            // another vendor): BYO was refused before a `Walk` could be made.
            Plan::Catalog(w) => self.walk(
                session,
                w,
                arrival,
                body_complete.as_deref(),
                shape,
                parsed_control.as_ref(),
                sub,
                session_field,
                inbound_responses,
            )?,
        };
        let Lineup {
            provider,
            usable,
            walk,
            arms: walk_arms,
            priority,
        } = lineup;

        let signed = self.sign(
            session,
            identity,
            model_route,
            &plan,
            body_complete.as_deref(),
        )?;

        // A sub-resource is its provider's own API: no wire check, no translation.
        let translate_state = match model_route.filter(|_| sub.is_none()) {
            Some(row) => wire(row, path)?,
            None => None,
        };

        // Dialect drives usage parsing and injection eligibility.
        //
        // For a model-routed request it is seeded from the **row** here and overwritten in
        // `upstream_peer` from **this candidate's path**. A provider can serve more than one
        // wire (OpenRouter Chat Completions *and* Messages), and a mixed row can list both, so
        // neither the provider table nor the row's declared wire is the serving dialect.
        // Reading the wrong one hands an Anthropic response to the OpenAI usage extractor,
        // which does not error — it trips the dialect-mismatch guard and emits a **zero-token
        // billing row**.
        //
        // A provider-routed request is the same problem without a catalog: `/{provider}/…`
        // forwards a path, and that path — not the provider — says which wire answers. OpenRouter
        // serves Messages at `/api/v1/messages` and Anthropic serves Chat Completions at
        // `/v1/chat/completions`, so the provider's default dialect metered both as zero.
        let dialect = match model_route {
            Some(r) => r.wire,
            None => plan
                .stripped()
                .map_or(provider.dialect, wire_of_forward_path),
        };
        // Mark OpenAI managed chat/completions streams for body buffering + `stream_options` injection
        // (handled in `request_body_filter`). A catalog walk asks of the arm this request will
        // actually walk, not the row's Chat Completions primary: a Responses walk must not inherit
        // `stream_options` injection (each candidate re-decides in `apply_serving_candidate`).
        let stream_options = match model_route {
            Some(_) => walk_arms
                .first()
                .is_some_and(|c| stream_options_eligible(managed, dialect, c.path)),
            None => plan
                .stripped()
                .is_some_and(|p| stream_options_eligible(managed, dialect, p)),
        };
        // Every billable managed `/{provider}/…` body is buffered too, so a model outside the
        // catalog is refused before a byte of it goes upstream: a managed key runs only what its
        // billing row can price. So is a `/responses` body's `background: true` (a
        // generation the provider runs after the request ends, which no usage tap can meter).
        // The free token counts are left to stream: nothing there bills. A catalog walk
        // checked its body above.
        let catalog_check = managed
            && plan
                .stripped()
                .and_then(route::forward_endpoint)
                .is_some_and(|e| e.sub.is_none_or(route::SubResource::billed));
        // That buffer counts against the body budget. A declared large body reserves up front,
        // before the tenant slot and the breaker permit, so a refusal holds neither; a chunked one
        // reserves as it grows (`request_body_filter`).
        if (stream_options || catalog_check)
            && model_route.is_none()
            && let Some(n) = declared_len.filter(|&n| past_replay_buffer(n))
            && !held.reserve_body(n)
        {
            return Err(Refusal::BodyMemory);
        }

        let control = self.capture(parsed_control, identity, request_seq);

        // Exact-match cache: lookup only when the pre-rewrite body is already in hand (headerless
        // managed catalog walk). A hit writes the stored 2xx and returns before the breaker, the
        // key-walk, and `upstream_peer`. A miss stays an unbuffered relay; the fill is a tap.
        let cache_bypass = !managed || cache::request_bypasses(session.req_header());
        let cache_look = if let Some(id) = identity
            && !cache_bypass
            && model_route.is_some()
        {
            body_complete.as_deref().and_then(|body| {
                let store = self.state.cache.as_ref()?;
                let req = session.req_header();
                let (ids, n) = walk.provider_ids(walk_arms);
                let ck = cache::key(&cache::KeyParts {
                    tenant_id: id.tenant_id,
                    method: req.method.as_str(),
                    inbound_path: req.uri.path(),
                    model: model_route.map_or("", |r| r.model),
                    headers: &req.headers,
                    body,
                    providers: &ids[..usize::from(n)],
                });
                match store.get(&ck) {
                    Some(hit) => Some(Err(Box::new(hit))),
                    None => Some(Ok((ck, store.max_bytes()))),
                }
            })
        } else {
            None
        };
        let forward_path = match plan {
            Plan::Provider { forward, .. } => forward,
            Plan::Catalog(_) => ForwardPath::Inbound,
        };
        let mut pending_cache: Option<cache::Pending> = None;
        match cache_look {
            Some(Err(hit)) => {
                let rc = RequestCtx {
                    billing: Billing::of(identity),
                    dialect,
                    leg: Leg::new(provider, None, 0),
                    forward_path,
                    control,
                    model: String::new(),
                    attempt: Attempt::new(Vec::new()),
                    resp_model_scanner: peek::ModelScanner::for_response(),
                    // Not an upstream stream — `response_filter` never ran, so `active_streams`
                    // was never incremented. The stored `hit.streaming` is emitted on `ai.usage`.
                    streaming: false,
                    plan: BodyPlan::new(false, false, model_route.is_some(), false),
                    resp_tail: UsageTail::default(),
                    resp_head: Vec::new(),
                    upstream_status: held_status(hit.status),
                    start,
                    deadline,
                    // The hit itself joins the walk once its answer is written.
                    auto: model_route.map(|route| {
                        Box::new(ModelRouting {
                            route,
                            sub,
                            candidate: first_usable(usable, 0).unwrap_or(0),
                            usable,
                            walk,
                            arms: walk_arms,
                            session_field,
                            attempt_start: start,
                            cache: None,
                            translate: None,
                            health: None,
                            addrs: 0,
                            attempted: false,
                            open_retry_after: None,
                            priority,
                        })
                    }),
                    request_id,
                    tally_eager: false,
                    resp_bytes: 0,
                    upstream_phase: UpstreamPhase::None,
                    redact: None,
                    terminal: TerminalTracker::default(),
                    signed: None,
                };
                return Ok(Admitted::Hit(hit, rc));
            }
            Some(Ok((ck, max_bytes))) => {
                pending_cache = Some(cache::Pending::Fill {
                    key: ck,
                    tap: cache::ResponseTap::new(max_bytes),
                    content_type: None,
                });
            }
            None => {}
        }

        let (slot, permit) = self.claim(
            early_slot,
            identity,
            &provider,
            model_route.is_some(),
            full_body.as_ref().is_some_and(|fb| fb.slot_held),
        )?;
        // Past any key cooling off from a 401; a catalog walk picks per candidate.
        let pool_key = provider.first_key();
        held.tenant = slot;

        let mut rc = RequestCtx {
            billing: Billing::of(identity),
            dialect,
            leg: Leg::new(provider, permit, pool_key),
            forward_path,
            control,
            model: String::new(),
            // Its body buffer is pre-sized below, once the context says whether this request
            // rewrites its body.
            attempt: Attempt::new(Vec::new()),
            // `for_response`, not `new`: a response may carry the model nested under `message`
            // (Anthropic's `message_start`), and a root-only scanner would neither find it nor ever
            // stop looking. See `ModelScanner::for_response`.
            resp_model_scanner: peek::ModelScanner::for_response(),
            streaming: false,
            plan: BodyPlan::new(
                stream_options,
                catalog_check,
                model_route.is_some(),
                signed.is_some(),
            ),
            // Grown lazily by the response tap (`response_body_filter`), not pre-reserved: a
            // non-streaming response — the common case — is a few hundred bytes, so reserving the
            // full 64KB cap up front would waste an allocation on every request to hold ~200B. A
            // long stream grows it geometrically to the bounded 2×cap and compacts; that handful of
            // reallocs is lost in the network noise of a stream we're already relaying chunk by chunk.
            resp_tail: UsageTail::default(),
            // Grown lazily, and only on the one path that needs it (Anthropic SSE) — see
            // `response_body_filter`. Every other response leaves this empty and never allocates.
            resp_head: Vec::new(),
            upstream_status: None,
            start,
            deadline,
            auto: model_route.map(|route| {
                Box::new(ModelRouting {
                    route,
                    sub,
                    // `first_usable` picked this candidate above; `upstream_peer` re-derives it from
                    // here on.
                    candidate: first_usable(usable, 0).unwrap_or(0),
                    usable,
                    walk,
                    arms: walk_arms,
                    session_field,
                    // Overwritten per attempt by `upstream_peer`; seeded so the first attempt is
                    // timed even if it fails before the prologue runs.
                    attempt_start: start,
                    cache: pending_cache,
                    translate: translate_state,
                    health: None,
                    addrs: 0,
                    attempted: false,
                    open_retry_after: None,
                    priority,
                })
            }),
            request_id,
            tally_eager: tally_eager(managed, model_route.is_some(), declared_len),
            resp_bytes: 0,
            upstream_phase: UpstreamPhase::None,
            redact: None,
            terminal: TerminalTracker::default(),
            signed,
        };
        // A body buffered for a rewrite (`BodyPlan::rewrites_body`) is pre-sized from the
        // declared Content-Length, so accumulation is a single allocation instead of a geometric
        // realloc chain; capped at `MAX_REQUEST_BODY` so a lying header can't pre-allocate
        // unbounded memory. Every other request leaves it empty and never buffers.
        //
        // The `+ STREAM_OPTIONS_FRAG.len()` is headroom for the splice, which
        // `ScannedBody::into_upstream` performs *in place*: with it the injection never
        // reallocates, so a body arrives, is spliced, and goes upstream on one allocation.
        if rc.plan.rewrites_body()
            && let Some(len) = declared_len
        {
            rc.attempt.req_buf =
                Vec::with_capacity(len.min(MAX_REQUEST_BODY) + STREAM_OPTIONS_FRAG.len());
        }
        Ok(Admitted::Proxy(rc))
    }

    /// A catalog walk's candidates: the arm it walks, the order, which candidates may take it,
    /// and the first one's provider. The body rules that read what the row accepts run first.
    #[allow(clippy::too_many_arguments)]
    fn walk(
        &self,
        session: &Session,
        w: &Walk,
        arrival: &Arrival,
        body_complete: Option<&[u8]>,
        shape: body_rules::Facts<'static>,
        parsed_control: Option<&control::Control>,
        sub: Option<route::SubResource>,
        session_field: Option<&'static str>,
        inbound_responses: bool,
    ) -> std::result::Result<Lineup, Refusal> {
        let full_body = &arrival.full_body;
        let (row, id) = (w.row, w.id);
        // The rest of the body rules, after the control surface is parsed (and counted). `shape`
        // passed, so the order is still `body_rules::Rule::ORDER`.
        let facts = body_rules::Facts {
            refused_input: body_complete
                .and_then(|b| route::refused_input(row, b))
                .map(|kind| (row.model, kind)),
            background: inbound_responses && body_complete.is_some_and(requests_background),
            ..shape
        };
        if let Some(refusal) = body_rules::refusal(body_rules::Point::Connect, facts) {
            return Err(refusal);
        }
        // A Responses request walks the row's Responses arm whenever the row has one,
        // `store: false` one-shots included: there it is a byte relay, and translation onto
        // Chat Completions would lose what only Responses has (Codex's `namespace` tools,
        // `custom` grammars, encrypted reasoning). Only a row without an arm translates,
        // and only a one-shot may (session state there is the 400 below). A failover stays
        // inside the arm it walks, in catalog order: the session pin orders `candidates`
        // only.
        // An embeddings row has no Responses arm either, but "store cannot be honored"
        // would name a field the caller may never have set: it walks its candidates and the
        // wire check rejects the endpoint, which is what is actually wrong.
        let embeddings_row = route::Endpoint::of_row(row) == route::Endpoint::Embeddings;
        // A sub-resource walks the arms that hold its provider's parent endpoint: OpenAI's
        // `/v1/responses` sits in a GPT row's Responses arm, or first in a Responses-first
        // row's candidates.
        let arms: &'static [route::Candidate] = match sub {
            Some(route::SubResource::InputTokens | route::SubResource::Compact)
                if !row.responses.is_empty() =>
            {
                row.responses
            }
            Some(_) => row.candidates,
            None if inbound_responses
                && !embeddings_row
                && (session_field.is_some() || !row.responses.is_empty()) =>
            {
                row.responses
            }
            // More tools than Chat Completions takes, from a translated client: the
            // Responses arm takes them (`route::tools_need_responses_arm`).
            None if match full_body {
                Some(fb) => fb.responses_tools,
                None => body_complete.is_some_and(|b| {
                    route::tools_need_responses_arm(
                        row,
                        route::implied_endpoint(session.req_header().uri.path()),
                        b,
                    )
                }),
            } =>
            {
                row.responses
            }
            None => row.candidates,
        };
        if inbound_responses
            && let Some(field) = session_field
            && arms.is_empty()
        {
            return Err(Refusal::SessionField {
                field,
                model: row.model,
            });
        }
        // Permute the row before the usable mask / `first_usable`. Failover, breakers, and
        // the 429 key-walk then see this sequence. Unknown names were already dropped;
        // an `only` filter that left nobody is the same 503 as an unkeyed row.
        //
        // `order` / `split` fix the walk; otherwise the caller's computed session pin orders
        // it (`pin`). `only` filters, then the pin applies to what is left. Responses walks
        // and sub-resources keep catalog order.
        let mut walk = parsed_control.map_or_else(
            || control::Walk::identity(arms.len()),
            |c| c.catalog_walk(arms, arrival.request_seq),
        );
        // Bit `orig` ⇒ `arms[orig]` can be sent to: registered here with a pool key, not
        // skipped by a large-body re-run or held off by a key walk, and serving this
        // sub-resource. Computed before the session pin, which moves what cannot take the
        // request behind what can; the usable mask below is this, by walk slot.
        let mut dispatchable = 0u8;
        // Bit `orig` ⇒ every pool key of that arm's provider is cooling off from a refusal
        // (a 401, an out-of-credit answer): left out below while another arm can serve.
        let mut cooling = 0u8;
        // Each arm's provider, as resolved here: a dispatchable arm has one.
        let mut resolved: [Option<&'static Arc<Provider>>; route::MAX_CANDIDATES] =
            [None; route::MAX_CANDIDATES];
        for (orig, c) in arms.iter().enumerate().take(route::MAX_CANDIDATES) {
            let provider = self.state.provider_by_id(c.provider);
            resolved[orig] = provider;
            let orig = orig as u8;
            let keyed = provider.is_some_and(|p| p.has_pool_key());
            if provider.is_some_and(|p| p.cooling()) {
                cooling |= 1 << orig;
            }
            // A re-run of a large body skips the candidates earlier attempts failed on. A
            // key walk in progress is not a filter: its candidate goes first below
            // (`FullBody::resume`), and the rest stay for failover.
            let failed = full_body
                .as_ref()
                .is_some_and(|fb| fb.skip & (1 << orig) != 0);
            let serves = sub.is_none_or(|sub| sub.serves(c));
            if keyed && !failed && serves {
                dispatchable |= 1 << orig;
            }
        }
        // A candidate that cannot serve this body (`route::unserved`: Bedrock and a
        // JSON-schema output) is not dispatched to or pinned onto, unless nothing else
        // can take the request, when the provider's own error is the answer.
        let unserved = match (full_body, body_complete) {
            // The parent computed it against the row's candidates.
            (Some(fb), _) if std::ptr::eq(arms, row.candidates) => fb.unserved,
            (Some(_), _) => 0,
            (None, Some(b)) => route::unserved(arms, b),
            (None, None) => 0,
        };
        // Anthropic fast mode. A Messages client's `speed: "fast"` is served fast or
        // refused, as Anthropic itself does: only a candidate that serves fast mode may
        // take it, and a walk with none is a 400 here rather than an answer at standard
        // speed (or Bedrock's own refusal) under a fast request. A Chat Completions or
        // Responses client's `service_tier: "priority"` is a best effort and moves nothing:
        // it is mapped to fast mode on whichever attempt reaches a candidate that serves it.
        let mut priority = false;
        if sub.is_none() {
            let scanned = match full_body {
                Some(fb) => Some(&fb.body[..]),
                None => body_complete,
            };
            let client = route::implied_endpoint(session.req_header().uri.path());
            match scanned.map(|b| route::speed_ask(client, b)) {
                Some(route::SpeedAsk::Fast) => {
                    let slow = route::fast_unserved(arms);
                    if walk.mask() & !slow == 0 {
                        return Err(Refusal::RefusedInput {
                            model: row.model,
                            kind: "fast mode",
                        });
                    }
                    dispatchable &= !slow;
                }
                Some(route::SpeedAsk::Priority) => priority = true,
                _ => {}
            }
        }
        let walked = walk.mask();
        if dispatchable & walked & !unserved != 0 {
            dispatchable &= !unserved;
        }
        // A provider whose every key was refused within `KEY_COOLDOWN` (revoked, or out of
        // credit) is not sent to while another candidate can take the request; when
        // none can, it is tried anyway, and its own answer is the client's.
        if dispatchable & walked & !cooling != 0 {
            dispatchable &= !cooling;
        }
        if self.state.config.session_pins()
            && sub.is_none()
            && std::ptr::eq(arms, row.candidates)
            && !parsed_control.is_some_and(control::Control::pins_walk)
        {
            let affinity = pin::affinity(id.tenant_id, id.vpc_id, id.key_id);
            if let Some(pinned) = pin::order(walk, row, affinity, dispatchable) {
                walk = pinned;
                self.state.metrics.session_pinned_total.inc();
            }
        }
        // A large body's re-run resumes its key walk on that candidate first; the rest of
        // the walk stays behind it for failover (see `FullBody::resume`).
        if let Some(orig) = full_body.as_ref().and_then(|fb| fb.resume) {
            walk_front(&mut walk, orig);
        }
        if walk.len == 0 {
            return Err(Refusal::NoCandidate);
        }
        let mut usable = 0u8;
        for i in 0..walk.len {
            if walk
                .catalog_index(i)
                .is_some_and(|orig| dispatchable & (1 << orig) != 0)
            {
                usable |= 1 << i;
            }
        }
        if let Some(sub) = sub
            && !arms.iter().any(|c| sub.serves(c))
        {
            return Err(Refusal::NoSubResource {
                model: row.model,
                provider: sub.provider_name(),
                path: session.req_header().uri.path().into(),
            });
        }
        // The first usable slot's provider: a usable slot is a dispatchable arm, which
        // resolved one.
        let first = first_usable(usable, 0)
            .and_then(|i| walk.catalog_index(i))
            .and_then(|orig| resolved.get(usize::from(orig)).copied().flatten());
        let Some(first) = first else {
            // Every remaining candidate is unkeyed. Distinct from `circuit_open`, which
            // means the candidates exist and are being skipped while they recover —
            // `doctor`'s `model_catalog` check exists to catch this configuration at boot
            // instead.
            return Err(Refusal::NoCandidate);
        };
        Ok(Lineup {
            provider: Arc::clone(first),
            usable,
            walk,
            arms,
            priority,
        })
    }

    /// Signed ids (`signed_id`): Responses state a provider keeps is tenant-bound. A managed
    /// Responses relay to a store (a GPT row's Responses arm, `/{provider}/…/responses`) must not
    /// carry an id another tenant could resolve. The ids the client sent back are checked here,
    /// before connecting, wherever the body is in hand (every catalog walk); a `/{provider}` body
    /// streams, and is checked as it is stripped in `request_body_filter`.
    fn sign(
        &self,
        session: &Session,
        identity: Option<Identity>,
        model_route: Option<&'static route::ModelRoute>,
        plan: &Plan,
        body_complete: Option<&[u8]>,
    ) -> std::result::Result<Option<Box<signed_id::Relay>>, Refusal> {
        let Some(id) = identity else {
            return Ok(None);
        };
        let applies = match model_route {
            Some(row) => signed_id::catalog_applies(session.req_header().uri.path(), row),
            None => plan
                .stripped()
                .is_some_and(signed_id::provider_route_applies),
        };
        if !applies {
            return Ok(None);
        }
        let Some(signer) = self.state.id_signer.as_ref() else {
            return Err(Refusal::IdSigningUnset);
        };
        if let Some(refusal) = body_complete.and_then(|b| {
            signer
                .unsign_request(id.tenant_id, b, model_route.is_some())
                .err()
        }) {
            return Err(Refusal::ForeignId(refusal));
        }
        Ok(Some(Box::new(signed_id::Relay::new(id.tenant_id))))
    }

    /// Capture decision from the control surface. The header wins in **both** directions over
    /// the tenant's control-plane rule (Cloudflare's `cf-aig-collect-log` semantics), and the two
    /// enablers serve different people: the control plane is the operator's, works
    /// retroactively, and needs no cooperation from a client we don't control; the header is the
    /// caller's, for per-request precision.
    fn capture(
        &self,
        parsed_control: Option<control::Control>,
        identity: Option<Identity>,
        request_seq: u64,
    ) -> Option<Box<RequestControl>> {
        let parsed = parsed_control?;
        let id = identity?;
        let capture = match parsed.capture {
            // Explicit suppression always wins — "not this one, it has PII".
            Some(false) => None,
            // Explicitly requested, so **never sampled away**: a caller who asks to log one
            // trace and silently gets nothing is the single outcome that makes this useless.
            Some(true) => Some(CaptureBufs::new(self.state.capture_defaults.max_bytes)),
            // No opinion — fall back to the tenant's rule. Sparse miss is ~1.5ns (see `deny`'s
            // hasher rationale, which this set shares) and is what the overwhelming majority of
            // managed requests hit.
            None => self
                .state
                .capture
                .load()
                .rule_for(id.tenant_id)
                .filter(|rule| rule.samples(request_seq))
                .map(|rule| CaptureBufs::new(rule.max_bytes)),
        };

        // Allocate the box only when something is actually on. A managed request that used
        // neither feature — nearly all of them — stays at `None`.
        (parsed.metadata.is_some() || capture.is_some()).then(|| {
            Box::new(RequestControl {
                metadata: parsed.metadata,
                capture,
            })
        })
    }

    /// The tenant slot and the breaker permit, the last claims before the connection.
    ///
    /// Per-tenant in-flight cap — the bound on overspend while the allowance-set lags (see
    /// `concurrency`). After the cache (a hit costs no provider spend) and before the breaker,
    /// so a refused request never holds a half-open probe permit.
    ///
    /// Circuit breaker (per provider, all traffic — a down provider is down regardless of whose
    /// key is used). Checked here, after every other rejection, so claiming a half-open probe
    /// permit corresponds to an *actual* upstream attempt — and balanced by exactly one
    /// `record_*` in `logging` (which runs once per admitted request), so a permit can't leak.
    /// When open, fast-fail 503 instead of piling the request against `read_timeout_secs` and
    /// exhausting connection/in-flight slots for every provider. 5xx/connect failures trip it;
    /// 429 never does (that's a healthy provider throttling — see `logging`).
    ///
    /// **Not** for a model-routed request: which provider it attempts is not settled until
    /// `upstream_peer` picks a candidate, and gating here would claim a permit against candidate
    /// 0 and then claim a second one against whichever candidate is actually tried. That path
    /// gates per candidate instead, at the same "last thing before the connection" position.
    fn claim(
        &self,
        early_slot: &mut Option<TenantSlot<'static>>,
        identity: Option<Identity>,
        provider: &Provider,
        catalog: bool,
        parent_holds_slot: bool,
    ) -> std::result::Result<(Option<TenantSlot<'static>>, Option<Permit>), Refusal> {
        let slot = match (early_slot.take(), identity) {
            // A `FullBody` attempt whose parent holds the slot: neither take nor release one.
            _ if parent_holds_slot => None,
            // Taken before the body was read.
            (Some(slot), _) => Some(slot),
            (None, Some(id)) => match self.take_slot(&id) {
                Ok(slot) => slot,
                Err(()) => return Err(Refusal::TenantBusy),
            },
            (None, None) => None,
        };
        let permit = match (catalog, &provider.breaker) {
            (false, Some(breaker)) => match breaker.allow() {
                Ok(permit) => Some(permit),
                // `slot` is dropped as this returns, before the refusal is written: a slow
                // client must not hold it.
                Err(_) => {
                    return Err(Refusal::CircuitOpen {
                        retry_after: breaker.retry_after_secs(),
                    });
                }
            },
            // The leg owes this permit until an outcome or a release resolves it (see `Leg`). A
            // model-routed request owes none here: it claims each candidate's in `upstream_peer`.
            _ => None,
        };
        Ok((slot, permit))
    }

    /// This tenant's concurrency slot. `Ok(None)` when no cap is configured; `Err` when the tenant
    /// is at its cap.
    pub(super) fn take_slot(
        &self,
        id: &Identity,
    ) -> std::result::Result<Option<TenantSlot<'static>>, ()> {
        match self.state.tenant_slots.as_ref() {
            None => Ok(None),
            Some(slots) => slots.take(id).map(Some).ok_or(()),
        }
    }

    /// `signed_id`: a large catalog body's ids, checked before its first attempt connects (its
    /// `FullBody` re-runs see the body only as it streams in). `Some` when the request is refused.
    pub(super) fn foreign_ids(
        &self,
        path: &str,
        route: &'static route::ModelRoute,
        body: &[u8],
        tenant_id: u64,
    ) -> Option<Refusal> {
        if !signed_id::catalog_applies(path, route) {
            return None;
        }
        match self.state.id_signer.as_ref() {
            None => Some(Refusal::IdSigningUnset),
            Some(signer) => signer
                .unsign_request(tenant_id, body, true)
                .err()
                .map(Refusal::ForeignId),
        }
    }
}

/// What the route is for this key.
pub(super) fn admit(
    route: Route,
    identity: Option<Identity>,
) -> std::result::Result<Admission, Refusal> {
    let managed = identity.is_some();
    Ok(match (route, identity) {
        // A managed `?key=` is the virtual key: never forward it. Catalog walks send the
        // candidate's own path with no query, so only `/{provider}/…` carries one.
        (
            Route::Provider {
                provider,
                forward_path,
            },
            _,
        ) => Admission::Ready(Plan::Provider {
            provider,
            forward: ForwardPath::Stripped(
                match managed {
                    true => strip_key_param(&forward_path).unwrap_or(forward_path),
                    false => forward_path,
                }
                .into_boxed_str(),
            ),
        }),
        (Route::BareDefault { provider }, None) => Admission::Ready(Plan::Provider {
            provider,
            forward: ForwardPath::Inbound,
        }),
        // Catalog walk is **managed-only**. BYO on `/auto` 400s before any peek — a BYO token
        // belongs to one vendor, and reading the body to pick among candidates would still be
        // a guess (and a failover would send that key to a different vendor). BYO on `/v1`
        // keeps the dialect-default provider (above) and never consults the catalog.
        (Route::Auto { .. }, None) => return Err(Refusal::ByoOnModelRoute),
        (Route::Auto { header: Some(row) }, Some(id)) => Admission::Ready(Plan::Catalog(Walk {
            id,
            row,
            relay: None,
        })),
        (Route::BareDefault { .. } | Route::Auto { header: None }, Some(id)) => {
            Admission::NeedsRow(id)
        }
    })
}

/// A managed key spends Beyond's shared pool key, so it reaches only metered generation calls:
/// POST, and on `/{provider}/…` only a generation endpoint. Anything else (listing or reading
/// stored files and responses, batches, fine-tuning, a GET relayed to a generation path) would
/// let one tenant reach another's data on the shared key and run unmetered. BYO keys are the
/// caller's own and are never checked.
pub(super) fn managed_endpoint(
    req: &pingora::http::RequestHeader,
    admission: &Admission,
) -> std::result::Result<(), Refusal> {
    // A protocol upgrade (WebSocket: OpenAI Realtime, Codex) would turn the request into an
    // opaque relay on the pool key that no usage tap can meter. Refused whatever the path, so
    // the allowlist below is not the only thing standing in its way.
    if req.headers.contains_key(http::header::UPGRADE) {
        return Err(Refusal::ProtocolUpgrade);
    }
    let method_ok = req.method == http::Method::POST;
    // Only a request that names its provider is held to the allowlist; a catalog walk's paths
    // are the catalog's own.
    let endpoint_ok = match admission {
        Admission::Ready(plan) => plan.stripped().is_none_or(|p| {
            is_managed_provider_endpoint(p.split_once('?').map_or(p, |(path, _)| path))
        }),
        Admission::NeedsRow(_) => true,
    };
    if method_ok && endpoint_ok {
        return Ok(());
    }
    // An endpoint outside the allowlist is 404 whatever the method; a wrong method on an allowed
    // endpoint (or any catalog path) is 405.
    Err(Refusal::ManagedEndpoint {
        method: req.method.clone(),
        path: clip_catalog_name(req.uri.path()).into(),
        wrong_method: endpoint_ok,
    })
}

/// The body rules (`body_rules`) that read the body's shape, checked as soon as it is scanned: a
/// second `model`, and an output limit that is not a whole number on a row with a published cap.
/// What the row accepts is checked once the walk is shaped. One walk of the body serves both.
pub(super) fn shape(
    model_route: Option<&'static route::ModelRoute>,
    body_complete: Option<&[u8]>,
) -> std::result::Result<body_rules::Facts<'static>, Refusal> {
    let scan = body_complete
        .filter(|_| model_route.is_some())
        .map(peek::scan_buffered);
    let shape = body_rules::Facts {
        duplicate_model: scan.as_ref().is_some_and(|s| s.duplicate_model),
        inexact_limit: model_route.is_some_and(|r| r.card.output_cap().is_some())
            && scan.as_ref().is_some_and(|s| s.inexact_limit),
        ..body_rules::Facts::default()
    };
    match body_rules::refusal(body_rules::Point::Connect, shape) {
        Some(refusal) => Err(refusal),
        None => Ok(shape),
    }
}

/// Catalog walk: the inbound path may name Chat Completions, Messages, or Responses while the
/// serving *candidate* speaks a different one of those three. The client endpoint is kept so
/// failover can translate onto the next path; embeddings against a generation row (or the
/// reverse) is a 400. Inbound Responses with session state already chose the Responses arm (or
/// 400'd) — same-endpoint is a byte relay (`from == to`). `/{provider}/…` never reaches this — it
/// has no row.
pub(super) fn wire(
    row: &'static route::ModelRoute,
    path: &str,
) -> std::result::Result<Option<translate::TranslateState>, Refusal> {
    let row_endpoint = route::Endpoint::of_row(row);
    match route::catalog_wire_action(path, row_endpoint) {
        route::WireAction::Reject => Err(Refusal::WrongWire {
            model: row.model,
            hint: row_endpoint.post_hint(),
        }),
        route::WireAction::Relay => Ok(Some(translate::TranslateState::new(
            route::implied_endpoint(path).unwrap_or(row_endpoint),
        ))),
        route::WireAction::Translate { client } => Ok(Some(translate::TranslateState::new(client))),
    }
}

/// The `x-beyond-model` header, viewed as a catalog lookup. Present-but-unknown is distinct from
/// absent: the header wins, so an unknown header is a 404 rather than a fall-through to the body.
#[derive(Clone, Copy)]
pub(super) enum CatalogHeader {
    Absent,
    Known(&'static route::ModelRoute),
    Unknown,
}

pub(super) fn catalog_from_header(session: &Session) -> CatalogHeader {
    match session
        .req_header()
        .headers
        .get(route::MODEL_HEADER)
        .and_then(|v| v.to_str().ok())
    {
        None => CatalogHeader::Absent,
        Some(name) => match route::model_route(name) {
            Some(r) => CatalogHeader::Known(r),
            None => CatalogHeader::Unknown,
        },
    }
}
