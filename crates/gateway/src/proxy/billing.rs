//! Who pays for a request, the facts its `ai.usage` row is built from, and how it ended.

use super::*;

/// Who pays for a request, and so whether anything tenant-scoped applies to it.
///
/// The only way to a tenant id is `Billing::Managed`, built from what `Keyring::verify` proved
/// (an [`Identity`]), so no unverified id can reach the deny set, a capture rule or a billing row,
/// and a BYO request has no tenant id at all rather than a zero one.
pub(super) enum Billing {
    /// A Beyond virtual key: the pool key is swapped in, and the request writes a billing row.
    Managed(Managed),
    /// The caller's own provider key, passed through untouched: no identity, no billing row.
    Byo,
}

/// A managed request's verified identity, and the billing facts its row reads.
pub(super) struct Managed {
    identity: Identity,
    /// Billing facts tapped beside the usage block ([`BillingTaps`]), made at the first one.
    /// Boxed for [`ModelRouting`]'s reason; `None` until a fact is tapped (a cache hit taps none).
    taps: Option<Box<BillingTaps>>,
}

impl Billing {
    /// Managed for a verified identity, BYO without one.
    pub(super) fn of(identity: Option<Identity>) -> Self {
        match identity {
            Some(identity) => Billing::Managed(Managed {
                identity,
                taps: None,
            }),
            None => Billing::Byo,
        }
    }

    pub(super) fn identity(&self) -> Option<&Identity> {
        match self {
            Billing::Managed(m) => Some(&m.identity),
            Billing::Byo => None,
        }
    }

    /// The facts tapped so far; `None` on BYO and before the first.
    pub(super) fn taps(&self) -> Option<&BillingTaps> {
        match self {
            Billing::Managed(m) => m.taps.as_deref(),
            Billing::Byo => None,
        }
    }

    /// See [`Self::taps`].
    pub(super) fn taps_mut(&mut self) -> Option<&mut BillingTaps> {
        match self {
            Billing::Managed(m) => m.taps.as_deref_mut(),
            Billing::Byo => None,
        }
    }

    /// A managed request's taps, made here at its first fact; `None` on BYO, which taps nothing.
    pub(super) fn tap(&mut self) -> Option<&mut BillingTaps> {
        match self {
            Billing::Managed(m) => Some(m.taps.get_or_insert_with(Box::default)),
            Billing::Byo => None,
        }
    }
}

/// Billing facts a managed request taps beside its usage block, for the `ai.usage` row.
#[derive(Default)]
pub(super) struct BillingTaps {
    /// The request's price knobs ([`peek::REQUEST_KNOB_KEYS`]), fed as the body streams only when
    /// no copy of it will be left for `logging` (`tally_eager`); otherwise `logging` scans the copy
    /// (see [`requested_knobs`]).
    pub(super) knobs: Option<(peek::ModelScanner, peek::Kept)>,
    /// What `resp_model_scanner` keeps beside the model: the response's id and serving host
    /// ([`peek::RESPONSE_EXTRA_KEYS`]).
    pub(super) resp_kept: Option<peek::Kept>,
    /// Hosted-tool items of a Responses answer (see [`usage::ToolTally`]).
    pub(super) tools: Option<usage::ToolTally>,
    /// OpenRouter's `X-Generation-Id` response header: the id its generation API takes.
    generation_id: Option<usage::IdStr>,
    /// The vendor's request id header (`request-id`, `x-request-id`, `x-amzn-requestid`): what its
    /// support and logs key on.
    pub(super) request_id: Option<usage::IdStr>,
    /// The request offers OpenAI's `web_search_preview` tool: its `web_search_call` items are
    /// counted as `web_search_preview`, priced apart (see `unpriced::Inspection`).
    pub(super) web_search_preview: bool,
}

/// What a request asked for that changes its price, as the client sent it (the served values come
/// from the response). String knobs are kept when they are short tokens; object and array knobs
/// (`provider`, `plugins`, an object `container`) as raw JSON, at most [`peek::RAW_CAPTURE`] bytes.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct RequestedKnobs {
    pub service_tier: Option<usage::ServiceTier>,
    pub speed: Option<usage::ServiceTier>,
    pub inference_geo: Option<usage::ServiceTier>,
    pub provider_routing: Option<String>,
    pub plugins: Option<String>,
    pub container: Option<String>,
}

impl RequestedKnobs {
    pub(super) fn from_scan(mut s: peek::Kept) -> Self {
        let tier = |v: Option<String>| v.as_deref().and_then(usage::tier_from);
        // A raw value is JSON the scanner copied byte for byte; a string knob arrives unquoted.
        let raw = |v: Option<String>| v.filter(|v| !v.is_empty());
        RequestedKnobs {
            service_tier: tier(s.take(0)),
            speed: tier(s.take(1)),
            inference_geo: tier(s.take(2)),
            provider_routing: raw(s.take(3)),
            plugins: raw(s.take(4)),
            container: raw(s.take(5)),
        }
    }
}

/// The request's price knobs: from the scan fed as the body streamed (`tally_eager`), else from
/// the body `logging` still holds — the `FullBody` parent's copy or pingora's retry buffer, the
/// copies [`input_estimate`] reads. That is the client's body, before any rewrite.
pub(super) fn requested_knobs(session: &Session, rc: &mut RequestCtx) -> RequestedKnobs {
    if let Some((_, k)) = rc.billing.taps_mut().and_then(|t| t.knobs.take()) {
        return RequestedKnobs::from_scan(k);
    }
    let mut kept = peek::Kept::request_knobs();
    if let Some(body) = full_body_ctx(session)
        .map(|fb| fb.body)
        .or_else(|| session.as_ref().get_retry_buffer())
    {
        peek::ModelScanner::new().feed_keeping(&body, &mut kept);
    }
    RequestedKnobs::from_scan(kept)
}

/// Which price a provider applies to the endpoint that served, where one provider has several:
/// Bedrock's `global.` inference profiles are its base price, and its geographic profiles (`us.`,
/// `eu.`, `jp.`, `au.`, `apac.`) and single-region model ids carry a 10% premium on Claude 4.5
/// and later; OpenRouter's in-region hosts (`us.openrouter.ai`, `eu.openrouter.ai`) pass on the
/// provider's regional surcharge. `None` where the provider has one price.
pub(super) fn price_variant(
    provider: &str,
    upstream_model: &str,
    host: &str,
) -> Option<&'static str> {
    match provider {
        "bedrock" => Some(if upstream_model.starts_with("global.") {
            "global"
        } else {
            "regional"
        }),
        "openrouter" => {
            (host != "openrouter.ai" && host.ends_with(".openrouter.ai")).then_some("regional")
        }
        _ => None,
    }
}

/// Providers documented to keep generating, and billing, after the client disconnects
/// mid-stream: a cancelled row's tokens are what was relayed, not what was billed.
pub(super) const MAY_CONTINUE: [ProviderId; 3] = [
    ProviderId::OpenRouter,
    ProviderId::Bedrock,
    ProviderId::Groq,
];

/// [`MAY_CONTINUE`] by name: a billing row names its provider, and its facts are derived from the
/// row's own fields (as its price is).
pub(super) const MAY_CONTINUE_PROVIDERS: [&str; 3] = [
    MAY_CONTINUE[0].name(),
    MAY_CONTINUE[1].name(),
    MAY_CONTINUE[2].name(),
];

/// Whether this row's provider may have generated (and billed) past the point the row counts:
/// a stream the client cancelled, or one cut short, on a provider in [`MAY_CONTINUE_PROVIDERS`],
/// unless the gateway read that stream to its end after the client left (`drained`): then the row
/// counts everything the provider generated.
pub(super) fn upstream_may_continue(
    provider: Option<&str>,
    outcome: &str,
    streaming: bool,
    drained: bool,
) -> bool {
    streaming
        && !drained
        && matches!(outcome, "client_cancelled" | "cut_short")
        && provider.is_some_and(|p| MAY_CONTINUE_PROVIDERS.contains(&p))
}

/// The upstream response header carrying the vendor's request id, in the order they are tried.
pub(super) const REQUEST_ID_HEADERS: [&str; 3] = ["request-id", "x-request-id", "x-amzn-requestid"];

/// At a managed response head: read the vendor's ids from its headers, and start counting
/// hosted-tool items when a Responses endpoint answered. A few map lookups per response; a fresh
/// tally per head, so an earlier attempt's never leaks into the one that serves.
pub(super) fn tap_response_head(rc: &mut RequestCtx, resp: &ResponseHeader) {
    let header_id = |name: &str| {
        resp.headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .and_then(usage::id_from)
    };
    let generation_id = header_id("x-generation-id");
    let request_id = REQUEST_ID_HEADERS.iter().find_map(|h| header_id(h));
    let responses = match rc.auto.as_ref() {
        Some(a) => catalog_serving_endpoint(a) == Some(route::Endpoint::Responses),
        None => {
            rc.dialect == Dialect::OpenAi
                && rc.forward_path.as_str().is_some_and(|p| {
                    let p = p.split_once('?').map_or(p, |(path, _)| path);
                    p.ends_with("/responses") || p.ends_with("/responses/compact")
                })
        }
    };
    // Always made on a managed response: `resp_model_scanner` keeps the response's id in it.
    let Some(taps) = rc.billing.tap() else {
        return;
    };
    taps.generation_id = generation_id;
    taps.request_id = request_id;
    taps.tools = responses.then(|| usage::ToolTally::new(rc.streaming));
}

/// Fold what the taps saw into the row's usage: the header generation id over the body's (the
/// response scanner's, from the head, over the usage block's), the serving host the head named,
/// and the Responses tool items.
pub(super) fn merge_taps(rc: &mut RequestCtx, usage: &mut usage::Usage) {
    let Some(t) = rc.billing.taps_mut() else {
        return;
    };
    let head_id = t
        .resp_kept
        .as_ref()
        .and_then(|k| k.get(0))
        .and_then(usage::id_from);
    let head_host = t
        .resp_kept
        .as_ref()
        .and_then(|k| k.get(1))
        .and_then(usage::host_from);
    let (hdr_id, tools, preview) = (t.generation_id, t.tools.take(), t.web_search_preview);
    if let Some(id) = hdr_id.or(head_id) {
        usage.upstream.generation_id = Some(id);
    }
    if head_host.is_some() {
        usage.upstream.served_by = head_host;
    }
    if let Some(mut t) = tools {
        t.finish();
        let mut counted = t.tools;
        if preview {
            counted.web_search_preview = counted.web_search;
            counted.web_search = 0;
        }
        usage.server_tools.merge_max(&counted);
        if usage.upstream.container_id.is_none() {
            usage.upstream.container_id = t.container_id;
        }
        usage.server_tool_calls = usage
            .server_tool_calls
            .max(u64::from(usage.server_tools.web_search));
    }
}

/// Which of a row's counts an estimate replaced (see `logging`), and what the estimate cannot see.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct EstimatedParts {
    pub(super) input: bool,
    pub(super) output: bool,
}

impl EstimatedParts {
    /// The row's `usage_estimated_parts`: `input`, `output`, `input,output`, or absent.
    pub(super) fn parts(self) -> Option<&'static str> {
        match (self.input, self.output) {
            (true, true) => Some("input,output"),
            (true, false) => Some("input"),
            (false, true) => Some("output"),
            (false, false) => None,
        }
    }

    /// The row's `usage_estimate_excludes`: `cache` when the input is estimated (the pre-token
    /// count is the whole prompt as plain input: some of it may have been cache reads, cheaper, or
    /// cache writes, dearer), `reasoning` when the output is estimated and the provider reported no
    /// reasoning count (hidden reasoning is billed as output and invisible in the stream).
    pub(super) fn excludes(self, u: &usage::Usage) -> Option<&'static str> {
        let reasoning = self.output && u.reasoning_tokens.is_none();
        match (self.input, reasoning) {
            (true, true) => Some("cache,reasoning"),
            (true, false) => Some("cache"),
            (false, true) => Some("reasoning"),
            (false, false) => None,
        }
    }
}

/// Who a request's error says ended it, as its billing row reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Fault {
    /// The client went away: any downstream error but the gateway's own answer.
    Client,
    /// The gateway's own answer to the client ([`gateway_refusal`]). Downstream-tagged, so a
    /// cancel once a response head arrived, and no row at all before one.
    Refusal,
    /// The upstream, or anything else not tagged downstream.
    Upstream,
}

impl Fault {
    pub(super) fn of(e: &pingora_core::Error) -> Self {
        if gateway_refusal(e) {
            Self::Refusal
        } else if e.esource() == &pingora_core::ErrorSource::Downstream {
            Self::Client
        } else {
            Self::Upstream
        }
    }
}

/// Whose word a billing row's token counts are.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Count {
    /// The provider's own report (or none at all, on an error body).
    #[default]
    Reported,
    /// The gateway's estimate of what the provider bills, and which counts it replaced (none
    /// when the provider's partial report already covered what the estimate saw).
    Estimated(EstimatedParts),
}

impl Count {
    pub(super) fn estimated(self) -> bool {
        matches!(self, Self::Estimated(_))
    }

    /// What the estimate replaced: nothing, for a reported count.
    pub(super) fn parts(self) -> EstimatedParts {
        match self {
            Self::Reported => EstimatedParts::default(),
            Self::Estimated(p) => p,
        }
    }
}

/// The facts `logging` reads about how a request ended, before it decides how the request bills.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct EndFacts {
    /// An abandoned `FullBody` attempt ([`Attempt::relay_abandoned`]).
    pub(super) relay_abandoned: bool,
    /// A free sub-resource (a token count).
    pub(super) free: bool,
    /// Answered from the exact-match cache.
    pub(super) cache_hit: bool,
    /// The provider's status; absent when no response head arrived.
    pub(super) status: Option<u16>,
    /// Whom the request's error blames; absent when it ended without one.
    pub(super) fault: Option<Fault>,
    /// The client hung up on a drained stream ([`crate::drain`]) before its terminal event.
    pub(super) client_left: bool,
    /// A provider was handed the request (`UpstreamPhase` past `None`).
    pub(super) called: bool,
    /// The drain read the stream to its end after the client left.
    pub(super) drained: bool,
    /// Whose word the counts are.
    pub(super) count: Count,
}

/// Every way a request can end, as billing sees it. `logging` builds one from its [`EndFacts`],
/// and [`Ending::row`] is a total match on it, so a new ending does not compile until someone
/// decides how it bills.
///
/// Every ending that writes a row carries its [`Count`], and the row logs it as it is, so an
/// estimate is never written as the provider's word. An estimate needs a 2xx head, or a connected
/// attempt that delivered the whole body ([`gave_up_waiting`]), so a cache hit, an error status
/// and a request no provider was handed carry `Reported`: [`Ending::of`] asserts it in debug
/// builds, where a producer that broke it would show. Only a cancel can be drained: a drain arms
/// only on a managed 2xx stream's head ([`drains_on_cancel`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Ending {
    /// An abandoned `FullBody` attempt: not the answer the client got. The attempt that serves
    /// writes the one row.
    RelayAbandoned,
    /// A free sub-resource (a token count): not a billable call.
    Free,
    /// Answered from the exact-match cache: the usage stored at fill, no provider called.
    CacheHit(Count),
    /// The gateway's own refusal before any response head (D267): no provider had the request
    /// whole, so it writes no row, as every other rejection writes none.
    Refused,
    /// The provider answered with an error status (4xx/5xx).
    UpstreamStatus(Count),
    /// The client left before the answer was whole, or left a drained stream however the request
    /// then ended. `drained`: the gateway then read the stream to its end.
    ClientCancelled { count: Count, drained: bool },
    /// No provider was called (every candidate's breaker open).
    NoCandidate(Count),
    /// The upstream failed before any response head: a connect, read-timeout or reset failure.
    UpstreamFailed(Count),
    /// A response that started and then died upstream.
    CutShort(Count),
    /// The request completed.
    Completed(Count),
}

/// The fields of an `ai.usage` row its [`Ending`] decides, as logged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct EndRow {
    /// What became of the request: a consumer must be able to tell a zero-token row for an
    /// upstream error or a cancel from a real zero-token generation.
    pub(super) outcome: &'static str,
    /// Whose word the counts are (`usage_estimated`, `usage_estimated_parts`).
    pub(super) count: Count,
    /// The counts are the cache entry's, stored at fill: no call was made.
    pub(super) cache_hit: bool,
    /// The client cancelled and the gateway read the stream to its end ([`crate::drain`]).
    pub(super) drained: bool,
}

impl EndRow {
    /// The row's `usage_settled`: `drained` when the counts after a cancel are the vendor's own.
    pub(super) fn usage_settled(self) -> Option<&'static str> {
        (self.drained && !self.count.estimated()).then_some("drained")
    }
}

impl Ending {
    /// The ending these facts describe. The order is the precedence: a row-less ending over any
    /// other, an error status over a cancel, a cancel over a missing call.
    pub(super) fn of(f: &EndFacts) -> Self {
        let count = f.count;
        if f.relay_abandoned {
            return Self::RelayAbandoned;
        }
        if f.free {
            return Self::Free;
        }
        // These three never see an estimate (see `Ending`): one here is a producer's bug.
        let reported = |ending: Self| {
            debug_assert!(!count.estimated(), "{ending:?} carries an estimate");
            ending
        };
        if f.cache_hit {
            return reported(Self::CacheHit(count));
        }
        if f.status.is_none() && f.fault == Some(Fault::Refusal) {
            return Self::Refused;
        }
        if f.status.is_some_and(|s| s >= 400) {
            return reported(Self::UpstreamStatus(count));
        }
        if f.client_left || matches!(f.fault, Some(Fault::Client | Fault::Refusal)) {
            return Self::ClientCancelled {
                count,
                drained: f.drained,
            };
        }
        if !f.called {
            return reported(Self::NoCandidate(count));
        }
        match (f.fault, f.status) {
            (Some(_), None) => Self::UpstreamFailed(count),
            (Some(_), Some(_)) => Self::CutShort(count),
            (None, _) => Self::Completed(count),
        }
    }

    /// How this ending bills; `None` writes no row.
    pub(super) fn row(self) -> Option<EndRow> {
        let row = |outcome, count| EndRow {
            outcome,
            count,
            cache_hit: false,
            drained: false,
        };
        Some(match self {
            Self::RelayAbandoned | Self::Free | Self::Refused => return None,
            Self::CacheHit(count) => EndRow {
                cache_hit: true,
                ..row("ok", count)
            },
            Self::UpstreamStatus(count) => row("upstream_error", count),
            Self::ClientCancelled { count, drained } => EndRow {
                drained,
                ..row("client_cancelled", count)
            },
            Self::NoCandidate(count) => row("no_candidate", count),
            Self::UpstreamFailed(count) => row("upstream_error", count),
            Self::CutShort(count) => row("cut_short", count),
            Self::Completed(count) => row("ok", count),
        })
    }
}

/// Upper bound on a model id we'll record. Real ids are short (`claude-opus-4-8`,
/// `accounts/fireworks/models/…`); anything longer is junk or an attempt to bloat the billing log.
pub(super) const MAX_MODEL_LEN: usize = 128;

/// Sanitize the model id extracted from the (client-controlled) request body before it lands in the
/// `ai.usage` billing log. `tracing`'s JSON layer escapes the value, but a downstream consumer
/// (logfwd/OTLP → ClickHouse) may re-handle it, so we refuse anything that could break out of a JSON
/// string or a line-oriented log: control bytes, `"`, `\`, `DEL`. A violating or over-long value is
/// recorded as `"unknown"` (matching `peek`'s non-UTF-8 fallback) rather than the raw bytes — a
/// mislabeled-but-safe usage row beats a corrupted or injected one.
pub(super) fn sanitize_model(model: String) -> Cow<'static, str> {
    let bad = model.len() > MAX_MODEL_LEN
        || model
            .bytes()
            .any(|b| b < 0x20 || b == b'"' || b == b'\\' || b == 0x7f);
    if bad {
        Cow::Borrowed("unknown")
    } else {
        Cow::Owned(model)
    }
}

/// The catalog row a billing row prices against: the provider-echoed id first, then the requested
/// one, each tried as spelled and then with a dated snapshot suffix removed. `None` when neither
/// names a row — an unpriced model, which a consumer must not silently price as something else.
///
/// The catalog lists aliases (`gpt-5`) and vendor slugs (`openai/gpt-5`), never the snapshots a
/// provider echoes (`gpt-5-2025-08-07`, `claude-sonnet-4-5-20250929`), so without the strip a
/// provider-routed row's `model` matches nothing. Runs once per billing row, on strings that are
/// already in hand.
pub(super) fn price_model(billed: &str, requested: &str) -> Option<&'static str> {
    [billed, requested]
        .into_iter()
        .find_map(|m| price_row(m).map(|r| r.model))
}

/// The catalog row one model id prices at: as spelled (a catalog name, or any candidate's
/// provider-specific spelling of it), then with a dated snapshot suffix removed. The one lookup
/// both [`price_model`] and the managed `/{provider}` refusal ([`BodyPlan::catalog_check`])
/// make on the requested model, so a managed `/{provider}` request is refused exactly when its
/// requested model would leave its billing row without a `price_model`.
pub(super) fn price_row(m: &str) -> Option<&'static route::ModelRoute> {
    route::model_route(m).or_else(|| strip_snapshot_date(m).and_then(route::model_route))
}

/// The input-token estimate for a billing row that needs one (`usage::InputTally`): the tally fed
/// as the body streamed when it was ([`RequestCtx::tally_eager`]), else a tally of the body itself,
/// made now: the `FullBody` parent's copy, or pingora's retry buffer (the whole body, within
/// 64 KiB). Most rows carry the provider's own count and never get here, so the body is not walked
/// for them at all.
pub(super) fn input_estimate(session: &Session, rc: &RequestCtx) -> u64 {
    if rc.tally_eager {
        return rc.attempt.input_tally.estimate_tokens();
    }
    let body = full_body_ctx(session)
        .map(|fb| fb.body)
        .or_else(|| session.as_ref().get_retry_buffer());
    let mut tally = usage::InputTally::default();
    if let Some(body) = body {
        tally.feed(&body);
    }
    tally.estimate_tokens()
}
