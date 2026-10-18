//! What the serving candidate of a catalog walk changes about the request.

use super::*;

/// The lowest set bit in `usable` at or after index `from`, or `None` if there is none.
///
/// The candidate walk's only cursor primitive. `from` strictly increases across a request, so the
/// walk always terminates — there is no way to revisit a candidate and claim a second breaker permit
/// against it.
pub(super) fn first_usable(usable: u8, from: u8) -> Option<u8> {
    if from >= route::MAX_CANDIDATES as u8 {
        return None;
    }
    // Mask off everything below `from`, then take the lowest remaining bit.
    let remaining = usable & !((1u8 << from) - 1);
    (remaining != 0).then(|| remaining.trailing_zeros() as u8)
}

/// Overwrite dialect and the body plan from **this** catalog candidate's path. Injection follows
/// the upstream candidate, not the client: a Messages body must not grow `stream_options`, and a
/// Chat Completions failover of a Messages client must.
pub(super) fn apply_serving_candidate(rc: &mut RequestCtx) {
    let Some(c) = rc.auto.as_ref().and_then(|a| a.candidate_at(a.candidate)) else {
        return;
    };
    rc.dialect = route::Endpoint::of_upstream_path(c.path).wire();
    rc.plan = BodyPlan::new(
        stream_options_eligible(rc.managed(), rc.dialect, c.path),
        false,
        true,
        rc.signed.is_some(),
    );
}

/// Whether a body sent on `wire` to `forward_path` (query allowed) must ask for its usage chunk:
/// a managed OpenAI Chat Completions request, which OpenAI meters only when it streams with
/// `stream_options.include_usage`. The one copy of the rule, for a `/{provider}` route, a catalog
/// walk's first arm, and each candidate it then serves from. BYO and every other endpoint stream
/// straight through. The query is stripped here, so no caller can hand `is_streamable_path` one
/// (Azure's `?api-version=` would silently disable injection).
pub(super) fn stream_options_eligible(managed: bool, wire: Dialect, forward_path: &str) -> bool {
    managed
        && wire == Dialect::OpenAi
        && is_streamable_path(
            forward_path
                .split_once('?')
                .map_or(forward_path, |(p, _)| p),
        )
}

pub(super) fn catalog_serving_endpoint(auto: &ModelRouting) -> Option<route::Endpoint> {
    auto.candidate_at(auto.candidate)
        .map(|c| route::Endpoint::of_upstream_path(c.path))
}

/// Whether this attempt, translated onto candidate `c`, asks it for Anthropic fast mode: the client
/// asked for priority processing and `c` serves fast mode (D266). One decision for the attempt's
/// `anthropic-beta` header and its translated body, so the two never disagree.
pub(super) fn translated_fast(auto: &ModelRouting, c: &route::Candidate) -> bool {
    auto.priority && providers::catalog::serves_fast_mode(c)
}

pub(super) fn catalog_translating(auto: &ModelRouting) -> bool {
    let Some(t) = auto.translate.as_ref() else {
        return false;
    };
    catalog_serving_endpoint(auto).is_some_and(|up| t.client != up)
}

/// This attempt asked a stream-only candidate for a stream its client did not ask for, and
/// assembles it into one body (D147; see `translate::SseBridge::assembling`).
pub(super) fn catalog_assembling(auto: &ModelRouting) -> bool {
    auto.translate.as_ref().is_some_and(|t| t.assemble)
}

/// A non-stream error from a same-endpoint candidate, buffered (it is small, and capped by
/// `MAX_TRANSLATE_BUFFER`) so a vendor's own error shape reaches the client in its API's envelope
/// (`translate::response_json_tools`, D100). One already in that envelope is relayed unchanged.
pub(super) fn catalog_error_relay(
    auto: &ModelRouting,
    status: Option<u16>,
    streaming: bool,
) -> bool {
    !streaming
        && status.is_some_and(|s| s >= 400)
        && auto.translate.is_some()
        && !catalog_translating(auto)
}

/// A catalog walk (not a sub-resource) served by a Chat Completions candidate of a vendor other
/// than OpenAI, whose stream may repeat what OpenAI's sends once (see `translate::ChatIdentity`).
/// It decides only for a Chat Completions client: any other is translated off that candidate
/// anyway (`catalog_translating`), which builds the same bridge.
pub(super) fn catalog_chat_relay(auto: &ModelRouting) -> bool {
    auto.translate.is_some()
        && auto.candidate_at(auto.candidate).is_some_and(|c| {
            c.provider != providers::ProviderId::OpenAi
                && route::Endpoint::of_upstream_path(c.path) == route::Endpoint::ChatCompletions
        })
}

/// OpenRouter's context compression is on by default for every endpoint of 8K context or less: a
/// prompt over the window has its middle dropped and is answered, where every other host (and
/// OpenRouter with compression off) answers the context error. The gateway relays what the client
/// sent or a context error, never an answer to a prompt it did not send, so a catalog walk to an
/// OpenRouter Chat Completions candidate turns it off, the way OpenRouter documents
/// (`plugins: [{"id": "context-compression", "enabled": false}]`).
pub(super) const OPENROUTER_NO_COMPRESSION: &[u8] =
    br#""plugins":[{"id":"context-compression","enabled":false}],"#;

/// An OpenRouter Chat Completions catalog candidate.
pub(super) fn openrouter_chat(c: &route::Candidate) -> bool {
    c.provider == providers::ProviderId::OpenRouter
        && route::Endpoint::of_upstream_path(c.path) == route::Endpoint::ChatCompletions
}

/// Splice [`OPENROUTER_NO_COMPRESSION`] just inside the root object. A body that already names
/// `plugins` is the client's choice and is left as sent, as is anything that is not a non-empty
/// JSON object (the provider rejects it by name).
pub(super) fn disable_openrouter_compression(mut body: Vec<u8>) -> Vec<u8> {
    if memchr::memmem::find(&body, br#""plugins""#).is_some() {
        return body;
    }
    let Some(open) = body.iter().position(|b| !b.is_ascii_whitespace()) else {
        return body;
    };
    let next = body
        .get(open + 1..)
        .and_then(|rest| rest.iter().find(|b| !b.is_ascii_whitespace()));
    if body.get(open) != Some(&b'{') || matches!(next, None | Some(b'}')) {
        return body;
    }
    body.splice(
        open + 1..open + 1,
        OPENROUTER_NO_COMPRESSION.iter().copied(),
    );
    body
}

/// Whether a catalog candidate is native OpenAI Chat Completions, which takes the output limit as
/// `max_completion_tokens` on every model and rejects `max_tokens` on its reasoning models.
pub(super) fn native_openai_chat(c: &route::Candidate) -> bool {
    c.provider == providers::ProviderId::OpenAi
        && route::Endpoint::of_upstream_path(c.path) == route::Endpoint::ChatCompletions
}
