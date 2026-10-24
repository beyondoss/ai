use super::*;

use crate::metrics::ProviderMetrics;
use crate::route::AuthScheme;

/// `max_tokens` is respelled for native OpenAI Chat; with both spellings the explicit
/// `max_completion_tokens` stays and the `max_tokens` member goes, whatever its value and
/// position, leaving valid JSON.
#[test]
fn max_tokens_is_respelled_max_completion_tokens() {
    let rename = |body: &str| {
        let mut buf = body.as_bytes().to_vec();
        let scan = peek::scan_buffered(&buf);
        let changed = rename_max_tokens(&mut buf, &scan.limit_keys);
        let out = String::from_utf8(buf).unwrap();
        serde_json::from_str::<serde_json::Value>(&out).expect(&out);
        (out, changed)
    };
    for (body, want, changed) in [
        (
            r#"{"model":"o3","max_tokens":256}"#,
            r#"{"model":"o3","max_completion_tokens":256}"#,
            true,
        ),
        (
            r#"{"max_tokens" : null, "model":"o3"}"#,
            r#"{"max_completion_tokens" : null, "model":"o3"}"#,
            true,
        ),
        (
            r#"{"max_tokens":256, "max_completion_tokens":512}"#,
            r#"{ "max_completion_tokens":512}"#,
            true,
        ),
        (
            r#"{"max_completion_tokens":512, "max_tokens" : {"x":[1,"}"]} }"#,
            r#"{"max_completion_tokens":512 }"#,
            true,
        ),
        (
            r#"{"max_completion_tokens":512,"messages":[{"max_tokens":1}]}"#,
            r#"{"max_completion_tokens":512,"messages":[{"max_tokens":1}]}"#,
            false,
        ),
    ] {
        assert_eq!(rename(body), (want.to_owned(), changed), "{body}");
    }
}

/// Every limit over the cap is cut to it, back to front so offsets hold; one under it, or a
/// cap of zero, is left alone. A value past `u64` is over any cap.
#[test]
fn output_limits_are_capped_never_raised() {
    let cap = |body: &str, max: u32| {
        let mut buf = body.as_bytes().to_vec();
        let scan = peek::scan_buffered(&buf);
        let changed = clamp_output_limits(&mut buf, &scan.limit_spans, max);
        (String::from_utf8(buf).unwrap(), changed)
    };
    assert_eq!(
        cap(
            r#"{"max_tokens":64000,"max_completion_tokens":99999999999999999999999}"#,
            16384
        ),
        (
            r#"{"max_tokens":16384,"max_completion_tokens":16384}"#.to_owned(),
            true
        )
    );
    assert_eq!(
        cap(r#"{"max_output_tokens":100}"#, 16384),
        (r#"{"max_output_tokens":100}"#.to_owned(), false)
    );
    assert_eq!(
        cap(r#"{"max_tokens":64000}"#, 0),
        (r#"{"max_tokens":64000}"#.to_owned(), false)
    );
}

/// A minimal `RequestCtx` for exercising the body-phase logic without a running proxy.
pub(super) fn test_ctx(stream_options: bool) -> RequestCtx {
    let provider = Provider::resolve(
        "openai",
        "api.openai.com:443".to_string(),
        Dialect::OpenAi,
        AuthScheme::Bearer,
        &["sk-pool"],
        ProviderMetrics::disconnected(),
        None,
    );
    RequestCtx {
        billing: Billing::of(Some(Identity::unverified_for_test(42, 7, None))),
        dialect: Dialect::OpenAi,
        leg: Leg::new(Arc::new(provider), None, 0),
        forward_path: ForwardPath::Inbound,
        model: String::new(),
        attempt: Attempt::new(Vec::new()),
        resp_model_scanner: peek::ModelScanner::for_response(),
        streaming: false,
        resp_tail: UsageTail::default(),
        resp_head: Vec::new(),
        upstream_status: None,
        plan: BodyPlan::new(stream_options, false, false, false),
        start: Instant::now(),
        deadline: crate::deadline::NONE,
        auto: None,
        control: None,
        request_id: RequestId::new(),
        tally_eager: false,
        resp_bytes: 0,
        upstream_phase: UpstreamPhase::None,
        redact: None,
        terminal: TerminalTracker::default(),
        signed: None,
    }
}

/// The Anthropic stream head is bounded at `USAGE_HEAD_CAP` however the stream is chunked: the
/// chunk that crosses the cap is cut there, and later chunks add nothing. Other responses keep
/// no head.
#[test]
fn the_usage_head_holds_at_most_its_cap() {
    let mut rc = test_ctx(false);
    rc.streaming = true;
    rc.keep_usage_head(&[b'a'; 1000]);
    assert!(rc.resp_head.is_empty(), "an OpenAI stream keeps no head");
    rc.dialect = Dialect::Anthropic;
    rc.keep_usage_head(&[b'a'; 5 * 1024]);
    assert_eq!(rc.resp_head.len(), 5 * 1024);
    rc.keep_usage_head(&[b'b'; 10 * 1024]);
    assert_eq!(rc.resp_head.len(), USAGE_HEAD_CAP);
    assert_eq!(rc.resp_head.last(), Some(&b'b'));
    rc.keep_usage_head(b"more");
    assert_eq!(rc.resp_head.len(), USAGE_HEAD_CAP);
}

/// The replay contract, exercised directly: `upstream_peer` starts a fresh attempt before pingora
/// re-feeds the buffered prefix through `request_body_filter`, so attempt 2 sees the body once.
///
/// This is the bug in miniature. Attempt 1 buffers a prefix and is cut off before end-of-stream,
/// so `req_buf` is never drained. Pingora then replays that prefix from byte 0. Without the
/// reset the two concatenate, the splice is planned against the first copy, and the upstream
/// gets a body with a duplicated fragment — which it rejects with a `400` that `logging` records
/// as a breaker *success*, so nothing surfaces it as ours.
///
/// Covered here rather than end-to-end because reproducing it over a socket needs the upstream
/// to die in the window where the gateway holds a *partial* body — a timing-dependent race. The
/// e2e suite proves the surrounding mechanism (pingora really does retry a reused connection and
/// really does replay through this filter); this pins what the reset itself must guarantee.
#[test]
fn resetting_the_body_phase_makes_a_replayed_prefix_idempotent() {
    let body = br#"{"messages":[{"role":"user","content":"hi"}],"model":"gpt-4o","stream":true}"#;
    let (prefix, rest) = body.split_at(30);

    // Attempt 1: a prefix arrives, then the upstream dies before end-of-stream. The chunk goes
    // through the production append step, as `request_body_filter` feeds it.
    let mut rc = test_ctx(true);
    assert!(rc.take_request_chunk(prefix));
    assert!(
        !rc.attempt.req_buf.is_empty(),
        "precondition: the partial body is still held, since end_of_stream never came",
    );

    // `upstream_peer` runs before any body byte of attempt 2.
    rc.begin_attempt();
    assert_eq!(
        rc.attempt.upload.0, 0,
        "the size guard must not double-count the replayed prefix",
    );

    // Attempt 2: pingora replays from byte 0 through the same filter, then the remainder.
    assert!(rc.take_request_chunk(prefix));
    assert!(rc.take_request_chunk(rest));

    assert_eq!(
        rc.attempt.req_buf, body,
        "the retried attempt must send the body exactly once",
    );
    assert_eq!(
        rc.attempt.upload.0,
        body.len(),
        "the replay counts the body once, as the client sent it",
    );
    let scan = peek::scan_buffered(&rc.attempt.req_buf);
    assert_eq!(
        scan.model.as_deref(),
        Some("gpt-4o"),
        "and the model must still be extractable — a duplicated prefix offsets the scanner's \
             depth permanently, shipping `requested_model` empty",
    );
    assert!(
        scan.inject_at.is_some(),
        "the stream_options splice must still be planned against a well-formed root object",
    );
}

/// A retried attempt replays the body prefix through the capture tap, so the reset must clear
/// the capture as well, or the captured request holds the prefix twice.
/// claim: O2
/// defect: D285
#[test]
fn a_retried_attempt_captures_the_request_body_once() {
    let body = br#"{"messages":[{"role":"user","content":"hi"}],"model":"gpt-4o","stream":true}"#;
    let (prefix, rest) = body.split_at(30);

    let mut rc = test_ctx(true);
    rc.control = Some(Box::new(RequestControl {
        metadata: None,
        capture: Some(CaptureBufs::new(4096)),
    }));

    // Attempt 1: a prefix arrives through the production append step (captured and buffered),
    // then the upstream dies.
    assert!(rc.take_request_chunk(prefix));
    rc.begin_attempt();

    // Attempt 2: pingora replays from byte 0 through the same step, then the remainder.
    assert!(rc.take_request_chunk(prefix));
    assert!(rc.take_request_chunk(rest));
    let captured = rc.control.as_ref().unwrap().capture.as_ref().unwrap();
    assert_eq!(
        captured.req_str(),
        std::str::from_utf8(body).ok(),
        "the capture holds the replayed prefix twice"
    );
}

/// A 2xx whose body turns out to be an error is counted against the breaker permit the head
/// resolved, so a late verdict cannot reopen a breaker that another request's probe owns by then.
/// claim: REL-6
/// defect: D286
#[test]
fn a_late_body_verdict_cannot_reopen_a_breaker_another_probe_owns() {
    static NOW: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1_000);
    fn clock() -> u64 {
        NOW.load(std::sync::atomic::Ordering::Relaxed)
    }
    let breaker = crate::circuit_breaker::CircuitBreaker::with_clock(
        crate::circuit_breaker::CircuitBreakerConfig::consecutive(1)
            .reset_timeout(std::time::Duration::from_secs(30))
            .half_open_permits(1),
        clock,
    );
    // This request was admitted while the breaker was closed, and its head was a 200.
    let admitted = breaker.allow().expect("a closed breaker admits");
    let admitted = breaker.record_success_for(admitted);
    // Another request trips the breaker; after the reset timeout its probe owns the breaker.
    let trip = breaker.allow().expect("a closed breaker admits");
    breaker.record_failure_for(trip);
    NOW.store(1_030, std::sync::atomic::Ordering::Relaxed);
    let _probe = breaker.allow().expect("the reset timeout admits one probe");

    let mut rc = test_ctx(true);
    rc.leg.provider = Arc::new(Provider::resolve(
        "openai",
        "api.openai.com:443".to_string(),
        Dialect::OpenAi,
        AuthScheme::Bearer,
        &["sk-pool"],
        ProviderMetrics::disconnected(),
        Some(breaker),
    ));
    let row = route::model_route("gpt-4o").expect("a catalog row");
    rc.auto = Some(Box::new(ModelRouting {
        route: row,
        sub: None,
        candidate: 0,
        usable: 1,
        walk: control::Walk::identity(row.candidates.len()),
        arms: row.candidates,
        session_field: None,
        attempt_start: Instant::now(),
        cache: None,
        translate: None,
        health: Some(PendingHealth {
            prefix: Vec::new(),
            permit: Some(admitted),
        }),
        addrs: 0,
        attempted: true,
        open_retry_after: None,
        priority: false,
    }));
    let state: &'static Arc<crate::state::GatewayState> = Box::leak(Box::new(
        crate::state::GatewayState::new(
            crate::config::AiConfig::default().validate().unwrap(),
            crate::state::test_metrics(),
        )
        .unwrap(),
    ));

    AiProxy { state }.settle_health(&mut rc, br#"{"error":{"message":"overloaded"}}"#, true);

    assert!(
        matches!(
            rc.leg.provider.breaker.as_ref().unwrap().state(),
            crate::circuit_breaker::CircuitState::HalfOpen { .. }
        ),
        "a late body verdict reopened a breaker owned by another request's probe"
    );
}

/// Without the reset, the same sequence corrupts the body — so the test above is not vacuous.
#[test]
fn a_replayed_prefix_without_the_reset_corrupts_the_body() {
    let body = br#"{"messages":[{"role":"user","content":"hi"}],"model":"gpt-4o","stream":true}"#;
    let (prefix, rest) = body.split_at(30);

    let mut rc = test_ctx(true);
    rc.attempt.req_buf.extend_from_slice(prefix);
    rc.attempt.upload.feed(prefix.len());
    // ...no reset...
    rc.attempt.req_buf.extend_from_slice(prefix);
    rc.attempt.req_buf.extend_from_slice(rest);

    assert_ne!(
        rc.attempt.req_buf, body,
        "the duplicated prefix must be observable"
    );
    assert!(
        serde_json::from_slice::<serde_json::Value>(&rc.attempt.req_buf).is_err(),
        "and the result is not valid JSON — this is the 400 the provider returns",
    );
    assert_eq!(
        rc.attempt.upload.0,
        prefix.len(),
        "and the size guard has double-counted",
    );
}

/// `RequestCtx` is touched on every hook and, for a streaming response, once per response
/// chunk, so its size is a real cost rather than bookkeeping. Growing it by 64 bytes to hold the
/// model-routing fields inline produced a reproducible ~2.5% regression on the
/// `managed_sse_latency` bench — non-streaming was unaffected, which is the signature of a
/// per-chunk cost. Boxing that state (see `ModelRouting`) bought it back.
///
/// A ceiling rather than an equality: padding and field order are the compiler's business, and a
/// few bytes either way is not what this guards. What it guards is someone adding a `String` or
/// an `Instant` here without noticing that the cost is paid per chunk on every stream.
/// `key_id` (identity, every managed request) is why 384 became 416 — do not spend that slack
/// on route-specific state.
#[test]
fn a_2xx_body_that_is_an_error_object_is_told_apart_from_an_answer() {
    // Non-streaming.
    let json = |b: &str| body_reports_error(b.as_bytes(), false);
    assert_eq!(
        json(r#"{"error":{"message":"overloaded","code":502}}"#),
        Some(true)
    );
    assert_eq!(json(r#" { "type" : "error", "error": {}}"#), Some(true));
    assert_eq!(
        json(r#"{"id":"chatcmpl-1","object":"chat.completion"}"#),
        Some(false)
    );
    assert_eq!(json(r#"{"type":"message","id":"msg_1"}"#), Some(false));
    assert_eq!(json(r#"{"err"#), None, "the first key has not ended");
    assert_eq!(json(""), None);
    assert_eq!(json("not json"), Some(false));
    // Streaming: the first data event decides; comments, ids and blank lines are skipped.
    let sse = |b: &str| body_reports_error(b.as_bytes(), true);
    assert_eq!(sse("data: {\"error\":{\"message\":\"x\"}}\n\n"), Some(true));
    assert_eq!(
        sse("event: error\ndata: {\"type\":\"error\",\"error\":{}}\n\n"),
        Some(true)
    );
    assert_eq!(
        sse(": OPENROUTER PROCESSING\n\ndata: {\"error\":1}\n\n"),
        Some(true)
    );
    assert_eq!(sse("data: {\"id\":\"c\",\"choices\":[]}\n\n"), Some(false));
    assert_eq!(
        sse("event: message_start\ndata: {\"type\":\"message_start\"}\n\n"),
        Some(false)
    );
    assert_eq!(sse("data: [DONE]\n\n"), Some(false));
    assert_eq!(sse(": keepalive\n"), None, "only a comment so far");
    assert_eq!(sse("data: {\"err"), None);
}

/// Building a request's context touches no shared reference count: every worker would otherwise
/// bounce the gateway state's `Arc` cache line on every request, including every fast reject.
/// claim: S1
/// defect: D92
#[test]
fn new_ctx_pays_no_arc_clone() {
    let state = GatewayState::new(
        crate::config::AiConfig::default().validate().unwrap(),
        crate::state::test_metrics(),
    )
    .unwrap();
    let proxy = AiProxy::new(state.clone());
    let before = Arc::strong_count(&state);
    let ctx = proxy.new_ctx();
    assert_eq!(
        Arc::strong_count(&state),
        before,
        "new_ctx cloned the state Arc"
    );
    drop(ctx);
}

/// The header sweep removes every match, past its stack batch and past its name slot, and keeps
/// everything else.
#[test]
fn remove_headers_where_removes_every_match_in_stack_batches() {
    let mut req = pingora::http::RequestHeader::build(http::Method::POST, b"/v1/x", None).unwrap();
    let long = format!("x-{}", "l".repeat(80));
    for i in 0..40 {
        req.insert_header(format!("x-drop-{i}"), "v").unwrap();
    }
    req.insert_header(long.clone(), "v").unwrap();
    req.insert_header("content-type", "application/json")
        .unwrap();
    req.insert_header("accept", "*/*").unwrap();
    remove_headers_where(&mut req, |n| n.starts_with("x-"));
    let left: Vec<&str> = req.headers.keys().map(http::HeaderName::as_str).collect();
    assert_eq!(left.len(), 2, "{left:?}");
    assert!(left.contains(&"content-type") && left.contains(&"accept"));
}

/// Each pool key's echo searcher is built once at boot, for exactly the bare key a provider
/// would echo: `response_filter` and `Redact` borrow it rather than rebuild it per
/// response or per chunk.
#[test]
fn a_pool_key_carries_its_boot_built_searcher() {
    let p = Provider::resolve(
        "openai",
        "api.openai.com:443".to_owned(),
        Dialect::OpenAi,
        AuthScheme::Bearer,
        &["sk-pool-a", "sk-pool-b"],
        ProviderMetrics::disconnected(),
        None,
    );
    for (auth, key) in p.pool_auth.iter().zip(["sk-pool-a", "sk-pool-b"]) {
        assert_eq!(auth.key(), key);
        assert_eq!(auth.finder().needle(), key.as_bytes());
    }
}

#[test]
fn request_ctx_stays_small_enough_to_be_cheap_per_chunk() {
    let size = std::mem::size_of::<RequestCtx>();
    assert!(
        // 448: the verified identity is held inline in `Billing::Managed` beside one lazily
        // boxed set of billing taps (`None` on BYO and until a tap first fires), and the
        // per-attempt and per-leg state packs into `Attempt` and `Leg` with the small fields.
        size <= 448,
        "RequestCtx grew to {size} bytes (ceiling 448). It is touched once per response chunk \
             on a stream — if the new state is only needed on one route, box it the way \
             `ModelRouting` is rather than paying for it on every request.",
    );
}

/// A model-routed request takes its dialect from **the serving candidate's path**: the row's
/// wire is the client's default, and a mixed-wire row (Claude → OpenRouter Chat Completions)
/// would otherwise parse a Chat Completions reply with the Anthropic extractor, or the reverse,
/// and emit a zero-token billing row. The assertions below check the derivation from the path
/// only; the provider's own wire is not what they exercise.
#[test]
fn a_model_route_takes_its_dialect_from_the_serving_candidate_path() {
    let openrouter = providers::by_id(providers::ProviderId::OpenRouter);
    assert_eq!(openrouter.wire, Dialect::OpenAi, "premise");

    let row = route::model_route("claude-opus-4-8").expect("seed row");
    assert_eq!(row.wire, Dialect::Anthropic, "premise: client default");
    let primary = row.candidates[0];
    let fallback = *row.candidates.last().expect("OpenRouter arm");
    assert_eq!(primary.provider, providers::ProviderId::Anthropic);
    assert_eq!(fallback.provider, providers::ProviderId::OpenRouter);
    assert_eq!(
        route::Endpoint::of_upstream_path(primary.path),
        route::Endpoint::Messages,
    );
    assert_eq!(
        route::Endpoint::of_upstream_path(fallback.path),
        route::Endpoint::ChatCompletions,
        "Claude's OpenRouter arm is Chat Completions, mixed-wire with the row",
    );
    assert_eq!(
        route::Endpoint::of_upstream_path(fallback.path).wire(),
        Dialect::OpenAi,
        "billing for the serving candidate must use the OpenAI extractor",
    );
    // The old derivation (row.wire, or provider.wire) is wrong for this fallback:
    assert_eq!(row.wire, Dialect::Anthropic);
    assert_eq!(openrouter.wire, Dialect::OpenAi);

    // The production derivation: serving the request from each arm sets the dialect from that
    // arm's path. Both attempts start from the row's wire, as `request_filter` leaves them.
    let serving = |candidate: usize| {
        let mut rc = test_ctx(true);
        rc.auto = Some(Box::new(ModelRouting {
            route: row,
            sub: None,
            candidate: u8::try_from(candidate).unwrap(),
            usable: 0b11,
            walk: control::Walk::identity(row.candidates.len()),
            arms: row.candidates,
            session_field: None,
            attempt_start: Instant::now(),
            cache: None,
            translate: None,
            health: None,
            addrs: 0,
            attempted: false,
            open_retry_after: None,
            priority: false,
        }));
        rc.dialect = row.wire;
        apply_serving_candidate(&mut rc);
        rc.dialect
    };
    assert_eq!(serving(0), Dialect::Anthropic, "the Messages primary");
    assert_eq!(
        serving(row.candidates.len() - 1),
        Dialect::OpenAi,
        "the OpenRouter Chat arm of a Claude row"
    );
}

/// Only a 401 walks and cools a key; 401, 402 and 403 are all catalog refusals. A 403's body
/// cools the key only when it names the credential.
#[test]
fn only_a_401_or_a_403_naming_the_key_is_a_key_failure() {
    assert!(is_pool_key_failure(401));
    assert!(!is_pool_key_failure(403) && !is_pool_key_failure(402));
    assert!((401..=403).all(is_candidate_refusal));
    assert!(!is_candidate_refusal(429) && !is_candidate_refusal(400));
    assert!(body_names_the_key(
        br#"{"error":{"message":"Incorrect API key provided","code":"invalid_api_key"}}"#
    ));
    assert!(body_names_the_key(
        br#"{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}"#
    ));
    for body in [
        &br#"{"type":"error","error":{"type":"permission_error","message":"no access"}}"#[..],
        br#"{"error":{"message":"flagged by moderation","code":403}}"#,
        br#"{"error":{"code":"model_not_found"}}"#,
    ] {
        assert!(!body_names_the_key(body));
    }
}

/// A resumed key walk's candidate goes first; everyone else keeps their order behind it.
#[test]
fn walk_front_moves_one_candidate_and_keeps_the_rest_in_order() {
    let mut walk = control::Walk::identity(4);
    walk_front(&mut walk, 2);
    assert_eq!(&walk.indices[..4], &[2, 0, 1, 3]);
    walk_front(&mut walk, 2);
    assert_eq!(&walk.indices[..4], &[2, 0, 1, 3]);
    walk_front(&mut walk, 7);
    assert_eq!(
        &walk.indices[..4],
        &[2, 0, 1, 3],
        "not in the walk: unchanged"
    );
}

/// The candidate cursor. `from` strictly increases across a request, which is what guarantees
/// the walk terminates and that no candidate can be revisited to claim a second breaker permit.
#[test]
fn first_usable_walks_set_bits_in_order_and_terminates() {
    // 0b1011 ⇒ candidates 0, 1, 3.
    assert_eq!(first_usable(0b1011, 0), Some(0));
    assert_eq!(first_usable(0b1011, 1), Some(1));
    assert_eq!(first_usable(0b1011, 2), Some(3));
    assert_eq!(first_usable(0b1011, 3), Some(3));
    assert_eq!(first_usable(0b1011, 4), None);

    // Nothing usable at all — the "every candidate is unkeyed" case.
    assert_eq!(first_usable(0, 0), None);

    // Past the bitmask width, including values a `saturating_add` could park on.
    assert_eq!(first_usable(0xFF, route::MAX_CANDIDATES as u8), None);
    assert_eq!(first_usable(0xFF, u8::MAX), None);

    // Walking a full mask visits every index exactly once, in order, then stops.
    let mut seen = Vec::new();
    let mut at = 0u8;
    while let Some(i) = first_usable(0xFF, at) {
        seen.push(i);
        at = i.saturating_add(1);
    }
    assert_eq!(seen, (0..route::MAX_CANDIDATES as u8).collect::<Vec<_>>());
}

/// The rewrite must replace exactly the value and leave the rest of the body byte-identical,
/// whether the new id is longer, shorter, or the same. A body that stays valid JSON with a
/// subtly wrong model is the failure this guards, and nothing downstream would catch it.
#[test]
fn model_rewrite_replaces_exactly_the_value() {
    let body = br#"{"model":"gpt-4o-mini","stream":true}"#.to_vec();
    let span = peek::scan_buffered(&body).model_span.unwrap();

    // Longer (the real failover case: OpenRouter prefixes the vendor).
    let grown = apply_model_rewrite(body.clone(), span, b"openai/gpt-4o-mini");
    assert_eq!(
        std::str::from_utf8(&grown).unwrap(),
        r#"{"model":"openai/gpt-4o-mini","stream":true}"#,
    );
    // Shorter.
    let shrunk = apply_model_rewrite(body.clone(), span, b"x");
    assert_eq!(
        std::str::from_utf8(&shrunk).unwrap(),
        r#"{"model":"x","stream":true}"#,
    );
    // Identical ⇒ untouched, and no memmove: the primary candidate's common case.
    let same = apply_model_rewrite(body.clone(), span, b"gpt-4o-mini");
    assert_eq!(same, body);

    // Both rewrites still parse, and the result is still injectable at the same offset — the
    // ordering invariant that lets the model rewrite run before the stream_options splice.
    let rescanned = peek::scan_buffered(&grown);
    assert_eq!(rescanned.model.as_deref(), Some("openai/gpt-4o-mini"));
    assert_eq!(
        rescanned.inject_at,
        peek::scan_buffered(&body).inject_at,
        "the injection offset sits before the model value, so a rewrite cannot move it",
    );
}

/// An out-of-range span must be ignored rather than panicking a worker.
#[test]
fn model_rewrite_ignores_an_impossible_span() {
    let body = br#"{"model":"a"}"#.to_vec();
    assert_eq!(apply_model_rewrite(body.clone(), (5, 2), b"z"), body);
    assert_eq!(apply_model_rewrite(body.clone(), (0, 999), b"z"), body);
}

/// A client-side abort must not count against the provider's breaker; everything else must.
///
/// This is the whole of the cancellation bug: `ErrorSource::Downstream` is a user hitting ESC or
/// a broken pipe writing the response back, and counting those opened breakers on healthy
/// providers. `Unset` is our own DNS failure out of `upstream_peer` and `Upstream` is a real
/// connect/read failure — both are genuine provider failures and must still count.
#[test]
fn only_non_downstream_errors_count_against_the_breaker() {
    use pingora_core::{Error, ErrorSource, ErrorType};

    assert!(!is_upstream_failure(None), "no error is not a failure");

    let mut down = *Error::new(ErrorType::WriteError);
    down.esource = ErrorSource::Downstream;
    assert!(
        !is_upstream_failure(Some(&down)),
        "a client abort must not trip the provider's breaker",
    );

    for source in [
        ErrorSource::Upstream,
        ErrorSource::Internal,
        ErrorSource::Unset,
    ] {
        let mut e = *Error::new(ErrorType::ConnectError);
        e.esource = source.clone();
        assert!(
            is_upstream_failure(Some(&e)),
            "{source:?} must count as a provider failure",
        );
    }
}

/// The premise behind `RequestCtx::begin_attempt`: a scanner fed a duplicated body
/// prefix is permanently broken, because the extra `{` offsets its brace depth so the root-level
/// key check never matches again.
///
/// Pingora replays its retry buffer through `request_body_filter`, so without the reset this is
/// exactly what the second attempt's scanner sees — and the billing row ships
/// `requested_model = ""`. The e2e coverage for the real replay path lives in `tests/e2e.rs`.
#[test]
fn a_replayed_body_prefix_breaks_a_scanner_that_was_not_reset() {
    // `model` deliberately sits after `messages`, so the scanner is still mid-walk when the
    // replayed prefix arrives — the case a short-circuit on an early `model` would hide.
    let body = br#"{"messages":[{"role":"user","content":"hi"}],"model":"gpt-4o"}"#;
    let (prefix, rest) = body.split_at(30);

    let mut fresh = peek::ModelScanner::new();
    fresh.feed(body);
    assert_eq!(
        fresh.take_model().as_deref(),
        Some("gpt-4o"),
        "a clean walk finds the root-level model",
    );

    let mut replayed = peek::ModelScanner::new();
    replayed.feed(prefix);
    replayed.feed(prefix); // pingora replays from byte 0 on the retry
    replayed.feed(rest);
    assert_eq!(
        replayed.take_model(),
        None,
        "the duplicated prefix offsets the depth, so the root `model` is never seen — \
             this is why upstream_peer resets the scanner every attempt",
    );
}

/// Build a `RequestHeader` for a given path (+ optional query) with the given headers set —
/// mirrors the shape `session.req_header()` hands `extract_virtual_key` on the real path.
fn req_with_headers(
    path: &str,
    headers: &[(&'static str, &'static str)],
) -> pingora::http::RequestHeader {
    let mut req =
        pingora::http::RequestHeader::build(http::Method::POST, path.as_bytes(), None).unwrap();
    for (k, v) in headers {
        req.insert_header(*k, *v).unwrap();
    }
    req
}

#[test]
fn extract_virtual_key_recognizes_anthropic_x_api_key() {
    let req = req_with_headers("/v1/messages", &[("x-api-key", "sk-ant-key")]);
    assert_eq!(extract_virtual_key(&req).map(|p| p.key), Some("sk-ant-key"));
}

#[test]
fn extract_virtual_key_recognizes_openai_bearer() {
    let req = req_with_headers(
        "/v1/chat/completions",
        &[("authorization", "Bearer sk-openai-key")],
    );
    assert_eq!(
        extract_virtual_key(&req).map(|p| p.key),
        Some("sk-openai-key")
    );
}

#[test]
fn extract_virtual_key_recognizes_azure_api_key_header() {
    // Azure OpenAI authenticates via the bare `api-key` header (no `Bearer` prefix, no
    // OAuth). Before this fix, a client presenting only this header got a 401.
    let req = req_with_headers("/v1/responses", &[("api-key", "azure-secret")]);
    assert_eq!(
        extract_virtual_key(&req).map(|p| p.key),
        Some("azure-secret")
    );
}

#[test]
fn extract_virtual_key_recognizes_google_goog_api_key_header() {
    // Google Gemini authenticates via `x-goog-api-key`.
    let req = req_with_headers(
        "/v1beta/models/gemini-2.5-pro:generateContent",
        &[("x-goog-api-key", "goog-secret")],
    );
    assert_eq!(
        extract_virtual_key(&req).map(|p| p.key),
        Some("goog-secret")
    );
}

#[test]
fn extract_virtual_key_recognizes_google_key_query_param() {
    // Google Gemini also accepts the key as a `?key=` query param — no header at all.
    let req = req_with_headers(
        "/v1beta/models/gemini-2.5-pro:generateContent?key=goog-query-secret",
        &[],
    );
    assert_eq!(
        extract_virtual_key(&req).map(|p| p.key),
        Some("goog-query-secret")
    );
}

#[test]
fn extract_virtual_key_query_param_only_used_as_last_resort() {
    // A header takes precedence over a `key=` query param if both are somehow present.
    let req = req_with_headers(
        "/v1beta/models/gemini-2.5-pro:generateContent?key=query-secret",
        &[("x-goog-api-key", "header-secret")],
    );
    assert_eq!(
        extract_virtual_key(&req).map(|p| p.key),
        Some("header-secret")
    );
}

/// claim: SEC-11
/// defect: D29
#[test]
fn a_managed_key_in_any_location_makes_the_request_managed() {
    const VK: &str = "bai_v1.1.payload.sig";
    for headers in [
        &[
            ("x-api-key", "junk"),
            ("authorization", "Bearer bai_v1.1.payload.sig"),
        ][..],
        &[
            ("x-api-key", ""),
            ("authorization", "Bearer bai_v1.1.payload.sig"),
        ][..],
        &[("api-key", "sk-real"), ("x-goog-api-key", VK)][..],
        &[
            ("x-api-key", "sk-ant-real"),
            ("authorization", "bearer bai_v1.1.payload.sig"),
        ][..],
    ] {
        let req = req_with_headers("/openai/v1/chat/completions", headers);
        assert_eq!(
            extract_virtual_key(&req).map(|p| p.key),
            Some(VK),
            "{headers:?}"
        );
    }
    let req = req_with_headers(
        "/openai/v1/chat/completions?key=bai_v1.1.payload.sig",
        &[("authorization", "Bearer sk-byo")],
    );
    assert_eq!(extract_virtual_key(&req).map(|p| p.key), Some(VK));
    // No managed key anywhere: the first non-empty location, as before.
    let req = req_with_headers(
        "/v1/chat/completions",
        &[("x-api-key", ""), ("authorization", "Bearer sk-byo")],
    );
    assert_eq!(extract_virtual_key(&req).map(|p| p.key), Some("sk-byo"));
}

#[test]
fn extract_virtual_key_returns_none_when_absent() {
    let req = req_with_headers("/v1/chat/completions", &[]);
    assert_eq!(extract_virtual_key(&req).map(|p| p.key), None);
}

#[test]
fn managed_client_headers_keep_only_the_allowlist_and_safe_betas() {
    let mut req = req_with_headers(
        "/v1/messages",
        &[
            ("content-type", "application/json"),
            ("user-agent", "claude-cli/2.0"),
            ("anthropic-version", "2023-06-01"),
            ("openai-organization", "org-evil"),
            ("cookie", "session=1"),
            ("x-stainless-lang", "js"),
            (
                "anthropic-beta",
                "prompt-caching-2024-07-31, context-1m-2025-08-07,interleaved-thinking-2025-05-14",
            ),
        ],
    );
    retain_managed_client_headers(&mut req, true, true).unwrap();
    let mut names: Vec<&str> = req.headers.keys().map(|k| k.as_str()).collect();
    names.sort_unstable();
    assert_eq!(
        names,
        [
            "anthropic-beta",
            "anthropic-version",
            "content-type",
            "user-agent"
        ]
    );
    assert_eq!(
        req.headers.get("anthropic-beta").unwrap(),
        "prompt-caching-2024-07-31,interleaved-thinking-2025-05-14"
    );

    // A value made only of unlisted tokens leaves no header at all; two header lines are
    // filtered as one list rather than letting the second through unread.
    let mut req = req_with_headers(
        "/v1/messages",
        &[("anthropic-beta", "mcp-client-2025-04-04")],
    );
    retain_managed_client_headers(&mut req, true, true).unwrap();
    assert!(req.headers.get("anthropic-beta").is_none());
    let mut req = req_with_headers("/v1/messages", &[]);
    req.append_header("anthropic-beta", "prompt-caching-2024-07-31")
        .unwrap();
    req.append_header("anthropic-beta", "code-execution-2025-05-22")
        .unwrap();
    retain_managed_client_headers(&mut req, true, true).unwrap();
    assert_eq!(
        req.headers
            .get_all("anthropic-beta")
            .iter()
            .collect::<Vec<_>>(),
        ["prompt-caching-2024-07-31"]
    );
}

#[test]
fn strip_key_param_drops_only_the_credential() {
    assert_eq!(
        strip_key_param("/v1/x?key=bai_v1.a.b.c&api-version=2024").as_deref(),
        Some("/v1/x?api-version=2024")
    );
    assert_eq!(
        strip_key_param("/v1/x?a=1&key=s&b=2&key=t").as_deref(),
        Some("/v1/x?a=1&b=2")
    );
    assert_eq!(strip_key_param("/v1/x?key=s").as_deref(), Some("/v1/x"));
    assert_eq!(strip_key_param("/v1/x?keys=1&monkey=2"), None);
    assert_eq!(strip_key_param("/v1/x"), None);
}

#[test]
fn key_params_finds_every_key_among_multiple_params() {
    let all = |q| key_params(q).collect::<Vec<_>>();
    assert_eq!(all("a=1&key=abc123&b=2"), ["abc123"]);
    assert_eq!(all("key=solo"), ["solo"]);
    assert_eq!(
        all("key=junk&k%65y=two&%6B%65%79=three"),
        ["junk", "two", "three"]
    );
    assert!(all("a=1&b=2&keys=3&monkey=4").is_empty());
    assert!(all("").is_empty());
}

/// A full-body attempt that ended with the client gone is the end of the request, whatever
/// retry the attempt recorded: re-running it would send the whole body to another candidate (or
/// key) for a client that is no longer there, a generation billed upstream that nobody reads.
/// An upstream failure with a recorded retry still retries, and nothing written ends the walk.
/// claim: REL-9, REL-21
/// defect: D206
#[test]
fn a_full_body_attempt_that_lost_its_client_is_never_retried() {
    use pingora_core::{Error, ErrorType};
    let down = || Err(Error::new(ErrorType::ConnectionClosed).into_down());
    let up = || Err(Error::new(ErrorType::ConnectRefused).into_up());
    let retry = Some(RelayRetry::Candidate(0));
    assert_eq!(attempt_end(&down(), retry), AttemptEnd::Fail);
    assert_eq!(
        attempt_end(
            &down(),
            Some(RelayRetry::Key {
                candidate: 0,
                key: 1
            })
        ),
        AttemptEnd::Fail
    );
    assert_eq!(attempt_end(&down(), None), AttemptEnd::Fail);
    assert_eq!(
        attempt_end(&up(), retry),
        AttemptEnd::Retry(RelayRetry::Candidate(0))
    );
    assert_eq!(attempt_end(&up(), None), AttemptEnd::Fail);
    assert_eq!(
        attempt_end(&Ok(Piped::Abandoned), retry),
        AttemptEnd::Retry(RelayRetry::Candidate(0))
    );
    assert_eq!(attempt_end(&Ok(Piped::Written), retry), AttemptEnd::Done);
    assert_eq!(attempt_end(&Ok(Piped::Empty), None), AttemptEnd::Fail);
}

/// A full-body attempt's task that panicked is reported (the parent logs it with the request
/// id, then answers 502), not dropped unseen; one that finished or is still running is not.
/// claim: REL-17, REL-21
/// defect: D207
#[tokio::test]
async fn a_panicked_full_body_attempt_is_reported() {
    let panicked = tokio::spawn(async { panic!("attempt blew up") });
    assert_eq!(reap_attempt(panicked, "rid-1").await, Reaped::Panicked);
    let finished = tokio::spawn(async {});
    assert_eq!(reap_attempt(finished, "rid-2").await, Reaped::Finished);
    let stuck = tokio::spawn(std::future::pending::<()>());
    tokio::time::pause();
    assert_eq!(reap_attempt(stuck, "rid-3").await, Reaped::StillRunning);
}

/// An error-in-200 is read from the root's members in any order; an answer beside `"error":
/// null`, or with partial `choices`, is not one.
/// claim: BIL-12, BIL-20
#[test]
fn json_is_error_only_reads_members_in_any_order() {
    for body in [
        r#"{"error":{"code":502,"message":"x"}}"#,
        r#"{"id":"gen-1","object":"chat.completion","error":{"code":502},"user_id":"u"}"#,
        r#"{"model":"m","error":"upstream failed"}"#,
    ] {
        assert!(json_is_error_only(body.as_bytes()), "{body}");
    }
    for body in [
        r#"{"id":"resp_1","object":"response","error":null,"output":[]}"#,
        r#"{"error":{"code":502},"choices":[{"finish_reason":"error"}]}"#,
        r#"{"type":"message","content":[],"error":{"x":1}}"#,
        r#"{"id":"x","error":null}"#,
        r#"{"error":{"code":502}"#,
        "[]",
    ] {
        assert!(!json_is_error_only(body.as_bytes()), "{body}");
    }
}

/// `background: true` is read as a provider's parser reads it: at the root, any copy, its name
/// escaped or not; never a nested member or a string that says so.
/// claim: BIL-2
#[test]
fn requests_background_reads_the_root_member() {
    for body in [
        r#"{"model":"m","background":true}"#,
        r#"{ "background" : true , "input":"x"}"#,
        r#"{"background":true}"#,
        r#"{"background":false,"background":true}"#,
    ] {
        assert!(requests_background(body.as_bytes()), "{body}");
    }
    for body in [
        r#"{"model":"m","background":false}"#,
        r#"{"model":"m","background":null,"stream":true}"#,
        r#"{"model":"m","background":"true"}"#,
        r#"{"model":"m","metadata":{"background":true}}"#,
        r#"{"model":"m","input":"background\":true"}"#,
        "not json true",
    ] {
        assert!(!requests_background(body.as_bytes()), "{body}");
    }
}

/// The pool key is scrubbed wherever it falls in a streamed body — inside one chunk, split
/// across two, repeated — and the body keeps its length (so its `Content-Length`).
/// claim: SEC-7
#[test]
fn redact_scrubs_a_key_split_across_chunks_and_keeps_the_length() {
    let raw = b"sk-pool-secret-0123456789";
    let key = &memchr::memmem::Finder::new(raw);
    let body = format!(
        r#"{{"error":{{"message":"bad key {k} (again: {k})","note":"{k}"}}}}"#,
        k = std::str::from_utf8(raw).unwrap()
    );
    for split in [1, 7, 20, 30, 47, body.len() - 1] {
        for step in [1, 3, 64] {
            let mut r = Redact::default();
            let mut out = Vec::new();
            let (head, tail) = body.as_bytes().split_at(split);
            let mut chunks: Vec<&[u8]> = head.chunks(step).collect();
            chunks.push(tail);
            for c in chunks {
                let got = r.feed(
                    std::slice::from_ref(key),
                    Some(Bytes::copy_from_slice(c)),
                    false,
                );
                out.extend_from_slice(got.as_deref().unwrap_or(&[]));
            }
            out.extend_from_slice(&r.feed(std::slice::from_ref(key), None, true).unwrap());
            let text = String::from_utf8(out).unwrap();
            assert_eq!(text.len(), body.len(), "{split}/{step}");
            assert!(!text.contains("sk-pool-secret"), "{split}/{step}: {text}");
            assert_eq!(text.matches("[redacted]").count(), 3, "{text}");
        }
    }
    // A JSON error is held whole: nothing until its end, then the key masked and an account
    // remedy neutralized; one past the cap streams with the key masked.
    let remedy = format!(
        r#"{{"error":{{"message":"bad key {k}; add your own key","code":429}}}}"#,
        k = std::str::from_utf8(raw).unwrap()
    );
    let mut r = Redact {
        whole: true,
        status: 429,
        ..Redact::default()
    };
    for c in remedy.as_bytes().chunks(5) {
        let got = r.feed(
            std::slice::from_ref(key),
            Some(Bytes::copy_from_slice(c)),
            false,
        );
        assert!(got.unwrap().is_empty());
    }
    let out = r.feed(std::slice::from_ref(key), None, true).unwrap();
    assert_eq!(
        &out[..],
        format!(
            r#"{{"error":{{"code":429,"message":"{}"}}}}"#,
            remedy::RATE_LIMITED
        )
        .as_bytes()
    );
    let big = format!(
        r#"{{"error":{{"message":"{k} add your own key","pad":"{}"}}}}"#,
        "x".repeat(remedy::MAX_BODY),
        k = std::str::from_utf8(raw).unwrap()
    );
    let mut r = Redact {
        whole: true,
        status: 429,
        ..Redact::default()
    };
    let mut out = r
        .feed(
            std::slice::from_ref(key),
            Some(Bytes::copy_from_slice(big.as_bytes())),
            false,
        )
        .unwrap()
        .to_vec();
    out.extend_from_slice(&r.feed(std::slice::from_ref(key), None, true).unwrap());
    assert_eq!(out.len(), big.len());
    assert!(!out.windows(raw.len()).any(|w| w == raw));

    // A key shorter than the marker is masked with a truncated marker.
    let mut buf = *b"x=abc;";
    assert!(mask_all(&mut buf, &memchr::memmem::Finder::new(b"abc")));
    assert_eq!(&buf, b"x=[re;");
}

/// A base64 key (`/`, `+`) echoed in a JSON encoder's spelling (`\/`, `+`, `+`) is
/// scrubbed on every path: streamed in small chunks, held whole with no remedy, and held whole
/// and re-serialized by the remedy rewrite, which decodes any escape (`A`) back to the key.
/// claim: SEC-7
#[test]
fn redact_scrubs_every_json_spelling_of_a_base64_key() {
    let key = "ABSKa2V5/cGFy+dA==";
    let keys = route::key_finders(key);
    assert_eq!(keys.len(), 6, "2 slash spellings x 3 plus spellings");
    for echo in [
        key.to_owned(),
        r"ABSKa2V5\/cGFy+dA==".to_owned(),
        r"ABSKa2V5\/cGFy+dA==".to_owned(),
        r"ABSKa2V5/cGFy+dA==".to_owned(),
    ] {
        for (whole, status, remedy) in [
            (false, 401, ""),
            (true, 401, ""),
            (true, 401, " see https://openrouter.ai/settings/keys"),
        ] {
            let body = format!(
                r#"{{"error":{{"message":"bad key {echo}{remedy}","param":"{echo}","raw":"ABSKa2V5\/cGFy+dA=="}}}}"#
            );
            let mut r = Redact {
                whole,
                status,
                ..Redact::default()
            };
            let mut out = Vec::new();
            for c in body.as_bytes().chunks(3) {
                let got = r.feed(&keys, Some(Bytes::copy_from_slice(c)), false);
                out.extend_from_slice(got.as_deref().unwrap_or(&[]));
            }
            out.extend_from_slice(&r.feed(&keys, None, true).unwrap());
            let text = String::from_utf8(out).unwrap();
            let decoded = serde_json::from_str::<serde_json::Value>(&text)
                .unwrap()
                .to_string();
            let rewritten = !remedy.is_empty();
            assert!(
                !text.contains(key) && !text.contains(&echo),
                "{echo} whole={whole} remedy={rewritten}: {text}"
            );
            // A held JSON body is also clean as a JSON client reads it: the `raw` field's
            // `A` spelling decodes to the key. A streamed one is scrubbed of the
            // spellings an encoder writes.
            if whole {
                assert!(
                    !decoded.contains(key),
                    "{echo} whole remedy={rewritten}: {text}"
                );
            }
        }
    }
}

/// Every spelling of a credential location is read, and a virtual key in any of them makes the
/// request managed: a repeated header whose first line is junk, a second `?key=`, an encoded
/// `key` name or value, extra whitespace after `Bearer`, a scheme-less virtual key.
/// claim: SEC-3, SEC-11
#[test]
fn a_managed_key_in_any_spelling_makes_the_request_managed() {
    const VK: &str = "bai_v1.1.payload.sig";
    let build = |path: &str, headers: &[(&'static str, &str)]| {
        let mut req =
            pingora::http::RequestHeader::build(http::Method::POST, path.as_bytes(), None).unwrap();
        for (k, v) in headers {
            req.append_header(*k, *v).unwrap();
        }
        req
    };
    let cases: [(&str, &[(&'static str, &str)]); 7] = [
        ("/x?key=junk&key=bai_v1.1.payload.sig", &[]),
        (
            "/x?k%65y=bai_v1.1.payload.sig",
            &[("authorization", "Bearer sk-byo")],
        ),
        ("/x", &[("x-api-key", "junk"), ("x-api-key", VK)]),
        ("/x", &[("authorization", "Bearer  bai_v1.1.payload.sig")]),
        ("/x", &[("authorization", "Bearer	bai_v1.1.payload.sig ")]),
        (
            "/x",
            &[("authorization", "Bearer sk-byo"), ("authorization", VK)],
        ),
        ("/x", &[("api-key", "sk-real"), ("api-key", VK)]),
    ];
    for (path, headers) in cases {
        let req = build(path, headers);
        let p = extract_virtual_key(&req).unwrap();
        assert!(p.managed, "{path} {headers:?}");
        assert_eq!(p.key, VK, "{path} {headers:?}");
    }
    // Managed only once decoded: managed (so stripped, never forwarded), and the encoded
    // value fails verification.
    let req = build("/x?key=%62ai_v1.1.payload.sig", &[]);
    assert!(extract_virtual_key(&req).unwrap().managed);
    // A BYO key in a lenient spelling is still the BYO key.
    let req = build("/x", &[("authorization", "bearer   sk-byo")]);
    assert_eq!(
        extract_virtual_key(&req),
        Some(Presented {
            key: "sk-byo",
            managed: false
        })
    );
    assert_eq!(
        strip_key_param("/x?k%65y=a&b=1&%6b%65%79=c&key=d").as_deref(),
        Some("/x?b=1")
    );
}

#[test]
fn the_gateways_own_refusal_is_not_giving_up_on_a_delivered_request() {
    use pingora_core::{Error, ErrorType as T};
    // A body the gateway withheld and refused: the provider never had it, nothing to bill.
    assert!(!gave_up_waiting(
        &Refusal::CatalogMiss {
            name: None,
            header: false
        }
        .into_error()
        .into_down()
    ));
    assert!(!gave_up_waiting(
        &Error::new(T::HTTPStatus(400)).into_down()
    ));
    // The client going away, and the gateway's read timeout, still are.
    assert!(gave_up_waiting(
        &Error::new(T::ConnectionClosed).into_down()
    ));
    assert!(gave_up_waiting(&Error::new(T::ReadTimedout).into_up()));
}

#[test]
fn price_model_resolves_echoed_snapshots_and_vendor_slugs() {
    assert_eq!(price_model("gpt-5-2025-08-07", "gpt-5"), Some("gpt-5"));
    assert_eq!(price_model("gpt-5-2025-08-07", "x"), Some("gpt-5"));
    assert_eq!(
        price_model("claude-opus-4-8-20260101", "x"),
        Some("claude-opus-4-8")
    );
    assert_eq!(
        price_model("anthropic/claude-opus-4.8", "x"),
        Some("claude-opus-4-8")
    );
    // The echo names nothing priced, the request does.
    assert_eq!(price_model("unknown", "gpt-5"), Some("gpt-5"));
    assert_eq!(price_model("my-finetune-2025-08-07", "nope"), None);
    // The managed `/{provider}` refusal asks exactly this of the requested model: any
    // candidate's provider spelling counts, as does a dated snapshot; nothing else does.
    let row = |m| price_row(m).map(|r| r.model);
    assert_eq!(
        row("accounts/fireworks/models/gpt-oss-120b"),
        price_model("", "accounts/fireworks/models/gpt-oss-120b")
    );
    assert!(row("accounts/fireworks/models/gpt-oss-120b").is_some());
    assert_eq!(row("claude-sonnet-4-5-20250929"), Some("claude-sonnet-4-5"));
    assert_eq!(row("gpt-4o-mini-2024-07-18"), Some("gpt-4o-mini"));
    for unpriced in ["my-finetune", "ft:gpt-4o:acme", "claude-2.1", "", "unknown"] {
        assert_eq!(row(unpriced), None, "{unpriced}");
        assert_eq!(price_model("", unpriced), None, "{unpriced}");
    }
    // Not a date: left alone.
    assert_eq!(strip_snapshot_date("gpt-4o-mini"), None);
    assert_eq!(strip_snapshot_date("model-12-34"), None);
}

#[test]
fn sanitize_model_passes_real_ids() {
    for id in [
        "gpt-4o",
        "claude-opus-4-8",
        "openrouter/meta-llama/llama-3.1",
        "accounts/fireworks/models/llama-v3p1-70b-instruct",
        "gpt-4o-mini-2024-07-18",
    ] {
        assert_eq!(sanitize_model(id.to_string()), id);
    }
}

#[test]
fn sanitize_model_rejects_json_and_log_injection() {
    // A `"` would close the JSON string; `\` could escape; a newline breaks line-oriented log
    // shipping. Any of them ⇒ recorded as "unknown" rather than injected into the billing log.
    for evil in [
        r#"real","injected":"x"#,
        r#"a\b"#,
        "line1\nline2",
        "ctrl\u{0}byte",
    ] {
        assert_eq!(sanitize_model(evil.to_string()), "unknown");
    }
}

#[test]
fn sanitize_model_rejects_overlong() {
    let long = "a".repeat(MAX_MODEL_LEN + 1);
    assert_eq!(sanitize_model(long), "unknown");
    // Exactly at the cap is fine.
    let ok = "a".repeat(MAX_MODEL_LEN);
    assert_eq!(sanitize_model(ok.clone()), ok);
}

#[test]
fn a_provider_route_takes_its_wire_from_the_forwarded_path() {
    for (path, wire) in [
        ("/api/v1/messages", Dialect::Anthropic),
        ("/v1/messages?beta=true", Dialect::Anthropic),
        ("/v1/messages/count_tokens", Dialect::Anthropic),
        ("/v1/messages/", Dialect::Anthropic),
        ("/v1/chat/completions", Dialect::OpenAi),
        (
            "/openai/deployments/x/chat/completions?api-version=1",
            Dialect::OpenAi,
        ),
        ("/v1/responses/compact", Dialect::OpenAi),
        ("/v1/embeddings", Dialect::OpenAi),
    ] {
        assert_eq!(wire_of_forward_path(path), wire, "{path}");
    }
}

#[test]
fn dialect_for_path_selects_anthropic_only_for_messages() {
    // The dialect drives usage parsing *and* stream injection: misclassifying an Anthropic
    // `/v1/messages` request as OpenAI mis-meters its tokens. The rule is a `/v1/messages`
    // prefix ⇒ Anthropic; everything else (chat completions, embeddings, the bare root) is
    // OpenAI-dialect. This locks that mapping so a refactor can't silently flip it.
    assert_eq!(dialect_for_path("/v1/messages"), Dialect::Anthropic);
    assert_eq!(dialect_for_path("/v1/messages/batches"), Dialect::Anthropic);
    assert_eq!(dialect_for_path("/v1/chat/completions"), Dialect::OpenAi);
    assert_eq!(dialect_for_path("/v1/embeddings"), Dialect::OpenAi);
    assert_eq!(dialect_for_path("/"), Dialect::OpenAi);
}

#[test]
fn bare_default_provider_name_rejects_gemini_v1beta_lookalike() {
    // A raw `path.starts_with("/v1")` would absorb Google Gemini's real path shape
    // (`/v1beta/models/{model}:generateContent`) into the bare-default branch, which routes it to
    // OpenAI — a silent misroute that 404s against `api.openai.com` instead of failing with a
    // clear "unknown provider" error. Boundary-checking must reject it.
    assert_eq!(
        bare_default_provider_name("/v1beta/models/gemini-2.5-pro:generateContent", None),
        None,
        "/v1beta must NOT be routed to OpenAI (or any provider) via the bare-default path"
    );
    assert_eq!(bare_default_provider_name("/v1beta", None), None);

    // The real bare-default shape still resolves correctly, dialect-picked.
    assert_eq!(
        bare_default_provider_name("/v1/messages", None),
        Some("anthropic")
    );
    assert_eq!(
        bare_default_provider_name("/v1/chat/completions", None),
        Some("openai")
    );
    assert_eq!(bare_default_provider_name("/v1", None), Some("openai"));

    // Other near-miss prefixes must also be rejected, not just /v1beta.
    assert_eq!(bare_default_provider_name("/v10/messages", None), None);
    assert_eq!(bare_default_provider_name("/v2/messages", None), None);
}

/// claim: SEC-11
/// defect: D30
#[test]
fn a_byo_x_api_key_on_bare_v1_routes_to_anthropic_on_every_path() {
    for path in [
        "/v1/files",
        "/v1/models/claude-opus-4-8",
        "/v1/chat/completions",
        "/v1",
    ] {
        assert_eq!(
            bare_default_provider_name(path, Some(Dialect::Anthropic)),
            Some("anthropic"),
            "{path}"
        );
    }
    assert_eq!(
        bare_default_provider_name("/v1beta/x", Some(Dialect::Anthropic)),
        None
    );
    let req = req_with_headers("/v1/files", &[("x-api-key", "sk-ant-byo")]);
    assert_eq!(byo_credential_dialect(&req), Ok(Some(Dialect::Anthropic)));
    for headers in [
        &[("x-api-key", "bai_v1.1.p.s")][..],
        &[("x-api-key", "")][..],
        &[("authorization", "Bearer opaque-token")][..],
    ] {
        let req = req_with_headers("/v1/files", headers);
        assert_eq!(byo_credential_dialect(&req), Ok(None), "{headers:?}");
    }
}

/// The BYO credential that will be forwarded picks the provider: by shape where it has
/// one, `x-api-key` meaning Anthropic otherwise, every value read; keys for two providers on one
/// request are ambiguous.
#[test]
fn byo_credentials_pick_the_provider_they_belong_to() {
    // Appends, so a repeated header keeps every line.
    let dialect = |headers: &[(&'static str, &'static str)]| {
        let mut req =
            pingora::http::RequestHeader::build(http::Method::POST, b"/v1/chat/completions", None)
                .unwrap();
        for (k, v) in headers {
            req.append_header(*k, *v).unwrap();
        }
        byo_credential_dialect(&req)
    };
    assert_eq!(
        dialect(&[("authorization", "Bearer sk-proj-abc")]),
        Ok(Some(Dialect::OpenAi))
    );
    assert_eq!(
        dialect(&[("authorization", "Bearer sk-ant-oat01-abc")]),
        Ok(Some(Dialect::Anthropic))
    );
    assert_eq!(
        dialect(&[("x-api-key", ""), ("x-api-key", "sk-ant-api03-x")]),
        Ok(Some(Dialect::Anthropic)),
        "every x-api-key line counts"
    );
    assert_eq!(
        dialect(&[
            ("x-api-key", "sk-ant-a"),
            ("authorization", "Bearer sk-ant-a")
        ]),
        Ok(Some(Dialect::Anthropic)),
        "the same provider twice is not ambiguous"
    );
    assert_eq!(
        dialect(&[
            ("x-api-key", "stray"),
            ("authorization", "Bearer sk-proj-abc")
        ]),
        Err(())
    );
    assert_eq!(
        dialect(&[("x-api-key", "sk-ant-a"), ("x-api-key", "sk-proj-b")]),
        Err(())
    );
    assert_eq!(
        dialect(&[
            ("x-api-key", "bai_v1.1.p.s"),
            ("authorization", "Bearer sk-proj-abc")
        ]),
        Ok(Some(Dialect::OpenAi)),
        "a managed value casts no vote"
    );
    assert_eq!(
        bare_default_provider_name("/v1/messages", Some(Dialect::OpenAi)),
        Some("openai"),
        "an OpenAI key never goes to Anthropic, whatever the path"
    );
}

/// The managed allowlist (a security boundary) and every routing decision read one endpoint
/// table, so they cannot drift: for every endpoint, under every provider's mount prefix, with
/// a trailing slash or a query, the allowlist admits it, its wire, sub-resource and
/// streamability follow from its row, and the catalog spelling (`/v1`, `/auto`) names the
/// same endpoint. A path in no row is refused by the allowlist and names nothing.
/// claim: SEC-1, CAT-14
/// defect: D209
#[test]
fn the_allowlist_and_the_routing_tables_agree_for_every_endpoint() {
    for e in &route::ENDPOINT_PATHS {
        let generation = e.sub.is_none();
        for prefix in [
            "/v1",
            "/api/v1",
            "/openai/v1",
            "/inference/v1",
            "/anthropic/v1",
            "/backend-api/codex",
            "/openai/deployments/x",
        ] {
            let base = format!("{prefix}{}", e.suffix);
            for path in [
                base.clone(),
                format!("{base}/"),
                format!("{base}?beta=true"),
            ] {
                let bare = path.split_once('?').map_or(path.as_str(), |(p, _)| p);
                assert!(is_managed_provider_endpoint(&path), "{path}");
                assert_eq!(wire_of_forward_path(&path), e.endpoint.wire(), "{path}");
                assert_eq!(route::SubResource::of_forward_path(&path), e.sub, "{path}");
                assert_eq!(
                    is_streamable_path(bare),
                    generation && e.endpoint == route::Endpoint::ChatCompletions,
                    "{path}"
                );
                assert_eq!(
                    route::forward_is_responses(bare),
                    generation && e.endpoint == route::Endpoint::Responses,
                    "{path}"
                );
            }
            if generation {
                assert_eq!(
                    route::Endpoint::of_upstream_path(&base),
                    e.endpoint,
                    "{base}"
                );
            }
        }
        for path in [
            format!("/v1{}", e.suffix),
            format!("/auto{}", e.suffix),
            format!("/auto/v1{}/", e.suffix),
        ] {
            assert_eq!(
                route::implied_endpoint(&path),
                generation.then_some(e.endpoint),
                "{path}"
            );
            assert_eq!(route::SubResource::of_path(&path), e.sub, "{path}");
        }
    }
    for path in [
        "/v1/files",
        "/v1/models",
        "/v1/batches",
        "/v1/responses/resp_123",
        "/v1/responses/resp_123/cancel",
        "/v1/messages/batches",
        "/v1/chat/completions/chatcmpl-1",
        "/v1/fine_tuning/jobs",
    ] {
        assert!(!is_managed_provider_endpoint(path), "{path}");
        assert_eq!(route::SubResource::of_forward_path(path), None, "{path}");
        assert_eq!(route::implied_endpoint(path), None, "{path}");
        assert!(!route::forward_is_responses(path), "{path}");
    }
}

#[test]
fn is_streamable_path_matches_generation_suffixes_across_prefixes() {
    // Only chat-completions gets buffered for `stream_options.include_usage` injection. The
    // check is by *suffix* so it holds whatever mount prefix the provider uses; a mismatch here
    // either skips injection on a streamable path (lost usage) or needlessly buffers a
    // non-streaming one.
    assert!(is_streamable_path("/v1/chat/completions"));
    assert!(is_streamable_path("/openai/v1/chat/completions"));
    assert!(is_streamable_path("/inference/v1/chat/completions"));
    // The Responses API must NOT be buffered/injected: it has no `stream_options` field, always
    // reports usage on its terminal event regardless, and splicing this fragment into its body
    // would inject a field the API doesn't recognize.
    assert!(!is_streamable_path("/v1/responses"));
    // Non-streaming endpoints must not be buffered.
    assert!(!is_streamable_path("/v1/embeddings"));
    assert!(!is_streamable_path("/v1/messages"));
    assert!(!is_streamable_path("/v1/models"));
}

#[test]
fn usage_tail_retains_exactly_the_last_cap_bytes() {
    // The ring must retain byte-for-byte what the old grow-and-compact buffer did: the whole
    // body while it fits, the last `USAGE_TAIL_CAP` bytes once it doesn't. Getting this wrong
    // silently truncates or misorders the usage event, which is unrecoverable once the request
    // completes — so drive it across chunk sizes that do and don't divide the cap, and across
    // the boundary itself.
    let body: Vec<u8> = (0..(3 * USAGE_TAIL_CAP + 1234))
        .map(|i| (i % 251) as u8)
        .collect();

    for chunk in [1usize, 7, 4096, 8192, USAGE_TAIL_CAP - 1, USAGE_TAIL_CAP] {
        for total in [
            0usize,
            1,
            USAGE_TAIL_CAP - 1,
            USAGE_TAIL_CAP,
            USAGE_TAIL_CAP + 1,
            2 * USAGE_TAIL_CAP + 77,
            body.len(),
        ] {
            let src = &body[..total];
            let mut tail = UsageTail::default();
            for c in src.chunks(chunk.max(1)) {
                tail.push(c);
            }
            let want = &src[src.len().saturating_sub(USAGE_TAIL_CAP)..];
            assert_eq!(
                tail.contiguous(),
                want,
                "chunk={chunk} total={total}: retained window differs"
            );
            // Idempotent — `contiguous` must not consume or re-rotate.
            assert_eq!(tail.contiguous(), want);
        }
    }

    // A single chunk larger than the whole window keeps only its last cap bytes.
    let mut tail = UsageTail::default();
    tail.push(&body);
    assert_eq!(tail.contiguous(), &body[body.len() - USAGE_TAIL_CAP..]);

    // Memory stays bounded no matter how long the stream runs.
    let mut tail = UsageTail::default();
    for _ in 0..64 {
        tail.push(&body[..USAGE_TAIL_CAP]);
    }
    assert_eq!(tail.contiguous().len(), USAGE_TAIL_CAP);
}

#[test]
fn a_client_sent_stream_options_always_asks_for_usage() {
    for (body, want_span_moves) in [
        (
            r#"{"model":"gpt-4o","stream":true,"stream_options":{"include_usage":false}}"#,
            false,
        ),
        (
            r#"{"stream_options":{},"stream":true,"model":"gpt-4o"}"#,
            true,
        ),
        (
            r#"{"stream":true,"stream_options": null ,"model":"gpt-4o"}"#,
            true,
        ),
        (
            r#"{"stream":true,"stream_options":{"x":1},"model":"gpt-4o"}"#,
            true,
        ),
    ] {
        let scan = peek::scan_buffered(body.as_bytes());
        assert_eq!(scan.inject_at, None, "{body}");
        let at = scan
            .stream_options_at
            .expect("stream_options value located");
        let (out, span) = force_include_usage(body.as_bytes().to_vec(), at, scan.model_span);
        let v: serde_json::Value = serde_json::from_slice(&out).expect("still JSON");
        assert_eq!(v["stream_options"]["include_usage"], true, "{body}");
        let (s, e) = span.expect("span");
        assert_eq!(&out[s..e], b"gpt-4o", "{body}");
        assert_eq!(span != scan.model_span, want_span_moves, "{body}");
    }
    // Already asking: untouched, byte for byte.
    let body = br#"{"stream":true,"stream_options":{ "include_usage" : true }}"#;
    let at = peek::scan_buffered(body)
        .stream_options_at
        .expect("located");
    assert_eq!(force_include_usage(body.to_vec(), at, None).0, body);
    // A non-stream body's stream_options is not ours to touch.
    let off = br#"{"stream":false,"stream_options":{"include_usage":false}}"#;
    assert_eq!(peek::scan_buffered(off).stream_options_at, None);
}

#[test]
fn in_place_splice_produces_the_same_bytes_as_a_copying_one() {
    // The splice moved from "allocate a second buffer and copy everything" to "shift the tail
    // right in place". The wire bytes must be identical, including when the buffer has no spare
    // capacity (a chunked upload, where `resize` has to grow) and when it has exactly the
    // headroom `request_filter` reserves.
    let copying = |body: &[u8], at: usize| -> Vec<u8> {
        let mut out = Vec::with_capacity(body.len() + STREAM_OPTIONS_FRAG.len());
        out.extend_from_slice(&body[..at]);
        out.extend_from_slice(STREAM_OPTIONS_FRAG);
        out.extend_from_slice(&body[at..]);
        out
    };

    for src in [
        &br#"{"model":"gpt-4o","stream":true,"messages":[]}"#[..],
        &br#"{"stream":true}"#[..],
        &b"  {  \"stream\" : true , \"model\" : \"m1\" }"[..],
    ] {
        let at = peek::scan_buffered(src).inject_at.expect("streaming body");

        // Exact-fit capacity, as `request_filter` pre-sizes it.
        let mut exact = Vec::with_capacity(src.len() + STREAM_OPTIONS_FRAG.len());
        exact.extend_from_slice(src);
        assert_eq!(
            apply_stream_usage_injection(exact, Some(at)),
            copying(src, at)
        );

        // No spare capacity at all — the chunked case.
        let tight = src.to_vec();
        let out = apply_stream_usage_injection(tight, Some(at));
        assert_eq!(out, copying(src, at));

        // ...and the result is still valid JSON carrying the option.
        let v: serde_json::Value = serde_json::from_slice(&out).expect("valid JSON");
        assert_eq!(
            v["stream_options"]["include_usage"],
            serde_json::json!(true)
        );
    }

    // Nothing to inject ⇒ the body is returned untouched, not grown.
    let untouched = br#"{"model":"gpt-4o"}"#.to_vec();
    assert_eq!(
        apply_stream_usage_injection(untouched.clone(), None),
        untouched
    );
}

#[test]
fn clip_catalog_name_caps_a_long_caller_string() {
    assert_eq!(clip_catalog_name("gpt-4o-mini"), "gpt-4o-mini");
    let long = "x".repeat(200);
    assert_eq!(clip_catalog_name(&long).len(), 128);
    // Byte 128 falls inside a two-byte character: the cut backs off to its start.
    let wide = format!("{}{}", "x".repeat(127), "é".repeat(10));
    assert_eq!(clip_catalog_name(&wide), "x".repeat(127));
}

/// Pingora's own error lines print the request without its query, where a `?key=` credential
/// rides.
/// claim: SEC-4
#[test]
fn the_summary_line_leaves_out_the_query() {
    let mut req = pingora::http::RequestHeader::build(
        http::Method::POST,
        b"/v1beta/models/g:generateContent?key=secret",
        None,
    )
    .unwrap();
    req.insert_header("host", "gw.example").unwrap();
    assert_eq!(
        summary_line(&req),
        "POST /v1beta/models/g:generateContent, Host: gw.example"
    );
}

/// The compression opt-out is spliced only just inside a non-empty root object; an empty
/// object, an array, or a body that already names `plugins` goes as sent.
/// claim: CAT-3
/// defect: D109
#[test]
fn compression_is_disabled_only_inside_a_non_empty_object() {
    let off = |b: &[u8]| disable_openrouter_compression(b.to_vec());
    for same in [
        &b"{}"[..],
        b" { } ",
        b"[1]",
        br#"[{"a":1}]"#,
        b"",
        b"  ",
        br#"{"plugins":[],"a":1}"#,
    ] {
        assert_eq!(off(same), same, "{}", String::from_utf8_lossy(same));
    }
    let spliced = |pre: &str, rest: &str| {
        let mut v = pre.as_bytes().to_vec();
        v.extend_from_slice(OPENROUTER_NO_COMPRESSION);
        v.extend_from_slice(rest.as_bytes());
        v
    };
    assert_eq!(off(br#"{"a":1}"#), spliced("{", r#""a":1}"#));
    assert_eq!(off(br#" {"a":1}"#), spliced(" {", r#""a":1}"#));
}

/// An empty `model` value (a zero-width span) is still rewritten.
/// claim: R1
#[test]
fn an_empty_model_span_is_rewritten() {
    let body = br#"{"model":""}"#.to_vec();
    assert_eq!(
        apply_model_rewrite(body, (10, 10), b"gpt-4o"),
        br#"{"model":"gpt-4o"}"#
    );
}

/// A full-body attempt that was cancelled rather than panicked is just finished: only a
/// panic has a payload to log.
/// claim: REL-17
#[tokio::test]
async fn a_cancelled_full_body_attempt_is_finished_not_a_panic() {
    let cancelled = tokio::spawn(std::future::pending::<()>());
    cancelled.abort();
    assert_eq!(reap_attempt(cancelled, "rid").await, Reaped::Finished);
}

#[test]
fn reject_bodies_are_valid_json_and_match_their_type_and_message() {
    // Every static body is a hand-written JSON literal standing in for what `serde_json::json!`
    // would build, so the thing to guard is that it still *says* what the `error_type` in the
    // log line and the metric label claim, byte for byte. A drifting literal would ship a
    // response whose `type` contradicts the reason the request ended.
    let refusals = crate::refusal::every();
    let fixed = refusals.iter().filter_map(Refusal::fixed);
    for f in fixed.chain(FAILURES) {
        let v: serde_json::Value =
            serde_json::from_str(f.body).unwrap_or_else(|e| panic!("{} is not JSON: {e}", f.body));
        assert_eq!(v["error"]["type"], f.kind, "type mismatch in {}", f.body);
        assert_eq!(
            v["error"]["message"], f.message,
            "message mismatch in {}",
            f.body
        );
        let built =
            serde_json::json!({ "error": { "type": f.kind, "message": f.message } }).to_string();
        assert_eq!(f.body, built, "constant diverges from the built body");
    }
    // A failure's lookup finds its static body rather than falling through to `serde_json`.
    for f in FAILURES {
        assert_eq!(
            error_body(f.kind, f.message).as_ptr(),
            f.body.as_ptr(),
            "{}: {}",
            f.kind,
            f.message
        );
    }
    // Every pair the old flat table held is still answered with a static body: a fixed
    // refusal's, or a failure's.
    for (typ, msg) in [
        ("invalid_request_error", "unknown provider"),
        ("authentication_error", "missing API key"),
        ("authentication_error", "invalid API key"),
        ("rate_limit_error", "rate limit exceeded"),
        ("invalid_request_error", "request body too large"),
        ("access_denied", "tenant is over limit or suspended"),
        ("api_error", "no provider key available"),
        ("api_error", "provider temporarily unavailable"),
        (
            "invalid_request_error",
            "model routing requires a managed key",
        ),
        ("insufficient_quota", "quota exhausted"),
        ("api_error", "allowance unavailable"),
        ("rate_limit_error", "too many concurrent requests"),
        ("api_error", "too many large request bodies in flight"),
        (
            "api_error",
            "upstream timed out after receiving the request",
        ),
        ("api_error", "upstream failed after receiving the request"),
    ] {
        let as_refusal = refusals
            .iter()
            .find(|r| r.fixed().is_some_and(|f| f.kind == typ && f.message == msg));
        let static_body = match as_refusal {
            Some(r) => r.body().as_ptr() == r.fixed().map_or(std::ptr::null(), |f| f.body.as_ptr()),
            None => FAILURES.iter().any(|f| f.kind == typ && f.message == msg),
        };
        assert!(static_body, "({typ:?}, {msg:?}) lost its static body");
    }
}

#[test]
fn every_reject_call_site_has_a_precomputed_body() {
    // A refusal is typed, so its body cannot miss its table. What can is a failure: each
    // message the production half passes to `fail_boxed` or `gateway_error`, or `fail_to_proxy`
    // assigns, must have its static body in `FAILURES`, under the type it is answered with,
    // and every entry there must have a producer (no entry left behind after its failure
    // went). And no body may be written past both tables: every writer is listed by the
    // function that calls it. Only the production half of the file: this test mentions the call
    // forms itself.
    // The proxy module's files, each up to its tests.
    let src: &'static str = [
        include_str!("../proxy.rs"),
        include_str!("billing.rs"),
        include_str!("full_body.rs"),
        include_str!("headers.rs"),
        include_str!("hooks.rs"),
        include_str!("phases.rs"),
        include_str!("relay.rs"),
        include_str!("replies.rs"),
        include_str!("serving.rs"),
        include_str!("upstream.rs"),
    ]
    .iter()
    .map(|file| {
        file.split("#[cfg(test)]")
            .next()
            .expect("split always yields at least one part")
    })
    .collect::<String>()
    .leak();
    let typ = |status: u16| match status {
        429 => "rate_limit_error",
        400..=499 => "invalid_request_error",
        _ => "api_error",
    };
    // The string literals in the call that starts at `at`, up to its closing `)` line.
    let literals = |at: usize| -> Vec<&str> {
        let tail = &src[at..];
        let end = tail.find(")\n").map_or(tail.len(), |e| e + 1);
        tail[..end]
            .match_indices('"')
            .collect::<Vec<_>>()
            .chunks(2)
            .filter_map(|c| match c {
                [(a, _), (b, _)] => Some(&tail[a + 1..*b]),
                _ => None,
            })
            .collect()
    };
    // The numeric status argument of `gateway_error(status, "…")`.
    let status_arg = |at: usize| -> u16 {
        let tail = &src[at + "gateway_error(".len()..];
        tail[..tail.find(',').expect("a status argument")]
            .trim()
            .parse()
            .expect("a numeric status")
    };
    // Every call of `fail_boxed` and `gateway_error` but their definitions passes its message as
    // a literal: a site whose message could not be read would escape the pairing below.
    let is_definition = |at: usize| src[..at].ends_with("fn ");
    let mut produced = Vec::new();
    let mut sites = (0usize, 0usize);
    for (i, _) in src.match_indices("fail_boxed(") {
        if is_definition(i) {
            continue;
        }
        let msg = *literals(i)
            .first()
            .expect("a fail_boxed site with a message literal");
        produced.push(("api_error", msg));
        sites.0 += 1;
    }
    for (i, _) in src.match_indices("gateway_error(") {
        if is_definition(i) {
            continue;
        }
        let msg = *literals(i)
            .first()
            .expect("a gateway_error site with a message literal");
        produced.push((typ(status_arg(i)), msg));
        sites.1 += 1;
    }
    assert!(
        sites.0 > 0 && sites.1 > 0,
        "the scan found no sites: {sites:?}"
    );
    // Every path that writes a response body, by the function it is written in: a refusal
    // (`write_refusal`; `fail_to_proxy` for one raised past `request_filter`), a failure
    // (`fail_boxed`, `fail_to_proxy`), the writers' own definitions, and the two bodies that
    // are no error (the models list, a cache hit). A body written anywhere else would pass
    // both tables, so it fails here until it goes through one of these.
    let mut writers = Vec::new();
    let mut current_fn = "";
    for line in src.lines() {
        let code = line.trim_start();
        if code.starts_with("//") {
            continue;
        }
        let decl = ["pub(crate) ", "pub(super) ", "pub ", "const ", "async "]
            .iter()
            .fold(code, |c, p| c.strip_prefix(p).unwrap_or(c));
        if let Some(rest) = decl.strip_prefix("fn ") {
            current_fn = &rest[..rest.find(['(', '<']).unwrap_or(rest.len())];
        }
        for needle in [
            "write_json_body(",
            "error_body(",
            "json_body(",
            "refusal.body()",
            "write_response_body(",
        ] {
            for (at, _) in line.match_indices(needle) {
                if !line[..at].ends_with(|c: char| c.is_ascii_alphanumeric() || c == '_') {
                    writers.push((needle, current_fn));
                }
            }
        }
    }
    writers.sort_unstable();
    let mut want = vec![
        ("error_body(", "error_body"),
        ("error_body(", "fail_boxed"),
        ("error_body(", "fail_to_proxy"),
        ("json_body(", "error_body"),
        ("refusal.body()", "fail_to_proxy"),
        ("refusal.body()", "write_refusal"),
        ("write_json_body(", "fail_boxed"),
        ("write_json_body(", "fail_to_proxy"),
        ("write_json_body(", "write_json_body"),
        ("write_json_body(", "write_refusal"),
        ("write_response_body(", "reply_cache_hit"),
        ("write_response_body(", "reply_models_list"),
        ("write_response_body(", "write_json_body"),
    ];
    want.sort_unstable();
    assert_eq!(
        writers, want,
        "a response body is written outside the tabled paths"
    );
    // `fail_to_proxy` names a delivered request's failure and a refused stream itself, under
    // the type `failure_response` gives the error class reaching that arm: the message is read
    // from its source and the type from the mapping, so either drifting fails here.
    let ftp = src
        .find("async fn fail_to_proxy(")
        .expect("fail_to_proxy is in the production half");
    let ftp = &src[ftp..ftp + src[ftp..].find("\n    }\n").expect("fail_to_proxy ends")];
    for (class, msg) in [
        (
            pingora_core::ErrorType::ReadTimedout,
            "upstream timed out after receiving the request",
        ),
        (
            pingora_core::ErrorType::ConnectionClosed,
            "upstream failed after receiving the request",
        ),
        (
            pingora_core::ErrorType::H2Error,
            "the provider refused the stream; it was not processed",
        ),
    ] {
        assert!(
            ftp.contains(&format!("msg = \"{msg}\";")),
            "fail_to_proxy no longer answers {msg:?}"
        );
        let (_, typ, _) = failure_response(&pingora_core::Error::new_up(class))
            .expect("an upstream error is answered");
        produced.push((typ, msg));
    }
    // Every class `failure_response` maps, by its source and type.
    {
        use pingora_core::{Error, ErrorType as T};
        let mut classes = vec![
            Error::new_down(T::ReadTimedout),
            Error::new_down(T::InvalidHTTPHeader),
            Error::new_up(T::ConnectTimedout),
            Error::new_up(T::ConnectRefused),
            Error::new_up(T::H2Error),
            Error::new(T::InternalError),
        ];
        for status in [400, 401, 413, 429, 500, 502, 503, 504] {
            classes.push(Error::new(T::HTTPStatus(status)));
        }
        for e in &classes {
            let (_, typ, msg) = failure_response(e).expect("answered");
            produced.push((typ, msg));
        }
    }
    // And the other way: every message `fail_to_proxy` assigns, and every message literal in
    // `failure_response`, is one produced above, so an arm added to either without a case here
    // fails rather than going unchecked.
    let quoted = |s: &'static str| -> Vec<&'static str> {
        s.match_indices('"')
            .collect::<Vec<_>>()
            .chunks(2)
            .filter_map(|c| match c {
                [(a, _), (b, _)] => Some(&s[a + 1..*b]),
                _ => None,
            })
            .collect()
    };
    for (at, _) in ftp.match_indices("msg = \"") {
        let msg = quoted(&ftp[at..])[0];
        assert!(
            produced.iter().any(|&(_, m)| m == msg),
            "fail_to_proxy answers {msg:?}, which no case here produces"
        );
    }
    let fr = src
        .find("fn failure_response(")
        .expect("failure_response is in the production half");
    let fr = &src[fr..fr + src[fr..].find("\n}\n").expect("failure_response ends")];
    let types = ["rate_limit_error", "invalid_request_error", "api_error"];
    for msg in quoted(fr).into_iter().filter(|m| !types.contains(m)) {
        assert!(
            produced.iter().any(|&(_, m)| m == msg),
            "failure_response answers {msg:?}, which no class here produces"
        );
    }
    for &(typ, msg) in &produced {
        assert!(
            FAILURES.iter().any(|f| f.kind == typ && f.message == msg),
            "({typ:?}, {msg:?}) has no FAILURES entry: it would build its body on every hit",
        );
    }
    for f in FAILURES {
        assert!(
            produced.contains(&(f.kind, f.message)),
            "FAILURES entry ({:?}, {:?}) has no producer",
            f.kind,
            f.message
        );
    }
}

#[test]
fn rejection_labels_round_trip_and_are_unique() {
    use crate::metrics::Rejection;
    use std::collections::HashSet;
    // `Rejection::ALL` rather than a copy of the variant list: a copy silently stops covering
    // whatever is added next, which is exactly what it happened to do.
    let all = Rejection::ALL;
    let labels: HashSet<&str> = all.iter().map(|r| r.label()).collect();
    assert_eq!(labels.len(), all.len(), "duplicate rejection label");
    for r in all {
        let l = r.label();
        assert!(!l.is_empty(), "{r:?} has an empty label");
        assert!(
            l.bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'),
            "{r:?} label {l:?} must be a lowercase snake_case Prometheus label value",
        );
    }
    // The two dashboards-facing strings must not drift — existing alerts key on them.
    assert_eq!(Rejection::RateLimit.label(), "rate_limit");
    assert_eq!(
        Rejection::RateLimitByoGlobal.label(),
        "rate_limit_byo_global"
    );
    // `Throttled` must map onto the same two.
    assert_eq!(
        Rejection::from(crate::ratelimit::Throttled::PerCredential),
        Rejection::RateLimit
    );
    assert_eq!(
        Rejection::from(crate::ratelimit::Throttled::ByoGlobal),
        Rejection::RateLimitByoGlobal
    );
}

#[test]
fn is_streamable_path_must_be_given_the_path_without_the_query() {
    // The suffix match cannot see past a query string. This locks in *why*
    // `stream_options_eligible` strips the query before asking it: Azure OpenAI requires
    // `?api-version=…` on every request, and testing the path+query here returned `false` for a
    // genuine chat/completions call — so managed Azure streams skipped `stream_options`
    // injection, got no usage chunk back, and billed zero tokens with nothing logged.
    assert!(
        !is_streamable_path("/v1/chat/completions?api-version=2024-10-21"),
        "a query string defeats the suffix match — callers must strip it first"
    );
    assert!(!is_streamable_path(
        "/openai/deployments/gpt4o/chat/completions?api-version=2024-10-21"
    ));
    // ...and the same paths, query stripped, are correctly streamable.
    assert!(is_streamable_path("/v1/chat/completions"));
    assert!(is_streamable_path(
        "/openai/deployments/gpt4o/chat/completions"
    ));
}

#[test]
fn openrouter_attribution_present_only_for_openrouter_managed_traffic() {
    let mut openrouter_req =
        pingora::http::RequestHeader::build(http::Method::POST, b"/api/v1/chat/completions", None)
            .unwrap();
    apply_provider_attribution(
        &mut openrouter_req,
        Some(providers::ProviderId::OpenRouter),
        true,
    )
    .unwrap();
    assert_eq!(
        openrouter_req.headers.get("HTTP-Referer").unwrap(),
        OPENROUTER_REFERER
    );
    assert_eq!(
        openrouter_req.headers.get("X-OpenRouter-Title").unwrap(),
        OPENROUTER_TITLE
    );
    assert_eq!(
        openrouter_req
            .headers
            .get("X-OpenRouter-Categories")
            .unwrap(),
        OPENROUTER_CATEGORY
    );

    for other in ["openai", "anthropic", "fireworks", "groq"] {
        let mut req =
            pingora::http::RequestHeader::build(http::Method::POST, b"/v1/chat/completions", None)
                .unwrap();
        apply_provider_attribution(
            &mut req,
            Some(providers::by_name(other).expect("a built-in provider").id),
            true,
        )
        .unwrap();
        assert!(
            req.headers.get("HTTP-Referer").is_none(),
            "{other} should not get HTTP-Referer"
        );
        assert!(
            req.headers.get("X-OpenRouter-Title").is_none(),
            "{other} should not get X-OpenRouter-Title"
        );
        assert!(
            req.headers.get("X-OpenRouter-Categories").is_none(),
            "{other} should not get X-OpenRouter-Categories"
        );
    }
}

#[test]
fn openrouter_attribution_gated_off_for_byo_traffic() {
    // Pi gates its OpenRouter dashboard-attribution headers
    // behind a user-controllable telemetry opt-out (`isInstallTelemetryEnabled`) — before this
    // fix, this gateway injected them unconditionally, including onto a BYO caller's own
    // OpenRouter key, misattributing *their* traffic to Beyond's dashboard app. `managed=false`
    // (BYO) must suppress every one of the three headers, even for the OpenRouter provider.
    let mut byo_req =
        pingora::http::RequestHeader::build(http::Method::POST, b"/api/v1/chat/completions", None)
            .unwrap();
    apply_provider_attribution(&mut byo_req, Some(providers::ProviderId::OpenRouter), false)
        .unwrap();
    assert!(
        byo_req.headers.get("HTTP-Referer").is_none(),
        "BYO OpenRouter traffic must not get HTTP-Referer"
    );
    assert!(
        byo_req.headers.get("X-OpenRouter-Title").is_none(),
        "BYO OpenRouter traffic must not get X-OpenRouter-Title"
    );
    assert!(
        byo_req.headers.get("X-OpenRouter-Categories").is_none(),
        "BYO OpenRouter traffic must not get X-OpenRouter-Categories"
    );
}

/// A leaked default gateway state, for the phases that look a provider up.
fn phase_state() -> &'static crate::state::GatewayState {
    Box::leak(Box::new(
        crate::state::GatewayState::new(
            crate::config::AiConfig::default().validate().unwrap(),
            crate::state::test_metrics(),
        )
        .unwrap(),
    ))
}

/// `admit` is the route for this key: BYO never walks the catalog, a managed `?key=` is never
/// forwarded, and a managed walk without a header waits for the body.
#[test]
fn admit_decides_the_route_for_the_key() {
    let state = phase_state();
    let openai = state.provider("openai").unwrap();
    let id = Identity::unverified_for_test(42, 7, None);
    let row = route::model_route("gpt-5").unwrap();
    let provider_route = || Route::Provider {
        provider: openai,
        forward_path: "/v1/chat/completions?key=bai_v1.x&a=1".into(),
    };

    for header in [None, Some(row)] {
        assert!(matches!(
            admit(Route::Auto { header }, None),
            Err(Refusal::ByoOnModelRoute)
        ));
    }
    assert!(matches!(
        admit(Route::Auto { header: Some(row) }, Some(id)),
        Ok(Admission::Ready(Plan::Catalog(Walk { relay: None, .. })))
    ));
    for route in [
        Route::Auto { header: None },
        Route::BareDefault { provider: openai },
    ] {
        assert!(matches!(admit(route, Some(id)), Ok(Admission::NeedsRow(_))));
    }
    assert!(matches!(
        admit(Route::BareDefault { provider: openai }, None),
        Ok(Admission::Ready(Plan::Provider {
            forward: ForwardPath::Inbound,
            ..
        }))
    ));
    let forwarded = |identity| match admit(provider_route(), identity) {
        Ok(Admission::Ready(plan)) => plan.stripped().map(str::to_owned),
        _ => None,
    };
    assert_eq!(
        forwarded(Some(id)).as_deref(),
        Some("/v1/chat/completions?a=1")
    );
    assert_eq!(
        forwarded(None).as_deref(),
        Some("/v1/chat/completions?key=bai_v1.x&a=1")
    );
}

/// A `FullBody` re-run walks its parent's row, whatever the route resolved.
#[test]
fn a_rerun_walks_the_parents_row() {
    let id = Identity::unverified_for_test(42, 7, None);
    let parent = route::model_route("gpt-5").unwrap();
    let other = route::model_route("claude-sonnet-4-5").unwrap();
    for admission in [
        Admission::NeedsRow(id),
        Admission::Ready(Plan::Catalog(Walk {
            id,
            row: other,
            relay: None,
        })),
    ] {
        match admission.rerun(parent) {
            Admission::Ready(plan) => assert!(std::ptr::eq(plan.row().unwrap(), parent)),
            Admission::NeedsRow(_) => panic!("a re-run has its row"),
        }
    }
}

/// `managed_endpoint`: an upgrade is refused anywhere, a provider route only on its generation
/// endpoints (404 otherwise; a path that only ends in an endpoint suffix is not one), and anything
/// but POST is a 405.
/// claim: SEC-1
/// defect: D289
/// defect: D295
#[test]
fn managed_endpoint_holds_a_managed_key_to_metered_calls() {
    let state = phase_state();
    let openai = state.provider("openai").unwrap();
    let id = Identity::unverified_for_test(42, 7, None);
    let req = |method: http::Method, path: &str| {
        pingora::http::RequestHeader::build(method, path.as_bytes(), None).unwrap()
    };
    let on = |path: &str| {
        Admission::Ready(Plan::Provider {
            provider: openai,
            forward: ForwardPath::Stripped(path.into()),
        })
    };

    assert!(managed_endpoint(&req(http::Method::POST, "/v1"), &Admission::NeedsRow(id)).is_ok());
    assert!(
        managed_endpoint(
            &req(http::Method::POST, "/openai/v1/chat/completions"),
            &on("/v1/chat/completions?x=1")
        )
        .is_ok()
    );
    let mut upgrade = req(http::Method::POST, "/v1");
    upgrade.insert_header("upgrade", "websocket").unwrap();
    assert!(matches!(
        managed_endpoint(&upgrade, &Admission::NeedsRow(id)),
        Err(Refusal::ProtocolUpgrade)
    ));
    assert!(matches!(
        managed_endpoint(&req(http::Method::GET, "/v1"), &Admission::NeedsRow(id)),
        Err(Refusal::ManagedEndpoint {
            wrong_method: true,
            ..
        })
    ));
    for method in [http::Method::POST, http::Method::GET] {
        assert!(matches!(
            managed_endpoint(&req(method, "/openai/v1/files"), &on("/v1/files")),
            Err(Refusal::ManagedEndpoint {
                wrong_method: false,
                ..
            })
        ));
    }
    // Anchored: a path that merely ends in an endpoint suffix is not that endpoint.
    for path in [
        "/v1/threads/t/messages",
        "/v1/x/chat/completions",
        "/v1/responses/abc/cancel",
        "/v1/responses/messages",
        "/v1/x/messages/count_tokens",
        "/openai/deployments/a/b/chat/completions",
        "/openai/deployments//chat/completions",
        "/openai/deployments/..%2Fv1%2Fthreads%2Ft/chat/completions",
        "/openai/deployments/..%2fv1%2fthreads%2ft/chat/completions",
        "/openai/deployments/../chat/completions",
        "/openai/deployments/./chat/completions",
        "/openai/deployments/%2e%2e/chat/completions",
        "/chat/completions",
        "//v1/messages",
    ] {
        assert!(
            matches!(
                managed_endpoint(&req(http::Method::POST, "/p"), &on(path)),
                Err(Refusal::ManagedEndpoint {
                    wrong_method: false,
                    ..
                })
            ),
            "{path}"
        );
    }
    for path in [
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/v1/responses/input_tokens",
        "/v1/responses/compact?x=1",
        "/v1/chat/completions/",
        "/api/v1/chat/completions",
        "/openai/v1/embeddings",
        "/inference/v1/chat/completions",
        "/anthropic/v1/messages",
        "/backend-api/codex/responses",
        "/openai/deployments/d/chat/completions?api-version=1",
        "/openai/deployments/gpt-4.1_mini.v2/chat/completions",
    ] {
        assert!(
            managed_endpoint(&req(http::Method::POST, "/p"), &on(path)).is_ok(),
            "{path}"
        );
    }
}

/// `shape` reads the body only on a catalog walk: a second root `model` is refused there, and a
/// provider route's body is left to `request_body_filter`.
#[test]
fn shape_refuses_a_second_model_on_a_catalog_walk() {
    let row = route::model_route("gpt-5").unwrap();
    let body = br#"{"model":"gpt-5","model":"gpt-5-mini","messages":[]}"#;
    assert!(shape(Some(row), Some(body)).is_err());
    assert!(shape(None, Some(body)).is_ok());
    assert!(shape(Some(row), Some(br#"{"model":"gpt-5"}"#)).is_ok());
}
