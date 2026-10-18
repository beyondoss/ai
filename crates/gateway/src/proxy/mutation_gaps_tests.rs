use super::*;
use crate::refusal::default_retry_after;
use pingora_core::{Error, ErrorType as T};

/// Each way a request ends maps to its own `outcome` on the billing row, so a zero-token row
/// for an error or a cancel can be told from a real zero-token generation.
/// claim: BIL-12
#[test]
fn every_ending_has_its_own_outcome() {
    let mut rc = tests::test_ctx(false);
    assert_eq!(outcome(&rc, None, false, false), "no_candidate");
    assert_eq!(
        outcome(&rc, None, true, false),
        "ok",
        "a cache hit made no call"
    );
    rc.upstream_phase = UpstreamPhase::Attempted;
    let up = Error::new_up(T::ConnectRefused);
    assert_eq!(outcome(&rc, Some(&up), false, false), "upstream_error");
    let down = Error::new_down(T::ReadError);
    assert_eq!(outcome(&rc, Some(&down), false, false), "client_cancelled");
    rc.upstream_phase = UpstreamPhase::Connected;
    rc.upstream_status = NonZeroU16::new(200);
    assert_eq!(outcome(&rc, None, false, false), "ok");
    assert_eq!(outcome(&rc, Some(&up), false, false), "cut_short");
    assert_eq!(outcome(&rc, Some(&down), false, false), "client_cancelled");
    // A drained stream (the client left first) is a cancel however the request then ended.
    assert_eq!(outcome(&rc, None, false, true), "client_cancelled");
    assert_eq!(outcome(&rc, Some(&up), false, true), "client_cancelled");
    rc.upstream_status = NonZeroU16::new(400);
    assert_eq!(outcome(&rc, None, false, false), "upstream_error");
    assert_eq!(outcome(&rc, Some(&down), false, false), "upstream_error");
}

/// Only a 401 is a pool key failing: a 402 or 403 is about the request or the account,
/// a candidate refusal that walks the catalog but never cools or walks keys.
/// claim: REL-4
#[test]
fn pool_key_failures_are_401_only() {
    for (status, key, candidate) in [
        (400, false, false),
        (401, true, true),
        (402, false, true),
        (403, false, true),
        (404, false, false),
        (429, false, false),
        (500, false, false),
    ] {
        assert_eq!(is_pool_key_failure(status), key, "{status}");
        assert_eq!(is_candidate_refusal(status), candidate, "{status}");
    }
}

/// A complete first `data:` line the error check cannot read is an answer, not "undecided".
/// claim: REL-8
#[test]
fn an_unreadable_complete_first_event_is_an_answer() {
    assert_eq!(body_reports_error(b"data: {\"err\n\n", true), Some(false));
    assert_eq!(
        body_reports_error(b"data: {\"err\ndata: {\"error\":{}}\n\n", true),
        Some(false),
        "the first event decides"
    );
}

/// Only the `Bearer` scheme carries a token; another six-letter scheme does not, and neither
/// does `Bearer` run into its token.
/// claim: SEC-11
#[test]
fn only_a_spaced_bearer_scheme_carries_a_token() {
    assert_eq!(bearer_token("Bearer bai_v1x"), Some("bai_v1x"));
    assert_eq!(bearer_token("bEaReR \t bai_v1x"), Some("bai_v1x"));
    assert_eq!(bearer_token("Digest bai_v1x"), None);
    assert_eq!(bearer_token("Bearerbai_v1x"), None);
}

/// A space is not a control byte: a model id with one is kept on the billing row.
/// claim: BIL-13
#[test]
fn a_model_id_with_a_space_is_kept() {
    assert_eq!(sanitize_model("my model".to_owned()), "my model");
    assert_eq!(sanitize_model("bad\u{1f}model".to_owned()), "unknown");
}

/// A dated snapshot prices as its base model, in either spelling the vendors use.
/// claim: BIL-13, BIL-24
/// defect: D278
#[test]
fn a_dated_snapshot_prices_as_its_base_model() {
    for id in ["gpt-4o-mini-2024-07-18", "gpt-4o-mini-20240718"] {
        assert_eq!(price_row(id).map(|r| r.model), Some("gpt-4o-mini"), "{id}");
    }
}

/// Only an eight-digit or `YYYY-MM-DD` suffix is a snapshot date; an all-digit suffix of
/// another length (`gpt-4-0613`) and an eight-letter one are not.
/// claim: BIL-13
#[test]
fn only_real_snapshot_dates_are_stripped() {
    assert_eq!(
        strip_snapshot_date("claude-sonnet-4-5-20250929"),
        Some("claude-sonnet-4-5")
    );
    assert_eq!(strip_snapshot_date("gpt-5-2025-08-07"), Some("gpt-5"));
    assert_eq!(strip_snapshot_date("gpt-4-0613"), None);
    assert_eq!(strip_snapshot_date("model-abcdefgh"), None);
    assert_eq!(strip_snapshot_date("model-v2-05-12"), None, "no year");
    assert_eq!(
        strip_snapshot_date("model-2025-5-12"),
        None,
        "one-digit month"
    );
}

/// `Retry-After` defaults: 5s for an unavailable allowance, 1s for any other 429/503, none
/// otherwise.
/// Rejection lines are capped per second: the second's first `per_sec` lines are logged, the
/// rest suppressed and counted, and the next logged line (in a later second) reports how many.
/// defect: D263
#[test]
fn rejection_lines_are_capped_per_second_and_the_rest_counted() {
    let rate = LogRate::new(3);
    assert_eq!(rate.admit(1), Some(0));
    assert_eq!(rate.admit(1), Some(0));
    assert_eq!(rate.admit(1), Some(0));
    assert_eq!(rate.admit(1), None);
    assert_eq!(rate.admit(1), None);
    // A new second refills the allowance; its first line reports the two suppressed.
    assert_eq!(rate.admit(2), Some(2));
    assert_eq!(rate.admit(2), Some(0));
    // A racing caller with an older second neither rewinds the window nor refills it.
    assert_eq!(rate.admit(1), Some(0));
    assert_eq!(rate.admit(2), None);
    assert_eq!(rate.admit(5), Some(1));
}

/// claim: REL-19
#[test]
fn default_retry_after_by_status_and_message() {
    assert_eq!(default_retry_after(503, "allowance unavailable"), Some(5));
    assert_eq!(
        default_retry_after(503, "provider temporarily unavailable"),
        Some(1)
    );
    assert_eq!(default_retry_after(429, "allowance unavailable"), Some(1));
    assert_eq!(default_retry_after(429, "rate limit exceeded"), Some(1));
    assert_eq!(default_retry_after(400, "bad request"), None);
    assert_eq!(default_retry_after(502, "upstream unavailable"), None);
}

/// The client-facing status, type and message for every class of error that ends a request.
/// claim: REL-1, REL-2
#[test]
fn failure_response_maps_every_error_class() {
    type Want = Option<(u16, &'static str, &'static str)>;
    let cases: [(Box<Error>, Want); 18] = [
        (
            Error::new(T::CustomCode("allowance unavailable", 503)),
            Some((503, "api_error", "allowance unavailable")),
        ),
        (
            Error::new(T::CustomCode("too many", 429)),
            Some((429, "rate_limit_error", "too many")),
        ),
        (
            Error::new(T::CustomCode("nope", 404)),
            Some((404, "invalid_request_error", "nope")),
        ),
        (
            Error::new(T::HTTPStatus(400)),
            Some((400, "invalid_request_error", "bad request")),
        ),
        (
            Error::new(T::HTTPStatus(413)),
            Some((413, "invalid_request_error", "request body too large")),
        ),
        (
            Error::new(T::HTTPStatus(429)),
            Some((429, "rate_limit_error", "rate limit exceeded")),
        ),
        (
            Error::new(T::HTTPStatus(502)),
            Some((502, "api_error", "upstream unavailable")),
        ),
        (
            Error::new(T::HTTPStatus(503)),
            Some((503, "api_error", "provider temporarily unavailable")),
        ),
        (
            Error::new(T::HTTPStatus(504)),
            Some((504, "api_error", "upstream timed out")),
        ),
        (
            Error::new(T::HTTPStatus(401)),
            Some((401, "invalid_request_error", "request rejected")),
        ),
        (
            Error::new(T::HTTPStatus(500)),
            Some((500, "api_error", "upstream error")),
        ),
        (Error::new_down(T::ConnectionClosed), None),
        (
            Error::new_down(T::ReadTimedout),
            Some((408, "invalid_request_error", "request body timed out")),
        ),
        (
            Error::new_down(T::InvalidHTTPHeader),
            Some((400, "invalid_request_error", "bad request")),
        ),
        (
            Error::new_up(T::ReadTimedout),
            Some((504, "api_error", "upstream timed out")),
        ),
        (
            Error::new_up(T::ConnectRefused),
            Some((502, "api_error", "could not connect to the provider")),
        ),
        (
            Error::new_up(T::ConnectionClosed),
            Some((502, "api_error", "upstream connection failed")),
        ),
        (
            Error::new(T::InternalError),
            Some((500, "api_error", "internal error")),
        ),
    ];
    for (e, want) in cases {
        assert_eq!(failure_response(&e), want, "{e}");
    }
}

/// Only OpenAI's own Chat Completions takes `max_completion_tokens` in place of `max_tokens`:
/// not another vendor's Chat endpoint, and not OpenAI's Responses.
/// claim: TRN-5
#[test]
fn only_native_openai_chat_is_respelled() {
    let c = |provider, path| route::Candidate {
        provider,
        upstream_model: "m",
        path,
    };
    use providers::ProviderId::{OpenAi, OpenRouter};
    assert!(native_openai_chat(&c(OpenAi, "/v1/chat/completions")));
    assert!(!native_openai_chat(&c(
        OpenRouter,
        "/api/v1/chat/completions"
    )));
    assert!(!native_openai_chat(&c(OpenAi, "/v1/responses")));
}

/// A limit already at the serving row's maximum is not rewritten.
/// claim: TRN-5
#[test]
fn a_limit_equal_to_the_cap_is_left_alone() {
    let mut body = br#"{"max_tokens":16384}"#.to_vec();
    let scan = peek::scan_buffered(&body);
    assert!(!clamp_output_limits(&mut body, &scan.limit_spans, 16384));
    assert_eq!(body, br#"{"max_tokens":16384}"#);
}

/// An empty `model` value is still replaced with the serving candidate's id.
/// claim: R1, BIL-13
#[test]
fn an_empty_model_value_is_rewritten() {
    let body = br#"{"model":"","messages":[]}"#.to_vec();
    let scan = peek::scan_buffered(&body);
    let span = scan.model_span.expect("an empty value still has a span");
    let out = apply_model_rewrite(body, span, b"gpt-4o");
    assert_eq!(out, br#"{"model":"gpt-4o","messages":[]}"#);
}

/// Forcing `include_usage` keeps the client's other `stream_options` members.
/// claim: BIL-2
#[test]
fn forcing_include_usage_keeps_other_stream_options() {
    let body =
        br#"{"stream":true,"stream_options":{"include_usage":false,"include_obfuscation":false}}"#
            .to_vec();
    let at = peek::scan_buffered(&body)
        .stream_options_at
        .expect("streams with options");
    let (out, _) = force_include_usage(body, at, None);
    let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(v["stream_options"]["include_usage"], true);
    assert_eq!(v["stream_options"]["include_obfuscation"], false);
}

/// A reused connection's failure resends only when **every** condition holds: a reused connection, pingora's
/// `ReusedOnly` verdict, a read error, and a reset or abort at the root. With any one false the
/// request may have reached a server, so it is never sent twice.
/// claim: REL-1, REL-11
#[test]
fn a_reset_before_reading_needs_every_condition() {
    use std::io::{Error as Io, ErrorKind as K};
    let err = |etype: T, kind: K, reused_only: bool| {
        let mut e = Error::because(etype, "upstream", Io::from(kind));
        if reused_only {
            e.retry = pingora_core::RetryType::ReusedOnly;
        }
        e
    };
    let reset = err(T::ReadError, K::ConnectionReset, true);
    assert!(reset_before_reading(&reset, true));
    let aborted = err(T::ReadError, K::ConnectionAborted, true);
    assert!(reset_before_reading(&aborted, true));
    // A fresh connection: the reset says nothing about an idle-close race.
    assert!(!reset_before_reading(&reset, false));
    // Pingora did not judge it retryable on reuse.
    let decided = err(T::ReadError, K::ConnectionReset, false);
    assert!(!reset_before_reading(&decided, true));
    // Not a read: a write that met the reset may have delivered part of the request.
    let write = err(T::WriteError, K::ConnectionReset, true);
    assert!(!reset_before_reading(&write, true));
    // A timeout is a liveness verdict on a peer that may have taken the request.
    let timed_out = err(T::ReadError, K::TimedOut, true);
    assert!(!reset_before_reading(&timed_out, true));
}

/// The usage tail at the exact moment it becomes a ring and across its wrap: a chunk that
/// lands flush on the end (no remainder, `head` back to 0), then a final usage event that
/// straddles the end. Any slip here truncates or misorders the event billing reads.
/// claim: BIL-15
#[test]
fn the_usage_tail_wraps_at_exactly_its_capacity() {
    let stream: Vec<u8> = (0..2 * USAGE_TAIL_CAP).map(|i| (i % 253) as u8).collect();
    let mut tail = UsageTail::default();
    tail.push(&stream[..USAGE_TAIL_CAP]);
    assert!(
        tail.ring(),
        "exactly the cap is a ring with nothing dropped"
    );
    assert_eq!(tail.head, 0);
    tail.push(&stream[USAGE_TAIL_CAP..USAGE_TAIL_CAP + 10]);
    tail.push(&stream[USAGE_TAIL_CAP + 10..2 * USAGE_TAIL_CAP - 3]);
    assert_eq!(tail.head, USAGE_TAIL_CAP - 3);
    tail.push(&stream[2 * USAGE_TAIL_CAP - 3..]);
    assert_eq!(tail.head, 0, "flush on the end wraps head to the front");
    assert_eq!(tail.contiguous(), &stream[USAGE_TAIL_CAP..]);

    // Outgrown (compacted to head 0), then filled to 5 bytes short of the end, so the usage
    // event's first 5 bytes land at the end and the rest wrap to the front.
    let mut tail = UsageTail::default();
    tail.push(&stream[..USAGE_TAIL_CAP + 5]);
    tail.push(&stream[USAGE_TAIL_CAP + 5..2 * USAGE_TAIL_CAP - 5]);
    assert_eq!(tail.head, USAGE_TAIL_CAP - 10);
    tail.push(&stream[2 * USAGE_TAIL_CAP - 5..]);
    assert_eq!(tail.head, USAGE_TAIL_CAP - 5);
    let event = br#"data: {"usage":{"total_tokens":7}}"#;
    tail.push(event);
    assert_eq!(tail.head, event.len() - 5);
    let mut whole = stream.clone();
    whole.extend_from_slice(event);
    assert_eq!(tail.contiguous(), &whole[whole.len() - USAGE_TAIL_CAP..]);
}

/// A body write that met an HTTP/2 stream already closed is pingora's `H2Error` with a
/// capacity context, found anywhere down the cause chain; another type or context is not it.
/// claim: REL-22
#[test]
fn only_an_h2_capacity_failure_is_an_unsent_body() {
    for ctx in ["cannot reserve capacity", "while waiting for capacity"] {
        assert!(h2_body_unsent(&Error::explain(T::H2Error, ctx)), "{ctx}");
        let wrapped = Error::because(
            T::WriteError,
            "writing the body",
            Error::explain(T::H2Error, ctx),
        );
        assert!(h2_body_unsent(&wrapped), "caused by {ctx}");
        assert!(
            !h2_body_unsent(&Error::explain(T::WriteError, ctx)),
            "not H2: {ctx}"
        );
    }
    assert!(!h2_body_unsent(&Error::explain(T::H2Error, "stream reset")));
    assert!(!h2_body_unsent(&Error::new(T::H2Error)));
}

/// With no key to scrub (none sent, or an empty one) and no body to hold, a chunk is relayed as
/// it came, never searched; a body held whole is held whether or not there is a key.
/// claim: SEC-7
#[test]
fn redact_holds_a_whole_body_even_with_no_key() {
    let chunk = Bytes::from_static(br#"{"error":{"message":"x"}}"#);
    let empty = memchr::memmem::Finder::new(b"");
    let mut r = Redact::default();
    let got = r.feed(std::slice::from_ref(&empty), Some(chunk.clone()), false);
    assert_eq!(got.map(|b| b.as_ptr()), Some(chunk.as_ptr()));
    let mut r = Redact {
        whole: true,
        status: 400,
        ..Redact::default()
    };
    assert_eq!(r.feed(&[], Some(chunk.clone()), false), Some(Bytes::new()));
    assert_eq!(r.feed(&[], None, true), Some(chunk));
}

/// An attempt that ended unanswered is a 502 that says whether it panicked.
/// claim: REL-1
/// defect: D207
#[test]
fn an_unanswered_attempt_says_whether_it_panicked() {
    let context = |e: Box<Error>| e.context.as_ref().map(|c| c.as_str().to_owned());
    let panicked = Reaped::Panicked.unanswered();
    assert_eq!(panicked.etype(), &T::HTTPStatus(502));
    assert_eq!(
        context(panicked).as_deref(),
        Some("full-body attempt panicked")
    );
    for quiet in [Reaped::Finished, Reaped::StillRunning] {
        assert_eq!(
            context(quiet.unanswered()).as_deref(),
            Some("full-body attempt ended without a response")
        );
    }
}

/// An `anthropic-beta` line whose every token is allowed goes out exactly as the client wrote
/// it, spacing included: only a value with a token to drop is rebuilt.
/// claim: SEC-6
#[test]
fn an_allowed_beta_line_is_forwarded_untouched() {
    let value = "prompt-caching-2024-07-31,  interleaved-thinking-2025-05-14";
    let mut req = pingora::http::RequestHeader::build("POST", b"/v1/messages", None).unwrap();
    req.insert_header("anthropic-beta", value).unwrap();
    retain_managed_client_headers(&mut req, true, true).unwrap();
    assert_eq!(req.headers.get("anthropic-beta").unwrap(), value);
}

/// Fast mode's beta rides on a managed request to direct Anthropic, the only provider with fast
/// mode, and is dropped on the way to any other.
/// claim: SEC-6
/// defect: D266
#[test]
fn the_fast_mode_beta_reaches_direct_anthropic_only() {
    let value = "prompt-caching-2024-07-31,fast-mode-2026-02-01";
    let header = |anthropic| {
        let mut req = pingora::http::RequestHeader::build("POST", b"/v1/messages", None).unwrap();
        req.insert_header("anthropic-beta", value).unwrap();
        retain_managed_client_headers(&mut req, anthropic, true).unwrap();
        req.headers
            .get("anthropic-beta")
            .map(|v| v.to_str().unwrap().to_owned())
    };
    assert_eq!(header(true).as_deref(), Some(value));
    assert_eq!(header(false).as_deref(), Some("prompt-caching-2024-07-31"));
}

/// The fine-grained-tool-streaming beta rides only to a candidate that takes fine-grained tool
/// streaming; for any other (an older Bedrock stack) it goes with each tool's
/// `eager_input_streaming`. Alone on the line, the header goes entirely.
/// claim: SEC-6
/// defect: D272
#[test]
fn the_fine_grained_tool_streaming_beta_reaches_only_a_candidate_that_takes_it() {
    let header = |value: &str, eager_tool_streaming| {
        let mut req = pingora::http::RequestHeader::build("POST", b"/v1/messages", None).unwrap();
        req.insert_header("anthropic-beta", value).unwrap();
        retain_managed_client_headers(&mut req, false, eager_tool_streaming).unwrap();
        req.headers
            .get("anthropic-beta")
            .map(|v| v.to_str().unwrap().to_owned())
    };
    let both = "fine-grained-tool-streaming-2025-05-14,prompt-caching-2024-07-31";
    assert_eq!(header(both, true).as_deref(), Some(both));
    assert_eq!(
        header(both, false).as_deref(),
        Some("prompt-caching-2024-07-31")
    );
    assert_eq!(header(FINE_GRAINED_TOOL_STREAMING_BETA, false), None);
}

/// The first chunk after the client left an armed stream starts the drain and gives the tenant's
/// slot back there, not at `logging`: the slot bounds what a client has open, and a drain can read
/// on long after it left.
/// claim: A3
#[test]
fn a_drain_frees_the_tenant_slot_at_its_first_chunk() {
    let state: &'static Arc<GatewayState> = Box::leak(Box::new(
        GatewayState::new(
            crate::config::AiConfig::default().validate().unwrap(),
            crate::state::test_metrics(),
        )
        .unwrap(),
    ));
    let slots: &'static crate::concurrency::TenantSlots =
        Box::leak(Box::new(crate::concurrency::TenantSlots::new(1).unwrap()));
    let id = Identity::unverified_for_test(42, 7, None);
    let mut held = Held {
        state,
        request_id: None,
        in_flight: false,
        active_stream: false,
        tenant: slots.take(&id),
        body_bytes: 0,
        body_exempt: false,
        client_left: AtomicBool::new(true),
        drain: Drain::Armed,
    };
    assert_eq!(slots.in_flight(42), 1);
    let mut rc = tests::test_ctx(false);
    assert!(drain_chunk(&mut rc, &mut held, false), "drained, not sent");
    assert_eq!(held.drain, Drain::Draining);
    assert_eq!(slots.in_flight(42), 0, "the slot went with the client");
}

/// A leg gives a half-open breaker's only probe permit back with no outcome, whether released
/// (the client left) or dropped with it still out (a panic unwinding the request), so the next
/// request can probe.
/// claim: REL-6
#[test]
fn a_released_or_dropped_leg_returns_its_probe_permit() {
    static NOW: AtomicU64 = AtomicU64::new(1_000);
    fn clock() -> u64 {
        NOW.load(Ordering::Relaxed)
    }
    let breaker = CircuitBreaker::with_clock(
        crate::circuit_breaker::CircuitBreakerConfig::consecutive(1)
            .reset_timeout(Duration::from_secs(30))
            .half_open_permits(1),
        clock,
    );
    let trip = breaker.allow().expect("a closed breaker admits");
    breaker.record_failure_for(trip);
    NOW.store(1_030, Ordering::Relaxed);
    let p = Arc::new(Provider::resolve(
        "openai",
        "api.openai.com:443".to_string(),
        Dialect::OpenAi,
        route::AuthScheme::Bearer,
        &["sk-pool"],
        crate::metrics::ProviderMetrics::disconnected(),
        Some(breaker),
    ));
    let breaker = p.breaker.as_ref().unwrap();
    let probe = breaker.allow().expect("the reset timeout admits one probe");
    let mut leg = Leg::new(Arc::clone(&p), Some(probe), 0);
    assert!(breaker.allow().is_err(), "the probe is out");
    leg.release();
    let probe = breaker.allow().expect("a released probe is free again");
    drop(Leg::new(Arc::clone(&p), Some(probe), 0));
    assert!(
        breaker.allow().is_ok(),
        "a dropped leg's probe is free again"
    );
}
