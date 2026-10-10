//! The pure pieces of the `ai.usage` row's new fields.

use super::*;

#[test]
fn price_variant_names_bedrock_profiles_and_openrouter_regions() {
    assert_eq!(
        price_variant(
            "bedrock",
            "global.anthropic.claude-opus-4-8",
            "bedrock-runtime.us-east-1.amazonaws.com"
        ),
        Some("global")
    );
    for m in [
        "us.anthropic.claude-opus-4-8",
        "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
        "anthropic.claude-opus-4-8",
    ] {
        assert_eq!(price_variant("bedrock", m, "h"), Some("regional"), "{m}");
    }
    assert_eq!(
        price_variant("openrouter", "x", "us.openrouter.ai"),
        Some("regional")
    );
    assert_eq!(price_variant("openrouter", "x", "openrouter.ai"), None);
    assert_eq!(price_variant("openrouter", "x", "evilopenrouter.ai"), None);
    assert_eq!(
        price_variant("anthropic", "global.x", "api.anthropic.com"),
        None
    );
}

#[test]
fn only_a_streamed_cancel_on_a_continuing_provider_may_continue() {
    for p in MAY_CONTINUE_PROVIDERS {
        assert!(
            upstream_may_continue(Some(p), "client_cancelled", true),
            "{p}"
        );
        assert!(upstream_may_continue(Some(p), "cut_short", true), "{p}");
        assert!(
            !upstream_may_continue(Some(p), "client_cancelled", false),
            "{p}"
        );
        assert!(!upstream_may_continue(Some(p), "ok", true), "{p}");
        assert!(
            !upstream_may_continue(Some(p), "upstream_error", true),
            "{p}"
        );
    }
    assert!(!upstream_may_continue(
        Some("anthropic"),
        "client_cancelled",
        true
    ));
    assert!(!upstream_may_continue(None, "client_cancelled", true));
}

#[test]
fn estimated_parts_and_exclusions() {
    let reported = usage::Usage {
        reasoning_tokens: Some(3),
        ..usage::Usage::default()
    };
    let unreported = usage::Usage::default();
    let p = |input, output| EstimatedParts { input, output };
    assert_eq!(p(false, false).parts(), None);
    assert_eq!(p(true, false).parts(), Some("input"));
    assert_eq!(p(false, true).parts(), Some("output"));
    assert_eq!(p(true, true).parts(), Some("input,output"));
    assert_eq!(p(false, false).excludes(&unreported), None);
    assert_eq!(p(true, false).excludes(&unreported), Some("cache"));
    assert_eq!(p(false, true).excludes(&unreported), Some("reasoning"));
    assert_eq!(p(false, true).excludes(&reported), None);
    assert_eq!(p(true, true).excludes(&unreported), Some("cache,reasoning"));
    assert_eq!(p(true, true).excludes(&reported), Some("cache"));
}

#[test]
fn only_the_gateways_own_downstream_answer_is_a_refusal() {
    let down = |t| pingora_core::Error::new(t).into_down();
    let up = |t| pingora_core::Error::new(t).into_up();
    assert!(gateway_refusal(&down(pingora_core::ErrorType::CustomCode(
        "x", 400
    ))));
    assert!(gateway_refusal(&down(pingora_core::ErrorType::HTTPStatus(
        413
    ))));
    assert!(!gateway_refusal(&down(pingora_core::ErrorType::ReadError)));
    assert!(!gateway_refusal(&up(pingora_core::ErrorType::HTTPStatus(
        400
    ))));
}
