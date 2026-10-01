//! Provider-account remedies in an upstream error (D174).
//!
//! A provider error reaches the client verbatim — status, type, code, `Retry-After`, message —
//! except advice about Beyond's own account with that provider: "add your own key", a billing or
//! settings page, "your credit balance is too low". A gateway client holds no account there and
//! cannot follow it, and the text names the upstream behind a row and describes Beyond's account
//! state. Such a message becomes [`NEUTRAL`]; what a client can act on (a context overflow, a bad
//! parameter, a content-policy refusal) never carries one of these phrases and is left alone.
//!
//! Run only on a managed error body (status >= 400, JSON, at most [`MAX_BODY`]) by the pool-key
//! scrub in `proxy.rs`, once, at end of body. A 2xx is never parsed here. A BYO error is the
//! caller's own account talking, so it is relayed as sent.

use serde_json::{Map, Value};

/// What a provider-account remedy reads as. From a client's side an exhausted provider account
/// behaves as an upstream rate limit: nothing it can change, and it clears without it.
pub const NEUTRAL: &str = "The provider is rate-limited upstream; retry later.";

/// The largest error body that is held whole to be checked. Real ones are under 2 KiB; a larger
/// one streams through unchanged (the pool-key scrub still applies).
pub const MAX_BODY: usize = 64 * 1024;

/// Phrases that occur only in advice about the provider account the request was sent on, matched
/// ASCII case-insensitively in the error's strings. Each is the remedy (or the account state it
/// remedies), never the client's mistake.
const REMEDIES: &[&str] = &[
    // OpenRouter: the shared-pool 429 ("add your own key to accumulate your rate limits:
    // https://openrouter.ai/settings/integrations"), the 402 ("purchase more at
    // https://openrouter.ai/settings/credits"), its key and BYOK pages.
    "add your own key",
    "openrouter.ai/settings",
    // OpenAI (Gemini uses the same words): insufficient_quota, and the organization rate limit,
    // which also names Beyond's org id and links its limits page.
    "check your plan and billing details",
    "platform.openai.com/account",
    // Anthropic: the billing 400 ("Please go to Plans & Billing to upgrade or purchase credits"),
    // and the organization rate limit's "contact sales" (it also names Beyond's org id).
    "credit balance is too low",
    "anthropic.com/contact-sales",
    "console.anthropic.com/settings",
    // xAI: credits spent or the monthly spending limit reached (it also names Beyond's team id).
    "purchase more credits",
    "raise your spending limit",
    "console.x.ai",
    // Groq: the organization rate limit's "Upgrade to Dev Tier" billing link.
    "console.groq.com/settings",
    // DeepSeek's 402.
    "insufficient balance",
];

/// OpenRouter error metadata that describes Beyond's account with it, not the error.
const ACCOUNT_METADATA: &[&str] = &["is_byok", "limit_source", "remedy_hint"];

/// `body` with every provider-account remedy rewritten, or `None` when it has none (or is not a
/// JSON object), so the caller relays the original bytes.
///
/// In the error object (`error` when it is an object, else the root: Bedrock's `{"message"}`,
/// xAI's `{"code", "error": "<string>"}`): a `message` or string `error` carrying a remedy becomes
/// [`NEUTRAL`], OpenRouter's `metadata.raw` is removed when it carries one (the translation quotes
/// it after the message), and `is_byok`, `limit_source` and `remedy_hint` are removed.
/// Everything else — `type`, `code`, `param`, `metadata.provider_name` — is kept.
pub fn neutralize(body: &[u8]) -> Option<Vec<u8>> {
    let mut v: Value = serde_json::from_slice(body).ok()?;
    let root = v.as_object_mut()?;
    let changed = match root.get_mut("error") {
        Some(Value::Object(err)) => scrub(err),
        _ => scrub(root),
    };
    if changed {
        serde_json::to_vec(&v).ok()
    } else {
        None
    }
}

fn scrub(err: &mut Map<String, Value>) -> bool {
    let mut changed = false;
    if let Some(meta) = err.get_mut("metadata").and_then(Value::as_object_mut) {
        for k in ACCOUNT_METADATA {
            changed |= meta.remove(*k).is_some();
        }
        if meta.get("raw").is_some_and(carries_remedy) {
            meta.remove("raw");
            changed = true;
        }
    }
    for field in ["message", "error"] {
        if let Some(s) = err.get_mut(field)
            && s.as_str().is_some_and(is_remedy)
        {
            *s = Value::from(NEUTRAL);
            changed = true;
        }
    }
    changed
}

/// Whether any string in `v` (OpenRouter's `raw` is a string, or the upstream's own JSON) carries a
/// remedy.
fn carries_remedy(v: &Value) -> bool {
    match v {
        Value::String(s) => is_remedy(s),
        Value::Array(a) => a.iter().any(carries_remedy),
        Value::Object(o) => o.values().any(carries_remedy),
        _ => false,
    }
}

fn is_remedy(s: &str) -> bool {
    let lower = s.to_ascii_lowercase();
    REMEDIES.iter().any(|r| lower.contains(r))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rewritten(body: &str) -> Option<Value> {
        neutralize(body.as_bytes()).map(|b| serde_json::from_slice(&b).unwrap())
    }

    /// OpenRouter's shared-pool 429: the message and `raw` carry the remedy and the account
    /// metadata goes; the code and the provider's name stay.
    #[test]
    fn openrouters_shared_pool_remedy_is_neutralized() {
        let v = rewritten(
            r#"{"error":{"message":"m is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations","code":429,"metadata":{"raw":"Add Your Own Key","provider_name":"Mistral","is_byok":false,"limit_source":"upstream_provider_shared_pool","remedy_hint":"byok"}},"user_id":"u"}"#,
        )
        .unwrap();
        assert_eq!(v["error"]["message"], NEUTRAL);
        assert_eq!(v["error"]["code"], 429);
        assert_eq!(
            v["error"]["metadata"],
            serde_json::json!({"provider_name": "Mistral"})
        );
        assert_eq!(v["user_id"], "u");
    }

    /// Each provider's account remedy, in its own envelope.
    #[test]
    fn every_providers_account_remedy_is_neutralized() {
        for (body, field) in [
            (
                r#"{"error":{"message":"You exceeded your current quota, please check your plan and billing details. For more information on this error, read the docs: https://platform.openai.com/docs/guides/error-codes/api-errors.","type":"insufficient_quota","param":null,"code":"insufficient_quota"}}"#,
                "/error/message",
            ),
            (
                r#"{"error":{"message":"Rate limit reached for gpt-4o in organization org-abc on tokens per min (TPM): Limit 30000, Used 29000, Requested 2000. Please try again in 2s. Visit https://platform.openai.com/account/rate-limits to learn more.","type":"tokens","param":null,"code":"rate_limit_exceeded"}}"#,
                "/error/message",
            ),
            (
                r#"{"type":"error","error":{"type":"invalid_request_error","message":"Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits."}}"#,
                "/error/message",
            ),
            (
                r#"{"code":"permission_denied","error":"Your team t-1 has either used all available credits or reached its monthly spending limit. To continue making API requests, please purchase more credits or raise your spending limit."}"#,
                "/error",
            ),
            (
                r#"{"error":{"message":"Insufficient credits. Purchase more at https://openrouter.ai/settings/credits","code":402}}"#,
                "/error/message",
            ),
            (
                r#"{"error":{"message":"Insufficient Balance","type":"unknown_error","param":null,"code":"invalid_request_error"}}"#,
                "/error/message",
            ),
        ] {
            let v = rewritten(body).unwrap_or_else(|| panic!("kept: {body}"));
            assert_eq!(v.pointer(field).unwrap(), NEUTRAL, "{body}");
            let orig: Value = serde_json::from_str(body).unwrap();
            for k in ["/error/type", "/error/code", "/code", "/type"] {
                assert_eq!(v.pointer(k), orig.pointer(k), "{k} of {body}");
            }
        }
    }

    /// What a client can act on is relayed as sent: a context overflow, a bad parameter, a content
    /// refusal, OpenRouter quoting an upstream's own error, and a plain rate limit.
    #[test]
    fn client_actionable_errors_are_kept() {
        for body in [
            r#"{"error":{"message":"This model's maximum context length is 128000 tokens. However, your messages resulted in 130000 tokens.","type":"invalid_request_error","param":"messages","code":"context_length_exceeded"}}"#,
            r#"{"error":{"message":"Unrecognized request argument supplied: foo","type":"invalid_request_error","param":null,"code":null}}"#,
            r#"{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 250000 tokens > 200000 maximum"}}"#,
            r#"{"error":{"message":"Your request was rejected as a result of our safety system.","type":"invalid_request_error","code":"content_policy_violation"}}"#,
            r#"{"error":{"message":"Provider returned error","code":400,"metadata":{"raw":"{\"error\":{\"message\":\"prompt is too long\"}}","provider_name":"Anthropic"}},"user_id":"u"}"#,
            r#"{"type":"error","error":{"type":"rate_limit_error","message":"Rate limited. Please try again later."}}"#,
            r#"{"message":"Too many tokens, please wait before trying again."}"#,
            "not json",
            "[]",
        ] {
            assert!(neutralize(body.as_bytes()).is_none(), "rewrote {body}");
        }
    }
}
