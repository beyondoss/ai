//! Provider-account remedies in an upstream error (D174), and the out-of-credit answers among
//! them (D180).
//!
//! A provider error reaches the client verbatim — status, type, code, `Retry-After`, message —
//! except advice about Beyond's own account with that provider: "add your own key", a billing or
//! settings page, "your credit balance is too low". A gateway client holds no account there and
//! cannot follow it, and the text names the upstream behind a row and describes Beyond's account
//! state. Such a message becomes [`RATE_LIMITED`] or [`UNAVAILABLE`] (or [`TOO_LARGE`], for a 429
//! that says the request alone is over the limit, D196); what a client can act on (a
//! context overflow, a bad parameter, a content-policy refusal) never carries one of these phrases
//! and is left alone.
//!
//! Two decisions, kept apart because only one of them is shared (D200). Whether the account is
//! **out of credit** ([`Neutralized::unfunded`]) cools Beyond's pool key for every tenant, so it is
//! read only from what the provider sets and the request cannot: the status, a `code` or `type`,
//! or a provider's whole sentence ([`unfunded`]). The **message rewrite** is cosmetic and matches
//! phrases, but only on a status about the account (401, 402, 403, 429), because a 400 or 404 may
//! quote the tenant's own model name or parameter back.
//!
//! Run only on a managed error body (status >= 400, JSON, at most [`MAX_BODY`]) by the pool-key
//! scrub in `proxy.rs`, once, at end of body. A 2xx is never parsed here. A BYO error is the
//! caller's own account talking, so it is relayed as sent.

use serde_json::{Map, Value};

/// What an account remedy on a rate limit (a 429 that is not out of credit) reads as: OpenRouter's
/// shared upstream pool, an organization's tokens per minute. Waiting clears it.
pub const RATE_LIMITED: &str = "The provider is rate-limited upstream; retry later.";

/// What an out-of-credit answer, or an account remedy on any other status, reads as. True from
/// the client's side without describing the account: this provider cannot take the request now,
/// and the walk sends later requests to the row's next candidate (D180).
pub const UNAVAILABLE: &str =
    "The provider cannot serve this request right now; retry later or use another model.";

/// What a rate limit reads as when the provider said the request alone is over the limit, so no
/// wait admits it (OpenAI's "Request too large for gpt-4.1 ... on tokens per min (TPM): Limit
/// 1000000, Requested 1101959. The input or output tokens must be reduced in order to run
/// successfully."): the client must shrink the request, and [`RATE_LIMITED`]'s "retry later"
/// would send it round the same 429 forever (D196).
pub const TOO_LARGE: &str = "The request is larger than the provider's per-minute token limit admits; \
     reduce the input or output tokens. Retrying it unchanged will not succeed.";

/// Phrases with which a 429 says the request by itself exceeds the limit (OpenAI's TPM "Request
/// too large"), matched ASCII case-insensitively.
const TOO_LARGE_PHRASES: &[&str] = &[
    "request too large",
    "must be reduced in order to run successfully",
];

/// The largest error body that is held whole to be checked. Real ones are under 2 KiB; a larger
/// one streams through unchanged (the pool-key scrub still applies).
pub const MAX_BODY: usize = 64 * 1024;

/// Phrases that occur only in advice about Beyond's account with the provider, matched ASCII
/// case-insensitively in the error's strings. **Cosmetic only**: they choose what a message is
/// rewritten to, and only on an account status ([`account_status`]). They never decide that a key
/// is out of credit ([`unfunded`] reads structured fields): a provider quotes the request in its
/// errors, so any of these can arrive in a tenant's own text (D200).
const REMEDIES: &[&str] = &[
    // OpenAI (Gemini uses the same words): insufficient_quota, and its billing page.
    "check your plan and billing details",
    // Anthropic's billing 400 ("Please go to Plans & Billing to upgrade or purchase credits").
    "credit balance is too low",
    "console.anthropic.com/settings",
    // xAI: credits spent or the monthly spending limit reached (it also names Beyond's team id).
    "purchase more credits",
    "raise your spending limit",
    "console.x.ai",
    // OpenRouter's 402 ("Insufficient credits ... https://openrouter.ai/settings/credits").
    "insufficient credits",
    // DeepSeek's 402.
    "insufficient balance",
    // OpenRouter: the shared-pool 429 ("add your own key to accumulate your rate limits:
    // https://openrouter.ai/settings/integrations"), its key, credits and BYOK pages.
    "add your own key",
    "openrouter.ai/settings",
    // OpenAI's organization rate limit and billing pages (the rate limit also names Beyond's org id).
    "platform.openai.com/account",
    // Anthropic's organization rate limit: "contact sales" (it also names Beyond's org id).
    "anthropic.com/contact-sales",
    // Groq: the organization rate limit's "Upgrade to Dev Tier" billing link.
    "console.groq.com/settings",
];

/// Anthropic's out-of-credit answer, a `400` `invalid_request_error`, compared whole: its type is
/// the one every request mistake carries, so only the provider's exact sentence tells it apart. A
/// prefix would not do: Anthropic's "`<field>`: Extra inputs are not permitted" starts with a name
/// the tenant chose.
const ANTHROPIC_NO_CREDIT: &str = "Your credit balance is too low to access the Anthropic API. \
     Please go to Plans & Billing to upgrade or purchase credits.";

/// xAI's spent-credits `403` (`permission_denied`, the code every permission error carries) is
/// "Your team `<id>` has either used all ...": the team id is xAI's, and everything around it is
/// compared whole.
const XAI_NO_CREDIT: (&str, &str) = (
    "Your team ",
    " has either used all available credits or reached its monthly spending limit. To continue \
     making API requests, please purchase more credits or raise your spending limit.",
);

/// OpenRouter's `402` for one request larger than the credit left can pay for ("This request
/// requires more credits, or fewer max_tokens. You requested up to N tokens, but can only afford
/// M ..."): the tenant chose that size, and a smaller request is still served, so it does not cool
/// the key. Anchored at the start, which is OpenRouter's own text.
const OPENROUTER_TOO_COSTLY: &str = "This request requires more credits";

/// OpenRouter error metadata that describes Beyond's account with it, not the error.
const ACCOUNT_METADATA: &[&str] = &["is_byok", "limit_source", "remedy_hint"];

/// A rewritten error body.
#[derive(Debug, PartialEq, Eq)]
pub struct Neutralized {
    pub body: Vec<u8>,
    /// The provider said the account is out of credit or quota: cool the key (D180). Read from
    /// structured fields only ([`unfunded`]), never from a phrase.
    pub unfunded: bool,
}

/// `body` (an error with this `status`) with every provider-account remedy rewritten, or `None`
/// when it has none (or is not a JSON object), so the caller relays the original bytes.
///
/// In the error object (`error` when it is an object, else the root: Bedrock's `{"message"}`,
/// xAI's `{"code", "error": "<string>"}`): when the provider said the account is out of credit
/// ([`unfunded`]), the `message` or string `error` becomes [`UNAVAILABLE`]. Otherwise, on an
/// account status ([`account_status`]: a 401, 402, 403 or 429), one carrying a [`REMEDIES`] phrase
/// becomes [`TOO_LARGE`] on a 429 that says the request alone is over the limit, [`RATE_LIMITED`]
/// on any other 429, and [`UNAVAILABLE`] on the rest. On any other status (a 400 or 404 about the
/// request, which may quote it) the message is left as the provider wrote it. OpenRouter's
/// `metadata.raw` is removed when it is rewritten for the same reasons (the translation quotes it
/// after the message), and `is_byok`, `limit_source` and `remedy_hint` are always removed.
/// Everything else — `type`, `code`, `param`, `metadata.provider_name` — is kept.
pub fn neutralize(body: &[u8], status: u16) -> Option<Neutralized> {
    let mut v: Value = serde_json::from_slice(body).ok()?;
    let root = v.as_object_mut()?;
    let unfunded = match root.get_mut("error") {
        Some(Value::Object(err)) => scrub(err, status),
        _ => scrub(root, status),
    }?;
    Some(Neutralized {
        body: serde_json::to_vec(&v).ok()?,
        unfunded,
    })
}

/// Whether a provider error says the account behind the key is out of credit or quota, read from
/// what the provider sets and the request cannot (D200):
///
/// - a `402` (Payment Required: OpenRouter's insufficient credits, DeepSeek's insufficient balance,
///   Anthropic's `billing_error`), except OpenRouter's one-request-too-costly answer
///   ([`OPENROUTER_TOO_COSTLY`]);
/// - OpenAI's `insufficient_quota` `code` or `type`, and Anthropic's `billing_error` `type`;
/// - Anthropic's credit-balance `400` and xAI's spent-credits `403`, whose message is compared to
///   the provider's whole sentence ([`ANTHROPIC_NO_CREDIT`], [`XAI_NO_CREDIT`]).
///
/// Never a phrase found somewhere in a message: providers quote the request in their errors (a
/// model name, an unknown parameter, an extra field), so a tenant could otherwise cool Beyond's
/// key for every tenant, for free and at will.
fn unfunded(err: &Map<String, Value>, status: u16) -> bool {
    let str_of = |k: &str| err.get(k).and_then(Value::as_str);
    let message = str_of("message").or_else(|| str_of("error")).unwrap_or("");
    if status == 402 {
        return !message.starts_with(OPENROUTER_TOO_COSTLY);
    }
    if [str_of("code"), str_of("type")]
        .iter()
        .any(|v| matches!(v, Some("insufficient_quota" | "billing_error")))
    {
        return true;
    }
    if message.trim_end() == ANTHROPIC_NO_CREDIT {
        return true;
    }
    let (head, tail) = XAI_NO_CREDIT;
    message
        .strip_prefix(head)
        .and_then(|rest| rest.find(' ').map(|i| rest.split_at(i)))
        .is_some_and(|(team, rest)| !team.is_empty() && rest.trim_end() == tail)
}

/// Whether `status` is one on which a provider talks about the account rather than the request: a
/// refused (401), unfunded (402) or forbidden (403) credential, or a rate limit (429). A remedy
/// phrase is rewritten only on these. A 400 or 404 is about the request and may quote it ("The
/// model `X` does not exist"), so a phrase there may be the tenant's own words (D200).
fn account_status(status: u16) -> bool {
    matches!(status, 401..=403 | 429)
}

/// Rewrite `err` in place. `None` when nothing changed, else whether it is out of credit.
fn scrub(err: &mut Map<String, Value>, status: u16) -> Option<bool> {
    let unfunded = unfunded(err, status);
    let account = unfunded || account_status(status);
    let mut changed = false;
    let mut remedy = false;
    let mut too_large = false;
    if account {
        for field in ["message", "error"] {
            if let Some(s) = err.get(field).and_then(Value::as_str) {
                let lower = s.to_ascii_lowercase();
                remedy |= has_remedy(&lower);
                too_large |= TOO_LARGE_PHRASES.iter().any(|p| lower.contains(p));
            }
        }
    }
    if let Some(meta) = err.get_mut("metadata").and_then(Value::as_object_mut) {
        for k in ACCOUNT_METADATA {
            changed |= meta.remove(*k).is_some();
        }
        if account && meta.get("raw").is_some_and(remedy_in) {
            remedy = true;
            meta.remove("raw");
            changed = true;
        }
    }
    if unfunded || remedy {
        let text = if unfunded || status != 429 {
            UNAVAILABLE
        } else if too_large {
            TOO_LARGE
        } else {
            RATE_LIMITED
        };
        let mut wrote = false;
        for field in ["message", "error"] {
            if let Some(s) = err.get_mut(field)
                && s.is_string()
            {
                *s = Value::from(text);
                wrote = true;
            }
        }
        if !wrote {
            err.insert("message".to_owned(), Value::from(text));
        }
        changed = true;
    }
    changed.then_some(unfunded)
}

/// Whether any string in `v` carries a remedy (OpenRouter's `raw` is a string, or the upstream's
/// own JSON).
fn remedy_in(v: &Value) -> bool {
    match v {
        Value::String(s) => has_remedy(&s.to_ascii_lowercase()),
        Value::Array(a) => a.iter().any(remedy_in),
        Value::Object(o) => o.values().any(remedy_in),
        _ => false,
    }
}

fn has_remedy(lower: &str) -> bool {
    REMEDIES.iter().any(|r| lower.contains(r))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rewritten(body: &str, status: u16) -> Option<(Value, bool)> {
        neutralize(body.as_bytes(), status)
            .map(|n| (serde_json::from_slice(&n.body).unwrap(), n.unfunded))
    }

    /// OpenRouter's shared-pool 429: a rate limit. The message and `raw` carry the remedy and the
    /// account metadata goes; the code and the provider's name stay.
    #[test]
    fn openrouters_shared_pool_remedy_is_a_rate_limit() {
        let (v, unfunded) = rewritten(
            r#"{"error":{"message":"m is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations","code":429,"metadata":{"raw":"Add Your Own Key","provider_name":"Mistral","is_byok":false,"limit_source":"upstream_provider_shared_pool","remedy_hint":"byok"}},"user_id":"u"}"#,
            429,
        )
        .unwrap();
        assert!(!unfunded);
        assert_eq!(v["error"]["message"], RATE_LIMITED);
        assert_eq!(v["error"]["code"], 429);
        assert_eq!(
            v["error"]["metadata"],
            serde_json::json!({"provider_name": "Mistral"})
        );
        assert_eq!(v["user_id"], "u");
    }

    /// Each provider's account remedy, in its own envelope and status: a rate limit reads as one,
    /// an out-of-credit answer (whatever its status) reads as the provider being unavailable and
    /// is reported unfunded.
    #[test]
    fn every_providers_account_remedy_is_neutralized() {
        for (status, body, field, unfunded) in [
            (
                429,
                r#"{"error":{"message":"You exceeded your current quota, please check your plan and billing details. For more information on this error, read the docs: https://platform.openai.com/docs/guides/error-codes/api-errors.","type":"insufficient_quota","param":null,"code":"insufficient_quota"}}"#,
                "/error/message",
                true,
            ),
            (
                429,
                r#"{"error":{"message":"Rate limit reached for gpt-4o in organization org-abc on tokens per min (TPM): Limit 30000, Used 29000, Requested 2000. Please try again in 2s. Visit https://platform.openai.com/account/rate-limits to learn more.","type":"tokens","param":null,"code":"rate_limit_exceeded"}}"#,
                "/error/message",
                false,
            ),
            (
                400,
                r#"{"type":"error","error":{"type":"invalid_request_error","message":"Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits."}}"#,
                "/error/message",
                true,
            ),
            (
                429,
                r#"{"type":"error","error":{"type":"rate_limit_error","message":"This request would exceed the rate limit for your organization (o-1) of 50,000 input tokens per minute. You may also contact sales at https://www.anthropic.com/contact-sales to discuss your options for a rate limit increase."}}"#,
                "/error/message",
                false,
            ),
            (
                403,
                r#"{"code":"permission_denied","error":"Your team t-1 has either used all available credits or reached its monthly spending limit. To continue making API requests, please purchase more credits or raise your spending limit."}"#,
                "/error",
                true,
            ),
            (
                402,
                r#"{"error":{"message":"Insufficient credits. Purchase more at https://openrouter.ai/settings/credits","code":402}}"#,
                "/error/message",
                true,
            ),
            (
                402,
                r#"{"error":{"message":"Insufficient Balance","type":"unknown_error","param":null,"code":"invalid_request_error"}}"#,
                "/error/message",
                true,
            ),
        ] {
            let (v, got) = rewritten(body, status).unwrap_or_else(|| panic!("kept: {body}"));
            let want = if unfunded { UNAVAILABLE } else { RATE_LIMITED };
            assert_eq!(v.pointer(field).unwrap(), want, "{body}");
            assert_eq!(got, unfunded, "{body}");
            let orig: Value = serde_json::from_str(body).unwrap();
            for k in ["/error/type", "/error/code", "/code", "/type"] {
                assert_eq!(v.pointer(k), orig.pointer(k), "{k} of {body}");
            }
        }
        // A remedy on a status that is not a rate limit never reads as one.
        let (v, unfunded) = rewritten(
            r#"{"error":{"message":"see https://openrouter.ai/settings/keys"}}"#,
            403,
        )
        .unwrap();
        assert_eq!(
            (v["error"]["message"].as_str(), unfunded),
            (Some(UNAVAILABLE), false)
        );
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
            for status in [400, 429] {
                assert!(
                    neutralize(body.as_bytes(), status).is_none(),
                    "rewrote {body}"
                );
            }
        }
    }

    /// A provider quoting the request: every out-of-credit phrase in a tenant's model name,
    /// unknown parameter or extra field. Never unfunded (that would cool the shared key, D200), and
    /// on a 400 or 404 never rewritten either. On an account status the phrase still reads as a
    /// remedy (the cosmetic rewrite), but it is not unfunded there either.
    #[test]
    fn an_echoed_phrase_is_never_unfunded() {
        for phrase in REMEDIES {
            for body in [
                format!(
                    r#"{{"error":{{"message":"The model `{phrase}` does not exist or you do not have access to it.","type":"invalid_request_error","param":null,"code":"model_not_found"}}}}"#
                ),
                format!(
                    r#"{{"error":{{"message":"Unrecognized request argument supplied: {phrase}","type":"invalid_request_error","param":null,"code":null}}}}"#
                ),
                format!(
                    r#"{{"type":"error","error":{{"type":"invalid_request_error","message":"{phrase}: Extra inputs are not permitted"}}}}"#
                ),
            ] {
                for status in [400, 404, 422] {
                    assert!(
                        neutralize(body.as_bytes(), status).is_none(),
                        "{status} {body}"
                    );
                }
                for status in [401, 403, 429] {
                    let (_, unfunded) = rewritten(&body, status).unwrap();
                    assert!(!unfunded, "{status} {body}");
                }
            }
        }
        // The anchored sentences, with tenant text around them.
        for (status, body) in [
            (
                400,
                r#"{"type":"error","error":{"type":"invalid_request_error","message":"Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits.: Extra inputs are not permitted"}}"#,
            ),
            (
                400,
                r#"{"type":"error","error":{"type":"invalid_request_error","message":"x: Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits."}}"#,
            ),
            (
                403,
                r#"{"code":"permission_denied","error":"Model Your team t-1 has either used all available credits or reached its monthly spending limit. To continue making API requests, please purchase more credits or raise your spending limit."}"#,
            ),
        ] {
            assert!(
                !rewritten(body, status).is_some_and(|(_, u)| u),
                "{status} {body}"
            );
        }
    }

    /// The structured signals: a 402 (but not OpenRouter's one-request-too-costly), Anthropic's
    /// `billing_error`, OpenAI's `insufficient_quota`.
    #[test]
    fn unfunded_is_read_from_structured_fields() {
        for (status, body, want) in [
            (
                402,
                r#"{"type":"error","error":{"type":"billing_error","message":"There's an issue with your billing."}}"#,
                true,
            ),
            (
                400,
                r#"{"type":"error","error":{"type":"billing_error","message":"x"}}"#,
                true,
            ),
            (
                402,
                r#"{"error":{"message":"This request requires more credits, or fewer max_tokens. You requested up to 64000 tokens, but can only afford 1234. To increase, visit https://openrouter.ai/settings/credits","code":402}}"#,
                false,
            ),
            (
                429,
                r#"{"error":{"message":"x","type":"insufficient_quota","code":"insufficient_quota"}}"#,
                true,
            ),
        ] {
            let (v, got) = rewritten(body, status).unwrap_or_else(|| panic!("kept: {body}"));
            assert_eq!(got, want, "{body}");
            assert_eq!(v["error"]["message"], UNAVAILABLE, "{body}");
        }
    }
}
