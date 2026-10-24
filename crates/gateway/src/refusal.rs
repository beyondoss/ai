//! Every request the gateway itself refuses, as one closed type.
//!
//! A [`Refusal`] answers everything the client is told: the status, the error `type` and
//! `message`, the exact JSON body, the `ai_rejections_total` label and the `Retry-After`. A
//! refusal written before any upstream (`proxy::AiProxy::refuse`) and one raised from a later hook
//! as pingora's error ([`Refusal::into_error`], answered by `fail_to_proxy`) read all of it from
//! here, so no refusal can be answered with one message and counted under another, or go uncounted.
//!
//! A refusal whose message is fixed has its body spelled out at compile time ([`Fixed`]): a flood
//! of bad keys or rate-limited requests is answered with `Bytes::from_static`, no allocation. One
//! whose message names something from the request (a model, a path, a field) builds its body with
//! `serde_json`, so caller text can never break out of the JSON.

use crate::deny::DenyReason;
use crate::metrics::Rejection;
use crate::ratelimit::Throttled;
use bytes::Bytes;
use std::borrow::Cow;

/// A refusal with a fixed message: its `type`, `message` and wire body, the body built from the
/// first two at compile time by [`fixed!`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fixed {
    pub kind: &'static str,
    pub message: &'static str,
    pub body: &'static str,
}

/// `{"error":{"message":…,"type":…}}` from literals, in the key order `serde_json` writes (sorted),
/// so a fixed body is byte-identical to the one it would build. A message holding a `"` passes its
/// JSON spelling third; the variant test checks every body parses back to its kind and message.
macro_rules! fixed {
    ($kind:literal, $message:literal) => {
        $crate::refusal::fixed!($kind, $message, $message)
    };
    ($kind:literal, $message:literal, $json:literal) => {
        $crate::refusal::Fixed {
            kind: $kind,
            message: $message,
            body: concat!(
                r#"{"error":{"message":""#,
                $json,
                r#"","type":""#,
                $kind,
                r#""}}"#
            ),
        }
    };
}
pub(crate) use fixed;

/// Why the gateway refused a request. One variant per distinct answer; a variant carries data only
/// where its message names something from the request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// `/{name}/…` names no provider, or a bare `/v1` request no default provider serves.
    UnknownProvider,
    /// BYO keys for two providers in two headers: neither can be chosen.
    AmbiguousCredential,
    /// No credential at all.
    MissingKey,
    /// A managed key that does not verify.
    InvalidKey,
    /// A `bai_v3` key whose signature verified but whose `exp` has passed (past the leeway).
    KeyExpired,
    /// Over a rate guardrail ([`crate::ratelimit`]).
    RateLimited(Throttled),
    /// A body past `MAX_REQUEST_BODY`: declared, read while choosing a row, or streamed.
    BodyTooLarge,
    /// The deny-set names the tenant or key.
    Denied(DenyReason),
    /// The allowance-set says the tenant or key is exhausted.
    QuotaExhausted,
    /// The allowance-set has not been read yet: fail-closed, and retryable.
    AllowanceUnavailable,
    /// A `/{provider}` route whose provider has no pool key.
    NoPoolKey,
    /// A catalog walk with no candidate this deployment can send to.
    NoCandidate,
    /// A BYO key on the managed-only model route.
    ByoOnModelRoute,
    /// A managed request asking to upgrade the connection (WebSocket).
    ProtocolUpgrade,
    /// A managed request outside the POST generation endpoints. `wrong_method`: the endpoint is
    /// allowed and the method is not (405), else the endpoint is not (404). `path` is clipped.
    ManagedEndpoint {
        method: http::Method,
        path: Box<str>,
        wrong_method: bool,
    },
    /// A managed Responses `background: true` request: generated after the request ends,
    /// where no usage tap can meter it.
    Background,
    /// The tenant already has `tenant_max_in_flight` requests open.
    TenantBusy,
    /// Holding this body would cross `max_buffered_body_bytes`.
    BodyMemory,
    /// The model names no catalog row (or none was sent). `header`: whether the route reads
    /// `x-beyond-model`, which the message then mentions.
    CatalogMiss {
        name: Option<Box<str>>,
        header: bool,
    },
    /// Two root `model` keys, refused while the request headers are still held.
    DuplicateModel,
    /// Two root `model` keys, refused once the request headers went upstream. Answered with the
    /// generic 400 the gateway gives an error with no message of its own there.
    DuplicateModelLate,
    /// An output limit that is not a whole number, on a row with a published cap.
    OutputLimit,
    /// The body asks for what the row's card does not accept (`route::refused_input`, fast mode).
    RefusedInput {
        model: &'static str,
        kind: &'static str,
    },
    /// A feature billed outside the usage a row can meter; the message is [`crate::unpriced`]'s.
    Unpriced(&'static str),
    /// Responses session state on a row whose walk has no Responses arm.
    SessionField {
        field: &'static str,
        model: &'static str,
    },
    /// A sub-resource (`count_tokens`, `input_tokens`, `compact`) no candidate of the row serves.
    NoSubResource {
        model: &'static str,
        provider: &'static str,
        path: Box<str>,
    },
    /// An inbound endpoint the row's wire is not translated from. `hint` names the right one.
    WrongWire {
        model: &'static str,
        hint: &'static str,
    },
    /// A managed Responses relay to a store with no `id_signing_keys` configured.
    IdSigningUnset,
    /// An id the client sent back that is not signed for its tenant, refused while the request
    /// headers are still held: the message names the field.
    ForeignId(crate::signed_id::Refusal),
    /// The same, refused once the request headers went upstream (a `/{provider}` body).
    ForeignIdLate,
    /// The provider's circuit breaker is open; `retry_after` is the seconds until it half-opens.
    CircuitOpen { retry_after: u64 },
    /// Translating this body would take more heap than the whole body budget.
    TranslateTooLarge,
    /// The request outlived `request_max_secs` before its response head.
    RequestDeadline,
}

impl Refusal {
    /// The kind, message and compile-time body of a refusal whose message is fixed.
    pub fn fixed(&self) -> Option<Fixed> {
        Some(match self {
            Self::UnknownProvider => fixed!("invalid_request_error", "unknown provider"),
            Self::AmbiguousCredential => fixed!(
                "invalid_request_error",
                "the request carries API keys for different providers (x-api-key and \
                 Authorization); send only the key for the provider you mean, or name it with \
                 /{provider}/..."
            ),
            Self::MissingKey => fixed!("authentication_error", "missing API key"),
            Self::InvalidKey => fixed!("authentication_error", "invalid API key"),
            Self::KeyExpired => fixed!("authentication_error", "key expired"),
            Self::RateLimited(_) => fixed!("rate_limit_error", "rate limit exceeded"),
            Self::BodyTooLarge => fixed!("invalid_request_error", "request body too large"),
            Self::Denied(_) => fixed!("access_denied", "tenant is over limit or suspended"),
            Self::QuotaExhausted => fixed!("insufficient_quota", "quota exhausted"),
            Self::AllowanceUnavailable => fixed!("api_error", "allowance unavailable"),
            Self::NoPoolKey | Self::NoCandidate => fixed!("api_error", "no provider key available"),
            Self::ByoOnModelRoute => {
                fixed!(
                    "invalid_request_error",
                    "model routing requires a managed key"
                )
            }
            Self::ProtocolUpgrade => fixed!(
                "invalid_request_error",
                "protocol upgrades (WebSocket) are not available with a managed key"
            ),
            Self::Background => fixed!(
                "invalid_request_error",
                "background responses are not available with a managed key: the provider \
                 generates them after the request ends, where they cannot be metered; send the \
                 request without \"background\": true",
                "background responses are not available with a managed key: the provider \
                 generates them after the request ends, where they cannot be metered; send the \
                 request without \\\"background\\\": true"
            ),
            Self::TenantBusy => fixed!("rate_limit_error", "too many concurrent requests"),
            Self::BodyMemory => fixed!("api_error", "too many large request bodies in flight"),
            Self::DuplicateModel => fixed!(
                "invalid_request_error",
                "the request body has more than one root \"model\" key; send exactly one",
                "the request body has more than one root \\\"model\\\" key; send exactly one"
            ),
            Self::DuplicateModelLate => fixed!("invalid_request_error", "bad request"),
            Self::OutputLimit => fixed!(
                "invalid_request_error",
                "an output limit must be a whole number of tokens"
            ),
            Self::IdSigningUnset => fixed!(
                "api_error",
                "managed Responses ids cannot be issued: the gateway has no id signing key \
                 configured"
            ),
            Self::ForeignIdLate => fixed!(
                "invalid_request_error",
                "an id in this request does not belong to this tenant: send back only ids from \
                 responses this gateway returned to the same tenant"
            ),
            Self::CircuitOpen { .. } => fixed!("api_error", "provider temporarily unavailable"),
            Self::TranslateTooLarge => fixed!(
                "invalid_request_error",
                "request body is too large to translate onto this model's API within the \
                 gateway's memory budget; send it on the model's own API, or send less"
            ),
            Self::RequestDeadline => {
                fixed!(
                    "api_error",
                    "request exceeded the gateway's maximum duration"
                )
            }
            Self::ManagedEndpoint { .. }
            | Self::CatalogMiss { .. }
            | Self::RefusedInput { .. }
            | Self::Unpriced(_)
            | Self::SessionField { .. }
            | Self::NoSubResource { .. }
            | Self::WrongWire { .. }
            | Self::ForeignId(_) => return None,
        })
    }

    pub fn status(&self) -> u16 {
        match self {
            Self::MissingKey | Self::InvalidKey | Self::KeyExpired => 401,
            Self::QuotaExhausted => 402,
            Self::Denied(reason) => reason.http_status(),
            Self::UnknownProvider | Self::CatalogMiss { .. } => 404,
            Self::ManagedEndpoint {
                wrong_method: true, ..
            } => 405,
            Self::ManagedEndpoint {
                wrong_method: false,
                ..
            } => 404,
            Self::BodyTooLarge | Self::TranslateTooLarge => 413,
            Self::RateLimited(_) | Self::TenantBusy => 429,
            Self::AllowanceUnavailable
            | Self::NoPoolKey
            | Self::NoCandidate
            | Self::BodyMemory
            | Self::IdSigningUnset
            | Self::CircuitOpen { .. } => 503,
            Self::RequestDeadline => 504,
            Self::AmbiguousCredential
            | Self::ByoOnModelRoute
            | Self::ProtocolUpgrade
            | Self::Background
            | Self::DuplicateModel
            | Self::DuplicateModelLate
            | Self::OutputLimit
            | Self::RefusedInput { .. }
            | Self::Unpriced(_)
            | Self::SessionField { .. }
            | Self::NoSubResource { .. }
            | Self::WrongWire { .. }
            | Self::ForeignId(_)
            | Self::ForeignIdLate => 400,
        }
    }

    /// The error `type` in the body and the `request rejected` log line. Every variant is named,
    /// with no default, so a new one does not compile until its type is chosen; a fixed one's must
    /// be the type its compile-time body spells (the variant test checks each).
    pub fn kind(&self) -> &'static str {
        match self {
            Self::MissingKey | Self::InvalidKey | Self::KeyExpired => "authentication_error",
            Self::RateLimited(_) | Self::TenantBusy => "rate_limit_error",
            Self::Denied(_) => "access_denied",
            Self::QuotaExhausted => "insufficient_quota",
            Self::AllowanceUnavailable
            | Self::NoPoolKey
            | Self::NoCandidate
            | Self::BodyMemory
            | Self::IdSigningUnset
            | Self::CircuitOpen { .. }
            | Self::RequestDeadline => "api_error",
            Self::UnknownProvider
            | Self::AmbiguousCredential
            | Self::BodyTooLarge
            | Self::ByoOnModelRoute
            | Self::ProtocolUpgrade
            | Self::ManagedEndpoint { .. }
            | Self::Background
            | Self::CatalogMiss { .. }
            | Self::DuplicateModel
            | Self::DuplicateModelLate
            | Self::OutputLimit
            | Self::RefusedInput { .. }
            | Self::Unpriced(_)
            | Self::SessionField { .. }
            | Self::NoSubResource { .. }
            | Self::WrongWire { .. }
            | Self::ForeignId(_)
            | Self::ForeignIdLate
            | Self::TranslateTooLarge => "invalid_request_error",
        }
    }

    pub fn message(&self) -> Cow<'static, str> {
        Cow::Owned(match self {
            Self::ManagedEndpoint { method, path, .. } => format!(
                "{method} {path} is not available with a managed key; managed keys reach POST \
                 generation endpoints only (chat/completions, messages, responses, embeddings, \
                 count_tokens)"
            ),
            Self::CatalogMiss { name, header } => catalog_miss_message(name.as_deref(), *header),
            Self::RefusedInput { model, kind } => format!("{model} does not accept {kind}"),
            Self::Unpriced(why) => return Cow::Borrowed(why),
            Self::SessionField { field, model } => {
                crate::translate::session_field_refusal(field, model)
            }
            Self::NoSubResource {
                model,
                provider,
                path,
            } => format!("{model} has no {provider} upstream for {path}"),
            Self::WrongWire { model, hint } => format!("{model} is {hint}"),
            Self::ForeignId(refusal) => refusal.to_string(),
            // `fixed` is exhaustive: every variant not named above has a fixed message.
            fixed => return Cow::Borrowed(fixed.fixed().map_or("", |f| f.message)),
        })
    }

    /// The `ai_rejections_total` label it is counted under.
    pub fn metric(&self) -> Rejection {
        match self {
            Self::UnknownProvider => Rejection::UnknownProvider,
            Self::AmbiguousCredential => Rejection::AmbiguousCredential,
            Self::MissingKey => Rejection::MissingKey,
            Self::InvalidKey => Rejection::Auth,
            Self::KeyExpired => Rejection::KeyExpired,
            Self::RateLimited(throttled) => (*throttled).into(),
            Self::BodyTooLarge => Rejection::BodyTooLarge,
            // Distinct per reason: an `Unknown` is a control plane ahead of this build, not fraud.
            Self::Denied(DenyReason::Spend) => Rejection::DenySpend,
            Self::Denied(DenyReason::Fraud) => Rejection::DenyFraud,
            Self::Denied(DenyReason::Unknown) => Rejection::DenyUnknown,
            Self::QuotaExhausted => Rejection::Quota,
            Self::AllowanceUnavailable => Rejection::AllowanceUnavailable,
            Self::NoPoolKey => Rejection::NoPoolKey,
            Self::NoCandidate => Rejection::NoCandidate,
            Self::ByoOnModelRoute => Rejection::ByoOnModelRoute,
            Self::ProtocolUpgrade | Self::ManagedEndpoint { .. } | Self::Background => {
                Rejection::ManagedEndpoint
            }
            Self::TenantBusy => Rejection::TenantConcurrency,
            Self::BodyMemory => Rejection::BodyMemory,
            Self::CatalogMiss { .. } => Rejection::UnknownModel,
            Self::DuplicateModel | Self::DuplicateModelLate => Rejection::DuplicateModel,
            Self::OutputLimit => Rejection::OutputLimit,
            Self::RefusedInput { .. } => Rejection::Modality,
            Self::Unpriced(_) => Rejection::UnpricedFeature,
            Self::SessionField { .. } | Self::NoSubResource { .. } | Self::WrongWire { .. } => {
                Rejection::WireMismatch
            }
            Self::IdSigningUnset => Rejection::IdSigningUnset,
            Self::ForeignId(_) | Self::ForeignIdLate => Rejection::ForeignId,
            Self::CircuitOpen { .. } => Rejection::CircuitOpen,
            Self::TranslateTooLarge => Rejection::TranslateTooLarge,
            Self::RequestDeadline => Rejection::RequestDeadline,
        }
    }

    /// The `Retry-After` (seconds) a refusal written before any upstream carries: every 429 and
    /// 503 has one ([`default_retry_after`]), the open breaker's is the time until it half-opens.
    ///
    /// One of the two sources of a gateway `Retry-After`: this one answers a refusal in
    /// `request_filter` (`proxy::write_refusal`). A 503 raised past `request_filter` is answered by
    /// `fail_to_proxy`, which sets its own: the soonest open breaker of the walk, else 1.
    pub fn retry_after(&self) -> Option<u64> {
        match self {
            Self::CircuitOpen { retry_after } => Some(*retry_after),
            // A deployment with no signing key: no retry fixes that, so none is invited.
            Self::IdSigningUnset => None,
            r => r
                .fixed()
                .and_then(|f| default_retry_after(r.status(), f.message)),
        }
    }

    /// The JSON error body: a static one for a fixed message, else built with `serde_json`.
    pub fn body(&self) -> Bytes {
        match self.fixed() {
            Some(f) => Bytes::from_static(f.body.as_bytes()),
            None => json_body(self.kind(), &self.message()),
        }
    }

    /// This refusal as pingora's error, for a hook that cannot write the answer itself. The type
    /// is a `CustomCode` (the metric label and the status), so every check that treats a
    /// `CustomCode` or `HTTPStatus` as the gateway's own decision still does; the refusal itself
    /// rides as the cause and is read back with [`Refusal::of`].
    pub fn into_error(self) -> Box<pingora_core::Error> {
        pingora_core::Error::create(
            pingora_core::ErrorType::CustomCode(self.metric().label(), self.status()),
            pingora_core::ErrorSource::Unset,
            None,
            Some(Box::new(self)),
        )
    }

    /// The refusal `e` answers with: its cause, however much context pingora wrapped it in, while
    /// `e` still has the refusal's type. A wrapper that gave it another type is answered as that
    /// type, as any error is.
    pub fn of(e: &pingora_core::Error) -> Option<&Refusal> {
        let refusal = e.root_cause().downcast_ref::<Refusal>()?;
        let own = pingora_core::ErrorType::CustomCode(refusal.metric().label(), refusal.status());
        (e.etype() == &own).then_some(refusal)
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message())
    }
}

impl std::error::Error for Refusal {}

/// The `Retry-After` (seconds) on a gateway-made 429 or 503, so a stock SDK backs off rather than
/// hammering or giving up. A rate limit or a tenant's concurrency cap frees up within a second; an
/// allowance-set not yet read takes a few; anything else gets 1.
pub fn default_retry_after(status: u16, msg: &str) -> Option<u64> {
    match status {
        503 if msg == "allowance unavailable" => Some(5),
        429 | 503 => Some(1),
        _ => None,
    }
}

/// `{"error":{"message":msg,"type":kind}}` built with `serde_json`, which escapes caller text.
pub fn json_body(kind: &str, msg: &str) -> Bytes {
    Bytes::from(serde_json::json!({ "error": { "type": kind, "message": msg } }).to_string())
}

/// The catalog-miss message, for the catalog walk and a managed `/{provider}` body alike:
/// the model by name (clipped), or that none was sent, and where the models a managed key can use
/// are listed. `header`: whether the route reads `x-beyond-model` (a catalog walk does,
/// `/{provider}` does not).
pub fn catalog_miss_message(name: Option<&str>, header: bool) -> String {
    match name.map(str::trim).filter(|s| !s.is_empty()) {
        None => format!(
            "missing model: set the JSON body's model{} to a catalog name; GET /v1/models lists \
             them",
            if header { " (or x-beyond-model)" } else { "" }
        ),
        Some(n) => format!(
            "model \"{}\" is not in the catalog; a managed key runs catalog models only (GET \
             /v1/models lists them)",
            clip_catalog_name(n)
        ),
    }
}

/// Cap a caller-supplied name before echoing it in an error. The peek is 64 KiB; we do not want
/// that in a JSON error body.
pub fn clip_catalog_name(name: &str) -> &str {
    &name[..name.floor_char_boundary(128)]
}

/// One of every variant, each data-carrying one with a value its message names. The `match` has no
/// wildcard, so a new variant does not compile until it is listed here, and `every_variant_is_listed`
/// fails until it has a sample.
#[cfg(test)]
pub(crate) fn every() -> Vec<Refusal> {
    use crate::deny::DenyReason as D;
    let all = vec![
        Refusal::UnknownProvider,
        Refusal::AmbiguousCredential,
        Refusal::MissingKey,
        Refusal::InvalidKey,
        Refusal::KeyExpired,
        Refusal::RateLimited(Throttled::PerCredential),
        Refusal::RateLimited(Throttled::ByoGlobal),
        Refusal::BodyTooLarge,
        Refusal::Denied(D::Spend),
        Refusal::Denied(D::Fraud),
        Refusal::Denied(D::Unknown),
        Refusal::QuotaExhausted,
        Refusal::AllowanceUnavailable,
        Refusal::NoPoolKey,
        Refusal::NoCandidate,
        Refusal::ByoOnModelRoute,
        Refusal::ProtocolUpgrade,
        Refusal::ManagedEndpoint {
            method: http::Method::GET,
            path: "/openai/v1/files".into(),
            wrong_method: false,
        },
        Refusal::ManagedEndpoint {
            method: http::Method::GET,
            path: "/v1/chat/completions".into(),
            wrong_method: true,
        },
        Refusal::Background,
        Refusal::TenantBusy,
        Refusal::BodyMemory,
        Refusal::CatalogMiss {
            name: Some("gpt-9".into()),
            header: true,
        },
        Refusal::CatalogMiss {
            name: None,
            header: true,
        },
        Refusal::CatalogMiss {
            name: Some(" ".into()),
            header: false,
        },
        Refusal::DuplicateModel,
        Refusal::DuplicateModelLate,
        Refusal::OutputLimit,
        Refusal::RefusedInput {
            model: "deepseek-v4-pro",
            kind: "image input",
        },
        Refusal::Unpriced(crate::unpriced::CONTAINERS),
        Refusal::SessionField {
            field: "previous_response_id",
            model: "grok-4.3",
        },
        Refusal::NoSubResource {
            model: "gpt-5",
            provider: "Anthropic",
            path: "/v1/messages/count_tokens".into(),
        },
        Refusal::WrongWire {
            model: "claude-opus-4-8",
            hint: "Anthropic Messages; POST /v1/messages",
        },
        Refusal::IdSigningUnset,
        Refusal::ForeignId(crate::signed_id::Refusal {
            field: "previous_response_id".into(),
        }),
        Refusal::ForeignIdLate,
        Refusal::CircuitOpen { retry_after: 17 },
        Refusal::TranslateTooLarge,
        Refusal::RequestDeadline,
    ];
    for r in &all {
        // Exhaustive on purpose: see the doc above.
        match r {
            Refusal::UnknownProvider
            | Refusal::AmbiguousCredential
            | Refusal::MissingKey
            | Refusal::InvalidKey
            | Refusal::KeyExpired
            | Refusal::RateLimited(_)
            | Refusal::BodyTooLarge
            | Refusal::Denied(_)
            | Refusal::QuotaExhausted
            | Refusal::AllowanceUnavailable
            | Refusal::NoPoolKey
            | Refusal::NoCandidate
            | Refusal::ByoOnModelRoute
            | Refusal::ProtocolUpgrade
            | Refusal::ManagedEndpoint { .. }
            | Refusal::Background
            | Refusal::TenantBusy
            | Refusal::BodyMemory
            | Refusal::CatalogMiss { .. }
            | Refusal::DuplicateModel
            | Refusal::DuplicateModelLate
            | Refusal::OutputLimit
            | Refusal::RefusedInput { .. }
            | Refusal::Unpriced(_)
            | Refusal::SessionField { .. }
            | Refusal::NoSubResource { .. }
            | Refusal::WrongWire { .. }
            | Refusal::IdSigningUnset
            | Refusal::ForeignId(_)
            | Refusal::ForeignIdLate
            | Refusal::CircuitOpen { .. }
            | Refusal::TranslateTooLarge
            | Refusal::RequestDeadline => {}
        }
    }
    all
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// How many variants `Refusal` has: the exhaustive `match` in [`every`] is the list.
    const VARIANTS: usize = 33;

    #[test]
    fn every_variant_is_listed() {
        let seen: HashSet<_> = every().iter().map(std::mem::discriminant).collect();
        assert_eq!(seen.len(), VARIANTS, "a variant has no sample in `every`");
    }

    /// `(status, metric label, body, Retry-After)` each refusal answers with, spelled out as the
    /// client sees them: literals written apart from the code, so a change to any refusal's wire
    /// fails here rather than passing with the code that changed it.
    fn before(r: &Refusal) -> (u16, &'static str, &'static str, Option<u64>) {
        use crate::deny::DenyReason as D;
        match r {
            Refusal::UnknownProvider => (
                404,
                "unknown_provider",
                r#"{"error":{"message":"unknown provider","type":"invalid_request_error"}}"#,
                None,
            ),
            Refusal::AmbiguousCredential => (
                400,
                "ambiguous_credential",
                r#"{"error":{"message":"the request carries API keys for different providers (x-api-key and Authorization); send only the key for the provider you mean, or name it with /{provider}/...","type":"invalid_request_error"}}"#,
                None,
            ),
            Refusal::MissingKey => (
                401,
                "missing_key",
                r#"{"error":{"message":"missing API key","type":"authentication_error"}}"#,
                None,
            ),
            Refusal::InvalidKey => (
                401,
                "auth",
                r#"{"error":{"message":"invalid API key","type":"authentication_error"}}"#,
                None,
            ),
            Refusal::KeyExpired => (
                401,
                "key_expired",
                r#"{"error":{"message":"key expired","type":"authentication_error"}}"#,
                None,
            ),
            Refusal::RateLimited(Throttled::PerCredential) => (
                429,
                "rate_limit",
                r#"{"error":{"message":"rate limit exceeded","type":"rate_limit_error"}}"#,
                Some(1),
            ),
            Refusal::RateLimited(Throttled::ByoGlobal) => (
                429,
                "rate_limit_byo_global",
                r#"{"error":{"message":"rate limit exceeded","type":"rate_limit_error"}}"#,
                Some(1),
            ),
            Refusal::BodyTooLarge => (
                413,
                "body_too_large",
                r#"{"error":{"message":"request body too large","type":"invalid_request_error"}}"#,
                None,
            ),
            Refusal::Denied(D::Spend) => (
                402,
                "deny_spend",
                r#"{"error":{"message":"tenant is over limit or suspended","type":"access_denied"}}"#,
                None,
            ),
            Refusal::Denied(D::Fraud) => (
                403,
                "deny_fraud",
                r#"{"error":{"message":"tenant is over limit or suspended","type":"access_denied"}}"#,
                None,
            ),
            Refusal::Denied(D::Unknown) => (
                403,
                "deny_unknown",
                r#"{"error":{"message":"tenant is over limit or suspended","type":"access_denied"}}"#,
                None,
            ),
            Refusal::QuotaExhausted => (
                402,
                "quota",
                r#"{"error":{"message":"quota exhausted","type":"insufficient_quota"}}"#,
                None,
            ),
            Refusal::AllowanceUnavailable => (
                503,
                "allowance_unavailable",
                r#"{"error":{"message":"allowance unavailable","type":"api_error"}}"#,
                Some(5),
            ),
            Refusal::NoPoolKey => (
                503,
                "no_pool_key",
                r#"{"error":{"message":"no provider key available","type":"api_error"}}"#,
                Some(1),
            ),
            Refusal::NoCandidate => (
                503,
                "no_candidate",
                r#"{"error":{"message":"no provider key available","type":"api_error"}}"#,
                Some(1),
            ),
            Refusal::ByoOnModelRoute => (
                400,
                "byo_on_model_route",
                r#"{"error":{"message":"model routing requires a managed key","type":"invalid_request_error"}}"#,
                None,
            ),
            Refusal::ProtocolUpgrade => (
                400,
                "managed_endpoint",
                r#"{"error":{"message":"protocol upgrades (WebSocket) are not available with a managed key","type":"invalid_request_error"}}"#,
                None,
            ),
            Refusal::ManagedEndpoint {
                wrong_method: false,
                ..
            } => (
                404,
                "managed_endpoint",
                r#"{"error":{"message":"GET /openai/v1/files is not available with a managed key; managed keys reach POST generation endpoints only (chat/completions, messages, responses, embeddings, count_tokens)","type":"invalid_request_error"}}"#,
                None,
            ),
            Refusal::ManagedEndpoint {
                wrong_method: true, ..
            } => (
                405,
                "managed_endpoint",
                r#"{"error":{"message":"GET /v1/chat/completions is not available with a managed key; managed keys reach POST generation endpoints only (chat/completions, messages, responses, embeddings, count_tokens)","type":"invalid_request_error"}}"#,
                None,
            ),
            Refusal::Background => (
                400,
                "managed_endpoint",
                r#"{"error":{"message":"background responses are not available with a managed key: the provider generates them after the request ends, where they cannot be metered; send the request without \"background\": true","type":"invalid_request_error"}}"#,
                None,
            ),
            Refusal::TenantBusy => (
                429,
                "tenant_concurrency",
                r#"{"error":{"message":"too many concurrent requests","type":"rate_limit_error"}}"#,
                Some(1),
            ),
            Refusal::BodyMemory => (
                503,
                "body_memory",
                r#"{"error":{"message":"too many large request bodies in flight","type":"api_error"}}"#,
                Some(1),
            ),
            Refusal::CatalogMiss { name: Some(n), .. } if &**n == "gpt-9" => (
                404,
                "unknown_model",
                r#"{"error":{"message":"model \"gpt-9\" is not in the catalog; a managed key runs catalog models only (GET /v1/models lists them)","type":"invalid_request_error"}}"#,
                None,
            ),
            Refusal::CatalogMiss { header: true, .. } => (
                404,
                "unknown_model",
                r#"{"error":{"message":"missing model: set the JSON body's model (or x-beyond-model) to a catalog name; GET /v1/models lists them","type":"invalid_request_error"}}"#,
                None,
            ),
            Refusal::CatalogMiss { header: false, .. } => (
                404,
                "unknown_model",
                r#"{"error":{"message":"missing model: set the JSON body's model to a catalog name; GET /v1/models lists them","type":"invalid_request_error"}}"#,
                None,
            ),
            Refusal::DuplicateModel => (
                400,
                "duplicate_model",
                r#"{"error":{"message":"the request body has more than one root \"model\" key; send exactly one","type":"invalid_request_error"}}"#,
                None,
            ),
            Refusal::DuplicateModelLate => (
                400,
                "duplicate_model",
                r#"{"error":{"message":"bad request","type":"invalid_request_error"}}"#,
                None,
            ),
            Refusal::OutputLimit => (
                400,
                "output_limit",
                r#"{"error":{"message":"an output limit must be a whole number of tokens","type":"invalid_request_error"}}"#,
                None,
            ),
            Refusal::RefusedInput { .. } => (
                400,
                "modality",
                r#"{"error":{"message":"deepseek-v4-pro does not accept image input","type":"invalid_request_error"}}"#,
                None,
            ),
            Refusal::Unpriced(_) => (
                400,
                "unpriced_feature",
                r#"{"error":{"message":"a managed key cannot use OpenAI's code_interpreter or hosted shell tool: container time is billed outside the usage the gateway meters; use your own provider key","type":"invalid_request_error"}}"#,
                None,
            ),
            Refusal::SessionField { .. } => (
                400,
                "wire_mismatch",
                r#"{"error":{"message":"previous_response_id cannot be honored for grok-4.3 (no Responses upstream)","type":"invalid_request_error"}}"#,
                None,
            ),
            Refusal::NoSubResource { .. } => (
                400,
                "wire_mismatch",
                r#"{"error":{"message":"gpt-5 has no Anthropic upstream for /v1/messages/count_tokens","type":"invalid_request_error"}}"#,
                None,
            ),
            Refusal::WrongWire { .. } => (
                400,
                "wire_mismatch",
                r#"{"error":{"message":"claude-opus-4-8 is Anthropic Messages; POST /v1/messages","type":"invalid_request_error"}}"#,
                None,
            ),
            Refusal::IdSigningUnset => (
                503,
                "id_signing_unset",
                r#"{"error":{"message":"managed Responses ids cannot be issued: the gateway has no id signing key configured","type":"api_error"}}"#,
                None,
            ),
            Refusal::ForeignId(_) => (
                400,
                "foreign_id",
                r#"{"error":{"message":"previous_response_id does not belong to this tenant: send back only ids from responses this gateway returned to the same tenant (an id from another tenant, from the provider directly, or altered, is refused)","type":"invalid_request_error"}}"#,
                None,
            ),
            Refusal::ForeignIdLate => (
                400,
                "foreign_id",
                r#"{"error":{"message":"an id in this request does not belong to this tenant: send back only ids from responses this gateway returned to the same tenant","type":"invalid_request_error"}}"#,
                None,
            ),
            Refusal::CircuitOpen { .. } => (
                503,
                "circuit_open",
                r#"{"error":{"message":"provider temporarily unavailable","type":"api_error"}}"#,
                Some(17),
            ),
            Refusal::TranslateTooLarge => (
                413,
                "translate_too_large",
                r#"{"error":{"message":"request body is too large to translate onto this model's API within the gateway's memory budget; send it on the model's own API, or send less","type":"invalid_request_error"}}"#,
                None,
            ),
            Refusal::RequestDeadline => (
                504,
                "request_deadline",
                r#"{"error":{"message":"request exceeded the gateway's maximum duration","type":"api_error"}}"#,
                None,
            ),
        }
    }

    /// Each refusal's whole answer is the one spelled out for it, its error type included: every
    /// variant's `kind` is read against the type in its literal body, and a fixed one's against its
    /// compile-time body too.
    #[test]
    fn every_refusal_answers_exactly_as_spelled_out() {
        for r in every() {
            let (status, label, body, retry_after) = before(&r);
            let wire: serde_json::Value = serde_json::from_str(body).expect("a JSON body");
            assert_eq!(wire["error"]["type"], r.kind(), "{r:?}");
            if let Some(f) = r.fixed() {
                assert_eq!(f.kind, r.kind(), "{r:?}");
            }
            assert_eq!(r.status(), status, "{r:?}");
            assert_eq!(r.metric().label(), label, "{r:?}");
            assert_eq!(r.body(), Bytes::from_static(body.as_bytes()), "{r:?}");
            assert_eq!(r.retry_after(), retry_after, "{r:?}");
        }
    }

    /// Every body is JSON that says what the log line and the metric claim, and is byte-identical
    /// to what `serde_json` builds for its kind and message. A fixed one is the static itself: no
    /// allocation, on the path a flood of bad keys drives.
    #[test]
    fn every_body_parses_to_its_kind_and_message() {
        for r in every() {
            let body = r.body();
            let v: serde_json::Value =
                serde_json::from_slice(&body).unwrap_or_else(|e| panic!("{r:?}: not JSON ({e})"));
            assert_eq!(v["error"]["type"], r.kind(), "{r:?}");
            assert_eq!(v["error"]["message"], *r.message(), "{r:?}");
            assert!(!r.message().is_empty(), "{r:?}");
            assert_eq!(body, json_body(r.kind(), &r.message()), "{r:?}");
            assert_eq!(r.to_string(), r.message(), "{r:?}");
            if let Some(f) = r.fixed() {
                assert_eq!(body.as_ptr(), f.body.as_ptr(), "{r:?}: not the static body");
                assert!(matches!(r.message(), Cow::Borrowed(_)), "{r:?}");
            }
        }
    }

    /// Every label but one is some refusal's: the exception is a translated response cut short
    /// mid-relay, which ends a stream rather than refusing a request.
    #[test]
    fn every_rejection_label_is_a_refusals() {
        let counted: HashSet<&str> = every().iter().map(|r| r.metric().label()).collect();
        for reason in Rejection::ALL {
            assert_eq!(
                counted.contains(reason.label()),
                reason != Rejection::ResponseTooLarge,
                "{reason:?}"
            );
        }
    }

    /// Raised as pingora's error, a refusal is still the gateway's own decision to every check
    /// that reads the type, and is read back whole through the context pingora wraps it in.
    #[test]
    fn a_refusal_rides_pingoras_error_whole() {
        use pingora_core::ErrorType as T;
        for r in every() {
            let e = r.clone().into_error().into_down();
            assert_eq!(
                e.etype(),
                &T::CustomCode(r.metric().label(), r.status()),
                "{r:?}"
            );
            assert_eq!(Refusal::of(&e), Some(&r));
            let wrapped = e.more_context("Peer: x");
            assert_eq!(Refusal::of(&wrapped), Some(&r));
            let retyped = pingora_core::Error::because(T::InternalError, "x", wrapped);
            assert_eq!(Refusal::of(&retyped), None, "{r:?}");
        }
        assert_eq!(
            Refusal::of(&pingora_core::Error::new(T::CustomCode("x", 504))),
            None
        );
    }
}
