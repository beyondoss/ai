//! The credential a request presents and the headers a managed request may forward.

use super::*;

/// OpenRouter's dashboard-attribution headers (https://openrouter.ai/docs/quickstart): purely
/// cosmetic on OpenRouter's side (their own cost/usage categorization), no effect on the request
/// or response. Static — this doesn't need to be configurable, just present. Only OpenRouter is in
/// `known_providers()` among the providers pi attributes to (NVIDIA NIM, Cloudflare, Vercel AI
/// Gateway aren't routed providers here), so this is the one case worth porting. Header names match
/// pi's current OpenRouter-specific set (`packages/coding-agent/src/core/provider-attribution.ts`):
/// `HTTP-Referer`, `X-OpenRouter-Title` (NOT the generic `X-Title` pi used for the now-removed Vercel
/// AI Gateway route), and `X-OpenRouter-Categories`.
pub(super) const OPENROUTER_REFERER: &str = "https://beyond.dev";

pub(super) const OPENROUTER_TITLE: &str = "Beyond Gateway";

pub(super) const OPENROUTER_CATEGORY: &str = "cli-agent";

/// Header names carrying a plain static API key (no OAuth/signing), checked in order. Anthropic:
/// `x-api-key`. Azure OpenAI: `api-key`. Google Gemini: `x-goog-api-key`. `Authorization: Bearer`
/// (OpenAI and everyone else) is checked separately below since it needs prefix-stripping.
pub(super) const STATIC_KEY_HEADERS: [&str; 3] = ["x-api-key", "api-key", "x-goog-api-key"];

/// The client headers a managed request forwards to the provider. Every other client header is
/// dropped in `upstream_request_filter` before the gateway adds its own (pool key, `Host`,
/// `accept-encoding`, OpenRouter attribution, a translated walk's `anthropic-version` /
/// `anthropic-beta`).
///
/// An allowlist rather than a blocklist because the request rides on Beyond's pool key: a header
/// the gateway has never heard of is one it cannot vouch for. `openai-organization` and
/// `openai-project` would switch the org or project the pool key bills to (and 401 for an SDK user
/// with `OPENAI_ORG_ID` set), `cookie` / `proxy-authorization` / `x-goog-user-project` are someone
/// else's credentials or billing selectors, and SDK telemetry (`x-stainless-*`) is noise. Framing
/// headers stay so the body arrives intact, and `expect` stays so a client waiting on
/// `100 Continue` is answered by the provider rather than by its own timeout (unless the gateway
/// already answered it while reading the body, in which case `peek_body_model` removed it).
pub(super) const MANAGED_FORWARD_HEADERS: [&str; 8] = [
    "content-type",
    "content-length",
    "transfer-encoding",
    "expect",
    "accept",
    "user-agent",
    "anthropic-version",
    "anthropic-beta",
];

/// The beta that turned on fine-grained tool input streaming before each tool's
/// `eager_input_streaming` field replaced it; kept only for a candidate that takes the field.
pub(super) const FINE_GRAINED_TOOL_STREAMING_BETA: &str = "fine-grained-tool-streaming-2025-05-14";

/// `anthropic-beta` tokens a managed request may send on the pool key. Each changes how a request
/// is parsed or streamed, and none changes what Anthropic charges per token or runs server-side
/// tools the gateway does not meter. Anything else is dropped: `context-1m-*` switches on premium
/// long-context pricing, `mcp-client-*` / `code-execution-*` / `files-api-*` reach servers, sandboxes
/// and storage on Beyond's account, and `oauth-*` is meaningless beside a pool API key. The gateway's
/// own [`translate::THINKING_BINDING_BETA`] is listed too, so a Messages client on a binding model
/// may send it itself. Fast mode's beta is not here: it doubles the per-token price, and is kept
/// only on a request to direct Anthropic (see [`retain_managed_client_headers`]).
pub(super) const MANAGED_ANTHROPIC_BETAS: [&str; 8] = [
    "claude-code-20250219",
    "prompt-caching-2024-07-31",
    "interleaved-thinking-2025-05-14",
    FINE_GRAINED_TOOL_STREAMING_BETA,
    "context-management-2025-06-27",
    "token-efficient-tools-2025-02-19",
    "output-128k-2025-02-19",
    translate::THINKING_BINDING_BETA,
];

/// Drop every client header a managed request may not forward ([`MANAGED_FORWARD_HEADERS`]), and
/// every `anthropic-beta` token not in [`MANAGED_ANTHROPIC_BETAS`], plus
/// [`translate::FAST_MODE_BETA`] when `anthropic` (the attempt goes to direct Anthropic, the only
/// provider with fast mode). Fast mode is a product Beyond sells at its price: the row records the
/// speed asked and the speed served (`usage.speed`), so it is billed as served (D266). Without
/// `eager_tool_streaming` (the attempt's candidate does not take fine-grained tool streaming,
/// `providers::catalog::serves_eager_input_streaming`), [`FINE_GRAINED_TOOL_STREAMING_BETA`] is
/// dropped too, as `request_body_filter` drops each tool's `eager_input_streaming` (D272). The
/// header sweep allocates nothing (D92: it collected the names into a `Vec` on nearly every
/// managed request); only an `anthropic-beta` value with a token to drop is rebuilt.
pub(super) fn retain_managed_client_headers(
    req: &mut pingora::http::RequestHeader,
    anthropic: bool,
    eager_tool_streaming: bool,
) -> Result<()> {
    let allowed = |t: &str| {
        (MANAGED_ANTHROPIC_BETAS.contains(&t)
            && (eager_tool_streaming || t != FINE_GRAINED_TOOL_STREAMING_BETA))
            || (anthropic && t == translate::FAST_MODE_BETA)
    };
    remove_headers_where(req, |name| !MANAGED_FORWARD_HEADERS.contains(&name));
    let betas = req.headers.get_all("anthropic-beta");
    let mut values = betas.iter();
    let clean = match (values.next(), values.next()) {
        (None, _) => true,
        (Some(v), None) => v
            .to_str()
            .is_ok_and(|v| v.split(',').map(str::trim).all(allowed)),
        _ => false,
    };
    if !clean {
        let kept = betas
            .iter()
            .filter_map(|v| v.to_str().ok())
            .flat_map(|v| v.split(','))
            .map(str::trim)
            .filter(|t| allowed(t))
            .collect::<Vec<_>>()
            .join(",");
        req.remove_header("anthropic-beta");
        if !kept.is_empty() {
            req.insert_header("anthropic-beta", kept)?;
        }
    }
    Ok(())
}

/// Drop every `x-beyond-*` header from a request about to go upstream (D67). Allocates nothing.
pub(super) fn strip_beyond_headers(req: &mut pingora::http::RequestHeader) {
    remove_headers_where(req, |name| name.starts_with("x-beyond-"));
}

/// Remove every header whose (lower-case) name `drop` matches, allocating nothing (D92): names are
/// copied in batches onto the stack, since the map cannot be borrowed while it is edited. A name
/// past the stack slot's length (no real header) is cloned instead.
pub(super) fn remove_headers_where(
    req: &mut pingora::http::RequestHeader,
    drop: impl Fn(&str) -> bool,
) {
    const BATCH: usize = 16;
    loop {
        let mut names = ArrayVec::<ArrayString<64>, BATCH>::new();
        let mut long = None;
        let mut more = false;
        for k in req.headers.keys().map(http::HeaderName::as_str) {
            if !drop(k) {
                continue;
            }
            if names.is_full() {
                more = true;
                break;
            }
            match ArrayString::from(k) {
                Ok(name) => names.push(name),
                Err(_) => {
                    long = http::HeaderName::from_bytes(k.as_bytes()).ok();
                    more = true;
                    break;
                }
            }
        }
        for name in &names {
            req.remove_header(name.as_str());
        }
        if let Some(name) = long {
            req.remove_header(&name);
        }
        if !more {
            return;
        }
    }
}

/// `s` with its `%XX` escapes decoded, borrowed when it has none. A malformed escape is kept as
/// written. Used only to *recognize* a credential the way a provider would read it (Google decodes
/// `k%65y` as `key`), never to rebuild what is forwarded.
pub(super) fn percent_decoded(s: &str) -> Cow<'_, str> {
    if !s.contains('%') {
        return Cow::Borrowed(s);
    }
    let hex = |b: u8| char::from(b).to_digit(16);
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && let (Some(h), Some(l)) = (
                bytes.get(i + 1).copied().and_then(hex),
                bytes.get(i + 2).copied().and_then(hex),
            )
        {
            out.push(u8::try_from(h * 16 + l).unwrap_or(b'%'));
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    Cow::Owned(String::from_utf8_lossy(&out).into_owned())
}

/// Whether a raw query pair (`k=v`) names the `key` credential param, under any spelling a provider
/// decodes to `key` (`k%65y`).
pub(super) fn is_key_param(pair: &str) -> bool {
    let name = pair.split_once('=').map_or(pair, |(k, _)| k);
    name == "key" || percent_decoded(name) == "key"
}

/// Every `key` query param's raw value, in order (Gemini's `?key=` convention — the sole
/// query-param credential shape among recognized providers).
pub(super) fn key_params(query: &str) -> impl Iterator<Item = &str> {
    query
        .split('&')
        .filter(|p| is_key_param(p))
        .map(|p| p.split_once('=').map_or("", |(_, v)| v))
}

/// The token of one `Authorization` value: `Bearer <token>`, scheme matched case-insensitively and
/// any run of whitespace around the token tolerated (a lenient provider trims it). `None` for any
/// other scheme.
pub(super) fn bearer_token(v: &str) -> Option<&str> {
    let v = v.trim();
    let scheme = v.get(..6)?;
    let rest = &v[6..];
    (scheme.eq_ignore_ascii_case("bearer") && rest.starts_with(|c: char| c.is_ascii_whitespace()))
        .then(|| rest.trim_start())
}

/// The key a request presents, and whether it is managed.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Presented<'a> {
    /// The value to verify (managed) or to rate-guard as a BYO token. Borrowed from the request.
    pub(super) key: &'a str,
    /// Some credential location carries a virtual key (`bai_v1…`/`bai_v2…`). The request is then
    /// managed whatever the others hold: verified (fail-closed, 401) and every location stripped.
    pub(super) managed: bool,
}

/// Extract the presented key (virtual or BYO) from wherever the client's SDK puts it. Every
/// recognized shape is a **plain static key** (no OAuth/signing), so one neutral virtual key works
/// in any of them: Anthropic's `x-api-key`, Azure OpenAI's `api-key`, Google Gemini's
/// `x-goog-api-key` (header, falling back to the `?key=` query param — Gemini accepts either),
/// OpenAI's `Authorization: Bearer` (scheme matched case-insensitively). Borrowed from the request —
/// no per-request copy. Empty values count as absent.
///
/// **A managed key anywhere wins** (D29, D65). Every value of every location is read: each line of a
/// repeated header, each `key` param (including a percent-encoded name such as `k%65y`, and a
/// percent-encoded value), and a `Bearer` with extra whitespace. When any of them carries a virtual
/// key the request is managed, and `upstream_request_filter` / `strip_key_param` strip every
/// location. Reading only the first value of each let a junk first line, a second `?key=`, an
/// encoded name or a double space classify the request as BYO, which forwards it untouched — the
/// virtual key included. Otherwise the first location in the order above wins; the query param is
/// last since keys in a URL end up in proxy/access logs.
///
/// A value that is managed only once percent-decoded is returned as written: it fails verification
/// (401), which is the right answer for a key no SDK would encode.
pub(super) fn extract_virtual_key(req: &pingora::http::RequestHeader) -> Option<Presented<'_>> {
    let headers = STATIC_KEY_HEADERS
        .iter()
        .flat_map(|h| req.headers.get_all(*h))
        .filter_map(|v| v.to_str().ok())
        .map(|v| (v.trim(), false));
    // A scheme-less `Authorization: bai_v1…` is no credential a provider reads, but it is one the
    // gateway must not forward on a BYO request: it counts as managed, never as the BYO key.
    let auth = req
        .headers
        .get_all("authorization")
        .into_iter()
        .filter_map(|v| v.to_str().ok())
        .filter_map(|v| match bearer_token(v) {
            Some(t) => Some((t, false)),
            None => key::is_managed_prefix(v.trim()).then(|| (v.trim(), true)),
        });
    let query = req
        .uri
        .query()
        .into_iter()
        .flat_map(key_params)
        .map(|v| (v, key::is_managed_prefix(&percent_decoded(v))));
    let mut first = None;
    for (v, decoded_managed) in headers.chain(auth).chain(query) {
        if v.is_empty() {
            continue;
        }
        if decoded_managed || key::is_managed_prefix(v) {
            return Some(Presented {
                key: v,
                managed: true,
            });
        }
        first = first.or(Some(v));
    }
    first.map(|key| Presented {
        key,
        managed: false,
    })
}

/// `path_and_query` without its `key` query params, or `None` when it carries none. Other params
/// (Azure's `api-version`) keep their order. Managed requests only: a `?key=` there is the virtual
/// key, which must not reach the provider, while a BYO `?key=` is the caller's own Gemini key. Every
/// spelling a provider decodes to `key` is dropped (`k%65y`), and every repeat of it.
pub(super) fn strip_key_param(path_and_query: &str) -> Option<String> {
    let (path, query) = path_and_query.split_once('?')?;
    if !query.split('&').any(is_key_param) {
        return None;
    }
    let mut out = String::with_capacity(path_and_query.len());
    out.push_str(path);
    let mut sep = '?';
    for pair in query.split('&').filter(|p| !is_key_param(p)) {
        out.push(sep);
        out.push_str(pair);
        sep = '&';
    }
    Some(out)
}

/// Set OpenRouter's attribution headers when (and only when) the provider is OpenRouter
/// **and** the request is managed — every other provider, and every BYO request regardless of
/// provider, is untouched. Task #22 (pi-parity, Medium): pi gates this dashboard-attribution
/// behind a user-controllable telemetry setting (`isInstallTelemetryEnabled`,
/// `packages/coding-agent/src/core/provider-attribution.ts`); this gateway has no per-user
/// telemetry-opt-out setting to consult, but it already has an unambiguous, always-correct proxy
/// for "is this Beyond's own attribution to make, or someone else's traffic passing through us": a
/// BYO request carries the *caller's own* OpenRouter key, not Beyond's — attributing *their*
/// traffic to Beyond's dashboard app would misrepresent whose usage it is, the same harm the
/// telemetry opt-out exists to prevent. Managed traffic (Beyond's own pool key) is the only case
/// these headers describe accurately.
pub(super) fn apply_provider_attribution(
    upstream_request: &mut pingora::http::RequestHeader,
    provider: Option<ProviderId>,
    managed: bool,
) -> Result<()> {
    if provider == Some(ProviderId::OpenRouter) && managed {
        upstream_request.insert_header("HTTP-Referer", OPENROUTER_REFERER)?;
        upstream_request.insert_header("X-OpenRouter-Title", OPENROUTER_TITLE)?;
        upstream_request.insert_header("X-OpenRouter-Categories", OPENROUTER_CATEGORY)?;
    }
    Ok(())
}

/// Resolve the provider name for a request whose first path segment matched no known/config
/// provider: `Some(name)` for the bare-path default — `path` is boundary-checked against
/// [`route::DEFAULT_PREFIX`] (see [`route::is_default_prefix`]) so a lookalike like Google
/// Gemini's `/v1beta/…` doesn't qualify — with the dialect picking openai/anthropic
/// ([`dialect_for_path`]); `None` for anything else, which the caller turns into a 404 rather than
/// silently guessing a provider (Task #7, pi-parity).
///
/// `credential`: the provider the request's BYO credentials belong to
/// ([`byo_credential_dialect`]). It wins over the path, so an Anthropic SDK call to `/v1/files`
/// never reaches OpenAI carrying `sk-ant-…`, and an OpenAI key on `/v1/messages` never reaches
/// Anthropic. With no credential that says, the path picks.
pub(super) fn bare_default_provider_name(
    path: &str,
    credential: Option<Dialect>,
) -> Option<&'static str> {
    let dialect = credential.unwrap_or_else(|| dialect_for_path(path));
    route::is_default_prefix(path).then(|| route::dialect_default(dialect))
}

/// The provider a BYO key belongs to, from its shape: `sk-ant-…` is Anthropic's, any other `sk-…`
/// OpenAI's. `None` when the shape says nothing.
pub(super) fn byo_key_dialect(key: &str) -> Option<Dialect> {
    if key.starts_with("sk-ant-") {
        Some(Dialect::Anthropic)
    } else if key.starts_with("sk-") {
        Some(Dialect::OpenAi)
    } else {
        None
    }
}

/// The provider the BYO credentials on a bare `/v1` request belong to, read from every value the
/// gateway would forward (D82): each `x-api-key` line (Anthropic's header, so Anthropic unless the
/// key's shape says otherwise) and each `Authorization: Bearer` token (by its shape only; an
/// opaque token says nothing). Managed and empty values do not count. `Ok(None)`: nothing says.
/// `Err(())`: they name different providers; every one would be forwarded, so whichever provider
/// the request went to would receive the other's key.
pub(super) fn byo_credential_dialect(
    req: &pingora::http::RequestHeader,
) -> Result<Option<Dialect>, ()> {
    let byo = |v: &&str| !v.is_empty() && !key::is_managed_prefix(v);
    let api_keys = req
        .headers
        .get_all("x-api-key")
        .into_iter()
        .filter_map(|v| v.to_str().ok())
        .map(str::trim)
        .filter(byo)
        .map(|v| Some(byo_key_dialect(v).unwrap_or(Dialect::Anthropic)));
    let bearers = req
        .headers
        .get_all("authorization")
        .into_iter()
        .filter_map(|v| v.to_str().ok())
        .filter_map(bearer_token)
        .filter(byo)
        .map(byo_key_dialect);
    let mut vote = None;
    for d in api_keys.chain(bearers).flatten() {
        match vote {
            None => vote = Some(d),
            Some(v) if v == d => {}
            Some(_) => return Err(()),
        }
    }
    Ok(vote)
}

/// Add `beta` to the request's `anthropic-beta` value, unless it is already there.
pub(super) fn merge_anthropic_beta(
    req: &mut pingora::http::RequestHeader,
    beta: &'static str,
) -> Result<()> {
    let existing = req
        .headers
        .get("anthropic-beta")
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|v| !v.is_empty());
    match existing {
        Some(v) if v.split(',').any(|b| b.trim() == beta) => {}
        Some(v) => {
            let merged = format!("{v},{beta}");
            req.insert_header("anthropic-beta", merged)?;
        }
        None => req.insert_header("anthropic-beta", beta)?,
    }
    Ok(())
}
