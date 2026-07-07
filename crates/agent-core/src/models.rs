//! A minimal per-model capability table.
//!
//! The agent — like the gateway it sits behind — holds no model catalog: a model id is forwarded
//! verbatim and only its *wire dialect* is derived from the name (see [`dialect`]). But a handful of
//! wire-shaping decisions genuinely need per-model knowledge the request can't carry on its own:
//!
//! - which output-ceiling field a model expects (`max_tokens` vs `max_completion_tokens`),
//! - whether it honors the 1-hour prompt-cache TTL,
//! - what extended-thinking shape it accepts,
//! - a sane context-window default for compaction.
//!
//! This is the *smallest* table those decisions need, matched by id prefix. It is deliberately **not**
//! pricing (the gateway meters tokens; a downstream consumer prices them) and **not** a routing
//! registry (the gateway routes). When a model id is unknown the table returns conservative defaults
//! that keep every wire decision safe.
//!
//! [`dialect`]: crate::dialect

/// Which output-ceiling field a model's wire expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaxTokensField {
    /// `max_tokens` — Anthropic, and OpenAI chat-completions on non-reasoning models.
    MaxTokens,
    /// `max_completion_tokens` — OpenAI reasoning models (o-series, gpt-5) reject `max_tokens`.
    MaxCompletionTokens,
}

/// Which OpenAI-wire API surface a model speaks. Anthropic ids ignore this field entirely (they
/// always speak `Dialect::Anthropic`); it only disambiguates within the OpenAI-wire branch of
/// [`crate::dialect::Dialect::for_model`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiKind {
    /// `/v1/chat/completions` — every third-party OpenAI-compatible provider (OpenRouter, DeepSeek,
    /// Together, Cerebras, xAI, Groq, Fireworks, Mistral), and the fallback for any unrecognized id.
    ChatCompletions,
    /// `/v1/responses` — every native OpenAI model id (gpt-4/4.1/4o, gpt-5 family, o-series). Ported
    /// from pi's live catalogue: every entry under `packages/ai/src/providers/openai.models.ts`
    /// carries `api: "openai-responses"`, with no exceptions among current ids.
    Responses,
}

/// Which OpenAI Chat-Completions-wire reasoning/thinking toggle a model expects — mirrors pi's
/// `compat.thinkingFormat` tag (`packages/ai/src/api/openai-completions.ts`). Only consulted by
/// [`crate::dialect::openai::build_body`]; the Anthropic dialect has its own shape
/// ([`ThinkingShape`]) and the OpenAI Responses dialect always speaks native OpenAI (`Standard`).
/// Every native OpenAI id and every model with no vendor-specific quirk stays `Standard` — the bare
/// `reasoning_effort` string this dialect emitted before third-party coverage existed, unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAiReasoningFormat {
    /// Bare `reasoning_effort: "<level>"` (or `"none"` when [`ModelCaps::reasoning_disableable`] and
    /// no level is requested) — OpenAI's own shape, and pi's default for every provider it doesn't
    /// special-case (xAI, Groq, Cerebras, and any unrecognized id).
    Standard,
    /// DeepSeek — and Moonshot/Kimi, which pi tags with this identical shape: `thinking: {"type":
    /// "enabled"}` / `{"type": "disabled"}`, plus — only when [`ModelCaps::reasoning_effort`] is also
    /// `true` (DeepSeek; not Kimi, which has no effort vocabulary at all) — a sibling top-level
    /// `reasoning_effort` string.
    DeepSeek,
    /// Z.ai/GLM: `thinking: {"type": "enabled", "clear_thinking": false}` / `{"type": "disabled"}`,
    /// sent unconditionally (unlike `DeepSeek`, pi never gates this on a disableable check), plus —
    /// only when [`ModelCaps::reasoning_effort`] is also `true` (GLM-5.2+ only) — a sibling top-level
    /// `reasoning_effort` string.
    Zai,
    /// Together: a nested `reasoning: {"enabled": bool}`, sent unconditionally, plus — only when
    /// [`ModelCaps::reasoning_effort`] is also `true` — a sibling top-level `reasoning_effort` string.
    Together,
    /// OpenRouter — and the generic vendor/model-shaped fallback for an uncatalogued third-party id:
    /// a nested `reasoning: {"effort": "<level>"}`, or `{"effort": "none"}` when
    /// [`ModelCaps::reasoning_disableable`] and no level is requested.
    OpenRouter,
    /// Ant-Ling: a nested `reasoning: {"effort": "<level>"}` sent *only* when a level is requested and
    /// the model's per-level remap maps it to a string — unlike every other variant above, no explicit
    /// "off"/disabled signal is ever sent; the field is simply omitted otherwise (pi:
    /// `compat.thinkingFormat === "ant-ling"`, `packages/ai/src/api/openai-completions.ts`). Not yet
    /// consulted by [`crate::dialect::openai::build_body`] — added so a manual `models.json` override
    /// (`ModelOverride::dialect`-style BYO catalogue) can already name this shape; wiring the actual
    /// wire-building branch is tracked as follow-up work (see `crate::models`'s Ant-Ling capability
    /// branch doc comment).
    AntLing,
}

/// The extended-thinking request shape a model accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingShape {
    /// No Anthropic-style thinking control — omit the `thinking` field (OpenAI reasoning models use
    /// `reasoning_effort` instead; see [`ModelCaps::reasoning_effort`]).
    None,
    /// Anthropic `{type:"enabled", budget_tokens}` — Claude 3.7 / 4.x extended thinking.
    Budget,
    /// Anthropic `{type:"adaptive", display}` (plus a sibling top-level `output_config.effort`) — the
    /// shape generation-6+ Claude/Fable models require (`Budget`'s `{type:"enabled", budget_tokens}` is
    /// rejected). This is our own default model's shape (`claude-opus-4-8`); the live smoke test
    /// exercises it directly rather than assuming `Budget` covers everything current.
    Adaptive,
}

/// Per-model capability flags the wire adapters and compaction consult.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelCaps {
    /// Context window in tokens — the compaction default when the caller doesn't override it.
    pub context_window: u32,
    /// A safe default per-turn output ceiling for the model.
    pub max_output: u32,
    /// Which output-ceiling field name to emit on the wire.
    pub max_tokens_field: MaxTokensField,
    /// Whether the model honors the 1-hour prompt-cache TTL (`cache_control.ttl = "1h"`).
    pub supports_long_cache: bool,
    /// Whether the model accepts image input.
    pub supports_vision: bool,
    /// Whether the model accepts a `temperature` parameter at all. `true` for almost every model;
    /// `false` for `claude-opus-4-7`/`claude-opus-4-8` specifically, which reject it outright (pi:
    /// `compat.supportsTemperature: false`, `anthropic.models.ts`) — distinct from the existing
    /// thinking-enabled gate (`ModelRequest::temperature`'s own doc comment), which omits the field
    /// only while extended thinking is on. A model can need both gates at once.
    pub supports_temperature: bool,
    /// The extended-thinking shape the model accepts.
    pub thinking: ThinkingShape,
    /// Whether the model takes an OpenAI `reasoning_effort` parameter.
    pub reasoning_effort: bool,
    /// Whether the model accepts an *explicit* "reasoning off" signal — Anthropic
    /// `{"type":"disabled"}`, or OpenAI `{"effort":"none"}` — when the caller isn't requesting
    /// thinking/reasoning this turn. A model that supports thinking/`reasoning_effort` at all doesn't
    /// necessarily support turning it off explicitly (e.g. `claude-fable-5`); one that supports
    /// neither has nothing to disable, so this is `false` there too.
    pub reasoning_disableable: bool,
    /// Whether the model's tool definitions should be marked with `eager_input_streaming: true` on the
    /// Anthropic wire (ignored outside the Anthropic dialect). `true` for every current Anthropic id;
    /// exists as a named capability (rather than a blanket constant) so a future model needing the
    /// mutually-exclusive `fine-grained-tool-streaming-2025-05-14` beta header instead has somewhere to
    /// say so.
    pub supports_eager_tool_streaming: bool,
    /// Whether tool definitions on the OpenAI Chat-Completions wire should be sent alongside a
    /// top-level `tool_stream: true` (ignored outside that dialect, and only ever emitted when the
    /// request actually carries tools — see [`crate::dialect::openai::build_body`]). Mirrors pi's
    /// `compat.zaiToolStream` (`openai-completions.ts:582-586`): every current Z.ai/GLM id sets it
    /// except `glm-4.5-air`, the one id pi's own `zai.models.ts`/`zai-coding-cn.models.ts` catalogue
    /// leaves it off for; no other family sets it at all.
    pub supports_tool_stream: bool,
    /// Which OpenAI-wire API surface the model speaks (ignored for Anthropic ids). See [`ApiKind`].
    pub api: ApiKind,
    /// The lowest [`crate::transport::ReasoningEffort`] this model's wire actually accepts — a request
    /// below this floor is clamped up (e.g. `gpt-5.5-pro` rejects both `minimal` and `low`, so its floor
    /// is `Medium`). Only consulted for a model that takes `reasoning_effort`/adaptive thinking at all;
    /// meaningless (and unread) otherwise. See [`clamp_reasoning_effort`].
    pub min_reasoning_effort: crate::transport::ReasoningEffort,
    /// Whether this model's wire has a distinct `xhigh` tier at all. Several current OpenAI reasoning
    /// models (o-series, bare/gpt-5.1-family gpt-5) and two Anthropic adaptive models (sonnet-4-6,
    /// sonnet-5) top out at `high` — requesting `xhigh` there must clamp down to `high`, not send a
    /// value the provider doesn't recognize. See [`clamp_reasoning_effort`].
    pub supports_xhigh_reasoning: bool,
    /// The Anthropic adaptive-thinking wire string for `xhigh`, once [`clamp_reasoning_effort`] has
    /// confirmed the model supports it at all. Only `claude-opus-4-6` differs (`"max"` — pi: "effort
    /// 'max' is only valid on Opus 4.6, while Opus 4.7+ and Fable 5 support 'xhigh'"); every other
    /// adaptive-shape id sends the literal `"xhigh"`. Unread outside `ThinkingShape::Adaptive`.
    pub adaptive_xhigh_effort_wire: &'static str,
    /// Which OpenAI Chat-Completions-wire reasoning-toggle shape this model expects. Only read by
    /// [`crate::dialect::openai::build_body`]; every other dialect ignores it. See
    /// [`OpenAiReasoningFormat`].
    pub openai_reasoning_format: OpenAiReasoningFormat,
    /// Whether the Anthropic dialect may stamp a `cache_control` breakpoint on the last tool definition
    /// (`dialect::anthropic::mark_last_tool`). `true` for every current Anthropic-wire model except the
    /// 14 Fireworks ids `Dialect::for_model`'s `is_fireworks_anthropic_wire_model` routes through this
    /// dialect (DeepSeek-V4, GLM-5.1, gpt-oss-120b/20b, Kimi-K2.6/K2.7-Code + their `-fast`/`-turbo`
    /// router variants, MiniMax-M2.7/M3, Qwen3.7-Plus) — pi's `fireworks.models.ts` sets
    /// `supportsCacheControlOnTools: false` on all of them (mirroring pi's own per-model
    /// `compat.supportsCacheControlOnTools` gate, `anthropic-messages.ts`). Ignored outside the
    /// Anthropic dialect.
    pub supports_cache_control_on_tools: bool,
}

impl ModelCaps {
    /// Conservative defaults for an unknown model id: the universally-accepted `max_tokens` field, no
    /// long cache, no thinking — every choice that can't make a request invalid.
    const fn unknown() -> Self {
        Self {
            context_window: 128_000,
            max_output: 4_096,
            max_tokens_field: MaxTokensField::MaxTokens,
            supports_long_cache: false,
            supports_vision: false,
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort: false,
            reasoning_disableable: false,
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::ChatCompletions,
            min_reasoning_effort: crate::transport::ReasoningEffort::Minimal,
            supports_xhigh_reasoning: true,
            adaptive_xhigh_effort_wire: "xhigh",
            openai_reasoning_format: OpenAiReasoningFormat::Standard,
            supports_cache_control_on_tools: true,
        }
    }
}

/// Whether a `gpt-5.*` id's wire has a distinct `xhigh` reasoning tier. True from generation 5.2
/// onward (`gpt-5.2`/`.3`/`.4`/`.5`, any suffix — codex/pro/spark/chat-latest all included); false for
/// bare `gpt-5` and every `gpt-5.1` variant, none of which list `xhigh` in pi's live catalogue at all.
/// Shared across all three `gpt-5` return sites below (chat-latest, the codex-spark special case, and
/// the general bucket) rather than duplicated per branch.
fn gpt5_supports_xhigh(m: &str) -> bool {
    m.starts_with("gpt-5.2")
        || m.starts_with("gpt-5.3")
        || m.starts_with("gpt-5.4")
        || m.starts_with("gpt-5.5")
}

/// Adjust a native-OpenAI-family `ModelCaps` (o-series/gpt-4/gpt-5) for a vendor-slug id (e.g.
/// OpenRouter's `"openai/gpt-5.2"`, matched via `family_id`). Every current OpenRouter entry serves
/// these ids over Chat Completions (`api: "openai-completions"` in pi's own `openrouter.models.ts`),
/// never the Responses API this table's native branches otherwise return, and tags them
/// `compat.thinkingFormat: "openrouter"` — so a vendor-slug id must not inherit the native branch's
/// `api`/`max_tokens_field`/`openai_reasoning_format` verbatim, even though its context/output/
/// vision/reasoning-effort numbers are a reasonable port of the native ones (pi's own OpenRouter
/// numbers for these ids are mostly close to, but not always identical to, the native ones — an
/// accepted approximation, the same "family-level, not exhaustive" tradeoff this table already makes
/// for Together/HuggingFace/NVIDIA's third-party catalogues elsewhere). A no-op for a bare (unprefixed)
/// native id.
fn openai_family_caps_for_vendor_slug(mut caps: ModelCaps, is_vendor_slug: bool) -> ModelCaps {
    if is_vendor_slug {
        caps.api = ApiKind::ChatCompletions;
        caps.max_tokens_field = MaxTokensField::MaxTokens;
        caps.openai_reasoning_format = OpenAiReasoningFormat::OpenRouter;
    }
    caps
}

/// Every id in pi's Mistral catalogue (`packages/ai/src/providers/mistral.models.ts`) starts with one
/// of these — there's no single common prefix for the whole provider the way there is for e.g.
/// DeepSeek: codestral/devstral/ministral/magistral/mistral itself/pixtral/open-mistral/open-mixtral are
/// all separate id families under this one native `KNOWN_PROVIDERS` gateway route
/// (`crates/gateway/src/route.rs`). `m` must already be lowercased (every caller below goes through
/// [`is_mistral_model`], the public, self-lowercasing wrapper).
const MISTRAL_ID_PREFIXES: &[&str] = &[
    "mistral",
    "codestral",
    "devstral",
    "ministral",
    "magistral",
    "pixtral",
    "open-mistral",
    "open-mixtral",
    "labs-devstral",
];

fn is_mistral_id(m: &str) -> bool {
    MISTRAL_ID_PREFIXES.iter().any(|p| m.starts_with(p))
}

/// Whether `model` is a Mistral id. Shared between this table's Mistral `capabilities()` branch and
/// `dialect::openai`'s Mistral-specific tool-call-id reshaping (Mistral's real API rejects a
/// `tool_call_id` that isn't exactly 9 alphanumeric characters) — both need the identical "is this a
/// Mistral model" check, so it lives here once rather than being duplicated per call site.
pub fn is_mistral_model(model: &str) -> bool {
    is_mistral_id(&model.to_ascii_lowercase())
}

/// Whether `model` is a DeepSeek id — `dialect::openai::build_body`'s assistant-replay path needs this
/// specifically (not the broader `OpenAiReasoningFormat::DeepSeek` shape, which Moonshot/Kimi shares
/// too) to backfill an empty `reasoning_content` on replay, matching pi's `compat
/// .requiresReasoningContentOnAssistantMessages` (`isDeepSeek`, `openai-completions.ts::detectCompat`) —
/// a DeepSeek-only quirk, not shared with any other family tagged with the same wire *shape*.
pub fn is_deepseek_model(model: &str) -> bool {
    model.to_ascii_lowercase().starts_with("deepseek")
}

/// Whether `model` (already lowercased) is a Fireworks-hosted id — every current id in pi's
/// `packages/ai/src/providers/fireworks.models.ts` carries this exact path-shaped prefix
/// (`"accounts/fireworks/models/…"` or `"accounts/fireworks/routers/…"`), distinctive enough (unlike a
/// generic `org/model` vendor slug) to reliably disambiguate a Fireworks id from an identically-suffixed
/// one on another host — see [`normalize_fireworks_p_separator`] and the MiniMax branch's own
/// Fireworks-vs-Together disambiguation below for the two places that distinction actually matters.
pub fn is_fireworks_model(m: &str) -> bool {
    m.starts_with("accounts/fireworks/")
}

/// Normalize Fireworks' "p"-for-"." version-separator spelling (`"glm-5p1"` → `"glm-5.1"`,
/// `"kimi-k2p6"` → `"kimi-k2.6"`) so this table's existing dot-spelled family/sub-id checks (written
/// against every other host's spelling) match a Fireworks id unchanged. Only ever called on an id
/// already confirmed Fireworks-shaped ([`is_fireworks_model`]); replaces exactly the ascii-digit-`p`-
/// ascii-digit pattern (Fireworks' own account/router path segments contain no such sequence, so this
/// is a safe, generic scan rather than a fixed set of substring replacements).
fn normalize_fireworks_p_separator(m: &str) -> String {
    let bytes = m.as_bytes();
    let mut out = String::with_capacity(m.len());
    let mut i = 0;
    while i < bytes.len() {
        if i + 2 < bytes.len()
            && bytes[i].is_ascii_digit()
            && bytes[i + 1] == b'p'
            && bytes[i + 2].is_ascii_digit()
        {
            out.push(bytes[i] as char);
            out.push('.');
            i += 2; // Re-visit `bytes[i+2]` (the second digit) as the next iteration's first byte.
        } else {
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    out
}

/// Every current id belonging to a provider pi's `openai-completions.ts::detectCompat` marks
/// `isNonStandard` (hence `supportsStore: false` — the provider rejects or simply doesn't understand
/// OpenAI's `store` extension field): DeepSeek, Z.ai/GLM, Moonshot/Kimi, xAI/Grok, Together-hosted
/// Qwen, Ant-Ling, and Cerebras's native (unprefixed) id set. `false` — meaning `store: false` gets
/// sent — for everything else, including native OpenAI-via-Chat-Completions, OpenRouter, Groq,
/// Fireworks, Mistral, and any uncatalogued id, matching pi's own default.
///
/// pi's real exclusion list also covers NVIDIA and Cloudflare (Workers AI and AI Gateway) — neither of
/// which has an id shape of their own this table could recognize (they're reached via arbitrary
/// vendor-native ids through a NIM/gateway proxy, not a fixed prefix), the same known limitation already
/// documented at this table's own "Third-party OpenAI-compatible providers" section header: a
/// route/provider-level distinction with no matching id-level signature can't be told apart from a
/// generic third-party id by name alone.
pub fn is_non_standard_store_provider(model: &str) -> bool {
    const CEREBRAS_NATIVE: &[&str] = &[
        "gpt-oss-120b",
        "gpt-oss-20b",
        "gpt-oss-safeguard-20b",
        "gemma-4-31b",
        "zai-glm-4.7",
    ];
    let m = model.to_ascii_lowercase();
    m.starts_with("deepseek")
        || m.starts_with("glm")
        || m.starts_with("kimi")
        || m.starts_with("grok")
        || (m.starts_with("qwen") && m != "qwen/qwen3-32b")
        || m.starts_with("ling-")
        || m.starts_with("ring-")
        || CEREBRAS_NATIVE.contains(&m.as_str())
}

/// GitHub Copilot-specific capability overrides for the four current gen6+ Claude ids whose real
/// numbers (`packages/ai/src/providers/github-copilot.models.ts`) diverge from Anthropic's own native
/// catalogue (`anthropic.models.ts`) enough to matter: a smaller context and/or output ceiling, and
/// (for opus-4-7/4-8) a `compat.supportsTemperature: false` entry the native table also carries for
/// these same two ids, so it's honored here too rather than left ungated. Only ever consulted for a
/// Copilot-hosted id (dot-spelled, no vendor-slug prefix — see the Claude branch's own caller) after
/// generation normalization, so `m` here is already dash-spelled. `None` for every other Claude id —
/// including the rest of the gen6+ family (sonnet-5, fable-5) and every earlier generation — which fall
/// through to this table's ordinary (native-numbered) rules unchanged.
fn github_copilot_claude_overrides(m: &str) -> Option<ModelCaps> {
    // (context_window, max_output, supports_temperature, adaptive_xhigh_effort_wire) — the four fields
    // Copilot's own catalogue diverges on for these ids; every other `ModelCaps` field below matches
    // the shared gen6+ shape (`Adaptive` thinking, vision, long cache, xhigh support) unchanged.
    let (context_window, max_output, supports_temperature, adaptive_xhigh_effort_wire) =
        if m.starts_with("claude-opus-4-6") {
            (1_000_000, 32_000, true, "max")
        } else if m.starts_with("claude-opus-4-7") {
            (200_000, 32_000, false, "xhigh")
        } else if m.starts_with("claude-opus-4-8") {
            (200_000, 64_000, false, "xhigh")
        } else if m.starts_with("claude-sonnet-4-6") {
            // pi-parity (models/dialects pass): Copilot's own `thinkingLevelMap` for this id is
            // `{"minimal":"low","xhigh":"max"}` (`github-copilot.models.ts`) — `"max"`, not the literal
            // `"xhigh"` every other adaptive-shape id on this route sends. Sibling id `opus-4-6` above
            // already gets this right; this one was hardcoded to the generic `"xhigh"` instead.
            (1_000_000, 32_000, true, "max")
        } else {
            return None;
        };
    Some(ModelCaps {
        context_window,
        max_output,
        max_tokens_field: MaxTokensField::MaxTokens,
        supports_long_cache: true,
        supports_vision: true,
        supports_temperature,
        thinking: ThinkingShape::Adaptive,
        reasoning_effort: false,
        // None of these four ids null `thinkingLevelMap.off` in Copilot's own catalogue (unlike
        // claude-fable-5's `{"off":null}` natively) — every one of them can be told to disable
        // thinking explicitly.
        reasoning_disableable: true,
        supports_eager_tool_streaming: true,
        supports_tool_stream: false,
        api: ApiKind::ChatCompletions,
        min_reasoning_effort: crate::transport::ReasoningEffort::Minimal,
        supports_xhigh_reasoning: true,
        adaptive_xhigh_effort_wire,
        openai_reasoning_format: OpenAiReasoningFormat::Standard,
        supports_cache_control_on_tools: true,
    })
}

/// NVIDIA's id-for-id capability table (`packages/ai/src/providers/nvidia.models.ts`) — see
/// `capabilities`'s own NVIDIA branch doc comment for why this is ported id-for-id (case-insensitive
/// exact match against the full vendor-slug id) rather than bucketed by family. `None` for any id not
/// in NVIDIA's current catalogue (the caller falls through to every other family branch, and
/// eventually the generic vendor-slug fallback, unchanged).
///
/// One of NVIDIA's own ids is deliberately *not* listed below, despite being in its real catalogue:
/// `"moonshotai/kimi-k2.6"` is id-for-id identical to an id Together/HuggingFace already serve under
/// the same literal string (and this table already has established, tested coverage for via the Kimi
/// `family_id` branch further below) — with a genuinely different real max_output on NVIDIA
/// specifically. Since this function is checked *first*, listing it here would silently steal that id
/// away from the coverage already tested for the other, equally-real host — the same known "one id,
/// more than one host" limitation already documented at this table's own section header, just newly
/// reachable because NVIDIA happens to reuse an org-slug shape another aggregator also uses. Left to
/// the Kimi branch instead; NVIDIA's own real number for this id remains unrepresented, same as any
/// other cross-host collision this table can't disambiguate.
///
/// `"minimaxai/minimax-m3"` *used* to be excluded for the identical reason (left to the MiniMax
/// `family_id` branch further below, which — like every host that branch's own "else" bucket covers —
/// reports 128,000 max_output, NVIDIA's real number here being merely "unrepresented"). That framing
/// undersold the bug (pi-parity pass 20, Task 2): NVIDIA's own real max_output for this id is 16,384 —
/// a **7.8x over-report**, not a harmless gap, whenever this id is actually served by NVIDIA. Listed
/// here now instead, using NVIDIA's real (and, conveniently, smallest of the three real hosts —
/// Together's own real is 250,000, HuggingFace's 128,000) number — the same "smaller number wins"
/// safe-direction tie-break this table already uses for every other unresolvable same-string
/// collision, so Together's/HuggingFace's requests for this identical string now merely under-report
/// (a usability loss) instead of NVIDIA's over-reporting outright (a 400).
///
/// A third, converse case is investigated (pi-parity Task #21) but deliberately left as-is rather than
/// "fixed": `"nvidia/nemotron-3-ultra-550b-a55b"` below is *also* re-hosted, id-for-id, on Together
/// (`together.models.ts:268`: real 512300/512300, vs NVIDIA's own real and already-tested 1,000,000/
/// 65,536 asserted by `nvidia_nemotron_ids_get_their_real_numbers`). Since this function runs first and
/// matches purely on the id string, a Together request for this exact id silently gets NVIDIA's numbers
/// instead of its own — the max_output direction is safe (NVIDIA's 65,536 is *smaller* than Together's
/// real 512,300 — a usability loss, not a 400), but there is no other family branch a la Kimi/MiniMax
/// that this one could be excluded in favor of ("nemotron" matches no other host's naming convention at
/// all) — excluding it here would only lose NVIDIA's own correct, tested numbers for the far more common
/// native case in exchange for a same-string collision this table still couldn't resolve for Together
/// either. Left unfixed: the same "one id, several real hosts, no route signal" limitation as every
/// other documented collision in this file, one level less tractable than the two ids just above.
///
/// `"nvidia/nemotron-3-super-120b-a12b"` looks like the same shape as `nemotron-3-ultra-550b-a55b`
/// above (also re-hosted on OpenRouter under the identical string), but the safe-smallest-number
/// reasoning that justifies leaving `-ultra` alone does *not* hold here (pi-parity pass 20, Task 1):
/// OpenRouter's real max_output for this exact id is 4096 (`openrouter.models.ts`), vs the 262,144 this
/// table used to return unconditionally — a **64x over-report** for an OpenRouter-routed request, not
/// a mere usability loss. Fixed below by narrowing to OpenRouter's smaller (and therefore safe-for-
/// both-hosts) number; NVIDIA-native's own real max_output for this id is also 262,144, so a genuine
/// NVIDIA-routed request now under-reports instead — the same tie-break this whole doc comment
/// otherwise describes.
fn nvidia_caps(m: &str) -> Option<ModelCaps> {
    use crate::transport::ReasoningEffort as RE;
    // (context_window, max_output, supports_vision) — reasoning/thinking fields are uniform "no
    // mechanism at all" across every current id regardless of pi's own per-id `reasoning: true/false`
    // flag (see this function's own doc comment), so they're not part of this tuple.
    let (context_window, max_output, supports_vision) = match m {
        "meta/llama-3.1-70b-instruct" => (128_000, 4_096, false),
        "meta/llama-3.1-8b-instruct" => (16_000, 4_096, false),
        "meta/llama-3.2-11b-vision-instruct" => (128_000, 4_096, true),
        "meta/llama-3.2-90b-vision-instruct" => (128_000, 8_192, true),
        "meta/llama-3.3-70b-instruct" => (128_000, 4_096, false),
        "minimaxai/minimax-m3" => (1_000_000, 16_384, true),
        "mistralai/mistral-large-3-675b-instruct-2512" => (262_144, 262_144, true),
        "mistralai/mistral-small-4-119b-2603" => (128_000, 8_192, true),
        "nvidia/nemotron-3-nano-30b-a3b" => (131_072, 131_072, false),
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning" => (256_000, 65_536, true),
        // pi-parity pass 20 Task 1: was (262_144, 262_144, false) — NVIDIA's own real number, but a
        // 64x over-report of OpenRouter's real 4096 for this identical vendor-slug string (unlike
        // `nemotron-3-ultra-550b-a55b` below, where NVIDIA's own number already IS the smaller/safer
        // one — see this function's own doc comment).
        "nvidia/nemotron-3-super-120b-a12b" => (262_144, 4_096, false),
        "nvidia/nemotron-3-ultra-550b-a55b" => (1_000_000, 65_536, false),
        "nvidia/nvidia-nemotron-nano-9b-v2" => (131_072, 131_072, false),
        "openai/gpt-oss-120b" => (128_000, 8_192, false),
        "openai/gpt-oss-20b" => (131_072, 32_768, false),
        "qwen/qwen3.5-122b-a10b" => (262_144, 65_536, true),
        "stepfun-ai/step-3.5-flash" => (256_000, 16_384, false),
        "stepfun-ai/step-3.7-flash" => (256_000, 16_384, true),
        // pi-parity pass 20 Task 1: was (1_000_000, 131_072, false) — NVIDIA's own real numbers, but
        // this exact vendor-slug string is *also* OpenRouter's own real spelling
        // (`openrouter.models.ts`: 1,048,576/128,000). Context stays at NVIDIA's smaller (safe) number;
        // max_output takes OpenRouter's smaller one instead (a ~2.4% over-report otherwise) — the same
        // per-field mix-and-match tie-break already used for `"qwen/qwen3-235b-a22b"` elsewhere in this
        // file.
        "z-ai/glm-5.2" => (1_000_000, 128_000, false),
        _ => return None,
    };
    Some(ModelCaps {
        context_window,
        max_output,
        max_tokens_field: MaxTokensField::MaxTokens,
        // pi's `isNvidia` denylist excludes this provider from `supportsLongCacheRetention`.
        supports_long_cache: false,
        supports_vision,
        supports_temperature: true,
        thinking: ThinkingShape::None,
        // `compat.supportsReasoningEffort: false` on every current id (`detectCompat`'s `isNvidia`
        // exclusion) — no client-steerable reasoning mechanism at all, matching xAI/Grok's identical
        // "reasons internally, no toggle" shape elsewhere in this table.
        reasoning_effort: false,
        reasoning_disableable: false,
        supports_eager_tool_streaming: false,
        supports_tool_stream: false,
        api: ApiKind::ChatCompletions,
        min_reasoning_effort: RE::Minimal,
        supports_xhigh_reasoning: false,
        adaptive_xhigh_effort_wire: "xhigh",
        openai_reasoning_format: OpenAiReasoningFormat::Standard,
        supports_cache_control_on_tools: true,
    })
}

/// Resolve a model id to its [`ModelCaps`]. Matching is by id prefix (most-specific first); unknown
/// ids fall back to [`ModelCaps::unknown`] (logged, since a silent conservative fallback can otherwise
/// mask a model we should have taught this table about).
///
/// The per-family numbers below are ported from the reference agent's live model catalogue
/// (`packages/ai/src/providers/{anthropic,openai}.models.ts` in `badlogic/pi-mono`) rather than
/// invented — re-check that catalogue when adding a new model family, since it's regenerated upstream
/// and may have moved on by the time you read this.
///
/// A thin wrapper around [`capabilities_impl`] (which does the actual per-family lookup, unchanged):
/// applies the one post-hoc override every one of that function's ~30 return sites would otherwise
/// need to remember individually — [`ModelCaps::supports_cache_control_on_tools`] is `false` for the 14
/// Fireworks ids [`crate::dialect::is_fireworks_anthropic_wire_model`] routes through the Anthropic
/// dialect (pi's `fireworks.models.ts` sets `supportsCacheControlOnTools: false` on all of them).
pub fn capabilities(model: &str) -> ModelCaps {
    let mut caps = capabilities_impl(model);
    if crate::dialect::is_fireworks_anthropic_wire_model(model) {
        caps.supports_cache_control_on_tools = false;
    }
    caps
}

fn capabilities_impl(model: &str) -> ModelCaps {
    use crate::transport::ReasoningEffort as RE;
    let m = model.to_ascii_lowercase();
    // Fireworks spells its models' version separator as the letter "p", not "." (`"glm-5p1"`,
    // `"kimi-k2p6"` vs every other host's `"glm-5.1"`/`"kimi-k2.6"`) — normalized here, before
    // `family_id` extraction and every family branch below, so a Fireworks id matches the exact same
    // dot-spelled sub-id checks every other host already does, rather than needing its own parallel
    // "p"-spelled copy of each one. Scoped to ids carrying Fireworks' own distinctive
    // `"accounts/fireworks/"` path-shaped prefix (`is_fireworks_model`) so an unrelated id that happens
    // to contain a literal "p" between two digits for some other reason is never touched.
    let m = if is_fireworks_model(&m) {
        normalize_fireworks_p_separator(&m)
    } else {
        m
    };
    // The id the third-party family branches below (DeepSeek, GLM, Kimi, MiniMax, Qwen, MiMo) match
    // their bare-id patterns against: the suffix after the last `/` for a vendor-slug id — the shape
    // aggregator hosts (Together, HuggingFace, NVIDIA NIM, …) prefix a model with the org that trained
    // it (`"moonshotai/Kimi-K2.6"`, `"zai-org/GLM-5.2"`, `"XiaomiMiMo/MiMo-V2.5-Pro"`) — or the id
    // itself when it isn't slug-prefixed at all. Using this (rather than the raw id, or a
    // `.contains(family)` substring check) fixes two failure modes at once: a slug-prefixed id whose
    // org slug doesn't happen to start with the family name (`"moonshotai/kimi-k2.6"` doesn't start
    // with `"kimi"`) used to fall all the way through to the generic 128k/32k vendor-slug fallback
    // further below; and an org slug that *does* happen to share a literal prefix with a different
    // family's own name — `"MiniMaxAI/MiniMax-M3"` lowercases to `"minimaxai/minimax-m3"`, which
    // itself begins with the literal string `"minimax"` one character early — used to land in the
    // right family branch but silently match the wrong per-id sub-case inside it (the flat "else"
    // shape, not `minimax-m3`'s real one). Deliberately not applied to Anthropic/OpenAI/Mistral/
    // xAI/Cerebras-native/Groq/Llama below: none of those are reachable through an aggregator's
    // vendor-slug id shape in pi's live catalogues, and blanket-applying it there would let a
    // vendor-slug id's *suffix* alone decide family membership for providers that never actually
    // appear slug-prefixed — an unnecessary widening of what each of those branches matches.
    let family_id: &str = match m.rfind('/') {
        Some(idx) => &m[idx + 1..],
        None => &m,
    };

    // ---- Anthropic Claude (+ Fable, which speaks the Anthropic wire) ----
    // Also matches a vendor-slug id (OpenRouter's `"anthropic/claude-opus-4.6"`) via `family_id` — see
    // its own doc comment above — and a dot-spelled generation separator (GitHub Copilot's own
    // `"claude-opus-4.6"`, and OpenRouter's identically dot-spelled vendor-slug ids), which every rule
    // below is written against the dash-spelled native shape (pi's own `anthropic.models.ts`) instead.
    if m.starts_with("claude")
        || m.starts_with("fable")
        || family_id.starts_with("claude")
        || family_id.starts_with("fable")
    {
        // The suffix this branch's own rules match against: the vendor-slug suffix if slug-shaped,
        // otherwise the bare id — same reasoning as the MiniMax/GLM/Kimi/DeepSeek/MiMo branches' own
        // `k`/`g`/`mm` locals below. Additionally normalizes a dot-spelled generation separator to a
        // dash: no real Claude/Fable id ever uses a literal `.` in its native (Anthropic-direct) form,
        // so a blanket replace is safe once we're already inside this family's own branch — unlike
        // doing it before family dispatch, which would corrupt an unrelated family's *meaningful* use
        // of `.` in its own version numbering (DeepSeek-V3.2, Kimi-K2.6, …).
        let is_vendor_slug = m.contains('/');
        let base = if is_vendor_slug { family_id } else { m.as_str() };
        let is_dot_spelled = base.contains('.');
        let normalized = base.replace('.', "-");
        let m: &str = &normalized;

        // GitHub Copilot hosts a handful of these ids with real numbers that diverge from Anthropic's
        // own native catalogue (`packages/ai/src/providers/github-copilot.models.ts` vs
        // `anthropic.models.ts`) enough to matter. Detected the same way pi's own gateway would key it:
        // this id reached here bare (no vendor-slug prefix) yet was originally dot-spelled — the one
        // shape unique to Copilot among the three hosts this branch now recognizes (native:
        // dash-spelled, unprefixed; OpenRouter: dot-spelled, "anthropic/"-prefixed; Copilot:
        // dot-spelled, unprefixed).
        if is_dot_spelled && !is_vendor_slug {
            if let Some(caps) = github_copilot_claude_overrides(m) {
                return caps;
            }
        }

        // Generation 6+ (opus-4-6/4-7/4-8, sonnet-4-6, sonnet-5, fable-5) require the newer `adaptive`
        // thinking shape (`Budget`'s `{type:"enabled", budget_tokens}` is rejected) and ship a 1M-token
        // context window — a full step up from every earlier generation. Matched by the exact
        // family+generation token so "claude-opus-4-5" can't collide with "claude-opus-4-6" and later.
        const GEN6_PLUS: &[&str] = &[
            "claude-opus-4-6",
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-sonnet-4-6",
            "claude-sonnet-5",
            "claude-fable-5",
            "fable-5",
        ];
        if GEN6_PLUS.iter().any(|p| m.starts_with(p)) {
            // sonnet-4-6 caps output at 64k; every other gen6+ family caps at 128k.
            let max_output = if m.starts_with("claude-sonnet-4-6") {
                64_000
            } else {
                128_000
            };
            // Every gen6+ id can be told to disable thinking explicitly, except claude-fable-5 (pi:
            // `thinkingLevelMap: {"off": null}` — there's no "off" wire shape for it at all).
            let reasoning_disableable =
                !(m.starts_with("claude-fable-5") || m.starts_with("fable-5"));
            // sonnet-4-6/sonnet-5 carry no `thinkingLevelMap` at all in pi's catalogue, and an unmapped
            // "xhigh" isn't a value their wire accepts — only opus-4-6/4-7/4-8 and fable-5 do (see
            // `adaptive_xhigh_effort_wire`'s doc comment).
            let supports_xhigh_reasoning =
                !(m.starts_with("claude-sonnet-4-6") || m.starts_with("claude-sonnet-5"));
            // Only opus-4-6 remaps xhigh to "max"; every other adaptive id that supports it sends the
            // literal "xhigh" (the field's own default).
            let adaptive_xhigh_effort_wire = if m.starts_with("claude-opus-4-6") {
                "max"
            } else {
                "xhigh"
            };
            // opus-4-7 and opus-4-8 reject `temperature` outright — pi's `compat.supportsTemperature:
            // false` is set on exactly these two ids in `anthropic.models.ts` and no other gen6+ entry
            // (opus-4-6, sonnet-4-6, sonnet-5, fable-5 all default to supported). Our own DEFAULT_MODEL
            // is `claude-opus-4-8`, so this is a live, reachable 400 without the gate.
            let supports_temperature =
                !(m.starts_with("claude-opus-4-7") || m.starts_with("claude-opus-4-8"));
            return ModelCaps {
                context_window: 1_000_000,
                max_output,
                max_tokens_field: MaxTokensField::MaxTokens,
                supports_long_cache: true,
                supports_vision: true,
                supports_temperature,
                thinking: ThinkingShape::Adaptive,
                reasoning_effort: false,
                reasoning_disableable,
                supports_eager_tool_streaming: true,
                supports_tool_stream: false,
                api: ApiKind::ChatCompletions,
                min_reasoning_effort: crate::transport::ReasoningEffort::Minimal,
                supports_xhigh_reasoning,
                adaptive_xhigh_effort_wire,
                openai_reasoning_format: OpenAiReasoningFormat::Standard,
                supports_cache_control_on_tools: true,
            };
        }

        // Generation 3 (pre-4.x, excluding 3.7-sonnet which *does* support extended thinking): no
        // thinking support at all — sending a `thinking` field would be rejected.
        let is_legacy_gen3 = m.starts_with("claude-3-5")
            || m.starts_with("claude-3-haiku")
            || m.starts_with("claude-3-opus")
            || m.starts_with("claude-3-sonnet");
        if is_legacy_gen3 {
            let max_output = if m.starts_with("claude-3-5") {
                8_192
            } else {
                4_096
            };
            return ModelCaps {
                context_window: 200_000,
                max_output,
                max_tokens_field: MaxTokensField::MaxTokens,
                // Prompt caching (the 1-hour TTL included) is supported on every current Claude 3.x
                // id — pi's catalogue defaults `supportsLongCacheRetention` to `true` and no gen-3
                // entry overrides it. `false` here would silently downgrade a `cache_long` request on
                // these ids to the standard 5-minute TTL instead of the 1-hour one asked for.
                supports_long_cache: true,
                supports_vision: true,
                supports_temperature: true,
                thinking: ThinkingShape::None,
                reasoning_effort: false,
                // No thinking support at all here — nothing to explicitly disable, so the `thinking`
                // field stays omitted entirely rather than sending a `{"type":"disabled"}` a model
                // that never supported thinking might reject.
                reasoning_disableable: false,
                supports_eager_tool_streaming: true,
                supports_tool_stream: false,
                api: ApiKind::ChatCompletions,
                min_reasoning_effort: crate::transport::ReasoningEffort::Minimal,
                supports_xhigh_reasoning: true,
                adaptive_xhigh_effort_wire: "xhigh",
                openai_reasoning_format: OpenAiReasoningFormat::Standard,
                supports_cache_control_on_tools: true,
            };
        }

        // GitHub Copilot's own bare "claude-sonnet-4" (no version-dot at all, unlike every other
        // Copilot Claude id — `github-copilot.models.ts`) has no native Anthropic-catalogue equivalent
        // to collide with (`anthropic.models.ts` has no bare "claude-sonnet-4" entry at all), so this is
        // a safe, unconditional exact-id override. The `is_dot_spelled` gate a few lines above (and
        // `github_copilot_claude_overrides` itself) can never catch this id — it has no "." to detect —
        // so it silently fell through to this generic bucket's `sonnet` default (64_000) instead of
        // Copilot's real, much smaller 16_000 ceiling (a 4x over-report) and 216_000 context (vs
        // 200_000) — pi-parity Task #5. Scoped to the bare (non-vendor-slug) form, matching the same
        // disambiguation discipline `github_copilot_claude_overrides` uses for its own four ids — a
        // same-suffix vendor-slug id (were one ever to exist) must not inherit Copilot's number.
        if m == "claude-sonnet-4" && !is_vendor_slug {
            return ModelCaps {
                context_window: 216_000,
                max_output: 16_000,
                max_tokens_field: MaxTokensField::MaxTokens,
                supports_long_cache: true,
                supports_vision: true,
                supports_temperature: true,
                thinking: ThinkingShape::Budget,
                reasoning_effort: false,
                reasoning_disableable: true,
                supports_eager_tool_streaming: true,
                supports_tool_stream: false,
                api: ApiKind::ChatCompletions,
                min_reasoning_effort: crate::transport::ReasoningEffort::Minimal,
                supports_xhigh_reasoning: false,
                adaptive_xhigh_effort_wire: "xhigh",
                openai_reasoning_format: OpenAiReasoningFormat::Standard,
                supports_cache_control_on_tools: true,
            };
        }

        // Everything else current (opus-4-0/4-1/4-5, sonnet-3-7/4-0/4-5, haiku-4-5, and future ids we
        // haven't special-cased above): `Budget`-shape extended thinking, 200k context.
        //
        // GitHub Copilot's own opus-4.5/sonnet-4.5 (dot-spelled, unprefixed — the same detection
        // `github_copilot_claude_overrides` uses for its own four ids) cap output at Copilot's real,
        // smaller 32_000 ceiling (vs this bucket's generic 64_000 — pi-parity Tasks #6/#7). Unlike those
        // four gen6+ ids, though, opus-4.5/sonnet-4.5 aren't Adaptive-shape on Copilot at all (no
        // `forceAdaptiveThinking`/`thinkingLevelMap` for either in `github-copilot.models.ts`), so they
        // can't simply be added to `github_copilot_claude_overrides`'s allowlist — that would also
        // wrongly switch their wire shape to Adaptive. Context stays 200_000 either way (Copilot's real
        // number already matches this bucket's own default for both ids).
        let copilot_smaller_output = is_dot_spelled
            && !is_vendor_slug
            && (m.starts_with("claude-opus-4-5") || m.starts_with("claude-sonnet-4-5"));
        let max_output = if copilot_smaller_output {
            32_000
        } else if m.contains("sonnet") || m.contains("haiku") || m.contains("opus-4-5") {
            64_000
        } else {
            32_000 // opus-4-0 / opus-4-1 / generic opus and fable ids
        };
        return ModelCaps {
            context_window: 200_000,
            max_output,
            max_tokens_field: MaxTokensField::MaxTokens,
            supports_long_cache: true,
            supports_vision: true,
            supports_temperature: true,
            thinking: ThinkingShape::Budget,
            reasoning_effort: false,
            reasoning_disableable: true,
            supports_eager_tool_streaming: true,
            supports_tool_stream: false,
            api: ApiKind::ChatCompletions,
            // Budget-shape thinking sends a numeric `budget_tokens`, never a named effort string, so
            // `min_reasoning_effort`/`adaptive_xhigh_effort_wire` are never read for this shape — left
            // at their harmless defaults. `supports_xhigh_reasoning` is *not* one of those unread
            // fields, despite the shape never putting a named effort string on the wire: it's read
            // shape-agnostically by `available_thinking_levels`/`clamp_reasoning_effort` to decide
            // whether to offer/accept the `xhigh` *portable* thinking-level rung at all (which then
            // maps to a numeric budget via `budget_for_effort`). pi's `thinkingLevelMap.xhigh` is only
            // defined for the four gen6+ Adaptive-shape ids — none of these classic Budget-shape models
            // (sonnet-4-5, opus-4-5, sonnet-3-7, haiku-4-5, opus-4-0/4-1, etc.) offer it.
            min_reasoning_effort: crate::transport::ReasoningEffort::Minimal,
            supports_xhigh_reasoning: false,
            adaptive_xhigh_effort_wire: "xhigh",
            openai_reasoning_format: OpenAiReasoningFormat::Standard,
            supports_cache_control_on_tools: true,
        };
    }

    // ---- OpenAI reasoning models: o-series ----
    // Reject `max_tokens` (require `max_completion_tokens`); 200k context, 100k max output.
    //
    // `supports_long_cache` below (and on every other OpenAI branch): pi's actual default is
    // *permissive* — `prompt_cache_retention`/24h is supported unless the specific provider opts out
    // (a denylist: Together, Cloudflare, NVIDIA, Ant-Ling — everything else, including native OpenAI,
    // OpenRouter, DeepSeek, Cerebras, Groq, xAI, Mistral, defaults to supporting it). Our table is
    // keyed by *model id*, not by which gateway route/provider is actually serving it, so it can't
    // reproduce that per-provider denylist — we default to `true` (matching pi's default) and accept
    // that a route to a genuinely non-supporting provider (Together is the one we route to) will send
    // a `prompt_cache_retention` field it ignores rather than one it needs but doesn't get.
    // Also matches a vendor-slug id (OpenRouter's `"openai/o3"`) via `family_id` — see its own doc
    // comment above. `family_id` is what the rest of this branch matches against below (shadowing `m`)
    // so a slug prefix never breaks e.g. the `o3-mini` exact-suffix check.
    let is_o_series = m.starts_with("o1")
        || m.starts_with("o3")
        || m.starts_with("o4")
        || family_id.starts_with("o1")
        || family_id.starts_with("o3")
        || family_id.starts_with("o4");
    if is_o_series {
        let is_vendor_slug = m.contains('/');
        let m: &str = if is_vendor_slug { family_id } else { m.as_str() };
        let caps = ModelCaps {
            context_window: 200_000,
            max_output: 100_000,
            // Every o-series id speaks the Responses API (`api` below), which always sends
            // `max_output_tokens` regardless of this field (see `dialect/openai_responses.rs::build_body`)
            // — `max_tokens_field` is only ever read by the Chat Completions dialect. Still set correctly
            // (o-series genuinely rejects `max_tokens`, requiring `max_completion_tokens` instead) rather
            // than left at a value that would be actively wrong if a future routing change ever sent one
            // of these ids through Chat Completions.
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            supports_long_cache: true,
            // o3-mini is text-only (pi's catalogue: `input: ["text"]`) — the one o-series id that isn't
            // vision-capable. o1-mini used to be excluded here too, but pi's live catalogue has fully
            // retired it (no entry at all, as of this writing) — nothing upstream still serves it, so
            // there's no longer a live model to special-case.
            supports_vision: !m.starts_with("o3-mini"),
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort: true,
            // o-series ids are disable-capable by default in pi's catalogue (no override).
            reasoning_disableable: true,
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::Responses,
            min_reasoning_effort: crate::transport::ReasoningEffort::Minimal,
            // No o-series id carries a `thinkingLevelMap` at all in pi's catalogue, and an unmapped
            // "xhigh" is excluded from `getSupportedThinkingLevels` — none of them accept it.
            supports_xhigh_reasoning: false,
            adaptive_xhigh_effort_wire: "xhigh", // unread: o-series never uses Adaptive shape.
            openai_reasoning_format: OpenAiReasoningFormat::Standard,
            supports_cache_control_on_tools: true,
        };
        return openai_family_caps_for_vendor_slug(caps, is_vendor_slug);
    }

    // ---- OpenAI GPT-5 family (reasoning) ----
    // Also matches a vendor-slug id (OpenRouter's `"openai/gpt-5.2"`) via `family_id` — see its own doc
    // comment above; `m` is shadowed to that suffix for the rest of this branch so every exact-id/
    // prefix check below keeps working unmodified against a slug-prefixed id too.
    if m.starts_with("gpt-5") || family_id.starts_with("gpt-5") {
        let is_vendor_slug = m.contains('/');
        let m: &str = if is_vendor_slug { family_id } else { m.as_str() };
        // The narrower "-chat-latest" variants share the family name but cap at the older
        // chat-completions ceiling (128k/16384), not the reasoning-model one, and aren't uniformly
        // `reasoning_effort`-driven — treat them like a non-reasoning chat model. Two of the four
        // current ids (5.1/5.2) are still `reasoning_effort`-driven per pi's catalogue, though;
        // gpt-5-chat-latest/gpt-5.3-chat-latest are not. None of the four support an explicit "off"
        // signal (pi: `"off": null` for this whole bucket).
        if m.contains("chat-latest") {
            let reasoning_effort =
                m.starts_with("gpt-5.1-chat-latest") || m.starts_with("gpt-5.2-chat-latest");
            let caps = ModelCaps {
                context_window: 128_000,
                max_output: 16_384,
                max_tokens_field: MaxTokensField::MaxTokens,
                supports_long_cache: true,
                supports_vision: true,
                supports_temperature: true,
                thinking: ThinkingShape::None,
                reasoning_effort,
                reasoning_disableable: false,
                supports_eager_tool_streaming: false,
                supports_tool_stream: false,
                api: ApiKind::Responses,
                min_reasoning_effort: crate::transport::ReasoningEffort::Minimal,
                supports_xhigh_reasoning: gpt5_supports_xhigh(m),
                adaptive_xhigh_effort_wire: "xhigh", // unread: this bucket never uses Adaptive shape.
                openai_reasoning_format: OpenAiReasoningFormat::Standard,
                supports_cache_control_on_tools: true,
            };
            return openai_family_caps_for_vendor_slug(caps, is_vendor_slug);
        }
        // "gpt-5.3-codex-spark" is a narrower model than the rest of the family — 128k context, 32k
        // output — not the generic 400k/128k every other gpt-5 id gets below. Not in pi's
        // disable-capable allowlist (that's `gpt-5.3-codex`, a different id).
        if m == "gpt-5.3-codex-spark" {
            let caps = ModelCaps {
                context_window: 128_000,
                max_output: 32_000,
                // Unread: Responses-routed (see the o-series branch above for why this is still set
                // correctly rather than to a harmless-but-wrong value).
                max_tokens_field: MaxTokensField::MaxCompletionTokens,
                supports_long_cache: true,
                supports_vision: true,
                supports_temperature: true,
                thinking: ThinkingShape::None,
                reasoning_effort: true,
                reasoning_disableable: false,
                supports_eager_tool_streaming: false,
                supports_tool_stream: false,
                api: ApiKind::Responses,
                min_reasoning_effort: crate::transport::ReasoningEffort::Minimal,
                supports_xhigh_reasoning: true,
                adaptive_xhigh_effort_wire: "xhigh",
                openai_reasoning_format: OpenAiReasoningFormat::Standard,
                supports_cache_control_on_tools: true,
            };
            return openai_family_caps_for_vendor_slug(caps, is_vendor_slug);
        }
        // "-pro" ships a much larger 1.05M context — but only the 5.4/5.5 generation of it;
        // gpt-5-pro/gpt-5.2-pro are 400k like the rest of the family. The bare "gpt-5.4"/"gpt-5.5"
        // release (no mini/nano/pro suffix) runs a smaller 272k window; every other family member is
        // 400k.
        let context_window = if m == "gpt-5.4-pro" || m == "gpt-5.5-pro" {
            1_050_000
        } else if m == "gpt-5.4" || m == "gpt-5.5" {
            272_000
        } else {
            400_000
        };
        // Disable-capability is a per-exact-id allowlist in pi's catalogue, not a blanket rule — an
        // id under this generic branch that isn't listed here (e.g. bare "gpt-5", "gpt-5-mini",
        // "gpt-5.3" without "-codex", any "-pro"/"-nano" variant not listed) has no "off" signal.
        const GPT5_DISABLE_CAPABLE: &[&str] = &[
            "gpt-5.1",
            "gpt-5.2",
            "gpt-5.3-codex",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.4-nano",
            "gpt-5.5",
        ];
        let reasoning_disableable = GPT5_DISABLE_CAPABLE.contains(&m);
        // gpt-5.5-pro excludes both "minimal" and "low" (floor: medium); gpt-5.5 excludes just
        // "minimal" (floor: low) — pi's `thinkingLevelMap` nulls those out explicitly. Every earlier
        // gpt-5 generation accepts the full ladder from minimal up.
        let min_reasoning_effort = if m == "gpt-5.5-pro" {
            crate::transport::ReasoningEffort::Medium
        } else if m == "gpt-5.5" {
            crate::transport::ReasoningEffort::Low
        } else {
            crate::transport::ReasoningEffort::Minimal
        };
        // pi-parity (models/dialects pass): OpenRouter's own vendor-slug "openai/gpt-5-nano" reports a
        // real `maxTokens` of 4096 (`openrouter.models.ts`), not this bucket's generic 128000 — no
        // collision with any other host reachable through this codebase's known providers (OpenRouter-
        // only vendor-slug spelling for this exact suffix).
        let max_output = if is_vendor_slug && m == "gpt-5-nano" {
            4_096
        } else {
            128_000
        };
        let caps = ModelCaps {
            context_window,
            max_output,
            // Unread: Responses-routed (see the o-series branch above for why this is still set
            // correctly rather than to a harmless-but-wrong value).
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            supports_long_cache: true,
            supports_vision: true,
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort: true,
            reasoning_disableable,
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::Responses,
            min_reasoning_effort,
            supports_xhigh_reasoning: gpt5_supports_xhigh(m),
            adaptive_xhigh_effort_wire: "xhigh", // unread: this bucket never uses Adaptive shape.
            openai_reasoning_format: OpenAiReasoningFormat::Standard,
            supports_cache_control_on_tools: true,
        };
        return openai_family_caps_for_vendor_slug(caps, is_vendor_slug);
    }

    // ---- OpenAI GPT-4 family (bare gpt-4 / 4-turbo / 4o / 4.1) ----
    // None of these take `reasoning_effort` at all, so there's nothing to explicitly disable. Also
    // matches a vendor-slug id (OpenRouter's `"openai/gpt-4o"`) via `family_id` — see its own doc
    // comment above.
    if m.starts_with("gpt-4") || family_id.starts_with("gpt-4") {
        let is_vendor_slug = m.contains('/');
        let m: &str = if is_vendor_slug { family_id } else { m.as_str() };
        // 4.1 shipped a ~1M-token context window, a full step up from the rest of the family.
        if m.starts_with("gpt-4.1") {
            // pi-parity (models/dialects pass): OpenRouter's own vendor-slug "openai/gpt-4.1" reports a
            // real `maxTokens` of 4096 (`openrouter.models.ts`), not this bucket's generic 32768 — no
            // collision with any other host reachable through this codebase's known providers (this
            // exact vendor-slug spelling is OpenRouter-only), so this is a plain vendor-slug-scoped
            // correction, not a `capabilities_for_route_with_host` case. Scoped to the exact bare suffix
            // "gpt-4.1" — OpenRouter's own "-mini"/"-nano" siblings already match this bucket's default
            // 32768 for real, so they must stay unaffected.
            let max_output = if is_vendor_slug && m == "gpt-4.1" {
                4_096
            } else {
                32_768
            };
            let caps = ModelCaps {
                context_window: 1_047_576,
                max_output,
                max_tokens_field: MaxTokensField::MaxTokens,
                supports_long_cache: true,
                supports_vision: true,
                supports_temperature: true,
                thinking: ThinkingShape::None,
                reasoning_effort: false,
                reasoning_disableable: false,
                supports_eager_tool_streaming: false,
                supports_tool_stream: false,
                api: ApiKind::Responses,
                // The gpt-4 family never takes `reasoning_effort` at all — these three are unread.
                min_reasoning_effort: crate::transport::ReasoningEffort::Minimal,
                supports_xhigh_reasoning: true,
                adaptive_xhigh_effort_wire: "xhigh",
                openai_reasoning_format: OpenAiReasoningFormat::Standard,
                supports_cache_control_on_tools: true,
            };
            return openai_family_caps_for_vendor_slug(caps, is_vendor_slug);
        }
        // This one pinned snapshot caps output tighter (4096) than every other 4o-family id (16384).
        if m == "gpt-4o-2024-05-13" {
            let caps = ModelCaps {
                context_window: 128_000,
                max_output: 4_096,
                max_tokens_field: MaxTokensField::MaxTokens,
                supports_long_cache: true,
                supports_vision: true,
                supports_temperature: true,
                thinking: ThinkingShape::None,
                reasoning_effort: false,
                reasoning_disableable: false,
                supports_eager_tool_streaming: false,
                supports_tool_stream: false,
                api: ApiKind::Responses,
                // The gpt-4 family never takes `reasoning_effort` at all — these three are unread.
                min_reasoning_effort: crate::transport::ReasoningEffort::Minimal,
                supports_xhigh_reasoning: true,
                adaptive_xhigh_effort_wire: "xhigh",
                openai_reasoning_format: OpenAiReasoningFormat::Standard,
                supports_cache_control_on_tools: true,
            };
            return openai_family_caps_for_vendor_slug(caps, is_vendor_slug);
        }
        // pi-parity pass 20 Task 3: OpenRouter's own vendor-slug "openai/gpt-4-turbo-preview" is a
        // genuinely text-only legacy alias (`input: ["text"]`, `openrouter.models.ts`) — unlike its
        // differently-named sibling "openai/gpt-4-turbo" (`input: ["text","image"]`, vision-capable),
        // which the generic bucket below still correctly covers via `m != "gpt-4"`. Native OpenAI's own
        // catalogue (`openai.models.ts`) has no "gpt-4-turbo-preview" entry at all — only OpenRouter
        // does — so this is scoped to the vendor-slug form specifically; a bare id under this name (if
        // one somehow existed) would still fall through to the generic bucket's `m != "gpt-4"` default.
        if is_vendor_slug && m == "gpt-4-turbo-preview" {
            let caps = ModelCaps {
                context_window: 128_000,
                max_output: 4_096,
                max_tokens_field: MaxTokensField::MaxTokens,
                supports_long_cache: true,
                supports_vision: false,
                supports_temperature: true,
                thinking: ThinkingShape::None,
                reasoning_effort: false,
                reasoning_disableable: false,
                supports_eager_tool_streaming: false,
                supports_tool_stream: false,
                api: ApiKind::Responses,
                min_reasoning_effort: crate::transport::ReasoningEffort::Minimal,
                supports_xhigh_reasoning: true,
                adaptive_xhigh_effort_wire: "xhigh",
                openai_reasoning_format: OpenAiReasoningFormat::Standard,
                supports_cache_control_on_tools: true,
            };
            return openai_family_caps_for_vendor_slug(caps, is_vendor_slug);
        }
        // Bare "gpt-4" (no suffix) is the original 8k-context model; 4-turbo caps output tighter than
        // 4o; everything else (4o, 4o-mini, and dated snapshots) shares a 128k/16384 ceiling.
        let (context_window, max_output) = if m == "gpt-4" {
            (8_192, 8_192)
        } else if m.starts_with("gpt-4-turbo") {
            (128_000, 4_096)
        } else {
            (128_000, 16_384)
        };
        let caps = ModelCaps {
            context_window,
            max_output,
            max_tokens_field: MaxTokensField::MaxTokens,
            supports_long_cache: true,
            // Bare "gpt-4" (pi's catalogue: `input: ["text"]`) predates GPT-4's vision support
            // entirely — that shipped separately as "gpt-4-vision-preview" and was later folded into
            // "gpt-4-turbo" (`input: ["text","image"]`) and every 4o-family id. Every other id in this
            // branch does accept image input.
            supports_vision: m != "gpt-4",
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort: false,
            reasoning_disableable: false,
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::Responses,
            min_reasoning_effort: crate::transport::ReasoningEffort::Minimal,
            supports_xhigh_reasoning: true,
            adaptive_xhigh_effort_wire: "xhigh",
            openai_reasoning_format: OpenAiReasoningFormat::Standard,
            supports_cache_control_on_tools: true,
        };
        return openai_family_caps_for_vendor_slug(caps, is_vendor_slug);
    }

    // ---- Third-party OpenAI-compatible providers ----
    //
    // The gateway's own provider table (`crates/gateway/src/route.rs::KNOWN_PROVIDERS`, plus several
    // Anthropic-wire vendors reachable only via config) actively routes to every family below; without
    // an entry here each one silently fell through to `ModelCaps::unknown()` — 128k context, a 4096
    // output ceiling regardless of the model's real one (`Agent::new` seeds `max_tokens` from
    // `max_output`), and no reasoning/thinking wire support at all (`--thinking high` on, say,
    // `deepseek-v4-pro`, sent no thinking parameter whatsoever). Numbers below are ported from pi's
    // live catalogue (`packages/ai/src/providers/{deepseek,zai,moonshotai,xai,groq,cerebras,together,
    // minimax}.models.ts`, `openai-completions.ts::detectCompat`) — re-check those when adding a new
    // model, same caveat as the Anthropic/OpenAI tables above. Coverage here is intentionally
    // family-level (matched by a recognizable id prefix/substring), not an exhaustive per-exact-id
    // table the way the native Anthropic/OpenAI branches above are: these providers' catalogues
    // (OpenRouter's especially, which mirrors nearly every model that exists) are large and change
    // often, and a reasonably-accurate family default is what actually closes the truncation/no-thinking
    // gap — an exact number a few percent off doesn't reopen it. A model id genuinely reachable through
    // more than one host (e.g. a Together-hosted "deepseek-ai/deepseek-r1" also starting with the
    // literal prefix "deepseek") can still land on the wrong sub-branch; this is a known limitation of a
    // table keyed on model id alone, with no route/provider context to disambiguate by.

    // NVIDIA (`packages/ai/src/providers/nvidia.models.ts`): every current id is vendor/model-shaped
    // (`"meta/llama-3.1-70b-instruct"`, `"z-ai/glm-5.2"`, `"mistralai/mistral-small-4-119b-2603"`, …)
    // but NVIDIA's own NIM catalogue behind those same org slugs is a *different* real model than what
    // Together/HuggingFace/native serve under the identical-looking id — e.g. NVIDIA's own
    // `"openai/gpt-oss-120b"` (128k/8k) isn't Groq's identically-named id (131k/64k) matched by the
    // `contains("gpt-oss")` check further below, and `"mistralai/mistral-large-3-675b-instruct-2512"`
    // doesn't match any of the Mistral branch's own exact ids at all (its org slug happens to start
    // with the literal string "mistral" too, so it would otherwise silently land in that branch's own
    // smaller `_` catch-all default). Checked here, before every other family/`family_id` branch below,
    // so NVIDIA's own real numbers always win. Ported id-for-id (`nvidia_caps`) rather than bucketed —
    // too much per-id variance for a family-level default, the same reasoning the Mistral branch below
    // gives for its own id-for-id table. pi's `isNvidia` denylist also unconditionally sets
    // `supportsReasoningEffort: false` on every current id regardless of its own `reasoning: true/false`
    // flag (the same "reasons internally, no client-steerable toggle" shape this table already gives
    // xAI/Grok) — so no NVIDIA id gets a thinking/reasoning mechanism here, matching that.
    if let Some(caps) = nvidia_caps(&m) {
        return caps;
    }

    // DeepSeek: 1M context, 384k output, a real reasoning-effort vocabulary (floor `high`, `xhigh`
    // wired as `"max"`) — pi: `compat.thinkingFormat: "deepseek"`, `thinkingLevelMap: {high:"high",
    // xhigh:"max"}`, `supportsReasoningEffort` left at its (`!isZai && …`) default of `true`.
    // DeepSeek's own auto-detected `maxTokensField` is `max_completion_tokens` (not in pi's
    // `useMaxTokens` allowlist), matching this table's other reasoning-model families. Also matches a
    // vendor-slug id whose suffix is a DeepSeek id (e.g. Together/HuggingFace's
    // `"deepseek-ai/DeepSeek-V4-Pro"`) via `family_id` — see its own doc comment above.
    if m.starts_with("deepseek") || family_id.starts_with("deepseek") {
        // HuggingFace's own bare-ish naming for three specific ids (`deepseek-ai/DeepSeek-R1`,
        // `-R1-0528`, `-V3.2`, `packages/ai/src/providers/huggingface.models.ts`) is dramatically
        // smaller than the family-wide default below (up to ~15x on context, ~12x on max_output) — the
        // generic bucket badly overstates them. Matched on the *full* id (not just `family_id`'s
        // suffix): HuggingFace's `"deepseek-ai/…"` prefix never collides with any other host's own
        // differently-prefixed id sharing the same bare suffix (e.g. OpenRouter's
        // `"deepseek/deepseek-r1"` is a different full string entirely), so this is a safe exact-id
        // override, not a family-level default.
        if let Some((context_window, max_output)) = match m.as_str() {
            "deepseek-ai/deepseek-r1" => Some((64_000, 32_768)),
            "deepseek-ai/deepseek-r1-0528" => Some((163_840, 163_840)),
            "deepseek-ai/deepseek-v3.2" => Some((163_840, 65_536)),
            // pi-parity Task #20: unlike the three ids above (HuggingFace-only, no collision), this
            // exact id is *also* served by Together with the same real max_output (384000, already
            // matching the family-wide default below) but a real context of only 512000
            // (`together.models.ts:156`) — half the generic bucket's 1,000,000, which is near-correct
            // for HuggingFace's own real 1,048,576 (`huggingface.models.ts:475`) but a 2x over-report
            // for Together. Together's smaller, safer number wins here (the accepted cost: HuggingFace's
            // real, larger context is now under-reported instead of Together's being over-reported).
            "deepseek-ai/deepseek-v4-pro" => Some((512_000, 384_000)),
            // pi-parity (models/dialects pass): HuggingFace-only (no collision) — real 1048576/384000
            // (`huggingface.models.ts`) vs the family-wide default below's 1_000_000/384_000: only
            // `context_window` was under-reported (~4.6%), `max_output` already matched.
            "deepseek-ai/deepseek-v4-flash" => Some((1_048_576, 384_000)),
            // pi-parity (models/dialects pass): OpenRouter's own vendor-slug "deepseek/deepseek-r1"
            // (a different full id string from HuggingFace's "deepseek-ai/deepseek-r1" just above, so
            // no collision) reports a real 163840/16000 (`openrouter.models.ts`) — dramatically smaller
            // than the family-wide default below (1,000,000/384,000), which an existing regression test
            // used to assert as this id's own number before this fix.
            "deepseek/deepseek-r1" => Some((163_840, 16_000)),
            _ => None,
        } {
            return ModelCaps {
                context_window,
                max_output,
                max_tokens_field: MaxTokensField::MaxCompletionTokens,
                supports_long_cache: true,
                supports_vision: false,
                supports_temperature: true,
                thinking: ThinkingShape::None,
                reasoning_effort: true,
                reasoning_disableable: true,
                supports_eager_tool_streaming: false,
                supports_tool_stream: false,
                api: ApiKind::ChatCompletions,
                min_reasoning_effort: RE::High,
                supports_xhigh_reasoning: true,
                adaptive_xhigh_effort_wire: "max",
                openai_reasoning_format: OpenAiReasoningFormat::DeepSeek,
                supports_cache_control_on_tools: true,
            };
        }
        return ModelCaps {
            context_window: 1_000_000,
            max_output: 384_000,
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            supports_long_cache: true,
            supports_vision: false,
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort: true,
            reasoning_disableable: true,
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::ChatCompletions,
            min_reasoning_effort: RE::High,
            supports_xhigh_reasoning: true,
            adaptive_xhigh_effort_wire: "max",
            openai_reasoning_format: OpenAiReasoningFormat::DeepSeek,
            supports_cache_control_on_tools: true,
        };
    }

    // Xiaomi/MiMo: no bare "mimo"/"xiaomi" branch existed at all before this — every current id
    // (`packages/ai/src/providers/xiaomi.models.ts`, identical across the ams/cn/sgp token-plan
    // variants — only `baseUrl`/`provider` differ) fell all the way through to `ModelCaps::unknown()`,
    // capping `max_output` at 4096 regardless of the model's real 65536-384000 one and disabling
    // reasoning entirely. pi tags every id `compat: {"requiresReasoningContentOnAssistantMessages":
    // true, "thinkingFormat": "deepseek"}` — the identical `thinking:{enabled/disabled}` toggle
    // DeepSeek and Kimi share, with a sibling `reasoning_effort` string since `supportsReasoningEffort`
    // resolves `true` for this provider (xiaomi isn't in pi's `isGrok`/`isZai`/`isMoonshot`/`isTogether`/
    // `isCloudflareAiGateway`/`isNvidia`/`isAntLing` exclusion list) — the exact same
    // `OpenAiReasoningFormat::DeepSeek` shape, reused rather than adding a new enum variant. No id
    // carries a `thinkingLevelMap` at all, so (unlike real DeepSeek) there's no floor/remap: every
    // effort level is legal and sent under its own literal name. `"mimo-v2.5-pro"` starts with the same
    // string as the vision-capable bare `"mimo-v2.5"`, so vision is matched by exact id, not prefix.
    // Also matches a vendor-slug id (HuggingFace's `"XiaomiMiMo/MiMo-V2.5-Pro"`) via `family_id`.
    if m.starts_with("mimo") || family_id.starts_with("mimo") {
        // Keyed on whether `m` is slug-shaped at all — same reasoning as the MiniMax/GLM/Kimi branches.
        let is_vendor_slug = m.contains('/');
        let k = if is_vendor_slug { family_id } else { m.as_str() };
        let (context_window, max_output, supports_vision) = if k == "mimo-v2-flash" {
            // HuggingFace's own "XiaomiMiMo/MiMo-V2-Flash" has a much smaller real `maxTokens` (4_096)
            // than the native (bare, unprefixed) id's 65_536 — an id-suffix collision `family_id`
            // alone can't resolve (see this table's own section-header doc comment), disambiguated
            // here by whether the id is vendor-slug-shaped at all (only HuggingFace hosts this family
            // slug-prefixed; the bare id is always native).
            if is_vendor_slug {
                (262_144, 4_096, false)
            } else {
                (262_144, 65_536, false)
            }
        } else if k == "mimo-v2-omni" {
            (262_144, 131_072, true)
        } else if k == "mimo-v2-pro" {
            (1_048_576, 131_072, false)
        } else if k == "mimo-v2.5" {
            (1_048_576, 131_072, true)
        } else {
            // "mimo-v2.5-pro", "mimo-v2.5-pro-ultraspeed", and any future id this exact-match table
            // doesn't recognize yet: most current ids share this 1M/131k, text-only shape.
            (1_048_576, 131_072, false)
        };
        return ModelCaps {
            context_window,
            max_output,
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            supports_long_cache: true,
            supports_vision,
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort: true,
            reasoning_disableable: true,
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::ChatCompletions,
            min_reasoning_effort: RE::Minimal,
            supports_xhigh_reasoning: true,
            adaptive_xhigh_effort_wire: "xhigh",
            openai_reasoning_format: OpenAiReasoningFormat::DeepSeek,
            supports_cache_control_on_tools: true,
        };
    }

    // Ant-Ling (`packages/ai/src/providers/ant-ling.models.ts`): three current ids, none reachable
    // through a vendor-slug id in pi's own catalogue (no aggregator hosts this family slug-prefixed),
    // so no `family_id` matching needed here unlike Kimi/GLM/MiniMax/DeepSeek/MiMo. This table had zero
    // branch for the family at all before now — every id fell through to `ModelCaps::unknown()`,
    // losing its real ~2-6x larger context/output ceiling. Only `Ring-2.6-1T` is reasoning-capable;
    // its `thinkingLevelMap` nulls minimal/low/medium/off entirely (floor: `High`), and its wire shape
    // (`compat.thinkingFormat: "ant-ling"`, `openai-completions.ts`) is unique among every format this
    // table already ports: a nested `reasoning: {effort}` sent *only* when a level is actually
    // requested and mapped to a string — never an explicit "off" signal the way DeepSeek/Zai/
    // OpenRouter/Together all send one. `OpenAiReasoningFormat::AntLing` exists for this shape and *is*
    // wired into `dialect::openai::build_body`'s `apply_reasoning_wire` (its `Fmt::AntLing` arm) — a
    // `ling-`/`ring-`-prefixed id isn't Anthropic-named and reports `ApiKind::ChatCompletions` here, so
    // `Dialect::for_model` already routes it to this dialect by default, no manual `models.json`
    // override needed to reach it. This entry closes the truncation gap (`context_window`/`max_output`)
    // every dialect reads regardless of which wire ends up serving the request.
    if m.starts_with("ling-") || m.starts_with("ring-") {
        let is_reasoning = m.starts_with("ring-2.6-1t");
        return ModelCaps {
            context_window: 262_144,
            max_output: 65_536,
            max_tokens_field: MaxTokensField::MaxTokens,
            // pi's `isAntLing` denylist excludes this provider from `supportsLongCacheRetention`.
            supports_long_cache: false,
            supports_vision: false,
            supports_temperature: true,
            thinking: ThinkingShape::None,
            // No top-level `reasoning_effort` string in this wire shape at all (the whole mechanism
            // lives in the nested `reasoning.effort` field itself, once wired) — same reasoning as
            // Kimi/pre-5.2 GLM's toggle-only families; `has_reasoning_mechanism`'s third arm (below)
            // is what correctly reports a mechanism exists despite this being `false`.
            reasoning_effort: false,
            // `thinkingLevelMap.off` is null — no explicit "off" wire signal exists at all for this
            // format, matching e.g. `claude-fable-5`/`kimi-k2.7-code`'s identical convention elsewhere
            // in this table.
            reasoning_disableable: false,
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::ChatCompletions,
            min_reasoning_effort: if is_reasoning { RE::High } else { RE::Minimal },
            supports_xhigh_reasoning: is_reasoning,
            adaptive_xhigh_effort_wire: "xhigh", // unread: never uses the Anthropic Adaptive shape.
            openai_reasoning_format: if is_reasoning {
                OpenAiReasoningFormat::AntLing
            } else {
                OpenAiReasoningFormat::Standard
            },
            supports_cache_control_on_tools: true,
        };
    }

    // Mistral: a native `KNOWN_PROVIDERS` gateway route (`crates/gateway/src/route.rs`) whose 30-id
    // catalogue (`packages/ai/src/providers/mistral.models.ts`) has no shared prefix and enough per-id
    // variance (bare "mistral-large-2411" is 131_072/16_384 while "mistral-large-2512" two generations
    // later is 262_144/262_144) that a family-level default would still misreport several ids by a wide
    // margin — so this is ported id-for-id, not bucketed, matched via `is_mistral_id` above. This
    // codebase has no bespoke Mistral wire client (pi's own `mistral-conversations.ts` speaks a
    // different, non-OpenAI-shaped API); every Mistral id is routed through the generic OpenAI Chat
    // Completions dialect instead — see `dialect::openai::build_body`'s Mistral-specific tool-call-id
    // reshaping and `reasoning_wire_override` below for the two wire-shape adjustments that gap needs.
    if is_mistral_id(&m) {
        let (context_window, max_output, reasoning_effort, supports_vision) = match m.as_str() {
            "codestral-latest" => (256_000, 4_096, false, false),
            "devstral-2512" => (262_144, 262_144, false, false),
            "devstral-latest" => (262_144, 262_144, false, false),
            "devstral-medium-2507" => (128_000, 128_000, false, false),
            "devstral-medium-latest" => (262_144, 262_144, false, false),
            "devstral-small-2505" => (128_000, 128_000, false, false),
            "devstral-small-2507" => (128_000, 128_000, false, false),
            "labs-devstral-small-2512" => (256_000, 256_000, false, true),
            "magistral-medium-latest" => (128_000, 16_384, true, false),
            "magistral-small" => (128_000, 128_000, true, false),
            "ministral-3b-latest" => (128_000, 128_000, false, false),
            "ministral-8b-latest" => (128_000, 128_000, false, false),
            "mistral-large-2411" => (131_072, 16_384, false, false),
            "mistral-large-2512" => (262_144, 262_144, false, true),
            "mistral-large-latest" => (262_144, 262_144, false, true),
            "mistral-medium-2505" => (131_072, 131_072, false, true),
            "mistral-medium-2508" => (262_144, 262_144, false, true),
            "mistral-medium-2604" => (262_144, 262_144, true, true),
            "mistral-medium-3.5" => (262_144, 262_144, true, true),
            "mistral-medium-latest" => (262_144, 262_144, false, true),
            "mistral-nemo" => (128_000, 128_000, false, false),
            "mistral-small-2506" => (128_000, 16_384, false, true),
            "mistral-small-2603" => (256_000, 256_000, true, true),
            "mistral-small-latest" => (256_000, 256_000, true, true),
            "open-mistral-7b" => (8_000, 8_000, false, false),
            "open-mistral-nemo" => (128_000, 128_000, false, false),
            "open-mixtral-8x22b" => (64_000, 64_000, false, false),
            "open-mixtral-8x7b" => (32_000, 32_000, false, false),
            "pixtral-12b" => (128_000, 128_000, false, true),
            "pixtral-large-latest" => (128_000, 128_000, false, true),
            // A future/uncatalogued Mistral id this exact-match table doesn't recognize yet: a
            // reasonable family-wide default (128k/128k, no reasoning, no vision) rather than falling
            // all the way through to `unknown()`'s flatter 4096-token ceiling.
            _ => (128_000, 128_000, false, false),
        };
        return ModelCaps {
            context_window,
            max_output,
            max_tokens_field: MaxTokensField::MaxTokens,
            // Not in pi's Together/Cloudflare/NVIDIA/Ant-Ling `supportsLongCacheRetention` denylist.
            supports_long_cache: true,
            supports_vision,
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort,
            // No Mistral id nulls "off" in pi's catalogue (none carry a `thinkingLevelMap` at all) —
            // every reasoning-capable id can be told to turn reasoning off.
            reasoning_disableable: reasoning_effort,
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::ChatCompletions,
            min_reasoning_effort: RE::Minimal,
            // No Mistral id defines an "xhigh" wire value at all.
            supports_xhigh_reasoning: false,
            adaptive_xhigh_effort_wire: "xhigh", // unread: Mistral never uses Adaptive shape.
            openai_reasoning_format: OpenAiReasoningFormat::Standard,
            supports_cache_control_on_tools: true,
        };
    }

    // Z.ai/GLM: `compat.thinkingFormat: "zai"` — a `thinking:{enabled/disabled}` toggle sent
    // unconditionally, with a `reasoning_effort` string only from GLM-5.2 onward
    // (`supportsReasoningEffort` is `false` on every earlier id in pi's catalogue). `glm-5v*` is the
    // one vision-capable id; every other current id is text-only. Also matches a vendor-slug id (e.g.
    // Together's/HuggingFace's `"zai-org/GLM-5.2"`) via `family_id` — see its own doc comment above.
    if m.starts_with("glm") || family_id.starts_with("glm") {
        // Keyed on whether `m` is slug-shaped at all (not on which disjunct above matched) — same
        // reasoning as the MiniMax branch below, applied uniformly even though no current GLM org
        // slug happens to collide with the literal prefix "glm" the way MiniMax's does.
        let g = if m.contains('/') { family_id } else { m.as_str() };
        let (context_window, max_output, reasoning_effort) = if g.starts_with("glm-5.2") {
            // pi-parity Task #15: Together's and HuggingFace's own vendor-slug "zai-org/GLM-5.2"
            // (`together.models.ts:363`, `huggingface.models.ts:853`) both report a real context of
            // 262144, far smaller than NVIDIA's/native's own 1,000,000 this branch otherwise returns —
            // checked against the *full* id (not just the `family_id` suffix both share with native),
            // so Fireworks' differently-prefixed "glm-5p2" (normalized to the same "glm-5.2" suffix
            // but under `accounts/fireworks/models/glm-5p2`) is unaffected. The two hosts disagree with
            // each other on max_output (164000 vs 131072) — out of scope for this fix; only context
            // (which they *do* agree on) is corrected here.
            if m == "zai-org/glm-5.2" {
                (262_144, 131_072, true)
            } else if m == "zai/glm-5.2" {
                // pi-parity Task #30: Vercel AI Gateway's own vendor-slug spelling ("zai/glm-5.2", no
                // "-org" — distinct from Together's/HuggingFace's "zai-org/glm-5.2" just above) reports
                // a real context of 1,040,000, not native's/NVIDIA's 1,000,000 — a negligible ~4%
                // under-report, fixed here for completeness. The sibling "zai/glm-5.2-fast" id already
                // matches its own real 1,000,000 via the `else` branch below and is left unaffected.
                (1_040_000, 131_072, true)
            } else {
                // Note: OpenRouter's own vendor-slug spelling ("z-ai/glm-5.2", a dash between "z" and
                // "ai") never reaches this arm at all — `nvidia_caps` (checked earlier, unconditionally)
                // already lists this exact string first, since it's *also* a real NVIDIA-native id. See
                // `nvidia_caps`'s own doc comment (pi-parity pass 20, Task 1) for that fix instead.
                (1_000_000, 131_072, true)
            }
        } else if g.starts_with("glm-4.5-air") || g == "glm-4.5" {
            // Bare "glm-4.5" (HuggingFace's own id, `huggingface.models.ts:709`) reports identical real
            // numbers to the already-correct "glm-4.5-air" bucket (131072/98304) — pi-parity Task #23.
            (131_072, 98_304, false)
        } else if g.starts_with("glm-4.7") {
            // pi-parity (models/dialects pass): HuggingFace's own "zai-org/GLM-4.7-Flash" reports a
            // real 200000/128000 (`huggingface.models.ts`) — a small (~2.4%) over-report vs this
            // bucket's own 204800/131072 default, which native "glm-4.7" itself keeps.
            if m == "zai-org/glm-4.7-flash" {
                (200_000, 128_000, false)
            } else {
                (204_800, 131_072, false)
            }
        } else if g.starts_with("glm-4.5v") {
            // HuggingFace's "zai-org/GLM-4.5V" — vision-capable, and notably smaller than every other
            // id in this family (65_536/16_384 vs the 200k+/131k+ every other bucket gets).
            (65_536, 16_384, false)
        } else if m == "zai-org/glm-4.6" {
            // pi-parity (models/dialects pass): HuggingFace-only (`huggingface.models.ts`), real
            // 204800/131072 — a small (~2.3%) context under-report vs this bucket's generic default.
            (204_800, 131_072, false)
        } else if m == "zai-org/glm-5" || m == "zai-org/glm-5.1" {
            // pi-parity (models/dialects pass): Together and HuggingFace both host this exact id
            // string and agree on the real numbers (202752/131072, `together.models.ts`/
            // `huggingface.models.ts`) — no collision, just a small (~1.4%) context correction vs this
            // bucket's generic default.
            (202_752, 131_072, false)
        } else if m == "z-ai/glm-5" {
            // pi-parity (models/dialects pass): OpenRouter's own vendor-slug spelling ("z-ai/", not
            // Together's/HuggingFace's "zai-org/" above — a genuinely different full id string, so no
            // collision) reports a real 202752/4096 (`openrouter.models.ts`) — this bucket's generic
            // 131072 max_output is a dangerous 32x over-report for this specific route.
            (202_752, 4_096, false)
        } else {
            (200_000, 131_072, false)
        };
        return ModelCaps {
            context_window,
            max_output,
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            supports_long_cache: true,
            // "glm-5v*" is the vision-capable id in the current native/coding-cn catalogue; HuggingFace
            // additionally hosts "GLM-4.5V" (`"glm-4.5v"` here) — a *different* vision-capable id this
            // check used to miss entirely (`"glm-4.5v".starts_with("glm-5v")` is false; different major
            // generation, same "v" vision suffix convention).
            supports_vision: g.starts_with("glm-5v") || g.starts_with("glm-4.5v"),
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort,
            reasoning_disableable: true,
            supports_eager_tool_streaming: false,
            // Every current GLM id sets pi's `compat.zaiToolStream: true` except "glm-4.5-air", the
            // one id both `zai.models.ts` and `zai-coding-cn.models.ts` leave it off for (see
            // `ModelCaps::supports_tool_stream`'s own doc comment).
            supports_tool_stream: !g.starts_with("glm-4.5-air"),
            api: ApiKind::ChatCompletions,
            // glm-5.2+ nulls "minimal" out of its `thinkingLevelMap` (excluded entirely, not just
            // remapped) — every earlier GLM id has no reasoning vocabulary at all, so this floor is
            // unread for them regardless.
            min_reasoning_effort: if g.starts_with("glm-5.2") {
                RE::Low
            } else {
                RE::Minimal
            },
            supports_xhigh_reasoning: reasoning_effort,
            adaptive_xhigh_effort_wire: "max",
            openai_reasoning_format: OpenAiReasoningFormat::Zai,
            supports_cache_control_on_tools: true,
        };
    }

    // Moonshot/Kimi: pi tags this family with the identical `"deepseek"` thinkingFormat (the same
    // `thinking:{enabled/disabled}` toggle), but `supportsReasoningEffort` is `false` on every current
    // id — no `reasoning_effort` string ever, just the toggle. Only the non-"thinking" ids
    // (0711/0905/turbo-preview previews) have `reasoning: false` in pi's catalogue, so they get
    // `OpenAiReasoningFormat::Standard` instead (with `reasoning_effort: false`, that emits nothing —
    // matching pi's own `model.reasoning` gate). `kimi-k2.7-code*` uniquely has no "off" wire value
    // (`thinkingLevelMap: {"off": null}`); every other reasoning-capable id can be disabled. Also
    // matches a vendor-slug id (e.g. Together's/HuggingFace's `"moonshotai/Kimi-K2.6"`) via
    // `family_id` — see its own doc comment above.
    //
    // KNOWN COLLISION (documented, not fixed — no id-level signal exists to disambiguate): GitHub
    // Copilot hosts its own `"kimi-k2.7-code"` (`github-copilot.models.ts`) with real numbers
    // (256_000 context / 32_000 max output) much smaller than moonshotai's own identically-spelled,
    // unprefixed native id (262_144/262_144, `moonshotai.models.ts`) this branch returns below. Unlike
    // the Claude branch's Copilot ids, there is no dot-vs-dash or vendor-slug tell here — Kimi ids are
    // dot-spelled on every host — so the bare id alone can't route between them. Kept at the native
    // (larger) numbers, matching every other id in this branch's own "one bare id, one host" default;
    // a future route-aware capability lookup (see Task 23-style `is_codex`/`is_azure` precedent) would
    // be the real fix.
    // "k2p7" is Kimi-Coding's own alias for its Kimi K2.7 Code offering (`kimi-coding.models.ts`) — it
    // doesn't start with "kimi" at all, so it needs an explicit entry into this branch alongside the
    // ordinary prefix check.
    if m.starts_with("kimi") || family_id.starts_with("kimi") || m == "k2p7" {
        // Keyed on whether `m` is slug-shaped at all — same reasoning as the MiniMax/GLM branches.
        let is_vendor_slug = m.contains('/');
        let k = if is_vendor_slug { family_id } else { m.as_str() };
        // Kimi-Coding (`api.kimi.com/coding`, pi's `kimi-coding.models.ts`) hosts three ids with a much
        // smaller real `maxTokens` (32_768) than this bucket's generic 262_144 default: its own "k2p7"
        // alias and "kimi-for-coding" (neither collides with any moonshotai-native id at all), plus
        // bare "kimi-k2-thinking" — which *does* collide with moonshotai-native's own identically-
        // spelled bare id (262_144/262_144 there, matching this branch's own "else" default; pi-parity
        // Task #14, same bug class as the documented kimi-k2.7-code/Copilot collision below). Scoped to
        // the bare (non-vendor-slug) id specifically: HuggingFace's own vendor-slug
        // "moonshotai/Kimi-K2-Thinking" reports the *native* (larger) numbers for this exact suffix
        // (`huggingface.models.ts:583`: 262144/262144), not Kimi-Coding's, so only the bare form should
        // take Kimi-Coding's smaller number — the safe-direction choice for the unresolvable bare-id
        // collision (an 8x max_output under-report for moonshotai-native is far safer than an 8x
        // over-report for Kimi-Coding).
        let is_kimi_coding_smaller_output =
            k == "k2p7" || k == "kimi-for-coding" || (!is_vendor_slug && k == "kimi-k2-thinking");
        // HuggingFace's own bare naming for the (non-reasoning) Instruct release — "Kimi-K2-Instruct"/
        // "-0905", no version-dot at all — drifted badly under the generic "else" bucket below (context
        // over-reported 2x for the plain Instruct id; max_output over-reported ~16x for both).
        let is_hf_bare_instruct = k == "kimi-k2-instruct" || k == "kimi-k2-instruct-0905";
        let non_reasoning = k.starts_with("kimi-k2-0711")
            || k.starts_with("kimi-k2-0905")
            || k.starts_with("kimi-k2-turbo-preview")
            || is_hf_bare_instruct;
        // Together hosts its own vendor-slug "moonshotai/Kimi-K2.6"/"moonshotai/Kimi-K2.7-Code" with a
        // real max_output smaller than this bucket's generic 262_144 (`together.models.ts:230,249`:
        // 131000/131072 respectively) — a dangerous ~2x over-report for that host (pi-parity Task #22).
        // HuggingFace serves the *identical* vendor-slug id strings at the generic bucket's own 262144
        // (`huggingface.models.ts:619,637`) — another same-string, no-host-signal collision; Together's
        // smaller, safer numbers win here, matching this table's established tradeoff elsewhere (e.g.
        // the llama-3.3-70b-instruct HuggingFace/OpenRouter collision further below).
        let together_vendor_slug_k2_6 = is_vendor_slug && k == "kimi-k2.6";
        // pi-parity pass 20 Task 1: OpenRouter's own catalogue serves this identical vendor-slug string
        // too (`openrouter.models.ts`: real 262144/16384) — a *third* real host for the same string,
        // and its max_output is smaller still than Together's own 131072 above (an 8x over-report for
        // OpenRouter specifically if Together's number were kept). OpenRouter's smaller number wins,
        // matching this table's established tie-break.
        let together_vendor_slug_k2_7_code = is_vendor_slug && k == "kimi-k2.7-code";
        // pi-parity (models/dialects pass): OpenRouter's own vendor-slug "moonshotai/kimi-k2.5"
        // (`openrouter.models.ts`) reports a real max_output of 4096, vs this bucket's generic 262144 —
        // no collision with any other host reachable through this codebase's known providers (native
        // moonshotai/moonshotai-cn/opencode all use the bare, unprefixed "kimi-k2.5" instead).
        let openrouter_vendor_slug_k2_5 = is_vendor_slug && k == "kimi-k2.5";
        // pi-parity pass 20 Task 1: OpenRouter's own vendor-slug "moonshotai/kimi-k2-thinking" reports a
        // real max_output of 100352 (`openrouter.models.ts`), vs this bucket's generic 262144 default —
        // a ~2.6x over-report. HuggingFace's identical vendor-slug spelling reports the native (larger)
        // 262144/262144 for this same suffix (`huggingface.models.ts:583`, already the "else" default
        // below) — another same-string, no-host-signal collision; OpenRouter's smaller number wins.
        let openrouter_vendor_slug_k2_thinking = is_vendor_slug && k == "kimi-k2-thinking";
        let (context_window, max_output) = if k.starts_with("kimi-k2-0711") {
            (131_072, 16_384)
        } else if is_kimi_coding_smaller_output {
            (262_144, 32_768)
        } else if k == "kimi-k2-instruct" {
            (131_072, 16_384)
        } else if k == "kimi-k2-instruct-0905" {
            (262_144, 16_384)
        } else if together_vendor_slug_k2_6 {
            (262_144, 131_000)
        } else if together_vendor_slug_k2_7_code {
            (262_144, 16_384)
        } else if openrouter_vendor_slug_k2_5 {
            (262_144, 4_096)
        } else if openrouter_vendor_slug_k2_thinking {
            (262_144, 100_352)
        } else {
            (262_144, 262_144)
        };
        let supports_vision = (k.starts_with("kimi-k2.5")
            || k.starts_with("kimi-k2.6")
            || k.starts_with("kimi-k2.7")
            || is_kimi_coding_smaller_output)
            // "kimi-k2-thinking" has no vision on either host (Kimi-Coding's and moonshotai-native's
            // catalogues both list `input: ["text"]`) — excluded even though it's now folded into
            // `is_kimi_coding_smaller_output` above (which otherwise correctly grants vision for the
            // genuinely vision-capable k2p7/kimi-for-coding). Together's own "moonshotai/Kimi-K2.7-Code"
            // is also text-only (`together.models.ts:249`), unlike the bare native id and HuggingFace's
            // identically-spelled vendor-slug entry (both real vision models) — the safe
            // (false-positive-avoiding) choice when the two hosts' identical id strings disagree.
            && k != "kimi-k2-thinking"
            && !together_vendor_slug_k2_7_code;
        return ModelCaps {
            context_window,
            max_output,
            max_tokens_field: MaxTokensField::MaxTokens,
            supports_long_cache: true,
            supports_vision,
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort: false,
            reasoning_disableable: !non_reasoning && !k.starts_with("kimi-k2.7-code"),
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::ChatCompletions,
            min_reasoning_effort: RE::Minimal,
            supports_xhigh_reasoning: true,
            adaptive_xhigh_effort_wire: "xhigh",
            openai_reasoning_format: if non_reasoning {
                OpenAiReasoningFormat::Standard
            } else {
                OpenAiReasoningFormat::DeepSeek
            },
            supports_cache_control_on_tools: true,
        };
    }

    // xAI/Grok: pi's `detectCompat` marks every xAI id `supportsReasoningEffort: false`
    // unconditionally — the reasoning models in this family (grok-4.2x-reasoning, grok-4.3,
    // grok-build) reason on their own, with no client-steerable toggle at all, so `Standard` format
    // with `reasoning_effort: false` correctly emits nothing (matching pi exactly) rather than
    // guessing at a shape xAI doesn't accept. `maxTokensField` auto-detects to
    // `max_completion_tokens` (xAI isn't in pi's `useMaxTokens` allowlist).
    // Also matches a vendor-slug id (OpenRouter's `"x-ai/grok-4.3"`) via `family_id` — see its own doc
    // comment above.
    if m.starts_with("grok") || family_id.starts_with("grok") {
        let is_vendor_slug = m.contains('/');
        let g = if is_vendor_slug { family_id } else { m.as_str() };
        let (context_window, max_output, supports_vision) = if g == "grok-3" || g == "grok-3-fast"
        {
            (131_072, 8_192, false)
        } else if g.starts_with("grok-code-fast") {
            (32_768, 8_192, false)
        } else if g.starts_with("grok-build") {
            (256_000, 256_000, true)
        } else {
            (1_000_000, 30_000, true)
        };
        // OpenRouter hosts every current xAI id with a much smaller real output ceiling than xAI's own
        // native API — pi's `openrouter.models.ts` lists `maxTokens: 4096` for every grok-4.x/grok-build
        // entry there (vs the much larger native ones `xai.models.ts` this table otherwise ports).
        // Before this clamp, a vendor-slug grok id fell through to the generic vendor-slug fallback
        // further below (`max_output: 32_000`) — barely better than not clamping at all, and still the
        // dangerous over-reporting direction (a `max_tokens` the real OpenRouter ceiling would reject).
        // Only the vendor-slug (OpenRouter) variant is clamped; the native, unprefixed id keeps its
        // real larger ceiling.
        let max_output = if is_vendor_slug {
            max_output.min(4_096)
        } else {
            max_output
        };
        return ModelCaps {
            context_window,
            max_output,
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            supports_long_cache: true,
            supports_vision,
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort: false,
            reasoning_disableable: false,
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::ChatCompletions,
            min_reasoning_effort: RE::Minimal,
            supports_xhigh_reasoning: false,
            adaptive_xhigh_effort_wire: "xhigh",
            openai_reasoning_format: OpenAiReasoningFormat::Standard,
            supports_cache_control_on_tools: true,
        };
    }

    // MiniMax: pi's own catalogue serves this family over the *Anthropic* wire
    // (`api: "anthropic-messages"`, `packages/ai/src/providers/minimax.models.ts`) for a *bare*
    // (unprefixed) id — `dialect::mod::routes_to_anthropic_by_default` (`NATIVE_ANTHROPIC_WIRE_BARE_IDS`)
    // now routes those correctly to the Anthropic dialect (a prior pass's fix; this comment used to
    // describe that gap as still open — it is closed for the bare-id case). A vendor-slug id
    // (Together's/HuggingFace's own `"MiniMaxAI/MiniMax-M3"`-shaped ids) still goes through this Chat
    // Completions dialect, matching those hosts' own real `api: "openai-completions"` catalogue entries
    // — so this entry closes the truncation gap (context/max output) `Agent::new` reads regardless of
    // which dialect ends up serving the request, and deliberately leaves `reasoning_effort`/
    // `openai_reasoning_format` at their inert defaults rather than emit an OpenAI-shaped reasoning
    // toggle for the native/bare-id case (now Anthropic-routed, where these OpenAI-wire fields are
    // simply unread). Also matches a vendor-slug id via `family_id` — see its own doc comment above;
    // this one matters even for a *bare* MiniMax-family match, not just a missed one:
    // HuggingFace/Together's real id is `"MiniMaxAI/MiniMax-M3"`, whose lowercased org slug
    // (`"minimaxai"`) itself starts with the literal string `"minimax"` one character early, so
    // matching against the raw `m` here would already (mis)fire this whole branch — just against the
    // wrong per-id sub-case inside it (the flat "else" shape below, not `minimax-m3`'s real one) —
    // without `family_id` correctly isolating the actual suffix first.
    if m.starts_with("minimax") || family_id.starts_with("minimax") {
        // Deliberately keyed on whether `m` is slug-shaped at all (`m.contains('/')`), not on which
        // disjunct above matched: `m.starts_with("minimax")` is *already* (coincidentally) true for
        // "minimaxai/minimax-m3" itself, so selecting on that would silently keep matching the raw,
        // slug-prefixed `m` below instead of the real suffix `family_id` isolated it to.
        let mm = if m.contains('/') { family_id } else { m.as_str() };
        let (context_window, max_output, supports_vision) = if mm.starts_with("minimax-m3") {
            // Fireworks' own "MiniMax-M3" (`accounts/fireworks/models/minimax-m3`) reduces to this
            // identical `family_id` suffix as Together's/HuggingFace's/native's own same-named ids —
            // an inherent "one id, more than one host" collision `family_id` alone can't resolve (see
            // this table's own section-header doc comment) — but Fireworks' id carries a distinctive,
            // reliable full-id prefix (`is_fireworks_model`) no other host's id shape uses, so that one
            // specific host can still be disambiguated by checking the raw id rather than just its
            // `family_id` suffix. Together's own real number (524_288/250_000, vision-capable,
            // `together.models.ts`) is NOT similarly special-cased here: its id ("MiniMaxAI/MiniMax-M3")
            // uses the identical org-slug prefix HuggingFace's own real (but numerically different,
            // 524_288/128_000) entry does, so special-casing that prefix would just trade one
            // undisambiguable collision for another the existing
            // `vendor_slug_minimax_m3_isnt_shadowed_by_its_own_org_slug_prefix_collision` regression
            // test already locks in behavior for — left at this branch's original default, which is at
            // least closer to Together's real context (1_000_000 vs 524_288, safe over-report
            // direction) than any of the alternatives.
            //
            // pi-parity pass 20 Task 2: the vendor-slug string this `else` tuple used to serve
            // ("minimaxai/minimax-m3", Together's/HuggingFace's own spelling too) no longer reaches it
            // at all — `nvidia_caps` (checked earlier, unconditionally) now lists NVIDIA's own real,
            // smaller max_output (16_384) for this exact string first, which wins for every host that
            // shares it (the established "smaller number wins" tie-break — see `nvidia_caps`'s own doc
            // comment). This `else` tuple is only still reachable for the bare native id (no vendor
            // slug at all, so `nvidia_caps`'s exact-string match never fires).
            if is_fireworks_model(&m) {
                (512_000, 512_000, false)
            } else {
                (1_000_000, 128_000, true)
            }
        } else if mm == "minimax-m2" {
            // pi-parity (models/dialects pass): HuggingFace-only (no collision), real 204800/128000
            // (`huggingface.models.ts`) — context already matches this bucket's own default; only
            // max_output was over-reported (~2.4%).
            (204_800, 128_000, false)
        } else if mm == "minimax-m2.7" {
            // pi-parity (models/dialects pass): Together (202752/131072, `together.models.ts`) and
            // HuggingFace (204800/131072, `huggingface.models.ts`) nearly agree — max_output already
            // matches this bucket's default; context differs by <1%. Together's smaller, safer number
            // wins, matching this table's established safe-direction tie-break elsewhere.
            (202_752, 131_072, false)
        } else {
            (204_800, 131_072, false)
        };
        return ModelCaps {
            context_window,
            max_output,
            max_tokens_field: MaxTokensField::MaxTokens,
            supports_long_cache: true,
            supports_vision,
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort: false,
            // Native MiniMax (`minimax.models.ts`/`minimax-cn.models.ts`) carries no `thinkingLevelMap`
            // at all for any of its 3 current ids — unlike Together's own MiniMax entries, which
            // explicitly null `off` (`{"off":null,…}`) — so the native id genuinely *is*
            // disable-capable, not hardcoded `false` regardless of host the way this used to read.
            // Currently inert either way in practice (`has_reasoning_mechanism` already reports no
            // mechanism at all for this whole branch, via `reasoning_effort`/`thinking`/
            // `openai_reasoning_format` staying at their Standard/off defaults — see that function's own
            // doc comment), but correct once this family's dialect-routing gap (`dialect/mod.rs`,
            // Tasks 19-21) or its Anthropic-shape wiring ever reads it.
            reasoning_disableable: !m.contains('/'),
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::ChatCompletions,
            min_reasoning_effort: RE::Minimal,
            supports_xhigh_reasoning: false,
            adaptive_xhigh_effort_wire: "xhigh",
            openai_reasoning_format: OpenAiReasoningFormat::Standard,
            supports_cache_control_on_tools: true,
        };
    }

    // Qwen: pi's real behavior differs by host. Hosted on Groq, the one current id
    // (`qwen/qwen3-32b`) takes a bare `reasoning_effort` (no `compat` override ⇒ pi's `detectCompat`
    // default of the plain OpenAI shape). Hosted on Together — the more common route for this family
    // in pi's catalogue — every entry is `compat.thinkingFormat: "together"` instead. Matched by exact
    // id first so the one Groq case doesn't get swallowed by the generic Together-shaped default below.
    //
    // pi-parity Task #18: HuggingFace hosts the *identically-spelled* id (`huggingface.models.ts:133`)
    // with a real max_output of 16384, not Groq's 40960 — a same-string, no-host-signal collision this
    // table can't resolve any further (context_window agrees at 131072 on both hosts, so only
    // max_output is affected). Groq's real, larger ceiling loses here: the smaller HuggingFace number is
    // the safe direction for both (an under-report is merely a lost-capability, not a 400).
    if m == "qwen/qwen3-32b" {
        return ModelCaps {
            context_window: 131_072,
            max_output: 16_384,
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            supports_long_cache: true,
            supports_vision: false,
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort: true,
            reasoning_disableable: false,
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::ChatCompletions,
            min_reasoning_effort: RE::High,
            supports_xhigh_reasoning: false,
            adaptive_xhigh_effort_wire: "xhigh",
            openai_reasoning_format: OpenAiReasoningFormat::Standard,
            supports_cache_control_on_tools: true,
        };
    }
    // In practice every Together/HuggingFace-hosted Qwen id's own org slug is literally "qwen/…", so
    // `m.starts_with("qwen")` already matches these without needing `family_id` — but check it too
    // (harmless: `family_id` reduces to the same suffix either way) so a future host whose org slug
    // *doesn't* happen to start with "qwen" isn't silently missed the way Kimi/GLM/MiniMax/DeepSeek's
    // vendor-slug ids used to be.
    if m.starts_with("qwen") || family_id.starts_with("qwen") {
        // HuggingFace's own bare "Qwen/Qwen3-235B-A22B" (no additional suffix — Together's own entry
        // for this same base model is a differently-suffixed "...-Instruct-2507-tput" id, so there's no
        // collision with Together) is far smaller than the generic bucket's 200k/40960 default
        // (`huggingface.models.ts:97`: real 40960/16384) — pi-parity Task #16.
        //
        // pi-parity pass 20 Task 1: this identical full id string is *also* OpenRouter's own real
        // vendor-slug spelling (`openrouter.models.ts`: real 131072/8192) — a genuine same-string
        // collision with HuggingFace this table can't disambiguate any further. HuggingFace's smaller
        // context (40960 vs OpenRouter's 131072) already wins safely; max_output disagrees the other
        // way (HuggingFace's 16384 is *larger* than OpenRouter's real 8192, a ~2x over-report there),
        // so `max_output` now takes OpenRouter's smaller number instead — the same "smaller number
        // wins" tie-break used throughout this table.
        if m == "qwen/qwen3-235b-a22b" {
            return ModelCaps {
                context_window: 40_960,
                max_output: 8_192,
                max_tokens_field: MaxTokensField::MaxTokens,
                supports_long_cache: false,
                supports_vision: false,
                supports_temperature: true,
                thinking: ThinkingShape::None,
                reasoning_effort: true,
                reasoning_disableable: true,
                supports_eager_tool_streaming: false,
                supports_tool_stream: false,
                api: ApiKind::ChatCompletions,
                min_reasoning_effort: RE::Minimal,
                supports_xhigh_reasoning: true,
                adaptive_xhigh_effort_wire: "xhigh",
                openai_reasoning_format: OpenAiReasoningFormat::Standard,
                supports_cache_control_on_tools: true,
            };
        }
        let q = if m.contains('/') { family_id } else { m.as_str() };
        // pi-parity pass 20 Task 1: these 6 ids are OpenRouter's own `"qwen/…"` vendor-slug entries
        // (`openrouter.models.ts`), each colliding on `family_id`'s shared suffix `q` with a Together
        // id `together_match` below already covers under a genuinely different (and, for every one of
        // these six, larger) real number — checked against the *full* id `m` (not just `q`), so
        // Together's own numbers for these same suffixes are unaffected. `openai_reasoning_format:
        // OpenRouter` (not `Together`) — OpenRouter's real `compat.thinkingFormat`, a nested
        // `reasoning:{effort}` shape, distinct from Together's own nested `reasoning:{enabled}` toggle.
        if let Some((context_window, max_output, supports_vision, is_reasoning)) = match m.as_str() {
            // 32_768 → 4_096: an 8x over-report (Together's own real max_output for this suffix is
            // 130_000; HuggingFace's identical vendor-slug id is 32_768 — already the shared bucket's
            // pick below — but OpenRouter's own real number is smaller still).
            "qwen/qwen3.5-397b-a17b" => Some((256_000, 4_096, true, true)),
            // 500_000 → 65_536: ~7.6x over-report (Together's own real 1,000,000/500,000 is correct
            // for Together itself — the shared bucket below — but OpenRouter's real max_output is far
            // smaller for this identical suffix). Vision is `false`, not OpenRouter's own real `true`
            // (`input: ["text","image"]`): Together's real entry for this exact string is text-only
            // (`input: ["text"]`) — the same "false is the safe, false-positive-avoiding choice when
            // hosts disagree" tie-break `together_vendor_slug_k2_7_code` already established elsewhere.
            "qwen/qwen3.6-plus" => Some((1_000_000, 65_536, false, true)),
            "qwen/qwen3.7-max" => Some((1_000_000, 65_536, false, true)),
            // Falls to this branch's generic 262_144/65_536 default today (~4x over-report vs
            // OpenRouter's real 262_144/16_384; `reasoning: false` on OpenRouter, so no toggle at all).
            "qwen/qwen3-next-80b-a3b-instruct" => Some((262_144, 16_384, false, false)),
            // ~4x over-report vs the HuggingFace-tuned 262_144/131_072 special case below (real for
            // this identical suffix on HuggingFace; OpenRouter's own real max_output is smaller).
            "qwen/qwen3-next-80b-a3b-thinking" => Some((262_144, 32_768, false, true)),
            // Falls to this branch's generic 262_144/65_536 default today; OpenRouter's real is
            // 160_000/32_768 (context 1.6x, max_output 2x over-reported) — HuggingFace's identical
            // vendor-slug id agrees with the generic default, so this is OpenRouter-specific.
            "qwen/qwen3-coder-30b-a3b-instruct" => Some((160_000, 32_768, false, false)),
            _ => None,
        } {
            return ModelCaps {
                context_window,
                max_output,
                max_tokens_field: MaxTokensField::MaxTokens,
                supports_long_cache: false,
                supports_vision,
                supports_temperature: true,
                thinking: ThinkingShape::None,
                reasoning_effort: is_reasoning,
                reasoning_disableable: true,
                supports_eager_tool_streaming: false,
                supports_tool_stream: false,
                api: ApiKind::ChatCompletions,
                min_reasoning_effort: RE::Minimal,
                supports_xhigh_reasoning: true,
                adaptive_xhigh_effort_wire: "xhigh",
                openai_reasoning_format: if is_reasoning {
                    OpenAiReasoningFormat::OpenRouter
                } else {
                    OpenAiReasoningFormat::Standard
                },
                supports_cache_control_on_tools: true,
            };
        }
        // Together's current Qwen lineup varies too much (context from 32_768 to 1_000_000, max_output
        // from 32_768 to 500_000 — up to ~12x either direction) for the generic 200k/40960 default
        // below to stay reasonably accurate — ported id-for-id instead, the same reasoning the Mistral/
        // NVIDIA branches give their own tables. `packages/ai/src/providers/together.models.ts`; every
        // entry there sets `compat.supportsReasoningEffort: false` regardless of the id's own
        // `reasoning: true/false` flag (mirrors NVIDIA's identical denylist elsewhere in this table),
        // so `reasoning_effort` stays `false` uniformly — only whether the family's real toggle
        // mechanism applies at all (`openai_reasoning_format`) varies by id.
        let together_match = match q {
            "qwen2.5-7b-instruct-turbo" => Some((32_768, 32_768, false, false)),
            "qwen3-235b-a22b-instruct-2507-tput" => Some((262_144, 262_144, false, false)),
            // pi-parity Task #17: this exact vendor-slug id is *also* served by HuggingFace under the
            // identical full string, with the same context (262144) but a real max_output of only 32768
            // (`huggingface.models.ts:295`) vs Together's own 130000 (`together.models.ts:81`) — a
            // same-string, no-host-signal collision (this table has no route/provider context to
            // disambiguate by). HuggingFace's smaller number wins: Together's real, larger ceiling is
            // now safely under-reported instead of HuggingFace's being dangerously over-reported.
            "qwen3.5-397b-a17b" => Some((262_144, 32_768, true, true)),
            "qwen3.5-9b" => Some((262_144, 65_536, true, true)),
            "qwen3.6-plus" => Some((1_000_000, 500_000, false, true)),
            "qwen3.7-max" => Some((1_000_000, 500_000, false, false)),
            _ => None,
        };
        if let Some((context_window, max_output, supports_vision, is_reasoning)) = together_match {
            return ModelCaps {
                context_window,
                max_output,
                max_tokens_field: MaxTokensField::MaxTokens,
                supports_long_cache: false,
                supports_vision,
                supports_temperature: true,
                thinking: ThinkingShape::None,
                reasoning_effort: false,
                reasoning_disableable: true,
                supports_eager_tool_streaming: false,
                supports_tool_stream: false,
                api: ApiKind::ChatCompletions,
                min_reasoning_effort: RE::Minimal,
                supports_xhigh_reasoning: true,
                adaptive_xhigh_effort_wire: "xhigh",
                // A non-reasoning id (`reasoning: false` in pi's catalogue) never gets the `together`
                // toggle sent at all (`openai-completions.ts`'s `thinkingFormat === "together" &&
                // model.reasoning` gate) — reporting `Together` format for one anyway would incorrectly
                // claim a mechanism that doesn't exist for it (`has_reasoning_mechanism`'s third arm).
                openai_reasoning_format: if is_reasoning {
                    OpenAiReasoningFormat::Together
                } else {
                    OpenAiReasoningFormat::Standard
                },
                supports_cache_control_on_tools: true,
            };
        }
        // pi-parity (models/dialects pass): "qwen3-next-80b-a3b-thinking" is HuggingFace-only (no
        // collision) with a real max_output of 131072 (`huggingface.models.ts`) — 2x this bucket's
        // generic 65536 default, context already correct at 262144.
        if q == "qwen3-next-80b-a3b-thinking" {
            return ModelCaps {
                context_window: 262_144,
                max_output: 131_072,
                max_tokens_field: MaxTokensField::MaxTokens,
                supports_long_cache: false,
                supports_vision: false,
                supports_temperature: true,
                thinking: ThinkingShape::None,
                reasoning_effort: false,
                reasoning_disableable: true,
                supports_eager_tool_streaming: false,
                supports_tool_stream: false,
                api: ApiKind::ChatCompletions,
                min_reasoning_effort: RE::Minimal,
                supports_xhigh_reasoning: true,
                adaptive_xhigh_effort_wire: "xhigh",
                openai_reasoning_format: OpenAiReasoningFormat::Together,
                supports_cache_control_on_tools: true,
            };
        }
        // pi-parity (models/dialects pass): "qwen3-235b-a22b-thinking-2507" is a genuine same-string
        // collision — OpenRouter's own vendor-slug id (`openrouter.models.ts`: real max_output 4096)
        // and HuggingFace's identically-spelled vendor-slug id (`huggingface.models.ts`: real 131072,
        // 32x larger) are the exact same full id string with no host signal in the id itself to
        // disambiguate. The host-agnostic default here matches OpenRouter's smaller, safer number (an
        // under-report on HuggingFace is a lost capability, not a 400); see
        // `capabilities_for_route_with_host` for the HuggingFace-specific bump to 131072 once a host
        // signal is actually available.
        if q == "qwen3-235b-a22b-thinking-2507" {
            return ModelCaps {
                context_window: 262_144,
                max_output: 4_096,
                max_tokens_field: MaxTokensField::MaxTokens,
                supports_long_cache: false,
                supports_vision: false,
                supports_temperature: true,
                thinking: ThinkingShape::None,
                reasoning_effort: false,
                reasoning_disableable: true,
                supports_eager_tool_streaming: false,
                supports_tool_stream: false,
                api: ApiKind::ChatCompletions,
                min_reasoning_effort: RE::Minimal,
                supports_xhigh_reasoning: true,
                adaptive_xhigh_effort_wire: "xhigh",
                openai_reasoning_format: OpenAiReasoningFormat::OpenRouter,
                supports_cache_control_on_tools: true,
            };
        }
        // pi-parity Task #28: pi's current catalogue lists ~7 more recent ids (Qwen3-Coder-30B/480B/
        // Next, Qwen3-Next-80B-A3B-Instruct/-Thinking, Qwen3.5-27B, Qwen3.6-27B/35B-A3B —
        // `huggingface.models.ts`) consistently at ~262144/65536, not this bucket's older 200000/40960
        // default — a systematic, safe-direction (under-report) refresh with no known collision (none
        // of these ids appear in `together.models.ts` under the same suffix).
        return ModelCaps {
            context_window: 262_144,
            max_output: 65_536,
            max_tokens_field: MaxTokensField::MaxTokens,
            supports_long_cache: false,
            // HuggingFace's Qwen3.5/3.6 lineup (`huggingface.models.ts`) is real vision-capable —
            // this generic fallback used to hardcode `false` unconditionally for the whole family. The
            // one exception already caught above (Together's own text-only "Qwen3.6-Plus"/"Qwen3.7-Max")
            // returns before ever reaching this default, so it's unaffected by widening this check.
            supports_vision: q.starts_with("qwen3.5") || q.starts_with("qwen3.6"),
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort: false,
            reasoning_disableable: true,
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::ChatCompletions,
            min_reasoning_effort: RE::Minimal,
            supports_xhigh_reasoning: true,
            adaptive_xhigh_effort_wire: "xhigh",
            openai_reasoning_format: OpenAiReasoningFormat::Together,
            supports_cache_control_on_tools: true,
        };
    }

    // Cerebras-native ids (no vendor-slug prefix — distinct from Groq's own "openai/gpt-oss-*",
    // matched below): all three current ids share the same 131k/40960 shape and take a bare
    // `reasoning_effort` (Cerebras isn't excluded from pi's `supportsReasoningEffort` default).
    if m == "gpt-oss-120b"
        || m == "gpt-oss-20b"
        || m == "gpt-oss-safeguard-20b"
        || m == "gemma-4-31b"
        || m == "zai-glm-4.7"
    {
        return ModelCaps {
            context_window: 131_072,
            max_output: 40_960,
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            supports_long_cache: true,
            supports_vision: m == "gemma-4-31b",
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort: true,
            reasoning_disableable: true,
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::ChatCompletions,
            min_reasoning_effort: RE::Minimal,
            supports_xhigh_reasoning: true,
            adaptive_xhigh_effort_wire: "xhigh",
            openai_reasoning_format: OpenAiReasoningFormat::Standard,
            supports_cache_control_on_tools: true,
        };
    }

    // GitHub Copilot's own "MAI-Code-1-Flash" id (`github-copilot.models.ts`) — doesn't match any
    // other family's id shape at all (not claude/gpt/deepseek/kimi/glm/grok/minimax/qwen/gpt-oss/
    // llama), so it fell all the way through to `ModelCaps::unknown()` before this entry existed. Text
    // only (pi: `input: ["text"]`); `reasoning: true` but `compat.supportsReasoningEffort: false`, so
    // no client-steerable reasoning toggle at all.
    if m == "mai-code-1-flash-picker" {
        return ModelCaps {
            context_window: 256_000,
            max_output: 128_000,
            max_tokens_field: MaxTokensField::MaxTokens,
            supports_long_cache: true,
            supports_vision: false,
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort: false,
            reasoning_disableable: false,
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::ChatCompletions,
            min_reasoning_effort: RE::Minimal,
            supports_xhigh_reasoning: false,
            adaptive_xhigh_effort_wire: "xhigh",
            openai_reasoning_format: OpenAiReasoningFormat::Standard,
            supports_cache_control_on_tools: true,
        };
    }

    // Groq-hosted open models under the vendor's own OpenAI-compatible endpoint. `openai/gpt-oss-*`
    // (Groq's id, vendor-prefixed — distinct from Cerebras's un-prefixed "gpt-oss-*" above) is
    // reasoning-capable; every Llama id on Groq is not.
    //
    // pi-parity Task #19, investigated and left as-is: "openai/gpt-oss-120b" specifically is a real id
    // on (at least) 4 providers with 4 different numbers — NVIDIA (128000/8192, `nvidia_caps` above,
    // which runs *before* this branch and unconditionally wins for this exact string — see its own
    // "must not inherit Groq's identically-named id's 65_536" regression test), Groq (real 65536,
    // `groq.models.ts:58`, this branch's own default below), Together (real 131072,
    // `together.models.ts:287`), and HuggingFace (real 32768, `huggingface.models.ts:655`). Since
    // `nvidia_caps` already intercepts this exact string unconditionally (a deliberate, tested
    // disambiguation for the NVIDIA-native case), a Groq/Together/HuggingFace request for the identical
    // id today actually gets NVIDIA's 8192 ceiling, not this branch's 65536 — already the *smallest* (and
    // therefore safest) of all four real numbers, not a dangerous over-report. Making this branch's own
    // default host-aware wouldn't even be reachable without first un-teaching `nvidia_caps` its own
    // correct, tested number for the NVIDIA-native case — the identical "one id, several real hosts, no
    // route signal to disambiguate" limitation documented throughout this table (see e.g. the
    // `nvidia/nemotron-3-ultra-550b-a55b` case in `nvidia_caps`'s own doc comment), just one level
    // deeper. Left unfixed rather than papering over it with a same-string check that would never
    // actually execute.
    if m.contains("gpt-oss") {
        return ModelCaps {
            context_window: 131_072,
            max_output: 65_536,
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            supports_long_cache: true,
            supports_vision: false,
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort: true,
            reasoning_disableable: true,
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::ChatCompletions,
            min_reasoning_effort: RE::Minimal,
            supports_xhigh_reasoning: true,
            adaptive_xhigh_effort_wire: "xhigh",
            openai_reasoning_format: OpenAiReasoningFormat::Standard,
            supports_cache_control_on_tools: true,
        };
    }
    if m.starts_with("llama") || m.contains("/llama") {
        // HuggingFace's own "meta-llama/Llama-3.3-70B-Instruct" has a real max_output (4_096) far
        // smaller than this branch's generic 32_768 default (8x over-report). Matched on the full id:
        // OpenRouter also lists a `"meta-llama/llama-3.3-70b-instruct"` entry, an identical full-string
        // collision this table can't disambiguate any further (unlike the DeepSeek/Llama-3.2-vision
        // fixes elsewhere in this pass, whose HuggingFace/NVIDIA ids carry a distinguishing prefix)
        // — HuggingFace's number wins here since that's this fix's specific target, at the accepted
        // cost of leaving OpenRouter's own (different, smaller: 16_384) real ceiling unrepresented for
        // this one id.
        if m == "meta-llama/llama-3.3-70b-instruct" {
            return ModelCaps {
                context_window: 131_072,
                max_output: 4_096,
                max_tokens_field: MaxTokensField::MaxCompletionTokens,
                supports_long_cache: true,
                supports_vision: false,
                supports_temperature: true,
                thinking: ThinkingShape::None,
                reasoning_effort: false,
                reasoning_disableable: false,
                supports_eager_tool_streaming: false,
                supports_tool_stream: false,
                api: ApiKind::ChatCompletions,
                min_reasoning_effort: RE::Minimal,
                supports_xhigh_reasoning: false,
                adaptive_xhigh_effort_wire: "xhigh",
                openai_reasoning_format: OpenAiReasoningFormat::Standard,
                supports_cache_control_on_tools: true,
            };
        }
        // pi-parity Task #26: the `-Turbo` suffixed Together id is a *different* string entirely from
        // the bare-native/OpenRouter-shaped exact match just above — no collision — and its real
        // numbers (131072/131072, `together.models.ts:212`) are much larger than this branch's generic
        // 32_768 default (a safe-direction under-report, fixed here for accuracy).
        if m == "meta-llama/llama-3.3-70b-instruct-turbo" {
            return ModelCaps {
                context_window: 131_072,
                max_output: 131_072,
                max_tokens_field: MaxTokensField::MaxCompletionTokens,
                supports_long_cache: true,
                supports_vision: false,
                supports_temperature: true,
                thinking: ThinkingShape::None,
                reasoning_effort: false,
                reasoning_disableable: false,
                supports_eager_tool_streaming: false,
                supports_tool_stream: false,
                api: ApiKind::ChatCompletions,
                min_reasoning_effort: RE::Minimal,
                supports_xhigh_reasoning: false,
                adaptive_xhigh_effort_wire: "xhigh",
                openai_reasoning_format: OpenAiReasoningFormat::Standard,
                supports_cache_control_on_tools: true,
            };
        }
        // pi-parity Task #24/#25: Groq's own bare ids, no collision with any other host's spelling.
        // "llama-4-scout-17b-16e-instruct" real max_output is 8192 (`groq.models.ts:41`) vs this
        // branch's generic 32_768 (a dangerous 4x over-report); "llama-3.1-8b-instant" real is 131072
        // (`groq.models.ts:7`), a safe-direction under-report fixed here for accuracy.
        if m == "meta-llama/llama-4-scout-17b-16e-instruct" {
            return ModelCaps {
                context_window: 131_072,
                max_output: 8_192,
                max_tokens_field: MaxTokensField::MaxCompletionTokens,
                supports_long_cache: true,
                supports_vision: true,
                supports_temperature: true,
                thinking: ThinkingShape::None,
                reasoning_effort: false,
                reasoning_disableable: false,
                supports_eager_tool_streaming: false,
                supports_tool_stream: false,
                api: ApiKind::ChatCompletions,
                min_reasoning_effort: RE::Minimal,
                supports_xhigh_reasoning: false,
                adaptive_xhigh_effort_wire: "xhigh",
                openai_reasoning_format: OpenAiReasoningFormat::Standard,
                supports_cache_control_on_tools: true,
            };
        }
        if m == "llama-3.1-8b-instant" {
            return ModelCaps {
                context_window: 131_072,
                max_output: 131_072,
                max_tokens_field: MaxTokensField::MaxCompletionTokens,
                supports_long_cache: true,
                supports_vision: false,
                supports_temperature: true,
                thinking: ThinkingShape::None,
                reasoning_effort: false,
                reasoning_disableable: false,
                supports_eager_tool_streaming: false,
                supports_tool_stream: false,
                api: ApiKind::ChatCompletions,
                min_reasoning_effort: RE::Minimal,
                supports_xhigh_reasoning: false,
                adaptive_xhigh_effort_wire: "xhigh",
                openai_reasoning_format: OpenAiReasoningFormat::Standard,
                supports_cache_control_on_tools: true,
            };
        }
        return ModelCaps {
            context_window: 131_072,
            max_output: 32_768,
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            supports_long_cache: true,
            supports_vision: m.contains("llama-4"),
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort: false,
            reasoning_disableable: false,
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::ChatCompletions,
            min_reasoning_effort: RE::Minimal,
            supports_xhigh_reasoning: false,
            adaptive_xhigh_effort_wire: "xhigh",
            openai_reasoning_format: OpenAiReasoningFormat::Standard,
            supports_cache_control_on_tools: true,
        };
    }

    // Google Gemini (`packages/ai/src/providers/google.models.ts`/`google-vertex.models.ts`) — this
    // table had zero branch for the family at all before now, so every bare `"gemini-*"` id (plus
    // Google's own bare, unprefixed `"gemma-4-26b-a4b-it"`/`"gemma-4-31b-it"` — distinct from the
    // "google/"-prefixed vendor-slug ids OpenRouter/HuggingFace host under the same model name, handled
    // separately just below) fell through to `ModelCaps::unknown()`'s flat 128k/4096 default: a
    // definitely-wrong data point even though nothing in this codebase actually speaks Gemini's wire
    // yet — the Gemini-direct dialect itself remains a deferred non-goal (no `Dialect` variant, no
    // `build_body`), so `thinking`/`reasoning_effort`/`openai_reasoning_format` below are inert
    // placeholders (kept at their safe "no mechanism" defaults deliberately, rather than reflecting
    // pi's own per-id `reasoning: true/false` flag, so an id here can never accidentally trigger the
    // OpenAI Chat Completions dialect's `reasoning_effort` emission if a Gemini id is ever misrouted
    // there before a real dialect exists) — this is data-only: just the context/output ceiling and
    // vision flag every dialect already reads regardless of wire shape.
    if m.starts_with("gemini") || m == "gemma-4-26b-a4b-it" || m == "gemma-4-31b-it" {
        let (context_window, max_output) = if m.starts_with("gemini-2.0") {
            // gemini-2.0-flash/-flash-lite: the two non-reasoning, smaller-output legacy ids.
            (1_048_576, 8_192)
        } else if m.starts_with("gemma-4") {
            (262_144, 32_768)
        } else {
            // Every gemini-2.5+/3.x id (flash, flash-lite, pro, and every dated/preview variant)
            // shares this shape.
            (1_048_576, 65_536)
        };
        return ModelCaps {
            context_window,
            max_output,
            max_tokens_field: MaxTokensField::MaxTokens,
            // Not in this table's Together/Cloudflare/NVIDIA/Ant-Ling `supportsLongCacheRetention`
            // denylist by name, but Gemini has no equivalent 1-hour prompt-cache concept in this
            // codebase at all yet — left conservatively `false` rather than claiming support for a
            // cache shape no dialect here builds.
            supports_long_cache: false,
            // Every current id (including both gemma-4 entries) is vision-capable — pi's `input` array
            // is `["text", "image"]` on all 16 `google.models.ts` entries and all 10
            // `google-vertex.models.ts` ones, no exception.
            supports_vision: true,
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort: false,
            reasoning_disableable: false,
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::ChatCompletions,
            min_reasoning_effort: RE::Minimal,
            supports_xhigh_reasoning: false,
            adaptive_xhigh_effort_wire: "xhigh",
            openai_reasoning_format: OpenAiReasoningFormat::Standard,
            supports_cache_control_on_tools: true,
        };
    }

    // HuggingFace's own "google/gemma-4-*-it" ids (`packages/ai/src/providers/huggingface.models.ts`)
    // don't match any family branch above at all — both fell through to the generic OpenRouter-shaped
    // vendor-slug fallback below, losing their real vision support entirely (`supports_vision: false`
    // there, vs both of these being real vision models). OpenRouter also lists a
    // `"google/gemma-4-*-it"` entry with yet another (also real, also different) max_output for each —
    // an id-only-keying collision this table can't resolve any further. NVIDIA's own
    // "stepfun-ai/step-3.5-flash"/"-3.7-flash" ids have the identical collision with HuggingFace (both
    // real, different numbers, no distinguishing prefix) — left at `nvidia_caps`'s existing, tested
    // numbers rather than re-litigated here, since that branch runs first and already wins the
    // tie-break for those two ids specifically.
    //
    // pi-parity Task #27 (investigated, kept as-is): Together *also* hosts the identical vendor-slug
    // "google/gemma-4-31B-it" string, with a real max_output of 131072 (`together.models.ts:193`) — 4x
    // HuggingFace's 32768 below. Since this table has no route/host signal to disambiguate Together
    // from HuggingFace/OpenRouter for this exact string (the same limitation as every other same-string
    // collision documented in this file), keeping HuggingFace's smaller number is the safe-direction
    // choice: Together's real, larger ceiling is under-reported (a usability loss, not a 400) rather
    // than HuggingFace's/OpenRouter's smaller ones being dangerously over-reported.
    //
    // pi-parity pass 20 Task 1: unlike "-31b-it" (where HuggingFace's 32768 is already the smallest of
    // the three real hosts and stays), "-26b-a4b-it"'s own OpenRouter real number (4096,
    // `openrouter.models.ts`) is *smaller* than HuggingFace's 32768 — an 8x over-report for OpenRouter
    // specifically. Narrowed to OpenRouter's smaller number for this one id only; "-31b-it" is
    // untouched.
    if m == "google/gemma-4-26b-a4b-it" || m == "google/gemma-4-31b-it" {
        let max_output = if m == "google/gemma-4-26b-a4b-it" {
            4_096
        } else {
            32_768
        };
        return ModelCaps {
            context_window: 262_144,
            max_output,
            max_tokens_field: MaxTokensField::MaxTokens,
            supports_long_cache: false,
            supports_vision: true,
            supports_temperature: true,
            thinking: ThinkingShape::None,
            // reasoning:true in pi's catalogue, and HuggingFace isn't in `detectCompat`'s
            // `supportsReasoningEffort` exclusion list (that denylist is Grok/Zai/Moonshot/Together/
            // Cloudflare-AI-Gateway/NVIDIA/Ant-Ling — not HuggingFace), so this gets the bare, permissive
            // `reasoning_effort` shape rather than one of the third-party toggle formats.
            reasoning_effort: true,
            reasoning_disableable: true,
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::ChatCompletions,
            min_reasoning_effort: RE::Minimal,
            supports_xhigh_reasoning: false,
            adaptive_xhigh_effort_wire: "xhigh",
            openai_reasoning_format: OpenAiReasoningFormat::Standard,
            supports_cache_control_on_tools: true,
        };
    }

    // pi-parity Task #29: Together's "essentialai/Rnj-1-Instruct" (`together.models.ts:175`) is a
    // brand-new model family with no other branch above recognizing it at all — falls to the generic
    // vendor-slug fallback below (128000/32000), a ~4x context over-report of its real, much smaller
    // 32768/32768. No collision with any other host's id.
    if m == "essentialai/rnj-1-instruct" {
        return ModelCaps {
            context_window: 32_768,
            max_output: 32_768,
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            supports_long_cache: false,
            supports_vision: false,
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort: false,
            reasoning_disableable: false,
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::ChatCompletions,
            min_reasoning_effort: RE::Minimal,
            supports_xhigh_reasoning: false,
            adaptive_xhigh_effort_wire: "xhigh",
            openai_reasoning_format: OpenAiReasoningFormat::Standard,
            supports_cache_control_on_tools: true,
        };
    }

    // pi-parity pass 20 Task 4: OpenRouter's own meta-routing pseudo-models (`openrouter.models.ts`) —
    // bare "auto" (no vendor slug at all) picks a model on your behalf; the vendor-slug-shaped
    // "openrouter/free"/"openrouter/fusion" (neither names a real hosting vendor — "openrouter" is
    // OpenRouter itself) round-robin across its free tier / a multi-model fusion router, respectively.
    // All three otherwise fell through to a *wrong* fallback: "auto" (no '/') went all the way to
    // `ModelCaps::unknown()`'s conservative 128k/4096; the other two hit the generic `m.contains('/')`
    // OpenRouter bucket below (128000/32000) — a genuine, dangerous over-report for "openrouter/free"
    // specifically (real max_output is smaller, 4096, not larger). Real numbers/shape all three share
    // with the generic OpenRouter bucket (`compat.thinkingFormat: "openrouter"`, no
    // `supportsReasoningEffort` override ⇒ permissive default) — only context/max_output/vision differ
    // per id.
    if m == "auto" || m == "openrouter/free" || m == "openrouter/fusion" {
        let (context_window, max_output, supports_vision) = match m.as_str() {
            "auto" => (2_000_000, 30_000, true),
            "openrouter/free" => (200_000, 4_096, true),
            _ => (1_000_000, 30_000, false),
        };
        return ModelCaps {
            context_window,
            max_output,
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            supports_long_cache: true,
            supports_vision,
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort: true,
            reasoning_disableable: true,
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::ChatCompletions,
            min_reasoning_effort: RE::Minimal,
            supports_xhigh_reasoning: true,
            adaptive_xhigh_effort_wire: "xhigh",
            openai_reasoning_format: OpenAiReasoningFormat::OpenRouter,
            supports_cache_control_on_tools: true,
        };
    }

    // pi-parity pass 20 Task 4: OpenCode Zen's own branded free-tier ids (`opencode.models.ts`) — bare,
    // unprefixed, and matching no other family's id shape above, so all three fell straight through to
    // `ModelCaps::unknown()`'s conservative 128k/4096 default. Real numbers ported id-for-id; all three
    // share an identical compat shape in pi's catalogue (`"maxTokensField":"max_tokens"`, no
    // `thinkingFormat` override ⇒ the plain OpenAI-style bare `reasoning_effort` shape, not the nested
    // OpenRouter one) — distinct from the "auto"/"openrouter/free"/"openrouter/fusion" trio just above,
    // which genuinely are OpenRouter-shaped. None of the three are vision-capable (`input: ["text"]`).
    if m == "big-pickle" || m == "nemotron-3-ultra-free" || m == "north-mini-code-free" {
        let (context_window, max_output) = match m.as_str() {
            "big-pickle" => (200_000, 32_000),
            "nemotron-3-ultra-free" => (1_000_000, 128_000),
            _ => (256_000, 64_000),
        };
        return ModelCaps {
            context_window,
            max_output,
            max_tokens_field: MaxTokensField::MaxTokens,
            supports_long_cache: true,
            supports_vision: false,
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort: true,
            reasoning_disableable: true,
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::ChatCompletions,
            min_reasoning_effort: RE::Minimal,
            supports_xhigh_reasoning: true,
            adaptive_xhigh_effort_wire: "xhigh",
            openai_reasoning_format: OpenAiReasoningFormat::Standard,
            supports_cache_control_on_tools: true,
        };
    }

    // OpenRouter — and the generic fallback for any other vendor/model-shaped id (`vendor/model`) this
    // table doesn't otherwise recognize (most commonly an uncatalogued Together/Groq/Cerebras
    // addition). pi tags virtually every OpenRouter catalogue entry `compat.thinkingFormat:
    // "openrouter"` with `supportsReasoningEffort` left at its permissive default (`true`); a nested
    // `reasoning:{effort}` object is a reasonably safe shape to send even to a provider that doesn't
    // recognize it (an extra top-level field a permissive JSON API typically ignores, rather than a
    // field substitution that could 400 outright). Deliberately last, so a recognizable family
    // (deepseek/glm/kimi/grok/qwen/gpt-oss/llama) always gets its own real numbers first even when
    // reached with a vendor-slug id.
    if m.contains('/') {
        return ModelCaps {
            context_window: 128_000,
            max_output: 32_000,
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            supports_long_cache: true,
            supports_vision: false,
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort: true,
            reasoning_disableable: true,
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::ChatCompletions,
            min_reasoning_effort: RE::Minimal,
            supports_xhigh_reasoning: true,
            adaptive_xhigh_effort_wire: "xhigh",
            openai_reasoning_format: OpenAiReasoningFormat::OpenRouter,
            supports_cache_control_on_tools: true,
        };
    }

    tracing::warn!(
        model,
        "unrecognized model id; falling back to conservative capabilities"
    );
    ModelCaps::unknown()
}

/// A third-party aggregator platform a request may be routed through — the host/route analogue of
/// [`ApiKind`]'s wire-format distinction, for the class of bug this table's own section-header doc
/// comment documents repeatedly: the same bare or vendor-slug model id served by two or more of these
/// hosts, each with genuinely different real numbers, that [`capabilities`] alone (keyed purely on the
/// id string) can't disambiguate. Consumed by [`capabilities_for_route_with_host`]; see
/// [`crate::transport::ModelRequest::host`]'s own doc comment for how (and how much of) this is
/// actually populated today.
///
/// Not every variant here is a genuine same-string collision in practice — Fireworks ids, for
/// instance, are already self-identifying by shape (`is_fireworks_model`) and need no host signal at
/// all — but all nine are named here together so the one mechanism covers every aggregator platform
/// this codebase routes to, rather than growing a new one-off per host as each collision is found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregatorHost {
    /// `api.together.ai` (`crates/gateway/src/route.rs`'s `"together"` `KNOWN_PROVIDERS` row).
    Together,
    /// `router.huggingface.co` — BYO-only (named via `ModelOverride::base_url`; no gateway-native
    /// route exists for it).
    HuggingFace,
    /// `api.groq.com` (`"groq"` `KNOWN_PROVIDERS` row).
    Groq,
    /// `api.fireworks.ai` (`"fireworks"` `KNOWN_PROVIDERS` row) — already self-identifying by id shape
    /// (`is_fireworks_model`); included for completeness of the mechanism, not because a host signal is
    /// actually required to recognize one.
    Fireworks,
    /// NVIDIA NIM — BYO-only.
    Nvidia,
    /// `openrouter.ai` (`"openrouter"` `KNOWN_PROVIDERS` row).
    OpenRouter,
    /// `api.kimi.com/coding` — BYO-only, and already disambiguated from moonshotai-native by a
    /// `ModelOverride::dialect`/`base_url` override rather than by id shape (see `dialect/mod.rs`'s own
    /// Kimi-Coding handling) — included here too since its capability numbers still benefit from a
    /// host signal the same way every other aggregator's do.
    KimiCoding,
    /// `opencode.ai/zen`(`/v1`) — pi's `opencode.models.ts` — BYO-only, named via
    /// `ModelOverride::base_url`. Distinct from [`Self::OpenCodeGo`] below: both are nested under the
    /// same registered domain (`opencode.ai`), but genuinely disagree on real numbers/wire dialect for
    /// a handful of shared bare ids (`"minimax-m3"`, `"glm-5.1"`) — a hostname-only check the way every
    /// other variant here uses can't tell the two apart; see
    /// `crates/agent::gateway_credential::aggregator_host_for_base_url`'s own doc comment for how the
    /// `/zen` vs `/zen/go` path segment disambiguates them.
    OpenCodeZen,
    /// `opencode.ai/zen/go`(`/v1`) — pi's `opencode-go.models.ts`. See [`Self::OpenCodeZen`]'s own doc
    /// comment for why this needs to be a separate variant rather than folded into it.
    OpenCodeGo,
}

/// Route-aware capability override for the handful of OpenAI ids whose real numbers/thinking-map
/// diverge by which route serves them — native (`packages/ai/src/providers/openai.models.ts`), OpenAI
/// Codex (`openai-codex.models.ts`), or Azure OpenAI Responses (`azure-openai-responses.models.ts`) —
/// despite [`capabilities`] being keyed purely on the bare model id with no route context at all.
/// Reuses [`crate::transport::ModelRequest`]'s own `is_codex`/`is_azure` flags (already resolved once
/// in `GatewayClient::stream` from the credential's route shape — a prior pass's own Codex/Azure work,
/// `openai_responses.rs`'s `build_body` already special-cases both) rather than inventing a second
/// route-detection mechanism — mirrors [`crate::dialect::Dialect::for_model_via_copilot`]'s identical
/// shape: a thin route-aware wrapper around the plain, route-blind lookup every other caller keeps
/// using unchanged. `is_codex`/`is_azure` are never both `true` for a real request (a request is routed
/// to exactly one place); both `false` (the overwhelming common case — native OpenAI, or any
/// non-OpenAI provider) is a complete no-op, returning [`capabilities`]'s own answer untouched.
///
/// Only 4 ids currently diverge in their raw numbers (verified against pi's three catalogues above):
/// `gpt-5.3-codex-spark` (Codex: 128k max output and no vision, vs native/Azure's 32k/vision-capable —
/// Codex's own entry is genuinely `input: ["text"]`, not `["text","image"]`), `gpt-5.4`/`gpt-5.5`
/// (Azure: a 1.05M context, vs native/Codex's 272k), and `gpt-5.4-mini` (Codex: a 272k context, vs
/// native/Azure's 400k). Separately, every id in the native `GPT5_DISABLE_CAPABLE` allowlist
/// (`gpt-5.1`/`gpt-5.2`/`gpt-5.3-codex`/`gpt-5.4`/`gpt-5.4-mini`/`gpt-5.4-nano`/`gpt-5.5`) loses its
/// explicit "off" wire signal entirely on both other routes — Azure nulls `thinkingLevelMap.off`
/// outright, and Codex's own catalogue simply omits the key rather than mapping it to a literal wire
/// string the way native's `"none"` does — so none of them can be told to disable reasoning explicitly
/// on either non-native route, even though the exact same id can natively.
///
/// Wired into every call site that already holds a `req: &ModelRequest` (carrying both flags):
/// `dialect::openai_responses::build_body`, `dialect::openai::build_body`,
/// `dialect::anthropic::build_body` (a no-op there in practice — see that call site's own comment),
/// and `client.rs`'s `needs_fine_grained_tool_streaming_beta` gate. `agent.rs`'s several callers
/// (`Agent::new` and others) call the plain `capabilities(&model)` still — they hold only a bare
/// model `String`, never a `ModelRequest`, so there's no `is_codex`/`is_azure` in scope to thread
/// through there.
pub fn capabilities_for_route(model: &str, is_codex: bool, is_azure: bool) -> ModelCaps {
    let mut caps = capabilities(model);
    let m = model.to_ascii_lowercase();
    match m.as_str() {
        "gpt-5.3-codex-spark" if is_codex => {
            caps.max_output = 128_000;
            caps.supports_vision = false;
        }
        "gpt-5.4" if is_azure => caps.context_window = 1_050_000,
        "gpt-5.4-mini" if is_codex => caps.context_window = 272_000,
        "gpt-5.5" if is_azure => caps.context_window = 1_050_000,
        _ => {}
    }
    // The 7 ids natively disable-capable (`GPT5_DISABLE_CAPABLE` in `capabilities`'s own gpt-5 branch)
    // all lose that explicit "off" signal on either non-native route — see this function's own doc
    // comment for the Azure-null vs Codex-key-absent distinction (functionally identical here: neither
    // gives this table a literal wire value to send for "off").
    const NOT_DISABLEABLE_OFF_NATIVE_ROUTE: &[&str] = &[
        "gpt-5.1",
        "gpt-5.2",
        "gpt-5.3-codex",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5.5",
    ];
    if (is_codex || is_azure) && NOT_DISABLEABLE_OFF_NATIVE_ROUTE.contains(&m.as_str()) {
        caps.reasoning_disableable = false;
    }
    caps
}

/// Same as [`capabilities_for_route`], additionally threading `is_copilot` for the handful of ids
/// GitHub Copilot's own proxy serves with real numbers that diverge from every other route
/// (`packages/ai/src/providers/github-copilot.models.ts`) — mirrors that function's own `is_codex`/
/// `is_azure` shape (and `Dialect::for_model_via_copilot`'s identical idea on the dialect side). A
/// separate function, rather than a 4th parameter on `capabilities_for_route` itself, so `client.rs`'s
/// existing 3-arg call site (which has no `is_copilot` in scope to thread through) is unaffected; only
/// the two callers that already hold `req.is_copilot` (`dialect::openai_responses::build_body`,
/// `dialect::openai::build_body`) need this variant — both now call it *indirectly*, through
/// [`capabilities_for_route_with_host`] (which also threads `req.host`), rather than directly.
///
/// Currently overrides 7 ids (pi-parity Tasks #8, #9, #10, #11): `gpt-4.1` (Copilot's own dialect
/// routing already forces Chat Completions for this one — see
/// [`crate::dialect`]'s `COPILOT_FORCES_CHAT_COMPLETIONS` — with a real 128k/16384 ceiling vs native's
/// ~1.05M/32768), `gpt-5-mini` (264000/64000 vs native's 400000/128000), `gpt-5.4`/`gpt-5.5` (400000
/// context, not native's 272000), and the 4 Copilot-hosted Gemini ids: `gemini-2.5-pro`/
/// `gemini-3-flash-preview` (128000/64000) and `gemini-3.1-pro-preview`/`gemini-3.5-flash`
/// (200000/64000), vs native Google's 1,048,576/65,536 — [`capabilities`]'s own Gemini branch has zero
/// Copilot-route awareness at all otherwise (no dialect actually serves Gemini's wire yet, so this is
/// data-only, same as every other Gemini entry in that branch — see its own doc comment).
pub fn capabilities_for_route_with_copilot(
    model: &str,
    is_codex: bool,
    is_azure: bool,
    is_copilot: bool,
) -> ModelCaps {
    let mut caps = capabilities_for_route(model, is_codex, is_azure);
    if !is_copilot {
        return caps;
    }
    let m = model.to_ascii_lowercase();
    match m.as_str() {
        "gpt-4.1" => {
            caps.context_window = 128_000;
            caps.max_output = 16_384;
        }
        "gpt-5-mini" => {
            caps.context_window = 264_000;
            caps.max_output = 64_000;
        }
        "gpt-5.4" | "gpt-5.5" => caps.context_window = 400_000,
        "gemini-2.5-pro" | "gemini-3-flash-preview" => {
            caps.context_window = 128_000;
            caps.max_output = 64_000;
        }
        "gemini-3.1-pro-preview" | "gemini-3.5-flash" => {
            caps.context_window = 200_000;
            caps.max_output = 64_000;
        }
        _ => {}
    }
    caps
}

/// Same as [`capabilities_for_route_with_copilot`], additionally threading [`AggregatorHost`] for the
/// handful of ids whose real numbers genuinely depend on *which aggregator* serves them, not just
/// which OpenAI route (Codex/Azure/Copilot) does — see [`AggregatorHost`]'s own doc comment for why
/// this is a separate signal from those three. A separate function rather than a 5th parameter on
/// `capabilities_for_route_with_copilot` itself, for the identical reason that function gives for
/// being separate from `capabilities_for_route`: most callers have no host in scope to thread through.
///
/// `host: None` (still the common case for a plain gateway-relayed request with no BYO `base_url`
/// override — see [`crate::transport::ModelRequest::host`]'s own doc comment for exactly which routes
/// populate it today, as of pi-parity pass 20's `crates/agent::gateway_credential` wiring) is a
/// complete no-op, returning [`capabilities_for_route_with_copilot`]'s own answer untouched — every fix
/// below only ever narrows an *already* host-ambiguous case, never changes the host-agnostic default.
///
/// Currently overrides 7 genuine same-string collisions:
/// `"qwen/qwen3-235b-a22b-thinking-2507"` — the identical id string on both OpenRouter (real
/// `max_output` 4096, matching this table's own host-agnostic default) and HuggingFace (real 131072,
/// 32x larger) — and `"openai/gpt-oss-20b"` — at least a 4-way collision (NVIDIA/Together/Groq/
/// OpenRouter, each with a different real `max_output`); the host-agnostic default silently returns
/// NVIDIA's number today (`nvidia_caps` intercepts this exact string first, unconditionally — see its
/// own doc comment), but this override still corrects the Together case regardless, since it runs
/// *after* that base lookup rather than trying to out-order it. Every other concrete collision the
/// models/dialects pass fixed turned out to be resolvable without a host signal at all (the vendor-slug
/// id string itself was already host-unique within this codebase's reachable provider set) — see
/// `capabilities`'s own per-family doc comments for those.
///
/// pi-parity pass 20 Task 5 adds 5 more, all OpenCode Zen/OpenCode-Go bare-id collisions (the two
/// aggregators disagree with each other, and/or with this table's host-agnostic default, for these
/// specific ids — see [`AggregatorHost::OpenCodeZen`]'s own doc comment): `"kimi-k2.5"`/`"kimi-k2.6"`
/// (real max_output 65536 on both OpenCode Zen and OpenCode-Go, vs this table's host-agnostic 262144),
/// `"glm-5.1"` (real max_output 32768 on OpenCode-Go specifically — OpenCode Zen's own real number,
/// 131072, already matches the host-agnostic default), and `"minimax-m3"` (real context 512000 on
/// OpenCode Zen specifically — OpenCode-Go's own real context, 1,000,000, already matches the
/// host-agnostic default).
pub fn capabilities_for_route_with_host(
    model: &str,
    is_codex: bool,
    is_azure: bool,
    is_copilot: bool,
    host: Option<AggregatorHost>,
) -> ModelCaps {
    let mut caps = capabilities_for_route_with_copilot(model, is_codex, is_azure, is_copilot);
    let m = model.to_ascii_lowercase();
    match (host, m.as_str()) {
        (Some(AggregatorHost::HuggingFace), "qwen/qwen3-235b-a22b-thinking-2507") => {
            caps.max_output = 131_072;
        }
        (Some(AggregatorHost::Together), "openai/gpt-oss-20b") => {
            caps.max_output = 131_072;
        }
        // pi-parity pass 20 Task 5: OpenCode Zen's and OpenCode-Go's real numbers for these bare ids
        // (`opencode.models.ts`/`opencode-go.models.ts`) — see this function's own doc comment for the
        // cross-host survey each arm below corrects.
        (Some(AggregatorHost::OpenCodeZen), "kimi-k2.5" | "kimi-k2.6") => {
            caps.max_output = 65_536;
        }
        (Some(AggregatorHost::OpenCodeGo), "kimi-k2.6") => {
            caps.max_output = 65_536;
        }
        (Some(AggregatorHost::OpenCodeGo), "glm-5.1") => {
            caps.max_output = 32_768;
        }
        (Some(AggregatorHost::OpenCodeZen), "minimax-m3") => {
            caps.context_window = 512_000;
        }
        _ => {}
    }
    caps
}

/// Per-model reasoning-effort wire-string remap for the OpenAI Chat-Completions dialect's third-party
/// reasoning-toggle shapes (`dialect::openai::apply_reasoning_wire`) — mirrors pi's per-model
/// `thinkingLevelMap` (`packages/ai/src/providers/*.models.ts`), which several providers use to spell a
/// given portable [`crate::transport::ReasoningEffort`] rung differently on the wire than that rung's
/// own name. [`ModelCaps::adaptive_xhigh_effort_wire`] already exists for the identical need on the
/// Anthropic adaptive-thinking shape, but it's a single xhigh-only string — insufficient here, where a
/// model can remap several rungs at once (GLM-5.2 remaps three: low, medium, and xhigh) — so this is a
/// standalone lookup instead of a new `ModelCaps` field, which would need touching every one of this
/// file's ~25 existing `ModelCaps` struct-literal construction sites for a table only four families
/// actually use.
///
/// `effort` is the *already-clamped* [`crate::transport::ReasoningEffort`] ([`clamp_reasoning_effort`]'s
/// output) — this function only remaps how an already-legal level is spelled on the wire, never decides
/// whether a level is legal in the first place (that's still `min_reasoning_effort`/
/// `supports_xhigh_reasoning`). `None` — the overwhelming common case — leaves the caller to fall back
/// to the clamped effort's own literal name, unchanged from before this function existed.
pub fn reasoning_wire_override(
    model: &str,
    effort: crate::transport::ReasoningEffort,
) -> Option<&'static str> {
    use crate::transport::ReasoningEffort as RE;
    let m = model.to_ascii_lowercase();
    // Matched the same way `capabilities` matches its own family branches — see `family_id`'s doc
    // comment there — so a vendor-slug id (`"zai-org/glm-5.2"`, Together/HuggingFace) remaps
    // identically to the bare id it's slug-prefixed with, instead of silently keeping the clamped
    // effort's own literal name just because the full id doesn't start with the family's prefix.
    let family_id: &str = match m.rfind('/') {
        Some(idx) => &m[idx + 1..],
        None => &m,
    };
    // DeepSeek: `thinkingLevelMap` nulls minimal/low/medium (already reflected in
    // `min_reasoning_effort: High`, so only High/XHigh can ever reach here); xhigh alone remaps, to
    // "max".
    if m.starts_with("deepseek") || family_id.starts_with("deepseek") {
        return (effort == RE::XHigh).then_some("max");
    }
    // GLM-5.2+: low/medium/high all collapse to the literal "high", and xhigh remaps to "max". Every
    // earlier GLM id has no reasoning_effort vocabulary at all (`caps.reasoning_effort == false`), so
    // `apply_reasoning_wire` never calls this for them regardless.
    if m.starts_with("glm-5.2") || family_id.starts_with("glm-5.2") {
        return match effort {
            RE::Low | RE::Medium | RE::High => Some("high"),
            RE::XHigh => Some("max"),
            // Excluded by `min_reasoning_effort: Low` above — never actually reaches here.
            RE::Minimal => None,
        };
    }
    // Groq's one qwen id: the only level it ever accepts (`min_reasoning_effort: High`,
    // `supports_xhigh_reasoning: false` clamp everything else away first) remaps from "high" to
    // "default".
    if m == "qwen/qwen3-32b" {
        return (effort == RE::High).then_some("default");
    }
    // Mistral reasoning-capable ids: pi's own client (`mistral-conversations.ts::mapReasoningEffort`)
    // has no per-level vocabulary for any Mistral id at all — none carry a `thinkingLevelMap` — so
    // every active level falls back to that function's own hardcoded default, "high", literally
    // regardless of which portable level was requested; only "off" ever omits the field. Mirrored here
    // rather than sending "minimal"/"low"/etc, which Mistral's real `reasoning_effort` wire vocabulary
    // (a bare `"none" | "high"` enum) doesn't recognize at all. Only ever reached when
    // `caps.reasoning_effort` is already true for the id (the caller's own gate), so no non-reasoning
    // Mistral id (e.g. "mistral-large-latest") ever hits this branch.
    if is_mistral_id(&m) {
        return Some("high");
    }
    None
}

/// Per-model "minimal"-effort wire remap for the OpenAI **Responses** dialect's Codex/Copilot routes
/// (`dialect::openai_responses::build_body`) — the Responses-dialect sibling of
/// [`reasoning_wire_override`] (which only covers the Chat-Completions dialect's third-party toggle
/// shapes). pi's `openai-codex.models.ts` and `github-copilot.models.ts` both set
/// `thinkingLevelMap: {"minimal":"low", …}` for these gpt-5 ids — neither catalogue defines a literal
/// "minimal" tier at all, unlike native OpenAI's own map (pi-parity Task #13).
///
/// `REMAPPED_IDS` mixes two different real catalogues, matched against the *same* gating condition
/// (`is_codex || is_copilot`) — safe to share one list rather than split it by route: `gpt-5.3-codex-
/// spark`/`gpt-5.4`/`gpt-5.4-mini`/`gpt-5.5` are the 4 ids `openai-codex.models.ts` actually carries
/// (reachable via either route); `gpt-5-mini`/`gpt-5.2`/`gpt-5.2-codex`/`gpt-5.3-codex`/`gpt-5.4-nano`
/// exist **only** in `github-copilot.models.ts` (pi-parity, models/dialects pass) — Codex's own
/// catalogue has no such ids to route to in the first place, so listing them unconditionally can never
/// fire on a real Codex-routed request.
///
/// `effort` is the *already-clamped* [`crate::transport::ReasoningEffort`] ([`clamp_reasoning_effort`]'s
/// output), matching [`reasoning_wire_override`]'s own contract. `None` — the overwhelming common
/// case (a non-"minimal" effort, a route that's neither Codex nor Copilot, or any other model) — leaves
/// the caller to fall back to the clamped effort's own literal name unchanged.
pub fn responses_minimal_effort_wire_override(
    model: &str,
    is_codex: bool,
    is_copilot: bool,
    effort: crate::transport::ReasoningEffort,
) -> Option<&'static str> {
    use crate::transport::ReasoningEffort as RE;
    if effort != RE::Minimal || !(is_codex || is_copilot) {
        return None;
    }
    const REMAPPED_IDS: &[&str] = &[
        "gpt-5.3-codex-spark",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.5",
        "gpt-5-mini",
        "gpt-5.2",
        "gpt-5.2-codex",
        "gpt-5.3-codex",
        "gpt-5.4-nano",
    ];
    let m = model.to_ascii_lowercase();
    REMAPPED_IDS.contains(&m.as_str()).then_some("low")
}

/// A portable thinking-depth level, independent of which wire mechanism the active model actually uses
/// — an Anthropic token budget (`ThinkingShape::Budget`), an Anthropic adaptive effort
/// (`ThinkingShape::Adaptive`), or an OpenAI `reasoning_effort` parameter. A client (or
/// `cycle_thinking_level`) can move through these six rungs and land at a comparable depth no matter
/// which model is currently active — see [`thinking_for_level`] for the translation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingLevel {
    Off,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
}

/// Ladder order `ThinkingLevel::next` cycles through, wrapping from `XHigh` back to `Off`.
const THINKING_LEVEL_LADDER: [ThinkingLevel; 6] = [
    ThinkingLevel::Off,
    ThinkingLevel::Minimal,
    ThinkingLevel::Low,
    ThinkingLevel::Medium,
    ThinkingLevel::High,
    ThinkingLevel::XHigh,
];

impl ThinkingLevel {
    /// The wire string this level round-trips as: `"off"`, plus `ReasoningEffort::as_str`'s vocabulary
    /// for the other five.
    pub fn as_str(self) -> &'static str {
        match self {
            ThinkingLevel::Off => "off",
            ThinkingLevel::Minimal => "minimal",
            ThinkingLevel::Low => "low",
            ThinkingLevel::Medium => "medium",
            ThinkingLevel::High => "high",
            ThinkingLevel::XHigh => "xhigh",
        }
    }

    /// Parse [`Self::as_str`]'s vocabulary (case-sensitive, matching `ReasoningEffort`'s own wire
    /// strings). `None` for anything else.
    pub fn parse(s: &str) -> Option<Self> {
        THINKING_LEVEL_LADDER
            .iter()
            .copied()
            .find(|l| l.as_str() == s)
    }

    /// The next rung, wrapping from `XHigh` back to `Off`.
    pub fn next(self) -> Self {
        let idx = THINKING_LEVEL_LADDER
            .iter()
            .position(|&l| l == self)
            .unwrap_or(0);
        THINKING_LEVEL_LADDER[(idx + 1) % THINKING_LEVEL_LADDER.len()]
    }

    /// The `ReasoningEffort` this level carries, or `None` for `Off`.
    pub fn reasoning_effort(self) -> Option<crate::transport::ReasoningEffort> {
        use crate::transport::ReasoningEffort as RE;
        match self {
            ThinkingLevel::Off => None,
            ThinkingLevel::Minimal => Some(RE::Minimal),
            ThinkingLevel::Low => Some(RE::Low),
            ThinkingLevel::Medium => Some(RE::Medium),
            ThinkingLevel::High => Some(RE::High),
            ThinkingLevel::XHigh => Some(RE::XHigh),
        }
    }
}

impl From<crate::transport::ReasoningEffort> for ThinkingLevel {
    fn from(effort: crate::transport::ReasoningEffort) -> Self {
        use crate::transport::ReasoningEffort as RE;
        match effort {
            RE::Minimal => ThinkingLevel::Minimal,
            RE::Low => ThinkingLevel::Low,
            RE::Medium => ThinkingLevel::Medium,
            RE::High => ThinkingLevel::High,
            RE::XHigh => ThinkingLevel::XHigh,
        }
    }
}

/// The built-in effort→token-budget ladder, before any caller override or the `max_output` clamp.
/// Factored out of [`budget_for_effort`]/[`budget_for_effort_with_override`] so there's exactly one
/// literal source of truth for the fixed numbers.
fn default_budget_for_effort(effort: crate::transport::ReasoningEffort) -> u32 {
    use crate::transport::ReasoningEffort as RE;
    match effort {
        RE::Minimal => 1_024,
        RE::Low => 2_048,
        RE::Medium => 8_192,
        // pi: `packages/ai/src/api/simple-options.ts`'s `defaultBudgets.high` is 16384, not 24000.
        RE::High => 16_384,
        RE::XHigh => 32_000,
    }
}

/// A fixed effort→token-budget ladder for [`ThinkingShape::Budget`]-shape models, clamped below
/// `max_output` (a thinking budget must leave room for the turn's actual output). For
/// [`ThinkingShape::Adaptive`]-shape models this value is a pure on/off gate on the wire — only its
/// `Some`-ness is checked, the actual depth comes from `output_config.effort` (see
/// `dialect::anthropic::build_body`) — so its exact magnitude doesn't matter there, but reusing the same
/// ladder keeps one code path instead of two.
///
/// [`thinking_for_level`] (the only internal caller) always uses the built-in ladder — equivalent to
/// calling [`budget_for_effort_with_override`] with `overrides: None`. A caller that has an
/// operator-supplied override table should call that sibling directly instead.
fn budget_for_effort(effort: crate::transport::ReasoningEffort, max_output: u32) -> u32 {
    budget_for_effort_with_override(effort, max_output, None)
}

/// Same as [`budget_for_effort`], but consults `overrides` first — the extension point a settings/CLI
/// layer can wire an operator-supplied thinking-token budget through (e.g. `--thinking-budget
/// high=40000`), without this crate needing any opinion on where that table comes from. An effort level
/// missing from `overrides` (or `overrides: None` entirely) falls back to
/// [`default_budget_for_effort`]'s fixed ladder unchanged. The `max_output` clamp applies either way —
/// an override still can't leave a turn with zero room for its actual output.
pub fn budget_for_effort_with_override(
    effort: crate::transport::ReasoningEffort,
    max_output: u32,
    overrides: Option<&std::collections::HashMap<crate::transport::ReasoningEffort, u32>>,
) -> u32 {
    let budget = overrides
        .and_then(|table| table.get(&effort))
        .copied()
        .unwrap_or_else(|| default_budget_for_effort(effort));
    budget.min(max_output.saturating_sub(1).max(1))
}

/// Clamp a requested [`ReasoningEffort`](crate::transport::ReasoningEffort) to what `caps`'s model
/// actually accepts on the wire — pi's `clampThinkingLevel`, specialized to the two edges every current
/// model's exclusions actually trim (never a gap in the middle): a floor (`min_reasoning_effort`, e.g.
/// `gpt-5.5-pro` rejects `minimal`/`low`) and an `xhigh` ceiling (`supports_xhigh_reasoning`, e.g.
/// o-series/bare-gpt-5/gpt-5.1\* and Anthropic sonnet-4-6/sonnet-5 top out at `high`). Called by every
/// dialect at the point it's about to put an effort string on the wire, rather than by
/// [`thinking_for_level`], so a raw `with_reasoning_effort`/`--reasoning-effort` call (which never goes
/// through `thinking_for_level`) is clamped too.
pub fn clamp_reasoning_effort(
    caps: &ModelCaps,
    effort: crate::transport::ReasoningEffort,
) -> crate::transport::ReasoningEffort {
    use crate::transport::ReasoningEffort as RE;
    let mut e = effort.max(caps.min_reasoning_effort);
    if e == RE::XHigh && !caps.supports_xhigh_reasoning {
        e = RE::High;
    }
    e
}

/// Whether `caps`'s model has *any* client-steerable thinking/reasoning mechanism at all — an
/// Anthropic `thinking` shape, a bare OpenAI `reasoning_effort`, or one of the third-party
/// [`OpenAiReasoningFormat`] toggle shapes (DeepSeek/Zai/Together/OpenRouter). The third arm matters
/// because several of those families (Kimi, GLM below 5.2) set `reasoning_effort: false` — they have no
/// graduated effort vocabulary — while still exposing a real on/off toggle
/// (`dialect::openai::build_body`'s family-specific branches send it regardless of `reasoning_effort`);
/// without this arm such a model would be misreported as having no mechanism at all, `Off`-locked, even
/// though `--thinking high` genuinely turns its reasoning on. Shared by [`available_thinking_levels`],
/// [`clamp_thinking_level`], and [`thinking_for_level`] so the three stay consistent with each other.
///
/// `pub` (not module-private): `crates/agent`'s own `default_reasoning_effort_for_model`/`model_info`
/// call this directly instead of maintaining their own narrower copy of the same check. A prior version
/// of both did exactly that — checking only `reasoning_effort`/`thinking` — which silently missed the
/// third arm above: Kimi-thinking and pre-5.2 GLM models (`openai_reasoning_format` toggle shapes with
/// `reasoning_effort: false` and `thinking: ThinkingShape::None`) read as "no reasoning mechanism at
/// all" under that narrower check, so a bare invocation picked no default reasoning effort for them at
/// all, even though pi's own reference defaults every reasoning-capable model — this family included —
/// to medium effort.
pub fn has_reasoning_mechanism(caps: &ModelCaps) -> bool {
    caps.reasoning_effort
        || caps.thinking != ThinkingShape::None
        || caps.openai_reasoning_format != OpenAiReasoningFormat::Standard
}

/// The [`ThinkingLevel`] rungs `caps`'s model actually accepts, in ladder order — `Off` excluded when
/// the model has a thinking/reasoning mechanism it can't explicitly disable
/// (`reasoning_disableable == false`), `XHigh` excluded when `!supports_xhigh_reasoning`, and any rung
/// below `min_reasoning_effort` excluded. A model with no thinking/reasoning mechanism at all
/// (see [`has_reasoning_mechanism`]) has exactly one available level, `Off` — nothing else could ever
/// mean anything to it. Used by [`clamp_thinking_level`] (via containment) and
/// [`next_available_thinking_level`] (to cycle only through what's real for this model, instead of the
/// raw 6-rung ladder).
pub fn available_thinking_levels(caps: &ModelCaps) -> Vec<ThinkingLevel> {
    if !has_reasoning_mechanism(caps) {
        return vec![ThinkingLevel::Off];
    }
    THINKING_LEVEL_LADDER
        .iter()
        .copied()
        .filter(|&level| match level {
            ThinkingLevel::Off => caps.reasoning_disableable,
            ThinkingLevel::XHigh => caps.supports_xhigh_reasoning,
            _ => level
                .reasoning_effort()
                .is_some_and(|effort| effort >= caps.min_reasoning_effort),
        })
        .collect()
}

/// Clamp a portable [`ThinkingLevel`] to the nearest rung `caps`'s model actually accepts — the
/// ladder-wide counterpart to [`clamp_reasoning_effort`], additionally handling `Off`: a model that
/// can't explicitly disable a reasoning mechanism it has (`reasoning_disableable == false`) has no
/// legal `Off` state at all — pi's `thinkingLevelMap.off === null` — so requesting it bumps up to the
/// model's own floor (`min_reasoning_effort`) instead of silently omitting the reasoning field and
/// leaving the provider to apply its own hidden default. Mirrors pi's `clampThinkingLevel`/
/// `getSupportedThinkingLevels`, called on every model switch (`sdk.ts:241`) and every level change
/// (`agent-session.ts` `setThinkingLevel`/`cycleThinkingLevel`). Every non-`Off` rung reuses
/// [`clamp_reasoning_effort`]'s existing floor/`xhigh`-ceiling behavior unchanged.
///
/// Callers: every point that sets a session's active thinking level for a given model — the initial
/// level at startup, `set_model`/`cycle_model` (re-clamp the *existing* level against the *new*
/// model), `set_reasoning_effort`, and branch-switch model/level restoration. `cycle_thinking_level`
/// uses [`next_available_thinking_level`] instead (see its own doc comment for why a plain
/// `next().then(clamp)` gets stuck for models missing `xhigh`).
pub fn clamp_thinking_level(caps: &ModelCaps, level: ThinkingLevel) -> ThinkingLevel {
    match level.reasoning_effort() {
        None => {
            // `level == ThinkingLevel::Off` — the only variant with no `ReasoningEffort`.
            if has_reasoning_mechanism(caps) && !caps.reasoning_disableable {
                ThinkingLevel::from(caps.min_reasoning_effort)
            } else {
                ThinkingLevel::Off
            }
        }
        Some(effort) => ThinkingLevel::from(clamp_reasoning_effort(caps, effort)),
    }
}

/// The rung `cycle_thinking_level` should land on next, for `caps`'s model, from `level`. Advances
/// through [`available_thinking_levels`] — not the raw 6-rung ladder via [`ThinkingLevel::next`] — so
/// cycling always reaches every level the model actually supports and wraps cleanly at its own ends,
/// rather than getting stuck: a naive `level.next()` then [`clamp_thinking_level`] would bounce forever
/// between `High` and a re-clamped `XHigh` for any model lacking `xhigh` support, since `XHigh` always
/// clamps back down to the same `High` it started from and the ladder never advances past it. If
/// `level` itself isn't currently available (e.g. state predates a model switch that narrowed the set),
/// lands on its clamp first rather than jumping an extra rung ahead.
pub fn next_available_thinking_level(caps: &ModelCaps, level: ThinkingLevel) -> ThinkingLevel {
    let available = available_thinking_levels(caps);
    match available.iter().position(|&l| l == level) {
        Some(i) => available[(i + 1) % available.len()],
        None => clamp_thinking_level(caps, level),
    }
}

/// The Anthropic adaptive-thinking `output_config.effort` wire string for a (clamped) reasoning
/// effort, mirroring pi's `mapThinkingLevelToEffort`: Anthropic's adaptive shape has no `minimal` tier
/// at all (always sent as `"low"`), and `xhigh` is model-specific (`adaptive_xhigh_effort_wire`) —
/// everything else is the effort's own name. Only meaningful for `ThinkingShape::Adaptive`; callers
/// should clamp with [`clamp_reasoning_effort`] first so a model that doesn't support `xhigh` never
/// reaches this function still holding it.
pub fn anthropic_adaptive_effort_wire(
    caps: &ModelCaps,
    effort: crate::transport::ReasoningEffort,
) -> &'static str {
    use crate::transport::ReasoningEffort as RE;
    match clamp_reasoning_effort(caps, effort) {
        RE::Minimal | RE::Low => "low",
        RE::Medium => "medium",
        RE::High => "high",
        RE::XHigh => caps.adaptive_xhigh_effort_wire,
    }
}

/// Translate a portable `level` into the `(thinking_budget, reasoning_effort)` pair
/// [`crate::Agent::with_thinking`]/[`crate::Agent::with_reasoning_effort`] need for a model with
/// capabilities `caps` — whichever combination its `thinking`/`reasoning_effort` capabilities actually
/// call for:
/// - `ThinkingShape::None` and no OpenAI `reasoning_effort` support: neither field is ever set — the
///   model has no thinking/reasoning mechanism at all.
/// - `ThinkingShape::Budget` (Claude 3.7/4.x): a token budget scaled from `level`, clamped below
///   `max_output`; `reasoning_effort` stays unset (that dialect arm never reads it).
/// - `ThinkingShape::Adaptive` (gen-6+ Claude/Fable) and OpenAI's `reasoning_effort: true` models both
///   want `reasoning_effort` set — for `Adaptive`, `thinking` is *also* set (any nonzero budget; it's
///   purely the on/off gate for that shape, see `budget_for_effort`'s doc comment) since Anthropic's
///   `output_config.effort` sibling field only takes effect when thinking itself is enabled.
pub fn thinking_for_level(
    caps: &ModelCaps,
    level: ThinkingLevel,
) -> (Option<u32>, Option<crate::transport::ReasoningEffort>) {
    let Some(effort) = level.reasoning_effort() else {
        return (None, None);
    };
    let thinking = matches!(
        caps.thinking,
        ThinkingShape::Budget | ThinkingShape::Adaptive
    )
    .then(|| budget_for_effort(effort, caps.max_output));
    // Populated whenever the model has *any* steerable mechanism (see `has_reasoning_mechanism`) —
    // not just `reasoning_effort`/`Adaptive` — so a toggle-only third-party family (Kimi, GLM below
    // 5.2: `reasoning_effort: false` but a real `openai_reasoning_format` toggle) still gets
    // `req.reasoning_effort` set, letting `dialect::openai::build_body`'s family-specific branch turn
    // the toggle on even though it never puts a graduated effort string on the wire for these ids.
    let reasoning_effort_applies = caps.reasoning_effort
        || caps.thinking == ThinkingShape::Adaptive
        || caps.openai_reasoning_format != OpenAiReasoningFormat::Standard;
    let reasoning_effort = reasoning_effort_applies.then_some(effort);
    (thinking, reasoning_effort)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claude_gen6_plus_is_adaptive_with_1m_context() {
        // Our own default model: adaptive thinking, 1M context, 128k output.
        let c = capabilities("claude-opus-4-8");
        assert_eq!(c.max_tokens_field, MaxTokensField::MaxTokens);
        assert!(c.supports_long_cache);
        assert_eq!(c.thinking, ThinkingShape::Adaptive);
        assert_eq!(c.context_window, 1_000_000);
        assert_eq!(c.max_output, 128_000);
        assert!(!c.reasoning_effort);

        for id in [
            "claude-opus-4-6",
            "claude-opus-4-7",
            "claude-sonnet-5",
            "claude-fable-5",
        ] {
            let c = capabilities(id);
            assert_eq!(
                c.thinking,
                ThinkingShape::Adaptive,
                "{id} should be adaptive"
            );
            assert_eq!(c.context_window, 1_000_000, "{id} should have 1M context");
        }
        // sonnet-4-6 is adaptive but caps output at 64k, not 128k.
        let c = capabilities("claude-sonnet-4-6");
        assert_eq!(c.thinking, ThinkingShape::Adaptive);
        assert_eq!(c.context_window, 1_000_000);
        assert_eq!(c.max_output, 64_000);
    }

    #[test]
    fn only_opus_4_7_and_opus_4_8_reject_temperature() {
        // pi: `compat.supportsTemperature: false` is set on exactly these two ids in
        // `anthropic.models.ts`; every other current id (including the rest of gen6+, legacy Budget-
        // shape Claude, and every OpenAI-wire model) defaults to supporting it.
        for id in ["claude-opus-4-7", "claude-opus-4-8"] {
            assert!(
                !capabilities(id).supports_temperature,
                "{id} should reject temperature"
            );
        }
        for id in [
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-sonnet-5",
            "claude-fable-5",
            "claude-opus-4-5",
            "claude-sonnet-4-5",
            "claude-3-5-sonnet-20241022",
            "gpt-4o",
            "o3",
        ] {
            assert!(
                capabilities(id).supports_temperature,
                "{id} should still support temperature"
            );
        }
    }

    #[test]
    fn claude_pre_gen6_stays_budget_shaped() {
        // opus-4-5 and sonnet-4-5 predate the adaptive requirement — still `Budget`, 200k context.
        for id in ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5"] {
            let c = capabilities(id);
            assert_eq!(
                c.thinking,
                ThinkingShape::Budget,
                "{id} should stay Budget-shaped"
            );
            assert_eq!(
                c.context_window, 200_000,
                "{id} should stay at 200k context"
            );
        }
        assert_eq!(capabilities("claude-opus-4-5").max_output, 64_000);
        assert_eq!(capabilities("claude-sonnet-4-5").max_output, 64_000);
        assert_eq!(capabilities("claude-haiku-4-5").max_output, 64_000);
        assert_eq!(capabilities("claude-opus-4-1").max_output, 32_000);
    }

    #[test]
    fn claude_gen3_legacy_has_no_thinking() {
        for id in [
            "claude-3-5-sonnet-20241022",
            "claude-3-haiku-20240307",
            "claude-3-opus-20240229",
        ] {
            let c = capabilities(id);
            assert_eq!(
                c.thinking,
                ThinkingShape::None,
                "{id} predates extended thinking"
            );
            // No thinking support doesn't mean no long-cache support — prompt caching (including the
            // 1-hour TTL) is a separate, orthogonal capability that every current Claude 3.x id has.
            assert!(c.supports_long_cache, "{id} does support the 1h cache TTL");
        }
        // 3.7-sonnet is the exception: it *does* support extended thinking (Budget shape).
        assert_eq!(
            capabilities("claude-3-7-sonnet-20250219").thinking,
            ThinkingShape::Budget
        );
    }

    #[test]
    fn openai_reasoning_models_use_completion_tokens_field() {
        for id in ["o1", "o3-mini", "o4-mini", "gpt-5", "gpt-5-mini"] {
            let c = capabilities(id);
            assert_eq!(
                c.max_tokens_field,
                MaxTokensField::MaxCompletionTokens,
                "{id} should use max_completion_tokens"
            );
            assert!(c.reasoning_effort, "{id} should take reasoning_effort");
            assert!(
                c.supports_long_cache,
                "{id} should default to supporting prompt_cache_retention (pi's denylist default)"
            );
            assert_eq!(c.thinking, ThinkingShape::None);
        }
    }

    #[test]
    fn openai_context_windows_match_family() {
        assert_eq!(capabilities("o1").context_window, 200_000);
        assert_eq!(capabilities("o1").max_output, 100_000);
        assert_eq!(capabilities("gpt-5").context_window, 400_000);
        assert_eq!(capabilities("gpt-5-mini").context_window, 400_000);
        assert_eq!(capabilities("gpt-5.4").context_window, 272_000);
        assert_eq!(capabilities("gpt-5.4-mini").context_window, 400_000);
        assert_eq!(capabilities("gpt-5.4-pro").context_window, 1_050_000);
        assert_eq!(
            capabilities("gpt-5-chat-latest").thinking,
            ThinkingShape::None
        );
        assert!(!capabilities("gpt-5-chat-latest").reasoning_effort);
    }

    #[test]
    fn o1_mini_is_no_longer_special_cased() {
        // LOW pi-parity gap (fixed): o1-mini used to be excluded from `supports_vision` alongside
        // o3-mini, but pi's live catalogue has fully retired the id — nothing upstream still serves
        // it, so it now falls through like any other (still-live) o-series id.
        assert!(capabilities("o1-mini").supports_vision);
    }

    #[test]
    fn o3_mini_is_text_only() {
        // The one o-series id that isn't vision-capable (pi: `input: ["text"]`); every other
        // o-series id is vision-capable.
        assert!(!capabilities("o3-mini").supports_vision);
        assert!(capabilities("o1").supports_vision);
        assert!(capabilities("o4-mini").supports_vision);
    }

    #[test]
    fn gpt5_pro_context_window_only_1_05m_for_5_4_and_5_5() {
        // Only the 5.4/5.5 generation of "-pro" gets the larger window; earlier -pro ids share the
        // rest of the family's 400k.
        assert_eq!(capabilities("gpt-5-pro").context_window, 400_000);
        assert_eq!(capabilities("gpt-5.2-pro").context_window, 400_000);
        assert_eq!(capabilities("gpt-5.4-pro").context_window, 1_050_000);
        assert_eq!(capabilities("gpt-5.5-pro").context_window, 1_050_000);
    }

    #[test]
    fn gpt_5_3_codex_spark_is_narrower_than_the_rest_of_the_family() {
        let c = capabilities("gpt-5.3-codex-spark");
        assert_eq!(c.context_window, 128_000);
        assert_eq!(c.max_output, 32_000);
    }

    #[test]
    fn chat_latest_reasoning_effort_is_id_specific() {
        // 5.1/5.2-chat-latest are still reasoning_effort-driven per pi's catalogue; the plain and
        // 5.3 chat-latest ids are not.
        assert!(capabilities("gpt-5.1-chat-latest").reasoning_effort);
        assert!(capabilities("gpt-5.2-chat-latest").reasoning_effort);
        assert!(!capabilities("gpt-5-chat-latest").reasoning_effort);
        assert!(!capabilities("gpt-5.3-chat-latest").reasoning_effort);
    }

    #[test]
    fn gpt4o_2024_05_13_pinned_snapshot_caps_output_at_4096() {
        // This one dated snapshot caps output tighter (4096) than every other 4o-family id (16384).
        assert_eq!(capabilities("gpt-4o-2024-05-13").max_output, 4_096);
        assert_eq!(capabilities("gpt-4o").max_output, 16_384);
    }

    #[test]
    fn reasoning_disableable_is_id_specific_not_family_wide() {
        // Anthropic: every gen6+ id can be told to disable thinking explicitly, except claude-fable-5.
        assert!(capabilities("claude-opus-4-8").reasoning_disableable);
        assert!(capabilities("claude-sonnet-5").reasoning_disableable);
        assert!(!capabilities("claude-fable-5").reasoning_disableable);
        assert!(!capabilities("fable-5").reasoning_disableable);
        // Budget-shaped (pre-gen6, post-gen3) Claude ids are disable-capable too.
        assert!(capabilities("claude-opus-4-5").reasoning_disableable);
        // Gen-3 legacy ids have no thinking support at all — nothing to disable.
        assert!(!capabilities("claude-3-5-sonnet-20241022").reasoning_disableable);

        // OpenAI o-series: disable-capable by default.
        assert!(capabilities("o3-mini").reasoning_disableable);
        assert!(capabilities("o1").reasoning_disableable);

        // OpenAI gpt-5: only the exact-match allowlist is disable-capable.
        assert!(capabilities("gpt-5.1").reasoning_disableable);
        assert!(capabilities("gpt-5.2").reasoning_disableable);
        assert!(capabilities("gpt-5.3-codex").reasoning_disableable);
        assert!(capabilities("gpt-5.4").reasoning_disableable);
        assert!(capabilities("gpt-5.4-mini").reasoning_disableable);
        assert!(capabilities("gpt-5.4-nano").reasoning_disableable);
        assert!(capabilities("gpt-5.5").reasoning_disableable);
        assert!(
            !capabilities("gpt-5").reasoning_disableable,
            "bare gpt-5 isn't in the allowlist"
        );
        assert!(
            !capabilities("gpt-5.1-chat-latest").reasoning_disableable,
            "chat-latest ids have no off signal even though they take reasoning_effort"
        );
        assert!(!capabilities("gpt-5.3-codex-spark").reasoning_disableable);

        // OpenAI gpt-4 family never takes reasoning_effort — nothing to disable.
        assert!(!capabilities("gpt-4o").reasoning_disableable);
        assert!(!capabilities("gpt-4.1").reasoning_disableable);
    }

    #[test]
    fn supports_eager_tool_streaming_is_anthropic_only() {
        // Every current Anthropic id (across all three generational branches) supports the per-tool
        // eager-input-streaming shape.
        for id in [
            "claude-opus-4-8",
            "claude-sonnet-5",
            "claude-fable-5",
            "claude-3-5-sonnet-20241022",
            "claude-opus-4-5",
        ] {
            assert!(
                capabilities(id).supports_eager_tool_streaming,
                "{id} should support eager tool streaming"
            );
        }
        // No current OpenAI id does — it's an Anthropic-wire-only concept.
        for id in [
            "o1",
            "gpt-5",
            "gpt-5-chat-latest",
            "gpt-5.3-codex-spark",
            "gpt-4o",
            "gpt-4.1",
        ] {
            assert!(
                !capabilities(id).supports_eager_tool_streaming,
                "{id} should not support eager tool streaming"
            );
        }
    }

    #[test]
    fn gpt4_uses_plain_max_tokens() {
        let c = capabilities("gpt-4o");
        assert_eq!(c.max_tokens_field, MaxTokensField::MaxTokens);
        assert!(!c.reasoning_effort);
        assert!(c.supports_vision);
        assert_eq!(c.context_window, 128_000);
        assert!(c.supports_long_cache);

        // 4.1 gets the ~1M-token window; bare "gpt-4" stays at the original 8k.
        assert_eq!(capabilities("gpt-4.1").context_window, 1_047_576);
        assert_eq!(capabilities("gpt-4.1-mini").context_window, 1_047_576);
        assert_eq!(capabilities("gpt-4").context_window, 8_192);
    }

    #[test]
    fn bare_gpt4_predates_vision_support_unlike_every_other_gpt4_id() {
        // Bare "gpt-4" (pi's catalogue: `input: ["text"]`) is the original, text-only model — vision
        // shipped later as "gpt-4-vision-preview" and was folded into "gpt-4-turbo"/4o onward. Getting
        // this wrong would let the loop attach an image to a request the model can't accept at all.
        assert!(
            !capabilities("gpt-4").supports_vision,
            "bare gpt-4 must not be marked vision-capable"
        );
        for id in ["gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "gpt-4.1"] {
            assert!(
                capabilities(id).supports_vision,
                "{id} should still be vision-capable"
            );
        }
    }

    #[test]
    fn unknown_model_is_conservative() {
        let c = capabilities("some-future-model-x");
        assert_eq!(c.max_tokens_field, MaxTokensField::MaxTokens);
        assert!(!c.supports_long_cache);
        assert_eq!(c.thinking, ThinkingShape::None);
    }

    #[test]
    fn matching_is_case_insensitive() {
        assert_eq!(
            capabilities("Claude-Opus-4-8").thinking,
            ThinkingShape::Adaptive
        );
    }

    #[test]
    fn native_openai_ids_speak_responses_api() {
        // Every native OpenAI id (o-series, gpt-5 family incl. chat-latest, gpt-4 family) routes
        // through the Responses API per pi's live catalogue — no exceptions among current ids.
        for id in [
            "o1",
            "o3-mini",
            "gpt-5",
            "gpt-5-chat-latest",
            "gpt-5.4-pro",
            "gpt-4o",
            "gpt-4.1-mini",
            "gpt-4-turbo",
            "gpt-4",
        ] {
            assert_eq!(
                capabilities(id).api,
                ApiKind::Responses,
                "{id} should speak the Responses API"
            );
        }
    }

    #[test]
    fn claude_and_unknown_ids_default_to_chat_completions_api() {
        // Anthropic ids ignore `api` (they always speak Dialect::Anthropic), but the field must still
        // resolve to a harmless default rather than accidentally inheriting `Responses`.
        assert_eq!(
            capabilities("claude-opus-4-8").api,
            ApiKind::ChatCompletions
        );
        // Third-party OpenAI-compatible ids (and anything unrecognized) stay on Chat Completions —
        // only native OpenAI ids get the Responses API.
        assert_eq!(capabilities("llama-3.1-70b").api, ApiKind::ChatCompletions);
        assert_eq!(
            capabilities("some-future-model-x").api,
            ApiKind::ChatCompletions
        );
    }

    #[test]
    fn thinking_level_next_cycles_through_all_six_rungs_and_wraps() {
        let mut level = ThinkingLevel::Off;
        let mut seen = vec![level];
        for _ in 0..5 {
            level = level.next();
            seen.push(level);
        }
        assert_eq!(
            seen,
            vec![
                ThinkingLevel::Off,
                ThinkingLevel::Minimal,
                ThinkingLevel::Low,
                ThinkingLevel::Medium,
                ThinkingLevel::High,
                ThinkingLevel::XHigh,
            ]
        );
        assert_eq!(
            level.next(),
            ThinkingLevel::Off,
            "the ladder must wrap from XHigh back to Off"
        );
    }

    #[test]
    fn thinking_level_as_str_and_parse_round_trip() {
        for level in [
            ThinkingLevel::Off,
            ThinkingLevel::Minimal,
            ThinkingLevel::Low,
            ThinkingLevel::Medium,
            ThinkingLevel::High,
            ThinkingLevel::XHigh,
        ] {
            assert_eq!(ThinkingLevel::parse(level.as_str()), Some(level));
        }
        assert_eq!(ThinkingLevel::parse("nonsense"), None);
    }

    #[test]
    fn thinking_for_level_is_off_for_a_model_with_no_thinking_mechanism_at_all() {
        // Legacy gen-3 Claude ids: `ThinkingShape::None`, `reasoning_effort: false`.
        let caps = capabilities("claude-3-5-sonnet-20241022");
        assert_eq!(caps.thinking, ThinkingShape::None);
        for level in [
            ThinkingLevel::Minimal,
            ThinkingLevel::High,
            ThinkingLevel::XHigh,
        ] {
            assert_eq!(
                thinking_for_level(&caps, level),
                (None, None),
                "a model with no thinking mechanism must never get either field set"
            );
        }
    }

    #[test]
    fn thinking_for_level_off_always_clears_both_fields_regardless_of_shape() {
        for id in ["claude-opus-4-5", "claude-opus-4-8", "o3", "gpt-4o"] {
            assert_eq!(
                thinking_for_level(&capabilities(id), ThinkingLevel::Off),
                (None, None),
                "{id}: Off must clear both fields"
            );
        }
    }

    #[test]
    fn thinking_for_level_sets_only_a_budget_for_budget_shape_models() {
        // "claude-opus-4-5" is pre-gen6: `ThinkingShape::Budget`, no OpenAI-style `reasoning_effort`.
        let caps = capabilities("claude-opus-4-5");
        assert_eq!(caps.thinking, ThinkingShape::Budget);
        let (thinking, effort) = thinking_for_level(&caps, ThinkingLevel::High);
        assert_eq!(thinking, Some(16_384), "pi's defaultBudgets.high is 16384, not 24000");
        assert_eq!(
            effort, None,
            "a Budget-shape model's dialect never reads reasoning_effort"
        );
    }

    #[test]
    fn thinking_for_level_clamps_the_budget_below_max_output() {
        // A small `max_output` (e.g. a legacy id) must clamp XHigh's 32_000 rungs down to something
        // that still leaves room for the turn's own output.
        let caps = ModelCaps {
            max_output: 4_096,
            ..capabilities("claude-opus-4-5")
        };
        let (thinking, _) = thinking_for_level(&caps, ThinkingLevel::XHigh);
        assert_eq!(thinking, Some(4_095), "must clamp to max_output - 1");
    }

    #[test]
    fn thinking_for_level_sets_both_fields_for_adaptive_shape_models() {
        // Gen6+ Claude (e.g. the default "claude-opus-4-8"): `ThinkingShape::Adaptive`. Both `thinking`
        // (the on/off gate) and `reasoning_effort` (the actual depth) must be set together.
        let caps = capabilities("claude-opus-4-8");
        assert_eq!(caps.thinking, ThinkingShape::Adaptive);
        let (thinking, effort) = thinking_for_level(&caps, ThinkingLevel::Medium);
        assert!(thinking.is_some(), "Adaptive needs a nonzero gate value");
        assert_eq!(effort, Some(crate::transport::ReasoningEffort::Medium));
    }

    #[test]
    fn thinking_for_level_sets_only_reasoning_effort_for_openai_style_models() {
        // OpenAI reasoning models (o-series): `ThinkingShape::None` + `reasoning_effort: true` — no
        // Anthropic-style budget field at all, only the named effort.
        let caps = capabilities("o3");
        assert_eq!(caps.thinking, ThinkingShape::None);
        assert!(caps.reasoning_effort);
        let (thinking, effort) = thinking_for_level(&caps, ThinkingLevel::Low);
        assert_eq!(
            thinking, None,
            "OpenAI reasoning models take no budget field"
        );
        assert_eq!(effort, Some(crate::transport::ReasoningEffort::Low));
    }

    #[test]
    fn clamp_reasoning_effort_drops_unsupported_xhigh_to_high() {
        use crate::transport::ReasoningEffort as RE;
        // o-series, bare/early gpt-5 ids, and sonnet-4-6/sonnet-5 carry no `xhigh` wire value at all in
        // pi's live catalogue — requesting it must clamp down to `high`, not send an invalid value.
        for id in [
            "o3",
            "o1",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-codex",
            "gpt-5-pro",
            "gpt-5.1",
            "gpt-5.1-codex",
            "gpt-5.1-chat-latest",
            "claude-sonnet-4-6",
            "claude-sonnet-5",
        ] {
            let caps = capabilities(id);
            assert_eq!(
                clamp_reasoning_effort(&caps, RE::XHigh),
                RE::High,
                "{id} should clamp xhigh down to high"
            );
        }
        // gpt-5.2+ and opus-4-6/4-7/4-8/fable-5 do support xhigh — must pass through unclamped.
        for id in [
            "gpt-5.2",
            "gpt-5.2-codex",
            "gpt-5.3-codex",
            "gpt-5.3-codex-spark",
            "gpt-5.4",
            "gpt-5.5",
            "gpt-5.5-pro",
            "claude-opus-4-6",
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-fable-5",
        ] {
            let caps = capabilities(id);
            assert_eq!(
                clamp_reasoning_effort(&caps, RE::XHigh),
                RE::XHigh,
                "{id} should support xhigh"
            );
        }
    }

    #[test]
    fn clamp_reasoning_effort_raises_below_a_models_floor() {
        use crate::transport::ReasoningEffort as RE;
        // gpt-5.5 excludes "minimal"; gpt-5.5-pro excludes "minimal" and "low" too.
        let gpt55 = capabilities("gpt-5.5");
        assert_eq!(clamp_reasoning_effort(&gpt55, RE::Minimal), RE::Low);
        assert_eq!(clamp_reasoning_effort(&gpt55, RE::Low), RE::Low);
        assert_eq!(clamp_reasoning_effort(&gpt55, RE::Medium), RE::Medium);

        let gpt55_pro = capabilities("gpt-5.5-pro");
        assert_eq!(clamp_reasoning_effort(&gpt55_pro, RE::Minimal), RE::Medium);
        assert_eq!(clamp_reasoning_effort(&gpt55_pro, RE::Low), RE::Medium);
        assert_eq!(clamp_reasoning_effort(&gpt55_pro, RE::Medium), RE::Medium);

        // Every other reasoning model accepts minimal unclamped.
        for id in ["o3", "gpt-5", "gpt-5.2", "claude-opus-4-8"] {
            assert_eq!(
                clamp_reasoning_effort(&capabilities(id), RE::Minimal),
                RE::Minimal,
                "{id} should accept minimal unclamped"
            );
        }
    }

    #[test]
    fn anthropic_adaptive_effort_wire_matches_pi_map_thinking_level_to_effort() {
        use crate::transport::ReasoningEffort as RE;
        // Anthropic's adaptive shape has no "minimal" tier at all — always sent as "low", on every
        // adaptive model, not just the ones with an explicit override.
        for id in [
            "claude-opus-4-6",
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-sonnet-4-6",
            "claude-sonnet-5",
            "claude-fable-5",
        ] {
            let caps = capabilities(id);
            assert_eq!(anthropic_adaptive_effort_wire(&caps, RE::Minimal), "low");
            assert_eq!(anthropic_adaptive_effort_wire(&caps, RE::Low), "low");
            assert_eq!(anthropic_adaptive_effort_wire(&caps, RE::Medium), "medium");
            assert_eq!(anthropic_adaptive_effort_wire(&caps, RE::High), "high");
        }
        // xhigh is model-specific: opus-4-6 uniquely sends "max"; opus-4-7/4-8/fable-5 send "xhigh"
        // literally; sonnet-4-6/sonnet-5 have already been clamped to High and so never reach "xhigh"
        // as an input in practice, but the function must still degrade gracefully if it ever did.
        assert_eq!(
            anthropic_adaptive_effort_wire(&capabilities("claude-opus-4-6"), RE::XHigh),
            "max"
        );
        for id in ["claude-opus-4-7", "claude-opus-4-8", "claude-fable-5"] {
            assert_eq!(
                anthropic_adaptive_effort_wire(&capabilities(id), RE::XHigh),
                "xhigh",
                "{id} should send xhigh literally"
            );
        }
        for id in ["claude-sonnet-4-6", "claude-sonnet-5"] {
            assert_eq!(
                anthropic_adaptive_effort_wire(&capabilities(id), RE::XHigh),
                "high",
                "{id} has no xhigh wire value; must degrade to high"
            );
        }
    }

    #[test]
    fn clamp_thinking_level_bumps_off_to_the_floor_for_non_disableable_reasoning_models() {
        // The CRITICAL gap this fix closes: a model that has a reasoning mechanism but can't
        // explicitly disable it must never be left at the stored level `Off` — that silently omits
        // the reasoning field entirely and lets the provider apply its own hidden default effort.
        for id in [
            "gpt-5",
            "gpt-5-codex",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-5-pro",
            "gpt-5.1-codex",
            "gpt-5.1-codex-max",
            "gpt-5.1-codex-mini",
            "gpt-5.1-chat-latest",
            "gpt-5.2-chat-latest",
            "gpt-5.2-codex",
            "gpt-5.2-pro",
            "gpt-5.3-codex-spark",
            "gpt-5.4-pro",
            "claude-fable-5",
        ] {
            let caps = capabilities(id);
            assert!(
                !caps.reasoning_disableable,
                "{id} must be a non-disableable model for this test to be meaningful"
            );
            assert_ne!(
                clamp_thinking_level(&caps, ThinkingLevel::Off),
                ThinkingLevel::Off,
                "{id}: Off must not survive the clamp for a non-disableable reasoning model"
            );
        }
        // Ordinary floor: bumps to Minimal.
        assert_eq!(
            clamp_thinking_level(&capabilities("gpt-5-codex"), ThinkingLevel::Off),
            ThinkingLevel::Minimal
        );
        // A model-specific floor above Minimal must be honored too, not just a hardcoded "Minimal".
        assert_eq!(
            clamp_thinking_level(&capabilities("gpt-5.5-pro"), ThinkingLevel::Off),
            ThinkingLevel::Medium,
            "gpt-5.5-pro's floor is Medium, not the generic Minimal"
        );
    }

    #[test]
    fn clamp_thinking_level_leaves_off_alone_when_legal() {
        // Disable-capable reasoning models: Off is a real, legal state.
        for id in ["claude-opus-4-8", "claude-sonnet-5", "o3", "o1", "gpt-5.1"] {
            assert_eq!(
                clamp_thinking_level(&capabilities(id), ThinkingLevel::Off),
                ThinkingLevel::Off,
                "{id}: Off is legal and must not be bumped"
            );
        }
        // Models with no thinking/reasoning mechanism at all: Off is the only sensible state.
        for id in ["gpt-4o", "gpt-4.1", "claude-3-5-sonnet-20241022"] {
            assert_eq!(
                clamp_thinking_level(&capabilities(id), ThinkingLevel::Off),
                ThinkingLevel::Off,
                "{id}: has no mechanism to disable; Off must pass through"
            );
        }
    }

    #[test]
    fn clamp_thinking_level_still_clamps_non_off_rungs_like_clamp_reasoning_effort() {
        use crate::transport::ReasoningEffort as RE;
        // xhigh ceiling: delegates to the same behavior as `clamp_reasoning_effort`.
        assert_eq!(
            clamp_thinking_level(&capabilities("o3"), ThinkingLevel::XHigh),
            ThinkingLevel::from(RE::High)
        );
        assert_eq!(
            clamp_thinking_level(&capabilities("claude-opus-4-8"), ThinkingLevel::XHigh),
            ThinkingLevel::XHigh
        );
        // floor: gpt-5.5-pro excludes Minimal/Low.
        assert_eq!(
            clamp_thinking_level(&capabilities("gpt-5.5-pro"), ThinkingLevel::Minimal),
            ThinkingLevel::Medium
        );
    }

    #[test]
    fn available_thinking_levels_excludes_off_for_non_disableable_models() {
        assert_eq!(
            available_thinking_levels(&capabilities("gpt-5-codex")),
            vec![
                ThinkingLevel::Minimal,
                ThinkingLevel::Low,
                ThinkingLevel::Medium,
                ThinkingLevel::High,
            ],
            "gpt-5-codex has no xhigh and no off"
        );
    }

    #[test]
    fn available_thinking_levels_is_the_full_ladder_for_a_fully_capable_model() {
        assert_eq!(
            available_thinking_levels(&capabilities("claude-opus-4-8")),
            vec![
                ThinkingLevel::Off,
                ThinkingLevel::Minimal,
                ThinkingLevel::Low,
                ThinkingLevel::Medium,
                ThinkingLevel::High,
                ThinkingLevel::XHigh,
            ]
        );
    }

    #[test]
    fn available_thinking_levels_is_just_off_for_a_model_with_no_mechanism() {
        assert_eq!(
            available_thinking_levels(&capabilities("gpt-4o")),
            vec![ThinkingLevel::Off]
        );
    }

    #[test]
    fn budget_shape_models_do_not_offer_xhigh_but_adaptive_shape_gen6_plus_does() {
        // Regression for the pi-parity gap: pi's `thinkingLevelMap.xhigh` is only defined for the four
        // gen6+ Adaptive-shape ids — classic Budget-shape Claude models (sonnet-4-5, opus-4-5,
        // sonnet-3-7, haiku-4-5, opus-4-0/4-1, etc.) must not advertise or accept the `xhigh` rung, even
        // though `thinking_for_level`/`budget_for_effort` would happily turn it into a plain numeric
        // `budget_tokens` value the Anthropic wire accepts fine — the bug is purely that we shouldn't
        // *offer* it for these models in the first place.
        for id in [
            "claude-sonnet-4-5",
            "claude-opus-4-5",
            "claude-3-7-sonnet-20250219",
            "claude-haiku-4-5",
        ] {
            let caps = capabilities(id);
            assert_eq!(
                caps.thinking,
                ThinkingShape::Budget,
                "{id} should be Budget-shape"
            );
            assert!(
                !caps.supports_xhigh_reasoning,
                "{id}: Budget-shape models must not support xhigh"
            );
            assert!(
                !available_thinking_levels(&caps).contains(&ThinkingLevel::XHigh),
                "{id}: xhigh must not be an available thinking level"
            );
            assert_eq!(
                clamp_thinking_level(&caps, ThinkingLevel::XHigh),
                ThinkingLevel::High,
                "{id}: requesting xhigh must clamp down to high"
            );
        }

        // Adaptive-shape gen6+ models are unaffected: xhigh is still offered where pi's catalogue
        // actually maps it (opus-4-6/4-7/4-8, fable-5 — sonnet-4-6/sonnet-5 are the two adaptive
        // exceptions covered by other tests in this module).
        let caps = capabilities("claude-opus-4-8");
        assert_eq!(caps.thinking, ThinkingShape::Adaptive);
        assert!(caps.supports_xhigh_reasoning);
        assert!(available_thinking_levels(&caps).contains(&ThinkingLevel::XHigh));
        assert_eq!(
            clamp_thinking_level(&caps, ThinkingLevel::XHigh),
            ThinkingLevel::XHigh
        );
    }

    #[test]
    fn next_available_thinking_level_wraps_within_the_models_own_ladder_without_getting_stuck() {
        // The regression this specifically guards: a naive `level.next()` + `clamp_thinking_level`
        // bounces forever between High and a re-clamped XHigh for a model lacking xhigh support,
        // since XHigh always clamps back down to the very High it started from.
        let caps = capabilities("gpt-5-codex");
        let mut level = ThinkingLevel::Minimal;
        let mut seen = vec![level];
        for _ in 0..3 {
            level = next_available_thinking_level(&caps, level);
            seen.push(level);
        }
        assert_eq!(
            seen,
            vec![
                ThinkingLevel::Minimal,
                ThinkingLevel::Low,
                ThinkingLevel::Medium,
                ThinkingLevel::High,
            ]
        );
        // Cycling past the top must wrap back to Minimal (the model's actual floor), not get stuck.
        assert_eq!(
            next_available_thinking_level(&caps, ThinkingLevel::High),
            ThinkingLevel::Minimal
        );
    }

    #[test]
    fn next_available_thinking_level_cycles_the_full_ladder_for_a_fully_capable_model() {
        let caps = capabilities("claude-opus-4-8");
        let mut level = ThinkingLevel::Off;
        let mut seen = vec![level];
        for _ in 0..5 {
            level = next_available_thinking_level(&caps, level);
            seen.push(level);
        }
        assert_eq!(
            seen,
            vec![
                ThinkingLevel::Off,
                ThinkingLevel::Minimal,
                ThinkingLevel::Low,
                ThinkingLevel::Medium,
                ThinkingLevel::High,
                ThinkingLevel::XHigh,
            ]
        );
        assert_eq!(
            next_available_thinking_level(&caps, ThinkingLevel::XHigh),
            ThinkingLevel::Off
        );
    }

    #[test]
    fn next_available_thinking_level_stays_at_off_for_a_model_with_no_mechanism() {
        let caps = capabilities("gpt-4o");
        assert_eq!(
            next_available_thinking_level(&caps, ThinkingLevel::Off),
            ThinkingLevel::Off
        );
    }

    #[test]
    fn next_available_thinking_level_reclamps_a_stale_unavailable_level_instead_of_skipping() {
        // If the stored level isn't in the model's available set at all (e.g. carried over from a
        // model switch that should have re-clamped it, but didn't), landing on its clamp first is
        // safer than jumping straight to the next rung after it.
        let caps = capabilities("gpt-5-codex"); // Off is not available here.
        assert_eq!(
            next_available_thinking_level(&caps, ThinkingLevel::Off),
            clamp_thinking_level(&caps, ThinkingLevel::Off),
        );
    }

    // ---- Third-party provider coverage (pi-parity: these used to fall through to `unknown()`) ----

    #[test]
    fn deepseek_gets_its_real_context_window_and_max_output_not_the_unknown_default() {
        // The CRITICAL bug this closes: before this table had a DeepSeek entry, `deepseek-v4-pro`
        // resolved to `ModelCaps::unknown()` — 128k context, a 4096-token output ceiling regardless of
        // the model's real 384k one (`Agent::new` seeds `max_tokens` from `max_output`).
        for id in ["deepseek-v4-pro", "deepseek-v4-flash"] {
            let c = capabilities(id);
            assert_eq!(c.context_window, 1_000_000, "{id}");
            assert_eq!(c.max_output, 384_000, "{id}");
            assert_eq!(c.max_tokens_field, MaxTokensField::MaxCompletionTokens, "{id}");
            assert!(c.reasoning_effort, "{id}");
            assert_eq!(c.openai_reasoning_format, OpenAiReasoningFormat::DeepSeek, "{id}");
            // thinkingLevelMap nulls minimal/low/medium — floor is High, xhigh wired as "max".
            assert_eq!(c.min_reasoning_effort, crate::transport::ReasoningEffort::High, "{id}");
            assert_eq!(c.adaptive_xhigh_effort_wire, "max", "{id}");
        }
    }

    #[test]
    fn deepseek_thinking_param_emits_the_deepseek_toggle_and_reasoning_effort() {
        use crate::transport::ReasoningEffort as RE;
        let caps = capabilities("deepseek-v4-pro");
        // A requested level turns on both the toggle (checked in dialect::openai's own tests) and the
        // portable (thinking_budget, reasoning_effort) pair `Agent::with_thinking`/`with_reasoning_effort`
        // are built from.
        let (thinking, effort) = thinking_for_level(&caps, ThinkingLevel::High);
        assert_eq!(thinking, None, "DeepSeek has no Anthropic-style budget field");
        assert_eq!(effort, Some(RE::High));
        // Off clears it.
        assert_eq!(thinking_for_level(&caps, ThinkingLevel::Off), (None, None));
    }

    #[test]
    fn kimi_thinking_models_have_a_mechanism_despite_no_reasoning_effort_string() {
        // Moonshot/Kimi: `supportsReasoningEffort: false` on every id (no graduated effort vocabulary
        // at all), but the "thinking"-suffixed ids still have a real client-steerable on/off toggle
        // (`OpenAiReasoningFormat::DeepSeek`, the same shape pi tags them with). Regression guard for
        // `has_reasoning_mechanism`: without its third arm, this model would incorrectly report having
        // no mechanism at all (`available_thinking_levels` collapsing to `[Off]`) even though
        // `--thinking high` genuinely turns its reasoning on.
        let caps = capabilities("kimi-k2-thinking");
        assert!(!caps.reasoning_effort, "Kimi has no effort vocabulary");
        assert_eq!(caps.openai_reasoning_format, OpenAiReasoningFormat::DeepSeek);
        assert_ne!(
            available_thinking_levels(&caps),
            vec![ThinkingLevel::Off],
            "a toggle-only mechanism must still offer more than just Off"
        );
        // thinking_for_level must still populate `reasoning_effort` (the toggle gate the dialect reads)
        // even though the model never gets a graduated wire string for it.
        let (_, effort) = thinking_for_level(&caps, ThinkingLevel::Medium);
        assert!(effort.is_some(), "the toggle gate must be populated");

        // The non-"thinking" preview ids (reasoning: false in pi's catalogue) get no mechanism at all.
        let non_reasoning = capabilities("kimi-k2-0711-preview");
        assert_eq!(non_reasoning.openai_reasoning_format, OpenAiReasoningFormat::Standard);
        assert!(!non_reasoning.reasoning_effort);
        assert_eq!(
            available_thinking_levels(&non_reasoning),
            vec![ThinkingLevel::Off]
        );

        // kimi-k2.7-code has no "off" wire value at all — Off must bump up to the floor, same as
        // claude-fable-5's identical shape.
        let code = capabilities("kimi-k2.7-code");
        assert!(!code.reasoning_disableable);
        assert_ne!(clamp_thinking_level(&code, ThinkingLevel::Off), ThinkingLevel::Off);
    }

    #[test]
    fn glm_5_2_gets_reasoning_effort_but_earlier_glm_ids_do_not() {
        let old = capabilities("glm-4.7");
        assert!(!old.reasoning_effort, "pre-5.2 GLM has no effort vocabulary");
        assert_eq!(old.openai_reasoning_format, OpenAiReasoningFormat::Zai);
        assert_eq!(old.context_window, 204_800);
        assert_eq!(old.max_output, 131_072);

        let new = capabilities("glm-5.2");
        assert!(new.reasoning_effort, "glm-5.2 gains a real effort vocabulary");
        assert_eq!(new.context_window, 1_000_000);
        assert_eq!(new.openai_reasoning_format, OpenAiReasoningFormat::Zai);

        // glm-5v-turbo is the one vision-capable id in this family.
        assert!(capabilities("glm-5v-turbo").supports_vision);
        assert!(!capabilities("glm-4.7").supports_vision);
    }

    #[test]
    fn xai_never_emits_a_reasoning_field_matching_pis_unconditional_opt_out() {
        // pi's `detectCompat` marks every xAI id `supportsReasoningEffort: false` unconditionally, even
        // for the reasoning-capable ids (grok-4.3, grok-4.20-*-reasoning) — they reason on their own
        // with no client-steerable toggle at all.
        for id in ["grok-3", "grok-4.3", "grok-4.20-0309-reasoning", "grok-build-0.1"] {
            let c = capabilities(id);
            assert!(!c.reasoning_effort, "{id}");
            assert_eq!(c.openai_reasoning_format, OpenAiReasoningFormat::Standard, "{id}");
            assert_eq!(c.max_tokens_field, MaxTokensField::MaxCompletionTokens, "{id}");
            // No mechanism at all — Off is the only available level.
            assert_eq!(available_thinking_levels(&c), vec![ThinkingLevel::Off], "{id}");
        }
        assert_eq!(capabilities("grok-3").context_window, 131_072);
        assert_eq!(capabilities("grok-4.3").context_window, 1_000_000);
    }

    #[test]
    fn together_hosted_qwen_gets_the_together_wire_format() {
        // pi-parity pass 20 Task 1: "Qwen/Qwen3.6-Plus" no longer proves this — that exact vendor-slug
        // string is also OpenRouter's own real spelling, and now resolves to OpenRouter's smaller,
        // safer numbers/format instead (see `openrouter_qwen_ids_get_their_own_real_numbers_not_the_
        // colliding_together_shape` below). "Qwen/Qwen3.5-9B" has no such collision (Together-only) and
        // still proves a reasoning-capable Together-shaped qwen id gets the real toggle.
        let c = capabilities("Qwen/Qwen3.5-9B".to_ascii_lowercase().as_str());
        assert_eq!(c.openai_reasoning_format, OpenAiReasoningFormat::Together);
        assert!(!c.reasoning_effort);

        // The one Groq-hosted exception is matched by its exact id first and keeps the plain shape.
        let groq = capabilities("qwen/qwen3-32b");
        assert_eq!(groq.openai_reasoning_format, OpenAiReasoningFormat::Standard);
        assert!(groq.reasoning_effort);
    }

    #[test]
    fn qwen3_32b_uses_huggingfaces_smaller_real_max_output_for_this_shared_exact_id() {
        // pi-parity Task #18: "qwen/qwen3-32b" is served identically-spelled by both Groq (real
        // max_output 40960, `groq.models.ts:109`) and HuggingFace (real 16384,
        // `huggingface.models.ts:133`) — a same-string collision with no host signal to disambiguate
        // (context_window agrees at 131072 on both). HuggingFace's smaller, safer number wins.
        let c = capabilities("qwen/qwen3-32b");
        assert_eq!(c.context_window, 131_072);
        assert_eq!(c.max_output, 16_384, "was 40_960, over-reporting HuggingFace's real ceiling 2.5x");
    }

    #[test]
    fn generic_vendor_slug_fallback_beats_the_flat_unknown_default() {
        // An uncatalogued vendor/model-shaped id (most commonly OpenRouter) still gets meaningfully
        // better numbers than `ModelCaps::unknown()`'s flat 128k/4096.
        let c = capabilities("some-vendor/some-new-model");
        assert_eq!(c.openai_reasoning_format, OpenAiReasoningFormat::OpenRouter);
        assert_eq!(c.max_output, 32_000);
        assert!(c.max_output > ModelCaps::unknown().max_output);

        // A bare (no-slash) unrecognized id still falls all the way through to `unknown()`.
        let bare = capabilities("some-future-model-x");
        assert_eq!(bare.openai_reasoning_format, OpenAiReasoningFormat::Standard);
        assert_eq!(bare.max_output, 4_096);
    }

    #[test]
    fn essentialai_rnj_1_instruct_gets_its_real_capabilities_not_the_generic_vendor_slug_fallback() {
        // pi-parity Task #29: a brand-new Together model family with zero prior coverage — real
        // numbers (`together.models.ts:175`): 32768/32768, a ~4x smaller context than the generic
        // vendor-slug fallback's 128000.
        let c = capabilities("essentialai/Rnj-1-Instruct");
        assert_eq!(c.context_window, 32_768);
        assert_eq!(c.max_output, 32_768);
        assert!(!c.supports_vision);
        assert!(!c.reasoning_effort);
        assert_ne!(
            c.openai_reasoning_format,
            OpenAiReasoningFormat::OpenRouter,
            "must not land on the generic vendor-slug fallback"
        );
    }

    #[test]
    fn openrouter_meta_routing_pseudo_models_get_their_real_numbers_not_unknown_or_the_generic_fallback() {
        // pi-parity pass 20 Task 4: bare "auto" used to fall all the way through to
        // `ModelCaps::unknown()` (128k/4096); "openrouter/free"/"openrouter/fusion" (vendor-slug-shaped,
        // though neither names a real hosting vendor) used to hit the generic `m.contains('/')`
        // OpenRouter fallback (128000/32000) — a dangerous over-report specifically for
        // "openrouter/free", whose real max_output (4096) is *smaller*, not larger.
        let auto = capabilities("auto");
        assert_eq!(auto.context_window, 2_000_000);
        assert_eq!(auto.max_output, 30_000, "was 4_096 under ModelCaps::unknown()");
        assert!(auto.supports_vision);
        assert_eq!(auto.openai_reasoning_format, OpenAiReasoningFormat::OpenRouter);
        assert!(auto.max_output > ModelCaps::unknown().max_output);

        let free = capabilities("openrouter/free");
        assert_eq!(free.context_window, 200_000);
        assert_eq!(free.max_output, 4_096, "was 32_000 under the generic fallback, itself an over-report");
        assert!(free.supports_vision);

        let fusion = capabilities("openrouter/fusion");
        assert_eq!(fusion.context_window, 1_000_000, "was 128_000 under the generic fallback");
        assert_eq!(fusion.max_output, 30_000);
        assert!(!fusion.supports_vision);
    }

    #[test]
    fn opencode_zen_branded_free_tier_ids_get_their_real_numbers_not_unknown() {
        // pi-parity pass 20 Task 4: bare, unprefixed ids matching no other family's shape at all —
        // `opencode.models.ts`'s own branded free-tier offerings, all previously falling straight
        // through to `ModelCaps::unknown()`.
        let big_pickle = capabilities("big-pickle");
        assert_eq!(big_pickle.context_window, 200_000);
        assert_eq!(big_pickle.max_output, 32_000);
        assert!(!big_pickle.supports_vision);
        assert!(big_pickle.max_output > ModelCaps::unknown().max_output);

        let nemotron_free = capabilities("nemotron-3-ultra-free");
        assert_eq!(nemotron_free.context_window, 1_000_000);
        assert_eq!(nemotron_free.max_output, 128_000);

        let north_mini = capabilities("north-mini-code-free");
        assert_eq!(north_mini.context_window, 256_000);
        assert_eq!(north_mini.max_output, 64_000);

        // All three share the plain OpenAI-style bare `reasoning_effort` shape (no `thinkingFormat`
        // override in pi's catalogue), not OpenRouter's nested one.
        for caps in [big_pickle, nemotron_free, north_mini] {
            assert_eq!(caps.openai_reasoning_format, OpenAiReasoningFormat::Standard);
            assert_eq!(caps.max_tokens_field, MaxTokensField::MaxTokens);
        }
    }

    #[test]
    fn vercel_ai_gateway_zai_glm_5_2_gets_its_real_slightly_larger_context() {
        // pi-parity Task #30: Vercel AI Gateway's own vendor-slug spelling "zai/glm-5.2" (no "-org")
        // reports a real context of 1,040,000, not native's/NVIDIA's 1,000,000 — negligible (~4%) but
        // fixed for completeness.
        let c = capabilities("zai/glm-5.2");
        assert_eq!(c.context_window, 1_040_000);
        // The sibling "-fast" id is unaffected — its own real number already matches native's.
        assert_eq!(capabilities("zai/glm-5.2-fast").context_window, 1_000_000);
        // The Together/HuggingFace vendor-slug spelling ("zai-org/…") is a different string entirely,
        // unaffected by this one.
        assert_eq!(capabilities("zai-org/GLM-5.2").context_window, 262_144);
    }

    // ---- Vendor-slug family matching + MiMo (pi-parity remediation) ----

    #[test]
    fn together_hosted_vendor_slug_ids_hit_their_real_family_not_the_generic_fallback() {
        // Before `family_id`, a vendor-slug id whose org prefix doesn't happen to start with the
        // family name (`"moonshotai/Kimi-K2.6"` doesn't start with "kimi") fell all the way through to
        // the generic 128k/32k `ApiKind::ChatCompletions`/`OpenRouter`-shaped fallback further below —
        // wrong by roughly 2-8x depending on the field. Numbers ported from
        // `packages/ai/src/providers/together.models.ts`.
        let kimi = capabilities("moonshotai/Kimi-K2.6");
        assert_eq!(kimi.context_window, 262_144);
        assert!(kimi.supports_vision, "kimi-k2.6 is vision-capable");
        assert_eq!(kimi.openai_reasoning_format, OpenAiReasoningFormat::DeepSeek);
        assert_ne!(
            kimi.openai_reasoning_format,
            OpenAiReasoningFormat::OpenRouter,
            "must not land on the generic vendor-slug fallback"
        );

        // pi-parity Task #22: Together's real max_output for this exact vendor-slug id (131000,
        // `together.models.ts:230`) is much smaller than the generic "else" bucket's 262144, which
        // HuggingFace's identically-spelled entry legitimately matches (`huggingface.models.ts:619`).
        assert_eq!(kimi.max_output, 131_000, "was 262_144, a dangerous 2x over-report for Together");

        // "moonshotai/Kimi-K2.7-Code" on Together is real max_output 131072 (`together.models.ts:249`)
        // and, uniquely among every other vision-capable id in this family (including the bare native
        // id and HuggingFace's own identically-spelled vendor-slug entry), text-only on Together
        // specifically (`input: ["text"]`) — both the number and the vision flag must reflect that.
        //
        // pi-parity pass 20 Task 1: OpenRouter serves this identical vendor-slug string too, with an
        // even smaller real max_output (16384, `openrouter.models.ts`) — an 8x over-report if
        // Together's own 131072 were kept. OpenRouter's smaller number now wins the 3-way collision
        // (Together/HuggingFace/OpenRouter all serve this exact string).
        let code = capabilities("moonshotai/Kimi-K2.7-Code");
        assert_eq!(code.context_window, 262_144);
        assert_eq!(code.max_output, 16_384, "OpenRouter's real, smaller number now wins the collision");
        assert!(!code.supports_vision, "Together's own K2.7-Code entry is text-only");

        // The bare native id is unaffected — still the family default (262144/262144), vision-capable.
        // (HuggingFace's own real entry for "moonshotai/Kimi-K2.7-Code" is identically spelled to
        // Together's/OpenRouter's — the same same-string collision `together_vendor_slug_k2_7_code`
        // accepts; OpenRouter's smaller, safer number and Together's text-only flag win for this one
        // shared string.)
        let native_code = capabilities("kimi-k2.7-code");
        assert_eq!(native_code.max_output, 262_144);
        assert!(native_code.supports_vision);

        // pi-parity Task #15: this exact vendor-slug id's real context (262144, both
        // `together.models.ts:363` and `huggingface.models.ts:853`) is far smaller than NVIDIA's/
        // native's own 1,000,000 the bare "glm-5.2" id gets.
        let glm = capabilities("zai-org/GLM-5.2");
        assert_eq!(glm.context_window, 262_144, "Together/HuggingFace's real context, not native's 1M");
        assert_eq!(glm.max_output, 131_072);
        assert!(glm.reasoning_effort, "glm-5.2 has a real effort vocabulary");
        assert_eq!(glm.openai_reasoning_format, OpenAiReasoningFormat::Zai);

        // The native (bare) id is completely unaffected — still the larger 1,000,000.
        assert_eq!(capabilities("glm-5.2").context_window, 1_000_000);
    }

    #[test]
    fn huggingface_hosted_vendor_slug_ids_also_hit_their_real_family() {
        // A different aggregator, same org-slug id shape (`packages/ai/src/providers/
        // huggingface.models.ts`) — confirms the fix isn't Together-specific.
        //
        // pi-parity pass 20 Task 1: max_output is 100352, not HuggingFace's own 262144 — OpenRouter
        // serves this identical vendor-slug string too, with a smaller real number that now wins the
        // collision (this table's established safe-direction tie-break).
        let kimi = capabilities("moonshotai/Kimi-K2-Thinking");
        assert_eq!(kimi.context_window, 262_144);
        assert_eq!(kimi.max_output, 100_352);
        assert_eq!(kimi.openai_reasoning_format, OpenAiReasoningFormat::DeepSeek);

        let glm = capabilities("zai-org/GLM-4.7");
        assert_eq!(glm.context_window, 204_800);
        assert_eq!(glm.max_output, 131_072);
        assert_eq!(glm.openai_reasoning_format, OpenAiReasoningFormat::Zai);
    }

    #[test]
    fn vendor_slug_minimax_m3_isnt_shadowed_by_its_own_org_slug_prefix_collision() {
        // "MiniMaxAI/MiniMax-M3" lowercases to "minimaxai/minimax-m3" — the org slug ("minimaxai")
        // itself begins with the literal string "minimax" one character early. Selecting the match
        // target by "did the raw id already start with the family name" (rather than by whether the
        // id is slug-shaped at all) would silently land in the right family branch but match the
        // wrong per-id sub-case inside it: the flat, smaller "else" shape below instead of
        // "minimax-m3"'s real 1M-context/vision-capable one.
        //
        // pi-parity pass 20 Task 2: `max_output` is 16_384, not the family branch's own 128_000 — this
        // exact string is now intercepted earlier by `nvidia_caps` (NVIDIA's real, smaller number wins
        // the cross-host tie-break; see that function's own doc comment). Still proves the point this
        // test exists for: the id resolves to *a* real, correct-shaped entry (1M context, vision-
        // capable), not the unrelated smaller/non-vision "minimax-m2"-style default the org-slug-prefix
        // bug used to risk.
        let c = capabilities("MiniMaxAI/MiniMax-M3");
        assert_eq!(c.context_window, 1_000_000);
        assert_eq!(c.max_output, 16_384);
        assert!(c.supports_vision);
    }

    #[test]
    fn vendor_slug_deepseek_and_groq_qwen_ids_are_unaffected_by_family_id() {
        // These two already matched correctly before `family_id` existed (DeepSeek's org slug
        // "deepseek-ai" itself starts with "deepseek"; Groq's real qwen id already carries its own
        // slash). Regression guard that the refactor didn't change either.
        //
        // pi-parity Task #20: this exact id now gets its own real (smaller) context — see
        // `deepseek_ai_deepseek_v4_pro_gets_togethers_smaller_real_context` below — rather than the
        // family-wide 1,000,000 default; max_output is unaffected (Together's real number already
        // matches that default).
        let ds = capabilities("deepseek-ai/DeepSeek-V4-Pro");
        assert_eq!(ds.context_window, 512_000);
        assert_eq!(ds.max_output, 384_000);

        let groq = capabilities("qwen/qwen3-32b");
        assert_eq!(groq.openai_reasoning_format, OpenAiReasoningFormat::Standard);
        assert!(groq.reasoning_effort);
    }

    // ---- Task 11: family_id/vendor-slug matching extended to Claude/OpenAI/Grok ----

    #[test]
    fn openrouter_vendor_slug_claude_ids_hit_the_real_claude_branch_not_the_generic_fallback() {
        // "anthropic/claude-opus-4.6" (OpenRouter) used to fall all the way through to the generic
        // vendor-slug fallback (128k/32k, `OpenAiReasoningFormat::OpenRouter`) since the Claude branch
        // never matched a vendor-slug id at all. It must now hit the real gen6+ Claude shape.
        let c = capabilities("anthropic/claude-opus-4.6");
        assert_eq!(c.context_window, 1_000_000);
        assert_eq!(c.thinking, ThinkingShape::Adaptive);
        assert!(c.supports_vision);
        // Claude ids never use the OpenAI-wire reasoning format at all, vendor-slug or not.
        assert_eq!(c.openai_reasoning_format, OpenAiReasoningFormat::Standard);

        let sonnet = capabilities("anthropic/claude-sonnet-5");
        assert_eq!(sonnet.context_window, 1_000_000);
        assert_eq!(sonnet.max_output, 128_000);
    }

    #[test]
    fn openrouter_vendor_slug_openai_ids_hit_the_real_family_but_keep_chat_completions_wire() {
        // "openai/gpt-5.2" (OpenRouter) must land on the real gpt-5 shape (context/output/reasoning
        // numbers), but — critically — must NOT inherit the native branch's `ApiKind::Responses`: every
        // current OpenRouter entry for these ids is served over Chat Completions
        // (`api: "openai-completions"` in pi's own `openrouter.models.ts`), and OpenRouter's endpoint
        // doesn't speak the Responses API at all. Getting this wrong would send a Responses-shaped
        // request to an endpoint that can't parse it.
        let c = capabilities("openai/gpt-5.2");
        assert_eq!(c.context_window, 400_000);
        assert_eq!(c.max_output, 128_000);
        assert!(c.reasoning_effort);
        assert_eq!(
            c.api,
            ApiKind::ChatCompletions,
            "a vendor-slug OpenAI id must stay off the Responses API"
        );
        assert_eq!(c.max_tokens_field, MaxTokensField::MaxTokens);
        assert_eq!(c.openai_reasoning_format, OpenAiReasoningFormat::OpenRouter);

        // The native (bare) id is completely unaffected: still Responses-routed, still the OpenAI
        // Standard reasoning format, still MaxCompletionTokens.
        let native = capabilities("gpt-5.2");
        assert_eq!(native.api, ApiKind::Responses);
        assert_eq!(native.max_tokens_field, MaxTokensField::MaxCompletionTokens);
        assert_eq!(native.openai_reasoning_format, OpenAiReasoningFormat::Standard);

        // o-series and gpt-4 vendor-slug ids get the same treatment.
        let o3 = capabilities("openai/o3");
        assert_eq!(o3.api, ApiKind::ChatCompletions);
        assert_eq!(o3.context_window, 200_000);
        let gpt4o = capabilities("openai/gpt-4o");
        assert_eq!(gpt4o.api, ApiKind::ChatCompletions);
        assert!(gpt4o.supports_vision);
    }

    #[test]
    fn openrouter_vendor_slug_grok_ids_get_the_real_family_with_openrouters_smaller_output_ceiling() {
        // Task 13's explicit regression: "x-ai/grok-4.3" used to land on the generic vendor-slug
        // fallback (max_output: 32_000) — already an over-report of OpenRouter's real ceiling
        // (`openrouter.models.ts`: maxTokens 4096 for every current xAI id there). Extending
        // `family_id` matching alone would still over-report (native's 30_000) — the vendor-slug clamp
        // is what actually fixes the dangerous direction.
        let c = capabilities("x-ai/grok-4.3");
        assert_eq!(c.context_window, 1_000_000);
        assert_eq!(
            c.max_output, 4_096,
            "must not over-report OpenRouter's real (much smaller) output ceiling"
        );
        assert!(c.supports_vision);
        // xAI never accepts a client-steerable reasoning toggle at all, vendor-slug or not — must not
        // gain a mechanism just because the id is now recognized.
        assert!(!c.reasoning_effort);
        assert_eq!(c.openai_reasoning_format, OpenAiReasoningFormat::Standard);

        // The native (bare) id keeps its real, larger ceiling — the clamp is vendor-slug-only.
        let native = capabilities("grok-4.3");
        assert_eq!(native.max_output, 30_000);

        // grok-build's real OpenRouter number happens to already fit the clamp's range for context
        // (its own branch already scopes it separately below); the max_output clamp still applies.
        let build = capabilities("x-ai/grok-build-0.1");
        assert_eq!(build.max_output, 4_096);
    }

    // ---- Task 12: GitHub Copilot's dot-spelled Claude ids ----

    #[test]
    fn copilot_dot_spelled_claude_ids_normalize_and_get_copilots_own_numbers() {
        // Before normalization, "claude-opus-4.6" (Copilot's own spelling, `github-copilot.models.ts`)
        // matched no `GEN6_PLUS` entry at all (all dash-spelled) and fell to the pre-gen6 Budget-shape
        // bucket — wrong shape entirely, plus Copilot's own numbers diverge from native Anthropic's.
        let opus46 = capabilities("claude-opus-4.6");
        assert_eq!(opus46.thinking, ThinkingShape::Adaptive, "must resolve to gen6+ shape");
        assert_eq!(opus46.context_window, 1_000_000);
        assert_eq!(opus46.max_output, 32_000, "Copilot's real ceiling, not native's 128_000");
        assert!(opus46.supports_temperature);

        let opus47 = capabilities("claude-opus-4.7");
        assert_eq!(
            opus47.context_window, 200_000,
            "Copilot's opus-4.7 context is smaller than native's 1M"
        );
        assert_eq!(opus47.max_output, 32_000);
        assert!(!opus47.supports_temperature, "opus-4.7 rejects temperature on every host");

        let opus48 = capabilities("claude-opus-4.8");
        assert_eq!(opus48.context_window, 200_000);
        assert_eq!(
            opus48.max_output, 64_000,
            "Copilot's real opus-4.8 max_output, not the 32_000 the generic bucket used to report"
        );
        assert!(!opus48.supports_temperature);

        let sonnet46 = capabilities("claude-sonnet-4.6");
        assert_eq!(
            sonnet46.context_window, 1_000_000,
            "Copilot's sonnet-4.6 context, not the 200_000 the generic bucket used to report"
        );
        assert_eq!(sonnet46.max_output, 32_000);

        // Every other dot-spelled Copilot Claude id (no numeric override) still normalizes and resolves
        // to a real Claude shape rather than falling through to a wrong bucket.
        let fable5 = capabilities("claude-fable-5");
        assert_eq!(fable5.thinking, ThinkingShape::Adaptive);
        let sonnet5 = capabilities("claude-sonnet-5");
        assert_eq!(sonnet5.context_window, 1_000_000);

        // The native, dash-spelled ids are completely unaffected by the Copilot override.
        let native_opus46 = capabilities("claude-opus-4-6");
        assert_eq!(native_opus46.max_output, 128_000);
        assert_eq!(native_opus46.context_window, 1_000_000);
    }

    /// pi-parity (models/dialects pass): Copilot's `thinkingLevelMap` for "claude-sonnet-4.6" is
    /// `{"minimal":"low","xhigh":"max"}` (`github-copilot.models.ts`) — `adaptive_xhigh_effort_wire`
    /// must be `"max"`, not the generic `"xhigh"` literal every other adaptive-shape id on this route
    /// sends. Sibling id `opus-4.6` already got this right before this fix.
    #[test]
    fn copilot_sonnet_4_6_remaps_xhigh_to_max_like_its_opus_sibling() {
        let sonnet46 = capabilities("claude-sonnet-4.6");
        assert_eq!(sonnet46.adaptive_xhigh_effort_wire, "max", "was \"xhigh\" before this fix");
        let opus46 = capabilities("claude-opus-4.6");
        assert_eq!(opus46.adaptive_xhigh_effort_wire, "max", "opus-4.6 must be unaffected");
        // The native, dash-spelled sonnet-4-6 has no xhigh wire value at all (`supports_xhigh_reasoning:
        // false`) — this field is unread there and must stay unaffected by the Copilot fix.
        assert!(!capabilities("claude-sonnet-4-6").supports_xhigh_reasoning);
    }

    #[test]
    fn copilot_bare_claude_sonnet_4_gets_its_own_real_numbers_not_the_generic_bucket() {
        // pi-parity Task #5: unlike every other Copilot Claude id, "claude-sonnet-4" has no
        // version-dot at all — `is_dot_spelled` can never catch it — so it used to fall through to the
        // generic Budget-shape bucket's `sonnet` default (context 200_000/max_output 64_000) instead of
        // Copilot's real, much smaller numbers (`github-copilot.models.ts`: 216_000/16_000, a 4x
        // max_output over-report).
        let c = capabilities("claude-sonnet-4");
        assert_eq!(c.context_window, 216_000);
        assert_eq!(c.max_output, 16_000);
        assert_eq!(c.thinking, ThinkingShape::Budget);

        // No native/OpenRouter id of this exact bare spelling exists — a vendor-slug id sharing the
        // same suffix is unaffected (falls to the generic bucket, not this exact-id override).
        let vendor_slug = capabilities("anthropic/claude-sonnet-4");
        assert_ne!(vendor_slug.max_output, 16_000);
    }

    #[test]
    fn copilot_opus_4_5_and_sonnet_4_5_get_copilots_smaller_max_output_but_stay_budget_shaped() {
        // pi-parity Tasks #6/#7: Copilot's own "claude-opus-4.5"/"claude-sonnet-4.5" cap max_output at
        // 32_000 (`github-copilot.models.ts`), not this bucket's generic 64_000 — but, unlike
        // opus-4-6/4-7/4-8/sonnet-4-6, neither is Adaptive-shape on Copilot (no `forceAdaptiveThinking`/
        // `thinkingLevelMap` for either), so they must NOT be added to
        // `github_copilot_claude_overrides`'s allowlist (which would incorrectly also switch them to
        // the Adaptive wire shape) — they stay in this generic Budget-shape bucket, just with a smaller
        // max_output.
        for id in ["claude-opus-4.5", "claude-sonnet-4.5"] {
            let c = capabilities(id);
            assert_eq!(c.context_window, 200_000, "{id}: Copilot's context matches this bucket already");
            assert_eq!(c.max_output, 32_000, "{id}: Copilot's real, smaller ceiling");
            assert_eq!(c.thinking, ThinkingShape::Budget, "{id}: must stay Budget-shape, not Adaptive");
        }

        // The native, dash-spelled ids are completely unaffected — still the generic 64_000.
        assert_eq!(capabilities("claude-opus-4-5").max_output, 64_000);
        assert_eq!(capabilities("claude-sonnet-4-5").max_output, 64_000);

        // A vendor-slug dot-spelled id (OpenRouter) is also unaffected — the override is bare-id-only.
        let vendor_slug = capabilities("anthropic/claude-opus-4.5");
        assert_eq!(vendor_slug.max_output, 64_000);
    }

    #[test]
    fn copilot_mai_code_1_flash_picker_gets_real_capabilities_not_the_unknown_default() {
        let c = capabilities("mai-code-1-flash-picker");
        assert_eq!(c.context_window, 256_000);
        assert_eq!(c.max_output, 128_000);
        assert!(!c.supports_vision);
        assert!(!c.reasoning_effort, "compat.supportsReasoningEffort is false despite reasoning:true");
        assert!(c.max_output > ModelCaps::unknown().max_output);
    }

    // ---- Task 14: Kimi-Coding catalog numbers ----

    #[test]
    fn kimi_coding_k2p7_alias_gets_real_capabilities_not_the_unknown_default() {
        // "k2p7" doesn't start with "kimi" at all, so it fell all the way through to
        // `ModelCaps::unknown()` before this fix (no vendor slug either, so the generic vendor-slug
        // fallback never applied). Real numbers: `kimi-coding.models.ts`.
        let c = capabilities("k2p7");
        assert_eq!(c.context_window, 262_144);
        assert_eq!(c.max_output, 32_768, "Kimi-Coding's real ceiling, not the generic bucket's 262_144");
        assert!(c.supports_vision);
        assert!(c.max_output > ModelCaps::unknown().max_output);
    }

    #[test]
    fn kimi_for_coding_gets_kimi_codings_smaller_real_max_output() {
        // "kimi-for-coding" matched the generic Kimi bucket before this fix (262_144/262_144) — an
        // over-report of its real, much smaller ceiling (32_768) and no vision, unlike every other id
        // in this bucket that shares the "else" default. Not a collision (no moonshotai-native id of
        // this exact name exists), so it's fully corrected rather than just documented.
        let c = capabilities("kimi-for-coding");
        assert_eq!(c.context_window, 262_144);
        assert_eq!(c.max_output, 32_768);
        assert!(c.supports_vision);
    }

    #[test]
    fn kimi_coding_kimi_k2_thinking_gets_kimi_codings_smaller_real_max_output() {
        // pi-parity Task #14: unlike "k2p7"/"kimi-for-coding", bare "kimi-k2-thinking" *does* collide
        // with moonshotai-native's own identically-spelled id (which correctly gets 262144/262144 via
        // this bucket's "else" default) — same bug class as the documented kimi-k2.7-code/Copilot
        // collision. Kimi-Coding's smaller, safer max_output (32768) wins here: an 8x over-report for
        // Kimi-Coding (dangerous) is worse than an 8x under-report for moonshotai-native (safe).
        // Neither host reports vision for this id, unlike k2p7/kimi-for-coding.
        let c = capabilities("kimi-k2-thinking");
        assert_eq!(c.context_window, 262_144);
        assert_eq!(c.max_output, 32_768, "was 262_144, a dangerous 8x over-report for Kimi-Coding");
        assert!(!c.supports_vision, "neither Kimi-Coding nor moonshotai-native report vision for this id");

        // HuggingFace's own vendor-slug spelling of this id reports the *native* (larger) numbers, not
        // Kimi-Coding's — the bare-id-only scoping must not affect it.
        //
        // pi-parity pass 20 Task 1: OpenRouter's identical vendor-slug spelling reports a real
        // max_output of 100352 — smaller than HuggingFace's 262144, so it now wins this same-string
        // collision instead (see `openrouter_vendor_slug_k2_thinking` in the capability table).
        let hf = capabilities("moonshotai/Kimi-K2-Thinking");
        assert_eq!(hf.context_window, 262_144);
        assert_eq!(hf.max_output, 100_352, "OpenRouter's smaller real number now wins the collision");
    }

    // ---- Task 25: Google/Gemini capability data ----

    #[test]
    fn gemini_ids_get_real_capabilities_not_the_unknown_default() {
        for id in ["gemini-2.5-pro", "gemini-3-flash-preview", "gemini-3.1-pro-preview", "gemini-flash-latest"] {
            let c = capabilities(id);
            assert_eq!(c.context_window, 1_048_576, "{id}");
            assert_eq!(c.max_output, 65_536, "{id}");
            assert!(c.supports_vision, "{id}");
            assert!(c.max_output > ModelCaps::unknown().max_output, "{id}");
        }
        // The 2.0-generation legacy ids share a smaller, non-reasoning shape.
        for id in ["gemini-2.0-flash", "gemini-2.0-flash-lite"] {
            let c = capabilities(id);
            assert_eq!(c.context_window, 1_048_576, "{id}");
            assert_eq!(c.max_output, 8_192, "{id}");
        }
    }

    #[test]
    fn google_native_gemma_4_ids_get_their_own_smaller_shape_distinct_from_the_vendor_slug_ones() {
        for id in ["gemma-4-26b-a4b-it", "gemma-4-31b-it"] {
            let c = capabilities(id);
            assert_eq!(c.context_window, 262_144, "{id}");
            assert_eq!(c.max_output, 32_768, "{id}");
            assert!(c.supports_vision, "{id}");
        }
        // Distinct from the Cerebras-native bare "gemma-4-31b" (no "-it" suffix) and the OpenRouter/
        // HuggingFace vendor-slug "google/gemma-4-31b-it" — three different id strings, three
        // (potentially) different real hosts, no collision between any of them.
        assert_eq!(capabilities("gemma-4-31b").max_output, 40_960);
        assert_eq!(capabilities("google/gemma-4-31b-it").max_output, 32_768);
    }

    #[test]
    fn gemini_never_claims_a_reasoning_mechanism_since_no_dialect_reads_it_yet() {
        // Deliberately inert: the Gemini-direct dialect remains a deferred non-goal, so this data-only
        // entry must never claim a mechanism a misrouted request could act on incorrectly.
        for id in ["gemini-2.5-pro", "gemini-3-pro-preview", "gemma-4-31b-it"] {
            assert!(!has_reasoning_mechanism(&capabilities(id)), "{id}");
        }
    }

    // ---- Task 24: native MiniMax reasoning_disableable ----

    #[test]
    fn native_minimax_ids_are_disable_capable_not_hardcoded_false() {
        // pi's minimax.models.ts/minimax-cn.models.ts carry no thinkingLevelMap at all for any of the
        // 3 current ids — no id nulls "off" — so the native (bare) id is disable-capable.
        for id in ["MiniMax-M2.7", "MiniMax-M2.7-highspeed", "MiniMax-M3"] {
            assert!(capabilities(id).reasoning_disableable, "{id}");
        }
    }

    #[test]
    fn vendor_slug_minimax_ids_stay_not_disable_capable() {
        // Together's own MiniMax entries explicitly null "off" (`thinkingLevelMap: {"off":null,…}`) —
        // a vendor-slug-reached id must not inherit the native fix above.
        assert!(!capabilities("MiniMaxAI/MiniMax-M2.7").reasoning_disableable);
        assert!(!capabilities("MiniMaxAI/MiniMax-M3").reasoning_disableable);
    }

    // ---- Task 23: route-aware OpenAI Codex/Azure capability overrides ----

    #[test]
    fn capabilities_for_route_is_a_no_op_for_the_native_route() {
        for id in ["gpt-5.3-codex-spark", "gpt-5.4", "gpt-5.4-mini", "gpt-5.5", "gpt-5.1", "gpt-4o"] {
            assert_eq!(
                capabilities_for_route(id, false, false),
                capabilities(id),
                "{id}: both flags false must be a complete no-op"
            );
        }
    }

    #[test]
    fn capabilities_for_route_codex_spark_loses_max_output_and_vision() {
        let native = capabilities_for_route("gpt-5.3-codex-spark", false, false);
        assert_eq!(native.max_output, 32_000);
        assert!(native.supports_vision);

        let codex = capabilities_for_route("gpt-5.3-codex-spark", true, false);
        assert_eq!(codex.max_output, 128_000, "Codex's own entry ships a 128k ceiling, not 32k");
        assert!(!codex.supports_vision, "Codex's own entry is input: [\"text\"] only");
        // Context window is unaffected — identical on all three routes.
        assert_eq!(codex.context_window, native.context_window);
    }

    #[test]
    fn capabilities_for_route_gpt_5_4_and_5_5_get_azures_larger_context() {
        for id in ["gpt-5.4", "gpt-5.5"] {
            let native = capabilities_for_route(id, false, false);
            assert_eq!(native.context_window, 272_000, "{id}");
            let azure = capabilities_for_route(id, false, true);
            assert_eq!(azure.context_window, 1_050_000, "{id}: Azure's real, much larger context");
            // Codex is unaffected for these two ids (only gpt-5.4-mini diverges on Codex).
            let codex = capabilities_for_route(id, true, false);
            assert_eq!(codex.context_window, 272_000, "{id}");
        }
    }

    #[test]
    fn capabilities_for_route_gpt_5_4_mini_gets_codexs_smaller_context() {
        let native = capabilities_for_route("gpt-5.4-mini", false, false);
        assert_eq!(native.context_window, 400_000);
        let codex = capabilities_for_route("gpt-5.4-mini", true, false);
        assert_eq!(codex.context_window, 272_000, "Codex's own real, smaller context");
        let azure = capabilities_for_route("gpt-5.4-mini", false, true);
        assert_eq!(azure.context_window, 400_000, "Azure matches native for this one id");
    }

    #[test]
    fn capabilities_for_route_disable_capable_ids_lose_their_off_signal_on_both_other_routes() {
        for id in [
            "gpt-5.1",
            "gpt-5.2",
            "gpt-5.3-codex",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.4-nano",
            "gpt-5.5",
        ] {
            assert!(
                capabilities_for_route(id, false, false).reasoning_disableable,
                "{id}: must stay disable-capable natively"
            );
            assert!(
                !capabilities_for_route(id, true, false).reasoning_disableable,
                "{id}: Codex's catalogue has no off wire value at all"
            );
            assert!(
                !capabilities_for_route(id, false, true).reasoning_disableable,
                "{id}: Azure nulls thinkingLevelMap.off outright"
            );
        }
        // An id that was never disable-capable natively (bare "gpt-5") stays that way regardless of
        // route — this override only ever removes a signal, never grants one.
        assert!(!capabilities_for_route("gpt-5", true, false).reasoning_disableable);
        assert!(!capabilities_for_route("gpt-5", false, true).reasoning_disableable);
    }

    #[test]
    fn capabilities_for_route_leaves_non_openai_ids_completely_unaffected() {
        // A route flag being (incorrectly, hypothetically) set true for a model Codex/Azure never
        // actually serves must still be harmless — nothing in this table keys route-awareness off
        // anything but the specific 4 (or 7, for the off-signal check) OpenAI ids above.
        for id in ["claude-opus-4-8", "deepseek-v4-pro", "glm-5.2", "kimi-k2-thinking"] {
            assert_eq!(capabilities_for_route(id, true, false), capabilities(id), "{id}");
            assert_eq!(capabilities_for_route(id, false, true), capabilities(id), "{id}");
        }
    }

    #[test]
    fn capabilities_for_route_with_copilot_is_a_no_op_when_not_copilot_routed() {
        for id in ["gpt-4.1", "gpt-5-mini", "gpt-5.4", "gpt-5.5", "gemini-2.5-pro", "gpt-4o"] {
            assert_eq!(
                capabilities_for_route_with_copilot(id, false, false, false),
                capabilities_for_route(id, false, false),
                "{id}: is_copilot=false must be a complete no-op"
            );
        }
    }

    #[test]
    fn capabilities_for_route_with_copilot_gpt_4_1_gets_copilots_real_numbers() {
        // pi-parity Task #9: native gpt-4.1 is ~1.05M/32768; Copilot's real numbers are a much smaller
        // 128000/16384 — an 8x context over-report was the single largest miss this pass.
        let native = capabilities_for_route_with_copilot("gpt-4.1", false, false, false);
        assert_eq!(native.context_window, 1_047_576);
        assert_eq!(native.max_output, 32_768);

        let copilot = capabilities_for_route_with_copilot("gpt-4.1", false, false, true);
        assert_eq!(copilot.context_window, 128_000, "Copilot's real, much smaller context");
        assert_eq!(copilot.max_output, 16_384, "Copilot's real, much smaller output ceiling");
    }

    #[test]
    fn capabilities_for_route_with_copilot_gpt_5_mini_gets_copilots_real_numbers() {
        // pi-parity Task #10.
        let native = capabilities_for_route_with_copilot("gpt-5-mini", false, false, false);
        assert_eq!(native.context_window, 400_000);
        assert_eq!(native.max_output, 128_000);

        let copilot = capabilities_for_route_with_copilot("gpt-5-mini", false, false, true);
        assert_eq!(copilot.context_window, 264_000);
        assert_eq!(copilot.max_output, 64_000);
    }

    #[test]
    fn capabilities_for_route_with_copilot_gpt_5_4_and_5_5_get_copilots_larger_context() {
        // pi-parity Task #11: bare gpt-5.4/gpt-5.5 report 272000 context natively; Copilot's real
        // number is the larger 400000 (a safe-direction under-report before this fix, not a hard bug,
        // but still worth correcting for accuracy).
        for id in ["gpt-5.4", "gpt-5.5"] {
            let native = capabilities_for_route_with_copilot(id, false, false, false);
            assert_eq!(native.context_window, 272_000, "{id}");
            let copilot = capabilities_for_route_with_copilot(id, false, false, true);
            assert_eq!(copilot.context_window, 400_000, "{id}: Copilot's real, larger context");
        }
    }

    #[test]
    fn capabilities_for_route_with_copilot_gemini_ids_get_copilots_real_numbers() {
        // pi-parity Task #8: native Google numbers are 1,048,576/65,536 for every current Gemini id;
        // Copilot's real numbers are much smaller and split into two pairs.
        for id in ["gemini-2.5-pro", "gemini-3-flash-preview"] {
            let native = capabilities_for_route_with_copilot(id, false, false, false);
            assert_eq!(native.context_window, 1_048_576, "{id}");
            let copilot = capabilities_for_route_with_copilot(id, false, false, true);
            assert_eq!(copilot.context_window, 128_000, "{id}");
            assert_eq!(copilot.max_output, 64_000, "{id}");
        }
        for id in ["gemini-3.1-pro-preview", "gemini-3.5-flash"] {
            let copilot = capabilities_for_route_with_copilot(id, false, false, true);
            assert_eq!(copilot.context_window, 200_000, "{id}");
            assert_eq!(copilot.max_output, 64_000, "{id}");
        }
    }

    #[test]
    fn capabilities_for_route_with_copilot_leaves_unrelated_ids_and_codex_azure_overrides_intact() {
        // Copilot-awareness must compose with, not replace, the existing Codex/Azure overrides.
        let codex = capabilities_for_route_with_copilot("gpt-5.3-codex-spark", true, false, false);
        assert_eq!(codex.max_output, 128_000, "Codex's own override must still apply");

        // An id Copilot never serves is untouched even if (hypothetically) is_copilot were true.
        let unaffected = capabilities_for_route_with_copilot("deepseek-v4-pro", false, false, true);
        assert_eq!(unaffected, capabilities("deepseek-v4-pro"));
    }

    #[test]
    fn responses_minimal_effort_wire_override_remaps_only_on_codex_or_copilot_routes() {
        use crate::transport::ReasoningEffort as RE;
        for id in ["gpt-5.3-codex-spark", "gpt-5.4", "gpt-5.4-mini", "gpt-5.5"] {
            assert_eq!(
                responses_minimal_effort_wire_override(id, true, false, RE::Minimal),
                Some("low"),
                "{id}: Codex must remap minimal to low"
            );
            assert_eq!(
                responses_minimal_effort_wire_override(id, false, true, RE::Minimal),
                Some("low"),
                "{id}: Copilot must remap minimal to low"
            );
            // Neither route: no remap.
            assert_eq!(
                responses_minimal_effort_wire_override(id, false, false, RE::Minimal),
                None,
                "{id}: native route must not remap"
            );
            // A non-minimal effort is never remapped, even on Codex/Copilot.
            assert_eq!(
                responses_minimal_effort_wire_override(id, true, false, RE::High),
                None,
                "{id}: only minimal is remapped"
            );
        }
        // An id not in the remap list is unaffected even on Codex/Copilot.
        assert_eq!(
            responses_minimal_effort_wire_override("gpt-5.1", true, false, RE::Minimal),
            None
        );
    }

    /// pi-parity (models/dialects pass): 5 Copilot-only gpt-5.x ids (`github-copilot.models.ts`) were
    /// missing from `REMAPPED_IDS` entirely — each sent the wire string `"minimal"` verbatim instead of
    /// Copilot's required `"low"`. None of these 5 exist in `openai-codex.models.ts`'s catalogue at all,
    /// so `is_codex` never legitimately fires for them in practice, but the shared gate is still
    /// exercised here for completeness.
    #[test]
    fn responses_minimal_effort_wire_override_covers_the_5_copilot_only_gpt_5_x_ids() {
        use crate::transport::ReasoningEffort as RE;
        for id in [
            "gpt-5-mini",
            "gpt-5.2",
            "gpt-5.2-codex",
            "gpt-5.3-codex",
            "gpt-5.4-nano",
        ] {
            assert_eq!(
                responses_minimal_effort_wire_override(id, false, true, RE::Minimal),
                Some("low"),
                "{id}: Copilot must remap minimal to low"
            );
            assert_eq!(
                responses_minimal_effort_wire_override(id, false, false, RE::Minimal),
                None,
                "{id}: native route must not remap"
            );
        }
    }

    // ---- Task 22: HuggingFace fresh gaps ----

    #[test]
    fn huggingface_qwen_3_5_and_3_6_ids_are_vision_capable_via_the_generic_bucket() {
        for id in [
            "Qwen/Qwen3.5-122B-A10B",
            "Qwen/Qwen3.5-27B",
            "Qwen/Qwen3.6-27B",
            "Qwen/Qwen3.6-35B-A3B",
        ] {
            assert!(capabilities(id).supports_vision, "{id}");
        }
        // The Together-specific exact table's own text-only ids are unaffected by widening the
        // fallback's vision check (they return before ever reaching it).
        assert!(!capabilities("Qwen/Qwen3.6-Plus").supports_vision);
        assert!(!capabilities("Qwen/Qwen3.7-Max").supports_vision);
    }

    #[test]
    fn huggingface_deepseek_r1_and_v3_2_get_their_real_much_smaller_numbers() {
        let r1 = capabilities("deepseek-ai/DeepSeek-R1");
        assert_eq!(r1.context_window, 64_000, "was 1_000_000 under the flat family bucket");
        assert_eq!(r1.max_output, 32_768, "was 384_000 under the flat family bucket");

        let r1_0528 = capabilities("deepseek-ai/DeepSeek-R1-0528");
        assert_eq!(r1_0528.context_window, 163_840);
        assert_eq!(r1_0528.max_output, 163_840);

        let v32 = capabilities("deepseek-ai/DeepSeek-V3.2");
        assert_eq!(v32.context_window, 163_840);
        assert_eq!(v32.max_output, 65_536);

        // A differently-prefixed id sharing the same bare suffix (OpenRouter's own naming) is a
        // *different* full string, matched by its own exact-id entry rather than falling through to
        // the flat family bucket — pi-parity (models/dialects pass): OpenRouter's real numbers
        // (163840/16000, `openrouter.models.ts`) are dramatically smaller than the family-wide default
        // this test used to assert here (1_000_000/384_000) as if it were OpenRouter's own real number,
        // when it was actually just the flat bucket's untouched fallback.
        let openrouter_style = capabilities("deepseek/deepseek-r1");
        assert_eq!(openrouter_style.context_window, 163_840, "was 1_000_000 under the flat family bucket");
        assert_eq!(openrouter_style.max_output, 16_000, "was 384_000 under the flat family bucket");
    }

    #[test]
    fn deepseek_ai_deepseek_v4_pro_gets_togethers_smaller_real_context() {
        // pi-parity Task #20: this exact id is served by both Together (real context 512000,
        // `together.models.ts:156`, max_output 384000 already matching the family default) and
        // HuggingFace (real context 1048576, `huggingface.models.ts:475`, near enough to the family
        // default's 384000 max_output too) — a same-string collision with no host signal to
        // disambiguate. Together's smaller, safer context wins: the family-wide 1,000,000 default was a
        // 2x over-report for Together.
        let c = capabilities("deepseek-ai/DeepSeek-V4-Pro");
        assert_eq!(c.context_window, 512_000, "was 1_000_000 under the flat family bucket");
        assert_eq!(c.max_output, 384_000, "unaffected — already matched the family default");

        // The bare native id (not this exact vendor-slug string) is unaffected.
        assert_eq!(capabilities("deepseek-v4-pro").context_window, 1_000_000);
    }

    #[test]
    fn huggingface_deepseek_v4_flash_gets_its_real_larger_context() {
        // pi-parity (models/dialects pass): HuggingFace-only, real 1048576/384000
        // (`huggingface.models.ts`) — max_output already matched the family default; only
        // context_window was under-reported (~4.6%).
        let c = capabilities("deepseek-ai/DeepSeek-V4-Flash");
        assert_eq!(c.context_window, 1_048_576, "was 1_000_000 under the flat family bucket");
        assert_eq!(c.max_output, 384_000, "unaffected — already matched the family default");
    }

    #[test]
    fn openrouter_gpt_4_1_and_gpt_5_nano_get_their_real_much_smaller_max_output() {
        // pi-parity (models/dialects pass): OpenRouter's own vendor-slug "openai/gpt-4.1" reports a
        // real max_output of 4096 (`openrouter.models.ts`), a dramatic over-report vs the 32768 the
        // generic vendor-slug adjustment used to leave untouched.
        let gpt41 = capabilities("openai/gpt-4.1");
        assert_eq!(gpt41.context_window, 1_047_576);
        assert_eq!(gpt41.max_output, 4_096, "was 32_768 before this fix");

        // The "-mini"/"-nano" siblings already matched the generic default for real — must stay
        // unaffected by the bare "gpt-4.1" fix.
        assert_eq!(capabilities("openai/gpt-4.1-mini").max_output, 32_768);
        assert_eq!(capabilities("openai/gpt-4.1-nano").max_output, 32_768);
        // The native, unprefixed id is also unaffected.
        assert_eq!(capabilities("gpt-4.1").max_output, 32_768);

        // pi-parity (models/dialects pass): OpenRouter's own vendor-slug "openai/gpt-5-nano" reports a
        // real max_output of 4096, vs the generic gpt-5 bucket's 128000.
        let nano = capabilities("openai/gpt-5-nano");
        assert_eq!(nano.context_window, 400_000);
        assert_eq!(nano.max_output, 4_096, "was 128_000 before this fix");
        // The native, unprefixed id is unaffected.
        assert_eq!(capabilities("gpt-5-nano").max_output, 128_000);
    }

    #[test]
    fn openrouter_z_ai_glm_5_and_moonshotai_kimi_k2_5_get_their_real_much_smaller_max_output() {
        // pi-parity (models/dialects pass): OpenRouter's own vendor-slug "z-ai/glm-5" (distinct from
        // Together's/HuggingFace's "zai-org/glm-5" below — a different full id string) reports a real
        // 202752/4096 (`openrouter.models.ts`) vs the generic GLM bucket's 200000/131072.
        let glm5_openrouter = capabilities("z-ai/glm-5");
        assert_eq!(glm5_openrouter.context_window, 202_752);
        assert_eq!(glm5_openrouter.max_output, 4_096, "was 131_072 before this fix");

        // Together and HuggingFace both host "zai-org/glm-5"/"zai-org/glm-5.1" and agree on the real
        // numbers (202752/131072) — no collision, just a small context correction.
        for id in ["zai-org/glm-5", "zai-org/glm-5.1"] {
            let c = capabilities(id);
            assert_eq!(c.context_window, 202_752, "{id}");
            assert_eq!(c.max_output, 131_072, "{id}");
        }

        // HuggingFace-only small drifts: GLM-4.6 (context only) and GLM-4.7-Flash (both fields).
        let glm46 = capabilities("zai-org/glm-4.6");
        assert_eq!(glm46.context_window, 204_800);
        assert_eq!(glm46.max_output, 131_072);
        let glm47flash = capabilities("zai-org/glm-4.7-flash");
        assert_eq!(glm47flash.context_window, 200_000, "was 204_800 before this fix");
        assert_eq!(glm47flash.max_output, 128_000, "was 131_072 before this fix");
        // Bare native "glm-4.7" (no "-flash" suffix) is unaffected.
        assert_eq!(capabilities("glm-4.7").context_window, 204_800);
        assert_eq!(capabilities("glm-4.7").max_output, 131_072);

        // pi-parity (models/dialects pass): OpenRouter's own vendor-slug "moonshotai/kimi-k2.5" reports
        // a real max_output of 4096 (`openrouter.models.ts`), vs the generic Kimi bucket's 262144.
        let kimi = capabilities("moonshotai/kimi-k2.5");
        assert_eq!(kimi.context_window, 262_144);
        assert_eq!(kimi.max_output, 4_096, "was 262_144 before this fix");
        assert!(kimi.supports_vision);
        // The native, unprefixed id is unaffected.
        assert_eq!(capabilities("kimi-k2.5").max_output, 262_144);
    }

    #[test]
    fn openrouter_qwen3_next_80b_a3b_thinking_gets_its_real_smaller_max_output() {
        // pi-parity (models/dialects pass): this exact vendor-slug string was originally believed
        // HuggingFace-only, real 262144/131072 (`huggingface.models.ts`) — 2x the generic Qwen
        // fallback's 65536, hence the fix at the time.
        //
        // pi-parity pass 20 Task 1: OpenRouter's own catalogue (`openrouter.models.ts`) turns out to
        // serve this identical full id string too, with a real max_output of just 32768 — smaller than
        // both HuggingFace's 131072 and the original generic fallback's 65536 (a ~4x over-report for
        // OpenRouter specifically). Same-string, no-host-signal collision this table can't disambiguate
        // any further; OpenRouter's smaller number now wins (this table's established safe-direction
        // tie-break), so HuggingFace itself now under-reports instead.
        let c = capabilities("qwen/qwen3-next-80b-a3b-thinking");
        assert_eq!(c.context_window, 262_144);
        assert_eq!(c.max_output, 32_768, "OpenRouter's real, smaller number now wins the collision");
    }

    #[test]
    fn huggingface_minimax_m2_and_m2_7_get_their_real_max_output() {
        // pi-parity (models/dialects pass): HuggingFace-only "MiniMaxAI/MiniMax-M2", real 204800/128000
        // (`huggingface.models.ts`) — context already matched the generic bucket's default; only
        // max_output was over-reported (~2.4%).
        let m2 = capabilities("MiniMaxAI/MiniMax-M2");
        assert_eq!(m2.context_window, 204_800);
        assert_eq!(m2.max_output, 128_000, "was 131_072 before this fix");

        // Together and HuggingFace nearly agree on "MiniMaxAI/MiniMax-M2.7" (202752 vs 204800 context,
        // 131072 max_output on both) — Together's smaller, safer context wins.
        let m2_7 = capabilities("MiniMaxAI/MiniMax-M2.7");
        assert_eq!(m2_7.context_window, 202_752, "was 204_800 before this fix");
        assert_eq!(m2_7.max_output, 131_072);
    }

    /// Genuine same-string collision (pi-parity, models/dialects pass): OpenRouter's and HuggingFace's
    /// own vendor-slug "qwen/qwen3-235b-a22b-thinking-2507" is the identical id string with a real
    /// max_output that differs 32x between the two hosts (4096 vs 131072) — exactly the case
    /// `capabilities_for_route_with_host`/`AggregatorHost` exist to resolve.
    #[test]
    fn capabilities_for_route_with_host_resolves_the_qwen3_235b_thinking_collision() {
        let id = "qwen/qwen3-235b-a22b-thinking-2507";
        // No host signal: falls back to the host-agnostic default, which matches OpenRouter's smaller,
        // safer number.
        let no_host = capabilities_for_route_with_host(id, false, false, false, None);
        assert_eq!(no_host.max_output, 4_096);
        assert_eq!(no_host, capabilities(id), "None must be a complete no-op");

        let openrouter = capabilities_for_route_with_host(
            id,
            false,
            false,
            false,
            Some(AggregatorHost::OpenRouter),
        );
        assert_eq!(openrouter.max_output, 4_096, "OpenRouter's real, smaller number");

        let huggingface = capabilities_for_route_with_host(
            id,
            false,
            false,
            false,
            Some(AggregatorHost::HuggingFace),
        );
        assert_eq!(huggingface.max_output, 131_072, "HuggingFace's real, 32x larger number");
        assert_eq!(
            huggingface.context_window, 262_144,
            "context agrees between hosts, so it's untouched by the host override"
        );
    }

    /// "openai/gpt-oss-20b" is (at least) a 4-way same-string collision — NVIDIA (131072/32768,
    /// `nvidia_caps`, checked first and already winning unconditionally — the same documented,
    /// deliberate precedent `nvidia_llama_ids_get_real_much_smaller_numbers_not_the_generic_llama_
    /// bucket`'s own `openai/gpt-oss-120b` assertion locks in ("must not inherit Groq's identically-
    /// named id's 65_536"), since un-teaching `nvidia_caps` its correct NVIDIA-native number isn't
    /// reachable without breaking that case),
    /// Together (131072/131072, `together.models.ts`), Groq (131072/65536, `groq.models.ts`), and
    /// OpenRouter (131072/4096, `openrouter.models.ts`). Without a host signal, every route silently
    /// gets NVIDIA's number (today's status quo, unchanged) — but `capabilities_for_route_with_host`'s
    /// override runs *after* that base lookup, so it corrects the Together case regardless of which
    /// number the host-agnostic base happened to return, a strictly better outcome than the base table
    /// alone can offer (it has no route context to disambiguate by at all).
    #[test]
    fn capabilities_for_route_with_host_resolves_the_gpt_oss_20b_collision_for_together() {
        let id = "openai/gpt-oss-20b";
        let no_host = capabilities_for_route_with_host(id, false, false, false, None);
        assert_eq!(no_host.max_output, 32_768, "NVIDIA's number wins unconditionally without a host signal");
        assert_eq!(no_host, capabilities(id), "None must be a complete no-op");

        let together =
            capabilities_for_route_with_host(id, false, false, false, Some(AggregatorHost::Together));
        assert_eq!(
            together.max_output, 131_072,
            "Together's real number, corrected despite NVIDIA's base-table interception"
        );
    }

    /// `capabilities_for_route_with_host` composes with the existing Codex/Azure/Copilot overrides
    /// rather than replacing them — an id neither route touches is unaffected regardless of host.
    #[test]
    fn capabilities_for_route_with_host_composes_with_existing_route_overrides() {
        let codex = capabilities_for_route_with_host(
            "gpt-5.3-codex-spark",
            true,
            false,
            false,
            Some(AggregatorHost::Fireworks),
        );
        assert_eq!(codex.max_output, 128_000, "Codex's own override must still apply");

        let unaffected = capabilities_for_route_with_host(
            "deepseek-v4-pro",
            false,
            false,
            false,
            Some(AggregatorHost::Together),
        );
        assert_eq!(unaffected, capabilities("deepseek-v4-pro"));
    }

    #[test]
    fn capabilities_for_route_with_host_resolves_the_opencode_zen_and_go_bare_id_collisions() {
        // pi-parity pass 20 Task 5: `kimi-k2.5`/`kimi-k2.6` are 4x over-reported (262144 vs. the real
        // 65536) on *both* OpenCode Zen and OpenCode-Go; `glm-5.1` is 4x over-reported (131072 vs. real
        // 32768) on OpenCode-Go specifically (OpenCode Zen's own real number, 131072, already matches
        // the host-agnostic default); `minimax-m3`'s *context* is 2x over-reported (1,000,000 vs. real
        // 512,000) on OpenCode Zen specifically (OpenCode-Go's own real context, 1,000,000, already
        // matches the host-agnostic default).
        let no_host_kimi = capabilities_for_route_with_host("kimi-k2.5", false, false, false, None);
        assert_eq!(no_host_kimi.max_output, 262_144, "no host signal: host-agnostic default unaffected");
        assert_eq!(no_host_kimi, capabilities("kimi-k2.5"), "None must be a complete no-op");

        for host in [AggregatorHost::OpenCodeZen, AggregatorHost::OpenCodeGo] {
            let kimi_2_6 = capabilities_for_route_with_host("kimi-k2.6", false, false, false, Some(host));
            assert_eq!(kimi_2_6.max_output, 65_536, "{host:?}: was 262_144, a 4x over-report");
        }
        let kimi_2_5_zen =
            capabilities_for_route_with_host("kimi-k2.5", false, false, false, Some(AggregatorHost::OpenCodeZen));
        assert_eq!(kimi_2_5_zen.max_output, 65_536, "OpenCode Zen: was 262_144, a 4x over-report");

        let glm_go =
            capabilities_for_route_with_host("glm-5.1", false, false, false, Some(AggregatorHost::OpenCodeGo));
        assert_eq!(glm_go.max_output, 32_768, "OpenCode-Go: was 131_072, a 4x over-report");
        let glm_zen =
            capabilities_for_route_with_host("glm-5.1", false, false, false, Some(AggregatorHost::OpenCodeZen));
        assert_eq!(
            glm_zen.max_output, 131_072,
            "OpenCode Zen's own real number already matches the host-agnostic default"
        );

        let minimax_zen = capabilities_for_route_with_host(
            "minimax-m3",
            false,
            false,
            false,
            Some(AggregatorHost::OpenCodeZen),
        );
        assert_eq!(minimax_zen.context_window, 512_000, "OpenCode Zen: was 1_000_000, a 2x over-report");
        let minimax_go = capabilities_for_route_with_host(
            "minimax-m3",
            false,
            false,
            false,
            Some(AggregatorHost::OpenCodeGo),
        );
        assert_eq!(
            minimax_go.context_window, 1_000_000,
            "OpenCode-Go's own real context already matches the host-agnostic default"
        );
    }

    #[test]
    fn huggingface_glm_4_5v_gets_vision_and_its_own_much_smaller_numbers() {
        let c = capabilities("zai-org/GLM-4.5V");
        assert!(c.supports_vision, "\"glm-4.5v\".starts_with(\"glm-5v\") is false; needs its own check");
        assert_eq!(c.context_window, 65_536);
        assert_eq!(c.max_output, 16_384);
        // Every other current GLM id is unaffected.
        assert!(!capabilities("glm-4.7").supports_vision);
        assert!(capabilities("glm-5v-turbo").supports_vision);
    }

    #[test]
    fn huggingface_bare_glm_4_5_gets_the_same_real_numbers_as_glm_4_5_air() {
        // pi-parity Task #23: bare "zai-org/GLM-4.5" (no "-air"/"-v" suffix — `huggingface.models.ts:
        // 709`) used to fall to the 200000/131072 else-bucket; its real numbers (131072/98304) are
        // identical to the already-correct "glm-4.5-air" special case.
        let c = capabilities("zai-org/GLM-4.5");
        assert_eq!(c.context_window, 131_072);
        assert_eq!(c.max_output, 98_304);
        // The "-air" and "-v" variants are unaffected — still their own distinct cases.
        assert_eq!(capabilities("glm-4.5-air").max_output, 98_304);
        assert!(capabilities("zai-org/GLM-4.5V").supports_vision);
    }

    #[test]
    fn huggingface_kimi_k2_instruct_bare_naming_gets_its_real_numbers() {
        let plain = capabilities("moonshotai/Kimi-K2-Instruct");
        assert_eq!(plain.context_window, 131_072, "was 262_144 under the generic else bucket");
        assert_eq!(plain.max_output, 16_384, "was 262_144 under the generic else bucket");
        assert!(!plain.supports_vision);
        assert!(!plain.reasoning_disableable, "non-reasoning: nothing to disable");

        let dated = capabilities("moonshotai/Kimi-K2-Instruct-0905");
        assert_eq!(dated.context_window, 262_144);
        assert_eq!(dated.max_output, 16_384, "was 262_144 under the generic else bucket");

        // The unrelated "-0905" *preview* id (a different, non-Instruct release) keeps its own
        // existing, separately-tested treatment.
        let preview = capabilities("kimi-k2-0905");
        assert_eq!(preview.openai_reasoning_format, OpenAiReasoningFormat::Standard);
    }

    #[test]
    fn huggingface_llama_3_3_70b_instruct_gets_its_real_much_smaller_max_output() {
        let c = capabilities("meta-llama/Llama-3.3-70B-Instruct");
        assert_eq!(c.context_window, 131_072);
        assert_eq!(c.max_output, 4_096, "was 32_768 under the generic llama branch's default");
        // Every other llama-shaped id is unaffected.
        assert_eq!(capabilities("llama-3.1-70b").max_output, 32_768);
    }

    #[test]
    fn together_llama_3_3_70b_instruct_turbo_gets_its_real_larger_max_output() {
        // pi-parity Task #26: the `-Turbo` suffixed Together id (`together.models.ts:212`) is a
        // *different* string entirely from the bare-native/OpenRouter-shaped exact match just above —
        // no collision — real numbers 131072/131072, vs the generic llama branch's smaller 32_768
        // default (a safe-direction under-report, fixed here for accuracy).
        let c = capabilities("meta-llama/Llama-3.3-70B-Instruct-Turbo");
        assert_eq!(c.context_window, 131_072);
        assert_eq!(c.max_output, 131_072, "was 32_768 under the generic llama branch's default");
        // The non-Turbo HuggingFace id is unaffected — still its own, much smaller real number.
        assert_eq!(capabilities("meta-llama/Llama-3.3-70B-Instruct").max_output, 4_096);
    }

    #[test]
    fn groq_llama_4_scout_and_llama_3_1_8b_instant_get_their_real_numbers() {
        // pi-parity Tasks #24/#25: both are Groq-exclusive spellings, no collision with any other
        // host's id.
        let scout = capabilities("meta-llama/llama-4-scout-17b-16e-instruct");
        assert_eq!(scout.context_window, 131_072);
        assert_eq!(
            scout.max_output, 8_192,
            "was 32_768 under the generic llama branch's default — a dangerous 4x over-report"
        );
        assert!(scout.supports_vision, "llama-4 ids are vision-capable");

        let instant = capabilities("llama-3.1-8b-instant");
        assert_eq!(instant.context_window, 131_072);
        assert_eq!(
            instant.max_output, 131_072,
            "was 32_768 under the generic llama branch's default — a safe-direction under-report, \
             fixed for accuracy"
        );
        assert!(!instant.supports_vision);
    }

    #[test]
    fn huggingface_mimo_v2_flash_vendor_slug_gets_its_real_smaller_max_output() {
        let hf = capabilities("XiaomiMiMo/MiMo-V2-Flash");
        assert_eq!(hf.context_window, 262_144);
        assert_eq!(hf.max_output, 4_096, "was 65_536 under the bare-id default");

        // The bare (native) id is completely unaffected — this regression already exists and must
        // keep passing unchanged.
        let native = capabilities("mimo-v2-flash");
        assert_eq!(native.max_output, 65_536);
    }

    #[test]
    fn huggingface_gemma_4_it_ids_get_real_vision_and_numbers_not_the_generic_openrouter_fallback() {
        for id in ["google/gemma-4-26b-a4b-it", "google/gemma-4-31b-it"] {
            let c = capabilities(id);
            assert!(c.supports_vision, "{id}");
            assert_eq!(c.context_window, 262_144, "{id}");
            assert_ne!(
                c.openai_reasoning_format,
                OpenAiReasoningFormat::OpenRouter,
                "{id}: must not land on the generic vendor-slug fallback"
            );
        }
        // pi-parity pass 20 Task 1: the two ids no longer share a single max_output. OpenRouter's own
        // real number for "-26b-a4b-it" (4096, `openrouter.models.ts`) is far smaller than
        // HuggingFace's shared 32768 default — an 8x over-report fixed by narrowing to the smaller
        // number. "-31b-it" is unaffected: OpenRouter's real number there (262144) is *larger* than
        // HuggingFace's 32768, so HuggingFace's smaller number remains the safe pick (see the
        // `..._togethers_larger_real_one` test below for that sibling id's own, separate collision).
        assert_eq!(capabilities("google/gemma-4-26b-a4b-it").max_output, 4_096);
        assert_eq!(capabilities("google/gemma-4-31b-it").max_output, 32_768);
    }

    #[test]
    fn google_gemma_4_31b_it_keeps_huggingfaces_smaller_safe_max_output_despite_togethers_larger_real_one() {
        // pi-parity Task #27 (investigated, kept as-is — documented, not a new numeric fix): Together
        // also hosts this exact vendor-slug id with a real max_output of 131072
        // (`together.models.ts:193`), 4x HuggingFace's 32768. No route/host signal exists to
        // disambiguate the two for this same-string collision, so the smaller (safe-direction) number
        // stays — this pins down the current, deliberate behavior so a future route-aware fix changes
        // it on purpose, not by accident.
        let c = capabilities("google/gemma-4-31b-it");
        assert_eq!(c.max_output, 32_768, "HuggingFace's smaller, safe-direction number");
        assert_ne!(c.max_output, 131_072, "not Together's real, larger ceiling");
    }

    // ---- Task 17: Together Qwen precision + Together/Fireworks MiniMax-M3 ----

    #[test]
    fn together_qwen_lineup_gets_id_for_id_numbers_not_the_stale_generic_bucket() {
        // Real numbers vary by up to ~12x from the generic 200k/40960 default in both directions.
        let turbo = capabilities("Qwen/Qwen2.5-7B-Instruct-Turbo");
        assert_eq!(turbo.context_window, 32_768);
        assert_eq!(turbo.max_output, 32_768);
        assert!(!turbo.supports_vision);
        assert_eq!(
            turbo.openai_reasoning_format,
            OpenAiReasoningFormat::Standard,
            "non-reasoning id must not claim the together toggle mechanism"
        );

        // pi-parity pass 20 Task 1: "Qwen/Qwen3.6-Plus"/"Qwen/Qwen3.7-Max" (Together's real numbers:
        // 1,000,000/500,000, matched here by this file's own `together_match` table) are *also*
        // OpenRouter's own identical vendor-slug spelling, with a real max_output of just 65536 — OpenRouter's
        // smaller number now wins this collision (see
        // `openrouter_qwen_ids_get_their_own_real_numbers_not_the_colliding_together_shape` below), so
        // this test no longer asserts Together's larger numbers for either id directly.

        let vision_9b = capabilities("Qwen/Qwen3.5-9B");
        assert!(vision_9b.supports_vision);
        assert_eq!(vision_9b.max_output, 65_536);

        // "Qwen/Qwen3.5-397B-A17B" used to be tested here as a Together/HuggingFace-only collision
        // (pi-parity Task #17: HuggingFace's smaller 32768 winning over Together's 130000) — OpenRouter
        // turns out to serve this identical string too, with an even smaller real max_output (4096); see
        // the dedicated test below for the fully updated 3-way resolution.

        // An uncatalogued Together-shaped Qwen id still falls back to the generic bucket, refreshed
        // (pi-parity Task #28) to match pi's current catalogue's ~262144/65536 default rather than the
        // older, stale 200000/40960.
        let uncatalogued = capabilities("Qwen/Qwen4-Hypothetical-Future-Id");
        assert_eq!(uncatalogued.context_window, 262_144);
        assert_eq!(uncatalogued.max_output, 65_536);
        assert_eq!(uncatalogued.openai_reasoning_format, OpenAiReasoningFormat::Together);
    }

    #[test]
    fn openrouter_qwen_ids_get_their_own_real_numbers_not_the_colliding_together_shape() {
        // pi-parity pass 20 Task 1: every id here is OpenRouter's own vendor-slug spelling
        // (`openrouter.models.ts`), each colliding on the identical full id string with a Together (and,
        // for two of them, HuggingFace) entry this file's `together_match`/HuggingFace-tuned buckets
        // already cover under a different, larger real number. OpenRouter's own smaller, safer number
        // now wins each collision.
        let plus = capabilities("qwen/qwen3.6-plus");
        assert_eq!(plus.context_window, 1_000_000);
        assert_eq!(plus.max_output, 65_536, "was 500_000, Together's number, a ~7.6x over-report");
        assert_eq!(plus.openai_reasoning_format, OpenAiReasoningFormat::OpenRouter);
        assert!(
            !plus.supports_vision,
            "Together's real entry for this exact string is text-only — the safe pick"
        );

        let max = capabilities("qwen/qwen3.7-max");
        assert_eq!(max.context_window, 1_000_000);
        assert_eq!(max.max_output, 65_536, "was 500_000, a ~7.6x over-report");
        assert!(!max.supports_vision);

        // Was a documented Together/HuggingFace-only collision (pi-parity Task #17); OpenRouter serves
        // this identical string too, with an even smaller real max_output.
        let vision_397b = capabilities("qwen/qwen3.5-397b-a17b");
        assert_eq!(vision_397b.context_window, 256_000);
        assert_eq!(vision_397b.max_output, 4_096, "was 32_768 (HuggingFace's number), an 8x over-report");
        assert!(vision_397b.supports_vision);

        let instruct_80b = capabilities("qwen/qwen3-next-80b-a3b-instruct");
        assert_eq!(instruct_80b.context_window, 262_144);
        assert_eq!(instruct_80b.max_output, 16_384, "was 65_536 under the generic bucket, a ~4x over-report");
        assert_eq!(instruct_80b.openai_reasoning_format, OpenAiReasoningFormat::Standard);

        let thinking_80b = capabilities("qwen/qwen3-next-80b-a3b-thinking");
        assert_eq!(thinking_80b.context_window, 262_144);
        assert_eq!(
            thinking_80b.max_output, 32_768,
            "was 131_072 (HuggingFace's number), a ~4x over-report"
        );

        let coder_30b = capabilities("qwen/qwen3-coder-30b-a3b-instruct");
        assert_eq!(coder_30b.context_window, 160_000, "was 262_144, ~1.6x over-report");
        assert_eq!(coder_30b.max_output, 32_768, "was 65_536, ~2x over-report");
    }

    #[test]
    fn openrouter_qwen3_235b_a22b_gets_openrouters_smaller_max_output_alongside_huggingfaces_context() {
        // pi-parity pass 20 Task 1: this exact vendor-slug string is a HuggingFace/OpenRouter collision
        // (Task #16 originally special-cased it for HuggingFace's smaller *context* alone) — OpenRouter's
        // own real max_output (8192) is in turn smaller than HuggingFace's 16384 this bucket used to
        // return unconditionally, a ~2x over-report for OpenRouter specifically.
        let c = capabilities("qwen/qwen3-235b-a22b");
        assert_eq!(c.context_window, 40_960, "HuggingFace's smaller context, unaffected by this fix");
        assert_eq!(c.max_output, 8_192, "was 16_384, OpenRouter's real number is smaller still");
    }

    #[test]
    fn openrouter_z_ai_glm_5_2_gets_its_real_smaller_max_output_via_nvidia_caps() {
        // pi-parity pass 20 Task 1: "z-ai/glm-5.2" (OpenRouter's own vendor-slug spelling — distinct
        // from Together's/HuggingFace's "zai-org/" and Vercel's "zai/") is *also* a real NVIDIA-native
        // id (`nvidia.models.ts`), which `nvidia_caps` intercepts first, unconditionally — so this
        // collision is resolved there, not in the GLM family branch. NVIDIA's own real max_output
        // (131_072) was a small (~2.4%) over-report of OpenRouter's real 128_000 for this identical
        // string; context (1_000_000, NVIDIA's own, smaller than OpenRouter's 1,048,576) is unaffected.
        let c = capabilities("z-ai/glm-5.2");
        assert_eq!(c.context_window, 1_000_000);
        assert_eq!(c.max_output, 128_000, "was 131_072, a small ~2.4% over-report for OpenRouter");
        // nvidia_caps's own shape: no client-steerable reasoning mechanism at all (unlike the GLM
        // branch's real effort vocabulary for bare "glm-5.2").
        assert!(!c.reasoning_effort);
    }

    #[test]
    fn openrouter_gpt_4_turbo_preview_is_text_only_unlike_its_gpt_4_turbo_sibling() {
        // pi-parity pass 20 Task 3: OpenRouter's own vendor-slug "openai/gpt-4-turbo-preview" is a
        // genuinely text-only legacy alias (`openrouter.models.ts`: `input: ["text"]`) — the shared
        // gpt-4-turbo* bucket's `m != "gpt-4"` vision gate incorrectly claimed vision support for it,
        // inherited from its differently-named, genuinely vision-capable sibling "openai/gpt-4-turbo".
        let preview = capabilities("openai/gpt-4-turbo-preview");
        assert!(!preview.supports_vision, "was true — this legacy alias has no vision support at all");
        assert_eq!(preview.context_window, 128_000);
        assert_eq!(preview.max_output, 4_096);

        // The differently-named sibling id keeps its real vision support, matching pi's `openai/
        // gpt-4-turbo` entry (`input: ["text","image"]`) — this fix must not regress it.
        let turbo = capabilities("openai/gpt-4-turbo");
        assert!(turbo.supports_vision, "openai/gpt-4-turbo itself is genuinely vision-capable");
    }

    #[test]
    fn fireworks_minimax_m3_gets_its_own_real_numbers() {
        let c = capabilities("accounts/fireworks/models/minimax-m3");
        assert_eq!(c.context_window, 512_000);
        assert_eq!(c.max_output, 512_000);
        assert!(!c.supports_vision);
    }

    // ---- Task 18: Fireworks "p"-encoded id normalization ----

    #[test]
    fn fireworks_p_separator_normalizes_only_digit_p_digit_sequences() {
        assert_eq!(normalize_fireworks_p_separator("glm-5p1"), "glm-5.1");
        assert_eq!(normalize_fireworks_p_separator("kimi-k2p6"), "kimi-k2.6");
        assert_eq!(
            normalize_fireworks_p_separator("accounts/fireworks/models/minimax-m2p7"),
            "accounts/fireworks/models/minimax-m2.7"
        );
        // No digit-p-digit sequence anywhere: untouched, including the account/router path segments.
        assert_eq!(
            normalize_fireworks_p_separator("accounts/fireworks/models/gpt-oss-120b"),
            "accounts/fireworks/models/gpt-oss-120b"
        );
        assert_eq!(
            normalize_fireworks_p_separator("accounts/fireworks/models/minimax-m3"),
            "accounts/fireworks/models/minimax-m3"
        );
    }

    #[test]
    fn fireworks_glm_5p2_gets_the_real_glm_5_2_context_not_the_stale_200k_else_bucket() {
        // Before normalization, `family_id` was "glm-5p2" — `g.starts_with("glm-5.2")` never matched
        // (dot vs "p"), so this silently fell to the GLM branch's flat 200k/131k "else" default instead
        // of the real glm-5.2 bucket (~1M context).
        let c = capabilities("accounts/fireworks/models/glm-5p2");
        assert_eq!(c.context_window, 1_000_000, "must resolve to the glm-5.2 bucket, not the else one");
        assert!(c.reasoning_effort, "glm-5.2 is the one GLM generation with a real effort vocabulary");
    }

    #[test]
    fn fireworks_kimi_k2p6_gets_correct_vision_not_silently_downgraded_to_text_only() {
        // Before normalization, `k.starts_with("kimi-k2.6")` never matched "kimi-k2p6" (dot vs "p"),
        // silently reporting a real vision-capable id as text-only.
        let c = capabilities("accounts/fireworks/models/kimi-k2p6");
        assert!(c.supports_vision, "kimi-k2.6 is vision-capable on every host, Fireworks included");
        assert_eq!(c.openai_reasoning_format, OpenAiReasoningFormat::DeepSeek);

        // The "-fast"/"-turbo" router variants share the same normalized prefix.
        assert!(capabilities("accounts/fireworks/routers/kimi-k2p6-fast").supports_vision);
        assert!(capabilities("accounts/fireworks/routers/kimi-k2p6-turbo").supports_vision);
    }

    #[test]
    fn fireworks_normalization_does_not_affect_non_fireworks_ids() {
        // A non-Fireworks id is never even passed to the normalizer (`capabilities` gates on
        // `is_fireworks_model` first) — confirm the gate itself doesn't fire for an id that merely
        // contains a literal "p" between two digits for some unrelated reason, without the
        // distinctive Fireworks path prefix.
        assert!(!is_fireworks_model("some-vendor/model-5p1"));
        assert!(is_fireworks_model("accounts/fireworks/models/glm-5p1"));
    }

    #[test]
    fn fireworks_anthropic_wire_ids_never_support_cache_control_on_tools() {
        // pi-parity Task #2: pi's `fireworks.models.ts` sets `supportsCacheControlOnTools: false` on
        // all 14 of its Anthropic-wire ids — a later routing change
        // (`dialect::is_fireworks_anthropic_wire_model`) now sends exactly these ids through the
        // Anthropic dialect, which used to stamp `cache_control` on the last tool unconditionally.
        for id in [
            "accounts/fireworks/models/deepseek-v4-pro",
            "accounts/fireworks/models/glm-5p1",
            "accounts/fireworks/models/gpt-oss-120b",
            "accounts/fireworks/models/kimi-k2p6",
            "accounts/fireworks/models/minimax-m3",
            "accounts/fireworks/models/qwen3p7-plus",
            "accounts/fireworks/routers/kimi-k2p6-fast",
        ] {
            assert!(
                !capabilities(id).supports_cache_control_on_tools,
                "{id} must not support cache_control on tools"
            );
        }
        // The two genuinely `openai-completions` Fireworks ids (never reached through the Anthropic
        // dialect at all) are unaffected — `true`, matching every other model's default.
        assert!(capabilities("accounts/fireworks/models/glm-5p2").supports_cache_control_on_tools);
        // Every non-Fireworks id, including plain Claude, defaults to `true` too.
        assert!(capabilities("claude-opus-4-8").supports_cache_control_on_tools);
        assert!(capabilities("deepseek-v4-pro").supports_cache_control_on_tools);
        assert!(ModelCaps::unknown().supports_cache_control_on_tools);
    }

    // ---- Task 16: NVIDIA id-for-id table ----

    #[test]
    fn nvidia_llama_ids_get_real_much_smaller_numbers_not_the_generic_llama_bucket() {
        // The generic `llama`/`/llama` branch's 131_072/32_768 defaults badly over-report every one of
        // these (dangerous direction: a `max_tokens` NVIDIA's real endpoint would reject).
        let c = capabilities("meta/llama-3.1-8b-instruct");
        assert_eq!(c.context_window, 16_000);
        assert_eq!(c.max_output, 4_096);

        let c70 = capabilities("meta/llama-3.1-70b-instruct");
        assert_eq!(c70.context_window, 128_000);
        assert_eq!(c70.max_output, 4_096);

        let c_oss120 = capabilities("openai/gpt-oss-120b");
        assert_eq!(c_oss120.context_window, 128_000);
        assert_eq!(
            c_oss120.max_output, 8_192,
            "must not inherit Groq's identically-named id's 65_536"
        );
    }

    #[test]
    fn nvidia_vision_models_are_correctly_flagged_unlike_the_generic_llama_prefix_check() {
        // The generic llama branch's `m.contains("llama-4")` vision check misses these two llama-3.2
        // vision ids entirely — NVIDIA's own catalogue does mark them vision-capable.
        assert!(capabilities("meta/llama-3.2-11b-vision-instruct").supports_vision);
        assert!(capabilities("meta/llama-3.2-90b-vision-instruct").supports_vision);
        assert_eq!(capabilities("meta/llama-3.2-90b-vision-instruct").max_output, 8_192);
    }

    #[test]
    fn nvidia_mistralai_ids_do_not_fall_into_the_mistral_branchs_catch_all() {
        // "mistralai/..." starts with the literal string "mistral" (its own org slug), so it would
        // otherwise silently match `is_mistral_id` and land on that branch's smaller `_` catch-all
        // default (128_000/128_000, no vision) instead of NVIDIA's real, larger numbers.
        let large = capabilities("mistralai/mistral-large-3-675b-instruct-2512");
        assert_eq!(large.context_window, 262_144);
        assert_eq!(large.max_output, 262_144);
        assert!(large.supports_vision);

        let small = capabilities("mistralai/mistral-small-4-119b-2603");
        assert_eq!(small.context_window, 128_000);
        assert_eq!(small.max_output, 8_192, "must not inherit the catch-all's 128_000");
        assert!(small.supports_vision);
    }

    #[test]
    fn nvidia_collision_ids_intentionally_left_to_the_other_hosts_established_coverage() {
        // "moonshotai/kimi-k2.6" is also a real NVIDIA id, but this table deliberately doesn't list
        // NVIDIA's own number for it (see `nvidia_caps`'s own doc comment) since it's id-for-id
        // identical to an id Together/HuggingFace already serve under the same literal string, with
        // existing, tested coverage via the Kimi `family_id` branch. Regression guard that adding
        // NVIDIA's table didn't silently steal this id away from that established behavior.
        //
        // "minimaxai/minimax-m3" used to be a sibling example here — pi-parity pass 20 Task 2 moved it
        // *into* `nvidia_caps` instead (a real 7.8x over-report, not a shrug-worthy gap); see
        // `nvidia_minimax_m3_gets_its_real_smaller_max_output` below for that fix's own regression
        // test.
        let kimi = capabilities("moonshotai/kimi-k2.6");
        assert_eq!(kimi.openai_reasoning_format, OpenAiReasoningFormat::DeepSeek);
    }

    #[test]
    fn nvidia_nemotron_ids_get_their_real_numbers() {
        let ultra = capabilities("nvidia/nemotron-3-ultra-550b-a55b");
        assert_eq!(ultra.context_window, 1_000_000);
        assert_eq!(ultra.max_output, 65_536);
        assert!(!ultra.supports_vision);

        let omni = capabilities("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning");
        assert_eq!(omni.context_window, 256_000);
        assert_eq!(omni.max_output, 65_536);
        assert!(omni.supports_vision);
    }

    #[test]
    fn nvidia_minimax_m3_gets_its_real_smaller_max_output() {
        // pi-parity pass 20 Task 2: NVIDIA's real max_output for "minimaxai/minimax-m3" is 16_384 — the
        // table used to omit it entirely (deferring to the MiniMax `family_id` branch's own 128_000,
        // Together's/HuggingFace's shared number), a 7.8x over-report for a genuine NVIDIA-routed
        // request. Now listed in `nvidia_caps` directly, using NVIDIA's own (and smallest-of-three)
        // real number.
        let c = capabilities("minimaxai/minimax-m3");
        assert_eq!(c.context_window, 1_000_000);
        assert_eq!(c.max_output, 16_384);
        assert!(c.supports_vision);
    }

    #[test]
    fn nvidia_nemotron_3_super_120b_no_longer_over_reports_openrouters_max_output() {
        // pi-parity pass 20 Task 1: this table used to return NVIDIA's own (262_144, 262_144) for this
        // id unconditionally — a 64x over-report of OpenRouter's real max_output (4096) for the
        // identical vendor-slug string, unlike the sibling `nemotron-3-ultra-550b-a55b`/`gpt-oss-120b`
        // ids where NVIDIA's own number already was the smallest/safest across hosts (see
        // `nvidia_caps`'s own doc comment for why that reasoning doesn't extend to this id).
        let c = capabilities("nvidia/nemotron-3-super-120b-a12b");
        assert_eq!(c.context_window, 262_144);
        assert_eq!(c.max_output, 4_096);
    }

    #[test]
    fn nvidia_nemotron_3_ultra_keeps_nvidias_native_numbers_pending_route_awareness() {
        // pi-parity Task #21, investigated and deliberately left as-is (see `nvidia_caps`'s own doc
        // comment): Together re-hosts this exact id with real numbers (512300/512300,
        // `together.models.ts:268`) that `nvidia_caps` — checked first, unconditionally, on the id
        // string alone — can't ever see, since it always wins for this literal string. The direction is
        // safe (NVIDIA's 65_536 max_output is smaller than Together's real 512_300), just a usability
        // loss for Together specifically. This pins down the current, documented behavior so a future
        // route-aware fix changes it on purpose.
        let c = capabilities("nvidia/nemotron-3-ultra-550b-a55b");
        assert_eq!(c.context_window, 1_000_000, "NVIDIA's own real, tested number");
        assert_eq!(c.max_output, 65_536, "NVIDIA's own real, tested number");
    }

    #[test]
    fn nvidia_ids_have_no_steerable_reasoning_mechanism_at_all() {
        // pi's `supportsReasoningEffort: false` on every current NVIDIA id, regardless of that id's own
        // `reasoning: true/false` flag — matches xAI/Grok's identical "reasons internally, no toggle"
        // shape elsewhere in this table.
        for id in [
            "nvidia/nemotron-3-nano-30b-a3b", // reasoning: true in pi's catalogue
            "meta/llama-3.1-70b-instruct",    // reasoning: false in pi's catalogue
        ] {
            let c = capabilities(id);
            assert!(!c.reasoning_effort, "{id}");
            assert!(!has_reasoning_mechanism(&c), "{id}");
        }
    }

    // ---- Task 15: Ant-Ling capability branch ----

    #[test]
    fn ant_ling_gets_real_capabilities_not_the_unknown_default() {
        for id in ["Ling-2.6-1T", "Ling-2.6-flash", "Ring-2.6-1T"] {
            let c = capabilities(id);
            assert_eq!(c.context_window, 262_144, "{id}");
            assert_eq!(c.max_output, 65_536, "{id}");
            assert!(!c.supports_vision, "{id}");
            assert!(!c.supports_long_cache, "{id}: pi's isAntLing denylist excludes long-cache");
            assert!(c.max_output > ModelCaps::unknown().max_output, "{id}");
        }
    }

    #[test]
    fn ant_ling_ring_2_6_1t_is_the_only_reasoning_capable_id_with_a_high_floor() {
        let ring = capabilities("Ring-2.6-1T");
        assert_eq!(ring.min_reasoning_effort, crate::transport::ReasoningEffort::High);
        assert!(ring.supports_xhigh_reasoning);
        assert!(!ring.reasoning_disableable, "thinkingLevelMap.off is null — no explicit off signal");
        assert_eq!(ring.openai_reasoning_format, OpenAiReasoningFormat::AntLing);
        // No graduated top-level `reasoning_effort` string exists for this format — the mechanism lives
        // entirely in the (currently-unwired) nested `reasoning.effort` shape.
        assert!(!ring.reasoning_effort);
        assert!(has_reasoning_mechanism(&ring), "the toggle-only third arm must still report a mechanism");

        let ling = capabilities("Ling-2.6-1T");
        assert_eq!(ling.openai_reasoning_format, OpenAiReasoningFormat::Standard);
        assert!(!has_reasoning_mechanism(&ling));
    }

    #[test]
    fn ant_ling_ids_count_as_a_non_standard_store_provider() {
        assert!(is_non_standard_store_provider("Ring-2.6-1T"));
        assert!(is_non_standard_store_provider("ling-2.6-flash"));
    }

    #[test]
    fn copilot_kimi_k2_7_code_keeps_the_native_moonshot_numbers_pending_route_awareness() {
        // Documented (not fixed) collision: Copilot's own "kimi-k2.7-code" has much smaller real
        // numbers (256_000/32_000) than moonshotai's identically-spelled native id (262_144/262_144),
        // and there's no id-only signal to tell them apart (see the branch's own doc comment). This
        // pins down the *documented* current behavior (native numbers win) so a future route-aware fix
        // changes this test deliberately, not by accident.
        let c = capabilities("kimi-k2.7-code");
        assert_eq!(c.context_window, 262_144);
        assert_eq!(c.max_output, 262_144);
    }

    #[test]
    fn mimo_gets_its_real_capabilities_not_the_unknown_default() {
        // The CRITICAL bug this closes: no "mimo"/"xiaomi" branch existed at all, so every bare
        // "mimo-*" id (`xiaomi.models.ts`'s ids aren't vendor-slug prefixed) fell all the way through
        // to `ModelCaps::unknown()` — a 4096-token output ceiling regardless of the model's real
        // 65536-384000 one, and no reasoning support at all.
        let flash = capabilities("mimo-v2-flash");
        assert_eq!(flash.context_window, 262_144);
        assert_eq!(flash.max_output, 65_536);
        assert!(!flash.supports_vision);

        let omni = capabilities("mimo-v2-omni");
        assert_eq!(omni.max_output, 131_072);
        assert!(omni.supports_vision);

        let v25 = capabilities("mimo-v2.5");
        assert_eq!(v25.context_window, 1_048_576);
        assert!(v25.supports_vision, "bare mimo-v2.5 is vision-capable");

        // "mimo-v2.5-pro" shares its prefix with "mimo-v2.5" but is NOT vision-capable — regression
        // guard that vision is matched by exact id, not by prefix.
        let v25_pro = capabilities("mimo-v2.5-pro");
        assert_eq!(v25_pro.context_window, 1_048_576);
        assert!(!v25_pro.supports_vision);

        for id in [
            "mimo-v2-flash",
            "mimo-v2-omni",
            "mimo-v2-pro",
            "mimo-v2.5",
            "mimo-v2.5-pro",
            "mimo-v2.5-pro-ultraspeed",
        ] {
            let c = capabilities(id);
            assert!(c.reasoning_effort, "{id}");
            assert!(c.reasoning_disableable, "{id}");
            assert_eq!(c.max_tokens_field, MaxTokensField::MaxCompletionTokens, "{id}");
            assert_eq!(c.openai_reasoning_format, OpenAiReasoningFormat::DeepSeek, "{id}");
            assert!(
                c.max_output > ModelCaps::unknown().max_output,
                "{id} should beat unknown()'s 4096 ceiling"
            );
        }
    }

    #[test]
    fn mimo_vendor_slug_id_also_matches_the_family() {
        // HuggingFace hosts this family vendor-slug-prefixed: "XiaomiMiMo/MiMo-V2.5-Pro".
        let c = capabilities("XiaomiMiMo/MiMo-V2.5-Pro");
        assert_eq!(c.context_window, 1_048_576);
        assert_eq!(c.max_output, 131_072);
        assert!(!c.supports_vision);
        assert_eq!(c.openai_reasoning_format, OpenAiReasoningFormat::DeepSeek);
    }

    #[test]
    fn supports_tool_stream_is_true_for_every_glm_id_except_glm_4_5_air() {
        for id in ["glm-4.7", "glm-5-turbo", "glm-5.1", "glm-5.2", "glm-5v-turbo"] {
            assert!(capabilities(id).supports_tool_stream, "{id} should set tool_stream");
        }
        assert!(
            !capabilities("glm-4.5-air").supports_tool_stream,
            "glm-4.5-air is the one id both zai.models.ts and zai-coding-cn.models.ts leave it off for"
        );
        // Vendor-slug GLM ids inherit the flag through the same family branch.
        assert!(capabilities("zai-org/GLM-5.2").supports_tool_stream);
        // No other family sets it.
        for id in [
            "deepseek-v4-pro",
            "kimi-k2-thinking",
            "gpt-4o",
            "claude-opus-4-8",
            "mimo-v2.5",
            "some-vendor/some-model",
        ] {
            assert!(!capabilities(id).supports_tool_stream, "{id}");
        }
    }

    #[test]
    fn budget_for_effort_with_override_prefers_the_override_table_but_still_clamps_to_max_output() {
        use crate::transport::ReasoningEffort as RE;
        let mut overrides = std::collections::HashMap::new();
        overrides.insert(RE::High, 40_000);
        // The overridden level uses the table's value...
        assert_eq!(
            budget_for_effort_with_override(RE::High, 100_000, Some(&overrides)),
            40_000
        );
        // ...a level missing from the table still falls back to the built-in ladder...
        assert_eq!(
            budget_for_effort_with_override(RE::Low, 100_000, Some(&overrides)),
            default_budget_for_effort(RE::Low)
        );
        // ...and the max_output clamp still applies even to an overridden value.
        assert_eq!(
            budget_for_effort_with_override(RE::High, 4_096, Some(&overrides)),
            4_095
        );
        // No table at all behaves exactly like `budget_for_effort`.
        assert_eq!(
            budget_for_effort_with_override(RE::Medium, 100_000, None),
            budget_for_effort(RE::Medium, 100_000)
        );
    }

    // ---- Mistral (pi-parity pass 15: this whole family used to fall through to `unknown()`) ----

    #[test]
    fn mistral_ids_get_real_capabilities_not_the_unknown_default() {
        // The CRITICAL bug this closes: every Mistral model id used to resolve to
        // `ModelCaps::unknown()` — 128k context, a 4096-token output ceiling regardless of the
        // model's real (often much larger) one, vision silently disabled on every vision-capable id,
        // and no reasoning/thinking wire support at all.
        let devstral = capabilities("devstral-2512");
        assert_eq!(devstral.context_window, 262_144);
        assert_eq!(devstral.max_output, 262_144);
        assert!(!devstral.reasoning_effort);
        assert!(!devstral.supports_vision);

        let large_2512 = capabilities("mistral-large-2512");
        assert_eq!(large_2512.context_window, 262_144);
        assert!(large_2512.supports_vision, "mistral-large-2512 gained vision");
        // The "-2411" predecessor is text-only and much smaller — a family-level default would have
        // misreported one of these by a wide margin.
        let large_2411 = capabilities("mistral-large-2411");
        assert_eq!(large_2411.context_window, 131_072);
        assert_eq!(large_2411.max_output, 16_384);
        assert!(!large_2411.supports_vision);

        assert!(capabilities("pixtral-12b").supports_vision);
        assert!(capabilities("pixtral-large-latest").supports_vision);
        assert!(!capabilities("open-mistral-7b").supports_vision);
        assert_eq!(capabilities("open-mistral-7b").context_window, 8_000);
    }

    #[test]
    fn mistral_reasoning_capable_ids_get_a_thinking_mechanism() {
        // pixtral/mistral-medium-*/mistral-small-2603+ used to silently have no thinking mechanism at
        // all (`reasoning_effort: false`, `openai_reasoning_format: Standard` with nothing to drive
        // it) — `--thinking high` was a silent no-op.
        for id in [
            "magistral-medium-latest",
            "magistral-small",
            "mistral-medium-2604",
            "mistral-medium-3.5",
            "mistral-small-2603",
            "mistral-small-latest",
        ] {
            let c = capabilities(id);
            assert!(c.reasoning_effort, "{id} should gain a thinking mechanism");
            assert_eq!(c.openai_reasoning_format, OpenAiReasoningFormat::Standard, "{id}");
            assert!(c.reasoning_disableable, "{id}: off is always legal (no id nulls it)");
            assert!(!c.supports_xhigh_reasoning, "{id}: no Mistral id defines xhigh");
        }
        // "mistral-medium-latest" is (per pi's live catalogue) an alias still pointing at a
        // non-reasoning snapshot, unlike its "-2604"/"-3.5" siblings.
        assert!(!capabilities("mistral-medium-latest").reasoning_effort);
        assert!(!capabilities("mistral-large-latest").reasoning_effort);
    }

    #[test]
    fn is_mistral_model_matches_every_current_family_prefix() {
        for id in [
            "codestral-latest",
            "devstral-2512",
            "ministral-3b-latest",
            "magistral-small",
            "mistral-large-latest",
            "pixtral-12b",
            "open-mistral-7b",
            "open-mixtral-8x22b",
            "labs-devstral-small-2512",
            "MISTRAL-SMALL-LATEST", // case-insensitive
        ] {
            assert!(is_mistral_model(id), "{id} should be recognized as Mistral");
        }
        for id in ["gpt-4o", "claude-opus-4-8", "deepseek-v4-pro", "some-vendor/model"] {
            assert!(!is_mistral_model(id), "{id} should not be recognized as Mistral");
        }
    }

    #[test]
    fn is_deepseek_model_is_narrower_than_the_shared_deepseek_wire_shape() {
        assert!(is_deepseek_model("deepseek-v4-pro"));
        assert!(is_deepseek_model("DeepSeek-V4-Flash"));
        // Kimi/Moonshot shares `OpenAiReasoningFormat::DeepSeek`'s wire *shape* but is a different
        // provider — `is_deepseek_model` must not conflate the two.
        assert!(!is_deepseek_model("kimi-k2-thinking"));
        assert!(!is_deepseek_model("gpt-4o"));
    }

    // ---- Per-model reasoning-effort wire remap (pi-parity pass 15) ----

    #[test]
    fn reasoning_wire_override_remaps_deepseek_xhigh_to_max() {
        use crate::transport::ReasoningEffort as RE;
        assert_eq!(reasoning_wire_override("deepseek-v4-pro", RE::XHigh), Some("max"));
        // High is a literal passthrough — no override.
        assert_eq!(reasoning_wire_override("deepseek-v4-pro", RE::High), None);
    }

    #[test]
    fn reasoning_wire_override_remaps_glm_5_2_low_medium_and_xhigh() {
        use crate::transport::ReasoningEffort as RE;
        for (effort, wire) in [(RE::Low, "high"), (RE::Medium, "high"), (RE::High, "high")] {
            assert_eq!(reasoning_wire_override("glm-5.2", effort), Some(wire), "{effort:?}");
        }
        assert_eq!(reasoning_wire_override("glm-5.2", RE::XHigh), Some("max"));
        // Minimal is excluded by the model's own floor (`min_reasoning_effort: Low`) before this
        // function would ever see it in practice; confirm that floor directly.
        assert_eq!(capabilities("glm-5.2").min_reasoning_effort, RE::Low);
        assert_eq!(capabilities("glm-4.7").min_reasoning_effort, RE::Minimal);
    }

    #[test]
    fn reasoning_wire_override_remaps_vendor_slug_glm_5_2_same_as_the_bare_id() {
        use crate::transport::ReasoningEffort as RE;
        // "zai-org/glm-5.2" (Together/HuggingFace) must remap identically to the bare "glm-5.2" it's
        // slug-prefixed with, not silently fall through to `None` because the full id doesn't start
        // with "glm-5.2".
        assert_eq!(reasoning_wire_override("zai-org/glm-5.2", RE::Low), Some("high"));
        assert_eq!(reasoning_wire_override("zai-org/glm-5.2", RE::XHigh), Some("max"));
    }

    #[test]
    fn reasoning_wire_override_remaps_groq_qwen_high_to_default() {
        use crate::transport::ReasoningEffort as RE;
        assert_eq!(reasoning_wire_override("qwen/qwen3-32b", RE::High), Some("default"));
        // A different qwen host (Together-shaped) isn't this one exact Groq id — no override.
        assert_eq!(reasoning_wire_override("qwen/qwen3.6-plus", RE::High), None);
    }

    #[test]
    fn reasoning_wire_override_sends_high_for_every_mistral_reasoning_id() {
        use crate::transport::ReasoningEffort as RE;
        // Mistral's real reasoning_effort vocabulary is a bare "none"|"high" enum with no per-level
        // map at all — every active level collapses to "high" on the wire, matching
        // `mapReasoningEffort`'s literal fallback in pi's own Mistral client.
        for effort in [RE::Minimal, RE::Low, RE::Medium, RE::High, RE::XHigh] {
            assert_eq!(
                reasoning_wire_override("mistral-small-2603", effort),
                Some("high"),
                "{effort:?}"
            );
        }
        // A non-reasoning Mistral id is never routed here by the caller, but the function itself is
        // still harmless if it somehow were.
        assert_eq!(
            reasoning_wire_override("mistral-large-latest", RE::High),
            Some("high")
        );
    }

    #[test]
    fn reasoning_wire_override_is_none_for_every_other_family() {
        use crate::transport::ReasoningEffort as RE;
        for id in ["o3", "gpt-5.2", "claude-opus-4-8", "grok-4.3", "kimi-k2-thinking"] {
            assert_eq!(reasoning_wire_override(id, RE::High), None, "{id}");
        }
    }

    #[test]
    fn is_non_standard_store_provider_matches_pis_denylist_where_the_id_alone_can_tell() {
        for id in [
            "deepseek-v4-pro",
            "glm-5.2",
            "kimi-k2-thinking",
            "grok-4.3",
            "qwen/qwen3.6-plus",
            "gpt-oss-120b",
            "zai-glm-4.7",
        ] {
            assert!(is_non_standard_store_provider(id), "{id} should be non-standard");
        }
        // The one Groq-hosted qwen exception is standard (Groq isn't in pi's denylist).
        assert!(!is_non_standard_store_provider("qwen/qwen3-32b"));
        for id in ["gpt-4o", "claude-opus-4-8", "mistral-large-latest", "some-vendor/model"] {
            assert!(!is_non_standard_store_provider(id), "{id} should be standard");
        }
    }
}
