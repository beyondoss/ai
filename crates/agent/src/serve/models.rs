//! Model ids and lists: the catalog the session offers, `--models` scoping, and resolving what a client typed.

use crate::serve::*;

/// The `--reasoning-effort` default (Fix 1, pi-parity gap) both `main.rs`'s plain `run` path and this
/// module's own `serve` startup fall back to when the operator passed neither an explicit
/// `--reasoning-effort` flag nor a stored `agent settings --default-reasoning-effort` override: pi's own
/// `DEFAULT_THINKING_LEVEL` ("medium", `packages/coding-agent/src/core/defaults.ts`) applies whenever
/// `model` supports reasoning/thinking at all — a bare `pi "task"` invocation runs with medium
/// adaptive-thinking effort on pi's own default model, not with thinking disabled outright. Previously
/// this crate left the effort `None` in that case, which — for a model whose thinking mechanism can be
/// explicitly disabled (`reasoning_disableable`) — wire-sends an explicit `{"type":"disabled"}"`/
/// `{"effort":"none"}` rather than picking any default depth at all.
///
/// `None` for a model with no thinking/reasoning mechanism whatsoever — see
/// [`agent_core::models::has_reasoning_mechanism`] — nothing to default there. Deliberately a
/// CLI/serve-layer default, not new `agent_core` plumbing: `agent_core::Agent::new`/
/// `with_reasoning_effort` are unchanged, still doing exactly nothing when the caller never calls the
/// latter — this only changes what `main.rs`/`serve` pass in when the operator didn't specify anything
/// themselves.
///
/// Calls agent-core's own [`has_reasoning_mechanism`](agent_core::models::has_reasoning_mechanism)
/// rather than reimplementing the check here: a prior version tested only `caps.thinking !=
/// ThinkingShape::None || caps.reasoning_effort`, narrower than agent-core's own internal gate by
/// exactly one arm (`caps.openai_reasoning_format != OpenAiReasoningFormat::Standard`) — so
/// Kimi-thinking and pre-5.2 GLM models, which signal their reasoning toggle only via that third arm,
/// silently fell through to "no default" here even though pi's own reference defaults them to medium
/// effort too.
pub fn default_reasoning_effort_for_model(model: &str) -> Option<agent_core::ReasoningEffort> {
    let caps = agent_core::capabilities(model);
    agent_core::models::has_reasoning_mechanism(&caps)
        .then_some(agent_core::ReasoningEffort::Medium)
}

/// A small, non-exhaustive list of model ids the [`capabilities`](agent_core::capabilities) table
/// recognizes, for a client's model picker. The gateway forwards any id verbatim, so this is a
/// convenience hint — not an allowlist; `set_model` accepts ids outside this list. Shared by `serve`'s
/// own `get_available_models`/`cycle_model` and `main`'s `list-models` CLI subcommand.
pub fn available_models() -> &'static [&'static str] {
    &[
        "claude-fable-5-1",
        "claude-opus-5",
        "claude-opus-4-8",
        "claude-sonnet-5",
        "claude-haiku-4-5",
        "gpt-6-astra",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.5",
        "gpt-5",
        "gpt-5-mini",
        "gpt-4o",
        "gpt-4.1",
        "o3",
        "o4-mini",
    ]
}

/// Every model to *show*: [`available_models`], then every other chat model in the provider
/// catalog that a native provider serves under its bare id (`providers::for_model_id`): the current
/// Claude, GPT and o-series, Grok, DeepSeek and Groq-hosted rows. What `list-models` and
/// `get_available_models` list, so a model's capabilities and compaction trigger are visible the day
/// its catalog row lands.
///
/// Not what a partial `--model`/`set_model` id resolves against: that stays [`available_models`],
/// whose short, unambiguous list keeps `--model sonnet` or `--model haiku` resolving to one id
/// instead of erroring on several. Vendor-slug rows (`qwen/…`, aggregator-only) and embedding models
/// are left out.
pub fn listed_models() -> &'static [&'static str] {
    static LISTED: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();
    LISTED.get_or_init(|| {
        let mut out = available_models().to_vec();
        for row in providers::catalog::MODEL_ROUTES {
            if !row.model.contains('/')
                && row.card.max_output_tokens > 0
                && providers::for_model_id(row.model).is_some()
                && !out.contains(&row.model)
            {
                out.push(row.model);
            }
        }
        out
    })
}

/// One resolved entry in a `--models` scope: a model id plus an optional thinking level pinned via a
/// trailing `:<level>` suffix on the pattern that produced it. Mirrors pi's own `ScopedModel`
/// (`model-resolver.ts`), minus the `provider/id` canonical form pi's real multi-provider registry
/// carries — this crate has no such registry (`model_info`'s doc comment), so a glob only matches
/// against the bare id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ScopedModel {
    pub(super) id: String,
    pub(super) thinking_level: Option<agent_core::ThinkingLevel>,
}

/// Parses `--models`'s comma-separated pattern list (already split by clap) into a resolved candidate
/// set — pi's own `resolveModelScopeWithDiagnostics`. Each pattern, independently:
///
/// - has a trailing `:<level>` suffix stripped off and remembered as that entry's pinned thinking
///   level (one of off/minimal/low/medium/high/xhigh — `agent_core::ThinkingLevel::parse`), applied
///   whenever `cycle_model` lands on it; a suffix that isn't a valid level is left alone and treated
///   as part of the id/pattern itself, same as pi's `parseModelPattern` recursion;
/// - if what's left contains a glob metacharacter (`*`, `?`, `[`), is matched case-insensitively
///   against every id in `catalog`, expanding to every match in `catalog`'s order — e.g. `claude-*`
///   or `*sonnet*` — each getting the same pinned level;
/// - otherwise is kept as a literal id verbatim, whether or not it's in `catalog` — matching
///   `available_models`'s own "hint, not an allowlist" contract, since the gateway forwards any id
///   verbatim regardless of what this process happens to know about.
///
/// A glob that matches nothing is dropped silently — pi's own resolver only ever warns, never
/// hard-fails, and cycling still works from whatever else resolved. A duplicate id keeps the first
/// occurrence's pin, same as pi's `modelsAreEqual` dedup.
///
/// Splits a trailing `:<level>` suffix (one of `agent_core::ThinkingLevel::parse`'s vocabulary) off
/// `pattern`, shared by [`resolve_model_scope`] (a `--models` entry) and [`resolve_model_id`] (a bare
/// `--model`/`set_model` id) — Fix 2 (pi-parity gap): `resolve_model_id` previously had no colon-suffix
/// handling at all, unlike pi's own `model-resolver.ts::parseModelPattern`, which applies it to
/// `--model`/`--provider` too (its own help text example: `pi --model sonnet:high`). A suffix that isn't
/// a valid level is left attached to `pattern` untouched — the whole string is then resolved (or kept
/// literal) as one id, same as if no colon were present at all.
pub(super) fn split_thinking_level_suffix(
    pattern: &str,
) -> (&str, Option<agent_core::ThinkingLevel>) {
    match pattern.rfind(':') {
        Some(idx) => {
            let suffix = &pattern[idx + 1..];
            match agent_core::ThinkingLevel::parse(suffix) {
                Some(level) => (&pattern[..idx], Some(level)),
                None => (pattern, None),
            }
        }
        None => (pattern, None),
    }
}

pub(super) fn resolve_model_scope(patterns: &[String], catalog: &[&str]) -> Vec<ScopedModel> {
    let mut resolved: Vec<ScopedModel> = Vec::new();
    for raw in patterns {
        let pattern = raw.trim();
        if pattern.is_empty() {
            continue;
        }
        let (base, thinking_level) = split_thinking_level_suffix(pattern);
        let is_glob = base.chars().any(|c| matches!(c, '*' | '?' | '['));
        if is_glob {
            let matcher = match globset::GlobBuilder::new(base)
                .case_insensitive(true)
                .build()
            {
                Ok(g) => g.compile_matcher(),
                Err(_) => continue,
            };
            for &id in catalog {
                if matcher.is_match(id) && !resolved.iter().any(|m| m.id == id) {
                    resolved.push(ScopedModel {
                        id: id.to_string(),
                        thinking_level,
                    });
                }
            }
        } else {
            // Fix 10 (pi-parity feature): a literal, non-glob pattern that doesn't exactly match a
            // catalog id now also gets the same partial/substring resolution `--model`/`set_model` get
            // — e.g. `--models sonnet` resolves to `claude-sonnet-5` instead of cycling to the
            // literal, almost-certainly-wrong string "sonnet". An ambiguous partial match can't fail
            // the whole `serve` startup the way an ambiguous `--model` does (this is a background
            // candidate-list build, not the one model actively in use) — it's warned about and kept
            // literal instead, same graceful-degrade this crate already applies to a glob matching
            // nothing.
            let resolved_id = match resolve_model_id_base(base, catalog) {
                Ok(id) => id,
                Err(e) => {
                    eprintln!("warning: --models {base:?}: {e}; using it literally");
                    base.to_string()
                }
            };
            if !resolved.iter().any(|m| m.id == resolved_id) {
                resolved.push(ScopedModel {
                    id: resolved_id,
                    thinking_level,
                });
            }
        }
    }
    resolved
}

/// Resolve a possibly-partial `--model`/`set_model` id against the known-model hint list
/// ([`available_models`]) — pi's own `model-resolver.ts` partial/substring matching, scoped down: this
/// crate has no per-provider catalog to rank un-dated aliases against dated snapshots with (pi's own
/// alias-preference rule), and — unlike pi's own `resolveCliModel`, which silently picks whichever
/// candidate sorts first — never silently guesses on an ambiguous match. Instead mirrors
/// `SessionRepo::find_path`'s own "ambiguous is a real error naming every candidate, not a guess"
/// philosophy (see `session_store/`), applied consistently to `--model`, `set_model`, and (via
/// `resolve_model_scope`, above) the literal entries in a `--models` scope.
///
/// - An exact, case-insensitive match against `catalog` resolves to the catalog's own canonical
///   spelling (so `--model Claude-Opus-4-8` still lands on `claude-opus-4-8`).
/// - Otherwise, every catalog id that *contains* `input` as a substring (case-insensitive) is a
///   candidate — e.g. `sonnet` matches `claude-sonnet-5`. Exactly one candidate resolves to it; more than
///   one is `Err`, naming every candidate so the caller can be specific instead of silently guessing
///   which one was meant (e.g. `gpt` matches `gpt-6-astra`/`gpt-5.6-*`/`gpt-5`/`gpt-4o`/…).
/// - No candidates at all returns `Ok(input)` unchanged — `available_models` is explicitly documented as
///   a non-exhaustive hint, not an allowlist, so an id genuinely outside it (a brand-new model, a
///   provider-specific id the gateway still understands) must still reach the gateway verbatim rather
///   than being rejected outright.
///
/// Fix 2 (pi-parity gap): `input` may also carry a trailing `:<level>` suffix (e.g. `sonnet:high`,
/// pi's own `--model <pattern>:<thinking-level>` shorthand) — split off via
/// [`split_thinking_level_suffix`] *before* resolution, so both `--model`'s own startup resolution and
/// the `set_model` RPC handler get it for free, the same way [`resolve_model_scope`]'s `--models`
/// entries already do. The returned level is the caller's to apply (setting the corresponding
/// reasoning effort); a suffix that isn't a valid level is left attached to the id itself, and resolved
/// (or kept literal) as one string, exactly as if no colon were present.
pub fn resolve_model_id(
    input: &str,
    catalog: &[&str],
) -> Result<(String, Option<agent_core::ThinkingLevel>), String> {
    let (base, thinking_level) = split_thinking_level_suffix(input.trim());
    resolve_model_id_base(base, catalog).map(|id| (id, thinking_level))
}

/// [`resolve_model_id`]'s resolution, without the colon-suffix split — the id-only lookup
/// [`resolve_model_scope`]'s own literal-pattern branch calls directly (it already extracted any
/// `:<level>` suffix itself, via the same [`split_thinking_level_suffix`] helper, so re-splitting here
/// would be a no-op at best and a double-strip of a legitimately colon-containing id at worst).
pub(super) fn resolve_model_id_base(input: &str, catalog: &[&str]) -> Result<String, String> {
    let trimmed = input.trim();
    if let Some(exact) = catalog.iter().find(|m| m.eq_ignore_ascii_case(trimmed)) {
        return Ok((*exact).to_string());
    }
    let lower = trimmed.to_ascii_lowercase();
    let candidates: Vec<&str> = catalog
        .iter()
        .copied()
        .filter(|m| m.to_ascii_lowercase().contains(&lower))
        .collect();
    match candidates.as_slice() {
        [] => Ok(trimmed.to_string()),
        [one] => Ok((*one).to_string()),
        many => Err(format!(
            "{trimmed:?} matches more than one known model id: {} — pass one of these exactly, or a \
             full id outside this hint list to bypass resolution entirely",
            many.join(", ")
        )),
    }
}

/// A `get_available_models` entry: enough per-model capability info (from the same
/// [`agent_core::capabilities`] table every wire decision already consults) for a client to build a
/// capability-aware picker without shipping its own hardcoded model registry. pi: `Model<any>`
/// (`rpc-types.ts:143-146`) — `id`/`contextWindow`/`reasoning` carry the same *kind* of information
/// here under this codebase's own field-naming convention (snake_case, matching `get_state`/
/// `get_session_stats`); `provider` is a coarse wire-family hint (`agent_core::dialect::Dialect::
/// for_model`, "anthropic"/"openai"), not a real routing provider id — this crate has no such registry,
/// the gateway owns routing. Pricing (pi's `cost`) is deliberately omitted: gateway-owned, out of scope
/// for this crate (see `ServeConfig`'s module doc).
pub(super) fn model_info(model: &str) -> Value {
    let caps = agent_core::capabilities(model);
    let provider = match agent_core::dialect::Dialect::for_model(model) {
        agent_core::dialect::Dialect::Anthropic => "anthropic",
        agent_core::dialect::Dialect::OpenAi | agent_core::dialect::Dialect::OpenAiResponses => {
            "openai"
        }
    };
    json!({
        "id": model,
        "provider": provider,
        "context_window": caps.context_window,
        "max_output": caps.max_output,
        // Shares `default_reasoning_effort_for_model`'s own `has_reasoning_mechanism` gate rather than a
        // separate `reasoning_effort || thinking != None` check — that narrower pair previously missed
        // the `openai_reasoning_format` toggle shapes (Kimi-thinking, pre-5.2 GLM), reporting `reasoning:
        // false` for a model that genuinely does support a client-steerable reasoning toggle.
        "reasoning": agent_core::models::has_reasoning_mechanism(&caps),
        "supports_vision": caps.supports_vision,
    })
}

/// [`model_info`] plus the thinking-level state a model *switch* also carries — the shared response
/// shape for `set_model`/`cycle_model`, so a client gets the same capability info
/// `get_available_models` already provides (context window, max output, reasoning/vision support)
/// instead of just a bare model id, matching pi's own `set_model`/`cycle_model` (`rpc-mode.ts`), which
/// embed the identical per-model object `get_available_models` uses. `model`/`reasoning_effort` are
/// kept as top-level fields (not folded into `model_info`'s own `id`) for wire-compatibility with a
/// client already reading this shape.
pub(super) fn model_switch_response(model: &str, level: agent_core::ThinkingLevel) -> Value {
    let mut info = model_info(model);
    if let Value::Object(map) = &mut info {
        map.insert("model".to_string(), json!(model));
        map.insert("reasoning_effort".to_string(), json!(level.as_str()));
    }
    info
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_natively_served_catalog_chat_model_is_listed() {
        // A model the catalog routes but `list-models`/`get_available_models` never show has no
        // visible capabilities or compaction trigger (claude-haiku-5-5's 90K was missing).
        let listed = listed_models();
        for row in providers::catalog::MODEL_ROUTES {
            if row.model.contains('/') || row.card.max_output_tokens == 0 {
                continue;
            }
            if providers::for_model_id(row.model).is_some() {
                assert!(listed.contains(&row.model), "{} is not listed", row.model);
            }
        }
        for id in [
            "claude-haiku-5-5",
            "claude-opus-5-5",
            "claude-sonnet-5-5",
            "gpt-6.1-sol",
        ] {
            assert!(listed.contains(&id), "{id}");
        }
        assert!(!listed.contains(&"text-embedding-3-small"));
        // The hint list leads, in its own order, and partial ids still resolve against it alone.
        assert_eq!(&listed[..available_models().len()], available_models());
        assert_eq!(
            resolve_model_id("haiku", available_models()).unwrap().0,
            "claude-haiku-4-5"
        );
    }

    #[test]
    fn resolve_model_scope_keeps_a_literal_id_verbatim_even_outside_the_catalog() {
        // `available_models` is a hint, not an allowlist (its own doc comment) — the gateway forwards
        // any id verbatim, so a literal the operator typed must survive resolution unchanged.
        let scoped = resolve_model_scope(&["totally-custom-id".to_string()], available_models());
        assert_eq!(
            scoped,
            vec![ScopedModel {
                id: "totally-custom-id".to_string(),
                thinking_level: None,
            }]
        );
    }

    #[test]
    fn resolve_model_scope_fuzzy_resolves_a_literal_partial_match_against_the_catalog() {
        // Fix 10: a literal, non-glob `--models` entry that partially matches exactly one catalog id
        // now resolves to it, the same as `--model`/`set_model` — `--models sonnet` must cycle to
        // `claude-sonnet-5`, not the literal, almost-certainly-wrong string "sonnet".
        let scoped = resolve_model_scope(&["sonnet".to_string()], available_models());
        assert_eq!(
            scoped,
            vec![ScopedModel {
                id: "claude-sonnet-5".to_string(),
                thinking_level: None,
            }]
        );
    }

    #[test]
    fn resolve_model_scope_an_ambiguous_literal_partial_match_warns_and_keeps_it_literal() {
        // Unlike `--model`, an ambiguous `--models` entry must not fail the whole `serve` startup —
        // this is a background candidate-list build, not the one model actively in use — so it falls
        // back to the literal string (with a warning), same graceful-degrade already applied to a glob
        // matching nothing.
        let scoped = resolve_model_scope(&["gpt".to_string()], available_models());
        assert_eq!(
            scoped,
            vec![ScopedModel {
                id: "gpt".to_string(),
                thinking_level: None,
            }]
        );
    }

    #[test]
    fn resolve_model_scope_expands_a_glob_against_the_catalog_in_catalog_order() {
        let scoped = resolve_model_scope(&["claude-*".to_string()], available_models());
        let ids: Vec<&str> = scoped.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(
            ids,
            vec![
                "claude-fable-5-1",
                "claude-opus-5",
                "claude-opus-4-8",
                "claude-sonnet-5",
                "claude-haiku-4-5",
            ]
        );
    }

    #[test]
    fn resolve_model_scope_glob_matching_is_case_insensitive() {
        let scoped = resolve_model_scope(&["CLAUDE-*".to_string()], available_models());
        assert_eq!(scoped.len(), 5, "got: {scoped:?}");
    }

    #[test]
    fn resolve_model_scope_a_glob_matching_nothing_is_dropped_silently() {
        let scoped = resolve_model_scope(
            &["no-such-provider/*".to_string(), "gpt-4o".to_string()],
            available_models(),
        );
        assert_eq!(
            scoped,
            vec![ScopedModel {
                id: "gpt-4o".to_string(),
                thinking_level: None,
            }]
        );
    }

    #[test]
    fn resolve_model_scope_pattern_level_suffix_pins_a_literal_entry() {
        let scoped = resolve_model_scope(&["gpt-4o:high".to_string()], available_models());
        assert_eq!(
            scoped,
            vec![ScopedModel {
                id: "gpt-4o".to_string(),
                thinking_level: Some(agent_core::ThinkingLevel::High),
            }]
        );
    }

    #[test]
    fn resolve_model_scope_pattern_level_suffix_pins_every_glob_match_the_same_way() {
        let scoped = resolve_model_scope(&["claude-*:low".to_string()], available_models());
        assert!(
            scoped
                .iter()
                .all(|m| m.thinking_level == Some(agent_core::ThinkingLevel::Low)),
            "got: {scoped:?}"
        );
        assert_eq!(scoped.len(), 5);
    }

    #[test]
    fn resolve_model_scope_an_invalid_level_suffix_is_kept_as_part_of_the_literal_id() {
        // pi's own scope-mode fallback: a trailing `:bogus` isn't a real thinking level, so the whole
        // string — colon included — is treated as the pattern/id rather than silently truncated.
        let scoped = resolve_model_scope(&["gpt-4o:bogus".to_string()], available_models());
        assert_eq!(
            scoped,
            vec![ScopedModel {
                id: "gpt-4o:bogus".to_string(),
                thinking_level: None,
            }]
        );
    }

    #[test]
    fn resolve_model_scope_dedupes_a_glob_overlapping_a_literal_keeping_the_first_occurrences_pin()
    {
        let scoped = resolve_model_scope(
            &["gpt-4o:high".to_string(), "gpt-*".to_string()],
            available_models(),
        );
        let gpt_4o = scoped.iter().find(|m| m.id == "gpt-4o").unwrap();
        assert_eq!(gpt_4o.thinking_level, Some(agent_core::ThinkingLevel::High));
        // The rest of the `gpt-*` glob still resolved past the one duplicate.
        assert!(scoped.iter().any(|m| m.id == "gpt-5"));
    }

    #[test]
    fn resolve_model_scope_trims_whitespace_and_skips_empty_patterns() {
        let scoped = resolve_model_scope(
            &[" gpt-4o ".to_string(), "".to_string(), "  ".to_string()],
            available_models(),
        );
        assert_eq!(
            scoped,
            vec![ScopedModel {
                id: "gpt-4o".to_string(),
                thinking_level: None,
            }]
        );
    }

    // Fix 1 (pi-parity gap): a bare invocation with no `--reasoning-effort` previously left the
    // effort/level `None`/`Off`, wire-disabling reasoning outright for a model that can explicitly
    // disable it, instead of picking pi's own "medium" default.

    #[test]
    fn default_reasoning_effort_for_model_is_medium_on_the_built_in_default_model() {
        assert_eq!(
            default_reasoning_effort_for_model("claude-opus-4-8"),
            Some(agent_core::ReasoningEffort::Medium)
        );
    }

    #[test]
    fn default_reasoning_effort_for_model_is_medium_for_any_reasoning_capable_model() {
        // Budget-shape (extended thinking, not an OpenAI `reasoning_effort` toggle) counts too.
        assert_eq!(
            default_reasoning_effort_for_model("claude-test"),
            Some(agent_core::ReasoningEffort::Medium)
        );
    }

    #[test]
    fn default_reasoning_effort_for_model_is_none_for_a_model_with_no_reasoning_mechanism_at_all() {
        assert_eq!(
            default_reasoning_effort_for_model("claude-3-5-sonnet-20241022"),
            None
        );
        assert_eq!(default_reasoning_effort_for_model("gpt-4o"), None);
    }

    // Fix 1 (pi-parity remediation, Round 2): `default_reasoning_effort_for_model` previously
    // reimplemented `has_reasoning_mechanism`'s own check narrower than the original — missing the
    // `openai_reasoning_format` arm — so a Kimi-thinking or pre-5.2 GLM model (both toggle reasoning
    // only via that third-party wire format, with `reasoning_effort: false` and
    // `thinking: ThinkingShape::None`) silently got no default reasoning effort at all, even though pi's
    // own reference defaults every reasoning-capable model to medium effort.
    #[test]
    fn default_reasoning_effort_for_model_is_medium_for_a_kimi_thinking_model() {
        assert_eq!(
            default_reasoning_effort_for_model("kimi-k2-thinking"),
            Some(agent_core::ReasoningEffort::Medium)
        );
        // K3 always thinks; the default is still Pi's medium, not the provider's hidden `max`.
        assert_eq!(
            default_reasoning_effort_for_model("moonshotai/kimi-k3"),
            Some(agent_core::ReasoningEffort::Medium)
        );
    }

    #[test]
    fn default_reasoning_effort_for_model_is_medium_for_a_pre_5_2_glm_model() {
        assert_eq!(
            default_reasoning_effort_for_model("glm-4.5"),
            Some(agent_core::ReasoningEffort::Medium)
        );
    }

    // Fix 1's sibling gap: `model_info`'s own `"reasoning"` field (surfaced via `get_available_models`/
    // `set_model`/`cycle_model`) reimplemented the identical narrower check independently, so it too
    // reported `false` for these families even though they have a genuine client-steerable toggle.
    #[test]
    fn model_info_reasoning_is_true_for_a_kimi_thinking_model() {
        assert_eq!(model_info("kimi-k2-thinking")["reasoning"], true);
    }

    #[test]
    fn model_info_reasoning_is_true_for_a_pre_5_2_glm_model() {
        assert_eq!(model_info("glm-4.5")["reasoning"], true);
    }

    // Fix 10 (pi-parity feature): `--model`/`set_model` previously forwarded any id verbatim with no
    // resolution at all. `resolve_model_id` ports a scoped-down version of pi's own
    // `model-resolver.ts` partial/substring matching, but never silently guesses on an ambiguous
    // match — mirrors `SessionRepo::find_path`'s own "list every candidate, don't guess" philosophy
    // instead of pi's own silent-pick-first behavior.

    #[test]
    fn resolve_model_id_an_exact_match_is_returned_in_the_catalogs_own_casing() {
        assert_eq!(
            resolve_model_id("Claude-Opus-4-8", available_models())
                .unwrap()
                .0,
            "claude-opus-4-8"
        );
    }

    #[test]
    fn resolve_model_id_an_unambiguous_partial_match_resolves_to_the_full_id() {
        assert_eq!(
            resolve_model_id("opus-5", available_models()).unwrap().0,
            "claude-opus-5"
        );
        assert_eq!(
            resolve_model_id("SONNET", available_models()).unwrap().0,
            "claude-sonnet-5"
        );
    }

    #[test]
    fn resolve_model_id_an_ambiguous_partial_match_errors_naming_every_candidate() {
        let err = resolve_model_id("gpt", available_models()).unwrap_err();
        assert!(err.contains("gpt-5"), "got: {err}");
        assert!(err.contains("gpt-5-mini"), "got: {err}");
        assert!(err.contains("gpt-6-astra"), "got: {err}");
        assert!(err.contains("gpt-4o"), "got: {err}");
        assert!(err.contains("gpt-4.1"), "got: {err}");
        let opus_err = resolve_model_id("opus", available_models()).unwrap_err();
        assert!(opus_err.contains("claude-opus-5"), "got: {opus_err}");
        assert!(opus_err.contains("claude-opus-4-8"), "got: {opus_err}");
    }

    #[test]
    fn resolve_model_id_no_match_at_all_is_forwarded_verbatim() {
        // `available_models` is a hint, not an allowlist — a brand-new or provider-specific id the
        // gateway still understands must reach it unchanged, not be rejected.
        assert_eq!(
            resolve_model_id("totally-custom-id", available_models())
                .unwrap()
                .0,
            "totally-custom-id"
        );
    }

    #[test]
    fn resolve_model_id_an_exact_match_short_circuits_before_any_ambiguity_check() {
        // "gpt-5" is itself a substring of "gpt-5-mini" too, but an exact match must win outright
        // rather than ever reaching the ambiguous-candidates error path.
        assert_eq!(
            resolve_model_id("gpt-5", available_models()).unwrap().0,
            "gpt-5"
        );
    }

    #[test]
    fn resolve_model_id_a_colon_level_suffix_resolves_the_id_and_returns_the_level() {
        // Fix 2 (pi-parity gap): `--model sonnet:high` must resolve to the sonnet model id *and*
        // report `high` as the level to apply — pi's own `--model <pattern>:<thinking-level>`
        // shorthand, previously only honored by `--models`/`resolve_model_scope`.
        let (id, level) = resolve_model_id("sonnet:high", available_models()).unwrap();
        assert_eq!(id, "claude-sonnet-5");
        assert_eq!(level, Some(agent_core::ThinkingLevel::High));
    }

    #[test]
    fn resolve_model_id_no_colon_suffix_returns_no_level() {
        let (id, level) = resolve_model_id("opus-5", available_models()).unwrap();
        assert_eq!(id, "claude-opus-5");
        assert_eq!(level, None);
    }

    #[test]
    fn resolve_model_id_an_invalid_colon_suffix_is_kept_as_part_of_the_literal_id() {
        // Matches `resolve_model_scope`'s identical treatment of an invalid suffix: not a valid
        // thinking level, so it's left attached and the whole string is resolved (or kept literal) as
        // one id rather than silently dropped.
        let (id, level) =
            resolve_model_id("totally-custom-id:notalevel", available_models()).unwrap();
        assert_eq!(id, "totally-custom-id:notalevel");
        assert_eq!(level, None);
    }
}
