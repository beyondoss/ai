//! What a managed request may not ask for: features a provider bills outside the usage the
//! gateway can meter, so a row could never price them (`providers::pricing` refuses such a row;
//! see `crates/providers/ARCHITECTURE.md`, "Provider × dimension matrix"). Refused with a 400
//! before any byte of the body goes upstream, so nothing runs unpriced. A BYO key is never checked:
//! its spend is the caller's own.
//!
//! | Asked for                                                        | Why it cannot be priced                                   |
//! | ---------------------------------------------------------------- | --------------------------------------------------------- |
//! | OpenAI `code_interpreter` / hosted `shell` tool                  | containers bill per session minute by memory, unreported  |
//! | `image_generation` tool (OpenAI, xAI)                            | image tokens are not in `usage`                           |
//! | `web_search*` tool on gpt-4o-mini / gpt-4.1-mini                 | a fixed 8,000-token content block, unmeasured             |
//! | Anthropic `code_execution_*` tool, root `container`              | container-hours, unreported                               |
//! | Anthropic `advisor_*` tool                                       | the advisor's tokens are outside top-level `usage`        |
//! | Groq `browser_search` (and `code_interpreter`)                   | no published fee                                          |
//! | OpenRouter `plugins` (web, file-parser), `:online` models, `web_search_options`, `openrouter:web_search` | engine and result count unreported |
//! | Messages `service_tier: "priority"`                              | Anthropic Priority Tier is contract-priced                |
//! | any other tool type                                              | not one the gateway meters (see [`allowed_tool`])         |
//!
//! Read structurally from the client's body (root members only, and each `tools` element's own
//! `type`), so a prompt that mentions a tool cannot trip it.

use crate::peek;
use providers::catalog::strip_snapshot_date;

/// What [`inspect`] found in a managed request body.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Inspection {
    /// Why the request is refused: the message for the client's 400.
    pub refused: Option<&'static str>,
    /// The request offers OpenAI's `web_search_preview` tool, priced apart from `web_search` (its
    /// calls are `web_search_call` items either way, so only the request tells them apart).
    pub web_search_preview: bool,
}

/// OpenRouter plugins that bill nothing beyond the tokens: everything else is refused.
const FREE_PLUGINS: [&str; 2] = ["context-compression", "response-healing"];

/// The 400 messages. Static, as the gateway's error path requires.
pub const CONTAINERS: &str = "a managed key cannot use OpenAI's code_interpreter or hosted shell tool: container time is billed outside the usage the gateway meters; use your own provider key";
pub const IMAGE_GENERATION: &str = "a managed key cannot use the image_generation tool: its image tokens are billed outside the usage the gateway meters; use your own provider key";
pub const MINI_WEB_SEARCH: &str = "a managed key cannot use web search on gpt-4o-mini or gpt-4.1-mini: each call bills a fixed content block the gateway cannot meter; use another model or your own provider key";
pub const CODE_EXECUTION: &str = "a managed key cannot use Anthropic code execution or a container: container-hours are billed outside the usage the gateway meters; use your own provider key";
pub const ADVISOR: &str = "a managed key cannot use the advisor tool: the advisor's tokens are billed outside the usage the gateway meters; use your own provider key";
pub const GROQ_TOOLS: &str = "a managed key cannot use Groq's browser_search tool: it has no published price; use your own provider key";
pub const OPENROUTER_WEB: &str = "a managed key cannot use OpenRouter web search or plugins (plugins, :online models, web_search_options, openrouter:web_search): their engine and results are billed outside the usage the gateway meters; use your own provider key";
pub const UNLISTED_TOOL: &str = "a managed key cannot use a tool the gateway does not meter: its usage could be billed outside what the gateway can price; use your own provider key";
pub const PRIORITY_TIER: &str = "a managed key cannot use Anthropic Priority Tier (service_tier: priority): it is priced by contract; omit service_tier or use your own provider key";

/// Inspect a managed request's body. `model` is the model it names (the catalog row or the
/// provider's id); `messages` is whether the client speaks the Messages API.
pub fn inspect(body: &[u8], model: &str, messages: bool) -> Inspection {
    let mut out = Inspection::default();
    if model.ends_with(":online") {
        out.refused = Some(OPENROUTER_WEB);
        return out;
    }
    let Some(members) = peek::root_members(body) else {
        return out;
    };
    for m in &members {
        let span = m.value;
        let refused = if m.key_is(body, "tools") {
            tools(body, span.0, model, &mut out.web_search_preview)
        } else if m.key_is(body, "plugins") {
            plugins(body, span.0)
        } else if m.key_is(body, "web_search_options") {
            Some(OPENROUTER_WEB)
        } else if m.key_is(body, "container") {
            // `null` asks for nothing.
            (!matches!(body.get(span.0..span.1), Some(b"null"))).then_some(CODE_EXECUTION)
        } else if messages && m.key_is(body, "service_tier") {
            peek::str_is(body, span, "priority").then_some(PRIORITY_TIER)
        } else {
            None
        };
        if refused.is_some() {
            out.refused = refused;
            return out;
        }
    }
    out
}

/// The first refused tool in a `tools` array whose value starts at `open`.
fn tools(body: &[u8], open: usize, model: &str, preview: &mut bool) -> Option<&'static str> {
    if body.get(open) != Some(&b'[') {
        return None;
    }
    let mini = matches!(base_model(model), "gpt-4o-mini" | "gpt-4.1-mini");
    for (start, _) in peek::array_elements(body, open)? {
        // Not an object: no tool the gateway can meter.
        let Some(member) = peek::last_member(body, start, "type") else {
            return Some(UNLISTED_TOOL);
        };
        // No `type` at all is Anthropic's client tool (`name`, `input_schema`): the client runs it,
        // so no provider charge, and there is nothing to meter.
        let Some(member) = member else {
            continue;
        };
        let Some(ty) = peek::str_value(body, member.value) else {
            return Some(UNLISTED_TOOL);
        };
        let ty = ty.as_ref();
        if ty.starts_with("web_search_preview") {
            *preview = true;
        }
        let refused = match ty {
            "code_interpreter" | "shell" => Some(CONTAINERS),
            "image_generation" => Some(IMAGE_GENERATION),
            "browser_search" => Some(GROQ_TOOLS),
            "openrouter:web_search" => Some(OPENROUTER_WEB),
            t if t.starts_with("web_search") && mini => Some(MINI_WEB_SEARCH),
            t if t.starts_with("code_execution") => Some(CODE_EXECUTION),
            t if t.starts_with("advisor") => Some(ADVISOR),
            t if allowed_tool(t) => None,
            _ => Some(UNLISTED_TOOL),
        };
        if refused.is_some() {
            return refused;
        }
    }
    None
}

/// Whether a tool type is one the gateway meters or one the client itself runs. An allowlist, so a
/// new tool type is refused until it is known here: a denylist let an unknown billed tool through.
///
/// A family matches its exact name, or its name followed by `_` and a version or variant
/// (`web_search_20250305`, `web_search_preview`, `computer_use_preview`, `text_editor_20250728`).
/// `bashful_x` is not `bash`. Names listed as exact-only (`file_search`, `tool_search`, `x_search`,
/// `function`, ...) take no suffix, so Anthropic's hosted `tool_search_tool_*` is refused.
/// A tool with no `type` (Anthropic's `{name, input_schema}` client tool) is passed by the caller.
///
/// Client-run tools carry no provider charge: the client executes them and the provider bills the
/// tokens they travel in. The rest are server tools the provider runs and the gateway counts in
/// [`ServerTools`]. The refusals above are checked first, so a family listed here that is also
/// refused there (`shell`, `code_execution`) stays refused.
fn allowed_tool(ty: &str) -> bool {
    // One spelling each: the function-call shapes and the server tools with no version suffix.
    const EXACT: [&str; 8] = [
        "function",
        "custom",
        "namespace",
        "apply_patch",
        "local_shell",
        "file_search",
        "tool_search",
        "x_search",
    ];
    // Client-run tools (bash, text_editor, computer, memory) and metered server tools (web_search,
    // web_fetch, mcp), each by its name with or without a `_` suffix.
    const VERSIONED: [&str; 7] = [
        "bash",
        "text_editor",
        "computer",
        "memory",
        "web_search",
        "web_fetch",
        "mcp",
    ];
    EXACT.contains(&ty)
        || VERSIONED.iter().any(|family| {
            ty == *family
                || ty
                    .strip_prefix(family)
                    .is_some_and(|rest| rest.starts_with('_'))
        })
}

/// A refusal when a `plugins` array (at `open`) names a plugin outside [`FREE_PLUGINS`].
fn plugins(body: &[u8], open: usize) -> Option<&'static str> {
    if body.get(open) != Some(&b'[') {
        // Not an array: OpenRouter rejects it; nothing to price.
        return None;
    }
    let free = peek::array_elements(body, open)?
        .into_iter()
        .all(|(start, _)| {
            matches!(peek::last_member(body, start, "id"), Some(Some(id))
            if peek::str_value(body, id.value).is_some_and(|v| FREE_PLUGINS.contains(&v.as_ref())))
        });
    (!free).then_some(OPENROUTER_WEB)
}

/// `model` without a vendor prefix (`openai/`) or a dated snapshot suffix, as the price lookup
/// strips it (`proxy::price_row`), so the gate and the price name the same model.
fn base_model(model: &str) -> &str {
    let m = model.rsplit_once('/').map_or(model, |(_, m)| m);
    strip_snapshot_date(m).unwrap_or(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refused(body: &str, model: &str, messages: bool) -> Option<&'static str> {
        inspect(body.as_bytes(), model, messages).refused
    }

    #[test]
    fn every_unpriced_feature_is_refused() {
        let cases: [(&str, &str, bool, &str); 16] = [
            (
                r#"{"tools":[{"type":"code_interpreter","container":{"type":"auto"}}]}"#,
                "gpt-5",
                false,
                CONTAINERS,
            ),
            (
                r#"{"tools":[{"type":"shell"}]}"#,
                "gpt-5",
                false,
                CONTAINERS,
            ),
            (
                r#"{"tools":[{"type":"function","name":"f"},{"type":"image_generation"}]}"#,
                "gpt-5",
                false,
                IMAGE_GENERATION,
            ),
            (
                r#"{"tools":[{"type":"image_generation"}]}"#,
                "grok-4.3",
                false,
                IMAGE_GENERATION,
            ),
            (
                r#"{"tools":[{"type":"web_search"}]}"#,
                "gpt-4o-mini",
                false,
                MINI_WEB_SEARCH,
            ),
            (
                r#"{"tools":[{"type":"web_search_preview"}]}"#,
                "openai/gpt-4.1-mini-2025-04-14",
                false,
                MINI_WEB_SEARCH,
            ),
            (
                r#"{"tools":[{"type":"code_execution_20250825","name":"code_execution"}]}"#,
                "claude-opus-4-8",
                true,
                CODE_EXECUTION,
            ),
            (
                r#"{"container":"container_011x","messages":[]}"#,
                "claude-opus-4-8",
                true,
                CODE_EXECUTION,
            ),
            (
                r#"{"tools":[{"type":"advisor_20260301","name":"advisor"}]}"#,
                "claude-opus-4-8",
                true,
                ADVISOR,
            ),
            (
                r#"{"tools":[{"type":"browser_search"}]}"#,
                "openai/gpt-oss-120b",
                false,
                GROQ_TOOLS,
            ),
            (
                r#"{"plugins":[{"id":"web","max_results":3}]}"#,
                "anthropic/claude-opus-4.8",
                false,
                OPENROUTER_WEB,
            ),
            (
                r#"{"plugins":[{"id":"file-parser","pdf":{"engine":"mistral-ocr"}}]}"#,
                "x",
                false,
                OPENROUTER_WEB,
            ),
            (
                r#"{"messages":[]}"#,
                "anthropic/claude-opus-4.8:online",
                false,
                OPENROUTER_WEB,
            ),
            (
                r#"{"web_search_options":{}}"#,
                "anthropic/claude-opus-4.8",
                false,
                OPENROUTER_WEB,
            ),
            (
                r#"{"tools":[{"type":"openrouter:web_search"}]}"#,
                "anthropic/claude-opus-4.8",
                false,
                OPENROUTER_WEB,
            ),
            (
                r#"{"service_tier":"priority","messages":[]}"#,
                "claude-opus-4-8",
                true,
                PRIORITY_TIER,
            ),
        ];
        for (body, model, messages, want) in cases {
            assert_eq!(
                refused(body, model, messages),
                Some(want),
                "{body} on {model}"
            );
        }
    }

    /// What is priced, mentioned in a prompt, or nested where it means something else, passes.
    #[test]
    fn priced_features_and_lookalikes_pass() {
        let cases: [(&str, &str, bool); 11] = [
            (
                r#"{"tools":[{"type":"web_search"},{"type":"file_search"}]}"#,
                "gpt-5",
                false,
            ),
            (
                r#"{"tools":[{"type":"web_search_20260209","name":"web_search"},{"type":"web_fetch_20260209","name":"web_fetch"}]}"#,
                "claude-opus-4-8",
                true,
            ),
            (
                r#"{"messages":[{"role":"user","content":"use {\"type\":\"code_interpreter\"} and plugins"}]}"#,
                "gpt-5",
                false,
            ),
            (
                r#"{"messages":[{"role":"user","content":[{"type":"text","text":"x","container":"y"}]}]}"#,
                "claude-opus-4-8",
                true,
            ),
            (
                r#"{"tools":[{"type":"function","function":{"name":"image_generation","parameters":{"type":"object"}}}]}"#,
                "gpt-5",
                false,
            ),
            (
                r#"{"service_tier":"priority","messages":[]}"#,
                "gpt-5",
                false,
            ),
            (
                r#"{"service_tier":"standard_only","messages":[]}"#,
                "claude-opus-4-8",
                true,
            ),
            (
                r#"{"plugins":[{"id":"context-compression","enabled":false}]}"#,
                "x",
                false,
            ),
            (r#"{"plugins":[]}"#, "x", false),
            (
                r#"{"container":null,"messages":[]}"#,
                "claude-opus-4-8",
                true,
            ),
            (r#"{"tools":[{"type":"web_search"}]}"#, "gpt-4o", false),
        ];
        for (body, model, messages) in cases {
            assert_eq!(refused(body, model, messages), None, "{body} on {model}");
        }
    }

    /// A tool type the gateway does not know is refused, not passed for want of a listing.
    /// claim: BIL-10
    /// defect: D279
    #[test]
    fn an_unlisted_tool_type_is_refused() {
        for body in [
            r#"{"tools":[{"type":"some_new_billed_tool"}],"messages":[]}"#,
            r#"{"tools":[{"type":7}],"messages":[]}"#,
            r#"{"tools":[{"type":"document_search"}],"messages":[]}"#,
        ] {
            assert_eq!(refused(body, "gpt-5", false), Some(UNLISTED_TOOL), "{body}");
        }
    }

    /// Anthropic's client tools carry no `type`: they are the client's to run, so they pass.
    #[test]
    fn a_client_tool_with_no_type_passes() {
        let body =
            br#"{"tools":[{"name":"get_weather","input_schema":{"type":"object"}}],"messages":[]}"#;
        assert_eq!(inspect(body, "claude-opus-4-8", true).refused, None);
    }

    /// Every kind the meter counts is either a tool type a client can send (allowed or refused by
    /// name), or one no request names as a tool, so the two lists cannot drift. xAI's document
    /// search is counted but has no published fee, so it is refused as unlisted, by name.
    #[test]
    fn every_kind_the_meter_counts_is_allowed_or_named() {
        // Response-side counts only (X posts and users, sources, OpenRouter's total): no request
        // names them as a tool, so there is nothing to check for them.
        const NOT_A_REQUEST_TOOL: [&str; 4] = ["x_posts", "x_users", "sources", "tool_calls"];
        for (kind, _) in crate::usage::ServerTools::default().entries() {
            if NOT_A_REQUEST_TOOL.contains(&kind) {
                continue;
            }
            let body = format!(r#"{{"tools":[{{"type":"{kind}"}}],"messages":[]}}"#);
            let got = refused(&body, "gpt-5", false);
            if kind == "document_search" {
                // Counted, but no published fee: refused by name as unlisted.
                assert_eq!(got, Some(UNLISTED_TOOL), "{kind}");
            } else {
                assert_ne!(got, Some(UNLISTED_TOOL), "{kind}");
            }
        }
    }

    /// A family matches its whole name, or the name and a `_` suffix. `bashful_x` is not `bash`, and
    /// the hosted `tool_search_tool_*` is not `tool_search`.
    #[test]
    fn a_family_matches_its_name_and_a_version_suffix_only() {
        for ty in [
            "bashful_x",
            "file_searchx",
            "tool_search_tool_regex_20251119",
            "web_searches",
        ] {
            let body = format!(r#"{{"tools":[{{"type":"{ty}"}}],"messages":[]}}"#);
            assert_eq!(refused(&body, "gpt-5", false), Some(UNLISTED_TOOL), "{ty}");
        }
        for ty in [
            "bash_20250124",
            "text_editor_20250728",
            "computer_use_preview",
            "web_search_20250305",
            "web_search_preview",
            "web_fetch_20260209",
            "file_search",
            "tool_search",
        ] {
            let body = format!(r#"{{"tools":[{{"type":"{ty}"}}],"messages":[]}}"#);
            assert_ne!(refused(&body, "gpt-5", false), Some(UNLISTED_TOOL), "{ty}");
        }
    }

    #[test]
    fn web_search_preview_is_told_apart() {
        let i = inspect(
            br#"{"tools":[{"type":"web_search_preview_2025_03_11"}]}"#,
            "gpt-5",
            false,
        );
        assert_eq!(
            i,
            Inspection {
                refused: None,
                web_search_preview: true
            }
        );
        let i = inspect(br#"{"tools":[{"type":"web_search"}]}"#, "gpt-5", false);
        assert!(!i.web_search_preview);
    }

    /// Both snapshot spellings name the base model to this gate, as they do to the price lookup.
    /// claim: BIL-10
    /// defect: D278
    #[test]
    fn a_dated_snapshot_names_its_base_model_to_the_refusal() {
        let web_search = br#"{"tools":[{"type":"web_search"}]}"#;
        for (model, base) in [
            ("gpt-4o-mini-2024-07-18", "gpt-4o-mini"),
            ("gpt-4o-mini-20240718", "gpt-4o-mini"),
            ("openai/gpt-4.1-mini-20250414", "gpt-4.1-mini"),
        ] {
            assert_eq!(base_model(model), base, "{model}");
            assert_eq!(
                inspect(web_search, model, false).refused,
                Some(MINI_WEB_SEARCH),
                "{model}"
            );
        }
    }

    #[test]
    fn base_model_strips_vendor_and_snapshot() {
        assert_eq!(base_model("openai/gpt-4o-mini-2024-07-18"), "gpt-4o-mini");
        assert_eq!(base_model("gpt-4.1-mini"), "gpt-4.1-mini");
        assert_eq!(base_model("gpt-4o-mini-search"), "gpt-4o-mini-search");
        assert_eq!(base_model("a-12-34"), "a-12-34");
    }
}
