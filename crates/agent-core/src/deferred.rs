//! Deferred tools: callable by name, but sent to the model only once a `tool_search` result names
//! them.
//!
//! The registry keeps deferred tools out of the advertised list
//! ([`crate::ToolRegistry::register_deferred`]); every request carries their definitions separately
//! as [`ModelRequest::deferred_tools`]. What loads one is the transcript itself: **a successful
//! [`TOOL_SEARCH`] result loads every deferred tool it names.** The model read those names in the
//! result, so the history is the one record of what is loaded. It survives a resume, a `serve`
//! reattach and a switch of dialect mid-session with nothing extra persisted, and a compaction that
//! summarizes the result away unloads the tool again (the model searches again if it still needs it).
//!
//! Each dialect then puts a loaded tool in front of the model the cheapest way its vendor allows:
//!
//! - **Anthropic, Claude 4.5 and later** ([`native`]): every deferred definition rides in `tools`
//!   with `defer_loading: true`, which keeps it out of the model's context and out of the billed
//!   input, and the `tool_search` result becomes `tool_reference` blocks the API expands in place.
//!   `tools` never changes, so the prompt cache survives a load.
//! - **OpenAI Responses, GPT-5.4 and later**: deferred definitions are not sent up front; an
//!   `additional_tools` item right after the `tool_search` output carries the loaded ones. That item
//!   is part of the conversation, not of the prefix, so the cache survives a load here too.
//! - **Everything else** ([`with_loaded`]): the loaded definitions are appended to `tools` from the
//!   next request on. Portable, but `tools` sits at the front of the prompt, so each load costs one
//!   cache rewrite of the whole prefix.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use crate::dialect::Dialect;
use crate::message::{ContentBlock, Role, ToolDef};
use crate::transport::ModelRequest;

/// The name of the tool whose results load deferred tools.
pub const TOOL_SEARCH: &str = "tool_search";

/// Which deferred tools each `tool_search` result loaded, in transcript order. A tool loads once:
/// a later result that names it again adds nothing.
#[derive(Debug, Default)]
pub(crate) struct Loads<'a> {
    by_result: Vec<(&'a str, Vec<&'a ToolDef>)>,
}

impl<'a> Loads<'a> {
    pub(crate) fn is_empty(&self) -> bool {
        self.by_result.is_empty()
    }

    /// The tools one `tool_search` result loaded, by its `tool_use_id`.
    pub(crate) fn for_result(&self, tool_use_id: &str) -> Option<&[&'a ToolDef]> {
        self.by_result
            .iter()
            .find(|(id, _)| *id == tool_use_id)
            .map(|(_, defs)| defs.as_slice())
    }

    /// Every loaded tool, in load order.
    pub(crate) fn all(&self) -> impl Iterator<Item = &'a ToolDef> + '_ {
        self.by_result
            .iter()
            .flat_map(|(_, defs)| defs.iter().copied())
    }
}

/// Scan the transcript for `tool_search` results and the deferred tools each names.
///
/// Free when nothing is deferred, and otherwise one pass over the history plus a token scan of the
/// (short) search results only.
pub(crate) fn loads(req: &ModelRequest) -> Loads<'_> {
    if req.deferred_tools.is_empty() {
        return Loads::default();
    }
    let mut search_ids: HashSet<&str> = HashSet::new();
    for m in req.messages.iter().filter(|m| m.role == Role::Assistant) {
        for b in &m.content {
            if let ContentBlock::ToolUse { id, name, .. } = b
                && name == TOOL_SEARCH
            {
                search_ids.insert(id);
            }
        }
    }
    if search_ids.is_empty() {
        return Loads::default();
    }
    let by_name: HashMap<&str, &ToolDef> = req
        .deferred_tools
        .iter()
        .map(|d| (d.name.as_str(), d))
        .collect();
    let mut seen: HashSet<&str> = HashSet::new();
    let mut by_result = Vec::new();
    for m in req.messages.iter().filter(|m| m.role == Role::User) {
        for b in &m.content {
            let ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error: false,
                ..
            } = b
            else {
                continue;
            };
            if !search_ids.contains(tool_use_id.as_str()) {
                continue;
            }
            let defs: Vec<&ToolDef> = name_tokens(content)
                .filter_map(|tok| by_name.get(tok).copied())
                .filter(|d| seen.insert(d.name.as_str()))
                .collect();
            if !defs.is_empty() {
                by_result.push((tool_use_id.as_str(), defs));
            }
        }
    }
    Loads { by_result }
}

/// Maximal runs of tool-name characters (`[A-Za-z0-9_-]`), so `` `mcp__a__b`. `` yields
/// `mcp__a__b` and never a prefix of a longer name.
fn name_tokens(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '-'))
        .filter(|t| !t.is_empty())
}

/// `req.tools` plus every loaded deferred tool — the portable fallback. Borrowed, at no cost, when
/// nothing is loaded.
pub(crate) fn with_loaded<'a>(req: &'a ModelRequest, loads: &Loads<'a>) -> Cow<'a, [ToolDef]> {
    if loads.is_empty() {
        return Cow::Borrowed(&req.tools);
    }
    let mut tools = req.tools.to_vec();
    for def in loads.all() {
        if !tools.iter().any(|t| t.name == def.name) {
            tools.push(def.clone());
        }
    }
    Cow::Owned(tools)
}

/// Whether `dialect` loads deferred tools natively for this request, keeping the prompt cache intact
/// — see the module doc. Only a vendor's own route qualifies: an aggregator or a re-hosted endpoint
/// serving the same model id may not accept the fields, and the fallback is always correct.
pub(crate) fn native(req: &ModelRequest, dialect: Dialect) -> bool {
    use crate::models::AggregatorHost as Host;
    if req.deferred_tools.is_empty() {
        return false;
    }
    // `host` is `None` through the gateway (which relays to the vendor) and names the provider on a
    // direct route.
    match dialect {
        Dialect::Anthropic => {
            matches!(req.host, None | Some(Host::Anthropic | Host::Bedrock))
                && claude_supports_tool_search(&req.model)
        }
        Dialect::OpenAiResponses => {
            matches!(req.host, None | Some(Host::OpenAi))
                && !req.is_codex
                && !req.is_azure
                && !req.is_copilot
                && gpt_supports_tool_search(&req.model)
        }
        Dialect::OpenAi => false,
    }
}

/// Claude Opus/Sonnet/Haiku 4.5 and later, and every Fable/Mythos — the models Anthropic's tool
/// search page lists. Accepts the Bedrock (`us.anthropic.claude-…`) and Vertex (`…@date`) spellings,
/// whose Messages endpoints honour `defer_loading` and `tool_reference` too.
fn claude_supports_tool_search(model: &str) -> bool {
    let model = model.rsplit_once("anthropic.").map_or(model, |(_, m)| m);
    let Some(rest) = model.strip_prefix("claude-") else {
        return false;
    };
    let mut parts = rest.split(['-', '@']);
    let family = parts.next().unwrap_or_default();
    let major: u32 = parts.next().and_then(|p| p.parse().ok()).unwrap_or(0);
    // A dated snapshot (`claude-sonnet-4-20250514`) has no minor version: its next part is the date.
    let minor: u32 = parts
        .next()
        .filter(|p| p.len() <= 2)
        .and_then(|p| p.parse().ok())
        .unwrap_or(0);
    match family {
        "opus" | "sonnet" | "haiku" => (major, minor) >= (4, 5),
        "fable" | "mythos" => true,
        _ => false,
    }
}

/// GPT-5.4 and later: OpenAI's tool search (and the `additional_tools` item) is documented for
/// those families only.
fn gpt_supports_tool_search(model: &str) -> bool {
    ["gpt-5.4", "gpt-5.5", "gpt-5.6", "gpt-6"]
        .iter()
        .any(|p| model.starts_with(p))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Message;
    use serde_json::json;

    fn def(name: &str) -> ToolDef {
        ToolDef {
            name: name.into(),
            description: format!("{name} does a thing"),
            input_schema: json!({ "type": "object" }),
        }
    }

    fn search_turn(id: &str, result: &str, is_error: bool) -> Vec<Message> {
        vec![
            Message::assistant(vec![ContentBlock::tool_use(
                id,
                TOOL_SEARCH,
                json!({ "query": "x" }),
            )]),
            Message::tool_results(vec![ContentBlock::ToolResult {
                tool_use_id: id.into(),
                content: result.into(),
                is_error,
                images: Vec::new(),
            }]),
        ]
    }

    fn req(messages: Vec<Message>) -> ModelRequest {
        ModelRequest::new("claude-sonnet-5-5", messages, 256)
            .with_tools(vec![def(TOOL_SEARCH)])
            .with_deferred_tools(vec![
                def("mcp__gh__list"),
                def("mcp__gh__list_issues"),
                def("mcp__web__open"),
            ])
    }

    #[test]
    fn a_search_result_loads_exactly_the_deferred_tools_it_names() {
        let mut msgs = vec![Message::user("go")];
        msgs.extend(search_turn(
            "t1",
            "Loaded 1 tool:\n- `mcp__gh__list_issues` — lists issues (see also gh)",
            false,
        ));
        let r = req(msgs);
        let loads = loads(&r);
        let names: Vec<&str> = loads.all().map(|d| d.name.as_str()).collect();
        // Not `mcp__gh__list`, a prefix of the name the result gave.
        assert_eq!(names, vec!["mcp__gh__list_issues"]);
        assert_eq!(loads.for_result("t1").map(<[_]>::len), Some(1));
    }

    #[test]
    fn an_error_or_another_tools_result_loads_nothing() {
        let mut msgs = search_turn("t1", "mcp__web__open", true);
        msgs.push(Message::assistant(vec![ContentBlock::tool_use(
            "t2",
            "bash",
            json!({}),
        )]));
        msgs.push(Message::tool_results(vec![ContentBlock::ToolResult {
            tool_use_id: "t2".into(),
            content: "mcp__web__open".into(),
            is_error: false,
            images: Vec::new(),
        }]));
        assert!(loads(&req(msgs)).is_empty());
    }

    #[test]
    fn a_tool_loads_once_and_the_fallback_appends_it_after_the_advertised_tools() {
        let mut msgs = search_turn("t1", "mcp__web__open", false);
        msgs.extend(search_turn("t2", "mcp__web__open mcp__gh__list", false));
        let r = req(msgs);
        let loads = loads(&r);
        assert_eq!(loads.for_result("t2").map(<[_]>::len), Some(1));
        let tools: Vec<String> = with_loaded(&r, &loads)
            .iter()
            .map(|d| d.name.clone())
            .collect();
        assert_eq!(tools, vec![TOOL_SEARCH, "mcp__web__open", "mcp__gh__list"]);
    }

    #[test]
    fn native_loading_is_for_the_vendors_own_recent_models() {
        for (model, dialect, want) in [
            ("claude-sonnet-5-5", Dialect::Anthropic, true),
            ("claude-opus-4-5-20251101", Dialect::Anthropic, true),
            ("claude-haiku-4-5", Dialect::Anthropic, true),
            (
                "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                Dialect::Anthropic,
                true,
            ),
            ("claude-fable-5-1", Dialect::Anthropic, true),
            ("claude-sonnet-4-20250514", Dialect::Anthropic, false),
            ("claude-opus-4-1", Dialect::Anthropic, false),
            ("claude-3-7-sonnet-latest", Dialect::Anthropic, false),
            ("gpt-6.1-sol", Dialect::OpenAiResponses, true),
            ("gpt-5.4-mini", Dialect::OpenAiResponses, true),
            ("gpt-5.2", Dialect::OpenAiResponses, false),
            ("gpt-6.1-sol", Dialect::OpenAi, false),
        ] {
            let r = ModelRequest::new(model, Vec::new(), 16).with_deferred_tools(vec![def("x")]);
            assert_eq!(native(&r, dialect), want, "{model} {dialect:?}");
        }
        // Nothing deferred: nothing to do natively either.
        let r = ModelRequest::new("claude-sonnet-5-5", Vec::new(), 16);
        assert!(!native(&r, Dialect::Anthropic));
        // The vendor's own direct route qualifies; an aggregator serving the same id does not.
        let mut r = ModelRequest::new("claude-sonnet-5-5", Vec::new(), 16)
            .with_deferred_tools(vec![def("x")]);
        r.host = Some(crate::models::AggregatorHost::Anthropic);
        assert!(native(&r, Dialect::Anthropic));
        r.host = Some(crate::models::AggregatorHost::OpenCodeZen);
        assert!(!native(&r, Dialect::Anthropic));
    }
}
