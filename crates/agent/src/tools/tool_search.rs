//! `tool_search` — load deferred MCP tools on demand, with no interpreter.
//!
//! By default MCP tools are not sent to the model up front. The registry keeps them as *deferred*
//! tools ([`ToolRegistry::register_deferred`]): callable by name, absent from the advertised list.
//! What the model sees instead is this one tool, whose description is a compact catalog — server
//! names and tool names, byte-stable across runs. A search returns the best matches and **loads
//! every tool its result names**: the dialect puts their full definitions in front of the model from
//! the next step on (`agent_core::deferred` — natively on Anthropic and OpenAI, so the prompt cache
//! survives the load; by appending to `tools` elsewhere).
//!
//! Stateless, like [`super::todo`]: what is loaded is read back out of the transcript, so a
//! registry rebuilt on `set_model` loses nothing and a resumed session keeps its loaded tools.
//!
//! [`McpLoading`] picks between this, eager loading (every MCP schema in every request) and, by
//! default, whichever of the two is cheaper for the connected catalog's size. Code Mode
//! (`--code-mode`) takes precedence over both: it hides the same catalog behind `execute` instead.

use std::collections::BTreeMap;
use std::sync::Arc;

use agent_core::tool::Tool;
use agent_core::{ToolError, ToolOutput, ToolRegistry};
use async_trait::async_trait;
use serde_json::{Value, json};

pub use agent_core::deferred::TOOL_SEARCH as NAME;

/// Results per search unless the model asks for more.
const DEFAULT_LIMIT: usize = 5;
/// The most one search loads: past this a query is too vague to be worth the context.
const MAX_LIMIT: usize = 20;
/// Bytes of catalog lines in the description. Past it a server collapses to its tool count — the
/// tool names are still one search away.
const CATALOG_BUDGET: usize = 4_000;
/// Characters of each loaded tool's description repeated in the result text.
const RESULT_DESCRIPTION_CHARS: usize = 160;

/// How MCP tools reach the model — `--mcp-tool-loading` / `AI_AGENT_MCP_TOOL_LOADING`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum McpLoading {
    /// Search when the catalog's schemas exceed [`AUTO_SEARCH_MIN_BYTES`], eager below it.
    #[default]
    Auto,
    /// Always defer MCP tools behind `tool_search`.
    Search,
    /// Always send every MCP tool's full schema in every request.
    Eager,
}

/// Where [`McpLoading::Auto`] switches to search: the summed size of the MCP tools' serialized
/// definitions. Below it a catalog costs less than one search round trip would (see
/// `crates/agent/ARCHITECTURE.md`, "Deferred MCP tools", for the measurement).
pub const AUTO_SEARCH_MIN_BYTES: usize = 8_000;

impl std::str::FromStr for McpLoading {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "auto" => Ok(Self::Auto),
            "search" => Ok(Self::Search),
            "eager" => Ok(Self::Eager),
            other => Err(format!(
                "invalid MCP tool loading {other:?}; expected one of auto/search/eager"
            )),
        }
    }
}

impl McpLoading {
    /// Whether this catalog's tools are deferred behind `tool_search`.
    pub fn defers(self, mcp_tools: &[Arc<dyn Tool>]) -> bool {
        match self {
            Self::Search => true,
            Self::Eager => false,
            Self::Auto => {
                let bytes: usize = mcp_tools
                    .iter()
                    .filter(|t| deferrable(t.name()))
                    .map(|t| serde_json::to_vec(&t.definition()).map_or(0, |b| b.len()))
                    .sum();
                bytes > AUTO_SEARCH_MIN_BYTES
            }
        }
    }
}

/// Every MCP tool but a server's skill loader (`mcp__<server>__skill__read`): `<available_skills>`
/// in the system prompt tells the model to call that one by name, so it stays advertised.
pub fn deferrable(name: &str) -> bool {
    !name.ends_with("__skill__read")
}

/// Register `mcp_tools` the way `loading` says — called by `default_registry_with_config` after
/// every built-in.
pub fn register_mcp(reg: &mut ToolRegistry, mcp_tools: &[Arc<dyn Tool>], loading: McpLoading) {
    if !loading.defers(mcp_tools) {
        for tool in mcp_tools {
            reg.register(tool.clone());
        }
        return;
    }
    for tool in mcp_tools {
        if deferrable(tool.name()) {
            reg.register_deferred(tool.clone());
        } else {
            reg.register(tool.clone());
        }
    }
    let deferred = reg.deferred_tools();
    if !deferred.is_empty() {
        reg.register(Arc::new(ToolSearch::new(&deferred)));
    }
}

/// Make `tool_search` agree with the deferred tools the registry still holds, after a filter
/// removed some (`apply_filter`): rebuilt over what is left, removed when nothing is. When
/// `tool_search` itself was filtered out, the deferred tools that survived — ones an allow-list
/// named explicitly — are advertised instead, since nothing could load them any more.
pub fn refresh(reg: &mut ToolRegistry) {
    let deferred = reg.deferred_tools();
    match (reg.get(NAME).is_some(), deferred.is_empty()) {
        (true, true) => {
            reg.retain(|n| n != NAME);
        }
        (true, false) => {
            reg.register(Arc::new(ToolSearch::new(&deferred)));
        }
        (false, false) => {
            for tool in deferred {
                reg.register(tool);
            }
        }
        (false, true) => {}
    }
}

const LEAD: &str = "Find and load MCP tools that are not loaded yet.";

/// One deferred tool, as the search sees it.
struct Entry {
    name: String,
    server: String,
    /// The name without its `mcp__<server>__` prefix.
    short: String,
    description: String,
    signature: String,
}

/// The model-facing tool.
pub struct ToolSearch {
    entries: Vec<Entry>,
    index: Index,
    description: String,
}

impl ToolSearch {
    /// Build over the deferred tools, which arrive sorted by name.
    pub fn new(deferred: &[Arc<dyn Tool>]) -> Self {
        let entries: Vec<Entry> = deferred
            .iter()
            .map(|t| {
                let name = t.name().to_string();
                let server = super::mcp::server_name_from_registered(&name)
                    .unwrap_or_default()
                    .to_string();
                let short = name
                    .strip_prefix(&format!("mcp__{server}__"))
                    .unwrap_or(&name)
                    .to_string();
                Entry {
                    signature: signature(&t.input_schema()),
                    description: t.description().to_string(),
                    name,
                    server,
                    short,
                }
            })
            .collect();
        let description = describe(&entries);
        let index = Index::new(
            deferred
                .iter()
                .map(|t| (t.name(), t.description(), t.input_schema())),
        );
        Self {
            entries,
            index,
            description,
        }
    }

    fn search(&self, query: &str, limit: usize) -> String {
        if Index::terms(query).is_empty() {
            return format!("Empty query. {}", self.servers_line());
        }
        let mut scored: Vec<&Entry> = self
            .index
            .rank(query)
            .into_iter()
            .map(|i| &self.entries[i])
            .collect();
        if scored.is_empty() {
            return format!(
                "No deferred tool matches {query:?}. {}",
                self.servers_line()
            );
        }
        let more = scored.len().saturating_sub(limit);
        scored.truncate(limit);
        let mut out = format!(
            "Loaded {} tool{} — call {} directly from now on:\n",
            scored.len(),
            if scored.len() == 1 { "" } else { "s" },
            if scored.len() == 1 { "it" } else { "them" },
        );
        for e in &scored {
            out.push_str(&format!(
                "- {}({}) — {}\n",
                e.name,
                e.signature,
                truncate(&e.description, RESULT_DESCRIPTION_CHARS)
            ));
        }
        if more > 0 {
            out.push_str(&format!(
                "{more} more matched; search again with a narrower query or a higher `limit`.\n"
            ));
        }
        out
    }

    fn servers_line(&self) -> String {
        let mut servers: Vec<&str> = self.entries.iter().map(|e| e.server.as_str()).collect();
        servers.dedup();
        format!("Servers: {}.", servers.join(", "))
    }
}

/// Keyword search over a tool catalog — shared by `tool_search` and Code Mode's `tools.search`.
///
/// Names and text are cut into normalized words: split on anything not a letter or digit and at
/// camelCase humps (`get_file_contents`, `get-file-contents`, `getFileContents` and `get file
/// contents` are all `get file content`), lowercased, plurals folded (`issues` → `issue`,
/// `entities` → `entity`). A query word scores most when it is a word of the tool's own name,
/// less as the start of one (`nav` → `navigate`) or as the server, least when it appears only in
/// the description or a parameter name. Only the tools matching the most query words come back —
/// "github add" is github's `add`, not every github tool and every `add` — best first, ties in name
/// order, so the same query always returns the same tools. An exact registered name
/// (`mcp__github__get_issue`, or `github__get_issue`) wins outright. No embeddings, no state.
pub struct Index {
    docs: Vec<Doc>,
}

struct Doc {
    /// `mcp__github__get_issue` and `github__get_issue`, lowercased.
    exact: [String; 2],
    server: String,
    name_words: Vec<String>,
    text_words: std::collections::HashSet<String>,
}

/// Words too common to tell tools apart.
const STOPWORDS: &[&str] = &[
    "a", "an", "and", "the", "to", "of", "for", "in", "on", "by", "with", "from", "or", "is", "it",
    "tool", "tools", "mcp",
];

impl Index {
    /// Index `(registered name, description, input schema)` triples, in the order results refer to.
    pub fn new<'a>(tools: impl IntoIterator<Item = (&'a str, &'a str, Value)>) -> Self {
        let docs = tools
            .into_iter()
            .map(|(name, description, schema)| {
                let server = super::mcp::server_name_from_registered(name).unwrap_or_default();
                let short = name
                    .strip_prefix(&format!("mcp__{server}__"))
                    .unwrap_or(name);
                let mut text_words: std::collections::HashSet<String> =
                    words(description).collect();
                if let Some(props) = schema.get("properties").and_then(Value::as_object) {
                    text_words.extend(props.keys().flat_map(|k| words(k)));
                }
                Doc {
                    exact: [
                        name.to_lowercase(),
                        format!("{server}__{short}").to_lowercase(),
                    ],
                    server: server.to_lowercase(),
                    name_words: words(short).collect(),
                    text_words,
                }
            })
            .collect();
        Self { docs }
    }

    /// The query's normalized words, stopwords dropped, each once.
    fn terms(query: &str) -> Vec<String> {
        let mut terms: Vec<String> = Vec::new();
        for w in words(query) {
            if !STOPWORDS.contains(&w.as_str()) && !terms.contains(&w) {
                terms.push(w);
            }
        }
        terms
    }

    /// Indices of the matching tools, best first (see the type's doc comment).
    pub fn rank(&self, query: &str) -> Vec<usize> {
        let raw: Vec<String> = query
            .split(|c: char| c == ',' || c.is_whitespace())
            .map(|t| {
                t.trim_matches(|c: char| !(c.is_alphanumeric() || c == '_' || c == '-'))
                    .to_lowercase()
            })
            .filter(|t| !t.is_empty())
            .collect();
        let exact: Vec<usize> = (0..self.docs.len())
            .filter(|&i| raw.iter().any(|r| self.docs[i].exact.contains(r)))
            .collect();
        if !exact.is_empty() {
            return exact;
        }
        let terms = Self::terms(query);
        let mut scored: Vec<(usize, u32, usize)> = self
            .docs
            .iter()
            .enumerate()
            .filter_map(|(i, d)| {
                let (mut matched, mut points) = (0, 0);
                for term in &terms {
                    let p = if d.name_words.contains(term) {
                        6
                    } else if term.len() >= 3
                        && d.name_words.iter().any(|w| w.starts_with(term.as_str()))
                    {
                        3
                    } else {
                        0
                    } + if d.server == *term { 3 } else { 0 }
                        + u32::from(d.text_words.contains(term));
                    if p > 0 {
                        matched += 1;
                        points += p;
                    }
                }
                (matched > 0).then_some((matched, points, i))
            })
            .collect();
        let best = scored.iter().map(|s| s.0).max().unwrap_or(0);
        scored.retain(|s| s.0 == best);
        scored.sort_by(|a, b| {
            b.1.cmp(&a.1)
                .then_with(|| self.docs[a.2].exact[0].cmp(&self.docs[b.2].exact[0]))
        });
        scored.into_iter().map(|s| s.2).collect()
    }
}

/// Normalized words of `s` — see [`Index`].
fn words(s: &str) -> impl Iterator<Item = String> + '_ {
    s.split(|c: char| !c.is_alphanumeric())
        .flat_map(split_camel)
        .map(|w| stem(&w.to_lowercase()))
        .filter(|w| !w.is_empty())
}

/// `getFileContents` → `get`, `File`, `Contents`; `HTTPServer` → `HTTP`, `Server`.
fn split_camel(s: &str) -> Vec<&str> {
    let chars: Vec<(usize, char)> = s.char_indices().collect();
    let mut out = Vec::new();
    let mut start = 0;
    for k in 1..chars.len() {
        let (i, c) = chars[k];
        let prev = chars[k - 1].1;
        let next_lower = chars.get(k + 1).is_some_and(|(_, n)| n.is_lowercase());
        if c.is_uppercase()
            && (prev.is_lowercase() || prev.is_ascii_digit() || (prev.is_uppercase() && next_lower))
        {
            out.push(&s[start..i]);
            start = i;
        }
    }
    out.push(&s[start..]);
    out
}

/// Fold an English plural to its singular, crudely but identically on both sides of a match.
fn stem(w: &str) -> String {
    if w.len() > 4 && w.ends_with("ies") {
        format!("{}y", &w[..w.len() - 3])
    } else if w.len() > 3 && w.ends_with('s') && !w.ends_with("ss") {
        w[..w.len() - 1].to_string()
    } else {
        w.to_string()
    }
}

/// `owner, repo, page?` — required parameters bare, optional ones marked.
fn signature(schema: &Value) -> String {
    let Some(props) = schema.get("properties").and_then(Value::as_object) else {
        return String::new();
    };
    let required: Vec<&str> = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|r| r.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let mut parts: Vec<String> = props
        .keys()
        .map(|k| {
            if required.contains(&k.as_str()) {
                k.clone()
            } else {
                format!("{k}?")
            }
        })
        .collect();
    parts.sort();
    parts.join(", ")
}

/// The description: what the tool does, then the catalog — one line per server listing its tool
/// names, within [`CATALOG_BUDGET`].
fn describe(entries: &[Entry]) -> String {
    let mut by_server: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for e in entries {
        by_server.entry(&e.server).or_default().push(&e.short);
    }
    let mut out = format!(
        "{LEAD} Their names are listed below; their parameters are not, so search before calling \
         one. Pass keywords, a server name, or exact tool names (comma-separated). Every tool a \
         result lists is loaded and callable directly by its full `mcp__<server>__<tool>` name from \
         then on; it stays loaded, so never search for it again.\n\nDeferred MCP tools by server:"
    );
    let mut used = 0;
    for (server, names) in &by_server {
        let line = format!("\n- {server}: {}", names.join(", "));
        let line = if used + line.len() <= CATALOG_BUDGET {
            line
        } else {
            format!("\n- {server}: {} tools", names.len())
        };
        used += line.len();
        out.push_str(&line);
    }
    out
}

fn truncate(s: &str, max_chars: usize) -> String {
    let flat: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
    match flat.char_indices().nth(max_chars) {
        Some((cut, _)) => format!("{}…", &flat[..cut]),
        None => flat,
    }
}

#[async_trait]
impl Tool for ToolSearch {
    fn name(&self) -> &str {
        NAME
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords (\"create issue\"), a server name (\"github\"), or exact tool names (\"mcp__github__get_issue, mcp__github__list_issues\").",
                },
                "limit": {
                    "type": "integer",
                    "description": format!("Most tools to load (default {DEFAULT_LIMIT}, at most {MAX_LIMIT})."),
                },
            },
            "required": ["query"],
        })
    }

    async fn run(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let query = input
            .get("query")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidInput("missing `query` string".into()))?;
        let limit = input
            .get("limit")
            .and_then(Value::as_u64)
            .map_or(DEFAULT_LIMIT, |n| usize::try_from(n).unwrap_or(MAX_LIMIT))
            .clamp(1, MAX_LIMIT);
        Ok(ToolOutput::text(self.search(query, limit)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake(&'static str, &'static str, Value);

    #[async_trait]
    impl Tool for Fake {
        fn name(&self) -> &str {
            self.0
        }
        fn description(&self) -> &str {
            self.1
        }
        fn input_schema(&self) -> Value {
            self.2.clone()
        }
        async fn run(&self, _: Value) -> Result<ToolOutput, ToolError> {
            Ok("ran".into())
        }
    }

    fn catalog() -> Vec<Arc<dyn Tool>> {
        let schema = |req: &[&str], opt: &[&str]| {
            let mut props = serde_json::Map::new();
            for p in req.iter().chain(opt) {
                props.insert((*p).into(), json!({ "type": "string" }));
            }
            json!({ "type": "object", "properties": props, "required": req })
        };
        vec![
            Arc::new(Fake(
                "mcp__github__get_issue",
                "Get details of a specific issue in a GitHub repository.",
                schema(&["owner", "repo", "issue_number"], &[]),
            )),
            Arc::new(Fake(
                "mcp__github__list_issues",
                "List issues in a GitHub repository with filtering options.",
                schema(&["owner", "repo"], &["state", "page"]),
            )),
            Arc::new(Fake(
                "mcp__github__skill__read",
                "Load a skill.",
                schema(&["uri"], &[]),
            )),
            Arc::new(Fake(
                "mcp__playwright__browser_navigate",
                "Navigate to a URL.",
                schema(&["url"], &[]),
            )),
            Arc::new(Fake(
                "mcp__playwright__browser_snapshot",
                "Capture an accessibility snapshot of the current page.",
                schema(&[], &[]),
            )),
        ]
    }

    fn registry(loading: McpLoading) -> ToolRegistry {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(Fake("read", "read a file", json!({}))));
        register_mcp(&mut reg, &catalog(), loading);
        reg
    }

    #[test]
    fn search_mode_defers_mcp_tools_behind_a_byte_stable_catalog() {
        let reg = registry(McpLoading::Search);
        let advertised: Vec<String> = reg.definitions().into_iter().map(|d| d.name).collect();
        // The skill loader stays advertised: `<available_skills>` names it.
        assert_eq!(advertised, vec!["mcp__github__skill__read", "read", NAME]);
        assert_eq!(reg.deferred_tools().len(), 4);
        let desc = reg.get(NAME).unwrap().description().to_string();
        assert!(
            desc.contains("\n- github: get_issue, list_issues"),
            "{desc}"
        );
        assert!(desc.contains("\n- playwright: browser_navigate, browser_snapshot"));
        // Rebuilding yields the same bytes — the description sits in the cached prefix.
        assert_eq!(
            registry(McpLoading::Search)
                .get(NAME)
                .unwrap()
                .description(),
            desc
        );
    }

    #[test]
    fn eager_mode_and_a_small_auto_catalog_advertise_everything() {
        for loading in [McpLoading::Eager, McpLoading::Auto] {
            let reg = registry(loading);
            assert!(reg.get(NAME).is_none());
            assert!(reg.deferred_tools().is_empty());
            assert_eq!(reg.definitions().len(), 6);
        }
    }

    #[tokio::test]
    async fn a_search_names_only_the_tools_it_loads() {
        let search = ToolSearch::new(&registry(McpLoading::Search).deferred_tools());
        let out = search
            .run(json!({ "query": "issue", "limit": 1 }))
            .await
            .unwrap()
            .text;
        assert!(out.starts_with("Loaded 1 tool"), "{out}");
        assert!(out.contains("mcp__github__get_issue(issue_number, owner, repo)"));
        assert!(!out.contains("list_issues"), "{out}");
        assert!(out.contains("1 more matched"));
        // An exact name beats keyword matches; a server name loads that server's tools.
        let out = search
            .run(json!({ "query": "mcp__github__list_issues" }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("mcp__github__list_issues(") && !out.contains("get_issue"));
        let out = search
            .run(json!({ "query": "playwright" }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("browser_navigate") && out.contains("browser_snapshot"));
        let out = search.run(json!({ "query": "zzz" })).await.unwrap().text;
        assert!(out.starts_with("No deferred tool matches"), "{out}");
        assert!(out.contains("Servers: github, playwright."));
    }

    #[test]
    fn refresh_follows_a_filter() {
        // Excluding tools rebuilds the catalog over what is left.
        let mut reg = registry(McpLoading::Search);
        reg.retain(|n| !n.starts_with("mcp__playwright"));
        refresh(&mut reg);
        assert!(!reg.get(NAME).unwrap().description().contains("playwright"));
        // Excluding `tool_search` advertises the deferred tools an allow-list kept.
        let mut reg = registry(McpLoading::Search);
        reg.retain(|n| n != NAME && n != "mcp__github__list_issues");
        refresh(&mut reg);
        assert!(reg.deferred_tools().is_empty());
        assert!(
            reg.definitions()
                .iter()
                .any(|d| d.name == "mcp__github__get_issue")
        );
        // Nothing deferred left: no empty `tool_search`.
        let mut reg = registry(McpLoading::Search);
        reg.retain(|n| !n.starts_with("mcp__") || n.ends_with("skill__read"));
        refresh(&mut reg);
        assert!(reg.get(NAME).is_none());
    }

    /// The real catalogs the measurements use (`eval/mcp_catalogs/`), registered as the agent would.
    fn real_index() -> (Index, Vec<String>) {
        let mut tools: Vec<(String, String, Value)> = Vec::new();
        for (server, json) in [
            (
                "github",
                include_str!("../../eval/mcp_catalogs/github.json"),
            ),
            (
                "playwright",
                include_str!("../../eval/mcp_catalogs/playwright.json"),
            ),
            (
                "filesystem",
                include_str!("../../eval/mcp_catalogs/filesystem.json"),
            ),
            (
                "memory",
                include_str!("../../eval/mcp_catalogs/memory.json"),
            ),
            (
                "everything",
                include_str!("../../eval/mcp_catalogs/everything.json"),
            ),
            (
                "notion",
                include_str!("../../eval/mcp_catalogs/notion.json"),
            ),
        ] {
            for t in serde_json::from_str::<Vec<Value>>(json).unwrap() {
                tools.push((
                    format!("mcp__{server}__{}", t["name"].as_str().unwrap()),
                    t["description"].as_str().unwrap_or_default().to_string(),
                    t["inputSchema"].clone(),
                ));
            }
        }
        let names = tools.iter().map(|t| t.0.clone()).collect();
        let index = Index::new(
            tools
                .iter()
                .map(|(n, d, s)| (n.as_str(), d.as_str(), s.clone())),
        );
        (index, names)
    }

    #[test]
    fn search_matches_how_real_tool_names_are_spelled() {
        let (index, names) = real_index();
        let top = |q: &str| {
            index
                .rank(q)
                .first()
                .map(|&i| names[i].clone())
                .unwrap_or_else(|| panic!("no match for {q:?}"))
        };
        for (query, want) in [
            // snake_case, from spaced words, from camelCase, from kebab-case
            ("get file contents", "mcp__github__get_file_contents"),
            ("getFileContents", "mcp__github__get_file_contents"),
            ("get-file-contents", "mcp__github__get_file_contents"),
            ("click", "mcp__playwright__browser_click"),
            ("list issues", "mcp__github__list_issues"),
            // singular for a plural name, and the reverse
            ("list issue", "mcp__github__list_issues"),
            ("create entity", "mcp__memory__create_entities"),
            ("get pull requests", "mcp__github__get_pull_request"),
            // kebab-case names, stopwords in the name or the query
            ("tiny image", "mcp__everything__get-tiny-image"),
            ("retrieve a page", "mcp__notion__API-retrieve-a-page"),
            (
                "directory with sizes",
                "mcp__filesystem__list_directory_with_sizes",
            ),
            // words only the description has
            (
                "accessibility snapshot",
                "mcp__playwright__browser_snapshot",
            ),
            (
                "knowledge graph observations",
                "mcp__memory__add_observations",
            ),
            // a prefix of a name word
            ("navig", "mcp__playwright__browser_navigate"),
            // an exact registered name
            ("mcp__github__search_code", "mcp__github__search_code"),
        ] {
            assert_eq!(top(query), want, "{query:?}");
        }
        // Only the tools matching the most query words: not every `get_*` and every `*_file*`.
        assert!(index.rank("get file contents").len() <= 3);
        assert!(index.rank("zzz qqq").is_empty());
        // One word that every name of a server shares, the server's own name, or a common verb lists
        // them all, not `[]` (the A/B's `tools.search("browser")` scare — see `mcp_stdio::stale`).
        let count = |q: &str, prefix: &str| {
            index
                .rank(q)
                .iter()
                .filter(|&&i| names[i].starts_with(prefix))
                .count()
        };
        assert_eq!(count("browser", "mcp__playwright__browser_"), 25);
        assert_eq!(count("github", "mcp__github__"), 26);
        assert!(
            index
                .rank("list")
                .iter()
                .any(|&i| names[i] == "mcp__github__list_issues")
        );
        // A word only descriptions have finds every tool whose description has it.
        let hits: Vec<&str> = index
            .rank("accessibility")
            .iter()
            .map(|&i| names[i].as_str())
            .collect();
        assert!(
            hits.contains(&"mcp__playwright__browser_snapshot"),
            "{hits:?}"
        );
    }

    #[test]
    fn loading_parses_from_the_flag() {
        assert_eq!("search".parse::<McpLoading>(), Ok(McpLoading::Search));
        assert_eq!("eager".parse::<McpLoading>(), Ok(McpLoading::Eager));
        assert!("lazy".parse::<McpLoading>().is_err());
    }
}
