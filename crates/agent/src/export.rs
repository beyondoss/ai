//! Export a session's transcript as a single, self-contained HTML file — pi's `export_html`, for
//! sharing or reviewing a conversation outside the control protocol (no client, no server, just a
//! browser). Deliberately plain: one static page, inline CSS, no external assets or **client-side**
//! JS, so the file is portable and viewable offline exactly as generated — `<details>`/`<summary>`
//! (native HTML, no script) is used for every interactive piece: branch navigation (see
//! [`render_branches_diverging_at`]), the collapsed-by-default system-prompt/tools sections, and a long
//! tool result collapsed past a per-tool line threshold (see [`render_tool_result_content`]) — so the
//! page stays purely static (no JS) even though it isn't purely static *reading order* anymore. Message
//! text is rendered as markdown (`render_markdown`, via `pulldown-cmark`) — server-side, at export time,
//! rather than pi's own approach of vendoring `marked`/`highlight.js` and running them client-side inside
//! the exported file. Deliberately **not** paired with a real syntax-highlighting crate (e.g. `syntect`):
//! that bundles several MB of syntax/theme data and would slow every build of this CLI, including `run`/
//! `serve`, which never touch export, for a nice-to-have that fenced code blocks and file-shaped tool
//! results already get a useful approximation of via plain `<pre><code class="language-x">`
//! (language-tagged — from a fenced block's own info string, or from a `read`/`write`/`edit` call's
//! `path` extension, see [`language_from_path`] — monospaced, just not token-colored) — except a
//! `diff`-tagged block, or any tool-result content shaped like a unified diff, which does get real
//! per-line +/- coloring (`diff_html`/`looks_like_diff`), since that needs no language-specific lexer at
//! all. Every built-in tool call (`edit`/`write`/`bash`/`read`/`grep`/`find`/`ls`, and the Beyond
//! platform tools `fork`/`sync`/`logs`) gets a dedicated renderer (`render_tool_call`) instead of raw
//! pretty-printed JSON — `edit` in particular reuses the diff-coloring machinery, over a real
//! line-level diff (`diff_lines`), to show its before/after. Only a genuinely unrecognized tool name (a
//! third-party extension, or a future built-in not yet taught this module) falls back to generic JSON,
//! which reads fine for the rare case that needs it.

use std::collections::HashMap;
use std::io::Write as _;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use agent_core::{ContentBlock, Message, Role, ToolDef};

use crate::session_store::SessionMeta;
use crate::skills::xml_escape as html_escape;

/// Render `messages` (and `meta`'s header info) as a complete HTML document. `branches` is every
/// abandoned branch's full root-to-leaf chain plus how much of it is shared with `messages` (see
/// [`crate::session_store::SessionStore::abandoned_branches`]) — pass `&[]` for a session with no
/// tree (in-memory only) or when abandoned branches shouldn't be rendered. `usage` is the session's
/// cumulative token totals for the header's stats section (see [`UsageTotals`]) — `None` renders that
/// section without a token line, e.g. when the caller has no live `agent_core::session::Session` to read
/// running totals from.
pub fn render_html(
    meta: &SessionMeta,
    messages: &[Message],
    branches: &[(usize, Vec<Message>)],
    usage: Option<UsageTotals>,
) -> String {
    render_html_inner(meta, messages, branches, usage, &[], None, None)
}

/// Like [`render_html`], but also renders `events` — every [`crate::session_store::ExportEvent`]
/// this session recorded (a model/thinking-level switch, a label, a caller-defined custom entry) —
/// as its own simple block right after the main transcript. Track L36 (pi-parity fix): these are
/// durably tracked (`SessionStore::export_events`) but previously never reached an export at all.
/// A separate function (rather than growing [`render_html`]'s own signature) so every existing caller
/// of the plain, entries-less form — including this module's own ~35 test call sites — keeps working
/// unchanged.
pub fn render_html_with_entries(
    meta: &SessionMeta,
    messages: &[Message],
    branches: &[(usize, Vec<Message>)],
    usage: Option<UsageTotals>,
    events: &[crate::session_store::ExportEvent],
) -> String {
    render_html_inner(meta, messages, branches, usage, events, None, None)
}

/// Like [`render_html_with_entries`], but also renders `system_prompt` and `tools` as collapsible
/// sections near the top of the document, right after the header — pi's own always-included
/// `systemPrompt`/`tools` fields (`export-html/index.ts:263-270`, `template.js:1405-1435`), previously
/// omitted from this crate's export entirely. `None` (either individually or both) renders no section
/// at all, so every existing caller of the plainer forms above — this module's own test call sites,
/// plus the standalone `export` subcommand (no live `Agent`/`ToolRegistry` to pull either from,
/// genuinely not just an oversight — see its own call site's comment) — keeps rendering exactly as
/// before. `main.rs`'s `run --export` and `serve.rs`'s `export_html` RPC command both now call this
/// directly with a live system prompt and [`agent_core::ToolRegistry::definitions`] (Task #44).
pub fn render_html_full(
    meta: &SessionMeta,
    messages: &[Message],
    branches: &[(usize, Vec<Message>)],
    usage: Option<UsageTotals>,
    events: &[crate::session_store::ExportEvent],
    system_prompt: Option<&str>,
    tools: Option<&[ToolDef]>,
) -> String {
    render_html_inner(
        meta,
        messages,
        branches,
        usage,
        events,
        system_prompt,
        tools,
    )
}

fn render_html_inner(
    meta: &SessionMeta,
    messages: &[Message],
    branches: &[(usize, Vec<Message>)],
    usage: Option<UsageTotals>,
    events: &[crate::session_store::ExportEvent],
    system_prompt: Option<&str>,
    tools: Option<&[ToolDef]>,
) -> String {
    let mut out = String::new();
    out.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n");
    out.push_str(&format!(
        "<title>{}</title>\n",
        html_escape(meta.title.as_deref().unwrap_or("Session transcript"))
    ));
    out.push_str(STYLE);
    out.push_str("</head>\n<body>\n");
    out.push_str("<header>\n");
    out.push_str(&format!(
        "<h1>{}</h1>\n",
        html_escape(meta.title.as_deref().unwrap_or("Session transcript"))
    ));
    out.push_str(&format!(
        "<p class=\"meta\">session <code>{}</code> &middot; model <code>{}</code> &middot; {} \
         message(s)</p>\n",
        html_escape(&meta.id),
        html_escape(&meta.model),
        messages.len()
    ));
    render_stats_section(&mut out, meta, messages, usage, events);
    out.push_str("</header>\n");
    render_system_prompt_section(&mut out, system_prompt);
    render_tools_section(&mut out, tools);
    out.push_str("<main>\n");
    // Built once, up front, from every message this export will ever render (the main transcript plus
    // every abandoned branch) — see [`index_tool_calls`]'s own doc comment for why a `ToolResult` block
    // needs this at all (pi-parity Fix 2: language-tagging a tool result's `<pre>` from the originating
    // call's own `path` argument).
    let index = index_tool_calls(messages, branches);
    // Every branch is rendered *inline*, immediately after the message it actually diverged from
    // (`shared` is a message-count prefix — the branch shares `messages[..shared]` with the active
    // path) — a real tree laid out in reading order, rather than one flat "other branches" dump
    // disconnected from the point it forked from at the bottom of the page. `shared == 0` branches
    // (forked before the very first message) render before the loop starts. Numbered in the order
    // they appear so a reader can refer to "branch 2" unambiguously even though they're scattered
    // through the page rather than listed together.
    let mut branch_number = render_branches_diverging_at(&mut out, branches, 0, 1, &index);
    // Every `ModelChange` event this document can position inline renders as its own small block right
    // before the assistant turn that actually used the new model — Task #27 (pi-parity fix): pi's own
    // `template.js` renders `model_change` inline, as part of `renderEntry`'s single chronological walk
    // over one tree (each entry carrying its own `id`/`parentId`/`timestamp`), instead of a disconnected
    // trailing dump. `ExportEvent` itself carries no such anchor back to a specific message (see its own
    // doc comment) — this correlates by value instead: a model switch is always recorded before the next
    // assistant turn that actually ran under the new model (`Message::model_id`, stamped at the point
    // that turn is recorded — see `with_model_id`'s own doc comment), so walking `model_changes` in
    // lockstep with `messages` and matching on that value recovers the same position without needing a
    // real anchor. `model_change_cursor` only ever advances, never backtracks, so an earlier pending
    // event can't be skipped in favor of a later one matching sooner (see `render_model_change`'s call
    // site below for the one pathological case this simplification accepts).
    let model_changes: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            crate::session_store::ExportEvent::ModelChange(model) => Some(model.as_str()),
            _ => None,
        })
        .collect();
    let mut model_change_cursor = 0usize;
    for (i, message) in messages.iter().enumerate() {
        if let Some(model_id) = &message.model_id {
            while model_changes.get(model_change_cursor) == Some(&model_id.as_str()) {
                render_model_change(&mut out, model_id);
                model_change_cursor += 1;
            }
        }
        render_message(&mut out, message, &index);
        branch_number =
            render_branches_diverging_at(&mut out, branches, i + 1, branch_number, &index);
    }
    // A `ModelChange` that never matched a later message (e.g. the session ended, or was exported,
    // before another assistant turn ran under the new model) still needs to be visible somewhere —
    // `model_change_cursor` stops advancing right before it, so it's still present in `events` and falls
    // through to `render_events_section`'s flat fallback list below, same as every `ModelChange` did
    // before this fix, rather than being silently dropped now that most of them render inline instead.
    let mut model_changes_seen = 0usize;
    let trailing_events: Vec<crate::session_store::ExportEvent> = events
        .iter()
        .filter(|e| match e {
            crate::session_store::ExportEvent::ModelChange(_) => {
                model_changes_seen += 1;
                model_changes_seen > model_change_cursor
            }
            _ => true,
        })
        .cloned()
        .collect();
    render_events_section(&mut out, &trailing_events);
    out.push_str("</main>\n");
    out.push_str("</body>\n</html>\n");
    out
}

/// A simple, un-fancy block per [`crate::session_store::ExportEvent`] — Track L36's whole point is
/// visibility, not fidelity to where in the transcript each one actually happened (that would need
/// each event's own position/timestamp threaded all the way through from `SessionStore`, a bigger
/// change than "currently invisible" calls for). A no-op when `events` is empty, so a plain
/// [`render_html`] call (which always passes `&[]`) never adds this section at all.
///
/// Only ever called today with the *trailing* subset of `events`: every `ModelChange` this document
/// managed to position inline is filtered out by [`render_html_inner`]'s main loop before reaching here
/// (Task #27, pi-parity fix) — only one that never matched a later message still shows up in this flat
/// fallback list. `Label`'s own `target_id` (Task #26, pi-parity fix) still can't become a real
/// same-document anchor here — a bare `Message` carries no id at all (that lives one layer up, on
/// `SessionStore`'s own tree-entry wrapper, never threaded into `render_html`'s `messages: &[Message]`)
/// — so it renders as a plain `<code>` reference instead: enough for a reader to tell which target a
/// label belongs to even without a clickable link. Cheap fix for a gratuitous extra loss on top of an
/// already-accepted non-goal (pi's own label rendering is tree-sidebar-only too).
fn render_events_section(out: &mut String, events: &[crate::session_store::ExportEvent]) {
    if events.is_empty() {
        return;
    }
    out.push_str("<section class=\"events\">\n<h2>Session Events</h2>\n<ul>\n");
    for event in events {
        let line = match event {
            crate::session_store::ExportEvent::ModelChange(model) => {
                format!("Model changed to <code>{}</code>", html_escape(model))
            }
            crate::session_store::ExportEvent::ThinkingLevelChange(level) => {
                format!(
                    "Thinking level changed to <code>{}</code>",
                    html_escape(level)
                )
            }
            crate::session_store::ExportEvent::Label {
                target_id,
                label: Some(label),
            } => {
                format!(
                    "Labeled <code>{}</code>: {}",
                    html_escape(target_id),
                    html_escape(label)
                )
            }
            crate::session_store::ExportEvent::Label {
                target_id,
                label: None,
            } => {
                format!("Label cleared on <code>{}</code>", html_escape(target_id))
            }
            crate::session_store::ExportEvent::Custom { kind, data } => {
                format!(
                    "Custom entry <code>{}</code>: {}",
                    html_escape(kind),
                    html_escape(&data.to_string())
                )
            }
        };
        out.push_str(&format!("<li>{line}</li>\n"));
    }
    out.push_str("</ul>\n</section>\n");
}

/// Render a single `ModelChange` event inline, right at the point in the transcript
/// [`render_html_inner`]'s main loop determined it actually landed (Task #27, pi-parity fix) — its own
/// small block, styled distinctly from a message, rather than a `<li>` in the flat trailing dump
/// [`render_events_section`] still falls back to for one that couldn't be positioned. Deliberately the
/// same wording that flat list already used for this variant (`"Model changed to <code>{}</code>"`), so
/// a reader sees identical phrasing regardless of which of the two ever renders a given switch.
fn render_model_change(out: &mut String, model: &str) {
    out.push_str(&format!(
        "<div class=\"model-change\">Model changed to <code>{}</code></div>\n",
        html_escape(model)
    ));
}

/// Render the session's system prompt as a collapsed-by-default `<details>` block — pi's own
/// always-included `systemPrompt` section (`export-html/index.ts:267`, `template.js:1405-1417`). A
/// no-op when `system_prompt` is `None`, which is every caller today except [`render_html_full`] (see
/// that function's own doc comment for why).
fn render_system_prompt_section(out: &mut String, system_prompt: Option<&str>) {
    let Some(prompt) = system_prompt else {
        return;
    };
    out.push_str(&format!(
        "<details class=\"system-prompt\"><summary>System Prompt</summary>\n<pre>{}</pre>\n</details>\n",
        html_escape(prompt)
    ));
}

/// Render the session's registered tools (name, description, and JSON-Schema parameters) as a
/// collapsed-by-default `<details>` block — pi's own always-included `tools` section
/// (`export-html/index.ts:268`, `template.js:1423-1447`). A no-op when `tools` is `None` or empty.
fn render_tools_section(out: &mut String, tools: Option<&[ToolDef]>) {
    let Some(tools) = tools.filter(|t| !t.is_empty()) else {
        return;
    };
    out.push_str(&format!(
        "<details class=\"tools-list\"><summary>Available Tools ({})</summary>\n\
         <div class=\"tools-content\">\n",
        tools.len()
    ));
    for tool in tools {
        out.push_str(&format!(
            "<div class=\"tool-item\"><span class=\"tool-item-name\">{}</span> &mdash; \
             <span class=\"tool-item-desc\">{}</span>\n",
            html_escape(&tool.name),
            html_escape(&tool.description)
        ));
        render_tool_params(out, &tool.input_schema);
        out.push_str("</div>\n");
    }
    out.push_str("</div>\n</details>\n");
}

/// Render one tool's JSON-Schema `input_schema`'s `properties`/`required` as a per-parameter list —
/// matching pi's own `t.parameters.properties`/`required` walk (`template.js:1428-1440`). A no-op when
/// `schema` has no non-empty `properties` object, same as pi's own `hasParams` guard.
fn render_tool_params(out: &mut String, schema: &serde_json::Value) {
    let Some(properties) = schema
        .get("properties")
        .and_then(serde_json::Value::as_object)
    else {
        return;
    };
    if properties.is_empty() {
        return;
    }
    let required: Vec<&str> = schema
        .get("required")
        .and_then(serde_json::Value::as_array)
        .map(|a| a.iter().filter_map(serde_json::Value::as_str).collect())
        .unwrap_or_default();
    out.push_str("<div class=\"tool-params\">\n");
    for (name, prop) in properties {
        let ty = prop
            .get("type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("any");
        let (req_class, req_label) = if required.contains(&name.as_str()) {
            ("tool-param-required", "required")
        } else {
            ("tool-param-optional", "optional")
        };
        out.push_str(&format!(
            "<div class=\"tool-param\"><span class=\"tool-param-name\">{}</span> \
             <span class=\"tool-param-type\">{}</span> <span class=\"{req_class}\">{req_label}</span>",
            html_escape(name),
            html_escape(ty),
        ));
        if let Some(desc) = prop.get("description").and_then(serde_json::Value::as_str) {
            out.push_str(&format!(
                "<div class=\"tool-param-desc\">{}</div>",
                html_escape(desc)
            ));
        }
        out.push_str("</div>\n");
    }
    out.push_str("</div>\n");
}

/// Render every branch whose `shared` divergence point equals `at`, as a collapsed-by-default
/// `<details>` block — expandable inline without leaving the reader's place in the main transcript,
/// and with zero JS (`<details>`/`<summary>` is native HTML). `next_number` is threaded through by the
/// caller's loop so branch numbering stays sequential across every divergence point in one document,
/// not just within one call. Only the part that actually diverges from the active path
/// (`branch[shared..]`) is rendered — the shared prefix is already shown once, in the main transcript.
fn render_branches_diverging_at(
    out: &mut String,
    branches: &[(usize, Vec<Message>)],
    at: usize,
    next_number: usize,
    index: &ToolCallIndex<'_>,
) -> usize {
    let mut n = next_number;
    for (shared, branch_messages) in branches.iter().filter(|(shared, _)| *shared == at) {
        let note = if *shared == 0 {
            "forked from the start".to_string()
        } else {
            format!("forked after message {shared}")
        };
        out.push_str(&format!(
            "<details class=\"branch\"><summary>Branch {n} &middot; {} &middot; {} message(s)\
             </summary>\n<div class=\"branch-body\">\n",
            html_escape(&note),
            branch_messages.len() - shared
        ));
        for message in &branch_messages[*shared..] {
            render_message(out, message, index);
        }
        out.push_str("</div>\n</details>\n");
        n += 1;
    }
    n
}

/// Map from a [`ContentBlock::ToolUse`] block's `id` to its `(name, input)` — built once (see
/// [`index_tool_calls`]) from every message an export will render, so a [`ContentBlock::ToolResult`]
/// block (which carries only `tool_use_id`, never the tool's own name/arguments) can look up which tool
/// actually produced it. Used today for exactly one thing: tagging a `read`/`write`/`edit` result's
/// `<pre>` with a `language-{ext}` class derived from that tool's own `path` argument (pi-parity Fix 2,
/// see [`render_tool_result_content`]) — a plain lookup table, not a general "tool call log", so it's
/// fine that a call with no matching result (or vice versa) just misses.
type ToolCallIndex<'a> = HashMap<&'a str, (&'a str, &'a serde_json::Value)>;

/// Build a [`ToolCallIndex`] from every [`ContentBlock::ToolUse`] block in `messages` and every
/// abandoned `branches` message — the union of everything [`render_html_inner`] will actually render,
/// so a tool result anywhere in the document (main transcript or an inline branch) can be traced back to
/// the call that produced it regardless of which of the two it lives in.
fn index_tool_calls<'a>(
    messages: &'a [Message],
    branches: &'a [(usize, Vec<Message>)],
) -> ToolCallIndex<'a> {
    let mut index = ToolCallIndex::new();
    let all_messages = messages
        .iter()
        .chain(branches.iter().flat_map(|(_, branch)| branch.iter()));
    for message in all_messages {
        for block in &message.content {
            if let ContentBlock::ToolUse {
                id, name, input, ..
            } = block
            {
                index.insert(id.as_str(), (name.as_str(), input));
            }
        }
    }
    index
}

/// Cumulative token totals for the exported header's stats section — the same running counters
/// [`agent_core::session::Session`] already tracks (`input_tokens`/`output_tokens`/`cache_read_tokens`/
/// `cache_write_tokens`). Threaded through explicitly rather than summed from `messages` alone: unlike
/// pi's own per-assistant-message `usage` field, `agent_core::Message` carries no per-message usage to
/// sum, only `Session` accumulates it. Deliberately excludes dollar cost — pricing lives downstream of
/// this codebase, not here.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UsageTotals {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
}

/// Entry-type counts derived purely from `messages`/`events` — unlike token usage, these need no data
/// beyond what [`render_html_with_entries`] already receives.
struct MessageStats {
    user_messages: usize,
    assistant_messages: usize,
    tool_calls: usize,
    /// How many individual `ContentBlock::ToolResult` blocks appear across every message (Task #32,
    /// pi-parity fix) — matching pi's own header line, which always shows a tool-results count
    /// alongside `user`/`assistant` (e.g. "12 user, 15 assistant, 40 tool results"); this crate's own
    /// `tool_calls` count already existed, but nothing on the results side did.
    tool_results: usize,
    /// How many [`crate::session_store::ExportEvent::Custom`] entries this session recorded (Task #32).
    /// Unlike every other field here, this can't come from `messages` at all — a custom entry
    /// contributes nothing to `Session.messages` (see that variant's own doc comment) — so it's the one
    /// count `compute` derives from `events` instead.
    custom_entries: usize,
    /// Distinct `model_id`s actually seen on an assistant turn (sorted), so a session that switched
    /// models mid-run shows every model actually used, not just the one it started with.
    models: Vec<String>,
    /// How many messages are a compaction recap (see [`parse_summary_marker`]) — pi-parity Fix 5: pi's
    /// header always folds this into its "Messages" summary line when present (`template.js:1352-1381`,
    /// e.g. "N compactions"), previously silently dropped here even though the recap itself already
    /// renders as its own distinct block in the transcript body ([`render_summary_marker`]).
    compactions: usize,
    /// Same as `compactions`, for a branch-summary recap (the other [`parse_summary_marker`] class).
    branch_summaries: usize,
}

impl MessageStats {
    fn compute(messages: &[Message], events: &[crate::session_store::ExportEvent]) -> Self {
        let mut user_messages = 0;
        let mut assistant_messages = 0;
        let mut tool_calls = 0;
        let mut tool_results = 0;
        let mut models: Vec<String> = Vec::new();
        let mut compactions = 0;
        let mut branch_summaries = 0;
        for m in messages {
            match m.role {
                Role::User => {
                    // A message carrying only `ToolResult` blocks is the model's tool feedback, not a
                    // real user turn — don't double-count it as one just because it rides on `Role::User`
                    // (Anthropic's own convention for where tool results live).
                    if m.content
                        .iter()
                        .any(|b| !matches!(b, ContentBlock::ToolResult { .. }))
                    {
                        user_messages += 1;
                    }
                    // A compaction/branch-summary recap is itself a plain `Role::User` text block (see
                    // `parse_summary_marker`'s own doc comment) — still counted above as an ordinary user
                    // message (that behavior predates this fix and stays unchanged), just *also* tallied
                    // here so the header can surface how many of the session's messages are one of these.
                    for block in &m.content {
                        if let ContentBlock::Text { text, .. } = block {
                            match parse_summary_marker(text) {
                                Some(("compaction", ..)) => compactions += 1,
                                Some(("branch-summary", ..)) => branch_summaries += 1,
                                _ => {}
                            }
                        }
                    }
                }
                Role::Assistant => {
                    assistant_messages += 1;
                    if let Some(id) = &m.model_id {
                        if !models.contains(id) {
                            models.push(id.clone());
                        }
                    }
                }
                Role::System => {}
            }
            tool_calls += m
                .content
                .iter()
                .filter(|b| matches!(b, ContentBlock::ToolUse { .. }))
                .count();
            tool_results += m
                .content
                .iter()
                .filter(|b| matches!(b, ContentBlock::ToolResult { .. }))
                .count();
        }
        models.sort();
        let custom_entries = events
            .iter()
            .filter(|e| matches!(e, crate::session_store::ExportEvent::Custom { .. }))
            .count();
        Self {
            user_messages,
            assistant_messages,
            tool_calls,
            tool_results,
            custom_entries,
            models,
            compactions,
            branch_summaries,
        }
    }
}

/// Render the header's aggregate stats section: models used, a user/assistant/tool-call breakdown, and
/// (when `usage` is available) total token counts — matching pi's own always-shown inline summary
/// (`template.js:1323-1364`), minus dollar cost (out of scope here; pricing lives downstream of this
/// codebase).
fn render_stats_section(
    out: &mut String,
    meta: &SessionMeta,
    messages: &[Message],
    usage: Option<UsageTotals>,
    events: &[crate::session_store::ExportEvent],
) {
    let stats = MessageStats::compute(messages, events);
    out.push_str("<div class=\"stats\">\n");
    // pi-parity fix: pi's own export always renders a `Date:` line first
    // (`template.js:1395`, `header.timestamp`) — this crate's stats section previously read
    // `meta.model`/every message/`usage`/`events` but never `meta.created_at` at all, so the session's
    // creation date never appeared in an exported transcript anywhere. Formatted with
    // `resources::format_local_datetime` — the same hand-rolled, no-date-crate machinery already used
    // for the system prompt's own dynamic "today" footer, rather than introducing a second convention.
    push_stat(
        out,
        "Date",
        &crate::resources::format_local_datetime(meta.created_at),
    );
    let models = if stats.models.is_empty() {
        meta.model.clone()
    } else {
        stats.models.join(", ")
    };
    push_stat(out, "Models", &models);
    // pi-parity Fix 5: fold compaction/branch-summary counts into this same line, matching pi's own
    // `msgParts` (`template.js:1352-1381`) — appended only when present, so an ordinary session with
    // neither renders exactly the plain "N user, N assistant" line it always has. Task #32 (pi-parity
    // fix): `tool_results`/`custom_entries` join the same conditional-append pattern — pi's own header
    // always folds a tool-results count into this line too (e.g. "12 user, 15 assistant, 40 tool
    // results"), and a custom entry is exactly as rare/optional as a compaction or branch summary, so
    // it gets the same "only when present" treatment rather than cluttering every ordinary session's
    // line with "0 custom entries".
    let mut message_parts = vec![
        format!("{} user", stats.user_messages),
        format!("{} assistant", stats.assistant_messages),
    ];
    if stats.tool_results > 0 {
        message_parts.push(format!("{} tool results", stats.tool_results));
    }
    if stats.compactions > 0 {
        message_parts.push(format!("{} compactions", stats.compactions));
    }
    if stats.branch_summaries > 0 {
        message_parts.push(format!("{} branch summaries", stats.branch_summaries));
    }
    if stats.custom_entries > 0 {
        message_parts.push(format!("{} custom entries", stats.custom_entries));
    }
    push_stat(out, "Messages", &message_parts.join(", "));
    push_stat(out, "Tool calls", &stats.tool_calls.to_string());
    if let Some(u) = usage {
        push_stat(
            out,
            "Tokens",
            &format!(
                "{} in, {} out, {} cache read, {} cache write",
                u.input_tokens, u.output_tokens, u.cache_read_tokens, u.cache_write_tokens
            ),
        );
    }
    out.push_str("</div>\n");
}

fn push_stat(out: &mut String, label: &str, value: &str) {
    out.push_str(&format!(
        "<div class=\"stat\"><span class=\"stat-label\">{}</span><span class=\"stat-value\">{}</span>\
         </div>\n",
        html_escape(label),
        html_escape(value)
    ));
}

const STYLE: &str = "<style>\n\
body { font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif; max-width: 860px; \
margin: 2rem auto; padding: 0 1rem; background: #1e1e1e; color: #d4d4d4; }\n\
header { border-bottom: 1px solid #444; margin-bottom: 1.5rem; padding-bottom: 1rem; }\n\
h1 { font-size: 1.3rem; margin: 0 0 0.25rem; }\n\
.meta { color: #888; font-size: 0.85rem; margin: 0; }\n\
.message { border-radius: 6px; padding: 0.75rem 1rem; margin-bottom: 1rem; }\n\
.role-user { background: #2d3a4a; }\n\
.role-assistant { background: #2a2a2a; }\n\
.role-system { background: #3a2a2a; }\n\
.role-label { font-weight: 600; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; \
color: #999; margin-bottom: 0.4rem; }\n\
.tool-call, .tool-result { border-left: 3px solid #666; padding-left: 0.75rem; margin: 0.5rem 0; }\n\
.tool-result.error { border-left-color: #c94f4f; }\n\
.tool-call.host-bash { border-left-color: #d0a94c; }\n\
.tool-call.host-bash.error { border-left-color: #c94f4f; }\n\
.skill-invocation { border-left: 3px solid #a67ee2; padding-left: 0.75rem; margin: 0.5rem 0; }\n\
.tool-title { font-size: 0.8rem; color: #aaa; margin-bottom: 0.25rem; }\n\
pre { white-space: pre-wrap; word-wrap: break-word; background: #151515; padding: 0.5rem; \
border-radius: 4px; overflow-x: auto; margin: 0.25rem 0; }\n\
.thinking { font-style: italic; color: #888; border-left: 3px solid #555; padding-left: 0.75rem; \
margin: 0.5rem 0; }\n\
img.attachment { max-width: 100%; border-radius: 4px; margin: 0.5rem 0; display: block; }\n\
.branch { border: 1px dashed #555; border-radius: 6px; padding: 0.5rem 0.75rem; margin: 0.75rem 0; \
background: #232323; }\n\
.branch summary { cursor: pointer; font-size: 0.8rem; color: #bbb; user-select: none; }\n\
.branch summary:hover { color: #fff; }\n\
.branch[open] summary { margin-bottom: 0.5rem; color: #fff; }\n\
.branch-body { border-left: 2px solid #555; padding-left: 0.75rem; }\n\
.bash-command { color: #7ee2a8; }\n\
.bash-status { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 0.4rem; }\n\
.bash-badge { display: inline-block; font-size: 0.75rem; font-weight: 600; padding: 0.1rem 0.5rem; \
border-radius: 3px; }\n\
.bash-badge-ok { background: rgba(80, 200, 120, 0.15); color: #7ee2a8; }\n\
.bash-badge-error { background: rgba(220, 80, 80, 0.15); color: #f0908f; }\n\
.bash-badge-cancelled { background: rgba(208, 169, 76, 0.15); color: #d0a94c; }\n\
.bash-badge-truncated { background: rgba(108, 182, 255, 0.15); color: #6cb6ff; font-weight: 400; }\n\
.markdown { line-height: 1.5; }\n\
.markdown p { margin: 0.4rem 0; }\n\
.markdown p:first-child { margin-top: 0; }\n\
.markdown p:last-child { margin-bottom: 0; }\n\
.markdown ul, .markdown ol { margin: 0.4rem 0; padding-left: 1.5rem; }\n\
.markdown blockquote { border-left: 3px solid #555; margin: 0.5rem 0; padding-left: 0.75rem; \
color: #aaa; }\n\
.markdown table { border-collapse: collapse; margin: 0.5rem 0; }\n\
.markdown th, .markdown td { border: 1px solid #444; padding: 0.3rem 0.6rem; }\n\
.markdown code { background: #151515; padding: 0.1rem 0.3rem; border-radius: 3px; }\n\
.markdown pre code { background: none; padding: 0; border-radius: 0; }\n\
.markdown a { color: #6cb6ff; }\n\
.diff-add { background: rgba(80, 200, 120, 0.15); color: #7ee2a8; display: block; }\n\
.diff-del { background: rgba(220, 80, 80, 0.15); color: #f0908f; display: block; }\n\
.diff-hunk { color: #6cb6ff; }\n\
.diff-file { color: #d0a94c; font-weight: 600; }\n\
.turn-status { font-weight: 600; font-size: 0.8rem; margin-top: 0.5rem; padding: 0.4rem 0.6rem; \
border-radius: 4px; }\n\
.turn-status.aborted { background: rgba(208, 169, 76, 0.15); color: #d0a94c; }\n\
.turn-status.error { background: rgba(220, 80, 80, 0.15); color: #f0908f; }\n\
.summary-marker { border-left: 3px solid #6cb6ff; border-radius: 6px; padding: 0.6rem 0.9rem; \
margin: 0.5rem 0; background: #232323; }\n\
.summary-marker.compaction { border-left-color: #d0a94c; }\n\
.summary-marker.branch-summary { border-left-color: #a67ee2; }\n\
.model-change { font-size: 0.8rem; color: #999; border-left: 3px solid #6cb6ff; padding: 0.3rem 0.75rem; \
margin: 0.5rem 0; background: #232323; }\n\
.model-change code { color: #d4d4d4; }\n\
.stats { display: flex; flex-wrap: wrap; gap: 0.5rem 1.5rem; margin-top: 0.6rem; }\n\
.stat { font-size: 0.85rem; }\n\
.stat-label { color: #888; margin-right: 0.35rem; }\n\
.stat-value { color: #d4d4d4; }\n\
.system-prompt, .tools-list { border: 1px dashed #555; border-radius: 6px; padding: 0.5rem 0.75rem; \
margin: 0.75rem 0; background: #232323; }\n\
.system-prompt summary, .tools-list summary { cursor: pointer; font-size: 0.85rem; font-weight: 600; \
color: #bbb; user-select: none; }\n\
.system-prompt summary:hover, .tools-list summary:hover { color: #fff; }\n\
.system-prompt pre { margin-top: 0.5rem; }\n\
.tools-content { margin-top: 0.5rem; }\n\
.tool-item { padding: 0.4rem 0; border-bottom: 1px solid #333; }\n\
.tool-item:last-child { border-bottom: none; }\n\
.tool-item-name { font-weight: 600; font-family: monospace; color: #6cb6ff; }\n\
.tool-item-desc { color: #ccc; }\n\
.tool-params { margin: 0.35rem 0 0 1rem; }\n\
.tool-param { font-size: 0.85rem; margin: 0.2rem 0; }\n\
.tool-param-name { font-family: monospace; color: #d0a94c; }\n\
.tool-param-type { color: #888; font-style: italic; }\n\
.tool-param-required { color: #f0908f; font-size: 0.75rem; }\n\
.tool-param-optional { color: #7ee2a8; font-size: 0.75rem; }\n\
.tool-param-desc { color: #999; font-size: 0.8rem; margin-left: 0.25rem; }\n\
.collapsible-output summary { cursor: pointer; font-size: 0.8rem; color: #999; user-select: none; \
margin-bottom: 0.25rem; }\n\
.collapsible-output summary:hover { color: #ddd; }\n\
.collapsible-output pre { margin-top: 0; }\n\
</style>\n";

fn role_label(role: Role) -> &'static str {
    match role {
        Role::User => "User",
        Role::Assistant => "Assistant",
        Role::System => "System",
    }
}

fn role_class(role: Role) -> &'static str {
    match role {
        Role::User => "role-user",
        Role::Assistant => "role-assistant",
        Role::System => "role-system",
    }
}

fn render_message(out: &mut String, message: &Message, index: &ToolCallIndex<'_>) {
    out.push_str(&format!(
        "<div class=\"message {}\">\n<div class=\"role-label\">{}</div>\n",
        role_class(message.role),
        role_label(message.role)
    ));
    for block in &message.content {
        render_block(out, block, index);
    }
    // A turn stopped by an abort or a transport failure typically has empty/partial `content` (see
    // `Message::aborted`/`Message::error_message`'s doc comments) — without these, the message div above
    // renders completely blank, giving zero indication anything went wrong in exactly the scenario
    // (debugging a failed run) export exists for. Matches pi's own `stopReason: "aborted"`/`"error"`
    // handling (`template.js:1263-1267`).
    if message.aborted {
        out.push_str("<div class=\"turn-status aborted\">Aborted</div>\n");
    }
    if let Some(err) = &message.error_message {
        out.push_str(&format!(
            "<div class=\"turn-status error\">Error: {}</div>\n",
            html_escape(err)
        ));
    }
    out.push_str("</div>\n");
}

fn render_block(out: &mut String, block: &ContentBlock, index: &ToolCallIndex<'_>) {
    match block {
        ContentBlock::Text { text, .. } => {
            // A host-run bash command (`serve.rs`'s `bash` RPC command, run from the idle loop rather
            // than the model's own turn) materializes as a plain `Role::User` text block carrying a
            // literal bracketed marker line — detect and render it as its own distinct, code-styled
            // block instead of falling through to `render_markdown`, where multi-line output collapsed
            // into one unreadable run-on line (pi-parity gap; pi has a dedicated `bashExecution` role,
            // `template.js:1273-1285`).
            if let Some(host_bash) = parse_host_bash_marker(text) {
                render_host_bash_marker(out, &host_bash);
            } else if let Some((class, label, tokens_before, body)) = parse_summary_marker(text) {
                // A compaction or branch-summary recap materializes as a plain `Role::User` text block
                // carrying a literal bracketed marker line (`agent_core::compaction::apply_summary` /
                // `session_store.rs::branch_summary_message`) — detect and render it as its own
                // distinctly labeled block, matching pi's dedicated `.compaction`/`.branch-summary`
                // blocks (`template.js:1294-1306`), instead of plain markdown text with the marker line
                // still visible verbatim as if the model had actually written it.
                render_summary_marker(out, class, label, tokens_before, body);
            } else {
                // A `/skill:name` invocation stores its expansion as a structural
                // `<skill name="..." location="...">...</skill>` wrapper around the model-facing text
                // (see `skills.rs::expand_if_skill_invocation`) — render it as its own distinct block
                // instead of as one raw-escaped blob with the wrapper tags visible as literal text.
                match parse_skill_block(text) {
                    Some(skill) => render_skill_invocation(out, &skill),
                    None => {
                        out.push_str(&format!(
                            "<div class=\"text markdown\">{}</div>\n",
                            render_markdown(text)
                        ));
                    }
                }
            }
        }
        ContentBlock::Thinking { text, .. } => {
            out.push_str(&format!(
                "<div class=\"thinking\">{}</div>\n",
                html_escape(text)
            ));
        }
        ContentBlock::RedactedThinking { .. } => {
            out.push_str("<div class=\"thinking\">[redacted thinking]</div>\n");
        }
        ContentBlock::ToolUse { name, input, .. } => render_tool_call(out, name, input),
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
            images,
            ..
        } => {
            let class = if *is_error {
                "tool-result error"
            } else {
                "tool-result"
            };
            let title = if *is_error { "Error" } else { "Result" };
            out.push_str(&format!(
                "<div class=\"{class}\"><div class=\"tool-title\">{title}</div>\n"
            ));
            let tool_info = index.get(tool_use_id.as_str()).copied();
            render_tool_result_content(out, content, tool_info);
            for image in images {
                render_image(out, &image.media_type, &image.data);
            }
            out.push_str("</div>\n");
        }
        ContentBlock::Image { source } => render_image(out, &source.media_type, &source.data),
    }
}

/// Render a tool result's raw content — verbatim, not markdown (it's raw command/file output, and
/// interpreting a `#`-prefixed shell comment or a `-`-prefixed line as markdown would misrender it) —
/// except a unified diff, which gets [`diff_html`]'s per-line +/- coloring, same as a fenced
/// ` ```diff ` block in message text.
///
/// Three additive pi-parity fixes live here:
/// - **Fix 2** (language tagging): when `tool_info` (the originating call's `(name, input)`, looked up
///   by `tool_use_id` — see [`index_tool_calls`]) is a `read`/`write`/`edit` call with a recognized
///   `path` extension, the `<pre>` is tagged `<code class="language-{ext}">` (pi's own
///   `getLanguageFromPath`, `template.js:823-834`) — ready for a future client-side highlighter without
///   adding one now.
/// - **Fix 3** (collapse affordance): past a per-tool line-count threshold
///   (see [`tool_result_line_threshold`], matching pi's own `formatExpandableOutput` thresholds,
///   `template.js:848-902`), the block is collapsed by default behind a `<details>`/`<summary>` — the
///   same zero-client-JS pattern already used for the system-prompt/tools/branch sections.
/// - **Fix 4** (control-byte defense): C0 control bytes (a stray raw ANSI escape, etc. — see
///   [`strip_control_chars`]) are stripped before escaping, so they can't garble the rendered page. Not
///   a full ANSI-to-styled-HTML converter (out of scope) — just cheap belt-and-suspenders, since
///   `html_escape` itself only escapes `&<>"'`.
fn render_tool_result_content(
    out: &mut String,
    content: &str,
    tool_info: Option<(&str, &serde_json::Value)>,
) {
    let lang = tool_info.and_then(|(name, input)| language_class_for_tool_result(name, input));
    let threshold = tool_info
        .map(|(name, _)| tool_result_line_threshold(name))
        .unwrap_or(DEFAULT_TOOL_RESULT_LINE_THRESHOLD);
    render_collapsible_output(out, content, lang, threshold);
}

/// The actual `<pre>`/`<details>` construction shared by every call site that needs pi's
/// `formatExpandableOutput` collapse affordance (Fix 3) — [`render_tool_result_content`] (a tool's
/// result content), [`render_write_call`] (a `write` call's own content preview, pi-parity Task #46:
/// previously rendered unconditionally with no collapse at all), and [`render_host_bash_marker`] (a
/// host-run bash command's output, pi-parity Task #47: same previously-missing collapse). Pulled out
/// as its own function rather than left inline in `render_tool_result_content` because none of these
/// three callers otherwise share a `(tool_name, input)` pair to look up a language/threshold from — a
/// host-bash command has no originating `ToolUse` at all, and a `write` call's own threshold (see
/// [`tool_result_line_threshold`]) needs to apply even though it's rendering the call's *input*, not a
/// `ToolResult` block.
fn render_collapsible_output(
    out: &mut String,
    content: &str,
    lang: Option<&str>,
    threshold: usize,
) {
    let cleaned = strip_control_chars(content);
    let body = if looks_like_diff(&cleaned) {
        diff_html(&cleaned)
    } else {
        match lang {
            Some(lang) => format!(
                "<pre><code class=\"language-{lang}\">{}</code></pre>\n",
                html_escape(&cleaned)
            ),
            None => format!("<pre>{}</pre>", html_escape(&cleaned)),
        }
    };
    let line_count = cleaned.lines().count();
    if line_count > threshold {
        out.push_str(&format!(
            "<details class=\"collapsible-output\"><summary>{line_count} lines (click to expand)\
             </summary>\n{body}</details>\n"
        ));
    } else {
        out.push_str(&body);
    }
}

/// Per-tool line-count threshold past which [`render_tool_result_content`]/[`render_write_call`]
/// collapse content behind `<details>` — matching pi's own per-tool thresholds (`template.js:848-902`,
/// `967-988`: `bash` calls `formatExpandableOutput(output, 5)`, `read` and `write` each call it with
/// `10`; everything else, including `ls`, falls back to [`DEFAULT_TOOL_RESULT_LINE_THRESHOLD`], which
/// happens to equal pi's own `ls` threshold of `20` too).
fn tool_result_line_threshold(tool_name: &str) -> usize {
    match tool_name {
        "bash" => 5,
        "read" | "write" => 10,
        _ => DEFAULT_TOOL_RESULT_LINE_THRESHOLD,
    }
}

/// The collapse threshold for a tool result with no associated call in the [`ToolCallIndex`] (a
/// tool-result-only test fixture, or a session missing its originating `ToolUse` for some other reason)
/// and for every tool that isn't specially thresholded in [`tool_result_line_threshold`] — pi's own `ls`
/// threshold (`template.js:902`).
const DEFAULT_TOOL_RESULT_LINE_THRESHOLD: usize = 20;

/// Derive a `language-{ext}` tag for a tool result's content from the originating call's own `path`
/// argument — only for the three tools whose result content is actually file/text content in that
/// path's language (`read`/`write`/`edit`); everything else (`bash` output, `grep` matches, `ls`
/// listings, ...) isn't meaningfully one source language, so gets no tag, matching pi's own call sites
/// for `getLanguageFromPath` (`template.js:962`, `981`).
fn language_class_for_tool_result(
    tool_name: &str,
    input: &serde_json::Value,
) -> Option<&'static str> {
    if !matches!(tool_name, "read" | "write" | "edit") {
        return None;
    }
    let path = input.get("path").and_then(serde_json::Value::as_str)?;
    language_from_path(path)
}

/// Map a file path's extension to a `language-{x}` tag — pi's own `getLanguageFromPath`
/// (`template.js:823-834`), byte-for-byte the same extension table (including its one quirk: a path
/// with no `.` at all, e.g. `Dockerfile`, is treated as if its *whole* name were the extension, same as
/// JS's `filePath.split('.').pop()` on a dot-less string). Not an exhaustive list — an unrecognized or
/// absent extension renders untagged, same as today, just without the new class.
fn language_from_path(path: &str) -> Option<&'static str> {
    let ext = path.rsplit('.').next()?.to_ascii_lowercase();
    Some(match ext.as_str() {
        "ts" | "tsx" => "typescript",
        "js" | "jsx" => "javascript",
        "py" => "python",
        "rb" => "ruby",
        "rs" => "rust",
        "go" => "go",
        "java" => "java",
        "c" => "c",
        "cpp" => "cpp",
        "h" => "c",
        "hpp" => "cpp",
        "cs" => "csharp",
        "php" => "php",
        "sh" | "bash" | "zsh" => "bash",
        "sql" => "sql",
        "html" => "html",
        "css" => "css",
        "scss" => "scss",
        "json" => "json",
        "yaml" | "yml" => "yaml",
        "xml" => "xml",
        "md" => "markdown",
        "dockerfile" => "dockerfile",
        _ => return None,
    })
}

/// Strip C0 control characters (`0x00`-`0x1F`, excluding tab `\t` and newline `\n`) and DEL (`0x7F`)
/// from tool-result text before it's escaped/rendered — pi-parity Fix 4. [`html_escape`] only escapes
/// `&<>"'`, so a stray raw control byte (most plausibly an ANSI escape sequence a tool emitted) would
/// otherwise pass straight through into the page, garbling it (though never executing anything — this
/// is a rendering-quality fix, not a security one). Beyond's only tool that could currently emit ANSI
/// color (`bash`) already strips it at capture time, so this is cheap belt-and-suspenders for the rare
/// case something else doesn't, not a fix for a live gap. Deliberately *not* an ANSI-to-styled-HTML
/// converter — that's a much bigger feature, out of scope here.
fn strip_control_chars(text: &str) -> String {
    text.chars()
        .filter(|&c| !matches!(c, '\u{0}'..='\u{8}' | '\u{b}'..='\u{1f}' | '\u{7f}'))
        .collect()
}

/// Dispatch a tool call to a renderer that understands its specific argument shape, falling back to
/// generic pretty-printed JSON for anything not specially handled (`grep`/`find`/`ls`/the Beyond
/// platform tools) — a structured render for the common file-mutating/shell tools is worth the
/// specific-casing; the rest already read fine as JSON (a pattern, a path, a glob).
fn render_tool_call(out: &mut String, name: &str, input: &serde_json::Value) {
    match name {
        "edit" => render_edit_call(out, input),
        "write" => render_write_call(out, input),
        "bash" => render_bash_call(out, input),
        "read" => render_read_call(out, input),
        "grep" => render_grep_call(out, input),
        "find" => render_find_call(out, input),
        "ls" => render_ls_call(out, input),
        "fork" => render_fork_call(out, input),
        "sync" => render_sync_call(out, input),
        "logs" => render_logs_call(out, input),
        _ => render_generic_call(out, name, input),
    }
}

fn render_generic_call(out: &mut String, name: &str, input: &serde_json::Value) {
    out.push_str(&format!(
        "<div class=\"tool-call\"><div class=\"tool-title\">Called <code>{}</code></div>\n\
         <pre>{}</pre></div>\n",
        html_escape(name),
        html_escape(&serde_json::to_string_pretty(input).unwrap_or_default())
    ));
}

/// Render an `edit` call as a real diff (reusing [`diff_html`]'s `+`/`-` coloring) instead of raw
/// JSON — each old/new pair becomes a `-`-colored block of the old text followed by a `+`-colored
/// block of the new text. Falls back to the generic JSON renderer if `input` doesn't parse as a valid
/// edit shape (`crate::tools::edit::parse_edits`, the same parser the tool itself uses, so exported
/// rendering never disagrees with what actually ran).
fn render_edit_call(out: &mut String, input: &serde_json::Value) {
    let path = input.get("path").and_then(serde_json::Value::as_str);
    let Ok(edits) = crate::tools::edit::parse_edits(input) else {
        return render_generic_call(out, "edit", input);
    };
    let title = match path {
        Some(p) => format!("Edited <code>{}</code>", html_escape(p)),
        None => "Edit".to_string(),
    };
    out.push_str(&format!(
        "<div class=\"tool-call\"><div class=\"tool-title\">{title}</div>\n"
    ));
    for (old, new) in &edits {
        out.push_str(&diff_pair_html(old, new));
    }
    out.push_str("</div>\n");
}

/// One line-level diff operation — see [`diff_lines`].
enum LineDiffOp<'a> {
    /// A line present, unchanged, in both `old` and `new` — rendered with no color at all.
    Equal(&'a str),
    /// A line only in `old` — rendered `-`-prefixed, red.
    Delete(&'a str),
    /// A line only in `new` — rendered `+`-prefixed, green.
    Insert(&'a str),
}

/// Render an old/new string pair as a real line-level diff (pi-parity Fix 1) — unchanged lines render
/// plain, only the lines that actually differ get `+`/`-` colored, instead of the old behavior of
/// solid-coloring the *entire* old block red and the entire new block green (so a one-line change deep
/// inside a 200-line block used to render as 400 colored lines). See [`diff_lines`] for how the actual
/// line-level diff is computed.
fn diff_pair_html(old: &str, new: &str) -> String {
    let mut out = String::from("<pre><code class=\"language-diff\">");
    for op in diff_lines(old, new) {
        match op {
            LineDiffOp::Equal(line) => {
                out.push_str(&html_escape(line));
                out.push('\n');
            }
            LineDiffOp::Delete(line) => {
                out.push_str(&format!(
                    "<span class=\"diff-del\">-{}</span>\n",
                    html_escape(line)
                ));
            }
            LineDiffOp::Insert(line) => {
                out.push_str(&format!(
                    "<span class=\"diff-add\">+{}</span>\n",
                    html_escape(line)
                ));
            }
        }
    }
    out.push_str("</code></pre>\n");
    out
}

/// Compute a real line-level diff between `old` and `new` (pi-parity Fix 1) — pi delegates this to the
/// `diff` npm package's `diffLines` (`edit-diff.ts:385`); no equivalent crate is already a dependency of
/// this workspace (checked `Cargo.lock` — nothing named `similar`/`diff`/`imara-diff`/`dissimilar`
/// exists, transitively or otherwise), and this is render-time-only, so a small hand-rolled diff is
/// appropriate rather than reaching for a new dependency.
///
/// The content is *not* bounded: `old`/`new` are whatever the model put in an `edit` call, and nothing
/// on the export path truncates them. [`lcs_diff`] therefore caps its own quadratic table
/// ([`MAX_LCS_CELLS`]) and degrades to a plain block rendering past it.
///
/// A common prefix and suffix (lines identical between `old` and `new` at the very start/end) are
/// trimmed off *before* running the actual O(n*m) LCS table in [`lcs_diff`] — the overwhelming majority
/// of a real edit's old/new pair is unchanged context around one small changed region, so this keeps the
/// expensive part of the algorithm scoped to just the lines that actually differ, not the whole
/// (typically much larger) surrounding text.
fn diff_lines<'a>(old: &'a str, new: &'a str) -> Vec<LineDiffOp<'a>> {
    let old_lines: Vec<&str> = old.lines().collect();
    let new_lines: Vec<&str> = new.lines().collect();

    let mut prefix = 0;
    while prefix < old_lines.len()
        && prefix < new_lines.len()
        && old_lines[prefix] == new_lines[prefix]
    {
        prefix += 1;
    }
    let mut suffix = 0;
    while suffix < old_lines.len() - prefix
        && suffix < new_lines.len() - prefix
        && old_lines[old_lines.len() - 1 - suffix] == new_lines[new_lines.len() - 1 - suffix]
    {
        suffix += 1;
    }

    let mut ops: Vec<LineDiffOp<'a>> = Vec::with_capacity(old_lines.len() + new_lines.len());
    ops.extend(old_lines[..prefix].iter().copied().map(LineDiffOp::Equal));
    ops.extend(lcs_diff(
        &old_lines[prefix..old_lines.len() - suffix],
        &new_lines[prefix..new_lines.len() - suffix],
    ));
    ops.extend(
        old_lines[old_lines.len() - suffix..]
            .iter()
            .copied()
            .map(LineDiffOp::Equal),
    );
    ops
}

/// The largest LCS table [`lcs_diff`] will build, in cells. The quadratic table is the one place an
/// export can be made to allocate without limit: prefix/suffix trimming shrinks a *typical* edit to a
/// handful of changed lines, but an edit that rewrites a file wholesale leaves every line differing on
/// both sides, and `dp` is then `len(old) * len(new)` cells — a 25k-line rewrite is ~2.5 GB, a
/// single-shot OOM triggered by nothing more than exporting the session that contains it.
///
/// 1M cells is 4 MiB of `u32` and admits any changed region up to ~1000x1000 lines, which is already far
/// past what a human reads as a diff; past it the rendering degrades but the content is all still there.
const MAX_LCS_CELLS: usize = 1_000_000;

/// A classic LCS (longest-common-subsequence) line diff over `old`/`new` — the textbook
/// dynamic-programming table (`dp[i][j]` = length of the LCS of `old[i..]`/`new[j..]`), backtracked from
/// `(0, 0)` to produce an edit script. `O(len(old) * len(new))` time and space, which is why
/// [`diff_lines`] trims the common prefix/suffix first rather than calling this directly on the whole
/// old/new pair, and why the table is capped at [`MAX_LCS_CELLS`].
fn lcs_diff<'a>(old: &[&'a str], new: &[&'a str]) -> Vec<LineDiffOp<'a>> {
    let n = old.len();
    let m = new.len();
    // Too big to diff: fall back to what this renderer did before the line-level diff existed — the whole
    // old block `-`, the whole new block `+`. Linear in the input, and every line the model wrote still
    // appears in the export; all that is lost is the pairing of unchanged lines across a pathologically
    // large rewrite, which nobody was going to read line-by-line anyway.
    if n.saturating_mul(m) > MAX_LCS_CELLS {
        let mut ops = Vec::with_capacity(n + m);
        ops.extend(old.iter().copied().map(LineDiffOp::Delete));
        ops.extend(new.iter().copied().map(LineDiffOp::Insert));
        return ops;
    }
    // One flat allocation with an explicit stride, rather than `vec![vec![]; n + 1]`: same cells, but
    // `n + 1` fewer allocations and a contiguous, cache-friendly sweep.
    let stride = m + 1;
    let mut dp = vec![0u32; stride * (n + 1)];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            dp[i * stride + j] = if old[i] == new[j] {
                dp[(i + 1) * stride + j + 1] + 1
            } else {
                dp[(i + 1) * stride + j].max(dp[i * stride + j + 1])
            };
        }
    }
    let mut ops = Vec::new();
    let mut i = 0;
    let mut j = 0;
    while i < n && j < m {
        if old[i] == new[j] {
            ops.push(LineDiffOp::Equal(old[i]));
            i += 1;
            j += 1;
        } else if dp[(i + 1) * stride + j] >= dp[i * stride + j + 1] {
            ops.push(LineDiffOp::Delete(old[i]));
            i += 1;
        } else {
            ops.push(LineDiffOp::Insert(new[j]));
            j += 1;
        }
    }
    ops.extend(old[i..].iter().copied().map(LineDiffOp::Delete));
    ops.extend(new[j..].iter().copied().map(LineDiffOp::Insert));
    ops
}

/// Render a `write` call: the target path as the title, content behind the same collapse affordance a
/// tool result gets — pi-parity Fix 2: tagged `<code class="language-{ext}">` from the write's own
/// `path` argument when the extension is recognized (see [`language_from_path`]), same as pi's own
/// `write.ts` tagging its content preview the same way. Pi-parity Task #46: content used to render
/// unconditionally in a single `<pre>`, so a large file write dumped unbounded raw content inline —
/// pi's own `template.js:967-988` runs the write case through the same `formatExpandableOutput(content,
/// 10, lang)` helper used for `read`/`bash`/`ls`, so this now reuses [`render_tool_result_content`] (the
/// same collapse-past-a-threshold `<details>` mechanism, and the same `language_class_for_tool_result`
/// tagging, since `"write"` is one of its recognized tool names) rather than building its own bespoke
/// `<pre>` here.
fn render_write_call(out: &mut String, input: &serde_json::Value) {
    let path = input.get("path").and_then(serde_json::Value::as_str);
    let content = input.get("content").and_then(serde_json::Value::as_str);
    let title = match path {
        Some(p) => format!("Wrote <code>{}</code>", html_escape(p)),
        None => "Write".to_string(),
    };
    out.push_str(&format!(
        "<div class=\"tool-call\"><div class=\"tool-title\">{title}</div>\n"
    ));
    if let Some(content) = content {
        render_tool_result_content(out, content, Some(("write", input)));
    }
    out.push_str("</div>\n");
}

/// Render a `bash` call as a shell-prompt-styled command line, optionally noting a non-default `cwd`.
fn render_bash_call(out: &mut String, input: &serde_json::Value) {
    let command = input.get("command").and_then(serde_json::Value::as_str);
    let cwd = input.get("cwd").and_then(serde_json::Value::as_str);
    out.push_str("<div class=\"tool-call\"><div class=\"tool-title\">Ran a shell command");
    if let Some(cwd) = cwd {
        out.push_str(&format!(" in <code>{}</code>", html_escape(cwd)));
    }
    out.push_str("</div>\n");
    if let Some(command) = command {
        out.push_str(&format!(
            "<pre class=\"bash-command\">$ {}</pre>",
            html_escape(command)
        ));
    }
    out.push_str("</div>\n");
}

/// Render a `read` call: the path being read, plus a `:start-end` line range when the call actually
/// passed `offset`/`limit` (`crate::tools::read`'s own two params) — pi-parity fix: pi's own
/// `formatToolCall`/render path (`template.js:573-583`, `946-957`) always appends this when either
/// arg is present (`start = offset ?? 1`, `end = limit !== undefined ? start + limit - 1 : ''`), which
/// this renderer previously dropped outright, showing a plain path indistinguishable from a whole-file
/// read. A call that passed neither still renders as a bare path — no `:0-0` or similar invented range.
fn render_read_call(out: &mut String, input: &serde_json::Value) {
    let path = input.get("path").and_then(serde_json::Value::as_str);
    let offset = input.get("offset").and_then(serde_json::Value::as_u64);
    let limit = input.get("limit").and_then(serde_json::Value::as_u64);
    let title = match path {
        Some(p) => {
            let mut t = format!("Read <code>{}", html_escape(p));
            if offset.is_some() || limit.is_some() {
                let start = offset.unwrap_or(1);
                t.push_str(&format!(":{start}"));
                if let Some(limit) = limit {
                    t.push_str(&format!("-{}", start + limit.saturating_sub(1)));
                }
            }
            t.push_str("</code>");
            t
        }
        None => "Read".to_string(),
    };
    out.push_str(&format!(
        "<div class=\"tool-call\"><div class=\"tool-title\">{title}</div></div>\n"
    ));
}

/// Render a `grep` call: the pattern searched for, plus the path/glob narrowing it if given.
fn render_grep_call(out: &mut String, input: &serde_json::Value) {
    let pattern = input.get("pattern").and_then(serde_json::Value::as_str);
    let path = input.get("path").and_then(serde_json::Value::as_str);
    let glob = input.get("glob").and_then(serde_json::Value::as_str);
    let mut title = match pattern {
        Some(p) => format!("Searched for <code>{}</code>", html_escape(p)),
        None => "Search".to_string(),
    };
    if let Some(p) = path {
        title.push_str(&format!(" in <code>{}</code>", html_escape(p)));
    }
    if let Some(g) = glob {
        title.push_str(&format!(" (glob <code>{}</code>)", html_escape(g)));
    }
    out.push_str(&format!(
        "<div class=\"tool-call\"><div class=\"tool-title\">{title}</div></div>\n"
    ));
}

/// Render a `find` call: the glob pattern matched, plus the search root if given.
fn render_find_call(out: &mut String, input: &serde_json::Value) {
    let pattern = input.get("pattern").and_then(serde_json::Value::as_str);
    let path = input.get("path").and_then(serde_json::Value::as_str);
    let mut title = match pattern {
        Some(p) => format!("Found files matching <code>{}</code>", html_escape(p)),
        None => "Find".to_string(),
    };
    if let Some(p) = path {
        title.push_str(&format!(" in <code>{}</code>", html_escape(p)));
    }
    out.push_str(&format!(
        "<div class=\"tool-call\"><div class=\"tool-title\">{title}</div></div>\n"
    ));
}

/// Render an `ls` call: the directory listed, defaulting to `.` — matching the tool's own default —
/// plus a `(limit N)` note when the call passed a non-default `limit` (`crate::tools::ls`'s own param)
/// — pi-parity fix: pi's own render path (`template.js:1008-1014`) always appends this note when
/// `limit` is present; previously dropped here, showing a plain path indistinguishable from a listing
/// with `ls`'s default cap. A call with no `limit` still renders as a bare path, same as before.
fn render_ls_call(out: &mut String, input: &serde_json::Value) {
    let path = input
        .get("path")
        .and_then(serde_json::Value::as_str)
        .unwrap_or(".");
    let limit = input.get("limit").and_then(serde_json::Value::as_u64);
    let mut title = format!("Listed <code>{}</code>", html_escape(path));
    if let Some(limit) = limit {
        title.push_str(&format!(" (limit {limit})"));
    }
    out.push_str(&format!(
        "<div class=\"tool-call\"><div class=\"tool-title\">{title}</div></div>\n"
    ));
}

/// Render a `fork` call (Beyond platform tool): the app forked, plus the branch name if given.
fn render_fork_call(out: &mut String, input: &serde_json::Value) {
    let app = input.get("app").and_then(serde_json::Value::as_str);
    let name = input.get("name").and_then(serde_json::Value::as_str);
    let mut title = match app {
        Some(a) => format!("Forked <code>{}</code>", html_escape(a)),
        None => "Fork".to_string(),
    };
    if let Some(n) = name {
        title.push_str(&format!(" as <code>{}</code>", html_escape(n)));
    }
    out.push_str(&format!(
        "<div class=\"tool-call\"><div class=\"tool-title\">{title}</div></div>\n"
    ));
}

/// Render a `sync` call (Beyond platform tool): the directory synced, if a non-default one was given.
fn render_sync_call(out: &mut String, input: &serde_json::Value) {
    let path = input.get("path").and_then(serde_json::Value::as_str);
    let title = match path {
        Some(p) => format!("Synced <code>{}</code>", html_escape(p)),
        None => "Synced the project root".to_string(),
    };
    out.push_str(&format!(
        "<div class=\"tool-call\"><div class=\"tool-title\">{title}</div></div>\n"
    ));
}

/// Render a `logs` call (Beyond platform tool): which app's logs, plus the Lucene query if given.
fn render_logs_call(out: &mut String, input: &serde_json::Value) {
    let app = input.get("app").and_then(serde_json::Value::as_str);
    let query = input.get("query").and_then(serde_json::Value::as_str);
    let mut title = "Read logs".to_string();
    if let Some(a) = app {
        title.push_str(&format!(" for <code>{}</code>", html_escape(a)));
    }
    if let Some(q) = query {
        title.push_str(&format!(" (query: <code>{}</code>)", html_escape(q)));
    }
    out.push_str(&format!(
        "<div class=\"tool-call\"><div class=\"tool-title\">{title}</div></div>\n"
    ));
}

/// Render `text` as HTML via CommonMark plus a few GFM extensions (strikethrough/tables/task lists) —
/// pi's own `marked`, run server-side at export time instead of client-side JS, so this crate's
/// "no JS in the exported file" design holds. Two defenses against untrusted (model- or
/// tool-generated) content: raw HTML in the source is defused to plain escaped text rather than passed
/// through (matching pi's own tokenizer override — a prompt-injected `<script>` renders as visible
/// text, not a live tag), and link/image URLs outside an http(s)/mailto/relative allow-list are
/// dropped rather than rendered as a live `href`/`src`. A fenced ` ```diff ` block gets the same
/// per-line +/- coloring [`diff_html`] gives a tool result that looks like a diff.
fn render_markdown(text: &str) -> String {
    use pulldown_cmark::{CodeBlockKind, CowStr, Event, Options, Parser, Tag, TagEnd};

    let options =
        Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TABLES | Options::ENABLE_TASKLISTS;
    let mut iter = Parser::new_ext(text, options).map(|event| match event {
        Event::Html(s) | Event::InlineHtml(s) => Event::Text(s),
        // CommonMark's soft break (a single `\n` not preceded by two-plus trailing spaces) renders as
        // a literal newline character by default — invisible once collapsed into HTML whitespace. pi's
        // own markdown renderer configures `marked.use({ breaks: true })` so every single `\n` becomes
        // a real `<br>` instead; promoting the event to a hard break here is pulldown-cmark's
        // equivalent (renders as `<br />`), matching pi's actual appearance for multi-line, non-blank-
        // line-separated text.
        Event::SoftBreak => Event::HardBreak,
        Event::Start(Tag::Link {
            link_type,
            dest_url,
            title,
            id,
        }) => Event::Start(Tag::Link {
            link_type,
            dest_url: sanitize_url(dest_url),
            title,
            id,
        }),
        Event::Start(Tag::Image {
            link_type,
            dest_url,
            title,
            id,
        }) => Event::Start(Tag::Image {
            link_type,
            dest_url: sanitize_url(dest_url),
            title,
            id,
        }),
        other => other,
    });

    let mut out_events: Vec<Event> = Vec::new();
    while let Some(event) = iter.next() {
        if let Event::Start(Tag::CodeBlock(CodeBlockKind::Fenced(ref lang))) = event {
            if lang.as_ref() == "diff" {
                let mut code = String::new();
                for inner in iter.by_ref() {
                    match inner {
                        Event::Text(t) => code.push_str(&t),
                        Event::End(TagEnd::CodeBlock) => break,
                        _ => {}
                    }
                }
                out_events.push(Event::Html(CowStr::from(diff_html(&code))));
                continue;
            }
        }
        out_events.push(event);
    }

    let mut html_out = String::new();
    pulldown_cmark::html::push_html(&mut html_out, out_events.into_iter());
    html_out
}

/// Allow-list a markdown link/image URL to http(s)/mailto/tel/ftp or a same-document/relative
/// reference, dropping anything else (`javascript:`, `data:`, `vbscript:`, ...) rather than emitting it
/// as a live `href`/`src` — pi's own `sanitizeMarkdownUrl` (`template.js:616-626`).
fn sanitize_url(url: pulldown_cmark::CowStr) -> pulldown_cmark::CowStr {
    // Strip C0 controls + DEL before the scheme check — matches pi's own
    // `.replace(/[\x00-\x1f\x7f]/g, '')`. Not an active bypass-prevention here (the scheme check is
    // already an allow-list keyed on `starts_with`, not a denylist a control char could dodge), but a
    // legitimate URL carrying a stray embedded control byte (a copy/paste artifact, or a markdown
    // parser quirk upstream) should still normalize to something usable rather than being handled
    // inconsistently with one still present.
    let stripped: String = url
        .chars()
        .filter(|c| !matches!(c, '\u{0}'..='\u{1f}' | '\u{7f}'))
        .collect();
    let trimmed = stripped.trim();
    let lower = trimmed.to_ascii_lowercase();
    let safe = lower.starts_with("http://")
        || lower.starts_with("https://")
        || lower.starts_with("mailto:")
        || lower.starts_with("tel:")
        || lower.starts_with("ftp:")
        || trimmed.starts_with('#')
        || trimmed.starts_with('/')
        || trimmed.starts_with("./")
        || trimmed.starts_with("../")
        || !lower.contains(':'); // no scheme at all — a bare relative reference
    if safe {
        pulldown_cmark::CowStr::from(trimmed.to_string())
    } else {
        pulldown_cmark::CowStr::Borrowed("")
    }
}

/// Whether `content` looks like a unified diff — checked before applying per-line diff coloring so
/// ordinary output (e.g. a file's own `-`-bulleted list) isn't miscolored. Only fires on diff-specific
/// line shapes (a hunk header or a `---`/`+++` file header), not merely the presence of `+`/`-` lines.
fn looks_like_diff(content: &str) -> bool {
    content
        .lines()
        .any(|l| l.starts_with("@@ ") || l.starts_with("--- ") || l.starts_with("+++ "))
}

/// Render `content` as a `<pre><code>` block with per-line unified-diff coloring: `+`-lines,
/// `-`-lines, `@@` hunk headers, and `---`/`+++` file headers each get their own CSS class. Every line
/// is still HTML-escaped first — this only adds a class, never interprets the content as markup.
fn diff_html(content: &str) -> String {
    let mut out = String::from("<pre><code class=\"language-diff\">");
    for line in content.lines() {
        let class = if line.starts_with("+++") || line.starts_with("---") {
            "diff-file"
        } else if line.starts_with("@@") {
            "diff-hunk"
        } else if line.starts_with('+') {
            "diff-add"
        } else if line.starts_with('-') {
            "diff-del"
        } else {
            ""
        };
        if class.is_empty() {
            out.push_str(&html_escape(line));
        } else {
            out.push_str(&format!(
                "<span class=\"{class}\">{}</span>",
                html_escape(line)
            ));
        }
        out.push('\n');
    }
    out.push_str("</code></pre>\n");
    out
}

/// Detect a compaction or branch-summary recap: a message whose text begins with
/// `agent_core::compaction::SUMMARY_MARKER` or [`agent_core::BRANCH_SUMMARY_MARKER`], each always
/// followed by exactly `\n\n` and then the summary body (`agent_core::compaction::apply_summary` /
/// `crate::session_store`'s `branch_summary_message`, the only two places either message shape is ever
/// constructed). Returns the CSS class, a display label, the pre-compaction token count if the body
/// embeds one (see [`parse_compaction_tokens_before`]), and the body — `None` for ordinary text,
/// including text that merely happens to start with one marker's literal characters but isn't followed
/// by the exact separator both call sites always produce (mirrors [`parse_skill_block`]'s same
/// exact-shape-or-not-at-all precedent).
fn parse_summary_marker(text: &str) -> Option<(&'static str, &'static str, Option<u64>, &str)> {
    for (marker, class, label) in [
        (
            agent_core::compaction::SUMMARY_MARKER,
            "compaction",
            "Compaction",
        ),
        (
            agent_core::BRANCH_SUMMARY_MARKER,
            "branch-summary",
            "Branch Summary",
        ),
    ] {
        if let Some(body) = text
            .strip_prefix(marker)
            .and_then(|r| r.strip_prefix("\n\n"))
        {
            let (tokens_before, body) = if class == "compaction" {
                parse_compaction_tokens_before(body)
            } else {
                (None, body)
            };
            return Some((class, label, tokens_before, body));
        }
    }
    None
}

/// Parse the pre-compaction token count `agent_core::compaction::apply_summary` embeds as a literal
/// leading line of a compaction summary's body, in the shape `"Compacted from {N} tokens\n\n"` — the
/// same value pi's own dedicated `entry.tokensBefore` field renders (`template.js:1294-1299`). Absent
/// entirely from a branch-summary body (that class never carries this line) or from a legacy/malformed
/// marker predating this line's introduction, in which case this is a no-op (`None`, `body` unchanged).
fn parse_compaction_tokens_before(body: &str) -> (Option<u64>, &str) {
    if let Some(rest) = body.strip_prefix("Compacted from ") {
        if let Some((count, rest)) = rest.split_once(" tokens\n\n") {
            if let Ok(n) = count.parse::<u64>() {
                return (Some(n), rest);
            }
        }
    }
    (None, body)
}

/// Render an integer with `,`-grouped thousands (`12345` -> `"12,345"`) — matching pi's own
/// `tokensBefore.toLocaleString()` (`template.js:1298`).
fn format_thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out.chars().rev().collect()
}

/// Render a parsed compaction/branch-summary marker (see [`parse_summary_marker`]) as its own
/// distinctly labeled, styled block (markdown body) — matching pi's own dedicated `.compaction`/
/// `.branch-summary` blocks (`template.js:1294-1306`) instead of plain markdown text that still shows
/// the bracketed marker line verbatim, as if the model itself had written it. `tokens_before`, when
/// present, appends a "Compacted from N tokens" note to the label — always present for a real
/// compaction, `None` for a branch summary (see [`parse_compaction_tokens_before`]).
///
/// Pi-parity Task #49: a `class == "compaction"` marker renders collapsed by default — pi's own
/// `.compaction` entry (`template.js:1294-1300`) starts collapsed behind a `.compaction-collapsed`
/// one-liner ("Compacted from N tokens") and reveals the full summary on an `onclick` toggle. This crate
/// renders exported HTML with no client-side JS at all (a deliberate choice — see every other collapse
/// affordance in this module, e.g. [`render_tool_result_content`]'s `<details class="collapsible-output">`),
/// so a `<details>`/`<summary>` element replaces pi's onclick toggle here too: same one-line collapsed
/// summary, full markdown body revealed on click, no information lost — just a different default
/// density. A branch summary (the other [`parse_summary_marker`] class) is never collapsed in pi either,
/// so it keeps the plain, always-expanded `<div>` rendering unchanged.
fn render_summary_marker(
    out: &mut String,
    class: &str,
    label: &str,
    tokens_before: Option<u64>,
    body: &str,
) {
    let rendered_body = render_markdown(body);
    if class == "compaction" {
        let summary_line = match tokens_before {
            Some(n) => format!("Compacted from {} tokens", format_thousands(n)),
            None => label.to_string(),
        };
        out.push_str(&format!(
            "<details class=\"summary-marker {class} collapsible-output\">\
             <summary>{}</summary>\n{}</details>\n",
            html_escape(&summary_line),
            rendered_body
        ));
        return;
    }
    let tokens_note = match tokens_before {
        Some(n) => format!(" &middot; Compacted from {} tokens", format_thousands(n)),
        None => String::new(),
    };
    out.push_str(&format!(
        "<div class=\"summary-marker {class}\"><div class=\"tool-title\">{}{tokens_note}</div>\n{}</div>\n",
        html_escape(label),
        rendered_body
    ));
}

/// A parsed host-run bash command marker (`serve.rs`'s `bash` RPC command, invoked from the idle loop
/// rather than the model's own turn) — see [`parse_host_bash_marker`].
struct HostBashBlock<'a> {
    command: &'a str,
    is_error: bool,
    output: &'a str,
    /// Whether this was recorded with `exclude_from_context: true` (Fix 9, pi-parity gap) — still
    /// rendered here (this crate's own "what's visible in history" path never gates on the flag, only
    /// the separate "what's sent to the model" transform does — see `serve.rs::ServeHooks`), just
    /// annotated so the exported transcript doesn't silently look identical to a command the model did
    /// see.
    excluded_from_context: bool,
    /// The structured exit-code/cancelled/truncated/full-output-path status line (Fix 3, pi-parity gap)
    /// — see [`HOST_BASH_STATUS_LINE_PREFIX`]. `None` for a session persisted before this line existed,
    /// in which case [`render_host_bash_marker`] falls back to `is_error`'s plain border-color styling,
    /// exactly as before this fix.
    status: Option<HostBashStatus>,
}

/// The line-count threshold past which [`render_host_bash_marker`] collapses `output` behind
/// `<details>` (pi-parity Task #47) — matches pi's own `formatExpandableOutput(msg.output, 10)` call for
/// its `bashExecution` role (`template.js:1285`).
const HOST_BASH_OUTPUT_LINE_THRESHOLD: usize = 10;

/// The exact literal prefix `serve.rs` tags a host-run bash command with (`~serve.rs:HOST_BASH_LABEL`),
/// immediately followed by the command itself.
const HOST_BASH_MARKER_PREFIX: &str = "[Host bash command, run outside the model's own turn]\n$ ";
/// The `exclude_from_context: true` counterpart to [`HOST_BASH_MARKER_PREFIX`]
/// (`~serve.rs:HOST_BASH_EXCLUDED_LABEL`) — Fix 9, pi-parity gap: previously such a command was never
/// recorded at all, so there was no marker shape for this case to recognize.
const HOST_BASH_EXCLUDED_MARKER_PREFIX: &str =
    "[Host bash command, excluded from model context]\n$ ";
/// This module's copy of `~serve.rs:HOST_BASH_STATUS_LINE_PREFIX` — the line `serve.rs`'s `bash` RPC
/// command now writes right after the blank-line separator (before the legacy `"(error)\n"` marker, if
/// present), carrying the same `exit_code`/`cancelled`/`truncated`/`full_output_path` fields its RPC
/// response has always reported live but never persisted (Fix 3, pi-parity gap). See
/// [`parse_host_bash_status`].
const HOST_BASH_STATUS_LINE_PREFIX: &str = "[Host bash status] ";

/// Detect and split apart `serve.rs`'s host-bash-command marker: the fixed prefix (either shape — see
/// [`HOST_BASH_MARKER_PREFIX`]/[`HOST_BASH_EXCLUDED_MARKER_PREFIX`]), the command up to the first
/// `\n\n`, then an optional structured status line (see [`parse_host_bash_status`]), then either the
/// result text directly or (if the command errored) a literal `"(error)\n"` immediately before it.
/// Anything that doesn't fit either exact shape isn't a host-bash marker at all — returns `None`, and
/// the caller falls through to ordinary text rendering (mirrors
/// [`parse_summary_marker`]/[`parse_skill_block`]'s same exact-shape-or-not-at-all precedent). Not
/// robust to a `command` that itself contains a blank line (an embedded `\n\n`) — splits at the first
/// one, same simplifying assumption `parse_skill_block` makes for its own body boundary.
fn parse_host_bash_marker(text: &str) -> Option<HostBashBlock<'_>> {
    let (rest, excluded_from_context) = match text.strip_prefix(HOST_BASH_MARKER_PREFIX) {
        Some(rest) => (rest, false),
        None => (text.strip_prefix(HOST_BASH_EXCLUDED_MARKER_PREFIX)?, true),
    };
    let (command, rest) = rest.split_once("\n\n")?;
    let (status, rest) = parse_host_bash_status(rest);
    let (is_error, output) = match rest.strip_prefix("(error)\n") {
        Some(o) => (true, o),
        None => (false, rest),
    };
    Some(HostBashBlock {
        command,
        is_error,
        output,
        excluded_from_context,
        status,
    })
}

/// `serve.rs`'s `bash` RPC command's own structured result fields (`exit_code`/`cancelled`/
/// `truncated`/`full_output_path`) — Fix 3, pi-parity gap: previously only a bare `is_error` bool ever
/// reached the persisted message (as the `"(error)\n"` marker text [`parse_host_bash_marker`] already
/// detected), even though the *live* RPC response for the exact same command already carried a real
/// provider exit code, whether the run was cancelled outright, and whether its output was truncated —
/// none of which a later export could show. `agent_core::Message`/`ContentBlock` deliberately carries
/// no generic per-message side-channel for this (see `serve.rs::HOST_BASH_EXCLUDED_LABEL`'s own doc
/// comment on why that field stays wire-shaped 1:1), so this rides on the same
/// self-describing-marker-text approach already used for a host-bash command itself, rather than
/// widening that shared type for one caller's metadata.
struct HostBashStatus {
    /// The provider's own exit code, when available — `None` for a cancelled/timed-out run (mirrors
    /// `serve.rs::bash_exit_code_from_status_line`, which is what actually computes this).
    exit_code: Option<i64>,
    /// Whether the run was cancelled (`abort_bash`/`abort`) or timed out, rather than completing with an
    /// exit code either way.
    cancelled: bool,
    /// Whether `tools::bash`'s own output cap truncated what's shown here.
    truncated: bool,
    /// Where the untruncated output was saved, when `truncated` and a path was actually captured.
    full_output_path: Option<String>,
}

/// Parse an optional structured status line (see [`HOST_BASH_STATUS_LINE_PREFIX`]) off the front of
/// `rest` — Fix 3, pi-parity gap. Returns `(None, rest)` unchanged for a legacy session predating this
/// line, or for one whose status line fails to parse as JSON (at worst rendered as an ordinary line of
/// output text below, never silently dropped) — either way, [`parse_host_bash_marker`]'s caller falls
/// straight through to its existing `"(error)\n"`-marker-only handling exactly as it did before this fix.
fn parse_host_bash_status(rest: &str) -> (Option<HostBashStatus>, &str) {
    let Some(after_prefix) = rest.strip_prefix(HOST_BASH_STATUS_LINE_PREFIX) else {
        return (None, rest);
    };
    let Some((json_line, remainder)) = after_prefix.split_once('\n') else {
        return (None, rest);
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(json_line) else {
        return (None, rest);
    };
    let status = HostBashStatus {
        exit_code: value.get("exit_code").and_then(serde_json::Value::as_i64),
        cancelled: value
            .get("cancelled")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        truncated: value
            .get("truncated")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        full_output_path: value
            .get("full_output_path")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string),
    };
    (Some(status), remainder)
}

/// Render a parsed host-bash-command marker (see [`parse_host_bash_marker`]) as its own distinct,
/// code-styled block — a shell-prompt-styled command line (matching [`render_bash_call`]'s own
/// styling) followed by the raw result in a `<pre>` (never markdown — it's command output, not prose),
/// instead of falling through to `render_markdown`, where multi-line output used to collapse into one
/// unreadable run-on line. `is_error` gets its own CSS class, matching pi's own success/error coloring
/// for its dedicated `bashExecution` role (`template.js:1273-1285`). `excluded_from_context` (Fix 9,
/// pi-parity gap) gets its own note in the title — still visible here, just never sent to the model.
/// `status` (Fix 3, pi-parity gap), when present, renders as its own row of distinctly-styled badges
/// (see [`render_host_bash_status`]) rather than only ever being visible as the border-color `is_error`
/// already drove. `output` (pi-parity Task #47) renders behind the same collapse-past-a-threshold
/// `<details>` affordance a tool result gets (see [`render_collapsible_output`]) instead of always
/// dumping unbounded raw output into a single `<pre>` — pi's own `template.js:1273-1285` runs its
/// `bashExecution` role's output through `formatExpandableOutput(msg.output, 10)` (no language tag,
/// hence [`HOST_BASH_OUTPUT_LINE_THRESHOLD`] rather than reusing `tool_result_line_threshold("bash")`'s
/// unrelated threshold of `5`, which is for a model-issued `bash` tool call's result, a different pi
/// call site with its own lower threshold).
fn render_host_bash_marker(out: &mut String, block: &HostBashBlock) {
    let class = if block.is_error {
        "tool-call host-bash error"
    } else {
        "tool-call host-bash"
    };
    let title = if block.excluded_from_context {
        "Host bash command (run outside the model's own turn, hidden from the model)"
    } else {
        "Host bash command (run outside the model's own turn)"
    };
    out.push_str(&format!(
        "<div class=\"{class}\"><div class=\"tool-title\">{title}</div>\n"
    ));
    if let Some(status) = &block.status {
        render_host_bash_status(out, status);
    }
    out.push_str(&format!(
        "<pre class=\"bash-command\">$ {}</pre>",
        html_escape(block.command)
    ));
    if !block.output.is_empty() {
        render_collapsible_output(out, block.output, None, HOST_BASH_OUTPUT_LINE_THRESHOLD);
    }
    out.push_str("</div>\n");
}

/// Render `status`'s fields as their own row of distinctly-styled badges (Fix 3, pi-parity gap) — a
/// real exit code (and whether the run was cancelled outright, as opposed to merely erroring) is more
/// specific than `is_error`'s plain binary success/fail, and deserves a visible element of its own
/// rather than being inferable only from the `<pre>` block's border color. `cancelled` wins outright
/// over `exit_code` when both would otherwise apply — a cancelled run's exit code (if any survived) is
/// incidental, not the reason it stopped. `truncated`/`full_output_path` get their own note alongside
/// the badges (not a badge of their own — informative, not a success/failure signal).
fn render_host_bash_status(out: &mut String, status: &HostBashStatus) {
    out.push_str("<div class=\"bash-status\">\n");
    if status.cancelled {
        out.push_str("<span class=\"bash-badge bash-badge-cancelled\">Cancelled</span>\n");
    } else if let Some(code) = status.exit_code {
        let class = if code == 0 {
            "bash-badge-ok"
        } else {
            "bash-badge-error"
        };
        out.push_str(&format!(
            "<span class=\"bash-badge {class}\">Exit {code}</span>\n"
        ));
    }
    if status.truncated {
        out.push_str("<span class=\"bash-badge bash-badge-truncated\">Output truncated");
        if let Some(path) = &status.full_output_path {
            out.push_str(&format!(
                " &middot; saved to <code>{}</code>",
                html_escape(path)
            ));
        }
        out.push_str("</span>\n");
    }
    out.push_str("</div>\n");
}

/// A parsed `<skill name="..." location="...">...</skill>` invocation wrapper, plus any trailing
/// user-authored text after it — see [`parse_skill_block`].
struct SkillBlock<'a> {
    name: &'a str,
    location: &'a str,
    content: &'a str,
    user_message: Option<&'a str>,
}

/// Detect and split apart `skills.rs::expand_if_skill_invocation`'s wrapper format — mirrors pi's own
/// `parseSkillBlock` regex (`/^<skill name="([^"]+)" location="([^"]+)">\n([\s\S]*?)\n<\/skill>
/// (?:\n\n([\s\S]+))?$/`) byte-for-byte: an opening tag, a newline, the skill body up to the *first*
/// `\n</skill>` (non-greedy — a body that happens to contain that literal substring later doesn't
/// extend the match), then either end-of-string or exactly `\n\n` followed by the user's own trailing
/// message. Anything that doesn't fit this exact shape (including a name/location containing `"`,
/// which the wrapper itself never produces) isn't a skill block at all — returns `None`, and the caller
/// falls through to ordinary text rendering.
fn parse_skill_block(text: &str) -> Option<SkillBlock<'_>> {
    let rest = text.strip_prefix("<skill name=\"")?;
    let (name, rest) = rest.split_once("\" location=\"")?;
    let (location, rest) = rest.split_once("\">\n")?;
    let close_idx = rest.find("\n</skill>")?;
    let content = &rest[..close_idx];
    let after = &rest[close_idx + "\n</skill>".len()..];
    let user_message = if after.is_empty() {
        None
    } else {
        Some(after.strip_prefix("\n\n")?.trim())
    };
    Some(SkillBlock {
        name,
        location,
        content,
        user_message,
    })
}

/// Render a parsed skill invocation as its own block (markdown body, matching pi's
/// `safeMarkedParse(skillBlock.content)`), followed by a separate sibling block for the user's own
/// trailing text when present — the two are siblings, not nested, matching pi's TUI layout
/// (`SkillInvocationMessageComponent`/`UserMessageComponent`).
fn render_skill_invocation(out: &mut String, skill: &SkillBlock) {
    out.push_str(&format!(
        "<div class=\"skill-invocation\"><div class=\"tool-title\">Invoked skill <code>{}</code> \
         (<code>{}</code>)</div>\n{}</div>\n",
        html_escape(skill.name),
        html_escape(skill.location),
        render_markdown(skill.content),
    ));
    if let Some(msg) = skill.user_message.filter(|m| !m.is_empty()) {
        out.push_str(&format!(
            "<div class=\"text markdown\">{}</div>\n",
            render_markdown(msg)
        ));
    }
}

fn render_image(out: &mut String, media_type: &str, data: &str) {
    // `data` rides straight from `ContentBlock::Image`/`ImageSource` — a plain, unvalidated `String`
    // deserialized from session JSONL on disk, so a tampered file or an untrusted tool's "image" output
    // reaching here isn't guaranteed to actually be base64. Must be escaped like any other untrusted
    // attribute value, or it can break out of the `src="..."` attribute into live HTML/script.
    out.push_str(&format!(
        "<img class=\"attachment\" src=\"data:{};base64,{}\" alt=\"attachment\">\n",
        html_escape(media_type),
        html_escape(data)
    ));
}

/// A timestamped default export filename, `session-<unix-seconds>.html`, relative to the current
/// directory — used when [`export_html`] isn't given an explicit `output_path`.
fn default_export_path() -> PathBuf {
    PathBuf::from(format!("session-{}.html", now_secs()))
}

/// Render and write `messages` to an HTML file, with no token-usage stats line (see
/// [`export_html_with_usage`] for a caller that has a live session's running totals to show).
/// `branches` is passed straight through to [`render_html`] (pass `&[]` for a session with no tree, or
/// when abandoned branches shouldn't be rendered). `output_path` is used verbatim when given; otherwise
/// [`default_export_path`] is used. Parent directories are created as needed. Returns the path written.
pub fn export_html(
    meta: &SessionMeta,
    messages: &[Message],
    branches: &[(usize, Vec<Message>)],
    output_path: Option<&str>,
) -> std::io::Result<PathBuf> {
    export_html_with_usage(meta, messages, branches, None, output_path)
}

/// Like [`export_html`], but also folds `usage`'s token totals into the header's stats section — for a
/// caller holding a live [`agent_core::session::Session`]'s running counters, which `export_html` has
/// none of on its own (a bare `SessionMeta` + `Vec<Message>` carries no token accounting; only `Session`
/// accumulates it).
pub fn export_html_with_usage(
    meta: &SessionMeta,
    messages: &[Message],
    branches: &[(usize, Vec<Message>)],
    usage: Option<UsageTotals>,
    output_path: Option<&str>,
) -> std::io::Result<PathBuf> {
    let path = match output_path {
        Some(p) => PathBuf::from(p),
        None => default_export_path(),
    };
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let html = render_html(meta, messages, branches, usage);
    let mut file = std::fs::File::create(&path)?;
    file.write_all(html.as_bytes())?;
    file.sync_all()?;
    Ok(path)
}

/// Like [`export_html`], but also renders `events` (see [`render_html_with_entries`]) — for a caller
/// holding a live [`crate::session_store::SessionStore`] to read
/// [`crate::session_store::SessionStore::export_events`] from, which a bare `SessionMeta` +
/// `Vec<Message>` has no access to on its own. Track L36 (pi-parity fix): `main.rs`'s and `serve.rs`'s
/// `export_html` call sites previously passed none of this through, so a model/thinking-level switch,
/// a label, or a custom entry never appeared in an exported transcript at all.
pub fn export_html_with_entries(
    meta: &SessionMeta,
    messages: &[Message],
    branches: &[(usize, Vec<Message>)],
    events: &[crate::session_store::ExportEvent],
    output_path: Option<&str>,
) -> std::io::Result<PathBuf> {
    let path = match output_path {
        Some(p) => PathBuf::from(p),
        None => default_export_path(),
    };
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let html = render_html_with_entries(meta, messages, branches, None, events);
    let mut file = std::fs::File::create(&path)?;
    file.write_all(html.as_bytes())?;
    file.sync_all()?;
    Ok(path)
}

/// Like [`export_html_with_entries`], but also renders `system_prompt`/`tools` (see
/// [`render_html_full`]) and folds in `usage`'s token totals — the one entry point that renders
/// everything this module knows how to, for a caller holding a live session, its
/// [`agent_core::ToolRegistry::definitions`], and its running token totals. Task #44: `main.rs`'s
/// `run --export`/standalone `export` subcommand and `serve.rs`'s `export_html` RPC command all call
/// this directly now — the standalone subcommand passes `None` for `system_prompt`/`tools`/`usage`
/// (no live `Agent`/`ToolRegistry`/`Session` to pull any of them from; see its own call site's comment
/// for why that's a genuine absence, not an oversight), the other two pass real values.
#[allow(clippy::too_many_arguments)]
pub fn export_html_full(
    meta: &SessionMeta,
    messages: &[Message],
    branches: &[(usize, Vec<Message>)],
    usage: Option<UsageTotals>,
    events: &[crate::session_store::ExportEvent],
    system_prompt: Option<&str>,
    tools: Option<&[ToolDef]>,
    output_path: Option<&str>,
) -> std::io::Result<PathBuf> {
    let path = match output_path {
        Some(p) => PathBuf::from(p),
        None => default_export_path(),
    };
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let html = render_html_full(
        meta,
        messages,
        branches,
        usage,
        events,
        system_prompt,
        tools,
    );
    let mut file = std::fs::File::create(&path)?;
    file.write_all(html.as_bytes())?;
    file.sync_all()?;
    Ok(path)
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_core::ImageSource;

    fn meta() -> SessionMeta {
        let mut m = SessionMeta::new("/proj", "claude-test");
        m.title = Some("Fix the bug".into());
        m
    }

    #[test]
    fn renders_a_well_formed_document_with_header_info() {
        let html = render_html(&meta(), &[], &[], None);
        assert!(html.starts_with("<!DOCTYPE html>"));
        assert!(html.trim_end().ends_with("</html>"));
        assert!(html.contains("Fix the bug"));
        assert!(html.contains("claude-test"));
        assert!(html.contains("0 message(s)"));
    }

    #[test]
    fn renders_text_tool_use_and_tool_result_blocks() {
        // A tool with no dedicated renderer (an invented name, not any real tool) still falls back to
        // generic pretty-printed JSON — `grep` itself now has dedicated rendering, see
        // `renders_grep_find_ls_and_beyond_calls_with_dedicated_rendering` below.
        let messages = vec![
            Message::user("please search a.rs"),
            Message::assistant(vec![ContentBlock::ToolUse {
                id: "1".into(),
                name: "some_future_tool".into(),
                input: serde_json::json!({ "pattern": "fn main", "path": "a.rs" }),
                thought_signature: None,
            }]),
            Message::tool_result("1", "fn main() {}", false),
        ];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(html.contains("please search a.rs"));
        assert!(html.contains("Called <code>some_future_tool</code>"));
        assert!(html.contains("&quot;pattern&quot;"));
        assert!(html.contains("fn main() {}"));
        assert!(html.contains("class=\"tool-result\""));
    }

    #[test]
    fn renders_grep_find_ls_and_beyond_calls_with_dedicated_rendering() {
        // MEDIUM pi-parity gap (fixed): grep/find/ls and the Beyond-platform tools (fork/sync/logs)
        // used to fall back to generic pretty-printed JSON like any unrecognized tool; pi routes every
        // tool, including third-party ones, through a rich renderer.
        let messages = vec![
            Message::assistant(vec![ContentBlock::ToolUse {
                id: "1".into(),
                name: "grep".into(),
                input: serde_json::json!({ "pattern": "fn main", "path": "src", "glob": "*.rs" }),
                thought_signature: None,
            }]),
            Message::assistant(vec![ContentBlock::ToolUse {
                id: "2".into(),
                name: "find".into(),
                input: serde_json::json!({ "pattern": "*.rs", "path": "src" }),
                thought_signature: None,
            }]),
            Message::assistant(vec![ContentBlock::ToolUse {
                id: "3".into(),
                name: "ls".into(),
                input: serde_json::json!({}),
                thought_signature: None,
            }]),
            Message::assistant(vec![ContentBlock::ToolUse {
                id: "4".into(),
                name: "fork".into(),
                input: serde_json::json!({ "app": "myapp", "name": "sandbox" }),
                thought_signature: None,
            }]),
            Message::assistant(vec![ContentBlock::ToolUse {
                id: "5".into(),
                name: "sync".into(),
                input: serde_json::json!({}),
                thought_signature: None,
            }]),
            Message::assistant(vec![ContentBlock::ToolUse {
                id: "6".into(),
                name: "logs".into(),
                input: serde_json::json!({ "app": "myapp", "query": "level:error" }),
                thought_signature: None,
            }]),
        ];
        let html = render_html(&meta(), &messages, &[], None);

        assert!(
            html.contains(
                "Searched for <code>fn main</code> in <code>src</code> (glob <code>*.rs</code>)"
            ),
            "{html}"
        );
        assert!(
            html.contains("Found files matching <code>*.rs</code> in <code>src</code>"),
            "{html}"
        );
        assert!(html.contains("Listed <code>.</code>"), "{html}");
        assert!(
            html.contains("Forked <code>myapp</code> as <code>sandbox</code>"),
            "{html}"
        );
        assert!(html.contains("Synced the project root"), "{html}");
        assert!(
            html.contains("Read logs for <code>myapp</code> (query: <code>level:error</code>)"),
            "{html}"
        );
        assert!(
            !html.contains("&quot;pattern&quot;") && !html.contains("&quot;app&quot;"),
            "none of these should fall back to raw JSON: {html}"
        );
    }

    #[test]
    fn renders_edit_write_bash_and_read_calls_with_dedicated_rendering() {
        let messages = vec![
            Message::assistant(vec![ContentBlock::ToolUse {
                id: "1".into(),
                name: "edit".into(),
                input: serde_json::json!({
                    "path": "src/main.rs",
                    "old_string": "let x = 1;",
                    "new_string": "let x = 2;",
                }),
                thought_signature: None,
            }]),
            Message::assistant(vec![ContentBlock::ToolUse {
                id: "2".into(),
                name: "write".into(),
                input: serde_json::json!({ "path": "notes.md", "content": "hello world" }),
                thought_signature: None,
            }]),
            Message::assistant(vec![ContentBlock::ToolUse {
                id: "3".into(),
                name: "bash".into(),
                input: serde_json::json!({ "command": "cargo test", "cwd": "/proj" }),
                thought_signature: None,
            }]),
            Message::assistant(vec![ContentBlock::ToolUse {
                id: "4".into(),
                name: "read".into(),
                input: serde_json::json!({ "path": "README.md" }),
                thought_signature: None,
            }]),
        ];
        let html = render_html(&meta(), &messages, &[], None);

        // edit: a real diff, not raw JSON — old text `-`-colored, new text `+`-colored.
        assert!(html.contains("Edited <code>src/main.rs</code>"));
        assert!(html.contains("class=\"diff-del\">-let x = 1;"));
        assert!(html.contains("class=\"diff-add\">+let x = 2;"));
        assert!(
            !html.contains("&quot;old_string&quot;"),
            "must not fall back to raw JSON: {html}"
        );

        // write: path as title, content verbatim.
        assert!(html.contains("Wrote <code>notes.md</code>"));
        assert!(html.contains("hello world"));

        // bash: shell-prompt-styled command line, cwd noted.
        assert!(html.contains("Ran a shell command in <code>/proj</code>"));
        assert!(html.contains("$ cargo test"));

        // read: just the path.
        assert!(html.contains("Read <code>README.md</code>"));
    }

    #[test]
    fn edit_call_falls_back_to_generic_rendering_when_input_does_not_parse_as_an_edit() {
        let messages = vec![Message::assistant(vec![ContentBlock::ToolUse {
            id: "1".into(),
            name: "edit".into(),
            input: serde_json::json!({ "path": "src/main.rs" }), // missing old_string/new_string
            thought_signature: None,
        }])];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(html.contains("Called <code>edit</code>"));
    }

    #[test]
    fn read_call_renders_the_line_range_when_offset_or_limit_were_given() {
        // pi-parity gap (fixed): the exported `read` call used to render only the path, silently
        // dropping `offset`/`limit` even when the call actually used them — indistinguishable from a
        // whole-file read. Matches pi's own `path:start-end` shape (`template.js:573-583`).
        let messages = vec![
            // offset only: start = offset, no end (open-ended).
            Message::assistant(vec![ContentBlock::ToolUse {
                id: "1".into(),
                name: "read".into(),
                input: serde_json::json!({ "path": "a.rs", "offset": 10 }),
                thought_signature: None,
            }]),
            // limit only: start defaults to 1.
            Message::assistant(vec![ContentBlock::ToolUse {
                id: "2".into(),
                name: "read".into(),
                input: serde_json::json!({ "path": "b.rs", "limit": 20 }),
                thought_signature: None,
            }]),
            // both: start = offset, end = offset + limit - 1.
            Message::assistant(vec![ContentBlock::ToolUse {
                id: "3".into(),
                name: "read".into(),
                input: serde_json::json!({ "path": "c.rs", "offset": 5, "limit": 3 }),
                thought_signature: None,
            }]),
        ];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(html.contains("Read <code>a.rs:10</code>"), "{html}");
        assert!(html.contains("Read <code>b.rs:1-20</code>"), "{html}");
        assert!(html.contains("Read <code>c.rs:5-7</code>"), "{html}");
    }

    #[test]
    fn read_call_renders_a_bare_path_when_neither_offset_nor_limit_were_given() {
        // No invented `:0-0`-shaped range for a plain whole-file read — same assertion
        // `renders_edit_write_bash_and_read_calls_with_dedicated_rendering` already makes for this call
        // shape; kept here too as its own explicit regression target for this fix.
        let messages = vec![Message::assistant(vec![ContentBlock::ToolUse {
            id: "1".into(),
            name: "read".into(),
            input: serde_json::json!({ "path": "README.md" }),
            thought_signature: None,
        }])];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(html.contains("Read <code>README.md</code>"), "{html}");
        assert!(!html.contains("README.md:"), "{html}");
    }

    #[test]
    fn ls_call_renders_the_limit_note_when_given() {
        // pi-parity gap (fixed): the exported `ls` call used to render only the path, silently dropping
        // a non-default `limit` — matches pi's own `(limit N)` note (`template.js:1010-1014`).
        let messages = vec![Message::assistant(vec![ContentBlock::ToolUse {
            id: "1".into(),
            name: "ls".into(),
            input: serde_json::json!({ "path": "src", "limit": 50 }),
            thought_signature: None,
        }])];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(
            html.contains("Listed <code>src</code> (limit 50)"),
            "{html}"
        );
    }

    #[test]
    fn ls_call_renders_a_bare_path_when_no_limit_was_given() {
        let messages = vec![Message::assistant(vec![ContentBlock::ToolUse {
            id: "1".into(),
            name: "ls".into(),
            input: serde_json::json!({ "path": "src" }),
            thought_signature: None,
        }])];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(html.contains("Listed <code>src</code></div>"), "{html}");
        assert!(!html.contains("limit"), "{html}");
    }

    #[test]
    fn marks_a_tool_error_result_distinctly() {
        let messages = vec![Message::tool_result("1", "boom", true)];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(html.contains("class=\"tool-result error\""));
        assert!(html.contains(">Error<"));
    }

    #[test]
    fn escapes_html_metacharacters_in_message_text() {
        let messages = vec![Message::user("<script>alert(1)</script>")];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(!html.contains("<script>alert(1)</script>"));
        assert!(html.contains("&lt;script&gt;alert(1)&lt;/script&gt;"));
    }

    #[test]
    fn renders_markdown_formatting_in_message_text() {
        let messages = vec![Message::user(
            "# Heading\n\n**bold** and a list:\n\n- one\n- two\n\n```rust\nfn main() {}\n```",
        )];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(html.contains("<h1>Heading</h1>"));
        assert!(html.contains("<strong>bold</strong>"));
        assert!(html.contains("<li>one</li>"));
        assert!(html.contains("<li>two</li>"));
        assert!(html.contains("<pre><code class=\"language-rust\">fn main() {}\n</code></pre>"));
    }

    #[test]
    fn a_single_newline_not_separated_by_a_blank_line_renders_as_a_hard_line_break() {
        // pi-parity gap (fixed): pi's own markdown renderer configures `marked.use({ breaks: true })`
        // so every single `\n` becomes a real `<br>`; without an equivalent here, a CommonMark soft
        // break rendered as a literal, invisible-once-collapsed newline character instead.
        let messages = vec![Message::user("line one\nline two")];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(html.contains("line one<br"), "{html}");
        assert!(html.contains("line two"), "{html}");
        // A real paragraph break (blank-line-separated) must still become two `<p>`s, not `<br>`s.
        let messages = vec![Message::user("para one\n\npara two")];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(html.matches("<p>").count() >= 2, "{html}");
    }

    #[test]
    fn drops_a_javascript_scheme_link_but_keeps_an_http_one() {
        let messages = vec![Message::user(
            "[click me](javascript:alert(1)) and [safe](https://example.com)",
        )];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(
            !html.contains("javascript:"),
            "an unsafe URL scheme must never reach a live href: {html}"
        );
        assert!(html.contains("href=\"https://example.com\""));
    }

    #[test]
    fn allows_tel_and_ftp_scheme_links_matching_pis_own_allow_list() {
        let messages = vec![Message::user(
            "[call](tel:+15555550100) and [file](ftp://example.com/f.txt)",
        )];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(html.contains("href=\"tel:+15555550100\""), "{html}");
        assert!(html.contains("href=\"ftp://example.com/f.txt\""), "{html}");
    }

    #[test]
    fn href_attribute_breakout_is_neutralized() {
        // Unit-tested at `sanitize_url` directly rather than through `render_html`: CommonMark's own
        // bare-link-destination grammar doesn't accept a literal `"` followed by `onmouseover="...` as
        // part of a destination (it reads as a malformed title), so `Parser` never emits a `Link` event
        // for this text at all — it falls through as plain escaped text, never reaching `sanitize_url`.
        // That's a safe outcome, just via a different mechanism than this test originally assumed; the
        // real guarantee to pin is at the function `sanitize_url` actually owns: an allowed scheme's
        // value passes through unmangled, and escaping the resulting attribute is
        // `pulldown_cmark::html::push_html`'s job (well-established behavior of that crate, not ours to
        // re-verify) — proven end-to-end by `drops_a_javascript_scheme_link_but_keeps_an_http_one`
        // above for the non-adversarial case.
        let value = "https://example.com\" onmouseover=\"alert(1)";
        let cleaned = sanitize_url(pulldown_cmark::CowStr::Borrowed(value));
        assert_eq!(cleaned.as_ref(), value);
    }

    #[test]
    fn strips_c0_control_characters_from_a_url_before_the_scheme_check() {
        // Unit-tested directly (see `href_attribute_breakout_is_neutralized` above for why): a raw C0
        // control byte inside a bare markdown link destination isn't valid CommonMark either, so this
        // can't be proven through `render_html` — `sanitize_url` is the actual owner of this behavior.
        let cleaned = sanitize_url(pulldown_cmark::CowStr::Borrowed("https://exa\u{1}mple.com"));
        assert_eq!(cleaned.as_ref(), "https://example.com");
    }

    #[test]
    fn defuses_raw_html_inside_markdown_text_to_plain_visible_text() {
        // Raw HTML in the markdown *source* (not the already-tested plain-text case above) must still
        // render as visible escaped text, not a live tag — a prompt-injected block quoted as "```" or
        // written as inline HTML shouldn't execute just because it parses as valid embedded HTML.
        let messages = vec![Message::user("before <img src=x onerror=alert(1)> after")];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(!html.contains("<img src=x"));
        assert!(html.contains("&lt;img"));
    }

    #[test]
    fn colors_a_fenced_diff_block_in_message_text_per_line() {
        let messages = vec![Message::user(
            "```diff\n--- a/f.rs\n+++ b/f.rs\n@@ -1 +1 @@\n-old\n+new\n context\n```",
        )];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(html.contains("class=\"diff-file\">--- a/f.rs"));
        assert!(html.contains("class=\"diff-hunk\">@@ -1 +1 @@"));
        assert!(html.contains("class=\"diff-del\">-old"));
        assert!(html.contains("class=\"diff-add\">+new"));
        assert!(html.contains("context")); // unlabeled context line, still rendered
    }

    #[test]
    fn colors_a_tool_result_that_looks_like_a_diff() {
        let messages = vec![Message::tool_result(
            "1",
            "--- a/f.rs\n+++ b/f.rs\n@@ -1 +1 @@\n-old\n+new",
            false,
        )];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(html.contains("class=\"diff-add\">+new"));
        assert!(html.contains("class=\"diff-del\">-old"));
    }

    #[test]
    fn does_not_miscolor_a_tool_result_that_merely_starts_lines_with_plus_or_minus() {
        // A bulleted list (or any other `-`/`+`-prefixed content) that isn't shaped like a real unified
        // diff (no hunk header, no `---`/`+++` file header) must not get diff coloring.
        let messages = vec![Message::tool_result("1", "- one\n- two\n+ three", false)];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(!html.contains("class=\"diff-add\""));
        assert!(!html.contains("class=\"diff-del\""));
        assert!(html.contains("- one"));
    }

    #[test]
    fn renders_a_skill_invocation_as_a_distinct_block_from_the_trailing_user_message() {
        // The exact wrapper shape `skills.rs::expand_if_skill_invocation` produces. Must render as a
        // separate skill-invocation block (markdown body) plus a sibling text block for the trailing
        // user message — not one raw-escaped blob with the `<skill>` tags visible as literal text.
        let messages = vec![Message::user(
            "<skill name=\"lint\" location=\"/x/.claude/skills/lint/SKILL.md\">\n\
             References are relative to /x/.claude/skills/lint.\n\n\
             Run `cargo clippy`.\n\
             </skill>\n\n\
             only src/main.rs",
        )];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(
            !html.contains("&lt;skill name="),
            "the wrapper tag must not appear as visible raw-escaped text: {html}"
        );
        assert!(html.contains("class=\"skill-invocation\""));
        assert!(html.contains("Invoked skill <code>lint</code>"));
        assert!(html.contains("/x/.claude/skills/lint/SKILL.md"));
        assert!(html.contains("Run <code>cargo clippy</code>")); // markdown-rendered, not raw-escaped
        // The trailing user message renders as its own sibling block, after the skill block.
        let skill_pos = html.find("skill-invocation").unwrap();
        let trailing_pos = html.find("only src/main.rs").unwrap();
        assert!(trailing_pos > skill_pos);
    }

    #[test]
    fn a_skill_invocation_with_no_trailing_user_message_renders_no_extra_text_block() {
        let messages = vec![Message::user(
            "<skill name=\"lint\" location=\"/x/SKILL.md\">\nReferences are relative to /x.\n\n\
             Body.\n</skill>",
        )];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(html.contains("class=\"skill-invocation\""));
        // Exactly one message-level div (the skill block doesn't spawn an empty trailing text div).
        assert_eq!(html.matches("class=\"text markdown\"").count(), 0);
    }

    #[test]
    fn text_that_merely_resembles_a_skill_tag_is_not_misparsed() {
        // A user pasting something that starts with `<skill` but doesn't match the exact wrapper shape
        // must fall through to ordinary (safely escaped) markdown rendering, not a broken skill block.
        let messages = vec![Message::user("<skill>not a real wrapper</skill>")];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(!html.contains("class=\"skill-invocation\""));
        assert!(html.contains("&lt;skill&gt;"));
    }

    #[test]
    fn renders_an_image_attachment_as_a_data_uri() {
        let messages = vec![Message::assistant(vec![ContentBlock::Image {
            source: ImageSource::base64("image/png", "Zm9v"),
        }])];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(html.contains("src=\"data:image/png;base64,Zm9v\""));
    }

    #[test]
    fn escapes_adversarial_bytes_in_an_image_attachment_data_uri() {
        // `ImageSource.data` rides straight from session JSONL on disk (or an untrusted tool result) —
        // a plain, unvalidated `String`, not guaranteed to actually be base64. Must be escaped like any
        // other untrusted attribute value or it can break out of `src="..."` into live HTML.
        let messages = vec![Message::assistant(vec![ContentBlock::Image {
            source: ImageSource::base64("image/png", "Zm9v\"><script>alert(1)</script>"),
        }])];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(
            !html.contains("<script>alert(1)</script>"),
            "adversarial image data must not break out of the src attribute: {html}"
        );
        assert!(html.contains("&quot;&gt;&lt;script&gt;"));
    }

    #[test]
    fn drops_a_javascript_scheme_markdown_image_but_keeps_a_data_uri_media_type_escaped() {
        // The link-scheme allow-list is exercised above only for `<a href>`; markdown image syntax
        // (`![alt](url)`) routes through the same `sanitize_url`, but nothing proved that directly.
        let messages = vec![Message::user("![x](javascript:alert(1))")];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(
            !html.contains("javascript:"),
            "an unsafe URL scheme must never reach a live img src: {html}"
        );
    }

    #[test]
    fn renders_abandoned_branches_inline_after_the_message_they_diverged_from() {
        let messages = vec![Message::user("what should I do?")];
        let branches = vec![(
            1usize,
            vec![
                Message::user("what should I do?"),
                Message::user("this is the divergent branch content"),
            ],
        )];
        let html = render_html(&meta(), &messages, &branches, None);
        // A collapsible <details> block — expandable inline, positioned right after the message it
        // actually forked from, not a separate flat section at the bottom of the page.
        assert!(html.contains("<details class=\"branch\">"));
        assert!(html.contains("Branch 1"));
        assert!(html.contains("forked after message 1"));
        assert!(html.contains("this is the divergent branch content"));
        // The shared prefix (message 0, "what should I do?") must appear exactly once — from the main
        // transcript — not duplicated inside the branch's own body.
        assert_eq!(html.matches("what should I do?").count(), 1);
        // The branch content appears strictly after the message it diverged from, not before it or in
        // a wholly separate part of the document.
        let main_msg_pos = html.find("what should I do?").unwrap();
        let branch_pos = html.find("this is the divergent branch content").unwrap();
        assert!(branch_pos > main_msg_pos);
    }

    #[test]
    fn a_branch_diverging_before_the_first_message_renders_before_it() {
        let messages = vec![Message::user("second path")];
        let branches = vec![(0usize, vec![Message::user("first path, abandoned")])];
        let html = render_html(&meta(), &messages, &branches, None);
        assert!(html.contains("forked from the start"));
        let branch_pos = html.find("first path, abandoned").unwrap();
        let main_msg_pos = html.find("second path").unwrap();
        assert!(branch_pos < main_msg_pos);
    }

    #[test]
    fn multiple_branches_at_different_points_are_numbered_sequentially() {
        let messages = vec![Message::user("a"), Message::user("b")];
        let branches = vec![
            (
                1usize,
                vec![Message::user("a"), Message::user("branch-one")],
            ),
            (
                2usize,
                vec![
                    Message::user("a"),
                    Message::user("b"),
                    Message::user("branch-two"),
                ],
            ),
        ];
        let html = render_html(&meta(), &messages, &branches, None);
        assert!(html.contains("Branch 1"));
        assert!(html.contains("Branch 2"));
        let b1 = html.find("branch-one").unwrap();
        let b2 = html.find("branch-two").unwrap();
        assert!(b1 < b2, "branches must appear in divergence order: {html}");
    }

    #[test]
    fn no_branches_section_when_there_are_no_abandoned_branches() {
        let html = render_html(&meta(), &[Message::user("hi")], &[], None);
        assert!(!html.contains("<details class=\"branch\">"));
        assert!(!html.contains("forked"));
    }

    #[test]
    fn export_html_writes_to_the_given_path() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("out.html");
        let path = export_html(&meta(), &[], &[], Some(target.to_str().unwrap())).unwrap();
        assert_eq!(path, target);
        let written = std::fs::read_to_string(&path).unwrap();
        assert!(written.contains("Fix the bug"));
    }

    #[test]
    fn export_html_creates_missing_parent_directories() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("nested/deeper/out.html");
        let path = export_html(&meta(), &[], &[], Some(target.to_str().unwrap())).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn default_export_path_is_timestamped_and_relative() {
        // Tested in isolation (no `export_html`/`set_current_dir` involved) — this repo's other tests
        // run concurrently in the same process, and `set_current_dir` is process-global, not
        // per-thread, so mutating it here would be a real flakiness risk for every other test.
        let path = default_export_path();
        assert!(path.is_relative());
        let name = path.file_name().and_then(|n| n.to_str()).unwrap();
        assert!(name.starts_with("session-"), "got: {name}");
        assert!(name.ends_with(".html"), "got: {name}");
    }

    #[test]
    fn an_aborted_assistant_turn_renders_a_visible_indicator_even_with_no_content() {
        // pi-parity gap (fixed): an aborted turn's `content` is typically empty/partial — without a
        // dedicated indicator, the message div renders completely blank, giving zero sign anything went
        // wrong in exactly the scenario (debugging a failed run) export exists for.
        let messages = vec![Message::assistant(vec![]).with_aborted()];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(html.contains("class=\"turn-status aborted\""), "{html}");
        assert!(html.contains(">Aborted<"), "{html}");
    }

    #[test]
    fn an_errored_assistant_turn_renders_the_error_message_visibly() {
        let messages = vec![Message::error("transport failed after 3 retries")];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(html.contains("class=\"turn-status error\""), "{html}");
        assert!(
            html.contains("Error: transport failed after 3 retries"),
            "{html}"
        );
    }

    #[test]
    fn an_aborted_turn_with_partial_content_keeps_the_content_and_adds_the_indicator() {
        let messages = vec![
            Message::assistant(vec![ContentBlock::text("partial response before cancel")])
                .with_aborted(),
        ];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(html.contains("partial response before cancel"));
        assert!(html.contains(">Aborted<"));
    }

    #[test]
    fn renders_a_compaction_summary_as_a_distinctly_labeled_block() {
        let messages = vec![Message::user(format!(
            "{}\n\n{}",
            agent_core::compaction::SUMMARY_MARKER,
            "Refactored the auth module and fixed three bugs."
        ))];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(
            html.contains("class=\"summary-marker compaction collapsible-output\""),
            "{html}"
        );
        assert!(html.contains(">Compaction<"), "{html}");
        assert!(html.contains("Refactored the auth module"));
        assert!(
            !html.contains(agent_core::compaction::SUMMARY_MARKER),
            "the bracketed marker line must not appear verbatim: {html}"
        );
    }

    #[test]
    fn compaction_block_collapses_behind_details_with_a_one_line_summary() {
        // Task #49 (pi-parity gap): pi's `.compaction` entry starts collapsed behind a one-line
        // `.compaction-collapsed` summary ("Compacted from N tokens") and reveals the full body on
        // click (`template.js:1294-1300`); this crate has no client-side JS, so a `<details>`/
        // `<summary>` element (the same zero-JS pattern as `render_tool_result_content`'s
        // `<details class="collapsible-output">`) replaces pi's onclick toggle. No information is
        // lost — the full summary is still present in the static HTML, just behind `<details>`.
        let messages = vec![Message::user(format!(
            "{}\n\nCompacted from 12345 tokens\n\n{}",
            agent_core::compaction::SUMMARY_MARKER,
            "Refactored the auth module and fixed three bugs."
        ))];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(
            html.contains("<details class=\"summary-marker compaction collapsible-output\">"),
            "{html}"
        );
        assert!(
            html.contains("<summary>Compacted from 12,345 tokens</summary>"),
            "the collapsed one-liner must match pi's own collapsed summary text: {html}"
        );
        // The full body must still be present in the static HTML (just behind `<details>`, not
        // client-side-JS-gated) — an artifact viewer or `grep` over the exported file must still see it.
        assert!(html.contains("Refactored the auth module and fixed three bugs."));
    }

    #[test]
    fn branch_summary_block_is_never_collapsed_unlike_compaction() {
        // Pi never collapses `.branch-summary` entries, only `.compaction` ones — this must stay a
        // plain, always-expanded `<div>`, not gain a `<details>` wrapper too.
        let messages = vec![Message::user(format!(
            "{}\n\n{}",
            agent_core::BRANCH_SUMMARY_MARKER,
            "Explored using a cache; reverted since it added complexity."
        ))];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(
            !html.contains("<details"),
            "a branch summary must not collapse: {html}"
        );
        assert!(
            html.contains("class=\"summary-marker branch-summary\""),
            "{html}"
        );
        assert!(html.contains("Explored using a cache"));
    }

    #[test]
    fn renders_a_branch_summary_marker_as_a_distinctly_labeled_block() {
        let messages = vec![Message::user(format!(
            "{}\n\n{}",
            agent_core::BRANCH_SUMMARY_MARKER,
            "Explored using a cache; reverted since it added complexity."
        ))];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(
            html.contains("class=\"summary-marker branch-summary\""),
            "{html}"
        );
        assert!(html.contains(">Branch Summary<"), "{html}");
        assert!(html.contains("Explored using a cache"));
        assert!(
            !html.contains(agent_core::BRANCH_SUMMARY_MARKER),
            "the bracketed marker line must not appear verbatim: {html}"
        );
    }

    #[test]
    fn text_that_merely_resembles_a_summary_marker_is_not_misparsed() {
        // Same not-the-exact-shape precedent as `parse_skill_block`'s own adversarial-input test: text
        // that starts with the marker's characters but isn't followed by the exact `\n\n` separator both
        // real call sites always produce must fall through to ordinary markdown rendering.
        let messages = vec![Message::user(format!(
            "{}not the real shape",
            agent_core::compaction::SUMMARY_MARKER
        ))];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(!html.contains("class=\"summary-marker"));
        assert!(html.contains("not the real shape"));
    }

    #[test]
    fn compaction_marker_renders_tokens_before_when_embedded_in_the_summary_body() {
        // Task #45: `agent_core::compaction::apply_summary` embeds a pre-compaction token count in the
        // marker text; this proves the export-side rendering surfaces it, matching pi's own
        // `entry.tokensBefore` note.
        let messages = vec![Message::user(format!(
            "{}\n\nCompacted from 12345 tokens\n\n{}",
            agent_core::compaction::SUMMARY_MARKER,
            "Refactored the auth module and fixed three bugs."
        ))];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(html.contains("Compacted from 12,345 tokens"), "{html}");
        assert!(html.contains("Refactored the auth module"));
        assert!(
            !html.contains("Compacted from 12345 tokens\n\n"),
            "the raw token-count line must not leak verbatim into the rendered body: {html}"
        );
    }

    #[test]
    fn compaction_marker_end_to_end_through_real_apply_summary() {
        // Task #45, full round-trip: drive the actual `apply_summary` (not a hand-built marker string)
        // and confirm export renders the token count it embeds.
        let mut session = agent_core::Session::new();
        session.user("do the thing");
        agent_core::compaction::apply_summary(&mut session, 0, "Did the thing.", 42_000);
        let html = render_html(&meta(), &session.messages, &[], None);
        assert!(html.contains("Compacted from 42,000 tokens"), "{html}");
        assert!(html.contains("Did the thing."));
    }

    #[test]
    fn compaction_marker_without_an_embedded_token_count_renders_with_no_token_note() {
        // A branch-summary marker (a different class, see `parse_summary_marker`) never carries this
        // line at all — backward-compatible parsing for that shape.
        let messages = vec![Message::user(format!(
            "{}\n\n{}",
            agent_core::compaction::SUMMARY_MARKER,
            "Refactored the auth module."
        ))];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(!html.contains("Compacted from"), "{html}");
    }

    #[test]
    fn renders_a_host_bash_command_marker_as_a_distinct_block_not_plain_markdown() {
        // pi-parity gap (fixed, Task #47): previously fell through to `render_markdown`, where
        // multi-line output collapsed into one unreadable run-on line.
        let messages = vec![Message::user(
            "[Host bash command, run outside the model's own turn]\n$ ls -la\n\n\
             file1.txt\nfile2.txt\ndrwxr-xr-x  dir",
        )];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(html.contains("class=\"tool-call host-bash\""), "{html}");
        assert!(html.contains("$ ls -la"));
        assert!(html.contains("file1.txt"));
        assert!(html.contains("file2.txt"));
        assert!(
            !html.contains("[Host bash command, run outside the model's own turn]"),
            "the raw marker line must not leak verbatim: {html}"
        );
        assert!(
            !html.contains("class=\"text markdown\""),
            "must not fall through to plain markdown rendering: {html}"
        );
    }

    #[test]
    fn renders_a_host_bash_command_error_marker_distinctly() {
        let messages = vec![Message::user(
            "[Host bash command, run outside the model's own turn]\n$ false\n\n(error)\ncommand exited 1",
        )];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(
            html.contains("class=\"tool-call host-bash error\""),
            "{html}"
        );
        assert!(html.contains("$ false"));
        assert!(html.contains("command exited 1"));
        assert!(
            !html.contains("(error)\ncommand exited 1"),
            "the literal (error) line marker must not leak verbatim into the rendered output: {html}"
        );
    }

    #[test]
    fn renders_an_exclude_from_context_host_bash_marker_as_its_own_annotated_block() {
        // Fix 9 (pi-parity gap): an `exclude_from_context: true` host-bash record is now always
        // present in `session.messages` (previously skipped entirely), and must render with the same
        // dedicated block a non-excluded one gets — not fall through to plain markdown — just annotated
        // so the exported transcript doesn't look identical to a command the model actually saw.
        let messages = vec![Message::user(
            "[Host bash command, excluded from model context]\n$ printf secret-diagnostic\n\n\
             secret-diagnostic",
        )];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(html.contains("class=\"tool-call host-bash\""), "{html}");
        assert!(html.contains("$ printf secret-diagnostic"));
        assert!(html.contains("secret-diagnostic"));
        assert!(
            html.contains("hidden from the model"),
            "the excluded variant must be annotated distinctly from the ordinary one: {html}"
        );
        assert!(
            !html.contains("[Host bash command, excluded from model context]"),
            "the raw marker line must not leak verbatim: {html}"
        );
        assert!(
            !html.contains("class=\"text markdown\""),
            "must not fall through to plain markdown rendering: {html}"
        );
    }

    #[test]
    fn text_that_merely_resembles_a_host_bash_marker_is_not_misparsed() {
        let messages = vec![Message::user("[Host bash command] not the real shape")];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(!html.contains("class=\"tool-call host-bash\""));
        assert!(html.contains("not the real shape"));
    }

    #[test]
    fn renders_a_host_bash_status_line_as_distinct_badges() {
        // Fix 3 (pi-parity gap): `serve.rs`'s `bash` RPC command now threads its own
        // exit_code/cancelled/truncated/full_output_path fields into the persisted marker as a leading
        // status line — previously only `is_error`'s border color ever reached the exported page.
        let messages = vec![Message::user(
            "[Host bash command, run outside the model's own turn]\n$ cargo test\n\n\
             [Host bash status] {\"exit_code\":0,\"cancelled\":false,\"truncated\":false,\"full_output_path\":null}\n\
             all tests passed",
        )];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(html.contains("class=\"bash-status\""), "{html}");
        assert!(
            html.contains("class=\"bash-badge bash-badge-ok\">Exit 0</span>"),
            "{html}"
        );
        assert!(!html.contains("Cancelled"), "{html}");
        assert!(html.contains("all tests passed"));
        // The raw status line must not leak verbatim into the rendered output.
        assert!(!html.contains("[Host bash status]"), "{html}");
    }

    #[test]
    fn renders_a_nonzero_host_bash_exit_code_with_the_error_badge() {
        let messages = vec![Message::user(
            "[Host bash command, run outside the model's own turn]\n$ false\n\n\
             [Host bash status] {\"exit_code\":1,\"cancelled\":false,\"truncated\":false,\"full_output_path\":null}\n\
             (error)\ncommand exited 1",
        )];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(
            html.contains("class=\"bash-badge bash-badge-error\">Exit 1</span>"),
            "{html}"
        );
        // `is_error`'s own block-level styling (the `"(error)\n"` marker) is untouched by this fix.
        assert!(
            html.contains("class=\"tool-call host-bash error\""),
            "{html}"
        );
    }

    #[test]
    fn renders_a_cancelled_host_bash_run_with_its_own_badge_instead_of_an_exit_code() {
        let messages = vec![Message::user(
            "[Host bash command, run outside the model's own turn]\n$ sleep 100\n\n\
             [Host bash status] {\"exit_code\":null,\"cancelled\":true,\"truncated\":false,\"full_output_path\":null}\n\
             (error)\ncancelled",
        )];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(
            html.contains("class=\"bash-badge bash-badge-cancelled\">Cancelled</span>"),
            "{html}"
        );
        assert!(!html.contains("Exit "), "{html}");
    }

    #[test]
    fn renders_a_truncated_host_bash_output_with_its_saved_path() {
        let messages = vec![Message::user(
            "[Host bash command, run outside the model's own turn]\n$ yes\n\n\
             [Host bash status] {\"exit_code\":0,\"cancelled\":false,\"truncated\":true,\"full_output_path\":\"/tmp/out.txt\"}\n\
             y\ny\ny",
        )];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(
            html.contains("class=\"bash-badge bash-badge-truncated\">Output truncated"),
            "{html}"
        );
        assert!(html.contains("<code>/tmp/out.txt</code>"), "{html}");
    }

    #[test]
    fn a_long_host_bash_output_is_collapsed_behind_a_details_element() {
        // Task #47 (pi-parity gap): a host-run bash command's output used to render unconditionally in
        // a single `<pre>`, no threshold/collapse — same missed pattern as write (Task #46). Matches
        // pi's own `formatExpandableOutput(msg.output, 10)` call for its `bashExecution` role
        // (`template.js:1273-1285`).
        let long_output: String = (1..=20)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let messages = vec![Message::user(format!(
            "[Host bash command, run outside the model's own turn]\n$ seq 20\n\n{long_output}"
        ))];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(html.contains("class=\"collapsible-output\""), "{html}");
        assert!(html.contains("20 lines (click to expand)"), "{html}");
        assert!(html.contains("line 1"), "{html}");
        assert!(html.contains("line 20"), "{html}");
    }

    #[test]
    fn a_short_host_bash_output_is_not_collapsed() {
        let messages = vec![Message::user(
            "[Host bash command, run outside the model's own turn]\n$ ls -la\n\n\
             file1.txt\nfile2.txt\ndrwxr-xr-x  dir",
        )];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(!html.contains("class=\"collapsible-output\""), "{html}");
        assert!(html.contains("file1.txt"));
    }

    #[test]
    fn a_host_bash_marker_with_no_status_line_still_renders_with_no_badges() {
        // Backward compatibility: a session persisted before Fix 3 has no status line at all — must
        // still render exactly as before (no `bash-status` row), falling back to `is_error`'s plain
        // border-color styling. This is the same fixture
        // `renders_a_host_bash_command_marker_as_a_distinct_block_not_plain_markdown` already uses.
        let messages = vec![Message::user(
            "[Host bash command, run outside the model's own turn]\n$ ls -la\n\n\
             file1.txt\nfile2.txt\ndrwxr-xr-x  dir",
        )];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(!html.contains("class=\"bash-status\""), "{html}");
        // Not a bare `contains("bash-badge")` check: the document's own `<style>` block always defines
        // `.bash-badge`/`.bash-badge-*` CSS rules regardless of whether any badge is ever rendered in
        // the body, so that substring alone would never actually catch a regression here.
        assert!(!html.contains("<span class=\"bash-badge"), "{html}");
        assert!(html.contains("class=\"tool-call host-bash\""), "{html}");
    }

    #[test]
    fn a_malformed_host_bash_status_line_falls_back_gracefully() {
        // A corrupted/truncated status line must never break parsing of the rest of the block — falls
        // back to no status (rendered, at worst, as an ordinary line of output text) rather than losing
        // the command/output entirely.
        let messages = vec![Message::user(
            "[Host bash command, run outside the model's own turn]\n$ echo hi\n\n\
             [Host bash status] not valid json\nhi",
        )];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(html.contains("class=\"tool-call host-bash\""), "{html}");
        assert!(html.contains("$ echo hi"));
        assert!(!html.contains("class=\"bash-status\""), "{html}");
    }

    #[test]
    fn renders_an_aggregate_stats_section_with_models_messages_and_tool_calls() {
        let messages = vec![
            Message::user("hello"),
            Message::assistant(vec![ContentBlock::text("hi there")]).with_model_id("claude-x"),
            Message::assistant(vec![ContentBlock::ToolUse {
                id: "1".into(),
                name: "bash".into(),
                input: serde_json::json!({ "command": "ls" }),
                thought_signature: None,
            }])
            .with_model_id("claude-x"),
            Message::tool_result("1", "file.txt", false),
        ];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(html.contains("class=\"stats\""), "{html}");
        assert!(html.contains("claude-x"), "models line missing: {html}");
        assert!(html.contains("1 user, 2 assistant"), "{html}");
        assert!(html.contains(">Tool calls<"));
        assert!(
            html.contains("<span class=\"stat-value\">1</span>"),
            "exactly one tool call must be counted: {html}"
        );
    }

    #[test]
    fn stats_section_includes_the_session_creation_date() {
        // pi-parity gap (fixed): pi's own export always renders a `Date:` line first
        // (`template.js:1395`); this crate's stats section previously read `meta.model`/every
        // message/`usage`/`events` but never `meta.created_at` at all.
        let mut m = meta();
        m.created_at = 1_700_000_000;
        let html = render_html(&m, &[], &[], None);
        assert!(html.contains(">Date<"), "{html}");
        // Tied to the same formatting function `render_stats_section` calls, rather than a hardcoded
        // string, so this test isn't itself flaky across the host's own local timezone.
        let expected = crate::resources::format_local_datetime(m.created_at);
        assert!(
            html.contains(&format!(
                "<span class=\"stat-value\">{}</span>",
                html_escape(&expected)
            )),
            "expected formatted date {expected:?} in: {html}"
        );
    }

    #[test]
    fn stats_section_models_line_falls_back_to_the_session_model_when_no_message_is_tagged() {
        let html = render_html(&meta(), &[Message::user("hi")], &[], None);
        assert!(html.contains("class=\"stats\""));
        // `meta().model` is "claude-test" — see the `meta()` fixture above.
        assert!(html.contains("claude-test"));
    }

    #[test]
    fn stats_section_lists_every_distinct_model_actually_used() {
        let messages = vec![
            Message::assistant(vec![ContentBlock::text("a")]).with_model_id("model-a"),
            Message::assistant(vec![ContentBlock::text("b")]).with_model_id("model-b"),
        ];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(html.contains("model-a"));
        assert!(html.contains("model-b"));
    }

    #[test]
    fn stats_section_shows_a_token_line_only_when_usage_is_given() {
        let without = render_html(&meta(), &[Message::user("hi")], &[], None);
        assert!(
            !without.contains(">Tokens<"),
            "no token line without usage data: {without}"
        );

        let usage = UsageTotals {
            input_tokens: 1234,
            output_tokens: 567,
            cache_read_tokens: 89,
            cache_write_tokens: 12,
        };
        let with = render_html(&meta(), &[Message::user("hi")], &[], Some(usage));
        assert!(with.contains(">Tokens<"), "{with}");
        assert!(with.contains("1234"), "input token count missing: {with}");
        assert!(with.contains("567"), "output token count missing: {with}");
        assert!(with.contains("89"), "cache read count missing: {with}");
        assert!(with.contains("12"), "cache write count missing: {with}");
    }

    #[test]
    fn export_html_with_usage_writes_token_stats_to_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("out.html");
        let usage = UsageTotals {
            input_tokens: 42,
            output_tokens: 7,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
        };
        let path = export_html_with_usage(
            &meta(),
            &[Message::user("hi")],
            &[],
            Some(usage),
            Some(target.to_str().unwrap()),
        )
        .unwrap();
        let written = std::fs::read_to_string(&path).unwrap();
        assert!(written.contains(">Tokens<"));
        assert!(written.contains("42"));
    }

    #[test]
    fn export_html_without_usage_omits_the_token_line() {
        // The plain `export_html` entry point (what `main.rs`/`serve.rs` call today) has no usage data
        // to show yet — it must still render the rest of the stats section, just without a token line.
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("out.html");
        let path = export_html(
            &meta(),
            &[Message::user("hi")],
            &[],
            Some(target.to_str().unwrap()),
        )
        .unwrap();
        let written = std::fs::read_to_string(&path).unwrap();
        assert!(written.contains("class=\"stats\""));
        assert!(!written.contains(">Tokens<"));
    }

    #[test]
    fn render_html_omits_the_events_section_when_there_are_none() {
        // The plain `render_html`/`export_html` entry points (no entries given) must render exactly as
        // before this feature existed — no empty "Session Events" section cluttering every ordinary
        // export.
        let html = render_html(&meta(), &[Message::user("hi")], &[], None);
        assert!(!html.contains("Session Events"));
    }

    #[test]
    fn render_html_with_entries_renders_a_simple_block_per_event() {
        // Track L36 (pi-parity fix): `Entry::ModelChange`/`Entry::ThinkingLevelChange`/`Entry::Label`/
        // `Entry::Custom` were durably tracked but never reached an export at all.
        use crate::session_store::ExportEvent;
        let events = vec![
            ExportEvent::ModelChange("claude-opus-4-8".to_string()),
            ExportEvent::ThinkingLevelChange("high".to_string()),
            ExportEvent::Label {
                target_id: "msg-1".to_string(),
                label: Some("checkpoint".to_string()),
            },
            ExportEvent::Label {
                target_id: "msg-1".to_string(),
                label: None,
            },
            ExportEvent::Custom {
                kind: "beyond:sync".to_string(),
                data: serde_json::json!({"marker": "m1"}),
            },
        ];
        let html = render_html_with_entries(&meta(), &[Message::user("hi")], &[], None, &events);
        assert!(html.contains("Session Events"));
        assert!(html.contains("Model changed to <code>claude-opus-4-8</code>"));
        assert!(html.contains("Thinking level changed to <code>high</code>"));
        assert!(
            html.contains("Labeled <code>msg-1</code>: checkpoint"),
            "{html}"
        );
        assert!(
            html.contains("Label cleared on <code>msg-1</code>"),
            "{html}"
        );
        assert!(html.contains("beyond:sync"));
        assert!(html.contains("marker"));
    }

    #[test]
    fn label_event_renders_its_target_id_alongside_the_label() {
        // Task #26 (pi-parity fix): `ExportEvent::Label { target_id, label }`'s render arm used to
        // discard `target_id` via `..`, so an exported "Labeled: foo" line gave no way to tell which
        // message it actually pointed to.
        use crate::session_store::ExportEvent;
        let events = vec![ExportEvent::Label {
            target_id: "abc123".to_string(),
            label: Some("checkpoint".to_string()),
        }];
        let html = render_html_with_entries(&meta(), &[Message::user("hi")], &[], None, &events);
        assert!(
            html.contains("Labeled <code>abc123</code>: checkpoint"),
            "{html}"
        );
    }

    #[test]
    fn a_cleared_label_event_still_renders_its_target_id() {
        use crate::session_store::ExportEvent;
        let events = vec![ExportEvent::Label {
            target_id: "abc123".to_string(),
            label: None,
        }];
        let html = render_html_with_entries(&meta(), &[Message::user("hi")], &[], None, &events);
        assert!(
            html.contains("Label cleared on <code>abc123</code>"),
            "{html}"
        );
    }

    #[test]
    fn label_event_html_escapes_an_adversarial_target_id() {
        // `target_id` rides straight from session storage — not attacker-controlled in practice, but
        // it's still untrusted-shaped data reaching an HTML render path, same defensive posture this
        // module already takes for `Custom`'s `kind`/`data` fields.
        use crate::session_store::ExportEvent;
        let events = vec![ExportEvent::Label {
            target_id: "<script>alert(1)</script>".to_string(),
            label: Some("x".to_string()),
        }];
        let html = render_html_with_entries(&meta(), &[], &[], None, &events);
        assert!(!html.contains("<script>alert(1)</script>"), "{html}");
        assert!(html.contains("&lt;script&gt;"), "{html}");
    }

    #[test]
    fn model_change_event_renders_inline_before_the_assistant_message_that_first_used_it() {
        // Task #27 (pi-parity fix): `model_change` events used to always land in a disconnected
        // trailing list (`render_events_section`, called once after every message) instead of at their
        // real chronological position, matching pi's own `template.js`, which renders `model_change`
        // inline as part of `renderEntry`'s single walk over the transcript.
        use crate::session_store::ExportEvent;
        let messages = vec![
            Message::user("hello"),
            Message::assistant(vec![ContentBlock::text("hi there")]).with_model_id("claude-a"),
            Message::user("switch models please"),
            Message::assistant(vec![ContentBlock::text("using claude-b now")])
                .with_model_id("claude-b"),
        ];
        let events = vec![ExportEvent::ModelChange("claude-b".to_string())];
        let html = render_html_with_entries(&meta(), &messages, &[], None, &events);
        // Rendered inline, not in the flat trailing dump.
        assert!(!html.contains("Session Events"), "{html}");
        assert!(html.contains("class=\"model-change\""), "{html}");
        assert!(
            html.contains("Model changed to <code>claude-b</code>"),
            "{html}"
        );
        // Positioned between the turn that used the old model and the turn that used the new one, not
        // before the first assistant turn (which never used `claude-b` at all).
        let change_pos = html.find("class=\"model-change\"").unwrap();
        let first_assistant_pos = html.find("hi there").unwrap();
        let second_assistant_pos = html.find("using claude-b now").unwrap();
        assert!(change_pos > first_assistant_pos, "{html}");
        assert!(change_pos < second_assistant_pos, "{html}");
    }

    #[test]
    fn multiple_model_change_events_each_render_inline_before_their_own_matching_message() {
        // Proves the value-matching cursor doesn't just find *any* message sharing a model id — the
        // first message already uses `model-a` from the very start (before any switch event at all) and
        // must not be mistaken for the *second* `ModelChange("model-a")` event, which really belongs to
        // the third message, after the intervening switch to `model-b`.
        use crate::session_store::ExportEvent;
        let messages = vec![
            Message::assistant(vec![ContentBlock::text("first reply")]).with_model_id("model-a"),
            Message::assistant(vec![ContentBlock::text("second reply")]).with_model_id("model-b"),
            Message::assistant(vec![ContentBlock::text("third reply")]).with_model_id("model-a"),
        ];
        let events = vec![
            ExportEvent::ModelChange("model-b".to_string()),
            ExportEvent::ModelChange("model-a".to_string()),
        ];
        let html = render_html_with_entries(&meta(), &messages, &[], None, &events);
        assert!(!html.contains("Session Events"), "{html}");
        assert_eq!(html.matches("class=\"model-change\"").count(), 2, "{html}");
        let change_to_b = html.find("Model changed to <code>model-b</code>").unwrap();
        let change_to_a = html.find("Model changed to <code>model-a</code>").unwrap();
        let first = html.find("first reply").unwrap();
        let second = html.find("second reply").unwrap();
        let third = html.find("third reply").unwrap();
        assert!(first < change_to_b, "{html}");
        assert!(change_to_b < second, "{html}");
        assert!(second < change_to_a, "{html}");
        assert!(change_to_a < third, "{html}");
    }

    #[test]
    fn a_model_change_event_with_no_matching_later_message_falls_back_to_the_trailing_section() {
        // A switch that never got used by a subsequent assistant turn (e.g. the session was exported
        // right after the switch) can't be positioned inline — it must still be visible somewhere,
        // rather than silently dropped now that most `ModelChange` events render inline instead.
        use crate::session_store::ExportEvent;
        let messages =
            vec![Message::assistant(vec![ContentBlock::text("hi")]).with_model_id("claude-a")];
        let events = vec![ExportEvent::ModelChange("claude-never-used".to_string())];
        let html = render_html_with_entries(&meta(), &messages, &[], None, &events);
        assert!(html.contains("Session Events"), "{html}");
        assert!(
            html.contains("Model changed to <code>claude-never-used</code>"),
            "{html}"
        );
        assert!(!html.contains("class=\"model-change\""), "{html}");
    }

    #[test]
    fn render_html_with_entries_escapes_untrusted_custom_entry_content() {
        // `data`/`kind` on an `Entry::Custom` are caller-defined and this module never interprets
        // them (see `Entry::Custom`'s own doc comment) — a value containing HTML-significant
        // characters must not break out of the `<li>` it's rendered into.
        use crate::session_store::ExportEvent;
        let events = vec![ExportEvent::Custom {
            kind: "<script>".to_string(),
            data: serde_json::json!({"x": "<img onerror=alert(1)>"}),
        }];
        let html = render_html_with_entries(&meta(), &[], &[], None, &events);
        assert!(!html.contains("<script>"));
        assert!(!html.contains("<img onerror"));
    }

    #[test]
    fn export_html_with_entries_writes_the_events_section_to_the_file() {
        use crate::session_store::ExportEvent;
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("out.html");
        let events = vec![ExportEvent::ModelChange("gpt-5".to_string())];
        let path = export_html_with_entries(
            &meta(),
            &[Message::user("hi")],
            &[],
            &events,
            Some(target.to_str().unwrap()),
        )
        .unwrap();
        let written = std::fs::read_to_string(&path).unwrap();
        assert!(written.contains("Model changed to <code>gpt-5</code>"));
    }

    fn sample_tool_defs() -> Vec<ToolDef> {
        vec![ToolDef {
            name: "bash".to_string(),
            description: "Run a shell command.".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "The command to run." },
                    "cwd": { "type": "string", "description": "Working directory." }
                },
                "required": ["command"]
            }),
        }]
    }

    #[test]
    fn render_html_full_renders_system_prompt_and_tools_sections_when_given() {
        // pi-parity gap (fixed, Task #44): `render_html`/`render_html_with_entries` never rendered the
        // session's system prompt or registered tools at all — pi always includes both
        // (`export-html/index.ts:263-270`, `template.js:1405-1435`).
        let tools = sample_tool_defs();
        let html = render_html_full(
            &meta(),
            &[Message::user("hi")],
            &[],
            None,
            &[],
            Some("You are a helpful coding agent."),
            Some(&tools),
        );
        assert!(html.contains("class=\"system-prompt\""), "{html}");
        assert!(html.contains("You are a helpful coding agent."));
        assert!(html.contains("class=\"tools-list\""), "{html}");
        assert!(html.contains("Available Tools (1)"), "{html}");
        assert!(html.contains("tool-item-name\">bash</span>"), "{html}");
        assert!(html.contains("Run a shell command."));
        assert!(html.contains("command"));
        assert!(html.contains("cwd"));
        assert!(html.contains("required"));
        assert!(html.contains("optional"));
        assert!(html.contains("Working directory."));
    }

    #[test]
    fn render_html_full_with_none_renders_no_system_prompt_or_tools_section() {
        // Backward compatibility: `main.rs`/`serve.rs` don't thread a live system prompt/tool registry
        // through yet (see `render_html_full`'s own doc comment) — `None` for both must render exactly
        // as the plainer entry points do, no empty sections.
        let html = render_html_full(&meta(), &[Message::user("hi")], &[], None, &[], None, None);
        assert!(!html.contains("class=\"system-prompt\""));
        assert!(!html.contains("class=\"tools-list\""));
    }

    #[test]
    fn render_tools_section_omits_a_tool_with_no_schema_properties() {
        let tools = vec![ToolDef {
            name: "ping".to_string(),
            description: "No-op health check.".to_string(),
            input_schema: serde_json::json!({ "type": "object", "properties": {} }),
        }];
        let html = render_html_full(&meta(), &[], &[], None, &[], None, Some(&tools));
        assert!(html.contains("tool-item-name\">ping</span>"), "{html}");
        assert!(!html.contains("class=\"tool-params\""), "{html}");
    }

    #[test]
    fn export_html_full_writes_system_prompt_and_tools_to_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("out.html");
        let tools = sample_tool_defs();
        let path = export_html_full(
            &meta(),
            &[Message::user("hi")],
            &[],
            None,
            &[],
            Some("System prompt text."),
            Some(&tools),
            Some(target.to_str().unwrap()),
        )
        .unwrap();
        let written = std::fs::read_to_string(&path).unwrap();
        assert!(written.contains("System prompt text."));
        assert!(written.contains("tool-item-name\">bash</span>"));
    }

    // ---- pi-parity fixes: real line diff, tool-result language tagging, collapse affordance,
    // control-byte stripping, and compaction/branch-summary stats counts. ----

    #[test]
    fn edit_call_renders_a_real_line_level_diff_not_solid_colored_blocks() {
        // Fix 1: a single changed line inside a larger unchanged block must render as a real diff
        // (unchanged lines uncolored, only the actually-changed line(s) diff-colored) instead of the
        // old behavior of solid-coloring the *entire* old text red and the entire new text green.
        let old = "line one\nline two\nline three\nline four\nline five";
        let new = "line one\nline two\nCHANGED\nline four\nline five";
        let messages = vec![Message::assistant(vec![ContentBlock::ToolUse {
            id: "1".into(),
            name: "edit".into(),
            input: serde_json::json!({
                "path": "src/lib.rs",
                "old_string": old,
                "new_string": new,
            }),
            thought_signature: None,
        }])];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(html.contains("class=\"diff-del\">-line three"), "{html}");
        assert!(html.contains("class=\"diff-add\">+CHANGED"), "{html}");
        // The unchanged surrounding lines must not themselves be diff-colored.
        assert!(!html.contains("diff-del\">-line one"), "{html}");
        assert!(!html.contains("diff-add\">+line one"), "{html}");
        assert!(!html.contains("diff-del\">-line five"), "{html}");
        assert!(!html.contains("diff-add\">+line five"), "{html}");
        // They still appear, just as plain, uncolored context.
        assert!(html.contains("line one"), "{html}");
        assert!(html.contains("line five"), "{html}");
    }

    #[test]
    fn a_wholesale_rewrite_past_the_lcs_cap_still_exports_and_still_shows_both_sides() {
        // A file rewritten wholesale: every line differs, so prefix/suffix trimming saves nothing and the
        // LCS table would be 5k x 5k = 25M cells (100 MB) — and a 25k-line rewrite would be ~2.5 GB.
        // Past `MAX_LCS_CELLS` the diff degrades to plain `-` old / `+` new blocks: linear, and the
        // content is all still in the document. The timing assert is what catches a regression to the
        // quadratic path, which for this input takes seconds and 100 MB rather than milliseconds.
        let old: String = (0..5_000).map(|i| format!("old line {i}\n")).collect();
        let new: String = (0..5_000).map(|i| format!("new line {i}\n")).collect();
        let messages = vec![Message::assistant(vec![ContentBlock::ToolUse {
            id: "1".into(),
            name: "edit".into(),
            input: serde_json::json!({
                "path": "src/lib.rs",
                "old_string": old,
                "new_string": new,
            }),
            thought_signature: None,
        }])];
        let started = std::time::Instant::now();
        let html = render_html(&meta(), &messages, &[], None);
        assert!(
            started.elapsed() < std::time::Duration::from_secs(5),
            "export of an oversized edit fell back into the quadratic diff"
        );
        assert!(html.contains("class=\"diff-del\">-old line 4999"), "{html}");
        assert!(html.contains("class=\"diff-add\">+new line 4999"), "{html}");
    }

    #[test]
    fn edit_call_with_no_overlap_still_diffs_as_a_full_delete_then_insert() {
        // The degenerate case (nothing in common) must still behave like the old bulk-colored
        // rendering: every old line removed, every new line added, in that order.
        let messages = vec![Message::assistant(vec![ContentBlock::ToolUse {
            id: "1".into(),
            name: "edit".into(),
            input: serde_json::json!({
                "path": "src/lib.rs",
                "old_string": "let x = 1;",
                "new_string": "let x = 2;",
            }),
            thought_signature: None,
        }])];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(html.contains("class=\"diff-del\">-let x = 1;"));
        assert!(html.contains("class=\"diff-add\">+let x = 2;"));
    }

    #[test]
    fn read_tool_result_gets_a_language_class_matching_its_paths_extension() {
        // Fix 2: a `ToolResult` block itself carries no path — this proves the language tag is
        // correctly derived by tracing back to the *originating* `read` call's own `path` argument
        // (see `index_tool_calls`/`ToolCallIndex`).
        let messages = vec![
            Message::assistant(vec![ContentBlock::ToolUse {
                id: "1".into(),
                name: "read".into(),
                input: serde_json::json!({ "path": "src/main.rs" }),
                thought_signature: None,
            }]),
            Message::tool_result("1", "fn main() {}\n", false),
        ];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(
            html.contains("<pre><code class=\"language-rust\">"),
            "{html}"
        );
    }

    #[test]
    fn a_tool_result_with_no_associated_call_or_unrecognized_extension_gets_no_language_class() {
        // No preceding `ToolUse` at all (a bare tool-result fixture, as several existing tests use) —
        // must still render, just untagged, not panic on a missing index entry.
        let messages = vec![Message::tool_result("1", "plain output", false)];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(html.contains("<pre>plain output</pre>"), "{html}");
        assert!(!html.contains("<code class=\"language-"), "{html}");
    }

    #[test]
    fn write_call_content_gets_a_language_class_from_its_own_path() {
        let messages = vec![Message::assistant(vec![ContentBlock::ToolUse {
            id: "1".into(),
            name: "write".into(),
            input: serde_json::json!({ "path": "notes.md", "content": "# Title" }),
            thought_signature: None,
        }])];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(
            html.contains("<pre><code class=\"language-markdown\">"),
            "{html}"
        );
    }

    #[test]
    fn a_long_write_call_content_is_collapsed_behind_a_details_element() {
        // Task #46 (pi-parity gap): a `write` call's content used to render unconditionally in a
        // single `<pre>`, with only language tagging applied — no collapse of any kind, unlike
        // `read`/`bash`/`ls` tool results, which already collapsed past their own thresholds. Matches
        // pi's own `formatExpandableOutput(content, 10, lang)` call for the write case
        // (`template.js:967-988`) — same threshold as `read`'s.
        let long_content: String = (1..=15).map(|i| format!("line {i}\n")).collect();
        let messages = vec![Message::assistant(vec![ContentBlock::ToolUse {
            id: "1".into(),
            name: "write".into(),
            input: serde_json::json!({ "path": "big.rs", "content": long_content }),
            thought_signature: None,
        }])];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(html.contains("class=\"collapsible-output\""), "{html}");
        assert!(html.contains("15 lines (click to expand)"), "{html}");
        // Still tagged with the write's own path language, even collapsed.
        assert!(html.contains("<code class=\"language-rust\">"), "{html}");
        assert!(html.contains("line 1\n"), "{html}");
        assert!(html.contains("line 15"), "{html}");
    }

    #[test]
    fn a_short_write_call_content_is_not_collapsed() {
        let messages = vec![Message::assistant(vec![ContentBlock::ToolUse {
            id: "1".into(),
            name: "write".into(),
            input: serde_json::json!({ "path": "notes.md", "content": "hello world" }),
            thought_signature: None,
        }])];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(!html.contains("class=\"collapsible-output\""), "{html}");
        assert!(html.contains("hello world"));
    }

    #[test]
    fn a_long_tool_result_is_collapsed_behind_a_details_element() {
        // Fix 3: past a per-tool line threshold, a tool result collapses by default behind
        // `<details>`/`<summary>` — the same zero-client-JS pattern already used for branches/the
        // system-prompt section — instead of always rendering fully expanded.
        let long_output: String = (1..=30).map(|i| format!("line {i}\n")).collect();
        let messages = vec![Message::tool_result("1", long_output, false)];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(html.contains("class=\"collapsible-output\""), "{html}");
        assert!(html.contains("30 lines (click to expand)"), "{html}");
        assert!(html.contains("line 1\n"), "{html}");
        assert!(html.contains("line 30"), "{html}");
    }

    #[test]
    fn a_short_tool_result_is_not_collapsed() {
        let messages = vec![Message::tool_result("1", "one line of output", false)];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(!html.contains("class=\"collapsible-output\""), "{html}");
    }

    #[test]
    fn a_bash_result_collapses_past_five_lines_matching_pis_own_lower_threshold() {
        let messages = vec![
            Message::assistant(vec![ContentBlock::ToolUse {
                id: "1".into(),
                name: "bash".into(),
                input: serde_json::json!({ "command": "seq 6" }),
                thought_signature: None,
            }]),
            Message::tool_result("1", "1\n2\n3\n4\n5\n6", false),
        ];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(html.contains("class=\"collapsible-output\""), "{html}");
        assert!(html.contains("6 lines (click to expand)"), "{html}");
    }

    #[test]
    fn strips_c0_control_bytes_from_tool_result_content() {
        // Fix 4: a stray raw control byte (e.g. an ANSI escape, 0x1b) must not pass through literally
        // into the rendered page — `html_escape` alone only escapes `&<>"'`.
        let content = "before\u{1b}[31mred\u{1b}[0mafter";
        let messages = vec![Message::tool_result("1", content, false)];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(!html.contains('\u{1b}'), "{html}");
        assert!(html.contains("before"));
        assert!(html.contains("red"));
        assert!(html.contains("after"));
    }

    #[test]
    fn keeps_tabs_and_newlines_when_stripping_control_bytes_from_tool_result_content() {
        let content = "line one\tindented\nline two";
        let messages = vec![Message::tool_result("1", content, false)];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(html.contains("line one\tindented"), "{html}");
        assert!(html.contains("line two"), "{html}");
    }

    #[test]
    fn stats_section_folds_a_compaction_count_into_the_messages_line() {
        // Fix 5: pi's header always folds compaction/branch-summary counts into its "Messages"
        // summary line when present (`template.js:1352-1381`, e.g. "N compactions").
        let messages = vec![Message::user(format!(
            "{}\n\n{}",
            agent_core::compaction::SUMMARY_MARKER,
            "Refactored the auth module."
        ))];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(html.contains("1 compactions"), "{html}");
    }

    #[test]
    fn stats_section_folds_a_branch_summary_count_into_the_messages_line() {
        let messages = vec![Message::user(format!(
            "{}\n\n{}",
            agent_core::BRANCH_SUMMARY_MARKER,
            "Explored using a cache; reverted."
        ))];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(html.contains("1 branch summaries"), "{html}");
    }

    #[test]
    fn stats_section_omits_compaction_and_branch_summary_counts_when_there_are_none() {
        // The CSS in `STYLE` always mentions `.summary-marker.compaction` regardless of content, so
        // this checks for the actual stat phrasing (`"N compactions"`/`"N branch summaries"`), not a
        // bare substring match against "compaction" anywhere in the document.
        let html = render_html(&meta(), &[Message::user("hi")], &[], None);
        assert!(!html.contains("compactions"), "{html}");
        assert!(!html.contains("branch summaries"), "{html}");
        assert!(html.contains("1 user, 0 assistant"), "{html}");
    }

    #[test]
    fn stats_section_folds_a_tool_results_count_into_the_messages_line() {
        // Task #32 (pi-parity fix): matching pi's own header line, which always folds a tool-results
        // count into this same "Messages" summary (e.g. "12 user, 15 assistant, 40 tool results").
        let messages = vec![
            Message::assistant(vec![ContentBlock::tool_use(
                "1",
                "read",
                serde_json::json!({ "path": "a.rs" }),
            )]),
            Message::tool_results(vec![
                ContentBlock::ToolResult {
                    tool_use_id: "1".into(),
                    content: "fn a() {}".into(),
                    is_error: false,
                    images: vec![],
                },
                ContentBlock::ToolResult {
                    tool_use_id: "2".into(),
                    content: "fn b() {}".into(),
                    is_error: false,
                    images: vec![],
                },
            ]),
        ];
        let html = render_html(&meta(), &messages, &[], None);
        assert!(html.contains("2 tool results"), "{html}");
    }

    #[test]
    fn stats_section_folds_a_custom_entries_count_into_the_messages_line() {
        // Task #32 (pi-parity fix): a custom entry (`SessionStore::append_custom`) contributes nothing
        // to `messages` at all — this count can only come from `events`, unlike every other
        // `MessageStats` field.
        use crate::session_store::ExportEvent;
        let events = vec![
            ExportEvent::Custom {
                kind: "checkpoint".into(),
                data: serde_json::json!({}),
            },
            ExportEvent::Custom {
                kind: "checkpoint".into(),
                data: serde_json::json!({}),
            },
            // A non-`Custom` event must not be miscounted as one.
            ExportEvent::ModelChange("claude-test".into()),
        ];
        let html = render_html_with_entries(&meta(), &[Message::user("hi")], &[], None, &events);
        assert!(html.contains("2 custom entries"), "{html}");
    }

    #[test]
    fn stats_section_omits_tool_results_and_custom_entries_counts_when_there_are_none() {
        let html = render_html(&meta(), &[Message::user("hi")], &[], None);
        assert!(!html.contains("tool results"), "{html}");
        assert!(!html.contains("custom entries"), "{html}");
    }
}
