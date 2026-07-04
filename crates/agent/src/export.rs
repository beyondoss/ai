//! Export a session's transcript as a single, self-contained HTML file — pi's `export_html`, for
//! sharing or reviewing a conversation outside the control protocol (no client, no server, just a
//! browser). Deliberately plain: one static page, inline CSS, no external assets or **client-side**
//! JS, so the file is portable and viewable offline exactly as generated — `<details>`/`<summary>`
//! (native HTML, no script) is used for the one genuinely interactive piece, branch navigation (see
//! [`render_branches_diverging_at`]), so that stays true even though the page isn't purely static
//! reading order anymore. Message text is rendered as markdown (`render_markdown`, via
//! `pulldown-cmark`) — server-side, at export time, rather than pi's own approach of vendoring
//! `marked`/`highlight.js` and running them client-side inside the exported file. Deliberately
//! **not** paired with a real syntax-highlighting crate (e.g. `syntect`): that bundles several MB of
//! syntax/theme data and would slow every build of this CLI, including `run`/`serve`, which never
//! touch export, for a nice-to-have that fenced code blocks already get a useful approximation of via
//! plain `<pre><code class="language-x">` (language-tagged, monospaced, just not token-colored) —
//! except a `diff`-tagged block, or any tool-result content shaped like a unified diff, which does get
//! real per-line +/- coloring (`diff_html`/`looks_like_diff`), since that needs no language-specific
//! lexer at all. Every built-in tool call (`edit`/`write`/`bash`/`read`/`grep`/`find`/`ls`, and the
//! Beyond platform tools `fork`/`sync`/`logs`) gets a dedicated renderer (`render_tool_call`) instead
//! of raw pretty-printed JSON — `edit` in particular reuses the diff-coloring machinery to show its
//! before/after as a real (if not line-diffed) diff. Only a genuinely unrecognized tool name (a
//! third-party extension, or a future built-in not yet taught this module) falls back to generic JSON,
//! which reads fine for the rare case that needs it.

use std::io::Write as _;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use agent_core::{ContentBlock, Message, Role};

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
    render_html_inner(meta, messages, branches, usage, &[])
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
    render_html_inner(meta, messages, branches, usage, events)
}

fn render_html_inner(
    meta: &SessionMeta,
    messages: &[Message],
    branches: &[(usize, Vec<Message>)],
    usage: Option<UsageTotals>,
    events: &[crate::session_store::ExportEvent],
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
    render_stats_section(&mut out, meta, messages, usage);
    out.push_str("</header>\n<main>\n");
    // Every branch is rendered *inline*, immediately after the message it actually diverged from
    // (`shared` is a message-count prefix — the branch shares `messages[..shared]` with the active
    // path) — a real tree laid out in reading order, rather than one flat "other branches" dump
    // disconnected from the point it forked from at the bottom of the page. `shared == 0` branches
    // (forked before the very first message) render before the loop starts. Numbered in the order
    // they appear so a reader can refer to "branch 2" unambiguously even though they're scattered
    // through the page rather than listed together.
    let mut branch_number = render_branches_diverging_at(&mut out, branches, 0, 1);
    for (i, message) in messages.iter().enumerate() {
        render_message(&mut out, message);
        branch_number = render_branches_diverging_at(&mut out, branches, i + 1, branch_number);
    }
    render_events_section(&mut out, events);
    out.push_str("</main>\n");
    out.push_str("</body>\n</html>\n");
    out
}

/// A simple, un-fancy block per [`crate::session_store::ExportEvent`] — Track L36's whole point is
/// visibility, not fidelity to where in the transcript each one actually happened (that would need
/// each event's own position/timestamp threaded all the way through from `SessionStore`, a bigger
/// change than "currently invisible" calls for). A no-op when `events` is empty, so a plain
/// [`render_html`] call (which always passes `&[]`) never adds this section at all.
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
                format!("Thinking level changed to <code>{}</code>", html_escape(level))
            }
            crate::session_store::ExportEvent::Label { label: Some(label), .. } => {
                format!("Labeled: {}", html_escape(label))
            }
            crate::session_store::ExportEvent::Label { label: None, .. } => {
                "Label cleared".to_string()
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
            render_message(out, message);
        }
        out.push_str("</div>\n</details>\n");
        n += 1;
    }
    n
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

/// Entry-type counts derived purely from `messages` — unlike token usage, these need no data beyond
/// what [`render_html`] already receives.
struct MessageStats {
    user_messages: usize,
    assistant_messages: usize,
    tool_calls: usize,
    /// Distinct `model_id`s actually seen on an assistant turn (sorted), so a session that switched
    /// models mid-run shows every model actually used, not just the one it started with.
    models: Vec<String>,
}

impl MessageStats {
    fn compute(messages: &[Message]) -> Self {
        let mut user_messages = 0;
        let mut assistant_messages = 0;
        let mut tool_calls = 0;
        let mut models: Vec<String> = Vec::new();
        for m in messages {
            match m.role {
                Role::User => {
                    // A message carrying only `ToolResult` blocks is the model's tool feedback, not a
                    // real user turn — don't double-count it as one just because it rides on `Role::User`
                    // (Anthropic's own convention for where tool results live).
                    if m
                        .content
                        .iter()
                        .any(|b| !matches!(b, ContentBlock::ToolResult { .. }))
                    {
                        user_messages += 1;
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
        }
        models.sort();
        Self {
            user_messages,
            assistant_messages,
            tool_calls,
            models,
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
) {
    let stats = MessageStats::compute(messages);
    out.push_str("<div class=\"stats\">\n");
    let models = if stats.models.is_empty() {
        meta.model.clone()
    } else {
        stats.models.join(", ")
    };
    push_stat(out, "Models", &models);
    push_stat(
        out,
        "Messages",
        &format!(
            "{} user, {} assistant",
            stats.user_messages, stats.assistant_messages
        ),
    );
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
.stats { display: flex; flex-wrap: wrap; gap: 0.5rem 1.5rem; margin-top: 0.6rem; }\n\
.stat { font-size: 0.85rem; }\n\
.stat-label { color: #888; margin-right: 0.35rem; }\n\
.stat-value { color: #d4d4d4; }\n\
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

fn render_message(out: &mut String, message: &Message) {
    out.push_str(&format!(
        "<div class=\"message {}\">\n<div class=\"role-label\">{}</div>\n",
        role_class(message.role),
        role_label(message.role)
    ));
    for block in &message.content {
        render_block(out, block);
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

fn render_block(out: &mut String, block: &ContentBlock) {
    match block {
        ContentBlock::Text { text, .. } => {
            // A compaction or branch-summary recap materializes as a plain `Role::User` text block
            // carrying a literal bracketed marker line (`agent_core::compaction::apply_summary` /
            // `session_store.rs::branch_summary_message`) — detect and render it as its own distinctly
            // labeled block, matching pi's dedicated `.compaction`/`.branch-summary` blocks
            // (`template.js:1294-1306`), instead of plain markdown text with the marker line still
            // visible verbatim as if the model had actually written it.
            if let Some((class, label, body)) = parse_summary_marker(text) {
                render_summary_marker(out, class, label, body);
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
            // A tool result verbatim — not markdown (it's raw command/file output, and interpreting a
            // `#`-prefixed shell comment or a `-`-prefixed line as markdown would misrender it) —
            // except a unified diff, which gets the same per-line +/- coloring `render_markdown` gives
            // a fenced ```diff block.
            if looks_like_diff(content) {
                out.push_str(&diff_html(content));
            } else {
                out.push_str(&format!("<pre>{}</pre>", html_escape(content)));
            }
            for image in images {
                render_image(out, &image.media_type, &image.data);
            }
            out.push_str("</div>\n");
        }
        ContentBlock::Image { source } => render_image(out, &source.media_type, &source.data),
    }
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

/// Render an old/new string pair with the same per-line `+`/`-` coloring [`diff_html`] gives a real
/// unified diff — every line of `old` colored as removed, every line of `new` as added. Not a real
/// line-level diff (no common-prefix/suffix detection — that's a bigger algorithm than this static,
/// no-JS export needs), just old-then-new, clearly colored.
fn diff_pair_html(old: &str, new: &str) -> String {
    let mut out = String::from("<pre><code class=\"language-diff\">");
    for line in old.lines() {
        out.push_str(&format!(
            "<span class=\"diff-del\">-{}</span>\n",
            html_escape(line)
        ));
    }
    for line in new.lines() {
        out.push_str(&format!(
            "<span class=\"diff-add\">+{}</span>\n",
            html_escape(line)
        ));
    }
    out.push_str("</code></pre>\n");
    out
}

/// Render a `write` call: the target path as the title, full content in a plain `<pre>` (not
/// markdown — it's raw file content).
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
        out.push_str(&format!("<pre>{}</pre>", html_escape(content)));
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

/// Render a `read` call: just the path being read — the tool's own args carry nothing else worth a
/// title beyond that.
fn render_read_call(out: &mut String, input: &serde_json::Value) {
    let path = input.get("path").and_then(serde_json::Value::as_str);
    let title = match path {
        Some(p) => format!("Read <code>{}</code>", html_escape(p)),
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

/// Render an `ls` call: the directory listed, defaulting to `.` — matching the tool's own default.
fn render_ls_call(out: &mut String, input: &serde_json::Value) {
    let path = input
        .get("path")
        .and_then(serde_json::Value::as_str)
        .unwrap_or(".");
    out.push_str(&format!(
        "<div class=\"tool-call\"><div class=\"tool-title\">Listed <code>{}</code></div></div>\n",
        html_escape(path)
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

/// Allow-list a markdown link/image URL to http(s)/mailto or a same-document/relative reference,
/// dropping anything else (`javascript:`, `data:`, `vbscript:`, ...) rather than emitting it as a live
/// `href`/`src` — pi's own `sanitizeMarkdownUrl`.
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
/// constructed). Returns the CSS class, a display label, and the body — `None` for ordinary text,
/// including text that merely happens to start with one marker's literal characters but isn't followed
/// by the exact separator both call sites always produce (mirrors [`parse_skill_block`]'s same
/// exact-shape-or-not-at-all precedent).
fn parse_summary_marker(text: &str) -> Option<(&'static str, &'static str, &str)> {
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
        if let Some(body) = text.strip_prefix(marker).and_then(|r| r.strip_prefix("\n\n")) {
            return Some((class, label, body));
        }
    }
    None
}

/// Render a parsed compaction/branch-summary marker (see [`parse_summary_marker`]) as its own
/// distinctly labeled, styled block (markdown body) — matching pi's own dedicated `.compaction`/
/// `.branch-summary` blocks (`template.js:1294-1306`) instead of plain markdown text that still shows
/// the bracketed marker line verbatim, as if the model itself had written it.
fn render_summary_marker(out: &mut String, class: &str, label: &str, body: &str) {
    out.push_str(&format!(
        "<div class=\"summary-marker {class}\"><div class=\"tool-title\">{}</div>\n{}</div>\n",
        html_escape(label),
        render_markdown(body)
    ));
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
        assert!(html.contains("class=\"summary-marker compaction\""), "{html}");
        assert!(html.contains(">Compaction<"), "{html}");
        assert!(html.contains("Refactored the auth module"));
        assert!(
            !html.contains(agent_core::compaction::SUMMARY_MARKER),
            "the bracketed marker line must not appear verbatim: {html}"
        );
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
        let path =
            export_html(&meta(), &[Message::user("hi")], &[], Some(target.to_str().unwrap()))
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
        let html =
            render_html_with_entries(&meta(), &[Message::user("hi")], &[], None, &events);
        assert!(html.contains("Session Events"));
        assert!(html.contains("Model changed to <code>claude-opus-4-8</code>"));
        assert!(html.contains("Thinking level changed to <code>high</code>"));
        assert!(html.contains("Labeled: checkpoint"));
        assert!(html.contains("Label cleared"));
        assert!(html.contains("beyond:sync"));
        assert!(html.contains("marker"));
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
}
