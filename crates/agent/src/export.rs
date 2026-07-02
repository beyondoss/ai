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
//! lexer at all. The most common file-mutating/shell tool calls (`edit`/`write`/`bash`/`read`) get a
//! dedicated renderer (`render_tool_call`) instead of raw pretty-printed JSON — `edit` in particular
//! reuses the diff-coloring machinery to show its before/after as a real (if not line-diffed) diff;
//! everything else (`grep`/`find`/`ls`, the Beyond platform tools) falls back to generic JSON, which
//! already reads fine for those.

use std::io::Write as _;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use agent_core::{ContentBlock, Message, Role};

use crate::session_store::SessionMeta;
use crate::skills::xml_escape as html_escape;

/// Render `messages` (and `meta`'s header info) as a complete HTML document. `branches` is every
/// abandoned branch's full root-to-leaf chain plus how much of it is shared with `messages` (see
/// [`crate::session_store::SessionStore::abandoned_branches`]) — pass `&[]` for a session with no
/// tree (in-memory only) or when abandoned branches shouldn't be rendered.
pub fn render_html(
    meta: &SessionMeta,
    messages: &[Message],
    branches: &[(usize, Vec<Message>)],
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
    out.push_str("</main>\n");
    out.push_str("</body>\n</html>\n");
    out
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
    out.push_str("</div>\n");
}

fn render_block(out: &mut String, block: &ContentBlock) {
    match block {
        ContentBlock::Text { text } => {
            out.push_str(&format!(
                "<div class=\"text markdown\">{}</div>\n",
                render_markdown(text)
            ));
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
    let trimmed = url.trim();
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
        url
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

fn render_image(out: &mut String, media_type: &str, data: &str) {
    out.push_str(&format!(
        "<img class=\"attachment\" src=\"data:{};base64,{}\" alt=\"attachment\">\n",
        html_escape(media_type),
        data
    ));
}

/// A timestamped default export filename, `session-<unix-seconds>.html`, relative to the current
/// directory — used when [`export_html`] isn't given an explicit `output_path`.
fn default_export_path() -> PathBuf {
    PathBuf::from(format!("session-{}.html", now_secs()))
}

/// Render and write `messages` to an HTML file. `branches` is passed straight through to
/// [`render_html`] (pass `&[]` for a session with no tree, or when abandoned branches shouldn't be
/// rendered). `output_path` is used verbatim when given; otherwise [`default_export_path`] is used.
/// Parent directories are created as needed. Returns the path written.
pub fn export_html(
    meta: &SessionMeta,
    messages: &[Message],
    branches: &[(usize, Vec<Message>)],
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
    let html = render_html(meta, messages, branches);
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
        let html = render_html(&meta(), &[], &[]);
        assert!(html.starts_with("<!DOCTYPE html>"));
        assert!(html.trim_end().ends_with("</html>"));
        assert!(html.contains("Fix the bug"));
        assert!(html.contains("claude-test"));
        assert!(html.contains("0 message(s)"));
    }

    #[test]
    fn renders_text_tool_use_and_tool_result_blocks() {
        // `grep` (and every other tool without a dedicated renderer — see
        // `renders_edit_write_bash_and_read_calls_with_dedicated_rendering` below for those) still
        // falls back to generic pretty-printed JSON.
        let messages = vec![
            Message::user("please search a.rs"),
            Message::assistant(vec![ContentBlock::ToolUse {
                id: "1".into(),
                name: "grep".into(),
                input: serde_json::json!({ "pattern": "fn main", "path": "a.rs" }),
            }]),
            Message::tool_result("1", "fn main() {}", false),
        ];
        let html = render_html(&meta(), &messages, &[]);
        assert!(html.contains("please search a.rs"));
        assert!(html.contains("Called <code>grep</code>"));
        assert!(html.contains("&quot;pattern&quot;"));
        assert!(html.contains("fn main() {}"));
        assert!(html.contains("class=\"tool-result\""));
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
            }]),
            Message::assistant(vec![ContentBlock::ToolUse {
                id: "2".into(),
                name: "write".into(),
                input: serde_json::json!({ "path": "notes.md", "content": "hello world" }),
            }]),
            Message::assistant(vec![ContentBlock::ToolUse {
                id: "3".into(),
                name: "bash".into(),
                input: serde_json::json!({ "command": "cargo test", "cwd": "/proj" }),
            }]),
            Message::assistant(vec![ContentBlock::ToolUse {
                id: "4".into(),
                name: "read".into(),
                input: serde_json::json!({ "path": "README.md" }),
            }]),
        ];
        let html = render_html(&meta(), &messages, &[]);

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
        }])];
        let html = render_html(&meta(), &messages, &[]);
        assert!(html.contains("Called <code>edit</code>"));
    }

    #[test]
    fn marks_a_tool_error_result_distinctly() {
        let messages = vec![Message::tool_result("1", "boom", true)];
        let html = render_html(&meta(), &messages, &[]);
        assert!(html.contains("class=\"tool-result error\""));
        assert!(html.contains(">Error<"));
    }

    #[test]
    fn escapes_html_metacharacters_in_message_text() {
        let messages = vec![Message::user("<script>alert(1)</script>")];
        let html = render_html(&meta(), &messages, &[]);
        assert!(!html.contains("<script>alert(1)</script>"));
        assert!(html.contains("&lt;script&gt;alert(1)&lt;/script&gt;"));
    }

    #[test]
    fn renders_markdown_formatting_in_message_text() {
        let messages = vec![Message::user(
            "# Heading\n\n**bold** and a list:\n\n- one\n- two\n\n```rust\nfn main() {}\n```",
        )];
        let html = render_html(&meta(), &messages, &[]);
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
        let html = render_html(&meta(), &messages, &[]);
        assert!(
            !html.contains("javascript:"),
            "an unsafe URL scheme must never reach a live href: {html}"
        );
        assert!(html.contains("href=\"https://example.com\""));
    }

    #[test]
    fn defuses_raw_html_inside_markdown_text_to_plain_visible_text() {
        // Raw HTML in the markdown *source* (not the already-tested plain-text case above) must still
        // render as visible escaped text, not a live tag — a prompt-injected block quoted as "```" or
        // written as inline HTML shouldn't execute just because it parses as valid embedded HTML.
        let messages = vec![Message::user("before <img src=x onerror=alert(1)> after")];
        let html = render_html(&meta(), &messages, &[]);
        assert!(!html.contains("<img src=x"));
        assert!(html.contains("&lt;img"));
    }

    #[test]
    fn colors_a_fenced_diff_block_in_message_text_per_line() {
        let messages = vec![Message::user(
            "```diff\n--- a/f.rs\n+++ b/f.rs\n@@ -1 +1 @@\n-old\n+new\n context\n```",
        )];
        let html = render_html(&meta(), &messages, &[]);
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
        let html = render_html(&meta(), &messages, &[]);
        assert!(html.contains("class=\"diff-add\">+new"));
        assert!(html.contains("class=\"diff-del\">-old"));
    }

    #[test]
    fn does_not_miscolor_a_tool_result_that_merely_starts_lines_with_plus_or_minus() {
        // A bulleted list (or any other `-`/`+`-prefixed content) that isn't shaped like a real unified
        // diff (no hunk header, no `---`/`+++` file header) must not get diff coloring.
        let messages = vec![Message::tool_result("1", "- one\n- two\n+ three", false)];
        let html = render_html(&meta(), &messages, &[]);
        assert!(!html.contains("class=\"diff-add\""));
        assert!(!html.contains("class=\"diff-del\""));
        assert!(html.contains("- one"));
    }

    #[test]
    fn renders_an_image_attachment_as_a_data_uri() {
        let messages = vec![Message::assistant(vec![ContentBlock::Image {
            source: ImageSource::base64("image/png", "Zm9v"),
        }])];
        let html = render_html(&meta(), &messages, &[]);
        assert!(html.contains("src=\"data:image/png;base64,Zm9v\""));
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
        let html = render_html(&meta(), &messages, &branches);
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
        let html = render_html(&meta(), &messages, &branches);
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
        let html = render_html(&meta(), &messages, &branches);
        assert!(html.contains("Branch 1"));
        assert!(html.contains("Branch 2"));
        let b1 = html.find("branch-one").unwrap();
        let b2 = html.find("branch-two").unwrap();
        assert!(b1 < b2, "branches must appear in divergence order: {html}");
    }

    #[test]
    fn no_branches_section_when_there_are_no_abandoned_branches() {
        let html = render_html(&meta(), &[Message::user("hi")], &[]);
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
}
