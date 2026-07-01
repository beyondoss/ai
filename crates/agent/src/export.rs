//! Export a session's transcript as a single, self-contained HTML file — pi's `export_html`, for
//! sharing or reviewing a conversation outside the control protocol (no client, no server, just a
//! browser). Deliberately plain: one static page, inline CSS, no external assets or JS, so the file
//! is portable and viewable offline exactly as generated. Message text is rendered as markdown
//! (`render_markdown`, via `pulldown-cmark`) — server-side, at export time, rather than pi's own
//! approach of vendoring `marked`/`highlight.js` and running them client-side inside the exported
//! file — so this crate's "no JS" design holds even for formatted output. Deliberately **not**
//! paired with a real syntax-highlighting crate (e.g. `syntect`): that bundles several MB of
//! syntax/theme data and would slow every build of this CLI, including `run`/`serve`, which never
//! touch export, for a nice-to-have that fenced code blocks already get a useful approximation of via
//! plain `<pre><code class="language-x">` (language-tagged, monospaced, just not token-colored) —
//! except a `diff`-tagged block, or any tool-result content shaped like a unified diff, which does get
//! real per-line +/- coloring (`diff_html`/`looks_like_diff`), since that needs no language-specific
//! lexer at all.

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
    for message in messages {
        render_message(&mut out, message);
    }
    out.push_str("</main>\n");
    if !branches.is_empty() {
        render_branches(&mut out, branches);
    }
    out.push_str("</body>\n</html>\n");
    out
}

/// Render every abandoned branch as its own labeled section after the main transcript — only the part
/// that actually diverges from `messages` (`branch[shared..]`), so the shared prefix already shown
/// above isn't duplicated. A session that's never branched never calls this ([`render_html`] skips it
/// when `branches` is empty), so the common case renders exactly as it always has.
fn render_branches(out: &mut String, branches: &[(usize, Vec<Message>)]) {
    out.push_str("<section class=\"branches\">\n");
    out.push_str(&format!("<h2>Other branches ({})</h2>\n", branches.len()));
    for (i, (shared, branch_messages)) in branches.iter().enumerate() {
        out.push_str("<div class=\"branch\">\n");
        let note = if *shared == 0 {
            "forked from the start".to_string()
        } else {
            format!("forked after message {shared}")
        };
        out.push_str(&format!(
            "<div class=\"branch-title\">Branch {} &middot; {}</div>\n",
            i + 1,
            html_escape(&note)
        ));
        for message in &branch_messages[*shared..] {
            render_message(out, message);
        }
        out.push_str("</div>\n");
    }
    out.push_str("</section>\n");
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
.branches { border-top: 1px dashed #444; margin-top: 2rem; padding-top: 1rem; }\n\
.branches h2 { font-size: 1rem; color: #aaa; margin: 0 0 1rem; }\n\
.branch { border: 1px dashed #444; border-radius: 6px; padding: 0.75rem; margin-bottom: 1rem; }\n\
.branch-title { font-size: 0.8rem; color: #999; margin-bottom: 0.5rem; }\n\
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
        ContentBlock::ToolUse { name, input, .. } => {
            out.push_str(&format!(
                "<div class=\"tool-call\"><div class=\"tool-title\">Called <code>{}</code></div>\n\
                 <pre>{}</pre></div>\n",
                html_escape(name),
                html_escape(&serde_json::to_string_pretty(input).unwrap_or_default())
            ));
        }
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
        let messages = vec![
            Message::user("please read a.rs"),
            Message::assistant(vec![ContentBlock::ToolUse {
                id: "1".into(),
                name: "read".into(),
                input: serde_json::json!({ "path": "a.rs" }),
            }]),
            Message::tool_result("1", "fn main() {}", false),
        ];
        let html = render_html(&meta(), &messages, &[]);
        assert!(html.contains("please read a.rs"));
        assert!(html.contains("Called <code>read</code>"));
        assert!(html.contains("&quot;path&quot;"));
        assert!(html.contains("fn main() {}"));
        assert!(html.contains("class=\"tool-result\""));
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
    fn renders_abandoned_branches_after_the_main_transcript_skipping_the_shared_prefix() {
        let messages = vec![Message::user("what should I do?")];
        let branches = vec![(
            1usize,
            vec![
                Message::user("what should I do?"),
                Message::user("this is the divergent branch content"),
            ],
        )];
        let html = render_html(&meta(), &messages, &branches);
        assert!(html.contains("Other branches (1)"));
        assert!(html.contains("forked after message 1"));
        assert!(html.contains("this is the divergent branch content"));
        // The shared prefix (message 0, "what should I do?") must appear exactly once — from the main
        // transcript — not duplicated inside the branch section.
        assert_eq!(html.matches("what should I do?").count(), 1);
    }

    #[test]
    fn no_branches_section_when_there_are_no_abandoned_branches() {
        let html = render_html(&meta(), &[Message::user("hi")], &[]);
        assert!(!html.contains("<section class=\"branches\">"));
        assert!(!html.contains("Other branches"));
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
