//! Export a session's transcript as a single, self-contained HTML file — pi's `export_html`, for
//! sharing or reviewing a conversation outside the control protocol (no client, no server, just a
//! browser). Deliberately plain: one static page, inline CSS, no external assets or JS, so the file
//! is portable and viewable offline exactly as generated.

use std::io::Write as _;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use agent_core::{ContentBlock, Message, Role};

use crate::session_store::SessionMeta;
use crate::skills::xml_escape as html_escape;

/// Render `messages` (and `meta`'s header info) as a complete HTML document.
pub fn render_html(meta: &SessionMeta, messages: &[Message]) -> String {
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
    out.push_str("</main>\n</body>\n</html>\n");
    out
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
.text { white-space: pre-wrap; word-wrap: break-word; }\n\
.tool-call, .tool-result { border-left: 3px solid #666; padding-left: 0.75rem; margin: 0.5rem 0; }\n\
.tool-result.error { border-left-color: #c94f4f; }\n\
.tool-title { font-size: 0.8rem; color: #aaa; margin-bottom: 0.25rem; }\n\
pre { white-space: pre-wrap; word-wrap: break-word; background: #151515; padding: 0.5rem; \
border-radius: 4px; overflow-x: auto; margin: 0.25rem 0; }\n\
.thinking { font-style: italic; color: #888; border-left: 3px solid #555; padding-left: 0.75rem; \
margin: 0.5rem 0; }\n\
img.attachment { max-width: 100%; border-radius: 4px; margin: 0.5rem 0; display: block; }\n\
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
                "<div class=\"text\">{}</div>\n",
                html_escape(text)
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
                "<div class=\"{class}\"><div class=\"tool-title\">{title}</div>\n<pre>{}</pre>",
                html_escape(content)
            ));
            for image in images {
                render_image(out, &image.media_type, &image.data);
            }
            out.push_str("</div>\n");
        }
        ContentBlock::Image { source } => render_image(out, &source.media_type, &source.data),
    }
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

/// Render and write `messages` to an HTML file. `output_path` is used verbatim when given; otherwise
/// [`default_export_path`] is used. Parent directories are created as needed. Returns the path written.
pub fn export_html(
    meta: &SessionMeta,
    messages: &[Message],
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
    let html = render_html(meta, messages);
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
        let html = render_html(&meta(), &[]);
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
        let html = render_html(&meta(), &messages);
        assert!(html.contains("please read a.rs"));
        assert!(html.contains("Called <code>read</code>"));
        assert!(html.contains("&quot;path&quot;"));
        assert!(html.contains("fn main() {}"));
        assert!(html.contains("class=\"tool-result\""));
    }

    #[test]
    fn marks_a_tool_error_result_distinctly() {
        let messages = vec![Message::tool_result("1", "boom", true)];
        let html = render_html(&meta(), &messages);
        assert!(html.contains("class=\"tool-result error\""));
        assert!(html.contains(">Error<"));
    }

    #[test]
    fn escapes_html_metacharacters_in_message_text() {
        let messages = vec![Message::user("<script>alert(1)</script>")];
        let html = render_html(&meta(), &messages);
        assert!(!html.contains("<script>alert(1)</script>"));
        assert!(html.contains("&lt;script&gt;alert(1)&lt;/script&gt;"));
    }

    #[test]
    fn renders_an_image_attachment_as_a_data_uri() {
        let messages = vec![Message::assistant(vec![ContentBlock::Image {
            source: ImageSource::base64("image/png", "Zm9v"),
        }])];
        let html = render_html(&meta(), &messages);
        assert!(html.contains("src=\"data:image/png;base64,Zm9v\""));
    }

    #[test]
    fn export_html_writes_to_the_given_path() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("out.html");
        let path = export_html(&meta(), &[], Some(target.to_str().unwrap())).unwrap();
        assert_eq!(path, target);
        let written = std::fs::read_to_string(&path).unwrap();
        assert!(written.contains("Fix the bug"));
    }

    #[test]
    fn export_html_creates_missing_parent_directories() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("nested/deeper/out.html");
        let path = export_html(&meta(), &[], Some(target.to_str().unwrap())).unwrap();
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
