//! HTML → markdown for the `web` tool's `markdown` mode — reading docs without the HTML noise.
//!
//! A thin wrapper over `htmd` (a maintained HTML→markdown converter) with `script`/`style`/nav chrome
//! dropped, so what reaches the model is the readable body rather than the page skeleton.

/// Tags whose content is noise for a reader — scripts, styling, and navigation chrome. Dropping them
/// trims the boilerplate that surrounds an article's actual text.
const SKIP_TAGS: [&str; 6] = ["script", "style", "nav", "header", "footer", "aside"];

/// Convert an HTML document to markdown. Never panics — on the (rare) converter error the caller gets a
/// message; the `web` tool maps it to a `ToolError`.
pub fn to_markdown(html: &str) -> Result<String, String> {
    htmd::HtmlToMarkdown::builder()
        .skip_tags(SKIP_TAGS.to_vec())
        .build()
        .convert(html)
        .map(|md| md.trim().to_string())
        .map_err(|e| format!("HTML→markdown conversion failed: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_headings_links_and_lists() {
        let html = r#"<html><body>
            <h1>Title</h1>
            <p>Some <a href="https://example.com">link</a> text.</p>
            <ul><li>one</li><li>two</li></ul>
        </body></html>"#;
        let md = to_markdown(html).unwrap();
        assert!(md.contains("# Title"), "{md}");
        assert!(md.contains("[link](https://example.com)"), "{md}");
        assert!(md.contains("one") && md.contains("two"), "{md}");
    }

    #[test]
    fn drops_script_and_nav_chrome() {
        let html = r#"<html><body>
            <nav>home about contact</nav>
            <script>alert('x')</script>
            <main><p>Real content.</p></main>
            <footer>copyright</footer>
        </body></html>"#;
        let md = to_markdown(html).unwrap();
        assert!(md.contains("Real content."), "{md}");
        assert!(!md.contains("alert"), "script must be dropped: {md}");
        assert!(
            !md.contains("copyright"),
            "footer chrome must be dropped: {md}"
        );
    }
}
