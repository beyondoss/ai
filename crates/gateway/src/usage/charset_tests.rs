//! The charsets and small helpers the row's string fields pass through.

use super::*;

#[test]
fn a_tier_is_lowercase_digits_underscore_and_hyphen_only() {
    for ok in ["priority", "standard_only", "us-east", "v2", "a_b-c"] {
        assert_eq!(tier_from(ok).as_deref(), Some(ok), "{ok}");
    }
    for bad in [
        "",
        "Priority",
        "a b",
        "a.b",
        "a/b",
        "a+b",
        "x\"y",
        "seventeen-letters",
    ] {
        assert_eq!(tier_from(bad), None, "{bad:?}");
    }
}

#[test]
fn ids_and_hosts_keep_their_charsets() {
    for ok in ["msg_01AbC", "gen-1760000000-abc", "resp.1:2", "chatcmpl-9"] {
        assert_eq!(id_from(ok).as_deref(), Some(ok), "{ok}");
    }
    for bad in ["", "a b", "a/b", "a\"b", "a\\b", &"x".repeat(129)] {
        assert_eq!(id_from(bad), None, "{bad:?}");
    }
    for ok in ["Amazon Bedrock", "Google (Vertex)", "a/b_c-d.e"] {
        assert_eq!(host_from(ok).as_deref(), Some(ok), "{ok}");
    }
    for bad in ["", "a\nb", "a\"b", "a:b", &"x".repeat(65)] {
        assert_eq!(host_from(bad), None, "{bad:?}");
    }
    assert!(id_byte(b'_') && id_byte(b'-') && id_byte(b'.') && id_byte(b':'));
    assert!(!id_byte(b' ') && !id_byte(b'/'));
}

#[test]
fn server_tools_render_nonzero_kinds_in_order_and_merge_by_max() {
    let mut t = ServerTools::default();
    assert!(!t.any());
    assert_eq!(t.to_row(), None);
    t.tool_calls = 1;
    assert!(t.any(), "the last kind counts too");
    t.web_search = 2;
    assert_eq!(t.to_row().as_deref(), Some("web_search=2,tool_calls=1"));
    let mut all = ServerTools::default();
    let n = all.entries().len();
    all.merge_max(&ServerTools {
        web_search: 1,
        web_search_preview: 2,
        web_search_page: 3,
        web_fetch: 4,
        code_execution: 5,
        file_search: 6,
        image_generation: 7,
        computer_use: 8,
        mcp: 9,
        shell: 10,
        tool_search: 11,
        x_search: 12,
        x_posts: 13,
        x_users: 14,
        document_search: 15,
        sources: 16,
        tool_calls: 17,
    });
    let counts: Vec<u32> = all.entries().iter().map(|(_, c)| *c).collect();
    assert_eq!(
        counts,
        (1..=n as u32).collect::<Vec<_>>(),
        "every kind merges"
    );
    all.merge_max(&ServerTools::default());
    assert_eq!(all.web_search, 1, "max, not overwrite");
}
