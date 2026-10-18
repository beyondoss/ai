//! The lexer's events and its string decoding.

use super::*;

/// Every event of a lex, with offsets made absolute, so chunkings can be compared.
#[derive(Debug, Default, PartialEq)]
struct Events {
    base: usize,
    log: Vec<(char, usize, u32, bool)>,
    content: Vec<u8>,
    /// Stop at the n-th close, if set.
    stop_after: Option<usize>,
}

impl Visit for Events {
    fn byte(&mut self, at: usize, b: u8, depth: u32) -> Flow {
        self.log.push((char::from(b), self.base + at, depth, false));
        Flow::Continue(())
    }
    fn open(&mut self, at: usize, depth: u32) {
        self.log.push(('<', self.base + at, depth, false));
    }
    fn content(&mut self, raw: &[u8]) {
        self.content.extend_from_slice(raw);
    }
    fn close(&mut self, at: usize, depth: u32, escaped: bool) -> Flow {
        self.log.push(('>', self.base + at, depth, escaped));
        let closes = self.log.iter().filter(|e| e.0 == '>').count();
        if self.stop_after == Some(closes) {
            Flow::Break(())
        } else {
            Flow::Continue(())
        }
    }
}

fn lex_in(chunks: &[&[u8]]) -> Events {
    let mut lexer = Lexer::new();
    let mut ev = Events::default();
    for c in chunks {
        assert_eq!(lexer.feed(c, &mut ev), None);
        ev.base += c.len();
    }
    ev.base = 0;
    ev
}

/// The same events whatever the chunking, an escape split from what it escapes included.
#[test]
fn events_do_not_depend_on_the_chunking() {
    let body = b" {\"a\\\"b\":[1,{\"c\":\"d\\\\\"}],\"e\":\"\\u00e9\\\\\\\"x\"}] } ";
    let whole = lex_in(&[body]);
    for cut in 0..=body.len() {
        let (a, b) = body.split_at(cut);
        assert_eq!(lex_in(&[a, b]), whole, "cut at {cut}");
    }
    let bytes: Vec<&[u8]> = body.chunks(1).collect();
    assert_eq!(lex_in(&bytes), whole);
    // Content is every byte between the quotes, escapes as sent.
    assert_eq!(whole.content, b"a\\\"bcd\\\\e\\u00e9\\\\\\\"x");
    // A stray close leaves the depth at zero.
    assert_eq!(whole.log.last(), Some(&(' ', body.len() - 1, 0, false)));
}

/// A stop ends the lex just past the byte that stopped it, and the lexer resumes from there.
#[test]
fn a_stop_reports_where_it_stopped() {
    let body = br#"["a","b"]"#;
    let mut lexer = Lexer::new();
    let mut ev = Events {
        stop_after: Some(1),
        ..Events::default()
    };
    assert_eq!(lexer.feed(body, &mut ev), Some(4));
    assert_eq!(lexer.depth(), 1);
    ev.stop_after = None;
    assert_eq!(lexer.feed(&body[4..], &mut ev), None);
    assert_eq!(lexer.depth(), 0);
}

#[test]
fn string_end_finds_the_unescaped_quote() {
    assert_eq!(string_end(br#""ab""#, 0), Some(4));
    assert_eq!(string_end(br#"x"a\"b"y"#, 1), Some(7));
    assert_eq!(string_end(br#""a\\"b"#, 0), Some(5));
    assert_eq!(string_end(br#""ab"#, 0), None);
    assert_eq!(string_end(br#""ab\"#, 0), None);
    assert_eq!(string_end(br#""ab\""#, 0), None);
    assert_eq!(string_end(b"\"", 0), None);
    assert_eq!(string_end(b"", 0), None);
    assert_eq!(string_end(b"\"\"", usize::MAX), None);
    // A backslash escapes one byte for lexing: `\u` with no digits still ends at its quote.
    assert_eq!(string_end(br#""\u""#, 0), Some(4));
}

/// Each case decodes as serde_json decodes it into a `String`, refusals included.
#[test]
fn unescape_decodes_as_serde_does() {
    let cases: &[&[u8]] = &[
        b"plain",
        b"",
        br#"a\"b\\c\/d\b\f\n\r\t"#,
        b"\\u0041\\u00e9\\u00E9\\u20AC",
        b"\\ud83d\\ude00",
        b"\\uD83D\\uDE00x",
        br#"\ud83d"#,
        br#"\ud83dx"#,
        b"\\ud83d\\u0041",
        b"\\ud83d\\ud83d\\ude00",
        br#"\ude00"#,
        br#"\u12"#,
        br#"\u12g4"#,
        br#"\x"#,
        br#"\U0041"#,
        br#"a\"#,
        b"a\"b",
        b"tab\there",
        b"\x7f",
        "é😀".as_bytes(),
        b"\xc3",
        b"\xc3\\n\xa9",
        br#"\u0000"#,
    ];
    for &raw in cases {
        let mut quoted = vec![b'"'];
        quoted.extend_from_slice(raw);
        quoted.push(b'"');
        let want = serde_json::from_slice::<String>(&quoted).ok();
        assert_eq!(unescape(raw), want, "{}", String::from_utf8_lossy(raw));
        if let Some(w) = &want {
            assert!(unescapes_to(raw, w), "{}", String::from_utf8_lossy(raw));
            assert!(!unescapes_to(raw, &format!("{w}x")));
            if !w.is_empty() {
                assert!(!unescapes_to(
                    raw,
                    &w[..w.len() - w.chars().last().unwrap().len_utf8()]
                ));
            }
        } else {
            for w in ["", "a", "plain", "\u{1F600}"] {
                assert!(!unescapes_to(raw, w), "{}", String::from_utf8_lossy(raw));
            }
        }
    }
}

/// A string far longer than the name is refused unread.
#[test]
fn unescapes_to_refuses_a_string_too_long_to_match() {
    let raw = "\\u0041".repeat(2);
    assert!(unescapes_to(raw.as_bytes(), "AA"));
    let long = "\\u0041".repeat(10_000);
    assert!(!unescapes_to(long.as_bytes(), "A"));
}
