//! Generators shared by the fuzz targets. Every target takes the input two ways: as raw bytes (the
//! parser must survive anything a command or an endpoint can print) and as a structured value that
//! is serialized exactly as the real tool prints it, parsed, and required to come back equal.

use beyond_sandbox::fs::{Hit, LINE_TRUNCATED_SUFFIX, MAX_LINE};
use libfuzzer_sys::arbitrary::{Result, Unstructured};

/// Fragments that have broken, or nearly broken, these parsers: group separators, separators that
/// look like `<sep><digits><sep>`, carriage returns, tabs, newlines, a dot-prefix.
const PIECES: &[&str] = &[
    "--", "--\n", "-", ":", "-1-", ":12:", "-3:", "\r", "\n", "\t", ".", "..", "a", "b/c", "é",
    " ", "0", "99",
];

/// A string of arbitrary bytes (lossily decoded, as every command's output is before a parser sees
/// it) spliced with [`PIECES`]. Never contains NUL; never contains `\n` unless `newlines`.
pub fn text(u: &mut Unstructured<'_>, max: usize, newlines: bool) -> Result<String> {
    let mut out = String::new();
    for _ in 0..u.int_in_range(0..=4usize)? {
        if u.ratio(1, 2)? {
            out.push_str(PIECES[u.choose_index(PIECES.len())?]);
        } else {
            let n = u.int_in_range(0..=max.min(8))?;
            out.push_str(&String::from_utf8_lossy(u.bytes(n.min(u.len()))?));
        }
    }
    out.retain(|c| c != '\0' && (newlines || c != '\n'));
    out.truncate(floor_boundary(&out, max));
    Ok(out)
}

fn floor_boundary(s: &str, mut i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Like [`text`] but never empty (a path is never empty).
pub fn name(u: &mut Unstructured<'_>, newlines: bool) -> Result<String> {
    let s = text(u, 24, newlines)?;
    Ok(if s.is_empty() { "p".to_string() } else { s })
}

/// Bytes a parser may add to its hits beyond its input: [`LINE_TRUNCATED_SUFFIX`] for each line it
/// clipped to [`MAX_LINE`] characters. A head clip keeps the first `MAX_LINE` characters of a line
/// that has more, so the suffix is the only growth; a clipped text is exactly those characters followed
/// by the suffix, which is what this counts.
pub fn clip_slack(hits: &[Hit]) -> usize {
    let clipped_chars = MAX_LINE + LINE_TRUNCATED_SUFFIX.chars().count();
    let clipped = hits
        .iter()
        .filter(|h| {
            h.text.chars().count() == clipped_chars && h.text.ends_with(LINE_TRUNCATED_SUFFIX)
        })
        .count();
    clipped * LINE_TRUNCATED_SUFFIX.len()
}
