//! `edit` — string replacement in a file (unique match unless `replace_all`). Matching is exact
//! first, then a normalized fuzzy fallback (NFKC + unified quotes/dashes/spaces + per-line
//! trailing-whitespace) so a model's slightly-off `old_string` still lands.

use agent_core::tool::Tool;
use agent_core::{ToolError, ToolOutput};
use async_trait::async_trait;
use serde_json::{Value, json};

pub struct Edit;

#[async_trait]
impl Tool for Edit {
    fn name(&self) -> &str {
        "edit"
    }
    fn description(&self) -> &str {
        "Apply exact-match replacements to a file. Pass an `edits` array of {old_string, new_string} \
         (each `old_string` must match exactly once), or a single old_string/new_string. With \
         `replace_all`, a single edit replaces every occurrence."
    }
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "File to edit." },
                "edits": {
                    "type": "array",
                    "description": "Ordered replacements; each old_string must match exactly once.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_string": { "type": "string" },
                            "new_string": { "type": "string" }
                        },
                        "required": ["old_string", "new_string"]
                    }
                },
                "old_string": { "type": "string", "description": "Single-edit form: exact text to replace." },
                "new_string": { "type": "string", "description": "Single-edit form: replacement text." },
                "replace_all": { "type": "boolean", "description": "Single-edit form: replace every occurrence." }
            },
            "required": ["path"]
        })
    }

    fn write_target(&self, input: &Value) -> Option<String> {
        input
            .get("path")
            .and_then(Value::as_str)
            .map(super::canonical_write_target)
    }

    async fn run(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let path = input
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidInput("missing `path`".into()))?;
        let edits = parse_edits(&input)?;
        let replace_all = input
            .get("replace_all")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if replace_all && edits.len() != 1 {
            return Err(ToolError::InvalidInput(
                "`replace_all` applies only to a single edit".into(),
            ));
        }

        let raw = std::fs::read_to_string(path)
            .map_err(|e| ToolError::Execution(format!("read {path}: {e}")))?;
        // Match in a normalized LF space with the BOM stripped, then restore the file's original
        // line endings + BOM on write. A CRLF file whose `old_string` uses `\n` (the common case)
        // would otherwise never match — a silent, unrecoverable failure.
        let had_bom = raw.starts_with('\u{feff}');
        let body = raw.strip_prefix('\u{feff}').unwrap_or(&raw);
        let had_crlf = body.contains("\r\n");
        let working = body.replace("\r\n", "\n");

        // Resolve every edit to byte ranges against the *original* text (not the running result), so
        // multi-edit semantics are order-independent and an earlier edit's output can't accidentally
        // match a later `old_string`.
        let mut ranges: Vec<(usize, usize, String)> = Vec::new();
        for (old, new) in &edits {
            let old = old.replace("\r\n", "\n");
            let new = new.replace("\r\n", "\n");
            for (start, end) in find_spans(&working, &old, replace_all)
                .map_err(|msg| ToolError::InvalidInput(format!("{msg} in {path}")))?
            {
                ranges.push((start, end, new.clone()));
            }
        }

        // Reject overlapping edits rather than silently corrupting.
        ranges.sort_by_key(|r| r.0);
        for pair in ranges.windows(2) {
            if pair[0].1 > pair[1].0 {
                return Err(ToolError::InvalidInput(format!(
                    "edits overlap in {path}; they must touch disjoint regions"
                )));
            }
        }

        // Splice the replacements in a single pass.
        let mut out = String::with_capacity(working.len());
        let mut cursor = 0;
        for (start, end, new) in &ranges {
            out.push_str(&working[cursor..*start]);
            out.push_str(new);
            cursor = *end;
        }
        out.push_str(&working[cursor..]);

        if out == working {
            return Err(ToolError::InvalidInput(format!(
                "edit made no changes to {path}"
            )));
        }

        let restored = restore(out, had_crlf, had_bom);
        super::write_atomic(path, restored.as_bytes())
            .map_err(|e| ToolError::Execution(format!("write {path}: {e}")))?;
        let applied = ranges.len();
        Ok(format!(
            "edited {path} ({applied} replacement{})",
            if applied == 1 { "" } else { "s" }
        )
        .into())
    }
}

/// Byte `(start, end)` spans in `working` to replace. Tries an **exact** match first; if that finds
/// nothing, falls back to a **fuzzy** match in a normalized space (NFKC + unified quotes/dashes/
/// spaces + per-line trailing-whitespace stripped) and maps the hit back to original byte offsets —
/// so a model's `old_string` carrying a curly quote, an em-dash, an nbsp, or stray trailing
/// whitespace still lands instead of failing "not found". Errors if `old` is empty, absent, or
/// (without `replace_all`) ambiguous. Spans can be wider than `old` when the file's bytes differ from
/// the normalized needle (e.g. a 3-byte em-dash matched by a 1-byte `-`), which is why this returns
/// real end offsets rather than `start + old.len()`.
fn find_spans(working: &str, old: &str, replace_all: bool) -> Result<Vec<(usize, usize)>, String> {
    if old.is_empty() {
        return Err("`old_string` is empty".into());
    }
    let exact: Vec<(usize, usize)> = working
        .match_indices(old)
        .map(|(i, _)| (i, i + old.len()))
        .collect();

    // Normalize both sides once, up front — the ambiguity check below needs it even when an exact
    // match wins, not only on the fuzzy-fallback path.
    let (norm_work, map) = normalize_with_map(working);
    let norm_old = normalize_with_map(old).0;

    if !exact.is_empty() {
        // Prefer exact spans for the actual splice — preserves every byte around the match instead of
        // rewriting the whole line from normalized text (see the module docs). But the *uniqueness*
        // check must still count in normalized space: every exact match is trivially also a fuzzy
        // match, so a near-duplicate that only fuzzy-matches (a curly-quoted twin of an ASCII-quoted
        // exact hit, say) is still a real ambiguity — it just doesn't show up in `exact`. Matches the
        // reference agent's `countOccurrences`, which always counts in normalized space regardless of
        // which candidate would eventually win.
        let fuzzy_count = if norm_old.is_empty() {
            exact.len()
        } else {
            norm_work.matches(norm_old.as_str()).count()
        };
        return disambiguate(exact, fuzzy_count, old, replace_all);
    }

    // Fuzzy fallback: no exact match anywhere, so search — and splice — in normalized space, mapping
    // matches back via the normalized→original byte-offset table.
    if norm_old.is_empty() {
        return Err(format!("`old_string` not found: {old:?}"));
    }
    let fuzzy: Vec<(usize, usize)> = norm_work
        .match_indices(&norm_old)
        .map(|(i, _)| (map[i] as usize, map[i + norm_old.len()] as usize))
        .collect();
    let count = fuzzy.len();
    disambiguate(fuzzy, count, old, replace_all)
}

/// Apply the uniqueness/absence rules shared by the exact and fuzzy passes. `spans` is what would
/// actually be spliced; `fuzzy_count` is how many matches exist in normalized space — the two differ
/// exactly when an exact match won but a fuzzy-only near-duplicate also exists elsewhere.
fn disambiguate(
    spans: Vec<(usize, usize)>,
    fuzzy_count: usize,
    old: &str,
    replace_all: bool,
) -> Result<Vec<(usize, usize)>, String> {
    match spans.len() {
        0 => Err(format!("`old_string` not found: {old:?}")),
        _ if fuzzy_count > 1 && !replace_all => Err(format!(
            "`old_string` is not unique ({fuzzy_count} matches): {old:?}; add surrounding context"
        )),
        _ => Ok(spans),
    }
}

/// Fold a single scalar toward an ASCII canonical for fuzzy matching: smart quotes → `'`/`"`, the
/// unicode dash family → `-`, and the assorted unicode spaces → a plain space. Returns `None` to keep
/// the char unchanged. (NFKC, applied first by the caller, already folds e.g. nbsp → space; this
/// catches the cases NFKC leaves alone, like curly quotes and em-dashes.)
fn fold_char(c: char) -> Option<char> {
    Some(match c {
        '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' | '\u{2032}' | '\u{2035}' => '\'',
        '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' | '\u{2033}' | '\u{2036}' => '"',
        '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' | '\u{2015}'
        | '\u{2212}' | '\u{FE58}' | '\u{FE63}' | '\u{FF0D}' => '-',
        '\u{00A0}'
        | '\u{1680}'
        | '\u{2000}'..='\u{200A}'
        | '\u{202F}'
        | '\u{205F}'
        | '\u{3000}' => ' ',
        _ => return None,
    })
}

/// Normalize `orig` for fuzzy matching and return the result plus a byte-offset map back to `orig`:
/// `map[i]` is the original byte offset that normalized byte `i` came from, with a trailing sentinel
/// `map[out.len()] == orig.len()`. Normalization is per-scalar NFKC, then [`fold_char`], then per-line
/// trailing-whitespace stripping (spaces/tabs immediately before a `\n` or EOF).
///
/// Done in one pass, emitting straight into `out` + `map`. Candidate trailing whitespace is held in
/// `pending` until a `\n`/EOF proves it trailing (drop) or another char proves it interior (flush), so
/// no intermediate copy of the whole input is built — the prior two-pass allocated `a`, `a_map`, `out`,
/// `out_map`, and a final string; this allocates `out` + `map` + a tiny `pending`. `map` is `u32`
/// (half the width of the old `usize`): `edit` reads the whole file into a `String` first, so a >4 GiB
/// file can't reach here.
pub fn normalize_with_map(orig: &str) -> (String, Vec<u32>) {
    use unicode_normalization::UnicodeNormalization;

    let mut out = String::with_capacity(orig.len());
    let mut map = Vec::<u32>::with_capacity(orig.len() + 1);
    // A held run of candidate trailing whitespace: (byte — always ASCII ' '/'\t', source offset).
    let mut pending: Vec<(u8, u32)> = Vec::new();
    let mut buf = [0u8; 4];

    for (off, c) in orig.char_indices() {
        let off = off as u32;
        for nc in c.nfkc() {
            let folded = fold_char(nc).unwrap_or(nc);
            match folded {
                // Whitespace: hold it — it's trailing only if a `\n`/EOF follows.
                ' ' | '\t' => pending.push((folded as u8, off)),
                // Newline: the held run was trailing → drop it.
                '\n' => {
                    pending.clear();
                    out.push('\n');
                    map.push(off);
                }
                // Any other char: the held run was interior → flush it, then emit the char.
                _ => {
                    for (b, o) in pending.drain(..) {
                        out.push(b as char);
                        map.push(o);
                    }
                    let s = folded.encode_utf8(&mut buf);
                    out.push_str(s);
                    for _ in 0..s.len() {
                        map.push(off);
                    }
                }
            }
        }
    }
    // A trailing whitespace run at EOF is dropped (`pending` left unflushed).
    map.push(orig.len() as u32); // sentinel
    (out, map)
}

/// Restore the file's original line endings (LF → CRLF) and a leading BOM after editing in LF space.
fn restore(out: String, had_crlf: bool, had_bom: bool) -> String {
    let body = if had_crlf {
        out.replace('\n', "\r\n")
    } else {
        out
    };
    if had_bom {
        format!("\u{feff}{body}")
    } else {
        body
    }
}

/// Accept either the `edits` array form (pi-style) or the single old_string/new_string form. Also
/// recovers the case where a model sends `edits` as a JSON-encoded *string* instead of an array.
fn parse_edits(input: &Value) -> Result<Vec<(String, String)>, ToolError> {
    // Some models emit `edits` as a JSON string; parse it back into a value first.
    let edits_value = match input.get("edits") {
        Some(Value::String(s)) => serde_json::from_str::<Value>(s).ok(),
        other => other.cloned(),
    };
    if let Some(arr) = edits_value.as_ref().and_then(Value::as_array) {
        if arr.is_empty() {
            return Err(ToolError::InvalidInput("`edits` is empty".into()));
        }
        return arr
            .iter()
            .map(|e| {
                let old = e.get("old_string").and_then(Value::as_str);
                let new = e.get("new_string").and_then(Value::as_str);
                match (old, new) {
                    (Some(o), Some(n)) => Ok((o.to_string(), n.to_string())),
                    _ => Err(ToolError::InvalidInput(
                        "each edit needs old_string and new_string".into(),
                    )),
                }
            })
            .collect();
    }
    let old = input.get("old_string").and_then(Value::as_str);
    let new = input.get("new_string").and_then(Value::as_str);
    match (old, new) {
        (Some(o), Some(n)) => Ok(vec![(o.to_string(), n.to_string())]),
        _ => Err(ToolError::InvalidInput(
            "provide `edits`, or `old_string` and `new_string`".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_tmp(contents: &str) -> tempfile::NamedTempFile {
        use std::io::Write as _;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(contents.as_bytes()).unwrap();
        f
    }

    #[tokio::test]
    async fn replaces_unique_match() {
        let f = write_tmp("the quick brown fox");
        let p = f.path().to_str().unwrap();
        Edit.run(json!({ "path": p, "old_string": "quick", "new_string": "slow" }))
            .await
            .unwrap();
        assert_eq!(std::fs::read_to_string(p).unwrap(), "the slow brown fox");
    }

    #[tokio::test]
    async fn rejects_ambiguous_match() {
        let f = write_tmp("a a a");
        let err = Edit
            .run(
                json!({ "path": f.path().to_str().unwrap(), "old_string": "a", "new_string": "b" }),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn replace_all_replaces_every_occurrence() {
        let f = write_tmp("a a a");
        let p = f.path().to_str().unwrap();
        Edit.run(json!({ "path": p, "old_string": "a", "new_string": "b", "replace_all": true }))
            .await
            .unwrap();
        assert_eq!(std::fs::read_to_string(p).unwrap(), "b b b");
    }

    #[tokio::test]
    async fn applies_edits_array_in_order() {
        let f = write_tmp("foo and bar");
        let p = f.path().to_str().unwrap();
        Edit.run(json!({
            "path": p,
            "edits": [
                { "old_string": "foo", "new_string": "baz" },
                { "old_string": "bar", "new_string": "qux" }
            ]
        }))
        .await
        .unwrap();
        assert_eq!(std::fs::read_to_string(p).unwrap(), "baz and qux");
    }

    #[tokio::test]
    async fn missing_string_is_invalid_input() {
        let f = write_tmp("hello");
        let err = Edit
            .run(json!({ "path": f.path().to_str().unwrap(), "old_string": "zzz", "new_string": "x" }))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn matches_across_crlf_line_endings() {
        // A CRLF file edited with an `\n`-joined `old_string` must still match, and the file must keep
        // its CRLF endings after the edit.
        let f = write_tmp("line one\r\nline two\r\nline three\r\n");
        let p = f.path().to_str().unwrap();
        Edit.run(json!({
            "path": p,
            "old_string": "line one\nline two",
            "new_string": "line one\nLINE TWO",
        }))
        .await
        .unwrap();
        let result = std::fs::read_to_string(p).unwrap();
        assert_eq!(result, "line one\r\nLINE TWO\r\nline three\r\n");
    }

    #[tokio::test]
    async fn rejects_overlapping_edits() {
        let f = write_tmp("abcdef");
        let err = Edit
            .run(json!({
                "path": f.path().to_str().unwrap(),
                "edits": [
                    { "old_string": "abcd", "new_string": "X" },
                    { "old_string": "cdef", "new_string": "Y" }
                ]
            }))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn rejects_no_op_edit() {
        let f = write_tmp("hello world");
        let err = Edit
            .run(json!({
                "path": f.path().to_str().unwrap(),
                "old_string": "hello",
                "new_string": "hello",
            }))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn recovers_edits_sent_as_json_string() {
        let f = write_tmp("foo and bar");
        let p = f.path().to_str().unwrap();
        // `edits` as a JSON-encoded string rather than an array.
        Edit.run(json!({
            "path": p,
            "edits": "[{\"old_string\":\"foo\",\"new_string\":\"baz\"}]",
        }))
        .await
        .unwrap();
        assert_eq!(std::fs::read_to_string(p).unwrap(), "baz and bar");
    }

    #[tokio::test]
    async fn matches_independent_of_application_order() {
        // edit1's output ("bar") must not be matched by a later edit; matching against the original
        // keeps this deterministic.
        let f = write_tmp("foo bar");
        let p = f.path().to_str().unwrap();
        Edit.run(json!({
            "path": p,
            "edits": [
                { "old_string": "foo", "new_string": "bar" },
                { "old_string": "bar", "new_string": "qux" }
            ]
        }))
        .await
        .unwrap();
        // Second edit targets the *original* "bar", not the "bar" produced by the first edit.
        assert_eq!(std::fs::read_to_string(p).unwrap(), "bar qux");
    }

    #[tokio::test]
    async fn fuzzy_matches_across_smart_quotes_and_dashes() {
        // The file has curly quotes and an em-dash; the model's old_string uses ASCII. Exact match
        // fails, the fuzzy fallback lands, and — crucially — the replacement preserves the rest of
        // the line byte-for-byte (the wider em-dash span is mapped back correctly).
        let f = write_tmp("title = \u{201c}hello\u{201d} \u{2014} world");
        let p = f.path().to_str().unwrap();
        Edit.run(json!({
            "path": p,
            "old_string": "\"hello\" - world",
            "new_string": "greeting",
        }))
        .await
        .unwrap();
        assert_eq!(std::fs::read_to_string(p).unwrap(), "title = greeting");
    }

    #[tokio::test]
    async fn fuzzy_matches_despite_trailing_whitespace() {
        // The file has trailing spaces after `foo`; the model's old_string omits them.
        let f = write_tmp("foo   \nbar\n");
        let p = f.path().to_str().unwrap();
        Edit.run(json!({
            "path": p,
            "old_string": "foo\nbar",
            "new_string": "baz\nqux",
        }))
        .await
        .unwrap();
        assert_eq!(std::fs::read_to_string(p).unwrap(), "baz\nqux\n");
    }

    #[tokio::test]
    async fn fuzzy_matches_nbsp_as_space() {
        let f = write_tmp("a\u{00a0}b\u{00a0}c");
        let p = f.path().to_str().unwrap();
        Edit.run(json!({ "path": p, "old_string": "a b c", "new_string": "x" }))
            .await
            .unwrap();
        assert_eq!(std::fs::read_to_string(p).unwrap(), "x");
    }

    #[tokio::test]
    async fn exact_and_fuzzy_duplicate_together_are_ambiguous() {
        // An ASCII-quoted exact match and a curly-quoted near-duplicate both fold to the same
        // normalized text — a real ambiguity, even though only one is byte-exact. The reference
        // agent's `countOccurrences` always counts in normalized space regardless of which candidate
        // would eventually win, so this must be rejected rather than silently picking the exact one.
        let f = write_tmp("'a' and \u{2018}a\u{2019}");
        let p = f.path().to_str().unwrap();
        let err = Edit
            .run(json!({ "path": p, "old_string": "'a'", "new_string": "X" }))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn exact_match_wins_and_preserves_surrounding_bytes_when_unambiguous() {
        // Exactly one occurrence in both exact and fuzzy space: not ambiguous, and the exact span is
        // used for the splice — every byte around the match is untouched rather than the whole line
        // being rewritten from normalized text.
        let f = write_tmp("'a' and something else");
        let p = f.path().to_str().unwrap();
        Edit.run(json!({ "path": p, "old_string": "'a'", "new_string": "X" }))
            .await
            .unwrap();
        assert_eq!(std::fs::read_to_string(p).unwrap(), "X and something else");
    }

    #[tokio::test]
    async fn leaves_no_temp_file_behind() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.txt");
        std::fs::write(&path, "the quick brown fox").unwrap();
        Edit.run(
            json!({ "path": path.to_str().unwrap(), "old_string": "quick", "new_string": "slow" }),
        )
        .await
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "the slow brown fox"
        );
        // The atomic write (shared with `write`'s tool) must not leave its sibling temp behind.
        let temps: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(temps.is_empty(), "atomic write left a temp file behind");
    }
}
