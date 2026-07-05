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
            .map(super::normalize_path)
            .map(|p| super::canonical_write_target(&p))
    }

    async fn run(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let path = input
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidInput("missing `path`".into()))?;
        let path = &super::normalize_path(path);
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

        let (raw, initial_mtime) = read_with_mtime(path)?;
        // Fail fast on a read-only file before spending any match/diff work on it (fuzzy matching
        // does NFKC normalization + ambiguity resolution — real CPU work, not free) — pi's own
        // `access(path, W_OK)` pre-check, just via a metadata read instead of a syscall that doesn't
        // exist on every platform Rust targets.
        let writable = std::fs::metadata(path)
            .map(|m| !m.permissions().readonly())
            .unwrap_or(true); // a metadata read failing here is surfaced by the write attempt below
        if !writable {
            return Err(ToolError::Execution(format!("{path} is not writable")));
        }
        // Match in a normalized LF space with the BOM stripped, then restore the file's original
        // line endings + BOM on write. A CRLF file whose `old_string` uses `\n` (the common case)
        // would otherwise never match — a silent, unrecoverable failure.
        let had_bom = raw.starts_with('\u{feff}');
        let body = raw.strip_prefix('\u{feff}').unwrap_or(&raw);
        // A file can mix `\r\n` and bare `\n` endings across lines (different tools/editors touched
        // different lines). CRLF is only reintroduced wholesale when the file is *consistently* CRLF;
        // otherwise untouched spans are spliced back in from the original bytes verbatim (via
        // `body_map`) so an edit to one line can never flip another, untouched line's ending.
        let has_bare_lf = body.replace("\r\n", "").contains('\n');
        let is_pure_crlf = !has_bare_lf && body.contains("\r\n");
        let (working, body_map) = strip_crlf_with_map(body);

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

        // Splice the replacements in a single pass. Untouched spans are copied from `body` (the
        // original bytes, via `body_map`), not `working`, so a mixed-line-ending file's untouched
        // lines keep their exact original ending rather than being reconstructed from LF space.
        let mut out = String::with_capacity(body.len());
        let mut cursor = 0usize;
        for (start, end, new) in &ranges {
            out.push_str(&body[body_map[cursor] as usize..body_map[*start] as usize]);
            if is_pure_crlf {
                out.push_str(&new.replace('\n', "\r\n"));
            } else {
                out.push_str(new);
            }
            cursor = *end;
        }
        out.push_str(&body[body_map[cursor] as usize..]);

        if out == body {
            return Err(ToolError::InvalidInput(format!(
                "edit made no changes to {path}"
            )));
        }

        let restored = if had_bom {
            format!("\u{feff}{out}")
        } else {
            out
        };
        write_if_unchanged(path, initial_mtime, restored.as_bytes())?;
        let applied = ranges.len();
        // No diff/patch attached, deliberately: pi computes both too, but only for its interactive
        // terminal's own rendering — the model gets this same bare confirmation there as well. See
        // ARCHITECTURE.md's "Why `edit`'s result carries no diff/patch data" for the full reasoning
        // (no reader exists today on the model, wire-protocol, or export side).
        Ok(format!(
            "edited {path} ({applied} replacement{})",
            if applied == 1 { "" } else { "s" }
        )
        .into())
    }
}

/// Read `path`'s contents together with the mtime it had at that exact moment, captured from the same
/// open file handle the read itself uses — so there's no separate stat-then-read (or read-then-stat)
/// syscall pair that could itself race a concurrent writer. [`write_if_unchanged`] later compares
/// against this baseline right before the final write, to catch a different process (another agent
/// session on the same repo, an editor autosave, a formatter/build step) modifying the file during the
/// fuzzy-match/splice work `run` does in between.
///
/// `None` when the platform/filesystem doesn't report a modification time at all — rare, but exactly
/// the same "best-effort, don't fail the whole operation over one unreadable timestamp" reasoning
/// `session_store`'s trash-listing already uses for its own `.modified()` read.
fn read_with_mtime(path: &str) -> Result<(String, Option<std::time::SystemTime>), ToolError> {
    use std::io::Read as _;
    let mut file =
        std::fs::File::open(path).map_err(|e| ToolError::Execution(format!("read {path}: {e}")))?;
    let initial_mtime = file.metadata().and_then(|m| m.modified()).ok();
    let mut raw = String::new();
    file.read_to_string(&mut raw)
        .map_err(|e| ToolError::Execution(format!("read {path}: {e}")))?;
    Ok((raw, initial_mtime))
}

/// Write `content` to `path` via [`super::write_atomic`] — but only if `path`'s mtime still matches
/// `initial_mtime` (the baseline [`read_with_mtime`] captured when this edit's `old_string`/`new_string`
/// work started). If it doesn't, a different process wrote to the file in the window between the read
/// and this write; applying this edit anyway would silently discard that other write (last-writer-wins,
/// with no signal to the model that called this tool) — so this refuses instead, surfacing a clear
/// error the model can react to by re-reading the file and retrying.
///
/// Same-turn races from this agent's own overlapping tool calls are already handled elsewhere (see
/// `write_target`/the write-lock registry that serializes calls targeting the same file) — this check
/// is specifically for an *external* writer this agent has no other way to see coming.
///
/// Compares mtimes rather than content hashes: mtime is what the writability pre-check right above in
/// `run` already reasons about via `std::fs::metadata`, and it's the same signal `auth_store`/
/// `trust_store`/`settings`'s own stale-lock detection uses elsewhere in this codebase. `std::fs::metadata`
/// (not `symlink_metadata`) follows a trailing symlink to the real target's metadata, matching
/// `write_atomic`'s own symlink-following behavior (see its doc comment) — so this compares mtimes on
/// the same resolved path `write_atomic` actually writes through, not the symlink's own (unchanging)
/// directory-entry metadata, which would otherwise make an ordinary symlinked-dotfile edit spuriously
/// fail this check every time.
///
/// `initial_mtime: None` — the initial read's own metadata call failed, or the platform doesn't report
/// mtimes at all — skips the check entirely rather than hard-failing, the same graceful degradation the
/// writability pre-check above already applies to its own failed metadata read.
fn write_if_unchanged(
    path: &str,
    initial_mtime: Option<std::time::SystemTime>,
    content: &[u8],
) -> Result<(), ToolError> {
    if let Some(initial) = initial_mtime {
        let current = std::fs::metadata(path).and_then(|m| m.modified()).ok();
        if current != Some(initial) {
            return Err(ToolError::Execution(format!(
                "{path} was modified on disk after it was read; the edit was aborted to avoid \
                 discarding that change — re-read the file and retry"
            )));
        }
    }
    super::write_atomic(path, content)
        .map_err(|e| ToolError::Execution(format!("write {path}: {e}")))
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
///
/// Deliberately a superset of pi's own `normalizeForFuzzyMatch` (`edit-diff.ts`), not a port of its
/// exact table: pi folds quotes `2018/2019/201A/201B` → `'` and `201C/201D/201E/201F` → `"`, dashes
/// `2010–2015`/`2212` → `-`, and spaces `00A0`/`2002–200A`/`202F`/`205F`/`3000` → ` `. This adds the
/// prime marks (`2032`/`2035` → `'`, `2033`/`2036` → `"` — a model quoting measurements or citing a
/// reconstructed string sometimes reaches for a prime instead of an apostrophe), the small/fullwidth
/// dash forms (`FE58`/`FE63`/`FF0D`), and widens the space range down to `2000`/`2001` (en/em quad)
/// and to include `1680` (Ogham space mark). Every addition is the same kind of narrowly-scoped,
/// single-family confusable pi's own set already folds — never two visually-distinct characters
/// collapsed together — so broadening the net only helps a model's slightly-off `old_string` land; it
/// can't make two genuinely different characters match each other.
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

/// Collapse `\r\n` to `\n`, returning the result plus a byte-offset map back to `body`: `map[i]` is
/// the original byte offset that output byte `i` *starts at*, with a trailing sentinel
/// `map[out.len()] == body.len()`. A collapsed `\r\n` pair maps its resulting `\n` to the `\r`'s
/// offset (the start of the pair, not the `\n`'s own offset) — the pair is one atomic unit, so a
/// match boundary landing right after it must still recover the `\r` when the caller slices the
/// original bytes back out via this map (see `run`'s untouched-span splice); mapping to the `\n`'s own
/// offset would silently drop the preceding `\r` whenever a match boundary falls exactly there. Safe
/// on UTF-8: `\r`/`\n` are single ASCII bytes, so collapsing a pair never touches a multi-byte
/// character's other bytes.
fn strip_crlf_with_map(body: &str) -> (String, Vec<u32>) {
    let bytes = body.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut map = Vec::with_capacity(bytes.len() + 1);
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\r' && bytes.get(i + 1) == Some(&b'\n') {
            out.push(b'\n');
            map.push(i as u32);
            i += 2;
            continue;
        }
        out.push(bytes[i]);
        map.push(i as u32);
        i += 1;
    }
    map.push(bytes.len() as u32);
    (
        // Removing lone `\r` bytes that precede a `\n` never touches a multi-byte character's
        // continuation bytes (those are always >= 0x80, never equal to the single-byte ASCII `\r`),
        // so `out` is still valid UTF-8 whenever `body` was.
        #[allow(clippy::expect_used)]
        String::from_utf8(out).expect("removing ASCII \\r bytes preserves UTF-8 validity"),
        map,
    )
}

/// Accept either the `edits` array form (pi-style) or the single old_string/new_string form. Also
/// recovers the case where a model sends `edits` as a JSON-encoded *string* instead of an array. When
/// a call sends **both** an `edits` array and top-level `old_string`/`new_string`, the legacy pair is
/// appended as an extra edit rather than silently discarded — matches pi's behavior (a model that
/// carries both shapes over from a prior turn/retry means both, not just the array).
///
/// `pub(crate)`: also used by `export` to render an `edit` tool call's before/after as a real diff
/// instead of raw JSON — reusing this rather than re-parsing `input` independently keeps the exported
/// rendering in sync with whatever shape this tool actually accepts (including the JSON-string quirk).
pub(crate) fn parse_edits(input: &Value) -> Result<Vec<(String, String)>, ToolError> {
    // Some models emit `edits` as a JSON string; parse it back into a value first.
    let edits_value = match input.get("edits") {
        Some(Value::String(s)) => serde_json::from_str::<Value>(s).ok(),
        other => other.cloned(),
    };
    let legacy = match (
        input.get("old_string").and_then(Value::as_str),
        input.get("new_string").and_then(Value::as_str),
    ) {
        (Some(o), Some(n)) => Some((o.to_string(), n.to_string())),
        _ => None,
    };
    if let Some(arr) = edits_value.as_ref().and_then(Value::as_array) {
        if arr.is_empty() {
            return Err(ToolError::InvalidInput("`edits` is empty".into()));
        }
        let mut edits: Vec<(String, String)> = arr
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
            .collect::<Result<_, _>>()?;
        edits.extend(legacy);
        return Ok(edits);
    }
    legacy.map(|pair| vec![pair]).ok_or_else(|| {
        ToolError::InvalidInput("provide `edits`, or `old_string` and `new_string`".into())
    })
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
    async fn run_normalizes_the_path_argument() {
        // Proves `run` actually calls `super::normalize_path`, via its `@`-prefix-strip behavior
        // (needs no `$HOME` mutation — see `expand_tilde`'s own direct unit tests for that half).
        let f = write_tmp("the quick brown fox");
        let at_prefixed = format!("@{}", f.path().to_str().unwrap());
        Edit.run(json!({ "path": at_prefixed, "old_string": "quick", "new_string": "slow" }))
            .await
            .unwrap();
        assert_eq!(
            std::fs::read_to_string(f.path()).unwrap(),
            "the slow brown fox"
        );
    }

    #[tokio::test]
    async fn write_target_normalizes_the_path_argument_too() {
        // Same-turn concurrency-grouping key must match `run`'s own normalization, or `write("~/f")`
        // and `edit("~/f")` in one turn would get different canonical keys.
        let f = write_tmp("x");
        let p = f.path().to_str().unwrap();
        let at_prefixed = format!("@{p}");
        let plain = Edit.write_target(&json!({ "path": p })).unwrap();
        let normalized = Edit.write_target(&json!({ "path": at_prefixed })).unwrap();
        assert_eq!(plain, normalized);
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
    async fn rejects_a_read_only_file_before_doing_any_match_work() {
        let f = write_tmp("the quick brown fox");
        let p = f.path().to_str().unwrap();
        let mut perms = std::fs::metadata(p).unwrap().permissions();
        perms.set_readonly(true);
        std::fs::set_permissions(p, perms).unwrap();

        // `old_string` doesn't even match — if the writability pre-check didn't fire first, this
        // would fail with "not found" instead of "not writable", proving the check runs before any
        // match/diff work rather than only surfacing at the final write.
        let err = Edit
            .run(json!({ "path": p, "old_string": "does not appear", "new_string": "x" }))
            .await
            .unwrap_err();
        match err {
            ToolError::Execution(msg) => assert!(msg.contains("not writable"), "got: {msg}"),
            other => panic!("expected Execution(\"... not writable\"), got {other:?}"),
        }

        // Restore write permission (owner-only, not world-writable) so the tempfile crate can clean
        // up the file on drop.
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(p, std::fs::Permissions::from_mode(0o600)).unwrap();
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
    async fn preserves_untouched_lines_original_ending_in_mixed_crlf_file() {
        // Mixed line endings: `b`'s line is bare-LF, `a`'s and `c`'s lines are CRLF. Editing `b` must
        // not rewrite `c`'s line ending to CRLF — the bug was a whole-file `had_crlf` flag blanket
        // converting every `\n` back to `\r\n`, silently corrupting untouched bare-LF lines in a
        // mixed file.
        let f = write_tmp("a\r\nb\nc\r\n");
        let p = f.path().to_str().unwrap();
        Edit.run(json!({ "path": p, "old_string": "b", "new_string": "B" }))
            .await
            .unwrap();
        assert_eq!(std::fs::read_to_string(p).unwrap(), "a\r\nB\nc\r\n");
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
    async fn a_syntactically_invalid_edits_json_string_falls_through_to_requiring_top_level_fields()
    {
        // pi: edit-tool-legacy-input.test.ts, "leaves edits alone when the string is not valid JSON" —
        // pi's `prepareArguments` passes a non-JSON `edits` string through unchanged, letting the
        // ordinary schema validation reject it downstream. Ours parses inline (`parse_edits`): the
        // `serde_json::from_str::<Value>(s).ok()` on a malformed string yields `None`, so `edits_value`
        // is `None` too — the `arr` branch is skipped entirely, and (with no top-level `old_string`/
        // `new_string` given either) this must fall through to the same generic "provide `edits`, or
        // `old_string` and `new_string`" error as an outright missing `edits`, not a JSON-parse-error
        // message that would leak `serde_json`'s own wording to the model.
        let f = write_tmp("foo and bar");
        let p = f.path().to_str().unwrap();
        let err = Edit
            .run(json!({ "path": p, "edits": "not json" }))
            .await
            .unwrap_err();
        match err {
            ToolError::InvalidInput(msg) => assert!(
                msg.contains("provide `edits`, or `old_string` and `new_string`"),
                "got: {msg}"
            ),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
        // The file must be untouched — this must fail before ever attempting to read/write it.
        assert_eq!(std::fs::read_to_string(p).unwrap(), "foo and bar");
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
    async fn fuzzy_matches_fullwidth_cjk_punctuation_as_ascii() {
        // pi: tools.test.ts, fullwidth CJK comma/parens folding to ASCII. NFKC (applied first, before
        // `fold_char`) already maps the Fullwidth-Forms block to its ASCII counterparts, so this should
        // hold without `fold_char` needing its own entry — pinning it as a real end-to-end test rather
        // than trusting that by reading the Unicode tables.
        let f = write_tmp("call(a\u{ff0c}b)"); // fullwidth comma U+FF0C, fullwidth parens U+FF08/FF09
        let p = f.path().to_str().unwrap();
        Edit.run(json!({ "path": p, "old_string": "call(a,b)", "new_string": "call(a, b)" }))
            .await
            .unwrap();
        assert_eq!(std::fs::read_to_string(p).unwrap(), "call(a, b)");
    }

    #[tokio::test]
    async fn preserves_lf_line_endings_for_an_lf_only_file() {
        // Every other line-ending test covers mixed-CRLF or pure-CRLF files; this proves the ordinary
        // pure-LF case (the common one) explicitly rather than only by absence of a CRLF-specific test.
        let f = write_tmp("line one\nline two\nline three\n");
        let p = f.path().to_str().unwrap();
        Edit.run(json!({ "path": p, "old_string": "line two", "new_string": "LINE TWO" }))
            .await
            .unwrap();
        assert_eq!(
            std::fs::read_to_string(p).unwrap(),
            "line one\nLINE TWO\nline three\n"
        );
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
    async fn appends_legacy_replacement_to_existing_edits() {
        // Both `edits` and top-level `old_string`/`new_string` present: the legacy pair must be
        // applied too, not silently dropped (pi: edit-tool-legacy-input.test.ts, "appends legacy
        // replacement to existing edits").
        let f = write_tmp("foo and bar and baz");
        let p = f.path().to_str().unwrap();
        Edit.run(json!({
            "path": p,
            "edits": [{ "old_string": "foo", "new_string": "FOO" }],
            "old_string": "baz",
            "new_string": "BAZ",
        }))
        .await
        .unwrap();
        assert_eq!(std::fs::read_to_string(p).unwrap(), "FOO and bar and BAZ");
    }

    #[tokio::test]
    async fn empty_edits_array_is_invalid_input() {
        let f = write_tmp("hello");
        let err = Edit
            .run(json!({ "path": f.path().to_str().unwrap(), "edits": [] }))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn editing_a_nonexistent_path_is_an_execution_error() {
        let err = Edit
            .run(json!({
                "path": "/nonexistent/definitely-not-here.txt",
                "old_string": "a",
                "new_string": "b",
            }))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Execution(_)), "got: {err:?}");
    }

    #[tokio::test]
    async fn ambiguous_match_error_reports_the_exact_count() {
        let f = write_tmp("a a a");
        let err = Edit
            .run(
                json!({ "path": f.path().to_str().unwrap(), "old_string": "a", "new_string": "b" }),
            )
            .await
            .unwrap_err();
        match err {
            ToolError::InvalidInput(msg) => assert!(msg.contains('3'), "got: {msg}"),
            other => panic!("expected InvalidInput mentioning 3 matches, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_failing_multi_edit_call_leaves_the_file_byte_for_byte_unchanged() {
        // One edit in the batch has an `old_string` that isn't found. Every edit's span is resolved
        // before any write happens, so the file must come back out exactly as it went in — no partial
        // application (pi: tools.test.ts, "keeps the file unchanged when any edit in the batch fails").
        let original = "foo and bar and baz";
        let f = write_tmp(original);
        let p = f.path().to_str().unwrap();
        let err = Edit
            .run(json!({
                "path": p,
                "edits": [
                    { "old_string": "foo", "new_string": "FOO" },
                    { "old_string": "does not exist", "new_string": "X" }
                ]
            }))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
        assert_eq!(std::fs::read_to_string(p).unwrap(), original);
    }

    #[tokio::test]
    async fn preserves_a_byte_order_mark_across_a_single_edit() {
        let f = write_tmp("\u{feff}foo bar");
        let p = f.path().to_str().unwrap();
        Edit.run(json!({ "path": p, "old_string": "foo", "new_string": "baz" }))
            .await
            .unwrap();
        let out = std::fs::read_to_string(p).unwrap();
        assert!(out.starts_with('\u{feff}'), "BOM was dropped: {out:?}");
        assert_eq!(out, "\u{feff}baz bar");
    }

    #[tokio::test]
    async fn preserves_a_byte_order_mark_with_crlf_and_multiple_edits() {
        let f = write_tmp("\u{feff}foo\r\nbar\r\n");
        let p = f.path().to_str().unwrap();
        Edit.run(json!({
            "path": p,
            "edits": [
                { "old_string": "foo", "new_string": "FOO" },
                { "old_string": "bar", "new_string": "BAR" }
            ]
        }))
        .await
        .unwrap();
        let out = std::fs::read_to_string(p).unwrap();
        assert_eq!(out, "\u{feff}FOO\r\nBAR\r\n");
    }

    #[tokio::test]
    async fn fuzzy_match_works_across_a_multi_edit_array_not_just_single_edits() {
        // Every fuzzy-matching test above uses the single-edit form; the `edits` array path shares
        // `find_spans` per-edit, but nothing previously proved that end-to-end.
        let f = write_tmp("\u{201c}hello\u{201d} and \u{2014}world\u{2014}");
        let p = f.path().to_str().unwrap();
        Edit.run(json!({
            "path": p,
            "edits": [
                { "old_string": "\"hello\"", "new_string": "greeting" },
                { "old_string": "-world-", "new_string": "planet" }
            ]
        }))
        .await
        .unwrap();
        assert_eq!(std::fs::read_to_string(p).unwrap(), "greeting and planet");
    }

    #[tokio::test]
    async fn rejects_a_write_when_the_file_changed_on_disk_after_it_was_read() {
        // Simulates a concurrent *external* writer — a different process, not this agent's own
        // same-turn write-lock (that's `write_target`/the write-lock registry's job, exercised
        // elsewhere) — modifying the file in the window between `edit`'s read and its final write.
        // `run` itself has no pause point a test can land in mid-flight, so this drives its two
        // internal halves (`read_with_mtime` / `write_if_unchanged`) directly, exactly the way `run`
        // chains them, with the concurrent write injected in between.
        let f = write_tmp("the quick brown fox");
        let p = f.path().to_str().unwrap();

        let (raw, initial_mtime) = read_with_mtime(p).unwrap();
        assert!(
            initial_mtime.is_some(),
            "this filesystem should report mtimes"
        );
        let restored = raw.replace("quick", "slow"); // the edit `run` would have computed

        // The external writer's change, landing after the read this edit is based on. The mtime is
        // bumped explicitly (rather than relying on clock resolution alone between two writes issued
        // microseconds apart in a test) so the race is deterministic.
        std::fs::write(p, "the quick brown fox, edited elsewhere").unwrap();
        let bumped = initial_mtime.unwrap() + std::time::Duration::from_secs(5);
        std::fs::OpenOptions::new()
            .write(true)
            .open(p)
            .unwrap()
            .set_modified(bumped)
            .unwrap();

        let err = write_if_unchanged(p, initial_mtime, restored.as_bytes()).unwrap_err();
        match err {
            ToolError::Execution(msg) => {
                assert!(msg.contains("modified") && msg.contains(p), "got: {msg}")
            }
            other => panic!("expected Execution(\"... modified ...\"), got {other:?}"),
        }
        // The concurrent writer's content must survive untouched — silently discarding it
        // (last-writer-wins) is exactly the bug this check exists to prevent.
        assert_eq!(
            std::fs::read_to_string(p).unwrap(),
            "the quick brown fox, edited elsewhere"
        );
    }

    #[tokio::test]
    async fn write_if_unchanged_succeeds_when_the_mtime_is_unchanged() {
        // The ordinary, non-racing path: nothing touched the file between the read and the write, so
        // the mtime comparison must not spuriously reject it.
        let f = write_tmp("the quick brown fox");
        let p = f.path().to_str().unwrap();
        let (raw, initial_mtime) = read_with_mtime(p).unwrap();
        write_if_unchanged(p, initial_mtime, raw.replace("quick", "slow").as_bytes()).unwrap();
        assert_eq!(std::fs::read_to_string(p).unwrap(), "the slow brown fox");
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
