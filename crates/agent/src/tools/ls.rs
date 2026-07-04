//! `ls` — list a directory's entries (directories suffixed with `/`).
//!
//! Two deliberate divergences from the reference agent, kept rather than "fixed" to match: dotfiles are
//! hidden by default (`all: true` opts back in — cuts real noise like `.git`/editor swapfiles without
//! losing access), and directories are always sorted before files (a stable UX improvement independent
//! of parity with anything else).
//!
//! Matches the reference agent on: a missing path or a path that isn't a directory gets a clean, specific
//! error (`Path not found: {path}` / `Not a directory: {path}`) rather than a raw OS errno string; each
//! entry's directory-ness is decided by following symlinks (`std::fs::metadata`, not the lstat-like
//! `DirEntry::file_type`), so a symlink to a directory gets the `/` suffix; and an entry that can't be
//! stat'd at all (permission race, dangling symlink) is silently skipped rather than guessed at.

use agent_core::tool::Tool;
use agent_core::{ToolError, ToolOutput};
use async_trait::async_trait;
use serde_json::{Value, json};
use unicode_normalization::UnicodeNormalization;

/// Default cap on entries returned before truncating, to protect the model's context from a
/// `node_modules`-sized directory. Overridable per call via the `limit` argument. The model can also
/// narrow with a more specific path or `find`/`grep`.
const DEFAULT_LIMIT: usize = 500;

/// A cheap Unicode-aware sort key: lowercase, then NFD-decompose and drop combining marks, so an
/// accented Latin letter collates next to its unaccented form (e.g. "café" sorts next to "cafe", ahead
/// of "cafz") instead of by raw codepoint (where "é" is U+00E9 — after "z" — so a plain codepoint
/// compare would sort "café.txt" *after* "cafz.txt", nowhere near where a human, or pi's own
/// `localeCompare`, would place it). Not full Unicode collation (DUCET; that needs an ICU-sized
/// dependency this crate deliberately doesn't take just for `ls`'s display order) — but `unicode-
/// normalization` is already linked directly by this crate (`edit.rs`'s NFKC folding), so this reuses
/// machinery already paid for rather than adding anything new for a cosmetic-only improvement.
fn collation_key(name: &str) -> String {
    name.to_lowercase()
        .nfd()
        .filter(|c| !unicode_normalization::char::is_combining_mark(*c))
        .collect()
}

pub struct Ls;

#[async_trait]
impl Tool for Ls {
    fn name(&self) -> &str {
        "ls"
    }
    fn description(&self) -> &str {
        "List the entries of a directory. Directories are suffixed with `/`. Hidden entries are \
         shown only when `all` is true."
    }
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Directory to list (default \".\")." },
                "all": { "type": "boolean", "description": "Include dot-files (default false)." },
                "limit": { "type": "integer", "description": "Max entries to list before truncating (default 500)." }
            }
        })
    }

    async fn run(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let path = input.get("path").and_then(Value::as_str).unwrap_or(".");
        let path = &super::normalize_path(path);
        let all = input.get("all").and_then(Value::as_bool).unwrap_or(false);
        let limit = input
            .get("limit")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(DEFAULT_LIMIT);

        // Sort by the bare name, not the display string (which gets a trailing `/` for directories) —
        // comparing display strings inverts order for sibling names sharing a prefix (e.g. "src" vs
        // "src-old": '-' sorts below '/' byte-wise, so "src-old/" would sort before "src/"). The
        // display suffix is appended only after sorting, matching the reference agent's own two-step
        // sort-then-suffix order.
        // Pi-parity fix: distinguish "doesn't exist" from "not a directory" with clean, specific
        // messages (matching pi's own `Path not found: {path}`/`Not a directory: {path}`) before ever
        // calling `read_dir` — rather than a raw OS errno string covering both. `metadata` (not
        // `symlink_metadata`) follows a symlink to check the *target*, matching pi's own `ops.stat`
        // (Node's `fs.stat` follows symlinks) and ordinary `ls`'s own expectation that listing a
        // symlink-to-a-directory lists the target's contents.
        match std::fs::metadata(path) {
            Ok(meta) => {
                if !meta.is_dir() {
                    return Err(ToolError::Execution(format!("Not a directory: {path}")));
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(ToolError::Execution(format!("Path not found: {path}")));
            }
            Err(e) => return Err(ToolError::Execution(format!("ls {path}: {e}"))),
        }
        let mut entries: Vec<(String, bool)> = Vec::new();
        let dir =
            std::fs::read_dir(path).map_err(|e| ToolError::Execution(format!("ls {path}: {e}")))?;
        for entry in dir {
            let entry = entry.map_err(|e| ToolError::Execution(e.to_string()))?;
            let fname = entry.file_name();
            let name = fname.to_string_lossy();
            if !all && name.starts_with('.') {
                continue;
            }
            // Pi-parity fix: `DirEntry::file_type` is lstat-like — it reports a symlink's own type, not
            // its target's, so a symlink pointing at a directory was mislabeled as a plain file (no
            // trailing `/`). pi's own `ls.ts` stats each entry with `fs.stat` (follows symlinks), so use
            // `std::fs::metadata` on the full entry path here to match. This also subsumes the
            // unstattable-entry case: a vanished-between-readdir-and-here entry or a dangling symlink
            // both fail `metadata` and are skipped (hidden), matching pi's own catch-and-skip behavior,
            // instead of silently guessing `is_dir = false` for something we couldn't actually confirm.
            let Ok(meta) = std::fs::metadata(entry.path()) else {
                continue;
            };
            entries.push((name.into_owned(), meta.is_dir()));
        }
        // Directories first, then case-insensitive alphabetical by bare name, collated in a way that
        // groups an accented letter with its unaccented form (`collation_key`) rather than a raw
        // codepoint compare — matches pi's `a.toLowerCase().localeCompare(b.toLowerCase())` far more
        // closely than a byte-order compare would (which would also sort every uppercase-leading name
        // before every lowercase one, e.g. "Zebra.txt" before "apple.txt", unlike what a human — or pi —
        // would expect). Falls back to a case-sensitive compare only to break an exact tie (e.g.
        // "Foo"/"foo" coexisting in one directory), so the order stays deterministic instead of
        // depending on the OS's arbitrary `readdir` order.
        entries.sort_by(|(a_name, a_dir), (b_name, b_dir)| {
            b_dir.cmp(a_dir).then_with(|| {
                collation_key(a_name)
                    .cmp(&collation_key(b_name))
                    .then_with(|| a_name.cmp(b_name))
            })
        });
        let mut entries: Vec<String> = entries
            .into_iter()
            .map(|(name, is_dir)| {
                // Built with room for the trailing `/` so a directory entry doesn't allocate a second
                // String beyond this one.
                let mut display = String::with_capacity(name.len() + usize::from(is_dir));
                display.push_str(&name);
                if is_dir {
                    display.push('/');
                }
                display
            })
            .collect();
        if entries.is_empty() {
            return Ok("(empty directory)".into());
        }
        // Cap the listing so a huge directory can't flood the model's context.
        let total = entries.len();
        let count_truncated = total > limit;
        if count_truncated {
            entries.truncate(limit);
        }
        let mut out = entries.join("\n");
        if count_truncated {
            out.push('\n');
        }
        // The byte cap is checked *before* the count marker, and takes priority when both would
        // otherwise fire — see `cap_listing_bytes`'s doc comment.
        let byte_capped = super::output::cap_listing_bytes(
            &mut out,
            "narrow with a subpath or use `find`/`grep` to see more",
        );
        if !byte_capped && count_truncated {
            out.push_str(&super::output::marker(format_args!(
                "{} more entries; {total} total — narrow with a subpath or use `find`/`grep`",
                total - limit
            )));
        }
        Ok(out.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn lists_dirs_first_and_hides_dotfiles() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("subdir")).unwrap();
        std::fs::write(dir.path().join("file.txt"), "x").unwrap();
        std::fs::write(dir.path().join(".hidden"), "x").unwrap();

        let out = Ls
            .run(json!({ "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert_eq!(out, "subdir/\nfile.txt");

        let all = Ls
            .run(json!({ "path": dir.path().to_str().unwrap(), "all": true }))
            .await
            .unwrap()
            .text;
        assert!(all.contains(".hidden"));
    }

    #[tokio::test]
    async fn reports_a_clean_not_found_message_for_a_missing_path() {
        // Pi-parity fix: a raw OS errno string ("No such file or directory (os error 2)") covered both
        // "doesn't exist" and "not a directory" alike — pi's own `ls.ts` distinguishes them with clean,
        // specific messages.
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("does-not-exist");
        let err = Ls
            .run(json!({ "path": missing.to_str().unwrap() }))
            .await
            .unwrap_err();
        let ToolError::Execution(msg) = err else {
            panic!("expected Execution, got {err:?}");
        };
        assert!(
            msg.contains(&format!("Path not found: {}", missing.to_str().unwrap())),
            "got: {msg}"
        );
        assert!(
            !msg.contains("os error"),
            "must not leak raw OS errno text: {msg}"
        );
    }

    #[tokio::test]
    async fn reports_a_clean_not_a_directory_message_for_a_plain_file() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("plain.txt");
        std::fs::write(&file, "x").unwrap();
        let err = Ls
            .run(json!({ "path": file.to_str().unwrap() }))
            .await
            .unwrap_err();
        let ToolError::Execution(msg) = err else {
            panic!("expected Execution, got {err:?}");
        };
        assert_eq!(msg, format!("Not a directory: {}", file.to_str().unwrap()));
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn a_symlink_to_a_directory_is_shown_with_the_directory_suffix() {
        // Pi-parity fix: `DirEntry::file_type()` is lstat-like and reported the symlink's own type
        // (never a directory), not the target's — pi's `ls.ts` follows symlinks via `fs.stat` when
        // deciding the `/` suffix.
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("real_dir");
        std::fs::create_dir(&target).unwrap();
        std::os::unix::fs::symlink(&target, dir.path().join("link_to_dir")).unwrap();
        let out = Ls
            .run(json!({ "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert!(
            out.contains("link_to_dir/"),
            "a symlink to a directory must be shown with a trailing slash: {out}"
        );
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn a_dangling_symlink_is_listed_as_a_plain_entry_not_mislabeled_a_directory() {
        // Related regression guard for the same fix: an entry whose type this process can't fully
        // confirm (here, a symlink target that doesn't exist) must never be mislabeled as a directory —
        // it should either be skipped (unstattable) or shown as a plain, unsuffixed entry, but never
        // get a `/` it hasn't earned.
        let dir = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(dir.path().join("nowhere"), dir.path().join("dangling"))
            .unwrap();
        let out = Ls
            .run(json!({ "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert!(
            !out.contains("dangling/"),
            "a dangling symlink must never be shown as a directory: {out}"
        );
    }

    #[tokio::test]
    async fn sorts_by_bare_name_not_the_slash_suffixed_display_string() {
        // MEDIUM pi-parity gap (fixed): sorting the display string (with `/` already appended) instead
        // of the bare name inverts order for sibling names sharing a prefix — '-' (0x2D) sorts below
        // '/' (0x2F) byte-wise, so "src-old/" would sort before "src/" even though a human (and the
        // reference agent, which sorts bare names then appends the suffix) expects "src" first since
        // it's a strict prefix of "src-old".
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("src-old")).unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        let out = Ls
            .run(json!({ "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert_eq!(out, "src/\nsrc-old/", "got: {out}");
    }

    #[tokio::test]
    async fn sorts_case_insensitively_like_pi_not_by_raw_byte_order() {
        // pi-parity fix: a byte-order compare sorts every uppercase-leading name before every
        // lowercase one ('Z' is 0x5A, 'a' is 0x61) — "Zebra.txt" would sort before "apple.txt", unlike
        // pi's `a.toLowerCase().localeCompare(b.toLowerCase())` or what a human expects.
        let dir = tempfile::tempdir().unwrap();
        for name in ["Zebra.txt", "apple.txt", "banana.txt"] {
            std::fs::write(dir.path().join(name), "x").unwrap();
        }
        let out = Ls
            .run(json!({ "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert_eq!(out, "apple.txt\nbanana.txt\nZebra.txt", "got: {out}");
    }

    #[tokio::test]
    async fn accented_names_collate_near_their_unaccented_form_not_by_raw_codepoint() {
        // pi-parity fix (task 47): a raw codepoint compare sorts "café.txt" (with U+00E9) *after*
        // "cafz.txt" ('z' is U+007A, well below U+00E9), unlike pi's own `localeCompare`, which groups
        // an accented letter with its unaccented form — "café.txt" should land between "cafe.txt" and
        // "cafz.txt", not after both.
        let dir = tempfile::tempdir().unwrap();
        for name in ["cafz.txt", "café.txt", "cafe.txt"] {
            std::fs::write(dir.path().join(name), "x").unwrap();
        }
        let out = Ls
            .run(json!({ "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert_eq!(out, "cafe.txt\ncafé.txt\ncafz.txt", "got: {out}");
    }

    #[tokio::test]
    async fn breaks_a_case_insensitive_tie_deterministically() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Foo.txt"), "x").unwrap();
        std::fs::write(dir.path().join("foo.txt"), "x").unwrap();
        let out = Ls
            .run(json!({ "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert_eq!(out, "Foo.txt\nfoo.txt", "got: {out}");
    }

    #[tokio::test]
    async fn run_normalizes_the_path_argument() {
        // Proves `run` actually calls `super::normalize_path`, via its `@`-prefix-strip behavior
        // (needs no `$HOME` mutation — see `expand_tilde`'s own direct unit tests for that half).
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("file.txt"), "x").unwrap();
        let at_prefixed = format!("@{}", dir.path().to_str().unwrap());
        let out = Ls.run(json!({ "path": at_prefixed })).await.unwrap().text;
        assert!(out.contains("file.txt"));
    }

    #[tokio::test]
    async fn caps_a_huge_directory() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..(DEFAULT_LIMIT + 50) {
            std::fs::write(dir.path().join(format!("f{i:04}")), "x").unwrap();
        }
        let out = Ls
            .run(json!({ "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("more entries"));
        // The body is capped to DEFAULT_LIMIT lines (plus the truncation note).
        assert_eq!(out.lines().count(), DEFAULT_LIMIT + 1);
    }

    #[tokio::test]
    async fn limit_param_overrides_the_default_cap() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..10 {
            std::fs::write(dir.path().join(format!("f{i:02}")), "x").unwrap();
        }
        let out = Ls
            .run(json!({ "path": dir.path().to_str().unwrap(), "limit": 3 }))
            .await
            .unwrap()
            .text;
        assert!(
            out.contains("more entries"),
            "a small limit must truncate: {out}"
        );
        assert!(out.contains("[7 more entries; 10 total"));
        // Three entry lines plus the truncation note.
        assert_eq!(out.lines().count(), 4);
    }

    #[tokio::test]
    async fn output_byte_cap_truncates_even_under_the_entry_limit() {
        // 300 entries with long names: well under the 500-entry default `limit` (so the entry-count
        // marker never fires), but the aggregate listing still blows past the 50KB output cap on its
        // own.
        let dir = tempfile::tempdir().unwrap();
        for i in 0..300 {
            let name = format!("{i:04}-{}", "x".repeat(200));
            std::fs::write(dir.path().join(name), "x").unwrap();
        }
        let out = Ls
            .run(json!({ "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert!(
            out.len() <= super::super::output::MAX_LISTING_BYTES + 256,
            "output should be capped near MAX_LISTING_BYTES, got {} bytes",
            out.len()
        );
        assert!(
            out.contains("[output truncated at 50.0KB"),
            "byte-cap marker missing: {}",
            &out[out.len().saturating_sub(200)..]
        );
        assert!(
            !out.contains("more entries;"),
            "entry-count marker must not co-fire when count never exceeded the limit"
        );
    }

    #[tokio::test]
    async fn output_byte_cap_takes_priority_over_the_entry_count_marker() {
        // 600 long-named entries against the default 500-entry `limit`: both the entry-count marker
        // (600 > 500) and the byte cap would fire on the same rendered text. The byte cap must win
        // outright rather than leaving a marker sliced in half.
        let dir = tempfile::tempdir().unwrap();
        for i in 0..600 {
            let name = format!("{i:04}-{}", "x".repeat(200));
            std::fs::write(dir.path().join(name), "x").unwrap();
        }
        let out = Ls
            .run(json!({ "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert!(
            out.contains("[output truncated at 50.0KB"),
            "byte-cap marker missing: {out}"
        );
        assert!(
            !out.contains("more entries;"),
            "the byte-cap marker must win cleanly, not leave a mangled count marker behind"
        );
        assert!(
            out.len() <= super::super::output::MAX_LISTING_BYTES + 256,
            "output should be capped near MAX_LISTING_BYTES, got {} bytes",
            out.len()
        );
    }
}
