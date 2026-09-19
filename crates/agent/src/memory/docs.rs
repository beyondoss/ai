//! Document-map semantics shared by every non-filesystem [`MemoryBackend`].
//!
//! A store is a `rel → body` map. Directories are implicit prefixes of those keys, not first-class
//! entries — there is nothing to persist for an empty folder, and a `view` of a prefix synthesizes
//! the listing. That is the natural model for a hash, a SQL table, or an in-process map.
//!
//! [`super::map`] holds the map in process. [`super::redis`] and [`super::postgres`] use the same
//! helpers, but they fetch **one document** (or keys + sizes) rather than the whole project — a
//! `view` of `notes.md` is one `HGET` / `SELECT`, and the injected index is `MEMORY.md` only.
//! Search still scans bodies; prefix `create`/`rename`/`delete` consult the key set.
//!
//! Text helpers ([`cap_index`], [`slice_range`], [`str_replace_once`], [`insert_at_line`],
//! [`search_in`]) are also used by [`super::file::FileBackend`] so a `str_replace` means the same
//! thing on disk as it does on Redis.

use std::collections::{BTreeMap, BTreeSet};

use super::{Entry, Hit, INDEX_FILE, INDEX_MAX_BYTES, INDEX_MAX_LINES, MemPath, MemoryError, View};

/// Keep the first [`INDEX_MAX_LINES`] lines / [`INDEX_MAX_BYTES`] bytes of the index — whichever
/// bites first — so the always-injected prefix stays bounded.
pub fn cap_index(raw: &str) -> String {
    let mut out = String::new();
    for (i, line) in raw.lines().enumerate() {
        if i >= INDEX_MAX_LINES {
            out.push_str("\n[index truncated: showing the first ");
            out.push_str(&INDEX_MAX_LINES.to_string());
            out.push_str(" lines]");
            break;
        }
        if out.len() + line.len() + 1 > INDEX_MAX_BYTES {
            out.push_str("\n[index truncated at ~25 KB]");
            break;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// 1-indexed inclusive line slice, matching the text-editor `view_range` semantics.
pub fn slice_range(text: &str, range: Option<(usize, usize)>) -> String {
    let Some((start, end)) = range else {
        return text.to_string();
    };
    let start = start.max(1);
    let lines: Vec<&str> = text.lines().collect();
    if start > lines.len() {
        return String::new();
    }
    let end = end.min(lines.len());
    if end >= start {
        lines[start - 1..end].join("\n")
    } else {
        String::new()
    }
}

/// Replace the single occurrence of `old` with `new`. [`MemoryError::NotUnique`] unless it matched
/// exactly once.
pub fn str_replace_once(
    text: &str,
    old: &str,
    new: &str,
    path: &str,
) -> Result<String, MemoryError> {
    let count = text.matches(old).count();
    if count != 1 {
        return Err(MemoryError::NotUnique {
            path: path.to_string(),
            old: old.to_string(),
            count,
        });
    }
    Ok(text.replacen(old, new, 1))
}

/// Insert `text` after 1-indexed line `line` (`0` = the start of the document). Preserves a trailing
/// newline when the original had one (or was empty).
pub fn insert_at_line(existing: &str, line: usize, text: &str) -> String {
    let mut lines: Vec<&str> = existing.lines().collect();
    let at = line.min(lines.len());
    let inserted: Vec<&str> = text.split('\n').collect();
    for (offset, l) in inserted.into_iter().enumerate() {
        lines.insert(at + offset, l);
    }
    let mut joined = lines.join("\n");
    if existing.ends_with('\n') || existing.is_empty() {
        joined.push('\n');
    }
    joined
}

/// Case-insensitive substring hits in one document. `path` is the full logical path already.
pub fn search_in(path: &str, text: &str, needle_lower: &str) -> Vec<Hit> {
    if needle_lower.is_empty() {
        return Vec::new();
    }
    let mut hits = Vec::new();
    let mut lower = String::new();
    for (i, line) in text.lines().enumerate() {
        lower.clear();
        lower.extend(line.chars().flat_map(char::to_lowercase));
        if lower.contains(needle_lower) {
            hits.push(Hit {
                path: path.to_string(),
                line: i + 1,
                text: line.to_string(),
            });
        }
    }
    hits
}

/// Whether `rel` is a directory prefix of at least one stored document.
pub fn is_dir(docs: &BTreeMap<String, String>, rel: &str) -> bool {
    is_dir_keys(docs.keys(), rel)
}

/// [`is_dir`] against a key set — networked backends check prefixes without loading bodies.
pub fn is_dir_keys<'a, I, S>(keys: I, rel: &str) -> bool
where
    I: IntoIterator<Item = S>,
    S: AsRef<str> + 'a,
{
    if rel.is_empty() {
        return true;
    }
    let prefix = format!("{rel}/");
    keys.into_iter().any(|k| k.as_ref().starts_with(&prefix))
}

/// The index document, already bounded. Empty when `MEMORY.md` is absent.
pub fn index(docs: &BTreeMap<String, String>) -> String {
    docs.get(INDEX_FILE)
        .map(|raw| cap_index(raw))
        .unwrap_or_default()
}

/// View a document or a (possibly implicit) directory.
pub fn view(
    docs: &BTreeMap<String, String>,
    path: &MemPath,
    range: Option<(usize, usize)>,
    root: &str,
) -> Result<View, MemoryError> {
    if path.is_root() {
        return Ok(View::Listing(listing(docs, "", root)));
    }
    if let Some(text) = docs.get(path.rel()) {
        return Ok(View::Document(slice_range(text, range)));
    }
    if is_dir(docs, path.rel()) {
        return Ok(View::Listing(listing(docs, path.rel(), root)));
    }
    Err(MemoryError::NotFound(path.display()))
}

/// Create a *new* document. Refuses to clobber a file or to occupy a path that is already a
/// directory prefix — silently replacing durable knowledge is the failure this store exists to
/// prevent.
pub fn create(
    docs: &mut BTreeMap<String, String>,
    path: &MemPath,
    text: &str,
) -> Result<(), MemoryError> {
    if path.is_root() {
        return Err(MemoryError::InvalidPath(
            "cannot create the memory root itself".to_string(),
        ));
    }
    create_conflict(docs, path)?;
    docs.insert(path.rel().to_string(), text.to_string());
    Ok(())
}

/// `str_replace` against a stored document.
pub fn str_replace(
    docs: &mut BTreeMap<String, String>,
    path: &MemPath,
    old: &str,
    new: &str,
) -> Result<(), MemoryError> {
    let text = read_doc(docs, path)?.to_string();
    let replaced = str_replace_once(&text, old, new, &path.display())?;
    docs.insert(path.rel().to_string(), replaced);
    Ok(())
}

/// `insert` against a stored document.
pub fn insert(
    docs: &mut BTreeMap<String, String>,
    path: &MemPath,
    line: usize,
    text: &str,
) -> Result<(), MemoryError> {
    let existing = read_doc(docs, path)?.to_string();
    docs.insert(
        path.rel().to_string(),
        insert_at_line(&existing, line, text),
    );
    Ok(())
}

/// Delete a document. A directory prefix with children is refused (same as the file backend's
/// non-empty-dir rule). Implicit empty directories do not exist, so deleting one is [`NotFound`].
pub fn delete(docs: &mut BTreeMap<String, String>, path: &MemPath) -> Result<(), MemoryError> {
    if path.is_root() {
        return Err(MemoryError::InvalidPath(
            "cannot delete the memory root".to_string(),
        ));
    }
    if docs.remove(path.rel()).is_some() {
        return Ok(());
    }
    if is_dir(docs, path.rel()) {
        return Err(MemoryError::InvalidPath(format!(
            "{} is a non-empty directory; delete its contents first",
            path.display()
        )));
    }
    Err(MemoryError::NotFound(path.display()))
}

/// Move a document, or every document under a directory prefix. `to` must be unoccupied.
pub fn rename(
    docs: &mut BTreeMap<String, String>,
    from: &MemPath,
    to: &MemPath,
) -> Result<(), MemoryError> {
    if from.is_root() || to.is_root() {
        return Err(MemoryError::InvalidPath(
            "cannot rename the memory root".to_string(),
        ));
    }
    if from.rel() == to.rel() {
        return Ok(());
    }
    let from_is_file = docs.contains_key(from.rel());
    let from_is_dir = is_dir(docs, from.rel());
    if !from_is_file && !from_is_dir {
        return Err(MemoryError::NotFound(from.display()));
    }
    if docs.contains_key(to.rel()) || is_dir(docs, to.rel()) {
        return Err(MemoryError::AlreadyExists(to.display()));
    }
    // Refuse to move a directory inside itself (`sub` → `sub/nested`).
    if from_is_dir && (to.rel() == from.rel() || to.rel().starts_with(&format!("{}/", from.rel())))
    {
        return Err(MemoryError::InvalidPath(format!(
            "cannot rename {} into itself",
            from.display()
        )));
    }
    ancestor_is_file(docs, to.rel(), to)?;

    if from_is_file {
        let Some(body) = docs.remove(from.rel()) else {
            return Err(MemoryError::NotFound(from.display()));
        };
        docs.insert(to.rel().to_string(), body);
        return Ok(());
    }

    let prefix = format!("{}/", from.rel());
    let moving: Vec<(String, String)> = docs
        .iter()
        .filter(|(k, _)| k.starts_with(&prefix))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    for (k, _) in &moving {
        docs.remove(k);
    }
    for (k, v) in moving {
        let tail = &k[prefix.len()..];
        docs.insert(format!("{}/{tail}", to.rel()), v);
    }
    Ok(())
}

/// Case-insensitive substring search across every document, in path then line order.
pub fn search(docs: &BTreeMap<String, String>, query: &str, root: &str) -> Vec<Hit> {
    let needle = query.to_lowercase();
    if needle.is_empty() {
        return Vec::new();
    }
    let mut hits = Vec::new();
    for (rel, text) in docs {
        let path = format!("{root}/{rel}");
        hits.extend(search_in(&path, text, &needle));
    }
    hits
}

fn read_doc<'a>(
    docs: &'a BTreeMap<String, String>,
    path: &MemPath,
) -> Result<&'a str, MemoryError> {
    if let Some(text) = docs.get(path.rel()) {
        return Ok(text);
    }
    if is_dir(docs, path.rel()) {
        return Err(MemoryError::InvalidPath(format!(
            "{} is a directory, not a document",
            path.display()
        )));
    }
    Err(MemoryError::NotFound(path.display()))
}

fn create_conflict(docs: &BTreeMap<String, String>, path: &MemPath) -> Result<(), MemoryError> {
    if docs.contains_key(path.rel()) {
        return Err(MemoryError::AlreadyExists(path.display()));
    }
    if is_dir(docs, path.rel()) {
        return Err(MemoryError::InvalidPath(format!(
            "{} is a directory",
            path.display()
        )));
    }
    ancestor_is_file(docs, path.rel(), path)
}

/// [`create`] conflict checks against a key set — networked backends do not load bodies.
pub fn create_conflict_keys(keys: &BTreeSet<String>, path: &MemPath) -> Result<(), MemoryError> {
    if keys.contains(path.rel()) {
        return Err(MemoryError::AlreadyExists(path.display()));
    }
    if is_dir_keys(keys.iter(), path.rel()) {
        return Err(MemoryError::InvalidPath(format!(
            "{} is a directory",
            path.display()
        )));
    }
    ancestor_is_file_keys(keys, path.rel(), path)
}

/// A path cannot live under a document (`a.md/b` when `a.md` exists) — there is no directory there.
fn ancestor_is_file(
    docs: &BTreeMap<String, String>,
    rel: &str,
    path: &MemPath,
) -> Result<(), MemoryError> {
    let mut acc = String::new();
    for comp in rel.split('/') {
        if !acc.is_empty() && docs.contains_key(&acc) {
            return Err(MemoryError::InvalidPath(format!(
                "{} sits under document `{acc}`, which is not a directory",
                path.display()
            )));
        }
        if !acc.is_empty() {
            acc.push('/');
        }
        acc.push_str(comp);
    }
    Ok(())
}

/// [`ancestor_is_file`] against a key set.
pub fn ancestor_is_file_keys(
    keys: &BTreeSet<String>,
    rel: &str,
    path: &MemPath,
) -> Result<(), MemoryError> {
    let mut acc = String::new();
    for comp in rel.split('/') {
        if !acc.is_empty() && keys.contains(&acc) {
            return Err(MemoryError::InvalidPath(format!(
                "{} sits under document `{acc}`, which is not a directory",
                path.display()
            )));
        }
        if !acc.is_empty() {
            acc.push('/');
        }
        acc.push_str(comp);
    }
    Ok(())
}

/// Recursive listing of every document under `under` plus a directory entry for each implicit
/// prefix — matching [`super::file::FileBackend`]'s walk (the whole subtree, not just immediate
/// children).
fn listing(docs: &BTreeMap<String, String>, under: &str, root: &str) -> Vec<Entry> {
    listing_with_sizes(
        docs.iter()
            .map(|(rel, body)| (rel.as_str(), body.len() as u64)),
        under,
        root,
    )
}

/// [`listing`] from `(rel, byte-size)` pairs — a directory view does not need document bodies.
pub fn listing_with_sizes<'a, I>(docs: I, under: &str, root: &str) -> Vec<Entry>
where
    I: IntoIterator<Item = (&'a str, u64)>,
{
    let prefix = if under.is_empty() {
        String::new()
    } else {
        format!("{under}/")
    };
    let mut dirs = BTreeSet::new();
    let mut out = Vec::new();
    for (rel, size) in docs {
        if !under.is_empty() && rel != under && !rel.starts_with(&prefix) {
            continue;
        }
        if rel == under {
            continue;
        }
        let rest = if under.is_empty() {
            rel
        } else {
            &rel[prefix.len()..]
        };
        out.push(Entry {
            path: format!("{root}/{rel}"),
            is_dir: false,
            size,
        });
        let mut acc = under.to_string();
        for comp in rest.split('/') {
            if acc.is_empty() {
                acc = comp.to_string();
            } else {
                acc = format!("{acc}/{comp}");
            }
            if acc != rel {
                dirs.insert(acc.clone());
            }
        }
    }
    for d in dirs {
        out.push(Entry {
            path: format!("{root}/{d}"),
            is_dir: true,
            size: 0,
        });
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MEMORY_ROOT;

    fn p(s: &str) -> MemPath {
        MemPath::parse(s).unwrap()
    }

    #[test]
    fn cap_index_bounds_lines() {
        let big: String = (0..500).map(|i| format!("line {i}\n")).collect();
        let capped = cap_index(&big);
        assert!(capped.lines().count() <= INDEX_MAX_LINES + 3);
        assert!(capped.lines().filter(|l| l.starts_with("line ")).count() <= INDEX_MAX_LINES);
        assert!(capped.contains("index truncated"));
    }

    #[test]
    fn create_view_rename_and_search_on_a_map() {
        let mut docs = BTreeMap::new();
        create(&mut docs, &p("/memories/sub/a.md"), "hello\nworld\n").unwrap();
        create(&mut docs, &p("/memories/notes.md"), "The Build Command\n").unwrap();

        let View::Listing(entries) = view(&docs, &p("/memories"), None, MEMORY_ROOT).unwrap()
        else {
            panic!("expected a listing");
        };
        assert!(
            entries
                .iter()
                .any(|e| e.path == "/memories/notes.md" && !e.is_dir)
        );
        assert!(
            entries
                .iter()
                .any(|e| e.path == "/memories/sub" && e.is_dir)
        );
        assert!(entries.iter().any(|e| e.path == "/memories/sub/a.md"));

        let View::Document(t) =
            view(&docs, &p("/memories/sub/a.md"), Some((2, 2)), MEMORY_ROOT).unwrap()
        else {
            panic!("expected a document");
        };
        assert_eq!(t, "world");

        assert!(matches!(
            create(&mut docs, &p("/memories/notes.md"), "x"),
            Err(MemoryError::AlreadyExists(_))
        ));
        assert!(matches!(
            create(&mut docs, &p("/memories/sub"), "x"),
            Err(MemoryError::InvalidPath(_))
        ));

        rename(&mut docs, &p("/memories/sub"), &p("/memories/moved")).unwrap();
        assert!(docs.contains_key("moved/a.md"));
        assert!(!docs.contains_key("sub/a.md"));

        let hits = search(&docs, "build command", MEMORY_ROOT);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "/memories/notes.md");
    }

    #[test]
    fn delete_refuses_a_non_empty_prefix() {
        let mut docs = BTreeMap::new();
        create(&mut docs, &p("/memories/sub/a.md"), "x").unwrap();
        assert!(matches!(
            delete(&mut docs, &p("/memories/sub")),
            Err(MemoryError::InvalidPath(_))
        ));
        delete(&mut docs, &p("/memories/sub/a.md")).unwrap();
        assert!(matches!(
            delete(&mut docs, &p("/memories/sub")),
            Err(MemoryError::NotFound(_))
        ));
    }

    #[test]
    fn prefix_checks_and_listings_work_from_keys_alone() {
        let keys: BTreeSet<String> = ["notes.md".into(), "sub/a.md".into()].into();
        assert!(is_dir_keys(keys.iter(), "sub"));
        assert!(!is_dir_keys(keys.iter(), "notes.md"));
        assert!(matches!(
            create_conflict_keys(&keys, &p("/memories/notes.md")),
            Err(MemoryError::AlreadyExists(_))
        ));
        assert!(matches!(
            create_conflict_keys(&keys, &p("/memories/sub")),
            Err(MemoryError::InvalidPath(_))
        ));
        create_conflict_keys(&keys, &p("/memories/fresh.md")).unwrap();
        let entries = listing_with_sizes([("notes.md", 3u64), ("sub/a.md", 10)], "", MEMORY_ROOT);
        assert!(
            entries
                .iter()
                .any(|e| e.path == "/memories/notes.md" && e.size == 3 && !e.is_dir)
        );
        assert!(
            entries
                .iter()
                .any(|e| e.path == "/memories/sub" && e.is_dir && e.size == 0)
        );
    }
}
