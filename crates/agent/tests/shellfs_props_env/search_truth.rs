//! A brute-force search over the real tree: the oracle for `ShellFs::search`.
//!
//! ## The pattern subset both sides agree on
//!
//! `literal` queries are any string without a newline. Regex queries are restricted to the subset on
//! which Rust's regex (ripgrep), POSIX ERE (GNU grep) and this matcher agree: ASCII letters, digits,
//! space and `_`, alternation `a|b`, and a leading `^` / trailing `$` on an alternative. Nothing else
//! (no classes, groups, repetition, backslashes): those are where the dialects differ, and the spec
//! says so (FILES-OVER-EXEC "Engines differ").
//!
//! ## What is required and what is merely allowed
//!
//! A file is **included** when the engine is specified to search it, **excluded** when the engine is
//! specified to skip it (a NUL in the first 32 KiB), and **ambiguous** when the spec leaves it to the
//! engine (a NUL later in a big file; invalid UTF-8 under GNU grep, whose answer depends on the locale).
//! A search whose tree has no ambiguous or unreadable entry is compared exactly. Otherwise every
//! returned hit must still be a true hit (path, line number, text, match flag) and the order and the
//! limit rule must hold, but completeness is not demanded.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::truth::can;

pub const MAX_LINE: usize = 500;
const FIRST_BUFFER: usize = 32 * 1024;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Engine {
    Rg,
    Grep,
}

#[derive(Clone, Debug)]
pub struct Spec {
    pub pattern: String,
    pub literal: bool,
    pub ignore_case: bool,
    /// `*.ext`, and whether it excludes.
    pub glob: Option<(String, bool)>,
    pub limit: usize,
    pub before: usize,
    pub after: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Th {
    pub path: String,
    pub line: usize,
    pub text: String,
    pub is_match: bool,
}

pub struct Expect {
    pub root_missing: bool,
    pub unreadable: bool,
    pub exact: bool,
    pub hits: Vec<Th>,
    pub truncated: bool,
}

struct Alt {
    lit: Vec<char>,
    start: bool,
    end: bool,
}

pub struct Matcher {
    alts: Vec<Alt>,
    ic: bool,
}

fn fold(c: char, ic: bool) -> char {
    if ic {
        c.to_lowercase().next().unwrap_or(c)
    } else {
        c
    }
}

impl Matcher {
    pub fn new(s: &Spec) -> Self {
        let ic = s.ignore_case;
        let mk = |lit: &str, start, end| Alt {
            lit: lit.chars().map(|c| fold(c, ic)).collect(),
            start,
            end,
        };
        let alts = if s.literal {
            vec![mk(&s.pattern, false, false)]
        } else {
            s.pattern
                .split('|')
                .map(|a| {
                    let (start, a) = a.strip_prefix('^').map_or((false, a), |r| (true, r));
                    let (end, a) = a.strip_suffix('$').map_or((false, a), |r| (true, r));
                    mk(a, start, end)
                })
                .collect()
        };
        Self { alts, ic }
    }

    /// The leftmost match, as char offsets.
    pub fn find(&self, line: &str) -> Option<(usize, usize)> {
        let l: Vec<char> = line.chars().map(|c| fold(c, self.ic)).collect();
        let mut best: Option<(usize, usize)> = None;
        for a in &self.alts {
            let n = a.lit.len();
            let m = if n > l.len() {
                None
            } else if a.start && a.end {
                (l == a.lit).then_some((0, n))
            } else if a.start {
                l.starts_with(&a.lit).then_some((0, n))
            } else if a.end {
                l.ends_with(&a.lit).then_some((l.len() - n, l.len()))
            } else {
                l.windows(n.max(1))
                    .position(|w| w == a.lit.as_slice())
                    .map(|i| (i, i + n))
            };
            if let Some(m) = m
                && best.is_none_or(|b| m.0 < b.0)
            {
                best = Some(m);
            }
        }
        best
    }
}

/// The spec's clipping (FILES-OVER-EXEC "Clipping").
pub fn clip(text: &str, is_match: bool, m: &Matcher) -> String {
    let chars: Vec<char> = text.chars().collect();
    let total = chars.len();
    if total <= MAX_LINE {
        return text.to_string();
    }
    let head = |chars: &[char]| {
        let mut s: String = chars[..MAX_LINE].iter().collect();
        s.push_str("… [truncated]");
        s
    };
    if !is_match {
        return head(&chars);
    }
    let Some((cs, ce)) = m.find(text) else {
        return head(&chars);
    };
    if ce <= MAX_LINE {
        return head(&chars);
    }
    let pad = MAX_LINE.saturating_sub(ce - cs) / 2;
    let mut start = cs.saturating_sub(pad);
    let mut end = start + MAX_LINE;
    if end > total {
        end = total;
        start = total - MAX_LINE;
    }
    let mut s = String::from("…");
    s.extend(&chars[start..end]);
    if end < total {
        s.push('…');
    }
    s.push_str(&format!(
        " [truncated; columns {}-{} of {}]",
        start + 1,
        end,
        total
    ));
    s
}

#[derive(PartialEq)]
enum Class {
    Included,
    Excluded,
    Ambiguous,
}

fn classify_file(bytes: &[u8], engine: Engine, explicit: bool) -> Class {
    let nul = bytes.contains(&0);
    if nul {
        return if explicit {
            Class::Ambiguous
        } else if bytes.len() <= FIRST_BUFFER || bytes[..FIRST_BUFFER].contains(&0) {
            Class::Excluded
        } else {
            Class::Ambiguous
        };
    }
    if engine == Engine::Grep && std::str::from_utf8(bytes).is_err() {
        return Class::Ambiguous;
    }
    Class::Included
}

fn components_key(p: &str) -> Vec<Vec<u8>> {
    p.split('/').map(|c| c.as_bytes().to_vec()).collect()
}

pub fn expect(
    root: &Path,
    map: &dyn Fn(&Path) -> String,
    spec: &Spec,
    engine: Engine,
) -> Expect {
    let mut out = Expect {
        root_missing: false,
        unreadable: false,
        exact: true,
        hits: vec![],
        truncated: false,
    };
    let Ok(rm) = std::fs::metadata(root) else {
        out.root_missing = std::fs::symlink_metadata(root).is_err()
            && !matches!(root.parent().map(std::fs::metadata), Some(Err(ref e)) if e.kind() == std::io::ErrorKind::PermissionDenied);
        out.exact = false;
        return out;
    };
    let mut files: Vec<(PathBuf, bool)> = Vec::new(); // (path, explicit)
    if rm.is_dir() {
        collect(root, spec, &mut files, &mut out.unreadable);
        if !can(root, 0o5) {
            out.unreadable = true;
        }
    } else if rm.is_file() {
        if can(root, 4) {
            files.push((root.to_path_buf(), true));
        } else {
            out.unreadable = true;
        }
    }
    let matcher = Matcher::new(spec);
    let mut by_file: BTreeMap<Vec<Vec<u8>>, Vec<Th>> = BTreeMap::new();
    let mut total_matches = 0usize;
    for (f, explicit) in &files {
        let bytes = std::fs::read(f).unwrap_or_default();
        match classify_file(&bytes, engine, *explicit) {
            Class::Excluded => continue,
            Class::Ambiguous => out.exact = false,
            Class::Included => {}
        }
        let path = map(f);
        let mut lines: Vec<&[u8]> = bytes.split(|b| *b == b'\n').collect();
        if lines.last().is_some_and(|l| l.is_empty()) {
            lines.pop();
        }
        let texts: Vec<String> = lines
            .iter()
            .map(|l| {
                let l = l.strip_suffix(b"\r").unwrap_or(l);
                String::from_utf8_lossy(l).into_owned()
            })
            .collect();
        let is_m: Vec<bool> = texts.iter().map(|t| matcher.find(t).is_some()).collect();
        let mut hits = Vec::new();
        let (before, after) = (spec.before.min(100), spec.after.min(100));
        for i in 0..texts.len() {
            let shown = is_m[i]
                || (i.saturating_sub(after)..i).any(|j| is_m[j])
                || ((i + 1)..=(i + before).min(texts.len() - 1)).any(|j| is_m[j]);
            if shown {
                total_matches += usize::from(is_m[i]);
                let t = texts[i].replace('\r', "");
                hits.push(Th {
                    path: path.clone(),
                    line: i + 1,
                    text: clip(&t, is_m[i], &matcher),
                    is_match: is_m[i],
                });
            }
        }
        if !hits.is_empty() {
            by_file.entry(components_key(&path)).or_default().extend(hits);
        }
    }
    let mut all: Vec<Th> = by_file.into_values().flatten().collect();
    // Per-file lists are already in line order; the map key orders files component-wise.
    if !out.exact || out.unreadable {
        out.exact = false;
        out.hits = all;
        return out;
    }
    let matches = all.iter().filter(|h| h.is_match).count();
    debug_assert_eq!(matches, total_matches);
    if matches > spec.limit {
        out.truncated = true;
        let mut seen = 0;
        let cut = all
            .iter()
            .position(|h| {
                if h.is_match {
                    seen += 1;
                }
                h.is_match && seen == spec.limit + 1
            })
            .unwrap_or(all.len());
        all.truncate(cut);
        while all.last().is_some_and(|h| !h.is_match) {
            all.pop();
        }
    }
    out.hits = all;
    out
}

fn collect(dir: &Path, spec: &Spec, files: &mut Vec<(PathBuf, bool)>, unreadable: &mut bool) {
    if !can(dir, 0o5) {
        *unreadable = true;
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else {
        *unreadable = true;
        return;
    };
    let mut entries: Vec<_> = rd.flatten().collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let p = e.path();
        if e.file_name() == ".git" {
            continue;
        }
        let Ok(m) = std::fs::symlink_metadata(&p) else {
            continue;
        };
        if m.file_type().is_symlink() {
            continue;
        }
        if m.is_dir() {
            collect(&p, spec, files, unreadable);
        } else if m.is_file() {
            if let Some((g, excl)) = &spec.glob {
                let suffix = g.trim_start_matches('*');
                let name = e.file_name().to_string_lossy().into_owned();
                if name.ends_with(suffix) == *excl {
                    continue;
                }
            }
            if !can(&p, 4) {
                *unreadable = true;
                continue;
            }
            files.push((p, false));
        }
    }
}
