//! The agent's structure stays where the refactor put it. Two rules over `crates/agent/src/**/*.rs`,
//! non-test code only:
//!
//! 1. **Mode checks live only in `mode.rs`.** Whether the process is local or a service replica is
//!    decided once, at startup, and every rule that depends on it is a method on `Mode` named for
//!    the rule (`mode.refused_command(..)`, `mode.reads_disk_overrides()`). Branching on the mode
//!    anywhere else puts a second copy of the rule next to the first, and the two drift. The rule
//!    flags, outside `mode.rs`:
//!    - `Mode::Local`, `Mode::Replica` and `Mode::Service`, in any position (an arm, a
//!      `matches!`, a constructor);
//!    - `service_mode` as an identifier;
//!    - `<path>.service.is_some()` / `.is_none()` and `service == ..` / `service != ..`;
//!    - `if let Some(x) = <path>.service` and `match <path>.service`.
//!
//!    It does not flag a method call named `.service()` (rmcp's peer handle in `tools/mcp.rs`), a
//!    string or comment that mentions these (an Azure host match in `gateway_credential.rs` is a
//!    string), or an identifier that merely contains the word (`reservice`, `service_url`).
//!    The sites that remain are in [`MODE_ALLOWED`], each with why it is not a rule about the mode.
//!
//! 2. **No function over [`BUDGET`] lines.** A function that long is several steps with one name. A
//!    function over budget today is in [`OVER_BUDGET`] with its current length and why it is still
//!    one function. The list is a ratchet: a listed function that grows fails, and so does any new
//!    function over budget. A listed function that has shrunk to within budget (or been renamed or
//!    removed) fails too, so the list only ever gets shorter.
//!
//! Length is counted from the line holding `fn` to the line of its closing brace, inclusive, in the
//! source with comments and string literals blanked so a brace in either cannot confuse the match.
//! Items under `#[cfg(test)]`, files named `tests.rs` or `*_tests.rs`, and any file under a
//! directory named `tests` are skipped. Functions are keyed `path::name`; two functions of one name
//! in one file are held to the longer one.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The longest a function may be, in lines.
const BUDGET: usize = 300;

/// Every file outside `mode.rs` with matches of the mode rule, how many, and why each is not a
/// rule about the mode. Adding one is a reviewed change here.
const MODE_ALLOWED: &[(&str, usize, &str)] = &[
    // The notify receiver's `service_mode` flag (the field and its one read) restates `hub.is_some()`.
    (
        "serve_ws/job_ended.rs",
        2,
        "Receiver.service_mode duplicates hub.is_some()",
    ),
    // The supervisor is process-level state that exists before any session has a `Mode`: it builds
    // the process `Mode` from whether a service config was given, picks its startup path and its
    // readiness probe by the same fact, and hands `verifies_grants()` to the notify receiver.
    (
        "serve_ws.rs",
        4,
        "supervisor config, before any session Mode exists",
    ),
];

/// Every function over [`BUDGET`]: `path::name`, its length today, and why it is one function.
const OVER_BUDGET: &[(&str, usize, &str)] = &[
    // The credential precedence ladder as one pure function, so every test drives the whole ladder
    // through a single seam.
    (
        "gateway_credential.rs::resolve_with",
        433,
        "the whole precedence ladder in one pure function",
    ),
    // The `bash` command: foreground, background and job-control paths share one set of locals.
    (
        "serve/commands/bash.rs::cmd_bash",
        321,
        "foreground, background and job paths share locals",
    ),
    // The busy-loop command table: one arm per command a running session answers.
    (
        "serve/control.rs::dispatch",
        383,
        "one match arm per control command",
    ),
    // The prompt busy loop, held inline because the run future holds `&mut session`.
    (
        "serve/prompt.rs::run_attempt",
        554,
        "busy loop inline: the run future holds &mut session",
    ),
    // Everything a prompt needs resolved before the run future is built.
    (
        "serve/session_setup.rs::setup_prompt",
        313,
        "sequential resolution of one prompt's inputs",
    ),
    // The WebSocket/daemon entry point: listeners, the accept loop and its per-connection task.
    (
        "serve_ws.rs::serve_ws",
        396,
        "listener setup and accept loop with per-connection closure",
    ),
    // Opening a session repo: probing the layout, then the three shapes of open.
    (
        "session_store/store.rs::open_with",
        409,
        "layout probe plus each open shape",
    ),
];

/// Source with comments and string/char literal contents blanked (newlines kept, so line numbers
/// hold).
fn blank(source: &str) -> Vec<char> {
    let s: Vec<char> = source.chars().collect();
    let mut out = s.clone();
    let wipe = |out: &mut Vec<char>, a: usize, b: usize| {
        for c in &mut out[a..b.min(s.len())] {
            if *c != '\n' {
                *c = ' ';
            }
        }
    };
    let mut i = 0;
    while i < s.len() {
        let c = s[i];
        let next = s.get(i + 1).copied();
        if c == '/' && next == Some('/') {
            let start = i;
            while i < s.len() && s[i] != '\n' {
                i += 1;
            }
            wipe(&mut out, start, i);
        } else if c == '/' && next == Some('*') {
            let start = i;
            let mut depth = 0;
            while i < s.len() {
                if s[i] == '/' && s.get(i + 1) == Some(&'*') {
                    depth += 1;
                    i += 2;
                } else if s[i] == '*' && s.get(i + 1) == Some(&'/') {
                    depth -= 1;
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    i += 1;
                }
            }
            wipe(&mut out, start, i);
        } else if (c == 'r' || (c == 'b' && next == Some('r')))
            && !(i > 0 && is_ident(s[i - 1]))
            && {
                let mut j = i + if c == 'b' { 2 } else { 1 };
                while s.get(j) == Some(&'#') {
                    j += 1;
                }
                s.get(j) == Some(&'"')
            }
        {
            // A raw string: `r#"…"#`, closed by a quote and the same number of hashes.
            let mut j = i + if c == 'b' { 2 } else { 1 };
            let mut hashes = 0;
            while s[j] == '#' {
                hashes += 1;
                j += 1;
            }
            let start = j + 1;
            let mut k = start;
            while k < s.len() {
                if s[k] == '"' && (0..hashes).all(|h| s.get(k + 1 + h) == Some(&'#')) {
                    break;
                }
                k += 1;
            }
            wipe(&mut out, start, k);
            i = k + 1 + hashes;
        } else if c == '"' {
            let start = i + 1;
            let mut k = start;
            while k < s.len() && s[k] != '"' {
                if s[k] == '\\' {
                    k += 1;
                }
                k += 1;
            }
            wipe(&mut out, start, k);
            i = k + 1;
        } else if c == '\'' {
            // A char literal (`'"'`, `'{'`, `'\''`), or a lifetime (`'a`), which has no closing quote.
            if next == Some('\\') {
                let mut k = i + 3;
                while k < s.len() && s[k] != '\'' {
                    k += 1;
                }
                wipe(&mut out, i + 1, k);
                i = k + 1;
            } else if s.get(i + 2) == Some(&'\'') {
                wipe(&mut out, i + 1, i + 2);
                i += 3;
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    out
}

fn is_ident(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Whether `code[at..]` starts with `word` as a whole identifier.
fn word_at(code: &[char], at: usize, word: &str) -> bool {
    let w: Vec<char> = word.chars().collect();
    code.get(at..at + w.len()) == Some(&w[..])
        && !(at > 0 && is_ident(code[at - 1]))
        && !code.get(at + w.len()).is_some_and(|c| is_ident(*c))
}

/// The index of the first non-whitespace character at or after `i`.
fn skip_ws(code: &[char], mut i: usize) -> usize {
    while code.get(i).is_some_and(|c| c.is_whitespace()) {
        i += 1;
    }
    i
}

/// The index of the `}` closing the `{` at `open`.
fn matching_brace(code: &[char], open: usize) -> usize {
    let mut depth = 0;
    for (i, &c) in code.iter().enumerate().skip(open) {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return i;
                }
            }
            _ => {}
        }
    }
    code.len().saturating_sub(1)
}

/// The end of the thing a `#[cfg(test)]` attribute applies to, starting at `from`. An item
/// (`fn`, `mod`, `use`, ...) ends past its `;` or its braced body. Anything else (a struct field, a
/// statement) ends at its `,` or `;`, at the close of the block holding it, or, for a braced
/// statement such as `if let .. { .. }`, at its closing brace.
fn item_end(code: &[char], from: usize) -> usize {
    const ITEMS: &[&str] = &[
        "pub",
        "fn",
        "async",
        "mod",
        "use",
        "impl",
        "struct",
        "enum",
        "trait",
        "type",
        "const",
        "static",
        "unsafe",
        "extern",
        "macro_rules",
    ];
    let is_item = ITEMS.iter().any(|k| word_at(code, from, k));
    let mut depth = 0i32;
    for (i, &c) in code.iter().enumerate().skip(from) {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            '{' if is_item && depth <= 0 => return matching_brace(code, i) + 1,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth < 0 {
                    return i;
                }
                if depth == 0 {
                    return i + 1;
                }
            }
            ';' if depth <= 0 => return i + 1,
            ',' if depth <= 0 && !is_item => return i + 1,
            _ => {}
        }
    }
    code.len()
}

/// The code with every `#[cfg(test)]` item blanked out (newlines kept).
fn without_tests(mut code: Vec<char>) -> Vec<char> {
    let mut i = 0;
    while i < code.len() {
        if code[i] == '#' && code.get(i + 1) == Some(&'[') {
            let Some(close) = code[i..].iter().position(|c| *c == ']').map(|p| p + i) else {
                break;
            };
            let attr: String = code[i + 2..close]
                .iter()
                .filter(|c| !c.is_whitespace())
                .collect();
            if attr == "cfg(test)" {
                // Any further attributes on the same item belong to it.
                let mut start = skip_ws(&code, close + 1);
                while code.get(start) == Some(&'#') {
                    let c = code[start..].iter().position(|c| *c == ']').unwrap() + start;
                    start = skip_ws(&code, c + 1);
                }
                let end = item_end(&code, start);
                for c in &mut code[i..end] {
                    if *c != '\n' {
                        *c = ' ';
                    }
                }
                i = end;
                continue;
            }
            i = close;
        }
        i += 1;
    }
    code
}

/// Each function with a body in `code` as (name, first line, last line), 1-based.
fn functions(code: &[char]) -> Vec<(String, usize, usize)> {
    let mut line_of = Vec::with_capacity(code.len());
    let mut line = 1;
    for &c in code {
        line_of.push(line);
        if c == '\n' {
            line += 1;
        }
    }
    let mut out = Vec::new();
    for i in 0..code.len() {
        if !word_at(code, i, "fn") {
            continue;
        }
        let name_at = skip_ws(code, i + 2);
        let name: String = code[name_at..]
            .iter()
            .take_while(|c| is_ident(**c))
            .collect();
        if name.is_empty() {
            continue; // `fn(` is a function pointer type.
        }
        // The body's `{`, unless a `;` (a trait method with no body) comes first.
        let mut paren = 0i32;
        let mut open = None;
        for (j, &c) in code.iter().enumerate().skip(name_at) {
            match c {
                '(' | '[' => paren += 1,
                ')' | ']' => paren -= 1,
                ';' if paren <= 0 => break,
                '{' if paren <= 0 => {
                    open = Some(j);
                    break;
                }
                _ => {}
            }
        }
        if let Some(open) = open {
            out.push((name, line_of[i], line_of[matching_brace(code, open)]));
        }
    }
    out
}

/// Whether the mode rule fires at the identifier `service` at `code[at..]`.
fn service_branch_at(code: &[char], at: usize) -> bool {
    if !word_at(code, at, "service") {
        return false;
    }
    let after = skip_ws(code, at + "service".len());
    let rest: String = code[after..].iter().take(10).collect();
    if rest.starts_with(".is_some()")
        || rest.starts_with(".is_none()")
        || rest.starts_with("==")
        || rest.starts_with("!=")
    {
        return true;
    }
    // Walk back over the path (`&self.supervisor.`) that `service` ends.
    let mut start = at;
    while start > 0 && (is_ident(code[start - 1]) || matches!(code[start - 1], '.' | '&')) {
        start -= 1;
    }
    // `match <path>.service` and `let Some(x) = <path>.service`: the field, not a bare local.
    if !code[start..at].contains(&'.') {
        return false;
    }
    let head: String = code[start.saturating_sub(60)..start].iter().collect();
    let head = head.split_whitespace().collect::<Vec<_>>().join(" ");
    head.ends_with("match") || (head.contains("let Some(") && head.ends_with('='))
}

/// How many times the mode rule fires in `code`.
fn mode_checks(code: &[char]) -> usize {
    (0..code.len())
        .filter(|&i| {
            (word_at(code, i, "Mode")
                && code.get(i + 4..i + 6) == Some(&[':', ':'])
                && ["Local", "Replica", "Service"]
                    .iter()
                    .any(|v| word_at(code, i + 6, v)))
                || word_at(code, i, "service_mode")
                || service_branch_at(code, i)
        })
        .count()
}

fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    entries.sort();
    for p in entries {
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        if p.is_dir() {
            if name != "tests" {
                sources(&p, out);
            }
        } else if name.ends_with(".rs")
            && name != "tests.rs"
            && !name.ends_with("_tests.rs")
            && name != "test_support.rs"
            && name != "tracing_test.rs"
        {
            out.push(p);
        }
    }
}

/// Every non-test source file as (path relative to `src`, code with comments, strings and test
/// items blanked).
fn agent_sources() -> Vec<(String, Vec<char>)> {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    sources(&src, &mut files);
    files
        .iter()
        .map(|f| {
            let rel = f.strip_prefix(&src).unwrap().to_string_lossy().into_owned();
            (
                rel,
                without_tests(blank(&std::fs::read_to_string(f).unwrap())),
            )
        })
        .collect()
}

#[test]
fn mode_is_decided_only_in_mode_rs() {
    let mut problems = Vec::new();
    let mut seen = Vec::new();
    for (rel, code) in agent_sources() {
        let n = mode_checks(&code);
        if rel == "mode.rs" || n == 0 {
            continue;
        }
        seen.push(rel.clone());
        let allowed = MODE_ALLOWED
            .iter()
            .find(|(p, _, _)| *p == rel)
            .map_or(0, |e| e.1);
        if n != allowed {
            problems.push(format!("  {rel}: {n} mode check(s), {allowed} allowed"));
        }
    }
    for (file, n, _) in MODE_ALLOWED {
        if !seen.iter().any(|s| s == file) {
            problems.push(format!(
                "  {file}: {n} allowed but none found; remove the entry"
            ));
        }
    }
    assert!(
        problems.is_empty(),
        "mode checks outside crates/agent/src/mode.rs:\n{}\n\nWhether the process is local or a \
         service replica is decided in `mode.rs` only. Add a method on `Mode` that names the rule \
         (for example `mode.reads_disk_overrides()`) and call it here, instead of branching on \
         `Mode::..`, `service.is_some()` or `service_mode`. If a site is not about the mode at \
         all, record it in MODE_ALLOWED in tests/structure_lint.rs with the reason.",
        problems.join("\n")
    );
}

#[test]
fn no_function_outgrows_the_budget() {
    let mut lengths: BTreeMap<String, usize> = BTreeMap::new();
    for (rel, code) in agent_sources() {
        for (name, first, last) in functions(&code) {
            let slot = lengths.entry(format!("{rel}::{name}")).or_insert(0);
            *slot = (*slot).max(last - first + 1);
        }
    }
    let mut problems = Vec::new();
    for (key, &len) in &lengths {
        match OVER_BUDGET.iter().find(|(k, _, _)| k == key).map(|e| e.1) {
            None if len > BUDGET => problems.push(format!("  {key}: {len} lines, budget {BUDGET}")),
            Some(max) if len > max => {
                problems.push(format!("  {key}: grew from {max} to {len} lines"));
            }
            Some(_) if len <= BUDGET => problems.push(format!(
                "  {key}: now {len} lines, within budget; remove it from OVER_BUDGET"
            )),
            _ => {}
        }
    }
    for (key, _, _) in OVER_BUDGET {
        if !lengths.contains_key(*key) {
            problems.push(format!(
                "  {key}: listed in OVER_BUDGET but not found; remove or rename the entry"
            ));
        }
    }
    assert!(
        problems.is_empty(),
        "function length budget ({BUDGET} lines):\n{}\n\nSplit the function into named steps \
         instead of raising a number. OVER_BUDGET in tests/structure_lint.rs is a ratchet: an \
         entry may be kept or removed, never raised.",
        problems.join("\n")
    );
}

#[cfg(test)]
mod unit {
    use super::*;

    fn checks(src: &str) -> usize {
        mode_checks(&without_tests(blank(src)))
    }

    #[test]
    fn flags_the_mode_branches() {
        assert_eq!(checks("fn f(s: &S) { if s.service.is_some() {} }"), 1);
        assert_eq!(checks("fn f(s: &S) { if s.service.is_none() {} }"), 1);
        assert_eq!(checks("let service_mode = true;"), 1);
        assert_eq!(checks("if let Some(svc) = &self.service { }"), 1);
        assert_eq!(checks("match &self.service { _ => {} }"), 1);
        assert_eq!(checks("match mode { Mode::Local => 1, _ => 2 }"), 1);
        assert_eq!(checks("matches!(m, Mode::Service(..))"), 1);
        assert_eq!(checks("if a.service == b.service {}"), 1);
    }

    #[test]
    fn ignores_what_is_not_a_mode_check() {
        assert_eq!(checks("let h = client.service().host;"), 0);
        assert_eq!(
            checks("// Mode::Local and service.is_some()\nlet a = 1;"),
            0
        );
        assert_eq!(checks("let u = \"openai.azure.com/service_mode\";"), 0);
        assert_eq!(checks("let reservice = 1; let service_url = 2;"), 0);
        assert_eq!(checks("#[cfg(test)]\nmod t { fn f() { Mode::Local; } }"), 0);
        assert_eq!(checks("let m = Mode::process(flag);"), 0);
    }

    #[test]
    fn counts_function_lines() {
        let code = without_tests(blank(
            "fn a() {\n    let s = \"}\"; // }\n    if x { y }\n}\n#[cfg(test)]\nmod t {\n    fn big() {\n\n\n    }\n}\nfn b() -> Vec<u8> where T: X {\n    c('{')\n}\ntrait T { fn m(&self); }\n",
        ));
        assert_eq!(
            functions(&code),
            vec![("a".to_string(), 1, 4), ("b".to_string(), 12, 14)]
        );
    }
}
