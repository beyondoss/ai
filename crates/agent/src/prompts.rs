//! Prompt templates — reusable `/name args` expansions.
//!
//! A template is a Markdown file under `~/.claude/prompts` (user) or `<cwd>/.claude/prompts`
//! (project). When a prompt message begins with `/name ...`, the matching template's body is expanded
//! with bash-style argument substitution and sent to the model in place of the slash line.

use std::fs;
use std::path::{Path, PathBuf};

/// A discovered prompt template.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptTemplate {
    /// Invoke name (the file stem), used as `/name`.
    pub name: String,
    /// One-line hint for arguments, from `argument-hint:` frontmatter (if any).
    pub argument_hint: Option<String>,
    /// One-line summary for autocomplete: `description:` frontmatter, else the first non-empty body
    /// line (truncated). Always populated so a caller can show *something* per command.
    pub description: String,
    /// The template body (frontmatter stripped).
    pub body: String,
}

/// Discover prompt templates under the user and project roots; project shadows user by name.
pub fn discover(cwd: &Path) -> Vec<PromptTemplate> {
    let mut found: Vec<PromptTemplate> = Vec::new();
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        roots.push(home.join(".claude/prompts"));
    }
    roots.push(cwd.join(".claude/prompts"));

    for root in roots {
        let Ok(entries) = fs::read_dir(&root) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("md") {
                continue;
            }
            let Some(name) = path.file_stem().map(|s| s.to_string_lossy().into_owned()) else {
                continue;
            };
            let Ok(text) = fs::read_to_string(&path) else {
                continue;
            };
            let (hint, description, body) = parse(&text);
            let template = PromptTemplate {
                name: name.clone(),
                argument_hint: hint,
                description,
                body,
            };
            if let Some(existing) = found.iter_mut().find(|t| t.name == name) {
                *existing = template;
            } else {
                found.push(template);
            }
        }
    }
    found.sort_by(|a, b| a.name.cmp(&b.name));
    found
}

/// Split optional `---` frontmatter (reading `argument-hint:` and `description:`) from the body, and
/// derive a `description` (frontmatter wins; otherwise the first non-empty body line, truncated like
/// pi to 60 chars). Returns `(argument_hint, description, body)`.
fn parse(text: &str) -> (Option<String>, String, String) {
    let mut lines = text.lines();
    let has_frontmatter = lines.next().map(str::trim) == Some("---");

    let mut hint = None;
    let mut description = None;
    let body = if has_frontmatter {
        let mut rest = String::new();
        let mut in_frontmatter = true;
        for line in lines {
            if in_frontmatter {
                if line.trim() == "---" {
                    in_frontmatter = false;
                    continue;
                }
                if let Some(v) = line.trim().strip_prefix("argument-hint:") {
                    hint = Some(v.trim().trim_matches(['"', '\'']).to_string());
                } else if let Some(v) = line.trim().strip_prefix("description:") {
                    description = Some(v.trim().trim_matches(['"', '\'']).to_string());
                }
            } else {
                rest.push_str(line);
                rest.push('\n');
            }
        }
        rest.trim_end().to_string()
    } else {
        text.trim_end().to_string()
    };

    let description = description
        .filter(|d| !d.is_empty())
        .unwrap_or_else(|| first_line_summary(&body));
    (hint, description, body)
}

/// A short summary from the first non-empty body line — pi truncates to 60 chars and appends `...`.
fn first_line_summary(body: &str) -> String {
    let Some(line) = body.lines().find(|l| !l.trim().is_empty()) else {
        return String::new();
    };
    // Count by chars so a multi-byte boundary never splits mid-codepoint.
    if line.chars().count() > 60 {
        let truncated: String = line.chars().take(60).collect();
        format!("{truncated}...")
    } else {
        line.to_string()
    }
}

/// If `message` is a `/name ...` invocation of a known template, expand and return it; otherwise
/// return the message unchanged.
pub fn expand_if_slash(message: &str, templates: &[PromptTemplate]) -> String {
    let Some(rest) = message.strip_prefix('/') else {
        return message.to_string();
    };
    let (name, args) = match rest.split_once(char::is_whitespace) {
        Some((n, a)) => (n, a.trim()),
        None => (rest, ""),
    };
    match templates.iter().find(|t| t.name == name) {
        Some(t) => substitute(&t.body, &parse_command_args(args)),
        None => message.to_string(),
    }
}

/// Split an argument string into fields, honoring single and double quotes so `"a b" c` is two args
/// (`a b`, `c`) rather than three. Mirrors pi's `parseCommandArgs`: a quote starts a span that runs to
/// the matching quote, and the quote characters themselves are dropped.
pub fn parse_command_args(input: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    let mut started = false; // tracks an in-progress (possibly empty, e.g. `""`) field

    for ch in input.chars() {
        match quote {
            Some(q) => {
                if ch == q {
                    quote = None;
                } else {
                    current.push(ch);
                }
            }
            None => match ch {
                '"' | '\'' => {
                    quote = Some(ch);
                    started = true;
                }
                ' ' | '\t' => {
                    if started {
                        args.push(std::mem::take(&mut current));
                        started = false;
                    }
                }
                _ => {
                    current.push(ch);
                    started = true;
                }
            },
        }
    }
    if started {
        args.push(current);
    }
    args
}

/// Bash-style argument substitution over a template body. Supports:
/// - `$ARGUMENTS` / `$@` → all args joined by a space.
/// - `$N` (any positive N, not just 1–9) → the Nth positional (1-based); out of range → empty.
/// - `${@:N}` / `${@:N:L}` → bash array slice: args from index N, optionally L of them, space-joined.
/// - `${N:-default}` → the Nth positional if present and non-empty, else `default`.
///
/// Done in a single left-to-right scan so a substituted value is never itself re-scanned (a plain
/// `replace` pass could rewrite a `$1` that came *out* of an argument).
fn substitute(body: &str, args: &[String]) -> String {
    let all = args.join(" ");
    let mut out = String::with_capacity(body.len());
    let bytes = body.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'$' {
            // Copy the current UTF-8 char verbatim (indices stay on char boundaries because we only
            // ever advance past whole `$...` tokens or single ASCII bytes that are < 0x80).
            let ch_len = utf8_len(bytes[i]);
            out.push_str(&body[i..i + ch_len]);
            i += ch_len;
            continue;
        }
        // We're at `$`. Try each form in turn; fall back to a literal `$` if none match.
        let after = &body[i + 1..];
        if let Some(stripped) = after.strip_prefix("ARGUMENTS") {
            out.push_str(&all);
            i = body.len() - stripped.len();
        } else if after.starts_with('@') {
            out.push_str(&all);
            i += 2; // `$@`
        } else if after.starts_with('{') {
            if let Some(end) = after.find('}') {
                let inner = &after[1..end];
                out.push_str(&expand_brace(inner, args, &all));
                i += 2 + end; // `$` + `{` … `}`
            } else {
                out.push('$');
                i += 1;
            }
        } else if let Some(digits) = leading_digits(after) {
            out.push_str(positional(args, digits));
            i += 1 + digits.len();
        } else {
            out.push('$');
            i += 1;
        }
    }
    out
}

/// Expand the contents of a `${...}` placeholder. `inner` is the text between the braces.
fn expand_brace(inner: &str, args: &[String], all: &str) -> String {
    if let Some(spec) = inner.strip_prefix("@:") {
        // `@:N` or `@:N:L` — a bash array slice.
        let (start_str, len_str) = match spec.split_once(':') {
            Some((s, l)) => (s, Some(l)),
            None => (spec, None),
        };
        let Some(start) = parse_index(start_str) else {
            return String::new();
        };
        let slice: &[String] = match len_str.and_then(parse_usize) {
            Some(len) => args
                .get(start..(start + len).min(args.len()))
                .unwrap_or(&[]),
            None => args.get(start..).unwrap_or(&[]),
        };
        return slice.join(" ");
    }
    if let Some((num, default)) = inner.split_once(":-") {
        // `${N:-default}` — default when the positional is missing or empty.
        if let Some(idx) = parse_index(num) {
            if let Some(v) = args.get(idx) {
                if !v.is_empty() {
                    return v.clone();
                }
            }
        }
        return default.to_string();
    }
    if inner == "@" || inner == "ARGUMENTS" {
        return all.to_string();
    }
    if inner.chars().all(|c| c.is_ascii_digit()) && !inner.is_empty() {
        return positional(args, inner).to_string();
    }
    String::new()
}

/// The 1-based positional `digits` as a `&str` slice into `args`, or `""` if missing/unparsable.
fn positional<'a>(args: &'a [String], digits: &str) -> &'a str {
    match parse_index(digits).and_then(|i| args.get(i)) {
        Some(v) => v.as_str(),
        None => "",
    }
}

/// Parse a 1-based positional number into a 0-based index (`"1"` → `0`). `"0"`/empty/overflow → None.
fn parse_index(s: &str) -> Option<usize> {
    let n: usize = s.parse().ok()?;
    n.checked_sub(1)
}

fn parse_usize(s: &str) -> Option<usize> {
    s.parse().ok()
}

/// The run of leading ASCII digits in `s`, or `None` if it doesn't start with a digit.
fn leading_digits(s: &str) -> Option<&str> {
    let end = s.bytes().take_while(u8::is_ascii_digit).count();
    (end > 0).then(|| &s[..end])
}

/// Length in bytes of the UTF-8 sequence whose lead byte is `b`.
fn utf8_len(b: u8) -> usize {
    match b {
        0x00..=0x7f => 1,
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        _ => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovers_and_expands_template() {
        let tmp = tempfile::tempdir().unwrap();
        let pdir = tmp.path().join(".claude/prompts");
        fs::create_dir_all(&pdir).unwrap();
        fs::write(
            pdir.join("fix.md"),
            "---\nargument-hint: <file>\n---\nFix the bug in $1 and explain: $ARGUMENTS",
        )
        .unwrap();

        // `discover` also scans the developer's real `~/.claude/prompts`; assert on our template by
        // name rather than the total count.
        let templates = discover(tmp.path());
        let fix = templates
            .iter()
            .find(|t| t.name == "fix")
            .expect("fix template");
        assert_eq!(fix.argument_hint.as_deref(), Some("<file>"));

        let expanded = expand_if_slash("/fix foo.rs urgently", &templates);
        assert_eq!(
            expanded,
            "Fix the bug in foo.rs and explain: foo.rs urgently"
        );
    }

    #[test]
    fn unknown_slash_is_passed_through() {
        assert_eq!(expand_if_slash("/nope x", &[]), "/nope x");
        assert_eq!(expand_if_slash("plain message", &[]), "plain message");
    }

    #[test]
    fn missing_positional_expands_empty() {
        let t = template("a $1 b $2 c");
        assert_eq!(expand_if_slash("/x only", &[t]), "a only b  c");
    }

    #[test]
    fn quoted_arguments_stay_together() {
        let args = parse_command_args(r#"foo "bar baz" 'qux quux' end"#);
        assert_eq!(args, vec!["foo", "bar baz", "qux quux", "end"]);
    }

    #[test]
    fn quote_aware_positional_substitution() {
        let t = template("first=[$1] second=[$2]");
        let expanded = expand_if_slash(r#"/x "a b" c"#, &[t]);
        assert_eq!(expanded, "first=[a b] second=[c]");
    }

    #[test]
    fn positional_beyond_nine() {
        let t = template("$10 $11");
        let expanded = expand_if_slash("/x 1 2 3 4 5 6 7 8 9 ten eleven", &[t]);
        assert_eq!(expanded, "ten eleven");
    }

    #[test]
    fn array_slice_substitution() {
        let t = template("rest=[${@:2}] two=[${@:2:2}]");
        let expanded = expand_if_slash("/x a b c d e", &[t]);
        assert_eq!(expanded, "rest=[b c d e] two=[b c]");
    }

    #[test]
    fn default_value_substitution() {
        let t = template("name=${2:-anon}");
        assert_eq!(
            expand_if_slash("/x given", std::slice::from_ref(&t)),
            "name=anon"
        );
        assert_eq!(expand_if_slash("/x given second", &[t]), "name=second");
    }

    #[test]
    fn description_from_frontmatter_then_first_line() {
        assert_eq!(
            parse("---\ndescription: Do the thing\n---\nBody line").1,
            "Do the thing"
        );
        // No `description:` → first non-empty body line.
        assert_eq!(
            parse("---\n---\n\nFirst real line\nmore").1,
            "First real line"
        );
        // No frontmatter at all → first non-empty line of the whole text.
        assert_eq!(parse("Just a body").1, "Just a body");
    }

    /// A template with an empty hint/description and the given body, for substitution tests.
    fn template(body: &str) -> PromptTemplate {
        PromptTemplate {
            name: "x".into(),
            argument_hint: None,
            description: String::new(),
            body: body.into(),
        }
    }
}
