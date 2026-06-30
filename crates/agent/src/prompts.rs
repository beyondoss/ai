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
            let (hint, body) = parse(&text);
            let template = PromptTemplate {
                name: name.clone(),
                argument_hint: hint,
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

/// Split optional `---` frontmatter (reading `argument-hint:`) from the body.
fn parse(text: &str) -> (Option<String>, String) {
    let mut lines = text.lines();
    if lines.next().map(str::trim) != Some("---") {
        return (None, text.to_string());
    }
    let mut hint = None;
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
            }
        } else {
            rest.push_str(line);
            rest.push('\n');
        }
    }
    (hint, rest.trim_end().to_string())
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
        Some(t) => substitute(&t.body, args),
        None => message.to_string(),
    }
}

/// Bash-style argument substitution: `$ARGUMENTS`/`$@` → all args; `$1`..`$9` → positional args
/// (whitespace-split); an out-of-range positional expands to empty.
fn substitute(body: &str, args: &str) -> String {
    let positional: Vec<&str> = args.split_whitespace().collect();
    let mut out = body.replace("$ARGUMENTS", args).replace("$@", args);
    for (i, val) in positional.iter().enumerate() {
        out = out.replace(&format!("${}", i + 1), val);
    }
    // Any remaining `$1`..`$9` past the supplied args expand to empty.
    for i in (positional.len() + 1)..=9 {
        out = out.replace(&format!("${i}"), "");
    }
    out
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
        let t = vec![PromptTemplate {
            name: "x".into(),
            argument_hint: None,
            body: "a $1 b $2 c".into(),
        }];
        assert_eq!(expand_if_slash("/x only", &t), "a only b  c");
    }
}
