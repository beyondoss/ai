//! Agent skills — progressive-disclosure capabilities discovered on disk.
//!
//! A skill is a directory containing a `SKILL.md` whose YAML frontmatter declares a `name` and
//! `description`. We discover them under `~/.claude/skills` (user) and `<cwd>/.claude/skills`
//! (project), and inject only their name/description/location into the system prompt — the body is
//! read on demand (by the `read` tool) when a task matches, so skills cost almost no context until used.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// A discovered skill: enough to advertise it; the body stays on disk until needed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skill {
    pub name: String,
    pub description: String,
    /// Absolute path to the skill's `SKILL.md`, given to the model so it can read it on demand.
    pub path: PathBuf,
    /// `disable-model-invocation: true` in the frontmatter: the model must not auto-select this skill
    /// from its description, but a user can still trigger it explicitly via `/skill:name`. Such skills
    /// are discovered (so the explicit lookup works) but omitted from the `<available_skills>` listing.
    pub disable_model_invocation: bool,
}

/// How deep we descend looking for a `SKILL.md`. A skill may live in a directory tree, but a sane
/// bound keeps a pathological/symlinked layout from turning discovery into an unbounded walk.
const MAX_DEPTH: usize = 8;

/// Discover skills under the user (`~/.claude/skills`) and project (`<cwd>/.claude/skills`) roots.
/// Project skills shadow user skills of the same name. Returns them sorted by name (stable output).
pub fn discover(cwd: &Path) -> Vec<Skill> {
    let mut found: Vec<Skill> = Vec::new();
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Some(home) = home_dir() {
        roots.push(home.join(".claude/skills"));
    }
    roots.push(cwd.join(".claude/skills"));

    for root in roots {
        for skill in discover_in(&root) {
            // Later roots (project) win over earlier (user) on name collisions.
            if let Some(existing) = found.iter_mut().find(|s| s.name == skill.name) {
                *existing = skill;
            } else {
                found.push(skill);
            }
        }
    }
    found.sort_by(|a, b| a.name.cmp(&b.name));
    found
}

/// Discover skills anywhere under `root`: walk the tree and load a `SKILL.md` at any depth (pi recurses
/// rather than scanning one level). Once a directory yields its `SKILL.md` we stop descending into it —
/// the manifest defines that skill's root, and anything nested is that skill's own resources.
fn discover_in(root: &Path) -> Vec<Skill> {
    let mut out = Vec::new();
    walk(root, 0, &mut out);
    out
}

/// Recursive worker for [`discover_in`]. Skips hidden directories and does not follow symlinked
/// directories, so a cyclic symlink (`a -> ..`) can't trap the walk.
fn walk(dir: &Path, depth: usize, out: &mut Vec<Skill>) {
    if depth > MAX_DEPTH {
        return;
    }
    // A manifest here defines this directory as a skill root; load it and don't descend further.
    let manifest = dir.join("SKILL.md");
    if manifest.is_file() {
        if let Some(skill) = parse_skill(&manifest) {
            out.push(skill);
        }
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return; // missing/unreadable dir is the normal case, not an error
    };
    for entry in entries.flatten() {
        // `file_type` does not traverse symlinks, so a symlinked directory reads as a symlink (not a
        // directory) and is skipped — that's what keeps cycles out of the walk.
        let Ok(ft) = entry.file_type() else { continue };
        if !ft.is_dir() {
            continue;
        }
        if entry.file_name().to_string_lossy().starts_with('.') {
            continue; // hidden directories (`.git`, `.cache`, …) never hold skills
        }
        walk(&entry.path(), depth + 1, out);
    }
}

/// Parse a `SKILL.md`'s frontmatter into a [`Skill`]. Requires a non-empty `description`; falls back to
/// the directory name for `name` if the frontmatter omits it.
fn parse_skill(manifest: &Path) -> Option<Skill> {
    let text = fs::read_to_string(manifest).ok()?;
    let fm = parse_frontmatter(&text);
    let description = fm
        .get("description")
        .filter(|d| !d.trim().is_empty())?
        .clone();
    let name = fm.get("name").cloned().or_else(|| {
        manifest
            .parent()?
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
    })?;
    let disable_model_invocation = fm
        .get("disable-model-invocation")
        .map(|v| v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    Some(Skill {
        name,
        description,
        path: manifest.to_path_buf(),
        disable_model_invocation,
    })
}

/// Parse a leading `---`-fenced YAML frontmatter block into its top-level scalar keys. Dependency-free
/// (no `serde_yaml`): enough of YAML for the Agent Skills spec — quoted values, and block scalars
/// (`key: |` / `key: >`) whose value spans the following more-indented lines. A `>` (folded) block is
/// joined with spaces, a `|` (literal) block with newlines; both let a long `description:` wrap across
/// lines. Anything fancier (anchors, nested maps) is out of scope and ignored.
fn parse_frontmatter(text: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mut lines = text.lines().peekable();
    if lines.next().map(str::trim) != Some("---") {
        return map;
    }
    while let Some(line) = lines.next() {
        if line.trim() == "---" {
            break;
        }
        // Top-level keys are unindented; an indented line is a continuation already consumed by the
        // block-scalar branch below, so skip any that reach here.
        if line.starts_with([' ', '\t']) {
            continue;
        }
        let Some((key, raw)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim().to_string();
        let raw = raw.trim();
        let value = match raw.chars().next() {
            // Block scalar: gather the lines indented under this key until a dedent or closing `---`.
            Some(folded @ ('|' | '>')) => {
                let mut parts: Vec<String> = Vec::new();
                while let Some(next) = lines.peek() {
                    if next.trim() == "---" {
                        break;
                    }
                    if !next.trim().is_empty() && !next.starts_with([' ', '\t']) {
                        break; // dedent to another top-level key ends the block
                    }
                    parts.push(next.trim().to_string());
                    lines.next();
                }
                let joined = if folded == '>' {
                    parts.join(" ")
                } else {
                    parts.join("\n")
                };
                joined.trim().to_string()
            }
            _ => unquote(raw),
        };
        map.insert(key, value);
    }
    map
}

/// Strip matching surrounding single or double quotes.
fn unquote(s: &str) -> String {
    let bytes = s.as_bytes();
    if bytes.len() >= 2
        && (bytes[0] == b'"' || bytes[0] == b'\'')
        && bytes[bytes.len() - 1] == bytes[0]
    {
        s[1..s.len() - 1].to_string()
    } else {
        s.to_string()
    }
}

/// Look up a discovered skill by exact name — used to resolve an explicit `/skill:name` invocation,
/// which is allowed even for skills flagged `disable-model-invocation`.
pub fn find_by_name<'a>(skills: &'a [Skill], name: &str) -> Option<&'a Skill> {
    skills.iter().find(|s| s.name == name)
}

/// Render skills into the `<available_skills>` block injected into the system prompt. Tells the model
/// each skill's name, what it's for, and where to read the full instructions when a task matches.
/// Skills flagged `disable-model-invocation` are omitted here (the model must not auto-select them);
/// they stay reachable via [`find_by_name`] for an explicit `/skill:name` invocation.
pub fn format_available(skills: &[Skill]) -> String {
    let mut out = String::from(
        "<available_skills>\nThese skills extend your capabilities. When a task matches a skill's \
         description, read its file for the full instructions before proceeding.\n",
    );
    for s in skills.iter().filter(|s| !s.disable_model_invocation) {
        out.push_str(&format!(
            "- {} — {} (read: {})\n",
            s.name,
            s.description,
            s.path.display()
        ));
    }
    out.push_str("</available_skills>");
    out
}

/// The user's home directory from the environment.
fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests target `discover_in` (a single root) so they don't pick up the developer's real
    // `~/.claude/skills` that `discover` also scans.
    fn write_skill(root: &Path, name: &str, frontmatter: &str) {
        let sd = root.join(name);
        fs::create_dir_all(&sd).unwrap();
        fs::write(sd.join("SKILL.md"), frontmatter).unwrap();
    }

    #[test]
    fn discovers_project_skills_with_frontmatter() {
        let tmp = tempfile::tempdir().unwrap();
        write_skill(
            tmp.path(),
            "lint",
            "---\nname: lint\ndescription: \"Run the project linter\"\n---\n\nBody here.",
        );
        let skills = discover_in(tmp.path());
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "lint");
        assert_eq!(skills[0].description, "Run the project linter");
        assert!(!skills[0].disable_model_invocation);
    }

    #[test]
    fn name_falls_back_to_directory() {
        let tmp = tempfile::tempdir().unwrap();
        write_skill(tmp.path(), "deploy", "---\ndescription: Ship it\n---\n");
        let skills = discover_in(tmp.path());
        assert_eq!(skills[0].name, "deploy");
    }

    #[test]
    fn skill_without_description_is_skipped() {
        let tmp = tempfile::tempdir().unwrap();
        write_skill(tmp.path(), "broken", "---\nname: broken\n---\n");
        assert!(discover_in(tmp.path()).is_empty());
    }

    #[test]
    fn discovers_skills_nested_at_any_depth() {
        let tmp = tempfile::tempdir().unwrap();
        // A `SKILL.md` several directories deep, under a category folder, is still found.
        write_skill(
            &tmp.path().join("category/sub"),
            "deep",
            "---\nname: deep\ndescription: A nested skill\n---\nBody.",
        );
        // A manifest nested *inside* a discovered skill is that skill's resource, not a new skill.
        write_skill(
            &tmp.path().join("category/sub/deep/inner"),
            "inner",
            "---\nname: inner\ndescription: should not surface\n---\n",
        );
        let names: Vec<String> = discover_in(tmp.path())
            .into_iter()
            .map(|s| s.name)
            .collect();
        assert_eq!(names, vec!["deep".to_string()]);
    }

    #[test]
    fn block_scalar_description_spans_lines() {
        let tmp = tempfile::tempdir().unwrap();
        write_skill(
            tmp.path(),
            "wrapped",
            "---\nname: wrapped\ndescription: >\n  first line\n  second line\n---\nBody.",
        );
        let skills = discover_in(tmp.path());
        assert_eq!(skills[0].description, "first line second line");
    }

    #[test]
    fn disable_model_invocation_is_parsed_hidden_but_findable() {
        let tmp = tempfile::tempdir().unwrap();
        write_skill(
            tmp.path(),
            "manual",
            "---\nname: manual\ndescription: Explicit only\ndisable-model-invocation: true\n---\n",
        );
        let skills = discover_in(tmp.path());
        assert!(skills[0].disable_model_invocation);
        // Discoverable for an explicit `/skill:name` lookup …
        assert!(find_by_name(&skills, "manual").is_some());
        // … but omitted from the model-facing listing.
        assert!(!format_available(&skills).contains("manual"));
    }

    #[test]
    fn format_lists_name_description_and_path() {
        let skills = vec![Skill {
            name: "lint".into(),
            description: "Run the linter".into(),
            path: PathBuf::from("/x/.claude/skills/lint/SKILL.md"),
            disable_model_invocation: false,
        }];
        let rendered = format_available(&skills);
        assert!(rendered.contains("<available_skills>"));
        assert!(rendered.contains("lint — Run the linter"));
        assert!(rendered.contains("/x/.claude/skills/lint/SKILL.md"));
    }
}
