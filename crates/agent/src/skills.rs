//! Agent skills — progressive-disclosure capabilities discovered on disk.
//!
//! A skill is a directory containing a `SKILL.md` whose YAML frontmatter declares a `name` and
//! `description`. We discover them under `~/.claude/skills` (user) and `<cwd>/.claude/skills`
//! (project), and inject only their name/description/location into the system prompt — the body is
//! read on demand (by the `read` tool) when a task matches, so skills cost almost no context until used.

use std::fs;
use std::path::{Path, PathBuf};

/// A discovered skill: enough to advertise it; the body stays on disk until needed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skill {
    pub name: String,
    pub description: String,
    /// Absolute path to the skill's `SKILL.md`, given to the model so it can read it on demand.
    pub path: PathBuf,
}

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

/// Discover skills directly under `root`: each immediate subdirectory holding a `SKILL.md`.
fn discover_in(root: &Path) -> Vec<Skill> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir(root) else {
        return out; // missing root is the normal case, not an error
    };
    for entry in entries.flatten() {
        let dir = entry.path();
        if !dir.is_dir() {
            continue;
        }
        let manifest = dir.join("SKILL.md");
        if !manifest.is_file() {
            continue;
        }
        if let Some(skill) = parse_skill(&manifest) {
            out.push(skill);
        }
    }
    out
}

/// Parse a `SKILL.md`'s frontmatter into a [`Skill`]. Requires both `name` and `description`; falls
/// back to the directory name for `name` if the frontmatter omits it.
fn parse_skill(manifest: &Path) -> Option<Skill> {
    let text = fs::read_to_string(manifest).ok()?;
    let (name, description) = parse_frontmatter(&text);
    let description = description?;
    let name = name.or_else(|| {
        manifest
            .parent()?
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
    })?;
    Some(Skill {
        name,
        description,
        path: manifest.to_path_buf(),
    })
}

/// Extract `name` and `description` from a leading `---`-fenced YAML frontmatter block. A minimal
/// single-line-value parser — enough for the Agent Skills spec without pulling in a YAML dependency.
fn parse_frontmatter(text: &str) -> (Option<String>, Option<String>) {
    let mut lines = text.lines();
    if lines.next().map(str::trim) != Some("---") {
        return (None, None);
    }
    let (mut name, mut description) = (None, None);
    for line in lines {
        let line = line.trim();
        if line == "---" {
            break;
        }
        if let Some(v) = line.strip_prefix("name:") {
            name = Some(unquote(v.trim()));
        } else if let Some(v) = line.strip_prefix("description:") {
            description = Some(unquote(v.trim()));
        }
    }
    (name, description)
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

/// Render skills into the `<available_skills>` block injected into the system prompt. Tells the model
/// each skill's name, what it's for, and where to read the full instructions when a task matches.
pub fn format_available(skills: &[Skill]) -> String {
    let mut out = String::from(
        "<available_skills>\nThese skills extend your capabilities. When a task matches a skill's \
         description, read its file for the full instructions before proceeding.\n",
    );
    for s in skills {
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
    fn format_lists_name_description_and_path() {
        let skills = vec![Skill {
            name: "lint".into(),
            description: "Run the linter".into(),
            path: PathBuf::from("/x/.claude/skills/lint/SKILL.md"),
        }];
        let rendered = format_available(&skills);
        assert!(rendered.contains("<available_skills>"));
        assert!(rendered.contains("lint — Run the linter"));
        assert!(rendered.contains("/x/.claude/skills/lint/SKILL.md"));
    }
}
