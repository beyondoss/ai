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
/// rather than scanning one level), plus any loose `*.md` file directly at the root (pi's other skill
/// shape, for something too small to need its own directory). Once a directory yields its `SKILL.md` we
/// stop descending into it — the manifest defines that skill's root, and anything nested is that
/// skill's own resources.
fn discover_in(root: &Path) -> Vec<Skill> {
    let mut out = Vec::new();
    walk(root, &mut out);
    for skill in loose_root_skills(root) {
        out.push(skill);
    }
    out
}

/// Walk `root` for `SKILL.md` manifests, gitignore-aware — reuses the same `ignore` crate `grep`/`find`
/// already depend on, rather than hand-rolling `.gitignore`/`.ignore` parsing, so a vendored or fixture
/// directory carrying its own ignore file doesn't leak a stray `SKILL.md` into the prompt.
/// `WalkBuilder`'s defaults already match the walk's prior hand-rolled semantics: hidden files/
/// directories are skipped and symlinked directories are not followed (so a cyclic symlink can't trap
/// the walk); `max_depth` reproduces the old sane-bound cutoff.
fn walk(root: &Path, out: &mut Vec<Skill>) {
    let mut candidates: Vec<PathBuf> = ignore::WalkBuilder::new(root)
        .max_depth(Some(MAX_DEPTH))
        // A skills root (`~/.claude/skills`, `<cwd>/.claude/skills`) is routinely *not* itself a git
        // repository (or is a subdirectory of one where that fact is incidental) — `.gitignore` files
        // placed within it should still be honored either way, not only when `require_git`'s default
        // finds an enclosing `.git`.
        .require_git(false)
        .build()
        .flatten() // missing/unreadable/inaccessible entries are the normal case, not an error
        .filter(|entry| {
            entry.file_name() == "SKILL.md" && entry.file_type().is_some_and(|t| t.is_file())
        })
        .map(ignore::DirEntry::into_path)
        .collect();

    // Shallowest first: a manifest nested inside an already-accepted skill's directory is that
    // skill's own resource, not a separate skill — the original recursive walk stopped descending the
    // instant it found one, which this reproduces by rejecting anything under an accepted skill root.
    candidates.sort_by_key(|p| p.components().count());
    let mut accepted_dirs: Vec<PathBuf> = Vec::new();
    for manifest in candidates {
        let Some(dir) = manifest.parent() else {
            continue;
        };
        if accepted_dirs.iter().any(|a| dir.starts_with(a)) {
            continue;
        }
        if let Some(skill) = parse_skill(&manifest) {
            out.push(skill);
        }
        accepted_dirs.push(dir.to_path_buf());
    }
}

/// Loose `*.md` files directly under `root` (not `SKILL.md`, which [`walk`] already handles, and not
/// nested in a subdirectory — only the root's immediate children) — pi's second skill shape, for one
/// small enough not to need its own directory and resources.
fn loose_root_skills(root: &Path) -> Vec<Skill> {
    let Ok(entries) = fs::read_dir(root) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        if path.file_name().and_then(|n| n.to_str()) == Some("SKILL.md") {
            continue; // a bare root-level SKILL.md is `walk`'s concern, not a loose skill file
        }
        if let Some(skill) = parse_skill(&path) {
            out.push(skill);
        }
    }
    out
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
    for issue in validate_skill_name(&name) {
        tracing::warn!(skill = %name, path = %manifest.display(), "{issue}");
    }
    Some(Skill {
        name,
        description,
        path: manifest.to_path_buf(),
        disable_model_invocation,
    })
}

/// Cap matching the reference agent's `MAX_NAME_LENGTH`.
const MAX_SKILL_NAME_LEN: usize = 64;

/// Non-fatal format checks on a skill's declared `name` (lowercase alphanumeric + hyphens, no
/// leading/trailing/consecutive hyphens, bounded length) — the same rules the reference agent enforces.
/// A skill failing these is still discovered and usable; violations are only `warn!`-logged (see
/// `parse_skill`), since nothing here reads or displays diagnostics for an operator to act on.
fn validate_skill_name(name: &str) -> Vec<String> {
    let mut issues = Vec::new();
    if name.len() > MAX_SKILL_NAME_LEN {
        issues.push(format!(
            "skill name exceeds {MAX_SKILL_NAME_LEN} characters ({})",
            name.len()
        ));
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
    {
        issues.push("skill name must be lowercase a-z, 0-9, and hyphens only".to_string());
    }
    if name.starts_with('-') || name.ends_with('-') {
        issues.push("skill name must not start or end with a hyphen".to_string());
    }
    if name.contains("--") {
        issues.push("skill name must not contain consecutive hyphens".to_string());
    }
    issues
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

/// If `message` is a `/skill:name ...` explicit invocation of a known skill, expand it into that
/// skill's full file contents (any trailing text after the name follows on its own paragraph);
/// otherwise return the message unchanged. This is the one path that honors a skill flagged
/// `disable-model-invocation` — that flag only keeps the model from *choosing* the skill on its own
/// (see [`format_available`]), not from a user naming it directly. Distinct prefix from
/// [`crate::prompts::expand_if_slash`]'s bare `/name`, so the two never collide; a caller should try
/// this first and fall through to prompt-template expansion when it returns the message unchanged.
pub fn expand_if_skill_invocation(message: &str, skills: &[Skill]) -> String {
    let Some(rest) = message.strip_prefix("/skill:") else {
        return message.to_string();
    };
    let (name, trailing) = match rest.split_once(char::is_whitespace) {
        Some((n, t)) => (n, t.trim()),
        None => (rest, ""),
    };
    let Some(skill) = find_by_name(skills, name) else {
        return message.to_string();
    };
    let Ok(body) = fs::read_to_string(&skill.path) else {
        return message.to_string();
    };
    if trailing.is_empty() {
        body
    } else {
        format!("{body}\n\n{trailing}")
    }
}

/// Render skills into the `<available_skills>` block injected into the system prompt. Tells the model
/// each skill's name, what it's for, and where to read the full instructions when a task matches.
/// Skills flagged `disable-model-invocation` are omitted here (the model must not auto-select them);
/// they stay reachable via [`find_by_name`] for an explicit `/skill:name` invocation.
pub fn format_available(skills: &[Skill]) -> String {
    let mut out = String::from(
        "<available_skills>\nThese skills extend your capabilities. When a task matches a skill's \
         description, read its file for the full instructions before proceeding. When a skill file \
         references a relative path, resolve it against the skill directory (the parent of SKILL.md, \
         or the loose skill file's own directory) and use that absolute path in tool commands.\n",
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
    fn validate_skill_name_flags_bad_shapes_but_accepts_good_ones() {
        assert!(validate_skill_name("lint").is_empty());
        assert!(validate_skill_name("pdf-processing-v2").is_empty());
        assert!(!validate_skill_name("Lint").is_empty()); // uppercase
        assert!(!validate_skill_name("lint tool").is_empty()); // space
        assert!(!validate_skill_name("-lint").is_empty()); // leading hyphen
        assert!(!validate_skill_name("lint-").is_empty()); // trailing hyphen
        assert!(!validate_skill_name("lint--tool").is_empty()); // consecutive hyphens
        assert!(!validate_skill_name(&"a".repeat(65)).is_empty()); // too long
    }

    #[test]
    fn a_badly_named_skill_is_still_discovered() {
        // Non-fatal: format issues are warn!-logged (see `parse_skill`), not a rejection — the skill
        // must still surface and be usable.
        let tmp = tempfile::tempdir().unwrap();
        write_skill(
            tmp.path(),
            "Bad_Name",
            "---\nname: Bad_Name\ndescription: still works\n---\n",
        );
        let skills = discover_in(tmp.path());
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "Bad_Name");
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
    fn gitignored_directories_are_not_walked() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join(".gitignore"), "vendor/\n").unwrap();
        // A stray SKILL.md under an ignored directory (e.g. a vendored dependency's fixtures) must
        // not surface — matching the reference agent's gitignore-aware discovery.
        write_skill(
            &tmp.path().join("vendor"),
            "leaked",
            "---\nname: leaked\ndescription: should not surface\n---\n",
        );
        write_skill(
            tmp.path(),
            "real",
            "---\nname: real\ndescription: a real skill\n---\n",
        );
        let names: Vec<String> = discover_in(tmp.path())
            .into_iter()
            .map(|s| s.name)
            .collect();
        assert_eq!(names, vec!["real".to_string()]);
    }

    #[test]
    fn loose_root_level_md_file_is_a_skill() {
        let tmp = tempfile::tempdir().unwrap();
        // A skill small enough not to need its own directory — a bare `*.md` file directly under the
        // skills root, distinct from the `dir/SKILL.md` shape `write_skill` produces.
        fs::write(
            tmp.path().join("quick.md"),
            "---\nname: quick\ndescription: A one-file skill\n---\nBody.",
        )
        .unwrap();
        let skills = discover_in(tmp.path());
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "quick");
        assert_eq!(skills[0].path, tmp.path().join("quick.md"));
    }

    #[test]
    fn loose_root_level_scan_ignores_skill_md_itself() {
        // A bare `SKILL.md` directly at the root is `walk`'s concern; the loose-`.md` scanner must not
        // double-count it as a second skill.
        let tmp = tempfile::tempdir().unwrap();
        fs::write(
            tmp.path().join("SKILL.md"),
            "---\nname: root-skill\ndescription: at the very root\n---\n",
        )
        .unwrap();
        let names: Vec<String> = discover_in(tmp.path())
            .into_iter()
            .map(|s| s.name)
            .collect();
        assert_eq!(names, vec!["root-skill".to_string()]);
    }

    #[test]
    fn expand_if_skill_invocation_reads_the_skill_file() {
        let tmp = tempfile::tempdir().unwrap();
        write_skill(
            tmp.path(),
            "lint",
            "---\nname: lint\ndescription: Run the linter\n---\n\nRun `cargo clippy`.",
        );
        let skills = discover_in(tmp.path());
        let expanded = expand_if_skill_invocation("/skill:lint", &skills);
        assert!(expanded.contains("Run `cargo clippy`."));
        assert!(!expanded.starts_with("/skill:"));
    }

    #[test]
    fn expand_if_skill_invocation_appends_trailing_text() {
        let tmp = tempfile::tempdir().unwrap();
        write_skill(
            tmp.path(),
            "lint",
            "---\nname: lint\ndescription: Run the linter\n---\n\nRun the linter.",
        );
        let skills = discover_in(tmp.path());
        let expanded = expand_if_skill_invocation("/skill:lint only src/main.rs", &skills);
        assert!(expanded.contains("Run the linter."));
        assert!(expanded.ends_with("only src/main.rs"));
    }

    #[test]
    fn expand_if_skill_invocation_bypasses_disable_model_invocation() {
        // The flag hides a skill from the model's own judgment (`format_available`), not from an
        // explicit `/skill:name` the user typed directly.
        let tmp = tempfile::tempdir().unwrap();
        write_skill(
            tmp.path(),
            "manual",
            "---\nname: manual\ndescription: Explicit only\ndisable-model-invocation: true\n---\n\nManual body.",
        );
        let skills = discover_in(tmp.path());
        let expanded = expand_if_skill_invocation("/skill:manual", &skills);
        assert!(expanded.contains("Manual body."));
    }

    #[test]
    fn expand_if_skill_invocation_passes_through_unmatched_input() {
        // Not a `/skill:` message at all.
        assert_eq!(
            expand_if_skill_invocation("plain message", &[]),
            "plain message"
        );
        // `/skill:` prefix but no such skill — unchanged, not an error.
        assert_eq!(
            expand_if_skill_invocation("/skill:nope", &[]),
            "/skill:nope"
        );
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
        // The model needs to be told how to resolve a relative path a skill file references, or it
        // may hand a tool a path relative to the wrong directory.
        assert!(rendered.contains("resolve it against the skill directory"));
    }
}
