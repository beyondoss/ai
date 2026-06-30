//! System-prompt assembly: base prompt + project instructions + skills + environment.
//!
//! The model's effective instructions are built here rather than baked into a constant, so it sees
//! the project's own `AGENTS.md`/`CLAUDE.md` files, any discovered skills, and the current date/cwd —
//! the context a coding agent needs to behave like it belongs in *this* repo.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::skills::{self, Skill};

/// Options controlling how the system prompt is assembled.
pub struct PromptOptions<'a> {
    /// The base agent identity/instructions.
    pub base: &'a str,
    /// Extra text appended after the base (e.g. `--append-system-prompt`).
    pub append: Option<&'a str>,
    /// The working directory whose project-instruction files and skills to load.
    pub cwd: &'a Path,
    /// Whether to discover and inject `AGENTS.md`/`CLAUDE.md` project-instruction files.
    pub include_context_files: bool,
    /// Whether to discover and advertise skills.
    pub include_skills: bool,
}

/// Build the full system prompt for a session. Pulls in project instruction files (global +
/// cwd-to-root walk), advertises discovered skills, and appends the current date and working directory.
pub fn build_system_prompt(opts: &PromptOptions) -> String {
    let mut s = String::from(opts.base);
    if let Some(extra) = opts.append {
        s.push_str("\n\n");
        s.push_str(extra);
    }

    if opts.include_context_files {
        for (path, body) in load_context_files(opts.cwd) {
            s.push_str(&format!(
                "\n\n<project_instructions path=\"{path}\">\n{}\n</project_instructions>",
                body.trim()
            ));
        }
    }

    if opts.include_skills {
        let skills: Vec<Skill> = skills::discover(opts.cwd);
        if !skills.is_empty() {
            s.push_str("\n\n");
            s.push_str(&skills::format_available(&skills));
        }
    }

    s.push_str(&format!(
        "\n\nCurrent date: {}\nCurrent working directory: {}",
        today(),
        opts.cwd.display()
    ));
    s
}

/// Collect project-instruction files as `(path, body)`, nearest-last so the model reads the most
/// specific instructions (the cwd's) after the broader ones: first the global `~/.claude/CLAUDE.md`
/// (and `AGENTS.md`), then every `AGENTS.md`/`CLAUDE.md` from the filesystem root down to `cwd`.
pub fn load_context_files(cwd: &Path) -> Vec<(String, String)> {
    let mut out = Vec::new();

    // Global instruction files.
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        for name in ["CLAUDE.md", "AGENTS.md"] {
            push_if_present(&mut out, &home.join(".claude").join(name));
        }
    }

    // Walk cwd → root, collecting each ancestor's files; reverse so the deepest (cwd) lands last.
    let mut ancestors: Vec<&Path> = cwd.ancestors().collect();
    ancestors.reverse(); // root-most first
    for dir in ancestors {
        for name in ["AGENTS.md", "CLAUDE.md"] {
            push_if_present(&mut out, &dir.join(name));
        }
    }
    out
}

fn push_if_present(out: &mut Vec<(String, String)>, path: &Path) {
    if let Ok(body) = std::fs::read_to_string(path) {
        if !body.trim().is_empty() {
            out.push((path.display().to_string(), body));
        }
    }
}

/// Today's date as `YYYY-MM-DD` (UTC), without pulling in a date crate.
fn today() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (y, m, d) = civil_from_days((secs / 86_400) as i64);
    format!("{y:04}-{m:02}-{d:02}")
}

/// Convert a count of days since the Unix epoch to a `(year, month, day)` civil date. Howard
/// Hinnant's `civil_from_days` algorithm — exact, branch-light, no leap-year special cases.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn civil_date_matches_known_epochs() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(18_993), (2022, 1, 1));
        // 2020-02-29 (a leap day) is day 18321.
        assert_eq!(civil_from_days(18_321), (2020, 2, 29));
    }

    #[test]
    fn context_files_are_nearest_last() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let nested = root.join("a/b");
        fs::create_dir_all(&nested).unwrap();
        fs::write(root.join("CLAUDE.md"), "root rules").unwrap();
        fs::write(nested.join("AGENTS.md"), "nested rules").unwrap();

        let files = load_context_files(&nested);
        let joined: Vec<&str> = files.iter().map(|(_, b)| b.as_str()).collect();
        // Both present, root-most before nearest.
        let root_pos = joined
            .iter()
            .position(|b| b.contains("root rules"))
            .unwrap();
        let nested_pos = joined
            .iter()
            .position(|b| b.contains("nested rules"))
            .unwrap();
        assert!(
            root_pos < nested_pos,
            "nearest file must come last: {joined:?}"
        );
    }

    #[test]
    fn system_prompt_includes_project_instructions_and_env() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("CLAUDE.md"), "Be excellent.").unwrap();
        let prompt = build_system_prompt(&PromptOptions {
            base: "You are an agent.",
            append: Some("Stay terse."),
            cwd: tmp.path(),
            include_context_files: true,
            include_skills: false,
        });
        assert!(prompt.contains("You are an agent."));
        assert!(prompt.contains("Stay terse."));
        assert!(prompt.contains("<project_instructions"));
        assert!(prompt.contains("Be excellent."));
        assert!(prompt.contains("Current date:"));
        assert!(prompt.contains("Current working directory:"));
    }
}
