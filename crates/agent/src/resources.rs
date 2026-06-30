//! System-prompt assembly: base prompt + project instructions + skills + environment.
//!
//! The model's effective instructions are built here rather than baked into a constant, so it sees
//! the project's own `AGENTS.md`/`CLAUDE.md` files, any discovered skills, and the current date/cwd —
//! the context a coding agent needs to behave like it belongs in *this* repo.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::skills::{self, Skill};

/// Options controlling how the system prompt is assembled.
pub struct PromptOptions<'a> {
    /// The base agent identity/instructions. A `SYSTEM.md` on disk (project, then user) overrides it.
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
    // An on-disk `SYSTEM.md` (project `<cwd>/.claude/`, else user `~/.claude/`) replaces the built-in
    // base entirely — that's how a project pins its own agent identity (pi's resource-loader does the
    // same). Absent one, the caller-supplied `base` stands.
    let mut s = system_prompt_override(opts.cwd).unwrap_or_else(|| opts.base.to_string());
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

/// A `SYSTEM.md` override: project-local (`<cwd>/.claude/SYSTEM.md`) takes precedence over the user
/// one (`~/.claude/SYSTEM.md`). Returns its raw contents, or `None` when neither exists / is blank.
fn system_prompt_override(cwd: &Path) -> Option<String> {
    let mut candidates = vec![cwd.join(".claude/SYSTEM.md")];
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        candidates.push(home.join(".claude/SYSTEM.md"));
    }
    for path in candidates {
        if let Ok(body) = fs::read_to_string(&path) {
            if !body.trim().is_empty() {
                return Some(body);
            }
        }
    }
    None
}

/// Collect project-instruction files as `(path, body)`, nearest-last so the model reads the most
/// specific instructions (the cwd's) after the broader ones: first the global `~/.claude` file, then
/// one file per directory from the filesystem root down to `cwd`.
///
/// At most one file is taken per directory: when both `AGENTS.md` and `CLAUDE.md` are present,
/// `AGENTS.md` wins (matching pi's resource-loader). Filename matching is case-insensitive.
pub fn load_context_files(cwd: &Path) -> Vec<(String, String)> {
    let mut out = Vec::new();

    // Global instruction file (one, AGENTS-wins).
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        if let Some(file) = load_context_file_from_dir(&home.join(".claude")) {
            out.push(file);
        }
    }

    // Walk cwd → root, collecting each ancestor's file; reverse so the deepest (cwd) lands last.
    let mut ancestors: Vec<&Path> = cwd.ancestors().collect();
    ancestors.reverse(); // root-most first
    for dir in ancestors {
        if let Some(file) = load_context_file_from_dir(dir) {
            out.push(file);
        }
    }
    out
}

/// Pick this directory's single context file: `AGENTS.md` if present, else `CLAUDE.md`, matched
/// case-insensitively. Returns `(path, body)` only when the chosen file exists and is non-empty —
/// preserving the empty-file skip. A directory we can't read just yields nothing.
fn load_context_file_from_dir(dir: &Path) -> Option<(String, String)> {
    let mut agents: Option<PathBuf> = None;
    let mut claude: Option<PathBuf> = None;
    for entry in fs::read_dir(dir).ok()?.flatten() {
        let lower = entry.file_name().to_string_lossy().to_ascii_lowercase();
        if lower == "agents.md" && agents.is_none() {
            agents = Some(entry.path());
        } else if lower == "claude.md" && claude.is_none() {
            claude = Some(entry.path());
        }
    }
    let chosen = agents.or(claude)?;
    let body = fs::read_to_string(&chosen).ok()?;
    if body.trim().is_empty() {
        return None;
    }
    Some((chosen.display().to_string(), body))
}

/// Today's date as `YYYY-MM-DD` in the host's local timezone, without pulling in a date crate. Local
/// (not UTC) so the injected date never reads a day behind/ahead of the user near midnight.
fn today() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let local = secs + local_utc_offset(secs);
    let (y, m, d) = civil_from_days(local.div_euclid(86_400));
    format!("{y:04}-{m:02}-{d:02}")
}

/// The host's UTC offset in seconds at `now`, read from the system zoneinfo (`/etc/localtime`). We have
/// no date crate and `unsafe_code` is forbidden (so no libc `localtime`); parsing the TZif file is the
/// dependency-free, safe way to get the local offset. Any problem — missing file, unknown format —
/// degrades to `0` (UTC), which is correct, just not local.
fn local_utc_offset(now: i64) -> i64 {
    fs::read("/etc/localtime")
        .ok()
        .and_then(|data| tzif_offset(&data, now))
        .unwrap_or(0)
}

/// Extract the UTC offset applicable at `now` from a TZif (RFC 8536) blob. A v2/v3 file carries a
/// second block with 64-bit transition times after the legacy 32-bit one; that block is authoritative.
fn tzif_offset(data: &[u8], now: i64) -> Option<i64> {
    if data.get(0..4)? != b"TZif" {
        return None;
    }
    let version = *data.get(4)?;
    let (offset32, consumed) = tzif_block(data, 0, 4, now)?;
    if version == b'2' || version == b'3' {
        let (offset64, _) = tzif_block(data, consumed, 8, now)?;
        Some(offset64)
    } else {
        Some(offset32)
    }
}

/// Parse one TZif header+data block starting at `start`, with transition times `time_size` bytes wide
/// (4 for the v1 block, 8 for the v2/v3 block). Returns `(offset_seconds_at_now, end_index)`.
fn tzif_block(data: &[u8], start: usize, time_size: usize, now: i64) -> Option<(i64, usize)> {
    let header = data.get(start..start + 44)?;
    if header.get(0..4)? != b"TZif" {
        return None;
    }
    // Six 4-byte counts begin at header offset 20 (after magic + version + 15 reserved bytes).
    let isutcnt = be32(header, 20)? as usize;
    let isstdcnt = be32(header, 24)? as usize;
    let leapcnt = be32(header, 28)? as usize;
    let timecnt = be32(header, 32)? as usize;
    let typecnt = be32(header, 36)? as usize;
    let charcnt = be32(header, 40)? as usize;

    let trans = start + 44;
    let type_idx = trans + timecnt * time_size;
    let ttinfo = type_idx + timecnt;
    let names = ttinfo + typecnt * 6;
    let leaps = names + charcnt;
    let end = leaps + leapcnt * (time_size + 4) + isstdcnt + isutcnt;
    if end > data.len() {
        return None;
    }

    // The active type is that of the latest transition at or before `now` (transitions are ascending).
    let mut active: Option<usize> = None;
    for k in 0..timecnt {
        if read_int(data, trans + k * time_size, time_size)? <= now {
            active = Some(*data.get(type_idx + k)? as usize);
        } else {
            break;
        }
    }
    // Before the first transition, fall back to the first non-DST type, else type 0.
    let ti = active.unwrap_or_else(|| {
        (0..typecnt)
            .find(|&t| data.get(ttinfo + t * 6 + 4) == Some(&0))
            .unwrap_or(0)
    });
    let ti = if ti < typecnt { ti } else { 0 };
    // A ttinfo record is 6 bytes: i32 gmtoff, u8 isdst, u8 abbrind.
    let gmtoff = read_int(data, ttinfo + ti * 6, 4)?;
    Some((gmtoff, end))
}

/// Read a big-endian signed integer of `size` (4 or 8) bytes at offset `o`.
fn read_int(b: &[u8], o: usize, size: usize) -> Option<i64> {
    let s = b.get(o..o + size)?;
    if size == 8 {
        Some(i64::from_be_bytes(s.try_into().ok()?))
    } else {
        Some(i32::from_be_bytes(s.try_into().ok()?) as i64)
    }
}

/// Read a big-endian `u32` at offset `o`.
fn be32(b: &[u8], o: usize) -> Option<u32> {
    let s = b.get(o..o + 4)?;
    Some(u32::from_be_bytes(s.try_into().ok()?))
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
    fn one_file_per_dir_prefers_agents() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("proj");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("AGENTS.md"), "agents-wins").unwrap();
        fs::write(dir.join("CLAUDE.md"), "claude-loses").unwrap();

        let joined: String = load_context_files(&dir)
            .iter()
            .map(|(_, b)| b.clone())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(joined.contains("agents-wins"));
        assert!(
            !joined.contains("claude-loses"),
            "CLAUDE.md must be skipped when AGENTS.md is present"
        );
    }

    #[test]
    fn context_filenames_are_case_insensitive() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("proj");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("Claude.MD"), "mixed-case rules").unwrap();

        let joined: String = load_context_files(&dir)
            .iter()
            .map(|(_, b)| b.clone())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(joined.contains("mixed-case rules"));
    }

    #[test]
    fn system_md_overrides_base_prompt() {
        let tmp = tempfile::tempdir().unwrap();
        let claude_dir = tmp.path().join(".claude");
        fs::create_dir_all(&claude_dir).unwrap();
        fs::write(claude_dir.join("SYSTEM.md"), "OVERRIDE IDENTITY").unwrap();

        let prompt = build_system_prompt(&PromptOptions {
            base: "DEFAULT IDENTITY",
            append: None,
            cwd: tmp.path(),
            include_context_files: false,
            include_skills: false,
        });
        assert!(prompt.contains("OVERRIDE IDENTITY"));
        assert!(
            !prompt.contains("DEFAULT IDENTITY"),
            "on-disk SYSTEM.md must replace the built-in base"
        );
    }

    #[test]
    fn tzif_offset_reads_fixed_offset_zone() {
        // A minimal v1 TZif blob with one type (gmtoff = +3600s) and no transitions: the parser must
        // fall back to that type and return +3600 regardless of `now`.
        let mut blob: Vec<u8> = Vec::new();
        blob.extend_from_slice(b"TZif"); // magic
        blob.push(0); // version '\0' → 32-bit block only
        blob.extend_from_slice(&[0u8; 15]); // reserved
        // counts: isut, isstd, leap, time, type, char
        for c in [0u32, 0, 0, 0, 1, 1] {
            blob.extend_from_slice(&c.to_be_bytes());
        }
        // ttinfo[0]: gmtoff=3600, isdst=0, abbrind=0
        blob.extend_from_slice(&3600i32.to_be_bytes());
        blob.push(0);
        blob.push(0);
        blob.push(0); // abbreviation chars (charcnt = 1)

        assert_eq!(tzif_offset(&blob, 1_700_000_000), Some(3600));
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
