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
    /// Extra text appended after the base (e.g. `--append-system-prompt`). When `None`, an on-disk
    /// `APPEND_SYSTEM.md` (project, then user — same discovery/trust order as `SYSTEM.md`) is used
    /// instead, if one exists; an explicit `append` here wins outright rather than combining with it.
    pub append: Option<&'a str>,
    /// The working directory whose project-instruction files and skills to load.
    pub cwd: &'a Path,
    /// Whether to discover and inject `AGENTS.md`/`CLAUDE.md` project-instruction files.
    pub include_context_files: bool,
    /// Whether to discover and advertise skills at all — an independent on/off switch, not a trust
    /// gate (see `project_trusted` below for that). A real caller always passes `true`; tests pass
    /// `false` to skip touching the developer's actual `~/.claude/skills`.
    pub include_skills: bool,
    /// Whether `cwd` is a trusted project (an explicit `--trust-project`/RPC override, or recorded in
    /// `TrustStore`). Gates the *project-local* `SYSTEM.md`/`APPEND_SYSTEM.md` overrides and the
    /// project-local skills root (`<cwd>/.claude/skills`, see `skills::discover`) — an untrusted
    /// checkout can't replace or extend the agent's identity or inject its own skills just by shipping
    /// the files (see `crate::trust_store`). The user-global `~/.claude/SYSTEM.md`/`APPEND_SYSTEM.md`
    /// overrides and `~/.claude/skills` are unaffected either way: they're the operator's own machine,
    /// not something a repo checkout controls.
    pub project_trusted: bool,
}

/// Build the full system prompt for a session: the static base (see [`build_static_system_prompt`])
/// plus the current dynamic footer (see [`dynamic_footer`]). A caller that rebuilds the prompt every
/// turn just to pick up the current date (e.g. `serve`'s per-turn refresh) should call the two pieces
/// separately instead — see `serve::build_agent`'s callers.
pub fn build_system_prompt(opts: &PromptOptions) -> String {
    let mut s = build_static_system_prompt(opts);
    s.push_str(&dynamic_footer(opts.cwd));
    s
}

/// Everything in the system prompt except the per-turn dynamic footer (current date/cwd, see
/// [`dynamic_footer`]): the base identity or its `SYSTEM.md` override, project instructions, and
/// discovered skills. Walks the filesystem (project-instruction files, skill directories), so it's
/// meant to be rebuilt only at startup and on a model/thinking-triggered `Agent` rebuild — not on every
/// turn, unlike the cheap dynamic footer.
pub fn build_static_system_prompt(opts: &PromptOptions) -> String {
    // An on-disk `SYSTEM.md` (project `<cwd>/.claude/`, else user `~/.claude/`) replaces the built-in
    // base entirely — that's how a project pins its own agent identity (pi's resource-loader does the
    // same). Absent one, the caller-supplied `base` stands.
    let mut s = system_prompt_override(opts.cwd, opts.project_trusted)
        .unwrap_or_else(|| opts.base.to_string());
    let append = opts
        .append
        .map(str::to_string)
        .or_else(|| append_system_prompt_override(opts.cwd, opts.project_trusted));
    if let Some(extra) = append {
        s.push_str("\n\n");
        s.push_str(&extra);
    }

    if opts.include_context_files {
        let files = load_context_files(opts.cwd);
        if !files.is_empty() {
            // Wrapped in an outer `<project_context>` element (matching the reference agent) so the
            // model sees these as a distinct, labeled section rather than bare instruction blocks
            // floating in the prompt.
            s.push_str("\n\n<project_context>\n\n");
            s.push_str("Project-specific instructions and guidelines:\n\n");
            for (path, body) in files {
                s.push_str(&format!(
                    "<project_instructions path=\"{path}\">\n{}\n</project_instructions>\n\n",
                    body.trim()
                ));
            }
            s.push_str("</project_context>");
        }
    }

    if opts.include_skills {
        let skills: Vec<Skill> = skills::discover(opts.cwd, opts.project_trusted);
        if !skills.is_empty() {
            s.push_str("\n\n");
            s.push_str(&skills::format_available(&skills));
        }
    }

    s
}

/// The cheap, time-varying tail of the system prompt: the current date and working directory. Does no
/// filesystem discovery (unlike [`build_static_system_prompt`]), so it's cheap enough to recompute
/// before every turn — the one part of the prompt that's actually time-varying.
pub fn dynamic_footer(cwd: &Path) -> String {
    format!(
        "\n\nCurrent date: {}\nCurrent working directory: {}",
        today(),
        cwd.display()
    )
}

/// A `SYSTEM.md` override: project-local (`<cwd>/.claude/SYSTEM.md`, only when `project_trusted`) takes
/// precedence over the user one (`~/.claude/SYSTEM.md`, always eligible — it's the operator's own
/// machine). Returns its raw contents, or `None` when neither exists / is blank / the project one
/// exists but isn't trusted (falls through to the global candidate exactly as if the project file were
/// simply absent — no different fallback than the untrusted-file-doesn't-exist case).
fn system_prompt_override(cwd: &Path, project_trusted: bool) -> Option<String> {
    discover_claude_file(cwd, project_trusted, "SYSTEM.md")
}

/// An `APPEND_SYSTEM.md` on disk (same project-then-user discovery/trust order as `SYSTEM.md`) is
/// additive rather than a replacement: its contents are appended after the base/override system prompt.
/// Only consulted when the caller didn't already supply an explicit `append` (e.g.
/// `--append-system-prompt`) — an explicit override wins outright rather than combining with the
/// on-disk file, matching pi's `resource-loader.ts` (`appendSystemPromptSource ?? discovered`).
fn append_system_prompt_override(cwd: &Path, project_trusted: bool) -> Option<String> {
    discover_claude_file(cwd, project_trusted, "APPEND_SYSTEM.md")
}

/// Shared project-then-user `.claude/<filename>` discovery: project-local (only when `project_trusted`)
/// takes precedence over the user-global one (always eligible — it's the operator's own machine).
/// Returns `None` when neither exists / is blank / the project one exists but isn't trusted (falls
/// through to the global candidate exactly as if the project file were simply absent).
fn discover_claude_file(cwd: &Path, project_trusted: bool, filename: &str) -> Option<String> {
    let mut candidates = Vec::new();
    if project_trusted {
        candidates.push(cwd.join(".claude").join(filename));
    }
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        candidates.push(home.join(".claude").join(filename));
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

/// The host's UTC offset in seconds at `now`. We have no date crate and `unsafe_code` is forbidden (so
/// no libc `localtime`); parsing a TZif file is the dependency-free, safe way to get the local offset.
/// `TZ` takes precedence over the system zoneinfo (`/etc/localtime`) when it names a resolvable zone —
/// the precedence every libc/ICU implementation uses, and the one that matters in a container: `TZ` is
/// routinely set to override a base image whose `/etc/localtime` is still UTC. Any problem — `TZ`
/// unset/unresolvable, missing file, unknown format — degrades to `0` (UTC), which is correct, just not
/// local.
fn local_utc_offset(now: i64) -> i64 {
    tz_env_offset(now)
        .or_else(|| {
            fs::read("/etc/localtime")
                .ok()
                .and_then(|data| tzif_offset(&data, now))
        })
        .unwrap_or(0)
}

/// The offset from the `TZ` environment variable, if set to a resolvable zone. `TZ` unset falls
/// through to `None` so the caller tries `/etc/localtime` next.
fn tz_env_offset(now: i64) -> Option<i64> {
    let tz = std::env::var("TZ").ok()?;
    tz_string_offset(&tz, now)
}

/// Resolve a `TZ`-style value — an IANA zone name (e.g. `America/New_York`, optionally `:`-prefixed per
/// POSIX convention) or an absolute zoneinfo path — to its UTC offset at `now`. The form containers/CI
/// set almost universally. A raw POSIX offset-rule string (`EST5EDT,M3.2.0,M11.1.0`) isn't parsed (no
/// rule-transition logic here, just TZif files); empty-but-for-the-`:`, or naming something we can't
/// resolve to a real zoneinfo file, falls through to `None`.
///
/// Split from [`tz_env_offset`] so the zone-resolution logic is unit-testable without mutating the
/// process's environment — `std::env::set_var` is unsafe to call from a test that may run in parallel
/// with others reading `TZ`.
fn tz_string_offset(tz: &str, now: i64) -> Option<i64> {
    let zone = tz.strip_prefix(':').unwrap_or(tz);
    if zone.is_empty() {
        return Some(0); // POSIX: an empty TZ value means UTC
    }
    let path = if zone.starts_with('/') {
        PathBuf::from(zone)
    } else {
        Path::new("/usr/share/zoneinfo").join(zone)
    };
    let data = fs::read(path).ok()?;
    tzif_offset(&data, now)
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
    fn system_md_overrides_base_prompt_when_trusted() {
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
            project_trusted: true,
        });
        assert!(prompt.contains("OVERRIDE IDENTITY"));
        assert!(
            !prompt.contains("DEFAULT IDENTITY"),
            "a trusted project's on-disk SYSTEM.md must replace the built-in base"
        );
    }

    #[test]
    fn system_md_is_ignored_when_project_is_untrusted() {
        // The whole point of the trust gate: an untrusted checkout can't hijack the agent's identity
        // just by shipping a `.claude/SYSTEM.md` — it falls through exactly as if the file were
        // absent, not a different (or erroring) fallback.
        let tmp = tempfile::tempdir().unwrap();
        let claude_dir = tmp.path().join(".claude");
        fs::create_dir_all(&claude_dir).unwrap();
        fs::write(claude_dir.join("SYSTEM.md"), "MALICIOUS OVERRIDE").unwrap();

        let prompt = build_system_prompt(&PromptOptions {
            base: "DEFAULT IDENTITY",
            append: None,
            cwd: tmp.path(),
            include_context_files: false,
            include_skills: false,
            project_trusted: false,
        });
        assert!(prompt.contains("DEFAULT IDENTITY"));
        assert!(!prompt.contains("MALICIOUS OVERRIDE"));
    }

    #[test]
    fn append_system_md_is_appended_when_trusted_and_no_explicit_override() {
        let tmp = tempfile::tempdir().unwrap();
        let claude_dir = tmp.path().join(".claude");
        fs::create_dir_all(&claude_dir).unwrap();
        fs::write(claude_dir.join("APPEND_SYSTEM.md"), "EXTRA HOUSE RULES").unwrap();

        let prompt = build_system_prompt(&PromptOptions {
            base: "DEFAULT IDENTITY",
            append: None,
            cwd: tmp.path(),
            include_context_files: false,
            include_skills: false,
            project_trusted: true,
        });
        assert!(prompt.contains("DEFAULT IDENTITY"));
        assert!(
            prompt.contains("EXTRA HOUSE RULES"),
            "a trusted project's on-disk APPEND_SYSTEM.md must be appended to the system prompt"
        );
    }

    #[test]
    fn append_system_md_is_ignored_when_project_is_untrusted() {
        let tmp = tempfile::tempdir().unwrap();
        let claude_dir = tmp.path().join(".claude");
        fs::create_dir_all(&claude_dir).unwrap();
        fs::write(claude_dir.join("APPEND_SYSTEM.md"), "MALICIOUS EXTRA RULES").unwrap();

        let prompt = build_system_prompt(&PromptOptions {
            base: "DEFAULT IDENTITY",
            append: None,
            cwd: tmp.path(),
            include_context_files: false,
            include_skills: false,
            project_trusted: false,
        });
        assert!(!prompt.contains("MALICIOUS EXTRA RULES"));
    }

    #[test]
    fn explicit_append_wins_outright_over_on_disk_append_system_md() {
        let tmp = tempfile::tempdir().unwrap();
        let claude_dir = tmp.path().join(".claude");
        fs::create_dir_all(&claude_dir).unwrap();
        fs::write(claude_dir.join("APPEND_SYSTEM.md"), "ON-DISK APPEND").unwrap();

        let prompt = build_system_prompt(&PromptOptions {
            base: "DEFAULT IDENTITY",
            append: Some("CLI APPEND"),
            cwd: tmp.path(),
            include_context_files: false,
            include_skills: false,
            project_trusted: true,
        });
        assert!(prompt.contains("CLI APPEND"));
        assert!(
            !prompt.contains("ON-DISK APPEND"),
            "an explicit --append-system-prompt must win outright, not combine with the on-disk file"
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
    fn tz_string_offset_resolves_a_real_iana_zone() {
        // UTC's zoneinfo file exists on essentially every Linux distro (including this test machine's)
        // and has a fixed 0 offset — a reliable resolvable-zone case without hardcoding a path.
        assert_eq!(tz_string_offset("UTC", 1_700_000_000), Some(0));
        // POSIX `:`-prefix convention (explicitly "this is a file reference") resolves the same way.
        assert_eq!(tz_string_offset(":UTC", 1_700_000_000), Some(0));
    }

    #[test]
    fn tz_string_offset_empty_value_is_utc() {
        // POSIX: an empty `TZ` value means UTC.
        assert_eq!(tz_string_offset("", 1_700_000_000), Some(0));
        assert_eq!(tz_string_offset(":", 1_700_000_000), Some(0));
    }

    #[test]
    fn tz_string_offset_falls_through_on_unresolvable_value() {
        // A raw POSIX offset-rule string isn't parsed (no rule-transition logic here) — falls through
        // to `None` so the caller tries `/etc/localtime` next, rather than erroring the date entirely.
        assert_eq!(
            tz_string_offset("EST5EDT,M3.2.0,M11.1.0", 1_700_000_000),
            None
        );
        assert_eq!(tz_string_offset("Not/A/Real/Zone", 1_700_000_000), None);
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
            project_trusted: false,
        });
        assert!(prompt.contains("You are an agent."));
        assert!(prompt.contains("Stay terse."));
        assert!(prompt.contains("<project_context>"));
        assert!(prompt.contains("</project_context>"));
        assert!(prompt.contains("<project_instructions"));
        assert!(prompt.contains("Be excellent."));
        assert!(prompt.contains("Current date:"));
        assert!(prompt.contains("Current working directory:"));
        // The instructions block must be nested *inside* the wrapper, not just present somewhere.
        let wrapper_start = prompt.find("<project_context>").unwrap();
        let wrapper_end = prompt.find("</project_context>").unwrap();
        let instructions_pos = prompt.find("<project_instructions").unwrap();
        assert!(instructions_pos > wrapper_start && instructions_pos < wrapper_end);
    }

    #[test]
    fn static_prompt_plus_dynamic_footer_equals_the_full_prompt() {
        let tmp = tempfile::tempdir().unwrap();
        let opts = PromptOptions {
            base: "You are an agent.",
            append: None,
            cwd: tmp.path(),
            include_context_files: false,
            include_skills: false,
            project_trusted: false,
        };
        let full = build_system_prompt(&opts);
        let static_part = build_static_system_prompt(&opts);
        let footer = dynamic_footer(tmp.path());
        assert_eq!(full, format!("{static_part}{footer}"));
        // The static half must carry no date/cwd at all — that's the whole point of the split.
        assert!(!static_part.contains("Current date:"));
        assert!(!static_part.contains("Current working directory:"));
        assert!(footer.contains("Current date:"));
        assert!(footer.contains("Current working directory:"));
    }

    #[test]
    fn project_context_wrapper_is_absent_when_context_files_are_disabled() {
        // `include_context_files: false` skips `load_context_files` entirely — the one way to prove
        // the wrapper is conditional without depending on `$HOME/.claude` being empty (which this test
        // can't safely control: mutating `HOME` via `std::env::set_var` is unsafe to do from a test
        // that may run in parallel with others, and this repo's own `~/.claude/CLAUDE.md` genuinely
        // exists on a real dev machine, so an empty-tempdir-cwd variant of this test would be flaky).
        let tmp = tempfile::tempdir().unwrap();
        let prompt = build_system_prompt(&PromptOptions {
            base: "You are an agent.",
            append: None,
            cwd: tmp.path(),
            include_context_files: false,
            include_skills: false,
            project_trusted: false,
        });
        assert!(!prompt.contains("<project_context>"));
    }
}
