//! Where a model id keeps its version, and which line of models it belongs to.
//!
//! `claude-haiku-4-5`, `anthropic/claude-haiku-4.5` and `us.anthropic.claude-haiku-4-5-20251001-v1:0`
//! are each version `[4, 5]` of a line: `claude-haiku-*`, `anthropic/claude-haiku-*`,
//! `us.anthropic.claude-haiku-*`. A dated snapshot (`-20251001`, `-0813`, `-2025-10-01`) and a
//! Bedrock revision (`-v1:0`) are not part of the line, so a vendor's dated and undated spellings
//! of one release share it. An id whose first number is glued to letters after it (`gpt-4o`,
//! `gpt-oss-120b`, `muse-glimmer-30b`) or that has no number (`inkling`) has no lineage: it is a
//! name, not a version, and nothing can succeed it mechanically.

/// An id split around its version.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lineage {
    /// The id, lowercased, with the version replaced by `*` and snapshot tokens dropped.
    pub line: String,
    pub version: Vec<u32>,
    /// Byte range of the version in the id.
    pub span: (usize, usize),
    /// How the version is written: `.` (`4.5`, and any one-number version) or `-` (`4-5`).
    pub sep: char,
    /// Whether the id carries a snapshot token (a date or a Bedrock revision).
    pub dated: bool,
}

fn digits(b: &[u8], from: usize) -> usize {
    b[from..].iter().take_while(|c| c.is_ascii_digit()).count()
}

/// The end of the token starting at `at` is the end of the id or a non-alphanumeric byte.
fn token_ends(b: &[u8], at: usize) -> bool {
    b.get(at).is_none_or(|c| !c.is_ascii_alphanumeric())
}

pub fn parse(id: &str) -> Option<Lineage> {
    let b = id.as_bytes();
    let start = b.iter().position(u8::is_ascii_digit)?;
    if start > 0 && b[start - 1] == b'.' {
        return None;
    }
    let mut version = Vec::new();
    let mut at = start;
    let mut sep = '.';
    loop {
        let n = digits(b, at);
        version.push(id[at..at + n].parse().ok()?);
        at += n;
        if b.get(at) == Some(&b'.') && b.get(at + 1).is_some_and(u8::is_ascii_digit) {
            if sep == '-' {
                return None;
            }
            at += 1;
            continue;
        }
        // `-5` after `4` (Anthropic's `4-5`) continues the version; `-20251001` and `-31b` don't.
        if version.len() == 1 || sep == '-' {
            let n = if b.get(at) == Some(&b'-') {
                digits(b, at + 1)
            } else {
                0
            };
            if (1..=2).contains(&n) && token_ends(b, at + 1 + n) && b.get(at + 1 + n) != Some(&b'.')
            {
                sep = '-';
                at += 1;
                continue;
            }
        }
        break;
    }
    if !token_ends(b, at) {
        return None;
    }
    let (rest, dated) = undate(&id[at..]);
    Some(Lineage {
        line: format!("{}*{rest}", &id[..start]).to_ascii_lowercase(),
        version,
        span: (start, at),
        sep,
        dated,
    })
}

/// `s` without its snapshot tokens: `-` then four or more digits (and a following `-MM-DD`), or a
/// trailing Bedrock revision `-v1:0` / `-v1`.
fn undate(s: &str) -> (String, bool) {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut dated = false;
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'-' {
            let n = digits(b, i + 1);
            if n >= 4 && token_ends(b, i + 1 + n) {
                let mut j = i + 1 + n;
                // `-2025-10-01`
                if n == 4 {
                    while b.get(j) == Some(&b'-') && digits(b, j + 1) == 2 && token_ends(b, j + 3) {
                        j += 3;
                    }
                }
                i = j;
                dated = true;
                continue;
            }
            // `-v1:0` or `-v1` at the end.
            if b.get(i + 1) == Some(&b'v') {
                let n = digits(b, i + 2);
                let mut j = i + 2 + n;
                if n > 0 && b.get(j) == Some(&b':') {
                    let m = digits(b, j + 1);
                    if m > 0 {
                        j += 1 + m;
                    }
                }
                if n > 0 && j == b.len() {
                    dated = true;
                    break;
                }
            }
        }
        out.push(char::from(b[i]));
        i += 1;
    }
    (out, dated)
}

/// The id with its snapshot tokens dropped (`claude-haiku-4-5-20251001` → `claude-haiku-4-5`).
pub fn undated(id: &str) -> String {
    match parse(id) {
        Some(l) => format!("{}{}", &id[..l.span.1], undate(&id[l.span.1..]).0),
        None => undate(id).0,
    }
}

/// `[5, 5]` written with `sep`.
pub fn format_version(v: &[u32], sep: char) -> String {
    v.iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(&sep.to_string())
}

/// `text` with its version replaced by `new`, written the way `text` writes its own (`.` for a
/// one-number version: `Claude Opus 5` → `Claude Opus 5.5`). `None` when `text` has no lineage.
pub fn with_version(text: &str, new: &[u32]) -> Option<String> {
    let l = parse(text)?;
    Some(format!(
        "{}{}{}",
        &text[..l.span.0],
        format_version(new, l.sep),
        &text[l.span.1..]
    ))
}

/// The family a name is in: its vendor prefix and the letters before its first number or
/// separator (`anthropic/claude-haiku-4.5` → `anthropic/claude`, `qwen/qwen3.8-flash` → `qwen/qwen`).
pub fn brand(id: &str) -> String {
    let (prefix, name) = id.rsplit_once('/').map_or(("", id), |(p, n)| (p, n));
    let stem: String = name
        .chars()
        .take_while(char::is_ascii_alphabetic)
        .collect::<String>()
        .to_ascii_lowercase();
    if prefix.is_empty() {
        stem
    } else {
        format!("{}/{stem}", prefix.to_ascii_lowercase())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn lv(id: &str) -> Option<(String, Vec<u32>, bool)> {
        parse(id).map(|l| (l.line, l.version, l.dated))
    }

    #[test]
    fn versions_and_lines() {
        type Case<'a> = (&'a str, Option<(&'a str, &'a [u32], bool)>);
        let cases: &[Case<'_>] = &[
            ("claude-haiku-4-5", Some(("claude-haiku-*", &[4, 5], false))),
            (
                "claude-haiku-4-5-20251001",
                Some(("claude-haiku-*", &[4, 5], true)),
            ),
            ("claude-opus-5", Some(("claude-opus-*", &[5], false))),
            (
                "anthropic/claude-haiku-4.5",
                Some(("anthropic/claude-haiku-*", &[4, 5], false)),
            ),
            (
                "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                Some(("us.anthropic.claude-haiku-*", &[4, 5], true)),
            ),
            (
                "us.anthropic.claude-opus-4-6-v1",
                Some(("us.anthropic.claude-opus-*", &[4, 6], true)),
            ),
            (
                "us.anthropic.claude-haiku-5-5",
                Some(("us.anthropic.claude-haiku-*", &[5, 5], false)),
            ),
            ("gpt-5.4-mini", Some(("gpt-*-mini", &[5, 4], false))),
            ("gpt-6-sol-pro", Some(("gpt-*-sol-pro", &[6], false))),
            ("gpt-5.4-2026-03-05", Some(("gpt-*", &[5, 4], true))),
            ("grok-4.20", Some(("grok-*", &[4, 20], false))),
            (
                "grok-4.20-multi-agent",
                Some(("grok-*-multi-agent", &[4, 20], false)),
            ),
            (
                "qwen/qwen3.8-flash",
                Some(("qwen/qwen*-flash", &[3, 8], false)),
            ),
            ("Qwen/Qwen3.5-9B", Some(("qwen/qwen*-9b", &[3, 5], false))),
            (
                "moonshotai/kimi-k3",
                Some(("moonshotai/kimi-k*", &[3], false)),
            ),
            (
                "deepseek-ai/DeepSeek-V4-Pro-0813",
                Some(("deepseek-ai/deepseek-v*-pro", &[4], true)),
            ),
            ("gemma-4-31b-it", Some(("gemma-*-31b-it", &[4], false))),
            (
                "llama-3.3-70b-versatile",
                Some(("llama-*-70b-versatile", &[3, 3], false)),
            ),
            (
                "claude-3-7-sonnet",
                Some(("claude-*-sonnet", &[3, 7], false)),
            ),
            ("Claude Haiku 4.5", Some(("claude haiku *", &[4, 5], false))),
            ("o3", Some(("o*", &[3], false))),
            ("gpt-4o", None),
            ("gpt-oss-120b", None),
            ("meta/muse-glimmer-30b", None),
            ("thinkingmachines/inkling", None),
        ];
        for (id, want) in cases {
            let want = want.map(|(l, v, d)| (l.to_owned(), v.to_vec(), d));
            assert_eq!(lv(id), want, "{id}");
        }
    }

    #[test]
    fn swaps_keep_the_spelling() {
        assert_eq!(
            with_version("Claude Opus 5", &[5, 5]).as_deref(),
            Some("Claude Opus 5.5")
        );
        assert_eq!(
            with_version("claude-haiku-4-5", &[5, 5]).as_deref(),
            Some("claude-haiku-5-5")
        );
        assert_eq!(
            with_version("Qwen/Qwen3.5-9B", &[3, 6]).as_deref(),
            Some("Qwen/Qwen3.6-9B")
        );
        assert_eq!(undated("claude-haiku-4-5-20251001"), "claude-haiku-4-5");
        assert_eq!(undated("gpt-5.4-2026-03-05"), "gpt-5.4");
        assert_eq!(brand("anthropic/claude-haiku-4.5"), "anthropic/claude");
        assert_eq!(brand("qwen/qwen3.8-flash"), "qwen/qwen");
        assert_eq!(brand("openai/gpt-oss-120b"), "openai/gpt");
    }
}
