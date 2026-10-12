//! `--tool-tuning KEY=VALUE` — the A/B knobs on how built-in tools present themselves to a model.
//!
//! One repeatable flag rather than a flag per knob: every key here exists so an eval can compare
//! arms, each defaults to today's behavior, and the losers get deleted. Keeping them behind one
//! typed parser means adding or removing a knob touches this file and the tool, not the CLI and
//! `serve` plumbing. An unknown key or value is a startup error, never a silent default — a typo in
//! an eval arm must not quietly run the control.
//!
//! Keys:
//! - `edit-schema=combined|edits|split` — see [`edit::EditSchema`].
//! - `edit-pair=strict|lenient` — see [`edit::EditPair`].
//! - `bash-max-bytes=N` (bytes, or `Nk` for KiB; 1k–50k) — `bash`'s display cap, see
//!   [`bash_output::BashOutput`].
//! - `bash-test-digest=off|pytest` — see [`bash_output::BashOutput::pytest_digest`].
//! - `test-guideline=default|pytest` — see [`TestGuideline`].
//! - `search-tool=separate|merged` — see [`SearchTool`].
//! - `tool-guidelines=default|neutral|none` — see [`ToolGuidelines`].

use super::{bash_output, edit};

/// The parsed knobs. `Default` is the production behavior.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ToolTuning {
    pub edit: edit::EditOptions,
    pub bash: bash_output::BashOutput,
    pub test_guideline: TestGuideline,
    pub search_tool: SearchTool,
    pub tool_guidelines: ToolGuidelines,
}

/// How content and file-name search reach the model (`search-tool=…`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SearchTool {
    /// `grep`, `find` and `ls`, as today.
    #[default]
    Separate,
    /// One `search` tool routing to those three — see [`super::search`].
    Merged,
}

/// What the system prompt says about the dedicated tools versus bash (`tool-guidelines=…`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ToolGuidelines {
    /// "Prefer grep/read over bash…" and "Never bash-cat…", as today.
    #[default]
    Default,
    /// One outcome-framed bullet: use the tools or bash, but bound what you print.
    Neutral,
    /// Neither bullet. pi's own one-line read guideline stays.
    None,
}

/// What the system prompt tells the model about running tests (`test-guideline=…`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TestGuideline {
    /// Today's only mention: "never `| head`/`| tail` installs or tests", inside the grep/read bullet.
    #[default]
    Default,
    /// A bullet of its own, gated on `bash` alone, that names the flags to use (`-q --tb=short`,
    /// `-x`, `-k`) — the alternative the default forbids without offering. GPT-6 Luna copied the
    /// task's bare `python -m pytest tests` in 41% of its test calls.
    Pytest,
}

fn parse_bytes(value: &str) -> Result<usize, String> {
    let (digits, scale) = match value.strip_suffix(['k', 'K']) {
        Some(d) => (d, 1024),
        None => (value, 1),
    };
    let n = digits
        .parse::<usize>()
        .ok()
        .and_then(|n| n.checked_mul(scale))
        .ok_or_else(|| format!("invalid bash-max-bytes {value:?}; expected bytes or Nk"))?;
    if !(1024..=super::output::DEFAULT_MAX_BYTES).contains(&n) {
        return Err(format!(
            "bash-max-bytes {value:?} out of range; expected 1k to 50k"
        ));
    }
    Ok(n)
}

impl ToolTuning {
    /// Parse `KEY=VALUE` entries (each may also hold several, comma-separated). Later entries win.
    pub fn parse<S: AsRef<str>>(entries: &[S]) -> Result<Self, String> {
        let mut t = Self::default();
        for entry in entries {
            for kv in entry
                .as_ref()
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                let (key, value) = kv
                    .split_once('=')
                    .ok_or_else(|| format!("--tool-tuning {kv:?}: expected KEY=VALUE"))?;
                let (key, value) = (key.trim(), value.trim());
                match key {
                    "edit-schema" => t.edit.schema = value.parse()?,
                    "edit-pair" => t.edit.pair = value.parse()?,
                    "bash-max-bytes" => t.bash.max_bytes = parse_bytes(value)?,
                    "bash-test-digest" => {
                        t.bash.pytest_digest = match value {
                            "off" => false,
                            "pytest" => true,
                            other => {
                                return Err(format!(
                                    "invalid bash-test-digest {other:?}; expected off or pytest"
                                ));
                            }
                        }
                    }
                    "test-guideline" => {
                        t.test_guideline = match value {
                            "default" => TestGuideline::Default,
                            "pytest" => TestGuideline::Pytest,
                            other => {
                                return Err(format!(
                                    "invalid test-guideline {other:?}; expected default or pytest"
                                ));
                            }
                        }
                    }
                    "search-tool" => {
                        t.search_tool = match value {
                            "separate" => SearchTool::Separate,
                            "merged" => SearchTool::Merged,
                            other => {
                                return Err(format!(
                                    "invalid search-tool {other:?}; expected separate or merged"
                                ));
                            }
                        }
                    }
                    "tool-guidelines" => {
                        t.tool_guidelines = match value {
                            "default" => ToolGuidelines::Default,
                            "neutral" => ToolGuidelines::Neutral,
                            "none" => ToolGuidelines::None,
                            other => {
                                return Err(format!(
                                    "invalid tool-guidelines {other:?}; expected default, neutral or none"
                                ));
                            }
                        }
                    }
                    other => {
                        return Err(format!(
                            "--tool-tuning: unknown key {other:?}; expected one of edit-schema, \
                             edit-pair, bash-max-bytes, bash-test-digest, test-guideline, \
                             search-tool, tool-guidelines"
                        ));
                    }
                }
            }
        }
        Ok(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use edit::{EditPair, EditSchema};

    #[test]
    fn no_entries_is_the_production_default() {
        assert_eq!(
            ToolTuning::parse::<&str>(&[]).unwrap(),
            ToolTuning::default()
        );
        assert_eq!(ToolTuning::default().edit.schema, EditSchema::Combined);
        assert_eq!(ToolTuning::default().edit.pair, EditPair::Strict);
    }

    #[test]
    fn parses_repeated_and_comma_separated_entries_later_wins() {
        let t = ToolTuning::parse(&["edit-schema=split,edit-pair=lenient", "edit-schema=edits"])
            .unwrap();
        assert_eq!(t.edit.schema, EditSchema::EditsOnly);
        assert_eq!(t.edit.pair, EditPair::Lenient);
    }

    #[test]
    fn parses_the_bash_output_and_test_guideline_knobs() {
        let t = ToolTuning::parse(&[
            "bash-max-bytes=16k,bash-test-digest=pytest,test-guideline=pytest",
        ])
        .unwrap();
        assert_eq!(t.bash.max_bytes, 16 * 1024);
        assert!(t.bash.pytest_digest);
        assert_eq!(t.test_guideline, TestGuideline::Pytest);
        assert_eq!(
            ToolTuning::parse(&["bash-max-bytes=20000"])
                .unwrap()
                .bash
                .max_bytes,
            20_000
        );
        assert_eq!(ToolTuning::default().bash.max_bytes, 50 * 1024);
        assert!(!ToolTuning::default().bash.pytest_digest);
    }

    #[test]
    fn parses_the_tool_choice_knobs() {
        let t = ToolTuning::parse(&["search-tool=merged,tool-guidelines=neutral"]).unwrap();
        assert_eq!(t.search_tool, SearchTool::Merged);
        assert_eq!(t.tool_guidelines, ToolGuidelines::Neutral);
        assert_eq!(
            ToolTuning::parse(&["tool-guidelines=none"])
                .unwrap()
                .tool_guidelines,
            ToolGuidelines::None
        );
        assert_eq!(ToolTuning::default().search_tool, SearchTool::Separate);
        assert_eq!(
            ToolTuning::default().tool_guidelines,
            ToolGuidelines::Default
        );
    }

    #[test]
    fn unknown_keys_values_and_shapes_are_errors_not_silent_defaults() {
        for bad in [
            "edit-schema=nope",
            "edit-pair=loose",
            "edit-shape=split",
            "edit-schema",
            "bash-max-bytes=0",
            "bash-max-bytes=100k",
            "bash-max-bytes=lots",
            "bash-test-digest=on",
            "test-guideline=terse",
            "search-tool=unified",
            "tool-guidelines=strong",
        ] {
            assert!(ToolTuning::parse(&[bad]).is_err(), "{bad} must be rejected");
        }
    }
}
