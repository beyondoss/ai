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

use super::edit;

/// The parsed knobs. `Default` is the production behavior.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ToolTuning {
    pub edit: edit::EditOptions,
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
                    other => {
                        return Err(format!(
                            "--tool-tuning: unknown key {other:?}; expected one of edit-schema, edit-pair"
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
    fn unknown_keys_values_and_shapes_are_errors_not_silent_defaults() {
        for bad in [
            "edit-schema=nope",
            "edit-pair=loose",
            "edit-shape=split",
            "edit-schema",
        ] {
            assert!(ToolTuning::parse(&[bad]).is_err(), "{bad} must be rejected");
        }
    }
}
