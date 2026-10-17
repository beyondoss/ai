//! The one rule every JSON config store follows when it rewrites its file — `settings.json`,
//! `auth.json`, `mcp_auth.json` and `trusted-projects.json`.
//!
//! Each store *reads* leniently: an entry or key it doesn't understand (a provider this binary
//! doesn't know, a field of the wrong type, a credential shape a newer binary wrote) is skipped
//! with a warning, and a file that can't be read at all counts as empty. That is right for using
//! the store, and wrong as the basis for a write: writing back "what parsed" deletes everything
//! that didn't. Refreshing one provider's token used to delete another provider's login whenever
//! two binary versions shared the file, and one bad byte turned the next `agent trust` into a
//! fresh, empty allowlist.
//!
//! So a mutation works on the raw JSON instead:
//!
//! 1. **Round-trip what isn't understood.** The write starts from the file's current raw object
//!    ([`read_object_for_write`]) and changes only the entries the mutation changed
//!    ([`apply_changes`]); every other key is written back byte-for-byte as it was parsed.
//! 2. **Never overwrite a file that can't be read.** A file that exists but can't be read, isn't
//!    valid JSON, or isn't the expected shape refuses the write with an error naming the file —
//!    the user fixes or removes it. Only a *missing* file starts from empty.

use std::fs;
use std::io::{self, ErrorKind};
use std::path::Path;

use serde_json::{Map, Value};

/// The file's raw JSON, for a write. `Ok(None)` when it doesn't exist (the store starts empty);
/// an error, naming the file, when it exists but can't be read or isn't valid JSON.
pub(crate) fn read_value_for_write(path: &Path) -> io::Result<Option<Value>> {
    let contents = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(refuse(path, &format!("it can't be read ({e})"))),
    };
    serde_json::from_str(&contents)
        .map(Some)
        .map_err(|e| refuse(path, &format!("it is not valid JSON ({e})")))
}

/// [`read_value_for_write`], for a store whose file is a JSON object: a missing file is an empty
/// object, and any other top-level shape refuses the write.
pub(crate) fn read_object_for_write(path: &Path) -> io::Result<Map<String, Value>> {
    match read_value_for_write(path)? {
        None => Ok(Map::new()),
        Some(Value::Object(map)) => Ok(map),
        Some(_) => Err(not_an_object(path)),
    }
}

/// The error for a file whose top level isn't the JSON object a store expects.
pub(crate) fn not_an_object(path: &Path) -> io::Error {
    refuse(path, "it is not a JSON object")
}

/// The error every refused write returns: what is wrong with the file, and what to do about it.
pub(crate) fn refuse(path: &Path, why: &str) -> io::Error {
    io::Error::new(
        ErrorKind::InvalidData,
        format!(
            "refusing to write {}: {why}; fix or remove the file and try again",
            path.display()
        ),
    )
}

/// Bring `raw` in line with a mutation, touching only the top-level keys it changed. `before` and
/// `after` are the store's typed view serialized before and after the mutation: a key whose value
/// differs is replaced, a key that disappeared is removed, and every key in neither — the entries
/// the store couldn't parse — is left exactly as it was. Returns whether anything changed.
pub(crate) fn apply_changes(
    raw: &mut Map<String, Value>,
    before: &Map<String, Value>,
    after: &Map<String, Value>,
) -> bool {
    let mut changed = false;
    for (key, value) in after {
        if before.get(key) != Some(value) {
            raw.insert(key.clone(), value.clone());
            changed = true;
        }
    }
    for key in before.keys() {
        if !after.contains_key(key) {
            raw.remove(key);
            changed = true;
        }
    }
    changed
}

/// Write `value` to `path` durably and atomically, creating the parent directory. With `private`,
/// the file is created `0600` and the mode is re-asserted after every write: `write_atomic` only
/// copies an existing file's mode forward, so a first write would otherwise land at the umask
/// default and a mode loosened out of band would stay loose.
pub(crate) fn write_json(path: &Path, value: &Value, private: bool) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    if private && !path.exists() {
        create_private(path)?;
    }
    let body = serde_json::to_string_pretty(value).map_err(io::Error::other)?;
    let path_str = path
        .to_str()
        .ok_or_else(|| io::Error::other(format!("non-UTF-8 path: {}", path.display())))?;
    crate::tools::write_atomic_durable(path_str, body.as_bytes())?;
    if private {
        set_private(path)?;
    }
    Ok(())
}

/// Create an empty file with `0600` set atomically at creation (Unix), so a secret is never
/// readable by others even for a moment.
fn create_private(path: &Path) -> io::Result<()> {
    let mut opts = fs::OpenOptions::new();
    opts.write(true).create(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    opts.open(path)?;
    Ok(())
}

/// Re-assert `0600` on a written file (Unix only).
fn set_private(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn obj(v: Value) -> Map<String, Value> {
        match v {
            Value::Object(m) => m,
            _ => unreachable!(),
        }
    }

    #[test]
    fn apply_changes_touches_only_what_the_mutation_changed() {
        let mut raw = obj(json!({ "a": 1, "b": 2, "garbage": "kept", "drop": 0 }));
        let before = obj(json!({ "a": 1, "b": 2, "drop": 0 }));
        let after = obj(json!({ "a": 1, "b": 3, "new": true }));
        assert!(apply_changes(&mut raw, &before, &after));
        assert_eq!(
            Value::Object(raw),
            json!({ "a": 1, "b": 3, "garbage": "kept", "new": true })
        );
    }

    #[test]
    fn apply_changes_reports_no_change_for_an_identical_view() {
        let mut raw = obj(json!({ "a": 1, "x": "unparsed" }));
        let view = obj(json!({ "a": 1 }));
        assert!(!apply_changes(&mut raw, &view, &view));
    }

    #[test]
    fn a_missing_file_reads_as_empty_but_a_corrupt_one_refuses() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("missing.json");
        assert!(read_object_for_write(&missing).unwrap().is_empty());

        let corrupt = dir.path().join("corrupt.json");
        fs::write(&corrupt, "{ not json").unwrap();
        let err = read_object_for_write(&corrupt).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidData);
        assert!(err.to_string().contains("corrupt.json"), "{err}");
        assert!(err.to_string().contains("fix or remove"), "{err}");

        let array = dir.path().join("array.json");
        fs::write(&array, "[1, 2]").unwrap();
        assert!(read_object_for_write(&array).is_err());
    }
}
