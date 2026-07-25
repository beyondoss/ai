//! `memory` — the agent's knowledge store, across one or more mounted roots.
//!
//! One tool with a `command` enum (`view`/`create`/`str_replace`/`insert`/`delete`/`rename`/`search`)
//! over logical roots, mirroring Anthropic's canonical `memory_20250818` tool so the model's existing
//! skills transfer. Two roots can be mounted at once: `/memories` (durable, cross-session project
//! knowledge) and `/session` (per-session working memory that survives compaction and is cleared when
//! the session ends). The tool is a thin, backend-agnostic adapter: it validates the model's arguments
//! (notably the [`MemPath`], which confines every operation to a store), **routes the path to the mount
//! whose root it names** ([`MemPath::classify`]), and forwards to that [`MemoryBackend`]. What actually
//! persists — local files today, a networked store later — is the backend's concern. See [`crate::memory`].
//!
//! **The mounts are host-owned.** Each is an `Arc<dyn MemoryBackend>` cloned into the tool at
//! registration, not a field the tool constructs — because the registry is rebuilt on every
//! `set_model`/`set_thinking`, durable state must survive a rebuild (the same reason
//! `structured_output`'s `OutputSlot` and `subagent`'s `SubagentCtx` are host-owned).
//!
//! **`additionalProperties: false`** keeps the schema tight, which means the dispatch loop's
//! `_model_supports_vision` stamp would otherwise fail validation — so it's stripped at the top of `run`,
//! exactly as `structured_output` does.

use std::fmt::Write;
use std::sync::Arc;

use agent_core::tool::{MODEL_SUPPORTS_VISION_KEY, Tool};
use agent_core::{ToolError, ToolOutput};
use async_trait::async_trait;
use serde_json::{Value, json};

use crate::memory::{Entry, Hit, MemPath, MemoryBackend, MemoryError, Mount, MountKind, View};
use crate::tools::output::cap_listing_bytes;

pub const NAME: &str = "memory";

const DESCRIPTION: &str = "Memory store. `/memories` is durable, cross-session project knowledge \
    (build/test commands, architecture decisions, debugging lessons, user preferences). `/session`, when \
    present, is private working memory for THIS task that survives your own context compaction and is \
    cleared when the session ends — write precise facts there you don't want summarized away. Keep each \
    root's `MEMORY.md` as a lean index of one-line pointers to topic files.";

/// The `memory` tool. Holds the mounted stores, each an `Arc` cloned in at registration.
pub struct Memory {
    mounts: Vec<Mount>,
}

impl Memory {
    /// Single durable mount — the common single-root case (and what tests use).
    pub fn new(backend: Arc<dyn MemoryBackend>) -> Self {
        Self {
            mounts: vec![Mount {
                kind: MountKind::Durable,
                backend,
            }],
        }
    }

    /// Mount an explicit set of stores (durable, and optionally session), in display order.
    pub fn mounted(mounts: Vec<Mount>) -> Self {
        Self { mounts }
    }

    /// The roots this tool serves, in mount order — the first is the default for a bare relative path.
    fn roots(&self) -> Vec<&'static str> {
        self.mounts.iter().map(Mount::root).collect()
    }

    /// Classify a model-supplied path across the mounted roots and return the owning backend plus the
    /// validated, rel-only [`MemPath`]. An unknown/foreign root is a retryable `InvalidInput`.
    fn resolve(&self, raw: &str) -> Result<(&Arc<dyn MemoryBackend>, MemPath), ToolError> {
        let roots = self.roots();
        let path = MemPath::classify(raw, &roots).map_err(map_err)?;
        let mount = self
            .mounts
            .iter()
            .find(|m| m.root() == path.root())
            .ok_or_else(|| {
                ToolError::InvalidInput(format!("no memory is mounted at {}", path.root()))
            })?;
        Ok((&mount.backend, path))
    }
}

/// Map a backend failure to a tool error: everything the model can fix from the message becomes an error
/// `tool_result` it retries against ([`ToolError::InvalidInput`]); a real store/IO failure it can't fix
/// becomes [`ToolError::Execution`].
fn map_err(e: MemoryError) -> ToolError {
    match e {
        MemoryError::Backend(_) => ToolError::Execution(e.to_string()),
        _ => ToolError::InvalidInput(e.to_string()),
    }
}

fn str_field<'a>(input: &'a Value, key: &str) -> Result<&'a str, ToolError> {
    input
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| ToolError::InvalidInput(format!("missing `{key}`")))
}

/// Parse an optional `view_range` of `[start, end]` (1-indexed, inclusive).
fn parse_range(input: &Value) -> Result<Option<(usize, usize)>, ToolError> {
    let Some(v) = input.get("view_range") else {
        return Ok(None);
    };
    if v.is_null() {
        return Ok(None);
    }
    let arr = v.as_array().ok_or_else(|| {
        ToolError::InvalidInput("`view_range` must be an array [start, end]".into())
    })?;
    if arr.len() != 2 {
        return Err(ToolError::InvalidInput(
            "`view_range` must be exactly [start, end]".into(),
        ));
    }
    let n = |x: &Value| {
        x.as_u64()
            .map(|u| u as usize)
            .ok_or_else(|| ToolError::InvalidInput("`view_range` values must be integers".into()))
    };
    Ok(Some((n(&arr[0])?, n(&arr[1])?)))
}

/// Render a document's text for the model, capped to a sane context budget.
fn render_doc(path: &MemPath, text: String) -> ToolOutput {
    if text.is_empty() {
        return format!("{} is empty.", path.display()).into();
    }
    let mut out = text;
    cap_listing_bytes(&mut out, "use view_range to read a slice");
    out.into()
}

/// Render a directory listing.
fn render_listing(entries: &[Entry]) -> ToolOutput {
    if entries.is_empty() {
        return "The memory store is empty.".into();
    }
    let mut out = String::new();
    for e in entries {
        // Infallible: writing into a String never errors.
        if e.is_dir {
            let _ = writeln!(out, "{}/", e.path);
        } else {
            let _ = writeln!(out, "{} ({} bytes)", e.path, e.size);
        }
    }
    cap_listing_bytes(&mut out, "narrow with search");
    out.into()
}

/// Render search hits as `path:line: text`.
fn render_hits(query: &str, hits: &[Hit]) -> ToolOutput {
    if hits.is_empty() {
        return format!("No memory matched `{query}`.").into();
    }
    let mut out = String::new();
    for h in hits {
        // Infallible: writing into a String never errors.
        let _ = writeln!(out, "{}:{}: {}", h.path, h.line, h.text);
    }
    cap_listing_bytes(&mut out, "narrow the query");
    out.into()
}

#[async_trait]
impl Tool for Memory {
    fn name(&self) -> &str {
        NAME
    }

    fn description(&self) -> &str {
        DESCRIPTION
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "str_replace", "insert", "delete", "rename", "search"],
                    "description": "The memory operation to perform."
                },
                "path": {
                    "type": "string",
                    "description": "Logical path under a mounted root — /memories/... (durable) or \
                                    /session/... (this-session working memory, when available), e.g. \
                                    /memories/notes.md. `view` of a bare root (/memories or /session) \
                                    lists that store. Required for every command except `search`."
                },
                "file_text": {
                    "type": "string",
                    "description": "For `create`: the full contents of a NEW document (fails if the path already exists — edit an existing one with `str_replace`/`insert`)."
                },
                "old_str": {
                    "type": "string",
                    "description": "For `str_replace`: the exact text to replace (must occur exactly once)."
                },
                "new_str": {
                    "type": "string",
                    "description": "For `str_replace`: the replacement text."
                },
                "insert_line": {
                    "type": "integer",
                    "description": "For `insert`: insert after this 1-indexed line (0 = start of document)."
                },
                "insert_text": {
                    "type": "string",
                    "description": "For `insert`: the text to insert."
                },
                "view_range": {
                    "type": "array",
                    "items": { "type": "integer" },
                    "description": "For `view`: optional [start, end] 1-indexed inclusive line range."
                },
                "new_path": {
                    "type": "string",
                    "description": "For `rename`: the destination path (same root as `path` — a rename \
                                    cannot move between /memories and /session)."
                },
                "query": {
                    "type": "string",
                    "description": "For `search`: case-insensitive substring to find across all memories."
                }
            },
            "required": ["command"],
            "additionalProperties": false
        })
    }

    fn write_target(&self, input: &Value) -> Option<String> {
        // Serialize same-target mutations in one turn (mirrors `write`/`edit`). Read-only commands stay
        // concurrent (None). `rename` groups on its source path.
        let command = input.get("command").and_then(Value::as_str)?;
        let key = match command {
            "create" | "str_replace" | "insert" | "delete" | "rename" => "path",
            _ => return None,
        };
        let raw = input.get(key).and_then(Value::as_str)?;
        // Group on the validated full logical path (root included) so `/memories/x` and `memories/x`
        // collide as they should, while `/memories/x` and `/session/x` stay independent.
        let logical = MemPath::classify(raw, &self.roots()).ok()?.display();
        Some(format!("memory:{logical}"))
    }

    async fn run(&self, mut input: Value) -> Result<ToolOutput, ToolError> {
        // The dispatch loop stamps `_model_supports_vision` onto every call; our schema forbids extra
        // properties, so strip it before anything looks at the input.
        if let Some(obj) = input.as_object_mut() {
            obj.remove(MODEL_SUPPORTS_VISION_KEY);
        }

        let command = str_field(&input, "command")?;
        match command {
            "view" => {
                let (backend, path) = self.resolve(str_field(&input, "path")?)?;
                let range = parse_range(&input)?;
                match backend.view(&path, range).await.map_err(map_err)? {
                    View::Document(text) => Ok(render_doc(&path, text)),
                    View::Listing(entries) => Ok(render_listing(&entries)),
                }
            }
            "create" => {
                let (backend, path) = self.resolve(str_field(&input, "path")?)?;
                let text = str_field(&input, "file_text")?;
                backend.create(&path, text).await.map_err(map_err)?;
                Ok(format!("Created {} ({} bytes).", path.display(), text.len()).into())
            }
            "str_replace" => {
                let (backend, path) = self.resolve(str_field(&input, "path")?)?;
                let old = str_field(&input, "old_str")?;
                // `new_str` may legitimately be absent (delete a snippet) → default to empty.
                let new = input.get("new_str").and_then(Value::as_str).unwrap_or("");
                backend
                    .str_replace(&path, old, new)
                    .await
                    .map_err(map_err)?;
                Ok(format!("Edited {}.", path.display()).into())
            }
            "insert" => {
                let (backend, path) = self.resolve(str_field(&input, "path")?)?;
                let line = input
                    .get("insert_line")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| ToolError::InvalidInput("missing `insert_line`".into()))?
                    as usize;
                let text =
                    str_field(&input, "insert_text").or_else(|_| str_field(&input, "file_text"))?;
                backend.insert(&path, line, text).await.map_err(map_err)?;
                Ok(format!("Inserted into {}.", path.display()).into())
            }
            "delete" => {
                let (backend, path) = self.resolve(str_field(&input, "path")?)?;
                backend.delete(&path).await.map_err(map_err)?;
                Ok(format!("Deleted {}.", path.display()).into())
            }
            "rename" => {
                let (backend, from) = self.resolve(str_field(&input, "path")?)?;
                let (_, to) = self.resolve(str_field(&input, "new_path")?)?;
                // A backend can only move within itself; a cross-root rename has no meaning here.
                if from.root() != to.root() {
                    return Err(ToolError::InvalidInput(format!(
                        "cannot rename across roots ({} → {}); copy the content instead",
                        from.root(),
                        to.root()
                    )));
                }
                backend.rename(&from, &to).await.map_err(map_err)?;
                Ok(format!("Renamed {} to {}.", from.display(), to.display()).into())
            }
            "search" => {
                let query = str_field(&input, "query")?;
                // Fan across every mount; each backend labels its hits with its own root, so the merged
                // listing is unambiguous. Durable first (mount order), then session.
                let mut hits: Vec<Hit> = Vec::new();
                for mount in &self.mounts {
                    hits.extend(mount.backend.search(query).await.map_err(map_err)?);
                }
                Ok(render_hits(query, &hits))
            }
            other => Err(ToolError::InvalidInput(format!(
                "unknown memory command `{other}` (expected view/create/str_replace/insert/delete/rename/search)"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::file::FileBackend;

    fn tool() -> (tempfile::TempDir, Memory) {
        let dir = tempfile::tempdir().unwrap();
        let backend = Arc::new(FileBackend::at(dir.path().join("memory")));
        (dir, Memory::new(backend))
    }

    fn dual_tool() -> (tempfile::TempDir, Memory) {
        let dir = tempfile::tempdir().unwrap();
        let durable = Arc::new(FileBackend::at(dir.path().join("memory")));
        let session = Arc::new(FileBackend::session_at(dir.path().join("session")));
        let mounts = vec![
            Mount {
                kind: MountKind::Durable,
                backend: durable,
            },
            Mount {
                kind: MountKind::Session,
                backend: session,
            },
        ];
        (dir, Memory::mounted(mounts))
    }

    #[tokio::test]
    async fn create_then_view_round_trips_through_the_tool() {
        let (_d, t) = tool();
        let out = t
            .run(json!({ "command": "create", "path": "/memories/x.md", "file_text": "hi\n" }))
            .await
            .unwrap();
        assert!(out.text.contains("Created /memories/x.md"));

        let viewed = t
            .run(json!({ "command": "view", "path": "/memories/x.md" }))
            .await
            .unwrap();
        assert_eq!(viewed.text, "hi\n");
    }

    #[tokio::test]
    async fn the_injected_vision_flag_is_stripped() {
        // Regression: additionalProperties:false + the loop's _model_supports_vision stamp would reject
        // every call unless stripped first.
        let (_d, t) = tool();
        let out = t
            .run(json!({
                "command": "create",
                "path": "/memories/x.md",
                "file_text": "hi",
                "_model_supports_vision": true
            }))
            .await;
        assert!(
            out.is_ok(),
            "vision flag must be stripped, not rejected: {out:?}"
        );
    }

    #[tokio::test]
    async fn a_bad_path_is_invalid_input_the_model_can_retry() {
        let (_d, t) = tool();
        let err = t
            .run(json!({ "command": "view", "path": "/etc/passwd" }))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn str_replace_ambiguity_is_reported_to_the_model() {
        let (_d, t) = tool();
        t.run(json!({ "command": "create", "path": "/memories/a.md", "file_text": "x x" }))
            .await
            .unwrap();
        let err = t
            .run(json!({ "command": "str_replace", "path": "/memories/a.md", "old_str": "x", "new_str": "y" }))
            .await
            .unwrap_err();
        match err {
            ToolError::InvalidInput(msg) => assert!(msg.contains("matched 2 times"), "got: {msg}"),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn write_target_groups_mutations_and_leaves_reads_concurrent() {
        let (_d, t) = tool();
        assert_eq!(
            t.write_target(&json!({ "command": "create", "path": "memories/a.md" })),
            Some("memory:/memories/a.md".to_string())
        );
        // A read is not a write target.
        assert_eq!(
            t.write_target(&json!({ "command": "view", "path": "/memories/a.md" })),
            None
        );
    }

    #[tokio::test]
    async fn dual_mounts_route_by_root_and_stay_isolated() {
        let (_d, t) = dual_tool();
        // Same rel path in each root writes to a different store.
        t.run(json!({ "command": "create", "path": "/memories/x.md", "file_text": "durable" }))
            .await
            .unwrap();
        t.run(json!({ "command": "create", "path": "/session/x.md", "file_text": "scratch" }))
            .await
            .unwrap();
        let d = t
            .run(json!({ "command": "view", "path": "/memories/x.md" }))
            .await
            .unwrap();
        let s = t
            .run(json!({ "command": "view", "path": "/session/x.md" }))
            .await
            .unwrap();
        assert_eq!(d.text, "durable");
        assert_eq!(s.text, "scratch");

        // write_target distinguishes the two roots so their mutations don't serialize together.
        assert_eq!(
            t.write_target(&json!({ "command": "create", "path": "session/x.md" })),
            Some("memory:/session/x.md".to_string())
        );

        // search fans across both mounts, each hit labeled with its own root.
        let hits = t
            .run(json!({ "command": "search", "query": "a" }))
            .await
            .unwrap();
        assert!(hits.text.contains("/memories/x.md"));
        assert!(hits.text.contains("/session/x.md"));
    }

    #[tokio::test]
    async fn cross_root_rename_is_rejected() {
        let (_d, t) = dual_tool();
        t.run(json!({ "command": "create", "path": "/memories/x.md", "file_text": "hi" }))
            .await
            .unwrap();
        let err = t
            .run(json!({ "command": "rename", "path": "/memories/x.md", "new_path": "/session/x.md" }))
            .await
            .unwrap_err();
        match err {
            ToolError::InvalidInput(msg) => assert!(msg.contains("across roots"), "got: {msg}"),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn an_unmounted_root_is_a_retryable_error() {
        // Only durable is mounted; a /session path is rejected with guidance the model can act on.
        let (_d, t) = tool();
        let err = t
            .run(json!({ "command": "view", "path": "/session/x.md" }))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn search_reports_no_matches_cleanly() {
        let (_d, t) = tool();
        t.run(json!({ "command": "create", "path": "/memories/a.md", "file_text": "hello" }))
            .await
            .unwrap();
        let out = t
            .run(json!({ "command": "search", "query": "zzz" }))
            .await
            .unwrap();
        assert!(out.text.contains("No memory matched"));
    }
}
