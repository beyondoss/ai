//! `write` — create or overwrite a file (creating parent directories).

use std::path::PathBuf;
use std::sync::Arc;

use agent_core::tool::Tool;
use agent_core::{ToolError, ToolOutput};
use async_trait::async_trait;
use serde_json::{Value, json};

use super::fs::local::LocalFs;
use super::fs::{FileKind, FsBackend};

pub struct Write {
    /// Relative `path` arguments resolve against this. Empty = the process cwd. See
    /// [`super::resolve_against`].
    root: PathBuf,
    /// Where the write actually lands. Defaults to the host filesystem.
    backend: Arc<dyn FsBackend>,
}

impl Default for Write {
    fn default() -> Self {
        Self::new(PathBuf::new())
    }
}

impl Write {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            backend: Arc::new(LocalFs::new()),
        }
    }

    /// Write somewhere other than the host filesystem. See [`super::fs`].
    pub fn with_backend(mut self, backend: Arc<dyn FsBackend>) -> Self {
        self.backend = backend;
        self
    }

    fn resolve(&self, path: &str) -> String {
        super::resolve_against_in(&self.root, path, &self.backend.world())
    }
}

#[async_trait]
impl Tool for Write {
    fn name(&self) -> &str {
        "write"
    }
    fn description(&self) -> &str {
        "Create or overwrite a file with the given contents. Parent directories are created."
    }
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "File path to write." },
                "content": { "type": "string", "description": "Full file contents." }
            },
            "required": ["path", "content"]
        })
    }

    fn write_target(&self, input: &Value) -> Option<String> {
        input
            .get("path")
            .and_then(Value::as_str)
            .map(|p| super::write_key(&self.root, p, &self.backend.world()))
    }

    async fn run(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let path = input
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidInput("missing `path`".into()))?;
        let path = &self.resolve(path);
        let content = input
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidInput("missing `content`".into()))?;

        // One `stat` answers both pre-checks, in the order that matters.
        //
        // pi-parity fix (pass 20): a FIFO/socket/device must be rejected on its *kind* before anything
        // opens it for writing — opening an existing FIFO write-only blocks until a reader connects,
        // the same unrecoverable hang class `read.rs` was fixed for. The backend's `stat` never
        // performs the access check on a non-regular path for exactly this reason.
        //
        // Then writability: `write_atomic` replaces the file via `rename(2)`, which consults only the
        // *containing directory's* permission — the target's own mode bits are never checked, so a
        // `chmod 444` file was silently overwritten. pi's `fs.writeFile` opens the existing path
        // directly, letting the OS enforce EACCES. `Meta::writable` is a real access check rather than
        // an inspection of mode bits, which say nothing about this process's actual uid/gid.
        // A missing path reports no metadata at all and is simply created below, matching
        // `is_writable`'s own `NotFound => true`.
        let p = std::path::Path::new(path.as_str());
        if let Some(meta) = self.backend.stat(p).await? {
            if meta.kind == FileKind::Other {
                return Err(super::non_regular_file_error(path, "write"));
            }
            if !meta.writable {
                return Err(ToolError::Execution(format!("{path} is not writable")));
            }
        }
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() {
                self.backend.create_dir_all(parent).await?;
            }
        }
        // Atomic temp-file + rename: an overwrite killed mid-write must not leave a half-written
        // file — the same guarantee `edit` makes (and which `serve` reattach depends on for the
        // session file). `create_dir_all` above ensures the sibling temp's directory exists.
        self.backend.write_bytes(p, content.as_bytes()).await?;
        Ok(format!("wrote {} bytes to {path}", content.len()).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn writes_and_creates_parents() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested/dir/file.txt");
        let out = Write::default()
            .run(json!({ "path": path.to_str().unwrap(), "content": "hello" }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("5 bytes"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello");
    }

    #[tokio::test]
    async fn run_normalizes_the_path_argument() {
        // Proves `run` actually calls `super::normalize_path` (not just that the shared function
        // itself works — see `tools::tests` for that) via its `@`-prefix-strip behavior, which needs
        // no `$HOME` mutation to exercise safely in a parallel test run.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file.txt");
        let at_prefixed = format!("@{}", path.to_str().unwrap());
        Write::default()
            .run(json!({ "path": at_prefixed, "content": "hello" }))
            .await
            .unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello");
    }

    #[tokio::test]
    async fn write_target_normalizes_the_path_argument_too() {
        // `write_target` computes the same-turn concurrency-grouping key from a *different* code path
        // than `run` — both must normalize identically, or `write("~/f")` and `edit("~/f")` in the same
        // turn would get different canonical keys and lose the write-race protection between them.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file.txt");
        std::fs::write(&path, "x").unwrap();
        let at_prefixed = format!("@{}", path.to_str().unwrap());
        let plain = Write::default()
            .write_target(&json!({ "path": path.to_str().unwrap() }))
            .unwrap();
        let normalized = Write::default()
            .write_target(&json!({ "path": at_prefixed }))
            .unwrap();
        assert_eq!(plain, normalized);
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn rejects_a_read_only_file_leaving_its_content_unchanged() {
        // pi-parity gap (fixed): unlike `edit`, `write` had no writability pre-check at all —
        // `write_atomic`'s `rename(2)`-based swap only consults the containing directory's write
        // permission, never the target file's own mode bits, so a `chmod 444` file was silently
        // overwritten. pi's `fs.writeFile` opens the existing path directly, so the OS enforces the
        // file's own write bit; this proves the same refusal `edit.rs` already has.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("locked.txt");
        std::fs::write(&path, "original").unwrap();
        let mut perms = std::fs::metadata(&path).unwrap().permissions();
        perms.set_readonly(true);
        std::fs::set_permissions(&path, perms).unwrap();

        let err = Write::default()
            .run(json!({ "path": path.to_str().unwrap(), "content": "new" }))
            .await
            .unwrap_err();
        match err {
            ToolError::Execution(msg) => assert!(msg.contains("not writable"), "got: {msg}"),
            other => panic!("expected Execution(\"... not writable\"), got {other:?}"),
        }
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "original",
            "a refused write must leave the file's content unchanged"
        );

        // Restore write permission (owner-only, not world-writable) so the tempfile crate can clean
        // up the file on drop.
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    }

    #[tokio::test]
    async fn missing_content_is_invalid_input() {
        let err = Write::default()
            .run(json!({ "path": "/tmp/x" }))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn a_fifo_is_rejected_immediately_instead_of_hanging_the_tool_call_forever() {
        // Regression (pi-parity pass 20): `is_writable`'s own `OpenOptions::write(true).open(path)`
        // blocks forever on an existing FIFO with no reader on the other end — reproduced hanging 30+
        // seconds in the original audit, with no timeout able to preempt it (the blocking syscall runs
        // inline in the future's own `poll()`, not behind an `await` point). `mkfifo x` then `write
        // path=x ...` reproduced this with two ordinary tool calls. `tokio::time::timeout` here is a
        // *test* safety net, not the fix itself (see `reject_non_regular_file` in `tools/mod.rs`): if
        // the gate ever regresses, this test fails fast in seconds instead of hanging CI indefinitely.
        let dir = tempfile::tempdir().unwrap();
        let fifo_path = dir.path().join("a.fifo");
        let status = std::process::Command::new("mkfifo")
            .arg(&fifo_path)
            .status()
            .expect("mkfifo must be available to run this test");
        assert!(status.success(), "mkfifo failed to create the test fixture");

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            Write::default().run(json!({ "path": fifo_path.to_str().unwrap(), "content": "x" })),
        )
        .await
        .expect("write must reject a FIFO immediately, not hang");
        let err = result.expect_err("a FIFO must be rejected, not written to");
        assert!(matches!(err, ToolError::Execution(_)), "got: {err:?}");
    }

    #[tokio::test]
    async fn overwrites_existing_file_and_leaves_no_temp() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.txt");
        std::fs::write(&path, "old contents").unwrap();
        Write::default()
            .run(json!({ "path": path.to_str().unwrap(), "content": "new" }))
            .await
            .unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "new");
        // The atomic write must not leave its sibling temp behind.
        let temps: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(temps.is_empty(), "atomic write left a temp file behind");
    }
}
