//! `write` — create or overwrite a file (creating parent directories).

use agent_core::tool::Tool;
use agent_core::{ToolError, ToolOutput};
use async_trait::async_trait;
use serde_json::{Value, json};

pub struct Write;

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
            .map(super::normalize_path)
            .map(|p| super::canonical_write_target(&p))
    }

    async fn run(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let path = input
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidInput("missing `path`".into()))?;
        let path = &super::normalize_path(path);
        let content = input
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidInput("missing `content`".into()))?;
        // pi-parity fix (pass 20): reject a FIFO/socket/device *before* `is_writable`'s own
        // `OpenOptions::write(true).open(path)` ever runs — opening an existing FIFO for write-only
        // blocks until a reader connects on the other end, the same unrecoverable hang class `read.rs`
        // was fixed for earlier (see `reject_non_regular_file`'s doc comment in `tools/mod.rs`). A bare
        // `mkfifo x` followed by `write path=x ...` reproduced this hanging indefinitely.
        super::reject_non_regular_file(path, "write")?;
        // `write_atomic` replaces the file via `rename(2)`, which only consults the *containing
        // directory's* write permission — the target file's own mode bits are never checked, so a
        // `chmod 444` file was silently overwritten. pi's `fs.writeFile` opens the existing path
        // directly, so the OS itself enforces the file's own write-permission bit (EACCES). Same
        // pre-check `edit.rs` already makes: `super::is_writable` performs a real access check
        // (open-for-write, then close without touching content) rather than inspecting mode bits,
        // which say nothing about *this process's* actual uid/gid (pi-parity task 50).
        if !super::is_writable(path) {
            return Err(ToolError::Execution(format!("{path} is not writable")));
        }
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    ToolError::Execution(format!("mkdir {}: {e}", parent.display()))
                })?;
            }
        }
        // Atomic temp-file + rename: an overwrite killed mid-write must not leave a half-written
        // file — the same guarantee `edit` makes (and which `serve` reattach depends on for the
        // session file). `create_dir_all` above ensures the sibling temp's directory exists.
        super::write_atomic(path, content.as_bytes())
            .map_err(|e| ToolError::Execution(format!("write {path}: {e}")))?;
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
        let out = Write
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
        Write
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
        let plain = Write
            .write_target(&json!({ "path": path.to_str().unwrap() }))
            .unwrap();
        let normalized = Write.write_target(&json!({ "path": at_prefixed })).unwrap();
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

        let err = Write
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
        let err = Write.run(json!({ "path": "/tmp/x" })).await.unwrap_err();
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
            Write.run(json!({ "path": fifo_path.to_str().unwrap(), "content": "x" })),
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
        Write
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
